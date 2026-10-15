"""Tests for the de-novo Add engine (Add-tab P1·1).

``add_segment_region`` segments a missed cell from an image channel inside a
box and emits new labels clipped to background. Pure NumPy/scipy/skimage — no
Qt, no napari.

Coverage:
  - Otsu + explicit-value threshold
  - background-mask exclusion (output never touches existing labels)
  - watershed separation of two touching blobs
  - min-size drop
  - relabel-from-next_label
  - clipping invariant (output ∩ existing == ∅)
  - degenerate inputs (constant image, empty foreground)
  - separate=False → connected components
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.add_segment import add_segment_region


def _bright_block(shape, zr, yr, xr, value=1.0, base=0.0):
    """An image with a bright cuboid against a dim background."""
    img = np.full(shape, base, dtype=np.float64)
    img[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]] = value
    return img


class TestThresholding:
    def test_otsu_segments_bright_region(self):
        img = _bright_block((6, 12, 12), (1, 5), (3, 9), (3, 9), value=5.0, base=0.0)
        existing = np.zeros((6, 12, 12), dtype=np.int32)

        out = add_segment_region(
            img, existing, threshold="otsu", separate=False, next_label=7
        )
        assert out.dtype == np.int32
        # The bright cuboid should be labeled; the dim background should not.
        assert out[2, 5, 5] == 7
        assert out[0, 0, 0] == 0
        assert set(np.unique(out)) == {0, 7}

    def test_explicit_value_threshold(self):
        img = _bright_block((4, 10, 10), (0, 4), (2, 8), (2, 8), value=10.0, base=2.0)
        existing = np.zeros((4, 10, 10), dtype=np.int32)

        # Cut at 5 → only the value-10 block survives.
        out = add_segment_region(
            img, existing, threshold=5.0, separate=False, next_label=1
        )
        assert np.any(out == 1)
        # Background (value 2 < 5) excluded.
        assert out[0, 0, 0] == 0

    def test_unknown_threshold_method_raises(self):
        img = np.ones((3, 4, 4))
        existing = np.zeros((3, 4, 4), dtype=np.int32)
        with pytest.raises(ValueError):
            add_segment_region(img, existing, threshold="triangle", next_label=1)


class TestBackgroundClipping:
    def test_existing_labels_never_overwritten(self):
        # Entire box is bright → threshold would select everything...
        img = np.full((4, 8, 8), 5.0, dtype=np.float64)
        existing = np.zeros((4, 8, 8), dtype=np.int32)
        existing[:, 0:4, :] = 99  # left half already labeled

        out = add_segment_region(
            img, existing, threshold=1.0, separate=False, next_label=100
        )
        # No output voxel may land on an existing label.
        assert not np.any(out[existing != 0])
        # Background half got segmented.
        assert np.any(out[existing == 0])

    def test_clipping_invariant_intersection_empty(self):
        rng = np.random.default_rng(0)
        img = rng.random((5, 10, 10)) + 1.0
        existing = np.zeros((5, 10, 10), dtype=np.int32)
        existing[2:4, 2:6, 2:6] = 3

        out = add_segment_region(
            img, existing, threshold=0.5, separate=True, next_label=10
        )
        # output ∩ existing == ∅
        assert np.all(out[existing != 0] == 0)
        overlap = (out != 0) & (existing != 0)
        assert not np.any(overlap)


class TestSeparation:
    def test_two_touching_blobs_split(self):
        # Dumbbell: two cubes joined by a thin neck → watershed should split.
        img = np.zeros((6, 10, 24), dtype=np.float64)
        img[1:5, 2:8, 2:9] = 5.0      # left blob
        img[1:5, 2:8, 15:22] = 5.0    # right blob
        img[2:4, 4:6, 9:15] = 5.0     # thin neck connecting them
        existing = np.zeros((6, 10, 24), dtype=np.int32)

        out = add_segment_region(
            img, existing, threshold=1.0, separate=True,
            min_distance=3, next_label=20,
        )
        ids = sorted(int(v) for v in np.unique(out) if v != 0)
        assert len(ids) == 2, f"expected 2 pieces, got {ids}"
        assert ids == [20, 21]

    def test_separate_false_one_label_per_blob(self):
        # Two clearly disjoint blobs, separate=False → connected components.
        img = np.zeros((4, 8, 16), dtype=np.float64)
        img[1:3, 2:6, 1:5] = 5.0
        img[1:3, 2:6, 11:15] = 5.0
        existing = np.zeros((4, 8, 16), dtype=np.int32)

        out = add_segment_region(
            img, existing, threshold=1.0, separate=False, next_label=1
        )
        ids = sorted(int(v) for v in np.unique(out) if v != 0)
        assert ids == [1, 2]


class TestMinSizeAndRelabel:
    def test_min_size_drops_specks(self):
        img = np.zeros((4, 10, 20), dtype=np.float64)
        img[1:3, 2:8, 1:9] = 5.0     # big blob (~96 vox)
        img[0, 0, 18] = 5.0          # 1-voxel speck
        existing = np.zeros((4, 10, 20), dtype=np.int32)

        out = add_segment_region(
            img, existing, threshold=1.0, separate=False,
            min_size=10, next_label=5,
        )
        ids = sorted(int(v) for v in np.unique(out) if v != 0)
        assert ids == [5], f"speck should be dropped, got {ids}"

    def test_relabel_starts_at_next_label(self):
        img = np.zeros((4, 8, 16), dtype=np.float64)
        img[1:3, 2:6, 1:5] = 5.0
        img[1:3, 2:6, 11:15] = 5.0
        existing = np.zeros((4, 8, 16), dtype=np.int32)

        out = add_segment_region(
            img, existing, threshold=1.0, separate=False, next_label=42
        )
        ids = sorted(int(v) for v in np.unique(out) if v != 0)
        assert ids == [42, 43]


class TestDegenerate:
    def test_constant_image_returns_empty(self):
        img = np.full((4, 6, 6), 3.0, dtype=np.float64)
        existing = np.zeros((4, 6, 6), dtype=np.int32)
        out = add_segment_region(img, existing, threshold="otsu", next_label=1)
        assert not np.any(out)

    def test_all_foreground_already_labeled_returns_empty(self):
        img = np.full((4, 6, 6), 5.0, dtype=np.float64)
        existing = np.full((4, 6, 6), 7, dtype=np.int32)  # everything taken
        out = add_segment_region(img, existing, threshold=1.0, next_label=8)
        assert not np.any(out)

    def test_shape_mismatch_raises(self):
        img = np.zeros((4, 6, 6))
        existing = np.zeros((4, 6, 7), dtype=np.int32)
        with pytest.raises(ValueError):
            add_segment_region(img, existing, next_label=1)

    def test_empty_array_returns_empty(self):
        img = np.zeros((0,))
        existing = np.zeros((0,), dtype=np.int32)
        out = add_segment_region(img, existing, next_label=1)
        assert out.size == 0


class TestClahe:
    def test_clahe_runs_and_clips_to_background(self):
        # Low-contrast bright-ish region; CLAHE should still segment + clip.
        rng = np.random.default_rng(1)
        img = rng.random((6, 16, 16)) * 0.1
        img[1:5, 4:12, 4:12] += 0.6  # a brighter cuboid
        existing = np.zeros((6, 16, 16), dtype=np.int32)
        existing[0, 0, 0] = 3

        out = add_segment_region(
            img, existing, clahe=True, threshold="otsu",
            separate=False, next_label=10,
        )
        assert out.dtype == np.int32
        # Clipping invariant holds even with CLAHE.
        assert np.all(out[existing != 0] == 0)

    def test_spacing_aware_edt_does_not_crash(self):
        img = np.zeros((4, 8, 16), dtype=np.float64)
        img[1:3, 2:6, 1:5] = 5.0
        img[1:3, 2:6, 11:15] = 5.0
        existing = np.zeros((4, 8, 16), dtype=np.int32)
        out = add_segment_region(
            img, existing, threshold=1.0, separate=True,
            spacing=(2.0, 1.0, 1.0), next_label=1,
        )
        assert np.any(out)


# --------------------------------------------------------------------------- #
# The two pure functions moved out of the annotator
# --------------------------------------------------------------------------- #
def test_edge_flag_ids_flags_the_edge_shell_not_a_lone_piece():
    from packages.epicycle_annotator.add_segment import edge_flag_ids

    lab = np.zeros((10, 10, 10), np.int32)
    lab[4:6, 4:6, 4:6] = 1           # interior
    lab[0:2, 4:6, 4:6] = 2           # touches z = 0
    assert edge_flag_ids(lab, 1) == {2}
    lone = np.zeros_like(lab)
    lone[0:2, 4:6, 4:6] = 3
    assert edge_flag_ids(lone, 1) == set()
    assert edge_flag_ids(np.zeros((4, 4), np.int32), 1) == set()


def test_the_annotator_methods_delegate_to_the_pure_functions():
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    from packages.epicycle_annotator.add_segment import (
        edge_flag_ids,
        smooth_pieces,
    )

    lab = np.zeros((12, 12, 12), np.int32)
    lab[3:9, 3:9, 3:9] = 5
    lab[0:3, 0:3, 0:3] = 6
    assert Curation3DAnnotator._add_edge_flags(None, lab, 2) == edge_flag_ids(lab, 2)
    a = Curation3DAnnotator._smooth_add_result(None, lab.copy(), 1.5)
    b = smooth_pieces(lab.copy(), 1.5)
    assert np.array_equal(a, b)

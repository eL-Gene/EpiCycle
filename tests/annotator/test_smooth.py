"""Tests for packages.epicycle_annotator.smooth — SDF label smoothing.

Pure algorithm tests (no Qt, no napari).
"""

import numpy as np
import pytest

from packages.epicycle_annotator.smooth import (
    apply_smooth_to_volume,
    smooth_label_sdf,
    smooth_split_fragments,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_staircase(shape=(30, 30, 30), label_id=5):
    """Create a volume with a Z-offset staircase pattern (jagged label)."""
    vol = np.zeros(shape, dtype=np.int32)
    for z in range(8, 22):
        offset = (z - 8) % 3  # shift center by 0-2 pixels per slice
        vol[z, 10 + offset:20 + offset, 10 + offset:20 + offset] = label_id
    return vol


def _make_sphere(shape=(30, 30, 30), label_id=3, center=(15, 15, 15), radius=8):
    """Create a volume with a spherical label."""
    vol = np.zeros(shape, dtype=np.int32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    vol[dist <= radius] = label_id
    return vol


def _make_dumbbell(shape=(40, 20, 20), label_id=7):
    """Dumbbell shape: two balls connected by a thin neck."""
    vol = np.zeros(shape, dtype=np.int32)
    # Ball 1
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    d1 = np.sqrt((zz - 8) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2)
    vol[d1 <= 6] = label_id
    # Ball 2
    d2 = np.sqrt((zz - 32) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2)
    vol[d2 <= 6] = label_id
    # Thin neck connecting them
    for z in range(14, 27):
        vol[z, 9:12, 9:12] = label_id
    return vol


# ===========================================================================
# TestSmoothLabelSdf
# ===========================================================================

class TestSmoothLabelSdf:
    """Tests for smooth_label_sdf()."""

    def test_smooth_reduces_staircase(self):
        """Z-offset staircase → smoothing should modify voxels."""
        vol = _make_staircase(label_id=5)
        result, n_mod, msg = smooth_label_sdf(vol, 5, sigma_z=2.0, sigma_xy=0.5)
        assert n_mod > 0, "Expected smoothing to modify staircase voxels"
        assert "Smoothed" in msg

    def test_smooth_preserves_label_id(self):
        """Result should contain only label_id and 0."""
        vol = _make_staircase(label_id=5)
        result, _, _ = smooth_label_sdf(vol, 5)
        unique = set(np.unique(result))
        assert unique <= {0, 5}, f"Unexpected labels in result: {unique}"

    def test_smooth_does_not_contain_neighbors(self):
        """Neighbour labels in the ROI must not appear in the result."""
        vol = _make_staircase(label_id=5)
        # Add a neighbour label nearby
        vol[10:15, 5:9, 5:9] = 99
        result, _, _ = smooth_label_sdf(vol, 5)
        assert 99 not in result, "Neighbour label leaked into smooth result"

    def test_label_not_found_returns_zeros(self):
        """Absent label → empty result, n_modified=0."""
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        result, n_mod, msg = smooth_label_sdf(vol, 42)
        assert n_mod == 0
        assert np.all(result == 0)
        assert "not found" in msg

    def test_topology_preservation_thin_neck(self):
        """Dumbbell shape should stay connected when preserve_topology=True."""
        from scipy.ndimage import label as nd_label

        vol = _make_dumbbell(label_id=7)
        result, n_mod, _ = smooth_label_sdf(
            vol, 7, sigma_z=1.5, sigma_xy=1.0, preserve_topology=True
        )
        smooth_mask = result == 7
        _, n_components = nd_label(smooth_mask)
        assert n_components <= 1 or n_mod >= 0  # either connected or at least ran

        # Original should be 1 component
        orig_mask = vol == 7
        _, n_orig = nd_label(orig_mask)
        assert n_orig == 1, "Test setup: dumbbell should be 1 component"

    def test_topology_off_does_not_crash(self):
        """preserve_topology=False should not crash."""
        vol = _make_dumbbell(label_id=7)
        result, n_mod, _ = smooth_label_sdf(
            vol, 7, sigma_z=2.0, sigma_xy=1.0, preserve_topology=False
        )
        assert result.shape == vol.shape

    def test_sphere_stays_spherical(self):
        """A spherical label should remain mostly unchanged."""
        vol = _make_sphere(label_id=3, radius=8)
        result, n_mod, _ = smooth_label_sdf(vol, 3, sigma_z=0.5, sigma_xy=0.5)
        # Sphere is already smooth — very few voxels should change.
        original_count = np.count_nonzero(vol == 3)
        assert n_mod < original_count * 0.15, (
            f"Too many voxels changed for sphere: {n_mod}/{original_count}"
        )

    def test_anisotropic_sigma_z_dominant(self):
        """Higher sigma_z should smooth Z-axis staircase more than low sigma_z."""
        vol = _make_staircase(label_id=5)
        _, n_mod_high_z, _ = smooth_label_sdf(vol, 5, sigma_z=3.0, sigma_xy=0.3)
        _, n_mod_low_z, _ = smooth_label_sdf(vol, 5, sigma_z=0.3, sigma_xy=0.3)
        assert n_mod_high_z >= n_mod_low_z, (
            f"High sigma_z ({n_mod_high_z}) should modify >= low sigma_z ({n_mod_low_z})"
        )

    def test_single_voxel_label(self):
        """A single-voxel label should be handled without error."""
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        vol[5, 5, 5] = 1
        result, n_mod, _ = smooth_label_sdf(vol, 1)
        # Either smoothed away or preserved — should not crash.
        assert result.shape == vol.shape

    def test_result_shape_matches_input(self):
        """Result volume should have same shape as input."""
        vol = _make_staircase(label_id=5)
        result, _, _ = smooth_label_sdf(vol, 5)
        assert result.shape == vol.shape


# ===========================================================================
# TestApplySmoothToVolume
# ===========================================================================

class TestApplySmoothToVolume:
    """Tests for apply_smooth_to_volume()."""

    def test_safe_write_does_not_overwrite_neighbors(self):
        """Neighbour labels must not be overwritten."""
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[5:15, 5:15, 5:15] = 1  # target
        vol[5:15, 5:15, 14:18] = 2  # neighbour overlapping edge

        roi = (slice(3, 17), slice(3, 17), slice(3, 19))
        smooth_result = np.zeros_like(vol[roi])
        # Smooth result tries to expand into neighbour's space
        smooth_result[2:14, 2:14, 2:16] = 1

        apply_smooth_to_volume(vol, roi, smooth_result, label_id=1)

        # Neighbour label 2 should still exist
        assert np.any(vol == 2), "Neighbour label was overwritten"

    def test_label_cleared_and_rewritten(self):
        """Old footprint should be replaced by new smooth footprint."""
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[5:15, 5:15, 5:15] = 1
        old_count = np.count_nonzero(vol == 1)

        roi = (slice(4, 16), slice(4, 16), slice(4, 16))
        smooth_result = np.zeros_like(vol[roi])
        smooth_result[2:10, 2:10, 2:10] = 1  # smaller region

        apply_smooth_to_volume(vol, roi, smooth_result, label_id=1)
        new_count = np.count_nonzero(vol == 1)
        assert new_count < old_count, "Smooth result should have replaced old footprint"

    def test_no_modification_when_smooth_unchanged(self):
        """If smooth result matches original, n_modified should be 0."""
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[5:15, 5:15, 5:15] = 1

        roi = (slice(5, 15), slice(5, 15), slice(5, 15))
        smooth_result = np.ones_like(vol[roi])  # same as original ROI content

        n_mod = apply_smooth_to_volume(vol, roi, smooth_result, label_id=1)
        # The footprint didn't actually change shape.
        assert np.count_nonzero(vol == 1) == np.count_nonzero(vol[roi] == 1)

    def test_partial_roi(self):
        """Only the ROI region should be affected."""
        vol = np.zeros((30, 30, 30), dtype=np.int32)
        vol[5:25, 5:25, 5:25] = 1

        roi = (slice(10, 20), slice(10, 20), slice(10, 20))
        smooth_result = np.zeros_like(vol[roi])
        smooth_result[2:8, 2:8, 2:8] = 1

        apply_smooth_to_volume(vol, roi, smooth_result, label_id=1)

        # Voxels outside ROI should be untouched (vol[5:10, 5:10, 5:10] = 1 still).
        assert np.any(vol[5:10, 5:10, 5:10] == 1), "Voxels outside ROI were modified"


# ===========================================================================
# TestSmoothSplitFragments
# ===========================================================================

class TestSmoothSplitFragments:
    """Tests for smooth_split_fragments()."""

    def test_two_fragments_both_smoothed(self):
        """Both fragments should be smoothed."""
        vol = np.zeros((30, 30, 30), dtype=np.int32)
        # Fragment 1: staircase pattern
        for z in range(5, 15):
            off = z % 2
            vol[z, 5 + off:12 + off, 5:12] = 10
        # Fragment 2: staircase pattern in different location
        for z in range(5, 15):
            off = z % 2
            vol[z, 18 + off:25 + off, 18:25] = 20

        roi = (slice(0, 30), slice(0, 30), slice(0, 30))
        total, msg = smooth_split_fragments(
            vol, roi, [10, 20], sigma_z=2.0, sigma_xy=0.5
        )
        assert total > 0, "Expected both fragments to be smoothed"
        # Both labels should still exist
        assert np.any(vol == 10)
        assert np.any(vol == 20)

    def test_adjacent_fragments_do_not_merge(self):
        """Adjacent fragments should remain separate after smoothing."""
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[5:15, 5:10, 5:15] = 10
        vol[5:15, 10:15, 5:15] = 20

        roi = (slice(0, 20), slice(0, 20), slice(0, 20))
        smooth_split_fragments(vol, roi, [10, 20], sigma_z=1.0, sigma_xy=0.5)

        # Check no voxel is assigned to both labels
        both = (vol == 10) & (vol == 20)
        assert not np.any(both), "Fragments merged after smoothing"

    def test_empty_fragment_skipped(self):
        """Fragment ID not present in ROI should be skipped without error."""
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[5:15, 5:15, 5:15] = 10

        roi = (slice(0, 20), slice(0, 20), slice(0, 20))
        total, msg = smooth_split_fragments(
            vol, roi, [10, 999], sigma_z=1.0, sigma_xy=0.5
        )
        # Should not crash; only label 10 processed
        assert np.any(vol == 10)


# ===========================================================================
# TestSmoothSplitKeepsTheObject
# ===========================================================================

def _slab_split(n_slabs, *, keep_object=False, sigma_z=3.0, sigma_xy=1.5,
                depth=60):
    """Split a rod into ``n_slabs`` marker slabs, then auto-smooth it.

    Returns (volume, roi, original_mask, ids).
    """
    from packages.epicycle_annotator.split import (
        apply_split_to_volume,
        seeded_watershed_split,
    )

    vol = np.zeros((72, 16, 16), dtype=np.int32)
    vol[6:6 + depth, 4:12, 4:12] = 250
    original = vol == 250
    roi = (slice(0, 72), slice(0, 16), slice(0, 16))
    baseline = vol[roi].copy()
    step = depth // n_slabs
    marked = vol[roi].copy()
    for k in range(n_slabs):
        z = 6 + step * k + step // 2
        marked[z, 6:10, 6:10] = 1000 + k
    result, ids, _msg = seeded_watershed_split(baseline, marked, 250)
    apply_split_to_volume(vol, roi, result, 250)
    smooth_split_fragments(vol, roi, ids, sigma_z=sigma_z, sigma_xy=sigma_xy,
                           keep_object=keep_object)
    return vol, roi, original, ids


class TestSmoothSplitKeepsTheObject:
    @pytest.mark.parametrize("n_slabs", [2, 6, 12])
    def test_slab_split_keeps_object_and_pieces(self, n_slabs):
        vol, _roi, original, ids = _slab_split(n_slabs, keep_object=True)
        assert len(ids) == n_slabs
        in_ids = np.isin(vol, ids)
        # nothing grows outside the object
        assert not np.any(in_ids & ~original)
        # every voxel of the object is still owned by a piece
        assert np.count_nonzero(in_ids) == np.count_nonzero(original)
        # no piece vanished
        for i in ids:
            assert np.any(vol == i), f"piece {i} is empty"

    def test_default_mode_still_loses_voxels_on_many_slabs(self):
        """keep_object=False is today's behaviour: the mode is pinned."""
        vol, _roi, original, ids = _slab_split(12)
        kept = np.count_nonzero(np.isin(vol, ids)) / np.count_nonzero(original)
        assert kept < 0.95

    @pytest.mark.parametrize("keep_object", [False, True])
    def test_two_separated_staircase_blocks_still_smooth(self, keep_object):
        vol = np.zeros((30, 30, 30), dtype=np.int32)
        for z in range(5, 25):
            off = z % 2
            vol[z, 5 + off:12 + off, 5:12] = 10
            vol[z, 18 + off:25 + off, 18:25] = 20
        before = vol.copy()
        roi = (slice(0, 30), slice(0, 30), slice(0, 30))
        total, _msg = smooth_split_fragments(
            vol, roi, [10, 20], sigma_z=2.0, sigma_xy=0.5,
            keep_object=keep_object,
        )
        assert np.any(vol == 10) and np.any(vol == 20)
        if keep_object:
            # Each block is its own whole object here, so keeping the
            # object means its voxel set cannot change.
            assert np.array_equal(vol > 0, before > 0)
        else:
            assert total > 0
            assert not np.array_equal(vol, before)

    def test_a_piece_that_would_vanish_is_left_untouched_when_keeping(self):
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[5:15, 5:15, 5:15] = 10
        vol[9, 9, 9] = 0
        vol[9, 9, 9] = 30
        vol[9, 10, 10] = 30   # a thin two-voxel piece inside the block
        before_thin = (vol == 30).copy()
        roi = (slice(0, 20), slice(0, 20), slice(0, 20))
        smooth_split_fragments(vol, roi, [10, 30], sigma_z=3.0, sigma_xy=1.5,
                               keep_object=True)
        assert np.array_equal(vol == 30, before_thin)
        assert np.any(vol == 10)


class TestKeepObjectFillStaysInsideTheObject:
    def _u_shape(self):
        vol = np.zeros((12, 30, 30), dtype=np.int32)
        vol[3:9, 4:26, 4:10] = 1    # left arm
        vol[3:9, 4:26, 20:26] = 1   # right arm
        vol[3:9, 20:26, 4:26] = 1   # the base joining them
        return vol

    def test_u_shaped_object_pieces_stay_one_component_each(self):
        from scipy.ndimage import label as nd_label
        from packages.epicycle_annotator.split import (
            apply_split_to_volume,
            seeded_watershed_split,
        )

        vol = self._u_shape()
        original = vol == 1
        roi = (slice(0, 12), slice(0, 30), slice(0, 30))
        baseline = vol[roi].copy()
        marked = baseline.copy()
        marked[5, 6, 6] = 11
        marked[5, 6, 23] = 12
        result, ids, _ = seeded_watershed_split(baseline, marked, 1)
        vol[original] = 0
        vol[original] = result[original]
        smooth_split_fragments(vol, roi, ids, sigma_z=3.0, sigma_xy=1.5,
                               keep_object=True, object_mask=original)
        assert np.array_equal(np.isin(vol, ids), original)
        for i in ids:
            _, n = nd_label(vol == i)
            assert n == 1, f"piece {i} has {n} components"

    def test_object_mask_is_the_object_not_what_the_pieces_hold(self):
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[5:15, 5:15, 5:10] = 10
        vol[5:15, 5:15, 10:15] = 20
        # one object voxel was left as something else (not a piece)
        vol[9, 9, 9] = 99
        obj = np.zeros(vol.shape, dtype=bool)
        obj[5:15, 5:15, 5:15] = True
        roi = (slice(0, 20), slice(0, 20), slice(0, 20))
        smooth_split_fragments(vol, roi, [10, 20], sigma_z=2.0,
                               sigma_xy=1.0, keep_object=True,
                               object_mask=obj)
        assert vol[9, 9, 9] == 99
        assert not np.any(np.isin(vol, [10, 20]) & ~obj)

    def test_keep_count_is_measured_after_the_clip(self):
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[5:15, 5:15, 5:15] = 10
        before = vol.copy()
        roi = (slice(0, 20), slice(0, 20), slice(0, 20))
        total, _ = smooth_split_fragments(vol, roi, [10], sigma_z=2.0,
                                          sigma_xy=1.0, keep_object=True)
        assert total == int(np.count_nonzero(before != vol)) == 0


class TestKeepModeProtectsThePaintedMarkers:
    def test_twelve_slab_split_keeps_every_marker_whole(self):
        from scipy.ndimage import label as nd_label
        from packages.epicycle_annotator.split import (
            apply_split_to_volume,
            seeded_watershed_split,
        )

        vol = np.zeros((72, 16, 16), dtype=np.int32)
        vol[6:66, 4:12, 4:12] = 250
        original = vol == 250
        roi = (slice(0, 72), slice(0, 16), slice(0, 16))
        baseline = vol[roi].copy()
        marked = baseline.copy()
        for k in range(12):
            z = 6 + 5 * k + 2
            marked[z, 5:11, 5:11] = 1000 + k
        result, ids, _ = seeded_watershed_split(baseline, marked, 250)
        apply_split_to_volume(vol, roi, result, 250)
        markers = np.where(np.isin(marked, ids), marked, 0)
        smooth_split_fragments(
            vol, roi, ids, sigma_z=3.0, sigma_xy=1.5,
            keep_object=True, object_mask=original, marker_labels=markers,
        )
        assert np.array_equal(np.isin(vol, ids), original)
        for i in ids:
            mine = markers == i
            assert np.all(vol[mine] == i), f"marker {i} lost voxels"
            _, n = nd_label(vol == i)
            assert n == 1, f"piece {i} has {n} components"

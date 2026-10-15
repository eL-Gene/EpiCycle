"""Tests for SDF contour interpolation engine.

Phase K5.7.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.interpolation import (
    compute_sdf,
    interpolate_label_all_orientations,
    interpolate_label_in_volume,
    interpolate_slices_sdf,
    postprocess_label,
)


# ---------------------------------------------------------------------------
# compute_sdf
# ---------------------------------------------------------------------------

class TestComputeSdf:
    def test_empty_mask_all_positive(self):
        mask = np.zeros((10, 10), dtype=bool)
        sdf = compute_sdf(mask)
        assert (sdf >= 0).all()

    def test_full_mask_all_negative(self):
        mask = np.ones((10, 10), dtype=bool)
        sdf = compute_sdf(mask)
        assert (sdf <= 0).all()

    def test_boundary_near_zero(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        sdf = compute_sdf(mask)
        assert sdf[10, 10] < 0, "Interior should be negative"
        assert sdf[0, 0] > 0, "Exterior should be positive"


# ---------------------------------------------------------------------------
# interpolate_slices_sdf
# ---------------------------------------------------------------------------

class TestInterpolateSlicesSdf:
    def test_same_shape_identity(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        results = interpolate_slices_sdf(mask, mask, 3)
        assert len(results) == 3
        for r in results:
            # All intermediates should be very similar to start
            overlap = (r & mask).sum()
            assert overlap > mask.sum() * 0.9

    def test_circle_interpolation_sizes_monotonic(self):
        y, x = np.ogrid[-15:16, -15:16]
        mask_small = (x ** 2 + y ** 2) < 5 ** 2
        mask_large = (x ** 2 + y ** 2) < 12 ** 2
        results = interpolate_slices_sdf(mask_small, mask_large, 4)
        assert len(results) == 4
        areas = [r.sum() for r in results]
        # Areas should be monotonically increasing
        for i in range(len(areas) - 1):
            assert areas[i + 1] >= areas[i]

    def test_zero_intermediate_returns_empty(self):
        mask = np.ones((5, 5), dtype=bool)
        assert interpolate_slices_sdf(mask, mask, 0) == []


# ---------------------------------------------------------------------------
# interpolate_label_in_volume
# ---------------------------------------------------------------------------

class TestInterpolateLabelInVolume:
    def test_xy_interpolation(self):
        vol = np.zeros((10, 20, 20), dtype=np.int32)
        vol[2, 5:15, 5:15] = 1
        vol[7, 5:15, 5:15] = 1
        count = interpolate_label_in_volume(vol, 1, [2, 7], "XY")
        assert count == 4  # slices 3, 4, 5, 6
        for z in range(3, 7):
            assert (vol[z] == 1).any(), f"Slice Z={z} should have interpolated content"

    def test_xz_interpolation(self):
        vol = np.zeros((20, 10, 20), dtype=np.int32)
        vol[:, 2, 5:15] = 1
        vol[:, 7, 5:15] = 1
        count = interpolate_label_in_volume(vol, 1, [2, 7], "XZ")
        assert count == 4
        for y in range(3, 7):
            assert (vol[:, y, :] == 1).any(), f"Slice Y={y} should have content"

    def test_yz_interpolation(self):
        vol = np.zeros((20, 20, 10), dtype=np.int32)
        vol[:, 5:15, 2] = 1
        vol[:, 5:15, 7] = 1
        count = interpolate_label_in_volume(vol, 1, [2, 7], "YZ")
        assert count == 4
        for x in range(3, 7):
            assert (vol[:, :, x] == 1).any(), f"Slice X={x} should have content"

    def test_insufficient_slices_returns_zero(self):
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        vol[5, :, :] = 1
        count = interpolate_label_in_volume(vol, 1, [5], "XY")
        assert count == 0

    def test_adjacent_slices_no_gap(self):
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        vol[3, 2:8, 2:8] = 1
        vol[4, 2:8, 2:8] = 1
        count = interpolate_label_in_volume(vol, 1, [3, 4], "XY")
        assert count == 0  # no gap between adjacent slices

    def test_empty_mask_at_one_end(self):
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        vol[2, 3:7, 3:7] = 1
        # slice 7 has no label 1 data
        count = interpolate_label_in_volume(vol, 1, [2, 7], "XY")
        assert count == 0  # can't interpolate when one end is empty

    def test_multiple_label_ids_independent(self):
        vol = np.zeros((10, 20, 20), dtype=np.int32)
        vol[2, 2:8, 2:8] = 1
        vol[7, 2:8, 2:8] = 1
        vol[2, 12:18, 12:18] = 2
        vol[7, 12:18, 12:18] = 2
        # Interpolate only label 1
        count = interpolate_label_in_volume(vol, 1, [2, 7], "XY")
        assert count == 4
        # Label 2 should be unaffected in intermediate slices
        for z in range(3, 7):
            assert not (vol[z, 12:18, 12:18] == 2).any()


# ---------------------------------------------------------------------------
# interpolate_label_all_orientations
# ---------------------------------------------------------------------------

class TestInterpolateLabelAllOrientations:
    def test_xy_full_pipeline(self):
        vol = np.zeros((10, 20, 20), dtype=np.int32)
        vol[2, 5:15, 5:15] = 1
        vol[7, 5:15, 5:15] = 1
        edited = {"XY": {1: [2, 7]}, "XZ": {}, "YZ": {}}
        total = interpolate_label_all_orientations(vol, 1, edited)
        assert total > 0
        for z in range(3, 7):
            assert (vol[z] == 1).any()

    def test_no_edits_returns_zero(self):
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        edited = {"XY": {}, "XZ": {}, "YZ": {}}
        total = interpolate_label_all_orientations(vol, 1, edited)
        assert total == 0

    def test_numpy_int_keys(self):
        """Label IDs stored by LabelSyncManager may be numpy integers."""
        vol = np.zeros((10, 20, 20), dtype=np.int32)
        vol[2, 5:15, 5:15] = 1
        vol[7, 5:15, 5:15] = 1
        edited = {"XY": {np.int64(1): [2, 7]}, "XZ": {}, "YZ": {}}
        total = interpolate_label_all_orientations(vol, 1, edited)
        assert total == 4

    def test_post_process_smooths_result(self):
        vol = np.zeros((10, 20, 20), dtype=np.int32)
        vol[2, 5:15, 5:15] = 1
        vol[7, 5:15, 5:15] = 1
        edited = {"XY": {1: [2, 7]}, "XZ": {}, "YZ": {}}
        total = interpolate_label_all_orientations(vol, 1, edited, post_process=True)
        assert total > 0
        # Intermediate slices should still have content after post-processing
        for z in range(3, 7):
            assert (vol[z] == 1).any()


# ---------------------------------------------------------------------------
# postprocess_label
# ---------------------------------------------------------------------------

class TestPostprocessLabel:
    def test_fills_internal_hole(self):
        vol = np.zeros((5, 10, 10), dtype=np.int32)
        vol[:, 2:8, 2:8] = 1
        vol[:, 4:6, 4:6] = 0  # create hole
        postprocess_label(vol, 1)
        assert vol[2, 5, 5] == 1, "Hole should be filled"

    def test_no_label_is_noop(self):
        vol = np.zeros((5, 5, 5), dtype=np.int32)
        postprocess_label(vol, 1)  # should not error
        assert (vol == 0).all()

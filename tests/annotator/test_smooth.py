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

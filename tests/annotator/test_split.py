"""Tests for seeded watershed split algorithm.

Pure algorithm tests — no Qt, no napari required.
Tests cover validation, watershed, safe-write, and EDT overlay.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.split import (
    apply_split_to_volume,
    compute_edt_overlay,
    seeded_watershed_split,
    validate_split_seeds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cube_label(shape, label_id, z_range, y_range, x_range):
    """Create a volume with a cubic label region."""
    vol = np.zeros(shape, dtype=np.int32)
    vol[z_range[0]:z_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]] = label_id
    return vol


# ===========================================================================
# TestValidateSplitSeeds
# ===========================================================================

class TestValidateSplitSeeds:
    """Tests for validate_split_seeds()."""

    def test_two_seeds_valid(self):
        """Two seed IDs within the mask → valid."""
        snapshot = _make_cube_label((10, 20, 20), 5, (2, 8), (5, 15), (5, 15))
        current = snapshot.copy()
        current[3:5, 7:9, 7:9] = 10   # seed 1
        current[5:7, 11:13, 11:13] = 11  # seed 2

        valid, seed_ids, msg = validate_split_seeds(snapshot, current, 5)

        assert valid
        assert sorted(seed_ids) == [10, 11]
        assert "2" in msg  # mentions 2 seeds

    def test_one_seed_invalid(self):
        """Only 1 seed → invalid."""
        snapshot = _make_cube_label((10, 20, 20), 5, (2, 8), (5, 15), (5, 15))
        current = snapshot.copy()
        current[3:5, 7:9, 7:9] = 10  # only one seed

        valid, seed_ids, msg = validate_split_seeds(snapshot, current, 5)

        assert not valid
        assert len(seed_ids) == 1
        assert "Need" in msg

    def test_seeds_outside_mask_invalid(self):
        """Seeds painted outside the label mask → invalid."""
        snapshot = _make_cube_label((10, 20, 20), 5, (2, 8), (5, 15), (5, 15))
        current = snapshot.copy()
        # Paint seeds in background area (not within label 5).
        current[0, 0, 0] = 10
        current[0, 0, 1] = 11

        valid, seed_ids, msg = validate_split_seeds(snapshot, current, 5)

        assert not valid

    def test_no_seeds_invalid(self):
        """No painting at all → invalid."""
        snapshot = _make_cube_label((10, 10, 10), 3, (2, 8), (2, 8), (2, 8))
        current = snapshot.copy()

        valid, seed_ids, msg = validate_split_seeds(snapshot, current, 3)

        assert not valid
        assert len(seed_ids) == 0

    def test_target_label_not_counted_as_seed(self):
        """Painting with the target label ID itself should not count as seed."""
        snapshot = _make_cube_label((10, 10, 10), 5, (2, 8), (2, 8), (2, 8))
        current = snapshot.copy()
        # Re-paint part with label 5 — same as target.
        current[3:5, 3:5, 3:5] = 5
        # Only one actual seed.
        current[5:7, 5:7, 5:7] = 10

        valid, _, _ = validate_split_seeds(snapshot, current, 5)

        assert not valid

    def test_label_not_in_snapshot(self):
        """Target label absent from snapshot → invalid."""
        snapshot = np.zeros((5, 5, 5), dtype=np.int32)
        current = snapshot.copy()

        valid, seed_ids, msg = validate_split_seeds(snapshot, current, 99)

        assert not valid
        assert "not found" in msg.lower()

    def test_three_seeds_valid(self):
        """Three seeds → valid."""
        snapshot = _make_cube_label((10, 30, 30), 7, (2, 8), (2, 28), (2, 28))
        current = snapshot.copy()
        current[4, 5, 5] = 20
        current[4, 15, 15] = 21
        current[4, 25, 25] = 22

        valid, seed_ids, msg = validate_split_seeds(snapshot, current, 7)

        assert valid
        assert sorted(seed_ids) == [20, 21, 22]


# ===========================================================================
# TestSeededWatershedSplit
# ===========================================================================

class TestSeededWatershedSplit:
    """Tests for seeded_watershed_split()."""

    def test_two_seeds_splits_label(self):
        """Mask mode: 2 seeds produce 2 non-overlapping regions."""
        snapshot = _make_cube_label((10, 20, 20), 5, (2, 8), (5, 15), (5, 15))
        current = snapshot.copy()
        # Place seeds at opposite corners of the label.
        current[4, 6, 6] = 10     # seed 1: top-left-ish
        current[4, 14, 14] = 11   # seed 2: bottom-right-ish

        result, new_ids, msg = seeded_watershed_split(
            snapshot, current, 5, method="mask"
        )

        assert result.shape == snapshot.shape
        assert set(new_ids) == {10, 11}
        # Within the original mask, all voxels should be assigned.
        mask = snapshot == 5
        assert np.all(result[mask] > 0)
        # Outside the mask, everything is 0.
        assert np.all(result[~mask] == 0)

    def test_three_seeds_three_regions(self):
        """Mask mode: 3 seeds → 3 distinct regions."""
        snapshot = _make_cube_label((10, 30, 30), 7, (2, 8), (2, 28), (2, 28))
        current = snapshot.copy()
        current[4, 5, 5] = 20
        current[4, 15, 15] = 21
        current[4, 25, 25] = 22

        result, new_ids, _ = seeded_watershed_split(snapshot, current, 7)

        assert set(new_ids) == {20, 21, 22}
        # Each seed ID should appear in the result.
        for sid in [20, 21, 22]:
            assert np.any(result == sid)

    def test_intensity_mode_without_image_falls_back(self):
        """Intensity mode with no image → falls back to mask mode."""
        snapshot = _make_cube_label((6, 12, 12), 3, (1, 5), (2, 10), (2, 10))
        current = snapshot.copy()
        current[3, 4, 4] = 10
        current[3, 8, 8] = 11

        result, new_ids, msg = seeded_watershed_split(
            snapshot, current, 3, method="intensity", image_roi=None
        )

        # Should still produce a valid split (falls back to mask mode).
        assert set(new_ids) == {10, 11}
        mask = snapshot == 3
        assert np.all(result[mask] > 0)

    def test_intensity_mode_with_image(self):
        """Intensity mode with image data → valid split."""
        snapshot = _make_cube_label((6, 12, 12), 3, (1, 5), (2, 10), (2, 10))
        current = snapshot.copy()
        current[3, 4, 4] = 10
        current[3, 8, 8] = 11

        # Create fake image with gradient.
        image = np.random.rand(6, 12, 12).astype(np.float32)

        result, new_ids, msg = seeded_watershed_split(
            snapshot, current, 3, method="intensity", image_roi=image
        )

        assert set(new_ids) == {10, 11}
        assert "Split into 2" in msg

    def test_label_not_in_snapshot_returns_empty(self):
        """Target label absent → empty result."""
        snapshot = np.zeros((5, 5, 5), dtype=np.int32)
        current = snapshot.copy()

        result, new_ids, msg = seeded_watershed_split(
            snapshot, current, 99
        )

        assert len(new_ids) == 0
        assert np.all(result == 0)

    def test_insufficient_seeds_returns_empty(self):
        """Only 1 seed → validation fails, empty result."""
        snapshot = _make_cube_label((6, 10, 10), 5, (1, 5), (2, 8), (2, 8))
        current = snapshot.copy()
        current[3, 5, 5] = 10  # only one seed

        result, new_ids, msg = seeded_watershed_split(
            snapshot, current, 5
        )

        assert len(new_ids) < 2
        assert "Need" in msg


# ===========================================================================
# TestApplySplitToVolume
# ===========================================================================

class TestApplySplitToVolume:
    """Tests for apply_split_to_volume()."""

    def test_safe_write_does_not_overwrite_neighbors(self):
        """Neighboring labels must remain untouched."""
        volume = np.zeros((10, 20, 20), dtype=np.int32)
        volume[2:8, 5:15, 5:15] = 5   # target
        volume[2:8, 5:15, 0:5] = 99   # neighbor

        roi = (slice(0, 10), slice(0, 20), slice(0, 20))
        split_result = np.zeros((10, 20, 20), dtype=np.int32)
        # Fill entire region (intentionally overlaps neighbor area).
        split_result[2:8, 5:15, 0:15] = 10

        n = apply_split_to_volume(volume, roi, split_result, 5)

        assert n > 0
        # Neighbor must be untouched.
        assert np.all(volume[2:8, 5:15, 0:5] == 99)
        # Target region should have the split result.
        assert np.all(volume[2:8, 5:15, 5:15] == 10)

    def test_zero_voxels_when_label_absent(self):
        """If target label doesn't exist in volume → 0 modifications."""
        volume = np.zeros((5, 5, 5), dtype=np.int32)
        roi = (slice(0, 5), slice(0, 5), slice(0, 5))
        split_result = np.ones((5, 5, 5), dtype=np.int32) * 10

        n = apply_split_to_volume(volume, roi, split_result, 99)

        assert n == 0

    def test_partial_roi(self):
        """Apply split to a sub-ROI of the full volume."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        volume[5:10, 5:10, 5:10] = 3  # target label

        roi = (slice(3, 12), slice(3, 12), slice(3, 12))
        split_result = np.zeros((9, 9, 9), dtype=np.int32)
        # Mark the label region within the ROI.
        split_result[2:7, 2:7, 2:7] = 20

        n = apply_split_to_volume(volume, roi, split_result, 3)

        assert n == 5 * 5 * 5  # 125 voxels
        # Verify the write happened.
        assert np.all(volume[5:10, 5:10, 5:10] == 20)

    def test_does_not_write_zero_result(self):
        """Zero values in split_result should not overwrite anything."""
        volume = np.zeros((5, 5, 5), dtype=np.int32)
        volume[1:4, 1:4, 1:4] = 7

        roi = (slice(0, 5), slice(0, 5), slice(0, 5))
        # split_result is all zeros — nothing should be written.
        split_result = np.zeros((5, 5, 5), dtype=np.int32)

        n = apply_split_to_volume(volume, roi, split_result, 7)

        assert n == 0
        # Original label should be unchanged.
        assert np.all(volume[1:4, 1:4, 1:4] == 7)


# ===========================================================================
# TestComputeEdtOverlay
# ===========================================================================

class TestComputeEdtOverlay:
    """Tests for compute_edt_overlay()."""

    def test_edt_values_positive_inside_mask(self):
        """EDT should be > 0 inside the label mask, 0 outside."""
        volume = _make_cube_label((10, 20, 20), 5, (2, 8), (5, 15), (5, 15))

        edt_crop, roi, edt_max = compute_edt_overlay(volume, 5)

        assert edt_crop is not None
        assert edt_crop.dtype == np.float32
        # All values in crop should be >= 0 and nonzero inside the label.
        assert np.any(edt_crop > 0)
        # edt_max should match the actual max of the crop.
        assert edt_max == float(edt_crop.max())
        assert edt_max > 0
        # Verify the crop is positioned correctly via the roi slices.
        assert all(isinstance(s, slice) for s in roi)

    def test_edt_peak_at_center(self):
        """Max EDT value should be near the center of a spherical-ish label."""
        volume = np.zeros((21, 21, 21), dtype=np.int32)
        # Create a sphere-ish label centered at (10, 10, 10) with radius 8.
        for z in range(21):
            for y in range(21):
                for x in range(21):
                    if (z - 10) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 64:
                        volume[z, y, x] = 3

        edt_crop, roi, edt_max = compute_edt_overlay(volume, 3)

        assert edt_crop is not None
        assert edt_max > 0
        # The maximum should be at or very near the center of the label.
        # Convert crop-local max position to full-volume coordinates.
        max_local = np.unravel_index(np.argmax(edt_crop), edt_crop.shape)
        max_full = tuple(s.start + p for s, p in zip(roi, max_local))
        for dim in range(3):
            assert abs(max_full[dim] - 10) <= 1, (
                f"Max EDT at {max_full}, expected near (10, 10, 10)"
            )

    def test_label_not_found_returns_none(self):
        """If target label is absent → (None, None, None)."""
        volume = np.zeros((5, 5, 5), dtype=np.int32)

        edt_crop, roi, edt_max = compute_edt_overlay(volume, 99)

        assert edt_crop is None
        assert roi is None
        assert edt_max is None

    def test_thin_neck_has_low_edt(self):
        """A label with a thin neck should have low EDT values at the neck."""
        volume = np.zeros((5, 20, 5), dtype=np.int32)
        # Two blobs connected by a thin 1-pixel neck.
        volume[0:5, 0:8, 0:5] = 1   # blob 1
        volume[0:5, 12:20, 0:5] = 1  # blob 2
        volume[2, 8:12, 2] = 1       # thin neck (1 pixel wide)

        edt_crop, roi, edt_max = compute_edt_overlay(volume, 1)

        assert edt_crop is not None
        assert edt_max > 0
        # Convert full-volume neck/center coords to crop-local coords.
        offsets = [s.start for s in roi]
        neck_local = (2 - offsets[0], 9 - offsets[1], 2 - offsets[2])
        center_local = (2 - offsets[0], 4 - offsets[1], 2 - offsets[2])
        neck_edt = edt_crop[neck_local]
        blob_center_edt = edt_crop[center_local]
        assert neck_edt < blob_center_edt, (
            f"Neck EDT {neck_edt} should be less than blob center {blob_center_edt}"
        )

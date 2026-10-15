"""Tests for EDT-based seed detection algorithm.

Pure algorithm tests — no Qt, no napari required.
Tests cover peak detection, threshold filtering, and seed volume generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.seed_detection import (
    detect_seed_peaks,
    peaks_to_seed_volume,
    threshold_peaks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sphere(shape, center, radius, label_id=1):
    """Create a volume with a spherical label."""
    vol = np.zeros(shape, dtype=np.int32)
    zz, yy, xx = np.ogrid[
        :shape[0], :shape[1], :shape[2]
    ]
    dist2 = (
        (zz - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (xx - center[2]) ** 2
    )
    vol[dist2 <= radius ** 2] = label_id
    return vol


def _make_dumbbell(shape, center1, center2, radius, neck_radius, label_id=1):
    """Create a dumbbell shape: two spheres connected by a thin neck."""
    vol = np.zeros(shape, dtype=np.int32)
    zz, yy, xx = np.ogrid[
        :shape[0], :shape[1], :shape[2]
    ]
    # Sphere 1
    d1 = (zz - center1[0]) ** 2 + (yy - center1[1]) ** 2 + (xx - center1[2]) ** 2
    vol[d1 <= radius ** 2] = label_id
    # Sphere 2
    d2 = (zz - center2[0]) ** 2 + (yy - center2[1]) ** 2 + (xx - center2[2]) ** 2
    vol[d2 <= radius ** 2] = label_id
    # Neck: cylinder connecting centers along Y axis
    mid_y = (center1[1] + center2[1]) // 2
    y_lo = min(center1[1], center2[1])
    y_hi = max(center1[1], center2[1])
    for y in range(y_lo, y_hi + 1):
        dist_cyl = (zz - center1[0]) ** 2 + (xx - center1[2]) ** 2
        vol[:, y, :][dist_cyl[:, 0, :] <= neck_radius ** 2] = label_id
    return vol


# ===========================================================================
# TestDetectSeedPeaks
# ===========================================================================

class TestDetectSeedPeaks:
    """Tests for detect_seed_peaks()."""

    def test_dumbbell_finds_two_peaks(self):
        """A dumbbell shape should yield 2 peaks (one per blob)."""
        vol = _make_dumbbell(
            (30, 60, 30),
            center1=(15, 15, 15),
            center2=(15, 45, 15),
            radius=10,
            neck_radius=2,
            label_id=5,
        )
        peaks, markers = detect_seed_peaks(vol, 5, min_distance=5)
        assert len(peaks) >= 2

    def test_single_sphere_finds_one_peak(self):
        """A single sphere should yield exactly 1 peak."""
        vol = _make_sphere((30, 30, 30), (15, 15, 15), 10, label_id=3)
        peaks, markers = detect_seed_peaks(vol, 3, min_distance=5)
        assert len(peaks) == 1

    def test_min_distance_controls_count(self):
        """Smaller min_distance may find more peaks; larger finds fewer."""
        vol = _make_dumbbell(
            (30, 60, 30),
            center1=(15, 15, 15),
            center2=(15, 45, 15),
            radius=10,
            neck_radius=2,
            label_id=1,
        )
        peaks_close, _ = detect_seed_peaks(vol, 1, min_distance=3)
        peaks_far, _ = detect_seed_peaks(vol, 1, min_distance=20)
        assert len(peaks_close) >= len(peaks_far)

    def test_empty_label_returns_empty(self):
        """Label not found → empty peaks list and zero markers."""
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        peaks, markers = detect_seed_peaks(vol, 99)
        assert len(peaks) == 0
        assert markers.shape == vol.shape
        assert np.all(markers == 0)

    def test_peaks_have_required_fields(self):
        """Each peak dict must have coord, edt_value, seed_id."""
        vol = _make_sphere((20, 20, 20), (10, 10, 10), 7, label_id=2)
        peaks, _ = detect_seed_peaks(vol, 2)
        assert len(peaks) >= 1
        for p in peaks:
            assert "coord" in p
            assert "edt_value" in p
            assert "seed_id" in p
            assert len(p["coord"]) == 3

    def test_edt_values_positive(self):
        """All peak EDT values should be positive."""
        vol = _make_sphere((20, 20, 20), (10, 10, 10), 7, label_id=1)
        peaks, _ = detect_seed_peaks(vol, 1)
        for p in peaks:
            assert p["edt_value"] > 0

    def test_seed_ids_sequential_from_one(self):
        """Seed IDs should start at 1 and increase."""
        vol = _make_dumbbell(
            (30, 60, 30),
            center1=(15, 15, 15),
            center2=(15, 45, 15),
            radius=10,
            neck_radius=2,
            label_id=1,
        )
        peaks, _ = detect_seed_peaks(vol, 1, min_distance=5)
        ids = [p["seed_id"] for p in peaks]
        assert ids == list(range(1, len(peaks) + 1))

    def test_peaks_sorted_by_edt_descending(self):
        """Peaks should be sorted by EDT value descending."""
        vol = _make_dumbbell(
            (30, 60, 30),
            center1=(15, 15, 15),
            center2=(15, 45, 15),
            radius=10,
            neck_radius=2,
            label_id=1,
        )
        peaks, _ = detect_seed_peaks(vol, 1, min_distance=5)
        if len(peaks) >= 2:
            edts = [p["edt_value"] for p in peaks]
            assert edts == sorted(edts, reverse=True)

    def test_markers_match_peaks(self):
        """Returned markers should contain one region per peak."""
        vol = _make_sphere((20, 20, 20), (10, 10, 10), 7, label_id=1)
        peaks, markers = detect_seed_peaks(vol, 1, min_distance=5)
        assert markers.shape == vol.shape
        marker_ids = set(int(v) for v in np.unique(markers) if v > 0)
        peak_ids = set(p["seed_id"] for p in peaks)
        assert marker_ids == peak_ids


# ===========================================================================
# TestThresholdPeaks
# ===========================================================================

class TestThresholdPeaks:
    """Tests for threshold_peaks()."""

    def test_threshold_zero_keeps_all(self):
        """Threshold=0 should keep all peaks."""
        peaks = [
            {"coord": (1, 2, 3), "edt_value": 5.0, "seed_id": 1},
            {"coord": (4, 5, 6), "edt_value": 3.0, "seed_id": 2},
            {"coord": (7, 8, 9), "edt_value": 1.0, "seed_id": 3},
        ]
        result = threshold_peaks(peaks, 0.0)
        assert len(result) == 3

    def test_threshold_above_max_returns_empty(self):
        """Threshold above all EDT values → empty list."""
        peaks = [
            {"coord": (1, 2, 3), "edt_value": 5.0, "seed_id": 1},
            {"coord": (4, 5, 6), "edt_value": 3.0, "seed_id": 2},
        ]
        result = threshold_peaks(peaks, 10.0)
        assert len(result) == 0

    def test_partial_filtering(self):
        """Threshold between values keeps only those above."""
        peaks = [
            {"coord": (1, 2, 3), "edt_value": 5.0, "seed_id": 1},
            {"coord": (4, 5, 6), "edt_value": 3.0, "seed_id": 2},
            {"coord": (7, 8, 9), "edt_value": 1.0, "seed_id": 3},
        ]
        result = threshold_peaks(peaks, 2.5)
        assert len(result) == 2
        assert result[0]["seed_id"] == 1
        assert result[1]["seed_id"] == 2

    def test_seed_ids_stable(self):
        """Filtering should NOT reassign seed IDs."""
        peaks = [
            {"coord": (1, 2, 3), "edt_value": 5.0, "seed_id": 1},
            {"coord": (4, 5, 6), "edt_value": 3.0, "seed_id": 2},
            {"coord": (7, 8, 9), "edt_value": 1.0, "seed_id": 3},
        ]
        result = threshold_peaks(peaks, 2.5)
        # seed_ids should be 1 and 2, not reassigned to 1 and 2
        assert [p["seed_id"] for p in result] == [1, 2]

    def test_empty_input(self):
        """Empty peaks list → empty result."""
        assert threshold_peaks([], 5.0) == []


# ===========================================================================
# TestPeaksToSeedVolume
# ===========================================================================

class TestPeaksToSeedVolume:
    """Tests for peaks_to_seed_volume()."""

    def test_output_shape_matches(self):
        """Output shape should match the requested shape."""
        peaks = [{"coord": (5, 5, 5), "edt_value": 3.0, "seed_id": 1}]
        vol = peaks_to_seed_volume((10, 10, 10), peaks)
        assert vol.shape == (10, 10, 10)

    def test_unique_ids_per_peak(self):
        """Each peak should have its own unique ID in the volume."""
        peaks = [
            {"coord": (3, 3, 3), "edt_value": 5.0, "seed_id": 1},
            {"coord": (7, 7, 7), "edt_value": 3.0, "seed_id": 2},
        ]
        vol = peaks_to_seed_volume((10, 10, 10), peaks, sphere_radius=1)
        ids = set(int(v) for v in np.unique(vol) if v > 0)
        assert ids == {1, 2}

    def test_spheres_at_peak_coords(self):
        """Seed markers should be centered at peak coordinates."""
        peaks = [{"coord": (5, 5, 5), "edt_value": 3.0, "seed_id": 1}]
        vol = peaks_to_seed_volume((11, 11, 11), peaks, sphere_radius=2)
        assert vol[5, 5, 5] == 1
        # Nearby voxels within radius should also be marked
        assert vol[5, 5, 6] == 1

    def test_radius_controls_size(self):
        """Larger radius → more voxels marked."""
        peaks = [{"coord": (10, 10, 10), "edt_value": 3.0, "seed_id": 1}]
        vol_small = peaks_to_seed_volume((21, 21, 21), peaks, sphere_radius=1)
        vol_large = peaks_to_seed_volume((21, 21, 21), peaks, sphere_radius=3)
        assert np.count_nonzero(vol_large) > np.count_nonzero(vol_small)

    def test_empty_peaks_returns_zeros(self):
        """Empty peaks → all-zeros volume."""
        vol = peaks_to_seed_volume((10, 10, 10), [])
        assert np.all(vol == 0)

    def test_clipping_at_boundaries(self):
        """Peaks near volume edges should not cause index errors."""
        peaks = [{"coord": (0, 0, 0), "edt_value": 3.0, "seed_id": 1}]
        vol = peaks_to_seed_volume((5, 5, 5), peaks, sphere_radius=3)
        assert vol[0, 0, 0] == 1
        assert vol.shape == (5, 5, 5)

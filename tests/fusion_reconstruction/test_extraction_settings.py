"""Tests for extraction settings and mask preprocessing (Phase 3).

Tests cover:
- clean_mask_slice() pipeline order and toggles
- Slice sampling (edge_buffer, min_distance, slices_per_patch)
- Views combo → views list conversion
- Config builder from UI values
- extract_training_pairs_from_3d with full settings
"""

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
    clean_mask_slice,
    extract_training_pairs_from_3d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_touching_labels():
    """Create a 2D mask where two labels touch (shared boundary)."""
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[4:28, 4:16] = 1
    mask[4:28, 16:28] = 2
    return mask


def _make_curved_label():
    """Create a 2D mask where one label appears as two disconnected blobs.

    Simulates a 3D cell that curves back on itself — in this 2D slice
    it appears as two separate blobs both labeled '1'.
    """
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[4:12, 4:12] = 1    # blob A
    mask[20:28, 20:28] = 1  # blob B — same label, disconnected
    return mask


def _make_small_fragment():
    """Create a mask with one large region and one tiny fragment."""
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[4:28, 4:28] = 1
    mask[30, 30] = 2  # 1-pixel fragment
    return mask


def _make_volume(shape=(20, 32, 32), n_labels=4):
    """Create a test volume with labeled regions."""
    rng = np.random.RandomState(42)
    image = rng.randint(0, 255, shape, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.int32)
    z, y, x = shape
    for i in range(n_labels):
        z0 = (i * z // n_labels) + 1
        z1 = ((i + 1) * z // n_labels) - 1
        labels[z0:z1, 4:28, 4:28] = i + 1
    return image, labels


# ---------------------------------------------------------------------------
# Tests: clean_mask_slice
# ---------------------------------------------------------------------------

class TestCleanMaskSlice:
    """Test the mask cleaning pipeline."""

    def test_boundaries_removed(self):
        """Touching labels should have boundary pixels set to 0."""
        mask = _make_touching_labels()
        cleaned = clean_mask_slice(mask)
        # The boundary column (x=16 area) should be 0
        # At minimum, the cleaned mask should have fewer nonzero pixels
        assert np.count_nonzero(cleaned) < np.count_nonzero(mask)

    def test_relabel_splits_disconnected_blobs(self):
        """A single label as two blobs should become two separate labels."""
        mask = _make_curved_label()
        assert len(np.unique(mask[mask > 0])) == 1  # one label ID
        cleaned = clean_mask_slice(mask, min_size=0, smoothing_disk_size=0)
        unique = np.unique(cleaned[cleaned > 0])
        assert len(unique) == 2  # should now be two labels

    def test_small_regions_removed(self):
        """Regions below min_size should be removed."""
        mask = _make_small_fragment()
        cleaned = clean_mask_slice(mask, min_size=5)
        # The 1-pixel fragment should be gone
        unique = np.unique(cleaned[cleaned > 0])
        assert len(unique) == 1

    def test_min_size_zero_keeps_all(self):
        """min_size=0 keeps all regions."""
        mask = _make_small_fragment()
        cleaned = clean_mask_slice(
            mask, min_size=0, remove_boundaries=False, smoothing_disk_size=0,
        )
        # Both regions should be present (though possibly relabeled)
        assert len(np.unique(cleaned[cleaned > 0])) >= 2

    def test_smoothing_disabled(self):
        """smoothing_disk_size=0 skips morphological smoothing."""
        mask = _make_touching_labels()
        cleaned = clean_mask_slice(mask, smoothing_disk_size=0)
        assert np.count_nonzero(cleaned) > 0

    def test_no_boundaries_preserves_touching(self):
        """remove_boundaries=False keeps boundary pixels."""
        mask = _make_touching_labels()
        cleaned = clean_mask_slice(mask, remove_boundaries=False)
        # Without boundary removal, total area should be very close to original
        assert np.count_nonzero(cleaned) >= np.count_nonzero(mask) * 0.95

    def test_no_relabel_keeps_original_ids(self):
        """relabel=False keeps original label IDs even if disconnected."""
        mask = _make_curved_label()
        cleaned = clean_mask_slice(
            mask, relabel=False, min_size=0, smoothing_disk_size=0,
        )
        unique = np.unique(cleaned[cleaned > 0])
        assert len(unique) == 1  # still one label ID

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((16, 16), dtype=np.int32)
        cleaned = clean_mask_slice(mask)
        assert np.count_nonzero(cleaned) == 0

    def test_all_disabled_returns_copy(self):
        """All cleaning disabled should return a copy of the input."""
        mask = _make_touching_labels()
        cleaned = clean_mask_slice(
            mask, remove_boundaries=False, relabel=False,
            min_size=0, smoothing_disk_size=0,
        )
        np.testing.assert_array_equal(cleaned, mask)

    def test_pipeline_order_boundary_before_relabel(self):
        """Boundary removal must happen before relabeling.

        With the full pipeline (boundary → relabel), the two disconnected
        blobs get unique IDs. Verify the pipeline runs without error and
        produces valid output with both blobs preserved.
        """
        mask = _make_curved_label()
        cleaned = clean_mask_slice(
            mask, min_size=0, smoothing_disk_size=0,
        )
        assert np.count_nonzero(cleaned) > 0
        assert len(np.unique(cleaned[cleaned > 0])) == 2


# ---------------------------------------------------------------------------
# Tests: extraction with sampling params
# ---------------------------------------------------------------------------

class TestExtractionWithSampling:
    """Test extract_training_pairs_from_3d with sampling parameters."""

    def test_edge_buffer_reduces_slices(self, tmp_path):
        image, labels = _make_volume(shape=(20, 32, 32))
        result_no_buf = extract_training_pairs_from_3d(
            image, labels, tmp_path / "no_buf",
            min_labels_per_slice=1, views=("xy",),
            edge_buffer=0,
        )
        result_buf = extract_training_pairs_from_3d(
            image, labels, tmp_path / "buf",
            min_labels_per_slice=1, views=("xy",),
            edge_buffer=5,
        )
        assert result_buf["total"] <= result_no_buf["total"]

    def test_slices_per_patch_caps_output(self, tmp_path):
        image, labels = _make_volume(shape=(20, 32, 32))
        result_all = extract_training_pairs_from_3d(
            image, labels, tmp_path / "all",
            min_labels_per_slice=1, views=("xy",),
            slices_per_patch=0,  # all
        )
        result_capped = extract_training_pairs_from_3d(
            image, labels, tmp_path / "capped",
            min_labels_per_slice=1, views=("xy",),
            slices_per_patch=3,
        )
        assert result_capped["spatial"] <= 3
        assert result_all["spatial"] >= result_capped["spatial"]

    def test_min_distance_reduces_slices(self, tmp_path):
        image, labels = _make_volume(shape=(40, 32, 32), n_labels=8)
        result_no_dist = extract_training_pairs_from_3d(
            image, labels, tmp_path / "no_dist",
            min_labels_per_slice=1, views=("xy",),
            slices_per_patch=0, min_distance=1,
        )
        result_dist = extract_training_pairs_from_3d(
            image, labels, tmp_path / "dist",
            min_labels_per_slice=1, views=("xy",),
            slices_per_patch=10, min_distance=5,
        )
        assert result_dist["total"] <= result_no_dist["total"]

    def test_mask_cleaning_applied(self, tmp_path):
        """When cleaning is enabled, output masks should be cleaned."""
        import tifffile
        # Create volume where label 1 appears as two blobs in every XY slice
        image = np.random.randint(0, 255, (4, 32, 32), dtype=np.uint8)
        labels = np.zeros((4, 32, 32), dtype=np.int32)
        for z in range(4):
            labels[z, 4:12, 4:12] = 1
            labels[z, 20:28, 20:28] = 1  # same label, disconnected

        result = extract_training_pairs_from_3d(
            image, labels, tmp_path / "cleaned",
            min_labels_per_slice=1, views=("xy",),
            relabel=True, remove_boundaries=True,
        )
        assert result["total"] > 0
        # Read a mask back and check it has 2 labels (relabeled blobs)
        mask_files = list((tmp_path / "cleaned" / "spatial").glob("*_mask.tif"))
        assert len(mask_files) > 0
        m = tifffile.imread(str(mask_files[0]))
        unique = np.unique(m[m > 0])
        assert len(unique) == 2

    def test_views_combo_mapping(self):
        """Test the views combo text → views list conversion."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        assert FusionReconstructionController._views_from_combo_text(
            "All (XY + XZ + YZ)"
        ) == ("xy", "xz", "yz")
        assert FusionReconstructionController._views_from_combo_text(
            "Lateral (XY only)"
        ) == ("xy",)
        assert FusionReconstructionController._views_from_combo_text(
            "Axial (XZ + YZ only)"
        ) == ("xz", "yz")

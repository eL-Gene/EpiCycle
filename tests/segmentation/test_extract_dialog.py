"""Tests for Extract Features dialog data model and feature manifest.

Tests the pure-Python foundations: PCA presets, feature manifest generation,
and dialog data model logic. No Qt required.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# PCA Presets
# ---------------------------------------------------------------------------

class TestPCAPresets:
    """Verify PCA_PRESETS structure and content."""

    def test_presets_dict_exists(self):
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        assert isinstance(PCA_PRESETS, dict)

    def test_epicycle_qc_preset_exists(self):
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        assert "EpiCycle QC" in PCA_PRESETS

    def test_epicycle_qc_has_15_features(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            PCA_PRESETS, PCA_FEATURES,
        )
        preset = PCA_PRESETS["EpiCycle QC"]
        assert len(preset["features"]) == 15
        assert preset["features"] is PCA_FEATURES

    def test_epicycle_qc_mode_is_robust(self):
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        assert PCA_PRESETS["EpiCycle QC"]["mode"] == "qc"

    def test_standard_preset_exists(self):
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        assert "Standard" in PCA_PRESETS

    def test_standard_preset_features_is_none(self):
        """Standard preset = all numeric, resolved at runtime."""
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        assert PCA_PRESETS["Standard"]["features"] is None

    def test_standard_preset_mode_is_eda(self):
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        assert PCA_PRESETS["Standard"]["mode"] == "eda"

    def test_none_preset_exists(self):
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        assert "None" in PCA_PRESETS

    def test_none_preset_empty_features(self):
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        assert PCA_PRESETS["None"]["features"] == []

    def test_none_preset_mode_is_none(self):
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        assert PCA_PRESETS["None"]["mode"] is None

    def test_every_preset_has_required_keys(self):
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        for name, preset in PCA_PRESETS.items():
            assert "features" in preset, f"{name} missing 'features'"
            assert "mode" in preset, f"{name} missing 'mode'"
            assert "description" in preset, f"{name} missing 'description'"

    def test_pca_features_count_is_15(self):
        """Guard against accidental additions/removals."""
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES
        assert len(PCA_FEATURES) == 15


# ---------------------------------------------------------------------------
# Feature Manifest
# ---------------------------------------------------------------------------

class TestFeatureManifest:
    """Verify get_feature_manifest() output."""

    def test_returns_dict(self):
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        manifest = get_feature_manifest()
        assert isinstance(manifest, dict)

    def test_core_categories_present(self):
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        manifest = get_feature_manifest()
        expected = {
            "Morphology", "Shape", "Surface",
            "Probability", "Neighbors", "Skeleton",
        }
        assert expected.issubset(set(manifest.keys()))
        # Intensity and Membrane removed — now per-channel only
        assert "Intensity" not in manifest
        assert "Membrane" not in manifest

    def test_each_category_has_features(self):
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        manifest = get_feature_manifest()
        for cat, feats in manifest.items():
            assert isinstance(feats, list), f"{cat} should be a list"
            assert len(feats) > 0, f"{cat} should not be empty"

    def test_no_channels_by_default(self):
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        manifest = get_feature_manifest()
        channel_cats = [k for k in manifest if k.startswith("Channel:")]
        assert len(channel_cats) == 0

    def test_channels_added_with_configs(self):
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        channels = [
            {"index": 0, "prefix": "membrane"},
            {"index": 1, "prefix": "nuclear"},
        ]
        manifest = get_feature_manifest(checked_channels=channels)
        assert "Channel: Membrane" in manifest
        assert "Channel: Nuclear" in manifest

    def test_channel_features_have_correct_prefix(self):
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        channels = [{"index": 0, "prefix": "dapi"}]
        manifest = get_feature_manifest(checked_channels=channels)
        dapi_feats = manifest["Channel: Dapi"]
        # Column pattern: {prefix}_intensity_mean, {prefix}_intensity_median, ...
        assert all(f.startswith("dapi_") for f in dapi_feats)

    def test_channel_features_have_11_stats(self):
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        channels = [{"index": 0, "prefix": "gfp"}]
        manifest = get_feature_manifest(checked_channels=channels)
        assert len(manifest["Channel: Gfp"]) == 11

    def test_manifest_excludes_metadata_columns(self):
        """label, is_border_cell, qc_tier, etc. should not appear."""
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        manifest = get_feature_manifest()
        all_feats = [f for feats in manifest.values() for f in feats]
        assert "label" not in all_feats
        assert "is_border_cell" not in all_feats
        assert "qc_tier" not in all_feats
        assert "label_status" not in all_feats

    def test_position_category_present(self):
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        manifest = get_feature_manifest()
        assert "Position" in manifest
        assert "centroid-0" in manifest["Position"]

    def test_composites_category_present(self):
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        manifest = get_feature_manifest()
        assert "Composites" in manifest
        assert "qc_score" in manifest["Composites"]


class TestChannelCategorization:
    """Verify dynamic channel feature categorization in the EDA table."""

    def test_extract_channel_prefix_simple(self):
        from napari_epicycle.segmentation.analysis_widget import (
            _extract_channel_prefix,
        )
        assert _extract_channel_prefix("membrane_intensity_mean") == "membrane"
        assert _extract_channel_prefix("nuclei_intensity_std") == "nuclei"

    def test_extract_channel_prefix_boundary(self):
        from napari_epicycle.segmentation.analysis_widget import (
            _extract_channel_prefix,
        )
        assert _extract_channel_prefix("membrane_boundary_intensity_mean") == "membrane"
        assert _extract_channel_prefix("ch0_interior_intensity_mean") == "ch0"

    def test_extract_channel_prefix_z_variation(self):
        from napari_epicycle.segmentation.analysis_widget import (
            _extract_channel_prefix,
        )
        assert _extract_channel_prefix("membrane_z_intensity_variation") == "membrane"
        assert _extract_channel_prefix("gfp_z_intensity_variation") == "gfp"

    def test_extract_channel_prefix_ratio(self):
        from napari_epicycle.segmentation.analysis_widget import (
            _extract_channel_prefix,
        )
        assert _extract_channel_prefix("ch2_boundary_interior_ratio") == "ch2"

    def test_non_channel_returns_none(self):
        from napari_epicycle.segmentation.analysis_widget import (
            _extract_channel_prefix,
        )
        assert _extract_channel_prefix("area") is None
        assert _extract_channel_prefix("neighbor_count") is None
        assert _extract_channel_prefix("skeleton_branches") is None

    def test_all_11_suffixes_detected(self):
        """All 11 per-channel features should resolve to the same prefix."""
        from napari_epicycle.segmentation.analysis_widget import (
            _extract_channel_prefix,
        )
        features = [
            "test_intensity_mean", "test_intensity_median",
            "test_intensity_std", "test_intensity_min",
            "test_intensity_max", "test_intensity_sum",
            "test_intensity_cv",
            "test_boundary_intensity_mean",
            "test_interior_intensity_mean",
            "test_boundary_interior_ratio",
            "test_z_intensity_variation",
        ]
        for feat in features:
            assert _extract_channel_prefix(feat) == "test", f"Failed for {feat}"

    def test_total_feature_count_reasonable(self):
        """Core features (no channels) should be 40+."""
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        manifest = get_feature_manifest()
        total = sum(len(feats) for feats in manifest.values())
        assert total >= 40

    def test_category_order_preserved(self):
        """Categories should follow _CATEGORY_ORDER."""
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        manifest = get_feature_manifest()
        keys = list(manifest.keys())
        # Morphology should come before Shape, Shape before Surface, etc.
        if "Morphology" in keys and "Shape" in keys:
            assert keys.index("Morphology") < keys.index("Shape")
        if "Surface" in keys and "Probability" in keys:
            assert keys.index("Surface") < keys.index("Probability")

    def test_channels_come_after_core_categories(self):
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )
        channels = [{"index": 0, "prefix": "gfp"}]
        manifest = get_feature_manifest(checked_channels=channels)
        keys = list(manifest.keys())
        channel_idx = keys.index("Channel: Gfp")
        # All core categories should precede channel categories
        for core_cat in ["Morphology", "Shape", "Surface"]:
            if core_cat in keys:
                assert keys.index(core_cat) < channel_idx

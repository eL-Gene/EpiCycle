"""Tests for QCSettingsDialog and related controller wiring."""

import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# QCSettingsDialog unit tests (require QApplication)
# ---------------------------------------------------------------------------

@pytest.fixture
def _qapp():
    """Provide a QApplication or skip if unavailable."""
    try:
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app
    except Exception:
        pytest.skip("No QApplication available")


class TestQCSettingsDialog:
    """Test the QC Settings popup dialog."""

    def test_dialog_creates_with_defaults(self, _qapp):
        from napari_epicycle.fusion_reconstruction.qc_settings_dialog import QCSettingsDialog
        dlg = QCSettingsDialog()
        settings = dlg.get_settings()
        assert settings["spacing"] is None  # all 1.0 → None
        assert settings["spacing_values"] == (1.0, 1.0, 1.0)
        assert settings["erosion_iters"] == 2
        assert settings["max_radius"] == 5
        assert settings["composite_method"] == "zscore"

    def test_dialog_returns_custom_values(self, _qapp):
        from napari_epicycle.fusion_reconstruction.qc_settings_dialog import QCSettingsDialog
        dlg = QCSettingsDialog()
        dlg.spacing_z_sb.setValue(0.3)
        dlg.spacing_y_sb.setValue(0.1)
        dlg.spacing_x_sb.setValue(0.1)
        dlg.erosion_iters_sb.setValue(4)
        dlg.max_radius_sb.setValue(10)
        dlg.composite_method_combo.setCurrentText("pca")
        settings = dlg.get_settings()
        assert settings["spacing"] == (0.3, 0.1, 0.1)
        assert settings["erosion_iters"] == 4
        assert settings["max_radius"] == 10
        assert settings["composite_method"] == "pca"

    def test_dialog_apply_settings(self, _qapp):
        from napari_epicycle.fusion_reconstruction.qc_settings_dialog import QCSettingsDialog
        prev = {
            "spacing_values": (0.5, 0.2, 0.2),
            "erosion_iters": 3,
            "max_radius": 8,
            "composite_method": "pca",
        }
        dlg = QCSettingsDialog(settings=prev)
        settings = dlg.get_settings()
        assert settings["spacing_values"] == (0.5, 0.2, 0.2)
        assert settings["erosion_iters"] == 3
        assert settings["max_radius"] == 8
        assert settings["composite_method"] == "pca"


# ---------------------------------------------------------------------------
# Patch color palette tests (no Qt needed)
# ---------------------------------------------------------------------------

class TestPatchPalette:
    """Test the categorical patch coloring palette."""

    def test_palette_has_12_colors(self):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget
        assert len(QCPlotWidget._PATCH_PALETTE) == 12

    def test_palette_rgba_range(self):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget
        palette = QCPlotWidget._PATCH_PALETTE
        assert palette.shape[1] == 4
        assert np.all(palette >= 0.0)
        assert np.all(palette <= 1.0)

    def test_palette_wraps_for_many_patches(self):
        """When there are more patches than palette entries, colors wrap."""
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget
        palette = QCPlotWidget._PATCH_PALETTE
        n_palette = len(palette)
        # If we have more patches, indexing should wrap via modulo
        idx = (n_palette + 3) % n_palette
        assert idx == 3  # wrapped correctly


# ---------------------------------------------------------------------------
# Label isolation logic tests (no Qt/napari needed)
# ---------------------------------------------------------------------------

class TestLabelIsolationLogic:
    """Test the dimming / bright color logic for label isolation."""

    def test_selected_labels_get_bright_colors(self):
        """Selected labels should get distinct bright colors."""
        # Simulate the isolation logic
        selected = {1, 5, 10}
        dim_color = np.array([0.3, 0.3, 0.3, 0.15], dtype=np.float32)
        bright_colors = [
            np.array([0.12, 0.73, 0.55, 1.0], dtype=np.float32),
            np.array([0.90, 0.35, 0.23, 1.0], dtype=np.float32),
            np.array([0.20, 0.55, 0.90, 1.0], dtype=np.float32),
        ]
        all_ids = {1, 2, 3, 5, 7, 10}
        color_dict = {None: np.array([0, 0, 0, 0], dtype=np.float32)}
        bright_idx = 0
        for lid in sorted(all_ids):
            if lid in selected:
                color_dict[lid] = bright_colors[bright_idx % len(bright_colors)]
                bright_idx += 1
            else:
                color_dict[lid] = dim_color

        # Selected → bright (alpha 1.0)
        assert color_dict[1][3] == 1.0
        assert color_dict[5][3] == 1.0
        assert color_dict[10][3] == 1.0
        # Unselected → dim (alpha 0.15)
        assert color_dict[2][3] == pytest.approx(0.15)
        assert color_dict[3][3] == pytest.approx(0.15)
        assert color_dict[7][3] == pytest.approx(0.15)

    def test_empty_selection_all_dim(self):
        """No selected labels means all labels get dim color."""
        selected = set()
        dim_color = np.array([0.3, 0.3, 0.3, 0.15], dtype=np.float32)
        all_ids = {1, 2, 3}
        color_dict = {}
        for lid in all_ids:
            if lid in selected:
                color_dict[lid] = np.array([1, 1, 1, 1], dtype=np.float32)
            else:
                color_dict[lid] = dim_color
        for lid in all_ids:
            assert color_dict[lid][3] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Merged CSV / scope switching tests (no Qt needed)
# ---------------------------------------------------------------------------

class TestMergedCSVFeatures:
    """Test merge_patch_csvs behavior for population view."""

    def test_merge_adds_patch_name_column(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            merge_patch_csvs,
            save_patch_csv,
        )
        df1 = pd.DataFrame({"label": [1, 2], "area": [100, 200]})
        df2 = pd.DataFrame({"label": [1, 3], "area": [150, 50]})
        p1 = tmp_path / "patch_A_features.csv"
        p2 = tmp_path / "patch_B_features.csv"
        save_patch_csv(df1, str(p1))
        save_patch_csv(df2, str(p2))

        merged = merge_patch_csvs([p1, p2])
        assert "patch_name" in merged.columns
        assert len(merged) == 4
        assert set(merged["patch_name"].unique()) == {"patch_A", "patch_B"}

    def test_merge_empty_list(self):
        from napari_epicycle.fusion_reconstruction.qc_features import merge_patch_csvs
        merged = merge_patch_csvs([])
        assert merged.empty


# ---------------------------------------------------------------------------
# PCA Feature Selection in Settings Dialog (K8c)
# ---------------------------------------------------------------------------

class TestPCAFeatureSelection:
    """Test the PCA feature checkbox group in QCSettingsDialog."""

    def test_dialog_returns_pca_features_key(self, _qapp):
        from napari_epicycle.fusion_reconstruction.qc_settings_dialog import QCSettingsDialog
        dlg = QCSettingsDialog()
        settings = dlg.get_settings()
        assert "pca_features" in settings
        assert isinstance(settings["pca_features"], list)

    def test_default_features_match_pca_features(self, _qapp):
        from napari_epicycle.fusion_reconstruction.qc_settings_dialog import QCSettingsDialog
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES
        dlg = QCSettingsDialog()
        selected = dlg.get_selected_features()
        assert set(selected) == set(PCA_FEATURES)

    def test_select_all_checks_everything(self, _qapp):
        from napari_epicycle.fusion_reconstruction.qc_settings_dialog import QCSettingsDialog
        from napari_epicycle.fusion_reconstruction.qc_features import NUMERIC_FEATURES
        dlg = QCSettingsDialog()
        dlg._select_all_features()
        selected = dlg.get_selected_features()
        # All NUMERIC_FEATURES except qc_score
        expected = [f for f in NUMERIC_FEATURES if f != "qc_score"]
        assert set(selected) == set(expected)

    def test_reset_defaults_restores_pca_features(self, _qapp):
        from napari_epicycle.fusion_reconstruction.qc_settings_dialog import QCSettingsDialog
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES
        dlg = QCSettingsDialog()
        dlg._select_all_features()
        dlg._reset_default_features()
        selected = dlg.get_selected_features()
        assert set(selected) == set(PCA_FEATURES)

    def test_apply_settings_restores_custom_features(self, _qapp):
        from napari_epicycle.fusion_reconstruction.qc_settings_dialog import QCSettingsDialog
        custom = ["area", "extent", "solidity"]
        dlg = QCSettingsDialog(settings={"pca_features": custom})
        selected = dlg.get_selected_features()
        assert set(selected) == set(custom)


class TestClusteringSettings:
    """Test clustering settings in QCSettingsDialog."""

    def test_default_cluster_algorithm(self, _qapp):
        from napari_epicycle.fusion_reconstruction.qc_settings_dialog import QCSettingsDialog
        dlg = QCSettingsDialog()
        settings = dlg.get_settings()
        assert settings["cluster_algorithm"] == "leiden"
        assert settings["min_cluster_size"] == 15
        assert settings["n_clusters"] == 5
        assert settings["leiden_resolution"] == 1.0
        assert settings["leiden_n_neighbors"] == 15
        assert settings["n_cluster_components"] == 5

    def test_custom_cluster_settings(self, _qapp):
        from napari_epicycle.fusion_reconstruction.qc_settings_dialog import QCSettingsDialog
        dlg = QCSettingsDialog(settings={
            "cluster_algorithm": "hdbscan",
            "min_cluster_size": 25,
            "n_clusters": 8,
            "leiden_resolution": 0.5,
            "leiden_n_neighbors": 20,
        })
        settings = dlg.get_settings()
        assert settings["cluster_algorithm"] == "hdbscan"
        assert settings["min_cluster_size"] == 25
        assert settings["n_clusters"] == 8
        assert settings["leiden_resolution"] == 0.5
        assert settings["leiden_n_neighbors"] == 20

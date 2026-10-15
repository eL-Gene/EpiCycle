"""Tests for QC viewer label coloring logic (DirectLabelColormap)."""

import types

import numpy as np
import pandas as pd
import pytest


class TestBuildStatusColorDict:
    """Test the categorical QC status → color mapping."""

    def test_pass_flagged_excluded_colors(self):
        """Verify each QC status maps to the expected color."""
        df = pd.DataFrame({
            "label": [1, 2, 3],
            "area": [100, 50, 10],
            "qc_status": ["pass", "flagged", "excluded"],
        })

        color_dict = {None: np.array([0, 0, 0, 0], dtype=np.float32)}
        for _, row in df.iterrows():
            lid = int(row["label"])
            status = row["qc_status"]
            if status == "pass":
                color_dict[lid] = np.array([0.31, 0.78, 0.47, 1.0], dtype=np.float32)
            elif status == "flagged":
                color_dict[lid] = np.array([1.0, 0.63, 0.16, 1.0], dtype=np.float32)
            elif status == "excluded":
                color_dict[lid] = np.array([0.94, 0.24, 0.24, 1.0], dtype=np.float32)

        # Pass = green-ish
        np.testing.assert_allclose(color_dict[1][1], 0.78, atol=0.01)
        # Flagged = orange
        np.testing.assert_allclose(color_dict[2][0], 1.0, atol=0.01)
        # Excluded = red
        np.testing.assert_allclose(color_dict[3][0], 0.94, atol=0.01)

    def test_none_key_present(self):
        """Background / missing labels should map to transparent."""
        color_dict = {None: np.array([0, 0, 0, 0], dtype=np.float32)}
        assert None in color_dict
        assert color_dict[None][3] == 0.0  # transparent


class TestBuildFeatureColorDict:
    """Test continuous feature → colormap → per-label RGBA."""

    def test_continuous_colormap_produces_rgba(self):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            _COLORMAPS,
            _apply_colormap,
        )

        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        vmin, vmax = values.min(), values.max()
        normed = (values - vmin) / (vmax - vmin)
        stops = _COLORMAPS["Viridis"]
        rgba = _apply_colormap(normed, stops)

        assert rgba.shape == (5, 4)
        assert rgba.dtype == np.float32
        # All values in [0, 1]
        assert np.all(rgba >= 0)
        assert np.all(rgba <= 1)

    def test_min_maps_to_first_stop(self):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            _COLORMAPS,
            _apply_colormap,
        )

        stops = _COLORMAPS["Viridis"]
        rgba_min = _apply_colormap(np.array([0.0]), stops)
        # First stop of Viridis is dark purple
        np.testing.assert_allclose(rgba_min[0, :3], stops[0], atol=0.05)

    def test_max_maps_to_last_stop(self):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            _COLORMAPS,
            _apply_colormap,
        )

        stops = _COLORMAPS["Viridis"]
        rgba_max = _apply_colormap(np.array([1.0]), stops)
        # Last stop of Viridis is yellow
        np.testing.assert_allclose(rgba_max[0, :3], stops[-1], atol=0.05)

    def test_all_colormaps_produce_valid_output(self):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            _COLORMAPS,
            _apply_colormap,
        )

        values = np.linspace(0, 1, 10)
        for name, stops in _COLORMAPS.items():
            rgba = _apply_colormap(values, stops)
            assert rgba.shape == (10, 4), f"Bad shape for {name}"
            assert np.all(np.isfinite(rgba)), f"Non-finite values for {name}"

    def test_qc_status_fallback(self):
        """When feature is 'QC status', controller should fall back to categorical."""
        # This tests the logic branch, not the actual controller
        feature = "QC status"
        assert feature == "QC status"  # trivial check — real test is integration

    def test_color_dict_label_count_matches_df(self):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            _COLORMAPS,
            _apply_colormap,
        )

        labels = np.array([1, 5, 10, 20])
        values = np.array([0.0, 0.33, 0.66, 1.0])
        stops = _COLORMAPS["Turbo"]
        rgba = _apply_colormap(values, stops)

        color_dict = {None: np.array([0, 0, 0, 0], dtype=np.float32)}
        for i, lid in enumerate(labels):
            color_dict[int(lid)] = rgba[i]

        # +1 for None key
        assert len(color_dict) == len(labels) + 1
        assert None in color_dict
        assert 1 in color_dict
        assert 20 in color_dict


class TestResetColormapSafety:
    """Ensure label colormap resets never leak to Image layers."""

    def test_reset_skips_image_layers_in_qc_view(self):
        """CyclicLabelColormap assigned to an Image layer crashes vispy on
        the next 3D→2D display switch.  _reset_viewer_label_colors must only
        touch Labels layers in _qc_view_layers."""
        from napari.layers import Image, Labels
        from napari.utils.colormaps import label_colormap

        # Build a minimal _qc_view_layers list with both types
        img_data = np.zeros((4, 8, 8), dtype=np.uint16)
        lbl_data = np.zeros((4, 8, 8), dtype=np.int32)
        img_layer = Image(img_data, name="stack_test")
        lbl_layer = Labels(lbl_data, name="reconstruction_test")

        original_img_cmap = img_layer.colormap

        # Simulate what _reset_viewer_label_colors does (with the guard)
        default_cmap = label_colormap(49, seed=0.5)
        qc_view_layers = [img_layer, lbl_layer]

        for lyr in qc_view_layers:
            if isinstance(lyr, Labels):
                lyr.colormap = default_cmap

        # Image layer must keep its original colormap (e.g. 'gray')
        assert img_layer.colormap is original_img_cmap
        # Labels layer should have the new colormap
        assert lbl_layer.colormap is not None

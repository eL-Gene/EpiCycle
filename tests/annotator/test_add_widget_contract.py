"""Contract tests for the de-novo Add tab in AdvancedFeaturesWidget.

Source-inspection / class-attribute tests only (Qt-free — no widget
instantiation, which needs a QApplication and can crash on teardown). These
pin the wiring contract the annotator orchestration depends on.
"""

from __future__ import annotations

import inspect


def _w():
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    return AdvancedFeaturesWidget


class TestAddTabWiring:
    def test_add_tab_appended_after_refine(self):
        """Add tab must be added AFTER Refine so Refine keeps index 5."""
        src = inspect.getsource(_w()._build_3d_curation_sections)
        i_refine = src.index('_build_refine_section(), "Refine"')
        i_add = src.index('_build_add_denovo_section(), "Add"')
        assert i_add > i_refine, "Add tab must be appended after Refine"

    def test_add_not_in_hidden_tabs(self):
        """Add is a real tab, not a hidden stub."""
        src = inspect.getsource(_w()._build_3d_curation_sections)
        # the hidden set must not contain "Add"
        hidden_line = [l for l in src.splitlines() if "_hidden_tabs = {" in l][0]
        assert '"Add"' not in hidden_line

    def test_public_api_methods_exist(self):
        W = _w()
        for m in (
            "_build_add_denovo_section",
            "set_add_callbacks",
            "update_add_ui_state",
            "refresh_add_image_combo",
            "get_add_params",
            "get_selected_add_pieces",
            "_ensure_add_plot",
            "_update_add_histogram",
            "_set_add_threshold",
            "_on_add_box_slider_changed",
            "_on_add_line_moved",
        ):
            assert hasattr(W, m), f"missing {m}"


class TestAddCallbackContract:
    def test_set_add_callbacks_signature(self):
        sig = inspect.signature(_w().set_add_callbacks)
        params = set(sig.parameters)
        for p in (
            "on_start", "on_box_changed", "on_clahe_toggled",
            "on_threshold_changed", "on_otsu_reset", "on_preview",
            "on_accept", "on_reject", "on_advanced_changed",
            "on_select_piece", "on_refine_piece", "on_merge_pieces",
            "on_add_another", "on_undo_all", "on_batch_refine",
        ):
            assert p in params, f"set_add_callbacks missing {p}"

    def test_refine_all_is_wired_to_on_batch_refine(self):
        """The kwarg that sat unused now drives
        "Refine all ▸"."""
        src = inspect.getsource(_w().set_add_callbacks)
        assert '_wire("add_batch_refine_btn", on_batch_refine)' in src

    def test_update_add_ui_state_keywords(self):
        sig = inspect.signature(_w().update_add_ui_state)
        params = set(sig.parameters)
        for p in (
            "box_active", "box_shape", "box_center", "box_size",
            "histogram_data", "threshold_value", "results_text",
            "preview_enabled", "accept_enabled", "accept_count",
            "reject_enabled", "start_enabled", "status_text",
            "review_active", "piece_ids", "marker_enabled",
            "add_batch_refine_enabled",
        ):
            assert p in params, f"update_add_ui_state missing {p}"

    def test_box_changed_fires_six_tuple(self):
        """on_box_changed must be invoked with (cz,cy,cx,sz,sy,sx)."""
        src = inspect.getsource(_w()._on_add_box_slider_changed)
        for key in ("cz", "cy", "cx", "sz", "sy", "sx"):
            assert f's["{key}"].value()' in src, f"box slider {key} not read"


class TestAddParams:
    def test_get_add_params_keys(self):
        src = inspect.getsource(_w().get_add_params)
        for key in (
            "channel", "clahe", "clahe_clip", "min_size", "separate",
            "min_distance", "clear_border", "edge_margin",
            # 2
            "preset", "click_radius_um", "core_erode_um",
            "dark_clearance_um", "dark_otsu_factor", "nuclei_threshold",
            "dark_outside",
        ):
            assert f'"{key}"' in src, f"get_add_params missing {key}"
        # Collected-but-never-passed "connectivity" and the
        # Boundary stub's "marker" are removed
        for gone in ("connectivity", "marker"):
            assert f'"{gone}"' not in src, f"get_add_params still has {gone}"

    def test_marker_filled_default_boundary_phase2(self):
        """Marker combo: Filled selectable, Boundary marked not-yet-available."""
        # The "Boundary -- soon" stub and its combo
        # are removed; the Cells preset replaced it.
        src = inspect.getsource(_w()._build_add_denovo_section)
        assert "add_marker_combo" not in src
        assert "Boundary" not in src
        assert "add_preset_combo" in src

    def test_review_swaps_tune(self):
        """review_active must hide TUNE and show REVIEW (swap in place)."""
        # the swap lives in the one visibility pass (Add rebuild 4.1)
        src = inspect.getsource(_w().update_add_ui_state)
        assert "review_active" in src and "_apply_add_visibility" in src
        vsrc = inspect.getsource(_w()._apply_add_visibility)
        assert "_add_review_container" in vsrc
        assert "_add_tune_container" in vsrc

"""Tests for Step 4 (3D Stack Segmentation) UI structure and theme.

Theme tests are pure-Python (no Qt). Widget structure tests are skipped
in CI and can be run manually with a display + QApplication.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Theme constants (pure Python — no Qt required)
# ---------------------------------------------------------------------------

class TestStackStatusThemeConstants:
    """STACK_STATUS_* (owner decisions 6, 14, 22)."""

    EXPECTED_STATUSES = [
        "unprocessed", "predicted", "segmented", "reconstructed",
        "qc_extracted", "smoothed", "finalized", "failed",
    ]
    SHARED = ("unprocessed", "predicted", "reconstructed", "finalized")

    def test_every_status_has_colour_icon_dots_and_word(self):
        from napari_epicycle.ui_shell.theme import (
            STACK_STATUS_COLORS, STACK_STATUS_DOTS, STACK_STATUS_ICONS,
            STACK_STATUS_WORDS,
        )
        for status in self.EXPECTED_STATUSES:
            assert status in STACK_STATUS_COLORS, status
            assert status in STACK_STATUS_ICONS, status
            assert status in STACK_STATUS_DOTS, status
            assert status in STACK_STATUS_WORDS, status

    def test_colors_are_hex_strings(self):
        from napari_epicycle.ui_shell.theme import STACK_STATUS_COLORS
        for status, color in STACK_STATUS_COLORS.items():
            assert color.startswith("#") and len(color) == 7, status

    def test_icons_are_mdi6_strings(self):
        from napari_epicycle.ui_shell.theme import STACK_STATUS_ICONS
        for status, icon in STACK_STATUS_ICONS.items():
            assert icon.startswith("mdi6."), status

    def test_shared_words_take_step3s_colour_and_dots(self):
        from napari_epicycle.ui_shell.theme import (
            PATCH_STATUS_COLORS, PATCH_STATUS_DOTS, STACK_STATUS_COLORS,
            STACK_STATUS_DOTS,
        )
        for status in self.SHARED:
            assert STACK_STATUS_COLORS[status] == PATCH_STATUS_COLORS[status]
            assert STACK_STATUS_DOTS[status] == PATCH_STATUS_DOTS[status]
        assert STACK_STATUS_COLORS["predicted"] == "#5ba4cf"
        assert STACK_STATUS_COLORS["reconstructed"] == "#d4a843"

    def test_step4s_own_statuses(self):
        from napari_epicycle.ui_shell.theme import (
            STACK_STATUS_COLORS, STACK_STATUS_ICONS, STACK_STATUS_WORDS,
        )
        assert STACK_STATUS_COLORS["qc_extracted"] == "#a78bca"
        assert STACK_STATUS_WORDS["qc_extracted"] == "measured"
        assert STACK_STATUS_COLORS["failed"] == "#e07070"
        assert STACK_STATUS_ICONS["unprocessed"] == "mdi6.circle-outline"

    def test_legacy_statuses_read_as_failed_and_measured(self):
        from napari_epicycle.ui_shell.theme import STACK_STATUS_WORDS
        assert STACK_STATUS_WORDS["segmented"] == "failed"
        assert STACK_STATUS_WORDS["smoothed"] == "measured"

    def test_stack_and_patch_dicts_are_separate(self):
        """Step 3 and Step 4 status dicts must not be the same object."""
        from napari_epicycle.ui_shell.theme import (
            PATCH_STATUS_COLORS, STACK_STATUS_COLORS,
        )
        assert PATCH_STATUS_COLORS is not STACK_STATUS_COLORS

    def test_protected_statuses_are_gone(self):
        import napari_epicycle.ui_shell.theme as th
        assert not hasattr(th, "STACK_PROTECTED_STATUSES")


# ---------------------------------------------------------------------------
# UI module importability (no widget instantiation — avoids Qt crash)
# ---------------------------------------------------------------------------

class TestSegmentStackModuleImport:
    """Verify the ui module can be imported and has expected classes."""

    def test_segment_stack_class_exists(self):
        from napari_epicycle.segmentation.ui import SegmentStack
        assert SegmentStack is not None

    def test_controller_class_exists(self):
        from napari_epicycle.segmentation.controller import SegmentStackController
        assert SegmentStackController is not None

    def test_ui_source_has_channel_combos(self):
        import inspect
        from napari_epicycle.segmentation.ui import SegmentStack
        src = inspect.getsource(SegmentStack._build_segment_settings)
        assert "chan1_combo" in src
        assert "chan2_combo" in src
        # axes are per stack now: no global axes combo
        assert "axes_combo" not in src

    def test_ui_source_has_reconstruction_params(self):
        import inspect
        from napari_epicycle.segmentation.ui import SegmentStack
        src = (inspect.getsource(SegmentStack._build_advanced)
               + inspect.getsource(SegmentStack._make_group_widgets)
               + inspect.getsource(SegmentStack._build_segment_settings))
        assert "method_combo" in src
        assert "do_mp_cb" in src
        assert "start_from_combo" in src       # replaces the two presets
        assert "gd_decay_sb" in src  # gradient descent
        assert "ccp_ksize_sb" in src  # combine_cell_probs
        assert "post_small_obj_sb" in src  # postprocess_binary
        assert "cc_min_area_sb" in src  # connected_component
        assert "sf_min_size_sb" in src  # size_filters

    def _mt_src(self):
        import inspect
        from napari_epicycle.segmentation.ui import SegmentStack
        return inspect.getsource(SegmentStack._build_measure_panel)

    def test_ui_source_has_the_one_qc_bar(self):
        """B1b: ③ has one bar; the three old bars are gone."""
        src = self._mt_src()
        assert "qc_progress" in src
        for gone in ("extract_progress", "smooth_progress", "mesh_progress"):
            assert gone not in src

    def test_ui_source_has_smooth_section(self):
        src = self._mt_src()
        assert "smooth_all_btn" in src
        assert "smooth_strength_combo" in src

    def test_spacing_inputs_left_measure_and_tidy(self):
        """B1a moved them into SetSpacingDialog (v2.7)."""
        src = self._mt_src()
        for gone in ("spacing_z_sb", "spacing_auto_btn",
                     "z_orientation_combo"):
            assert gone not in src

    def test_ui_source_has_review_block(self):
        src = self._mt_src()
        assert "delete_bad_btn" in src
        assert "undo_btn" in src
        assert "save_tiers_btn" in src

    def test_ui_source_has_smooth_sigma_params(self):
        src = self._mt_src()
        assert "smooth_sigma_z_sb" in src
        assert "smooth_sigma_xy_sb" in src
        assert "smooth_preserve_topology_cb" in src

    def test_finalize_and_export_section(self):
        src = self._mt_src()
        assert "finalize_btn" in src
        assert "export_btn" in src
        assert "export_all_btn" not in src

    def test_status_palette_uses_stack_colors(self):
        from napari_epicycle.segmentation.ui import _STATUS_COLORS
        from napari_epicycle.ui_shell.theme import STACK_STATUS_COLORS
        # Verify _STATUS_COLORS was built from STACK_STATUS_COLORS
        assert len(_STATUS_COLORS) == len(STACK_STATUS_COLORS)
        for status in STACK_STATUS_COLORS:
            assert status in _STATUS_COLORS


# ---------------------------------------------------------------------------
# Controller signal wiring (pure inspection, no Qt widgets)
# ---------------------------------------------------------------------------

class TestControllerSignalContract:
    """Verify the controller references the expected UI widget attributes.

    This uses source inspection rather than widget instantiation to
    avoid Qt teardown crashes.
    """

    def _get_connect_source(self):
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        return inspect.getsource(SegmentStackController._connect_signals)

    def test_wires_browse_buttons(self):
        src = self._get_connect_source()
        assert "browse_workdir_btn" in src
        assert "add_folder_btn" in src          # replaces browse_input_btn
        assert "model_change_btn" in src        # replaces the model pack

    def test_wires_run_execution(self):
        src = self._get_connect_source()
        assert "predict_btn" in src
        assert "reconstruct_btn" in src
        assert "cancel_btn" in src

    def test_wires_defaults_presets(self):
        src = self._get_connect_source()
        assert "start_from_combo" in src
        assert "epicycle_defaults_btn" not in src

    def test_no_spacing_auto_detect(self):
        src = self._get_connect_source()
        assert "spacing_auto_btn" not in src

    def test_wires_qc_load(self):
        """A ③ row click loads; the Load button is gone."""
        src = self._get_connect_source()
        assert "mt_load_requested" in src
        assert "load_reconstruction_btn" not in src

    def test_wires_feature_extraction(self):
        src = self._get_connect_source()
        assert "extract_features_btn" in src
        assert "extract_all_features_btn" not in src

    def test_wires_tier_buttons(self):
        """Delete/Save wired in _connect_signals, Mark Good/Bad in EDA dock."""
        src = self._get_connect_source()
        assert "delete_bad_btn" in src
        assert "save_tiers_btn" in src
        # Mark Good/Bad moved to EDA Workbench — wired in _open_analysis_dock
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        eda_src = inspect.getsource(SegmentStackController._open_analysis_dock)
        assert "mark_good_requested" in eda_src
        assert "mark_bad_requested" in eda_src

    def test_wires_smooth_button(self):
        src = self._get_connect_source()
        assert "smooth_all_btn" in src

    def test_smart_extract_detects_context(self):
        """Extract shows full transparency dialog with PCA presets."""
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        src = inspect.getsource(SegmentStackController._extract_features)
        assert "ExtractFeaturesDialog" in src
        assert "skip_probability" in src
        assert "_pending_border_removal" in src

    def test_wires_finalize_and_export(self):
        src = self._get_connect_source()
        assert "finalize_btn" in src
        assert "export_btn" in src
        assert "features_requested" in src and "meshes_requested" in src

    def test_lifecycle_hooks_exist(self):
        from napari_epicycle.segmentation.controller import SegmentStackController
        assert hasattr(SegmentStackController, "on_step_exit")
        assert hasattr(SegmentStackController, "on_step_enter")

    def test_controller_has_state_manager(self):
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        src = inspect.getsource(SegmentStackController.__init__)
        assert "state_manager" in src
        assert "StackStateManager" in src

    def test_extract_uses_registry(self):
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        src = inspect.getsource(SegmentStackController._run_feature_extraction)
        assert "_StackQCFeatureWorker" in src

    def test_smooth_all_uses_worker(self):
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        src = inspect.getsource(SegmentStackController._smooth_all)
        assert "_SmoothBatchWorker" in src          # F2 replaced _SmoothAllWorker

    def test_export_all_uses_merge(self):
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        src = inspect.getsource(SegmentStackController._export_all_stacks)
        assert "merge_patch_csvs" in src
        assert "stack_name" in src

    def test_mark_good_sets_tier(self):
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        src = inspect.getsource(SegmentStackController._mark_good)
        assert "good" in src

    def test_delete_bad_zeros_volume(self):
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        src = inspect.getsource(SegmentStackController._delete_bad_labels)
        assert "deleted" in src

    def test_extract_has_border_removal_dialog(self):
        """Extract dialog exposes border removal + skip prob + PCA presets."""
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        src = inspect.getsource(SegmentStackController._extract_features)
        assert "remove_border" in src
        assert "skip_probability" in src
        assert "pca_presets" in src.lower() or "PCA_PRESETS" in src

    def test_worker_classes_exist(self):
        from napari_epicycle.segmentation.controller import (
            _StackQCFeatureWorker, _SmoothBatchWorker,
        )
        assert _StackQCFeatureWorker is not None
        assert _SmoothBatchWorker is not None

    def test_worker_has_required_signals(self):
        import inspect
        from napari_epicycle.segmentation.controller import _StackQCFeatureWorker
        src = inspect.getsource(_StackQCFeatureWorker)
        assert "finished" in src
        assert "progress" in src
        assert "error" in src
        assert "log" in src
        assert "csv_saved" in src

    def test_smooth_worker_has_signals(self):
        import inspect
        from napari_epicycle.segmentation.controller import _SmoothBatchWorker
        src = inspect.getsource(_SmoothBatchWorker)
        assert "finished" in src
        assert "progress" in src
        assert "run_batch_recipe" in src            # F2: Batch Refine's path

    def test_lasso_selection_wired(self):
        """Lasso selection now wired in _open_analysis_dock (EDA owns plot)."""
        import inspect
        from napari_epicycle.segmentation.controller import SegmentStackController
        src = inspect.getsource(SegmentStackController._open_analysis_dock)
        assert "labels_selected" in src
        assert "_on_lasso_selection" in src


# ---------------------------------------------------------------------------
# Every ``self.ui.<name>`` the controller reads exists on the panel
# ---------------------------------------------------------------------------

class TestEveryControllerReadExists:
    def test_every_ui_attribute_the_controller_reads_is_built(self, qapp):
        import ast
        from pathlib import Path

        from qtpy.QtWidgets import QWidget

        import napari_epicycle.segmentation.controller as C
        from napari_epicycle.segmentation.ui import SegmentStack

        tree = ast.parse(Path(C.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "ui"
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "self"):
                names.add(node.attr)
        w = SegmentStack.__new__(SegmentStack)
        QWidget.__init__(w)
        w.viewer = None
        w._build_ui()
        try:
            missing = sorted(n for n in names if not hasattr(w, n))
            assert not missing, missing
        finally:
            w.deleteLater()

    def test_the_deleted_widgets_are_gone(self, qapp):
        from qtpy.QtWidgets import QWidget

        from napari_epicycle.segmentation.ui import SegmentStack

        w = SegmentStack.__new__(SegmentStack)
        QWidget.__init__(w)
        w.viewer = None
        w._build_ui()
        try:
            for name in ("input_stacks_dir_line", "browse_input_btn",
                         "model_pack_line", "browse_model_pack_btn",
                         "run_name_line", "clear_cache_btn", "downsample_sb",
                         "axes_combo", "diameter_sb", "compute_masks_cb",
                         "epicycle_defaults_btn", "useg3d_defaults_btn",
                         "_recon_collapsible", "_auto_model_widget",
                         "_import_widget", "_pretrained_widget"):
                assert not hasattr(w, name), name
        finally:
            w.deleteLater()


class TestShellRows:
    """F17: the stale "Finalize Model" advisory is skipped for
    Step 4; Step 5 keeps it."""

    def test_the_advisory_is_step5s_only(self, monkeypatch):
        import types

        from napari_epicycle.ui_shell import _workspace_facts
        from napari_epicycle.ui_shell.workflow_shell import (
            EpiCycleWorkflowShell,
        )
        monkeypatch.setattr(_workspace_facts, "trained_rounds",
                            lambda ws: {"lateral": 2, "axial": 1})
        fake = types.SimpleNamespace(
            _read_workspace_path=lambda: "D:/ws",
            _read_step_workspace_path=lambda step: "",
            _read_model_choice=lambda bucket: "")
        guide = EpiCycleWorkflowShell._build_step_guidance
        assert guide(fake, "apply_3d") == ""
        assert "No final model" in guide(fake, "whole_tissue")

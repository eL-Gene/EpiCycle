"""Step 2 "persistence + bucket switching" brick (2026-09-07).

User symptom: "I train something in axial, then switch to lateral, and some
UI fields don't change."

Each class below covers one ranked item from the bucket-switch audit.
All fakes are Qt-free.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

BS = chr(92)

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


# ---------------------------------------------------------------------------
# Qt-free fakes
# ---------------------------------------------------------------------------

class _Edit:
    def __init__(self, text=""):
        self._t = str(text)
        self.set_calls = 0
        self.enabled = True
        self.blocked = False
        self.block_history = []

    def text(self):
        return self._t

    def setText(self, t):
        self._t = str(t)
        self.set_calls += 1

    def setEnabled(self, v):
        self.enabled = bool(v)

    def blockSignals(self, v):
        self.blocked = bool(v)
        self.block_history.append(bool(v))


class _Label(_Edit):
    def setStyleSheet(self, s):
        self.style = s


class _Combo:
    def __init__(self, items=None, current=""):
        self._items = list(items or [])
        self._t = current or (self._items[0] if self._items else "")
        self.blocked = False
        self.block_history = []

    def count(self):
        return len(self._items)

    def itemText(self, i):
        return self._items[i]

    def currentText(self):
        return self._t

    def setCurrentText(self, t):
        self._t = str(t)

    def setCurrentIndex(self, i):
        if 0 <= i < len(self._items):
            self._t = self._items[i]

    def blockSignals(self, v):
        self.blocked = bool(v)
        self.block_history.append(bool(v))


class _CheckBox:
    def __init__(self, checked=True):
        self._c = checked

    def isChecked(self):
        return self._c

    def setChecked(self, v):
        self._c = bool(v)


class _UI:
    pass


def _bare_ctrl():
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl._logged = []
    ctrl._append_ui_log = lambda msg: ctrl._logged.append(msg)
    ctrl._saves = 0

    def _sched():
        ctrl._saves += 1

    ctrl._schedule_state_save = _sched
    return ctrl


# ---------------------------------------------------------------------------
# Item 1 - the trained round checkpoint must NOT leak into the workspace
#          "External model file" (imported) slot.
# ---------------------------------------------------------------------------

class TestRoundCheckpointDoesNotLeakIntoImportedSlot:

    def _make(self):
        ctrl = _bare_ctrl()
        ui = ctrl.ui
        ui.imported_model_path = _Edit("")           # user's own slot: empty
        ui.model_source_combo = _Combo(
            ["Auto (Recommended)", "Round model", "External model file",
             "Cellpose pretrained"],
            "Auto (Recommended)",
        )
        ui.auto_update_model_cb = _CheckBox(True)
        ctrl.curation_model_source_mode = "auto"
        ctrl.imported_curation_model_path = ""
        ctrl.active_curation_checkpoints = {"lateral": "", "axial": ""}
        ctrl.selected_view = "XZ"
        ctrl.view_target = "axial"
        ctrl._applied = []
        ctrl._apply_curation_checkpoint_for_view = (
            lambda *a, **k: ctrl._applied.append(k)
        )
        return ctrl

    def test_imported_path_unchanged_after_axial_training(self, tmp_path):
        ctrl = self._make()
        ckpt = tmp_path / "cell_axial_round_00"
        ckpt.write_text("weights")
        ctrl._promote_round_checkpoints_to_curation({"axial": str(ckpt)})

        assert ctrl.ui.imported_model_path.text() == "", (
            "the freshly trained round checkpoint leaked into the External "
            "model file slot - that slot must only hold a user-browsed path"
        )
        assert ctrl.imported_curation_model_path == ""
        # ...but the active axial checkpoint IS promoted.
        assert ctrl.active_curation_checkpoints["axial"] == str(ckpt)

    def test_no_setText_on_imported_path_in_promote_source(self):
        src = inspect.getsource(
            TrainingAnnotationController._promote_round_checkpoints_to_curation
        )
        assert "imported_model_path" not in src, (
            "_promote_round_checkpoints_to_curation must not touch the "
            "imported/External-model-file widget"
        )

    def test_lateral_prediction_never_resolves_the_axial_checkpoint(self, tmp_path):
        """With only an axial checkpoint trained and the imported slot empty,
        resolving the LATERAL model must yield nothing - never the axial one."""
        ctrl = self._make()
        ckpt = tmp_path / "cell_axial_round_00"
        ckpt.write_text("weights")
        ctrl._promote_round_checkpoints_to_curation({"axial": str(ckpt)})

        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        resolved = str(
            ctrl.active_curation_checkpoints.get("lateral", "")
        ) or str(ctrl.imported_curation_model_path or "")
        assert resolved == "", (
            "lateral prediction would run the AXIAL checkpoint"
        )


# ---------------------------------------------------------------------------
# Item 2 - the picker/view keys are PER-RUN, with a workspace-level
#          fallback read for old state files.
# ---------------------------------------------------------------------------

class TestPickerAndViewKeysArePerRun:

    PER_RUN_KEYS = (
        "selected_view",
        "view_target",
        "axial_plane",
        "curation_model_source_mode",
        "imported_curation_model_path",
        "curation_pretrained_model_type",
    )

    def test_keys_are_not_workspace_level(self):
        from napari_epicycle.hitl_training_annotation._migrate_step2_layout import (
            WORKSPACE_LEVEL_KEYS_V2,
        )
        for key in self.PER_RUN_KEYS:
            assert key not in WORKSPACE_LEVEL_KEYS_V2, (
                f"{key} is workspace-level, so it bleeds across runs"
            )

    def test_wrap_puts_them_in_the_run(self):
        from napari_epicycle.hitl_training_annotation._migrate_step2_layout import (
            wrap_state_schema_v2_to_v3,
        )
        v2 = {
            "schema_version": 2,
            "selected_view": "XZ",
            "imported_curation_model_path": "/tmp/mine.model",
            "curation_model_source_mode": "imported",
        }
        v3 = wrap_state_schema_v2_to_v3(v2, active_run_id="cell")
        run = v3["runs"]["cell"]
        assert run["selected_view"] == "XZ"
        assert run["imported_curation_model_path"] == "/tmp/mine.model"
        assert "imported_curation_model_path" not in v3

    def test_two_runs_keep_their_own_values(self, tmp_path):
        from napari_epicycle.hitl_training_annotation import _paths as p
        v3 = {
            "schema_version": 3,
            "active_run_id": "cell",
            "runs": {
                "cell": {
                    "selected_view": "XZ",
                    "imported_curation_model_path": "/models/cell.model",
                    "curation_model_source_mode": "imported",
                },
                "nuc": {
                    "selected_view": "XY",
                    "imported_curation_model_path": "/models/nuc.model",
                    "curation_model_source_mode": "auto",
                },
            },
        }
        sp = p.state_json_path(tmp_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(v3), encoding="utf-8")

        flat_cell = p.load_step2_state_flat(tmp_path, active_run_id="cell")
        flat_nuc = p.load_step2_state_flat(tmp_path, active_run_id="nuc")
        assert flat_cell["imported_curation_model_path"] == "/models/cell.model"
        assert flat_nuc["imported_curation_model_path"] == "/models/nuc.model"
        assert flat_cell["selected_view"] == "XZ"
        assert flat_nuc["selected_view"] == "XY"

    def test_old_format_file_still_loads(self, tmp_path):
        """Backward-compatible read: a pre-change file has the keys ONLY at
        workspace level; they must still reach the flat state."""
        from napari_epicycle.hitl_training_annotation import _paths as p
        v3_old = {
            "schema_version": 3,
            "active_run_id": "cell",
            "selected_view": "XZ",
            "imported_curation_model_path": "/models/legacy.model",
            "curation_model_source_mode": "imported",
            "curation_pretrained_model_type": "nuclei",
            "axial_plane": "YZ",
            "view_target": "axial",
            "runs": {"cell": {"current_round": 3}},
        }
        sp = p.state_json_path(tmp_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(v3_old), encoding="utf-8")

        flat = p.load_step2_state_flat(tmp_path, active_run_id="cell")
        assert flat["imported_curation_model_path"] == "/models/legacy.model"
        assert flat["selected_view"] == "XZ"
        assert flat["curation_pretrained_model_type"] == "nuclei"
        assert flat["axial_plane"] == "YZ"
        assert flat["view_target"] == "axial"
        assert flat["current_round"] == 3

    def test_per_run_value_wins_over_legacy_workspace_level(self, tmp_path):
        from napari_epicycle.hitl_training_annotation import _paths as p
        v3 = {
            "schema_version": 3,
            "active_run_id": "cell",
            "imported_curation_model_path": "/models/legacy.model",
            "runs": {"cell": {"imported_curation_model_path": "/models/new.model"}},
        }
        sp = p.state_json_path(tmp_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(v3), encoding="utf-8")
        flat = p.load_step2_state_flat(tmp_path, active_run_id="cell")
        assert flat["imported_curation_model_path"] == "/models/new.model"

    def test_save_carries_legacy_toplevel_forward_for_unsaved_runs(self):
        """The save path must not silently drop the legacy workspace-level
        values - runs that have not been re-saved still fall back to them."""
        src = inspect.getsource(TrainingAnnotationController._save_internal_state)
        assert "_LEGACY_WORKSPACE_FALLBACK_KEYS" in src, (
            "_save_internal_state must carry the legacy workspace-level "
            "picker/view keys forward so never-re-saved runs keep reading them"
        )


# ---------------------------------------------------------------------------
# Item 3 - status "Last checkpoint" must be bucket-correct.
# ---------------------------------------------------------------------------

class TestLastCheckpointIsBucketScoped:

    def _make(self, view):
        ctrl = _bare_ctrl()
        ctrl.selected_view = view
        ctrl.view_target = "lateral" if view == "XY" else "axial"
        ctrl.current_round_by_view = {"lateral": 2, "axial": 1}
        ctrl.current_round = 2
        ctrl.model_checkpoint_paths = {
            "lateral_0": "/ws/cell_lateral_round_00",
            "lateral_1": "/ws/cell_lateral_round_01",
            "axial_0": "/ws/cell_axial_round_00",
        }
        return ctrl

    def test_axial_view_does_not_show_a_lateral_checkpoint(self):
        ctrl = self._make("XZ")
        assert ctrl._last_checkpoint_text() == "cell_axial_round_00"

    def test_lateral_view_shows_the_latest_lateral_checkpoint(self):
        ctrl = self._make("XY")
        assert ctrl._last_checkpoint_text() == "cell_lateral_round_01"

    def test_no_checkpoint_for_the_bucket_reads_None(self):
        ctrl = self._make("XZ")
        ctrl.model_checkpoint_paths = {"lateral_0": "/ws/cell_lateral_round_00"}
        assert ctrl._last_checkpoint_text() == "None"

    def test_empty_reads_None(self):
        ctrl = self._make("XY")
        ctrl.model_checkpoint_paths = {}
        assert ctrl._last_checkpoint_text() == "None"

    def test_exact_round_key_wins(self):
        ctrl = self._make("XY")
        ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
        assert ctrl._last_checkpoint_text() == "cell_lateral_round_00"


# ---------------------------------------------------------------------------
# Item 4 - workspace open restores the last patch, like a run switch does.
# ---------------------------------------------------------------------------

class TestWorkspaceOpenRestoresLastPatch:

    def test_browse_workdir_calls_the_shared_auto_reload_helper(self):
        src = inspect.getsource(TrainingAnnotationController._browse_workdir)
        assert "_auto_reload_saved_patch" in src, (
            "opening a workspace must restore the saved patch, the same way "
            "a run switch does"
        )

    def test_switch_active_run_uses_the_same_helper(self):
        src = inspect.getsource(TrainingAnnotationController._switch_active_run)
        assert "_auto_reload_saved_patch" in src, (
            "the auto-reload block must be shared, not duplicated"
        )

    def _reload_ctrl(self, tmp_path, patch, files=()):
        ctrl = _bare_ctrl()
        stack = tmp_path / "extractions"
        stack.mkdir(exist_ok=True)
        for f in files:
            (stack / f).write_bytes(b"x")
        ctrl.loaded_stack_path = str(stack)
        ctrl.loaded_patch_name = patch
        ctrl._selected = []
        ctrl._loads = []
        ctrl._select_patch_item = lambda n: (ctrl._selected.append(n) or True)
        ctrl._load_patch = lambda: ctrl._loads.append(ctrl.loaded_patch_name)
        return ctrl

    def test_helper_reloads_when_the_patch_file_is_present(self, tmp_path):
        ctrl = self._reload_ctrl(tmp_path, "patch_01", ["patch_01.tiff"])
        ctrl._auto_reload_saved_patch()
        assert ctrl._selected == ["patch_01"]
        assert ctrl._loads == ["patch_01"]

    def test_helper_accepts_a_tif_extension_too(self, tmp_path):
        ctrl = self._reload_ctrl(tmp_path, "patch_02", ["patch_02.tif"])
        ctrl._auto_reload_saved_patch()
        assert ctrl._loads == ["patch_02"]

    def test_helper_logs_and_skips_when_the_patch_row_is_missing(self, tmp_path):
        ctrl = self._reload_ctrl(tmp_path, "patch_03", ["patch_03.tiff"])
        ctrl._select_patch_item = lambda n: False
        ctrl._auto_reload_saved_patch()
        assert ctrl._loads == []
        assert any("not in the Patches list" in m for m in ctrl._logged)

    def test_helper_is_a_noop_when_the_stack_dir_is_gone(self, tmp_path):
        ctrl = self._reload_ctrl(tmp_path, "gone")
        ctrl.loaded_stack_path = str(tmp_path / "vanished")
        ctrl._auto_reload_saved_patch()
        assert ctrl._loads == []
        assert any("not found at" in m for m in ctrl._logged)

    def test_helper_is_a_noop_with_no_saved_patch(self, tmp_path):
        ctrl = self._reload_ctrl(tmp_path, "")
        ctrl.loaded_stack_path = ""
        ctrl._auto_reload_saved_patch()
        assert ctrl._loads == []


# ---------------------------------------------------------------------------
# Item 5 - Training mode is per bucket.
# ---------------------------------------------------------------------------

class TestTrainingModeIsPerBucket:

    def _make(self):
        ctrl = _bare_ctrl()
        ctrl.ui.training_mode_combo = _Combo(
            ["Cumulative - reuse prior model (default)",
             "From Scratch - ignore prior rounds"],
            "Cumulative - reuse prior model (default)",
        )
        ctrl.selected_view = "XZ"
        ctrl.view_target = "axial"
        ctrl.training_mode = "cumulative"
        ctrl.training_mode_by_bucket = {"lateral": "cumulative", "axial": "cumulative"}
        ctrl._refresh_init_hint = lambda: None
        return ctrl

    def test_change_records_only_the_active_bucket(self):
        ctrl = self._make()
        ctrl._on_training_mode_changed("From Scratch - ignore prior rounds")
        assert ctrl.training_mode_by_bucket["axial"] == "from_scratch"
        assert ctrl.training_mode_by_bucket["lateral"] == "cumulative"
        assert ctrl.training_mode == "from_scratch"  # legacy alias = active

    def test_switching_view_reapplies_the_other_buckets_mode(self):
        ctrl = self._make()
        ctrl._on_training_mode_changed("From Scratch - ignore prior rounds")
        # switch to lateral
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl._apply_training_mode_for_active_bucket()
        assert ctrl.training_mode == "cumulative"
        assert "cumulative" in ctrl.ui.training_mode_combo.currentText().lower()
        # back to axial
        ctrl.selected_view = "XZ"
        ctrl.view_target = "axial"
        ctrl._apply_training_mode_for_active_bucket()
        assert ctrl.training_mode == "from_scratch"
        assert "scratch" in ctrl.ui.training_mode_combo.currentText().lower()

    def test_reapply_blocks_signals_so_it_does_not_persist_what_it_shows(self):
        ctrl = self._make()
        ctrl._on_training_mode_changed("From Scratch - ignore prior rounds")
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        combo = ctrl.ui.training_mode_combo
        combo.block_history.clear()
        ctrl._apply_training_mode_for_active_bucket()
        assert combo.block_history[:1] == [True]
        assert combo.blocked is False, "signals must be un-blocked afterwards"
        assert ctrl.training_mode_by_bucket["axial"] == "from_scratch", (
            "the refresh must not overwrite the OTHER bucket's stored mode"
        )

    def test_persisted_and_restored(self):
        save_src = inspect.getsource(TrainingAnnotationController._save_internal_state)
        assert '"training_mode_by_bucket"' in save_src
        assert '"training_mode"' in save_src, (
            "keep writing the legacy single-valued key for compatibility"
        )
        load_src = inspect.getsource(TrainingAnnotationController._load_internal_state)
        assert "training_mode_by_bucket" in load_src

    def test_old_state_without_the_new_key_falls_back(self):
        ctrl = _bare_ctrl()
        got = ctrl._coerce_training_mode_by_bucket(None, fallback="from_scratch")
        assert got == {"lateral": "from_scratch", "axial": "from_scratch"}

    def test_partial_dict_is_filled_in(self):
        ctrl = _bare_ctrl()
        got = ctrl._coerce_training_mode_by_bucket(
            {"axial": "from_scratch"}, fallback="cumulative"
        )
        assert got == {"lateral": "cumulative", "axial": "from_scratch"}

    def test_junk_values_are_coerced(self):
        ctrl = _bare_ctrl()
        got = ctrl._coerce_training_mode_by_bucket(
            {"axial": "banana", "lateral": 17}, fallback="cumulative"
        )
        assert got == {"lateral": "cumulative", "axial": "cumulative"}


# ---------------------------------------------------------------------------
# Item 6 - the view-change refresh covers the whole Model/Training block.
# ---------------------------------------------------------------------------

class TestViewChangeRefreshesModelAndTrainingWidgets:

    def test_refresh_delegates_to_the_shared_display_only_helpers(self):
        src = inspect.getsource(
            TrainingAnnotationController._update_training_ui_for_view
        )
        for needle in (
            "_refresh_model_widgets_from_state",
            "_apply_training_mode_for_active_bucket",
            "status_last_checkpoint_value",
        ):
            assert needle in src, f"view change must refresh {needle}"

    def test_model_widget_refresh_is_display_only(self):
        src = inspect.getsource(
            TrainingAnnotationController._refresh_model_widgets_from_state
        )
        for needle in ("model_source_combo", "imported_model_path",
                       "curation_pretrained_combo"):
            assert needle in src
        assert "blockSignals" in src, (
            "the refresh must block signals so displaying a value does not "
            "persist it (restore-clobber guard)"
        )
        assert "_schedule_state_save" not in src

    def test_refresh_does_not_schedule_a_save(self):
        ctrl = _bare_ctrl()
        ui = ctrl.ui
        ui.model_source_combo = _Combo(
            ["Auto (Recommended)", "Round model", "External model file"],
            "Round model",
        )
        ui.imported_model_path = _Edit("/models/mine.model")
        ui.curation_pretrained_combo = _Combo(["cyto3", "nuclei"], "nuclei")
        ui.training_mode_combo = _Combo(
            ["Cumulative - reuse prior model (default)",
             "From Scratch - ignore prior rounds"],
            "Cumulative - reuse prior model (default)",
        )
        ui.starting_checkpoint_path = _Edit("")
        ui.status_last_checkpoint_value = _Label("")
        ui.loss_plot = None
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl.current_round_by_view = {"lateral": 1, "axial": 0}
        ctrl.current_round = 1
        ctrl.model_checkpoint_paths = {"lateral_0": "/ws/cell_lateral_round_00"}
        ctrl.curation_model_source_mode = "auto"
        ctrl.imported_curation_model_path = "/models/mine.model"
        ctrl.curation_pretrained_model_type = "cyto3"
        ctrl.training_mode = "cumulative"
        ctrl.training_mode_by_bucket = {"lateral": "cumulative", "axial": "from_scratch"}
        ctrl._init_transparency_loading = False
        ctrl._resolve_prior_model_checkpoint = lambda *a, **k: {}
        ctrl._refresh_init_hint = lambda: None

        ctrl._update_training_ui_for_view()

        assert ctrl._saves == 0, "a display-only refresh must not persist"
        # widgets now agree with state
        assert ctrl.ui.model_source_combo.currentText().startswith("Auto")
        assert ctrl.ui.curation_pretrained_combo.currentText() == "cyto3"
        assert ctrl.ui.status_last_checkpoint_value.text() == "cell_lateral_round_00"
        # and the OTHER bucket's stored mode is untouched
        assert ctrl.training_mode_by_bucket["axial"] == "from_scratch"


# ---------------------------------------------------------------------------
# Item 7 - no automatic root training_config.json.
# ---------------------------------------------------------------------------

class TestNoAutomaticRootTrainingConfig:

    def test_no_caller_or_definition_remains_anywhere(self):
        import napari_epicycle.hitl_training_annotation.controller.training_annotation_controller as m
        text = Path(m.__file__).read_text(encoding="utf-8")
        assert "_auto_save_training_config" not in text, (
            "training must not drop a training_config.json at the workspace root"
        )

    def test_manual_save_defaults_into_the_active_run_dir(self):
        src = inspect.getsource(
            TrainingAnnotationController._save_and_set_training_config
        )
        assert "_training_config_dialog_dir" in src

    def test_manual_load_defaults_into_the_active_run_dir(self):
        src = inspect.getsource(
            TrainingAnnotationController._load_training_config
        )
        assert "_training_config_dialog_dir" in src

    def test_dialog_dir_is_the_run_dir_when_it_exists(self, tmp_path):
        from napari_epicycle.hitl_training_annotation import _paths as p
        ctrl = _bare_ctrl()
        ctrl.working_dir = str(tmp_path)
        ctrl.active_run_id = "cell"
        run = p.run_root(tmp_path, "cell")
        run.mkdir(parents=True)
        got = Path(ctrl._training_config_dialog_dir())
        assert got == run

    def test_dialog_dir_falls_back_to_the_workspace(self, tmp_path):
        ctrl = _bare_ctrl()
        ctrl.working_dir = str(tmp_path)
        ctrl.active_run_id = ""
        assert Path(ctrl._training_config_dialog_dir()) == tmp_path

    def test_dialog_dir_is_empty_without_a_workspace(self):
        ctrl = _bare_ctrl()
        ctrl.working_dir = ""
        ctrl.active_run_id = "cell"
        assert ctrl._training_config_dialog_dir() == ""


# ---------------------------------------------------------------------------
# Item 8 - the Benchmarking section is hidden for v1 (but still built).
# ---------------------------------------------------------------------------

class TestBenchmarkingHidden:

    def test_control_rail_hides_the_benchmarking_panel(self):
        import inspect as _i
        from napari_epicycle.hitl_training_annotation.ui.main_widget import (
            TrainingAnnotationHITL2D,
        )
        src = _i.getsource(TrainingAnnotationHITL2D._create_control_rail)
        assert "_benchmarking_panel" in src
        assert "setVisible(False)" in src or "setItemVisible" in src, (
            "the Benchmarking section must be hidden for v1"
        )

    def test_the_panel_is_still_built(self):
        import inspect as _i
        from napari_epicycle.hitl_training_annotation.ui.main_widget import (
            TrainingAnnotationHITL2D,
        )
        src = _i.getsource(TrainingAnnotationHITL2D._create_control_rail)
        assert "_create_benchmarking_panel()" in src, (
            "keep the panel BUILT so its child widgets are never orphaned"
        )


# ---------------------------------------------------------------------------
# Item 9 - plain-language copy.
# ---------------------------------------------------------------------------

class TestPlainLanguageCopy:

    def _ui_source(self):
        import napari_epicycle.hitl_training_annotation.ui.main_widget as m
        return Path(m.__file__).read_text(encoding="utf-8")

    def test_external_model_file_replaces_imported_model(self):
        text = self._ui_source()
        assert '"External model file"' in text
        assert '"Imported model"' not in text

    def test_model_source_key_still_maps_the_new_label(self):
        ctrl = _bare_ctrl()
        assert ctrl._model_source_key("External model file") == "imported"
        # legacy label still understood (old persisted UI text / tests)
        assert ctrl._model_source_key("Imported model") == "imported"

    def test_amber_model_name_row_says_what_it_is(self):
        # brick T: "Next round will train:" became "Writes:" (Setup)
        from napari_epicycle.hitl_training_annotation.ui import _step2_text as T
        text = self._ui_source()
        assert T.WRITES_LABEL == "Writes:"
        assert "Next round will train:" not in text
        assert 'addRow("Model name:"' not in text

    def test_active_model_row_is_labelled(self):
        text = self._ui_source()
        assert "Active model:" in text

    def test_from_scratch_label_no_longer_claims_no_checkpoint(self):
        text = self._ui_source()
        assert "From Scratch - no prior checkpoint" not in text
        assert "From Scratch — no prior checkpoint" not in text

    def test_mode_key_derivation_is_unchanged_by_the_rename(self):
        ctrl = _bare_ctrl()
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl.training_mode = "cumulative"
        ctrl.training_mode_by_bucket = {"lateral": "cumulative", "axial": "cumulative"}
        ctrl.ui.training_mode_combo = _Combo(
            ["Cumulative - reuse prior model (default)",
             "From Scratch - ignore prior rounds"],
        )
        ctrl._refresh_init_hint = lambda: None
        ctrl._on_training_mode_changed("From Scratch - ignore prior rounds")
        assert ctrl.training_mode == "from_scratch"
        ctrl._on_training_mode_changed("Cumulative - reuse prior model (default)")
        assert ctrl.training_mode == "cumulative"


# ---------------------------------------------------------------------------
# Item 10 - the run picker counts TRAINED rounds.
# ---------------------------------------------------------------------------

class TestRunPickerCountsTrainedRounds:

    def _mk(self, tmp_path, slug, bucket, rounds_with_model, rounds_without):
        from napari_epicycle.hitl_training_annotation import _paths as p
        for i in rounds_with_model:
            p.model_dir(tmp_path, slug, bucket, i).mkdir(parents=True, exist_ok=True)
        for i in rounds_without:
            p.round_dir(tmp_path, slug, bucket, i).mkdir(parents=True, exist_ok=True)

    def test_only_rounds_with_a_model_are_counted(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._run_management import _count_rounds
        self._mk(tmp_path, "cell", "lateral", [0, 1], [2])
        assert _count_rounds(tmp_path, "cell", "lateral") == 2

    def test_zero_when_nothing_is_trained(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._run_management import _count_rounds
        self._mk(tmp_path, "cell", "axial", [], [0])
        assert _count_rounds(tmp_path, "cell", "axial") == 0

    def test_legacy_split_layout_still_counted(self, tmp_path):
        from napari_epicycle.hitl_training_annotation import _paths as p
        from napari_epicycle.hitl_training_annotation._run_management import _count_rounds
        legacy = p.bucket_rounds_root(tmp_path, "cell", "lateral") / "round_00" / "model"
        legacy.mkdir(parents=True)
        assert _count_rounds(tmp_path, "cell", "lateral") == 1

    def test_missing_bucket_is_zero(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._run_management import _count_rounds
        assert _count_rounds(tmp_path, "cell", "lateral") == 0

    def test_the_same_round_in_both_layouts_counts_once(self, tmp_path):
        from napari_epicycle.hitl_training_annotation import _paths as p
        from napari_epicycle.hitl_training_annotation._run_management import _count_rounds
        p.model_dir(tmp_path, "cell", "lateral", 0).mkdir(parents=True)
        (p.bucket_rounds_root(tmp_path, "cell", "lateral")
         / "round_00" / "model").mkdir(parents=True)
        assert _count_rounds(tmp_path, "cell", "lateral") == 1


class TestLastCheckpointLegacyWorkspaces:
    """Pre-per-bucket workspaces used bare int round keys.  There is no bucket
    to attribute them to, so keep showing them rather than regressing to
    "None"."""

    def test_bare_int_keys_still_display(self):
        ctrl = _bare_ctrl()
        ctrl.selected_view = "XZ"
        ctrl.view_target = "axial"
        ctrl.current_round_by_view = {"lateral": 0, "axial": 1}
        ctrl.current_round = 1
        ctrl.model_checkpoint_paths = {0: "/ws/old_round_00", 1: "/ws/old_round_01"}
        assert ctrl._last_checkpoint_text() == "old_round_01"

    def test_bare_int_fallback_is_off_once_bucket_keys_exist(self):
        ctrl = _bare_ctrl()
        ctrl.selected_view = "XZ"
        ctrl.view_target = "axial"
        ctrl.current_round_by_view = {"lateral": 1, "axial": 0}
        ctrl.current_round = 0
        ctrl.model_checkpoint_paths = {0: "/ws/old_round_00",
                                       "lateral_1": "/ws/cell_lateral_round_01"}
        assert ctrl._last_checkpoint_text() == "None"


class TestPoisonedImportedSlotSelfHeals:
    """Workspaces saved before the item-#1 fix carry the leaked round
    checkpoint in the External-model-file slot.  Loading clears it -- but
    only when it is byte-identical to one of this run's own active round
    checkpoints, which is the exact thing the leak wrote."""

    HEAL = staticmethod(TrainingAnnotationController._is_leaked_round_checkpoint)

    def test_this_runs_own_round_checkpoint_is_a_leak(self):
        assert self.HEAL(
            "/ws/artifacts/step2/runs/cell/axial/round_00/model/cell_axial_round_00",
            {"lateral": "",
             "axial": "/ws/artifacts/step2/runs/cell/axial/round_00/model/cell_axial_round_00"},
        )

    def test_a_copied_or_moved_workspace_still_matches(self):
        """The leaked path was stored Achr(92)OLUTE while active checkpoints are
        stored RELATIVE, so their roots differ after a copy or move -- and
        Windows separators differ from the stored forward slashes."""
        win = ("C:@Users@me@Desktop@Wound@artifacts@step2@runs@cell@axial"
               "@round_00@model@cell_axial_round_00").replace("@", chr(92))
        rel = ("artifacts@step2@runs@cell@axial@round_00@model"
               "@cell_axial_round_00").replace("@", chr(92))
        assert self.HEAL(win, {"axial": rel, "lateral": ""})

    def test_a_genuinely_browsed_path_is_left_alone(self):
        assert not self.HEAL(
            "/downloads/models/somebody_elses.model",
            {"lateral": "artifacts/step2/runs/cell/lateral/round_01/model/x",
             "axial": "artifacts/step2/runs/cell/axial/round_00/model/y"},
        )

    def test_another_runs_model_is_left_alone(self):
        assert not self.HEAL(
            "/ws/artifacts/step2/runs/nuc/lateral/round_00/model/nuc_lateral_round_00",
            {"lateral": "artifacts/step2/runs/cell/lateral/round_01/model/x",
             "axial": ""},
        )

    def test_a_different_round_of_the_same_bucket_is_left_alone(self):
        assert not self.HEAL(
            "/ws/artifacts/step2/runs/cell/axial/round_01/model/cell_axial_round_01",
            {"axial": "artifacts/step2/runs/cell/axial/round_00/model/cell_axial_round_00",
             "lateral": ""},
        )

    def test_empty_and_junk_are_not_leaks(self):
        assert not self.HEAL("", {"lateral": "", "axial": ""})
        assert not self.HEAL(None, {"lateral": "", "axial": ""})
        assert not self.HEAL("/a/b", None)
        assert not self.HEAL("   ", {"lateral": "", "axial": ""})
        assert not self.HEAL("short.model", {"axial": "short.model"})

    def test_the_load_path_uses_it(self):
        src = inspect.getsource(TrainingAnnotationController._load_internal_state)
        assert "_should_heal_imported_slot" in src
        gate = inspect.getsource(
            TrainingAnnotationController._should_heal_imported_slot
        )
        assert "_is_leaked_round_checkpoint" in gate

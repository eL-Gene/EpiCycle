"""Fix pass for the Step 2 "persistence + bucket switching" brick
(2026-09-07, two independent audits).

Reuses the Qt-free fakes from ``test_step2_bucket_switch_persistence``.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)

from .test_step2_bucket_switch_persistence import (
    _bare_ctrl,
    _Combo,
    _Edit,
    _Label,
)


# ---------------------------------------------------------------------------
# Audit fix #1 - the PER-BUCKET training mode must reach the trainer.
# ---------------------------------------------------------------------------

class TestPerBucketModeReachesTheTrainer:
    """A mixed XY+XZ round used to read the flat ``training_mode`` ONCE and
    apply it to every bucket, so "From Scratch on axial + Cumulative on
    lateral" trained BOTH the same way."""

    def _ctrl(self, modes):
        ctrl = _bare_ctrl()
        ctrl.training_mode_by_bucket = dict(modes)
        ctrl.training_mode = modes["lateral"]
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl.current_round_by_view = {"lateral": 2, "axial": 2}
        ctrl.current_round = 2
        ctrl.model_checkpoint_paths = {
            "lateral_1": "/ws/cell_lateral_round_01",
            "axial_1": "/ws/cell_axial_round_01",
        }
        return ctrl

    def test_from_scratch_axial_plus_cumulative_lateral(self):
        ctrl = self._ctrl({"lateral": "cumulative", "axial": "from_scratch"})
        got = ctrl._resolve_prior_checkpoints_for_buckets({"lateral", "axial"})
        assert got == {"spatial": "/ws/cell_lateral_round_01"}, (
            "axial is From Scratch, so it must contribute NO prior "
            "checkpoint, and lateral must still get its own"
        )

    def test_the_reverse(self):
        ctrl = self._ctrl({"lateral": "from_scratch", "axial": "cumulative"})
        got = ctrl._resolve_prior_checkpoints_for_buckets({"lateral", "axial"})
        assert got == {"axial": "/ws/cell_axial_round_01"}

    def test_both_cumulative(self):
        ctrl = self._ctrl({"lateral": "cumulative", "axial": "cumulative"})
        got = ctrl._resolve_prior_checkpoints_for_buckets({"lateral", "axial"})
        assert got == {
            "spatial": "/ws/cell_lateral_round_01",
            "axial": "/ws/cell_axial_round_01",
        }

    def test_both_from_scratch(self):
        ctrl = self._ctrl({"lateral": "from_scratch", "axial": "from_scratch"})
        assert ctrl._resolve_prior_checkpoints_for_buckets(
            {"lateral", "axial"}
        ) == {}

    def test_a_single_bucket_round_is_unaffected(self):
        ctrl = self._ctrl({"lateral": "cumulative", "axial": "from_scratch"})
        assert ctrl._resolve_prior_checkpoints_for_buckets({"lateral"}) == {
            "spatial": "/ws/cell_lateral_round_01"
        }
        assert ctrl._resolve_prior_checkpoints_for_buckets({"axial"}) == {}

    def test_it_logs_the_per_bucket_decision(self):
        ctrl = self._ctrl({"lateral": "cumulative", "axial": "from_scratch"})
        ctrl._resolve_prior_checkpoints_for_buckets({"lateral", "axial"})
        joined = " ".join(ctrl._logged).lower()
        assert "axial" in joined and "scratch" in joined

    def test_the_train_path_uses_the_per_bucket_resolver(self):
        src = inspect.getsource(
            TrainingAnnotationController._train_model
        )
        assert "_resolve_prior_checkpoints_for_buckets" in src

    def test_the_worker_no_longer_gates_on_the_flat_mode(self):
        """prior_checkpoints is already empty for a from-scratch bucket, so
        re-gating on the SCALAR mode would undo the per-bucket decision."""
        src = inspect.getsource(TrainingAnnotationController._run_training_work)
        assert 'if training_mode == "cumulative" else None' not in src
        assert "pretrained = prior_checkpoints.get(view_type)" in src


# ---------------------------------------------------------------------------
# Audit fix #2 - the Active-model readout says "external model file".
# ---------------------------------------------------------------------------

class TestActiveModelReadoutCopy:

    def _ctrl(self, mode, imported=""):
        ctrl = _bare_ctrl()
        ctrl.current_config = {"model_type": "cyto3"}
        ctrl.active_curation_checkpoints = {"lateral": "", "axial": ""}
        ctrl.curation_model_source_mode = mode
        ctrl.imported_curation_model_path = imported
        ctrl.curation_pretrained_model_type = "cyto3"
        ctrl._ensure_curation_checkpoint_state = lambda: None
        ctrl._texts = {}
        ctrl._safe_ui_set_text = (
            lambda attr, val: ctrl._texts.__setitem__(attr, str(val))
        )
        return ctrl

    def test_bare_imported_mode(self):
        ctrl = self._ctrl("imported")
        ctrl._refresh_curation_model_status()
        assert ctrl._texts["active_model_source_value"] == "external model file"

    def test_imported_mode_with_a_path(self):
        ctrl = self._ctrl("imported", "/models/mine.model")
        ctrl._refresh_curation_model_status()
        assert ctrl._texts["active_model_source_value"] == (
            "external model file (/models/mine.model)"
        )

    def test_no_imported_wording_survives_anywhere_in_the_readout(self):
        for mode in ("auto", "round", "imported", "pretrained", "none"):
            ctrl = self._ctrl(mode, "/models/mine.model")
            ctrl._refresh_curation_model_status()
            summary = ctrl._texts["active_model_summary_value"]
            assert "imported model" not in summary


# ---------------------------------------------------------------------------
# Audit fix #3 - the manual "Save Config..." is still ATOMIC (behavioural).
# ---------------------------------------------------------------------------

class TestManualSaveConfigIsAtomic:

    def test_no_automatic_root_training_config_remains(self):
        import napari_epicycle.hitl_training_annotation.controller.training_annotation_controller as m
        text = Path(m.__file__).read_text(encoding="utf-8")
        assert "_auto_save_training_config" not in text, (
            "training must not drop a training_config.json at the "
            "workspace root"
        )

    def test_a_failed_replace_leaves_the_previous_file_intact(
        self, tmp_path, monkeypatch
    ):
        import napari_epicycle.hitl_training_annotation.controller.training_annotation_controller as m

        target = tmp_path / "training_config.json"
        target.write_text(json.dumps({"epochs": 10}), encoding="utf-8")

        ctrl = _bare_ctrl()
        ctrl.working_dir = str(tmp_path)
        ctrl.active_run_id = ""
        ctrl.current_config = {}
        ctrl._collect_training_config = lambda: {"epochs": 999}
        ctrl._sync_inference_controls_from_config = lambda *a, **k: None
        ctrl._apply_model_source_selection = lambda *a, **k: None
        ctrl._save_internal_state = lambda *a, **k: None

        monkeypatch.setattr(
            m.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a, **k: (str(target), "")),
        )
        monkeypatch.setattr(
            os, "replace",
            lambda *a, **k: (_ for _ in ()).throw(OSError("crash")),
        )
        with pytest.raises(OSError):
            ctrl._save_and_set_training_config()

        assert json.loads(target.read_text(encoding="utf-8")) == {"epochs": 10}, (
            "a crash mid-write must leave the PREVIOUS config byte-intact"
        )

    def test_a_successful_save_writes_the_new_config(self, tmp_path, monkeypatch):
        import napari_epicycle.hitl_training_annotation.controller.training_annotation_controller as m

        target = tmp_path / "training_config.json"
        target.write_text(json.dumps({"epochs": 10}), encoding="utf-8")

        ctrl = _bare_ctrl()
        ctrl.working_dir = str(tmp_path)
        ctrl.active_run_id = ""
        ctrl.current_config = {}
        ctrl._collect_training_config = lambda: {"epochs": 999}
        ctrl._sync_inference_controls_from_config = lambda *a, **k: None
        ctrl._apply_model_source_selection = lambda *a, **k: None
        ctrl._save_internal_state = lambda *a, **k: None
        monkeypatch.setattr(
            m.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a, **k: (str(target), "")),
        )
        ctrl._save_and_set_training_config()
        assert json.loads(target.read_text(encoding="utf-8")) == {"epochs": 999}


# ---------------------------------------------------------------------------
# Audit fix #4 - lateral prediction goes through the REAL resolver.
# ---------------------------------------------------------------------------

class TestLateralPredictionUsesTheRealResolver:

    def _ctrl(self, tmp_path):
        ctrl = _bare_ctrl()
        ctrl.working_dir = str(tmp_path)
        ctrl.ui.imported_model_path = _Edit("")
        ctrl.ui.model_source_combo = _Combo(
            ["Auto (Recommended)", "Round model", "External model file"],
            "Auto (Recommended)",
        )
        ctrl.ui.auto_update_model_cb = None
        ctrl.curation_model_source_mode = "auto"
        ctrl.imported_curation_model_path = ""
        ctrl.curation_pretrained_model_type = "cyto3"
        ctrl.current_config = {"model_type": "cyto3"}
        ctrl.active_curation_checkpoints = {"lateral": "", "axial": ""}
        ctrl._ensure_curation_checkpoint_state = lambda: None
        ctrl._apply_curation_checkpoint_for_view = lambda *a, **k: None
        return ctrl

    def test_after_axial_training_lateral_resolves_no_checkpoint(self, tmp_path):
        """The real predict-time resolver — `_resolve_curation_checkpoint` —
        must not hand lateral the axial model."""
        ctrl = self._ctrl(tmp_path)
        ckpt = tmp_path / "cell_axial_round_00"
        ckpt.write_text("weights")
        ctrl._promote_round_checkpoints_to_curation({"axial": str(ckpt)})

        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        resolved = ctrl._resolve_model_source_for_view(view_name="XY")
        assert resolved["checkpoint_path"] == ""

    def test_the_axial_view_does_resolve_it(self, tmp_path):
        ctrl = self._ctrl(tmp_path)
        ckpt = tmp_path / "cell_axial_round_00"
        ckpt.write_text("weights")
        ctrl._promote_round_checkpoints_to_curation({"axial": str(ckpt)})

        ctrl.selected_view = "XZ"
        ctrl.view_target = "axial"
        resolved = ctrl._resolve_model_source_for_view(view_name="XZ")
        assert resolved["checkpoint_path"] == str(ckpt)

    def test_the_external_slot_is_still_empty_after_training(self, tmp_path):
        ctrl = self._ctrl(tmp_path)
        ckpt = tmp_path / "cell_axial_round_00"
        ckpt.write_text("weights")
        ctrl._promote_round_checkpoints_to_curation({"axial": str(ckpt)})
        assert ctrl.ui.imported_model_path.text() == ""
        assert ctrl.imported_curation_model_path == ""


# ---------------------------------------------------------------------------
# Audit fix #5 - Load Config must not import ANOTHER run's checkpoint.
# ---------------------------------------------------------------------------

class TestLoadConfigRejectsAnotherRunsCheckpoint:

    def _ctrl(self, tmp_path):
        ctrl = _bare_ctrl()
        ctrl.working_dir = str(tmp_path)
        ctrl.active_run_id = "cell"
        return ctrl

    def _run_path(self, tmp_path, run):
        return str(
            tmp_path / "artifacts" / "step2" / "runs" / run / "lateral"
            / "round_00" / "model" / f"{run}_lateral_round_00"
        )

    def test_another_runs_checkpoint_is_detected(self, tmp_path):
        ctrl = self._ctrl(tmp_path)
        assert ctrl._foreign_run_checkpoint(self._run_path(tmp_path, "nuc")) == "nuc"

    def test_this_runs_checkpoint_is_fine(self, tmp_path):
        ctrl = self._ctrl(tmp_path)
        assert ctrl._foreign_run_checkpoint(self._run_path(tmp_path, "cell")) is None

    def test_an_outside_path_is_fine(self, tmp_path):
        ctrl = self._ctrl(tmp_path)
        assert ctrl._foreign_run_checkpoint("/downloads/some.model") is None
        assert ctrl._foreign_run_checkpoint("") is None
        assert ctrl._foreign_run_checkpoint(None) is None

    def test_load_config_drops_the_foreign_path_and_warns(self, tmp_path):
        ctrl = self._ctrl(tmp_path)
        foreign = self._run_path(tmp_path, "nuc")
        cleaned = ctrl._sanitize_loaded_training_config({
            "epochs": 42,
            "pretrained_model_path": foreign,
            "starting_checkpoint": foreign,
        })
        assert cleaned["epochs"] == 42
        assert cleaned.get("pretrained_model_path", "") == ""
        assert cleaned.get("starting_checkpoint", "") == ""
        assert any("nuc" in m for m in ctrl._logged)

    def test_a_config_from_this_run_is_untouched(self, tmp_path):
        ctrl = self._ctrl(tmp_path)
        mine = self._run_path(tmp_path, "cell")
        cfg = {"epochs": 7, "pretrained_model_path": mine,
               "starting_checkpoint": mine}
        assert ctrl._sanitize_loaded_training_config(cfg) == cfg
        assert ctrl._logged == []

    def test_a_config_with_no_paths_is_untouched(self, tmp_path):
        ctrl = self._ctrl(tmp_path)
        assert ctrl._sanitize_loaded_training_config({"epochs": 3}) == {"epochs": 3}

    def test_load_config_routes_through_the_sanitizer(self):
        src = inspect.getsource(TrainingAnnotationController._load_training_config)
        assert "_sanitize_loaded_training_config" in src


# ---------------------------------------------------------------------------
# Audit fix #6 - display-only refreshes RESTORE the prior block state.
# ---------------------------------------------------------------------------

class _AlreadyBlockedCombo(_Combo):
    """A widget whose signals were ALREADY blocked by an outer guard."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.blocked = True

    def signalsBlocked(self):
        return self.blocked


class _AlreadyBlockedEdit(_Edit):

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.blocked = True

    def signalsBlocked(self):
        return self.blocked


class TestBlockSignalsIsRestoredNotForced:

    def test_training_mode_refresh_restores_a_prior_block(self):
        ctrl = _bare_ctrl()
        combo = _AlreadyBlockedCombo(
            ["Cumulative - reuse prior model (default)",
             "From Scratch - ignore prior rounds"],
        )
        ctrl.ui.training_mode_combo = combo
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl.training_mode = "cumulative"
        ctrl.training_mode_by_bucket = {"lateral": "from_scratch",
                                        "axial": "cumulative"}
        ctrl._apply_training_mode_for_active_bucket()
        assert combo.blocked is True, (
            "an outer guard had signals blocked; the refresh must restore "
            "that state, not force-unblock the widget mid-guard"
        )

    def test_model_widget_refresh_restores_a_prior_block(self):
        ctrl = _bare_ctrl()
        ui = ctrl.ui
        ui.model_source_combo = _AlreadyBlockedCombo(
            ["Auto (Recommended)", "Round model", "External model file"],
        )
        ui.imported_model_path = _AlreadyBlockedEdit("")
        ui.curation_pretrained_combo = _AlreadyBlockedCombo(["cyto3", "nuclei"])
        ctrl.curation_model_source_mode = "auto"
        ctrl.imported_curation_model_path = ""
        ctrl.curation_pretrained_model_type = "cyto3"
        ctrl._refresh_model_widgets_from_state()
        assert ui.model_source_combo.blocked is True
        assert ui.imported_model_path.blocked is True
        assert ui.curation_pretrained_combo.blocked is True

    def test_an_unblocked_widget_is_left_unblocked(self):
        ctrl = _bare_ctrl()
        combo = _Combo(["Cumulative - x", "From Scratch - y"])
        ctrl.ui.training_mode_combo = combo
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl.training_mode = "cumulative"
        ctrl.training_mode_by_bucket = {"lateral": "cumulative",
                                        "axial": "cumulative"}
        ctrl._apply_training_mode_for_active_bucket()
        assert combo.blocked is False

    def test_the_model_refresh_still_applies_the_values(self):
        ctrl = _bare_ctrl()
        ui = ctrl.ui
        ui.model_source_combo = _Combo(
            ["Auto (Recommended)", "Round model", "External model file"],
            "Round model",
        )
        ui.imported_model_path = _Edit("stale")
        ui.curation_pretrained_combo = _Combo(["cyto3", "nuclei"], "nuclei")
        ctrl.curation_model_source_mode = "imported"
        ctrl.imported_curation_model_path = "/models/mine.model"
        ctrl.curation_pretrained_model_type = "cyto3"
        ctrl._refresh_model_widgets_from_state()
        assert ui.model_source_combo.currentText() == "External model file"
        assert ui.imported_model_path.text() == "/models/mine.model"
        assert ui.curation_pretrained_combo.currentText() == "cyto3"


# ---------------------------------------------------------------------------
# Audit fix #7 - a deliberately browsed EXTERNAL file survives the self-heal.
# ---------------------------------------------------------------------------

class TestSelfHealSkipsDeliberateExternalPicks:

    def _ctrl(self, tmp_path, mode, path):
        ctrl = _bare_ctrl()
        ctrl.working_dir = str(tmp_path / "ws")
        ctrl.curation_model_source_mode = mode
        ctrl.imported_curation_model_path = path
        ctrl.active_curation_checkpoints = {
            "lateral": ("artifacts/step2/runs/cell/lateral/round_01/model/"
                        "cell_lateral_round_01"),
            "axial": ("artifacts/step2/runs/cell/axial/round_00/model/"
                      "cell_axial_round_00"),
        }
        return ctrl

    def _outside(self, tmp_path):
        return str(
            tmp_path / "other_ws" / "artifacts" / "step2" / "runs" / "cell"
            / "axial" / "round_00" / "model" / "cell_axial_round_00"
        )

    def _inside(self, tmp_path):
        return str(
            tmp_path / "ws" / "artifacts" / "step2" / "runs" / "cell"
            / "axial" / "round_00" / "model" / "cell_axial_round_00"
        )

    def test_explicit_external_pick_outside_the_workspace_is_kept(self, tmp_path):
        """The user deliberately chose External model file and browsed to a
        file OUTSIDE this workspace -- even a path that happens to end in the
        same run/bucket/round/model segments must not be wiped."""
        ctrl = self._ctrl(tmp_path, "imported", self._outside(tmp_path))
        assert ctrl._should_heal_imported_slot() is False

    def test_the_leak_inside_this_workspace_is_still_healed(self, tmp_path):
        ctrl = self._ctrl(tmp_path, "imported", self._inside(tmp_path))
        assert ctrl._should_heal_imported_slot() is True

    def test_auto_mode_is_healed_wherever_the_file_sits(self, tmp_path):
        """The leak wrote the path while the mode was auto/round -- the user
        never picked External, so there is nothing deliberate to protect."""
        ctrl = self._ctrl(tmp_path, "auto", self._outside(tmp_path))
        assert ctrl._should_heal_imported_slot() is True

    def test_round_mode_is_healed_too(self, tmp_path):
        ctrl = self._ctrl(tmp_path, "round", self._outside(tmp_path))
        assert ctrl._should_heal_imported_slot() is True

    def test_a_genuinely_unrelated_path_is_never_healed(self, tmp_path):
        for mode in ("imported", "auto"):
            ctrl = self._ctrl(tmp_path, mode, "/downloads/mine.model")
            assert ctrl._should_heal_imported_slot() is False

    def test_an_empty_slot_is_never_healed(self, tmp_path):
        ctrl = self._ctrl(tmp_path, "imported", "")
        assert ctrl._should_heal_imported_slot() is False

    def test_no_workspace_still_heals_a_matching_leak(self, tmp_path):
        ctrl = self._ctrl(tmp_path, "imported", self._inside(tmp_path))
        ctrl.working_dir = ""
        assert ctrl._should_heal_imported_slot() is True

    def test_the_load_path_uses_the_gate(self):
        src = inspect.getsource(TrainingAnnotationController._load_internal_state)
        assert "_should_heal_imported_slot" in src

"""Step 3's refinement config moves under `artifacts/` — Brick B4.

`refinement_config.json` used to sit loose in the workspace root.  It now
lives at `artifacts/step3/refinement/refinement_config.json`, and the loader
falls back to the old root file FOREVER, so a project that already has one
still opens with its settings intact.  The root copy is never moved or
deleted; the next save simply goes to the new place.

It stays WORKSPACE-GLOBAL, not run-scoped — only the location changed.
"""
from __future__ import annotations

import json

from napari_epicycle.fusion_reconstruction._paths import (
    legacy_step3_refinement_config_path,
    step3_refinement_config_path,
)
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)

CONFIG = {"n_epochs": 42, "learning_rate": 0.001}


class _Spin:
    def __init__(self):
        self.value_set = None

    def setValue(self, v):
        self.value_set = v

    def value(self):
        return self.value_set


class _UI:
    """Only the widgets the loader touches for the keys used here."""

    def __init__(self):
        self.refine_epochs_sb = _Spin()
        self.refine_lr_sb = _Spin()


def _controller(ws):
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.working_dir = str(ws)
    ctrl.ui = _UI()
    ctrl.logs = []
    ctrl._append_log = lambda msg: ctrl.logs.append(msg)
    ctrl._build_training_config = lambda: dict(CONFIG)
    return ctrl


class TestThePathHelpers:
    def test_new_location_is_under_artifacts(self, tmp_path):
        p = step3_refinement_config_path(tmp_path)
        assert p.parent == tmp_path / "artifacts" / "step3" / "refinement"
        assert p.name == "refinement_config.json"

    def test_it_is_workspace_global_not_run_scoped(self, tmp_path):
        assert "runs" not in step3_refinement_config_path(tmp_path).parts

    def test_the_legacy_helper_names_the_root_file(self, tmp_path):
        assert legacy_step3_refinement_config_path(tmp_path) == (
            tmp_path / "refinement_config.json")


class TestSave:
    def test_the_write_goes_to_the_new_place_and_makes_the_folder(self, tmp_path):
        ctrl = _controller(tmp_path)
        ctrl._save_refinement_config()
        target = step3_refinement_config_path(tmp_path)
        assert target.exists()
        assert json.loads(target.read_text(encoding="utf-8")) == CONFIG

    def test_nothing_lands_in_the_workspace_root(self, tmp_path):
        ctrl = _controller(tmp_path)
        ctrl._save_refinement_config()
        assert not (tmp_path / "refinement_config.json").exists()

    def test_a_pre_existing_root_copy_is_left_alone(self, tmp_path):
        (tmp_path / "refinement_config.json").write_text(
            json.dumps({"n_epochs": 7}), encoding="utf-8")
        ctrl = _controller(tmp_path)
        ctrl._save_refinement_config()
        assert json.loads(
            (tmp_path / "refinement_config.json").read_text(
                encoding="utf-8")) == {"n_epochs": 7}


class TestLoad:
    def test_reads_the_new_location(self, tmp_path):
        target = step3_refinement_config_path(tmp_path)
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(CONFIG), encoding="utf-8")
        ctrl = _controller(tmp_path)
        ctrl._load_refinement_config()
        assert ctrl.ui.refine_epochs_sb.value_set == 42

    def test_falls_back_to_the_old_root_file(self, tmp_path):
        (tmp_path / "refinement_config.json").write_text(
            json.dumps({"n_epochs": 13}), encoding="utf-8")
        ctrl = _controller(tmp_path)
        ctrl._load_refinement_config()
        assert ctrl.ui.refine_epochs_sb.value_set == 13

    def test_the_new_location_wins_over_the_root_copy(self, tmp_path):
        (tmp_path / "refinement_config.json").write_text(
            json.dumps({"n_epochs": 13}), encoding="utf-8")
        target = step3_refinement_config_path(tmp_path)
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"n_epochs": 99}), encoding="utf-8")
        ctrl = _controller(tmp_path)
        ctrl._load_refinement_config()
        assert ctrl.ui.refine_epochs_sb.value_set == 99

    def test_no_config_anywhere_warns_and_returns(self, tmp_path):
        ctrl = _controller(tmp_path)
        ctrl._load_refinement_config()
        assert ctrl.ui.refine_epochs_sb.value_set is None
        assert any("No refinement config" in m for m in ctrl.logs)

    def test_a_save_after_a_legacy_load_writes_to_the_new_place(self, tmp_path):
        (tmp_path / "refinement_config.json").write_text(
            json.dumps({"n_epochs": 13}), encoding="utf-8")
        ctrl = _controller(tmp_path)
        ctrl._load_refinement_config()
        ctrl._save_refinement_config()
        assert step3_refinement_config_path(tmp_path).exists()

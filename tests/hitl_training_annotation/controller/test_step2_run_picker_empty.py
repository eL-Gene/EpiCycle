"""The run picker when a workspace has NO runs — second B2 audit, FIX-FIRST #1.

Adopt skips Browse's first-run modal, so the picker is where a fresh project
says what is missing.  Two things were wrong:

* it invited the user to "click New Run" while `new_run_button` was disabled —
  on a NEW-layout workspace whose runs were all deleted, and on a genuinely
  fresh folder, the one button being pointed at was greyed out;
* a totally empty folder showed "Default (legacy layout)" in the picker while
  the banner said "No run yet" — two surfaces, two different stories about the
  same folder.

Qt-free: stub combo + buttons on a bare controller.
"""
from __future__ import annotations

import json

import pytest

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (  # noqa: E501
    TrainingAnnotationController,
)
from napari_epicycle.ui_shell import _workspace_facts as wf


class _Combo:
    def __init__(self):
        self.items = []
        self.enabled = None

    def blockSignals(self, on):
        pass

    def clear(self):
        self.items = []

    def addItem(self, text, data=None):
        self.items.append((text, data))

    def count(self):
        return len(self.items)

    def itemText(self, i):
        return self.items[i][0]

    def insertSeparator(self, i):
        pass

    def setCurrentIndex(self, i):
        pass

    def setEnabled(self, v):
        self.enabled = v

    def isEnabled(self):
        return bool(self.enabled)


class _Button:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, v):
        self.enabled = v

    def isEnabled(self):
        return bool(self.enabled)


class _UI:
    def __init__(self):
        self.run_picker_combo = _Combo()
        self.new_run_button = _Button()
        self.import_pairs_button = _Button()
        self.import_external_button = _Button()
        self.manage_runs_button = _Button()


def _controller(workspace):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.working_dir = str(workspace)
    ctrl.active_run_id = "default"
    return ctrl


def _new_layout(ws, runs=()):
    """A migrated (v3) workspace, optionally with run directories."""
    (ws / "step2_state.json").write_text(
        json.dumps({"schema_version": 3, "active_run_id": "default",
                    "runs": {}}), encoding="utf-8")
    for r in runs:
        (ws / "artifacts" / "step2" / "runs" / r).mkdir(parents=True)


class TestNewRunIsReachableWhenItIsTheOnlyThingToDo:
    def test_new_layout_with_zero_runs_enables_new_run(self, tmp_path):
        _new_layout(tmp_path)
        ctrl = _controller(tmp_path)
        ctrl._refresh_run_picker()
        assert ctrl.ui.run_picker_combo.itemText(0) == wf.MSG_STEP2_NO_RUN
        assert ctrl.ui.new_run_button.isEnabled() is True, (
            "the picker points at New Run; New Run must be pressable")

    def test_a_fresh_empty_folder_enables_new_run(self, tmp_path):
        ctrl = _controller(tmp_path)
        ctrl._refresh_run_picker()
        assert ctrl.ui.new_run_button.isEnabled() is True

    def test_no_workspace_still_disables_everything(self, tmp_path):
        ctrl = _controller("")
        ctrl._refresh_run_picker()
        assert ctrl.ui.run_picker_combo.itemText(0) == "(no workspace)"
        assert ctrl.ui.new_run_button.isEnabled() is False

    def test_a_legacy_workspace_still_needs_its_migration_first(self, tmp_path):
        (tmp_path / "training_data").mkdir()
        ctrl = _controller(tmp_path)
        ctrl._refresh_run_picker()
        assert ctrl.ui.new_run_button.isEnabled() is False, (
            "LEGACY / MIXED must migrate before a run can be made")


class TestPickerAndBannerAgree:
    def test_an_empty_folder_says_no_run_in_both_places(self, tmp_path):
        ctrl = _controller(tmp_path)
        ctrl._refresh_run_picker()
        picker = ctrl.ui.run_picker_combo.itemText(0)
        wf.clear_cache()
        banner = (wf.step2_layout_hint(str(tmp_path))
                  or (wf.MSG_STEP2_NO_RUN
                      if not wf.step2_has_runs(str(tmp_path)) else ""))
        assert picker == banner == wf.MSG_STEP2_NO_RUN, (
            f"picker={picker!r} banner={banner!r}")

    def test_a_legacy_folder_says_legacy_in_both_places(self, tmp_path):
        (tmp_path / "training_data").mkdir()
        ctrl = _controller(tmp_path)
        ctrl._refresh_run_picker()
        assert "legacy layout" in ctrl.ui.run_picker_combo.itemText(0).lower()
        wf.clear_cache()
        assert wf.step2_layout_hint(str(tmp_path)) == wf.MSG_STEP2_LEGACY_LAYOUT

    def test_the_two_no_run_strings_are_the_same_string(self, tmp_path):
        """They are written in two files; a drifting dash or wording would
        read as two different states to the user."""
        _new_layout(tmp_path)
        ctrl = _controller(tmp_path)
        ctrl._refresh_run_picker()
        assert ctrl.ui.run_picker_combo.itemText(0) == wf.MSG_STEP2_NO_RUN

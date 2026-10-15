"""Step 1's snapshot exports move under `artifacts/` — Brick B4.

Two things changed about `_ask_snapshot_path`:

* the folder it proposes is `artifacts/step1/exports/` (matching Step 4's
  `artifacts/step4/exports`) rather than a bare `exports/` in the workspace
  root;
* it is created only when the user actually confirms a destination.  Merely
  opening — and cancelling — the dialog used to leave an empty `exports/`
  folder behind in every workspace anyone ever looked at.
"""
from __future__ import annotations

import os

import pytest

from napari_epicycle.data_preparation._paths import step1_exports_dir


class _Line:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text


class _Widget:
    def __init__(self, ws):
        self.output_file_path = _Line(str(ws))


def _controller(ws):
    from napari_epicycle.data_preparation.controller import (
        DataPreparationController,
    )
    ctrl = DataPreparationController.__new__(DataPreparationController)
    ctrl.widget = _Widget(ws)
    return ctrl


@pytest.fixture
def dialog(monkeypatch):
    """Drive `QFileDialog.getSaveFileName`; records what was proposed."""
    seen = {}

    def install(answer):
        from qtpy.QtWidgets import QFileDialog

        def _fake(parent, title, default_name, filt):
            seen["default_name"] = default_name
            return (answer, filt)

        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(_fake))
        return seen

    return install


class TestThePathHelper:
    def test_exports_live_under_artifacts_step1(self, tmp_path):
        assert step1_exports_dir(tmp_path) == (
            tmp_path / "artifacts" / "step1" / "exports")


class TestTheDialogProposesTheNewFolder:
    def test_the_proposed_name_is_inside_artifacts(self, tmp_path, dialog):
        seen = dialog("")                       # user cancels
        ctrl = _controller(tmp_path)
        ctrl._ask_snapshot_path()
        assert seen["default_name"] == os.path.join(
            str(step1_exports_dir(tmp_path)), "snapshot.tiff")


class TestNothingIsCreatedUntilTheUserSaves:
    def test_cancelling_creates_no_folder(self, tmp_path, dialog):
        dialog("")
        ctrl = _controller(tmp_path)
        assert ctrl._ask_snapshot_path() == ""
        assert not (tmp_path / "artifacts").exists()
        assert not (tmp_path / "exports").exists()

    def test_confirming_creates_the_folder(self, tmp_path, dialog):
        target = step1_exports_dir(tmp_path) / "snapshot.tiff"
        dialog(str(target))
        ctrl = _controller(tmp_path)
        got = ctrl._ask_snapshot_path()
        assert got == str(target)
        assert step1_exports_dir(tmp_path).is_dir()

    def test_a_missing_suffix_is_still_added(self, tmp_path, dialog):
        target = step1_exports_dir(tmp_path) / "snapshot"
        dialog(str(target))
        ctrl = _controller(tmp_path)
        assert ctrl._ask_snapshot_path() == str(target) + ".tiff"

    def test_nothing_lands_in_the_workspace_root(self, tmp_path, dialog):
        target = step1_exports_dir(tmp_path) / "snapshot.tiff"
        dialog(str(target))
        ctrl = _controller(tmp_path)
        ctrl._ask_snapshot_path()
        assert not (tmp_path / "exports").exists()

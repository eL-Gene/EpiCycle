"""F22: a workspace change with unsaved edits or marks asks Save / Discard
/ Stay, and any workspace change unloads the stack (F22, grill 1 G1-15).  Qt-free: the controller runs on the fakes.

Today (before B1b) a Browse swapped the state but kept the loaded stack and
its dirty flag; Save then looked the record up in the NEW workspace, wrote
nothing and cleared the flag (S4c:4337, :4389).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.segmentation import _step4_text as T

from ._step4_fakes import controller


@pytest.fixture
def ctrl(tmp_path, monkeypatch):
    ws = tmp_path / "ws1"
    ws.mkdir()
    c = controller(ws)
    c.state_manager.set_stack_status("s.tif", "qc_extracted")
    c._current_stack_name = "s.tif"
    c._labels_volume = np.zeros((2, 2, 2), np.uint32)
    c._qc_features_df = pd.DataFrame({"label": [1]})
    c._labels_dirty = True
    c._undo_slot = {"kind": "smooth", "stack": "s.tif"}
    other = tmp_path / "ws2"
    other.mkdir()
    from qtpy.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(other)))
    c._other = str(other)
    # the Browse's own refreshes are not under test here
    c._restore_session_from_state = lambda *a, **k: None
    c._refresh_stacks = lambda: None
    return c


def _answer(monkeypatch, choice):
    import napari_epicycle.ui_shell.theme as theme
    asked = []
    monkeypatch.setattr(theme, "themed_three_choice",
                        lambda *a, **k: asked.append((a[1], a[2], k))
                        or choice)
    return asked


def test_stay_cancels_the_browse(ctrl, monkeypatch):
    asked = _answer(monkeypatch, "cancel")
    ws = ctrl.working_dir
    ctrl._browse_workdir()
    assert len(asked) == 1
    title, body, k = asked[0]
    assert title == T.LOAD_UNSAVED_TITLE
    assert body == T.LOAD_UNSAVED_BODY.format(stack="s.tif")
    assert (k["primary_text"], k["secondary_text"], k["cancel_text"]) == (
        T.LOAD_UNSAVED_SAVE, T.LOAD_UNSAVED_DISCARD, T.LOAD_UNSAVED_STAY)
    assert ctrl.working_dir == ws
    assert ctrl._current_stack_name == "s.tif" and ctrl._labels_dirty


def test_discard_unloads_then_changes(ctrl, monkeypatch):
    _answer(monkeypatch, "secondary")
    ctrl._browse_workdir()
    assert ctrl.working_dir == ctrl._other
    assert ctrl._current_stack_name is None
    assert ctrl._labels_volume is None and ctrl._qc_features_df is None
    assert not ctrl._has_unsaved_changes and ctrl._undo_slot is None


def test_a_failed_save_stays(ctrl, monkeypatch):
    _answer(monkeypatch, "primary")
    ws = ctrl.working_dir
    ctrl.state_manager.save_blocked = True
    ctrl._browse_workdir()
    assert ctrl.working_dir == ws
    assert ctrl._current_stack_name == "s.tif" and ctrl._labels_dirty


def test_a_good_save_then_changes_and_unloads(ctrl, monkeypatch):
    _answer(monkeypatch, "primary")
    monkeypatch.setattr(ctrl, "_save_tier_assignments",
                        lambda confirm=True: setattr(
                            ctrl, "_has_unsaved_changes", False) or True)
    ctrl._browse_workdir()
    assert ctrl.working_dir == ctrl._other
    assert ctrl._current_stack_name is None


def test_no_unsaved_work_asks_nothing_but_still_unloads(ctrl, monkeypatch):
    asked = _answer(monkeypatch, "cancel")
    ctrl._labels_dirty = False
    ctrl._browse_workdir()
    assert asked == []
    assert ctrl.working_dir == ctrl._other
    assert ctrl._current_stack_name is None


def test_the_same_workspace_keeps_the_stack(ctrl, monkeypatch):
    asked = _answer(monkeypatch, "cancel")
    from qtpy.QtWidgets import QFileDialog
    ws = ctrl.working_dir
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ws))
    ctrl._browse_workdir()
    assert asked == [] and ctrl._current_stack_name == "s.tif"


def test_adopt_asks_too(ctrl, monkeypatch):
    """The shell moves the shared workspace: same question."""
    asked = _answer(monkeypatch, "cancel")
    ws = ctrl.working_dir
    ctrl.adopt_workspace(ctrl._other, previous=ws)
    assert len(asked) == 1 and ctrl.working_dir == ws
    _answer(monkeypatch, "secondary")
    ctrl.adopt_workspace(ctrl._other, previous=ws)
    assert ctrl.working_dir == ctrl._other
    assert ctrl._current_stack_name is None


def test_browse_is_refused_while_a_job_runs(ctrl, monkeypatch):
    asked = _answer(monkeypatch, "secondary")
    ctrl._smooth_running = True
    ws = ctrl.working_dir
    ctrl._browse_workdir()
    assert asked == [] and ctrl.working_dir == ws
    assert T.LOCK_WORKSPACE_REFUSED in ctrl.ui.log_output.lines

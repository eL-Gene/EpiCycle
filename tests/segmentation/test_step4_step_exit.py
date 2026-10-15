"""B2: leaving Step 4 ("Step exit", F23 / audit
D13).

On exit the Workbench dock is hidden (not destroyed) and the Export window
is hidden; a Save before leaving that FAILS stops the step change, with the
reason on screen in ③ (``STEP_EXIT_SAVE_FAILED``); the shell honours an
exit hook's explicit ``False`` and nothing else.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import Qt

from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif",))
    w = panel(qapp, ws)
    c = w.controller
    run = ws / "artifacts" / "step4" / "runs" / "r1"
    run.mkdir(parents=True)
    lab = np.zeros((4, 8, 8), np.uint32)
    lab[1, 1:3, 1:3] = 1
    tifffile.imwrite(str(run / "labels_3d.tif"), lab)
    c.state_manager.set_stack_status(
        "P1.tif", "reconstructed", labels_path=str(run / "labels_3d.tif"))
    c._refresh_stacks()
    c._apply_loaded("P1.tif", {"labels": lab.copy(), "n_cells": 1,
                               "backup": "", "raw": None, "source_path": ""})
    c._step4_active = True
    pump(qapp)
    yield w
    close(w)


def _fake_viewer_with_dock(c):
    from qtpy.QtWidgets import QDockWidget, QLabel, QMainWindow
    main = QMainWindow()
    dock = QDockWidget(c._EDA_DOCK_NAME, main)
    dock.setWidget(QLabel("workbench"))
    main.addDockWidget(Qt.LeftDockWidgetArea, dock)
    main.show()
    dock.show()
    c.viewer = types.SimpleNamespace(window=types.SimpleNamespace(
        _qt_window=main))
    return main, dock


def test_exit_hides_the_workbench_dock_and_the_export_window(qapp, w):
    c = w.controller
    main, dock = _fake_viewer_with_dock(c)
    try:
        c._open_export_window()
        pump(qapp)
        assert w.export_dialog.isVisible()
        assert c.on_step_exit() is None
        pump(qapp)
        assert not w.export_dialog.isVisible()
        assert dock.isHidden()
        assert c._find_eda_dock() is dock          # hidden, not destroyed
        assert c._step4_active is False
    finally:
        c.viewer = None
        main.close()
        main.deleteLater()


def test_the_workbench_refuses_to_open_outside_step4(qapp, w):
    c = w.controller
    c._step4_active = False
    c._open_analysis_dock()
    assert c._analysis_widget is None


def test_an_empty_workbench_dock_is_skipped(qapp, w):
    """Napari's X can leave an EMPTY dock with the title:
    it is never the dock the Workbench is found in."""
    from qtpy.QtWidgets import QDockWidget, QMainWindow
    c = w.controller
    main = QMainWindow()
    empty = QDockWidget(c._EDA_DOCK_NAME, main)
    main.addDockWidget(Qt.LeftDockWidgetArea, empty)
    c.viewer = types.SimpleNamespace(window=types.SimpleNamespace(
        _qt_window=main))
    try:
        assert c._find_eda_dock() is None
    finally:
        c.viewer = None
        main.deleteLater()


def test_a_failed_save_before_leaving_stops_the_exit(qapp, w, monkeypatch):
    c = w.controller
    c._labels_dirty = True
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)
    notices = []
    monkeypatch.setattr(theme, "themed_notice",
                        lambda parent, title, message, **k:
                        notices.append((title, message)))
    monkeypatch.setattr(c, "_save_tier_assignments",
                        lambda confirm=True: False)
    assert c.on_step_exit() is False
    pump(qapp)
    msg = T.STEP_EXIT_SAVE_FAILED.format(stack="P1.tif")
    # R1-4: the whole sentence in a themed notice (the user just answered
    # a modal); the short form on ③'s one-line status
    assert notices == [(T.STEP_EXIT_SAVE_FAILED_TITLE, msg)]
    assert w.qc_status_line.text() == T.STEP_EXIT_SAVE_FAILED_SHORT.format(
        stack="P1.tif")
    assert w.accordion._current == 2
    assert c._step4_active is True                  # still in Step 4
    assert c._labels_dirty is True                  # the edits are kept
    assert msg in w.log_output.toPlainText()


def test_leave_without_saving_goes_on(qapp, w, monkeypatch):
    c = w.controller
    c._labels_dirty = True
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: False)
    assert c.on_step_exit() is None
    assert c._step4_active is False


def test_a_good_save_before_leaving_goes_on(qapp, w, monkeypatch):
    c = w.controller
    c._labels_dirty = True
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)
    monkeypatch.setattr(c, "_save_tier_assignments",
                        lambda confirm=True: True)
    assert c.on_step_exit() is None


def test_exit_is_refused_while_a_job_runs(qapp, w):
    c = w.controller
    c._predict_running = True
    try:
        assert c.on_step_exit() is False
        assert c._step4_active is True
    finally:
        c._predict_running = False


def _shell_for_select(ctrl_exit_result):
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
    shell = MagicMock()
    shell._step_index_by_id = {"apply_3d": 0, "project_setup": 1}
    shell._step_nav_locked = False
    shell._active_step_id = "apply_3d"
    shell._step_specs = [types.SimpleNamespace(step_id="apply_3d"),
                         types.SimpleNamespace(step_id="project_setup")]
    prev = types.SimpleNamespace(on_step_exit=lambda: ctrl_exit_result)
    shell._get_step_controller = lambda sid: prev if sid == "apply_3d" \
        else None
    shell._select_step = EpiCycleWorkflowShell._select_step.__get__(shell)
    return shell


def test_the_shell_stays_when_the_exit_hook_says_false():
    shell = _shell_for_select(False)
    shell._select_step("project_setup")
    assert shell._active_step_id == "apply_3d"
    shell.step_stack.setCurrentIndex.assert_not_called()


def test_the_shell_goes_on_when_the_exit_hook_returns_none():
    shell = _shell_for_select(None)
    try:
        shell._select_step("project_setup")
    except Exception:      # noqa: BLE001 -- the MagicMock shell stops later
        pass
    shell.step_stack.setCurrentIndex.assert_called_once_with(1)


def test_the_exit_prompt_names_marks_and_hand_edits():
    """D13 body: a mark and a hand paint are unsaved work too."""
    assert "(deletions, smoothing, marks or hand edits)" in T.STEP_EXIT_BODY

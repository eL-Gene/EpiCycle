"""B2: the full Step 4 lock (I-8, I-9).

Every job of ``STEP4_BLOCK_FLAGS`` against every control of the lock set:
grey or live per the lock table, the reason on screen (the job's one
``STEP4_LOCK_<JOB>`` sentence in the lock line of ①, ② and ③ alike), the
Export window's buttons grey with a sentence on its own status line, and
the step nav held under the ``"step4"`` owner only while Step 4 is the
active step -- and released on the job's end even when its worker dies.
A Workbench map (``_eda_running``) never locks the nav.  Step 2's training
lock is never released by a Step 4 job (a real shell method, two owners).

The real panel with its controller (``viewer=None``); a plain QApplication,
no pytest-qt.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import Qt

from napari_epicycle.segmentation import _step4_lock as L
from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump, tick

#: One sentence per job: the flag -> the lock line's text.
JOB_SENTENCE = {
    "_predict_running": T.STEP4_LOCK_PREDICT,
    "_reconstruct_running": T.STEP4_LOCK_RECONSTRUCT,
    "_measure_running": T.STEP4_LOCK_MEASURE,
    "_smooth_running": T.STEP4_LOCK_SMOOTH.format(stack="P1.tif"),
    "_save_in_flight": T.STEP4_LOCK_SAVE.format(stack="P1.tif"),
    "_load_in_flight": T.STEP4_LOCK_LOAD.format(stack="P1.tif"),
    "_preview_in_flight": T.STEP4_LOCK_LOAD.format(stack="P1.tif"),
    "_export_running": T.STEP4_LOCK_EXPORT,
    "_mesh_running": T.STEP4_LOCK_EXPORT,
}

#: ① and ② controls every job greys.
SURFACE = ("workdir_line", "browse_workdir_btn", "add_folder_btn",
           "set_spacing_btn", "run_picker_combo", "model_change_btn",
           "start_from_combo", "chan1_combo", "chan2_combo",
           "cell_width_um_sb", "predict_btn", "reconstruct_btn")
#: ③'s controls every job greys (the live one is "What is measured?").
MT = ("extract_features_btn", "open_analysis_btn", "delete_bad_btn",
      "undo_btn", "smooth_all_btn", "smooth_strength_combo",
      "save_tiers_btn", "finalize_btn", "reset_original_btn", "export_btn")


class _Shell:
    """The REAL ``lock_step_nav`` bound to a stand-in with real buttons."""

    def __init__(self):
        from qtpy.QtWidgets import QPushButton
        from napari_epicycle.ui_shell.workflow_shell import (
            EpiCycleWorkflowShell,
        )
        self._nav_items = {k: QPushButton(k) for k in
                           ("project_setup", "roi_hitl_2d", "apply_3d")}
        self.lock_step_nav = EpiCycleWorkflowShell.lock_step_nav.__get__(self)

    def locked(self) -> bool:
        return not any(b.isEnabled() for b in self._nav_items.values())

    def owners(self) -> set:
        return set(getattr(self, "_step_nav_owners", set()))


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"))
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
    c._workflow_shell = _Shell()
    # R1-1: a model, a membrane channel and a ticked stack, so Predict is
    # LIVE when idle and a job must grey it (not a vacuous pass)
    c._on_model_use({"_kind": "pretrained", "name": "cyto3"})
    w.set_channel_choice(w.chan1_combo, 0)
    c._on_channels_changed()
    pump(qapp)
    tick(w, qapp, "P1.tif", "P2.tif")
    pump(qapp)
    yield w
    for flag in L.ALL_FLAGS:
        setattr(c, flag, False)
    close(w)


def _lock(w, qapp, flag, on=True):
    w.controller._set_step4_flag(flag, on)
    pump(qapp)


def _checkable(lw):
    return [bool(lw.item(i).flags() & Qt.ItemIsUserCheckable)
            for i in range(lw.count())]


# -------------------------------------------------------------- every job


@pytest.mark.parametrize("flag", L.STEP4_BLOCK_FLAGS)
def test_every_job_greys_the_surface_with_its_sentence(qapp, w, flag):
    c = w.controller
    assert any(_checkable(w.stacks_list))           # idle: ticks live
    assert w.predict_btn.isEnabled()                # idle: Predict live
    _lock(w, qapp, flag)
    for name in SURFACE + MT:
        assert not getattr(w, name).isEnabled(), f"{name} live under {flag}"
    assert not any(_checkable(w.stacks_list))
    assert not any(_checkable(w.session_stacks_list))
    # I-9: the one sentence, in ①, ② and ③ alike, on screen
    for line in (w.session_lock_line, w.seg_lock_line, w.mt_lock_line):
        assert line.text() == JOB_SENTENCE[flag]
        assert not line.isHidden()
    # the live one of ③ (D3)
    assert w.feature_reference_btn.isEnabled()
    # R1-1: View is live only under Predict / Reconstruct
    if flag not in ("_predict_running", "_reconstruct_running"):
        assert not w.view_btn.isEnabled()
        # and ②'s status line never says "Ready" under a lock
        assert T.seg_idle_line(2) not in (w.run_status_label.toolTip()
                                           or w.run_status_label.text())
    _lock(w, qapp, flag, on=False)
    for name in ("workdir_line", "browse_workdir_btn", "add_folder_btn",
                 "run_picker_combo", "model_change_btn", "start_from_combo",
                 "export_btn"):
        assert getattr(w, name).isEnabled(), f"{name} stays grey"
    assert any(_checkable(w.stacks_list))
    for line in (w.session_lock_line, w.seg_lock_line, w.mt_lock_line):
        assert line.isHidden()
    assert c._step4_nav_held is False


def test_a_tick_during_a_job_changes_nothing(qapp, w):
    c = w.controller
    before = list(c.state_manager.get_ticked())
    _lock(w, qapp, "_predict_running")
    it = w.stacks_list.item(0)
    # the delegate refuses a click on a box without ItemIsUserCheckable
    assert not it.flags() & Qt.ItemIsUserCheckable
    pump(qapp)
    assert list(c.state_manager.get_ticked()) == before


def test_rows_rebuilt_during_a_job_stay_unticked(qapp, w):
    c = w.controller
    _lock(w, qapp, "_reconstruct_running")
    c._refresh_stacks()                     # a progress rebuild mid-run
    pump(qapp)
    assert not any(_checkable(w.stacks_list))
    assert not any(_checkable(w.session_stacks_list))


def test_a_setting_keeps_its_own_grey_after_unlock(qapp, w):
    """The settings go grey through their container, so a widget grey for
    its own reason stays grey after the job ends."""
    w.cell_width_um_sb.setEnabled(False)
    _lock(w, qapp, "_predict_running")
    _lock(w, qapp, "_predict_running", on=False)
    assert not w.cell_width_um_sb.isEnabled()
    assert w.start_from_combo.isEnabled()


# -------------------------------------------------- the live set per job


@pytest.mark.parametrize("flag", ("_predict_running", "_reconstruct_running"))
def test_cancel_and_view_stay_live_during_a_run(qapp, w, flag):
    c = w.controller
    w.cancel_btn.setEnabled(True)           # the launch enables Cancel
    # highlight a row with a result so View has something to show
    for i in range(w.stacks_list.count()):
        if w.stacks_list.item(i).data(Qt.UserRole) == "P1.tif":
            w.stacks_list.setCurrentRow(i)
    pump(qapp)
    _lock(w, qapp, flag)
    assert w.cancel_btn.isEnabled()
    assert w.view_btn.isEnabled()
    assert not w.predict_btn.isEnabled()
    # a ② row click still selects (select only)
    w.stacks_list.setCurrentRow(1)
    pump(qapp)
    assert w.stacks_list.currentRow() == 1
    # a ③ row click does not load
    loads = []
    c._launch_load = lambda *a, **k: loads.append(a)
    c._request_load("P2.tif")
    assert c._current_stack_name == "P1.tif"


def test_the_menu_does_not_open_under_the_lock(qapp, w):
    c = w.controller
    built = []
    c._build_stack_menu = lambda name: built.append(name)
    _lock(w, qapp, "_measure_running")
    c._on_stack_menu("P1.tif", None)
    assert built == []
    assert T.LOCK_MENU_REFUSED in c.ui.log_output.toPlainText()


def test_workspace_refused_under_the_lock(qapp, w, tmp_path):
    c = w.controller
    before = c.working_dir
    _lock(w, qapp, "_predict_running")
    c.adopt_workspace(str(tmp_path / "other"), previous=before)
    c._browse_workdir()
    assert c.working_dir == before
    assert T.LOCK_WORKSPACE_REFUSED in c.ui.log_output.toPlainText()


# ------------------------------------------------------ the Export window


def test_the_export_window_greys_with_the_running_jobs_sentence(qapp, w):
    c = w.controller
    dlg = w.export_dialog
    c._open_export_window()
    pump(qapp)
    _lock(w, qapp, "_predict_running")
    assert not dlg.export_features_btn.isEnabled()
    assert not dlg.export_meshes_btn.isEnabled()
    assert not dlg.export_rows_list.isEnabled()
    assert dlg.export_status_label.text() == T.STEP4_LOCK_PREDICT
    # R1-3: the window's sentence labels word-wrap -- the whole lock
    # sentence is on screen, not cut at ~75 characters
    from qtpy.QtWidgets import QLabel
    lbl = dlg.export_status_label
    assert lbl.wordWrap()
    assert QLabel.text(lbl) == T.STEP4_LOCK_PREDICT
    assert lbl.height() >= 2 * lbl.fontMetrics().lineSpacing()
    _lock(w, qapp, "_predict_running", on=False)
    assert dlg.export_status_label.text() != T.STEP4_LOCK_PREDICT
    assert dlg.export_rows_list.isEnabled()
    _lock(w, qapp, "_export_running")
    assert dlg.export_status_label.text() == T.EXPORT_RUNNING
    dlg.hide()


# ------------------------------------------------------------ the nav lock


def test_the_nav_is_taken_by_step4_and_released(qapp, w):
    c = w.controller
    shell = c._workflow_shell
    _lock(w, qapp, "_measure_running")
    assert shell.locked() and shell.owners() == {"step4"}
    _lock(w, qapp, "_measure_running", on=False)
    assert not shell.locked() and shell.owners() == set()


def test_the_nav_is_not_taken_when_step4_is_not_active(qapp, w):
    c = w.controller
    c._step4_active = False
    _lock(w, qapp, "_predict_running")
    assert not c._workflow_shell.locked()
    assert c._workflow_shell.owners() == set()


def test_a_workbench_map_never_locks_the_nav(qapp, w):
    c = w.controller
    _lock(w, qapp, "_eda_running")
    assert not c._workflow_shell.locked()
    assert w.browse_workdir_btn.isEnabled()          # not a Step 4 lock
    assert w.session_lock_line.isHidden()


def test_step2s_training_lock_is_not_released_by_a_step4_job(qapp, w):
    c = w.controller
    shell = c._workflow_shell
    shell.lock_step_nav(True)               # Step 2's call: the default owner
    _lock(w, qapp, "_save_in_flight")
    _lock(w, qapp, "_save_in_flight", on=False)
    assert shell.locked()
    assert shell.owners() == {"default"}
    shell.lock_step_nav(False)
    assert not shell.locked()


def test_step4s_lock_outlives_a_default_owner_release(qapp, w):
    c = w.controller
    shell = c._workflow_shell
    _lock(w, qapp, "_predict_running")
    shell.lock_step_nav(False)              # Step 2 / 3 releasing theirs
    assert shell.locked() and shell.owners() == {"step4"}


def test_two_step4_jobs_release_the_nav_once_both_end(qapp, w):
    c = w.controller
    shell = c._workflow_shell
    _lock(w, qapp, "_predict_running")
    _lock(w, qapp, "_load_in_flight")
    _lock(w, qapp, "_predict_running", on=False)
    assert shell.locked()
    _lock(w, qapp, "_load_in_flight", on=False)
    assert not shell.locked()


def test_re_entering_step4_with_nothing_running_leaves_the_nav_free(qapp, w):
    c = w.controller
    c._step4_active = False
    c.on_step_enter()
    assert c._step4_active is True
    assert not c._workflow_shell.locked()


# --------------------------------------- a dying worker releases the nav


def test_a_worker_that_dies_releases_the_nav(qapp, tmp_path, monkeypatch):
    """The import error of B0's backstop test, with a shell attached: the
    thread's ``finished`` clears the flag and gives the owner back."""
    import packages.epicycle_reconstruction.logic as logic
    from napari_epicycle.segmentation import _step4_models as M

    from ._step4_fakes import controller

    monkeypatch.delattr(logic, "segment_views")
    ctrl = controller(tmp_path)
    shell = _Shell()
    ctrl._workflow_shell = shell
    ctrl._step4_active = True
    ctrl._seg_start_time = time.monotonic()
    ctrl._active_model = M.model_snapshot("cyto3", "cyto3",
                                          kind="pretrained")
    ctrl.state_manager.register_stacks(["s.tif"])
    ctrl.state_manager.set_stack_status(
        "s.tif", "unprocessed", axes="CZYX", stack_shape=[2, 6, 24, 24],
        n_channels=2)
    ctrl._launch_prediction_worker([{
        "stack_name": "s.tif", "source_stack_path": "x.tif",
        "lateral_model_path": "L", "axial_model_path": "A"}])
    assert shell.owners() == {"step4"}
    t0 = time.monotonic()
    while ctrl._predict_running and time.monotonic() - t0 < 20:
        qapp.processEvents()
        time.sleep(0.01)
    assert ctrl._predict_running is False
    assert shell.owners() == set() and not shell.locked()
    for th in getattr(ctrl, "_retired_threads", []):
        th.wait(3000)


# --------------------------------------------------------- the Workbench


def test_the_workbench_shows_wb_locked_and_greys_its_marks(qapp, w):
    from napari_epicycle.segmentation.analysis_widget import (
        AnalysisControlsWidget,
    )
    from napari_epicycle.segmentation import _workbench_text as WT

    c = w.controller
    aw = AnalysisControlsWidget()
    c._analysis_widget = aw
    try:
        _lock(w, qapp, "_measure_running")
        assert aw._locked_job == WT.LOCK_JOB_WORDS["_measure_running"]
        assert not aw.mark_good_btn.isEnabled()
        assert not aw.compute_umap_btn.isEnabled()
        locked_text = WT.WB_LOCKED.format(
            job=WT.LOCK_JOB_WORDS["_measure_running"])
        assert aw.sel_msg_label.text() == locked_text
        # R1-2: a lasso / pick / gate line never replaces WB_LOCKED while
        # the Mark buttons are grey
        for text in (WT.LASSO_DONE, WT.CAT_PICKED, ""):
            aw.show_card_message(text, "ok")
            assert aw.sel_msg_label.text() == locked_text
        _lock(w, qapp, "_measure_running", on=False)
        assert aw._locked_job is None
        assert aw.compute_umap_btn.isEnabled()
    finally:
        c._analysis_widget = None
        aw.deleteLater()


# ------------------------------------------------ K-lens audit (part 2)


@pytest.mark.parametrize("flag", ("_load_in_flight", "_measure_running",
                                  "_smooth_running", "_save_in_flight",
                                  "_export_running"))
def test_the_eye_refuses_under_a_non_run_job(qapp, w, flag):
    """K3: the eye reads a TIFF and adds layers, so it is live under
    Predict / Reconstruct only; the delegate dims it with the lock."""
    c = w.controller
    calls = []
    c._preview_selected_stack = lambda *a, **k: calls.append(a)
    c.viewer = type("V", (), {"layers": []})()
    try:
        _lock(w, qapp, flag)
        c._on_stack_eye("P1.tif")
        assert calls == []
        assert w.session_stacks_list.property("step4EyeLocked") is True
    finally:
        c.viewer = None


def test_the_eye_stays_live_under_predict(qapp, w):
    c = w.controller
    calls = []
    c._preview_selected_stack = lambda *a, **k: calls.append(a)
    c.viewer = type("V", (), {"layers": []})()
    try:
        _lock(w, qapp, "_predict_running")
        c._on_stack_eye("P1.tif")
        assert len(calls) == 1
        assert not w.session_stacks_list.property("step4EyeLocked")
    finally:
        c.viewer = None


@pytest.mark.parametrize("flag", ("_smooth_running", "_measure_running",
                                  "_load_in_flight"))
def test_the_labels_layer_is_locked_during_a_volume_job(qapp, w, flag):
    """K4: a paint mid-job would make the job's result stale, so the
    labels layer is not editable while Smooth / Measure / Load run."""
    import types
    c = w.controller
    layer = types.SimpleNamespace(editable=True)
    c._find_labels_layer = lambda: layer
    _lock(w, qapp, flag)
    assert layer.editable is False
    _lock(w, qapp, flag, on=False)
    assert layer.editable is True


def test_the_labels_layer_stays_editable_under_predict(qapp, w):
    import types
    c = w.controller
    layer = types.SimpleNamespace(editable=True)
    c._find_labels_layer = lambda: layer
    _lock(w, qapp, "_predict_running")
    assert layer.editable is True


def test_help_and_the_advanced_disclosure_stay_live(qapp, w):
    """K5: the lock greys the settings, never ②'s ? button or the
    Advanced disclosure."""
    _lock(w, qapp, "_predict_running")
    for btn, _guide in w._help_buttons:
        assert btn.isEnabled()
    assert w._adv_collapsible._toggle_btn.isEnabled()
    assert not w.start_from_combo.isEnabled()


def test_step4_starts_inactive_and_a_job_ending_then_is_a_no_op(qapp,
                                                                tmp_path):
    """K6: _step4_active is False until on_step_enter."""
    from ._step4_fakes import controller
    ctrl = controller(tmp_path)
    ctrl._step4_active = False
    shell = _Shell()
    ctrl._workflow_shell = shell
    ctrl._set_step4_flag("_measure_running", True)
    ctrl._set_step4_flag("_measure_running", False)
    assert shell.owners() == set() and not shell.locked()
    from napari_epicycle.segmentation.controller import (
        SegmentStackController,
    )
    fresh = SegmentStackController.__new__(SegmentStackController)
    fresh._init_runtime_state()
    assert fresh._step4_active is False


def test_the_workbench_refuses_marks_outside_step4(qapp, w):
    """K7: a Workbench dock re-shown from napari's Window menu in another
    step greys Mark / Unmark / Undo / Clear with a sentence on its card."""
    from napari_epicycle.segmentation.analysis_widget import (
        AnalysisControlsWidget,
    )
    from napari_epicycle.segmentation import _workbench_text as WT
    c = w.controller
    aw = AnalysisControlsWidget()
    c._analysis_widget = aw
    try:
        aw.set_mark_state(has_selection=True, can_undo=True)
        c.on_step_exit()
        assert not aw.mark_good_btn.isEnabled()
        assert not aw.unmark_btn.isEnabled()
        assert not aw.undo_mark_btn.isEnabled()
        assert not aw.clear_selection_btn.isEnabled()
        assert aw.sel_msg_label.text() == WT.WB_STEP_INACTIVE
        aw.show_card_message("x")
        assert aw.sel_msg_label.text() == WT.WB_STEP_INACTIVE
        assert c._mark_refused("Mark") is True
        c.on_step_enter()
        assert aw.mark_good_btn.isEnabled()
        assert aw.sel_msg_label.text() != WT.WB_STEP_INACTIVE
    finally:
        c._analysis_widget = None
        aw.deleteLater()


# ------------------------------------------ lifecycle re-audit (H1, M1)


def test_every_spinbox_still_types_after_a_lock(qapp, w):
    """H1: findChildren also yields each spinbox's own QLineEdit; after a
    lock / unlock cycle every ② spin box must still accept typing."""
    from qtpy.QtWidgets import QAbstractSpinBox
    _lock(w, qapp, "_predict_running")
    _lock(w, qapp, "_predict_running", on=False)
    _lock(w, qapp, "_measure_running")
    _lock(w, qapp, "_measure_running", on=False)
    dead = [sb.objectName() or type(sb).__name__
            for sb in w.seg_right_content.findChildren(QAbstractSpinBox)
            if sb.isEnabled() and not sb.lineEdit().isEnabled()]
    assert dead == []


def test_the_flow_gate_is_re_run_after_the_unlock(qapp, w):
    """M1: a gate changed during the lock is applied at unlock, not a
    stale prior."""
    c = w.controller
    calls = []
    orig_flow = w._update_flow_controls
    w._update_flow_controls = lambda *a, **k: (calls.append("flow"),
                                               orig_flow(*a, **k))
    _lock(w, qapp, "_predict_running")
    _lock(w, qapp, "_predict_running", on=False)
    assert "flow" in calls


def test_smooth_is_handed_a_copy_of_the_volume(qapp, w, monkeypatch):
    """L1: Ctrl+Z writes the layer data without a paint event, so the
    Smooth worker must get its own copy, taken on the main thread."""
    from napari_epicycle.segmentation import controller as CM
    c = w.controller
    seen = []

    class _W:
        def __init__(self, volume, **kw):
            seen.append(volume)

    monkeypatch.setattr(CM, "_SmoothBatchWorker", _W)
    monkeypatch.setattr(c, "_launch_smooth_worker", lambda worker: None)
    c._smooth_all()
    assert seen and seen[0] is not c._labels_volume
    assert (seen[0] == c._labels_volume).all()


def test_the_layer_mode_comes_back_after_the_unlock(qapp, w):
    """L2: editable=False forces pan/zoom; the unlock puts the mode back."""
    import types
    c = w.controller
    layer = types.SimpleNamespace(editable=True, mode="paint")
    c._find_labels_layer = lambda: layer
    _lock(w, qapp, "_smooth_running")
    layer.mode = "pan_zoom"                   # what napari does
    _lock(w, qapp, "_smooth_running", on=False)
    assert layer.editable is True and layer.mode == "paint"


def test_a_layer_that_was_not_editable_stays_so(qapp, w):
    """L4: only a layer Step 4 locked is unlocked."""
    import types
    c = w.controller
    layer = types.SimpleNamespace(editable=False, mode="pan_zoom")
    c._find_labels_layer = lambda: layer
    _lock(w, qapp, "_measure_running")
    _lock(w, qapp, "_measure_running", on=False)
    assert layer.editable is False


def test_dead_stack_layers_are_pruned_on_re_entry(qapp, w):
    """L3: layers removed while away are not kept (nor their data)."""
    import types
    c = w.controller
    gone = types.SimpleNamespace(name="old")
    c._stack_layers = [gone]
    c.viewer = type("V", (), {"layers": []})()
    try:
        c._labels_volume = None
        c.on_step_enter()
        assert c._stack_layers == []
    finally:
        c.viewer = None

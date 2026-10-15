"""The Predict + Reconstruct lock line and surface lock
(brick V1).

* ``_refresh_step3_lock`` ends with the lock line: the first set flag of
  ``S3_LOCK_SENTENCES`` (specific first, ``_curation_busy`` last); two
  sentence families -- jobs in Step 4's form, annotator sessions in the
  imperative -- and every flag of ``STEP3_BLOCK_FLAGS`` has one.
* ``_patch_load_in_flight`` / ``_curation_busy`` show the line only after
  ``S3_LOCK_DELAY_MS`` and grey nothing.
* Only Predict / Reconstruct grey: the run row, Change…, the settings'
  top-level inputs (each back to its OWN state; the ? buttons and the
  disclosures stay live; the column's gates re-run), the ticks; Cancel and
  View stay live.  View is grey under every other flag.
* ``lock_step_nav(locked, owner="step3")``.

Step 4's R1-1 lesson: each control is first proved live WITHOUT the lock.
Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _P():
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    return P


@pytest.fixture
def full(app, tmp_path):
    import numpy as np
    import tifffile

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration(viewer=None)
    c = w.controller
    c.working_dir = str(tmp_path)
    c.state_manager.active_run_id = "default"
    c.state_manager.load(str(tmp_path))
    folder = tmp_path / "patches"
    folder.mkdir()
    for name in ("a.tif", "b.tif"):
        tifffile.imwrite(str(folder / name), np.zeros((4, 8, 8), np.uint8))
    c.loaded_stack_dir = str(folder)
    c.state_manager._state["patches"] = {
        "a.tif": {"status": "unprocessed"}, "b.tif": {"status": "predicted"}}
    c.state_manager._state["ticked"] = ["a.tif", "b.tif"]
    w.lateral_model_path.setText("cyto3")
    w.axial_model_path.setText("cyto3")
    c._refresh_patch_state_list()
    w.resize(900, 1200)
    w.show()
    app.processEvents()
    yield w, c
    for f in ("_prediction_running", "_reconstruction_running",
              "_patch_load_in_flight", "_curation_busy", "_refine_running"):
        setattr(c, f, False)
    timer = getattr(c.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    w.close()
    w.deleteLater()
    app.processEvents()


def _set(c, flag, on=True):
    setattr(c, flag, on)
    c._refresh_step3_lock()


GREYED_BY_RUN = ("run_picker_combo", "new_run_button", "manage_runs_button",
                 "model_change_btn", "start_from_combo",
                 "chan1_combo", "diameter_sb", "norm_cb", "batch_sb")


# ------------------------------------------------------------ the sentences

def test_every_block_flag_has_a_sentence_of_the_right_family():
    from napari_epicycle.fusion_reconstruction._switch_run_guard import (
        STEP3_BLOCK_FLAGS,
    )
    P = _P()
    table = dict(P.S3_LOCK_SENTENCES)
    assert set(table) == set(STEP3_BLOCK_FLAGS)
    for flag, sentence in table.items():
        if flag in P.S3_LOCK_SESSION_FLAGS:       # the imperative
            assert sentence.endswith(" there."), flag
            assert "wait" not in sentence.lower(), flag
        else:                                     # a job: Step 4's form
            assert ("Cancel stops it" in sentence
                    or "wait for it to finish" in sentence), flag


def test_the_priority_is_specific_first_and_busy_last():
    P = _P()
    order = [f for f, _s in P.S3_LOCK_SENTENCES]
    assert order[0] == "_prediction_running"
    assert order[1] == "_reconstruction_running"
    assert order[-1] == "_curation_busy"
    assert order.index("_refine_running") < order.index(
        "_patch_load_in_flight")


def test_the_job_sentences_are_step_4s_form():
    P = _P()
    assert P.S3_LOCK_PREDICT == (
        "Predicting — Cancel stops it after the current patch. The run, "
        "the ticks, the model and the settings are locked.")
    assert P.S3_LOCK_REFINE == (
        "A refine step is staged in the annotator — Accept or Discard it "
        "there.")


# ------------------------------------------------------------ the line

def test_predict_shows_its_line_and_unlock_hides_it(full):
    w, c = full
    assert w.pr_lock_line.isHidden()
    _set(c, "_prediction_running")
    assert not w.pr_lock_line.isHidden()
    assert w.pr_lock_line.text() == _P().S3_LOCK_PREDICT
    _set(c, "_prediction_running", False)
    assert w.pr_lock_line.isHidden()


def test_the_first_flag_wins(full):
    w, c = full
    c._curation_busy = True
    _set(c, "_reconstruction_running")
    assert w.pr_lock_line.text() == _P().S3_LOCK_RECONSTRUCT
    c._curation_busy = False


@pytest.mark.parametrize("flag", ["_patch_load_in_flight", "_curation_busy"])
def test_a_brief_flag_shows_its_line_only_after_the_delay(app, full, flag):
    from qtpy.QtTest import QTest
    w, c = full
    _set(c, flag)
    assert w.pr_lock_line.isHidden()              # not at once
    QTest.qWait(_P().S3_LOCK_DELAY_MS + 150)
    assert not w.pr_lock_line.isHidden()
    assert w.pr_lock_line.text() == dict(_P().S3_LOCK_SENTENCES)[flag]
    _set(c, flag, False)
    assert w.pr_lock_line.isHidden()


def test_a_brief_flag_that_ends_in_time_never_shows(app, full):
    from qtpy.QtTest import QTest
    w, c = full
    _set(c, "_patch_load_in_flight")
    _set(c, "_patch_load_in_flight", False)
    QTest.qWait(_P().S3_LOCK_DELAY_MS + 150)
    assert w.pr_lock_line.isHidden()


@pytest.mark.parametrize("flag", ["_refine_running", "_patch_load_in_flight",
                                  "_curation_busy"])
def test_every_other_flag_greys_nothing_of_the_surface(full, flag):
    w, c = full
    for name in GREYED_BY_RUN:
        assert getattr(w, name).isEnabled(), name      # live first
    _set(c, flag)
    for name in GREYED_BY_RUN:
        assert getattr(w, name).isEnabled(), (flag, name)
    assert not w._patch_state_list.property("step3Locked")


# ------------------------------------------------------------ the greying

def test_predict_greys_the_surface_and_keeps_cancel_and_view(full):
    w, c = full
    for name in GREYED_BY_RUN:
        assert getattr(w, name).isEnabled(), name      # live first
    assert w.view_reconstructed_button.isEnabled()
    help_buttons = [b for b, _g in w._pr_help_buttons]
    _set(c, "_prediction_running")
    c._update_button_states()
    for name in GREYED_BY_RUN:
        assert not getattr(w, name).isEnabled(), name
    assert w.pipeline_cancel_btn.isEnabled()
    assert w.view_reconstructed_button.isEnabled()
    assert all(b.isEnabled() for b in help_buttons)
    assert w._pr_adv_collapsible.isEnabled()          # the disclosure
    _set(c, "_prediction_running", False)
    for name in GREYED_BY_RUN:
        assert getattr(w, name).isEnabled(), name


def test_view_is_grey_under_every_other_flag(full):
    w, c = full
    assert w.view_reconstructed_button.isEnabled()
    _set(c, "_curation_busy")
    assert not w.view_reconstructed_button.isEnabled()
    _set(c, "_curation_busy", False)
    assert w.view_reconstructed_button.isEnabled()


def test_each_input_gets_its_own_state_back_and_the_gates_rerun(full):
    w, c = full
    w.ps_smooth_flows_cb.setChecked(False)        # the gate greys the two
    assert not w.ps_mask_thresh_sb.isEnabled()
    w.batch_sb.setEnabled(False)                  # greyed by hand
    _set(c, "_prediction_running")
    w.ps_smooth_flows_cb.setChecked(True)         # changed during the lock
    _set(c, "_prediction_running", False)
    assert not w.batch_sb.isEnabled()             # its own state
    assert w.ps_mask_thresh_sb.isEnabled()        # the gate re-ran
    w.batch_sb.setEnabled(True)


def test_a_spin_box_takes_typing_after_the_lock(full):
    from qtpy.QtWidgets import QLineEdit
    w, c = full
    inputs = w._pr_setting_inputs()
    assert w.diameter_sb in inputs
    assert not any(isinstance(x, QLineEdit)
                   and x.parentWidget() is w.diameter_sb for x in inputs)
    _set(c, "_prediction_running")
    _set(c, "_prediction_running", False)
    assert w.diameter_sb.lineEdit().isEnabled()


def test_the_ticks_go_off_quietly_under_predict(full):
    from qtpy.QtCore import Qt
    w, c = full
    lst = w._patch_state_list
    assert lst.item(0).flags() & Qt.ItemIsUserCheckable   # live first
    _set(c, "_prediction_running")
    assert lst.property("step3Locked")
    for i in range(lst.count()):
        assert not (lst.item(i).flags() & Qt.ItemIsUserCheckable)
        assert lst.item(i).flags() & Qt.ItemIsSelectable   # View's
    lst.item(0).setCheckState(Qt.Unchecked)       # a programmatic try
    # the run's ticks are read back unchanged, and the state agrees
    assert c._ticked_patch_names() == ["a.tif", "b.tif"]
    assert c.state_manager.get_ticked() == ["a.tif", "b.tif"]
    # a rebuild during the run keeps them off
    c._refresh_patch_state_list()
    assert not (lst.item(0).flags() & Qt.ItemIsUserCheckable)
    assert w.pr_locked_list_note.isVisible()
    assert not w.pr_ticked_predict_label.isVisible()
    assert not w.pr_tick_all_link.isVisible()
    _set(c, "_prediction_running", False)
    assert lst.item(0).flags() & Qt.ItemIsUserCheckable
    assert not w.pr_locked_list_note.isVisible()
    assert w.pr_tick_all_link.isVisible()


def test_the_nav_lock_is_step_3s_own(full):
    w, c = full
    calls = []

    class _Shell:
        def lock_step_nav(self, locked, owner="default"):
            calls.append((locked, owner))

    c._workflow_shell = _Shell()
    _set(c, "_reconstruction_running")
    _set(c, "_reconstruction_running", False)
    assert calls[-2:] == [(True, "step3"), (False, "step3")]

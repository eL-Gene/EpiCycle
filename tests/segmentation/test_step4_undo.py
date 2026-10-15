"""One-level Undo in ③ Review (grill 1 G1-10 /
G1-36).  Qt-free: the controller runs on the Qt-free fakes.

Undo takes back the last Remove bad cells or Smooth outlines, whichever
came last; it restores the volume and the rows it changed and sets the
labels-dirty flag; it never brings the maps back; a second edit replaces
the slot; Save, Measure, Reset, a load and a workspace change empty it; a
press with nothing to undo says ``MT_UNDO_EMPTY`` (never grey).
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.segmentation import _step4_text as T

from ._step4_fakes import controller


@pytest.fixture
def red(monkeypatch):
    """Answer the red Remove confirm with yes, and record its accent."""
    import napari_epicycle.ui_shell.theme as theme
    seen = []

    def _confirm(*a, **k):
        seen.append(k.get("accent"))
        return True
    monkeypatch.setattr(theme, "themed_confirm", _confirm)
    return seen


@pytest.fixture
def ctrl(tmp_path):
    c = controller(tmp_path)
    c.state_manager.set_stack_status("s.tif", "qc_extracted")
    c._current_stack_name = "s.tif"
    vol = np.zeros((2, 4, 4), np.uint32)
    vol[0, 0, 0] = 1
    vol[0, 1, 1] = 2
    vol[1, 2, 2:4] = 3
    c._labels_volume = vol
    c._n_cells = 3
    c._qc_features_df = pd.DataFrame({"label": [1, 2, 3],
                                      "qc_tier": ["bad", "good", "bad"]})
    c._open_analysis_dock = lambda: None
    return c


def test_undo_of_remove_restores_the_voxels_and_the_rows(ctrl, red):
    before = ctrl._labels_volume.copy()
    ctrl._delete_bad_labels()
    assert red == ["red"]                          # REMOVE_CONFIRM is red
    assert int((ctrl._labels_volume == 1).sum()) == 0
    assert list(ctrl._qc_features_df["label_status"]) == [
        "deleted", "active", "deleted"]
    assert ctrl.ui.qc_status_line.sets[-1] == T.say(T.MT_DELETED, n=2, m=1)
    ctrl._undo_last_edit()
    assert np.array_equal(ctrl._labels_volume, before)
    assert list(ctrl._qc_features_df["label_status"]) == ["active"] * 3
    assert ctrl._labels_dirty
    assert ctrl.ui.qc_status_line.sets[-1] == T.MT_UNDONE.format(
        what=T.UNDO_WHAT_REMOVE)
    assert ctrl._undo_slot is None                  # one level only


def test_undo_never_writes_over_a_newer_id(ctrl, red):
    ctrl._delete_bad_labels()
    ctrl._labels_volume[0, 0, 0] = 9                # painted since
    ctrl._undo_last_edit()
    assert ctrl._labels_volume[0, 0, 0] == 9
    assert (ctrl._labels_volume == 3).sum() == 2


def test_undo_of_smooth_restores_the_array_and_the_tidy_fields(ctrl):
    old = ctrl._labels_volume
    ctrl._pending_edits = {"smoothed_at": "T0", "n_thin_kept": 1,
                           "_stack": "s.tif"}
    smoothed = old.copy()
    smoothed[0, 0, 1] = 1
    result = types.SimpleNamespace(working_volume=smoothed,
                                   emptied_ids=[2, 3], touched_ids=[1],
                                   n_changed_total=1)
    ctrl._on_smooth_finished(result)
    assert ctrl._labels_volume is smoothed
    assert ctrl._pending_edits["n_thin_kept"] == 2
    ctrl._undo_last_edit()
    assert ctrl._labels_volume is old
    assert ctrl._pending_edits["smoothed_at"] == "T0"     # G1-36
    assert ctrl._pending_edits["n_thin_kept"] == 1
    assert ctrl.ui.qc_status_line.sets[-1] == T.MT_UNDONE.format(
        what=T.UNDO_WHAT_SMOOTH)


def test_undo_of_a_first_smooth_drops_the_held_fields(ctrl):
    result = types.SimpleNamespace(working_volume=ctrl._labels_volume.copy(),
                                   emptied_ids=[], touched_ids=[1],
                                   n_changed_total=0)
    ctrl._on_smooth_finished(result)
    ctrl._undo_last_edit()
    assert "smoothed_at" not in ctrl._pending_edits
    assert "n_thin_kept" not in ctrl._pending_edits


def test_a_second_edit_replaces_the_slot(ctrl, red):
    ctrl._delete_bad_labels()
    result = types.SimpleNamespace(working_volume=ctrl._labels_volume.copy(),
                                   emptied_ids=[], touched_ids=[2],
                                   n_changed_total=0)
    ctrl._on_smooth_finished(result)
    assert ctrl._undo_slot["kind"] == "smooth"
    ctrl._undo_last_edit()
    ctrl._undo_last_edit()
    assert ctrl.ui.qc_status_line.sets[-1] == T.MT_UNDO_EMPTY
    assert (ctrl._labels_volume == 1).sum() == 0    # the Remove stays


def test_nothing_to_undo_says_so_and_is_never_grey(ctrl):
    ctrl._undo_last_edit()
    assert ctrl.ui.qc_status_line.sets[-1] == T.MT_UNDO_EMPTY
    ctrl._refresh_mt_panel()
    assert ctrl.ui.undo_btn.isEnabled()


def test_the_maps_are_not_brought_back(ctrl, red):
    class _Plot:
        def __init__(self):
            self.calls = []

        def set_features(self, df, *, preserve_caches=False):
            self.calls.append(preserve_caches)

    plot = _Plot()
    ctrl._analysis_widget = types.SimpleNamespace(qc_plot_widget=plot)
    ctrl._delete_bad_labels()
    ctrl._undo_last_edit()
    assert plot.calls == [False, False]     # the caches stay cleared


@pytest.mark.parametrize("how", ["save", "measure", "reset_flag", "load",
                                 "workspace"])
def test_the_slot_is_emptied(ctrl, red, how, monkeypatch):
    ctrl._delete_bad_labels()
    assert ctrl._undo_slot is not None
    if how == "save":
        ctrl._has_unsaved_changes = False            # what Save does
    elif how == "measure":
        ctrl._measure_launch = {"stack": "s.tif"}
        ctrl._on_extract_finished(pd.DataFrame({"label": [2]}), "s.tif")
    elif how == "reset_flag":
        ctrl._forget_loaded_state()                  # Reset / a load
    elif how == "load":
        ctrl._unload_stack("test")
    else:
        monkeypatch.setattr(ctrl, "_ask_unsaved", lambda s: "discard")
        assert ctrl._settle_unsaved("a workspace change")
    assert ctrl._undo_slot is None


def test_undo_after_another_stack_loaded_is_empty(ctrl, red):
    ctrl._delete_bad_labels()
    ctrl._current_stack_name = "other.tif"
    ctrl._undo_last_edit()
    assert ctrl.ui.qc_status_line.sets[-1] == T.MT_UNDO_EMPTY


def test_undo_is_refused_while_a_job_runs(ctrl, red):
    ctrl._delete_bad_labels()
    ctrl._smooth_running = True
    ctrl._undo_last_edit()
    assert ctrl._undo_slot is not None
    assert any("refused" in l for l in ctrl.ui.log_output.lines)

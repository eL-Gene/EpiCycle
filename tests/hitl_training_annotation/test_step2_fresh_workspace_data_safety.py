"""Step 2 never loses or moves aside state because of a passing lock.

The rules under test, in plain words:

* A run list that could not be OPENED just now (a lock held by a backup, a
  sync tool or a virus scanner right after a write) is not a damaged one:
  that save is refused (or goes through), no copy is set aside, and the next
  save reads the file again.  The session never turns "unreadable" for good.
* New Run while the run list is locked says "could not be written just now;
  try again" and leaves no unlisted run folder behind.
* Leaving a workspace (Browse) saves it; what a refused save could not write
  goes to a rescue file, and that is said.
* A run switch whose pending drawing could not be written is cancelled
  before anything is cleared (the question promised the drawing is saved
  first); the drawing stays pending.
* Train does not start when the save before it is refused.
* The small ones: an old-layout save that raises is a refusal, the once per
  workspace lines are said again after a good save, and a folder's first
  run is always written active.

Qt-free: the controller is built with ``__new__`` and plain fakes; boxes
are recorded, so no test builds a real modal.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from napari_epicycle.hitl_training_annotation import _paths as P
from napari_epicycle.hitl_training_annotation import _run_management as RM
from napari_epicycle.hitl_training_annotation.controller import (
    training_annotation_controller as C,
)
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T

from test_step2_fresh_workspace_first_run import _ctrl, _run_entry, asked  # noqa: F401


def _locked_open(monkeypatch, module, times):
    """``open`` of step2_state.json raises PermissionError ``times`` times."""
    left = {"n": times}
    real = builtins.open

    def _open(path, *a, **k):
        if Path(str(path)).name == "step2_state.json" and left["n"] > 0:
            left["n"] -= 1
            raise PermissionError(13, "locked by another program", str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(module, "open", _open, raising=False)
    return left


def _copies(ws):
    return list(Path(ws).glob("step2_state.json.corrupt-*"))


# ---------------------------------------------------------------------------
# a lock is not a damaged file
# ---------------------------------------------------------------------------

def test_one_lock_never_makes_the_session_unreadable(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    before = (tmp_path / "step2_state.json").read_bytes()
    _locked_open(monkeypatch, C, 1)
    first = ctrl._save_internal_state()
    assert first in ("saved-v3", "refused")
    assert ctrl._step2_write_target() == "run"
    assert _copies(tmp_path) == []
    assert ctrl._save_internal_state() == "saved-v3"
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert "pilot" in state["runs"] and before


def test_a_lock_on_the_save_itself_refuses_only_that_save(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    _locked_open(monkeypatch, C, 2)       # the check and the save's own read
    assert ctrl._save_internal_state() == "refused"
    assert _copies(tmp_path) == []
    assert ctrl._step2_write_target() == "run"
    assert ctrl._save_internal_state() == "saved-v3"
    assert _copies(tmp_path) == []


def test_a_damaged_list_is_still_unreadable(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    assert ctrl._step2_write_target() == "unreadable"


def test_run_list_loader_never_copies_a_locked_file(tmp_path, monkeypatch):
    RM.create_run_on_disk(tmp_path, "cells", display_name="cells")
    RM.add_run_to_state_json(tmp_path, "cells", activate=True)
    _locked_open(monkeypatch, RM, 1)
    with pytest.raises(OSError):
        RM.add_run_to_state_json(tmp_path, "nuclei")
    assert _copies(tmp_path) == []
    RM.add_run_to_state_json(tmp_path, "nuclei")
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert set(state["runs"]) == {"cells", "nuclei"}


def test_new_run_on_a_locked_list_says_try_again_and_leaves_no_folder(
        tmp_path, monkeypatch, asked):
    ctrl = _ctrl(tmp_path, run="cells")
    _locked_open(monkeypatch, RM, 1)
    with pytest.raises(RuntimeError) as err:
        ctrl._create_new_run("nuclei", "Nuclei")
    assert err.value.sentence == T.NEW_RUN_LIST_LOCKED
    assert not (tmp_path / "artifacts" / "step2" / "runs" / "nuclei").exists()
    ctrl._say_new_run_failed("nuclei", err.value)
    assert ctrl.told[-1] == (T.NEW_RUN_FAILED_TITLE, T.NEW_RUN_LIST_LOCKED)
    assert "PermissionError" in "\n".join(ctrl.ui.training_log_output.lines)
    assert _copies(tmp_path) == []
    ctrl._create_new_run("nuclei", "Nuclei")          # the retry works
    assert ctrl.active_run_id == "nuclei"


# ---------------------------------------------------------------------------
# leaving a workspace
# ---------------------------------------------------------------------------

def _refuse_v3_saves(monkeypatch):
    from napari_epicycle.hitl_training_annotation import _migrate_step2_layout

    def _boom(*_a, **_k):
        raise OSError("the disk went away")

    monkeypatch.setattr(_migrate_step2_layout, "wrap_state_schema_v2_to_v3",
                        _boom)


def _rescues(ws):
    return list((Path(ws) / "artifacts" / "step2" / "rescue").glob("*.json"))


def test_leaving_a_workspace_whose_save_is_refused_rescues_it(
        tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl.sample_records = {"stack01_XZ_12": {"status": "curating"}}
    _refuse_v3_saves(monkeypatch)
    ctrl._teardown_previous_workspace()
    rescues = _rescues(tmp_path)
    assert len(rescues) == 1
    assert "stack01_XZ_12" in rescues[0].read_text("utf-8")
    assert [title for title, _s in ctrl.told] == [T.STATE_NOT_SAVED_TITLE]
    assert str(rescues[0]) in ctrl.told[0][1]


def test_leaving_a_workspace_saves_it(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl.sample_records = {"stack01_XZ_12": {"status": "curating"}}
    ctrl._teardown_previous_workspace()
    _state, entry = _run_entry(tmp_path, "pilot")
    assert "stack01_XZ_12" in entry["sample_records"]
    assert _rescues(tmp_path) == []
    assert ctrl.told == []


# ---------------------------------------------------------------------------
# a run switch and the drawing it promised to save first
# ---------------------------------------------------------------------------

def _two_runs(ws):
    ctrl = _ctrl(ws, run="cells")
    RM.create_run_on_disk(ws, "nuclei", display_name="nuclei")
    RM.add_run_to_state_json(ws, "nuclei")
    return ctrl


def test_a_refused_drawing_cancels_the_switch(tmp_path, asked):
    ctrl = _two_runs(tmp_path)
    ctrl._flush_curating_persist_pending = lambda: False
    ctrl._last_pair_refusal = "folder"
    ctrl._switch_active_run("nuclei")
    assert ctrl.active_run_id == "cells"
    assert "clear" not in ctrl.calls and "close" not in ctrl.calls
    assert ctrl.loaded_patch_name == "stack01"
    assert len(ctrl.told) == 1
    title, sentence = ctrl.told[0]
    assert title == T.RUN_SWITCH_FAILED_TITLE
    assert sentence == T.RUN_SWITCH_DRAWING_FAILED.format(
        error=T.DRAWING_CAUSE_FOLDER)
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert state["active_run_id"] == "cells"


def test_a_saved_drawing_lets_the_switch_go(tmp_path, asked):
    ctrl = _two_runs(tmp_path)
    ctrl._flush_curating_persist_pending = lambda: True
    ctrl._switch_active_run("nuclei")
    assert ctrl.active_run_id == "nuclei"


def test_the_first_run_is_not_stopped_by_a_drawing_with_no_run(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    flushes = []
    ctrl._flush_curating_persist_pending = lambda: flushes.append(1) or False
    ctrl._last_pair_refusal = "no-run"
    ctrl._create_new_run("pilot", "Pilot")
    assert ctrl.active_run_id == "pilot"
    assert len(flushes) == 2, "flushed before, and again into the new run"


class _LabelsLayer:
    def __init__(self, data):
        self.data = data


def _pending_ctrl(tmp_path, written):
    import numpy as np
    ctrl = _ctrl(tmp_path, run="cells")
    ctrl.segmentation_manager = type("SM", (), {"data_masks": {}})()
    ctrl._curating_persist_pending = {
        "stack01_XY_3": ("stack01_XY_3", "XY", 3)}
    ctrl._work_labels_layer_name_for_view = lambda **_k: "work"
    ctrl._layer_by_name = lambda _n: _LabelsLayer(np.zeros((5, 4, 4), np.int32))
    ctrl._say_pair_refused = lambda *a, **k: None
    ctrl._refresh_run_picker = lambda: None

    def _persist(sl, async_write=False):
        if written:
            return {"img": "x"}
        ctrl._last_pair_refusal = "folder"
        return {}

    ctrl._persist_curated_training_pair = _persist
    return ctrl


def test_a_refused_drawing_stays_pending(tmp_path):
    ctrl = _pending_ctrl(tmp_path, written=False)
    assert ctrl._flush_curating_persist_pending() is False
    assert "stack01_XY_3" in ctrl._curating_persist_pending


def test_a_written_drawing_leaves_the_queue(tmp_path):
    ctrl = _pending_ctrl(tmp_path, written=True)
    assert ctrl._flush_curating_persist_pending() is True
    assert ctrl._curating_persist_pending == {}


def test_the_timers_flush_reports_a_refused_drawing(tmp_path):
    ctrl = _pending_ctrl(tmp_path, written=False)
    assert ctrl._flush_pending_state_timers() is False


# ---------------------------------------------------------------------------
# Train and the save before it
# ---------------------------------------------------------------------------

def _train_ctrl(tmp_path, save_result):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl._collect_training_config = lambda: {"model_name": "m", "epochs": 1}
    ctrl._training_thread = None
    ctrl._save_internal_state = lambda: save_result
    ctrl._last_save_error = "OSError: the disk went away"
    ctrl.went_on = []
    ctrl._sync_data_to_add_list = lambda: ctrl.went_on.append(1)
    return ctrl


def test_train_does_not_start_when_its_save_is_refused(tmp_path):
    ctrl = _train_ctrl(tmp_path, "refused")
    ctrl._train_model()
    assert ctrl.went_on == [], "nothing after the save ran"
    assert ctrl.told == [(T.TRAIN_SAVE_REFUSED_TITLE, T.TRAIN_SAVE_REFUSED.format(
        error="OSError: the disk went away"))]


def test_train_goes_on_after_a_good_save(tmp_path):
    ctrl = _train_ctrl(tmp_path, "saved-v3")
    try:
        ctrl._train_model()
    except Exception:      # the round itself needs more of the UI
        pass
    assert ctrl.went_on == [1]


def test_a_refused_save_after_training_reaches_the_close_rescue(
        tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    _refuse_v3_saves(monkeypatch)
    ctrl.current_round = 3                     # a finished round, in memory
    assert ctrl._save_internal_state() == "refused"
    ctrl.shutdown()
    rescues = _rescues(tmp_path)
    assert len(rescues) == 1
    assert json.loads(rescues[0].read_text("utf-8"))["current_round"] == 3


# ---------------------------------------------------------------------------
# the small ones
# ---------------------------------------------------------------------------

def test_an_old_layout_save_that_raises_is_a_refusal(tmp_path):
    (tmp_path / "training_data").mkdir()
    ctrl = _ctrl(tmp_path)
    ctrl.active_run_id = "default"

    class _Boom:
        def save_hitl_state(self, **_k):
            raise OSError("locked")

    ctrl.persistence = _Boom()
    ctrl.sample_records = {"stack01_XY_1": {"status": "curating"}}
    assert ctrl._save_internal_state() == "refused"
    assert ctrl._refused_state["sample_records"]


def test_the_once_lines_are_said_again_after_a_good_save(tmp_path, capsys):
    ctrl = _ctrl(tmp_path)
    ctrl._debounced_state_save()
    assert "no run yet: nothing saved" in capsys.readouterr().out
    RM.create_run_on_disk(tmp_path, "pilot", display_name="pilot")
    RM.add_run_to_state_json(tmp_path, "pilot", activate=True)
    ctrl.active_run_id = "pilot"
    ctrl._debounced_state_save()
    ctrl.active_run_id = ""
    ctrl._debounced_state_save()
    assert "no run yet: nothing saved" in capsys.readouterr().out


def test_a_folders_first_run_is_always_written_active(tmp_path):
    RM.create_run_on_disk(tmp_path, "pilot", display_name="pilot")
    RM.add_run_to_state_json(tmp_path, "pilot", activate=False)
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert state["active_run_id"] == "pilot"
    RM.create_run_on_disk(tmp_path, "nuclei", display_name="nuclei")
    RM.add_run_to_state_json(tmp_path, "nuclei", activate=False)
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert state["active_run_id"] == "pilot"


# ---------------------------------------------------------------------------
# a lock at load: its own state, every write refused, read again next time
# ---------------------------------------------------------------------------

class _Lock:
    """``open`` of step2_state.json raises PermissionError while ``on``."""

    def __init__(self, monkeypatch, module):
        self.on = True
        real = builtins.open

        def _open(path, *a, **k):
            if self.on and Path(str(path)).name == "step2_state.json":
                raise PermissionError(13, "held by another program", str(path))
            return real(path, *a, **k)

        monkeypatch.setattr(module, "open", _open, raising=False)


def _locked_at_load(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl.active_run_id = ""
    lock = _Lock(monkeypatch, C)
    ctrl._load_internal_state()
    return ctrl, lock


def test_a_lock_at_load_is_its_own_state(tmp_path, monkeypatch, capsys):
    ctrl, lock = _locked_at_load(tmp_path, monkeypatch)
    before = (tmp_path / "step2_state.json").read_bytes()
    assert ctrl.active_run_id == ""
    assert ctrl._step2_write_target() == "locked"
    assert ctrl._save_internal_state() == "refused"
    assert ctrl._last_save_error == T.RUN_LIST_LOCKED
    for _ in range(3):
        ctrl._step2_write_target()
    out = capsys.readouterr().out
    assert out.count(T.RUN_LIST_LOCKED) == 1, "said once"
    assert "\n".join(ctrl.ui.training_log_output.lines).count(
        T.RUN_LIST_LOCKED) == 1
    assert _copies(tmp_path) == []
    assert (tmp_path / "step2_state.json").read_bytes() == before
    assert not (tmp_path / "hitl_state.json").exists()


def test_the_next_check_after_the_lock_finds_the_runs(tmp_path, monkeypatch):
    ctrl, lock = _locked_at_load(tmp_path, monkeypatch)
    ctrl.sample_records = {"stack01_XY_4": {"status": "curating"}}
    lock.on = False
    assert ctrl._step2_write_target() == "run"
    assert ctrl.active_run_id == "pilot"
    assert [s["run_id"] for s in ctrl._list_runs_for_ui()] == ["pilot"]
    assert "stack01_XY_4" in ctrl.sample_records, "kept across the reload"
    assert ctrl._save_internal_state() == "saved-v3"


def test_an_action_while_locked_is_told_not_offered_new_run(
        tmp_path, monkeypatch):
    ctrl, _lock = _locked_at_load(tmp_path, monkeypatch)
    asked = []
    ctrl._ask_for_run = lambda *a: asked.append(a) or False
    assert ctrl._require_run("Predict") is False
    assert asked == [("Predict", T.RUN_LIST_LOCKED, "locked")]
    assert ctrl._layout_allows_new_run() is False


def test_new_run_while_locked_is_refused_before_any_folder(
        tmp_path, monkeypatch):
    ctrl, _lock = _locked_at_load(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError) as err:
        ctrl._create_new_run("nuclei", "Nuclei")
    assert err.value.sentence == T.RUN_LIST_LOCKED
    assert not (tmp_path / "artifacts" / "step2" / "runs" / "nuclei").exists()


def test_the_opened_just_now_line_is_said_once_until_it_opens(
        tmp_path, monkeypatch, capsys):
    ctrl = _ctrl(tmp_path, run="pilot")
    lock = _Lock(monkeypatch, C)
    for _ in range(3):
        assert ctrl._step2_write_target() == "run"
    assert capsys.readouterr().out.count("could not be opened just now") == 1
    lock.on = False
    ctrl._step2_write_target()                  # it opens again
    lock.on = True
    ctrl._step2_state_parse_cache = None        # so the file is opened again
    ctrl._step2_write_target()
    assert capsys.readouterr().out.count("could not be opened just now") == 1



def test_new_run_on_a_fresh_folder_whose_list_write_fails_leaves_it_fresh(
        tmp_path, monkeypatch, asked):
    ctrl = _ctrl(tmp_path)
    real = RM._atomic_write_json
    left = {"n": 1}

    def _write(path, data):
        if Path(path).name == "step2_state.json" and left["n"]:
            left["n"] -= 1
            raise PermissionError(13, "held by another program", str(path))
        return real(path, data)

    monkeypatch.setattr(RM, "_atomic_write_json", _write)
    with pytest.raises(RuntimeError):
        ctrl._create_new_run("pilot", "Pilot")
    assert list(tmp_path.iterdir()) == [], "the disk as before the click"
    assert ctrl._step2_write_target() is None
    ctrl._create_new_run("pilot", "Pilot")          # the retry works
    assert ctrl.active_run_id == "pilot"


def test_leaving_a_mixed_folder_untouched_writes_nothing_and_says_nothing(
        tmp_path):
    (tmp_path / "training_data").mkdir()
    (tmp_path / "artifacts" / "step2" / "runs" / "x").mkdir(parents=True)
    ctrl = _ctrl(tmp_path)
    ctrl.sample_records = {}
    ctrl._teardown_previous_workspace()
    assert _rescues(tmp_path) == []
    assert ctrl.told == []


def test_leaving_a_fresh_folder_writes_nothing(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl._teardown_previous_workspace()
    assert list(tmp_path.iterdir()) == []
    assert ctrl.told == []



# ---------------------------------------------------------------------------
# a lock during a run switch: the switch is cancelled or undone, never a
# "no active run" that later copies one run's rows into another
# ---------------------------------------------------------------------------

def test_a_lock_right_after_new_run_never_clears_the_screen(
        tmp_path, monkeypatch, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.sample_records = {"stack01_XY_4": {"status": "curating"}}
    lock = _Lock(monkeypatch, C)          # the run list helpers still write
    ctrl._create_new_run("pilot", "Pilot")
    assert ctrl.active_run_id == ""
    assert ctrl.calls == [], "nothing closed, cleared or reloaded"
    assert "stack01_XY_4" in ctrl.sample_records
    assert ctrl.told == [(T.RUN_SWITCH_FAILED_TITLE, T.RUN_LIST_LOCKED)]
    lock.on = False
    ctrl._switch_active_run("pilot")      # picked later: still the new run
    assert ctrl.active_run_id == "pilot"
    assert ctrl.left_no_run == [], "it is the run made for these slices"
    _state, entry = _run_entry(tmp_path, "pilot")
    assert "stack01_XY_4" in entry["sample_records"]


def test_a_lock_on_the_switch_reload_puts_the_old_run_back(
        tmp_path, monkeypatch, asked):
    ctrl = _two_runs(tmp_path)
    ctrl.sample_records = {"stack01_XY_4": {"status": "accepted",
                                            "img_uri": "runs/cells/x_img.tif"}}
    assert ctrl._save_internal_state() == "saved-v3"
    lock = _Lock(monkeypatch, C)
    lock.on = False

    def _clear():
        ctrl.calls.append("clear")
        if ctrl.calls.count("clear") == 1:
            lock.on = True                 # the lock arrives mid-switch, once

    ctrl._clear_patch_viewer_layers_safely = _clear
    with pytest.raises(RuntimeError) as err:
        ctrl._switch_active_run("nuclei")
    assert err.value.sentence == T.RUN_LIST_LOCKED
    assert ctrl.active_run_id == "cells"
    assert ctrl.loaded_patch_name == "stack01"
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert state["active_run_id"] == "cells"
    lock.on = False
    ctrl._switch_active_run("nuclei")
    assert ctrl.active_run_id == "nuclei"
    assert "stack01_XY_4" not in ctrl.sample_records
    _state, entry = _run_entry(tmp_path, "nuclei")
    assert "stack01_XY_4" not in json.dumps(entry), "no cross-run rows"


def test_a_lock_before_a_switch_cancels_it(tmp_path, monkeypatch, asked):
    ctrl = _two_runs(tmp_path)
    _Lock(monkeypatch, C)
    ctrl._switch_active_run("nuclei")
    assert ctrl.active_run_id == "cells"
    assert ctrl.calls == []
    assert ctrl.told == [(T.RUN_SWITCH_FAILED_TITLE, T.RUN_LIST_LOCKED)]


# ---------------------------------------------------------------------------
# a drawing whose cause does not go away: "Switch anyway"
# ---------------------------------------------------------------------------

def test_switch_anyway_drops_the_drawing_and_switches(tmp_path, asked):
    ctrl = _two_runs(tmp_path)
    ctrl._curating_persist_pending = {"stack01_XY_3": ("stack01_XY_3", "XY", 3)}
    ctrl._flush_curating_persist_pending = lambda: False
    ctrl._last_pair_refusal = "no-run"
    offered = []
    ctrl._ask_switch_anyway = lambda sentence: offered.append(sentence) or True
    ctrl._switch_active_run("nuclei")
    assert offered == [T.RUN_SWITCH_DRAWING_FAILED.format(
        error=T.DRAWING_CAUSE_NO_RUN)]
    assert ctrl.active_run_id == "nuclei"
    assert ctrl._curating_persist_pending == {}


def test_switch_anyway_is_not_the_default(tmp_path, monkeypatch):
    from napari_epicycle.ui_shell import theme
    seen = []
    monkeypatch.setattr(theme, "themed_confirm",
                        lambda *a, **k: seen.append(k) or False)
    ctrl = _ctrl(tmp_path, run="cells")
    ctrl._can_show_modal = lambda: True
    assert ctrl._ask_switch_anyway("x") is False
    assert seen[0]["default"] == "cancel"
    assert seen[0]["confirm_text"] == T.RUN_SWITCH_ANYWAY


# ---------------------------------------------------------------------------
# Train, the first run's writes, leaving with a drawing
# ---------------------------------------------------------------------------

def test_a_pre_train_save_that_raises_does_not_start_training(tmp_path):
    ctrl = _train_ctrl(tmp_path, "saved-v3")

    def _boom():
        raise OSError("the disk went away")

    ctrl._save_internal_state = _boom
    ctrl._train_model()
    assert ctrl.went_on == []
    assert [t for t, _s in ctrl.told] == [T.TRAIN_SAVE_REFUSED_TITLE]


def test_a_dialog_model_does_not_take_the_widgets_checkpoint(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.current_config["starting_checkpoint"] = "C:/elsewhere/round_3.pth"
    ctrl._create_new_run(
        "pilot", "Pilot",
        init_spec={"mode": "pretrained", "pretrained_name": "nuclei"})
    _state, entry = _run_entry(tmp_path, "pilot")
    assert not entry["current_config"].get("starting_checkpoint")


def test_without_a_dialog_model_the_widgets_checkpoint_comes_along(
        tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.curation_model_source_mode = "auto"
    ctrl.curation_pretrained_model_type = "cyto3"
    ctrl.current_config["starting_checkpoint"] = "C:/elsewhere/round_3.pth"
    ctrl._create_new_run("pilot", "Pilot")
    _state, entry = _run_entry(tmp_path, "pilot")
    assert entry["current_config"]["starting_checkpoint"].endswith("round_3.pth")


def test_the_first_run_is_written_active_once(tmp_path, monkeypatch, asked):
    calls = []
    real = RM.set_active_run_in_state_json
    monkeypatch.setattr(RM, "set_active_run_in_state_json",
                        lambda ws, slug: calls.append(slug) or real(ws, slug))
    ctrl = _ctrl(tmp_path)
    ctrl._create_new_run("pilot", "Pilot")
    assert calls == []
    ctrl._create_new_run("nuclei", "Nuclei")
    assert calls == ["nuclei"]


def test_leaving_with_a_drawing_that_could_not_be_saved_says_so(
        tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl._curating_persist_pending = {"stack01_XY_3": ("stack01_XY_3", "XY", 3)}
    ctrl._flush_curating_persist_pending = lambda: False
    _refuse_v3_saves(monkeypatch)
    ctrl._teardown_previous_workspace()
    assert ctrl._curating_persist_pending == {}
    assert T.STATE_LEFT_DRAWING_LOST in ctrl.told[0][1]



# ---------------------------------------------------------------------------
# the last corners: the reload's second read, the first run's undo, a
# background write that keeps failing, the switch's own write
# ---------------------------------------------------------------------------

def test_a_lock_on_the_reloads_second_read_puts_the_old_run_back(
        tmp_path, monkeypatch, asked):
    ctrl = _two_runs(tmp_path)
    ctrl.sample_records = {"stack01_XY_4": {"status": "accepted"}}
    assert ctrl._save_internal_state() == "saved-v3"
    lock = _Lock(monkeypatch, P)           # the run's records, not the peek
    lock.on = False

    def _clear():
        ctrl.calls.append("clear")
        if ctrl.calls.count("clear") == 1:
            lock.on = True

    ctrl._clear_patch_viewer_layers_safely = _clear
    with pytest.raises(RuntimeError) as err:
        ctrl._switch_active_run("nuclei")
    assert err.value.sentence == T.RUN_LIST_LOCKED
    assert ctrl.active_run_id == "cells"
    assert "stack01_XY_4" in ctrl.sample_records
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert state["active_run_id"] == "cells"
    lock.on = False
    ctrl._switch_active_run("nuclei")
    assert "stack01_XY_4" not in ctrl.sample_records
    assert ctrl._save_internal_state() == "saved-v3"
    _state, entry = _run_entry(tmp_path, "nuclei")
    assert "stack01_XY_4" not in json.dumps(entry), "no cross-run rows"


def test_a_lock_on_the_second_read_at_load_loads_no_run(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl.active_run_id = ""
    _Lock(monkeypatch, P)                  # the lock starts after the peek
    ctrl._load_internal_state()
    assert ctrl.active_run_id == ""
    _Lock(monkeypatch, C)                  # and is still there at the check
    assert ctrl._step2_write_target() == "locked"


def test_a_lock_after_the_first_runs_carry_keeps_the_screen_and_the_mark(
        tmp_path, monkeypatch, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.sample_records = {"stack01_XY_4": {"status": "curating"}}
    lock = _Lock(monkeypatch, C)
    lock.on = False
    real_carry = ctrl._carry_into_first_run

    armed = []

    def _carry(slug, settings):
        real_carry(slug, settings)
        if not armed:
            armed.append(1)
            lock.on = True                 # once, right after the carry

    ctrl._carry_into_first_run = _carry
    with pytest.raises(RuntimeError):
        ctrl._create_new_run("pilot", "Pilot")
    assert ctrl.calls == [], "nothing cleared, so nothing reloaded"
    assert ctrl._run_just_made == "pilot"
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert ctrl.active_run_id == state["active_run_id"] == ""
    assert "stack01_XY_4" in ctrl.sample_records
    lock.on = False
    ctrl._switch_active_run("pilot")
    assert ctrl.left_no_run == [], "picked later, it still takes the screen"
    assert ctrl.active_run_id == "pilot"
    _state, entry = _run_entry(tmp_path, "pilot")
    assert "stack01_XY_4" in entry["sample_records"]


def test_a_lock_on_the_switchs_own_write_is_undone(tmp_path, monkeypatch, asked):
    ctrl = _two_runs(tmp_path)

    def _locked(ws, slug):
        raise PermissionError(13, "held by another program")

    monkeypatch.setattr(RM, "set_active_run_in_state_json", _locked)
    with pytest.raises(RuntimeError) as err:
        ctrl._switch_active_run("nuclei")
    assert err.value.sentence == T.RUN_LIST_LOCKED
    assert ctrl.active_run_id == "cells"
    assert ctrl.loaded_patch_name == "stack01"
    assert "reload-patch" in ctrl.calls, "the cleared patch comes back"


def test_a_patch_alone_switches_without_the_no_run_question(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.sample_records = {}
    for slug in ("cells",):
        RM.create_run_on_disk(tmp_path, slug, display_name=slug)
        RM.add_run_to_state_json(tmp_path, slug, activate=False)
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    state["active_run_id"] = "gone"
    (tmp_path / "step2_state.json").write_text(json.dumps(state), "utf-8")
    ctrl._switch_active_run("cells")
    assert ctrl.left_no_run == []
    assert ctrl.active_run_id == "cells"


def test_the_no_run_question_without_a_screen_is_logged(tmp_path):
    ctrl = _ctrl(tmp_path)
    del ctrl._confirm_leave_no_run
    ctrl._can_show_modal = lambda: False
    assert ctrl._confirm_leave_no_run("cells") is False
    assert T.RUN_SWITCH_NO_RUN_UNASKED.format(name="cells") in "\n".join(
        ctrl.ui.training_log_output.lines)


def test_train_stops_on_a_drawing_that_could_not_be_saved(tmp_path):
    ctrl = _train_ctrl(tmp_path, "saved-v3")
    ctrl._flush_pending_state_timers = lambda: False
    ctrl._last_pair_refusal = "folder"
    ctrl._train_model()
    assert ctrl.went_on == []
    assert ctrl.told == [(T.TRAIN_SAVE_REFUSED_TITLE,
                          T.TRAIN_DRAWING_NOT_SAVED.format(
                              error=T.DRAWING_CAUSE_FOLDER))]

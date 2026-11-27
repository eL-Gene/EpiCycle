"""Step 2 on a folder with no run: every action that needs a run asks for one.

The rule under test, in plain words:

* An action that would write a Step 2 file (Predict, the empty mask, both
  Accepts, Finalize accepted, Train, opening the 2D annotator, the two
  imports, the training history) does nothing on a folder with no run.  It
  prints ``[STEP2-RUN] <action>: no run yet``, writes the state's sentence to
  the log, and shows a box that offers New Run -- never for a run list Step 2
  cannot read.  After New Run the user presses the action again.
* The sentence says which state the folder is in: no workspace, no run yet,
  runs but none active, or a run list Step 2 cannot read.
* New Run from the box is the New Run button's own handler and gate; a New
  Run that fails says why (log line and box), it is never swallowed.
* With no run: only New Run is pressable in the run row, the Run summary says
  "No run yet", the shell header names no run, and the New Run dialog has no
  "Create only".
* A slice whose pair could not be written says so in one box per cause,
  naming the button to press again.

Qt-free: the controller is built with ``__new__`` and plain fakes; the two
box seams (``_ask_for_run``, ``_tell_step2``) are replaced by recorders, so no
test builds a real modal.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_epicycle.hitl_training_annotation import _paths as P
from napari_epicycle.hitl_training_annotation import _run_management as RM
from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)
from napari_epicycle.hitl_training_annotation.controller.training_manager import (
    TrainingManager,
)
from napari_epicycle.hitl_training_annotation.ui import _step2_lists as L
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T
from napari_epicycle.io_registry import HITLPersistenceService
from napari_epicycle.ui_shell import _workspace_facts as wf


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _Item:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _ListWidget:
    def __init__(self, items=None):
        self._items = [_Item(x) for x in (items or [])]

    def count(self):
        return len(self._items)

    def item(self, i):
        return self._items[i]

    def addItem(self, text):
        self._items.append(_Item(text))

    def clear(self):
        self._items = []


class _Label:
    def __init__(self, text=""):
        self.value = text

    def setText(self, text):
        self.value = text

    def text(self):
        return self.value


class _LogOutput:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


class _Button:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, v):
        self.enabled = bool(v)

    def isEnabled(self):
        return bool(self.enabled)


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
        self.enabled = bool(v)


class _ProgressBar:
    def setRange(self, *_a):
        pass

    def setValue(self, _v):
        pass


class _SegManager:
    def __init__(self):
        self.data_inputs = {}
        self.data_masks = {}
        self.data_names = {}
        self.data_inputs_rgb = {}
        self.segmenter = object()

    def get_input_stack(self, label):
        return self.data_inputs.get(label)

    def get_masks(self, label):
        return self.data_masks.get(label)


class _Layer:
    def __init__(self, data):
        self.data = data

    def refresh(self):
        pass


class _Layers(list):
    def __contains__(self, name):
        return False


class _UI:
    def __init__(self):
        self.training_log_output = _LogOutput()
        self.status_run_value = _Label()
        self.run_picker_combo = _Combo()
        self.new_run_button = _Button()
        self.import_pairs_button = _Button()
        self.import_external_button = _Button()
        self.manage_runs_button = _Button()
        self.data_to_add_list = _ListWidget()
        self.segment_slice_data_list = _ListWidget()
        self.accepted_queue_list = _ListWidget()
        self.staged_queue_list = _ListWidget()
        self.sample_queue_list = self.accepted_queue_list
        self.training_progress_bar = _ProgressBar()


def _noop(*_a, **_k):
    return None


_QUIET = (
    "_sync_data_to_add_list", "_refresh_sample_queue", "_refresh_round_status",
    "_refresh_annotator_edit_state_visuals", "_schedule_state_save",
    "_begin_busy", "_end_busy", "_exit_refine_mode",
    "_refresh_patch_list_badges",
)


def _ctrl(ws, *, run=None, layout=None):
    """A controller on folder ``ws`` with both box seams recorded.

    ``run``: make that run on disk, activate it and point the controller at it.
    ``layout="legacy"``: give the folder an old-layout ``training_data/``.
    ``ws=""``: no workspace.
    """
    if ws and layout == "legacy":
        (Path(ws) / "training_data").mkdir(parents=True, exist_ok=True)
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.persistence = HITLPersistenceService()
    ctrl.training_manager = TrainingManager()
    ctrl.segmentation_manager = _SegManager()
    ctrl.working_dir = str(ws) if ws else ""
    ctrl.current_round = 0
    ctrl.current_config = {"model_name": "epi", "model_type": "cyto3"}
    ctrl.training_data_by_round = {}
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.round_backend_info = {}
    ctrl.loaded_patch_name = "stack01"
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"
    ctrl.axial_plane = "XZ"
    ctrl.sample_records = {}
    ctrl.training_mode = "cumulative"
    ctrl.active_curation_checkpoints = {"lateral": "", "axial": ""}
    ctrl.curation_model_source_mode = "auto"
    ctrl.imported_curation_model_path = ""
    ctrl.curation_pretrained_model_type = "cyto3"
    ctrl._use_sync_training = True
    ctrl.active_run_id = ""
    ctrl.ui = _UI()
    ctrl.viewer = SimpleNamespace(layers=_Layers())
    for name in _QUIET:
        setattr(ctrl, name, _noop)
    ctrl.asked = []
    ctrl.told = []
    ctrl.answer = False

    def _ask(action, sentence, state):
        ctrl.asked.append((action, sentence, state))
        return ctrl.answer

    ctrl._ask_for_run = _ask
    ctrl._tell_step2 = lambda title, sentence: ctrl.told.append((title, sentence))
    if run:
        RM.create_run_on_disk(ws, run, display_name=run)
        RM.add_run_to_state_json(ws, run, activate=True)
        ctrl.active_run_id = run
    return ctrl


def _files_outside_thumbs(ws):
    ws = Path(ws)
    thumbs = ws / "artifacts" / "step2" / "thumbs"
    return sorted(
        str(p.relative_to(ws)) for p in ws.rglob("*")
        if p.is_file() and thumbs not in p.parents
    )


def _log(ctrl):
    return "\n".join(ctrl.ui.training_log_output.lines)


def _raise(*_a, **_k):
    raise OSError("disk went away")


def _two_runs_none_active(ws):
    """Runs exist on disk, but the run list names a run with no folder."""
    for slug in ("cells", "nuclei"):
        RM.create_run_on_disk(ws, slug, display_name=slug)
        RM.add_run_to_state_json(ws, slug, activate=False)
    state = json.loads((Path(ws) / "step2_state.json").read_text("utf-8"))
    state["active_run_id"] = "gone"
    (Path(ws) / "step2_state.json").write_text(json.dumps(state), "utf-8")


# ---------------------------------------------------------------------------
# the sentence for each state
# ---------------------------------------------------------------------------

def test_the_no_run_sentence_is_the_banner_sentence():
    assert T.MSG_STEP2_NO_RUN == wf.MSG_STEP2_NO_RUN


def test_refusal_sentence_no_workspace(tmp_path):
    ctrl = _ctrl("")
    assert ctrl._no_run_state() == "no-workspace"
    assert ctrl._no_run_sentence() == T.RUN_NEED_WORKSPACE == "Pick a workspace first."


def test_refusal_sentence_no_runs(tmp_path):
    ctrl = _ctrl(tmp_path)
    assert ctrl._no_run_state() == "no-runs"
    assert ctrl._no_run_sentence() == T.MSG_STEP2_NO_RUN


def test_refusal_sentence_runs_but_none_active(tmp_path):
    _two_runs_none_active(tmp_path)
    ctrl = _ctrl(tmp_path)
    assert ctrl._step2_write_target() is None
    assert ctrl._no_run_state() == "none-active"
    assert ctrl._no_run_sentence() == T.RUN_NONE_ACTIVE
    assert "pick a run in the Run picker, or click New Run" in T.RUN_NONE_ACTIVE


@pytest.mark.parametrize("damage", ["corrupt", "runs-without-list"])
def test_refusal_sentence_unreadable(tmp_path, damage):
    if damage == "corrupt":
        (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    else:
        (tmp_path / "artifacts" / "step2" / "runs" / "x").mkdir(parents=True)
    ctrl = _ctrl(tmp_path)
    assert ctrl._no_run_state() == "unreadable"
    want = (T.RUN_LIST_UNREADABLE if damage == "corrupt"
            else T.RUN_LIST_MISSING)
    assert ctrl._no_run_sentence() == want
    assert "New Run" not in want


# ---------------------------------------------------------------------------
# every gated action refuses and offers
# ---------------------------------------------------------------------------

def _call_predict(c):
    c._run_segmentation()


def _call_empty_mask(c):
    c._create_empty_mask_seed()


def _call_accept(c):
    c._parse_all_selected_sample_ids = lambda: ["stack01_XY_12"]
    c._accept_annotation()


def _call_annotator_accept(c):
    result = c._accept_native_annotation("XY", 12)
    assert result["ok"] is False
    assert result["message"] == c._no_run_sentence()


def _call_stage(c):
    c.sample_records = {"stack01_XY_12": {"status": "accepted",
                                          "bucket": "lateral", "view": "XY"}}
    c._stage_accepted_in_current_view()
    assert c.sample_records["stack01_XY_12"]["status"] == "accepted"


def _call_train(c):
    c._train_model()


def _call_open_annotator(c):
    c._launch_multiview_annotator()


def _call_import_pairs(c):
    c._on_import_pairs_clicked()


def _call_import_external(c):
    c._on_import_external_clicked()


def _call_history(c):
    c._open_training_history_dialog()


GATED = [
    (_call_predict, "Predict"),
    (_call_empty_mask, "Empty mask"),
    (_call_accept, T.CUR_BTN_ACCEPT),
    (_call_annotator_accept, T.CUR_BTN_ACCEPT),
    (_call_stage, T.CUR_BTN_FINALIZE),
    (_call_train, T.TRAIN_BTN_TEXT),
    (_call_open_annotator, T.CUR_BTN_OPEN),
    (_call_import_pairs, T.CUR_IMPORT_PAIRS_ACTION),
    (_call_import_external, T.CUR_IMPORT_EXTERNAL_ACTION),
    (_call_history, "History"),
]


@pytest.mark.parametrize("call,word", GATED, ids=[c.__name__[6:] for c, _ in GATED])
def test_action_without_run_refuses(tmp_path, capsys, call, word):
    ctrl = _ctrl(tmp_path)
    call(ctrl)
    assert ctrl.asked == [(word, T.MSG_STEP2_NO_RUN, "no-runs")]
    assert f"[STEP2-RUN] {word}: no run yet" in capsys.readouterr().out
    assert T.MSG_STEP2_NO_RUN in _log(ctrl)
    assert _files_outside_thumbs(tmp_path) == []


@pytest.mark.parametrize("call,word", GATED, ids=[c.__name__[6:] for c, _ in GATED])
def test_action_with_unreadable_run_list_never_offers_new_run(
        tmp_path, call, word):
    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    ctrl.answer = True                       # even a "yes" makes no run
    called = []
    ctrl._on_new_run_button_clicked = lambda: called.append(True)
    call(ctrl)
    assert ctrl.asked == [(word, T.RUN_LIST_UNREADABLE, "unreadable")]
    assert called == []
    assert (tmp_path / "step2_state.json").read_text("utf-8") == "{oops"


def test_action_with_a_run_is_not_asked(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    assert ctrl._require_run("Predict") is True
    assert ctrl.asked == []


def test_action_on_an_old_layout_folder_is_not_asked(tmp_path):
    ctrl = _ctrl(tmp_path, layout="legacy")
    ctrl.active_run_id = "default"
    assert ctrl._require_run("Predict") is True
    assert ctrl.asked == []


def test_runs_but_none_active_asks_to_pick_or_make(tmp_path):
    _two_runs_none_active(tmp_path)
    ctrl = _ctrl(tmp_path)
    assert ctrl._require_run("Train") is False
    assert ctrl.asked == [("Train", T.RUN_NONE_ACTIVE, "none-active")]


# ---------------------------------------------------------------------------
# the box: New Run goes through the button's path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer", [True, False])
def test_box_new_run_uses_the_button_path(tmp_path, answer):
    ctrl = _ctrl(tmp_path)
    ctrl.answer = answer
    called = []
    ctrl._on_new_run_button_clicked = lambda: called.append(True)
    assert ctrl._require_run("Predict") is False      # press it again after
    assert called == ([True] if answer else [])


def test_default_box_offers_new_run_and_not_now(tmp_path, monkeypatch):
    """The real seam, with the themed box itself replaced."""
    from napari_epicycle.ui_shell import theme

    ctrl = _ctrl(tmp_path)
    del ctrl._ask_for_run                     # the class's own seam
    ctrl._can_show_modal = lambda: True
    seen = []

    def _confirm(parent, title, message, confirm_text, cancel_text, **_k):
        seen.append((parent, title, message, confirm_text, cancel_text))
        return True

    monkeypatch.setattr(theme, "themed_confirm", _confirm)
    called = []
    ctrl._on_new_run_button_clicked = lambda: called.append(True)
    assert ctrl._require_run(T.CUR_BTN_OPEN) is False
    assert seen == [(ctrl.ui, "No run yet",
                     "Step 2 keeps your slices and models in a run; make one, "
                     "then press Open 2D annotator again.",
                     "New Run…", "Not now")]
    assert called == [True]


def test_default_box_with_runs_names_the_picker(tmp_path, monkeypatch):
    from napari_epicycle.ui_shell import theme

    _two_runs_none_active(tmp_path)
    ctrl = _ctrl(tmp_path)
    del ctrl._ask_for_run
    ctrl._can_show_modal = lambda: True
    seen = []
    monkeypatch.setattr(theme, "themed_confirm",
                        lambda *a, **k: seen.append(k["message"]) or False)
    ctrl._require_run(T.CUR_IMPORT_PAIRS_ACTION)
    assert seen == [
        "Step 2 keeps your slices and models in a run; pick a run in the Run "
        "picker (your slices from before are not kept there), or click New "
        "Run to keep them, then press Import pairs again."]


def test_default_box_for_an_unreadable_list_only_tells(tmp_path, monkeypatch):
    from napari_epicycle.ui_shell import theme

    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    del ctrl._ask_for_run
    ctrl._can_show_modal = lambda: True
    monkeypatch.setattr(theme, "themed_confirm",
                        lambda *a, **k: pytest.fail("no New Run offer"))
    assert ctrl._require_run("Predict") is False
    assert ctrl.told == [(T.RUN_UNREADABLE_TITLE, T.RUN_LIST_UNREADABLE)]


def test_default_box_shows_nothing_without_a_display(tmp_path, monkeypatch):
    from napari_epicycle.ui_shell import theme

    ctrl = _ctrl(tmp_path)
    del ctrl._ask_for_run
    monkeypatch.setattr(theme, "themed_confirm",
                        lambda *a, **k: pytest.fail("no modal headless"))
    assert ctrl._require_run("Predict") is False        # sync test mode


# ---------------------------------------------------------------------------
# New Run: the gate, the dialog flag, failures said
# ---------------------------------------------------------------------------

class _FakeDialog:
    built = []

    def __init__(self, **kwargs):
        _FakeDialog.built.append(kwargs)

    def exec_(self):
        return True

    def get_result(self):
        return {"slug": "pilot", "display_name": "Pilot", "activate": True}


@pytest.fixture
def fake_dialog(monkeypatch):
    from napari_epicycle.hitl_training_annotation import _new_run_dialog

    _FakeDialog.built = []
    monkeypatch.setattr(_new_run_dialog, "NewRunDialog", _FakeDialog)
    return _FakeDialog


def _make(ctrl, error):
    def _create(**_k):
        raise error
    ctrl._create_new_run = _create


def test_new_run_failure_already_there_is_said(tmp_path, fake_dialog, capsys):
    ctrl = _ctrl(tmp_path)
    _make(ctrl, RM.RunAlreadyExistsError("pilot"))
    ctrl._on_new_run_button_clicked()
    sentence = "A folder for the run 'pilot' is already there. Pick another name."
    assert ctrl.told == [(T.NEW_RUN_FAILED_TITLE, sentence)]
    assert sentence in _log(ctrl)
    assert "[STEP2-RUN] New Run failed" in capsys.readouterr().out


def test_new_run_failure_other_is_said(tmp_path, fake_dialog):
    ctrl = _ctrl(tmp_path)
    _make(ctrl, OSError("disk full"))
    ctrl._on_new_run_button_clicked()
    assert ctrl.told == [(T.NEW_RUN_FAILED_TITLE,
                          "The run could not be made: OSError: disk full.")]


def test_new_run_made_but_not_switched_is_said(tmp_path, fake_dialog):
    ctrl = _ctrl(tmp_path)

    def _switch(_slug):
        raise RuntimeError("the switch broke")

    ctrl._switch_active_run = _switch
    ctrl._on_new_run_button_clicked()
    assert (tmp_path / "artifacts" / "step2" / "runs" / "pilot").is_dir()
    [(title, sentence)] = ctrl.told
    assert title == T.NEW_RUN_FAILED_TITLE
    assert sentence.startswith("The run 'pilot' was made, but Step 2 could not "
                               "switch to it: RuntimeError: the switch broke.")


def test_new_run_on_an_old_layout_folder_is_refused_and_said(
        tmp_path, fake_dialog):
    ctrl = _ctrl(tmp_path, layout="legacy")
    ctrl._on_new_run_button_clicked()
    assert fake_dialog.built == []
    assert ctrl.told == [(T.NEW_RUN_FAILED_TITLE, wf.MSG_STEP2_LEGACY_LAYOUT)]


def test_new_run_on_an_unreadable_list_is_refused_and_said(
        tmp_path, fake_dialog):
    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    ctrl._on_new_run_button_clicked()
    assert fake_dialog.built == []
    assert ctrl.told == [(T.NEW_RUN_FAILED_TITLE, T.RUN_LIST_UNREADABLE)]
    assert (tmp_path / "step2_state.json").read_text("utf-8") == "{oops"


def test_create_only_not_offered_for_the_first_run(tmp_path, fake_dialog):
    ctrl = _ctrl(tmp_path)
    ctrl._create_new_run = lambda **_k: None
    ctrl._on_new_run_button_clicked()
    assert fake_dialog.built[-1]["allow_create_only"] is False


def test_create_only_offered_with_a_current_run(tmp_path, fake_dialog):
    ctrl = _ctrl(tmp_path, run="cells")
    ctrl._create_new_run = lambda **_k: None
    ctrl._on_new_run_button_clicked()
    assert fake_dialog.built[-1]["allow_create_only"] is True


def test_the_first_run_is_written_active(tmp_path):
    """The run list never names an active run that has no folder."""
    ctrl = _ctrl(tmp_path)
    switched = []
    ctrl._switch_active_run = lambda slug: switched.append(slug)
    ctrl._create_new_run("pilot", "Pilot")
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert state["active_run_id"] == "pilot"
    assert switched == ["pilot"]


def test_a_later_run_is_written_inactive_then_switched(tmp_path):
    ctrl = _ctrl(tmp_path, run="cells")
    switched = []
    ctrl._switch_active_run = lambda slug: switched.append(slug)
    ctrl._create_new_run("nuclei", "Nuclei")
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert state["active_run_id"] == "cells"
    assert switched == ["nuclei"]


# ---------------------------------------------------------------------------
# the first switch: its busy line, and no made-up "default"
# ---------------------------------------------------------------------------

class _StopSwitch(Exception):
    pass


def _busy_line_of_switch(ctrl, slug, monkeypatch):
    from napari_epicycle.hitl_training_annotation import _switch_run_guard as G

    monkeypatch.setattr(G, "check_switch_safety",
                        lambda _c, _v: (G.SwitchSafety.SAFE, []))
    lines = []

    def _busy(message="", cursor=True):
        lines.append(message)
        raise _StopSwitch

    ctrl._begin_busy = _busy
    with pytest.raises(_StopSwitch):
        ctrl._switch_active_run(slug)
    return lines


def test_first_run_busy_text(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path)
    RM.create_run_on_disk(tmp_path, "pilot", display_name="Pilot")
    RM.add_run_to_state_json(tmp_path, "pilot", activate=True)
    ctrl._run_just_made = "pilot"           # as New Run leaves it
    assert _busy_line_of_switch(ctrl, "pilot", monkeypatch) == [
        "[STEP2-RUN] Making the run 'Pilot' …"]


def test_a_later_switch_keeps_its_busy_text(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="cells")
    RM.create_run_on_disk(tmp_path, "nuclei", display_name="Nuclei")
    RM.add_run_to_state_json(tmp_path, "nuclei", activate=False)
    assert _busy_line_of_switch(ctrl, "nuclei", monkeypatch) == [
        "[STEP2-RUN] Switching to 'Nuclei' \u2026"]


def test_a_first_run_called_default_is_switched_to(tmp_path, monkeypatch):
    """With no current run, a first run named "default" is not mistaken for
    the one already active (the old fallback id made it a no-op)."""
    ctrl = _ctrl(tmp_path)
    RM.create_run_on_disk(tmp_path, "default", display_name="default")
    RM.add_run_to_state_json(tmp_path, "default", activate=True)
    ctrl._run_just_made = "default"         # as New Run leaves it
    assert _busy_line_of_switch(ctrl, "default", monkeypatch) == [
        "[STEP2-RUN] Making the run 'default' …"]


# ---------------------------------------------------------------------------
# the run row, the Run summary, the header
# ---------------------------------------------------------------------------

def test_empty_picker_enables_only_new_run(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl._refresh_run_picker()
    ui = ctrl.ui
    assert ui.run_picker_combo.itemText(0) == T.MSG_STEP2_NO_RUN
    assert ui.new_run_button.isEnabled() is True
    assert ui.import_pairs_button.isEnabled() is False
    assert ui.import_external_button.isEnabled() is False
    assert ui.manage_runs_button.isEnabled() is False


@pytest.mark.parametrize("damage", ["corrupt", "runs-without-list"])
def test_unreadable_picker_offers_nothing(tmp_path, damage):
    if damage == "corrupt":
        RM.create_run_on_disk(tmp_path, "cells", display_name="cells")
        (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    else:
        (tmp_path / "artifacts" / "step2" / "runs" / "x").mkdir(parents=True)
    ctrl = _ctrl(tmp_path)
    assert ctrl._list_runs_for_ui() == []
    assert ctrl._layout_allows_new_run() is False
    ctrl._refresh_run_picker()
    ui = ctrl.ui
    want = (T.RUN_LIST_UNREADABLE if damage == "corrupt"
            else T.RUN_LIST_MISSING)
    assert ui.run_picker_combo.items == [(want, "")]
    for btn in (ui.new_run_button, ui.import_pairs_button,
                ui.import_external_button, ui.manage_runs_button):
        assert btn.isEnabled() is False


def test_fresh_and_unreadable_folders_never_list_a_legacy_row(tmp_path):
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    runs_only = tmp_path / "runs_only"
    (runs_only / "artifacts" / "step2" / "runs" / "x").mkdir(parents=True)
    for ws in (fresh, runs_only):
        assert _ctrl(ws)._list_runs_for_ui() == []


def _round_status_ctrl(ws, **kw):
    ctrl = _ctrl(ws, **kw)
    del ctrl._refresh_round_status                 # the class's own method
    for name in ("_collect_prior_round_pairs", "_refresh_curation_model_status",
                 "_refresh_model_name_display", "_write_staged_count_text",
                 "_refresh_training_stage_line", "_refresh_init_hint"):
        setattr(ctrl, name, lambda *a, **k: [])
    ctrl._last_checkpoint_text = lambda: ""
    ctrl._format_target_label = lambda: ""
    ctrl._current_round_key = lambda: "lateral"
    ctrl._prior_pair_total_excluding_staged = lambda *_a: 0
    return ctrl


def test_fresh_run_label_not_legacy(tmp_path):
    ctrl = _round_status_ctrl(tmp_path)
    ctrl._refresh_round_status()
    assert ctrl.ui.status_run_value.value == "No run yet"
    assert "legacy" not in ctrl.ui.status_run_value.value


def test_run_label_names_the_run(tmp_path):
    ctrl = _round_status_ctrl(tmp_path, run="pilot")
    ctrl._refresh_round_status()
    assert ctrl.ui.status_run_value.value == "pilot"


def test_old_layout_run_label_says_legacy(tmp_path):
    ctrl = _round_status_ctrl(tmp_path, layout="legacy")
    ctrl.active_run_id = "default"
    ctrl._refresh_round_status()
    assert ctrl.ui.status_run_value.value == "default (legacy layout)"


def test_unreadable_run_label(tmp_path):
    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    ctrl = _round_status_ctrl(tmp_path)
    ctrl._refresh_round_status()
    assert ctrl.ui.status_run_value.value == T.RUN_SUMMARY_UNREADABLE


def test_run_summary_model_line_names_no_made_up_run():
    rounds = L.run_summary_rounds({"lateral": 3, "axial": 0}, "")
    assert rounds["lateral"]["model"] == T.RUN_SUMMARY_NONE
    assert L.run_summary_rounds({"lateral": 3}, "Cell")["lateral"]["model"] \
        == "Cell · round 2"


def test_session_payload_carries_no_made_up_run(tmp_path):
    ctrl = _ctrl(tmp_path)
    del ctrl._refresh_patch_list_badges            # the class's own method
    seen = []
    ctrl.ui.train_patch_list = _ListWidget()
    ctrl.ui.on_session_patches = seen.append
    ctrl._effective_stack_path = lambda: ""
    ctrl._refresh_patch_list_badges()
    assert len(seen) == 1
    assert seen[0]["run_id"] == ""


def test_the_empty_card_no_longer_promises_a_default_run():
    assert T.RUN_SUMMARY_EMPTY_LINES[1] == \
        "Pick one above, then make a run with New Run in Curation."


def test_shell_header_names_no_run_on_a_fresh_folder(tmp_path):
    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell

    def _status(run_id):
        shell = SimpleNamespace(
            _opening_workspace="", _workspace_path=str(tmp_path),
            _shell_ready=True,
            _STEP_SHORT_NAMES=EpiCycleWorkflowShell._STEP_SHORT_NAMES,
            _read_workspace_path=lambda: str(tmp_path),
            _get_step_controller=lambda _s: SimpleNamespace(active_run_id=run_id),
        )
        return EpiCycleWorkflowShell._read_active_step_status(shell, "roi_hitl_2d")

    assert _status("").startswith("Step 2  ·  lateral")
    assert "run" not in _status("").split("·")[0]
    assert _status("pilot").startswith("Step 2  ·  run pilot")


# ---------------------------------------------------------------------------
# a slice that could not be saved: one box per cause, the right word
# ---------------------------------------------------------------------------

def _add_slice(ctrl, label="stack01_XY_12", status=None):
    ctrl.segmentation_manager.data_inputs[label] = \
        np.random.rand(2, 16, 16).astype(np.float32)
    ctrl.segmentation_manager.data_masks[label] = np.ones((16, 16), np.uint16)
    if status:
        ctrl._touch_sample(label, status=status, view="XY")
    return label


def _folder_box(word):
    return (T.PAIR_NOT_SAVED_TITLE,
            "This slice could not be saved, so it is only on screen. Press "
            f"{word} again; if it fails again, check that the workspace "
            "folder is still there.")


def test_refused_finalize_box_names_finalize(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    label = _add_slice(ctrl, status="accepted")
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    ctrl._stage_accepted_in_current_view()
    ctrl._stage_accepted_in_current_view()
    assert ctrl.told == [_folder_box("Finalize accepted")]      # once per cause
    assert _log(ctrl).count(label) == 2                      # logged each time
    assert not (tmp_path / "training_data").exists()


def _accept_ready(ctrl, label):
    volume = np.zeros((20, 16, 16), dtype=np.int32)
    volume[12, 2:6, 2:6] = 1
    ctrl._parse_all_selected_sample_ids = lambda: [label]
    ctrl._work_labels_layer_name_for_view = lambda **_k: "work"
    ctrl._layer_by_name = lambda _n: _Layer(volume)
    ctrl._get_patch_view_image_layers = lambda **_k: [_Layer(np.zeros((20, 16, 16)))]
    ctrl.curation_sessions = SimpleNamespace(record_accept=_noop)
    ctrl.interactive_manager = SimpleNamespace(segmenter=None)
    ctrl.segmented_slice_list = set()


def test_refused_accept_box_names_accept(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    label = _add_slice(ctrl, status="curating")
    _accept_ready(ctrl, label)
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    ctrl._accept_annotation()
    assert ctrl.told == [_folder_box("Accept")]
    assert ctrl.sample_records[label]["status"] == "curating"


def test_refused_annotator_accept_box(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    label = "stack01_XY_12"
    _accept_ready(ctrl, label)
    ctrl._touch_sample(label, status="curating", view="XY")
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    result = ctrl._accept_native_annotation("XY", 12)
    assert result["ok"] is False
    assert ctrl.told == [_folder_box("Accept")]


def test_refused_predict_box_names_predict(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    label = "stack01_XY_12"
    stack = np.random.rand(2, 16, 16).astype(np.float32)
    ctrl._apply_model_source_selection = lambda **_k: {"can_segment": True}
    ctrl._ensure_segmentation_config_defaults = _noop
    ctrl._apply_inference_params_from_ui = _noop
    ctrl._collect_current_slice_input = lambda: (label, stack)
    ctrl._adjust_channels_for_input = _noop
    ctrl._confirm_resegment = lambda _l: True
    ctrl._layer_by_name = lambda _n: None
    ctrl.segmented_slice_list = set()
    seg = ctrl.segmentation_manager

    def _segment_image(inp, _cfg, lbl):
        seg.data_masks[lbl] = np.ones((16, 16), dtype=np.int32)
        return seg.data_masks[lbl], None, None

    seg.segment_image = _segment_image
    ctrl.image_manager = SimpleNamespace(
        convert_to_rgb=lambda **_k: np.zeros((16, 16, 3)))
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    ctrl._run_segmentation()
    assert ctrl.told == [_folder_box("Predict")]
    assert label not in seg.data_masks             # the cache goes back too


def test_refused_drawing_save_logs_once_and_shows_no_box(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl._last_pair_refusal = "folder"
    ctrl._say_pair_refused("drawing", ["stack01_XY_12"], box=False)
    ctrl._say_pair_refused("drawing", ["stack01_XY_13"], box=False)
    assert ctrl.told == []
    assert _log(ctrl).count(T.PAIR_NOT_SAVED_DRAWING) == 1


def test_refused_pair_without_a_run_says_the_no_run_sentence(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl._last_pair_refusal = "no-run"
    ctrl._say_pair_refused("Accept", ["stack01_XY_12"])
    assert ctrl.told == [(T.PAIR_NOT_SAVED_TITLE, T.MSG_STEP2_NO_RUN)]


# ---------------------------------------------------------------------------
# saves and refusals that used to reach only the console
# ---------------------------------------------------------------------------

def test_no_run_save_is_said_in_the_log_once(tmp_path):
    ctrl = _ctrl(tmp_path)
    for _ in range(3):
        ctrl._debounced_state_save()
    assert _log(ctrl).count(T.MSG_STEP2_NO_RUN) == 1
    assert ctrl.told == []                     # a save is not an action: no box


def test_failed_save_box_once_per_distinct_error(tmp_path, monkeypatch):
    from napari_epicycle.hitl_training_annotation.controller import (
        training_annotation_controller as tac,
    )

    ctrl = _ctrl(tmp_path, run="pilot")
    messages = iter(["the disk is full", "the disk is full", "access denied"])
    real = tac._atomic_write_json

    def _broken(path, payload, *a, **k):
        if Path(path).name == P.STEP2_STATE_FILENAME:
            raise OSError(next(messages))
        return real(path, payload, *a, **k)

    monkeypatch.setattr(tac, "_atomic_write_json", _broken)
    for _ in range(3):
        ctrl._debounced_state_save()
    assert ctrl.told == [(T.STATE_NOT_SAVED_TITLE, T.STATE_NOT_SAVED)] * 2
    assert _log(ctrl).count(T.STATE_NOT_SAVED) == 2


def test_refused_stage_does_not_claim_nothing_was_accepted(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    _add_slice(ctrl, status="accepted")
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    ctrl._stage_accepted_in_current_view()
    assert "No accepted samples to stage" not in _log(ctrl)



def test_import_on_a_runs_list_that_is_a_list_asks_nothing_of_the_disk(tmp_path):
    """A run list whose ``runs`` is not a table: the imports refuse at the
    gate (they never copy pairs into a run whose records cannot be saved)."""
    RM.create_run_on_disk(tmp_path, "pilot", display_name="pilot")
    (tmp_path / "step2_state.json").write_text(
        '{"schema_version": 3, "active_run_id": "pilot", "runs": []}',
        encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    ctrl.active_run_id = "pilot"
    ctrl._on_import_pairs_clicked()
    ctrl._on_import_external_clicked()
    assert [a[2] for a in ctrl.asked] == ["unreadable", "unreadable"]


def test_a_mixed_folder_names_the_migration(tmp_path):
    (tmp_path / "training_data").mkdir()
    (tmp_path / "artifacts" / "step2" / "runs" / "x").mkdir(parents=True)
    ctrl = _ctrl(tmp_path)
    ctrl.active_run_id = "default"
    assert ctrl._require_run("Predict") is False
    assert ctrl.asked == [("Predict", T.RUN_LIST_MIXED, "unreadable")]
    assert "Browse" in T.RUN_LIST_MIXED


# ---------------------------------------------------------------------------
# the retry the box promises: Accept -> box -> New Run -> Accept writes
# ---------------------------------------------------------------------------

_SWITCH_QUIET = (
    "_populate_patch_list_from_disk", "_refresh_model_source_controls",
    "_apply_model_source_selection", "_apply_training_mode_for_active_bucket",
    "_set_axial_combo_enabled", "_sync_training_controls_from_config",
    "_sync_inference_controls_from_config", "_refresh_current_config_from_widgets",
    "_refresh_run_picker",
)


def test_accept_after_new_run_from_the_box_writes_the_drawn_slice(
        tmp_path, fake_dialog, monkeypatch):
    """Draw, press Accept with no run, choose New Run in the box: the run is
    made and switched to, the drawn slice's row and the open patch are still
    there, and pressing Accept again writes the pair into the new run."""
    from napari_epicycle.hitl_training_annotation import _switch_run_guard as G

    monkeypatch.setattr(G, "_annotator_dock_open", lambda _v: False)
    ctrl = _ctrl(tmp_path)
    for name in _SWITCH_QUIET:
        setattr(ctrl, name, _noop)
    label = _add_slice(ctrl)
    # the drawing's own record (the per-stroke save was refused: no run)
    ctrl.sample_records[label] = {"status": "curating", "bucket": "lateral",
                                  "view": "XY", "updated_at": "1"}
    _accept_ready(ctrl, label)
    ctrl.answer = True                      # the box: "New Run…"
    ctrl._accept_annotation()
    assert ctrl.asked and ctrl.asked[0][0] == T.CUR_BTN_ACCEPT
    assert ctrl.active_run_id == "pilot"
    assert ctrl.loaded_patch_name == "stack01"
    assert ctrl.sample_records[label]["status"] == "curating"
    assert label in ctrl.segmentation_manager.data_masks
    ctrl._accept_annotation()               # "press Accept again"
    assert ctrl.sample_records[label]["status"] == "accepted"
    accepted = P.round_accepted_dir(tmp_path, "pilot", "lateral", 0)
    assert sorted(p.name for p in accepted.iterdir()) == [
        "stack01_XY_z0012_img.tif", "stack01_XY_z0012_mask.tif"]
    assert not (tmp_path / "training_data").exists()
    assert not (tmp_path / "hitl_state.json").exists()
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    assert label in state["runs"]["pilot"]["sample_records"]



# ---------------------------------------------------------------------------
# a background drawing write that fails is written again and said
# ---------------------------------------------------------------------------

class _FakeWriter:
    def __init__(self, ok):
        self.ok = ok
        self.paths = []

    def submit(self, path, data, on_done=None):
        self.paths.append(Path(path).name)
        if on_done is not None:
            on_done(self.ok, None if self.ok else OSError("the disk is full"))

    def flush(self, timeout=None):
        return True


def test_a_failed_background_write_is_written_again_and_said(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    label = _add_slice(ctrl, status="curating")
    ctrl._pair_writer = _FakeWriter(ok=False)
    assert ctrl._persist_curated_training_pair(label, async_write=True)
    sigs = list(ctrl._persisted_pair_sigs.values())[0]
    assert "mask" not in sigs, "no signature for a file that was not written"
    assert ctrl._flush_pending_state_timers() is False
    assert T.PAIR_NOT_SAVED_DRAWING in _log(ctrl)
    ctrl._pair_writer = _FakeWriter(ok=True)
    ctrl._persist_curated_training_pair(label, async_write=True)
    assert "stack01_XY_z0012_mask.tif" in ctrl._pair_writer.paths
    assert "mask" in list(ctrl._persisted_pair_sigs.values())[0]



def test_a_failed_background_write_stays_pending_until_it_writes(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    label = _add_slice(ctrl, status="curating")
    _accept_ready(ctrl, label)
    ctrl._refresh_run_picker = _noop
    ctrl._curating_persist_pending = {label: (label, "XY", 12)}
    ctrl._pair_writer = _FakeWriter(ok=False)
    assert ctrl._flush_pending_state_timers() is False
    assert label in ctrl._curating_persist_pending, "back in the queue"
    assert ctrl._flush_pending_state_timers() is False, "still not written"
    ctrl._pair_writer = _FakeWriter(ok=True)
    assert ctrl._flush_pending_state_timers() is True
    assert ctrl._curating_persist_pending == {}
    assert not (ctrl._pair_write_failures.get(str(tmp_path)) or {})

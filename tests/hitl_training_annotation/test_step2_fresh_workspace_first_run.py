"""Step 2: making the first run of a folder keeps what is on screen.

The rule under test, in plain words:

* With no current run (a fresh folder), making and switching to the first
  run keeps the open patch, its view and its slice: no question is asked
  (there is no old run to name), no layer is cleared and the patch is not
  reloaded.
* The settings set up before the run existed (channels, epochs, learning
  rate, a loaded config, the model source, the training mode, the
  auto-update tick) go into the new run: in memory, in the widgets and in
  the run list on disk.  A model the New Run dialog chose wins over the
  widget's.
* A switch between two runs still asks (when the annotator is open), clears
  the patch layers and loads the new run's own patch, as before.
* With runs but none active, the Run picker shows its "pick a run" line,
  not the first run as if it were the current one.
* "Create only" on a fresh folder still makes the run the current one, so
  the run list and Step 2 agree.

Qt-free: the controller is built with ``__new__`` and plain fakes; the
question box is replaced by a recorder, so no test builds a real modal.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_epicycle.hitl_training_annotation import _run_management as RM
from napari_epicycle.hitl_training_annotation import _switch_run_dialog as D
from napari_epicycle.hitl_training_annotation import _switch_run_guard as G
from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T
from napari_epicycle.io_registry import HITLPersistenceService


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _Label:
    def __init__(self, text=""):
        self.value = text

    def setText(self, text):
        self.value = text


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


class _Combo:
    def __init__(self):
        self.items = []
        self.enabled = None
        self.current = -1

    def blockSignals(self, on):
        pass

    def clear(self):
        self.items = []
        self.current = -1

    def addItem(self, text, data=None):
        self.items.append((text, data))
        if self.current < 0:
            self.current = 0      # Qt makes the first item current

    def insertItem(self, i, text, data=None):
        self.items.insert(i, (text, data))
        if self.current < 0:
            self.current = 0

    def count(self):
        return len(self.items)

    def itemText(self, i):
        return self.items[i][0]

    def itemData(self, i):
        return self.items[i][1]

    def insertSeparator(self, i):
        self.items.insert(i, ("---", None))

    def setCurrentIndex(self, i):
        self.current = i

    def currentText(self):
        return self.items[self.current][0] if self.current >= 0 else ""

    def currentData(self):
        return self.items[self.current][1] if self.current >= 0 else None

    def setEnabled(self, v):
        self.enabled = bool(v)


class _UI:
    def __init__(self):
        self.training_log_output = _LogOutput()
        self.loaded_patch_label = _Label("Patch: stack01")
        self.run_picker_combo = _Combo()
        self.new_run_button = _Button()
        self.import_pairs_button = _Button()
        self.import_external_button = _Button()
        self.manage_runs_button = _Button()


class _Dims:
    def __init__(self):
        self.ndim = 3
        self.order = (1, 0, 2)          # the XZ view
        self.current_step = (0, 12, 0)  # slice 12 of the XZ view
        self.moves = []

    def set_current_step(self, axis, value):
        self.moves.append((axis, value))
        step = list(self.current_step)
        step[axis] = value
        self.current_step = tuple(step)


class _Layers(list):
    def __contains__(self, name):
        return any(getattr(x, "name", None) == name for x in self)


def _noop(*_a, **_k):
    return None


_QUIET = (
    "_sync_data_to_add_list", "_refresh_sample_queue", "_refresh_round_status",
    "_refresh_patch_list_badges", "_begin_busy", "_end_busy",
    "_populate_patch_list_from_disk", "_refresh_model_source_controls",
    "_apply_model_source_selection", "_apply_training_mode_for_active_bucket",
    "_set_axial_combo_enabled", "_schedule_state_save",
    "_write_per_bucket_metadata_for_active_run",
)

#: what the training / inference widgets hold before any run exists
_WIDGETS = {
    "epochs": 77,
    "learning_rate": 0.0042,
    "channel1": "1: red",
    "channel2": "0: gray",
    "diameter": 31,
}


def _ctrl(ws, *, run=None):
    """A controller on ``ws`` with an open patch (stack01, XZ view, slice 12)
    and widget settings that are not the defaults.  ``run``: make that run on
    disk, activate it and point the controller at it."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.persistence = HITLPersistenceService()
    ctrl.working_dir = str(ws)
    ctrl.current_round = 0
    # a loaded config's key that no widget shows
    ctrl.current_config = {"model_type": "cyto3", "weight_decay": 0.07}
    ctrl.training_data_by_round = {}
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.round_backend_info = {}
    ctrl.loaded_patch_name = "stack01"
    ctrl.loaded_stack_path = str(Path(ws) / "extractions")
    ctrl.selected_view = "XZ"
    ctrl.view_target = "axial"
    ctrl.axial_plane = "XZ"
    ctrl.sample_records = {}
    ctrl.training_mode = "from_scratch"
    ctrl.training_mode_by_bucket = {"lateral": "cumulative", "axial": "from_scratch"}
    ctrl.active_curation_checkpoints = {"lateral": "", "axial": ""}
    ctrl.curation_model_source_mode = "pretrained"
    ctrl.imported_curation_model_path = ""
    ctrl.curation_pretrained_model_type = "cyto2"
    ctrl.active_run_id = ""
    ctrl._run_switch_in_flight = False
    ctrl._patch_load_token = 5
    ctrl.ui = _UI()
    ctrl.viewer = SimpleNamespace(
        layers=_Layers([SimpleNamespace(name="stack01", visible=True),
                        SimpleNamespace(name="work_labels_axial", visible=True)]),
        dims=_Dims(),
    )
    for name in _QUIET:
        setattr(ctrl, name, _noop)
    ctrl._get_auto_update_model_checked = lambda: False

    def _from_widgets():
        ctrl.current_config.update(_WIDGETS)

    ctrl._refresh_current_config_from_widgets = _from_widgets
    ctrl.pushed = []      # the config each widget push received
    ctrl._sync_training_controls_from_config = (
        lambda: ctrl.pushed.append(("training", dict(ctrl.current_config))))
    ctrl._sync_inference_controls_from_config = (
        lambda: ctrl.pushed.append(("inference", dict(ctrl.current_config))))
    ctrl.calls = []
    ctrl._close_step2_annotator_dock_safely = lambda: ctrl.calls.append("close")
    ctrl._clear_patch_viewer_layers_safely = lambda: ctrl.calls.append("clear")
    ctrl._auto_reload_saved_patch = lambda: ctrl.calls.append("reload-patch")
    ctrl._launch_multiview_annotator = lambda: ctrl.calls.append("reopen")
    ctrl.told = []
    ctrl._tell_step2 = lambda title, sentence: ctrl.told.append((title, sentence))
    ctrl.left_no_run = []        # the "slices belong to no run" question
    ctrl._confirm_leave_no_run = (
        lambda name: ctrl.left_no_run.append(name) or True)
    if run:
        RM.create_run_on_disk(ws, run, display_name=run)
        RM.add_run_to_state_json(ws, run, activate=True)
        ctrl.active_run_id = run
    return ctrl


@pytest.fixture
def asked(monkeypatch):
    """Every switch question: the annotator counts as open, so a switch
    between runs asks; the answer is yes."""
    seen = []
    monkeypatch.setattr(
        G, "check_switch_safety",
        lambda _c, _v: (G.SwitchSafety.NEEDS_CONFIRMATION, ["annotator open"]))

    def _confirm(**kw):
        seen.append(kw)
        return True

    monkeypatch.setattr(D, "show_switch_confirm", _confirm)
    monkeypatch.setattr(G, "_annotator_dock_open", lambda _v: False)
    return seen


def _run_entry(ws, slug):
    state = json.loads((Path(ws) / "step2_state.json").read_text("utf-8"))
    return state, state["runs"][slug]


# ---------------------------------------------------------------------------
# the first run keeps the screen
# ---------------------------------------------------------------------------

def test_first_run_keeps_patch_and_slice(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl._create_new_run("pilot", "Pilot")
    assert ctrl.active_run_id == "pilot"
    assert ctrl.loaded_patch_name == "stack01"
    assert ctrl.loaded_stack_path == str(tmp_path / "extractions")
    assert ctrl.ui.loaded_patch_label.value == "Patch: stack01"
    assert (ctrl.selected_view, ctrl.view_target) == ("XZ", "axial")
    assert ctrl.viewer.dims.current_step == (0, 12, 0)
    assert ctrl.viewer.dims.moves == []
    assert ctrl._patch_load_token == 5, "the open patch's load is not cancelled"
    assert "clear" not in ctrl.calls and "close" not in ctrl.calls
    assert "reload-patch" not in ctrl.calls, "the open patch is not reloaded"
    _state, entry = _run_entry(tmp_path, "pilot")
    assert entry["loaded_patch_name"] == "stack01"
    assert entry["selected_view"] == "XZ"


def test_first_run_asks_nothing(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl._create_new_run("pilot", "Pilot")
    assert asked == [], "no question names an old run that does not exist"


def test_first_run_carries_widget_config(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl._create_new_run("pilot", "Pilot")
    for key, value in _WIDGETS.items():
        assert ctrl.current_config[key] == value, key
    assert ctrl.current_config["weight_decay"] == 0.07, "a loaded config's key"
    # the widgets were given the carried settings, never the run's empty ones
    assert ctrl.pushed, "the reload pushes the config to the widgets"
    for _which, cfg in ctrl.pushed:
        assert cfg.get("epochs") == 77 and cfg.get("channel1") == "1: red"
    assert ctrl.curation_model_source_mode == "pretrained"
    assert ctrl.curation_pretrained_model_type == "cyto2"
    assert ctrl.training_mode_by_bucket == {"lateral": "cumulative",
                                            "axial": "from_scratch"}
    _state, entry = _run_entry(tmp_path, "pilot")
    for key, value in _WIDGETS.items():
        assert entry["current_config"][key] == value, key
    assert entry["curation_model_source_mode"] == "pretrained"
    assert entry["curation_pretrained_model_type"] == "cyto2"
    assert entry["auto_update_model_after_training"] is False


def test_init_source_beats_widget_default(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.curation_model_source_mode = "auto"
    ctrl.curation_pretrained_model_type = "cyto3"
    ctrl._create_new_run(
        "default", "default",
        init_spec={"mode": "pretrained", "pretrained_name": "nuclei"})
    assert ctrl.active_run_id == "default"
    assert ctrl.curation_model_source_mode == "pretrained"
    assert ctrl.curation_pretrained_model_type == "nuclei"
    _state, entry = _run_entry(tmp_path, "default")
    assert entry["curation_model_source_mode"] == "pretrained"
    assert entry["curation_pretrained_model_type"] == "nuclei"
    # the dialog chose the model; the widget settings still came along
    assert entry["current_config"]["epochs"] == 77


def test_init_source_beats_a_widget_model(tmp_path, asked):
    """A model the widgets picked never replaces the one the dialog chose."""
    ctrl = _ctrl(tmp_path)          # widgets: Cellpose pretrained cyto2
    ext = tmp_path / "my_model"
    ext.write_bytes(b"x")
    ctrl._create_new_run(
        "pilot", "Pilot",
        init_spec={"mode": "browse", "external_path": str(ext)})
    assert ctrl.curation_model_source_mode == "imported"
    assert ctrl.imported_curation_model_path == str(ext)
    _state, entry = _run_entry(tmp_path, "pilot")
    assert entry["curation_model_source_mode"] == "imported"
    assert entry["current_config"]["starting_checkpoint"] == str(ext)


def test_first_run_on_a_folder_with_no_patch_open(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.loaded_patch_name = ""
    ctrl._create_new_run("pilot", "Pilot")
    assert ctrl.active_run_id == "pilot"
    assert ctrl.loaded_patch_name == ""
    assert "clear" not in ctrl.calls


# ---------------------------------------------------------------------------
# a switch between runs is as before
# ---------------------------------------------------------------------------

def test_a_second_run_still_asks_and_clears(tmp_path, asked):
    ctrl = _ctrl(tmp_path, run="cells")
    ctrl._create_new_run("nuclei", "Nuclei")
    assert ctrl.active_run_id == "nuclei"
    assert len(asked) == 1
    assert asked[0]["old_run_id"] == "cells"
    assert "close" in ctrl.calls and "clear" in ctrl.calls
    assert "reload-patch" in ctrl.calls
    assert ctrl.loaded_patch_name == "", "the new run has no patch of its own"
    assert ctrl._patch_load_token == 6


def test_a_second_run_takes_its_own_config(tmp_path, asked):
    """Between runs the new run's own (empty) settings load, as before."""
    ctrl = _ctrl(tmp_path, run="cells")
    ctrl._create_new_run("nuclei", "Nuclei")
    assert "epochs" not in ctrl.current_config
    _state, entry = _run_entry(tmp_path, "nuclei")
    assert "epochs" not in entry["current_config"]


def test_runs_but_none_active_picking_a_used_run_loads_its_patch(
        tmp_path, asked):
    """No current run, but the picked run has a patch of its own: its patch
    is the one loaded (the screen's is cleared), and nothing is asked."""
    ctrl = _ctrl(tmp_path)
    RM.create_run_on_disk(tmp_path, "cells", display_name="cells")
    RM.add_run_to_state_json(tmp_path, "cells", activate=False)
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    state["runs"]["cells"]["loaded_patch_name"] = "stack02"
    state["runs"]["cells"]["current_config"] = {"epochs": 5}
    state["active_run_id"] = "gone"
    (tmp_path / "step2_state.json").write_text(json.dumps(state), "utf-8")
    ctrl._switch_active_run("cells")
    assert asked == [], "no question names an old run that does not exist"
    # an open patch alone is not lost work: no question, as before
    assert ctrl.left_no_run == []
    assert ctrl.active_run_id == "cells"
    assert "clear" in ctrl.calls and "reload-patch" in ctrl.calls
    assert ctrl.loaded_patch_name == "stack02"
    assert ctrl.current_config.get("epochs") == 5, "the run's own settings"


# ---------------------------------------------------------------------------
# "Create only" with no current run
# ---------------------------------------------------------------------------

def test_create_only_on_a_fresh_folder_still_switches(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl._create_new_run("pilot", "Pilot", activate=False)
    state, _entry = _run_entry(tmp_path, "pilot")
    assert state["active_run_id"] == "pilot"
    assert ctrl.active_run_id == "pilot", "the run list and Step 2 agree"
    assert ctrl.loaded_patch_name == "stack01"


def test_create_only_with_a_current_run_stays(tmp_path, asked):
    ctrl = _ctrl(tmp_path, run="cells")
    ctrl._create_new_run("nuclei", "Nuclei", activate=False)
    state, _entry = _run_entry(tmp_path, "nuclei")
    assert state["active_run_id"] == "cells"
    assert ctrl.active_run_id == "cells"


# ---------------------------------------------------------------------------
# the Run picker with runs but none active
# ---------------------------------------------------------------------------

def _two_runs_none_active(ws):
    for slug in ("cells", "nuclei"):
        RM.create_run_on_disk(ws, slug, display_name=slug)
        RM.add_run_to_state_json(ws, slug, activate=False)
    state = json.loads((Path(ws) / "step2_state.json").read_text("utf-8"))
    state["active_run_id"] = "gone"
    (Path(ws) / "step2_state.json").write_text(json.dumps(state), "utf-8")


def test_picker_with_runs_but_none_active_shows_its_line(tmp_path):
    ctrl = _ctrl(tmp_path)
    _two_runs_none_active(tmp_path)
    ctrl._refresh_run_picker()
    combo = ctrl.ui.run_picker_combo
    assert combo.currentText() == T.RUN_PICKER_NONE_ACTIVE
    assert combo.currentData() == "", "the line is not a run"
    assert {combo.itemData(i) for i in range(combo.count())} >= {"cells",
                                                                  "nuclei"}
    assert ctrl.ui.new_run_button.enabled is True


def test_picker_line_when_the_list_names_a_run_with_no_folder(tmp_path):
    """The run list's active run has lost its folder: Step 2 has no current
    run, so the picker does not show that run as the current one."""
    ctrl = _ctrl(tmp_path)
    for slug in ("cells", "nuclei"):
        RM.create_run_on_disk(tmp_path, slug, display_name=slug)
        RM.add_run_to_state_json(tmp_path, slug, activate=(slug == "cells"))
    import shutil
    shutil.rmtree(tmp_path / "artifacts" / "step2" / "runs" / "cells")
    ctrl._refresh_run_picker()
    assert ctrl.ui.run_picker_combo.currentText() == T.RUN_PICKER_NONE_ACTIVE


def test_picker_with_an_active_run_shows_it(tmp_path):
    ctrl = _ctrl(tmp_path, run="cells")
    RM.create_run_on_disk(tmp_path, "nuclei", display_name="nuclei")
    RM.add_run_to_state_json(tmp_path, "nuclei", activate=False)
    ctrl._refresh_run_picker()
    combo = ctrl.ui.run_picker_combo
    assert combo.currentData() == "cells"
    assert T.RUN_PICKER_NONE_ACTIVE not in [combo.itemText(i)
                                     for i in range(combo.count())]


def test_picking_the_line_switches_nothing(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    _two_runs_none_active(tmp_path)
    ctrl._refresh_run_picker()
    ctrl._on_run_picker_activated(0)
    assert ctrl.active_run_id == ""
    assert ctrl.calls == []


# ---------------------------------------------------------------------------
# the slices worked on before the run, and the busy line
# ---------------------------------------------------------------------------

def test_first_run_carries_the_slices_worked_on_before_it(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.sample_records = {"stack01_XZ_12": {"status": "curating",
                                             "bucket": "axial", "view": "XZ"}}
    ctrl._create_new_run("pilot", "Pilot")
    assert ctrl.sample_records["stack01_XZ_12"]["status"] == "curating"
    _state, entry = _run_entry(tmp_path, "pilot")
    assert entry["sample_records"]["stack01_XZ_12"]["status"] == "curating"


def test_a_second_run_does_not_take_the_first_runs_slices(tmp_path, asked):
    ctrl = _ctrl(tmp_path, run="cells")
    ctrl.sample_records = {"stack01_XZ_12": {"status": "curating"}}
    ctrl._create_new_run("nuclei", "Nuclei")
    assert ctrl.sample_records == {}


class _Busy(Exception):
    pass


def _busy_line(ctrl, call):
    lines = []

    def _begin(message="", cursor=True):
        lines.append(message)
        raise _Busy

    ctrl._begin_busy = _begin
    with pytest.raises(_Busy):
        call()
    return lines


def test_the_run_just_made_says_making(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    assert _busy_line(ctrl, lambda: ctrl._create_new_run("pilot", "Pilot")) == [
        "[STEP2-RUN] " + T.RUN_MAKING.format(name="Pilot")]


def test_picking_an_existing_run_with_none_active_says_switching(
        tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    _two_runs_none_active(tmp_path)
    assert _busy_line(ctrl, lambda: ctrl._switch_active_run("cells")) == [
        "[STEP2-RUN] " + T.RUN_SWITCHING.format(name="cells")]
    assert T.RUN_SWITCHING.endswith("\u2026")


# ---------------------------------------------------------------------------
# the picker's tooltip holds the sentence the narrow box cuts off
# ---------------------------------------------------------------------------

class _TipCombo(_Combo):
    def __init__(self):
        super().__init__()
        self.tip = "Pick the active run."

    def toolTip(self):
        return self.tip

    def setToolTip(self, tip):
        self.tip = tip


def test_unreadable_picker_tooltip_is_the_whole_sentence(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl.ui.run_picker_combo = _TipCombo()
    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    ctrl._refresh_run_picker()
    assert ctrl.ui.run_picker_combo.tip == T.RUN_LIST_UNREADABLE


def test_none_active_picker_tooltip_then_back(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl.ui.run_picker_combo = _TipCombo()
    _two_runs_none_active(tmp_path)
    ctrl._refresh_run_picker()
    assert ctrl.ui.run_picker_combo.tip == T.RUN_PICKER_NONE_ACTIVE
    ctrl.active_run_id = "cells"
    state = json.loads((tmp_path / "step2_state.json").read_text("utf-8"))
    state["active_run_id"] = "cells"
    (tmp_path / "step2_state.json").write_text(json.dumps(state), "utf-8")
    ctrl._refresh_run_picker()
    assert ctrl.ui.run_picker_combo.tip == "Pick the active run."


# ---------------------------------------------------------------------------
# an old-layout folder beside a folder of runs is not the old single run
# ---------------------------------------------------------------------------

def _mixed_folder(ws):
    (Path(ws) / "training_data").mkdir()
    (Path(ws) / "artifacts" / "step2" / "runs" / "x").mkdir(parents=True)


def test_mixed_folder_without_run_list_loads_no_run(tmp_path):
    _mixed_folder(tmp_path)
    ctrl = _ctrl(tmp_path)
    ctrl.active_run_id = "default"
    ctrl._load_internal_state()
    assert ctrl._step2_write_target() == "unreadable"
    assert ctrl.active_run_id == ""


def test_old_layout_folder_still_loads_its_single_run(tmp_path):
    (tmp_path / "training_data").mkdir()
    ctrl = _ctrl(tmp_path)
    ctrl._load_internal_state()
    assert ctrl.active_run_id == "default"


# ---------------------------------------------------------------------------
# the words: console, box titles, the card, the import word, one title
# ---------------------------------------------------------------------------

def test_load_line_before_any_copy_says_first_save(tmp_path, capsys):
    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    ctrl._load_internal_state()
    out = capsys.readouterr().out
    assert "a copy could not be set aside" not in out
    assert "nothing saved; a copy is set aside at the first save" in out
    assert not list(tmp_path.glob("step2_state.json.corrupt-*"))


def test_the_copy_line_is_per_workspace(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for ws in (a, b):
        ws.mkdir()
        (ws / "step2_state.json").write_text("{oops", encoding="utf-8")
    ctrl = _ctrl(a)
    ctrl._save_internal_state()
    assert str(a) in ctrl._unreadable_console_line()
    ctrl.working_dir = str(b)
    assert "set aside at the first save" in ctrl._unreadable_console_line()
    ctrl._save_internal_state()
    assert str(b) in ctrl._unreadable_console_line()


@pytest.mark.parametrize("state,word", [
    ("no-runs", "no run yet"), ("none-active", "no run is active"),
    ("unreadable", "the run list cannot be used"),
    ("no-workspace", "no workspace")])
def test_console_word_follows_the_state(tmp_path, capsys, state, word):
    if state == "none-active":
        _two_runs_none_active(tmp_path)
    elif state == "unreadable":
        (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    if state == "no-workspace":
        ctrl.working_dir = ""
    ctrl._ask_for_run = lambda *_a: False
    assert ctrl._require_run("Predict") is False
    assert f"[STEP2-RUN] Predict: {word}" in capsys.readouterr().out


def _box_ctrl(tmp_path, monkeypatch, state):
    from napari_epicycle.ui_shell import theme
    seen = []
    monkeypatch.setattr(theme, "themed_confirm",
                        lambda *a, **k: seen.append(("confirm", k)) or False)
    monkeypatch.setattr(theme, "themed_notice",
                        lambda *a, **k: seen.append(("notice", a)))
    if state == "none-active":
        _two_runs_none_active(tmp_path)
    ctrl = _ctrl(tmp_path)
    ctrl._can_show_modal = lambda: True
    ctrl._layout_allows_new_run = lambda: True
    return ctrl, seen


def test_box_title_with_runs_but_none_active(tmp_path, monkeypatch):
    ctrl, seen = _box_ctrl(tmp_path, monkeypatch, "none-active")
    ctrl._ask_for_run("Predict", T.RUN_NONE_ACTIVE, "none-active")
    assert seen[0][1]["title"] == T.RUN_ASK_TITLE_PICK == "No run is active"


def test_box_title_with_no_runs(tmp_path, monkeypatch):
    ctrl, seen = _box_ctrl(tmp_path, monkeypatch, "no-runs")
    ctrl._ask_for_run("Predict", T.MSG_STEP2_NO_RUN, "no-runs")
    assert seen[0][1]["title"] == T.RUN_ASK_TITLE


@pytest.mark.parametrize("action,word", [
    (T.CUR_IMPORT_PAIRS_ACTION, "Import pairs"),
    (T.CUR_IMPORT_EXTERNAL_ACTION, "Import external pairs")])
def test_box_names_the_import_action_plainly(tmp_path, monkeypatch, action,
                                             word):
    ctrl, seen = _box_ctrl(tmp_path, monkeypatch, "no-runs")
    ctrl._ask_for_run(action, T.MSG_STEP2_NO_RUN, "no-runs")
    message = seen[0][1]["message"]
    assert f"press {word} again" in message
    assert "…" not in message


def _run_word(ctrl):
    seen = {}
    ctrl._safe_ui_set_text = lambda name, text: seen.setdefault(name, text)
    del ctrl._refresh_round_status          # the real one (quiet elsewhere)
    try:
        ctrl._refresh_round_status()
    except Exception:      # the card's other rows need more of the UI
        pass
    return seen.get("status_run_value")


@pytest.mark.parametrize("damage,word", [
    ("corrupt", T.RUN_SUMMARY_UNREADABLE),
    ("mixed", T.RUN_SUMMARY_UNREADABLE_MIXED),
    ("runs-only", T.RUN_SUMMARY_UNREADABLE_MISSING)])
def test_card_word_follows_the_kind(tmp_path, damage, word):
    if damage == "corrupt":
        (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    elif damage == "mixed":
        _mixed_folder(tmp_path)
    else:
        (tmp_path / "artifacts" / "step2" / "runs" / "x").mkdir(parents=True)
    ctrl = _ctrl(tmp_path)
    assert _run_word(ctrl) == word
    assert word[0].isupper()


def test_refused_switch_save_uses_the_switch_title(tmp_path, asked):
    ctrl = _ctrl(tmp_path, run="cells")
    RM.create_run_on_disk(tmp_path, "nuclei", display_name="nuclei")
    RM.add_run_to_state_json(tmp_path, "nuclei")
    ctrl._save_internal_state = lambda: "refused"
    ctrl._last_save_error = "OSError: locked"
    ctrl._switch_active_run("nuclei")
    assert ctrl.active_run_id == "cells"
    assert [title for title, _s in ctrl.told] == [T.RUN_SWITCH_FAILED_TITLE]


# ---------------------------------------------------------------------------
# a run list Step 2 cannot use: a rescue file only when something was done
# ---------------------------------------------------------------------------

def _rescues(ws):
    return list((Path(ws) / "artifacts" / "step2" / "rescue").glob("*.json"))


def test_closing_an_unreadable_project_untouched_writes_no_rescue(tmp_path):
    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    ctrl.shutdown()
    assert _rescues(tmp_path) == []


def test_closing_an_unreadable_project_with_slices_writes_a_rescue(tmp_path):
    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    ctrl.sample_records = {"stack01_XY_1": {"status": "curating"}}
    ctrl.shutdown()
    assert len(_rescues(tmp_path)) == 1



# ---------------------------------------------------------------------------
# picking an existing run while none is active
# ---------------------------------------------------------------------------

def _used_run_none_active(ws):
    RM.create_run_on_disk(ws, "cells", display_name="cells")
    RM.add_run_to_state_json(ws, "cells", activate=False)
    state = json.loads((Path(ws) / "step2_state.json").read_text("utf-8"))
    state["runs"]["cells"]["loaded_patch_name"] = "stack02"
    state["active_run_id"] = "gone"
    (Path(ws) / "step2_state.json").write_text(json.dumps(state), "utf-8")


def test_picking_a_run_with_work_on_screen_asks_first(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.sample_records = {"stack01_XY_4": {"status": "curating"}}
    _used_run_none_active(tmp_path)
    ctrl._confirm_leave_no_run = (
        lambda name: ctrl.left_no_run.append(name) or False)
    ctrl._switch_active_run("cells")
    assert ctrl.left_no_run == ["cells"]
    assert ctrl.active_run_id == ""
    assert "stack01_XY_4" in ctrl.sample_records
    assert ctrl.calls == [], "nothing closed, cleared or reloaded"


def test_picking_a_run_and_choosing_to_switch_drops_the_drawings(
        tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.sample_records = {"stack01_XY_4": {"status": "curating"}}
    ctrl._curating_persist_pending = {"stack01_XY_4": ("stack01_XY_4", "XY", 4)}
    ctrl._flush_curating_persist_pending = lambda: True
    _used_run_none_active(tmp_path)
    ctrl._switch_active_run("cells")
    assert ctrl.active_run_id == "cells"
    assert ctrl.loaded_patch_name == "stack02"
    assert ctrl._curating_persist_pending == {}


def test_picking_a_run_with_nothing_on_screen_asks_nothing(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    ctrl.loaded_patch_name = ""
    _used_run_none_active(tmp_path)
    ctrl._switch_active_run("cells")
    assert ctrl.left_no_run == []
    assert ctrl.active_run_id == "cells"


def test_the_no_run_question_words():
    assert T.RUN_SWITCH_NO_RUN_BODY == (
        "Your slices from before a run was active are not kept; make a New "
        "Run to keep them, or switch and lose them.")
    assert "(your slices from before are not kept there)" in T.RUN_ASK_BODY_PICK


def test_card_word_with_runs_but_none_active(tmp_path):
    ctrl = _ctrl(tmp_path)
    _two_runs_none_active(tmp_path)
    assert _run_word(ctrl) == T.RUN_SUMMARY_NONE_ACTIVE == "None active"


def test_first_run_line_names_the_patch_only_when_one_is_open(
        tmp_path, asked, capsys):
    ctrl = _ctrl(tmp_path)
    ctrl.loaded_patch_name = ""
    ctrl._create_new_run("pilot", "Pilot")
    out = capsys.readouterr().out
    assert "first run 'pilot': carried" in out
    assert "kept the open patch" not in out


def test_the_first_runs_refused_save_is_logged(tmp_path, asked):
    ctrl = _ctrl(tmp_path)
    real = ctrl._save_internal_state
    ctrl._save_internal_state = lambda: "refused"
    ctrl._last_save_error = "OSError: locked"
    ctrl._create_new_run("pilot", "Pilot")
    assert T.STATE_NOT_SAVED in "\n".join(ctrl.ui.training_log_output.lines)
    assert real

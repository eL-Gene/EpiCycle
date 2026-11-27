"""A project that already has an active run behaves as it did before the
no-run work.

The no-run work (no Step 2 file without a run, the no-run box, the first
run, the lock handling) only changes what happens when there is no active
run, when a save or a write is refused, or when the run list is locked.  An
existing project with an active run -- every user so far -- must see the
same questions, the same closes / clears / reloads, the same saves, the same
files and the same words on every other path.

This file drives a two-run project ("cells" active, "nuclei" beside it)
through a run switch (without and with a pending drawing), Accept, Finalize
accepted, the start of Train, leaving the workspace (Browse) and closing,
and compares what happened with the values the same script recorded on the
code as it stood before the no-run work (the literals in ``EXPECTED``; the
script was run on an extract of that code and its printout pasted here).
Four differences, listed in ``CHANGED_ON_PURPOSE`` with the value before,
the value now and why, so none is hidden: two that change what happens --
leaving a workspace (Browse) now saves it first, and the busy line of a
switch has one ellipsis character -- and two cosmetic ones: the save now
returns what it did (it returned nothing), and the save before training
prints that line instead of the old fixed one.

Qt-free: the controller is built with ``__new__`` and plain fakes; the
switch question is recorded (answered yes), so no test builds a modal.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from napari_epicycle.hitl_training_annotation import _paths as P
from napari_epicycle.hitl_training_annotation import _run_management as RM
from napari_epicycle.hitl_training_annotation import _switch_run_dialog as D
from napari_epicycle.hitl_training_annotation import _switch_run_guard as G
from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)
from napari_epicycle.hitl_training_annotation.controller.training_manager import (
    TrainingManager,
)
from napari_epicycle.io_registry import HITLPersistenceService


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _Item:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _ListWidget:
    def __init__(self):
        self._items = []

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


class _Log:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


class _Button:
    def setEnabled(self, v):
        pass


class _Combo:
    def __init__(self):
        self.items = []

    def blockSignals(self, on):
        pass

    def clear(self):
        self.items = []

    def addItem(self, text, data=None):
        self.items.append((text, data))

    def insertItem(self, i, text, data=None):
        self.items.insert(i, (text, data))

    def count(self):
        return len(self.items)

    def itemText(self, i):
        return self.items[i][0]

    def insertSeparator(self, i):
        pass

    def setCurrentIndex(self, i):
        pass

    def setEnabled(self, v):
        pass


class _Bar:
    def setRange(self, *_a):
        pass

    def setValue(self, _v):
        pass


class _UI:
    def __init__(self):
        self.training_log_output = _Log()
        self.status_run_value = _Label()
        self.loaded_patch_label = _Label("Patch: stack01")
        self.run_picker_combo = _Combo()
        for name in ("new_run_button", "import_pairs_button",
                     "import_external_button", "manage_runs_button"):
            setattr(self, name, _Button())
        self.data_to_add_list = _ListWidget()
        self.segment_slice_data_list = _ListWidget()
        self.accepted_queue_list = _ListWidget()
        self.staged_queue_list = _ListWidget()
        self.sample_queue_list = self.accepted_queue_list
        self.training_progress_bar = _Bar()


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


class _Stop(Exception):
    pass


def _noop(*_a, **_k):
    return None


_QUIET = (
    "_sync_data_to_add_list", "_refresh_sample_queue", "_refresh_round_status",
    "_refresh_annotator_edit_state_visuals", "_schedule_state_save",
    "_exit_refine_mode", "_refresh_patch_list_badges",
    "_populate_patch_list_from_disk", "_refresh_model_source_controls",
    "_apply_model_source_selection", "_apply_training_mode_for_active_bucket",
    "_set_axial_combo_enabled", "_sync_training_controls_from_config",
    "_sync_inference_controls_from_config", "_refresh_current_config_from_widgets",
)

LABEL = "stack01_XY_12"


def _project(ws):
    """Two runs on disk, "cells" active; a controller pointed at it with a
    patch open, one slice and recorders on the switch's steps."""
    for slug in ("cells", "nuclei"):
        RM.create_run_on_disk(ws, slug, display_name=slug)
    RM.add_run_to_state_json(ws, "cells", activate=True)
    RM.add_run_to_state_json(ws, "nuclei")
    c = TrainingAnnotationController.__new__(TrainingAnnotationController)
    c.persistence = HITLPersistenceService()
    c.training_manager = TrainingManager()
    c.segmentation_manager = _SegManager()
    c.working_dir = str(ws)
    c.current_round = 0
    c.current_config = {"model_name": "epi", "model_type": "cyto3"}
    c.training_data_by_round = {}
    c.losses_by_round = {}
    c.model_checkpoint_paths = {}
    c.round_backend_info = {}
    c.loaded_patch_name = "stack01"
    c.loaded_stack_path = str(Path(ws) / "extractions")
    c.selected_view = "XY"
    c.view_target = "lateral"
    c.axial_plane = "XZ"
    c.sample_records = {}
    c.training_mode = "cumulative"
    c.active_curation_checkpoints = {"lateral": "", "axial": ""}
    c.curation_model_source_mode = "auto"
    c.imported_curation_model_path = ""
    c.curation_pretrained_model_type = "cyto3"
    c._use_sync_training = True
    c.active_run_id = "cells"
    c._run_switch_in_flight = False
    c._patch_load_token = 0
    c.ui = _UI()
    c.viewer = SimpleNamespace(layers=_Layers())
    for name in _QUIET:
        setattr(c, name, _noop)
    c.events = []
    c.told = []
    c._ask_for_run = lambda *a: c.events.append(("asked",) + a) or False
    c._tell_step2 = lambda title, sentence: c.told.append((title, sentence))
    c._begin_busy = lambda message="", cursor=True: c.events.append(
        ("busy", message))
    c._end_busy = _noop
    c._close_step2_annotator_dock_safely = lambda: c.events.append(("close",))
    c._clear_patch_viewer_layers_safely = lambda: c.events.append(("clear",))
    c._auto_reload_saved_patch = lambda: c.events.append(("reload-patch",))
    c._launch_multiview_annotator = lambda: c.events.append(("reopen",))
    real_save = c._save_internal_state

    def _save():
        result = real_save()
        c.events.append(("save", result))
        return result

    c._save_internal_state = _save
    # the slice: an image and a drawn mask on the labels layer
    image = np.random.RandomState(0).rand(2, 16, 16).astype(np.float32)
    c.segmentation_manager.data_inputs[LABEL] = image
    c.segmentation_manager.data_masks[LABEL] = np.ones((16, 16), np.uint16)
    volume = np.zeros((20, 16, 16), dtype=np.int32)
    volume[12, 2:6, 2:6] = 1
    c._parse_all_selected_sample_ids = lambda: [LABEL]
    c._work_labels_layer_name_for_view = lambda **_k: "work"
    c._layer_by_name = lambda _n: _Layer(volume)
    c._get_patch_view_image_layers = lambda **_k: [_Layer(np.zeros((20, 16, 16)))]
    c.curation_sessions = SimpleNamespace(record_accept=_noop)
    c.interactive_manager = SimpleNamespace(segmenter=None)
    c.segmented_slice_list = set()
    return c


def _files(ws):
    ws = Path(ws)
    out = []
    for p in ws.rglob("*"):
        if p.is_file() and "thumbs" not in p.parts:
            out.append(str(p.relative_to(ws)).replace("\\", "/"))
    return sorted(out)


_STAMP = re.compile(r"\d{4}-\d\d-\d\dT[0-9:.+]+|\d{8}T\d{6}\d*Z?")


def _norm(text, ws):
    text = str(text).replace(str(ws), "<ws>").replace("\\", "/")
    return _STAMP.sub("<time>", text)


_PREFIXES = ("[STEP2-RUN]", "[STATE-SAVE]", "[STEP2-WRITE]", "[TRAIN]",
             "[WORKSPACE-SWITCH]", "[TRAIN-SAFETY]")


def _scene(name, setup, act):
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        with contextlib.redirect_stdout(io.StringIO()):   # setup's own lines
            c = _project(ws)
            setup(c)
        before = set(_files(ws))
        out = io.StringIO()
        error = None
        with contextlib.redirect_stdout(out):
            try:
                act(c)
            except _Stop:
                pass
            except Exception as exc:      # recorded: a difference to see
                error = f"{type(exc).__name__}: {exc}"
        after = _files(ws)
        state = json.loads((ws / "step2_state.json").read_text("utf-8"))
        return {
            "events": [_norm(" | ".join(map(str, e)), ws) for e in c.events],
            "new_files": sorted(set(after) - before),
            "gone_files": sorted(before - set(after)),
            "active_on_disk": state.get("active_run_id"),
            "active": c.active_run_id,
            "records": {k: v.get("status") for k, v in
                        (c.sample_records or {}).items()},
            "told": [[_norm(t, ws), _norm(s, ws)] for t, s in c.told],
            "log": [_norm(x, ws) for x in c.ui.training_log_output.lines],
            "console": [_norm(x, ws) for x in out.getvalue().splitlines()
                        if x.startswith(_PREFIXES)],
            "error": error,
        }


def _switch_setup(drawing):
    def setup(c):
        G.check_switch_safety = lambda _c, _v: (
            G.SwitchSafety.NEEDS_CONFIRMATION, ["annotator open"])
        G._annotator_dock_open = lambda _v: True

        def _confirm(**kw):
            c.events.append(("confirm", kw["old_run_id"], kw["new_run_id"],
                             kw.get("old_display_name"),
                             kw.get("new_display_name")))
            return True

        D.show_switch_confirm = _confirm
        if drawing:
            c._touch_sample(LABEL, status="curating", view="XY")
            c._curating_persist_pending = {LABEL: (LABEL, "XY", 12)}
    return setup


def _accept_setup(c):
    c._touch_sample(LABEL, status="curating", view="XY")


def _finalize_setup(c):
    c._touch_sample(LABEL, status="curating", view="XY")
    c._accept_annotation()
    c.events.clear()
    c.ui.training_log_output.lines.clear()


def _train_setup(c):
    _finalize_setup(c)
    c._stage_accepted_in_current_view()
    c.events.clear()
    c.ui.training_log_output.lines.clear()
    c._collect_training_config = lambda: {"model_name": "epi", "epochs": 1}
    c._training_thread = None

    def _went_on():
        c.events.append(("training goes on",))
        raise _Stop

    c._sync_data_to_add_list = _went_on


def _leave_setup(c):
    c._touch_sample(LABEL, status="curating", view="XY")


def drive():
    """Every scene, as a dict of plain values."""
    real = (G.check_switch_safety, G._annotator_dock_open, D.show_switch_confirm)
    try:
        return {
            "switch": _scene("switch", _switch_setup(False),
                             lambda c: c._switch_active_run("nuclei")),
            "switch with a drawing": _scene(
                "switch with a drawing", _switch_setup(True),
                lambda c: c._switch_active_run("nuclei")),
            "accept": _scene("accept", _accept_setup,
                             lambda c: c._accept_annotation()),
            "finalize": _scene("finalize", _finalize_setup,
                               lambda c: c._stage_accepted_in_current_view()),
            "train start": _scene("train start", _train_setup,
                                  lambda c: c._train_model()),
            "browse away": _scene("browse away", _leave_setup,
                                  lambda c: c._teardown_previous_workspace()),
            "close": _scene("close", _leave_setup, lambda c: c.shutdown()),
        }
    finally:
        G.check_switch_safety, G._annotator_dock_open, D.show_switch_confirm = real


# ---------------------------------------------------------------------------
# the values before the no-run work (pasted from a run of ``drive()`` on an
# extract of that code)
# ---------------------------------------------------------------------------

EXPECTED = json.loads(r'''
{
 "switch": {
  "events": [
   "confirm | cells | nuclei | cells | nuclei",
   "busy | [STEP2-RUN] Switching to 'nuclei' ...",
   "save | None",
   "close",
   "clear",
   "reload-patch"
  ],
  "new_files": [],
  "gone_files": [],
  "active_on_disk": "nuclei",
  "active": "nuclei",
  "records": {},
  "told": [],
  "log": [],
  "console": [
   "[STATE-SAVE] Phase B.1 derivation: lateral_round_idx=0 (from counter=0-1), axial_round_idx=0 (from counter=0-1), active_run='cells'"
  ],
  "error": null
 },
 "switch with a drawing": {
  "events": [
   "confirm | cells | nuclei | cells | nuclei",
   "busy | [STEP2-RUN] Switching to 'nuclei' ...",
   "save | None",
   "close",
   "clear",
   "reload-patch"
  ],
  "new_files": [
   "artifacts/step2/runs/cells/lateral/round_00/accepted/stack01_XY_z0012_img.tif",
   "artifacts/step2/runs/cells/lateral/round_00/accepted/stack01_XY_z0012_mask.tif"
  ],
  "gone_files": [],
  "active_on_disk": "nuclei",
  "active": "nuclei",
  "records": {},
  "told": [],
  "log": [],
  "console": [
   "[STEP2-WRITE] _persist_curated_training_pair -> NEW layout: XY -> lateral bucket / run='cells' / round_00",
   "[STATE-SAVE] Phase B.1 derivation: lateral_round_idx=0 (from counter=0-1), axial_round_idx=0 (from counter=0-1), active_run='cells'"
  ],
  "error": null
 },
 "accept": {
  "events": [],
  "new_files": [
   "artifacts/step2/runs/cells/lateral/round_00/accepted/stack01_XY_z0012_img.tif",
   "artifacts/step2/runs/cells/lateral/round_00/accepted/stack01_XY_z0012_mask.tif"
  ],
  "gone_files": [],
  "active_on_disk": "cells",
  "active": "cells",
  "records": {
   "stack01_XY_12": "accepted"
  },
  "told": [],
  "log": [
   "[CURATION] Accepted 1 slice(s). Finalize to add to training."
  ],
  "console": [
   "[STEP2-WRITE] _persist_curated_training_pair -> NEW layout: XY -> lateral bucket / run='cells' / round_00"
  ],
  "error": null
 },
 "finalize": {
  "events": [],
  "new_files": [],
  "gone_files": [],
  "active_on_disk": "cells",
  "active": "cells",
  "records": {
   "stack01_XY_12": "staged"
  },
  "told": [],
  "log": [
   "[QUEUE] Staged 1 accepted sample(s) in lateral."
  ],
  "console": [
   "[STEP2-WRITE] _persist_curated_training_pair -> NEW layout: XY -> lateral bucket / run='cells' / round_00"
  ],
  "error": null
 },
 "train start": {
  "events": [
   "save | None",
   "training goes on"
  ],
  "new_files": [],
  "gone_files": [],
  "active_on_disk": "cells",
  "active": "cells",
  "records": {
   "stack01_XY_12": "staged"
  },
  "told": [],
  "log": [],
  "console": [
   "[STATE-SAVE] Phase B.1 derivation: lateral_round_idx=0 (from counter=0-1), axial_round_idx=0 (from counter=0-1), active_run='cells'",
   "[TRAIN-SAFETY] F2.1 explicit state save after flush",
   "[TRAIN-SAFETY] F2.1 flushed pending state/persist timers + saved + drained"
  ],
  "error": null
 },
 "browse away": {
  "events": [
   "close",
   "clear"
  ],
  "new_files": [],
  "gone_files": [],
  "active_on_disk": "cells",
  "active": "cells",
  "records": {
   "stack01_XY_12": "curating"
  },
  "told": [],
  "log": [],
  "console": [
   "[WORKSPACE-SWITCH] tearing down previous workspace before load: prev_patch='stack01'",
   "[WORKSPACE-SWITCH] annotator dock + stale patch layers cleared",
   "[WORKSPACE-SWITCH] segmenter reset (next segment rebuilds from new workspace config)"
  ],
  "error": null
 },
 "close": {
  "events": [
   "save | None"
  ],
  "new_files": [],
  "gone_files": [],
  "active_on_disk": "cells",
  "active": "cells",
  "records": {
   "stack01_XY_12": "curating"
  },
  "told": [],
  "log": [],
  "console": [
   "[STATE-SAVE] Phase B.1 derivation: lateral_round_idx=0 (from counter=0-1), axial_round_idx=0 (from counter=0-1), active_run='cells'"
  ],
  "error": null
 }
}
''')

#: where the behaviour of an existing project changed on purpose:
#: (scene, key) -> (value before, value now, why)
CHANGED_ON_PURPOSE = {
    ('switch', 'events'): (
        ['confirm | cells | nuclei | cells | nuclei', "busy | [STEP2-RUN] Switching to 'nuclei' ...", 'save | None', 'close', 'clear', 'reload-patch'],
        ['confirm | cells | nuclei | cells | nuclei', "busy | [STEP2-RUN] Switching to 'nuclei' …", 'save | saved-v3', 'close', 'clear', 'reload-patch'],
        'the busy line has one ellipsis character, like the other busy lines; the save now says what it did (it returned nothing): a value, not a behaviour'),
    ('switch with a drawing', 'events'): (
        ['confirm | cells | nuclei | cells | nuclei', "busy | [STEP2-RUN] Switching to 'nuclei' ...", 'save | None', 'close', 'clear', 'reload-patch'],
        ['confirm | cells | nuclei | cells | nuclei', "busy | [STEP2-RUN] Switching to 'nuclei' …", 'save | saved-v3', 'close', 'clear', 'reload-patch'],
        'the busy line has one ellipsis character, like the other busy lines; the save now says what it did (it returned nothing): a value, not a behaviour'),
    ('train start', 'events'): (
        ['save | None', 'training goes on'],
        ['save | saved-v3', 'training goes on'],
        'the save now says what it did (it returned nothing): a value, not a behaviour'),
    ('train start', 'console'): (
        ["[STATE-SAVE] Phase B.1 derivation: lateral_round_idx=0 (from counter=0-1), axial_round_idx=0 (from counter=0-1), active_run='cells'", '[TRAIN-SAFETY] F2.1 explicit state save after flush', '[TRAIN-SAFETY] F2.1 flushed pending state/persist timers + saved + drained'],
        ["[STATE-SAVE] Phase B.1 derivation: lateral_round_idx=0 (from counter=0-1), axial_round_idx=0 (from counter=0-1), active_run='cells'", '[STATE-SAVE] saved before training', '[TRAIN-SAFETY] F2.1 flushed pending state/persist timers + saved + drained'],
        'the save before training prints what it did ("saved before training") instead of the old fixed line'),
    ('browse away', 'events'): (
        ['close', 'clear'],
        ['save | saved-v3', 'close', 'clear'],
        'leaving a workspace saves it first (a pending save used to be stopped, not fired), so a refused save can be rescued'),
    ('browse away', 'console'): (
        ["[WORKSPACE-SWITCH] tearing down previous workspace before load: prev_patch='stack01'", '[WORKSPACE-SWITCH] annotator dock + stale patch layers cleared', '[WORKSPACE-SWITCH] segmenter reset (next segment rebuilds from new workspace config)'],
        ["[STATE-SAVE] Phase B.1 derivation: lateral_round_idx=0 (from counter=0-1), axial_round_idx=0 (from counter=0-1), active_run='cells'", "[WORKSPACE-SWITCH] tearing down previous workspace before load: prev_patch='stack01'", '[WORKSPACE-SWITCH] annotator dock + stale patch layers cleared', '[WORKSPACE-SWITCH] segmenter reset (next segment rebuilds from new workspace config)'],
        "the save's own console line, from the save above"),
    ('close', 'events'): (
        ['save | None'],
        ['save | saved-v3'],
        'the save now says what it did (it returned nothing): a value, not a behaviour'),
}


def _expected_now():
    want = json.loads(json.dumps(EXPECTED))
    for (scene, key), (_before, now, _why) in CHANGED_ON_PURPOSE.items():
        want[scene][key] = now
    return want


def test_an_existing_project_behaves_as_before():
    got = drive()
    want = _expected_now()
    for scene in want:
        for key in want[scene]:
            assert got[scene][key] == want[scene][key], (scene, key)


def test_every_purposeful_change_still_differs_from_before():
    """A listed change that no longer differs is not a change: drop it."""
    for (scene, key), (before, now, why) in CHANGED_ON_PURPOSE.items():
        assert before != now, (scene, key, why)
        assert EXPECTED[scene][key] == before, (scene, key)


if __name__ == "__main__":
    json.dump(drive(), sys.stdout, indent=1, ensure_ascii=False)

"""Step 2 on a fresh workspace: no run, no writes.

The rule under test, in plain words:

* A folder with no Step 2 run gets NO Step 2 file from a save, a pair write
  or a training round -- never the old layout's ``hitl_state.json``,
  ``training_data/``, ``models/``, ``predictions/`` or
  ``artifacts/models|inference/`` (only the patch thumbnails are drawn).
* A folder that already holds old-layout data keeps its old-layout writes
  until it is migrated.
* A folder with an active run writes the new layout only.
* A run list Step 2 cannot read is never overwritten, and nothing is saved
  beside it.
* An error never sends a write into an old-layout folder: it refuses and
  the data stays in memory (at close, the unsaved state goes to a rescue
  file inside ``artifacts/step2/rescue/``).

Qt-free: the controller is built with ``__new__`` and plain fakes.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from napari_epicycle.io_registry import HITLPersistenceService
from napari_epicycle.ui_shell import _workspace_facts


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _Item:
    def __init__(self, text):
        self._text = text
        self._data = {}

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text

    def setForeground(self, brush):
        pass

    def setData(self, role, value):
        self._data[role] = value

    def data(self, role):
        return self._data.get(role)

    def setIcon(self, icon):
        pass


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


class _ProgressBar:
    def __init__(self):
        self.value = 0

    def setRange(self, _a, _b):
        pass

    def setValue(self, value):
        self.value = value


class _LogOutput:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


class _Label:
    def __init__(self):
        self.value = ""

    def setText(self, text):
        self.value = text


class _Button:
    def __init__(self):
        self._enabled = True

    def setEnabled(self, v):
        self._enabled = v

    def isEnabled(self):
        return self._enabled


class _SegManager:
    """The parts of the segmentation manager the writers read."""

    def __init__(self):
        self.data_inputs = {}
        self.data_masks = {}
        self.data_names = {}
        self.data_inputs_rgb = {}
        self.segmenter = object()

    def set(self, label, inp, mask):
        self.data_inputs[label] = inp
        self.data_masks[label] = mask

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


class _Viewer:
    def __init__(self):
        self.layers = _Layers()


class _UI:
    def __init__(self):
        self.data_to_add_list = _ListWidget()
        self.segment_slice_data_list = _ListWidget()
        self.training_progress_bar = _ProgressBar()
        self.training_log_output = _LogOutput()
        self.running_training_loss_plot = _Label()
        self.update_and_train_button = _Button()
        self.accepted_queue_list = _ListWidget()
        self.staged_queue_list = _ListWidget()
        self.sample_queue_list = self.accepted_queue_list


def _noop(*_a, **_k):
    return None


_QUIET = (
    "_sync_data_to_add_list", "_refresh_sample_queue", "_refresh_round_status",
    "_refresh_annotator_edit_state_visuals", "_refresh_run_picker",
    "_schedule_state_save", "_begin_busy", "_end_busy", "_exit_refine_mode",
)


def _ctrl(ws, *, run=None, layout=None, quiet=True):
    """A controller on folder ``ws``.

    ``run``: make that run on disk, activate it and point the controller at it.
    ``layout="legacy"``: give the folder an old-layout ``training_data/``.
    """
    ws = Path(ws)
    if layout == "legacy":
        (ws / "training_data").mkdir(parents=True, exist_ok=True)
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.persistence = HITLPersistenceService()
    ctrl.training_manager = TrainingManager()
    ctrl.segmentation_manager = _SegManager()
    ctrl.working_dir = str(ws)
    ctrl.current_round = 0
    ctrl.current_config = {"model_name": "epi", "model_type": "cyto3", "epochs": 120}
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
    ctrl.viewer = _Viewer()
    if quiet:
        for name in _QUIET:
            setattr(ctrl, name, _noop)
    if run:
        _make_run(ws, run)
        ctrl.active_run_id = run
    return ctrl


def _make_run(ws, slug):
    RM.create_run_on_disk(ws, slug, display_name=slug)
    RM.add_run_to_state_json(ws, slug, activate=True)


def _files_outside_thumbs(ws):
    ws = Path(ws)
    thumbs = ws / "artifacts" / "step2" / "thumbs"
    return sorted(
        str(p.relative_to(ws)) for p in ws.rglob("*")
        if p.is_file() and thumbs not in p.parents
    )


_LEGACY_MARKS = (
    "hitl_state.json", "training_data", "models", "predictions",
    "artifacts/models", "artifacts/inference",
)


def _legacy_marks(ws):
    return [m for m in _LEGACY_MARKS if (Path(ws) / m).exists()]


def _add_slice(ctrl, label="stack01_XY_12", status=None, shape=(16, 16)):
    ctrl.segmentation_manager.set(
        label,
        np.random.rand(2, *shape).astype(np.float32),
        np.ones(shape, dtype=np.uint16),
    )
    if status:
        ctrl._touch_sample(label, status=status, view=label.rsplit("_", 2)[1])
    return label


def _raise(*_a, **_k):
    raise OSError("disk went away")


# ---------------------------------------------------------------------------
# the predicate
# ---------------------------------------------------------------------------

def test_fresh_folder_target_is_none(tmp_path):
    assert _ctrl(tmp_path)._step2_write_target() is None


def test_no_workspace_target_is_none(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl.working_dir = ""
    assert ctrl._step2_write_target() is None


def test_old_folder_target_is_legacy(tmp_path):
    assert _ctrl(tmp_path, layout="legacy")._step2_write_target() == "legacy"


def test_stray_hitl_state_alone_is_fresh(tmp_path):
    (tmp_path / "hitl_state.json").write_text("{}", encoding="utf-8")
    assert _ctrl(tmp_path)._step2_write_target() is None


def test_run_folder_target_is_run(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    assert ctrl._step2_write_target() == "run"
    assert ctrl.active_run_id == "pilot"


def test_run_folder_without_active_run_is_none(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl.active_run_id = ""
    assert ctrl._step2_write_target() is None


def test_active_run_without_folder_is_none(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl.active_run_id = "gone"
    assert ctrl._step2_write_target() is None


def test_corrupt_state_is_unreadable(tmp_path):
    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    assert _ctrl(tmp_path)._step2_write_target() == "unreadable"


def test_state_without_a_runs_table_is_unreadable(tmp_path):
    (tmp_path / "step2_state.json").write_text('{"runs": []}', encoding="utf-8")
    assert _ctrl(tmp_path)._step2_write_target() == "unreadable"


def test_runs_without_state_file(tmp_path):
    (tmp_path / "artifacts" / "step2" / "runs" / "x").mkdir(parents=True)
    assert _ctrl(tmp_path)._step2_write_target() == "unreadable"


def test_predicate_error_never_means_legacy(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, layout="legacy")
    monkeypatch.setattr(P, "is_new_layout_state", _raise)
    assert ctrl._step2_write_target() is None


def test_predicate_reads_the_disk_live(tmp_path):
    ctrl = _ctrl(tmp_path)
    assert ctrl._step2_write_target() is None
    _make_run(tmp_path, "pilot")
    ctrl.active_run_id = "pilot"
    assert ctrl._step2_write_target() == "run"
    (tmp_path / "step2_state.json").write_text("{oops", encoding="utf-8")
    assert ctrl._step2_write_target() == "unreadable"


# ---------------------------------------------------------------------------
# an unreadable run list is never overwritten
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("call", ["add", "activate", "remove"])
def test_corrupt_state_never_overwritten(tmp_path, call):
    state = tmp_path / "step2_state.json"
    state.write_bytes(b"{oops")
    with pytest.raises(Exception):
        if call == "add":
            RM.add_run_to_state_json(tmp_path, "pilot", activate=True)
        elif call == "activate":
            RM.set_active_run_in_state_json(tmp_path, "pilot")
        else:
            RM.remove_run_from_state_json(tmp_path, "pilot")
    assert state.read_bytes() == b"{oops"
    copies = list(tmp_path.glob("step2_state.json.corrupt-*"))
    assert len(copies) == 1
    assert copies[0].read_bytes() == b"{oops"


def test_readable_state_still_gains_a_run(tmp_path):
    _make_run(tmp_path, "pilot")
    RM.create_run_on_disk(tmp_path, "second", display_name="second")
    RM.add_run_to_state_json(tmp_path, "second")
    data = json.loads((tmp_path / "step2_state.json").read_text(encoding="utf-8"))
    assert set(data["runs"]) == {"pilot", "second"}
    assert data["active_run_id"] == "pilot"


# ---------------------------------------------------------------------------
# active_run_id at load
# ---------------------------------------------------------------------------

def test_fresh_load_sets_empty_run_id(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl.active_run_id = "left-over"
    ctrl._load_internal_state()
    assert ctrl.active_run_id == ""
    assert _files_outside_thumbs(tmp_path) == []


def test_old_folder_load_keeps_the_default_run_id(tmp_path):
    ctrl = _ctrl(tmp_path, layout="legacy")
    ctrl._load_internal_state()
    assert ctrl.active_run_id == P.DEFAULT_RUN_ID


def test_missing_active_folder_loads_empty_run_id(tmp_path):
    (tmp_path / "step2_state.json").write_text(json.dumps({
        "schema_version": 3, "active_run_id": "gone", "runs": {"gone": {}},
    }), encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    ctrl._load_internal_state()
    assert ctrl.active_run_id == ""
    assert ctrl._step2_write_target() is None


def test_run_folder_load_keeps_its_run_id(tmp_path):
    _make_run(tmp_path, "pilot")
    ctrl = _ctrl(tmp_path)
    ctrl._load_internal_state()
    assert ctrl.active_run_id == "pilot"


def test_old_folder_load_writes_no_new_layout(tmp_path):
    ctrl = _ctrl(tmp_path, layout="legacy")
    ctrl._load_internal_state()
    assert not (tmp_path / "step2_state.json").exists()
    assert not (tmp_path / "artifacts" / "step2" / "runs").exists()


# ---------------------------------------------------------------------------
# saves
# ---------------------------------------------------------------------------

def test_fresh_save_writes_nothing(tmp_path):
    ctrl = _ctrl(tmp_path)
    assert ctrl._save_internal_state() == "no-run"
    assert _files_outside_thumbs(tmp_path) == []


def test_unreadable_save_writes_nothing(tmp_path):
    state = tmp_path / "step2_state.json"
    state.write_bytes(b"{oops")
    ctrl = _ctrl(tmp_path)
    assert ctrl._save_internal_state() == "unreadable"
    assert ctrl._save_internal_state() == "unreadable"
    assert state.read_bytes() == b"{oops"
    # one byte copy set aside, however many saves are refused
    files = _files_outside_thumbs(tmp_path)
    assert files[0] == "step2_state.json" and len(files) == 2
    assert files[1].startswith("step2_state.json.corrupt-")
    assert (tmp_path / files[1]).read_bytes() == b"{oops"


def test_run_save_writes_the_new_layout(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl.sample_records = {"stack01_XY_12": {"status": "curating"}}
    assert ctrl._save_internal_state() == "saved-v3"
    data = json.loads((tmp_path / "step2_state.json").read_text(encoding="utf-8"))
    assert "stack01_XY_12" in data["runs"]["pilot"]["sample_records"]
    assert _legacy_marks(tmp_path) == []


def test_old_folder_save_keeps_the_old_layout(tmp_path):
    ctrl = _ctrl(tmp_path, layout="legacy")
    assert ctrl._save_internal_state() == "saved-legacy"
    assert (tmp_path / "hitl_state.json").is_file()
    assert not (tmp_path / "step2_state.json").exists()


def test_no_run_line_printed_once(tmp_path, capsys):
    ctrl = _ctrl(tmp_path)
    for _ in range(3):
        ctrl._debounced_state_save()
    out = capsys.readouterr().out
    assert out.count("[STATE-SAVE] no run yet: nothing saved") == 1
    assert "saved hitl_state.json" not in out


def test_shutdown_on_fresh_folder_writes_nothing(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl._load_internal_state()
    ctrl.shutdown()
    assert _files_outside_thumbs(tmp_path) == []


def _break_v3_write(monkeypatch, message="the disk is full"):
    from napari_epicycle.hitl_training_annotation.controller import (
        training_annotation_controller as tac,
    )
    real = tac._atomic_write_json
    calls = {"n": 0}

    def _write(path, payload, *a, **k):
        if Path(path).name == P.STEP2_STATE_FILENAME:
            calls["n"] += 1
            raise OSError(message)
        return real(path, payload, *a, **k)

    monkeypatch.setattr(tac, "_atomic_write_json", _write)
    return real, calls


def test_a_failed_save_is_said_once_per_error(tmp_path, monkeypatch, capsys):
    ctrl = _ctrl(tmp_path, run="pilot")
    before = (tmp_path / "step2_state.json").read_bytes()
    ctrl.sample_records = {"stack01_XY_12": {"status": "curating"}}
    real, calls = _break_v3_write(monkeypatch)
    for _ in range(3):
        ctrl._debounced_state_save()
    out = capsys.readouterr().out
    assert calls["n"] == 3
    assert out.count("could not save") == 1
    assert "the disk is full" in out
    assert (tmp_path / "step2_state.json").read_bytes() == before
    assert ctrl.sample_records == {"stack01_XY_12": {"status": "curating"}}
    assert _legacy_marks(tmp_path) == []
    # the disk is back: the next save goes through
    from napari_epicycle.hitl_training_annotation.controller import (
        training_annotation_controller as tac,
    )
    monkeypatch.setattr(tac, "_atomic_write_json", real)
    assert ctrl._save_internal_state() == "saved-v3"


def test_a_failed_save_at_close_writes_a_rescue_file(tmp_path, monkeypatch, capsys):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl.sample_records = {"stack01_XY_12": {"status": "curating"}}
    _break_v3_write(monkeypatch)
    ctrl.shutdown()
    rescues = list((tmp_path / "artifacts" / "step2" / "rescue").glob("*.json"))
    assert len(rescues) == 1
    payload = json.loads(rescues[0].read_text(encoding="utf-8"))
    assert "stack01_XY_12" in payload["sample_records"]
    assert _legacy_marks(tmp_path) == []
    assert P.detect_layout_version(tmp_path) == P.LAYOUT_VERSION_NEW
    assert "your state is in" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the pair funnel
# ---------------------------------------------------------------------------

def test_fresh_pair_write_refuses(tmp_path):
    ctrl = _ctrl(tmp_path)
    label = _add_slice(ctrl)
    assert ctrl._persist_curated_training_pair(label) == {}
    assert ctrl._last_pair_refusal == "no-run"
    assert _files_outside_thumbs(tmp_path) == []


def test_silent_bootstrap_is_gone():
    assert not hasattr(TrainingAnnotationController,
                       "_auto_bootstrap_default_run_if_fresh")


def test_run_pair_write_lands_in_the_run(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    label = _add_slice(ctrl, "stack01_XZ_9")
    out = ctrl._persist_curated_training_pair(label)
    acc = P.round_accepted_dir(tmp_path, "pilot", "axial", 0)
    assert out["img_uri"] == str(acc / "stack01_XZ_z0009_img.tif")
    assert (acc / "stack01_XZ_z0009_mask.tif").is_file()
    assert ctrl._last_pair_refusal is None
    assert _legacy_marks(tmp_path) == []


def test_old_folder_pair_write_keeps_the_old_layout(tmp_path):
    ctrl = _ctrl(tmp_path, layout="legacy")
    label = _add_slice(ctrl)
    out = ctrl._persist_curated_training_pair(label)
    assert "training_data" in out["img_uri"]
    assert not (tmp_path / "step2_state.json").exists()


def test_a_pair_write_error_never_goes_to_the_old_layout(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    label = _add_slice(ctrl)
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    assert ctrl._persist_curated_training_pair(label) == {}
    assert ctrl._last_pair_refusal == "folder"
    assert _legacy_marks(tmp_path) == []
    assert "img_uri" not in ctrl.sample_records.get(label, {})


def test_a_refused_finalize_puts_the_status_back(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    label = _add_slice(ctrl, status="accepted")
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    ctrl._stage_accepted_in_current_view()
    assert ctrl.sample_records[label]["status"] == "accepted"
    assert _legacy_marks(tmp_path) == []


def test_a_refused_accept_puts_the_status_back(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    label = _add_slice(ctrl, status="curating")
    volume = np.zeros((20, 16, 16), dtype=np.int32)
    volume[12, 2:6, 2:6] = 1
    ctrl._parse_all_selected_sample_ids = lambda: [label]
    ctrl._work_labels_layer_name_for_view = lambda **_k: "work"
    ctrl._layer_by_name = lambda _n: _Layer(volume)
    ctrl._get_patch_view_image_layers = lambda **_k: [_Layer(np.zeros((20, 16, 16)))]

    class _Sessions:
        def record_accept(self, **_k):
            pass

    ctrl.curation_sessions = _Sessions()

    class _Interactive:
        segmenter = None

    ctrl.interactive_manager = _Interactive()
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    ctrl._accept_annotation()
    assert ctrl.sample_records[label]["status"] == "curating"
    assert _legacy_marks(tmp_path) == []


def test_a_refused_predict_puts_the_record_back(tmp_path, monkeypatch):
    """Predict keeps the prediction on screen; the record goes back to what
    it was before the predict wrote it (here: no record at all)."""
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
    seg = ctrl.segmentation_manager

    def _segment_image(inp, _cfg, lbl):
        seg.data_masks[lbl] = np.ones((16, 16), dtype=np.int32)
        return seg.data_masks[lbl], None, None

    seg.segment_image = _segment_image

    class _ImageManager:
        def convert_to_rgb(self, **_k):
            return np.zeros((16, 16, 3))

    ctrl.image_manager = _ImageManager()
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    ctrl._run_segmentation()
    assert label not in ctrl.sample_records
    # the cached mask goes back too (what is drawn in the viewer stays)
    assert label not in seg.data_masks
    assert _legacy_marks(tmp_path) == []


def test_a_refused_annotator_accept_is_not_ok(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    volume = np.zeros((20, 16, 16), dtype=np.int32)
    volume[12, 2:6, 2:6] = 1
    ctrl._work_labels_layer_name_for_view = lambda **_k: "work"
    ctrl._layer_by_name = lambda _n: _Layer(volume)
    ctrl._get_patch_view_image_layers = lambda **_k: [_Layer(np.zeros((20, 16, 16)))]
    ctrl.segmented_slice_list = set()
    label = "stack01_XY_12"
    ctrl._touch_sample(label, status="curating", view="XY")
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    result = ctrl._accept_native_annotation("XY", 12)
    assert result["ok"] is False
    assert ctrl.sample_records[label]["status"] == "curating"
    assert _legacy_marks(tmp_path) == []


def test_fresh_annotator_accept_returns_not_ok(tmp_path):
    ctrl = _ctrl(tmp_path)
    volume = np.zeros((20, 16, 16), dtype=np.int32)
    volume[12, 2:6, 2:6] = 1
    ctrl._work_labels_layer_name_for_view = lambda **_k: "work"
    ctrl._layer_by_name = lambda _n: _Layer(volume)
    ctrl._get_patch_view_image_layers = lambda **_k: [_Layer(np.zeros((20, 16, 16)))]
    ctrl.segmented_slice_list = set()
    result = ctrl._accept_native_annotation("XY", 12)
    assert result["ok"] is False
    assert result["refusal"] == "no-run"        # the run gate, not a folder error
    assert "stack01_XY_12" not in ctrl.sample_records
    assert _files_outside_thumbs(tmp_path) == []


# ---------------------------------------------------------------------------
# training decides from one main-thread snapshot
# ---------------------------------------------------------------------------

def _stage_for_training(ctrl, label="stack01_XY_12"):
    _add_slice(ctrl, label)
    ctrl._persist_curated_training_pair(label)
    ctrl._touch_sample(label, status="staged", view="XY")
    ctrl.ui.data_to_add_list = _ListWidget([label])
    return label


def _record_errors(ctrl):
    errors = []
    ctrl._on_training_error = lambda msg: errors.append(str(msg))
    return errors


def test_a_round_without_a_run_trains_nothing(tmp_path):
    ctrl = _ctrl(tmp_path)
    _stage_for_training(ctrl)
    errors = _record_errors(ctrl)
    # Train asks for a run first; the worker's own refusal is the second
    # line of defence (a run gone between the ask and the launch).
    ctrl._require_run = lambda _action: True
    ctrl._train_model()
    assert len(errors) == 1
    assert "nothing was trained" in errors[0]
    assert _files_outside_thumbs(tmp_path) == []


def test_worker_ignores_live_layout_reads(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    _stage_for_training(ctrl)
    errors = _record_errors(ctrl)
    real = P.is_new_layout_state
    # the snapshot is taken on the main thread; every later read "fails"
    monkeypatch.setattr(ctrl, "_step2_write_target",
                        lambda: "run")
    monkeypatch.setattr(P, "is_new_layout_state", lambda _ws: False)
    ctrl._train_model()
    monkeypatch.setattr(P, "is_new_layout_state", real)
    assert errors == []
    assert P.model_dir(tmp_path, "pilot", "lateral", 0).is_dir()
    assert _legacy_marks(tmp_path) == []


def test_a_model_folder_error_trains_nothing(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    _stage_for_training(ctrl)
    errors = _record_errors(ctrl)
    monkeypatch.setattr(P, "model_dir", _raise)
    ctrl._train_model()
    assert errors and "No successful training view" in errors[0]
    assert _legacy_marks(tmp_path) == []


def test_a_model_folder_error_fails_only_that_view(tmp_path, monkeypatch):
    """One bucket's model folder fails: that view is recorded as failed at
    the "folder" stage and the other bucket still trains; nothing lands in
    the old ``models/`` / ``predictions/`` folders."""
    ctrl = _ctrl(tmp_path, run="pilot")
    xy = _stage_for_training(ctrl, "stack01_XY_12")
    xz = _stage_for_training(ctrl, "stack01_XZ_3")
    real = P.model_dir

    def _model_dir(ws, run_id, bucket, round_idx):
        if bucket == "lateral":
            raise OSError("disk went away")
        return real(ws, run_id, bucket, round_idx)

    monkeypatch.setattr(P, "model_dir", _model_dir)
    result = ctrl._run_training_work(
        progress_cb=_noop, round_id="round_00",
        staged_slice_labels=[xy, xz], training_mode="cumulative",
        prior_pairs=[], prior_checkpoints={},
        persisted_pairs={lbl: dict(ctrl.sample_records[lbl]) for lbl in (xy, xz)},
        config_snapshot=dict(ctrl.current_config),
        working_dir_snapshot=str(tmp_path),
        active_run_snapshot="pilot", write_target_snapshot="run",
        selected_view_snapshot="XY",
        current_round_by_view_snapshot={"lateral": 0, "axial": 0},
        loaded_patch_snapshot="stack01",
    )
    failed = result["round_backend"]["failed_views"]
    assert [f["stage"] for f in failed] == ["folder"]
    assert result["write_target"] == "run"
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "predictions").exists()


def test_a_run_round_writes_no_old_layout_json(tmp_path, monkeypatch):
    ctrl = _ctrl(tmp_path, run="pilot")
    monkeypatch.setattr(P, "is_new_layout_state", _raise)
    result = {
        "samples": [], "round_losses": {"spatial": [0.5]},
        "round_checkpoints": {"spatial": str(tmp_path / "ckpt")},
        "round_inference": {"spatial": str(tmp_path / "pred")},
        "round_backend": {"training": {}, "inference": {}, "failed_views": []},
        "is_new_layout": True, "write_target": "run",
        "used_slice_labels": [],
    }
    try:
        ctrl._on_training_finished(result, [], "round_00")
    except Exception:
        pass        # the round bookkeeping may need more fakes; the gate runs first
    assert not (tmp_path / "artifacts" / "models").exists()
    assert not (tmp_path / "artifacts" / "inference").exists()


def test_finished_without_a_write_target_writes_no_legacy_json(tmp_path):
    ctrl = _ctrl(tmp_path)
    result = {
        "samples": [], "round_losses": {"spatial": [0.5]},
        "round_checkpoints": {"spatial": str(tmp_path / "ckpt")},
        "round_inference": {"spatial": str(tmp_path / "pred")},
        "round_backend": {"training": {}, "inference": {}, "failed_views": []},
        "is_new_layout": False, "used_slice_labels": [],
    }
    try:
        ctrl._on_training_finished(result, [], "round_00")
    except Exception:
        pass
    assert not (tmp_path / "artifacts" / "models").exists()
    assert not (tmp_path / "artifacts" / "inference").exists()


def test_old_folder_training_keeps_the_old_layout(tmp_path):
    ctrl = _ctrl(tmp_path, layout="legacy")
    _stage_for_training(ctrl)
    errors = _record_errors(ctrl)
    ctrl._train_model()
    assert errors == []
    assert (tmp_path / "artifacts" / "inference").is_dir()
    assert not (tmp_path / "step2_state.json").exists()


# ---------------------------------------------------------------------------
# the walk: fresh folder -> first run -> save -> pair -> train
# ---------------------------------------------------------------------------

def _fresh_walk(tmp_path):
    """Yield after every step of the walk so a test can look at the disk."""
    ctrl = _ctrl(tmp_path)
    ctrl._load_internal_state()
    yield "load"
    ctrl._save_internal_state()
    yield "save before a run"
    # the first run, made the way New Run makes it
    _make_run(tmp_path, "pilot")
    ctrl.active_run_id = "pilot"
    yield "run"
    ctrl._save_internal_state()
    yield "save"
    label = _stage_for_training(ctrl)
    yield "pair"
    errors = _record_errors(ctrl)
    ctrl._train_model()
    assert errors == [], errors
    assert ctrl.sample_records[label]["status"] == "committed"
    yield "train"


def test_fresh_walk_writes_only_new_layout(tmp_path):
    for _ in _fresh_walk(tmp_path):
        pass
    run = tmp_path / "artifacts" / "step2" / "runs" / "pilot"
    rnd = run / "lateral" / "round_00"
    for path in (
        tmp_path / "step2_state.json",
        run / "run_manifest.json",
        rnd / "accepted" / "stack01_XY_z0012_img.tif",
        rnd / "accepted" / "stack01_XY_z0012_mask.tif",
        rnd / "round_metadata.json",
    ):
        assert path.is_file(), path
    assert (rnd / "model").is_dir()
    assert _legacy_marks(tmp_path) == []


def test_fresh_walk_layout_never_legacy(tmp_path):
    for step in _fresh_walk(tmp_path):
        assert P.detect_layout_version(tmp_path) in (
            P.LAYOUT_VERSION_UNKNOWN, P.LAYOUT_VERSION_NEW,
        ), step


def test_fresh_walk_banner_never_old_layout(tmp_path):
    for step in _fresh_walk(tmp_path):
        assert _workspace_facts.step2_layout_hint(str(tmp_path)) == "", step



# ---------------------------------------------------------------------------
# review fixes: what a refusal puts back, what a save never overwrites, the
# saves that act on their result, an unreadable run list said by its kind
# ---------------------------------------------------------------------------

from napari_epicycle.hitl_training_annotation.ui import _step2_text as T  # noqa: E402
import tifffile  # noqa: E402


def _told(ctrl):
    told = []
    ctrl._tell_step2 = lambda title, sentence: told.append(sentence)
    return told


def _log_text(ctrl):
    return "\n".join(ctrl.ui.training_log_output.lines)


def test_a_refused_re_accept_is_written_on_retry(tmp_path, monkeypatch):
    """An edited slice whose Accept was refused is written by the next
    Accept -- never skipped as "unchanged" while the disk holds the old mask."""
    ctrl = _ctrl(tmp_path, run="pilot")
    volume = np.zeros((20, 16, 16), dtype=np.int32)
    volume[12, 2:6, 2:6] = 1
    ctrl._work_labels_layer_name_for_view = lambda **_k: "work"
    ctrl._layer_by_name = lambda _n: _Layer(volume)
    ctrl._get_patch_view_image_layers = lambda **_k: [_Layer(np.zeros((20, 16, 16)))]
    ctrl.segmented_slice_list = set()
    assert ctrl._accept_native_annotation("XY", 12)["ok"] is True
    volume[12, 8:12, 8:12] = 2                                 # an edit
    real = P.round_accepted_dir
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    assert ctrl._accept_native_annotation("XY", 12)["ok"] is False
    monkeypatch.setattr(P, "round_accepted_dir", real)
    result = ctrl._accept_native_annotation("XY", 12)
    assert result["ok"] is True and not result.get("skipped")
    mask = tifffile.imread(str(P.round_accepted_dir(tmp_path, "pilot", "lateral", 0)
                               / "stack01_XY_z0012_mask.tif"))
    assert mask[8:12, 8:12].all()


def _predict_ready(ctrl, label, value):
    stack = np.random.rand(2, 16, 16).astype(np.float32)
    ctrl._apply_model_source_selection = lambda **_k: {"can_segment": True}
    ctrl._ensure_segmentation_config_defaults = _noop
    ctrl._apply_inference_params_from_ui = _noop
    ctrl._collect_current_slice_input = lambda: (label, stack)
    ctrl._adjust_channels_for_input = _noop
    ctrl._confirm_resegment = lambda _l: True
    ctrl._layer_by_name = lambda _n: None
    seg = ctrl.segmentation_manager

    def _segment_image(inp, _cfg, lbl):
        seg.data_masks[lbl] = np.full((16, 16), value, dtype=np.int32)
        return seg.data_masks[lbl], None, None

    seg.segment_image = _segment_image

    class _ImageManager:
        def convert_to_rgb(self, **_k):
            return np.zeros((16, 16, 3))

    ctrl.image_manager = _ImageManager()


def test_a_refused_predict_never_reaches_a_later_finalize(tmp_path, monkeypatch):
    """A prediction that was not saved is not what Finalize writes as the
    slice's accepted pair: the slice's cached mask goes back with its record."""
    ctrl = _ctrl(tmp_path, run="pilot")
    label = _add_slice(ctrl)                       # its accepted mask is all 1
    assert ctrl._persist_curated_training_pair(label)
    ctrl._touch_sample(label, status="accepted", view="XY")
    _predict_ready(ctrl, label, 7)
    real = P.round_accepted_dir
    monkeypatch.setattr(P, "round_accepted_dir", _raise)
    ctrl._run_segmentation()
    monkeypatch.setattr(P, "round_accepted_dir", real)
    assert ctrl.sample_records[label]["status"] == "accepted"
    assert int(np.max(ctrl.segmentation_manager.data_masks[label])) == 1
    ctrl._stage_accepted_in_current_view()
    mask = tifffile.imread(str(P.round_accepted_dir(tmp_path, "pilot", "lateral", 0)
                               / "stack01_XY_z0012_mask.tif"))
    assert int(mask.max()) == 1


def test_a_run_list_read_failure_never_rewrites_it(tmp_path, monkeypatch):
    """The save re-reads step2_state.json to carry the other runs forward; a
    read that fails (a lock held by a backup, a sync tool) refuses the save
    and leaves the file byte for byte -- never a file with one run in it."""
    import builtins

    ctrl = _ctrl(tmp_path, run="pilot")
    RM.create_run_on_disk(tmp_path, "second", display_name="second")
    RM.add_run_to_state_json(tmp_path, "second")
    state = tmp_path / "step2_state.json"
    before = state.read_bytes()
    assert ctrl._step2_write_target() == "run"        # the parse is remembered
    real_open = builtins.open
    calls = {"n": 0}

    def _open(file, mode="r", *a, **k):
        if (calls["n"] == 0 and "r" in str(mode)
                and Path(str(file)).name == "step2_state.json"):
            calls["n"] += 1
            raise PermissionError("locked by a backup")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", _open)
    assert ctrl._save_internal_state() == "refused"
    monkeypatch.setattr(builtins, "open", real_open)
    assert calls["n"] == 1
    assert state.read_bytes() == before
    assert list(tmp_path.glob("step2_state.json.corrupt-*")) == []
    assert ctrl._save_internal_state() == "saved-v3"
    assert set(json.loads(state.read_text(encoding="utf-8"))["runs"]) == {
        "pilot", "second"}


def test_a_run_list_that_lost_its_runs_table_is_never_rewritten(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    ctrl._step2_write_target = lambda: "run"     # it changed after the check
    state = tmp_path / "step2_state.json"
    state.write_text('{"schema_version": 3, "runs": []}', encoding="utf-8")
    before = state.read_bytes()
    assert ctrl._save_internal_state() == "refused"
    assert state.read_bytes() == before


def _switch_ready(ctrl, monkeypatch):
    from napari_epicycle.hitl_training_annotation import _switch_run_guard as G

    monkeypatch.setattr(G, "check_switch_safety",
                        lambda _c, _v: (G.SwitchSafety.SAFE, []))
    touched = []
    ctrl._flush_pending_state_timers = _noop
    ctrl._clear_patch_viewer_layers_safely = lambda: touched.append("layers")
    ctrl._close_step2_annotator_dock_safely = lambda: touched.append("dock")
    return touched


def test_a_switch_whose_save_fails_is_cancelled(tmp_path, monkeypatch):
    """The old run's unsaved edits are never replaced by the new run's
    state: a refused save cancels the switch before anything is closed."""
    ctrl = _ctrl(tmp_path, run="pilot")
    RM.create_run_on_disk(tmp_path, "second", display_name="second")
    RM.add_run_to_state_json(tmp_path, "second")
    state = tmp_path / "step2_state.json"
    before = state.read_bytes()
    ctrl.sample_records = {"stack01_XY_12": {"status": "curating",
                                             "note": object()}}   # TypeError
    touched = _switch_ready(ctrl, monkeypatch)
    told = _told(ctrl)
    ctrl._switch_active_run("second")
    assert ctrl.active_run_id == "pilot"
    assert "note" in ctrl.sample_records["stack01_XY_12"]
    assert touched == []
    assert state.read_bytes() == before
    assert told and "the switch was cancelled" in told[0]
    assert "TypeError" in told[0]


@pytest.mark.parametrize("damage", ["gone", "unreadable"])
def test_a_switch_that_cannot_happen_leaves_the_screen_alone(
        tmp_path, monkeypatch, damage):
    ctrl = _ctrl(tmp_path, run="pilot")
    if damage == "unreadable":
        RM.create_run_on_disk(tmp_path, "second", display_name="second")
        (tmp_path / "step2_state.json").write_bytes(b"{oops")
        target, sentence = "second", T.RUN_LIST_UNREADABLE
    else:
        target, sentence = "gone", T.RUN_FOLDER_GONE.format(name="gone")
    touched = _switch_ready(ctrl, monkeypatch)
    told = _told(ctrl)
    ctrl._switch_active_run(target)
    assert touched == []
    assert ctrl.active_run_id == "pilot"
    assert told == [sentence]


def test_a_failed_picker_switch_is_said(tmp_path):
    ctrl = _ctrl(tmp_path, run="pilot")
    told = _told(ctrl)

    class _Combo:
        def itemData(self, _i):
            return "second"

    ctrl.ui.run_picker_combo = _Combo()

    def _boom(_slug):
        raise OSError("disk went away")

    ctrl._switch_active_run = _boom
    ctrl._on_run_picker_activated(0)
    assert told == [T.RUN_SWITCH_FAILED.format(
        name="second", error="OSError: disk went away")]


@pytest.mark.parametrize("damage", ["corrupt", "mixed", "runs-only"])
def test_an_unreadable_list_save_is_its_own_result(tmp_path, capsys, damage):
    if damage == "corrupt":
        (tmp_path / "step2_state.json").write_bytes(b"{oops")
        line = "the run list file (step2_state.json) could not be read; a copy was set aside"
        sentence = T.RUN_LIST_UNREADABLE
    elif damage == "mixed":
        (tmp_path / "training_data").mkdir()
        (tmp_path / "artifacts" / "step2" / "runs" / "x").mkdir(parents=True)
        line = "Browse offers the migration"
        sentence = T.RUN_LIST_MIXED
    else:
        (tmp_path / "artifacts" / "step2" / "runs" / "x").mkdir(parents=True)
        line = "run folders but no run list"
        sentence = T.RUN_LIST_MISSING
    ctrl = _ctrl(tmp_path)
    ctrl.active_run_id = "default" if damage == "mixed" else ""
    ctrl.sample_records = {"stack01_XY_12": {"status": "curating"}}
    assert ctrl._step2_write_target() == "unreadable"
    assert ctrl._no_run_sentence("unreadable") == sentence
    for _ in range(3):
        ctrl._debounced_state_save()
    out = capsys.readouterr().out
    assert out.count(line) == 1
    assert "no run yet" not in out
    assert _log_text(ctrl).count(sentence) == 1
    ctrl.shutdown()
    rescues = list((tmp_path / "artifacts" / "step2" / "rescue").glob("*.json"))
    assert len(rescues) == 1
    saved = json.loads(rescues[0].read_text(encoding="utf-8"))
    assert "stack01_XY_12" in json.dumps(saved)
    assert not (tmp_path / "hitl_state.json").exists()
    if damage == "corrupt":
        assert (tmp_path / "step2_state.json").read_bytes() == b"{oops"
        assert len(list(tmp_path.glob("step2_state.json.corrupt-*"))) == 1


def test_a_saved_run_forgets_its_old_save_error(tmp_path, monkeypatch, capsys):
    ctrl = _ctrl(tmp_path, run="pilot")
    real, _calls = _break_v3_write(monkeypatch)
    ctrl._debounced_state_save()
    from napari_epicycle.hitl_training_annotation.controller import (
        training_annotation_controller as tac,
    )
    monkeypatch.setattr(tac, "_atomic_write_json", real)
    ctrl._debounced_state_save()
    _break_v3_write(monkeypatch)
    ctrl._debounced_state_save()
    assert capsys.readouterr().out.count("could not save step2_state.json") == 2


def test_new_run_on_an_unreadable_list_makes_no_folder(tmp_path):
    (tmp_path / "step2_state.json").write_bytes(b"{oops")
    ctrl = _ctrl(tmp_path)
    for _ in range(2):
        with pytest.raises(RM.Step2StateUnreadableError):
            ctrl._create_new_run("pilot", "Pilot")
    assert not (tmp_path / "artifacts" / "step2" / "runs").exists()
    assert len(list(tmp_path.glob("step2_state.json.corrupt-*"))) == 1
    assert (tmp_path / "step2_state.json").read_bytes() == b"{oops"


def test_the_run_list_copy_is_set_aside_once(tmp_path):
    (tmp_path / "step2_state.json").write_bytes(b"{oops")
    for _ in range(3):
        with pytest.raises(RM.Step2StateUnreadableError):
            RM.add_run_to_state_json(tmp_path, "pilot", activate=True)
    assert len(list(tmp_path.glob("step2_state.json.corrupt-*"))) == 1


def test_a_run_list_without_runs_is_unreadable_to_the_loader_too(tmp_path):
    (tmp_path / "step2_state.json").write_text(
        '{"schema_version": 3, "active_run_id": "pilot"}', encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    assert ctrl._step2_write_target() == "unreadable"
    assert ctrl._layout_allows_new_run() is False
    with pytest.raises(RM.Step2StateUnreadableError):
        RM.add_run_to_state_json(tmp_path, "pilot", activate=True)


def test_browse_opens_no_new_run_on_an_unreadable_list(tmp_path):
    (tmp_path / "step2_state.json").write_bytes(b"{oops")
    ctrl = _ctrl(tmp_path)
    opened = []
    ctrl._show_first_run_intro_dialog = lambda _ws: opened.append("intro")
    ctrl._on_new_run_button_clicked = lambda: opened.append("new run")
    ctrl._maybe_prompt_first_run(str(tmp_path))
    assert opened == []


def test_the_pair_refusal_line_is_per_workspace_and_said_again(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    ctrl = _ctrl(a)
    label = _add_slice(ctrl)
    ctrl._persist_curated_training_pair(label)
    ctrl._persist_curated_training_pair(label)
    ctrl.working_dir = str(b)
    ctrl._persist_curated_training_pair(label)
    assert capsys.readouterr().out.count("not saved to disk: no run yet") == 2
    _make_run(b, "pilot")
    ctrl.active_run_id = "pilot"
    assert ctrl._persist_curated_training_pair(label)       # written in b
    ctrl.active_run_id = ""
    ctrl._persist_curated_training_pair(label)
    assert capsys.readouterr().out.count("not saved to disk: no run yet") == 1


def test_a_round_on_an_unreadable_list_names_the_list(tmp_path):
    ctrl = _ctrl(tmp_path)
    with pytest.raises(RuntimeError) as err:
        ctrl._run_training_work(
            progress_cb=_noop, round_id="round_00", staged_slice_labels=[],
            training_mode="cumulative", prior_pairs=[], prior_checkpoints={},
            persisted_pairs={}, config_snapshot={}, working_dir_snapshot=str(tmp_path),
            active_run_snapshot="", write_target_snapshot="unreadable")
    assert str(err.value) == T.TRAIN_NO_RUN_LIST
    assert "New Run" not in T.TRAIN_NO_RUN_LIST


def test_no_active_run_loads_no_records(tmp_path):
    """A v3 file whose active run has no folder loads no run's records, and
    a stray hitl_state.json beside it is not read either."""
    (tmp_path / "step2_state.json").write_text(json.dumps({
        "schema_version": 3, "active_run_id": "gone",
        "runs": {"default": {"sample_records": {
            "stack01_XY_1": {"status": "accepted"}}}},
    }), encoding="utf-8")
    (tmp_path / "hitl_state.json").write_text(json.dumps({
        "schema_version": 2,
        "sample_records": {"stack01_XY_2": {"status": "accepted"}},
    }), encoding="utf-8")
    ctrl = _ctrl(tmp_path)
    ctrl._load_internal_state()
    assert ctrl.active_run_id == ""
    assert "stack01_XY_1" not in ctrl.sample_records
    assert "stack01_XY_2" not in ctrl.sample_records


def test_the_saves_called_directly_say_a_refusal(tmp_path, monkeypatch, capsys):
    """The save before a round and the one after it act on their result: a
    refused save before the round stops it (nothing is trained on a state
    that cannot be saved); a refused save after it is said."""
    ctrl = _ctrl(tmp_path, run="pilot")
    _stage_for_training(ctrl)
    errors = _record_errors(ctrl)
    _break_v3_write(monkeypatch)
    ctrl._train_model()
    out = capsys.readouterr().out
    assert errors == []
    assert "[TRAIN] " + T.TRAIN_SAVE_REFUSED.format(
        error="OSError: the disk is full") in out
    assert "could not save after training" not in out
    # one line, not two that disagree ("tries again" and "fix the folder")
    assert T.STATE_NOT_SAVED not in _log_text(ctrl)
    assert T.TRAIN_SAVE_REFUSED.format(
        error="OSError: the disk is full") in _log_text(ctrl)


def test_a_refused_save_after_training_is_said(tmp_path, monkeypatch, capsys):
    ctrl = _ctrl(tmp_path, run="pilot")
    _stage_for_training(ctrl)
    errors = _record_errors(ctrl)
    from napari_epicycle.hitl_training_annotation.controller import (
        training_annotation_controller as tac,
    )
    real = tac._atomic_write_json
    seen = {"n": 0}

    def _write(path, payload, *a, **k):
        if Path(path).name == P.STEP2_STATE_FILENAME:
            seen["n"] += 1
            if seen["n"] > 1:          # the save before the round goes through
                raise OSError("the disk is full")
        return real(path, payload, *a, **k)

    monkeypatch.setattr(tac, "_atomic_write_json", _write)
    ctrl._train_model()
    out = capsys.readouterr().out
    assert errors == []
    assert "could not save before training" not in out
    assert "could not save after training" in out
    assert T.STATE_NOT_SAVED in _log_text(ctrl)


def test_closing_with_no_run_says_so(tmp_path, capsys):
    ctrl = _ctrl(tmp_path)
    ctrl.shutdown()
    assert "closing with no run: nothing saved" in capsys.readouterr().out

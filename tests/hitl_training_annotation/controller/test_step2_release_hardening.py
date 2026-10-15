"""Step 2 release-hardening tests (2026-09-07 assessment, Brick 1).

Each class covers one ranked defect.
All fakes are Qt-free.
"""

from __future__ import annotations

import json
import os

import pytest

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


class _Signal:
    """Minimal Qt-signal stand-in (connect + emit)."""

    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class _LogOutput:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


# ---------------------------------------------------------------------------
# #1 — shutdown() must be reachable on napari 0.8 (no viewer.events.close)
# ---------------------------------------------------------------------------

class _Viewer08:
    """napari 0.8 viewer: `events` exists but has NO `close` attribute."""

    class _Events:
        pass

    def __init__(self):
        self.events = self._Events()


def _make_shutdown_ctrl():
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.viewer = _Viewer08()

    class _UI:
        pass

    ui = _UI()
    ui.destroyed = _Signal()
    ctrl.ui = ui

    ctrl._shutdown_calls = []
    ctrl.shutdown = lambda *a, **k: ctrl._shutdown_calls.append(1)
    return ctrl


def test_ui_destroyed_triggers_shutdown_on_napari_08():
    ctrl = _make_shutdown_ctrl()
    ctrl._connect_shutdown_hooks()

    assert ctrl._shutdown_calls == []
    ctrl.ui.destroyed.emit()
    assert len(ctrl._shutdown_calls) == 1, (
        "ui.destroyed must reach shutdown() on napari 0.8 where "
        "viewer.events.close does not exist"
    )


def test_viewer_close_still_wired_when_available():
    ctrl = _make_shutdown_ctrl()
    ctrl.viewer.events.close = _Signal()
    ctrl._connect_shutdown_hooks()

    ctrl.viewer.events.close.emit()
    assert len(ctrl._shutdown_calls) == 1


def test_shutdown_is_idempotent(tmp_path):
    """Two closes (viewer.close then ui.destroyed) must not double-run."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.viewer = _Viewer08()
    ctrl.viewer.events.close = _Signal()

    class _UI:
        pass

    ui = _UI()
    ui.destroyed = _Signal()
    ctrl.ui = ui
    ctrl.working_dir = str(tmp_path)

    saves = []
    ctrl._save_internal_state = lambda: saves.append(1)
    ctrl._flush_curating_persist_pending = lambda: None
    ctrl._connect_shutdown_hooks()

    ctrl.viewer.events.close.emit()
    ctrl.ui.destroyed.emit()
    assert len(saves) == 1, "shutdown() must be idempotent across close signals"


# ---------------------------------------------------------------------------
# #2 — a per-view training/inference failure must be VISIBLE
# ---------------------------------------------------------------------------

def _failed_views_ctrl():
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)

    class _UI:
        pass

    ui = _UI()
    ui.training_log_output = _LogOutput()
    ctrl.ui = ui
    ctrl._use_sync_training = True  # headless: no modal
    return ctrl


def test_report_failed_views_logs_every_failure():
    ctrl = _failed_views_ctrl()
    ctrl._report_failed_views({
        "failed_views": [
            {"view_type": "axial", "stage": "training", "error": "CUDA OOM"},
            {"view_type": "spatial", "stage": "inference", "error": "boom"},
        ]
    })
    text = "\n".join(ctrl.ui.training_log_output.lines)
    assert "axial" in text and "CUDA OOM" in text
    assert "spatial" in text and "boom" in text
    assert "TRAIN FAILED" in text


def test_report_failed_views_silent_when_none():
    ctrl = _failed_views_ctrl()
    ctrl._report_failed_views({"failed_views": []})
    assert ctrl.ui.training_log_output.lines == []


def test_training_finished_reports_failed_inference(tmp_path):
    """End-to-end: training succeeds, inference raises -> the log must say so."""
    from .test_training_round_flow import _make_train_ctrl, _ListWidget
    import numpy as np
    from napari_epicycle.hitl_training_annotation.controller.training_manager import (
        TrainingManager,
    )

    ckpt = tmp_path / "models" / "round_00" / "epi_spatial_round_00"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("checkpoint", encoding="utf-8")

    def trainer_backend(**kwargs):
        return {
            "checkpoint_path": str(ckpt),
            "losses": [0.8, 0.4],
            "mode": "custom_backend",
            "view_type": kwargs["view_type"],
        }

    def inferencer_backend(**kwargs):
        raise RuntimeError("inference exploded")

    ctrl = _make_train_ctrl(
        tmp_path,
        config={"model_name": "epi", "model_type": "cyto3", "epochs": 5},
        training_manager=TrainingManager(
            backend_trainer=trainer_backend,
            backend_inferencer=inferencer_backend,
        ),
    )
    ctrl.loaded_patch_name = "solo"
    xy = "solo_XY_1"
    ctrl.segmentation_manager.set(
        xy,
        np.random.rand(2, 16, 16).astype(np.float32),
        np.ones((16, 16), dtype=np.uint16),
    )
    ctrl._touch_sample(xy, status="staged", view="XY")
    ctrl.ui.data_to_add_list = _ListWidget([xy])

    ctrl._train_model()

    text = chr(10).join(ctrl.ui.training_log_output.lines)
    assert "TRAIN FAILED" in text, "inference failure was silent"
    assert "inference exploded" in text


# ---------------------------------------------------------------------------
# #3 — atomic JSON writes + corrupt-file quarantine
# ---------------------------------------------------------------------------

from packages.epicycle_io.state_store import (  # noqa: E402
    atomic_write_json,
    quarantine_corrupt_file,
)


def test_atomic_write_json_leaves_old_file_intact_on_crash(tmp_path, monkeypatch):
    target = tmp_path / "step2_state.json"
    target.write_text('{"runs": {"default": {"keep": true}}}', encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("simulated crash between temp write and replace")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        atomic_write_json(target, {"runs": {}})

    assert json.loads(target.read_text(encoding="utf-8"))["runs"]["default"]["keep"] is True
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_atomic_write_json_replaces_content(tmp_path):
    target = tmp_path / "nested" / "cfg.json"
    atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}


def test_quarantine_corrupt_file_moves_aside(tmp_path):
    bad = tmp_path / "step2_state.json"
    bad.write_text("{not json", encoding="utf-8")

    moved = quarantine_corrupt_file(bad)

    assert moved is not None and moved.is_file()
    assert moved.name.startswith("step2_state.json.corrupt-")
    assert not bad.exists()
    assert moved.read_text(encoding="utf-8") == "{not json"


def test_quarantine_missing_file_is_noop(tmp_path):
    assert quarantine_corrupt_file(tmp_path / "nope.json") is None


def test_no_automatic_root_training_config_is_written():
    """Superseded 2026-09-07 (bucket-switch item #7): the automatic
    workspace-root ``training_config.json`` is gone -- it was a global
    snapshot of one bucket's widgets that nothing read back.  The manual
    "Save Config..." remains and is still atomic."""
    import inspect
    from pathlib import Path as _P
    import napari_epicycle.hitl_training_annotation.controller.training_annotation_controller as m

    assert "_auto_save_training_config" not in _P(m.__file__).read_text(
        encoding="utf-8"
    )
    src = inspect.getsource(
        TrainingAnnotationController._save_and_set_training_config
    )
    assert "_atomic_write_json" in src


# ---------------------------------------------------------------------------
# #4 — a training error must be visible, not just a log line
# ---------------------------------------------------------------------------

class _ProgressBar:
    def __init__(self):
        self.value = None

    def setValue(self, v):
        self.value = v


def _error_ctrl():
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)

    class _UI:
        pass

    ui = _UI()
    ui.training_log_output = _LogOutput()
    ui.training_progress_bar = _ProgressBar()
    ctrl.ui = ui
    ctrl._use_sync_training = True
    ctrl._restore_train_button = lambda: None
    ctrl._set_training_ui_locked = lambda *_: None
    ctrl._clear_training_in_progress_marker = lambda: None
    shown = []
    ctrl._show_training_error_dialog = lambda msg: shown.append(msg)
    ctrl._shown = shown
    return ctrl


def test_on_training_error_shows_dialog_with_first_line():
    ctrl = _error_ctrl()
    ctrl._on_training_error("CUDA out of memory\nstack frame 1\nstack frame 2")

    assert any("CUDA out of memory" in line for line in ctrl.ui.training_log_output.lines)
    assert ctrl._shown, "training error must raise a visible dialog"
    assert "CUDA out of memory" in ctrl._shown[0]
    # The dialog itself shows only the headline, never the whole traceback.
    headline = TrainingAnnotationController._training_error_headline(
        "CUDA out of memory" + chr(10) + "stack frame 1"
    )
    assert headline == "CUDA out of memory"


def test_training_error_dialog_is_headless_safe():
    """The real dialog helper must no-op (not raise) in sync/test mode."""
    ctrl = _error_ctrl()
    del ctrl._show_training_error_dialog
    ctrl._on_training_error("boom")
    assert any("boom" in line for line in ctrl.ui.training_log_output.lines)


# ---------------------------------------------------------------------------
# #9 — never promote a checkpoint path that isn't a real file
# ---------------------------------------------------------------------------

def _promote_ctrl():
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)

    class _UI:
        pass

    ui = _UI()
    ui.training_log_output = _LogOutput()
    ctrl.ui = ui
    ctrl.active_curation_checkpoints = {"lateral": "", "axial": ""}
    ctrl.curation_model_source_mode = "auto"
    ctrl.current_config = {}
    ctrl._apply_curation_checkpoint_for_view = lambda **k: ""
    return ctrl


def test_promotion_skips_missing_checkpoint_file(tmp_path):
    ctrl = _promote_ctrl()
    ghost = tmp_path / "never_written"

    ctrl._promote_round_checkpoints_to_curation({"spatial": str(ghost)})

    assert ctrl.active_curation_checkpoints["lateral"] == "", (
        "a checkpoint path that does not exist (failed lift) must not become "
        "the active curation model"
    )
    text = chr(10).join(ctrl.ui.training_log_output.lines)
    assert "not a file" in text or "missing" in text.lower()


def test_promotion_accepts_real_checkpoint_file(tmp_path):
    ctrl = _promote_ctrl()
    real = tmp_path / "weights"
    real.write_text("w", encoding="utf-8")

    ctrl._promote_round_checkpoints_to_curation({"spatial": str(real)})

    assert ctrl.active_curation_checkpoints["lateral"] == str(real)


# ---------------------------------------------------------------------------
# #8 — cancel wording must not promise "a few seconds"
# ---------------------------------------------------------------------------

class _TrainButton:
    def __init__(self):
        self.text_value = "Train"
        self.enabled = True

    def setText(self, t):
        self.text_value = t

    def setEnabled(self, v):
        self.enabled = v


def test_cancel_states_the_real_epoch_checkpoint_bound():
    """Assert the text `cancel_training` actually emits, not its source."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)

    class _UI:
        pass

    ui = _UI()
    ui.training_log_output = _LogOutput()
    ui.update_and_train_button = _TrainButton()
    ctrl.ui = ui

    class _Thread:
        def isRunning(self):
            return True

    class _Flag:
        def __init__(self):
            self.set_called = False

        def set(self):
            self.set_called = True

        def is_set(self):
            return self.set_called

    ctrl._training_thread = _Thread()
    ctrl._training_abort_flag = _Flag()
    ctrl._use_sync_training = True

    ctrl.cancel_training()

    text = chr(10).join(ctrl.ui.training_log_output.lines)
    assert ctrl._training_abort_flag.set_called
    assert "few seconds" not in text
    assert "10 epochs" in text
    assert "epoch checkpoint" in text
    assert ctrl.ui.update_and_train_button.text_value == "Cancelling..."


# ---------------------------------------------------------------------------
# #11 — a failed inference must still leave a manifest (with a flag)
# ---------------------------------------------------------------------------

def _manifest_ctrl(tmp_path):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)

    class _UI:
        pass

    ui = _UI()
    ui.training_log_output = _LogOutput()
    ctrl.ui = ui
    ctrl.working_dir = str(tmp_path)
    ctrl._snapshot_viewer_channels = lambda: []
    return ctrl


def _write_manifest(tmp_path, **kwargs):
    from napari_epicycle.hitl_training_annotation import _paths as p
    ctrl = _manifest_ctrl(tmp_path)
    ctrl._write_step2_model_manifest_for_round(
        run_id="default",
        bucket="lateral",
        round_idx=0,
        view_type="spatial",
        losses=[0.5, 0.3],
        val_losses=[],
        config={"epochs": 5, "batch_size": 2, "model_type": "cyto3"},
        n_train_pairs=1,
        pretrained_model=None,
        weights_filename="default_lateral",
        working_dir=str(tmp_path),
        **kwargs,
    )
    path = p.model_manifest_path(str(tmp_path), "default", "lateral", 0)
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_flags_failed_inference(tmp_path):
    payload = _write_manifest(tmp_path, inference_failed=True)
    assert payload.get("inference_failed") is True


def test_manifest_omits_flag_on_success(tmp_path):
    payload = _write_manifest(tmp_path)
    assert payload.get("inference_failed", False) is False
    assert payload.get("final_loss", {}).get("val") is None


# ---------------------------------------------------------------------------
# Brick 1b #1 — shutdown() must signal the abort flag FIRST, and never block
# forever on the pair-writer queue
# ---------------------------------------------------------------------------

class _RecordingFlag:
    def __init__(self, order):
        self._order = order
        self._set = False

    def set(self):
        self._order.append("abort")
        self._set = True

    def is_set(self):
        return self._set


def test_shutdown_sets_abort_flag_before_flushing_writes(tmp_path):
    order = []

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.working_dir = str(tmp_path)
    ctrl._flush_curating_persist_pending = lambda: order.append("curating")
    ctrl._save_internal_state = lambda: order.append("save")
    ctrl._training_abort_flag = _RecordingFlag(order)

    class _Writer:
        def flush(self, timeout=None):
            order.append("flush")
            return True

    ctrl._pair_writer = _Writer()

    ctrl.shutdown()

    assert order[0] == "abort", (
        f"abort flag must be set before any blocking work, got order={order}"
    )
    assert "flush" in order and "save" in order


def test_pair_writer_flush_is_bounded_and_reports_backlog(capsys):
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        _AsyncPairWriter,
    )

    writer = _AsyncPairWriter()
    writer._ensure_thread = lambda: None  # no worker -> queue can never drain
    writer._q.put(("nowhere.tif", None))

    drained = writer.flush(timeout=0.15)

    assert drained is False, "bounded flush must give up instead of hanging"
    assert "not drained" in capsys.readouterr().out.lower()


def test_pair_writer_flush_unbounded_by_default(tmp_path):
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        _AsyncPairWriter,
    )
    import numpy as np

    writer = _AsyncPairWriter()
    writer.submit(str(tmp_path / "a.tif"), np.zeros((4, 4), dtype=np.uint16))
    assert writer.flush() is True
    assert (tmp_path / "a.tif").is_file()


# ---------------------------------------------------------------------------
# Brick 1b #2 — a user cancel is not a failure: no "Training failed" modal
# ---------------------------------------------------------------------------

def test_cancel_routes_through_error_signal_without_failure_dialog():
    """Drive the REAL worker signal path: abort -> error() -> _on_training_error."""
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        _TrainingWorker,
    )
    from packages.epicycle_training.exceptions import TrainingAbortedError

    ctrl = _error_ctrl()
    del ctrl._show_training_error_dialog  # use the real one
    ctrl._use_sync_training = False       # would otherwise skip every modal
    ctrl._training_abort_flag = None

    shown = []
    import napari_epicycle.hitl_training_annotation.controller.training_annotation_controller as mod
    original = mod.TrainingAnnotationController._show_training_error_dialog
    try:
        mod.TrainingAnnotationController._show_training_error_dialog = (
            lambda self, msg: shown.append(msg)
        )

        def _boom(**kwargs):
            raise TrainingAbortedError("Training cancelled by user.")

        worker = _TrainingWorker(_boom)
        worker.error.connect(ctrl._on_training_error)
        worker.run()
    finally:
        mod.TrainingAnnotationController._show_training_error_dialog = original

    text = chr(10).join(ctrl.ui.training_log_output.lines)
    assert "cancelled by user" in text.lower(), "the log line must remain"
    assert shown == [], "a user cancel must not pop a 'Training failed' modal"


def test_real_failure_still_shows_dialog_when_abort_flag_clear():
    ctrl = _error_ctrl()
    ctrl._training_abort_flag = None
    ctrl._on_training_error("CUDA out of memory")
    assert ctrl._shown == ["CUDA out of memory"]


def test_failure_dialog_suppressed_while_abort_flag_is_set():
    class _Flag:
        def is_set(self):
            return True

    ctrl = _error_ctrl()
    ctrl._training_abort_flag = _Flag()
    ctrl._on_training_error("some late error after the user hit Cancel")
    assert ctrl._shown == []


# ---------------------------------------------------------------------------
# Brick 1b #3 — modals must be display-guarded, and say the truth when BOTH
# buckets failed
# ---------------------------------------------------------------------------

class _RecordingBox:
    """Records every QMessageBox call so the dialog branch can be driven."""

    instances = []

    Warning = "warning"
    Critical = "critical"
    Ok = "ok"

    def __init__(self, parent=None):
        self.parent = parent
        self.icon = None
        self.title = None
        self.text = None
        self.informative = None
        self.executed = False
        _RecordingBox.instances.append(self)

    def setIcon(self, icon):
        self.icon = icon

    def setWindowTitle(self, t):
        self.title = t

    def setText(self, t):
        self.text = t

    def setInformativeText(self, t):
        self.informative = t

    def setStandardButtons(self, _b):
        pass

    def exec(self):
        self.executed = True


def _patch_qt(monkeypatch, *, has_app=True):
    from qtpy import QtWidgets

    _RecordingBox.instances = []
    monkeypatch.setattr(QtWidgets, "QMessageBox", _RecordingBox, raising=False)

    class _App:
        @staticmethod
        def instance():
            return object() if has_app else None

    monkeypatch.setattr(QtWidgets, "QApplication", _App, raising=False)
    return _RecordingBox


def test_error_dialog_is_skipped_without_a_qapplication(monkeypatch):
    box = _patch_qt(monkeypatch, has_app=False)
    ctrl = _error_ctrl()
    del ctrl._show_training_error_dialog
    ctrl._use_sync_training = False   # only the display check may stop us
    ctrl._training_abort_flag = None

    ctrl._on_training_error("boom")

    assert box.instances == [], "no QMessageBox may be built without a QApplication"
    assert any("boom" in line for line in ctrl.ui.training_log_output.lines)


def test_failed_views_dialog_is_skipped_without_a_qapplication(monkeypatch):
    box = _patch_qt(monkeypatch, has_app=False)
    ctrl = _failed_views_ctrl()
    ctrl._use_sync_training = False

    ctrl._report_failed_views({"failed_views": [{"view_type": "axial",
                                                 "stage": "training",
                                                 "error": "x"}]})

    assert box.instances == []
    assert ctrl.ui.training_log_output.lines


def test_error_dialog_branch_runs_with_a_qapplication(monkeypatch):
    box = _patch_qt(monkeypatch, has_app=True)
    ctrl = _error_ctrl()
    del ctrl._show_training_error_dialog
    ctrl._use_sync_training = False
    ctrl._training_abort_flag = None

    ctrl._on_training_error("CUDA out of memory" + chr(10) + "frame 1")

    assert len(box.instances) == 1
    shown = box.instances[0]
    assert shown.text == "CUDA out of memory"
    assert "log" in shown.informative.lower()
    assert shown.executed is True


def test_failed_views_dialog_says_no_model_when_both_buckets_failed(monkeypatch):
    box = _patch_qt(monkeypatch, has_app=True)
    ctrl = _failed_views_ctrl()
    ctrl._use_sync_training = False

    ctrl._report_failed_views(
        {"failed_views": [
            {"view_type": "spatial", "stage": "training", "error": "a"},
            {"view_type": "axial", "stage": "training", "error": "b"},
        ]},
        successful_buckets=set(),
    )

    text = chr(10).join(ctrl.ui.training_log_output.lines)
    assert "No model was trained this round" in text
    assert "the other" not in text.lower()
    assert len(box.instances) == 1
    assert "No model was trained this round" in (
        box.instances[0].text + box.instances[0].informative
    )


def test_failed_views_dialog_names_the_surviving_bucket(monkeypatch):
    _patch_qt(monkeypatch, has_app=True)
    ctrl = _failed_views_ctrl()
    ctrl._use_sync_training = False

    ctrl._report_failed_views(
        {"failed_views": [{"view_type": "axial", "stage": "training", "error": "b"}]},
        successful_buckets={"lateral"},
    )

    text = chr(10).join(ctrl.ui.training_log_output.lines)
    assert "lateral" in text


# ---------------------------------------------------------------------------
# Brick 1b #4 — the canonical artifact writer + manual Save Config are atomic
# ---------------------------------------------------------------------------

def test_artifact_registry_write_json_is_atomic(tmp_path, monkeypatch):
    from packages.epicycle_io.artifact_registry import FileArtifactRegistry

    reg = FileArtifactRegistry()
    target = tmp_path / "round_metadata.json"
    reg._write_json(target, {"round_id": "round_00"})
    assert json.loads(target.read_text(encoding="utf-8"))["round_id"] == "round_00"

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        reg._write_json(target, {"round_id": "clobbered"})

    assert json.loads(target.read_text(encoding="utf-8"))["round_id"] == "round_00"
    assert [p for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_atomic_write_json_fsyncs_before_replace(tmp_path, monkeypatch):
    from packages.epicycle_io import state_store

    synced = []
    monkeypatch.setattr(state_store.os, "fsync", lambda fd: synced.append(fd))
    state_store.atomic_write_json(tmp_path / "x.json", {"a": 1})
    assert synced, "the temp file must be fsynced before os.replace"


# ---------------------------------------------------------------------------
# Brick 1b #5 / #6b — the corrupt-state branch quarantines and says what is lost
# ---------------------------------------------------------------------------

def _new_layout_ctrl(tmp_path):
    from napari_epicycle.io_registry import HITLPersistenceService

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.persistence = HITLPersistenceService()
    ctrl.working_dir = str(tmp_path)
    ctrl.current_round = 0
    ctrl.training_data_by_round = {}
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.current_config = {"epochs": 1}
    ctrl.selected_view = "XY"
    ctrl.active_run_id = "default"
    ctrl._refresh_current_config_from_widgets = lambda: None
    ctrl._write_per_bucket_metadata_for_active_run = lambda **k: None

    class _UI:
        pass

    ui = _UI()
    ui.training_log_output = _LogOutput()
    ctrl.ui = ui

    (tmp_path / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)
    return ctrl


def test_save_internal_state_quarantines_a_corrupt_state_file(tmp_path):
    ctrl = _new_layout_ctrl(tmp_path)
    state_path = tmp_path / "step2_state.json"
    state_path.write_text('{"runs": {"other": ', encoding="utf-8")  # truncated

    ctrl._save_internal_state()

    quarantined = list(tmp_path.glob("step2_state.json.corrupt-*"))
    assert len(quarantined) == 1, "the corrupt file must be kept, not overwritten"
    assert quarantined[0].read_text(encoding="utf-8") == '{"runs": {"other": '
    # A fresh, valid state file replaced it.
    assert json.loads(state_path.read_text(encoding="utf-8"))["runs"]

    text = chr(10).join(ctrl.ui.training_log_output.lines)
    assert "OTHER run" in text
    assert "Recover them from the quarantined copy" in text
    assert str(quarantined[0]) in text


def test_save_internal_state_leaves_a_valid_state_file_alone(tmp_path):
    ctrl = _new_layout_ctrl(tmp_path)
    state_path = tmp_path / "step2_state.json"
    state_path.write_text(
        json.dumps({"schema_version": 3, "active_run_id": "default",
                    "runs": {"other": {"display_name": "Other run"}}}),
        encoding="utf-8",
    )

    ctrl._save_internal_state()

    assert list(tmp_path.glob("step2_state.json.corrupt-*")) == []
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["runs"]["other"]["display_name"] == "Other run"


# ---------------------------------------------------------------------------
# Brick 1b #6a — the inference_failed manifest gate is covered end to end
# ---------------------------------------------------------------------------

def test_train_model_writes_manifest_flagged_when_inference_fails(tmp_path):
    """A revert of the #11 gate must fail a test, not ship silently."""
    from .test_training_round_flow import _make_train_ctrl, _ListWidget
    import numpy as np
    from napari_epicycle.hitl_training_annotation import _paths as p
    from napari_epicycle.hitl_training_annotation.controller.training_manager import (
        TrainingManager,
    )

    # New-layout workspace: step2_state.json + runs/default/ must exist.
    (tmp_path / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)
    (tmp_path / "step2_state.json").write_text(
        json.dumps({"schema_version": 3, "active_run_id": "default",
                    "runs": {"default": {}}}),
        encoding="utf-8",
    )

    model_dir = p.model_dir(str(tmp_path), "default", "lateral", 0)
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = model_dir / "default_lateral_round_00"
    weights.write_text("w", encoding="utf-8")

    def trainer_backend(**kwargs):
        return {
            "checkpoint_path": str(weights),
            "losses": [0.5],
            "val_losses": [],
            "mode": "custom_backend",
            "view_type": kwargs["view_type"],
            "new_layout": True,
            "weights_lifted": True,
        }

    def inferencer_backend(**kwargs):
        raise RuntimeError("inference exploded")

    ctrl = _make_train_ctrl(
        tmp_path,
        config={"model_name": "epi", "model_type": "cyto3", "epochs": 1},
        training_manager=TrainingManager(
            backend_trainer=trainer_backend,
            backend_inferencer=inferencer_backend,
        ),
    )
    ctrl.active_run_id = "default"
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    ctrl.loaded_patch_name = "stk"
    ctrl._snapshot_viewer_channels = lambda: []
    xy = "stk_XY_1"
    ctrl.segmentation_manager.set(
        xy,
        np.random.rand(2, 16, 16).astype(np.float32),
        np.ones((16, 16), dtype=np.uint16),
    )
    ctrl._touch_sample(xy, status="staged", view="XY")
    ctrl.ui.data_to_add_list = _ListWidget([xy])

    ctrl._train_model()

    manifest_path = p.model_manifest_path(str(tmp_path), "default", "lateral", 0)
    assert manifest_path.is_file(), (
        "a failed inference must still leave provenance next to real weights"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["inference_failed"] is True
    assert payload.get("final_loss", {}).get("val") is None

"""Step 2 Brick D tests: indicators, error surfacing, honest UI.

Covers five ranked Step 2 defects and the hide list.  All fakes are Qt-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


class _LogOutput:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


class _Bar:
    """QProgressBar stand-in that records every range/value it is given."""

    def __init__(self):
        self.range = (0, 100)
        self.value = 0
        self.visible = True          # permanent, like the real widget
        self.history = []

    def setRange(self, lo, hi):
        self.range = (lo, hi)
        self.history.append(("range", lo, hi))

    def setValue(self, v):
        self.value = v
        self.history.append(("value", v))

    def setFormat(self, text):
        self.history.append(("format", text))

    def setTextVisible(self, flag):
        self.history.append(("textvisible", bool(flag)))

    def setVisible(self, flag):
        self.visible = bool(flag)
        self.history.append(("visible", bool(flag)))


class _UI:
    def __init__(self):
        self.training_log_output = _LogOutput()
        self.training_progress_bar = _Bar()
        # 2026-09-07: the busy funnel drives its OWN bar in the patches
        # column; `training_progress_bar` is training-only.
        self.patch_busy_bar = _Bar()


# ---------------------------------------------------------------------------
# #6 (error half) — a failed predict must be VISIBLE, and the bar must reset
# ---------------------------------------------------------------------------

class _BoomSegmentationManager:
    segmenter = None

    def set_model_preference(self, *a, **k):
        raise RuntimeError("no such file: bogus_checkpoint\nsecond line")

    def segment_image(self, *a, **k):  # pragma: no cover - never reached
        raise AssertionError("should not get past set_model_preference")


def _make_predict_ctrl(*, manager=None):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.working_dir = ""
    ctrl.selected_view = "XY"
    ctrl.current_config = {
        "model_type": "custom",
        "channel1": "1: membrane",
        "channel2": "0: none",
    }
    ctrl.segmentation_manager = manager or _BoomSegmentationManager()
    ctrl._ensure_runtime_defaults = lambda: None
    ctrl._exit_refine_mode = lambda: None
    ctrl._apply_model_source_selection = lambda **k: {
        "can_segment": True,
        "mode": "round",
        "checkpoint_path": r"C:\ws\runs\cell\lateral\round_01\model\cell_lateral_round_01",
    }
    ctrl._ensure_segmentation_config_defaults = lambda: None
    ctrl._apply_inference_params_from_ui = lambda: None
    ctrl._collect_current_slice_input = lambda: (
        "patch1_XY_40", np.zeros((2, 8, 8), dtype=np.uint16),
    )
    ctrl._adjust_channels_for_input = lambda stack: None
    ctrl._confirm_resegment = lambda label: True
    ctrl._can_show_modal = lambda: True

    shown = []
    ctrl._show_modal_error = lambda title, headline, detail="": shown.append(
        (title, headline, detail)
    )
    return ctrl, shown


def test_failed_predict_shows_a_modal_with_the_first_error_line():
    ctrl, shown = _make_predict_ctrl()
    ctrl._run_segmentation()

    assert len(shown) == 1, "a failed predict must raise a modal, not just print"
    title, headline, _detail = shown[0]
    assert "no such file: bogus_checkpoint" == headline, (
        "only the FIRST line of the error belongs in the dialog"
    )
    assert "predict" in title.lower() or "segment" in title.lower()


def test_failed_predict_logs_the_checkpoint_and_channels():
    ctrl, _shown = _make_predict_ctrl()
    ctrl._run_segmentation()

    text = "\n".join(ctrl.ui.training_log_output.lines)
    assert "no such file: bogus_checkpoint" in text
    assert "cell_lateral_round_01" in text, "the log must name the checkpoint used"
    assert "1: membrane" in text, "the log must name the channels used"


def test_predict_busy_indicator_is_reset_on_failure():
    ctrl, _shown = _make_predict_ctrl()
    bar = ctrl.ui.patch_busy_bar
    ctrl._run_segmentation()

    assert ("range", 0, 0) in bar.history, (
        "a synchronous predict must put the bar in marquee mode"
    )
    assert bar.range == (0, 100) and bar.value == 0, (
        "the busy indicator must be reset in a finally, even when predict fails"
    )


def test_predict_busy_indicator_is_reset_on_success():
    class _OKManager:
        segmenter = object()

        def __init__(self):
            self.data_inputs = {}
            self.data_inputs_rgb = {}

        def segment_image(self, stack, config, label):
            return (
                np.zeros((8, 8), dtype=np.int32),
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((8, 8), dtype=np.float32),
            )

        def get_masks(self, label):
            return None

    class _ImageManager:
        def convert_to_rgb(self, **kwargs):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    ctrl, shown = _make_predict_ctrl(manager=_OKManager())
    ctrl.image_manager = _ImageManager()
    ctrl.segmented_slice_list = {"patch1_XY_40"}
    ctrl._touch_sample = lambda *a, **k: None
    ctrl._persist_curated_training_pair = lambda label: False
    ctrl._refresh_sample_queue = lambda: None
    ctrl._refresh_round_status = lambda: None
    ctrl._schedule_state_save = lambda: None
    ctrl._run_segmentation()

    bar = ctrl.ui.patch_busy_bar
    assert ("range", 0, 0) in bar.history
    assert bar.range == (0, 100) and bar.value == 0
    assert bar.visible is True, "the bar is PERMANENT, never hidden"
    assert ctrl.ui.training_progress_bar.history == [], "training's bar is untouched"
    assert shown == [], "a successful predict must not raise an error modal"


# ---------------------------------------------------------------------------
# #6 (load half) — the patch TIFF read runs on a worker, with a scan bar
# ---------------------------------------------------------------------------

class _FakeLayer:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data
        self.visible = True
        self.colormap = None
        self.blending = None

    def refresh(self):
        pass


class _FakeLayers(list):
    def remove(self, layer):
        list.remove(self, layer)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayers()

    def add_image(self, data, name=None, **kwargs):
        layer = _FakeLayer(name, data)
        self.layers.append(layer)
        return layer


def _make_load_ctrl(tmp_path):
    """Controller wired for the real `_load_patch` prelude/read/apply split."""
    import json

    import tifffile

    stack_dir = tmp_path / "patches"
    stack_dir.mkdir()
    vol = np.zeros((2, 4, 8, 8), dtype=np.uint16)
    tifffile.imwrite(str(stack_dir / "p1.tif"), vol)
    (stack_dir / "p1_metadata.json").write_text(
        json.dumps({"patch_name": "p1", "channel_names": ["c0", "c1"]})
    )

    from napari_epicycle.hitl_training_annotation.controller.image_manager import (
        ImageManager,
    )

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.viewer = _FakeViewer()
    ctrl.image_manager = ImageManager()
    ctrl.loaded_stack_path = str(stack_dir)
    ctrl.loaded_patch_name = "old_patch"
    ctrl.working_dir = str(tmp_path)
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"
    ctrl._curating_persist_pending = {}
    ctrl._flush_curating_persist_pending = lambda: None
    ctrl._exit_refine_mode = lambda: None

    class _IM:
        def cleanup_prompt_layers(self, viewer):
            pass

    ctrl.interactive_manager = _IM()
    ctrl.layer_lifecycle_manager = None
    ctrl._read_view_selection_from_ui = lambda: ("lateral", "XY")
    ctrl._set_axial_combo_enabled = lambda: None
    ctrl._clean_patch_filename = lambda item: "p1.tif"
    ctrl._refresh_round_status = lambda: None
    ctrl.apply_active_view_to_viewer = lambda **k: None
    ctrl._set_work_labels_visibility_for_context = lambda **k: None
    ctrl._refresh_inference_channel_options = lambda: None
    ctrl._layer_by_name = lambda name: None
    ctrl.active_run_id = "default"

    class _List:
        def selectedItems(self):
            return ["p1.tif"]

    ctrl.ui.train_patch_list = _List()
    return ctrl


def test_worker_driven_load_reaches_the_viewer_through_the_finished_slot(tmp_path):
    ctrl = _make_load_ctrl(tmp_path)
    ctx = ctrl._load_patch_prelude(force_axis="XY")
    assert ctx is not None
    assert ctrl.loaded_patch_name == "", (
        "the prelude must NOT claim the new patch (only the finished slot may) "
        "and must not leave the OLD name standing without its layers"
    )

    payload = ctrl._read_patch_payload(ctx["file_path"])
    ctrl._on_patch_load_finished(payload, ctx)

    assert ctrl.loaded_patch_name == "p1"
    names = [l.name for l in ctrl.viewer.layers]
    assert "p1_c0_XY_view" in names and "p1_c1_XY_view" in names


def test_a_superseded_load_is_discarded_and_its_bar_hidden(tmp_path):
    ctrl = _make_load_ctrl(tmp_path)
    ctx_a = ctrl._load_patch_prelude(force_axis="XY")
    payload = ctrl._read_patch_payload(ctx_a["file_path"])
    # The user clicks another patch: a second prelude bumps the token.
    ctrl._load_patch_prelude(force_axis="XY")

    ctrl._on_patch_load_finished(payload, ctx_a)

    assert ctrl.loaded_patch_name == "", (
        "a stale finish must not install the patch the user moved away from"
    )
    assert ctrl.viewer.layers == []
    assert ctrl.ui.patch_busy_bar.range == (0, 100)


def test_load_error_shows_a_dialog_and_resets_the_bar(tmp_path):
    ctrl = _make_load_ctrl(tmp_path)
    shown = []
    ctrl._show_modal_error = lambda t, h, d="": shown.append((t, h, d))
    ctx = ctrl._load_patch_prelude(force_axis="XY")
    ctrl._begin_busy("[PATCH] Loading ...", cursor=False)

    ctrl._on_patch_load_error("Metadata JSON not found: p1_metadata.json", ctx)

    assert len(shown) == 1
    assert "Metadata JSON not found" in shown[0][1]
    assert ctrl.ui.patch_busy_bar.range == (0, 100)
    assert ctrl.loaded_patch_name == "old_patch", (
        "a failed load leaves the previously-loaded patch in place"
    )


def test_load_after_leaving_the_step_is_discarded(tmp_path):
    ctrl = _make_load_ctrl(tmp_path)
    ctx = ctrl._load_patch_prelude(force_axis="XY")
    payload = ctrl._read_patch_payload(ctx["file_path"])
    ctrl._step2_left = True

    ctrl._on_patch_load_finished(payload, ctx)

    assert ctrl.loaded_patch_name == "old_patch", (
        "discarded, so the name the session had is restored (never persisted "
        "away as an empty string)"
    )
    assert ctrl.viewer.layers == []


def test_load_after_workspace_switch_is_discarded(tmp_path):
    ctrl = _make_load_ctrl(tmp_path)
    ctx = ctrl._load_patch_prelude(force_axis="XY")
    payload = ctrl._read_patch_payload(ctx["file_path"])
    ctrl.working_dir = str(tmp_path / "other_workspace")

    ctrl._on_patch_load_finished(payload, ctx)

    assert ctrl.loaded_patch_name == ""


def test_sync_load_patch_still_works_end_to_end(tmp_path):
    """The five in-annotator callers still get a synchronous `_load_patch`."""
    ctrl = _make_load_ctrl(tmp_path)
    ctrl._load_patch(force_axis="XY")
    assert ctrl.loaded_patch_name == "p1"
    assert len(ctrl.viewer.layers) == 2


def test_shutdown_joins_a_running_patch_load_thread(tmp_path):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.working_dir = str(tmp_path)
    ctrl._save_internal_state = lambda: None
    ctrl._flush_curating_persist_pending = lambda: None

    joined = []

    class _Thread:
        def isRunning(self):
            return True

        def quit(self):
            joined.append("quit")

        def wait(self, ms):
            joined.append(("wait", ms))
            return True

    ctrl._patch_load_thread = _Thread()
    ctrl.shutdown()

    assert "quit" in joined and any(j[0] == "wait" for j in joined if isinstance(j, tuple)), (
        "shutdown must join the patch-load thread, not abandon it mid-read"
    )


# ---------------------------------------------------------------------------
# #13 — silent total failures become modals; blocking ops show a busy state
# ---------------------------------------------------------------------------

def _make_import_ctrl(monkeypatch, tmp_path):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.working_dir = str(tmp_path)
    ctrl.active_run_id = "default"
    ctrl.loaded_patch_name = ""
    ctrl._can_show_modal = lambda: True
    ctrl._flush_pending_state_timers = lambda: None
    ctrl._get_run_display_name = lambda rid: rid
    shown = []
    ctrl._show_modal_error = lambda t, h, d="": shown.append((t, h, d))
    return ctrl, shown


def test_cherry_pick_total_failure_is_a_modal(monkeypatch, tmp_path):
    ctrl, shown = _make_import_ctrl(monkeypatch, tmp_path)
    ctrl._report_import_failure("Cherry-pick", RuntimeError("disk full\ntrace"))

    assert len(shown) == 1
    assert shown[0][1] == "disk full"
    assert any("disk full" in l for l in ctrl.ui.training_log_output.lines)


def test_post_import_failures_are_one_modal_not_three(monkeypatch, tmp_path):
    ctrl, shown = _make_import_ctrl(monkeypatch, tmp_path)
    failures = []
    ctrl._run_post_import_step("save state", lambda: 1 / 0, failures)
    ctrl._run_post_import_step("reload pairs", lambda: [][1], failures)
    ctrl._run_post_import_step("refresh UI", lambda: None, failures)
    ctrl._report_post_import_failures("Cherry-pick", failures)

    assert len(failures) == 2, "only the failing steps are collected"
    assert len(shown) == 1, "the user gets ONE dialog, not one per step"
    text = "\n".join(ctrl.ui.training_log_output.lines)
    assert "save state" in text and "reload pairs" in text
    assert "refresh UI" not in text


def test_no_post_import_modal_when_everything_worked(monkeypatch, tmp_path):
    ctrl, shown = _make_import_ctrl(monkeypatch, tmp_path)
    failures = []
    ctrl._run_post_import_step("save state", lambda: None, failures)
    ctrl._report_post_import_failures("Cherry-pick", failures)
    assert shown == []


def test_skip_forever_sentinel_failure_tells_the_user(monkeypatch, tmp_path):
    from napari_epicycle.hitl_training_annotation import (
        _migrate_split_to_flat_dialog as dlg,
    )

    warned = []
    monkeypatch.setattr(
        dlg, "_show_sentinel_write_failed",
        lambda parent, error: warned.append(str(error)),
        raising=False,
    )

    class _Logic:
        @staticmethod
        def plan_split_to_flat_migration(ws):
            class _P:
                is_empty = False
                has_hard_errors = False
                total_operations = 3
            return _P()

        @staticmethod
        def is_migration_skipped(ws):
            return False

        @staticmethod
        def mark_migration_skipped(ws):
            raise OSError("read-only workspace")

    monkeypatch.setattr(dlg, "mlogic", _Logic, raising=False)
    monkeypatch.setattr(
        dlg, "_prompt_migration",
        lambda parent, plan: dlg.UserChoice.SKIP_FOREVER,
    )

    choice, result = dlg.prompt_and_migrate_if_needed(None, tmp_path)

    assert choice == dlg.UserChoice.SKIP_FOREVER
    assert result is None
    assert warned and "read-only workspace" in warned[0], (
        "a 'Don't ask again' that could not be saved must be surfaced — "
        "otherwise the prompt silently returns next session"
    )


# ---------------------------------------------------------------------------
# #18 — deleting a run is recoverable
# ---------------------------------------------------------------------------

def test_deleted_run_lands_in_trash_not_oblivion(tmp_path):
    from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
    from napari_epicycle.hitl_training_annotation._run_management import (
        create_run_on_disk, delete_run_from_disk,
    )

    create_run_on_disk(tmp_path, "cells", display_name="Cells")
    run_dir = _step2_paths.run_root(tmp_path, "cells")
    (run_dir / "lateral").mkdir(parents=True, exist_ok=True)
    (run_dir / "lateral" / "x.bin").write_bytes(b"data")

    delete_run_from_disk(tmp_path, "cells", confirm=True)

    assert not run_dir.exists(), "the run must leave the runs/ tree"
    trashed = sorted((tmp_path / "artifacts" / "step2" / ".trash").glob("cells*"))
    assert trashed, "a deleted run must be recoverable from .trash/"
    assert (trashed[0] / "lateral" / "x.bin").read_bytes() == b"data"


def test_two_deletes_of_the_same_slug_do_not_collide(tmp_path):
    from napari_epicycle.hitl_training_annotation._run_management import (
        create_run_on_disk, delete_run_from_disk,
    )

    for _ in range(2):
        create_run_on_disk(tmp_path, "cells", display_name="Cells")
        delete_run_from_disk(tmp_path, "cells", confirm=True)

    trashed = sorted((tmp_path / "artifacts" / "step2" / ".trash").glob("cells*"))
    assert len(trashed) == 2, "the second delete must not overwrite the first"


# ---------------------------------------------------------------------------
# #17 — the deferred dock close-hook must not touch a destroyed dock
# ---------------------------------------------------------------------------

def test_deferred_close_hook_survives_a_destroyed_dock():
    from napari_epicycle.annotator import launcher

    class _DeadDock:
        def __getattr__(self, name):
            raise RuntimeError("wrapped C/C++ object of type QtViewerDockWidget "
                               "has been deleted")

    # Must not raise: the 600 ms timer routinely outlives a dock the user
    # closed immediately after opening it.
    launcher._install_dock_close_hook_if_alive(_DeadDock(), object())


# ---------------------------------------------------------------------------
# Brick D fix pass (2026-09-07) - worker lifecycle, run identity, shutdown
# ---------------------------------------------------------------------------

class _Signal:
    """Minimal Qt-signal stand-in (connect + emit)."""

    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class _FakeThread:
    """QThread stand-in whose `start()` runs the worker synchronously."""

    def __init__(self):
        self.started = _Signal()
        self.finished = _Signal()
        self.deleted = False
        self._running = False

    def start(self):
        self._running = True
        self.started.emit()

    def quit(self, *_args):
        # Qt slots ignore surplus signal arguments; the fake must too
        # (worker.finished.connect(thread.quit) passes the payload).
        self._running = False
        self.finished.emit()

    def isRunning(self):
        return self._running

    def wait(self, ms=0):
        return True

    def deleteLater(self):
        self.deleted = True


class _FakeWorker:
    """Stand-in for `_Step2PatchLoadWorker` with the same signal surface."""

    def __init__(self, read_fn, file_path):
        self._read_fn = read_fn
        self._file_path = file_path
        self.finished = _Signal()
        self.error = _Signal()
        self.moved_to = None

    def moveToThread(self, thread):
        self.moved_to = thread

    def run(self):
        try:
            self.finished.emit(self._read_fn(self._file_path))
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


def _install_fake_qt(monkeypatch, threads):
    from napari_epicycle.hitl_training_annotation.controller import (
        training_annotation_controller as tac,
    )

    def _make_thread():
        t = _FakeThread()
        threads.append(t)
        return t

    monkeypatch.setattr(tac, "QThread", _make_thread)
    monkeypatch.setattr(tac, "_Step2PatchLoadWorker", _FakeWorker)
    return tac


def test_load_patch_async_drives_the_whole_worker_path(monkeypatch, tmp_path):
    """The real `_load_patch_async`: busy start, wiring, finish, cleanup."""
    threads = []
    _install_fake_qt(monkeypatch, threads)

    ctrl = _make_load_ctrl(tmp_path)
    ctrl._can_show_modal = lambda: True          # take the threaded branch
    ctrl._use_sync_training = False
    done = []

    started = ctrl._load_patch_async(on_done=lambda: done.append(1))

    assert started is True
    assert len(threads) == 1, "the async path must construct a thread"
    assert ctrl.loaded_patch_name == "p1"
    assert [l.name for l in ctrl.viewer.layers] == [
        "p1_c0_XY_view", "p1_c1_XY_view",
    ]
    assert done == [1]
    assert ctrl.ui.patch_busy_bar.range == (0, 100), "bar reset"
    assert threads[0].deleted is True
    assert ctrl._patch_load_thread is None


def test_real_worker_emits_payload_and_errors():
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        _Step2PatchLoadWorker,
    )

    got = []
    worker = _Step2PatchLoadWorker(lambda p: {"path": p}, "somewhere.tif")
    worker.finished.connect(got.append)
    worker.run()
    assert got == [{"path": "somewhere.tif"}]

    errs = []

    def _boom(_p):
        raise FileNotFoundError("no such patch")

    worker2 = _Step2PatchLoadWorker(_boom, "gone.tif")
    worker2.error.connect(errs.append)
    worker2.run()
    assert errs and "no such patch" in errs[0]


def test_a_late_first_load_does_not_delete_the_second_thread(monkeypatch, tmp_path):
    """Two overlapping loads: #1 finishing late must not retire #2."""
    threads = []
    _install_fake_qt(monkeypatch, threads)

    ctrl = _make_load_ctrl(tmp_path)
    ctrl._can_show_modal = lambda: True
    ctrl._use_sync_training = False

    # Build load #1 by hand so it does NOT auto-finish.
    ctx1 = ctrl._load_patch_prelude(force_axis="XY")
    assert ctx1 is not None
    thread1 = _FakeThread()
    ctrl._patch_load_thread = thread1
    ctrl._patch_load_worker = object()
    thread1.finished.connect(lambda t=thread1: ctrl._cleanup_patch_load_thread(t))
    thread1._running = True

    # The user clicks another patch: load #2 takes over (and, with the fake
    # thread, finishes immediately, clearing the handles it owns).
    ctrl._load_patch_async()
    thread2 = threads[-1]

    # ... and only now does load #1's thread finish.
    thread1.quit()

    assert thread1.deleted is True
    assert thread2.deleted is True, "load #2 retired itself, not thread1"
    assert thread2 is not thread1


def test_cleanup_only_clears_handles_for_the_thread_that_finished(tmp_path):
    ctrl = _make_load_ctrl(tmp_path)
    old_thread = _FakeThread()
    live_thread = _FakeThread()
    live_worker = object()
    ctrl._patch_load_thread = live_thread
    ctrl._patch_load_worker = live_worker

    ctrl._cleanup_patch_load_thread(old_thread)

    assert old_thread.deleted is True
    assert live_thread.deleted is False
    assert ctrl._patch_load_thread is live_thread
    assert ctrl._patch_load_worker is live_worker


def test_load_after_a_run_switch_is_discarded(tmp_path):
    ctrl = _make_load_ctrl(tmp_path)
    ctx = ctrl._load_patch_prelude(force_axis="XY")
    payload = ctrl._read_patch_payload(ctx["file_path"])

    ctrl.active_run_id = "second_run"          # user switched runs mid-read
    ctrl._on_patch_load_finished(payload, ctx)

    assert ctrl.loaded_patch_name == "", (
        "the old run's patch must not install into the new run's session"
    )
    assert ctrl.viewer.layers == []


def test_shutdown_disarms_the_load_before_joining(tmp_path):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.working_dir = str(tmp_path)
    ctrl._save_internal_state = lambda: None
    ctrl._flush_curating_persist_pending = lambda: None
    ctrl._patch_load_token = 7

    order = []

    class _Thread:
        def isRunning(self):
            return True

        def quit(self):
            order.append(("quit", ctrl._step2_left, ctrl._patch_load_token))

        def wait(self, ms):
            order.append(("wait", None, None))
            return True

    ctrl._patch_load_thread = _Thread()
    ctrl.shutdown()

    assert order and order[0][0] == "quit"
    assert order[0][1] is True, "_step2_left must be set BEFORE the join"
    assert order[0][2] != 7, "the load token must be bumped BEFORE the join"


def test_on_step_exit_marks_the_step_as_left():
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.viewer = _FakeViewer()
    ctrl.layer_lifecycle_manager = None

    class _IM:
        def cleanup_prompt_layers(self, viewer):
            pass

    ctrl.interactive_manager = _IM()
    ctrl._step2_left = False

    ctrl.on_step_exit()
    assert ctrl._step2_left is True, (
        "an in-flight patch load must be discardable after leaving Step 2"
    )

    ctrl.on_step_enter()
    assert ctrl._step2_left is False


# ---------------------------------------------------------------------------
# Audit fix 5 - a failed predict must not silently demote the slice
# ---------------------------------------------------------------------------

def test_failed_predict_restores_the_slice_status(monkeypatch):
    """`_confirm_resegment` demotes to `predicted` BEFORE cellpose runs.

    When cellpose then fails, the mask is untouched but the slice has been
    knocked out of the training set with no sign to the user.
    """
    from qtpy.QtWidgets import QMessageBox

    from napari_epicycle.hitl_training_annotation.controller import (
        training_annotation_controller as tac,
    )

    ctrl, shown = _make_predict_ctrl()
    ctrl.sample_records = {"patch1_XY_40": {"status": "accepted", "view": "XY"}}
    ctrl._RESEGMENT_GUARDED_STATUSES = (
        TrainingAnnotationController._RESEGMENT_GUARDED_STATUSES
    )

    def _touch(label, *, status=None, **kw):
        if status is not None:
            ctrl.sample_records.setdefault(label, {})["status"] = status

    ctrl._touch_sample = _touch
    ctrl._curating_persist_pending = {}
    monkeypatch.setattr(
        tac.QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.Yes),
    )
    # Real guard, real demotion.
    del ctrl._confirm_resegment

    ctrl._run_segmentation()

    assert shown, "the failure still has to be visible"
    assert ctrl.sample_records["patch1_XY_40"]["status"] == "accepted", (
        "cellpose failed and the mask is untouched, so the slice must keep "
        "the status the user gave it"
    )
    assert any("still marked accepted" in line
               for line in ctrl.ui.training_log_output.lines)


def test_failed_predict_ends_the_busy_state_before_the_modal():
    ctrl, _shown = _make_predict_ctrl()
    seen = []
    ctrl._show_modal_error = lambda t, h, d="": seen.append(
        ctrl.ui.patch_busy_bar.range
    )

    ctrl._run_segmentation()

    assert seen == [(0, 100)], (
        "the wait cursor / marquee must be released before a modal blocks"
    )


def test_segment_failure_dialog_title_matches_the_button():
    ctrl, shown = _make_predict_ctrl()
    ctrl._run_segmentation()
    assert shown[0][0] == "Segmentation failed"


def test_deferred_close_hook_respects_sip_isdeleted(monkeypatch):
    """The guard must actually consult sip, not just catch RuntimeError."""
    from napari_epicycle.annotator import launcher

    class _LooksAlive:
        """A dock whose Python side answers fine but whose C++ side is gone."""

        def installEventFilter(self, *_a):
            raise AssertionError("must not reach the real hook")

    installs = []
    monkeypatch.setattr(
        launcher, "_install_dock_close_hook",
        lambda d, w: installs.append((d, w)),
    )

    from qtpy import sip as _sip
    monkeypatch.setattr(_sip, "isdeleted", lambda obj: True, raising=False)

    launcher._install_dock_close_hook_if_alive(_LooksAlive(), object())
    assert installs == [], "a deleted dock must not get the hook installed"

    monkeypatch.setattr(_sip, "isdeleted", lambda obj: False, raising=False)
    launcher._install_dock_close_hook_if_alive(_LooksAlive(), object())
    assert len(installs) == 1, "a live dock still gets the hook"


# ---------------------------------------------------------------------------
# Audit fix 8 - a busy op during training must not reset the training bar
# ---------------------------------------------------------------------------

def test_busy_never_touches_the_training_bar_even_mid_training():
    """The two indicators are separate widgets, so they can never collide."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    train_bar = ctrl.ui.training_progress_bar
    train_bar.setRange(0, 100)
    train_bar.setValue(42)
    train_bar.history.clear()

    class _RunningThread:
        def isRunning(self):
            return True

    ctrl._training_thread = _RunningThread()

    ctrl._begin_busy("[IMPORT] Copying pairs ...")
    assert ctrl.ui.patch_busy_bar.range == (0, 0), (
        "a busy op sweeps its own bar even while training runs"
    )
    assert ctrl.ui.patch_busy_bar.visible is True

    ctrl._end_busy()
    assert train_bar.history == [], "the training bar is never written to"
    assert train_bar.range == (0, 100) and train_bar.value == 42
    assert any("Copying pairs" in line
               for line in ctrl.ui.training_log_output.lines), (
        "the user still gets told what is running"
    )


def test_busy_sweeps_and_then_hides_the_patch_bar():
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl._training_thread = None
    bar = ctrl.ui.patch_busy_bar

    ctrl._begin_busy("[IMPORT] Copying pairs ...")
    assert bar.range == (0, 0) and bar.visible is True
    ctrl._end_busy()
    assert bar.range == (0, 100) and bar.value == 0
    assert bar.visible is True, "the bar is permanent; idle == a static strip"


def test_nested_busy_only_the_outermost_end_hides_the_bar():
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    bar = ctrl.ui.patch_busy_bar

    ctrl._begin_busy("[PATCH] outer ...", cursor=False)
    ctrl._begin_busy("[PATCH] inner ...", cursor=False)
    ctrl._end_busy()
    assert bar.range == (0, 0), "the inner end must not release the scope"
    ctrl._end_busy()
    assert bar.range == (0, 100)


# ---------------------------------------------------------------------------
# Audit fix 7 - the orphan-archive scan must not reach into .trash/
# ---------------------------------------------------------------------------

def test_orphan_archive_scan_skips_deleted_runs(tmp_path):
    """A trashed run's archive must not be auto-restored back into life."""
    from packages.epicycle_training.training_backend_adapter import (
        scan_orphan_archives,
    )

    def _make_archive(save_dir):
        archive = save_dir / ".pre_train_archive" / "20260907-120000_model_m"
        archive.mkdir(parents=True)
        (archive / "model_m").write_bytes(b"weights")

    live = tmp_path / "artifacts" / "step2" / "runs" / "cells" / "lateral" / "round_00" / "model"
    live.mkdir(parents=True)
    _make_archive(live)

    trashed = tmp_path / "artifacts" / "step2" / ".trash" / "old-20260101" / "lateral" / "round_00" / "model"
    trashed.mkdir(parents=True)
    _make_archive(trashed)

    found = scan_orphan_archives(tmp_path)
    paths = [str(row["archive_path"]) for row in found]

    assert any("runs" in p for p in paths), "the live archive is still found"
    assert not any(".trash" in p for p in paths), (
        "an archive inside .trash/ must never be offered for restore"
    )


# ---------------------------------------------------------------------------
# Audit fix 10 - drive the REAL import handlers, not just the helpers
# ---------------------------------------------------------------------------

class _FakeDialog:
    """Import dialog stand-in: always accepted, always the same result."""

    RESULT = None

    def __init__(self, *a, **k):
        pass

    def exec_(self):
        return 1

    def get_result(self):
        return dict(self.RESULT)


def _arm_import_ctrl(monkeypatch, tmp_path):
    from napari_epicycle.hitl_training_annotation import _paths as _step2_paths

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.working_dir = str(tmp_path)
    ctrl.active_run_id = "default"
    ctrl.loaded_patch_name = ""
    ctrl._training_thread = None
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    ctrl._can_show_modal = lambda: True
    ctrl._flush_pending_state_timers = lambda: None
    ctrl._list_runs_for_ui = lambda: []
    ctrl._get_run_display_name = lambda rid: rid
    ctrl._begin_busy_calls = []
    monkeypatch.setattr(_step2_paths, "is_new_layout_state", lambda ws: True)

    shown = []
    ctrl._show_modal_error = lambda t, h, d="": shown.append((t, h, d))
    return ctrl, shown


def test_real_cherry_pick_handler_surfaces_a_failed_copy(monkeypatch, tmp_path):
    """Reverting `_report_import_failure` must break a test."""
    from napari_epicycle.hitl_training_annotation import _import_pairs
    from napari_epicycle.hitl_training_annotation import _import_pairs_dialog

    ctrl, shown = _arm_import_ctrl(monkeypatch, tmp_path)

    _FakeDialog.RESULT = {
        "source_run_id": "other",
        "source_bucket": "lateral",
        "source_round_idx": 0,
        "target_bucket": "lateral",
        "target_round_idx": 1,
        "selected_labels": ["p1_XY_10"],
        "overwrite_collisions": False,
    }
    monkeypatch.setattr(
        _import_pairs_dialog, "ImportPairsDialog", _FakeDialog,
    )

    def _boom(*a, **k):
        raise OSError("the target folder is read-only")

    monkeypatch.setattr(_import_pairs, "copy_pairs_to_target_round", _boom)
    monkeypatch.setattr(
        _import_pairs, "list_source_pairs", lambda *a, **k: [], raising=False,
    )
    monkeypatch.setattr(
        _import_pairs, "list_source_run_rounds", lambda *a, **k: [],
        raising=False,
    )

    ctrl._on_import_pairs_clicked()

    assert len(shown) == 1, "a copy that moved nothing must reach a modal"
    assert "read-only" in shown[0][1]
    assert any("read-only" in line for line in ctrl.ui.training_log_output.lines)
    assert ctrl.ui.patch_busy_bar.range == (0, 100), (
        "the busy state must be released on the failure path"
    )


def test_real_external_import_handler_surfaces_a_failed_copy(monkeypatch, tmp_path):
    from napari_epicycle.hitl_training_annotation import _import_external_pairs
    from napari_epicycle.hitl_training_annotation import (
        _import_external_pairs_dialog,
    )

    ctrl, shown = _arm_import_ctrl(monkeypatch, tmp_path)

    _FakeDialog.RESULT = {
        "target_bucket": "lateral",
        "target_round_idx": 0,
        "pair_records": [{"label": "ext_1"}],
        "relabel_basename": "ext",
        "overwrite_collisions": False,
        "source_channel_axis": "first",
    }
    monkeypatch.setattr(
        _import_external_pairs_dialog, "ImportExternalPairsDialog", _FakeDialog,
    )

    def _boom(*a, **k):
        raise OSError("could not read the source folder")

    monkeypatch.setattr(
        _import_external_pairs, "copy_external_pairs_to_target_round", _boom,
    )
    monkeypatch.setattr(
        _import_external_pairs, "enumerate_pairs", lambda *a, **k: [],
        raising=False,
    )

    ctrl._on_import_external_clicked()

    assert len(shown) == 1
    assert "could not read the source folder" in shown[0][1]
    assert ctrl.ui.patch_busy_bar.range == (0, 100)


def test_real_cherry_pick_handler_reports_a_broken_post_import_step(
    monkeypatch, tmp_path,
):
    """Copy works, the session update does not: one modal, honest text."""
    from napari_epicycle.hitl_training_annotation import _import_pairs
    from napari_epicycle.hitl_training_annotation import _import_pairs_dialog

    ctrl, shown = _arm_import_ctrl(monkeypatch, tmp_path)
    _FakeDialog.RESULT = {
        "source_run_id": "other",
        "source_bucket": "lateral",
        "source_round_idx": 0,
        "target_bucket": "lateral",
        "target_round_idx": 1,
        "selected_labels": ["p1_XY_10"],
        "overwrite_collisions": False,
    }
    monkeypatch.setattr(
        _import_pairs_dialog, "ImportPairsDialog", _FakeDialog,
    )
    monkeypatch.setattr(
        _import_pairs, "copy_pairs_to_target_round",
        lambda *a, **k: {"copied": ["p1_XY_10"], "skipped": [], "errors": []},
    )
    monkeypatch.setattr(
        _import_pairs, "list_source_pairs", lambda *a, **k: [], raising=False,
    )
    monkeypatch.setattr(
        _import_pairs, "list_source_run_rounds", lambda *a, **k: [],
        raising=False,
    )

    ctrl._add_sample_records_for_imported_pairs = lambda **k: None
    ctrl._save_internal_state = lambda: 1 / 0          # boom
    ctrl._reload_persisted_pairs_from_disk = lambda: None
    ctrl.segmentation_manager = None
    ctrl._refresh_run_picker = lambda: None
    ctrl._refresh_round_status = lambda: None
    ctrl._refresh_sample_queue = lambda: None
    ctrl._refresh_patch_list_badges = lambda: None
    ctrl._refresh_annotator_edit_state_visuals = lambda: None

    ctrl._on_import_pairs_clicked()

    assert len(shown) == 1, "one dialog for the whole post-import phase"
    assert shown[0][0] == "Copied, but not loaded"
    assert "saving your workspace" in "\n".join(
        ctrl.ui.training_log_output.lines
    ), "the step name the user reads must be plain language"


# ---------------------------------------------------------------------------
# Audit fix 12 - the busy indicator is actually reached from each blocking op
# ---------------------------------------------------------------------------

def test_busy_is_reached_from_the_split_to_flat_migration(monkeypatch, tmp_path):
    from napari_epicycle.hitl_training_annotation import (
        _migrate_split_to_flat_dialog as dlg,
    )

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    calls = []
    ctrl._begin_busy = lambda *a, **k: calls.append("begin")
    ctrl._end_busy = lambda: calls.append("end")
    monkeypatch.setattr(
        dlg, "prompt_and_migrate_if_needed",
        lambda parent, ws: (dlg.UserChoice.SKIP_SESSION, None),
    )

    ctrl._maybe_prompt_split_to_flat_migration(str(tmp_path))
    assert calls == ["begin", "end"]


def test_busy_is_reached_from_the_run_switch(monkeypatch, tmp_path):
    """The switch is synchronous end to end — it must show a busy state."""
    from napari_epicycle.hitl_training_annotation import _run_management

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.viewer = _FakeViewer()
    ctrl.working_dir = str(tmp_path)
    ctrl.active_run_id = "default"
    ctrl._run_switch_in_flight = False
    ctrl._get_run_display_name = lambda rid: rid
    ctrl._flush_pending_state_timers = lambda: None
    ctrl._save_internal_state = lambda: None
    ctrl._patch_load_token = 0
    calls = []
    ctrl._begin_busy = lambda *a, **k: calls.append("begin")
    ctrl._end_busy = lambda: calls.append("end")
    monkeypatch.setattr(
        _run_management, "set_active_run_in_state_json",
        lambda ws, slug: (_ for _ in ()).throw(RuntimeError("stop here")),
    )
    ctrl._close_step2_annotator_dock_safely = lambda: None
    ctrl._clear_patch_viewer_layers_safely = lambda: None

    try:
        ctrl._switch_active_run("other")
    except Exception:
        pass

    assert calls[0] == "begin" and calls[-1] == "end", (
        "the switch must show a busy state and always release it"
    )


# ---------------------------------------------------------------------------
# Copy fix 13 - the delete-run confirm must describe what actually happens
# ---------------------------------------------------------------------------

def test_delete_run_confirm_says_the_run_goes_to_the_trash():
    from napari_epicycle.hitl_training_annotation._manage_runs_panel import (
        ManageRunsPanel,
    )

    panel = ManageRunsPanel.__new__(ManageRunsPanel)
    panel._selected_summary = lambda: {
        "run_id": "cells",
        "display_name": "Cells",
        "lateral_round_count": 3,
        "axial_round_count": 1,
    }
    panel._on_delete = lambda slug: True
    panel._find_dependents = None
    panel._reload_from_callback = lambda: None
    panel._show_error = lambda *a: None
    asked = {}

    def _confirm(title, body, confirm_text=None, destructive=False):
        asked["title"] = title
        asked["body"] = body
        asked["confirm_text"] = confirm_text
        return False

    panel._confirm = _confirm
    panel._on_delete_clicked()

    assert asked["title"] == "Delete run?"
    assert asked["confirm_text"] == "Delete"
    body = asked["body"]
    assert "artifacts/step2/.trash/" in body
    assert "3 lateral and 1 axial" in body
    for lie in ("Permanently", "CANNOT be undone", "Delete forever"):
        assert lie not in body, f"the confirm still claims {lie!r}"


def test_benchmark_button_is_unwired_and_disabled():
    """The panel is behind a coming-soon veil; the button must be inert."""
    import inspect

    from napari_epicycle.hitl_training_annotation.ui import main_widget
    from napari_epicycle.hitl_training_annotation.controller import (
        training_annotation_controller as tac,
    )

    ui_src = inspect.getsource(main_widget.TrainingAnnotationHITL2D._create_benchmarking_panel)
    assert "setEnabled(False)" in ui_src

    connect_src = inspect.getsource(tac.TrainingAnnotationController._connect_signals)
    live = [
        line for line in connect_src.splitlines()
        if "run_cross_round_benchmark_button" in line
        and not line.strip().startswith("#")
    ]
    assert live == [], "the benchmark button must not be connected"

    from napari_epicycle.ui_shell import workflow_shell
    shell_src = inspect.getsource(workflow_shell.EpiCycleWorkflowShell._wire_button_history)
    live_shell = [
        line for line in shell_src.splitlines()
        if "run_cross_round_benchmark_button" in line
        and not line.strip().startswith("#")
    ]
    assert live_shell == [], (
        "the activity log must not claim a benchmark was triggered"
    )


# ---------------------------------------------------------------------------
# Re-audit fix - the cleared patch name must not be PERSISTED as ""
# ---------------------------------------------------------------------------

def _arm_saved_name_spy(ctrl):
    """Record what `loaded_patch_name` would be written to disk."""
    saved = []
    ctrl._schedule_state_save = lambda: saved.append(
        str(getattr(ctrl, "loaded_patch_name", "") or "")
    )
    return saved


def test_a_failed_load_does_not_persist_an_empty_patch_name(tmp_path):
    """The prelude clears the name; a failure must put it back.

    `_flush_curating_persist_pending` in the prelude schedules a 500 ms state
    save, and `_save_internal_state` writes `loaded_patch_name`
    unconditionally — so without a restore the next session forgets the
    patch (and Step 1's downstream-used guard flips to "not used").
    """
    ctrl = _make_load_ctrl(tmp_path)
    ctrl._show_modal_error = lambda t, h, d="": None
    saved = _arm_saved_name_spy(ctrl)

    ctx = ctrl._load_patch_prelude(force_axis="XY")
    assert ctrl.loaded_patch_name == ""

    ctrl._on_patch_load_error("FileNotFoundError: gone", ctx)

    assert ctrl.loaded_patch_name == "old_patch", (
        "a failed load must leave the previously-loaded patch named"
    )
    assert saved and saved[-1] == "old_patch", (
        "and must re-persist that name, not the empty string"
    )


def test_a_step_exit_discard_restores_the_patch_name(tmp_path):
    ctrl = _make_load_ctrl(tmp_path)
    saved = _arm_saved_name_spy(ctrl)
    ctx = ctrl._load_patch_prelude(force_axis="XY")
    payload = ctrl._read_patch_payload(ctx["file_path"])
    ctrl._step2_left = True

    ctrl._on_patch_load_finished(payload, ctx)

    assert ctrl.loaded_patch_name == "old_patch"
    assert saved[-1] == "old_patch"


def test_a_superseded_discard_leaves_the_name_to_the_newer_load(tmp_path):
    """The newer load owns the name — the stale one must not put back the old."""
    ctrl = _make_load_ctrl(tmp_path)
    _arm_saved_name_spy(ctrl)
    ctx_a = ctrl._load_patch_prelude(force_axis="XY")
    payload = ctrl._read_patch_payload(ctx_a["file_path"])
    ctrl._load_patch_prelude(force_axis="XY")     # second click

    ctrl._on_patch_load_finished(payload, ctx_a)

    assert ctrl.loaded_patch_name == ""


def test_a_load_that_outlives_the_debounce_persists_the_new_name(tmp_path):
    """Slow success: the 500 ms save may already have fired with ""."""
    ctrl = _make_load_ctrl(tmp_path)
    saved = _arm_saved_name_spy(ctrl)
    ctx = ctrl._load_patch_prelude(force_axis="XY")
    saved.append(str(ctrl.loaded_patch_name or ""))     # the debounce fires
    payload = ctrl._read_patch_payload(ctx["file_path"])

    ctrl._on_patch_load_finished(payload, ctx)

    assert ctrl.loaded_patch_name == "p1"
    assert saved[-1] == "p1", (
        "a successful load must re-schedule the save so the new name wins "
        "over the empty one the debounce may already have written"
    )


def test_end_busy_leaves_a_training_run_that_started_meanwhile_alone():
    """An import that outlives the Train click must not zero the real bar."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl._training_thread = None

    ctrl._begin_busy("[IMPORT] Copying pairs ...")
    assert ctrl.ui.patch_busy_bar.range == (0, 0)

    class _StartedMeanwhile:
        def isRunning(self):
            return True

    ctrl._training_thread = _StartedMeanwhile()
    ctrl.ui.training_progress_bar.setRange(0, 100)
    ctrl.ui.training_progress_bar.setValue(37)

    ctrl._end_busy()

    assert ctrl.ui.training_progress_bar.value == 37, (
        "training's real percentage survives an unrelated busy op ending"
    )
    assert ctrl.ui.patch_busy_bar.range == (0, 100)


def test_restored_status_reaches_the_queue_and_the_saved_state():
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.sample_records = {"p_XY_3": {"status": "predicted"}}

    def _touch(label, *, status=None, **kw):
        ctrl.sample_records[label]["status"] = status

    ctrl._touch_sample = _touch
    ctrl._resegment_prior_status = ("p_XY_3", "staged")
    calls = []
    ctrl._refresh_sample_queue = lambda: calls.append("queue")
    ctrl._schedule_state_save = lambda: calls.append("save")

    ctrl._restore_status_after_failed_segment("p_XY_3")

    assert ctrl.sample_records["p_XY_3"]["status"] == "staged"
    assert calls == ["queue", "save"], (
        "the Slices list and step2_state.json must not keep saying predicted"
    )

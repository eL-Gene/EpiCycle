"""`RefineWorker`, and the guards that stand around it.

Two halves:

* the WORKER itself -- thin, synchronous, directly callable, and a refusal
  arrives on ``finished`` because a refusal is an answer;
* the ORCHESTRATION guards in `Curation3DAnnotator`, bound to a bare object
  so this file never builds a napari viewer: the stale QUADRUPLE, the token
  stamped on the worker and read back off the sender (never compared against
  the live attribute, which a second launch has already bumped), the
  abandon-and-park (a QThread garbage-collected while running aborts the
  process), the drawing lock taken at LAUNCH and released on all four exits,
  and the Stopping... window in which a second Preview is refused.
"""

from __future__ import annotations

import inspect
import threading

import pytest

pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover
            pytest.skip("cannot create a QApplication: {}".format(exc))
    return app


def _ann():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


# --------------------------------------------------------------------------- #
# the worker
# --------------------------------------------------------------------------- #
class TestTheWorkerIsThinAndSynchronous:
    def test_it_forwards_the_call_and_emits_finished(self, qapp):
        from napari_epicycle.annotator.reshape_worker import (
            RefineOutcome,
            RefineWorker,
        )

        seen = []
        w = RefineWorker(lambda a, b: a + b, {"a": 2, "b": 3}, kind="settle")
        w.finished.connect(seen.append)
        w.run()
        assert len(seen) == 1
        assert isinstance(seen[0], RefineOutcome)
        assert seen[0].result == 5
        assert seen[0].kind == "settle"
        assert seen[0].elapsed_s >= 0.0

    def test_a_refusal_arrives_on_finished_not_on_error(self, qapp):
        from napari_epicycle.annotator.reshape_worker import RefineWorker
        from packages.epicycle_annotator.reshape.settle import settle_label

        import numpy as np

        vol = np.zeros((6, 10, 10), np.int32)
        vol[2:4, 3:7, 3:7] = 7
        done, failed = [], []
        w = RefineWorker(
            settle_label,
            {"labels_roi": vol, "elevation": np.zeros((2, 2, 2), np.float32),
             "label_id": 7},
            kind="settle",
        )
        w.finished.connect(done.append)
        w.error.connect(failed.append)
        w.run()
        assert failed == []
        assert done and done[0].result.status == "refused"

    def test_an_exception_arrives_on_error_with_a_traceback(self, qapp):
        from napari_epicycle.annotator.reshape_worker import RefineWorker

        def boom():
            raise RuntimeError("no")

        done, failed = [], []
        w = RefineWorker(boom, {}, kind="contour")
        w.finished.connect(done.append)
        w.error.connect(failed.append)
        w.run()
        assert done == []
        assert failed and "Traceback" in failed[0]

    def test_a_cancel_that_lands_before_run_skips_the_work_entirely(self, qapp):
        from napari_epicycle.annotator.reshape_worker import RefineWorker

        ran = []
        w = RefineWorker(lambda: ran.append(1), {}, kind="settle")
        w.cancel()
        w.run()
        assert ran == []
        assert w.cancel_event.is_set()

    def test_it_only_injects_progress_and_log_when_asked(self, qapp):
        from napari_epicycle.annotator.reshape_worker import RefineWorker

        got = {}

        def fn(**kwargs):
            got.update(kwargs)
            return None

        RefineWorker(fn, {"x": 1}, kind="contour").run()
        assert set(got) == {"x"}
        got.clear()
        RefineWorker(fn, {"x": 1}, kind="contour",
                     wants_progress=True, wants_log=True).run()
        assert set(got) == {"x", "progress", "log"}

    def test_progress_and_messages_reach_the_signals(self, qapp):
        from napari_epicycle.annotator.reshape_worker import RefineWorker

        steps, notes = [], []

        def fn(progress=None, log=None):
            progress("contour", 4, 11)
            log("halfway")
            return None

        w = RefineWorker(fn, {}, kind="contour",
                         wants_progress=True, wants_log=True)
        w.progress.connect(lambda s, i, n: steps.append((s, i, n)))
        w.message.connect(notes.append)
        w.run()
        assert steps == [("contour", 4, 11)]
        assert notes == ["halfway"]

    def test_a_results_own_messages_are_mirrored_once(self, qapp):
        from napari_epicycle.annotator.reshape_worker import RefineWorker

        class _R:
            messages = ["a", "b"]

        notes = []
        w = RefineWorker(lambda: _R(), {}, kind="settle")
        w.message.connect(notes.append)
        w.run()
        assert notes == ["a", "b"]

    def test_the_caller_can_hand_in_its_own_cancel_event(self, qapp):
        from napari_epicycle.annotator.reshape_worker import RefineWorker

        ev = threading.Event()
        w = RefineWorker(lambda: None, {}, cancel_event=ev)
        assert w.cancel_event is ev
        w.cancel()
        assert ev.is_set()


class TestTheStageTable:
    @pytest.mark.parametrize("stage", ["prepare", "elevation", "band",
                                       "contour", "rebuild", "settle"])
    def test_every_refine_stage_has_a_sentence(self, stage):
        from napari_epicycle.annotator.reshape_worker import (
            STAGE_TEXT,
            stage_text,
        )

        assert stage in STAGE_TEXT
        assert stage_text(stage) != stage

    def test_the_reshape_stages_are_untouched(self):
        from napari_epicycle.annotator.reshape_worker import STAGE_TEXT

        assert STAGE_TEXT["solve"] == "Cutting the cell..."
        assert STAGE_TEXT["seeds"] == "Placing the seeds..."


# --------------------------------------------------------------------------- #
# the orchestration guards -- bound to a bare object, no viewer
# --------------------------------------------------------------------------- #
class _Bare:
    """Just enough attributes for the guards under test.

    They are pure predicates over instance state, so binding them rather
    than building an annotator keeps this test about the contract.
    """

    def __init__(self, **kw):
        cls = _ann()
        self._active_labels_name = "labels"
        self._refine_launch_labels_name = "labels"
        self._refine_launch_patch = "patch_a"
        self._refine_launch_run_epoch = 3
        self._refine_token = 5
        self._smooth_running = True
        self._refine_worker = None
        self._refine_thread = None
        self._refine_busy_kind = "settle"
        self._refine_running = False
        self.__dict__.update(kw)
        for name in ("_refine_is_stale", "_result_token", "_signal_sender",
                     "_abandon_refine_worker", "_refine_thread_cleanup"):
            setattr(self, name, getattr(cls, name).__get__(self, _Bare))

    def _reshape_context(self):
        return {"patch_name": "patch_a", "run_epoch": 3,
                "spacing": (1.0, 1.0, 1.0)}


class TestTheStaleQuadruple:
    def test_a_matching_answer_is_not_stale(self):
        assert _Bare()._refine_is_stale(result_token=5) is False

    def test_a_superseded_token_is_stale(self):
        assert _Bare()._refine_is_stale(result_token=4) is True

    def test_a_changed_labels_layer_is_stale(self):
        obj = _Bare(_active_labels_name="other")
        assert obj._refine_is_stale(result_token=5) is True

    def test_a_changed_patch_is_stale(self):
        obj = _Bare(_refine_launch_patch="patch_b")
        assert obj._refine_is_stale(result_token=5) is True

    def test_a_changed_run_epoch_is_stale(self):
        obj = _Bare(_refine_launch_run_epoch=99)
        assert obj._refine_is_stale(result_token=5) is True

    def test_a_session_that_is_gone_is_stale(self):
        obj = _Bare(_smooth_running=False)
        assert obj._refine_is_stale(result_token=5) is True

    def test_it_never_compares_the_live_attribute_to_itself(self):
        """The tautology to avoid: a guard that reads
        `self._refine_token` on both sides can never fire, because a second
        launch has already bumped it.  The token must be STAMPED on the
        worker and read back off the sender."""
        src = inspect.getsource(_ann()._refine_is_stale)
        assert "result_token" in src
        src2 = inspect.getsource(_ann()._result_token)
        assert "_epicycle_token" in src2
        assert "sender" in inspect.getsource(_ann()._signal_sender)

    def test_every_finish_handler_opens_with_the_guard(self):
        for name in ("_on_refine_finished", "_on_refine_error",
                     "_on_refine_progress"):
            src = inspect.getsource(getattr(_ann(), name))
            assert "_refine_is_stale" in src, (
                "{} does not check for a late answer".format(name))


class TestTheLaunchStampsTheToken:
    def test_the_launch_stamps_the_token_on_the_worker(self):
        src = inspect.getsource(_ann()._launch_refine_worker)
        assert "_epicycle_token" in src
        assert "moveToThread" in src
        assert "started.connect" in src


class TestAbandonAndPark:
    def test_abandon_cancels_clears_and_is_idempotent(self):
        class _W:
            def __init__(self):
                self.cancelled = 0

            def cancel(self):
                self.cancelled += 1

        w = _W()
        obj = _Bare(_refine_worker=w, _refine_running=True)
        assert obj._abandon_refine_worker() == "abandoned"
        assert w.cancelled == 1
        assert obj._refine_running is False
        assert obj._refine_busy_kind is None
        assert obj._abandon_refine_worker() == "idle"

    def test_a_thread_that_will_not_stop_is_PARKED_not_dropped(self):
        """A QThread garbage-collected while running aborts the whole
        process (observed live 2026-09-08)."""
        src = inspect.getsource(_ann()._refine_thread_cleanup)
        assert "park" in src.lower()
        assert "wait(" in src

        class _T:
            def quit(self):
                pass

            def wait(self, _ms):
                return False

        class _W:
            class _S:
                def disconnect(self):
                    pass

            progress = message = finished = error = _S()

        obj = _Bare(_refine_thread=_T(), _refine_worker=_W())
        obj._refine_thread_cleanup()
        assert obj._refine_thread is None
        assert len(getattr(obj, "_refine_parked_threads", [])) == 1

    def test_a_thread_that_stops_is_joined_and_not_parked(self):
        class _T:
            def quit(self):
                pass

            def wait(self, _ms):
                return True

        obj = _Bare(_refine_thread=_T())
        obj._refine_thread_cleanup()
        assert obj._refine_thread is None
        assert not getattr(obj, "_refine_parked_threads", [])

    def test_every_exit_abandons_the_worker(self):
        """Names the sites: they are the ones that leave the
        staged window while a worker may still be running."""
        for name in ("_reject_smooth", "_accept_smooth", "_reset_smooth_state",
                     "_close_cleanup", "_teardown_reshape_for_switch",
                     "_load_selected_layers"):
            src = inspect.getsource(getattr(_ann(), name))
            assert ("_abandon_refine_worker" in src
                    or "_teardown_refine_for_switch" in src), (
                "{} can strand a running RefineWorker".format(name))

    def test_the_shared_teardown_abandons_it(self):
        src = inspect.getsource(_ann()._teardown_refine_for_switch)
        assert "_abandon_refine_worker" in src
        assert "_drop_refine_walls" in src


class TestTheDrawingLock:
    def test_the_lock_is_taken_at_LAUNCH_not_after_the_result(self):
        """A worker opens a window in which the user can paint into an ROI
        that is about to be overwritten by a result copied at launch --
        exactly the loss `_lock_subviewers_for_preview` exists to prevent.  So the lock goes on BEFORE `thread.start()`."""
        src = inspect.getsource(_ann()._launch_refine_worker)
        assert "_lock_subviewers_for_preview" in src
        assert (src.index("_lock_subviewers_for_preview")
                < src.index("start()")), "the lock is taken after the launch"

    def test_it_is_released_on_all_four_exits(self):
        for name in ("_on_refine_finished", "_on_refine_error",
                     "_on_refine_stopped", "_launch_refine_worker"):
            src = inspect.getsource(getattr(_ann(), name))
            assert "_restore_ready_to_draw_state" in src, (
                "{} can leave drawing locked".format(name))

    def test_the_strip_no_longer_carries_the_drawing_off_sentence(self):
        """Owner decision on B2 audit UX-M7 (B3): the one-line stage strip
        clipped "Drawing is off until this finishes." behind the plane count
        on every method, and the side views are visibly locked -- so the
        sentence came off the strip, and its constant with it (nothing else
        read it)."""
        from napari_epicycle.annotator import advanced_features_widget as M

        assert not hasattr(M, "REFINE_DRAWING_OFF")
        for name in ("_launch_refine_worker", "_on_refine_progress"):
            src = inspect.getsource(getattr(_ann(), name))
            assert "REFINE_DRAWING_OFF" not in src, name
            assert "Drawing is off" not in src, name


class TestStopIsNotInstant:
    def test_stop_enters_a_stopping_state_rather_than_pretending(self):
        """`random_walker` and `diffuse_labels3D` cannot be interrupted once
        entered, so Stop disables itself, the bar keeps running and the
        stage line reads Stopping... until the thread exits."""
        src = inspect.getsource(_ann()._cancel_refine_preview)
        assert "Stopping" in src
        assert "stop_enabled=False" in src

    def test_a_second_preview_is_refused_while_a_worker_is_in_flight(self):
        src = inspect.getsource(_ann()._preview_smooth)
        assert "_refine_running" in src
        assert "Still working" in src

    def test_stop_clears_every_piece_of_staged_state(self):
        """A Stop that leaves them set makes the next Preview chain onto a
        result that never existed."""
        src = inspect.getsource(_ann()._on_refine_stopped)
        for attr in ("_smooth_running", "_smooth_snapshot", "_smooth_roi",
                     "_smooth_chain_count", "_refine_busy_kind"):
            assert attr in src, "{} survives a Stop".format(attr)


# --------------------------------------------------------------------------- #
# C-M2 -- ONE test that runs a real RefineWorker on a real QThread.
#
# Everything above this line is either the worker called synchronously or a
# guard read off a bare object.  Neither can see the thing the audit asked
# about: what happens when the answer arrives QUEUED, from another thread,
# after the user has pressed Stop -- which is exactly the sequence that
# wrote a refused result into the volume on 2026-09-22.
# --------------------------------------------------------------------------- #
class TestOnARealThread:
    """An `Event`-gated function on a real QThread, driven through the
    annotator's own launch and stop paths.

    Slow by construction (it waits on a thread), and deliberately the ONLY
    test in this file that is.
    """

    def _annotator(self, qapp, vol):
        """A bare **QObject** carrying the real launch / stop / finish methods.

        A QObject and not a plain object, deliberately: PyQt gives a
        non-QObject receiver a DIRECT connection, so the finish handler
        would run on the worker thread and the test would prove something
        other than what the app does.  With a QObject the connection is
        QUEUED across the thread boundary and the handler runs on the main
        thread when this test pumps -- which is the shape the 2026-09-22
        bug had.
        """
        import numpy as np
        from qtpy.QtCore import QObject

        K = _ann()

        class _Sync:
            def __init__(self, v):
                self.shared_labels = {"labels": v}

        class _A(QObject):
            pass

        a = _A()
        a._active_labels_name = "labels"
        a.sync_manager = _Sync(vol)
        a._smooth_running = True
        a._smooth_snapshot = vol.copy()
        a._smooth_roi = (slice(None), slice(None), slice(None))
        a._smooth_label_id = 7
        a._smooth_chain_count = 1
        a._refine_steps = []
        a._refine_staged_method = None
        a._refine_running = False
        a._refine_stopping = False
        a._refine_lock_fired = False
        a._refine_busy_kind = None
        a._refine_worker = None
        a._refine_thread = None
        a._refine_token = 0
        a._refine_parked_threads = []
        a._refine_elevation = None
        a._refine_elevation_roi = None
        a._refine_walls_wanted = False
        a._temp_guide_layers = {}
        a._refine_lock_cb = None
        a.ui = []
        a.restored = 0

        a._update_smooth_ui = lambda **kw: a.ui.append(kw)
        a._sync_all_views_after_operation = lambda **kw: None
        a._refresh_refine_walls_toggle = lambda: None
        a._lock_subviewers_for_preview = lambda: None
        a._restore_ready_to_draw_state = lambda: setattr(
            a, "restored", a.restored + 1)
        a._refresh_merge_button_state = lambda: None
        a._refresh_label_action_buttons_state = lambda: None
        a._remove_temp_layer = lambda key: None
        a._get_advanced_features_widget = lambda: None
        a._reshape_context = lambda: {
            "patch_name": "p", "run_epoch": 0, "spacing": (1.0, 1.0, 1.0)}
        for name in ("_launch_refine_worker", "_cancel_refine_preview",
                     "_on_refine_finished", "_on_refine_error",
                     "_on_refine_progress", "_on_refine_message",
                     "_on_refine_stopped", "_on_refine_thread_finished",
                     "_refine_is_stale", "_result_token", "_signal_sender",
                     "_abandon_refine_worker", "_refine_thread_cleanup",
                     "_fire_refine_lock", "_drop_refine_walls",
                     "_drop_refine_walls_layer", "_finish_refine_preview",
                     "_reset_smooth_state", "_refine_breadcrumb_text",
                     "_refine_spacing", "_refine_spacing_known"):
            setattr(a, name, getattr(K, name).__get__(a, _A))
        for name in ("_REFINE_NOTHING_MOVED", "_REFINE_STEP_WORD",
                     "_REFINE_METHOD_WORD"):
            setattr(a, name, getattr(K, name))
        return a

    def _pump(self, qapp, predicate, seconds=10.0):
        import time

        t0 = time.time()
        while time.time() - t0 < seconds:
            qapp.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_a_queued_answer_after_Stop_is_thrown_away_and_the_object_restored(
            self, qapp):
        """The 2026-09-22 sequence, on a real thread.

        The function blocks on an Event, so Stop lands while the worker is
        genuinely mid-flight; releasing the Event then delivers `finished`
        from the worker thread, QUEUED, after the stop.  It must be
        discarded -- and the object must come back.
        """
        import threading

        import numpy as np

        vol = np.zeros((6, 12, 12), np.int32)
        vol[1:5, 3:9, 3:9] = 7
        before = vol.copy()
        a = self._annotator(qapp, vol)

        gate = threading.Event()
        entered = threading.Event()

        def fn(**_kw):
            entered.set()
            gate.wait(10.0)
            out = np.zeros_like(vol)
            out[0:6, 1:11, 1:11] = 7          # a big, obvious change
            return (out, 999, "the outline moved.")

        a._launch_refine_worker(fn, {}, "contour", 7, a._smooth_roi)
        assert self._pump(qapp, entered.is_set), "the worker never started"
        assert a._refine_running is True
        assert a._refine_thread is not None

        a._cancel_refine_preview()
        assert a._refine_stopping is True, "Stop must be a STATE, not a wait"
        # ...and it returned immediately, without joining: the thread is
        # still in the slot and still inside `fn`.
        assert a._refine_thread is not None

        gate.set()
        assert self._pump(qapp, lambda: not a._refine_stopping), (
            "the stopped path never ran")

        assert np.array_equal(vol, before), (
            "{} voxels of a REFUSED answer were written".format(
                int(np.count_nonzero(vol != before))))
        assert a._smooth_running is False
        assert a._smooth_snapshot is None
        assert a._refine_running is False
        assert a._refine_thread is None, "the thread was not joined"
        assert a.restored >= 1, "drawing was left locked"

    def test_a_second_preview_cannot_launch_until_the_first_thread_exits(
            self, qapp):
        """Preview stays disabled for the whole Stopping window,
        so a second solve cannot start while the first still holds the GIL
        and the ROI."""
        import threading

        import numpy as np

        vol = np.zeros((6, 12, 12), np.int32)
        vol[1:5, 3:9, 3:9] = 7
        a = self._annotator(qapp, vol)

        gate = threading.Event()
        entered = threading.Event()

        def fn(**_kw):
            entered.set()
            gate.wait(10.0)
            return (np.zeros_like(vol), 0, "nothing.")

        a._launch_refine_worker(fn, {}, "settle", 7, a._smooth_roi)
        assert self._pump(qapp, entered.is_set)

        def preview_enabled():
            for kw in reversed(a.ui):
                if "preview_enabled" in kw:
                    return kw["preview_enabled"]
            return None

        assert preview_enabled() is False, "Preview was live during the flight"
        a._cancel_refine_preview()
        assert preview_enabled() is False, "Preview came back while stopping"

        gate.set()
        assert self._pump(qapp, lambda: not a._refine_stopping)
        assert preview_enabled() is True, "Preview never came back"

    def test_a_normal_finish_on_a_real_thread_stages_and_joins(self, qapp):
        import threading

        import numpy as np

        vol = np.zeros((6, 12, 12), np.int32)
        vol[1:5, 3:9, 3:9] = 7
        a = self._annotator(qapp, vol)

        entered = threading.Event()

        def fn(**_kw):
            entered.set()
            out = np.zeros_like(vol)
            out[1:5, 2:10, 3:9] = 7
            return (out, 48, "the outline moved.")

        a._launch_refine_worker(fn, {}, "contour", 7, a._smooth_roi)
        assert self._pump(qapp, lambda: a._refine_thread is None
                          and not a._refine_running)
        assert entered.is_set()
        assert a._refine_staged_method == "contour"
        assert int((vol == 7).sum()) > 0
        assert a._refine_parked_threads == [], "it should have joined cleanly"

    def test_the_token_is_read_off_the_SENDER_on_a_real_thread(self, qapp):
        """A second launch bumps `_refine_token`, so the first answer must
        be discarded by the token stamped on ITS worker."""
        import threading

        import numpy as np

        vol = np.zeros((6, 12, 12), np.int32)
        vol[1:5, 3:9, 3:9] = 7
        before = vol.copy()
        a = self._annotator(qapp, vol)

        gate = threading.Event()
        entered = threading.Event()

        def slow(**_kw):
            entered.set()
            gate.wait(10.0)
            out = np.zeros_like(vol)
            out[:, :, :] = 7
            return (out, 999, "session A.")

        a._launch_refine_worker(slow, {}, "contour", 7, a._smooth_roi)
        assert self._pump(qapp, entered.is_set)
        first = a._refine_worker

        # session B supersedes it without ever running
        a._refine_token += 1

        gate.set()
        # The stale branch returns WITHOUT clearing `_refine_running`: that
        # flag belongs to session B now, and clearing it would unlock a
        # surface session B is still using.  What it must do is join the
        # thread and write nothing.
        assert self._pump(qapp, lambda: a._refine_thread is None, 10.0), (
            "session A's thread was never joined")
        assert np.array_equal(vol, before), (
            "session A's answer landed under session B's token")
        assert a._refine_staged_method is None
        assert first is not None


def test_a_cancel_before_run_still_answers():
    """Recipes audit H2: a cancel that lands before `run()` used to return
    without emitting anything, and nothing quits the thread, so the Stop
    path never ran.  It answers with an empty outcome now."""
    from napari_epicycle.annotator.reshape_worker import (
        RefineOutcome,
        RefineWorker,
    )

    w = RefineWorker(lambda **kw: 1 / 0, {}, kind="settle")
    got, errs = [], []
    w.finished.connect(got.append)
    w.error.connect(errs.append)
    w.cancel()
    w.run()
    assert errs == []
    assert len(got) == 1 and isinstance(got[0], RefineOutcome)
    assert got[0].result is None and got[0].kind == "settle"

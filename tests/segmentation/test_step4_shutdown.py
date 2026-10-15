"""B2: ``shutdown()`` at napari's close.

A REAL ``QThread`` per job-thread kind (Predict / Reconstruct = ``_seg``,
Measure = ``_qc``, Smooth = ``_smooth``, Load = ``_load``), each worker
blocked on an ``Event``: ``shutdown()`` asks it to stop, silences it and
its thread, quits and waits; no handler of the controller -- the result
handler nor the thread-finished backstop -- runs after it.  A worker that
ignores the cancel is parked, never destroyed.  The panel's ``destroyed``
runs ``shutdown()`` too.
"""
from __future__ import annotations

import threading
import time

import pytest
from qtpy.QtCore import QObject, Signal

from napari_epicycle.segmentation import controller as C

from ._step4_fakes import controller

KINDS = (("_seg", "_predict_running"), ("_qc", "_measure_running"),
         ("_smooth", "_smooth_running"), ("_load", "_load_in_flight"))


class _Blocked(QObject):
    finished = Signal(object)
    progress = Signal(int)

    def __init__(self, obey_cancel=True, early=False):
        super().__init__()
        self.release = threading.Event()
        self.started = threading.Event()
        self.emitted = threading.Event()
        self.obey = obey_cancel
        self.early = early

    def request_cancel(self):
        if self.obey:
            self.release.set()

    def run(self):
        try:
            if self.early:
                # K2: a result posted BEFORE shutdown() -- already queued
                self.progress.emit(1)
                self.emitted.set()
            self.started.set()
            self.release.wait(20)
        finally:
            self.finished.emit("done")


def _wait(qapp, cond, deadline=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return cond()


@pytest.mark.parametrize("attr,flag", KINDS)
def test_shutdown_stops_every_job_thread_and_no_handler_runs(
        qapp, tmp_path, attr, flag):
    ctrl = controller(tmp_path)
    ran = []
    worker = _Blocked()
    worker.finished.connect(lambda _r: ran.append("result"))
    thread = ctrl._start_step4_thread(
        worker, flag, quit_on=("finished",), attr=attr,
        cleanup=lambda: ran.append("cleanup"))
    assert thread is not None and getattr(ctrl, flag) is True
    assert worker.started.wait(5)
    ctrl.shutdown()
    assert not thread.isRunning()
    for _ in range(20):
        qapp.processEvents()
    assert ran == []                            # silenced: nothing after
    assert getattr(ctrl, f"{attr}_thread") is None
    assert getattr(ctrl, flag) is False


def test_a_worker_that_ignores_the_cancel_is_parked(qapp, tmp_path):
    ctrl = controller(tmp_path)
    ran = []
    worker = _Blocked(obey_cancel=False)
    worker.finished.connect(lambda _r: ran.append("result"))
    thread = ctrl._start_step4_thread(
        worker, "_predict_running", quit_on=("finished",), attr="_seg",
        cleanup=lambda: ran.append("cleanup"))
    assert worker.started.wait(5)
    t0 = time.monotonic()
    ctrl.shutdown()
    assert time.monotonic() - t0 < 6               # waited at most ~3 s
    assert thread.isRunning()
    assert any(t is thread for t, _w in C._LIVE_STEP4_THREADS)
    worker.release.set()
    assert thread.wait(5000)
    for _ in range(20):
        qapp.processEvents()
    assert ran == []


def test_shutdown_twice_is_harmless(qapp, tmp_path):
    ctrl = controller(tmp_path)
    ctrl.shutdown()
    ctrl.shutdown()


def test_shutdown_gives_the_nav_owner_back(qapp, tmp_path):
    from .test_step4_lock_full import _Shell
    ctrl = controller(tmp_path)
    shell = _Shell()
    ctrl._workflow_shell = shell
    ctrl._step4_active = True
    worker = _Blocked()
    ctrl._start_step4_thread(worker, "_measure_running",
                             quit_on=("finished",), attr="_qc")
    assert shell.owners() == {"step4"}
    ctrl.shutdown()
    assert shell.owners() == set()


def test_the_panels_destroyed_runs_shutdown(qapp, tmp_path):
    from ._step4_panel import panel
    w = panel(qapp)
    c = w.controller
    calls = []
    c.shutdown = lambda: calls.append(1)
    from qtpy.QtCore import QEvent
    w.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert _wait(qapp, lambda: bool(calls), 5)


@pytest.mark.parametrize("attr,flag", KINDS)
def test_a_signal_queued_before_shutdown_never_lands(qapp, tmp_path, attr,
                                                     flag):
    """K2: the worker posted a signal just before the close; shutdown()
    runs BEFORE the event loop delivers it (the ui.destroyed path keeps the
    loop running) -- it must never reach a handler."""
    ctrl = controller(tmp_path)
    ran = []
    worker = _Blocked(early=True)
    worker.progress.connect(lambda _v: ran.append("progress"))
    ctrl._start_step4_thread(worker, flag, quit_on=("finished",), attr=attr,
                             cleanup=lambda: ran.append("cleanup"))
    assert worker.emitted.wait(5)
    ctrl.shutdown()                      # before any pump
    for _ in range(30):
        qapp.processEvents()
    assert ran == []
    for tok in ("_load_token", "_measure_token", "_smooth_token"):
        assert getattr(ctrl, tok, None) is None

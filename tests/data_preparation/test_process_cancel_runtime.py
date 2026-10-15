"""Runtime (real QThread) cancel test for the whole-volume process worker.

This is the test the prior parallel build LACKED: it exercises the actual
``_PreprocProcessWorker`` on a real ``QThread`` and pulls the Cancel lever
mid-run, asserting the worker STOPS and reports a clean "cancelled" (not a
"failed"), and the thread finishes so it can be torn down without the
"QThread destroyed while running" abort.

It injects a slow ``execute`` that polls ``should_cancel`` so the mid-run cancel
is DETERMINISTIC (no racing a fast real op).  Kept in its own file + guarded so a
missing Qt binding skips cleanly rather than erroring the batch.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from napari_epicycle.data_preparation import controller
from packages.epicycle_preprocessing.exceptions import PreprocessingCancelled

pytest.importorskip("qtpy")
from qtpy.QtCore import QCoreApplication, QThread  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402


def _slow_execute(recipe, channels, spacing, *, max_workers=None,
                  bounds_per_channel=None, parallel=False,
                  should_cancel=None, block_workers=None, on_progress=None):
    """Stand-in for the real pipeline: polls should_cancel so a mid-run Cancel
    reliably raises (and would otherwise return zeros after ~5s)."""
    for _ in range(1000):
        if should_cancel is not None and should_cancel():
            raise PreprocessingCancelled("cancelled between ops")
        time.sleep(0.005)   # sleeps the WORKER thread; main loop keeps pumping
    return ([np.zeros(c.shape, dtype=np.float32) for c in channels],
            tuple(spacing), {})


@pytest.fixture
def app():
    a = QCoreApplication.instance() or QCoreApplication([])
    yield a


def test_worker_cancel_mid_run_emits_cancelled(app, monkeypatch):
    monkeypatch.setattr(controller, "_pp_execute", _slow_execute)

    worker = controller._PreprocProcessWorker(
        {"channels": [[]], "geometry": None},
        [np.zeros((4, 4, 4), dtype=np.float32)],
        (1.0, 1.0, 1.0),
        parallel=True,
    )
    thread = QThread()
    worker.moveToThread(thread)

    got = []
    worker.error.connect(lambda m: got.append(("error", m)))
    worker.finished.connect(lambda *a: got.append(("finished", None)))
    worker.error.connect(thread.quit)
    worker.finished.connect(thread.quit)
    thread.started.connect(worker.run)

    thread.start()
    QTest.qWait(60)               # let the worker enter the polling loop
    worker.request_cancel()       # Cancel from the MAIN thread

    waited = 0
    while thread.isRunning() and waited < 4000:
        QTest.qWait(20)
        waited += 20

    assert not thread.isRunning(), "worker did not stop after Cancel"
    thread.wait(2000)
    # exactly one terminal signal, and it is the clean "cancelled" (grill B4)
    assert ("error", "cancelled") in got
    assert all(kind != "finished" for kind, _ in got)

    thread.deleteLater()
    worker.deleteLater()
    QTest.qWait(10)

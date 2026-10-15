"""The Add preview and box-info workers on a REAL ``QThread`` (B2 fix pass,
audit L-H1 / L-H2 / L-M3).

``add_harness.FakeThread`` used to fire ``finished`` whether or not anything
had called ``quit()``, so every lifecycle test stayed green while the real
launch never connected ``quit``: a real thread's event loop ran on after the
worker answered, ``thread.finished`` never came, busy stayed up until Accept
or Cancel, and the box-info worker answered exactly once per session.  These
tests drive the same borrowed orchestrator with the real ``QThread`` and the
real ``RefineWorker`` (``moveToThread`` included) and pump a plain
``QApplication`` until a deadline -- no pytest-qt.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytest.importorskip("qtpy")

from packages.epicycle_annotator import add_preview as AP  # noqa: E402

from .add_harness import Harness, nuclei_volume  # noqa: E402
from .test_add_nuclei_golden import box_of  # noqa: E402

DEADLINE_S = 20.0


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _pump(app, until, deadline=DEADLINE_S):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(until())


def _join(*threads):
    for th in threads:
        if th is not None:
            try:
                th.quit()
                th.wait(5000)
            except Exception:
                pass


@pytest.fixture
def nuc(qapp):
    img, labels = nuclei_volume()
    h = Harness(labels, images={"nuc": img})
    h.place_box(box_of(labels.shape))
    yield h
    _join(h._add_thread, h._add_info_thread,
          *[t for t, _w in h._refine_parked_threads])


def test_a_preview_lands_and_busy_clears(qapp, nuc):
    nuc._preview_add()
    assert nuc._add_busy and nuc._add_thread is not None
    assert _pump(qapp, lambda: nuc._add_preview_result is not None), (
        "the preview never landed")
    assert _pump(qapp, lambda: not nuc._add_busy), (
        "busy stuck on after the answer: the thread was never told to quit")
    assert nuc._add_thread is None
    assert nuc.adv.last("preview_enabled") is True


def test_stop_then_a_preview_works_once_the_thread_has_finished(qapp, nuc):
    nuc._preview_add()
    first = nuc._add_thread
    nuc._stop_add_preview()
    nuc._preview_add()                       # immediate: still refused
    assert nuc._add_transient == AP.ADD_S1_BUSY_REFUSED
    assert nuc._add_thread is first
    assert _pump(qapp, lambda: not nuc._add_busy), (
        "a Stopped preview never released busy")
    assert nuc._add_preview_result is None
    nuc._add_transient = None
    nuc._preview_add()
    assert nuc._add_busy and nuc._add_thread is not None
    assert nuc._add_thread is not first
    assert _pump(qapp, lambda: nuc._add_preview_result is not None)
    assert _pump(qapp, lambda: not nuc._add_busy)


def test_the_info_worker_answers_again_after_a_box_move(qapp):
    from .test_reshape_region import _tissue

    mem, nucl, cells = _tissue()
    h = Harness(np.zeros_like(cells), images={"mem": mem, "nuc": nucl},
                spacing=(0.5, 0.5, 0.5))
    h.adv.preset = "cells"
    h.adv.source = ("labels", "run1")
    h._add_run_guide = cells.copy()
    h.place_box(tuple(slice(0, n) for n in cells.shape))
    try:
        h._request_add_box_info()
        assert _pump(qapp, lambda: h._add_info is not None), "no first answer"
        assert _pump(qapp, lambda: h._add_info_thread is None), (
            "the info thread never finished")
        b = h._add_box
        h._on_add_box_changed(b.cz, b.cy, b.cx, b.sz - 2, b.sy - 2, b.sx - 2)
        assert h._add_info is None
        h._request_add_box_info()
        assert h._add_info_thread is not None, (
            "the second request only set dirty: the first thread is stuck")
        assert _pump(qapp, lambda: h._add_info is not None), (
            "no answer after the box moved")
        assert _pump(qapp, lambda: h._add_info_thread is None)
    finally:
        _join(h._add_info_thread, h._add_thread)

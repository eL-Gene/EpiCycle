"""③'s load on a worker (I-1, I-13, grill 1 G1-22).

The REAL panel with its controller and a REAL ``QThread`` (the Add-rebuild
lesson: a fake thread that finishes by itself hides a thread that never
quits).  A row click reads the stack on ``_StackLoadWorker`` with
``_load_in_flight`` raised; the UI stays live while it reads; a finish for
a request that is no longer current changes nothing; the worker never
writes the record; ``shutdown`` stops it.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import QTimer

from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.segmentation import controller as C

from ._step4_panel import close, make_workspace, panel, pump


def _reconstruct(ws: Path, c, name: str, n: int = 2) -> Path:
    run = ws / "artifacts" / "step4" / "runs" / f"r_{Path(name).stem}"
    run.mkdir(parents=True, exist_ok=True)
    lab = np.zeros((6, 24, 24), np.uint32)
    for i in range(n):
        lab[1:3, 2 + 5 * i:5 + 5 * i, 2:5] = i + 1
    path = run / "labels_3d.tif"
    tifffile.imwrite(str(path), lab)
    c.state_manager.set_stack_status(name, "reconstructed",
                                     labels_path=str(path))
    return path


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"))
    w = panel(qapp, ws)
    c = w.controller
    _reconstruct(ws, c, "P1.tif", 2)
    _reconstruct(ws, c, "P2.tif", 3)
    c._refresh_stacks()
    pump(qapp)
    w._ws = ws
    yield w
    close(w)
    pump(qapp)
    for th in getattr(w.controller, "_retired_threads", []):
        th.wait(3000)


def _wait(qapp, cond, timeout=20.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.005)
    return cond()


def test_a_row_click_loads_on_a_real_thread(qapp, w):
    c = w.controller
    assert w.qc_stacks_list.count() == 2
    item = w.qc_stacks_list.item(0)
    w.qc_stacks_list.itemClicked.emit(item)       # the user's click
    assert c._load_in_flight                      # the flag, at once
    assert c._load_thread is not None
    thread = c._load_thread
    assert _wait(qapp, lambda: not c._load_in_flight)
    assert thread.wait(3000)                      # quit was reached
    assert c._current_stack_name == item.data(0x0100)
    assert c._labels_volume is not None and c._n_cells in (2, 3)
    assert w.stack_card_name.text() == c._current_stack_name
    rec = c.state_manager.get_record(c._current_stack_name)
    assert Path(rec["original_labels_path"]).exists()


def test_the_ui_is_never_blocked_while_it_reads(qapp, w, monkeypatch):
    """A slow read (0.6 s) runs while a 20 ms timer keeps ticking on the
    main thread: the event loop never waits for the read."""
    real = tifffile.imread

    def slow(path, *a, **k):
        time.sleep(0.6)
        return real(path, *a, **k)
    monkeypatch.setattr(tifffile, "imread", slow)
    ticks = []
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    c = w.controller
    t0 = time.monotonic()
    c._request_load("P1.tif")
    launch_ms = (time.monotonic() - t0) * 1000
    assert launch_ms < 300, launch_ms                # the click returns
    assert _wait(qapp, lambda: c._current_stack_name == "P1.tif")
    timer.stop()
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert len(ticks) > 20
    assert max(gaps) < 0.4, max(gaps)


def test_a_stale_load_is_discarded(qapp, w):
    """A finish whose token is no longer current changes nothing (I-1)."""
    c = w.controller
    c._request_load("P1.tif")
    assert _wait(qapp, lambda: c._current_stack_name == "P1.tif")
    vol = c._labels_volume
    fake = {"labels": np.ones((2, 2, 2), np.uint32), "n_cells": 1,
            "backup": "", "raw": None, "source_path": ""}
    c._on_load_worker_finished(fake, "P2.tif", 999)
    assert c._current_stack_name == "P1.tif"
    assert c._labels_volume is vol


def test_the_worker_never_writes_the_record(qapp, w, monkeypatch):
    """G1-22: only the finish handler writes ``original_labels_path``."""
    c = w.controller
    writes = []
    orig = c.state_manager.update_record
    monkeypatch.setattr(c.state_manager, "update_record",
                        lambda *a, **k: writes.append(
                            (__import__("threading").get_ident(), a, k))
                        or orig(*a, **k))
    main = __import__("threading").get_ident()
    c._request_load("P2.tif")
    assert _wait(qapp, lambda: c._current_stack_name == "P2.tif")
    assert writes and all(tid == main for tid, _a, _k in writes)


def test_a_failed_read_says_so(qapp, w, monkeypatch):
    c = w.controller

    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(tifffile, "imread", boom)
    c._request_load("P1.tif")
    assert _wait(qapp, lambda: not c._load_in_flight)
    assert c._current_stack_name is None
    assert w.qc_status_line.text() == T.MT_LOAD_FAILED.format(stack="P1.tif")
    assert any("disk gone" in l for l in
               w.log_output.toPlainText().splitlines())


def test_a_click_while_loading_is_refused(qapp, w, monkeypatch):
    real = tifffile.imread

    def slow(path, *a, **k):
        time.sleep(0.4)
        return real(path, *a, **k)
    monkeypatch.setattr(tifffile, "imread", slow)
    c = w.controller
    c._request_load("P1.tif")
    c._request_load("P2.tif")                    # refused: a load runs
    assert _wait(qapp, lambda: not c._load_in_flight)
    assert c._current_stack_name == "P1.tif"
    assert "refused" in w.log_output.toPlainText()


def test_the_rows_cannot_be_clicked_while_it_loads(qapp, w, monkeypatch):
    real = tifffile.imread

    def slow(path, *a, **k):
        time.sleep(0.4)
        return real(path, *a, **k)
    monkeypatch.setattr(tifffile, "imread", slow)
    c = w.controller
    c._request_load("P1.tif")
    assert not w.qc_stacks_list.isEnabled()
    assert w.mt_lock_line.text() == T.STEP4_LOCK_LOAD.format(stack="P1.tif")
    assert _wait(qapp, lambda: not c._load_in_flight)
    assert w.qc_stacks_list.isEnabled()
    assert w.mt_lock_line.text() == ""


def test_shutdown_stops_it(qapp, w, monkeypatch):
    real = tifffile.imread

    def slow(path, *a, **k):
        time.sleep(0.5)
        return real(path, *a, **k)
    monkeypatch.setattr(tifffile, "imread", slow)
    c = w.controller
    c._request_load("P1.tif")
    thread = c._load_thread
    c.shutdown()
    assert thread.wait(3000)
    pump(qapp, 10)
    assert c._current_stack_name is None          # the late finish is stale


def test_the_worker_emits_once_on_every_exit():
    """I-13, without Qt threads: a worker whose read raises emits error."""
    w = C._StackLoadWorker(stack_name="x", labels_path="Z:/nope.tif",
                           token=4)
    got = []
    w.finished.connect(lambda *a: got.append(("finished", a)))
    w.error.connect(lambda *a: got.append(("error", a)))
    w.run()
    assert [g[0] for g in got] == ["error"]
    assert got[0][1][1:] == ("x", 4)

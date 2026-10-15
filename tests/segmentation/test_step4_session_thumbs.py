"""① Session thumbnails wired to the worker
(I-20): placeholders until ``ready``; a ``ready`` fills the row
keyed by its stack PATH; a ``ready`` for a stack no longer listed is
dropped; ``shutdown()`` stops the worker; the main thread never reads a
TIFF for a thumbnail."""
from __future__ import annotations

import threading
import time

import pytest
from qtpy.QtCore import Qt

from napari_epicycle.segmentation import _stack_thumbs as ST

from ._step4_panel import close, make_workspace, panel, pump

ROLE_THUMB = Qt.UserRole + 3


def _thumbs(w):
    lw = w.session_stacks_list
    return {lw.item(i).data(Qt.UserRole): lw.item(i).data(ROLE_THUMB)
            for i in range(lw.count())}


def _png_bytes():
    from qtpy.QtCore import QBuffer, QByteArray, QIODevice
    from qtpy.QtGui import QColor, QImage
    img = QImage(8, 8, QImage.Format_RGB32)
    img.fill(QColor("#336699"))
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


@pytest.fixture
def ws(tmp_path):
    return make_workspace(tmp_path, patches=("P1.tif", "P2.tif"))


def test_rows_wait_with_a_placeholder_and_ready_fills_by_path(
        qapp, ws, monkeypatch):
    started = []
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (started.append((a, k)) or
                                         (None, None)))
    w = panel(qapp, ws)
    try:
        assert started, "one worker per Session refresh"
        requests = started[-1][0][1]
        assert sorted(r.name for r in requests) == ["P1.tif", "P2.tif"]
        assert all(r.axes == "CZYX" for r in requests)
        assert _thumbs(w) == {"P1.tif": None, "P2.tif": None}
        c = w.controller
        p1 = next(r.path for r in requests if r.name == "P1.tif")
        c._on_thumb_ready("P1.tif", p1, "", _png_bytes())
        pump(qapp)
        thumbs = _thumbs(w)
        assert thumbs["P1.tif"] is not None and not thumbs["P1.tif"].isNull()
        assert thumbs["P2.tif"] is None
        # a stack no longer listed: dropped, nothing breaks
        c._on_thumb_ready("gone.tif", str(ws / "gone.tif"), "", _png_bytes())
        # the picture survives a list rebuild
        c._refresh_stacks()
        pump(qapp)
        assert _thumbs(w)["P1.tif"] is not None
    finally:
        close(w)


def test_a_cached_thumbnail_is_used_without_a_worker(qapp, ws, monkeypatch):
    w = panel(qapp, ws)
    try:
        deadline = time.monotonic() + 20
        while (None in _thumbs(w).values()
               and time.monotonic() < deadline):
            pump(qapp)
            time.sleep(0.02)
        assert None not in _thumbs(w).values()
        started = []
        monkeypatch.setattr(ST, "start_thumb_worker",
                            lambda *a, **k: (started.append(a) or
                                             (None, None)))
        w.controller._rescan_stacks(persist=True)
        pump(qapp)
        assert not started            # every picture came from the cache
        assert None not in _thumbs(w).values()
    finally:
        close(w)


def test_the_main_thread_never_reads_a_tiff_for_a_thumbnail(
        qapp, ws, monkeypatch):
    import tifffile
    w = panel(qapp, ws)
    try:
        c = w.controller
        # the listing (header reads) is done; now only the thumbnails run
        c._thumb_cache = None
        for p in (ws / "artifacts/step4/thumbs").glob("*.png"):
            p.unlink()
        calls = []
        real = tifffile.TiffFile.__init__

        def spy(self, *a, **k):
            calls.append(threading.current_thread() is threading.main_thread())
            return real(self, *a, **k)

        monkeypatch.setattr(tifffile.TiffFile, "__init__", spy)
        c._start_thumbs()
        deadline = time.monotonic() + 20
        while (None in _thumbs(w).values()
               and time.monotonic() < deadline):
            pump(qapp)
            time.sleep(0.02)
        assert None not in _thumbs(w).values()
        assert calls and not any(calls), calls
    finally:
        close(w)


def test_shutdown_stops_the_worker(qapp, ws):
    w = panel(qapp, ws)
    try:
        c = w.controller
        c._thumb_cache = None
        for p in (ws / "artifacts/step4/thumbs").glob("*.png"):
            p.unlink()
        c._start_thumbs()
        thread = c._thumb_thread
        assert thread is not None
        c.shutdown()
        assert c._thumb_thread is None
        assert not thread.isRunning()
    finally:
        close(w)


def test_an_adopt_writes_no_thumbnail_into_the_workspace(qapp, ws):
    """I12: the read-only adopt lists the stacks WITH pictures, but the PNGs
    go to a temp folder, never into the workspace."""
    from napari_epicycle.segmentation.ui import SegmentStack
    w = SegmentStack(viewer=None)
    try:
        before = sorted(p.relative_to(ws).as_posix() for p in ws.rglob("*"))
        w.controller.adopt_workspace(str(ws))
        deadline = time.monotonic() + 20
        while (None in _thumbs(w).values()
               and time.monotonic() < deadline):
            pump(qapp)
            time.sleep(0.02)
        assert None not in _thumbs(w).values()
        after = sorted(p.relative_to(ws).as_posix() for p in ws.rglob("*"))
        assert after == before
    finally:
        close(w)


def test_an_older_workers_picture_is_dropped(qapp, ws, monkeypatch):
    """A1-7: a ``ready`` carries the stamp its own worker was asked for;
    when the file has changed since (a newer stamp is listed), the picture
    is dropped instead of cached under the newer stamp."""
    started = []
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (started.append((a, k)) or
                                         (None, None)))
    w = panel(qapp, ws)
    try:
        c = w.controller
        on_ready = started[-1][1]["on_ready"]
        p1 = next(r.path for r in started[-1][0][1] if r.name == "P1.tif")
        from napari_epicycle.segmentation import _step4_rows as R
        key = R.norm_path(p1)
        old = c._thumb_requested[key]
        c._thumb_requested[key] = (old[0] + 1, old[1])     # file changed
        on_ready("P1.tif", p1, "", _png_bytes())
        pump(qapp)
        assert _thumbs(w)["P1.tif"] is None
        c._thumb_requested[key] = old                      # same stamp
        on_ready("P1.tif", p1, "", _png_bytes())
        pump(qapp)
        assert _thumbs(w)["P1.tif"] is not None
    finally:
        close(w)

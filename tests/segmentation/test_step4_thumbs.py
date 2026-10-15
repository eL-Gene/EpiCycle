"""Step 4 Session thumbnails (invariant I-20).

The pure parts, then a REAL ``QThread`` under a plain offscreen
``QApplication`` (no pytest-qt): a fake that emits ``finished`` by itself
would hide a thread that never quits.  ``qapp`` comes from
``tests/segmentation/conftest.py``.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
tifffile = pytest.importorskip("tifffile")

from napari_epicycle.segmentation import _stack_thumbs as T  # noqa: E402

DEADLINE_S = 20.0
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def tiff_calls(monkeypatch):
    """I-20: every tifffile read records its thread; the main thread raises."""
    calls = []
    real_tf, real_imread = tifffile.TiffFile, tifffile.imread

    def _guard(what):
        th = threading.current_thread()
        calls.append((what, th.name))
        if th is threading.main_thread():
            raise AssertionError(f"tifffile.{what} called on the main thread")

    class GuardedTiffFile(real_tf):
        def __init__(self, *a, **k):
            _guard("TiffFile")
            super().__init__(*a, **k)

    def guarded_imread(*a, **k):
        _guard("imread")
        return real_imread(*a, **k)

    monkeypatch.setattr(tifffile, "TiffFile", GuardedTiffFile)
    monkeypatch.setattr(tifffile, "imread", guarded_imread)
    return calls


def _write(path, arr, **kw):
    tifffile.imwrite(str(path), arr, **kw)
    return int(os.stat(path).st_mtime_ns)


def _req(name, path, axes, **kw):
    st = os.stat(path) if os.path.exists(path) else None
    return T.ThumbRequest(name, str(path), axes,
                          kw.get("mtime_ns", st.st_mtime_ns if st else 1),
                          kw.get("st_size", st.st_size if st else 0))


def _vol(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=shape, dtype=np.uint16)


def _pngs(d):
    d = Path(d)
    return sorted(p.name for p in d.glob("*.png")) if d.is_dir() else []


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_stride_keeps_small_and_shrinks_large():
    small = np.zeros((40, 50))
    assert T.thumb_stride(small, 64) is small
    assert T.thumb_stride(np.zeros((300, 130)), 64).shape == (75, 65)
    assert max(T.thumb_stride(np.zeros((640, 640)), 64).shape) == 64


def test_norm8_clips_to_uint8_and_survives_flat_and_nan_planes():
    a = np.linspace(0, 1000, 10000).reshape(100, 100)
    n = T.thumb_norm8(a)
    assert n.dtype == np.uint8 and n.min() == 0 and n.max() == 255
    assert not T.thumb_norm8(np.full((8, 8), 7.0)).any()
    b = a.copy()
    b[:10] = np.nan
    nb = T.thumb_norm8(b)
    assert nb.dtype == np.uint8 and nb.max() == 255 and not nb[:10].any()
    assert not T.thumb_norm8(np.full((4, 4), np.nan)).any()


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_compose_rgb_channels(n):
    """Decided 2026-10-06 (it was
    channel 0 red, 1 green, 2 blue): channel 0 MAGENTA (nuclei by
    EpiCycle's convention), channel 1 GREEN (membrane), a third channel
    adds blue; additive and clipped; one channel grey."""
    planes = np.zeros((n, 10, 12), np.uint16)
    for i in range(n):
        planes[i, :, 6:] = 100 * (i + 1)
    rgb = T.compose_rgb(planes, 64)
    assert rgb.shape == (10, 12, 3) and rgb.dtype == np.uint8
    if n == 1:
        assert np.array_equal(rgb[..., 0], rgb[..., 1])
        assert np.array_equal(rgb[..., 1], rgb[..., 2])
        assert rgb[..., 0].any()
    else:
        on = rgb[:, 6:]
        assert on[..., 0].min() == 255 and on[..., 1].min() == 255
        assert on[..., 2].min() == 255        # ch0's blue (+ ch2's, clipped)
        assert not rgb[:, :6].any()


def test_compose_rgb_magenta_and_green_one_at_a_time():
    """Each channel alone: channel 0 draws magenta, channel 1 green, a
    third channel blue."""
    planes = np.zeros((3, 4, 4), np.uint8)
    planes[0, 0, 0] = 200
    planes[1, 1, 1] = 200
    planes[2, 2, 2] = 200
    for i in range(3):
        planes[i, 3, 3] = 1                  # the normalisation's floor
    rgb = T.compose_rgb(planes, 64)
    assert tuple(rgb[0, 0]) == (255, 0, 255)
    assert tuple(rgb[1, 1]) == (0, 255, 0)
    assert tuple(rgb[2, 2]) == (0, 0, 255)
    two = T.compose_rgb(planes[:2], 64)
    assert tuple(two[0, 0]) == (255, 0, 255)
    assert tuple(two[1, 1]) == (0, 255, 0)
    assert T.THUMB_COLOURS[:2] == ((255, 0, 255), (0, 255, 0))


def test_compose_rgb_takes_other_colours():
    planes = np.zeros((2, 4, 4), np.uint8)
    planes[0, 0, 0] = 9
    planes[1, 1, 1] = 9
    rgb = T.compose_rgb(planes, 64, colours=((255, 0, 0), (0, 0, 255)))
    assert tuple(rgb[0, 0]) == (255, 0, 0)
    assert tuple(rgb[1, 1]) == (0, 0, 255)


def test_png_name_carries_path_hash_mtime_and_size():
    a = T.thumb_png_path("/c", "stack_1", "/x/a.tif", 123, 9)
    b = T.thumb_png_path("/c", "stack_1", "/y/a.tif", 123, 9)
    # the colour-version token (row 30): an old red / green PNG never hits
    assert a.name.startswith("stack_1.")
    # A3: the size the worker draws at is part of the name too
    assert a.name.endswith(f".123.9.{T.THUMB_STYLE}.s{T.THUMB_WIDE_PX}.png")
    assert T.THUMB_STYLE == "c2"
    assert a != b                                      # C1-3: folder in the key
    assert T.thumb_png_path("/c", "stack_1", "/x/a.tif", 123, 10) != a   # size
    k = T.ThumbCache.key
    assert k("x.tif", 5, 1) == k("./x.tif", 5, 1)
    assert k("x.tif", 5, 1) != k("x.tif", 6, 1) != k("x.tif", 5, 2)


def test_safe_stack_name_is_hardened():
    assert T.safe_stack_name("a/b:c") == "a_b_c"
    assert T.safe_stack_name("a\x01b\tc") == "a_b_c"
    assert T.safe_stack_name("CON").upper() != "CON"
    assert T.safe_stack_name("lpt1.x").split(".")[0].upper() != "LPT1"
    assert not T.safe_stack_name("name. ").endswith((".", " "))
    assert len(T.safe_stack_name("x" * 500)) <= T.MAX_NAME_LEN
    assert T.safe_stack_name("") == "_"


def test_stale_pngs_match_name_and_path_hash_only(tmp_path):
    keep = T.thumb_png_path(tmp_path, "stack", "/x/s.tif", 2, 5)
    old = T.thumb_png_path(tmp_path, "stack", "/x/s.tif", 1, 5)
    other_folder = T.thumb_png_path(tmp_path, "stack", "/y/s.tif", 1, 5)
    other_name = T.thumb_png_path(tmp_path, "stack_2", "/x/s.tif", 1, 5)
    for p in (keep, old, other_folder, other_name):
        p.write_bytes(b"x")
    stale = T.stale_pngs(tmp_path, "stack", "/x/s.tif", keep)
    assert [p.name for p in stale] == [old.name]


def test_an_old_colour_png_is_a_miss_and_is_swept(tmp_path):
    """Row 30: a PNG named before the colour token (today's red / green
    one, same mtime and size) is not a cache hit, and is deleted when the
    new one is written."""
    stack = tmp_path / "s.tif"
    tifffile.imwrite(str(stack), np.arange(2 * 3 * 8 * 8, dtype=np.uint16)
                     .reshape(2, 3, 8, 8), metadata={"axes": "CZYX"})
    st = stack.stat()
    cache = tmp_path / "th"
    cache.mkdir()
    new = T.thumb_png_path(cache, "s", str(stack), st.st_mtime_ns,
                           st.st_size)
    legacy = cache / (new.name[:-len(
        f".{T.THUMB_STYLE}.s{T.THUMB_WIDE_PX}.png")] + ".png")
    assert legacy.name.endswith(f".{st.st_mtime_ns}.{st.st_size}.png")
    legacy.write_bytes(b"old red/green png")
    assert T.stale_pngs(cache, "s", str(stack), new) == [legacy]
    ok, path, data = T.render_thumb(
        T.ThumbRequest("s", str(stack), "CZYX", st.st_mtime_ns, st.st_size),
        cache)
    assert ok and path == str(new) and data != b"old red/green png"
    assert not legacy.exists() and new.is_file()


def test_the_colour_token_has_the_shape_stale_pngs_recognises(tmp_path):
    """``stale_pngs`` recognises only a ``c<digits>`` colour token, so a
    ``THUMB_STYLE`` of another shape would orphan every PNG drawn with it:
    fail loudly here instead."""
    import re
    assert re.fullmatch(r"c\d+", T.THUMB_STYLE), T.THUMB_STYLE
    stack = tmp_path / "s.tif"
    stack.write_bytes(b"x")
    cache = tmp_path / "th"
    cache.mkdir()
    old = T.thumb_png_path(cache, "s", str(stack), 1, 1)
    old.write_bytes(b"png")
    new = T.thumb_png_path(cache, "s", str(stack), 2, 2)
    assert T.stale_pngs(cache, "s", str(stack), new) == [old]


def test_midz_index_follows_the_resolved_axes():
    assert T.midz_index("ZYX", (9, 8, 8), 0) == (4, slice(None), slice(None))
    assert T.midz_index("CZYX", (2, 9, 8, 8), 1)[:2] == (1, 4)
    assert T.midz_index("ZCYX", (9, 2, 8, 8), 1)[:2] == (4, 1)
    assert T.n_channels("CZYX", (5, 4, 8, 8)) == 3
    for axes, shape in [("ZYX", (2, 9, 8, 8)), ("XYZ", (8, 8, 8)),
                        ("TZYX", (2, 3, 8, 8)), ("ZZYX", (2, 2, 8, 8))]:
        with pytest.raises(ValueError):
            T.n_channels(axes, shape)


def test_czyx_and_zcyx_read_the_same_mid_plane(tmp_path):
    czyx = _vol((2, 7, 16, 16))
    _write(tmp_path / "a.tif", czyx)
    _write(tmp_path / "b.tif", np.ascontiguousarray(czyx.transpose(1, 0, 2, 3)))
    assert np.array_equal(T.read_midz_planes(str(tmp_path / "a.tif"), "CZYX"), czyx[:, 3])
    assert np.array_equal(T.read_midz_planes(str(tmp_path / "b.tif"), "ZCYX"), czyx[:, 3])


def test_planar_rgb_written_without_photometric_reads_right(tmp_path):
    """C1-1: imwrite without photometric stores C=3 as planar RGB pages."""
    zcyx = _vol((7, 3, 32, 32), 3)
    _write(tmp_path / "z.tif", zcyx)
    assert np.array_equal(T.read_midz_planes(str(tmp_path / "z.tif"), "ZCYX"), zcyx[3])
    czyx = _vol((2, 3, 32, 32), 4)                     # Z = 3
    _write(tmp_path / "c.tif", czyx)
    assert np.array_equal(T.read_midz_planes(str(tmp_path / "c.tif"), "CZYX"), czyx[:, 1])
    zyx = _vol((3, 20, 20), 5)                          # Z = 3, written as RGB
    _write(tmp_path / "s.tif", zyx)
    assert np.array_equal(T.read_midz_planes(str(tmp_path / "s.tif"), "ZYX"), zyx[1][None])


def test_a_file_whose_own_axes_disagree_fails(tmp_path, qapp):
    """C1-2: a CZYX-tagged file asked for as ZCYX is never drawn."""
    arr = _vol((2, 2, 16, 16))
    _write(tmp_path / "t.tif", arr, metadata={"axes": "CZYX"})
    with tifffile.TiffFile(str(tmp_path / "t.tif")) as tf:
        assert tf.series[0].axes == "CZYX"
    ok, reason = T.render_thumb(_req("t", tmp_path / "t.tif", "ZCYX"), tmp_path / "th")[:2]
    assert not ok and reason == T.THUMB_FAIL_AXES
    assert _pngs(tmp_path / "th") == []
    ok, _p = T.render_thumb(_req("t", tmp_path / "t.tif", "CZYX"), tmp_path / "th")[:2]
    assert ok


def test_wrong_axes_fail_and_write_nothing(tmp_path, qapp):
    _write(tmp_path / "s.tif", _vol((2, 5, 16, 16)))
    ok, reason = T.render_thumb(_req("s", tmp_path / "s.tif", "ZYX"), tmp_path / "th")[:2]
    assert not ok and reason == T.THUMB_FAIL_AXES
    assert _pngs(tmp_path / "th") == []


def test_missing_and_unreadable_fail(tmp_path, qapp):
    ok, reason = T.render_thumb(_req("m", tmp_path / "nope.tif", "ZYX"), tmp_path)[:2]
    assert not ok and reason == T.THUMB_FAIL_MISSING
    (tmp_path / "junk.tif").write_bytes(b"not a tiff")
    ok, reason = T.render_thumb(_req("j", tmp_path / "junk.tif", "ZYX"), tmp_path)[:2]
    assert not ok and reason == T.THUMB_FAIL_UNREADABLE


def test_render_returns_png_bytes_and_a_read_only_cache_still_draws(tmp_path, qapp, monkeypatch):
    _write(tmp_path / "s.tif", _vol((5, 16, 16)))
    ok, png, data = T.render_thumb(_req("s", tmp_path / "s.tif", "ZYX"), tmp_path / "th")
    assert ok and os.path.isfile(png) and data == Path(png).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def no_write(*a, **k):
        raise PermissionError("read-only")

    monkeypatch.setattr(T, "_write_atomic", no_write)
    ok, png2, data2 = T.render_thumb(_req("s", tmp_path / "s.tif", "ZYX", mtime_ns=7),
                                     tmp_path / "th")
    assert ok and png2 == "" and data2[:4] == b"\x89PNG"


def test_cache_hit_never_opens_the_tiff(tmp_path, monkeypatch, qapp):
    _write(tmp_path / "s.tif", _vol((5, 16, 16)))
    req = _req("s", tmp_path / "s.tif", "ZYX")
    ok, first, data = T.render_thumb(req, tmp_path / "th")
    assert ok and os.path.isfile(first)

    def boom(*a, **k):
        raise AssertionError("the TIFF was opened on a cache hit")

    monkeypatch.setattr(tifffile, "TiffFile", boom)
    monkeypatch.setattr(tifffile, "imread", boom)
    ok, again, data2 = T.render_thumb(req, tmp_path / "th")
    assert ok and again == first and data2 == data


def test_a_new_mtime_writes_a_new_png_and_deletes_the_old(tmp_path, qapp):
    _write(tmp_path / "s.tif", _vol((5, 16, 16)))
    d = tmp_path / "th"
    ok, p1, _ = T.render_thumb(_req("s", tmp_path / "s.tif", "ZYX"), d)
    ok2, p2, _ = T.render_thumb(_req("s", tmp_path / "s.tif", "ZYX", mtime_ns=99), d)
    assert ok and ok2 and p1 != p2
    assert not os.path.exists(p1) and os.path.exists(p2)


def test_png_is_the_wide_size(tmp_path, qapp):
    from qtpy.QtGui import QImage

    _write(tmp_path / "s.tif", _vol((3, 200, 100)), photometric="minisblack")
    ok, p, _ = T.render_thumb(_req("s", tmp_path / "s.tif", "ZYX"), tmp_path / "th")
    img = QImage(p)
    assert ok and max(img.width(), img.height()) == T.THUMB_WIDE_PX == 128
    assert f".s{T.THUMB_WIDE_PX}." in Path(p).name
    # a worker asked for another size draws that size, under another name
    ok2, p2, _ = T.render_thumb(_req("s", tmp_path / "s.tif", "ZYX"),
                                tmp_path / "th", T.THUMB_NARROW_PX)
    assert ok2 and p2 != p and ".s80." in Path(p2).name
    assert max(QImage(p2).width(), QImage(p2).height()) == 80


def test_stale_tmp_files_are_swept(tmp_path):
    d = tmp_path / "th"
    d.mkdir()
    old = d / "a.png.1.2.tmp"
    new = d / "b.png.1.2.tmp"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    past = time.time() - 3600
    os.utime(old, (past, past))
    T.sweep_stale_tmp(d)
    assert not old.exists() and new.exists()


def test_thumb_cache_lru_null_png_and_main_thread_only(tmp_path, qapp):
    _write(tmp_path / "s.tif", _vol((3, 20, 20)), photometric="minisblack")
    ok, png, data = T.render_thumb(_req("s", tmp_path / "s.tif", "ZYX"), tmp_path / "th")
    c = T.ThumbCache(maxsize=2)
    assert c.get("a", 1, 1) is None
    assert c.load("a", 1, 1, png_path=png) is not None
    c.load("b", 1, 1, png_bytes=data)                  # from bytes, no file
    c.get("a", 1, 1)
    c.load("c", 1, 1, png_path=png)                    # evicts b
    assert len(c) == 2 and c.get("b", 1, 1) is None and c.get("a", 1, 1) is not None

    bad = tmp_path / "th" / "bad.png"                  # C1-5: a null PNG is deleted
    bad.write_bytes(b"not a png")
    assert c.load("d", 1, 1, png_path=bad) is None
    assert not bad.exists()

    errors = []

    def off_main():
        try:
            c.load("e", 1, 1, png_path=png)
        except RuntimeError as exc:
            errors.append(exc)

    th = threading.Thread(target=off_main)
    th.start()
    th.join()
    assert errors, "ThumbCache.load ran off the main thread"


# ---------------------------------------------------------------------------
# Queue closing (T1-3)
# ---------------------------------------------------------------------------

def test_enqueue_after_the_queue_drained_is_refused(tmp_path, qapp):
    _write(tmp_path / "s.tif", _vol((4, 20, 20)))
    w = T._StackThumbWorker([_req("s", tmp_path / "s.tif", "ZYX")], tmp_path / "th")
    assert w.enqueue(_req("s2", tmp_path / "s.tif", "ZYX")) is True
    w.run()
    assert w.enqueue(_req("s3", tmp_path / "s.tif", "ZYX")) is False


def test_enqueue_from_on_finished_is_refused_never_dropped(tmp_path, qapp):
    _write(tmp_path / "s.tif", _vol((4, 20, 20)))
    w = T._StackThumbWorker([_req("s", tmp_path / "s.tif", "ZYX")], tmp_path / "th")
    ready, answers = [], []

    class Slots:
        def on_ready(self, name, path, png, data):
            ready.append(name)

        def on_finished(self):
            answers.append(w.enqueue(_req("late", tmp_path / "s.tif", "ZYX")))

    s = Slots()
    w.ready.connect(s.on_ready)
    w.finished.connect(s.on_finished)
    w.run()
    assert answers == [False] or "late" in ready
    assert "late" not in ready or answers == [True]


def test_cancel_drains_and_closes_the_queue(tmp_path, qapp):
    _write(tmp_path / "s.tif", _vol((4, 20, 20)))
    reqs = [_req(f"s{i}", tmp_path / "s.tif", "ZYX") for i in range(5)]
    w = T._StackThumbWorker(reqs, tmp_path / "th")
    assert w.pending() == 5
    w.cancel()
    assert w.pending() == 0 and w.enqueue(reqs[0]) is False
    got = []

    class S:
        def fin(self):
            got.append(1)

    s = S()
    w.finished.connect(s.fin)
    w.run()
    assert got == [1] and _pngs(tmp_path / "th") == []


def test_run_emits_error_and_finished_after_an_unexpected_error(tmp_path, qapp, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(T, "render_thumb", boom)
    w = T._StackThumbWorker([("s", "x", "ZYX", 1, 1)], tmp_path)
    errs, fin = [], []

    class S:
        def e(self, m):
            errs.append(m)

        def f(self):
            fin.append(1)

    s = S()
    w.error.connect(s.e)
    w.finished.connect(s.f)
    w.run()
    assert errs and fin == [1]


# ---------------------------------------------------------------------------
# A real QThread
# ---------------------------------------------------------------------------

def _wait_finished(app, thread, deadline=DEADLINE_S):
    from qtpy.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    thread.finished.connect(loop.quit)           # connect BEFORE the check
    if thread.isFinished():
        thread.finished.disconnect(loop.quit)
        ok = True
    else:
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(int(deadline * 1000))
        loop.exec_()
        timer.stop()
        ok = thread.isFinished()
    t0 = time.monotonic()                        # drain queued ready / failed
    while time.monotonic() - t0 < 0.2:
        app.processEvents()
    return ok


class _Spy:
    def __init__(self):
        self.ready, self.failed, self.errors, self.finished = [], [], [], 0

    def on_ready(self, name, stack_path, png_path, data):
        self.ready.append((name, stack_path, png_path, data,
                           threading.current_thread() is threading.main_thread()))

    def on_failed(self, name, stack_path, reason):
        self.failed.append((name, reason))

    def on_error(self, msg):
        self.errors.append(msg)

    def on_finished(self):
        self.finished += 1


def _start(tmp_path, requests, spy, parent=None):
    return T.start_thumb_worker(
        parent, requests, tmp_path / "th", on_ready=spy.on_ready,
        on_failed=spy.on_failed, on_finished=spy.on_finished,
        on_error=spy.on_error)


def _park(thread, worker):
    """Tests' finally: stop; a stuck thread stays parked in the live set."""
    T.stop_thumb_worker(thread, worker)


def test_real_qthread_writes_two_pngs_and_quits(tmp_path, qapp, tiff_calls):
    _write(tmp_path / "zyx.tif", _vol((6, 40, 30), 1))
    _write(tmp_path / "czyx.tif", _vol((2, 6, 40, 30), 2))
    reqs = [_req("zyx", tmp_path / "zyx.tif", "ZYX"),
            _req("czyx", tmp_path / "czyx.tif", "CZYX")]
    spy = _Spy()
    thread, worker = _start(tmp_path, reqs, spy)
    try:
        assert thread in T._LIVE_THUMB_WORKERS
        assert _wait_finished(qapp, thread), "the thumb thread never finished"
        assert not thread.isRunning()
        assert thread.quit_called, "finished never reached thread.quit"
        assert sorted(r[0] for r in spy.ready) == ["czyx", "zyx"]
        assert all(r[4] for r in spy.ready)
        for _n, sp, p, data, _m in spy.ready:
            assert os.path.isfile(p) and Path(p).read_bytes() == data
            assert os.path.isfile(sp)
        assert not spy.failed and not spy.errors and spy.finished == 1
        assert tiff_calls and all(th != "MainThread" for _w, th in tiff_calls)
        assert thread not in T._LIVE_THUMB_WORKERS
    finally:
        _park(thread, worker)


def test_two_same_name_stacks_in_different_folders(tmp_path, qapp):
    """C1-3: same stack name, two folders, one run -> two PNGs that load."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write(tmp_path / "a" / "s.tif", _vol((5, 20, 20), 1))
    _write(tmp_path / "b" / "s.tif", _vol((5, 20, 20), 2))
    reqs = [_req("s", tmp_path / "a" / "s.tif", "ZYX"),
            _req("s", tmp_path / "b" / "s.tif", "ZYX")]
    spy = _Spy()
    thread, worker = _start(tmp_path, reqs, spy)
    try:
        assert _wait_finished(qapp, thread)
        assert len(spy.ready) == 2
        assert {r[1] for r in spy.ready} == {str(tmp_path / "a" / "s.tif"),
                                             str(tmp_path / "b" / "s.tif")}
        pngs = {r[2] for r in spy.ready}
        assert len(pngs) == 2 and all(os.path.isfile(p) for p in pngs)
        c = T.ThumbCache()
        for _n, sp, p, _d, _m in spy.ready:
            assert c.load(sp, 1, 1, png_path=p) is not None
    finally:
        _park(thread, worker)


def test_real_qthread_reload_is_a_cache_hit_and_a_bad_stack_fails(
        tmp_path, qapp, tiff_calls):
    _write(tmp_path / "zyx.tif", _vol((6, 40, 30), 1))
    _write(tmp_path / "bad.tif", _vol((2, 6, 40, 30), 2))
    spy = _Spy()
    th, w = _start(tmp_path, [_req("zyx", tmp_path / "zyx.tif", "ZYX")], spy)
    try:
        assert _wait_finished(qapp, th)
    finally:
        _park(th, w)
    n_reads = len(tiff_calls)
    assert n_reads >= 1

    spy2 = _Spy()
    reqs = [_req("zyx", tmp_path / "zyx.tif", "ZYX"),
            _req("bad", tmp_path / "bad.tif", "ZYX")]      # axes do not fit
    th2, w2 = _start(tmp_path, reqs, spy2)
    try:
        assert _wait_finished(qapp, th2)
        assert th2.quit_called and not th2.isRunning()
        assert [r[0] for r in spy2.ready] == ["zyx"]
        assert spy2.failed == [("bad", T.THUMB_FAIL_AXES)]
        assert len(tiff_calls) == n_reads + 1                # the hit read nothing
        assert not any(n.startswith("bad.") for n in _pngs(tmp_path / "th"))
    finally:
        _park(th2, w2)


def test_the_thread_holds_the_worker(tmp_path, qapp):
    """T1-2: a caller keeping only the thread cannot lose the worker."""
    _write(tmp_path / "s.tif", _vol((4, 20, 20)))
    spy = _Spy()
    thread, worker = _start(tmp_path, [_req("s", tmp_path / "s.tif", "ZYX")], spy)
    assert thread._thumb_worker is worker
    del worker
    gc.collect()
    try:
        assert _wait_finished(qapp, thread)
        assert [r[0] for r in spy.ready] == ["s"]
    finally:
        _park(thread, None)


def test_wait_finished_is_false_without_a_quit_connection(tmp_path, qapp):
    """Negative control: with no finished -> quit, the thread never ends."""
    thread = T._ThumbThread()
    worker = T._StackThumbWorker([], tmp_path / "th")
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    thread.start()
    try:
        assert _wait_finished(qapp, thread, 1.0) is False
        assert thread.isRunning()
    finally:
        thread.quit()
        assert thread.wait(5000)


def test_stop_disconnects_the_callers_slots(tmp_path, qapp):
    """T1-4: a ready already queued when stop runs never reaches the caller."""
    _write(tmp_path / "s.tif", _vol((4, 20, 20)))
    spy = _Spy()
    thread, worker = _start(tmp_path, [_req("s", tmp_path / "s.tif", "ZYX")], spy)
    assert thread.wait(int(DEADLINE_S * 1000))     # emitted, not yet delivered
    T.stop_thumb_worker(thread, worker)
    for _ in range(50):
        qapp.processEvents()
    assert spy.ready == [] and spy.finished == 0


def test_a_widget_parent_is_ignored_and_the_set_empties():
    """T1-1: deleting the would-be parent while the worker runs is safe."""
    code = textwrap.dedent(f"""
        import os, sys, threading, time, tempfile, gc
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        sys.path.insert(0, {str(REPO)!r})
        import numpy as np, tifffile
        from qtpy.QtWidgets import QApplication, QWidget
        from qtpy import sip
        from napari_epicycle.segmentation import _stack_thumbs as T
        app = QApplication([])
        gate = threading.Event()
        real = T.render_thumb
        def blocked(*a, **k):
            gate.wait(20)
            return real(*a, **k)
        T.render_thumb = blocked
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, "s.tif")
        tifffile.imwrite(p, np.ones((4, 16, 16), np.uint16))
        st = os.stat(p)
        w = QWidget()
        th, wk = T.start_thumb_worker(w, [("s", p, "ZYX", st.st_mtime_ns, st.st_size)],
                                      os.path.join(tmp, "th"))
        assert th.parent() is None, "thread parented"
        del th, wk
        time.sleep(0.2)
        sip.delete(w)
        del w
        gc.collect()
        for _ in range(20):
            app.processEvents()
        assert len(T._LIVE_THUMB_WORKERS) == 1
        gate.set()
        t0 = time.monotonic()
        while T._LIVE_THUMB_WORKERS and time.monotonic() - t0 < 20:
            app.processEvents(); time.sleep(0.01)
        assert not T._LIVE_THUMB_WORKERS, "the live set never emptied"
        print("CHILD-OK")
    """)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=180, env=env, cwd=str(REPO))
    assert r.returncode == 0, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    assert "CHILD-OK" in r.stdout


def test_without_stop_the_queued_ready_is_delivered(tmp_path, qapp):
    """Control for the test above: the pending ready is really in the queue."""
    _write(tmp_path / "s.tif", _vol((4, 20, 20)))
    spy = _Spy()
    thread, worker = _start(tmp_path, [_req("s", tmp_path / "s.tif", "ZYX")], spy)
    try:
        assert thread.wait(int(DEADLINE_S * 1000))
        for _ in range(50):
            qapp.processEvents()
        assert [r[0] for r in spy.ready] == ["s"] and spy.finished == 1
    finally:
        _park(thread, worker)

"""Step 3's thumbnail engine, brick V2a.

No UI: the request builder, the cache folder, the start / stop wrappers over
the SHARED worker (``ui_shell/_stack_thumbs.py``), the sweep of PNGs whose
patch is no longer listed, and the failure tooltip.

The owner's rule (2026-10-07): a ONE-channel patch -- a 3-D ``ZYX`` file, or
a 4-D ``ZCYX`` / ``CZYX`` file with C = 1 -- draws GREY, never tinted; two
channels draw magenta (channel 0) + green (channel 1), a third adds blue; the
axes come from Step 3's PER-PATCH resolver, so a mixed folder works row by
row; no ``tifffile`` call on the main thread; a failure is a console line and
a tooltip, never a crash.

Every picture here is made by the REAL worker on a REAL ``QThread`` under a
plain offscreen ``QApplication`` (no pytest-qt), as
``tests/segmentation/test_step4_thumbs.py`` drives it.
"""

from __future__ import annotations

import ast
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
tifffile = pytest.importorskip("tifffile")

from napari_epicycle.fusion_reconstruction import _step3_thumbs as S3  # noqa: E402
from napari_epicycle.ui_shell import _stack_thumbs as T  # noqa: E402

DEADLINE_S = 20.0
MODULE = (Path(__file__).resolve().parents[2] / "napari_epicycle"
          / "fusion_reconstruction" / "_step3_thumbs.py")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def tiff_calls(monkeypatch):
    """Every tifffile read records its thread; the main thread RAISES."""
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


def _vol(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=shape, dtype=np.uint16)


def _write(path, arr):
    tifffile.imwrite(str(path), arr)
    return Path(path)


def _resolved_axes(path) -> str:
    """Step 3's own per-patch resolver, header only (the controller's
    ``_detect_and_cache_patch_axes``).  Runs in the test BEFORE any spy."""
    from napari_epicycle.fusion_reconstruction.qc_features import (
        resolve_patch_axes)

    with tifffile.TiffFile(str(path)) as tf:
        shape = tf.series[0].shape
    view = np.broadcast_to(np.uint8(0), shape)
    return resolve_patch_axes(view, image_path=str(path), fallback="CZYX")


def _wait_finished(app, thread, deadline=DEADLINE_S):
    from qtpy.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    thread.finished.connect(loop.quit)
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
        self.ready, self.failed = [], []

    def on_ready(self, name, stack_path, png_path, data):
        self.ready.append((name, stack_path, png_path, bytes(data),
                           threading.current_thread() is threading.main_thread()))

    def on_failed(self, name, stack_path, reason):
        self.failed.append((name, stack_path, reason))

    def picture(self, name):
        hits = [r for r in self.ready if r[0] == name]
        assert len(hits) == 1, f"{name}: {len(hits)} pictures"
        return _decode(hits[0][3])


def _decode(png_bytes) -> np.ndarray:
    """PNG bytes -> ``(h, w, 3)`` uint8 (QImage, in the test's main thread)."""
    from qtpy.QtGui import QImage

    img = QImage()
    assert img.loadFromData(bytes(png_bytes), "PNG")
    img = img.convertToFormat(QImage.Format_RGB888)
    h, w, bpl = img.height(), img.width(), img.bytesPerLine()
    ptr = img.constBits()
    try:
        ptr.setsize(h * bpl)                       # PyQt5
    except AttributeError:
        pass
    buf = np.frombuffer(bytes(ptr) if not hasattr(ptr, "asstring")
                        else ptr.asstring(h * bpl), np.uint8)
    return buf.reshape(h, bpl)[:, : w * 3].reshape(h, w, 3).copy()


def _run(app, tmp_path, requests, cache="th", state=None):
    spy = _Spy()
    st = S3.start_step3_thumbs(None, requests, tmp_path / cache,
                               spy.on_ready, spy.on_failed, state=state)
    try:
        if st.thread is not None:
            assert _wait_finished(app, st.thread), "the thumb thread never ended"
    finally:
        S3.stop_step3_thumbs(st)
    return spy, st


def _is_grey(rgb):
    return (np.array_equal(rgb[..., 0], rgb[..., 1])
            and np.array_equal(rgb[..., 1], rgb[..., 2]))


def _near_inferno(rgb):
    """Inferno's shape, robust to the PNG's smooth scaling of noisy planes
    (a chord between two table rows leaves the curve): red never trails
    green, and blue is not red -- magenta/green/grey would break one."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return bool((r - g >= -12).all() and (np.abs(r - b) > 8).any()
                and (g - r <= 12).all())


def _is_inferno(rgb):
    """A one-channel patch draws in inferno, not grey (A1, owner
    2026-10-10)."""
    return _near_inferno(rgb) and not _is_grey(rgb)


# ---------------------------------------------------------------------------
# The cache folder
# ---------------------------------------------------------------------------

def test_the_cache_folder_is_workspace_level_for_every_run(tmp_path):
    want = tmp_path / "artifacts" / "step3" / "thumbs"
    assert S3.thumb_cache_dir(tmp_path, None) == want
    assert S3.thumb_cache_dir(tmp_path, "cells") == want
    assert S3.thumb_cache_dir(str(tmp_path), "nuclei") == want
    assert not want.exists(), "naming the folder must not create it"


# ---------------------------------------------------------------------------
# The request builder (main thread: one os.stat per file, no file opened)
# ---------------------------------------------------------------------------

def test_requests_skip_npy_and_missing_files_and_carry_each_axes(
        tmp_path, tiff_calls):
    stack = tmp_path / "patches"
    stack.mkdir()
    _write(stack / "a.tif", _vol((4, 10, 12)))
    _write(stack / "b.TIFF", _vol((2, 4, 10, 12)))
    np.save(stack / "c.npy", _vol((4, 10, 12)))
    (stack / "d.tif").mkdir()                     # a folder, not a file
    axes = {"a.tif": "zyx", "b.TIFF": "CZYX", "c.npy": "ZYX",
            "gone.tif": "ZYX", "d.tif": "ZYX"}
    asked = []

    def axes_for(name):
        asked.append(name)
        return axes[name]

    reqs = S3.patch_thumb_requests(tmp_path, stack,
                                   ["a.tif", "b.TIFF", "c.npy", "gone.tif",
                                    "d.tif", "a.tif"], axes_for)
    assert [r.name for r in reqs] == ["a.tif", "b.TIFF"]
    assert [r.axes for r in reqs] == ["ZYX", "CZYX"]
    for r in reqs:
        assert isinstance(r, T.ThumbRequest)
        st = os.stat(r.path)
        assert Path(r.path) == stack / r.name
        assert (r.mtime_ns, r.st_size) == (st.st_mtime_ns, st.st_size)
    assert "c.npy" not in asked, ".npy is skipped before its axes are asked"
    assert tiff_calls == [], "the request builder opened a TIFF"


def test_a_relative_stack_folder_is_read_under_the_workspace(tmp_path):
    (tmp_path / "rel").mkdir()
    _write(tmp_path / "rel" / "a.tif", _vol((4, 10, 12)))
    reqs = S3.patch_thumb_requests(tmp_path, "rel", ["a.tif"],
                                   lambda _n: "ZYX")
    assert [Path(r.path) for r in reqs] == [tmp_path / "rel" / "a.tif"]


def test_unknown_or_failing_axes_skip_the_patch_and_never_raise(tmp_path):
    for n in ("a.tif", "b.tif", "c.tif"):
        _write(tmp_path / n, _vol((4, 10, 12)))

    def axes_for(name):
        if name == "a.tif":
            raise KeyError(name)
        return {"b.tif": "", "c.tif": "ZYX"}[name]

    reqs = S3.patch_thumb_requests(tmp_path, tmp_path,
                                   ["a.tif", "b.tif", "c.tif"], axes_for)
    assert [r.name for r in reqs] == ["c.tif"]


def test_no_stack_folder_or_no_names_is_no_request(tmp_path):
    assert S3.patch_thumb_requests(tmp_path, None, ["a.tif"],
                                   lambda _n: "ZYX") == []
    assert S3.patch_thumb_requests(tmp_path, tmp_path, [],
                                   lambda _n: "ZYX") == []
    assert S3.patch_thumb_requests(tmp_path, tmp_path / "nope", ["a.tif"],
                                   lambda _n: "ZYX") == []


# ---------------------------------------------------------------------------
# Grey vs colour, through the REAL worker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label, shape", [
    ("ZYX", (6, 40, 30)),
    ("ZCYX", (6, 1, 40, 30)),
    ("CZYX", (1, 6, 40, 30)),
])
def test_one_channel_draws_inferno_and_not_flat(tmp_path, qapp, tiff_calls,
                                             label, shape):
    path = tmp_path / ("p_" + label.lower() + ".tif")
    tifffile.imwrite(str(path), _vol(shape, 3))     # (the spy guards reads)
    axes = label
    reqs = S3.patch_thumb_requests(tmp_path, tmp_path, [path.name],
                                   lambda _n: axes)
    assert [r.axes for r in reqs] == [label]
    spy, _st = _run(qapp, tmp_path, reqs)
    assert not spy.failed, spy.failed
    rgb = spy.picture(path.name)
    assert _is_inferno(rgb), f"{label} C=1 is not inferno"
    assert rgb[..., 0].std() > 5, "a picture with no content"
    assert spy.ready[0][4], "ready did not land on the main thread"
    assert tiff_calls and all(th != "MainThread" for _w, th in tiff_calls)


def test_the_resolver_gives_one_channel_files_one_channel_axes(tmp_path):
    """The axes the controller will pass (Step 3's own resolver) for each
    one-channel layout are axes the worker draws grey."""
    for name, shape in (("zyx.tif", (6, 20, 24)), ("zc1.tif", (6, 1, 20, 24)),
                        ("c1z.tif", (1, 6, 20, 24))):
        _write(tmp_path / name, _vol(shape))
        axes = _resolved_axes(tmp_path / name)
        assert T.n_channels(axes, shape) == 1, (name, axes)


def _halves(shape_czyx, order):
    """Channel 0 bright on the LEFT half, channel 1 bright on the RIGHT
    half (channel 2, when there, bright in the TOP band), laid out as
    ``order``."""
    c, z, y, x = shape_czyx
    a = np.full(shape_czyx, 50, np.uint16)
    a[0, :, :, : x // 2] = 3000
    a[1, :, :, x // 2:] = 3000
    if c > 2:
        a[2, :, : y // 4, :] = 3000
    a += _vol(shape_czyx, 9) % 40                  # never a flat plane
    return a if order == "CZYX" else np.ascontiguousarray(
        a.transpose(1, 0, 2, 3))


@pytest.mark.parametrize("order", ["CZYX", "ZCYX"])
def test_two_channels_draw_magenta_and_green(tmp_path, qapp, order):
    arr = _halves((2, 6, 40, 40), order)
    _write(tmp_path / "two.tif", arr)
    reqs = S3.patch_thumb_requests(tmp_path, tmp_path, ["two.tif"],
                                   lambda _n: order)
    spy, _ = _run(qapp, tmp_path, reqs)
    rgb = spy.picture("two.tif").astype(int)
    h, w = rgb.shape[:2]
    left = rgb[h // 4: 3 * h // 4, 2: w // 2 - 4].reshape(-1, 3).mean(0)
    right = rgb[h // 4: 3 * h // 4, w // 2 + 4: w - 2].reshape(-1, 3).mean(0)
    # left: channel 0 only -> magenta (red = blue high, green low)
    assert left[0] > 150 and left[2] > 150 and left[1] < 60, left
    assert abs(left[0] - left[2]) < 10, left
    # right: channel 1 only -> green
    assert right[1] > 150 and right[0] < 60 and right[2] < 60, right
    assert not _is_grey(spy.picture("two.tif"))


def test_a_third_channel_adds_blue(tmp_path, qapp):
    arr = _halves((3, 6, 40, 40), "CZYX")
    _write(tmp_path / "three.tif", arr)
    reqs = S3.patch_thumb_requests(tmp_path, tmp_path, ["three.tif"],
                                   lambda _n: "CZYX")
    spy, _ = _run(qapp, tmp_path, reqs)
    rgb = spy.picture("three.tif").astype(int)
    h, w = rgb.shape[:2]
    # bottom right: channel 1 only -> green, no blue
    br = rgb[h // 2: h - 2, w // 2 + 4: w - 2].reshape(-1, 3).mean(0)
    assert br[1] > 150 and br[2] < 60, br
    # top right: channels 1 + 2 -> green + blue (cyan)
    tr = rgb[1: h // 4 - 3, w // 2 + 4: w - 2].reshape(-1, 3).mean(0)
    assert tr[1] > 150 and tr[2] > 150 and tr[0] < 60, tr


def test_the_same_bytes_read_as_zcyx_and_czyx_give_different_pictures(
        tmp_path, qapp):
    a = _vol((2, 2, 40, 40), 11)
    a[0, 1, :, :20] = 3900                          # differs by plane
    a[1, 0, 20:, :] = 3900
    _write(tmp_path / "amb.tif", a)
    r_z = S3.patch_thumb_requests(tmp_path, tmp_path, ["amb.tif"],
                                  lambda _n: "ZCYX")
    r_c = S3.patch_thumb_requests(tmp_path, tmp_path, ["amb.tif"],
                                  lambda _n: "CZYX")
    spy_z, _ = _run(qapp, tmp_path, r_z, cache="th_z")
    spy_c, _ = _run(qapp, tmp_path, r_c, cache="th_c")
    pz, pc = spy_z.picture("amb.tif"), spy_c.picture("amb.tif")
    assert pz.shape == pc.shape
    assert not np.array_equal(pz, pc), "the axes were not honoured"


def test_a_mixed_folder_draws_row_by_row(tmp_path, qapp, tiff_calls):
    """Grey and two-channel patches in ONE folder, each with its own
    resolved axes: the one-channel rows are inferno, the colour row is colour."""
    files = {
        "g3d.tif": _vol((6, 30, 30), 1),
        "gz1.tif": _vol((6, 1, 30, 30), 2),
        "gc1.tif": _vol((1, 6, 30, 30), 3),
        "col.tif": _halves((2, 6, 30, 30), "CZYX"),
        "old.npy": None,
    }
    from napari_epicycle.fusion_reconstruction.qc_features import (
        resolve_patch_axes)

    axes = {}
    for name, arr in files.items():
        if arr is None:
            np.save(tmp_path / name, _vol((6, 30, 30)))
            continue
        _write(tmp_path / name, arr)
        # Step 3's per-patch resolver on the patch's shape (no file opened)
        axes[name] = resolve_patch_axes(np.broadcast_to(np.uint8(0),
                                                        arr.shape))
    assert axes == {"g3d.tif": "ZYX", "gz1.tif": "ZCYX", "gc1.tif": "CZYX",
                    "col.tif": "CZYX"}
    reqs = S3.patch_thumb_requests(tmp_path, tmp_path, sorted(files),
                                   axes.__getitem__)
    assert sorted(r.name for r in reqs) == ["col.tif", "g3d.tif", "gc1.tif",
                                            "gz1.tif"]
    spy, _ = _run(qapp, tmp_path, reqs)
    assert not spy.failed, spy.failed
    for one in ("g3d.tif", "gz1.tif", "gc1.tif"):
        assert _is_inferno(spy.picture(one)), one
    assert not _is_grey(spy.picture("col.tif"))
    assert all(th != "MainThread" for _w, th in tiff_calls)


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

def test_the_png_name_carries_the_colour_token(tmp_path, qapp):
    _write(tmp_path / "p.tif", _vol((4, 20, 20)))
    reqs = S3.patch_thumb_requests(tmp_path, tmp_path, ["p.tif"],
                                   lambda _n: "ZYX")
    spy, _ = _run(qapp, tmp_path, reqs)
    png = Path(spy.ready[0][2])
    assert png.parent == tmp_path / "th"
    assert png.name.endswith(f".{T.THUMB_STYLE}i.s{T.THUMB_SIZE}.png")
    st = os.stat(tmp_path / "p.tif")
    assert png == T.thumb_png_path(tmp_path / "th", "p.tif",
                                   str(tmp_path / "p.tif"),
                                   st.st_mtime_ns, st.st_size,
                                   single="inferno")


def test_a_second_run_is_a_cache_hit_and_opens_no_tiff(tmp_path, qapp,
                                                        tiff_calls):
    _write(tmp_path / "p.tif", _vol((4, 20, 20)))
    _write(tmp_path / "q.tif", _vol((2, 4, 20, 20)))
    ax = {"p.tif": "ZYX", "q.tif": "CZYX"}
    reqs = S3.patch_thumb_requests(tmp_path, tmp_path, ["p.tif", "q.tif"],
                                   ax.__getitem__)
    spy1, _ = _run(qapp, tmp_path, reqs)
    opened = len(tiff_calls)
    assert opened >= 2 and len(spy1.ready) == 2
    spy2, _ = _run(qapp, tmp_path, reqs)
    assert len(tiff_calls) == opened, "a cached picture opened the TIFF"
    assert sorted(r[3] for r in spy2.ready) == sorted(r[3] for r in spy1.ready)


# ---------------------------------------------------------------------------
# Sweep of PNGs whose patch is no longer listed
# ---------------------------------------------------------------------------

def _png(d, name, style="c2"):
    p = Path(d) / f"{T.safe_stack_name(name)}.0123abcd.1700000000.4096.{style}.png"
    p.write_bytes(b"x")
    return p


def test_sweep_unlisted_removes_only_unlisted_pngs_inside_the_folder(tmp_path):
    ws = tmp_path / "ws"
    d = S3.thumb_cache_dir(ws, None)
    d.mkdir(parents=True)
    keep = [_png(d, "a.tif"), _png(d, "b.v2.tif"), _png(d, "a.tif", "c1")]
    gone = [_png(d, "old.tif"), _png(d, "b.tif")]
    tmp_file = d / "x.png.123.456.tmp"
    tmp_file.write_bytes(b"t")
    other = d / "notes.png"                        # not our name shape
    other.write_bytes(b"o")
    sub = d / "sub"
    sub.mkdir()
    nested = _png(sub, "old.tif")
    outside = _png(tmp_path, "old.tif")
    n = S3.sweep_unlisted(d, ["a.tif", "b.v2.tif"], workspace=ws)
    assert n == 2
    assert all(p.exists() for p in keep)
    assert not any(p.exists() for p in gone)
    for p in (tmp_file, other, nested, outside):
        assert p.exists(), p


def test_sweep_unlisted_refuses_a_folder_outside_the_workspace(tmp_path):
    d = tmp_path / "elsewhere"
    d.mkdir()
    victim = _png(d, "old.tif")
    assert S3.sweep_unlisted(d, [], workspace=tmp_path / "ws") == 0
    assert victim.exists()


def test_sweep_unlisted_on_a_missing_folder_is_zero(tmp_path):
    assert S3.sweep_unlisted(tmp_path / "nope", ["a.tif"]) == 0
    assert S3.sweep_unlisted(None, ["a.tif"]) == 0


def test_sweep_unlisted_keeps_a_png_whose_listed_name_was_made_safe(tmp_path):
    d = tmp_path / "th"
    d.mkdir()
    p = _png(d, "a:b.tif")                         # stored as "a_b.tif"
    assert S3.sweep_unlisted(d, ["a:b.tif"]) == 0
    assert p.exists()


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------

def test_stop_is_idempotent_and_leaves_no_live_thread(tmp_path, qapp):
    assert S3.stop_step3_thumbs(None) is True
    st = S3.Step3ThumbState()
    assert S3.stop_step3_thumbs(st) is True
    _write(tmp_path / "p.tif", _vol((4, 20, 20)))
    reqs = S3.patch_thumb_requests(tmp_path, tmp_path, ["p.tif"],
                                   lambda _n: "ZYX")
    spy = _Spy()
    st = S3.start_step3_thumbs(None, reqs, tmp_path / "th", spy.on_ready,
                               spy.on_failed)
    thread = st.thread
    assert thread is not None and st.running() in (True, False)
    assert S3.stop_step3_thumbs(st) is True
    assert st.thread is None and st.worker is None and not st.running()
    assert S3.stop_step3_thumbs(st) is True        # again: nothing to do
    assert thread.wait(int(DEADLINE_S * 1000))
    t0 = time.monotonic()
    while thread in T._LIVE_THUMB_WORKERS and time.monotonic() - t0 < 5:
        qapp.processEvents()
    assert thread not in T._LIVE_THUMB_WORKERS
    assert not thread.isRunning()


def test_a_new_batch_stops_the_older_one_first(tmp_path, qapp):
    names = [f"p{i:02d}.tif" for i in range(12)]
    for i, n in enumerate(names):
        _write(tmp_path / n, _vol((4, 60, 60), i))
    reqs = S3.patch_thumb_requests(tmp_path, tmp_path, names,
                                   lambda _n: "ZYX")
    old_spy, new_spy = _Spy(), _Spy()
    st = S3.start_step3_thumbs(None, reqs, tmp_path / "th", old_spy.on_ready,
                               old_spy.on_failed)
    old_thread, old_worker = st.thread, st.worker
    st2 = S3.start_step3_thumbs(None, reqs[:2], tmp_path / "th",
                                new_spy.on_ready, new_spy.on_failed, state=st)
    try:
        assert st2 is st
        assert st.thread is not old_thread
        assert old_worker.pending() == 0, "the old queue was not drained"
        assert _wait_finished(qapp, st.thread)
        assert old_thread.wait(int(DEADLINE_S * 1000))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.3:
            qapp.processEvents()
        assert old_spy.ready == [] and old_spy.failed == [], \
            "the stopped batch still reached its caller"
        assert sorted(r[0] for r in new_spy.ready) == names[:2]
    finally:
        S3.stop_step3_thumbs(st)


def test_an_empty_batch_still_stops_the_older_one(tmp_path, qapp):
    _write(tmp_path / "p.tif", _vol((4, 20, 20)))
    reqs = S3.patch_thumb_requests(tmp_path, tmp_path, ["p.tif"],
                                   lambda _n: "ZYX")
    spy = _Spy()
    st = S3.start_step3_thumbs(None, reqs, tmp_path / "th", spy.on_ready,
                               spy.on_failed)
    old_thread = st.thread
    S3.start_step3_thumbs(None, [], tmp_path / "th", spy.on_ready,
                          spy.on_failed, state=st)
    assert st.thread is None and st.worker is None
    assert old_thread.wait(int(DEADLINE_S * 1000))


# ---------------------------------------------------------------------------
# Failures: a sentence, never an exception
# ---------------------------------------------------------------------------

def test_a_corrupt_tiff_fails_with_a_sentence(tmp_path, qapp, capsys):
    (tmp_path / "bad.tif").write_bytes(b"II*\x00" + b"\x00garbage" * 20)
    _write(tmp_path / "good.tif", _vol((4, 20, 20)))
    reqs = S3.patch_thumb_requests(tmp_path, tmp_path, ["bad.tif", "good.tif"],
                                   lambda _n: "ZYX")
    assert [r.name for r in reqs] == ["bad.tif", "good.tif"]
    spy, _ = _run(qapp, tmp_path, reqs)
    assert [r[0] for r in spy.ready] == ["good.tif"]
    assert len(spy.failed) == 1
    name, path, reason = spy.failed[0]
    assert name == "bad.tif" and reason == T.THUMB_FAIL_UNREADABLE
    tip = S3.thumb_tooltip_for_failure(reason)
    assert tip.endswith(".") and "patch" in tip and "stack" not in tip
    assert not any(p.name.startswith("bad.tif.") for p in (tmp_path / "th")
                   .glob("*.png"))
    assert "bad.tif" in capsys.readouterr().out    # the console line


def test_failure_tooltips_speak_of_the_patch():
    for reason in (T.THUMB_FAIL_MISSING, T.THUMB_FAIL_UNREADABLE,
                   T.THUMB_FAIL_AXES, T.THUMB_FAIL_ENCODE):
        tip = S3.thumb_tooltip_for_failure(reason)
        assert tip.startswith(S3.S3_THUMB_FAILED.split("{")[0]), tip
        assert "stack" not in tip.lower() and "patch" in tip.lower(), tip
    assert "Something odd." in S3.thumb_tooltip_for_failure("Something odd.")
    assert S3.thumb_tooltip_for_failure("").strip()
    assert S3.thumb_tooltip_for_failure(None).strip()


def test_the_words_are_named_constants():
    for name in ("S3_THUMB_TIP", "S3_THUMB_PENDING", "S3_THUMB_FAILED"):
        assert isinstance(getattr(S3, name), str) and getattr(S3, name)


def test_the_module_imports_no_qt_and_no_tifffile_at_module_level():
    """The request builder and the sweep are Qt-free; only the start / stop
    wrappers reach the shared worker, and they import it inside."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        for m in mods:
            assert not m.startswith(("qtpy", "PyQt", "PySide", "tifffile")), m
    for node in ast.walk(tree):                     # nor anywhere inside
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("tifffile") for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tifffile")


def test_the_sweep_deletes_inferno_and_outlined_pngs_of_unlisted_patches(
        tmp_path):
    """A1 audit HIGH: ``.c2i.png`` / ``.c2i.l<ns>.png`` match the PNG shape."""
    ws = tmp_path / "ws"
    d = S3.thumb_cache_dir(ws, None)
    d.mkdir(parents=True)
    plain = T.thumb_png_path(d, "old.tif", "x/old.tif", 5, 6, single="inferno")
    lab = T.thumb_png_path(d, "old2.tif", "x/old2.tif", 5, 6,
                           single="inferno", labels_mtime_ns=1)
    for p in (plain, lab):
        p.write_bytes(b"x")
    assert S3.sweep_unlisted(d, ["other.tif"], workspace=ws) == 2
    assert not plain.exists() and not lab.exists()

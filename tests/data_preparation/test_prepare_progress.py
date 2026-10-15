"""Brick 5 fix 1 - the Prepare (convert to OME-Zarr) progress indicator.

Prepare is the longest operation in Step 1 and it showed a MARQUEE: a bar that
says "something is happening" for minutes without ever saying how much is left.
The converter has exactly one real denominator - the pyramid levels - so it now
reports each one through an additive ``on_level(done, total)`` callback, which
``_LargeDataPrepWorker`` re-emits as a Qt ``progress`` signal (worker thread ->
GUI thread, queued) and the controller paints with ``_busy_progress``.

Qt-free: the worker's ``run`` is exercised with a fake ``convert_to_ome_zarr``
and a recording signal; the controller handler runs on a ``__new__`` stand-in.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation import controller as ctlmod
from packages.epicycle_largedata.multiscale import convert_to_ome_zarr


# --------------------------------------------------------------------------- #
# 1. the converter reports every level
# --------------------------------------------------------------------------- #

def _src(tmp_path):
    arr = np.arange(16 * 96 * 96, dtype=np.uint16).reshape(16, 96, 96)
    p = tmp_path / "src.tif"
    tifffile.imwrite(str(p), arr)
    return p


def test_convert_reports_the_denominator_before_the_first_write(tmp_path):
    """A determinate bar is only useful if it is determinate from the START -
    level 0 is the long one, so the total must arrive before it, not after."""
    seen = []
    man = convert_to_ome_zarr(_src(tmp_path), tmp_path / "o.zarr",
                              chunk=32, downsample=2, min_size=48,
                              on_level=lambda d, t: seen.append((d, t)))
    n = len(man["level_shapes"])
    assert seen[0] == (0, n), seen
    assert seen[-1] == (n, n), seen
    assert [d for d, _ in seen] == list(range(n + 1)), "one tick per level"
    assert all(t == n for _, t in seen)


def test_a_raising_progress_callback_never_breaks_the_conversion(tmp_path):
    """Progress is decoration: a disconnected signal / torn-down widget must
    not turn a healthy multi-GB convert into a failure."""
    def boom(_d, _t):
        raise RuntimeError("widget gone")

    man = convert_to_ome_zarr(_src(tmp_path), tmp_path / "o.zarr",
                              chunk=32, downsample=2, min_size=48,
                              on_level=boom)
    assert man["level_shapes"]


def test_on_level_is_optional(tmp_path):
    """Every existing caller passes no callback."""
    assert convert_to_ome_zarr(_src(tmp_path), tmp_path / "o.zarr",
                               chunk=32, downsample=2, min_size=48)


# --------------------------------------------------------------------------- #
# 2. the worker re-emits it as a signal
# --------------------------------------------------------------------------- #

class _Sig:
    def __init__(self):
        self.emitted = []

    def emit(self, *a):
        self.emitted.append(a)


def test_worker_threads_on_level_into_its_progress_signal(monkeypatch):
    seen = {}

    def fake_convert(src, out, **kw):
        seen.update(kw)
        kw["on_level"](0, 3)
        kw["on_level"](1, 3)
        return {}

    monkeypatch.setattr(ctlmod, "convert_to_ome_zarr", fake_convert)

    w = ctlmod._LargeDataPrepWorker.__new__(ctlmod._LargeDataPrepWorker)
    w._src, w._out_store = "s.tif", "o.zarr"
    w._voxel_size = w._image_axes = None
    w._num_workers = w._chunk = w._compressor = None
    w._downsample = w._min_size = None
    w._cancel_requested = False
    w.progress, w.finished, w.error, w.cancelled = _Sig(), _Sig(), _Sig(), _Sig()

    w.run()

    assert callable(seen["on_level"])
    assert w.progress.emitted == [(0, 3), (1, 3)]
    assert w.finished.emitted, "the terminal signal still fires"
    # progress must be a real Qt Signal on the class, not a plain callback:
    # emitting from the worker thread is only safe because Qt queues it.
    assert hasattr(ctlmod._LargeDataPrepWorker, "progress")


# --------------------------------------------------------------------------- #
# 3. the controller handler
# --------------------------------------------------------------------------- #

class _Bar:
    """The `_busy_*` recorder used across the Step-1 busy tests."""

    def __init__(self):
        self.calls = []

    def _busy_progress(self, label, done, total):
        self.calls.append(("progress", label, done, total))

    def _busy_show(self, label):
        self.calls.append(("show", label))

    def _busy_hide(self, idle="— idle"):
        self.calls.append(("hide",))


def _ctl(token=7):
    c = ctlmod.DataPreparationController.__new__(ctlmod.DataPreparationController)
    c.widget = _Bar()
    c._largedata_token = token
    c._largedata_preparing = True
    c._largedata_busy_label = "Preparing wound.tif for large-data viewing…"
    c._left_step1 = False
    c._log = lambda *a, **k: None
    return c


def test_progress_paints_a_real_percentage():
    c = _ctl()
    c._on_large_data_progress(2, 5, 7)
    assert c.widget.calls == [
        ("progress", "Preparing wound.tif for large-data viewing…", 2, 5)]


def test_a_superseded_run_cannot_drive_the_new_bar():
    """The whole point of binding the token in the connect lambda: a cancelled
    Prepare keeps converting until its current level lands."""
    c = _ctl(token=8)
    c._on_large_data_progress(1, 5, 7)          # old token
    assert c.widget.calls == []


def test_progress_declines_after_the_step_is_left_or_the_lock_released():
    c = _ctl()
    c._left_step1 = True
    c._on_large_data_progress(1, 5, 7)
    c._left_step1 = False
    c._largedata_preparing = False              # terminal handler owns the bar
    c._on_large_data_progress(1, 5, 7)
    assert c.widget.calls == []


def test_progress_never_raises_on_a_dead_widget():
    c = _ctl()
    c.widget = None
    c._on_large_data_progress(1, 5, 7)          # must not raise


def test_the_launch_binds_the_token_in_the_progress_lambda():
    import inspect

    src = inspect.getsource(
        ctlmod.DataPreparationController.prepare_large_data)
    assert "worker.progress.connect(" in src
    i = src.index("worker.progress.connect(")
    assert "_t=token" in src[i:i + 260], "the token must be bound in the lambda"

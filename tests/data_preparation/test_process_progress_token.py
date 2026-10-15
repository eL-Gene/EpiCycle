"""Brick 5 fix 2 - the Process progress slots are token-guarded.

Both Process launches connected their worker's ``progress`` signal to a BARE
bound method, so the only thing standing between a draining worker and the live
bar was the `_processing` boolean.  That flag is not a stale-guard: cancelling a
run and starting another turns it straight back ON, and a streaming Process is
NOT interruptible mid-block, so the old worker's block counts went on painting
the new run's percentage.  The launch now binds the token in the lambda and the
slot re-checks it - the same shape the finish / error handlers already used.

Qt-free: `__new__` controllers + a `_Bar` recorder.
"""
from __future__ import annotations

import inspect

from napari_epicycle.data_preparation import controller as ctlmod


class _Bar:
    def __init__(self):
        self.calls = []

    def _busy_progress(self, label, done, total):
        self.calls.append(("progress", label, done, total))

    def _busy_show(self, label):
        self.calls.append(("show", label))


def _ctl(token=3):
    c = ctlmod.DataPreparationController.__new__(ctlmod.DataPreparationController)
    c.widget = _Bar()
    c._processing = True
    c._process_token = token
    return c


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #

def test_streaming_progress_paints_for_the_live_run():
    c = _ctl()
    c._on_process_store_progress(4, 10, 3)
    assert c.widget.calls == [("progress", "Processing (streaming) …", 4, 10)]


def test_a_draining_streaming_worker_cannot_drive_the_new_bar():
    c = _ctl(token=4)                     # a newer run owns the bar
    c._on_process_store_progress(4, 10, 3)
    assert c.widget.calls == []


def test_streaming_progress_still_declines_when_not_processing():
    c = _ctl()
    c._processing = False
    c._on_process_store_progress(4, 10, 3)
    assert c.widget.calls == []


def test_streaming_progress_without_a_token_still_paints():
    """`token=None` keeps the pre-fix contract for any bare caller."""
    c = _ctl()
    c._on_process_store_progress(4, 10)
    assert c.widget.calls


def test_streaming_zero_total_falls_back_to_the_marquee():
    c = _ctl()
    c._on_process_store_progress(0, 0, 3)
    assert c.widget.calls == [("show", "Processing (streaming) …")]


# --------------------------------------------------------------------------- #
# in-memory (the identical hole in the sibling launch)
# --------------------------------------------------------------------------- #

def test_in_memory_progress_paints_for_the_live_run():
    c = _ctl()
    c._on_process_progress(0.25, "Processing …", False, 3)
    assert c.widget.calls == [("progress", "Processing …", 250, 1000)]


def test_a_draining_in_memory_worker_cannot_drive_the_new_bar():
    c = _ctl(token=4)
    c._on_process_progress(0.25, "Processing …", False, 3)
    c._on_process_progress(0.25, "Processing …", True, 3)
    assert c.widget.calls == []


def test_in_memory_indeterminate_uses_the_marquee():
    c = _ctl()
    c._on_process_progress(0.0, "Blurring …", True, 3)
    assert c.widget.calls == [("show", "Blurring …")]


# --------------------------------------------------------------------------- #
# the wiring
# --------------------------------------------------------------------------- #

def test_both_launches_bind_the_token_in_the_progress_lambda():
    """A bare `worker.progress.connect(self._on_...)` is exactly the bug."""
    src = inspect.getsource(ctlmod.DataPreparationController)
    for name in ("_on_process_progress", "_on_process_store_progress"):
        bare = "worker.progress.connect(self.%s)" % name
        assert bare not in src, "%s is connected without a token" % name
    for hook in ("self._on_process_progress(frac, label, indet, _t)",
                 "self._on_process_store_progress(done, total, _t)"):
        assert hook in src, hook

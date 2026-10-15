"""Stale-finish guard for the async EDA/auto-cluster compute (hole 2).

`_launch_async_compute` runs cluster / PCA / UMAP / tSNE on a background
QThread via `_AsyncComputeWorker`, then applies the result on the main thread
in an `on_done` closure that writes back to `_qc_features_df` AND re-resolves
the patch's CSV path at apply-time.  It was gated only by `_curation_busy`
(which blocks a patch switch) but had NO stale-finish guard — so any path that
changes the active patch between launch and finish would land the OLD df in the
NEW patch's context (a stale result written into the wrong patch).

The fix captures a monotonic dispatch token + the launch-time patch name and
discards the result (skips `on_done`) in the finish handler if either changed.

These tests monkeypatch `_AsyncComputeWorker` + `QThread` with Qt-free fakes so
the wired `done` slot can be captured and fired manually (no event loop).
"""

from __future__ import annotations

import inspect

from napari_epicycle.fusion_reconstruction import controller as ctrl_mod
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


# ---------------------------------------------------------------------------
# Qt-free fakes
# ---------------------------------------------------------------------------


class _FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for s in list(self.slots):
            s(*args)


class _FakeWorker:
    def __init__(self, fn):
        self._fn = fn
        self.done = _FakeSignal()
        self.error = _FakeSignal()

    def moveToThread(self, _t):
        pass

    def run(self):  # referenced by thread.started.connect(worker.run)
        pass


class _FakeThread:
    def __init__(self):
        self.started = _FakeSignal()
        self.finished = _FakeSignal()
        self._started = False

    def start(self):
        self._started = True

    def quit(self):
        pass


def _make_compute_controller(monkeypatch, workers):
    def _factory(fn):
        w = _FakeWorker(fn)
        workers.append(w)
        return w

    monkeypatch.setattr(ctrl_mod, "_AsyncComputeWorker", _factory)
    monkeypatch.setattr(ctrl_mod, "QThread", _FakeThread)

    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._compute_dispatch_token = 0
    c._compute_dispatch_patch = None
    c._qc_view_patch_name = "patchA"
    c._busy_hidden = []
    c._busy_hide = lambda: c._busy_hidden.append(True)
    c._append_log = lambda m: None
    return c


def _done_slot(worker):
    # worker.done.connect order: _handle_done first, then thread.quit.
    return worker.done.slots[0]


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------


def test_async_compute_applies_when_not_stale(monkeypatch):
    workers = []
    c = _make_compute_controller(monkeypatch, workers)
    applied = []

    c._launch_async_compute("TEST", lambda: "RESULT", lambda r: applied.append(r))
    # Active patch unchanged → result must apply.
    _done_slot(workers[0])("RESULT")
    assert applied == ["RESULT"]
    assert c._busy_hidden, "busy_hide must run on the happy path"


def test_async_compute_discards_stale_finish_on_patch_change(monkeypatch):
    workers = []
    c = _make_compute_controller(monkeypatch, workers)
    applied = []

    c._launch_async_compute("TEST", lambda: "RESULT", lambda r: applied.append(r))
    # The active patch changes AFTER launch (a path that bypassed the busy
    # guard, or a future run-switch).
    c._qc_view_patch_name = "patchB"
    _done_slot(workers[0])("RESULT")
    assert applied == [], "stale finish (patch changed) must NOT apply on_done"
    assert c._busy_hidden, "busy_hide must still run on a stale finish"


def test_async_compute_discards_superseded_finish(monkeypatch):
    workers = []
    c = _make_compute_controller(monkeypatch, workers)
    applied1 = []
    applied2 = []

    # First launch (token 1).
    c._launch_async_compute("OP1", lambda: "R1", lambda r: applied1.append(r))
    # A second launch supersedes it (token 2) — same patch.
    c._launch_async_compute("OP2", lambda: "R2", lambda r: applied2.append(r))

    # The FIRST (now superseded) finish fires → discarded.
    _done_slot(workers[0])("R1")
    assert applied1 == [], "superseded finish (older token) must be discarded"

    # The SECOND (current) finish applies.
    _done_slot(workers[1])("R2")
    assert applied2 == ["R2"]


def test_async_compute_token_increments_per_launch(monkeypatch):
    workers = []
    c = _make_compute_controller(monkeypatch, workers)
    c._launch_async_compute("A", lambda: None, lambda r: None)
    t1 = c._compute_dispatch_token
    c._launch_async_compute("B", lambda: None, lambda r: None)
    t2 = c._compute_dispatch_token
    assert t2 == t1 + 1
    assert c._compute_dispatch_patch == "patchA"


# ---------------------------------------------------------------------------
# Source-introspection tests
# ---------------------------------------------------------------------------


def test_compute_dispatch_guards_initialized_in_init_source():
    src = inspect.getsource(FusionReconstructionController.__init__)
    assert "self._compute_dispatch_token: int = 0" in src
    assert "self._compute_dispatch_patch" in src


def test_launch_async_compute_captures_and_checks_identity_source():
    src = inspect.getsource(
        FusionReconstructionController._launch_async_compute
    )
    assert "_compute_dispatch_token" in src, (
        "_launch_async_compute must bump + capture the dispatch token"
    )
    assert "_qc_view_patch_name" in src, (
        "_launch_async_compute must capture the launch-time patch name"
    )
    # on_done must be gated behind the staleness check.  Find the staleness
    # check, then the on_done CALL at/after it (the first textual
    # "on_done(result)" is the docstring mention, not the real call).
    assert "if _is_stale():" in src
    stale_idx = src.index("if _is_stale():")
    done_idx = src.index("on_done(result)", stale_idx)
    assert stale_idx < done_idx, (
        "the staleness check must run BEFORE on_done writes the result back"
    )

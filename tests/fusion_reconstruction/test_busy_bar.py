"""Tests for the reusable Step 3 busy/progress indicator
(integration/step3-patch-perf, 2026-06-05; redesigned 2026-06-06).

The patch-load progress bar was generalized into a single PERMANENT
``curation_busy_bar`` (ui.py) + the single permanent ``curation_status_label``
directly below it, driven by three controller helpers.

2026-06-06 redesign (PERMANENT bar): the bar is NEVER hidden.  While IDLE it
is a static determinate-empty strip (``setRange(0, 100)`` + ``setValue(0)``,
no marquee).  While BUSY it is flipped to INDETERMINATE "scan mode"
(``setRange(0, 0)`` → marquee).  The descriptive per-op text is written to
``curation_status_label`` (the same permanent label that shows the idle patch
status); ``_busy_hide`` restores that label to the remembered idle text
(``_curation_idle_status``, set via ``_set_curation_status``) so it is never
left blank or showing stale op text.  ``curation_busy_label`` was removed.

  * ``_busy_show(label, *, indeterminate=True, lock=True)`` — flip the bar to
    scan mode (marquee) + write the op text to ``curation_status_label``, and
    (optionally) lock the curation UI.  ``indeterminate`` accepted-but-ignored.
  * ``_busy_progress(label)`` — update ONLY ``curation_status_label``.
  * ``_busy_hide()`` — return the bar to idle static + restore the idle text
    on ``curation_status_label``, unlock.  The bar stays visible throughout.
  * ``_set_curation_status(text)`` — set ``curation_status_label`` + remember
    the text as the idle status for ``_busy_hide`` to restore.

Every Step 3 long-running op now routes through these helpers so the
curation UI is locked one-op-at-a-time and a patch switch is blocked while
ANY busy op runs (``_curation_busy`` flag).  The EDA recompute ops (PCA /
UMAP / tSNE / clustering) are worker-ified (``_AsyncComputeWorker``) so the
marquee animates; Commit/Save remain synchronous (sync-paint).

Source-introspection + light-fake tests (no Qt display / napari viewer) so
they run in CI alongside the rest of ``tests/fusion_reconstruction/``.
"""

from __future__ import annotations

import inspect

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


# ---------------------------------------------------------------------------
# Helpers exist with the right signatures.
# ---------------------------------------------------------------------------

def test_busy_helpers_exist():
    for name in ("_busy_show", "_busy_progress", "_busy_hide",
                 "_set_curation_status"):
        assert hasattr(FusionReconstructionController, name), (
            f"controller must define {name}"
        )


def test_busy_show_supports_indeterminate_kwarg():
    sig = inspect.signature(FusionReconstructionController._busy_show)
    assert "indeterminate" in sig.parameters, (
        "_busy_show must accept an `indeterminate` keyword (scan vs determinate)"
    )
    assert "lock" in sig.parameters, (
        "_busy_show must accept a `lock` keyword"
    )
    # Default is indeterminate scan mode (the common case for worker/blocking
    # ops); patch load passes indeterminate=False explicitly.
    assert sig.parameters["indeterminate"].default is True
    assert sig.parameters["lock"].default is True


def test_busy_show_always_scan_mode_source():
    """2026-06-06: _busy_show is ALWAYS scan mode — setRange(0, 0) marquee.
    It must write the op text to curation_status_label + lock when lock=True.
    It must NOT toggle the bar's visibility (the bar is permanent now)."""
    src = inspect.getsource(FusionReconstructionController._busy_show)
    assert "setRange(0, 0)" in src, "scan mode must use setRange(0, 0)"
    assert "curation_status_label" in src, (
        "_busy_show must write the op text to curation_status_label"
    )
    assert "curation_busy_label" not in src, (
        "curation_busy_label was removed — _busy_show must not reference it"
    )
    assert "setVisible" not in src, (
        "_busy_show must NOT toggle visibility — the bar is permanent"
    )
    assert "_set_curation_locked(True)" in src, "_busy_show must lock when lock=True"
    assert "[BUSY]" in src


def test_busy_hide_unlocks_source():
    """2026-06-06: _busy_hide must NOT hide the bar (it is permanent); it
    returns the bar to idle static + restores the remembered idle status
    text on curation_status_label, and unlocks."""
    src = inspect.getsource(FusionReconstructionController._busy_hide)
    assert "_set_curation_locked(False)" in src, "_busy_hide must unlock the UI"
    assert "setVisible(False)" not in src, (
        "_busy_hide must NOT call setVisible(False) — the bar is permanent"
    )
    assert "curation_status_label" in src, (
        "_busy_hide must restore the idle text on curation_status_label"
    )
    assert "_curation_idle_status" in src, (
        "_busy_hide must restore the remembered idle status text"
    )
    assert "curation_busy_label" not in src, (
        "curation_busy_label was removed — _busy_hide must not reference it"
    )
    assert "[BUSY]" in src


def test_busy_progress_updates_label_source():
    """2026-06-06: _busy_progress updates ONLY curation_status_label
    (the bar stays in scan-mode marquee — no determinate promotion anymore)."""
    src = inspect.getsource(FusionReconstructionController._busy_progress)
    assert "curation_status_label" in src, (
        "_busy_progress must update curation_status_label"
    )
    assert "curation_busy_label" not in src, (
        "curation_busy_label was removed — _busy_progress must not reference it"
    )
    assert "setText" in src
    # The bar is no longer touched here — only the label.
    assert "curation_busy_bar" not in src, (
        "_busy_progress must NOT touch the bar (scan mode is permanent)"
    )


def test_set_curation_status_source():
    """_set_curation_status writes curation_status_label + remembers the idle
    text in _curation_idle_status so _busy_hide can restore it."""
    src = inspect.getsource(
        FusionReconstructionController._set_curation_status
    )
    assert "curation_status_label" in src
    assert "_curation_idle_status" in src
    assert "setText" in src


# ---------------------------------------------------------------------------
# Light-fake runtime behavior: show (both modes) / progress / hide.
# ---------------------------------------------------------------------------

class _FakeBar:
    def __init__(self):
        self._min = 0
        self._max = 100
        self._value = 0
        self._format = ""
        # The bar is PERMANENT (2026-06-06) — constructed visible in ui.py.
        # The busy helpers no longer toggle visibility, so the fake must
        # mirror the real construction-time state.
        self._visible = True

    def setRange(self, lo, hi):
        self._min, self._max = lo, hi

    def maximum(self):
        return self._max

    def setValue(self, v):
        self._value = v

    def setFormat(self, f):
        self._format = f

    def setVisible(self, v):
        self._visible = v


class _FakeLabel:
    def __init__(self):
        self._text = ""
        self._visible = False

    def setText(self, t):
        self._text = t

    def setVisible(self, v):
        self._visible = v


class _FakeUI:
    pass


def _make_controller_with_bar():
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    ui = _FakeUI()
    ui.curation_busy_bar = _FakeBar()
    # The single permanent status label below the bar (was curation_busy_label).
    lbl = _FakeLabel()
    lbl._visible = True  # permanent — visible by default
    ui.curation_status_label = lbl
    c.ui = ui
    # Stub the lock so _busy_show/_busy_hide don't need real widgets.
    c._lock_calls = []
    c._set_curation_locked = lambda locked: c._lock_calls.append(locked)
    return c


def test_busy_show_sets_scan_mode_and_label():
    c = _make_controller_with_bar()
    bar = c.ui.curation_busy_bar
    lbl = c.ui.curation_status_label
    c._busy_show("Extracting QC features…")  # default indeterminate=True
    assert c._curation_busy is True
    assert bar._max == 0, "scan mode must set range max to 0 (marquee)"
    # The bar is permanent — _busy_show must NOT have hidden it.
    assert bar._visible is True
    # Op text now lives in the permanent curation_status_label.
    assert lbl._text == "Extracting QC features…"
    assert c._lock_calls == [True]


def test_busy_show_does_not_toggle_bar_visibility():
    """The bar starts visible (permanent) and _busy_show keeps it visible."""
    c = _make_controller_with_bar()
    bar = c.ui.curation_busy_bar
    bar._visible = True
    c._busy_show("op")
    assert bar._visible is True, "_busy_show must not hide the permanent bar"


def test_busy_show_indeterminate_kwarg_ignored_still_scan_mode():
    """The indeterminate kwarg is accepted-but-ignored — the bar is always
    scan mode now (no determinate %-bar)."""
    c = _make_controller_with_bar()
    bar = c.ui.curation_busy_bar
    c._busy_show("Loading", indeterminate=False)
    assert bar._max == 0, "bar must STILL be scan mode even with indeterminate=False"


def test_busy_show_no_lock():
    c = _make_controller_with_bar()
    c._busy_show("PCA…", lock=False)
    assert c._curation_busy is True
    assert c._lock_calls == [], "lock=False must not call _set_curation_locked"


def test_busy_progress_updates_only_label():
    c = _make_controller_with_bar()
    bar = c.ui.curation_busy_bar
    lbl = c.ui.curation_status_label
    c._busy_show("Loading patch001…")
    assert bar._max == 0
    # A progress push updates the status label text only; the bar stays scan mode.
    c._busy_progress("Loading patch001 — downsampling…")
    assert lbl._text == "Loading patch001 — downsampling…"
    assert bar._max == 0, "bar must STAY in scan mode (marquee)"


def test_busy_hide_returns_to_idle_and_restores_text():
    c = _make_controller_with_bar()
    bar = c.ui.curation_busy_bar
    lbl = c.ui.curation_status_label
    # An idle patch status is established first (e.g. patch load completed).
    c._set_curation_status("Curating: patch001")
    c._busy_show("Saving edits…")
    assert lbl._text == "Saving edits…"
    c._busy_hide()
    assert c._curation_busy is False
    # Bar is NOT hidden — it returns to idle static (empty determinate).
    assert bar._visible is True, "the bar must stay visible (permanent)"
    assert bar._min == 0 and bar._max == 100  # idle static range
    assert bar._value == 0
    # Label restored to the remembered idle status, NOT blank / stale op text.
    assert lbl._text == "Curating: patch001"
    assert c._lock_calls[-1] is False


def test_busy_hide_restores_default_when_no_idle_set():
    """If no idle status was ever set, _busy_hide falls back to the default
    'No active curation session.' rather than leaving the label blank."""
    c = _make_controller_with_bar()
    lbl = c.ui.curation_status_label
    c._busy_show("Computing PCA…")
    c._busy_hide()
    assert lbl._text == "No active curation session."


def test_set_curation_status_remembers_idle_text():
    c = _make_controller_with_bar()
    lbl = c.ui.curation_status_label
    c._set_curation_status("Viewing: patch042")
    assert lbl._text == "Viewing: patch042"
    assert c._curation_idle_status == "Viewing: patch042"


def test_busy_helpers_guard_missing_bar():
    """Helpers must never raise when the bar/label attributes are absent."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c.ui = _FakeUI()  # no curation_busy_bar / curation_status_label
    c._set_curation_locked = lambda locked: None
    c._busy_show("x")          # must not raise
    c._busy_progress("x")      # must not raise
    c._busy_hide()             # must not raise
    c._set_curation_status("x")  # must not raise
    assert c._curation_busy is False


# ---------------------------------------------------------------------------
# Switch is blocked while ANY busy op runs (not just patch load).
# ---------------------------------------------------------------------------

def test_switch_guards_on_curation_busy_source():
    src = inspect.getsource(
        FusionReconstructionController._switch_to_curation_patch
    )
    assert "_curation_busy" in src, (
        "_switch_to_curation_patch must consult _curation_busy so Extract / "
        "PCA / Save / etc. also block a patch switch while running"
    )


def test_switch_guard_returns_early_when_curation_busy():
    """Live: with _curation_busy True (but no patch load in flight), the
    switch returns early before touching any surface."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._patch_load_in_flight = False
    c._curation_busy = True
    c._qc_view_patch_name = "other_patch"
    appended = []
    c._append_log = lambda m: appended.append(m)

    # No _hard_clear_curation_surfaces / _guard_unsaved_changes defined on the
    # bare instance — if the guard fails to early-return, this raises
    # AttributeError, surfacing as a test failure.
    c._switch_to_curation_patch("some_patch")

    assert any("[LOAD-LOCK]" in m for m in appended), (
        "the busy switch should log a [LOAD-LOCK] ignore marker"
    )


# ---------------------------------------------------------------------------
# Op wiring: extract launches busy, finish/error hide busy.
# ---------------------------------------------------------------------------

def test_extract_features_current_shows_busy():
    src = inspect.getsource(
        FusionReconstructionController._extract_features_current
    )
    assert "_busy_show(" in src, (
        "_extract_features_current must show the busy bar (scan mode) at launch"
    )


def test_on_qc_extract_finished_hides_busy():
    src = inspect.getsource(
        FusionReconstructionController._on_qc_extract_finished
    )
    assert "_busy_hide()" in src, (
        "_on_qc_extract_finished must hide the busy bar (every terminal path)"
    )


def test_on_qc_extract_error_hides_busy():
    src = inspect.getsource(
        FusionReconstructionController._on_qc_extract_error
    )
    assert "_busy_hide()" in src, (
        "_on_qc_extract_error must hide the busy bar"
    )


# ---------------------------------------------------------------------------
# Synchronous ops (Commit / Save) use the sync-paint pattern (busy_show + a
# targeted bar.repaint() + try/finally busy_hide).  The marquee won't animate
# for these (main thread blocked) but the bar + label show; animating them
# needs worker-ification (future follow-up — disk writes are out of scope here).
#
# 2026-06-05 (audit fix FIX 5): the sync-paint mechanism changed from
# ``QApplication.processEvents()`` to a targeted ``curation_busy_bar.repaint()``.
# repaint() forces an immediate synchronous paint WITHOUT pumping the Qt event
# loop, eliminating the (theoretical) re-entrancy where processEvents() could
# flush queued timers/signals mid-op.  These tests assert the new mechanism.
# ---------------------------------------------------------------------------

def _assert_sync_paint(method_name):
    method = getattr(FusionReconstructionController, method_name)
    src = inspect.getsource(method)
    assert "_busy_show(" in src, f"{method_name} must call _busy_show"
    assert ".repaint()" in src, (
        f"{method_name} must call a targeted curation_busy_bar.repaint() to "
        f"paint the busy bar before the synchronous compute blocks the event "
        f"loop (repaint does NOT pump the event loop, avoiding mid-op "
        f"re-entrancy that processEvents() could cause)"
    )
    # Negative check targets the CALL form (``QApplication.processEvents(``)
    # so the explanatory comments — which still mention ``processEvents()`` in
    # prose — don't trip the assertion.
    assert "QApplication.processEvents(" not in src, (
        f"{method_name} must NOT call QApplication.processEvents() — it pumps "
        f"the event loop and can re-enter mid-op; use a targeted bar.repaint() "
        f"instead"
    )
    assert "_busy_hide()" in src, f"{method_name} must call _busy_hide"
    assert "finally" in src, (
        f"{method_name} must hide the busy bar in a finally so it always clears"
    )


# ---------------------------------------------------------------------------
# 2026-06-07: Commit Triage was ASYNC-ified (background QThread persist) so the
# busy-bar marquee actually animates + the UI stays pannable during a big-patch
# commit.  It NO LONGER follows the synchronous sync-paint pattern: _busy_show +
# the worker launch happen in _commit_triage, but _busy_hide moved into the
# main-thread finish/error handlers (a busy_hide in _commit_triage itself would
# fire BEFORE the off-thread write completes).  The launch-failure guard keeps
# its own _busy_hide so a construction failure can't strand the lock.
# ---------------------------------------------------------------------------

def test_commit_triage_async_launch():
    """_commit_triage shows the busy bar + launches a worker thread; the bar
    is hidden by the finish/error handlers, NOT a finally in this method."""
    src = inspect.getsource(FusionReconstructionController._commit_triage)
    assert "_busy_show(" in src, "_commit_triage must call _busy_show"
    assert ".repaint()" in src, (
        "_commit_triage should repaint the bar so the marquee shows immediately"
    )
    assert "thread.start()" in src, (
        "_commit_triage must launch a background QThread (async persist)"
    )
    assert "_CommitTriageWorker" in src, (
        "_commit_triage must construct the _CommitTriageWorker"
    )
    assert "_commit_triage_in_flight = True" in src, (
        "_commit_triage must set the in-flight gate before launching"
    )
    # The launch-failure guard keeps a _busy_hide so a construction failure
    # never strands the lock.
    assert "_busy_hide()" in src, (
        "_commit_triage must hide the bar in its launch-failure guard"
    )
    assert "QApplication.processEvents(" not in src, (
        "_commit_triage must not pump the event loop"
    )


def test_commit_triage_handlers_hide_busy():
    """Both terminal handlers (finished/error) must hide the busy bar so the
    lock always releases on the main thread."""
    for name in ("_on_commit_triage_finished", "_on_commit_triage_error"):
        src = inspect.getsource(getattr(FusionReconstructionController, name))
        assert "_busy_hide()" in src, f"{name} must call _busy_hide"
        assert "finally" in src, (
            f"{name} must hide the busy bar in a finally so it always clears"
        )


def test_save_curation_checked_sync_paint():
    _assert_sync_paint("_save_curation_checked")


# ---------------------------------------------------------------------------
# 2026-06-07: the Save BUTTON path (_save_curation_async) was ASYNC-ified
# (background QThread persist) so the busy-bar marquee actually animates + the
# UI stays pannable during a big-patch mask save.  _busy_show + the worker
# launch happen in _save_curation_async, but _busy_hide moved into the
# main-thread finish/error handlers (a busy_hide in the launcher itself would
# fire BEFORE the off-thread write completes).  The launch-failure guard keeps
# its own _busy_hide so a construction failure can't strand the lock.  Note the
# GATE handler _save_curation_checked stays SYNC (sync-paint) — see above.
# ---------------------------------------------------------------------------

def test_save_curation_async_launch():
    """_save_curation_async shows the busy bar + launches a worker thread; the
    bar is hidden by the finish/error handlers, NOT a finally in this method."""
    src = inspect.getsource(FusionReconstructionController._save_curation_async)
    assert "_busy_show(" in src, "_save_curation_async must call _busy_show"
    assert ".repaint()" in src, (
        "_save_curation_async should repaint the bar so the marquee shows "
        "immediately"
    )
    assert "thread.start()" in src, (
        "_save_curation_async must launch a background QThread (async persist)"
    )
    assert "_SaveCurationWorker" in src, (
        "_save_curation_async must construct the _SaveCurationWorker"
    )
    assert "_save_in_flight = True" in src, (
        "_save_curation_async must set the in-flight gate before launching"
    )
    # The launch-failure guard keeps a _busy_hide so a construction failure
    # never strands the lock.
    assert "_busy_hide()" in src, (
        "_save_curation_async must hide the bar in its launch-failure guard"
    )
    assert "QApplication.processEvents(" not in src, (
        "_save_curation_async must not pump the event loop"
    )


def test_save_curation_handlers_hide_busy():
    """Both terminal handlers (finished/error) must hide the busy bar so the
    lock always releases on the main thread (async path only)."""
    for name in ("_on_save_curation_finished", "_on_save_curation_error"):
        src = inspect.getsource(getattr(FusionReconstructionController, name))
        assert "_busy_hide()" in src, f"{name} must call _busy_hide"
        assert "finally" in src, (
            f"{name} must hide the busy bar in a finally so it always clears"
        )


# ---------------------------------------------------------------------------
# Worker-ified ops (PCA / UMAP / tSNE / clustering) — 2026-06-06.
#
# These now launch an _AsyncComputeWorker (via _launch_async_compute) so the
# busy marquee animates while the compute runs off the main thread.  Each must:
#   * call _busy_show(...) at launch,
#   * reference _launch_async_compute (the worker launcher),
#   * NOT use the .repaint() sync-paint pattern (no longer blocks the loop).
# The _busy_hide() is owned by _launch_async_compute's done/error handlers
# (try/finally), not the op method itself.
# ---------------------------------------------------------------------------

def _assert_worker_ified(method_name):
    method = getattr(FusionReconstructionController, method_name)
    src = inspect.getsource(method)
    assert "_busy_show(" in src, f"{method_name} must call _busy_show at launch"
    assert "_launch_async_compute(" in src, (
        f"{method_name} must launch the compute via _launch_async_compute so "
        f"the busy marquee animates off the main thread"
    )
    assert ".repaint()" not in src, (
        f"{method_name} must NOT use the sync-paint .repaint() pattern anymore "
        f"— it is worker-ified; the marquee animates on its own"
    )


def test_run_clustering_worker_ified():
    _assert_worker_ified("_run_clustering")


def test_recompute_pca_worker_ified():
    _assert_worker_ified("_recompute_pca")


def test_recompute_umap_worker_ified():
    _assert_worker_ified("_recompute_umap")


def test_recompute_tsne_worker_ified():
    _assert_worker_ified("_recompute_tsne")


# ---------------------------------------------------------------------------
# _AsyncComputeWorker exists + _launch_async_compute wires done/error → busy_hide.
# ---------------------------------------------------------------------------

def test_async_compute_worker_exists():
    from napari_epicycle.fusion_reconstruction.controller import (
        _AsyncComputeWorker,
    )
    assert hasattr(_AsyncComputeWorker, "done")
    assert hasattr(_AsyncComputeWorker, "error")
    assert hasattr(_AsyncComputeWorker, "run")


def test_async_compute_worker_run_emits_done():
    """run() emits done(result) on success; error(msg) on exception — without
    touching any shared controller state (pure compute_fn contract).

    Uses a tiny fake-signal shim + ``__new__`` so we can drive ``run()``
    without a Qt event loop (matching the rest of this CI-safe module)."""
    from napari_epicycle.fusion_reconstruction.controller import (
        _AsyncComputeWorker,
    )

    class _Sig:
        def __init__(self):
            self._cbs = []

        def connect(self, cb):
            self._cbs.append(cb)

        def emit(self, *a):
            for cb in self._cbs:
                cb(*a)

    # done path
    w = _AsyncComputeWorker.__new__(_AsyncComputeWorker)
    w._fn = lambda: 7
    w.done = _Sig()
    w.error = _Sig()
    results, errors = [], []
    w.done.connect(results.append)
    w.error.connect(errors.append)
    w.run()
    assert results == [7]
    assert errors == []

    # error path
    def _boom():
        raise ValueError("kaboom")

    w2 = _AsyncComputeWorker.__new__(_AsyncComputeWorker)
    w2._fn = _boom
    w2.done = _Sig()
    w2.error = _Sig()
    res2, err2 = [], []
    w2.done.connect(res2.append)
    w2.error.connect(err2.append)
    w2.run()
    assert res2 == []
    assert len(err2) == 1 and "kaboom" in err2[0]


def test_launch_async_compute_done_handler_hides_busy_source():
    """_launch_async_compute's done handler must hide the busy bar in a
    finally so a raise in the apply-back logic can't strand the lock on."""
    src = inspect.getsource(
        FusionReconstructionController._launch_async_compute
    )
    assert "_AsyncComputeWorker(" in src
    assert "_busy_hide()" in src
    assert "finally" in src, (
        "_launch_async_compute must hide the busy bar in a finally on the done "
        "path (Qt swallows slot exceptions)"
    )
    # Teardown is signal-based (thread.quit), NOT a blocking main-thread wait().
    assert "thread.quit" in src, (
        "_launch_async_compute must tear the thread down via thread.quit"
    )
    assert "thread.wait(" not in src, (
        "_launch_async_compute must NOT call thread.wait() on the main thread"
    )


# ---------------------------------------------------------------------------
# Guaranteed-unlock audit fix (2026-06-05, integration/step3-patch-perf).
#
# Several Qt-slot / launch paths call _busy_show(...) (which sets
# _curation_busy=True + disables the curation UI) but could raise BEFORE
# reaching _busy_hide().  Because Qt swallows slot exceptions, that would
# leave _curation_busy=True and the UI permanently locked (patch switching
# blocked until restart).  These ops were wrapped so _busy_hide() ALWAYS runs
# (try/finally) or so a launch-time raise hides the bar (try/except).
# ---------------------------------------------------------------------------

def _assert_busy_hide_protected(method_name):
    """The method must hide the busy bar on every exit path — i.e. its source
    contains a _busy_hide() AND a finally/except guard so a mid-handler raise
    can't strand the curation lock on (Qt swallows slot exceptions)."""
    method = getattr(FusionReconstructionController, method_name)
    src = inspect.getsource(method)
    assert "_busy_hide" in src, (
        f"{method_name} must call _busy_hide() to clear the busy bar + lock"
    )
    assert ("finally" in src) or ("except" in src), (
        f"{method_name} must guard _busy_hide() with try/finally (or try/except "
        f"on the launch path) so a raise can't leave _curation_busy=True and "
        f"the curation UI permanently locked"
    )


def test_on_patch_loaded_busy_hide_protected():
    # FIX 1: entire body after the stale-token early-return wrapped in
    # try/finally so _busy_hide() always runs even if a GPU add raises.
    _assert_busy_hide_protected("_on_patch_loaded")


def test_on_qc_extract_finished_busy_hide_protected():
    # FIX 3: normal-completion body wrapped in try/finally: _busy_hide().
    _assert_busy_hide_protected("_on_qc_extract_finished")


def test_load_patch_for_qc_view_busy_hide_protected():
    # FIX 2: worker/thread construction + start() wrapped in try/except that
    # hides the busy bar + clears the in-flight guard on a launch-time raise.
    _assert_busy_hide_protected("_load_patch_for_qc_view")


def test_extract_features_current_busy_hide_protected():
    # FIX 4: worker/thread launch zone wrapped in try/except that hides the
    # busy bar + re-enables the Extract button on a launch-time raise.
    _assert_busy_hide_protected("_extract_features_current")

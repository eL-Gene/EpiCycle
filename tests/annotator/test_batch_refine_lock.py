"""Tests for the Batch-Refine → controller lock wiring (2026-06-07).

Batch Refine is now a first-class LOCKED Step-3 op: the annotator drives a
controller flag (_batch_refine_running) via an injected callback so the WHOLE
Step-3 surface (predict / reconstruct / extract / commit / save / step-nav /
patch-switch + the navigator action buttons) locks for the duration of BOTH the
worker compute AND the staged-preview window.

Two layers:
  * source-introspection — every terminal path (START + accept / reject / error
    / cancel / dialog close / discard) calls _fire_batch_refine_lock with the
    right arg; the STAGED path must NOT release.
  * live light-fake — _fire_batch_refine_lock routes through the injected
    callback (guarded when absent), set_curation_callbacks stores it, and the
    _open_batch_refine_dialog load-lock gate refuses while load-locked.

Qt-free: source introspection + Curation3DAnnotator.__new__ stubs (no display).
"""

from __future__ import annotations

import inspect

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator


# --------------------------------------------------------------------------- #
# set_curation_callbacks stores the lock callback
# --------------------------------------------------------------------------- #
def test_set_curation_callbacks_accepts_lock_cb():
    sig = inspect.signature(Curation3DAnnotator.set_curation_callbacks)
    assert "on_batch_refine_lock" in sig.parameters, (
        "set_curation_callbacks must accept on_batch_refine_lock"
    )


def test_set_curation_callbacks_stores_lock_cb_live():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    sentinel = lambda active: None  # noqa: E731
    ann.set_curation_callbacks(on_batch_refine_lock=sentinel)
    assert ann._batch_refine_lock_cb is sentinel


# --------------------------------------------------------------------------- #
# _fire_batch_refine_lock — routes through the cb, guards absence
# --------------------------------------------------------------------------- #
def test_fire_lock_invokes_callback_live():
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    seen = []
    ann._batch_refine_lock_cb = lambda active: seen.append(active)
    ann._fire_batch_refine_lock(True)
    ann._fire_batch_refine_lock(False)
    assert seen == [True, False]


def test_fire_lock_noop_when_cb_none():
    """Standalone annotator (no controller) → callback is None → no-op, no
    raise.  This is the steady state after set_curation_callbacks with no
    controller wiring (or after close cleanup), where _batch_refine_lock_cb
    is None."""
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    ann._batch_refine_lock_cb = None
    ann._fire_batch_refine_lock(True)  # must not raise


def test_fire_lock_swallows_callback_errors():
    """A buggy controller callback must not propagate into the batch flow."""
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)

    def _boom(active):
        raise RuntimeError("kaboom")

    ann._batch_refine_lock_cb = _boom
    ann._fire_batch_refine_lock(True)  # must not raise


# --------------------------------------------------------------------------- #
# START — _run_batch_refine fires the lock True before launching the worker
# --------------------------------------------------------------------------- #
def test_run_batch_refine_fires_lock_true_at_start_source():
    # The snapshot / lock / launch live in `_launch_batch_worker`, shared by
    # the geometry and the chain paths; both
    # paths reach it only after their early returns.
    run_src = inspect.getsource(Curation3DAnnotator._run_batch_refine)
    assert "self._launch_batch_worker(" in run_src
    chain_src = inspect.getsource(Curation3DAnnotator._run_batch_chain)
    assert "self._launch_batch_worker(" in chain_src
    src = inspect.getsource(Curation3DAnnotator._launch_batch_worker)
    assert "self._fire_batch_refine_lock(True)" in src, (
        "_run_batch_refine must fire the lock True at START"
    )
    # The lock-on must come AFTER the early-return validation (so we don't lock
    # on a bail) — i.e. after the running flag is set, before thread.start().
    lock_idx = src.index("self._fire_batch_refine_lock(True)")
    run_flag_idx = src.index("self._batch_running = True")
    start_idx = src.index("thread.start()")
    assert run_flag_idx < lock_idx < start_idx, (
        "lock-on must be set after _batch_running=True and before thread.start()"
    )


# --------------------------------------------------------------------------- #
# RESOLVE — every terminal path fires the lock False
# --------------------------------------------------------------------------- #
def test_worker_finished_discard_paths_fire_lock_false_source():
    """The three discard branches of _on_batch_worker_finished (dialog gone /
    patch changed / cancelled-no-op) are terminal → release the lock."""
    src = inspect.getsource(Curation3DAnnotator._on_batch_worker_finished)
    # Three discard branches each return; count the False-fires before the
    # staged section (which must NOT fire).  Find the staged marker first.
    staged_idx = src.index('dlg.set_state("staged")')
    pre_staged = src[:staged_idx]
    assert pre_staged.count("self._fire_batch_refine_lock(False)") == 3, (
        "all THREE discard branches must release the lock (dialog-gone / "
        "patch-changed / cancelled-or-no-op)"
    )


def test_worker_finished_staged_path_does_not_release_source():
    """The STAGED (success) path must NOT release the lock — the preview is
    pending accept/reject and the volume is not final."""
    src = inspect.getsource(Curation3DAnnotator._on_batch_worker_finished)
    staged_idx = src.index('dlg.set_state("staged")')
    post_staged = src[staged_idx:]
    assert "self._fire_batch_refine_lock(False)" not in post_staged, (
        "the staged path must keep the lock held until accept/reject"
    )


def test_worker_error_fires_lock_false_source():
    src = inspect.getsource(Curation3DAnnotator._on_batch_worker_error)
    assert "self._fire_batch_refine_lock(False)" in src, (
        "_on_batch_worker_error is terminal → must release the lock"
    )


def test_batch_accept_fires_lock_false_source():
    src = inspect.getsource(Curation3DAnnotator._on_batch_accept)
    assert "self._fire_batch_refine_lock(False)" in src, (
        "_on_batch_accept is terminal → must release the lock"
    )


def test_batch_reject_fires_lock_false_source():
    src = inspect.getsource(Curation3DAnnotator._on_batch_reject)
    assert "self._fire_batch_refine_lock(False)" in src, (
        "_on_batch_reject is terminal → must release the lock"
    )


def test_batch_close_fires_lock_false_source():
    """_on_batch_close — both the staged-restore path and the no-live-worker
    path must release.  (The worker-cancel path defers to the finished handler,
    which releases there; firing here too is harmless — idempotent.)"""
    src = inspect.getsource(Curation3DAnnotator._on_batch_close)
    assert src.count("self._fire_batch_refine_lock(False)") >= 2, (
        "_on_batch_close staged-restore + no-worker paths must release the lock"
    )


def test_discard_batch_refine_fires_lock_false_source():
    """_discard_batch_refine (annotator-close discard) — the running path is a
    synchronous discard (the finished handler won't fire), so it must release
    directly; the staged path delegates to _on_batch_reject (which releases)."""
    src = inspect.getsource(Curation3DAnnotator._discard_batch_refine)
    assert "self._fire_batch_refine_lock(False)" in src, (
        "_discard_batch_refine running path must release the lock directly"
    )


# --------------------------------------------------------------------------- #
# Open-dialog gate — refuse while load-locked (controller op in flight)
# --------------------------------------------------------------------------- #
def test_open_batch_refine_dialog_has_load_lock_gate_source():
    src = inspect.getsource(Curation3DAnnotator._open_batch_refine_dialog)
    assert 'getattr(self, "_load_locked", False)' in src, (
        "_open_batch_refine_dialog must refuse to start while load-locked "
        "(a controller op is in flight)"
    )
    # The gate must come before the can_start_batch_refine check + dialog build.
    gate_idx = src.index('getattr(self, "_load_locked", False)')
    clean_idx = src.index("can_start_batch_refine")
    assert gate_idx < clean_idx, (
        "the load-lock gate must precede the clean-state check"
    )


def test_set_load_locked_tracks_load_locked_flag_source():
    src = inspect.getsource(Curation3DAnnotator.set_load_locked)
    assert "self._load_locked = bool(locked)" in src, (
        "set_load_locked must record _load_locked so the open-dialog gate can "
        "consult it"
    )


def test_open_batch_refine_dialog_refuses_when_load_locked_live():
    """Live: with _load_locked True, _open_batch_refine_dialog returns before
    touching _active_labels_name / sync_manager / building the dialog."""
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    ann._load_locked = True
    # No _active_labels_name / sync_manager / _refine_state defined — if the
    # gate fails to early-return the body raises AttributeError, surfacing as a
    # test failure.
    ann._open_batch_refine_dialog()  # must early-return, no raise

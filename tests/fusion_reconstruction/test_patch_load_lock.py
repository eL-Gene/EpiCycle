"""Tests for the Step 3 async patch-load *load lock + progress bar*
(integration/step3-patch-perf, 2026-06-05).

While a patch is loading off-thread (``_load_patch_for_qc_view`` →
``_PatchLoadWorker`` → ``_on_patch_loaded``) the controller:
  (a) shows a determinate progress bar with stage text, and
  (b) LOCKS every curation control that touches the patch's data (and the
      open 3D annotator's drawing + navigator action buttons), allowing only
      passive pan/zoom/rotate, re-enabling on finish/error.

These are source-introspection + light-fake tests (no Qt display, no napari
viewer) so they run in CI alongside the rest of ``tests/fusion_reconstruction/``.
The actual signal/thread wiring is exercised by the running app
(``[ASYNC-LOAD]`` / ``[LOAD-LOCK]`` console markers).
"""

from __future__ import annotations

import inspect

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
    _PatchLoadWorker,
)


# ---------------------------------------------------------------------------
# Functional guard: a second switch while a load is in flight is ignored.
# ---------------------------------------------------------------------------

def test_switch_guards_on_patch_load_in_flight():
    """``_switch_to_curation_patch`` must early-return when a load is in
    flight (no queuing — the click is dropped)."""
    src = inspect.getsource(
        FusionReconstructionController._switch_to_curation_patch
    )
    assert "_patch_load_in_flight" in src, (
        "_switch_to_curation_patch must consult _patch_load_in_flight"
    )
    assert "[LOAD-LOCK]" in src, (
        "_switch_to_curation_patch must log a [LOAD-LOCK] marker when it "
        "ignores a switch during a load"
    )


def test_switch_guard_returns_early_when_in_flight():
    """Live behavior: with the in-flight flag set, the switch returns before
    touching any surface (uses a bare instance + the flag only)."""
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._patch_load_in_flight = True
    c._qc_view_patch_name = "other_patch"

    appended = []
    c._append_log = lambda m: appended.append(m)

    # No _hard_clear_curation_surfaces / _guard_unsaved_changes defined on the
    # bare instance — if the guard fails to early-return, this raises
    # AttributeError, which the test would surface as a failure.
    c._switch_to_curation_patch("some_patch")

    assert any("[LOAD-LOCK]" in m for m in appended), (
        "the in-flight switch should log a [LOAD-LOCK] ignore marker"
    )


# ---------------------------------------------------------------------------
# _set_curation_locked exists + references the right widgets.
# ---------------------------------------------------------------------------

def test_set_curation_locked_exists():
    assert hasattr(FusionReconstructionController, "_set_curation_locked")


def test_set_curation_locked_references_key_widgets():
    """The lock must cover the boxlist + at least the extract-features button
    + the QC plot widget (spot-check of the documented lock set).

    2026-06-06: the widget loop moved into the state-driven
    ``_refresh_step3_lock`` (``_set_curation_locked`` now just delegates), so the
    setEnabled / [LOAD-LOCK] assertions check that method's source.  The
    ``_CURATION_LOCK_WIDGETS`` tuple is still the authority for which widgets."""
    src = inspect.getsource(
        FusionReconstructionController._refresh_step3_lock
    )
    # The lock iterates the _CURATION_LOCK_WIDGETS class tuple, so check the
    # combined source of the method + the tuple definition.
    locked = set(FusionReconstructionController._CURATION_LOCK_WIDGETS)
    assert "_curation_bucket" in locked
    assert "qc_plot_widget" in locked
    # 2026-06-07: extract_features_btn is NO LONGER in the bulk tuple — it is
    # dual-purpose (Extract ⇄ Cancel Extraction) and SPECIAL-CASED in
    # _refresh_step3_lock so it stays ENABLED as "Cancel Extraction" during its
    # own QC-extract subprocess (mirrors extract_training_pairs_btn).  Assert it
    # is still covered by the lock — now via the method's special-case branch,
    # not the bulk tuple.
    assert "extract_features_btn" not in locked
    assert "extract_features_btn" in src
    assert "_qc_extract_running" in src
    # And it should disable (not enable) — setEnabled(not locked).
    assert "setEnabled" in src
    assert "[LOAD-LOCK]" in src


def test_set_curation_locked_locks_annotator():
    """When an annotator is open, the lock must forward to its
    ``set_load_locked``.

    2026-06-06: the annotator-forwarding logic moved into
    ``_refresh_step3_lock`` (``_set_curation_locked`` delegates)."""
    src = inspect.getsource(
        FusionReconstructionController._refresh_step3_lock
    )
    assert "set_load_locked" in src
    assert "_get_open_annotator_widget" in src


def test_set_curation_locked_runtime_disables_widgets():
    """Light-fake: locking flips setEnabled(False) on present widgets and
    getattr-guards absent ones.

    2026-06-06: the lock is now STATE-DRIVEN — ``_set_curation_locked`` delegates
    to ``_refresh_step3_lock`` which reads the live op flags rather than its
    argument.  Callers set ``_curation_busy`` BEFORE calling, so the fake mirrors
    that (busy=True before the True call, busy=False before the False call)."""
    class _W:
        def __init__(self):
            self.enabled = True

        def setEnabled(self, v):
            self.enabled = v

    class _UI:
        pass

    c = FusionReconstructionController.__new__(FusionReconstructionController)
    ui = _UI()
    # Provide a couple of the lock-set widgets; deliberately omit the rest so
    # the getattr-guard path is exercised.
    ui._curation_bucket = _W()
    ui.extract_features_btn = _W()
    ui.qc_plot_widget = _W()
    c.ui = ui
    c._get_open_annotator_widget = lambda: None

    # Mirror _busy_show: flip the busy flag True before the lock call.
    c._curation_busy = True
    c._set_curation_locked(True)
    assert ui._curation_bucket.enabled is False
    assert ui.extract_features_btn.enabled is False
    assert ui.qc_plot_widget.enabled is False

    # Mirror _busy_hide: flip the busy flag False before the unlock call.
    c._curation_busy = False
    c._set_curation_locked(False)
    assert ui._curation_bucket.enabled is True
    assert ui.extract_features_btn.enabled is True
    assert ui.qc_plot_widget.enabled is True


# ---------------------------------------------------------------------------
# Wiring: load start locks; finish unlocks + drives the bar.
# ---------------------------------------------------------------------------

def test_load_patch_locks_and_shows_bar():
    src = inspect.getsource(
        FusionReconstructionController._load_patch_for_qc_view
    )
    # 2026-06-06 redesign: the busy bar is ALWAYS scan mode (marquee); the
    # patch load streams stage TEXT into the underneath label (no determinate
    # %-bar).  _busy_show locks the curation UI for us, so we assert the helper
    # call rather than the old inline _set_curation_locked(True).
    assert "_busy_show(" in src, (
        "_load_patch_for_qc_view must show the busy bar via _busy_show"
    )
    # The initial label is descriptive ("Loading {stem}…").
    assert "Loading " in src, (
        "_load_patch_for_qc_view must pass a descriptive 'Loading …' label"
    )
    assert "worker.progress.connect" in src, (
        "_load_patch_for_qc_view must connect the worker progress signal"
    )


def test_on_patch_loaded_unlocks_and_finishes_bar():
    src = inspect.getsource(
        FusionReconstructionController._on_patch_loaded
    )
    # Post-rename (2026-06-05): unlock + bar-hide now go through _busy_hide().
    assert "_busy_hide()" in src, (
        "_on_patch_loaded must hide the busy bar + unlock via _busy_hide on success"
    )
    # 2026-06-06: the GPU/chain phases stream stage text ("Viewing {stem} — …")
    # into the label via the local _bar() helper instead of pushing a % to 100.
    assert "_bar(" in src, (
        "_on_patch_loaded must push stage text through its local _bar() helper"
    )


def test_on_patch_load_error_unlocks_and_hides_bar():
    src = inspect.getsource(
        FusionReconstructionController._on_patch_load_error
    )
    # Post-rename (2026-06-05): unlock + bar-hide now go through _busy_hide().
    assert "_busy_hide()" in src, (
        "_on_patch_load_error must hide the busy bar + unlock via _busy_hide on failure"
    )


def test_on_patch_load_progress_handler_exists():
    assert hasattr(
        FusionReconstructionController, "_on_patch_load_progress"
    )
    src = inspect.getsource(
        FusionReconstructionController._on_patch_load_progress
    )
    # 2026-06-06: the handler takes just a STAGE string, builds the
    # "Loading {stem}[ (downsampled)] — {stage}" line, and delegates to
    # _busy_progress (which updates the underneath status label).
    assert "_busy_progress(" in src
    assert "Loading " in src


# ---------------------------------------------------------------------------
# Worker emits a progress signal across its stages (stage TEXT, 2026-06-06).
# ---------------------------------------------------------------------------

def test_worker_has_progress_signal():
    # The Signal lives on the class.
    assert hasattr(_PatchLoadWorker, "progress")


def test_worker_run_emits_progress():
    src = inspect.getsource(_PatchLoadWorker.run)
    # 2026-06-06: stages are now descriptive TEXT, not percents.  The
    # always-emitted bookends are "reading…" and "rendering…".
    assert 'self.progress.emit("reading' in src
    assert 'self.progress.emit("rendering' in src
    # Conditional stages (only when downsample / source apply).
    assert 'self.progress.emit("downsampling' in src
    assert 'self.progress.emit("processing channels' in src


def test_worker_run_emits_progress_live(tmp_path):
    """Run the worker synchronously and confirm it emits at least the
    'reading…' and 'rendering…' bookend stages, plus a non-None labels
    payload.  (downsample_factor=1 + no source → those stages are skipped.)"""
    import numpy as np
    import tifffile

    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[0:2, 0:2, 0:2] = 1
    labels_path = tmp_path / "labels_3d.tif"
    tifffile.imwrite(str(labels_path), labels)

    worker = _PatchLoadWorker(
        str(labels_path), "", need_source=False,
        downsample_factor=1, to_channel_first=lambda a: a,
    )
    progress = []
    loaded = []
    errors = []
    worker.progress.connect(lambda stage: progress.append(stage))
    worker.loaded.connect(loaded.append)
    worker.error.connect(errors.append)
    worker.run()

    assert not errors, f"worker emitted error: {errors}"
    assert len(loaded) == 1
    assert "reading…" in progress, f"expected a 'reading…' emit, got {progress}"
    assert "rendering…" in progress, f"expected a 'rendering…' emit, got {progress}"
    # downsample off + no source → those two intermediate stages are skipped.
    assert "downsampling…" not in progress
    assert "processing channels…" not in progress


# ---------------------------------------------------------------------------
# Annotator side.
# ---------------------------------------------------------------------------

def test_annotator_has_set_load_locked():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )
    assert hasattr(Curation3DAnnotator, "set_load_locked")


def test_annotator_set_load_locked_behavior():
    """Locked → subviewers locked + navigator buttons disabled.  Unlocked
    (no op active) → navigator buttons re-enabled AND draw state restored.

    UPDATED 2026-06-06 (audit FIX-2): unlock now ALSO calls
    _restore_ready_to_draw_state when no op is active — curate ops / predict /
    reconstruct lock via _refresh_step3_lock with NO reload, so without this the
    subviewers stay editable=False (paint-freeze).  During a patch load it's a
    safe no-op (empty _active_labels_name → early-return)."""
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )
    src = inspect.getsource(Curation3DAnnotator.set_load_locked)
    # Lock path.
    assert "_lock_subviewers_for_preview" in src
    assert "set_label_action_buttons_enabled" in src
    assert "set_merge_enabled" in src
    # Unlock must gate on the refinement-op predicate.
    assert "_is_op_active_for_navigator" in src
    # Diagnostic markers.
    assert "[LOAD-LOCK] annotator locked" in src
    assert "[LOAD-LOCK] annotator unlocked" in src
    # FIX-2: unlock branch must *call* _restore_ready_to_draw_state so curate-op
    # / predict / reconstruct unlocks un-freeze drawing (patch-load path is a
    # safe no-op via the empty-name early-return).
    assert "self._restore_ready_to_draw_state()" in src


def test_annotator_unlock_restores_draw_state_only_when_no_op():
    """The _restore_ready_to_draw_state call must live in the unlock branch and
    be gated on no-op-active (so an in-flight op's gate isn't clobbered)."""
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )
    src = inspect.getsource(Curation3DAnnotator.set_load_locked)
    # The restore call should appear after the unlock's `else:` and after the
    # `op_active` predicate evaluation (i.e. gated, not in the lock branch).
    restore_idx = src.index("self._restore_ready_to_draw_state()")
    op_active_idx = src.index("op_active = bool(self._is_op_active_for_navigator())")
    assert restore_idx > op_active_idx, (
        "restore must be gated on the op_active predicate in the unlock branch"
    )
    # Safety reasoning preserved in a comment near the restore call.
    assert "early-return" in src, (
        "the FIX-2 safety reasoning (patch-load empty-name early-return) must be "
        "documented in set_load_locked"
    )

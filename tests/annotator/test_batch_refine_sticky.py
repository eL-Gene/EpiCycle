"""Tests for the sticky Batch-Refine recipe behavior (2026-06-07).

Sticky = the last-RUN recipe survives across Reject and patch switches so the
user can iterate without rebuilding the recipe by hand:

  * Reject KEEPS the dialog open in editable/idle (recipe rows intact) instead
    of tearing it down — but still RESTORES the pre-batch labels + RELEASES the
    controller lock (only the result is discarded, the dialog survives).
  * _run_batch_refine captures the validated recipe into _batch_last_recipe
    (serialized) on EVERY run (accepted OR later rejected).
  * _open_batch_refine_dialog pre-loads _batch_last_recipe into the dialog's
    recipe builder (steps ONLY — the target is left at default per patch); a
    malformed last-recipe must NOT break dialog-open.
  * _close_cleanup resets _batch_last_recipe (session-only).
  * a patch switch tears down an OPEN idle dialog (controller-side hook) so a
    stale dialog can't operate on the wrong patch — the recipe persists in
    _batch_last_recipe so reopening on the new patch pre-loads it.

Qt-free: Curation3DAnnotator.__new__ stubs (no display) + source introspection.
"""

from __future__ import annotations

import inspect
import types

import numpy as np

from packages.epicycle_annotator.batch_refine import RefineStep, recipe_to_json
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator


def _recipe():
    return [RefineStep(method="sdf", params={"sigma_z": 1.5, "sigma_xy": 1.5})]


# --------------------------------------------------------------------------- #
# 1) Reject keeps the dialog open (set_state idle), restores, releases lock
# --------------------------------------------------------------------------- #
class _RejectDialog:
    def __init__(self):
        self.state = "staged"

    def set_state(self, s):
        self.state = s


def _reject_ann(dlg):
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    ann._batch_staged = True
    ann._active_labels_name = "lbl"
    vol = np.ones((4, 4, 4), dtype=np.int32)
    ann.sync_manager = types.SimpleNamespace(shared_labels={"lbl": vol})
    union = (slice(0, 4), slice(0, 4), slice(0, 4))
    ann._batch_union_roi = union
    ann._batch_snapshot = np.zeros((4, 4, 4), dtype=np.int32)  # pre-batch: empty
    ann._batch_touched_ids = [1]
    ann._batch_refine_dialog = dlg
    # Stub the side-effect helpers.
    ann._sync_calls = []
    ann._sync_all_views_after_operation = lambda roi=None: ann._sync_calls.append(roi)
    ann._lock_fires = []
    ann._fire_batch_refine_lock = lambda active: ann._lock_fires.append(active)
    ann._restore_ready_to_draw_state = lambda: None
    ann._sync_refine_state = lambda: None
    ann._refresh_merge_button_state = lambda: None
    ann._refresh_label_action_buttons_state = lambda: None
    ann._teardown_called = []
    ann._teardown_batch_dialog = lambda: ann._teardown_called.append(True)
    return ann, vol, union


def test_reject_keeps_dialog_open_idle():
    dlg = _RejectDialog()
    ann, vol, union = _reject_ann(dlg)
    ann._on_batch_reject()
    # Dialog ref STAYS set (not nulled), set_state("idle") called, NOT torn down.
    assert ann._batch_refine_dialog is dlg
    assert dlg.state == "idle"
    assert ann._teardown_called == []


def test_reject_restores_labels_and_releases_lock():
    dlg = _RejectDialog()
    ann, vol, union = _reject_ann(dlg)
    ann._on_batch_reject()
    # Pre-batch (empty) snapshot restored into the live volume over the union.
    assert np.array_equal(vol[union], np.zeros((4, 4, 4), dtype=np.int32))
    # Lock released (False fired).
    assert ann._lock_fires == [False]
    # Views re-synced over the union ROI.
    assert ann._sync_calls == [union]
    # Staged flag cleared.
    assert ann._batch_staged is False


def test_reject_no_op_when_not_staged():
    dlg = _RejectDialog()
    ann, vol, union = _reject_ann(dlg)
    ann._batch_staged = False
    ann._on_batch_reject()
    # Early return — nothing fired, dialog untouched.
    assert ann._lock_fires == []
    assert dlg.state == "staged"


def test_reject_falls_back_to_teardown_when_dialog_gone():
    """Defensive: if the dialog ref is somehow None on a staged reject, fall
    back to teardown rather than dereferencing None."""
    ann, vol, union = _reject_ann(None)
    ann._on_batch_reject()
    assert ann._teardown_called == [True]
    assert ann._lock_fires == [False]


def test_reject_source_keeps_dialog_and_releases_lock():
    src = inspect.getsource(Curation3DAnnotator._on_batch_reject)
    # Still fires the lock False (terminal release).
    assert "self._fire_batch_refine_lock(False)" in src
    # Restores the snapshot over the union ROI.
    assert "shared[union_roi] = snap" in src
    # set_state("idle") path present; does NOT unconditionally teardown.
    assert 'set_state("idle")' in src


# --------------------------------------------------------------------------- #
# 2) _run_batch_refine captures the recipe into _batch_last_recipe
# --------------------------------------------------------------------------- #
class _RunDialog:
    def __init__(self, recipe):
        self._recipe = recipe
        self.state = None
        self.status = []

    def recipe(self):
        return self._recipe

    def set_state(self, s):
        self.state = s

    def set_status(self, t):
        self.status.append(t)

    def set_progress(self, *a):
        pass

    def set_busy(self):
        # The running state and its marquee now go up BEFORE the snapshot
        # copy (UX audit L7), so the fake answers them too.
        pass

    def repaint(self):
        pass


class _StopBeforeWorker(Exception):
    """Sentinel raised by a stubbed helper to halt _run_batch_refine AFTER the
    recipe-capture but BEFORE the real QThread/BatchRefineWorker launch — keeps
    the test Qt-free (no thread to crash on process teardown)."""


def _run_ann(recipe):
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    ann._active_labels_name = "lbl"
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[0, 0, 0] = 1
    ann.sync_manager = types.SimpleNamespace(shared_labels={"lbl": vol})
    ann._batch_refine_dialog = _RunDialog(recipe)
    # Pre-seed so getattr(...) returns the default cleanly (Qt's __getattr__ on a
    # bare __new__ instance raises RuntimeError rather than returning None for a
    # truly-missing attr).
    ann._batch_last_recipe = None
    ann._fire_batch_refine_lock = lambda active: None
    # Capture happens BEFORE _batch_snapshot = shared.copy(); raise here (which
    # runs AFTER capture, BEFORE the QThread launch) so we never construct a real
    # worker thread.  The raise propagates out of _run_batch_refine (the internal
    # try/except only wraps the thread block), so callers wrap the call.
    def _stop():
        raise _StopBeforeWorker()

    ann._lock_subviewers_for_preview = _stop
    ann._restore_ready_to_draw_state = lambda: None
    ann._sync_refine_state = lambda: None
    return ann


def test_run_captures_last_recipe():
    recipe = _recipe()
    ann = _run_ann(recipe)
    try:
        ann._run_batch_refine([1])
    except _StopBeforeWorker:
        pass
    assert getattr(ann, "_batch_last_recipe", None) == recipe_to_json(recipe)


def test_run_does_not_capture_on_empty_recipe():
    ann = _run_ann([])  # empty recipe → early return before capture
    ann._run_batch_refine([1])
    assert getattr(ann, "_batch_last_recipe", None) is None


def test_run_does_not_capture_on_invalid_recipe():
    ann = _run_ann(None)  # dialog.recipe() returns None → invalid → early return
    ann._run_batch_refine([1])
    assert getattr(ann, "_batch_last_recipe", None) is None


def test_run_captures_before_the_launch():
    """Behavioural (wiring audit M2; was an inspect.getsource index check):
    the sticky recipe is already captured when the snapshot / lock / launch
    begins."""
    recipe = _recipe()
    ann = _run_ann(recipe)
    seen = {}

    def spy(dlg, shared, labels_name, label_ids, rec, **kw):
        seen["sticky"] = ann._batch_last_recipe
        seen["recipe"] = rec

    ann._launch_batch_worker = spy
    ann._run_batch_refine([1])
    assert seen["sticky"] == recipe_to_json(recipe)
    assert [s.method for s in seen["recipe"]] == ["sdf"]


# --------------------------------------------------------------------------- #
# 3) _open_batch_refine_dialog pre-loads the last recipe (steps only)
# --------------------------------------------------------------------------- #
class _OpenDialog:
    def __init__(self):
        self.set_recipe_calls = []
        self.shown = False

    def set_recipe(self, recipe):
        self.set_recipe_calls.append(recipe)

    def show(self):
        self.shown = True

    def raise_(self):
        pass


def _open_ann(last_recipe, dlg, make_raises=False):
    """Set up an annotator far enough into _open_batch_refine_dialog that the
    pre-load block runs.  We monkeypatch the heavy dialog-construction +
    history-refresh so the method reaches the pre-load + show()."""
    ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
    # Pre-seed _batch_refine_dialog so the stale-dialog teardown check at the top
    # of _open_batch_refine_dialog returns cleanly (Qt's __getattr__ on a bare
    # __new__ instance raises rather than returning the default for a missing
    # attr).
    ann._batch_refine_dialog = None
    ann._load_locked = False
    ann._refine_state = types.SimpleNamespace(can_start_batch_refine=lambda: True)
    ann._sync_refine_state = lambda: None
    ann._active_labels_name = "lbl"
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[0, 0, 0] = 1
    ann.sync_manager = types.SimpleNamespace(shared_labels={"lbl": vol})
    ann._navigator_widget = types.SimpleNamespace(
        _items=[], selected_label_ids=lambda: set()
    )
    ann._save_batch_recipe_cb = None
    ann._batch_last_recipe = last_recipe
    ann._refresh_merge_button_state = lambda: None
    ann._refresh_label_action_buttons_state = lambda: None
    # Refresh-history is a no-op for these tests.
    ann._refresh_batch_recipe_history = lambda d=None: None
    ann._resolve_batch_targets = lambda spec: []
    return ann, vol, dlg


def _patch_dialog_ctor(monkeypatch, dlg):
    """Replace BatchRefineDialog so _open_batch_refine_dialog builds our fake."""
    import napari_epicycle.annotator.batch_refine_dialog as brd_mod

    monkeypatch.setattr(brd_mod, "BatchRefineDialog", lambda **kw: dlg)


def test_open_preloads_last_recipe(monkeypatch):
    dlg = _OpenDialog()
    rj = recipe_to_json(_recipe())
    ann, vol, _ = _open_ann(rj, dlg)
    _patch_dialog_ctor(monkeypatch, dlg)
    ann._open_batch_refine_dialog()
    # set_recipe called once with the round-tripped last recipe.
    assert len(dlg.set_recipe_calls) == 1
    loaded = dlg.set_recipe_calls[0]
    assert [s.method for s in loaded] == ["sdf"]
    assert dlg.shown is True


def test_open_does_not_preload_when_none(monkeypatch):
    dlg = _OpenDialog()
    ann, vol, _ = _open_ann(None, dlg)
    _patch_dialog_ctor(monkeypatch, dlg)
    ann._open_batch_refine_dialog()
    assert dlg.set_recipe_calls == []
    assert dlg.shown is True


def test_open_malformed_last_recipe_does_not_break(monkeypatch):
    """A malformed _batch_last_recipe must NOT block dialog-open — it logs +
    opens with an empty builder."""
    class _RaisingDialog(_OpenDialog):
        def set_recipe(self, recipe):
            raise ValueError("bad recipe")

    dlg = _RaisingDialog()
    # Deliberately malformed payload.
    ann, vol, _ = _open_ann([{"method": "NOT_A_METHOD"}], dlg)
    _patch_dialog_ctor(monkeypatch, dlg)
    ann._open_batch_refine_dialog()  # must not raise
    assert dlg.shown is True  # dialog still shown


def test_open_preload_source_target_not_restored():
    src = inspect.getsource(Curation3DAnnotator._open_batch_refine_dialog)
    # Pre-load uses set_recipe (steps), wrapped in try/except.
    assert "set_recipe(" in src
    assert "recipe_from_json" in src
    # Steps-only: there is no target/tier restore in the pre-load block.
    # (set_recipe itself is reload-only per the dialog contract.)
    preload_idx = src.index("recipe_from_json")
    # No 'target_spec' or 'set_target' restore call near the pre-load.
    assert "set_target" not in src


# --------------------------------------------------------------------------- #
# 4) _close_cleanup resets _batch_last_recipe (source)
# --------------------------------------------------------------------------- #
def test_close_cleanup_resets_last_recipe_source():
    src = inspect.getsource(Curation3DAnnotator._close_cleanup)
    assert "self._batch_last_recipe = None" in src


# --------------------------------------------------------------------------- #
# 5) Patch-switch teardown — controller tears down an OPEN idle dialog
# --------------------------------------------------------------------------- #
def test_post_switch_sync_tears_down_open_dialog_source():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    src = inspect.getsource(
        FusionReconstructionController._post_switch_annotator_sync
    )
    # Guards on an OPEN dialog and tears it down.
    assert 'getattr(widget, "_batch_refine_dialog", None) is not None' in src
    assert "widget._teardown_batch_dialog()" in src


def test_post_switch_sync_tears_down_open_dialog_behavior():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    # Stub everything _post_switch_annotator_sync touches after the teardown.
    ctrl._collect_flagged_labels = lambda: ([], {})
    ctrl._on_annotator_mark_tier = lambda *a, **k: None
    ctrl._on_annotator_label_deleted = lambda *a, **k: None
    ctrl._on_annotator_label_merged = lambda *a, **k: None
    ctrl._get_active_prob_path = lambda: None

    torn = []

    widget = types.SimpleNamespace(
        _batch_refine_dialog=object(),
        _teardown_batch_dialog=lambda: torn.append(True),
        _suppress_layer_removal_cleanup=True,
        layer_manager=types.SimpleNamespace(_suppress_events=True),
        set_focused_refinement=lambda ids, c: None,
        set_triage_callbacks=lambda **k: None,
        set_prob_volume_path=lambda p: None,
        _active_labels_name="old",
        _active_image_name="old",
        _schedule_deferred_auto_load=lambda ms: None,
    )
    FusionReconstructionController._post_switch_annotator_sync(
        ctrl, widget, "new_patch"
    )
    assert torn == [True]


def test_post_switch_sync_no_teardown_when_no_dialog():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl._collect_flagged_labels = lambda: ([], {})
    ctrl._on_annotator_mark_tier = lambda *a, **k: None
    ctrl._on_annotator_label_deleted = lambda *a, **k: None
    ctrl._on_annotator_label_merged = lambda *a, **k: None
    ctrl._get_active_prob_path = lambda: None

    torn = []
    widget = types.SimpleNamespace(
        _batch_refine_dialog=None,
        _teardown_batch_dialog=lambda: torn.append(True),
        _suppress_layer_removal_cleanup=True,
        layer_manager=types.SimpleNamespace(_suppress_events=True),
        set_focused_refinement=lambda ids, c: None,
        set_triage_callbacks=lambda **k: None,
        set_prob_volume_path=lambda p: None,
        _active_labels_name="old",
        _active_image_name="old",
        _schedule_deferred_auto_load=lambda ms: None,
    )
    FusionReconstructionController._post_switch_annotator_sync(
        ctrl, widget, "new_patch"
    )
    assert torn == []  # no open dialog → no teardown

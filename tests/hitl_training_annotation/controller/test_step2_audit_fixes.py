"""Tests for Step 2 audit fixes (2026-05-22 audit).

Covers three fixes derived from independent 5-agent audit:

* **C2** — X/R event filter installed on the 2D annotator canvas does
  not accumulate across close→relaunch cycles, otherwise X (atomic
  erase) and R (pick-refine) fire 2-3× per keystroke.

* **C3** — ``_curating_persist_pending`` accepts MULTIPLE in-flight
  slice edits (dict keyed by slice_label) instead of a single tuple
  that gets overwritten when the user paints A then B within the 2s
  debounce window.  Previously caused silent data loss on A.

* **C4** — ``shutdown()`` flushes ``_curating_persist_timer`` (in
  addition to the existing ``_state_save_timer`` flush) so pending
  curated-edit persists complete before the process exits.  Previously
  caused silent data loss when napari was closed within 2s of the last
  paint stroke.

**C1 reverted 2026-05-22:** Initial fix set
``_epicycle_defer_main_refresh = True`` on the main work_labels layer
in ``_ensure_work_labels_layer``, but the ``DrawingHandler`` reads the
flag from the **subviewer mirror** (`drawing.py:172`), not the main
layer.  Flag never reached the read site → no perf improvement.
Reverted pending a proper fix at the subviewer-mirror creation site
(`packages/epicycle_annotator/subviewers.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


# ----------------------------------------------------------------------------
# Lightweight mocks
# ----------------------------------------------------------------------------

class _MockLabelsLayer:
    """Minimal stand-in for napari.layers.Labels."""

    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.opacity = 0.8
        self.colormap = None


class _MockImageLayer:
    """Minimal stand-in for napari.layers.Image (used as ref_source)."""

    def __init__(self, data, name):
        self.data = data
        self.name = name


class _MockLayerList:
    """Minimal subset of napari's LayerList interface."""

    def __init__(self):
        self._layers = []

    def __iter__(self):
        return iter(self._layers)

    def __contains__(self, name):
        return any(l.name == name for l in self._layers)

    def __getitem__(self, name):
        for l in self._layers:
            if l.name == name:
                return l
        raise KeyError(name)


class _MockViewer:
    """Minimal stand-in that captures the layer added via add_labels."""

    def __init__(self):
        self.layers = _MockLayerList()
        self._added_labels = []  # capture history

    def add_labels(self, data, *, name, opacity=0.8):
        layer = _MockLabelsLayer(data, name)
        layer.opacity = opacity
        self.layers._layers.append(layer)
        self._added_labels.append(layer)
        return layer


# ----------------------------------------------------------------------------
# C3: curating-persist pending dict (not single tuple)
# ----------------------------------------------------------------------------

def _make_persist_ctrl():
    """Minimal controller for testing _schedule_curating_persist /
    _flush_curating_persist_pending without going through QTimer."""
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.loaded_patch_name = "patch01"

    # Mock segmentation_manager with a mutable data_masks dict.
    class _SM:
        def __init__(self):
            self.data_masks = {}
    ctrl.segmentation_manager = _SM()

    # Mock _layer_by_name to return a small labels layer.
    layer = _MockLabelsLayer(np.zeros((4, 8, 8), dtype=np.int32), "patch01_work_labels_lateral_3d")
    # Stuff distinct values into slices so we can verify each was saved.
    layer.data[0, :, :] = 11
    layer.data[1, :, :] = 22
    layer.data[2, :, :] = 33
    ctrl._layer_by_name = lambda name: layer if name == "patch01_work_labels_lateral_3d" else None
    ctrl._work_labels_layer_name_for_view = (
        lambda patch_name=None, view_name=None: "patch01_work_labels_lateral_3d"
    )

    # Capture persist calls.  (async_write kwarg added 2026-06-02 Stage 2 —
    # the debounce path now calls persist with async_write=True.)
    ctrl._persist_calls = []
    def _fake_persist(sl, async_write=False):
        ctrl._persist_calls.append(sl)
        return {"ok": True}
    ctrl._persist_curated_training_pair = _fake_persist

    # No-op state save scheduler.
    ctrl._schedule_state_save = lambda: None

    return ctrl, layer


def test_c3_pending_is_dict_not_tuple():
    """C3 — _schedule_curating_persist stores pending edits as a dict
    keyed by slice_label so multiple in-flight slice edits coexist.
    Previously a single tuple was overwritten on each call, silently
    losing edits to all-but-the-last slice when the user painted A
    then B within the 2s debounce window.
    """
    ctrl, _ = _make_persist_ctrl()

    # Schedule three different slices in rapid succession.  Without a
    # real QTimer running, the dict should hold all three pending
    # entries — proving the regression is gone (the prior bug would
    # collapse them to just the last tuple).
    ctrl._schedule_curating_persist("patch01_XY_0", "XY", 0)
    ctrl._schedule_curating_persist("patch01_XY_1", "XY", 1)
    ctrl._schedule_curating_persist("patch01_XY_2", "XY", 2)

    pending = ctrl._curating_persist_pending
    assert isinstance(pending, dict), (
        f"C3 regression: _curating_persist_pending should be a dict "
        f"(C3 fix 2026-05-22), got {type(pending).__name__!s}."
    )
    assert set(pending.keys()) == {"patch01_XY_0", "patch01_XY_1", "patch01_XY_2"}, (
        "C3 regression: expected all three slice labels to be pending, "
        f"got {set(pending.keys())}.  The single-tuple bug would only "
        "preserve the last call."
    )


def test_c3_flush_persists_all_pending_slices():
    """C3 — _flush_curating_persist_pending drains ALL queued slices,
    not just the latest one.  The single-tuple bug caused only the
    most recent slice to be persisted on timer fire.
    """
    ctrl, layer = _make_persist_ctrl()

    ctrl._schedule_curating_persist("patch01_XY_0", "XY", 0)
    ctrl._schedule_curating_persist("patch01_XY_1", "XY", 1)
    ctrl._schedule_curating_persist("patch01_XY_2", "XY", 2)

    # Drain (simulates the QTimer firing).
    ctrl._flush_curating_persist_pending()

    assert sorted(ctrl._persist_calls) == [
        "patch01_XY_0", "patch01_XY_1", "patch01_XY_2",
    ], (
        f"C3 regression: expected all three slice persists, got "
        f"{ctrl._persist_calls}.  Single-tuple bug would only persist "
        "the last one."
    )

    # Each slice's actual mask bytes should match what was extracted
    # from the volume — verifies the per-slice view-aware indexing.
    sm = ctrl.segmentation_manager
    assert sm.data_masks["patch01_XY_0"][0, 0] == 11
    assert sm.data_masks["patch01_XY_1"][0, 0] == 22
    assert sm.data_masks["patch01_XY_2"][0, 0] == 33

    # Pending dict should be empty after successful drain.
    assert ctrl._curating_persist_pending == {}


def test_c3_repeat_call_for_same_slice_overwrites_only_that_slice():
    """C3 — calling _schedule_curating_persist twice for the SAME
    slice should still latest-wins for that slice (within the dict
    entry), but not affect OTHER slices' entries.  This preserves the
    intentional "latest edit per slice" behaviour.
    """
    ctrl, _ = _make_persist_ctrl()

    ctrl._schedule_curating_persist("patch01_XY_0", "XY", 0)
    ctrl._schedule_curating_persist("patch01_XY_1", "XY", 1)
    # Repeat call for slice 0 with a different idx (shouldn't happen
    # in practice, but verifies semantics).
    ctrl._schedule_curating_persist("patch01_XY_0", "XY", 0)

    pending = ctrl._curating_persist_pending
    assert "patch01_XY_0" in pending
    assert "patch01_XY_1" in pending
    assert len(pending) == 2  # not 3 — repeat for same key just overwrites


# ----------------------------------------------------------------------------
# C4: shutdown flushes curating-persist timer + drains pending
# ----------------------------------------------------------------------------

def test_c4_shutdown_drains_pending_curating_persists():
    """C4 — :meth:`shutdown` calls
    :meth:`_flush_curating_persist_pending` so that closing napari
    within the 2-second debounce window doesn't silently lose pending
    curated-edit persists.  Previously only ``_state_save_timer`` was
    flushed; the curating-persist timer would be destroyed with the
    pending dict still populated, and the edits would never reach
    disk.
    """
    ctrl, _ = _make_persist_ctrl()

    # Queue some pending edits as if the user was mid-edit.
    ctrl._schedule_curating_persist("patch01_XY_0", "XY", 0)
    ctrl._schedule_curating_persist("patch01_XY_1", "XY", 1)

    # Stub the bits shutdown also touches but we don't care about here.
    ctrl._save_internal_state = lambda: None
    ctrl._training_abort_flag = None
    ctrl._training_thread = None

    # User closes napari before the 2s timer fires.
    ctrl.shutdown()

    # Both pending edits should have been persisted to disk via the
    # shutdown flush — NOT silently lost.
    assert sorted(ctrl._persist_calls) == [
        "patch01_XY_0", "patch01_XY_1",
    ], (
        "C4 regression: shutdown() did not flush pending curating "
        "persists.  Closing napari within the debounce window will "
        "silently lose user edits."
    )
    # Pending dict should be empty post-shutdown.
    assert ctrl._curating_persist_pending == {}


def test_c4_shutdown_with_no_pending_is_safe():
    """C4 — shutdown() must be safe to call with no pending curating
    edits and no curating-persist timer (e.g. immediate close after
    workspace load, before any annotation activity).
    """
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl._save_internal_state = lambda: None
    ctrl._training_abort_flag = None
    ctrl._training_thread = None

    # Should not raise.
    ctrl.shutdown()


def test_c4_shutdown_survives_deleted_timers():
    """C4 — the two debounce timers are parentless QTimers; at napari close
    Qt can delete their C++ side first, and ``isActive()`` on such a wrapper
    raises ``RuntimeError: wrapped C/C++ object of type QTimer has been
    deleted``.  shutdown() must not raise, and everything after the timers
    (the curating flush, the pair-writer flush, the state save) must still
    run, in order.
    """
    calls = []

    class _DeletedTimer:
        def isActive(self):
            raise RuntimeError(
                "wrapped C/C++ object of type QTimer has been deleted"
            )

        def stop(self):
            raise RuntimeError(
                "wrapped C/C++ object of type QTimer has been deleted"
            )

    class _Writer:
        def flush(self, timeout=None):
            calls.append("writer_flush")

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl._state_save_timer = _DeletedTimer()
    ctrl._curating_persist_timer = _DeletedTimer()
    ctrl._flush_curating_persist_pending = lambda: calls.append("curating_flush")
    ctrl._pair_writer = _Writer()
    ctrl._save_internal_state = lambda: calls.append("save_state")
    ctrl._training_abort_flag = None
    ctrl._training_thread = None

    ctrl.shutdown()  # must not raise

    assert calls == ["curating_flush", "writer_flush", "save_state"], calls


# ----------------------------------------------------------------------------
# C2: 2D hotkey filter — canvas-attribute lifecycle (Qt-free unit test)
# ----------------------------------------------------------------------------

def test_c2_teardown_clears_per_canvas_filter_attribute():
    """C2 — :meth:`_teardown_2d_hotkeys` must clear the per-canvas
    ``_epicycle_curation_keyfilter`` attribute in addition to calling
    ``removeEventFilter``.  This ensures a subsequent setup can detect
    a stale filter via the canvas attribute and remove it explicitly,
    defending against the case where ``removeEventFilter`` silently
    failed on the previous teardown.
    """
    from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator

    class _FakeCanvas:
        """Minimal stand-in supporting removeEventFilter + attr assignment."""
        def __init__(self):
            self._installed = []
            self._removed = []

        def installEventFilter(self, filt):
            self._installed.append(filt)

        def removeEventFilter(self, filt):
            self._removed.append(filt)

    annot = Curation2DAnnotator.__new__(Curation2DAnnotator)

    canvas_a = _FakeCanvas()
    canvas_b = _FakeCanvas()
    filt_a = object()
    filt_b = object()
    canvas_a._epicycle_curation_keyfilter = filt_a
    canvas_b._epicycle_curation_keyfilter = filt_b
    annot._key_filters = [(canvas_a, filt_a), (canvas_b, filt_b)]

    annot._teardown_2d_hotkeys()

    # Tracking list cleared.
    assert annot._key_filters == []
    # Both filters had removeEventFilter called.
    assert filt_a in canvas_a._removed
    assert filt_b in canvas_b._removed
    # Per-canvas attribute cleared on both (C2 fix).
    assert not hasattr(canvas_a, "_epicycle_curation_keyfilter"), (
        "C2 regression: teardown did not clear the per-canvas "
        "_epicycle_curation_keyfilter attribute.  A subsequent setup "
        "won't be able to detect+remove this stale filter."
    )
    assert not hasattr(canvas_b, "_epicycle_curation_keyfilter")


def test_c2_teardown_swallows_remove_event_filter_failure():
    """C2 — even if ``removeEventFilter`` raises (e.g. filter object
    was sip-deleted), teardown must still clear the tracking list and
    the per-canvas attribute so the next setup can start clean.
    """
    from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator

    class _FailingCanvas:
        def removeEventFilter(self, filt):
            raise RuntimeError("simulated sip-deletion")

    annot = Curation2DAnnotator.__new__(Curation2DAnnotator)

    canvas = _FailingCanvas()
    filt = object()
    canvas._epicycle_curation_keyfilter = filt
    annot._key_filters = [(canvas, filt)]

    # Must not raise.
    annot._teardown_2d_hotkeys()

    # Tracking list still cleared despite removeEventFilter failure.
    assert annot._key_filters == []
    # Per-canvas attribute still cleared despite removeEventFilter failure.
    assert not hasattr(canvas, "_epicycle_curation_keyfilter"), (
        "C2 regression: when removeEventFilter raises, the per-canvas "
        "attribute is left in place, defeating the defensive lookup "
        "on the next setup."
    )


# ----------------------------------------------------------------------------
# C5: _refresh_label_data_in_subviewers re-applies edit constraints
# ----------------------------------------------------------------------------

def test_c5_refresh_label_data_in_subviewers_calls_apply_edit_constraints():
    """C5 (2026-05-23) — :meth:`_refresh_label_data_in_subviewers` MUST
    call :meth:`_apply_edit_constraints` at the end.

    **Why this matters:** napari's ``Labels._reset_editable`` (in
    ``napari/layers/labels/labels.py``) runs as a side effect of any
    ``layer.data = ...`` assignment.  It unconditionally sets
    ``editable=True`` on the layer regardless of the current mode-aware
    constraint state.

    Without the explicit re-application, after any data refresh on
    subviewer mirrors (e.g. post-Cellpose-prediction in Step 2 HITL),
    the XZ and YZ axial subviewer mirrors flip from ``editable=False``
    (correct for lateral mode) back to ``editable=True``.  This let
    ``_do_pick_refine`` set ``mode='paint'`` on them and the user
    could then paint on axial subviewers while in lateral mode.

    Trace evidence (caller stack from [DIAG-WATCH] on 2026-05-23):
        labels.py:814(_reset_editable)
          <- labels.py:586(data)
          <- native_multiview_widget.py:2583(_refresh_label_data_in_subviewers)
          <- native_multiview_widget.py:2422(_request_segment_slice)
          <- native_multiview_widget.py:2481(_request_segment_active_mode)
    """
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator,
    )

    annot = NativeMultiviewAnnotator.__new__(NativeMultiviewAnnotator)
    # No layers — _refresh loop short-circuits but
    # _apply_edit_constraints still must be called once at the end.
    annot._active_labels_name = ""
    annot._active_labels_names = []

    # Stub the viewer + subviewer attributes the refresh loop reads.
    class _Dims:
        ndisplay = 2
    class _Layers:
        def __iter__(self):
            return iter([])
    class _VM:
        dims = _Dims()
        layers = _Layers()
    class _MainViewer:
        layers = _Layers()
    annot.viewer = _MainViewer()
    annot.viewer_model_xy = _VM()
    annot.viewer_model_xz = _VM()
    annot.viewer_model_yz = _VM()
    annot._layer_by_name = lambda layers, name: None  # no main layer found

    # Spy on _apply_edit_constraints.
    annot._apply_edit_constraints_call_count = 0
    def _spy():
        annot._apply_edit_constraints_call_count += 1
    annot._apply_edit_constraints = _spy

    annot._refresh_label_data_in_subviewers()

    assert annot._apply_edit_constraints_call_count == 1, (
        f"C5 regression: _refresh_label_data_in_subviewers MUST call "
        f"_apply_edit_constraints exactly once at the end to override "
        f"napari's Labels._reset_editable side effect.  Got "
        f"{annot._apply_edit_constraints_call_count} call(s).  Without "
        f"this, axial subviewer mirrors flip to editable=True after any "
        f"data refresh, breaking the lateral/axial editability lockdown."
    )


def test_c5_apply_edit_constraints_failure_is_swallowed():
    """C5 — if :meth:`_apply_edit_constraints` raises (e.g. during
    teardown when subviewer attributes are torn down), the refresh
    itself must still complete.  Defensive try/except wrap.
    """
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator,
    )

    annot = NativeMultiviewAnnotator.__new__(NativeMultiviewAnnotator)
    annot._active_labels_name = ""
    annot._active_labels_names = []

    class _Dims:
        ndisplay = 2
    class _Layers:
        def __iter__(self):
            return iter([])
    class _VM:
        dims = _Dims()
        layers = _Layers()
    class _MainViewer:
        layers = _Layers()
    annot.viewer = _MainViewer()
    annot.viewer_model_xy = _VM()
    annot.viewer_model_xz = _VM()
    annot.viewer_model_yz = _VM()
    annot._layer_by_name = lambda layers, name: None

    def _raise():
        raise RuntimeError("simulated teardown race")
    annot._apply_edit_constraints = _raise

    # Must not propagate the exception out of the refresh.
    annot._refresh_label_data_in_subviewers()

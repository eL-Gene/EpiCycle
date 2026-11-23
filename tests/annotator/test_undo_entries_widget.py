"""Widget side of the curation-undo family: one edit, one entry.

* Interpolate -> Refine Further -> Accept fires ONE callback that carries
  the refine layer as a ``pre_layers`` entry.
* A Refine Further chain on reshape fragments defers its layers to the
  final Accept all instead of pushing its own entry.
* A paint stroke records the plane it wrote and pushes one entry.
* Reconciling a label that is already gone never pushes.
"""
from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K

_BOUND = ("_iso_owner_alive", "_iso_bracket_open", "_iso_take", "_iso_drop",
          "_iso_begin", "_iso_suspend", "_iso_end", "_iso_live_ids",
          "_reshape_pieces_cap")

ROI_I = (slice(0, 6), slice(0, 8), slice(0, 8))
ROI_S = (slice(0, 6), slice(0, 8), slice(0, 10))


@pytest.fixture(autouse=True)
def _plain_provenance(monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as W
    monkeypatch.setattr(W, "_refine_provenance", lambda self: ([], []))
    monkeypatch.setattr(W, "_refine_clear_records", lambda self: None)


def _fake():
    f = MagicMock()
    f._load_locked = False
    f._is_auto_advance_enabled = lambda: False
    f._iso_bracket = None
    f._ISO_OWNER_KINDS = K._ISO_OWNER_KINDS
    f._get_advanced_features_widget = lambda: None
    f._navigator_widget = None
    f._split_fragment_refine = False
    f._frag_undo_layers = []
    for n in _BOUND:
        setattr(f, n, types.MethodType(getattr(K, n), f))
    return f


# ---------------------------------------------------------------- F18 ------
def test_interp_refine_accept_fires_one_callback_with_both_layers():
    f = _fake()
    f._smooth_running = True
    f._interp_refine_mode = True
    f._smooth_label_id = 431
    f._interp_label_id = 431
    f._smooth_roi, f._smooth_snapshot = ROI_S, "smooth-snap"
    f._interp_roi, f._interp_snapshot = ROI_I, "interp-snap"
    smoothed, interp = MagicMock(), MagicMock()
    f._on_label_smoothed_cb = smoothed
    f._on_interpolation_accepted_cb = interp

    K._accept_smooth(f)

    smoothed.assert_not_called()
    interp.assert_called_once()
    args, kw = interp.call_args
    assert args == (431,)
    assert kw["roi_slices"] == ROI_I
    assert kw["old_roi_data"] == "interp-snap"
    assert kw["pre_layers"] == [(ROI_S, "smooth-snap")]
    assert kw["steps"] == [] and kw["recipes"] == []


# ---------------------------------------------------------------- F10 ------
def test_fragment_smooth_accept_defers_its_layer():
    f = _fake()
    f._split_fragment_refine = True
    f._interp_refine_mode = False
    f._smooth_running = True
    f._smooth_label_id = 686
    f._smooth_roi, f._smooth_snapshot = ROI_S, "frag-snap"
    smoothed = MagicMock()
    f._on_label_smoothed_cb = smoothed

    K._accept_smooth(f)

    smoothed.assert_not_called()
    assert f._frag_undo_layers == [(ROI_S, "frag-snap")]


def test_accept_all_passes_the_deferred_layers_once():
    f = _fake()
    f._split_fragment_refine = True
    f._smooth_running = False
    f._split_label_id = 250
    f._split_fragment_ids = [686, 687]
    f._split_roi, f._split_snapshot = ROI_I, "parent-snap"
    f._reshape_fragment_ticked = None
    f._reshape_result = None
    f._frag_undo_layers = [(ROI_S, "frag-snap")]
    f._reset_split_state = types.MethodType(K._reset_split_state, f)
    split_cb = MagicMock()
    f._on_label_split_cb = split_cb

    K._accept_split(f)

    split_cb.assert_called_once()
    _, kw = split_cb.call_args
    assert kw["roi_slices"] == ROI_I and kw["old_roi_data"] == "parent-snap"
    assert kw["pre_layers"] == [(ROI_S, "frag-snap")]
    assert f._frag_undo_layers == []


def test_reset_split_state_clears_deferred_layers():
    f = _fake()
    f._frag_undo_layers = [(ROI_S, "x")]
    K._reset_split_state(f)
    assert f._frag_undo_layers == []


# ---------------------------------------------------------------- F4 -------
def test_reconcile_of_a_gone_label_never_pushes():
    f = _fake()
    cb = MagicMock()
    f._on_delete_label_cb = cb
    K._handle_manually_deleted_label(f, 633)
    cb.assert_called_once_with(
        633, roi_slices=None, old_roi_data=None, reconcile=True)


def test_navigator_delete_forgets_the_last_navigated_label():
    f = _fake()
    f._last_navigated_label_id = 633
    vol = np.zeros((4, 4, 4), np.int32)
    vol[1:3, 1:3, 1:3] = 633
    f._active_labels_name = "labels"
    f.sync_manager.shared_labels.get.return_value = vol
    f._label_action_buttons_active = lambda: True
    f._snapshot_label_roi = lambda v, lid: (
        (slice(0, 4),) * 3, v[(slice(0, 4),) * 3].copy())
    f._on_delete_label_cb = MagicMock()

    K._on_delete_label_from_navigator(f, 633)

    assert f._last_navigated_label_id is None
    f._on_delete_label_cb.assert_called_once()


# ---------------------------------------------------------------- F17 ------
def _stroke_fake(shared):
    f = _fake()
    f._split_running = False
    f._smooth_running = False
    f._interp_refine_mode = False
    f.staged_kinds = lambda: frozenset()
    f._active_labels_name = "labels"
    f.sync_manager.shared_labels.get.return_value = shared
    f._get_selected_label = lambda: 91
    f._draw_label_id = 91          # not the first stroke: no Clear Edits arm
    f._isolation_proxy_active = False
    f._stroke_plane = None
    f._stroke_plane_consumed = False
    f._draw_session = 0
    f._on_paint_stroke_cb = MagicMock()
    f._on_new_label_painted_cb = MagicMock()
    for n in ("_on_subviewer_stroke_begin", "_push_paint_stroke",
              "_on_subviewer_new_label_painted", "_on_subviewer_draw_complete"):
        setattr(f, n, types.MethodType(getattr(K, n), f))
    return f


@pytest.mark.parametrize("orient,expect", [
    ("XY", (3, slice(None), slice(None))),
    ("XZ", (slice(None), 3, slice(None))),
    ("YZ", (slice(None), slice(None), 3)),
])
def test_stroke_pushes_one_entry_for_its_plane(orient, expect):
    shared = np.zeros((6, 8, 8), np.int32)
    f = _stroke_fake(shared)
    f._on_subviewer_stroke_begin(orient, 3)
    before = shared[expect].copy()
    shared[expect] = 91
    f._on_subviewer_draw_complete()
    cb = f._on_paint_stroke_cb
    cb.assert_called_once()
    args, kw = cb.call_args
    assert args == (91,)
    assert kw["roi_slices"] == expect
    assert np.array_equal(kw["old_roi_data"], before)
    f._on_new_label_painted_cb.assert_not_called()


def test_stroke_that_changed_nothing_pushes_nothing():
    shared = np.zeros((6, 8, 8), np.int32)
    f = _stroke_fake(shared)
    f._on_subviewer_stroke_begin("XY", 3)
    f._on_subviewer_draw_complete()
    f._on_paint_stroke_cb.assert_not_called()


def test_new_label_stroke_carries_the_plane_and_pushes_once():
    shared = np.zeros((6, 8, 8), np.int32)
    f = _stroke_fake(shared)
    f._on_subviewer_stroke_begin("XY", 2)
    shared[2, 1:3, 1:3] = 687
    f._on_subviewer_new_label_painted(687)
    f._on_subviewer_draw_complete()
    f._on_new_label_painted_cb.assert_called_once()
    _, kw = f._on_new_label_painted_cb.call_args
    assert kw["roi_slices"] == (2, slice(None), slice(None))
    assert not kw["old_roi_data"].any()
    f._on_paint_stroke_cb.assert_not_called()


def test_no_stroke_entry_while_an_op_is_staged():
    shared = np.zeros((6, 8, 8), np.int32)
    f = _stroke_fake(shared)
    f.staged_kinds = lambda: frozenset({"x"})
    f._on_subviewer_stroke_begin("XY", 3)
    shared[3] = 91
    f._on_subviewer_draw_complete()
    f._on_paint_stroke_cb.assert_not_called()


def test_drawing_handler_reports_the_plane_before_any_sync():
    from packages.epicycle_annotator.drawing import DrawingHandler

    order = []
    sync = MagicMock()
    sync.sync_to_main_view.side_effect = lambda *a, **k: order.append("sync")
    h = DrawingHandler(sync)
    h.stroke_begin_callback = lambda o, i: order.append(("begin", o, i))
    h.pre_sync_callback = lambda: order.append("pre")
    layer = SimpleNamespace(mode="erase", name="labels_XY", selected_label=1,
                            refresh=lambda: None)
    vm = SimpleNamespace(
        dims=SimpleNamespace(current_step=(4, 0, 0)),
        cursor=SimpleNamespace(position=(4, 1, 1)))
    ev = SimpleNamespace(type="mouse_press", button=1)
    gen = h._on_mouse_drag(layer, ev, "XY", vm)
    next(gen)
    ev.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(gen)
    assert order[0] == ("begin", "XY", 4)
    assert order.index("pre") < order.index("sync")


# ---------------------------------------------------------------- H1 -------
def _real_layer_with_handler(insert_first):
    """A real napari Labels layer in paint mode with our drag callback
    installed the way the subviewer manager does."""
    from napari.layers import Labels
    from packages.epicycle_annotator.drawing import DrawingHandler

    shared = np.zeros((4, 6, 6), np.int32)
    f = _stroke_fake(shared)
    layer = Labels(shared)
    layer.mode = "paint"            # napari's draw callback is in the list now
    layer.selected_label = 91
    layer.name = "labels_XY"
    sync = MagicMock()
    h = DrawingHandler(sync)
    h.stroke_begin_callback = f._on_subviewer_stroke_begin
    h.post_sync_callback = f._on_subviewer_draw_complete
    vm = SimpleNamespace(
        dims=SimpleNamespace(current_step=(2, 0, 0)),
        cursor=SimpleNamespace(position=(2, 3, 3)))
    cb = h.create_mouse_drag_callback("XY", vm)
    if insert_first:
        layer.mouse_drag_callbacks.insert(0, cb)
    else:
        layer.mouse_drag_callbacks.append(cb)
    return layer, shared, f


def _click(layer):
    from napari.utils.interactions import (
        mouse_press_callbacks, mouse_release_callbacks)
    ev = SimpleNamespace(
        type="mouse_press", button=1, position=(2, 3, 3),
        dims_displayed=[1, 2], dims_point=[2, 0, 0], view_direction=None,
        modifiers=(), handled=False)
    mouse_press_callbacks(layer, ev)
    ev.type = "mouse_release"
    mouse_release_callbacks(layer, ev)


def test_single_click_snapshot_excludes_the_press_dab():
    layer, shared, f = _real_layer_with_handler(insert_first=True)
    _click(layer)
    assert (shared[2] == 91).any()          # napari painted the dab
    cb = f._on_paint_stroke_cb
    cb.assert_called_once()
    _, kw = cb.call_args
    assert not kw["old_roi_data"].any(), "snapshot already held the press voxel"


def test_the_order_matters_appended_callback_sees_the_dab():
    """Documents why the callback is inserted first."""
    layer, shared, f = _real_layer_with_handler(insert_first=False)
    _click(layer)
    assert not f._on_paint_stroke_cb.called or \
        f._on_paint_stroke_cb.call_args[1]["old_roi_data"].any()


def test_subviewer_manager_installs_our_callback_first():
    import inspect
    from packages.epicycle_annotator import subviewers
    src = inspect.getsource(subviewers)
    assert "mouse_drag_callbacks.insert(0, drag_callback)" in src


# ---------------------------------------------------------------- MED-2 ----
def test_intent_zero_snapshots_the_label_so_undo_restores_all_of_it():
    vol = np.zeros((8, 8, 8), np.int32)
    vol[2:6, 2:6, 2:6] = 5
    before = vol.copy()
    f = _fake()
    f._active_labels_name = "labels"
    f.viewer.layers = {"labels": SimpleNamespace(data=vol)}
    f._snapshot_label_roi = K._snapshot_label_roi
    roi, snap = K._zero_label_in_volume(f, 5)
    assert not (vol == 5).any()
    vol[roi] = snap
    assert np.array_equal(vol, before)


def test_manual_delete_with_a_snapshot_gets_a_real_entry():
    f = _fake()
    cb = MagicMock()
    f._on_delete_label_cb = cb
    K._handle_manually_deleted_label(f, 5, roi_slices=ROI_I, old_roi_data="s")
    cb.assert_called_once_with(5, roi_slices=ROI_I, old_roi_data="s")


def test_clear_edits_tells_the_controller_to_drop_the_stroke_entries():
    shared = np.zeros((6, 8, 8), np.int32)
    shared[1:3, 1:3, 1:3] = 91
    f = _stroke_fake(shared)
    f._interp_snapshot = None
    f._draw_snapshot = shared.copy()
    f._draw_roi = ROI_I
    f._draw_label_id = 91
    f._handle_manually_deleted_label = MagicMock()
    K._clear_draw_edits(f)
    f._on_paint_stroke_cb.assert_called_once_with(
        91, discard=True, draw_session=f._draw_session)


# ------------------------------------------------------------ MED-3 / 4 ----
def _frag_reject_fake(shared):
    f = _fake()
    f._split_fragment_refine = True
    f._split_running = True
    f._split_label_id = 250
    f._split_roi = (slice(2, 4), slice(2, 4), slice(2, 4))
    f._split_snapshot = shared[f._split_roi].copy()
    f._active_labels_name = "labels"
    f.sync_manager.shared_labels.get.return_value = shared
    f._interp_snapshot = None
    f._interp_refine_mode = False
    f._reset_split_state = types.MethodType(K._reset_split_state, f)
    f._restore_frag_layers = types.MethodType(K._restore_frag_layers, f)
    return f


def test_reject_in_fragment_mode_restores_layers_grown_past_the_split_roi():
    shared = np.zeros((8, 8, 8), np.int32)
    shared[2:4, 2:4, 2:4] = 250
    before = shared.copy()
    f = _frag_reject_fake(shared)
    # fragment entry wrote pieces; a fragment smooth grew past _split_roi
    grown = (slice(1, 6), slice(1, 6), slice(1, 6))
    layer_snap = shared[grown].copy()
    shared[2:4, 2:4, 2:4] = 686
    shared[4:6, 4:6, 4:6] = 686       # outside _split_roi
    f._frag_undo_layers = [(grown, layer_snap)]
    K._reject_split(f)
    assert np.array_equal(shared, before)


def test_restore_reshape_target_restores_the_layers_too():
    shared = np.zeros((8, 8, 8), np.int32)
    shared[2:4, 2:4, 2:4] = 250
    before = shared.copy()
    f = _frag_reject_fake(shared)
    grown = (slice(1, 6), slice(1, 6), slice(1, 6))
    f._frag_undo_layers = [(grown, shared[grown].copy())]
    shared[4:6, 4:6, 4:6] = 686
    shared[2:4, 2:4, 2:4] = 686
    K._restore_reshape_target(f, "test")
    assert np.array_equal(shared, before)


def test_reject_in_fragment_mode_rejects_an_open_interpolation_first():
    shared = np.zeros((8, 8, 8), np.int32)
    f = _frag_reject_fake(shared)
    f._frag_undo_layers = []
    order = []
    f._interp_snapshot = "snap"
    f._reject_interpolation = lambda: order.append("interp")
    f._reset_smooth_state = lambda: order.append("smooth-reset")
    K._reject_split(f)
    assert order[:2] == ["interp", "smooth-reset"]


def test_close_cleanup_rejects_the_interpolation_before_the_split():
    import inspect
    src = inspect.getsource(K._close_cleanup)
    assert src.index("self._reject_interpolation()") < src.index(
        "self._reject_split()")


# ------------------------------------------------------ re-audit items -----
def test_the_alive_check_waits_for_a_stroke_in_flight():
    f = _fake()
    f.drawing_handler = SimpleNamespace(_is_editing=True)
    f._alive_check_deferrals = 0
    f._schedule_label_alive_check = MagicMock()
    f._parse_undo_candidates = MagicMock()
    K._check_deleted_labels(f)
    f._schedule_label_alive_check.assert_called_once()
    f._parse_undo_candidates.assert_not_called()


def test_zero_label_scans_the_volume_once():
    class Counting(np.ndarray):
        eq = 0

        def __eq__(self, other):
            Counting.eq += 1
            return np.ndarray.__eq__(self, other)
    vol = np.zeros((8, 8, 8), np.int32)
    vol[2:6, 2:6, 2:6] = 5
    vol = vol.view(Counting)
    f = _fake()
    f._active_labels_name = "labels"
    f.viewer.layers = {"labels": SimpleNamespace(data=vol)}
    f._snapshot_label_roi = K._snapshot_label_roi
    Counting.eq = 0
    K._zero_label_in_volume(f, 5)
    assert Counting.eq == 1


def test_restore_frag_layers_returns_the_union_roi():
    f = _fake()
    shared = np.zeros((8, 8, 8), np.int32)
    f._split_roi = (slice(2, 4), slice(2, 4), slice(2, 4))
    f._frag_undo_layers = [((slice(1, 6), slice(0, 3), slice(2, 7)),
                            np.zeros((5, 3, 5), np.int32))]
    u = K._restore_frag_layers(f, shared)
    assert u == (slice(1, 6), slice(0, 4), slice(2, 7))
    f._frag_undo_layers = []
    assert K._restore_frag_layers(f, shared) is None


def test_strokes_carry_the_draw_session_token():
    shared = np.zeros((6, 8, 8), np.int32)
    f = _stroke_fake(shared)
    f._draw_session = 4
    f._on_subviewer_stroke_begin("XY", 3)
    shared[3] = 91
    f._on_subviewer_draw_complete()
    assert f._on_paint_stroke_cb.call_args[1]["draw_session"] == 4

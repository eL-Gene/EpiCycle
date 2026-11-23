"""Clear Edits keeps working after an interpolation Preview is rejected.

Reject puts the hand-drawn slices back and keeps them tracked, so the draw
snapshot taken before the first stroke must survive Preview and Reject.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .test_add_click_markers import h  # noqa: F401 - the fixture

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K


class _Btn:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, v):
        self.enabled = bool(v)


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t


class _Lenient(SimpleNamespace):
    """Unset private hooks are quiet no-ops."""

    def __getattr__(self, name):
        if name.startswith(("_refresh", "_set_", "_sync", "_restore",
                            "_end_", "_abandon", "_drop", "_fire",
                            "_update", "_remove", "_lock", "_unlock")):
            return lambda *a, **k: None
        if name.startswith("_"):
            return None            # an unset flag reads falsy
        raise AttributeError(name)


def _widget(volume, *, draw_roi, draw_snapshot, label_id=5):
    f = _Lenient(
        staged_kinds=lambda: frozenset(),
        _draw_roi=draw_roi, _draw_snapshot=draw_snapshot,
        _draw_label_id=label_id,
        _interp_snapshot=None, _interp_roi=None, _interp_label_id=None,
        _interp_refine_mode=False, _split_running=False,
        _smooth_running=False,
        _active_labels_name="lab",
        sync_manager=SimpleNamespace(
            shared_labels={"lab": volume},
            clear_edited_slices_for_label=lambda n, l: None),
        _clear_edits_btn=_Btn(), _accept_interp_btn=_Btn(),
        _reject_interp_btn=_Btn(), _interp_refine_btn=_Btn(),
        _interpolate_btn=_Btn(),
        _interp_status_label=_Label(),
        _refresh_merge_button_state=lambda: None,
        _refresh_label_action_buttons_state=lambda: None,
        _refresh_refine_preview_gate=lambda: None,
        _restore_ready_to_draw_state=lambda: None,
        _sync_all_views_after_operation=lambda **kw: None,
    )
    f._clear_draw_snapshot = lambda: K._clear_draw_snapshot(f)
    return f


def _drawn_scene():
    """Label 5 drawn on slice 2; returns volume, pre-draw copy, widget."""
    vol = np.zeros((6, 6, 6), dtype=np.int32)
    vol[1, 1:3, 1:3] = 5
    roi = (slice(0, 6), slice(0, 6), slice(0, 6))
    before = vol.copy()
    snap = vol[roi].copy()
    vol[3, 2:5, 2:5] = 5  # hand-drawn stroke
    return vol, before, _widget(vol, draw_roi=roi, draw_snapshot=snap)


def _preview_then_reject(f, vol):
    f._interp_roi = (slice(0, 6),) * 3
    f._interp_snapshot = vol.copy()
    f._interp_label_id = 5
    f._clear_edits_btn.setEnabled(False)  # held preview
    vol[2, 2:4, 2:4] = 5  # algorithm fill
    K._reject_interpolation(f)


def test_reject_keeps_draw_snapshot_and_clear_edits_enabled():
    vol, _, f = _drawn_scene()
    _preview_then_reject(f, vol)
    assert f._draw_snapshot is not None
    assert f._draw_roi is not None
    assert f._draw_label_id == 5
    assert f._clear_edits_btn.enabled is True


def test_clear_edits_after_reject_restores_pre_draw_voxels():
    vol, before, f = _drawn_scene()
    _preview_then_reject(f, vol)
    K._clear_draw_edits(f)
    assert np.array_equal(vol, before)
    assert f._draw_snapshot is None
    assert f._clear_edits_btn.enabled is False


def test_clear_edits_is_refused_while_preview_is_held():
    vol, _, f = _drawn_scene()
    f._interp_snapshot = vol.copy()
    K._clear_draw_edits(f)
    assert f._interp_status_label.text == "Use Reject to undo interpolation preview."
    assert f._draw_snapshot is not None


def test_accept_drops_the_draw_snapshot():
    import inspect
    assert "self._clear_draw_snapshot()" in inspect.getsource(
        K._accept_interpolation)
    vol, _, f = _drawn_scene()
    K._clear_draw_snapshot(f)
    assert f._draw_snapshot is None and f._draw_label_id is None
    assert f._clear_edits_btn.enabled is False


def test_reject_without_draw_snapshot_leaves_clear_edits_grey():
    vol, _, f = _drawn_scene()
    f._draw_snapshot = None
    f._draw_roi = None
    _preview_then_reject(f, vol)
    assert f._clear_edits_btn.enabled is False


def test_refine_further_discard_leaves_clear_edits_live(monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as mod

    monkeypatch.setattr(mod, "_refine_clear_records", lambda self: None)
    vol, _, f = _drawn_scene()
    f._interp_refine_mode = True
    f._interp_roi = (slice(0, 6),) * 3
    f._interp_snapshot = vol.copy()
    f._interp_label_id = 5
    f._abandon_refine_worker = lambda: None
    f._drop_refine_walls = lambda: None
    f._fire_refine_lock = lambda on: None
    f._update_smooth_ui = lambda **kw: None
    f._reject_interpolation = lambda: K._reject_interpolation(f)
    K._reject_smooth(f)
    assert f._draw_snapshot is not None
    assert f._clear_edits_btn.enabled is True




def _run_fake(f, vol):
    f._interp_running = False
    f._batch_staged = False
    f._batch_running = False
    f.staged_kinds = lambda: frozenset()
    f._ensure_shared_data_aligned = lambda name: vol
    f._get_selected_label = lambda: 5
    f.sync_manager.get_edited_slices = lambda name: {
        "z": {5: {1, 3}}}
    f._snapshot_label_roi = lambda data, lid, edited: (
        (slice(0, 6),) * 3, data.copy())
    f._lock_subviewers_for_preview = lambda: None
    return f


def test_real_preview_keeps_draw_snapshot_and_greys_clear_edits(monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as mod
    import packages.epicycle_annotator.interpolation as interp

    monkeypatch.setattr(
        interp, "interpolate_label_all_orientations",
        lambda data, lid, edited: 1)
    vol, _, f = _drawn_scene()
    _run_fake(f, vol)
    f._clear_edits_btn.setEnabled(True)
    K._run_interpolation(f)
    assert f._interp_snapshot is not None
    assert f._draw_snapshot is not None and f._draw_label_id == 5
    assert f._clear_edits_btn.enabled is False
    K._reject_interpolation(f)
    assert f._clear_edits_btn.enabled is True


def _smooth_reject_fake(vol, *, refine_mode):
    f = _widget(vol, draw_roi=(slice(0, 6),) * 3,
                draw_snapshot=vol.copy())
    f._interp_refine_mode = refine_mode
    f._smooth_running = False
    f._smooth_snapshot = vol.copy()
    f._smooth_roi = (slice(0, 6),) * 3
    f._interp_label_id = 5
    f._interp_roi = (slice(0, 6),) * 3
    f._interp_snapshot = vol.copy()
    for n in ("_abandon_refine_worker", "_drop_refine_walls"):
        setattr(f, n, lambda *a, **k: None)
    f._fire_refine_lock = lambda on: None
    f._update_smooth_ui = lambda **kw: None
    f._smooth_label_id = 5
    return f


def test_plain_refine_discard_keeps_clear_edits_live(monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as mod

    monkeypatch.setattr(mod, "_refine_clear_records", lambda self: None)
    vol, _, _ = _drawn_scene()
    f = _smooth_reject_fake(vol, refine_mode=False)
    f._smooth_running = True
    f._clear_edits_btn.setEnabled(False)
    K._reject_smooth(f)
    assert f._draw_snapshot is not None
    assert f._clear_edits_btn.enabled is True


def test_discard_of_a_refine_preview_inside_refine_further_keeps_clear_edits(
        monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as mod

    monkeypatch.setattr(mod, "_refine_clear_records", lambda self: None)
    vol, _, _ = _drawn_scene()
    f = _smooth_reject_fake(vol, refine_mode=True)
    f._smooth_running = True
    f._clear_edits_btn.setEnabled(False)
    K._reject_smooth(f)
    assert f._draw_snapshot is not None
    assert f._clear_edits_btn.enabled is True


def test_clear_edits_refuses_while_any_edit_is_staged():
    from napari_epicycle.annotator._staged_op_text import STAGED_WORK_OPEN

    vol, _, f = _drawn_scene()
    f.staged_kinds = lambda: frozenset({"add"})
    before = vol.copy()
    K._clear_draw_edits(f)
    assert f._interp_status_label.text == STAGED_WORK_OPEN
    assert np.array_equal(vol, before) and f._draw_snapshot is not None


def test_finalize_merge_accept_drops_the_draw_snapshot():
    vol, _, f = _drawn_scene()
    f._load_locked = False
    f._merge_dialog = None
    f._on_merge_cb = None
    f._navigator_widget = None
    f._on_isolation_changed = lambda ids, **k: None
    K._finalize_merge_accept(f, 5, [7])
    assert f._draw_snapshot is None and f._draw_label_id is None
    assert f._clear_edits_btn.enabled is False


def _with_draw_snapshot(h):
    h._draw_snapshot = np.zeros((2, 2, 2), np.int32)
    h._draw_roi = (slice(0, 2),) * 3
    h._draw_label_id = 7
    h._clear_edits_btn = _Btn()
    h._refresh_merge_button_state = lambda: None
    h._refresh_label_action_buttons_state = lambda: None
    h._clear_draw_snapshot = lambda: K._clear_draw_snapshot(h)
    return h


def test_a_refused_add_start_keeps_the_draw_snapshot(h):
    _with_draw_snapshot(h)
    h._add_running = False
    h.staged_kinds = lambda: frozenset({"smooth"})
    h._add_refused_while_staged = lambda door: True
    h._start_add()
    assert h._draw_snapshot is not None


def test_add_start_then_cancel_keeps_the_draw_snapshot(h):
    _with_draw_snapshot(h)
    h._add_running = False
    h._start_add()
    assert h._add_running
    h._add_preview_result = None
    h._add_secondary()                       # Cancel
    assert not h._add_running
    assert h._draw_snapshot is not None and h._draw_label_id == 7


def test_add_accept_drops_the_draw_snapshot(h):
    _with_draw_snapshot(h)
    res = np.zeros((16, 26, 26), np.int32)
    res[8:12, 15:19, 15:19] = 40
    h._add_preview_result = res
    h._add_preview_roi = tuple(h._add_roi)
    h.adv.checked = [40]
    h._on_add_committed_cb = None
    h._accept_add()
    assert h._add_review
    assert h._draw_snapshot is None and h._draw_label_id is None


def test_patch_switch_greys_clear_edits_through_the_shared_helper():
    import inspect
    assert "self._clear_draw_snapshot()" in inspect.getsource(
        K._load_selected_layers)


def test_clear_edits_on_a_new_label_removes_its_row_once():
    vol = np.zeros((6, 6, 6), dtype=np.int32)
    roi = (slice(0, 6),) * 3
    snap = vol[roi].copy()
    vol[3, 2:5, 2:5] = 5          # the label's only voxels are strokes
    f = _widget(vol, draw_roi=roi, draw_snapshot=snap)
    healed = []
    f._handle_manually_deleted_label = lambda lid: healed.append(lid)
    K._clear_draw_edits(f)
    assert not (vol == 5).any()
    assert healed == [5]


def test_clear_edits_on_an_existing_cell_does_not_remove_its_row():
    vol, _, f = _drawn_scene()    # label 5 existed before the stroke
    healed = []
    f._handle_manually_deleted_label = lambda lid: healed.append(lid)
    K._clear_draw_edits(f)
    assert (vol == 5).any()
    assert healed == []

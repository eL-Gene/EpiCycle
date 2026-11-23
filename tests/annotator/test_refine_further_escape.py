"""Refine Further has a way out before any Preview: the Refine tab's
secondary button reads Discard and puts the pre-interpolation volume back."""
from __future__ import annotations

from types import SimpleNamespace

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K


class _Btn:
    def setEnabled(self, v):
        pass


def test_entering_refine_further_arms_discard_with_a_plain_preview_word():
    pushed = []
    f = SimpleNamespace(
        _interp_label_id=4, _interp_refine_mode=False,
        _interpolate_btn=_Btn(), _accept_interp_btn=_Btn(),
        _reject_interp_btn=_Btn(), _interp_refine_btn=_Btn(),
        _interp_status_label=SimpleNamespace(setText=lambda t: None),
        _navigator_widget=None, _isolation_active=False,
        _refresh_merge_button_state=lambda: None,
        _refresh_label_action_buttons_state=lambda: None,
        _set_selected_label_all=lambda lid: None,
        _on_isolation_changed=lambda ids, **k: None,
        _iso_begin=lambda kind, target=None: None,
        _restore_ready_to_draw_state=lambda: None,
        _open_advanced_features=lambda: None,
        _get_advanced_features_widget=lambda: SimpleNamespace(
            focus_tab=lambda name: True),
        _update_smooth_ui=lambda **kw: pushed.append(kw),
    )
    K._enter_interp_refine_mode(f)
    assert {"reject_enabled": True, "preview_text": "Preview"}.items() <= {
        k: v for kw in pushed for k, v in kw.items()}.items()


def _reject_fake(*, refine_mode, smooth_running):
    calls = []
    f = SimpleNamespace(
        _interp_refine_mode=refine_mode, _smooth_running=smooth_running,
        _abandon_refine_worker=lambda: None, _drop_refine_walls=lambda: None,
        _fire_refine_lock=lambda on: None,
        _reject_interpolation=lambda: calls.append("reject_interp"),
        _update_smooth_ui=lambda **kw: calls.append(("ui", kw)),
        calls=calls,
    )
    return f


def test_discard_before_any_preview_routes_to_reject_interpolation(monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as mod

    monkeypatch.setattr(mod, "_refine_clear_records", lambda self: None)
    f = _reject_fake(refine_mode=True, smooth_running=False)
    K._reject_smooth(f)
    assert f.calls[0] == "reject_interp"
    assert ("ui", {"reject_enabled": False, "accept_enabled": False}) in f.calls


def test_discard_with_nothing_staged_and_no_refine_further_still_returns(monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as mod

    monkeypatch.setattr(mod, "_refine_clear_records", lambda self: None)
    f = _reject_fake(refine_mode=False, smooth_running=False)
    K._reject_smooth(f)
    assert f.calls == []


class _Rec(_Btn):
    def __init__(self):
        self.enabled = None

    def setEnabled(self, v):
        self.enabled = bool(v)


def _real_reject_fake(refine_mode):
    import numpy as np

    vol = np.ones((4, 4, 4), np.int32)
    f = SimpleNamespace(
        _active_labels_name="L",
        sync_manager=SimpleNamespace(shared_labels={"L": vol}),
        _interp_snapshot=np.zeros((2, 2, 2), np.int32),
        _interp_roi=(slice(0, 2),) * 3, _interp_label_id=3,
        _interp_refine_mode=refine_mode,
        _interpolate_btn=_Rec(), _accept_interp_btn=_Rec(),
        _reject_interp_btn=_Rec(), _interp_refine_btn=_Rec(),
        _interp_status_label=SimpleNamespace(setText=lambda t: None),
        _sync_all_views_after_operation=lambda **k: None,
        _refresh_merge_button_state=lambda: None,
        _refresh_label_action_buttons_state=lambda: None,
        _clear_draw_snapshot=lambda: None,
        _draw_snapshot=None, _clear_edits_btn=_Rec(),
        _restore_ready_to_draw_state=lambda: None,
        gate=[],
    )
    f._refresh_refine_preview_gate = lambda: f.gate.append(1)
    return f, vol


def test_real_discard_from_refine_further_gives_interpolate_back():
    f, vol = _real_reject_fake(refine_mode=True)
    K._reject_interpolation(f)
    assert f._interpolate_btn.enabled is True
    # every Reject re-arms the gate once; leaving Refine Further re-arms it
    # again after Interpolate is handed back
    assert f.gate == [1, 1]
    assert f._interp_refine_mode is False and f._interp_snapshot is None
    assert int(vol[:2, :2, :2].sum()) == 0      # the volume went back


def test_a_plain_reject_leaves_the_interpolate_button_alone():
    f, _ = _real_reject_fake(refine_mode=False)
    K._reject_interpolation(f)
    assert f._interpolate_btn.enabled is None and f.gate == [1]


def _ui_fake(refine_mode, smooth=False):
    seen = []
    f = SimpleNamespace(
        _interp_refine_mode=refine_mode, _smooth_running=smooth,
        _get_advanced_features_widget=lambda: SimpleNamespace(
            update_smooth_ui_state=lambda **kw: seen.append(kw)),
        _interp_status_label=SimpleNamespace(setText=lambda t: None),
        seen=seen)
    return f


def test_a_step_that_ends_with_nothing_staged_keeps_discard_in_refine_further():
    f = _ui_fake(True)
    K._update_smooth_ui(f, reject_enabled=False, accept_enabled=False)
    assert f.seen == [{"reject_enabled": True, "accept_enabled": False,
                       "preview_text": "Preview"}]


def test_discard_is_not_forced_while_busy_staged_or_outside_refine_further():
    for f, kw in ((_ui_fake(True), {"reject_enabled": False, "busy": True}),
                  (_ui_fake(True, smooth=True), {"reject_enabled": False}),
                  (_ui_fake(False), {"reject_enabled": False})):
        K._update_smooth_ui(f, **kw)
        assert f.seen == [kw]

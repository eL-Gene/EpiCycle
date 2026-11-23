"""A staged (not running) interpolation preview gates the label-action buttons."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
from napari_epicycle.annotator.refinement_state import RefinementOpState


class _Nav:
    def __init__(self):
        self.enabled = None
        self.merge = None

    def set_label_action_buttons_enabled(self, v):
        self.enabled = bool(v)

    def set_merge_enabled(self, v):
        self.merge = bool(v)


def _fake(**flags):
    f = SimpleNamespace(
        _refine_state=RefinementOpState(),
        _navigator_widget=_Nav(),
        _interp_running=False,
        _interp_snapshot=None,
        _interp_refine_mode=False,
    )
    f.__dict__.update(flags)
    # bind the real methods to the fake
    f._sync_refine_state = lambda: Curation3DAnnotator._sync_refine_state(f)
    return f


def _refresh(f):
    Curation3DAnnotator._refresh_label_action_buttons_state(f)
    return f._navigator_widget.enabled


def test_preview_then_staged_then_patch_switch():
    f = _fake()
    assert _refresh(f) is True                      # idle

    f._interp_running = True                        # Preview computing
    assert _refresh(f) is False

    f._interp_running = False                       # result staged, nothing running
    f._interp_snapshot = object()
    assert _refresh(f) is False
    assert f._refine_state.any_op_active

    f._interp_snapshot = None                       # the patch switch clears it
    assert _refresh(f) is True


def test_refine_mode_alone_counts_as_active():
    f = _fake(_interp_refine_mode=True)
    assert _refresh(f) is False


def test_patch_switch_refreshes_buttons_after_clearing_snapshots():
    src = inspect.getsource(Curation3DAnnotator._load_selected_layers)
    clear = src.index("self._interp_snapshot = None")
    for call in ("self._refresh_merge_button_state()",
                 "self._refresh_label_action_buttons_state()"):
        assert src.index(call) > clear, (
            f"{call} must run after the staged-preview state is cleared")


class _Btn:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, v):
        self.enabled = bool(v)


class _Label:
    text = ""

    def setText(self, t):
        self.text = t


def test_reject_with_the_volume_missing_frees_the_buttons():
    f = _fake(_interp_refine_mode=True, _interp_snapshot=object())
    f._active_labels_name = "labels"
    f.sync_manager = SimpleNamespace(shared_labels={})
    f._interp_status_label = _Label()
    f._interp_roi = (slice(0, 1),)
    f._interp_label_id = 3
    f._accept_interp_btn = _Btn()
    f._reject_interp_btn = _Btn()
    f._interp_refine_btn = _Btn()
    f._interpolate_btn = _Btn()
    # Reject re-arms Clear Edits from the draw snapshot, so the fake carries both
    f._clear_edits_btn = _Btn()
    f._draw_snapshot = None
    seen = []

    def refresh_label_buttons():
        Curation3DAnnotator._refresh_label_action_buttons_state(f)
        seen.append(f._navigator_widget.enabled)

    f._refresh_label_action_buttons_state = refresh_label_buttons
    f._refresh_merge_button_state = lambda: None
    Curation3DAnnotator._reject_interpolation(f)

    assert f._interp_refine_mode is False
    assert not f._refine_state.any_op_active
    assert seen and seen[-1] is True

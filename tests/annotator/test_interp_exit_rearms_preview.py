"""An interpolation Accept / Reject hands Refine's Preview button back.

Refine Further and an open interpolation grey the Preview button.  The
Refine accept / discard exits re-run the Preview gate; the interpolation
Accept and Reject did not, so Preview stayed grey until the dock was
relaunched.  The chained Preview that selects another object also now says
why it refused.  Qt-free: the annotator is a mock with the method under test
bound to it.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from napari_epicycle.annotator.curation_3d_widget import (
    Curation3DAnnotator as K)


def _fake():
    w = MagicMock()
    w._load_locked = False
    w._interp_label_id = 7
    w._interp_roi = None
    w._interp_snapshot = None
    w._interp_refine_mode = False
    w._split_fragment_refine = False
    w._on_interpolation_accepted_cb = None
    w._is_auto_advance_enabled.return_value = False
    w._draw_snapshot = None
    w.sync_manager.shared_labels.get.return_value = object()
    return w


def test_accept_interpolation_rearms_preview():
    w = _fake()
    K._accept_interpolation(w)
    assert w._refresh_refine_preview_gate.call_count == 1


def test_reject_interpolation_rearms_preview():
    w = _fake()
    K._reject_interpolation(w)
    assert w._refresh_refine_preview_gate.call_count == 1


def test_chain_refusal_says_why(capsys):
    w = _fake()
    w._split_running = False
    w._batch_staged = False
    w._batch_running = False
    w._refine_running = False
    w._refine_sync_busy = False
    w._smooth_running = True
    w._smooth_label_id = 5
    w._interp_snapshot = None
    w.staged_kinds.return_value = []
    w._ensure_shared_data_aligned.return_value = object()
    w._get_advanced_features_widget.return_value = None
    w._layer_by_name.return_value = MagicMock(selected_label=9)
    import napari_epicycle.annotator.curation_3d_widget as mod
    orig = mod.Labels
    mod.Labels = MagicMock  # the fake layer stands in for a Labels layer
    try:
        K._preview_smooth(w)
    finally:
        mod.Labels = orig
    out = capsys.readouterr().out
    assert "[REFINE] chain refused - selected 9 != staged 5" in out

"""Refine Further isolates the object; Accept or Discard puts the isolation
back the way it was before Refine Further (off stays off, on stays on, a
group of isolated cells comes back as the same group).  With auto-advance the
next cell is isolated instead."""
from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K
from napari_epicycle.annotator.refinement_state import StagedOp

_BOUND = ("_iso_owner_alive", "_iso_bracket_open", "_iso_take", "_iso_drop",
          "_iso_begin", "_iso_suspend", "_iso_end", "_iso_live_ids",
          "_reshape_pieces_cap")


def _fake(*, iso_was_on, nav=True, auto=False, multi=None):
    """*multi* is a set of cell ids isolated before Refine Further."""
    f = MagicMock()
    f._interp_label_id = 3
    f._interp_refine_mode = False
    f._is_auto_advance_enabled = lambda: auto
    f._iso_bracket = None
    f._ISO_OWNER_KINDS = K._ISO_OWNER_KINDS
    f.staged_kinds = lambda: frozenset({StagedOp.INTERPOLATE, StagedOp.REFINE})
    f._get_advanced_features_widget = lambda: None
    f._active_labels_name = "labels"
    f._layer_by_name = lambda layers, name: SimpleNamespace(
        visible=False, selected_label=3)
    f._multi_label_proxy_set = frozenset(multi) if multi else None
    f._isolation_label_set = None
    f._isolation_proxy_active = iso_was_on
    f._isolation_proxy_label_id = 3 if iso_was_on and not multi else None
    f._isolation_active = iso_was_on
    f._pre_isolation_colormap = None
    f._colormap_authority = None
    for n in _BOUND:
        setattr(f, n, types.MethodType(getattr(K, n), f))
    if nav:
        f._navigator_widget = MagicMock()
        f._navigator_widget.is_isolate_checked.return_value = iso_was_on
    else:
        f._navigator_widget = None
    return f


def _in_refine(f):
    """Enter Refine Further, then stand where Accept / Discard find the
    widget: a staged refine result in interp-refine mode."""
    K._enter_interp_refine_mode(f)
    f._on_isolation_changed.reset_mock()
    f._navigator_widget and f._navigator_widget.set_isolate_checked.reset_mock()
    f._load_locked = False
    f._smooth_running = True
    f._smooth_label_id = 3
    f._smooth_roi = None
    f._smooth_snapshot = None
    f._interp_roi = None
    f._interp_snapshot = None
    f._interp_label_id = 3
    f._active_labels_name = "labels"
    f.sync_manager.shared_labels.get.return_value = None
    f._refine_steps = []
    f._refine_provenance = None
    return f


@pytest.fixture(autouse=True)
def _plain_provenance(monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as W
    monkeypatch.setattr(W, "_refine_provenance", lambda self: ([], []))
    monkeypatch.setattr(W, "_refine_clear_records", lambda self: None)


def _forced(f):
    """Refine Further isolated the cell (what its entry did to the view)."""
    f._isolation_active = True
    f._isolation_proxy_active = True
    f._isolation_proxy_label_id = 3


def test_entering_refine_records_that_isolation_was_off():
    f = _fake(iso_was_on=False)
    K._enter_interp_refine_mode(f)
    snap = f._iso_bracket.snapshot
    assert (snap.mode, snap.checkbox) == ("off", False)
    assert f._iso_bracket.kind == "interp" and f._iso_bracket.target == 3
    f._on_isolation_changed.assert_called_once_with({3})


def test_entering_refine_records_that_isolation_was_on():
    f = _fake(iso_was_on=True)
    K._enter_interp_refine_mode(f)
    snap = f._iso_bracket.snapshot
    assert (snap.mode, snap.ids, snap.checkbox) == ("single", {3}, True)


def test_the_bracket_opens_before_the_forced_isolation():
    f = _fake(iso_was_on=False)
    order = []
    f._on_isolation_changed = lambda ids, **k: order.append(
        ("OI", f._iso_bracket is not None))
    K._enter_interp_refine_mode(f)
    assert order == [("OI", True)]


def test_entering_refine_without_a_navigator_still_records():
    f = _fake(iso_was_on=True, nav=False)
    K._enter_interp_refine_mode(f)
    assert f._iso_bracket.snapshot.mode == "single"
    assert f._iso_bracket.snapshot.checkbox is False


@pytest.mark.parametrize("op", ["_accept_smooth", "_reject_smooth"])
def test_isolation_that_was_off_is_cleared_on_exit(op):
    f = _in_refine(_fake(iso_was_on=False))
    _forced(f)
    getattr(K, op)(f)
    f._on_isolation_changed.assert_called_once_with(set(), reason="forget")
    f._navigator_widget.set_isolate_checked.assert_called_once_with(False)
    assert f._iso_bracket is None


@pytest.mark.parametrize("op", ["_accept_smooth", "_reject_smooth"])
def test_isolation_that_was_on_comes_back_at_exit(op):
    f = _in_refine(_fake(iso_was_on=True))
    getattr(K, op)(f)
    f._on_isolation_changed.assert_called_once_with({3})
    f._navigator_widget.set_isolate_checked.assert_called_once_with(True)
    assert f._iso_bracket is None


@pytest.mark.parametrize("op", ["_accept_smooth", "_reject_smooth"])
def test_a_group_isolation_comes_back_as_the_same_group(op):
    f = _in_refine(_fake(iso_was_on=True, multi={3, 8, 9}))
    getattr(K, op)(f)
    f._on_isolation_changed.assert_called_once_with({3, 8, 9})


def test_accept_with_auto_advance_drops_without_restoring():
    f = _in_refine(_fake(iso_was_on=True, auto=True))

    def _next():
        # The next row's own isolation: a different cell.
        f._isolation_proxy_label_id = 9
        return True

    f._navigator_widget.select_next_triage.side_effect = _next
    K._accept_smooth(f)
    f._on_isolation_changed.assert_not_called()
    f._navigator_widget.set_isolate_checked.assert_not_called()
    f._navigator_widget.select_next_triage.assert_called_once()
    assert f._iso_bracket is None


def test_accept_with_auto_advance_and_the_box_off_before_still_restores():
    f = _in_refine(_fake(iso_was_on=False, auto=True))
    _forced(f)
    K._accept_smooth(f)
    # The clear is a "forget" (the plot must not echo it and deselect the
    # row the advance selects), and it is the only isolation call.
    assert f._on_isolation_changed.call_args_list == [
        call(set(), reason="forget")]
    f._navigator_widget.set_isolate_checked.assert_called_once_with(False)
    f._navigator_widget.select_next_triage.assert_called_once()


def test_accept_with_auto_advance_and_no_next_row_restores_exactly():
    f = _in_refine(_fake(iso_was_on=True, auto=True))
    f._navigator_widget.select_next_triage.return_value = False
    K._accept_smooth(f)
    # One isolation call: the snapshot back, not an extra clear.
    assert f._on_isolation_changed.call_args_list == [call({3})]
    f._navigator_widget.set_isolate_checked.assert_called_once_with(True)
    assert f._iso_bracket is None


def test_accept_with_auto_advance_and_a_next_row_makes_no_call():
    f = _in_refine(_fake(iso_was_on=True, auto=True))

    def _next():
        # The next row's own isolation: a different cell.
        f._isolation_proxy_label_id = 9
        return True

    f._navigator_widget.select_next_triage.side_effect = _next
    K._accept_smooth(f)
    f._on_isolation_changed.assert_not_called()
    assert f._iso_bracket is None


def test_accept_restores_when_the_next_row_isolates_nothing_new():
    # Refine Further forced the refined cell into view; the advance selected
    # a row but left that isolation exactly as it was.  The view must go back
    # to the snapshot, not stay on the refined cell.
    f = _in_refine(_fake(iso_was_on=True, auto=True, multi={3, 8, 9}))
    _forced(f)
    f._navigator_widget.select_next_triage.return_value = True
    K._accept_smooth(f)
    assert f._on_isolation_changed.call_args_list == [call({3, 8, 9})]
    assert f._iso_bracket is None


def test_entering_refine_with_a_deleted_checkbox_records_the_box_as_off():
    f = _fake(iso_was_on=True)
    f._navigator_widget.is_isolate_checked.side_effect = RuntimeError(
        "wrapped C/C++ object has been deleted")
    K._enter_interp_refine_mode(f)
    assert f._iso_bracket.snapshot.checkbox is False
    # ...and the box-off branch of Accept restores the isolation it saw.
    f = _in_refine(f)
    f._navigator_widget.is_isolate_checked.side_effect = None
    K._accept_smooth(f)
    f._on_isolation_changed.assert_called_with({3})
    f._navigator_widget.set_isolate_checked.assert_called_once_with(False)


def test_discard_never_advances():
    f = _in_refine(_fake(iso_was_on=True, auto=True))
    K._reject_smooth(f)
    f._on_isolation_changed.assert_called_once_with({3})
    f._navigator_widget.select_next_triage.assert_not_called()


def _in_reject_interpolation(iso_was_on):
    f = _in_refine(_fake(iso_was_on=iso_was_on))
    f._interp_refine_mode = True
    f._interp_snapshot = None
    f._interp_roi = None
    f.sync_manager.shared_labels.get.return_value = MagicMock()
    return f


def test_reject_from_refine_further_puts_isolation_that_was_on_back():
    f = _in_reject_interpolation(True)
    K._reject_interpolation(f)
    f._on_isolation_changed.assert_called_once_with({3})
    f._navigator_widget.set_isolate_checked.assert_called_once_with(True)
    assert f._iso_bracket is None


def test_reject_from_refine_further_clears_isolation_that_was_off():
    f = _in_reject_interpolation(False)
    _forced(f)
    K._reject_interpolation(f)
    f._on_isolation_changed.assert_called_once_with(set(), reason="forget")
    f._navigator_widget.set_isolate_checked.assert_called_once_with(False)


def test_reject_with_the_volume_missing_drops_without_restoring():
    f = _in_reject_interpolation(True)
    f.sync_manager.shared_labels.get.return_value = None
    K._reject_interpolation(f)
    f._on_isolation_changed.assert_not_called()
    assert f._iso_bracket is None


def test_a_plain_interpolation_reject_leaves_isolation_alone():
    f = _fake(iso_was_on=True)
    f._interp_refine_mode = False
    f._interp_snapshot = None
    f._interp_roi = None
    f.sync_manager.shared_labels.get.return_value = MagicMock()
    K._reject_interpolation(f)
    f._on_isolation_changed.assert_not_called()

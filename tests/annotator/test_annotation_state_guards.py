"""Tests for annotation state guards — uncommitted work detection + discard + navigation guards.

Guards prevent users from accidentally navigating away from in-progress
drawing, interpolation, split, or smooth work.  The popup dialog (WP-6)
is Qt-dependent and verified manually; these tests mock it.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from packages.epicycle_annotator.sync import LabelSyncManager


# ---------------------------------------------------------------------------
# Lightweight fakes (no Qt / napari required)
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def disconnect(self, cb):
        try:
            self._callbacks.remove(cb)
        except ValueError:
            pass


class _FakeSyncManager:
    """Minimal sync manager for guard tests."""

    def __init__(self):
        self.shared_labels: dict[str, np.ndarray] = {}
        self._edited: dict[str, dict] = {}

    def get_edited_slices(self, layer_name: str) -> dict:
        return self._edited.setdefault(layer_name, {"XY": {}, "XZ": {}, "YZ": {}})

    def clear_edited_slices_for_label(self, layer_name: str, label_id: int) -> None:
        edited = self._edited.get(layer_name)
        if edited is None:
            return
        for orient_dict in edited.values():
            orient_dict.pop(label_id, None)
            for key in list(orient_dict.keys()):
                if int(key) == label_id:
                    orient_dict.pop(key, None)


def _make_guard_test_annotator(**overrides):
    """Build a SimpleNamespace annotator with controllable state flags.

    Returns an object that has all the state attributes checked by
    ``_has_uncommitted_work`` and ``_discard_uncommitted_work``, plus
    a fake sync_manager.  The real guard methods from Curation3DAnnotator
    are bound via ``types.MethodType``.

    ``overrides`` let tests set specific flags (e.g. ``_smooth_running=True``).
    """
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
        OperationSlot,
    )

    sync = _FakeSyncManager()

    ann = types.SimpleNamespace(
        # Identity
        sync_manager=sync,
        _active_labels_name="labels",

        # Interpolation state
        _interp_snapshot=None,
        _interp_roi=None,
        _interp_label_id=None,
        _interp_running=False,
        _interp_refine_mode=False,

        # Split state
        _split_snapshot=None,
        _split_roi=None,
        _split_label_id=None,
        _split_running=False,
        _split_previewed=False,
        _split_fragment_refine=False,
        _split_fragment_ids=[],
        _split_seed_ids=[],
        _split_isolate_was_on=False,
        _split_main_visible=True,
        _refine_pre_preview=None,
        _detected_peaks=None,
        _auto_detect_active=False,

        # Smooth state
        _smooth_running=False,
        _smooth_snapshot=None,
        _smooth_roi=None,
        _smooth_label_id=None,
        _smooth_chain_count=0,

        # Navigator
        _navigator_widget=None,
        _last_navigated_label_id=None,

        # Selected label (for _get_selected_label stub)
        _selected_label=1,

        # Triage callbacks
        _on_mark_good_cb=None,
        _on_delete_label_cb=None,

        # Tracking for test assertions
        _reject_interpolation_called=False,
        _reject_split_called=False,
        _reject_smooth_called=False,
        _set_selected_label_all_called_with=None,
        _close_cleanup_called=False,
        _close_subviewers_called=False,
    )

    # Default _get_selected_label stub
    ann._get_selected_label = lambda: ann._selected_label

    # Default reject stubs (tests may override)
    ann._reject_smooth = lambda: setattr(ann, "_reject_smooth_called", True)
    ann._reject_split = lambda: setattr(ann, "_reject_split_called", True)
    ann._reject_interpolation = lambda: setattr(ann, "_reject_interpolation_called", True)

    # Default dialog stub — returns True (Discard).  Tests override as needed.
    ann._confirm_discard_dialog = lambda reason: True

    # Build operation slots (mirrors _register_operation_slots)
    ann._operation_slots = [
        OperationSlot(
            name="smooth",
            is_active=lambda: ann._smooth_running,
            description=lambda: (
                f"Smooth preview active for label {ann._smooth_label_id}."
            ),
            reject=lambda: ann._reject_smooth(),
            priority=30,
        ),
        OperationSlot(
            name="split",
            is_active=lambda: ann._split_running,
            description=lambda: (
                f"Split fragment refine active for label {ann._split_label_id}."
                if ann._split_fragment_refine
                else f"Split in progress for label {ann._split_label_id}."
            ),
            reject=lambda: ann._reject_split(),
            priority=20,
        ),
        OperationSlot(
            name="interpolation",
            is_active=lambda: (
                ann._interp_snapshot is not None
                or ann._interp_refine_mode
            ),
            description=lambda: (
                f"Interpolation preview active for label {ann._interp_label_id}."
            ),
            reject=lambda: ann._reject_interpolation(),
            priority=10,
        ),
    ]

    # Bind the real guard methods from the class
    for method_name in (
        "_has_uncommitted_work",
        "_discard_uncommitted_work",
        "_next_label_id_after_accept",
    ):
        method = getattr(Curation3DAnnotator, method_name, None)
        if method is not None:
            setattr(ann, method_name, types.MethodType(method, ann))

    # Apply overrides
    for k, v in overrides.items():
        setattr(ann, k, v)

    return ann


# ---------------------------------------------------------------------------
# WP-1: _has_uncommitted_work tests
# ---------------------------------------------------------------------------


class TestHasUncommittedWork:
    """Verify _has_uncommitted_work detects all in-progress states."""

    def _call(self, ann):
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        return Curation3DAnnotator._has_uncommitted_work(ann)

    def test_idle_returns_false(self):
        ann = _make_guard_test_annotator()
        has_work, reason = self._call(ann)
        assert not has_work
        assert reason == ""

    def test_smooth_running_detected(self):
        ann = _make_guard_test_annotator(_smooth_running=True, _smooth_label_id=5)
        has_work, reason = self._call(ann)
        assert has_work
        assert "smooth" in reason.lower()

    def test_split_running_detected(self):
        ann = _make_guard_test_annotator(_split_running=True, _split_label_id=3)
        has_work, reason = self._call(ann)
        assert has_work
        assert "split" in reason.lower()

    def test_split_fragment_refine_detected(self):
        ann = _make_guard_test_annotator(
            _split_running=True, _split_fragment_refine=True, _split_label_id=3
        )
        has_work, reason = self._call(ann)
        assert has_work
        assert "split" in reason.lower()

    def test_interp_snapshot_detected(self):
        ann = _make_guard_test_annotator(
            _interp_snapshot=np.zeros((5, 5, 5)),
            _interp_label_id=2,
        )
        has_work, reason = self._call(ann)
        assert has_work
        assert "interpolation" in reason.lower()

    def test_interp_refine_mode_detected(self):
        ann = _make_guard_test_annotator(_interp_refine_mode=True, _interp_label_id=7)
        has_work, reason = self._call(ann)
        assert has_work
        assert "refine" in reason.lower() or "interpolation" in reason.lower()

    def test_edited_slices_for_current_label_detected(self):
        ann = _make_guard_test_annotator()
        ann.sync_manager._edited["labels"] = {"XY": {1: [0, 3]}, "XZ": {}, "YZ": {}}
        ann._selected_label = 1
        ann._get_selected_label = lambda: 1
        has_work, reason = self._call(ann)
        assert has_work
        assert "edit" in reason.lower() or "unsaved" in reason.lower()

    def test_edited_slices_with_numpy_key(self):
        ann = _make_guard_test_annotator()
        ann.sync_manager._edited["labels"] = {
            "XY": {np.int64(1): [0, 3]}, "XZ": {}, "YZ": {}
        }
        ann._selected_label = 1
        ann._get_selected_label = lambda: 1
        has_work, reason = self._call(ann)
        assert has_work

    def test_edited_slices_for_different_label_returns_false(self):
        ann = _make_guard_test_annotator()
        ann.sync_manager._edited["labels"] = {"XY": {99: [0]}, "XZ": {}, "YZ": {}}
        ann._selected_label = 1
        ann._get_selected_label = lambda: 1
        has_work, reason = self._call(ann)
        assert not has_work

    def test_priority_smooth_over_split(self):
        """Smooth is checked first (highest priority)."""
        ann = _make_guard_test_annotator(
            _smooth_running=True, _smooth_label_id=5,
            _split_running=True, _split_label_id=3,
        )
        has_work, reason = self._call(ann)
        assert has_work
        assert "smooth" in reason.lower()


# ---------------------------------------------------------------------------
# WP-2: _discard_uncommitted_work tests
# ---------------------------------------------------------------------------


class TestDiscardUncommittedWork:
    """Verify _discard_uncommitted_work calls the right reject methods."""

    def _call(self, ann):
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        Curation3DAnnotator._discard_uncommitted_work(ann)

    def test_discard_smooth(self):
        ann = _make_guard_test_annotator(_smooth_running=True, _smooth_label_id=5)
        ann._reject_smooth = lambda: setattr(ann, "_reject_smooth_called", True)
        self._call(ann)
        assert ann._reject_smooth_called

    def test_discard_split(self):
        ann = _make_guard_test_annotator(_split_running=True, _split_label_id=3)
        ann._reject_split = lambda: setattr(ann, "_reject_split_called", True)
        self._call(ann)
        assert ann._reject_split_called

    def test_discard_interp_snapshot(self):
        ann = _make_guard_test_annotator(
            _interp_snapshot=np.zeros((5, 5, 5)),
            _interp_label_id=2,
        )
        ann._reject_interpolation = lambda: setattr(ann, "_reject_interpolation_called", True)
        self._call(ann)
        assert ann._reject_interpolation_called

    def test_discard_interp_refine_mode(self):
        """Interp refine mode calls _reject_smooth (which does full undo)."""
        ann = _make_guard_test_annotator(
            _interp_refine_mode=True, _interp_label_id=7,
            _smooth_running=True, _smooth_label_id=7,
        )
        ann._reject_smooth = lambda: setattr(ann, "_reject_smooth_called", True)
        self._call(ann)
        assert ann._reject_smooth_called

    def test_discard_edited_slices(self):
        ann = _make_guard_test_annotator()
        ann.sync_manager._edited["labels"] = {"XY": {1: [0, 3]}, "XZ": {}, "YZ": {}}
        ann._selected_label = 1
        ann._get_selected_label = lambda: 1
        self._call(ann)
        edited = ann.sync_manager.get_edited_slices("labels")
        # Edited slices should be cleared for label 1
        assert 1 not in edited["XY"]

    def test_discard_noop_when_idle(self):
        ann = _make_guard_test_annotator()
        ann._reject_smooth = lambda: setattr(ann, "_reject_smooth_called", True)
        ann._reject_split = lambda: setattr(ann, "_reject_split_called", True)
        ann._reject_interpolation = lambda: setattr(ann, "_reject_interpolation_called", True)
        self._call(ann)
        assert not ann._reject_smooth_called
        assert not ann._reject_split_called
        assert not ann._reject_interpolation_called

    def test_discard_smooth_then_split(self):
        """Both smooth and split active — both should be rejected."""
        ann = _make_guard_test_annotator(
            _smooth_running=True, _smooth_label_id=5,
            _split_running=True, _split_label_id=3,
        )
        smooth_called = []
        split_called = []
        ann._reject_smooth = lambda: smooth_called.append(True)
        ann._reject_split = lambda: split_called.append(True)
        self._call(ann)
        assert len(smooth_called) == 1
        assert len(split_called) == 1

    def test_discard_interp_then_clears_edited_slices(self):
        ann = _make_guard_test_annotator(
            _interp_snapshot=np.zeros((5, 5, 5)),
            _interp_label_id=2,
        )
        ann.sync_manager._edited["labels"] = {"XY": {2: [0, 3]}, "XZ": {}, "YZ": {}}
        ann._selected_label = 2
        ann._get_selected_label = lambda: 2
        ann._reject_interpolation = lambda: setattr(ann, "_reject_interpolation_called", True)
        self._call(ann)
        assert ann._reject_interpolation_called


# ---------------------------------------------------------------------------
# WP-4: Navigation guard tests — silent blocking (no dialog)
# ---------------------------------------------------------------------------


class TestNavigationGuard:
    """Verify _on_navigator_select_label silently blocks during active ops.

    Navigation guards use silent return (no dialog popup). This preserves
    baseline usability — the user stays in the current operation and must
    accept/reject before navigating.
    """

    def _make_navigable(self, **overrides):
        """Build a fake annotator with navigation-related methods."""
        ann = _make_guard_test_annotator(**overrides)
        ann._selected_label = 1
        ann._get_selected_label = lambda: ann._selected_label

        # Stub navigation methods
        ann._set_selected_label_all_calls = []
        ann._set_selected_label_all = lambda lid: ann._set_selected_label_all_calls.append(lid)

        ann._check_label_exists_in_volume = lambda lid: True
        ann._handle_manually_deleted_label = lambda lid: None
        ann._clear_draw_snapshot = lambda: None

        ann._on_navigator_label_selected_cb = None

        return ann

    def _call_navigate(self, ann, target_label_id):
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        Curation3DAnnotator._on_navigator_select_label(ann, target_label_id)

    def test_idle_navigates_freely(self):
        ann = self._make_navigable()
        self._call_navigate(ann, 5)
        assert 5 in ann._set_selected_label_all_calls

    def test_drawing_does_not_block_navigation(self):
        """Edited slices (drawing) should NOT block navigation — drawing is
        normal workflow input for interpolation, not uncommitted work."""
        ann = self._make_navigable()
        ann.sync_manager._edited["labels"] = {"XY": {1: [0, 3]}, "XZ": {}, "YZ": {}}
        self._call_navigate(ann, 5)
        assert 5 in ann._set_selected_label_all_calls

    def test_split_normal_silently_blocks(self):
        """During active split, navigation is silently blocked (no dialog)."""
        ann = self._make_navigable(
            _split_running=True,
            _split_label_id=3,
        )
        self._call_navigate(ann, 5)
        assert 5 not in ann._set_selected_label_all_calls

    def test_split_fragment_refine_allows_navigation(self):
        """Fragment refine exempts navigation — user selects fragments."""
        ann = self._make_navigable(
            _split_running=True,
            _split_fragment_refine=True,
            _split_label_id=3,
        )
        self._call_navigate(ann, 5)
        assert 5 in ann._set_selected_label_all_calls

    def test_smooth_silently_blocks(self):
        """During active smooth, navigation is silently blocked."""
        ann = self._make_navigable(
            _smooth_running=True,
            _smooth_label_id=5,
        )
        self._call_navigate(ann, 10)
        assert 10 not in ann._set_selected_label_all_calls

    def test_interp_preview_silently_blocks(self):
        """During active interpolation preview, navigation is silently blocked."""
        ann = self._make_navigable(
            _interp_snapshot=np.zeros((5, 5, 5)),
            _interp_label_id=1,
        )
        self._call_navigate(ann, 5)
        assert 5 not in ann._set_selected_label_all_calls

    def test_interp_refine_silently_blocks(self):
        """During interp refine mode, navigation is silently blocked."""
        ann = self._make_navigable(
            _interp_refine_mode=True,
            _interp_label_id=7,
        )
        self._call_navigate(ann, 5)
        assert 5 not in ann._set_selected_label_all_calls


# ---------------------------------------------------------------------------
# WP-5: Delete, Mark Good, Close guards
# ---------------------------------------------------------------------------


class TestDeleteGuard:
    """Verify _on_delete_from_navigator blocks only during split (baseline)."""

    def _make_deletable(self, **overrides):
        ann = _make_guard_test_annotator(**overrides)
        ann._selected_label = 1

        # Stub the actual delete logic — viewer.layers must support `in`
        class _EmptyLayers:
            def __contains__(self, name):
                return False
        ann.labels_combo = types.SimpleNamespace(currentText=lambda: "labels")
        ann.viewer = types.SimpleNamespace(layers=_EmptyLayers())
        ann._snapshot_label_roi = lambda data, lid: (None, None)
        ann._refresh_label_data_in_subviewers = lambda: None
        ann._on_delete_label_cb = None

        return ann

    def _call_delete(self, ann, label_id):
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        Curation3DAnnotator._on_delete_from_navigator(ann, label_id)

    def test_idle_deletes_freely(self):
        ann = self._make_deletable()
        self._call_delete(ann, 5)

    def test_split_blocks_delete(self):
        """Delete is blocked during active split (baseline behavior)."""
        ann = self._make_deletable(_split_running=True, _split_label_id=3)
        # Should silently return — no crash, no dialog
        self._call_delete(ann, 5)

    def test_smooth_does_not_block_delete(self):
        """Smooth does NOT block delete (baseline behavior)."""
        ann = self._make_deletable(_smooth_running=True, _smooth_label_id=5)
        # Should proceed to delete (no crash)
        self._call_delete(ann, 5)


class TestMarkGoodGuard:
    """Verify _on_mark_good_from_navigator has no guard (baseline).

    Mark-good is always allowed — it's a triage action that naturally
    concludes the curation workflow for a label.
    """

    def _make_markable(self, **overrides):
        ann = _make_guard_test_annotator(**overrides)
        ann._mark_good_calls = []
        ann._on_mark_good_cb = lambda lid: ann._mark_good_calls.append(lid)
        return ann

    def _call_mark_good(self, ann, label_id):
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        Curation3DAnnotator._on_mark_good_from_navigator(ann, label_id)

    def test_idle_marks_freely(self):
        ann = self._make_markable()
        self._call_mark_good(ann, 5)
        assert 5 in ann._mark_good_calls

    def test_mark_good_during_smooth(self):
        """Mark-good is allowed even during smooth (no guard)."""
        ann = self._make_markable(_smooth_running=True, _smooth_label_id=5)
        self._call_mark_good(ann, 5)
        assert 5 in ann._mark_good_calls

    def test_mark_good_during_split(self):
        """Mark-good is allowed even during split (no guard)."""
        ann = self._make_markable(_split_running=True, _split_label_id=3)
        self._call_mark_good(ann, 5)
        assert 5 in ann._mark_good_calls


# ---------------------------------------------------------------------------
# WP-7: Navigator-based next label ID after accept
# ---------------------------------------------------------------------------


class TestNextLabelIdSelection:
    """Verify _next_label_id_after_accept picks from navigator or volume."""

    def _call(self, ann):
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        return Curation3DAnnotator._next_label_id_after_accept(ann)

    def test_uses_navigator_items(self):
        ann = _make_guard_test_annotator()
        nav = types.SimpleNamespace(
            _items=[
                {"label_id": 1}, {"label_id": 5}, {"label_id": 10},
            ]
        )
        ann._navigator_widget = nav
        result = self._call(ann)
        assert result == 11  # max(1, 5, 10) + 1

    def test_falls_back_to_volume(self):
        ann = _make_guard_test_annotator()
        ann._navigator_widget = None
        volume = np.zeros((10, 10, 10), dtype=np.int32)
        volume[0, 0, 0] = 8
        ann.sync_manager.shared_labels["labels"] = volume
        result = self._call(ann)
        assert result == 9  # max(volume) + 1

    def test_navigator_empty_falls_back_to_volume(self):
        ann = _make_guard_test_annotator()
        nav = types.SimpleNamespace(_items=[])
        ann._navigator_widget = nav
        volume = np.zeros((10, 10, 10), dtype=np.int32)
        volume[2, 3, 4] = 12
        ann.sync_manager.shared_labels["labels"] = volume
        result = self._call(ann)
        assert result == 13

    def test_no_navigator_no_volume_returns_fallback(self):
        ann = _make_guard_test_annotator()
        ann._navigator_widget = None
        result = self._call(ann)
        assert result >= 1

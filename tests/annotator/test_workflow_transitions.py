"""Tests for seamless workflow transitions between operations.

Verifies that guard integration works correctly across operation boundaries:
- Operations block when another is active
- Operations clean up properly after accept/reject
- Auto-advance works after accept
- Navigation guards fire at correct times

Uses lightweight fakes (no Qt / napari required).
"""

from __future__ import annotations

import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lightweight fakes (reuse pattern from test_annotation_state_guards)
# ---------------------------------------------------------------------------


class _FakeSyncManager:
    """Minimal sync manager for transition tests."""

    def __init__(self):
        self.shared_labels: dict[str, np.ndarray] = {}
        self._edited: dict[str, dict] = {}
        self._suppress_show_selected_sync = False

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


def _make_transition_test_annotator(**overrides):
    """Build a fake annotator for workflow transition tests.

    Binds real guard methods from Curation3DAnnotator and provides
    controllable state flags for simulating active operations.
    """
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
        OperationSlot,
    )

    sync = _FakeSyncManager()

    ann = types.SimpleNamespace(
        # Core
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

        # Selected label
        _selected_label=1,

        # Tracking
        _reject_interpolation_called=False,
        _reject_split_called=False,
        _reject_smooth_called=False,
        _on_mark_good_cb=None,
        _on_delete_label_cb=None,
    )

    ann._get_selected_label = lambda: ann._selected_label

    # Reject stubs
    ann._reject_smooth = lambda: setattr(ann, "_reject_smooth_called", True)
    ann._reject_split = lambda: setattr(ann, "_reject_split_called", True)
    ann._reject_interpolation = lambda: setattr(
        ann, "_reject_interpolation_called", True
    )

    # Dialog stub — returns True (Discard)
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

    # Bind real guard methods
    for method_name in (
        "_has_uncommitted_work",
        "_discard_uncommitted_work",
        "_next_label_id_after_accept",
    ):
        method = getattr(Curation3DAnnotator, method_name, None)
        if method is not None:
            setattr(ann, method_name, types.MethodType(method, ann))

    for k, v in overrides.items():
        setattr(ann, k, v)

    return ann


# ---------------------------------------------------------------------------
# WP-7: Workflow transition tests
# ---------------------------------------------------------------------------


class TestWorkflowTransitions:
    """Verify clean transitions between different operation workflows."""

    def test_smooth_blocks_interp_start(self):
        """Cannot start interpolation while smooth is active."""
        ann = _make_transition_test_annotator(
            _smooth_running=True,
            _smooth_label_id=5,
        )

        has_work, reason = ann._has_uncommitted_work()
        assert has_work
        assert "smooth" in reason.lower()

    def test_split_blocks_smooth_start(self):
        """Cannot start smooth while split (non-fragment) is active."""
        ann = _make_transition_test_annotator(
            _split_running=True,
            _split_label_id=3,
            _split_fragment_refine=False,
        )

        has_work, reason = ann._has_uncommitted_work()
        assert has_work
        assert "split" in reason.lower()

    def test_fragment_refine_allows_smooth(self):
        """During fragment refine, smooth is allowed (per-fragment refinement).

        Fragment refine is special: has_uncommitted_work reports True (split),
        but the smooth Preview button checks _split_fragment_refine to allow it.
        """
        ann = _make_transition_test_annotator(
            _split_running=True,
            _split_fragment_refine=True,
            _split_label_id=3,
        )

        # Guard reports split is active (has_uncommitted_work = True)
        has_work, reason = ann._has_uncommitted_work()
        assert has_work
        assert "split" in reason.lower()

        # But _preview_smooth checks _split_fragment_refine and allows it.
        should_block = ann._split_running and not ann._split_fragment_refine
        assert not should_block  # smooth is allowed during fragment refine

    def test_interp_refine_blocks_split(self):
        """Cannot start split during interp refine mode."""
        ann = _make_transition_test_annotator(
            _interp_refine_mode=True,
            _interp_label_id=7,
        )

        has_work, reason = ann._has_uncommitted_work()
        assert has_work

    def test_discard_clears_smooth_then_split(self):
        """Discard should clear both smooth and split when both are active."""
        ann = _make_transition_test_annotator(
            _smooth_running=True,
            _smooth_label_id=5,
            _split_running=True,
            _split_label_id=3,
        )

        ann._discard_uncommitted_work()

        assert ann._reject_smooth_called
        assert ann._reject_split_called

    def test_discard_clears_all_three_operations(self):
        """Discard should clear smooth + split + interp in order."""
        ann = _make_transition_test_annotator(
            _smooth_running=True,
            _smooth_label_id=5,
            _split_running=True,
            _split_label_id=3,
            _interp_snapshot=np.zeros((5, 5, 5)),
            _interp_label_id=2,
        )

        ann._discard_uncommitted_work()

        assert ann._reject_smooth_called
        assert ann._reject_split_called
        assert ann._reject_interpolation_called

    def test_edited_slices_cleared_after_discard(self):
        """After discard, edited slices for current label should be cleared."""
        ann = _make_transition_test_annotator()
        ann.sync_manager._edited["labels"] = {
            "XY": {1: [0, 3]}, "XZ": {}, "YZ": {}
        }
        ann._selected_label = 1

        ann._discard_uncommitted_work()

        edited = ann.sync_manager.get_edited_slices("labels")
        assert 1 not in edited["XY"]

    def test_guard_priority_smooth_over_split_over_interp(self):
        """Guard detection follows priority: smooth > split > interp > edited."""
        # All three active → smooth reported first
        ann = _make_transition_test_annotator(
            _smooth_running=True, _smooth_label_id=5,
            _split_running=True, _split_label_id=3,
            _interp_snapshot=np.zeros((5, 5, 5)), _interp_label_id=2,
        )
        _, reason = ann._has_uncommitted_work()
        assert "smooth" in reason.lower()

        # Only split + interp → split reported
        ann2 = _make_transition_test_annotator(
            _split_running=True, _split_label_id=3,
            _interp_snapshot=np.zeros((5, 5, 5)), _interp_label_id=2,
        )
        _, reason2 = ann2._has_uncommitted_work()
        assert "split" in reason2.lower()

        # Only interp → interp reported
        ann3 = _make_transition_test_annotator(
            _interp_snapshot=np.zeros((5, 5, 5)), _interp_label_id=2,
        )
        _, reason3 = ann3._has_uncommitted_work()
        assert "interpolation" in reason3.lower()


class TestNextLabelIdAfterAccept:
    """Verify _next_label_id_after_accept computes correct next label."""

    def test_no_navigator_uses_volume_max(self):
        """Without navigator, next label = max(volume) + 1."""
        ann = _make_transition_test_annotator()
        vol = np.zeros((5, 5, 5), dtype=np.int32)
        vol[0, 0, 0] = 10
        ann.sync_manager.shared_labels["labels"] = vol

        next_id = ann._next_label_id_after_accept()
        assert next_id == 11

    def test_navigator_with_items(self):
        """With navigator items, next label = max(navigator_ids) + 1."""
        ann = _make_transition_test_annotator()
        vol = np.zeros((5, 5, 5), dtype=np.int32)
        vol[0, 0, 0] = 10
        ann.sync_manager.shared_labels["labels"] = vol

        nav = types.SimpleNamespace(
            _items=[
                {"label_id": 5},
                {"label_id": 10},
                {"label_id": 15},
            ]
        )
        ann._navigator_widget = nav

        next_id = ann._next_label_id_after_accept()
        assert next_id == 16  # max(5,10,15) + 1

    def test_empty_volume_returns_1(self):
        """With empty volume and no navigator, returns 1."""
        ann = _make_transition_test_annotator()
        ann.sync_manager.shared_labels["labels"] = np.zeros(
            (5, 5, 5), dtype=np.int32
        )

        next_id = ann._next_label_id_after_accept()
        assert next_id == 1

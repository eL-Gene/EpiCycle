"""Tests for the curation undo stack.

Pure Python — no Qt or napari required.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.curation_undo import (
    CurationUndoEntry,
    CurationUndoStack,
)


# ---------------------------------------------------------------------------
# CurationUndoEntry construction
# ---------------------------------------------------------------------------


class TestCurationUndoEntry:
    def test_minimal_entry(self):
        entry = CurationUndoEntry(action_type="delete", label_id=42)
        assert entry.action_type == "delete"
        assert entry.label_id == 42
        assert entry.roi_slices is None
        assert entry.old_mask_roi is None
        assert entry.old_df_snapshot is None
        assert entry.description == ""
        assert entry.extra == {}

    def test_full_entry(self):
        roi = (slice(0, 10), slice(5, 15), slice(2, 8))
        mask_data = np.ones((10, 10, 6), dtype=np.uint32)
        entry = CurationUndoEntry(
            action_type="split",
            label_id=7,
            roi_slices=roi,
            old_mask_roi=mask_data,
            old_df_snapshot={"label": [7], "area": [100]},
            description="Split label 7",
            extra={"new_ids": [8, 9]},
        )
        assert entry.action_type == "split"
        assert entry.label_id == 7
        assert entry.roi_slices == roi
        assert np.array_equal(entry.old_mask_roi, mask_data)
        assert entry.extra["new_ids"] == [8, 9]

    def test_mark_good_no_mask(self):
        entry = CurationUndoEntry(
            action_type="mark_good",
            label_id=3,
            description="Mark label 3 as good",
        )
        assert entry.roi_slices is None
        assert entry.old_mask_roi is None

    def test_extra_defaults_to_empty_dict(self):
        """Each entry gets its own dict (no shared mutable default)."""
        e1 = CurationUndoEntry(action_type="delete", label_id=1)
        e2 = CurationUndoEntry(action_type="delete", label_id=2)
        e1.extra["key"] = "val"
        assert "key" not in e2.extra


# ---------------------------------------------------------------------------
# CurationUndoStack
# ---------------------------------------------------------------------------


class TestCurationUndoStack:
    @staticmethod
    def _entry(label_id: int, action: str = "delete") -> CurationUndoEntry:
        return CurationUndoEntry(
            action_type=action,
            label_id=label_id,
            description=f"{action} label {label_id}",
        )

    def test_empty_stack(self):
        stack = CurationUndoStack()
        assert len(stack) == 0
        assert not stack
        assert stack.pop() is None
        assert stack.peek() is None

    def test_push_pop(self):
        stack = CurationUndoStack()
        e1 = self._entry(1)
        e2 = self._entry(2)
        stack.push(e1)
        stack.push(e2)
        assert len(stack) == 2
        assert stack

        popped = stack.pop()
        assert popped is e2
        assert len(stack) == 1

        popped = stack.pop()
        assert popped is e1
        assert len(stack) == 0

    def test_peek_does_not_remove(self):
        stack = CurationUndoStack()
        e1 = self._entry(1)
        stack.push(e1)
        assert stack.peek() is e1
        assert len(stack) == 1

    def test_clear(self):
        stack = CurationUndoStack()
        for i in range(5):
            stack.push(self._entry(i))
        assert len(stack) == 5
        stack.clear()
        assert len(stack) == 0
        assert not stack

    def test_max_depth_eviction(self):
        stack = CurationUndoStack(max_depth=3)
        for i in range(5):
            stack.push(self._entry(i))
        # Only the last 3 should remain
        assert len(stack) == 3
        assert stack.pop().label_id == 4
        assert stack.pop().label_id == 3
        assert stack.pop().label_id == 2
        assert stack.pop() is None

    def test_max_depth_one(self):
        stack = CurationUndoStack(max_depth=1)
        stack.push(self._entry(10))
        stack.push(self._entry(20))
        assert len(stack) == 1
        assert stack.pop().label_id == 20

    def test_max_depth_property(self):
        stack = CurationUndoStack(max_depth=15)
        assert stack.max_depth == 15

    def test_push_pop_with_mask_data(self):
        """Verify numpy arrays survive push/pop without corruption."""
        stack = CurationUndoStack()
        roi = (slice(0, 5), slice(0, 5), slice(0, 5))
        data = np.arange(125, dtype=np.uint32).reshape(5, 5, 5)
        entry = CurationUndoEntry(
            action_type="delete",
            label_id=1,
            roi_slices=roi,
            old_mask_roi=data.copy(),
        )
        stack.push(entry)
        popped = stack.pop()
        assert np.array_equal(popped.old_mask_roi, data)

    def test_lifo_order(self):
        """Stack is strictly LIFO."""
        stack = CurationUndoStack(max_depth=10)
        ids = [10, 20, 30, 40, 50]
        for lid in ids:
            stack.push(self._entry(lid))
        result = []
        while stack:
            result.append(stack.pop().label_id)
        assert result == [50, 40, 30, 20, 10]

    def test_mixed_action_types(self):
        stack = CurationUndoStack()
        stack.push(self._entry(1, "delete"))
        stack.push(self._entry(2, "split"))
        stack.push(self._entry(3, "mark_good"))
        stack.push(self._entry(4, "interpolation"))
        stack.push(self._entry(5, "smooth"))
        stack.push(self._entry(6, "merge"))
        assert len(stack) == 6
        assert stack.pop().action_type == "merge"
        assert stack.pop().action_type == "smooth"

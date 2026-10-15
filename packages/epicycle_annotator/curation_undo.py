"""Undo stack for 3D curation actions.

Pure Python — no Qt or napari dependencies.  Used by the fusion
reconstruction controller to support undo of delete, split, merge,
interpolation, smooth, and mark-good actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class CurationUndoEntry:
    """Snapshot of state before a single curation action."""

    action_type: str
    """One of: delete, split, merge, interpolation, smooth, mark_good, discard_garbage."""

    label_id: int
    """Primary label ID affected by the action."""

    roi_slices: Optional[tuple] = None
    """(slice, slice, slice) bounding the affected mask region.  None for
    actions that don't modify the mask (e.g. mark_good)."""

    old_mask_roi: Optional[Any] = None
    """Copy of ``volume[roi_slices]`` *before* the action.  ``None`` when the
    mask was not changed (e.g. mark_good)."""

    old_df_snapshot: Optional[Any] = None
    """Copy of the full QC features DataFrame *before* the action."""

    description: str = ""
    """Human-readable summary shown in the log (e.g. 'Delete label 42')."""

    extra: dict = field(default_factory=dict)
    """Arbitrary per-action metadata (e.g. ``new_ids`` for split)."""


class CurationUndoStack:
    """Fixed-depth undo stack for curation actions.

    Parameters
    ----------
    max_depth : int
        Maximum number of undo entries kept.  When exceeded the oldest
        entry is silently discarded.
    """

    def __init__(self, max_depth: int = 20) -> None:
        self._stack: List[CurationUndoEntry] = []
        self._max_depth = max_depth

    # -- mutators ----------------------------------------------------------

    def push(self, entry: CurationUndoEntry) -> None:
        """Record a new undo entry, evicting the oldest if at capacity."""
        if len(self._stack) >= self._max_depth:
            self._stack.pop(0)
        self._stack.append(entry)

    def pop(self) -> Optional[CurationUndoEntry]:
        """Remove and return the most recent entry, or ``None`` if empty."""
        return self._stack.pop() if self._stack else None

    def clear(self) -> None:
        """Discard all entries."""
        self._stack.clear()

    # -- queries -----------------------------------------------------------

    def peek(self) -> Optional[CurationUndoEntry]:
        """Return the most recent entry without removing it."""
        return self._stack[-1] if self._stack else None

    def __len__(self) -> int:
        return len(self._stack)

    def __bool__(self) -> bool:
        return bool(self._stack)

    @property
    def max_depth(self) -> int:
        return self._max_depth

"""Centralized refinement state for the 3D curation annotator.

Mirrors the existing per-op flags on `Curation3DAnnotator` (`_interp_running`,
`_split_running`, `_smooth_running`, etc.) into a single typed dataclass
so cross-op queries become named predicates instead of boolean expressions
duplicated across files.

Phase 1 (this file): dataclass + named predicates only. No existing flag
is removed; the annotator keeps a `RefinementOpState` instance and a
`_sync_refine_state()` adapter that mirrors current flags into it on
demand. External callers (e.g. the merge dialog gate) read predicates
through the dataclass.

Phase 2 (later): per-op flags migrate from instance attributes on the
annotator into fields on this dataclass. The `_sync_refine_state()`
adapter goes away. All cross-op call sites read/write directly on
`self._refine_state.<field>`.

Chaining is preserved by design: a smooth re-entry on the same label
reuses its snapshot, smooth is allowed inside the interpolation-refine
and split fragment-refine sub-modes, and interpolation is allowed inside
split's fragment-refine sub-mode (see the predicates below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RefinementOpState:
    """Centralized refinement state.

    All fields default to a clean / idle state. Fields are grouped by
    op family (Interp / Split / Smooth) so additions for new families
    follow the same shape.
    """

    # --- Interpolation family ---
    interp_active: bool = False
    interp_refine_mode: bool = False
    interp_snapshot: Any = None
    interp_label_id: int | None = None
    interp_roi: tuple | None = None

    # --- Split family ---
    split_active: bool = False
    split_previewed: bool = False
    split_fragment_refine: bool = False
    split_snapshot: Any = None
    split_label_id: int | None = None
    split_roi: tuple | None = None
    split_fragment_ids: list = field(default_factory=list)

    # --- Smooth / Refine family (umbrella for SDF Smooth, MorphGAC,
    #     MorphACWE, Dilate/Erode, Guided Grow, Fill Holes, Flood Fill) ---
    smooth_active: bool = False
    smooth_chain_count: int = 0
    smooth_snapshot: Any = None
    smooth_label_id: int | None = None
    smooth_roi: tuple | None = None

    # --- Batch Refine family (Step 3 batch tool — applies an ordered
    #     recipe of geometry-only ops across many labels at once).  Gated
    #     like a full refinement op: while a batch is staged (dialog open
    #     / preview applied) no other op may start and no destructive
    #     label action / merge / patch-switch may run. ---
    batch_refine_active: bool = False

    # --- Drawing / merge (mirrors annotator-side flags that are NOT
    #     refinement-op state but DO gate destructive label-action
    #     buttons like Delete Label + + New Label). ---
    draw_snapshot_present: bool = False
    merge_dialog_open: bool = False

    # --- Predicates: cross-op gating ---

    @property
    def any_op_active(self) -> bool:
        """True if any refinement op is currently in a staged state.

        Includes ``batch_refine_active`` (2026-06-05) so the existing
        cross-op gates — merge dialog (``can_open_merge_dialog``),
        destructive label actions (``can_start_destructive_label_op``),
        and the widget-side navigation gates that read this property —
        automatically refuse while a batch is staged.
        """
        return (
            self.interp_active
            or self.split_active
            or self.smooth_active
            or self.batch_refine_active
        )

    def can_start_interp(self) -> bool:
        """Encodes the existing rule at curation_3d_widget.py:3101-3104.

        Interp is blocked while smooth is running, and while split is
        running (unless we're inside the fragment-refine sub-mode).
        Also blocked while a batch refine is staged.
        """
        if self.batch_refine_active:
            return False
        if self.smooth_active:
            return False
        if self.split_active and not self.split_fragment_refine:
            return False
        return True

    def can_start_split(self) -> bool:
        """Encodes the existing rule at curation_3d_widget.py:3432-3436.

        Split requires a clean state — no interp, no smooth, no other
        split, and no staged batch refine.
        """
        return not (
            self.smooth_active
            or self.split_active
            or self.interp_active
            or self.batch_refine_active
        )

    def can_start_smooth(self, label_id: int) -> bool:
        """Encodes the existing rule at curation_3d_widget.py:4438-4490.

        Smooth chains on itself (same label only). Allowed during
        interp_refine_mode and split_fragment_refine sub-modes.
        Blocked by an active split unless we're in fragment-refine, and
        blocked while a batch refine is staged.
        """
        if self.batch_refine_active:
            return False
        if self.split_active and not self.split_fragment_refine:
            return False
        if self.smooth_active and self.smooth_label_id != label_id:
            return False
        return True

    def is_smooth_chaining(self, label_id: int) -> bool:
        """True iff a smooth re-entry on this label should reuse the
        existing snapshot rather than taking a new one.

        See curation_3d_widget.py:4473-4490 for the chain-preservation
        pattern this predicate guards.
        """
        return self.smooth_active and self.smooth_label_id == label_id

    def can_open_merge_dialog(self) -> bool:
        """The merge dialog requires a clean state (no staged ops).

        Flag-only triage actions (Mark Good / Mark Review / Mark Bad /
        Mark Garbage / Mark Curated) are NOT gated by this — they only
        update the qc_tier column and don't touch voxels.  However the
        restored destructive Delete Label (2026-06-01) DOES gate — see
        ``can_start_destructive_label_op`` for that contract.
        """
        return not self.any_op_active

    def can_start_destructive_label_op(self) -> bool:
        """True iff Delete Label / + New Label / similar destructive
        label-action buttons should be ENABLED.

        Destructive label-actions zero or replace voxels.  They MUST
        be gated against:
        - Any staged refinement op (interp / split / smooth) — concurrent
          volume mutation would corrupt the staged ROI snapshot.
        - Uncommitted draw strokes (``draw_snapshot_present``) — the
          user is mid-paint and a destructive op would race with the
          pending stroke-commit flow.
        - An open merge dialog (``merge_dialog_open``) — merge has its
          own staged snapshot.

        Added 2026-06-01 for the restored Delete Label button (see
        ``_on_delete_label_from_navigator`` on Curation3DAnnotator).
        The prior fec23be-era assumption that "triage actions don't
        conflict with refinement staging" no longer holds for
        destructive Delete — see the updated docstring on
        ``can_open_merge_dialog`` above.
        """
        if self.any_op_active:
            return False
        if self.draw_snapshot_present:
            return False
        if self.merge_dialog_open:
            return False
        return True

    def can_start_batch_refine(self) -> bool:
        """True iff the Batch Refine dialog may be opened.

        Batch refine mutates many labels at once over a working copy of
        the volume, so it requires a fully clean state — the same
        contract as ``can_start_destructive_label_op``:
        - No staged refinement op (interp / split / smooth) AND no batch
          already active (both covered by ``any_op_active``, which now
          includes ``batch_refine_active``).
        - No uncommitted draw stroke (``draw_snapshot_present``).
        - No open merge dialog (``merge_dialog_open``).
        """
        if self.any_op_active:
            return False
        if self.draw_snapshot_present:
            return False
        if self.merge_dialog_open:
            return False
        return True

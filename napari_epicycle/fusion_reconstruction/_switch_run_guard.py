"""Step-3 run-switch safety guard (Phase 3a, 2026-06-27).

When the user switches Step 3 runs (the future run-picker UI), several
live-UI side-effects must be handled or warned about BEFORE the switch
orchestrator (``controller._switch_active_run``) tears anything down:

* A long-running Step-3 operation is in flight (predict / reconstruct /
  refine / extract / commit-triage / save / batch-refine / extract-all /
  QC-extract / async patch-load / curate busy) — **HARD BLOCK**.  These
  workers read + write run-scoped artifacts on background threads; a switch
  mid-op would re-point the run context out from under them, corrupting the
  OLD run's outputs or writing them under the NEW run's dirs.  The same 11
  flags ``_refresh_step3_lock`` OR's into the surface lock are the block set.

* The 3D curation annotator dock is open — **WARN + CONFIRM**.  The
  annotator is scoped to the current run's patch; switching closes it and
  reopens on the new run's patch.  Unsaved curation / tier edits are
  guarded separately by the orchestrator's ``_guard_unsaved_changes``.

This module is PURE (it only reads controller attributes via ``getattr``)
and NEVER raises — a bare / partially-initialised controller yields a
defensive verdict rather than a crash.  Mirrors the Step 2 analogue
``hitl_training_annotation/_switch_run_guard.py``.  The Qt blocked / confirm
dialogs live in the controller (Phase 3a uses a minimal themed message;
the polished run-picker dialog is Phase 4).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


# The 11 HARD-BLOCK flags.  This is exactly the OR-set ``_refresh_step3_lock``
# uses for the surface lock (the 10 unified flags + the special-cased
# ``_qc_extract_running`` that the lock handles separately at its tail).  Any
# one of these being truthy means a Step-3 worker is live → BLOCK the switch.
STEP3_BLOCK_FLAGS: Tuple[str, ...] = (
    "_curation_busy",
    "_patch_load_in_flight",
    "_prediction_running",
    "_reconstruction_running",
    "_refinement_running",
    "_extraction_running",
    "_commit_triage_in_flight",
    "_save_in_flight",
    "_batch_refine_running",
    "_extract_all_running",
    "_qc_extract_running",
    # 2026-09-09 (Brick R1 audit): the two MODE flags.  A run switch runs
    # `_guard_unsaved_changes()` BEFORE the annotator close that would put the
    # cell back, so a user who answers "Save" mid-split writes a volume with a
    # HOLE in it to disk.  `_add_mode_running` had the same gap since 2026-06-30
    # (a staged de-novo Add is lost, though nothing is corrupted); both are
    # hard-blocked here rather than left to a confirmation dialog.
    "_reshape_running",
    "_add_mode_running",
    # 2026-09-22 (Brick B1): the Refine tab's worker + staged window.  A
    # staged refine preview is already written into the shared volume, and
    # a run switch runs `_guard_unsaved_changes()` BEFORE the annotator
    # close that would put it back -- so a user who answers "Save" mid-
    # refine writes an unaccepted shape to disk (REFINE).
    "_refine_running",
)

# Human-readable reason per flag (headline shown in the blocked dialog).
_FLAG_REASONS = {
    "_curation_busy": "A curation operation is in progress.",
    "_patch_load_in_flight": "A patch is still loading.",
    "_prediction_running": "A prediction run is in progress.",
    "_reconstruction_running": "A reconstruction run is in progress.",
    "_refinement_running": "Model refinement (training) is in progress.",
    "_extraction_running": "A training-pair extraction is in progress.",
    "_commit_triage_in_flight": "A Commit-Triage save is writing to disk.",
    "_refine_running": (
        "A refine step is open. The object you are refining has an "
        "unaccepted shape in the volume. Accept or Discard it first."
    ),
    "_reshape_running": (
        "A guided split is open. The cell you are splitting is not in the "
        "volume until you accept or cancel it."
    ),
    "_add_mode_running": "A de-novo Add session is open.",
    "_save_in_flight": "A curation save is writing to disk.",
    "_batch_refine_running": "A Batch Refine operation is in progress.",
    "_extract_all_running": "An 'Extract All' QC batch run is in progress.",
    "_qc_extract_running": "A QC feature extraction is in progress.",
}


class SwitchSafety(str, Enum):
    """Outcome of the pre-switch safety check."""

    SAFE = "safe"
    """No in-flight work; proceed without a prompt."""

    NEEDS_CONFIRMATION = "needs_confirmation"
    """Annotator open — reversible; warn + confirm (and run the unsaved guard)."""

    BLOCKED = "blocked"
    """A Step-3 worker is live — hard block; cannot switch safely."""


def check_switch_safety(controller, viewer) -> Tuple[SwitchSafety, List[str]]:
    """Classify whether a Step-3 run switch is safe right now.

    Args:
        controller: the Step-3 ``ReconstructionController`` (or any object
            exposing the block flags + ``_get_open_annotator_widget``).
        viewer: the napari viewer (unused today; kept for signature parity
            with the Step 2 analogue + future viewer-layer inspection).

    Returns:
        ``(SwitchSafety, reasons)`` where ``reasons`` is an ordered list of
        human-readable strings (most-severe first; use the first as the
        dialog headline).  Pure + defensive — never raises.
    """
    # 1. Any long-running Step-3 op → HARD BLOCK.
    blockers: List[str] = []
    for flag in STEP3_BLOCK_FLAGS:
        try:
            if getattr(controller, flag, False):
                blockers.append(_FLAG_REASONS.get(flag, f"{flag} is active."))
        except Exception:
            # A property/attr that raises is treated as "indeterminate" →
            # err on the side of blocking rather than silently switching.
            blockers.append(f"{flag} state is indeterminate; cannot switch.")
    if blockers:
        return SwitchSafety.BLOCKED, blockers

    # 2. 3D curation annotator dock open → WARN + CONFIRM.
    try:
        getter = getattr(controller, "_get_open_annotator_widget", None)
        if callable(getter) and getter():
            return (
                SwitchSafety.NEEDS_CONFIRMATION,
                [
                    "The 3D Curation Annotator is open on the current run's "
                    "patch.  Switching will close it and reopen on the new "
                    "run's last-used patch.  Unsaved edits are checked "
                    "separately before the switch proceeds."
                ],
            )
    except Exception:
        # If the detector itself fails, do NOT block — fall through to SAFE;
        # the orchestrator's own unsaved-changes guard still runs.
        pass

    # 3. Nothing in flight, annotator closed → SAFE.
    return SwitchSafety.SAFE, []


__all__ = [
    "SwitchSafety",
    "check_switch_safety",
    "STEP3_BLOCK_FLAGS",
]

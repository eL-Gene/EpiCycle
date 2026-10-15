"""Switch-run safety guard (Phase 8a polish, 2026-05-23).

When the user switches Step 2 runs via the Curation header dropdown,
several side-effects on the live UI need to be handled or warned about:

* Training is in flight (QThread running) — **HARD BLOCK**.  Mid-write
  state corruption + a stale model file getting written under the
  WRONG run's models/ dir is unacceptable.
* 2D annotator dock is open — **WARN + CONFIRM**.  The annotator is
  scoped to the previous run's patch; switching mid-edit silently
  would orphan unsaved strokes.  User confirms; we close the dock on
  proceed.
* Main napari viewer has stale patch layers (image / labels /
  predictions) — **CLEAR on switch**.  The new run starts fresh; the
  old run's image stack is no longer the right context.

User-reported bug 2026-05-23: after a run switch, the napari viewer
kept the old patch loaded + the dropdown's switch operation succeeded
silently — so the user saw labels for default's patch while editing
in CP-membrane's context.  This module catches that.

Pure functions for testability; Qt dialog lives in
``_switch_run_guard_dialog.py`` to keep this module Qt-free.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Tuple

from napari_epicycle.hitl_training_annotation.ui._step2_text import (
    CANCEL_BTN_TEXT,
)


class SwitchSafety(str, Enum):
    """Outcome of pre-switch safety check."""

    SAFE = "safe"
    """No in-flight work; proceed without prompt."""

    NEEDS_CONFIRMATION = "needs_confirmation"
    """Annotator open or some other reversible state — warn user."""

    BLOCKED = "blocked"
    """Training thread running — hard block, can't switch safely."""


def check_switch_safety(controller, viewer) -> Tuple[SwitchSafety, List[str]]:
    """Inspect the controller + viewer; classify whether a run switch is safe.

    Args:
        controller: TrainingAnnotationController instance.
        viewer: napari viewer instance (may be None in tests).

    Returns:
        Tuple of (SwitchSafety enum, list of human-readable reason strings).
        Reasons are ordered most-severe-first; use the first one as the
        headline message in the dialog.
    """
    blockers: List[str] = []
    warnings: List[str] = []

    # 1. Training thread in flight — HARD BLOCK
    train_thread = getattr(controller, "_training_thread", None)
    if train_thread is not None:
        try:
            if hasattr(train_thread, "isRunning") and train_thread.isRunning():
                blockers.append(
                    "Training is currently running.  Switching runs now "
                    "would leave the half-trained model file unusable.  "
                    "Wait for training to finish, or press "
                    f"{CANCEL_BTN_TEXT} and "
                    "wait for it to stop, then try again."
                )
        except Exception:
            # Defensive: if isRunning raises, assume blocked
            blockers.append(
                "Training thread state is indeterminate.  Cannot safely switch."
            )

    if blockers:
        return SwitchSafety.BLOCKED, blockers

    # 2. 2D annotator dock open — WARN + CONFIRM
    if viewer is not None and _annotator_dock_open(viewer):
        warnings.append(
            "The 2D Training Annotator is open showing the current run's "
            "patch.  Switching will reload it on the new run's last-used "
            "patch (if any).  Your drawing is saved first: the switch "
            "finishes writing any edits to disk before anything changes.  "
            "Continue?"
        )

    # 3. Main viewer has stale patch layers — informational only (we'll clear)
    if viewer is not None:
        patch_layer_count = _count_patch_layers(viewer, controller)
        if patch_layer_count > 0 and not warnings:
            # Don't add a warning row JUST for this — we'll clean up
            # silently.  But surface in the confirm dialog if a warning
            # is already showing.
            pass
        elif patch_layer_count > 0:
            warnings.append(
                f"The main viewer has {patch_layer_count} layers from the "
                f"current patch.  These will be cleared on switch."
            )

    if warnings:
        return SwitchSafety.NEEDS_CONFIRMATION, warnings

    return SwitchSafety.SAFE, []


def _annotator_dock_open(viewer) -> bool:
    """True if the Step 2 annotator dock is currently mounted on the viewer."""
    try:
        from napari_epicycle.annotator.launcher import _find_dock_by_title
        dock = _find_dock_by_title(viewer, "EpiCycle 2D Training Annotator")
        if dock is None:
            return False
        # Some napari versions report a dock object even after it's been
        # closed; check visibility too.
        if hasattr(dock, "isVisible"):
            try:
                return bool(dock.isVisible())
            except Exception:
                return True  # err on the side of "open" if we can't tell
        return True
    except Exception:
        return False


def _count_patch_layers(viewer, controller) -> int:
    """Count napari layers tied to the current patch (image / labels)."""
    try:
        loaded_patch = str(getattr(controller, "loaded_patch_name", "") or "")
        if not loaded_patch:
            return 0
        layers = getattr(viewer, "layers", [])
        n = 0
        for layer in layers:
            try:
                name = str(getattr(layer, "name", ""))
            except Exception:
                continue
            # Heuristic: layer name contains the patch name (image),
            # or starts with "work_labels_" (controller-managed labels).
            if loaded_patch in name or name.startswith("work_labels_"):
                n += 1
        return n
    except Exception:
        return 0


def collect_patch_layer_names(viewer, controller) -> List[str]:
    """List napari layer names that should be cleared on run switch.

    Conservative sweep: any layer that's part of the patch context.
    Audit-found 2026-05-23: original implementation only checked for
    ``loaded_patch in name OR starts_with("work_labels_")`` which
    missed many in-flight artifacts.  Now matches:

    * Layer name contains loaded_patch_name (image stack + variants)
    * Starts with any of the known controller-managed prefixes:
      ``work_labels_``, ``predictions_``, ``pred_``, ``edt_overlay``,
      ``sam_``, ``isolation_proxy``, ``curation_3d_``, ``refine_``,
      ``fragment_refine_``, ``_mask_``, ``current_prediction``

    Returns:
        List of layer name strings.  Empty list if viewer missing or no
        patch loaded.
    """
    PATCH_LAYER_PREFIXES = (
        "work_labels_",
        "predictions_", "pred_",
        "current_prediction",
        "edt_overlay",
        "sam_",
        "isolation_proxy",
        "curation_3d_",
        "refine_", "fragment_refine_",
    )
    PATCH_LAYER_SUBSTRINGS = ("_mask_",)

    try:
        loaded_patch = str(getattr(controller, "loaded_patch_name", "") or "")
        if not loaded_patch:
            # Even with no loaded_patch_name, still sweep controller-managed
            # prefixes — they're safe to remove (controller will recreate as
            # needed when user picks a patch in the new run).
            pass
        layers = getattr(viewer, "layers", [])
        out: List[str] = []
        for layer in layers:
            try:
                name = str(getattr(layer, "name", ""))
            except Exception:
                continue
            # Match #1: name contains loaded_patch_name
            if loaded_patch and loaded_patch in name:
                out.append(name)
                continue
            # Match #2: controller-managed prefix
            if any(name.startswith(p) for p in PATCH_LAYER_PREFIXES):
                out.append(name)
                continue
            # Match #3: known substring (e.g., '_mask_' in any name)
            if any(s in name for s in PATCH_LAYER_SUBSTRINGS):
                out.append(name)
                continue
        return out
    except Exception:
        return []


__all__ = [
    "SwitchSafety",
    "check_switch_safety",
    "collect_patch_layer_names",
]

"""Backend-agnostic Cellpose post-train output lift.

Cellpose's ``train_seg`` auto-inserts a ``models/`` sub-directory
under ``save_path``::

    cellpose/train.py:452-453:
        filename = save_path / "models" / model_name
        (save_path / "models").mkdir(exist_ok=True)

So when a caller hands Cellpose ``save_path = bundle_dir``, it
actually writes weights to ``bundle_dir/models/{model_name}`` —
not the flat ``bundle_dir/{model_name}`` the rest of EpiCycle
expects.  This module owns the post-train lift logic that
reorganizes Cellpose's output into the canonical bundle layout.

**The lift:**

1. Move ``bundle_dir/models/{weights_filename}`` up to
   ``bundle_dir/{weights_filename}``
2. Move companion files (``_train_loss.npy``, ``_val_loss.npy``,
   ``_training_config.json``, any other strays) to the bundle root
3. Delete the now-empty ``bundle_dir/models/`` subdir

Idempotent: re-running on an already-lifted bundle returns a no-op
result without raising.  Defensive against partial state from a
previous crash or interruption.

**Callers:**

* Step 3 refinement training: ``napari_epicycle/fusion_reconstruction/
  _refinement_bundle.py`` calls this with ``weights_filename`` being
  the bucket name (``"lateral"`` or ``"axial"``) — that's the
  convention Step 3 chose when invoking Cellpose with the bucket as
  ``model_name``.

* Step 2 HITL training (Phase 6+): calls this with ``weights_filename
  = "weights"`` (Step 2 invokes Cellpose with ``model_name="weights"``
  as the canonical name for the new bucket-first layout).

**Why this is its own module:**

Originally lived in ``_refinement_bundle.py``.  Extracted to
``packages/epicycle_training/`` per the Step 2 directory restructure
so both Step 2 and Step 3 share one
implementation without forking.

All filesystem operations emit ``[PERVIEW-CELLPOSE-IO]`` print
lines so future silent layout mismatches are visible in the log.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union


def lift_cellpose_output(
    save_path: Union[str, Path],
    weights_filename: str,
    *,
    log_fn=None,
) -> Dict[str, Any]:
    """Lift Cellpose's auto-inserted ``models/{weights_filename}`` output
    to the bundle root.

    Args:
        save_path: The directory passed to Cellpose as ``save_path``
            (typically a bundle dir, e.g.,
            ``artifacts/step3/refinement/models/my_model/`` or
            ``artifacts/step2/runs/default/lateral/models/round_03/``).
        weights_filename: The name passed to Cellpose as ``model_name``
            (e.g., ``"lateral"`` / ``"axial"`` for Step 3, or
            ``"weights"`` for Step 2).  This is what Cellpose used as
            the filename of the actual weights file.
        log_fn: Optional callable that accepts a single string for
            ``[PERVIEW-CELLPOSE-IO]`` print discipline.  If ``None``,
            falls back to :func:`print`.

    Returns:
        ``{
            "weights_lifted": bool,   # main weights file moved
            "weights_path": Path,      # final post-lift location
            "companions_lifted": list[Path],  # post-lift companion paths
            "models_subdir_removed": bool,
            "skipped_reason": str | None,  # "src_missing" | "dst_exists"
                                            # | "already_lifted" | None
         }``

    Idempotent: re-running on an already-lifted bundle returns a
    no-op result without raising.  Defensive against partial state
    from a previous crash or interruption.

    Raises:
        FileNotFoundError: if ``save_path`` does not exist as a dir.
        ValueError: if ``weights_filename`` is empty.
    """
    if not isinstance(weights_filename, str) or not weights_filename.strip():
        raise ValueError(
            f"weights_filename must be a non-empty str; got {weights_filename!r}"
        )
    log = log_fn if log_fn is not None else print
    save_path = Path(save_path)
    if not save_path.is_dir():
        raise FileNotFoundError(
            f"save_path does not exist as a directory: {save_path}"
        )

    cellpose_subdir = save_path / "models"
    src_weights = cellpose_subdir / weights_filename
    dst_weights = save_path / weights_filename

    result: Dict[str, Any] = {
        "weights_lifted": False,
        "weights_path": dst_weights,
        "companions_lifted": [],
        "models_subdir_removed": False,
        "skipped_reason": None,
    }

    # 1. Main weights lift.
    if not src_weights.exists():
        # Could be: (a) training failed before save, (b) already lifted,
        # (c) name mismatch.  Distinguish a/c (no-op) from b (success).
        if dst_weights.exists():
            log(
                f"[PERVIEW-CELLPOSE-IO] already lifted {dst_weights} "
                f"(idempotent re-call, skipping)"
            )
            result["weights_lifted"] = True  # post-condition holds
            result["skipped_reason"] = "already_lifted"
        else:
            log(
                f"[PERVIEW-CELLPOSE-IO] WARN no weights at {src_weights} "
                f"AND none at {dst_weights} — training may have failed."
            )
            result["skipped_reason"] = "src_missing"
        # Even on weights miss, try companion + cleanup for resilience.
    else:
        if dst_weights.exists():
            log(
                f"[PERVIEW-CELLPOSE-IO] WARN destination {dst_weights} "
                f"already exists; refusing to overwrite (src kept at "
                f"{src_weights} for manual resolution)."
            )
            result["skipped_reason"] = "dst_exists"
        else:
            log(
                f"[PERVIEW-CELLPOSE-IO] cellpose wrote {src_weights}"
            )
            shutil.move(str(src_weights), str(dst_weights))
            log(
                f"[PERVIEW-CELLPOSE-IO] lifted to {dst_weights}"
            )
            result["weights_lifted"] = True

    # 2. Companion lifts (loss arrays, training config, anything else
    # Cellpose dropped in the models/ subdir).  Glob whatever's there
    # — defensive against future Cellpose versions adding new companion
    # file conventions.
    if cellpose_subdir.is_dir():
        for stray in list(cellpose_subdir.iterdir()):
            if not stray.is_file():
                continue
            stray_dst = save_path / stray.name
            if stray_dst.exists():
                log(
                    f"[PERVIEW-CELLPOSE-IO] skip companion lift "
                    f"{stray.name} (dst exists)"
                )
                continue
            shutil.move(str(stray), str(stray_dst))
            log(
                f"[PERVIEW-CELLPOSE-IO] lifted companion "
                f"{stray.name} -> {stray_dst}"
            )
            result["companions_lifted"].append(stray_dst)

    # 3. Clean up empty models/ subdir.
    if cellpose_subdir.is_dir() and not any(cellpose_subdir.iterdir()):
        cellpose_subdir.rmdir()
        log(
            f"[PERVIEW-CELLPOSE-IO] cleanup done; removed empty "
            f"{cellpose_subdir}"
        )
        result["models_subdir_removed"] = True

    # 4. Final bundle-contents log line — visibility into what
    # the lift leaves on disk.
    try:
        bundle_contents = sorted(
            p.name for p in save_path.iterdir()
        )
        log(
            f"[PERVIEW-CELLPOSE-IO] bundle contents: {bundle_contents}"
        )
    except OSError:
        pass

    return result


__all__ = ["lift_cellpose_output"]

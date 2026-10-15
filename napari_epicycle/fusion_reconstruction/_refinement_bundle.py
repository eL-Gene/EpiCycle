"""Post-train Cellpose-output lift + schema 2.0 manifest writer.

This module closes the original wrong-path bug that triggered the
2026-05-20 Step 3 directory restructure.  Phase 6 work.

**The bug:** Cellpose's ``train_seg`` auto-inserts a ``models/``
sub-directory under ``save_path``::

    cellpose/train.py:452-453:
        filename = save_path / "models" / model_name
        (save_path / "models").mkdir(exist_ok=True)

So when we hand Cellpose ``save_path = bundle_dir``, it actually
writes weights to ``bundle_dir/models/{model_name}`` — not the
flat ``bundle_dir/{model_name}`` the rest of EpiCycle expects.

**The fix:** Let Cellpose write naturally (the
"no fighting Cellpose"), then immediately AFTER training:

  1. Move ``bundle/models/<bucket>`` up to ``bundle/<bucket>``
  2. Move companion files (``_train_loss.npy``, ``_val_loss.npy``,
     ``_training_config.json``, any other strays) to the bundle root
  3. Delete the now-empty ``bundle/models/`` subdir
  4. Write/merge the schema 2.0 manifest at ``bundle/manifest.json``

All filesystem operations emit ``[PERVIEW-CELLPOSE-IO]`` print
lines so future silent layout mismatches are visible in the log.

**Manifest schema 2.0** is backend-agnostic to support
future StarDist / Omnipose / 3D models without a schema bump.
Each bucket records its own entry; passes against the same bundle
merge in via ``write_or_merge_bundle_manifest``.

**Bucket convention:** ``"lateral"`` / ``"axial"`` (always — not
patch-name-derived; this is the per-orientation training partition).

**2026-05-23 update (Phase 6 of Step 2 restructure):** The actual
lift-the-files logic moved to ``packages/epicycle_training/cellpose_lift.py``
so Step 2 + Step 3 share one implementation.  This module's
``lift_cellpose_output`` is now a thin Step-3-specific wrapper that
validates the bucket name then delegates to the shared module.
``write_or_merge_bundle_manifest`` stays here because the schema 2.0
manifest is currently Step-3-specific (nested lateral/axial sub-dicts
in one bundle); Step 2's per-bucket-flat manifest lives in its own
writer (added in this phase).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Shared Cellpose post-train lift (extracted to packages/epicycle_training/
# in Phase 6 of the Step 2 restructure).
from packages.epicycle_training.cellpose_lift import (
    lift_cellpose_output as _shared_lift_cellpose_output,
)


SCHEMA_VERSION = "2.0"

# Valid bucket names.  Validated by both the lift + manifest helpers
# so callers can't accidentally pass a typo / patch-name.
VALID_BUCKETS = ("lateral", "axial")


def _validate_bucket(bucket: str) -> None:
    if bucket not in VALID_BUCKETS:
        raise ValueError(
            f"Invalid bucket {bucket!r}; expected one of {VALID_BUCKETS}."
        )


# ---------------------------------------------------------------------------
# Lift
# ---------------------------------------------------------------------------


def lift_cellpose_output(
    bundle_dir: Union[str, Path],
    bucket: str,
    *,
    log_fn=None,
) -> Dict[str, Any]:
    """Lift Cellpose's auto-inserted ``models/<bucket>`` output to the
    bundle root.

    Step-3-specific thin wrapper that validates ``bucket`` is one of
    ``("lateral", "axial")`` and then delegates to the shared
    ``packages.epicycle_training.cellpose_lift.lift_cellpose_output``.

    Step 2 callers (Phase 6+) should import from the shared module
    directly with ``weights_filename="weights"``; the per-bucket
    validation here is a Step-3 convention (Step 3 passes the bucket
    as Cellpose's ``model_name`` so the lift filename IS the bucket
    name).

    Args / Returns / Idempotency — see
    :func:`packages.epicycle_training.cellpose_lift.lift_cellpose_output`.
    """
    _validate_bucket(bucket)
    return _shared_lift_cellpose_output(
        bundle_dir, weights_filename=bucket, log_fn=log_fn,
    )


# ---------------------------------------------------------------------------
# Manifest (schema 2.0)
# ---------------------------------------------------------------------------


def write_or_merge_bundle_manifest(
    bundle_dir: Union[str, Path],
    bundle_name: str,
    bucket: str,
    *,
    init_mode: str = "scratch",
    init_path: Optional[str] = None,
    init_combo_text: Optional[str] = None,
    final_train_loss: Optional[float] = None,
    final_val_loss: Optional[float] = None,
    n_train_pairs: Optional[int] = None,
    n_val_pairs: Optional[int] = None,
    n_epochs: Optional[int] = None,
    source_workspace: Optional[str] = None,
) -> dict:
    """Write or merge the bundle's schema 2.0 manifest.

    Schema 2.0 is backend-agnostic: each bucket entry has
    a ``backend`` field, and ``weights_path`` is opaque to readers
    (file for Cellpose, directory for StarDist's SavedModel, etc.).

    Merge semantics: if a manifest already exists (e.g., this is the
    second pass of a Mode=Both training and the lateral pass already
    wrote its half), the existing top-level + sibling-bucket entries
    are preserved.  Only the named ``bucket`` entry is replaced.

    Returns the manifest dict that was written.

    Example:

        {
          "schema_version": "2.0",
          "name": "test",
          "created_at": "2026-05-20T11:08:30Z",
          "source_workspace": "...",
          "lateral": {
            "backend": "cellpose",
            "weights_path": "lateral",
            "loss_train_path": "lateral_train_loss.npy",
            "loss_val_path":   "lateral_val_loss.npy",
            "training_config_path": "lateral_training_config.json",
            "init": {"mode": "scratch", "path": null, ...},
            "final_loss": {"train": 0.882, "val": 0.0},
            "n_train_pairs": 129,
            "n_val_pairs": 21,
            "n_epochs": 10,
          },
          "axial": { ... }
        }
    """
    _validate_bucket(bucket)
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = bundle_dir / "manifest.json"

    # Load existing manifest if present (merge-on-write).
    existing: dict = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Build top-level fields (preserve existing where present).
    manifest = dict(existing)  # shallow copy of existing
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["name"] = bundle_name
    # created_at: keep the first write's timestamp; update last-modified.
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.setdefault("created_at", now_iso)
    manifest["last_updated"] = now_iso
    if source_workspace is not None:
        manifest["source_workspace"] = source_workspace

    # Build the bucket entry.
    bucket_entry = {
        "backend": "cellpose",
        "weights_path": bucket,  # relative to bundle dir
        "loss_train_path": f"{bucket}_train_loss.npy",
        "loss_val_path": f"{bucket}_val_loss.npy",
        "training_config_path": f"{bucket}_training_config.json",
        "init": {
            "mode": init_mode,
            "path": init_path,
            "init_combo_text": init_combo_text,
        },
        "final_loss": {
            "train": final_train_loss,
            "val": final_val_loss,
        },
        "n_train_pairs": n_train_pairs,
        "n_val_pairs": n_val_pairs,
        "n_epochs": n_epochs,
    }
    manifest[bucket] = bucket_entry

    # Atomic write via tempfile + os.replace (parity with state_manager).
    import os
    import tempfile
    fd, tmp_path = tempfile.mkstemp(
        prefix="manifest.", suffix=".tmp", dir=str(bundle_dir),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, str(manifest_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return manifest


# ---------------------------------------------------------------------------
# Combined: lift + write manifest in one call
# ---------------------------------------------------------------------------


def finalize_refinement_pass(
    bundle_dir: Union[str, Path],
    bundle_name: str,
    bucket: str,
    *,
    log_fn=None,
    **manifest_kwargs,
) -> Dict[str, Any]:
    """Top-level helper called by the training worker after Cellpose
    finishes.  Combines :func:`lift_cellpose_output` +
    :func:`write_or_merge_bundle_manifest` so worker code stays short.

    Returns a dict combining both helpers' results::

        {
            "lift": <result of lift_cellpose_output>,
            "manifest": <manifest dict written>,
        }

    Both phases emit
    ``[PERVIEW-CELLPOSE-IO]`` log lines via ``log_fn``.
    """
    log = log_fn if log_fn is not None else print
    log(
        f"[PERVIEW-CELLPOSE-IO] finalize_refinement_pass start: "
        f"bundle={bundle_name!r} bucket={bucket!r}"
    )
    lift_result = lift_cellpose_output(
        bundle_dir, bucket, log_fn=log,
    )
    manifest = write_or_merge_bundle_manifest(
        bundle_dir, bundle_name, bucket, **manifest_kwargs,
    )
    log(
        f"[PERVIEW-CELLPOSE-IO] wrote schema {SCHEMA_VERSION} manifest "
        f"at {Path(bundle_dir) / 'manifest.json'}; "
        f"buckets in manifest: "
        f"{[k for k in manifest if k in VALID_BUCKETS]}"
    )
    return {"lift": lift_result, "manifest": manifest}


__all__ = [
    "SCHEMA_VERSION",
    "VALID_BUCKETS",
    "lift_cellpose_output",
    "write_or_merge_bundle_manifest",
    "finalize_refinement_pass",
]

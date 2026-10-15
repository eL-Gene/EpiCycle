"""Phase 8d — Import accepted training pairs from one run into another.

User scenario (2026-05-23): "take say the round3 lateral pairs into a
new run so you can start from that and train moving forward".  This
module handles the disk + metadata side of cherry-picking pairs across
runs.

Architecture:

* Pure orchestration (Qt-free) — testable without a controller.
* Two operations:
  - ``list_source_pairs`` — enumerate available pairs in a source run
    + round + bucket, returning rich dicts (label, img_path, mask_path,
    img_mtime).
  - ``copy_pairs_to_target_round`` — atomic copy of selected pairs into
    the active run's chosen round, with per-pair ``inherited_from_pair``
    provenance recorded in the target round's ``round_metadata.json``.
* All file operations use temp-file + os.replace where possible.
* Provenance schema bump: SliceAnnotationRecord gets optional
  ``inherited_from_pair = {source_run_id, source_bucket, source_round_id,
  source_label}``.  Schema 1.1 of round_metadata.json.

What does NOT belong here:
* The Qt dialog (lives in ``_import_pairs_dialog.py``)
* Controller wiring (lives in the controller methods)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import _paths as _step2_paths


class ImportError_(Exception):
    """Raised when an import operation fails partway."""


def _canonicalize_slice_label(label: str) -> Optional[tuple]:
    """Parse a slice label into a canonical ``(stack_id, view, idx)`` tuple.

    2026-05-25 fix for the cherry-pick "finalized shows as accepted" bug.

    Two label formats coexist in the codebase:

    * **z-prefixed + 4-digit padded** (used by disk filenames written by
      ``_persist_curated_training_pair`` and by ``parse_slice_label``
      in ``_paths.py``): ``patch_XY_z0042``
    * **non-prefixed + non-padded** (used by ``sample_records`` keys
      constructed in multiple controller paths as
      ``f"{patch}_{view}_{idx}"``): ``patch_XY_42``

    Both formats refer to the SAME slice but are not string-equal.
    The cherry-pick status lookup must canonicalize both sides to a
    common key, or finalized status from sample_records (which uses
    the non-padded format) will never match disk-walked labels
    (z-prefixed format) — and the dialog falls back to ``"accepted"``
    silently for every finalized pair.

    Returns:
        ``(stack_id, view, slice_idx)`` tuple on success, ``None`` on
        parse failure.  Caller should treat None as "unmatchable" and
        fall back to default status.
    """
    if not isinstance(label, str) or not label:
        return None
    # Strip optional filename suffixes if the caller passed a stem.
    stripped = label
    for suffix in ("_pred_mask", "_mask", "_img"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            break
    parts = stripped.rsplit("_", 2)
    if len(parts) != 3:
        return None
    stack_id, view, idx_part = parts
    view_upper = view.upper()
    if view_upper not in ("XY", "XZ", "YZ"):
        return None
    # Strip 'z' prefix if present (z-prefixed format).
    if idx_part.startswith("z") and len(idx_part) > 1:
        idx_part = idx_part[1:]
    try:
        idx = int(idx_part)
    except (TypeError, ValueError):
        return None
    return (stack_id, view_upper, idx)


def _load_source_pair_statuses(workspace: Path, source_run_id: str) -> Dict[tuple, str]:
    """Build canonical-label-tuple → status lookup from a source run's state JSON.

    Used to enrich `list_source_pairs` results with per-pair status so
    the Import Pairs dialog can filter (default: finalized only).

    2026-05-25 fix: keys are now canonical ``(stack_id, view, idx)``
    tuples produced by ``_canonicalize_slice_label`` (NOT raw label
    strings).  This collapses the two label-format variants
    ('patch_XY_42' and 'patch_XY_z0042') to the same key so the
    cherry-pick lookup hits regardless of which format was used to
    store the sample_record.

    Returns:
        Dict mapping ``(stack_id, view, idx)`` tuple → status
        ("predicted" / "curating" / "accepted" / "staged" / "committed").
        Empty dict if state JSON missing or unparseable.
    """
    state_path = _step2_paths.state_json_path(workspace)
    if not state_path.is_file():
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            v3 = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[WARN] cherry-pick: state JSON parse failed at {state_path}: "
              f"{type(e).__name__}: {e} (all source pairs will default to 'accepted')")
        return {}
    runs = v3.get("runs") if isinstance(v3, dict) else None
    if not isinstance(runs, dict):
        return {}
    source = runs.get(source_run_id)
    if not isinstance(source, dict):
        return {}
    samples = source.get("sample_records")
    if not isinstance(samples, dict):
        return {}
    out: Dict[tuple, str] = {}
    skipped_parse = 0
    for label, rec in samples.items():
        if not isinstance(rec, dict):
            continue
        status = rec.get("status")
        if not isinstance(status, str):
            continue
        canonical = _canonicalize_slice_label(label)
        if canonical is None:
            skipped_parse += 1
            continue
        out[canonical] = status
    # Sample 5 entries for visibility
    for i, (tup, st) in enumerate(out.items()):
        if i >= 5:
            break
    return out


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------

def list_source_pairs(
    workspace: Path | str,
    source_run_id: str,
    source_bucket: str,
    source_round_idx: int,
) -> List[Dict[str, Any]]:
    """Enumerate accepted pairs in a source run/round/bucket.

    Returns one entry per pair (img + mask file pairing detected by
    ``_img/_mask`` suffix convention or by sibling presence).

    Args:
        workspace: workspace root.
        source_run_id: source run slug (must exist).
        source_bucket: "lateral" or "axial".
        source_round_idx: source round index (int, e.g. 3 for round_03).

    Returns:
        List of dicts: ``{label, img_path, mask_path, img_mtime_iso,
        size_bytes}``.  Empty list if source dir missing or no pairs.
    """
    ws = Path(workspace)
    if source_bucket not in {"lateral", "axial"}:
        raise ValueError(f"source_bucket must be lateral|axial, got {source_bucket!r}")

    # Phase B.2 (2026-05-25) dual-path: prefer NEW FLAT layout's
    # `{bucket}/round_NN/accepted/`; fall back to LEGACY SPLIT
    # `{bucket}/rounds/round_NN/accepted/` for un-migrated runs.
    accepted_dir = _step2_paths.round_accepted_dir(
        ws, source_run_id, source_bucket, source_round_idx
    )
    if not accepted_dir.is_dir():
        legacy_accepted = (
            _step2_paths.legacy_split_round_dir(
                ws, source_run_id, source_bucket, source_round_idx
            ) / "accepted"
        )
        if legacy_accepted.is_dir():
            accepted_dir = legacy_accepted
        else:
            return []

    # Build (stack_id, view, idx)-tuple → status lookup from the source
    # run's state JSON.  2026-05-25 fix: keys are now canonical tuples,
    # not raw label strings — so legacy non-padded sample_records keys
    # ('patch_XY_42') match disk-walked labels in z-prefixed format
    # ('patch_XY_z0042').  Without this canonicalization, finalize-
    # then-cherry-pick silently returns "accepted" for every pair.
    label_status = _load_source_pair_statuses(ws, source_run_id)

    # Pair detection — look for _img.tif files, find matching _mask.tif
    out: List[Dict[str, Any]] = []
    n_status_hit = 0
    n_status_miss = 0
    for img_file in sorted(accepted_dir.iterdir()):
        if not img_file.is_file():
            continue
        name = img_file.name
        # Convention: {label}_img.tif + {label}_mask.tif
        if not name.endswith("_img.tif"):
            continue
        label = name[: -len("_img.tif")]
        mask_file = accepted_dir / f"{label}_mask.tif"
        if not mask_file.is_file():
            continue  # orphan — skip
        try:
            mtime = img_file.stat().st_mtime
            mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except OSError:
            mtime_iso = None
        try:
            size_bytes = img_file.stat().st_size + mask_file.stat().st_size
        except OSError:
            size_bytes = 0
        # Canonicalize the disk label so it matches the format used to
        # store sample_records keys.  See _canonicalize_slice_label
        # docstring for the two-format problem.
        canonical = _canonicalize_slice_label(label)
        if canonical is not None and canonical in label_status:
            pair_status = label_status[canonical]
            n_status_hit += 1
        else:
            pair_status = "accepted"  # default: pair-on-disk = at-least-accepted
            n_status_miss += 1
        out.append({
            "label": label,
            "img_path": str(img_file),
            "mask_path": str(mask_file),
            "img_mtime_iso": mtime_iso,
            "size_bytes": size_bytes,
            "status": pair_status,
        })
    return out


def list_source_run_rounds(
    workspace: Path | str,
    source_run_id: str,
) -> List[Dict[str, Any]]:
    """List all rounds × buckets in a source run that have accepted pairs.

    Used to populate the source-round picker in the import dialog.

    Returns:
        List of dicts: ``{bucket, round_id, round_idx, pair_count}``.
        Empty list if run has no accepted pairs.
    """
    ws = Path(workspace)
    out: List[Dict[str, Any]] = []
    for bucket in ("lateral", "axial"):
        # Phase B.2 (2026-05-25) dual-path scan:
        # NEW FLAT layout — round_NN dirs at `{bucket}/round_NN/`
        # LEGACY SPLIT layout — round_NN dirs at `{bucket}/rounds/round_NN/`
        seen_round_idxs: set = set()
        scan_roots: List[Path] = []
        try:
            scan_roots.append(_step2_paths.bucket_root(ws, source_run_id, bucket))
        except (ValueError, TypeError):
            pass
        try:
            scan_roots.append(_step2_paths.bucket_rounds_root(ws, source_run_id, bucket))
        except (ValueError, TypeError):
            pass

        for root in scan_roots:
            if not root.is_dir():
                continue
            for round_dir in sorted(root.iterdir()):
                if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
                    continue
                round_id = round_dir.name
                try:
                    round_idx = int(round_id[len("round_"):])
                except ValueError:
                    continue
                if round_idx in seen_round_idxs:
                    # New-layout wins (scan_roots is bucket_root first)
                    continue
                pairs = list_source_pairs(ws, source_run_id, bucket, round_idx)
                if not pairs:
                    continue
                seen_round_idxs.add(round_idx)
                out.append({
                    "bucket": bucket,
                    "round_id": round_id,
                    "round_idx": round_idx,
                    "pair_count": len(pairs),
                })
    return out


# ----------------------------------------------------------------------------
# Copy / import
# ----------------------------------------------------------------------------

def copy_pairs_to_target_round(
    workspace: Path | str,
    *,
    source_run_id: str,
    source_bucket: str,
    source_round_idx: int,
    target_run_id: str,
    target_bucket: str,
    target_round_idx: int,
    selected_labels: Sequence[str],
    overwrite_collisions: bool = False,
) -> Dict[str, Any]:
    """Copy selected pairs from source → target round; record provenance.

    Args:
        workspace: workspace root.
        source_run_id, source_bucket, source_round_idx: source coords.
        target_run_id, target_bucket, target_round_idx: target coords.
        selected_labels: subset of source pair labels to copy.  Empty
            list = no-op.
        overwrite_collisions: if False (default), labels that already
            exist in the target round are skipped (logged); if True,
            they're overwritten.

    Returns:
        ``{copied: [labels...], skipped: [labels...], errors: [(label, msg)...]}``

    Raises:
        ImportError_: on critical failures (target dir uncreatable,
        provenance write fails).
    """
    ws = Path(workspace)
    if source_bucket not in {"lateral", "axial"}:
        raise ValueError(f"source_bucket must be lateral|axial, got {source_bucket!r}")
    if target_bucket not in {"lateral", "axial"}:
        raise ValueError(f"target_bucket must be lateral|axial, got {target_bucket!r}")
    if (source_run_id == target_run_id and source_bucket == target_bucket
            and source_round_idx == target_round_idx):
        raise ValueError("source and target are the same; copy would be a no-op")

    # Phase C.5 fix (2026-05-26, audit 2 LOW): mirror the dual-path
    # fallback that `list_source_pairs` already has — try FLAT first,
    # fall back to LEGACY SPLIT.  Pre-fix the dialog could show a
    # legacy-split source run via `list_source_run_rounds` +
    # `list_source_pairs` (both dual-path), user picks pairs, then
    # `copy_pairs_to_target_round` raises "source dir does not exist".
    # Loud failure (not silent), but inconsistent across the dialog
    # flow.  Post-fix same dual-path resolution everywhere.
    source_accepted = _step2_paths.round_accepted_dir(
        ws, source_run_id, source_bucket, source_round_idx
    )
    if not source_accepted.is_dir():
        legacy_split = (
            _step2_paths.legacy_split_round_dir(
                ws, source_run_id, source_bucket, source_round_idx
            ) / "accepted"
        )
        if legacy_split.is_dir():
            source_accepted = legacy_split
        else:
            raise ImportError_(
                f"source dir does not exist: {source_accepted} "
                f"(also checked legacy split path)"
            )

    target_accepted = _step2_paths.round_accepted_dir(
        ws, target_run_id, target_bucket, target_round_idx
    )
    target_accepted.mkdir(parents=True, exist_ok=True)

    # Build label → (img_path, mask_path) lookup from source
    source_pairs_by_label = {
        p["label"]: p
        for p in list_source_pairs(ws, source_run_id, source_bucket, source_round_idx)
    }

    copied: List[str] = []
    skipped: List[str] = []
    errors: List[tuple[str, str]] = []

    for label in selected_labels:
        if label not in source_pairs_by_label:
            errors.append((label, "not found in source"))
            continue
        src = source_pairs_by_label[label]
        target_img = target_accepted / f"{label}_img.tif"
        target_mask = target_accepted / f"{label}_mask.tif"
        if (target_img.exists() or target_mask.exists()) and not overwrite_collisions:
            skipped.append(label)
            continue
        try:
            shutil.copy2(src["img_path"], target_img)
            shutil.copy2(src["mask_path"], target_mask)
            copied.append(label)
        except (OSError, shutil.Error) as e:
            errors.append((label, f"copy failed: {e}"))
            # Cleanup partial copy
            for p in (target_img, target_mask):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass

    # Update target round_metadata.json with provenance records for copied
    # pairs.  Append (don't replace) so existing native pairs are preserved.
    if copied:
        _update_target_metadata_with_provenance(
            ws,
            target_run_id=target_run_id,
            target_bucket=target_bucket,
            target_round_idx=target_round_idx,
            source_run_id=source_run_id,
            source_bucket=source_bucket,
            source_round_idx=source_round_idx,
            source_labels=copied,
        )

    return {
        "copied": copied,
        "skipped": skipped,
        "errors": errors,
    }


def _update_target_metadata_with_provenance(
    workspace: Path,
    *,
    target_run_id: str,
    target_bucket: str,
    target_round_idx: int,
    source_run_id: str,
    source_bucket: str,
    source_round_idx: int,
    source_labels: List[str],
) -> None:
    """Append provenance records to the target's round_metadata.json.

    Each new record:
        {
          "label_id": <label>,
          "status": "accepted",
          "inherited_from_pair": {
            "source_run_id": ...,
            "source_bucket": ...,
            "source_round_id": "round_NN",
            "source_label": <label>,
          },
          "imported_at": <ISO>
        }

    Schema bump: ``round_metadata.json::schema_version`` becomes
    ``"1.1"`` (forward-compat — older readers ignore the new field).
    """
    meta_path = _step2_paths.round_metadata_path(
        workspace, target_run_id, target_bucket, target_round_idx
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing or initialize
    existing: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.setdefault("schema_version", "1.1")
    existing["schema_version"] = "1.1"  # bump even if was 1.0
    existing.setdefault("round_id", _step2_paths._round_segment(target_round_idx))
    samples = existing.setdefault("samples", [])
    if not isinstance(samples, list):
        samples = []
        existing["samples"] = samples

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_round_id = _step2_paths._round_segment(source_round_idx)
    existing_labels = {
        s.get("label_id") for s in samples
        if isinstance(s, dict) and s.get("label_id")
    }
    for label in source_labels:
        if label in existing_labels:
            # Already in metadata — skip to avoid duplicate rows
            continue
        samples.append({
            "label_id": label,
            "status": "accepted",
            "inherited_from_pair": {
                "source_run_id": source_run_id,
                "source_bucket": source_bucket,
                "source_round_id": source_round_id,
                "source_label": label,
            },
            "imported_at": now_iso,
        })

    # Atomic write
    payload = json.dumps(existing, indent=2, ensure_ascii=False)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".round_metadata_", suffix=".json.tmp", dir=str(meta_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, meta_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise ImportError_(f"failed to write target round_metadata.json: {meta_path}")


__all__ = [
    "ImportError_",
    "list_source_pairs",
    "list_source_run_rounds",
    "copy_pairs_to_target_round",
]

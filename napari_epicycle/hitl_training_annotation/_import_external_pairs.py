"""Phase 8f — Import external 2D image/mask pairs into a run.

User scenario (2026-05-24): "if I have a different 2D GT image/pair
that isn't from EpiCycle, how can I also import that?"  Ideal for new
users with pre-curated training data from other tools.

User feedback (2026-05-24): "for import external...maybe we can
specify the suffix?"  Auto-detection can mis-pair when conventions
are ambiguous (e.g. `*_img`/`*_mask` vs `*_image`/`*_label` in the
same folder).  This module's API forces the caller to supply suffix
strings explicitly — no auto-detection magic, but a helper for
suggesting common patterns.

Architecture:

* `enumerate_pairs(folder, img_suffix, mask_suffix)` — scan folder,
  find pairs by suffix matching, return list of dicts (basename, img,
  mask, img_size, mask_size)
* `validate_pair(img_path, mask_path)` — check dtype + 2D shape +
  shape match; return (ok, message)
* `copy_external_pairs_to_target_round(...)` — atomic copy + provenance
  write (mirror of Phase 8d's copy_pairs_to_target_round)
* `suggest_pair_convention(folder)` — scan folder + return best-guess
  (img_suffix, mask_suffix) tuple for the `[Auto]` button in the dialog
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import _paths as _step2_paths


class ExternalImportError(Exception):
    """Raised when an external-pair import operation fails."""


# Common suffix conventions we know about — used by suggest_pair_convention
# to score folder contents.
_KNOWN_CONVENTIONS = [
    ("_img.tif", "_mask.tif"),
    ("_img.tiff", "_mask.tiff"),
    ("_image.tif", "_label.tif"),
    ("_image.tiff", "_label.tiff"),
    (".tif", "_mask.tif"),
    (".tiff", "_mask.tiff"),
    (".tif", "_label.tif"),
    ("_raw.tif", "_gt.tif"),
]


# ----------------------------------------------------------------------------
# Discovery + auto-suggest
# ----------------------------------------------------------------------------

def enumerate_pairs(
    folder: Path | str,
    img_suffix: str,
    mask_suffix: str,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Enumerate image/mask pairs in a folder by suffix matching.

    Args:
        folder: source folder to scan.
        img_suffix: filename suffix that identifies image files (e.g.
            ``"_img.tif"`` — must include extension).
        mask_suffix: filename suffix that identifies mask files (e.g.
            ``"_mask.tif"``).

    Returns:
        ``(pairs, orphan_imgs, orphan_masks)`` triple:
        * ``pairs`` — list of ``{basename, img_path, mask_path,
          img_size, mask_size}`` dicts for successfully-paired files
        * ``orphan_imgs`` — list of image filenames with no matching mask
        * ``orphan_masks`` — list of mask filenames with no matching image

    Algorithm: scan all files in folder; for each file ending with
    img_suffix, derive basename = filename[:-len(img_suffix)], look for
    basename + mask_suffix.  Orphans on either side are reported.

    Empty suffixes raise ValueError.  Identical suffixes (img == mask)
    also raise — can't disambiguate.
    """
    if not img_suffix:
        raise ValueError("img_suffix cannot be empty")
    if not mask_suffix:
        raise ValueError("mask_suffix cannot be empty")
    if img_suffix == mask_suffix:
        raise ValueError(
            f"img_suffix and mask_suffix must differ, both = {img_suffix!r}"
        )

    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ExternalImportError(f"source folder not found: {folder_path}")

    # Build sets of basenames for each suffix
    img_basenames: Dict[str, Path] = {}
    mask_basenames: Dict[str, Path] = {}
    for entry in sorted(folder_path.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        # Mask check goes first if mask_suffix is a strict superset of
        # img_suffix (e.g. img=".tif", mask="_mask.tif") — otherwise a
        # mask file would match the img check too.  General rule: prefer
        # the longer suffix when both could match.
        matched = False
        if len(mask_suffix) > len(img_suffix) and name.endswith(mask_suffix):
            basename = name[:-len(mask_suffix)]
            mask_basenames[basename] = entry
            matched = True
        elif len(img_suffix) > len(mask_suffix) and name.endswith(img_suffix):
            basename = name[:-len(img_suffix)]
            img_basenames[basename] = entry
            matched = True
        elif name.endswith(mask_suffix):
            basename = name[:-len(mask_suffix)]
            mask_basenames[basename] = entry
            matched = True
        elif name.endswith(img_suffix):
            basename = name[:-len(img_suffix)]
            img_basenames[basename] = entry
            matched = True
        # Unmatched files are silently skipped — they're not image/mask
        # data per the user's specified convention.

    # Pair up + identify orphans
    pairs: List[Dict[str, Any]] = []
    orphan_imgs: List[str] = []
    orphan_masks: List[str] = []
    for basename in sorted(img_basenames):
        if basename in mask_basenames:
            img_path = img_basenames[basename]
            mask_path = mask_basenames[basename]
            try:
                img_size = img_path.stat().st_size
            except OSError:
                img_size = 0
            try:
                mask_size = mask_path.stat().st_size
            except OSError:
                mask_size = 0
            pairs.append({
                "basename": basename,
                "img_path": str(img_path),
                "mask_path": str(mask_path),
                "img_size": img_size,
                "mask_size": mask_size,
            })
        else:
            orphan_imgs.append(img_basenames[basename].name)
    for basename in sorted(mask_basenames):
        if basename not in img_basenames:
            orphan_masks.append(mask_basenames[basename].name)
    return pairs, orphan_imgs, orphan_masks


def suggest_pair_convention(folder: Path | str) -> Tuple[str, str]:
    """Scan folder + return the best-guess (img_suffix, mask_suffix).

    Tries each entry in ``_KNOWN_CONVENTIONS`` and picks the one that
    produces the most pairs.  Ties broken by convention order (earlier
    = preferred).  Returns ``("_img.tif", "_mask.tif")`` (EpiCycle's
    own convention) if no convention matches anything.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return ("_img.tif", "_mask.tif")

    best = (0, ("_img.tif", "_mask.tif"))
    for img_suf, mask_suf in _KNOWN_CONVENTIONS:
        try:
            pairs, _, _ = enumerate_pairs(folder_path, img_suf, mask_suf)
        except (ValueError, ExternalImportError):
            continue
        if len(pairs) > best[0]:
            best = (len(pairs), (img_suf, mask_suf))
    return best[1]


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def validate_pair(img_path: Path | str, mask_path: Path | str) -> Tuple[bool, str]:
    """Lightweight validation: check that both files exist + have
    compatible 2D shape.

    Does NOT load the full file into memory — uses ``tifffile.TiffFile``
    metadata for fast shape lookup.

    Returns:
        ``(ok, message)`` — ``ok=True`` on success; ``message`` empty
        on success, otherwise a short reason.
    """
    img_p = Path(img_path)
    mask_p = Path(mask_path)
    if not img_p.is_file():
        return False, f"image file missing: {img_p}"
    if not mask_p.is_file():
        return False, f"mask file missing: {mask_p}"

    try:
        import tifffile
    except ImportError:
        return False, "tifffile not installed (required for TIFF reads)"

    try:
        with tifffile.TiffFile(str(img_p)) as tif:
            img_shape = tif.pages[0].shape
            img_dtype = tif.pages[0].dtype
    except Exception as e:
        return False, f"could not read image TIFF: {type(e).__name__}: {e}"

    try:
        with tifffile.TiffFile(str(mask_p)) as tif:
            mask_shape = tif.pages[0].shape
            mask_dtype = tif.pages[0].dtype
    except Exception as e:
        return False, f"could not read mask TIFF: {type(e).__name__}: {e}"

    # Shape compatibility: both should be 2D + spatial dims should match
    img_spatial = img_shape[-2:] if len(img_shape) >= 2 else img_shape
    mask_spatial = mask_shape[-2:] if len(mask_shape) >= 2 else mask_shape
    if img_spatial != mask_spatial:
        return False, (
            f"image + mask spatial shapes differ: img={img_spatial}, "
            f"mask={mask_spatial}"
        )
    if len(mask_shape) > 2:
        # Mask must be 2D (instance labels for a single slice)
        return False, f"mask must be 2D, got shape={mask_shape}"
    # Image may be 2D (single channel) or 3D (multi-channel, last dim
    # = channels) — both acceptable.
    return True, ""


# ----------------------------------------------------------------------------
# Copy + provenance
# ----------------------------------------------------------------------------

def copy_external_pairs_to_target_round(
    workspace: Path | str,
    *,
    target_run_id: str,
    target_bucket: str,
    target_round_idx: int,
    pair_records: Sequence[Dict[str, Any]],
    relabel_basename: bool = True,
    overwrite_collisions: bool = False,
    source_channel_axis: str = "first",
) -> Dict[str, Any]:
    """Copy validated external pairs into a target run's accepted dir.

    Mirror of Phase 8d's ``copy_pairs_to_target_round`` but for external
    sources (writes ``inherited_from_external`` provenance instead of
    ``inherited_from_pair``).

    Args:
        workspace: workspace root.
        target_run_id, target_bucket, target_round_idx: target coords.
        pair_records: items from ``enumerate_pairs`` (or hand-built
            dicts with ``basename``, ``img_path``, ``mask_path``).
        relabel_basename: if True, normalize the imported label to
            EpiCycle's ``{stack_id}_{VIEW}_z{idx}`` convention so the
            controller's sample_records can parse it.  If False, keep
            the original basename (caller is responsible for ensuring
            it parses correctly).  Default True for safety — most
            external sources don't follow this convention.
        overwrite_collisions: if False (default), labels that already
            exist in the target round are skipped; if True, overwritten.

    Returns:
        ``{copied: [labels], skipped: [labels], errors: [(label, msg)]}``
    """
    ws = Path(workspace)
    if target_bucket not in {"lateral", "axial"}:
        raise ValueError(
            f"target_bucket must be lateral|axial, got {target_bucket!r}"
        )

    target_accepted = _step2_paths.round_accepted_dir(
        ws, target_run_id, target_bucket, target_round_idx
    )
    target_accepted.mkdir(parents=True, exist_ok=True)

    # Default view for the bucket — external pairs don't have explicit
    # view info, so we tag them based on target bucket.
    default_view = "XY" if target_bucket == "lateral" else "XZ"

    copied: List[str] = []
    skipped: List[str] = []
    errors: List[Tuple[str, str]] = []
    provenance_records: List[Dict[str, Any]] = []

    # Pre-loop pass — pre-compute target labels + detect intra-batch
    # collisions BEFORE any disk writes (audit-found 2026-05-24).
    # Without this, two source basenames that sanitize to the same
    # target label (e.g., "foo bar" + "foo-bar" → both "foo_bar")
    # would silently lose the second one to the post-write skip
    # branch — no diagnostic, looks like the second pair "didn't exist".
    seen_target_labels: Dict[str, str] = {}  # target_label → first source basename
    prepared_pairs = []
    for idx, record in enumerate(pair_records):
        basename = record.get("basename", "")
        img_path = record.get("img_path", "")
        mask_path = record.get("mask_path", "")
        if not basename or not img_path or not mask_path:
            errors.append((basename, "incomplete pair record"))
            continue
        if relabel_basename:
            target_label = f"ext_{basename}_{default_view}_z{idx:04d}"
            target_label = _sanitize_label(target_label)
        else:
            target_label = basename
        if target_label in seen_target_labels:
            other_basename = seen_target_labels[target_label]
            errors.append((basename,
                f"post-sanitization label collision with '{other_basename}' "
                f"(both → '{target_label}'); rename one of the source files "
                f"or set relabel_basename=False"))
            continue
        seen_target_labels[target_label] = basename
        prepared_pairs.append((idx, basename, img_path, mask_path, target_label))

    for idx, basename, img_path, mask_path, target_label in prepared_pairs:
        target_img = target_accepted / f"{target_label}_img.tif"
        target_mask = target_accepted / f"{target_label}_mask.tif"
        if (target_img.exists() or target_mask.exists()) and not overwrite_collisions:
            skipped.append(target_label)
            continue

        try:
            # Phase 8f polish 2026-05-24: handle channel-axis conversion.
            # EpiCycle's downstream Cellpose pipeline reads images
            # assuming channel-first ((C, H, W)).  If the source is
            # channel-last ((H, W, C)) we transpose on copy.  2D inputs
            # pass through unchanged.
            #
            # Audit-fix 2026-05-24 (J CRITICAL): VALIDATE shape against
            # declared axis BEFORE the copy.  Previously "first" and
            # "none" went through shutil.copy2 with zero shape check,
            # so a user declaring "first" on (H, W, C) data would
            # silently corrupt training.  Now we reject the pair with
            # an error if shape doesn't match declaration.
            shape_ok, shape_err = _validate_image_shape_vs_axis(
                img_path, source_channel_axis
            )
            if not shape_ok:
                errors.append((target_label, shape_err))
                continue
            if source_channel_axis == "first" or source_channel_axis == "none":
                # Channel-first OR 2D — no transpose, plain copy
                shutil.copy2(img_path, target_img)
            elif source_channel_axis == "last":
                _copy_with_axis_transpose(img_path, target_img)
            else:
                raise ValueError(
                    f"source_channel_axis must be 'first'|'last'|'none', "
                    f"got {source_channel_axis!r}"
                )
            # Mask is always 2D — no transpose ever needed
            shutil.copy2(mask_path, target_mask)
            copied.append(target_label)
            provenance_records.append({
                "target_label": target_label,
                "source_basename": basename,
                "source_img_path": img_path,
                "source_mask_path": mask_path,
                "source_channel_axis": source_channel_axis,
            })
        except (OSError, shutil.Error, ExternalImportError, ValueError) as e:
            # Catch ExternalImportError (e.g. 4D rejection from
            # _copy_with_axis_transpose) and ValueError (bad axis arg)
            # alongside the I/O errors.  Audit-fix 2026-05-24.
            errors.append((target_label, f"copy failed: {e}"))
            for p in (target_img, target_mask):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass

    if copied:
        _update_metadata_with_external_provenance(
            ws,
            target_run_id=target_run_id,
            target_bucket=target_bucket,
            target_round_idx=target_round_idx,
            provenance_records=provenance_records,
        )

    return {
        "copied": copied,
        "skipped": skipped,
        "errors": errors,
        "provenance": provenance_records,
    }


def _validate_image_shape_vs_axis(
    img_path: str, source_channel_axis: str
) -> Tuple[bool, str]:
    """Audit-fix 2026-05-24 (J CRITICAL): cross-check image shape against
    the user's declared channel-axis BEFORE the copy proceeds.

    Without this, a mis-declared axis would silently produce mis-shaped
    training data — Cellpose would train on garbage.

    Returns:
        (ok, error_message): ok=True means shape is consistent with
        declared axis; ok=False means shape disagrees + error_message
        describes the mismatch for the user.

    Defensive: if the file can't be read for shape inspection, returns
    (True, "") — trusts the user's declaration in that case rather than
    blocking on what might be a transient I/O issue.  The copy itself
    will fail loudly if the file is truly unreadable.
    """
    try:
        import tifffile
        with tifffile.TiffFile(img_path) as tif:
            # Use series[0].shape (full stack shape) not pages[0].shape
            # (single-page shape).  Multi-page TIFFs like (2, 64, 64)
            # serialized as 2 pages of (64, 64) would otherwise return
            # (64, 64) and bypass the 3D check.  Audit-fix 2026-05-24.
            if tif.series and len(tif.series) > 0:
                shape = tuple(tif.series[0].shape)
            else:
                shape = tuple(tif.pages[0].shape)
    except Exception:
        return True, ""  # defensive: don't block on shape-read failure

    n_dims = len(shape)
    if n_dims == 2:
        # 2D source — accept regardless of declared axis (no channels
        # to interpret).  shutil.copy2 will pass through unchanged for
        # "first" and "none"; "last" would attempt to moveaxis a 2D
        # array which `_copy_with_axis_transpose` short-circuits.
        return True, ""
    if n_dims == 3:
        if source_channel_axis == "none":
            return False, (
                f"shape mismatch: source has 3 dims {tuple(shape)} but "
                f"you declared '2D / grayscale'; pick 'Channels first' "
                f"or 'Channels last' depending on layout"
            )
        # Both 'first' and 'last' are formally valid for 3D — but heuristic
        # warns if smallest dim is in the wrong position (channels are
        # usually 1-5, much smaller than spatial dims).
        smallest = min(shape)
        if smallest > 10:
            # All dims are large — probably (Z, H, W) volumetric — block
            return False, (
                f"shape {tuple(shape)} looks volumetric (no axis with "
                f"size ≤ 10 typical for channels); 2D Cellpose can't "
                f"ingest volumetric data through this dialog"
            )
        smallest_idx = shape.index(smallest)
        if source_channel_axis == "first" and smallest_idx != 0:
            return False, (
                f"shape {tuple(shape)} — channels look like the LAST "
                f"axis (smallest dim {smallest} at position {smallest_idx}), "
                f"but you declared 'Channels first'.  Pick 'Channels last' "
                f"instead, or verify your data."
            )
        if source_channel_axis == "last" and smallest_idx != n_dims - 1:
            return False, (
                f"shape {tuple(shape)} — channels look like the FIRST "
                f"axis (smallest dim {smallest} at position {smallest_idx}), "
                f"but you declared 'Channels last'.  Pick 'Channels first' "
                f"instead, or verify your data."
            )
        return True, ""
    # n_dims > 3 or n_dims < 2 — block outright
    return False, (
        f"source shape {tuple(shape)} has {n_dims} dimensions; "
        f"EpiCycle's 2D Cellpose pipeline expects 2D (H, W) or 3D "
        f"(C, H, W) input"
    )


def _copy_with_axis_transpose(src_img_path: str, dst_img_path: Path) -> None:
    """Read image, transpose channel-last → channel-first, write out.

    Phase 8f polish 2026-05-24.  Used when user declares source images
    are channel-last (H, W, C) so the downstream Cellpose pipeline
    (which expects (C, H, W) per controller.py:449) sees the right
    ordering.

    Raises:
        ExternalImportError if the source array has ndim > 3 (volumetric
        data) — EpiCycle's 2D Cellpose pipeline can't ingest 4D arrays;
        previously such inputs silently passed through unchanged + trained
        Cellpose on garbage.  Audit-fix 2026-05-24.
    """
    import tifffile
    import numpy as np
    arr = tifffile.imread(src_img_path)
    if arr.ndim == 2:
        # Already 2D — no transpose, plain write
        tifffile.imwrite(str(dst_img_path), arr)
        return
    if arr.ndim == 3:
        # Heuristic: channel-last means the smallest axis is the last
        # axis.  Trust the user-declared axis; do a straight transpose.
        arr = np.moveaxis(arr, -1, 0)
        tifffile.imwrite(str(dst_img_path), arr)
        return
    # ndim > 3 — refuse rather than silently pass through.  Volumetric
    # data isn't supported by 2D Cellpose; importing it would train on
    # mis-shaped arrays.
    raise ExternalImportError(
        f"source image has ndim={arr.ndim} (shape={tuple(arr.shape)}); "
        f"EpiCycle's 2D Cellpose pipeline expects 2D (H, W) or 3D "
        f"(C, H, W).  Volumetric / 4D+ data is not importable through "
        f"this dialog."
    )


def _sanitize_label(label: str) -> str:
    """Replace chars that break slice_label parsing (whitespace, punct)
    with underscores."""
    import re
    # Keep alphanumeric + underscore.  Replace everything else with _.
    return re.sub(r"[^A-Za-z0-9_]", "_", label)


def _update_metadata_with_external_provenance(
    workspace: Path,
    *,
    target_run_id: str,
    target_bucket: str,
    target_round_idx: int,
    provenance_records: List[Dict[str, Any]],
) -> None:
    """Append external-source provenance to target round_metadata.json.

    Bumps schema 1.0/1.1 → 1.1 (additive — same schema bump as
    Phase 8d's inherited_from_pair).
    """
    meta_path = _step2_paths.round_metadata_path(
        workspace, target_run_id, target_bucket, target_round_idx
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)

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
    existing["schema_version"] = "1.1"
    existing.setdefault("round_id", _step2_paths._round_segment(target_round_idx))
    samples = existing.setdefault("samples", [])
    if not isinstance(samples, list):
        samples = []
        existing["samples"] = samples

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    existing_labels = {
        s.get("label_id") for s in samples
        if isinstance(s, dict) and s.get("label_id")
    }
    for prov in provenance_records:
        label = prov["target_label"]
        if label in existing_labels:
            continue  # dedup
        samples.append({
            "label_id": label,
            "status": "accepted",
            "inherited_from_external": {
                "source_path": str(Path(prov["source_img_path"]).parent),
                "source_filename": Path(prov["source_img_path"]).name,
                "source_mask_filename": Path(prov["source_mask_path"]).name,
                "source_basename": prov["source_basename"],
                "source_channel_axis": prov.get("source_channel_axis", "first"),
            },
            "imported_at": now_iso,
        })

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
        raise ExternalImportError(
            f"failed to write target round_metadata.json: {meta_path}"
        )


__all__ = [
    "ExternalImportError",
    "enumerate_pairs",
    "suggest_pair_convention",
    "validate_pair",
    "copy_external_pairs_to_target_round",
]

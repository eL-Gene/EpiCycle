"""Pure-function helpers for Step 2 multi-run management (Phase 8a).

Owns the disk-level + state-JSON-level operations that the controller's
``create_run`` / ``switch_run`` / ``list_runs`` methods need.  Keeping
this in a separate module (not on the controller) makes it testable
without spinning up a Qt + napari + controller stack.

The controller is responsible for:
* gathering parameters from the UI
* calling these helpers
* saving + reloading its in-memory state around a switch
* emitting ``[STEP2-RUN]`` console diagnostics

This module is responsible for:
* atomically updating ``step2_state.json`` (add run entry, set active)
* creating the run directory + writing ``run_manifest.json``
* scanning ``runs/`` and returning enriched summaries (round counts,
  last activity, archived flag)

The interaction flows of the New Run dialog drive each function's contract.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from packages.epicycle_io.step2_run_manifest import (
    RUN_MANIFEST_FILENAME,
    build_run_manifest,
    is_run_archived,
    list_runs_with_manifests,
    read_run_manifest,
    write_run_manifest,
)

from . import _paths as _step2_paths


# Commit 2 audit LOW-3 fix (2026-05-26): module-level constants for the
# fallback-scanner exclusion sets (was being rebuilt per round_dir).
# Zero-cost hoist; readability win + single source of truth.
#
# `_FALLBACK_EXCLUDED_NAMES`: exact filenames that are never canonical
# weights.  Includes:
#   * "manifest.json" + "models" (canonical artifacts inside the bundle)
#   * "weights" (Audit L-2: pre-Phase-6c bundles stored weights here;
#     post-Phase-6c the canonical filename is `{run_id}_{bucket}`)
#   * "Thumbs.db" + "desktop.ini" (Windows OS metadata files; Commit 6
#     audit Audit-3/5 LOW: these don't start with `.` or `~` so the
#     dotfile/backup-suffix exclusions miss them — Windows users with
#     File Explorer browsing the round dir could leave these behind).
_FALLBACK_EXCLUDED_NAMES = frozenset({
    "manifest.json", "models", "weights",
    "Thumbs.db", "desktop.ini",
})

# `_FALLBACK_EXCLUDED_SUFFIXES`: filename suffixes that mark Cellpose
# companion outputs (loss arrays + training config).  The `_validation_`
# variant is defensive only — Cellpose never wrote that name; kept in
# case any out-of-band tool ever did.
_FALLBACK_EXCLUDED_SUFFIXES = (
    "_train_loss.npy",
    "_val_loss.npy",
    "_validation_loss.npy",  # defensive — Cellpose uses abbreviated name
    "_training_config.json",
)


class RunAlreadyExistsError(Exception):
    """Raised when create_run sees the target slug already on disk or in state."""


class RunNotFoundError(Exception):
    """Raised when switch_run / delete_run target a slug that doesn't exist."""


class InvalidRunIdError(Exception):
    """Raised when the slug fails ``_paths.validate_run_id``."""


# ----------------------------------------------------------------------------
# Disk-level: create / delete run directory
# ----------------------------------------------------------------------------

def create_run_on_disk(
    workspace: Path | str,
    slug: str,
    *,
    display_name: str,
    target: str = "",
    description: str = "",
    backend: str = "cellpose",
    init_spec: Optional[Dict[str, Any]] = None,
    channels_snapshot: Optional[List[str]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Create ``runs/{slug}/`` and write its ``run_manifest.json``.

    Atomic-ish: creates the directory first; if manifest write fails, the
    empty dir is cleaned up.  Race-safe via ``mkdir(exist_ok=False)``.

    Args:
        workspace: workspace root path.
        slug: run ID (must validate via ``_paths.validate_run_id``).
        display_name, target, description, backend, init_spec,
        channels_snapshot, notes: forwarded to ``build_run_manifest``.

    Returns:
        The manifest dict that was written (useful for callers that want
        to use the canonical ``created_at`` etc.).

    Raises:
        InvalidRunIdError: if slug fails validation.
        RunAlreadyExistsError: if the directory already exists.
        ValueError: if build_run_manifest rejects the args.
        OSError: on filesystem failures during directory creation.
    """
    # Validate slug first
    _validate_or_raise(slug)

    ws = Path(workspace)
    runs_root = _step2_paths.step2_runs_root(ws)
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dir = runs_root / slug
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        raise RunAlreadyExistsError(
            f"run directory already exists: {run_dir}"
        ) from e

    # Manifest write — if it fails, clean up the empty run dir so the
    # workspace doesn't end up with a half-created run.
    try:
        manifest = build_run_manifest(
            run_id=slug,
            display_name=display_name,
            target=target,
            description=description,
            backend=backend,
            created_from=init_spec,
            channels_snapshot=channels_snapshot,
            notes=notes,
        )
        write_run_manifest(run_dir / RUN_MANIFEST_FILENAME, manifest)
    except Exception:
        # Cleanup: rmdir only succeeds if directory is empty (it is — we
        # just made it).  Defensive: catch any OS error.
        try:
            run_dir.rmdir()
        except OSError:
            pass
        raise

    return manifest


def delete_run_from_disk(
    workspace: Path | str,
    slug: str,
    *,
    confirm: bool = False,
) -> Path:
    """Move ``runs/{slug}/`` out of the run tree, into ``step2/.trash/``.

    Brick D / journey #18 (2026-09-07): this used to be ``shutil.rmtree`` —
    one mis-click and a run's rounds, pairs and models were gone with no
    recourse.  The run still disappears from the UI and from every scan (it
    is no longer under ``runs/``), but it stays on disk under
    ``artifacts/step2/.trash/{slug}-{timestamp}`` so the user can move it
    back or delete it in Explorer.  ``.trash/`` is a plain directory, not a
    schema: no reader looks inside it.

    Caller MUST still pass ``confirm=True`` after running the provenance
    dependency check (no other runs inherited from this).

    Args:
        workspace: workspace root.
        slug: run ID.
        confirm: safety flag; must be True or this raises.

    Returns:
        the path the run was moved to.

    Raises:
        ValueError: if confirm is not True.
        RunNotFoundError: if the run directory doesn't exist.
    """
    if not confirm:
        raise ValueError(
            "delete_run_from_disk requires confirm=True; caller did not "
            "complete the provenance check"
        )
    ws = Path(workspace)
    run_dir = _step2_paths.run_root(ws, slug)
    if not run_dir.is_dir():
        raise RunNotFoundError(f"run not found: {run_dir}")

    from datetime import datetime
    import shutil

    trash_root = run_dir.parent.parent / ".trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = trash_root / f"{slug}-{stamp}"
    n = 1
    while dest.exists():
        n += 1
        dest = trash_root / f"{slug}-{stamp}-{n}"
    shutil.move(str(run_dir), str(dest))
    print(f"[STEP2-RUN] deleted run {slug!r}: moved to {dest} "
          f"(recoverable — delete it there to reclaim the disk)")
    return dest


# ----------------------------------------------------------------------------
# State-JSON-level: add run entry / set active / get active
# ----------------------------------------------------------------------------

def add_run_to_state_json(
    workspace: Path | str,
    slug: str,
    *,
    activate: bool = False,
    init_spec: Optional[Dict[str, Any]] = None,
    target_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """Add an empty per-run entry to ``runs[{slug}]`` in step2_state.json.

    If the state file doesn't exist, creates a minimal v3 envelope.
    If the slug already exists in state, this is a no-op for the entry
    (but ``activate`` is still honored, and ``init_spec`` overrides are
    NOT applied — only fresh entries get pre-populated).

    Args:
        workspace: workspace root.
        slug: run ID to add.
        activate: if True, also set ``active_run_id = slug``.
        init_spec: optional ``created_from`` dict for the new run.  When
            provided AND the slug is fresh, resolves the spec to a
            checkpoint (via :func:`resolve_init_spec_to_checkpoint`) and
            pre-populates the Model section fields of the per-run state
            (Bug D, 2026-05-29).  No effect for existing slugs, for
            ``scratch`` mode, or when the spec can't be resolved.
        target_bucket: optional ``"lateral"`` / ``"axial"`` of the new
            run, used by ``fork`` resolution to prefer the source's
            matching bucket.

    Returns:
        The updated v3 state dict (post-write).

    Raises:
        InvalidRunIdError: if slug fails validation.
        OSError: on filesystem failures.
    """
    _validate_or_raise(slug)

    ws = Path(workspace)
    state_path = _step2_paths.state_json_path(ws)

    # Load existing v3 or create a fresh envelope
    v3 = _load_v3_or_empty(state_path)

    runs = v3.setdefault("runs", {})
    if not isinstance(runs, dict):
        runs = {}
        v3["runs"] = runs

    # Add empty entry if missing.  Don't clobber an existing one.
    is_fresh = slug not in runs
    if is_fresh:
        runs[slug] = _empty_per_run_state()
        if init_spec is not None:
            try:
                resolved = resolve_init_spec_to_checkpoint(
                    ws,
                    init_spec,
                    active_run_id=slug,
                    target_bucket=target_bucket,
                )
            except Exception:
                resolved = None
            overrides = init_spec_to_model_state_overrides(
                init_spec, resolved_checkpoint=resolved
            )
            apply_overrides_to_run_state(runs[slug], overrides)

    if activate:
        v3["active_run_id"] = slug

    _atomic_write_json(state_path, v3)
    return v3


def set_active_run_in_state_json(
    workspace: Path | str,
    slug: str,
) -> Dict[str, Any]:
    """Set ``active_run_id = slug`` in step2_state.json (must already exist).

    Args:
        workspace: workspace root.
        slug: run ID; must already have a ``runs[{slug}]`` entry.

    Returns:
        The updated v3 state dict.

    Raises:
        RunNotFoundError: if state file doesn't exist OR slug isn't in
            ``runs``.
    """
    _validate_or_raise(slug)
    ws = Path(workspace)
    state_path = _step2_paths.state_json_path(ws)
    if not state_path.is_file():
        raise RunNotFoundError(
            f"no step2_state.json at {state_path} — can't set active run"
        )
    v3 = _load_v3_or_empty(state_path)
    runs = v3.get("runs")
    if not isinstance(runs, dict) or slug not in runs:
        raise RunNotFoundError(
            f"run {slug!r} not in step2_state.json (known: {sorted((runs or {}).keys())})"
        )
    v3["active_run_id"] = slug
    _atomic_write_json(state_path, v3)
    return v3


def get_active_run_id(workspace: Path | str) -> str:
    """Read ``active_run_id`` from step2_state.json, defaulting to "default"."""
    ws = Path(workspace)
    state_path = _step2_paths.state_json_path(ws)
    if not state_path.is_file():
        return _step2_paths.DEFAULT_RUN_ID
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            v3 = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return _step2_paths.DEFAULT_RUN_ID
    if not isinstance(v3, dict):
        return _step2_paths.DEFAULT_RUN_ID
    val = v3.get("active_run_id")
    return str(val) if val else _step2_paths.DEFAULT_RUN_ID


def remove_run_from_state_json(
    workspace: Path | str,
    slug: str,
) -> Dict[str, Any]:
    """Remove ``runs[{slug}]`` from step2_state.json (Phase 8c delete).

    If the deleted run was active, falls back to ``"default"`` (which
    may or may not exist — caller must handle that).
    """
    _validate_or_raise(slug)
    ws = Path(workspace)
    state_path = _step2_paths.state_json_path(ws)
    if not state_path.is_file():
        return {}
    v3 = _load_v3_or_empty(state_path)
    runs = v3.get("runs", {})
    if isinstance(runs, dict) and slug in runs:
        del runs[slug]
    if v3.get("active_run_id") == slug:
        v3["active_run_id"] = _step2_paths.DEFAULT_RUN_ID
    _atomic_write_json(state_path, v3)
    return v3


# ----------------------------------------------------------------------------
# Discovery: enriched summaries
# ----------------------------------------------------------------------------

def list_run_summaries(workspace: Path | str) -> List[Dict[str, Any]]:
    """Return one summary per run for the dropdown / Manage panel.

    Each summary has:
        - ``run_id``: slug
        - ``display_name``: human name (falls back to slug if manifest
          missing)
        - ``target``: free-text target (or "")
        - ``backend``: "cellpose" (or whatever's in manifest)
        - ``archived``: bool
        - ``manifest_ok``: bool — False if manifest missing/corrupt
        - ``is_active``: bool — True for the currently-active run
        - ``lateral_round_count``: int (number of trained rounds in
          ``runs/{slug}/lateral/rounds/``)
        - ``axial_round_count``: int
        - ``last_activity_iso``: str | None — latest mtime across the
          run's training artifacts, ISO format

    Defensive across all failure modes — never raises.  Empty list if
    runs root missing.
    """
    ws = Path(workspace)
    runs_root = _step2_paths.step2_runs_root(ws)
    if not runs_root.is_dir():
        return []

    active_id = get_active_run_id(ws)
    entries = list_runs_with_manifests(runs_root)

    out: List[Dict[str, Any]] = []
    for entry in entries:
        slug = entry["run_id"]
        manifest = entry.get("manifest")
        if manifest is None:
            display_name = slug
            target = ""
            backend = "cellpose"
            archived = False
        else:
            display_name = manifest.get("display_name") or slug
            target = manifest.get("target", "") or ""
            backend = manifest.get("backend", "cellpose") or "cellpose"
            archived = is_run_archived(manifest)
            description = manifest.get("description", "") or ""

        if manifest is None:
            description = ""

        lateral_count = _count_rounds(ws, slug, "lateral")
        axial_count = _count_rounds(ws, slug, "axial")
        # 2026-05-25 Group D: accepted-pair count.
        accepted_pair_count = _count_accepted_pairs(ws, slug)
        last_activity = _latest_mtime(ws, slug)

        out.append({
            "run_id": slug,
            "display_name": display_name,
            "target": target,
            "backend": backend,
            "description": description,  # Phase 8c edit dialog pre-populate
            "archived": archived,
            "manifest_ok": entry.get("manifest_ok", False),
            "is_active": (slug == active_id),
            "lateral_round_count": lateral_count,
            "axial_round_count": axial_count,
            "accepted_pair_count": accepted_pair_count,
            "last_activity_iso": last_activity,
            "manifest_path": entry.get("manifest_path", ""),
        })
    return out


# ----------------------------------------------------------------------------
# Provenance dependency check (Phase 8c — block delete if inherited from)
# ----------------------------------------------------------------------------

def list_run_model_sources(
    workspace: Path | str,
    exclude_run_id: Optional[str] = None,
    include_archived: bool = False,
) -> List[Dict[str, Any]]:
    """List all Step 2 trained models in this workspace that could be used
    as init sources for a new run (Phase 8b).

    Walks ``runs/{run_id}/{bucket}/models/round_NN/weights`` for every
    run except ``exclude_run_id`` (typically the run currently being
    created — can't init from yourself).

    Args:
        workspace: workspace root.
        exclude_run_id: skip this run's models (typically the new run's
            slug — but new run won't have models yet anyway, this is
            mostly defensive).
        include_archived: if False (default), skip runs whose manifest
            is marked archived.

    Returns:
        List of dicts, each with:
            - run_id: source run slug
            - run_display_name: from run manifest, or run_id
            - bucket: "lateral" | "axial"
            - round_id: "round_NN"
            - weights_path: absolute path to weights file
            - mtime_iso: ISO timestamp of last modification (None if unknowable)
            - exists: True if weights file is currently on disk
        Sorted by (run_id, bucket, round_id) — caller can re-sort.
    """
    ws = Path(workspace)
    runs_root = _step2_paths.step2_runs_root(ws)
    out: List[Dict[str, Any]] = []

    # Walk canonical multi-run layout if present (Phase 6c-deferred
    # workspaces still have models at the legacy path — handled below
    # after this block).
    if not runs_root.is_dir():
        # No new-layout runs yet, but legacy models may still exist
        # (audit-fix 2026-05-23).  Fall through to the legacy scan.
        return _scan_legacy_step2_models(ws, exclude_run_id=exclude_run_id)

    for entry in list_runs_with_manifests(runs_root):
        run_id = entry["run_id"]
        if exclude_run_id and run_id == exclude_run_id:
            continue
        manifest = entry.get("manifest")
        if not include_archived and is_run_archived(manifest):
            continue
        display = (manifest or {}).get("display_name", run_id) if manifest else run_id

        for bucket in ("lateral", "axial"):
            # Phase B.2 (2026-05-25): dual-path scan.
            # NEW layout: model dirs at `{bucket}/round_NN/model/`
            # LEGACY split layout: model dirs at `{bucket}/models/round_NN/`
            # Build a unified list of `(round_id, round_dir)` pairs from
            # BOTH sources.  When a round_NN exists in BOTH locations
            # (e.g. partial migration), prefer the NEW-layout entry
            # (skip the legacy duplicate by round_id).
            try:
                bucket_root_path = _step2_paths.bucket_root(ws, run_id, bucket)
            except (ValueError, TypeError):
                continue

            seen_round_ids: set = set()
            round_pairs: List[tuple] = []  # (round_id_str, round_dir_path)

            # NEW layout: bucket_root.iterdir() finds round_NN dirs
            # directly; each has a `model/` subdir.
            if bucket_root_path.is_dir():
                for rd in sorted(bucket_root_path.iterdir()):
                    if not rd.is_dir() or not rd.name.startswith("round_"):
                        continue
                    model_sub = rd / "model"
                    if not model_sub.is_dir():
                        continue
                    round_pairs.append((rd.name, model_sub))
                    seen_round_ids.add(rd.name)

            # LEGACY split layout: scan `{bucket}/models/round_NN/`
            # ONLY for round_ids not already found in the new layout.
            try:
                legacy_models_root = _step2_paths.bucket_models_root(ws, run_id, bucket)
            except (ValueError, TypeError):
                legacy_models_root = None
            if legacy_models_root and legacy_models_root.is_dir():
                for rd in sorted(legacy_models_root.iterdir()):
                    if not rd.is_dir() or not rd.name.startswith("round_"):
                        continue
                    if rd.name in seen_round_ids:
                        # Prefer the new-layout duplicate
                        continue
                    round_pairs.append((rd.name, rd))
                    seen_round_ids.add(rd.name)

            for round_id, round_dir in round_pairs:
                # 2026-05-26 audit C-1 fix: the actual weights filename
                # in new layout is `{run_id}_{bucket}` (per Phase 6c
                # followup `7a36217`), NOT the legacy `weights` constant
                # the old code hardcoded.  Read manifest.json (schema
                # 2.0) to get the canonical filename.  If no manifest
                # (training mid-flight, lift failed, etc.) fall back to
                # scanning the model dir for any non-companion file.
                weights_filename = None
                inference_failed = False
                manifest_path = round_dir / "manifest.json"
                if manifest_path.is_file():
                    try:
                        from packages.epicycle_training.step2_model_manifest import (
                            read_step2_model_manifest,
                        )
                        manifest = read_step2_model_manifest(manifest_path)
                        weights_filename = manifest.get("weights_filename")
                        # Release-hardening 1b#7 (2026-09-07): give the
                        # `inference_failed` flag a reader.  The weights are
                        # real and usable, but that round's post-train
                        # inference never ran, so predictions from it were
                        # never sanity-checked — worth seeing in the picker.
                        inference_failed = bool(manifest.get("inference_failed"))
                    except Exception as _mre:
                        # Commit 1 audit LOW finding: silent except meant
                        # a corrupt manifest in production was invisible.
                        # Falls through to fallback scan — log so the
                        # diagnostic is visible.
                        print(f"[WARN] manifest read failed at "
                              f"{manifest_path}: {type(_mre).__name__}: {_mre}")
                        weights_filename = None
                if not weights_filename:
                    # Fallback scanner: pick the first non-companion,
                    # non-excluded file.  See module-level
                    # `_FALLBACK_EXCLUDED_NAMES` + `_FALLBACK_EXCLUDED_SUFFIXES`
                    # for what's excluded and why.
                    # Sort + dotfile/backup skip is Commit 1 audit MEDIUM
                    # fix — without it, `round_dir.iterdir()` order is
                    # filesystem-dependent and could return OS metadata
                    # files (`.DS_Store`, `Thumbs.db`) first.
                    for child in sorted(round_dir.iterdir(), key=lambda p: p.name):
                        if not child.is_file():
                            continue
                        nm = child.name
                        if nm in _FALLBACK_EXCLUDED_NAMES:
                            continue
                        if nm.startswith(".") or nm.startswith("~"):
                            continue
                        if nm.endswith((".tmp", ".bak")):
                            continue
                        if any(nm.endswith(suf) for suf in _FALLBACK_EXCLUDED_SUFFIXES):
                            continue
                        weights_filename = nm
                        break
                if not weights_filename:
                    # No manifest + no recognizable weights file; skip.
                    continue
                weights_path = round_dir / weights_filename
                exists = weights_path.is_file()
                mtime_iso = None
                if exists:
                    try:
                        mtime = weights_path.stat().st_mtime
                        from datetime import datetime, timezone
                        mtime_iso = datetime.fromtimestamp(
                            mtime, tz=timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ")
                    except OSError:
                        pass
                out.append({
                    "run_id": run_id,
                    "run_display_name": display,
                    # 1b#7: display-only label; `run_display_name` stays the
                    # clean identity string other consumers compare against.
                    "display_label": (
                        f"{display} ⚠ inference failed"
                        if inference_failed else display
                    ),
                    "inference_failed": inference_failed,
                    "bucket": bucket,
                    "round_id": round_id,
                    "weights_path": str(weights_path),
                    "weights_filename": weights_filename,
                    "mtime_iso": mtime_iso,
                    "exists": exists,
                })

    # Dual-path scan (audit-found CRITICAL 2026-05-23): Phase 6c is
    # deferred — Cellpose-trained models post-migration STILL land in
    # legacy `artifacts/models/round_NN/` paths.  Without scanning those,
    # users on migrated workspaces see empty Run model picker even though
    # they have trained models.  Mirror Step 3's _discover_workspace_models
    # dual-path approach.
    out.extend(_scan_legacy_step2_models(ws, exclude_run_id=exclude_run_id))
    return out


def _scan_legacy_step2_models(
    workspace: Path,
    exclude_run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scan legacy Step 2 model paths for run-model init sources.

    Phase 6c is deferred — Cellpose-trained models post-migration land
    in the legacy ``artifacts/models/round_NN/<model_name>`` path
    pattern.  This scanner finds them and emits entries with run_id =
    "default" (since legacy paths are not run-aware) + bucket inferred
    from the model filename (Cellpose preserves the bucket-keyword
    naming convention).

    Args:
        workspace: workspace root.
        exclude_run_id: skip if it's "default" (legacy → default only).

    Returns:
        List of source dicts in the same shape as ``list_run_model_sources``.
        Empty list if legacy models dir missing.
    """
    if exclude_run_id == "default":
        return []
    # Legacy Step 2 path: {ws}/models/round_NN/[models/]<model_name>.
    # Cellpose auto-creates a 'models/' subdir under round_NN; some
    # workspaces have files DIRECTLY in round_NN (older HITL versions)
    # and some have them under round_NN/models/.  Walk both.
    legacy_root = workspace / "models"
    if not legacy_root.is_dir():
        return []

    out: List[Dict[str, Any]] = []
    for round_dir in sorted(legacy_root.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
            continue
        round_id = round_dir.name
        # Two possible locations: round_NN/ directly, or round_NN/models/
        candidate_dirs = [round_dir]
        nested_models = round_dir / "models"
        if nested_models.is_dir():
            candidate_dirs.append(nested_models)

        for scan_dir in candidate_dirs:
            for model_file in sorted(scan_dir.iterdir(), key=lambda p: p.name):
                if not model_file.is_file():
                    continue
                # Skip loss companions (.npy files for train_loss / val_loss)
                if model_file.suffix in {".npy", ".json"}:
                    continue
                # Commit 2 audit MEDIUM-1 fix (2026-05-26): skip dotfiles
                # (`.DS_Store`, `Thumbs.db`) + editor backups (`~weights`,
                # `weights.tmp`, `weights.bak`).  Without this, OS metadata
                # or stale editor state could be surfaced as a "model".
                nm = model_file.name
                if nm.startswith(".") or nm.startswith("~"):
                    continue
                if nm.endswith((".tmp", ".bak")):
                    continue
                name_lower = nm.lower()
                if ("spatial" in name_lower or "lateral" in name_lower
                        or "_xy" in name_lower):
                    bucket = "lateral"
                elif ("axial" in name_lower or "_xz" in name_lower
                        or "_yz" in name_lower):
                    bucket = "axial"
                else:
                    # Unknown bucket — skip (don't misattribute)
                    continue
                try:
                    mtime = model_file.stat().st_mtime
                    from datetime import datetime, timezone
                    mtime_iso = datetime.fromtimestamp(
                        mtime, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                except OSError:
                    mtime_iso = None
                out.append({
                    "run_id": "default",
                    "run_display_name": "Default (legacy path)",
                    "bucket": bucket,
                    "round_id": round_id,
                    "weights_path": str(model_file),
                    "mtime_iso": mtime_iso,
                    "exists": True,
                })
    return out


def list_step3_refined_sources(workspace: Path | str) -> List[Dict[str, Any]]:
    """List Step 3 refinement bundles in this workspace that could be
    used as init sources for a Step 2 run (Phase 8b).

    Scans ``artifacts/step3/refinement/models/`` for bundle directories,
    each of which may have lateral + axial weights.

    Args:
        workspace: workspace root.

    Returns:
        List of dicts, each with:
            - bundle_name: directory name under refinement/models/
            - bucket: "lateral" | "axial"
            - weights_path: absolute path to the per-bucket weights file
            - mtime_iso: ISO timestamp or None
            - exists: True if weights file on disk
        Defensive — empty list if Step 3 dir missing or not a valid layout.
    """
    ws = Path(workspace)
    # Hardcoded path — Step 3's _paths module owns the canonical location
    # but importing it here would couple Step 2 to Step 3 internals.
    # The path is stable + documented; if it ever moves, this scanner
    # falls back to empty list (no crash).
    step3_refined_root = ws / "artifacts" / "step3" / "refinement" / "models"
    if not step3_refined_root.is_dir():
        return []

    out: List[Dict[str, Any]] = []
    for bundle_dir in sorted(step3_refined_root.iterdir()):
        if not bundle_dir.is_dir() or bundle_dir.name.startswith("."):
            continue
        for bucket in ("lateral", "axial"):
            weights_path = bundle_dir / bucket
            # Step 3 refined bundles store weights as a file with the
            # bucket name (no extension) — see cellpose_lift module.
            if not weights_path.is_file():
                continue
            try:
                mtime = weights_path.stat().st_mtime
                from datetime import datetime, timezone
                mtime_iso = datetime.fromtimestamp(
                    mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            except OSError:
                mtime_iso = None
            out.append({
                "bundle_name": bundle_dir.name,
                "bucket": bucket,
                "weights_path": str(weights_path),
                "mtime_iso": mtime_iso,
                "exists": True,
            })
    return out


def _extract_round_idx(round_id: str) -> int:
    """Extract integer N from 'round_NN' string."""
    # Strip 'round_' prefix and convert; raises ValueError on malformed
    if not round_id.startswith("round_"):
        raise ValueError(f"expected 'round_NN' format, got {round_id!r}")
    return int(round_id[len("round_"):])


def resolve_init_spec_to_checkpoint(
    workspace: Path | str,
    init_spec: Optional[Dict[str, Any]],
    active_run_id: str = "",
    target_bucket: Optional[str] = None,
) -> Optional[str]:
    """Resolve a run's ``created_from`` init_spec to an on-disk checkpoint
    path (Commit 2 audit H1 fix, 2026-05-26).

    For Phase 8b/c init modes that point at a real file
    (``inherited``, ``cross_bucket``, ``step3_refined``, ``browse``,
    ``fork``), this helper returns an absolute path string that can be
    passed straight to Cellpose as ``pretrained_model``.  For modes
    that don't have a file path (``scratch``, ``pretrained`` Cellpose
    base), returns ``None``.

    Why this exists: prior to this fix, ``_create_new_run`` wrote the
    ``init_spec`` to ``run_manifest.json::created_from`` but never wired
    it into training.  On round 0 of an "inherited from run X" run,
    training silently started from scratch (or cyto3 base) — a
    TRAINING CORRECTNESS bug from Phase 8b.

    Args:
        workspace: workspace root.
        init_spec: the ``created_from`` dict from ``run_manifest.json``,
            or ``None`` for legacy runs without one.
        active_run_id: the run this resolution is for.  Used for
            ``cross_bucket`` mode (no ``source_run_id`` in spec; it's
            implicit = active run).
        target_bucket: the bucket of the run being initialized.  Used
            for ``fork`` mode (no source_bucket in spec) to PREFER
            the target bucket of the source run over mtime ordering.
            Prevents Cellpose lateral init from a source's axial
            weights just because axial trained more recently.

    Returns:
        Absolute checkpoint path string, or ``None`` if:
            - mode is scratch / pretrained (no file)
            - spec is malformed / missing required fields
            - resolved path doesn't exist on disk

        ``None`` is the "use config.model_type or scratch" signal at
        the training-call site.
    """
    if not isinstance(init_spec, dict):
        return None
    mode = str(init_spec.get("mode", "scratch"))
    ws = Path(workspace)

    if mode in ("scratch", "pretrained"):
        # No file path — caller falls through to config["model_type"]
        # (Cellpose base) or scratch (random weights).
        return None

    if mode == "browse":
        # External file the user picked at run-create time.
        ext = init_spec.get("external_path")
        if ext and Path(ext).is_file():
            return str(ext)
        return None

    if mode == "step3_refined":
        # Step 3 refinement bundle in this workspace.  Lateral + axial
        # weights live at `artifacts/step3/refinement/models/{bundle}/
        # {bucket}` (no extension).  Spec stores bucket explicitly.
        # Audit 1 H-B / Audit 2 M-1 fix (2026-05-26): use the canonical
        # Step 3 path helper instead of hardcoded path — if Step 3's
        # layout shifts (Tier work in flight), hardcoded breaks
        # silently.
        bundle = init_spec.get("source_step3_bundle")
        bucket = init_spec.get("source_step3_bucket")
        if not (bundle and bucket):
            return None
        try:
            # Audit 4/5 LOW: prefer `refined_model_weights_path`
            # (single canonical Step 3 symbol returning the weights
            # file directly) over `refined_model_dir(...) / bucket`
            # so if Step 3 ever adds an extension or restructures the
            # bundle's per-bucket layout, this resolver picks it up
            # without a second edit.
            #
            # HIGH-2 (layout-agnostic): after a Step-3 default-run "Convert
            # to unified layout", refined bundles live at the NESTED
            # ``runs/default/refinement/models/...`` instead of the flat
            # ``step3/refinement/models/...``.  Check BOTH candidate paths —
            # flat first (run_id=None), then the nested default
            # (run_id="default") — and return the first that exists.  This
            # avoids threading the Step-3 nested flag across the layer
            # boundary; a two-path probe is robust + simple.
            from napari_epicycle.fusion_reconstruction import _paths as _step3_paths
            candidates = [
                _step3_paths.refined_model_weights_path(ws, bundle, bucket),
                _step3_paths.refined_model_weights_path(
                    ws, bundle, bucket, run_id="default"
                ),
            ]
        except Exception:
            # Defensive fallback if Step 3 path module is unavailable.
            # Mirror both layouts the canonical helper would produce.
            base = ws / "artifacts" / "step3"
            candidates = [
                base / "refinement" / "models" / bundle / bucket,
                base / "runs" / "default" / "refinement" / "models" / bundle / bucket,
            ]
        for cand in candidates:
            if cand.is_file():
                return str(cand)
        return None

    if mode in ("inherited", "fork"):
        # Within-workspace cross-run init.  Source can be specified
        # with full triple (run/bucket/round) or partial (just run for
        # `fork`).  For `fork`, no source_round_id is set; we use the
        # source run's terminal trained round (highest round_NN).
        # Audit 1 H-A fix: pass target_bucket as prefer_bucket so fork
        # from Mode=Both source prefers the target bucket over the
        # mtime-latest opposite bucket (lateral target ← lateral source
        # weights, not axial just because axial trained last).
        source_run = init_spec.get("source_run_id")
        if not source_run:
            return None
        source_bucket = init_spec.get("source_bucket")
        source_round = init_spec.get("source_round_id")
        return _resolve_source_run_checkpoint(
            ws, source_run, source_bucket, source_round,
            prefer_bucket=target_bucket,
        )

    if mode == "cross_bucket":
        # Within active run, opposite bucket of same round.  Spec has
        # source_bucket + source_round_id; source_run_id is implicit
        # = active_run_id.
        if not active_run_id:
            return None
        source_bucket = init_spec.get("source_bucket")
        source_round = init_spec.get("source_round_id")
        if not (source_bucket and source_round):
            return None
        return _resolve_source_run_checkpoint(
            ws, active_run_id, source_bucket, source_round
        )

    # Unknown mode — defensive: no resolution
    return None


def _resolve_source_run_checkpoint(
    workspace: Path,
    source_run: str,
    source_bucket: Optional[str],
    source_round: Optional[str],
    prefer_bucket: Optional[str] = None,
) -> Optional[str]:
    """Helper: find the checkpoint file for (source_run, source_bucket,
    source_round_id) in the new-layout workspace.

    If ``source_bucket`` is None, tries both buckets.  When
    ``prefer_bucket`` is supplied (e.g. the TARGET bucket of the
    inheriting run), it's tried FIRST and wins ties — preventing a
    Mode=Both source from initializing a lateral target with axial
    weights just because axial trained more recently (Cellpose models
    are bucket-specialized).  Without prefer_bucket, the older
    mtime-latest-wins behavior is preserved.

    If ``source_round`` is None, picks the highest round_NN in the
    bucket.

    Prefers ``manifest.json::weights_filename`` (canonical source of
    truth post-Phase-6c.7); falls back to the canonical Phase 6c
    naming ``{source_run}_{source_bucket}``, then the legacy bare
    ``weights`` file (the latter is intentional asymmetry with
    ``list_run_model_sources``'s `_FALLBACK_EXCLUDED_NAMES`: the
    scanner excludes `weights` when enumerating candidates because
    multiple files in the same round_dir would be ambiguous; the
    resolver accepts it as a last-resort because we know exactly
    which round we want — see Commit 5 audit M-A note).

    Note: archived runs ARE resolved (archive is metadata, not a
    soft-delete).  If the user inherits from run X, then archives X,
    then trains, training correctly inherits from X's weights.

    Returns absolute path string or None.
    """
    if not source_run:
        return None

    # Build bucket order: prefer_bucket first if specified, then the
    # remaining bucket as fallback.  If source_bucket is explicitly set,
    # it takes precedence (caller knows the right bucket).
    if source_bucket:
        buckets_to_try = [source_bucket]
    elif prefer_bucket and prefer_bucket in {"lateral", "axial"}:
        # Audit 1 H-A fix (2026-05-26): cross-bucket fork footgun.
        # Try target bucket first, then opposite as fallback.
        other = "axial" if prefer_bucket == "lateral" else "lateral"
        buckets_to_try = [prefer_bucket, other]
    else:
        buckets_to_try = ["lateral", "axial"]

    # Audit 1 H-A fix: when prefer_bucket is set, return the FIRST
    # match (target-bucket priority) rather than mtime-latest across
    # both.  Without prefer_bucket, fall back to mtime-latest across
    # all candidates (preserves legacy fork behavior when caller can't
    # express a preference).
    use_first_match = prefer_bucket is not None and source_bucket is None

    candidates = []  # list of (mtime, path, bucket_index)
    for bucket_idx, bucket in enumerate(buckets_to_try):
        if bucket not in {"lateral", "axial"}:
            continue
        # Phase B.2 (2026-05-25): dual-path scan over BOTH new-layout
        # `{bucket}/round_NN/model/` AND legacy split `{bucket}/models/
        # round_NN/`.  For each scan, the inner round-selection logic
        # picks the rounds to consider.

        # Collect candidate round dirs from both layouts.  Each entry
        # is `(round_idx, round_dir_path)` where round_dir_path is the
        # MODEL dir (containing weights + manifest).
        scan_locations: List[tuple] = []  # (round_idx, model_dir)

        # NEW layout: `{bucket}/round_NN/model/`
        try:
            new_bucket_root = _step2_paths.bucket_root(workspace, source_run, bucket)
        except (ValueError, TypeError):
            new_bucket_root = None
        if new_bucket_root and new_bucket_root.is_dir():
            for rd in new_bucket_root.iterdir():
                if not rd.is_dir() or not rd.name.startswith("round_"):
                    continue
                try:
                    ridx = _extract_round_idx(rd.name)
                except ValueError:
                    continue
                model_sub = rd / "model"
                if not model_sub.is_dir():
                    continue
                scan_locations.append((ridx, model_sub))

        # LEGACY split layout: `{bucket}/models/round_NN/`
        try:
            legacy_models_dir = _step2_paths.bucket_models_root(workspace, source_run, bucket)
        except (ValueError, TypeError):
            legacy_models_dir = None
        if legacy_models_dir and legacy_models_dir.is_dir():
            for rd in legacy_models_dir.iterdir():
                if not rd.is_dir() or not rd.name.startswith("round_"):
                    continue
                try:
                    ridx = _extract_round_idx(rd.name)
                except ValueError:
                    continue
                # Skip if new-layout already covers this round_idx
                if any(r == ridx for r, _ in scan_locations):
                    continue
                scan_locations.append((ridx, rd))

        if not scan_locations:
            continue

        # Round: if None, pick highest round_NN across both layouts
        if source_round:
            # Audit 2 H-2 fix (2026-05-26): user-touchable manifest
            # could have non-string source_round (e.g. int 3).
            try:
                target_round_idx = _extract_round_idx(str(source_round))
            except (ValueError, AttributeError, TypeError) as _re:
                print(f"[WARN] resolve: bad source_round "
                      f"{source_round!r} ({type(_re).__name__}); skipping "
                      f"bucket={bucket}")
                continue
            scan_locations = [
                (ridx, rd) for ridx, rd in scan_locations
                if ridx == target_round_idx
            ]
        else:
            # Pick highest round_idx across both layouts
            max_idx = max(ridx for ridx, _ in scan_locations)
            scan_locations = [
                (ridx, rd) for ridx, rd in scan_locations if ridx == max_idx
            ]

        for round_idx, round_dir in scan_locations:
            if not round_dir.is_dir():
                continue
            # Prefer manifest.json::weights_filename
            cand_path = None
            mp = round_dir / "manifest.json"
            if mp.is_file():
                try:
                    from packages.epicycle_training.step2_model_manifest import (
                        read_step2_model_manifest,
                    )
                    m = read_step2_model_manifest(mp)
                    wf = m.get("weights_filename")
                    if wf:
                        c = round_dir / wf
                        if c.is_file():
                            cand_path = c
                except Exception as _e:
                    # Audit 2 L-1 fix (2026-05-26): diagnostic instead
                    # of silent fall-through.  Falls back to canonical
                    # naming — still functional.
                    print(f"[WARN] resolve: manifest read failed at "
                          f"{mp}: {type(_e).__name__}: {_e}")
            # Phase B.2.1 fix (audit 3 MEDIUM): try Phase B.2 naming
            # FIRST (post-B.2 weights file is `{run}_{bucket}_round_NN`).
            # Without this, a B.2 trained model with deleted manifest
            # would silently fall back to either pre-B.2 name (which
            # doesn't exist) or legacy `weights` (also missing) → resolver
            # returns None → silent inheritance miss.
            if cand_path is None:
                c = round_dir / f"{source_run}_{bucket}_round_{round_idx:02d}"
                if c.is_file():
                    cand_path = c
            # Fallback: pre-B.2 Phase 6c naming
            if cand_path is None:
                c = round_dir / f"{source_run}_{bucket}"
                if c.is_file():
                    cand_path = c
            # Legacy bare "weights" (audit M-A: deliberate asymmetry
            # with `_FALLBACK_EXCLUDED_NAMES` in `list_run_model_sources`
            # — see docstring).
            if cand_path is None:
                c = round_dir / "weights"
                if c.is_file():
                    cand_path = c
            if cand_path is not None:
                try:
                    candidates.append((
                        cand_path.stat().st_mtime, cand_path, bucket_idx
                    ))
                except OSError:
                    continue
        # Audit 1 H-A fix: if prefer_bucket was specified + we found a
        # candidate in this bucket, stop searching.  Don't fall through
        # to the opposite bucket when target bucket has a match.
        if use_first_match and any(c[2] == bucket_idx for c in candidates):
            break

    if not candidates:
        return None
    # If prefer_bucket-priority is active, the loop above already
    # returns only target-bucket candidates (or fallback if target had
    # none).  Sort by mtime within whatever set we have.
    candidates.sort(key=lambda x: x[0], reverse=True)
    return str(candidates[0][1])


def archive_run(workspace: Path | str, slug: str) -> None:
    """Mark a run as archived in its run_manifest.json (Phase 8c).

    Sets ``archived_at`` to the current UTC timestamp.  The run is NOT
    moved on disk — Phase 8c keeps the directory in-place; the flag is
    purely metadata.  ``list_run_summaries`` reads the flag + the UI
    groups archived runs at the bottom of the dropdown.

    Raises:
        InvalidRunIdError: if slug is malformed.
        RunNotFoundError: if the run dir or manifest doesn't exist.
    """
    _validate_or_raise(slug)
    ws = Path(workspace)
    run_dir = _step2_paths.run_root(ws, slug)
    if not run_dir.is_dir():
        raise RunNotFoundError(f"run not found: {run_dir}")
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    manifest = read_run_manifest(manifest_path)
    if manifest is None:
        raise RunNotFoundError(
            f"run manifest missing or corrupt: {manifest_path}"
        )
    manifest["archived_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    write_run_manifest(manifest_path, manifest)


def unarchive_run(workspace: Path | str, slug: str) -> None:
    """Clear the archived_at flag on a run (Phase 8c).  Inverse of archive_run.

    Raises:
        InvalidRunIdError / RunNotFoundError: as archive_run.
    """
    _validate_or_raise(slug)
    ws = Path(workspace)
    manifest_path = _step2_paths.run_root(ws, slug) / RUN_MANIFEST_FILENAME
    manifest = read_run_manifest(manifest_path)
    if manifest is None:
        raise RunNotFoundError(
            f"run manifest missing or corrupt: {manifest_path}"
        )
    manifest["archived_at"] = None
    write_run_manifest(manifest_path, manifest)


def edit_run_metadata(
    workspace: Path | str,
    slug: str,
    *,
    display_name: Optional[str] = None,
    target: Optional[str] = None,
    description: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Edit user-mutable fields on a run's manifest (Phase 8c).

    Slug + backend + created_at + created_from are immutable post-
    creation (slug is the directory name; backend + init source define
    the run's identity).  display_name / target / description / notes
    can be freely edited via the Manage Runs panel.

    Passing ``None`` for a field leaves it unchanged.  Pass empty
    string "" to clear an optional field (target / description / notes).
    display_name CANNOT be cleared — must be a non-empty string.

    Returns:
        The updated manifest dict (post-write).

    Raises:
        InvalidRunIdError / RunNotFoundError: standard.
        ValueError: if display_name is set to empty.
    """
    _validate_or_raise(slug)
    ws = Path(workspace)
    manifest_path = _step2_paths.run_root(ws, slug) / RUN_MANIFEST_FILENAME
    manifest = read_run_manifest(manifest_path)
    if manifest is None:
        raise RunNotFoundError(
            f"run manifest missing or corrupt: {manifest_path}"
        )
    if display_name is not None:
        if not display_name.strip():
            raise ValueError("display_name cannot be empty")
        manifest["display_name"] = display_name
    if target is not None:
        manifest["target"] = target
    if description is not None:
        manifest["description"] = description
    if notes is not None:
        manifest["notes"] = notes
    write_run_manifest(manifest_path, manifest)
    return manifest


def find_runs_inheriting_from(
    workspace: Path | str,
    target_slug: str,
) -> List[str]:
    """Find runs whose ``created_from`` references ``target_slug``.

    Used by Phase 8c's delete flow to block destructive removal of a
    run that other runs depend on for their init.

    Returns:
        List of slugs (other runs) that inherited from ``target_slug``.
        Empty list if none.
    """
    ws = Path(workspace)
    runs_root = _step2_paths.step2_runs_root(ws)
    if not runs_root.is_dir():
        return []

    dependents: List[str] = []
    for entry in list_runs_with_manifests(runs_root):
        slug = entry["run_id"]
        if slug == target_slug:
            continue
        manifest = entry.get("manifest")
        if not isinstance(manifest, dict):
            continue
        created_from = manifest.get("created_from") or {}
        if not isinstance(created_from, dict):
            continue
        # Check both "inherited" (run model init) and "fork" (data import)
        if created_from.get("source_run_id") == target_slug:
            dependents.append(slug)
    return sorted(dependents)


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

def _validate_or_raise(slug: str) -> None:
    """Validate slug via ``_paths.validate_run_id``, raising
    InvalidRunIdError on failure (translates ValueError to a domain
    error so callers can catch it specifically)."""
    try:
        _step2_paths.validate_run_id(slug)
    except ValueError as e:
        raise InvalidRunIdError(str(e)) from e


def _empty_per_run_state() -> Dict[str, Any]:
    """Empty per-run state dict — what a fresh run looks like."""
    return {
        "current_round": 0,
        "current_round_by_view": {"XY": 0, "XZ": 0, "YZ": 0},
        "training_data_by_round": {},
        "losses_by_round": {},
        "model_checkpoint_paths": {},
        "round_backend_info": {},
        "current_config": {},
        "selected_view": "XY",
        "view_target": "lateral",
        "axial_plane": "XZ",
        "sample_records": {},
        "active_curation_checkpoints": {"lateral": "", "axial": ""},
        "curation_model_source_mode": "auto",
        "imported_curation_model_path": "",
        "curation_pretrained_model_type": "cyto3",
        "training_mode": "cumulative",
    }


def init_spec_to_model_state_overrides(
    init_spec: Optional[Any],
    *,
    resolved_checkpoint: Optional[str] = None,
) -> Dict[str, str]:
    """Map a New-Run ``init_spec`` (= ``run_manifest.json::created_from``)
    to a dict of per-run-state field overrides for the Inference Model
    section.

    Returned dict (only the keys that apply for the mode):
      * ``curation_model_source_mode`` — ``"imported"`` or ``"pretrained"``.
      * ``imported_curation_model_path`` — file path (Imported modes).
      * ``curation_pretrained_model_type`` — Cellpose builtin name.
      * ``starting_checkpoint`` — cosmetic Training Checkpoint pre-pop
        for Imported modes.  Round-0 training_spec wins regardless
        (Phase 6c.7), but the widget value drives the "what will training
        start from" display.

    Empty dict means "no overrides" (scratch, malformed spec, or an
    unresolvable inherited/fork path).

    Args:
        init_spec: ``created_from`` dict from ``run_manifest.json``, or
            ``None``.
        resolved_checkpoint: pre-resolved path from
            :func:`resolve_init_spec_to_checkpoint`, required for the
            modes that depend on workspace context
            (``inherited``, ``cross_bucket``, ``step3_refined``, ``fork``).
    """
    if not isinstance(init_spec, dict):
        return {}
    mode = str(init_spec.get("mode", "scratch"))
    if mode == "scratch":
        return {}
    if mode == "browse":
        path = str(init_spec.get("external_path") or "").strip()
        if not path:
            return {}
        return {
            "curation_model_source_mode": "imported",
            "imported_curation_model_path": path,
            "starting_checkpoint": path,
        }
    if mode == "pretrained":
        name = str(init_spec.get("pretrained_name") or "").strip()
        if not name:
            return {}
        return {
            "curation_model_source_mode": "pretrained",
            "curation_pretrained_model_type": name,
        }
    if mode in ("inherited", "cross_bucket", "step3_refined", "fork"):
        path = str(resolved_checkpoint or "").strip()
        if not path:
            return {}
        return {
            "curation_model_source_mode": "imported",
            "imported_curation_model_path": path,
            "starting_checkpoint": path,
        }
    return {}


def apply_overrides_to_run_state(
    state: Dict[str, Any],
    overrides: Dict[str, str],
) -> None:
    """Apply Model-section field overrides into a per-run state dict, but
    only for fields still at their empty / default value.  Mutates the
    state in place.

    Don't-clobber rule: each field is written only if its current value
    matches the "fresh empty state" default produced by
    :func:`_empty_per_run_state`.  Re-applying overrides on a run the
    user has since adjusted leaves the user's choices intact.

    Used at New Run creation to pre-populate the Inference Model section
    from the init_spec the user picked in the dialog.
    """
    if not isinstance(overrides, dict) or not overrides:
        return
    defaults = {
        "curation_model_source_mode": "auto",
        "imported_curation_model_path": "",
        "curation_pretrained_model_type": "cyto3",
    }
    current_config = state.setdefault("current_config", {})
    if not isinstance(current_config, dict):
        current_config = {}
        state["current_config"] = current_config
    for key, value in overrides.items():
        if key == "starting_checkpoint":
            if not str(current_config.get(key) or "").strip():
                current_config[key] = value
        elif key in defaults:
            if str(state.get(key) or "") == defaults[key]:
                state[key] = value
        # Audit hardening (2026-05-29): unknown keys silently IGNORED.
        # The mapper produces only the 4 known keys above; an unknown key
        # would mean a future mode added a field without updating the
        # `defaults` table, which is a bug — silently clobbering user state
        # is the worse failure mode, so we drop it instead.


def _load_v3_or_empty(state_path: Path) -> Dict[str, Any]:
    """Load v3 from disk, or return a minimal empty v3 envelope."""
    if state_path.is_file():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Ensure required keys exist (defensive against partial files)
                data.setdefault("schema_version", 3)
                data.setdefault("runs", {})
                data.setdefault("active_run_id", _step2_paths.DEFAULT_RUN_ID)
                return data
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return {
        "schema_version": 3,
        "active_run_id": _step2_paths.DEFAULT_RUN_ID,
        "runs": {},
    }


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomic JSON write (tempfile + os.replace).  Mirror migration code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".step2_state_", suffix=".json.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _count_rounds(workspace: Path, slug: str, bucket: str) -> int:
    """Count TRAINED rounds for a run/bucket.

    Release-hardening 2026-09-07 (bucket-switch item #10): this used to
    count round DIRECTORIES, so the run picker's ``(lat:N ax:M)`` was one
    ahead of the status bar as soon as a round dir existed for the round
    the user is currently annotating (no model in it yet).  A round counts
    only once it has a ``model/`` subdir -- i.e. it has actually trained.

    Phase B.2.1 fix (2026-05-25, audit 2 HIGH): pre-fix this read only
    the LEGACY SPLIT layout (`{bucket}/rounds/round_NN/`).  Post-Phase-
    B.2 round dirs live FLAT at `{bucket}/round_NN/`.  A run trained
    purely on the flat layout would report 0 rounds — the user-visible
    Run-dropdown label `(lat:N ax:M)` shows 0 everywhere.

    Dual-path scan: count unique round_NN entries across BOTH the new
    flat layout AND the legacy split layout.
    """
    seen_round_names: set = set()
    # NEW FLAT layout: `{bucket}/round_NN/` directly under bucket_root
    try:
        b_root = _step2_paths.bucket_root(workspace, slug, bucket)
    except (ValueError, TypeError):
        b_root = None
    if b_root and b_root.is_dir():
        for p in b_root.iterdir():
            if p.is_dir() and p.name.startswith("round_") and (p / "model").is_dir():
                seen_round_names.add(p.name)
    # LEGACY SPLIT layout: `{bucket}/rounds/round_NN/`
    try:
        rounds_root = _step2_paths.bucket_rounds_root(workspace, slug, bucket)
    except (ValueError, TypeError):
        rounds_root = None
    if rounds_root and rounds_root.is_dir():
        for p in rounds_root.iterdir():
            if p.is_dir() and p.name.startswith("round_") and (p / "model").is_dir():
                seen_round_names.add(p.name)
    return len(seen_round_names)


def _count_accepted_pairs(workspace: Path, slug: str) -> int:
    """Count accepted img+mask pairs across all rounds of all buckets.

    2026-05-25 Group D — surfaces in run-summary dict as
    ``accepted_pair_count`` for the Run dropdown + Manage Runs column.

    Counts ``*_img.tif`` files in `{round_dir}/accepted/`.  A pair is
    exactly one image + one mask, so img-file count == pair count.
    Sums across both buckets and all rounds.

    Phase B.2.1 fix (2026-05-25, audit 2 HIGH): dual-path scan across
    NEW FLAT (`{bucket}/round_NN/`) and LEGACY SPLIT (`{bucket}/rounds/
    round_NN/`) layouts.  When same round_NN exists in BOTH, prefer
    the new-layout `accepted/` contents (avoids double-count on
    partial migration).
    """
    total = 0
    for bucket in ("lateral", "axial"):
        seen_round_names: set = set()
        # Collect (round_NN_name, accepted_dir) pairs from both layouts.
        round_accepted_dirs: List[tuple] = []  # (round_name, accepted_path)

        # NEW FLAT layout first (wins on duplicates)
        try:
            b_root = _step2_paths.bucket_root(workspace, slug, bucket)
        except (ValueError, TypeError):
            b_root = None
        if b_root and b_root.is_dir():
            for rd in b_root.iterdir():
                if not rd.is_dir() or not rd.name.startswith("round_"):
                    continue
                accepted = rd / "accepted"
                if accepted.is_dir():
                    round_accepted_dirs.append((rd.name, accepted))
                    seen_round_names.add(rd.name)

        # LEGACY SPLIT layout — only for round names NOT already seen
        try:
            rounds_root = _step2_paths.bucket_rounds_root(workspace, slug, bucket)
        except (ValueError, TypeError):
            rounds_root = None
        if rounds_root and rounds_root.is_dir():
            for rd in rounds_root.iterdir():
                if not rd.is_dir() or not rd.name.startswith("round_"):
                    continue
                if rd.name in seen_round_names:
                    continue
                accepted = rd / "accepted"
                if accepted.is_dir():
                    round_accepted_dirs.append((rd.name, accepted))
                    seen_round_names.add(rd.name)

        for _, accepted_dir in round_accepted_dirs:
            try:
                total += sum(
                    1 for f in accepted_dir.iterdir()
                    if f.is_file() and f.name.endswith("_img.tif")
                )
            except OSError:
                continue
    return total


def _latest_mtime(workspace: Path, slug: str) -> Optional[str]:
    """Latest mtime across the run's tree, as ISO string, or None.

    Walks at most 2 levels deep (run_dir → bucket → rounds) for speed;
    finer granularity isn't useful for "last activity" display.
    """
    run_dir = _step2_paths.run_root(workspace, slug)
    if not run_dir.is_dir():
        return None
    latest: Optional[float] = None
    try:
        for p in run_dir.rglob("*"):
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue
            if latest is None or mt > latest:
                latest = mt
    except OSError:
        return None
    if latest is None:
        return None
    return datetime.fromtimestamp(latest, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    # Exceptions
    "RunAlreadyExistsError",
    "RunNotFoundError",
    "InvalidRunIdError",
    # Disk ops
    "create_run_on_disk",
    "delete_run_from_disk",
    # Phase 8c — metadata mutations
    "archive_run",
    "unarchive_run",
    "edit_run_metadata",
    # State JSON ops
    "add_run_to_state_json",
    "set_active_run_in_state_json",
    "get_active_run_id",
    "remove_run_from_state_json",
    # Discovery
    "list_run_summaries",
    "list_run_model_sources",
    "list_step3_refined_sources",
    # Dependency check
    "find_runs_inheriting_from",
    # Init-spec → Model section pre-population (Bug D, 2026-05-29)
    "init_spec_to_model_state_overrides",
    "apply_overrides_to_run_state",
]

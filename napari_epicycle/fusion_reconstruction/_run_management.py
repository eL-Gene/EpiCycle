"""Pure-function helpers for Step 3 multi-run management (Phase 2).

The Step 3 mirror of ``hitl_training_annotation/_run_management.py``.
Owns the **disk-dir + manifest** side of run lifecycle plus the
**read-only enriched summaries** that the (Phase 4) UI needs.  Keeping
this in a separate pure module (not on the controller) makes it testable
without a Qt + napari + controller stack.

Responsibility split (do NOT add a second writer to ``step3_state.json``):

* **State-JSON run-entry CRUD** (add / remove / set-active run entries in
  the v3 envelope) is OWNED by ``state_manager.ReconstructionStateManager``
  (``create_run`` / ``switch_active_run`` / ``list_run_ids`` / ``has_run``,
  already built Phase 1a).  This module reads ``step3_state.json`` ONLY
  read-only (for the per-run patch-status counts in the summaries).
* **This module** owns the run-context DIRECTORY + its
  ``run_manifest.json``: create / delete the dir, archive / unarchive /
  edit the manifest, scan + summarize runs, and resolve a run's model
  lineage to checkpoint paths.

**The default run is FLAT + manifest-less** (Option A):
it has NO ``runs/default/`` context dir.  Hence:

* :func:`create_run_on_disk` refuses ``slug == "default"`` and refuses a
  slug that collides (case-folded) with ANY existing child of ``runs/`` —
  whether that child is a named-run context OR a default-run per-patch
  reconstruction dir (both live directly under ``runs/``).
* :func:`delete_run_from_disk` / :func:`archive_run` / etc. refuse the
  default run (no context dir / no manifest to operate on).
* :func:`list_run_summaries` SYNTHESIZES the default run's summary from
  ``step3_state.json::runs.default.patches`` (it has no manifest dir to
  scan), then lists the named-run context dirs.

Pure-Python — no Qt / napari dependency..
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from packages.epicycle_io.step3_run_manifest import (
    RUN_MANIFEST_FILENAME,
    build_step3_run_manifest,
    is_run_archived,
    is_run_context_dir,
    list_runs_with_manifests,
    read_run_manifest,
    write_run_manifest,
)

from . import _paths as _step3_paths
from ._migrate_step3_runs import DEFAULT_RUN_ID


# ----------------------------------------------------------------------------
# Exceptions (mirror Step 2)
# ----------------------------------------------------------------------------

class RunAlreadyExistsError(Exception):
    """Raised when create_run sees the target slug already on disk."""


class RunNotFoundError(Exception):
    """Raised when an operation targets a slug that has no context dir."""


class InvalidRunIdError(Exception):
    """Raised when the slug fails ``_paths.validate_run_id``."""


class ModelMaterializationError(Exception):
    """Raised when copy-on-reuse can't resolve OR copy the source model.

    The controller maps this to its M2 rollback (delete the new run's disk
    dir + state entry) so a failed model copy never leaves a half-created
    run behind.
    """


#: Canonical bundle name for a run's copied-on-reuse init model.  A Step 3
#: run starts from exactly ONE model choice, so a single fixed name under the
#: run's ``refinement/models/`` is unambiguous (copy-on-reuse).
INIT_MODEL_NAME = "init_model"

#: Filename for the copied init-model bundle's provenance manifest.  Distinct
#: from a refinement-TRAINED bundle's ``manifest.json`` (schema 2.0) — this
#: one records the copy lineage (``copied_from`` + ``created_from``).
MODEL_MANIFEST_FILENAME = "model_manifest.json"


# ----------------------------------------------------------------------------
# Disk-level: create / delete the run-context directory
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
    model: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Create ``runs/{slug}/`` and write its ``run_manifest.json``.

    Atomic-ish: creates the directory first; if manifest write fails, the
    empty dir is cleaned up (``rmdir``).  Race-safe via
    ``mkdir(exist_ok=False)``.

    **Collision guard:** the slug must NOT case-fold-collide
    with any existing child of ``runs/`` — neither an existing named-run
    context dir NOR an existing DEFAULT-run per-patch reconstruction dir
    (a ``runs/{X}/`` with ``labels_3d`` / ``manifest.json`` but no
    ``run_manifest.json``).  Both live directly under ``runs/``, so a
    case-folded compare against every child name covers both.

    The default run is FLAT (no context dir) — ``slug == "default"`` is
    refused.

    Returns:
        The manifest dict that was written.

    Raises:
        InvalidRunIdError: if slug fails validation.
        RunAlreadyExistsError: slug is "default", or collides (case-folded)
            with an existing ``runs/`` child (run context or default-run
            patch-recon dir).
        ValueError: if build_step3_run_manifest rejects the args.
        OSError: on filesystem failures during directory creation.
    """
    _validate_or_raise(slug)

    if slug.casefold() == DEFAULT_RUN_ID.casefold():
        raise RunAlreadyExistsError(
            f"{slug!r} is the implicit FLAT default run — it has no context "
            f"dir and cannot be created as a named run"
        )

    ws = Path(workspace)
    runs_root = _step3_paths.step3_runs_root(ws)

    # Collision guard: case-folded compare against EVERY existing child of
    # runs/ (named-run contexts AND default-run patch-recon dirs).
    if runs_root.is_dir():
        slug_cf = slug.casefold()
        for child in runs_root.iterdir():
            if not child.is_dir():
                continue
            if child.name.casefold() == slug_cf:
                kind = (
                    "run context"
                    if is_run_context_dir(child)
                    else "default-run reconstruction dir"
                )
                raise RunAlreadyExistsError(
                    f"slug {slug!r} collides (case-folded) with existing "
                    f"{kind} {child.name!r} under {runs_root}"
                )

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
        manifest = build_step3_run_manifest(
            run_id=slug,
            display_name=display_name,
            target=target,
            description=description,
            backend=backend,
            created_from=init_spec,
            model=model,
            notes=notes,
        )
        write_run_manifest(run_dir / RUN_MANIFEST_FILENAME, manifest)
    except Exception:
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
) -> None:
    """Permanently remove ``runs/{slug}/`` from disk.  DESTRUCTIVE.

    Caller MUST pass ``confirm=True`` (after running the provenance
    dependency check + UI confirmation).

    Raises:
        ValueError: if confirm is not True, or slug is the default run
            (the default run is the flat layout — deleting it would wipe
            every patch reconstruction; not this function's job).
        InvalidRunIdError: if slug is malformed.
        RunNotFoundError: if the run-context dir doesn't exist.
    """
    if not confirm:
        raise ValueError(
            "delete_run_from_disk requires confirm=True; caller did not "
            "complete the provenance check"
        )
    _validate_or_raise(slug)
    if slug.casefold() == DEFAULT_RUN_ID.casefold():
        raise ValueError(
            "refusing to delete the default run — it is the FLAT layout, "
            "not a named-run context dir"
        )
    ws = Path(workspace)
    run_dir = _step3_paths.run_root(ws, slug)
    if not run_dir.is_dir():
        raise RunNotFoundError(f"run not found: {run_dir}")
    shutil.rmtree(run_dir)


# ----------------------------------------------------------------------------
# Manifest mutations: archive / unarchive / edit metadata
# ----------------------------------------------------------------------------

def archive_run(workspace: Path | str, slug: str) -> None:
    """Mark a run archived in its ``run_manifest.json`` (sets ``archived_at``).

    The run is NOT moved on disk — the flag is purely metadata.

    Raises:
        InvalidRunIdError: malformed slug.
        RunNotFoundError: default run (no manifest), or dir/manifest
            missing/corrupt.
    """
    manifest, manifest_path = _load_manifest_or_raise(workspace, slug)
    manifest["archived_at"] = _utc_now_iso()
    write_run_manifest(manifest_path, manifest)


def unarchive_run(workspace: Path | str, slug: str) -> None:
    """Clear the ``archived_at`` flag on a run.  Inverse of :func:`archive_run`."""
    manifest, manifest_path = _load_manifest_or_raise(workspace, slug)
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
    """Edit user-mutable fields on a run's manifest.

    ``slug`` / ``backend`` / ``created_at`` / ``created_from`` / ``model``
    are immutable post-creation (identity / lineage).  Passing ``None``
    leaves a field unchanged; pass ``""`` to clear an optional field
    (target / description / notes).  ``display_name`` cannot be cleared.

    Returns:
        The updated manifest dict (post-write).

    Raises:
        InvalidRunIdError / RunNotFoundError: standard.
        ValueError: if display_name is set to empty.
    """
    manifest, manifest_path = _load_manifest_or_raise(workspace, slug)
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


# ----------------------------------------------------------------------------
# Discovery: enriched summaries (read-only — reads step3_state.json)
# ----------------------------------------------------------------------------

def list_run_summaries(workspace: Path | str) -> List[Dict[str, Any]]:
    """Return one summary per run — the DEFAULT run plus every named run.

    The DEFAULT run is SYNTHESIZED (it has no manifest dir): its identity
    comes from ``step3_state.json::runs.default`` (display_name / target,
    if a save ever set them) and its per-patch status counts come from
    ``runs.default.patches``.  The named runs are scanned from ``runs/``
    context dirs (those with a ``run_manifest.json`` — default-run
    per-patch reconstruction dirs are classified as NOT runs and skipped).

    Each summary dict:
        - ``run_id`` (slug)
        - ``display_name`` (falls back to slug / "Default")
        - ``target`` (free text, or "")
        - ``backend`` ("cellpose" or manifest value)
        - ``description`` (or "")
        - ``archived`` (bool; default run never archived)
        - ``manifest_ok`` (bool; True for the synthesized default,
          False for a named run whose manifest is missing/corrupt)
        - ``is_active`` (== active_run_id in step3_state.json)
        - ``patch_counts`` ({reconstructed, curated, accepted, finalized,
          total})
        - ``last_activity_iso`` (str | None)
        - ``manifest_path`` ("" for the default run)

    Defensive across all failure modes — NEVER raises.  Always returns at
    least the synthesized default-run summary.
    """
    ws = Path(workspace)
    try:
        envelope = _read_state_envelope(ws)
        runs_state = envelope.get("runs")
        if not isinstance(runs_state, dict):
            runs_state = {}
        active_id = str(envelope.get("active_run_id") or DEFAULT_RUN_ID)

        out: List[Dict[str, Any]] = []

        # 1. Synthesize the DEFAULT run (always present; flat + manifest-less).
        default_entry = runs_state.get(DEFAULT_RUN_ID)
        if not isinstance(default_entry, dict):
            default_entry = {}
        default_patches = default_entry.get("patches")
        out.append({
            "run_id": DEFAULT_RUN_ID,
            "display_name": default_entry.get("display_name") or "Default",
            "target": default_entry.get("target") or "",
            "backend": "cellpose",
            "description": "",
            "archived": False,
            # The default run is valid by construction — no manifest file
            # is expected, so it is NOT flagged as broken.
            "manifest_ok": True,
            "is_active": (active_id == DEFAULT_RUN_ID),
            "patch_counts": _count_patch_statuses(default_patches),
            "last_activity_iso": _latest_patch_activity(default_patches),
            "manifest_path": "",
        })

        # 2. Scan runs/ for named-run CONTEXT dirs (run_manifest.json present).
        for entry in list_runs_with_manifests(_step3_paths.step3_runs_root(ws)):
            slug = entry["run_id"]
            if slug == DEFAULT_RUN_ID:
                # A stray runs/default/ context dir would be an Option-B
                # artifact — ignore it (default is synthesized above).
                continue
            manifest = entry.get("manifest")
            if isinstance(manifest, dict):
                display_name = manifest.get("display_name") or slug
                target = manifest.get("target", "") or ""
                backend = manifest.get("backend", "cellpose") or "cellpose"
                description = manifest.get("description", "") or ""
                archived = is_run_archived(manifest)
            else:
                display_name = slug
                target = ""
                backend = "cellpose"
                description = ""
                archived = False

            run_patches = None
            run_entry = runs_state.get(slug)
            if isinstance(run_entry, dict):
                run_patches = run_entry.get("patches")

            out.append({
                "run_id": slug,
                "display_name": display_name,
                "target": target,
                "backend": backend,
                "description": description,
                "archived": archived,
                "manifest_ok": entry.get("manifest_ok", False),
                "is_active": (slug == active_id),
                "patch_counts": _count_patch_statuses(run_patches),
                "last_activity_iso": _latest_mtime(
                    _step3_paths.run_root(ws, slug)
                ),
                "manifest_path": entry.get("manifest_path", ""),
            })
        return out
    except Exception as _e:  # noqa: BLE001 — summaries must never raise
        print(f"[WARN] list_run_summaries failed: {type(_e).__name__}: {_e}")
        return [{
            "run_id": DEFAULT_RUN_ID,
            "display_name": "Default",
            "target": "",
            "backend": "cellpose",
            "description": "",
            "archived": False,
            "manifest_ok": True,
            "is_active": True,
            "patch_counts": _count_patch_statuses(None),
            "last_activity_iso": None,
            "manifest_path": "",
        }]


# ----------------------------------------------------------------------------
# Provenance dependency scan (soft / informational — copy-on-reuse)
# ----------------------------------------------------------------------------

def find_runs_inheriting_from(
    workspace: Path | str,
    target_slug: str,
) -> List[str]:
    """Find runs whose ``created_from`` references ``target_slug``.

    Step 3 uses **copy-on-reuse** for models (the chosen model is copied
    into the new run), so a dependent run does NOT actually
    need the source on disk; this scan is INFORMATIONAL (mirrors Step 2's
    scan shape so the delete flow can warn "run X was the source for Y").

    Checks both a Step-3-run source (``created_from.source_step3_bundle``
    pointing at the slug) and a generic ``source_run_id`` reference.

    Returns:
        Sorted list of dependent slugs.  Empty if none.
    """
    ws = Path(workspace)
    runs_root = _step3_paths.step3_runs_root(ws)
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
        refs = {
            created_from.get("source_step3_bundle"),
            created_from.get("source_step2_run_id"),
            created_from.get("source_run_id"),
        }
        if target_slug in refs:
            dependents.append(slug)
    return sorted(dependents)


# ----------------------------------------------------------------------------
# Model-lineage resolution: created_from spec → (lateral, axial) checkpoints
# ----------------------------------------------------------------------------

def resolve_init_spec_to_checkpoint(
    workspace: Path | str,
    init_spec: Optional[Dict[str, Any]],
) -> Optional[Tuple[str, str]]:
    """Resolve a run's ``created_from`` lineage to ``(lateral, axial)`` paths.

    A Step 3 run carries BOTH a lateral and an axial model, so this returns
    a ``(lateral_path, axial_path)`` tuple (or ``None`` if unresolvable).

    Modes:

    * ``pretrained`` → ``(name, name)`` (the Cellpose built-in name is the
      same "path" for both buckets), or ``None`` if no name.
    * ``browse`` → ``(external_path, external_path)`` (the one picked file
      drives both buckets), or ``None`` if no path.
    * ``step3_refined`` → the bundle's lateral + axial weights under
      ``refinement/models/{bundle}/{bucket}`` — returned only if BOTH
      weight files exist on disk; else ``None``.
    * ``step2_run`` → **Phase-5 STUB** — the Step 2↔3 model bridge is wired
      in Phase 5; for now returns ``None`` (the New-Run dialog will resolve
      the Step 2 run's model pair itself once that lands).

    Returns ``None`` for malformed / missing specs.
    """
    if not isinstance(init_spec, dict):
        return None
    mode = str(init_spec.get("mode", ""))
    ws = Path(workspace)

    if mode == "pretrained":
        name = init_spec.get("pretrained_name")
        if name:
            return (str(name), str(name))
        return None

    if mode == "browse":
        ext = init_spec.get("external_path")
        if ext:
            return (str(ext), str(ext))
        return None

    if mode == "step3_refined":
        bundle = init_spec.get("source_step3_bundle")
        if not bundle:
            return None
        # Run-disambiguated resolution (Phase 5 audit HIGH-1B): the same
        # bundle NAME can exist under the flat default tree AND under any
        # named run's ``refinement/models/``.  The init_spec carries the
        # owning run via ``source_step3_run_id`` so resolution pins to
        # exactly ``(run_id, bundle)`` — NOT a name-scan that could pick the
        # wrong run silently.  A missing / empty run_id means the FLAT
        # default tree.
        run_id = init_spec.get("source_step3_run_id")
        return _locate_refined_bundle_pair(
            ws, str(bundle), str(run_id) if run_id else None
        )

    if mode == "step2_run":
        # Phase 5: resolve the Step 2 run's lateral/axial model PAIR via the
        # Step 2↔3 model bridge (Step 2's ``list_run_model_sources``).  The
        # source_step2_run_id is Step 2's run slug; this returns the latest
        # lateral + axial weights for that run, or None if either bucket is
        # missing.
        source = init_spec.get("source_step2_run_id")
        return _resolve_step2_run_pair(ws, str(source) if source else "")

    return None


def _locate_refined_bundle_pair(
    ws: Path, bundle: str, run_id: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Resolve a step3_refined bundle's ``(lateral, axial)`` weights EXACTLY.

    Resolves the bundle under the SPECIFIC owning run identified by
    ``run_id`` (Phase 5 audit HIGH-1B).  ``run_id=None`` (or ``""``) means
    the FLAT default tree (``refinement/models/{bundle}``); a named run
    resolves ``runs/{run_id}/refinement/models/{bundle}``.  Returns the
    ``(lateral, axial)`` pair only if BOTH weight files exist at that exact
    location; else ``None``.

    This deliberately does NOT name-scan across runs: the same bundle name
    can live under multiple runs, and a flat-first / alphabetical scan would
    silently pick the wrong run (the audit's HIGH-1B finding).  The owning
    run is carried in the init_spec's ``source_step3_run_id``.
    """
    if not bundle:
        return None
    rid = run_id or None  # normalize "" → None (flat default)
    try:
        lat = _step3_paths.refined_model_weights_path(ws, bundle, "lateral", rid)
        ax = _step3_paths.refined_model_weights_path(ws, bundle, "axial", rid)
    except Exception:
        return None
    if lat.is_file() and ax.is_file():
        return (str(lat), str(ax))
    return None


def _resolve_step2_run_pair(
    ws: Path, source_run_id: str,
) -> Optional[Tuple[str, str]]:
    """Resolve a Step 2 run's latest lateral+axial weights (the bridge).

    Uses Step 2's ``list_run_model_sources`` lister (per-bucket / per-round
    weight entries), filters to ``source_run_id``, and picks the LATEST
    round per bucket (highest ``round_NN``, tie-broken by mtime).  Returns
    ``(lateral, axial)`` only if BOTH buckets resolve to an existing weights
    file; else ``None`` (a Step 3 run needs the full pair to reconstruct).

    Defensive: any import / scan failure → ``None`` (logged).
    """
    if not source_run_id:
        return None
    try:
        from napari_epicycle.hitl_training_annotation._run_management import (
            list_run_model_sources,
        )
        sources = list_run_model_sources(ws) or []
    except Exception as exc:  # noqa: BLE001 — bridge must never crash create
        print(f"[STEP3-RUNS] step2_run resolve failed: "
              f"{type(exc).__name__}: {exc}")
        return None

    best: Dict[str, Tuple[Tuple[int, str], str]] = {}  # bucket -> (key, path)
    for src in sources:
        if not isinstance(src, dict):
            continue
        if src.get("run_id") != source_run_id:
            continue
        bucket = src.get("bucket")
        if bucket not in ("lateral", "axial"):
            continue
        if not src.get("exists"):
            continue
        wp = src.get("weights_path")
        if not wp:
            continue
        round_id = str(src.get("round_id", ""))
        try:
            round_idx = (
                int(round_id[len("round_"):])
                if round_id.startswith("round_") else -1
            )
        except ValueError:
            round_idx = -1
        key = (round_idx, str(src.get("mtime_iso") or ""))
        if bucket not in best or key > best[bucket][0]:
            best[bucket] = (key, str(wp))

    lat = best.get("lateral")
    ax = best.get("axial")
    if lat and ax:
        return (lat[1], ax[1])
    return None


# ----------------------------------------------------------------------------
# Cross-run refined-model discovery (the "models_index", built by scanning)
# ----------------------------------------------------------------------------

def is_seed_bundle(
    workspace: Path | str, bundle: str, run_id: Optional[str] = None,
) -> bool:
    """True iff ``bundle`` is a copy-on-reuse SEED (not a refinement-trained).

    Discriminator (Phase 5 audit HIGH-1A): a SEED bundle is the model copy
    that :func:`materialize_run_model` drops into a new run when it reuses an
    existing model.  It is identified by EITHER:

    * its dir name being :data:`INIT_MODEL_NAME` (``"init_model"`` — the
      fixed copy-on-reuse bundle name), OR
    * the presence of a :data:`MODEL_MANIFEST_FILENAME` (``model_manifest.json``)
      seed marker in the bundle dir (written ONLY by ``materialize_run_model``).

    A refinement-TRAINED bundle (the user's own refinement output) instead
    carries a ``manifest.json`` (schema 2.0, written by the refinement
    trainer's ``write_or_merge_bundle_manifest``) and never a
    ``model_manifest.json`` — so the two are unambiguous.
    """
    if bundle == INIT_MODEL_NAME:
        return True
    try:
        seed_marker = (
            _step3_paths.refined_model_dir(Path(workspace), bundle, run_id)
            / MODEL_MANIFEST_FILENAME
        )
    except Exception:
        return False
    return seed_marker.is_file()


def list_cross_run_refined_models(
    workspace: Path | str,
    *,
    trained_only: bool = False,
    default_nested: bool = False,
) -> List[Dict[str, Any]]:
    """Discover refined-model bundles ACROSS the default + every named run.

    Walks the flat default ``refinement/models/`` PLUS each named run's
    ``runs/{slug}/refinement/models/`` ("models_index" — built by
    scanning; a persisted index file is optional for v1).  Returns one entry
    per (bundle, bucket) — the shape the New Run dialog's Step-3 table needs
    (``bundle_name`` / ``bucket`` / ``mtime_iso``) plus the owning ``run_id``
    + the bundle ``manifest`` / ``backend`` for the multi-backend future.

    Args:
        trained_only: when True (Phase 5 audit HIGH-1A), EXCLUDE copy-on-reuse
            SEED bundles (``init_model`` / ``model_manifest.json``-marked — see
            :func:`is_seed_bundle`) so the New-Run *reuse source* list offers
            ONLY refinement-TRAINED bundles.  A seed is just a duplicate of an
            already-listed source model, so listing it pollutes the picker with
            indistinguishable ``init_model`` rows.  Default False preserves the
            full-scan behavior for callers that want every bundle.

    Each entry:
        - ``bundle_name`` (dir name under ``refinement/models/``)
        - ``run_id`` (owning run slug, or ``None`` for the flat default)
        - ``bucket`` ("lateral" | "axial")
        - ``weights_path`` (absolute path to the per-bucket weights file)
        - ``mtime_iso`` (ISO str | None)
        - ``exists`` (always True — only existing weights are yielded)
        - ``manifest`` (dict | None — the bundle's manifest, if any)
        - ``backend`` ("cellpose" or the manifest's backend)

    Defensive — NEVER raises; returns whatever it gathered (``[]`` on a
    top-level failure).
    """
    ws = Path(workspace)
    out: List[Dict[str, Any]] = []
    try:
        # Scan list: default run first, then each named-run context dir.
        # Phase 7: when the workspace ran the optional unified-layout
        # migration (``default_nested=True``) the default run's bundles live
        # under ``runs/default/refinement/models`` and the owning run_id is
        # the "default" slug (NOT None / flat).
        default_path_id = DEFAULT_RUN_ID if default_nested else None
        scan_roots: List[Tuple[Optional[str], Path]] = [
            (default_path_id, _step3_paths.refined_models_root(ws, default_path_id)),
        ]
        runs_root = _step3_paths.step3_runs_root(ws)
        if runs_root.is_dir():
            for entry in list_runs_with_manifests(runs_root):
                slug = entry["run_id"]
                if slug == DEFAULT_RUN_ID:
                    continue
                scan_roots.append(
                    (slug, _step3_paths.refined_models_root(ws, slug))
                )

        for run_id, models_root in scan_roots:
            if not models_root.is_dir():
                continue
            for bundle_dir in sorted(models_root.iterdir(), key=lambda p: p.name):
                if not bundle_dir.is_dir() or bundle_dir.name.startswith("."):
                    continue
                bundle = bundle_dir.name
                # Phase 5 audit HIGH-1A: a reuse SOURCE list must offer only
                # refinement-TRAINED bundles, never copy-on-reuse seeds.
                if trained_only and is_seed_bundle(ws, bundle, run_id):
                    continue
                manifest = _read_bundle_manifest(ws, bundle, run_id)
                backend = "cellpose"
                if isinstance(manifest, dict):
                    backend = manifest.get("backend", "cellpose") or "cellpose"
                for bucket in ("lateral", "axial"):
                    try:
                        wp = _step3_paths.refined_model_weights_path(
                            ws, bundle, bucket, run_id
                        )
                    except Exception:
                        continue
                    if not wp.is_file():
                        continue
                    out.append({
                        "bundle_name": bundle,
                        "run_id": run_id,
                        "bucket": bucket,
                        "weights_path": str(wp),
                        "mtime_iso": _file_mtime_iso(wp),
                        "exists": True,
                        "manifest": manifest,
                        "backend": backend,
                    })
    except Exception as exc:  # noqa: BLE001 — discovery must never raise
        print(f"[STEP3-RUNS] list_cross_run_refined_models failed: "
              f"{type(exc).__name__}: {exc}")
    return out


def _read_bundle_manifest(
    ws: Path, bundle: str, run_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Read a model bundle's manifest (best-effort).

    Tries the refinement-bundle ``manifest.json`` first (schema 2.0), then
    the copied-init-model ``model_manifest.json``.  Returns ``None`` if
    neither is present / parseable.
    """
    try:
        candidates = [
            _step3_paths.refined_model_manifest_path(ws, bundle, run_id),
            _step3_paths.refined_model_dir(ws, bundle, run_id)
            / MODEL_MANIFEST_FILENAME,
        ]
    except Exception:
        return None
    for path in candidates:
        data = _read_json_or_none(path)
        if isinstance(data, dict):
            return data
    return None


# ----------------------------------------------------------------------------
# Copy-on-reuse: give a new run its own self-contained model copy
# ----------------------------------------------------------------------------

def materialize_run_model(
    workspace: Path | str,
    slug: str,
    init_spec: Optional[Dict[str, Any]],
    *,
    target: str = "",
) -> Dict[str, Any]:
    """COPY-ON-REUSE — give a new run its own model copy + resolved paths.

    Called by the controller AFTER ``create_run_on_disk`` (the run-context
    dir must exist to copy into).  Resolves ``init_spec`` to the source
    ``(lateral, axial)`` weights and COPIES them into the run's own model
    tree (``runs/{slug}/refinement/models/init_model/{lateral,axial}``),
    writing a ``model_manifest.json`` (backend / target / created_from /
    copied_from) so the run is SELF-CONTAINED.  Cellpose bundles are small,
    so the duplication is cheap.

    Modes:
      * ``pretrained`` → NO copy; both paths = the Cellpose built-in NAME.
      * ``browse`` → COPY the one picked file into BOTH buckets.
      * ``step2_run`` / ``step3_refined`` → resolve + COPY each bucket.
      * missing / unknown mode → blank selection (no copy; run still made).

    Atomic-ish: weights copy to a ``.tmp`` sibling then ``os.replace``; on
    ANY failure the whole ``init_model`` bundle dir is removed before
    re-raising (so the controller's M2 rollback sees a clean tree).

    Returns:
        ``{"lateral_model_path", "axial_model_path", "model_source_mode"}``
        — pass straight into ``state_manager.create_run`` as per-run model
        fields.

    Raises:
        ModelMaterializationError: source unresolvable, or copy failed.
    """
    ws = Path(workspace)
    mode = str(init_spec.get("mode", "")) if isinstance(init_spec, dict) else ""

    # No source / unknown mode → blank selection; the run is still created
    # (defensive — Step 3 always starts from a model, but never crash here).
    if not isinstance(init_spec, dict) or not mode:
        return {
            "lateral_model_path": None,
            "axial_model_path": None,
            "model_source_mode": None,
        }

    if mode == "pretrained":
        name = init_spec.get("pretrained_name")
        if not name:
            raise ModelMaterializationError(
                "pretrained mode missing pretrained_name"
            )
        # Built-in name — no weights file to copy; both buckets use the name.
        return {
            "lateral_model_path": str(name),
            "axial_model_path": str(name),
            "model_source_mode": "pretrained",
        }

    # browse / step2_run / step3_refined: resolve a real (lat, ax) pair.
    pair = resolve_init_spec_to_checkpoint(ws, init_spec)
    if pair is None:
        raise ModelMaterializationError(
            f"could not resolve a (lateral, axial) model pair for "
            f"mode={mode!r}"
        )
    src_lat, src_ax = pair

    model_dir = _step3_paths.refined_model_dir(ws, INIT_MODEL_NAME, run_id=slug)
    dst_lat = _step3_paths.refined_model_weights_path(
        ws, INIT_MODEL_NAME, "lateral", run_id=slug
    )
    dst_ax = _step3_paths.refined_model_weights_path(
        ws, INIT_MODEL_NAME, "axial", run_id=slug
    )
    manifest_path = model_dir / MODEL_MANIFEST_FILENAME

    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        _atomic_copy(Path(src_lat), dst_lat)
        _atomic_copy(Path(src_ax), dst_ax)
        if not dst_lat.exists() or not dst_ax.exists():
            raise ModelMaterializationError(
                "copied weights missing immediately after copy"
            )
        manifest = {
            "schema_version": "1.0",
            "backend": "cellpose",
            "target": target or "",
            "model_source_mode": mode,
            "created_from": init_spec,
            "copied_from": {"lateral": str(src_lat), "axial": str(src_ax)},
            "lateral_weights": "lateral",
            "axial_weights": "axial",
            "created_at": _utc_now_iso(),
        }
        _atomic_write_json(manifest_path, manifest)
    except Exception:
        # Clean up the partial bundle so a retry / rollback is clean.
        try:
            if model_dir.is_dir():
                shutil.rmtree(model_dir)
        except OSError:
            pass
        raise

    return {
        "lateral_model_path": str(dst_lat),
        "axial_model_path": str(dst_ax),
        "model_source_mode": mode,
    }


# ----------------------------------------------------------------------------
# Clean intermediates: free a run's cached cellpose prediction outputs
# (flows/probs/masks) for its RECONSTRUCTED-or-beyond patches
# ----------------------------------------------------------------------------

#: Patch statuses that are "reconstructed-or-beyond" — a 3D reconstruction
#: (and possibly curation) EXISTS for the patch, so its cached 2D prediction
#: flows/probs/masks are safe to discard (re-PREDICT regenerates them).  A
#: patch that is merely ``predicted`` (NOT yet reconstructed) is EXCLUDED:
#: deleting its cache would lose un-backed-up prediction work.
RECONSTRUCTED_OR_BEYOND = frozenset(
    {"reconstructed", "curating", "accepted", "finalized"}
)


def run_path_id_for(slug: str, *, default_nested: bool) -> Optional[str]:
    """Resolve a run slug to the path-id the per-run ``_paths`` helpers want.

    Napari-free mirror of the controller's ``_active_run_path_id`` resolver,
    parameterized by the workspace's ``default_run_nested`` flag so it has no
    state-manager / Qt dependency:

    * ``slug == "default"`` + ``default_nested=False`` → ``None`` (FLAT
      default tree — byte-identical to today).
    * ``slug == "default"`` + ``default_nested=True`` → ``"default"`` (the
      Phase-7 unified nested tree, ``runs/default/...``).
    * any other slug → the slug itself (a named-run context dir).
    """
    if slug == DEFAULT_RUN_ID:
        return DEFAULT_RUN_ID if default_nested else None
    return slug


def _reconstructed_patch_stems(ws: Path, slug: str) -> List[str]:
    """Patch STEMS for a run whose status is reconstructed-or-beyond.

    Reads ``step3_state.json::runs.{slug}.patches`` read-only.  Never raises
    (missing / corrupt state → empty list).
    """
    try:
        envelope = _read_state_envelope(ws)
        runs_state = envelope.get("runs")
        if not isinstance(runs_state, dict):
            return []
        run_entry = runs_state.get(slug)
        if not isinstance(run_entry, dict):
            return []
        patches = run_entry.get("patches")
        if not isinstance(patches, dict):
            return []
        stems: List[str] = []
        for patch_name, record in patches.items():
            if not isinstance(record, dict):
                continue
            if record.get("status") in RECONSTRUCTED_OR_BEYOND:
                stems.append(Path(str(patch_name)).stem)
        return stems
    except Exception:  # noqa: BLE001 — never raise
        return []


def summarize_run_intermediates(
    ws: Path | str, slug: str, *, run_path_id: Optional[str],
) -> Dict[str, int]:
    """Summarize a run's reclaimable prediction-cache (flows/probs/masks).

    For each of the run's RECONSTRUCTED-or-beyond patches (see
    :data:`RECONSTRUCTED_OR_BEYOND`), globs ``{stem}_*`` in the run's
    prediction-cache dir and tallies file sizes.  A merely-``predicted``
    patch (no reconstruction yet) is EXCLUDED — its cache is the only copy
    of un-backed-up work.

    Returns ``{"n_patches", "bytes", "n_files"}`` where ``n_patches`` counts
    only reconstructed patches that actually HAVE cache files.  Never raises
    (missing dir / corrupt state → all-zero).
    """
    out = {"n_patches": 0, "bytes": 0, "n_files": 0}
    try:
        wsp = Path(ws)
        cache_dir = _step3_paths.prediction_cache_dir(wsp, run_path_id)
        if not cache_dir.is_dir():
            return out
        for stem in _reconstructed_patch_stems(wsp, slug):
            patch_files = 0
            for f in cache_dir.glob(f"{stem}_*"):
                try:
                    if not f.is_file():
                        continue
                    out["bytes"] += f.stat().st_size
                    out["n_files"] += 1
                    patch_files += 1
                except OSError:
                    continue
            if patch_files:
                out["n_patches"] += 1
    except Exception:  # noqa: BLE001 — never raise
        return {"n_patches": 0, "bytes": 0, "n_files": 0}
    return out


def clean_run_intermediates(
    ws: Path | str,
    slug: str,
    *,
    run_path_id: Optional[str],
    reconstructed_only: bool = True,
) -> Dict[str, int]:
    """Delete a run's cached prediction outputs for reconstructed patches.

    Unlinks every ``{stem}_*`` flows/probs/masks file in the run's
    prediction-cache dir for each RECONSTRUCTED-or-beyond patch.  Does NOT
    touch labels / manifests / qc / curation, and does NOT change any patch
    status — purely a disk-space reclaim (a re-PREDICT regenerates the cache
    if the user later wants to re-reconstruct).

    ``reconstructed_only`` is the safe default (and the only supported mode):
    a ``predicted``/``unprocessed`` patch's cache is its only copy of work,
    so it is NEVER deleted.  The kwarg exists for call-site explicitness; a
    ``False`` value is intentionally treated the same as ``True`` here (no
    code path deletes non-reconstructed cache).

    Returns ``{"n_cleared", "bytes_freed"}``.  Never raises (per-file
    ``try/except``; missing dir → zeros).
    """
    res = {"n_cleared": 0, "bytes_freed": 0}
    try:
        wsp = Path(ws)
        cache_dir = _step3_paths.prediction_cache_dir(wsp, run_path_id)
        if not cache_dir.is_dir():
            return res
        for stem in _reconstructed_patch_stems(wsp, slug):
            for f in cache_dir.glob(f"{stem}_*"):
                try:
                    if not f.is_file():
                        continue
                    size = f.stat().st_size
                except OSError:
                    size = 0
                try:
                    f.unlink()
                    res["n_cleared"] += 1
                    res["bytes_freed"] += size
                except OSError:
                    continue
    except Exception:  # noqa: BLE001 — never raise
        return res
    return res


# ----------------------------------------------------------------------------
# model_source_mode vocabulary mapping (materialize → state)  [audit MED-1]
# ----------------------------------------------------------------------------

def map_model_source_mode_to_state(mode: Optional[str]) -> str:
    """Map ``materialize_run_model``'s mode onto the STATE vocabulary.

    ``materialize_run_model`` returns ``model_source_mode`` in the RICH
    init-spec vocabulary ``{pretrained, browse, step2_run, step3_refined}``
    (or ``None`` for a blank selection).  But the run's STATE entry's
    ``model_source_mode`` is consumed by ``state_manager.set_model_selection``
    + the ``model_source_combo`` + ``_restore_persisted_model_selection``,
    which only understand ``{auto, import, pretrained}``.  Storing a rich mode
    there desynchronizes the combo + the persisted-selection restore (Phase 5
    audit MED-1).

    Mapping:

    * ``pretrained`` → ``"pretrained"`` (a Cellpose built-in name)
    * ``browse`` / ``step2_run`` / ``step3_refined`` → ``"import"`` (all
      resolve to an on-disk model PATH the run imported a copy of)
    * anything else (``None`` / unknown / blank) → ``"auto"`` (auto-discover)

    The RICH provenance is NOT lost — it stays in the run_manifest
    ``created_from`` (the init_spec) + the copied bundle's
    ``model_manifest.json``.  Only the STATE field is normalized.
    """
    if mode == "pretrained":
        return "pretrained"
    if mode in ("browse", "step2_run", "step3_refined"):
        return "import"
    return "auto"


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy a weights file (or dir) to ``dst`` atomically-ish.

    File → copy to a ``.tmp`` sibling then ``os.replace``.  Directory
    (future SavedModel-style backends) → copytree to a temp sibling then
    ``os.replace``.  Any stale temp from a prior interrupted copy is cleared
    first.

    Raises:
        ModelMaterializationError: if ``src`` does not exist.
        OSError: on filesystem failures (propagated to the caller's cleanup).
    """
    if not src.exists():
        raise ModelMaterializationError(f"source weights missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp")
    try:
        if tmp.is_dir():
            shutil.rmtree(tmp)
        elif tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    if src.is_dir():
        shutil.copytree(src, tmp)
    else:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write a JSON dict (tempfile + fsync + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def _read_json_or_none(path: Path) -> Optional[Any]:
    """Read + parse a JSON file; ``None`` on missing / unreadable / invalid."""
    try:
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _file_mtime_iso(path: Path) -> Optional[str]:
    """ISO 8601 UTC mtime for a file, or ``None`` if unstattable."""
    try:
        mt = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mt, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _validate_or_raise(slug: str) -> None:
    """Validate slug via ``_paths.validate_run_id`` → :class:`InvalidRunIdError`."""
    try:
        _step3_paths.validate_run_id(slug)
    except ValueError as e:
        raise InvalidRunIdError(str(e)) from e


def _load_manifest_or_raise(
    workspace: Path | str, slug: str,
) -> Tuple[Dict[str, Any], Path]:
    """Load a named run's manifest (validating + rejecting the default run).

    Returns ``(manifest_dict, manifest_path)``.

    Raises:
        InvalidRunIdError: malformed slug.
        RunNotFoundError: the default run (no manifest), or the dir /
            manifest is missing or corrupt.
    """
    _validate_or_raise(slug)
    if slug.casefold() == DEFAULT_RUN_ID.casefold():
        raise RunNotFoundError(
            "the default run is the flat layout — it has no run_manifest.json "
            "to operate on"
        )
    ws = Path(workspace)
    run_dir = _step3_paths.run_root(ws, slug)
    if not run_dir.is_dir():
        raise RunNotFoundError(f"run not found: {run_dir}")
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    manifest = read_run_manifest(manifest_path)
    if manifest is None:
        raise RunNotFoundError(
            f"run manifest missing or corrupt: {manifest_path}"
        )
    return manifest, manifest_path


def _read_state_envelope(workspace: Path) -> Dict[str, Any]:
    """Read ``step3_state.json`` read-only → dict (``{}`` on missing/corrupt)."""
    path = _step3_paths.state_json_path(workspace)
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _count_patch_statuses(patches: Any) -> Dict[str, int]:
    """Count per-patch statuses into the summary's ``patch_counts`` dict.

    Keys: ``reconstructed`` / ``curated`` / ``accepted`` / ``finalized`` /
    ``total``.  ``curated`` counts the in-progress ``"curating"`` status
    (the design's per-patch flow names the state ``curating``; the summary
    surfaces it as ``curated``).  Non-dict input → all-zero counts.
    """
    counts = {
        "reconstructed": 0,
        "curated": 0,
        "accepted": 0,
        "finalized": 0,
        "total": 0,
    }
    if not isinstance(patches, dict):
        return counts
    for record in patches.values():
        if not isinstance(record, dict):
            continue
        counts["total"] += 1
        status = record.get("status", "unprocessed")
        if status == "reconstructed":
            counts["reconstructed"] += 1
        elif status == "curating":
            counts["curated"] += 1
        elif status == "accepted":
            counts["accepted"] += 1
        elif status == "finalized":
            counts["finalized"] += 1
    return counts


def _latest_patch_activity(patches: Any) -> Optional[str]:
    """Latest ``updated_at`` ISO string across a run's patch records, or None.

    Used for the DEFAULT run's ``last_activity_iso`` (it has no dir to walk
    for an mtime — its activity is recorded in the state JSON's per-patch
    ``updated_at`` timestamps written by ``set_patch_status``).
    """
    if not isinstance(patches, dict):
        return None
    stamps = [
        record.get("updated_at")
        for record in patches.values()
        if isinstance(record, dict) and record.get("updated_at")
    ]
    if not stamps:
        return None
    # ISO 8601 strings sort lexicographically in time order.
    return max(str(s) for s in stamps)


def _latest_mtime(run_dir: Path) -> Optional[str]:
    """Latest mtime across a named run's context tree, as ISO string, or None."""
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
    # Include the dir itself (an empty run dir still has a useful mtime).
    try:
        dir_mt = run_dir.stat().st_mtime
        if latest is None or dir_mt > latest:
            latest = dir_mt
    except OSError:
        pass
    if latest is None:
        return None
    return datetime.fromtimestamp(latest, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _utc_now_iso() -> str:
    """ISO 8601 UTC timestamp with trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    # Exceptions
    "RunAlreadyExistsError",
    "RunNotFoundError",
    "InvalidRunIdError",
    "ModelMaterializationError",
    # Constants
    "INIT_MODEL_NAME",
    "MODEL_MANIFEST_FILENAME",
    # Disk ops
    "create_run_on_disk",
    "delete_run_from_disk",
    # Manifest mutations
    "archive_run",
    "unarchive_run",
    "edit_run_metadata",
    # Discovery
    "list_run_summaries",
    "list_cross_run_refined_models",
    "is_seed_bundle",
    # Dependency scan
    "find_runs_inheriting_from",
    # Model-lineage resolution
    "resolve_init_spec_to_checkpoint",
    # Copy-on-reuse
    "materialize_run_model",
    "map_model_source_mode_to_state",
    # Clean intermediates
    "RECONSTRUCTED_OR_BEYOND",
    "run_path_id_for",
    "summarize_run_intermediates",
    "clean_run_intermediates",
]

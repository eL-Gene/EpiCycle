"""Centralized Step 3 workspace path constants and helpers.

**Source of truth** for every filesystem path that the
``fusion_reconstruction`` package reads or writes.  Replaces the
ad-hoc ``Path(self.working_dir) / "artifacts" / ...`` constructions
that were scattered across ~95 call sites in controller.py +
state_manager.py + training_pair_extractor.py.

Two layouts are recognized:

* **Legacy** (pre-2026-05-20): artifacts scattered at workspace root —
  ``artifacts/{qc,reconstruction,snapshots,prediction_cache,
  refinement_training_data}/`` + ``models/refinement/``.  State JSON
  filename is ``reconstruction_state.json``.

* **New (v2)** (2026-05-20+): everything under ``artifacts/step3/`` —
  ``runs/qc/snapshots/prediction_cache/curation/refinement/
  training_archive/``.  State JSON renamed to ``step3_state.json``.
  Refinement training pairs + refined model bundles consolidated
  under ``artifacts/step3/refinement/``.

This module provides:

1. Path constants — workspace-rooted directory + file paths for both
   layouts.  Each `<thing>_path()` function returns a ``pathlib.Path``.
2. Layout detection — ``detect_layout_version(ws) -> int`` returning
   ``1`` (legacy), ``2`` (new), or ``0`` (workspace is empty / no
   step-3 markers).
3. Mixed-layout detection — ``is_mixed_layout(ws) -> bool``.  This
   indicates a partially-migrated workspace (hazard state).

**Format-agnostic path returns.**  For image artifacts (labels, source
stacks, etc.) the returned path is the **stem** without an extension.
Callers append ``.tif`` (current default), ``.zarr`` (future OME-Zarr),
or whatever else at the read/write site.  This way the format choice
lives in the I/O layer, not in the path helpers.  The exceptions are:

* Manifest JSONs — always ``.json`` so the helper returns the full
  filename.
* Cellpose loss / config companions — always ``.npy`` / ``.json``
  (Cellpose-imposed).

The format-aware caller convention:

::

    path = patch_labels_stem(ws, "patch_001")
    # Write a TIFF:
    tifffile.imwrite(str(path) + ".tif", labels_3d)
    # Or in the future, a Zarr store:
    # zarr.save(str(path) + ".zarr", labels_3d)

Helper functions are PURE (no side effects, no filesystem mutation).
``mkdir`` is the caller's responsibility — these helpers just answer
"where should X live?"
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Union


# ----------------------------------------------------------------------------
# Filename + layout constants
# ----------------------------------------------------------------------------

#: State JSON filename in the new layout (v2).
STEP3_STATE_FILENAME = "step3_state.json"

#: Legacy state JSON filename (v1, pre-2026-05-20).
LEGACY_STATE_FILENAME = "reconstruction_state.json"

#: Sub-tree name under ``artifacts/`` for the new layout.
STEP3_ARTIFACTS_SUBDIR = "step3"

#: Layout version constants.  Stored in the state JSON to mark which
#: layout the workspace is on.  ``LAYOUT_VERSION_MIXED`` is a runtime-
#: only sentinel — never persisted; indicates a partially-migrated
#: workspace that the user must resolve before operations resume.
LAYOUT_VERSION_UNKNOWN = 0
LAYOUT_VERSION_LEGACY = 1
LAYOUT_VERSION_NEW = 2
LAYOUT_VERSION_MIXED = 3
CURRENT_LAYOUT_VERSION = LAYOUT_VERSION_NEW

#: Per-run manifest filename (lives at ``runs/{run_slug}/``).  Mirrors
#: Step 2's ``RUN_MANIFEST_FILENAME`` + the value owned by
#: ``packages/epicycle_io/step3_run_manifest.py``.
RUN_MANIFEST_FILENAME = "run_manifest.json"

#: Maximum length of a run_id slug (filesystem-friendly).  Mirrors Step 2.
RUN_ID_MAX_LENGTH = 64

#: Slug pattern — alphanumeric + underscore + dash, starts with
#: alphanumeric (not _ or -), max RUN_ID_MAX_LENGTH chars.  Identical to
#: Step 2's ``_RUN_ID_PATTERN`` (filesystem-friendly on every OS).
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


# Type alias for "anything that can be turned into a Path".  Used to
# accept either str or Path arguments in every helper.
_WS = Union[str, Path]


def _ws(workspace: _WS) -> Path:
    """Coerce the workspace argument to a ``Path``.  Helper internal."""
    return Path(workspace)


def _run_segment(ws: Path, run_id: Optional[str]) -> Path:
    """Per-run root for a Step 3 run context (multi-run foundation).

    This is the single branch point that makes every per-run path helper
    back-compatible: with ``run_id=None`` it returns today's flat Step 3
    sub-tree (``artifacts/step3/``) so the per-run helpers reproduce the
    pre-multi-run paths BYTE-FOR-BYTE; with a run slug it returns that
    run's context dir under ``artifacts/step3/runs/{run_id}/``.

    Examples::

        _run_segment(ws, None)      -> {ws}/artifacts/step3
        _run_segment(ws, "cells")   -> {ws}/artifacts/step3/runs/cells

    Note ``runs_dir(ws)`` is deliberately reused for the run-id case so
    the literal ``runs/`` segment matches today's reconstruction-runs
    directory name (see :func:`reconstructions_dir`).
    """
    return step3_root(ws) if run_id is None else runs_dir(ws) / run_id


# ----------------------------------------------------------------------------
# Workspace root + state JSON
# ----------------------------------------------------------------------------

def workspace_root(workspace: _WS) -> Path:
    """The workspace directory itself."""
    return _ws(workspace)


def state_json_path(workspace: _WS) -> Path:
    """Path to the NEW-layout state JSON (``step3_state.json``)."""
    return _ws(workspace) / STEP3_STATE_FILENAME


def legacy_state_json_path(workspace: _WS) -> Path:
    """Path to the LEGACY state JSON (``reconstruction_state.json``).

    Used by the migration script to detect + read the legacy state
    before rewriting it to the new filename.
    """
    return _ws(workspace) / LEGACY_STATE_FILENAME


# ----------------------------------------------------------------------------
# Step 3 root
# ----------------------------------------------------------------------------

def step3_root(workspace: _WS) -> Path:
    """``{ws}/artifacts/step3/`` — the canonical Step 3 sub-tree."""
    return _ws(workspace) / "artifacts" / STEP3_ARTIFACTS_SUBDIR


def step3_artifacts_root(workspace: _WS) -> Path:
    """Alias for :func:`step3_root` — explicit name where helpful."""
    return step3_root(workspace)


# ----------------------------------------------------------------------------
# Reconstruction runs (per-patch)
# ----------------------------------------------------------------------------

def runs_dir(workspace: _WS) -> Path:
    """``{ws}/artifacts/step3/runs/`` — parent of all per-patch run dirs.

    Kept run_id-free for back-compat: in the flat (single-context)
    layout this directory holds the per-patch reconstruction dirs
    directly.  The run_id-aware successor is :func:`reconstructions_dir`.
    """
    return step3_root(workspace) / "runs"


def reconstructions_dir(workspace: _WS, run_id: Optional[str] = None) -> Path:
    """Parent of all per-patch reconstruction output dirs.

    The run_id-aware successor to :func:`runs_dir`.

    * ``run_id=None`` → ``{ws}/artifacts/step3/runs/`` — BYTE-IDENTICAL
      to :func:`runs_dir` (today's flat layout, where per-patch
      reconstruction dirs live directly under ``runs/``).
    * ``run_id="cells"`` → ``{ws}/artifacts/step3/runs/cells/reconstructions/``
      — nested under the run context (the ``runs/{patch}`` → ``runs/{run}/
      reconstructions/{patch}`` rename).
    """
    if run_id is None:
        return runs_dir(workspace)
    return _run_segment(_ws(workspace), run_id) / "reconstructions"


def patch_run_dir(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Per-patch reconstruction output directory.

    Holds the patch's ``labels_3d.tif`` (or ``.zarr``) + run manifest.
    Re-reconstruction overwrites this directory in place (no run
    accumulation) — confirmed user preference 2026-05-20.

    With ``run_id=None`` returns ``runs/{patch}`` (flat, byte-identical);
    with a run slug returns ``runs/{run_id}/reconstructions/{patch}``.
    """
    return reconstructions_dir(workspace, run_id) / patch_name


def patch_labels_stem(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Stem of the per-patch labels file (no extension).

    Format-agnostic: caller appends ``.tif`` (TIFF), ``.zarr`` (Zarr),
    or any other extension at the I/O site.
    """
    return patch_run_dir(workspace, patch_name, run_id) / "labels_3d"


def patch_run_manifest_path(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Per-patch reconstruction manifest (``manifest.json``)."""
    return patch_run_dir(workspace, patch_name, run_id) / "manifest.json"


# ----------------------------------------------------------------------------
# Run CONTEXT helpers (named-run directories — distinct from the per-run
# path helpers above, which take ``run_id`` and resolve per-artifact paths
# WITHIN a run).  These answer "where does run {slug}'s context dir + its
# manifest live?" — the parent of every per-run artifact tree.
#
# NOTE on the default run: under the Option-A runtime decision the DEFAULT run is the FLAT layout — it has NO context dir and NO
# ``run_manifest.json``.  ``run_root(ws, "default")`` therefore names a dir
# that, by design, should never exist; these helpers are only meaningful
# for NON-default slugs (named runs).  They do NOT special-case "default"
# (callers own that policy — see ``_run_management.create_run_on_disk``).
# ----------------------------------------------------------------------------

def step3_runs_root(workspace: _WS) -> Path:
    """``{ws}/artifacts/step3/runs/`` — parent of all run-context dirs.

    Alias of :func:`runs_dir` (in the flat layout this same directory also
    holds the default run's per-patch reconstruction dirs).  Provided under
    the Step-2-symmetric name so run-management code reads naturally.
    """
    return runs_dir(workspace)


def run_root(workspace: _WS, run_id: str) -> Path:
    """``{ws}/artifacts/step3/runs/{run_id}/`` — a named-run context dir.

    Only meaningful for non-default slugs (the default run is flat — see
    the section note above).
    """
    return runs_dir(workspace) / run_id


def run_manifest_path(workspace: _WS, run_id: str) -> Path:
    """``runs/{run_id}/run_manifest.json`` — a named run's manifest path."""
    return run_root(workspace, run_id) / RUN_MANIFEST_FILENAME


def validate_run_id(run_id: str) -> None:
    """Raise ``ValueError`` if ``run_id`` is not a valid run handle.

    Run-handle rules (mirror Step 2's ``_paths.validate_run_id``):
    alphanumeric + underscore + dash, starts with alphanumeric (not ``_``
    or ``-``), max :data:`RUN_ID_MAX_LENGTH` chars, case-sensitive.
    Rejects path-traversal patterns (``../``, ``/``, ``\\``) and any
    character outside ``[A-Za-z0-9_-]``.
    """
    if not isinstance(run_id, str):
        raise ValueError(
            f"Run handle must be str, got {type(run_id).__name__}"
        )
    if not run_id:
        raise ValueError("Run handle must not be empty")
    if len(run_id) > RUN_ID_MAX_LENGTH:
        raise ValueError(
            f"Run handle must be <= {RUN_ID_MAX_LENGTH} chars, got {len(run_id)}"
        )
    if not _RUN_ID_PATTERN.match(run_id):
        raise ValueError(
            f"Run handle {run_id!r} is invalid.  Allowed: alphanumeric + "
            f"underscore + dash, starting with alphanumeric, max "
            f"{RUN_ID_MAX_LENGTH} chars.  Path-traversal patterns "
            f"(../, /, \\) and special characters are rejected."
        )


# ----------------------------------------------------------------------------
# QC features + adjacency (per-patch)
# ----------------------------------------------------------------------------

def qc_dir(workspace: _WS, run_id: Optional[str] = None) -> Path:
    """``{ws}/artifacts/step3/qc/`` — per-patch QC features + adjacency.

    With ``run_id=None`` returns the flat ``step3/qc/`` (byte-identical);
    with a run slug returns ``step3/runs/{run_id}/qc/``.
    """
    return _run_segment(_ws(workspace), run_id) / "qc"


def qc_features_csv(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Per-patch QC features CSV (the active, possibly-edited copy)."""
    return qc_dir(workspace, run_id) / f"{patch_name}_features.csv"


def qc_features_raw_csv(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Per-patch raw QC features CSV (auto-backup before first edit)."""
    return qc_dir(workspace, run_id) / f"{patch_name}_features_raw.csv"


def qc_adjacency_raw_csv(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Pre-curation adjacency edge list (label_a, label_b, contact_voxels).

    NET-NEW file as of 2026-05-20 restructure.  Step 3 previously
    computed adjacency in memory but did not persist it (see
    ``qc_features.py:1517-1530``).
    """
    return qc_dir(workspace, run_id) / f"{patch_name}_adjacency_raw.csv"


def qc_adjacency_curated_csv(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Post-curation adjacency edge list.

    Re-written after each Save Curation / Finalize so the post-edit
    neighbor graph is queryable for post-hoc merging + the future
    classifier training feature.
    """
    return qc_dir(workspace, run_id) / f"{patch_name}_adjacency_curated.csv"


# ----------------------------------------------------------------------------
# Snapshots (append-only)
# ----------------------------------------------------------------------------

def snapshots_dir(workspace: _WS) -> Path:
    """``{ws}/artifacts/step3/snapshots/`` — pre-commit append-only snapshots.

    Files written here are NEVER deleted by re-reconstruction or other
    destructive actions.  They are durable training-data archive.  Each
    file is timestamp-suffixed.
    """
    return step3_root(workspace) / "snapshots"


# ----------------------------------------------------------------------------
# Prediction cache
# ----------------------------------------------------------------------------

def prediction_cache_dir(workspace: _WS, run_id: Optional[str] = None) -> Path:
    """``{ws}/artifacts/step3/prediction_cache/`` — Cellpose flow/prob/mask cache.

    With ``run_id=None`` returns the flat ``step3/prediction_cache/``
    (byte-identical); with a run slug returns
    ``step3/runs/{run_id}/prediction_cache/``.
    """
    return _run_segment(_ws(workspace), run_id) / "prediction_cache"


# ----------------------------------------------------------------------------
# Batch-refine recipes (workspace-level)
# ----------------------------------------------------------------------------

def batch_refine_recipes_path(workspace: _WS) -> Path:
    """``{ws}/artifacts/step3/batch_refine_recipes.json``.

    Single workspace-level JSON store of named Batch-Refine recipes
    (``{name: recipe_json}``).  Recipes are geometry-only step lists,
    reload-only (the TARGET — all/selected/by-tier — is NOT persisted).
    Written atomically by ``_batch_recipe_store``.
    """
    return step3_root(workspace) / "batch_refine_recipes.json"


def refine_chain_recipes_path(workspace: _WS) -> Path:
    """``{ws}/artifacts/step3/refine_chain_recipes.json``.

    The Refine tab's saved chains: one
    JSON object mapping ``name -> recipe`` (a dict, not Batch's list), so it
    is read and written with ``value_type=dict``.  Workspace-SHARED, like
    ``batch_refine_recipes.json`` -- every patch and every run sees it.
    """
    return step3_root(workspace) / "refine_chain_recipes.json"


# ----------------------------------------------------------------------------
# Curation outputs (per-patch)
# ----------------------------------------------------------------------------

def curation_root(workspace: _WS, run_id: Optional[str] = None) -> Path:
    """``{ws}/artifacts/step3/curation/`` — parent of all per-patch curation dirs.

    With ``run_id=None`` returns the flat ``step3/curation/``
    (byte-identical); with a run slug returns
    ``step3/runs/{run_id}/curation/``.
    """
    return _run_segment(_ws(workspace), run_id) / "curation"


def patch_curation_dir(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Per-patch curation output directory."""
    return curation_root(workspace, run_id) / patch_name


def patch_curation_labels_stem(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Stem of the curated labels file (no extension)."""
    return patch_curation_dir(workspace, patch_name, run_id) / "labels_curated"


def patch_curation_labels_bak_stem(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Stem of the curated labels backup file (.bak rotation)."""
    return patch_curation_dir(workspace, patch_name, run_id) / "labels_curated.bak"


def patch_curation_tiers_csv(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Per-patch tier assignment CSV."""
    return patch_curation_dir(workspace, patch_name, run_id) / "tiers.csv"


# ----------------------------------------------------------------------------
# Refinement: training pairs + refined-model bundles
# ----------------------------------------------------------------------------

def refinement_root(workspace: _WS, run_id: Optional[str] = None) -> Path:
    """``{ws}/artifacts/step3/refinement/`` — training pairs + refined models.

    With ``run_id=None`` returns the flat ``step3/refinement/``
    (byte-identical); with a run slug returns
    ``step3/runs/{run_id}/refinement/``.
    """
    return _run_segment(_ws(workspace), run_id) / "refinement"


def step3_refinement_config_path(workspace: _WS) -> Path:
    """``{ws}/artifacts/step3/refinement/refinement_config.json``.

    The saved Model-Refinement training config (B4).  It used to sit in the workspace ROOT.
    Deliberately **workspace-global, not run-scoped** — the semantics are
    unchanged, only the location.  A pre-existing root copy is still read
    (the loader falls back to it) and is never deleted.
    """
    return refinement_root(workspace, None) / "refinement_config.json"


def legacy_step3_refinement_config_path(workspace: _WS) -> Path:
    """``{ws}/refinement_config.json`` — where the config used to live."""
    return _ws(workspace) / "refinement_config.json"


def refinement_manifest_path(workspace: _WS, run_id: Optional[str] = None) -> Path:
    """Top-level refinement manifest (was ``extraction_manifest.json``).

    Records per-patch pair counts, mtimes, settings snapshots.
    """
    return refinement_root(workspace, run_id) / "manifest.json"


def refinement_pairs_root(workspace: _WS, run_id: Optional[str] = None) -> Path:
    """``{ws}/artifacts/step3/refinement/pairs/`` — per-patch training pairs."""
    return refinement_root(workspace, run_id) / "pairs"


def patch_pairs_dir(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Per-patch training pairs directory.

    Contains ``spatial/`` (XY) and ``axial/`` (XZ + YZ) sub-dirs.
    """
    return refinement_pairs_root(workspace, run_id) / patch_name


def patch_pairs_spatial_dir(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Per-patch lateral (XY) training pairs."""
    return patch_pairs_dir(workspace, patch_name, run_id) / "spatial"


def patch_pairs_axial_dir(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Per-patch axial (XZ + YZ) training pairs."""
    return patch_pairs_dir(workspace, patch_name, run_id) / "axial"


# Refined model bundles --------------------------------------------------------

def refined_models_root(workspace: _WS, run_id: Optional[str] = None) -> Path:
    """``{ws}/artifacts/step3/refinement/models/`` — parent of all refined-model bundles."""
    return refinement_root(workspace, run_id) / "models"


def refined_model_dir(
    workspace: _WS, model_name: str, run_id: Optional[str] = None,
) -> Path:
    """A single refined-model bundle directory."""
    return refined_models_root(workspace, run_id) / model_name


def refined_model_manifest_path(
    workspace: _WS, model_name: str, run_id: Optional[str] = None,
) -> Path:
    """Bundle manifest (``manifest.json``, schema 2.0).

    Replaces the legacy ``model_pack.json``.
    """
    return refined_model_dir(workspace, model_name, run_id) / "manifest.json"


def refined_model_weights_path(
    workspace: _WS, model_name: str, bucket: str, run_id: Optional[str] = None,
) -> Path:
    """Lifted weights file in the bundle root.

    Post-lift layout (per 2026-05-20 user decision Q3): Cellpose's
    auto-inserted ``models/{name}`` is moved up to the bundle root.
    Backend-agnostic — the file may have no extension (Cellpose),
    a directory suffix (StarDist's SavedModel), etc.  The manifest
    records the exact path; this helper just constructs the canonical
    location.

    Args:
        bucket: "lateral" or "axial".
    """
    _validate_bucket(bucket)
    return refined_model_dir(workspace, model_name, run_id) / bucket


def refined_model_train_loss_path(
    workspace: _WS, model_name: str, bucket: str, run_id: Optional[str] = None,
) -> Path:
    """Per-bucket train loss curve (.npy)."""
    _validate_bucket(bucket)
    return refined_model_dir(workspace, model_name, run_id) / f"{bucket}_train_loss.npy"


def refined_model_val_loss_path(
    workspace: _WS, model_name: str, bucket: str, run_id: Optional[str] = None,
) -> Path:
    """Per-bucket validation loss curve (.npy)."""
    _validate_bucket(bucket)
    return refined_model_dir(workspace, model_name, run_id) / f"{bucket}_val_loss.npy"


def refined_model_training_config_path(
    workspace: _WS, model_name: str, bucket: str, run_id: Optional[str] = None,
) -> Path:
    """Per-bucket Cellpose training config JSON."""
    _validate_bucket(bucket)
    return refined_model_dir(workspace, model_name, run_id) / f"{bucket}_training_config.json"


def refined_model_cellpose_intermediate_path(
    workspace: _WS, model_name: str, bucket: str, run_id: Optional[str] = None,
) -> Path:
    """Where Cellpose ACTUALLY writes during training (before our lift).

    Cellpose unconditionally creates a ``models/{model_name}`` subdir
    under whatever ``save_path`` we hand it.  After training completes,
    the controller MOVES files up from this path to the bundle root
    (see :func:`refined_model_weights_path`).
    """
    _validate_bucket(bucket)
    return refined_model_dir(workspace, model_name, run_id) / "models" / bucket


# ----------------------------------------------------------------------------
# Training archive (future classifier feature)
# ----------------------------------------------------------------------------

def training_archive_root(workspace: _WS) -> Path:
    """``{ws}/artifacts/step3/training_archive/`` — future classifier training data.

    Per-patch snapshots of raw + curated labels + source intensity +
    adjacency + provenance.  Reserved slot — population is a future
    feature; the directory may not exist on most workspaces.
    """
    return step3_root(workspace) / "training_archive"


def patch_training_archive_dir(workspace: _WS, patch_name: str) -> Path:
    """Per-patch training-archive directory."""
    return training_archive_root(workspace) / patch_name


def training_archive_raw_labels_stem(workspace: _WS, patch_name: str) -> Path:
    """Stem of the raw (pre-curation) labels snapshot."""
    return patch_training_archive_dir(workspace, patch_name) / "raw_labels"


def training_archive_curated_labels_stem(workspace: _WS, patch_name: str) -> Path:
    """Stem of the curated (post-curation) labels snapshot."""
    return patch_training_archive_dir(workspace, patch_name) / "curated_labels"


def training_archive_source_stack_stem(workspace: _WS, patch_name: str) -> Path:
    """Stem of the intensity source stack snapshot."""
    return patch_training_archive_dir(workspace, patch_name) / "source_stack"


def training_archive_metadata_path(workspace: _WS, patch_name: str) -> Path:
    """Per-archive metadata JSON (provenance: when curated, params, etc.)."""
    return patch_training_archive_dir(workspace, patch_name) / "metadata.json"


# ----------------------------------------------------------------------------
# Legacy paths (for migration script reference)
# ----------------------------------------------------------------------------

def legacy_artifacts_root(workspace: _WS) -> Path:
    """``{ws}/artifacts/`` — used by both legacy and new layouts."""
    return _ws(workspace) / "artifacts"


def legacy_reconstruction_root(workspace: _WS) -> Path:
    """``{ws}/artifacts/reconstruction/`` (legacy)."""
    return legacy_artifacts_root(workspace) / "reconstruction"


def legacy_qc_dir(workspace: _WS) -> Path:
    """``{ws}/artifacts/qc/`` (legacy — moves to step3/qc/)."""
    return legacy_artifacts_root(workspace) / "qc"


def legacy_snapshots_dir(workspace: _WS) -> Path:
    """``{ws}/artifacts/snapshots/`` (legacy)."""
    return legacy_artifacts_root(workspace) / "snapshots"


def legacy_prediction_cache_dir(workspace: _WS) -> Path:
    """``{ws}/artifacts/prediction_cache/`` (legacy)."""
    return legacy_artifacts_root(workspace) / "prediction_cache"


def legacy_refinement_training_data(workspace: _WS) -> Path:
    """``{ws}/artifacts/refinement_training_data/`` (legacy)."""
    return legacy_artifacts_root(workspace) / "refinement_training_data"


def legacy_models_root(workspace: _WS) -> Path:
    """``{ws}/models/`` — also used by Step 2 HITL (untouched)."""
    return _ws(workspace) / "models"


def legacy_refinement_models_root(workspace: _WS) -> Path:
    """``{ws}/models/refinement/`` (legacy)."""
    return legacy_models_root(workspace) / "refinement"


# ----------------------------------------------------------------------------
# Layout detection
# ----------------------------------------------------------------------------

def detect_layout_version(workspace: _WS) -> int:
    """Detect which Step 3 layout the workspace is on.

    Returns:
        ``LAYOUT_VERSION_NEW`` (2)     — new layout markers present, no
                                          legacy markers
        ``LAYOUT_VERSION_LEGACY`` (1)  — legacy markers present, no new
                                          markers
        ``LAYOUT_VERSION_MIXED`` (3)   — BOTH sets of markers present.
                                          The workspace was likely interrupted
                                          mid-migration; user must resolve.
        ``LAYOUT_VERSION_UNKNOWN`` (0) — no markers (empty workspace or
                                          non-Step-3 workspace)

    The predicates :func:`is_legacy_layout`, :func:`is_new_layout`,
    :func:`is_mixed_layout` consume this return value and agree with
    it — i.e. exactly one predicate returns True for any workspace.
    """
    has_new = _has_new_layout_markers(workspace)
    has_legacy = _has_legacy_layout_markers(workspace)
    if has_new and has_legacy:
        return LAYOUT_VERSION_MIXED
    if has_new:
        return LAYOUT_VERSION_NEW
    if has_legacy:
        return LAYOUT_VERSION_LEGACY
    return LAYOUT_VERSION_UNKNOWN


def is_legacy_layout(workspace: _WS) -> bool:
    """True iff the workspace has only legacy markers."""
    return detect_layout_version(workspace) == LAYOUT_VERSION_LEGACY


def is_new_layout(workspace: _WS) -> bool:
    """True iff the workspace has only new-layout markers."""
    return detect_layout_version(workspace) == LAYOUT_VERSION_NEW


def is_mixed_layout(workspace: _WS) -> bool:
    """True iff BOTH legacy and new markers are present.

    Indicates a partially-migrated workspace.  The migration script
    should detect this and offer either roll-back or completion;
    runtime code paths should warn the user and refuse to operate
    until the state is resolved.
    """
    return detect_layout_version(workspace) == LAYOUT_VERSION_MIXED


def _has_new_layout_markers(workspace: _WS) -> bool:
    """Internal: True if any new-layout marker exists."""
    ws = _ws(workspace)
    return (
        state_json_path(ws).exists()
        or step3_root(ws).is_dir()
    )


def _has_legacy_layout_markers(workspace: _WS) -> bool:
    """Internal: True if any legacy-layout marker exists."""
    ws = _ws(workspace)
    return (
        legacy_state_json_path(ws).exists()
        or legacy_reconstruction_root(ws).is_dir()
        or legacy_refinement_training_data(ws).is_dir()
        or legacy_refinement_models_root(ws).is_dir()
    )


# ----------------------------------------------------------------------------
# Utility validators
# ----------------------------------------------------------------------------

def _validate_bucket(bucket: str) -> None:
    """Raise if ``bucket`` is not a valid orientation name."""
    if bucket not in ("lateral", "axial"):
        raise ValueError(
            f"Invalid bucket {bucket!r}; expected 'lateral' or 'axial'."
        )


# ----------------------------------------------------------------------------
# Model paths on a copied workspace
# ----------------------------------------------------------------------------

_SEPS = chr(92) + "/"


def _split_model_path(path):
    """Parse ``path`` by its own separators.

    Returns ``(parts, is_absolute, flavour)``.  A string with a backslash, a
    drive letter or a ``//server`` prefix is read as a Windows path
    (flavour ``"nt"``), anything else as a posix path (``"posix"``),
    whatever the host is.
    """
    from pathlib import PurePosixPath, PureWindowsPath
    text = str(path or "").strip()
    if not text:
        return (), False, "posix"
    if (chr(92) in text or re.match(r"^[A-Za-z]:", text)
            or re.match(r"^//[^/]", text)):
        pure = PureWindowsPath(text)
        return pure.parts, pure.is_absolute(), "nt"
    pure = PurePosixPath(text)
    return pure.parts, pure.is_absolute(), "posix"


def _norm_part(p):
    return p.replace(chr(92), "/").strip("/") or p


def _anchored(path):
    """(parts, anchor index, flavour) for an absolute anchored path, else None."""
    parts, is_abs, flavour = _split_model_path(path)
    if not is_abs:
        return None
    idx = _last_anchor(parts)
    if idx is None or idx == 0:
        return None
    # An uplevel or no-op segment in the tail could point outside the
    # workspace once rebuilt.
    if any(p in ("..", ".") for p in parts[idx:]):
        return None
    return parts, idx, flavour


def model_path_old_root(path) -> Optional[str]:
    """The workspace folder a stored model path was written under.

    The anchor is the LAST ``artifacts/step2`` or ``artifacts/step3`` pair
    (matched ignoring case); the old root is everything before it, with
    slashes unified and case folded, so two spellings of one folder compare
    equal.  Relative strings, bare names and paths with no anchor give None.
    """
    hit = _anchored(path)
    if hit is None:
        return None
    parts, idx, _flavour = hit
    return "/".join(_norm_part(p) for p in parts[:idx]).casefold()


def _last_anchor(parts):
    low = [p.casefold() for p in parts]
    for i in range(len(low) - 2, -1, -1):
        if low[i] == "artifacts" and low[i + 1] in ("step2", "step3"):
            return i
    return None


def remap_model_path_to_workspace(path, workspace) -> Optional[Path]:
    """Re-root a stored model path onto ``workspace``, or None.

    Returns ``workspace / artifacts / step{2,3} / ...`` (the tail from the
    last anchor pair) only when the path is absolute, has an anchor, was
    written under a different folder than ``workspace``, that other folder
    does not exist on this machine at all (a real copy to another machine
    never has it; a folder that is still here is a live workspace whose
    models must be left alone), and the rebuilt path exists.  A model path
    never defines the workspace root for anything else: an imported model
    may live in another workspace, so anything doubtful returns None and
    the caller keeps its own behaviour.
    """
    if not workspace:
        return None
    hit = _anchored(path)
    if hit is None:
        return None
    parts, idx, flavour = hit
    ws = Path(workspace)
    old_root = model_path_old_root(path)
    ws_key = "/".join(_norm_part(p) for p in ws.parts).casefold()
    if old_root is None or old_root == ws_key:
        return None
    # A string written for the other kind of machine cannot name a folder
    # here, and is never resolved against the working directory.
    if flavour == os.name:
        try:
            if Path(*parts[:idx]).exists():
                return None
        except (OSError, ValueError):
            return None
    # The anchor matched ignoring case; the rebuilt path spells it the way
    # the plugin writes the folders, so it is found on a case-sensitive
    # filesystem too.
    rebuilt = ws.joinpath("artifacts", parts[idx + 1].casefold(), *parts[idx + 2:])
    try:
        return rebuilt if rebuilt.exists() else None
    except OSError:
        return None


__all__ = [
    # Constants
    "STEP3_STATE_FILENAME",
    "LEGACY_STATE_FILENAME",
    "STEP3_ARTIFACTS_SUBDIR",
    "RUN_MANIFEST_FILENAME",
    "RUN_ID_MAX_LENGTH",
    "LAYOUT_VERSION_UNKNOWN",
    "LAYOUT_VERSION_LEGACY",
    "LAYOUT_VERSION_NEW",
    "LAYOUT_VERSION_MIXED",
    "CURRENT_LAYOUT_VERSION",
    # Root
    "workspace_root",
    "state_json_path",
    "legacy_state_json_path",
    "step3_root",
    "step3_artifacts_root",
    # Runs
    "runs_dir",
    "reconstructions_dir",
    "patch_run_dir",
    "patch_labels_stem",
    "patch_run_manifest_path",
    # Run-context helpers (named runs)
    "step3_runs_root",
    "run_root",
    "run_manifest_path",
    "validate_run_id",
    # QC + adjacency
    "qc_dir",
    "qc_features_csv",
    "qc_features_raw_csv",
    "qc_adjacency_raw_csv",
    "qc_adjacency_curated_csv",
    # Snapshots
    "snapshots_dir",
    # Prediction cache
    "prediction_cache_dir",
    # Batch-refine recipes
    "batch_refine_recipes_path",
    "refine_chain_recipes_path",
    # Curation
    "curation_root",
    "patch_curation_dir",
    "patch_curation_labels_stem",
    "patch_curation_labels_bak_stem",
    "patch_curation_tiers_csv",
    # Refinement
    "refinement_root",
    "step3_refinement_config_path",
    "legacy_step3_refinement_config_path",
    "refinement_manifest_path",
    "refinement_pairs_root",
    "patch_pairs_dir",
    "patch_pairs_spatial_dir",
    "patch_pairs_axial_dir",
    # Refined model bundles
    "refined_models_root",
    "refined_model_dir",
    "refined_model_manifest_path",
    "refined_model_weights_path",
    "refined_model_train_loss_path",
    "refined_model_val_loss_path",
    "refined_model_training_config_path",
    "refined_model_cellpose_intermediate_path",
    # Training archive
    "training_archive_root",
    "patch_training_archive_dir",
    "training_archive_raw_labels_stem",
    "training_archive_curated_labels_stem",
    "training_archive_source_stack_stem",
    "training_archive_metadata_path",
    # Legacy (for migration script)
    "legacy_artifacts_root",
    "legacy_reconstruction_root",
    "legacy_qc_dir",
    "legacy_snapshots_dir",
    "legacy_prediction_cache_dir",
    "legacy_refinement_training_data",
    "legacy_models_root",
    "legacy_refinement_models_root",
    # Model paths on a copied workspace
    "model_path_old_root",
    "remap_model_path_to_workspace",
    # Layout detection
    "detect_layout_version",
    "is_legacy_layout",
    "is_new_layout",
    "is_mixed_layout",
]

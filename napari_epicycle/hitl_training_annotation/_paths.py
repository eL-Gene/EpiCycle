"""Centralized Step 2 (HITL 2D) workspace path constants and helpers.

**Source of truth** for every filesystem path that the
``hitl_training_annotation`` package reads or writes in the new
multi-run + bucket-first layout.

Two layouts are recognized:

* **Legacy** (pre-restructure): ``training_data/round_NN/``,
  ``models/round_NN/``, ``predictions/round_NN/`` plus
  ``artifacts/{models,inference,benchmarks}/``.  State JSON
  filename is ``hitl_state.json`` (schema v2).

* **New (v3)** (post-restructure): everything under
  ``artifacts/step2/runs/{run_id}/`` with bucket-first structure
  (``{lateral,axial}/rounds/round_NN/`` for training data,
  ``{lateral,axial}/models/round_NN/`` for trained models).
  State JSON renamed to ``step2_state.json``; schema bumped to v3.

This module provides:

1. Path constructors — pure helpers, no filesystem mutation.
   ``mkdir`` is the caller's responsibility.
2. Slug + slice-label parsing helpers (replacing scattered
   ``rsplit("_", 2)`` sites).
3. Legacy path aliases for the migration code.
4. Layout detection — ``detect_layout_version(ws) -> int``.

**Naming convention for trained-model files** (Cellpose-driven):
Step 2 invokes Cellpose with ``model_name=f"{run_id}_{bucket}"``
(post Phase 6c followup `7a36217`), so the post-lift filenames are
predictable:

* ``{run_id}_{bucket}`` (no extension — the actual weights file)
* ``{run_id}_{bucket}_train_loss.npy``
* ``{run_id}_{bucket}_val_loss.npy`` (Cellpose's actual on-disk name
  — abbreviated, NOT the full word "validation")
* ``{run_id}_{bucket}_training_config.json`` (if Cellpose writes one)

The lift function MOVES these files AS-IS from Cellpose's auto-
created ``models/`` subdir up one level into the canonical model
dir.  No filename rewriting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Union


# ----------------------------------------------------------------------------
# Filename + layout constants
# ----------------------------------------------------------------------------

#: State JSON filename in the new layout (v3).
STEP2_STATE_FILENAME = "step2_state.json"

#: Legacy state JSON filename (v2, pre-restructure).
LEGACY_STATE_FILENAME = "hitl_state.json"

#: Sub-tree name under ``artifacts/`` for the new layout.
STEP2_ARTIFACTS_SUBDIR = "step2"

#: Default run slug created during migration.  All pre-existing
#: workspace data is wrapped into this run.
DEFAULT_RUN_ID = "default"

#: Per-run manifest filename (lives at ``runs/{run_id}/``).
RUN_MANIFEST_FILENAME = "run_manifest.json"

#: Per-bucket-round model manifest filename (lives at
#: ``{bucket}/models/round_NN/``).
MODEL_MANIFEST_FILENAME = "manifest.json"

#: Per-round metadata filename (lives at ``{bucket}/rounds/round_NN/``).
ROUND_METADATA_FILENAME = "round_metadata.json"

#: Per-round selected-indices filename.
SELECTED_INDICES_FILENAME = "selected_indices.json"

#: Per-round pairs-manifest filename (settings provenance).
PAIRS_MANIFEST_FILENAME = "pairs_manifest.json"

#: Per-round predictions manifest filename (lives at
#: ``{bucket}/rounds/round_NN/predictions/``).
PREDICTIONS_MANIFEST_FILENAME = "manifest.json"

#: Cellpose-driven filename constants.  Step 2 passes
#: ``model_name="weights"`` at training-invoke time, so Cellpose
#: writes these exact filenames; the lift function moves them
#: AS-IS into the canonical model dir.
WEIGHTS_FILENAME = "weights"
TRAIN_LOSS_FILENAME = "weights_train_loss.npy"
VAL_LOSS_FILENAME = "weights_val_loss.npy"  # Cellpose writes _val_loss.npy (matches Step 3)
TRAINING_CONFIG_FILENAME = "weights_training_config.json"

#: Allowed bucket names in the NEW layout.  Note that ``"spatial"`` is
#: a LEGACY-only name; migration translates ``spatial/`` → ``lateral/``.
#: New code must use ``"lateral"``.
VALID_BUCKETS: Tuple[str, str] = ("lateral", "axial")

#: Allowed 3D-orientation VIEW codes in slice labels.  These remain
#: in filenames forever (they describe slice orientation, not bucket).
VALID_VIEWS: Tuple[str, str, str] = ("XY", "XZ", "YZ")

#: Layout version constants.  ``LAYOUT_VERSION_MIXED`` is a runtime-
#: only sentinel — never persisted; indicates a partially-migrated
#: workspace that the user must resolve before operations resume.
LAYOUT_VERSION_UNKNOWN = 0
LAYOUT_VERSION_LEGACY = 1
LAYOUT_VERSION_NEW = 2
LAYOUT_VERSION_MIXED = 3
CURRENT_LAYOUT_VERSION = LAYOUT_VERSION_NEW

#: Maximum length of a run_id slug (filesystem-friendly).
RUN_ID_MAX_LENGTH = 64

#: Slug pattern — alphanumeric + underscore + dash, starts with
#: alphanumeric (not _ or -), max RUN_ID_MAX_LENGTH chars.
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

#: Slice-label parsing regex.  Finds the LAST occurrence of the
#: ``_{VIEW}_z{digits}`` pattern so underscored stack_ids parse
#: correctly (e.g., ``my_patch_v2_XY_z0042`` → stack_id="my_patch_v2").
#: Anchored at start + end so trailing garbage is rejected.
_SLICE_LABEL_RE = re.compile(r"^(.+)_(XY|XZ|YZ)_z(\d+)$")


# Type alias for "anything that can be turned into a Path".  Mirrors
# Step 3's _paths.py convention so the two modules feel symmetric.
_WS = Union[str, Path]


def _ws(workspace: _WS) -> Path:
    """Coerce the workspace argument to a ``Path``.  Helper internal."""
    return Path(workspace)


def _validate_bucket(bucket: str) -> None:
    """Raise ValueError if ``bucket`` is not a valid NEW-layout bucket."""
    if bucket not in VALID_BUCKETS:
        raise ValueError(
            f"bucket must be one of {VALID_BUCKETS}, got {bucket!r}. "
            f"Note: 'spatial' is a LEGACY-only name; new code uses 'lateral'."
        )


def _round_segment(round_idx: int) -> str:
    """Zero-padded ``round_NN`` directory name.

    Pads to 2 digits for ``round_idx < 100``; uses full width above.
    Raises ``ValueError`` for negative input.
    """
    if not isinstance(round_idx, int) or isinstance(round_idx, bool):
        raise ValueError(
            f"round_idx must be int, got {type(round_idx).__name__}"
        )
    if round_idx < 0:
        raise ValueError(f"round_idx must be non-negative, got {round_idx}")
    return f"round_{round_idx:02d}"


# ----------------------------------------------------------------------------
# Workspace root + state JSON
# ----------------------------------------------------------------------------

def workspace_root(workspace: _WS) -> Path:
    """The workspace directory itself."""
    return _ws(workspace)


def state_json_path(workspace: _WS) -> Path:
    """``{ws}/step2_state.json`` — new-layout state JSON at workspace root.

    Matches Step 3's convention (`step3_state.json` at workspace root).
    """
    return _ws(workspace) / STEP2_STATE_FILENAME


def legacy_state_json_path(workspace: _WS) -> Path:
    """``{ws}/hitl_state.json`` — legacy state JSON at workspace root."""
    return _ws(workspace) / LEGACY_STATE_FILENAME


# ----------------------------------------------------------------------------
# Step 2 root + runs root
# ----------------------------------------------------------------------------

def step2_root(workspace: _WS) -> Path:
    """``{ws}/artifacts/step2/`` — the canonical Step 2 sub-tree."""
    return _ws(workspace) / "artifacts" / STEP2_ARTIFACTS_SUBDIR


def step2_runs_root(workspace: _WS) -> Path:
    """``{ws}/artifacts/step2/runs/`` — parent of all per-run dirs."""
    return step2_root(workspace) / "runs"


# ----------------------------------------------------------------------------
# Per-run paths
# ----------------------------------------------------------------------------

def run_root(workspace: _WS, run_id: str) -> Path:
    """``{ws}/artifacts/step2/runs/{run_id}/`` — single run's root."""
    validate_run_id(run_id)
    return step2_runs_root(workspace) / run_id


def run_manifest_path(workspace: _WS, run_id: str) -> Path:
    """Path to the per-run ``run_manifest.json``."""
    return run_root(workspace, run_id) / RUN_MANIFEST_FILENAME


# ----------------------------------------------------------------------------
# Bucket paths (lateral / axial)
# ----------------------------------------------------------------------------

def bucket_root(workspace: _WS, run_id: str, bucket: str) -> Path:
    """``{run_root}/{bucket}/`` — one bucket within a run."""
    _validate_bucket(bucket)
    return run_root(workspace, run_id) / bucket


def bucket_rounds_root(workspace: _WS, run_id: str, bucket: str) -> Path:
    """``{bucket_root}/rounds/`` — Phase B.2 LEGACY split layout helper.

    Pre-Phase-B.2 the per-round dirs lived under ``{bucket}/rounds/``.
    Post-Phase-B.2 they live flat at ``{bucket}/round_NN/``.  This
    helper is retained for dual-path reading of legacy data; the
    canonical reader is `round_dir`.
    """
    return bucket_root(workspace, run_id, bucket) / "rounds"


def bucket_models_root(workspace: _WS, run_id: str, bucket: str) -> Path:
    """``{bucket_root}/models/`` — Phase B.2 LEGACY split layout helper.

    Pre-Phase-B.2 the per-round model dirs lived at
    ``{bucket}/models/round_NN/``.  Post-Phase-B.2 they live at
    ``{bucket}/round_NN/model/`` (under the round dir, co-located with
    accepted/ and predictions/).  This helper is retained for dual-
    path reading of legacy data; the canonical reader is `model_dir`.
    """
    return bucket_root(workspace, run_id, bucket) / "models"


# ----------------------------------------------------------------------------
# Per-round data paths (training pairs, predictions, metadata)
#
# Phase B.2 (2026-05-25) FLAT LAYOUT:
#   round_dir   = {bucket}/round_NN/         (was {bucket}/rounds/round_NN/)
#   model_dir   = {bucket}/round_NN/model/   (was {bucket}/models/round_NN/)
# Legacy SPLIT layout is reachable via `legacy_split_round_dir` +
# `legacy_split_model_dir` for dual-path readers during migration.
# ----------------------------------------------------------------------------

def round_dir(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """``{bucket}/round_NN/`` — flat per-round directory (Phase B.2).

    Contents: ``accepted/``, ``predictions/``, ``round_metadata.json``,
    ``selected_indices.json``, ``pairs_manifest.json``, and the
    co-located ``model/`` subdir (via `model_dir`).
    """
    return bucket_root(workspace, run_id, bucket) / _round_segment(round_idx)


def legacy_split_round_dir(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """Pre-Phase-B.2 LEGACY ``{bucket}/rounds/round_NN/`` path.

    Use this for dual-path readers (e.g. cumulative-training prior-
    pair scanner) to find legacy data not yet migrated.  Writers
    should use `round_dir` (the canonical flat path) so new outputs
    land in the post-Phase-B.2 layout.
    """
    return bucket_rounds_root(workspace, run_id, bucket) / _round_segment(round_idx)


def round_accepted_dir(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """``{round_dir}/accepted/`` — curated training pairs (source of truth)."""
    return round_dir(workspace, run_id, bucket, round_idx) / "accepted"


def round_staged_dir(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """``{round_dir}/staged/`` — staged pairs awaiting training."""
    return round_dir(workspace, run_id, bucket, round_idx) / "staged"


def round_predictions_dir(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """``{round_dir}/predictions/`` — per-round inference outputs."""
    return round_dir(workspace, run_id, bucket, round_idx) / "predictions"


def round_predictions_manifest_path(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """Path to ``{round_dir}/predictions/manifest.json``."""
    return round_predictions_dir(workspace, run_id, bucket, round_idx) / PREDICTIONS_MANIFEST_FILENAME


def round_metadata_path(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """Path to ``{round_dir}/round_metadata.json`` (SliceAnnotationRecord[])."""
    return round_dir(workspace, run_id, bucket, round_idx) / ROUND_METADATA_FILENAME


def round_pairs_manifest_path(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """Path to ``{round_dir}/pairs_manifest.json`` (extraction settings provenance)."""
    return round_dir(workspace, run_id, bucket, round_idx) / PAIRS_MANIFEST_FILENAME


def round_selected_indices_path(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """Path to ``{round_dir}/selected_indices.json``."""
    return round_dir(workspace, run_id, bucket, round_idx) / SELECTED_INDICES_FILENAME


# ----------------------------------------------------------------------------
# Per-round model paths (weights, manifest, loss curves, training config)
# ----------------------------------------------------------------------------

def model_dir(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """``{bucket}/round_NN/model/`` — per-round trained model bundle dir.

    Phase B.2 (2026-05-25) flat layout: the model dir now lives UNDER
    the round dir (sibling of accepted/, predictions/) so model +
    training data + predictions for one round are structurally
    co-located.  Pre-B.2 path was ``{bucket}/models/round_NN/``;
    that's still reachable via `legacy_split_model_dir` for dual-path
    readers.
    """
    return round_dir(workspace, run_id, bucket, round_idx) / "model"


def legacy_split_model_dir(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """Pre-Phase-B.2 LEGACY ``{bucket}/models/round_NN/`` path.

    Use this for dual-path readers (Step 3 picker, Step 2 init-source
    enumerator) to find legacy models not yet migrated.  Writers
    should use `model_dir` (the canonical flat path) so new outputs
    land in the post-Phase-B.2 layout.
    """
    return bucket_models_root(workspace, run_id, bucket) / _round_segment(round_idx)


def model_weights_path(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """``{model_dir}/weights`` — STALE helper retained for back-compat.

    Phase 6c followup `7a36217` shifted the canonical weights filename
    from bare ``"weights"`` to dynamic ``{run_id}_{bucket}``.  Phase B.2
    further extended to ``{run_id}_{bucket}_round_NN`` (round-id-in-
    name).  Production callers build the dynamic filename inline
    (`f"{run_id}_{bucket}_round_{round_idx:02d}"`); this helper still
    returns the bare-``"weights"`` legacy path purely for test fixtures
    that haven't been migrated.  DO NOT use in new code.
    """
    return model_dir(workspace, run_id, bucket, round_idx) / WEIGHTS_FILENAME


def model_manifest_path(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """Path to the per-bucket-round ``manifest.json`` (schema 2.0)."""
    return model_dir(workspace, run_id, bucket, round_idx) / MODEL_MANIFEST_FILENAME


def model_manifest_history_dir(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """``{model_dir}/manifest_history/`` — archived prior manifests.

    Phase B.2: hoisted from controller-inline construction in
    `_write_step2_model_manifest_for_round` (C-5 archive feature) to
    a first-class helper so readers can enumerate it consistently.
    """
    return model_dir(workspace, run_id, bucket, round_idx) / "manifest_history"


def model_loss_path(
    workspace: _WS, run_id: str, bucket: str, round_idx: int, kind: str
) -> Path:
    """Path to a Cellpose-written loss ``.npy`` file.

    ``kind`` must be ``"train"`` or ``"val"``.  Returns the
    Cellpose-driven filename (``weights_train_loss.npy`` or
    ``weights_val_loss.npy`` — abbreviated as Cellpose actually
    writes on disk per ``cellpose_trainer.py:237``; the lift function
    does NOT rename).  Note: the ``weights`` prefix is illustrative —
    the actual filename is ``{model_name}_val_loss.npy`` where
    model_name is the canonical ``{run_id}_{bucket}`` post-Phase-6c.
    """
    if kind == "train":
        filename = TRAIN_LOSS_FILENAME
    elif kind == "val":
        filename = VAL_LOSS_FILENAME
    else:
        raise ValueError(f"kind must be 'train' or 'val', got {kind!r}")
    return model_dir(workspace, run_id, bucket, round_idx) / filename


def model_training_config_path(
    workspace: _WS, run_id: str, bucket: str, round_idx: int
) -> Path:
    """Path to the Cellpose-written per-bucket-round training config JSON."""
    return model_dir(workspace, run_id, bucket, round_idx) / TRAINING_CONFIG_FILENAME


# ----------------------------------------------------------------------------
# Slug + slice-label parsing helpers
# ----------------------------------------------------------------------------

def validate_run_id(run_id: str) -> None:
    """Raise ``ValueError`` if ``run_id`` is not a valid run handle.

    Run-handle rules: alphanumeric + underscore + dash, starts with
    alphanumeric (not _ or -), max ``RUN_ID_MAX_LENGTH`` chars,
    case-sensitive.  Designed to be filesystem-friendly on every OS.

    Rejects path-traversal patterns (``../``, ``/``, ``\\``) and
    any character outside ``[A-Za-z0-9_-]``.

    Note (2026-05-25 terminology sweep): the error messages emitted here
    use ``"Run handle"`` (user-facing term) instead of ``"run_id"``
    (internal field name).  Callers that surface these errors directly
    to the user (e.g. `validate_slug_input` in `_new_run_dialog.py`)
    get clean strings; internal stack traces still show the symbol name
    in tracebacks for diagnostic purposes.
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


def parse_slice_label(label: str) -> Tuple[str, str, int]:
    """Parse a slice label ``{stack_id}_{VIEW}_z{idx}`` into components.

    Handles underscored stack_ids via right-to-left anchored regex.
    Optionally strips a trailing ``_img`` / ``_mask`` / ``_pred_mask``
    suffix so filename stems work too.

    Returns ``(stack_id, view, slice_idx)`` tuple.  Raises ``ValueError``
    on any parse failure.

    Replaces the ~7 ``rsplit("_", 2)`` sites in
    ``training_annotation_controller.py`` (a single parser instead of scattered splits).

    Examples::

        >>> parse_slice_label("SAMPLE01_XY_z0042")
        ('SAMPLE01', 'XY', 42)
        >>> parse_slice_label("my_patch_v2_XY_z0001")
        ('my_patch_v2', 'XY', 1)
        >>> parse_slice_label("weird name with spaces_YZ_z0007")
        ('weird name with spaces', 'YZ', 7)
        >>> parse_slice_label("SAMPLE01_XY_z0042_img")
        ('SAMPLE01', 'XY', 42)
    """
    if not isinstance(label, str) or not label:
        raise ValueError(f"label must be non-empty str, got {label!r}")

    # Strip known suffixes if present (caller may pass a filename stem).
    # Order matters: check longest suffix first ("_pred_mask" before "_mask").
    stripped = label
    for suffix in ("_pred_mask", "_mask", "_img"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            break

    match = _SLICE_LABEL_RE.match(stripped)
    if match is None:
        raise ValueError(
            f"label {label!r} does not match expected pattern "
            f"'{{stack_id}}_{{VIEW}}_z{{idx}}' (VIEW ∈ {VALID_VIEWS})"
        )

    stack_id, view, idx_str = match.groups()
    try:
        slice_idx = int(idx_str)
    except ValueError as e:
        raise ValueError(
            f"label {label!r} has non-integer slice index {idx_str!r}"
        ) from e

    return stack_id, view, slice_idx


def bucket_for_view(view: str) -> str:
    """Map a 3D-orientation VIEW code to its bucket name.

    ``XY`` → ``lateral``.  ``XZ``/``YZ`` → ``axial``.

    Used by the migration to fan out legacy ``accepted_pairs/{spatial,axial}/``
    into the new bucket-first layout by reading each file's VIEW code.
    """
    if view == "XY":
        return "lateral"
    elif view in ("XZ", "YZ"):
        return "axial"
    else:
        raise ValueError(f"view must be one of {VALID_VIEWS}, got {view!r}")


# ----------------------------------------------------------------------------
# Legacy path aliases (for migration code)
# ----------------------------------------------------------------------------
#
# These are the OLD-layout paths the migration code needs to detect +
# read.  Kept here so the migration logic can use the same module as
# the new-layout writers — single source of truth for "where things
# live" on both sides of the move.
#
# These are STRIP-AT-RELEASE.  Once the migration tooling is removed
# these aliases go with it.

def legacy_training_data_root(workspace: _WS) -> Path:
    """``{ws}/training_data/`` — pre-restructure training-data root."""
    return _ws(workspace) / "training_data"


def legacy_round_dir(workspace: _WS, round_idx: int) -> Path:
    """``{ws}/training_data/round_NN/`` — pre-restructure round dir."""
    return legacy_training_data_root(workspace) / _round_segment(round_idx)


def legacy_round_accepted_pairs_dir(
    workspace: _WS, round_idx: int, legacy_bucket: str
) -> Path:
    """``{ws}/training_data/round_NN/accepted_pairs/{spatial|axial}/``.

    Note: ``legacy_bucket`` is ``"spatial"`` or ``"axial"`` (the OLD
    names).  Migration translates ``spatial`` → ``lateral`` when
    fanning out to the new bucket-first layout.
    """
    if legacy_bucket not in ("spatial", "axial"):
        raise ValueError(
            f"legacy_bucket must be 'spatial' or 'axial' (legacy "
            f"names), got {legacy_bucket!r}.  For new-layout paths "
            f"use bucket ∈ {VALID_BUCKETS}."
        )
    return legacy_round_dir(workspace, round_idx) / "accepted_pairs" / legacy_bucket


def legacy_round_staged_pairs_dir(
    workspace: _WS, round_idx: int, legacy_bucket: str
) -> Path:
    """``{ws}/training_data/round_NN/staged_pairs/{spatial|axial}/``."""
    if legacy_bucket not in ("spatial", "axial"):
        raise ValueError(
            f"legacy_bucket must be 'spatial' or 'axial', got {legacy_bucket!r}"
        )
    return legacy_round_dir(workspace, round_idx) / "staged_pairs" / legacy_bucket


def legacy_models_root(workspace: _WS) -> Path:
    """``{ws}/models/`` — pre-restructure trained-models root."""
    return _ws(workspace) / "models"


def legacy_round_models_dir(workspace: _WS, round_idx: int) -> Path:
    """``{ws}/models/round_NN/`` — pre-restructure per-round models."""
    return legacy_models_root(workspace) / _round_segment(round_idx)


def legacy_predictions_root(workspace: _WS) -> Path:
    """``{ws}/predictions/`` — pre-restructure predictions root."""
    return _ws(workspace) / "predictions"


def legacy_round_predictions_dir(workspace: _WS, round_idx: int) -> Path:
    """``{ws}/predictions/round_NN/`` — pre-restructure per-round predictions."""
    return legacy_predictions_root(workspace) / _round_segment(round_idx)


def legacy_artifacts_models_root(workspace: _WS) -> Path:
    """``{ws}/artifacts/models/`` — pre-restructure ModelArtifact registry."""
    return _ws(workspace) / "artifacts" / "models"


def legacy_artifacts_inference_root(workspace: _WS) -> Path:
    """``{ws}/artifacts/inference/`` — pre-restructure InferenceArtifact registry."""
    return _ws(workspace) / "artifacts" / "inference"


def legacy_artifacts_benchmarks_root(workspace: _WS) -> Path:
    """``{ws}/artifacts/benchmarks/`` — pre-restructure benchmarks root."""
    return _ws(workspace) / "artifacts" / "benchmarks"


# ----------------------------------------------------------------------------
# Layout detection
# ----------------------------------------------------------------------------

def detect_layout_version(workspace: _WS) -> int:
    """Return the layout version of a workspace.

    Returns:
        * ``LAYOUT_VERSION_UNKNOWN`` — no Step 2 markers found
        * ``LAYOUT_VERSION_LEGACY`` — has ``training_data/`` or
          ``models/round_*/`` but no ``artifacts/step2/runs/``
        * ``LAYOUT_VERSION_NEW`` — has ``artifacts/step2/runs/``
        * ``LAYOUT_VERSION_MIXED`` — has both legacy AND new markers
          (partially-migrated; resolve before further operations)
    """
    ws = _ws(workspace)

    has_new = (ws / "artifacts" / STEP2_ARTIFACTS_SUBDIR / "runs").is_dir()

    has_legacy_training = (ws / "training_data").is_dir()

    has_legacy_models = False
    legacy_models = ws / "models"
    if legacy_models.is_dir():
        try:
            has_legacy_models = any(
                d.is_dir() and d.name.startswith("round_")
                for d in legacy_models.iterdir()
            )
        except (OSError, PermissionError):
            has_legacy_models = False

    has_legacy = has_legacy_training or has_legacy_models

    if has_new and has_legacy:
        return LAYOUT_VERSION_MIXED
    if has_new:
        return LAYOUT_VERSION_NEW
    if has_legacy:
        return LAYOUT_VERSION_LEGACY
    return LAYOUT_VERSION_UNKNOWN


def is_mixed_layout(workspace: _WS) -> bool:
    """Convenience predicate: workspace has both legacy + new markers."""
    return detect_layout_version(workspace) == LAYOUT_VERSION_MIXED


# ----------------------------------------------------------------------------
# State loading — unified v3-first / v2-fallback helper
# ----------------------------------------------------------------------------

def load_step2_state_flat(
    workspace: _WS,
    active_run_id: str = DEFAULT_RUN_ID,
):
    """Load Step 2 state as a flat dict (controller-compatible shape).

    Tries v3 (``step2_state.json``) first; falls back to v2 legacy
    (``hitl_state.json``).  This is the bridge that lets the existing
    controller (which expects flat v2 keys) function on a migrated
    workspace without requiring a full controller refactor.

    Behavior:
    * v3 found: unwraps ``runs[active_run_id]`` into a flat dict;
      merges top-level workspace-level keys; returns the merged dict.
    * v2 legacy found: returns as-is (already flat).
    * Neither found: returns ``None``.

    Args:
        workspace: workspace root path.
        active_run_id: which run to unwrap (defaults to "default";
            future Phase 8 UI passes the user-selected active run).

    Returns:
        Flat state dict, or ``None`` if no state file exists.
    """
    import json
    ws = _ws(workspace)

    # Try v3 first
    v3_path = state_json_path(ws)
    if v3_path.is_file():
        try:
            with open(v3_path, "r", encoding="utf-8") as f:
                v3 = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            v3 = None
        if isinstance(v3, dict):
            return _unwrap_v3_state(v3, active_run_id)

    # Fallback to v2 legacy
    v2_path = legacy_state_json_path(ws)
    if v2_path.is_file():
        try:
            with open(v2_path, "r", encoding="utf-8") as f:
                v2 = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            v2 = None
        if isinstance(v2, dict):
            return v2

    return None


def _unwrap_v3_state(v3: dict, active_run_id: str) -> dict:
    """Convert a v3 ``step2_state.json`` dict into the flat shape the
    controller expects (v2-compatible).

    Merges the run's per-run state with the top-level workspace-level
    keys.  The result is a flat dict — exactly what
    ``_load_internal_state`` consumed pre-restructure.
    """
    flat = {}

    # Start with workspace-level keys (top-level v3 keys other than runs/
    # active_run_id/schema_version which are v3-specific).
    SKIP_KEYS = {"schema_version", "active_run_id", "runs"}
    for k, v in v3.items():
        if k not in SKIP_KEYS:
            flat[k] = v

    # Layer in the per-run state.  If active_run_id is missing or its
    # entry doesn't exist, fall back to "default" (defensive).
    runs = v3.get("runs", {})
    if not isinstance(runs, dict):
        runs = {}
    run_state = runs.get(active_run_id)
    if not isinstance(run_state, dict):
        run_state = runs.get(DEFAULT_RUN_ID, {})
    if not isinstance(run_state, dict):
        run_state = {}

    # Skip run-manifest-only keys that shouldn't leak into flat state
    RUN_MANIFEST_ONLY = {"display_name", "target", "backend", "created_at", "created_from"}
    for k, v in run_state.items():
        if k not in RUN_MANIFEST_ONLY:
            flat[k] = v

    # Always include schema_version so callers can detect what they loaded
    # (legacy callers expecting v2 will see schema_version=3 — they should
    # treat any version >= 2 as "modern enough" or check explicitly).
    flat["schema_version"] = v3.get("schema_version", 3)
    return flat


def is_new_layout_state(workspace: _WS) -> bool:
    """True if the workspace has a v3 ``step2_state.json`` on disk."""
    return state_json_path(_ws(workspace)).is_file()


# ----------------------------------------------------------------------------
# Public API surface
# ----------------------------------------------------------------------------
#
# Explicit ``__all__`` mirrors Step 3's `_paths.py` convention.  Internal
# helpers (``_ws``, ``_validate_bucket``, ``_round_segment``, the regex
# patterns) are intentionally NOT exported.
#
# ``parse_slice_label`` is exported as PUBLIC (not ``_parse_slice_label``
# as first sketched) because:
#   * migration code will import + call it
#   * test fixtures use it for adversarial-filename generation
#   * future external tooling (Phase 7B picker, etc.) benefits from
#     a centralized parser
# The leading-underscore convention was a sketch convention, not strict.

__all__ = [
    # Filename + layout constants
    "STEP2_STATE_FILENAME",
    "LEGACY_STATE_FILENAME",
    "STEP2_ARTIFACTS_SUBDIR",
    "DEFAULT_RUN_ID",
    "RUN_MANIFEST_FILENAME",
    "MODEL_MANIFEST_FILENAME",
    "ROUND_METADATA_FILENAME",
    "SELECTED_INDICES_FILENAME",
    "PAIRS_MANIFEST_FILENAME",
    "PREDICTIONS_MANIFEST_FILENAME",
    "WEIGHTS_FILENAME",
    "TRAIN_LOSS_FILENAME",
    "VAL_LOSS_FILENAME",
    "TRAINING_CONFIG_FILENAME",
    "VALID_BUCKETS",
    "VALID_VIEWS",
    "LAYOUT_VERSION_UNKNOWN",
    "LAYOUT_VERSION_LEGACY",
    "LAYOUT_VERSION_NEW",
    "LAYOUT_VERSION_MIXED",
    "CURRENT_LAYOUT_VERSION",
    "RUN_ID_MAX_LENGTH",
    # Workspace root + state JSON
    "workspace_root",
    "state_json_path",
    "legacy_state_json_path",
    # Step 2 root + runs
    "step2_root",
    "step2_runs_root",
    # Per-run paths
    "run_root",
    "run_manifest_path",
    # Bucket paths
    "bucket_root",
    "bucket_rounds_root",
    "bucket_models_root",
    # Per-round data paths
    "round_dir",
    "round_accepted_dir",
    "round_staged_dir",
    "round_predictions_dir",
    "round_predictions_manifest_path",
    "round_metadata_path",
    "round_pairs_manifest_path",
    "round_selected_indices_path",
    # Per-round model paths
    "model_dir",
    "model_weights_path",
    "model_manifest_path",
    "model_manifest_history_dir",
    "model_loss_path",
    "model_training_config_path",
    # Slug + parsing helpers
    "validate_run_id",
    "parse_slice_label",
    "bucket_for_view",
    # Phase B.2 LEGACY split-layout helpers (pre-Phase-B.2 paths
    # retained for dual-path readers + migration tooling; strip after
    # Phase D migration window).
    "legacy_split_round_dir",
    "legacy_split_model_dir",
    # Legacy aliases (for migration code; strip at release)
    "legacy_training_data_root",
    "legacy_round_dir",
    "legacy_round_accepted_pairs_dir",
    "legacy_round_staged_pairs_dir",
    "legacy_models_root",
    "legacy_round_models_dir",
    "legacy_predictions_root",
    "legacy_round_predictions_dir",
    "legacy_artifacts_models_root",
    "legacy_artifacts_inference_root",
    "legacy_artifacts_benchmarks_root",
    # Layout detection
    "detect_layout_version",
    "is_mixed_layout",
    "is_new_layout_state",
    # State loading (unified v3-first / v2-fallback)
    "load_step2_state_flat",
]

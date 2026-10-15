"""Step 2 layout migration: legacy → new (v3) on-disk reshape.

Phase 2 of the Step 2 directory restructure.

Two-stage flow mirrors Step 3's migration pattern:

1. :func:`plan_migration` — READ-ONLY scan.  Returns a
   :class:`MigrationPlan` describing every file move, metadata split,
   state JSON rewrite, and run-manifest init that WOULD happen.  No
   filesystem mutation.  Safe to call repeatedly.

2. :func:`execute_migration` — applies the plan.  Mutates the
   workspace.  Caller MUST have created a backup via
   :func:`create_backup` first (the dialog enforces this).

Rollback: :func:`rollback_migration` restores from the backup
directory.  Idempotent — re-running on an already-rolled-back
workspace is a no-op.

Pure-Python — no Qt / napari dependency.  All Qt-related interaction
(themed dialog, user choices) lives in the sibling
``_migrate_step2_dialog`` module so this one can be unit-tested
without a display.

Key Step 2-specific migration challenges (vs Step 3):

* **Bucket fan-out** — legacy ``training_data/round_NN/accepted_pairs/
  {spatial,axial}/`` must be split into new
  ``runs/default/{lateral,axial}/rounds/round_NN/accepted/`` by
  parsing each file's VIEW code (``XY`` → lateral, ``XZ``/``YZ`` →
  axial).  Mis-sorted files (e.g., an ``XY`` file in legacy
  ``axial/``) HARD-FAIL (see the validation rules below).

* **Metadata splits** — legacy ``round_metadata.json`` and
  ``selected_indices.json`` are joint (cover both buckets in one
  file).  New layout is per-bucket.  Split happens in-memory during
  execute.

* **Schema wrap** — legacy ``hitl_state.json`` (v2) wraps into new
  ``step2_state.json`` (v3) with the existing per-run state
  hoisted into ``runs["default"]``.  Workspace-level keys (UI state,
  picker config) stay at top-level.

* **Default run creation** — migration auto-creates
  ``runs/default/run_manifest.json`` with sensible defaults
  (``backend="cellpose"``, ``target=None``, ``display_name="Default"``).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import _paths as p


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

#: Sentinel filename for "user opted out of migration this session"
MIGRATION_SKIPPED_SENTINEL = ".step2_migration_skipped"

#: Backup directory prefix.  Timestamp appended at backup time.
BACKUP_DIR_PREFIX = ".pre_step2_restructure_"

#: Workspace-level v2 state keys that DON'T move into runs.default —
#: they stay at v3 top-level.
#:
#: Release-hardening 2026-09-07 (bucket-switch item #2): the six UI/picker
#: keys that used to live here — ``selected_view``, ``view_target``,
#: ``axial_plane``, ``curation_model_source_mode``,
#: ``imported_curation_model_path``, ``curation_pretrained_model_type`` —
#: are now PER-RUN.  Workspace-level meant one run's external model file and
#: one run's active view bled into every other run.  Reads stay backward
#: compatible: ``_paths._unwrap_v3_state`` merges the top-level keys FIRST
#: and lets the per-run entry win, so an old state file (values only at
#: top level) still loads unchanged.
WORKSPACE_LEVEL_KEYS_V2: frozenset = frozenset({
    "schema_version",          # special — bumped to 3, NOT copied verbatim
})


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------

@dataclass
class FileMove:
    """A single file or directory move operation in the migration plan."""

    src: Path
    dst: Path
    kind: str         # "dir" or "file"
    purpose: str      # human-readable description for the dialog


@dataclass
class MetadataSplit:
    """A joint metadata JSON that must be split into per-bucket files.

    Used for legacy ``round_metadata.json`` + ``selected_indices.json``
    which are joint per-round but per-bucket in the new layout.
    """

    src: Path
    dst_lateral: Path
    dst_axial: Path
    kind: str         # "round_metadata" or "selected_indices"
    round_idx: int
    purpose: str


@dataclass
class StateSchemaWrap:
    """Wrap legacy ``hitl_state.json`` (v2) into new ``step2_state.json`` (v3).

    Per-run state lifted into ``runs["default"]``; workspace-level keys
    stay at top-level.
    """

    src: Path
    dst: Path
    purpose: str = "Wrap hitl_state.json (v2) -> step2_state.json (v3)"


@dataclass
class RunManifestInit:
    """Auto-create a ``run_manifest.json`` for the migration-created run.

    Default values: backend="cellpose", target=None, display_name="Default".
    """

    dst: Path
    run_id: str
    manifest_data: Dict[str, Any]
    purpose: str = "Initialize runs/default/run_manifest.json"


@dataclass
class LegacyDelete:
    """A legacy directory that should be deleted post-migration.

    Used for ``artifacts/{models,inference,benchmarks}/`` and the
    working-copy ``training_data/round_NN/{spatial,axial}/`` dirs
    (which are regenerable from accepted_pairs).
    """

    path: Path
    purpose: str


@dataclass
class MigrationPlan:
    """Read-only description of every change a migration would make.

    Returned by :func:`plan_migration`.  Consumed by
    :func:`execute_migration` and the UI dialog (for enumeration).
    """

    workspace: Path
    moves: List[FileMove] = field(default_factory=list)
    splits: List[MetadataSplit] = field(default_factory=list)
    state_wrap: Optional[StateSchemaWrap] = None
    run_manifest_init: Optional[RunManifestInit] = None
    legacy_deletes: List[LegacyDelete] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    hard_errors: List[str] = field(default_factory=list)
    is_mixed_layout: bool = False

    @property
    def total_operations(self) -> int:
        return (
            len(self.moves)
            + len(self.splits)
            + (1 if self.state_wrap else 0)
            + (1 if self.run_manifest_init else 0)
            + len(self.legacy_deletes)
        )

    @property
    def is_empty(self) -> bool:
        return self.total_operations == 0

    @property
    def has_hard_errors(self) -> bool:
        return len(self.hard_errors) > 0


@dataclass
class ExecutionResult:
    """Outcome of :func:`execute_migration`."""

    success: bool
    completed_steps: List[str] = field(default_factory=list)
    failed_step: Optional[str] = None
    error: Optional[str] = None
    backup_dir: Optional[Path] = None
    auto_rolled_back: bool = False


# ----------------------------------------------------------------------------
# Schema wrap (pure function — also called directly by callers that
# don't want to run a full migration)
# ----------------------------------------------------------------------------

def wrap_state_schema_v2_to_v3(
    v2_state: Dict[str, Any],
    migration_timestamp: Optional[str] = None,
    *,
    active_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap a v2 ``hitl_state.json`` dict into a v3 ``step2_state.json`` dict.

Workspace-level UI/picker state stays at
    top-level; everything else moves into ``runs[active_run_id]``.

    Unknown keys default to per-run placement (robust against schema
    drift — if the v2 schema grows new per-run keys, they'll
    automatically end up in the right place).

    Args:
        v2_state: parsed contents of legacy ``hitl_state.json``.
        migration_timestamp: ISO timestamp to record as
            ``runs[active_run_id].created_at``.  Defaults to now (UTC).
        active_run_id: which per-run slot to populate.  Defaults to
            ``DEFAULT_RUN_ID`` ("default") — the migration use case.
            For multi-run saves (Phase 8a), pass the controller's
            ``self.active_run_id`` so the wrap writes to the correct
            slot.  Without this kwarg, saves on a non-default run
            would CLOBBER the default run's state (audit-found
            CRITICAL bug, fixed 2026-05-23).

    Returns:
        v3 dict ready to write to ``step2_state.json``.
    """
    if not isinstance(v2_state, dict):
        raise ValueError(f"v2_state must be dict, got {type(v2_state).__name__}")

    incoming_schema = v2_state.get("schema_version")
    if incoming_schema not in (None, 2):
        raise ValueError(
            f"Expected v2 schema (schema_version=2 or absent), "
            f"got schema_version={incoming_schema!r}"
        )

    if migration_timestamp is None:
        migration_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

    target_run_id = active_run_id or p.DEFAULT_RUN_ID

    v3: Dict[str, Any] = {
        "schema_version": 3,
        "active_run_id": target_run_id,
        "runs": {
            target_run_id: {
                "display_name": (
                    "Default" if target_run_id == p.DEFAULT_RUN_ID else target_run_id
                ),
                "target": None,
                "backend": "cellpose",
                "created_at": migration_timestamp,
                "created_from": None,
            }
        },
    }

    target_run = v3["runs"][target_run_id]

    # Pass 1: copy workspace-level keys at top-level (except schema_version
    # which we already set above to the new value).
    for key in WORKSPACE_LEVEL_KEYS_V2:
        if key == "schema_version":
            continue
        if key in v2_state:
            v3[key] = v2_state[key]

    # Pass 2: copy everything else into runs[target_run_id].  Unknown keys
    # default here, which is the safer placement.
    #
    # TODO(Phase 6): rewrite absolute paths in `model_checkpoint_paths` and
    # `sample_records.{img_uri,mask_uri}` from absolute → relative-to-
    # workspace.  Deferred to Phase 6 because:
    #   1. New layout's absolute paths still resolve correctly post-migration
    #      (files are moved to the new location, paths in state need to
    #      point at the new locations).
    #   2. Step 3's _persist_curation pattern (controller.py) shows how
    #      relativization should work; Phase 6 can mirror that.
    # Current behavior (copy as-is): if the workspace is later moved to a
    # different disk path, the absolute paths will break.  Acceptable for
    # now since the typical user doesn't move workspaces post-migration.
    for key, value in v2_state.items():
        if key in WORKSPACE_LEVEL_KEYS_V2:
            continue
        target_run[key] = value

    return v3


def build_default_run_manifest(
    workspace: p._WS,
    migration_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the default-run ``run_manifest.json`` for a migration.

    Returns a dict ready to write at
    ``{ws}/artifacts/step2/runs/default/run_manifest.json``.
    """
    if migration_timestamp is None:
        migration_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

    return {
        "schema_version": "2.0",
        "run_id": p.DEFAULT_RUN_ID,
        "display_name": "Default",
        "target": None,
        "backend": "cellpose",
        "created_at": migration_timestamp,
        "created_from": None,
        "description": "Default run auto-created by Step 2 layout migration.",
        "channels_snapshot": None,
        "current_round": {"lateral": 0, "axial": 0},
    }


# ----------------------------------------------------------------------------
# Plan builder
# ----------------------------------------------------------------------------

def plan_migration(workspace: p._WS) -> MigrationPlan:
    """Build the migration plan for a workspace.  READ-ONLY.

    Scans the legacy layout, enumerates every move + metadata split +
    state wrap + run-manifest init.  Validates against the layout rules
    (mis-sorted files, duplicate basenames, etc.).  Returns the plan
    with any hard errors collected — caller must check
    ``plan.has_hard_errors`` before calling :func:`execute_migration`.

    Returns an empty plan (``is_empty=True``) if the workspace is
    already on the new layout or has no legacy markers.
    """
    ws = Path(workspace)
    plan = MigrationPlan(workspace=ws)

    layout = p.detect_layout_version(ws)
    layout_name = {
        p.LAYOUT_VERSION_UNKNOWN: "UNKNOWN",
        p.LAYOUT_VERSION_LEGACY:  "LEGACY",
        p.LAYOUT_VERSION_NEW:     "NEW",
        p.LAYOUT_VERSION_MIXED:   "MIXED",
    }.get(layout, str(layout))
    print(f"[STEP2-MIGRATE] plan_migration: layout={layout_name} workspace={ws}")

    if layout == p.LAYOUT_VERSION_NEW:
        return plan  # already migrated

    if layout == p.LAYOUT_VERSION_UNKNOWN:
        return plan  # empty workspace

    if layout == p.LAYOUT_VERSION_MIXED:
        plan.is_mixed_layout = True
        plan.warnings.append(
            "Workspace has BOTH legacy and new layout markers — "
            "likely a previous migration was interrupted.  Review the "
            "plan carefully before executing."
        )

    # Step 1: training_data fan-out (per-round, per-legacy-bucket → per-VIEW-code)
    _plan_training_data_fan_out(plan, ws)

    # Step 2: models fan-out (per-round → per-VIEW-code)
    _plan_models_fan_out(plan, ws)

    # Step 3: predictions fan-out
    _plan_predictions_fan_out(plan, ws)

    # Step 4: metadata splits (round_metadata.json + selected_indices.json)
    _plan_metadata_splits(plan, ws)

    # Step 5: state JSON v2 → v3 wrap
    _plan_state_wrap(plan, ws)

    # Step 6: run_manifest.json init
    _plan_run_manifest_init(plan, ws)

    # Step 7: legacy deletes (artifacts/{models,inference,benchmarks}/ +
    # working-copy {spatial,axial}/ dirs)
    _plan_legacy_deletes(plan, ws)

    print(f"[STEP2-MIGRATE] plan_migration: built plan with "
          f"{len(plan.moves)} moves, {len(plan.splits)} splits, "
          f"state_wrap={'yes' if plan.state_wrap else 'no'}, "
          f"run_init={'yes' if plan.run_manifest_init else 'no'}, "
          f"{len(plan.legacy_deletes)} deletes, "
          f"{len(plan.warnings)} warnings, "
          f"{len(plan.hard_errors)} hard_errors")

    return plan


def _plan_training_data_fan_out(plan: MigrationPlan, ws: Path) -> None:
    """Plan moves for ``training_data/round_NN/accepted_pairs/{spatial,axial}/*``."""
    training_data_root = p.legacy_training_data_root(ws)
    if not training_data_root.is_dir():
        return

    seen_basenames_per_round_bucket: Dict[Tuple[int, str], set] = {}

    for round_dir in sorted(training_data_root.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
            continue
        try:
            round_idx = int(round_dir.name.split("_")[-1])
        except (ValueError, IndexError):
            plan.warnings.append(
                f"Skipping unrecognized legacy round dir {round_dir}"
            )
            continue

        for stage in ("accepted_pairs", "staged_pairs"):
            stage_root = round_dir / stage
            if not stage_root.is_dir():
                continue

            for legacy_bucket in ("spatial", "axial"):
                legacy_bucket_dir = stage_root / legacy_bucket
                if not legacy_bucket_dir.is_dir():
                    continue

                for file in sorted(legacy_bucket_dir.iterdir()):
                    if not file.is_file():
                        continue
                    _plan_one_pair_file(
                        plan, ws, round_idx, stage, legacy_bucket, file,
                        seen_basenames_per_round_bucket,
                    )


def _plan_one_pair_file(
    plan: MigrationPlan,
    ws: Path,
    round_idx: int,
    stage: str,
    legacy_bucket: str,
    file: Path,
    seen: Dict[Tuple[int, str], set],
) -> None:
    """Plan one training-pair file move, validating bucket placement."""
    stem = file.stem
    try:
        stack_id, view, slice_idx = p.parse_slice_label(stem)
    except ValueError as e:
        plan.warnings.append(
            f"Skipping unparseable filename {file.name} (in "
            f"round_{round_idx:02d}/{stage}/{legacy_bucket}/): {e}.  "
            f"File preserved in backup; not migrated to new layout."
        )
        return

    expected_bucket = p.bucket_for_view(view)

    # rule 3: mis-sorted files → HARD FAIL
    legacy_canonical_bucket = "lateral" if legacy_bucket == "spatial" else "axial"
    if expected_bucket != legacy_canonical_bucket:
        plan.hard_errors.append(
            f"Mis-sorted file: {file.name!r} has VIEW code {view!r} "
            f"(maps to {expected_bucket!r} bucket) but is in legacy "
            f"{legacy_bucket!r} directory ({stage}/{legacy_bucket}/).  "
            f"This is a corrupted legacy workspace.  Resolve manually "
            f"before migrating: move XY files into spatial/ and XZ/YZ "
            f"files into axial/, OR delete misplaced files."
        )
        return

    # rule 5: duplicate basename across buckets → HARD FAIL
    bucket_key = (round_idx, expected_bucket)
    if bucket_key not in seen:
        seen[bucket_key] = set()
    if file.name in seen[bucket_key]:
        plan.hard_errors.append(
            f"Duplicate basename: {file.name!r} appears in both buckets "
            f"for round_{round_idx:02d}/{stage}.  Resolve manually."
        )
        return
    seen[bucket_key].add(file.name)

    stage_segment = "accepted" if stage == "accepted_pairs" else "staged"
    new_round_dir = p.round_dir(ws, p.DEFAULT_RUN_ID, expected_bucket, round_idx)
    dst = new_round_dir / stage_segment / file.name

    plan.moves.append(FileMove(
        src=file,
        dst=dst,
        kind="file",
        purpose=f"round_{round_idx:02d} {stage_segment} pair → {expected_bucket} bucket",
    ))


def _plan_models_fan_out(plan: MigrationPlan, ws: Path) -> None:
    """Plan moves for ``models/round_NN/*`` files."""
    models_root = p.legacy_models_root(ws)
    if not models_root.is_dir():
        return

    for round_dir in sorted(models_root.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
            continue
        try:
            round_idx = int(round_dir.name.split("_")[-1])
        except (ValueError, IndexError):
            plan.warnings.append(
                f"Skipping unrecognized legacy model round dir {round_dir}"
            )
            continue

        for file in sorted(round_dir.iterdir()):
            if not file.is_file():
                continue
            _plan_one_model_file(plan, ws, round_idx, file)


def _plan_one_model_file(
    plan: MigrationPlan,
    ws: Path,
    round_idx: int,
    file: Path,
) -> None:
    """Plan one model file move.  Legacy filenames are typically
    ``{model_name}_{view_type}_{timestamp}.ckpt.json`` or similar.

    For migration we use a coarse heuristic: filename contains
    ``lateral``/``spatial`` → lateral bucket; contains ``axial`` →
    axial bucket; otherwise, log a warning and skip (user can manually
    move if needed).
    """
    name_lower = file.name.lower()
    bucket: Optional[str] = None
    if "axial" in name_lower:
        bucket = "axial"
    elif "spatial" in name_lower or "lateral" in name_lower:
        bucket = "lateral"

    if bucket is None:
        plan.warnings.append(
            f"Cannot determine bucket for legacy model file {file.name!r} "
            f"(round_{round_idx:02d}/models/).  No 'lateral', 'spatial', "
            f"or 'axial' substring in filename.  File preserved in backup "
            f"but not migrated; you may need to move it manually."
        )
        return

    new_model_dir = p.model_dir(ws, p.DEFAULT_RUN_ID, bucket, round_idx)
    dst = new_model_dir / file.name
    plan.moves.append(FileMove(
        src=file,
        dst=dst,
        kind="file",
        purpose=f"round_{round_idx:02d} model → {bucket}/models/",
    ))


def _plan_predictions_fan_out(plan: MigrationPlan, ws: Path) -> None:
    """Plan moves for ``predictions/round_NN/{spatial,axial}/*``."""
    predictions_root = p.legacy_predictions_root(ws)
    if not predictions_root.is_dir():
        return

    for round_dir in sorted(predictions_root.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
            continue
        try:
            round_idx = int(round_dir.name.split("_")[-1])
        except (ValueError, IndexError):
            continue

        for legacy_bucket in ("spatial", "axial"):
            legacy_bucket_dir = round_dir / legacy_bucket
            if not legacy_bucket_dir.is_dir():
                continue

            for file in sorted(legacy_bucket_dir.iterdir()):
                if not file.is_file():
                    continue
                _plan_one_prediction_file(
                    plan, ws, round_idx, legacy_bucket, file
                )


def _plan_one_prediction_file(
    plan: MigrationPlan,
    ws: Path,
    round_idx: int,
    legacy_bucket: str,
    file: Path,
) -> None:
    """Plan one prediction-file move.  Validates VIEW code matches bucket."""
    stem = file.stem
    # manifest.json doesn't have a slice-label stem; route by legacy bucket.
    if stem == "manifest":
        canonical_bucket = "lateral" if legacy_bucket == "spatial" else "axial"
        dst = p.round_predictions_manifest_path(
            ws, p.DEFAULT_RUN_ID, canonical_bucket, round_idx
        )
        plan.moves.append(FileMove(
            src=file, dst=dst, kind="file",
            purpose=f"round_{round_idx:02d} predictions manifest → {canonical_bucket}/predictions/",
        ))
        return

    try:
        _stack_id, view, _slice_idx = p.parse_slice_label(stem)
    except ValueError:
        # Pred-mask files have suffix '_pred_mask' which parse_slice_label
        # handles; if it still doesn't parse, log + route by legacy bucket.
        canonical_bucket = "lateral" if legacy_bucket == "spatial" else "axial"
        plan.warnings.append(
            f"Unparseable prediction filename {file.name!r}, routing by "
            f"legacy bucket to {canonical_bucket}."
        )
        dst = p.round_predictions_dir(
            ws, p.DEFAULT_RUN_ID, canonical_bucket, round_idx
        ) / file.name
        plan.moves.append(FileMove(
            src=file, dst=dst, kind="file",
            purpose=f"round_{round_idx:02d} prediction → {canonical_bucket}/predictions/",
        ))
        return

    expected_bucket = p.bucket_for_view(view)
    dst = p.round_predictions_dir(
        ws, p.DEFAULT_RUN_ID, expected_bucket, round_idx
    ) / file.name
    plan.moves.append(FileMove(
        src=file, dst=dst, kind="file",
        purpose=f"round_{round_idx:02d} prediction → {expected_bucket}/predictions/",
    ))


def _plan_metadata_splits(plan: MigrationPlan, ws: Path) -> None:
    """Plan per-bucket splits of ``round_metadata.json`` + ``selected_indices.json``."""
    training_data_root = p.legacy_training_data_root(ws)
    if not training_data_root.is_dir():
        return

    for round_dir in sorted(training_data_root.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
            continue
        try:
            round_idx = int(round_dir.name.split("_")[-1])
        except (ValueError, IndexError):
            continue

        meta = round_dir / "round_metadata.json"
        if meta.is_file():
            plan.splits.append(MetadataSplit(
                src=meta,
                dst_lateral=p.round_metadata_path(ws, p.DEFAULT_RUN_ID, "lateral", round_idx),
                dst_axial=p.round_metadata_path(ws, p.DEFAULT_RUN_ID, "axial", round_idx),
                kind="round_metadata",
                round_idx=round_idx,
                purpose=f"Split round_{round_idx:02d} metadata per bucket",
            ))

        sel = round_dir / "selected_indices.json"
        if sel.is_file():
            plan.splits.append(MetadataSplit(
                src=sel,
                dst_lateral=p.round_selected_indices_path(ws, p.DEFAULT_RUN_ID, "lateral", round_idx),
                dst_axial=p.round_selected_indices_path(ws, p.DEFAULT_RUN_ID, "axial", round_idx),
                kind="selected_indices",
                round_idx=round_idx,
                purpose=f"Split round_{round_idx:02d} selected_indices per bucket",
            ))


def _plan_state_wrap(plan: MigrationPlan, ws: Path) -> None:
    """Plan the v2 → v3 state JSON wrap."""
    legacy_state = p.legacy_state_json_path(ws)
    if not legacy_state.is_file():
        return  # no legacy state to wrap
    plan.state_wrap = StateSchemaWrap(
        src=legacy_state,
        dst=p.state_json_path(ws),
    )


def _plan_run_manifest_init(plan: MigrationPlan, ws: Path) -> None:
    """Plan the auto-creation of the default run's run_manifest.json."""
    plan.run_manifest_init = RunManifestInit(
        dst=p.run_manifest_path(ws, p.DEFAULT_RUN_ID),
        run_id=p.DEFAULT_RUN_ID,
        manifest_data=build_default_run_manifest(ws),
    )


def _plan_legacy_deletes(plan: MigrationPlan, ws: Path) -> None:
    """Plan deletes of subsumed legacy artifact dirs + working copies."""
    training_data_root = p.legacy_training_data_root(ws)
    if training_data_root.is_dir():
        for round_dir in sorted(training_data_root.iterdir()):
            if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
                continue

            # Working-copy {spatial,axial}/ dirs (regenerable, never moved)
            for legacy_bucket in ("spatial", "axial"):
                working = round_dir / legacy_bucket
                if working.is_dir():
                    plan.legacy_deletes.append(LegacyDelete(
                        path=working,
                        purpose=f"Working copy ({round_dir.name}/{legacy_bucket}/) — regenerable",
                    ))

            # Legacy joint metadata files: their content was split into
            # per-bucket files in the new layout (see MetadataSplit
            # operations); the original sources are now stale.
            for meta_name in ("round_metadata.json", "selected_indices.json"):
                meta_path = round_dir / meta_name
                if meta_path.is_file():
                    plan.legacy_deletes.append(LegacyDelete(
                        path=meta_path,
                        purpose=f"Stale joint {meta_name} (split into per-bucket files)",
                    ))

    # artifacts/{models,inference,benchmarks}/ — subsumed by per-bundle manifests
    for sub in ("models", "inference", "benchmarks"):
        legacy = ws / "artifacts" / sub
        if legacy.is_dir():
            plan.legacy_deletes.append(LegacyDelete(
                path=legacy,
                purpose=f"artifacts/{sub}/ — subsumed by per-bundle manifest.json",
            ))


# ----------------------------------------------------------------------------
# Backup + rollback
# ----------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Console log for the migration flow (visible to user in napari console)."""
    print(f"[STEP2-MIGRATE] {msg}")


def create_backup(workspace: p._WS, timestamp: Optional[str] = None) -> Path:
    """Create a backup of all legacy artifacts that the migration will touch.

    The backup is a sibling directory at workspace root named
    ``.pre_step2_restructure_<timestamp>/``.  Contents are deep-copies of:

    * ``training_data/``
    * ``models/``
    * ``predictions/``
    * ``artifacts/{models,inference,benchmarks}/``
    * ``hitl_state.json`` (legacy state file)
    * ``training_config.json`` (workspace-root config — preserved
      across migrations but backed up to protect against accidental
      modification during a failed migration)

    Returns the backup directory path.  Idempotent: re-creating the
    same backup is a no-op (skips already-present subtrees).

    NOT backed up: ``workspace.lock`` (advisory file, recreated as
    needed) and anything under ``artifacts/step2/`` (the new layout —
    if it exists at backup time, the workspace is mid-migration or
    already migrated and rollback would handle it).
    """
    ws = Path(workspace)
    if timestamp is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_dir = ws / f"{BACKUP_DIR_PREFIX}{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    _log(f"create_backup: copying legacy artifacts to {backup_dir}")

    # Top-level legacy dirs
    for legacy_subdir in ("training_data", "models", "predictions"):
        src = ws / legacy_subdir
        if src.is_dir():
            dst = backup_dir / legacy_subdir
            if not dst.exists():
                shutil.copytree(src, dst)

    # Legacy artifacts subdirs (only the ones we'll delete)
    for sub in ("models", "inference", "benchmarks"):
        src = ws / "artifacts" / sub
        if src.is_dir():
            dst = backup_dir / "artifacts" / sub
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst)

    # Legacy state JSON
    legacy_state = p.legacy_state_json_path(ws)
    if legacy_state.is_file():
        dst = backup_dir / p.LEGACY_STATE_FILENAME
        if not dst.exists():
            shutil.copy2(legacy_state, dst)

    # Workspace-root training_config.json — workspace-level config
    # that migration leaves in place, but back it up defensively
    # against accidental modification during a failed migration.
    training_config = ws / "training_config.json"
    if training_config.is_file():
        dst = backup_dir / "training_config.json"
        if not dst.exists():
            shutil.copy2(training_config, dst)

    _log(f"create_backup: complete at {backup_dir}")
    return backup_dir


def rollback_migration(backup_dir: Path, workspace: p._WS) -> None:
    """Restore a workspace from a migration backup.

    Removes any post-migration artifacts (the new
    ``artifacts/step2/`` subtree + ``step2_state.json``) and restores
    every backed-up subtree.  Idempotent: re-running on an already-
    rolled-back workspace is safe.
    """
    ws = Path(workspace)
    backup_dir = Path(backup_dir)
    _log(f"rollback_migration: restoring {ws} from backup {backup_dir}")

    if not backup_dir.is_dir():
        raise ValueError(f"Backup dir does not exist: {backup_dir}")

    # 1. Remove new-layout artifacts
    new_state = p.state_json_path(ws)
    if new_state.is_file():
        new_state.unlink()

    new_root = p.step2_root(ws)
    if new_root.is_dir():
        shutil.rmtree(new_root)

    # 2. Restore from backup
    for legacy_subdir in ("training_data", "models", "predictions"):
        backed_up = backup_dir / legacy_subdir
        if backed_up.is_dir():
            dst = ws / legacy_subdir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(backed_up, dst)

    for sub in ("models", "inference", "benchmarks"):
        backed_up = backup_dir / "artifacts" / sub
        if backed_up.is_dir():
            dst = ws / "artifacts" / sub
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(backed_up, dst)

    backed_state = backup_dir / p.LEGACY_STATE_FILENAME
    if backed_state.is_file():
        legacy_state = p.legacy_state_json_path(ws)
        shutil.copy2(backed_state, legacy_state)

    # Restore training_config.json from backup (workspace-level config)
    backed_training_config = backup_dir / "training_config.json"
    if backed_training_config.is_file():
        shutil.copy2(backed_training_config, ws / "training_config.json")


# ----------------------------------------------------------------------------
# Execute
# ----------------------------------------------------------------------------

def execute_migration(
    plan: MigrationPlan,
    backup_dir: Path,
    auto_rollback_on_failure: bool = True,
) -> ExecutionResult:
    """Apply a migration plan.  MUTATES the workspace.

    Caller must have created ``backup_dir`` via :func:`create_backup`
    first.  On any failure mid-execute, if ``auto_rollback_on_failure``
    is True (default), the workspace is restored from ``backup_dir``
    and the result reports the failure.
    """
    if plan.has_hard_errors:
        _log(f"execute_migration: REFUSING (plan has {len(plan.hard_errors)} hard errors)")
        return ExecutionResult(
            success=False,
            error=(
                "Plan has hard errors; refusing to execute.  "
                f"Errors: {plan.hard_errors}"
            ),
            backup_dir=backup_dir,
        )

    _log(f"execute_migration: starting ({plan.total_operations} operations)")
    result = ExecutionResult(success=True, backup_dir=backup_dir)

    # Track the current phase explicitly so failed_step reflects what
    # was ACTUALLY happening when the exception fired, not just the
    # last successful operation.  Without this, an error during step 2
    # (metadata splits) gets reported as "Failed at: <last move>" which
    # is misleading.
    current_phase = "(not started)"

    try:
        # Step 1: file moves
        current_phase = "step 1 — file moves"
        _log(f"execute_migration: step 1 — {len(plan.moves)} file moves")
        for move in plan.moves:
            move.dst.parent.mkdir(parents=True, exist_ok=True)
            if move.dst.exists():
                # Idempotent: dst already present means a prior partial
                # execute or copy.  Don't clobber.
                continue
            shutil.move(str(move.src), str(move.dst))
            result.completed_steps.append(f"move: {move.src} → {move.dst}")

        # Step 2: metadata splits
        if plan.splits:
            _log(f"execute_migration: step 2 — {len(plan.splits)} metadata splits")
        for split in plan.splits:
            current_phase = (
                f"step 2 — split: round_{split.round_idx:02d} {split.kind} "
                f"(src={split.src})"
            )
            _execute_metadata_split(split)
            result.completed_steps.append(
                f"split: round_{split.round_idx:02d} {split.kind}"
            )

        # Step 3: state JSON wrap
        if plan.state_wrap:
            current_phase = "step 3 — state JSON wrap (v2 -> v3)"
            _log("execute_migration: step 3 — state JSON wrap (v2 -> v3)")
            _execute_state_wrap(plan.state_wrap)
            result.completed_steps.append(
                f"wrap: {p.LEGACY_STATE_FILENAME} → {p.STEP2_STATE_FILENAME}"
            )

        # Step 4: run manifest init
        if plan.run_manifest_init:
            current_phase = "step 4 — init run_manifest.json for default run"
            _log("execute_migration: step 4 — init run_manifest.json for default run")
            _execute_run_manifest_init(plan.run_manifest_init)
            result.completed_steps.append(
                f"init: runs/{plan.run_manifest_init.run_id}/run_manifest.json"
            )

        # Step 5: legacy deletes (last — after everything else succeeded)
        if plan.legacy_deletes:
            current_phase = f"step 5 — legacy deletes"
            _log(f"execute_migration: step 5 — {len(plan.legacy_deletes)} legacy deletes")
        for delete in plan.legacy_deletes:
            if delete.path.is_dir():
                shutil.rmtree(delete.path)
            elif delete.path.is_file():
                delete.path.unlink()
            result.completed_steps.append(f"delete: {delete.path}")

        # Step 6: prune empty legacy parent dirs.  After all moves +
        # explicit deletes succeed, the legacy roots (training_data/,
        # models/, predictions/) typically contain only empty subdirs
        # (round_NN/ folders whose content has all moved or been
        # deleted).  Walk bottom-up and remove anything empty.  This
        # is what makes detect_layout_version() correctly transition
        # from MIXED → NEW after a successful migration.
        current_phase = "step 6 — prune empty legacy dirs"
        _log("execute_migration: step 6 — prune empty legacy dirs")
        ws = plan.workspace
        for legacy_root_name in ("training_data", "models", "predictions"):
            legacy_root = ws / legacy_root_name
            if legacy_root.is_dir():
                _prune_empty_dirs(legacy_root)
        result.completed_steps.append("prune empty legacy dirs")

        _log(f"execute_migration: SUCCESS ({len(result.completed_steps)} ops applied)")

    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {e}"
        # current_phase reflects the operation we were ATTEMPTING when
        # the exception fired — not the last successful step.  This is
        # what the dialog shows, so it must be accurate or users get
        # confused (the displayed "last successful move" leads them to
        # think the move itself failed).
        result.failed_step = current_phase
        _log(f"execute_migration: FAILED at '{result.failed_step}': {result.error}")
        if auto_rollback_on_failure:
            _log("execute_migration: attempting auto-rollback")
            try:
                rollback_migration(backup_dir, plan.workspace)
                result.auto_rolled_back = True
                _log("execute_migration: auto-rollback SUCCEEDED")
            except Exception as rollback_e:
                result.error = (
                    f"{result.error} (rollback ALSO FAILED: "
                    f"{type(rollback_e).__name__}: {rollback_e})"
                )
                _log(f"execute_migration: auto-rollback FAILED: {rollback_e}")

    return result


def _execute_metadata_split(split: MetadataSplit) -> None:
    """Split a joint metadata JSON into per-bucket files.

    The canonical schemas (per ``packages/epicycle_domain/contracts.py``):

    * ``round_metadata.json`` (``RoundMetadata`` dataclass):
        {"schema_version": str, "round_id": str, "created_at": str,
         "config_snapshot": dict, "parent_round": str|null,
         "samples": [SliceAnnotationRecord.to_dict(), ...]}

    * ``selected_indices.json`` (``SelectedIndicesRecord`` dataclass):
        {"schema_version": str, "round_id": str,
         "selections": {"XY": {stack_id: [idx]},
                        "XZ": {...},
                        "YZ": {...}}}

    Per-bucket split rules:
      * lateral bucket gets records with view=="XY" (round_metadata)
        or {"XY": ...} (selected_indices).
      * axial bucket gets records with view in ("XZ", "YZ").

    Defensive: also accepts legacy flat shapes (top-level list for
    round_metadata, top-level view-keyed dict for selected_indices)
    for backward compat with hand-written / older files.
    """
    with open(split.src, "r", encoding="utf-8") as f:
        data = json.load(f)

    if split.kind == "selected_indices":
        # Canonical: data["selections"] = {"XY": ..., "XZ": ..., "YZ": ...}
        # Legacy flat: data = {"XY": ..., "XZ": ..., "YZ": ...}
        if isinstance(data, dict) and "selections" in data and isinstance(data["selections"], dict):
            selections = data["selections"]
            # Preserve the envelope keys when writing per-bucket
            envelope = {k: v for k, v in data.items() if k != "selections"}
        elif isinstance(data, dict):
            # Legacy flat — treat data itself as the selections dict
            selections = data
            envelope = {}
        else:
            raise ValueError(
                f"Expected dict for selected_indices, got {type(data).__name__}"
            )

        lateral_selections = {"XY": selections.get("XY", {})} if "XY" in selections else {}
        axial_selections = {}
        if "XZ" in selections:
            axial_selections["XZ"] = selections["XZ"]
        if "YZ" in selections:
            axial_selections["YZ"] = selections["YZ"]

        lateral_data = {**envelope, "selections": lateral_selections} if lateral_selections else None
        axial_data = {**envelope, "selections": axial_selections} if axial_selections else None

    elif split.kind == "round_metadata":
        # Canonical: data["samples"] = [SliceAnnotationRecord, ...]
        # Legacy flat: data = [SliceAnnotationRecord, ...]
        if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], list):
            samples = data["samples"]
            envelope = {k: v for k, v in data.items() if k != "samples"}
        elif isinstance(data, list):
            # Legacy flat — treat data itself as the samples list
            samples = data
            envelope = {}
        else:
            raise ValueError(
                f"Expected dict (with 'samples' key) or list for round_metadata, "
                f"got {type(data).__name__}; src={split.src}"
            )

        lateral_samples = [
            r for r in samples
            if isinstance(r, dict) and r.get("view") == "XY"
        ]
        axial_samples = [
            r for r in samples
            if isinstance(r, dict) and r.get("view") in ("XZ", "YZ")
        ]

        lateral_data = {**envelope, "samples": lateral_samples} if lateral_samples else None
        axial_data = {**envelope, "samples": axial_samples} if axial_samples else None

    else:
        raise ValueError(f"Unknown split kind: {split.kind!r}")

    # Only write per-bucket files that have actual content
    if lateral_data:
        split.dst_lateral.parent.mkdir(parents=True, exist_ok=True)
        with open(split.dst_lateral, "w", encoding="utf-8") as f:
            json.dump(lateral_data, f, indent=2)
    if axial_data:
        split.dst_axial.parent.mkdir(parents=True, exist_ok=True)
        with open(split.dst_axial, "w", encoding="utf-8") as f:
            json.dump(axial_data, f, indent=2)


def _execute_state_wrap(wrap: StateSchemaWrap) -> None:
    """Read legacy state JSON, wrap v2→v3, write new state JSON.

    Idempotent: if ``wrap.dst`` already exists (prior partial execute
    or repeat-execute scenario) the call is a no-op.  Defensive
    against the v3-input-to-wrap crash that would otherwise leave the
    workspace in a partial state mid-execute.
    """
    if wrap.dst.exists():
        return  # already wrapped, skip
    with open(wrap.src, "r", encoding="utf-8") as f:
        v2 = json.load(f)
    # Belt-and-suspenders: if src is already v3 (e.g., user manually
    # renamed legacy state then re-ran migration), skip wrap.
    if isinstance(v2, dict) and v2.get("schema_version") == 3:
        return
    v3 = wrap_state_schema_v2_to_v3(v2)
    wrap.dst.parent.mkdir(parents=True, exist_ok=True)
    with open(wrap.dst, "w", encoding="utf-8") as f:
        json.dump(v3, f, indent=2)


def _prune_empty_dirs(root: Path) -> None:
    """Bottom-up walk: remove any directory that is empty (or becomes
    empty after pruning its children).  Files are never touched.

    Used as a post-execute cleanup pass to remove empty legacy roots
    (``training_data/``, ``models/``, ``predictions/``) once all their
    content has been moved or explicitly deleted by the migration.

    Safe for arbitrary trees: directories containing any files (e.g.,
    user-added notes) are preserved.
    """
    if not root.is_dir():
        return
    # Walk children first (bottom-up)
    for child in list(root.iterdir()):
        if child.is_dir():
            _prune_empty_dirs(child)
    # Then check if THIS dir is now empty
    try:
        next(root.iterdir())
        return  # has content (files or non-empty subdirs); keep
    except StopIteration:
        pass  # empty
    try:
        root.rmdir()
    except OSError:
        # Race / permission — leave it; non-fatal
        pass


def _execute_run_manifest_init(init: RunManifestInit) -> None:
    """Write the auto-generated run_manifest.json for the default run.

    Idempotent: if the manifest already exists, skip — don't clobber
    a manifest the user may have manually edited between migration
    runs (e.g., set a meaningful display_name or target).
    """
    if init.dst.exists():
        return
    init.dst.parent.mkdir(parents=True, exist_ok=True)
    with open(init.dst, "w", encoding="utf-8") as f:
        json.dump(init.manifest_data, f, indent=2)


# ----------------------------------------------------------------------------
# Sentinel (skip-and-don't-ask)
# ----------------------------------------------------------------------------

def is_migration_skipped(workspace: p._WS) -> bool:
    """True if the user has previously chosen to skip migration for this workspace."""
    return (Path(workspace) / MIGRATION_SKIPPED_SENTINEL).is_file()


def mark_migration_skipped(workspace: p._WS) -> None:
    """Write the skip-sentinel so the migration dialog won't re-prompt."""
    sentinel = Path(workspace) / MIGRATION_SKIPPED_SENTINEL
    sentinel.write_text(
        dt.datetime.now(dt.timezone.utc).isoformat() + "\n",
        encoding="utf-8",
    )


def clear_migration_skipped(workspace: p._WS) -> None:
    """Remove the skip-sentinel.  Used when user explicitly chooses to migrate after all."""
    sentinel = Path(workspace) / MIGRATION_SKIPPED_SENTINEL
    if sentinel.is_file():
        sentinel.unlink()


# ----------------------------------------------------------------------------
# Public API surface
# ----------------------------------------------------------------------------

__all__ = [
    # Constants
    "MIGRATION_SKIPPED_SENTINEL",
    "BACKUP_DIR_PREFIX",
    "WORKSPACE_LEVEL_KEYS_V2",
    # Data classes
    "FileMove",
    "MetadataSplit",
    "StateSchemaWrap",
    "RunManifestInit",
    "LegacyDelete",
    "MigrationPlan",
    "ExecutionResult",
    # Schema wrap
    "wrap_state_schema_v2_to_v3",
    "build_default_run_manifest",
    # Plan + execute
    "plan_migration",
    "create_backup",
    "execute_migration",
    "rollback_migration",
    # Sentinel
    "is_migration_skipped",
    "mark_migration_skipped",
    "clear_migration_skipped",
]

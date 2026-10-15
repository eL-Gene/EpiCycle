"""Step 2 split-layout → flat-layout migration (Phase C of B-series).

This module migrates an already-Phase-6c-migrated workspace from the
LEGACY SPLIT layout (`bucket/rounds/round_NN/` + `bucket/models/round_NN/`)
to the Phase B.2 FLAT layout (`bucket/round_NN/{accepted, predictions,
model}/`).  Mirrors the structure of `_migrate_step2_layout.py` (the
v2→v3 migration) so the two migrators are independent + composable.


Two-stage flow:

1. :func:`plan_split_to_flat_migration` — READ-ONLY scan.  Returns a
   :class:`SplitToFlatPlan` describing every move + manifest rewrite
   + state JSON path rewrite.  No filesystem mutation.

2. :func:`execute_split_to_flat_migration` (Phase C.2) — applies the
   plan.  Mutates the workspace.  Caller MUST have created a backup
   via :func:`create_backup` first.

Rollback (Phase C.2): :func:`rollback_split_to_flat_migration`
restores from the backup directory.

Pure-Python — no Qt / napari dependency.  All Qt-related interaction
(themed dialog) lives in a sibling ``_migrate_split_to_flat_dialog``
module (Phase C.3) so this one can be unit-tested without a display.

Key v3→v4 (split→flat) migration challenges (vs v2→v3):

* **Round-dir consolidation** — split layout has rounds and models in
  TWO sibling subtrees (`bucket/rounds/round_NN/` + `bucket/models/
  round_NN/`).  Flat layout merges them under ONE round dir
  (`bucket/round_NN/{accepted, predictions, model}/`).  Round-NN dirs
  in both subtrees → ONE round dir in flat with model/ inside.

* **Weights filename rename** — Phase B.2 changed `{run}_{bucket}` to
  `{run}_{bucket}_round_NN` so a model file carries its round in the
  name.  Migration renames the weights + 3 companion files (loss
  arrays, training_config).

* **Manifest path rewrites** — schema 2.0/2.1 manifest has 4 path
  fields (`weights_path`, `loss_train_path`, `loss_val_path`,
  `training_config_path`) plus `weights_filename`.  All must be
  updated to the new round-suffixed names.  manifest_history/
  archives must also be rewritten.

* **State JSON path rewrites** — closes the long-deferred
  `_migrate_step2_layout.py:282` TODO ("rewrite absolute paths in
  model_checkpoint_paths" — never done in v2→v3 migration).
  v3→v4 rewrites every absolute path that references the legacy
  split layout into the flat layout.

* **No schema bump** — the v3 state schema is unchanged.  The migration
  doesn't add new state keys; it just rewrites the path strings inside
  the existing keys.  Layout version is detected on-disk per
  `_paths.detect_layout_version`.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import _paths as p


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

#: Sentinel filename for "user opted out of split-to-flat migration"
SKIP_SENTINEL = ".step2_split_to_flat_migration_skipped"

#: Backup directory prefix.  Timestamp appended at backup time.
BACKUP_DIR_PREFIX = ".pre_step2_flat_layout_"

#: Filenames Cellpose writes alongside the weights file.  These must
#: be renamed alongside the weights file during split→flat migration
#: (the renamed weights file is `{run}_{bucket}_round_NN`; companions
#: get the same prefix).
CELLPOSE_COMPANION_SUFFIXES = (
    "_train_loss.npy",
    "_val_loss.npy",
    "_training_config.json",
)


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------

@dataclass
class RoundMove:
    """Plan for migrating ONE (run_id, bucket, round_NN) triple.

    Captures the source paths in LEGACY SPLIT layout + the destination
    paths in FLAT layout + the rename mapping for weights + companions
    + the manifest rewrite plan.
    """

    run_id: str
    bucket: str
    round_idx: int

    # Legacy SPLIT-layout source paths (may or may not exist; the
    # executor checks `is_dir()` before moving).
    src_round_dir: Path        # bucket/rounds/round_NN/  (has accepted/, predictions/, etc.)
    src_model_dir: Path        # bucket/models/round_NN/  (has weights, manifest, etc.)

    # FLAT-layout destination paths.
    dst_round_dir: Path        # bucket/round_NN/
    dst_model_dir: Path        # bucket/round_NN/model/

    # Weights filename rename.  Old: `{run}_{bucket}` (Phase 6c followup).
    # New: `{run}_{bucket}_round_NN` (Phase B.2 — round in name).
    # Empty strings if no weights file present in source.
    old_weights_filename: str = ""
    new_weights_filename: str = ""

    # Companion file renames (`{old}_train_loss.npy` → `{new}_train_loss.npy`
    # etc.).  Populated by the planner; one entry per CELLPOSE_COMPANION_SUFFIXES
    # that exists on disk.
    companion_renames: List[Tuple[str, str]] = field(default_factory=list)

    # Whether the source has a manifest.json that needs path rewriting.
    has_manifest: bool = False

    # Whether the source has a manifest_history/ dir whose archived
    # manifests also need path rewriting.
    has_manifest_history: bool = False

    # Whether the destination already exists (partial-migration case
    # — skip or warn).
    dst_round_dir_exists: bool = False
    dst_model_dir_exists: bool = False


@dataclass
class StateJsonPathRewrite:
    """Plan for rewriting absolute paths inside ``step2_state.json``.

    Closes the deferred `_migrate_step2_layout.py:282` TODO that was
    never addressed in the v2→v3 migration.  v3→v4 (split→flat) is
    where these paths actually need to change.
    """

    state_json_path: Path
    # Keys whose VALUES are path strings to be rewritten.  Per-run
    # nested keys are expressed with dot-paths like
    # `runs.{run_id}.model_checkpoint_paths.{round_key}`.
    keys_to_rewrite: List[str] = field(default_factory=list)
    # Per-run sample_records URI rewrites.  Each entry:
    # `runs.{run_id}.sample_records.{slice_label}.{img_uri|mask_uri}`.
    sample_record_uri_count: int = 0


@dataclass
class SplitToFlatPlan:
    """Read-only description of every change a split→flat migration
    would make to a workspace.
    """

    workspace: Path
    round_moves: List[RoundMove] = field(default_factory=list)
    state_rewrite: Optional[StateJsonPathRewrite] = None
    # Empty `rounds/` and `models/` parent dirs to clean up after all
    # round-moves complete.
    empty_dirs_to_remove: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    hard_errors: List[str] = field(default_factory=list)
    # Workspace is in a mixed state (some rounds flat, some split).
    # Migration completes by moving the remaining split rounds.
    is_mixed_layout: bool = False

    @property
    def total_operations(self) -> int:
        return (
            len(self.round_moves)
            + (1 if self.state_rewrite else 0)
            + len(self.empty_dirs_to_remove)
        )

    @property
    def is_empty(self) -> bool:
        return self.total_operations == 0

    @property
    def has_hard_errors(self) -> bool:
        return len(self.hard_errors) > 0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

_ROUND_DIR_RE = re.compile(r"^round_(\d+)$")


def _parse_round_dir_name(name: str) -> Optional[int]:
    """Parse `round_NN` directory name → round_idx int, or None."""
    m = _ROUND_DIR_RE.match(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _canonical_b2_weights_filename(run_id: str, bucket: str, round_idx: int) -> str:
    """The Phase B.2 canonical weights filename: `{run}_{bucket}_round_NN`.

    Mirror of the construction in
    `training_annotation_controller.py:5400` (the production writer).
    """
    return f"{run_id}_{bucket}_round_{round_idx:02d}"


# ----------------------------------------------------------------------------
# Plan builder
# ----------------------------------------------------------------------------

def plan_split_to_flat_migration(workspace: Path | str) -> SplitToFlatPlan:
    """Scan a workspace for LEGACY SPLIT layout artifacts that need
    migration to FLAT layout.

    READ-ONLY — does NOT mutate the workspace.  Safe to call repeatedly
    (idempotent).

    Args:
        workspace: workspace root.

    Returns:
        :class:`SplitToFlatPlan` describing every move + rewrite + cleanup
        that :func:`execute_split_to_flat_migration` (Phase C.2) would
        perform.

    The scan covers:

    * Every run under ``artifacts/step2/runs/``
    * For each run: both buckets (lateral, axial)
    * For each bucket: both legacy-split subtrees
      (`bucket/rounds/round_NN/` + `bucket/models/round_NN/`)
    * Union of round_NN dirs from BOTH subtrees → one `RoundMove` per
      unique round_NN

    Per-round move planning:
    * Detect weights filename in source `bucket/models/round_NN/`
      (read `manifest.json::weights_filename` if present; fall back to
      bare `{run}_{bucket}` heuristic)
    * Compute new filename `{run}_{bucket}_round_NN`
    * Enumerate companion files that exist on disk

    State JSON path rewrite planning:
    * Scan `step2_state.json` for absolute-path values that reference
      the legacy split layout.  Record per-key rewrites.

    Warnings (non-blocking):
    * Round_NN in `rounds/` but not `models/` (data without model)
    * Round_NN in `models/` but not `rounds/` (model without data)
    * Empty round_NN in one of the subtrees

    Hard errors (blocking):
    * Destination flat-layout round_NN already exists AND source still
      present → ambiguous which is canonical
    """
    ws = Path(workspace)
    plan = SplitToFlatPlan(workspace=ws)

    if not ws.is_dir():
        plan.hard_errors.append(f"Workspace does not exist: {ws}")
        return plan

    runs_root = p.step2_runs_root(ws)
    if not runs_root.is_dir():
        # No runs at all → nothing to migrate
        return plan

    # Discover per-run, per-bucket, per-round work
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        run_id = run_dir.name
        for bucket in ("lateral", "axial"):
            bucket_dir = run_dir / bucket
            if not bucket_dir.is_dir():
                continue

            legacy_rounds_root = bucket_dir / "rounds"
            legacy_models_root = bucket_dir / "models"

            # Collect unique round_idx values from BOTH legacy subtrees.
            round_idxs: Dict[int, Dict[str, bool]] = {}
            for subtree_root, key in (
                (legacy_rounds_root, "rounds"),
                (legacy_models_root, "models"),
            ):
                if not subtree_root.is_dir():
                    continue
                for sub in subtree_root.iterdir():
                    if not sub.is_dir():
                        continue
                    ridx = _parse_round_dir_name(sub.name)
                    if ridx is None:
                        continue
                    round_idxs.setdefault(ridx, {"rounds": False, "models": False})
                    round_idxs[ridx][key] = True

            # Also detect FLAT-layout round_NN dirs already present.
            # If a round_NN exists FLAT and SPLIT, that's a partial-
            # migration mixed state — flag in plan.
            flat_round_idxs: set = set()
            for sub in bucket_dir.iterdir():
                if not sub.is_dir():
                    continue
                if sub.name in ("rounds", "models"):
                    continue
                ridx = _parse_round_dir_name(sub.name)
                if ridx is not None:
                    flat_round_idxs.add(ridx)

            if flat_round_idxs and round_idxs:
                plan.is_mixed_layout = True

            for ridx, present in round_idxs.items():
                src_round_dir = legacy_rounds_root / f"round_{ridx:02d}"
                src_model_dir = legacy_models_root / f"round_{ridx:02d}"
                dst_round_dir = p.round_dir(ws, run_id, bucket, ridx)
                dst_model_dir = p.model_dir(ws, run_id, bucket, ridx)

                move = RoundMove(
                    run_id=run_id,
                    bucket=bucket,
                    round_idx=ridx,
                    src_round_dir=src_round_dir,
                    src_model_dir=src_model_dir,
                    dst_round_dir=dst_round_dir,
                    dst_model_dir=dst_model_dir,
                    dst_round_dir_exists=dst_round_dir.is_dir(),
                    dst_model_dir_exists=dst_model_dir.is_dir(),
                )

                # Warn if only one side present
                if present["rounds"] and not present["models"]:
                    plan.warnings.append(
                        f"{run_id}/{bucket}/rounds/round_{ridx:02d} has "
                        f"data but no sibling model dir — accepted pairs "
                        f"will move but no model will land."
                    )
                if present["models"] and not present["rounds"]:
                    plan.warnings.append(
                        f"{run_id}/{bucket}/models/round_{ridx:02d} has "
                        f"a model but no sibling rounds dir — model will "
                        f"move but no accepted/ data co-located."
                    )

                # Detect weights filename in source model dir.  Prefer
                # manifest.json::weights_filename; fall back to bare
                # `{run}_{bucket}` heuristic (the Phase 6c followup
                # canonical pre-B.2 name).
                if src_model_dir.is_dir():
                    old_wf = _detect_legacy_weights_filename(
                        src_model_dir, run_id, bucket
                    )
                    if old_wf:
                        move.old_weights_filename = old_wf
                        move.new_weights_filename = _canonical_b2_weights_filename(
                            run_id, bucket, ridx
                        )
                        # Enumerate companion files that exist
                        for suf in CELLPOSE_COMPANION_SUFFIXES:
                            old_comp = old_wf + suf
                            new_comp = move.new_weights_filename + suf
                            if (src_model_dir / old_comp).is_file():
                                move.companion_renames.append((old_comp, new_comp))
                    if (src_model_dir / "manifest.json").is_file():
                        move.has_manifest = True
                    if (src_model_dir / "manifest_history").is_dir():
                        move.has_manifest_history = True

                # Hard-error if destination flat round_NN exists AND
                # source still has data (ambiguous which is canonical).
                if move.dst_round_dir_exists and src_round_dir.is_dir():
                    plan.hard_errors.append(
                        f"{run_id}/{bucket}/round_{ridx:02d} exists in "
                        f"BOTH flat ({dst_round_dir}) AND legacy split "
                        f"({src_round_dir}).  Ambiguous canonical source — "
                        f"resolve manually before migrating: either move "
                        f"the legacy split content out, or delete the "
                        f"flat dir (data would be re-migrated)."
                    )
                if move.dst_model_dir_exists and src_model_dir.is_dir():
                    plan.hard_errors.append(
                        f"{run_id}/{bucket}/round_{ridx:02d}/model exists "
                        f"in both flat and legacy split.  Resolve manually."
                    )

                plan.round_moves.append(move)

            # Plan empty-dir cleanup for the post-migration step
            if legacy_rounds_root.is_dir():
                plan.empty_dirs_to_remove.append(legacy_rounds_root)
            if legacy_models_root.is_dir():
                plan.empty_dirs_to_remove.append(legacy_models_root)

    # State JSON path rewrite planning
    state_path = p.state_json_path(ws)
    if state_path.is_file() and plan.round_moves:
        # Plan a state-JSON rewrite ONLY if there's actual round-move
        # work — otherwise no paths to rewrite.
        rewrite = StateJsonPathRewrite(state_json_path=state_path)
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            _enumerate_state_rewrite_keys(state, rewrite, ws)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            plan.warnings.append(
                f"step2_state.json read failed during plan ({type(e).__name__}: "
                f"{e}); state-JSON paths will NOT be rewritten during "
                f"migration.  Manifest + on-disk paths still migrate."
            )
            rewrite = None
        plan.state_rewrite = rewrite

    return plan


def _detect_legacy_weights_filename(
    src_model_dir: Path, run_id: str, bucket: str
) -> Optional[str]:
    """Detect the weights filename in a legacy split-layout model dir.

    Priority chain (mirrors the dual-path readers added in Phase B.2):
    1. manifest.json::weights_filename (canonical source of truth)
    2. Bare `{run}_{bucket}` (Phase 6c followup default)
    3. Bare `weights` (pre-Phase-6c-followup legacy)
    4. None — caller treats as "no model in this round dir"
    """
    # Tier 1: manifest read
    mp = src_model_dir / "manifest.json"
    if mp.is_file():
        try:
            with open(mp, "r", encoding="utf-8") as f:
                m = json.load(f)
            wf = m.get("weights_filename")
            if wf and (src_model_dir / wf).is_file():
                return str(wf)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    # Tier 2: canonical Phase 6c filename
    cand = f"{run_id}_{bucket}"
    if (src_model_dir / cand).is_file():
        return cand
    # Tier 3: legacy bare `weights`
    if (src_model_dir / "weights").is_file():
        return "weights"
    return None


def _enumerate_state_rewrite_keys(
    state: Dict[str, Any], rewrite: StateJsonPathRewrite, workspace: Path
) -> None:
    """Walk a v3 state dict + record every path-bearing key whose value
    would change in the split→flat migration.

    For now this is a SHALLOW scan — it counts URI-bearing fields per
    run.  The actual rewrite logic (path translation) lives in Phase
    C.2's executor; this planner records WHAT will be rewritten so the
    dialog can show a count.
    """
    runs = state.get("runs", {})
    if not isinstance(runs, dict):
        return
    for run_id, run_state in runs.items():
        if not isinstance(run_state, dict):
            continue

        # model_checkpoint_paths: dict[round_key, path | {view: path}]
        mcp = run_state.get("model_checkpoint_paths", {})
        if isinstance(mcp, dict):
            for round_key, value in mcp.items():
                rewrite.keys_to_rewrite.append(
                    f"runs.{run_id}.model_checkpoint_paths.{round_key}"
                )

        # active_curation_checkpoints: dict[bucket, path]
        acc = run_state.get("active_curation_checkpoints", {})
        if isinstance(acc, dict):
            for bkt in acc.keys():
                rewrite.keys_to_rewrite.append(
                    f"runs.{run_id}.active_curation_checkpoints.{bkt}"
                )

        # imported_curation_model_path: str (single path)
        if run_state.get("imported_curation_model_path"):
            rewrite.keys_to_rewrite.append(
                f"runs.{run_id}.imported_curation_model_path"
            )

        # current_config.{pretrained_model_path, starting_checkpoint}
        cc = run_state.get("current_config", {})
        if isinstance(cc, dict):
            for key in ("pretrained_model_path", "starting_checkpoint"):
                if cc.get(key):
                    rewrite.keys_to_rewrite.append(
                        f"runs.{run_id}.current_config.{key}"
                    )

        # sample_records: dict[slice_label, {img_uri, mask_uri, ...}]
        sr = run_state.get("sample_records", {})
        if isinstance(sr, dict):
            for label, rec in sr.items():
                if not isinstance(rec, dict):
                    continue
                if rec.get("img_uri") or rec.get("mask_uri"):
                    rewrite.sample_record_uri_count += 1


@dataclass
class ExecutionResult:
    """Outcome of :func:`execute_split_to_flat_migration`."""

    success: bool
    completed_steps: List[str] = field(default_factory=list)
    failed_step: Optional[str] = None
    error: Optional[str] = None
    backup_dir: Optional[Path] = None
    auto_rolled_back: bool = False
    # Phase C.4 (2026-05-25, audit 3 LOW): non-fatal warnings surfaced
    # so the dialog can show them.  Pre-fix, the OSError on
    # `src_round_dir.rmdir()` was silently swallowed — could leave
    # stragglers if a stray file blocked cleanup without anyone knowing.
    warnings: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Skip-sentinel helpers (for "Don't Ask Again" option)
# ----------------------------------------------------------------------------

def is_migration_skipped(workspace: Path | str) -> bool:
    """Return True if the user previously opted out of split→flat migration
    for this workspace (sentinel file present)."""
    ws = Path(workspace)
    return (ws / SKIP_SENTINEL).is_file()


def mark_migration_skipped(workspace: Path | str) -> None:
    """Persist the "don't ask again" sentinel for this workspace.

    Subsequent workspace loads will skip the migration prompt.  User
    can re-enable by deleting the sentinel file manually.
    """
    ws = Path(workspace)
    (ws / SKIP_SENTINEL).touch(exist_ok=True)


# ----------------------------------------------------------------------------
# Backup + rollback (Phase C.2)
# ----------------------------------------------------------------------------

def create_backup(
    workspace: Path | str, timestamp: Optional[str] = None
) -> Path:
    """Create a backup of the v3 layout artifacts before migration.

    Backs up `artifacts/step2/runs/` (the entire current state — the
    migration mutates the legacy split subtrees in place) +
    `step2_state.json`.  Backup lives at
    `{ws}/.pre_step2_flat_layout_<timestamp>/`.

    Idempotent — if a backup with the same timestamp exists, skips
    already-backed-up subtrees.

    NOT backed up: `artifacts/step2/.trash/` (runs the user deleted).  A
    rollback wipes the whole `artifacts/step2/` subtree and restores only what
    is in the backup, so anything sitting in `.trash/` at rollback time is
    gone for good.  Empty the trash deliberately, not by rolling back.
    """
    ws = Path(workspace)
    if timestamp is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ws / f"{BACKUP_DIR_PREFIX}{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Back up the entire step2 subtree (covers both legacy + flat
    # round/model dirs so rollback can restore exactly whatever was
    # there pre-migration).
    src_step2 = p.step2_root(ws)
    if src_step2.is_dir():
        dst_step2 = backup_dir / "step2"
        if not dst_step2.exists():
            shutil.copytree(src_step2, dst_step2)

    # Back up state JSON
    state_path = p.state_json_path(ws)
    if state_path.is_file():
        dst_state = backup_dir / p.STEP2_STATE_FILENAME
        if not dst_state.exists():
            shutil.copy2(state_path, dst_state)

    return backup_dir


def rollback_split_to_flat_migration(
    backup_dir: Path | str, workspace: Path | str
) -> None:
    """Restore a workspace from a Phase C migration backup.

    Removes the (potentially partially-migrated) `artifacts/step2/`
    subtree + `step2_state.json` + restores the backed-up versions.
    Idempotent — re-running on an already-rolled-back workspace is safe.

    This DESTROYS `artifacts/step2/.trash/`: `create_backup` never copied it,
    and step 1 below removes the whole step2 root.  Deleted runs waiting in
    the trash do not survive a rollback.
    """
    ws = Path(workspace)
    backup_dir = Path(backup_dir)
    if not backup_dir.is_dir():
        raise ValueError(f"Backup dir does not exist: {backup_dir}")

    # 1. Remove current step2 subtree
    cur_step2 = p.step2_root(ws)
    if cur_step2.is_dir():
        shutil.rmtree(cur_step2)

    # 2. Remove current state JSON
    cur_state = p.state_json_path(ws)
    if cur_state.is_file():
        cur_state.unlink()

    # 3. Restore step2 subtree
    backup_step2 = backup_dir / "step2"
    if backup_step2.is_dir():
        cur_step2.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(backup_step2, cur_step2)

    # 4. Restore state JSON
    backup_state = backup_dir / p.STEP2_STATE_FILENAME
    if backup_state.is_file():
        shutil.copy2(backup_state, cur_state)


# ----------------------------------------------------------------------------
# Path translation (used for state JSON rewrite)
# ----------------------------------------------------------------------------

# Legacy split-layout absolute path patterns we recognize:
# 1. .../runs/{run_id}/{bucket}/models/round_NN/{weights_filename}
# 2. .../runs/{run_id}/{bucket}/models/round_NN/<companion-file>
# 3. .../runs/{run_id}/{bucket}/rounds/round_NN/accepted/<file>
# 4. .../runs/{run_id}/{bucket}/rounds/round_NN/predictions/<file>
# 5. .../runs/{run_id}/{bucket}/rounds/round_NN/<file>
# The translator rewrites segments + weights-filename atomically.

# Captures `.../runs/{run_id}/{bucket}/models/round_NN/{rest}`
_SPLIT_MODELS_PATH_RE = re.compile(
    r"^(?P<prefix>.+/runs/(?P<run_id>[^/]+)/(?P<bucket>lateral|axial))"
    r"/models/round_(?P<round>\d+)(?P<suffix>/.*)?$"
)

# Captures `.../runs/{run_id}/{bucket}/rounds/round_NN/{rest}`
_SPLIT_ROUNDS_PATH_RE = re.compile(
    r"^(?P<prefix>.+/runs/(?P<run_id>[^/]+)/(?P<bucket>lateral|axial))"
    r"/rounds/round_(?P<round>\d+)(?P<suffix>/.*)?$"
)


def translate_split_path_to_flat(
    path_str: str, round_moves_by_key: Optional[Dict[Tuple[str, str, int], "RoundMove"]] = None,
) -> Optional[str]:
    """Translate an absolute path from LEGACY SPLIT to FLAT layout.

    Returns the translated path, OR None if `path_str` doesn't match
    a known split-layout pattern (caller leaves it unchanged).

    Path patterns translated:
    - `runs/{run}/{bucket}/models/round_NN/{file}` →
        - If `{file}` is the OLD weights filename (`{run}_{bucket}`):
          → `runs/{run}/{bucket}/round_NN/model/{run}_{bucket}_round_NN`
        - If `{file}` is an OLD companion file (`{run}_{bucket}_{suffix}`):
          → `runs/{run}/{bucket}/round_NN/model/{run}_{bucket}_round_NN_{suffix}`
        - Else (manifest.json, manifest_history/, etc.):
          → `runs/{run}/{bucket}/round_NN/model/{file}`
    - `runs/{run}/{bucket}/rounds/round_NN/{file_or_subdir}` →
        `runs/{run}/{bucket}/round_NN/{file_or_subdir}`

    Args:
        path_str: absolute path string to translate.
        round_moves_by_key: optional map of (run_id, bucket, round_idx)
            → RoundMove for precise filename rewriting (uses the
            move's `old_weights_filename` for the heuristic).  When
            absent, falls back to `{run_id}_{bucket}` heuristic.

    Returns:
        Translated path string, OR None if no pattern matched.
    """
    norm = path_str.replace("\\", "/")

    # Try models/ pattern first (more specific than rounds/)
    m = _SPLIT_MODELS_PATH_RE.match(norm)
    if m:
        run_id = m.group("run_id")
        bucket = m.group("bucket")
        round_idx = int(m.group("round"))
        suffix = (m.group("suffix") or "").lstrip("/")
        new_prefix = f"{m.group('prefix')}/round_{round_idx:02d}/model"

        # Look up the move (if provided) for the exact old/new filename
        move = None
        if round_moves_by_key is not None:
            move = round_moves_by_key.get((run_id, bucket, round_idx))

        if not suffix:
            # Bare path .../models/round_NN — translate to .../round_NN/model
            return new_prefix

        # Translate the filename portion if it's the weights file or companion
        old_wf = (move.old_weights_filename if move and move.old_weights_filename
                  else f"{run_id}_{bucket}")
        new_wf = (move.new_weights_filename if move and move.new_weights_filename
                  else _canonical_b2_weights_filename(run_id, bucket, round_idx))

        # Try matching against old_wf + each companion suffix first
        if suffix == old_wf:
            return f"{new_prefix}/{new_wf}"
        for comp_suffix in CELLPOSE_COMPANION_SUFFIXES:
            if suffix == old_wf + comp_suffix:
                return f"{new_prefix}/{new_wf}{comp_suffix}"

        # Other files (manifest.json, manifest_history/...) — keep filename as-is
        return f"{new_prefix}/{suffix}"

    m = _SPLIT_ROUNDS_PATH_RE.match(norm)
    if m:
        round_idx = int(m.group("round"))
        suffix = (m.group("suffix") or "").lstrip("/")
        new_prefix = f"{m.group('prefix')}/round_{round_idx:02d}"
        if not suffix:
            return new_prefix
        return f"{new_prefix}/{suffix}"

    return None


# ----------------------------------------------------------------------------
# Manifest rewrite helpers
# ----------------------------------------------------------------------------

def rewrite_manifest_in_place(
    manifest_path: Path, old_weights_filename: str, new_weights_filename: str,
) -> bool:
    """Rewrite a schema 2.0/2.1 model manifest's path fields.

    Updates `weights_filename`, `weights_path`, `loss_train_path`,
    `loss_val_path`, `training_config_path` from old → new naming.
    All path fields are relative to the bundle root, so rewriting is
    a pure string substitution.

    Returns True if the manifest was updated, False if the rewrite was
    a no-op (already-rewritten manifest, or `old_weights_filename`
    doesn't match anything inside).  Phase C.4 hardening: raises
    `ValueError` on corrupt JSON or non-dict payload instead of
    silently returning False — the executor's auto-rollback then
    fires.  Pre-C.4, a corrupt manifest mid-migration could leave
    renamed files with a stale (un-rewritten) manifest pointing at
    non-existent paths.

    Idempotent: re-running on an already-rewritten manifest is safe
    (old_weights_filename won't match → returns False, no exception).
    """
    if not manifest_path.is_file():
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, ValueError) as _e:
        # Phase C.4 audit 1 MEDIUM fix: raise (was silent False).
        raise ValueError(
            f"Manifest {manifest_path} is corrupt JSON: {_e}"
        ) from _e
    except OSError as _ioe:
        raise OSError(
            f"Manifest {manifest_path} read failed: {_ioe}"
        ) from _ioe

    if not isinstance(manifest, dict):
        raise ValueError(
            f"Manifest {manifest_path} is not a JSON object "
            f"(got {type(manifest).__name__})"
        )

    changed = False
    if manifest.get("weights_filename") == old_weights_filename:
        manifest["weights_filename"] = new_weights_filename
        changed = True
    if manifest.get("weights_path") == old_weights_filename:
        manifest["weights_path"] = new_weights_filename
        changed = True
    for key, suffix in (
        ("loss_train_path", "_train_loss.npy"),
        ("loss_val_path", "_val_loss.npy"),
        ("training_config_path", "_training_config.json"),
    ):
        old_value = old_weights_filename + suffix
        new_value = new_weights_filename + suffix
        if manifest.get(key) == old_value:
            manifest[key] = new_value
            changed = True

    if not changed:
        return False

    # Atomic write via temp
    tmp = manifest_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(manifest_path)
    return True


# ----------------------------------------------------------------------------
# State JSON rewrite helper
# ----------------------------------------------------------------------------

def rewrite_state_json_paths(
    state_json_path: Path, plan: "SplitToFlatPlan",
) -> int:
    """Walk `step2_state.json` and rewrite every absolute path that
    references the legacy split layout to the corresponding flat-
    layout path.

    Returns the count of path strings rewritten.

    Idempotent: re-running on an already-rewritten state JSON is safe
    (paths that don't match the split-layout pattern are left
    unchanged).
    """
    if not state_json_path.is_file():
        return 0
    try:
        with open(state_json_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return 0

    # Build round_moves_by_key for precise filename translation
    moves_by_key: Dict[Tuple[str, str, int], "RoundMove"] = {
        (m.run_id, m.bucket, m.round_idx): m for m in plan.round_moves
    }

    n_rewritten = 0

    def _maybe_translate(value: Any) -> Tuple[Any, bool]:
        if not isinstance(value, str) or not value:
            return value, False
        translated = translate_split_path_to_flat(value, moves_by_key)
        if translated is not None and translated != value:
            return translated, True
        return value, False

    runs = state.get("runs", {})
    if not isinstance(runs, dict):
        return 0

    for run_id, run_state in runs.items():
        if not isinstance(run_state, dict):
            continue

        # model_checkpoint_paths: dict[round_key, path | {view: path}]
        mcp = run_state.get("model_checkpoint_paths", {})
        if isinstance(mcp, dict):
            for round_key, value in list(mcp.items()):
                if isinstance(value, str):
                    new_val, changed = _maybe_translate(value)
                    if changed:
                        mcp[round_key] = new_val
                        n_rewritten += 1
                elif isinstance(value, dict):
                    for view_key, sub_val in list(value.items()):
                        new_sub, changed = _maybe_translate(sub_val)
                        if changed:
                            value[view_key] = new_sub
                            n_rewritten += 1

        # active_curation_checkpoints: dict[bucket, path]
        acc = run_state.get("active_curation_checkpoints", {})
        if isinstance(acc, dict):
            for bkt, value in list(acc.items()):
                new_val, changed = _maybe_translate(value)
                if changed:
                    acc[bkt] = new_val
                    n_rewritten += 1

        # imported_curation_model_path: str (single path)
        for key in ("imported_curation_model_path",):
            new_val, changed = _maybe_translate(run_state.get(key))
            if changed:
                run_state[key] = new_val
                n_rewritten += 1

        # current_config.{pretrained_model_path, starting_checkpoint}
        cc = run_state.get("current_config", {})
        if isinstance(cc, dict):
            for key in ("pretrained_model_path", "starting_checkpoint"):
                new_val, changed = _maybe_translate(cc.get(key))
                if changed:
                    cc[key] = new_val
                    n_rewritten += 1

        # sample_records: dict[slice_label, {img_uri, mask_uri, ...}]
        sr = run_state.get("sample_records", {})
        if isinstance(sr, dict):
            for label, rec in sr.items():
                if not isinstance(rec, dict):
                    continue
                for uri_key in ("img_uri", "mask_uri"):
                    new_val, changed = _maybe_translate(rec.get(uri_key))
                    if changed:
                        rec[uri_key] = new_val
                        n_rewritten += 1

    if n_rewritten == 0:
        return 0

    # Atomic write
    tmp = state_json_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_json_path)
    return n_rewritten


# ----------------------------------------------------------------------------
# Executor (Phase C.2)
# ----------------------------------------------------------------------------

def execute_split_to_flat_migration(
    plan: SplitToFlatPlan,
    backup_dir: Path | str,
    auto_rollback_on_failure: bool = True,
) -> ExecutionResult:
    """Apply a split→flat migration plan.

    Caller MUST have created a backup via :func:`create_backup` first.

    Steps:
    1. For each `RoundMove`: move rounds/round_NN/* to round_NN/ (data
       side), then move models/round_NN/* to round_NN/model/ (model
       side), then rename weights + companions to B.2 naming, then
       rewrite manifest + manifest_history paths.
    2. Rewrite `step2_state.json` absolute path strings.
    3. Clean up empty `bucket/rounds/` and `bucket/models/` dirs.

    On any failure:
    - If `auto_rollback_on_failure=True`, calls
      :func:`rollback_split_to_flat_migration` to restore the backup.
    - Returns :class:`ExecutionResult` with `success=False` +
      `failed_step` + `error` populated.

    Idempotent on re-run: per-round moves skip if destination already
    has the content; manifest rewrites skip if old_weights_filename
    doesn't match anything; state-JSON rewrites only update strings
    that still match the split-layout pattern.
    """
    workspace = plan.workspace
    backup_dir = Path(backup_dir)
    result = ExecutionResult(success=False, backup_dir=backup_dir)

    if plan.has_hard_errors:
        result.error = (
            "Plan has hard errors; refuse to execute.  Resolve manually:\n  "
            + "\n  ".join(plan.hard_errors)
        )
        result.failed_step = "plan_validation"
        return result

    try:
        # Step 1: per-round moves
        for move in plan.round_moves:
            step_id = (
                f"round_move_{move.run_id}_{move.bucket}_round_{move.round_idx:02d}"
            )
            _execute_one_round_move(move, warnings=result.warnings)
            result.completed_steps.append(step_id)

        # Step 2: state JSON path rewrite
        if plan.state_rewrite is not None:
            n_rewritten = rewrite_state_json_paths(
                plan.state_rewrite.state_json_path, plan
            )
            result.completed_steps.append(
                f"state_json_rewrite (n={n_rewritten})"
            )

        # Step 3: cleanup empty legacy parent dirs
        for legacy_dir in plan.empty_dirs_to_remove:
            if legacy_dir.is_dir():
                try:
                    # Only remove if truly empty (defensive — a stray
                    # untracked file should NOT be silently deleted).
                    any_content = any(legacy_dir.iterdir())
                    if not any_content:
                        legacy_dir.rmdir()
                        result.completed_steps.append(
                            f"cleanup_empty_dir: {legacy_dir.name}"
                        )
                except OSError:
                    pass  # non-fatal

        result.success = True
        return result

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.failed_step = (
            result.completed_steps[-1] if result.completed_steps
            else "before_any_step"
        )
        if auto_rollback_on_failure and backup_dir.is_dir():
            try:
                rollback_split_to_flat_migration(backup_dir, workspace)
                result.auto_rolled_back = True
            except Exception as rb_err:
                result.error = (
                    f"{result.error}; rollback also failed: "
                    f"{type(rb_err).__name__}: {rb_err}"
                )
        return result


def _execute_one_round_move(
    move: RoundMove, *, warnings: Optional[List[str]] = None,
) -> None:
    """Execute a single RoundMove: move rounds/ + models/ data into
    the flat round_NN/ + round_NN/model/ structure + rename + rewrite.

    Internal helper for `execute_split_to_flat_migration`.  Raises on
    any failure; caller's rollback handles the partial-state.

    Phase C.4 (audit 3 LOW): `warnings` list accepts non-fatal
    notifications (e.g. stale-rmdir-target file blocking cleanup).
    """
    if warnings is None:
        warnings = []
    label = f"{move.run_id}/{move.bucket}/round_{move.round_idx:02d}"

    # 1. Move legacy rounds/round_NN/* to flat round_NN/
    if move.src_round_dir.is_dir():
        move.dst_round_dir.mkdir(parents=True, exist_ok=True)
        # Move contents one by one (allows merge if dst partially exists)
        for child in list(move.src_round_dir.iterdir()):
            target = move.dst_round_dir / child.name
            if target.exists():
                # Skip — partial-migration recovery (idempotent)
                continue
            shutil.move(str(child), str(target))
        # Remove the now-empty source dir; collect warning if blocked
        try:
            move.src_round_dir.rmdir()
        except OSError as _e:
            warnings.append(
                f"{label}: could not rmdir source rounds dir "
                f"({move.src_round_dir.name}) — likely a stray file "
                f"left behind: {_e}"
            )

    # 2. Move legacy models/round_NN/* to flat round_NN/model/
    if move.src_model_dir.is_dir():
        move.dst_model_dir.mkdir(parents=True, exist_ok=True)
        # Rename weights + companions first (in source — pre-move)
        if move.old_weights_filename and move.new_weights_filename and \
                move.old_weights_filename != move.new_weights_filename:
            old_w = move.src_model_dir / move.old_weights_filename
            new_w = move.src_model_dir / move.new_weights_filename
            if old_w.is_file() and not new_w.exists():
                old_w.rename(new_w)
            for old_comp, new_comp in move.companion_renames:
                old_c = move.src_model_dir / old_comp
                new_c = move.src_model_dir / new_comp
                if old_c.is_file() and not new_c.exists():
                    old_c.rename(new_c)

        # Rewrite the LIVE manifest.json (only — manifest_history
        # archives are intentionally LEFT AT THEIR PRE-MIGRATION
        # FILENAMES per Phase C.4 audit 3 design decision).  Each
        # archived manifest captures the live manifest at the time
        # of archive creation — a historical snapshot.  Retro-
        # rewriting them would falsify history.  Readers don't
        # consume archive filenames for resolution today; the live
        # manifest is the source of truth.
        if move.has_manifest and move.old_weights_filename and move.new_weights_filename:
            mp = move.src_model_dir / "manifest.json"
            rewrite_manifest_in_place(
                mp, move.old_weights_filename, move.new_weights_filename,
            )

        # Move all (renamed) contents to dst
        for child in list(move.src_model_dir.iterdir()):
            target = move.dst_model_dir / child.name
            if target.exists():
                continue
            shutil.move(str(child), str(target))
        try:
            move.src_model_dir.rmdir()
        except OSError as _e:
            warnings.append(
                f"{label}: could not rmdir source models dir "
                f"({move.src_model_dir.name}) — likely a stray file "
                f"left behind: {_e}"
            )


__all__ = [
    "SKIP_SENTINEL",
    "BACKUP_DIR_PREFIX",
    "CELLPOSE_COMPANION_SUFFIXES",
    "RoundMove",
    "StateJsonPathRewrite",
    "SplitToFlatPlan",
    "ExecutionResult",
    "plan_split_to_flat_migration",
    "create_backup",
    "rollback_split_to_flat_migration",
    "translate_split_path_to_flat",
    "rewrite_manifest_in_place",
    "rewrite_state_json_paths",
    "execute_split_to_flat_migration",
    "is_migration_skipped",
    "mark_migration_skipped",
]

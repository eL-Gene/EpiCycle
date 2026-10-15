"""Step 3 multi-run state + layout migration (single-context → ``default`` run).

Phase 0 of the Step 3 multi-run foundations.  This module
mirrors the structure of the Step 2 migration
(``hitl_training_annotation/_migrate_step2_layout.py``) but for Step 3's
much simpler single-context → ``default`` run reshape.

Two responsibilities:

1. **State envelope wrap / unwrap** (pure functions, no filesystem):

   * :func:`wrap_step3_state_to_v3` — wrap a flat state dict
     (``schema_version`` absent / 1 / 2) into the **schema v3**
     ``runs``-nested envelope, mirroring Step 2's
     ``wrap_state_schema_v2_to_v3``.  Workspace-level keys stay
     top-level; everything else (incl. ``patches`` + the model
     selection) moves under ``runs[active_run_id]``.

   * :func:`unwrap_step3_state_v3` — collapse a v3 envelope back into the
     flat dict the ``ReconstructionStateManager`` consumes, mirroring
     Step 2's ``_unwrap_v3_state``.

   * :func:`merge_forward_runs` — the no-clobber merge that copies every
     OTHER run's entry (and the active run's manifest fields) forward
     from the existing on-disk envelope, mirroring Step 2's controller
     save path.

   These are exact inverses (modulo the schema-version bump) so the
   state manager can keep its existing flat accessors untouched.

   **Schema number = 3 (NOT 2).**  ``schema_version 2`` is already
   claimed by ``_migrate_step3_layout`` for the *flat* layout-v2 shape
   (``{schema_version:2, layout_version:2, patches:{…top-level…}}``).
   The runs-envelope therefore mirrors Step 2's v2→v3 bump to avoid the
   collision; the wrap accepts BOTH the original flat-v1 and the
   layout-migrated flat-v2 as inputs.

2. **Physical on-disk migration** (plan / execute / rollback) of the
   ``artifacts/step3/runs/{patch}/`` → ``artifacts/step3/runs/{default}/
   reconstructions/{patch}/`` reshape, plus a state-JSON content rewrite
   that drops the per-patch ``reconstruction_run_id`` field (mirrors the
   existing ``_migrate_step3_layout.migrate_state_json_content``).

ACTIVATION STATUS (updated Phase 1a, 2026-06-27):

* The **state envelope (wrap / unwrap / merge_forward_runs) is LIVE** — as
  of Phase 1a the ``ReconstructionStateManager`` imports and calls these on
  every load/save (``_SCHEMA_VERSION = 3``).  Consequence (accepted): the first save flips ``step3_state.json`` to the
  v3 envelope, which an **un-merged baseline build reads as empty** (the
  cross-build trap).  The forward upgrade (v1/flat-v2 → v3) is lossless; the
  loss path is *downgrade-then-save* on an older build.  Safe once the full
  feature merges to baseline.  Until then: do NOT open a real workspace with
  a baseline build (or a baseline dev-PC pull loop) after this branch has
  touched it — test on a throwaway/copy workspace.

* The **physical on-disk migration (plan/execute/rollback) is still DORMANT**
  — no production caller.  ⚠️ Under the **Option A** runtime decision
  (default run = FLAT ``runs/{patch}``; only NAMED runs nest under
  ``runs/{slug}/reconstructions/``) the default run needs NO physical move —
  the flat layout *is* the default run.  This module's default-run target
  (``runs/{default}/reconstructions/{patch}``) therefore represents the
  OPTIONAL **symmetric-layout (Option B)** conversion only.  It is
  **incompatible with the Option-A resolver** (``_active_run_path_id`` maps
  default → ``None`` → flat): wiring it for the default run without ALSO
  switching the resolver to map default → the ``"default"`` slug would move
  data to ``runs/default/reconstructions/`` while the live controller keeps
  reading flat ``runs/`` → "reconstructed but not found".  Do NOT auto-wire
  this for the default run in Phase 2; it is a Phase-6 opt-in only.

Pure-Python — no Qt / napari dependency.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import _paths

# Discriminator between a NAMED-run context dir and a default-run per-patch
# reconstruction dir under ``runs/`` — both live at the same level.
# A named-run context dir ALWAYS carries a
# ``run_manifest.json``; a default-run patch dir NEVER does.  Imported from
# the (pure-Python, napari-free) packages layer.
try:
    from packages.epicycle_io.step3_run_manifest import is_run_context_dir
except Exception:  # pragma: no cover - defensive import guard
    is_run_context_dir = None  # type: ignore[assignment]


def _is_named_run_context(child: Path) -> bool:
    """True iff ``child`` is a NAMED-run context dir (must NOT be moved).

    Prefers the canonical :func:`is_run_context_dir` helper; falls back to a
    direct ``run_manifest.json`` presence check (belt-and-suspenders for the
    import-unavailable edge case).
    """
    try:
        if is_run_context_dir is not None:
            return bool(is_run_context_dir(child))
    except Exception:
        pass
    return (child / "run_manifest.json").exists()


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

#: The run slug a single-context (pre-multi-run) workspace migrates into.
DEFAULT_RUN_ID = "default"

#: Sentinel filename for "user opted out of the runs migration this session".
MIGRATION_SKIPPED_SENTINEL = ".step3_runs_migration_skipped"

#: Backup directory prefix.  Timestamp appended at backup time.
BACKUP_DIR_PREFIX = ".pre_step3_runs_"

#: Workspace-level flat state keys that DON'T move into ``runs[active]``
#: — they stay at the v3 envelope top-level..
#: ``schema_version`` + ``active_run_id`` are handled specially (set on
#: the envelope directly), the rest are copied verbatim if present.
#: ``layout_version`` is here so the layout-migrated flat-v2 shape's
#: top-level ``layout_version`` stays top-level (audit flagged it getting
#: buried under ``runs[default]``).
WORKSPACE_LEVEL_KEYS_V1: frozenset = frozenset({
    "schema_version",                 # special — bumped to 3, NOT copied
    "active_run_id",                  # special — set on the envelope
    "active_patch",                   # workspace-level (which patch is loaded)
    "loaded_stack_dir",               # workspace-level (stack source dir)
    "downsample_mode",                # workspace-level toggle
    "refinement_extraction_config",   # workspace-level extraction params
    "last_updated",                   # workspace-level save timestamp
    "layout_version",                 # workspace-level layout marker (flat-v2)
    "default_run_nested",             # workspace-level unified-layout flag
    # R7a: one voxel size per patch
    # for every run -- {patch: {"zyx": [z, y, x], "source": ...}}
    "patch_spacing",
})

#: The four per-run sibling sub-trees that the OPTIONAL unified-layout
#: (Option B) conversion moves WHOLE from the flat step-3 root into the
#: default run's context dir.  ``(label, getter)`` where ``getter(ws,
#: run_id)`` returns the dir for that run-id (``None`` = flat source,
#: ``"default"`` = nested dest).  Workspace-SHARED trees (``snapshots/``,
#: ``training_archive/``, ``batch_refine_recipes.json``) are intentionally
#: absent — they stay at the step-3 root by design.
#: ``refine_chain_recipes.json`` (Refine recipes) is one of them.
_SIBLING_DIR_GETTERS = (
    ("qc", _paths.qc_dir),
    ("curation", _paths.curation_root),
    ("prediction_cache", _paths.prediction_cache_dir),
    ("refinement", _paths.refinement_root),
)

#: Run-manifest-only keys that live under ``runs[id]`` but must NOT leak
#: back into the flat view on unwrap (they describe the run, not the
#: per-patch state the controller consumes).
RUN_MANIFEST_ONLY_KEYS: frozenset = frozenset({
    "display_name",
    "target",
    "created_from",
})


# ----------------------------------------------------------------------------
# State envelope wrap / unwrap (pure)
# ----------------------------------------------------------------------------

def wrap_step3_state_to_v3(
    flat_state: Dict[str, Any],
    *,
    active_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap a flat ``step3_state.json`` dict into the v3 ``runs`` envelope.

    Mirrors Step 2's ``wrap_state_schema_v2_to_v3``.  Workspace-level
    keys (see :data:`WORKSPACE_LEVEL_KEYS_V1`) stay top-level; everything
    else — including the entire ``patches`` map and the model selection
    fields — moves under ``runs[target_run_id]``.

    Accepts BOTH flat input shapes:

    * the original flat-v1 (``schema_version`` absent or ``1``); and
    * the layout-migrated flat-v2
      (``{schema_version:2, layout_version:2, patches:{…top-level…}}``).

    In both cases ``layout_version`` (if present) is preserved at the
    envelope top-level, not buried under ``runs[default]``.

    Args:
        flat_state: parsed contents of a flat (single-context) state dict.
        active_run_id: which per-run slot to populate.  Defaults to
            :data:`DEFAULT_RUN_ID`.  **Never hardcode "default" in the
            save path** — pass the manager's ``active_run_id`` so a save
            on a non-default run doesn't CLOBBER the default run (the
            Step 2 Phase 8a audit-found bug).

    Returns:
        v3 dict ready to write to ``step3_state.json``.

    Raises:
        ValueError: if ``flat_state`` is not a dict, or its
            ``schema_version`` is anything other than absent, ``1``, or
            ``2``.  ``schema_version >= 3`` is rejected (it is already a
            runs-envelope — feed it to :func:`unwrap_step3_state_v3`).
    """
    if not isinstance(flat_state, dict):
        raise ValueError(
            f"flat_state must be dict, got {type(flat_state).__name__}"
        )

    incoming_schema = flat_state.get("schema_version")
    if incoming_schema not in (None, 1, 2):
        raise ValueError(
            f"Expected flat schema (schema_version absent, 1, or 2), "
            f"got schema_version={incoming_schema!r}"
        )

    target_run_id = active_run_id or DEFAULT_RUN_ID

    v3: Dict[str, Any] = {
        "schema_version": 3,
        "active_run_id": target_run_id,
        "runs": {
            target_run_id: {
                "display_name": (
                    "Default" if target_run_id == DEFAULT_RUN_ID else target_run_id
                ),
                "target": None,
                "created_from": None,
            }
        },
    }

    target_run = v3["runs"][target_run_id]

    # Pass 1: copy workspace-level keys at top-level (except the two
    # specials, already set on the envelope above).
    for key in WORKSPACE_LEVEL_KEYS_V1:
        if key in ("schema_version", "active_run_id"):
            continue
        if key in flat_state:
            v3[key] = flat_state[key]

    # Pass 2: everything else moves under runs[target_run_id].  Unknown
    # keys default here (the safer placement — if the flat schema grows
    # new per-run keys they land in the right slot automatically).
    for key, value in flat_state.items():
        if key in WORKSPACE_LEVEL_KEYS_V1:
            continue
        target_run[key] = value

    return v3


def unwrap_step3_state_v3(
    v3_state: Dict[str, Any],
    active_run_id: str,
) -> Dict[str, Any]:
    """Collapse a v3 envelope into the flat dict the manager uses.

    Mirrors Step 2's ``_unwrap_v3_state``.  Merges the active run's
    per-run state with the top-level workspace-level keys; the result is
    a flat dict identical in shape to what the state manager consumed
    pre-multi-run (so all its existing accessors are untouched).

    Defensive: if ``active_run_id`` is missing or its entry doesn't
    exist, falls back to ``runs[DEFAULT_RUN_ID]`` then ``{}``.

    Args:
        v3_state: parsed v3 envelope dict.
        active_run_id: which run's per-run state to surface.

    Returns:
        Flat state dict (``schema_version`` re-stamped from the envelope).
    """
    flat: Dict[str, Any] = {}

    # Start with workspace-level keys (top-level v3 keys other than the
    # envelope-specific keys).
    skip = {"schema_version", "active_run_id", "runs"}
    for key, value in v3_state.items():
        if key not in skip:
            flat[key] = value

    # Layer in the per-run state (defensive fallback chain).
    runs = v3_state.get("runs", {})
    if not isinstance(runs, dict):
        runs = {}
    run_state = runs.get(active_run_id)
    if not isinstance(run_state, dict):
        run_state = runs.get(DEFAULT_RUN_ID, {})
    if not isinstance(run_state, dict):
        run_state = {}

    for key, value in run_state.items():
        if key not in RUN_MANIFEST_ONLY_KEYS:
            flat[key] = value

    # Re-stamp schema_version so callers can detect what they loaded.
    flat["schema_version"] = v3_state.get("schema_version", 3)
    return flat


def merge_forward_runs(
    new_envelope: Dict[str, Any],
    existing_on_disk: Optional[Dict[str, Any]],
    active_run_id: str,
) -> Dict[str, Any]:
    """No-clobber merge: carry OTHER runs forward from the on-disk file.

    LIVE as of Phase 1a — ``ReconstructionStateManager.save`` calls this.

    Mirrors ``training_annotation_controller.py:9117-9135`` (Step 2's
    controller save).  A save while active on one run must NOT drop the
    other runs' entries (the Phase 8a audit-found clobber bug).  This
    copies every ``other_id != active_run_id`` run entry forward from
    ``existing_on_disk["runs"]`` into ``new_envelope["runs"]``, and
    preserves the ACTIVE run's manifest fields (``display_name`` /
    ``target`` / ``created_from`` — see :data:`RUN_MANIFEST_ONLY_KEYS`)
    from disk so a plain per-patch save doesn't blank them.

    ``existing_on_disk=None`` / not-a-dict / missing-or-corrupt ``runs``
    are all handled gracefully — the ``new_envelope`` is returned with
    only its own active run.  Mutates + returns ``new_envelope`` in place.

    Args:
        new_envelope: the freshly-wrapped v3 envelope for the active run.
        existing_on_disk: parsed contents of the current on-disk file
            (any shape; only a dict with a ``runs`` dict contributes).
        active_run_id: the run currently being saved.

    Returns:
        ``new_envelope`` with the other runs merged forward.
    """
    new_runs = new_envelope.setdefault("runs", {})
    if not isinstance(new_runs, dict):
        new_runs = {}
        new_envelope["runs"] = new_runs

    if not isinstance(existing_on_disk, dict):
        return new_envelope

    existing_runs = existing_on_disk.get("runs")
    if not isinstance(existing_runs, dict):
        return new_envelope

    # Copy every OTHER run's entry forward verbatim.
    for other_id, other_state in existing_runs.items():
        if other_id != active_run_id and isinstance(other_state, dict):
            new_runs[other_id] = other_state

    # Preserve the active run's manifest fields from disk.
    existing_active = existing_runs.get(active_run_id)
    if isinstance(existing_active, dict):
        active_entry = new_runs.get(active_run_id)
        if isinstance(active_entry, dict):
            for preserve_key in RUN_MANIFEST_ONLY_KEYS:
                if preserve_key in existing_active:
                    active_entry[preserve_key] = existing_active[preserve_key]

    return new_envelope


# ----------------------------------------------------------------------------
# Physical migration: data classes
# ----------------------------------------------------------------------------

@dataclass
class FileMove:
    """A single file/dir move in the runs migration plan."""

    src: Path
    dst: Path
    kind: str         # "dir" or "file"
    purpose: str      # human-readable description for the dialog


@dataclass
class MigrationPlan:
    """Read-only description of every change the runs migration would make.

    Returned by :func:`plan_migration`.  Consumed by
    :func:`execute_migration` (and, later, the UI dialog).
    """

    workspace: Path
    default_run_id: str = DEFAULT_RUN_ID
    moves: List[FileMove] = field(default_factory=list)
    state_rewrite_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def total_operations(self) -> int:
        return len(self.moves) + (1 if self.state_rewrite_path else 0)

    @property
    def is_empty(self) -> bool:
        return self.total_operations == 0


@dataclass
class ExecutionResult:
    """Outcome of :func:`execute_migration`."""

    success: bool
    completed_steps: List[str] = field(default_factory=list)
    failed_step: Optional[str] = None
    error: Optional[str] = None
    backup_dir: Optional[Path] = None
    auto_rolled_back: bool = False
    warnings: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Console log for the migration flow (visible in napari console)."""
    print(f"[STEP3-RUNS-MIGRATE] {msg}")


# ----------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------

def plan_migration(
    workspace: _paths._WS,
    *,
    default_run_id: str = DEFAULT_RUN_ID,
) -> MigrationPlan:
    """Build the runs-migration plan for a workspace.  READ-ONLY.

    Enumerates the per-patch reconstruction dir moves
    (``runs/{patch}`` → ``runs/{default}/reconstructions/{patch}``) plus
    the state-JSON content rewrite.  No filesystem mutation — safe to
    call repeatedly.

    Idempotent: a child dir already named ``default_run_id`` (i.e. the
    run context dir itself) is skipped, so re-planning a partially- or
    fully-migrated workspace yields only the remaining moves.

    dst-orphan guard (MED-2): a src whose destination ALREADY exists is
    treated as already-migrated — it is NOT added to the move list (a
    move would be a no-op skip in :func:`execute_migration` and would
    leave the src orphaned forever, making :func:`is_migration_needed`
    loop true permanently).  A warning is recorded instead; the src is
    left in place (never auto-deleted — user data).

    Returns an empty plan if there is nothing under ``runs/`` to move
    and no state file to rewrite.
    """
    ws = Path(workspace)
    plan = MigrationPlan(workspace=ws, default_run_id=default_run_id)

    runs_root = _paths.runs_dir(ws)                                  # step3/runs
    recon_root = _paths.reconstructions_dir(ws, default_run_id)      # runs/{default}/reconstructions

    if runs_root.is_dir():
        for child in sorted(runs_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name == default_run_id:
                # The run-context dir itself (already-migrated target).
                continue
            if _is_named_run_context(child):
                # HIGH-1: a NAMED run's context dir (``runs/{slug}/`` with a
                # ``run_manifest.json``) lives at the SAME level as the
                # default-run per-patch dirs.  It is ALREADY nested — moving
                # it into ``runs/default/reconstructions/{slug}/`` would
                # orphan the whole named run silently.  Skip it entirely.
                continue
            dst = recon_root / child.name
            if dst.exists():
                # dst-orphan: destination already present.  Skip the move
                # (do not orphan the src behind a no-op move) + warn.
                plan.warnings.append(
                    f"destination already exists for {child.name!r} "
                    f"({dst}); skipping move, leaving source in place"
                )
                continue
            plan.moves.append(FileMove(
                src=child,
                dst=dst,
                kind="dir",
                purpose=(
                    f"per-patch reconstruction {child.name!r} → "
                    f"runs/{default_run_id}/reconstructions/"
                ),
            ))

    # Sibling step-3 sub-trees (qc / curation / prediction_cache /
    # refinement) move WHOLE into the default run's context dir.  Flat
    # source = run_id=None (e.g. ``step3/qc``); nested dest = the default
    # run slug (e.g. ``step3/runs/default/qc``).  Same dst-orphan guard as
    # the per-patch loop: a src whose dest already exists is left in place
    # (never clobbered) + a warning is recorded.  Workspace-SHARED trees
    # (``snapshots/`` / ``training_archive/`` / ``batch_refine_recipes.json``)
    # are deliberately NOT moved — they live at the step-3 root by design.
    # So is ``refine_chain_recipes.json`` (Refine recipes).
    for label, getter in _SIBLING_DIR_GETTERS:
        src = getter(ws, None)
        dst = getter(ws, default_run_id)
        if not src.is_dir():
            continue
        if dst.exists():
            plan.warnings.append(
                f"destination already exists for {label!r} "
                f"({dst}); skipping move, leaving source in place"
            )
            continue
        plan.moves.append(FileMove(
            src=src,
            dst=dst,
            kind="dir",
            purpose=f"{label} → runs/{default_run_id}/{label}/",
        ))

    state_path = _paths.state_json_path(ws)
    if state_path.is_file():
        plan.state_rewrite_path = state_path

    _log(
        f"plan_migration: {len(plan.moves)} moves, "
        f"state_rewrite={'yes' if plan.state_rewrite_path else 'no'}"
    )
    return plan


# ----------------------------------------------------------------------------
# Backup + rollback
# ----------------------------------------------------------------------------

def create_backup(
    workspace: _paths._WS, timestamp: Optional[str] = None,
) -> Path:
    """Create a backup of the Step 3 sub-tree + state JSON before migration.

    The backup is a sibling directory at workspace root named
    ``.pre_step3_runs_<timestamp>/``.  Copies ``artifacts/step3/`` (which
    contains ``runs/``, ``qc/``, ``curation/`` …) and the
    ``step3_state.json`` file.  Idempotent — re-creating the same backup
    skips already-present subtrees.

    Returns the backup directory path.
    """
    ws = Path(workspace)
    if timestamp is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_dir = ws / f"{BACKUP_DIR_PREFIX}{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    _log(f"create_backup: copying step3 artifacts to {backup_dir}")

    step3 = _paths.step3_root(ws)
    if step3.is_dir():
        dst = backup_dir / "artifacts" / _paths.STEP3_ARTIFACTS_SUBDIR
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(step3, dst)

    state = _paths.state_json_path(ws)
    if state.is_file():
        dst = backup_dir / _paths.STEP3_STATE_FILENAME
        if not dst.exists():
            shutil.copy2(state, dst)

    _log(f"create_backup: complete at {backup_dir}")
    return backup_dir


def rollback_migration(backup_dir: _paths._WS, workspace: _paths._WS) -> None:
    """Restore a workspace from a runs-migration backup.

    Destroys the (possibly partially-migrated) ``artifacts/step3/``
    subtree + ``step3_state.json`` and re-installs them from
    ``backup_dir``.  Idempotent — re-running on an already-rolled-back
    workspace is safe.
    """
    backup = Path(backup_dir)
    ws = Path(workspace)
    _log(f"rollback_migration: restoring {ws} from backup {backup}")

    if not backup.is_dir():
        raise FileNotFoundError(f"Backup directory does not exist: {backup}")

    # 1. Tear down any post-migration step3 subtree.
    step3 = _paths.step3_root(ws)
    if step3.is_dir():
        shutil.rmtree(step3)

    # 2. Restore step3 subtree from backup.
    backup_step3 = backup / "artifacts" / _paths.STEP3_ARTIFACTS_SUBDIR
    if backup_step3.is_dir():
        step3.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(backup_step3, step3, symlinks=False)

    # 3. Restore state JSON.
    backup_state = backup / _paths.STEP3_STATE_FILENAME
    if backup_state.is_file():
        shutil.copy2(backup_state, _paths.state_json_path(ws))


# ----------------------------------------------------------------------------
# Execute
# ----------------------------------------------------------------------------

def execute_migration(
    plan: MigrationPlan,
    backup_dir: Optional[Path],
    auto_rollback_on_failure: bool = True,
) -> ExecutionResult:
    """Apply a runs-migration plan.  MUTATES the workspace.

    Caller SHOULD have created ``backup_dir`` via :func:`create_backup`
    first.  On any failure mid-execute, if ``auto_rollback_on_failure``
    is True (default) and a backup was supplied, the workspace is
    restored from ``backup_dir`` and the failure is reported.

    Move semantics are idempotent: a destination that already exists is
    skipped (prior partial execute), so re-running is safe.
    """
    result = ExecutionResult(
        success=True,
        backup_dir=Path(backup_dir) if backup_dir is not None else None,
    )
    current_phase = "(not started)"

    try:
        # Step 1: per-patch dir moves.
        current_phase = "step 1 — reconstruction dir moves"
        _log(f"execute_migration: step 1 — {len(plan.moves)} dir moves")
        for move in plan.moves:
            move.dst.parent.mkdir(parents=True, exist_ok=True)
            if move.dst.exists():
                # dst-orphan / idempotent: dst already present (prior
                # partial execute, or appeared between plan + execute).
                # Skip the move (never clobber) + warn; the src is left
                # in place rather than orphaned behind a silent no-op.
                result.warnings.append(
                    f"destination already exists for {move.src.name!r} "
                    f"({move.dst}); skipped move, source left in place"
                )
                _log(
                    f"execute_migration: dst exists, skipping move "
                    f"{move.src} → {move.dst}"
                )
                continue
            shutil.move(str(move.src), str(move.dst))
            result.completed_steps.append(f"move: {move.src} → {move.dst}")

        # Step 2: state-JSON content rewrite (drop reconstruction_run_id).
        if plan.state_rewrite_path is not None:
            current_phase = "step 2 — state JSON rewrite"
            _log("execute_migration: step 2 — state JSON rewrite")
            _execute_state_rewrite(plan.state_rewrite_path)
            result.completed_steps.append(
                f"rewrite: {plan.state_rewrite_path.name}"
            )

        _log(
            f"execute_migration: SUCCESS "
            f"({len(result.completed_steps)} ops applied)"
        )

    except Exception as e:  # noqa: BLE001 — migration must never half-apply silently
        result.success = False
        result.error = f"{type(e).__name__}: {e}"
        result.failed_step = current_phase
        _log(f"execute_migration: FAILED at '{current_phase}': {result.error}")
        if auto_rollback_on_failure and backup_dir is not None:
            _log("execute_migration: attempting auto-rollback")
            try:
                rollback_migration(backup_dir, plan.workspace)
                result.auto_rolled_back = True
                _log("execute_migration: auto-rollback SUCCEEDED")
            except Exception as rollback_e:  # noqa: BLE001
                result.error = (
                    f"{result.error} (rollback ALSO FAILED: "
                    f"{type(rollback_e).__name__}: {rollback_e})"
                )
                _log(f"execute_migration: auto-rollback FAILED: {rollback_e}")

    return result


def migrate_state_json_content(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pure content rewrite: drop the per-patch ``reconstruction_run_id``.

    Mirrors ``_migrate_step3_layout.migrate_state_json_content`` (which
    drops the same field per design Q1).  Handles both the flat shape
    (``patches`` top-level) and the v2-nested shape
    (``runs[*].patches``) defensively.  Returns a NEW dict (deep copy);
    the input is not mutated.
    """
    out = json.loads(json.dumps(state))  # deep copy via JSON round-trip

    def _scrub_patches(patches: Any) -> None:
        if isinstance(patches, dict):
            for record in patches.values():
                if isinstance(record, dict):
                    record.pop("reconstruction_run_id", None)
                    record.pop("run_id", None)

    if isinstance(out, dict):
        _scrub_patches(out.get("patches"))
        runs = out.get("runs")
        if isinstance(runs, dict):
            for run_state in runs.values():
                if isinstance(run_state, dict):
                    _scrub_patches(run_state.get("patches"))
    return out


def _execute_state_rewrite(state_path: Path) -> None:
    """Read state JSON, drop per-patch ``reconstruction_run_id``, atomic-write."""
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return  # nothing to rewrite

    rewritten = migrate_state_json_content(data)

    base = state_path.parent
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=".step3_runs_state_", dir=str(base),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rewritten, f, indent=2)
        os.replace(tmp_path, str(state_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------------
# Convenience predicate
# ----------------------------------------------------------------------------

def is_migration_needed(
    workspace: _paths._WS, *, default_run_id: str = DEFAULT_RUN_ID,
) -> bool:
    """True if the flat ``runs/{patch}/`` layout has un-migrated patch dirs.

    i.e. there is at least one child of ``artifacts/step3/runs/`` that is
    a directory, is NOT the ``default_run_id`` run-context dir, AND whose
    migration destination does NOT already exist.

    dst-orphan guard (MED-2): a src whose destination
    (``runs/{default}/reconstructions/{patch}``) already exists is
    treated as already-migrated — it does NOT keep this predicate true.
    Without this clause a dst-orphan (src + dst both present) would make
    the migration appear perpetually "needed" even though every move is
    a no-op skip.
    """
    ws = Path(workspace)
    runs_root = _paths.runs_dir(ws)
    if runs_root.is_dir():
        recon_root = _paths.reconstructions_dir(ws, default_run_id)
        for child in runs_root.iterdir():
            if not child.is_dir() or child.name == default_run_id:
                continue
            if _is_named_run_context(child):
                # HIGH-1: a named-run context dir is already nested — it is
                # NOT a pending default-run move (mirrors plan_migration).
                continue
            if (recon_root / child.name).exists():
                # dst already present — orphan / already-migrated; not pending.
                continue
            return True
    # Sibling sub-trees (qc / curation / prediction_cache / refinement)
    # with a flat source + absent nested dest are also un-migrated.  Same
    # dst-orphan guard as the per-patch loop.
    for _label, getter in _SIBLING_DIR_GETTERS:
        src = getter(ws, None)
        dst = getter(ws, default_run_id)
        if src.is_dir() and not dst.exists():
            return True
    return False


# ----------------------------------------------------------------------------
# Skip sentinel
# ----------------------------------------------------------------------------

def is_migration_skipped(workspace: _paths._WS) -> bool:
    """True if the user previously chose to skip the runs migration."""
    return (Path(workspace) / MIGRATION_SKIPPED_SENTINEL).is_file()


def mark_migration_skipped(workspace: _paths._WS) -> None:
    """Write the skip-sentinel so the migration prompt won't re-fire."""
    sentinel = Path(workspace) / MIGRATION_SKIPPED_SENTINEL
    sentinel.write_text(
        dt.datetime.now(dt.timezone.utc).isoformat() + "\n",
        encoding="utf-8",
    )


def clear_migration_skip(workspace: _paths._WS) -> None:
    """Remove the skip-sentinel (so the prompt fires again).  Idempotent."""
    sentinel = Path(workspace) / MIGRATION_SKIPPED_SENTINEL
    if sentinel.is_file():
        sentinel.unlink()


# ----------------------------------------------------------------------------
# Public API surface
# ----------------------------------------------------------------------------

__all__ = [
    # Constants
    "DEFAULT_RUN_ID",
    "MIGRATION_SKIPPED_SENTINEL",
    "BACKUP_DIR_PREFIX",
    "WORKSPACE_LEVEL_KEYS_V1",
    "RUN_MANIFEST_ONLY_KEYS",
    # State envelope wrap / unwrap / merge-forward
    "wrap_step3_state_to_v3",
    "unwrap_step3_state_v3",
    "merge_forward_runs",
    # Data classes
    "FileMove",
    "MigrationPlan",
    "ExecutionResult",
    # Plan / execute / rollback
    "plan_migration",
    "create_backup",
    "execute_migration",
    "rollback_migration",
    "migrate_state_json_content",
    # Predicate + sentinel
    "is_migration_needed",
    "is_migration_skipped",
    "mark_migration_skipped",
    "clear_migration_skip",
]

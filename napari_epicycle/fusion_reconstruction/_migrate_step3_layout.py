"""Step 3 layout migration: legacy → new (v2) on-disk reshape.

Phase 2 of the directory restructure (2026-05-20).


Two-stage flow:

1. :func:`plan_migration` — READ-ONLY scan.  Returns a
   :class:`MigrationPlan` describing every file move + state-JSON
   rewrite that would happen.  No filesystem mutation.  Safe to
   call repeatedly.

2. :func:`execute_migration` — applies the plan.  Mutates the
   workspace.  Caller MUST have created a backup via
   :func:`create_backup` first (the dialog enforces this).

Rollback: :func:`rollback_migration` restores from the backup dir
created at step 1.  Idempotent — re-running it on an already-rolled-
back workspace is a no-op.

Pure-Python — no Qt / napari dependency.  All Qt-related interaction
(dialog, user choices) is in the sibling ``_migrate_step3_dialog``
module so this one can be unit-tested without a display.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import _paths as p


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FileMove:
    """A single file or directory move operation in the migration plan."""

    src: Path
    dst: Path
    kind: str               # "dir" or "file"
    purpose: str            # human-readable description for the dialog


@dataclass
class CellposeLift:
    """A post-move lift of a Cellpose weights file from the auto-inserted
    ``models/<bucket>`` subdir up to the bundle root ``<bucket>``."""

    src: Path
    dst: Path
    bundle_name: str
    bucket: str             # "lateral" or "axial"


@dataclass
class MigrationPlan:
    """Read-only description of every change a migration would make.

    Returned by :func:`plan_migration`.  Consumed by
    :func:`execute_migration` and the UI dialog (for enumeration).
    """

    workspace: Path
    moves: list = field(default_factory=list)                  # list[FileMove]
    cellpose_lifts: list = field(default_factory=list)         # list[CellposeLift]
    legacy_state_path: Optional[Path] = None
    new_state_path: Optional[Path] = None
    warnings: list = field(default_factory=list)               # list[str]
    is_mixed_layout: bool = False

    @property
    def total_operations(self) -> int:
        return len(self.moves) + len(self.cellpose_lifts) + (
            1 if self.legacy_state_path else 0
        )

    @property
    def is_empty(self) -> bool:
        return self.total_operations == 0


@dataclass
class ExecutionResult:
    """Outcome of :func:`execute_migration`."""

    success: bool
    completed_steps: list = field(default_factory=list)        # list[str]
    failed_step: Optional[str] = None
    error: Optional[str] = None
    backup_dir: Optional[Path] = None
    auto_rolled_back: bool = False  # set to True iff failure triggered + completed auto-rollback


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def plan_migration(workspace) -> MigrationPlan:
    """Build the migration plan for a workspace.  READ-ONLY.

    Detects legacy markers and enumerates every move that would be
    needed to bring the workspace to the v2 layout.  Returns an
    empty plan (``is_empty=True``) if the workspace is already on v2
    or is empty / non-Step-3.
    """
    ws = Path(workspace)
    plan = MigrationPlan(workspace=ws)

    if p.is_new_layout(ws):
        # Already migrated — nothing to do.
        return plan

    if p.is_mixed_layout(ws):
        plan.is_mixed_layout = True
        plan.warnings.append(
            "Workspace has BOTH legacy and new layout markers — "
            "likely a previous migration was interrupted.  Review the "
            "plan carefully before executing."
        )

    if not p.is_legacy_layout(ws) and not p.is_mixed_layout(ws):
        # No legacy markers and not mixed — empty workspace, nothing to do.
        return plan

    # 1. Top-level legacy dirs that move wholesale -----------------------------

    _maybe_plan_move(
        plan,
        src=p.legacy_qc_dir(ws),
        dst=p.qc_dir(ws),
        purpose="QC features + raw-feature backups",
    )
    _maybe_plan_move(
        plan,
        src=p.legacy_snapshots_dir(ws),
        dst=p.snapshots_dir(ws),
        purpose="Pre-commit snapshots (append-only training data)",
    )
    _maybe_plan_move(
        plan,
        src=p.legacy_prediction_cache_dir(ws),
        dst=p.prediction_cache_dir(ws),
        purpose="Cellpose prediction cache",
    )

    # 2. Per-patch reconstruction runs ----------------------------------------
    # Maps run_id (UUID-ish dir under artifacts/reconstruction/) →
    # patch_name from the legacy state JSON.

    legacy_state_path = p.legacy_state_json_path(ws)
    if legacy_state_path.is_file():
        plan.legacy_state_path = legacy_state_path
        plan.new_state_path = p.state_json_path(ws)
        _plan_reconstruction_runs(plan, ws, legacy_state_path)

    # 3. Refinement training pairs (per-patch) --------------------------------

    _plan_refinement_pairs(plan, ws)

    # 4. Refined model bundles + Cellpose lift --------------------------------

    _plan_refined_model_bundles(plan, ws)

    return plan


def _maybe_plan_move(plan: MigrationPlan, *, src: Path, dst: Path,
                     purpose: str) -> None:
    """Add a directory-or-file move to the plan iff ``src`` exists."""
    if not src.exists():
        return
    if dst.exists():
        plan.warnings.append(
            f"Destination already exists for '{purpose}': {dst}.  "
            "Skipping this move to avoid clobbering."
        )
        return
    kind = "dir" if src.is_dir() else "file"
    plan.moves.append(FileMove(src=src, dst=dst, kind=kind, purpose=purpose))


def _plan_reconstruction_runs(plan: MigrationPlan, ws: Path,
                              legacy_state_path: Path) -> None:
    """Plan moves for each per-patch reconstruction run directory."""
    try:
        state = json.loads(legacy_state_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        plan.warnings.append(
            f"Could not parse legacy state JSON ({legacy_state_path}): {e}.  "
            "Per-patch reconstruction runs will be left in place."
        )
        return

    # Defensive: if a state JSON named `reconstruction_state.json`
    # declares it's already on layout v2, the workspace is in an
    # inconsistent state (filename says legacy but content says new).
    # Refuse to plan any moves; surface a warning so the user
    # investigates.
    declared_layout = state.get("layout_version")
    if declared_layout is not None and declared_layout >= p.CURRENT_LAYOUT_VERSION:
        plan.warnings.append(
            f"Legacy-named state JSON ({legacy_state_path.name}) "
            f"declares layout_version={declared_layout}.  Workspace is "
            "in an inconsistent state — investigate manually before "
            "migrating."
        )
        return

    patches = state.get("patches", {})
    if not isinstance(patches, dict):
        return

    seen_dsts: set = set()
    for patch_name, record in patches.items():
        if not isinstance(record, dict):
            continue
        run_id = (
            record.get("reconstruction_run_id")
            or record.get("run_id")
        )
        if not run_id:
            continue
        src = p.legacy_reconstruction_root(ws) / str(run_id)
        if not src.is_dir():
            plan.warnings.append(
                f"Legacy state references run_id={run_id!r} for "
                f"patch {patch_name!r} but {src} does not exist."
            )
            continue
        dst = p.patch_run_dir(ws, patch_name)
        if dst in seen_dsts:
            plan.warnings.append(
                f"Duplicate destination for patch {patch_name!r}; "
                "skipping subsequent occurrences."
            )
            continue
        seen_dsts.add(dst)
        if dst.exists():
            plan.warnings.append(
                f"Destination already exists for patch {patch_name!r}: "
                f"{dst}.  Skipping."
            )
            continue
        plan.moves.append(FileMove(
            src=src, dst=dst, kind="dir",
            purpose=f"Reconstruction run for {patch_name}",
        ))


def _plan_refinement_pairs(plan: MigrationPlan, ws: Path) -> None:
    """Plan moves for ``artifacts/refinement_training_data/*``."""
    legacy_root = p.legacy_refinement_training_data(ws)
    if not legacy_root.is_dir():
        return

    for entry in sorted(legacy_root.iterdir()):
        if entry.is_dir():
            dst = p.patch_pairs_dir(ws, entry.name)
            if dst.exists():
                plan.warnings.append(
                    f"Destination already exists for training pairs of "
                    f"patch {entry.name!r}: {dst}.  Skipping."
                )
                continue
            plan.moves.append(FileMove(
                src=entry, dst=dst, kind="dir",
                purpose=f"Training pairs for {entry.name}",
            ))
        elif entry.is_file() and entry.name == "extraction_manifest.json":
            dst = p.refinement_manifest_path(ws)
            if dst.exists():
                plan.warnings.append(
                    f"Refinement manifest already exists at {dst}.  Skipping."
                )
                continue
            plan.moves.append(FileMove(
                src=entry, dst=dst, kind="file",
                purpose="Refinement extraction manifest",
            ))


def _plan_refined_model_bundles(plan: MigrationPlan, ws: Path) -> None:
    """Plan bundle moves + per-bundle Cellpose-output lifts."""
    legacy_models = p.legacy_refinement_models_root(ws)
    if not legacy_models.is_dir():
        return

    for bundle in sorted(legacy_models.iterdir()):
        if not bundle.is_dir():
            continue
        new_bundle = p.refined_model_dir(ws, bundle.name)
        if new_bundle.exists():
            plan.warnings.append(
                f"Destination already exists for model bundle "
                f"{bundle.name!r}: {new_bundle}.  Skipping."
            )
            continue
        plan.moves.append(FileMove(
            src=bundle, dst=new_bundle, kind="dir",
            purpose=f"Refined model bundle '{bundle.name}'",
        ))
        # After the bundle moves, Cellpose's auto-inserted
        # bundle/models/<bucket> path needs to be lifted up to
        # bundle/<bucket>.  We don't know yet which buckets exist —
        # the plan lists BOTH lateral and axial, and execute will
        # skip the ones whose src doesn't exist.
        for bucket in ("lateral", "axial"):
            plan.cellpose_lifts.append(CellposeLift(
                src=new_bundle / "models" / bucket,
                dst=new_bundle / bucket,
                bundle_name=bundle.name,
                bucket=bucket,
            ))


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------


def create_backup(workspace, timestamp: Optional[str] = None) -> Path:
    """Create a pre-migration backup of the workspace's mutable state.

    Copies ``artifacts/``, ``models/``, and the legacy state JSON
    (``reconstruction_state.json``) into
    ``{workspace}/.legacy_pre_restructure_{ts}/``.

    Returns the backup directory path.  The backup is the rollback
    target; per design Q3 it is retained forever — the user manually
    deletes it when confident the migration is good.
    """
    ws = Path(workspace)
    ts = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ws / f".legacy_pre_restructure_{ts}"
    if backup_dir.exists():
        raise FileExistsError(
            f"Backup directory already exists: {backup_dir}.  Either "
            "delete it or use a different timestamp."
        )
    backup_dir.mkdir(parents=True, exist_ok=False)

    # Subtree copies — only present items.
    for top in ("artifacts", "models"):
        src = ws / top
        if src.is_dir():
            shutil.copytree(src, backup_dir / top, symlinks=False)

    # Legacy state JSON file.
    legacy_state = p.legacy_state_json_path(ws)
    if legacy_state.is_file():
        shutil.copy2(legacy_state, backup_dir / legacy_state.name)

    return backup_dir


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def execute_migration(plan: MigrationPlan, *,
                      backup_dir: Optional[Path] = None) -> ExecutionResult:
    """Apply the migration plan to the workspace.

    Caller is responsible for having created a backup via
    :func:`create_backup` first.  ``backup_dir`` is recorded in the
    result for downstream rollback if anything fails.

    On success, the workspace is in the v2 layout and the legacy
    state JSON has been deleted.  On failure, the workspace may be
    in a partially-migrated state — the caller should invoke
    :func:`rollback_migration` against the recorded ``backup_dir``.
    """
    ws = plan.workspace
    completed: list = []

    try:
        # 1. Skeleton.
        p.step3_root(ws).mkdir(parents=True, exist_ok=True)
        p.runs_dir(ws).mkdir(parents=True, exist_ok=True)
        p.refinement_root(ws).mkdir(parents=True, exist_ok=True)
        p.refinement_pairs_root(ws).mkdir(parents=True, exist_ok=True)
        p.refined_models_root(ws).mkdir(parents=True, exist_ok=True)
        completed.append("Created step3 skeleton")

        # 2. File / directory moves.
        for move in plan.moves:
            move.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(move.src), str(move.dst))
            completed.append(f"Moved {move.src} -> {move.dst}")

        # 2b. Rewrite per-patch reconstruction manifest.json files.
        # The directories moved in step 2, but each manifest's inner
        # ``labels_path`` field still points at the old legacy
        # ``artifacts/reconstruction/{run_id}/labels_3d.tif`` location.
        # Two sources of truth (state JSON + inner manifest) would
        # disagree until rewritten.  Audit finding 2026-05-20.
        for move in plan.moves:
            if not move.purpose.startswith("Reconstruction run for "):
                continue
            patch_name = move.purpose.removeprefix(
                "Reconstruction run for "
            )
            inner_manifest = move.dst / "manifest.json"
            if not inner_manifest.is_file():
                continue
            try:
                payload = json.loads(
                    inner_manifest.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError) as e:
                completed.append(
                    f"[STEP3-MIGRATE] WARN could not rewrite inner "
                    f"manifest {inner_manifest}: {e}"
                )
                continue
            new_labels = move.dst / "labels_3d.tif"
            if new_labels.exists():
                payload["labels_path"] = str(new_labels)
            try:
                inner_manifest.write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
                completed.append(
                    f"Rewrote inner manifest {inner_manifest} "
                    f"(patch {patch_name})"
                )
            except OSError as e:
                completed.append(
                    f"[STEP3-MIGRATE] WARN inner manifest write failed "
                    f"{inner_manifest}: {e}"
                )

        # 3. Cellpose lifts.  Per, "no fighting Cellpose" —
        # we let Cellpose write to bundle/models/<bucket> and lift the
        # files up post-train.  For pre-existing legacy bundles, the
        # original wrong-path assumption means the legacy data may
        # already be at the flat path OR under models/ — we handle both.
        for lift in plan.cellpose_lifts:
            if lift.src.exists():
                if lift.dst.exists():
                    completed.append(
                        f"[PERVIEW-CELLPOSE-IO] skip lift "
                        f"{lift.src} (dst {lift.dst} already present)"
                    )
                    continue
                shutil.move(str(lift.src), str(lift.dst))
                completed.append(
                    f"[PERVIEW-CELLPOSE-IO] lifted {lift.src} -> {lift.dst}"
                )

        # 3b. Defensive companion-file lift.  Audit finding 2026-05-20:
        # the per-bucket lifts above only cover the weights file at
        # bundle/models/<bucket>.  Some Cellpose versions or
        # user-side reorganization may have COMPANION files (e.g.,
        # `_train_loss.npy`, `_val_loss.npy`, `_training_config.json`)
        # ALSO under bundle/models/.  Catch any straggler at top-level
        # of the cellpose-internal models/ subdir and lift it to the
        # bundle root.
        cellpose_subdirs_seen: set = set()
        for lift in plan.cellpose_lifts:
            cellpose_subdir = lift.dst.parent / "models"
            if cellpose_subdir in cellpose_subdirs_seen:
                continue
            cellpose_subdirs_seen.add(cellpose_subdir)
            if not cellpose_subdir.is_dir():
                continue
            for stray in list(cellpose_subdir.iterdir()):
                if not stray.is_file():
                    continue
                stray_dst = lift.dst.parent / stray.name
                if stray_dst.exists():
                    completed.append(
                        f"[PERVIEW-CELLPOSE-IO] skip stray lift "
                        f"{stray} (dst exists)"
                    )
                    continue
                shutil.move(str(stray), str(stray_dst))
                completed.append(
                    f"[PERVIEW-CELLPOSE-IO] lifted companion "
                    f"{stray} -> {stray_dst}"
                )

        # 4. Clean up any now-empty bundle/models/ subdirs.
        for lift in plan.cellpose_lifts:
            cellpose_subdir = lift.dst.parent / "models"
            if (
                cellpose_subdir.is_dir()
                and not any(cellpose_subdir.iterdir())
            ):
                cellpose_subdir.rmdir()
                completed.append(
                    f"[PERVIEW-CELLPOSE-IO] removed empty {cellpose_subdir}"
                )

        # 5. Rewrite state JSON.  Atomic: write to a temp sibling
        # then os.replace into place, then delete the legacy file
        # only after the new file is durable on disk.  Without this,
        # a crash between write_text and unlink leaves a mixed-layout
        # workspace (legacy AND new state JSONs both present, both
        # with content — a hazard).
        if plan.legacy_state_path is not None and plan.legacy_state_path.is_file():
            legacy_content = json.loads(plan.legacy_state_path.read_text())
            new_content = migrate_state_json_content(legacy_content, ws)
            assert plan.new_state_path is not None
            import os
            import tempfile
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=plan.new_state_path.name + ".",
                suffix=".tmp",
                dir=str(plan.new_state_path.parent),
            )
            try:
                with os.fdopen(tmp_fd, "w") as fh:
                    fh.write(json.dumps(new_content, indent=2))
                os.replace(tmp_name, str(plan.new_state_path))
            except Exception:
                # Best-effort cleanup of the temp file on failure.
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
            completed.append(f"Wrote new state JSON {plan.new_state_path}")
            plan.legacy_state_path.unlink()
            completed.append(
                f"Removed legacy state JSON {plan.legacy_state_path}"
            )

        # 6. Remove now-empty legacy directories.
        for legacy_dir in (
            p.legacy_reconstruction_root(ws),
            p.legacy_refinement_training_data(ws),
            p.legacy_refinement_models_root(ws),
        ):
            if legacy_dir.is_dir() and not any(legacy_dir.iterdir()):
                legacy_dir.rmdir()
                completed.append(f"Removed empty {legacy_dir}")

        return ExecutionResult(
            success=True,
            completed_steps=completed,
            backup_dir=backup_dir,
        )

    except Exception as e:
        # Auto-rollback on failure when a backup is available.  The
        # There are three rollback layers; the dialog-based
        # rollback (layer 2) is the user-facing one, but if the
        # dialog itself fails to render (Qt teardown, parent
        # destroyed) the user is stuck with a half-migrated workspace.
        # Auto-rolling-back here ensures the workspace is recoverable
        # even when the failure dialog never fires.  The user is then
        # shown the failure dialog WITHOUT a rollback button (we've
        # already done it).
        auto_rolled_back = False
        if backup_dir is not None and Path(backup_dir).is_dir():
            try:
                rollback_migration(backup_dir, plan.workspace)
                auto_rolled_back = True
                completed.append(
                    f"[STEP3-MIGRATE] Auto-rolled-back from {backup_dir}"
                )
            except Exception as rb_err:
                completed.append(
                    f"[STEP3-MIGRATE] Auto-rollback FAILED: "
                    f"{type(rb_err).__name__}: {rb_err}"
                )
        return ExecutionResult(
            success=False,
            completed_steps=completed,
            failed_step=completed[-1] if completed else None,
            error=(
                f"{type(e).__name__}: {e}"
                + (
                    "  (workspace auto-rolled-back to pre-migration state)"
                    if auto_rolled_back else ""
                )
            ),
            backup_dir=backup_dir,
            auto_rolled_back=auto_rolled_back,
        )


def migrate_state_json_content(state: dict, workspace) -> dict:
    """Rewrite the state JSON dict in-place style (returns a new dict).

    Field-by-field changes (state-JSON migration table):

    * ``schema_version``: bumped to 2.
    * ``layout_version``: NEW, set to ``CURRENT_LAYOUT_VERSION`` (2).
    * ``patches[name].labels_path``: rewritten to the new runs/ path.
    * ``patches[name].qc_features_csv``: rewritten to the new qc/ path.
    * ``patches[name].reconstruction_run_id``: dropped entirely
      (design Q1).
    * ``lateral_model_path`` / ``axial_model_path``: rewritten to the
      refined-bundle layout when the legacy path is under
      ``models/refinement/``.
    """
    ws = Path(workspace)
    out = json.loads(json.dumps(state))  # deep copy via JSON round-trip
    out["layout_version"] = p.CURRENT_LAYOUT_VERSION
    out["schema_version"] = 2

    patches = out.get("patches")
    if isinstance(patches, dict):
        for patch_name, record in patches.items():
            if not isinstance(record, dict):
                continue
            # labels_path → new runs/ location (TIFF assumed; format-
            # agnostic readers handle the .tif suffix at the I/O site).
            if "labels_path" in record:
                record["labels_path"] = (
                    str(p.patch_labels_stem(ws, patch_name)) + ".tif"
                )
            # qc_features_csv → new qc/ location.
            if "qc_features_csv" in record:
                record["qc_features_csv"] = str(
                    p.qc_features_csv(ws, patch_name)
                )
            # Drop reconstruction_run_id entirely per design Q1.
            record.pop("reconstruction_run_id", None)
            record.pop("run_id", None)

    # Top-level model paths → refined bundle layout.
    for key, bucket in (("lateral_model_path", "lateral"),
                        ("axial_model_path", "axial")):
        old_value = out.get(key)
        if not old_value:
            continue
        rewritten = _rewrite_model_path(old_value, ws, bucket)
        if rewritten is not None:
            out[key] = rewritten

    return out


def _rewrite_model_path(old_value: str, ws: Path, bucket: str) -> Optional[str]:
    """Best-effort rewrite of a legacy model path to the new bundle layout.

    Returns ``None`` if the path can't be normalized (caller leaves
    the field as-is).  Conservative — paths that don't clearly live
    under the legacy ``models/refinement/`` tree are left alone, the
    user can fix them via the registry view post-migration.
    """
    try:
        old_path = Path(old_value)
        legacy_root = p.legacy_refinement_models_root(ws)
        # Use raw comparison rather than .resolve() — file may not
        # exist if it was already deleted, and resolve() failures
        # would short-circuit here.
        try:
            rel = old_path.relative_to(legacy_root)
        except ValueError:
            # Not under the legacy tree.  Try resolved forms in case
            # of symlinks / .. — but only if both sides exist.
            if old_path.exists() and legacy_root.exists():
                rel = old_path.resolve().relative_to(legacy_root.resolve())
            else:
                return None
        # rel starts with bundle name; first component is the bundle.
        bundle_name = rel.parts[0]
        return str(p.refined_model_weights_path(ws, bundle_name, bucket))
    except (ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def rollback_migration(backup_dir, workspace) -> None:
    """Restore the workspace from a pre-migration backup.

    Idempotent — safe to call against an already-rolled-back
    workspace.  Destroys any partial v2 state in the workspace and
    re-installs the legacy state from ``backup_dir``.
    """
    backup = Path(backup_dir)
    ws = Path(workspace)

    if not backup.is_dir():
        raise FileNotFoundError(
            f"Backup directory does not exist: {backup}"
        )

    # 1. Tear down any new-layout state that exists.
    if p.step3_root(ws).is_dir():
        shutil.rmtree(p.step3_root(ws))
    new_state = p.state_json_path(ws)
    if new_state.is_file():
        new_state.unlink()

    # 2. Restore from backup.  Wipe + replace each top-level subtree
    # that exists in the backup.
    for top in ("artifacts", "models"):
        backup_src = backup / top
        ws_dst = ws / top
        if backup_src.is_dir():
            if ws_dst.exists():
                if ws_dst.is_dir():
                    shutil.rmtree(ws_dst)
                else:
                    ws_dst.unlink()
            shutil.copytree(backup_src, ws_dst, symlinks=False)

    # 3. Restore legacy state JSON.
    legacy_state_backup = backup / p.LEGACY_STATE_FILENAME
    if legacy_state_backup.is_file():
        shutil.copy2(legacy_state_backup, p.legacy_state_json_path(ws))


# ---------------------------------------------------------------------------
# Convenience predicates
# ---------------------------------------------------------------------------


def is_migration_needed(workspace) -> bool:
    """True if the workspace has any legacy markers (incl. mixed state)."""
    return (
        p.is_legacy_layout(workspace)
        or p.is_mixed_layout(workspace)
    )


# ---------------------------------------------------------------------------
# Skip-flag sentinel (for the "Don't ask again" choice)
# ---------------------------------------------------------------------------

#: Workspace-level sentinel file.  If present, the auto-prompt is
#: suppressed for this workspace.  User can re-trigger migration via
#: a menu action that ignores the sentinel.
_SKIP_SENTINEL_NAME = ".step3_migration_skipped"


def is_migration_skipped(workspace) -> bool:
    """True if the user previously chose "Don't ask again" for this workspace."""
    return (Path(workspace) / _SKIP_SENTINEL_NAME).exists()


def mark_migration_skipped(workspace) -> None:
    """Write the skip sentinel.  Idempotent."""
    sentinel = Path(workspace) / _SKIP_SENTINEL_NAME
    sentinel.touch(exist_ok=True)


def clear_migration_skip(workspace) -> None:
    """Remove the skip sentinel (so the prompt fires again).  Idempotent."""
    sentinel = Path(workspace) / _SKIP_SENTINEL_NAME
    if sentinel.exists():
        sentinel.unlink()


__all__ = [
    # Data classes
    "FileMove",
    "CellposeLift",
    "MigrationPlan",
    "ExecutionResult",
    # Planning
    "plan_migration",
    # Backup
    "create_backup",
    # Execute / rollback
    "execute_migration",
    "migrate_state_json_content",
    "rollback_migration",
    # Predicates + skip-flag
    "is_migration_needed",
    "is_migration_skipped",
    "mark_migration_skipped",
    "clear_migration_skip",
]

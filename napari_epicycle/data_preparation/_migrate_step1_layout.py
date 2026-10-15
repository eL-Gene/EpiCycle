"""Step 1 layout migration — legacy flat patches → artifacts/step1/patches/.

Pure-Python (no Qt). Mirrors the Step 2/3 migration tooling shape:
``plan_migration`` (dry-run) → ``create_backup`` → ``execute_migration`` →
``rollback_migration``. Data-safety properties (matching Step 2/3):

* the backup is created (complete) BEFORE any file is moved;
* the backup dir gets a UNIQUE name (``tempfile.mkdtemp``) so it can never
  collide with / delete a user directory;
* ``execute_migration`` re-validates each move at execute time (never
  clobbers an existing target, never crashes on an already-moved source);
* ``execute_migration`` AUTO-ROLLS-BACK on any failure;
* idempotent (a second run finds nothing to move);
* corrupt-sidecar tolerant; never overwrites (conflicts are reported).

A *legacy patch* is a workspace-root ``{stem}.tif(f)`` with a sibling
``{stem}_metadata.json`` sidecar that looks patch-like (or is unreadable —
tolerated). Migration moves the pair into ``artifacts/step1/patches/`` and
builds the ``step1_state.json`` registry from the sidecars (provenance
``source_id`` is unknown for legacy patches → empty).

Also exposes the skip-persistence used by the auto-modal trigger
(``is_migration_needed`` / ``is_migration_skipped`` / ``mark_migration_skipped``
/ ``clear_migration_skipped``).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from ._paths import step1_patches_dir, step1_state_path
from ._state import load_state, register_patch, save_state

_SIDECAR_SUFFIX = "_metadata.json"
_BACKUP_PREFIX = ".step1_migration_backup_"
_SKIP_MARKER = ".step1_migration_skip"
_PATCH_SIDECAR_KEYS = ("patch_name", "bounding_box", "channel_names")


# ---------------------------------------------------------------------------
# Plan model
# ---------------------------------------------------------------------------
@dataclass
class FileMove:
    src: Path
    dst: Path


@dataclass
class PatchMigration:
    patch_name: str
    tiff: FileMove
    sidecar: FileMove
    sidecar_readable: bool


@dataclass
class MigrationPlan:
    workspace: Path
    patches: List[PatchMigration] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)  # ambiguous / target exists → skipped

    @property
    def is_empty(self) -> bool:
        return not self.patches


@dataclass
class ExecutionResult:
    moved: int
    registered: int
    conflicts: List[str]            # detected at plan time (ambiguous / target exists)
    skipped: List[str]              # skipped at execute time (dst appeared / src vanished)
    backup_dir: Optional[Path]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _sidecar_readable(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (OSError, ValueError):
        return False


def _looks_like_patch_sidecar(path: Path) -> bool:
    """A readable sidecar must carry at least one patch field; an UNREADABLE
    sidecar is tolerated (treated as a patch) so corrupt sidecars still migrate."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True  # corrupt/unreadable → tolerate
    return isinstance(data, dict) and any(k in data for k in _PATCH_SIDECAR_KEYS)


def _iter_legacy_pairs(root: Path) -> Iterator[Tuple[str, Path, List[Path]]]:
    """Yield (stem, sidecar_path, [tiff_paths]) for legacy root patch candidates."""
    if not root.is_dir():
        return
    for entry in sorted(os.listdir(root)):
        if not entry.endswith(_SIDECAR_SUFFIX):
            continue
        stem = entry[: -len(_SIDECAR_SUFFIX)]
        if not stem:
            continue  # a bare "_metadata.json" (empty stem)
        sidecar = root / entry
        if not sidecar.is_file():
            continue
        if not _looks_like_patch_sidecar(sidecar):
            continue  # readable but not patch-like → not our file
        tiffs = [
            root / f"{stem}{ext}"
            for ext in (".tiff", ".tif")
            if (root / f"{stem}{ext}").is_file()
        ]
        if not tiffs:
            continue  # orphan sidecar (no TIFF)
        yield stem, sidecar, tiffs


def plan_migration(workspace) -> MigrationPlan:
    """Dry-run: enumerate legacy patches to move. No filesystem mutation."""
    root = Path(workspace)
    patches_dir = step1_patches_dir(workspace)
    plan = MigrationPlan(workspace=root)
    for stem, sidecar, tiffs in _iter_legacy_pairs(root):
        if len(tiffs) > 1:
            plan.conflicts.append(stem)  # ambiguous .tif AND .tiff — don't guess
            continue
        tiff = tiffs[0]
        dst_tiff = patches_dir / tiff.name
        dst_sidecar = patches_dir / sidecar.name
        if dst_tiff.exists() or dst_sidecar.exists():
            plan.conflicts.append(stem)  # never overwrite
            continue
        plan.patches.append(
            PatchMigration(
                patch_name=stem,
                tiff=FileMove(tiff, dst_tiff),
                sidecar=FileMove(sidecar, dst_sidecar),
                sidecar_readable=_sidecar_readable(sidecar),
            )
        )
    return plan


def is_migration_needed(workspace) -> bool:
    """True if any *movable* legacy root patches are present.

    Note: a pure-conflict state (legacy files present but all blocked by an
    existing target) returns False on purpose — re-offering a migration that
    can only no-op would nag the user.  Conflicts are surfaced via
    ``plan_migration(...).conflicts`` / ``ExecutionResult.conflicts`` instead.
    """
    return not plan_migration(workspace).is_empty


def should_offer_migration(workspace) -> bool:
    """True iff the auto-modal should offer a migration: needed AND not skipped.

    Pure decision used by the Step 1 load hook + the dialog entry point.
    """
    return is_migration_needed(workspace) and not is_migration_skipped(workspace)


# ---------------------------------------------------------------------------
# Skip persistence (for the auto-modal "Skip forever")
# ---------------------------------------------------------------------------
def _skip_marker_path(workspace) -> Path:
    return Path(workspace) / _SKIP_MARKER


def is_migration_skipped(workspace) -> bool:
    return _skip_marker_path(workspace).exists()


def mark_migration_skipped(workspace) -> None:
    Path(workspace).mkdir(parents=True, exist_ok=True)
    _skip_marker_path(workspace).write_text("skip", encoding="utf-8")


def clear_migration_skipped(workspace) -> None:
    try:
        _skip_marker_path(workspace).unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Backup / execute / rollback
# ---------------------------------------------------------------------------
def create_backup(plan: MigrationPlan) -> Path:
    """Copy the to-be-moved originals + any pre-existing state into a UNIQUE
    backup dir (created complete before any move). Cleans itself up on failure."""
    base = plan.workspace
    base.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=_BACKUP_PREFIX, dir=str(base)))
    try:
        files_dir = backup / "root_files"
        files_dir.mkdir()
        for pm in plan.patches:
            shutil.copy2(pm.tiff.src, files_dir / pm.tiff.src.name)
            shutil.copy2(pm.sidecar.src, files_dir / pm.sidecar.src.name)
        state_path = step1_state_path(plan.workspace)
        if state_path.exists():
            shutil.copy2(state_path, backup / "step1_state.json")
    except BaseException:
        shutil.rmtree(backup, ignore_errors=True)  # never leave a partial backup
        raise
    return backup


def execute_migration(
    plan: MigrationPlan,
    *,
    make_backup: bool = True,
    timestamp: Optional[str] = None,
    auto_rollback: bool = True,
) -> ExecutionResult:
    """Move legacy patches into artifacts/step1/patches/ and build the registry.

    Re-validates each move at execute time (never clobbers an existing target,
    never crashes on an already-moved source). On ANY failure, auto-rolls-back
    from the backup and re-raises. Idempotent and corrupt-sidecar tolerant.
    """
    if plan.is_empty:
        return ExecutionResult(0, 0, list(plan.conflicts), [], None)

    backup_dir = create_backup(plan) if make_backup else None
    try:
        patches_dir = step1_patches_dir(plan.workspace)
        patches_dir.mkdir(parents=True, exist_ok=True)

        migrated: List[PatchMigration] = []
        skipped: List[str] = []
        for pm in plan.patches:
            # Re-validate (TOCTOU / stale plan / idempotency): don't clobber a
            # target that appeared, don't crash on a source that vanished.
            if pm.tiff.dst.exists() or pm.sidecar.dst.exists():
                skipped.append(pm.patch_name)
                continue
            if not pm.tiff.src.exists() or not pm.sidecar.src.exists():
                skipped.append(pm.patch_name)
                continue
            shutil.move(str(pm.tiff.src), str(pm.tiff.dst))
            shutil.move(str(pm.sidecar.src), str(pm.sidecar.dst))
            migrated.append(pm)

        state = load_state(plan.workspace)
        for pm in migrated:
            register_patch(
                state,
                patch_name=pm.patch_name,
                relative_path=f"artifacts/step1/patches/{pm.tiff.dst.name}",
                sidecar_path=f"artifacts/step1/patches/{pm.sidecar.dst.name}",
                source_id="",  # provenance unknown for legacy patches
                timestamp=timestamp,
            )
        save_state(plan.workspace, state)
    except BaseException:
        if backup_dir is not None and auto_rollback:
            try:
                rollback_migration(plan, backup_dir)
            except Exception:
                pass
        raise

    return ExecutionResult(
        moved=len(migrated), registered=len(migrated),
        conflicts=list(plan.conflicts), skipped=skipped, backup_dir=backup_dir,
    )


def rollback_migration(plan: MigrationPlan, backup_dir, *, remove_backup: bool = True) -> None:
    """Undo a migration: restore root originals, remove migrated copies + state.

    Call ONCE. Removing the backup at the end (default) makes a second call a
    safe no-op and prevents clobbering files the user may have since edited.
    """
    backup_dir = Path(backup_dir)
    files_dir = backup_dir / "root_files"
    root = plan.workspace

    for pm in plan.patches:
        for dst in (pm.tiff.dst, pm.sidecar.dst):
            if dst.exists():
                dst.unlink()
        for name in (pm.tiff.src.name, pm.sidecar.src.name):
            b = files_dir / name
            if b.exists():
                shutil.copy2(b, root / name)

    state_backup = backup_dir / "step1_state.json"
    state_path = step1_state_path(root)
    if state_backup.exists():
        shutil.copy2(state_backup, state_path)
    elif state_path.exists():
        state_path.unlink()

    if remove_backup:
        shutil.rmtree(backup_dir, ignore_errors=True)

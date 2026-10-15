"""Centralized Step 1 (Data Preparation) workspace path helpers.

**Source of truth** for where Step 1 artifacts live on disk under a
workspace.  Mirrors the Step 3 ``fusion_reconstruction/_paths.py`` pattern.

Layout (new, 2026-06-29+)::

    {workspace}/
    ├── step1_state.json                  # registry + provenance + settings
    └── artifacts/
        └── step1/
            └── patches/
                ├── {patch}.tiff
                └── {patch}_metadata.json # sidecar (schema frozen)

Legacy layout (pre-restructure): patches written flat at the workspace
root.  Migration moves them into ``artifacts/step1/patches/`` — see
``_migrate_step1_layout.py``.

Helper functions are PURE — they answer "where should X live?" and never
touch the filesystem.  ``mkdir`` is the caller's responsibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

# Filename / directory constants
STEP1_STATE_FILENAME = "step1_state.json"


def step1_artifacts_dir(workspace: PathLike) -> Path:
    """Return ``{workspace}/artifacts/step1`` (the Step 1 artifact root)."""
    return Path(workspace) / "artifacts" / "step1"


def step1_patches_dir(workspace: PathLike) -> Path:
    """Return ``{workspace}/artifacts/step1/patches`` (extracted-patch home)."""
    return step1_artifacts_dir(workspace) / "patches"


def step1_state_path(workspace: PathLike) -> Path:
    """Return ``{workspace}/step1_state.json`` (the Step 1 state file)."""
    return Path(workspace) / STEP1_STATE_FILENAME


def step1_state_backup_path(workspace: PathLike) -> Path:
    """Return ``{workspace}/artifacts/step1/state_backup/step1_state.json.bak``.

    The rotated predecessor of the state file (B4).
    It used to sit beside `step1_state.json` in the workspace ROOT; nothing
    ever read it, so an old root ``.bak`` is simply ignored -- not read, not
    migrated, not deleted.
    """
    return (step1_artifacts_dir(workspace) / "state_backup"
            / (STEP1_STATE_FILENAME + ".bak"))


def step1_exports_dir(workspace: PathLike) -> Path:
    """Return ``{workspace}/artifacts/step1/exports`` (snapshot exports).

    Mirrors Step 4's ``artifacts/step4/exports``.  Created only when the user
    actually saves something (B4) -- opening and cancelling the dialog leaves
    no folder behind.
    """
    return step1_artifacts_dir(workspace) / "exports"

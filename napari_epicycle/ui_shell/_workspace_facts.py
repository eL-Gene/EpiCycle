"""Workspace facts read from DISK, plus the pure parts of the shared-workspace
banner.

Brick B1 of the shared-workspace banner.

Why this module exists: the shell used to answer "does
this workspace have trained rounds?" by reaching into Step 2's *live*
controller (`current_round`).  That attribute is empty until Step 2 has loaded
the workspace in this session, so a workspace with three trained lateral
rounds still showed "No completed HITL training round detected yet…".  Facts
about a workspace come from the workspace, not from session state.

Everything here is PURE — no Qt, no napari, no controller import.  The shell
must not depend on a step package (`hitl_training_annotation`), so the *rule*
from `_run_management._count_rounds` is reproduced here rather than imported:
a round counts only once it has a ``model/`` subdir, and the scan is
dual-path (new flat ``{bucket}/round_NN/`` and legacy split
``{bucket}/rounds/round_NN/``).

Nothing in this module raises: every reader is wrapped and degrades to a
neutral answer (zeros / "" / no strip).  These functions are called from
`_select_step`, and a step transition must never die on a filesystem hiccup.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

__all__ = [
    "trained_rounds",
    "step2_layout_hint",
    "step2_has_runs",
    "step3_pre_v3",
    "MSG_NO_TRAINED_MODEL",
    "MSG_STEP2_LEGACY_LAYOUT",
    "MSG_STEP2_NO_RUN",
    "MSG_STEP3_PRE_V3",
    "STATE_LOCAL",
    "patch_count",
    "clear_cache",
    "load_last_workspace",
    "save_last_workspace",
    "banner_state",
    "banner_strips",
    "advisory_color",
    "SHARED_ACCENT",
    "SHARED_ACCENT_KEY",
    "STATE_NONE",
    "STATE_SHARED",
    "STATE_DIVERGED",
    "STATE_STALE",
]

_BUCKETS = ("lateral", "axial")

#: {(normcase(workspace), mtime_ns of step2_state.json): {bucket: n}}
_ROUNDS_CACHE: Dict[Tuple[str, int], Dict[str, int]] = {}
#: {(normcase(workspace), mtime_ns of the patches dir): count}
_PATCHES_CACHE: Dict[Tuple[str, int], int] = {}
#: The three B2 banner facts, each keyed on the mtime of what it reads.
#: Uncached, they cost a directory walk (or a `json.loads`) on EVERY 3 s
#: sidebar poll — and, since the advisory follows the step's OWN path, that
#: could be a walk over a network share.
_LAYOUT_CACHE: Dict[Tuple[str, int], str] = {}
_RUNS_CACHE: Dict[Tuple[str, int], bool] = {}
_PRE_V3_CACHE: Dict[Tuple[str, int], bool] = {}

LAST_WORKSPACE_FILENAME = "last_workspace.json"
LAST_WORKSPACE_SCHEMA = 1


def clear_cache() -> None:
    """Drop every memoized workspace fact (tests + workspace switches)."""
    _ROUNDS_CACHE.clear()
    _PATCHES_CACHE.clear()
    _LAYOUT_CACHE.clear()
    _RUNS_CACHE.clear()
    _PRE_V3_CACHE.clear()


def _key(workspace: str) -> str:
    try:
        return os.path.normcase(os.path.normpath(workspace))
    except Exception:      # noqa: BLE001 — a key is never worth an exception
        return str(workspace)


def _mtime_ns(path: Path) -> int:
    """``st_mtime_ns`` or 0 — a missing file is a valid cache key."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Step 2 — trained rounds
# ---------------------------------------------------------------------------

def _count_bucket_rounds(bucket_root: Path) -> int:
    """Rounds under one bucket that have actually TRAINED (have ``model/``).

    Dual-path: the new flat layout ``{bucket}/round_NN/`` and the legacy
    split layout ``{bucket}/rounds/round_NN/``.  A round present in both is
    counted once (union of round names) — mirrors
    ``_run_management._count_rounds``.
    """
    seen: set = set()
    for root in (bucket_root, bucket_root / "rounds"):
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for p in entries:
            try:
                if (p.is_dir() and p.name.startswith("round_")
                        and (p / "model").is_dir()):
                    seen.add(p.name)
            except OSError:
                continue
    return len(seen)


_LEGACY_LATERAL_HINTS = ("spatial", "lateral", "_xy")
_LEGACY_AXIAL_HINTS = ("axial", "_xz", "_yz")


def _count_legacy_root_rounds(workspace: Path) -> Dict[str, int]:
    """Pre-Phase-6c layout: models at ``{ws}/models/round_NN/[models/]<file>``.

    Mirrors `_run_management._scan_legacy_step2_models`: a round counts for a
    bucket when it holds at least one model FILE whose name carries that
    bucket's keyword (Cellpose preserves the naming convention); dotfiles,
    editor backups and the `.npy` / `.json` companions are skipped, and a
    file with no bucket keyword is not attributed to either.
    """
    counts = {"lateral": 0, "axial": 0}
    legacy_root = workspace / "models"
    try:
        round_dirs = sorted(legacy_root.iterdir())
    except OSError:
        return counts
    for round_dir in round_dirs:
        try:
            if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
                continue
            scan_dirs = [round_dir]
            nested = round_dir / "models"
            if nested.is_dir():
                scan_dirs.append(nested)
            buckets: set = set()
            for scan_dir in scan_dirs:
                for f in scan_dir.iterdir():
                    if not f.is_file():
                        continue
                    name = f.name
                    if name.startswith((".", "~")):
                        continue
                    if f.suffix in {".npy", ".json"} or name.endswith(
                            (".tmp", ".bak")):
                        continue
                    low = name.lower()
                    if any(h in low for h in _LEGACY_LATERAL_HINTS):
                        buckets.add("lateral")
                    elif any(h in low for h in _LEGACY_AXIAL_HINTS):
                        buckets.add("axial")
            for b in buckets:
                counts[b] += 1
        except OSError:
            continue
    return counts


def trained_rounds(workspace: str) -> Dict[str, int]:
    """``{'lateral': n, 'axial': m}`` — TRAINED rounds across all Step 2 runs.

    Read from ``{ws}/artifacts/step2/runs/*/{bucket}/`` on disk and summed
    over every run.  Cached per ``(workspace, step2_state.json mtime)``; a
    missing state file means mtime 0 and the scan still runs once.  Never
    raises: any unreadable path yields zeros.
    """
    zero = {"lateral": 0, "axial": 0}
    if not workspace:
        return dict(zero)
    ws = Path(workspace)
    # Keyed on BOTH state filenames: a legacy (pre-v3) workspace has only
    # `hitl_state.json`, so keying on `step2_state.json` alone would freeze
    # its count behind a constant mtime of 0.
    cache_key = (
        _key(workspace),
        max(_mtime_ns(ws / "step2_state.json"),
            _mtime_ns(ws / "hitl_state.json")),
    )
    hit = _ROUNDS_CACHE.get(cache_key)
    if hit is not None:
        return dict(hit)

    counts = dict(zero)
    runs_root = ws / "artifacts" / "step2" / "runs"
    try:
        run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    except OSError:
        run_dirs = []
    for run_dir in run_dirs:
        for bucket in _BUCKETS:
            try:
                counts[bucket] += _count_bucket_rounds(run_dir / bucket)
            except Exception:      # noqa: BLE001 — a fact is never fatal
                continue

    # Pre-Phase-6c workspaces keep their trained models in the workspace root.
    try:
        legacy = _count_legacy_root_rounds(ws)
        counts["lateral"] += legacy["lateral"]
        counts["axial"] += legacy["axial"]
    except Exception:          # noqa: BLE001
        pass

    _ROUNDS_CACHE[cache_key] = dict(counts)
    return counts


# ---------------------------------------------------------------------------
# Step 2 / Step 3 — the second-line status the banner shows on adopt (B2)
# ---------------------------------------------------------------------------

#: Adopt never runs a migration modal, so a workspace on the old
#: folder layout is told, not asked.
MSG_STEP2_LEGACY_LAYOUT = (
    "This project uses the old folder layout. Press Browse and pick this "
    "same folder to update it.")
#: Adopt also skips Step 2's first-run modal, so the invitation moves here.
MSG_STEP2_NO_RUN = "No run yet — click New Run to start training."
#: Step 3's state file is only UPGRADED by an explicit Browse: adopt
#: reads it and writes nothing.
MSG_STEP3_PRE_V3 = (
    "This project uses the old folder layout. Press Browse and pick this "
    "same folder to update it.")


def step2_layout_hint(workspace: str) -> str:
    """`MSG_STEP2_LEGACY_LAYOUT` when Step 2's folders are pre-restructure.

    Reproduces the LEGACY / MIXED half of
    `hitl_training_annotation._paths.detect_layout_version` — the shell must
    not import a step package.  Legacy markers are ``training_data/``
    and ``models/round_*/``; the new marker is
    ``artifacts/step2/runs/``.  A workspace carrying BOTH is partially
    migrated and still needs the Browse, so it gets the hint too.
    """
    if not workspace or not _is_local_drive_path(workspace):
        return ""
    ws = Path(workspace)
    # Keyed on the workspace root AND `models/`: creating or removing
    # `training_data/` bumps the root, while adding a `models/round_NN`
    # bumps only `models/` itself.
    cache_key = (_key(workspace),
                 max(_mtime_ns(ws), _mtime_ns(ws / "models")))
    hit = _LAYOUT_CACHE.get(cache_key)
    if hit is not None:
        return hit
    try:
        has_legacy = (ws / "training_data").is_dir()
        if not has_legacy:
            legacy_models = ws / "models"
            if legacy_models.is_dir():
                for d in legacy_models.iterdir():
                    if d.is_dir() and d.name.startswith("round_"):
                        has_legacy = True
                        break
    except OSError:
        return ""
    except Exception:          # noqa: BLE001 — a fact is never fatal
        return ""
    answer = MSG_STEP2_LEGACY_LAYOUT if has_legacy else ""
    _LAYOUT_CACHE[cache_key] = answer
    return answer


def step2_has_runs(workspace: str) -> bool:
    """True when ``artifacts/step2/runs/`` holds at least one run directory."""
    if not workspace or not _is_local_drive_path(workspace):
        return False
    runs = Path(workspace) / "artifacts" / "step2" / "runs"
    # The runs dir's mtime bumps when a run is created or deleted, which is
    # exactly when this answer can change.
    cache_key = (_key(workspace), _mtime_ns(runs))
    hit = _RUNS_CACHE.get(cache_key)
    if hit is not None:
        return hit
    try:
        answer = any(p.is_dir() for p in runs.iterdir())
    except OSError:
        answer = False
    except Exception:          # noqa: BLE001
        answer = False
    _RUNS_CACHE[cache_key] = answer
    return answer


def step3_pre_v3(workspace: str) -> bool:
    """True when ``step3_state.json`` exists and predates the v3 envelope.

    Only a real, parseable, pre-v3 file counts: a missing or corrupt file
    would be REPLACED by a fresh v3 state on the next Browse anyway, so
    there is nothing for the user to protect and no line to show.
    """
    if not workspace or not _is_local_drive_path(workspace):
        return False
    path = Path(workspace) / "step3_state.json"
    cache_key = (_key(workspace), _mtime_ns(path))
    hit = _PRE_V3_CACHE.get(cache_key)
    if hit is not None:
        return hit
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:          # noqa: BLE001 — missing / corrupt / unreadable
        data = None
    answer = (isinstance(data, dict)
              and data.get("schema_version") in (None, 1, 2))
    _PRE_V3_CACHE[cache_key] = answer
    return answer


# ---------------------------------------------------------------------------
# Step 1 — extracted patches
# ---------------------------------------------------------------------------

def patch_count(workspace: str, allow_network: bool = False) -> int:
    """Number of extracted patch TIFFs in ``{ws}/artifacts/step1/patches/``.

    Cached on the patches dir's own mtime (which bumps whenever a patch is
    written or deleted), so the 3 s sidebar poll costs one ``stat``.

    ``allow_network``: a `stat`/`iterdir` on a disconnected
    network path can stall for seconds on Windows, so a non-drive-letter
    path is NOT walked unless the caller knows it is reachable — the shell
    passes True only once the user has actually browsed to it.
    """
    if not workspace:
        return 0
    if not allow_network and not _is_local_drive_path(workspace):
        return 0
    patches = Path(workspace) / "artifacts" / "step1" / "patches"
    cache_key = (_key(workspace), _mtime_ns(patches))
    hit = _PATCHES_CACHE.get(cache_key)
    if hit is not None:
        return hit
    total = 0
    try:
        for p in patches.iterdir():
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
                total += 1
    except OSError:
        total = 0
    except Exception:          # noqa: BLE001
        total = 0
    _PATCHES_CACHE[cache_key] = total
    return total


# ---------------------------------------------------------------------------
# The remembered workspace — a SUGGESTION, never an action
# ---------------------------------------------------------------------------

def _epicycle_dir() -> Path:
    """``~/.epicycle`` — the crash logger's directory."""
    return Path.home() / ".epicycle"


def _is_local_drive_path(path: str) -> bool:
    """True for ``C:\\…`` style paths only.

    Invariant I13: ``os.path.isdir`` on a disconnected network drive can
    stall for seconds on Windows, so a UNC (``\\\\server\\share``) or any
    non-drive-letter path is accepted unvalidated and left for Browse to
    resolve.
    """
    return bool(path) and re.match(r"^[A-Za-z]:[\\/]", path) is not None


def load_last_workspace() -> str:
    """The remembered workspace path, or ``""``.

    Corrupt, missing, wrong-shaped or (for a local drive letter) deleted →
    ``""``, silently.  Performs no I/O inside the workspace beyond the single
    guarded ``isdir``.
    """
    try:
        raw = (_epicycle_dir() / LAST_WORKSPACE_FILENAME).read_text(
            encoding="utf-8")
        payload = json.loads(raw)
    except Exception:          # noqa: BLE001 — a suggestion never fails loudly
        return ""
    if not isinstance(payload, dict):
        return ""
    path = payload.get("path") or ""
    if not isinstance(path, str) or not path.strip():
        return ""
    path = path.strip()
    if _is_local_drive_path(path):
        try:
            if not os.path.isdir(path):
                return ""
        except OSError:
            return ""
    return path


def save_last_workspace(path: str) -> None:
    """Remember ``path`` for the next session.  Best-effort, never raises."""
    if not path:
        return
    try:
        from packages.epicycle_io.state_store import atomic_write_json
        atomic_write_json(
            _epicycle_dir() / LAST_WORKSPACE_FILENAME,
            {
                "schema_version": LAST_WORKSPACE_SCHEMA,
                "path": path,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as exc:   # noqa: BLE001
        print(f"[WORKSPACE] could not remember the workspace: {exc}")


# ---------------------------------------------------------------------------
# The banner state machine — pure text + colour, rendered by the shell
# ---------------------------------------------------------------------------

#: THE colour of "this step is on the shared workspace" — the banner's green
#: strip and every neutral advisory read this one name, so the whole shared-
#: workspace accent is a ONE-LINE change.
#:
#: Today: Apply-green `#6abf69` (`theme.py:373`).  Teals that already exist in
#: the codebase, if the user prefers one:
#:   * `#00d4aa` — the sidebar's own CPU resource-bar fill
#:                 (`workflow_shell.py::_build_ui`), the closest match to the
#:                 "CPU-bar teal" and the only place that hex appears;
#:   * `#3fb6b6` — the Step 1 capture viewer's accent
#:                 (`data_preparation/_capture_widget.py`);
#:   * `#4ecdc4` — `theme.PRIMARY_BTN_STYLE_CYAN` (`theme.py:627`);
#:   * `#5ba4cf` — `theme.py`'s "teal (Setup)", which is really a blue.
#: Switching means editing this line and nothing else.
SHARED_ACCENT = "#00d4aa"

#: The colour KEY the banner uses for that accent.  Separate from the hex so
#: `advisory_color` and `banner_strips` name the ROLE, not a colour.
SHARED_ACCENT_KEY = "green"

STATE_NONE = "none"          # 1 — no workspace anywhere        (red)
STATE_SHARED = "shared"      # 2 — this step is on the shared ws (green)
STATE_DIVERGED = "diverged"  # 3 — this step was Browsed elsewhere (amber)
#: 2b — this step has a folder but no workspace has been SHARED yet (green,
#: without the "(shared across steps)" qualifier).  A box filled by a step's
#: own state restore is not a shared workspace; only a Browse shares one.
STATE_LOCAL = "local"
#: 3b — this step ADOPTED a path and the shared workspace has since moved
#: without it following (amber).  The banner is drawn for the ACTIVE step, and
#: the active step is adopted synchronously by `set_workspace`, so this means
#: its controller DECLINED the re-adopt — Step 1 does that while one of its
#: jobs is running (B5).  Not `shared` (that folder is not the shared one) and
#: not `diverged` (the user did not browse it; it moves on its own once the
#: job ends).
STATE_STALE = "stale"

_MSG_STEP1_NONE = "Choose a workspace folder to begin."
_MSG_OTHER_NONE = "Set a workspace in Project Setup before running this step."
_MSG_MISSING = ("That workspace folder is no longer where it was. "
                "Use Browse to find it.")
#: The Step 3/4 round advisory, in the user's language (no "HITL", no
#: "artifacts") — it says what is missing and what to do about it.
MSG_NO_TRAINED_MODEL = (
    "No trained 2D model in this workspace yet. You can continue, but there "
    "is nothing to reconstruct with until Step 2 trains a round.")


#: Advisories that are an INVITATION, not a warning.  A brand-new project has
#: nothing wrong with it; painting "No run yet" the same amber as "old folder
#: layout" and "no trained 2D model" made three quite different situations
#: look equally alarming (second B2 audit).
_NEUTRAL_ADVISORIES = frozenset({MSG_STEP2_NO_RUN})


def advisory_color(advisory: str) -> str:
    """The colour key for an advisory strip — amber unless it is neutral."""
    return SHARED_ACCENT_KEY if advisory in _NEUTRAL_ADVISORIES else "amber"


def banner_state(shared: str, step_path: str, browsed: bool = False) -> str:
    """Which of the three states this step is in.

    ``browsed`` says the USER chose this step's path with its own Browse
    button.  Under the symmetric rule the most recent Browse in
    ANY step sets the shared workspace, so a step is amber only when it was
    browsed somewhere else *earlier* and the shared workspace has since moved
    on.  A path a step merely INHERITED is never amber: it re-adopts the new
    shared workspace the next time the step is entered.

    An empty ``step_path`` means the step simply follows the shared
    workspace (state 2).
    """
    if not shared and not step_path:
        return STATE_NONE
    if not shared:
        return STATE_LOCAL
    if not step_path:
        return STATE_SHARED
    if _key(shared) == _key(step_path):
        return STATE_SHARED
    # Different from the shared workspace.  Which kind depends on how this
    # step got its path: the user browsed it (diverged, and it stays until
    # they browse again), or it adopted and the re-adopt was declined
    # (stale — it follows on its own once whatever declined it is done).
    return STATE_DIVERGED if browsed else STATE_STALE


def banner_strips(
    step_id: str,
    shared: str,
    step_path: str,
    advisory: str = "",
    missing: bool = False,
    browsed: bool = False,
) -> List[Tuple[str, str, str]]:
    """The banner as a list of strips, top first.

    Returns ``[(text, colour_key, path_for_elision), …]`` where ``colour_key``
    is one of ``"red" | "green" | "amber"`` and ``path_for_elision`` is the
    substring of ``text`` that may be middle-elided (``""`` when nothing in
    the strip is a path).

    Stacking: the workspace line first,
    the advisory below it, each keeping its own colour.  Precedence is
    state 1 > advisory > state 3 > state 2 — state 1 is the only state that
    *replaces* everything, because with no workspace there is nothing to
    advise about.

    ``browsed`` — see `banner_state`: amber requires that the user browsed
    THIS step somewhere else, not merely that its path is stale.
    """
    state = banner_state(shared, step_path, browsed=browsed)
    if state == STATE_NONE:
        msg = _MSG_STEP1_NONE if step_id == "project_setup" else _MSG_OTHER_NONE
        return [(msg, "red", "")]

    effective = step_path or shared
    strips: List[Tuple[str, str, str]] = []
    if state == STATE_LOCAL and not missing:
        # No qualifier: nothing is shared yet, this is just where this step
        # is pointed.
        strips.append((f"Workspace: {effective}", SHARED_ACCENT_KEY,
                       effective))
        if advisory:
            strips.append((advisory, advisory_color(advisory), ""))
        return strips
    if missing:
        strips.append((_MSG_MISSING, "red", ""))
    elif state == STATE_DIVERGED:
        # The mockup's wording.  The shared path is NOT inlined: with two
        # full paths on one line the step's own folder collapsed to a few
        # characters and the tail ran off the panel.  The shell puts both
        # paths in the tooltip instead.
        strips.append((
            f"Workspace: {effective}  (different from the shared workspace)",
            "amber", effective))
    elif state == STATE_STALE:
        # It has not moved to the shared workspace YET, and saying "(shared
        # across steps)" here would name a folder that is not shared.  Plain
        # words about why, since the user cannot act on it beyond waiting.
        strips.append((
            f"Workspace: {effective}  "
            f"(not switched yet — a job is still running here)",
            "amber", effective))
    else:
        strips.append((
            f"Workspace: {effective}  (shared across steps)",
            SHARED_ACCENT_KEY, effective))

    if advisory:
        strips.append((advisory, advisory_color(advisory), ""))
    return strips

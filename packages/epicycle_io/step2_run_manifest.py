"""Step 2 per-run manifest (``run_manifest.json``).

Each Step 2 training run lives under ``artifacts/step2/runs/{run_id}/``
and carries a sidecar ``run_manifest.json`` describing the run's
identity, target, backend, and the source it was bootstrapped from.

This module owns the manifest schema + read/write helpers, plus the
multi-run-discovery scanner used by the Phase 8 UI.  It is intentionally
pure-Python (no Qt, no napari, no controller deps) and lives under
``packages/`` per the layer-separation rule (packages never import the plugin).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------------
# Schema constants
# ----------------------------------------------------------------------------

RUN_MANIFEST_SCHEMA_VERSION = "1.0"
"""Schema version for ``run_manifest.json``.  Bump on incompatible change."""

RUN_MANIFEST_FILENAME = "run_manifest.json"

# Allowed values for ``created_from.mode``.  Mirrors the init-source picker
# cards of the New Run dialog.
VALID_INIT_MODES = frozenset({
    "scratch",         # random-weights init
    "inherited",       # init from another run in this workspace
    "cross_bucket",    # init from the opposite bucket of the same run/round
    "step3_refined",   # init from a Step 3 refinement bundle
    "pretrained",      # init from a Cellpose built-in (cyto3, nuclei, etc.)
    "browse",          # init from an arbitrary file the user picked
    "fork",            # this run forked another (Phase 8d)
})

# Allowed values for ``backend``.  Cellpose is the only one enabled today;
# Omnipose / StarDist / Custom 2D U-Net are reserved.
VALID_BACKENDS = frozenset({
    "cellpose",
    "omnipose",   # reserved, not yet implemented
    "stardist",   # reserved, not yet implemented
    "unet2d",     # reserved, not yet implemented
})

ENABLED_BACKENDS = frozenset({"cellpose"})
"""Subset of VALID_BACKENDS that are wired up today.  UI greys the rest."""


# ----------------------------------------------------------------------------
# Build / write / read
# ----------------------------------------------------------------------------

def build_run_manifest(
    *,
    run_id: str,
    display_name: str,
    target: str = "",
    description: str = "",
    backend: str = "cellpose",
    created_from: Optional[Dict[str, Any]] = None,
    channels_snapshot: Optional[List[str]] = None,
    notes: str = "",
    created_at: Optional[str] = None,
    archived_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a run_manifest dict from explicit fields.

    All validation happens here so the writer can stay a thin dump.

    Args:
        run_id: slug; must match the parent dir name.  Validated via
            ``hitl_training_annotation._paths.validate_run_id`` upstream;
            this module just checks non-empty.
        display_name: human-friendly name shown in the dropdown.
        target: biological target hint (free text, "nuclei", "membrane",
            "actin", "cytoplasm", ...).  Empty allowed.
        description: optional multiline description.
        backend: must be in ``VALID_BACKENDS``; UI today only enables
            "cellpose" (checked elsewhere).
        created_from: dict describing init source.  See ``build_init_spec``.
            If None, defaults to ``{"mode": "scratch"}``.
        channels_snapshot: optional list of channel-name strings captured
            at run creation.
        notes: optional free text.
        created_at: optional ISO timestamp.  If None, populated as
            ``utcnow().isoformat() + "Z"``.
        archived_at: optional ISO timestamp (Phase 8c).  Defaults to None.

    Returns:
        Manifest dict ready to be JSON-dumped.

    Raises:
        ValueError: on empty run_id, empty display_name, invalid backend,
            or invalid created_from mode.
    """
    if not run_id or not isinstance(run_id, str):
        raise ValueError("run_id must be a non-empty string")
    if not display_name or not isinstance(display_name, str):
        raise ValueError("display_name must be a non-empty string")
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"backend {backend!r} not in VALID_BACKENDS={sorted(VALID_BACKENDS)}"
        )

    cf = dict(created_from) if created_from else {"mode": "scratch"}
    if "mode" not in cf:
        raise ValueError("created_from must include 'mode'")
    if cf["mode"] not in VALID_INIT_MODES:
        raise ValueError(
            f"created_from.mode {cf['mode']!r} not in VALID_INIT_MODES="
            f"{sorted(VALID_INIT_MODES)}"
        )
    # Normalize the created_from dict — fill in missing optional fields
    # with None so downstream consumers don't KeyError.
    cf.setdefault("source_run_id", None)
    cf.setdefault("source_bucket", None)
    cf.setdefault("source_round_id", None)
    cf.setdefault("source_step3_bundle", None)
    cf.setdefault("source_step3_bucket", None)
    cf.setdefault("pretrained_name", None)
    cf.setdefault("external_path", None)
    cf.setdefault("external_label", None)

    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "display_name": display_name,
        "target": target or "",
        "description": description or "",
        "backend": backend,
        "created_at": created_at or _utc_now_iso(),
        "created_from": cf,
        "archived_at": archived_at,
        "channels_snapshot": list(channels_snapshot) if channels_snapshot else [],
        "notes": notes or "",
    }


def write_run_manifest(manifest_path: Path | str, manifest: Dict[str, Any]) -> None:
    """Atomically write a run_manifest dict to disk.

    Uses tempfile + os.replace for atomicity (mirror Step 3 pattern).
    Parent directory must already exist; this function does NOT mkdir
    (the caller's create_run flow is responsible for the directory
    structure so a half-created run isn't possible).

    Args:
        manifest_path: target file path (typically
            ``{workspace}/artifacts/step2/runs/{run_id}/run_manifest.json``).
        manifest: dict; should have come from ``build_run_manifest``.

    Raises:
        OSError: on filesystem failures.
        ValueError: on missing required keys in manifest.
    """
    manifest_path = Path(manifest_path)
    parent = manifest_path.parent
    if not parent.is_dir():
        raise OSError(f"parent directory does not exist: {parent}")
    # Quick sanity: required keys must be present
    for required in ("schema_version", "run_id", "display_name", "backend",
                     "created_at", "created_from"):
        if required not in manifest:
            raise ValueError(f"manifest missing required key: {required!r}")

    payload = json.dumps(manifest, indent=2, ensure_ascii=False)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".run_manifest_", suffix=".json.tmp", dir=str(parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, manifest_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_run_manifest(manifest_path: Path | str) -> Optional[Dict[str, Any]]:
    """Read a run_manifest from disk.

    Defensive: returns ``None`` on missing file, unreadable file, or
    invalid JSON.  The caller can distinguish file-not-found from
    other failures by checking the path exists first.

    Args:
        manifest_path: file path.

    Returns:
        Dict on success; ``None`` on any failure.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as _e:
        # Commit 2 audit LOW-1 fix (2026-05-26): the silent-None return
        # was masking corrupt run_manifests in production.  Centralize
        # the diagnostic here so all 4 callers benefit (list_run_summaries,
        # _list_runs_for_ui, training_annotation_controller H-5 read,
        # state-merge code path).  Still returns None for callers that
        # treat that as "missing or corrupt" — adds visibility, no
        # behavior change.
        print(f"[WARN] run_manifest read failed at {manifest_path}: "
              f"{type(_e).__name__}: {_e}")
        return None
    if not isinstance(data, dict):
        print(f"[WARN] run_manifest at {manifest_path} is not a "
              f"JSON object (got {type(data).__name__})")
        return None
    return data


# ----------------------------------------------------------------------------
# Init-spec builders (one per mode, for ergonomic call sites)
# ----------------------------------------------------------------------------

def init_spec_scratch() -> Dict[str, Any]:
    """Init from random weights."""
    return {"mode": "scratch"}


def init_spec_inherited(
    *, source_run_id: str, source_bucket: str, source_round_id: str
) -> Dict[str, Any]:
    """Init from another Step 2 run's trained model.

    Args:
        source_run_id: e.g., "default"
        source_bucket: "lateral" or "axial"
        source_round_id: e.g., "round_03"
    """
    if not source_run_id:
        raise ValueError("source_run_id required")
    if source_bucket not in {"lateral", "axial"}:
        raise ValueError(f"source_bucket must be lateral|axial, got {source_bucket!r}")
    if not source_round_id:
        raise ValueError("source_round_id required")
    return {
        "mode": "inherited",
        "source_run_id": source_run_id,
        "source_bucket": source_bucket,
        "source_round_id": source_round_id,
    }


def init_spec_cross_bucket(
    *, source_bucket: str, source_round_id: str
) -> Dict[str, Any]:
    """Init the new run's one bucket from the same run's opposite bucket.

    Used when Mode=Both and the user wants lateral to init from this
    same round's axial weights (or vice versa).  Phase 8b/c territory.
    """
    if source_bucket not in {"lateral", "axial"}:
        raise ValueError(f"source_bucket must be lateral|axial, got {source_bucket!r}")
    if not source_round_id:
        raise ValueError("source_round_id required")
    return {
        "mode": "cross_bucket",
        "source_bucket": source_bucket,
        "source_round_id": source_round_id,
    }


def init_spec_step3_refined(
    *, source_step3_bundle: str, source_step3_bucket: str
) -> Dict[str, Any]:
    """Init from a Step 3 refinement bundle in this workspace."""
    if not source_step3_bundle:
        raise ValueError("source_step3_bundle required")
    if source_step3_bucket not in {"lateral", "axial"}:
        raise ValueError(
            f"source_step3_bucket must be lateral|axial, got {source_step3_bucket!r}"
        )
    return {
        "mode": "step3_refined",
        "source_step3_bundle": source_step3_bundle,
        "source_step3_bucket": source_step3_bucket,
    }


def init_spec_pretrained(*, pretrained_name: str) -> Dict[str, Any]:
    """Init from a Cellpose built-in model (cyto3, nuclei, etc.)."""
    if not pretrained_name:
        raise ValueError("pretrained_name required")
    return {"mode": "pretrained", "pretrained_name": pretrained_name}


def init_spec_browse(*, external_path: str, external_label: str = "") -> Dict[str, Any]:
    """Init from an arbitrary user-picked file."""
    if not external_path:
        raise ValueError("external_path required")
    return {
        "mode": "browse",
        "external_path": external_path,
        "external_label": external_label or Path(external_path).name,
    }


def init_spec_fork(
    *, source_run_id: str, copied_pair_count: int = 0
) -> Dict[str, Any]:
    """Phase 8d: this run was forked from another, with N pairs copied."""
    if not source_run_id:
        raise ValueError("source_run_id required")
    return {
        "mode": "fork",
        "source_run_id": source_run_id,
        "copied_pair_count": int(copied_pair_count),
    }


# ----------------------------------------------------------------------------
# Multi-run discovery (scanner)
# ----------------------------------------------------------------------------

def list_run_ids(step2_runs_root: Path | str) -> List[str]:
    """Return sorted list of run IDs (sub-directory names) under runs/.

    Pure filesystem scan — does NOT validate that each dir contains a
    valid manifest.  Use ``list_runs_with_manifests`` for that.

    Args:
        step2_runs_root: path to ``{workspace}/artifacts/step2/runs/``.

    Returns:
        Sorted list of slugs (str).  Empty list if root missing or empty.
    """
    root = Path(step2_runs_root)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def list_runs_with_manifests(
    step2_runs_root: Path | str,
) -> List[Dict[str, Any]]:
    """Scan runs/ and return one entry per run with its manifest (if any).

    For each subdirectory of ``runs/``, attempts to read the manifest.
    Always returns one entry per directory so callers can show "run
    folder exists but manifest missing/corrupt" cases in the UI rather
    than silently dropping them.

    Args:
        step2_runs_root: path to ``{workspace}/artifacts/step2/runs/``.

    Returns:
        List of dicts, one per run dir.  Each dict has:
            - ``run_id`` (str): the directory name
            - ``manifest`` (dict | None): the parsed manifest, or None
            - ``manifest_ok`` (bool): True if manifest was parsed
            - ``manifest_path`` (str): absolute path to the manifest file
              (even if missing — useful for diagnostics)
    """
    root = Path(step2_runs_root)
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for run_id in list_run_ids(root):
        manifest_path = root / run_id / RUN_MANIFEST_FILENAME
        manifest = read_run_manifest(manifest_path)
        out.append({
            "run_id": run_id,
            "manifest": manifest,
            "manifest_ok": manifest is not None,
            "manifest_path": str(manifest_path),
        })
    return out


def is_run_archived(manifest: Optional[Dict[str, Any]]) -> bool:
    """True if the manifest indicates the run is archived (Phase 8c).

    Defensive: missing/None manifest → False (not archived).  Field
    absent → False.  Field set to truthy ISO timestamp → True.
    """
    if not isinstance(manifest, dict):
        return False
    return bool(manifest.get("archived_at"))


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """ISO 8601 UTC timestamp with trailing Z (matches migration code)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    # Schema constants
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RUN_MANIFEST_FILENAME",
    "VALID_INIT_MODES",
    "VALID_BACKENDS",
    "ENABLED_BACKENDS",
    # Build / write / read
    "build_run_manifest",
    "write_run_manifest",
    "read_run_manifest",
    # Init-spec builders
    "init_spec_scratch",
    "init_spec_inherited",
    "init_spec_cross_bucket",
    "init_spec_step3_refined",
    "init_spec_pretrained",
    "init_spec_browse",
    "init_spec_fork",
    # Discovery
    "list_run_ids",
    "list_runs_with_manifests",
    "is_run_archived",
]

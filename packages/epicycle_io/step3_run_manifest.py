"""Step 3 per-run manifest (``run_manifest.json``).

Each *named* Step 3 run lives under ``artifacts/step3/runs/{run_slug}/``
and carries a sidecar ``run_manifest.json`` describing the run's
identity, biological target, backend, the model it segments with, and
the source the run was bootstrapped from (its model lineage).

This module owns the manifest schema + read/write helpers, plus the
multi-run-discovery scanner used by the (Phase 4) UI.  It is the Step 3
mirror of ``packages/epicycle_io/step2_run_manifest.py`` — intentionally
pure-Python (no Qt, no napari, no controller deps), living under
``packages/`` per the layer-separation rule (packages never import the plugin).

**The default run is FLAT and manifest-less**: the single-context
``runs/{patch}/`` reconstruction dirs ARE the default run, so the default
run never has a ``runs/default/`` context dir and never a
``run_manifest.json``.  Therefore the **discriminator** between a
run-context dir and a default-run patch-recon dir under ``runs/`` is the
**presence of ``run_manifest.json``**: a dir that has one is a run
context; a dir that doesn't is a per-patch reconstruction output of the
default run.  :func:`is_run_context_dir` + :func:`list_runs_with_manifests`
encode that rule.
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

# Allowed values for ``created_from.mode`` (the run's MODEL lineage).
# Step 3 always starts from a model — there is NO "scratch".  Mirrors the
# New-Run picker: pretrained / Step-2 run-model / Step-3-refined /
# browse.  (Step 2's inherited / cross_bucket / fork / scratch modes have
# NO Step 3 analogue — runs reuse models by copy-on-reuse, not by data
# import — so they are deliberately absent.)
VALID_INIT_MODES = frozenset({
    "pretrained",      # a Cellpose built-in (cyto3, nuclei, ...)
    "step2_run",       # a Step 2 trained run-model in this workspace
    "step3_refined",   # a Step 3 refinement bundle in this workspace
    "browse",          # an arbitrary file the user picked
})

# Allowed values for ``backend``.  Cellpose is the only one enabled today;
# StarDist / Omnipose are reserved (the per-model manifest
# carries ``backend`` so the layout is multi-backend-ready).
VALID_BACKENDS = frozenset({
    "cellpose",
    "stardist",   # reserved, not yet implemented
    "omnipose",   # reserved, not yet implemented
})

ENABLED_BACKENDS = frozenset({"cellpose"})
"""Subset of VALID_BACKENDS wired up today.  UI greys the rest."""

# Canonical biological-target labels — reuse Step 2's labels
# and add "cells").  This is low-stakes TRACKING metadata, NOT a gate — free
# text is accepted; this set is the suggested vocabulary for the picker.
VALID_TARGETS = frozenset({"", "cells", "nuclei", "membrane", "general"})


# ----------------------------------------------------------------------------
# Build / write / read
# ----------------------------------------------------------------------------

def build_step3_run_manifest(
    *,
    run_id: str,
    display_name: str,
    target: str = "",
    description: str = "",
    backend: str = "cellpose",
    created_from: Optional[Dict[str, Any]] = None,
    model: Optional[Dict[str, Any]] = None,
    notes: str = "",
    created_at: Optional[str] = None,
    archived_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Step 3 run_manifest dict from explicit fields.

    All validation happens here so the writer can stay a thin dump.

    Args:
        run_id: slug; must match the parent dir name.  Validated via
            ``fusion_reconstruction._paths.validate_run_id`` upstream;
            this module just checks non-empty.
        display_name: human-friendly name shown in the run picker.
        target: biological target hint (free text — ``""``, ``"cells"``,
            ``"nuclei"``, ``"membrane"``, ``"general"``, ...).  Empty
            allowed.  NOT strictly validated (tracking metadata only).
        description: optional multiline description.
        backend: must be in :data:`VALID_BACKENDS`; UI today only enables
            ``"cellpose"`` (checked elsewhere via :data:`ENABLED_BACKENDS`).
        created_from: model-lineage dict.  See the ``init_spec_*``
            builders.  ``None`` allowed (a run whose source is unknown /
            not recorded); a provided dict MUST have a valid ``mode``.
        model: the run's model selection dict — keys
            ``lateral_model_path`` / ``axial_model_path`` /
            ``model_source_mode``.  ``None`` → a blank model block.
        notes: optional free text.
        created_at: optional ISO timestamp.  If None, populated as
            ``utcnow`` (``...Z``).
        archived_at: optional ISO timestamp.  Defaults to None.

    Returns:
        Manifest dict ready to be JSON-dumped.

    Raises:
        ValueError: on empty run_id, empty display_name, invalid backend,
            or an invalid created_from mode.
    """
    if not run_id or not isinstance(run_id, str):
        raise ValueError("run_id must be a non-empty string")
    if not display_name or not isinstance(display_name, str):
        raise ValueError("display_name must be a non-empty string")
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"backend {backend!r} not in VALID_BACKENDS={sorted(VALID_BACKENDS)}"
        )

    cf = _normalize_created_from(created_from)
    model_block = _normalize_model(model)

    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "display_name": display_name,
        "target": target or "",
        "description": description or "",
        "backend": backend,
        "created_at": created_at or _utc_now_iso(),
        "archived_at": archived_at,
        "created_from": cf,
        "model": model_block,
        "notes": notes or "",
    }


def _normalize_created_from(
    created_from: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Validate + fill-in a ``created_from`` (model-lineage) dict.

    ``None`` passes through as ``None`` (source unknown).  A provided dict
    MUST contain ``mode`` in :data:`VALID_INIT_MODES`; the optional
    per-mode keys are filled with ``None`` so downstream readers don't
    ``KeyError``.
    """
    if created_from is None:
        return None
    if not isinstance(created_from, dict):
        raise ValueError(
            f"created_from must be a dict or None, got {type(created_from).__name__}"
        )
    cf = dict(created_from)
    if "mode" not in cf:
        raise ValueError("created_from must include 'mode'")
    if cf["mode"] not in VALID_INIT_MODES:
        raise ValueError(
            f"created_from.mode {cf['mode']!r} not in VALID_INIT_MODES="
            f"{sorted(VALID_INIT_MODES)}"
        )
    cf.setdefault("source_step2_run_id", None)
    cf.setdefault("source_step3_bundle", None)
    cf.setdefault("source_step3_run_id", None)
    cf.setdefault("pretrained_name", None)
    cf.setdefault("external_path", None)
    cf.setdefault("external_label", None)
    return cf


def _normalize_model(model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize the per-run ``model`` block to its three canonical keys."""
    m = dict(model) if isinstance(model, dict) else {}
    return {
        "lateral_model_path": m.get("lateral_model_path"),
        "axial_model_path": m.get("axial_model_path"),
        "model_source_mode": m.get("model_source_mode"),
    }


def write_run_manifest(manifest_path: Path | str, manifest: Dict[str, Any]) -> None:
    """Atomically write a run_manifest dict to disk.

    Uses tempfile + fsync + os.replace for atomicity (mirror Step 2).
    Parent directory must already exist; this function does NOT mkdir
    (the caller's create_run flow owns the directory structure so a
    half-created run isn't possible).

    Raises:
        OSError: on filesystem failures (incl. missing parent dir).
        ValueError: on missing required keys in manifest.
    """
    manifest_path = Path(manifest_path)
    parent = manifest_path.parent
    if not parent.is_dir():
        raise OSError(f"parent directory does not exist: {parent}")
    for required in ("schema_version", "run_id", "display_name", "backend",
                     "created_at"):
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

    Defensive: returns ``None`` on missing file, unreadable file, invalid
    JSON, or a non-object payload.

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
        print(f"[WARN] step3 run_manifest read failed at {manifest_path}: "
              f"{type(_e).__name__}: {_e}")
        return None
    if not isinstance(data, dict):
        print(f"[WARN] step3 run_manifest at {manifest_path} is not a "
              f"JSON object (got {type(data).__name__})")
        return None
    return data


# ----------------------------------------------------------------------------
# Init-spec builders (one per mode, for ergonomic call sites)
# ----------------------------------------------------------------------------

def init_spec_pretrained(*, pretrained_name: str) -> Dict[str, Any]:
    """Init from a Cellpose built-in model (cyto3, nuclei, ...)."""
    if not pretrained_name:
        raise ValueError("pretrained_name required")
    return {"mode": "pretrained", "pretrained_name": pretrained_name}


def init_spec_step2_run(*, source_step2_run_id: str) -> Dict[str, Any]:
    """Init from a Step 2 trained run-model in this workspace."""
    if not source_step2_run_id:
        raise ValueError("source_step2_run_id required")
    return {"mode": "step2_run", "source_step2_run_id": source_step2_run_id}


def init_spec_step3_refined(
    *,
    source_step3_bundle: str,
    source_step3_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Init from a Step 3 refinement bundle in this workspace.

    Unlike Step 2's ``init_spec_step3_refined`` the bucket is NOT pinned
    here — a Step 3 run carries BOTH a lateral and an axial model, so the
    bundle's lateral + axial weights are both resolved at use time (see
    :func:`..._run_management.resolve_init_spec_to_checkpoint`).

    ``source_step3_run_id`` DISAMBIGUATES the bundle's owning run.  A bundle
    NAME alone is ambiguous because the same name can exist under the flat
    default tree AND under any named run's ``refinement/models/`` — and a
    name-only scan resolves the WRONG run silently.  Pass the owning run's
    slug to pin resolution to exactly ``(run_id, bundle)``.  ``None`` (or
    ``""``) means the FLAT default tree.  When omitted (``None``) the key is
    NOT stored — the spec stays backward-compatible with name-only specs,
    and resolution treats the absent/empty run_id as the flat default.
    """
    if not source_step3_bundle:
        raise ValueError("source_step3_bundle required")
    spec: Dict[str, Any] = {
        "mode": "step3_refined",
        "source_step3_bundle": source_step3_bundle,
    }
    if source_step3_run_id:
        spec["source_step3_run_id"] = source_step3_run_id
    return spec


def init_spec_browse(*, external_path: str, external_label: str = "") -> Dict[str, Any]:
    """Init from an arbitrary user-picked file."""
    if not external_path:
        raise ValueError("external_path required")
    return {
        "mode": "browse",
        "external_path": external_path,
        "external_label": external_label or Path(external_path).name,
    }


# ----------------------------------------------------------------------------
# Run-context discrimination + discovery (scanner)
# ----------------------------------------------------------------------------

def is_run_context_dir(path: Path | str) -> bool:
    """True iff ``path`` is a *named-run context* dir (not a patch recon dir).

    The discriminator: a run context ALWAYS has a
    ``run_manifest.json`` sidecar (written atomically at create); a
    default-run per-patch reconstruction dir under ``runs/`` NEVER does
    (it holds ``labels_3d.*`` + ``manifest.json`` only).  A present-but-
    corrupt ``run_manifest.json`` STILL marks a run context (so the UI can
    surface the corruption rather than silently treating it as a patch
    dir) — hence ``.exists()``, not a successful read.
    """
    p = Path(path)
    return p.is_dir() and (p / RUN_MANIFEST_FILENAME).exists()


def list_runs_with_manifests(runs_root: Path | str) -> List[Dict[str, Any]]:
    """Scan ``runs/`` and return one entry per *run-context* dir.

    Only directories that ARE run contexts (have a ``run_manifest.json``
    file present — see :func:`is_run_context_dir`) are yielded; default-run
    per-patch reconstruction dirs are skipped.  Always yields one entry per
    run-context dir even when its manifest is missing/corrupt (so callers
    can show "run folder exists but manifest unreadable" cases rather than
    silently dropping them).

    Each entry dict has:
        - ``run_id`` (str): the directory name (slug)
        - ``manifest`` (dict | None): parsed manifest, or None if corrupt
        - ``manifest_ok`` (bool): True if the manifest parsed
        - ``manifest_path`` (str): absolute path to the manifest file
    """
    root = Path(runs_root)
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.name.startswith("."):
            continue
        if not is_run_context_dir(child):
            continue
        manifest_path = child / RUN_MANIFEST_FILENAME
        manifest = read_run_manifest(manifest_path)
        out.append({
            "run_id": child.name,
            "manifest": manifest,
            "manifest_ok": manifest is not None,
            "manifest_path": str(manifest_path),
        })
    return out


def is_run_archived(manifest: Optional[Dict[str, Any]]) -> bool:
    """True if the manifest indicates the run is archived.

    Defensive: missing/None manifest → False; field absent → False;
    field set to a truthy ISO timestamp → True.
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
    "VALID_TARGETS",
    # Build / write / read
    "build_step3_run_manifest",
    "write_run_manifest",
    "read_run_manifest",
    # Init-spec builders
    "init_spec_pretrained",
    "init_spec_step2_run",
    "init_spec_step3_refined",
    "init_spec_browse",
    # Discovery / discrimination
    "is_run_context_dir",
    "list_runs_with_manifests",
    "is_run_archived",
]

"""QC extraction-context sidecar — record + compare spacing / downsample at
extraction time so the controller can notify the user when their current
settings have drifted from what produced the stored features.

Background (2026-05-31, fix/qc-stale-notice): QC features include centroid
columns in physical units (when spacing != (1,1,1)) and physical-unit area /
volume / surface measurements.  Both depend on the spacing handed to
``regionprops_table`` at extraction time, and on the downsample factor that
shaped the in-memory labels at extraction time.  If the user changes either
after extraction and reloads, the navigator's auto-pan + the area / volume
columns silently reference old context — the autopan lands at the wrong
position, and physical measurements have drifted.

This module writes a small JSON sidecar next to the features CSV at
extraction time -- since R7a one per patch, ``<patch>_extraction_context.json``
(the shared ``extraction_context.json`` is read only for a patch with none), and provides a comparison helper for the
controller to detect mismatch.  It is intentionally **read/write only — no UI,
no controller logic** — so it stays trivially testable without Qt / napari.

The notification UX (modal dialog + persistent badge) lives in the controller.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

#: the SHARED sidecar every patch of a QC folder wrote before R7a; read
#: only for a patch that has no file of its own (legacy)
EXTRACTION_CONTEXT_FILENAME = "extraction_context.json"
#: since R7a one sidecar per patch: ``<patch>_extraction_context.json``
#: beside ``<patch>_features.csv``
PATCH_CONTEXT_SUFFIX = "_extraction_context.json"
_FEATURES_CSV_SUFFIX = "_features.csv"

# Tolerance for spacing equality — JSON round-trip + numpy float reductions
# can introduce sub-1e-6 drift that should not trigger a staleness signal.
_SPACING_REL_TOL = 1e-6
_SPACING_ABS_TOL = 1e-9


def _legacy_sidecar_path(csv_path: Union[str, Path]) -> Path:
    """The shared ``extraction_context.json`` of the CSV's folder."""
    return Path(csv_path).parent / EXTRACTION_CONTEXT_FILENAME


def _sidecar_path(csv_path: Union[str, Path]) -> Path:
    """``<patch>_extraction_context.json`` beside ``<patch>_features.csv``
    (R7a: the shared file let every extraction overwrite every other
    patch's context).  A CSV whose name carries no patch (not
    ``*_features.csv``) keeps the shared file, as before."""
    p = Path(csv_path)
    name = p.name
    if name.endswith(_FEATURES_CSV_SUFFIX) and \
            len(name) > len(_FEATURES_CSV_SUFFIX):
        return p.parent / (name[:-len(_FEATURES_CSV_SUFFIX)]
                           + PATCH_CONTEXT_SUFFIX)
    return _legacy_sidecar_path(p)


def write_extraction_context(
    csv_path: Union[str, Path],
    *,
    spacing: Sequence[float],
    downsample_factor: int,
) -> None:
    """Write the extraction context as a JSON sidecar next to ``csv_path``.

    Overwrites any existing sidecar, atomically (``_step3_manifests.
    write_json_atomic``: a reader sees the old file or the whole new one).
    Creates parent directories if needed.

    Args:
        csv_path: path to the features CSV (sidecar goes next to it).
        spacing: (s_z, s_y, s_x) per-voxel physical size at extraction time.
        downsample_factor: downsample factor of the labels at extraction time.
    """
    path = _sidecar_path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spacing": [float(s) for s in spacing],
        "downsample_factor": int(downsample_factor),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    from . import _step3_manifests as _sm
    _sm.write_json_atomic(path, payload)


def read_extraction_context(
    csv_path: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """Read the sidecar next to ``csv_path`` if present and valid.

    The patch's own ``<patch>_extraction_context.json`` first; the shared
    ``extraction_context.json`` only when the patch has none (a workspace
    extracted before R7a).  A patch file that exists but is malformed reads
    as ``None`` -- it never falls back to another patch's context.

    Returns:
        Parsed dict or ``None`` if the sidecar doesn't exist, the CSV path
        is invalid, or the sidecar is malformed.  Never raises.
    """
    try:
        sidecar = _sidecar_path(csv_path)
        if not sidecar.is_file():
            sidecar = _legacy_sidecar_path(csv_path)
    except (TypeError, ValueError):
        return None
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _spacings_match(a: Sequence[float], b: Sequence[float]) -> bool:
    if len(a) != len(b):
        return False
    return all(
        math.isclose(float(x), float(y), rel_tol=_SPACING_REL_TOL, abs_tol=_SPACING_ABS_TOL)
        for x, y in zip(a, b)
    )


def stale_reason(
    stored: Optional[Dict[str, Any]],
    *,
    current_spacing: Sequence[float],
    current_downsample: int,
) -> Optional[str]:
    """Compare a stored extraction context against the current state.

    Returns a short human-readable reason string when they differ, or ``None``
    when they match (or when no comparison is possible — missing sidecar or
    incomplete schema → not flagged, backward-compat).

    The reason string is suitable for showing in the notification UI:
    ``"spacing changed from (1.0, 1.0, 1.0) to (4.0, 0.25, 0.25)"`` etc.

    Args:
        stored: dict from :func:`read_extraction_context`, or ``None``.
        current_spacing: live `(s_z, s_y, s_x)` from QC settings.
        current_downsample: live downsample factor from the patch state.

    Returns:
        Short reason string, or ``None``.
    """
    if not isinstance(stored, dict):
        return None

    stored_spacing = stored.get("spacing")
    stored_factor = stored.get("downsample_factor")

    # Backward-compat: missing keys → can't compare reliably → don't flag.
    if stored_spacing is None or stored_factor is None:
        return None
    if not isinstance(stored_spacing, (list, tuple)):
        return None

    if not _spacings_match(stored_spacing, current_spacing):
        return (
            f"spacing changed from {tuple(float(s) for s in stored_spacing)} "
            f"to {tuple(float(s) for s in current_spacing)}"
        )

    try:
        stored_factor_int = int(stored_factor)
    except (TypeError, ValueError):
        return None
    if stored_factor_int != int(current_downsample):
        return (
            f"downsample factor changed from {stored_factor_int} "
            f"to {int(current_downsample)}"
        )

    return None


__all__ = [
    "EXTRACTION_CONTEXT_FILENAME",
    "PATCH_CONTEXT_SUFFIX",
    "write_extraction_context",
    "read_extraction_context",
    "stale_reason",
]

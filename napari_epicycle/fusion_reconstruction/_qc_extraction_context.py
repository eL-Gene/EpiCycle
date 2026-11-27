"""QC extraction-context sidecar — record + compare spacing / downsample at
extraction time so the controller can notify the user when their current
settings have drifted from what produced the stored features.

Background (2026-05-31, fix/qc-stale-notice): QC features include area /
volume / surface in physical units when a voxel size is known; the centroid
stays in voxel indices.  The measurements depend on the spacing handed to
``regionprops_table`` at extraction time, and on the downsample factor that
shaped the in-memory labels at extraction time.  If the user changes either
after extraction and reloads, the navigator's auto-pan + the area / volume
columns silently reference old context — the autopan lands at the wrong
position, and physical measurements have drifted.

This module writes a small JSON sidecar next to the features CSV at
extraction time -- since R7a one per patch,
``<patch>_extraction_context.json`` (the shared ``extraction_context.json``
is read only for a patch with none), and provides a comparison helper for
the controller to detect mismatch.  It is intentionally **read/write only — no UI,
no controller logic** — so it stays trivially testable without Qt / napari.

An unknown voxel size is written as unknown: ``"spacing": null`` (the
features were measured in voxel units), never ``[1, 1, 1]`` -- a real 1 µm
voxel and "nobody knows" must not read alike.  ``"spacing_source"`` beside it
names where the spacing came from (``"unknown"`` when nothing is known,
else the resolver's word: ``user`` / ``step1`` / ``file`` / ``qc_settings``).
An entered QC Settings 1 / 1 / 1 is written as unknown, like its factory
default; the Reshape context keeps (1, 1, 1) in memory.  Every reader goes through :func:`measured_zyx`, which takes ``null`` as
voxel units (:data:`VOXEL_UNITS`).  A sidecar written before the word existed
keeps its ``[1, 1, 1]`` and no ``spacing_source``; it reads as that list.

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

#: what a ``"spacing": null`` sidecar was measured in: one unit per voxel
VOXEL_UNITS = (1.0, 1.0, 1.0)

#: the keys of an extraction: written by :func:`write_extraction_context`
#: only, never by a pairing field or :func:`update_extraction_context`
_EXTRACTION_KEYS = ("spacing", "spacing_source", "downsample_factor",
                    "extracted_at")


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
    spacing: Optional[Sequence[float]],
    downsample_factor: int,
    pairing: Optional[Dict[str, Any]] = None,
    spacing_source: Optional[str] = None,
) -> None:
    """Write the extraction context as a JSON sidecar next to ``csv_path``.

    Overwrites any existing sidecar, atomically (``_step3_manifests.
    write_json_atomic``: a reader sees the old file or the whole new one).
    Creates parent directories if needed.

    Args:
        csv_path: path to the features CSV (sidecar goes next to it).
        spacing: (s_z, s_y, s_x) per-voxel physical size at extraction
            time, or ``None`` when no voxel size was known (the features
            are in voxel units): written as ``"spacing": null``.
        downsample_factor: downsample factor of the labels at extraction time.
        pairing: the pairing check's fields of this extraction (the other
            run and its file, the floors, the patch file's size and mtime),
            added as top-level keys.  Recorded only: :func:`stale_reason`
            never reads them.  The keys of the extraction always win.
        spacing_source: where ``spacing`` came from (``"unknown"`` when
            nothing was known); written beside it, and left out when not
            given.
    """
    path = _sidecar_path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    for key, value in dict(pairing or {}).items():
        payload[str(key)] = value
    for key in _EXTRACTION_KEYS:
        payload.pop(key, None)
    payload["spacing"] = (None if spacing is None
                          else [float(s) for s in spacing])
    if spacing_source is not None:
        payload["spacing_source"] = str(spacing_source)
    payload.update({
        "downsample_factor": int(downsample_factor),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    })
    from . import _step3_manifests as _sm
    _sm.write_json_atomic(path, payload)


def update_extraction_context(csv_path: Union[str, Path], **fields) -> bool:
    """Add ``fields`` to the patch's own sidecar beside ``csv_path``, keeping
    everything else (atomically, like :func:`write_extraction_context`).

    Only the patch's own ``<patch>_extraction_context.json`` is touched: a
    patch without one (never extracted) gets none, and the shared file of
    an older workspace is never written.  The keys of an extraction
    (``spacing``, ``spacing_source``, ``downsample_factor``,
    ``extracted_at``) cannot be set here.  ``True`` when the sidecar was
    written.  Never raises.
    """
    try:
        path = _sidecar_path(csv_path)
        if path == _legacy_sidecar_path(csv_path) or not path.is_file():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        for key, value in fields.items():
            if key not in _EXTRACTION_KEYS:
                data[str(key)] = value
        from . import _step3_manifests as _sm
        _sm.write_json_atomic(path, data)
        return True
    except (OSError, TypeError, ValueError):
        return False


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


def measured_zyx(ctx: Optional[Dict[str, Any]]) -> Optional[list]:
    """The ``[z, y, x]`` a sidecar's features were measured with.

    A three-number list -> that list (floats); ``"spacing": null`` -> voxel
    units, a fresh ``[1.0, 1.0, 1.0]``; the key absent, or anything else ->
    ``None`` (nothing can be said).  Never raises.
    """
    if not isinstance(ctx, dict) or "spacing" not in ctx:
        return None
    raw = ctx.get("spacing")
    if raw is None:
        return list(VOXEL_UNITS)
    if not isinstance(raw, (list, tuple)):
        return None
    try:
        out = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    return out if len(out) == 3 else None


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
    current_spacing: Optional[Sequence[float]],
    current_downsample: int,
) -> Optional[str]:
    """Compare a stored extraction context against the current state.

    Returns a short human-readable reason string when they differ, or ``None``
    when they match (or when no comparison is possible: no sidecar, no
    ``spacing`` key, an unreadable spacing, or no ``downsample_factor`` ->
    not flagged, backward-compat).

    The stored spacing is read through :func:`measured_zyx`: ``null`` means
    the features were measured in voxel units, so it is compared as
    ``(1, 1, 1)`` -- a later voxel size flags the features stale, and the
    downsample factor is compared as for any other sidecar.  A
    ``current_spacing`` of ``None`` (no voxel size known now) is voxel units
    too, named "voxel units" in the reason.

    The reason string is suitable for showing in the notification UI:
    ``"spacing changed from (1.0, 1.0, 1.0) to (4.0, 0.25, 0.25)"``, or
    ``"spacing changed from voxel units to (4.0, 0.25, 0.25)"`` when the
    stored spacing is ``null``, etc.

    Args:
        stored: dict from :func:`read_extraction_context`, or ``None``.
        current_spacing: the patch's `(s_z, s_y, s_x)` now, or ``None``
            when none is known (voxel units).
        current_downsample: live downsample factor from the patch state.

    Returns:
        Short reason string, or ``None``.
    """
    if not isinstance(stored, dict):
        return None

    stored_zyx = measured_zyx(stored)
    stored_factor = stored.get("downsample_factor")

    # Backward-compat: missing keys → can't compare reliably → don't flag.
    if stored_zyx is None or stored_factor is None:
        return None

    now_zyx = (VOXEL_UNITS if current_spacing is None
               else tuple(float(s) for s in current_spacing))
    if not _spacings_match(stored_zyx, now_zyx):
        was = ("voxel units" if stored.get("spacing") is None
               else f"{tuple(stored_zyx)}")
        now = "voxel units" if current_spacing is None else f"{now_zyx}"
        return f"spacing changed from {was} to {now}"

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
    "VOXEL_UNITS",
    "measured_zyx",
    "write_extraction_context",
    "read_extraction_context",
    "stale_reason",
]

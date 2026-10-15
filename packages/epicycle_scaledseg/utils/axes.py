"""
Axis-order resolution + canonicalization for input Z-stacks.

The engine's internal canonical layout is **ZCYX** — everything downstream of
``ZStackTiler.__init__`` (the ``self.Z, self.C, self.H, self.W = shape`` unpack,
the ``patchify`` ``(1, 1, p, p)`` window, the ``channel_index`` slicing, the
hardcoded ``axes="ZCYX"`` tile tag) assumes it. A raw input TIFF, however, may
be 3D ``ZYX`` or 4D ``ZCYX`` / ``CZYX``. Feeding a ``CZYX`` stack into the hard
ZCYX unpack silently swaps Z and C and mis-tiles the whole volume; a 3D stack
crashes the 4-tuple unpack outright.

This module resolves the layout ONCE at the single input boundary (Stage 0
tiling) and canonicalizes to ZCYX, so every downstream stage is correct by
construction.

Ported from the monorepo's per-patch resolver
(``napari_epicycle/fusion_reconstruction/qc_features.py::resolve_patch_axes`` /
``detect_axes_order``). Two deliberate divergences from the monorepo:
  * the ``labels``-shape disambiguation tier is DROPPED — there are no labels at
    tiling time (this is the raw input stack, before any segmentation);
  * the fallback default + final hard-return are **ZCYX** (the engine's canonical
    form), not the monorepo's CZYX. Parity is the resolution *logic*; the
    canonical *target* matches the engine's convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def detect_axes_order(shape: Tuple[int, ...]) -> str:
    """Auto-detect axis order for a stack from dimension sizes.

    Heuristic: the channel axis is the smallest dimension (typically 1-5
    channels vs hundreds of Z/Y/X pixels).

    - 3D -> ``"ZYX"`` (no channel axis)
    - 4D, axis-0 smallest -> ``"CZYX"`` (channel-first, common in microscopy)
    - 4D, axis-1 smallest -> ``"ZCYX"`` (Z-first)

    Ported verbatim from the monorepo ``detect_axes_order``.
    """
    if len(shape) == 3:
        return "ZYX"
    if len(shape) != 4:
        return "CZYX"  # not 3D/4D — defer to the resolver's fallback handling

    dim0, dim1 = shape[0], shape[1]
    if dim0 <= dim1 and dim0 <= shape[2]:
        return "CZYX"  # first axis smallest -> channel-first
    elif dim1 <= dim0 and dim1 <= shape[2]:
        return "ZCYX"  # second axis smallest -> Z-first, then channel
    else:
        return "CZYX"  # fallback to most common


def resolve_input_axes(
    image: np.ndarray,
    *,
    image_path: Optional[Union[str, Path]] = None,
    fallback: str = "ZCYX",
) -> str:
    """Resolve the axis order of an input Z-stack at the tiling boundary.

    Priority (most -> least authoritative):
      1. 3D short-circuit: ``image.ndim == 3`` -> ``"ZYX"`` (no C/Z ambiguity).
      2. TIFF metadata: if ``image_path`` given, ``TiffFile(path).series[0].axes``,
         used ONLY if exactly ``"ZCYX"`` or ``"CZYX"`` (upper-cased).
      3. Size heuristic: ``detect_axes_order(image.shape)`` if ``"ZCYX"`` / ``"CZYX"``.
      4. ``fallback`` (normalized to ``"ZCYX"`` / ``"CZYX"``; if it isn't one of
         those, use ``"ZCYX"`` — the engine's canonical layout).

    Every step is guarded so a failure falls through to the next signal — this
    function NEVER raises. Returns one of ``"ZYX"`` (3D), ``"ZCYX"``, ``"CZYX"``.

    NOTE — the monorepo's deterministic labels-shape tier is unavailable here
    (no labels exist at tiling), so a genuinely thin-Z stack (Z <= C) with no
    axes metadata is IRREDUCIBLY ambiguous and the heuristic may mis-call it.
    Pass ``--input-axes`` explicitly (or embed axes metadata) for such stacks.
    """
    # 1) 3D input has no C/Z ambiguity.
    try:
        if getattr(image, "ndim", None) == 3:
            return "ZYX"
    except Exception:
        pass

    # 2) TIFF metadata (most authoritative when present + recognizable).
    if image_path is not None:
        try:
            import tifffile

            with tifffile.TiffFile(str(image_path)) as tf:
                meta_axes = getattr(tf.series[0], "axes", "") or ""
            meta_axes = meta_axes.upper()
            if meta_axes in ("ZCYX", "CZYX"):
                return meta_axes
        except Exception:
            pass

    # 3) Size heuristic.
    try:
        heur = detect_axes_order(tuple(image.shape))
        if heur in ("ZCYX", "CZYX"):
            return heur
    except Exception:
        pass

    # 4) Fallback (normalized; engine canonical = ZCYX).
    try:
        fb = str(fallback).upper()
        if fb in ("ZCYX", "CZYX"):
            return fb
    except Exception:
        pass
    return "ZCYX"


def canonicalize_to_zcyx(image: np.ndarray, axes: str) -> np.ndarray:
    """Return ``image`` reshaped to the engine's canonical ``(Z, C, Y, X)``.

    - ``"ZYX"``  -> ``image[:, None, :, :]``      (insert a C=1 axis)
    - ``"CZYX"`` -> ``np.moveaxis(image, 0, 1)``   (C,Z,... -> Z,C,...)
    - ``"ZCYX"`` -> ``image``                      (already canonical, no-op)

    Content is preserved (``out[z, c]`` holds the same plane the input did);
    only the axis ORDER changes. Raises ``ValueError`` on an unrecognized axes
    string — the resolver only ever emits ZYX/ZCYX/CZYX, so a bad value here is
    a programming error, not a runtime input condition.
    """
    a = str(axes).upper()
    if a == "ZCYX":
        return image
    if a == "CZYX":
        return np.moveaxis(image, 0, 1)
    if a == "ZYX":
        return image[:, None, :, :]
    raise ValueError(f"canonicalize_to_zcyx: unsupported axes {axes!r}")

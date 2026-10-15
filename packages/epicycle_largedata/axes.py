"""Axis-order resolution + canonicalization for large-data sources.

Three responsibilities, all napari-free:

* **Resolve** the on-disk axis order of a raw stack (``resolve_axes`` ->
  :class:`AxisResolution`) — the UI-facing resolver moved here from Step 1's
  ``_largedata.py``.  It is deliberately RICHER than the standalone engine's
  string-only ``resolve_input_axes``: it carries the channel axis, channel
  count, a confident/inferred flag, and the signal source, so the Step 1 Order
  combo can show "detected: CZYX · 2 ch — confirm".  It also honours tifffile's
  ``'Q'`` unknown-axis marker.
* **Canonicalize** an array to a known layout — ``canonicalize_to_czyx`` (the
  convention Step 1's display + extract use: channel-first) and
  ``canonicalize_to_zcyx`` (the scaled-seg engine's convention: Z-first,
  channel-second).  Ported from the engine's ``utils/axes.py`` so the two
  repos share the same canonicalization semantics (INTENDED to stay in parity;
  a cross-repo parity test is deferred).
* **Read** physical voxel spacing (``read_voxel_spacing`` -> ``(z, y, x)`` µm)
  from a TIFF (ImageJ / OME) or ``.zarr`` (NGFF level-0 scale) header — the
  "detected" tier of the Step-1 voxel authority.  Never raises.

The resolution *heuristic* is shared with the engine (channel = the small
leading axis); the two differ only in their canonical *target*, which is why we
ship both canonicalizers here rather than pick one.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Axis detection
# ---------------------------------------------------------------------------

#: A leading axis this small (or smaller) reads as CHANNELS, not Z.  Real data
#: has 1-8 channels and tens-to-hundreds of Z slices, so the small axis is the
#: channel axis and the large one is Z.
CHANNEL_AXIS_MAX = 8

#: tifffile marks an axis it could not resolve with 'Q' (e.g. 'QQYX' on the
#: WoundT01 test file).  A string containing 'Q' is treated as unresolved.
_UNKNOWN_AXIS = "Q"


@dataclass(frozen=True)
class AxisResolution:
    """The resolved axis layout of a source, for the UI to show + confirm.

    Attributes
    ----------
    axes:
        On-disk axis order in canonical letters: ``"CZYX"`` / ``"ZCYX"`` (4D)
        or ``"ZYX"`` (3D single-channel).  Maps 1:1 onto Step 1's Order combo,
        whose only two entries are ``CZYX`` and ``ZCYX``.
    channel_axis:
        Index of the channel axis in the on-disk array, or ``None`` when the
        source is single-channel (3D).
    n_channels:
        Number of channels (``shape[channel_axis]``, or 1 for 3D).
    confident:
        ``True`` when read from a CLEAN metadata axes string; ``False`` when
        inferred from the shape (so the UI can prompt "detected … — confirm").
    source:
        ``"metadata"`` | ``"inferred"`` | ``"single-channel"``.
    """

    axes: str
    channel_axis: Optional[int]
    n_channels: int
    confident: bool
    source: str


def _clean_axes(meta_axes) -> str:
    """Upper-cased, stripped metadata axes string (``""`` if missing)."""
    if not meta_axes:
        return ""
    return str(meta_axes).upper().strip()


def _is_clean_4d_axes(ax: str) -> bool:
    """True only for a fully-resolved 4D order naming exactly C, Z, Y, X once
    each.  Rejects any string carrying an unknown ('Q') marker (e.g. 'QQYX',
    'QCYX') and anything that is not a permutation of CZYX (e.g. 'TCYX')."""
    if len(ax) != 4 or _UNKNOWN_AXIS in ax:
        return False
    return sorted(ax) == sorted("CZYX")


def _infer_channel_axis(a0: int, a1: int) -> int:
    """Which of the first two axes is the CHANNEL axis, from sizes alone.

    Rule: a small axis (<= :data:`CHANNEL_AXIS_MAX`) is channels, the large one
    is Z.  When that does not decide it (both small, or both large), the smaller
    axis is the channel axis; a tie resolves to axis 0 (``CZYX``), which is
    exactly the historical ``shape[0] > shape[1]`` auto-transpose default the
    loader used before detection existed.
    """
    a0_small = a0 <= CHANNEL_AXIS_MAX
    a1_small = a1 <= CHANNEL_AXIS_MAX
    if a0_small and not a1_small:
        return 0
    if a1_small and not a0_small:
        return 1
    return 0 if a0 <= a1 else 1


def resolve_axes(shape, meta_axes=None) -> AxisResolution:
    """Resolve the on-disk axis order of a source from its shape + tifffile axes.

    Handles 4D (``CZYX`` / ``ZCYX``) and 3D (``ZYX``).  A clean metadata axes
    string is trusted; a missing / unresolved one (tifffile's ``'Q'`` marker)
    is inferred from the shape via :func:`_infer_channel_axis`.  Deterministic
    — the guess is returned (with ``confident=False``) so the UI can show and
    let the user confirm / flip it, rather than silently applying a mis-detected
    channel axis.

    Parameters
    ----------
    shape:
        The array shape (any sequence of ints).
    meta_axes:
        The tifffile ``series.axes`` string, or ``None``.

    Raises
    ------
    ValueError:
        If ``shape`` has fewer than 3 or more than 4 dimensions (Step 1 handles
        only 3D/4D microscopy stacks).
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    ax = _clean_axes(meta_axes)

    if ndim == 3:
        # A 3D stack is ZYX (single channel).
        confident = bool(ax) and _UNKNOWN_AXIS not in ax and "C" not in ax
        return AxisResolution(axes="ZYX", channel_axis=None, n_channels=1,
                              confident=confident, source="single-channel")

    if ndim == 4:
        if _is_clean_4d_axes(ax):
            c_idx = ax.index("C")
            z_idx = ax.index("Z")
            order = "CZYX" if c_idx < z_idx else "ZCYX"
            return AxisResolution(axes=order, channel_axis=c_idx,
                                  n_channels=shape[c_idx], confident=True,
                                  source="metadata")
        c_idx = _infer_channel_axis(shape[0], shape[1])
        order = "CZYX" if c_idx == 0 else "ZCYX"
        return AxisResolution(axes=order, channel_axis=c_idx,
                              n_channels=shape[c_idx], confident=False,
                              source="inferred")

    raise ValueError(
        f"resolve_axes: unsupported ndim={ndim} (shape={shape}); "
        "handles 3D (ZYX) and 4D (CZYX/ZCYX) stacks only"
    )


# ---------------------------------------------------------------------------
# Physical voxel spacing (header detect) — the "detected" tier of the Step-1
# voxel authority.  Pure + napari-free; NEVER raises.
# ---------------------------------------------------------------------------

#: micrometer-per-unit conversion factors for the physical-size unit strings a
#: TIFF/OME header may carry.  Unknown / absent → assume micron (the bio-imaging
#: default), so a bare ``spacing`` still reads as µm.
_UM_FACTOR = {
    # both the micro sign (U+00B5) and Greek small mu (U+03BC) — TIFF/OME use
    # either — plus the ASCII spellings.
    "um": 1.0, "µm": 1.0, "μm": 1.0, "micron": 1.0, "microns": 1.0,
    "micrometer": 1.0, "micrometers": 1.0, "micrometre": 1.0, "micrometres": 1.0,
    "nm": 1e-3, "nanometer": 1e-3, "nanometers": 1e-3,
    "mm": 1e3, "millimeter": 1e3, "millimeters": 1e3, "millimetre": 1e3,
    "cm": 1e4, "centimeter": 1e4, "centimeters": 1e4, "centimetre": 1e4,
    "m": 1e6, "meter": 1e6, "meters": 1e6, "metre": 1e6,
    "inch": 25400.0, "in": 25400.0, "\"": 25400.0,
}


def _um_factor(unit) -> float:
    """µm-per-*unit* conversion factor; unknown / empty → 1.0 (assume micron)."""
    return _UM_FACTOR.get(str(unit or "").strip().lower(), 1.0)


#: Unit strings that name a genuine PHYSICAL length usable as microscopy spacing.
#: ``inch`` is deliberately EXCLUDED: a TIFF ``XResolution`` in inches is almost
#: always print DPI (tifffile's default ``ResolutionUnit`` when ``resolution=`` is
#: written), NOT a physical voxel size — treating 300 DPI as ~84.67 µm would seed
#: a bogus "detected" spacing.
_PHYSICAL_UNITS = frozenset(
    k for k in _UM_FACTOR if k not in ("inch", "in", "\"")
)


def _is_physical_unit(unit) -> bool:
    """True only for a real physical length unit (micron/nm/mm/cm/m…); False for
    inch (print DPI), empty, or unrecognized."""
    return str(unit or "").strip().lower() in _PHYSICAL_UNITS


def read_voxel_spacing(path) -> Optional[Tuple[float, float, float]]:
    """Best-effort physical voxel spacing ``(z, y, x)`` in **micrometers** read
    from a source file's HEADER, or ``None`` when no spacing is recorded.

    This is the "detected" tier of the Step-1 voxel authority:
    it is consulted only when the registry has no user override and no non-unit
    store NGFF scale.  Supported:

    * **TIFF** — ImageJ ``spacing`` (Z) + ``XResolution`` / ``YResolution``
      (XY), scaled by the ImageJ ``unit``; OME-TIFF ``PhysicalSize{X,Y,Z}`` +
      their units (preferred when present).
    * **``.zarr`` / ``.zip``** — the OME-NGFF level-0 ``coordinateTransformations``
      scale (the durable physical spacing a Prepare baked in).

    NEVER raises: an unreadable / non-spatial / foreign file returns ``None`` so
    the caller can use it as a plain optional.  A missing axis is filled from the
    others (a lone XY spacing yields ``(xy, y, x)``); an all-unit ``(1, 1, 1)`` is
    returned as-is (the caller treats unit spacing as "unknown").
    """
    try:
        p = str(path)
    except Exception:
        return None
    low = p.lower()
    try:
        if low.endswith((".zarr", ".zip")) or (
                os.path.isdir(p) and os.path.exists(os.path.join(p, ".zattrs"))):
            return _voxel_from_zarr(p)
        if low.endswith((".tif", ".tiff")):
            return _voxel_from_tiff(p)
    except Exception:
        return None
    return None


def _fill_zyx(z, y, x) -> Optional[Tuple[float, float, float]]:
    """Coalesce a partial ``(z, y, x)`` into a full triple, or ``None`` if all
    three are absent.  A missing axis falls back to a present XY, then Z, then
    ``1.0``."""
    if z is None and y is None and x is None:
        return None
    xy = x if x is not None else y
    zz = z if z is not None else (xy if xy is not None else 1.0)
    yy = y if y is not None else (xy if xy is not None else 1.0)
    xx = x if x is not None else (xy if xy is not None else 1.0)
    try:
        return (float(zz), float(yy), float(xx))
    except (TypeError, ValueError):
        return None


def _parse_ome_voxel(xml) -> Optional[Tuple[float, float, float]]:
    """PhysicalSize{Z,Y,X} (+ per-axis unit) from an OME-XML metadata string, in
    µm, or ``None``.  Regex-based (no XML dependency); never raises."""
    try:
        s = str(xml)
    except Exception:
        return None

    def _num(attr):
        m = re.search(attr + r'\s*=\s*"([0-9eE.+-]+)"', s)
        try:
            return float(m.group(1)) if m else None
        except (TypeError, ValueError):
            return None

    def _unit(attr):
        m = re.search(attr + r'\s*=\s*"([^"]+)"', s)
        return m.group(1) if m else None

    z = _num("PhysicalSizeZ")
    y = _num("PhysicalSizeY")
    x = _num("PhysicalSizeX")
    if z is not None:
        z *= _um_factor(_unit("PhysicalSizeZUnit") or "um")
    if y is not None:
        y *= _um_factor(_unit("PhysicalSizeYUnit") or "um")
    if x is not None:
        x *= _um_factor(_unit("PhysicalSizeXUnit") or "um")
    return _fill_zyx(z, y, x)


def _voxel_from_tiff(path) -> Optional[Tuple[float, float, float]]:
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        ome = getattr(tf, "ome_metadata", None)
        if ome:
            v = _parse_ome_voxel(ome)
            if v is not None:
                return v
        ij = getattr(tf, "imagej_metadata", None) or {}
        unit = str(ij.get("unit", "") or "")
        # Z from the ImageJ `spacing` — ONLY when a genuine PHYSICAL length unit
        # governs it: an ImageJ stack with `unit="inch"` (or none /
        # missing) must NOT yield e.g. 0.5 * 25400 = 12700 µm — consistent with
        # the XY inch/none rejection below.
        z = None
        if _is_physical_unit(unit) and ij.get("spacing") is not None:
            try:
                z = float(ij["spacing"]) * _um_factor(unit)
            except (TypeError, ValueError, KeyError):
                z = None

        page = tf.pages[0]

        def _res(tagname):
            tag = page.tags.get(tagname)
            if tag is None:
                return None
            val = tag.value
            try:            # rational (num, den) → pixels per unit
                num, den = val
                num, den = float(num), float(den)
                return (den / num) if num else None
            except (TypeError, ValueError):
                try:
                    fv = float(val)
                    return (1.0 / fv) if fv else None
                except (TypeError, ValueError):
                    return None

        # XY from XResolution/YResolution — ONLY when a genuine PHYSICAL length
        # unit governs it, so a print-DPI TIFF (ResolutionUnit=INCH, tifffile's
        # DEFAULT when `resolution=` is written) or an aspect-ratio-only TIFF
        # (ResolutionUnit=NONE / missing) does NOT masquerade as a physical voxel.  The ImageJ `unit` (when physical) governs the
        # resolution; else the TIFF ResolutionUnit, but only its length values
        # (CENTIMETER; MILLIMETER/MICROMETER/NANOMETER if a writer used them).
        xy_factor = None
        if unit and _is_physical_unit(unit):
            xy_factor = _um_factor(unit)
        elif not unit:
            runit = page.tags.get("ResolutionUnit")
            rv = getattr(runit, "value", None)
            try:
                rv = int(rv) if rv is not None else 1
            except (TypeError, ValueError):
                rv = 1
            # TIFF ResolutionUnit: 1=NONE (aspect only), 2=INCH (print DPI) -> NOT
            # spacing.  3=CENTIMETER; extended 4=MILLIMETER, 5=MICROMETER,
            # 6=NANOMETER (some writers).
            xy_factor = {3: 1e4, 4: 1e3, 5: 1.0, 6: 1e-3}.get(rv)
        if xy_factor is not None:
            x = _res("XResolution")
            y = _res("YResolution")
            if x is not None:
                x *= xy_factor
            if y is not None:
                y *= xy_factor
        else:
            x = y = None
        return _fill_zyx(z, y, x)


def _voxel_from_zarr(path) -> Optional[Tuple[float, float, float]]:
    import zarr

    p = str(path)
    store = zarr.ZipStore(p, mode="r") if p.lower().endswith(".zip") \
        else zarr.DirectoryStore(p)
    try:
        group = zarr.open_group(store=store, mode="r")
        ms = group.attrs.get("multiscales")
    finally:
        if hasattr(store, "close"):
            try:
                store.close()
            except Exception:
                pass
    if not ms:
        return None
    block = ms[0] if isinstance(ms, (list, tuple)) and ms else ms
    datasets = block.get("datasets") if isinstance(block, dict) else None
    if not datasets:
        return None
    try:
        scale = datasets[0]["coordinateTransformations"][0]["scale"]
    except (KeyError, IndexError, TypeError):
        return None
    axes_meta = block.get("axes") or []
    names = [str((a or {}).get("name", "")).lower() for a in axes_meta] \
        if axes_meta else []
    if names and len(names) == len(scale):
        d = {n: scale[i] for i, n in enumerate(names)}
        z, y, x = d.get("z"), d.get("y"), d.get("x")
        if z is not None and y is not None and x is not None:
            try:
                return (float(z), float(y), float(x))
            except (TypeError, ValueError):
                return None
    if len(scale) >= 3:               # CZYX ([1,z,y,x]) or ZYX ([z,y,x])
        try:
            return (float(scale[-3]), float(scale[-2]), float(scale[-1]))
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Canonicalization (ported from the scaled-seg engine's utils/axes.py)
# ---------------------------------------------------------------------------

def canonicalize_to_czyx(image: np.ndarray, axes: str) -> np.ndarray:
    """Return ``image`` reshaped to **channel-first** ``(C, Z, Y, X)``.

    This is the convention Step 1's display + extract already use (see
    ``controller._crop_source_for``, which reorders ZCYX -> CZYX before a crop).

    - ``"CZYX"`` -> ``image``                     (already canonical, no-op)
    - ``"ZCYX"`` -> ``np.moveaxis(image, 1, 0)``  (Z,C,... -> C,Z,...)
    - ``"ZYX"``  -> ``image[None, :, :, :]``      (insert a leading C=1 axis)

    Content is preserved (``out[c, z]`` holds the same plane the input did);
    only the axis ORDER changes.  Raises ``ValueError`` on an unrecognized axes
    string — :func:`resolve_axes` only ever emits ZYX/ZCYX/CZYX, so a bad value
    here is a programming error, not a runtime input condition.
    """
    a = str(axes).upper()
    if a == "CZYX":
        return image
    if a == "ZCYX":
        return np.moveaxis(image, 1, 0)
    if a == "ZYX":
        return image[None, :, :, :]   # explicit 4-index form asserts 3D input
    raise ValueError(f"canonicalize_to_czyx: unsupported axes {axes!r}")


def canonicalize_to_zcyx(image: np.ndarray, axes: str) -> np.ndarray:
    """Return ``image`` reshaped to **Z-first** ``(Z, C, Y, X)`` — the scaled-seg
    engine's canonical layout (ported from its ``utils/axes.py`` for parity).

    - ``"ZCYX"`` -> ``image``                     (already canonical, no-op)
    - ``"CZYX"`` -> ``np.moveaxis(image, 0, 1)``  (C,Z,... -> Z,C,...)
    - ``"ZYX"``  -> ``image[:, None, :, :]``      (insert a C=1 axis after Z)

    See :func:`canonicalize_to_czyx` for the content-preservation guarantee.
    """
    a = str(axes).upper()
    if a == "ZCYX":
        return image
    if a == "CZYX":
        return np.moveaxis(image, 0, 1)
    if a == "ZYX":
        return image[:, None, :, :]   # explicit 4-index form asserts 3D input
    raise ValueError(f"canonicalize_to_zcyx: unsupported axes {axes!r}")

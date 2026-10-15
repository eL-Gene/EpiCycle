"""Read a raw channel / per-tile probability at a GLOBAL bbox — read-only, no re-run.

For the cell<->nucleus reconcile's membrane adjudication: given a problematic cell's
GLOBAL bbox (from ``label_provenance(_clean).csv``), read the raw cadherin (or nuclei
prob) there from the EXISTING engine tiles. **No re-tiling, no re-reconstruction** —
this mirrors the Mode-B stitch's read path (``iou_stitch_stage.compose_slice`` reads a
tile page at a local rect and places it at a global rect) but (a) points at the raw
``*_segmentrange_*.tif`` (4-D ZCYX) / per-tile ``*_probabilities.npy`` (3-D ZYX) instead
of the mask, (b) reads the cell's bbox+halo rect directly (no whole-plane core
partition — a per-cell crop just reads from the tile(s) covering it), (c) does no LUT.

Verified facts the reader relies on:

* The raw input tile is canonicalized to **ZCYX** at write
  (``tiling.py`` -> ``axes.canonicalize_to_zcyx``; CZYX inputs are moved to ZCYX), so
  **axis 1 is the channel axis** regardless of the original ordering. The membrane
  (cadherin) index = the cell run's signal channel; NOTE the seg config stores it as a
  **Cellpose 1-based** number (``0``=grayscale, ``1``=first array channel, …), so the
  ZCYX array index is ``cellpose - 1`` (:func:`resolve_channel_layout` does this).
  Confirm by rendering the channels on one cell — it's a hint, not authority.
* The raw tile holds the **FULL original Z** (``patch_stack = zstack[:, :, y0:y0+h,
  x0:x0+w]`` takes all Z); the ``segmentrange_<z0>_<z1>`` in the filename is metadata,
  NOT a Z crop. So ``global_z == tile_z`` — **do NOT apply the per-tile
  ``truncated_range``** (the seg stage does, downstream; this reader must not).
* The provenance bbox is **half-open on the high end** (``z1 = z_max + 1``) and indexes
  the global volume directly. Tile origins (``tile_geometry[*].origin = (y0, x0)``) give
  ``local = global - origin``. 25% tile overlap supplies the halo for interior cells;
  the FOV border clips it (those are the ``touches_border`` cells).
* The per-tile prob is ``(Z, Y, X)`` float16 and likely a **pre-sigmoid logit** — pass
  ``apply_sigmoid=True`` if a [0, 1] value is needed.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# (z0, z1, y0, y1, x0, x1) — HALF-OPEN on the high end (z1/y1/x1 exclusive),
# matching the provenance convention (z1 = z_max + 1).
Bbox = Tuple[int, int, int, int, int, int]


# --------------------------------------------------------------------------- #
# geometry + channel layout (all from the run's configs — nothing is asked)
# --------------------------------------------------------------------------- #

def load_geometry(tiles_dir) -> dict:
    """Read tile geometry + volume frame from ``<tiles_dir>/tiling_config.json``.

    Returns ``{full=(H,W), depth, channels, channel_index,
    geoms={tile_name: ((y0,x0),(h,w))}}``.
    """
    cfg_path = Path(tiles_dir) / "tiling_config.json"
    cfg = json.loads(cfg_path.read_text())
    full_h, full_w = cfg["full_size"]
    geoms: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
    for g in cfg["tile_geometry"]:
        oy, ox = g["origin"]
        h, w = g["extent"]
        geoms[g["tile_name"]] = ((int(oy), int(ox)), (int(h), int(w)))
    return {
        "full": (int(full_h), int(full_w)),
        "depth": int(cfg["depth"]),
        "channels": int(cfg.get("channels", 1)),
        "channel_index": int(cfg.get("channel_index", 1)),
        "input_tif": cfg.get("input_tif"),
        "geoms": geoms,
    }


def resolve_channel_layout(tiles_dir) -> dict:
    """Resolve the channel layout from disk (membrane + nucleus index; confirm by render).

    Returns ``{n_channels, signal_channel, nucleus_channel, signal_channel_cellpose,
    nucleus_channel_cellpose, input_tif}``. ``signal_channel`` / ``nucleus_channel`` are
    **ZCYX array indices** derived from the run pipeline's ``CHANNELS = [signal, nucleus]``
    (persisted per-tile as ``segmentation_params["channels"]``). Those values are **Cellpose
    1-based** channel numbers (``0``=grayscale/none, ``1``=first array channel, ``2``=second,
    …), so the array index is ``cellpose - 1`` (e.g. ``[2, 1]`` -> signal idx 1, nucleus idx
    0). A Cellpose nucleus of ``0`` means "no nucleus channel used" -> ``nucleus_channel`` is
    ``None`` (not index 0). Each field is ``None`` if no seg config is found or the value is
    out of range. The nucleus index is AUTHORITATIVE (it's what you segmented with), but still
    worth a one-cell render confirm — membrane vs nuclei is visually unmistakable.

    (The Z-range ``channel_index`` is deliberately NOT returned as a cadherin hint: it is
    the intensity-analysis channel and is not even persisted to ``tiling_config.json``.)
    """
    geo = load_geometry(tiles_dir)
    cellpose_channels = None
    for name in geo["geoms"]:
        cfgs = sorted(glob.glob(str(Path(tiles_dir) / name / "*_config.json")))
        for cp in cfgs:
            try:
                d = json.loads(Path(cp).read_text())
                ch = (d.get("segmentation_params") or {}).get("channels")
                if ch:
                    cellpose_channels = list(ch)   # Cellpose [signal, nucleus], 1-based
                    break
            except Exception:
                continue
        if cellpose_channels is not None:
            break

    def _to_index(cellpose_val):
        # Cellpose channel (1-based; 0 = grayscale/none) -> ZCYX array index, None if out of range.
        if cellpose_val is None:
            return None
        idx = max(0, int(cellpose_val) - 1)
        return idx if idx < geo["channels"] else None

    cellpose_signal = cellpose_channels[0] if cellpose_channels else None
    cellpose_nucleus = (cellpose_channels[1]
                        if cellpose_channels and len(cellpose_channels) > 1 else None)
    signal_channel = _to_index(cellpose_signal)
    # Cellpose nucleus == 0 means "no nucleus channel used" -> None (NOT array index 0).
    nucleus_channel = _to_index(cellpose_nucleus) if (cellpose_nucleus or 0) > 0 else None
    return {
        "n_channels": geo["channels"],
        "signal_channel": signal_channel,
        "nucleus_channel": nucleus_channel,
        "signal_channel_cellpose": cellpose_signal,
        "nucleus_channel_cellpose": cellpose_nucleus,
        "input_tif": geo["input_tif"],
    }


# --------------------------------------------------------------------------- #
# ROI assembly (shared core: pick covering tile(s), read local rect, place)
# --------------------------------------------------------------------------- #

def _clamp_roi(geo: dict, bbox: Bbox, halo: int, halo_z: Optional[int]) -> Bbox:
    H, W = geo["full"]
    depth = geo["depth"]
    hz = halo if halo_z is None else halo_z
    z0, z1, y0, y1, x0, x1 = bbox
    z0 = max(0, z0 - hz); z1 = min(depth, z1 + hz)
    y0 = max(0, y0 - halo); y1 = min(H, y1 + halo)
    x0 = max(0, x0 - halo); x1 = min(W, x1 + halo)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"degenerate ROI after halo+clamp: {(z0, z1, y0, y1, x0, x1)}")
    return (z0, z1, y0, y1, x0, x1)


def _assemble(geo: dict, roi: Bbox, read_tile_crop) -> np.ndarray:
    """Tile the clamped ROI from whichever tiles cover it; assert full coverage.

    ``read_tile_crop(tile_name, z0, z1, (ly0, ly1, lx0, lx1)) -> (Z, h, w) array``.
    Overlap double-writes are harmless (raw signal is identical across tiles). A cell+
    halo wider than one tile is filled from multiple tiles; any uncovered voxel raises.
    """
    z0, z1, y0, y1, x0, x1 = roi
    out: Optional[np.ndarray] = None
    written = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for name, ((oy, ox), (h, w)) in geo["geoms"].items():
        ty0, ty1, tx0, tx1 = oy, oy + h, ox, ox + w           # tile global footprint
        iy0 = max(y0, ty0); iy1 = min(y1, ty1)
        ix0 = max(x0, tx0); ix1 = min(x1, tx1)
        if iy1 <= iy0 or ix1 <= ix0:
            continue
        local = (iy0 - oy, iy1 - oy, ix0 - ox, ix1 - ox)       # global - origin
        crop = read_tile_crop(name, z0, z1, local)
        if out is None:
            out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=crop.dtype)
        out[:, iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = crop
        written[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = True
    if out is None or not written.all():
        cov = 0.0 if out is None else float(written.mean())
        raise RuntimeError(f"ROI not fully covered by tiles ({cov:.1%}); ROI={roi}")
    return out


# --------------------------------------------------------------------------- #
# raw channel reader (from the input tiles)
# --------------------------------------------------------------------------- #

def _raw_tile_path(tiles_dir, tile_name: str) -> str:
    # glob (the segmentrange suffix is per-tile, not constructible)
    for pat in ("*_segmentrange_*.tif", "*.tif"):
        cands = sorted(glob.glob(str(Path(tiles_dir) / tile_name / pat)))
        if cands:
            return cands[0]
    raise FileNotFoundError(f"no raw tile TIFF in {Path(tiles_dir) / tile_name}")


def _open_raw_tile(path: str, n_channels: int = 1) -> np.ndarray:
    """Open a raw input tile as a ``(Z, C, Y, X)`` array-like (memmap when possible).

    A 4-D tile is ZCYX as written. A 3-D tile is AMBIGUOUS: ImageJ drops a singleton
    axis, so it is ``ZYX`` (the C=1 case — add a C axis) OR ``CYX`` (the Z=1, C>=2 case —
    which we must NOT silently treat as ZYX). Use ``n_channels`` to disambiguate; raise
    on the genuinely-ambiguous C>1 collapse rather than mis-axing it.
    """
    import tifffile
    try:
        arr = tifffile.memmap(path)          # contiguous imagej tiff -> lazy slicing
    except Exception:
        arr = tifffile.imread(path)
    arr = np.asarray(arr) if not hasattr(arr, "ndim") else arr
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        if n_channels <= 1:                  # C=1 tile squeezed to ZYX -> add C axis
            return arr[:, None, :, :]
        raise ValueError(
            f"raw tile {path} is 3-D with {n_channels} channels declared — ambiguous "
            f"ZYX-vs-CYX collapse; cannot safely index the channel axis")
    raise ValueError(f"raw tile {path} has ndim {arr.ndim}, expected 4 (ZCYX)")


def read_global_roi(tiles_dir, bbox: Bbox, channel: int, halo: int = 0,
                    halo_z: Optional[int] = None) -> np.ndarray:
    """Raw channel ``channel`` (C-axis index) at a global ``bbox`` (+ halo). Returns (Z,Y,X).

    ``bbox`` is half-open on the high end (the provenance convention). ``channel`` indexes
    the ZCYX channel axis directly (``tiling_config["channel_index"]`` /
    ``resolve_channel_layout`` give the cadherin index). Reads the FULL-Z raw tile — does
    NOT apply ``truncated_range``.
    """
    geo = load_geometry(tiles_dir)
    if channel < 0 or channel >= geo["channels"]:
        raise IndexError(f"channel {channel} out of range for {geo['channels']}-channel tiles")
    roi = _clamp_roi(geo, bbox, halo, halo_z)

    def _read(name, z0, z1, local):
        ly0, ly1, lx0, lx1 = local
        arr = _open_raw_tile(_raw_tile_path(tiles_dir, name), geo["channels"])
        return np.asarray(arr[z0:z1, channel, ly0:ly1, lx0:lx1])

    return _assemble(geo, roi, _read)


# --------------------------------------------------------------------------- #
# per-tile probability reader (from stage2_reconstruction)
# --------------------------------------------------------------------------- #

def _prob_tile_path(recon_dir, tile_name: str) -> str:
    for pat in (f"{tile_name}_probabilities.npy", "*_probabilities.npy"):
        cands = sorted(glob.glob(str(Path(recon_dir) / tile_name / pat)))
        if cands:
            return cands[0]
    raise FileNotFoundError(f"no *_probabilities.npy in {Path(recon_dir) / tile_name} "
                            f"(was the run done with --no-prob3d?)")


def read_prob_roi(tiles_dir, recon_dir, bbox: Bbox, halo: int = 0,
                  halo_z: Optional[int] = None, apply_sigmoid: bool = False) -> np.ndarray:
    """Per-tile probability (Z,Y,X, full-Z, float16 LOGIT) at a global ``bbox`` (+ halo).

    ``tiles_dir`` supplies the tile geometry; ``recon_dir`` (``stage2_reconstruction``)
    holds the ``<tile>_probabilities.npy``. The prob volume has NO channel axis. Pass
    ``apply_sigmoid=True`` to map the logits to [0, 1] via ``scipy.special.expit``.
    """
    geo = load_geometry(tiles_dir)
    roi = _clamp_roi(geo, bbox, halo, halo_z)

    def _read(name, z0, z1, local):
        ly0, ly1, lx0, lx1 = local
        vol = np.load(_prob_tile_path(recon_dir, name), mmap_mode="r")  # (Z,Y,X)
        return np.asarray(vol[z0:z1, ly0:ly1, lx0:lx1])

    out = _assemble(geo, roi, _read)
    if apply_sigmoid:
        from scipy.special import expit
        out = expit(out.astype(np.float32))
    return out

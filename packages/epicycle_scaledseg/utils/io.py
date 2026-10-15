"""
I/O utilities for EpiCycle ScaledSegmentation.

Functions for loading/saving arrays, managing tile configs, and filesystem helpers.
"""

import os
import json
import logging
import datetime
import numpy as np

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Union


def base_name(patch_path: str) -> str:
    """Extract the base name (without extension) from a patch file path."""
    return os.path.basename(patch_path).split('.')[0]


def ensure_dir(p: Union[str, Path]):
    """Create directory (and parents) if it doesn't exist."""
    Path(p).mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """Return the current time as an ISO-formatted string (seconds precision)."""
    return datetime.datetime.now().isoformat(timespec='seconds')


def update_tile_config(cfg_path: Union[str, Path], updates: dict):
    """
    Merge `updates` into a JSON config file (shallow merge for dicts).

    If the file doesn't exist, it is created. Existing dict-valued keys
    are merged (one level deep); all other keys are overwritten.
    """
    cfg_path = Path(cfg_path)
    meta = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(meta.get(k), dict):
            meta[k].update(v)
        else:
            meta[k] = v
    with open(cfg_path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_array(p: Union[str, Path]) -> np.ndarray:
    """
    Load a .npy or .npz file and return as a float32 numpy array.

    For .npz files, looks for the key 'arr' first, then falls back
    to the single array if there's only one.
    """
    p = Path(p)
    if p.suffix == ".npz":
        with np.load(p) as z:
            if "arr" in z.files:
                arr = z["arr"]
            elif len(z.files) == 1:
                arr = z[z.files[0]]
            else:
                raise ValueError(f"Unexpected npz contents in {p}: {z.files}")
    else:
        arr = np.load(p)
    return np.asarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Volume IO seam — format dispatch (TIFF default, raw zarr opt-in)
# ---------------------------------------------------------------------------
#
# A thin format-dispatch layer so the engine can write/read large volumes as
# BigTIFF (the plugin ingest default, zero install) OR raw zarr (chunked, RAM-
# bounded, the scale path). This is NOT OME-Zarr / NGFF — just a backend seam.
# Its first use (Stack 7) is fixing the guaranteed-OOM full-volume float32
# allocation in stitch_prob_volume via a chunked zarr working store.
#
# HPC note: on Lustre/GPFS, a per-chunk zarr DirectoryStore creates one
# file per chunk -> an inode storm WORSE than the monolithic TIFF it replaces.
# Prefer a ZipStore (a single .zip) on networked scratch. save_volume routes a
# ``.zip`` path to ZipStore automatically.


def _resolve_volume_fmt(path: Union[str, Path], fmt: str) -> str:
    """Resolve ``fmt`` ('auto'|'tiff'|'zarr') from an explicit value or the suffix."""
    if fmt != "auto":
        if fmt not in ("tiff", "zarr"):
            raise ValueError(f"unknown volume format {fmt!r}; use 'auto', 'tiff', or 'zarr'")
        return fmt
    suf = Path(path).suffix.lower()
    if suf in (".tif", ".tiff"):
        return "tiff"
    if suf in (".zarr", ".zip"):
        return "zarr"
    raise ValueError(
        f"cannot auto-detect volume format from {Path(path).name!r}; "
        f"use a .tif/.tiff or .zarr/.zip suffix, or pass fmt='tiff'/'zarr'"
    )


def _zarr_store(path: Union[str, Path], mode: str):
    """Open the right zarr store for ``path``: ZipStore for ``.zip`` (GPFS-safe,
    single file), DirectoryStore otherwise."""
    import zarr  # deferred heavy import

    p = str(path)
    if p.lower().endswith(".zip"):
        return zarr.ZipStore(p, mode=mode)
    return zarr.DirectoryStore(p)


def save_volume(arr: np.ndarray, path: Union[str, Path], fmt: str = "auto", chunks=None):
    """Save a volume as BigTIFF (default) or raw zarr.

    ``fmt`` is 'auto' (by suffix: .tif/.tiff -> tiff; .zarr/.zip -> zarr), 'tiff',
    or 'zarr'. For zarr, a ``.zip`` path uses a (GPFS-safe) ZipStore; otherwise a
    DirectoryStore. ``chunks`` is passed through to zarr (None -> zarr auto-chunks).
    """
    import tifffile

    resolved = _resolve_volume_fmt(path, fmt)
    if resolved == "tiff":
        tifffile.imwrite(str(path), np.asarray(arr), bigtiff=True)
        return
    import zarr

    store = _zarr_store(path, mode="w")
    try:
        z = zarr.creation.open_array(
            store=store, mode="w", shape=arr.shape, dtype=arr.dtype,
            chunks=chunks if chunks is not None else True,
        )
        z[:] = arr
    finally:
        if hasattr(store, "close"):
            store.close()


def load_volume(path: Union[str, Path], fmt: str = "auto") -> np.ndarray:
    """Load a volume written by :func:`save_volume` back to a numpy array."""
    import tifffile

    resolved = _resolve_volume_fmt(path, fmt)
    if resolved == "tiff":
        return np.asarray(tifffile.imread(str(path)))
    import zarr

    store = _zarr_store(path, mode="r")
    try:
        z = zarr.open(store, mode="r")
        return np.asarray(z[:])
    finally:
        if hasattr(store, "close"):
            store.close()


def open_working_volume(shape, dtype, backend: str = "numpy", store_path=None, chunks=None):
    """A RAM-bounded accumulation store for stitching.

    ``backend='numpy'`` returns ``np.zeros(shape, dtype)`` (full volume in RAM —
    the legacy behavior). ``backend='zarr'`` returns a zero-filled, chunked,
    on-disk zarr array (RAM = a few chunks): it supports region ``+=`` and
    chunk-wise reads, so a stitch can accumulate without materializing the whole
    float32 volume (the OOM at 7680²×Z). ``store_path`` is required for zarr.
    """
    if backend == "numpy":
        return np.zeros(shape, dtype=dtype)
    if backend != "zarr":
        raise ValueError(f"unknown working-volume backend {backend!r}; use 'numpy' or 'zarr'")
    if store_path is None:
        raise ValueError("backend='zarr' requires a store_path")
    import zarr

    store = _zarr_store(store_path, mode="w")
    z = zarr.creation.open_array(
        store=store, mode="w", shape=tuple(shape), dtype=dtype,
        chunks=chunks if chunks is not None else True, fill_value=0,
    )
    return z


def pick_array_path(tile_dir: Union[str, Path], name_noext: str) -> Path:
    """
    Resolve a .npz or .npy file path, preferring .npz if both exist.

    Parameters
    ----------
    tile_dir : path to directory
    name_noext : filename stem without extension (e.g. 'tile_0_0_xy_flows')

    Returns
    -------
    Path to the existing file.

    Raises
    ------
    FileNotFoundError if neither .npz nor .npy exists.
    """
    tile_dir = Path(tile_dir)
    pz = tile_dir / f"{name_noext}.npz"
    py = tile_dir / f"{name_noext}.npy"
    if pz.exists():
        return pz
    if py.exists():
        return py
    raise FileNotFoundError(f"Missing {name_noext}.npz/.npy in {tile_dir}")

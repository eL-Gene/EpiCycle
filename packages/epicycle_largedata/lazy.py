"""Lazy on-disk open + crop + display-downsample for large-data sources.

napari-free.  Moved here from Step 1's ``_largedata.py`` (originally ported from
the EpiCycle-Viz figure tool).  The whole point: never ``imread`` everything — a
source is opened LAZY on disk, and only the crop the caller asks for is ever
materialized into RAM.

* :func:`open_lazy` — memmap (uncompressed TIFF) / dask (compressed / zarr).
* :func:`materialize` — realize a (lazy) array-like into RAM.
* :func:`crop_zyx` / :func:`crop_czyx` — materialize ONLY a bounded ROI.
* :func:`downsample_for_display` — a label-safe strided overview (VRAM cap).
* :func:`content_bbox` — cheap nonzero bounding box for auto-framing.
"""
from __future__ import annotations

import os

import numpy as np


def open_lazy(path, cache_bytes=None):
    """Open a source LAZY on disk — slicing reads only what is asked for.

    * uncompressed TIFF -> :func:`tifffile.memmap` (zero-copy numpy memmap)
    * compressed TIFF (DEFLATE / LZW / …) -> ``tifffile.imread(aszarr=True)``
      wrapped as a dask array.  Compressed pages are NOT memory-mappable (the
      toy-data lesson — the WoundT01 DEFLATE stack), so the memmap attempt
      raises and this branch takes over.
    * ``.zarr`` store / directory -> :func:`dask.array.from_zarr`, optionally
      over a bounded ``LRUStoreCache`` (``cache_bytes``; ``None`` -> the package
      default, which is 0 = UNCACHED, ``0`` -> uncached explicitly).

    Returns an array-like supporting numpy-style slicing that exposes
    ``.shape`` / ``.ndim`` / ``.dtype``.  dask results carry ``.compute`` (use
    :func:`materialize` on a crop); the memmap result is a numpy ``memmap``.
    """
    import dask.array as da
    import tifffile

    p = os.fspath(path)
    lower = p.lower()
    if lower.endswith(".zarr") or (
        os.path.isdir(p) and not lower.endswith((".tif", ".tiff"))
    ):
        # Brick 4b: an OPTIONAL bounded LRU read cache.  OFF by default
        # (DEFAULT_READ_CACHE_BYTES == 0) -- measured in napari it made
        # scrolling worse, since the OS page cache already keeps these files
        # resident and the LRU caches COMPRESSED blocks (decompression still
        # runs per access).  Worth enabling only on slow / remote storage.
        from .multiscale import (
            DEFAULT_READ_CACHE_BYTES, _cached_store, _zarr_store)

        nbytes = (DEFAULT_READ_CACHE_BYTES if cache_bytes is None
                  else int(cache_bytes))
        try:
            store = _cached_store(_zarr_store(p, "r"), nbytes)
            try:
                return da.from_zarr(store, component="0")
            except Exception:
                return da.from_zarr(store)
        except Exception:
            # any store-open oddity -> the original, uncached path
            try:
                return da.from_zarr(p, component="0")
            except Exception:
                return da.from_zarr(p)

    try:
        return tifffile.memmap(p)
    except Exception:
        import zarr

        store = tifffile.imread(p, aszarr=True)
        z = zarr.open(store, mode="r")
        if not hasattr(z, "shape"):
            keys = (sorted(z.array_keys()) if hasattr(z, "array_keys")
                    else sorted(z.keys()))
            z = z[keys[0]]
        return da.from_array(z, chunks=getattr(z, "chunks", "auto"))


def materialize(a):
    """Realize an array-like into RAM: ``.compute()`` a dask array, else
    ``np.asarray`` (a no-op view for an ndarray, a read for a memmap slice)."""
    return a.compute() if hasattr(a, "compute") else np.asarray(a)


def crop_zyx(arr, bounds):
    """Materialize ONLY the ``bounds`` crop of a 3D ``(Z, Y, X)`` array-like.

    ``bounds`` is a half-open dict ``{"z0","z1","y0","y1","x0","x1"}``.  The
    slice is lazy (a dask view / memmap view); :func:`materialize` reads just
    the crop off disk.
    """
    b = bounds
    return materialize(arr[b["z0"]:b["z1"], b["y0"]:b["y1"], b["x0"]:b["x1"]])


def crop_czyx(arr, bounds):
    """Materialize ONLY the ``bounds`` crop of a 4D ``(C, Z, Y, X)`` array-like
    (all channels kept).  See :func:`crop_zyx` for ``bounds``."""
    b = bounds
    return materialize(
        arr[:, b["z0"]:b["z1"], b["y0"]:b["y1"], b["x0"]:b["x1"]]
    )


def downsample_for_display(arr, factors):
    """Per-axis nearest/strided downsample of the last three ``(Z, Y, X)`` axes.

    Label-safe (no averaging — striding preserves integer label ids exactly),
    ported from EpiCycle-Viz.  Accepts 3D ``(Z,Y,X)`` or 4D ``(C,Z,Y,X)``
    numpy / dask arrays; the channel axis of a 4D array is left untouched.
    Factors are clamped to ``>= 1``.  Returns ``(in_RAM_array, (fz, fy, fx))`` —
    only the downsampled copy is realized into RAM, so VRAM is capped by the
    factor while the full-res handle stays lazy for extraction.
    """
    fz, fy, fx = (max(1, int(f)) for f in factors)
    if arr.ndim == 3:
        sub = arr[::fz, ::fy, ::fx]
    elif arr.ndim == 4:
        sub = arr[:, ::fz, ::fy, ::fx]
    else:
        raise ValueError(
            f"downsample_for_display expected 3D ZYX or 4D CZYX, got "
            f"ndim={arr.ndim}"
        )
    sub = sub.compute() if hasattr(sub, "compute") else np.ascontiguousarray(sub)
    return sub, (fz, fy, fx)


def content_bbox(arr, ny=256):
    """Nonzero bounding box in voxel coords, on a cheap strided subsample.

    Ported from EpiCycle-Viz.  Accepts 3D ``(Z,Y,X)`` or 4D ``(C,Z,Y,X)`` (uses
    channel 0).  Pads one stride so the tissue edge is never clipped.  Returns
    ``(mn_zyx, mx_zyx)`` in that array's voxels; an all-zero array returns the
    full extent.
    """
    a = arr if arr.ndim == 3 else arr[0]
    sz = max(1, a.shape[0] // 64)
    sy = max(1, a.shape[1] // ny)
    sx = max(1, a.shape[2] // ny)
    sub = a[::sz, ::sy, ::sx]
    sub = sub.compute() if hasattr(sub, "compute") else np.asarray(sub)
    nz = np.argwhere(sub != 0)
    if len(nz) == 0:
        return (0, 0, 0), tuple(int(s) for s in a.shape)
    step = np.array([sz, sy, sx])
    mn = nz.min(0) * step - step  # pad one stride so we never clip the edge
    mx = (nz.max(0) + 1) * step + step
    mn = np.clip(mn, 0, None)
    return (tuple(int(v) for v in mn),
            tuple(int(min(m, s)) for m, s in zip(mx, a.shape)))

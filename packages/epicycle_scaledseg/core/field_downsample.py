"""Streaming coarse-foreground + coarse-labels helpers for the downsampled FIELD
passes (Phase C / scaling stages S2-S3).

The truly-GLOBAL field passes — :func:`epicycle.core.qc_features.compute_tissue_depth`,
:func:`~epicycle.core.qc_features.compute_watershed_territory`,
:func:`~epicycle.core.qc_features.compute_lumen_features` — each need the WHOLE label
volume resident to run a 3D EDT / watershed / ``binary_fill_holes`` over it. At
whole-tissue scale (7680²×1000) that is a guaranteed OOM. The S2/S3 fix computes the
field on a foreground DOWNSAMPLED by an integer factor ``N`` and samples each cell at
its centroid mapped onto the coarse grid. This is an APPROXIMATION for ``N > 1``
(validated by ``scripts/bench_field_downsample.py``); ``N == 1`` never enters this
module — the field functions keep their original code as the ``downsample == 1``
branch (byte-identical).

This module's job is purely to BUILD the coarse arrays without ever loading the
full-resolution volume resident:

* :func:`coarse_foreground` — a ``bool`` coarse foreground (max-pool / logical-OR of
  ``labels > 0`` per ``N×N×N`` block). When ``labels`` is a ``np.memmap`` (or a path)
  it STREAMS the volume in z-chunks of ``N`` (a few full YX planes at a time), so peak
  RAM is ``O(N · Y · X)`` + the coarse volume ``O(volume / N³)`` — never the full
  volume.
* :func:`coarse_labels` — a coarse INTEGER label volume (seeds for the watershed
  territory pass), taking the block-CENTRE voxel's label per block (cheap +
  deterministic). Streamed the same way for a memmap input.

Both clamp the LAST block on each axis when the shape is not divisible by ``N`` (the
partial block is reduced over whatever voxels it contains).
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np

_logger = logging.getLogger("epicycle.features")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _is_memmap(arr) -> bool:
    """True for a numpy memmap (the streaming trigger). A path string is handled by
    the caller (it loads a memmap first)."""
    return isinstance(arr, np.memmap)


def coarse_foreground(labels: Union[np.ndarray, "np.memmap", str], factor: int) -> np.ndarray:
    """Build a coarse BOOLEAN foreground = ``labels > 0`` max-pooled by ``factor``.

    Shape is ``ceil(dim / factor)`` per axis (the partial last block is kept). The
    coarse value of a block is ``True`` iff ANY full-res voxel in that ``N×N×N`` block
    is foreground (logical OR / max-pool) — the standard binary downsample that never
    drops a thin sheet of tissue.

    Parameters
    ----------
    labels
        A 3D integer label volume (Z, Y, X). Either a RESIDENT ``np.ndarray`` (handled
        in one shot via the block reduction), a ``np.memmap`` (STREAMED in z-chunks of
        ``factor`` so the full-res volume is never resident — the scaling path), or a
        path to a ``.npy`` (opened as a memmap, then streamed).
    factor
        Integer downsample factor ``N >= 1``. ``factor == 1`` returns ``labels > 0``
        directly (a plain boolean copy at full resolution).

    Returns
    -------
    np.ndarray (bool), shape ``(ceil(Z/N), ceil(Y/N), ceil(X/N))``.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")

    # A path -> open as a memmap (so we stream it, not load it resident).
    if isinstance(labels, str):
        labels = np.load(labels, mmap_mode="r")

    if factor == 1:
        # No reduction — a resident boolean copy. (N=1 never enters the field-pass
        # downsample branch, but keep this correct + cheap for the helper's own use.)
        return np.asarray(labels) > 0

    z, y, x = labels.shape
    cz, cy, cx = _ceil_div(z, factor), _ceil_div(y, factor), _ceil_div(x, factor)
    if cz == 0 or cy == 0 or cx == 0:
        raise ValueError("factor too large for volume")

    if _is_memmap(labels):
        # STREAMING path — read `factor` full YX planes per coarse-z, max-pool the YX,
        # OR across the z-chunk. Peak RAM = factor full planes + the coarse volume.
        out = np.zeros((cz, cy, cx), dtype=bool)
        for kz in range(cz):
            z0 = kz * factor
            z1 = min(z0 + factor, z)
            slab = np.asarray(labels[z0:z1]) > 0     # (<=factor, Y, X) bool, resident
            # OR over the z-chunk -> a single full-res YX foreground plane.
            plane = slab.any(axis=0)                 # (Y, X) bool
            out[kz] = _maxpool_2d_or(plane, factor, cy, cx)
        return out

    # RESIDENT path — block-reduce the whole array at once.
    return _maxpool_3d_or(np.asarray(labels) > 0, factor, cz, cy, cx)


def _maxpool_2d_or(plane: np.ndarray, factor: int, cy: int, cx: int) -> np.ndarray:
    """Logical-OR max-pool a full-res 2D boolean plane to ``(cy, cx)`` by ``factor``,
    clamping the partial last block on each axis."""
    y, x = plane.shape
    out = np.zeros((cy, cx), dtype=bool)
    yd, xd = (y // factor) * factor, (x // factor) * factor
    cyd, cxd = yd // factor, xd // factor          # # of FULL coarse blocks per axis

    # (1) the fully-divisible interior — one vectorised reshape (the common, fast bulk).
    if yd > 0 and xd > 0:
        core = plane[:yd, :xd].reshape(cyd, factor, cxd, factor)
        out[:cyd, :cxd] = core.any(axis=(1, 3))

    # (2) the partial X tail (a column of blocks at the right edge) over the FULL coarse
    #     rows — reduce over the (partial) x block + the full y block in one any().
    if xd < x and yd > 0:
        tail = plane[:yd, xd:x].reshape(cyd, factor, x - xd)   # (cyd, factor, x_tail)
        out[:cyd, cxd] = tail.any(axis=(1, 2))

    # (3) the partial Y tail (a row of blocks at the bottom edge) over EVERY coarse
    #     column incl. the bottom-right corner block — reduce over the (partial) y rows.
    if yd < y:
        ytail = plane[yd:y]                          # (y_tail, X)
        for jx in range(cx):
            x0 = jx * factor
            out[cyd, jx] = bool(ytail[:, x0:min(x0 + factor, x)].any())
    return out


def _maxpool_3d_or(vol: np.ndarray, factor: int, cz: int, cy: int, cx: int) -> np.ndarray:
    """Logical-OR max-pool a full-res 3D boolean volume to ``(cz, cy, cx)``, clamping
    every partial last block. Resident path (the input is already a real array)."""
    z, y, x = vol.shape
    out = np.zeros((cz, cy, cx), dtype=bool)
    for kz in range(cz):
        z0 = kz * factor
        plane = vol[z0:min(z0 + factor, z)].any(axis=0)   # (Y, X) bool
        out[kz] = _maxpool_2d_or(plane, factor, cy, cx)
    return out


def _block_majority_nonzero(slab: np.ndarray, factor: int, cy: int, cx: int) -> np.ndarray:
    """Per-``factor×factor`` YX-block MAJORITY-of-nonzero label over a full-res slab.

    ``slab`` is the ``(<=factor, Y, X)`` full-res label sub-volume for one coarse-z
    block (already resident). For each coarse ``(ky, kx)`` block we take the MOST
    COMMON NONZERO label among the up-to-``factor³`` voxels in that block (ties broken
    toward the smaller id, deterministically); an all-background block stays 0. This
    replaces the old block-CENTRE seed, which silently dropped a cell that did not
    occupy its block centre (15% of cells at N=2, 44% at N=8 — see H1).

    Streaming-compatible: only this slab (``factor`` full YX planes) is resident, so
    peak RAM is ``O(factor·Y·X)`` + the coarse volume — never the full volume.
    """
    y, x = slab.shape[1], slab.shape[2]
    out = np.zeros((cy, cx), dtype=slab.dtype)
    for ky in range(cy):
        y0, y1 = ky * factor, min((ky + 1) * factor, y)
        for kx in range(cx):
            x0, x1 = kx * factor, min((kx + 1) * factor, x)
            block = slab[:, y0:y1, x0:x1].reshape(-1)
            nz = block[block != 0]
            if nz.size == 0:
                continue                               # background block -> 0
            # most common nonzero label; np.unique gives sorted ids so argmax of the
            # counts breaks ties toward the SMALLER id (deterministic).
            vals, counts = np.unique(nz, return_counts=True)
            out[ky, kx] = vals[int(np.argmax(counts))]
    return out


def coarse_labels(labels: Union[np.ndarray, "np.memmap", str], factor: int) -> np.ndarray:
    """Build a coarse INTEGER label volume (watershed seeds) by taking the
    MAJORITY-of-nonzero label per ``N×N×N`` block.

    Majority-of-nonzero sampling (vs. the old block-CENTRE voxel): each coarse voxel
    inherits the MOST COMMON NONZERO label among its ``N³`` full-res voxels, and an
    all-background block stays 0 (no spurious seeds). This fixes the H1 silent
    dropout — the old block-CENTRE seed gave NO seed to a cell that did not happen to
    occupy its block centre, NaN-ing its ``ws_*`` / ``territory_*`` columns for 15% of
    cells at N=2 and 44% at N=8. Majority-of-nonzero retains ~97-99% of cells (a cell
    still drops only when it is the minority in EVERY block it touches — i.e. it is
    sub-``N`` and always overlapped by a bigger neighbour). Ties break toward the
    smaller id (deterministic).

    Parameters
    ----------
    labels
        3D integer label volume (resident ``ndarray``, ``np.memmap``, or ``.npy`` path
        — a path / memmap is STREAMED, one coarse-z slab of ``factor`` full planes at a
        time, so the full-res volume is never resident).
    factor
        Integer downsample factor ``N >= 1``. ``factor == 1`` returns ``labels`` as an
        ``ndarray`` copy.

    Returns
    -------
    np.ndarray (same integer dtype as ``labels``), shape ``(ceil(Z/N), ...)``.

    Notes
    -----
    Logs a ``WARNING`` (logger ``epicycle.features``) reporting the fraction of
    full-res cells that STILL have no coarse seed after the majority pass, so the
    residual dropout is no longer silent.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")

    if isinstance(labels, str):
        labels = np.load(labels, mmap_mode="r")

    if factor == 1:
        return np.asarray(labels).copy()

    z, y, x = labels.shape
    cz, cy, cx = _ceil_div(z, factor), _ceil_div(y, factor), _ceil_div(x, factor)
    if cz == 0 or cy == 0 or cx == 0:
        raise ValueError("factor too large for volume")

    dtype = np.asarray(labels[:1]).dtype          # cheap dtype peek (one plane)
    out = np.zeros((cz, cy, cx), dtype=dtype)

    # Track which full-res cell ids survive into the coarse seed volume so we can
    # warn about the residual dropout (the input ids that got NO coarse seed).
    # ONE streaming pass: per coarse-z slab, compute the majority-of-nonzero coarse
    # plane AND collect both the full-res id set (``all_ids``) and the surviving coarse
    # seed set (``seen_ids``) from the SAME slab read — so the dropout warning costs no
    # extra I/O (no second pass over the volume).
    seen_ids: set = set()
    all_ids: set = set()
    for kz in range(cz):
        z0 = kz * factor
        z1 = min(z0 + factor, z)
        slab = np.asarray(labels[z0:z1])          # (<=factor, Y, X), resident
        nz_slab = slab[slab != 0]
        if nz_slab.size:
            all_ids.update(int(v) for v in np.unique(nz_slab))
        coarse_plane = _block_majority_nonzero(slab, factor, cy, cx)
        out[kz] = coarse_plane
        nz = coarse_plane[coarse_plane != 0]
        if nz.size:
            seen_ids.update(int(v) for v in np.unique(nz))

    # Residual-dropout warning: fraction of full-res cell ids with no coarse seed.
    n_total = len(all_ids)
    if n_total:
        n_dropped = len(all_ids - seen_ids)
        frac = n_dropped / n_total
        if n_dropped:
            _logger.warning(
                "coarse_labels(factor=%d): %d/%d cells (%.1f%%) have NO coarse seed "
                "after majority-of-nonzero downsampling -> their watershed-territory "
                "columns are NaN (sub-%d-voxel cells always overlapped by a larger "
                "neighbour).", factor, n_dropped, n_total, 100.0 * frac, factor)
    return out

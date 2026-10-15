"""Streaming (out-of-core) whole-volume Step-1 preprocessing — the PURE ENGINE.

napari-free BY DESIGN (mirrors the rest of ``packages/epicycle_preprocessing`` +
``packages/epicycle_largedata``): numpy + stdlib at module top; the heavy libs
(``dask`` / ``zarr``) are deferred-imported INSIDE the functions, exactly like
:mod:`packages.epicycle_largedata.multiscale`.  No Qt, no napari, no controller.

Why streaming
------------------------------------------------------------------------
The in-memory path materializes the whole channel + float32-doubles it + copies
it per op → ~5× the file at peak (~200 GB on a 35 GB source) and is
memory-bandwidth bound.  Streaming instead reads the source ONE cubic block at a
time straight off disk, runs the recipe on that block across cores, and writes
the uint16 result into a chunked OME-Zarr store — never holding the whole volume.
Peak RAM is bounded to ``workers × (block + 2·halo)³ × ~3`` (file-size
independent, verified), all cores stay busy, and the store is exactly what patch
extraction reads next.

The engine, the invariants, the proofs (all probed)
--------------------------------------------------------
* **Fused float32 apply, one pass.**  A channel's recipe is composed
  into ONE per-block function ``percentile_normalize(bounds) → local ops in order``
  kept in float32 throughout, baking ``to_uint16`` ONLY at the final write.  The
  op-by-op alternative (a uint16 store between local ops) is NOT byte-identical
  (the round-trip loses precision).  The fused block halo is the **SUM** of the
  fused local ops' reaches (gaussian ``int(4σ+0.5)`` + background ``2r``); a local
  op adds its reach, a pointwise op / normalize adds 0.  Probed byte-identical for
  a summed halo; ``max(...)`` seams (the negative-control test).
* **Chunk-aligned output blocks — a WRITE-CORRECTNESS invariant.**  A zarr
  chunk write is a whole-chunk read-modify-write; two blocks writing the SAME
  on-disk chunk from different threads RACE and silently corrupt (probed
  ``max|Δ|≈60000``).  The output block edge is therefore rounded UP to a multiple
  of the store chunk (``ceil(adaptive_edge / chunk) · chunk``) so each on-disk
  chunk is written by EXACTLY ONE block → the concurrent region-writes are safe.
* **Shared normalize window.**  With ``bounds`` given, normalize is
  pure pointwise, so ``normalize(block) == normalize(whole)[block]`` — the fused
  block result trims to a byte-identical interior.  The window is measured (once,
  if not supplied) via the EXACT 65536-bin uint16 histogram
  (:func:`ops.percentile_bounds_from_hist`), order-independent across blocks.
* **Direct zarr reads.**  Each block is read straight off the underlying
  zarr array (``zarr[read_sl]``), NOT ``da.from_zarr(...)[read_sl].compute()``
  (~1.6× slower — a fresh dask graph per block).  A dask handle falls back to
  ``.compute()``; the controller passes the resolved zarr array for the fast path.
* **Manual bounded loop, NOT ``map_overlap``.**  ``map_overlap → to_zarr``
  is unbounded (peak scales with workers).  Level-0 uses the wave-submit /
  cancel / progress harness mirrored from :func:`chunked.block_parallel_apply`;
  coarser levels reuse :mod:`multiscale`'s bounded coarsen-from-previous loop.
* **Atomic, Windows-safe commit.**  Write to ``<out>.tmp.<unique>``; on
  success ``rmtree`` any stale final then ``os.rename(tmp, final)`` (rename to a
  non-existent target — ``os.replace`` onto an existing dir → WinError 5).  On
  cancel / error the partial ``.tmp`` is removed — NEVER a half-store at ``final``.

Phase 2a scope: this engine streams the LEADING local + pointwise + normalize run
of each channel's recipe.  A CLAHE / bleach / geometry op STOPS the fused run;
the manifest flags ``whole_array_remainder`` so the controller handles the
remainder (out of scope here).  The 2a tests use gaussian / gaussian+background.
"""

from __future__ import annotations

import gc
import itertools
import logging
import math
import os
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import ops as _ops
from . import recipe as _recipe
from . import registry as _registry
from .chunked import _INFLIGHT_PER_WORKER, _MAX_BLOCK_WORKERS, _adaptive_block_edge, _axis_blocks
from .exceptions import PreprocessingCancelled

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# small pure helpers
# --------------------------------------------------------------------------- #

def _aligned_block(edge: int, chunk) -> int:
    """Round *edge* UP to a whole number of store chunks on EVERY axis.

    Guarantees each on-disk chunk is written by exactly ONE output block, so the
    concurrent region-writes never RMW-race a shared chunk.

    *chunk* is either a cubic int (legacy) or a ``(cz, cy, cx)`` tuple.  The
    block edge is the SAME on all three axes, so the edge must be a common
    multiple of all three chunk axes -- i.e. a multiple of their **lcm**.

    Aligning to ``max(...)`` is NOT enough and was a real hole: for the default
    ``(32, 128, 128)`` it happens to work (128 is a multiple of 32), which is
    exactly why it looked correct.  But a FOREIGN store -- one this pipeline did
    not write, opened through the inherit-the-source-shape path -- can have any
    chunk grid.  At ``(24, 100, 100)`` max-alignment yields multiples of 100, and
    ``100 % 24 != 0``, so a block boundary lands mid-chunk in Z and two worker
    threads read-modify-write the same on-disk chunk: silent corruption.  The lcm
    (600 here) is the smallest edge that divides cleanly on all three.

    This is why the lcm is the invariant rather than a validation rule
    elsewhere: alignment alone makes ANY chunk grid safe, so
    ``normalize_chunk_zyx`` does not need to reject "awkward" shapes (it must
    not -- it also has to round-trip whatever a foreign store already has on
    disk).  The cost is a larger block for a pathological grid, never a wrong
    write.
    """
    if isinstance(chunk, int) and not isinstance(chunk, bool):
        c = max(1, int(chunk))
    else:
        axes = [max(1, int(v)) for v in chunk]
        c = math.lcm(*axes) if axes else 1
    c = max(1, c)
    e = max(1, int(edge))
    return int(math.ceil(e / c) * c)


def _source_chunk_zyx(channel) -> tuple:
    """The ZYX chunk shape of *channel*'s backing store, else the Brick-4b default.

    Reads a zarr array's ``.chunks`` (or a dask array's ``.chunksize``) and takes
    the trailing three axes, so a processed store INHERITS the shape of the
    prepared store it was streamed from -- parity without a caller having to
    thread the value through.
    """
    from packages.epicycle_largedata.config import DEFAULT_CHUNK_ZYX

    for attr in ("chunks", "chunksize"):
        c = getattr(channel, attr, None)
        if c is None:
            continue
        try:
            vals = [int(v) for v in c]
        except (TypeError, ValueError):
            continue
        if len(vals) >= 3 and all(v >= 1 for v in vals[-3:]):
            return tuple(vals[-3:])
    return tuple(int(v) for v in DEFAULT_CHUNK_ZYX)


def _count_tiles(shape_zyx: Sequence[int], edge: int) -> int:
    e = max(1, int(edge))
    n = 1
    for d in shape_zyx:
        n *= int(math.ceil(int(d) / e))
    return int(n)


def _tile_specs(shape_zyx: Sequence[int], edge: int, halo: int) -> List[tuple]:
    """The ``(out, read, trim)`` block grid over a ZYX shape (see ``_axis_blocks``)."""
    axis = [_axis_blocks(int(n), int(edge), int(halo), int(n)) for n in shape_zyx]
    return list(itertools.product(*axis))


def _voxel_size_from_spacing(spacing: Any) -> Optional[Tuple[float, float, float]]:
    """The ``(z, y, x)`` voxel size for the NGFF ``scale`` metadata, or ``None``."""
    try:
        vs = tuple(float(s) for s in spacing)
    except (TypeError, ValueError):
        return None
    if len(vs) < 3:
        return None
    vs = vs[-3:]
    if any((not math.isfinite(v)) or v <= 0.0 for v in vs):
        return None
    return (float(vs[0]), float(vs[1]), float(vs[2]))


def _recipe_hash(norm_recipe: Any) -> str:
    """A stable hex digest of the normalized recipe (for the processed-store key)."""
    import hashlib
    import json

    try:
        payload = json.dumps(_recipe.to_json(norm_recipe), sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 — a hash is best-effort provenance, never fatal
        payload = repr(norm_recipe)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _remove_store_quiet(path: Any) -> None:
    """Best-effort delete of a store dir / file — swallow errors (cleanup path)."""
    import shutil

    p = Path(path)
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    except Exception:  # noqa: BLE001
        pass


def _make_block_reader(channel: Any) -> Callable[[tuple], np.ndarray]:
    """A ``read_slice -> ndarray`` reader that reads a block DIRECTLY.

    * numpy / zarr array / duck-array → ``channel[read_sl]`` (the direct read; for
      a zarr array this is the fast path).
    * dask array → ``channel[read_sl].compute()`` (fallback; the controller passes
      the underlying zarr array for the fast path — a fresh dask graph per block
      is ~1.6× slower).
    """
    try:
        import dask.array as da

        if isinstance(channel, da.Array):
            return lambda sl: np.asarray(channel[sl].compute())
    except Exception:  # noqa: BLE001 — dask absent / not a dask array → direct read
        pass
    return lambda sl: np.asarray(channel[sl])


# --------------------------------------------------------------------------- #
# fused per-channel composition
# --------------------------------------------------------------------------- #

def _geometry_is_identity(step: Any, spacing: Optional[Sequence[float]]) -> bool:
    """True iff this geometry step resamples NOTHING (all factors ≈ 1).

    The default recipe seeds a shared ``isotropic`` step.  On a source whose
    spacing already equals ``target_um`` (the common ``(1, 1, 1)``
    unknown-voxel case) it is a pure no-op, yet it still stopped the fused run
    — so every prepared source was refused with "resize step that can't stream
    yet".  An identity step can be SKIPPED with
    no effect on the result.

    ``iso`` factors are derived from the source spacing, so with no spacing
    this fails SAFE (``False`` → the step keeps blocking) rather than risk
    silently dropping a real resize.  ``aniso`` factors are explicit and need
    no spacing.
    """
    config = getattr(step, "config", None)
    if not isinstance(config, dict):
        return False
    mode = str(config.get("mode", "iso")).lower()
    if mode != "aniso" and spacing is None:
        return False
    axes = tuple(float(s) for s in (spacing or ())) or (1.0, 1.0, 1.0)
    try:
        factors = _ops._resolve_factors(config, _ops._align_axes(axes, 3), 3)
    except Exception:  # noqa: BLE001 — unresolvable → treat as a real resize
        return False
    return all(abs(float(f) - 1.0) < 1e-9 for f in factors)


def _fused_plan(
    flat_steps: Sequence[Any],
    *,
    spacing: Optional[Sequence[float]] = None,
) -> Tuple[List[Tuple[Callable, dict]], int, List[str]]:
    """Split a channel's flattened op list into the LEADING fused run + remainder.

    Walks the ops left-to-right accumulating LOCAL (bounded-neighborhood) and
    pointwise ops into the fused run; a geometry / adaptive (CLAHE) / global-stat
    (bleach) op STOPS the run (everything after is the whole-array remainder the
    controller must handle — Phase 2a).  An IDENTITY geometry step (see
    :func:`_geometry_is_identity`, needs *spacing* for the ``iso`` mode) is a
    no-op and is skipped instead of stopping the run.  Returns
    ``(fused_ops, halo, remainder_keys)`` where ``fused_ops`` is an ordered list of
    ``(fn, params)`` and ``halo`` is the SUM of the fused local ops' reaches (a
    pointwise op adds 0).  ``halo`` is independent of the normalize window, so this
    can be planned before the window is known.
    """
    fused: List[Tuple[Callable, dict]] = []
    halo = 0
    remainder: List[str] = []
    stopped = False
    for step in flat_steps:
        key = getattr(step, "key", None)
        op = _registry.find(key) if key else None
        if op is None or not op.implemented or op.fn is None:
            continue  # dropped by normalize already, belt-and-suspenders
        if stopped:
            remainder.append(str(key))
            continue
        chunk_class = getattr(op, "chunk_class", None)
        if op.kind == "geometry" and _geometry_is_identity(step, spacing):
            continue                      # resamples nothing → not a barrier
        if op.kind == "geometry" or chunk_class in ("adaptive", "global_stat"):
            stopped = True
            remainder.append(str(key))
            continue
        params = dict(getattr(step, "params", {}) or {})
        if chunk_class == "local":
            try:
                reach = int(op.halo(**params)) if callable(getattr(op, "halo", None)) else 0
            except Exception:  # noqa: BLE001 — a bad param → can't tile this op safely
                stopped = True
                remainder.append(str(key))
                continue
            fused.append((op.fn, params))
            halo += max(0, reach)
        elif chunk_class == "pointwise":
            fused.append((op.fn, params))  # +0 to the halo (per-voxel, no reach)
        else:
            # unknown taxonomy → stop the fused run rather than risk an undersized halo
            stopped = True
            remainder.append(str(key))
    return fused, int(halo), remainder


def _make_fused_fn(
    fused_ops: Sequence[Tuple[Callable, dict]],
    bounds: Optional[Tuple[float, float]],
) -> Callable[[np.ndarray], np.ndarray]:
    """The fused float32 per-block function: ``normalize(bounds) → fused ops``.

    Normalize runs FIRST (matching ``execute``'s hoisted stage), pointwise with a
    shared ``bounds`` window so ``normalize(block) == normalize(whole)[block]``.
    Returns float32 in ``[0, 1]``; the caller bakes ``to_uint16`` after trimming.
    """

    def _fused(block_f32: np.ndarray) -> np.ndarray:
        arr = _ops.percentile_normalize(block_f32, bounds=bounds)
        for fn, params in fused_ops:
            arr = np.asarray(fn(arr, **params), dtype=np.float32)
        return arr

    return _fused


# --------------------------------------------------------------------------- #
# bounded wave-submit block runner (mirror chunked.block_parallel_apply)
# --------------------------------------------------------------------------- #

def _run_blocks(
    specs: Sequence[tuple],
    do_block: Callable[[tuple], None],
    workers: int,
    should_cancel: Optional[Callable[[], bool]],
    on_wave: Callable[[int], None],
) -> None:
    """Run *do_block* over the *specs* grid, bounded + cancellable.

    Mirrors :func:`chunked.block_parallel_apply`'s harness: never queue the whole
    grid (only ``workers × _INFLIGHT_PER_WORKER`` outstanding), poll *should_cancel*
    once up front + once per wave, and ALWAYS ``shutdown(cancel_futures=True)`` in
    ``finally`` so a cancel drops the not-yet-started blocks and only the ≤ workers
    in-flight ones finish.  *on_wave* is called with the cumulative done count per
    wave (progress).  Raises :class:`PreprocessingCancelled` on cancel.
    """

    def _cancelled() -> bool:
        return should_cancel is not None and should_cancel()

    if _cancelled():
        raise PreprocessingCancelled("cancelled before start")

    total = len(specs)
    if total == 0:
        return
    w = max(1, min(int(workers), total))

    if w <= 1:
        done = 0
        for spec in specs:
            if _cancelled():
                raise PreprocessingCancelled("cancelled between blocks")
            do_block(spec)
            done += 1
            on_wave(done)
        return

    pool = ThreadPoolExecutor(max_workers=w)
    try:
        it = iter(specs)
        inflight: set = set()
        max_inflight = max(1, w * _INFLIGHT_PER_WORKER)
        for _ in range(max_inflight):
            spec = next(it, None)
            if spec is None:
                break
            inflight.add(pool.submit(do_block, spec))
        done = 0
        while inflight:
            finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                fut.result()  # surface a per-block exception (no silent partial)
                done += 1
            on_wave(done)
            if _cancelled():
                raise PreprocessingCancelled("cancelled between waves")
            for _ in range(len(finished)):
                spec = next(it, None)
                if spec is None:
                    break
                inflight.add(pool.submit(do_block, spec))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


class _Progress:
    """Threads per-wave block counts to an ``on_progress(done, total)`` callback.

    ``total`` is the full block budget (measure + apply blocks across channels +
    coarser levels), set once up front; ``base`` accumulates completed blocks so
    each phase reports globally-monotonic ``(base + done_in_phase, total)``.  Best
    effort — a raising callback never propagates into the write path.
    """

    def __init__(self, cb: Optional[Callable[[int, int], None]]) -> None:
        self._cb = cb
        self.base = 0
        self.total = 0

    def emit(self, done_in_phase: int) -> None:
        if self._cb is None:
            return
        try:
            self._cb(min(self.base + done_in_phase, self.total), self.total)
        except Exception:  # noqa: BLE001 — progress is emit-only; never fail the write
            pass

    def advance(self, n: int) -> None:
        self.base += int(n)


# --------------------------------------------------------------------------- #
# measure pass (streaming exact-histogram normalize window)
# --------------------------------------------------------------------------- #

def _measure_bounds(
    reader: Callable[[tuple], np.ndarray],
    shape_zyx: Sequence[int],
    edge: int,
    workers: int,
    should_cancel: Optional[Callable[[], bool]],
    prog: _Progress,
) -> Optional[Tuple[float, float]]:
    """The exact per-channel ``(lo, hi)`` normalize window via a streamed histogram.

    Reads the RAW uint16 source block-by-block (no halo needed — a histogram is
    order-independent), sums each block's :func:`ops.histogram_uint16` into one
    accumulator, and returns :func:`ops.percentile_bounds_from_hist`.  ``None`` for
    a flat / empty source (the all-zeros normalize fallback).
    """
    specs = _tile_specs(shape_zyx, edge, 0)
    acc = _ops.Uint16HistogramAccumulator()
    lock = threading.Lock()

    def _do(spec: tuple) -> None:
        read_sl = tuple(s[1] for s in spec)
        block = reader(read_sl)
        with lock:
            acc.add(block)

    _run_blocks(specs, _do, workers, should_cancel, prog.emit)
    prog.advance(len(specs))
    return _ops.percentile_bounds_from_hist(acc.histogram)


# --------------------------------------------------------------------------- #
# apply pass (fused level-0 block write)
# --------------------------------------------------------------------------- #

def _apply_channel(
    z_out: Any,
    reader: Callable[[tuple], np.ndarray],
    ci: int,
    canon_axes: str,
    fused_fn: Callable[[np.ndarray], np.ndarray],
    halo: int,
    block_edge: int,
    shape_zyx: Sequence[int],
    workers: int,
    should_cancel: Optional[Callable[[], bool]],
    prog: _Progress,
) -> None:
    """Stream one channel's fused float32 apply into the level-0 zarr array.

    Each block: read ``src[read+halo]`` DIRECTLY → float32 → ``fused_fn`` →
    ``to_uint16`` → write the trimmed interior to ``z_out`` at the channel's slot.
    ``block_edge`` is chunk-aligned by the caller so the region-writes are
    thread-safe.  The trimmed interior is byte-identical to the whole-array result
    (the halo covers the fused reach).
    """
    specs = _tile_specs(shape_zyx, block_edge, halo)

    def _do(spec: tuple) -> None:
        out_sl = tuple(s[0] for s in spec)
        read_sl = tuple(s[1] for s in spec)
        trim_sl = tuple(s[2] for s in spec)
        raw = reader(read_sl)
        block_f32 = np.asarray(raw, dtype=np.float32)
        result = fused_fn(block_f32)
        baked = _ops.to_uint16(result)
        trimmed = baked[trim_sl]
        if canon_axes == "CZYX":
            z_out[(ci,) + out_sl] = trimmed
        else:
            z_out[out_sl] = trimmed

    _run_blocks(specs, _do, workers, should_cancel, prog.emit)
    prog.advance(len(specs))


# --------------------------------------------------------------------------- #
# pyramid plan + coarser-level writes (reuse multiscale verbatim)
# --------------------------------------------------------------------------- #

def _plan_pyramid(
    canon_axes: str,
    full_shape: Tuple[int, ...],
    min_size: int,
    downsample: int,
    max_levels: Optional[int],
) -> Tuple[List[tuple], List[tuple], List[Optional[tuple]]]:
    """The anisotropic, size-aware pyramid oracle (ported from ``multiscale``).

    Returns ``(level_factors, level_shapes, per_level_factors)`` — the cumulative
    per-ZYX-axis factor per level (for the NGFF ``scale``), the full array shape
    per level, and the per-level per-axis STEP factor (for ``coarsen``).  Level 0
    is ``(1, 1, 1)`` / the full shape / ``None``.  Each ZYX axis halves
    independently only while the result stays ``>= min_size``.
    """
    from packages.epicycle_largedata.multiscale import _spatial_axis_indices

    zyx_idx = _spatial_axis_indices(canon_axes)
    ds = int(downsample)
    min_sz = int(min_size)
    lead = (full_shape[0],) if canon_axes == "CZYX" else ()

    cur_zyx = [int(full_shape[a]) for a in zyx_idx]
    level_factors: List[tuple] = [(1, 1, 1)]
    per_level_factors: List[Optional[tuple]] = [None]
    level_shapes: List[tuple] = [tuple(int(s) for s in full_shape)]
    cumulative = [1, 1, 1]
    while True:
        if max_levels is not None and len(level_shapes) >= int(max_levels):
            break
        per_level = [ds if (sz // ds) >= min_sz else 1 for sz in cur_zyx]
        if all(f == 1 for f in per_level):
            break
        cur_zyx = [sz // f if f > 1 else sz for sz, f in zip(cur_zyx, per_level)]
        cumulative = [c * f for c, f in zip(cumulative, per_level)]
        per_level_factors.append(tuple(int(f) for f in per_level))
        level_factors.append(tuple(int(c) for c in cumulative))
        level_shapes.append(lead + tuple(int(s) for s in cur_zyx))
    return level_factors, level_shapes, per_level_factors


def _write_coarser_levels(
    store: Any,
    canon_axes: str,
    cubic: tuple,
    dtype: Any,
    level_shapes: Sequence[tuple],
    per_level_factors: Sequence[Optional[tuple]],
    workers: Optional[int],
    prog: _Progress,
    compressor_obj: Any = None,
) -> None:
    """Write levels 1..N by coarsening the previous level READ BACK from the store.

    Verbatim reuse of :func:`multiscale.convert_to_ome_zarr`'s coarsen loop
    (bounded, non-overlapping graph): ``da.from_zarr(prev) → coarsen(mean) →
    astype → to_zarr``, releasing each read graph immediately.
    """
    n_levels = len(level_shapes)
    if n_levels <= 1:
        return
    import dask
    import dask.array as da

    from packages.epicycle_largedata.multiscale import (
        _clamp_chunks_to_shape,
        _coarsen_factor_map,
    )

    nw = max(1, min(int(workers) if workers else 4, os.cpu_count() or 1))
    sched = (
        {"scheduler": "synchronous"}
        if nw <= 1
        else {"scheduler": "threads", "num_workers": nw}
    )
    with dask.config.set(**sched):
        for i in range(1, n_levels):
            prev = da.from_zarr(store, component=str(i - 1))
            factor_map = _coarsen_factor_map(canon_axes, per_level_factors[i])
            nxt = da.coarsen(np.mean, prev, factor_map, trim_excess=True)
            # Brick 4b item 6: ROUND before the integer cast (a bare .astype
            # truncates and biases every coarse level ~0.5 LSB dim).  Must match
            # multiscale.convert_to_ome_zarr exactly or a processed store would
            # drift from the prepared store it mirrors.
            nxt = np.rint(nxt)
            nxt = nxt.astype(dtype).rechunk(
                _clamp_chunks_to_shape(cubic, nxt.shape))
            da.to_zarr(nxt, url=store, component=str(i), overwrite=False,
                       compressor=compressor_obj)
            del prev, nxt
            prog.advance(1)
            prog.emit(0)


# --------------------------------------------------------------------------- #
# the public entry
# --------------------------------------------------------------------------- #

def stream_process_to_store(
    src_channels: Sequence[Any],
    recipe_json: Any,
    spacing: Sequence[float],
    out_store_path: Any,
    *,
    bounds_per_channel: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    block_workers: Optional[int] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    chunk=None,
    compressor="zstd",
    min_size: int = 256,
    downsample: int = 2,
    max_levels: Optional[int] = None,
    source_fingerprint: Optional[dict] = None,
    block: Optional[int] = None,
    _force_halo: Optional[int] = None,
) -> dict:
    """Stream *src_channels* through *recipe_json* into a multiscale OME-Zarr store.

    Reads each per-channel ZYX source block-by-block DIRECTLY off its zarr array,
    runs the recipe's leading fused float32 run per block (normalize with a shared
    window → local ops → ``to_uint16``), and writes the result into a cubic-chunked
    uint16 store — never materializing the whole volume.  Then builds the coarser
    pyramid levels + writes OME-NGFF metadata, and atomically promotes the store.

    Parameters
    ----------
    src_channels:
        One array-like per channel, each **ZYX** (a zarr array — the direct-read
        fast path — a dask array, or a numpy array).  All channels share one ZYX
        shape; ``len == 1`` → a ZYX store, ``> 1`` → a CZYX store.
    recipe_json:
        The recipe (dict or :class:`recipe.Recipe`); normalized against the channel
        count.  Phase 2a streams the leading local + pointwise + normalize run; a
        CLAHE / bleach / geometry op stops the run and sets
        ``manifest["whole_array_remainder"]``.
    spacing:
        The source ``(z, y, x)`` voxel spacing (µm); used for the NGFF ``scale``
        metadata + the manifest ``new_spacing`` (Phase 2a does not regrid).
    out_store_path:
        The final store path (default a ``.zarr`` DirectoryStore).  Written via a
        unique ``.tmp`` and atomically promoted; a partial ``.tmp`` is removed on
        cancel / error — never a half-store at this path.
    bounds_per_channel:
        Optional precomputed ``(lo, hi)`` normalize window per channel.  When
        supplied the measure pass is SKIPPED (and the byte-identity gate holds it
        constant); when absent it is measured via the exact uint16 histogram.
    block_workers:
        Block-pool width (``None`` → ``min(cpu, 32)``).
    should_cancel / on_progress:
        ``() -> bool`` polled per wave (raises :class:`PreprocessingCancelled`);
        ``(done, total) -> None`` per wave over the measure + apply + coarsen budget.
    chunk / min_size / downsample / max_levels:
        Output store cubic-chunk edge + the anisotropic size-aware pyramid policy
        (ported from :func:`multiscale.convert_to_ome_zarr`).  The cubic edge
        default is **64**, matching the Prepare convert.

        Brick 3a #2 raised it to 128 for WRITE throughput (measured on the wound
        stack: apply 13.4 s -> 5.8 s, pyramid 3.8 s -> 1.2 s, since an aligned
        output block is then a whole number of chunks and the coarsener reads 8x
        fewer of them).  Brick 3c #3 REVERTED that: the store is not write-once,
        it is SCROLLED.  With CUBIC chunks and no read cache, one 2D XY slice
        touches a whole row of chunks in Z as well, so a 128 edge decompresses
        ~256 MB to paint a ~2 MB slice vs ~128 MB at 64 - it made interactive
        slice scrolling ~2x WORSE, which is the complaint that started this
        brick.  The write win returns in Brick 4b as an ANISOTROPIC
        (32, 128, 128) chunk plus an LRU read cache, the only shape that helps
        both directions at once.

        Stores written at EITHER edge open unchanged - the chunk shape is
        per-store metadata and is never assumed by the reader.
    source_fingerprint:
        Optional provenance dict embedded in the store's ``epicycle_processed``
        attrs (NOT ``epicycle_source``: a processed store must never be
        mistaken for a raw prepared source).
    block:
        Test seam (mirrors ``chunked.block_parallel_apply``'s ``block``): the exact
        per-axis output block edge, used VERBATIM.  ``None`` (production) → the
        chunk-aligned edge ``ceil(adaptive_edge / chunk) · chunk``.  An UNALIGNED
        value is the write-race negative control.
    _force_halo:
        Test seam: override the computed fused halo (the ``sum-1`` seam control).

    Returns
    -------
    dict
        ``{"out_store_path", "axes", "n_levels", "level_shapes", "chunks",
        "dtype", "new_spacing", "bounds", "stamp", "whole_array_remainder"}``.
    """
    import zarr

    from packages.epicycle_largedata.multiscale import (
        _build_multiscales,
        _clamp_chunks_to_shape,
        _cubic_chunks,
        _make_compressor,
        _remove_store,
        _store_exists,
        _zarr_store,
    )

    n = len(src_channels)
    if n < 1:
        raise ValueError("stream_process_to_store: at least one channel required")
    shape_zyx = tuple(int(s) for s in np.shape(src_channels[0]))
    if len(shape_zyx) != 3:
        raise ValueError(
            f"stream_process_to_store: each channel must be 3D ZYX (got {shape_zyx})"
        )
    for ci, ch in enumerate(src_channels):
        if tuple(int(s) for s in np.shape(ch)) != shape_zyx:
            raise ValueError(
                f"stream_process_to_store: channel {ci} shape "
                f"{tuple(np.shape(ch))} != channel 0 {shape_zyx}"
            )

    canon_axes = "CZYX" if n > 1 else "ZYX"
    full_shape: Tuple[int, ...] = (n,) + shape_zyx if n > 1 else shape_zyx
    # Brick 4b: chunk=None -> INHERIT the source store's own ZYX chunk shape, so
    # the processed store is in structural parity with the prepared store BY
    # CONSTRUCTION (no caller has to remember to pass the same value).  A source
    # without a usable ``.chunks`` (numpy / an odd duck-array) falls back to the
    # anisotropic default.
    chunk_arg = chunk if chunk is not None else _source_chunk_zyx(src_channels[0])
    chunk_shape = _cubic_chunks(canon_axes, chunk_arg)
    cubic = _clamp_chunks_to_shape(chunk_shape, full_shape)

    norm = _recipe.normalize_recipe(recipe_json, n)
    readers = [_make_block_reader(ch) for ch in src_channels]
    workers = int(block_workers) if block_workers else _MAX_BLOCK_WORKERS
    prog = _Progress(on_progress)

    def _provided_bounds(ci: int) -> Optional[Tuple[float, float]]:
        if bounds_per_channel is None:
            return None
        if ci >= len(bounds_per_channel):
            return None
        b = bounds_per_channel[ci]
        if b is None:
            return None
        try:
            return (float(b[0]), float(b[1]))
        except (TypeError, ValueError, IndexError):
            return None

    # --- planning pass: halos → aligned block edges → total block budget ------
    plans: List[dict] = []
    for ci in range(n):
        flat = _recipe.flatten_for_channel(norm, ci)
        fused_ops, halo, remainder = _fused_plan(flat, spacing=spacing)
        if _force_halo is not None:
            halo = int(_force_halo)
        # Same cancel-granularity cap as the in-memory path (Brick 3c #4): size
        # it from the source shape + the block-pool width so a streaming pass has
        # several waves and Cancel does not have to wait for the whole channel.
        _cap_kw = {"shape": getattr(src_channels[ci], "shape", None),
                   "workers": (block_workers or _MAX_BLOCK_WORKERS)}
        apply_edge = int(block) if block is not None else _aligned_block(
            _adaptive_block_edge(halo, **_cap_kw), chunk_shape[-3:]
        )
        measure_edge = int(block) if block is not None else _aligned_block(
            _adaptive_block_edge(0, **_cap_kw), chunk_shape[-3:]
        )
        need_measure = _provided_bounds(ci) is None
        plans.append(
            {
                "fused_ops": fused_ops,
                "halo": halo,
                "remainder": remainder,
                "apply_edge": apply_edge,
                "measure_edge": measure_edge,
                "need_measure": need_measure,
            }
        )

    level_factors, level_shapes, per_level_factors = _plan_pyramid(
        canon_axes, full_shape, int(min_size), int(downsample), max_levels
    )

    total_blocks = 0
    for p in plans:
        total_blocks += _count_tiles(shape_zyx, p["apply_edge"])
        if p["need_measure"]:
            total_blocks += _count_tiles(shape_zyx, p["measure_edge"])
    total_blocks += max(0, len(level_shapes) - 1)  # one tick per coarser level
    prog.total = total_blocks

    bounds_out: List[Optional[Tuple[float, float]]] = [None] * n
    remainder_any = False

    tmp_path = f"{out_store_path}.tmp.{uuid.uuid4().hex[:8]}"
    _remove_store_quiet(tmp_path)

    try:
        # Brick 4b: NEW processed stores get the nested chunk-directory layout +
        # the zstd-3 codec, matching what the Prepare convert now writes (the
        # separator is read out of each array's own .zarray, so old flat stores
        # are unaffected).
        compressor_obj = _make_compressor(compressor)
        store = _zarr_store(tmp_path, "w", dimension_separator="/")
        try:
            root = zarr.open_group(store=store, mode="w")
            z0 = root.create_dataset(
                "0", shape=full_shape, chunks=cubic, dtype="uint16",
                compressor=compressor_obj, overwrite=True
            )

            for ci in range(n):
                plan = plans[ci]
                if plan["remainder"]:
                    remainder_any = True
                b = _provided_bounds(ci)
                if b is None:
                    b = _measure_bounds(
                        readers[ci],
                        shape_zyx,
                        plan["measure_edge"],
                        workers,
                        should_cancel,
                        prog,
                    )
                bounds_out[ci] = b
                fused_fn = _make_fused_fn(plan["fused_ops"], b)
                _apply_channel(
                    z0,
                    readers[ci],
                    ci,
                    canon_axes,
                    fused_fn,
                    plan["halo"],
                    plan["apply_edge"],
                    shape_zyx,
                    workers,
                    should_cancel,
                    prog,
                )

            _write_coarser_levels(
                store,
                canon_axes,
                cubic,
                np.dtype("uint16"),
                level_shapes,
                per_level_factors,
                workers,
                prog,
                compressor_obj,
            )

            vs = _voxel_size_from_spacing(spacing)
            root.attrs["multiscales"] = _build_multiscales(
                canon_axes, Path(str(out_store_path)).stem, level_factors, vs
            )
            # A processed store carries `epicycle_processed`, NEVER
            # `epicycle_source` — so a raw-prepared-source freshness probe
            # (`read_store_source`) can never serve processed bytes AS the raw source.
            root.attrs["epicycle_processed"] = {
                "source": source_fingerprint,
                "recipe_hash": _recipe_hash(norm),
                "bounds": [list(b) if b is not None else None for b in bounds_out],
                "scale": "fixed_unit_x65535",
                "whole_array_remainder": bool(remainder_any),
            }
            # consolidate LAST, after every attrs write (same ordering rule as
            # the Prepare convert: a snapshot taken earlier would omit them)
            zarr.consolidate_metadata(store)
        finally:
            if hasattr(store, "close"):
                try:
                    store.close()
                except Exception:  # noqa: BLE001
                    pass
            gc.collect()  # release any read graphs before the promote / cleanup

        # --- atomic, Windows-safe promote ---------------------------
        if _store_exists(out_store_path):
            _remove_store(out_store_path)  # rename needs a NON-EXISTENT target
        os.rename(tmp_path, str(out_store_path))
    except BaseException:
        # cancel or error → NEVER leave a partial at the final path
        gc.collect()
        _remove_store_quiet(tmp_path)
        raise

    return {
        "out_store_path": str(out_store_path),
        "axes": canon_axes,
        "n_levels": len(level_shapes),
        "level_shapes": [tuple(int(s) for s in sh) for sh in level_shapes],
        "chunks": tuple(int(c) for c in cubic),
        "chunk_zyx": tuple(int(c) for c in cubic[-3:]),
        "compressor": (str(compressor).lower() if compressor is not None
                       else None),
        "dtype": "uint16",
        "new_spacing": tuple(float(s) for s in spacing),
        "bounds": [list(b) if b is not None else None for b in bounds_out],
        "stamp": {
            "recipe_hash": _recipe_hash(norm),
            "scale": "fixed_unit_x65535",
            "source": source_fingerprint,
        },
        "whole_array_remainder": bool(remainder_any),
    }

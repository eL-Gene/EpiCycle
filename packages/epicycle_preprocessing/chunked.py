"""Op-level bounded block-parallel primitive (Chunked-processing Phase 0-1, R1).

napari-free BY DESIGN (mirrors the other ``packages/epicycle_preprocessing`` /
``packages/*`` modules): numpy + stdlib only, no Qt / napari / dask / skimage.
The per-block op math is whatever ``op_fn`` the caller passes (the registry's
``gaussian`` / ``background`` wrappers) — this module only owns HOW the array is
traversed, never the math.

Why a manual thread-pool overlap-and-trim, NOT dask ``map_overlap``
------------------------------------------------------------------
Dask ``map_overlap`` on the
threaded scheduler has a structural 3D memory blow-up (27-neighbour fan-in, no
backpressure) and gives no clean per-block bound.  Instead we split the array
into a grid of **interior** blocks and, for each, read ``interior + halo`` (a
zero-copy VIEW into the resident array), run ``op_fn`` on that padded read, then
keep ONLY the interior (trim the halo) into a pre-allocated output.  A
``ThreadPoolExecutor`` runs the blocks across cores — scipy/skimage release the
GIL inside their C loops, so threads scale (same idiom as ``ops.py::_parallel_map``).
Only ``max_workers`` blocks are ever mid-flight, so peak scratch is
``max_workers × (block + 2·halo)ᴺ × op_scratch`` — file-size-independent.

Byte-identity (the training-data safety, R3/R4)
-----------------------------------------------
For a **local** op whose output at a voxel depends only on a bounded
neighbourhood of radius ``≤ halo``, every kept (interior) voxel saw its full
real neighbourhood → the trimmed result is **bit-identical** (``np.array_equal``,
not ``allclose``) to running ``op_fn`` on the whole array.  The one requirement
is the boundary contract at the TRUE array edge:

* ``boundary="none"`` (the only mode used + mandated by R3) — at the true edge
  we do NOT pad the read; we pass the TRUNCATED real read (clamped to ``[0, n]``)
  to ``op_fn`` exactly as the whole-array call sees its own edge, so scipy /
  skimage applies ITS native border internally.  Empirically edge-identical for
  gaussian (scipy ``reflect``) + white-tophat (skimage) — verified ``max|Δ|=0.0``
  over the FULL array incl. edge voxels.

The correct halo is the op's TRUE reach (gaussian ``int(4σ+0.5)``; white-tophat
``2·radius``) — an undersized halo seams (a FAILED ``array_equal`` test), which
is the whole point of the gate.  This module does not know the reach; the caller
passes ``halo`` from the registry op's ``halo(**params)``.
"""

from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

from .exceptions import PreprocessingCancelled

logger = logging.getLogger(__name__)


#: Cap the block pool so it saturates cores without oversubscribing.  The
#: whole-volume process runs CHANNELS serially (``execute(max_workers=1)``) and
#: leans on THIS pool for its multi-core speed, so it may use all cores.
_MAX_BLOCK_WORKERS: int = min(os.cpu_count() or 1, 32)

#: Below this many voxels, chunking overhead isn't worth it — run ``op_fn`` whole
#: (still bit-identical; it IS the op's own whole-array call).  ~1 Mvox ≈ 100³.
_SMALL_ARRAY_VOXELS: int = 1 << 20

#: Block-edge clamps.  ``block_edge = clip(round(6·halo), _MIN_BLOCK, _MAX_BLOCK)``
#: keeps the halo-overhead volume ratio ``((block+2·halo)/block)ᴺ`` modest AND
#: amortises the per-block Python round-trip.  The ``6·halo`` rule alone bounds
#: the ratio but says nothing about block COUNT: at halo=6 it gives 64³ blocks,
#: i.e. 768 tiles/channel on a 143×1024×1024 stack, and the measured cost is
#: dominated by that overhead + the 1.67× halo re-read, not by the filter
#: (measured: block 64→256 halves the apply).
#: So the FLOOR is ``_MIN_BLOCK=256`` (ratio at halo=6 → (268/256)³≈1.15×) with
#: ``6·halo`` retained as the LOWER bound for large halos and ``_MAX_BLOCK=512``
#: still capping the extreme ones.
#:
#: RAM: a 256³ tile is ~4.6× the scratch of a 64³ one, so the floor MUST be read
#: through :func:`ops.max_local_tile_scratch_bytes` (it calls
#: :func:`_adaptive_block_edge`) by the whole-volume caller's
#: ``_derive_block_workers`` — a small box then runs with FEWER workers instead
#: of blowing the budget.  That wiring already exists; do not bypass it.
_MIN_BLOCK: int = 256
_MAX_BLOCK: int = 512

#: Wave-submit width: never queue the whole tile grid at once (a 35GB gaussian is
#: ~10^4-10^5 tiles → tens of MB of Future objects + a slow cancel_futures sweep).
#: Keep only ``workers × this`` tiles outstanding, re-checking cancel each wave —
#: bounds the Future burst AND the cancel-response latency to about one wave.
_INFLIGHT_PER_WORKER: int = 4


#: Cancel granularity (Brick 3c #4).  The wave-submit loop keeps
#: ``workers x _INFLIGHT_PER_WORKER`` tiles outstanding and re-checks cancel once
#: per completed wave, so cancel latency is about ONE wave - but only if the grid
#: actually HAS several waves.  At the 256 floor a 143x1024x1024 pass is a 1x4x4
#: grid = 16 tiles, fewer than the 80 in-flight slots, so every tile is submitted
#: at once, there is nothing for ``cancel_futures`` to drop, and Cancel waits for
#: the WHOLE pass (~4.5 s per channel).  So the adaptive edge is additionally
#: capped so the largest axis is cut into at least ``_WAVES_WANTED x workers``
#: pieces.  This is a SOFT cap: it never goes below :data:`_MIN_TILE_FOR_WAVES`,
#: which is where the per-block overhead the 256 floor exists to amortise starts
#: to come back (measured: most of the block-floor win survives at 128).
_WAVES_WANTED: int = 3
_MIN_TILE_FOR_WAVES: int = 128


def _adaptive_block_edge(halo: int, *, shape=None, workers: int | None = None) -> int:
    """The default per-axis block edge for *halo* (see :data:`_MIN_BLOCK`).

    With *shape* and *workers* the result is additionally capped for cancel
    granularity (see :data:`_WAVES_WANTED`).  WITHOUT them it returns the
    uncapped value - which is what :func:`ops.max_local_tile_scratch_bytes`
    wants, since a RAM budget must use the LARGEST edge the engine could pick
    (over-estimating the tile only makes the caller pick fewer workers, which is
    safe; under-estimating it would blow the budget).
    """
    halo_edge = int(min(_MAX_BLOCK, round(6.0 * float(halo))))
    edge = int(min(_MAX_BLOCK, max(_MIN_BLOCK, halo_edge)))
    if shape is None or workers is None:
        return edge
    try:
        largest = max(int(d) for d in shape)
        w = max(1, int(workers))
    except (TypeError, ValueError):     # unusable shape/workers -> uncapped
        return edge
    if largest <= 0:
        return edge
    wanted = -(-largest // (_WAVES_WANTED * w))                # ceil
    capped = min(edge, max(_MIN_TILE_FOR_WAVES, wanted))
    # The 6*halo LOWER bound survives the cap: it is what keeps the halo-overhead
    # volume ratio ((block+2*halo)/block)^3 sane, and a large-halo op (background
    # r=15 -> halo 30 -> 180) would be crippled by a 128 tile.  Cancel latency is
    # a UX property; the halo ratio is a cost property - the cost wins.
    return int(max(1, capped, halo_edge))


def _axis_blocks(n: int, block: int, halo: int, arr_len: int) -> List[Tuple[slice, slice, slice]]:
    """Per-axis ``(out_slice, read_slice, trim_slice)`` tiling of one axis.

    ``out`` slices tile ``[0, n)`` disjointly (interior).  ``read`` extends each
    interior by ``halo`` on both sides, CLAMPED to ``[0, arr_len)`` at the true
    edge (``boundary="none"``).  ``trim`` selects the interior back out of the
    ``op_fn(read)`` result (offset = how far the read reached before the interior).
    """
    out: List[Tuple[slice, slice, slice]] = []
    start = 0
    while start < n:
        stop = min(start + block, n)
        r0 = max(0, start - halo)
        r1 = min(arr_len, stop + halo)
        t0 = start - r0
        t1 = t0 + (stop - start)
        out.append((slice(start, stop), slice(r0, r1), slice(t0, t1)))
        start = stop
    return out


def block_parallel_apply(
    op_fn: Callable[[np.ndarray], np.ndarray],
    arr: np.ndarray,
    halo: int,
    *,
    boundary: str = "none",
    max_workers: int | None = None,
    block: int | None = None,
    small_size: int = _SMALL_ARRAY_VOXELS,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Apply a **local** *op_fn* to *arr* block-parallel, byte-identical to whole.

    *op_fn* is a single-arg callable ``op_fn(read) -> ndarray`` (same shape as its
    input) — a local neighbourhood op whose reach is ``≤ halo`` voxels.  *arr* is
    a resident numpy array (ZYX or any ndim).  *halo* is the op's TRUE reach.

    Returns a NEW float32 array, equal (``np.array_equal``) to ``op_fn(arr)`` run
    on the whole array — provided ``halo`` covers the op's real reach.  Peak extra
    RAM is ``max_workers × (block+2·halo)ᴺ × op_scratch`` (bounded, file-size
    independent).  Pure — no Qt / napari / I/O.

    * *boundary* — only ``"none"`` is supported (R3): the true-edge read is the
      truncated real array, matching the whole-array call's own border.
    * *max_workers* — block-pool width (``None`` → ``min(cpu, 32)``).  This is
      SEPARATE from ``execute``'s channel ``max_workers`` (channels stay serial;
      this pool saturates cores per op — no oversubscription).
    * *block* — explicit per-axis block edge (overrides the adaptive default;
      used by tests to force multi-block tiling on a small array).  ``None`` →
      ``clip(round(6·halo), _MIN_BLOCK, _MAX_BLOCK)`` = ``clip(…, 256, 512)``.
    * *small_size* — arrays below this voxel count run ``op_fn`` whole (no chunk
      overhead).  Bypassed when *block* is given explicitly (test rigor).
    * *should_cancel* — optional ``() -> bool`` polled ONCE up front and once per
      wave of completed tiles.  When it returns True the pending tiles are dropped
      (``shutdown(cancel_futures=True)``), the ≤ ``workers`` in-flight ones finish,
      and :class:`PreprocessingCancelled` (a ``BaseException``) is raised so it
      propagates through the pipeline's ``except Exception`` guards.  Cancel
      latency ≈ one wave (a running tile can't be interrupted mid-call).  A plain
      cross-thread bool read is fine under CPython's GIL (py312).
    * *on_progress* — optional ``(done, total) -> None`` called once per wave with
      the running tile count (``total`` constant, ``done`` monotonic to ``total``).
      MUST be emit-only + self-throttling at the source (it runs on the worker
      thread; never touch a widget here).

    Falls back to a single whole-array ``op_fn(arr)`` call — still bit-identical —
    when there is no neighbourhood (``halo <= 0``), the array is 0-d, the array is
    below *small_size* (adaptive path only), or a single block already spans every
    axis.  Any of those makes chunking pointless, not wrong.
    """
    if boundary != "none":
        raise ValueError(
            f"block_parallel_apply: only boundary='none' is supported (got {boundary!r})"
        )

    def _cancelled() -> bool:
        return should_cancel is not None and should_cancel()

    def _report(done: int, total: int) -> None:
        # Best-effort: a raising progress callback must NOT propagate into the
        # caller's `except Exception` and silently skip the op (it's emit-only per
        # the contract, but guard it so a bug can't corrupt the training-data path).
        if on_progress is not None:
            try:
                on_progress(done, total)
            except Exception:
                pass

    if _cancelled():                       # fast early-out before any allocation
        raise PreprocessingCancelled("cancelled before start")

    a = np.asarray(arr, dtype=np.float32)
    h = int(halo)

    def _whole() -> np.ndarray:
        out = np.asarray(op_fn(a), dtype=np.float32)
        _report(1, 1)
        return out

    if a.ndim == 0 or h <= 0:
        return _whole()

    explicit_block = block is not None
    # `workers` is not known yet (it depends on the tile count, which depends on
    # the edge), so size the cancel-granularity cap with the pool WIDTH the caller
    # asked for - the upper bound on how many tiles can be in flight.
    edge = int(block) if explicit_block else _adaptive_block_edge(
        h, shape=a.shape, workers=(max_workers or _MAX_BLOCK_WORKERS))
    edge = max(1, edge)

    # Whole-array short-circuits (still bit-identical — it's the op's own call).
    # These are a single opaque op call: not interruptible mid-call (the caller's
    # between-op cancel check is the seam), which is fine for a below-gate array.
    if not explicit_block and int(a.size) < int(small_size):
        return _whole()
    if all(int(d) <= edge for d in a.shape):
        return _whole()

    # Per-axis (out, read, trim) tilings → the Cartesian product is the block grid.
    axis_tilings = [_axis_blocks(int(d), edge, h, int(d)) for d in a.shape]
    out = np.empty_like(a, dtype=np.float32)

    def _do_block(spec: Sequence[Tuple[slice, slice, slice]]) -> None:
        out_sl = tuple(s[0] for s in spec)
        read_sl = tuple(s[1] for s in spec)
        trim_sl = tuple(s[2] for s in spec)
        res = np.asarray(op_fn(a[read_sl]), dtype=np.float32)
        out[out_sl] = res[trim_sl]  # disjoint interior tile — thread-safe write

    specs = list(itertools.product(*axis_tilings))
    total = len(specs)
    workers = min(max_workers or _MAX_BLOCK_WORKERS, total)
    done = 0
    _report(0, total)

    if workers <= 1:
        # Serial (1-core box, or a RAM-forced width): check cancel BETWEEN tiles so
        # even the single-thread path stops promptly.
        for spec in specs:
            if _cancelled():
                raise PreprocessingCancelled("cancelled between tiles")
            _do_block(spec)
            done += 1
            _report(done, total)
        return out

    # Explicit pool — NOT ``with ThreadPoolExecutor``: its ``__exit__`` runs
    # ``shutdown(wait=True)`` WITHOUT ``cancel_futures``, so raising inside the
    # ``with`` would block the unwind until EVERY pending tile drains (the exact
    # "Cancel can't stop it" bug this rebuild fixes).  Manage the pool explicitly
    # and ALWAYS ``shutdown(wait=True, cancel_futures=True)`` in ``finally`` so on
    # cancel the not-yet-started tiles are dropped and only the ≤ ``workers``
    # in-flight ones finish — bounded, clean teardown.
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        it = iter(specs)
        inflight: set = set()
        # Wave-submit: keep only ``workers × _INFLIGHT_PER_WORKER`` tiles
        # outstanding (never the whole grid), re-checking cancel each wave.
        max_inflight = max(1, workers * _INFLIGHT_PER_WORKER)
        for _ in range(max_inflight):
            spec = next(it, None)
            if spec is None:
                break
            inflight.add(pool.submit(_do_block, spec))
        while inflight:
            finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                fut.result()          # surface a per-tile exception (no partial out)
                done += 1
            _report(done, total)
            if _cancelled():
                raise PreprocessingCancelled("cancelled between tiles")
            for _ in range(len(finished)):   # refill the wave
                spec = next(it, None)
                if spec is None:
                    break
                inflight.add(pool.submit(_do_block, spec))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return out

"""TIFF -> cubic-chunked multiscale OME-Zarr conversion + a lazy reader.

The "keystone" of the Step 1 large-data pipeline (Brick 1.2): turn a big TIFF into a
multiscale OME-NGFF store **without materializing the whole volume**, and open it
back as a **lazy multiscale array** napari can consume via
``add_image(data=levels, multiscale=True)``.

napari-free BY DESIGN (see the package docstring).  The heavy libraries
(``dask``, ``tifffile``, ``zarr``) are deferred-imported inside the functions,
mirroring :mod:`packages.epicycle_largedata.lazy`.

Design choices worth knowing
----------------------------
* **Cubic chunks** ``(1, chunk, chunk, chunk)`` (CZYX) / ``(chunk, chunk,
  chunk)`` (ZYX).  This is the *axial-lag fix*: EpiCycle curates XZ/YZ views in
  Steps 2/3, and slice-thick TIFF chunks force decoding whole Z-columns for an
  axial slice.  Cubic chunks make all three orientations cheap.
* **Streaming write** via ``dask.array.to_zarr`` (chunk-by-chunk).  Peak RAM
  depends on how the source is opened:

  - the **memmap (uncompressed-TIFF)** path reads the source one cubic chunk at
    a time (see :class:`_LazyReadProxy`), so peak is bounded to ``num_workers`` x
    a cubic chunk -- never a whole level.  This is the acceptance gate.
  - the **compressed** path (``open_lazy`` -> a dask array chunked by TIFF
    *pages*, typically thin-Z ``(1, Y, X)``, then rechunked cubic) must buffer a
    **Z-slab** of ~``chunk * Y * X`` to assemble cubic chunks out of the thin
    pages.  That is still bounded (flat in *total Z* -- it does NOT grow with
    depth) but it grows with the XY field, so a thin-Z / whole-tissue-XY
    *compressed* source can buffer a large slab.  (Whole-tissue XY is a Step-5
    concern, out of scope here.)

  **Source is read exactly ONCE** (OPT 1): only *level 0* streams from the
  on-disk source; every coarser level is coarsened from the *previous level read
  back from the store* (much smaller), not re-derived from the source.  The
  earlier design re-read (and re-decompressed) the whole source once PER LEVEL --
  ~N x the I/O, brutal at tens/hundreds of GB.  **Bounded parallelism** (OPT 2):
  ``num_workers`` (default ``min(4, os.cpu_count())``) caps a threaded scheduler
  so read/decompress/coarsen/write overlap; ``num_workers == 1`` falls back to
  the synchronous scheduler and reproduces the original deterministic
  single-chunk-in-flight RAM guarantee bit-for-bit.  Threading changes no output
  bytes.
* **Anisotropic, size-aware mean-downsample pyramid** via ``dask.array.coarsen``.
  Each spatial (ZYX) axis is halved *independently*, and only while the result
  stays ``>= min_size`` -- so a thin-Z wide-XY stack downsamples XY without
  vanishing its already-thin Z (the old isotropic policy stopped the whole
  pyramid the moment the smallest axis could not halve, giving such a stack
  almost no overview).  Each coarser level is built lazily by coarsening the
  previous level *read back from the store* (never ``.compute()``-d whole, and so
  the on-disk source is read exactly once -- see the "Streaming write" note
  above).  ``coarsen`` returns float64, so
  each coarser level is cast back to the source dtype -- a **truncation toward
  zero** of the block mean; level 0 is byte-exact, only coarser levels round.
* **Store handling** ported from the scaled-seg engine's ``utils/io.py``: a
  ``.zip`` path uses a (GPFS/Lustre-safe, single-file) ``ZipStore``; anything
  else a ``DirectoryStore``.  The HPC footgun a ``DirectoryStore`` on networked
  scratch causes (one inode per chunk -> inode storm) is why ``.zip`` exists.
* **OME-NGFF v0.4** ``multiscales`` metadata is written **by hand** as JSON into
  the group's ``.zattrs`` (``ome-zarr`` is intentionally not a dependency).

The axis-resolution + zarr conventions are shared (by PORTING, not importing)
with the standalone scaled-seg engine so the two repos stay in parity.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from .config import ALLOWED_COMPRESSORS, DEFAULT_CHUNK_ZYX, DEFAULT_COMPRESSOR

_LOG = logging.getLogger(__name__)

#: Default bound (bytes) for the per-open LRU read cache -- **0 = OFF**.
#:
#: Brick 4b first shipped this at 256 MiB on the strength of a RAW-READ
#: benchmark (zarr slices in a bare loop), where it looked like a 1.7x win.
#: Measuring the thing users actually do -- scrolling Z IN NAPARI, fresh process
#: per case, level 0, 40 steps -- reversed the sign:
#:
#:     store                      no cache        256 MiB LRU
#:     old cubic 64 + lz4          45 ms/step      75 ms/step
#:     new (32,128,128) + zstd     36 ms/step      45 ms/step
#:
#: The OS page cache already keeps the chunk files resident, so the Python-level
#: cache only adds bookkeeping and a SECOND copy of bytes that are already in
#: RAM; and because it stores COMPRESSED blocks, it never skips the
#: decompression that actually dominates.  It is kept as an OPT-IN for callers on
#: genuinely slow or remote storage (a network mount, object storage), where
#: avoiding the re-fetch does pay.
#:
#: NOTE the ceiling is PER STORE OPEN, not global: N concurrently open stores can
#: hold N x cache_bytes.  Another reason not to switch it on by default.
DEFAULT_READ_CACHE_BYTES = 0


# ---------------------------------------------------------------------------
# lazy memmap wrapper -- avoids da.from_array reading the whole volume
# ---------------------------------------------------------------------------

class _LazyReadProxy:
    """A thin, chunk-lazy wrapper around a numpy ``memmap`` (or a strided view
    of one) so ``dask.array.from_array`` reads it **one cubic chunk at a time**
    instead of materializing the whole volume.

    Two footguns in ``da.from_array`` this dodges (both would read the entire
    source into RAM -- verified against dask 2026.7.1):

    * a **defensive copy** -- ``from_array`` runs ``x = x.copy()`` on any
      arraylike exposing ``.copy()``, and ``memmap.copy()`` reads every byte.
      This proxy deliberately does NOT expose ``copy``.
    * a **content-hash tokenize** -- ``from_array`` names the graph by hashing
      the input's *contents* unless an explicit ``name=`` is passed; hashing a
      big array reads it all.  Callers pass ``name=`` (see :func:`_lazy_source`).

    ``__array_ufunc__ = None`` is a classification marker so dask's
    ``is_arraylike`` recognises the proxy as a duck-array (it needs a
    ``__array_function__`` / ``__array_ufunc__`` attribute); it is never actually
    invoked, because dask slices the proxy into real ndarrays before any op.

    INVARIANT (do not break): this class MUST NEVER expose a ``.copy`` attribute
    (method or otherwise).  ``da.from_array`` runs ``x = x.copy()`` on any
    arraylike advertising ``.copy()``, and ``memmap.copy()`` reads every byte of
    the source into RAM -- exactly the whole-volume read this proxy exists to
    avoid.  A ``test_lazyproxy_never_exposes_copy`` guards the invariant.
    """

    __array_ufunc__ = None

    def __init__(self, arr):
        self._arr = arr
        self.shape = tuple(int(s) for s in arr.shape)
        self.dtype = arr.dtype
        self.ndim = int(arr.ndim)

    def __getitem__(self, key):
        # a memmap (or memmap-view) slice reads only the requested bytes;
        # np.asarray makes it a small, contiguous in-RAM chunk for dask.
        return np.asarray(self._arr[key])


def _lazy_source(canon, cubic):
    """A cubic-chunked lazy dask array over ``canon`` (the canonicalized source).

    ``canon`` is either an already-lazy dask array (compressed TIFF / zarr, from
    :func:`open_lazy`) -- rechunked cubic -- or a numpy ``memmap`` / strided view
    (uncompressed TIFF) -- wrapped so it is read chunk-by-chunk with an explicit,
    unique ``name`` (see :class:`_LazyReadProxy`).
    """
    import dask.array as da

    if isinstance(canon, da.Array):
        return canon.rechunk(cubic)
    return da.from_array(
        _LazyReadProxy(canon), chunks=cubic,
        name="epicycle-ms-src-" + uuid.uuid4().hex,
    )


# ---------------------------------------------------------------------------
# store handling (ported from epicycle-scaledseg-engine/epicycle/utils/io.py)
# ---------------------------------------------------------------------------

def _zarr_store(path, mode: str, dimension_separator=None):
    """Open the right zarr store for ``path``: a ``ZipStore`` for a ``.zip``
    (GPFS-safe, a single file), a ``DirectoryStore`` otherwise.

    Ported verbatim from the scaled-seg engine's ``utils/io.py`` so the two
    repos share store conventions.  ``DirectoryStore`` ignores ``mode`` (it is
    stateless -- one OS file per chunk); ``ZipStore`` honours it (``"w"`` to
    write, ``"r"`` to read).
    """
    import zarr  # deferred heavy import

    p = str(path)
    if p.lower().endswith(".zip"):
        return zarr.ZipStore(p, mode=mode)
    if mode == "w" and dimension_separator is not None:
        # NEW stores only.  A NESTED separator puts chunk files in per-axis
        # subdirectories (``0/12/3/4``) instead of one flat directory holding
        # every chunk (``0/12.3.4``).  On NTFS a flat directory of ~130k entries
        # makes every chunk open a slow directory scan; nested keeps each
        # directory small.  READS never pass this: zarr reads the separator out
        # of each array's own ``.zarray``, so flat legacy stores open unchanged.
        return zarr.DirectoryStore(p, dimension_separator=dimension_separator)
    return zarr.DirectoryStore(p)


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------

def _read_tiff_axes(src_path) -> Optional[str]:
    """The tifffile ``series.axes`` string for a TIFF source, else ``None``
    (a zarr source, or an unreadable / non-TIFF path)."""
    p = str(src_path).lower()
    if not p.endswith((".tif", ".tiff")):
        return None
    try:
        import tifffile

        with tifffile.TiffFile(str(src_path)) as tf:
            return tf.series[0].axes
    except Exception:
        return None


def _cached_store(store, cache_bytes=DEFAULT_READ_CACHE_BYTES):
    """Wrap *store* in a bounded ``zarr.LRUStoreCache``, or return it unchanged.

    OFF BY DEFAULT (:data:`DEFAULT_READ_CACHE_BYTES` is 0) -- measured in napari,
    it made scrolling WORSE, because the OS page cache already serves the chunk
    files.  Opt in only for slow / remote storage.

    IMPORTANT, and the reason the original rationale for this was wrong: an
    ``LRUStoreCache`` caches the store's raw values, i.e. the **COMPRESSED**
    chunk blocks.  It saves the re-READ of a chunk, never the re-DECOMPRESSION --
    zarr decompresses on every access regardless.  Since decompression is what
    dominates a slice read here, the ceiling on what this can buy is small, and
    on local disk the OS page cache has already removed the part it does help
    with.

    ``cache_bytes <= 0`` (or ``None``) returns *store* untouched.

    When enabled it is a MEMORY bound, not a semantic change: the cache is
    read-only and bounded, so a ROI crop still reads only the chunks that ROI
    touches -- the lazy-extract "bounded read" invariant is unaffected.  The
    bound is PER STORE OPEN, not global.
    """
    if not cache_bytes or int(cache_bytes) <= 0:
        return store
    import zarr

    return zarr.LRUStoreCache(store, max_size=int(cache_bytes))


def _make_compressor(name=DEFAULT_COMPRESSOR):
    """The numcodecs compressor for *name*.

    * ``"zstd"`` -- Blosc zstd clevel 3 + byte SHUFFLE.  The Brick 4b default
      for NEW stores: it packs a 16-bit microscopy volume far smaller than
      zarr's stock lz4-clevel-5 (4.5x on the wound stack), and less on-disk data
      is less to read per 2D slice.
    * ``"lz4"``  -- Blosc lz4 clevel 3 + SHUFFLE, the "decode speed above all"
      alternative.
    * ``"default"`` -- zarr's OWN stock codec, i.e. exactly what every store
      written before Brick 4b used.  Kept so the legacy layout stays
      reproducible (the legacy round-trip tests pin it).
    * ``None`` -- genuinely UNCOMPRESSED.  This is a deliberate, explicit
      option, NOT a "let zarr choose" sentinel: zarr treats an explicit
      ``compressor=None`` as "no codec", so using it as a stand-in for the
      default silently writes a ~4x larger store (it produced a wrong baseline
      during this brick's own benchmarking).

    Reading is unaffected by all of this -- the codec is recorded per array in
    ``.zarray``, so any store opens under any setting.
    """
    if name is None:
        return None
    key = str(name).lower()
    if key not in ALLOWED_COMPRESSORS:
        raise ValueError(
            f"_make_compressor: unknown compressor {name!r}; expected one of "
            f"{ALLOWED_COMPRESSORS} (or None for genuinely uncompressed)"
        )
    if key == "default":
        import zarr.storage

        return zarr.storage.default_compressor
    from numcodecs import Blosc

    return Blosc(cname=key, clevel=3, shuffle=Blosc.SHUFFLE)


def _cubic_chunks(canon_axes: str, chunk):
    """Store chunk tuple for the canonical layout (channel axis chunked at 1).

    *chunk* is either the LEGACY cubic int (``N`` -> ``(N, N, N)``, what every
    store written before Brick 4b used) or a ``(cz, cy, cx)`` tuple (the new
    ANISOTROPIC default ``(32, 128, 128)``: thin in Z so a 2D slice decompresses
    little unread depth, big in-plane so it touches few chunks).  The name is
    kept because it is imported by ``epicycle_preprocessing.streaming``.
    """
    if isinstance(chunk, bool):
        raise ValueError(f"_cubic_chunks: chunk must be an int or a (cz, cy, cx)"
                         f" tuple (got {chunk!r})")
    if isinstance(chunk, int):
        if chunk < 1:
            raise ValueError(f"_cubic_chunks: chunk must be >= 1 (got {chunk!r})")
        zyx = (int(chunk),) * 3
    else:
        try:
            vals = [int(v) for v in chunk]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"_cubic_chunks: chunk must be an int or a (cz, cy, cx) tuple "
                f"(got {chunk!r})") from exc
        if len(vals) != 3 or any(v < 1 for v in vals):
            raise ValueError(
                f"_cubic_chunks: chunk tuple must be 3 positive ints "
                f"(got {chunk!r})")
        zyx = tuple(vals)
    if canon_axes == "CZYX":
        return (1,) + zyx
    if canon_axes == "ZYX":
        return zyx
    raise ValueError(f"_cubic_chunks: unsupported axes {canon_axes!r}")


def _clamp_chunks_to_shape(chunks, shape):
    """Clip each chunk axis to the level's extent (never below 1).

    zarr accepts a chunk larger than the array, but it wastes a whole padded
    chunk of storage per axis and the progressive reader then fetches far more
    than the level holds.  A coarse pyramid level can easily be smaller than a
    ``(32, 128, 128)`` chunk, so every level's chunks are clamped to its shape.
    """
    return tuple(max(1, min(int(c), int(s))) for c, s in zip(chunks, shape))


def _spatial_axis_indices(canon_axes: str):
    """Indices of the (Z, Y, X) axes in the canonical array."""
    return (1, 2, 3) if canon_axes == "CZYX" else (0, 1, 2)


def _coarsen_factor_map(canon_axes: str, per_axis_factors) -> dict:
    """``da.coarsen`` factor map for one pyramid step from a per-ZYX-axis factor
    tuple ``(fz, fy, fx)`` (each ``1`` = keep that axis, ``downsample`` = halve
    it).  The channel axis is always ``1`` (never pooled)."""
    fz, fy, fx = (int(f) for f in per_axis_factors)
    if canon_axes == "CZYX":
        return {0: 1, 1: fz, 2: fy, 3: fx}
    return {0: fz, 1: fy, 2: fx}


def _ngff_axes(canon_axes: str, voxel_size=None):
    """OME-NGFF ``axes`` list (name + type) for the canonical layout.

    When ``voxel_size`` is given, each SPACE axis carries ``"unit":
    "micrometer"`` (the channel axis never does); when ``None`` no unit field is
    emitted (the physical-spacing-unknown default)."""
    def space(n):
        d = {"name": n, "type": "space"}
        if voxel_size is not None:
            d["unit"] = "micrometer"
        return d

    if canon_axes == "CZYX":
        return [{"name": "c", "type": "channel"},
                space("z"), space("y"), space("x")]
    return [space("z"), space("y"), space("x")]


def _build_multiscales(canon_axes: str, name: str, level_factors, voxel_size=None):
    """Build the OME-NGFF v0.4 ``multiscales`` value (a list with one block).

    ``level_factors`` is the per-level list of **per-ZYX-axis cumulative
    downsample factors** ``(fz, fy, fx)`` (level 0 is ``(1, 1, 1)``); it is
    computed anisotropically by the converter so a clamped axis keeps a factor
    of ``1`` across levels.

    Per-level ``scale``: the channel axis stays ``1.0``; each ZYX axis is
    ``(voxel_size[axis] if voxel_size else 1.0) * cumulative_factor[axis]``.  A
    downsampled axis therefore increases across levels (coarser = larger scale,
    the ordering a reader uses); a clamped axis stays flat.  With
    ``voxel_size`` the scale is in micrometers, otherwise in (dimensionless)
    voxel units.

    NOTE: the scale is the cumulative **block factor** (``downsample ** level``
    for an unclamped axis), NOT the exact shape ratio.  For an extent that is not
    a power of ``downsample``, ``coarsen(trim_excess=True)`` drops the far-edge
    remainder, so the real shape ratio drifts slightly (e.g. 65 -> 32 is a 2.03x
    ratio but a 2.0 block factor).  This is the ecosystem-standard, origin-
    anchored convention (ome-zarr-py does the same; napari aligns levels by
    origin), so no ``translation`` transform is needed.
    """
    axes_meta = _ngff_axes(canon_axes, voxel_size)
    vz, vy, vx = (voxel_size if voxel_size is not None else (1.0, 1.0, 1.0))
    datasets = []
    for i, (fz, fy, fx) in enumerate(level_factors):
        sz, sy, sx = float(vz) * fz, float(vy) * fy, float(vx) * fx
        scale = [1.0, sz, sy, sx] if canon_axes == "CZYX" else [sz, sy, sx]
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [{"type": "scale", "scale": scale}],
        })
    return [{
        "version": "0.4",
        "name": str(name),
        "axes": axes_meta,
        "datasets": datasets,
    }]


def _store_exists(path) -> bool:
    return Path(path).exists()


def _remove_store(path) -> None:
    """Delete an existing store: ``rmtree`` a DirectoryStore dir, ``unlink`` a
    ``.zip`` file."""
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


def _write_result_manifest(out_path, manifest: dict) -> Path:
    """Write ``result_manifest.json`` next to the store (the engine's mode-
    agnostic ingest-contract convention).  Tuples serialize as JSON arrays.

    NOTE: this assumes one output store per parent directory (the engine's
    "one run dir per result" convention) -- two conversions into the same parent
    would overwrite each other's sidecar.  The returned dict is authoritative;
    the sidecar is a convenience for downstream ingest.
    """
    target = Path(out_path).parent / "result_manifest.json"
    with open(target, "w") as f:
        json.dump(manifest, f, indent=2)
    return target


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

def _check_cancel(should_cancel) -> None:
    """Poll the caller's cancel predicate; raise ``ConvertCancelled`` if set.

    A predicate that itself raises is treated as "not cancelled" (a broken probe
    must not masquerade as a user cancel, nor abort a healthy convert).
    """
    if should_cancel is None:
        return
    try:
        wants_stop = bool(should_cancel())
    except Exception:                          # noqa: BLE001 - broken probe
        return
    if wants_stop:
        from .exceptions import ConvertCancelled

        raise ConvertCancelled("conversion cancelled by the caller")


def _report_level(on_level, done: int, total: int) -> None:
    """Call the caller's progress callback, swallowing anything it raises.

    Progress reporting is decoration: a broken callback (a disconnected Qt
    signal, a torn-down widget) must never turn a healthy multi-GB conversion
    into a failed one.
    """
    if on_level is None:
        return
    try:
        on_level(int(done), int(total))
    except Exception:                          # noqa: BLE001 - broken callback
        pass


def convert_to_ome_zarr(src_path, out_path, *, chunk=DEFAULT_CHUNK_ZYX,
                        compressor=DEFAULT_COMPRESSOR,
                        max_levels: Optional[int] = None, min_size: int = 256,
                        downsample: int = 2, image_axes: Optional[str] = None,
                        voxel_size: Optional[tuple] = None,
                        overwrite: bool = False,
                        num_workers: Optional[int] = None,
                        should_cancel=None, on_level=None) -> dict:
    """Convert a big TIFF (or zarr) into a cubic-chunked multiscale OME-Zarr
    store WITHOUT materializing the whole volume.

    The source is opened LAZY (:func:`packages.epicycle_largedata.lazy.open_lazy`
    -- memmap for an uncompressed TIFF, dask for a compressed / zarr source),
    canonicalized to **CZYX** (4D) or kept as **ZYX** (3D single-channel), rechunked
    cubic, mean-downsampled into an **anisotropic, size-aware** pyramid, and
    streamed level-by-level into a zarr group with hand-written OME-NGFF v0.4
    metadata.

    Parameters
    ----------
    src_path:
        A ``.tif`` / ``.tiff`` (or ``.zarr``) source.
    out_path:
        Output store path -- ``.zip`` -> a single-file ``ZipStore`` (GPFS-safe),
        anything else -> a ``DirectoryStore`` directory.
    chunk:
        Cubic chunk edge length for the ZYX axes (channel axis chunked at 1).
    max_levels:
        Optional cap on the number of pyramid levels (``None`` = unbounded).
    min_size:
        Per-axis floor.  At each level a ZYX axis is halved **only if** the
        result would stay ``>= min_size``; otherwise that axis is clamped (kept
        at its current size for this and all coarser levels).  The pyramid stops
        when NO ZYX axis can be halved any further (or ``max_levels`` caps first)
        -- so a thin-Z stack still gets a full XY overview.
    downsample:
        Per-level per-axis downsample factor (``>= 2``); mean-pooling via coarsen.
    image_axes:
        Explicit on-disk axis order (``"CZYX"``/``"ZCYX"``/``"ZYX"``) for the
        irreducibly-ambiguous case; overrides shape/metadata detection.  REQUIRED
        for a 4D source whose axis order cannot be resolved confidently from
        metadata (the Step 1 UI passes it after user confirmation).
    voxel_size:
        Optional physical voxel spacing ``(z, y, x)`` in micrometers (order
        matches the spatial axes).  When given, per-level NGFF ``scale`` values
        are ``voxel_size * cumulative_downsample_factor`` per axis and each SPACE
        axis is tagged ``"unit": "micrometer"``.  ``None`` (default) leaves the
        scales in dimensionless voxel units with no unit field.
    overwrite:
        Refuse to clobber an existing store unless ``True``.
    should_cancel:
        Optional zero-arg predicate polled BEFORE each pyramid level is written.
        When it returns truthy the convert raises
        :class:`packages.epicycle_largedata.exceptions.ConvertCancelled`, whose
        propagation runs the ``finally`` below -- so the half-written store is
        removed and the ``epicycle_source`` freshness marker (written only after
        the last level) never appears.  A level write is a single dask stream
        and is NOT interruptible, so the effective granularity is one level.
    on_level:
        Optional progress callback ``on_level(done, total)`` invoked on the
        CALLING thread with ``done`` = pyramid levels fully written so far and
        ``total`` = levels in the pyramid.  Called once with ``(0, total)``
        before the first write (so a caller can size its bar immediately) and
        once after each level lands.  A level write is a single dask stream, so
        one level is the granularity available without instrumenting dask.  A
        callback that raises is swallowed: progress reporting must never abort a
        conversion.
    num_workers:
        Bounded parallelism for the level writes.  ``None`` (default) picks
        ``min(4, os.cpu_count())``.  ``1`` uses the SYNCHRONOUS dask scheduler --
        exactly one chunk task in flight, the original deterministic RAM
        guarantee -- and reproduces the pre-parallel behaviour bit-for-bit.  ``>1``
        uses a THREADED scheduler capped at ``num_workers`` so
        read/decompress/coarsen/write overlap (uses the otherwise-idle CPU on the
        I/O-bound convert).  Threading does NOT change output bytes.  WORKING
        (committed) RAM stays bounded and **independent of file size**: a few
        cubic chunks per worker (~``num_workers`` x a small multiple of
        ``chunk**3 * itemsize`` -- an output chunk pulls its 2x2x2 read-back
        inputs + float64 coarsen temporaries) plus the small ``from_zarr`` read
        buffers of the coarser levels (never larger than ~a Z-slab on the
        compressed path); it never scales with the volume.  CAVEAT: on the
        *uncompressed/memmap* path, process RSS (working set) does grow toward the
        source size as level 0's single read pages the memmap in -- these are
        file-backed, reclaimable pages, not committed heap, and this is unchanged
        from the pre-parallel code; a compressed source has no whole-file memmap
        so its RSS stays bounded too.

    Returns
    -------
    dict
        ``{"path", "source_path", "source_mtime", "source_size", "axes",
        "n_levels", "level_shapes", "chunks", "dtype", "downsample",
        "voxel_size"}``.  ``source_path`` is the abspath of the input,
        ``source_mtime`` its ``os.path.getmtime`` and ``source_size`` its
        ``os.path.getsize`` at convert time — together they let a reader decide
        whether this store is still fresh for its source (an edit moves the
        mtime; an mtime-preserving replacement still changes the size).  Also
        written as ``result_manifest.json`` alongside.
    """
    import dask.array as da

    from .axes import canonicalize_to_czyx, resolve_axes
    from .lazy import open_lazy

    ds = int(downsample)
    if ds < 2:
        raise ValueError(f"downsample must be >= 2 (got {downsample!r})")
    # validated (int OR (cz, cy, cx) tuple) by _cubic_chunks below; call it
    # early so a bad value fails before any I/O.
    _cubic_chunks("ZYX", chunk)
    # validated eagerly so a bad codec name fails before any I/O
    compressor_obj = _make_compressor(compressor)
    if int(min_size) < 1:
        # min_size=0 would let a size-1 axis coarsen to 0 -> a cryptic dask
        # "Empty tuples are not allowed in chunks"; fail cleanly instead.
        raise ValueError(f"min_size must be >= 1 (got {min_size!r})")

    # OPT 2: bounded parallelism.  Default to a small cap so a 64-core HPC node
    # does not over-subscribe an I/O-bound convert; num_workers=1 is the original
    # synchronous, single-chunk-in-flight behaviour (see the write phase below).
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    num_workers = max(1, int(num_workers))

    if voxel_size is not None:
        vs = tuple(float(v) for v in voxel_size)
        if len(vs) != 3 or any(v <= 0 for v in vs):
            raise ValueError(
                "voxel_size must be a length-3 (z, y, x) tuple of positive "
                f"floats in micrometers (got {voxel_size!r})"
            )
        voxel_size = vs

    out_p = Path(out_path)
    if _store_exists(out_p):
        if not overwrite:
            raise FileExistsError(
                f"output store already exists: {out_p} "
                f"(pass overwrite=True to replace it)"
            )
        _remove_store(out_p)

    # --- source provenance (for downstream "already prepared?" checks) -----
    # Captured at convert time so a reader can tell whether this store is still
    # fresh for its source: the abspath identifies the source, the mtime AND the
    # byte-size detect an edit.  Size matters because an mtime-preserving
    # replacement (``cp -p`` / ``rsync -a`` / "restore previous version" / many
    # cloud-sync clients) leaves the mtime untouched — mtime alone would then
    # serve the OLD store for a NEW source (silent wrong data for display AND
    # extract).  A content change almost always changes the byte-size, so the
    # pair closes essentially every real replacement vector.  Computed before
    # the (lazy) open; the converter never writes the source, so neither can
    # drift during the run.
    src_abspath = os.path.abspath(str(src_path))
    src_mtime = os.path.getmtime(src_path)
    src_size = os.path.getsize(src_path)
    src_head_sha = sampled_file_sha(src_path)

    # --- open lazy + resolve layout (never reads the whole volume) ---------
    # cache_bytes=0: the convert reads each source chunk EXACTLY ONCE, so a read
    # cache can only cost memory here -- there is nothing to re-hit.
    lazy = open_lazy(src_path, cache_bytes=0)
    shape = tuple(int(s) for s in lazy.shape)
    ndim = len(shape)
    dtype = np.dtype(lazy.dtype)

    if image_axes is not None:
        axes = str(image_axes).upper()
        if ndim == 4 and axes not in ("CZYX", "ZCYX"):
            raise ValueError(
                f"image_axes must be 'CZYX' or 'ZCYX' for a 4D source "
                f"(got {image_axes!r} for shape {shape})"
            )
    else:
        res = resolve_axes(shape, _read_tiff_axes(src_path))
        if ndim == 4 and not res.confident:
            # A 4D source whose channel/Z order was only GUESSED from the shape
            # (no clean metadata axes string) must be confirmed, not silently
            # applied -- a wrong guess mis-orders every voxel.  The Step 1 UI
            # passes image_axes= after the user confirms; a headless caller must
            # do the same.  (3D ZYX and confident-metadata 4D proceed as before.)
            raise ValueError(
                f"convert_to_ome_zarr: ambiguous 4D axis order for {src_path!r} "
                f"(inferred {res.axes!r} from shape {shape}, not confident); "
                "pass image_axes= (e.g. 'CZYX' or 'ZCYX') to confirm the order"
            )
        axes = res.axes

    if ndim == 4:
        canon_axes = "CZYX"
        canon = canonicalize_to_czyx(lazy, axes)   # np view (memmap) or lazy dask
    elif ndim == 3:
        canon_axes = "ZYX"
        canon = lazy
    else:
        raise ValueError(
            f"convert_to_ome_zarr: unsupported ndim={ndim} (shape={shape}); "
            "handles 3D (ZYX) and 4D (CZYX/ZCYX) sources only"
        )

    # --- level 0: cubic-chunked lazy dask over the on-disk source ---------
    chunk_shape = _cubic_chunks(canon_axes, chunk)
    chunk_zyx = chunk_shape[-3:]
    cubic = _clamp_chunks_to_shape(chunk_shape, canon.shape)
    level0 = _lazy_source(canon, cubic)

    # --- build the pyramid (all lazy; nothing computed here) --------------
    # Anisotropic + size-aware: each ZYX axis halves INDEPENDENTLY, and only
    # while the halved size stays >= min_size.  The pyramid stops when no axis
    # can be halved further.  ``cumulative`` tracks the per-axis product of the
    # per-level factors (ZYX order); level 0's cumulative factors are all 1.
    # NOTE (OPT 1): this loop is now a SHAPE + FACTOR oracle -- the chained
    # ``levels[1:]`` arrays are used only for their (lazy) ``.shape`` (the
    # manifest ``level_shapes``); the WRITE phase does NOT compute them.  It
    # instead re-derives each coarser level by coarsening the previous level read
    # back from the store, using the per-level step factor recorded in
    # ``per_level_factors``.  That is byte-identical: the chained graph already
    # casts each level to ``dtype`` before feeding the next ``coarsen``, so
    # reading the cast values back from disk feeds ``coarsen`` the identical
    # inputs (same shape, same cubic chunking, same integers) -> identical float
    # means -> identical truncation.
    zyx_idx = _spatial_axis_indices(canon_axes)
    min_sz = int(min_size)

    levels = [level0]
    level_factors = [(1, 1, 1)]                      # per-level ZYX cumulative
    per_level_factors = [None]                       # per-level ZYX step factor
    current = level0
    cumulative = [1, 1, 1]
    while True:
        if max_levels is not None and len(levels) >= int(max_levels):
            break
        cur_zyx = [int(current.shape[a]) for a in zyx_idx]
        # per-level factor per axis: halve iff the result stays >= min_size,
        # else clamp this axis to 1 (never shrinks below min_size, never vanishes)
        per_level = [ds if (sz // ds) >= min_sz else 1 for sz in cur_zyx]
        if all(f == 1 for f in per_level):
            break                                   # no axis qualifies -> done
        factor_map = _coarsen_factor_map(canon_axes, per_level)
        nxt = da.coarsen(np.mean, current, factor_map, trim_excess=True)
        # (see the rint note in the write phase below)
        nxt = np.rint(nxt)
        # each level gets the SAME chunk shape, clamped to that level's extent
        nxt = nxt.astype(dtype).rechunk(
            _clamp_chunks_to_shape(chunk_shape, nxt.shape))
        cumulative = [c * f for c, f in zip(cumulative, per_level)]
        levels.append(nxt)
        level_factors.append(tuple(cumulative))
        per_level_factors.append(tuple(int(f) for f in per_level))
        current = nxt

    # --- stream each level into the zarr group ----------------------------
    # Scheduler (OPT 2): num_workers==1 -> the SYNCHRONOUS scheduler (one chunk
    # task in flight, peak RAM deterministic + machine-independent, the original
    # guarantee); num_workers>1 -> a THREADED scheduler capped at num_workers, so
    # read/decompress/coarsen/write overlap on the otherwise-idle CPU of this
    # I/O-bound convert.  Peak is num_workers cubic chunks on the memmap path (or
    # ~a Z-slab on the compressed path) + the small from_zarr read buffers of the
    # coarser levels -- bounded, independent of file size, on any host.
    #
    # Write order (OPT 1): level 0 streams from the on-disk SOURCE (the ONLY
    # source read); each coarser level is coarsened from the previous level READ
    # BACK from the store, so the (multi-GB) source is read exactly once instead
    # of once per level.  Byte-identical to the source-chained pyramid (see the
    # pyramid-build note above).
    import dask

    # Built here, APPLIED only inside the ``with`` below (never at instantiation
    # before the try) so a store-open failure cannot leak the scheduler config
    # into the rest of the process.
    sched_kwargs = ({"scheduler": "synchronous"} if num_workers <= 1
                    else {"scheduler": "threads", "num_workers": num_workers})

    store = _zarr_store(out_p, mode="w", dimension_separator="/")
    convert_failed = True                        # cleared on the success path
    try:
        import zarr

        root = zarr.open_group(store=store, mode="w")
        root.attrs["multiscales"] = _build_multiscales(
            canon_axes, out_p.stem, level_factors, voxel_size
        )
        # PER-STORE source provenance in the store's OWN attrs, so every store
        # is self-describing regardless of how many share one parent dir.  The
        # shared ``result_manifest.json`` (below) is last-writer-wins — only the
        # most-recently-prepared source is recognisable from it — so a reader
        # that must recognise EVERY prepared source (Step-1 Smart Load) reads
        # this instead.  Same fields the manifest records; kept in lockstep.
        #
        # ``source_axes`` is the TRUE ON-DISK axis order (the in-scope ``axes``
        # resolved above — ``CZYX`` / ``ZCYX`` / ``ZYX``), NOT ``canon_axes``:
        # the store's NGFF ``multiscales.axes`` is always CANONICAL, so the raw
        # order would otherwise be nowhere durable for a Prepare-only source
        # (the "axes lie").  ``voxel_size`` is the physical
        # spacing param (``[z, y, x]`` µm) or ``None``.  Both additive.
        #
        # WRITTEN LAST, AFTER every level (2026-09-05): this marker is what
        # Step 1's freshness probe (``_fresh_prepared_store_for`` ->
        # :func:`read_store_source`) reads to decide "this store is a complete
        # conversion of that source".  Written BEFORE the level loop, an
        # interrupted Prepare (crash / disk full / cancel) left a store the
        # probe accepted as good whose missing chunks read back as ZEROS,
        # forever and silently.  Keep this assignment after the ``with`` block.
        source_provenance = {
            "path": src_abspath,
            "source_mtime": src_mtime,
            "source_size": src_size,
            # Brick 4c, ADDITIVE: a SAMPLED content digest (see
            # :func:`sampled_file_sha`).  Size + mtime cannot see an in-place
            # pixel edit of an UNCOMPRESSED TIFF -- ImageJ "Save" over the
            # original, a napari `imwrite` back onto it, an in-place
            # deconvolution -- which rewrites bytes while keeping the byte
            # count.  A legacy store omits this key; readers must ``.get`` it.
            "source_head_sha": src_head_sha,
            "source_axes": axes,
            "voxel_size": (list(voxel_size) if voxel_size is not None else None),
            # Brick 4b, ADDITIVE: the store's own chunk shape, so a reader (and
            # the processed-store parity path) never has to guess it.  A legacy
            # store omits it -> read it off the level-0 ``.zarray`` instead.
            "chunk_zyx": [int(c) for c in chunk_zyx],
            "compressor": (str(compressor).lower()
                           if compressor is not None else None),
        }
        n_levels = len(levels)
        with dask.config.set(**sched_kwargs):
            _check_cancel(should_cancel)
            # Announce the denominator before the first (longest) write so the
            # caller's bar is determinate from the start rather than after it.
            _report_level(on_level, 0, n_levels)
            # level 0: the ONLY read of the on-disk source.  chunked, RAM-bounded
            # write; a fresh store means no component collision, so overwrite=False
            # is correct (and ZipStore-safe).
            da.to_zarr(levels[0], url=store, component="0", overwrite=False,
                       compressor=compressor_obj)
            _report_level(on_level, 1, n_levels)
            # coarser levels: coarsen the JUST-WRITTEN previous level read back
            # from the store (small), NOT the source-chained lazy graph.  Same op,
            # same cast-to-dtype inputs, same cubic chunking -> byte-identical.
            for i in range(1, len(levels)):
                _check_cancel(should_cancel)
                prev = da.from_zarr(store, component=str(i - 1))
                factor_map = _coarsen_factor_map(
                    canon_axes, per_level_factors[i])
                nxt = da.coarsen(np.mean, prev, factor_map, trim_excess=True)
                # Brick 4b item 6: ROUND the block mean before the integer cast.
                # A bare .astype() TRUNCATES, so every coarser level loses ~0.5
                # LSB and the bias COMPOUNDS down the pyramid — the overview a
                # user judges brightness on ends up systematically dimmer than
                # the data.  np.rint is round-half-to-EVEN, which is unbiased in
                # both directions.  Level 0 is never coarsened, so it is
                # untouched; only NEW stores' coarse levels change.
                nxt = np.rint(nxt)
                nxt = nxt.astype(dtype).rechunk(
                    _clamp_chunks_to_shape(chunk_shape, nxt.shape))
                da.to_zarr(nxt, url=store, component=str(i), overwrite=False,
                           compressor=compressor_obj)
                _report_level(on_level, i + 1, n_levels)
                # Drop the read-back dask graphs immediately: on a ZipStore, a
                # from_zarr read opens an extra handle to the .zip that outlives
                # store.close() on Windows (blocking a later overwrite's unlink)
                # until the referencing graph is released.  del + the gc.collect()
                # below make that release deterministic.
                del prev, nxt
        # Every level is on disk -> NOW the store is a complete, trustworthy
        # conversion.  Only here does it become recognisable as "prepared".
        root.attrs["epicycle_source"] = source_provenance
        # Consolidate LAST -- after the provenance attrs.  Consolidating before
        # them would snapshot a store with NO ``epicycle_source``, and every
        # consolidated read (the Step-1 freshness probe included) would then see
        # an unprepared source and re-convert forever.  A consolidated store
        # still opens the ordinary way, so this is purely a read-speed win (one
        # ``.zmetadata`` read instead of a per-level ``.zattrs`` + ``.zarray``
        # stat walk) and old stores without it are unaffected.
        zarr.consolidate_metadata(store)
        convert_failed = False
    finally:
        if hasattr(store, "close"):
            store.close()
        # belt-and-suspenders for the Windows ZipStore handle above: break any
        # lingering read-handle reference cycles so the .zip is unlinkable now.
        gc.collect()
        if convert_failed:
            # Best-effort: drop the half-written store so nothing downstream can
            # mistake it for data (and so a retry starts from a clean path).
            # Never mask the original failure -- log and move on.
            try:
                _remove_store(out_p)
            except Exception as exc:      # pragma: no cover - filesystem-dependent
                _LOG.warning(
                    "could not remove the partial store %s after a failed "
                    "conversion: %r", out_p, exc)

    manifest = {
        "path": str(out_p),
        "source_path": src_abspath,
        "source_mtime": src_mtime,
        "source_size": src_size,
        "axes": canon_axes,
        "n_levels": len(levels),
        "level_shapes": [tuple(int(s) for s in lvl.shape) for lvl in levels],
        "chunks": tuple(int(c) for c in cubic),
        "chunk_zyx": tuple(int(c) for c in chunk_zyx),
        "compressor": (str(compressor).lower() if compressor is not None
                       else None),
        "dtype": str(dtype),
        "downsample": ds,
        "voxel_size": (tuple(voxel_size) if voxel_size is not None else None),
    }
    _write_result_manifest(out_p, manifest)
    return manifest


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

def open_multiscale(store_path, cache_bytes=DEFAULT_READ_CACHE_BYTES):
    """Open a multiscale OME-Zarr store as a list of **lazy** dask levels.

    Returns ``(levels, meta)`` where ``levels`` is level 0 (finest) first --
    the order napari's ``add_image(data=levels, multiscale=True)`` expects --
    and ``meta`` carries the parsed ``multiscales`` block (``version``, ``name``,
    ``axes``, ``datasets``, ``n_levels``, per-level ``scales``).

    Nothing is materialized: each level is ``dask.array.from_zarr`` over the
    store.  The store is intentionally **not** closed here -- the returned lazy
    levels read from it on ``.compute()``.  For a ``DirectoryStore`` that is
    free (it is stateless, one file open per chunk read); for a ``ZipStore`` the
    single file handle stays open, kept alive by the levels' references to it,
    and released only when they are garbage-collected.  On Windows that open
    handle can BLOCK an overwrite of the same ``.zip`` until the levels are
    dropped, so for local Step-1 use a ``.zarr`` DirectoryStore (no lingering
    handle) sidesteps the problem.  Handles both a ``.zarr`` DirectoryStore and
    a ``.zip`` ZipStore path.
    """
    import dask.array as da
    import zarr

    p = str(store_path)
    store = _cached_store(_zarr_store(p, mode="r"), cache_bytes)
    # a consolidated store (Brick 4b item 3) answers every metadata read from a
    # single `.zmetadata`; an older store has none, so fall back to the ordinary
    # open.  The reader NEVER assumes either form.
    try:
        group = zarr.open_consolidated(store=store, mode="r")
    except Exception:
        group = zarr.open_group(store=store, mode="r")

    ms_list = group.attrs.get("multiscales")
    if not ms_list:
        raise ValueError(
            f"no OME-NGFF 'multiscales' metadata found in store: {p}"
        )
    block = ms_list[0]

    # A hand-written / foreign 'multiscales' block may be missing 'datasets',
    # a per-dataset 'path', or the scale transform.  Surface a clean ValueError
    # rather than leaking a bare KeyError/IndexError/TypeError to the caller.
    try:
        datasets = block["datasets"]
        paths = [str(d["path"]) for d in datasets]
        scales = [d["coordinateTransformations"][0]["scale"] for d in datasets]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"malformed OME-NGFF 'multiscales' metadata in store {p}: {exc!r}"
        ) from exc

    levels = [da.from_zarr(store, component=pth) for pth in paths]
    meta = {
        "version": block.get("version"),
        "name": block.get("name"),
        "axes": block.get("axes"),
        "datasets": datasets,
        "n_levels": len(datasets),
        "scales": scales,
        # Per-store source provenance (``{"path", "source_mtime",
        # "source_size"}``) written by :func:`convert_to_ome_zarr`, or ``None``
        # for a foreign / legacy store.
        "source": group.attrs.get("epicycle_source"),
        # the cache bound actually applied to these levels (0 = uncached)
        "cache_bytes": (int(cache_bytes) if cache_bytes and int(cache_bytes) > 0
                        else 0),
    }
    return levels, meta


#: Bumped whenever the sampling PATTERN changes.  It rides in the digest string
#: itself, so an old value simply compares unequal to a new one and the caller
#: falls back to the raw file -- the conservative direction.
SAMPLED_SHA_VERSION = 2


def sampled_file_sha(path, sample_bytes: int = 1 << 20) -> Optional[str]:
    """A SAMPLED blake2b digest of *path*: head, tail AND eight interior windows.

    Step 1's store-freshness check is ``{size, mtime}``.  Both are blind to an
    in-place pixel edit of an UNCOMPRESSED TIFF -- ImageJ "Save" back over the
    original, a napari ``imwrite`` onto the same path, an in-place
    deconvolution: the pixel bytes change, the byte COUNT does not, and (for a
    tool that preserves it, or a same-second write) neither need the mtime.  The
    stale store was then served for display AND extract.

    Sampled, not whole-file, on purpose: these sources are tens to hundreds of
    GB and this runs on the UI thread at every load.  It is a strong heuristic,
    not a cryptographic guarantee, and it is only ever consulted as a TIE-BREAK
    when the byte size already matches.

    Head + tail alone were not enough (v1 -> v2): a targeted memmap write into
    ONE interior Z slice -- exactly what an in-place deconvolution or a scripted
    per-slice fix does -- lands in neither window and read back as unchanged.
    So the pattern is now the first and last *sample_bytes* PLUS eight 64 KiB
    windows spaced evenly through the interior (at ``size * i / 9``, i = 1..8).
    Total read stays ~2.5 MiB regardless of file size.

    What that buys, stated precisely rather than optimistically: any contiguous
    rewrite of roughly ``size / 9`` or more is caught, because it cannot fit
    between two windows.  One Z slice of a real stack is ~1/depth of the file,
    so a per-slice rewrite of a stack under ~9 slices is caught outright and a
    deeper one is caught as soon as it touches more than one slice (a
    deconvolution or a bleach correction rewrites all of them).  A single
    surgical write of a few KiB into a gap can still slip through -- this is a
    tie-break heuristic on top of the size check, not an integrity guarantee.

    The file's SIZE and the pattern VERSION are folded into the digest, so two
    files can only collide by sharing their length and all ten windows, and a
    digest recorded under an older pattern compares unequal (-> the caller falls
    back to the raw file, the conservative direction) instead of being trusted.

    Returns ``None`` when the file cannot be read -- never raises.
    """
    try:
        import hashlib

        n = int(sample_bytes)
        interior = 1 << 16                      # 64 KiB per interior window
        size = os.path.getsize(path)
        h = hashlib.blake2b(digest_size=16)
        h.update(f"v{SAMPLED_SHA_VERSION}:{size}".encode("ascii"))
        with open(str(path), "rb") as f:
            h.update(f.read(n))
            if size > 2 * n:
                f.seek(size - n)
                h.update(f.read(n))
            for i in range(1, 9):
                off = (size * i) // 9
                if off <= n or off >= max(size - n, 0):
                    continue                    # already inside head / tail
                f.seek(off)
                h.update(f.read(interior))
        return f"v{SAMPLED_SHA_VERSION}:{h.hexdigest()}"
    except Exception:
        return None


def read_store_source(store_path) -> Optional[dict]:
    """The per-store ``epicycle_source`` provenance dict for a multiscale store,
    or ``None``.

    A LIGHT, attrs-only read: opens the store group read-only and returns the
    ``{"path", "source_mtime", "source_size", "source_axes", "voxel_size"}``
    block :func:`convert_to_ome_zarr` writes into the store's OWN ``.zattrs``
    (``source_axes`` / ``voxel_size`` are the schema-v2 additions — the durable
    on-disk order + physical spacing; a legacy store omits them, so a reader
    must ``.get`` them defensively).  No pyramid levels are opened — this is a
    cheap freshness + provenance probe.

    NEVER raises: a missing store, a store without the block (a foreign / legacy
    store), or a corrupt store all return ``None``, so a caller (the Step-1
    Smart-Load freshness gate) can use it as a plain predicate.  Because the
    provenance lives in each store's own attrs, this recognises EVERY prepared
    source in a shared parent dir — unlike the last-writer-wins
    ``result_manifest.json`` sidecar.
    """
    try:
        store = _zarr_store(str(store_path), "r")
        try:
            import zarr

            group = zarr.open_group(store=store, mode="r")
            src = group.attrs.get("epicycle_source")
        finally:
            if hasattr(store, "close"):
                store.close()
        return src if isinstance(src, dict) else None
    except Exception:
        return None

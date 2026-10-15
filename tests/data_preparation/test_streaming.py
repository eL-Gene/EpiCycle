"""Streaming (out-of-core) whole-volume processing tests — the byte-identity GATE
with TEETH.

The streaming writer
(``packages/epicycle_preprocessing/streaming.py::stream_process_to_store``) reads
a cubic-chunked uint16 source block-by-block, runs the recipe's FUSED float32 run
per block, and writes a uint16 OME-Zarr store — never materializing the whole
volume.  For the local + pointwise + normalize ops the level-0 store MUST be
**bit-identical** (``np.array_equal``, incl. true-edge chunks) to the in-memory
``to_uint16(ops.execute(...))`` result.

Teeth (each positive test is paired with a negative control that MUST fail if the
invariant is broken):
  * T1 byte-identical fused-chain streamed-vs-in-memory (edge chunks included).
  * T2 a ``sum-1`` fused halo MUST seam (proves the SUM halo is required + minimal).
  * T3 an UNALIGNED output block + parallel writes MUST corrupt; aligned passes
        (proves the chunk-alignment WRITE-CORRECTNESS invariant).
  * T4 the exact-histogram measure == ``np.percentile`` (uint16), order-independent.
  * T5 a cancel mid-write raises + leaves NO store (partial ``.tmp`` cleaned).
  * T6 memory sanity — the writer only ever reads bounded blocks, never the whole
        array (a counting reader proves it deterministically).
  * (validity) the store is a real OME-NGFF multiscale + carries
        ``epicycle_processed`` (NOT ``epicycle_source``).

Pure — no napari / Qt (deferred dask/zarr imports live inside the engine).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("dask")

from packages.epicycle_preprocessing import ops, streaming
from packages.epicycle_preprocessing.exceptions import PreprocessingCancelled


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _write_source(path, vol, chunk):
    """Write *vol* (ZYX uint16) to a cubic-chunked zarr store and return a lazy
    read handle (the direct-read fast path the streaming writer uses)."""
    z = zarr.open(
        str(path), mode="w", shape=vol.shape, chunks=(chunk,) * vol.ndim, dtype="uint16"
    )
    z[:] = vol
    return zarr.open(str(path), mode="r")


def _read_level(store_path, level=0):
    """Read a level of a multiscale store back as a numpy array."""
    group = zarr.open_group(zarr.DirectoryStore(str(store_path)), mode="r")
    return np.asarray(group[str(level)][:])


def _in_memory_ref(recipe, channels_u16, bounds):
    """The in-memory reference: ``to_uint16(ops.execute(...))`` per channel."""
    chans_f32 = [c.astype(np.float32) for c in channels_u16]
    processed, _sp, _st = ops.execute(
        recipe, chans_f32, (1.0, 1.0, 1.0), parallel=False, bounds_per_channel=bounds
    )
    return [ops.to_uint16(p) for p in processed]


def _rand_u16(shape, seed, hi=4000):
    return np.random.default_rng(seed).integers(0, hi, shape, dtype=np.uint16)


# a recipe that genuinely tiles + fuses two local ops (gaussian + background)
_FUSED_RECIPE = {
    "channels": [
        [
            {"key": "gaussian", "params": {"sigma": 1.5}},
            {"key": "background", "params": {"radius": 3}},
        ]
    ],
    "geometry": None,
}
_GAUSS_RECIPE = {
    "channels": [[{"key": "gaussian", "params": {"sigma": 1.5}}]],
    "geometry": None,
}


# --------------------------------------------------------------------------- #
# T1 — byte-identical (fused vs in-memory), including EDGE chunks
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "recipe, tag",
    [(_GAUSS_RECIPE, "gaussian"), (_FUSED_RECIPE, "gaussian+background")],
)
def test_t1_byte_identical_incl_edges(tmp_path, recipe, tag):
    """Level-0 store == ``to_uint16(execute(...))`` EXACTLY, edges included.  The
    volume tiles on all three axes (block edge 96 < 130/140) so this genuinely
    exercises interior seams + all six true-edge faces."""
    chunk = 32
    vol = _rand_u16((130, 140, 140), seed=1)
    src = _write_source(tmp_path / "src.zarr", vol, chunk)
    bounds = [ops.percentile_bounds(vol.astype(np.float32))]

    out = tmp_path / "out.zarr"
    man = streaming.stream_process_to_store(
        [src], recipe, (1.0, 1.0, 1.0), out,
        bounds_per_channel=bounds, block_workers=6, chunk=chunk,
    )
    got = _read_level(out)
    ref = _in_memory_ref(recipe, [vol], bounds)[0]

    assert man["whole_array_remainder"] is False
    assert got.shape == vol.shape
    assert np.array_equal(got, ref), (
        f"{tag}: streamed != in-memory, "
        f"max|Δ|={int(np.abs(got.astype(int) - ref.astype(int)).max())}"
    )
    # name the edges explicitly — a wrong-boundary bug hides ONLY at the edge.
    for axis in range(got.ndim):
        lo = np.take(got, 0, axis=axis)
        hi = np.take(got, got.shape[axis] - 1, axis=axis)
        assert np.array_equal(lo, np.take(ref, 0, axis=axis)), f"low edge axis {axis}"
        assert np.array_equal(
            hi, np.take(ref, ref.shape[axis] - 1, axis=axis)
        ), f"high edge axis {axis}"


def test_t1_czyx_two_channel_byte_identical(tmp_path):
    """A multi-channel source writes a CZYX store; every channel is byte-identical."""
    chunk = 32
    v0 = _rand_u16((96, 128, 128), seed=10)
    v1 = (v0.astype(np.uint32) // 2 + 7).astype(np.uint16)
    s0 = _write_source(tmp_path / "c0.zarr", v0, chunk)
    s1 = _write_source(tmp_path / "c1.zarr", v1, chunk)
    bounds = [
        ops.percentile_bounds(v0.astype(np.float32)),
        ops.percentile_bounds(v1.astype(np.float32)),
    ]
    recipe = {
        "channels": [
            [{"key": "gaussian", "params": {"sigma": 1.5}}],
            [{"key": "gaussian", "params": {"sigma": 1.5}}],
        ],
        "geometry": None,
    }
    out = tmp_path / "czyx.zarr"
    man = streaming.stream_process_to_store(
        [s0, s1], recipe, (1.0, 1.0, 1.0), out,
        bounds_per_channel=bounds, block_workers=6, chunk=chunk,
    )
    assert man["axes"] == "CZYX"
    got = _read_level(out)
    assert got.shape == (2, 96, 128, 128)
    ref = _in_memory_ref(recipe, [v0, v1], bounds)
    assert np.array_equal(got[0], ref[0])
    assert np.array_equal(got[1], ref[1])


def test_t1_measure_pass_matches_exact_bounds_in_memory(tmp_path):
    """With bounds NOT supplied, the streamed measure (exact histogram) drives a
    result byte-identical to the in-memory run given those same exact bounds."""
    chunk = 32
    vol = _rand_u16((130, 140, 140), seed=4)
    src = _write_source(tmp_path / "src.zarr", vol, chunk)
    out = tmp_path / "out.zarr"
    man = streaming.stream_process_to_store(
        [src], _GAUSS_RECIPE, (1.0, 1.0, 1.0), out, block_workers=6, chunk=chunk,
    )
    exact = ops.percentile_bounds_from_hist(ops.histogram_uint16(vol))
    assert tuple(man["bounds"][0]) == exact
    got = _read_level(out)
    ref = _in_memory_ref(_GAUSS_RECIPE, [vol], [exact])[0]
    assert np.array_equal(got, ref)


# --------------------------------------------------------------------------- #
# T2 — sum-1 fused halo MUST seam (negative control for the SUM halo)
# --------------------------------------------------------------------------- #

def test_t2_sum_minus_one_halo_seams(tmp_path):
    """Forcing the fused halo to SUM-1 must produce a seam (NOT array_equal).  The
    correct SUM halo is byte-identical (T1); this proves the sum is required +
    minimal — a wrong composed halo would otherwise ship green."""
    chunk = 32
    vol = _rand_u16((130, 140, 140), seed=2)
    src = _write_source(tmp_path / "src.zarr", vol, chunk)
    bounds = [ops.percentile_bounds(vol.astype(np.float32))]
    ref = _in_memory_ref(_FUSED_RECIPE, [vol], bounds)[0]

    h_sum = ops._halo_gaussian(1.5) + ops._halo_background(3)
    out = tmp_path / "seam.zarr"
    streaming.stream_process_to_store(
        [src], _FUSED_RECIPE, (1.0, 1.0, 1.0), out,
        bounds_per_channel=bounds, block_workers=6, chunk=chunk, block=64,
        _force_halo=h_sum - 1,
    )
    got = _read_level(out)
    assert not np.array_equal(got, ref), (
        "sum-1 fused halo must seam — if this passes, the halo test has no teeth"
    )


# --------------------------------------------------------------------------- #
# T3 — unaligned output block + parallel writes MUST corrupt
# --------------------------------------------------------------------------- #

def test_t3_unaligned_parallel_write_is_unsafe_aligned_ok(tmp_path):
    """An UNALIGNED output block edge (not a chunk multiple) with >1 worker is
    UNSAFE — two blocks share an on-disk chunk and the concurrent read-modify-write
    either silently corrupts it (probed ``max|Δ|≈60000``) or, on Windows, raises a
    chunk-file sharing violation (``PermissionError``).  The chunk-ALIGNED block is
    deterministically clean.  This is the WRITE-CORRECTNESS invariant.

    Uses a normalize-only recipe (a pure pointwise map, halo 0) so the mismatch can
    ONLY be a WRITE race, never a halo seam.  ``unsafe`` = raises OR result != ref;
    the unaligned config is ~10/10 unsafe (measured) so a few attempts is robust;
    the aligned path is deterministically byte-identical."""
    chunk = 32
    vol = _rand_u16((64, 256, 256), seed=7, hi=30000)
    src = _write_source(tmp_path / "src.zarr", vol, chunk)
    bounds = [ops.percentile_bounds(vol.astype(np.float32))]
    ref = ops.to_uint16(ops.percentile_normalize(vol.astype(np.float32), bounds=bounds[0]))
    norm_only = {"channels": [[]], "geometry": None}

    # aligned block (64 = 2·chunk) → deterministically byte-identical, parallel.
    out_a = tmp_path / "aligned.zarr"
    streaming.stream_process_to_store(
        [src], norm_only, (1.0, 1.0, 1.0), out_a,
        bounds_per_channel=bounds, block_workers=8, chunk=chunk, block=64,
    )
    assert np.array_equal(_read_level(out_a), ref), "aligned parallel write must be clean"

    # unaligned block (48, not a multiple of 32) → shared-chunk RMW hazard.
    def _unsafe_once(i):
        out_u = tmp_path / f"unaligned_{i}.zarr"
        try:
            streaming.stream_process_to_store(
                [src], norm_only, (1.0, 1.0, 1.0), out_u,
                bounds_per_channel=bounds, block_workers=8, chunk=chunk, block=48,
            )
        except Exception:  # loud sharing violation IS the hazard manifesting
            return True
        return not np.array_equal(_read_level(out_u), ref)  # silent corruption

    assert any(_unsafe_once(i) for i in range(6)), (
        "unaligned block + parallel writes must be unsafe (corrupt or raise) — the "
        "chunk-alignment invariant is exactly what prevents this"
    )


# --------------------------------------------------------------------------- #
# T4 — exact-histogram measure parity + order-independence
# --------------------------------------------------------------------------- #

def test_t4_histogram_measure_matches_np_percentile_exactly():
    """``percentile_bounds_from_hist(histogram_uint16(arr))`` EQUALS
    ``np.percentile(arr, [low, high])`` bit-for-bit for a uint16 array (incl. the
    interpolated high percentile)."""
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 600, (60, 80, 80), dtype=np.uint16)
    mask = rng.random(arr.shape) < 0.02
    arr[mask] = rng.integers(3000, 60000, size=int(mask.sum()), dtype=np.uint16)

    lo, hi = ops.percentile_bounds_from_hist(ops.histogram_uint16(arr), 0.1, 99.5)
    np_lo, np_hi = np.percentile(arr, [0.1, 99.5])
    assert lo == float(np_lo)
    assert hi == float(np_hi)


def test_t4_histogram_accumulator_order_independent():
    """Summing per-block histograms is commutative → any read order gives the same
    total (and the same window)."""
    rng = np.random.default_rng(12)
    arr = rng.integers(0, 5000, (40, 50, 50), dtype=np.uint16)
    blocks = [arr[i::4] for i in range(4)]

    fwd = ops.Uint16HistogramAccumulator()
    for b in blocks:
        fwd.add(b)
    rev = ops.Uint16HistogramAccumulator()
    for b in reversed(blocks):
        rev.add(b)

    assert np.array_equal(fwd.histogram, rev.histogram)
    assert np.array_equal(fwd.histogram, ops.histogram_uint16(arr))
    assert ops.percentile_bounds_from_hist(fwd.histogram) == ops.percentile_bounds_from_hist(
        ops.histogram_uint16(arr)
    )


def test_t4_flat_histogram_returns_none():
    """A degenerate (flat) source → None (the all-zeros normalize fallback)."""
    flat = np.full((16, 16, 16), 1234, dtype=np.uint16)
    assert ops.percentile_bounds_from_hist(ops.histogram_uint16(flat)) is None
    assert ops.percentile_bounds_from_hist(np.zeros(65536, dtype=np.int64)) is None


# --------------------------------------------------------------------------- #
# T5 — cancel mid-write raises + leaves NO store (partial .tmp cleaned)
# --------------------------------------------------------------------------- #

def test_t5_cancel_mid_write_leaves_no_store(tmp_path):
    chunk = 32
    vol = _rand_u16((130, 140, 140), seed=3)
    src = _write_source(tmp_path / "src.zarr", vol, chunk)
    bounds = [ops.percentile_bounds(vol.astype(np.float32))]

    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] > 5  # flip after a few waves, mid-apply

    out = tmp_path / "cancelled.zarr"
    with pytest.raises(PreprocessingCancelled):
        streaming.stream_process_to_store(
            [src], _FUSED_RECIPE, (1.0, 1.0, 1.0), out,
            bounds_per_channel=bounds, block_workers=4, chunk=chunk, block=64,
            should_cancel=should_cancel,
        )
    assert not os.path.exists(out), "cancelled run must leave NO store at the final path"
    assert glob.glob(str(out) + ".tmp*") == [], "the partial .tmp must be cleaned up"


def test_t5_error_mid_write_leaves_no_store(tmp_path):
    """A per-block error also cleans up the partial — no half-store at final."""
    chunk = 32
    vol = _rand_u16((130, 140, 140), seed=33)
    src = _write_source(tmp_path / "src.zarr", vol, chunk)

    calls = {"n": 0}

    class _Boom(Exception):
        pass

    def boom_bounds():
        # a should_cancel that RAISES a non-cancel error mid-run
        calls["n"] += 1
        if calls["n"] > 5:
            raise _Boom("boom")
        return False

    out = tmp_path / "errored.zarr"
    with pytest.raises(_Boom):
        streaming.stream_process_to_store(
            [src], _FUSED_RECIPE, (1.0, 1.0, 1.0), out,
            bounds_per_channel=[ops.percentile_bounds(vol.astype(np.float32))],
            block_workers=4, chunk=chunk, block=64, should_cancel=boom_bounds,
        )
    assert not os.path.exists(out)
    assert glob.glob(str(out) + ".tmp*") == []


# --------------------------------------------------------------------------- #
# T6 — memory sanity: the writer only reads bounded blocks, never the whole array
# --------------------------------------------------------------------------- #

class _CountingArray:
    """A duck-array over a numpy volume that records the LARGEST single read.

    Proves deterministically that the streaming writer materializes only bounded
    blocks (``<= (block + 2·halo)³``) and NEVER the whole array — a robust
    memory-bound proxy that doesn't depend on flaky RSS / tracemalloc sampling."""

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.ndim = arr.ndim
        self.max_read_voxels = 0
        self.n_reads = 0

    def __getitem__(self, key):
        block = np.asarray(self._arr[key])
        self.max_read_voxels = max(self.max_read_voxels, int(block.size))
        self.n_reads += 1
        return block


def test_t6_writer_never_materializes_whole_array(tmp_path):
    chunk = 32
    vol = _rand_u16((96, 200, 200), seed=21)
    reader = _CountingArray(vol)
    bounds = [ops.percentile_bounds(vol.astype(np.float32))]

    out = tmp_path / "out.zarr"
    man = streaming.stream_process_to_store(
        [reader], _GAUSS_RECIPE, (1.0, 1.0, 1.0), out,
        bounds_per_channel=bounds, block_workers=4, chunk=chunk, block=64,
    )

    # output correct + shape matches without a full in-RAM copy
    assert man["level_shapes"][0] == vol.shape
    got = _read_level(out)
    ref = _in_memory_ref(_GAUSS_RECIPE, [vol], bounds)[0]
    assert np.array_equal(got, ref)

    # genuinely tiled - many bounded reads, never the whole volume
    halo = ops._halo_gaussian(1.5)                       # 6
    block_edge = streaming._aligned_block(64, chunk)     # explicit block above
    max_tile = (block_edge + 2 * halo) ** 3
    full = int(np.prod(vol.shape))
    assert reader.n_reads > 1, "must read multiple blocks (genuinely tiled)"
    assert reader.max_read_voxels <= max_tile, (
        f"a single read ({reader.max_read_voxels}) exceeded the block bound ({max_tile})"
    )
    assert reader.max_read_voxels < full, "the writer must NEVER read the whole array at once"


# --------------------------------------------------------------------------- #
# validity — real OME-NGFF multiscale + epicycle_processed (NOT epicycle_source)
# --------------------------------------------------------------------------- #

def test_store_is_valid_ome_ngff_and_processed_provenance(tmp_path):
    """The processed store loads through the shared ``open_multiscale`` path with a
    pyramid, carries the physical scale, and is tagged ``epicycle_processed`` — and
    is NEVER mistaken for a raw prepared source (no ``epicycle_source``)."""
    from packages.epicycle_largedata.multiscale import open_multiscale, read_store_source

    chunk = 32
    vol = _rand_u16((96, 96, 96), seed=5)
    src = _write_source(tmp_path / "src.zarr", vol, chunk)
    bounds = [ops.percentile_bounds(vol.astype(np.float32))]

    out = tmp_path / "proc.zarr"
    man = streaming.stream_process_to_store(
        [src], _GAUSS_RECIPE, (0.5, 0.2, 0.2), out,
        bounds_per_channel=bounds, block_workers=4, chunk=chunk,
        min_size=32, downsample=2,
    )
    # a coarser level built (96 -> 48 >= 32)
    assert man["n_levels"] >= 2

    levels, meta = open_multiscale(out)
    assert meta["n_levels"] == man["n_levels"]
    assert tuple(levels[0].shape) == vol.shape
    # physical scale carried from `spacing` (z, y, x) µm on level 0
    assert list(meta["scales"][0]) == [0.5, 0.2, 0.2]

    # processed provenance present; raw-source provenance absent (never confusable)
    assert read_store_source(out) is None
    group = zarr.open_group(zarr.DirectoryStore(str(out)), mode="r")
    assert "epicycle_processed" in group.attrs
    assert "epicycle_source" not in group.attrs
    proc = dict(group.attrs["epicycle_processed"])
    assert proc["scale"] == "fixed_unit_x65535"
    assert proc["whole_array_remainder"] is False
    assert proc["bounds"][0] == list(bounds[0])


def test_clahe_recipe_flags_whole_array_remainder(tmp_path):
    """A CLAHE (adaptive) op stops the fused run → the manifest flags a remainder
    the controller must handle whole-array (Phase 2a scope boundary)."""
    chunk = 32
    vol = _rand_u16((96, 96, 96), seed=6)
    src = _write_source(tmp_path / "src.zarr", vol, chunk)
    recipe = {
        "channels": [
            [
                {"key": "gaussian", "params": {"sigma": 1.5}},
                {"key": "clahe", "params": {"clip": 0.01, "kernel": 64}},
            ]
        ],
        "geometry": None,
    }
    out = tmp_path / "out.zarr"
    man = streaming.stream_process_to_store(
        [src], recipe, (1.0, 1.0, 1.0), out,
        bounds_per_channel=[ops.percentile_bounds(vol.astype(np.float32))],
        block_workers=4, chunk=chunk,
    )
    assert man["whole_array_remainder"] is True


# ---------------------------------------------------------------------------
# identity geometry must not block streaming
# ---------------------------------------------------------------------------

class TestIdentityGeometryStreams:
    """The DEFAULT recipe seeds a shared ``isotropic`` geometry step.  At
    identity (target == the source spacing / unit factors) it resamples
    nothing, yet it used to stop the fused run — so every prepared source was
    refused with "resize step that can't stream yet".  A REAL resize must still
    be refused."""

    from packages.epicycle_preprocessing import recipe as _r
    from packages.epicycle_preprocessing import streaming as _s

    def _plan(self, recipe, spacing):
        flat = self._r.flatten_for_channel(recipe, 0)
        return self._s._fused_plan(flat, spacing=spacing)

    def test_default_recipe_streams_on_unit_spacing(self):
        rec = self._r.default_recipe(1)
        fused, halo, remainder = self._plan(rec, (1.0, 1.0, 1.0))
        assert remainder == [], (
            "the default recipe's identity isotropic step blocked streaming")
        assert len(fused) == 1                      # the gaussian still fuses
        assert halo > 0

    def test_identity_aniso_factors_stream_without_spacing(self):
        rec = self._r.default_recipe(1)
        rec.geometry.config = {"mode": "aniso", "factors": (1.0, 1.0, 1.0),
                               "method": "linear"}
        _fused, _halo, remainder = self._plan(rec, None)
        assert remainder == []

    def test_real_isotropic_resize_still_refuses(self):
        rec = self._r.default_recipe(1)
        # anisotropic source, target 1 µm -> a genuine Z upsample
        _fused, _halo, remainder = self._plan(rec, (2.0, 0.5, 0.5))
        assert remainder == ["isotropic"]

    def test_real_aniso_resize_still_refuses(self):
        rec = self._r.default_recipe(1)
        rec.geometry.config = {"mode": "aniso", "factors": (2.0, 1.0, 1.0),
                               "method": "linear"}
        _fused, _halo, remainder = self._plan(rec, (1.0, 1.0, 1.0))
        assert remainder == ["isotropic"]

    def test_unknown_spacing_with_iso_geometry_is_refused(self):
        """Fail SAFE: iso factors depend on the spacing, so with no spacing the
        step stays blocking rather than risk a silently-dropped resize."""
        rec = self._r.default_recipe(1)
        _fused, _halo, remainder = self._plan(rec, None)
        assert remainder == ["isotropic"]

    def test_ops_after_an_identity_geometry_still_fuse(self):
        """Skipping (not stopping on) an identity step keeps the rest of the
        chain fused — it is a genuine no-op, so order is unaffected."""
        rec = self._r.default_recipe(1)
        rec.channels[0].append(
            self._r.IntensityStep(key="gaussian", params={"sigma": 1.0}))
        fused, _halo, remainder = self._plan(rec, (1.0, 1.0, 1.0))
        assert remainder == []
        assert len(fused) == 2


# --------------------------------------------------------------------------- #
# Brick 3c #3 - the store chunk default is 64 again, and stores written at EITHER
# edge still open (the 3a #2 bump to 128 helped writes but hurt slice scrolling)
# --------------------------------------------------------------------------- #

def test_default_store_chunk_is_inherited_from_the_source():
    """Brick 4b: `stream_process_to_store` no longer hardcodes a cubic edge.  Its
    `chunk` default is None = INHERIT the source store's own `.zarray` chunks, so
    a processed store is in layout parity with the prepared store it was streamed
    from BY CONSTRUCTION -- no caller has to remember to pass a matching value.
    The Prepare convert's own default is the anisotropic (32, 128, 128)."""
    import inspect

    from packages.epicycle_largedata import config as _cfg
    from packages.epicycle_largedata.multiscale import convert_to_ome_zarr

    assert inspect.signature(
        streaming.stream_process_to_store).parameters["chunk"].default is None
    assert tuple(inspect.signature(
        convert_to_ome_zarr).parameters["chunk"].default) == (32, 128, 128)

    # every preset writes the same SHAPE; they differ only in worker count
    for preset in ("safe", "balanced", "fast"):
        rec = _cfg.recommend_convert_settings(
            preset, cpu_count=8, total_ram_bytes=64 << 30,
            avail_ram_bytes=64 << 30)
        assert tuple(rec["chunk_zyx"]) == (32, 128, 128), preset


@pytest.mark.parametrize("chunk", [64, 128])
def test_stores_written_at_either_chunk_open_and_read_back(tmp_path, chunk):
    """The chunk edge is per-store METADATA and is never assumed by the reader,
    so both a legacy 64 store and a Brick-3a-era 128 store must open, expose
    every pyramid level, and be byte-identical to the in-memory reference."""
    vol = _rand_u16((70, 140, 140), seed=90 + chunk)
    src = _write_source(tmp_path / "src.zarr", vol, 64)
    bounds = [ops.percentile_bounds(vol.astype(np.float32))]
    out = tmp_path / f"store{chunk}.zarr"
    man = streaming.stream_process_to_store(
        [src], _GAUSS_RECIPE, (1.0, 1.0, 1.0), out,
        bounds_per_channel=bounds, block_workers=2, chunk=chunk, block=chunk,
    )
    # Brick 4b: each level's chunks are CLAMPED to that level's extent, so a
    # chunk axis can be smaller than the requested edge on a small test volume.
    got = tuple(man["chunks"])[-3:]
    assert all(g <= chunk for g in got)
    assert any(g == chunk for g in got)

    group = zarr.open_group(zarr.DirectoryStore(str(out)), mode="r")
    for lvl in range(int(man["n_levels"])):
        assert np.asarray(group[str(lvl)][:]).size > 0
    assert np.array_equal(_read_level(out),
                          _in_memory_ref(_GAUSS_RECIPE, [vol], bounds)[0])


def test_bounded_read_holds_at_the_production_defaults(tmp_path):
    """The Brick 3a T6 test pins the bounded-read invariant with an EXPLICIT
    `block=64`.  That is the right choice there (it keeps a small fixture
    genuinely tiled), but it means nothing checks the invariant on the edge the
    engine actually picks.  This one passes NO block: the planner chooses the
    adaptive + wave-capped edge, and the reader must still never see more than
    one padded tile, nor the whole array."""
    chunk = 64
    vol = _rand_u16((96, 400, 400), seed=22)
    reader = _CountingArray(vol)
    bounds = [ops.percentile_bounds(vol.astype(np.float32))]

    out = tmp_path / "prod.zarr"
    workers = 4
    man = streaming.stream_process_to_store(
        [reader], _GAUSS_RECIPE, (1.0, 1.0, 1.0), out,
        bounds_per_channel=bounds, block_workers=workers, chunk=chunk,
    )

    halo = ops._halo_gaussian(1.5)
    edge = streaming._aligned_block(
        streaming._adaptive_block_edge(halo, shape=vol.shape, workers=workers),
        chunk)
    max_tile = (edge + 2 * halo) ** 3
    full = int(np.prod(vol.shape))
    assert reader.n_reads > 1, "the production edge must still tile this volume"
    assert reader.max_read_voxels <= max_tile, (
        f"a single read ({reader.max_read_voxels}) exceeded the production block "
        f"bound ({max_tile}) at edge {edge}"
    )
    assert reader.max_read_voxels < full
    assert np.array_equal(_read_level(out),
                          _in_memory_ref(_GAUSS_RECIPE, [vol], bounds)[0])
    assert man["level_shapes"][0] == vol.shape

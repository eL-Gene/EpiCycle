"""Brick 4b item 4 — a bounded LRU read cache over a zarr store.

Sequential Z scrolling reads the SAME chunk row 32 times in a row (the store's
Z chunk is 32).  Without a cache each of those 32 slices re-reads and
re-decompresses the whole row.  ``zarr.LRUStoreCache`` keeps the compressed
blocks so the row is fetched once.

The invariant that must NOT break: the cache is a MEMORY bound, not a
correctness change.  Cached and uncached reads are byte-identical, the cache is
bounded (it can never grow into the whole volume), and the lazy-extract path
still sees a real zarr array.
"""
from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_largedata.lazy import open_lazy
from packages.epicycle_largedata.multiscale import (
    DEFAULT_READ_CACHE_BYTES,
    convert_to_ome_zarr,
    open_multiscale,
)


def _store(tmp_path, shape=(96, 300, 300), **kw):
    import tifffile

    a = (np.arange(int(np.prod(shape))) % 5000).astype(np.uint16).reshape(shape)
    src = tmp_path / "src.tif"
    tifffile.imwrite(str(src), a)
    out = tmp_path / "s.zarr"
    convert_to_ome_zarr(src, out, chunk=(32, 128, 128), max_levels=2, **kw)
    return out, a


def test_the_cache_is_OFF_by_default():
    """Measured IN NAPARI (fresh process per case, level 0, 40 Z steps): the
    256 MB LRU made scrolling WORSE -- old store 45 -> 75 ms/step, new store
    36 -> 45 ms/step.  The OS page cache already serves the chunk files, so the
    Python-level cache adds bookkeeping and a second copy of bytes that are
    already resident, and buys nothing.  It stays available as an option for
    callers on slow/remote storage, but it must not be the default."""
    assert DEFAULT_READ_CACHE_BYTES == 0


def test_open_multiscale_is_uncached_by_default(tmp_path):
    out, a = _store(tmp_path)
    levels, meta = open_multiscale(out)
    assert meta["cache_bytes"] == 0
    np.testing.assert_array_equal(np.asarray(levels[0]), a)


def test_the_cache_can_still_be_opted_into(tmp_path):
    out, a = _store(tmp_path)
    levels, meta = open_multiscale(out, cache_bytes=64 << 20)
    assert meta["cache_bytes"] == 64 << 20
    np.testing.assert_array_equal(np.asarray(levels[0]), a)


def test_open_lazy_is_uncached_by_default(tmp_path):
    """The convert's own source read must not pay for a cache either."""
    import zarr

    out, a = _store(tmp_path)
    lz = open_lazy(str(out))
    np.testing.assert_array_equal(np.asarray(lz), a)


def test_cached_and_uncached_reads_are_identical(tmp_path):
    out, a = _store(tmp_path)
    hot, _ = open_multiscale(out, cache_bytes=64 << 20)
    cold, _ = open_multiscale(out, cache_bytes=0)
    for h, c in zip(hot, cold):
        np.testing.assert_array_equal(np.asarray(h), np.asarray(c))
    np.testing.assert_array_equal(np.asarray(cold[0]), a)


def test_cache_can_be_disabled(tmp_path):
    out, _a = _store(tmp_path)
    _levels, meta = open_multiscale(out, cache_bytes=0)
    assert meta["cache_bytes"] == 0


def test_the_cache_is_bounded_not_unbounded(tmp_path):
    """The whole point of LRUStoreCache over a plain dict: a scroll through a
    volume far bigger than the cache must EVICT, never grow without limit."""
    import zarr

    from packages.epicycle_largedata.multiscale import _cached_store, _zarr_store

    out, _a = _store(tmp_path)
    cache = _cached_store(_zarr_store(str(out), "r"), 1 << 20)   # 1 MiB
    assert isinstance(cache, zarr.LRUStoreCache)
    z = zarr.open_group(store=cache, mode="r")["0"]
    for zi in range(0, 96, 4):
        _ = np.asarray(z[zi])
    assert cache._max_size == (1 << 20)
    assert cache._current_size <= cache._max_size


def test_a_hot_reread_hits_the_cache(tmp_path):
    """Re-reading the SAME slice must not re-fetch from the store."""
    import zarr

    from packages.epicycle_largedata.multiscale import _cached_store, _zarr_store

    out, _a = _store(tmp_path)
    cache = _cached_store(_zarr_store(str(out), "r"), 64 << 20)
    z = zarr.open_group(store=cache, mode="r")["0"]
    _ = np.asarray(z[10])
    misses_after_first = cache.misses
    for _ in range(5):
        _ = np.asarray(z[10])
    assert cache.hits > 0
    assert cache.misses == misses_after_first      # no new store reads


def test_a_sequential_scroll_inside_one_chunk_row_reuses_it(tmp_path):
    """Z 0..31 all live in the SAME chunk row (chunk_z = 32): after the first
    slice the other 31 must be cache hits, not 31 more store fetches."""
    import zarr

    from packages.epicycle_largedata.multiscale import _cached_store, _zarr_store

    out, _a = _store(tmp_path)
    cache = _cached_store(_zarr_store(str(out), "r"), 128 << 20)
    z = zarr.open_group(store=cache, mode="r")["0"]
    _ = np.asarray(z[0])
    after_first = cache.misses
    for zi in range(1, 32):
        _ = np.asarray(z[zi])
    assert cache.misses == after_first             # zero extra fetches
    assert cache.hits >= 31


# --------------------------------------------------------------------------- #
# open_lazy: the extract path must still see a zarr array
# --------------------------------------------------------------------------- #

def test_open_lazy_on_a_zarr_store_is_still_a_dask_array_over_zarr(tmp_path):
    import dask.array as da

    out, a = _store(tmp_path)
    lz = open_lazy(str(out))
    assert isinstance(lz, da.Array)
    assert lz.shape == a.shape
    np.testing.assert_array_equal(np.asarray(lz), a)


def test_open_lazy_crop_reads_only_the_crop(tmp_path):
    """The bounded-read invariant: the cache bounds MEMORY, it must not turn a
    bounded ROI read into a whole-volume read."""
    from packages.epicycle_largedata.lazy import crop_zyx

    out, a = _store(tmp_path)
    lz = open_lazy(str(out))
    b = {"z0": 10, "z1": 20, "y0": 30, "y1": 60, "x0": 5, "x1": 45}
    got = crop_zyx(lz, b)
    assert got.shape == (10, 30, 40)
    np.testing.assert_array_equal(got, a[10:20, 30:60, 5:45])


def test_open_lazy_cache_can_be_opted_into(tmp_path):
    out, a = _store(tmp_path)
    lz = open_lazy(str(out), cache_bytes=64 << 20)
    np.testing.assert_array_equal(np.asarray(lz), a)


def test_open_lazy_on_a_tiff_is_unchanged_by_the_cache_arg(tmp_path):
    import tifffile

    a = np.arange(8 * 16 * 16, dtype=np.uint16).reshape(8, 16, 16)
    p = tmp_path / "plain.tif"
    tifffile.imwrite(str(p), a)
    got = open_lazy(str(p))                    # memmap path, no store to wrap
    np.testing.assert_array_equal(np.asarray(got), a)


def test_a_legacy_cubic_store_opts_in_the_same_way(tmp_path):
    import tifffile

    a = (np.arange(64 * 128 * 128) % 900).astype(np.uint16).reshape(64, 128, 128)
    src = tmp_path / "old.tif"
    tifffile.imwrite(str(src), a)
    out = tmp_path / "old.zarr"
    convert_to_ome_zarr(src, out, chunk=64, compressor="default", max_levels=1)
    levels, meta = open_multiscale(out, cache_bytes=32 << 20)
    assert meta["cache_bytes"] == 32 << 20
    np.testing.assert_array_equal(np.asarray(levels[0]), a)

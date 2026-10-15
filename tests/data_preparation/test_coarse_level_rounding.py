"""Brick 4b item 6 — coarser pyramid levels ROUND the block mean instead of
truncating it.

`coarsen(np.mean, ...).astype(uint16)` truncates: a block mean of 100.9 becomes
100.  Every level therefore loses ~0.5 LSB, and the bias COMPOUNDS down the
pyramid, so the overview a user judges brightness on is systematically dimmer
than the data.  `np.rint` first removes the bias.

Level 0 is byte-identical either way (it is never coarsened) — only the coarse
levels of NEW stores change.
"""
from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_largedata.multiscale import (
    convert_to_ome_zarr,
    open_multiscale,
)


def _tif(tmp_path, a, name="src.tif"):
    import tifffile

    p = tmp_path / name
    tifffile.imwrite(str(p), a)
    return p


def test_a_block_that_rounds_up_is_not_truncated(tmp_path):
    """A 2x2x2 block of {100,100,100,100,100,100,101,101} has mean 100.25 -> 100;
    one of {101,101,101,101,101,101,100,100} has mean 100.75 -> 101 with rint but
    100 with truncation."""
    a = np.full((2, 2, 2), 101, dtype=np.uint16)
    a[0, 0, 0] = 100
    a[0, 0, 1] = 100                      # mean = 100.75
    src = _tif(tmp_path, a)
    out = tmp_path / "r.zarr"
    convert_to_ome_zarr(src, out, chunk=2, downsample=2, min_size=1)
    levels, _meta = open_multiscale(out)
    assert len(levels) >= 2
    assert int(np.asarray(levels[1]).ravel()[0]) == 101      # not 100


def test_level_zero_is_untouched(tmp_path):
    a = (np.arange(8 * 8 * 8) % 251).astype(np.uint16).reshape(8, 8, 8)
    src = _tif(tmp_path, a)
    out = tmp_path / "l0.zarr"
    convert_to_ome_zarr(src, out, chunk=4, downsample=2, min_size=2)
    levels, _meta = open_multiscale(out)
    np.testing.assert_array_equal(np.asarray(levels[0]), a)


def test_coarse_levels_match_an_independent_rint_reference(tmp_path):
    import dask.array as da

    rng = np.random.default_rng(3)
    a = rng.integers(0, 4000, size=(16, 16, 16)).astype(np.uint16)
    src = _tif(tmp_path, a)
    out = tmp_path / "ref.zarr"
    convert_to_ome_zarr(src, out, chunk=8, downsample=2, min_size=4)
    levels, _meta = open_multiscale(out)

    ref = da.from_array(a, chunks=(8, 8, 8))
    for i in range(1, len(levels)):
        ref = np.rint(da.coarsen(np.mean, ref, {0: 2, 1: 2, 2: 2},
                                 trim_excess=True)).astype(np.uint16)
        np.testing.assert_array_equal(np.asarray(levels[i]), np.asarray(ref))


def test_rounding_removes_the_downward_bias(tmp_path):
    """The point of the change: the coarse level's mean must track the source's,
    not sit systematically below it."""
    rng = np.random.default_rng(11)
    a = rng.integers(1000, 1100, size=(32, 32, 32)).astype(np.uint16)
    src = _tif(tmp_path, a)
    out = tmp_path / "bias.zarr"
    convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=4)
    levels, _meta = open_multiscale(out)
    l1 = np.asarray(levels[1]).astype(np.float64)
    # truncation would sit ~0.5 low; rint keeps it within a few hundredths
    assert abs(l1.mean() - a.astype(np.float64).mean()) < 0.1


def test_the_streamed_processed_store_rounds_too(tmp_path):
    """The processed store reuses the same coarsen loop; it must not drift from
    the prepared store it mirrors."""
    import zarr

    from packages.epicycle_preprocessing.streaming import stream_process_to_store

    # A RANDOM field: the recipe rescales to uint16, so we cannot predict
    # absolute values -- but a random field reliably produces block means with a
    # fractional part, which is what makes rint and truncation differ.  The
    # vacuity guard at the bottom asserts that it actually did.
    rng = np.random.default_rng(17)
    a = rng.integers(0, 4000, size=(4, 8, 8)).astype(np.uint16)
    z = zarr.open(str(tmp_path / "s.zarr"), mode="w", shape=a.shape,
                  chunks=(4, 8, 8), dtype="uint16")
    z[:] = a
    out = tmp_path / "p.zarr"
    man = stream_process_to_store(
        [z], {"channels": [[]]}, (1.0, 1.0, 1.0), str(out),
        downsample=2, min_size=2)
    assert man["n_levels"] >= 2
    levels, _meta = open_multiscale(out)
    l0 = np.asarray(levels[0]).astype(np.float64)
    l1 = np.asarray(levels[1])

    # Independent reference over the ACTUAL level-0 the stream wrote (the recipe
    # rescales, so we cannot predict absolute values -- but we CAN require that
    # every coarse voxel equals rint(block mean), and that at least one of them
    # would differ under truncation.  Without that second assertion this test
    # passes for a truncating implementation too, which is how it was vacuous.
    blocks = l0.reshape(2, 2, 4, 2, 4, 2).mean(axis=(1, 3, 5))
    np.testing.assert_array_equal(l1, np.rint(blocks).astype(np.uint16))
    n_diff = int((np.rint(blocks).astype(np.uint16)
                  != blocks.astype(np.uint16)).sum())
    assert n_diff > 0, (
        "fixture is vacuous: no coarse voxel distinguishes rint from truncation")

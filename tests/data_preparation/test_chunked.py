"""Chunked-processing Phase 0-1 tests — the byte-identical correctness GATE.

This is the training-data safety net.  For the LOCAL ops (gaussian, background) the
block-parallel overlap-and-trim primitive
(``packages/epicycle_preprocessing/chunked.py::block_parallel_apply``) MUST be
**bit-identical** (``np.array_equal``, NOT loose ``allclose``) to the whole-array
op — INCLUDING the true-volume-edge voxels.  A grid seam is a FAILED test, never
a silent corruption.

The tests have TEETH: a NEGATIVE control proves an UNDERSIZED halo actually
seams (``NOT array_equal``), so a green positive test can't be a false pass.

Pure — no napari / Qt.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from packages.epicycle_preprocessing import chunked, ops
from packages.epicycle_preprocessing.recipe import IntensityStep, Recipe


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _vol(shape, seed=0):
    return np.random.RandomState(seed).rand(*shape).astype(np.float32)


def _assert_edges_match(got, whole):
    """Explicitly assert the 6 true-edge faces match (belt-and-suspenders on top
    of the full-array array_equal — the wrong-boundary bug hides ONLY at the edge,
    so name it)."""
    for axis in range(got.ndim):
        lo = np.take(got, 0, axis=axis)
        hi = np.take(got, got.shape[axis] - 1, axis=axis)
        assert np.array_equal(lo, np.take(whole, 0, axis=axis)), f"low edge axis {axis}"
        assert np.array_equal(
            hi, np.take(whole, whole.shape[axis] - 1, axis=axis)
        ), f"high edge axis {axis}"


# --------------------------------------------------------------------------- #
# gaussian — byte-identical for the TRUE halo (int(4σ+0.5)); FULL array incl edge
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sigma", [1.0, 1.5, 3.0])
def test_gaussian_chunked_is_byte_identical_to_whole(sigma):
    vol = _vol((40, 48, 44), seed=1)
    halo = ops._halo_gaussian(sigma)          # == int(4σ + 0.5), the true reach
    whole = ops.gaussian(vol, sigma=sigma)
    got = chunked.block_parallel_apply(
        lambda a: ops.gaussian(a, sigma=sigma), vol, halo, block=16
    )
    # FULL array (incl. every true-edge voxel) is bit-identical — a seam fails here.
    assert np.array_equal(got, whole), (
        f"sigma={sigma} halo={halo}: chunked != whole "
        f"max|Δ|={float(np.abs(got.astype(np.float64) - whole).max()):.3e}"
    )
    _assert_edges_match(got, whole)


def test_gaussian_undersized_halo_seams_negative_control():
    """TEETH: the old ``ceil(3σ)`` halo is undersized → it MUST seam (NOT
    array_equal).  Without this, the positive test could pass with a broken op."""
    sigma = 3.0
    vol = _vol((40, 48, 44), seed=1)
    under = int(math.ceil(3.0 * sigma))       # 9 — the pre-fix undersized halo
    assert under < ops._halo_gaussian(sigma)  # sanity: it really is smaller than 4σ
    whole = ops.gaussian(vol, sigma=sigma)
    got = chunked.block_parallel_apply(
        lambda a: ops.gaussian(a, sigma=sigma), vol, under, block=16
    )
    assert not np.array_equal(got, whole), (
        "undersized halo must seam — if this passes, the test has no teeth"
    )


# --------------------------------------------------------------------------- #
# background (white top-hat) — byte-identical for the TRUE halo (2r); incl edge
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("radius", [6, 15])
def test_background_chunked_is_byte_identical_to_whole(radius):
    vol = _vol((48, 48, 48), seed=2)
    halo = ops._halo_background(radius=radius)   # == 2·radius, the true reach
    whole = ops.background(vol, radius=radius)
    got = chunked.block_parallel_apply(
        lambda a: ops.background(a, radius=radius), vol, halo, block=24
    )
    assert np.array_equal(got, whole), (
        f"radius={radius} halo={halo}: chunked != whole "
        f"max|Δ|={float(np.abs(got.astype(np.float64) - whole).max()):.3e}"
    )
    _assert_edges_match(got, whole)


def test_background_undersized_halo_seams_negative_control():
    """TEETH: halo == r (not 2r) is undersized by ×2 → it MUST seam."""
    radius = 15
    vol = _vol((48, 48, 48), seed=2)
    under = radius                                # the pre-fix undersized halo (r)
    assert under < ops._halo_background(radius=radius)
    whole = ops.background(vol, radius=radius)
    got = chunked.block_parallel_apply(
        lambda a: ops.background(a, radius=radius), vol, under, block=24
    )
    assert not np.array_equal(got, whole), (
        "undersized (halo=r) must seam — the true reach is 2r"
    )


# --------------------------------------------------------------------------- #
# the adaptive (production) block path also matches whole (no explicit block)
# --------------------------------------------------------------------------- #

def test_gaussian_adaptive_block_path_matches_whole():
    """The PRODUCTION path (adaptive block edge, real multi-block tiling) — a
    volume larger than one block on Y/X → chunks via the default heuristic and is
    still byte-identical."""
    sigma = 1.5
    vol = _vol((40, 300, 300), seed=3)   # 3.6 Mvox > small-size gate, Y/X > 256
    edge = chunked._adaptive_block_edge(ops._halo_gaussian(sigma))
    assert edge < max(vol.shape), "test volume must exceed one adaptive block"
    assert vol.size >= chunked._SMALL_ARRAY_VOXELS, "volume must exceed small-size gate"
    whole = ops.gaussian(vol, sigma=sigma)
    got = chunked.block_parallel_apply(
        lambda a: ops.gaussian(a, sigma=sigma), vol, ops._halo_gaussian(sigma)
    )
    assert np.array_equal(got, whole)


# --------------------------------------------------------------------------- #
# small-array path — below the size gate → whole-array call, identical, no crash
# --------------------------------------------------------------------------- #

def test_small_array_short_circuits_and_is_identical():
    vol = _vol((8, 8, 8), seed=4)
    halo = ops._halo_gaussian(1.5)                # 6, but the array is < one block
    got = chunked.block_parallel_apply(
        lambda a: ops.gaussian(a, sigma=1.5), vol, halo
    )
    assert np.array_equal(got, ops.gaussian(vol, sigma=1.5))


def test_halo_zero_and_0d_short_circuit():
    vol = _vol((16, 16, 16), seed=5)
    # halo <= 0 → no neighborhood → whole-array op call.
    got = chunked.block_parallel_apply(lambda a: a * 2.0, vol, 0)
    assert np.array_equal(got, (vol * 2.0).astype(np.float32))


def test_unsupported_boundary_raises():
    vol = _vol((8, 8, 8), seed=6)
    with pytest.raises(ValueError):
        chunked.block_parallel_apply(lambda a: a, vol, 4, boundary="reflect")


# --------------------------------------------------------------------------- #
# execute(parallel=True) == execute(parallel=False) — per channel, array_equal
# --------------------------------------------------------------------------- #

def test_execute_parallel_matches_serial_local_recipe():
    """The whole-volume seam: a [gaussian, background] recipe run with the LOCAL
    ops routed block-parallel (parallel=True) must be BYTE-IDENTICAL, per channel,
    to the whole-array path (parallel=False).  The volume is above the small-size
    gate + wider than one adaptive block, so gaussian genuinely chunks."""
    # 2 DISTINCT channels; (24,300,300)=2.16 Mvox > small_size, Y/X > block(256).
    chans = [_vol((24, 300, 300), seed=10), (_vol((24, 300, 300), seed=11) * 0.6)]
    assert chans[0].size >= chunked._SMALL_ARRAY_VOXELS
    assert chunked._adaptive_block_edge(ops._halo_gaussian(1.5)) < 300
    recipe = Recipe(
        channels=[
            [IntensityStep("gaussian", {"sigma": 1.5}),
             IntensityStep("background", {"radius": 3})]
            for _ in range(2)
        ],
        geometry=None,
    )
    par, _sp_p, _st_p = ops.execute(recipe, chans, (1.0, 1.0, 1.0), parallel=True)
    ser, _sp_s, _st_s = ops.execute(recipe, chans, (1.0, 1.0, 1.0), parallel=False)
    assert len(par) == len(ser) == 2
    for i, (p, s) in enumerate(zip(par, ser)):
        np.testing.assert_array_equal(
            p, s, err_msg=f"channel {i}: parallel != serial (a seam)"
        )


def test_execute_parallel_matches_serial_small_array():
    """Below the size gate, parallel=True falls through to whole-array everywhere
    → still identical + no crash."""
    chans = [_vol((6, 8, 7), seed=12)]
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"sigma": 1.2}),
                   IntensityStep("background", {"radius": 2})]],
        geometry=None,
    )
    par, _, _ = ops.execute(recipe, chans, (1.0, 1.0, 1.0), parallel=True)
    ser, _, _ = ops.execute(recipe, chans, (1.0, 1.0, 1.0), parallel=False)
    np.testing.assert_array_equal(par[0], ser[0])


# --------------------------------------------------------------------------- #
# Brick 3a #1 — the raised block FLOOR (_MIN_BLOCK 64 -> 256) is byte-identical
# --------------------------------------------------------------------------- #

def test_min_block_floor_is_256_with_six_halo_lower_bound():
    """The floor is 256; ``6·halo`` still governs LARGER halos and 512 still caps."""
    assert chunked._MIN_BLOCK == 256
    assert chunked._adaptive_block_edge(6) == 256        # 6·6=36 -> floored
    assert chunked._adaptive_block_edge(50) == 300       # 6·halo rule wins
    assert chunked._adaptive_block_edge(200) == 512      # cap still binds


def test_block_64_and_256_are_byte_identical():
    """The floor change is a PURE performance knob: the same volume tiled at the
    old 64 edge and the new 256 edge must produce bit-identical results (and both
    must equal the whole-array call)."""
    sigma = 1.5
    halo = ops._halo_gaussian(sigma)
    vol = _vol((40, 300, 300), seed=77)      # > 256 on Y/X -> 256 genuinely tiles
    op = lambda a: ops.gaussian(a, sigma=sigma)  # noqa: E731
    whole = ops.gaussian(vol, sigma=sigma)
    at64 = chunked.block_parallel_apply(op, vol, halo, block=64)
    at256 = chunked.block_parallel_apply(op, vol, halo, block=256)
    assert np.array_equal(at64, at256)
    assert np.array_equal(at256, whole)
    _assert_edges_match(at256, whole)


def test_adaptive_production_path_is_byte_identical():
    """The PRODUCTION (no explicit block) path - which since Brick 3c #4 also
    applies the cancel-granularity cap - stays byte-identical."""
    sigma = 1.5
    halo = ops._halo_gaussian(sigma)
    vol = _vol((40, 300, 300), seed=78)
    assert chunked._adaptive_block_edge(halo) == 256          # uncapped value
    got = chunked.block_parallel_apply(
        lambda a: ops.gaussian(a, sigma=sigma), vol, halo)
    assert np.array_equal(got, ops.gaussian(vol, sigma=sigma))


# --------------------------------------------------------------------------- #
# Brick 3c #4 - the cancel-granularity cap on the adaptive edge
# --------------------------------------------------------------------------- #

def test_cap_is_inert_without_shape_and_workers():
    """The RAM budget calls `_adaptive_block_edge(halo)` bare and must keep
    getting the LARGEST edge the engine could pick (over-estimating the tile is
    safe; under-estimating it would blow the budget)."""
    assert chunked._adaptive_block_edge(6) == 256
    assert chunked._adaptive_block_edge(6, shape=(143, 1024, 1024)) == 256
    assert chunked._adaptive_block_edge(6, workers=20) == 256


def test_cap_cuts_the_edge_so_several_waves_exist():
    """1024 / (3 waves x 20 workers) = 18 -> below the soft floor, so 128."""
    edge = chunked._adaptive_block_edge(6, shape=(143, 1024, 1024), workers=20)
    assert edge == chunked._MIN_TILE_FOR_WAVES == 128
    grid = 1
    for d in (143, 1024, 1024):
        grid *= -(-d // edge)
    inflight = 20 * chunked._INFLIGHT_PER_WORKER
    assert grid >= chunked._WAVES_WANTED * 20
    assert grid > inflight, (
        "the grid must exceed the in-flight window or Cancel has nothing to drop"
    )


def test_without_the_cap_the_same_pass_is_one_wave():
    """NEGATIVE control: at the uncapped 256 edge every tile fits in the
    in-flight window, which is exactly the stuck-Cancel case."""
    edge = chunked._adaptive_block_edge(6)
    grid = 1
    for d in (143, 1024, 1024):
        grid *= -(-d // edge)
    assert grid == 16
    assert grid < 20 * chunked._INFLIGHT_PER_WORKER


def test_cap_never_breaks_the_halo_lower_bound():
    """A large-halo op must keep its `6*halo` edge - the halo-overhead volume
    ratio is a cost property and outranks cancel latency (a 128 tile with halo 50
    would read (228/128)^3 = 5.6x the interior)."""
    for halo, expected in ((6, 128), (30, 180), (50, 300), (200, 512)):
        assert chunked._adaptive_block_edge(
            halo, shape=(143, 1024, 1024), workers=20) == expected, halo


def test_cap_relaxes_when_there_are_few_workers():
    """One worker already means one tile at a time - no wave problem to solve."""
    assert chunked._adaptive_block_edge(
        6, shape=(143, 1024, 1024), workers=1) == 256


def test_cap_tolerates_a_junk_shape():
    assert chunked._adaptive_block_edge(6, shape=(), workers=20) == 256
    assert chunked._adaptive_block_edge(6, shape=(0, 0), workers=20) == 256
    assert chunked._adaptive_block_edge(6, shape=("x",), workers=20) == 256


def test_capped_production_path_is_byte_identical():
    """The cap changes only HOW the array is traversed."""
    sigma = 1.5
    halo = ops._halo_gaussian(sigma)
    vol = _vol((24, 400, 400), seed=79)
    got = chunked.block_parallel_apply(
        lambda a: ops.gaussian(a, sigma=sigma), vol, halo, max_workers=4)
    whole = ops.gaussian(vol, sigma=sigma)
    assert np.array_equal(got, whole)
    _assert_edges_match(got, whole)


def test_raised_floor_flows_through_the_ram_budget():
    """The floor MUST be visible to the scratch-RAM budget (that is what makes a
    small box back off to fewer workers instead of blowing up)."""
    recipe = {"channels": [[{"key": "gaussian", "params": {"sigma": 1.5}}]],
              "geometry": None}
    tile = ops.max_local_tile_scratch_bytes(recipe)
    edge = chunked._adaptive_block_edge(ops._halo_gaussian(1.5))
    halo = ops._halo_gaussian(1.5)
    assert tile == (edge + 2 * halo) ** 3 * 4 * 3
    assert tile == 268 ** 3 * 12      # ~0.23 GB/tile at the new floor (was ~11 MB)


def test_block_64_and_256_identical_when_no_axis_is_short():
    """The Brick 3a case used (40, 300, 300) - only Y/X exceed 256, so Z was a
    single block and the Z halo seam was never exercised at the 256 edge.  This
    volume is >= 256 on EVERY axis, so both edges tile in all three dimensions."""
    sigma = 1.5
    halo = ops._halo_gaussian(sigma)
    vol = _vol((260, 260, 260), seed=81)
    op = lambda a: ops.gaussian(a, sigma=sigma)  # noqa: E731
    whole = ops.gaussian(vol, sigma=sigma)
    at64 = chunked.block_parallel_apply(op, vol, halo, block=64, max_workers=4)
    at256 = chunked.block_parallel_apply(op, vol, halo, block=256, max_workers=4)
    for axis_tiling in (at64, at256):
        assert np.array_equal(axis_tiling, whole)
    assert np.array_equal(at64, at256)
    _assert_edges_match(at256, whole)

"""
Tests for the pure flush-anchored tile-grid planner (Stack 5).

Adapted from TDD list to the flush-no-pad design. Everything here
is pure geometry (no IO, no GPU). The two BLOCKING acceptance gates
(output-shape-equals-input + instance-registration) are tile->assemble
round-trips and live in tests/test_flush_roundtrip.py.
"""

import pytest

from packages.epicycle_scaledseg.utils.tile_grid import (
    TileGeom,
    TileGrid,
    compute_tile_grid,
    resolve_step,
)


# ---------------------------------------------------------------------------
# resolve_step — fraction vs pixels
# ---------------------------------------------------------------------------


def test_resolve_step_fraction():
    step, ov = resolve_step(16, 0.25)
    assert step == 12 and ov == 4  # int(16*0.75)=12


def test_resolve_step_pixels():
    step, ov = resolve_step(16, 4)  # 4 px overlap
    assert step == 12 and ov == 4


def test_resolve_step_zero_overlap():
    step, ov = resolve_step(16, 0.0)
    assert step == 16 and ov == 0


def test_resolve_step_floored_at_one():
    step, _ = resolve_step(10, 0.999)  # would be int(10*0.001)=0
    assert step == 1


def test_resolve_step_negative_raises():
    with pytest.raises(ValueError):
        resolve_step(16, -0.1)


# ---------------------------------------------------------------------------
# compute_tile_grid — coverage / no-pad / flush
# ---------------------------------------------------------------------------


def test_grid_covers_full_extent():
    """Every pixel in [0,H)x[0,W) is in at least one tile; nothing exceeds the
    boundary; nothing is padded beyond the image."""
    grid = compute_tile_grid(20, 20, 16, 0.25)
    H = W = 20
    covered_y = [False] * H
    covered_x = [False] * W
    for t in grid.tiles:
        y0, x0 = t.origin
        h, w = t.extent
        assert y0 + h <= H and x0 + w <= W  # never exceeds boundary (no pad)
        for y in range(y0, y0 + h):
            covered_y[y] = True
        for x in range(x0, x0 + w):
            covered_x[x] = True
    assert all(covered_y) and all(covered_x)


def test_no_dropped_remainder():
    """The classic failure of the old perfect-tiling constraint: H=7000,
    patch=1024 must be fully covered with the last tile flush to 7000."""
    grid = compute_tile_grid(7000, 7000, 1024, 0.25)
    max_y1 = max(t.origin[0] + t.extent[0] for t in grid.tiles)
    max_x1 = max(t.origin[1] + t.extent[1] for t in grid.tiles)
    assert max_y1 == 7000 and max_x1 == 7000  # flush to the boundary, no overrun


def test_single_tile_when_smaller_than_patch():
    """dim < patch -> ONE tile whose extent is the real image size (NOT padded
    up to patch). The model only ever sees real data."""
    grid = compute_tile_grid(8, 10, 16, 0.25)
    assert grid.n_rows == 1 and grid.n_cols == 1
    t = grid.tiles[0]
    assert t.origin == (0, 0)
    assert t.extent == (8, 10)  # = (dim_y, dim_x), sub-patch
    assert not t.is_flush


def test_single_tile_when_equal_to_patch():
    grid = compute_tile_grid(16, 16, 16, 0.25)
    assert grid.n_rows == 1 and grid.n_cols == 1
    assert grid.tiles[0].extent == (16, 16)
    assert not grid.tiles[0].is_flush


def test_overlap_fraction_vs_pixels_equivalent():
    """overlap=0.25 on patch 16 (step 12) == overlap_px=4 -> identical origins."""
    frac = compute_tile_grid(50, 50, 16, 0.25)
    px = compute_tile_grid(50, 50, 16, 4)
    assert [t.origin for t in frac.tiles] == [t.origin for t in px.tiles]
    assert frac.step == px.step == 12


def test_tilespec_origins_exact():
    """Hand-derived origin sequence (incl. the flush anchor) for (20,20),16,0.25:
    step=12 -> y/x origins [0, 4(flush)]."""
    grid = compute_tile_grid(20, 20, 16, 0.25)
    origins = {t.grid_index: t.origin for t in grid.tiles}
    assert origins == {
        (0, 0): (0, 0), (0, 1): (0, 4),
        (1, 0): (4, 0), (1, 1): (4, 4),
    }


def test_flush_tile_flagged():
    """The boundary-anchored tile (origin != idx*step) is flagged is_flush."""
    grid = compute_tile_grid(20, 20, 16, 0.25)  # step=12; second origin 4 != 12
    flush = {t.grid_index for t in grid.tiles if t.is_flush}
    # any tile with a flush row (i=1) or flush col (j=1) is flush.
    assert flush == {(0, 1), (1, 0), (1, 1)}
    # (0,0) sits on the regular grid -> not flush.
    assert not next(t for t in grid.tiles if t.grid_index == (0, 0)).is_flush


def test_flush_anchor_coincides_with_grid_no_flush_flag():
    """When (dim-patch) IS a multiple of step, the last tile lands on the grid
    and is NOT flagged flush (no extra overlap). (40,40),16,0.25 -> [0,12,24]."""
    grid = compute_tile_grid(40, 40, 16, 0.25)
    y_origins = sorted({t.origin[0] for t in grid.tiles})
    assert y_origins == [0, 12, 24]  # 24 == 2*step, exact
    assert not any(t.is_flush for t in grid.tiles)


def test_neighbor_overlap_region_matches():
    """Adjacent (non-flush) tiles share exactly overlap_px in the seam — the
    region the Mode-B IoU stitch will read. (40,40),16,0.25 -> overlap 4px."""
    grid = compute_tile_grid(40, 40, 16, 0.25)
    by_idx = {t.grid_index: t for t in grid.tiles}
    t0, t1 = by_idx[(0, 0)], by_idx[(0, 1)]  # x origins 0 and 12
    overlap = (t0.origin[1] + t0.extent[1]) - t1.origin[1]  # 16 - 12
    assert overlap == grid.overlap_px == 4


def test_max_origin_never_exceeds_boundary():
    """Fuzz a range of awkward dims: no tile ever exceeds the image."""
    for dim in (17, 31, 100, 257, 1000, 4097):
        grid = compute_tile_grid(dim, dim, 64, 0.25)
        for t in grid.tiles:
            assert t.origin[0] + t.extent[0] <= dim
            assert t.origin[1] + t.extent[1] <= dim
        # and full coverage: the last origin is flush.
        assert max(t.origin[0] + t.extent[0] for t in grid.tiles) == dim


def test_dim_patch_plus_one():
    """Smallest non-trivial flush case: dim=patch+1 -> origins [0, 1]."""
    grid = compute_tile_grid(17, 17, 16, 0.25)
    y_origins = sorted({t.origin[0] for t in grid.tiles})
    assert y_origins == [0, 1]
    assert max(t.origin[0] + t.extent[0] for t in grid.tiles) == 17


def test_one_axis_sub_patch_mixed_grid():
    """One axis sub-patch (H=8 < patch), the other multi-tile (W=40). Every tile
    gets extent (8, 16); the H axis is a single row, W tiles normally."""
    grid = compute_tile_grid(8, 40, 16, 0.25)
    assert grid.n_rows == 1 and grid.n_cols == 3  # W origins [0,12,24]
    assert all(t.extent == (8, 16) for t in grid.tiles)
    # full coverage on both axes
    assert max(t.origin[0] + t.extent[0] for t in grid.tiles) == 8
    assert max(t.origin[1] + t.extent[1] for t in grid.tiles) == 40


def test_overlap_one_float_is_pixels():
    """overlap == 1.0 (>= 1) is treated as 1 PIXEL of overlap, not a fraction."""
    step, ov = resolve_step(16, 1.0)
    assert step == 15 and ov == 1


def test_zero_overlap_tiles_abut():
    """overlap=0 -> step==patch -> tiles abut with no overlap, full coverage."""
    grid = compute_tile_grid(48, 48, 16, 0.0)
    y_origins = sorted({t.origin[0] for t in grid.tiles})
    assert y_origins == [0, 16, 32]
    assert grid.overlap_px == 0


def test_compute_tile_grid_rejects_bad_inputs():
    with pytest.raises(ValueError):
        compute_tile_grid(20, 20, 0, 0.25)      # patch <= 0
    with pytest.raises(ValueError):
        compute_tile_grid(0, 20, 16, 0.25)      # dim <= 0
    with pytest.raises(ValueError):
        compute_tile_grid(20, 20, 16, -0.1)     # overlap < 0


def test_grid_index_matches_position():
    """grid_index (i,j) is consistent with sorted origin order."""
    grid = compute_tile_grid(50, 50, 16, 0.25)
    y_origins = sorted({t.origin[0] for t in grid.tiles})
    x_origins = sorted({t.origin[1] for t in grid.tiles})
    for t in grid.tiles:
        i, j = t.grid_index
        assert t.origin == (y_origins[i], x_origins[j])

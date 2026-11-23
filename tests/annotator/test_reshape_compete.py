"""Cells from markers -- the competition with a growth limit (``reshape/compete.py``).

Pure pytest on small synthetic volumes -- no Qt, no napari.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from packages.epicycle_annotator.reshape import ReshapeCancelled
from packages.epicycle_annotator.reshape import compete as C

ISO = (1.0, 1.0, 1.0)

#: The local driver data (a read-only Wound patch copy); tests that need it
#: skip when it is absent.
DRIVERS = Path(os.environ.get(
    "EPICYCLE_DERIVE_DRIVERS", Path.home() / "Desktop" / "EpiCycle-drivers"))


def _flat(shape, value=0.0):
    return np.full(shape, value, np.float32)


def _all(shape):
    return np.ones(shape, bool)


# --------------------------------------------------------------------------- #
# the step cost and the limit
# --------------------------------------------------------------------------- #
def test_a_step_costs_its_length_times_the_mean_price():
    elev = np.array([0.0, 0.5, 0.0, 0.0], np.float32).reshape(1, 1, 4)
    mk = np.array([1, 0, 0, 0], np.int32).reshape(1, 1, 4)
    lab, cost = C.compete(elev, mk, _all(elev.shape), (1.0, 1.0, 2.0),
                          wall_cost=40.0, growth_limit=100.0)
    # x-steps are 2 um long; prices 1, 21, 1, 1
    assert cost[0, 0, 0] == 0.0
    assert cost[0, 0, 1] == pytest.approx(2.0 * (1 + 21) / 2)
    assert cost[0, 0, 2] == pytest.approx(22.0 + 2.0 * (21 + 1) / 2)
    assert cost[0, 0, 3] == pytest.approx(44.0 + 2.0)
    assert (lab == 1).all()


def test_a_diagonal_step_is_its_physical_length():
    elev = _flat((2, 2, 1))
    mk = np.zeros((2, 2, 1), np.int32)
    mk[0, 0, 0] = 1
    _lab, cost = C.compete(elev, mk, _all(elev.shape), (2.0, 1.0, 1.0),
                           growth_limit=10.0)
    assert cost[1, 1, 0] == pytest.approx(np.sqrt(4.0 + 1.0), rel=1e-6)
    _lab, cost6 = C.compete(elev, mk, _all(elev.shape), (2.0, 1.0, 1.0),
                            growth_limit=10.0, neighbours=6)
    assert cost6[1, 1, 0] == pytest.approx(3.0)


def test_a_voxel_is_claimed_only_while_its_total_is_within_the_limit():
    elev = _flat((1, 1, 30))
    mk = np.zeros(elev.shape, np.int32)
    mk[0, 0, 0] = 3
    lab, cost = C.compete(elev, mk, _all(elev.shape), ISO, growth_limit=14.0)
    assert (lab[0, 0, :15] == 3).all()          # cost 14 is still in
    assert (lab[0, 0, 15:] == 0).all()
    assert np.isinf(cost[0, 0, 15:]).all()
    assert cost.dtype == np.float32 and lab.dtype == np.int32


def test_a_membrane_of_0_6_one_um_thick_is_never_crossed_at_40_and_14():
    sp = (0.5, 0.5, 0.5)
    shape = (16, 16, 40)
    elev = _flat(shape, 0.05)
    elev[:, :, 20:22] = 0.6                      # 2 voxels = 1 um
    mk = np.zeros(shape, np.int32)
    mk[6:10, 6:10, 14:18] = 1                     # left of the wall
    lab, _cost = C.compete(elev, mk, _all(shape), sp, wall_cost=40.0,
                           growth_limit=14.0)
    assert (lab[:, :, 22:] == 0).all()
    assert (lab[:, :, 19] == 1).any()             # it reached the wall
    mk[6:10, 6:10, 26:30] = 2                     # a rival right of it
    lab, _cost = C.compete(elev, mk, _all(shape), sp)
    assert not (lab[:, :, :20] == 2).any()
    assert not (lab[:, :, 22:] == 1).any()


def test_wall_cost_0_gives_the_nearest_marker_by_physical_distance():
    # on (2, 1, 1) spacing a marker 3 slices away in z (6 um) loses to one
    # 5 voxels away in y (5 um)
    shape = (9, 13, 3)
    rng = np.random.default_rng(0)
    elev = rng.random(shape).astype(np.float32)   # ignored at wall cost 0
    mk = np.zeros(shape, np.int32)
    mk[0, 6, 1] = 1                               # z = 0
    mk[3, 11, 1] = 2                              # y = 11
    lab, cost = C.compete(elev, mk, _all(shape), (2.0, 1.0, 1.0),
                          wall_cost=0.0, growth_limit=100.0)
    assert lab[3, 6, 1] == 2
    assert cost[3, 6, 1] == pytest.approx(5.0)
    # on a straight line the winner is simply the nearer one
    line = _flat((1, 1, 10))
    m = np.zeros(line.shape, np.int32)
    m[0, 0, 0], m[0, 0, 9] = 7, 4
    lab, _ = C.compete(line, m, _all(line.shape), ISO, wall_cost=0.0)
    assert lab[0, 0].tolist() == [7] * 5 + [4] * 5


def _reference(elev, markers, space, spacing, wall_cost, neighbours=26):
    """Every marker's cost map by scipy's Dijkstra on the same graph."""
    shape = elev.shape
    idx = np.arange(elev.size).reshape(shape)
    price = 1.0 + wall_cost * elev.astype(np.float64)
    rows, cols, vals = [], [], []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                order = abs(dz) + abs(dy) + abs(dx)
                if order == 0 or (neighbours == 6 and order != 1):
                    continue
                if neighbours == 18 and order == 3:
                    continue
                length = np.sqrt((dz * spacing[0]) ** 2 + (dy * spacing[1]) ** 2
                                 + (dx * spacing[2]) ** 2)
                src = tuple(slice(max(0, -d), n - max(0, d)) for d, n in zip((dz, dy, dx), shape))
                dst = tuple(slice(max(0, d), n - max(0, -d)) for d, n in zip((dz, dy, dx), shape))
                ok = space[src] & space[dst]
                rows.append(idx[src][ok])
                cols.append(idx[dst][ok])
                vals.append((length * 0.5 * (price[src] + price[dst]))[ok])
    g = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(elev.size, elev.size)).tocsr()
    ids = [int(v) for v in np.unique(markers) if v]
    maps = []
    for i in ids:
        src = np.flatnonzero((markers.ravel() == i) & space.ravel())
        d = dijkstra(g, indices=src, min_only=True) if src.size else np.full(elev.size, np.inf)
        maps.append(d)
    return np.array(ids), np.array(maps)


@pytest.mark.parametrize("spacing,neighbours", [(ISO, 26), ((2.0, 0.7, 1.0), 26),
                                                (ISO, 6), ((2.0, 0.7, 1.0), 6),
                                                (ISO, 18), ((2.0, 0.7, 1.0), 18)])
def test_the_solver_agrees_with_an_independent_dijkstra(spacing, neighbours):
    rng = np.random.default_rng(3)
    shape = (7, 11, 12)
    elev = ndi.gaussian_filter(rng.random(shape), 1.0).astype(np.float32)
    elev = (elev - elev.min()) / (elev.max() - elev.min())
    space = rng.random(shape) > 0.1
    mk = np.zeros(shape, np.int32)
    for i, p in enumerate(rng.integers(0, [7, 11, 12], size=(5, 3)), start=1):
        mk[tuple(p)] = i * 3
    limit = 9.0
    lab, cost = C.compete(elev, mk, space, spacing, wall_cost=5.0,
                          growth_limit=limit, neighbours=neighbours)
    ids, maps = _reference(elev, mk, space, spacing, 5.0, neighbours)
    best = maps.min(axis=0)
    order = np.sort(maps, axis=0)
    clear = (order[1] - order[0] > 1e-3) if len(ids) > 1 else np.ones(best.shape, bool)
    reach = best <= limit - 1e-3
    assert np.allclose(cost.ravel()[reach], best[reach], rtol=1e-5, atol=1e-4)
    win = ids[maps.argmin(axis=0)]
    assert (lab.ravel()[reach & clear] == win[reach & clear]).all()
    assert (lab.ravel()[best > limit + 1e-3] == 0).all()


# --------------------------------------------------------------------------- #
# ties, space, kept objects
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("left,right", [(1, 2), (2, 1), (2_147_483_647, 5)])
def test_an_exact_tie_goes_to_the_lower_id_whichever_side_it_is_on(left, right):
    shape = (9, 9, 11)
    elev = _flat(shape, 0.1)
    mk = np.zeros(shape, np.int32)
    mk[:, :, 0] = left
    mk[:, :, 10] = right
    lab, cost = C.compete(elev, mk, _all(shape), ISO, growth_limit=40.0)
    mid = lab[:, :, 5]
    assert np.allclose(cost[:, :, 5], cost[:, :, 5].flat[0])
    assert (mid == min(left, right)).all()
    assert (lab[:, :, :5] == left).all() and (lab[:, :, 6:] == right).all()


def test_voxels_outside_the_space_are_impassable_and_never_claimed():
    shape = (6, 10, 20)
    elev = _flat(shape)
    existing = np.zeros(shape, np.int32)
    existing[:, :, 10] = 9                         # a thin kept sheet, wall to wall
    mk = np.zeros(shape, np.int32)
    mk[3, 5, 2] = 1
    lab, cost = C.compete(elev, mk, existing == 0, ISO, wall_cost=0.0,
                          growth_limit=100.0)
    assert (lab[:, :, :10] == 1).all()
    assert (lab[:, :, 10:] == 0).all()
    assert np.isinf(cost[:, :, 10:]).all()


def test_a_marker_voxel_outside_the_space_is_ignored():
    shape = (3, 3, 3)
    mk = np.zeros(shape, np.int32)
    mk[1, 1, 1] = 4
    space = _all(shape)
    space[1, 1, 1] = False
    lab, _ = C.compete(_flat(shape), mk, space, ISO)
    assert not lab.any()


def test_kept_competitors_keep_a_new_cell_off_their_side_of_a_gap():
    shape = (5, 5, 21)
    elev = _flat(shape)
    existing = np.zeros(shape, np.int32)
    existing[:, :, :4] = 123_456_789
    mk = np.zeros(shape, np.int32)
    mk[:, :, 18] = 1
    space = existing == 0
    lab_block, _ = C.compete(elev, mk, space, ISO, wall_cost=0.0, growth_limit=50.0)
    assert (lab_block[:, :, 4:] == 1).all()        # blocking only: takes it all
    sources, n_new = C.kept_competitors(mk, existing, space)
    assert n_new == 1
    assert (sources[:, :, 4] == 2).all() and (sources[:, :, :4] == 0).all()
    lab, _ = C.compete(elev, sources, space, ISO, wall_cost=0.0, growth_limit=50.0)
    lab = np.where(lab <= n_new, lab, 0)
    assert (lab[:, :, :4] == 0).all()              # never written on a kept voxel
    assert (lab[:, :, 4:11] == 0).all()            # the kept cell's side of the gap
    assert (lab[:, :, 12:] == 1).all()


def test_kept_competitors_number_kept_objects_by_ascending_id():
    shape = (3, 3, 9)
    existing = np.zeros(shape, np.uint32)
    existing[:, :, 0] = 4_000_000_000
    existing[:, :, 8] = 7
    mk = np.zeros(shape, np.int32)
    mk[1, 1, 4] = 2
    sources, n_new = C.kept_competitors(mk, existing, existing == 0)
    assert n_new == 2
    assert (sources[:, :, 7] == 3).all()           # id 7 -> first competitor
    assert (sources[:, :, 1] == 4).all()           # the huge id -> second


# --------------------------------------------------------------------------- #
# the membrane step
# --------------------------------------------------------------------------- #
def test_the_membrane_step_hands_voxels_only_to_new_objects():
    sp = (0.5, 0.5, 0.5)
    shape = (3, 3, 13)
    labels = np.zeros(shape, np.int32)
    existing = np.zeros(shape, np.int32)
    labels[:, :, :3] = 5                            # a new object, x < 3
    existing[:, :, 10:] = 8                         # a kept one, x >= 10
    elev = _flat(shape, 0.6)
    elev[:, :, 4] = 0.1                             # dim: never handed
    before = labels.copy(), existing.copy(), elev.copy()
    out = C.include_membrane(labels, elevation=elev, existing=existing,
                             space=_all(shape), spacing_um=sp)
    row = out[1, 1]
    assert row[3] == 5                              # 0.5 um from the new object
    assert row[4] == 0                              # dim
    assert row[5] == 0                              # 1.5 um away: too far
    assert row[8] == 0 and row[9] == 0              # nearest is the kept object
    assert (out[:, :, 10:] == 0).all()              # never on a kept voxel
    assert (out[:, :, :3] == 5).all()
    for a, b in zip(before, (labels, existing, elev)):
        assert np.array_equal(a, b)                 # inputs untouched
    assert out.dtype == np.int32


def test_the_membrane_step_measures_the_distance_over_new_and_kept():
    sp = ISO
    shape = (1, 1, 5)
    labels = np.zeros(shape, np.int32)
    existing = np.zeros(shape, np.int32)
    labels[0, 0, 0] = 2
    existing[0, 0, 2] = 9
    out = C.include_membrane(labels, elevation=_flat(shape, 0.9), existing=existing,
                             space=_all(shape), spacing_um=sp, within_um=2.0)
    # x = 1: 1 um from both -> the kept object wins the tie, stays empty
    assert out[0, 0, 1] == 0
    labels2 = np.zeros(shape, np.int32)
    labels2[0, 0, 0], labels2[0, 0, 2] = 6, 3
    out = C.include_membrane(labels2, elevation=_flat(shape, 0.9),
                             existing=np.zeros(shape, np.int32), space=_all(shape),
                             spacing_um=sp)
    assert out[0, 0, 1] == 3                        # a new-new tie -> lower id


def test_the_membrane_step_stays_inside_the_space():
    shape = (1, 1, 4)
    labels = np.zeros(shape, np.int32)
    labels[0, 0, 0] = 1
    space = _all(shape)
    space[0, 0, 1] = False
    kw = dict(elevation=_flat(shape, 0.9), existing=np.zeros(shape, np.int32),
              spacing_um=ISO)
    out = C.include_membrane(labels, space=space, **kw)
    assert out[0, 0, 1] == 0
    out = C.include_membrane(labels, space=_all(shape), **kw)
    assert out[0, 0, 1] == 1


def test_the_membrane_step_takes_keywords_only_and_checks_them():
    shape = (2, 3, 4)
    lab = np.zeros(shape, np.int32)
    e, ex, s = _flat(shape, 0.5), np.zeros(shape, np.int32), _all(shape)
    with pytest.raises(TypeError):
        C.include_membrane(lab, e, ex, ISO)                   # positional: refused
    with pytest.raises(TypeError):
        C.include_membrane(lab, elevation=e, existing=ex, spacing_um=ISO)  # no space
    kw = dict(elevation=e, existing=ex, space=s, spacing_um=ISO)
    with pytest.raises(ValueError, match="float wall map"):
        C.include_membrane(lab, **dict(kw, elevation=np.ones(shape, np.uint8)))
    bad = e.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match=">= 0"):
        C.include_membrane(lab, **dict(kw, elevation=bad))
    with pytest.raises(ValueError, match="shape"):
        C.include_membrane(lab, **dict(kw, elevation=e[:, :, :3]))
    with pytest.raises(ValueError, match="existing must hold integer ids"):
        C.include_membrane(lab, **dict(kw, existing=ex.astype(np.float32)))
    with pytest.raises(ValueError, match="existing must hold integer ids"):
        C.include_membrane(lab, **dict(kw, existing=ex.astype(bool)))
    with pytest.raises(ValueError, match="boolean"):
        C.include_membrane(lab, **dict(kw, space=s.astype(np.uint8)))


# --------------------------------------------------------------------------- #
# slivers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_a_marker_mostly_over_kept_objects_is_dropped(dtype):
    shape = (1, 1, 30)
    mk = np.zeros(shape, dtype)
    empty = np.zeros(shape, bool)
    big = 2_147_483_647
    mk[0, 0, 0:10] = 3                              # 6 of 10 on empty: kept
    empty[0, 0, 0:6] = True
    mk[0, 0, 10:20] = big                           # 4 of 10: dropped
    empty[0, 0, 10:14] = True
    mk[0, 0, 20:30] = 9                             # exactly half: kept
    empty[0, 0, 20:25] = True
    kept, n = C.drop_sliver_markers(mk, empty)
    assert n == 1
    assert kept.dtype == mk.dtype
    assert (kept[0, 0, 0:10] == 3).all()            # not clipped to empty
    assert not kept[0, 0, 10:20].any()
    assert (kept[0, 0, 20:30] == 9).all()
    assert (mk[0, 0, 10:20] == big).all()           # the input untouched


# --------------------------------------------------------------------------- #
# ids and inputs
# --------------------------------------------------------------------------- #
def test_uint32_markers_and_the_largest_id_are_fine():
    shape = (3, 3, 6)
    mk = np.zeros(shape, np.uint32)
    mk[1, 1, 0] = 2_147_483_647
    mk[1, 1, 5] = 2
    lab, _ = C.compete(_flat(shape), mk, _all(shape), ISO)
    assert lab.dtype == np.int32
    assert lab[1, 1, 0] == 2_147_483_647
    mk[1, 1, 5] = 2_147_483_648
    with pytest.raises(ValueError, match="largest"):
        C.compete(_flat(shape), mk, _all(shape), ISO)


@pytest.mark.parametrize("kwargs,match", [
    (dict(growth_limit=0.0), "growth_limit"),
    (dict(growth_limit=float("nan")), "growth_limit"),
    (dict(wall_cost=-1.0), "wall_cost"),
    (dict(neighbours=8), "neighbours"),
])
def test_bad_settings_are_refused_in_plain_words(kwargs, match):
    shape = (2, 2, 2)
    with pytest.raises(ValueError, match=match):
        C.compete(_flat(shape), np.ones(shape, np.int32), _all(shape), ISO, **kwargs)


def test_bad_volumes_are_refused_in_plain_words():
    shape = (2, 3, 4)
    e, m, s = _flat(shape), np.zeros(shape, np.int32), _all(shape)
    with pytest.raises(ValueError, match="shape"):
        C.compete(e, m[:, :, :3], s, ISO)
    with pytest.raises(ValueError, match="integer"):
        C.compete(e, m.astype(np.float32), s, ISO)
    with pytest.raises(ValueError, match="negative"):
        C.compete(e, m - 1, s, ISO)
    with pytest.raises(ValueError, match="boolean"):
        C.compete(e, m, s.astype(np.uint8), ISO)
    bad = e.copy()
    bad[0, 0, 0] = -0.1
    with pytest.raises(ValueError, match=">= 0"):
        C.compete(bad, m, s, ISO)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match=">= 0"):
        C.compete(bad, m, s, ISO)
    with pytest.raises(ValueError, match="spacing"):
        C.compete(e, m, s, (1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="3D"):
        C.compete(e[0], m[0], s[0], ISO)


def test_compete_never_writes_its_inputs():
    rng = np.random.default_rng(1)
    shape = (6, 7, 8)
    e = rng.random(shape).astype(np.float32)
    m = np.zeros(shape, np.int32)
    m[3, 3, 3] = 1
    s = rng.random(shape) > 0.2
    for a in (e, m, s):
        a.setflags(write=False)
    C.compete(e, m, s, ISO)
    C.include_membrane(m, elevation=e, existing=np.zeros(shape, np.int32), space=s,
                       spacing_um=ISO)


def test_a_cancel_before_the_solve_raises():
    ev = threading.Event()
    ev.set()
    shape = (2, 2, 2)
    with pytest.raises(ReshapeCancelled):
        C.compete(_flat(shape), np.ones(shape, np.int32), _all(shape), ISO, cancel=ev)


def test_without_numba_the_error_names_it(monkeypatch):
    monkeypatch.setattr(C, "_KERNELS", {})
    monkeypatch.setitem(sys.modules, "numba", None)
    shape = (2, 2, 2)
    with pytest.raises(ImportError, match="numba"):
        C.compete(_flat(shape), np.ones(shape, np.int32), _all(shape), ISO)
    assert C.have_numba() is False
    assert C._KERNELS == {}                         # a later call tries again


def test_a_numba_cache_failure_is_said_in_plain_words(monkeypatch):
    import numba

    def no_locator(*_a, **_k):
        def deco(fn):
            raise RuntimeError("cannot cache function {!r}: no locator available "
                               "for file 'x'".format(fn.__name__))
        return deco

    monkeypatch.setattr(C, "_KERNELS", {})
    monkeypatch.setattr(numba, "njit", no_locator)
    shape = (2, 2, 2)
    with pytest.raises(RuntimeError) as info:
        C.compete(_flat(shape), np.ones(shape, np.int32), _all(shape), ISO)
    assert str(info.value) == C.NUMBA_CACHE_FAILED
    assert "NUMBA_CACHE_DIR" in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)   # the original kept
    assert C.have_numba() is False
    assert C._KERNELS == {}


def test_have_numba_is_true_here():
    assert C.have_numba() is True


# --------------------------------------------------------------------------- #
# tiles
# --------------------------------------------------------------------------- #
def test_the_halo_and_the_budget_follow_the_growth_limit():
    assert C.tile_halo((0.5, 0.5, 0.5), 14.0) == (32, 32, 32)
    assert C.tile_halo((2.0, 0.5, 0.5), 14.0) == (10, 32, 32)   # per axis
    assert C.tile_halo(ISO, 3.0) == (6, 6, 6)
    tiles = C.plan_tiles((81, 1024, 880), (0.5, 0.5, 0.5), 14.0)
    assert len(tiles) == 6                                # 1 x 3 x 2
    for core, halo in tiles:
        assert np.prod([s.stop - s.start for s in halo]) <= 16_000_000
    # a large limit raises the budget to 4 hz x 4 hy x 4 hx
    h = C.tile_halo(ISO, 40.0)
    big = C.plan_tiles((200, 200, 200), ISO, 40.0, budget=1)
    cap = np.prod([4 * v for v in h])
    assert max(np.prod([s.stop - s.start for s in hs]) for _c, hs in big) <= cap


def test_a_coarse_z_gets_a_thinner_z_halo():
    tiles = C.plan_tiles((200, 512, 512), (2.0, 0.5, 0.5), 14.0, budget=1)
    for core, halo in tiles:
        for axis, width in ((0, 10), (1, 32), (2, 32)):
            n = (200, 512, 512)[axis]
            assert core[axis].start - halo[axis].start == min(width, core[axis].start)
            assert halo[axis].stop - core[axis].stop == min(width, n - core[axis].stop)


@pytest.mark.parametrize("shape", [(81, 512, 512), (400, 512, 512), (5, 7, 3)])
def test_the_tile_cores_partition_the_volume(shape):
    tiles = C.plan_tiles(shape, (0.5, 0.5, 0.5), 14.0)
    seen = np.zeros(shape, np.uint8) if np.prod(shape) < 2e8 else None
    total = 0
    for core, halo in tiles:
        total += int(np.prod([s.stop - s.start for s in core]))
        for c, h, n in zip(core, halo, shape):
            assert 0 <= h.start <= c.start < c.stop <= h.stop <= n
        if seen is not None:
            seen[core] += 1
    assert total == int(np.prod(shape))
    if seen is not None:
        assert (seen == 1).all()


def _landscape(shape, seed):
    rng = np.random.default_rng(seed)
    elev = ndi.gaussian_filter(rng.random(shape), 1.5).astype(np.float32)
    elev = ((elev - elev.min()) / (elev.max() - elev.min())) ** 2
    space = ndi.gaussian_filter(rng.random(shape), 2.0) > 0.47
    mk = np.zeros(shape, np.int32)
    pts = rng.integers(0, shape, size=(60, 3))
    for i, p in enumerate(pts, start=1):
        mk[tuple(slice(max(0, v - 1), v + 1) for v in p)] = i
    return elev, mk, space


def test_tiles_equal_one_call_voxel_for_voxel():
    sp = (1.0, 0.8, 0.9)
    shape = (44, 50, 47)
    elev, mk, space = _landscape(shape, 5)
    limit = 3.0
    tiles = C.plan_tiles(shape, sp, limit, budget=1)
    cores = [c for c, _h in tiles]
    for axis in range(3):
        assert len({c[axis].start for c in cores}) >= 2   # >= 2 x 2 x 2 tiles
    one_lab, one_cost = C.compete(elev, mk, space, sp, wall_cost=10.0, growth_limit=limit)
    lab, cost = C.compete_tiled(elev, mk, space, sp, wall_cost=10.0,
                                growth_limit=limit, budget=1, return_cost=True)
    assert one_lab.any()
    assert np.array_equal(lab, one_lab)
    assert np.array_equal(cost, one_cost)
    only = C.compete_tiled(elev, mk, space, sp, wall_cost=10.0, growth_limit=limit,
                           budget=1)
    assert isinstance(only, np.ndarray) and np.array_equal(only, one_lab)


def test_the_halo_is_exactly_wide_enough_along_a_coarse_z():
    """One marker, wall cost 0, limit 14 on (2, 0.5, 0.5): the farthest claim
    is exactly 7 slices away in z.  With the core boundary there, tiles
    equal one call, and a z halo of 6 slices would not."""
    sp = (2.0, 0.5, 0.5)
    shape = (30, 24, 24)
    e = _flat(shape)
    mk = np.zeros(shape, np.int32)
    mk[8, 12, 12] = 1
    space = _all(shape)
    one, _ = C.compete(e, mk, space, sp, wall_cost=0.0, growth_limit=14.0)
    reach = np.flatnonzero(one.any(axis=(1, 2)))
    assert (reach.min(), reach.max()) == (1, 15)          # 7 slices each way
    assert C.tile_halo(sp, 14.0)[0] == 10

    def tiled(hz):
        out = np.zeros(shape, np.int32)
        for z0, z1 in ((0, 15), (15, 30)):                # the core boundary at z 15
            halo = (slice(max(0, z0 - hz), min(30, z1 + hz)), slice(0, 24), slice(0, 24))
            lab, _ = C.compete(e[halo], mk[halo], space[halo], sp, wall_cost=0.0,
                               growth_limit=14.0)
            out[z0:z1] = lab[z0 - halo[0].start:z1 - halo[0].start]
        return out

    assert np.array_equal(tiled(C.tile_halo(sp, 14.0)[0]), one)
    assert np.array_equal(tiled(7), one)
    assert not np.array_equal(tiled(6), one)              # the test has teeth
    # and the planner itself, cut at that boundary
    lab = C.compete_tiled(e, mk, space, sp, wall_cost=0.0, growth_limit=14.0,
                          budget=1)
    assert np.array_equal(lab, one)


def test_one_tile_is_a_single_call(monkeypatch):
    shape = (6, 7, 8)
    elev, mk, space = _landscape(shape, 2)
    calls = []
    real = C.compete

    def spy(*a, **k):
        calls.append(a[0].shape)
        return real(*a, **k)

    monkeypatch.setattr(C, "compete", spy)
    progress = []
    lab, cost = C.compete_tiled(elev, mk, space, ISO, growth_limit=3.0,
                                progress=lambda i, k: progress.append((i, k)),
                                return_cost=True)
    assert calls == [shape]
    assert progress == [(1, 1)]
    one_lab, one_cost = real(elev, mk, space, ISO, growth_limit=3.0)
    assert np.array_equal(lab, one_lab) and np.array_equal(cost, one_cost)


def test_a_cancel_is_honoured_between_tiles():
    shape = (40, 40, 40)
    elev, mk, space = _landscape(shape, 6)
    ev = threading.Event()
    calls = []

    def progress(i, k):
        calls.append((i, k))
        if i == 2:
            ev.set()

    with pytest.raises(ReshapeCancelled):
        C.compete_tiled(elev, mk, space, ISO, growth_limit=3.0, budget=1,
                        cancel=ev, progress=progress)
    assert [i for i, _k in calls] == [1, 2]
    assert calls[0][1] > 2


@pytest.mark.parametrize("exc", [ReshapeCancelled, KeyError])
def test_a_raising_progress_callback_stops_the_run(exc):
    shape = (40, 40, 40)
    elev, mk, space = _landscape(shape, 6)
    calls = []

    def progress(i, k):
        calls.append(i)
        raise exc("stop")

    with pytest.raises(exc):
        C.compete_tiled(elev, mk, space, ISO, growth_limit=3.0, budget=1,
                        progress=progress)
    assert calls == [1]


def test_a_stop_inside_one_solve_is_answered(monkeypatch):
    """The cancel set while the compiled solver runs stops it mid-solve
    (here the flag is read every 1024 pops so the test stays small)."""
    monkeypatch.setattr(C, "_STOP_CHECK_MASK", 1023)
    shape = (40, 120, 120)
    elev = _flat(shape, 0.0)
    mk = np.zeros(shape, np.int32)
    mk[20, 60, 60] = 1
    space = _all(shape)
    C.compete(elev[:4, :4, :4], mk[:4, :4, :4], space[:4, :4, :4], ISO)  # compiled
    t = time.perf_counter()
    C.compete(elev, mk, space, ISO, growth_limit=1000.0)
    full = time.perf_counter() - t

    class SetAfterFirstLook:
        """Not set at the poll before the solve, set from then on."""

        def __init__(self):
            self.n = 0

        def is_set(self):
            self.n += 1
            return self.n > 1

    ev = SetAfterFirstLook()
    t = time.perf_counter()
    with pytest.raises(ReshapeCancelled):
        C.compete(elev, mk, space, ISO, growth_limit=1000.0, cancel=ev)
    stopped = time.perf_counter() - t
    assert ev.n >= 2                                   # the watcher looked
    assert stopped < max(0.5 * full, 0.3), (stopped, full)


def test_the_stop_flag_never_changes_an_answer(monkeypatch):
    monkeypatch.setattr(C, "_STOP_CHECK_MASK", 0)       # read at every pop
    shape = (44, 50, 47)
    elev, mk, space = _landscape(shape, 5)
    ev = threading.Event()
    a = C.compete(elev, mk, space, (1.0, 0.8, 0.9), wall_cost=10.0, growth_limit=3.0,
                  cancel=ev)
    b = C.compete(elev, mk, space, (1.0, 0.8, 0.9), wall_cost=10.0, growth_limit=3.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def _wound_case():
    root = DRIVERS / "derive_d0" / "ws"
    files = [root / "patches" / "Wound_patch_1.tiff",
             root / "nuclei" / "Wound_patch_1" / "labels_3d.tif"]
    if not all(f.is_file() for f in files):
        pytest.skip("the Wound patch driver copy is not on this machine")
    import tifffile

    from packages.epicycle_annotator.reshape import region as R
    from packages.epicycle_annotator.reshape.walls import build_elevation

    img = tifffile.imread(str(files[0]))[:, :, :128, :128]
    nuc = tifffile.imread(str(files[1]))[:, :128, :128]
    sp = (0.5, 0.5, 0.5)
    elev, _, _ = build_elevation("membrane", np.asarray(img[1], np.float32), None,
                                 spacing=sp, clip=0.01, smooth=1.0, ridge=False)
    cores, _ = R.eroded_cores(nuc, erode_um=1.0, spacing=sp)
    mk, _n = R._relabel_ascending(cores)
    return np.asarray(elev, np.float32), mk, sp


def test_tiles_equal_one_call_on_a_real_patch():
    elev, mk, sp = _wound_case()
    space = np.ones(elev.shape, bool)
    # a limit of 6 keeps the halo small enough for this crop to need tiles
    one, _ = C.compete(elev, mk, space, sp, growth_limit=6.0)
    tiled = C.compete_tiled(elev, mk, space, sp, growth_limit=6.0, budget=1)
    assert len(C.plan_tiles(elev.shape, sp, 6.0, budget=1)) >= 4
    assert one.any()
    assert np.array_equal(one, tiled)

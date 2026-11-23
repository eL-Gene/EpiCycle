"""Inside each cell: two holes in the recipe's rules, closed.

1. With the smoothing ticked, a nucleus must still end at its owner cell's
   border (cells own the membrane; nuclei are cut at the cell border, with
   a chip).  The smoothing can round a nucleus back across the border, so
   every new nucleus is cut to its owner cell after the smoothing, and the
   chips ("cut at edge", "two in a cell") and the second-nucleus guard are
   decided on the shape that ships.
2. On a box, "a cell that already owns a kept object is skipped" must see
   the kept objects of each cell over the whole patch, not only the box's
   crop: a kept nucleus just outside the box still means the cell has one.

No Qt, no napari.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from packages.epicycle_annotator import add_derive as AD
from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator import add_segment as AS
from packages.epicycle_annotator.add_derive import (
    DeriveResult,
    nuclei_in_cells,
    owning_cells,
)

ISO = (0.5, 0.5, 0.5)
ANISO = (1.0, 0.5, 0.5)


def _inside_params(**kw):
    p = dict(look_for=AP.LOOK_INSIDE, spacing_known=True, auto_smooth=False,
             smooth_sigma=1.5, clear_border=True, edge_margin=3)
    p.update(kw)
    return p


def _whole(shape):
    return tuple(slice(0, n) for n in shape)


# --------------------------------------------------------------------------- #
# 1. the smoothing never carries a nucleus past its owner cell
# --------------------------------------------------------------------------- #
def _finger_scene():
    """One cell; a neighbour's finger (1 voxel wide in x, 2 in z) reaches
    into the middle of its nucleus."""
    Z, Y, X = 24, 48, 48
    sp = ANISO
    rng = np.random.default_rng(2)
    nuc = rng.normal(10, 1, (Z, Y, X)).astype(np.float32)
    zz, yy, xx = np.indices((Z, Y, X)).astype(float)
    ball = (((zz - 12) * sp[0]) ** 2 + ((yy - 24) * sp[1]) ** 2
            + ((xx - 24) * sp[2]) ** 2 <= 3.5 ** 2)
    nuc[ball] += 100
    cells = np.zeros((Z, Y, X), np.uint32)
    cells[2:22, 4:44, 4:44] = 1001
    cells[11:13, 4:25, 24:25] = 2002   # the finger
    cells[2:22, 4:8, 4:44] = 2002      # its body at the y-front
    return nuc, cells, sp


def _finger_run(smooth):
    nuc, cells, sp = _finger_scene()
    out = AP.run_add_preview(
        preset="nuclei", roi=_whole(cells.shape),
        existing_roi=np.zeros(cells.shape, np.int32), next_label=500,
        spacing=sp, params=_inside_params(auto_smooth=smooth),
        image_roi=nuc, run_guide_roi=cells, id_max=2 ** 32 - 1)
    return out, cells


def _outside_owner(out, cells):
    return sum(int(np.count_nonzero((out.result == n) & (cells != c)))
               for n, c in out.owner.items())


@pytest.mark.parametrize("smooth", [False, True])
def test_a_nucleus_stays_inside_its_owner_cell_smoothed_or_not(smooth):
    out, cells = _finger_run(smooth)
    assert out.ok, out.refusal
    assert out.ids
    assert _outside_owner(out, cells) == 0
    assert int(np.count_nonzero((out.result > 0) & (cells == 2002))) == 0
    for n in out.ids:
        assert n in out.owner


def test_the_smoothing_still_rounds_the_nucleus():
    """The cut runs after the smoothing; it does not undo it."""
    raw, _ = _finger_run(False)
    smoothed, _ = _finger_run(True)
    assert not np.array_equal(raw.result > 0, smoothed.result > 0)


# -- the chips and the guard on the shape that ships (fakes) --------------- #
# cells along x: A = 11 (x < 20), B = 22 (20 <= x < 40), C = 33 (x >= 40)
_SHAPE = (10, 20, 60)


def _abc_cells():
    cells = np.zeros(_SHAPE, np.uint32)
    cells[:, :, :20] = 11
    cells[:, :, 20:40] = 22
    cells[:, :, 40:] = 33
    return cells


def _abc_result():
    """The recipe's answer: in A a big nucleus (1) and a second one (2,
    49 um^3 at 0.5 um voxels: above the 40 um^3 guard); one nucleus in B
    (3); one in C (4), flagged cut by the recipe."""
    lab = np.zeros(_SHAPE, np.int32)
    lab[1:9, 2:18, 2:10] = 1      # 1024 voxels, 128 um^3
    lab[1:9, 2:9, 12:19] = 2      # 392 voxels, 49 um^3
    lab[1:9, 12:18, 24:32] = 3    # 384 voxels
    lab[1:9, 4:12, 44:52] = 4
    return DeriveResult(labels=lab, owner={1: 11, 2: 11, 3: 22, 4: 33},
                        two_in_cell={11: [1, 2]}, cut_at_edge={4},
                        n_cells=3, n_found=4)


def _shift_smooth(moves):
    """A stand-in smoothing that moves the named (minted) ids along x."""

    def smooth(result, sigma=1.5, **_kw):
        out = result.copy()
        for i in moves:
            out[result == i] = 0
        for i, dx in moves.items():
            m = np.roll(result == i, dx, axis=2)
            out[m & (out == 0)] = i
        return out

    return smooth


def _abc_run(monkeypatch, *, smooth, moves=None, res=None):
    res = res if res is not None else _abc_result()
    monkeypatch.setattr(AD, "nuclei_in_cells", lambda *a, **k: res)
    monkeypatch.setattr(AS, "smooth_pieces", _shift_smooth(moves or {}))
    cells = _abc_cells()
    out = AP.run_add_preview(
        preset="nuclei", roi=_whole(_SHAPE),
        existing_roi=np.zeros(_SHAPE, np.int32), next_label=500,
        spacing=ISO, params=_inside_params(auto_smooth=smooth,
                                           clear_border=False),
        image_roi=np.zeros(_SHAPE, np.float32), run_guide_roi=cells,
        id_max=2 ** 31 - 1)
    assert out.ok, out.refusal
    return out, cells


def test_without_the_smoothing_the_recipes_chips_ship_as_they_were(
        monkeypatch):
    out, cells = _abc_run(monkeypatch, smooth=False)
    assert out.ids == [500, 501, 502, 503]
    assert out.two_in_cell == {11: [500, 501]}
    assert out.cut_at_edge == {503}
    assert out.flags == {500: ("two",), 501: ("two",), 503: ("cut",)}


def test_a_nucleus_the_smoothing_pushes_over_the_border_is_cut_and_says_so(
        monkeypatch):
    # 502 moves 7 voxels into A: 3 of its 8 columns (37.5 %) leave B
    out, cells = _abc_run(monkeypatch, smooth=True, moves={502: -7})
    assert _outside_owner(out, cells) == 0
    assert int(np.count_nonzero(out.result[:, :, :20] == 502)) == 0
    assert out.sizes[502] == 8 * 6 * 5
    assert "cut" in out.flags[502]
    assert 502 in out.cut_at_edge
    # the recipe's own cut chip is kept
    assert "cut" in out.flags[503]


def test_a_small_push_over_the_border_is_cut_without_a_chip(monkeypatch):
    # 1 of 8 columns (12.5 %) leaves B: cut, but under a quarter
    out, cells = _abc_run(monkeypatch, smooth=True, moves={502: -5})
    assert _outside_owner(out, cells) == 0
    assert 502 not in out.cut_at_edge
    assert "cut" not in out.flags.get(502, ())


def test_the_cut_after_the_smoothing_adds_to_the_recipes_own_cut(
        monkeypatch):
    # the recipe already cut 20 % off nucleus 3; the smoothing pushes one of
    # its 8 columns over the border: 48 voxels of the 480 it had before any
    # cut is 10 % more, 30 % in all, over the quarter
    res = _abc_result()
    res.cut_share = {3: 0.2}
    out, cells = _abc_run(monkeypatch, smooth=True, moves={502: -5},
                          res=res)
    assert "cut" in out.flags[502]
    # alone, the same column (12.5 %) is no chip
    out, cells = _abc_run(monkeypatch, smooth=True, moves={502: -5})
    assert "cut" not in out.flags.get(502, ())


def test_the_guard_is_not_judged_again_on_the_smoothed_cut_shape(
        monkeypatch):
    # 501 (second in A, 49 um^3 as the recipe found it) moves 3 voxels into
    # B: the cut leaves 8 x 7 x 5 voxels = 35 um^3, under 40 um^3 -- the
    # guard was judged on the recipe's shape, so both still ship
    out, cells = _abc_run(monkeypatch, smooth=True, moves={501: 3})
    assert 501 in out.ids and 500 in out.ids
    assert out.two_in_cell == {11: [500, 501]}
    assert out.sizes[501] == 8 * 7 * 5
    assert _outside_owner(out, cells) == 0


def _pair_scene(sp, r2):
    shape = (30, 60, 100)
    lab = np.zeros(shape, np.int32)
    lab[_ball(shape, sp, (15, 30, 25), 3.5)] = 1
    lab[_ball(shape, sp, (15, 30, 60), r2)] = 2
    cells = np.zeros(shape, np.uint32)
    cells[2:28, 5:55, 5:95] = 77
    return lab, cells


def test_a_pair_above_the_guard_ships_whole_after_the_real_smoothing(
        monkeypatch):
    """182 + 58 um^3 in one cell: both above 40 um^3 as found; the real
    smoothing shrinks the small one, and it still ships."""
    sp = ANISO
    lab, cells = _pair_scene(sp, 2.4)
    vox = float(np.prod(sp))
    assert (lab == 2).sum() * vox > AD.SECOND_MIN_UM3
    res = DeriveResult(labels=lab, owner={1: 77, 2: 77},
                       two_in_cell={77: [1, 2]}, n_cells=1, n_found=2)
    monkeypatch.setattr(AD, "nuclei_in_cells", lambda *a, **k: res)
    out = AP.run_add_preview(
        preset="nuclei", roi=_whole(lab.shape),
        existing_roi=np.zeros(lab.shape, np.int32), next_label=500,
        spacing=sp, params=_inside_params(auto_smooth=True,
                                          clear_border=False),
        image_roi=np.zeros(lab.shape, np.float32), run_guide_roi=cells,
        id_max=2 ** 31 - 1)
    assert out.ok, out.refusal
    assert out.ids == [500, 501]
    assert out.two_in_cell == {77: [500, 501]}


def test_the_owner_cut_keeps_ids_it_has_no_owner_for():
    r = np.zeros((2, 2, 4), np.int32)
    r[..., :2] = 7
    r[..., 2:] = 9
    guide = np.zeros(r.shape, np.uint32)
    guide[..., :3] = 1
    out, lost, dropped = AP._clip_to_owner(r, guide, {7: 1})
    assert np.array_equal(out, r)
    assert lost == {} and dropped == []


def test_the_owner_cut_takes_huge_ids_and_uint64_cells():
    big = 2 ** 31 - 2
    r = np.zeros((2, 2, 4), np.int64)
    r[..., :3] = big
    guide = np.zeros(r.shape, np.uint64)
    guide[..., :2] = 2 ** 40
    out, lost, dropped = AP._clip_to_owner(r, guide, {big: 2 ** 40})
    assert int(np.count_nonzero(out == big)) == 2 * 2 * 2
    assert lost == {big: 4}
    assert dropped == []


def test_a_wall_through_a_nucleus_leaves_one_piece():
    """A neighbour's wall cuts the smoothed nucleus in two: the larger
    piece stays, as the recipe keeps every nucleus in one piece."""
    from scipy import ndimage as ndi

    guide = np.ones((1, 9, 9), np.int32)
    guide[0, :, 4] = 2
    r = np.zeros_like(guide)
    r[0, 3:6, 2:8] = 100        # 2 columns left of the wall, 3 right
    out, lost, dropped = AP._clip_to_owner(r, guide, {100: 1})
    assert ndi.label(out == 100)[1] == 1
    assert int(np.count_nonzero(out[0, :, 5:] == 100)) == 9
    assert int(np.count_nonzero(out[0, :, :4] == 100)) == 0
    assert lost == {100: 9}
    assert dropped == []


def test_a_piece_under_the_floor_is_left_out():
    guide = np.ones((1, 9, 9), np.int32)
    guide[0, :, 4:] = 2
    r = np.zeros_like(guide)
    r[0, 3:6, 2:8] = 100        # 6 voxels stay in cell 1, 12 cross
    out, lost, dropped = AP._clip_to_owner(r, guide, {100: 1},
                                           min_voxels=7)
    assert int(np.count_nonzero(out)) == 0
    assert dropped == [100]
    assert lost == {100: 18}


def test_a_nucleus_the_cut_empties_is_not_said_vanished_when_smoothed(
        monkeypatch):
    # 502 moves wholly into C: nothing of it is left in its owner B
    out, cells = _abc_run(monkeypatch, smooth=True, moves={502: 16})
    assert 502 not in out.ids
    assert out.n_emptied_by_smooth == 0
    assert any("owner cell" in m and "502" in m for m in out.messages), (
        out.messages)


def test_a_box_whose_patch_arrays_disagree_is_a_plain_refusal():
    nuc, cells, target, sp = _long_cell_scene()
    out = AP.run_add_preview(
        preset="nuclei", roi=_BOX, existing_roi=target[_BOX].copy(),
        next_label=500, spacing=sp, params=_inside_params(),
        image_roi=nuc[_BOX], run_guide_roi=cells[_BOX],
        id_max=2 ** 31 - 1, run_guide_patch=cells,
        existing_patch=target[:, :, :40])
    assert out.status == "refused"
    assert out.refusal


# --------------------------------------------------------------------------- #
# 2. on a box, a cell's kept nucleus outside the box still counts
# --------------------------------------------------------------------------- #
def _ball(shape, sp, c, r):
    zz, yy, xx = np.indices(shape).astype(float)
    return (((zz - c[0]) * sp[0]) ** 2 + ((yy - c[1]) * sp[1]) ** 2
            + ((xx - c[2]) * sp[2]) ** 2 <= r * r)


def _long_cell_scene(*, kept_at=48, second_cell=False):
    """One long cell (4242) along x; its kept nucleus at ``kept_at`` (x);
    a second bright body at x ~ 16.  With ``second_cell`` a cell 5151
    wholly inside x < 30 holds a bright body of its own."""
    shape = (24, 48, 64)
    sp = ANISO
    nuc = np.random.default_rng(4).normal(10, 1, shape).astype(np.float32)
    cells = np.zeros(shape, np.uint32)
    target = np.zeros(shape, np.int32)
    if second_cell:
        cells[2:22, 4:24, 4:60] = 4242
        cells[2:22, 24:44, 4:28] = 5151
        y_long = 14
        nuc[_ball(shape, sp, (12, 34, 16), 3.0)] += 100
    else:
        cells[2:22, 4:44, 4:60] = 4242
        y_long = 24
    kept = _ball(shape, sp, (12, y_long, kept_at), 3.0)
    target[kept] = 7
    nuc[kept] += 100
    nuc[_ball(shape, sp, (12, y_long, 16), 3.0)] += 100
    return nuc, cells, target, sp


def _box_run(nuc, cells, target, sp, roi, *, patch=True, whole=False):
    kw = {}
    if patch:
        kw = dict(run_guide_patch=cells, existing_patch=target)
    return AP.run_add_preview(
        preset="nuclei", roi=roi, existing_roi=target[roi].copy(),
        next_label=500, spacing=sp,
        params=_inside_params(whole_patch=whole), image_roi=nuc[roi],
        run_guide_roi=cells[roi], id_max=2 ** 31 - 1, **kw)


_BOX = (slice(0, 24), slice(0, 48), slice(0, 30))


def test_a_kept_nucleus_outside_the_box_still_means_the_cell_has_one():
    nuc, cells, target, sp = _long_cell_scene()
    out = _box_run(nuc, cells, target, sp, _BOX)
    assert out.status == "refused"
    assert out.refusal == AP.ADD_S1_INSIDE_ALL_TAKEN
    assert out.n_cell_has_one == 1
    assert out.result is None


def test_the_whole_patch_says_the_same():
    nuc, cells, target, sp = _long_cell_scene()
    roi = _whole(cells.shape)
    out = _box_run(nuc, cells, target, sp, roi, patch=False, whole=True)
    assert out.status == "refused"
    assert out.n_cell_has_one == 1


def test_a_kept_nucleus_inside_the_box_is_still_honoured():
    nuc, cells, target, sp = _long_cell_scene(kept_at=24)
    for patch in (True, False):
        out = _box_run(nuc, cells, target, sp, _BOX, patch=patch)
        assert out.status == "refused", (patch, out.ids)
        assert out.n_cell_has_one == 1


def test_only_the_cell_that_owns_one_is_skipped():
    nuc, cells, target, sp = _long_cell_scene(second_cell=True)
    out = _box_run(nuc, cells, target, sp, _BOX)
    assert out.ok, out.refusal
    assert set(out.owner.values()) == {5151}
    assert out.n_cell_has_one == 1


def test_a_whole_patch_never_measures_the_owners_twice(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("measured again on a whole patch")

    monkeypatch.setattr(AD, "owning_cells", boom)
    nuc, cells, target, sp = _long_cell_scene()
    roi = _whole(cells.shape)
    out = _box_run(nuc, cells, target, sp, roi, patch=True, whole=True)
    assert out.n_cell_has_one == 1


def test_the_patch_owners_give_the_same_answer_as_the_crop_on_a_whole_patch():
    nuc, cells, target, sp = _long_cell_scene(second_cell=True)
    a = nuclei_in_cells(nuc, cells, target, sp)
    b = nuclei_in_cells(nuc, cells, target, sp,
                        owning=owning_cells(cells, target))
    assert np.array_equal(a.labels, b.labels)
    assert a.owner == b.owner
    assert a.n_cell_has_one == b.n_cell_has_one == 1


# -- owning_cells itself ----------------------------------------------------- #
def test_owning_cells_by_majority_with_at_least_half_inside():
    cells = np.zeros((1, 1, 20), np.uint32)
    cells[..., 0:10] = 3
    cells[..., 10:16] = 9
    t = np.zeros(cells.shape, np.int64)
    t[..., 6:12] = 5          # 4 in cell 3, 2 in cell 9: cell 3
    t[..., 14:20] = 6         # 2 in cell 9, 4 outside the cells: nobody
    assert owning_cells(cells, t) == {3}


def test_owning_cells_breaks_a_tie_to_the_higher_cell():
    cells = np.zeros((1, 1, 8), np.uint32)
    cells[..., :4] = 3
    cells[..., 4:] = 9
    t = np.zeros(cells.shape, np.int32)
    t[..., 2:6] = 5
    assert owning_cells(cells, t) == {9}


def test_owning_cells_among_measures_whole_objects_of_those_cells_only():
    cells = np.zeros((1, 1, 30), np.uint32)
    cells[..., 0:10] = 3
    cells[..., 10:20] = 9
    cells[..., 20:30] = 12
    t = np.zeros(cells.shape, np.uint32)
    t[..., 8:16] = 4_000_000_000   # 2 in 3, 6 in 9: owned by 9
    t[..., 22:25] = 77             # owned by 12
    assert owning_cells(cells, t, among=[3]) == set()
    assert owning_cells(cells, t, among=[3, 9]) == {9}
    assert owning_cells(cells, t, among=[12]) == {12}
    assert owning_cells(cells, t) == {9, 12}


def test_owning_cells_with_nothing_kept():
    cells = np.ones((2, 2, 2), np.uint32)
    assert owning_cells(cells, np.zeros(cells.shape, np.int32)) == set()
    assert owning_cells(cells, np.zeros(cells.shape, np.int32),
                        among=[1]) == set()


def test_owning_cells_cost_on_a_256_cube():
    """The extra cost a box pays: the copy of the run being edited the page
    hands the worker, and the owners of the box's cells over the patch."""
    n = 256
    rng = np.random.default_rng(0)
    g = np.arange(n) // 16
    cid = (g[:, None, None] * 256 + g[None, :, None] * 16
           + g[None, None, :] + 1).astype(np.int32)
    perm = rng.permutation(4096).astype(np.uint32)
    cells = perm[cid - 1] * 1000 + 7
    target = np.zeros((n, n, n), np.int32)
    # one kept cube of 6^3 in every cell
    k = (np.arange(n) % 16 >= 5) & (np.arange(n) % 16 < 11)
    target[np.ix_(k, k, k)] = cid[np.ix_(k, k, k)]
    box = (slice(64, 128), slice(64, 128), slice(64, 128))
    t0 = time.perf_counter()
    copy = np.array(target, copy=True)
    t1 = time.perf_counter()
    among = np.unique(cells[box])
    among = among[among > 0]
    got = owning_cells(cells, copy, among=among)
    t2 = time.perf_counter()
    print("[ADD-PERF] 256^3: copy of the edited run {:.1f} ms, owners of "
          "{} box cells over the patch {:.1f} ms".format(
              (t1 - t0) * 1e3, among.size, (t2 - t1) * 1e3))
    assert got == {int(c) for c in among}

"""Pure tests of ``add_derive.nuclei_in_cells`` -- the Add tab's "Inside each
cell" recipe (nuclei first, then owners).

Every scene is drawn in micrometres and rendered at a given voxel spacing:
cells are boxes, nuclei are bright balls (or a bright capsule), the image is
blurred by 0.5 um and carries a little seeded noise.  No Qt, no napari.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from packages.epicycle_annotator import add_derive as ad
from packages.epicycle_annotator.add_derive import (
    CANCEL_EVERY,
    DERIVE_LOOP_CELLS_WITH,
    DERIVE_LOOP_EMPTY_CELLS,
    DERIVE_LOOP_NUCLEI,
    DERIVE_STAGE_CENTRES,
    DERIVE_STAGE_EDGES,
    DERIVE_STAGE_READ,
    DERIVE_STAGE_RESCUE,
    DERIVE_STAGE_TIDY,
    DERIVE_STAGE_TWO,
    DeriveResult,
    PatchLevels,
    nuclei_in_cells,
    patch_levels,
)
from packages.epicycle_annotator.reshape.engine import ReshapeCancelled

ISO = (0.5, 0.5, 0.5)


# --------------------------------------------------------------------------- #
# scenes
# --------------------------------------------------------------------------- #
def render(shape_um, sp, cell_boxes, blobs, *, cyto=10.0, outside=2.0,
           noise=0.3, seed=0, cell_dtype=np.int32):
    """``cell_boxes`` {id: (z0, z1, y0, y1, x0, x1) um}; ``blobs`` a list of
    ``("ball", (z, y, x), r, value)`` or ``("cap", p, q, r, value)`` (um)."""
    n = [int(round(s / p)) for s, p in zip(shape_um, sp)]
    c = [(np.arange(k) + 0.5) * p for k, p in zip(n, sp)]
    Z, Y, X = np.meshgrid(*c, indexing="ij")
    cells = np.zeros(n, cell_dtype)
    for cid, (z0, z1, y0, y1, x0, x1) in cell_boxes.items():
        cells[(Z >= z0) & (Z < z1) & (Y >= y0) & (Y < y1)
              & (X >= x0) & (X < x1)] = cid
    img = np.where(cells > 0, cyto, outside).astype(np.float32)
    for b in blobs:
        if b[0] == "ball":
            _, (cz, cy, cx), r, v = b
            d = np.sqrt((Z - cz) ** 2 + (Y - cy) ** 2 + (X - cx) ** 2)
        else:
            _, p, q, r, v = b
            p = np.asarray(p, float)
            u = np.asarray(q, float) - p
            t = np.clip(((Z - p[0]) * u[0] + (Y - p[1]) * u[1]
                         + (X - p[2]) * u[2]) / float(u @ u), 0, 1)
            d = np.sqrt((Z - p[0] - t * u[0]) ** 2 + (Y - p[1] - t * u[1]) ** 2
                        + (X - p[2] - t * u[2]) ** 2)
        img = np.where(d <= r, np.maximum(img, v), img)
    img = ndi.gaussian_filter(img, [0.5 / p for p in sp])
    img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return img.astype(np.float32), cells


ONE_CELL = {1: (0, 14, 1, 19, 1, 31)}
TWO_CELLS = {1: (0, 14, 1, 19, 0, 16), 2: (0, 14, 1, 19, 16, 32)}
SHAPE = (14, 20, 32)


def gap_scene(sp=ISO):
    """One cell, two nuclei 1.5 um apart with cytoplasm between them."""
    return render(SHAPE, sp, ONE_CELL, [("ball", (7, 10, 12.25), 3.0, 100),
                                        ("ball", (7, 10, 19.75), 3.0, 100)])


def straddle_scene(off_um, *, b_has_own=False, sp=ISO):
    """A nucleus centred ``off_um`` inside cell 1 from the 1|2 border."""
    blobs = [("ball", (7, 10, 16 - off_um), 3.0, 100)]
    if b_has_own:
        blobs.append(("ball", (7, 10, 26), 3.0, 100))
    return render(SHAPE, sp, TWO_CELLS, blobs)


def three_cells_scene(sp=ISO, cell_ids=(1, 2, 3)):
    """Three cells along x: one nucleus, two nuclei (dark gap), one nucleus."""
    a, b, c = cell_ids
    boxes = {a: (0, 14, 1, 19, 0, 14), b: (0, 14, 1, 19, 14, 34),
             c: (0, 14, 1, 19, 34, 48)}
    blobs = [("ball", (7, 10, 7), 3.0, 100),
             ("ball", (7, 10, 20.25), 3.0, 100),
             ("ball", (7, 10, 27.75), 2.6, 100),
             ("ball", (7, 10, 41), 3.0, 100)]
    return render((14, 20, 48), sp, boxes, blobs)


def assert_inside_owner(r, cells):
    for nid, cell in r.owner.items():
        under = np.asarray(cells)[r.labels == nid]
        assert under.size > 0
        assert np.all(under == cell), "nucleus {} leaves cell {}".format(nid, cell)


def assert_compact(r):
    ids = np.unique(r.labels)
    ids = ids[ids > 0]
    assert r.labels.dtype == np.int32
    assert ids.tolist() == list(range(1, len(ids) + 1))
    assert sorted(r.owner) == ids.tolist()


def per_cell(r):
    out = {}
    for nid, c in r.owner.items():
        out.setdefault(c, []).append(nid)
    return out


# --------------------------------------------------------------------------- #
# two nuclei or one
# --------------------------------------------------------------------------- #
def test_dark_gap_gives_two_nuclei_flagged_two_in_a_cell():
    img, cells = gap_scene()
    r = nuclei_in_cells(img, cells, None, ISO)
    assert r.n_nuclei == 2
    assert set(r.owner.values()) == {1}
    assert r.two_in_cell == {1: [1, 2]}
    assert r.n_two_in_cell == 1 and r.n_no_nucleus == 0
    assert_compact(r)
    assert_inside_owner(r, cells)


def test_bright_waist_gives_one_nucleus():
    img, cells = render(SHAPE, ISO, ONE_CELL, [
        ("ball", (7, 10, 12.25), 3.0, 100), ("ball", (7, 10, 19.75), 3.0, 100),
        ("cap", (7, 10, 12.25), (7, 10, 19.75), 2.0, 100)])
    r = nuclei_in_cells(img, cells, None, ISO)
    assert r.n_nuclei == 1
    assert r.two_in_cell == {}
    assert r.n_one_nucleus == 1


def test_gap_knob_decides_two_or_one():
    """The same dark gap merges when the gap share is set below its pass."""
    img, cells = gap_scene()
    assert nuclei_in_cells(img, cells, None, ISO, gap=0.95).n_nuclei == 2
    assert nuclei_in_cells(img, cells, None, ISO, gap=0.30).n_nuclei == 1


# --------------------------------------------------------------------------- #
# owners, the split at the border, the cut flag
# --------------------------------------------------------------------------- #
def _share_in_cell2(off_um):
    """The straddler's share in cell 2, measured whole (cell 2 holds its own
    nucleus, so nothing is split)."""
    img, cells = straddle_scene(off_um, b_has_own=True)
    whole = nuclei_in_cells(img, cells, None, ISO, cut_at_border=False)
    (nid,) = [i for i, c in whole.owner.items() if c == 1]  # the cell holding most
    m = whole.labels == nid
    return float((cells[m] == 2).mean()), whole


def test_straddle_split_when_other_cell_empty_and_share_at_least_30_percent():
    share, _ = _share_in_cell2(0.5)
    assert 0.30 <= share < 0.5
    img, cells = straddle_scene(0.5)
    r = nuclei_in_cells(img, cells, None, ISO)
    assert r.n_split_border == 1
    assert sorted(r.owner.values()) == [1, 2]
    assert r.n_no_nucleus == 0
    assert r.cut_at_edge == set()
    assert_inside_owner(r, cells)
    a, b = sorted(r.owner, key=r.owner.get)
    assert (r.labels == a).sum() > (r.labels == b).sum()   # owner 1 holds most


def test_straddle_not_split_when_other_cell_has_its_own_nucleus():
    img, cells = straddle_scene(0.5, b_has_own=True)
    r = nuclei_in_cells(img, cells, None, ISO)
    assert r.n_split_border == 0
    assert sorted(r.owner.values()) == [1, 2]
    # the straddler went whole to cell 1 and was cut at the border:
    # more than a quarter lost -> flagged
    (straddler,) = [i for i, c in r.owner.items() if c == 1]
    assert r.cut_at_edge == {straddler}
    assert r.n_cut_at_edge == 1
    assert_inside_owner(r, cells)


def test_straddle_under_30_percent_not_split_cell_left_empty():
    share, _ = _share_in_cell2(1.0)
    assert 0.25 < share < 0.30
    img, cells = straddle_scene(1.0)
    r = nuclei_in_cells(img, cells, None, ISO)
    assert r.n_split_border == 0
    assert r.owner == {1: 1}
    assert r.n_no_nucleus == 1
    assert r.cut_at_edge == {1}                     # lost more than a quarter


def test_cut_flag_only_above_a_quarter_lost():
    share, _ = _share_in_cell2(1.5)
    assert share < 0.25
    img, cells = straddle_scene(1.5)
    r = nuclei_in_cells(img, cells, None, ISO)
    assert r.owner == {1: 1}
    assert r.cut_at_edge == set() and r.n_cut_at_edge == 0


def test_whole_mode_keeps_the_nucleus_whole_and_flags_nothing():
    img, cells = straddle_scene(0.5, b_has_own=True)
    r = nuclei_in_cells(img, cells, None, ISO, cut_at_border=False)
    assert r.cut_at_edge == set()
    (straddler,) = [i for i, c in r.owner.items() if c == 1]
    assert (cells[r.labels == straddler] == 2).any()


def _pocket_scene(pocket_id):
    """One nucleus in cell 1 with a 1.5 um cube under its centre that is
    ``pocket_id`` (a small cell nested inside cell 1, or 0 = no cell)."""
    img, cells = render(SHAPE, ISO, ONE_CELL, [("ball", (7, 10, 16), 3.0, 100)])
    cells = cells.copy()
    cells[13:16, 19:22, 31:34] = pocket_id
    return img, cells


@pytest.mark.parametrize("pocket_id", [2, 0], ids=["nested_cell", "background"])
def test_tidy_does_not_refill_a_pocket_outside_the_owner(pocket_id):
    """The cut leaves a hole where the pocket is; filling holes must not
    write the nucleus back into it (27 voxels did before)."""
    img, cells = _pocket_scene(pocket_id)
    pocket = np.zeros(cells.shape, bool)
    pocket[13:16, 19:22, 31:34] = True
    for target in (None, np.zeros(cells.shape, np.int32)):
        r = nuclei_in_cells(img, cells, target, ISO)
        assert r.owner == {1: 1}
        assert not (r.labels[pocket] > 0).any()
        assert_inside_owner(r, cells)
    # whole mode keeps the nucleus whole: the pocket is filled
    w = nuclei_in_cells(img, cells, None, ISO, cut_at_border=False)
    assert (w.labels[pocket] > 0).all()


def test_nucleus_mostly_outside_every_cell_is_dropped_and_counted():
    img, cells = render(SHAPE, ISO, {1: (0, 14, 1, 19, 1, 14)}, [
        ("ball", (7, 10, 15.5), 3.0, 100), ("ball", (7, 10, 6), 3.0, 100)])
    r = nuclei_in_cells(img, cells, None, ISO)
    assert r.n_outside_cells == 1
    assert r.n_nuclei == 1
    assert_inside_owner(r, cells)


def test_rescue_finds_a_nucleus_inside_an_empty_cell():
    """A small nucleus pressed against a big one outside every cell: the
    patch-wide edge step loses it; the cell-only look finds it, inside the
    cell only."""
    img, cells = render(SHAPE, ISO, {1: (0, 14, 1, 19, 1, 16)}, [
        ("ball", (7, 10, 19.5), 3.0, 100), ("ball", (7, 10, 14.5), 2.0, 80)])
    r = nuclei_in_cells(img, cells, None, ISO)
    assert r.n_rescued == 1
    assert r.n_outside_cells == 1
    assert r.owner == {1: 1}
    assert np.all(cells[r.labels > 0] == 1)


# --------------------------------------------------------------------------- #
# sizes, the guard, units
# --------------------------------------------------------------------------- #
SMALL_ONE = {1: (0, 14, 1, 19, 1, 19)}


@pytest.mark.parametrize("s", [0.5, 1.0])
def test_size_floor_is_in_cubic_um_at_both_spacings(s):
    sp = (s, s, s)
    img, cells = render((14, 20, 20), sp, SMALL_ONE,
                        [("ball", (7, 10, 10), 2.0, 100)])
    r = nuclei_in_cells(img, cells, None, sp)
    assert r.n_nuclei == 1
    vol = float((r.labels > 0).sum()) * s ** 3
    assert vol > ad.MIN_NUCLEUS_UM3
    if s == 1.0:
        # under 100 voxels: a floor read in 0.5-um voxels would drop it
        assert (r.labels > 0).sum() < ad.MIN_NUCLEUS_UM3 / 0.125
    assert nuclei_in_cells(img, cells, None, sp, min_um3=vol + 1.0).n_nuclei == 0


def test_second_nucleus_guard_with_and_without():
    img, cells = render(SHAPE, ISO, ONE_CELL, [
        ("ball", (7, 10, 10), 3.0, 100), ("ball", (7, 10, 19), 1.9, 100)])
    off = nuclei_in_cells(img, cells, None, ISO, second_min_um3=None)
    assert off.n_nuclei == 2 and off.two_in_cell == {1: [1, 2]}
    small_um3 = float((off.labels == 2).sum()) * 0.125
    assert small_um3 < ad.SECOND_MIN_UM3
    on = nuclei_in_cells(img, cells, None, ISO)
    assert on.n_nuclei == 1 and on.n_guard_removed == 1
    assert on.two_in_cell == {}
    zero = nuclei_in_cells(img, cells, None, ISO, second_min_um3=0)
    assert zero.n_nuclei == 2


def test_guard_never_drops_a_lone_small_nucleus():
    img, cells = render((14, 20, 20), ISO, SMALL_ONE,
                        [("ball", (7, 10, 10), 1.9, 100)])
    r = nuclei_in_cells(img, cells, None, ISO, second_min_um3=1000.0)
    assert r.n_nuclei == 1 and r.n_guard_removed == 0


def test_unit_conversions_per_axis():
    assert ad._halfw(1.0, (0.5, 0.5, 0.5)) == (2, 2, 2)
    assert ad._halfw(1.0, (1.0, 0.5, 0.25)) == (1, 2, 4)
    assert ad._sig(1.25, (1.0, 0.5, 0.25)) == (1.25, 2.5, 5.0)
    # the tidy opening: a one-voxel cross at 0.5 um, none at 1.0 um, and a
    # flat yx cross when z rounds to 0
    fp = ad._opening_footprint((0.5, 0.5, 0.5))
    assert fp.shape == (3, 3, 3) and fp.sum() == 7
    assert ad._opening_footprint((1.0, 1.0, 1.0)) is None
    fz = ad._opening_footprint((1.0, 0.5, 0.5))
    assert fz.shape == (1, 3, 3) and fz.sum() == 5
    fine = ad._opening_footprint((0.25, 0.5, 0.5))
    assert fine.shape == (5, 3, 3)
    # the rescue's 1 um border shell: an L1 ball in um
    assert ad._l1_ball_um(1.0, (0.5, 0.5, 0.5)).sum() == 25
    assert ad._l1_ball_um(1.0, (1.0, 1.0, 1.0)).sum() == 7


@pytest.mark.parametrize("radius", [1.0, 1.5])
def test_small_radius_at_1um_voxels_finds_the_nuclei(radius):
    """At 1 um voxels the centre window 0.6 r / 1 um rounds down to 0 voxels;
    it must stay at least one voxel, or every voxel is a maximum (hundreds of
    centres, each basin under the size floor, no nucleus found)."""
    sp = (1.0, 1.0, 1.0)
    img, cells = three_cells_scene(sp)
    r = nuclei_in_cells(img, cells, None, sp, radius_um=radius)
    assert {c: len(v) for c, v in per_cell(r).items()} == {1: 1, 2: 2, 3: 1}
    assert r.n_centres <= 30                        # about the nuclei, not hundreds


def test_small_radius_with_coarse_z_does_not_over_seed():
    """At 1.0 um z and 0.5 um yx with r 1.5 um the z window is one voxel, as
    at 0.5 um everywhere: about as many centres as the isotropic scene."""
    iso_img, iso_cells = three_cells_scene(ISO)
    iso = nuclei_in_cells(iso_img, iso_cells, None, ISO, radius_um=1.5)
    sp = (1.0, 0.5, 0.5)
    img, cells = three_cells_scene(sp)
    r = nuclei_in_cells(img, cells, None, sp, radius_um=1.5)
    assert {c: len(v) for c, v in per_cell(r).items()} == {1: 1, 2: 2, 3: 1}
    assert r.n_centres <= 1.5 * iso.n_centres


def test_physical_log_matches_across_spacing():
    """The spot filter is in um: a ball's peak response is the same whether z
    is sampled at 0.5 or 1.0 um."""
    class _NoStop:
        def poll(self):
            pass

    vals = []
    for sp in (ISO, (1.0, 0.5, 0.5)):
        img, _cells = render((14, 20, 20), sp, SMALL_ONE,
                             [("ball", (7, 10, 10), 3.0, 100)], noise=0.0)
        vals.append(float(ad._log_max(img, 2.5, sp, _NoStop()).max()))
    assert vals[1] == pytest.approx(vals[0], rel=0.1)


@pytest.mark.parametrize("sp", [(1.0, 0.5, 0.5), (0.25, 0.5, 0.5),
                                (1.0, 1.0, 1.0)])
def test_every_um_constant_scales_with_spacing(sp):
    """The same physical scene gives the same nuclei per cell and about the
    same volumes in um^3 at every spacing, anisotropic included."""
    ref_img, ref_cells = three_cells_scene(ISO)
    ref = nuclei_in_cells(ref_img, ref_cells, None, ISO)
    img, cells = three_cells_scene(sp)
    r = nuclei_in_cells(img, cells, None, sp)
    counts = lambda res: {c: len(v) for c, v in per_cell(res).items()}  # noqa: E731
    assert counts(ref) == {1: 1, 2: 2, 3: 1}
    assert counts(r) == counts(ref)
    vox = float(np.prod(sp))
    for c in (1, 3):
        v_ref = float((ref.labels == per_cell(ref)[c][0]).sum()) * 0.125
        v = float((r.labels == per_cell(r)[c][0]).sum()) * vox
        assert v == pytest.approx(v_ref, rel=0.25)


# The scenes below each fail when one length is read in voxels of 0.5 um
# instead of um along its own axis (the filter's blur or weights, the centre
# suppression, the 1 um peak window, the 1 um border shell).
def _grid(shape_um, sp):
    n = [int(round(s / p)) for s, p in zip(shape_um, sp)]
    c = [(np.arange(k) + 0.5) * p for k, p in zip(n, sp)]
    return n, np.meshgrid(*c, indexing="ij")


@pytest.mark.parametrize("sp, axis, sep_um, expected", [
    ((1.0, 0.5, 0.5), 0, 2.0, 2),     # 2 voxels of 1 um: farther apart than 1.5 um
    ((0.5, 0.5, 0.5), 0, 2.0, 2),
    ((1.0, 0.5, 0.5), 2, 2.0, 2),
    ((0.5, 0.5, 0.25), 2, 1.0, 1),    # 4 voxels of 0.25 um: closer than 1.5 um
    ((0.5, 0.5, 0.5), 2, 1.0, 1),
])
def test_centre_suppression_is_in_um_along_every_axis(sp, axis, sep_um, expected):
    """Two peaks ``sep_um`` apart; the radius 2.5 um suppresses centres
    closer than 0.6 r = 1.5 um, whatever the axis and its spacing."""
    L = np.zeros((12, 12, 12), np.float32)
    p = [6, 6, 6]
    L[tuple(p)] = 1.0
    p[axis] += int(round(sep_um / sp[axis]))
    L[tuple(p)] = 0.8
    pts = ad._centres(L, np.ones(L.shape, bool), sp, ad._CENTRE_SPACING * 2.5)
    assert len(pts) == expected


@pytest.mark.parametrize("sp, lo, hi", [
    ((1.0, 0.5, 0.5), 33, 40),
    ((0.25, 0.5, 0.5), 46, 62),
    ((1.0, 1.0, 1.0), 21, 29),
])
def test_spot_count_at_a_small_radius_per_spacing(sp, lo, hi):
    """At r 1.0 um the finder also picks up the scene's seeded noise; how
    many spots it finds at each spacing is pinned (36, 54 and 25 when
    measured), so a filter or a suppression read in the wrong units shows
    up as a different count.  The four nuclei are found at every spacing."""
    img, cells = three_cells_scene(sp)
    r = nuclei_in_cells(img, cells, None, sp, radius_um=1.0)
    assert lo <= r.n_centres <= hi
    assert {c: len(v) for c, v in per_cell(r).items()} == {1: 1, 2: 2, 3: 1}


@pytest.mark.parametrize("sp", [(1.0, 0.5, 0.5), (1.0, 1.0, 1.0)])
def test_rescue_border_shell_is_1um_along_coarse_z(sp):
    """Step 6c alone: a cell wall across z, a nucleus whose centre sits
    1.5 um inside it -- outside the 1 um border shell, so it is the cell's
    own centre and the nucleus is found (a 2 um shell would refuse it)."""
    n, (Z, Y, X) = _grid((12, 12, 12), sp)
    cells = (Z >= 3).astype(np.int32)
    img = np.where(cells > 0, 10.0, 2.0)
    img = np.where(np.sqrt((Z - 4.5) ** 2 + (Y - 6) ** 2 + (X - 6) ** 2) <= 1.8,
                   100.0, img).astype(np.float32)
    g = ndi.gaussian_filter(img, ad._sig(0.5, sp)).astype(np.float32)
    g2 = ndi.gaussian_filter(img, ad._sig(1.0, sp)).astype(np.float32)
    pt = np.array([[int(4.5 / sp[0]), int(6 / sp[1]), int(6 / sp[2])]], np.intp)
    run = ad._Run(None, None)
    res = ad._rescue(pt, g, g2, cells, ndi.find_objects(cells), 50.0,
                     np.full(n[0], 10.0, np.float32), cells_list=[1], forbid=None,
                     sp=sp, edge=0.25, minv=ad.MIN_NUCLEUS_UM3 / float(np.prod(sp)),
                     next_id=1, run=run)
    assert len(res) == 1
    (cell, _sl, m), = res.values()
    assert cell == 1 and float(m.sum()) * float(np.prod(sp)) > ad.MIN_NUCLEUS_UM3


def test_peak_window_is_1um_along_fine_z():
    """Step 4 on one nucleus whose brightness rises 20 % per um along z,
    from a fixed centre: its peak is the brightest point within 1 um, so
    its edge -- and its volume -- is the same at 0.25 um z as at 0.5 um.
    A window of 2 voxels (0.5 um at 0.25 um z) draws it about a fifth
    bigger."""
    vols = {}
    for sp in (ISO, (0.25, 0.5, 0.5)):
        n, (Z, Y, X) = _grid((14, 14, 14), sp)
        img = np.full(n, 10.0)
        d = np.sqrt((Z - 7) ** 2 + (Y - 7) ** 2 + (X - 7) ** 2)
        img = np.where(d <= 3.0, 100.0 * (1 + 0.2 * (Z - 7)), img).astype(np.float32)
        g = ndi.gaussian_filter(img, ad._sig(0.5, sp)).astype(np.float32)
        g2 = ndi.gaussian_filter(img, ad._sig(1.0, sp)).astype(np.float32)
        pt = np.array([[int(7 / sp[0]), int(7 / sp[1]), int(7 / sp[2])]], np.intp)
        lab = ad._own_edges(pt, g, g2, np.full(n[0], 10.0, np.float32), g >= 17.5,
                            alpha=0.7, floor_abs=17.5, min_contrast=12.5,
                            hw=ad._halfw(ad._PEAK_WINDOW_UM, sp))
        vols[sp] = float((lab > 0).sum()) * float(np.prod(sp))
    assert vols[(0.25, 0.5, 0.5)] == pytest.approx(vols[ISO], rel=0.10)


@pytest.mark.parametrize("sp", [ISO, (1.0, 0.5, 0.5)])
def test_small_nucleus_against_a_wall_across_z_is_found(sp):
    """The rescue scene turned so the cell wall runs across z: a small
    nucleus inside the cell pressed against a big one outside every cell.
    The small one is the cell's nucleus at 0.5 um and at 1.0 um z alike
    (the big one is counted outside)."""
    img, cells = render((32, 20, 14), sp, {1: (1, 16, 1, 19, 0, 14)},
                        [("ball", (19.5, 10, 7), 3.0, 100),
                         ("ball", (14.5, 10, 7), 2.0, 80)])
    r = nuclei_in_cells(img, cells, None, sp)
    assert r.owner == {1: 1} and r.n_outside_cells == 1
    assert_inside_owner(r, cells)


# --------------------------------------------------------------------------- #
# Add's rules against the kept objects
# --------------------------------------------------------------------------- #
BIG = (14, 20, 48)
ONE_WIDE = {1: (0, 14, 1, 19, 1, 31)}                # x 31-48 um is outside


def test_a_nucleus_split_by_a_kept_object_keeps_its_largest_piece():
    """A thin kept slab (bulk outside every cell, under a quarter of the
    nucleus) cuts the nucleus in two: only its largest piece is added."""
    img, cells = render(BIG, ISO, ONE_WIDE, [("ball", (7, 10, 16), 3.0, 100)])
    target = np.zeros(cells.shape, np.int32)
    target[:, :, 35] = 7                            # x 17.5-18 um
    target[:, :, 68:] = 7                           # the bulk
    free = nuclei_in_cells(img, cells, None, ISO)
    assert ((free.labels > 0) & (target > 0)).sum() < 0.25 * (free.labels > 0).sum()
    assert (free.labels[:, :, 36:] > 0).any()       # the nucleus reaches past it
    r = nuclei_in_cells(img, cells, target, ISO)
    assert r.owner == {1: 1}
    assert ndi.label(r.labels == 1)[1] == 1
    assert not (r.labels[:, :, 36:] > 0).any()      # the far piece is gone


def test_guard_runs_again_after_the_kept_objects_trim():
    """A second nucleus over the guard volume, trimmed under it by a kept
    object (15 % of it, so not dropped by the quarter rule), is dropped."""
    img, cells = render(BIG, ISO, ONE_WIDE, [("ball", (7, 10, 9), 3.0, 100),
                                             ("ball", (7, 10, 22), 1.95, 100)])
    free = nuclei_in_cells(img, cells, None, ISO, second_min_um3=50.0)
    assert free.n_nuclei == 2 and free.n_guard_removed == 0
    second = free.labels == 2
    assert float(second.sum()) * 0.125 > 50.0
    zz, yy, xx = np.nonzero(second)
    order = np.argsort(-xx, kind="stable")[:int(round(0.15 * second.sum()))]
    target = np.zeros(cells.shape, np.int32)
    target[zz[order], yy[order], xx[order]] = 7
    target[:, :, 68:] = 7
    off = nuclei_in_cells(img, cells, target, ISO, second_min_um3=None)
    assert off.n_nuclei == 2 and float((off.labels == 2).sum()) * 0.125 < 50.0
    r = nuclei_in_cells(img, cells, target, ISO, second_min_um3=50.0)
    assert r.n_nuclei == 1 and r.two_in_cell == {}
    assert r.n_found == 2 and r.n_all_taken == 1


def test_cell_owning_a_kept_object_gets_nothing():
    img, cells = render(SHAPE, ISO, TWO_CELLS, [
        ("ball", (7, 10, 8), 3.0, 100), ("ball", (7, 10, 24), 3.0, 100)])
    target = np.zeros(cells.shape, np.int32)
    target[2:6, 4:8, 2:6] = 9                       # a kept object in cell 1
    r = nuclei_in_cells(img, cells, target, ISO)
    assert r.n_cell_has_one == 1
    assert set(r.owner.values()) == {2}
    assert not (r.labels[cells == 1] > 0).any()
    assert r.n_found == 2 and r.n_all_taken == 1
    assert r.n_no_nucleus == 0
    assert not r.all_taken


def _kept_covering(share):
    """A kept object covering ``share`` of cell 2's nucleus; most of it lies
    outside every cell, so no cell owns it."""
    img, cells = render(SHAPE, ISO, {1: (0, 14, 1, 19, 0, 16),
                                      2: (0, 14, 1, 10, 16, 32)},
                        [("ball", (7, 10, 8), 3.0, 100),
                         ("ball", (7, 5.5, 24), 3.0, 100)])
    free = nuclei_in_cells(img, cells, None, ISO)
    (nid,) = [i for i, c in free.owner.items() if c == 2]
    nucleus = free.labels == nid
    zz, yy, xx = np.nonzero(nucleus)
    order = np.argsort(-xx, kind="stable")
    k = int(round(share * order.size))
    target = np.zeros(cells.shape, np.int32)
    target[zz[order[:k]], yy[order[:k]], xx[order[:k]]] = 7
    target[:, 22:, 34:] = 7                         # the bulk, outside cells
    return img, cells, target, nucleus


def test_found_nucleus_a_quarter_under_a_kept_object_is_dropped():
    img, cells, target, _nucleus = _kept_covering(0.30)
    r = nuclei_in_cells(img, cells, target, ISO)
    assert r.n_cell_has_one == 0
    assert set(r.owner.values()) == {1}
    assert r.n_no_nucleus == 1                      # cell 2 is left empty
    assert r.n_all_taken == 1


def test_output_only_on_empty_target_and_inside_the_owner():
    img, cells, target, nucleus = _kept_covering(0.10)
    r = nuclei_in_cells(img, cells, target, ISO)
    assert set(r.owner.values()) == {1, 2}
    assert not ((r.labels > 0) & (target > 0)).any()
    (nid,) = [i for i, c in r.owner.items() if c == 2]
    assert (r.labels == nid).sum() < nucleus.sum()  # trimmed, not dropped
    assert_inside_owner(r, cells)


def test_every_nucleus_on_kept_objects_is_all_taken():
    img, cells = render(SHAPE, ISO, TWO_CELLS, [
        ("ball", (7, 10, 8), 3.0, 100), ("ball", (7, 10, 24), 3.0, 100)])
    free = nuclei_in_cells(img, cells, None, ISO)
    r = nuclei_in_cells(img, cells, free.labels, ISO)
    assert r.n_nuclei == 0 and r.n_found == 2
    assert r.all_taken and r.n_all_taken == 2
    assert r.n_cell_has_one == 2 and r.n_no_nucleus == 0
    assert not r.labels.any()


def test_no_nucleus_found_is_not_all_taken():
    img, cells = render((14, 20, 20), ISO, SMALL_ONE, [], outside=10.0)
    r = nuclei_in_cells(img, cells, np.zeros(cells.shape, np.int32), ISO)
    assert r.n_found == 0 and not r.all_taken and r.n_no_nucleus == 1


# --------------------------------------------------------------------------- #
# Stop
# --------------------------------------------------------------------------- #
def test_stop_set_before_the_call_raises():
    img, cells = gap_scene()
    ev = threading.Event()
    ev.set()
    with pytest.raises(ReshapeCancelled):
        nuclei_in_cells(img, cells, None, ISO, cancel_event=ev)


def test_stop_is_polled_between_the_six_filter_passes():
    img, cells = gap_scene()
    stages = []
    polls = {"after_centres": 0}

    def progress(text, i, n):
        stages.append(text)

    def cancel():
        if stages and stages[-1] == DERIVE_STAGE_CENTRES:
            polls["after_centres"] += 1
            return polls["after_centres"] >= 3     # the third pass
        return False

    with pytest.raises(ReshapeCancelled):
        nuclei_in_cells(img, cells, None, ISO, cancel_event=cancel,
                        progress=progress)
    assert stages[-1] == DERIVE_STAGE_CENTRES
    assert DERIVE_STAGE_EDGES not in stages
    assert polls["after_centres"] == 3


def _many_cells():
    """441 small cells, a flat channel: no nucleus, every cell empty."""
    cells = np.zeros((4, 63, 63), np.int32)
    k = 0
    for y in range(0, 63, 3):
        for x in range(0, 63, 3):
            k += 1
            cells[:, y:y + 3, x:x + 3] = k
    return np.full(cells.shape, 10.0, np.float32), cells


def test_per_cell_loops_poll_and_report_every_200_cells():
    img, cells = _many_cells()
    texts = []
    r = nuclei_in_cells(img, cells, None, ISO,
                        progress=lambda t, i, n: texts.append(t))
    assert r.n_cells == 441 and r.n_no_nucleus == 441
    cell_texts = [t for t in texts if " of 441" in t]
    assert cell_texts == [
        "Empty cell {} of 441 · {}".format(i, DERIVE_STAGE_RESCUE)
        for i in range(CANCEL_EVERY, 442, CANCEL_EVERY)]


def test_stop_inside_a_per_cell_loop_is_honoured_at_the_next_200():
    img, cells = _many_cells()
    texts = []
    state = {"stop": False}

    def progress(text, i, n):
        texts.append(text)
        if text.startswith("Empty cell 200 of"):
            state["stop"] = True

    with pytest.raises(ReshapeCancelled):
        nuclei_in_cells(img, cells, None, ISO, progress=progress,
                        cancel_event=lambda: state["stop"])
    assert texts[-1].startswith("Empty cell 200 of 441")


class _Texts:
    """A ``_Run`` that records its progress texts."""

    def __init__(self, stage):
        self.texts = []
        self.run = ad._Run(None, lambda t, i, n: self.texts.append(t))
        self.run._stage = stage


def test_tidy_loop_counts_nuclei():
    lab = np.zeros((1, 40, 40), np.int32)
    lab[0, ::2, ::2] = np.arange(1, 401).reshape(20, 20)
    rec = _Texts(DERIVE_STAGE_TIDY)
    out = ad._tidy(lab, None, 1, rec.run)
    assert int(out.max()) == 400
    assert rec.texts == [DERIVE_LOOP_NUCLEI.format(i=i, k=400, stage=DERIVE_STAGE_TIDY)
                         for i in (200, 400)]
    assert rec.texts[0] == "Nucleus 200 of 400 · " + DERIVE_STAGE_TIDY


def test_two_in_a_cell_loop_counts_cells_with_a_nucleus_and_keeps_int32():
    lab = np.zeros((1, 40, 40), np.int32)
    lab[0, ::2, ::2] = np.arange(1, 401).reshape(20, 20)
    cells = lab.copy()
    rec = _Texts(DERIVE_STAGE_TWO)
    new, _own = ad._two_in_a_cell(lab, {x: x for x in range(1, 401)}, cells,
                                  np.ones(lab.shape, np.float32), ratio=0.7,
                                  pad=(1, 1, 1), run=rec.run)
    assert new.dtype == np.int32 and np.array_equal(new, lab)
    assert rec.texts == [DERIVE_LOOP_CELLS_WITH.format(i=i, k=400, stage=DERIVE_STAGE_TWO)
                         for i in (200, 400)]
    assert rec.texts[0] == "Cell 200 of 400 with a nucleus · " + DERIVE_STAGE_TWO
    assert DERIVE_LOOP_EMPTY_CELLS.startswith("Empty cell ")


def test_stop_is_polled_between_the_two_compactions(monkeypatch):
    img, cells = gap_scene()
    calls = []
    real = ad._compact

    def counted(a):
        calls.append(1)
        return real(a)

    monkeypatch.setattr(ad, "_compact", counted)
    texts, polls = [], []

    def cancel():
        polls.append(len(calls))
        return len(calls) == 1                     # after the cells, before the target

    with pytest.raises(ReshapeCancelled):
        nuclei_in_cells(img, cells, np.zeros(cells.shape, np.int32), ISO,
                        cancel_event=cancel, progress=lambda t, i, n: texts.append(t))
    assert texts == [DERIVE_STAGE_READ]
    assert polls == [0, 1] and len(calls) == 1


# --------------------------------------------------------------------------- #
# ids
# --------------------------------------------------------------------------- #
def test_ids_compact_ordered_by_cell_then_size():
    img, cells = three_cells_scene(cell_ids=(30, 10, 20))
    r = nuclei_in_cells(img, cells, None, ISO)
    assert_compact(r)
    owners = [r.owner[i] for i in sorted(r.owner)]
    assert owners == sorted(owners) == [10, 10, 20, 30]
    a, b = r.two_in_cell[10]
    assert (r.labels == a).sum() >= (r.labels == b).sum()


def test_huge_ids_in_cells_and_target():
    img, cells = render(SHAPE, ISO, TWO_CELLS, [
        ("ball", (7, 10, 8), 3.0, 100), ("ball", (7, 10, 24), 3.0, 100)])
    small = nuclei_in_cells(img, cells, None, ISO)
    huge = np.where(cells == 2, 2 ** 31 - 1, np.where(cells == 1, 5, 0)).astype(np.int64)
    r = nuclei_in_cells(img, huge, None, ISO)
    assert np.array_equal(r.labels, small.labels)
    assert sorted(r.owner.values()) == [5, 2 ** 31 - 1]
    target = np.zeros(cells.shape, np.int64)
    target[2:6, 4:8, 40:44] = 2 ** 31 - 1           # kept, in the huge cell
    rk = nuclei_in_cells(img, huge, target, ISO)
    assert set(rk.owner.values()) == {5} and rk.n_cell_has_one == 1


def test_uint32_cells_above_the_int32_range():
    img, cells = render(SHAPE, ISO, TWO_CELLS, [
        ("ball", (7, 10, 8), 3.0, 100), ("ball", (7, 10, 24), 3.0, 100)])
    small = nuclei_in_cells(img, cells, None, ISO)
    big = np.where(cells == 2, 4_000_000_000, cells).astype(np.uint32)
    r = nuclei_in_cells(img, big, big * 0, ISO)
    assert np.array_equal(r.labels, small.labels)
    assert sorted(r.owner.values()) == [1, 4_000_000_000]


@pytest.mark.parametrize("where", ["cells", "target"])
def test_ids_of_2_63_or_more_are_refused(where):
    img, cells = render(SHAPE, ISO, TWO_CELLS, [("ball", (7, 10, 8), 3.0, 100)])
    big = np.where(cells == 2, np.uint64(2 ** 63 + 5), cells.astype(np.uint64))
    args = (big, None) if where == "cells" else (cells, big)
    with pytest.raises(ValueError, match=r"2\*\*63"):
        nuclei_in_cells(img, *args, ISO)
    ok = np.where(cells == 2, np.uint64(2 ** 63 - 1), cells.astype(np.uint64))
    r = nuclei_in_cells(img, ok, None, ISO)
    assert set(r.owner.values()) <= {1, 2 ** 63 - 1}


def test_compact_paths_agree():
    lab = np.array([[[0, 7, 7, 3]]], np.int64)
    a, ida = ad._compact(lab)
    b, idb = ad._compact(lab + np.where(lab > 0, 2 ** 40, 0))
    assert np.array_equal(a, b) and a.tolist() == [[[0, 2, 2, 1]]]
    assert ida.tolist() == [3, 7]


# --------------------------------------------------------------------------- #
# inputs and levels
# --------------------------------------------------------------------------- #
def test_inputs_are_not_modified_and_channel_dtype_is_free():
    img, cells = gap_scene()
    img16 = np.round(img * 100).astype(np.uint16)
    keep_img, keep_cells = img16.copy(), cells.copy()
    target = np.zeros(cells.shape, np.int32)
    r = nuclei_in_cells(img16, cells, target, ISO)
    assert r.n_nuclei == 2
    assert np.array_equal(img16, keep_img) and np.array_equal(cells, keep_cells)
    assert not target.any()


def test_patch_levels_given_equal_measured():
    img, cells = gap_scene()
    lv = patch_levels(img, cells, ISO)
    a = nuclei_in_cells(img, cells, None, ISO)
    b = nuclei_in_cells(img, cells, None, ISO, levels=lv)
    assert a.otsu == pytest.approx(lv.otsu)
    assert np.array_equal(a.labels, b.labels)
    sub = lv.crop_z(2, 10)
    assert sub.background_z.shape == (8,)
    with pytest.raises(ValueError, match="crop_z"):
        nuclei_in_cells(img, cells, None, ISO, levels=sub)


def test_patch_levels_validates_like_the_recipe():
    img, cells = gap_scene()
    with pytest.raises(TypeError, match="spacing_um"):
        patch_levels(img, cells, None)
    with pytest.raises(TypeError, match="spacing_um"):
        patch_levels(img, cells, "0.5")
    with pytest.raises(TypeError, match="numeric"):
        patch_levels(img > 50, cells, ISO)
    bad = img.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        patch_levels(bad, cells, ISO)
    with pytest.raises(ValueError, match="3D"):
        patch_levels(img[0], cells[0], ISO)


@pytest.mark.parametrize("make", ["zeros", "negative", "levels"])
def test_a_threshold_of_zero_or_less_is_refused_in_plain_words(make):
    img, cells = gap_scene()
    kw = {}
    if make == "zeros":
        img = np.zeros_like(img)
    elif make == "negative":
        img = img - 1000.0
    else:
        kw["levels"] = PatchLevels(0.0, np.zeros(img.shape[0], np.float32))
    with pytest.raises(ValueError, match="no bright level above zero"):
        nuclei_in_cells(img, cells, None, ISO, **kw)


def test_no_cells_returns_an_empty_result():
    img, cells = gap_scene()
    r = nuclei_in_cells(img, np.zeros_like(cells), None, ISO)
    assert isinstance(r, DeriveResult)
    assert r.n_cells == 0 and r.n_nuclei == 0 and not r.labels.any()


@pytest.mark.parametrize("kwargs, err, match", [
    (dict(spacing_um=(0.5, 0, 0.5)), ValueError, "spacing_um"),
    (dict(spacing_um=(0.5, 0.5)), ValueError, "spacing_um"),
    (dict(spacing_um=(0.5, -1, 0.5)), ValueError, "spacing_um"),
    (dict(radius_um=0.5), ValueError, "radius_um"),
    (dict(radius_um=12.0), ValueError, "radius_um"),
    (dict(edge=0.95), ValueError, "edge"),
    (dict(gap=0.2), ValueError, "gap"),
    (dict(min_um3=-1), ValueError, "min_um3"),
    (dict(second_min_um3=-5), ValueError, "second_min_um3"),
    (dict(edge="x"), TypeError, "edge must be a number"),
    (dict(gap=None), TypeError, "gap must be a number"),
    (dict(radius_um=float("nan")), ValueError, "radius_um"),
    (dict(min_um3="a"), TypeError, "min_um3 must be a number"),
    (dict(second_min_um3="a"), TypeError, "second_min_um3 must be a number"),
    (dict(spacing_um="0.5"), TypeError, "spacing_um"),
    (dict(spacing_um=None), TypeError, "spacing_um"),
    (dict(spacing_um=("a", 1, 1)), TypeError, "spacing_um"),
])
def test_boundary_validation_of_numbers(kwargs, err, match):
    img, cells = render((4, 4, 4), ISO, {1: (0, 2, 0, 2, 0, 2)}, [])
    kw = dict(spacing_um=ISO)
    kw.update(kwargs)
    sp = kw.pop("spacing_um")
    with pytest.raises(err, match=match):
        nuclei_in_cells(img, cells, None, sp, **kw)


def test_boundary_validation_of_arrays():
    img, cells = render((4, 4, 4), ISO, {1: (0, 2, 0, 2, 0, 2)}, [])
    with pytest.raises(ValueError, match="shape"):
        nuclei_in_cells(img, cells[:-1], None, ISO)
    with pytest.raises(ValueError, match="shape"):
        nuclei_in_cells(img, cells, cells[:, :-1], ISO)
    with pytest.raises(TypeError, match="integer"):
        nuclei_in_cells(img, cells.astype(np.float32), None, ISO)
    with pytest.raises(ValueError, match="negative"):
        nuclei_in_cells(img, cells - 1, None, ISO)
    with pytest.raises(ValueError, match="3D"):
        nuclei_in_cells(img[0], cells[0], None, ISO)
    bad = img.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        nuclei_in_cells(bad, cells, None, ISO)


# --------------------------------------------------------------------------- #
# parity with the reference run on a real patch (local data only)
# --------------------------------------------------------------------------- #
_DRIVERS = Path(os.environ.get(
    "EPICYCLE_DERIVE_DRIVERS", Path.home() / "Desktop" / "EpiCycle-drivers"))
_WS = _DRIVERS / "derive_d0" / "ws"
_REF = _DRIVERS / "derive_in_add" / "b0" / "out" / "s1_Wound_patch_1.npz"


@pytest.mark.skipif(not (_WS.is_dir() and _REF.is_file()),
                    reason="the Wound patch and its reference arrays are local only")
def test_parity_with_the_reference_on_wound_patch_1():
    tifffile = pytest.importorskip("tifffile")
    img = tifffile.imread(str(_WS / "patches" / "Wound_patch_1.tiff"))
    cells = tifffile.imread(str(_WS / "default" / "Wound_patch_1" / "labels_3d.tif"))
    ref = np.load(str(_REF))["clip"]
    r = nuclei_in_cells(img[0].astype(np.float32), cells.astype(np.int32), None,
                        ISO, second_min_um3=None)
    assert (r.n_no_nucleus, r.n_one_nucleus, r.n_two_in_cell) == (62, 769, 13)
    assert r.n_cut_at_edge == 59
    # the reference refilled 52 voxels outside their owner (holes in 7
    # nuclei, after the cut); the tidy now stays inside the owner, which also
    # leaves 6 voxels of 3 of those nuclei as loose pieces, dropped
    assert not ((r.labels > 0) & (ref == 0)).any()
    lost = (ref > 0) & (r.labels == 0)
    ref_owner = dict(np.load(str(_REF))["clip_owner"].tolist())
    lut = np.zeros(int(ref.max()) + 1, np.int64)
    for k, v in ref_owner.items():
        lut[k] = v
    outside = (ref > 0) & (cells != lut[ref])
    assert int(outside.sum()) == 52 and len(np.unique(ref[outside])) == 7
    assert int(lost.sum()) == 58 and not (outside & ~lost).any()
    assert_inside_owner(r, cells)
    pairs = np.unique(r.labels[ref > 0].astype(np.int64) * (int(ref.max()) + 1)
                      + ref[ref > 0])
    pairs = pairs[pairs >= int(ref.max()) + 1]      # drop (0, ref id) pairs
    assert pairs.size == r.n_nuclei == int(len(np.unique(ref)) - 1)

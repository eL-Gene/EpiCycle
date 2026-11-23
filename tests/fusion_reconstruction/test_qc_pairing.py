"""The pairing check between two runs of one patch.

For every object of the active run, which objects of the other run it
shares voxels with, how many, and how well.  Two objects are linked when
their shared voxels are at least the link floor of the smaller one (the
overlap coefficient), and an object below its run's size floor never links.

Every case is a small synthetic 3D volume built here, checked with cells
active and again with nuclei active.  The rule itself is numpy only; it is
imported through the package (whose ``__init__`` loads the Qt controller),
but no QApplication is needed.
"""

from __future__ import annotations

import json
import math
import threading
import tracemalloc

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction import qc_pairing as QP
from packages.epicycle_annotator.reshape import seeds as S

SHAPE = (10, 20, 40)
FLOOR = 20          # both runs' size floor in these cases, in voxels
LINK = 0.15


@pytest.fixture(params=[3, 16], ids=["slab3", "slab16"])
def slab(request):
    """z-planes per slab: 3 makes every object below cross slab edges."""
    return request.param


def _vol(shape=SHAPE, dtype=np.uint16):
    return np.zeros(shape, dtype)


def _two_cells(dtype=np.uint16):
    """Cell 1 fills x 0-19, cell 2 fills x 20-39, over the whole z and y."""
    c = _vol(dtype=dtype)
    c[:, :, :20] = 1
    c[:, :, 20:] = 2
    return c


def _pair(active, other, *, link=LINK, fa=FLOOR, fo=FLOOR,
          slab=QP.SLAB_PLANES):
    return QP.pair_runs(active, other, link_floor=link,
                        floor_active=fa, floor_other=fo, slab=slab)


def _row(t, label):
    i = int(np.flatnonzero(t.label == label)[0])
    return {
        "count": int(t.match_count[i]),
        "id": int(t.match_id[i]),
        "overlap": float(t.match_overlap[i]),
        "shared": int(t.match_shared[i]),
        "minor": float(t.match_minor_ratio[i]),
        "too_small": bool(t.too_small[i]),
    }


def _status(t, label):
    r = _row(t, label)
    return QP.match_status(r["count"], r["shared"], r["too_small"])


# --------------------------------------------------------------------------- #
# the eight cases, both directions, two slab sizes
# --------------------------------------------------------------------------- #
def test_merged_cell_holding_two_nuclei(slab):
    cells = _vol()
    cells[:, :, :] = 1                       # one cell over the whole patch
    nuc = _vol()
    nuc[2:8, 2:8, 2:8] = 5                   # 216 voxels each, wholly inside
    nuc[2:8, 2:8, 30:36] = 6
    ca = _pair(cells, nuc, slab=slab)
    r = _row(ca, 1)
    assert r["count"] == 2
    assert r["id"] == 5                      # tie on shared and O -> lower id
    assert r["overlap"] == pytest.approx(1.0)
    assert r["minor"] == pytest.approx(1.0)
    assert r["shared"] == 1
    assert _status(ca, 1) == QP.STATUS_SEVERAL

    na = _pair(nuc, cells, slab=slab)
    for n in (5, 6):
        r = _row(na, n)
        assert r["count"] == 1 and r["id"] == 1
        assert r["overlap"] == pytest.approx(1.0)
        assert r["shared"] == 2              # my cell is also another's
        assert math.isnan(r["minor"])
        assert _status(na, n) == QP.STATUS_SHARED


def _straddle():
    cells = _two_cells()
    nuc = _vol()
    nuc[2:7, 2:7, 14:24] = 7                 # 250 voxels: 150 in cell 1, 100 in 2
    return cells, nuc


def test_straddling_nucleus_60_40(slab):
    cells, nuc = _straddle()
    na = _pair(nuc, cells, slab=slab)
    r = _row(na, 7)
    assert r["count"] == 2
    assert r["id"] == 1
    assert r["overlap"] == pytest.approx(0.6)
    assert r["minor"] == pytest.approx(100 / 150)
    assert r["shared"] == 1
    assert _status(na, 7) == QP.STATUS_SEVERAL

    ca = _pair(cells, nuc, slab=slab)
    r1, r2 = _row(ca, 1), _row(ca, 2)
    assert (r1["count"], r1["id"], r1["shared"]) == (1, 7, 2)
    assert r1["overlap"] == pytest.approx(0.6)
    assert (r2["count"], r2["id"], r2["shared"]) == (1, 7, 2)
    assert r2["overlap"] == pytest.approx(0.4)
    assert _status(ca, 1) == QP.STATUS_SHARED
    assert _status(ca, 2) == QP.STATUS_SHARED


def test_minor_ratio_reads_the_straddle_from_the_nucleus_side():
    cells, nuc = _straddle()
    assert _row(_pair(nuc, cells), 7)["minor"] == pytest.approx(0.6667,
                                                                abs=1e-3)


def test_grazing_nucleus_90_10(slab):
    cells = _two_cells()
    nuc = _vol()
    nuc[2:7, 2:7, 11:21] = 8                 # 9 columns in cell 1, 1 in cell 2
    na = _pair(nuc, cells, slab=slab)
    r = _row(na, 8)
    assert (r["count"], r["id"], r["shared"]) == (1, 1, 1)
    assert r["overlap"] == pytest.approx(0.9)
    # the 10 % in cell 2 (O 0.1) is a contact, not a link: no minor ratio
    assert math.isnan(r["minor"])
    assert _status(na, 8) == QP.STATUS_ONE

    ca = _pair(cells, nuc, slab=slab)
    assert _row(ca, 1)["count"] == 1
    assert _row(ca, 1)["overlap"] == pytest.approx(0.9)
    assert _status(ca, 1) == QP.STATUS_ONE
    r2 = _row(ca, 2)
    assert (r2["count"], r2["id"]) == (0, 0)
    assert _status(ca, 2) == QP.STATUS_NONE


def _fragment():
    cells = _two_cells()
    nuc = _vol()
    nuc[2:7, 2:7, 1:21] = 5                  # 500 voxels: 475 in cell 1 (O 0.95)
    nuc[2:5, 10:14, 5:12] = 6                # 84-voxel fragment, wholly inside 1
    return cells, nuc


def test_nuclear_fragment_beside_its_real_nucleus(slab):
    cells, nuc = _fragment()
    ca = _pair(cells, nuc, slab=slab)
    r = _row(ca, 1)
    assert r["count"] == 2
    assert r["id"] == 5                      # the O = 1 fragment loses
    assert r["overlap"] == pytest.approx(0.95)
    assert r["minor"] == pytest.approx(84 / 475)
    assert r["shared"] == 1
    # cell 2 holds 25 voxels of nucleus 5 (O 0.05): no link
    assert _row(ca, 2)["count"] == 0

    na = _pair(nuc, cells, slab=slab)
    r5, r6 = _row(na, 5), _row(na, 6)
    assert (r5["count"], r5["id"], r5["shared"]) == (1, 1, 2)
    assert r5["overlap"] == pytest.approx(0.95)
    assert (r6["count"], r6["id"], r6["shared"]) == (1, 1, 2)
    assert r6["overlap"] == pytest.approx(1.0)
    assert _status(na, 6) == QP.STATUS_SHARED


def test_cell_with_no_nucleus(slab):
    cells = _two_cells()
    cells[:, :, 30:] = 3                     # cell 3, empty of nuclei
    nuc = _vol()
    nuc[2:7, 2:7, 2:10] = 5
    ca = _pair(cells, nuc, slab=slab)
    for lab in (2, 3):
        r = _row(ca, lab)
        assert (r["count"], r["id"], r["shared"]) == (0, 0, 0)
        assert r["overlap"] == 0.0
        assert math.isnan(r["minor"])
        assert _status(ca, lab) == QP.STATUS_NONE
    na = _pair(nuc, cells, slab=slab)
    assert _row(na, 5)["count"] == 1
    assert _status(na, 5) == QP.STATUS_ONE


def test_debris_below_the_floor_never_links_either_side(slab):
    cells = _two_cells()
    cells[0:2, 0:2, 0:2] = 4                 # 8-voxel debris cell inside cell 1's box
    nuc = _vol()
    nuc[0:5, 0:5, 0:10] = 5                  # covers the debris cell and cell 1
    nuc[5:7, 10:12, 25:27] = 9               # 8-voxel speck inside cell 2
    ca = _pair(cells, nuc, slab=slab)
    # the debris cell is too small: never links, and never raises the
    # match_shared of the nucleus's real cell
    r4 = _row(ca, 4)
    assert r4["too_small"] and r4["count"] == 0 and r4["id"] == 0
    assert _status(ca, 4) == QP.STATUS_TOO_SMALL
    r1 = _row(ca, 1)
    assert (r1["count"], r1["id"], r1["shared"]) == (1, 5, 1)
    # the speck nucleus is below the other run's floor: cell 2 has none
    assert _row(ca, 2)["count"] == 0
    assert _status(ca, 2) == QP.STATUS_NONE

    na = _pair(nuc, cells, slab=slab)
    r9 = _row(na, 9)
    assert r9["too_small"] and r9["count"] == 0 and r9["id"] == 0
    assert math.isnan(r9["overlap"])
    assert _status(na, 9) == QP.STATUS_TOO_SMALL
    r5 = _row(na, 5)
    assert (r5["count"], r5["id"], r5["shared"]) == (1, 1, 1)   # debris cell 4 not linked
    # the debris cell lies wholly inside nucleus 5 (O = 1) but is below its
    # floor: an unlinked contact, so no minor ratio
    assert math.isnan(r5["minor"])


def test_active_debris_never_raises_a_neighbours_shared():
    # nuclei active: a speck nucleus in the same cell as a real nucleus
    cells = _two_cells()
    nuc = _vol()
    nuc[2:7, 2:7, 2:10] = 5
    nuc[8:10, 15:17, 15:17] = 9              # 8 voxels, cell 1
    na = _pair(nuc, cells)
    assert _row(na, 9)["too_small"]
    assert _row(na, 5)["shared"] == 1
    assert _status(na, 5) == QP.STATUS_ONE


def test_coarse_cells_one_cell_over_several_nuclei(slab):
    cells = _vol()
    cells[:] = 1
    nuc = _vol()
    nuc[1:4, 1:4, 1:4] = 11                  # 27
    nuc[1:6, 8:13, 8:13] = 12                # 125 (the largest)
    nuc[1:5, 1:5, 20:24] = 13                # 64
    nuc[6:9, 14:18, 30:34] = 14              # 48
    ca = _pair(cells, nuc, slab=slab)
    r = _row(ca, 1)
    assert r["count"] == 4
    assert r["id"] == 12
    assert r["minor"] == pytest.approx(64 / 125)
    na = _pair(nuc, cells, slab=slab)
    for n in (11, 12, 13, 14):
        r = _row(na, n)
        assert (r["count"], r["id"], r["shared"]) == (1, 1, 4)
        assert _status(na, n) == QP.STATUS_SHARED


def test_two_same_channel_models_nested_object(slab):
    a = _vol()
    a[0:10, 0:20, 0:20] = 1                  # 4000
    b = _vol()
    b[2:8, 2:18, 2:18] = 2                   # 1536, nested inside a's 1
    for act, oth, lab, match in ((a, b, 1, 2), (b, a, 2, 1)):
        t = _pair(act, oth, slab=slab)
        r = _row(t, lab)
        assert (r["count"], r["id"], r["shared"]) == (1, match, 1)
        assert r["overlap"] == pytest.approx(1.0)
        assert _status(t, lab) == QP.STATUS_ONE


# --------------------------------------------------------------------------- #
# the rule's properties
# --------------------------------------------------------------------------- #
def test_overlap_is_the_same_number_from_either_side():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 6, size=(12, 16, 16)).astype(np.uint16)
    b = rng.integers(0, 9, size=(12, 16, 16)).astype(np.int32)
    ab = QP.count_overlaps(a, b, slab=5)
    ba = QP.count_overlaps(b, a, slab=5)
    o_ab = {(int(x), int(y)): v for x, y, v in
            zip(ab.pair_active, ab.pair_other, ab.overlap())}
    o_ba = {(int(y), int(x)): v for x, y, v in
            zip(ba.pair_active, ba.pair_other, ba.overlap())}
    assert o_ab.keys() == o_ba.keys() and len(o_ab) > 0
    for k, v in o_ab.items():
        assert v == pytest.approx(o_ba[k])


def test_link_at_exactly_the_floor():
    # 10 shared voxels, the smaller object has 100: O = 0.1 exactly
    a = _vol(shape=(1, 10, 30))
    a[0, :, :10] = 1                         # 100
    b = _vol(shape=(1, 10, 30))
    b[0, :, 9:30] = 2                        # 210; one shared column
    t = QP.pair_runs(a, b, link_floor=0.1, floor_active=1, floor_other=1)
    assert _row(t, 1)["overlap"] == pytest.approx(0.1)
    assert _row(t, 1)["count"] == 1
    t = QP.pair_runs(a, b, link_floor=0.11, floor_active=1, floor_other=1)
    assert _row(t, 1)["count"] == 0


def test_equal_shared_voxels_the_smaller_match_wins():
    # cell 1 shares 200 voxels with nucleus 5 (400 voxels, O 0.5) and 200
    # with nucleus 6 (200 voxels, wholly inside, O 1.0): the higher O wins
    # over the lower id
    cells = _two_cells()
    nuc = _vol()
    nuc[0:5, 0:4, 10:30] = 5                 # 400: 200 in cell 1, 200 in 2
    nuc[0:5, 10:14, 0:10] = 6                # 200, wholly inside cell 1
    t = _pair(cells, nuc)
    r = _row(t, 1)
    assert r["count"] == 2
    assert r["id"] == 6
    assert r["overlap"] == pytest.approx(1.0)
    assert r["minor"] == pytest.approx(1.0)


def test_an_object_of_exactly_the_floor_links():
    a = _vol(shape=(4, 5, 5))
    a[0, 0:4, 0:5] = 1                       # 20 voxels
    b = _vol(shape=(4, 5, 5))
    b[0, 0:4, 0:5] = 3                       # the same 20 voxels
    t = QP.pair_runs(a, b, link_floor=LINK, floor_active=20, floor_other=20)
    r = _row(t, 1)
    assert not r["too_small"] and (r["count"], r["id"]) == (1, 3)
    t = QP.pair_runs(a, b, link_floor=LINK, floor_active=21, floor_other=20)
    assert _row(t, 1)["too_small"]
    t = QP.pair_runs(a, b, link_floor=LINK, floor_active=20, floor_other=21)
    r = _row(t, 1)
    assert not r["too_small"] and r["count"] == 0


@pytest.mark.parametrize("slab", [1, 3, 16, 100])
def test_slab_counter_equals_a_whole_volume_unique(slab):
    rng = np.random.default_rng(7)
    a = rng.integers(0, 40, size=(23, 31, 29)).astype(np.uint32)
    b = rng.integers(-5, 30, size=(23, 31, 29)).astype(np.int16)
    a[a < 8] = 0
    got = QP.count_overlaps(a, b, slab=slab)
    m = (a != 0) & (b != 0)
    pairs = np.stack([a[m].astype(np.int64), b[m].astype(np.int64)], axis=1)
    want, counts = np.unique(pairs, axis=0, return_counts=True)
    got_pairs = np.stack([got.pair_active.astype(np.int64),
                          got.pair_other.astype(np.int64)], axis=1)
    order = np.lexsort((got_pairs[:, 1], got_pairs[:, 0]))
    np.testing.assert_array_equal(got_pairs[order], want)
    np.testing.assert_array_equal(got.pair_shared[order], counts)
    ua, ca = np.unique(a[a != 0], return_counts=True)
    np.testing.assert_array_equal(got.active_ids, ua)
    np.testing.assert_array_equal(got.active_voxels, ca)
    ub, cb = np.unique(b[b != 0], return_counts=True)
    np.testing.assert_array_equal(got.other_ids, ub)
    np.testing.assert_array_equal(got.other_voxels, cb)


def test_negative_active_ids_take_the_general_path():
    a = np.zeros((4, 6, 6), np.int64)
    a[:2] = 2 ** 32 - 3                      # an int32 cast reads -3
    a[2:] = 3
    b = np.zeros((4, 6, 6), np.int32)
    b[1:3] = 9
    got = QP.count_overlaps(a, b, slab=1)
    d = {(int(x), int(y)): int(c) for x, y, c in
         zip(got.pair_active, got.pair_other, got.pair_shared)}
    assert d == {(-3, 9): 36, (3, 9): 36}
    assert got.active_ids.dtype == np.int32


def test_label_dtype_is_the_cut_view_and_floors_are_plain_numbers():
    a = _two_cells(dtype=np.uint32)
    nuc = _vol(dtype=np.int16)
    nuc[2:5, 2:5, 2:5] = 3
    t = QP.pair_runs(a, nuc, link_floor=np.float64(LINK),
                     floor_active=np.int64(FLOOR), floor_other=20.5)
    assert t.label.dtype == np.int32         # uint32 is not int32-safe
    assert t.too_small.dtype == bool
    assert len(t) == 2
    assert (t.link_floor, t.floor_active, t.floor_other) == (LINK, FLOOR, 20.5)
    assert type(t.link_floor) is float
    assert type(t.floor_active) is int
    assert type(t.floor_other) is float
    json.dumps([t.link_floor, t.floor_active, t.floor_other])
    assert _pair(_two_cells(), nuc).label.dtype == np.uint16


def test_wide_active_ids_wrap_the_way_an_int32_cast_does():
    labels = _vol(dtype=np.uint32)
    labels[:, :, :10] = 2 ** 31 + 4
    labels[:, :, 10:20] = 2 ** 32 - 1
    labels[:, :, 20:30] = 7
    t = _pair(labels, _vol())
    cast = labels.astype(np.int32)
    want = np.unique(cast[cast != 0])
    np.testing.assert_array_equal(t.label, want)
    assert t.label.dtype == np.int32
    assert set(t.label.tolist()) == {-(2 ** 31) + 4, -1, 7}


def test_wide_active_object_across_slabs_through_pair_runs():
    cells = _vol(dtype=np.uint32)
    big = 2 ** 31 + 9
    cells[:, :, :20] = big                   # 4000 voxels over every slab
    cells[:, :, 20:] = 2
    nuc = _vol(dtype=np.uint16)
    nuc[1:9, 2:7, 2:10] = 5                  # 320 voxels, crosses z 3, 6
    nuc[1:9, 2:7, 22:30] = 6
    for slab in (1, 3, 16):
        t = _pair(cells, nuc, slab=slab)
        wrapped = int(np.array([big], np.uint32).astype(np.int32)[0])
        i = int(np.flatnonzero(t.label == wrapped)[0])
        assert int(t.voxels[i]) == 4000
        assert (int(t.match_count[i]), int(t.match_id[i])) == (1, 5)
        assert float(t.match_overlap[i]) == pytest.approx(1.0)
        r2 = _row(t, 2)
        assert (r2["count"], r2["id"]) == (1, 6)
        na = _pair(nuc, cells, slab=slab)
        assert _row(na, 5)["id"] == wrapped


def test_int16_other_run_ids_with_negative_values_agree_with_the_cut():
    cells = _two_cells()
    guide = _vol(dtype=np.int16)
    guide[2:7, 2:7, 2:10] = -300
    guide[2:7, 2:7, 22:30] = 12
    cut = S.seeds_from_labels(guide)         # the ids the cut works with
    t = _pair(cells, guide)
    assert _row(t, 1)["id"] == -300
    assert _row(t, 2)["id"] == 12
    assert set(np.unique(cut[cut != 0]).tolist()) == {-300, 12}


def test_wide_other_run_ids_wrap_the_way_the_cut_wraps():
    cells = _two_cells()
    guide = _vol(dtype=np.uint32)
    guide[2:7, 2:7, 2:10] = 2 ** 31 + 5      # the cut reads it as int32
    t = _pair(cells, guide)
    want = int(S.seeds_from_labels(guide)[3, 3, 3])
    assert _row(t, 1)["id"] == want and want < 0


def test_shapes_must_agree():
    with pytest.raises(ValueError):
        QP.pair_runs(_vol(), _vol(shape=(10, 20, 39)), link_floor=LINK,
                     floor_active=FLOOR, floor_other=FLOOR)


@pytest.mark.parametrize("bad", ["0.15", True, np.bool_(True), None,
                                 float("nan"), float("inf"), 0.0, 1.5])
def test_link_floor_must_be_a_real_number_in_range(bad):
    with pytest.raises(ValueError):
        QP.pair_runs(_two_cells(), _two_cells(), link_floor=bad,
                     floor_active=FLOOR, floor_other=FLOOR)


@pytest.mark.parametrize("which", ["floor_active", "floor_other"])
@pytest.mark.parametrize("bad", ["20", True, None, float("nan"),
                                 float("inf"), -1])
def test_size_floors_must_be_real_finite_numbers(which, bad):
    kw = {"floor_active": FLOOR, "floor_other": FLOOR, which: bad}
    with pytest.raises(ValueError):
        QP.pair_runs(_two_cells(), _two_cells(), link_floor=LINK, **kw)


def test_empty_volumes_give_an_empty_table():
    t = _pair(_vol(), _vol())
    assert len(t) == 0 and t.label.dtype == np.uint16


def test_cancel_between_slabs():
    ev = threading.Event()
    ev.set()
    with pytest.raises(QP.PairingCancelled):
        QP.pair_runs(_two_cells(), _two_cells(), link_floor=LINK,
                     floor_active=FLOOR, floor_other=FLOOR, cancel_event=ev)


def test_inputs_are_never_mutated():
    cells, nuc = _fragment()
    c0, n0 = cells.copy(), nuc.copy()
    _pair(cells, nuc)
    _pair(nuc, cells)
    np.testing.assert_array_equal(cells, c0)
    np.testing.assert_array_equal(nuc, n0)


def test_no_whole_volume_int64_copy():
    """73 M voxels: the peak stays below half of one int64 copy."""
    shape = (73, 1000, 1000)
    rng = np.random.default_rng(11)
    cells = np.empty(shape, np.uint16)
    # dense cells in 50 x 50 columns, every voxel labelled
    yy = (np.arange(1000) // 50)[:, None]
    xx = (np.arange(1000) // 50)[None, :]
    cells[:] = (yy * 20 + xx + 1).astype(np.uint16)[None, :, :]
    nuc = np.zeros(shape, np.uint16)
    for i in range(400):                     # one nucleus per column, apart
        gy, gx = divmod(i, 20)
        z = int(rng.integers(0, 63))
        y, x = gy * 50 + 10, gx * 50 + 10
        nuc[z:z + 10, y:y + 15, x:x + 15] = i + 1
    budget = cells.nbytes + nuc.nbytes       # half of one int64 copy
    tracemalloc.start()
    try:
        t = QP.pair_runs(cells, nuc, link_floor=LINK, floor_active=FLOOR,
                         floor_other=FLOOR)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(t) == 400
    assert peak < budget, f"peak {peak / 1e6:.0f} MB, budget {budget / 1e6:.0f} MB"


# --------------------------------------------------------------------------- #
# the floors
# --------------------------------------------------------------------------- #
def test_object_volumes_and_border():
    lab = np.zeros((6, 8, 8), np.uint16)
    lab[0, 0, 0] = 1                         # touches faces
    lab[2:4, 2:4, 2:4] = 2                   # interior
    lab[2:4, 2:4, 6:8] = 3                   # touches the last x face
    lab[5, 3:5, 3:5] = 4                     # touches the last z face
    ids, vols, border = QP.object_volumes_and_border(lab, slab=2)
    assert ids.tolist() == [1, 2, 3, 4]
    assert vols.tolist() == [1, 8, 8, 4]
    assert border.tolist() == [True, False, True, True]


def test_run_floor_from_the_interior_median():
    vols = np.array([1000] * 25 + [10] * 5)
    border = np.array([False] * 25 + [True] * 5)
    assert QP.run_floor(vols, border) == 100                # ceil(0.1 x 1000)
    assert QP.run_floor(np.array([205] * 20), np.zeros(20, bool)) == 21
    # no speck term and no lower bound: small objects give a small floor
    assert QP.run_floor(np.array([100] * 30), np.zeros(30, bool)) == 10


def test_run_floor_even_count_median_rounds_up():
    vols = np.array([1000] * 10 + [1001] * 10)              # median 1000.5
    assert QP.run_floor(vols, np.zeros(20, bool)) == 101    # ceil(100.05)


def test_run_floor_with_too_few_interior_objects_falls_back():
    vols = np.array([1000] * 19 + [5000] * 10)
    border = np.array([False] * 19 + [True] * 10)
    assert QP.run_floor(vols, border) == 20
    assert QP.run_floor(vols, border, fallback=7.5) == 8
    assert QP.run_floor(np.array([], int), np.array([], bool)) == 20
    assert QP.run_floor(vols, border, min_count=19) == 100


@pytest.mark.parametrize("name", ["factor", "min_count", "fallback"])
@pytest.mark.parametrize("bad", [True, "20", None, float("nan"),
                                 float("inf"), -1])
def test_run_floor_rejects_bad_settings(name, bad):
    vols = np.array([1000] * 25)
    with pytest.raises(ValueError):
        QP.run_floor(vols, np.zeros(25, bool), **{name: bad})


def test_run_floor_takes_no_speck_value():
    with pytest.raises(TypeError):
        QP.run_floor(np.array([1000] * 25), np.zeros(25, bool), 100)


# --------------------------------------------------------------------------- #
# the status
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("count, shared, too_small, measured, word", [
    (0, 0, True, True, QP.STATUS_TOO_SMALL),
    (3, 2, True, False, QP.STATUS_TOO_SMALL),
    (2, 2, False, False, QP.STATUS_NOT_MEASURED),
    (float("nan"), float("nan"), False, True, QP.STATUS_NOT_MEASURED),
    (float("nan"), float("nan"), float("nan"), True, QP.STATUS_NOT_MEASURED),
    (1, 1, float("nan"), True, QP.STATUS_ONE),
    (0, 0, False, True, QP.STATUS_NONE),
    (2, 1, False, True, QP.STATUS_SEVERAL),
    (2, 3, False, True, QP.STATUS_SEVERAL),
    (1, 2, False, True, QP.STATUS_SHARED),
    (1, 1, False, True, QP.STATUS_ONE),
])
def test_status_precedence(count, shared, too_small, measured, word):
    assert QP.match_status(count, shared, too_small, measured) == word


def _every_status():
    """Cells with every word on the cells side: cell 1 one nucleus, cell 6
    none, cell 2 two nuclei, cells 3 and 4 one shared nucleus, cell 5 an
    8-voxel debris cell; a speck nucleus too."""
    cells = _vol()
    cells[:, :10, 0:10] = 1
    cells[:, 10:, 0:10] = 6
    cells[:, :, 10:20] = 2
    cells[:, :, 20:30] = 3
    cells[:, :, 30:40] = 4
    cells[0:2, 0:2, 36:38] = 5
    nuc = _vol()
    nuc[2:7, 2:7, 2:8] = 11
    nuc[2:7, 2:7, 12:18] = 12
    nuc[2:7, 12:17, 12:18] = 13
    nuc[2:7, 2:7, 26:34] = 14
    nuc[8:10, 15:17, 15:17] = 15
    return cells, nuc


@pytest.mark.parametrize("cells_active", [True, False])
def test_status_from_columns_matches_the_table(cells_active):
    cells, nuc = _every_status()
    t = _pair(cells, nuc) if cells_active else _pair(nuc, cells)
    got = QP.status_from_columns(t.match_count, t.match_shared,
                                 t.match_overlap)
    want = [QP.match_status(t.match_count[i], t.match_shared[i],
                              t.too_small[i]) for i in range(len(t))]
    assert got.tolist() == want
    if cells_active:
        assert dict(zip(t.label.tolist(), want)) == {
            1: QP.STATUS_ONE, 6: QP.STATUS_NONE, 2: QP.STATUS_SEVERAL,
            3: QP.STATUS_SHARED, 4: QP.STATUS_SHARED,
            5: QP.STATUS_TOO_SMALL}
    else:
        assert dict(zip(t.label.tolist(), want)) == {
            11: QP.STATUS_ONE, 12: QP.STATUS_SHARED, 13: QP.STATUS_SHARED,
            14: QP.STATUS_SEVERAL, 15: QP.STATUS_TOO_SMALL}


def test_status_from_columns_reads_a_blanked_row_as_not_measured():
    cells, nuc = _every_status()
    t = _pair(cells, nuc)
    count = t.match_count.astype(np.float64)
    shared = t.match_shared.astype(np.float64)
    overlap = t.match_overlap.copy()
    before = QP.status_from_columns(count, shared, overlap)
    assert before[0] != QP.STATUS_NOT_MEASURED
    count[0] = shared[0] = overlap[0] = np.nan
    words = QP.status_from_columns(count, shared, overlap)
    assert words[0] == QP.STATUS_NOT_MEASURED
    assert words[1:].tolist() == before[1:].tolist()


@pytest.mark.parametrize("count, shared, overlap, word", [
    (0, 0, float("nan"), QP.STATUS_TOO_SMALL),       # below the floor
    (0, 0, 0.0, QP.STATUS_NONE),                     # measured, no link
    (float("nan"), float("nan"), float("nan"), QP.STATUS_NOT_MEASURED),
    (1, 1, 0.9, QP.STATUS_ONE),                      # linked
    (2, 1, 0.6, QP.STATUS_SEVERAL),
    (1, 2, 0.6, QP.STATUS_SHARED),
])
def test_status_from_columns_four_encodings(count, shared, overlap, word):
    got = QP.status_from_columns([count], [shared], [overlap])
    assert got.tolist() == [word]


def test_status_words():
    assert QP.STATUS_WORDS == (
        "no match", "one match", "several matches", "shared match",
        "too small to judge", "not measured")

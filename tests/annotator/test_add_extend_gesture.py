"""Extending an object: several seeds, one object (the engine and the gesture).

A plain click makes a new object, which becomes current; Shift+click adds a
seed to the current object wherever it lands; Esc leaves no object current;
Undo last (Ctrl+Z in a side view) takes back the last seed placed, and when
that was an object's only seed the object goes and nothing is current.

The engine makes ONE marker of every object's seeds before the flood and
the cut, so two seeds on one nucleus give the whole nucleus (no cut between
them, no cap left out), the size floor counts the object, and a seed on a
bright piece its object's earlier seed is not on is refused (the call runs
as if it had not been placed).  Cells grows one cell from several seeds.

The keys reach Add through the side views' own key filters, never napari's
keymap: the curated layer's undo is never run by an Add Ctrl+Z.  Shift, Alt
and Ctrl travel with a click on all three click routes.
"""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication

from napari_epicycle.annotator import curation_3d_widget as CW
from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator import add_segment as AS

from .add_harness import FakeThread
from .add_widget_fixture import qapp  # noqa: F401
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_nuclei_box import (  # noqa: F401
    NUC_A,
    NUC_B,
    _active,
    _finish,
    _fire,
    lv,
    off,
)

_APP = QApplication.instance() or QApplication([])

ISO = (1.0, 1.0, 1.0)
SPLITS = ("off", "single", "always")


# --------------------------------------------------------------------------- #
# synthetic nuclei
# --------------------------------------------------------------------------- #
def _sphere(shape, c, r):
    zz, yy, xx = np.indices(shape)
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r


#: one "nucleus that came out too small": two touching r = 10 spheres whose
#: neck at x = 23 the cut finds -- one click keeps one lobe
SHAPE = (30, 30, 50)
LOBE_1, LOBE_2 = (15, 15, 15), (15, 15, 31)


def _pair():
    return _sphere(SHAPE, LOBE_1, 10) | _sphere(SHAPE, LOBE_2, 10)


def _seg(fg, points, **kw):
    img = fg.astype(np.float64)
    kw.setdefault("radius_um", 1.0)
    kw.setdefault("spacing", ISO)
    kw.setdefault("threshold", 0.5)
    return AS.segment_at_markers(img, np.zeros(fg.shape, np.int32), points,
                                 **kw)


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", SPLITS)
def test_every_point_alone_is_the_call_without_groups(split):
    """One key per point: the same pieces and reasons as no groups at all,
    a dim click included (it keeps its number)."""
    fg = _pair()
    pts = [(15, 15, 7), (2, 2, 2), (15, 15, 40)]          # the 2nd is dim
    plain, why = _seg(fg, pts, split=split)
    got, gwhy, seed_why = _seg(fg, pts, split=split, groups=[70, 71, 72])
    assert np.array_equal(plain, got)
    assert gwhy == why == {2: "dim"}
    assert seed_why == {2: "dim"}


@pytest.mark.parametrize("split", SPLITS)
def test_two_seeds_give_the_whole_nucleus_no_notch_no_cap(split):
    fg = _pair()
    pieces, why, seed_why = _seg(fg, [(15, 15, 7), (15, 15, 40)],
                                 split=split, groups=[5, 5])
    assert why == {} and seed_why == {}
    assert set(np.unique(pieces).tolist()) == {0, 1}
    # every voxel of the bright piece, and nothing else
    assert np.array_equal(pieces == 1, fg)


def test_two_plain_clicks_on_the_same_nucleus_still_cut_it():
    """The difference the gesture makes: two OBJECTS split the piece."""
    fg = _pair()
    pieces, _why = _seg(fg, [(15, 15, 7), (15, 15, 40)], split="always")
    assert (pieces == 1).any() and (pieces == 2).any()
    assert not np.array_equal(pieces == 1, fg)


@pytest.mark.parametrize("split", ("single", "always"))
def test_one_seed_with_the_cut_keeps_one_lobe_and_a_second_seed_adds_the_other(
        split):
    fg = _pair()
    one, _w = _seg(fg, [(15, 15, 7)], split=split)
    assert 0 < int((one == 1).sum()) < int(fg.sum())
    two, _w, _s = _seg(fg, [(15, 15, 7), (15, 15, 40)], split=split,
                       groups=["n", "n"])
    assert np.all(two[one == 1] == 1)                 # the first lobe kept
    assert int((two == 1).sum()) == int(fg.sum())


@pytest.mark.parametrize("split", SPLITS)
def test_two_seeds_on_a_lone_round_nucleus_equal_one(split):
    fg = _sphere((24, 24, 24), (12, 12, 12), 8)
    one, _w = _seg(fg, [(12, 12, 12)], split=split)
    two, _w, _s = _seg(fg, [(12, 12, 12), (12, 12, 17)], split=split,
                       groups=[1, 1])
    assert np.array_equal(one, two)


@pytest.mark.parametrize("split", SPLITS)
def test_a_seed_on_another_piece_is_refused_and_changes_nothing(split):
    fg = _sphere(SHAPE, LOBE_1, 10) | _sphere(SHAPE, (15, 15, 42), 4)
    alone, why0 = _seg(fg, [(15, 15, 15)], split=split)
    pieces, why, seed_why = _seg(fg, [(15, 15, 15), (15, 15, 42)],
                                 split=split, groups=[9, 9])
    assert seed_why == {2: "apart"}
    assert why == why0 == {}
    assert np.array_equal(pieces, alone)
    assert not pieces[15, 15, 42]


def test_the_earliest_seed_that_claims_a_piece_decides_it():
    """A first seed off the stain claims nothing: the group's piece is its
    second seed's, and a third seed on another piece is the refused one."""
    fg = _sphere(SHAPE, LOBE_1, 10) | _sphere(SHAPE, (15, 15, 42), 4)
    pieces, why, seed_why = _seg(
        fg, [(2, 2, 2), (15, 15, 42), (15, 15, 15)], groups=[3, 3, 3])
    assert seed_why == {1: "dim", 3: "apart"}
    assert why == {}
    assert pieces[15, 15, 42] == 1 and not pieces[15, 15, 15]


def test_a_group_none_of_whose_seeds_claims_takes_its_first_seed_s_reason():
    fg = _sphere(SHAPE, LOBE_1, 10)
    existing = np.zeros(SHAPE, np.int32)
    existing[0:6, 0:6, 0:6] = 4                            # a kept object
    _p, why, seed_why = AS.segment_at_markers(
        fg.astype(np.float64), existing, [(2, 2, 40), (2, 2, 2)],
        radius_um=1.0, spacing=ISO, threshold=0.5, groups=[1, 1])
    assert seed_why == {1: "dim", 2: "kept"}
    assert why == {1: "dim"}


def test_groups_are_numbered_in_the_order_they_first_appear():
    fg = _sphere(SHAPE, LOBE_1, 10) | _sphere(SHAPE, (15, 15, 42), 4)
    pieces, why, seed_why = _seg(
        fg, [(15, 15, 15), (15, 15, 42), (15, 15, 9)], groups=[50, 20, 50])
    assert why == {} and seed_why == {}
    assert pieces[15, 15, 15] == 1 and pieces[15, 15, 9] == 1
    assert pieces[15, 15, 42] == 2


@pytest.mark.parametrize("split", SPLITS)
def test_the_size_floor_counts_the_whole_object(split):
    fg = _pair()
    floor = int(0.75 * fg.sum())
    plain, why = _seg(fg, [(15, 15, 7), (15, 15, 40)], split=split,
                      min_size=floor)
    assert not plain.any() and why == {1: "small", 2: "small"}
    pieces, gwhy, _s = _seg(fg, [(15, 15, 7), (15, 15, 40)], split=split,
                            min_size=floor, groups=[1, 1])
    assert gwhy == {}
    assert int((pieces == 1).sum()) == int(fg.sum())


def test_groups_must_match_the_points():
    with pytest.raises(ValueError):
        _seg(_pair(), [(15, 15, 7)], groups=[1, 1])


#: lobes in a row (r = 10, necks between them), the cut keeps one per click
ROW = (15, 15, 12), (15, 15, 30), (15, 15, 48), (15, 15, 66)
ROW_SHAPE = (30, 30, 80)


def _row(n):
    fg = np.zeros(ROW_SHAPE, bool)
    for c in ROW[:n]:
        fg |= _sphere(ROW_SHAPE, c, 10)
    return fg


@pytest.mark.parametrize("split", ("single", "always"))
@pytest.mark.parametrize("peak_um", (None, 3.0))
@pytest.mark.parametrize("n,far", [(3, 2), (4, 3)])
def test_seeds_on_far_lobes_take_every_lobe_between(split, peak_um, n, far):
    """Nothing between two seeds of one object is left out: seeds on the
    first and the last lobe give the whole row, the lobes between too."""
    fg = _row(n)
    one, _w = _seg(fg, [ROW[0]], split=split, peak_spacing_um=peak_um)
    assert int((one == 1).sum()) < int(fg.sum())     # one lobe alone
    pieces, why, seed_why = _seg(fg, [ROW[0], ROW[far]], split=split,
                                 peak_spacing_um=peak_um, groups=[1, 1])
    assert why == {} and seed_why == {}
    assert np.array_equal(pieces == 1, fg)


@pytest.mark.parametrize("split", SPLITS)
def test_a_group_never_takes_another_object_s_lobe(split):
    """Four lobes, another object on lobe 3, one object on lobes 1 and 2:
    lobe 3 stays the other object's."""
    fg = _row(4)
    pieces, _why, seed_why = _seg(fg, [ROW[2], ROW[0], ROW[1]], split=split,
                                  groups=[2, 1, 1])
    assert seed_why == {}
    assert pieces[ROW[2]] == 1
    assert pieces[ROW[0]] == pieces[ROW[1]] == 2
    lobe3 = _sphere(ROW_SHAPE, ROW[2], 10) & ~_sphere(ROW_SHAPE, ROW[1], 10)
    assert not (pieces[lobe3] == 2).any()


@pytest.mark.parametrize("split", SPLITS)
def test_a_seed_cut_off_by_another_object_is_refused_as_never_placed(split):
    """Seeds on lobes 1 and 4 of one object, another object on lobe 2: the
    way to lobe 4 crosses the other object, so the second seed is refused
    and the answer is the one without it."""
    fg = _row(4)
    without, why0, _s = _seg(fg, [ROW[0], ROW[1]], split=split,
                             groups=[1, 2])
    pieces, why, seed_why = _seg(fg, [ROW[0], ROW[3], ROW[1]], split=split,
                                 groups=[1, 1, 2])
    assert seed_why == {2: "apart"}
    assert why == why0
    assert np.array_equal(pieces, without)


# --------------------------------------------------------------------------- #
# the preview: markers sharing a reserved id are one object
# --------------------------------------------------------------------------- #
def _nuclei_markers(fg, points, ids, separate=True):
    return AP.run_add_preview(
        preset="nuclei", roi=tuple(slice(0, n) for n in fg.shape),
        existing_roi=np.zeros(fg.shape, np.int32), next_label=500,
        spacing=ISO,
        params=dict(look_for=AP.LOOK_MARKERS, threshold=0.5, clahe=False,
                    min_size=0, separate=separate, peak_spacing_um=5.0,
                    click_radius_um=1.0, auto_smooth=False, clear_border=False,
                    edge_margin=0),
        image_roi=fg.astype(np.float64), click_points=list(points),
        reserved_ids=list(ids))


@pytest.mark.parametrize("separate", (True, False))
def test_two_markers_with_one_id_are_one_whole_object(separate):
    fg = _pair()
    out = _nuclei_markers(fg, [(15, 15, 7), (15, 15, 40)], [500, 500],
                          separate)
    assert out.ok, out.refusal
    assert out.ids == [500]
    assert np.array_equal(out.result == 500, fg)
    assert out.seed_why == [None, None] and out.apart_ids == []
    assert out.marker_why == {} and out.absorbed_ids == []


def test_a_marker_not_touching_its_object_is_named_and_changes_nothing():
    fg = _sphere(SHAPE, LOBE_1, 10) | _sphere(SHAPE, (15, 15, 42), 4)
    alone = _nuclei_markers(fg, [(15, 15, 15)], [500])
    out = _nuclei_markers(fg, [(15, 15, 15), (15, 15, 42)], [500, 500])
    assert out.ok, out.refusal
    assert out.apart_ids == [500]
    assert out.seed_why == [None, "apart"]
    assert np.array_equal(out.result, alone.result)
    assert "not_connected=[500]" in out.log_line()


def test_unshared_ids_are_never_sent_as_groups(monkeypatch):
    """One id per marker: the engine call is the one it always was."""
    seen = []
    real = AS.segment_at_markers

    def _rec(*a, **kw):
        seen.append(dict(kw))
        return real(*a, **kw)

    monkeypatch.setattr(AS, "segment_at_markers", _rec)
    fg = _pair()
    _nuclei_markers(fg, [(15, 15, 7), (15, 15, 40)], [500, 501])
    _nuclei_markers(fg, [(15, 15, 7), (15, 15, 40)], [500, 500])
    assert "groups" not in seen[0]
    assert seen[1]["groups"] == [500, 500]


def _cell_tissue(two=False):
    """One cell walled on every side, its inside clear of the box faces;
    ``two``: two such cells side by side, open space between them."""
    shape = (20, 30, 30)
    rng = np.random.default_rng(0)
    mem = (np.full(shape, 0.05, np.float32)
           + rng.normal(0, 0.005, shape).astype(np.float32))
    shells = ([(1, 14), (16, 29)] if two else [(5, 25)])
    for x0, x1 in shells:
        mem[3:17, 5:25, x0:x1] = 1.0
        mem[5:15, 7:23, x0 + 2:x1 - 2] = (
            0.05 + rng.normal(0, 0.005, (10, 16, x1 - x0 - 4))
        ).astype(np.float32)
    return mem


def _cells_clicks(points, ids, two=False):
    mem = _cell_tissue(two)
    return AP.run_add_preview(
        preset="cells", roi=tuple(slice(0, n) for n in mem.shape),
        existing_roi=np.zeros(mem.shape, np.int32), next_label=500,
        spacing=ISO,
        params=dict(source=AP.SOURCE_CLICKS, wall_kind="membrane",
                    wall_chain="none", dark_outside=False, core_erode_um=1.0,
                    click_radius_um=1.0, min_size=0, clear_border=True,
                    edge_margin=3, auto_smooth=False, smooth_sigma=1.5),
        wall_roi=mem, dark_rois={"membrane": mem}, click_points=list(points),
        reserved_ids=list(ids))


def test_a_cells_two_seed_union_equals_the_one_cell_fill():
    one = _cells_clicks([(10, 15, 15)], [500])
    two = _cells_clicks([(10, 15, 15), (7, 9, 20)], [500, 500])
    assert one.ok and two.ok, (one.refusal, two.refusal)
    assert int((one.result == 500).sum()) > 1000
    assert two.ids == [500]
    assert np.array_equal(one.result, two.result)
    assert two.seed_why == [None, None] and two.apart_ids == []


def test_a_cells_seed_whose_cell_does_not_touch_is_refused():
    """Two walled cells with open space between them: a second seed of the
    object in the other cell grows a cell that does not touch the first,
    so it is refused and its cell left empty; the first is unchanged."""
    one = _cells_clicks([(10, 15, 7)], [500], two=True)
    out = _cells_clicks([(10, 15, 7), (10, 15, 22)], [500, 500], two=True)
    assert one.ok and out.ok, (one.refusal, out.refusal)
    assert out.apart_ids == [500]
    assert out.seed_why == [None, "apart"]
    assert not out.result[10, 15, 22]
    assert np.array_equal(out.result == 500, one.result == 500)
    assert "not_connected=[500]" in out.log_line()


def test_a_cells_refused_seed_leaves_the_answer_it_never_placed():
    """A(#500) in one cell, B(#500) and C(#501) in the other: B is refused,
    and C's cell is the one it grows with B never placed (B's share is not
    left empty); no faint pair names an engine id no object kept."""
    A, B, C = (10, 15, 7), (10, 10, 22), (10, 20, 22)
    never = _cells_clicks([A, C], [500, 501], two=True)
    for pts, ids, why in (([A, B, C], [500, 500, 501], [None, "apart", None]),
                          ([A, C, B], [500, 501, 500], [None, None, "apart"])):
        out = _cells_clicks(pts, ids, two=True)
        assert out.ok, out.refusal
        assert out.seed_why == why and out.apart_ids == [500]
        assert np.array_equal(out.result, never.result)
        for a, b, _m in out.faint_pairs:
            assert a in out.ids and b in out.ids


# --------------------------------------------------------------------------- #
# the gesture (the Qt-free harness; Nuclei ▸ At the markers I place)
# --------------------------------------------------------------------------- #
#: a second spot on nucleus A, outside the first click's ball
A2 = (12, 22, 22)


def _ids(hh):
    return [int(m["id"]) for m in hh._add_click_markers]


def test_a_plain_click_makes_a_new_object_which_becomes_current(off):
    assert off._add_marker_click(NUC_A)
    assert off._add_current_rec() is off._add_click_markers[-1]
    assert off._add_marker_click(NUC_B)
    assert off._add_current_rec() is off._add_click_markers[-1]
    assert len(set(_ids(off))) == 2


def test_shift_click_adds_a_seed_to_the_current_object(off):
    assert off._add_marker_click(NUC_A)
    a = _ids(off)[0]
    e0 = off._add_markers_epoch
    assert off._add_marker_click(A2, shift=True)
    assert _ids(off) == [a, a]
    assert off._add_markers_epoch == e0 + 1
    # the reference moves to the new seed: still the same object
    assert off._add_current_rec() is off._add_click_markers[-1]
    assert off._add_current_rec()["id"] == a
    # one row per click, both naming the object
    assert [r[0] for r in off.adv.last("click_markers")] == [a, a]
    # its ball has the object's colour
    layer = off._add_marker_layer
    assert layer.data[A2] == layer.data[NUC_A] == AP.ball_value(a)


def test_shift_click_wherever_it_lands_joins_the_current_object(off):
    """Even on the other nucleus: the seed is placed (the engine refuses
    it later if it does not touch)."""
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(NUC_B, shift=True)
    assert len(set(_ids(off))) == 1


def test_shift_click_with_no_object_current_is_a_plain_click(off):
    assert off._add_marker_click(NUC_A)
    assert off._add_esc_current()
    assert off._add_current_rec() is None
    assert off._add_marker_click(NUC_B, shift=True)
    assert len(set(_ids(off))) == 2
    assert off._add_current_rec() is off._add_click_markers[-1]


def test_shift_click_keeps_the_refusals_of_a_click(off):
    assert off._add_marker_click(NUC_A)
    n = len(off._add_click_markers)
    assert not off._add_marker_click((12, 21, 20), shift=True)   # in a ball
    assert not off._add_marker_click((6, 6, 6), shift=True)      # kept obj
    assert not off._add_marker_click((1, 1, 1), shift=True)      # outside
    assert len(off._add_click_markers) == n


def test_ctrl_wins_over_shift_and_alt_places_nothing_yet(off):
    assert off._add_marker_click(NUC_A)
    assert not off._add_marker_click(NUC_B, alt=True)
    assert len(off._add_click_markers) == 1
    assert off._add_marker_click(NUC_A, ctrl=True, shift=True)
    assert off._add_click_markers == []


def test_undo_last_takes_back_seeds_and_then_the_object(off):
    assert off._add_marker_click(NUC_A)
    a = _ids(off)[0]
    head = off._add_click_markers[0]
    assert off._add_marker_click(A2, shift=True)
    off._add_click_undo()
    assert _ids(off) == [a]
    assert off._add_current_rec() is head          # still the object
    off._add_click_undo()
    assert off._add_click_markers == []
    assert off._add_current_rec() is None          # its last seed went


def test_ids_stay_put_through_removal(off):
    assert off._add_marker_click(NUC_A)
    a = _ids(off)[0]
    assert off._add_marker_click(NUC_B)
    b = _ids(off)[1]
    assert off._add_marker_click((12, 10, 23), shift=True)
    assert _ids(off) == [a, b, b]
    # the first object goes: the other keeps its id and stays current
    assert off._add_marker_click(NUC_A, ctrl=True)
    assert _ids(off) == [b, b]
    assert off._add_current_rec()["id"] == b
    # the object's first seed goes: its other seed keeps the id
    assert off._add_marker_click(NUC_B, ctrl=True)
    assert _ids(off) == [b]
    assert off._add_current_rec() is off._add_click_markers[0]


def test_removing_the_current_record_moves_it_to_the_object_s_last_seed(off):
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(A2, shift=True)
    seed = off._add_click_markers[-1]
    head = off._add_click_markers[0]
    assert off._add_marker_click(A2, ctrl=True)      # the current record
    assert seed not in off._add_click_markers
    assert off._add_current_rec() is head


def test_clear_all_markers_leaves_nothing_current(off):
    assert off._add_marker_click(NUC_A)
    off._add_click_clear()
    assert off._add_current_rec() is None


def test_the_re_base_moves_the_current_object_with_its_seeds(off):
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(A2, shift=True)
    a = _ids(off)[0]
    assert off._add_rebase_markers(int(off._add_markers_base) + 4)
    assert _ids(off) == [a + 4, a + 4]
    assert off._add_current_rec()["id"] == a + 4
    # the next seed re-bases first (the volume's next free number did not
    # move): the object, its seeds and the reference move back together
    assert off._add_marker_click((12, 18, 18), shift=True)
    assert _ids(off) == [a] * 3
    assert off._add_current_rec()["id"] == a


def test_a_stale_reference_is_not_current(off):
    assert off._add_marker_click(NUC_A)
    off._add_click_markers = []                     # an exit cleared it
    assert off._add_current_rec() is None
    assert off._add_current is None


def test_a_tick_ticks_every_seed_and_a_new_seed_inherits_it(off):
    assert off._add_marker_click(NUC_A)
    a = _ids(off)[0]
    assert off._add_marker_click(A2, shift=True)
    off._on_add_click_tick(a, False)
    assert [m["ticked"] for m in off._add_click_markers] == [False, False]
    assert off._add_marker_click((12, 18, 18), shift=True)
    assert off._add_click_markers[-1]["ticked"] is False
    off._on_add_click_tick(a, True)
    assert all(m["ticked"] for m in off._add_click_markers)


def test_an_object_s_marks_go_only_with_its_last_seed(off):
    assert off._add_marker_click(NUC_A)
    a = _ids(off)[0]
    assert off._add_marker_click(A2, shift=True)
    off._add_marker_why = {a: "dim"}
    off._add_click_undo()
    assert off._add_marker_why == {a: "dim"}
    off._add_click_undo()
    assert off._add_marker_why == {}


def test_a_box_move_dropping_the_current_object_clears_it(off):
    assert off._add_marker_click(NUC_B)
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(A2, shift=True)
    # the box moves to y 4..11: B stays, the current object's seeds go
    off._on_add_box_changed(12, 8, 20, 10, 8, 10)
    assert [tuple(m["zyx"]) for m in off._add_click_markers] == [NUC_B]
    assert off._add_current_rec() is None


def test_esc_clears_current_and_live_off_keeps_the_result(off):
    assert off._add_marker_click(NUC_A)
    off._preview_add()
    _finish(off)
    assert off._add_preview_result is not None
    e0 = off._add_markers_epoch
    assert off._add_esc_current()
    assert off._add_current_rec() is None
    assert off._add_markers_epoch == e0                  # no engine input
    assert off._add_preview_result is not None           # nothing dropped
    assert not off._add_esc_current()                  # nothing current


def test_the_manual_flow_extends_and_refuses_with_a_sentence(off):
    assert off._add_marker_click(NUC_A)
    off._preview_add()
    _finish(off)
    a = _ids(off)[0]
    one = off._add_preview_result == a
    # a seed on A itself: the same nucleus
    assert off._add_marker_click(A2, shift=True)
    off._preview_add()
    _finish(off)
    assert np.array_equal(off._add_preview_result == a, one)
    assert off._add_transient != AP.ADD_S1_NOT_CONNECTED.format(id=a)
    # a seed on B: refused, the object unchanged, S1 says why
    assert off._add_marker_click(NUC_B, shift=True)
    off._preview_add()
    _finish(off)
    assert off._add_preview_res.apart_ids == [a]
    assert np.array_equal(off._add_preview_result == a, one)
    assert set(np.unique(off._add_preview_result).tolist()) <= {0, a}
    assert off._add_transient == AP.ADD_S1_NOT_CONNECTED.format(id=a)
    # Undo last removes the refused seed; the next result says nothing
    off._add_click_undo()
    off._preview_add()
    _finish(off)
    assert off._add_preview_res.apart_ids == []
    assert off._add_transient is None


# --------------------------------------------------------------------------- #
# live: a seed and an undo are marker changes, as a click is; Esc is not
# --------------------------------------------------------------------------- #
def _draw(hh):
    _fire(hh)
    _finish(hh)
    assert hh._add_preview_result is not None, hh._add_transient


def test_live_a_seed_and_an_undo_each_schedule_one_run(lv):
    assert lv._add_marker_click(NUC_A)
    _draw(lv)
    a = _ids(lv)[0]
    one = lv._add_preview_result == a
    for change in (lambda: lv._add_marker_click(A2, shift=True),
                   lv._add_click_undo):
        e0 = lv._add_markers_epoch
        n0 = len(FakeThread.instances)
        assert change() is not False
        assert lv._add_markers_epoch == e0 + 1
        assert _active(lv), "a live run is scheduled"
        # the drawn result stays until the next one replaces it
        assert lv._add_preview_result is not None
        _draw(lv)
        assert len(FakeThread.instances) == n0 + 1
        assert np.array_equal(lv._add_preview_result == a, one)


def test_live_esc_runs_nothing_and_keeps_the_result(lv):
    """Esc changes no engine input: no epoch, no run, the result stays --
    and it is not refused while an Accept waits."""
    assert lv._add_marker_click(NUC_A)
    _draw(lv)
    e0 = lv._add_markers_epoch
    n0 = len(FakeThread.instances)
    drawn = lv._add_preview_result
    lv._add_accept_pending = True
    assert lv._add_esc_current()
    lv._add_accept_pending = False
    assert lv._add_current_rec() is None
    assert lv._add_markers_epoch == e0
    assert not _active(lv) and not lv._add_live_dirty
    assert len(FakeThread.instances) == n0
    assert lv._add_preview_result is drawn


def test_live_a_seed_mid_run_makes_the_run_stale_and_one_more_follows(lv):
    assert lv._add_marker_click(NUC_A)
    _fire(lv)
    th = FakeThread.instances[-1]
    assert lv._add_marker_click(A2, shift=True)          # mid-run
    assert lv._add_live_dirty
    th.run()
    th.end()
    assert len(FakeThread.instances) == 2               # one more run
    _finish(lv)
    assert lv._add_preview_result is not None


def test_not_connected_is_said_once_not_on_every_later_result(lv):
    assert lv._add_marker_click(NUC_A)
    _draw(lv)
    a = _ids(lv)[0]
    assert lv._add_marker_click(NUC_B, shift=True)
    _draw(lv)
    assert lv._add_transient == AP.ADD_S1_NOT_CONNECTED.format(id=a)
    assert lv._add_marker_click((12, 15, 26))     # an unrelated new object
    _draw(lv)
    assert lv._add_transient != AP.ADD_S1_NOT_CONNECTED.format(id=a)


def test_not_connected_never_covers_a_sentence_a_clear_owes(lv):
    assert lv._add_marker_click(NUC_A)
    _draw(lv)
    assert lv._add_marker_click(NUC_B, shift=True)
    lv._add_live_keep_transient = "kept"
    _draw(lv)
    assert lv._add_transient == "kept"


def test_a_plain_click_on_a_refused_seed_replaces_it(off):
    """What the refusal advises -- click the spot plainly -- works: the
    refused marker goes and the click is a new object, in one step."""
    assert off._add_marker_click(NUC_A)
    a = _ids(off)[0]
    assert off._add_marker_click(NUC_B, shift=True)
    off._preview_add()
    _finish(off)
    assert off._add_transient == AP.ADD_S1_NOT_CONNECTED.format(id=a)
    assert off._add_marker_click(NUC_B)
    assert off._add_transient is None
    assert [tuple(m["zyx"]) for m in off._add_click_markers] == [NUC_A, NUC_B]
    b = _ids(off)[1]
    assert b != a
    assert off._add_current_rec() is off._add_click_markers[-1]
    off._preview_add()
    _finish(off)
    assert off._add_preview_res.apart_ids == []
    assert (off._add_preview_result == b).any()
    assert off._add_transient != AP.ADD_S1_NOT_CONNECTED.format(id=a)


def test_a_shift_click_on_a_refused_seed_is_still_refused(off):
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(NUC_B, shift=True)
    off._preview_add()
    _finish(off)
    assert not off._add_marker_click(NUC_B, shift=True)
    assert len(off._add_click_markers) == 2


def test_live_a_not_connected_seed_says_so_and_keeps_the_object(lv):
    assert lv._add_marker_click(NUC_A)
    _draw(lv)
    a = _ids(lv)[0]
    one = lv._add_preview_result == a
    assert lv._add_marker_click(NUC_B, shift=True)
    _draw(lv)
    assert np.array_equal(lv._add_preview_result == a, one)
    assert not lv._add_preview_result[NUC_B]
    assert lv._add_transient == AP.ADD_S1_NOT_CONNECTED.format(id=a)


# --------------------------------------------------------------------------- #
# the keys: the side views' own filters, by focus, never napari's keymap
# --------------------------------------------------------------------------- #
def _key(key, mods=Qt.NoModifier, kind=None):
    return QKeyEvent(kind or QEvent.KeyPress, key, mods)


def _ctrl_z(kind=None):
    return _key(Qt.Key_Z, Qt.ControlModifier, kind)


class _Dims:
    current_step = (0,)
    range = ((0, 10, 1),)

    def set_current_step(self, *_a):
        pass


class _VMd:
    dims = _Dims()


class _Step:
    def value(self):
        return 1


def _filters(hh, orient):
    nav = []
    f = CW._SliceStepFilter(_VMd(), _Step(), navigate_callback=nav.append,
                            nudge_callback=hh._add_nudge_box, orient=orient,
                            marker_key_callback=hh._add_marker_key)
    b = CW._UpDownBlocker(navigate_callback=nav.append,
                          nudge_callback=hh._add_nudge_box, orient=orient,
                          marker_key_callback=hh._add_marker_key)
    return f, b


def _with_history(layer):
    """A curated side copy with one paint of its own to undo: a history
    entry as napari records one, and the write itself (napari's own
    ``data_setitem`` refreshes the displayed slice and refuses a voxel
    outside it)."""
    idx = tuple(int(v) for v in np.argwhere(layer.data == 0)[0])
    ind = tuple(np.array([v]) for v in idx)
    layer._save_history((ind, np.array(layer.data[ind], copy=True), 3))
    layer.data[ind] = 3
    assert len(layer._undo_history) >= 1
    return idx


@pytest.mark.parametrize("orient", ("XY", "XZ", "YZ"))
@pytest.mark.parametrize("which", ("filter", "blocker"))
def test_ctrl_z_in_a_side_view_is_undo_last_never_the_layer_s_undo(
        h, orient, which):  # noqa: F811
    copies = h.main_copies
    for c in copies.values():
        _with_history(c)
    before = {o: (len(c._undo_history), c.data.copy())
              for o, c in copies.items()}
    assert h._add_marker_click((12, 20, 20))
    assert h._add_marker_click((12, 24, 24), shift=True)
    f, b = _filters(h, orient)
    flt = f if which == "filter" else b
    over = _ctrl_z(QEvent.ShortcutOverride)
    assert flt.eventFilter(None, over) is True
    assert over.isAccepted()
    took = flt.eventFilter(None, _ctrl_z())
    if not took:
        copies[orient].undo()                 # what napari would do next
    assert took is True
    assert len(h._add_click_markers) == 1
    for o, c in copies.items():
        n, data = before[o]
        assert len(c._undo_history) == n, o
        assert np.array_equal(c.data, data), o


@pytest.mark.parametrize("orient", ("XY", "XZ", "YZ"))
def test_the_key_is_the_focused_view_s_wherever_the_pointer_is(
        h, orient):  # noqa: F811
    """The click was made in XY; the key reaches whichever view's filter
    has the focus -- the filter reads no pointer position."""
    assert h._add_marker_click((12, 20, 20))
    f, _b = _filters(h, orient)
    assert f.eventFilter(None, _ctrl_z()) is True
    assert h._add_click_markers == []


@pytest.mark.parametrize("orient", ("XY", "XZ", "YZ"))
def test_esc_in_a_side_view_clears_the_current_object(h, orient):  # noqa: F811
    assert h._add_marker_click((12, 20, 20))
    f, b = _filters(h, orient)
    esc = _key(Qt.Key_Escape, kind=QEvent.ShortcutOverride)
    assert f.eventFilter(None, esc) is True and esc.isAccepted()
    assert b.eventFilter(None, _key(Qt.Key_Escape)) is True
    assert h._add_current_rec() is None
    assert len(h._add_click_markers) == 1                # nothing removed


@pytest.mark.parametrize("orient", ("XY", "XZ", "YZ"))
def test_with_place_markers_off_the_keys_do_nothing(h, orient):  # noqa: F811
    assert h._add_marker_click((12, 20, 20))
    cur = h._add_current_rec()
    h._add_place_markers_on = False
    f, b = _filters(h, orient)
    for flt in (f, b):
        assert flt.eventFilter(None, _ctrl_z()) is False
        assert flt.eventFilter(None, _key(Qt.Key_Escape)) is False
        over = _ctrl_z(QEvent.ShortcutOverride)
        assert flt.eventFilter(None, over) is False
    assert len(h._add_click_markers) == 1
    assert h._add_current_rec() is cur


def test_outside_a_box_session_the_keys_are_not_add_s(h):  # noqa: F811
    h._add_running = False
    f, b = _filters(h, "XY")
    assert f.eventFilter(None, _ctrl_z()) is False
    assert b.eventFilter(None, _key(Qt.Key_Escape)) is False


def test_other_keys_pass_through(h):  # noqa: F811
    assert h._add_marker_click((12, 20, 20))
    f, b = _filters(h, "XY")
    for ev in (_key(Qt.Key_Z),
               _key(Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier),
               _key(Qt.Key_Escape, Qt.ShiftModifier)):
        assert CW._add_marker_key_route(ev, "XY", h._add_marker_key) is False
    assert len(h._add_click_markers) == 1


def test_filters_without_the_callback_are_as_today(h):  # noqa: F811
    assert h._add_marker_click((12, 20, 20))
    f = CW._SliceStepFilter(_VMd(), _Step(), orient="XY")
    b = CW._UpDownBlocker(orient="XY")
    assert f.eventFilter(None, _ctrl_z()) is False
    assert b.eventFilter(None, _ctrl_z()) is False
    assert len(h._add_click_markers) == 1


def test_live_ctrl_z_is_a_marker_change(lv):
    assert lv._add_marker_click(NUC_A)
    _draw(lv)
    e0 = lv._add_markers_epoch
    f, _b = _filters(lv, "XZ")
    assert f.eventFilter(None, _ctrl_z()) is True
    assert lv._add_markers_epoch == e0 + 1
    assert lv._add_click_markers == []
    assert lv._add_current_rec() is None


def test_the_installed_filters_carry_the_marker_key():
    import inspect

    src = inspect.getsource(CW.Curation3DAnnotator._install_arrow_key_filters)
    assert src.count("marker_key_callback=key_cb") == 2
    assert "key_cb = self._add_marker_key" in src


# --------------------------------------------------------------------------- #
# the three click routes carry Shift, Alt and Ctrl
# --------------------------------------------------------------------------- #
@pytest.fixture
def routes(monkeypatch):
    from napari.components import ViewerModel

    from .add_harness import Harness, install_fakes
    from .test_add_box_drag_wiring import BOX, _Overlay, _QtV, _Tabs

    install_fakes(monkeypatch)
    hh = Harness(np.zeros((40, 60, 80), np.int32))
    for o in ("xy", "xz", "yz"):
        setattr(hh, f"viewer_model_{o}", ViewerModel())
        setattr(hh, f"qt_viewer_{o}", _QtV())
    hh.adv._tool_tabs = _Tabs("Add")
    hh.deferred = []
    hh._add_defer = hh.deferred.append
    hh._add_box_overlay = _Overlay()
    hh.place_box(BOX)
    hh._install_add_side_callbacks()
    hh._add_place_markers_on = True
    hh.seen = []

    def _rec(zyx, ctrl=False, shift=False, alt=False):
        hh.seen.append((ctrl, shift, alt))
        return True

    hh._add_marker_click = _rec
    return hh


def _route(hh, route, mods):
    from .test_add_box_drag_wiring import _dispatch, _Ev

    vm = hh.viewer_model_xy
    if route == "dbl":
        _dispatch(vm, "dbl", _Ev("mouse_double_click", (20, 30, 45),
                                 (100, 100), modifiers=mods))
    else:
        # the grab band: a press on the box's outline (y = 20) that
        # releases as a click
        at = (20, 30, 45) if route == "plain" else (20, 20, 45)
        _dispatch(vm, "press", _Ev("mouse_press", at, (100, 100),
                                   modifiers=mods))
        _dispatch(vm, "release", _Ev("mouse_release", at, (101, 100),
                                     modifiers=mods))
    while hh.deferred:
        hh.deferred.pop(0)()


@pytest.mark.parametrize("route", ("plain", "band", "dbl"))
@pytest.mark.parametrize("mods,want", [
    ((), (False, False, False)),
    (("Shift",), (False, True, False)),
    (("Alt",), (False, False, True)),
    (("Control",), (True, False, False)),
    (("Control", "Shift"), (True, True, False)),
])
def test_every_click_route_carries_the_modifiers(routes, route, mods, want):
    _route(routes, route, mods)
    assert routes.seen == [want]


def test_undo_last_and_remove_all_never_take_the_keyboard(qapp):  # noqa: F811
    """Clicking either button leaves the focus on the side view, so a Ctrl+Z
    after it still reaches Add's marker undo there."""
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget()
    try:
        for b in (w.add_click_undo_btn, w.add_click_clear_btn):
            assert b.focusPolicy() == Qt.NoFocus, b.text()
    finally:
        w.setParent(None)
        w.deleteLater()

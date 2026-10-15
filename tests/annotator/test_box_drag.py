"""Box drag in the side views -- the pure half.

``packages/epicycle_annotator/box_drag.py`` and
``orientation.subview_point_to_zyx``: a side view's point mapped back to
volume ZYX (checked against ``transpose_data`` itself, for all three views,
with a scale and a translate, and with a 4-entry position), the outline hit
on every edge and corner within ``6 / zoom`` and its misses, the
click-vs-drag threshold at exactly ``ADD_CLICK_DRAG_PX``, a moved box
clamped to the volume, and the Shift+arrow nudge per view.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import box_drag as BD
from packages.epicycle_annotator.orientation import (
    subview_point_to_zyx,
    transpose_data,
)
from packages.epicycle_annotator.roi_box import BoxRec


# --------------------------------------------------------------------------
# subview_point_to_zyx
# --------------------------------------------------------------------------

@pytest.mark.parametrize("orient", ["XY", "XZ", "YZ"])
def test_point_round_trips_against_transpose_data(orient):
    """Every voxel of a small volume: the side view's index of that voxel
    maps back to the voxel's own ZYX."""
    shape = (3, 4, 5)
    vol = np.arange(np.prod(shape)).reshape(shape)
    view = transpose_data(vol, orient)
    for idx in np.ndindex(view.shape):
        z, y, x = subview_point_to_zyx(orient, idx)
        assert vol[int(z), int(y), int(x)] == view[idx]


def test_axis_maps_as_the_spec_says():
    assert subview_point_to_zyx("XY", (1, 2, 3)) == (1, 2, 3)
    assert subview_point_to_zyx("XZ", (1, 2, 3)) == (2, 1, 3)
    assert subview_point_to_zyx("YZ", (1, 2, 3)) == (3, 2, 1)


def test_a_four_entry_position_maps_its_last_three():
    """Grill E-L11: never ``position[:3]``."""
    assert subview_point_to_zyx("XZ", (9, 1, 2, 3)) == (2, 1, 3)
    assert subview_point_to_zyx("YZ", (7, 7, 1, 2, 3)) == (3, 2, 1)


@pytest.mark.parametrize("orient", ["XY", "XZ", "YZ"])
def test_scale_and_translate_are_undone_in_the_view_order(orient):
    """A world point = translate + scale * data (napari), in the side
    view's own axis order."""
    scale = (2.0, 0.5, 4.0)
    translate = (10.0, -3.0, 1.0)
    data = (1.0, 6.0, 2.0)
    world = tuple(t + s * d for t, s, d in zip(translate, scale, data))
    got = subview_point_to_zyx(orient, world, scale=scale,
                               translate=translate)
    want = subview_point_to_zyx(orient, data)
    assert np.allclose(got, want)


def test_anisotropic_scale_alone():
    got = subview_point_to_zyx("XZ", (6.0, 4.0, 3.0), scale=(3.0, 2.0, 1.0))
    # data (y=2, z=2, x=3) shown as (y, z, x)
    assert np.allclose(got, (2.0, 2.0, 3.0))


def test_unknown_orientation_raises():
    with pytest.raises(ValueError):
        subview_point_to_zyx("ZZ", (1, 2, 3))


# --------------------------------------------------------------------------
# outline_hit
# --------------------------------------------------------------------------

B = {"z0": 10, "z1": 30, "y0": 20, "y1": 60, "x0": 40, "x1": 100}


def _pt(orient, row, col, slc):
    """(z, y, x) for a point at (row, col) on screen and slice *slc*."""
    if orient == "XY":
        return (slc, row, col)
    if orient == "XZ":
        return (row, slc, col)
    return (col, row, slc)          # YZ: rows are Y, cols are Z


# (row0, row1, col0, col1, a slice inside the box) per view
RECT = {
    "XY": (20, 60, 40, 100, 20),
    "XZ": (10, 30, 40, 100, 40),
    "YZ": (20, 60, 10, 30, 70),
}


@pytest.mark.parametrize("orient", ["XY", "XZ", "YZ"])
def test_hit_on_each_edge_and_corner(orient):
    r0, r1, c0, c1, s = RECT[orient]
    rm, cm = (r0 + r1) / 2, (c0 + c1) / 2
    for row, col in ((r0, cm), (r1, cm), (rm, c0), (rm, c1),
                     (r0, c0), (r0, c1), (r1, c0), (r1, c1)):
        assert BD.outline_hit(B, orient, _pt(orient, row, col, s), 1.5)


@pytest.mark.parametrize("orient", ["XY", "XZ", "YZ"])
def test_hit_within_tolerance_and_miss_beyond(orient):
    r0, r1, c0, c1, s = RECT[orient]
    cm = (c0 + c1) / 2
    tol = 3.0
    assert BD.outline_hit(B, orient, _pt(orient, r0 - 2.9, cm, s), tol)
    assert BD.outline_hit(B, orient, _pt(orient, r0 + 2.9, cm, s), tol)
    assert not BD.outline_hit(B, orient, _pt(orient, r0 - 3.1, cm, s), tol)
    # a corner, diagonally just outside both edges, still within tol
    assert BD.outline_hit(B, orient, _pt(orient, r1 + 2, c1 + 2, s), tol)
    assert not BD.outline_hit(B, orient, _pt(orient, r1 + 4, c1 + 4, s), tol)


@pytest.mark.parametrize("orient", ["XY", "XZ", "YZ"])
def test_miss_inside_and_outside(orient):
    r0, r1, c0, c1, s = RECT[orient]
    rm, cm = (r0 + r1) / 2, (c0 + c1) / 2
    assert not BD.outline_hit(B, orient, _pt(orient, rm, cm, s), 2.0)
    assert not BD.outline_hit(B, orient, _pt(orient, r1 + 30, cm, s), 2.0)
    assert not BD.outline_hit(B, orient, _pt(orient, rm, c0 - 30, s), 2.0)
    # the right row but far beyond the edge's own length
    assert not BD.outline_hit(B, orient, _pt(orient, r0, c1 + 30, s), 2.0)


@pytest.mark.parametrize("orient", ["XY", "XZ", "YZ"])
def test_the_slice_must_be_inside_the_box_range(orient):
    r0, r1, c0, c1, _s = RECT[orient]
    cm = (c0 + c1) / 2
    lo, hi = {"XY": (10, 30), "XZ": (20, 60), "YZ": (40, 100)}[orient]
    assert BD.outline_hit(B, orient, _pt(orient, r0, cm, lo), 2.0)
    assert BD.outline_hit(B, orient, _pt(orient, r0, cm, hi - 1), 2.0)
    assert not BD.outline_hit(B, orient, _pt(orient, r0, cm, lo - 1), 2.0)
    assert not BD.outline_hit(B, orient, _pt(orient, r0, cm, hi), 2.0)


@pytest.mark.parametrize("orient", ["XY", "XZ", "YZ"])
def test_slice_in_box_matches_outline_hit(orient):
    """The dimmed rectangle (LOW-4) and the grab agree: in range exactly
    when a press on the edge would be a hit."""
    _r0, _r1, _c0, _c1, _s = RECT[orient]
    axis = BD.view_axes(orient)[2]
    k = "zyx"[axis]
    lo, hi = B[f"{k}0"], B[f"{k}1"]
    for s in (lo - 1, lo, (lo + hi) // 2, hi - 1, hi):
        want = BD.outline_hit(B, orient, _pt(orient, _r0, _c0, s), 1.0)
        assert BD.slice_in_box(B, orient, s) == want == (lo <= s < hi)
    assert BD.slice_in_box(B, orient, None) is True
    assert BD.slice_in_box(B, "??", 5) is True


def test_inside_box_is_a_separate_question():
    s = RECT["XY"][4]
    assert BD.inside_box(B, _pt("XY", 30, 50, s))
    assert not BD.inside_box(B, _pt("XY", 30, 150, s))
    assert not BD.inside_box(B, (None, 30, 50))


def test_tolerance_follows_zoom():
    assert BD.outline_tolerance(1.0) == pytest.approx(6.0)
    assert BD.outline_tolerance(3.0) == pytest.approx(2.0)
    assert BD.outline_tolerance(0.5) == pytest.approx(12.0)
    # a zoom of 0 or nonsense never divides by zero
    assert BD.outline_tolerance(0.0) > 0
    assert BD.outline_tolerance(None) == pytest.approx(6.0)
    # at zoom 3 a point 2.5 voxels off the edge (7.5 px) misses
    r0, _r1, _c0, _c1, s = RECT["XY"]
    tol = BD.outline_tolerance(3.0)
    assert not BD.outline_hit(B, "XY", _pt("XY", r0 - 2.5, 60, s), tol)
    assert BD.outline_hit(B, "XY", _pt("XY", r0 - 1.9, 60, s), tol)


def test_tolerance_is_capped_by_max_vox():
    assert BD.outline_tolerance(0.6) == pytest.approx(10.0)
    assert BD.outline_tolerance(0.6, max_vox=5) == pytest.approx(5.0)
    assert BD.outline_tolerance(3.0, max_vox=5) == pytest.approx(2.0)
    assert BD.outline_tolerance(1.0, max_vox=None) == pytest.approx(6.0)


@pytest.mark.parametrize("orient,cap", [("XY", 5.0), ("XZ", 2.5),
                                        ("YZ", 2.5)])
def test_outline_cap_is_a_quarter_of_the_smaller_displayed_extent(orient,
                                                                   cap):
    b = {"z0": 0, "z1": 10, "y0": 0, "y1": 20, "x0": 0, "x1": 30}
    assert BD.outline_cap(b, orient) == pytest.approx(cap)
    assert BD.outline_cap(b, "??") is None


# --------------------------------------------------------------------------
# is_click / drag_delta / moved_box / nudge_delta
# --------------------------------------------------------------------------

def test_click_threshold_is_four_pixels():
    assert BD.ADD_CLICK_DRAG_PX == 4
    assert BD.is_click((10, 10), (10, 10))
    assert BD.is_click((10, 10), (14, 10))            # exactly 4 px
    assert BD.is_click((10, 10), (10, 6))
    assert not BD.is_click((10, 10), (14.01, 10))
    assert not BD.is_click((10, 10), (13, 13))        # 4.24 px
    assert BD.is_click((0, 0), (5, 0), threshold_px=5)
    assert not BD.is_click(None, (0, 0))              # unknown -> not a click


_ROW = {"XY": 1, "XZ": 0, "YZ": 1}
_COL = {"XY": 2, "XZ": 2, "YZ": 0}


@pytest.mark.parametrize("orient,moved", [
    ("XY", (0, 3, -2)),
    ("XZ", (3, 0, -2)),
    ("YZ", (-2, 3, 0)),
])
def test_drag_delta_moves_only_the_displayed_axes(orient, moved):
    a = (10.0, 20.0, 30.0)
    b = list(a)
    b[_ROW[orient]] += 3
    b[_COL[orient]] -= 2
    # ... and on the slice axis too, which must be ignored
    sl = ({0, 1, 2} - {_ROW[orient], _COL[orient]}).pop()
    b[sl] += 7
    assert BD.drag_delta(orient, a, tuple(b)) == moved


def test_drag_delta_rounds_to_whole_voxels():
    assert BD.drag_delta("XY", (0, 0, 0), (0, 2.6, -1.4)) == (0, 3, -1)


def test_moved_box_slides_inside_the_volume_and_keeps_its_size():
    """B3 fix pass (orchestrator decision): a drag past an edge SLIDES the
    box back inside the volume at its full size -- it never truncates."""
    shape = (40, 80, 120)
    box = BoxRec(20, 40, 60, 10, 20, 30)
    inside = BD.moved_box(box, (0, 5, -7), shape, axes=(1, 2))
    assert (inside.cz, inside.cy, inside.cx) == (20, 45, 53)
    assert (inside.sz, inside.sy, inside.sx) == (10, 20, 30)
    far = BD.moved_box(box, (0, 500, -500), shape, axes=(1, 2))
    b = far.bounds_dict(shape)
    assert b["y1"] == shape[1] and b["y1"] - b["y0"] == 20
    assert b["x0"] == 0 and b["x1"] - b["x0"] == 30
    assert far.cz == 20


_DISPLAYED = {"XY": (1, 2), "XZ": (0, 2), "YZ": (1, 0)}


@pytest.mark.parametrize("orient", ["XY", "XZ", "YZ"])
@pytest.mark.parametrize("sign", [1000, -1000])
def test_a_far_drag_keeps_the_size_on_each_displayed_axis(orient, sign):
    shape = (60, 70, 80)
    box = BoxRec(30, 35, 40, 20, 20, 20)
    axes = BD.view_axes(orient)[:2]
    for axis in _DISPLAYED[orient]:
        d = [0, 0, 0]
        d[axis] = sign
        nb = BD.moved_box(box, tuple(d), shape, axes=axes)
        assert nb.shape_in(shape) == (20, 20, 20), (orient, axis, sign)
        b = nb.bounds_dict(shape)
        k = "zyx"[axis]
        if sign > 0:
            assert b[f"{k}1"] == shape[axis]
        else:
            assert b[f"{k}0"] == 0
        # idempotent: a second pass of the same far drag is a fixed point
        assert BD.moved_box(nb, tuple(d), shape, axes=axes) == nb


def test_the_slicing_axis_is_never_touched():
    """A box already overflowing on the slicing axis (z centre 9, size 6,
    depth 10) is unchanged by an XY drag of (0, 0, 0)."""
    shape = (10, 50, 50)
    box = BoxRec(9, 25, 25, 6, 10, 10)
    assert BD.moved_box(box, (0, 0, 0), shape, axes=(1, 2)) == box
    nb = BD.moved_box(box, (0, 3, -2), shape, axes=(1, 2))
    assert (nb.cz, nb.sz) == (9, 6)
    assert (nb.cy, nb.cx) == (28, 23)


def test_a_zero_delta_returns_the_normalised_box():
    shape = (40, 40, 40)
    box = BoxRec(20, 38, 20, 10, 10, 10)          # overflows y's high edge
    nb = BD.moved_box(box, (0, 0, 0), shape, axes=(1, 2))
    assert nb == box.slid_inward(shape)
    assert nb.shape_in(shape) == (10, 10, 10)


def test_a_nudge_at_the_bottom_edge_does_not_move_it_up():
    """XZ Shift+Down on a box touching the bottom stays put."""
    shape = (40, 60, 80)
    box = BoxRec(30, 30, 40, 20, 20, 20)           # z 20..40, the bottom
    d = BD.nudge_delta("XZ", "Down")
    nb = BD.moved_box(box, d, shape)
    assert nb == box
    assert nb.bounds_dict(shape)["z1"] == 40


def test_one_left_nudge_from_a_high_edge_box_moves_x0_by_one():
    shape = (40, 60, 80)
    box = BoxRec(20, 30, 40, 20, 20, 20)
    high = BD.moved_box(box, (0, 0, 1000), shape, axes=(1, 2))
    b0 = high.bounds_dict(shape)
    assert b0["x1"] == 80
    left = BD.moved_box(high, BD.nudge_delta("XY", "Left"), shape)
    b1 = left.bounds_dict(shape)
    assert b1["x0"] == b0["x0"] - 1
    assert b1["x1"] - b1["x0"] == 20


def test_moved_box_does_not_mutate_the_start_box():
    box = BoxRec(5, 5, 5, 4, 4, 4)
    BD.moved_box(box, (1, 1, 1), (20, 20, 20))
    assert box == BoxRec(5, 5, 5, 4, 4, 4)


@pytest.mark.parametrize("orient,key,delta", [
    ("XY", "Up", (0, -1, 0)), ("XY", "Down", (0, 1, 0)),
    ("XY", "Left", (0, 0, -1)), ("XY", "Right", (0, 0, 1)),
    ("XZ", "Up", (-1, 0, 0)), ("XZ", "Down", (1, 0, 0)),
    ("XZ", "Left", (0, 0, -1)), ("XZ", "Right", (0, 0, 1)),
    ("YZ", "Up", (0, -1, 0)), ("YZ", "Down", (0, 1, 0)),
    ("YZ", "Left", (-1, 0, 0)), ("YZ", "Right", (1, 0, 0)),
])
def test_nudge_delta_per_view(orient, key, delta):
    assert BD.nudge_delta(orient, key) == delta


def test_nudge_follows_the_screen_axes_of_transpose_data():
    """Right moves one column right in that view's own picture; Down one
    row down -- checked against ``transpose_data`` rather than the table."""
    shape = (6, 7, 8)
    vol = np.arange(np.prod(shape)).reshape(shape)
    start = (2, 3, 4)
    for orient in ("XY", "XZ", "YZ"):
        view = transpose_data(vol, orient)
        idx = [i for i in np.ndindex(view.shape)
               if view[i] == vol[start]][0]
        for key, d_row, d_col in (("Right", 0, 1), ("Down", 1, 0)):
            dz, dy, dx = BD.nudge_delta(orient, key)
            moved = (start[0] + dz, start[1] + dy, start[2] + dx)
            want = (idx[0], idx[1] + d_row, idx[2] + d_col)
            assert view[want] == vol[moved], (orient, key)


def test_nudge_delta_unknown_key_is_none():
    assert BD.nudge_delta("XY", "PageUp") is None
    assert BD.nudge_delta("??", "Up") is None

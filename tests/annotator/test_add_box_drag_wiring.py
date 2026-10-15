"""Box drag in the side views -- the wiring.

The real orchestrator on the Qt-free harness, with REAL napari
``ViewerModel``s as the side views and napari's own dispatch functions
(``mouse_press_callbacks`` / ``mouse_move_callbacks`` /
``mouse_release_callbacks``) driving the generators, so the drag protocol
is napari's, not a guess.  The vispy camera is the one thing that is not
real: :func:`_dispatch` stands in for it -- a move PANS unless it came
back ``handled`` or ``camera.mouse_pan`` is off, the two mechanisms of a pan.

Covered: install / remove at every exit, none in REVIEW, button 2
untouched, inert off the Add tab, the busy refusal, the release deferred
behind the token, ``"Control" in event.modifiers``, the camera centre
unchanged across a drag, a drag started inside the box panning, the hover
cursor.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari.components import ViewerModel
from napari.utils._proxies import ReadOnlyWrapper
from napari.utils.interactions import (
    mouse_double_click_callbacks,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.roi_box import BoxRec

from .add_harness import Harness, install_fakes, marker_zyx


class _Ev:
    """The napari mouse event, as far as the callbacks read it."""

    def __init__(self, type_, position, pos, button=1, modifiers=(),
                 dragging=False):
        self.type = type_
        self.position = tuple(float(v) for v in position)
        self.pos = pos
        self.button = button
        self.modifiers = tuple(modifiers)
        self.handled = False
        self.is_dragging = dragging


class _Tabs:
    def __init__(self, text="Add"):
        self.text = text

    def currentIndex(self):  # noqa: N802 - Qt name
        return 0

    def tabText(self, _i):  # noqa: N802 - Qt name
        return self.text


class _Native:
    def __init__(self):
        self.cursor = None
        self.unset = 0

    def setCursor(self, c):  # noqa: N802 - Qt name
        self.cursor = c

    def unsetCursor(self):  # noqa: N802 - Qt name
        self.cursor = None
        self.unset += 1


class _QtV:
    def __init__(self):
        class _C:
            pass

        self.canvas = _C()
        self.canvas.native = _Native()


class _Overlay:
    def __init__(self):
        self.updates = []
        self.in_range = {}

    def update(self, box, shape, **kw):
        self.updates.append((box, kw.get("keys")))
        self.in_range.update(kw.get("in_range") or {})

    def set_in_range(self, key, inside, **_kw):
        self.in_range[key] = bool(inside)

    def destroy(self):
        pass


def _dispatch(vm, kind, ev):
    """napari's canvas: one read-only wrapper per event, viewer callbacks
    run, ``handled`` copied back -- then the stand-in camera pans a move
    that is neither handled nor on a camera with ``mouse_pan`` off."""
    w = ReadOnlyWrapper(ev, exceptions=("handled",))
    {"press": mouse_press_callbacks, "move": mouse_move_callbacks,
     "release": mouse_release_callbacks,
     "dbl": mouse_double_click_callbacks}[kind](vm, w)
    if kind == "move" and ev.is_dragging and not ev.handled \
            and vm.camera.mouse_pan:
        cy, cx = vm.camera.center[-2:]
        vm.camera.center = (cy + 1.0, cx + 1.0)
    return ev


# the box: z 10..30, y 20..40, x 30..60 in a (40, 60, 80) volume
BOX = (slice(10, 30), slice(20, 40), slice(30, 60))


@pytest.fixture
def h(monkeypatch):
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
    return hh


def _run_deferred(h):
    while h.deferred:
        h.deferred.pop(0)()


def _drag(h, vm, start, end, pos0=(100, 100), pos1=(160, 100), n=3,
          button=1):
    evs = [_dispatch(vm, "press", _Ev("mouse_press", start, pos0,
                                      button=button))]
    for i in range(1, n + 1):
        f = i / n
        p = tuple(a + (b - a) * f for a, b in zip(start, end))
        px = tuple(a + (b - a) * f for a, b in zip(pos0, pos1))
        evs.append(_dispatch(vm, "move", _Ev("mouse_move", p, px,
                                             button=button, dragging=True)))
    evs.append(_dispatch(vm, "release", _Ev("mouse_release", end, pos1,
                                            button=button)))
    return evs


# --------------------------------------------------------------------------
# install / remove
# --------------------------------------------------------------------------

def _ours(h, vm):
    """The callbacks Add put on *vm* (napari's own ``drag_to_zoom`` is
    there from the start and stays)."""
    mine = {id(cb) for (_vm, d, m) in h._add_side_cbs.values()
            for cb in (d, m)}
    return ([cb for cb in vm.mouse_drag_callbacks if id(cb) in mine],
            [cb for cb in vm.mouse_move_callbacks if id(cb) in mine])


def _names(lst):
    return [getattr(cb, "__name__", "") for cb in lst]


def test_install_puts_one_drag_and_one_hover_callback_on_each_side_view(h):
    for o in ("xy", "xz", "yz"):
        vm = getattr(h, f"viewer_model_{o}")
        drag, move = _ours(h, vm)
        assert len(drag) == 1 and len(move) == 1
        # appended AFTER napari's own, never replacing it
        assert vm.mouse_drag_callbacks[-1] is drag[0]
    assert h._add_side_callbacks_installed


def test_install_twice_installs_once(h):
    for o in ("xy", "xz", "yz"):
        n = len(getattr(h, f"viewer_model_{o}").mouse_drag_callbacks)
        h._install_add_side_callbacks()
        assert len(getattr(h, f"viewer_model_{o}").mouse_drag_callbacks) == n


def test_remove_is_idempotent_and_leaves_the_lists_empty(h):
    h._remove_add_side_callbacks()
    h._remove_add_side_callbacks()
    for o in ("xy", "xz", "yz"):
        vm = getattr(h, f"viewer_model_{o}")
        assert _names(vm.mouse_drag_callbacks) == ["drag_to_zoom"]
        assert vm.mouse_move_callbacks == []
    assert not h._add_side_callbacks_installed


def test_remove_keeps_callbacks_it_did_not_install(h):
    vm = h.viewer_model_xy

    def other(_vm, _e):
        pass

    vm.mouse_drag_callbacks.append(other)
    h._remove_add_side_callbacks()
    assert _names(vm.mouse_drag_callbacks) == ["drag_to_zoom", "other"]


def test_reset_path_removes_the_callbacks(h):
    """Cancel / patch switch / close all end in the reset path."""
    h._add_preview_result = None
    h._add_secondary()                        # Cancel
    assert not h._add_side_callbacks_installed
    assert _names(h.viewer_model_yz.mouse_drag_callbacks) == ["drag_to_zoom"]


# --------------------------------------------------------------------------
# the drag
# --------------------------------------------------------------------------

@pytest.mark.parametrize("orient,start,end,centre", [
    # XY: position (z, y, x); grab the top edge y=20, move +5 y, +6 x
    ("xy", (20, 20, 45), (20, 25, 51), (20, 35, 51)),
    # XZ: position (y, z, x); grab the z=10 edge, move +4 z, -5 x
    ("xz", (30, 10, 45), (30, 14, 40), (24, 30, 40)),
    # YZ: position (x, y, z); grab the z=30 edge, move +3 y, -2 z
    ("yz", (45, 30, 30), (45, 33, 28), (18, 33, 45)),
])
def test_a_drag_on_the_outline_moves_the_whole_box(h, orient, start, end,
                                                   centre):
    vm = getattr(h, f"viewer_model_{orient}")
    cam0 = tuple(vm.camera.center)
    evs = _drag(h, vm, start, end)
    assert all(e.handled for e in evs), "every event of a drag is handled"
    assert tuple(vm.camera.center) == cam0, "the camera did not pan"
    assert vm.camera.mouse_pan is True, "pan is back after the release"
    # the side rectangles followed, the 3D wireframe did not
    assert h._add_box_overlay.updates
    assert all(k == ("XY", "XZ", "YZ") for _b, k in h._add_box_overlay.updates)
    # the box itself moves only on the deferred release
    assert (h._add_box.cz, h._add_box.cy, h._add_box.cx) == (20, 30, 45)
    _run_deferred(h)
    b = h._add_box
    assert (b.cz, b.cy, b.cx) == centre
    assert (b.sz, b.sy, b.sx) == (20, 20, 30)
    assert h.adv.last("box_center") == centre


def test_pan_is_off_during_the_drag_and_on_after(h):
    vm = h.viewer_model_xy
    _dispatch(vm, "press", _Ev("mouse_press", (20, 20, 45), (0, 0)))
    assert vm.camera.mouse_pan is False
    _dispatch(vm, "move", _Ev("mouse_move", (20, 22, 45), (0, 9),
                              dragging=True))
    assert vm.camera.mouse_pan is False
    _dispatch(vm, "release", _Ev("mouse_release", (20, 22, 45), (0, 9)))
    assert vm.camera.mouse_pan is True


def test_a_drag_started_inside_the_box_pans(h):
    vm = h.viewer_model_xy
    cam0 = tuple(vm.camera.center)
    evs = _drag(h, vm, (20, 30, 45), (20, 35, 50))
    assert not any(e.handled for e in evs)
    assert tuple(vm.camera.center) != cam0, "napari panned"
    _run_deferred(h)
    assert (h._add_box.cy, h._add_box.cx) == (30, 45)


def test_a_drag_started_off_the_box_pans(h):
    vm = h.viewer_model_xy
    cam0 = tuple(vm.camera.center)
    _drag(h, vm, (20, 5, 5), (20, 9, 9))
    assert tuple(vm.camera.center) != cam0
    assert not h.deferred


def test_the_slice_outside_the_box_is_not_a_hit(h):
    vm = h.viewer_model_xy
    evs = _drag(h, vm, (35, 20, 45), (35, 25, 50))    # z 35 not in 10..30
    assert not any(e.handled for e in evs)
    _run_deferred(h)
    assert h._add_box.cy == 30


def test_the_tolerance_is_six_pixels_at_the_view_zoom(h):
    vm = h.viewer_model_xy
    vm.camera.zoom = 3.0                   # 6 px = 2 voxels
    evs = _drag(h, vm, (20, 17.5, 45), (20, 20, 45))
    assert not evs[0].handled, "2.5 voxels off at zoom 3 is 7.5 px"
    evs = _drag(h, vm, (20, 18.2, 45), (20, 22.2, 45))
    assert evs[0].handled


def test_button_2_is_never_touched(h):
    vm = h.viewer_model_xy
    evs = _drag(h, vm, (20, 20, 45), (20, 25, 50), button=2)
    assert not any(e.handled for e in evs)
    assert vm.camera.mouse_pan is True
    assert not h.deferred and not h._add_box_overlay.updates


def test_inert_off_the_add_tab(h):
    h.adv._tool_tabs.text = "Refine"
    vm = h.viewer_model_xy
    evs = _drag(h, vm, (20, 20, 45), (20, 25, 50))
    assert not any(e.handled for e in evs)
    h._add_place_markers_on = True
    _drag(h, vm, (20, 30, 45), (20, 30, 45), pos1=(100, 100))
    assert not h.deferred


def test_none_in_review(h):
    h._add_review = True
    evs = _drag(h, h.viewer_model_xy, (20, 20, 45), (20, 25, 50))
    assert not any(e.handled for e in evs)
    assert not h.deferred


def test_busy_refuses_the_drag_with_its_sentence(h):
    """B3 fix pass L3: refused means no drag AND no pan -- every event of
    the refused press is handled."""
    h._add_busy = True
    vm = h.viewer_model_xy
    cam0 = tuple(vm.camera.center)
    evs = _drag(h, vm, (20, 20, 45), (20, 25, 50))
    assert all(e.handled for e in evs)
    assert tuple(vm.camera.center) == cam0, "a refused press does not pan"
    assert vm.camera.mouse_pan is True
    assert h._add_transient == AP.ADD_S1_BUSY_REFUSED
    assert not h.deferred
    assert (h._add_box.cy, h._add_box.cx) == (30, 45)


def test_a_preview_started_mid_drag_ends_the_drag(h):
    vm = h.viewer_model_xy
    _dispatch(vm, "press", _Ev("mouse_press", (20, 20, 45), (0, 0)))
    h._add_busy = True
    _dispatch(vm, "move", _Ev("mouse_move", (20, 25, 45), (0, 9),
                              dragging=True))
    assert vm.camera.mouse_pan is True
    _dispatch(vm, "release", _Ev("mouse_release", (20, 25, 45), (0, 9)))
    assert not h.deferred


def test_the_release_is_deferred_behind_the_drag_token(h):
    vm = h.viewer_model_xy
    _drag(h, vm, (20, 20, 45), (20, 25, 50))
    assert len(h.deferred) == 1
    h._remove_add_side_callbacks()          # e.g. Accept before the turn
    _run_deferred(h)
    assert (h._add_box.cy, h._add_box.cx) == (30, 45)


def test_a_release_after_the_session_ended_is_dropped(h):
    vm = h.viewer_model_xy
    _drag(h, vm, (20, 20, 45), (20, 25, 50))
    h._add_running = False
    _run_deferred(h)
    assert (h._add_box.cy, h._add_box.cx) == (30, 45)


def test_a_drag_to_the_volume_edge_is_clamped(h):
    vm = h.viewer_model_xy
    _drag(h, vm, (20, 20, 45), (20, 20, 500))
    _run_deferred(h)
    b = h._add_box.bounds_dict((40, 60, 80))
    assert b["x1"] == 80 and b["x0"] >= 0


@pytest.mark.parametrize("orient,start,far", [
    ("xy", (20, 20, 45), [(20, 20, 1045), (20, 20, -955),
                          (20, 1020, 45), (20, -980, 45)]),
    ("xz", (30, 10, 45), [(30, 10, 1045), (30, 10, -955),
                          (30, 1010, 45), (30, -990, 45)]),
    ("yz", (45, 30, 30), [(45, 1030, 30), (45, -970, 30),
                          (45, 30, 1030), (45, 30, -970)]),
])
def test_a_far_drag_keeps_the_box_size(h, orient, start, far):
    """B3 fix pass: past either edge, on either displayed axis, the box
    slides back inside at its full size (20 x 20 x 30) -- never truncated."""
    vm = getattr(h, f"viewer_model_{orient}")
    for end in far:
        h.place_box(BOX)
        _drag(h, vm, start, end)
        _run_deferred(h)
        b = h._add_box
        assert b.shape_in((40, 60, 80)) == (20, 20, 30), (orient, end)
        assert (b.sz, b.sy, b.sx) == (20, 20, 30)
        assert h.adv.last("box_size") == (20, 20, 30)


def test_a_nudge_after_a_far_drag_moves_one_voxel(h):
    _drag(h, h.viewer_model_xy, (20, 20, 45), (20, 20, 1045))
    _run_deferred(h)
    b0 = h._add_box.bounds_dict((40, 60, 80))
    assert b0["x1"] == 80 and b0["x0"] == 50
    for i in range(1, 4):
        assert h._add_nudge_box("XY", "Left") is True
        b = h._add_box.bounds_dict((40, 60, 80))
        assert b["x0"] == 50 - i and b["x1"] == 80 - i


def test_a_zero_move_drag_over_a_slice_overflow_does_not_commit(
        monkeypatch):
    """z centre 9, size 6, depth 10: the box overflows on XY's slicing
    axis; an XY drag that never moves a voxel changes nothing."""
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((10, 60, 80), np.int32))
    for o in ("xy", "xz", "yz"):
        setattr(hh, f"viewer_model_{o}", ViewerModel())
        setattr(hh, f"qt_viewer_{o}", _QtV())
    hh.adv._tool_tabs = _Tabs("Add")
    hh.deferred = []
    hh._add_defer = hh.deferred.append
    hh._add_box_overlay = _Overlay()
    hh._add_running = True
    hh._add_box = BoxRec(9, 30, 45, 6, 20, 30)
    hh._add_roi = hh._add_box.bounds_slices((10, 60, 80))
    hh._install_add_side_callbacks()
    box = hh._add_box
    evs = _drag(hh, hh.viewer_model_xy, (7, 20, 45), (7, 20.3, 45.2),
                pos0=(100, 100), pos1=(130, 100))
    assert evs[0].handled, "an outline grab"
    assert hh.deferred == [], "nothing moved: no commit"
    assert hh._add_box == box
    assert (hh._add_box.cz, hh._add_box.sz) == (9, 6)


def test_a_drag_drops_clicks_outside_the_moved_box(h):
    h._add_click_markers = [(20, 22, 35), (20, 38, 58)]
    _drag(h, h.viewer_model_xy, (20, 20, 45), (20, 20, 35))   # -10 x
    _run_deferred(h)
    assert marker_zyx(h) == [(20, 22, 35)]


# --------------------------------------------------------------------------
# the click path
# --------------------------------------------------------------------------

def _click(h, vm, position, pos1=(102, 101), modifiers=()):
    _dispatch(vm, "press", _Ev("mouse_press", position, (100, 100),
                               modifiers=modifiers))
    _dispatch(vm, "release", _Ev("mouse_release", position, pos1,
                                 modifiers=modifiers))


def test_a_click_inside_with_place_markers_on_places_a_marker(h):
    h._add_place_markers_on = True
    _click(h, h.viewer_model_xy, (20, 30, 45))
    assert h._add_click_markers == [], "deferred"
    _run_deferred(h)
    assert marker_zyx(h) == [(20, 30, 45)]


def test_a_click_in_xz_maps_its_axes(h):
    h._add_place_markers_on = True
    _click(h, h.viewer_model_xz, (33, 20, 50))       # (y, z, x)
    _run_deferred(h)
    assert marker_zyx(h) == [(20, 33, 50)]


def test_a_press_moved_more_than_the_threshold_is_a_pan(h):
    h._add_place_markers_on = True
    _click(h, h.viewer_model_xy, (20, 30, 45), pos1=(105, 100))
    assert not h.deferred
    _click(h, h.viewer_model_xy, (20, 30, 45), pos1=(104, 100))
    assert len(h.deferred) == 1


def test_no_click_without_place_markers(h):
    _click(h, h.viewer_model_xy, (20, 30, 45))
    assert not h.deferred


def test_ctrl_click_removes(h):
    h._add_place_markers_on = True
    seen = []
    h._add_marker_click = lambda zyx, ctrl=False: seen.append((zyx, ctrl))
    _click(h, h.viewer_model_xy, (20, 30, 45), modifiers=("Control",))
    _click(h, h.viewer_model_xy, (20, 31, 45))
    _run_deferred(h)
    assert seen == [((20.0, 30.0, 45.0), True), ((20.0, 31.0, 45.0), False)]


def test_a_click_outside_the_box_says_so(h):
    h._add_place_markers_on = True
    _click(h, h.viewer_model_xy, (20, 5, 5))
    _run_deferred(h)
    assert h._add_transient == AP.ADD_S1_CLICK_OUTSIDE


def test_a_click_on_the_outline_with_place_markers_places_a_marker(h):
    """B3 fix pass (item 4): a press in the grab band that releases as a
    click, never moved, is still a marker click when Place markers is on
    (it used to be a silent zero-move drag)."""
    h._add_place_markers_on = True
    _click(h, h.viewer_model_xy, (20, 20, 45))
    _run_deferred(h)
    assert marker_zyx(h) == [(20, 20, 45)]
    assert (h._add_box.cy, h._add_box.cx) == (30, 45), "the box stayed"


def test_a_click_on_the_outline_without_place_markers_does_nothing(h):
    _click(h, h.viewer_model_xy, (20, 20, 45))
    _run_deferred(h)
    assert h._add_click_markers == []
    assert (h._add_box.cy, h._add_box.cx) == (30, 45)


def test_an_outline_drag_with_place_markers_places_no_marker(h):
    h._add_place_markers_on = True
    _drag(h, h.viewer_model_xy, (20, 20, 45), (20, 25, 50))
    _run_deferred(h)
    assert h._add_click_markers == []
    assert (h._add_box.cy, h._add_box.cx) == (35, 50)


def test_the_grab_band_is_capped_when_zoomed_out(h):
    """At zoom 0.6 the band would be 10 voxels; capped at a quarter of the
    box's smaller displayed extent (20 -> 5), a press 7 voxels inside the
    top edge is a marker click, not a grab."""
    h._add_place_markers_on = True
    vm = h.viewer_model_xy
    vm.camera.zoom = 0.6
    _click(h, vm, (20, 27, 45))
    _run_deferred(h)
    assert marker_zyx(h) == [(20, 27, 45)]
    evs = _drag(h, vm, (20, 24, 45), (20, 30, 45))     # 4 voxels: a grab
    assert evs[0].handled


# --------------------------------------------------------------------------
# the hover cursor
# --------------------------------------------------------------------------

def _hover(h, vm, position):
    _dispatch(vm, "move", _Ev("mouse_move", position, (0, 0)))


def test_hover_over_the_outline_sets_the_move_cursor_and_leaving_restores(h):
    from qtpy.QtCore import Qt

    native = h.qt_viewer_xy.canvas.native
    _hover(h, h.viewer_model_xy, (20, 20, 45))
    assert native.cursor == Qt.SizeAllCursor
    _hover(h, h.viewer_model_xy, (20, 30, 45))
    assert native.cursor is None and native.unset == 1
    # the main viewer and the other side views are never touched
    assert h.qt_viewer_xz.canvas.native.cursor is None


def test_hover_is_inert_off_the_add_tab_and_a_tab_change_clears_it(h):
    from qtpy.QtCore import Qt

    native = h.qt_viewer_xy.canvas.native
    _hover(h, h.viewer_model_xy, (20, 20, 45))
    assert native.cursor == Qt.SizeAllCursor
    h.adv._tool_tabs.text = "Refine"
    h._add_clear_side_cursors()               # what the tab change calls
    assert native.cursor is None
    _hover(h, h.viewer_model_xy, (20, 20, 45))
    assert native.cursor is None


def test_remove_clears_the_cursor(h):
    native = h.qt_viewer_yz.canvas.native
    _hover(h, h.viewer_model_yz, (45, 30, 30))
    assert native.cursor is not None
    h._remove_add_side_callbacks()
    assert native.cursor is None


def test_the_box_record_is_a_boxrec(h):
    assert isinstance(h._add_box, BoxRec)


# --------------------------------------------------------------------------
# B3 fix pass: pan truth, other buttons, removal mid-drag, double click,
# zoom and the cursor flag, the dropped release
# --------------------------------------------------------------------------

def test_a_lost_release_does_not_leave_pan_off(h):
    """Focus stolen mid-drag: no release.  The next drag must not read
    ``mouse_pan = False`` as the value to restore."""
    vm = h.viewer_model_xy
    _dispatch(vm, "press", _Ev("mouse_press", (20, 20, 45), (0, 0)))
    _dispatch(vm, "move", _Ev("mouse_move", (20, 22, 45), (0, 9),
                              dragging=True))
    assert vm.camera.mouse_pan is False
    # ... the release never arrives; a new drag starts
    _drag(h, vm, (20, 20, 45), (20, 23, 45))
    assert vm.camera.mouse_pan is True
    assert h._add_drag_state is None


def test_pan_is_restored_from_the_active_layer(h):
    """With an active layer, the restore is that layer's ``mouse_pan``
    (napari's own rule), not True and not the press-time value."""
    from napari.layers import Labels

    vm = h.viewer_model_xy
    layer = Labels(np.zeros((40, 60, 80), np.int32))
    vm.layers.append(layer)
    vm.layers.selection.active = layer
    layer.mode = "paint"
    assert vm.camera.mouse_pan is False
    _drag(h, vm, (20, 20, 45), (20, 23, 45))
    assert vm.camera.mouse_pan is False
    layer.mode = "pan_zoom"
    assert vm.camera.mouse_pan is True
    _drag(h, vm, (20, 20, 45), (20, 23, 45))
    assert vm.camera.mouse_pan is True


def test_a_stale_drag_is_ended_and_the_real_box_redrawn(h):
    renders = []
    h._render_add_box = lambda: renders.append(1)
    vm = h.viewer_model_xy
    _dispatch(vm, "press", _Ev("mouse_press", (20, 20, 45), (0, 0)))
    _dispatch(vm, "press", _Ev("mouse_press", (20, 20, 45), (0, 0)))
    assert renders, "the rectangles went back on the real box"
    assert h._add_drag_state is not None
    # the stale generator is closed and napari no longer holds its event
    assert len(vm._mouse_drag_gen) == 1
    assert len(vm._persisted_mouse_event) == 1
    _dispatch(vm, "release", _Ev("mouse_release", (20, 20, 45), (0, 0)))
    assert len(vm._persisted_mouse_event) == 0
    assert vm.camera.mouse_pan is True


def test_another_buttons_release_ends_the_drag_without_commit(h):
    vm = h.viewer_model_xy
    _dispatch(vm, "press", _Ev("mouse_press", (20, 20, 45), (0, 0)))
    _dispatch(vm, "move", _Ev("mouse_move", (20, 25, 45), (0, 9),
                              dragging=True))
    _dispatch(vm, "press", _Ev("mouse_press", (20, 25, 45), (0, 9),
                               button=2))
    _dispatch(vm, "release", _Ev("mouse_release", (20, 25, 45), (0, 9),
                                 button=2))
    assert not h.deferred, "no early commit"
    _run_deferred(h)
    assert (h._add_box.cy, h._add_box.cx) == (30, 45)
    assert vm.camera.mouse_pan is True
    assert h._add_drag_state is None


def test_remove_mid_drag_closes_the_generator_and_redraws(h):
    renders = []
    h._render_add_box = lambda: renders.append(1)
    vm = h.viewer_model_xy
    _dispatch(vm, "press", _Ev("mouse_press", (20, 20, 45), (0, 0)))
    _dispatch(vm, "move", _Ev("mouse_move", (20, 25, 45), (0, 9),
                              dragging=True))
    assert len(vm._mouse_drag_gen) == 1
    h._remove_add_side_callbacks()
    assert len(vm._mouse_drag_gen) == 0
    assert len(vm._persisted_mouse_event) == 0
    assert renders, "the generator's finally put the real box back"
    assert vm.camera.mouse_pan is True


def _dbl(h, vm, position, modifiers=()):
    return _dispatch(vm, "dbl", _Ev("mouse_double_click", position,
                                    (100, 100), modifiers=modifiers))


def test_a_double_click_with_place_markers_places_a_marker_not_a_zoom(h):
    h._add_place_markers_on = True
    vm = h.viewer_model_xy
    z0 = vm.camera.zoom
    _click(h, vm, (20, 30, 45))
    # a second marker a ball's width away (a click INSIDE an
    # existing marker's ball is refused as the same spot -- next test)
    _dbl(h, vm, (20, 36, 46))
    assert vm.camera.zoom == z0, "no 2x zoom"
    _run_deferred(h)
    assert marker_zyx(h) == [(20, 30, 45), (20, 36, 46)]


def test_a_double_click_on_the_same_spot_is_refused_not_a_zoom(h):
    """The second click of a real double click lands in the
    first marker's ball -- refused, one marker, still no 2x zoom.  Since the
    A5 UX audit the refusal is QUIET inside Qt's
    double-click interval of that marker's placement: no amber
    ADD_S1_CLICK_SAME_SPOT for a gesture the user made on purpose; a slow
    second click (outside the interval) is the ordinary refusal."""
    h._add_place_markers_on = True
    h._add_double_click_s = lambda: 60.0        # well inside the interval
    vm = h.viewer_model_xy
    z0 = vm.camera.zoom
    _click(h, vm, (20, 30, 45))
    _dbl(h, vm, (20, 31, 46))
    assert vm.camera.zoom == z0, "no 2x zoom"
    _run_deferred(h)
    assert marker_zyx(h) == [(20, 30, 45)]
    first = h._add_click_markers[0]["id"]
    assert h._add_transient != AP.ADD_S1_CLICK_SAME_SPOT.format(id=first)
    # outside the interval it is the amber refusal
    h._add_double_click_s = lambda: 0.0
    _dbl(h, vm, (20, 31, 46))
    _run_deferred(h)
    assert marker_zyx(h) == [(20, 30, 45)]
    assert h._add_transient == AP.ADD_S1_CLICK_SAME_SPOT.format(id=first)


def test_a_double_click_without_place_markers_still_zooms(h):
    vm = h.viewer_model_xy
    z0 = vm.camera.zoom
    _dbl(h, vm, (20, 30, 45))
    assert vm.camera.zoom == pytest.approx(2 * z0)
    assert not h.deferred


def test_removal_puts_double_click_to_zoom_back(h):
    for o in ("xy", "xz", "yz"):
        vm = getattr(h, f"viewer_model_{o}")
        assert "double_click_to_zoom" not in _names(
            vm.mouse_double_click_callbacks)
    h._remove_add_side_callbacks()
    for o in ("xy", "xz", "yz"):
        vm = getattr(h, f"viewer_model_{o}")
        assert _names(vm.mouse_double_click_callbacks) == [
            "double_click_to_zoom"]
    h._install_add_side_callbacks()
    h._install_add_side_callbacks()
    assert len(h.viewer_model_xy.mouse_double_click_callbacks) == 1


def test_a_zoom_clears_the_hover_cursor_flag(h):
    from qtpy.QtCore import Qt

    native = h.qt_viewer_xy.canvas.native
    _hover(h, h.viewer_model_xy, (20, 20, 45))
    assert native.cursor == Qt.SizeAllCursor
    h.viewer_model_xy.camera.zoom = 2.5
    assert not h._add_side_cursor.get("XY")
    # the next hover on the outline sets it again (napari reset it)
    _hover(h, h.viewer_model_xy, (20, 20, 45))
    assert h._add_side_cursor.get("XY")
    h._remove_add_side_callbacks()
    h._add_side_cursor["XY"] = True
    h.viewer_model_xy.camera.zoom = 1.5          # disconnected: untouched
    assert h._add_side_cursor["XY"] is True


def test_a_dropped_release_redraws_the_real_box(h):
    renders = []
    h._render_add_box = lambda: renders.append(1)
    _drag(h, h.viewer_model_xy, (20, 20, 45), (20, 25, 50))
    h.adv._tool_tabs.text = "Refine"           # the session moved on
    _run_deferred(h)
    assert (h._add_box.cy, h._add_box.cx) == (30, 45)
    assert renders, "the side rectangles went back on the real box"



# --------------------------------------------------------------------------
# B3 fix pass: the dimmed rectangle off the slice range; the orient labels
# --------------------------------------------------------------------------

def test_the_rectangle_dims_off_the_box_slice_range(h):
    from napari.layers import Labels

    vm = h.viewer_model_xy
    vm.layers.append(Labels(np.zeros((40, 60, 80), np.int32)))
    vm.dims.set_point(0, 35)                 # z 35: outside 10..30
    assert h._add_box_overlay.in_range.get("XY") is False
    vm.dims.set_point(0, 20)
    assert h._add_box_overlay.in_range.get("XY") is True
    vm.dims.set_point(0, 30)                 # half-open: 30 is outside
    assert h._add_box_overlay.in_range.get("XY") is False
    # the drag redraw carries it too
    vm.dims.set_point(0, 20)
    _drag(h, vm, (20, 20, 45), (20, 25, 45))
    assert h._add_box_overlay.in_range.get("XY") is True
    # removal: full colour, and a slice move no longer reaches it
    h._remove_add_side_callbacks()
    assert h._add_box_overlay.in_range.get("XY") is True
    vm.dims.set_point(0, 35)
    assert h._add_box_overlay.in_range.get("XY") is True


def test_the_side_view_orient_labels_are_in_order():
    """Item 15: the install sites pair XY / XZ / YZ with their own views --
    a swap of XZ and YZ must fail here."""
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator as C,
    )

    assert C._ADD_SIDE_VIEWS == (
        ("XY", "viewer_model_xy", "qt_viewer_xy"),
        ("XZ", "viewer_model_xz", "qt_viewer_xz"),
        ("YZ", "viewer_model_yz", "qt_viewer_yz"),
    )


@pytest.mark.parametrize("o", ["xy", "xz", "yz"])
def test_each_installed_callback_carries_its_own_view(h, o):
    vm = getattr(h, f"viewer_model_{o}")
    assert h._add_side_cbs[o.upper()][0] is vm
    assert h._add_side_extra[o.upper()]["vm"] is vm

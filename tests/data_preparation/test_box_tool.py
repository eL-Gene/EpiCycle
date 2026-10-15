"""Qt-free tests for the Step 1 ③ box tool's pure core (`_box_tool.py`).

The interactive 3D parts (Shapes layer, ndisplay, click-to-place, live sliders)
need a real napari viewer and are smoke-tested by the user on WoundT01.  What is
unit-testable — and what the extract path's byte-identical guarantee rests on —
is the pure geometry: the source-shape resolution, the full-Z box construction
(design D1), the box→(4,2)-YX-rectangle mapping that feeds the UNCHANGED extract
path, and the redraw-data assembly.  These are functions of numbers/arrays only.
"""
from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.roi_box import BoxRec
from napari_epicycle.data_preparation._box_tool import (
    _ACTIVE_RGBA,
    _ACTIVE_WIDTH,
    _IDLE_RGBA,
    _IDLE_WIDTH,
    centered_box,
    full_shape_from_source,
    overview_factors,
    place_box,
    rect_for_box,
    redraw_paths,
)
from napari_epicycle.data_preparation.logic import extract_patches_from_shapes


# ---------------------------------------------------------------------------
# full_shape_from_source — drop the channel axis per the resolved order
# ---------------------------------------------------------------------------

class TestFullShapeFromSource:
    def test_3d_is_zyx(self):
        assert full_shape_from_source((143, 1024, 1024), "ZYX") == (143, 1024, 1024)

    def test_4d_czyx_drops_leading_channel(self):
        # (C, Z, Y, X) -> (Z, Y, X)
        assert full_shape_from_source((2, 143, 1024, 1024), "CZYX") == (143, 1024, 1024)

    def test_4d_zcyx_drops_second_channel(self):
        # (Z, C, Y, X) -> (Z, Y, X)
        assert full_shape_from_source((143, 2, 1024, 1024), "ZCYX") == (143, 1024, 1024)

    def test_4d_unknown_order_defaults_channel_first(self):
        assert full_shape_from_source((2, 143, 1024, 1024), None) == (143, 1024, 1024)

    def test_none_and_bad_ndim_return_none(self):
        assert full_shape_from_source(None, "ZYX") is None
        assert full_shape_from_source((1024, 1024), "YX") is None       # 2D
        assert full_shape_from_source((2, 3, 4, 5, 6), "TCZYX") is None  # 5D


# ---------------------------------------------------------------------------
# place_box / centered_box — full-Z (D1)
# ---------------------------------------------------------------------------

class TestFullZInvariant:
    SHAPE = (40, 400, 600)  # Z, Y, X

    def test_place_box_spans_full_z(self):
        box = place_box(self.SHAPE, cy=100, cx=200, sy=80, sx=90)
        assert box.sz == self.SHAPE[0]
        assert box.cz == self.SHAPE[0] // 2
        # XY carried through
        assert (box.cy, box.cx, box.sy, box.sx) == (100, 200, 80, 90)

    def test_bounds_cover_all_z(self):
        box = place_box(self.SHAPE, cy=100, cx=200, sy=80, sx=90)
        b = box.bounds_dict(self.SHAPE)
        assert (b["z0"], b["z1"]) == (0, self.SHAPE[0])

    def test_size_clamped_to_at_least_one(self):
        box = place_box(self.SHAPE, cy=10, cx=10, sy=0, sx=-5)
        assert box.sy >= 1 and box.sx >= 1

    def test_centered_box_is_quarter_field_and_full_z(self):
        box = centered_box(self.SHAPE)
        Z, Y, X = self.SHAPE
        assert (box.cy, box.cx) == (Y // 2, X // 2)
        assert (box.sy, box.sx) == (max(1, Y // 4), max(1, X // 4))
        assert box.sz == Z and box.cz == Z // 2


# ---------------------------------------------------------------------------
# rect_for_box — the (4,2) YX rectangle that feeds the extract path
# ---------------------------------------------------------------------------

class TestRectForBox:
    SHAPE = (20, 200, 300)

    def test_rectangle_matches_clamped_bounds(self):
        box = place_box(self.SHAPE, cy=100, cx=150, sy=40, sx=60)
        b = box.bounds_dict(self.SHAPE)
        rect = rect_for_box(box, self.SHAPE)
        assert rect.shape == (4, 2)
        # min/max over the 4 corners == the box's clamped bounds
        assert int(rect[:, 0].min()) == b["y0"]
        assert int(rect[:, 0].max()) == b["y1"]
        assert int(rect[:, 1].min()) == b["x0"]
        assert int(rect[:, 1].max()) == b["x1"]

    def test_rectangle_winding_is_ccw_corners(self):
        box = place_box(self.SHAPE, cy=100, cx=150, sy=40, sx=60)
        b = box.bounds_dict(self.SHAPE)
        rect = rect_for_box(box, self.SHAPE)
        expected = np.array([[b["y0"], b["x0"]], [b["y0"], b["x1"]],
                             [b["y1"], b["x1"]], [b["y1"], b["x0"]]], dtype=float)
        assert np.array_equal(rect, expected)

    def test_rect_drives_extract_to_the_right_full_z_crop(self):
        """The load-bearing bridge: a box's rectangle, run through the SAME
        extract helper the controller uses, yields exactly the box's XY crop
        across ALL Z (full-Z)."""
        Z, Y, X = self.SHAPE
        vol = np.arange(Z * Y * X, dtype=np.int32).reshape(Z, Y, X)
        box = place_box(self.SHAPE, cy=100, cx=150, sy=40, sx=60)
        b = box.bounds_dict(self.SHAPE)
        rect = rect_for_box(box, self.SHAPE)

        patches = extract_patches_from_shapes(vol, [rect])
        (patch,) = patches.values()
        # Full-Z + the exact XY window.
        assert patch.shape == (Z, b["y1"] - b["y0"], b["x1"] - b["x0"])
        assert np.array_equal(patch, vol[:, b["y0"]:b["y1"], b["x0"]:b["x1"]])


# ---------------------------------------------------------------------------
# redraw_paths — per-shape (data, edge_colors, widths) for the Shapes layer
# ---------------------------------------------------------------------------

class TestRedrawPaths:
    SHAPE = (10, 100, 120)

    def test_empty_gives_three_empty_lists(self):
        data, ecolors, widths = redraw_paths([], None, self.SHAPE)
        assert data == [] and ecolors == [] and widths == []

    def test_none_shape_is_safe(self):
        box = place_box(self.SHAPE, 50, 50, 20, 20)
        data, ecolors, widths = redraw_paths([box], 0, None)
        assert data == [] and ecolors == [] and widths == []

    def test_corners_yields_24_segments_per_box(self):
        # 8 corners x 3 bracket segments = 24 path segments.
        box = place_box(self.SHAPE, 50, 50, 20, 20)
        data, ecolors, widths = redraw_paths([box], 0, self.SHAPE, "corners")
        assert len(data) == 24
        assert len(ecolors) == 24 == len(widths)
        # every segment is a (2, 3) ZYX pair
        assert all(seg.shape == (2, 3) for seg in data)

    def test_wireframe_yields_12_edges_per_box(self):
        box = place_box(self.SHAPE, 50, 50, 20, 20)
        data, _, _ = redraw_paths([box], 0, self.SHAPE, "wireframe")
        assert len(data) == 12

    def test_active_box_is_highlighted(self):
        boxes = [place_box(self.SHAPE, 30, 30, 10, 10),
                 place_box(self.SHAPE, 60, 60, 10, 10)]
        data, ecolors, widths = redraw_paths(boxes, 1, self.SHAPE, "corners")
        # first 24 segments = idle box 0, next 24 = active box 1
        assert ecolors[0] == _IDLE_RGBA and widths[0] == _IDLE_WIDTH
        assert ecolors[24] == _ACTIVE_RGBA and widths[24] == _ACTIVE_WIDTH

    def test_no_active_index_draws_all_idle(self):
        boxes = [place_box(self.SHAPE, 30, 30, 10, 10)]
        _, ecolors, widths = redraw_paths(boxes, None, self.SHAPE, "corners")
        assert set(ecolors) == {_IDLE_RGBA}
        assert set(widths) == {_IDLE_WIDTH}


# ---------------------------------------------------------------------------
# overview_factors — large-source display downsample
# ---------------------------------------------------------------------------

class TestOverviewFactors:
    def test_small_source_not_downsampled(self):
        assert overview_factors((40, 256, 256)) == (1, 1, 1)

    def test_large_xy_strided_z_kept(self):
        fz, fy, fx = overview_factors((60, 4096, 4096), max_xy=512)
        assert fz == 1                # Z always kept (Viz default)
        assert fy == fx and fy >= 8   # 4096 / 512 = 8
        assert 4096 // fy <= 512

    def test_uses_the_larger_of_y_x(self):
        # X is the big axis -> factor sized off it, applied to both Y and X.
        fz, fy, fx = overview_factors((10, 300, 2000), max_xy=500)
        assert fz == 1 and fy == fx == 4


# ---------------------------------------------------------------------------
# Step1BoxTool — the headless (no viewer / no widget) surface: box CRUD +
# the extract bridge.  Everything viewer/Qt-bound is guarded to no-op, so the
# non-visual logic is fully testable without napari.
# ---------------------------------------------------------------------------

class _FakeLazy:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)


class _FakeController:
    """Just the L0 state Step1BoxTool reads (viewer/widget deliberately None)."""

    def __init__(self, shape, order):
        self._source_lazy = _FakeLazy(shape)
        self._source_axes_resolved = order
        self._loaded_source_layers = []
        self.viewer = None
        self.widget = None

    def _log(self, *_a, **_k):
        pass


def _tool(shape=(30, 400, 600), order="CZYX"):
    from napari_epicycle.data_preparation._box_tool import Step1BoxTool
    return Step1BoxTool(_FakeController((2, *shape), order))


class TestBoxToolHeadless:
    def test_add_box_appends_a_centered_full_z_box(self):
        tool = _tool()
        assert not tool.has_boxes()
        idx = tool.add_box()
        assert idx == 0 and tool.has_boxes()
        box = tool.boxes[0]
        assert box.sz == 30 and box.cz == 15          # full-Z (D1)
        assert (box.cy, box.cx) == (200, 300)         # volume centre

    def test_click_to_place_center_is_carried(self):
        tool = _tool()
        tool.add_box(center_yx=(120, 340))
        box = tool.boxes[0]
        assert (box.cy, box.cx) == (120, 340)
        assert box.sz == 30                           # still full-Z

    def test_rectangles_feed_extract_all_current(self):
        tool = _tool()
        tool.add_box(center_yx=(100, 100), size_yx=(40, 50))
        tool.add_box(center_yx=(200, 300), size_yx=(60, 60))
        rects = tool.rectangles()
        assert len(rects) == 2
        assert all(r.shape == (4, 2) for r in rects)
        # matches rect_for_box on the same boxes/shape
        for box, r in zip(tool.boxes, rects):
            assert np.array_equal(r, rect_for_box(box, tool._full_shape))

    def test_delete_active_removes_and_reselects(self):
        tool = _tool()
        tool.add_box(center_yx=(100, 100))
        tool.add_box(center_yx=(200, 200))
        tool.delete_active()
        assert len(tool.boxes) == 1
        assert tool.active_index == 0

    def test_reset_drops_all_boxes(self):
        tool = _tool()
        tool.add_box()
        tool.reset()
        assert not tool.has_boxes() and tool.active_index is None

    def test_clear_all_drops_boxes_without_reset(self):
        # "Clear" must actually remove the box-tool boxes (else Clear looks like
        # a no-op yet wipes the committed-sig guard, re-exposing captured boxes).
        tool = _tool()
        tool.add_box()
        tool.add_box()
        assert len(tool.boxes) == 2
        tool.clear_all()
        assert not tool.has_boxes()
        assert tool.active_index is None
        assert tool.rectangles() == []

    def test_full_shape_resolves_from_zcyx_source(self):
        # (Z, C, Y, X) source -> (Z, Y, X)
        tool = Step1BoxToolFromShape((30, 2, 400, 600), "ZCYX")
        tool.add_box()
        assert tool._full_shape == (30, 400, 600)

    def test_geometry_maxima_helper_calls_widget_setter(self):
        # The box tool hands its resolved shape to the widget so the geometry
        # ceilings track the source (a >4096 source would otherwise clamp+crop).
        tool = _tool(shape=(5000, 400, 600))
        tool._full_shape = (5000, 400, 600)
        seen = {}

        class _W:
            def set_box_geometry_maxima(self, shape):
                seen["shape"] = shape

        tool.widget = _W()
        tool._apply_geometry_spin_maxima()
        assert seen["shape"] == (5000, 400, 600)

    def test_geometry_maxima_helper_is_safe_without_widget_or_shape(self):
        # widget=None (headless) and/or no resolved shape -> silent no-op.
        tool = _tool()
        tool.widget = None
        tool._full_shape = None
        tool._apply_geometry_spin_maxima()      # must not raise


def Step1BoxToolFromShape(shape, order):
    from napari_epicycle.data_preparation._box_tool import Step1BoxTool

    class _C:
        def __init__(self):
            self._source_lazy = _FakeLazy(shape)
            self._source_axes_resolved = order
            self._loaded_source_layers = []
            self.viewer = None
            self.widget = None

        def _log(self, *_a, **_k):
            pass

    return Step1BoxTool(_C())


# ---------------------------------------------------------------------------
# Item B — editable Z (place_box cz/sz + z_ranges bridge to extract)
# ---------------------------------------------------------------------------

class TestEditableZ:
    SHAPE = (40, 400, 600)  # Z, Y, X

    def test_place_box_defaults_full_z(self):
        box = place_box(self.SHAPE, 100, 200, 80, 90)
        assert box.cz == self.SHAPE[0] // 2 and box.sz == self.SHAPE[0]

    def test_place_box_accepts_custom_z(self):
        box = place_box(self.SHAPE, 100, 200, 80, 90, cz=12, sz=8)
        assert box.cz == 12 and box.sz == 8
        assert (box.cy, box.cx, box.sy, box.sx) == (100, 200, 80, 90)

    def test_z_ranges_full_for_default_box(self):
        tool = _tool(shape=self.SHAPE)
        tool.add_box(center_yx=(100, 200))
        # A default box spans all Z -> extract stays byte-identical.
        assert tool.z_ranges() == [[0, self.SHAPE[0]]]

    def test_z_ranges_reflects_narrowed_box(self):
        tool = _tool(shape=self.SHAPE)
        tool.add_box(center_yx=(100, 200))
        tool.boxes[0] = place_box(self.SHAPE, 100, 200, 80, 90, cz=10, sz=8)
        (zr,) = tool.z_ranges()
        b = tool.boxes[0].bounds_dict(self.SHAPE)
        assert zr == [b["z0"], b["z1"]]
        assert zr[1] - zr[0] == 8

    def test_z_ranges_align_to_rectangles_order(self):
        tool = _tool(shape=self.SHAPE)
        tool.add_box(center_yx=(100, 100))
        tool.add_box(center_yx=(300, 400))
        tool.boxes[1] = place_box(self.SHAPE, 300, 400, 40, 40, cz=6, sz=4)
        rects, zrs = tool.rectangles(), tool.z_ranges()
        assert len(rects) == len(zrs) == 2
        assert zrs[0] == [0, self.SHAPE[0]]        # box 0 still full-Z
        assert zrs[1][1] - zrs[1][0] == 4          # box 1 narrowed


# ---------------------------------------------------------------------------
# Item C — one tick per box + list order == extraction order
# ---------------------------------------------------------------------------

class TestBoxCheckedAndReorder:
    def test_add_appends_checked_true(self):
        tool = _tool()
        tool.add_box(center_yx=(10, 10))
        tool.add_box(center_yx=(20, 20))
        assert tool.box_checked == [True, True]

    def test_delete_pops_the_matching_tick(self):
        tool = _tool()
        tool.add_box(center_yx=(10, 10))
        tool.add_box(center_yx=(20, 20))
        tool.box_checked[0] = False
        tool.active_index = 0
        tool.delete_active()
        assert len(tool.boxes) == 1
        assert tool.box_checked == [True]          # row 0 (the unticked) removed

    def test_reorder_permutes_boxes_and_ticks_and_active(self):
        tool = _tool()
        for cy in (10, 20, 30):
            tool.add_box(center_yx=(cy, cy))
        tool.box_checked = [True, False, True]
        b0, b1, b2 = tool.boxes
        tool.active_index = 0
        assert tool._reorder_boxes([2, 0, 1])      # old indices in new order
        assert tool.boxes[0] is b2 and tool.boxes[1] is b0 and tool.boxes[2] is b1
        assert tool.box_checked == [True, True, False]
        assert tool.active_index == 1              # active box b0 moved to row 1

    def test_reorder_checked_override_wins(self):
        tool = _tool()
        tool.add_box(center_yx=(10, 10))
        tool.add_box(center_yx=(20, 20))
        assert tool._reorder_boxes([1, 0], checked=[False, True])
        assert tool.box_checked == [False, True]

    def test_reorder_rejects_invalid_permutation(self):
        tool = _tool()
        tool.add_box(center_yx=(10, 10))
        tool.add_box(center_yx=(20, 20))
        b0, b1 = tool.boxes
        assert tool._reorder_boxes([0, 0]) is False   # not a permutation
        assert tool.boxes[0] is b0 and tool.boxes[1] is b1

    def test_clear_all_resets_ticks(self):
        tool = _tool()
        tool.add_box(center_yx=(10, 10))
        tool.clear_all()
        assert tool.box_checked == []


# ---------------------------------------------------------------------------
# Item A — placement at the current view + sticky S/M/L size
# ---------------------------------------------------------------------------

class _FakeCamera:
    def __init__(self, center):
        self.center = center


class _FakeViewerCam:
    def __init__(self, center):
        self.camera = _FakeCamera(center)


class TestCameraPlacementAndSizePresets:
    SHAPE = (30, 400, 600)

    def _tool_shape(self):
        tool = _tool(shape=self.SHAPE)
        tool._full_shape = self.SHAPE
        return tool

    def test_no_viewer_falls_back_to_volume_centre(self):
        tool = self._tool_shape()
        assert tool.viewer is None
        assert tool._camera_center_yx() == (200, 300)

    def test_camera_target_becomes_clamped_voxel(self):
        tool = self._tool_shape()
        tool.viewer = _FakeViewerCam((5.0, 120.4, 340.6))
        assert tool._camera_center_yx() == (120, 341)

    def test_out_of_range_camera_is_clamped_into_volume(self):
        tool = self._tool_shape()
        tool.viewer = _FakeViewerCam((5.0, 999.0, -50.0))
        assert tool._camera_center_yx() == (399, 0)

    def test_non_finite_camera_falls_back(self):
        tool = self._tool_shape()
        tool.viewer = _FakeViewerCam((5.0, float("nan"), 300.0))
        assert tool._camera_center_yx() == (200, 300)

    def test_add_box_uses_camera_and_sticky_size(self):
        tool = self._tool_shape()
        tool.viewer = _FakeViewerCam((5.0, 120.0, 340.0))
        tool.set_size_fraction(0.5)                # sticky L
        tool.add_box()                             # Add-button path
        box = tool.boxes[0]
        Z, Y, X = self.SHAPE
        assert (box.cy, box.cx) == (120, 340)
        assert (box.sy, box.sx) == (int(Y * 0.5), int(X * 0.5))
        assert box.sz == Z                         # full-Z default

    def test_set_size_fraction_is_sticky_only_not_retroactive(self):
        """Bug fix 2026-07-27: an S/M/L preset sets the size of the NEXT box; it
        must NOT retroactively resize an already-placed box (size is per-box)."""
        tool = self._tool_shape()
        tool.add_box(center_yx=(100, 200))         # M default (0.25)
        before = (tool.boxes[0].sy, tool.boxes[0].sx)
        tool.set_size_fraction(0.125)              # S — sticky for next box only
        assert (tool.boxes[0].sy, tool.boxes[0].sx) == before   # UNCHANGED
        assert tool._last_size_frac == 0.125
        # the NEXT box is born at the new fraction
        tool.add_box(center_yx=(50, 60))
        Z, Y, X = self.SHAPE
        assert (tool.boxes[1].sy, tool.boxes[1].sx) == (int(Y * 0.125), int(X * 0.125))

    def test_click_to_place_reuses_sticky_size(self):
        tool = self._tool_shape()
        tool.set_size_fraction(0.5)
        tool.add_box(center_yx=(50, 60))           # click-to-place path
        box = tool.boxes[0]
        Z, Y, X = self.SHAPE
        assert (box.sy, box.sx) == (int(Y * 0.5), int(X * 0.5))

    def test_bad_fraction_is_ignored(self):
        tool = self._tool_shape()
        tool.set_size_fraction(0.25)
        tool.set_size_fraction(0)                  # out of (0, 1] -> no-op
        tool.set_size_fraction("nope")             # non-numeric -> no-op
        assert tool._last_size_frac == 0.25


# ---------------------------------------------------------------------------
# Item D (v2 / feedback #6) — absolute-voxel size presets (128³/256³/512³ + custom)
# ---------------------------------------------------------------------------

class TestAbsoluteSizePresets:
    SHAPE = (30, 400, 600)  # Z, Y, X

    def _tool_shape(self):
        tool = _tool(shape=self.SHAPE)
        tool._full_shape = self.SHAPE
        return tool

    def test_set_size_voxels_sets_sticky_absolute(self):
        tool = self._tool_shape()
        tool.set_size_voxels(128)
        assert tool._last_abs_size == 128
        assert tool._last_size_mode == "abs"

    def test_add_box_uses_absolute_size_and_full_z(self):
        tool = self._tool_shape()
        tool.set_size_voxels(256)
        tool.add_box(center_yx=(100, 200))
        box = tool.boxes[0]
        Z, _, _ = self.SHAPE
        assert (box.sy, box.sx) == (256, 256)
        assert box.sz == Z                         # full depth by default

    def test_set_size_voxels_is_sticky_only_not_retroactive(self):
        """Bug fix 2026-07-27: an absolute-size preset is sticky for the NEXT box
        only — it must NOT resize (nor Z-crop) an already-placed box."""
        tool = self._tool_shape()
        tool.add_box(center_yx=(100, 200))         # full-Z M box
        before = (tool.boxes[0].cy, tool.boxes[0].cx, tool.boxes[0].sy,
                  tool.boxes[0].sx, tool.boxes[0].cz, tool.boxes[0].sz)
        tool.set_size_voxels(128)                  # sticky for next box only
        box = tool.boxes[0]
        assert (box.cy, box.cx, box.sy, box.sx, box.cz, box.sz) == before  # UNCHANGED
        assert tool._last_abs_size == 128
        # the NEXT box gets the absolute sticky size (full-Z by default)
        tool.add_box(center_yx=(50, 60))
        Z, _, _ = self.SHAPE
        assert (tool.boxes[1].sy, tool.boxes[1].sx) == (128, 128)
        assert tool.boxes[1].sz == Z

    def test_last_chosen_size_model_wins(self):
        tool = self._tool_shape()
        _, Y, X = self.SHAPE
        tool.set_size_voxels(128)                  # abs sticky
        tool.set_size_fraction(0.5)                # frac now wins
        tool.add_box(center_yx=(50, 60))
        assert (tool.boxes[0].sy, tool.boxes[0].sx) == (int(Y * 0.5), int(X * 0.5))
        tool.set_size_voxels(64)                   # abs wins again
        tool.add_box(center_yx=(70, 80))
        assert (tool.boxes[1].sy, tool.boxes[1].sx) == (64, 64)

    def test_bad_voxel_size_ignored(self):
        tool = self._tool_shape()
        tool.set_size_voxels(200)
        tool.set_size_voxels(0)                    # < 1 -> no-op
        tool.set_size_voxels("nope")               # non-numeric -> no-op
        assert tool._last_abs_size == 200


# ---------------------------------------------------------------------------
# Item C (v2 / feedback #2) — placement is a 2D operation
# ---------------------------------------------------------------------------

class _FakeDims:
    def __init__(self, ndisplay):
        self.ndisplay = ndisplay


class _FakeViewerND:
    """A viewer with BOTH dims.ndisplay and camera.center (the 2D/3D-aware
    placement path reads both)."""

    def __init__(self, ndisplay, center):
        self.dims = _FakeDims(ndisplay)
        self.camera = _FakeCamera(center)


class TestPlacementIs2D:
    SHAPE = (30, 400, 600)

    def _tool_shape(self):
        tool = _tool(shape=self.SHAPE)
        tool._full_shape = self.SHAPE
        return tool

    def test_2d_reads_camera_and_records_last_centre(self):
        tool = self._tool_shape()
        tool.viewer = _FakeViewerND(2, (5.0, 120.0, 340.0))
        assert tool._camera_center_yx() == (120, 340)
        assert tool._last_2d_center == (120, 340)

    def test_3d_uses_last_2d_centre_not_orbit_target(self):
        tool = self._tool_shape()
        tool.viewer = _FakeViewerND(2, (5.0, 120.0, 340.0))
        tool._camera_center_yx()                   # records the 2D centre
        # Flip to 3D with a DIFFERENT (meaningless) orbit target.
        tool.viewer = _FakeViewerND(3, (5.0, 10.0, 10.0))
        assert tool._camera_center_yx() == (120, 340)   # last 2D centre wins

    def test_3d_without_prior_2d_falls_back_to_volume_centre(self):
        tool = self._tool_shape()
        tool.viewer = _FakeViewerND(3, (5.0, 10.0, 10.0))
        # The 3D orbit target is NEVER read for placement.
        assert tool._camera_center_yx() == (200, 300)

    def test_add_box_in_3d_places_at_last_2d_centre(self):
        tool = self._tool_shape()
        tool.viewer = _FakeViewerND(2, (5.0, 120.0, 340.0))
        tool._camera_center_yx()                   # record
        tool.viewer = _FakeViewerND(3, (5.0, 10.0, 10.0))
        tool.add_box()                             # Add-button path in 3D
        assert (tool.boxes[0].cy, tool.boxes[0].cx) == (120, 340)


# ---------------------------------------------------------------------------
# Brick A — placement world→voxel division at a PHYSICAL (non-unit) voxel
# ---------------------------------------------------------------------------

class _VoxelCtl:
    """Controller fake exposing `_preview_spacing` (the physical voxel the
    placement funnel divides by) alongside the L0 state the tool reads."""

    def __init__(self, shape, order, voxel):
        self._source_lazy = _FakeLazy((2, *shape))
        self._source_axes_resolved = order
        self._loaded_source_layers = []
        self.viewer = None
        self.widget = None
        self._voxel = voxel

    def _preview_spacing(self):
        return self._voxel

    def _log(self, *_a, **_k):
        pass


def _tool_voxel(shape=(30, 400, 600), order="CZYX", voxel=(1.0, 1.0, 1.0)):
    from napari_epicycle.data_preparation._box_tool import Step1BoxTool
    return Step1BoxTool(_VoxelCtl(shape, order, voxel))


class TestPlacementWorldToVoxelDivision:
    SHAPE = (30, 400, 600)
    VOXEL = (4.0, 2.0, 0.5)  # (vz, vy, vx) — anisotropic, non-unit

    # World µm for the target voxel (cy=120, cx=340): wy = 120*vy = 240,
    # wx = 340*vx = 170.  Placement must recover (120, 340) by DIVIDING.
    WORLD_CENTER = (5.0, 240.0, 170.0)
    TARGET_VOXEL = (120, 340)

    def _tool(self):
        tool = _tool_voxel(shape=self.SHAPE, voxel=self.VOXEL)
        tool._full_shape = self.SHAPE
        return tool

    def test_camera_center_divides_physical_world_by_voxel(self):
        tool = self._tool()
        tool.viewer = _FakeViewerCam(self.WORLD_CENTER)   # nd=None path
        assert tool._camera_center_yx() == self.TARGET_VOXEL

    def test_2d_records_last_centre_in_VOXEL_not_world(self):
        tool = self._tool()
        tool.viewer = _FakeViewerND(2, self.WORLD_CENTER)
        assert tool._camera_center_yx() == self.TARGET_VOXEL
        # `_last_2d_center` must be VOXEL (so a later 3D Add re-uses voxel, not µm).
        assert tool._last_2d_center == self.TARGET_VOXEL
        # Flip to 3D with a meaningless orbit target -> the VOXEL centre wins.
        tool.viewer = _FakeViewerND(3, (5.0, 1.0, 1.0))
        assert tool._camera_center_yx() == self.TARGET_VOXEL

    def test_add_box_via_camera_lands_at_the_divided_voxel(self):
        tool = self._tool()
        tool.viewer = _FakeViewerND(2, self.WORLD_CENTER)
        tool.add_box()                                    # ③ Add / ② Place funnel
        box = tool.boxes[0]
        assert (box.cy, box.cx) == self.TARGET_VOXEL

    def test_click_to_place_divides_cursor_world_by_voxel(self):
        tool = self._tool()
        tool._active = True
        tool._place_enabled = lambda: True                # gate open

        class _Cursor:
            position = (5.0, 240.0, 170.0)                # world µm (z, y, x)

        class _Dims:
            ndisplay = 2

        class _Viewer:
            dims = _Dims()
            cursor = _Cursor()

        class _Ev:
            type = "mouse_press"                          # a click, not a drag

        cb = tool._make_place_callback()
        gen = cb(_Viewer(), _Ev())
        next(gen)                                         # run to the first yield
        with pytest.raises(StopIteration):
            next(gen)                                     # resume -> add_box -> end
        box = tool.boxes[0]
        assert (box.cy, box.cx) == self.TARGET_VOXEL

    def test_unknown_voxel_is_identity_placement(self):
        """`voxel=(1,1,1)` (unknown) → world == voxel, byte-identical to today."""
        tool = _tool_voxel(shape=self.SHAPE, voxel=(1.0, 1.0, 1.0))
        tool._full_shape = self.SHAPE
        tool.viewer = _FakeViewerCam((5.0, 120.4, 340.6))
        assert tool._camera_center_yx() == (120, 341)     # same as pre-physical

    def test_missing_preview_spacing_falls_back_to_unit(self):
        """A controller without `_preview_spacing` (legacy fakes) reads as unit —
        no division, no crash."""
        tool = _tool(shape=self.SHAPE)                    # `_FakeController`, no hook
        tool._full_shape = self.SHAPE
        tool.viewer = _FakeViewerCam((5.0, 120.4, 340.6))
        assert tool._camera_center_yx() == (120, 341)

    def test_extraction_rectangles_are_voxel_and_voxel_independent(self):
        """Does-NOT-change guard: the extract rectangles read box DATA (voxel),
        so they are IDENTICAL at a unit vs a non-unit voxel — the physical scale is
        display-only and never reaches extraction."""
        def rects_for(voxel):
            tool = _tool_voxel(shape=self.SHAPE, voxel=voxel)
            tool._full_shape = self.SHAPE
            tool.add_box(center_yx=(120, 340), size_yx=(40, 50))  # VOXEL center
            return tool.rectangles()

        r_unit = rects_for((1.0, 1.0, 1.0))
        r_aniso = rects_for(self.VOXEL)
        assert len(r_unit) == len(r_aniso) == 1
        assert np.array_equal(r_unit[0], r_aniso[0])


# ---------------------------------------------------------------------------
# Part C (2026-07-27) — ④'s Render/style controls drive the box overlay
# ---------------------------------------------------------------------------

class _FakeOverlay:
    def __init__(self):
        self.style_kwargs = None
    def set_style(self, **kw):
        self.style_kwargs = kw
    def update(self, *_a):
        pass
    def update_ghost(self, *_a):
        pass


class TestBoxStyleWiring:
    def test_set_box_style_stores_and_pushes_to_overlay(self):
        from napari_epicycle.data_preparation._box_tool import _BOX_COLOR_CHOICES
        tool = _tool()
        tool._full_shape = (30, 400, 600)
        tool._overlay = _FakeOverlay()
        tool.set_box_style(thickness=8, opacity=0.4, color_name="green")
        assert tool._style_thickness == 8
        assert tool._style_opacity == 0.4
        assert tool._style_color == _BOX_COLOR_CHOICES["green"]
        # pushed through to the overlay
        assert tool._overlay.style_kwargs["thickness"] == 8
        assert tool._overlay.style_kwargs["opacity"] == 0.4
        assert tool._overlay.style_kwargs["color"] == _BOX_COLOR_CHOICES["green"]

    def test_status_colour_maps_to_none(self):
        tool = _tool()
        tool._overlay = _FakeOverlay()
        tool.set_box_style(color_name="status colour")
        assert tool._style_color is None
        assert tool._overlay.style_kwargs["color"] is None

    def test_set_box_style_no_op_without_overlay_but_remembers(self):
        tool = _tool()
        assert tool._overlay is None
        tool.set_box_style(thickness=5, opacity=0.6)   # must not raise
        assert tool._style_thickness == 5 and tool._style_opacity == 0.6

    def test_ensure_overlay_reapplies_stored_style(self, monkeypatch):
        # A fresh overlay (box-mode re-toggle) must inherit the remembered look.
        from napari_epicycle.data_preparation._box_tool import _BOX_COLOR_CHOICES
        import napari_epicycle.data_preparation._box_overlay as ov_mod
        tool = _tool()
        tool.set_box_style(thickness=9, opacity=0.3, color_name="magenta")
        captured = {}

        class _Ov:
            def set_style(self, **kw):
                captured.update(kw)

        monkeypatch.setattr(ov_mod, "Step1BoxOverlay", lambda *_a, **_k: _Ov())
        monkeypatch.setattr(tool, "_main_qt_viewer", lambda: object())
        tool._ensure_overlay()
        assert captured.get("thickness") == 9
        assert captured.get("opacity") == 0.3
        assert captured.get("color") == _BOX_COLOR_CHOICES["magenta"]


# ---------------------------------------------------------------------------
# Persist-boxes (2026-08-17) — box mode OFF keeps the placed boxes EDITABLE
#
# The ③ "📦 Box mode" toggle-OFF keeps the placed boxes on the canvas AND still
# selectable / movable — the geometry + navigator-list handlers stay connected
# (they don't gate on `_active`), so only the ghost placement-guide + click-to-
# place-new go away.  stop(keep_visible=True) must NOT destroy the overlay or
# disconnect geometry/list when boxes exist.  The source-lifecycle callers
# (step-exit, reset, app-quit) keep the default keep_visible=False → full
# teardown (overlay + connections destroyed) so nothing strands on another
# step's shared canvas.
# ---------------------------------------------------------------------------

class _LifecycleOverlay:
    def __init__(self):
        self.cleared = 0
        self.updates = []
        self.destroyed = 0
    def clear_ghost(self):
        self.cleared += 1
    def update(self, boxes, active, shape):
        self.updates.append((list(boxes), active, shape))
    def update_ghost(self, *_a):
        pass
    def destroy(self):
        self.destroyed += 1


class TestStopKeepVisible:
    def test_toggle_off_keeps_overlay_editable_and_connected(self):
        tool = _tool()
        tool.add_box(center_yx=(100, 100))      # active_index -> 0
        tool._full_shape = (30, 400, 600)
        tool._geom_connected = True
        tool._list_connected = True
        ov = _LifecycleOverlay()
        tool._overlay = ov
        tool.stop(keep_visible=True)
        assert tool._overlay is ov              # overlay NOT destroyed
        assert ov.destroyed == 0
        assert ov.cleared == 1                  # ghost cleared
        assert ov.updates and ov.updates[-1][1] == 0  # active-box highlight kept
        assert not tool.is_active()             # box mode is off
        assert len(tool.boxes) == 1             # boxes kept
        # editing stays wired so select/move works with box mode off
        assert tool._geom_connected is True
        assert tool._list_connected is True

    def test_toggle_off_with_no_boxes_destroys_overlay(self):
        tool = _tool()
        ov = _LifecycleOverlay()
        tool._overlay = ov
        tool.stop(keep_visible=True)            # nothing to keep -> full teardown
        assert tool._overlay is None
        assert ov.destroyed == 1


class TestBoxDimText:
    """C1 — per-row Z×Y×X reads from bounds_dict (matches the extracted extent)."""

    def test_dim_text_matches_bounds_dict(self):
        tool = _tool()
        tool.add_box(center_yx=(100, 100), size_yx=(40, 50))
        tool._full_shape = (30, 400, 600)
        b = tool.boxes[0].bounds_dict(tool._full_shape)
        expected = (f"{b['z1']-b['z0']}×{b['y1']-b['y0']}×{b['x1']-b['x0']}")
        assert tool._box_dim_text(0) == expected

    def test_dim_text_full_z_default(self):
        tool = _tool()
        tool.add_box(center_yx=(200, 300), size_yx=(60, 60))  # full-Z default
        tool._full_shape = (30, 400, 600)
        assert tool._box_dim_text(0).startswith("30×")        # sz == Z == 30

    def test_dim_text_empty_when_out_of_range(self):
        tool = _tool()
        tool._full_shape = (30, 400, 600)
        assert tool._box_dim_text(0) == ""      # no boxes
        assert tool._box_dim_text(-1) == ""


class TestExtractFromProcessed:
    """P1 — extract on the Processed view no longer HARD-blocks; it routes to
    the processed store via `_resolve_extract_source` -> `_crop_source_for_processed`,
    while the Raw view stays on the raw crop source (byte-identical)."""

    def test_processed_view_routes_to_processed_source(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController,
        )

        class _Handle:
            shape = (1, 4, 8, 8)

        ctl = DataPreparationController.__new__(DataPreparationController)
        ctl._is_processed_view_showing = lambda: True
        seen = {"processed": False, "raw": False}
        ctl._crop_source_for_processed = (
            lambda names: seen.__setitem__("processed", True) or (_Handle(), [0]))
        ctl._crop_source_for = (
            lambda names: seen.__setitem__("raw", True) or (None, None))

        crop, idx, shape, image_data = ctl._resolve_extract_source(["c0"])
        assert seen["processed"] is True     # processed store read…
        assert seen["raw"] is False          # …raw handle NOT touched
        assert image_data is None            # no whole-volume array on this path
        assert idx == [0]
        assert shape == (1, 4, 8, 8)

    def test_raw_view_uses_raw_crop_source(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController,
        )

        class _Handle:
            shape = (2, 4, 8, 8)

        ctl = DataPreparationController.__new__(DataPreparationController)
        ctl._is_processed_view_showing = lambda: False
        seen = {"processed": False, "raw": False}
        ctl._crop_source_for_processed = (
            lambda names: seen.__setitem__("processed", True) or (None, None))
        ctl._crop_source_for = (
            lambda names: seen.__setitem__("raw", True) or (_Handle(), [0, 1]))

        crop, idx, shape, image_data = ctl._resolve_extract_source(["a", "b"])
        assert seen["raw"] is True           # raw crop source used…
        assert seen["processed"] is False    # …processed branch NOT entered on raw
        assert idx == [0, 1]
        assert image_data is None

    def test_extract_on_raw_view_does_not_early_return(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController,
        )
        ctl = DataPreparationController.__new__(DataPreparationController)
        ctl._step1_busy = False
        ctl._is_processed_view_showing = lambda: False
        reached = {"past_guard": False}

        # The migration-timer `.stop()` is the first statement AFTER the processed
        # guard — bail there to prove raw view got past the guard, without
        # entering the heavy extract path.
        class _Timer:
            def stop(_self):
                reached["past_guard"] = True
                raise RuntimeError("bail after guard")

        ctl._migration_offer_timer = _Timer()
        try:
            ctl.extract_patches()
        except RuntimeError:
            pass
        assert reached["past_guard"] is True    # raw view is NOT blocked


class TestDisplayLayersPresentProcessed:
    """Entering box mode while viewing Processed must NOT build a downsampled
    overview on top (the "4 layers" bug: 2 processed + 2 overview).  The
    processed view counts as the source display even though the clean swap
    removed the raw layers."""

    def test_processed_view_counts_as_display_present(self):
        tool = _tool()
        tool._ctl._loaded_source_layers = []       # raw removed by the clean swap
        tool._ctl._is_processed_view_showing = lambda: True
        assert tool._display_layers_present() is True

    def test_no_raw_no_processed_is_not_present(self):
        tool = _tool()
        tool._ctl._loaded_source_layers = []
        tool._ctl._is_processed_view_showing = lambda: False
        assert tool._display_layers_present() is False

    def test_default_stop_destroys_overlay_and_disconnects(self):
        tool = _tool()
        tool.add_box(center_yx=(100, 100))
        tool._full_shape = (30, 400, 600)
        tool._geom_connected = True
        tool._list_connected = True
        ov = _LifecycleOverlay()
        tool._overlay = ov
        tool.stop()                             # keep_visible=False (lifecycle path)
        assert tool._overlay is None
        assert ov.destroyed == 1
        assert len(tool.boxes) == 1             # box definitions still in memory
        # full teardown drops the editing connections
        assert tool._geom_connected is False
        assert tool._list_connected is False


# ---------------------------------------------------------------------------
# Brick 1 (2026-08-17) — the on_props_change handle-match re-seed
#
# The live geometry edit re-seeds the stored box (and the spins) to the RENDERED
# (clamped) box so the slider handle can't lag the drawn box near an edge.  Two
# edge modes (`_edge_mode`): "shrink" (default, byte-identical extraction) and
# "keep" (preserve size, slide inward — changes edge extraction).  Both must be
# idempotent (no creep).  Driven through the real `on_props_change` with a
# Qt-free fake widget of spins.
# ---------------------------------------------------------------------------

class _FakeSpin:
    """The slice of QSpinBox `on_props_change` + `_set_spin` touch."""

    def __init__(self, value=0, lo=0, hi=100000):
        self._v = int(value)
        self._lo = int(lo)
        self._hi = int(hi)

    def value(self):
        return self._v

    def setValue(self, v):
        self._v = int(v)

    def minimum(self):
        return self._lo

    def maximum(self):
        return self._hi

    def setEnabled(self, _enabled):
        pass


class _FakeGeomWidget:
    """The six geometry spins `on_props_change` reads / `_load_geometry_into_spins`
    writes (X/Y/Z centre + W/H/D size).  No lock checkbox: the lock path reads
    `tool.box_locked`, not the widget."""

    def __init__(self, *, cx, cy, cz, sx, sy, sz, hi=100000):
        self.box_center_x_spin = _FakeSpin(cx, hi=hi)
        self.box_center_y_spin = _FakeSpin(cy, hi=hi)
        self.box_center_z_spin = _FakeSpin(cz, hi=hi)
        self.box_size_w_spin = _FakeSpin(sx, hi=hi)   # W = X
        self.box_size_h_spin = _FakeSpin(sy, hi=hi)   # H = Y
        self.box_size_d_spin = _FakeSpin(sz, hi=hi)   # D = Z


class TestEdgeModeAndReseed:
    SHAPE = (50, 512, 512)          # Z, Y, X — X overflows for a high-edge box

    def _tool_with_high_edge_box(self, edge_mode="shrink"):
        # A box whose X bounds truncate at the HIGH edge: cx=450, sx=200 on a
        # 512 axis → X bounds [350, 512).  (Z spans the full 50 depth.)
        tool = _tool(shape=self.SHAPE)
        tool._full_shape = self.SHAPE
        tool._edge_mode = edge_mode
        tool.boxes = [place_box(self.SHAPE, cy=256, cx=450, sy=200, sx=200)]
        tool.box_checked = [True]
        tool.box_locked = [False]
        tool.active_index = 0
        tool.widget = _FakeGeomWidget(cx=450, cy=256, cz=25, sx=200, sy=200, sz=50)
        return tool

    # -- set_edge_mode ------------------------------------------------------

    def test_default_edge_mode_is_shrink(self):
        assert _tool()._edge_mode == "shrink"

    def test_set_edge_mode_validates(self):
        tool = _tool()
        tool.set_edge_mode("keep")
        assert tool._edge_mode == "keep"
        tool.set_edge_mode("bogus")            # invalid -> ignored
        assert tool._edge_mode == "keep"
        tool.set_edge_mode("shrink")
        assert tool._edge_mode == "shrink"

    # -- guard --------------------------------------------------------------

    def test_on_props_change_without_full_shape_is_safe(self):
        tool = _tool()
        tool._full_shape = None
        tool.widget = _FakeGeomWidget(cx=10, cy=10, cz=5, sx=8, sy=8, sz=4)
        tool.boxes = [BoxRec(5, 5, 5, 4, 4, 4)]
        tool.box_locked = [False]
        tool.active_index = 0
        tool.on_props_change()                 # must NOT raise
        assert tool.boxes[0] == BoxRec(5, 5, 5, 4, 4, 4)   # untouched

    # -- shrink (default) ---------------------------------------------------

    def test_shrink_reseed_no_creep_across_N_passes(self):
        # N repeated on_props_change passes on a high-edge box must not drift
        # the rendered/extracted bounds (the no-creep guarantee).
        tool = self._tool_with_high_edge_box("shrink")
        tool.on_props_change()
        first = tool.boxes[0].bounds_dict(self.SHAPE)
        for _ in range(5):
            tool.on_props_change()
            assert tool.boxes[0].bounds_dict(self.SHAPE) == first

    def test_shrink_extraction_byte_identical_to_unfixed(self):
        # bounds_dict after the fix == what today's un-fixed place_box produced,
        # because extraction already reads bounds_dict (render truncated at edge).
        tool = self._tool_with_high_edge_box("shrink")
        unfixed = place_box(self.SHAPE, cy=256, cx=450, sy=200, sx=200,
                            cz=25, sz=50)
        tool.on_props_change()
        assert tool.boxes[0].bounds_dict(self.SHAPE) == unfixed.bounds_dict(self.SHAPE)
        # And the actual extract inputs (rectangles + z_ranges) are unchanged.
        assert np.array_equal(tool.rectangles()[0], rect_for_box(unfixed, self.SHAPE))
        b = unfixed.bounds_dict(self.SHAPE)
        assert tool.z_ranges()[0] == [b["z0"], b["z1"]]

    def test_shrink_reseeds_spins_to_rendered_box(self):
        # After the edit the spins show the rendered box (handle no longer lags).
        tool = self._tool_with_high_edge_box("shrink")
        tool.on_props_change()
        b = tool.boxes[0].bounds_dict(self.SHAPE)
        w = tool.widget
        assert w.box_center_x_spin.value() == (b["x0"] + b["x1"]) // 2
        assert w.box_size_w_spin.value() == b["x1"] - b["x0"]     # X shrank to fit

    # -- keep (opt-in) ------------------------------------------------------

    def test_keep_preserves_size_and_is_idempotent(self):
        tool = self._tool_with_high_edge_box("keep")
        tool.on_props_change()
        box_keep = tool.boxes[0]
        assert box_keep.sx == 200                     # requested X size preserved
        b_keep = box_keep.bounds_dict(self.SHAPE)
        assert b_keep["x1"] - b_keep["x0"] == 200     # full extent, slid inward
        # Idempotent: a second pass (reading the reseeded spins) is a no-op.
        tool.on_props_change()
        assert tool.boxes[0].bounds_dict(self.SHAPE) == b_keep

    def test_keep_bounds_differ_from_shrink(self):
        # Documents the intended edge-extraction difference between the modes.
        keep = self._tool_with_high_edge_box("keep")
        keep.on_props_change()
        shrink = self._tool_with_high_edge_box("shrink")
        shrink.on_props_change()
        assert (keep.boxes[0].bounds_dict(self.SHAPE)
                != shrink.boxes[0].bounds_dict(self.SHAPE))


# ---------------------------------------------------------------------------
# Brick 2 (2026-08-17) — box mode no longer FORCES 3D (Change 1, option b)
#
# `start()` used to call `_set_ndisplay(3)`; the progressive-loading engine then
# re-derived the image's world placement per detail level on zoom-out, making
# the box appear to drift off the signal.  The fix DROPS the forced set entirely
# (not a substitute `_set_ndisplay(2)` — that would double-swap per box cycle),
# so the box renders in whatever view is current, like the ② Preview box.  The
# `_prev_ndisplay` snapshot + `stop()` restore stay in place as genuine no-ops.
# ---------------------------------------------------------------------------

class _RecordingDims:
    """A `dims` stub that records every ndisplay WRITE (a forced view change),
    while a plain read of the property records nothing."""

    def __init__(self, ndisplay=2):
        self._nd = int(ndisplay)
        self.set_calls = []

    @property
    def ndisplay(self):
        return self._nd

    @ndisplay.setter
    def ndisplay(self, v):
        self._nd = int(v)
        self.set_calls.append(int(v))


class _StartFakeViewer:
    """Just enough of a napari viewer for `Step1BoxTool.start()` to run
    headless: a recording `dims`, a camera with NO `events` (so
    `_connect_camera` no-ops), a mouse-callback list, and `window=None` (so the
    overlay build resolves no QtViewer and no-ops).  The point of the fake is to
    observe ndisplay writes, not to render anything."""

    def __init__(self, ndisplay=2):
        self.dims = _RecordingDims(ndisplay)
        self.camera = _FakeCamera((5.0, 100.0, 100.0))
        self.mouse_drag_callbacks = []
        self.window = None


class TestStartDoesNotForceNdisplay:
    SHAPE = (30, 400, 600)

    def _tool_with_viewer(self, ndisplay):
        tool = _tool(shape=self.SHAPE)
        viewer = _StartFakeViewer(ndisplay)
        tool.viewer = viewer
        # No in-RAM display layers in the fake -> skip the large-source overview
        # build (which would read the lazy source); box mode entry is the SUT.
        tool._display_layers_present = lambda: True
        return tool, viewer

    def test_start_from_2d_leaves_the_view_untouched(self):
        tool, viewer = self._tool_with_viewer(2)
        assert tool.start() is True
        assert tool.is_active()
        assert viewer.dims.set_calls == []        # was `_set_ndisplay(3)` — gone
        assert viewer.dims.ndisplay == 2          # stays in the current 2D view

    def test_start_from_3d_does_not_force_2d_either(self):
        # Option (b): start() drops the forced set ENTIRELY — it must NOT
        # substitute `_set_ndisplay(2)`; a user already in 3D stays in 3D.
        tool, viewer = self._tool_with_viewer(3)
        assert tool.start() is True
        assert viewer.dims.set_calls == []
        assert viewer.dims.ndisplay == 3

    def test_prev_ndisplay_snapshot_is_taken_but_only_read(self):
        # The snapshot survives (kept for a minimal/reversible diff) — start()
        # reads dims.ndisplay to record it, but never writes it.
        tool, viewer = self._tool_with_viewer(2)
        tool.start()
        assert tool._prev_ndisplay == 2
        assert viewer.dims.set_calls == []

    def test_stop_restore_is_a_no_op_equal_valued_write(self):
        # `stop()` still restores `_prev_ndisplay`, but since start() never
        # changed the view the restore is a write of the SAME value (harmless).
        tool, viewer = self._tool_with_viewer(2)
        tool.start()
        viewer.dims.set_calls.clear()
        tool.stop()
        assert all(v == 2 for v in viewer.dims.set_calls)   # only equal-valued
        assert viewer.dims.ndisplay == 2
        assert not tool.is_active()
        assert tool._prev_ndisplay is None


# ---------------------------------------------------------------------------
# Brick 2 — the "⌖ Click-to-place" toggle forces 2D (Change 1, click-to-place
# fix).  Click-to-place is 2D-gated (`_make_place_callback` no-ops unless
# ndisplay == 2); with box mode no longer forcing 3D, ticking the checkbox while
# in a 3D inspect look must flip the canvas back to 2D so the click isn't inert.
# Exercised through the REAL controller method on a `__new__`-built instance.
# ---------------------------------------------------------------------------

class _FakePlaceBoxTool:
    def __init__(self, active=False):
        self._active = active
        self.start_calls = 0
        self.nd_calls = []

    def is_active(self):
        return self._active

    def start(self):
        self._active = True
        self.start_calls += 1
        return True

    def _set_ndisplay(self, nd):
        self.nd_calls.append(int(nd))


def _place_controller(box_tool):
    from napari_epicycle.data_preparation.controller import (
        DataPreparationController,
    )
    ctl = DataPreparationController.__new__(DataPreparationController)
    ctl.box_tool = box_tool
    ctl._teardown_calls = 0

    # `_arbitrate_for_box_mode` now calls `_teardown_preview(drop_processed=False)`
    # (region-only), so the stub must accept the kwarg; the counter still tracks
    # that the mode-arbitration teardown ran.  The rest of `_arbitrate_for_box_mode`
    # (`_is_processed_view_showing`, `_log`) is real + safe on this bare controller
    # (no processed layers → shows raw path skipped; `_log` prints when no widget).
    def _fake_teardown(drop_processed=True):
        ctl._teardown_calls += 1

    ctl._teardown_preview = _fake_teardown
    return ctl


class TestOnPlaceToggledForces2D:
    def test_toggle_on_when_inactive_starts_and_forces_2d(self):
        bt = _FakePlaceBoxTool(active=False)
        ctl = _place_controller(bt)
        ctl.on_place_toggled(True)
        assert bt.start_calls == 1
        assert ctl._teardown_calls == 1           # mode arbitration ran
        assert bt.nd_calls == [2]                  # forced to the flat 2D view

    def test_toggle_on_when_already_active_only_forces_2d(self):
        bt = _FakePlaceBoxTool(active=True)
        ctl = _place_controller(bt)
        ctl.on_place_toggled(True)
        assert bt.start_calls == 0                 # already running — not restarted
        assert ctl._teardown_calls == 0
        assert bt.nd_calls == [2]                  # still flipped to 2D

    def test_toggle_off_does_not_force_2d(self):
        bt = _FakePlaceBoxTool(active=True)
        ctl = _place_controller(bt)
        ctl.on_place_toggled(False)
        assert bt.nd_calls == []                   # toggle-OFF leaves the view alone

    def test_no_box_tool_is_safe(self):
        ctl = _place_controller(None)
        ctl.on_place_toggled(True)                 # must not raise


class _FakeToggleBoxTool:
    """Records the kwargs `on_box_mode_toggled` passes to stop()/start()."""
    def __init__(self, active=True, boxes=1):
        self._active = active
        self.stop_calls = []
        self.start_calls = 0
        self.boxes = [object()] * boxes
    def is_active(self):
        return self._active
    def start(self):
        self._active = True
        self.start_calls += 1
        return True
    def stop(self, restore_ndisplay=True, keep_visible=False):
        self.stop_calls.append(keep_visible)
        self._active = False
    def _sync_mode_button(self, _v):
        pass


class TestBoxModeToggleOffKeepsVisible:
    def test_toggle_off_keeps_boxes_visible(self):
        bt = _FakeToggleBoxTool(active=True, boxes=3)
        ctl = _place_controller(bt)
        ctl.on_box_mode_toggled(False)
        assert bt.stop_calls == [True]             # keep_visible=True on toggle-OFF

    def test_toggle_on_does_not_stop(self):
        bt = _FakeToggleBoxTool(active=False, boxes=0)
        ctl = _place_controller(bt)
        ctl.on_box_mode_toggled(True)              # arbitration + start, no stop()
        assert bt.stop_calls == []
        assert bt.start_calls == 1


# ---------------------------------------------------------------------------
# F3 — load_committed_boxes: restore exported boxes into the box tool
# ---------------------------------------------------------------------------

class TestLoadCommittedBoxes:
    """Restoring already-exported patches into the CRASH-SAFE box tool (F3)
    instead of the legacy napari Shapes layer."""

    def _recs(self, tool):
        fs = tool._resolve_full_shape()
        return [
            place_box(fs, cy=100, cx=200, sy=40, sx=50, cz=10, sz=8),
            place_box(fs, cy=250, cx=300, sy=60, sx=60, cz=15, sz=6),
        ]

    def test_appends_as_committed_locked_and_checked(self):
        tool = _tool()
        recs = self._recs(tool)
        n = tool.load_committed_boxes(recs)
        assert n == 2
        assert tool.boxes == recs
        # box_locked doubles as the "exported/committed" visual (frozen), and a
        # committed box reads as included.
        assert tool.box_locked == [True, True]
        assert tool.box_checked == [True, True]
        # full (Z, Y, X) resolved so rectangles()/z_ranges()/dim text all work.
        assert tool._full_shape == (30, 400, 600)

    def test_restore_renders_on_load_when_overlay_available(self, monkeypatch):
        # The restored boxes must SHOW on load (box mode off) — like the old
        # Shapes layer did — not only once box mode opens.  With a live viewer +
        # overlay, load_committed_boxes builds the overlay + redraws.
        tool = _tool()
        tool.viewer = object()                     # non-None gates the render path
        ov = _LifecycleOverlay()

        def _fake_ensure():                        # mirror the real one: sets _overlay
            tool._overlay = ov
            return ov

        monkeypatch.setattr(tool, "_ensure_overlay", _fake_ensure)
        monkeypatch.setattr(tool, "_connect_geometry", lambda: None)
        monkeypatch.setattr(tool, "_connect_list", lambda: None)
        tool.load_committed_boxes(self._recs(tool))
        assert ov.updates                          # redraw() painted the boxes…
        assert len(ov.updates[-1][0]) == 2         # …both restored boxes

    def test_restore_without_viewer_is_a_safe_noop_render(self):
        # No viewer -> no overlay build, no crash; start() renders later.
        tool = _tool()
        tool.viewer = None
        assert tool.load_committed_boxes(self._recs(tool)) == 2

    def test_idempotent_reload_no_duplicates(self):
        tool = _tool()
        recs = self._recs(tool)
        tool.load_committed_boxes(recs)
        tool.load_committed_boxes(recs)      # re-restore of the same source
        assert len(tool.boxes) == 2
        assert tool.box_locked == [True, True]
        assert tool.box_checked == [True, True]

    def test_reload_replaces_prior_committed_set(self):
        tool = _tool()
        tool.load_committed_boxes(self._recs(tool))
        new = [place_box(tool._full_shape, cy=50, cx=50, sy=20, sx=20, cz=5, sz=4)]
        tool.load_committed_boxes(new)       # a different exported set on reload
        assert len(tool.boxes) == 1
        assert tool.boxes == new

    def test_user_drawn_unlocked_boxes_survive_reload(self):
        tool = _tool()
        tool.load_committed_boxes(self._recs(tool))
        # A user-drawn (unlocked) box appended after restore must NOT be dropped
        # by the idempotent replace (only the committed/locked rows are).
        tool.boxes.append(place_box(tool._full_shape, cy=10, cx=10, sy=8, sx=8))
        tool.box_checked.append(True)
        tool.box_locked.append(False)
        tool.load_committed_boxes(self._recs(tool))
        assert len(tool.boxes) == 3          # 1 user + 2 restored
        assert tool.box_locked.count(False) == 1
        assert tool.box_locked.count(True) == 2

    def test_committed_boxes_are_deduped_by_select_uncommitted(self):
        from napari_epicycle.data_preparation.logic import (
            box_signature, select_uncommitted_boxes)
        tool = _tool()
        tool.load_committed_boxes(self._recs(tool))
        rects = tool.rectangles()
        zrs = tool.z_ranges()
        committed = set()
        for r, zr in zip(rects, zrs):
            committed.add(box_signature(r))
            committed.add(box_signature(r, zr))
        # Every restored box's signature is committed -> nothing to re-extract.
        assert select_uncommitted_boxes(rects, committed, zrs) == []

    def test_empty_input_is_safe(self):
        tool = _tool()
        assert tool.load_committed_boxes([]) == 0
        assert not tool.has_boxes()
        assert tool.load_committed_boxes(None) == 0

    def test_reset_clears_committed_boxes(self):
        tool = _tool()
        tool.load_committed_boxes(self._recs(tool))
        tool.reset()
        assert not tool.has_boxes()
        assert tool.box_locked == [] and tool.box_checked == []

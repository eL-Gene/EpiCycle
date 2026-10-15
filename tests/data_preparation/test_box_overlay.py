"""Qt/vispy-free tests for the Step 1 ③ crash-safe box overlay (`_box_overlay`).

The real vispy ``Line`` visuals + the ``glDrawArrays`` access violation they
fix can only be exercised in a live napari session (user smoke on WoundT01).
What IS headlessly testable — and what this covers — is the pure segment
assembly (the ZYX→XYZ, scale=1/translate=0 geometry the extract-visible box
rests on) and the overlay's idle/active/ghost wiring via injected fake lines.
"""
from __future__ import annotations

import numpy as np

from napari_epicycle.data_preparation._box_overlay import (
    Step1BoxOverlay,
    _box_segments_zyx,
    boxes_segments_xyz,
    boxes_segments_and_colors_xyz,
)
from napari_epicycle.data_preparation._box_tool import place_box

SHAPE = (10, 100, 120)  # Z, Y, X
_DEF = (0.5, 0.5, 0.5, 0.9)   # a default colour for the per-box tests


# ---------------------------------------------------------------------------
# Per-box segments + colours (P2 Brick 2 — one Line, per-vertex RGBA)
# ---------------------------------------------------------------------------

class TestPerBoxSegmentsAndColors:
    def test_default_styles_match_boxes_segments_xyz(self):
        # None styles → every box "corners" at the default colour: the SEGMENTS
        # must equal the single-style path (byte-identical geometry).
        boxes = [place_box(SHAPE, 30, 30, 10, 10), place_box(SHAPE, 60, 60, 10, 10)]
        seg, colors = boxes_segments_and_colors_xyz(boxes, SHAPE, None, _DEF)
        assert np.allclose(seg, boxes_segments_xyz(boxes, SHAPE, "corners"))
        assert seg.shape[0] == colors.shape[0] and colors.shape[1] == 4
        assert np.allclose(colors, np.tile(_DEF, (colors.shape[0], 1)))

    def test_per_box_color_is_per_vertex(self):
        boxes = [place_box(SHAPE, 30, 30, 10, 10), place_box(SHAPE, 60, 60, 10, 10)]
        styles = [{"color": [1.0, 0.0, 0.0, 1.0]}, {"color": [0.0, 1.0, 0.0, 1.0]}]
        seg, colors = boxes_segments_and_colors_xyz(boxes, SHAPE, styles, _DEF)
        # 48 corner verts per box → first 48 red, next 48 green.
        assert np.allclose(colors[:48], [1.0, 0.0, 0.0, 1.0])
        assert np.allclose(colors[48:], [0.0, 1.0, 0.0, 1.0])

    def test_per_box_shape_mixes_corner_and_wireframe(self):
        boxes = [place_box(SHAPE, 30, 30, 10, 10), place_box(SHAPE, 60, 60, 10, 10)]
        styles = [{"shape": "corners"}, {"shape": "wireframe"}]
        seg, colors = boxes_segments_and_colors_xyz(boxes, SHAPE, styles, _DEF)
        assert seg.shape[0] == 48 + 24   # corners(48) + wireframe(24)

    def test_status_and_none_use_default_color(self):
        boxes = [place_box(SHAPE, 30, 30, 10, 10), place_box(SHAPE, 60, 60, 10, 10)]
        styles = [{"color": "status"}, None]
        _seg, colors = boxes_segments_and_colors_xyz(boxes, SHAPE, styles, _DEF)
        assert np.allclose(colors, np.tile(_DEF, (colors.shape[0], 1)))

    def test_opacity_scales_alpha(self):
        boxes = [place_box(SHAPE, 30, 30, 10, 10)]
        styles = [{"color": [1.0, 0.0, 0.0, 1.0], "opacity": 0.5}]
        _seg, colors = boxes_segments_and_colors_xyz(boxes, SHAPE, styles, _DEF)
        assert np.allclose(colors[:, 3], 0.5)

    def test_empty_inputs(self):
        seg, colors = boxes_segments_and_colors_xyz([], SHAPE, None, _DEF)
        assert seg.shape == (0, 3) and colors.shape == (0, 4)

    def test_anisotropic_voxel_scales_segments(self):
        boxes = [place_box(SHAPE, 30, 30, 10, 10)]
        seg1, _ = boxes_segments_and_colors_xyz(
            boxes, SHAPE, None, _DEF, voxel_size=(1.0, 1.0, 1.0))
        seg2, _ = boxes_segments_and_colors_xyz(
            boxes, SHAPE, None, _DEF, voxel_size=(4.0, 2.0, 0.5))
        # XYZ columns after permute: X·0.5, Y·2, Z·4 vs unit.
        assert np.allclose(seg2[:, 0], seg1[:, 0] * 0.5)
        assert np.allclose(seg2[:, 1], seg1[:, 1] * 2.0)
        assert np.allclose(seg2[:, 2], seg1[:, 2] * 4.0)


# ---------------------------------------------------------------------------
# Pure segment assembly
# ---------------------------------------------------------------------------

class TestSegmentAssembly:
    def test_empty_boxes_gives_empty(self):
        assert boxes_segments_xyz([], SHAPE).shape == (0, 3)

    def test_none_shape_gives_empty(self):
        box = place_box(SHAPE, 50, 50, 20, 20)
        assert boxes_segments_xyz([box], None).shape == (0, 3)

    def test_corners_gives_48_points_per_box(self):
        # 24 bracket segments * 2 endpoints (connect="segments").
        box = place_box(SHAPE, 50, 50, 20, 20)
        assert boxes_segments_xyz([box], SHAPE, "corners").shape == (48, 3)

    def test_wireframe_gives_24_points_per_box(self):
        # 12 cube edges * 2 endpoints.
        box = place_box(SHAPE, 50, 50, 20, 20)
        assert boxes_segments_xyz([box], SHAPE, "wireframe").shape == (24, 3)

    def test_two_boxes_concatenate(self):
        b0 = place_box(SHAPE, 30, 30, 10, 10)
        b1 = place_box(SHAPE, 60, 60, 10, 10)
        assert boxes_segments_xyz([b0, b1], SHAPE, "corners").shape == (96, 3)

    def test_axis_order_is_xyz_reversal_only(self):
        """Default view (no dims_order): scale=1 / translate=0 — the ONLY
        transform is the ZYX -> XYZ reversal."""
        box = place_box(SHAPE, 50, 50, 20, 20)
        zyx = _box_segments_zyx(box, SHAPE, "wireframe")
        xyz = boxes_segments_xyz([box], SHAPE, "wireframe")
        assert np.array_equal(xyz, zyx[:, ::-1])

    def test_identity_dims_order_is_byte_identical_to_default(self):
        """order (0,1,2) reduces to the plain ZYX->XYZ reversal — so the default
        orientation (extract box + preview box) is pixel-for-pixel as before."""
        box = place_box(SHAPE, 50, 50, 20, 20)
        default = boxes_segments_xyz([box], SHAPE, "wireframe")
        ident = boxes_segments_xyz([box], SHAPE, "wireframe", dims_order=(0, 1, 2))
        assert np.array_equal(default, ident)

    def test_rolled_dims_order_permutes_to_napari_displayed_order(self):
        """A rolled 2D view (XZ/YZ) permutes the scene columns to napari's
        displayed order = dims.order[::-1] (quirk 2, the transpose fix)."""
        box = place_box(SHAPE, 50, 50, 20, 20)
        zyx = _box_segments_zyx(box, SHAPE, "wireframe")
        # XZ roll: order (1,0,2) -> cols [2,0,1] -> canvas axes (X, Z).
        xz = boxes_segments_xyz([box], SHAPE, "wireframe", dims_order=(1, 0, 2))
        assert np.array_equal(xz, zyx[:, [2, 0, 1]])
        # YZ roll: order (2,0,1) -> cols [1,0,2] -> canvas axes (Y, Z).
        yz = boxes_segments_xyz([box], SHAPE, "wireframe", dims_order=(2, 0, 1))
        assert np.array_equal(yz, zyx[:, [1, 0, 2]])

    def test_non_permutation_dims_order_falls_back_to_default(self):
        """A malformed / non-length-3 order is ignored (plain reversal) — never
        raises, never scrambles the box."""
        box = place_box(SHAPE, 50, 50, 20, 20)
        default = boxes_segments_xyz([box], SHAPE, "wireframe")
        for bad in ((0, 1), (0, 1, 1), (0, 1, 2, 3), None):
            got = boxes_segments_xyz([box], SHAPE, "wireframe", dims_order=bad)
            assert np.array_equal(got, default)


# ---------------------------------------------------------------------------
# Physical-µm scale (Brick A) — the geometry multiplies by the anisotropic voxel
# ---------------------------------------------------------------------------

VOXEL = (4.0, 2.0, 0.5)  # (vz, vy, vx) — anisotropic, non-unit


class TestPhysicalVoxelScale:
    def test_default_voxel_is_byte_identical_to_before(self):
        """`voxel_size=(1,1,1)` (the default + the "unknown" case) → pixel-for-
        pixel the same as the pre-physical geometry (the regression guard)."""
        box = place_box(SHAPE, 50, 50, 20, 20)
        before = boxes_segments_xyz([box], SHAPE, "wireframe")
        ident = boxes_segments_xyz(
            [box], SHAPE, "wireframe", voxel_size=(1.0, 1.0, 1.0))
        # The explicit (1,1,1) must equal the parameter default (no voxel passed).
        assert np.array_equal(before, ident)

    def test_anisotropic_voxel_scales_zyx_then_reverses(self):
        """A box renders at `voxel_index × voxel` in each ZYX axis, THEN the
        ZYX→XYZ reversal — i.e. the multiply is PRE-permute (mirrors napari's
        `world = data×scale` → `scene[order[::-1]]`)."""
        box = place_box(SHAPE, 50, 50, 20, 20)
        zyx = _box_segments_zyx(box, SHAPE, "wireframe")
        vs = np.asarray(VOXEL, dtype=float)                 # (vz, vy, vx)
        expected = (zyx * vs)[:, ::-1]                       # scale THEN reverse
        got = boxes_segments_xyz([box], SHAPE, "wireframe", voxel_size=VOXEL)
        assert np.allclose(got, expected)
        # And it genuinely differs from the unit-voxel geometry (proves it acted).
        assert not np.allclose(
            got, boxes_segments_xyz([box], SHAPE, "wireframe"))

    def test_multiply_is_pre_permute_in_a_rolled_view(self):
        """The decisive anisotropy test: in a rolled XZ view the multiply must
        happen on the ZYX columns BEFORE the permutation — a post-permute multiply
        (or a fixed STTransform) would scale the WRONG displayed axis and
        misalign anisotropic Z."""
        box = place_box(SHAPE, 50, 50, 20, 20)
        zyx = _box_segments_zyx(box, SHAPE, "wireframe")
        vs = np.asarray(VOXEL, dtype=float)
        # XZ roll: order (1,0,2) -> cols [2,0,1].  Correct (pre-permute):
        expected = (zyx * vs)[:, [2, 0, 1]]
        got = boxes_segments_xyz(
            [box], SHAPE, "wireframe", dims_order=(1, 0, 2), voxel_size=VOXEL)
        assert np.allclose(got, expected)
        # A POST-permute multiply would instead be (zyx[:, [2,0,1]]) * vs — assert
        # our result is NOT that, so the pre/post distinction is actually tested.
        wrong_post = (zyx[:, [2, 0, 1]]) * vs
        assert not np.allclose(got, wrong_post)


class TestSetVoxelSize:
    def test_set_voxel_size_stores_a_valid_triple(self):
        ov = Step1BoxOverlay(None)
        ov.set_voxel_size((4.0, 2.0, 0.5))
        assert ov._voxel_size == (4.0, 2.0, 0.5)

    def test_bad_voxel_is_ignored_and_never_raises(self):
        ov = Step1BoxOverlay(None)
        assert ov._voxel_size == (1.0, 1.0, 1.0)           # default
        for bad in (None, (1.0, 2.0), (1, 2, 3, 4), (0.0, 1.0, 1.0),
                    (-1.0, 1.0, 1.0), ("x", "y", "z")):
            ov.set_voxel_size(bad)                          # must not raise
            assert ov._voxel_size == (1.0, 1.0, 1.0)        # unchanged

    def test_none_viewer_overlay_stays_a_safe_noop_with_voxel(self):
        ov = Step1BoxOverlay(None)                          # no real vispy lines
        ov.set_voxel_size((4.0, 2.0, 0.5))
        box = place_box(SHAPE, 50, 50, 20, 20)
        # None of these may raise even with a non-unit voxel set.
        ov.update([box], 0, SHAPE)
        ov.update_ghost(box, SHAPE)
        ov.destroy()

    def test_overlay_render_applies_the_voxel_to_lines(self):
        """The overlay threads `_voxel_size` into `boxes_segments_xyz`, so a
        rendered line's coords equal the scaled+reversed geometry."""
        ov = _overlay_with_fakes()
        ov.set_voxel_size(VOXEL)
        box = place_box(SHAPE, 50, 50, 20, 20)
        ov.update([box], 0, SHAPE)
        zyx = _box_segments_zyx(box, SHAPE, "corners")
        expected = (zyx * np.asarray(VOXEL, dtype=float))[:, ::-1]
        assert np.allclose(ov._lines["active"].pos, expected)


# ---------------------------------------------------------------------------
# Overlay wiring (injected fake vispy lines)
# ---------------------------------------------------------------------------

class _FakeLine:
    def __init__(self):
        self.pos = None
        self.connect = None
        self.visible = None
        self.parent = "scene"
        self.color = None
        self.width = None

    def set_data(self, pos=None, connect=None, color=None, width=None):
        if pos is not None:
            self.pos = np.asarray(pos, dtype=float)
        if connect is not None:
            self.connect = connect
        if color is not None:
            self.color = color
        if width is not None:
            self.width = width


def _overlay_with_fakes():
    ov = Step1BoxOverlay(None)  # no qt viewer -> no real vispy lines created
    ov._lines = {"idle": _FakeLine(), "active": _FakeLine(), "ghost": _FakeLine()}
    return ov


class TestOverlayWiring:
    def test_no_qt_viewer_is_a_safe_noop(self):
        ov = Step1BoxOverlay(None)
        assert ov._lines == {}
        box = place_box(SHAPE, 50, 50, 20, 20)
        # None of these may raise out of the (would-be) Qt slot.
        ov.update([box], 0, SHAPE)
        ov.update_ghost(box, SHAPE)
        ov.clear_ghost()
        ov.repaint()
        ov.destroy()

    def test_update_splits_active_from_idle(self):
        ov = _overlay_with_fakes()
        b0 = place_box(SHAPE, 30, 30, 10, 10)
        b1 = place_box(SHAPE, 60, 60, 10, 10)
        ov.update([b0, b1], 1, SHAPE)
        assert ov._lines["active"].pos.shape == (48, 3)   # box 1 only
        assert ov._lines["active"].visible is True
        assert ov._lines["idle"].pos.shape == (48, 3)     # box 0 only
        assert ov._lines["idle"].visible is True

    def test_no_active_index_all_idle(self):
        ov = _overlay_with_fakes()
        b0 = place_box(SHAPE, 30, 30, 10, 10)
        b1 = place_box(SHAPE, 60, 60, 10, 10)
        ov.update([b0, b1], None, SHAPE)
        assert ov._lines["idle"].pos.shape == (96, 3)
        assert ov._lines["active"].visible is False       # nothing active -> hidden

    def test_ghost_update_and_clear(self):
        ov = _overlay_with_fakes()
        box = place_box(SHAPE, 50, 50, 20, 20)
        ov.update_ghost(box, SHAPE)
        assert ov._lines["ghost"].pos.shape == (48, 3)
        assert ov._lines["ghost"].visible is True
        ov.clear_ghost()
        assert ov._lines["ghost"].visible is False

    def test_destroy_detaches_and_clears(self):
        ov = _overlay_with_fakes()
        lines = list(ov._lines.values())
        ov.destroy()
        assert ov._lines == {}
        assert all(line.parent is None for line in lines)


# ---------------------------------------------------------------------------
# Part C (2026-07-27) — ④ Render/style -> live overlay appearance
# ---------------------------------------------------------------------------

class TestOverlayStyle:
    def test_defaults_reproduce_the_builtin_look(self):
        from napari_epicycle.data_preparation._box_overlay import (
            _IDLE_COLOR, _ACTIVE_COLOR, _IDLE_WIDTH, _ACTIVE_WIDTH)
        ov = Step1BoxOverlay(None)
        assert ov._idle_width == _IDLE_WIDTH
        assert ov._active_width == _ACTIVE_WIDTH
        assert ov._effective_color("idle")[:3] == _IDLE_COLOR[:3]
        assert ov._effective_color("active")[:3] == _ACTIVE_COLOR[:3]

    def test_set_style_updates_width_alpha_and_colour(self):
        ov = Step1BoxOverlay(None)
        ov.set_style(thickness=6, opacity=0.5, color=(0.1, 0.2, 0.3))
        assert ov._idle_width == 6.0
        assert ov._active_width == 7.5              # active kept a touch thicker
        assert ov._alpha == 0.5
        idle = ov._effective_color("idle")
        assert idle[:3] == (0.1, 0.2, 0.3)          # global colour applied
        assert abs(idle[3] - 0.5 * 0.9) < 1e-9      # opacity folds base alpha

    def test_status_colour_is_none_and_keeps_defaults(self):
        from napari_epicycle.data_preparation._box_overlay import (
            _IDLE_COLOR, _ACTIVE_COLOR)
        ov = Step1BoxOverlay(None)
        ov.set_style(color=(0.5, 0.5, 0.5))         # override first
        ov.set_style(color=None)                    # "status colour" resets
        assert ov._color_override is None
        assert ov._effective_color("idle")[:3] == _IDLE_COLOR[:3]
        assert ov._effective_color("active")[:3] == _ACTIVE_COLOR[:3]

    def test_keep_sentinel_leaves_colour_unchanged(self):
        ov = Step1BoxOverlay(None)
        ov.set_style(color=(0.5, 0.5, 0.5))
        ov.set_style(thickness=3)                    # color defaults to "__keep__"
        assert ov._color_override == (0.5, 0.5, 0.5)

    def test_bad_values_are_ignored(self):
        ov = Step1BoxOverlay(None)
        ov.set_style(thickness="x", opacity="y")
        assert ov._idle_width == 2.0 and ov._alpha == 1.0

    def test_update_applies_style_to_lines(self):
        ov = _overlay_with_fakes()
        ov.set_style(thickness=5, opacity=0.4, color=(0.2, 0.4, 0.6))
        b0 = place_box(SHAPE, 30, 30, 10, 10)
        b1 = place_box(SHAPE, 60, 60, 10, 10)
        ov.update([b0, b1], 1, SHAPE)
        assert ov._lines["idle"].width == 5.0
        assert ov._lines["active"].width == 6.5
        assert ov._lines["idle"].color[:3] == (0.2, 0.4, 0.6)

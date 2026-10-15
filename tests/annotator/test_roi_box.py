"""Tests for the 3D ROI box geometry (Add-tab P1·2 pure spine).

``roi_box`` lifts the headlessly-tested core of the EpiCycle-Viz prototype: a
center+size box, clamped bounds (slices-tuple), and 3D wireframe geometry. Pure
NumPy — no Qt, no napari.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.roi_box import (
    BoxRec,
    box_corners,
    box_edges,
    box_face_rectangle,
    box_wireframe,
)


class TestBounds:
    def test_central_box_bounds(self):
        box = BoxRec(cz=10, cy=20, cx=30, sz=4, sy=6, sx=8)
        b = box.bounds_dict((40, 60, 80))
        assert (b["z0"], b["z1"]) == (8, 12)
        assert (b["y0"], b["y1"]) == (17, 23)
        assert (b["x0"], b["x1"]) == (26, 34)

    def test_bounds_slices_match_dict(self):
        box = BoxRec(5, 5, 5, 4, 4, 4)
        shape = (20, 20, 20)
        b = box.bounds_dict(shape)
        sl = box.bounds_slices(shape)
        assert sl == (
            slice(b["z0"], b["z1"]),
            slice(b["y0"], b["y1"]),
            slice(b["x0"], b["x1"]),
        )
        # The ROI actually crops a 3D array to the box extent.
        vol = np.zeros(shape, dtype=np.int32)
        assert vol[sl].shape == box.shape_in(shape)

    def test_clamps_to_lower_edge(self):
        box = BoxRec(cz=0, cy=0, cx=0, sz=10, sy=10, sx=10)
        b = box.bounds_dict((20, 20, 20))
        assert b["z0"] == 0 and b["y0"] == 0 and b["x0"] == 0
        assert b["z1"] >= 1 and b["y1"] >= 1 and b["x1"] >= 1

    def test_clamps_to_upper_edge(self):
        box = BoxRec(cz=19, cy=19, cx=19, sz=10, sy=10, sx=10)
        b = box.bounds_dict((20, 20, 20))
        assert b["z1"] == 20 and b["y1"] == 20 and b["x1"] == 20
        assert b["z0"] < 20 and b["y0"] < 20 and b["x0"] < 20

    def test_always_at_least_one_voxel(self):
        # Degenerate size 0 is bumped to 1; never an empty slice.
        box = BoxRec(cz=5, cy=5, cx=5, sz=0, sy=0, sx=0)
        b = box.bounds_dict((10, 10, 10))
        assert b["z1"] > b["z0"]
        assert b["y1"] > b["y0"]
        assert b["x1"] > b["x0"]

    def test_bounds_never_exceed_shape(self):
        for cz, cy, cx in [(-5, -5, -5), (100, 100, 100), (3, 7, 1)]:
            box = BoxRec(cz, cy, cx, 9, 9, 9)
            b = box.bounds_dict((12, 14, 16))
            assert 0 <= b["z0"] < b["z1"] <= 12
            assert 0 <= b["y0"] < b["y1"] <= 14
            assert 0 <= b["x0"] < b["x1"] <= 16


class TestConstructionHelpers:
    def test_centered_default_quarter_size(self):
        box = BoxRec.centered((40, 80, 120))
        assert (box.cz, box.cy, box.cx) == (20, 40, 60)
        assert (box.sz, box.sy, box.sx) == (10, 20, 30)

    def test_centered_explicit_size(self):
        box = BoxRec.centered((40, 40, 40), size=(4, 6, 8))
        assert (box.sz, box.sy, box.sx) == (4, 6, 8)

    def test_with_center_and_size_are_pure(self):
        box = BoxRec(1, 1, 1, 2, 2, 2)
        moved = box.with_center(5, 6, 7)
        resized = box.with_size(8, 9, 10)
        assert (moved.cz, moved.cy, moved.cx) == (5, 6, 7)
        assert (resized.sz, resized.sy, resized.sx) == (8, 9, 10)
        # original untouched (dataclass replace returns a new instance)
        assert (box.cz, box.sz) == (1, 2)

    def test_with_size_floors_at_one(self):
        box = BoxRec(5, 5, 5, 4, 4, 4).with_size(0, -3, 0)
        assert (box.sz, box.sy, box.sx) == (1, 1, 1)

    def test_clamped_recenters_at_edge(self):
        box = BoxRec(cz=0, cy=0, cx=0, sz=10, sy=10, sx=10)
        shape = (20, 20, 20)
        c = box.clamped(shape)
        b = box.bounds_dict(shape)
        assert c.cz == (b["z0"] + b["z1"]) // 2


class TestShrunkToBounds:
    """Brick 1 — the handle-match re-seed (SHRINK, the default edge mode).

    ``shrunk_to_bounds`` collapses center+size onto ``bounds_dict``'s own output,
    so the geometry widgets can be seeded to what is actually drawn/extracted
    without the ``clamped()`` creep.
    """

    def test_high_edge_bounds_unchanged(self):
        # A box overflowing the HIGH edge (the case test_clamped_recenters_at_edge
        # misses — it uses a low-edge box).  full=(50,512,512); X center=450,
        # size=200 → X bounds truncate at 512.
        shape = (50, 512, 512)
        box = BoxRec(cz=25, cy=256, cx=450, sz=50, sy=200, sx=200)
        b_before = box.bounds_dict(shape)
        reseeded = box.shrunk_to_bounds(shape)
        # Extraction (which reads bounds_dict) is byte-identical to the raw box.
        assert reseeded.bounds_dict(shape) == b_before

    def test_high_edge_is_idempotent(self):
        # Applying the re-seed a SECOND time changes nothing (no creep) — unlike
        # clamped(), which re-grows the extent on the next pass.
        shape = (50, 512, 512)
        box = BoxRec(cz=25, cy=256, cx=450, sz=50, sy=200, sx=200)
        once = box.shrunk_to_bounds(shape)
        twice = once.shrunk_to_bounds(shape)
        assert twice == once
        assert twice.bounds_dict(shape) == once.bounds_dict(shape)

    def test_center_and_size_match_clamped_extent(self):
        shape = (50, 512, 512)
        box = BoxRec(cz=25, cy=256, cx=450, sz=50, sy=200, sx=200)
        b = box.bounds_dict(shape)
        r = box.shrunk_to_bounds(shape)
        # Per axis: center = (a0+a1)//2, size = a1-a0.
        assert (r.cx, r.sx) == ((b["x0"] + b["x1"]) // 2, b["x1"] - b["x0"])
        assert (r.cy, r.sy) == ((b["y0"] + b["y1"]) // 2, b["y1"] - b["y0"])
        assert (r.cz, r.sz) == ((b["z0"] + b["z1"]) // 2, b["z1"] - b["z0"])

    def test_low_edge_is_idempotent_too(self):
        box = BoxRec(cz=0, cy=0, cx=0, sz=10, sy=10, sx=10)
        shape = (20, 20, 20)
        once = box.shrunk_to_bounds(shape)
        assert once.shrunk_to_bounds(shape) == once
        assert once.bounds_dict(shape) == box.bounds_dict(shape)

    def test_interior_box_is_unchanged(self):
        # A box wholly inside the volume is already its own bounds → no-op.
        box = BoxRec(cz=10, cy=20, cx=30, sz=4, sy=6, sx=8)
        shape = (40, 60, 80)
        assert box.shrunk_to_bounds(shape) == box

    def test_does_not_creep_like_clamped(self):
        # The documented clamped() pathology: dim=10, size=6, center=8 → [5,10);
        # clamped() re-widens to [4,10) (gains a column); shrunk_to_bounds does
        # not.  (Single-axis check via the X axis; other axes interior.)
        shape = (20, 20, 10)
        box = BoxRec(cz=10, cy=10, cx=8, sz=4, sy=4, sx=6)
        assert box.bounds_dict(shape)["x0"] == 5
        assert box.bounds_dict(shape)["x1"] == 10
        # clamped() creeps:
        assert box.clamped(shape).bounds_dict(shape)["x0"] == 4      # widened
        # shrunk_to_bounds does not:
        assert box.shrunk_to_bounds(shape).bounds_dict(shape)["x0"] == 5


class TestSlidInward:
    """Brick 1 — the opt-in KEEP-SIZE edge mode.

    ``slid_inward`` re-centers to keep the requested size within the volume via a
    FIXED idempotent clamp (unlike ``clamped()``), so the extracted region near
    an edge deliberately differs from ``shrunk_to_bounds``.
    """

    def test_preserves_size(self):
        shape = (50, 512, 512)
        box = BoxRec(cz=25, cy=256, cx=450, sz=50, sy=200, sx=200)
        r = box.slid_inward(shape)
        assert (r.sz, r.sy, r.sx) == (box.sz, box.sy, box.sx)   # size kept

    def test_full_size_fits_inside(self):
        # Keeping size 200 on a 512 axis → the box now fits fully inside.
        shape = (50, 512, 512)
        box = BoxRec(cz=25, cy=256, cx=450, sz=50, sy=200, sx=200)
        r = box.slid_inward(shape)
        b = r.bounds_dict(shape)
        assert b["x1"] - b["x0"] == 200        # full requested extent preserved
        assert 0 <= b["x0"] < b["x1"] <= 512

    def test_is_idempotent(self):
        shape = (50, 512, 512)
        box = BoxRec(cz=25, cy=256, cx=450, sz=50, sy=200, sx=200)
        once = box.slid_inward(shape)
        assert once.slid_inward(shape) == once

    def test_idempotent_even_when_size_exceeds_dim(self):
        # size > dim → the fixed clamp still converges to a single fixed point.
        shape = (8, 8, 8)
        box = BoxRec(cz=4, cy=4, cx=4, sz=20, sy=20, sx=20)
        once = box.slid_inward(shape)
        assert once.slid_inward(shape) == once

    def test_differs_from_shrink_at_edge(self):
        # Documents the intended edge-extraction difference between the modes.
        shape = (50, 512, 512)
        box = BoxRec(cz=25, cy=256, cx=450, sz=50, sy=200, sx=200)
        keep = box.slid_inward(shape).bounds_dict(shape)
        shrink = box.shrunk_to_bounds(shape).bounds_dict(shape)
        assert keep != shrink
        assert keep["x1"] - keep["x0"] == 200        # keep: full size
        assert shrink["x1"] - shrink["x0"] == 162     # shrink: truncated [350,512)


class TestWireframe:
    def test_box_edges_returns_twelve_edges(self):
        b = BoxRec(5, 5, 5, 4, 4, 4).bounds_dict((20, 20, 20))
        edges = box_edges(b)
        assert len(edges) == 12
        for e in edges:
            assert e.shape == (2, 3)
            # each edge differs in exactly one coordinate
            assert np.count_nonzero(e[0] != e[1]) == 1

    def test_box_edges_span_the_bounds(self):
        b = BoxRec(5, 5, 5, 4, 6, 8).bounds_dict((20, 20, 20))
        edges = np.array(box_edges(b))
        zs = edges[..., 0]
        ys = edges[..., 1]
        xs = edges[..., 2]
        assert zs.min() == b["z0"] and zs.max() == b["z1"]
        assert ys.min() == b["y0"] and ys.max() == b["y1"]
        assert xs.min() == b["x0"] and xs.max() == b["x1"]

    def test_box_corners_are_short_segments(self):
        b = BoxRec(10, 10, 10, 8, 8, 8).bounds_dict((40, 40, 40))
        segs = box_corners(b)
        # 8 corners × 3 brackets each
        assert len(segs) == 24
        for s in segs:
            assert s.shape == (2, 3)

    def test_box_wireframe_dispatch(self):
        box = BoxRec(5, 5, 5, 4, 4, 4)
        shape = (20, 20, 20)
        assert len(box_wireframe(box, shape, "edges")) == 12
        assert len(box_wireframe(box, shape, "corners")) == 24


class TestFaceRectangle:
    @pytest.mark.parametrize("axis,const_idx", [("z", 0), ("y", 1), ("x", 2)])
    def test_face_rectangle_is_closed_planar(self, axis, const_idx):
        b = BoxRec(10, 10, 10, 6, 8, 10).bounds_dict((40, 40, 40))
        rect = box_face_rectangle(b, axis)
        assert rect.shape == (5, 3)
        # closed loop: first == last
        assert np.array_equal(rect[0], rect[-1])
        # all points share the same value on the constant axis (mid-plane)
        assert len(np.unique(rect[:, const_idx])) == 1

    def test_face_rectangle_mid_plane_value(self):
        b = BoxRec(10, 10, 10, 6, 8, 10).bounds_dict((40, 40, 40))
        rect = box_face_rectangle(b, "z")
        assert rect[0, 0] == (b["z0"] + b["z1"]) // 2

    def test_invalid_axis_raises(self):
        b = BoxRec(5, 5, 5, 4, 4, 4).bounds_dict((20, 20, 20))
        with pytest.raises(ValueError):
            box_face_rectangle(b, "w")

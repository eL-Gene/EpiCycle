"""
TDD suite for the Mode-B ("full") full-resolution IoU seam-stitch core.

The module under test (``epicycle/core/iou_stitch.py``) is the PURE-ALGORITHM
centerpiece of Stack 6: it matches per-tile instance labels across seam overlaps
by IoU, merges the matches via union-find, and produces a dense global relabel
LUT — all without ever loading a whole full-Z tile volume into RAM.

These tests are GPU/Qt/napari-free. They build on the conftest fixtures
(``make_tile_mask``, ``two_overlapping_tiles``, ``four_corner_tiles``) and the
flush-anchored tile geometry from ``compute_tile_grid``.

The three highest-stakes behaviors each get a BLOCKING gate test:
  * 1:1 max-IoU one-to-many resolution (Q3 under-merge bias),
  * a corner-only cell resolving across 4 tiles,
  * a Z-band seam mismatch (shared-Z matches, exclusive-Z stays singleton).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from packages.epicycle_scaledseg.core import iou_stitch as S
from packages.epicycle_scaledseg.utils.tile_grid import TileGeom, compute_tile_grid


# ---------------------------------------------------------------------------
# Small geometry helpers for building TileGeom-likes inline
# ---------------------------------------------------------------------------


def _geom(name, y0, x0, h, w, i=0, j=0):
    return TileGeom(
        name=name,
        origin=(y0, x0),
        extent=(h, w),
        grid_index=(i, j),
        is_edge=True,
        is_flush=False,
    )


def _geom_dict(name, y0, x0, h, w, i=0, j=0):
    return {
        "tile_name": name,
        "origin": [y0, x0],
        "extent": [h, w],
        "grid_index": [i, j],
        "is_edge": True,
        "is_flush": False,
    }


def _global_to_local_rect(grect, origin):
    """(y0,y1,x0,x1) global rect -> local rect within a tile at ``origin``."""
    gy0, gy1, gx0, gx1 = grect
    oy, ox = origin
    return (gy0 - oy, gy1 - oy, gx0 - ox, gx1 - ox)


def _present_global_ids(spec, off):
    """Global ids for the labels that ACTUALLY exist in a tile (no phantoms).

    A writer builds its id universe from the labels really present in each tile,
    not from ``range(1, nmax+1)`` (which would invent singleton components for
    unused ids between 1 and nmax).
    """
    local = [int(v) for v in np.unique(spec.array) if v != 0]
    return [ell + off[spec.name] for ell in local]


# ===========================================================================
# Geometry — overlap_rect / enumerate_adjacent_pairs / cores
# ===========================================================================


def test_overlap_rect_horizontal():
    # Two X-adjacent 16x16 tiles, origins x=0 and x=12 -> overlap global X[12,16).
    a = _geom("a", 0, 0, 16, 16)
    b = _geom("b", 0, 12, 16, 16)
    rect = S.overlap_rect(a, b)
    assert rect == (0, 16, 12, 16)


def test_overlap_rect_vertical():
    a = _geom("a", 0, 0, 16, 16)
    b = _geom("b", 12, 0, 16, 16)
    rect = S.overlap_rect(a, b)
    assert rect == (12, 16, 0, 16)


def test_overlap_rect_corner_4tiles():
    # The shared 4-way corner of a 2x2 step-12 grid is global Y[12,16) x X[12,16).
    a = _geom("00", 0, 0, 16, 16)
    d = _geom("11", 12, 12, 16, 16)
    rect = S.overlap_rect(a, d)
    assert rect == (12, 16, 12, 16)


def test_overlap_rect_disjoint_none():
    a = _geom("a", 0, 0, 16, 16)
    # origin x=20 -> spans X[20,36); a spans X[0,16) -> disjoint.
    b = _geom("b", 0, 20, 16, 16)
    assert S.overlap_rect(a, b) is None


def test_overlap_rect_accepts_dict():
    a = _geom_dict("a", 0, 0, 16, 16)
    b = _geom_dict("b", 0, 12, 16, 16)
    assert S.overlap_rect(a, b) == (0, 16, 12, 16)


def test_enumerate_adjacent_pairs_2x2():
    grid = compute_tile_grid(28, 28, 16, 0.25)  # step 12 -> 2x2 grid
    pairs = S.enumerate_adjacent_pairs(grid.tiles)
    names = {(a, b) for (a, b, _r) in pairs}
    # All 4 tiles pairwise overlap (corner makes 0_0 <-> 1_1 overlap too): C(4,2)=6.
    assert len(pairs) == 6
    assert ("tile_0_0", "tile_1_1") in names or ("tile_1_1", "tile_0_0") in names


def test_slab_alignment(two_overlapping_tiles):
    """The overlap rect maps to the SAME global voxels read from both tiles."""
    t0, t1 = two_overlapping_tiles
    a = _geom("tile_0_0", t0.y0, t0.x0, t0.patch_size, t0.patch_size)
    b = _geom("tile_0_1", t1.y0, t1.x0, t1.patch_size, t1.patch_size)
    grect = S.overlap_rect(a, b)
    la = _global_to_local_rect(grect, (t0.y0, t0.x0))
    lb = _global_to_local_rect(grect, (t1.y0, t1.x0))
    # Read each tile's overlap sub-rect; the straddle cell (label 3) sits on the
    # same (Z,Y) voxels in both crops.
    _, _, ax0, ax1 = la
    _, _, bx0, bx1 = lb
    cropA = t0.array[:, :, ax0:ax1]
    cropB = t1.array[:, :, bx0:bx1]
    assert np.array_equal(cropA == 3, cropB == 3)


def test_cores_are_disjoint():
    grid = compute_tile_grid(28, 28, 16, 0.25)
    cores = S.compute_tile_cores(grid.tiles, full_size=(28, 28))
    # Build a coverage map of (H,W); each voxel owned by exactly one tile core.
    owner = np.full((28, 28), -1, dtype=int)
    for idx, g in enumerate(grid.tiles):
        y0, y1, x0, x1 = cores[g.name]["dst_core"]
        block = owner[y0:y1, x0:x1]
        assert (block == -1).all(), f"core overlap at tile {g.name}"
        owner[y0:y1, x0:x1] = idx


def test_cores_cover_full_extent():
    grid = compute_tile_grid(28, 28, 16, 0.25)
    cores = S.compute_tile_cores(grid.tiles, full_size=(28, 28))
    owner = np.full((28, 28), -1, dtype=int)
    for idx, g in enumerate(grid.tiles):
        y0, y1, x0, x1 = cores[g.name]["dst_core"]
        owner[y0:y1, x0:x1] = idx
    assert (owner >= 0).all(), "some voxel unowned by any tile core"


def test_edge_tiles_keep_full_extent():
    grid = compute_tile_grid(28, 28, 16, 0.25)
    cores = S.compute_tile_cores(grid.tiles, full_size=(28, 28))
    # Top-left tile owns the outer corner (0,0); bottom-right owns (28,28).
    tl = cores["tile_0_0"]["dst_core"]
    assert tl[0] == 0 and tl[2] == 0
    br = cores["tile_1_1"]["dst_core"]
    assert br[1] == 28 and br[3] == 28


def test_core_src_inset_maps_to_dst():
    """A tile's src_inset (local) maps to its dst_core (global) by the origin."""
    grid = compute_tile_grid(28, 28, 16, 0.25)
    cores = S.compute_tile_cores(grid.tiles, full_size=(28, 28))
    for g in grid.tiles:
        oy, ox = g.origin
        dy0, dy1, dx0, dx1 = cores[g.name]["dst_core"]
        sy0, sy1, sx0, sx1 = cores[g.name]["src_inset"]
        assert (dy0 - sy0, dx0 - sx0) == (oy, ox)
        assert (dy1 - dy0, dx1 - dx0) == (sy1 - sy0, sx1 - sx0)


# ===========================================================================
# Uniquify — global offsets / tile_nmax
# ===========================================================================


def test_global_offsets_contiguous():
    nmax = {"tile_0_0": 3, "tile_0_1": 5, "tile_1_0": 2}
    off = S.compute_global_offsets(nmax)
    # Sorted by name: 0_0(0), 0_1(3), 1_0(3+5=8).
    assert off["tile_0_0"] == 0
    assert off["tile_0_1"] == 3
    assert off["tile_1_0"] == 8


def test_offset_blocks_nonoverlap_at_boundary():
    """Adjacent tiles' global-id blocks must not collide at the boundary.

    A naive ``offset[t] = prev_offset + prev_nmax - 1`` (off-by-one) would make
    block t's lowest id == block t-1's highest id.
    """
    nmax = {"a": 4, "b": 4}
    off = S.compute_global_offsets(nmax)
    # a: local 1..4 -> global 1..4 ; b: local 1..4 -> global off[b]+1..+4.
    a_ids = {ell + off["a"] for ell in range(1, 5)}
    b_ids = {ell + off["b"] for ell in range(1, 5)}
    assert a_ids.isdisjoint(b_ids)
    # specifically the boundary: a's max (4) != b's min.
    assert max(a_ids) != min(b_ids)


def test_tile_nmax_streaming(make_tile_mask):
    spec = make_tile_mask(
        labels={5: (slice(0, 4), slice(0, 4)), 9: (slice(8, 12), slice(8, 12))}
    )
    assert S.tile_nmax(str(spec.mask_path)) == 9
    # matches np.max of the whole volume
    assert S.tile_nmax(str(spec.mask_path)) == int(tifffile.imread(str(spec.mask_path)).max())


def test_tile_nmax_all_background(make_tile_mask):
    spec = make_tile_mask(labels={})
    assert S.tile_nmax(str(spec.mask_path)) == 0


def test_offset_map_does_not_mutate_tiles(two_overlapping_tiles):
    """Building the offset map reads nmax but never rewrites the tile masks."""
    t0, t1 = two_overlapping_tiles
    before0 = tifffile.imread(str(t0.mask_path)).copy()
    before1 = tifffile.imread(str(t1.mask_path)).copy()
    nmax = {t0.name: S.tile_nmax(str(t0.mask_path)),
            t1.name: S.tile_nmax(str(t1.mask_path))}
    _ = S.compute_global_offsets(nmax)
    assert np.array_equal(tifffile.imread(str(t0.mask_path)), before0)
    assert np.array_equal(tifffile.imread(str(t1.mask_path)), before1)


# ===========================================================================
# IoU edges
# ===========================================================================


def _edges_for_pair(t0, t1, **kw):
    """Helper: run overlap_iou_edges on the two-tile fixture seam."""
    a = _geom("tile_0_0", t0.y0, t0.x0, t0.patch_size, t0.patch_size)
    b = _geom("tile_0_1", t1.y0, t1.x0, t1.patch_size, t1.patch_size)
    grect = S.overlap_rect(a, b)
    la = _global_to_local_rect(grect, (t0.y0, t0.x0))
    lb = _global_to_local_rect(grect, (t1.y0, t1.x0))
    nmax = {t0.name: S.tile_nmax(str(t0.mask_path)),
            t1.name: S.tile_nmax(str(t1.mask_path))}
    off = S.compute_global_offsets(nmax)
    depth = t0.full_z
    return S.overlap_iou_edges(
        str(t0.mask_path), str(t1.mask_path), la, lb,
        off[t0.name], off[t1.name], depth, **kw
    ), off


def test_iou_identical_cell(two_overlapping_tiles):
    """The straddle cell (perfect co-occurrence, IoU=1.0) yields one edge."""
    t0, t1 = two_overlapping_tiles
    edges, off = _edges_for_pair(t0, t1)
    # Exactly one edge, fusing local 3<->3 (the straddle cell).
    assert len(edges) == 1
    ga, gb, iou = edges[0]
    assert ga == 3 + off[t0.name]
    assert gb == 3 + off[t1.name]
    assert iou == pytest.approx(1.0)


def test_iou_partial_above_threshold(make_tile_mask, tmp_path):
    """A cell sharing ~half the slab still clears tau=0.25 -> edge."""
    base = tmp_path / "tiles"
    # tile A origin x0=0 spans X[0,16); tile B origin x0=12 spans X[12,28).
    # overlap slab = global X[12,16). A's cell fills full slab; B's cell fills
    # half the slab voxels (in Y) -> IoU = inter/union ~ 0.5.
    tA = make_tile_mask(name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
                        labels={4: (slice(4, 12), slice(12, 16))})
    tB = make_tile_mask(name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
                        labels={7: (slice(4, 8), slice(0, 4))})  # half the Y span
    edges, off = _edges_for_pair(tA, tB)
    assert len(edges) == 1
    _, _, iou = edges[0]
    assert iou == pytest.approx(0.5, abs=0.05)


def test_iou_below_threshold_no_edge(make_tile_mask, tmp_path):
    """A small overlap whose IoU is strictly below tau yields no edge.

    A fills the whole 4px-wide slab over Y[4,12) (8 rows x 4 cols = 32 vox/z).
    B touches only Y[4,8) x 1 col (4 rows x 1 col = 4 vox/z). inter = 4,
    union = 32 + 4 - 4 = 32 -> IoU = 0.125 < tau=0.25. vmin is small so the IoU
    (not vmin) is what rejects the edge.
    """
    base = tmp_path / "tiles"
    tA = make_tile_mask(name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
                        labels={4: (slice(4, 12), slice(12, 16))})  # full slab (8*4 per z)
    tB = make_tile_mask(name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
                        labels={7: (slice(4, 8), slice(0, 1))})  # 4 rows x 1 col -> IoU 0.125
    edges, _ = _edges_for_pair(tA, tB, tau=0.25, vmin=1)
    assert edges == []


def test_vmin_rejects_coincidental(make_tile_mask, tmp_path):
    """A perfectly-aligned but TINY overlap (high IoU, low voxel count) is rejected by vmin."""
    base = tmp_path / "tiles"
    # Both cells occupy the SAME tiny global voxel set (IoU=1.0) but only a few voxels.
    # global overlap X[12,16). Place a 1x1xZsmall cell at global Y[5,6) X[12,13).
    tA = make_tile_mask(name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
                        z_range=(4, 6),
                        labels={4: (slice(4, 6), slice(5, 6), slice(12, 13))})
    tB = make_tile_mask(name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
                        z_range=(4, 6),
                        labels={7: (slice(4, 6), slice(5, 6), slice(0, 1))})
    # inter = 2 voxels (z in [4,6), 1 y, 1 x). IoU = 1.0 but inter < vmin=10.
    edges_hi_vmin, _ = _edges_for_pair(tA, tB, tau=0.25, vmin=10)
    assert edges_hi_vmin == []
    # With vmin=1, the same coincidence now produces an edge (proves vmin was the gate).
    edges_lo_vmin, _ = _edges_for_pair(tA, tB, tau=0.25, vmin=1)
    assert len(edges_lo_vmin) == 1


def test_containment_rescue_off_by_default(make_tile_mask, tmp_path):
    """A big-vs-small containment pair (low IoU, high containment) is NOT merged by default."""
    base = tmp_path / "tiles"
    # A's cell fills the full slab; B's cell is fully CONTAINED in a corner of it.
    # IoU is low (small/large) but inter/min(area)=1.0. Default off -> no edge.
    tA = make_tile_mask(name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
                        labels={4: (slice(4, 12), slice(12, 16))})  # big: 8*4 per z
    tB = make_tile_mask(name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
                        labels={7: (slice(4, 6), slice(0, 1))})  # small: 1*1 per z, 2 z
    # default containment_rescue=False -> below-tau IoU => no edge
    edges_off, _ = _edges_for_pair(tA, tB, tau=0.25, vmin=1, containment_rescue=False)
    assert edges_off == []
    # rescue ON -> the contained cell is accepted
    edges_on, _ = _edges_for_pair(tA, tB, tau=0.25, vmin=1, containment_rescue=True)
    assert len(edges_on) == 1


# ===========================================================================
# Connected components — union-find / compaction
# ===========================================================================


def test_union_find_transitive_3tiles():
    ids = [1, 2, 3]
    edges = [(1, 2, 0.9), (2, 3, 0.8)]  # A-B, B-C -> all one component
    roots = S.union_find_merge(ids, edges)
    assert roots[1] == roots[2] == roots[3]


def test_four_tile_corner():
    ids = [10, 11, 12, 13]
    # corner cell present in all 4 tiles, pairwise edges via the corner overlaps
    edges = [(10, 11, 0.9), (10, 12, 0.9), (11, 13, 0.9), (12, 13, 0.9)]
    roots = S.union_find_merge(ids, edges)
    assert len({roots[i] for i in ids}) == 1


def test_singleton_keeps_id():
    ids = [5, 6, 7]
    edges = [(5, 6, 0.9)]
    roots = S.union_find_merge(ids, edges)
    assert roots[5] == roots[6]
    assert roots[7] != roots[5]  # singleton stays its own component


def test_two_distinct_stay_separate():
    ids = [1, 2, 3, 4]
    edges = [(1, 2, 0.9), (3, 4, 0.9)]
    roots = S.union_find_merge(ids, edges)
    assert roots[1] == roots[2]
    assert roots[3] == roots[4]
    assert roots[1] != roots[3]


def test_compact_roots_dense_and_deterministic():
    roots = {1: 1, 2: 1, 3: 3, 5: 3, 7: 7}
    lut = S.compact_roots(roots)
    # 3 distinct roots {1,3,7} -> dense {1,2,3} sorted; background 0 stays 0.
    assert set(lut.values()) == {1, 2, 3}
    # All members of a root share its dense id.
    assert lut[1] == lut[2]
    assert lut[3] == lut[5]
    assert lut[1] != lut[3] != lut[7]
    # determinism: sorted roots -> sorted dense ids
    assert lut[1] < lut[3] < lut[7]


def test_compact_roots_background_stays_zero():
    roots = {0: 0, 1: 1, 2: 1}
    lut = S.compact_roots(roots)
    assert lut[0] == 0


# ===========================================================================
# Seam provenance — clean / seam / corner classification (QC keystone)
# ===========================================================================


def test_classify_seam_status():
    assert S.classify_seam_status(0) == "clean"
    assert S.classify_seam_status(1) == "clean"
    assert S.classify_seam_status(2) == "seam"
    assert S.classify_seam_status(3) == "corner"
    assert S.classify_seam_status(4) == "corner"


def test_compute_label_provenance_clean_seam_corner():
    """A cell from 1 tile -> clean; merged across 2 -> seam; across 4 -> corner."""
    offsets = {"t00": 0, "t01": 10, "t10": 20, "t11": 30}
    # global id = local + offset[tile]
    present_ids = [5, 13, 2, 12, 22, 32, 8]
    gid_to_tile = {5: "t00", 13: "t01", 2: "t00", 12: "t01",
                   22: "t10", 32: "t11", 8: "t00"}
    # dense 1 = t00:5 + t01:3 (seam); dense 2 = 4-way corner; dense 3 = t00:8 only
    dense_lut = {5: 1, 13: 1, 2: 2, 12: 2, 22: 2, 32: 2, 8: 3}

    prov = S.compute_label_provenance(present_ids, gid_to_tile, offsets, dense_lut)

    assert prov[1]["seam_status"] == "seam" and prov[1]["n_tiles"] == 2
    assert prov[1]["tiles"] == ["t00", "t01"]
    assert prov[1]["constituents"] == [("t00", 5), ("t01", 3)]
    assert prov[2]["seam_status"] == "corner" and prov[2]["n_tiles"] == 4
    assert prov[3]["seam_status"] == "clean" and prov[3]["n_tiles"] == 1
    assert 0 not in prov  # background excluded


def test_compute_label_provenance_excludes_background():
    assert S.compute_label_provenance([5], {5: "t"}, {"t": 0}, {5: 0}) == {}


def test_compute_label_provenance_min_match_iou():
    """With ``edges``, each merged cell carries the WEAKEST internal seam IoU; a
    clean (single-tile, no-edge) cell carries ``min_match_iou=None``."""
    offsets = {"t00": 0, "t01": 10, "t10": 20, "t11": 30}
    present_ids = [5, 13, 2, 12, 22, 32, 8]
    gid_to_tile = {5: "t00", 13: "t01", 2: "t00", 12: "t01",
                   22: "t10", 32: "t11", 8: "t00"}
    dense_lut = {5: 1, 13: 1, 2: 2, 12: 2, 22: 2, 32: 2, 8: 3}
    # dense 1 built by ONE edge (iou 0.40). dense 2 (corner) built by 3 edges
    # whose ious are 0.90/0.30/0.55 -> weakest 0.30. dense 3 has no edge.
    edges = [(5, 13, 0.40),
             (2, 12, 0.90), (12, 22, 0.30), (22, 32, 0.55)]

    prov = S.compute_label_provenance(
        present_ids, gid_to_tile, offsets, dense_lut, edges=edges)

    assert prov[1]["min_match_iou"] == 0.40
    assert prov[2]["min_match_iou"] == 0.30  # the weakest of the corner's 3 merges
    assert prov[3]["min_match_iou"] is None  # clean cell: nothing for a seam to break
    # Without edges the key is present but None for every cell (back-compat).
    prov_noedge = S.compute_label_provenance(
        present_ids, gid_to_tile, offsets, dense_lut)
    assert all(p["min_match_iou"] is None for p in prov_noedge.values())


# ===========================================================================
# BLOCKING GATE 1 — one-to-many merge policy (Q3: 1:1 max-IoU)
# ===========================================================================


def test_one_to_many_merge_policy(make_tile_mask, tmp_path):
    """Local A:5 overlaps B:12 (higher IoU) AND B:13 (lower, both >= tau).

    Decided behavior (Q3, 1:1 max-IoU under-merge): 5<->12 merges; 13 stays a
    singleton (NOT merged into 5).
    """
    base = tmp_path / "tiles"
    # overlap slab = global X[12,16) (local A x[12,16), local B x[0,4)).
    # A's cell 5 fills the whole slab Y[4,12).
    tA = make_tile_mask(name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
                        labels={5: (slice(4, 12), slice(12, 16))})
    # B has two cells in the slab: 12 covers Y[4,11) (7 rows -> high IoU vs A),
    # 13 covers Y[11,12) (1 row -> lower IoU but still inter>=vmin if vmin small).
    tB = make_tile_mask(name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
                        labels={12: (slice(4, 11), slice(0, 4)),
                                13: (slice(11, 12), slice(0, 4))})
    edges, off = _edges_for_pair(tA, tB, tau=0.05, vmin=1)
    # 1:1 max-IoU: only the best match (5<->12) survives; 5 is consumed so 13
    # cannot also bind to 5.
    assert len(edges) == 1
    ga, gb, _iou = edges[0]
    assert ga == 5 + off[tA.name]
    assert gb == 12 + off[tB.name]

    # And greedy mode emits BOTH (proves the policy flag is real).
    edges_greedy, _ = _edges_for_pair(tA, tB, tau=0.05, vmin=1, one_to_many="greedy")
    matched_b = {gb for (_ga, gb, _i) in edges_greedy}
    assert {12 + off[tB.name], 13 + off[tB.name]} <= matched_b


# ===========================================================================
# BLOCKING GATE 2 — 4-way corner-only cell
# ===========================================================================


def test_four_tile_corner_only_cell(four_corner_tiles):
    """A cell present ONLY in the 4-way corner (absent from every 2-way edge slab)
    must resolve to ONE global id across all four tiles."""
    specs = four_corner_tiles
    geoms = [_geom(s.name, s.y0, s.x0, s.patch_size, s.patch_size,
                   i=s.row, j=s.col) for s in specs]
    by_name = {s.name: s for s in specs}

    nmax = {s.name: S.tile_nmax(str(s.mask_path)) for s in specs}
    off = S.compute_global_offsets(nmax)
    depth = specs[0].full_z

    all_edges = []
    all_ids = []
    for s in specs:
        all_ids += _present_global_ids(s, off)

    for (nameA, nameB, grect) in S.enumerate_adjacent_pairs(geoms):
        sA, sB = by_name[nameA], by_name[nameB]
        la = _global_to_local_rect(grect, (sA.y0, sA.x0))
        lb = _global_to_local_rect(grect, (sB.y0, sB.x0))
        all_edges += S.overlap_iou_edges(
            str(sA.mask_path), str(sB.mask_path), la, lb,
            off[nameA], off[nameB], depth, tau=0.25, vmin=1,
        )

    roots = S.union_find_merge(all_ids, all_edges)
    # The corner cell is local label 3 in every tile -> 4 global ids that must
    # collapse to one component.
    corner_globals = [3 + off[s.name] for s in specs]
    corner_roots = {roots[g] for g in corner_globals}
    assert len(corner_roots) == 1, "corner-only cell did not unify across 4 tiles"

    # And the 4 private singletons stay 4 distinct components.
    singletons = []
    for s in specs:
        local = [int(v) for v in np.unique(s.array) if v not in (0, 3)]
        singletons += [ell + off[s.name] for ell in local]
    assert len(singletons) == 4
    assert len({roots[g] for g in singletons}) == len(singletons)


# ===========================================================================
# BLOCKING GATE 3 — Z-band seam mismatch
# ===========================================================================


def test_z_band_seam_mismatch(make_tile_mask, tmp_path):
    """Shared-Z cell matches; an A-exclusive-Z cell stays a singleton."""
    base = tmp_path / "tiles"
    full_z = 64
    # Cell 3: A spans z[10,60), B spans z[5,55) -> shared z[10,55) -> match.
    # Cell 8: present ONLY in A's exclusive z[55,60) -> no co-occurrence in B -> singleton.
    tA = make_tile_mask(
        name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
        full_z=full_z, z_range=(0, full_z),
        labels={
            3: (slice(10, 60), slice(None), slice(12, 16)),   # shared cell, A's Z
            8: (slice(55, 60), slice(2, 6), slice(12, 16)),   # A-exclusive Z cell
        },
    )
    tB = make_tile_mask(
        name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
        full_z=full_z, z_range=(0, full_z),
        labels={
            3: (slice(5, 55), slice(None), slice(0, 4)),      # shared cell, B's Z
        },
    )
    edges, off = _edges_for_pair(tA, tB, tau=0.25, vmin=1)
    matched_a = {ga for (ga, _gb, _i) in edges}
    # cell 3 matches across the shared Z band...
    assert (3 + off[tA.name]) in matched_a
    # ...cell 8 (A-exclusive Z) never co-occurs in B -> not in any edge -> singleton.
    assert (8 + off[tA.name]) not in matched_a


# ===========================================================================
# End-to-end
# ===========================================================================


def test_two_tile_straddling_cell(two_overlapping_tiles):
    """End-to-end: the straddle cell collapses to one global id; 3 cells -> 3 ids."""
    t0, t1 = two_overlapping_tiles
    edges, off = _edges_for_pair(t0, t1)
    ids = _present_global_ids(t0, off) + _present_global_ids(t1, off)
    roots = S.union_find_merge(ids, edges)
    lut = S.compact_roots(roots)
    # tile0 has {1,3}, tile1 has {2,3}. Distinct cells: 1, 2, and the fused 3 = 3.
    assert len(set(lut.values())) == 3
    # The two label-3 globals share a dense id.
    assert lut[3 + off[t0.name]] == lut[3 + off[t1.name]]


def test_two_tile_no_false_merge(make_tile_mask, tmp_path):
    """Two cells that touch the seam but on disjoint voxels do NOT merge."""
    base = tmp_path / "tiles"
    # A's cell at slab Y[4,8); B's cell at slab Y[8,12) -> zero intersection.
    tA = make_tile_mask(name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
                        labels={4: (slice(4, 8), slice(12, 16))})
    tB = make_tile_mask(name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
                        labels={7: (slice(8, 12), slice(0, 4))})
    edges, off = _edges_for_pair(tA, tB, tau=0.25, vmin=1)
    assert edges == []
    ids = [4 + off[tA.name], 7 + off[tB.name]]
    roots = S.union_find_merge(ids, edges)
    assert roots[ids[0]] != roots[ids[1]]


def test_relabel_round_trip(two_overlapping_tiles, tmp_path):
    """The relabel map survives a JSON round-trip with int-key normalization."""
    t0, t1 = two_overlapping_tiles
    edges, off = _edges_for_pair(t0, t1)
    ids = _present_global_ids(t0, off) + _present_global_ids(t1, off)
    roots = S.union_find_merge(ids, edges)
    lut = S.compact_roots(roots)

    # Serialize: JSON turns int keys into strings.
    p = tmp_path / "relabel.json"
    p.write_text(json.dumps({str(k): v for k, v in lut.items()}))
    loaded = {int(k): v for k, v in json.loads(p.read_text()).items()}
    assert loaded == lut


def test_build_lut_array_and_apply(two_overlapping_tiles):
    """The dense LUT array remaps a 2D slice (global ids -> dense ids) correctly."""
    t0, t1 = two_overlapping_tiles
    edges, off = _edges_for_pair(t0, t1)
    ids = _present_global_ids(t0, off) + _present_global_ids(t1, off)
    roots = S.union_find_merge(ids, edges)
    lut = S.compact_roots(roots)
    max_global = max(ids)
    lut_arr = S.build_lut_array(off, lut, max_global)

    # Take a Z-slice from tile0 (local ids), shift to global, remap to dense.
    sl = t0.array[6]  # a 2D (Y,X) slice with local ids {0,1,3}
    glob = np.where(sl > 0, sl + off[t0.name], 0).astype(np.int64)
    dense = S.apply_relabel_to_slice(glob, lut_arr)
    # background stays 0
    assert (dense[sl == 0] == 0).all()
    # local 3 -> its dense id
    assert (dense[sl == 3] == lut[3 + off[t0.name]]).all()


def test_build_lut_array_overflow_guard():
    """build_lut_array fail-fasts if the dense count would overflow int32."""
    huge = np.iinfo(np.int32).max
    # A dense lut whose max value exceeds the int32 ceiling must assert.
    bad_lut = {1: huge + 1}
    with pytest.raises(AssertionError):
        S.build_lut_array({"t": 0}, bad_lut, max_global=1)


# ===========================================================================
# Memory invariant — concrete spy: never reads more than z_block pages at once
# ===========================================================================


def test_chunk_incremental_equivalence(make_tile_mask, tmp_path):
    """overlap_iou_edges gives IDENTICAL edges regardless of z_block — the
    accumulator's headline correctness property (block result == single-block
    result). The cells straddle multiple z-block boundaries, and their label
    magnitudes differ wildly (5 vs 60) so the per-block packing key (block_kmax)
    varies across blocks — exactly the case a re-keying bug would corrupt."""
    base = tmp_path / "tiles"
    fz = 20
    tA = make_tile_mask(name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
                        full_z=fz, z_range=(0, fz),
                        labels={5:  (slice(2, 18), slice(4, 8), slice(12, 16)),
                                60: (slice(6, 10), slice(8, 12), slice(12, 16))})
    tB = make_tile_mask(name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
                        full_z=fz, z_range=(0, fz),
                        labels={5:  (slice(2, 18), slice(4, 8), slice(0, 4)),
                                60: (slice(6, 10), slice(8, 12), slice(0, 4))})
    results = []
    for zb in (1, 3, 5, 7, 20):
        edges, _ = _edges_for_pair(tA, tB, z_block=zb)
        results.append(sorted((a, b) for a, b, _ in edges))
    assert all(r == results[0] for r in results), f"z_block-dependent edges: {results}"
    assert len(results[0]) == 2  # both straddle cells matched


def test_one_to_many_equal_iou_tiebreak(make_tile_mask, tmp_path):
    """Two B-cells overlap A:5 at the SAME IoU; the 1:1 policy must still pick
    exactly one, deterministically by the (-iou, aL, bL) tie-break -> lower bL."""
    base = tmp_path / "tiles"
    # A:5 fills the slab over Y[4,12); B:12 (Y[4,8)) and B:13 (Y[8,12)) each cover
    # half -> equal IoU 0.5. 1:1 picks one; tie-break -> bL=12.
    tA = make_tile_mask(name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
                        labels={5: (slice(4, 12), slice(12, 16))})
    tB = make_tile_mask(name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
                        labels={12: (slice(4, 8), slice(0, 4)),
                                13: (slice(8, 12), slice(0, 4))})
    edges, off = _edges_for_pair(tA, tB, tau=0.05, vmin=1)
    assert len(edges) == 1
    ga, gb, _ = edges[0]
    assert ga == 5 + off[tA.name]
    assert gb == 12 + off[tB.name]  # lower bL wins the tie


def test_memory_never_loads_full_volume(two_overlapping_tiles, monkeypatch):
    """Concrete spy: the seam reader never decodes more than ``z_block`` pages
    in a single read call (no whole full-Z slab load)."""
    import packages.epicycle_scaledseg.core.iou_stitch as mod

    z_block = 4
    max_pages_seen = {"n": 0}
    real_read = mod._read_overlap_block

    def spy_read(path, rect_local, z_start, z_stop):
        n_pages = z_stop - z_start
        max_pages_seen["n"] = max(max_pages_seen["n"], n_pages)
        # Hard ceiling: a single read must never exceed z_block pages.
        assert n_pages <= z_block, f"read {n_pages} pages > z_block={z_block}"
        block = real_read(path, rect_local, z_start, z_stop)
        # And the returned block is cropped to the overlap rect (Y/X), not the
        # whole patch page — a full-page-decode regression would fail here.
        y0, y1, x0, x1 = rect_local
        assert block.shape[1:] == (y1 - y0, x1 - x0), (
            f"block {block.shape[1:]} not cropped to rect {(y1 - y0, x1 - x0)}"
        )
        return block

    monkeypatch.setattr(mod, "_read_overlap_block", spy_read)

    t0, t1 = two_overlapping_tiles
    a = _geom("tile_0_0", t0.y0, t0.x0, t0.patch_size, t0.patch_size)
    b = _geom("tile_0_1", t1.y0, t1.x0, t1.patch_size, t1.patch_size)
    grect = S.overlap_rect(a, b)
    la = _global_to_local_rect(grect, (t0.y0, t0.x0))
    lb = _global_to_local_rect(grect, (t1.y0, t1.x0))
    nmax = {t0.name: S.tile_nmax(str(t0.mask_path)),
            t1.name: S.tile_nmax(str(t1.mask_path))}
    off = S.compute_global_offsets(nmax)
    S.overlap_iou_edges(
        str(t0.mask_path), str(t1.mask_path), la, lb,
        off[t0.name], off[t1.name], t0.full_z, z_block=z_block,
    )
    # full_z is 16 -> blocks of 4 means at least one multi-block read happened.
    assert max_pages_seen["n"] > 0
    assert max_pages_seen["n"] <= z_block

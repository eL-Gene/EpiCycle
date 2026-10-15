"""Tests for CARVE mode in cleanup_volume (split a straddling nucleus by the cell boundary).

The shipped default (`split_straddling_nuclei=False`) MAJORITY-assigns a straddling nucleus wholly
to one cell and discards the overhang — covered by test_reconcile_cleanup.py and preserved here.
CARVE mode (`split_straddling_nuclei=True`) instead keeps each cell's slice of the nucleus as that
cell's OWN nucleus (majority cell keeps the original id; other qualifying cells get fresh ids), so
a cell that would otherwise read `no_nucleus_strung_victim` gets its nucleus back and no nucleus
volume is truncated. A `min_piece_frac` guard clips a tiny overhang (registration slop) to the
majority instead of spawning a sliver.
"""

import numpy as np

from packages.epicycle_scaledseg.reconcile.cleanup import cleanup_volume


def _two_cells_x(seam, width_x, Z=6, Y=10):
    """Two cells side by side along X, abutting at column ``seam`` (cell 1 left, cell 2 right)."""
    cells = np.zeros((Z, Y, width_x), dtype=np.int32)
    cells[1:Z - 1, 1:Y - 1, 1:seam] = 1
    cells[1:Z - 1, 1:Y - 1, seam:width_x - 1] = 2
    return cells


def test_carve_splits_a_5050_straddle_into_one_nucleus_per_cell():
    # nucleus split exactly 50/50 across the 1|2 seam at x=10
    cells = _two_cells_x(seam=10, width_x=19)
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 3:7, 8:12] = 7           # cols 8,9 in cell 1 ; cols 10,11 in cell 2 -> 2/2 split

    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                 split_straddling_nuclei=True, min_piece_frac=0.2)

    # BOTH cells end up with exactly one nucleus (the whole point)
    ids_c1 = set(int(x) for x in np.unique(on[oc == 1]) if x)
    ids_c2 = set(int(x) for x in np.unique(on[oc == 2]) if x)
    assert len(ids_c1) == 1 and len(ids_c2) == 1, (ids_c1, ids_c2)
    # they are DIFFERENT ids (relabeled), and one keeps the original 7
    assert ids_c1 != ids_c2
    assert 7 in (ids_c1 | ids_c2)
    # the non-majority piece got a FRESH id above every existing label
    fresh = (ids_c1 | ids_c2) - {7}
    assert min(fresh) > max(int(cells.max()), int(nuclei.max()))
    # NO truncation: every in-cell nucleus voxel survives (clip mode would lose ~half)
    assert int((on > 0).sum()) == int(((nuclei > 0) & (cells > 0)).sum())
    # consistency + no false victim
    assert (on[oc == 0] == 0).all()
    assert rep.get("no_nucleus_strung_victim", 0) == 0
    assert rep.get("split_nucleus", 0) == 1


def test_carve_splits_a_6040_straddle():
    cells = _two_cells_x(seam=10, width_x=19)
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 3:7, 7:12] = 5           # cols 7,8,9 -> cell1 (3) ; cols 10,11 -> cell2 (2) = 60/40
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                 split_straddling_nuclei=True, min_piece_frac=0.2)
    assert len(set(int(x) for x in np.unique(on[oc == 1]) if x)) == 1
    assert len(set(int(x) for x in np.unique(on[oc == 2]) if x)) == 1
    # majority cell (cell 1, 60%) keeps the original id
    assert set(int(x) for x in np.unique(on[oc == 1]) if x) == {5}
    assert int((on > 0).sum()) == int(((nuclei > 0) & (cells > 0)).sum())


def test_carve_clips_a_tiny_overhang_below_min_piece_frac():
    # ~95/5 straddle: the 5% sliver in cell 2 is below min_piece_frac -> DISCARDED (not moved),
    # the majority keeps only its own portion, and cell 2 is an honest strung victim
    cells = np.zeros((6, 10, 30), dtype=np.int32)
    cells[1:5, 1:9, 1:20] = 1
    cells[1:5, 1:9, 20:29] = 2
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 3:7, 2:21] = 9           # cols 2..19 -> cell1 (18) ; col 20 -> cell2 (1) ~ 5%
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                 split_straddling_nuclei=True, min_piece_frac=0.2)
    # only ONE nucleus, wholly in the majority cell; cell 2 got nothing -> victim (honest)
    assert set(int(x) for x in np.unique(on[oc == 1]) if x) == {9}
    assert set(int(x) for x in np.unique(on[oc == 2]) if x) == set()
    assert rep.get("split_nucleus", 0) == 0
    assert rep.get("no_nucleus_strung_victim", 0) == 1
    # the sub-threshold sliver is DISCARDED, NOT relocated to the majority: only cell 1's own
    # portion of nucleus 9 survives, and there WAS a real cell-2 sliver that got dropped
    assert int((on > 0).sum()) == int(((nuclei == 9) & (cells == 1)).sum())
    assert int(((nuclei == 9) & (cells == 2)).sum()) > 0            # a sliver existed...
    assert int(((on == 9) & (cells == 2)).sum()) == 0              # ...and it was dropped


def test_carve_three_cell_straddle_makes_three_nuclei():
    cells = np.zeros((6, 10, 20), dtype=np.int32)
    cells[1:5, 1:9, 1:7] = 1
    cells[1:5, 1:9, 7:13] = 2
    cells[1:5, 1:9, 13:19] = 3
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 3:7, 3:17] = 4           # cell1 cols 3-6 (4), cell2 7-12 (6), cell3 13-16 (4)
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                 split_straddling_nuclei=True, min_piece_frac=0.2)
    got = {c: set(int(x) for x in np.unique(on[oc == c]) if x) for c in (1, 2, 3)}
    assert all(len(v) == 1 for v in got.values()), got            # each cell one nucleus
    assert len({next(iter(v)) for v in got.values()}) == 3        # three distinct ids
    assert int((on > 0).sum()) == int(((nuclei > 0) & (cells > 0)).sum())
    assert rep.get("split_nucleus", 0) == 2                       # two receivers got fresh nuclei


def test_carve_off_is_the_current_majority_assign_behavior():
    # default (carve off) must be unchanged: strung nucleus -> majority cell only, other = victim
    cells = _two_cells_x(seam=10, width_x=19)
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 3:7, 8:12] = 7
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                 split_straddling_nuclei=False)
    assert set(int(x) for x in np.unique(oc[on == 7]) if x) == {1}   # wholly majority cell
    assert rep.get("no_nucleus_strung_victim", 0) == 1
    assert rep.get("split_nucleus", 0) == 0


def test_carve_ids_are_globally_unique():
    cells = _two_cells_x(seam=10, width_x=19)
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 3:7, 8:12] = 7
    nuclei[2:4, 1:3, 3:6] = 8            # a second, clean nucleus fully in cell 1
    # a nucleus whose id == cells.max()+1 (== 3): an allocator that used cells.max()+1 (dropping
    # the nuclei.max() term) would hand the carved slice this same id -> collision. This pins it.
    collide_id = int(cells.max()) + 1
    nuclei[2:4, 5:7, 3:6] = collide_id
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                 split_straddling_nuclei=True, min_piece_frac=0.2)
    originals = {7, 8, collide_id}
    ids = set(int(x) for x in np.unique(on) if x)
    fresh = ids - originals
    assert len(fresh) == 1                               # exactly one carved slice
    assert fresh.pop() > max(int(cells.max()), 8)        # above EVERY existing label
    assert originals <= ids                              # all originals preserved


def test_carve_serial_matches_parallel(tmp_path):
    Z, Y, X = 6, 14, 90
    cells = np.zeros((Z, Y, X), np.int32)
    nuclei = np.zeros((Z, Y, X), np.int32)
    for i in range(13):
        cells[1:5, 1:13, 1 + i * 7:6 + i * 7] = i + 1
        if i >= 2:                                       # cells 1|2 have NO own nucleus...
            nuclei[2:4, 4:9, 2 + i * 7:5 + i * 7] = 100 + i
    nuclei[2:4, 4:9, 5:9] = 200          # ...so this straddler splits cleanly across them
    cells_p = tmp_path / "c.npy"; np.save(cells_p, cells)
    nuc_p = tmp_path / "n.npy"; np.save(nuc_p, nuclei)

    oc_s, on_s, rep_s = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                       split_straddling_nuclei=True, parallel=False)
    oc_p, on_p, rep_p = cleanup_volume(
        cells, nuclei, smooth=False, recover=False, split_straddling_nuclei=True,
        parallel=True, n_workers=2, cells_path=str(cells_p), nuclei_path=str(nuc_p))
    assert np.array_equal(oc_s, oc_p)
    assert np.array_equal(on_s, on_p)
    assert rep_s == rep_p
    assert rep_s.get("split_nucleus", 0) >= 1            # carve actually fired (not a no-op parity)


def test_carve_mass_is_conserved_across_the_split():
    # the split must neither create nor destroy in-cell nucleus voxels
    cells = _two_cells_x(seam=10, width_x=19)
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 3:7, 8:12] = 7
    in_cell = int(((nuclei > 0) & (cells > 0)).sum())
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                 split_straddling_nuclei=True, min_piece_frac=0.2)
    assert int((on > 0).sum()) == in_cell
    # and the two pieces partition that mass (disjoint, no overlap)
    piece_ids = [int(x) for x in np.unique(on) if x]
    total = sum(int((on == pid).sum()) for pid in piece_ids)
    assert total == in_cell


def test_carve_min_piece_frac_boundary_is_inclusive():
    # a slice at EXACTLY min_piece_frac qualifies (the gate is >=, not >)
    cells = np.zeros((6, 10, 20), dtype=np.int32)
    cells[1:5, 1:9, 1:9] = 1
    cells[1:5, 1:9, 9:18] = 2
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 3:7, 5:10] = 6           # cols 5,6,7,8 -> cell1 (4) ; col 9 -> cell2 (1) = exactly 0.2
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                 split_straddling_nuclei=True, min_piece_frac=0.2)
    assert len(set(int(x) for x in np.unique(on[oc == 2]) if x)) == 1   # 0.2 slice QUALIFIES
    assert rep.get("split_nucleus", 0) == 1
    assert rep.get("no_nucleus_strung_victim", 0) == 0


def test_carve_triage_rows_carry_split_nucleus():
    cells = _two_cells_x(seam=10, width_x=19)
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 3:7, 8:12] = 7
    oc, on, rep, triage = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                         split_straddling_nuclei=True, min_piece_frac=0.2,
                                         return_triage=True)
    split_rows = [r for r in triage if r["status"] == "split_nucleus"]
    assert len(split_rows) == 1
    r = split_rows[0]
    assert r["n_nuclei"] == 1 and len(r["nucleus_ids"]) == 1
    assert r["nucleus_ids"][0] > max(int(cells.max()), int(nuclei.max()))   # carried the fresh id


def test_carve_fraction_is_over_in_cell_voxels_only():
    # a nucleus straddles cell1|cell2 AND pokes into background; the fraction denominator is the
    # IN-CELL voxels only (background voxels are dropped and don't count)
    cells = np.zeros((6, 10, 24), dtype=np.int32)
    cells[1:5, 1:9, 1:9] = 1
    cells[1:5, 1:9, 9:17] = 2            # x >= 17 is background
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 3:7, 7:20] = 3           # cell1 cols 7,8 (2) ; cell2 9..16 (8) ; bg 17,18,19 (3)
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                 split_straddling_nuclei=True, min_piece_frac=0.2)
    # in-cell = 2 + 8 = 10 -> cell1 share 0.2 QUALIFIES (would be 2/13 < 0.2 if bg were counted)
    assert len(set(int(x) for x in np.unique(on[oc == 1]) if x)) == 1
    assert len(set(int(x) for x in np.unique(on[oc == 2]) if x)) == 1
    assert rep.get("split_nucleus", 0) == 1
    # only in-cell voxels survive; the 3 background voxels are gone
    assert int((on > 0).sum()) == int(((nuclei == 3) & (cells > 0)).sum())
    assert (on[oc == 0] == 0).all()


def test_carve_slice_lost_to_a_dropped_fragment_is_counted():
    # a fragment cell (< min_cell_vox) that is a carve TARGET: its slice is dropped WITH the
    # fragment and must be COUNTED (dropped_nucleus_with_fragment), never silently lost
    cells = np.zeros((6, 10, 30), dtype=np.int32)
    cells[1:5, 1:9, 1:20] = 1               # big majority cell
    cells[2:4, 4:6, 20:23] = 2              # tiny fragment cell (2*2*3 = 12 vox)
    nuclei = np.zeros_like(cells)
    nuclei[2:4, 4:6, 16:22] = 5             # cell1 cols 16-19 (4) ; cell2 cols 20,21 (2) -> 2/6=0.33
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                 split_straddling_nuclei=True, min_piece_frac=0.2,
                                 min_cell_vox=50)
    assert 2 not in np.unique(oc)                            # fragment dropped
    assert rep.get("dropped_fragment", 0) == 1
    assert rep.get("dropped_nucleus_with_fragment", 0) == 1  # the carved slice is COUNTED, not lost
    assert rep.get("split_nucleus", 0) == 0                  # no split written (target was dropped)
    assert set(int(x) for x in np.unique(on[oc == 1]) if x) == {5}   # majority keeps its own id


def _two_cells_wide():
    cells = np.zeros((10, 20, 30), dtype=np.int32)
    cells[1:9, 1:19, 1:15] = 1
    cells[1:9, 1:19, 15:29] = 2
    return cells


def test_smooth_split_off_is_byte_identical_to_carve():
    cells = _two_cells_wide()
    nuclei = np.zeros_like(cells)
    nuclei[2:8, 4:16, 8:22] = 7           # 50/50 straddle
    a = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                       split_straddling_nuclei=True, min_piece_frac=0.2, smooth_split_nuclei=False)
    b = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                       split_straddling_nuclei=True, min_piece_frac=0.2)   # default off
    assert np.array_equal(a[1], b[1]) and a[2] == b[2]       # out_nuclei + report identical


def test_smooth_split_smooths_carved_pieces_and_stays_in_cell():
    cells = _two_cells_wide()
    nuclei = np.zeros_like(cells)
    nuclei[2:8, 4:16, 8:22] = 7
    _, on0, _ = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                               split_straddling_nuclei=True, min_piece_frac=0.2, smooth_split_nuclei=False)
    oc1, on1, r1 = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                  split_straddling_nuclei=True, min_piece_frac=0.2, smooth_split_nuclei=True)
    # both cells still have exactly one nucleus; consistency holds
    assert len(set(int(x) for x in np.unique(on1[oc1 == 1]) if x)) == 1
    assert len(set(int(x) for x in np.unique(on1[oc1 == 2]) if x)) == 1
    assert (on1[oc1 == 0] == 0).all()
    # smoothing ACTUALLY changed the carved pieces (rounded the razor cut)
    assert (on1 != on0).any(), "smoothing did not change the carved pieces"
    # each nucleus id lives in ONLY its own cell — the rounded piece never crossed the wall
    for nid in [int(x) for x in np.unique(on1) if x]:
        assert len(set(int(c) for c in np.unique(oc1[on1 == nid]) if c)) == 1
    assert r1.get("carved_nucleus_pieces", 0) == 2          # both halves flagged


def test_smooth_split_is_targeted_clean_nuclei_untouched():
    cells = _two_cells_wide()
    nuclei = np.zeros_like(cells)
    nuclei[2:8, 4:16, 8:22] = 7           # split
    nuclei[3:6, 4:8, 3:7] = 8             # clean, fully inside cell 1 (away from the seam)
    _, on0, _ = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                               split_straddling_nuclei=True, min_piece_frac=0.2, smooth_split_nuclei=False)
    _, on1, _ = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                               split_straddling_nuclei=True, min_piece_frac=0.2, smooth_split_nuclei=True)
    # nucleus 8 (clean, not split) is IDENTICAL on vs off -> only carved pieces are smoothed
    assert np.array_equal(on0 == 8, on1 == 8), "a clean (non-split) nucleus was smoothed"


def test_smooth_split_triage_lists_both_carved_halves():
    cells = _two_cells_wide()
    nuclei = np.zeros_like(cells)
    nuclei[2:8, 4:16, 8:22] = 7
    oc, on, rep, triage = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                         split_straddling_nuclei=True, min_piece_frac=0.2,
                                         smooth_split_nuclei=True, return_triage=True)
    carved = sorted(cid for row in triage for cid in row["carved_nucleus_ids"])
    assert 7 in carved and len(carved) == 2 and max(carved) > 7   # original majority + fresh receiver


def test_smooth_split_no_phantom_ids():
    # every id reported in triage (nucleus_ids AND carved_nucleus_ids) must actually exist in
    # out_nuclei -> smoothing that writes nothing must NOT leave a zero-voxel phantom label
    cells = _two_cells_wide()
    nuclei = np.zeros_like(cells)
    nuclei[2:8, 4:16, 8:22] = 7
    oc, on, rep, triage = cleanup_volume(cells, nuclei, smooth=False, recover=False,
                                         split_straddling_nuclei=True, min_piece_frac=0.2,
                                         smooth_split_nuclei=True, return_triage=True)
    present = set(int(x) for x in np.unique(on))
    for row in triage:
        for nid in row["nucleus_ids"]:
            assert nid in present, f"phantom nucleus id {nid} absent from out_nuclei"
        for cid in row["carved_nucleus_ids"]:
            assert cid in present, f"phantom carved id {cid} absent from out_nuclei"
    # report count matches the real carved ids present
    assert rep.get("carved_nucleus_pieces", 0) == sum(len(r["carved_nucleus_ids"]) for r in triage)

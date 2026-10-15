"""Tests for the descoped cleanup primitives (epicycle.reconcile.cleanup).

carve_nucleus_by_cells clips a nucleus to its cell(s) and splits a strung nucleus at the seam;
recover_nucleus pulls a dim nucleus out of the nuclear stain via local rescale; fill_smooth_cell
fills holes + smooths one cell."""

import numpy as np
import pytest

from packages.epicycle_scaledseg.reconcile.cleanup import (
    carve_nucleus_by_cells, recover_nucleus, fill_smooth_cell, tidy_labels,
)


def test_carve_clips_a_sticking_out_nucleus_to_its_cell():
    cells = np.zeros((6, 10, 10), dtype=np.int32)
    cells[1:5, 1:9, 1:6] = 3                       # cell 3
    nuc = np.zeros((6, 10, 10), dtype=bool)
    nuc[2:4, 3:7, 3:9] = True                      # bulges out past the cell (x>=6 is background)
    pieces = carve_nucleus_by_cells(nuc, cells)
    assert set(pieces) == {3}                      # only the in-cell part survives
    assert (pieces[3] & ~(cells == 3)).sum() == 0  # nothing sticks out
    assert pieces[3].any()


def test_carve_splits_a_strung_nucleus_at_the_cell_seam():
    cells = np.zeros((6, 10, 20), dtype=np.int32)
    cells[1:5, 1:9, 1:10] = 3                       # cell 3 (left)
    cells[1:5, 1:9, 10:19] = 4                      # cell 4 (right), abuts at x=10
    nuc = np.zeros((6, 10, 20), dtype=bool)
    nuc[2:4, 3:7, 7:13] = True                      # straddles the 3|4 seam
    pieces = carve_nucleus_by_cells(nuc, cells)
    assert set(pieces) == {3, 4}                    # one piece per cell
    assert (pieces[3] & (cells != 3)).sum() == 0
    assert (pieces[4] & (cells != 4)).sum() == 0
    assert not (pieces[3] & pieces[4]).any()


def test_carve_orphan_nucleus_returns_empty():
    cells = np.zeros((4, 6, 6), dtype=np.int32)     # no cells at all
    nuc = np.zeros((4, 6, 6), dtype=bool); nuc[1:3, 2:4, 2:4] = True
    assert carve_nucleus_by_cells(nuc, cells) == {}


def test_recover_a_dim_nucleus():
    cell = np.zeros((8, 14, 14), dtype=bool)
    cell[1:7, 1:13, 1:13] = True
    rng = np.random.RandomState(0)
    nuc = (100.0 + rng.randn(8, 14, 14) * 3.0).astype(np.float32)   # dim cytoplasm
    nuc[3:5, 5:9, 5:9] = 130.0                       # a faint-but-real nuclear blob (+~10 sigma)
    rec = recover_nucleus(nuc, cell)
    assert rec is not None
    assert rec.sum() >= 10
    assert (rec & ~cell).sum() == 0                  # stays inside the cell


def test_recover_return_reason_explains_rejections():
    from packages.epicycle_scaledseg.reconcile.cleanup import recover_nucleus
    cell = np.zeros((8, 14, 14), dtype=bool); cell[1:7, 1:13, 1:13] = True
    rng = np.random.RandomState(0)
    # compact bright blob -> accepted, reason "ok", and default call still returns a bare mask
    nuc = (100.0 + rng.randn(8, 14, 14) * 3.0).astype(np.float32); nuc[3:5, 5:9, 5:9] = 130.0
    mask, reason = recover_nucleus(nuc, cell, return_reason=True)
    assert reason == "ok" and mask is not None
    assert recover_nucleus(nuc, cell) is not None            # default API unchanged (bare mask)
    # flat stain -> low SNR (the gate, not a silent None)
    flat = (100.0 + rng.randn(8, 14, 14) * 3.0).astype(np.float32)
    assert recover_nucleus(flat, cell, return_reason=True)[1] == "low_snr"
    # a blob filling most of the cell -> too_big (rejected as a gradient/whole-cell)
    big = np.full((8, 14, 14), 100.0, np.float32); big[1:7, 1:13, 1:13] = 200.0
    assert recover_nucleus(big, cell, return_reason=True)[1] in ("too_big", "low_snr")


def test_recover_returns_none_on_flat_cell():
    cell = np.zeros((8, 14, 14), dtype=bool); cell[1:7, 1:13, 1:13] = True
    rng = np.random.RandomState(1)
    nuc = (100.0 + rng.randn(8, 14, 14) * 3.0).astype(np.float32)   # flat, no blob
    assert recover_nucleus(nuc, cell) is None


def test_recover_rejects_a_whole_cell_gradient():
    # a smooth gradient filling most of the cell is NOT a compact nucleus -> reject (max_frac)
    cell = np.zeros((8, 14, 14), dtype=bool); cell[1:7, 1:13, 1:13] = True
    zz = np.indices((8, 14, 14))[2].astype(np.float32)
    nuc = (50.0 + zz * 10.0).astype(np.float32)      # ramp across the whole cell
    assert recover_nucleus(nuc, cell) is None


def test_fill_smooth_fills_a_hole():
    cell = np.zeros((6, 12, 12), dtype=bool)
    cell[1:5, 2:10, 2:10] = True
    cell[2:4, 5:7, 5:7] = False                       # a hole inside
    out = fill_smooth_cell(cell, smooth=False)
    assert out[3, 6, 6]                               # hole filled
    assert out.any()


def test_fill_smooth_runs_with_smoothing():
    cell = np.zeros((8, 16, 16), dtype=bool)
    cell[2:6, 3:13, 3:13] = True
    out = fill_smooth_cell(cell, sigma_z=1.0, sigma_xy=1.0, smooth=True)
    assert out.any()                                 # smoothing didn't vanish the cell
    assert out.dtype == bool


def test_cleanup_volume_makes_a_consistent_pair():
    from packages.epicycle_scaledseg.reconcile.cleanup import cleanup_volume
    Z, Y, X = 6, 12, 30
    cells = np.zeros((Z, Y, X), dtype=np.int32)
    cells[1:5, 1:11, 1:9] = 1            # cell 1 (clean)
    cells[1:5, 1:11, 9:17] = 2           # cell 2 (abuts 1) -> strung nucleus across 1|2
    cells[1:5, 1:11, 17:25] = 3          # cell 3 (merged: 2 nuclei)
    nuclei = np.zeros((Z, Y, X), dtype=np.int32)
    nuclei[2:4, 3:8, 3:7] = 10           # nucleus 10 in cell 1
    nuclei[2:4, 3:8, 7:12] = 20          # nucleus 20 straddles cell 1|2 (mostly in cell 2)
    nuclei[2:4, 3:8, 18:20] = 30         # nucleus 30 in cell 3
    nuclei[2:4, 3:8, 22:24] = 40         # nucleus 40 ALSO in cell 3 (merged cell)
    nuclei[2:4, 3:8, 26:28] = 50         # nucleus 50 OUTSIDE any cell (orphan)

    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False)

    # consistency: every output nucleus voxel is inside a cell
    assert (on[oc == 0] == 0).all(), "a nucleus sticks out of all cells"
    # the orphan nucleus (50, outside cells) is gone
    assert 50 not in np.unique(on)
    # the strung nucleus 20 is assigned WHOLLY to its majority cell (cell 2), not split/fabricated
    assert set(int(x) for x in np.unique(oc[on == 20]) if x) == {2}
    # cell 1 keeps only its own nucleus (10) — the 20-sliver intruding into it is dropped, so
    # cell 1 is NOT falsely flagged as merged
    assert set(int(x) for x in np.unique(on[oc == 1]) if x) == {10}
    # merged cell 3 keeps BOTH genuine nuclei (residual, not split)
    assert set(int(x) for x in np.unique(on[oc == 3]) if x) == {30, 40}
    assert rep.get("merged_cell", 0) == 1, rep


def test_cleanup_volume_recovers_a_missed_nucleus():
    from packages.epicycle_scaledseg.reconcile.cleanup import cleanup_volume
    cells = np.zeros((8, 14, 14), dtype=np.int32)
    cells[1:7, 1:13, 1:13] = 1           # one cell, NO nucleus label
    nuclei = np.zeros((8, 14, 14), dtype=np.int32)
    rng = np.random.RandomState(0)
    nuc_raw = (100.0 + rng.randn(8, 14, 14) * 3.0).astype(np.float32)
    nuc_raw[3:5, 5:9, 5:9] = 130.0       # a recoverable dim nuclear blob
    oc, on, rep = cleanup_volume(cells, nuclei, nuclei_raw=nuc_raw, smooth=False, recover=True)
    assert rep.get("recovered_nucleus", 0) == 1
    assert (on > 0).any()                # a nucleus was recovered + written
    assert (on[oc == 0] == 0).all()      # and it's inside the cell


def test_cleanup_volume_reports_strung_victim():
    from packages.epicycle_scaledseg.reconcile.cleanup import cleanup_volume
    cells = np.zeros((6, 10, 20), dtype=np.int32)
    cells[1:5, 1:9, 1:10] = 1
    cells[1:5, 1:9, 10:19] = 2          # cell 2 abuts cell 1 -> the victim
    nuclei = np.zeros((6, 10, 20), dtype=np.int32)
    nuclei[2:4, 3:7, 6:12] = 9          # nucleus mostly in cell 1 (4 cols) vs cell 2 (2 cols)
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False)
    assert set(int(x) for x in np.unique(oc[on == 9]) if x) == {1}   # nucleus 9 -> majority cell 1
    assert rep.get("no_nucleus_strung_victim", 0) == 1               # cell 2 lost it -> reported
    assert rep.get("no_nucleus_genuine", 0) == 0


def test_cleanup_volume_smooth_true_keeps_nuclei():
    # the smooth=True path was previously untested -> a shrink could silently drop a nucleus
    from packages.epicycle_scaledseg.reconcile.cleanup import cleanup_volume
    cells = np.zeros((8, 16, 16), dtype=np.int32)
    cells[2:6, 3:13, 3:13] = 1
    nuclei = np.zeros((8, 16, 16), dtype=np.int32)
    nuclei[3:5, 6:10, 6:10] = 5
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=True, sigma=1.0, recover=False)
    assert (on == 5).any()                      # nucleus survived smoothing
    assert (on[oc == 0] == 0).all()             # consistency holds after smoothing
    assert rep.get("clean", 0) == 1


def test_cleanup_volume_parallel_matches_serial(tmp_path):
    # the memmap-pool path (assign + clean) must be byte-identical to the serial path
    from packages.epicycle_scaledseg.reconcile.cleanup import cleanup_volume
    Z, Y, X = 6, 14, 90
    cells = np.zeros((Z, Y, X), np.int32)
    nuclei = np.zeros((Z, Y, X), np.int32)
    for i in range(13):
        cells[1:5, 1:13, 1 + i * 7:6 + i * 7] = i + 1
        nuclei[2:4, 4:9, 2 + i * 7:5 + i * 7] = 100 + i
    nuclei[2:4, 4:9, 5:9] = 200            # straddles cell 1|2
    nuclei[2:4, 4:9, 16:18] = 201          # 2nd nucleus in cell 3 -> merged
    cells_p = tmp_path / "c.npy"; np.save(cells_p, cells)
    nuc_p = tmp_path / "n.npy"; np.save(nuc_p, nuclei)

    oc_s, on_s, rep_s = cleanup_volume(cells, nuclei, smooth=False, recover=False, parallel=False)
    oc_p, on_p, rep_p = cleanup_volume(
        cells, nuclei, smooth=False, recover=False, parallel=True, n_workers=2,
        cells_path=str(cells_p), nuclei_path=str(nuc_p))
    assert np.array_equal(oc_s, oc_p)
    assert np.array_equal(on_s, on_p)
    assert rep_s == rep_p
    assert (on_p[oc_p == 0] == 0).all()


def test_majority_assign_parallel_matches_serial(tmp_path):
    from packages.epicycle_scaledseg.reconcile.cleanup import _majority_cell_per_nucleus
    Z, Y, X = 5, 12, 40
    cells = np.zeros((Z, Y, X), np.int32)
    cells[1:4, 1:11, 1:10] = 1
    cells[1:4, 1:11, 10:19] = 2            # abuts 1
    cells[1:4, 1:11, 20:30] = 3
    nuclei = np.zeros((Z, Y, X), np.int32)
    nuclei[2:3, 3:8, 3:7] = 10            # in cell 1
    nuclei[2:3, 3:8, 7:13] = 11           # straddles 1|2 (mostly 1)
    nuclei[2:3, 3:8, 22:26] = 12          # in cell 3
    nuclei[2:3, 3:8, 33:37] = 13          # orphan (no cell there)
    cells_p = tmp_path / "c.npy"; np.save(cells_p, cells)
    nuc_p = tmp_path / "n.npy"; np.save(nuc_p, nuclei)

    s = _majority_cell_per_nucleus(cells, nuclei)
    p = _majority_cell_per_nucleus(cells, nuclei, parallel=True, n_workers=2,
                                   cells_path=str(cells_p), nuclei_path=str(nuc_p))
    assert s[0] == p[0]            # nuc_to_cell
    assert s[1] == p[1]            # victim_cells
    assert s[2] == p[2] == 1       # one orphan (nucleus 13)


def test_tidy_labels_drops_fragments_and_fills():
    from packages.epicycle_scaledseg.reconcile.cleanup import tidy_labels
    labels = np.zeros((6, 12, 12), np.int32)
    labels[1:5, 1:10, 1:10] = 1                 # a real object with a hole
    labels[2:4, 4:6, 4:6] = 0                   # interior hole
    labels[0:1, 0:1, 0:1] = 2                   # a 1-voxel fragment
    out, rep = tidy_labels(labels, min_vox=8, smooth=False)
    assert 2 not in np.unique(out)              # fragment dropped
    assert 1 in np.unique(out)                  # real object kept
    assert out[3, 5, 5] == 1                    # hole filled
    assert rep["dropped_fragment"] == 1 and rep["kept"] == 1


def test_tidy_labels_parallel_matches_serial(tmp_path):
    rng = np.random.RandomState(0)
    labels = np.zeros((8, 30, 30), np.int32)
    for i in range(1, 7):
        z, y, x = rng.randint(0, 5), rng.randint(0, 24), rng.randint(0, 24)
        labels[z:z + 3, y:y + 5, x:x + 5] = i
    p = tmp_path / "labels.npy"; np.save(p, labels)
    s, rs = tidy_labels(labels, min_vox=4, smooth=False, parallel=False)
    q, rq = tidy_labels(labels, min_vox=4, smooth=False, parallel=True, n_workers=2,
                        labels_path=str(p))
    assert np.array_equal(s, q)
    assert rs == rq


def test_tidy_labels_empty():
    from packages.epicycle_scaledseg.reconcile.cleanup import tidy_labels
    out, rep = tidy_labels(np.zeros((0, 4, 4), np.int32))
    assert out.shape == (0, 4, 4) and rep == {}


def test_cleanup_volume_drops_fragments():
    from packages.epicycle_scaledseg.reconcile.cleanup import cleanup_volume
    cells = np.zeros((6, 10, 10), dtype=np.int32)
    cells[1:5, 1:9, 1:9] = 1             # a real cell
    cells[0:1, 0:1, 0:1] = 2             # a 1-voxel fragment
    nuclei = np.zeros((6, 10, 10), dtype=np.int32); nuclei[2:4, 3:7, 3:7] = 10
    oc, on, rep = cleanup_volume(cells, nuclei, smooth=False, recover=False, min_cell_vox=8)
    assert 2 not in np.unique(oc)        # fragment dropped
    assert 1 in np.unique(oc)            # real cell kept
    assert rep.get("dropped_fragment", 0) == 1

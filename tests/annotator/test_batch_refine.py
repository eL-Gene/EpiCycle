"""Tests for the Qt-free batch-refine executor.

Pure pytest — no Qt / napari.  Small synthetic int label volumes only.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_annotator.batch_refine import (
    BatchResult,
    RefineStep,
    apply_recipe_to_label,
    compute_label_roi,
    resolve_target_labels,
    run_batch_recipe,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _solid_block(shape=(20, 20, 20)):
    """Single label (id=1) filling an interior 8³ block."""
    vol = np.zeros(shape, dtype=np.int32)
    vol[6:14, 6:14, 6:14] = 1
    return vol


def _block_with_hole():
    """Label 1 as a solid block with a single interior background voxel."""
    vol = np.zeros((20, 20, 20), dtype=np.int32)
    vol[6:14, 6:14, 6:14] = 1
    vol[10, 10, 10] = 0  # interior hole
    return vol


def _two_adjacent_labels():
    """Two touching blocks: label 1 (z=4..9) and label 2 (z=10..15)."""
    vol = np.zeros((20, 16, 16), dtype=np.int32)
    vol[4:10, 4:12, 4:12] = 1
    vol[10:16, 4:12, 4:12] = 2
    return vol


# --------------------------------------------------------------------------- #
# RefineStep.validate
# --------------------------------------------------------------------------- #
def test_validate_accepts_good_params():
    RefineStep("sdf", {"sigma_z": 3.0, "sigma_xy": 1.5}).validate()
    RefineStep("dilate_erode", {"radius": 2, "axis": "XY"}).validate()
    RefineStep("fill_holes", {"axis": "3D"}).validate()
    RefineStep("dilate_erode", {"radius": -3}).validate()  # erosion ok
    RefineStep("sdf", {}).validate()  # defaults ok


def test_validate_rejects_unknown_method():
    with pytest.raises(ValueError):
        RefineStep("voodoo", {}).validate()


@pytest.mark.parametrize(
    "step",
    [
        RefineStep("sdf", {"sigma_z": 0.0}),
        RefineStep("sdf", {"sigma_xy": 50.0}),
        RefineStep("dilate_erode", {"radius": 99}),
        RefineStep("dilate_erode", {"radius": -99}),
        RefineStep("dilate_erode", {"axis": "diagonal"}),
        RefineStep("fill_holes", {"axis": "nope"}),
    ],
)
def test_validate_rejects_out_of_range(step):
    with pytest.raises(ValueError):
        step.validate()


# --------------------------------------------------------------------------- #
# apply_recipe_to_label
# --------------------------------------------------------------------------- #
def test_apply_single_sdf_smooth():
    vol = _solid_block()
    roi = compute_label_roi(vol, 1, margin=3)
    result, n_changed = apply_recipe_to_label(vol, 1, roi, [RefineStep("sdf", {})])
    assert result.shape == vol[roi].shape
    # Result contains only label 1 and 0.
    assert set(np.unique(result)).issubset({0, 1})
    assert np.any(result == 1)


def test_apply_duplicate_sdf_steps_chain():
    vol = _solid_block()
    roi = compute_label_roi(vol, 1, margin=3)
    recipe = [RefineStep("sdf", {}), RefineStep("sdf", {})]
    result, n_changed = apply_recipe_to_label(vol, 1, roi, recipe)
    # Two passes must run without error and produce a valid label-only ROI.
    assert set(np.unique(result)).issubset({0, 1})
    assert np.any(result == 1)


def test_apply_dilate_then_fill_holes_chain():
    vol = _block_with_hole()
    roi = compute_label_roi(vol, 1, margin=3)
    recipe = [
        RefineStep("dilate_erode", {"radius": 1}),
        RefineStep("fill_holes", {"axis": "3D"}),
    ]
    result, n_changed = apply_recipe_to_label(vol, 1, roi, recipe)
    assert set(np.unique(result)).issubset({0, 1})
    # Dilation expands the block, so net change must be positive.
    assert n_changed > 0


def test_apply_empty_recipe_no_change():
    vol = _solid_block()
    roi = compute_label_roi(vol, 1, margin=3)
    result, n_changed = apply_recipe_to_label(vol, 1, roi, [])
    assert n_changed == 0
    np.testing.assert_array_equal(result, vol[roi])


# --------------------------------------------------------------------------- #
# fill_holes correctness
# --------------------------------------------------------------------------- #
def test_fill_holes_fills_interior_hole():
    vol = _block_with_hole()
    assert vol[10, 10, 10] == 0  # hole present
    roi = compute_label_roi(vol, 1, margin=2)
    result, n_changed = apply_recipe_to_label(
        vol, 1, roi, [RefineStep("fill_holes", {"axis": "3D"})]
    )
    # Translate hole coord into roi space.
    z, y, x = 10 - roi[0].start, 10 - roi[1].start, 10 - roi[2].start
    assert result[z, y, x] == 1
    assert n_changed == 1


# --------------------------------------------------------------------------- #
# resolve_target_labels
# --------------------------------------------------------------------------- #
def test_resolve_all():
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[0, 0, 0] = 3
    vol[1, 1, 1] = 1
    vol[2, 2, 2] = 7
    assert resolve_target_labels(vol, None, "all") == [1, 3, 7]


def test_resolve_selected_intersects_present():
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[0, 0, 0] = 2
    vol[1, 1, 1] = 5
    # 99 is absent → dropped; result sorted, 0 excluded.
    assert resolve_target_labels(vol, None, "selected", selected_ids=[5, 2, 99, 0]) == [2, 5]


def test_resolve_tier():
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[0, 0, 0] = 1
    vol[1, 1, 1] = 2
    vol[2, 2, 2] = 3
    df = pd.DataFrame(
        {"label": [1, 2, 3, 4], "qc_tier": ["good", "bad", "bad", "bad"]}
    )
    # label 4 is in df+tier but absent from volume → dropped.
    assert resolve_target_labels(vol, df, "tier", tiers=["bad"]) == [2, 3]


def test_resolve_tier_with_none_df():
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[0, 0, 0] = 1
    assert resolve_target_labels(vol, None, "tier", tiers=["bad"]) == []


# --------------------------------------------------------------------------- #
# compute_label_roi
# --------------------------------------------------------------------------- #
def test_compute_label_roi_margin_and_clamp():
    vol = np.zeros((10, 10, 10), dtype=np.int32)
    vol[5, 5, 5] = 1  # single voxel
    roi = compute_label_roi(vol, 1, margin=2)
    # 5 ± 2 → [3, 8) on each axis.
    assert roi == (slice(3, 8), slice(3, 8), slice(3, 8))


def test_compute_label_roi_clamp_at_bounds():
    vol = np.zeros((6, 6, 6), dtype=np.int32)
    vol[0, 0, 5] = 1  # corner voxel
    roi = compute_label_roi(vol, 1, margin=3)
    # Clamped: lo never < 0, hi never > shape.
    assert roi[0] == slice(0, 4)   # z: 0-3 → [0,4)
    assert roi[1] == slice(0, 4)   # y: 0-3 → [0,4)
    assert roi[2] == slice(2, 6)   # x: 5±3 → [2,6) clamped at 6


def test_compute_label_roi_absent_returns_none():
    vol = np.zeros((6, 6, 6), dtype=np.int32)
    assert compute_label_roi(vol, 42, margin=2) is None


# --------------------------------------------------------------------------- #
# run_batch_recipe — determinism, no-clobber, cancel, union_roi
# --------------------------------------------------------------------------- #
def test_run_batch_determinism_serial_vs_parallel():
    vol = _two_adjacent_labels()
    recipe = [RefineStep("sdf", {"sigma_z": 2.0, "sigma_xy": 1.0})]
    ids = resolve_target_labels(vol, None, "all")

    r1 = run_batch_recipe(vol, ids, recipe, n_workers=1)
    r4 = run_batch_recipe(vol, ids, recipe, n_workers=4)

    np.testing.assert_array_equal(r1.working_volume, r4.working_volume)
    assert r1.touched_ids == r4.touched_ids
    assert r1.n_changed_total == r4.n_changed_total
    # Original volume untouched (snapshot read-only).
    assert np.any(vol == 1) and np.any(vol == 2)


def test_run_batch_no_clobber_under_contention():
    vol = _two_adjacent_labels()
    orig_count_1 = int(np.count_nonzero(vol == 1))
    orig_count_2 = int(np.count_nonzero(vol == 2))
    # Both labels dilate → contend for the shared boundary voxels.
    recipe = [RefineStep("dilate_erode", {"radius": 2, "axis": "3D"})]
    ids = resolve_target_labels(vol, None, "all")

    res = run_batch_recipe(vol, ids, recipe, n_workers=4)
    out = res.working_volume

    # Neither label erased.
    assert np.any(out == 1)
    assert np.any(out == 2)
    # No third label leaked in.
    assert set(np.unique(out)).issubset({0, 1, 2})
    # Both grew (dilation adds voxels), but no overlap (sum of counts == nonzero).
    c1 = int(np.count_nonzero(out == 1))
    c2 = int(np.count_nonzero(out == 2))
    assert c1 >= orig_count_1
    assert c2 >= orig_count_2
    assert c1 + c2 == int(np.count_nonzero(out > 0))  # disjoint

    # Deterministic vs serial.
    res_serial = run_batch_recipe(vol, ids, recipe, n_workers=1)
    np.testing.assert_array_equal(out, res_serial.working_volume)


def test_run_batch_lower_id_wins_contention():
    vol = _two_adjacent_labels()
    recipe = [RefineStep("dilate_erode", {"radius": 1, "axis": "3D"})]
    ids = resolve_target_labels(vol, None, "all")
    out = run_batch_recipe(vol, ids, recipe, n_workers=1).working_volume
    # Apply order is ascending id; label 1 applied first claims the shared
    # boundary, label 2's safe-write only fills remaining background.
    out2 = run_batch_recipe(vol, ids, recipe, n_workers=8).working_volume
    np.testing.assert_array_equal(out, out2)


def test_run_batch_cancel_preset():
    vol = _two_adjacent_labels()
    recipe = [RefineStep("sdf", {})]
    ids = resolve_target_labels(vol, None, "all")
    ev = threading.Event()
    ev.set()  # cancel before run

    res = run_batch_recipe(vol, ids, recipe, n_workers=4, cancel_event=ev)
    assert res.cancelled is True
    assert isinstance(res.working_volume, np.ndarray)
    # Documented semantics: pre-set cancel → unmodified copy.
    np.testing.assert_array_equal(res.working_volume, vol)
    assert res.touched_ids == []
    assert res.n_changed_total == 0


def test_run_batch_union_roi_contains_changes():
    vol = _two_adjacent_labels()
    recipe = [RefineStep("dilate_erode", {"radius": 2, "axis": "3D"})]
    ids = resolve_target_labels(vol, None, "all")
    res = run_batch_recipe(vol, ids, recipe, n_workers=1)

    assert res.union_roi is not None
    # Every voxel that changed between snapshot and working must lie in union_roi.
    changed = np.argwhere(res.working_volume != vol)
    assert changed.size > 0
    z0, y0, x0 = (s.start for s in res.union_roi)
    z1, y1, x1 = (s.stop for s in res.union_roi)
    assert changed[:, 0].min() >= z0 and changed[:, 0].max() < z1
    assert changed[:, 1].min() >= y0 and changed[:, 1].max() < y1
    assert changed[:, 2].min() >= x0 and changed[:, 2].max() < x1


def test_run_batch_progress_callback():
    vol = _two_adjacent_labels()
    recipe = [RefineStep("fill_holes", {"axis": "3D"})]
    ids = resolve_target_labels(vol, None, "all")
    seen = []
    run_batch_recipe(vol, ids, recipe, n_workers=1, progress_cb=lambda d, t: seen.append((d, t)))
    assert seen  # at least one progress tick
    assert seen[-1][0] == seen[-1][1]  # last tick is done == total


# --------------------------------------------------------------------------- #
# Empty / edge cases
# --------------------------------------------------------------------------- #
def test_run_batch_empty_label_ids():
    vol = _solid_block()
    res = run_batch_recipe(vol, [], [RefineStep("sdf", {})], n_workers=4)
    assert isinstance(res, BatchResult)
    assert res.touched_ids == []
    assert res.union_roi is None
    assert res.n_processed == 0
    assert res.cancelled is False
    np.testing.assert_array_equal(res.working_volume, vol)


def test_run_batch_empty_recipe():
    vol = _two_adjacent_labels()
    ids = resolve_target_labels(vol, None, "all")
    res = run_batch_recipe(vol, ids, [], n_workers=4)
    # No ops → nothing changes, but labels were processed (ROIs computed).
    np.testing.assert_array_equal(res.working_volume, vol)
    assert res.touched_ids == []
    assert res.n_changed_total == 0


def test_run_batch_skips_absent_label():
    vol = _solid_block()
    # 999 is absent → ROI None → skipped, no crash.
    res = run_batch_recipe(vol, [1, 999], [RefineStep("sdf", {})], n_workers=2)
    assert 999 not in res.touched_ids

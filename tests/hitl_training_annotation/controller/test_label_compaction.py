"""Tests for Step 2 label-ID compaction (2026-05-28, fix/step2-label-compaction-v2).

The bloat: `_insert_slice_mask_with_unique_labels` adds a monotonic
`base_max` offset on every insert; persisted masks carry the offset back
to disk; reload re-offsets + re-persists → IDs compound into the MILLIONS
across sessions, slowing napari's 3D label LUT (sized to max_id).

The fix: compact the 2D mask to dense 1..N IDs right before the TIFF
write in `_persist_curated_training_pair`.  DISK-ONLY — the in-memory
volume + displayed layers are untouched; the restore loop re-offsets on
reload, so views stay cross-slice-unique.

Pure-Python tests — no Qt/napari.
"""
from __future__ import annotations

import inspect
import numpy as np

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    _compact_label_ids,
    TrainingAnnotationController,
)


# ---------------------------------------------------------------------------
# _compact_label_ids helper correctness
# ---------------------------------------------------------------------------


def test_compact_preserves_background():
    mask = np.array([[0, 0], [0, 0]], dtype=np.int32)
    out = _compact_label_ids(mask)
    assert np.array_equal(out, mask)
    assert out.dtype == mask.dtype


def test_compact_renumbers_to_dense_1_to_n():
    # Bloated IDs: 0, 4000, 4000, 9999
    mask = np.array([[0, 4000], [4000, 9999]], dtype=np.int32)
    out = _compact_label_ids(mask)
    # FG values become a dense 1..2 set; bg stays 0.
    assert set(np.unique(out).tolist()) == {0, 1, 2}
    # 4000 (smaller) → 1, 9999 (larger) → 2 (sorted-unique order).
    assert out[0, 1] == 1 and out[1, 0] == 1
    assert out[1, 1] == 2


def test_compact_preserves_identity():
    # Two pixels sharing old label 500 must share the same new label.
    mask = np.array([[500, 500, 0], [800, 0, 500]], dtype=np.int32)
    out = _compact_label_ids(mask)
    # All three 500-pixels map to the same new id.
    new_for_500 = out[0, 0]
    assert out[0, 1] == new_for_500
    assert out[1, 2] == new_for_500
    # 800 maps to a different id.
    assert out[1, 0] != new_for_500
    assert out[1, 0] != 0


def test_compact_preserves_voxel_count():
    rng = np.random.default_rng(0)
    mask = (rng.integers(0, 5, size=(16, 16)) * 1000).astype(np.int32)
    out = _compact_label_ids(mask)
    assert np.count_nonzero(out) == np.count_nonzero(mask)
    # Background partition identical.
    assert np.array_equal(out == 0, mask == 0)


def test_compact_preserves_dtype_and_shape():
    mask = np.array([[0, 7], [7, 13]], dtype=np.int32)
    out = _compact_label_ids(mask)
    assert out.dtype == mask.dtype
    assert out.shape == mask.shape


def test_compact_idempotent():
    mask = np.array([[0, 3000], [3000, 6000]], dtype=np.int32)
    once = _compact_label_ids(mask)
    twice = _compact_label_ids(once)
    assert np.array_equal(once, twice)
    # Already-compact mask returns equivalent.
    compact = np.array([[0, 1], [1, 2]], dtype=np.int32)
    assert np.array_equal(_compact_label_ids(compact), compact)


def test_compact_all_background():
    mask = np.zeros((4, 4), dtype=np.int32)
    out = _compact_label_ids(mask)
    assert np.array_equal(out, mask)


def test_compact_all_foreground_no_zero():
    # Edge case: no background pixel at all.  Smallest FG must NOT collapse
    # to 0 (that would invent a background region).
    mask = np.array([[5, 5], [9, 9]], dtype=np.int32)
    out = _compact_label_ids(mask)
    assert 0 not in np.unique(out), "all-FG mask must not gain a background"
    assert set(np.unique(out).tolist()) == {1, 2}
    assert np.count_nonzero(out) == out.size


def test_compact_caps_max_id_to_cell_count():
    # The whole point: a mask with 3 cells at million-range IDs ends up
    # with max_id == 3, not millions.
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[0:5, 0:5] = 4_832_019
    mask[10:15, 10:15] = 7_001_233
    mask[20:25, 20:25] = 9_999_999
    out = _compact_label_ids(mask)
    assert int(out.max()) == 3, f"expected max_id 3, got {int(out.max())}"
    assert np.count_nonzero(out) == np.count_nonzero(mask)


# ---------------------------------------------------------------------------
# Persist-time wiring (source inspection) + DISK-ONLY guarantee
# ---------------------------------------------------------------------------


def test_persist_compacts_mask_before_write():
    src = inspect.getsource(
        TrainingAnnotationController._persist_curated_training_pair
    )
    assert "_compact_label_ids(arr_mask" in src, (
        "_persist_curated_training_pair must compact the mask before "
        "persisting the TIFF"
    )
    # The compacted mask must be the one persisted, not the raw arr_mask.
    # 2026-06-02 (Stage 2): the write was routed through the `_emit` helper
    # (async background writer for the debounce path, synchronous atomic write
    # otherwise) rather than a direct `imwrite`.  Intent unchanged: compact_mask
    # is what lands on disk.
    assert "_emit(mask_path, compact_mask)" in src


def test_persist_is_disk_only_does_not_mutate_volume_or_data_masks():
    """The compaction must NOT touch the in-memory volume or
    segmentation_manager.data_masks — only the disk write.  This is what
    keeps the in-session views (which render the offset in-memory volume)
    collision-free.  Source guard: the compaction is local to the write,
    not assigned back to data_masks."""
    src = inspect.getsource(
        TrainingAnnotationController._persist_curated_training_pair
    )
    # compact_mask is used only for the imwrite, never stored back.
    assert "data_masks[slice_label] = compact_mask" not in src
    assert "data_masks[slice_label] = _compact_label_ids" not in src


def test_insert_still_applies_cross_slice_offset():
    """Regression guard: the restore/insert path must STILL re-offset
    masks for cross-slice uniqueness.  Compaction is disk-only and relies
    on this re-offset on reload to keep the rebuilt volume collision-free.
    If this offset were removed, compact disk masks would collide across
    slices in the volume/3D view."""
    src = inspect.getsource(
        TrainingAnnotationController._insert_slice_mask_with_unique_labels
    )
    assert "base_max" in src
    assert "+ base_max" in src or "+= base_max" in src or "relabeled[fg]" in src


# ---------------------------------------------------------------------------
# _finalize_labels cross-slice uniqueness (audit 🟡 HIGH fix)
# ---------------------------------------------------------------------------


def test_finalize_applies_running_cross_slice_offset():
    """Audit fix: _finalize_labels must offset each slice so distinct cells
    across slices stay distinct in the final_labels 3D volume.  Without
    this, COMPACT data_masks (post-reload, after disk compaction) would
    collide — slice-1-cell-1 and slice-5-cell-1 both label 1 → merged in
    the 3D view."""
    src = inspect.getsource(TrainingAnnotationController._finalize_labels)
    assert "running_offset" in src, (
        "_finalize_labels must apply a running cross-slice offset when "
        "stacking masks, or compact per-slice masks will collide in 3D"
    )
    assert "_compact_label_ids" in src, (
        "_finalize_labels should normalize each slice via _compact_label_ids "
        "before offsetting (robust to compact OR offset input)"
    )


def _simulate_finalize_stack(slice_masks):
    """Reproduce the _finalize_labels stacking arithmetic in isolation
    (the loop body that builds final_stack) so we can assert cross-slice
    uniqueness without instantiating the Qt controller."""
    final_stack = []
    running_offset = 0
    for mask in slice_masks:
        mask = _compact_label_ids(np.asarray(mask, dtype=np.int32))
        fg = mask > 0
        if np.any(fg):
            mask = mask.copy()
            mask[fg] = mask[fg] + running_offset
            running_offset = int(mask.max())
        final_stack.append(mask.astype(np.int32))
    return np.stack(final_stack, axis=0)


def test_finalize_stack_no_cross_slice_collision_from_compact_masks():
    """The exact collision scenario: two slices, each COMPACT (cells start
    at 1).  After stacking with the running offset, NO label appears on
    two different slices (no merge)."""
    # Slice 0: cells 1,2.  Slice 1: cells 1,2 (compact, would collide).
    s0 = np.array([[0, 1], [2, 2]], dtype=np.int32)
    s1 = np.array([[1, 1], [0, 2]], dtype=np.int32)
    vol = _simulate_finalize_stack([s0, s1])
    # Every non-zero label must be unique to a single Z slice.
    for lbl in np.unique(vol):
        if lbl == 0:
            continue
        zs_with_label = [z for z in range(vol.shape[0]) if np.any(vol[z] == lbl)]
        assert len(zs_with_label) == 1, (
            f"label {lbl} appears on slices {zs_with_label} — cross-slice "
            "collision (cells would merge in 3D)"
        )


def test_finalize_stack_also_safe_for_offset_masks():
    """Robustness: even if data_masks holds OFFSET masks (in-session, no
    reload), the normalize-then-offset still yields cross-slice-unique
    output (no double-inflation, no collision)."""
    s0 = np.array([[0, 401], [402, 402]], dtype=np.int32)   # offset slice
    s1 = np.array([[501, 501], [0, 502]], dtype=np.int32)   # offset slice
    vol = _simulate_finalize_stack([s0, s1])
    for lbl in np.unique(vol):
        if lbl == 0:
            continue
        zs_with_label = [z for z in range(vol.shape[0]) if np.any(vol[z] == lbl)]
        assert len(zs_with_label) == 1
    # And bounded (compact-ish), not re-inflated to offsets+offsets.
    assert int(vol.max()) <= 4, f"expected bounded max, got {int(vol.max())}"


# ---------------------------------------------------------------------------
# End-to-end reload cycle — the compounding loop the fix must break
# ---------------------------------------------------------------------------
#
# The bug lives in the LOOP across sessions, not in any single method:
#   reload (re-offset every disk slice) -> persist (write IDs to disk) ->
#   reload (re-offset the already-offset disk) -> ...
# Pre-fix, persist wrote the offset IDs verbatim, so each reload stacked a
# running offset ON TOP of the prior session's offsets and max(label_id)
# compounded into the millions while any slice still held ~100 cells.
# Compaction at persist resets every disk slice to dense 1..k, so each
# reload starts clean and the rebuilt-volume max stays ~= total cell count
# cycle after cycle.
#
# The two helpers below reproduce the REAL arithmetic of the two methods in
# that loop.  Drift is guarded by the source-inspection tests above:
#   - re-offset:  test_insert_still_applies_cross_slice_offset
#   - persist:    test_persist_compacts_mask_before_write
# so we can assert the multi-cycle behavior without standing up the Qt
# controller (same approach as _simulate_finalize_stack).


def _reload_reoffset(disk_slices):
    """Restore loop, reproduced: re-insert each persisted 2D disk mask into
    a fresh 3D volume using `_insert_slice_mask_with_unique_labels`'s
    running-max offset (relabeled[fg] = raw[fg] + base_max; base_max =
    current volume max).  Single-volume (lateral XY) model — axial adds a
    cross-volume max that couples XZ/YZ but does not change the compounding
    mechanism."""
    h, w = np.asarray(disk_slices[0]).shape
    volume = np.zeros((len(disk_slices), h, w), dtype=np.int32)
    for idx, raw in enumerate(disk_slices):
        raw = np.asarray(raw, dtype=np.int32)
        base_max = int(volume.max())
        relabeled = raw.astype(np.int64, copy=True)
        fg = relabeled > 0
        relabeled[fg] = relabeled[fg] + base_max
        volume[idx] = relabeled.astype(np.int32)
    return volume


def _persist_to_disk(volume, *, compact):
    """Persist, reproduced: each Z slice is written back to 'disk'.  With
    the fix, _compact_label_ids runs before the write; pre-fix the offset
    IDs are written verbatim."""
    return [
        _compact_label_ids(volume[z]) if compact else volume[z].copy()
        for z in range(volume.shape[0])
    ]


def _run_reload_cycles(initial_disk, *, compact, cycles):
    """N load->persist->reload cycles; returns (per-cycle in-memory max,
    final volume)."""
    disk = [np.asarray(s, dtype=np.int32) for s in initial_disk]
    maxes = []
    volume = None
    for _ in range(cycles):
        volume = _reload_reoffset(disk)                    # reload = re-offset
        maxes.append(int(volume.max()))
        disk = _persist_to_disk(volume, compact=compact)   # persist
    return maxes, volume


# Three slices: 2 + 3 + 1 = 6 distinct cells, each slice COMPACT (1..k),
# as it would be right after a first compacting persist.
_INITIAL_DISK = [
    np.array([[0, 1], [2, 2]], dtype=np.int32),          # 2 cells
    np.array([[1, 2], [3, 3]], dtype=np.int32),          # 3 cells
    np.array([[0, 0], [1, 1]], dtype=np.int32),          # 1 cell
]
_TOTAL_CELLS = 6


def test_reload_cycle_with_compaction_stays_bounded():
    """The fix: across 5 reload->persist cycles the rebuilt-volume max stays
    == total cell count, never compounding."""
    maxes, _ = _run_reload_cycles(_INITIAL_DISK, compact=True, cycles=5)
    assert maxes == [_TOTAL_CELLS] * 5, (
        f"compaction must keep the rebuilt-volume max == total cells "
        f"({_TOTAL_CELLS}) every cycle; got {maxes}"
    )


def test_reload_cycle_without_compaction_compounds():
    """Meaningfulness guard: the pre-fix behavior (no compaction) MUST
    visibly compound, otherwise the bounded assertion above is a tautology
    that would pass even if compaction were a no-op."""
    maxes, _ = _run_reload_cycles(_INITIAL_DISK, compact=False, cycles=5)
    assert all(b > a for a, b in zip(maxes, maxes[1:])), (
        f"without compaction the max must compound across cycles; got {maxes}"
    )
    assert maxes[-1] > 4 * _TOTAL_CELLS, (
        f"expected runaway growth without compaction, got {maxes}"
    )


def test_reload_cycle_compact_volume_has_no_cross_slice_collision():
    """The rebuilt volume after compacting reloads keeps every cell distinct
    across slices (no two slices share a label id) — the 3D-merge failure
    mode the finalize audit also guards, here verified end-to-end."""
    _, volume = _run_reload_cycles(_INITIAL_DISK, compact=True, cycles=3)
    for lbl in np.unique(volume):
        if lbl == 0:
            continue
        zs = [z for z in range(volume.shape[0]) if np.any(volume[z] == lbl)]
        assert len(zs) == 1, (
            f"label {lbl} appears on slices {zs} — cross-slice collision"
        )
    fg_labels = [int(l) for l in np.unique(volume) if l != 0]
    assert len(fg_labels) == _TOTAL_CELLS, (
        f"expected {_TOTAL_CELLS} distinct cells, got {len(fg_labels)}"
    )


# ---------------------------------------------------------------------------
# Re-segment climb fix: skip redundant restore + zero-before-rebuild
# ---------------------------------------------------------------------------
#
# User-reported: re-segmenting slices made in-memory label IDs grow without
# bound.  Root cause: `_segment_current_slice_from_native` called
# `_ensure_work_labels_layer` on EVERY segment, whose restore loop re-inserted
# every slice onto the non-zeroed volume -> `base_max` ratcheted up each call
# (and it was a per-segment rebuild that scales with slice count -> lag).
# Fix: the segment path passes `restore_existing=False`, so a populated layer
# is reused as-is (the new mask was already inserted by `_run_segmentation`);
# the genuine restore paths (load / patch switch) zero the volume first so the
# rebuilt IDs stay bounded.


def test_ensure_layer_has_restore_existing_param():
    import inspect
    sig = inspect.signature(TrainingAnnotationController._ensure_work_labels_layer)
    assert "restore_existing" in sig.parameters, (
        "_ensure_work_labels_layer must accept restore_existing to let the "
        "segment path skip the redundant per-segment restore"
    )
    # Default True preserves the rebuild behavior for load / patch-switch callers.
    assert sig.parameters["restore_existing"].default is True


def test_segment_path_skips_redundant_restore():
    import inspect
    src = inspect.getsource(
        TrainingAnnotationController._segment_current_slice_from_native
    )
    assert "restore_existing=False" in src, (
        "the segment path must pass restore_existing=False to skip the "
        "redundant per-segment restore (the label-ID climb + lag source)"
    )


def test_ensure_layer_reuse_fastpath_and_zero_before_rebuild():
    import inspect
    src = inspect.getsource(
        TrainingAnnotationController._ensure_work_labels_layer
    )
    # Steady-state reuse fast path: skip seed+restore on an already-populated
    # layer that the caller does not want rebuilt.
    assert "not restore_existing and not layer_created" in src, (
        "must guard the reuse fast-path on (not restore_existing and not "
        "layer_created)"
    )
    # The genuine rebuild path zeros the volume first so base_max resets to 0
    # and the rebuilt IDs stay bounded (no compounding across rebuilds).
    assert "labels_layer.data = np.zeros(ref_shape" in src
    assert "layer_created" in src

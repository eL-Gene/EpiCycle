"""Controller-level tests for ``_on_annotator_batch_refined`` (Step 3 Batch
Refine accept path).

Mirrors the smooth/interp accept df-contract and guards the same class of
bug (exponential ghost growth, undo fidelity).  Qt-free: builds a bare
controller via ``__new__`` + stubs, same pattern as
``test_qc_df_duplicates.py``.

Verifies:
- each touched label → exactly ONE ``superseded`` ghost + its active row
  set ``qc_tier="curated"`` / ``feature_state="stale"``;
- accepting on a label that ALREADY has ghosts adds exactly one new ghost
  (does NOT copy prior ghosts → no exponential growth);
- a single ``_undo_last_curation`` reverts both the DataFrame and the
  mask ROI in one step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class _FakeLayer:
    def __init__(self, data):
        self.data = data
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1


def _make_controller(df, layer=None):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    from packages.epicycle_annotator.curation_undo import CurationUndoStack

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl._qc_features_df = df
    ctrl._curation_undo_stack = CurationUndoStack(max_depth=20)
    ctrl._curation_dirty = False
    # Stub everything that touches Qt / the live volume.
    ctrl._update_label_basic_stats = lambda lid: None
    ctrl._scan_qc_df_for_duplicates = lambda msg: None
    ctrl._append_log = lambda msg: None
    ctrl._refresh_curation_ui = lambda: None
    ctrl._refresh_annotator_subviewers = lambda: None
    ctrl._find_active_labels_layer = lambda: layer
    return ctrl


def test_batch_accept_marks_each_label_curated_with_one_ghost():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    df = pd.DataFrame({
        "label": [1, 2, 3],
        "qc_tier": ["review", "review", "good"],
        "label_status": ["active", "active", "active"],
        "feature_state": ["fresh", "fresh", "fresh"],
    })
    ctrl = _make_controller(df)
    FusionReconstructionController._on_annotator_batch_refined(ctrl, [1, 2])

    res = ctrl._qc_features_df
    active = res[res["label_status"] == "active"]
    superseded = res[res["label_status"] == "superseded"]

    # Labels 1,2 touched → one ghost each; label 3 untouched.
    assert len(superseded) == 2
    assert set(superseded["label"]) == {1, 2}

    amap = dict(zip(active["label"], active["qc_tier"]))
    assert amap[1] == "curated"
    assert amap[2] == "curated"
    assert amap[3] == "good"  # untouched

    smap = dict(zip(active["label"], active["feature_state"]))
    assert smap[1] == "stale"
    assert smap[2] == "stale"
    assert smap[3] == "fresh"

    assert ctrl._curation_dirty is True


def test_batch_accept_does_not_copy_prior_ghosts():
    """A label that already has 2 superseded ghosts gains exactly ONE new
    ghost on accept — not copies of the existing ghosts (the exponential
    bug guarded by the active-only filter)."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    df = pd.DataFrame({
        "label": [1, 1, 1, 9],
        "qc_tier": ["curated", "review", "review", "good"],
        "label_status": ["active", "superseded", "superseded", "active"],
        "feature_state": ["fresh", "stale", "stale", "fresh"],
    })
    ctrl = _make_controller(df)
    FusionReconstructionController._on_annotator_batch_refined(ctrl, [1])

    res = ctrl._qc_features_df
    ghosts_1 = res[(res["label"] == 1) & (res["label_status"] == "superseded")]
    active_1 = res[(res["label"] == 1) & (res["label_status"] == "active")]

    assert len(ghosts_1) == 3  # 2 prior + exactly 1 new
    assert len(active_1) == 1
    assert active_1.iloc[0]["qc_tier"] == "curated"


def test_batch_accept_keep_tier_no_reclassify_no_ghost():
    """mark_curated=False: labels keep their tier, no superseded ghost is
    created, but geometry is flagged stale."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    df = pd.DataFrame({
        "label": [1, 2, 3],
        "qc_tier": ["good", "review", "good"],
        "label_status": ["active", "active", "active"],
        "feature_state": ["fresh", "fresh", "fresh"],
    })
    ctrl = _make_controller(df)
    FusionReconstructionController._on_annotator_batch_refined(
        ctrl, [1, 2], mark_curated=False
    )
    res = ctrl._qc_features_df
    # No ghosts created, no extra rows.
    assert len(res) == 3
    assert (res["label_status"] == "active").all()
    # Tiers UNCHANGED.
    tmap = dict(zip(res["label"], res["qc_tier"]))
    assert tmap[1] == "good"
    assert tmap[2] == "review"
    assert tmap[3] == "good"
    # Touched labels flagged stale; untouched stays fresh.
    smap = dict(zip(res["label"], res["feature_state"]))
    assert smap[1] == "stale"
    assert smap[2] == "stale"
    assert smap[3] == "fresh"


def test_batch_accept_then_undo_restores_df_and_mask():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    df = pd.DataFrame({
        "label": [1, 2],
        "qc_tier": ["review", "bad"],
        "label_status": ["active", "active"],
        "feature_state": ["fresh", "fresh"],
    })
    df_before = df.copy()

    # Post-batch volume + the pre-batch crop the annotator would pass.
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[1:3, 1:3, 1:3] = 1
    layer = _FakeLayer(vol.copy())
    union = (slice(0, 4), slice(0, 4), slice(0, 4))
    old_crop = np.zeros((4, 4, 4), dtype=np.int32)  # pre-batch: empty

    ctrl = _make_controller(df, layer=layer)
    FusionReconstructionController._on_annotator_batch_refined(
        ctrl, [1, 2], roi_slices=union, old_roi_data=old_crop
    )
    assert (ctrl._qc_features_df["qc_tier"] == "curated").any()

    FusionReconstructionController._undo_last_curation(ctrl)

    restored = ctrl._qc_features_df.reset_index(drop=True)
    assert restored.equals(df_before.reset_index(drop=True))
    assert np.array_equal(layer.data[union], old_crop)
    assert layer.refreshed >= 1

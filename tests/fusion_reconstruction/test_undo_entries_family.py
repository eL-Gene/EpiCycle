"""Curation undo: every edit is ONE entry that puts the volume back.

Five bugs, one rule.  An op that pushed zero entries (a paint stroke), two
(Interpolate -> Refine -> Accept) or an entry that only held part of the
change (a Refine Further chain) left Undo restoring the wrong voxels or an
active row with no voxels behind it.  Qt-free: the controller is built with
``__new__`` and stubs, a real undo stack and a real DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "curation_3d_patch"
ROI = (slice(0, 6), slice(0, 8), slice(0, 8))


class _Layer:
    def __init__(self, data):
        self.name = NAME
        self.data = data

    def refresh(self):
        pass


def _df(labels):
    return pd.DataFrame({
        "label": list(labels),
        "qc_tier": ["review"] * len(labels),
        "feature_state": ["fresh"] * len(labels),
        "label_status": ["active"] * len(labels),
        "area": [10] * len(labels),
        "centroid-0": [1.0] * len(labels),
        "centroid-1": [1.0] * len(labels),
        "centroid-2": [1.0] * len(labels),
    })


def _ctrl(vol, labels, tmp_path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    from packages.epicycle_annotator.curation_undo import CurationUndoStack

    c = object.__new__(FusionReconstructionController)
    c.working_dir = str(tmp_path)
    c._qc_view_patch_name = "p01.tif"
    c._qc_features_df = _df(labels)
    c._curation_undo_stack = CurationUndoStack(max_depth=20)
    c._curation_dirty = False
    c._tiers_dirty = False
    c._curation_dirty_epoch = 0
    c._run_epoch = 1
    c._qc_settings = {"spacing": None, "spacing_values": (1.0, 1.0, 1.0)}
    c._current_downsample_factor = lambda *a, **k: 1
    c._append_log = lambda m: None
    c._refresh_curation_ui = lambda: None
    layer = _Layer(vol)
    c._find_active_labels_layer = lambda: layer
    c._refresh_annotator_subviewers = lambda: None
    c._on_annotator_reshape_context = lambda: {
        "workspace": str(tmp_path), "patch_name": "p01", "run_id": None}
    c.layer = layer
    return c


def _active(df):
    return set(df[df["label_status"] == "active"]["label"])


# ---------------------------------------------------------------- F10 ------
def test_refine_further_chain_is_one_undo(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    vol[1:5, 1:7, 1:7] = 250
    original = vol.copy()
    c = _ctrl(vol, [250], tmp_path)

    # fragment entry writes 686/687 into the volume (no entry), a fragment
    # smooth accept records its snapshot as a layer, then Accept all.
    vol[1:5, 1:7, 1:4] = 686
    vol[1:5, 1:7, 4:7] = 687
    frag_snap = vol[ROI].copy()
    vol[1:5, 1:7, 1:3] = 0  # the smooth
    c._on_annotator_reshape_committed(
        250, [686, 687], roi_slices=ROI, old_roi_data=original[ROI].copy(),
        pre_layers=[(ROI, frag_snap)],
    )
    assert len(c._curation_undo_stack) == 1
    c._undo_last_curation()
    assert np.array_equal(c.layer.data, original)
    assert _active(c._qc_features_df) == {250}


# ---------------------------------------------------------------- F18 ------
def test_interpolate_refine_accept_is_one_entry_and_one_ghost(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    keyframes = vol.copy()
    keyframes[0, 2:4, 2:4] = 431
    keyframes[5, 2:4, 2:4] = 431
    vol[:] = keyframes
    c = _ctrl(vol, [431], tmp_path)
    interp_snap = keyframes[ROI].copy()
    vol[1:5, 2:4, 2:4] = 431           # the interpolation fill
    smooth_snap = vol[ROI].copy()
    vol[1:5, 2:4, 2:4] = 0             # the refine

    c._on_annotator_interpolation_accepted(
        431, roi_slices=ROI, old_roi_data=interp_snap,
        pre_layers=[(ROI, smooth_snap)], steps=[], recipes=[],
    )
    df = c._qc_features_df
    assert len(c._curation_undo_stack) == 1
    assert (df["label"] == 431).sum() == 2
    c._undo_last_curation()
    assert np.array_equal(c.layer.data, keyframes)
    assert (c._qc_features_df["label"] == 431).sum() == 1


def test_pre_layers_restore_newest_first(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    c = _ctrl(vol, [1], tmp_path)
    old, mid, new = [np.full(vol[ROI].shape, v, np.int32) for v in (1, 2, 3)]
    vol[ROI] = 9
    c._commit_curation_action(
        "smooth", 1, "x", roi_slices=ROI, old_roi_data=old,
        extra={"pre_layers": [(ROI, mid), (ROI, new)]},
    )
    c._undo_last_curation()
    assert (c.layer.data == 1).all()   # the oldest wins, the parent is last


# ---------------------------------------------------------------- F4 -------
def test_commit_with_push_false_pushes_nothing(tmp_path):
    c = _ctrl(np.zeros((6, 8, 8), np.int32), [1], tmp_path)
    c._commit_curation_action("delete", 1, "x", push=False)
    assert len(c._curation_undo_stack) == 0
    assert c._curation_dirty is True


def test_reconcile_delete_marks_row_without_an_entry(tmp_path):
    c = _ctrl(np.zeros((6, 8, 8), np.int32), [633, 4], tmp_path)
    c._on_annotator_label_deleted(633, reconcile=True)
    assert len(c._curation_undo_stack) == 0
    assert _active(c._qc_features_df) == {4}


def test_label_deleted_with_only_a_superseded_row_changes_nothing(tmp_path):
    c = _ctrl(np.zeros((6, 8, 8), np.int32), [5], tmp_path)
    c._qc_features_df["label_status"] = "superseded"
    before = c._qc_features_df.copy()
    c._on_annotator_label_deleted(5)
    c._on_annotator_label_deleted(5, reconcile=True)
    assert c._qc_features_df.equals(before)
    assert len(c._curation_undo_stack) == 0


# ---------------------------------------------------------------- F17 ------
def test_paint_stroke_entry_undo_zeroes_the_plane(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    vol[2, 1:3, 1:3] = 91
    c = _ctrl(vol, [91], tmp_path)
    roi = (3, slice(None), slice(None))
    snap = vol[roi].copy()
    vol[3, 4:6, 4:6] = 91
    c._on_annotator_paint_stroke(91, roi_slices=roi, old_roi_data=snap)
    assert len(c._curation_undo_stack) == 1
    entry = c._curation_undo_stack.peek()
    assert entry.action_type == "paint_stroke"
    assert entry.roi_slices == roi
    c._undo_last_curation()
    assert not (c.layer.data[3] == 91).any()
    assert (c.layer.data[2] == 91).any()


def test_consecutive_strokes_same_label_and_plane_coalesce(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    c = _ctrl(vol, [91], tmp_path)
    roi = (3, slice(None), slice(None))
    first = vol[roi].copy()
    vol[3, 1, 1] = 91
    c._on_annotator_paint_stroke(91, roi_slices=roi, old_roi_data=first)
    second = vol[roi].copy()
    vol[3, 2, 2] = 91
    c._on_annotator_paint_stroke(91, roi_slices=roi, old_roi_data=second)
    assert len(c._curation_undo_stack) == 1
    assert c._curation_undo_stack.peek().old_mask_roi is first
    # a different plane of the same label joins the entry
    c._on_annotator_paint_stroke(
        91, roi_slices=(4, slice(None), slice(None)),
        old_roi_data=vol[4].copy())
    assert len(c._curation_undo_stack) == 1


def test_paint_stroke_is_mask_mutating():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    assert "paint_stroke" in C._MASK_MUTATING_ACTIONS


# ---------------------------------------------------------------- F23 ------
def test_new_label_with_plane_roi_undo_removes_voxels(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    c = _ctrl(vol, [4], tmp_path)
    roi = (3, slice(None), slice(None))
    snap = vol[roi].copy()
    vol[3, 2:4, 2:4] = 687
    c._on_annotator_new_label_painted(
        687, roi_slices=roi, old_roi_data=snap)
    assert 687 in set(c._qc_features_df["label"])
    c._undo_last_curation()
    assert not (c.layer.data == 687).any()
    assert 687 not in set(c._qc_features_df["label"])


# ------------------------------------------------------------ M2 / M1 ------
def test_strokes_on_many_planes_fold_into_one_entry(tmp_path):
    vol = np.zeros((30, 8, 8), np.int32)
    c = _ctrl(vol, [91], tmp_path)
    for z in range(25):
        roi = (z, slice(None), slice(None))
        snap = vol[roi].copy()
        vol[z, 2:4, 2:4] = 91
        c._on_annotator_paint_stroke(91, roi_slices=roi, old_roi_data=snap)
    assert len(c._curation_undo_stack) == 1
    c._undo_last_curation()
    assert not (c.layer.data == 91).any()


def test_fold_across_planes_keeps_the_oldest_snapshot_per_plane(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    c = _ctrl(vol, [91], tmp_path)
    roi = (1, slice(None), slice(None))
    first = vol[roi].copy()
    vol[1, 1, 1] = 91
    c._on_annotator_paint_stroke(91, roi_slices=roi, old_roi_data=first)
    c._on_annotator_paint_stroke(
        91, roi_slices=(2, slice(None), slice(None)), old_roi_data=vol[2].copy())
    vol[1, 2, 2] = 91
    c._on_annotator_paint_stroke(91, roi_slices=roi, old_roi_data=vol[roi].copy())
    top = c._curation_undo_stack.peek()
    assert top.old_mask_roi is first
    assert len(top.extra["pre_layers"]) == 1


def test_a_different_label_is_a_new_entry(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    c = _ctrl(vol, [91, 92], tmp_path)
    roi = (1, slice(None), slice(None))
    c._on_annotator_paint_stroke(91, roi_slices=roi, old_roi_data=vol[roi].copy())
    c._on_annotator_paint_stroke(92, roi_slices=roi, old_roi_data=vol[roi].copy())
    assert len(c._curation_undo_stack) == 2


def test_a_fold_does_not_refresh_the_whole_curation_ui(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    c = _ctrl(vol, [91], tmp_path)
    calls = []
    c._refresh_curation_ui = lambda: calls.append(1)
    roi = (1, slice(None), slice(None))
    c._on_annotator_paint_stroke(91, roi_slices=roi, old_roi_data=vol[roi].copy())
    assert len(calls) == 1
    c._on_annotator_paint_stroke(91, roi_slices=roi, old_roi_data=vol[roi].copy())
    c._on_annotator_paint_stroke(
        91, roi_slices=(2, slice(None), slice(None)), old_roi_data=vol[2].copy())
    assert len(calls) == 1
    assert c._curation_dirty is True


# ------------------------------------------------------------- MED-1 -------
def test_reconcile_without_a_qc_table_still_sets_dirty(tmp_path):
    c = _ctrl(np.zeros((6, 8, 8), np.int32), [1], tmp_path)
    c._qc_features_df = None
    c._note_commit_without_qc = lambda *_a, **_k: None
    c._on_annotator_label_deleted(633, reconcile=True)
    assert c._curation_dirty is True
    assert len(c._curation_undo_stack) == 0


def test_reconcile_of_a_superseded_only_label_sets_dirty_but_edits_no_row(tmp_path):
    c = _ctrl(np.zeros((6, 8, 8), np.int32), [5], tmp_path)
    c._qc_features_df["label_status"] = "superseded"
    before = c._qc_features_df.copy()
    c._on_annotator_label_deleted(5, reconcile=True)
    assert c._qc_features_df.equals(before)
    assert c._curation_dirty is True


# ------------------------------------------------------------- LOW-1 -------
def test_interpolation_stats_roi_covers_the_pre_layers(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    vol[2:4, 2:4, 2:4] = 431
    c = _ctrl(vol, [431], tmp_path)
    seen = []
    c._update_label_basic_stats = lambda lid, *, roi_slices=None: seen.append(
        roi_slices)
    small = (slice(1, 5), slice(1, 5), slice(1, 5))
    grown = (slice(0, 6), slice(0, 8), slice(1, 7))
    c._on_annotator_interpolation_accepted(
        431, roi_slices=small, old_roi_data=vol[small].copy(),
        pre_layers=[(grown, vol[grown].copy())])
    assert seen == [(slice(0, 6), slice(0, 8), slice(1, 7))]


# ------------------------------------------------------------- MED-5 -------
def test_clear_edits_pops_the_stroke_entries_and_undo_hits_the_op_before(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    vol[1:3, 1:3, 1:3] = 91
    c = _ctrl(vol, [91, 4], tmp_path)
    c._on_annotator_label_deleted(4, roi_slices=ROI, old_roi_data=vol[ROI].copy())
    assert len(c._curation_undo_stack) == 1
    for z in (0, 4):
        roi = (z, slice(None), slice(None))
        snap = vol[roi].copy()
        vol[z, 5, 5] = 91
        c._on_annotator_paint_stroke(91, roi_slices=roi, old_roi_data=snap)
    c._on_annotator_paint_stroke(
        91, roi_slices=(5, slice(None), slice(None)),
        old_roi_data=vol[5].copy())
    c._on_annotator_paint_stroke(91, discard=True)
    assert len(c._curation_undo_stack) == 1
    assert c._curation_undo_stack.peek().action_type == "delete"


def test_discard_leaves_other_labels_entries_alone(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    c = _ctrl(vol, [91, 92], tmp_path)
    roi = (1, slice(None), slice(None))
    c._on_annotator_paint_stroke(92, roi_slices=roi, old_roi_data=vol[roi].copy())
    c._on_annotator_paint_stroke(91, discard=True)
    assert len(c._curation_undo_stack) == 1


# ------------------------------------------------------ MED-a / MED-b ------
def test_clear_edits_pops_only_the_current_draw_sessions_entry(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    c = _ctrl(vol, [7], tmp_path)
    roi = (1, slice(None), slice(None))
    vol[1, 1, 1] = 7
    c._on_annotator_paint_stroke(
        7, roi_slices=roi, old_roi_data=np.zeros((8, 8), np.int32),
        draw_session=1)
    # a second session on the same label and plane must not fold into it
    c._on_annotator_paint_stroke(
        7, roi_slices=roi, old_roi_data=vol[roi].copy(), draw_session=2)
    assert len(c._curation_undo_stack) == 2
    c._on_annotator_paint_stroke(7, discard=True, draw_session=2)
    assert len(c._curation_undo_stack) == 1
    c._undo_last_curation()
    assert not (c.layer.data == 7).any()


def test_clear_edits_on_a_new_label_leaves_no_row_of_it(tmp_path):
    vol = np.zeros((6, 8, 8), np.int32)
    c = _ctrl(vol, [4], tmp_path)
    roi = (3, slice(None), slice(None))
    snap = vol[roi].copy()
    vol[3, 2:4, 2:4] = 687
    c._on_annotator_new_label_painted(
        687, roi_slices=roi, old_roi_data=snap, draw_session=1)
    assert 687 in set(c._qc_features_df["label"])
    c._on_annotator_paint_stroke(687, discard=True, draw_session=1)
    assert 687 not in set(c._qc_features_df["label"])
    assert len(c._curation_undo_stack) == 0


# -------------------------------------------------------- re-isolate -------
class _IsoWidget:
    def __init__(self, live):
        self.live = frozenset(live)
        self.calls = []
        self._isolation_bridge_dispatched = False

    def _iso_live_ids(self):
        return self.live

    def _on_isolation_changed(self, ids, **kw):
        self.calls.append((set(ids), self._isolation_bridge_dispatched))


def _reshape_entry_ctrl(tmp_path, live):
    vol = np.zeros((6, 8, 8), np.int32)
    vol[1:5, 1:7, 1:7] = 250
    c = _ctrl(vol, [250], tmp_path)
    vol[1:5, 1:7, 1:4] = 11
    vol[1:5, 1:7, 4:7] = 12
    c._on_annotator_reshape_committed(
        250, [11, 12], roi_slices=ROI, old_roi_data=np.where(
            vol[ROI] > 0, 250, 0).astype(np.int32))
    widget = _IsoWidget(live)
    c._get_curation_annotator_widget = lambda: widget
    return c, widget


def test_undoing_a_reshape_isolates_the_original_again(tmp_path):
    c, widget = _reshape_entry_ctrl(tmp_path, {11, 12})
    c._undo_last_curation()
    assert widget.calls == [({250}, True)]
    assert widget._isolation_bridge_dispatched is False


def test_undo_keeps_the_other_isolated_cells(tmp_path):
    c, widget = _reshape_entry_ctrl(tmp_path, {11, 12, 40})
    c._undo_last_curation()
    assert widget.calls == [({250, 40}, True)]


def test_undo_leaves_an_unrelated_isolation_alone(tmp_path):
    c, widget = _reshape_entry_ctrl(tmp_path, {40, 41})
    c._undo_last_curation()
    assert widget.calls == []


def test_undo_selects_the_rows_of_what_it_isolates(tmp_path):
    c, widget = _reshape_entry_ctrl(tmp_path, {11, 12, 40})
    rows = []
    widget._navigator_widget = type("N", (), {
        "select_rows_silently": lambda self, ids: rows.append(set(ids))})()
    c._undo_last_curation()
    assert rows == [{250, 40}]


def test_undo_selects_the_restored_row_after_the_table_refresh(tmp_path):
    c, widget = _reshape_entry_ctrl(tmp_path, {11, 12})
    order = []
    c._refresh_curation_ui = lambda: order.append("refresh")
    widget._navigator_widget = type("N", (), {
        "select_rows_silently":
            lambda self, ids: order.append(("select", set(ids)))})()
    c._undo_last_curation()
    assert order == ["refresh", ("select", {250})]

"""The discard callback that Add's Clear objects commits through.

* it marks the ACTIVE rows of the cleared ids deleted; an older superseded
  row with the same id keeps its status;
* the caller's undo text and extra fields reach the undo entry, and the
  navigator's call (no text) keeps today's words;
* ``_live_add_clear_seqs`` lists the Clear entries still on the undo stack;
* ``_add_kept_tier_count`` counts the active rows marked Good or curated;
* a re-extraction keeps one removed row per id that Clear objects removed
  in this session, and per removed id above the fresh table's highest id
  (its deleted row, else its superseded one, the status kept), so a
  cleared id is never handed out again.

Qt-free: the controller is built with ``__new__``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from napari_epicycle.fusion_reconstruction import controller as CM
from packages.epicycle_annotator.curation_undo import CurationUndoStack

from .test_step3_commit_before_qc_guard import OLD_ROI, ROI, _make_controller


def _df():
    return pd.DataFrame({
        "label": [7, 8, 7, 9],
        "qc_tier": ["good", "review", "review", "curated"],
        "feature_state": ["fresh"] * 4,
        "label_status": ["active", "active", "superseded", "active"],
        "area": [10, 11, 12, 13],
    })


def _ctrl(df=None):
    c = _make_controller(df=_df() if df is None else df)
    c._curation_undo_stack = CurationUndoStack(max_depth=20)
    c._qc_table_patch = "P"
    c._qc_view_patch_name = "P"
    c._active_run_path_id = lambda: None
    return c


def test_only_active_rows_are_marked_deleted():
    c = _ctrl()
    c._on_annotator_discard_garbage([7, 8], roi_slices=ROI,
                                    old_roi_data=OLD_ROI)
    df = c._qc_features_df
    assert list(df["label_status"]) == ["deleted", "deleted", "superseded",
                                        "active"]


def test_the_callers_words_and_extra_reach_the_entry():
    c = _ctrl()
    c._on_annotator_discard_garbage(
        [7], roi_slices=ROI, old_roi_data=OLD_ROI,
        description="Cleared 1 object (Add)", extra={"add_clear_seq": 4})
    e = c._curation_undo_stack.peek()
    assert e.action_type == "discard_garbage"
    assert e.description == "Cleared 1 object (Add)"
    assert e.extra["add_clear_seq"] == 4
    assert e.extra["garbage_ids"] == [7]


def test_the_navigators_call_keeps_its_words():
    c = _ctrl()
    c._on_annotator_discard_garbage([7, 8], roi_slices=ROI,
                                    old_roi_data=OLD_ROI)
    e = c._curation_undo_stack.peek()
    assert e.description == "Discard 2 garbage labels"
    assert "add_clear_seq" not in e.extra
    assert c._live_add_clear_seqs() == set()


def test_live_clear_seqs_follow_the_stack():
    c = _ctrl()
    c._on_annotator_discard_garbage([7], roi_slices=ROI, old_roi_data=OLD_ROI,
                                    extra={"add_clear_seq": 1})
    c._on_annotator_discard_garbage([8], roi_slices=ROI, old_roi_data=OLD_ROI,
                                    extra={"add_clear_seq": 2})
    assert c._live_add_clear_seqs() == {1, 2}
    c._curation_undo_stack.pop()
    assert c._live_add_clear_seqs() == {1}
    c._curation_undo_stack.clear()
    assert c._live_add_clear_seqs() == set()


def test_live_clear_seqs_without_a_stack_is_empty():
    c = _ctrl()
    c._curation_undo_stack = None
    assert c._live_add_clear_seqs() == set()


def test_the_kept_tier_count():
    c = _ctrl()
    assert c._add_kept_tier_count([7, 8, 9]) == 2      # 7 good, 9 curated
    assert c._add_kept_tier_count([8]) == 0
    c._qc_features_df = None
    assert c._add_kept_tier_count([7]) == 0


# --------------------------------------------------------------------------- #
# re-extraction keeps the cleared ids
# --------------------------------------------------------------------------- #
def test_a_clear_records_its_ids_for_the_patch_and_run():
    c = _ctrl()
    c._on_annotator_discard_garbage([7, 9], roi_slices=ROI,
                                    old_roi_data=OLD_ROI,
                                    extra={"add_clear_seq": 1})
    c._on_annotator_discard_garbage([8], roi_slices=ROI,
                                    old_roi_data=OLD_ROI)  # not a Clear
    assert c._add_cleared_ids_for("P", None) == {7, 9}
    assert c._add_cleared_ids_for("Q", None) == set()


def test_cleared_rows_are_carried_one_per_id():
    prev = pd.DataFrame({
        "label": [3, 7, 7, 9, 12, 40],
        "qc_tier": ["good", "good", "review", "curated", "bad", "review"],
        "label_status": ["active", "deleted", "deleted", "deleted",
                         "deleted", "superseded"],
        "area": [5, 6, 7, 8, 9, 10],
        "old_only": [1, 2, 3, 4, 5, 6],
    })
    fresh = pd.DataFrame({"label": [3, 9], "label_status": ["active"] * 2,
                          "area": [5.0, 8.0], "new_col": [0.1, 0.2]})
    out = CM._carry_cleared_rows(prev, fresh, {7, 9, 40, 55})
    # 7: one deleted row (the last); 9 is active again (it is in the fresh
    # table); 40 is kept as superseded (a removed id holds the floor whatever
    # its inactive status); 55 has no row at all; 12 was not cleared
    assert list(out["label"]) == [3, 9, 7, 40]
    carried = out[out["label"] == 7].iloc[0]
    assert carried["label_status"] == "deleted"
    assert carried["area"] == 7
    assert out[out["label"] == 40].iloc[0]["label_status"] == "superseded"
    assert list(out.columns) == list(fresh.columns), "no stale column"
    assert CM._carry_cleared_rows(None, fresh, {7}) is fresh
    assert CM._carry_cleared_rows(prev, fresh, set()) is fresh


def test_the_floor_survives_a_reextract_through_the_carried_rows():
    prev = pd.DataFrame({"label": [3, 50], "label_status": ["active",
                                                            "deleted"]})
    fresh = pd.DataFrame({"label": [3], "label_status": ["active"]})
    out = CM._carry_cleared_rows(prev, fresh, {50})
    c = _ctrl(out)
    assert c._add_label_id_floor() == 50
    assert np.int64(out["label"].max()) == 50


def test_a_row_with_no_status_counts_as_active():
    """One rule for "active" in the discard and in ``_mark_ids_deleted``:
    every status but deleted and superseded."""
    df = pd.DataFrame({"label": [7, 8, 9],
                       "label_status": [np.nan, "superseded", "active"],
                       "area": [1, 2, 3]})
    c = _ctrl(df)
    c._on_annotator_discard_garbage([7, 8, 9], roi_slices=ROI,
                                    old_roi_data=OLD_ROI)
    assert list(c._qc_features_df["label_status"]) == [
        "deleted", "superseded", "deleted"]


# --------------------------------------------------------------------------- #
# the id floor with no table, across a switch away and back
# --------------------------------------------------------------------------- #
def test_the_floor_remembers_a_clear_with_no_table_across_a_switch():
    c = _ctrl()
    c._qc_features_df = None
    c._qc_table_patch = ""
    c._on_annotator_discard_garbage([7, 60], roi_slices=ROI,
                                    old_roi_data=OLD_ROI,
                                    extra={"add_clear_seq": 3})
    assert c._add_label_id_floor() == 60
    c._qc_view_patch_name = "Q"             # another patch in view
    assert c._add_label_id_floor() == 0
    c._qc_view_patch_name = "P"             # back
    assert c._add_label_id_floor() == 60
    # a table that came back without the cleared ids (an extraction that
    # finished after the switch) does not lower it
    c._qc_features_df = pd.DataFrame({"label": [3, 9],
                                      "label_status": ["active"] * 2})
    c._qc_table_patch = "P"
    assert c._add_label_id_floor() == 60


# --------------------------------------------------------------------------- #
# the re-extract merge itself (the real handler, not only the helper)
# --------------------------------------------------------------------------- #
def _reextract(prev, fresh, tmp_path, record=None):
    from .test_qc_tier_preservation import _make_bare_controller

    ctrl = _make_bare_controller(prev_df=prev, working_dir=tmp_path)
    if record is not None:
        ctrl._add_cleared_ids = record
    CM.FusionReconstructionController._on_qc_extract_finished(
        ctrl, fresh, "p.tif")
    return ctrl


def _fresh(labels):
    return pd.DataFrame({"label": labels, "area": [10] * len(labels),
                         "qc_score": [0.5] * len(labels)})


def test_a_reextract_keeps_the_deleted_rows_above_the_highest_id(tmp_path):
    """Clear all, Save, close napari, reopen, Extract: the deleted rows the
    saved table holds above the fresh table's highest id are kept -- with
    no session record -- and written by the auto-save, so the next
    re-extraction keeps them again and no removed id is handed out."""
    prev = pd.DataFrame({
        "label": [3, 5, 50, 60, 60],
        "qc_tier": ["good", "review", "bad", "review", "good"],
        "label_status": ["active", "deleted", "deleted", "deleted",
                         "superseded"],
        "area": [1, 2, 3, 4, 5],
    })
    ctrl = _reextract(prev, _fresh([3, 10]), tmp_path)
    df = ctrl._qc_features_df
    ghosts = df[df["label_status"] != "active"]
    assert sorted(ghosts["label"]) == [50, 60]
    assert set(ghosts["label_status"]) == {"deleted"}
    assert 5 not in set(df["label"]), "below the highest id: dropped"
    assert ctrl._add_label_id_floor() == 60
    on_disk = pd.read_csv(next(tmp_path.rglob("p_features.csv")))
    assert sorted(on_disk.loc[on_disk["label_status"] == "deleted",
                              "label"]) == [50, 60]
    # a restart: a new controller, the saved table, no session record
    again = _reextract(on_disk, _fresh([3, 10]), tmp_path)
    assert again._add_label_id_floor() == 60


def test_a_reextract_keeps_the_rows_this_sessions_clears_removed(tmp_path):
    prev = pd.DataFrame({
        "label": [3, 7, 9],
        "qc_tier": ["good", "review", "good"],
        "label_status": ["active", "deleted", "active"],
        "area": [1, 2, 3],
    })
    # 7 is below the fresh table's highest id: only the record keeps it
    ctrl = _reextract(prev, _fresh([3, 9]), tmp_path,
                      record={("p.tif", None): {7}})
    df = ctrl._qc_features_df
    assert df.loc[df["label"] == 7, "label_status"].tolist() == ["deleted"]
    plain = _reextract(prev, _fresh([3, 9]), tmp_path / "b")
    assert 7 not in set(plain._qc_features_df["label"])

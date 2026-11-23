"""Commit Triage deletes only labels whose ACTIVE row is garbage, and the
tier thresholds never re-tier a ghost row or a row with no finite score."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)
from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers


def _ctrl(tmp_path, df):
    c = C.__new__(C)
    c.working_dir = str(tmp_path)
    c._qc_view_patch_name = "A.tif"
    c._selected_curation_patch = lambda: ""
    c._selected_patch_name = lambda: "A.tif"
    c._append_log = lambda t: None
    c.ui = type("U", (), {})()
    c._qc_features_df = df
    zeroed = []
    c._zeroed = zeroed
    c._zero_labels_in_viewer = lambda ids: zeroed.append(set(ids))
    c._get_active_labels_path = lambda: ""
    c._find_active_labels_layer = lambda: None
    return c


def _prep(c, monkeypatch):
    class _SM:
        def get_active_run_id(self):
            return "default"
    c.state_manager = _SM()
    return c._commit_triage_prepare()


def test_a_superseded_garbage_row_does_not_delete_the_active_label(
        tmp_path, monkeypatch):
    df = pd.DataFrame({
        "label": [7, 7],
        "qc_tier": ["curated", "garbage"],
        "label_status": ["active", "superseded"],
    })
    c = _ctrl(tmp_path, df)
    prep = _prep(c, monkeypatch)
    assert prep["persist_kwargs"]["garbage_ids"] == set()
    assert c._zeroed == []
    assert list(c._qc_features_df["label_status"]) == ["active", "superseded"]


def test_an_active_garbage_row_is_still_deleted(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "label": [7, 8],
        "qc_tier": ["garbage", "good"],
        "label_status": ["active", "active"],
    })
    c = _ctrl(tmp_path, df)
    prep = _prep(c, monkeypatch)
    assert prep["persist_kwargs"]["garbage_ids"] == {7}
    assert list(c._qc_features_df["label_status"]) == ["deleted", "active"]


def test_a_frame_without_label_status_treats_every_row_as_active(
        tmp_path, monkeypatch):
    df = pd.DataFrame({"label": [7, 8], "qc_tier": ["garbage", "good"]})
    c = _ctrl(tmp_path, df)
    prep = _prep(c, monkeypatch)
    assert prep["persist_kwargs"]["garbage_ids"] == {7}
    assert list(c._qc_features_df["label_status"]) == ["deleted", "active"]


def test_thresholds_keep_the_tier_of_a_row_with_no_finite_score():
    df = pd.DataFrame({
        "label": [1, 2],
        "qc_score": [np.nan, 0.0],
        "qc_tier": ["", ""],
    })
    out = assign_qc_tiers(df, (1.0, 2.0, 3.0))
    assert list(out["qc_tier"]) == ["", "good"]


def test_thresholds_keep_the_tier_of_a_ghost_row():
    df = pd.DataFrame({
        "label": [1, 1],
        "qc_score": [9.0, 9.0],
        "qc_tier": ["curated", "good"],
        "label_status": ["active", "superseded"],
    })
    out = assign_qc_tiers(df, (1.0, 2.0, 3.0))
    # active row: curated stays; ghost row keeps what it had
    assert list(out["qc_tier"]) == ["curated", "good"]


def test_thresholds_default_a_missing_tier_to_empty_for_non_finite():
    df = pd.DataFrame({"label": [1], "qc_score": [np.nan]})
    out = assign_qc_tiers(df, (1.0, 2.0, 3.0))
    assert list(out["qc_tier"]) == [""]


class _Area:
    def __init__(self):
        self.cards = []

    def set_card_message(self, text, kind="info"):
        self.cards.append((text, kind))


def test_a_navigator_mark_is_refused_while_commit_triage_runs():
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T
    df = pd.DataFrame({"label": [7], "qc_tier": ["review"],
                       "label_status": ["active"]})
    c = C.__new__(C)
    c._qc_features_df = df
    c._commit_triage_in_flight = True
    area = _Area()
    c._set_plot_status = lambda text, kind="info": area.set_card_message(text, kind)
    c._commit_curation_action = lambda *a, **k: pytest.fail("marked")
    c._append_log = lambda t: None
    assert c._on_annotator_mark_tier(7, "good") is False
    assert df["qc_tier"].tolist() == ["review"]
    assert area.cards == [(T.CARD_WARN_MARK_WHILE_SAVING, "warn")]


class _CurationStack:
    def __init__(self, entries=()):
        self.entries = list(entries)

    def push(self, e):
        self.entries.append(e)

    def pop(self):
        return self.entries.pop() if self.entries else None

    def __len__(self):
        return len(self.entries)

    def __bool__(self):
        return bool(self.entries)


def _undo_ctrl():
    c = C.__new__(C)
    c._qc_features_df = pd.DataFrame({
        "label": [7], "qc_tier": ["review"], "label_status": ["active"]})
    c._curation_undo_stack = _CurationStack()
    c._tier_undo_stack = [pd.Series(["review"], name="qc_tier")]
    c._curation_dirty = False
    c._tiers_dirty = False
    c._curation_dirty_epoch = 0
    c._append_log = lambda t: None
    c._refresh_curation_ui = lambda: None
    c._find_active_labels_layer = lambda: None
    return c


def test_a_curation_action_empties_the_tier_undo_stack():
    c = _undo_ctrl()
    c._commit_curation_action("mark_good", 7, "Mark label 7 as good")
    assert c._tier_undo_stack == []


def test_a_curation_undo_empties_the_tier_undo_stack():
    c = _undo_ctrl()
    c._commit_curation_action("mark_good", 7, "Mark label 7 as good")
    c._tier_undo_stack.append(pd.Series(["good"], name="qc_tier"))
    c._undo_last_curation()
    assert c._tier_undo_stack == []

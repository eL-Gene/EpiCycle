"""An extract that finishes for a patch other than the one the table in
memory belongs to must not merge that table's tiers into its frame, nor
auto-save over it."""
from __future__ import annotations

import pandas as pd

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
    _extract_autosave_will_run,
)

from .test_qc_tier_preservation import _make_bare_controller


def test_the_table_of_another_patch_is_not_merged_or_saved(tmp_path):
    prev = pd.DataFrame({"label": [1, 2], "qc_tier": ["garbage", "garbage"],
                         "label_status": ["active", "active"]})
    ctrl = _make_bare_controller(prev_df=prev, working_dir=tmp_path)
    ctrl._qc_table_patch = "P1.tif"
    ctrl._qc_view_patch_name = ""
    fresh = pd.DataFrame({"label": [1, 2], "area": [10, 20],
                          "qc_score": [0.1, 0.2],
                          "qc_tier": ["good", "review"]})
    C._on_qc_extract_finished(ctrl, fresh, "P3.tif")
    out = ctrl._qc_features_df
    assert dict(zip(out["label"], out["qc_tier"])) == {1: "good", 2: "review"}
    assert not (tmp_path / "artifacts").exists() or not list(
        tmp_path.rglob("P1_features.csv"))


def test_the_tier_count_does_not_wait_for_a_save_that_never_runs():
    c = type("X", (), {})()
    c._qc_features_df = pd.DataFrame({"label": [1]})
    c._qc_table_patch = "P1.tif"
    c._qc_view_patch_name = "P1.tif"
    assert _extract_autosave_will_run(c, "P3.tif") is False
    assert _extract_autosave_will_run(c, "P1.tif") is True


def test_an_extract_of_another_patch_saves_its_own_frame_then_counts(
        tmp_path, capsys):
    from unittest.mock import MagicMock

    prev = pd.DataFrame({"label": [1, 2], "qc_tier": ["garbage", "garbage"],
                         "label_status": ["active", "active"]})
    ctrl = _make_bare_controller(prev_df=prev, working_dir=tmp_path)
    ctrl._qc_table_patch = "P1.tif"
    ctrl._qc_view_patch_name = ""
    seen = []

    def _count(why, names):
        seen.append((why, list(names),
                     list(tmp_path.rglob("P3_features.csv"))))

    ctrl._start_tier_counts = MagicMock(side_effect=_count)
    fresh = pd.DataFrame({"label": [1, 2], "area": [10, 20],
                          "qc_score": [0.1, 0.2],
                          "qc_tier": ["good", "review"]})
    C._on_qc_extract_finished(ctrl, fresh, "P3.tif")
    assert len(seen) == 1 and seen[0][:2] == ("extraction", ["P3.tif"])
    assert seen[0][2], "the count ran before P3's file was written"
    on_disk = pd.read_csv(seen[0][2][0])
    assert dict(zip(on_disk["label"], on_disk["qc_tier"])) == {
        1: "good", 2: "review"}
    assert not list(tmp_path.rglob("P1_features.csv"))
    assert "[TABLE-OWNER]" in capsys.readouterr().out

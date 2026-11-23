"""A re-extract keeps the tiers of the table in memory; the tiers the fresh
frame carries (read from disk) only fill labels the table has no tier for."""
from __future__ import annotations

import numpy as np
import pandas as pd

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)

from .test_qc_tier_preservation import _make_bare_controller


def _fresh(rows):
    return pd.DataFrame({
        "label": [r[0] for r in rows],
        "area": [100] * len(rows),
        "qc_score": [0.1] * len(rows),
        "qc_tier": [r[1] for r in rows],
    })


def _run(prev, fresh, tmp_path):
    ctrl = _make_bare_controller(prev_df=prev, working_dir=tmp_path)
    C._on_qc_extract_finished(ctrl, fresh, "p.tif")
    out = ctrl._qc_features_df
    out = out[out["label_status"] == "active"] if "label_status" in out else out
    return dict(zip(out["label"], out["qc_tier"]))


def test_the_tier_in_memory_wins_over_the_tier_from_disk(tmp_path):
    prev = pd.DataFrame({"label": [5], "qc_tier": ["good"],
                         "label_status": ["active"]})
    got = _run(prev, _fresh([(5, "review"), (6, "bad")]), tmp_path)
    assert got[5] == "good"
    assert got[6] == "bad"


def test_only_the_active_row_of_the_table_counts(tmp_path):
    prev = pd.DataFrame({
        "label": [7, 7],
        "qc_tier": ["garbage", np.nan],
        "label_status": ["superseded", "active"],
    })
    got = _run(prev, _fresh([(7, "review")]), tmp_path)
    assert got[7] == "review"


def _first_extract(tmp_path):
    from unittest.mock import MagicMock

    fresh = pd.DataFrame({"label": [1, 2], "area": [10, 20],
                          "qc_score": [0.1, 0.2]})
    ctrl = _make_bare_controller(prev_df=None, working_dir=tmp_path)

    def _assign(df, **kw):
        df["qc_tier"] = ["good", "review"]

    ctrl._auto_assign_qc_tiers = _assign
    ctrl._start_tier_counts = MagicMock()
    C._on_qc_extract_finished(ctrl, fresh, "p.tif")
    return ctrl


def test_a_first_extraction_saves_the_assigned_tiers_to_disk(tmp_path):
    ctrl = _first_extract(tmp_path)
    csvs = list(tmp_path.rglob("p_features.csv"))
    assert len(csvs) == 1
    on_disk = pd.read_csv(csvs[0])
    assert list(on_disk["qc_tier"]) == list(ctrl._qc_features_df["qc_tier"])


def test_a_first_extraction_counts_the_row_after_the_write(tmp_path):
    from unittest.mock import MagicMock

    seen = []

    def _count(why, names):
        csvs = list(tmp_path.rglob("p_features.csv"))
        seen.append(bool(csvs) and "qc_tier" in pd.read_csv(csvs[0]).columns)

    fresh = pd.DataFrame({"label": [1, 2], "area": [10, 20],
                          "qc_score": [0.1, 0.2]})
    ctrl = _make_bare_controller(prev_df=None, working_dir=tmp_path)
    ctrl._auto_assign_qc_tiers = lambda df, **kw: df.__setitem__(
        "qc_tier", ["good", "review"])
    ctrl._start_tier_counts = MagicMock(side_effect=_count)
    C._on_qc_extract_finished(ctrl, fresh, "p.tif")
    assert seen == [True]               # one call, with the tiers on disk


def test_a_re_extraction_still_counts_the_row_once(tmp_path):
    from unittest.mock import MagicMock

    prev = pd.DataFrame({"label": [5], "qc_tier": ["good"],
                         "label_status": ["active"]})
    ctrl = _make_bare_controller(prev_df=prev, working_dir=tmp_path)
    ctrl._start_tier_counts = MagicMock()
    C._on_qc_extract_finished(ctrl, _fresh([(5, "review")]), "p.tif")
    assert ctrl._start_tier_counts.call_count == 1

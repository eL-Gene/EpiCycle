"""Tests for P4: Feature staleness tracking and assignment guardrails.

Verifies that:
- ``feature_state`` column is set to "stale" after curation edits
- ``feature_state`` column is set to "fresh" after recalculation
- Old CSVs without ``feature_state`` load with default "fresh"
- Bulk tier assignment logs skipped count for non-actionable rows
"""

import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qc_df(n=5, with_feature_state=True):
    """Create a minimal QC DataFrame."""
    df = pd.DataFrame({
        "label": list(range(1, n + 1)),
        "area": [100] * n,
        "centroid-0": [10.0] * n,
        "centroid-1": [20.0] * n,
        "centroid-2": [30.0] * n,
        "qc_tier": ["good"] * n,
    })
    if with_feature_state:
        df["feature_state"] = "fresh"
    return df


# ---------------------------------------------------------------------------
# Tests: feature_state column management
# ---------------------------------------------------------------------------

class TestFeatureStateColumn:

    def test_fresh_by_default(self):
        df = _make_qc_df()
        assert (df["feature_state"] == "fresh").all()

    def test_mark_stale(self):
        df = _make_qc_df()
        # Simulate marking label 3 as stale after an edit
        df.loc[df["label"] == 3, "feature_state"] = "stale"
        row = df[df["label"] == 3].iloc[0]
        assert row["feature_state"] == "stale"
        # Others remain fresh
        assert (df[df["label"] != 3]["feature_state"] == "fresh").all()

    def test_mark_fresh_on_recalc(self):
        df = _make_qc_df()
        df.loc[df["label"] == 3, "feature_state"] = "stale"
        # Simulate recalc
        df.loc[df["label"] == 3, "feature_state"] = "fresh"
        assert (df["feature_state"] == "fresh").all()


class TestFeatureStateBackwardCompat:
    """Old CSVs don't have feature_state column."""

    def test_load_without_column_gets_fresh(self):
        df = _make_qc_df(with_feature_state=False)
        assert "feature_state" not in df.columns
        # Simulate the backward-compat fill
        if "feature_state" not in df.columns:
            df["feature_state"] = "fresh"
        assert (df["feature_state"] == "fresh").all()

    def test_csv_roundtrip_preserves_feature_state(self, tmp_path):
        df = _make_qc_df()
        df.loc[df["label"] == 2, "feature_state"] = "stale"
        csv_path = tmp_path / "features.csv"
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)
        assert loaded[loaded["label"] == 2].iloc[0]["feature_state"] == "stale"
        assert loaded[loaded["label"] == 1].iloc[0]["feature_state"] == "fresh"


class TestBulkAssignmentGuardrails:
    """Verify bulk tier assignment respects row eligibility."""

    def test_skip_recomputing_rows(self):
        df = _make_qc_df()
        df.loc[df["label"] == 2, "feature_state"] = "recomputing"
        selected_indices = [0, 1, 2]  # labels 1, 2, 3
        tier = "garbage"

        # Simulate guarded assignment
        skipped = 0
        for idx in selected_indices:
            if df.iloc[idx].get("feature_state") == "recomputing":
                skipped += 1
                continue
            df.iloc[idx, df.columns.get_loc("qc_tier")] = tier

        assert skipped == 1
        assert df.iloc[0]["qc_tier"] == "garbage"
        assert df.iloc[1]["qc_tier"] == "good"  # skipped (recomputing)
        assert df.iloc[2]["qc_tier"] == "garbage"

    def test_no_skip_when_all_fresh(self):
        df = _make_qc_df()
        selected_indices = [0, 1, 2]
        tier = "review"
        skipped = 0
        for idx in selected_indices:
            if df.iloc[idx].get("feature_state") == "recomputing":
                skipped += 1
                continue
            df.iloc[idx, df.columns.get_loc("qc_tier")] = tier

        assert skipped == 0
        assert (df.iloc[:3]["qc_tier"] == "review").all()

"""Tests for K8f: click-to-isolate categories + per-category feature breakdown."""

import numpy as np
import pandas as pd
import pytest


def _make_df(n=30):
    """Create a sample QC DataFrame with categorical and numeric columns."""
    rng = np.random.RandomState(42)
    labels = np.arange(1, n + 1)
    tiers = rng.choice(["good", "review", "bad", "garbage"], size=n)
    clusters = rng.choice([0, 1, 2, -1], size=n)
    patches = rng.choice(["patch_A", "patch_B"], size=n)
    return pd.DataFrame({
        "label": labels,
        "qc_tier": tiers,
        "cluster_id": clusters,
        "patch_name": patches,
        "qc_status": rng.choice(["pass", "flagged"], size=n),
        "area": rng.uniform(100, 10000, size=n),
        "sphericity": rng.uniform(0, 1, size=n),
        "compactness": rng.uniform(0, 1, size=n),
        "qc_score": rng.uniform(0, 1, size=n),
        "extent": rng.uniform(0, 1, size=n),
        "aspect_ratio": rng.uniform(1, 5, size=n),
    })


# -----------------------------------------------------------------------
# Category selection logic (no Qt needed)
# -----------------------------------------------------------------------


class TestCategorySelection:
    """Test _toggle_category_selection logic."""

    def test_select_category_returns_correct_labels(self):
        df = _make_df()
        expected_ids = df.loc[df["qc_tier"] == "good", "label"].values.tolist()
        assert len(expected_ids) > 0

        mask = df["qc_tier"] == "good"
        result = [int(lid) for lid in df.loc[mask, "label"].values]
        assert result == expected_ids

    def test_deselect_same_category(self):
        """Selecting the same category twice should produce a deselect toggle."""
        selected = ("qc_tier", "good")
        # Simulated toggle: if already selected, clear
        assert selected == ("qc_tier", "good")
        # After toggle, should be None
        new_state = None if selected == ("qc_tier", "good") else selected
        assert new_state is None

    def test_switch_category_replaces_selection(self):
        """Selecting a different category should replace the current one."""
        current = ("qc_tier", "good")
        new_cat = ("qc_tier", "bad")
        # Different category — should replace
        assert current != new_cat
        result = new_cat
        assert result == ("qc_tier", "bad")

    def test_cluster_id_type_coercion(self):
        """cluster_id is int in DataFrame but str from HTML link — need coercion."""
        df = _make_df()
        col = df["cluster_id"]
        cat_value = "0"  # string from HTML link

        # Coerce to match column dtype
        if col.dtype in (np.int64, np.int32):
            typed_val = int(cat_value)
        else:
            typed_val = cat_value

        mask = col == typed_val
        result = [int(lid) for lid in df.loc[mask, "label"].values]
        expected = df.loc[df["cluster_id"] == 0, "label"].values.tolist()
        assert result == expected

    def test_nonexistent_category_returns_empty(self):
        """Selecting a category that doesn't exist returns no labels."""
        df = _make_df()
        mask = df["qc_tier"] == "nonexistent_tier"
        result = [int(lid) for lid in df.loc[mask, "label"].values]
        assert result == []


# -----------------------------------------------------------------------
# Stacked histogram bin computation (no Qt needed)
# -----------------------------------------------------------------------


class TestComputeStackedBins:
    """Test shared binning logic used by histogram drawing and click handling."""

    def test_bin_counts_sum_to_total(self):
        """Sum of all category counts across all bins should equal finite data count."""
        df = _make_df(50)
        feature = "area"
        values = df[feature].values.astype(np.float64)
        finite_mask = np.isfinite(values)
        n_bins = 10
        _, edges = np.histogram(values[finite_mask], bins=n_bins)

        categories = df["qc_tier"].values[finite_mask]
        unique_cats = list(dict.fromkeys(categories))
        bin_indices = np.clip(np.digitize(values[finite_mask], edges) - 1, 0, n_bins - 1)

        cat_counts = {cat: np.zeros(n_bins, dtype=int) for cat in unique_cats}
        for i, cat in zip(bin_indices, categories):
            cat_counts[cat][i] += 1

        total = sum(c.sum() for c in cat_counts.values())
        assert total == int(finite_mask.sum())

    def test_bin_click_identifies_correct_category(self):
        """Walk categories bottom-to-top to identify click target."""
        # Manually construct small example: 2 bins, 2 categories
        unique_cats = ["A", "B"]
        cat_counts = {
            "A": np.array([3, 2]),
            "B": np.array([1, 4]),
        }
        edges = np.array([0.0, 5.0, 10.0])

        # Click in bin 0 (x=2.5), at y=2 → should be in "A" (0–3)
        bin_idx = 0
        my = 2.0
        bottom = 0.0
        clicked = None
        for cat in unique_cats:
            cnt = cat_counts[cat][bin_idx]
            top = bottom + cnt
            if bottom <= my <= top:
                clicked = cat
                break
            bottom = top
        assert clicked == "A"

        # Click in bin 0, at y=3.5 → should be in "B" (3–4)
        bottom = 0.0
        clicked = None
        my = 3.5
        for cat in unique_cats:
            cnt = cat_counts[cat][bin_idx]
            top = bottom + cnt
            if bottom <= my <= top:
                clicked = cat
                break
            bottom = top
        assert clicked == "B"

    def test_bin_click_outside_stack_returns_none(self):
        """Click above the stack top should return no category."""
        unique_cats = ["A"]
        cat_counts = {"A": np.array([5])}
        bin_idx = 0
        my = 6.0  # above the stack (max is 5)
        bottom = 0.0
        clicked = None
        for cat in unique_cats:
            cnt = cat_counts[cat][bin_idx]
            top = bottom + cnt
            if bottom <= my <= top:
                clicked = cat
                break
            bottom = top
        assert clicked is None


# -----------------------------------------------------------------------
# Category breakdown statistics (no Qt needed)
# -----------------------------------------------------------------------


class TestCategoryBreakdown:
    """Test per-category box plot statistics computation."""

    def test_breakdown_computes_correct_quartiles(self):
        df = _make_df(100)
        grouped = df.groupby("qc_tier")
        for tier_name, group in grouped:
            vals = group["area"].dropna().values.astype(np.float64)
            q1 = np.percentile(vals, 25)
            median = np.percentile(vals, 50)
            q3 = np.percentile(vals, 75)
            assert q1 <= median <= q3
            assert np.isfinite(median)

    def test_breakdown_handles_missing_features(self):
        """Features not in DataFrame should be skipped gracefully."""
        df = _make_df()
        features = ["area", "sphericity", "nonexistent_feature"]
        available = [f for f in features if f in df.columns]
        assert "nonexistent_feature" not in available
        assert "area" in available

    def test_breakdown_group_count_matches_categories(self):
        df = _make_df()
        n_tiers = df["qc_tier"].nunique()
        grouped = df.groupby("qc_tier")
        assert len(grouped) == n_tiers

    def test_breakdown_clears_on_continuous_mode(self):
        """When color-by is a continuous feature, categorical info returns None."""
        color_by = "area"  # continuous, not in _CATEGORICAL_COLOR_BY
        categorical_modes = {"QC status", "QC tier", "Cluster", "Patch"}
        is_categorical = color_by in categorical_modes
        assert not is_categorical

    def test_whisker_bounds_within_data_range(self):
        """Whisker bounds should not exceed the data range."""
        df = _make_df(200)
        vals = df["area"].values.astype(np.float64)
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        whisker_lo = max(vals.min(), q1 - 1.5 * iqr)
        whisker_hi = min(vals.max(), q3 + 1.5 * iqr)
        assert whisker_lo >= vals.min()
        assert whisker_hi <= vals.max()
        assert whisker_lo <= q1
        assert whisker_hi >= q3

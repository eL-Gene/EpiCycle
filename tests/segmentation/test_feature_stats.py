"""Tests for EDA Workbench — feature_stats pure computation."""

import math

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.segmentation.feature_stats import (
    TRANSFORM_METHODS,
    apply_transform,
    compute_correlations,
    compute_detail_stats,
    compute_feature_profile,
    compute_histogram,
    compute_normality,
    compute_skewness,
    profile_dataframe,
    resolve_auto_method,
)


# -----------------------------------------------------------------------
# compute_histogram
# -----------------------------------------------------------------------

class TestComputeHistogram:
    def test_basic_uniform(self):
        rng = np.random.default_rng(42)
        vals = rng.uniform(0, 1, 200)
        result = compute_histogram(vals, n_bins=8)
        assert result["counts"].shape == (8,)
        assert result["edges"].shape == (9,)
        assert result["max_count"] == int(result["counts"].max())
        assert result["counts"].sum() == 200

    def test_n_bins_respected(self):
        vals = np.arange(100, dtype=float)
        for n in (4, 10, 20):
            result = compute_histogram(vals, n_bins=n)
            assert len(result["counts"]) == n
            assert len(result["edges"]) == n + 1

    def test_empty_array(self):
        result = compute_histogram(np.array([]), n_bins=8)
        assert np.all(result["counts"] == 0)
        assert result["max_count"] == 0

    def test_single_value(self):
        result = compute_histogram(np.array([5.0]), n_bins=8)
        assert np.all(result["counts"] == 0)
        assert result["max_count"] == 0

    def test_nan_values_dropped(self):
        vals = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0])
        result = compute_histogram(vals, n_bins=4)
        # Only 1, 2, 3, 4 should be counted (inf dropped too)
        assert result["counts"].sum() == 4

    def test_constant_array(self):
        vals = np.full(50, 3.14)
        result = compute_histogram(vals, n_bins=8)
        # All values in one bin
        assert result["counts"].sum() == 50
        assert result["max_count"] == 50


# -----------------------------------------------------------------------
# compute_skewness
# -----------------------------------------------------------------------

class TestComputeSkewness:
    def test_symmetric_distribution(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 1000)
        result = compute_skewness(vals)
        assert abs(result["value"]) < 1.0
        assert result["arrow"] == ""
        assert result["severity"] == "low"
        assert result["color"] == "#6abf69"

    def test_right_skewed(self):
        rng = np.random.default_rng(42)
        vals = rng.exponential(1.0, 1000)
        result = compute_skewness(vals)
        assert result["value"] > 1.0
        assert result["arrow"] == "↗"
        assert result["severity"] in ("moderate", "high")

    def test_left_skewed(self):
        rng = np.random.default_rng(42)
        vals = -rng.exponential(1.0, 1000)
        result = compute_skewness(vals)
        assert result["value"] < -1.0
        assert result["arrow"] == "↙"

    def test_high_skewness_color(self):
        # Very skewed: e.g., [1]*100 + [1000]
        vals = np.concatenate([np.ones(100), np.array([1000.0])])
        result = compute_skewness(vals)
        assert result["severity"] == "high"
        assert result["color"] == "#ff6b6b"

    def test_moderate_skewness_color(self):
        # Moderately skewed
        rng = np.random.default_rng(42)
        vals = rng.lognormal(0, 0.5, 500)
        result = compute_skewness(vals)
        # lognormal with sigma=0.5 typically has skew ~1.5-2
        assert result["color"] in ("#d4a843", "#ff6b6b", "#6abf69")

    def test_insufficient_data(self):
        result = compute_skewness(np.array([1.0, 2.0]))
        assert math.isnan(result["value"])
        assert result["arrow"] == ""

    def test_all_nan(self):
        result = compute_skewness(np.array([np.nan, np.nan, np.nan]))
        assert math.isnan(result["value"])


# -----------------------------------------------------------------------
# compute_normality
# -----------------------------------------------------------------------

class TestComputeNormality:
    def test_normal_distribution(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 500)
        result = compute_normality(vals)
        assert result["normal"] is True
        assert result["symbol"] == "✓"
        assert result["color"] == "#6abf69"
        assert result["test"] == "shapiro"
        assert result["p_value"] > 0.05

    def test_non_normal_distribution(self):
        rng = np.random.default_rng(42)
        vals = rng.exponential(1.0, 500)
        result = compute_normality(vals)
        assert result["normal"] is False
        assert result["symbol"] == "✗"
        assert result["color"] == "#ff6b6b"

    def test_shapiro_used_for_small_n(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 100)
        result = compute_normality(vals)
        assert result["test"] == "shapiro"

    def test_dagostino_used_for_large_n(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 6000)
        result = compute_normality(vals)
        assert result["test"] == "dagostino"

    def test_insufficient_data(self):
        result = compute_normality(np.array([1.0, 2.0, 3.0]))
        assert math.isnan(result["p_value"])
        assert result["symbol"] == "—"
        assert result["test"] == "none"

    def test_nan_values_excluded(self):
        rng = np.random.default_rng(42)
        vals = np.concatenate([rng.normal(0, 1, 200), np.full(50, np.nan)])
        result = compute_normality(vals)
        assert result["test"] == "shapiro"
        assert result["p_value"] > 0  # successfully computed


# -----------------------------------------------------------------------
# compute_feature_profile
# -----------------------------------------------------------------------

class TestComputeFeatureProfile:
    def test_returns_all_sections(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 200)
        profile = compute_feature_profile(vals)
        assert "histogram" in profile
        assert "skewness" in profile
        assert "normality" in profile
        assert "counts" in profile["histogram"]
        assert "value" in profile["skewness"]
        assert "p_value" in profile["normality"]

    def test_custom_bins(self):
        vals = np.arange(100, dtype=float)
        profile = compute_feature_profile(vals, n_bins=20)
        assert len(profile["histogram"]["counts"]) == 20


# -----------------------------------------------------------------------
# profile_dataframe
# -----------------------------------------------------------------------

class TestProfileDataframe:
    def _make_df(self, n=200):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "area": rng.normal(100, 20, n),
            "aspect_ratio": rng.normal(1.5, 0.3, n),
            "sphericity": rng.normal(0.8, 0.1, n),
            "is_border_cell": [True] * 20 + [False] * (n - 20),
            "label": np.arange(1, n + 1),
        })

    def test_basic_profiling(self):
        df = self._make_df()
        features = ["area", "aspect_ratio", "sphericity"]
        profiles = profile_dataframe(df, features)
        assert set(profiles.keys()) == set(features)
        for feat, p in profiles.items():
            assert "histogram" in p
            assert "skewness" in p
            assert "normality" in p

    def test_border_cells_excluded(self):
        df = self._make_df(n=200)
        features = ["area"]
        profiles = profile_dataframe(df, features, exclude_border=True)
        # 180 non-border cells
        assert profiles["area"]["histogram"]["counts"].sum() == 180

    def test_border_cells_included(self):
        df = self._make_df(n=200)
        features = ["area"]
        profiles = profile_dataframe(df, features, exclude_border=False)
        assert profiles["area"]["histogram"]["counts"].sum() == 200

    def test_no_border_column(self):
        """If is_border_cell column doesn't exist, exclude_border is a no-op."""
        df = pd.DataFrame({
            "area": np.arange(50, dtype=float),
        })
        profiles = profile_dataframe(df, ["area"], exclude_border=True)
        assert profiles["area"]["histogram"]["counts"].sum() == 50

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"area": np.arange(50, dtype=float)})
        profiles = profile_dataframe(df, ["area", "nonexistent"])
        assert "area" in profiles
        assert "nonexistent" not in profiles

    def test_empty_dataframe(self):
        df = pd.DataFrame({"area": pd.array([], dtype=float)})
        profiles = profile_dataframe(df, ["area"])
        assert profiles["area"]["histogram"]["max_count"] == 0


# -----------------------------------------------------------------------
# apply_transform
# -----------------------------------------------------------------------

class TestApplyTransform:
    def test_raw_passthrough(self):
        vals = np.array([1.0, 2.0, 3.0])
        result = apply_transform(vals, "raw")
        np.testing.assert_array_equal(result, vals)

    def test_log_positive(self):
        vals = np.array([0.0, 1.0, 10.0, 100.0])
        result = apply_transform(vals, "log")
        expected = np.log1p(vals)
        np.testing.assert_allclose(result, expected)

    def test_log_with_negatives(self):
        """log should shift values so min=0 before log1p."""
        vals = np.array([-5.0, 0.0, 5.0, 10.0])
        result = apply_transform(vals, "log")
        shifted = vals - vals.min()  # [0, 5, 10, 15]
        expected = np.log1p(shifted)
        np.testing.assert_allclose(result, expected)

    def test_sqrt_positive(self):
        vals = np.array([0.0, 1.0, 4.0, 9.0])
        result = apply_transform(vals, "sqrt")
        np.testing.assert_allclose(result, np.sqrt(vals))

    def test_sqrt_with_negatives(self):
        vals = np.array([-4.0, 0.0, 4.0])
        result = apply_transform(vals, "sqrt")
        shifted = vals - vals.min()  # [0, 4, 8]
        np.testing.assert_allclose(result, np.sqrt(shifted))

    def test_rank(self):
        vals = np.array([10.0, 30.0, 20.0, 40.0])
        result = apply_transform(vals, "rank")
        # rankdata: 10→1, 20→2, 30→3, 40→4; then /4
        np.testing.assert_allclose(result, [0.25, 0.75, 0.5, 1.0])

    def test_rank_output_range(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 100)
        result = apply_transform(vals, "rank")
        assert result.min() > 0
        assert result.max() <= 1.0

    def test_zscore(self):
        vals = np.array([10.0, 20.0, 30.0])
        result = apply_transform(vals, "zscore")
        assert abs(np.mean(result)) < 1e-10
        assert abs(np.std(result, ddof=1) - 1.0) < 1e-10

    def test_winsorize(self):
        rng = np.random.default_rng(42)
        vals = np.concatenate([rng.normal(0, 1, 100), np.array([50.0, -50.0])])
        result = apply_transform(vals, "winsorize", winsorize_sigma=3.0)
        # Extreme values should be pulled in
        assert result[-2] < 50.0
        assert result[-1] > -50.0
        # The bulk of the data should be mostly unchanged
        np.testing.assert_allclose(result[:5], vals[:5], atol=0.01)

    def test_nan_preserved(self):
        vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        for method in ("raw", "log", "sqrt", "rank", "zscore", "winsorize"):
            result = apply_transform(vals, method)
            assert np.isnan(result[1])
            assert np.isnan(result[3])
            assert np.isfinite(result[0])
            assert np.isfinite(result[2])
            assert np.isfinite(result[4])

    def test_empty_array(self):
        for method in TRANSFORM_METHODS:
            result = apply_transform(np.array([]), method)
            assert len(result) == 0

    def test_constant_array(self):
        vals = np.full(50, 3.14)
        # Should not crash for any method
        for method in ("raw", "log", "sqrt", "rank", "zscore", "winsorize"):
            result = apply_transform(vals, method)
            assert len(result) == 50
            assert np.all(np.isfinite(result))

    def test_does_not_modify_input(self):
        vals = np.array([1.0, 2.0, 3.0])
        original = vals.copy()
        apply_transform(vals, "log")
        np.testing.assert_array_equal(vals, original)


# -----------------------------------------------------------------------
# resolve_auto_method / auto transform
# -----------------------------------------------------------------------

class TestAutoTransform:
    def test_skewed_gets_log(self):
        rng = np.random.default_rng(42)
        vals = rng.exponential(1.0, 500)  # right-skewed
        method = resolve_auto_method(vals)
        assert method == "log"

    def test_symmetric_stays_raw(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 500)
        method = resolve_auto_method(vals)
        assert method == "raw"

    def test_bounded_feature_stays_raw(self):
        """Bounded [0,1] features should never be log-transformed."""
        rng = np.random.default_rng(42)
        # Even if skewed, bounded features stay raw
        vals = rng.beta(0.5, 5, 500)  # very right-skewed but bounded
        method = resolve_auto_method(vals, feature_name="extent")
        assert method == "raw"

    def test_bounded_features_comprehensive(self):
        """All known bounded features should resolve to raw."""
        from napari_epicycle.fusion_reconstruction.qc_features import BOUNDED_FEATURES
        rng = np.random.default_rng(42)
        vals = rng.exponential(1.0, 100)  # skewed
        for feat in list(BOUNDED_FEATURES)[:5]:
            assert resolve_auto_method(vals, feat) == "raw"

    def test_auto_transform_reduces_skewness(self):
        """Auto-transforming a skewed feature should reduce its skewness."""
        rng = np.random.default_rng(42)
        vals = rng.exponential(1.0, 500)
        raw_skew = abs(compute_skewness(vals)["value"])
        transformed = apply_transform(vals, "auto")
        new_skew = abs(compute_skewness(transformed)["value"])
        assert new_skew < raw_skew

    def test_auto_on_insufficient_data(self):
        vals = np.array([1.0, 2.0])
        method = resolve_auto_method(vals)
        assert method == "raw"

    def test_profile_after_transform(self):
        """Recomputing profile on transformed data should reflect the change."""
        rng = np.random.default_rng(42)
        raw = rng.exponential(2.0, 300)
        raw_profile = compute_feature_profile(raw)
        transformed = apply_transform(raw, "log")
        new_profile = compute_feature_profile(transformed)
        # Log transform should reduce skewness
        assert abs(new_profile["skewness"]["value"]) < abs(raw_profile["skewness"]["value"])


# -----------------------------------------------------------------------
# compute_detail_stats
# -----------------------------------------------------------------------

class TestComputeDetailStats:
    def test_basic_stats(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(100, 20, 500)
        stats = compute_detail_stats(vals)
        assert stats["n"] == 500
        assert abs(stats["mean"] - 100) < 5
        assert abs(stats["median"] - 100) < 5
        assert abs(stats["std"] - 20) < 5
        assert stats["min"] < stats["max"]
        assert stats["iqr"] > 0

    def test_skew_and_kurtosis(self):
        rng = np.random.default_rng(42)
        vals = rng.exponential(1.0, 500)
        stats = compute_detail_stats(vals)
        assert stats["skew"] > 1.0  # exponential is right-skewed
        assert "kurtosis" in stats

    def test_empty_array(self):
        stats = compute_detail_stats(np.array([]))
        assert stats["n"] == 0
        assert math.isnan(stats["mean"])
        assert math.isnan(stats["iqr"])

    def test_single_value(self):
        stats = compute_detail_stats(np.array([5.0]))
        assert stats["n"] == 1
        assert stats["mean"] == 5.0
        assert stats["std"] == 0.0

    def test_constant_array(self):
        stats = compute_detail_stats(np.full(100, 42.0))
        assert stats["n"] == 100
        assert stats["skew"] == 0.0
        assert stats["kurtosis"] == 0.0

    def test_nan_excluded(self):
        vals = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        stats = compute_detail_stats(vals)
        assert stats["n"] == 4


# -----------------------------------------------------------------------
# compute_correlations
# -----------------------------------------------------------------------

class TestComputeCorrelations:
    def _make_df(self, n=200):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, n)
        return pd.DataFrame({
            "area": x,
            "volume": x * 2 + rng.normal(0, 0.1, n),  # highly correlated
            "shape": rng.normal(0, 1, n),  # uncorrelated
            "noise": rng.normal(0, 1, n),
            "is_border_cell": [False] * n,
        })

    def test_basic_correlations(self):
        df = self._make_df()
        features = ["area", "volume", "shape", "noise"]
        result = compute_correlations(df, "area", features)
        assert "spearman" in result
        assert "pearson" in result
        assert len(result["spearman"]) <= 5
        # "volume" should be the top correlate
        assert result["spearman"][0][0] == "volume"
        assert result["pearson"][0][0] == "volume"
        assert abs(result["pearson"][0][1]) > 0.9

    def test_sorted_by_abs_correlation(self):
        df = self._make_df()
        features = ["area", "volume", "shape", "noise"]
        result = compute_correlations(df, "area", features)
        # Should be descending by |r|
        spearman_abs = [abs(r) for _, r in result["spearman"]]
        assert spearman_abs == sorted(spearman_abs, reverse=True)

    def test_top_n_limit(self):
        df = self._make_df()
        features = ["area", "volume", "shape", "noise"]
        result = compute_correlations(df, "area", features, top_n=2)
        assert len(result["spearman"]) <= 2
        assert len(result["pearson"]) <= 2

    def test_missing_feature(self):
        df = self._make_df()
        result = compute_correlations(df, "nonexistent", ["area"])
        assert result["spearman"] == []
        assert result["pearson"] == []

    def test_excludes_self(self):
        df = self._make_df()
        features = ["area", "volume"]
        result = compute_correlations(df, "area", features)
        names = [name for name, _ in result["spearman"]]
        assert "area" not in names

    def test_constant_feature_skipped(self):
        df = pd.DataFrame({
            "area": np.arange(50, dtype=float),
            "constant": np.full(50, 1.0),
        })
        result = compute_correlations(df, "area", ["area", "constant"])
        names = [name for name, _ in result["spearman"]]
        assert "constant" not in names

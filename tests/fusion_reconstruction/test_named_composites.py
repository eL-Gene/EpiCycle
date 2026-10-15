"""Tests for named composite signal features (Group D)."""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.qc_features import extract_features


class TestComputeNamedComposites:
    """Tests for merge_signal, fragment_signal, shape_abnormality."""

    def _make_df_with_features(self, n_labels=5):
        """Create a DataFrame with all prerequisite features."""
        labels = np.zeros((20, 50, 50), dtype=np.int32)
        step = 50 // n_labels
        for i in range(n_labels):
            y0 = i * step + 1
            labels[3:17, y0:y0 + step - 2, 3:12] = i + 1
        df = extract_features(labels)
        # Add stub columns that composites depend on
        rng = np.random.RandomState(42)
        for col in ("boundary_interior_ratio", "inertia_elongation",
                     "skeleton_branches", "euler_number",
                     "cross_section_bulge", "mean_neighbor_size_ratio",
                     "neighbor_count", "background_fraction",
                     "apical_basal_area_ratio", "cross_section_taper",
                     "true_sphericity"):
            if col not in df.columns:
                df[col] = rng.rand(len(df))
        return df

    def test_adds_expected_columns(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df_with_features()
        compute_named_composites(df)
        for col in ("merge_signal", "fragment_signal", "shape_abnormality"):
            assert col in df.columns, f"Missing column: {col}"

    def test_values_are_finite(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df_with_features()
        compute_named_composites(df)
        for col in ("merge_signal", "fragment_signal", "shape_abnormality"):
            assert np.all(np.isfinite(df[col].values)), f"Non-finite in {col}"

    def test_values_are_nonnegative(self):
        """Composite signals are mean absolute z-scores — always >= 0."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df_with_features()
        compute_named_composites(df)
        for col in ("merge_signal", "fragment_signal", "shape_abnormality"):
            assert np.all(df[col].values >= 0), f"Negative values in {col}"

    def test_high_merge_signal_for_large_elongated(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df_with_features()
        # Make label 1 look like a merge: high elongation, high branches,
        # low boundary_interior_ratio, large relative to neighbor
        df.loc[df["label"] == 1, "inertia_elongation"] = 20.0
        df.loc[df["label"] == 1, "skeleton_branches"] = 10.0
        df.loc[df["label"] == 1, "boundary_interior_ratio"] = 0.5
        df.loc[df["label"] == 1, "mean_neighbor_size_ratio"] = 5.0
        df.loc[df["label"] == 1, "cross_section_bulge"] = 0.5
        compute_named_composites(df)
        merge_1 = df[df["label"] == 1].iloc[0]["merge_signal"]
        merge_2 = df[df["label"] == 2].iloc[0]["merge_signal"]
        assert merge_1 > merge_2

    def test_high_fragment_signal_for_small_isolated(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df_with_features()
        # Make label 1 look like a fragment: small, isolated, small relative to neighbors
        df.loc[df["label"] == 1, "area"] = 10.0
        df.loc[df["label"] == 1, "neighbor_count"] = 0.0
        df.loc[df["label"] == 1, "background_fraction"] = 0.9
        df.loc[df["label"] == 1, "mean_neighbor_size_ratio"] = 0.1
        compute_named_composites(df)
        frag_1 = df[df["label"] == 1].iloc[0]["fragment_signal"]
        frag_2 = df[df["label"] == 2].iloc[0]["fragment_signal"]
        assert frag_1 > frag_2

    def test_boundary_confidence_present_when_prob_features_exist(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df_with_features()
        # Add probability features
        for col in ("prob_edge_gradient", "prob_boundary_entropy",
                     "prob_boundary_mean"):
            df[col] = np.random.RandomState(7).rand(len(df))
        compute_named_composites(df)
        assert "boundary_confidence" in df.columns
        assert np.all(np.isfinite(df["boundary_confidence"].values))

    def test_boundary_confidence_absent_without_prob_features(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df_with_features()
        # No prob features → boundary_confidence should be NaN or not added
        compute_named_composites(df)
        if "boundary_confidence" in df.columns:
            # If added, values should be NaN (no prob data)
            assert np.all(np.isnan(df["boundary_confidence"].values))

    def test_empty_df(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        labels = np.zeros((5, 5, 5), dtype=np.int32)
        df = extract_features(labels)
        compute_named_composites(df)
        assert "merge_signal" in df.columns

    def test_single_label(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[5:15, 5:15, 5:15] = 1
        df = extract_features(labels)
        for col in ("boundary_interior_ratio", "inertia_elongation",
                     "skeleton_branches", "cross_section_bulge",
                     "mean_neighbor_size_ratio", "neighbor_count",
                     "background_fraction", "apical_basal_area_ratio",
                     "cross_section_taper", "true_sphericity"):
            if col not in df.columns:
                df[col] = 1.0
        compute_named_composites(df)
        # Single label → z-score is 0 → composites should be 0
        assert df.iloc[0]["merge_signal"] == 0.0


class TestDirectionalComposites:
    """Verify that negative-weight features are directionally inverted."""

    def _make_df(self, n=20):
        """Create a DataFrame with controlled feature values."""
        import pandas as pd
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "label": np.arange(1, n + 1),
            "boundary_interior_ratio": rng.normal(2.0, 0.3, n),
            "inertia_elongation": rng.normal(3.0, 0.5, n),
            "skeleton_branches": rng.normal(2.0, 0.5, n),
            "euler_number": rng.normal(1.0, 0.2, n),
            "cross_section_bulge": rng.normal(0.1, 0.05, n),
            "mean_neighbor_size_ratio": rng.normal(1.0, 0.3, n),
            "area": rng.normal(500, 100, n),
            "neighbor_count": rng.normal(6, 2, n),
            "background_fraction": rng.normal(0.1, 0.05, n),
            "apical_basal_area_ratio": rng.normal(1.0, 0.3, n),
            "cross_section_taper": rng.normal(0.0, 0.1, n),
            "true_sphericity": rng.normal(0.7, 0.1, n),
        })

    def test_low_bir_scores_high_merge(self):
        """Low boundary_interior_ratio (weight=-1) should increase merge_signal."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df()
        # Make cell 0 have very LOW boundary_interior_ratio (merge-like)
        df.loc[0, "boundary_interior_ratio"] = 0.2
        compute_named_composites(df)
        # Cell 0 should have higher merge_signal than median
        median_merge = df["merge_signal"].median()
        assert df.loc[0, "merge_signal"] > median_merge

    def test_high_bir_does_not_score_merge(self):
        """High boundary_interior_ratio (healthy) should NOT increase merge_signal."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df()
        # Make cell 0 have very HIGH boundary_interior_ratio (healthy)
        df.loc[0, "boundary_interior_ratio"] = 10.0
        # Keep everything else at population mean
        for col in ["inertia_elongation", "skeleton_branches", "euler_number",
                     "cross_section_bulge", "mean_neighbor_size_ratio"]:
            df.loc[0, col] = df[col].median()
        compute_named_composites(df)
        # Cell 0's bir contribution should be ~0 (healthy direction ignored)
        assert df.loc[0, "merge_signal"] < 0.5

    def test_small_area_scores_high_fragment(self):
        """Small area (weight=-1) should increase fragment_signal."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df()
        df.loc[0, "area"] = 10.0  # very small
        df.loc[0, "neighbor_count"] = 0.0  # isolated
        compute_named_composites(df)
        median_frag = df["fragment_signal"].median()
        assert df.loc[0, "fragment_signal"] > median_frag

    def test_large_area_does_not_score_fragment(self):
        """Large area (healthy for fragment) should NOT increase fragment_signal."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df()
        df.loc[0, "area"] = 5000.0  # very large
        # Keep everything else at population mean
        for col in ["neighbor_count", "background_fraction",
                     "mean_neighbor_size_ratio"]:
            df.loc[0, col] = df[col].median()
        compute_named_composites(df)
        assert df.loc[0, "fragment_signal"] < 0.5

    def test_prefixed_bir_fallback(self):
        """When unprefixed boundary_interior_ratio is missing, use channel version."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_named_composites,
        )
        df = self._make_df()
        # Rename to prefixed version (Step 4 pattern)
        df.rename(columns={
            "boundary_interior_ratio": "membrane_boundary_interior_ratio"
        }, inplace=True)
        compute_named_composites(df)
        # Should still compute merge_signal (using prefixed column)
        assert "merge_signal" in df.columns
        assert not np.all(np.isnan(df["merge_signal"].values))

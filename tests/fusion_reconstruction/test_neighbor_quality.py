"""Tests for neighbor quality features (Group B)."""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.qc_features import (
    compute_neighbor_features, extract_features,
)


class TestNeighborQualityFeatures:
    """Tests for mean_neighbor_size_ratio, neighbor_size_cv, etc."""

    def test_adds_quality_columns(self):
        labels = np.zeros((10, 20, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        labels[2:8, 12:18, 12:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        for col in ("mean_neighbor_size_ratio", "neighbor_size_cv",
                     "contact_weighted_neighbor_size_ratio"):
            assert col in df.columns, f"Missing column: {col}"

    def test_equal_size_neighbors_ratio_near_one(self):
        # Two identical touching labels
        labels = np.zeros((10, 10, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:10] = 1
        labels[2:8, 2:8, 10:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        r1 = df.iloc[0]["mean_neighbor_size_ratio"]
        r2 = df.iloc[1]["mean_neighbor_size_ratio"]
        assert 0.8 < r1 < 1.2, f"Equal neighbors should have ratio ~1, got {r1}"
        assert 0.8 < r2 < 1.2, f"Equal neighbors should have ratio ~1, got {r2}"

    def test_large_cell_has_high_ratio(self):
        # One large label, one small label, touching
        labels = np.zeros((10, 20, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:15] = 1   # large
        labels[2:8, 2:8, 15:18] = 2  # small
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        large = df[df["label"] == 1].iloc[0]
        small = df[df["label"] == 2].iloc[0]
        assert large["mean_neighbor_size_ratio"] > 1.5
        assert small["mean_neighbor_size_ratio"] < 0.7

    def test_cv_zero_for_uniform_neighbors(self):
        # One center label with two equal-sized neighbors
        labels = np.zeros((10, 10, 30), dtype=np.int32)
        labels[2:8, 2:8, 2:10] = 1
        labels[2:8, 2:8, 10:18] = 2  # center
        labels[2:8, 2:8, 18:26] = 3
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        mid = df[df["label"] == 2].iloc[0]
        assert mid["neighbor_size_cv"] < 0.2  # neighbors 1 and 3 are same size

    def test_isolated_label_has_zero_quality(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[5:15, 5:15, 5:15] = 1
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        assert df.iloc[0]["mean_neighbor_size_ratio"] == 0
        assert df.iloc[0]["neighbor_size_cv"] == 0

    def test_contact_weighted_ratio(self):
        # Two neighbors touching center, one with more contact area
        labels = np.zeros((10, 10, 30), dtype=np.int32)
        labels[2:8, 2:8, 2:10] = 1
        labels[2:8, 2:8, 10:18] = 2  # center
        labels[2:8, 2:8, 18:26] = 3
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        mid = df[df["label"] == 2].iloc[0]
        # Contact-weighted should be finite and positive
        assert mid["contact_weighted_neighbor_size_ratio"] > 0
        assert np.isfinite(mid["contact_weighted_neighbor_size_ratio"])

    def test_values_are_finite(self):
        labels = np.zeros((10, 20, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        labels[2:8, 12:18, 12:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        for col in ("mean_neighbor_size_ratio", "neighbor_size_cv",
                     "contact_weighted_neighbor_size_ratio"):
            assert np.all(np.isfinite(df[col].values)), f"Non-finite in {col}"

    def test_empty_volume(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        assert "mean_neighbor_size_ratio" in df.columns

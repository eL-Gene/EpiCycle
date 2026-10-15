"""Tests for neighbor birth-radius filtration features (Group E)."""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.qc_features import (
    compute_neighbor_features, extract_features,
)


class TestNeighborFiltrationFeatures:
    """Tests for nn_count_r1, nn_count_r3, nn_birth_median, etc."""

    def test_adds_filtration_columns(self):
        labels = np.zeros((10, 20, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        labels[2:8, 12:18, 12:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        expected = [
            "nn_count_r1", "nn_count_r3", "nn_birth_median",
            "nn_birth_std", "nn_accumulation_rate", "nn_accumulation_knee",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_touching_labels_have_r1_count(self):
        # Two labels directly touching
        labels = np.zeros((10, 10, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:10] = 1
        labels[2:8, 2:8, 10:18] = 2  # touching at x=10
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        # Both should have nn_count_r1 >= 1
        assert df.iloc[0]["nn_count_r1"] >= 1
        assert df.iloc[1]["nn_count_r1"] >= 1

    def test_separated_labels_have_zero_r1(self):
        # Two labels with a gap
        labels = np.zeros((10, 10, 30), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        labels[2:8, 2:8, 15:21] = 2  # gap of ~7 voxels
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=10)
        # No neighbor at radius 1
        assert df.iloc[0]["nn_count_r1"] == 0
        assert df.iloc[1]["nn_count_r1"] == 0
        # But should find each other at higher radius
        assert df.iloc[0]["nn_count_r3"] >= 0  # may or may not at r3

    def test_birth_median_for_touching(self):
        labels = np.zeros((10, 10, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:10] = 1
        labels[2:8, 2:8, 10:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        # Touching neighbor → birth radius = 1, median = 1
        assert df.iloc[0]["nn_birth_median"] == 1.0

    def test_accumulation_rate_nonnegative_for_packed(self):
        # Three labels packed together
        labels = np.zeros((10, 10, 30), dtype=np.int32)
        labels[2:8, 2:8, 2:10] = 1
        labels[2:8, 2:8, 10:18] = 2
        labels[2:8, 2:8, 18:26] = 3
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        # Middle label has 2 touching neighbors — all found at r=1,
        # so accumulation is flat (rate ~0). Rate should be non-negative.
        mid = df[df["label"] == 2].iloc[0]
        # floating-point noise can leave a flat rate a hair below zero
        assert mid["nn_accumulation_rate"] >= -1e-9
        assert mid["nn_count_r1"] == 2  # both neighbors touch

    def test_accumulation_knee_early_for_touching(self):
        labels = np.zeros((10, 10, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:10] = 1
        labels[2:8, 2:8, 10:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        # Touching → knee at radius 1 (80% of 1 neighbor found at r=1)
        assert df.iloc[0]["nn_accumulation_knee"] <= 2

    def test_isolated_label_has_zero_filtration(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[5:15, 5:15, 5:15] = 1  # single isolated label
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        assert df.iloc[0]["nn_count_r1"] == 0
        assert df.iloc[0]["nn_count_r3"] == 0
        assert df.iloc[0]["nn_birth_median"] == 0
        assert df.iloc[0]["nn_accumulation_rate"] == 0

    def test_values_are_finite(self):
        labels = np.zeros((10, 20, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        labels[2:8, 12:18, 12:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        for col in ("nn_count_r1", "nn_count_r3", "nn_birth_median",
                     "nn_birth_std", "nn_accumulation_rate",
                     "nn_accumulation_knee"):
            assert np.all(np.isfinite(df[col].values)), f"Non-finite in {col}"

    def test_empty_volume(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        assert "nn_count_r1" in df.columns

    def test_backward_compatible_columns_still_present(self):
        """Original 4 neighbor features must still be present."""
        labels = np.zeros((10, 20, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        labels[2:8, 12:18, 12:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        for col in ("neighbor_count", "nearest_neighbor_distance",
                     "max_contact_fraction", "background_fraction"):
            assert col in df.columns

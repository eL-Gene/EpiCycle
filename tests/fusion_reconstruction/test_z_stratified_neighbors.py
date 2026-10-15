"""Tests for Z-stratified neighbor topology features (Group F)."""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.qc_features import (
    compute_neighbor_features, extract_features,
)


class TestZStratifiedNeighborTopology:
    """Tests for basal/apical neighbor count, turnover, persistence."""

    def test_adds_topology_columns(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[2:18, 2:8, 2:8] = 1
        labels[2:18, 12:18, 12:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        for col in ("basal_neighbor_count", "apical_neighbor_count",
                     "neighbor_z_turnover", "neighbor_persistence_ratio"):
            assert col in df.columns, f"Missing column: {col}"

    def test_columnar_neighbors_have_low_turnover(self):
        """Side-by-side columns spanning full Z → same neighbors top and bottom."""
        labels = np.zeros((20, 10, 20), dtype=np.int32)
        labels[2:18, 2:8, 2:10] = 1
        labels[2:18, 2:8, 10:18] = 2  # side by side, full Z
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=3)
        # Same neighbor at top and bottom → turnover ≈ 0
        assert df.iloc[0]["neighbor_z_turnover"] < 0.3
        # Persistent → ratio close to 1
        assert df.iloc[0]["neighbor_persistence_ratio"] > 0.5

    def test_stacked_neighbors_have_high_turnover(self):
        """Vertically stacked labels → different neighbors at top vs bottom."""
        labels = np.zeros((30, 10, 10), dtype=np.int32)
        # Center cell spans full height
        labels[2:28, 2:8, 2:8] = 1
        # Bottom neighbor (only in lower third)
        labels[2:10, 2:8, 8:10] = 2  # padding needed to be adjacent
        # Top neighbor (only in upper third)
        labels[20:28, 2:8, 8:10] = 3
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=3)
        cell_1 = df[df["label"] == 1].iloc[0]
        # Different neighbors at top vs bottom → high turnover
        assert cell_1["neighbor_z_turnover"] > 0.5

    def test_basal_apical_counts(self):
        labels = np.zeros((30, 10, 20), dtype=np.int32)
        # Center cell
        labels[2:28, 2:8, 5:15] = 1
        # Bottom-only neighbor
        labels[2:10, 2:8, 15:18] = 2
        # Top-only neighbor
        labels[20:28, 2:8, 15:18] = 3
        # Full-height neighbor
        labels[2:28, 2:8, 2:5] = 4
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=3)
        cell_1 = df[df["label"] == 1].iloc[0]
        # Basal should include neighbors 2 and 4
        assert cell_1["basal_neighbor_count"] >= 2
        # Apical should include neighbors 3 and 4
        assert cell_1["apical_neighbor_count"] >= 2

    def test_thin_cell_gets_zero_topology(self):
        """Cell with < 5 Z slices → topology features should be 0."""
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[3:6, 3:7, 3:7] = 1  # only 3 Z slices
        labels[3:6, 3:7, 7:10] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=3)
        assert df.iloc[0]["neighbor_z_turnover"] == 0
        assert df.iloc[0]["neighbor_persistence_ratio"] == 0

    def test_isolated_label_zero_topology(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[5:15, 5:15, 5:15] = 1
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        assert df.iloc[0]["neighbor_z_turnover"] == 0
        assert df.iloc[0]["neighbor_persistence_ratio"] == 0
        assert df.iloc[0]["basal_neighbor_count"] == 0
        assert df.iloc[0]["apical_neighbor_count"] == 0

    def test_values_are_finite(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[2:18, 2:8, 2:8] = 1
        labels[2:18, 12:18, 12:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        for col in ("basal_neighbor_count", "apical_neighbor_count",
                     "neighbor_z_turnover", "neighbor_persistence_ratio"):
            assert np.all(np.isfinite(df[col].values)), f"Non-finite in {col}"

    def test_turnover_bounded_zero_one(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[2:18, 2:8, 2:8] = 1
        labels[2:18, 12:18, 12:18] = 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        assert np.all(df["neighbor_z_turnover"].values >= 0)
        assert np.all(df["neighbor_z_turnover"].values <= 1)
        assert np.all(df["neighbor_persistence_ratio"].values >= 0)
        assert np.all(df["neighbor_persistence_ratio"].values <= 1)

    def test_empty_volume(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        assert "neighbor_z_turnover" in df.columns

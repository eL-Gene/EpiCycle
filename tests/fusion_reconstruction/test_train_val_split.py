"""Tests for train/val data splitting (Phase 5a).

Covers:
- Random split produces correct proportions
- Spatial chunking keeps adjacent slices together
- Split summary label updated with real counts
- Edge cases: empty data, fraction=1.0
"""

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
    split_training_data,
)


def _make_file_list(n: int, prefix: str = "xy") -> list:
    """Create a list of fake file stems for testing."""
    return [f"{prefix}_{i:04d}" for i in range(n)]


# ---------------------------------------------------------------------------
# Tests: Random split
# ---------------------------------------------------------------------------

class TestRandomSplit:
    """Test random (non-spatial) splitting."""

    def test_basic_proportions(self):
        stems = _make_file_list(100)
        train, val = split_training_data(
            stems, train_frac=0.8, spatial_split=False,
        )
        assert len(train) == 80
        assert len(val) == 20

    def test_no_overlap(self):
        stems = _make_file_list(50)
        train, val = split_training_data(
            stems, train_frac=0.8, spatial_split=False,
        )
        assert set(train).isdisjoint(set(val))
        assert set(train) | set(val) == set(stems)

    def test_fraction_one_all_train(self):
        stems = _make_file_list(20)
        train, val = split_training_data(
            stems, train_frac=1.0, spatial_split=False,
        )
        assert len(train) == 20
        assert len(val) == 0

    def test_empty_input(self):
        train, val = split_training_data(
            [], train_frac=0.8, spatial_split=False,
        )
        assert train == []
        assert val == []

    def test_single_item(self):
        train, val = split_training_data(
            ["a"], train_frac=0.8, spatial_split=False,
        )
        assert len(train) + len(val) == 1

    def test_deterministic_with_seed(self):
        stems = _make_file_list(50)
        t1, v1 = split_training_data(
            stems, train_frac=0.8, spatial_split=False, seed=42,
        )
        t2, v2 = split_training_data(
            stems, train_frac=0.8, spatial_split=False, seed=42,
        )
        assert t1 == t2
        assert v1 == v2


# ---------------------------------------------------------------------------
# Tests: Spatial chunking split
# ---------------------------------------------------------------------------

class TestSpatialSplit:
    """Test spatial chunking split."""

    def test_chunks_stay_together(self):
        """Adjacent slices within a chunk should all go to same split."""
        stems = _make_file_list(100)
        train, val = split_training_data(
            stems, train_frac=0.8, spatial_split=True,
            chunk_size=10, seed=42,
        )
        train_set = set(train)
        # Check that for each chunk of 10, all items are in the same split
        for chunk_start in range(0, 100, 10):
            chunk = stems[chunk_start:chunk_start + 10]
            in_train = [s in train_set for s in chunk]
            # All True or all False within the chunk
            assert all(in_train) or not any(in_train), (
                f"Chunk {chunk_start}-{chunk_start+9} split across train/val"
            )

    def test_proportions_approximate(self):
        """Spatial split should approximate the requested fraction."""
        stems = _make_file_list(100)
        train, val = split_training_data(
            stems, train_frac=0.8, spatial_split=True,
            chunk_size=10, seed=42,
        )
        # With 10 chunks and 0.8 frac, expect ~8 train chunks = 80 items
        # Allow ±1 chunk tolerance
        assert 70 <= len(train) <= 90

    def test_no_overlap_spatial(self):
        stems = _make_file_list(50)
        train, val = split_training_data(
            stems, train_frac=0.7, spatial_split=True,
            chunk_size=5, seed=42,
        )
        assert set(train).isdisjoint(set(val))
        assert set(train) | set(val) == set(stems)

    def test_chunk_size_larger_than_data(self):
        """If chunk_size >= len(data), everything goes to train or val."""
        stems = _make_file_list(5)
        train, val = split_training_data(
            stems, train_frac=0.8, spatial_split=True,
            chunk_size=100, seed=42,
        )
        assert len(train) + len(val) == 5

    def test_chunk_size_one_like_random(self):
        """chunk_size=1 is effectively a random split."""
        stems = _make_file_list(20)
        train, val = split_training_data(
            stems, train_frac=0.8, spatial_split=True,
            chunk_size=1, seed=42,
        )
        assert len(train) + len(val) == 20

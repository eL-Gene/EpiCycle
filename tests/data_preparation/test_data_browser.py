"""Tests for Step 1 data browser controller logic.

Uses lightweight fakes — no Qt, no napari required for logic tests.
Tests the scan_tiff_folder, load/unload, and transpose workflows.
"""
import os
import tempfile

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation.logic import (
    scan_tiff_folder,
    split_channels,
    transpose_to_czyx,
)


# ---------------------------------------------------------------------------
# Scan + populate workflow
# ---------------------------------------------------------------------------

class TestScanAndPopulate:
    """Verify folder scanning discovers files with correct metadata."""

    def test_scan_returns_correct_fields(self):
        with tempfile.TemporaryDirectory() as d:
            tifffile.imwrite(os.path.join(d, "s1.tiff"), np.zeros((2, 10, 32, 32)))
            results = scan_tiff_folder(d)
            assert len(results) == 1
            r = results[0]
            assert r["filename"] == "s1.tiff"
            assert r["stem"] == "s1"
            assert r["shape"] == (2, 10, 32, 32)
            assert r["path"] == os.path.join(d, "s1.tiff")

    def test_scan_multiple_files_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["z.tiff", "a.tiff", "m.tiff"]:
                tifffile.imwrite(os.path.join(d, name), np.zeros((4, 4, 4)))
            results = scan_tiff_folder(d)
            assert [r["stem"] for r in results] == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# Load / unload workflow
# ---------------------------------------------------------------------------

class TestLoadWorkflow:
    """Test the data loading pipeline: imread → transpose → split."""

    def test_load_4d_splits_channels(self):
        """4D CZYX data should split into C separate 3D arrays."""
        data = np.random.rand(3, 10, 32, 32).astype(np.float32)
        channels = split_channels(data, "kidney")
        assert len(channels) == 3
        for i, (name, arr) in enumerate(channels):
            assert name == f"kidney_ch{i}"
            assert arr.shape == (10, 32, 32)
            np.testing.assert_array_equal(arr, data[i])

    def test_load_3d_single_channel(self):
        """3D ZYX data should return a single channel."""
        data = np.random.rand(10, 32, 32).astype(np.float32)
        channels = split_channels(data, "mono_sample")
        assert len(channels) == 1
        assert channels[0][0] == "mono_sample"
        np.testing.assert_array_equal(channels[0][1], data)

    def test_transpose_then_split(self):
        """ZCYX (10,2,32,32) → transpose → CZYX (2,10,32,32) → 2 channels."""
        data = np.random.rand(10, 2, 32, 32).astype(np.float32)
        transposed = transpose_to_czyx(data)
        assert transposed.shape == (2, 10, 32, 32)
        channels = split_channels(transposed, "sample")
        assert len(channels) == 2
        assert channels[0][1].shape == (10, 32, 32)

    def test_roundtrip_tiff_load(self):
        """Write a TIFF, read it back, split channels — verify shapes match."""
        with tempfile.TemporaryDirectory() as d:
            shape = (2, 15, 64, 64)
            original = np.random.randint(0, 255, shape, dtype=np.uint8)
            path = os.path.join(d, "sample.tiff")
            tifffile.imwrite(path, original, metadata={"axes": "CZYX"})

            loaded = tifffile.imread(path)
            assert loaded.shape == shape

            channels = split_channels(loaded, "test")
            assert len(channels) == 2
            for _, ch in channels:
                assert ch.shape == (15, 64, 64)

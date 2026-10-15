"""Tests for epicycle.utils.erosion module.

Tests cover CPU erosion fallback and helper functions.
GPU erosion tests are skipped unless CUDA is available.
"""

import numpy as np
import pytest

from packages.epicycle_scaledseg.utils.erosion import (
    get_instance_bboxes,
    _erode_instances_cpu,
    erode_instances,
    protect_border_margin,
)


class TestGetInstanceBboxes:
    def test_filters_small_instances(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[0, 0, 0] = 1  # 1 voxel — too small
        labels[3:8, 3:8, 3:8] = 2  # 125 voxels
        filtered, slices = get_instance_bboxes(labels, min_voxels=10)
        assert 1 not in filtered
        assert 2 in filtered

    def test_returns_slices(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[2:5, 2:5, 2:5] = 1
        _, slices = get_instance_bboxes(labels, min_voxels=1)
        assert slices[0] is not None


class TestErodeInstancesCPU:
    def test_basic_erosion(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[3:17, 3:17, 3:17] = 1  # 14^3 cube
        result = _erode_instances_cpu(labels, kernel_size=3, min_voxels=10)
        # Eroded cube should be smaller
        assert result.sum() > 0
        assert (result > 0).sum() < (labels > 0).sum()

    def test_preserves_label_ids(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[3:17, 3:17, 3:17] = 5
        result = _erode_instances_cpu(labels, kernel_size=3, min_voxels=10)
        unique = np.unique(result[result > 0])
        assert list(unique) == [5]

    def test_kernel_1_no_erosion(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[3:17, 3:17, 3:17] = 1
        result = _erode_instances_cpu(labels, kernel_size=1, min_voxels=10)
        np.testing.assert_array_equal(result > 0, labels > 0)

    def test_tiny_instance_preserved(self):
        """If erosion removes everything, original mask is kept."""
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[4, 4, 4] = 1  # single voxel
        result = _erode_instances_cpu(labels, kernel_size=3, min_voxels=0)
        # Should preserve because erosion would remove it entirely
        assert result[4, 4, 4] == 1


class TestErodeInstances:
    def test_force_cpu(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[3:17, 3:17, 3:17] = 1
        result = erode_instances(labels, kernel_size=3, min_voxels=10, force_cpu=True)
        assert result.sum() > 0
        assert (result > 0).sum() < (labels > 0).sum()

    def test_auto_selects_without_error(self):
        """erode_instances should work regardless of CUDA availability."""
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[3:17, 3:17, 3:17] = 1
        result = erode_instances(labels, kernel_size=3, min_voxels=10)
        assert result.sum() > 0


class TestProtectBorderMargin:
    def test_restores_borders(self):
        original = np.ones((10, 10, 10), dtype=np.int32)
        eroded = np.zeros((10, 10, 10), dtype=np.int32)
        result = protect_border_margin(eroded, original, margin=2)
        # Borders should be restored
        assert result[0, 0, 0] == 1
        assert result[9, 9, 9] == 1
        # Interior should remain eroded
        assert result[5, 5, 5] == 0

    def test_zero_margin_noop(self):
        original = np.ones((10, 10, 10), dtype=np.int32)
        eroded = np.zeros((10, 10, 10), dtype=np.int32)
        result = protect_border_margin(eroded, original, margin=0)
        np.testing.assert_array_equal(result, eroded)

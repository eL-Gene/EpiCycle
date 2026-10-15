"""Tests for orientation transpose utilities.

Phase K11d — comprehensive curation test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.orientation import (
    inverse_transpose_data,
    transpose_data,
)


class TestTransposeData3D:
    """Verify 3D (Z,Y,X) transpositions."""

    def test_xy_is_identity(self):
        data = np.arange(24).reshape(2, 3, 4)
        result = transpose_data(data, "XY")
        np.testing.assert_array_equal(result, data)

    def test_xz_transposes_axes_0_1(self):
        data = np.arange(24).reshape(2, 3, 4)
        result = transpose_data(data, "XZ")
        assert result.shape == (3, 2, 4)
        np.testing.assert_array_equal(result, data.transpose(1, 0, 2))

    def test_yz_transposes_axes_0_2(self):
        data = np.arange(24).reshape(2, 3, 4)
        result = transpose_data(data, "YZ")
        assert result.shape == (4, 3, 2)
        np.testing.assert_array_equal(result, data.transpose(2, 1, 0))

    def test_case_insensitive(self):
        data = np.arange(24).reshape(2, 3, 4)
        result_lower = transpose_data(data, "xz")
        result_upper = transpose_data(data, "XZ")
        np.testing.assert_array_equal(result_lower, result_upper)

    def test_invalid_orientation_raises(self):
        data = np.arange(24).reshape(2, 3, 4)
        with pytest.raises(ValueError, match="Unsupported orientation"):
            transpose_data(data, "ZZ")

    def test_empty_orientation_raises(self):
        data = np.arange(24).reshape(2, 3, 4)
        with pytest.raises(ValueError):
            transpose_data(data, "")

    def test_returns_view_not_copy(self):
        """Transposed result should be a numpy view sharing memory."""
        data = np.arange(24).reshape(2, 3, 4)
        result = transpose_data(data, "XZ")
        assert result.base is data or np.shares_memory(result, data)


class TestTransposeData4D:
    """Verify 4D (Z,Y,X,C) transpositions for RGB images."""

    def test_xy_4d_is_identity(self):
        data = np.arange(72).reshape(2, 3, 4, 3)
        result = transpose_data(data, "XY")
        np.testing.assert_array_equal(result, data)

    def test_xz_4d_preserves_channel_axis(self):
        data = np.arange(72).reshape(2, 3, 4, 3)
        result = transpose_data(data, "XZ")
        assert result.shape == (3, 2, 4, 3)
        np.testing.assert_array_equal(result, data.transpose(1, 0, 2, 3))

    def test_yz_4d_preserves_channel_axis(self):
        data = np.arange(72).reshape(2, 3, 4, 3)
        result = transpose_data(data, "YZ")
        assert result.shape == (4, 3, 2, 3)
        np.testing.assert_array_equal(result, data.transpose(2, 1, 0, 3))


class TestInverseTranspose:
    """Verify inverse_transpose_data round-trips correctly."""

    def test_xy_roundtrip(self):
        data = np.arange(24).reshape(2, 3, 4)
        result = inverse_transpose_data(transpose_data(data, "XY"), "XY")
        np.testing.assert_array_equal(result, data)

    def test_xz_roundtrip(self):
        data = np.arange(24).reshape(2, 3, 4)
        result = inverse_transpose_data(transpose_data(data, "XZ"), "XZ")
        np.testing.assert_array_equal(result, data)

    def test_yz_roundtrip(self):
        data = np.arange(24).reshape(2, 3, 4)
        result = inverse_transpose_data(transpose_data(data, "YZ"), "YZ")
        np.testing.assert_array_equal(result, data)

    def test_4d_xz_roundtrip(self):
        data = np.arange(72).reshape(2, 3, 4, 3)
        result = inverse_transpose_data(transpose_data(data, "XZ"), "XZ")
        np.testing.assert_array_equal(result, data)

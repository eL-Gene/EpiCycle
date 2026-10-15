"""Tests for epicycle.utils.array_ops module."""

import numpy as np
import pytest

from packages.epicycle_scaledseg.utils.array_ops import (
    recover_original_dimensions,
    split_channels,
    permute_axes,
    combine_channels,
    get_orthoview,
    convert_to_rgb_stack,
    get_tile_zranges_from_filenames,
    smooth_flows_3d,
    smooth_probs_3d,
)


class TestRecoverOriginalDimensions:
    def test_basic(self):
        region = np.ones((10, 64, 64), dtype=np.uint32)
        result = recover_original_dimensions(region, (5, 15), 100)
        assert result.shape == (100, 64, 64)
        assert result[5:15].sum() == region.sum()
        assert result[:5].sum() == 0
        assert result[15:].sum() == 0

    def test_preserves_dtype(self):
        region = np.ones((5, 8, 8), dtype=np.float16)
        result = recover_original_dimensions(region, (0, 5), 20)
        assert result.dtype == np.float16


class TestSplitChannels:
    def test_shape(self):
        stack = np.random.rand(10, 3, 64, 64).astype(np.float32)
        channels = split_channels(stack)
        assert len(channels) == 3
        assert all(c.shape == (10, 64, 64) for c in channels)


class TestPermuteAxes:
    def test_xy_noop(self):
        arr = np.random.rand(10, 64, 32)
        result = permute_axes(arr, 'xy')
        np.testing.assert_array_equal(result, arr)

    def test_xz(self):
        arr = np.random.rand(10, 64, 32)
        result = permute_axes(arr, 'xz')
        assert result.shape == (64, 10, 32)

    def test_yz(self):
        arr = np.random.rand(10, 64, 32)
        result = permute_axes(arr, 'yz')
        assert result.shape == (32, 10, 64)


class TestCombineChannels:
    def test_roundtrip(self):
        stack = np.random.randint(0, 255, (10, 2, 64, 64), dtype=np.uint8)
        channels = split_channels(stack)
        recombined = combine_channels(channels)
        np.testing.assert_array_equal(recombined, stack)


class TestGetOrthoview:
    def test_xy_preserves_shape(self):
        stack = np.random.randint(0, 255, (10, 2, 64, 32), dtype=np.uint8)
        result = get_orthoview(stack, 'xy')
        assert result.shape == stack.shape

    def test_xz_transposes(self):
        stack = np.random.randint(0, 255, (10, 2, 64, 32), dtype=np.uint8)
        result = get_orthoview(stack, 'xz')
        # xz: Z=10,X=64,Y=32 → (X=64, C=2, Z=10, Y=32)
        assert result.shape == (64, 2, 10, 32)


class TestConvertToRgbStack:
    def test_output_shape(self):
        stack = np.random.randint(0, 255, (5, 2, 32, 32), dtype=np.uint8)
        rgb = convert_to_rgb_stack(stack)
        assert rgb.shape == (5, 32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_blue_channel_zero(self):
        stack = np.random.randint(0, 255, (3, 2, 16, 16), dtype=np.uint8)
        rgb = convert_to_rgb_stack(stack)
        np.testing.assert_array_equal(rgb[:, :, :, 2], 0)

    def test_wrong_channels_raises(self):
        stack = np.random.rand(3, 3, 16, 16)
        with pytest.raises(ValueError, match="2 channels"):
            convert_to_rgb_stack(stack)

    def test_constant_slice_produces_zeros(self):
        stack = np.full((1, 2, 8, 8), 42, dtype=np.uint8)
        rgb = convert_to_rgb_stack(stack)
        # constant input → range=0 → zeros
        np.testing.assert_array_equal(rgb, 0)


class TestGetTileZranges:
    def test_parses_filenames(self, tmp_path):
        (tmp_path / "tile_0_800.tif").touch()
        (tmp_path / "tile_800_1600.tif").touch()
        (tmp_path / "noise.txt").touch()
        result = get_tile_zranges_from_filenames(str(tmp_path))
        assert result["tile_0_800.tif"] == (0, 800)
        assert result["tile_800_1600.tif"] == (800, 1600)
        assert "noise.txt" not in result


class TestSmoothProbs3d:
    def test_output_shape_matches_input(self):
        probs = np.random.rand(10, 32, 32).astype(np.float32)
        result = smooth_probs_3d(probs, sigma=(1.0, 1.0, 1.0))
        assert result.shape == probs.shape

    def test_zero_sigma_is_identity(self):
        probs = np.random.rand(10, 32, 32).astype(np.float32)
        result = smooth_probs_3d(probs, sigma=(0, 0, 0))
        np.testing.assert_array_almost_equal(result, probs)

    def test_anisotropic_smooths_along_z(self):
        """High sigma along Z should reduce variance along Z more than XY."""
        probs = np.random.rand(20, 32, 32).astype(np.float32)
        result = smooth_probs_3d(probs, sigma=(5.0, 0.0, 0.0))
        # Variance along Z should be reduced
        var_orig = np.var(probs, axis=0).mean()
        var_smooth = np.var(result, axis=0).mean()
        assert var_smooth < var_orig

    def test_preserves_dtype_float32(self):
        probs = np.random.rand(5, 8, 8).astype(np.float32)
        result = smooth_probs_3d(probs, sigma=(1.0, 1.0, 1.0))
        assert result.dtype == np.float64 or result.dtype == np.float32


class TestSmoothFlows3d:
    def test_output_shape_matches_input(self):
        flows = np.random.rand(2, 10, 32, 32).astype(np.float32)
        result = smooth_flows_3d(flows, sigma=(1.0, 1.0, 1.0))
        assert result.shape == flows.shape

    def test_zero_sigma_is_identity(self):
        flows = np.random.rand(2, 10, 32, 32).astype(np.float32)
        result = smooth_flows_3d(flows, sigma=(0, 0, 0))
        np.testing.assert_array_almost_equal(result, flows)

    def test_mask_threshold_zeros_low_prob(self):
        flows = np.ones((2, 5, 8, 8), dtype=np.float32)
        # Probs are logits: -10 → sigmoid ≈ 0, well below threshold
        probs = np.full((5, 8, 8), -10.0, dtype=np.float32)
        result = smooth_flows_3d(
            flows, probs=probs, sigma=(0, 0, 0),
            mask_threshold=0.5,
        )
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_soft_weighting_scales_by_sigmoid(self):
        flows = np.ones((2, 5, 8, 8), dtype=np.float32)
        # Probs at 0 → sigmoid = 0.5
        probs = np.zeros((5, 8, 8), dtype=np.float32)
        result = smooth_flows_3d(
            flows, probs=probs, sigma=(0, 0, 0),
            soft_weighting=True,
        )
        np.testing.assert_array_almost_equal(result, 0.5)

    def test_anisotropic_sigma_tuple(self):
        flows = np.random.rand(2, 20, 32, 32).astype(np.float32)
        result = smooth_flows_3d(flows, sigma=(3.0, 1.0, 1.0))
        assert result.shape == flows.shape
        # Should reduce variance along Z
        var_orig = np.var(flows[0], axis=0).mean()
        var_smooth = np.var(result[0], axis=0).mean()
        assert var_smooth < var_orig

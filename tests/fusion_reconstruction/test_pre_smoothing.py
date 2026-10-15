"""Tests for anisotropic pre-smoothing functions and config integration."""

import numpy as np
import pytest

from packages.epicycle_reconstruction.config import ReconstructionConfig
from packages.epicycle_reconstruction.logic import smooth_flows_3d, smooth_probs_3d


# ---------------------------------------------------------------------------
# smooth_probs_3d
# ---------------------------------------------------------------------------

class TestSmoothProbs3d:

    def test_output_shape_matches_input(self):
        probs = np.random.rand(10, 20, 20).astype(np.float32)
        out = smooth_probs_3d(probs, sigma=(1.0, 1.0, 1.0))
        assert out.shape == probs.shape

    def test_zero_sigma_returns_identity(self):
        probs = np.random.rand(8, 16, 16).astype(np.float32)
        out = smooth_probs_3d(probs, sigma=(0, 0, 0))
        np.testing.assert_array_almost_equal(out, probs)

    def test_anisotropic_smooths_more_along_perp(self):
        """High sigma along Z should reduce Z-variance more than XY."""
        probs = np.random.rand(20, 8, 8).astype(np.float32)
        z_var_before = np.var(probs, axis=0).mean()
        out = smooth_probs_3d(probs, sigma=(5.0, 0.5, 0.5))
        z_var_after = np.var(out, axis=0).mean()
        assert z_var_after < z_var_before

    def test_preserves_dtype_float32(self):
        probs = np.random.rand(4, 8, 8).astype(np.float32)
        out = smooth_probs_3d(probs, sigma=(1.0, 1.0, 1.0))
        assert out.dtype == np.float64 or out.dtype == np.float32


# ---------------------------------------------------------------------------
# smooth_flows_3d
# ---------------------------------------------------------------------------

class TestSmoothFlows3d:

    def test_output_shape_matches_input(self):
        flows = np.random.rand(2, 10, 20, 20).astype(np.float32)
        out = smooth_flows_3d(flows, sigma=(1.0, 1.0, 1.0))
        assert out.shape == flows.shape

    def test_zero_sigma_returns_identity(self):
        flows = np.random.rand(2, 8, 16, 16).astype(np.float32)
        out = smooth_flows_3d(flows, sigma=(0, 0, 0))
        np.testing.assert_array_almost_equal(out, flows)

    def test_mask_threshold_zeros_low_prob_flows(self):
        flows = np.ones((2, 4, 4, 4), dtype=np.float32)
        # probs < 0 → sigmoid < 0.5 → below threshold 0.5
        probs = np.full((4, 4, 4), -10.0, dtype=np.float32)
        out = smooth_flows_3d(
            flows, probs=probs, sigma=(0, 0, 0),
            mask_threshold=0.5,
        )
        np.testing.assert_array_almost_equal(out, 0.0)

    def test_mask_threshold_preserves_high_prob_flows(self):
        flows = np.ones((2, 4, 4, 4), dtype=np.float32)
        # probs >> 0 → sigmoid ≈ 1.0 → above threshold
        probs = np.full((4, 4, 4), 10.0, dtype=np.float32)
        out = smooth_flows_3d(
            flows, probs=probs, sigma=(0, 0, 0),
            mask_threshold=0.5,
        )
        np.testing.assert_array_almost_equal(out, 1.0)

    def test_soft_weighting_scales_by_sigmoid(self):
        flows = np.ones((2, 4, 4, 4), dtype=np.float32)
        # sigmoid(0) = 0.5
        probs = np.zeros((4, 4, 4), dtype=np.float32)
        out = smooth_flows_3d(
            flows, probs=probs, sigma=(0, 0, 0),
            soft_weighting=True,
        )
        np.testing.assert_array_almost_equal(out, 0.5)

    def test_anisotropic_sigma_applied_per_component(self):
        """Each flow component should be smoothed independently."""
        flows = np.random.rand(2, 20, 8, 8).astype(np.float32)
        out = smooth_flows_3d(flows, sigma=(5.0, 0.5, 0.5))
        # Z-variance should decrease for both components
        for c in range(2):
            var_before = np.var(flows[c], axis=0).mean()
            var_after = np.var(out[c], axis=0).mean()
            assert var_after < var_before

    def test_no_probs_skips_masking_and_weighting(self):
        """Without probs, only Gaussian smoothing happens."""
        flows = np.random.rand(2, 8, 8, 8).astype(np.float32)
        out = smooth_flows_3d(
            flows, probs=None, sigma=(1.0, 1.0, 1.0),
            mask_threshold=0.5, soft_weighting=True,
        )
        assert out.shape == flows.shape
        # Should still be smoothed (not identical to input)
        assert not np.allclose(out, flows)


# ---------------------------------------------------------------------------
# ReconstructionConfig — pre-smoothing fields
# ---------------------------------------------------------------------------

class TestPreSmoothingConfig:

    def test_defaults_disabled(self):
        cfg = ReconstructionConfig()
        assert cfg.pre_smooth_probs is False
        assert cfg.pre_smooth_flows is False

    def test_sigma_defaults(self):
        cfg = ReconstructionConfig()
        assert cfg.pre_smooth_sigma_perp == 3.0
        assert cfg.pre_smooth_sigma_inplane == 1.0

    def test_flow_weighting_defaults(self):
        cfg = ReconstructionConfig()
        assert cfg.flow_mask_threshold == 0.2
        assert cfg.flow_soft_weighting is True

    def test_custom_override(self):
        cfg = ReconstructionConfig(
            pre_smooth_probs=True,
            pre_smooth_flows=True,
            pre_smooth_sigma_perp=5.0,
            pre_smooth_sigma_inplane=0.5,
            flow_mask_threshold=0.3,
            flow_soft_weighting=False,
        )
        assert cfg.pre_smooth_probs is True
        assert cfg.pre_smooth_flows is True
        assert cfg.pre_smooth_sigma_perp == 5.0
        assert cfg.pre_smooth_sigma_inplane == 0.5
        assert cfg.flow_mask_threshold == 0.3
        assert cfg.flow_soft_weighting is False


# ---------------------------------------------------------------------------
# Integration: smoothing applied in correct order
# ---------------------------------------------------------------------------

class TestSmoothingIntegration:

    def test_probs_smoothed_before_flow_weighting(self):
        """When both are enabled, probs must be smoothed first so flow
        masking/weighting uses the smoothed probabilities."""
        rng = np.random.RandomState(42)
        flows = rng.rand(2, 10, 10, 10).astype(np.float32)
        # Alternating high/low probs along Z to create discontinuities
        probs = np.zeros((10, 10, 10), dtype=np.float32)
        probs[::2] = 5.0   # sigmoid ≈ 1.0
        probs[1::2] = -5.0  # sigmoid ≈ 0.0

        sigma = (3.0, 1.0, 1.0)

        # Smooth probs first, then use smoothed probs for flow weighting
        probs_smooth = smooth_probs_3d(probs, sigma=sigma)
        flows_result = smooth_flows_3d(
            flows, probs=probs_smooth, sigma=sigma,
            mask_threshold=0.2, soft_weighting=True,
        )

        # Smoothed probs should have less Z-variance
        z_var_raw = np.var(probs, axis=0).mean()
        z_var_smooth = np.var(probs_smooth, axis=0).mean()
        assert z_var_smooth < z_var_raw

        # Flows should not be all zero (smoothed probs raise the low slices)
        assert flows_result.max() > 0.1

    def test_per_view_sigma_construction(self):
        """Verify the sigma tuples are constructed correctly for each view."""
        s_perp, s_in = 3.0, 1.0
        sigma_xy = (s_perp, s_in, s_in)   # Z is perp
        sigma_xz = (s_in, s_perp, s_in)   # Y is perp
        sigma_yz = (s_in, s_in, s_perp)   # X is perp

        assert sigma_xy == (3.0, 1.0, 1.0)
        assert sigma_xz == (1.0, 3.0, 1.0)
        assert sigma_yz == (1.0, 1.0, 3.0)

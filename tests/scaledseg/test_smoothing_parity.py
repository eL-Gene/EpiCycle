"""
Smoothing + reconstruction-param parity lock (Stack 3).

Locks the anisotropic pre-smoothing math and the reconstruction-param tuning
against the EpiCycle monorepo (packages/epicycle_reconstruction), which is the
declared single source of truth. The two repos are a MIRROR pair; these tests
make any drift LOUD (a red test) instead of silent on an HPC run.

What this stack changes in production:
  * defaults.py: renorm_gradient False -> True (the monorepo's deliberate
    tuning; locked decision Q1). This resolves the Stack-2 xfail bug-lock
    `test_stage2_direct_renorm_gradient_should_be_true` (now a plain pass).
  * array_ops.py: extracted the per-view sigma build into `per_view_sigmas`
    so the perpendicular-axis assignment is unit-testable without a GPU.
  * reconstruction.py: uses `per_view_sigmas` (behavior-preserving).

Smoothing math itself needed NO porting — `smooth_probs_3d`/`smooth_flows_3d`
are already byte-for-byte identical to the monorepo (verified 2026-06-12). The
"float64 upcast" the survey worried about is REFUTED here, not "fixed".
"""

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from packages.epicycle_scaledseg.utils.array_ops import (
    per_view_sigmas,
    smooth_flows_3d,
    smooth_probs_3d,
)

pytestmark = pytest.mark.characterization

_REFERENCE = Path(__file__).resolve().parent / "fixtures" / "reconstruction_params.reference.json"


# ---------------------------------------------------------------------------
# Per-view sigma assignment (the extracted pure helper)
# ---------------------------------------------------------------------------


def test_per_view_sigmas_perp_axis_assignment():
    """Each view's PERPENDICULAR axis carries sigma_perp; in-plane axes carry
    sigma_inplane. Catches an XZ/YZ (or XY) perpendicular-axis swap that a
    full-reconstruction test would never isolate. perp=3, in=1:
      XY -> Z perp -> (3,1,1); XZ -> Y perp -> (1,3,1); YZ -> X perp -> (1,1,3)."""
    sigma_xy, sigma_xz, sigma_yz = per_view_sigmas(3.0, 1.0)
    assert sigma_xy == (3.0, 1.0, 1.0)  # XY: perp on axis 0 (Z)
    assert sigma_xz == (1.0, 3.0, 1.0)  # XZ: perp on axis 1 (Y)
    assert sigma_yz == (1.0, 1.0, 3.0)  # YZ: perp on axis 2 (X)


def test_per_view_sigmas_each_view_perp_is_distinct_axis():
    """The perp value lands on a DIFFERENT axis for each view (no two views
    share a perpendicular axis) — the property the anisotropy relies on."""
    sigmas = per_view_sigmas(9.0, 1.0)
    perp_axis = [s.index(9.0) for s in sigmas]
    assert perp_axis == [0, 1, 2]


# ---------------------------------------------------------------------------
# Smoothing math parity vs the monorepo (verbatim reference impls)
# ---------------------------------------------------------------------------


def _monorepo_smooth_probs_3d(probs, sigma=(1.0, 1.0, 1.0)):
    """VERBATIM copy of packages/epicycle_reconstruction/logic.py:35-56
    (as of 2026-06-12). Drift guard: the standalone must match this exactly."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(probs, sigma=sigma)


def _monorepo_smooth_flows_3d(flows, probs=None, sigma=(1.0, 1.0, 1.0),
                              mask_threshold=None, soft_weighting=False):
    """VERBATIM copy of packages/epicycle_reconstruction/logic.py:59-95."""
    from scipy.ndimage import gaussian_filter
    from scipy.special import expit
    flows_smooth = np.zeros_like(flows)
    for c in range(flows.shape[0]):
        flows_smooth[c] = gaussian_filter(flows[c], sigma=sigma)
    if probs is not None:
        prob_sigmoid = expit(probs)
        if mask_threshold is not None:
            mask = (prob_sigmoid >= mask_threshold).astype(np.float32)
            flows_smooth *= mask[np.newaxis, ...]
        if soft_weighting:
            flows_smooth *= prob_sigmoid[np.newaxis, ...]
    return flows_smooth


def test_smooth_probs_matches_monorepo_reference():
    rng = np.random.default_rng(0)
    x = rng.random((6, 8, 8)).astype(np.float32)
    sigma = (3.0, 1.0, 1.0)
    got = smooth_probs_3d(x, sigma=sigma)
    ref = _monorepo_smooth_probs_3d(x, sigma=sigma)
    assert np.array_equal(got, ref)
    # array_equal is dtype-blind (float64==float32 if values match), so assert
    # dtype explicitly — this is what would catch a stray .astype upcast.
    assert got.dtype == ref.dtype


def test_smooth_flows_matches_monorepo_reference():
    rng = np.random.default_rng(1)
    flows = rng.random((2, 6, 8, 8)).astype(np.float32)
    probs = rng.random((6, 8, 8)).astype(np.float32)
    sigma = (1.0, 3.0, 1.0)
    got = smooth_flows_3d(flows, probs=probs, sigma=sigma,
                          mask_threshold=0.2, soft_weighting=True)
    ref = _monorepo_smooth_flows_3d(flows, probs=probs, sigma=sigma,
                                    mask_threshold=0.2, soft_weighting=True)
    assert np.array_equal(got, ref)
    assert got.dtype == ref.dtype  # dtype-blind array_equal needs this


# ---------------------------------------------------------------------------
# dtype: REFUTE the survey's "float64 upcast" claim (verify, don't fix)
# ---------------------------------------------------------------------------


def test_smooth_probs_preserves_float32_dtype():
    """The survey claimed smooth_probs_3d upcasts to float64 — REFUTED here.
    scipy.ndimage.gaussian_filter preserves the input dtype, so NO `.astype`
    'fix' is warranted. If this ever fails, investigate the premise first."""
    out = smooth_probs_3d(np.zeros((4, 4, 4), dtype=np.float32))
    assert out.dtype == np.float32


def test_smooth_flows_preserves_float32_dtype():
    out = smooth_flows_3d(np.zeros((2, 4, 4, 4), dtype=np.float32))
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Pre-smooth default OFF (opt-in) — both flags False
# ---------------------------------------------------------------------------


def test_reconstruction_stage_pre_smooth_defaults_off():
    """Pre-smoothing is opt-in: ReconstructionStage defaults both flags False,
    matching the monorepo (config.py:41-42). Importing the stage is segment3D-free."""
    from packages.epicycle_scaledseg.core.reconstruction import ReconstructionStage
    stage = ReconstructionStage()
    assert stage.pre_smooth_flows is False
    assert stage.pre_smooth_probs is False


# ---------------------------------------------------------------------------
# Stage 4 (global indirect) must NOT pre-smooth — it operates on label images
# ---------------------------------------------------------------------------


def test_stage4_does_not_reference_smoothing():
    """Stage 4 derives EDT gradients from label images internally; there are no
    per-view flow/prob arrays to pre-smooth. Lock that it never calls the
    smoothing helpers, so a naive 'smooth Stage 4' PR fails loudly here.

    NOTE: this is a source-text tripwire, not a proof — it catches a direct
    call (the likely way someone adds smoothing) but could be defeated by an
    aliased import or an indirect call. A runtime spy would need segment3D."""
    from packages.epicycle_scaledseg.core import global_reconstruction
    src = inspect.getsource(global_reconstruction)
    assert "smooth_flows_3d" not in src
    assert "smooth_probs_3d" not in src
    assert "per_view_sigmas" not in src


def test_pre_smooth_off_gate_wraps_smoothing_calls():
    """OFF path is a true no-op: the `smooth_*` calls (and the helper import)
    live INSIDE `if pre_smooth_probs or pre_smooth_flows:`, so both-flags-False
    skips them entirely. A runtime call_count spy would need segment3D (the
    worker imports it before reaching the gate), so this locks the gate
    STRUCTURALLY instead — and confirms `per_view_sigmas` feeds the sigmas.
    Mirrors the Stage-4 source-scan tripwire (same caveat applies)."""
    from packages.epicycle_scaledseg.core import reconstruction
    src = inspect.getsource(reconstruction)
    gate = "if pre_smooth_probs or pre_smooth_flows:"
    assert gate in src
    g = src.index(gate)
    # per_view_sigmas builds the sigmas, after the gate.
    assert "per_view_sigmas(" in src and src.index("per_view_sigmas(") > g
    # The smooth_* CALLS (with paren; the import has none) sit after the gate.
    for call in ("smooth_probs_3d(", "smooth_flows_3d("):
        assert call in src and src.index(call) > g


def test_renorm_gradient_true_is_locked_segment3d_free():
    """The headline Q1 decision (renorm_gradient=True) is enforced WITHOUT
    segment3D, so it's covered in CI too — `get_default_reconstruct_params`
    imports segment3D, so its value test would otherwise only run on dev/HPC.
    Source-text tripwire on the literal assignment in defaults.py."""
    from packages.epicycle_scaledseg.config import defaults
    src = inspect.getsource(defaults.get_default_reconstruct_params)
    assert "renorm_gradient'] = True" in src


# ---------------------------------------------------------------------------
# Golden-reference param parity (the drift-management lock)
# ---------------------------------------------------------------------------


def test_reconstruct_params_match_reference():
    """The shared deliberately-tuned gradient_descent values (incl.
    renorm_gradient=True after the Stack-3 flip) match the committed reference
    snapshot of the monorepo's config.py. Drift -> red test."""
    pytest.importorskip("segment3D")
    from packages.epicycle_scaledseg.config.defaults import get_default_reconstruct_params
    ref = json.loads(_REFERENCE.read_text())["gradient_descent"]
    gd = get_default_reconstruct_params()["gradient_descent"]
    for key, expected in ref.items():
        assert gd[key] == expected, f"gradient_descent[{key!r}] drift: {gd[key]} != {expected}"


def test_reconstruction_stage_pre_smooth_matches_reference():
    """ReconstructionStage pre_smooth defaults mirror the monorepo reference
    (segment3D-free — these are plain constructor defaults)."""
    from packages.epicycle_scaledseg.core.reconstruction import ReconstructionStage
    ref = json.loads(_REFERENCE.read_text())["pre_smooth_defaults"]
    stage = ReconstructionStage()
    assert stage.pre_smooth_probs == ref["pre_smooth_probs"]
    assert stage.pre_smooth_flows == ref["pre_smooth_flows"]
    assert stage.pre_smooth_sigma_perp == ref["pre_smooth_sigma_perp"]
    assert stage.pre_smooth_sigma_inplane == ref["pre_smooth_sigma_inplane"]
    assert stage.flow_mask_threshold == ref["flow_mask_threshold"]
    assert stage.flow_soft_weighting == ref["flow_soft_weighting"]

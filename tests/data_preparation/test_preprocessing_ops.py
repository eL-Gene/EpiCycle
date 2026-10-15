"""Brick 2a tests — the pure Step-1 preprocessing op math (``ops.py``).

Covers the preprocessing op set: each op vs a
KNOWN small array; the hoisted percentile-normalize working space; the shared
geometry resample (canonical mode/method tokens); the uint16 bake dtype; and the
``execute`` pipeline (per-channel normalize-first, correct flatten order, shared
geometry re-grids every channel identically, corrupt recipes never raise).

Pure — no napari / Qt.  All ops operate on normalized float32 in [0, 1] and
return float32 in [0, 1] (except the geometry resampler + the uint16 bake).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

_OPS_LOGGER = "packages.epicycle_preprocessing.ops"

from packages.epicycle_preprocessing import ops
from packages.epicycle_preprocessing import registry as reg
from packages.epicycle_preprocessing.recipe import (
    GeometryStep,
    IntensityStep,
    Recipe,
    default_geometry_config,
    default_recipe,
)


# --------------------------------------------------------------------------- #
# gaussian
# --------------------------------------------------------------------------- #

def test_gaussian_spreads_a_delta_and_preserves_sum():
    d = np.zeros((11, 11, 11), dtype=np.float32)
    d[5, 5, 5] = 1.0
    out = ops.gaussian(d, sigma=1.5)

    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Peak spreads: the delta drops, an immediate neighbour rises off zero.
    assert out[5, 5, 5] < 1.0 and out[5, 5, 5] > 0.0
    assert out[5, 5, 6] > 0.0
    # Sum preserved-ish (reflect boundary conserves mass for a centred delta).
    assert abs(float(out.sum()) - 1.0) < 0.02


def test_gaussian_zero_sigma_is_noop():
    a = np.random.RandomState(0).rand(6, 6, 6).astype(np.float32)
    assert np.array_equal(ops.gaussian(a, sigma=0.0), a)


# --------------------------------------------------------------------------- #
# clahe
# --------------------------------------------------------------------------- #

def test_clahe_changes_local_contrast_and_stays_in_unit_range():
    rng = np.random.RandomState(1)
    patch = np.clip(0.45 + 0.02 * rng.rand(24, 24, 24), 0.0, 1.0).astype(np.float32)
    out = ops.clahe(patch, clip=0.01, kernel=8)

    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # A low-contrast patch is genuinely changed (contrast stretched locally).
    assert np.abs(out - patch).max() > 1e-3


def test_clahe_constant_input_is_noop():
    const = np.full((16, 16, 16), 0.5, dtype=np.float32)
    out = ops.clahe(const, clip=0.01, kernel=8)
    # CLAHE is undefined on a constant image; guard returns it unchanged.
    assert np.array_equal(out, const)


def test_clahe_default_kernel_larger_than_array_ok():
    rng = np.random.RandomState(2)
    patch = np.clip(0.4 + 0.1 * rng.rand(12, 12, 12), 0.0, 1.0).astype(np.float32)
    out = ops.clahe(patch, clip=0.01, kernel=64)  # kernel > array dims
    assert out.shape == patch.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


# --------------------------------------------------------------------------- #
# background (white top-hat)
# --------------------------------------------------------------------------- #

def test_background_removes_gradient_keeps_small_bright_spot():
    # Linear gradient background (0.2..0.7 along Y) + one small bright voxel.
    _, y, _ = np.mgrid[0:20, 0:40, 0:40]
    vol = (0.2 + 0.5 * (y / 39.0)).astype(np.float32)
    vol[10, 20, 20] += 0.25  # a small bright spot (smaller than the SE)
    out = ops.background(vol, radius=6)

    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # A pure-background interior region collapses to ~0 (gradient removed).
    # Tolerance 0.05 (not 0.02): the fast `decomposition="sequence"` ball is an
    # approximation of the solid ball (~0.02 max|Δ| on [0,1], data-dependent), so
    # the residual background can sit slightly above an ultra-tight 0.02 bound.
    bg = out[5:15, 5:15, 5:15]
    assert float(bg.max()) < 0.05
    # The small bright spot survives the top-hat.
    assert float(out[10, 20, 20]) > 0.1


def test_background_zero_radius_is_noop():
    a = np.random.RandomState(3).rand(6, 6, 6).astype(np.float32)
    assert np.array_equal(ops.background(a, radius=0), a)


# --------------------------------------------------------------------------- #
# bleach
# --------------------------------------------------------------------------- #

def _exp_decay_volume(z=12, k=0.15, seed=4):
    base = (0.1 + 0.6 * np.random.RandomState(seed).rand(24, 24)).astype(np.float32)
    return np.stack([base * np.exp(-k * zz) for zz in range(z)]).astype(np.float32), base


def _increasing_volume(z=12, k=0.10, seed=7):
    """A stack that BRIGHTENS with depth, kept inside [0, 1] (not a bleach
    signature) — for the ``on_non_decreasing`` behaviour tests."""
    base = (0.05 + 0.05 * np.random.RandomState(seed).rand(24, 24)).astype(np.float32)
    vol = np.stack([base * np.exp(k * zz) for zz in range(z)]).astype(np.float32)
    return vol, base


def test_bleach_flattens_exp_z_decay():
    vol, _ = _exp_decay_volume()
    z = vol.shape[0]
    means_before = vol.reshape(z, -1).mean(axis=1)
    out = ops.bleach(vol, model="exp")
    means_after = out.reshape(z, -1).mean(axis=1)

    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Decay is real before; ~flat (all slice means equal) after.
    assert float(means_before.std()) > 0.05
    assert float(means_after.std()) < 1e-3
    assert float(means_after.max() - means_after.min()) < 1e-3


def test_bleach_2d_and_single_slice_are_noops():
    two_d = np.random.RandomState(5).rand(16, 16).astype(np.float32)
    assert np.array_equal(ops.bleach(two_d), two_d)
    one_slice = np.random.RandomState(6).rand(1, 8, 8).astype(np.float32)
    assert np.array_equal(ops.bleach(one_slice), one_slice)


def test_bleach_unknown_model_is_noop():
    vol, _ = _exp_decay_volume()
    assert np.array_equal(ops.bleach(vol, model="linear"), vol)


def test_bleach_mean_reference_flattens_without_extra_overshoot():
    """Defect 1: the MEAN fitted level is the reference, not the first/brightest
    slice.  The Z-mean profile still flattens, and the pre-clip result never
    overshoots MORE than the old ``fitted[0]/fitted`` reference would have."""
    vol, _ = _exp_decay_volume()
    z = vol.shape[0]
    means_before = vol.reshape(z, -1).mean(axis=1)
    out = ops.bleach(vol, model="exp")
    means_after = out.reshape(z, -1).mean(axis=1)

    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    # Genuinely flatter: decay is real before, ~flat after.
    assert float(means_before.std()) > 0.05
    assert float(means_after.std()) < float(means_before.std())
    assert float(means_after.std()) < 1e-3

    # THE discriminator vs the old first-slice reference: the mean reference dims
    # the bright FIRST slice (correction = mean(fitted)/fitted[0] < 1), whereas
    # `fitted[0]/fitted` left slice 0 untouched (correction = 1) and multiplied
    # the deep slices by >1.  So the first slice must now be dimmed.
    assert float(out[0].max()) < float(vol[0].max()), (
        "the mean reference must dim the bright first slice (old first-slice "
        "reference left it unchanged)"
    )

    # No pre-clip overshoot beyond the OLD first-slice reference: since
    # mean(fitted) <= fitted[0], the mean-ref correction is <= the first-ref
    # correction everywhere, so its pre-clip product cannot exceed the old one.
    means = vol.reshape(z, -1).mean(axis=1)
    zs = np.arange(z, dtype=np.float64)
    slope, intercept = np.polyfit(zs, np.log(means), 1)
    fitted = np.exp(intercept + slope * zs)
    corr_mean = (float(np.mean(fitted)) / fitted).reshape((z, 1, 1)).astype(np.float32)
    corr_first = (fitted[0] / fitted).reshape((z, 1, 1)).astype(np.float32)
    preclip_mean = vol * corr_mean
    preclip_first = vol * corr_first
    assert float(preclip_mean.max()) <= float(preclip_first.max()) + 1e-6


def test_bleach_non_decreasing_skip_is_noop_and_logs(caplog):
    """Defect 2: a brightening (non-decreasing) profile is not a bleach
    signature.  ``on_non_decreasing="skip"`` (default) returns the input
    unchanged AND emits a visible status line."""
    vol, _ = _increasing_volume()
    with caplog.at_level(logging.INFO, logger=_OPS_LOGGER):
        out = ops.bleach(vol, model="exp", on_non_decreasing="skip")
    assert np.array_equal(out, vol), "skip must leave a non-decreasing profile untouched"
    assert any("not decreasing" in r.getMessage() for r in caplog.records), (
        "skip must emit a status line naming the non-decreasing profile"
    )


def test_bleach_non_decreasing_flatten_changes_it():
    """Defect 2: ``on_non_decreasing="flatten"`` opts into the both-directions
    rescale, so a brightening profile IS altered (and stays in [0, 1])."""
    vol, _ = _increasing_volume()
    out = ops.bleach(vol, model="exp", on_non_decreasing="flatten")
    assert not np.array_equal(out, vol), "flatten must apply a correction"
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # It really flattens the brightening: the Z-mean spread shrinks.
    z = vol.shape[0]
    before = float(vol.reshape(z, -1).mean(axis=1).std())
    after = float(out.reshape(z, -1).mean(axis=1).std())
    assert after < before


def test_bleach_underflow_guard_prevents_nan(caplog):
    """Defect 3: a steep negative slope underflows exp() to 0.0 at depth (finite,
    so a bare isfinite check passes) — the old ``fitted[0] > 0`` guard let this
    through and divided → inf/NaN.  The all-``> 0`` guard must fire instead."""
    z, y, x = 50, 4, 4
    vol = np.zeros((z, y, x), dtype=np.float32)
    # A very steep decay in the first few slices → slope ≈ -20; the fitted line
    # evaluated over zs=0..49 underflows exp() to 0.0 well before z=49.  The
    # deeper all-zero slices are excluded from the fit but keep the zs span.
    for zz in range(5):
        vol[zz] = np.float32(np.exp(-20.0 * zz))

    with caplog.at_level(logging.INFO, logger=_OPS_LOGGER):
        out = ops.bleach(vol, model="exp")

    assert out.dtype == np.float32
    assert np.all(np.isfinite(out)), "the all-positive fitted guard must prevent inf/NaN"
    # Guard fired → input returned unchanged.
    assert np.array_equal(out, vol)
    assert any("underflowed" in r.getMessage() or "non-positive" in r.getMessage()
               for r in caplog.records)


# --------------------------------------------------------------------------- #
# percentile_normalize (the hoisted, non-removable first stage)
# --------------------------------------------------------------------------- #

def test_percentile_normalize_maps_to_unit_range_and_clips_outliers():
    a = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    a[0, 0, 0] = 100000.0  # a wild high outlier
    out = ops.percentile_normalize(a, low=0.1, high=99.5)

    assert out.dtype == np.float32
    assert float(out.min()) == 0.0
    assert float(out.max()) == 1.0
    # The outlier is clipped to the top of the range, not left dominating.
    assert float(out[0, 0, 0]) == 1.0


def test_percentile_normalize_flat_input_does_not_raise():
    flat = np.full((5, 5, 5), 3.0, dtype=np.float32)
    out = ops.percentile_normalize(flat)  # must NOT raise (div-by-zero guarded)
    assert out.shape == flat.shape
    assert np.all(out == 0.0)


def test_percentile_normalize_empty_and_nan_inputs_do_not_raise():
    empty = np.zeros((0,), dtype=np.float32)
    assert ops.percentile_normalize(empty).size == 0
    all_nan = np.full((4, 4), np.nan, dtype=np.float32)
    out = ops.percentile_normalize(all_nan)
    assert np.all(out == 0.0)


def test_percentile_bounds_is_single_source_of_truth_for_normalize():
    """`percentile_bounds` is the ONE place the clip bounds live; a ``None`` from
    it is EXACTLY the degenerate case where `percentile_normalize` returns
    all-zeros (so the two can never drift apart)."""
    # Normal crop → finite (lo, hi) with hi > lo.
    a = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    bounds = ops.percentile_bounds(a, low=0.1, high=99.5)
    assert bounds is not None
    lo, hi = bounds
    assert hi > lo
    # normalize's output IS clip((a - lo)/(hi - lo)) using those exact bounds.
    recomputed = np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    np.testing.assert_array_equal(ops.percentile_normalize(a, 0.1, 99.5), recomputed)
    # Degenerate / empty / all-NaN → None, and normalize → all-zeros there.
    for degenerate in (
        np.full((5, 5), 3.0, np.float32),
        np.zeros((0,), np.float32),
        np.full((4, 4), np.nan, np.float32),
    ):
        assert ops.percentile_bounds(degenerate) is None
    assert np.all(ops.percentile_normalize(np.full((5, 5), 3.0, np.float32)) == 0.0)


def test_percentile_normalize_uses_explicit_bounds():
    # With an explicit window, normalize maps lo→0 / hi→1 using THOSE endpoints
    # (the shared whole-volume window), not the array's own percentiles.
    a = np.linspace(0.0, 10.0, 100, dtype=np.float32)
    out = ops.percentile_normalize(a, bounds=(2.0, 8.0))
    assert float(out.min()) == 0.0 and float(out.max()) == 1.0
    assert float(out[0]) == 0.0    # 0.0 < 2.0 → clips low
    assert float(out[-1]) == 1.0   # 10.0 > 8.0 → clips high
    mid = float(np.interp(5.0, a, out))
    assert abs(mid - 0.5) < 0.02   # midpoint of (2,8) → 0.5


def test_percentile_normalize_degenerate_explicit_bounds_are_zeros():
    a = np.linspace(0, 1, 20, dtype=np.float32)
    assert np.all(ops.percentile_normalize(a, bounds=(0.5, 0.5)) == 0.0)  # hi<=lo


def test_percentile_normalize_malformed_bounds_fall_back_not_raise():
    # A malformed explicit window must NOT raise out of execute() (which
    # advertises "never raises") — it falls back to the array's own percentiles.
    a = np.linspace(0.0, 10.0, 100, dtype=np.float32)
    ref = ops.percentile_normalize(a)  # the from-data result
    for bad in [(5.0,), 5.0, (None, 2.0), ("x", "y"), ()]:
        out = ops.percentile_normalize(a, bounds=bad)
        assert out.shape == a.shape and out.dtype == np.float32
        np.testing.assert_array_equal(out, ref)  # fell back to per-array percentiles


def test_execute_malformed_bounds_entry_never_raises():
    ch = np.linspace(0.0, 1.0, 4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    recipe = Recipe(channels=[[]], geometry=None)
    # A malformed bounds entry for the channel → falls back, no raise.
    processed, _, _ = ops.execute(recipe, [ch], (1, 1, 1),
                                  bounds_per_channel=[(5.0,)])
    assert len(processed) == 1 and processed[0].dtype == np.float32


# --------------------------------------------------------------------------- #
# apply_geometry + method_to_order
# --------------------------------------------------------------------------- #

def test_method_to_order_maps_canonical_tokens():
    assert ops.method_to_order("nearest") == 0
    assert ops.method_to_order("linear") == 1
    assert ops.method_to_order("cubic") == 3
    assert ops.method_to_order("area") == 1  # fallback order when area can't block-mean
    assert ops.method_to_order("Linear") == 1  # case-insensitive
    assert ops.method_to_order("bogus") == 1  # unknown → linear


def test_apply_geometry_iso_resamples_to_expected_shape():
    arr = np.random.RandomState(7).rand(10, 20, 20).astype(np.float32)
    config = {"mode": "iso", "target_um": 1.0, "factors": (1.0, 1.0, 1.0),
              "method": "linear"}
    (out,), new_spacing = ops.apply_geometry([arr], (2.0, 0.5, 0.5), config)

    # factors = spacing/target = (2.0, 0.5, 0.5) → (20, 10, 10)
    assert out.shape == (20, 10, 10)
    assert out.dtype == np.float32
    # Physical extent preserved → new voxel size ≈ target (1 µm) each axis.
    assert new_spacing == pytest.approx((1.0, 1.0, 1.0))


def test_apply_geometry_aniso_uses_factors_directly():
    arr = np.random.RandomState(8).rand(8, 16, 16).astype(np.float32)
    config = {"mode": "aniso", "target_um": 1.0, "factors": (2.0, 1.0, 0.5),
              "method": "nearest"}
    (out,), new_spacing = ops.apply_geometry([arr], (1.0, 1.0, 1.0), config)
    assert out.shape == (16, 16, 8)
    # new spacing = old * old_shape / new_shape = (0.5, 1.0, 2.0)
    assert new_spacing == pytest.approx((0.5, 1.0, 2.0))


def test_apply_geometry_regrids_all_channels_identically():
    rng = np.random.RandomState(9)
    ch0 = rng.rand(6, 12, 12).astype(np.float32)
    ch1 = rng.rand(6, 12, 12).astype(np.float32)
    config = {"mode": "iso", "target_um": 1.0, "factors": (1.0, 1.0, 1.0),
              "method": "linear"}
    out, _ = ops.apply_geometry([ch0, ch1], (2.0, 1.0, 1.0), config)
    assert len(out) == 2
    assert out[0].shape == out[1].shape == (12, 12, 12)  # Z doubled for both


def test_apply_geometry_noop_returns_copies_and_same_spacing():
    arr = np.random.RandomState(10).rand(5, 5, 5).astype(np.float32)
    config = {"mode": "iso", "target_um": 1.0, "factors": (1.0, 1.0, 1.0),
              "method": "linear"}
    (out,), new_spacing = ops.apply_geometry([arr], (1.0, 1.0, 1.0), config)
    assert np.array_equal(out, arr)
    assert out is not arr  # a copy, not the same object
    assert new_spacing == (1.0, 1.0, 1.0)


def test_apply_geometry_empty_channels():
    out, sp = ops.apply_geometry([], (1.0, 1.0, 1.0), default_geometry_config())
    assert out == []
    assert sp == (1.0, 1.0, 1.0)


def test_apply_geometry_area_downscale_uses_block_mean():
    arr = np.random.RandomState(11).rand(10, 20, 20).astype(np.float32)
    config = {"mode": "aniso", "target_um": 1.0, "factors": (1.0, 0.5, 0.5),
              "method": "area"}
    (out,), _ = ops.apply_geometry([arr], (1.0, 1.0, 1.0), config)
    assert out.shape == (10, 10, 10)  # block-mean by (1, 2, 2)
    assert out.min() >= 0.0 and out.max() <= 1.0


# --------------------------------------------------------------------------- #
# to_uint16
# --------------------------------------------------------------------------- #

def test_to_uint16_fixed_scale_maps_unit_interval():
    # Fixed [0,1]→×65535 (G6): 0→0, 1→65535, 0.5→~32768. The ABSOLUTE value sets
    # the code — NOT a per-channel min-max stretch.
    a = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = ops.to_uint16(a)
    assert out.dtype == np.uint16
    assert list(out) == [0, 32768, 65535]


def test_to_uint16_fixed_scale_preserves_dim_values():
    # THE correctness discriminator vs the old per-channel min-max: a uniformly
    # DIM channel (e.g. after background subtraction) stays dim, it is NOT
    # re-stretched.  The old min-max mapped a constant to all-zeros and stretched
    # any dim gradient to the full range — silently UNDOING a dimming op.  The
    # fixed scale preserves the recipe's actual output.
    dim = np.full((4, 4, 4), 0.2, dtype=np.float32)
    out = ops.to_uint16(dim)
    expected = int(round(0.2 * 65535.0))  # ~13107
    assert np.all(out == expected)
    assert int(out.max()) < 20000, "dim channel must NOT be stretched to full range"


def test_to_uint16_clips_out_of_range_and_nan():
    a = np.array([-0.5, 0.0, 1.0, 1.5, np.nan], dtype=np.float32)
    out = ops.to_uint16(a)
    assert list(out) == [0, 0, 65535, 65535, 0]  # clip to [0,1]; nan→0


def test_to_uint16_all_nan_is_zeros():
    out = ops.to_uint16(np.full((4, 4), np.nan, dtype=np.float32))
    assert out.dtype == np.uint16
    assert np.all(out == 0)


# --------------------------------------------------------------------------- #
# execute
# --------------------------------------------------------------------------- #

def test_execute_default_recipe_two_channels():
    rng = np.random.RandomState(12)
    ch0 = (0.3 + 0.1 * rng.rand(8, 16, 16)).astype(np.float32)
    ch1 = (0.5 + 0.1 * rng.rand(8, 16, 16)).astype(np.float32)
    recipe = default_recipe(2)  # per channel [gaussian] + shared isotropic @ boundary 1

    # spacing (2,1,1), target 1 µm → iso factors (2,1,1): Z doubles for BOTH channels.
    processed, new_spacing, stamp = ops.execute(recipe, [ch0, ch1], (2.0, 1.0, 1.0))

    assert len(processed) == 2
    # Geometry re-grids every channel identically.
    assert processed[0].shape == processed[1].shape == (16, 16, 16)
    for p in processed:
        assert p.dtype == np.float32
        assert p.min() >= 0.0 and p.max() <= 1.0
    assert new_spacing == pytest.approx((1.0, 1.0, 1.0))
    # Stamp carries the recipe json + the intended bake dtype / fixed-scale mode.
    assert stamp["dtype"] == "uint16"
    assert stamp["scale"] == "fixed_unit_x65535"
    assert "recipe" in stamp and isinstance(stamp["recipe"], dict)


def test_execute_threads_bounds_per_channel():
    # A per-channel window passed to execute is used for that channel's normalize
    # instead of the crop's own percentiles (the shared whole-volume window).
    ch = np.linspace(0.0, 10.0, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    recipe = Recipe(channels=[[]], geometry=None)  # normalize only
    p_full, _, _ = ops.execute(recipe, [ch], (1, 1, 1),
                               bounds_per_channel=[(0.0, 10.0)])
    p_half, _, _ = ops.execute(recipe, [ch], (1, 1, 1),
                               bounds_per_channel=[(5.0, 10.0)])
    frac_zero_full = float((p_full[0] == 0.0).mean())
    frac_zero_half = float((p_half[0] == 0.0).mean())
    # The (5,10) window clips the lower half to zero; (0,10) barely clips.
    assert frac_zero_half > frac_zero_full + 0.3


def test_execute_normalizes_first_even_with_no_ops():
    # A channel far outside [0,1]; only normalize-FIRST brings it back with
    # structure preserved (an end-clip alone would flatten it to all-ones).
    grad = np.linspace(100.0, 500.0, 8 * 16 * 16).reshape(8, 16, 16).astype(np.float32)
    recipe = Recipe(channels=[[]], geometry=None)  # nothing but the hoisted normalize
    processed, _, _ = ops.execute(recipe, [grad], (1.0, 1.0, 1.0))
    out = processed[0]
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert float(out.max() - out.min()) > 0.5  # spread preserved, not clipped flat


def test_execute_runs_ops_in_flattened_order(monkeypatch):
    calls = []

    def rec_gaussian(arr, **_):
        calls.append("gaussian")
        return arr

    def rec_background(arr, **_):
        calls.append("background")
        return arr

    def rec_geom(channels, spacing, config, **kwargs):  # **kwargs: accepts max_workers
        calls.append("isotropic")
        return list(channels), spacing

    monkeypatch.setattr(reg.get("gaussian"), "fn", rec_gaussian)
    monkeypatch.setattr(reg.get("background"), "fn", rec_background)
    monkeypatch.setattr(ops, "apply_geometry", rec_geom)

    # channel [gaussian, background] with geometry @ boundary 1
    #   → flatten [gaussian, isotropic, background]
    # A GENUINE resize (target 0.5 um vs a 1.0 um source): since Brick 3a #5 an
    # IDENTITY geometry step is skipped, so the default config would no longer
    # exercise the boundary this test is about.
    geom_cfg = dict(default_geometry_config())
    geom_cfg.update({"mode": "iso", "target_um": 0.5})
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"sigma": 1.0}),
                   IntensityStep("background", {"radius": 3})]],
        geometry=GeometryStep("isotropic", 1, geom_cfg),
    )
    ops.execute(recipe, [np.zeros((4, 4, 4), dtype=np.float32)], (1.0, 1.0, 1.0))
    assert calls == ["gaussian", "isotropic", "background"]


def test_execute_garbage_recipe_never_raises():
    ch = np.full((4, 4, 4), 0.5, dtype=np.float32)
    bad = Recipe(
        channels=[[
            IntensityStep("does_not_exist", {}),        # unknown key → dropped
            IntensityStep("median", {"radius": 2}),      # declared, not implemented → dropped
            IntensityStep("gaussian", {"sigma": 1.0}),   # the one survivor
        ]],
        geometry=GeometryStep("not_a_geom", 5, {"mode": "weird"}),  # bad geometry → dropped
    )
    processed, _, stamp = ops.execute(bad, [ch], (1.0, 1.0, 1.0))
    assert len(processed) == 1
    assert processed[0].dtype == np.float32
    assert processed[0].min() >= 0.0 and processed[0].max() <= 1.0

    # Total garbage (not even a Recipe) is coerced by normalize_recipe → no raise.
    processed2, _, _ = ops.execute("total garbage", [ch], (1.0, 1.0, 1.0))
    assert len(processed2) == 1


def test_execute_channel_count_clamped_to_data():
    # Recipe declares 3 channels but only 2 arrays given → normalize clamps to 2.
    rng = np.random.RandomState(13)
    recipe = default_recipe(3)
    data = [rng.rand(4, 8, 8).astype(np.float32) for _ in range(2)]
    processed, _, _ = ops.execute(recipe, data, (1.0, 1.0, 1.0))
    assert len(processed) == 2


def test_execute_parallel_channels_match_serial_reference():
    """Channel parallelism must not change results OR cross-map channels: each
    output channel must equal the exact same pipeline run serially on that same
    input channel (a mis-assignment would swap outputs and fail array-equal)."""
    rng = np.random.RandomState(21)
    # Three DISTINCT channels (different content + different intensity ranges).
    chans = [
        (0.3 * (i + 1) * rng.rand(6, 12, 12)).astype(np.float32) for i in range(3)
    ]
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"sigma": 1.0})] for _ in range(3)],
        geometry=None,
    )
    processed, _, _ = ops.execute(recipe, chans, (1.0, 1.0, 1.0))

    assert len(processed) == 3
    for i, ch in enumerate(chans):
        # execute() runs percentile_normalize → gaussian (→ clip) per channel.
        ref = np.clip(
            ops.gaussian(ops.percentile_normalize(ch), sigma=1.0), 0.0, 1.0
        ).astype(np.float32)
        np.testing.assert_allclose(
            processed[i], ref, rtol=0.0, atol=1e-6,
            err_msg=f"channel {i} parallel result != serial reference",
        )


def test_execute_serial_max_workers_matches_parallel():
    """The memory-bounded whole-volume path (max_workers=1 → serial, one
    channel's working set at a time) must produce BYTE-IDENTICAL results to the
    default parallel width — the seam Slice 1 uses to honor without
    changing the output."""
    rng = np.random.RandomState(22)
    chans = [rng.rand(6, 12, 12).astype(np.float32) for _ in range(3)]
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"sigma": 1.2}),
                   IntensityStep("background", {"radius": 3})] for _ in range(3)],
        geometry=GeometryStep("isotropic", 1, default_geometry_config()),
    )
    # spacing (2,1,1), iso target 1 → Z doubles; flatten = [gaussian, iso, background].
    par, sp_par, _ = ops.execute(recipe, chans, (2.0, 1.0, 1.0))
    ser, sp_ser, _ = ops.execute(recipe, chans, (2.0, 1.0, 1.0), max_workers=1)

    assert sp_par == sp_ser
    assert len(par) == len(ser) == 3
    for i, (p, s) in enumerate(zip(par, ser)):
        np.testing.assert_array_equal(p, s, err_msg=f"channel {i} serial != parallel")


# --------------------------------------------------------------------------- #
# Brick 3a #3 — to_uint16 single-pass rewrite is BYTE-IDENTICAL to the old form
# --------------------------------------------------------------------------- #

def _to_uint16_reference(arr_1ch):
    """A VERBATIM copy of the pre-Brick-3a `ops.to_uint16` body — the byte-identity
    oracle.  Do not "simplify" it; its whole job is to be the old expression."""
    a = np.asarray(arr_1ch, dtype=np.float32)
    a = np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0)
    return np.round(a * 65535.0).astype(np.uint16)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_to_uint16_matches_the_old_expression_on_random_float32(seed):
    a = np.random.RandomState(seed).rand(7, 33, 41).astype(np.float32)
    got = ops.to_uint16(a)
    assert got.dtype == np.uint16
    assert np.array_equal(got, _to_uint16_reference(a))


def test_to_uint16_matches_on_exact_endpoints_and_nonfinite():
    """The edge cases the rewrite could plausibly move: exact 0.0 / 1.0, the
    half-way codes (round-half-to-even), out-of-range, +-inf and NaN."""
    a = np.array(
        [0.0, -0.0, 1.0, 1.0 - 2 ** -24, 2 ** -24, 0.5, 0.25, 0.75,
         0.5 / 65535.0, 1.5 / 65535.0, 2.5 / 65535.0,     # exact .5 codes
         -1e-8, -1.0, 1.0 + 1e-6, 2.0, 1e30,
         np.inf, -np.inf, np.nan],
        dtype=np.float32,
    )
    got = ops.to_uint16(a)
    ref = _to_uint16_reference(a)
    assert np.array_equal(got, ref), list(zip(a.tolist(), got.tolist(), ref.tolist()))
    assert got[0] == 0 and got[2] == 65535
    assert got[-1] == 0            # NaN -> 0 preserved
    assert got[-3] == 65535 and got[-2] == 0   # +inf -> 65535, -inf -> 0


def test_to_uint16_does_not_mutate_its_input():
    a = np.linspace(-0.5, 1.5, 512, dtype=np.float32).reshape(8, 8, 8)
    before = a.copy()
    ops.to_uint16(a)
    assert np.array_equal(a, before)


def test_to_uint16_all_nan_and_empty():
    assert np.array_equal(ops.to_uint16(np.full((4, 4), np.nan, np.float32)),
                          np.zeros((4, 4), np.uint16))
    assert ops.to_uint16(np.zeros((0,), np.float32)).shape == (0,)


# --------------------------------------------------------------------------- #
# Brick 3c #1 - ops.gaussian's trailing clip is REQUIRED (the 3a #4 drop was a
# bug: `apply_geometry` does NOT clip, and `method="cubic"` rings outside [0, 1])
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 3.0, 5.0])
@pytest.mark.parametrize("kind", ["uniform", "binary01", "mixed01"])
def test_gaussian_on_in_range_input_equals_the_clipped_form(sigma, kind):
    """On [0, 1] input the filter is a convex combination, so clipping changes
    nothing - this is the case Brick 3a measured, and it is still true.  It is
    NOT the whole domain: see the cubic-geometry tests below."""
    from scipy.ndimage import gaussian_filter

    r = np.random.RandomState(abs(hash((sigma, kind))) % 2 ** 31)
    a = r.rand(12, 48, 48).astype(np.float32)
    if kind == "binary01":
        a = (a > 0.5).astype(np.float32)
    elif kind == "mixed01":
        a[a < 0.4] = 0.0
        a[a > 0.6] = 1.0
    ref = np.clip(gaussian_filter(a, sigma=sigma), 0.0, 1.0).astype(np.float32)
    got = ops.gaussian(a, sigma=sigma)
    assert got.dtype == np.float32
    assert np.array_equal(got, ref)
    assert got.min() >= 0.0 and got.max() <= 1.0


def _cubic_resampled_channel(target_um=0.5, seed=0):
    """A channel that has been through a CUBIC geometry resample - i.e. exactly
    what `execute` stage 3 hands to a post-geometry gaussian.  Cubic
    interpolation OVERSHOOTS at sharp edges, and `apply_geometry` deliberately
    does not clip, so this array genuinely leaves [0, 1]."""
    vol = np.zeros((16, 48, 48), np.float32)
    vol[:, 8:20, 8:20] = 1.0
    vol[:, 30:44, 30:44] = 1.0
    cfg = {"mode": "iso", "target_um": target_um,
           "factors": (1.0, 1.0, 1.0), "method": "cubic"}
    out, _sp = ops.apply_geometry(
        [ops.percentile_normalize(vol)], (1.0, 1.0, 1.0), cfg)
    return out[0]


def test_apply_geometry_cubic_really_leaves_the_unit_range():
    """The premise of the two tests below - if this ever stops being true they
    become vacuous, so assert it explicitly."""
    g = _cubic_resampled_channel()
    assert g.min() < 0.0 and g.max() > 1.0


@pytest.mark.parametrize("sigma", [0.5, 0.8, 1.0])
def test_gaussian_clips_a_cubic_resampled_channel(sigma):
    """REGRESSION (Brick 3c #1).  `ops.gaussian` must return [0, 1] even when its
    INPUT does not - a bare `gaussian_filter` here differs from the contract by a
    visible margin, and the difference survives into the bake."""
    from scipy.ndimage import gaussian_filter

    g = _cubic_resampled_channel()
    unclipped = gaussian_filter(g, sigma=sigma).astype(np.float32)
    assert unclipped.min() < 0.0 or unclipped.max() > 1.0, (
        "this sigma no longer exercises the out-of-range case"
    )
    got = ops.gaussian(g, sigma=sigma)
    assert np.array_equal(got, np.clip(unclipped, 0.0, 1.0).astype(np.float32))
    assert got.min() >= 0.0 and got.max() <= 1.0


def test_cubic_geometry_then_gaussian_then_background_needs_the_clip():
    """The end-to-end shape of the bug the audit found: a cubic geometry step,
    then gaussian, then background.  `background` is NOT convex - it reads the
    gaussian's out-of-range values through a morphological opening - so the
    missing clip changes THOUSANDS of DN in the uint16 bake, not last-ULP noise.
    """
    from scipy.ndimage import gaussian_filter

    g = _cubic_resampled_channel()
    with_clip = ops.background(ops.gaussian(g, sigma=0.5), radius=3)
    without_clip = ops.background(
        gaussian_filter(g, sigma=0.5).astype(np.float32), radius=3)

    baked_ok = ops.to_uint16(with_clip)
    baked_bad = ops.to_uint16(without_clip)
    delta = np.abs(baked_ok.astype(np.int64) - baked_bad.astype(np.int64))
    assert delta.max() > 1000, (
        "the negative control lost its teeth - the two paths must diverge "
        f"visibly (got max delta {int(delta.max())})"
    )
    # and the SHIPPED path is the clipped one
    assert np.array_equal(baked_ok, ops.to_uint16(
        ops.background(ops.gaussian(g, sigma=0.5), radius=3)))


def test_execute_cubic_geometry_recipe_stays_in_range_at_every_stage():
    """Through the real `execute` seam, not hand-composed ops."""
    from packages.epicycle_preprocessing.recipe import GeometryStep

    vol = np.zeros((16, 48, 48), np.float32)
    vol[:, 8:20, 8:20] = 1.0
    vol[:, 30:44, 30:44] = 1.0
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"sigma": 0.5}),
                   IntensityStep("background", {"radius": 3})]],
        geometry=GeometryStep(
            "isotropic", 0,
            {"mode": "iso", "target_um": 0.5, "factors": (1.0, 1.0, 1.0),
             "method": "cubic"}),
    )
    out, _sp, _st = ops.execute(recipe, [vol], (1.0, 1.0, 1.0))
    assert out[0].min() >= 0.0 and out[0].max() <= 1.0


def test_gaussian_sigma_zero_is_still_a_noop():
    a = np.random.RandomState(5).rand(4, 8, 8).astype(np.float32)
    assert np.array_equal(ops.gaussian(a, sigma=0.0), a)
    assert np.array_equal(ops.gaussian(a, sigma=-1.0), a)


# --------------------------------------------------------------------------- #
# Brick 3a #5 — execute skips an IDENTITY geometry step, like streaming does
# --------------------------------------------------------------------------- #

def _iso_recipe(target_um, n_channels=1):
    from packages.epicycle_preprocessing.recipe import GeometryStep

    return Recipe(
        channels=[[] for _ in range(n_channels)],
        geometry=GeometryStep(
            key="isotropic", boundary_index=0,
            config={"mode": "iso", "target_um": target_um,
                    "factors": (1.0, 1.0, 1.0), "method": "linear"},
        ),
    )


def test_execute_skips_identity_geometry(monkeypatch):
    """Spacing already AT the isotropic target -> every factor is 1 -> the resample
    must not run at all (it was a full per-channel `zoom` for nothing)."""
    calls = []
    real = ops.apply_geometry
    monkeypatch.setattr(
        ops, "apply_geometry",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    chans = [np.random.RandomState(1).rand(6, 20, 20).astype(np.float32)]
    out, spacing, _stamp = ops.execute(
        _iso_recipe(1.0), chans, (1.0, 1.0, 1.0))
    assert calls == [], "identity geometry must be skipped, not resampled"
    assert spacing == (1.0, 1.0, 1.0)
    assert out[0].shape == chans[0].shape


def test_execute_still_runs_a_real_geometry_step(monkeypatch):
    """Negative control — a genuine resize must STILL go through apply_geometry."""
    calls = []
    real = ops.apply_geometry
    monkeypatch.setattr(
        ops, "apply_geometry",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    chans = [np.random.RandomState(2).rand(6, 20, 20).astype(np.float32)]
    out, spacing, _stamp = ops.execute(
        _iso_recipe(0.5), chans, (1.0, 1.0, 1.0))
    assert calls == [1], "a real resize must not be skipped"
    assert out[0].shape != chans[0].shape


def test_execute_identity_skip_uses_the_streaming_predicate(monkeypatch):
    """ONE definition of "identity" across the two regimes (no duplicate).

    Asserted by OBJECT IDENTITY, not by reading `execute`s source: a
    `inspect.getsource(...) in` check passes on a comment and rots the moment the
    import moves (that whole class of test is what rotted `tests/ui_shell/`).
    Here the streaming predicate is replaced with a recorder - if `execute` had
    its own copy, the recorder would never be called."""
    from packages.epicycle_preprocessing import streaming as _st

    calls = []
    real = _st._geometry_is_identity

    def spy(step, spacing):
        calls.append((step, tuple(spacing or ())))
        return real(step, spacing)

    monkeypatch.setattr(_st, "_geometry_is_identity", spy)
    chans = [np.random.RandomState(9).rand(5, 16, 16).astype(np.float32)]
    ops.execute(_iso_recipe(1.0), chans, (1.0, 1.0, 1.0))
    assert len(calls) == 1, "execute must call THE streaming predicate, not a copy"


def test_execute_identity_skip_result_matches_the_resampled_noop():
    """Skipping must be a pure perf change: identical bytes to letting the
    identity `zoom` run."""
    chans = [np.random.RandomState(3).rand(5, 16, 16).astype(np.float32)]
    rec = _iso_recipe(1.0)
    skipped, sp_a, _ = ops.execute(rec, chans, (1.0, 1.0, 1.0))
    resampled, sp_b = ops.apply_geometry(
        [ops.percentile_normalize(chans[0])], (1.0, 1.0, 1.0), rec.geometry.config)
    resampled = [np.clip(np.asarray(a, np.float32), 0.0, 1.0) for a in resampled]
    assert sp_a == sp_b
    assert np.array_equal(skipped[0], resampled[0])


# --------------------------------------------------------------------------- #
# Brick 3a #6 — channel-parallel CLAHE, RAM-gated (the SAFE half; NOT per-slab)
# --------------------------------------------------------------------------- #

class _VM:
    def __init__(self, avail):
        self.available = avail


def _clahe_recipe(n_channels=2):
    return Recipe(
        channels=[[IntensityStep("clahe", {"clip": 0.01, "kernel": 8})]
                  for _ in range(n_channels)],
        geometry=None,
    )


def _concurrency_recorder():
    """A fake op fn that records the MAXIMUM number of concurrent invocations."""
    import threading
    import time

    state = {"live": 0, "max": 0, "calls": 0}
    lock = threading.Lock()

    def fn(arr, **_kw):
        with lock:
            state["live"] += 1
            state["calls"] += 1
            state["max"] = max(state["max"], state["live"])
        time.sleep(0.05)          # hold the slot so an overlap is observable
        with lock:
            state["live"] -= 1
        return arr

    return state, fn


def test_adaptive_channel_workers_widens_when_ram_fits(monkeypatch):
    import psutil
    chans = [np.zeros((4, 64, 64), np.float32)] * 2      # 64 KB each
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM(64 * 1024 ** 3))
    assert ops.adaptive_channel_workers(chans) == 2


def test_adaptive_channel_workers_stays_serial_when_ram_is_tight(monkeypatch):
    import psutil
    # 600 MB/channel x the measured ~20x peak = ~12 GB extra; 8 GB free -> serial
    chans = [np.zeros((143, 1024, 1024), np.float32)] * 2
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM(8 * 1024 ** 3))
    assert ops.adaptive_channel_workers(chans) == 1


def test_adaptive_channel_workers_is_capped_and_never_guesses(monkeypatch):
    import builtins
    import psutil
    chans = [np.zeros((2, 8, 8), np.float32)] * 5
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM(64 * 1024 ** 3))
    assert ops.adaptive_channel_workers(chans) == 2        # capped at 2, not 5
    assert ops.adaptive_channel_workers(chans[:1]) == 1    # single channel

    real_import = builtins.__import__

    def _no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    assert ops.adaptive_channel_workers(chans) == 1        # no probe -> serial


def test_execute_runs_clahe_channels_concurrently_when_ram_fits(monkeypatch):
    import psutil
    state, fn = _concurrency_recorder()
    monkeypatch.setattr(reg.get("clahe"), "fn", fn)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM(64 * 1024 ** 3))

    chans = [np.random.RandomState(i).rand(4, 32, 32).astype(np.float32)
             for i in range(2)]
    ops.execute(_clahe_recipe(2), chans, (1.0, 1.0, 1.0), max_workers=1)
    assert state["calls"] == 2
    assert state["max"] == 2, "the two channels' CLAHE must overlap"


def test_execute_keeps_clahe_channels_serial_when_ram_is_tight(monkeypatch):
    import psutil
    state, fn = _concurrency_recorder()
    monkeypatch.setattr(reg.get("clahe"), "fn", fn)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM(1024 ** 2))

    chans = [np.random.RandomState(i).rand(4, 32, 32).astype(np.float32)
             for i in range(2)]
    ops.execute(_clahe_recipe(2), chans, (1.0, 1.0, 1.0), max_workers=1)
    assert state["calls"] == 2
    assert state["max"] == 1, "a tight RAM budget must keep channels serial"


def test_execute_keeps_local_only_recipes_serial(monkeypatch):
    """No adaptive op -> the channel pool must stay at the caller's width, so the
    BLOCK pool keeps owning the cores (no channel x block oversubscription)."""
    import psutil
    state, fn = _concurrency_recorder()
    monkeypatch.setattr(reg.get("gaussian"), "fn", fn)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM(64 * 1024 ** 3))

    chans = [np.random.RandomState(i).rand(4, 32, 32).astype(np.float32)
             for i in range(2)]
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"sigma": 1.0})] for _ in range(2)],
        geometry=None,
    )
    ops.execute(recipe, chans, (1.0, 1.0, 1.0), max_workers=1)
    assert state["max"] == 1


def test_channel_parallel_clahe_is_byte_identical(monkeypatch):
    """The widening must be a PURE scheduling change.

    Both legs are FORCED (the 3a version only forced the serial one and let the
    parallel leg fall back to the ambient RAM budget - on a small box that made
    it serial too, i.e. vacuous)."""
    import packages.epicycle_preprocessing.ops as _o

    chans = [np.random.RandomState(i).rand(5, 40, 40).astype(np.float32)
             for i in range(2)]
    rec = _clahe_recipe(2)

    monkeypatch.setattr(_o, "adaptive_channel_workers", lambda *a, **k: 2)
    par, _sp, _st = ops.execute(rec, chans, (1.0, 1.0, 1.0), max_workers=1)

    monkeypatch.setattr(_o, "adaptive_channel_workers", lambda *a, **k: 1)
    ser, _sp2, _st2 = ops.execute(rec, chans, (1.0, 1.0, 1.0), max_workers=1)

    for a, b in zip(par, ser):
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Brick 3c #2 - the widening must not double the BLOCK pool, and must not catch
# the preview path (which passes no max_workers at all)
# --------------------------------------------------------------------------- #

def _record_block_workers(monkeypatch):
    """Record the `block_workers` every `_run_intensity_ops` tiled op receives."""
    seen = []
    real = ops.block_parallel_apply

    def spy(op_fn, arr, halo, **kw):
        seen.append(kw.get("max_workers"))
        return real(op_fn, arr, halo, **kw)

    monkeypatch.setattr(ops, "block_parallel_apply", spy)
    return seen


def test_widening_divides_the_block_pool(monkeypatch):
    """`stage_workers` applies to the WHOLE stage, so two channels are inside
    `_run_intensity_ops` at once.  Each must get `block_workers // stage_workers`
    or the peak tile scratch is 2x what `_derive_block_workers` budgeted."""
    import packages.epicycle_preprocessing.ops as _o

    seen = _record_block_workers(monkeypatch)
    monkeypatch.setattr(_o, "adaptive_channel_workers", lambda *a, **k: 2)

    chans = [np.random.RandomState(i).rand(6, 40, 40).astype(np.float32)
             for i in range(2)]
    recipe = Recipe(
        channels=[[IntensityStep("clahe", {"clip": 0.01, "kernel": 8}),
                   IntensityStep("gaussian", {"sigma": 1.0})]
                  for _ in range(2)],
        geometry=None,
    )
    ops.execute(recipe, chans, (1.0, 1.0, 1.0),
                max_workers=1, parallel=True, block_workers=20)
    assert seen and set(seen) == {10}, seen


def test_no_widening_leaves_the_block_pool_whole(monkeypatch):
    import packages.epicycle_preprocessing.ops as _o

    seen = _record_block_workers(monkeypatch)
    monkeypatch.setattr(_o, "adaptive_channel_workers", lambda *a, **k: 1)

    chans = [np.random.RandomState(i).rand(6, 40, 40).astype(np.float32)
             for i in range(2)]
    recipe = Recipe(
        channels=[[IntensityStep("clahe", {"clip": 0.01, "kernel": 8}),
                   IntensityStep("gaussian", {"sigma": 1.0})]
                  for _ in range(2)],
        geometry=None,
    )
    ops.execute(recipe, chans, (1.0, 1.0, 1.0),
                max_workers=1, parallel=True, block_workers=20)
    assert seen and set(seen) == {20}, seen


def test_divided_block_pool_never_goes_below_one(monkeypatch):
    import packages.epicycle_preprocessing.ops as _o

    seen = _record_block_workers(monkeypatch)
    monkeypatch.setattr(_o, "adaptive_channel_workers", lambda *a, **k: 2)
    chans = [np.random.RandomState(i).rand(6, 40, 40).astype(np.float32)
             for i in range(2)]
    recipe = Recipe(
        channels=[[IntensityStep("clahe", {"clip": 0.01, "kernel": 8}),
                   IntensityStep("gaussian", {"sigma": 1.0})]
                  for _ in range(2)],
        geometry=None,
    )
    ops.execute(recipe, chans, (1.0, 1.0, 1.0),
                max_workers=1, parallel=True, block_workers=1)
    assert seen and set(seen) == {1}, seen


def test_preview_path_is_never_widened(monkeypatch):
    """`_compute_preview` calls `execute` with NO `max_workers` (None = "no
    opinion", already a wide channel pool).  The old `(max_workers or 1) <= 1`
    gate caught it; the explicit `== 1` gate must not."""
    import packages.epicycle_preprocessing.ops as _o

    asked = []
    monkeypatch.setattr(
        _o, "adaptive_channel_workers",
        lambda *a, **k: (asked.append(1), 2)[1])

    chans = [np.random.RandomState(i).rand(4, 24, 24).astype(np.float32)
             for i in range(2)]
    ops.execute(_clahe_recipe(2), chans, (1.0, 1.0, 1.0))   # no max_workers
    assert asked == [], "the preview path must not consult the widening budget"


def test_progress_indices_are_unique_under_concurrent_channels(monkeypatch):
    """`_ProgressCtx.op` captures its index inside the lock - two concurrent
    channels must never report the same `index`."""
    import packages.epicycle_preprocessing.ops as _o

    monkeypatch.setattr(_o, "adaptive_channel_workers", lambda *a, **k: 2)
    events = []
    chans = [np.random.RandomState(i).rand(4, 24, 24).astype(np.float32)
             for i in range(2)]
    ops.execute(_clahe_recipe(2), chans, (1.0, 1.0, 1.0), max_workers=1,
                on_progress=events.append)
    idx = [e["index"] for e in events if e.get("kind") == "op"]
    assert idx and len(idx) == len(set(idx)), idx
    assert sorted(idx) == list(range(1, len(idx) + 1)), idx


# --------------------------------------------------------------------------- #
# Brick 3c #5 - NaN really does reach to_uint16 (the docstring used to deny it)
# --------------------------------------------------------------------------- #

def test_clip_propagates_nan_so_execute_cannot_remove_it():
    """The premise: `execute`'s trailing `clip(..., 0, 1)` does NOT sanitize NaN,
    because `np.clip` propagates it.  `to_uint16`'s NaN branch is therefore load-
    bearing, not belt-and-suspenders."""
    assert np.isnan(np.clip(np.float32("nan"), 0.0, 1.0))
    assert np.isnan(ops.percentile_normalize(
        np.array([0.0, 0.5, np.nan, 1.0], np.float32))).any()


def test_nan_in_the_source_survives_execute_and_is_zeroed_by_the_bake():
    src = np.random.RandomState(4).rand(4, 16, 16).astype(np.float32)
    src[0, 0, 0] = np.nan
    out, _sp, _st = ops.execute(
        Recipe(channels=[[IntensityStep("gaussian", {"sigma": 1.0})]],
               geometry=None),
        [src], (1.0, 1.0, 1.0))
    assert np.isnan(out[0]).any(), "execute cannot remove NaN - clip propagates it"
    baked = ops.to_uint16(out[0])
    assert baked.dtype == np.uint16
    assert not np.isnan(baked.astype(np.float64)).any()
    assert baked[np.isnan(out[0])].tolist() == [0] * int(np.isnan(out[0]).sum())

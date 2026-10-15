"""Slice-0 smoke + timing guard for the Step-1 preprocessing ops.

Purpose (user ask, 2026-08-11): confirm EVERY implemented op actually RUNS
(doesn't throw), honors the working-space contract (float32 in [0, 1], shape
preserved for intensity ops), and stays TIMELY.  The background op's solid-ball
footprint had regressed to ~4 min at the default radius before the
``decomposition="sequence"`` fix; the generous timing bounds here fail loudly if
that (or an equally slow footprint) ever comes back, while staying safe on
slower CI machines.

Pure — no napari / Qt.  Lives in ``tests/data_preparation`` (NOT a CI batch), so
the wall-clock asserts never flake CI; they are a local regression net.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from packages.epicycle_preprocessing import ops
from packages.epicycle_preprocessing import registry as reg
from packages.epicycle_preprocessing.recipe import (
    GeometryStep,
    IntensityStep,
    Recipe,
    default_geometry_config,
)


def _normalized_vol(shape=(32, 128, 128), seed=0):
    """A structured, already-normalized [0, 1] float32 volume (so ops do real
    work: a smooth central bump = background for top-hat, plus texture)."""
    rng = np.random.RandomState(seed)
    _, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    bump = 0.4 * np.exp(
        -((yy - shape[1] / 2) ** 2 + (xx - shape[2] / 2) ** 2) / (2 * 30.0 ** 2)
    )
    a = 0.3 * rng.rand(*shape) + bump
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def _intensity_keys():
    return [k for k in reg.implemented_keys() if reg.get(k).kind == "intensity"]


def test_every_implemented_intensity_op_runs_and_honors_contract():
    """Each implemented intensity op runs on a 3D crop without throwing and
    returns float32 in [0, 1] with the shape preserved."""
    vol = _normalized_vol()
    keys = _intensity_keys()
    assert keys, "expected at least the v1 intensity ops (gaussian/clahe/background/bleach)"
    for key in keys:
        op = reg.get(key)
        params = {p.name: p.default for p in op.params}
        out = op.fn(vol, **params)
        assert out is not None, f"{key} returned None"
        out = np.asarray(out)
        assert out.dtype == np.float32, f"{key} dtype was {out.dtype}"
        assert out.shape == vol.shape, f"{key} changed shape {vol.shape} -> {out.shape}"
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, f"{key} left [0, 1]"


def test_geometry_op_runs_via_apply_geometry():
    """The shared geometry op runs (routed via apply_geometry, not a per-channel
    fn) and preserves the float32 contract."""
    vol = _normalized_vol((12, 32, 32))
    out, spacing = ops.apply_geometry([vol], (2.0, 1.0, 1.0), default_geometry_config())
    assert len(out) == 1
    assert out[0].dtype == np.float32
    assert len(spacing) == 3


@pytest.mark.parametrize("radius", [3, 8, 15])
def test_background_is_timely_at_all_radii(radius):
    """Regression guard for the solid-ball perf cliff: background at the default
    radius (15) took ~22 s on 0.5 Mvox with a solid ball; the decomposition fix
    brings it to ~0.2 s.  A generous 5 s bound fails loudly if the slow footprint
    ever returns, while staying safe on slower machines."""
    vol = _normalized_vol((32, 128, 128))  # 0.52 Mvox
    t0 = time.perf_counter()
    out = ops.background(vol, radius=radius)
    dt = time.perf_counter() - t0
    assert out.shape == vol.shape
    assert out.dtype == np.float32
    assert dt < 5.0, f"background(radius={radius}) took {dt:.2f}s — perf regression?"


def test_background_2d_disk_path_runs():
    """The 2D disk-decomposition footprint path (the 3D tests only hit ``ball``)."""
    a = _normalized_vol((1, 64, 64))[0]  # a genuine 2D (Y, X) array
    out = ops.background(a, radius=10)
    assert out.ndim == 2 and out.shape == a.shape and out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_execute_all_ops_recipe_runs_clean_and_timely():
    """A recipe chaining ALL implemented intensity ops + the shared geometry step
    over a 2-channel volume runs without throwing and quickly (channel-parallel).
    Geometry is a no-op here (spacing == target), so the array size is stable."""
    chans = [_normalized_vol(seed=1), _normalized_vol(seed=2)]
    intensity = [
        IntensityStep(k, {p.name: p.default for p in reg.get(k).params})
        for k in _intensity_keys()
    ]
    recipe = Recipe(
        channels=[list(intensity), list(intensity)],
        geometry=GeometryStep(
            "isotropic", boundary_index=len(intensity), config=default_geometry_config()
        ),
    )
    t0 = time.perf_counter()
    processed, spacing, stamp = ops.execute(recipe, chans, (1.0, 1.0, 1.0))
    dt = time.perf_counter() - t0

    assert len(processed) == 2
    for p in processed:
        assert p.dtype == np.float32
        assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    assert "recipe" in stamp
    assert dt < 12.0, f"execute(all ops, 2ch) took {dt:.2f}s — perf regression?"

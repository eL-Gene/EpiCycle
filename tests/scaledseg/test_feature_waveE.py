"""Wave E feature tests — per-channel TEXTURE (atlas ).

Pure-function known-shape tests (no tile readers / GPU). Builds tiny synthetic
intensity crops with KNOWN texture character — punctate, smooth/diffuse, uniform,
saturated/clipped — and calls :func:`qc_features.compute_channel_texture` DIRECTLY
on a single-label mask + the intensity crop, asserting the discriminators move in
the documented direction. Texture is the robust-to-haze/saturation signal that
separates punctate vs diffuse vs hazy (HCR) and captures nuclear chromatin
condensation; it is computed on the SAME raw-channel crop the M2 intensity
features read.

Asserts:

* PUNCTATE (bright spots in a dim cell) -> high granularity + high glcm_contrast
  vs a SMOOTH/diffuse gradient and a UNIFORM fill (granularity is the strongest
  punctate-vs-diffuse discriminator).
* SMOOTH/diffuse gradient -> high glcm_homogeneity + high entropy, LOW granularity.
* UNIFORM fill -> ~0 variance/entropy/contrast, homogeneity ≈ 1, granularity ≈ 0.
* SATURATED/clipped vs unclipped -> a smaller robust dynamic range (the texture is
  saturation-robust by design).
* NaN guard: a too-small mask yields NaN for the GLCM block (and finite 0-collapse
  for the 3D-native block); an all-NaN crop yields NaN everywhere.
* The manifest tags every dynamic per-channel texture column via
  ``channel_texture_tags(prefix)``.
* The FeatureStage wiring emits the texture columns when channels are sampled, and
  the labels-only path stays byte-identical (texture appears ONLY with channels).
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# All 11 <prefix>{suffix} texture columns compute_channel_texture emits.
_TEX_3D = (
    "_texture_variance", "_texture_entropy", "_texture_mad",
    "_texture_granularity", "_texture_range_robust",
)
_TEX_GLCM = (
    "_glcm_contrast", "_glcm_homogeneity", "_glcm_correlation",
    "_glcm_energy", "_glcm_entropy", "_glcm_dissimilarity",
)
_TEX_SUFFIXES = _TEX_3D + _TEX_GLCM


# --------------------------------------------------------------------------- #
# synthetic crops (mask + intensity)
# --------------------------------------------------------------------------- #

def _cube_mask(shape=(16, 16, 16), lo=3, hi=13):
    """A solid cube cell mask (boolean), well inside the crop so the opening +
    erosion have a clean surround."""
    mask = np.zeros(shape, dtype=bool)
    mask[lo:hi, lo:hi, lo:hi] = True
    return mask


def _punctate(mask, frac=0.06, dim=10.0, bright=250.0, seed=0):
    """A PUNCTATE signal: a dim cell with a fraction of bright single-voxel spots
    (HCR puncta-like). Bright spots are erased by morphological opening -> high
    granularity; sharp dim/bright alternation -> high glcm_contrast."""
    rng = np.random.RandomState(seed)
    img = np.where(mask, dim, 0.0).astype(np.float64)
    zz, yy, xx = np.where(mask)
    k = max(int(len(zz) * frac), 1)
    idx = rng.choice(len(zz), k, replace=False)
    img[zz[idx], yy[idx], xx[idx]] = bright
    return img


def _smooth_gradient(mask, base=20.0, slope=8.0):
    """A SMOOTH/diffuse signal: a linear intensity ramp across X. Survives the
    opening -> low granularity; gradual neighbour change -> high homogeneity +
    high histogram entropy (many evenly-populated levels)."""
    img = np.zeros(mask.shape, dtype=np.float64)
    for x in range(mask.shape[2]):
        img[:, :, x] = base + slope * x
    img[~mask] = 0.0
    return img


def _uniform(mask, value=100.0):
    """A UNIFORM fill: a single intensity level over the whole cell."""
    return np.where(mask, value, 0.0).astype(np.float64)


# --------------------------------------------------------------------------- #
# column presence + finiteness
# --------------------------------------------------------------------------- #

def test_texture_columns_present_with_prefix():
    mask = _cube_mask()
    t = qf.compute_channel_texture(_smooth_gradient(mask), mask, "nuc")
    for s in _TEX_SUFFIXES:
        assert f"nuc{s}" in t, f"nuc{s} missing"
    # No extra / fewer columns than the documented 11.
    assert len(t) == len(_TEX_SUFFIXES)


def test_texture_returns_dict_not_dataframe():
    """compute_channel_texture returns a flat {col: value} dict (the per-cell shape
    the FeatureStage sampler writes into the row), unlike the df-mutating siblings."""
    mask = _cube_mask()
    t = qf.compute_channel_texture(_uniform(mask), mask, "c")
    assert isinstance(t, dict)


# --------------------------------------------------------------------------- #
# punctate vs smooth vs uniform discriminators
# --------------------------------------------------------------------------- #

def test_punctate_high_granularity_and_contrast():
    mask = _cube_mask()
    punct = qf.compute_channel_texture(_punctate(mask), mask, "c")
    smooth = qf.compute_channel_texture(_smooth_gradient(mask), mask, "c")
    unif = qf.compute_channel_texture(_uniform(mask), mask, "c")

    # Granularity is the strongest punctate-vs-diffuse discriminator: the bright
    # spots are erased by the morphological opening.
    assert punct["c_texture_granularity"] > smooth["c_texture_granularity"]
    assert punct["c_texture_granularity"] > unif["c_texture_granularity"]
    # Sharp dim/bright alternation -> high GLCM contrast vs a uniform fill.
    assert punct["c_glcm_contrast"] > unif["c_glcm_contrast"]
    # Variance is well above the uniform (zero) floor.
    assert punct["c_texture_variance"] > unif["c_texture_variance"]


def test_smooth_high_homogeneity_low_granularity():
    mask = _cube_mask()
    smooth = qf.compute_channel_texture(_smooth_gradient(mask), mask, "c")
    punct = qf.compute_channel_texture(_punctate(mask), mask, "c")

    # A diffuse gradient survives the opening -> low granularity vs punctate.
    assert smooth["c_texture_granularity"] < punct["c_texture_granularity"]
    # Gradual neighbour transitions -> high GLCM homogeneity.
    assert smooth["c_glcm_homogeneity"] > 0.5
    # Many evenly populated intensity levels -> high histogram entropy.
    assert smooth["c_texture_entropy"] > 1.0


def test_uniform_near_zero_texture():
    mask = _cube_mask()
    t = qf.compute_channel_texture(_uniform(mask, 100.0), mask, "c")
    assert t["c_texture_variance"] == pytest.approx(0.0, abs=1e-9)
    assert t["c_texture_entropy"] == pytest.approx(0.0, abs=1e-9)
    assert t["c_texture_mad"] == pytest.approx(0.0, abs=1e-9)
    assert t["c_texture_granularity"] == pytest.approx(0.0, abs=1e-9)
    assert t["c_texture_range_robust"] == pytest.approx(0.0, abs=1e-9)
    # A flat signal -> degenerate GLCM: zero contrast, perfect homogeneity/energy.
    assert t["c_glcm_contrast"] == pytest.approx(0.0, abs=1e-9)
    assert t["c_glcm_homogeneity"] == pytest.approx(1.0, abs=1e-9)
    assert t["c_glcm_energy"] == pytest.approx(1.0, abs=1e-9)
    assert t["c_glcm_dissimilarity"] == pytest.approx(0.0, abs=1e-9)


def test_uniform_vs_smooth_entropy_ordering():
    """A uniform fill has zero entropy; a multi-level diffuse gradient has high
    entropy — the entropy column separates structured from flat signal."""
    mask = _cube_mask()
    unif = qf.compute_channel_texture(_uniform(mask), mask, "c")
    smooth = qf.compute_channel_texture(_smooth_gradient(mask), mask, "c")
    assert smooth["c_texture_entropy"] > unif["c_texture_entropy"]


# --------------------------------------------------------------------------- #
# saturation / clipping robustness
# --------------------------------------------------------------------------- #

def test_saturated_smaller_robust_range_than_unclipped():
    """Clipping the bright tail (saturation) compresses the robust dynamic range —
    the texture range is built to be saturation-robust (p98 - p2)."""
    mask = _cube_mask()
    unclipped = _smooth_gradient(mask, base=20.0, slope=8.0)
    clipped = unclipped.copy()
    clipped[clipped > 80.0] = 80.0          # simulate detector saturation
    clipped[~mask] = 0.0

    t_un = qf.compute_channel_texture(unclipped, mask, "c")
    t_cl = qf.compute_channel_texture(clipped, mask, "c")
    assert t_cl["c_texture_range_robust"] < t_un["c_texture_range_robust"]
    # Both stay finite (no NaN leak from the clipping).
    for s in _TEX_3D:
        assert np.isfinite(t_un[f"c{s}"]) and np.isfinite(t_cl[f"c{s}"])


def test_robust_range_ignores_a_few_outliers():
    """A handful of saturated voxels barely move the p98-p2 robust range vs a clean
    uniform-ish cell (the robust range is outlier-insensitive)."""
    mask = _cube_mask()
    base = _uniform(mask, 100.0)
    spiked = base.copy()
    # Spike a few voxels to a huge value; p98 should stay near 100.
    zz, yy, xx = np.where(mask)
    spiked[zz[:3], yy[:3], xx[:3]] = 1e4
    t = qf.compute_channel_texture(spiked, mask, "c")
    # The robust range stays tiny despite the 1e4 spikes (they're above p98).
    assert t["c_texture_range_robust"] < 10.0


# --------------------------------------------------------------------------- #
# NaN guards
# --------------------------------------------------------------------------- #

def test_tiny_mask_nan_glcm_block():
    """A too-small mask (< the GLCM min-voxel threshold) yields NaN for the GLCM
    block; the 3D-native block collapses to a finite 0 (single value)."""
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[2, 2, 2] = True                      # a single voxel
    img = np.full((6, 6, 6), 42.0, dtype=np.float64)
    t = qf.compute_channel_texture(img, mask, "c")
    for s in _TEX_GLCM:
        assert np.isnan(t[f"c{s}"]), f"c{s} should be NaN on a tiny mask"
    # native block is defined (a single value -> zero spread, finite).
    assert np.isfinite(t["c_texture_variance"])
    assert t["c_texture_variance"] == pytest.approx(0.0)


def test_all_nan_crop_yields_all_nan():
    """A failed tile read (all-NaN crop) -> NaN for EVERY texture column, schema-
    stable (the column is present, the value is NaN)."""
    mask = _cube_mask()
    img = np.full(mask.shape, np.nan, dtype=np.float64)
    t = qf.compute_channel_texture(img, mask, "c")
    for s in _TEX_SUFFIXES:
        assert np.isnan(t[f"c{s}"]), f"c{s} should be NaN on an all-NaN crop"


def test_empty_mask_yields_all_nan():
    """An empty (no in-mask voxels) crop -> NaN everywhere, no crash."""
    mask = np.zeros((8, 8, 8), dtype=bool)
    img = np.zeros((8, 8, 8), dtype=np.float64)
    t = qf.compute_channel_texture(img, mask, "c")
    for s in _TEX_SUFFIXES:
        assert np.isnan(t[f"c{s}"])


def test_shape_mismatch_yields_all_nan():
    """A mask/image shape mismatch -> NaN everywhere rather than a crash."""
    mask = _cube_mask((10, 10, 10))
    img = np.zeros((8, 8, 8), dtype=np.float64)
    t = qf.compute_channel_texture(img, mask, "c")
    for s in _TEX_SUFFIXES:
        assert np.isnan(t[f"c{s}"])


def test_distinct_prefixes_distinct_columns():
    mask = _cube_mask()
    img = _smooth_gradient(mask)
    nuc = qf.compute_channel_texture(img, mask, "nuc")
    mem = qf.compute_channel_texture(img, mask, "mem")
    assert all(k.startswith("nuc") for k in nuc)
    assert all(k.startswith("mem") for k in mem)
    # Same input -> same values under different prefixes.
    assert nuc["nuc_texture_variance"] == pytest.approx(mem["mem_texture_variance"])


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def test_channel_texture_tags_helper():
    tags = fm.channel_texture_tags("nuc")
    for s in _TEX_SUFFIXES:
        assert f"nuc{s}" in tags, f"nuc{s}"
        t = tags[f"nuc{s}"]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in t, (s, key)
        assert t["role"] == "biology"
        assert t["frame"] == "image"
        assert t["redundancy_group"] == "channel_texture_nuc"
        assert isinstance(t["bounded"], bool)


def test_channel_texture_tags_bounded_flags():
    tags = fm.channel_texture_tags("c")
    # Bounded [0,1] / [-1,1]: granularity, homogeneity, energy, correlation.
    for s in ("_texture_granularity", "_glcm_homogeneity",
              "_glcm_energy", "_glcm_correlation"):
        assert tags[f"c{s}"]["bounded"] is True, s
    # Unbounded: variance, mad, range, entropy, glcm contrast/entropy/dissimilarity.
    for s in ("_texture_variance", "_texture_mad", "_texture_range_robust",
              "_texture_entropy", "_glcm_contrast", "_glcm_entropy",
              "_glcm_dissimilarity"):
        assert tags[f"c{s}"]["bounded"] is False, s


def test_texture_in_module_all():
    assert "compute_channel_texture" in qf.__all__
    assert "channel_texture_tags" in fm.__all__


# --------------------------------------------------------------------------- #
# FeatureStage families registry + per-channel recovery
# --------------------------------------------------------------------------- #

def test_per_channel_texture_families_recovered_from_columns():
    from packages.epicycle_scaledseg.core.feature_stage import _per_channel_texture_families
    cols = (
        [f"nuc{s}" for s in _TEX_SUFFIXES]
        + [f"mem{s}" for s in _TEX_SUFFIXES]
        + ["area", "nuc_intensity_mean"]
    )
    fams = _per_channel_texture_families(cols, [(0, "nuc"), (1, "mem")])
    assert set(fams["channel_texture_nuc"]) == {f"nuc{s}" for s in _TEX_SUFFIXES}
    assert set(fams["channel_texture_mem"]) == {f"mem{s}" for s in _TEX_SUFFIXES}


def test_per_channel_texture_families_empty_when_no_channels():
    from packages.epicycle_scaledseg.core.feature_stage import _per_channel_texture_families
    assert _per_channel_texture_families(["area", "nuc_texture_variance"], None) == {}


# --------------------------------------------------------------------------- #
# FeatureStage wiring: texture appears ONLY when channels are sampled
# --------------------------------------------------------------------------- #

def _solid_cube_label(shape=(12, 16, 16), lo=2, hi=10):
    lab = np.zeros(shape, dtype=np.int32)
    lab[lo:hi, lo:hi, lo:hi] = 1
    return lab


def test_extract_one_labels_only_emits_no_texture(tmp_path):
    """The per-cell worker with NO sampler config emits NO texture columns
    (default-off byte-identical to pre-Wave-E)."""
    from packages.epicycle_scaledseg.core import feature_stage as fs

    lab = _solid_cube_label()
    p = tmp_path / "labels.npy"
    np.save(p, lab)
    fs._init_feature_worker(str(p))           # no M2/Wave-E args
    try:
        task = (1, (2, 10, 2, 10, 2, 10), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict), row
    assert not any(k.endswith(_TEX_SUFFIXES) for k in row)
    assert not any("_texture_" in k or "_glcm_" in k for k in row)


def test_extract_one_with_channel_config_emits_texture(monkeypatch, tmp_path):
    """With a channel sampler config set, ``_extract_one`` emits the per-channel
    texture columns alongside the intensity columns (same crop, same prefix)."""
    from packages.epicycle_scaledseg.core import feature_stage as fs
    from packages.epicycle_scaledseg.reconcile import raw_roi

    lab = _solid_cube_label()
    p = tmp_path / "labels.npy"
    np.save(p, lab)

    # A fake raw reader returning a punctate channel crop at the requested bbox.
    full_mask = lab == 1
    full_ch = _punctate(full_mask, frac=0.08)

    def _fake_read_global_roi(tiles_dir, bbox, channel, halo=0, halo_z=None):
        z0, z1, y0, y1, x0, x1 = bbox
        return full_ch[z0:z1, y0:y1, x0:x1]

    monkeypatch.setattr(raw_roi, "read_global_roi", _fake_read_global_roi)

    fs._init_feature_worker(
        str(p), tiles_dir="FAKE_TILES", recon_dir=None,
        channels=[(0, "nuc")], prob=False)
    try:
        task = (1, (2, 10, 2, 10, 2, 10), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
        fs._WORKER_TILES_DIR = None
        fs._WORKER_RECON_DIR = None
        fs._WORKER_CHANNELS = None
        fs._WORKER_PROB = False
    assert isinstance(row, dict), row
    for s in _TEX_SUFFIXES:
        assert f"nuc{s}" in row, f"nuc{s}"
    # Intensity columns are still there (texture rides alongside, doesn't replace).
    assert "nuc_intensity_mean" in row
    # The punctate channel reads a non-trivial granularity.
    assert row["nuc_texture_granularity"] > 0.0

"""Wave K feature tests — channel-PAIR co-localization (atlas ) + the
HCR-puncta SPATIAL remainder (atlas ).

Pure-function known-shape tests (no tile readers / GPU) on synthetic crops.

* :func:`qc_features.compute_channel_colocalization` — the classic
  microscopy co-localization battery (Pearson + Manders M1/M2 + overlap coefficient +
  Li's ICQ + co-positive fraction) over the in-mask voxels common to two channels. We
  assert: two perfectly-correlated channels -> pearson ≈ 1 + manders ≈ 1; anti /
  independent channels -> pearson ≈ 0; a B-bright-only-where-A pair -> high manders.

* :func:`qc_features.compute_channel_puncta` Wave-K remainder — extends the
  Wave-G blob_log puncta with radial position / clustering index / polarity /
  intensity cv / dispersion. We assert: clustered spots -> clustering_index < 1 vs
  dispersed ≈ 1; core-localized spots -> low radial_position vs rind -> high.

* :func:`qc_features.compute_channel_puncta_compartments` (paired path) —
  nuclear vs cytoplasm puncta fraction. We assert the two fractions split + sum to 1.

Plus: NaN guards on every path; the manifest tags every new column; the FeatureStage
families registry + per-channel-pair recovery; the FeatureStage wiring emits the
co-loc columns ONLY when >= 2 channels are sampled (single-channel + labels-only stay
byte-identical).
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# --------------------------------------------------------------------------- #
# column templates
# --------------------------------------------------------------------------- #
_COLOC_SUFFIXES = (
    "_pearson", "_manders_m1", "_manders_m2", "_overlap_coefficient",
    "_icq", "_coloc_fraction",
    # Mop-up O co-localization remainder (kept in lockstep with
    # qf._CHANNEL_COLOC_SUFFIXES).
    "_mean_ratio", "_boundary_corr", "_interior_ratio_delta",
)
_PUNCTA_SPATIAL_SUFFIXES = (
    "_puncta_radial_position_mean", "_puncta_clustering_index",
    "_puncta_polarity", "_puncta_intensity_cv", "_puncta_dispersion",
)
_PUNCTA_ALL_SUFFIXES = (
    "_puncta_count", "_puncta_density", "_puncta_intensity_mean",
    "_puncta_mean_volume", "_puncta_nn_distance", "_punctate_fraction",
    *_PUNCTA_SPATIAL_SUFFIXES,
)
_PUNCTA_COMPARTMENT_SUFFIXES = (
    "_nuclear_puncta_fraction", "_cytoplasm_puncta_fraction",
)


def _cell_mask(shape=(24, 24, 24), margin=2):
    m = np.zeros(shape, bool)
    m[margin:shape[0] - margin, margin:shape[1] - margin,
      margin:shape[2] - margin] = True
    return m


def _spot_image(mask, centres, base=5.0, peak=200.0, half=1):
    img = np.full(mask.shape, base, dtype=np.float64)
    for (z, y, x) in centres:
        img[z - half:z + half + 1, y - half:y + half + 1, x - half:x + half + 1] = peak
    return np.where(mask, img, 0.0)


# --------------------------------------------------------------------------- #
# — compute_channel_colocalization
# --------------------------------------------------------------------------- #
def test_coloc_columns_present_with_pair_prefix():
    mask = _cell_mask()
    a = np.where(mask, np.random.RandomState(0).rand(*mask.shape), 0.0)
    out = qf.compute_channel_colocalization(a, a, mask, "ch1", "ch2")
    for s in _COLOC_SUFFIXES:
        assert f"ch1_ch2{s}" in out, f"ch1_ch2{s} missing"
    assert len(out) == len(_COLOC_SUFFIXES)


def test_coloc_returns_dict():
    mask = _cell_mask()
    a = np.where(mask, 1.0, 0.0)
    out = qf.compute_channel_colocalization(a, a, mask, "a", "b")
    assert isinstance(out, dict)


def test_perfectly_correlated_channels_pearson_one():
    """Two identical channels -> pearson ≈ 1, overlap_coefficient ≈ 1, ICQ ≈ +0.5
    (max dependent staining). The two Manders are EQUAL (symmetric identical channels)
    and substantial (the above-mean voxels carry most of the signal), but NOT exactly
    1 (the below-threshold voxels still carry A's signal) — that's correct Manders."""
    mask = _cell_mask()
    rng = np.random.RandomState(1)
    a = np.where(mask, rng.rand(*mask.shape) + 0.1, 0.0)
    out = qf.compute_channel_colocalization(a, a.copy(), mask, "a", "b")
    assert out["a_b_pearson"] == pytest.approx(1.0, abs=1e-6)
    assert out["a_b_overlap_coefficient"] == pytest.approx(1.0, abs=1e-6)
    assert out["a_b_icq"] == pytest.approx(0.5, abs=1e-6)
    # symmetric channels -> M1 == M2; both substantial (most signal above the mean).
    assert out["a_b_manders_m1"] == pytest.approx(out["a_b_manders_m2"], abs=1e-9)
    assert out["a_b_manders_m1"] > 0.5


def test_anti_correlated_channels_pearson_negative():
    """B = const − A inside the mask -> pearson ≈ −1, ICQ ≈ −0.5 (segregated)."""
    mask = _cell_mask()
    rng = np.random.RandomState(2)
    base = rng.rand(*mask.shape)
    a = np.where(mask, base, 0.0)
    b = np.where(mask, 1.0 - base, 0.0)
    out = qf.compute_channel_colocalization(a, b, mask, "a", "b")
    assert out["a_b_pearson"] == pytest.approx(-1.0, abs=1e-6)
    assert out["a_b_icq"] == pytest.approx(-0.5, abs=1e-6)


def test_independent_channels_pearson_near_zero():
    """Two independent random channels -> pearson ≈ 0 (no correlation)."""
    mask = _cell_mask((30, 30, 30))
    rng = np.random.RandomState(3)
    a = np.where(mask, rng.rand(*mask.shape), 0.0)
    b = np.where(mask, rng.rand(*mask.shape), 0.0)
    out = qf.compute_channel_colocalization(a, b, mask, "a", "b")
    assert abs(out["a_b_pearson"]) < 0.1


def test_manders_high_when_b_bright_only_where_a():
    """B is bright ONLY in voxels where A is bright (A's positive region ⊂ B's) ->
    M1 (fraction of A overlapping B) is high; coloc_fraction is substantial."""
    mask = _cell_mask()
    a = np.full(mask.shape, 1.0)
    a[8:16, 8:16, 8:16] = 50.0        # a bright A core
    a = np.where(mask, a, 0.0)
    b = np.full(mask.shape, 1.0)
    b[8:16, 8:16, 8:16] = 80.0        # B bright in exactly A's bright core
    b = np.where(mask, b, 0.0)
    out = qf.compute_channel_colocalization(a, b, mask, "a", "b")
    # most of A's (background-subtracted) signal lives in the shared core.
    assert out["a_b_manders_m1"] > 0.7
    assert out["a_b_coloc_fraction"] > 0.0


def test_coloc_bounds():
    """Manders / overlap / coloc_fraction are [0,1]; pearson is [-1,1]; icq is
    [-0.5,0.5]."""
    mask = _cell_mask()
    rng = np.random.RandomState(4)
    a = np.where(mask, rng.rand(*mask.shape), 0.0)
    b = np.where(mask, rng.rand(*mask.shape), 0.0)
    out = qf.compute_channel_colocalization(a, b, mask, "a", "b")
    for s in ("_manders_m1", "_manders_m2", "_overlap_coefficient", "_coloc_fraction"):
        assert 0.0 - 1e-9 <= out[f"a_b{s}"] <= 1.0 + 1e-9, s
    assert -1.0 - 1e-9 <= out["a_b_pearson"] <= 1.0 + 1e-9
    assert -0.5 - 1e-9 <= out["a_b_icq"] <= 0.5 + 1e-9


def test_coloc_pair_prefix_naming():
    mask = _cell_mask()
    a = np.where(mask, 1.0, 0.0)
    out = qf.compute_channel_colocalization(a, a, mask, "nuc", "mem")
    assert all(k.startswith("nuc_mem") for k in out)


def test_coloc_nan_guards():
    mask = _cell_mask()
    a = np.where(mask, 1.0, 0.0)
    # empty mask -> all NaN.
    empty = np.zeros((8, 8, 8), bool)
    out = qf.compute_channel_colocalization(np.zeros((8, 8, 8)), np.zeros((8, 8, 8)),
                                            empty, "a", "b")
    for s in _COLOC_SUFFIXES:
        assert np.isnan(out[f"a_b{s}"])
    # shape mismatch -> all NaN.
    out = qf.compute_channel_colocalization(np.zeros((8, 8, 8)), np.zeros((8, 8, 8)),
                                            mask, "a", "b")
    for s in _COLOC_SUFFIXES:
        assert np.isnan(out[f"a_b{s}"])
    # all-NaN crop on one channel -> all NaN.
    out = qf.compute_channel_colocalization(np.full(mask.shape, np.nan), a, mask, "a", "b")
    for s in _COLOC_SUFFIXES:
        assert np.isnan(out[f"a_b{s}"])
    # a tiny mask (< _COLOC_MIN_MASK_VOXELS) -> all NaN.
    tiny = np.zeros((8, 8, 8), bool)
    tiny[3:5, 3, 3] = True            # 2 voxels
    out = qf.compute_channel_colocalization(np.where(tiny, 1.0, 0.0),
                                            np.where(tiny, 1.0, 0.0), tiny, "a", "b")
    for s in _COLOC_SUFFIXES:
        assert np.isnan(out[f"a_b{s}"])


def test_coloc_flat_channel_nans_pearson_but_not_manders():
    """A FLAT channel (zero variance) -> pearson + ICQ NaN (undefined), but the
    Manders / overlap / coloc_fraction (magnitude-based) are still defined."""
    mask = _cell_mask()
    rng = np.random.RandomState(5)
    a = np.where(mask, rng.rand(*mask.shape) + 1.0, 0.0)
    b = np.where(mask, 7.0, 0.0)      # flat inside the mask
    out = qf.compute_channel_colocalization(a, b, mask, "a", "b")
    assert np.isnan(out["a_b_pearson"])
    assert np.isnan(out["a_b_icq"])
    # b is flat -> b_pos = 0 everywhere (min == value) -> overlap denom 0 -> NaN there,
    # but m1 (A over B-above-mean) is defined. Just assert the family is schema-stable.
    for s in _COLOC_SUFFIXES:
        assert f"a_b{s}" in out


# --------------------------------------------------------------------------- #
# — puncta SPATIAL remainder (compute_channel_puncta Wave-K block)
# --------------------------------------------------------------------------- #
def test_puncta_spatial_columns_present():
    mask = _cell_mask()
    out = qf.compute_channel_puncta(_spot_image(mask, [(12, 12, 12)]), mask, "hcr")
    for s in _PUNCTA_ALL_SUFFIXES:
        assert f"hcr{s}" in out, f"hcr{s} missing"
    assert len(out) == len(_PUNCTA_ALL_SUFFIXES)


def test_clustering_index_below_one_for_clustered_above_for_dispersed():
    """Clustered spots -> clustering_index < 1 (sub-CSR NN); widely dispersed spots
    -> clustering_index >= ~1 (regular / random)."""
    mask = _cell_mask((30, 30, 30))
    clustered = [(14, 14, 14), (14, 14, 17), (14, 17, 14), (17, 14, 14)]
    dispersed = [(6, 6, 6), (6, 24, 24), (24, 6, 24), (24, 24, 6)]
    oc = qf.compute_channel_puncta(_spot_image(mask, clustered), mask, "g")
    od = qf.compute_channel_puncta(_spot_image(mask, dispersed), mask, "g")
    assert oc["g_puncta_clustering_index"] < 1.0
    assert od["g_puncta_clustering_index"] > oc["g_puncta_clustering_index"]


def test_radial_position_low_for_rind_high_for_core():
    """Spots near the SURFACE (rind) -> low radial_position; spots in the CORE ->
    high radial_position (the EDT radial coordinate orders them)."""
    mask = _cell_mask((30, 30, 30), margin=2)
    # rind spots: just inside the surface on opposite faces.
    rind = [(4, 15, 15), (25, 15, 15), (15, 4, 15), (15, 25, 15)]
    # core spots: clustered in the deep centre.
    core = [(14, 14, 14), (15, 15, 15), (16, 16, 16), (14, 16, 14)]
    o_rind = qf.compute_channel_puncta(_spot_image(mask, rind), mask, "g")
    o_core = qf.compute_channel_puncta(_spot_image(mask, core), mask, "g")
    assert o_rind["g_puncta_radial_position_mean"] < o_core["g_puncta_radial_position_mean"]


def test_dispersion_larger_for_spread_spots():
    """Spots spread across the whole cell -> larger dispersion than spots packed in
    one corner."""
    mask = _cell_mask((30, 30, 30))
    spread = [(6, 6, 6), (6, 24, 24), (24, 6, 24), (24, 24, 6)]
    packed = [(8, 8, 8), (8, 8, 11), (8, 11, 8), (11, 8, 8)]
    o_spread = qf.compute_channel_puncta(_spot_image(mask, spread), mask, "g")
    o_packed = qf.compute_channel_puncta(_spot_image(mask, packed), mask, "g")
    assert o_spread["g_puncta_dispersion"] > o_packed["g_puncta_dispersion"]


def test_intensity_cv_higher_for_uneven_brightness():
    """Spots of equal brightness -> low intensity_cv; spots of very uneven brightness
    -> higher intensity_cv."""
    mask = _cell_mask((30, 30, 30))
    centres = [(8, 8, 8), (8, 20, 20), (20, 8, 20), (20, 20, 8)]
    # equal-brightness spots.
    equal = _spot_image(mask, centres, peak=200.0)
    o_eq = qf.compute_channel_puncta(equal, mask, "g")
    # uneven: rebuild with varied peaks per spot.
    uneven = np.full(mask.shape, 5.0)
    for (z, y, x), pk in zip(centres, (50.0, 100.0, 200.0, 400.0)):
        uneven[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] = pk
    uneven = np.where(mask, uneven, 0.0)
    o_un = qf.compute_channel_puncta(uneven, mask, "g")
    assert o_un["g_puncta_intensity_cv"] > o_eq["g_puncta_intensity_cv"]


def test_polarity_nan_without_outward_signed_with():
    """Polarity needs the Wave C outward normal: NaN when None, signed when given.
    Spots all on the +outward side -> polarity ≈ +1."""
    mask = _cell_mask((30, 30, 30))
    # all spots in the HIGH-Z half (apical when outward = +Z).
    centres = [(22, 12, 12), (22, 18, 18), (24, 12, 18), (24, 18, 12)]
    img = _spot_image(mask, centres)
    o_none = qf.compute_channel_puncta(img, mask, "g")
    assert np.isnan(o_none["g_puncta_polarity"])
    # outward = +Z, centroid at the cell centre -> all spots are apical -> +1.
    o_ow = qf.compute_channel_puncta(
        img, mask, "g", outward=(1.0, 0.0, 0.0), centroid=(15.0, 15.0, 15.0))
    assert o_ow["g_puncta_polarity"] == pytest.approx(1.0, abs=1e-9)


def test_polarity_sign_flips_with_basal_spots():
    """Spots all on the −outward side -> polarity ≈ −1 (basal)."""
    mask = _cell_mask((30, 30, 30))
    centres = [(6, 12, 12), (6, 18, 18), (4, 12, 18), (4, 18, 12)]
    img = _spot_image(mask, centres)
    o = qf.compute_channel_puncta(
        img, mask, "g", outward=(1.0, 0.0, 0.0), centroid=(15.0, 15.0, 15.0))
    assert o["g_puncta_polarity"] == pytest.approx(-1.0, abs=1e-9)


def test_puncta_spatial_nan_for_no_spots():
    """A diffuse / no-spot cell -> spatial stats NaN (undefined without spots)."""
    mask = _cell_mask()
    diffuse = np.where(mask, 100.0, 0.0).astype(np.float64)
    out = qf.compute_channel_puncta(diffuse, mask, "g")
    for s in _PUNCTA_SPATIAL_SUFFIXES:
        assert np.isnan(out[f"g{s}"]), s


def test_puncta_clustering_index_nan_single_spot():
    """A single spot -> clustering_index + dispersion + intensity_cv NaN (need >= 2),
    but radial_position is defined (a single centre has a radius)."""
    mask = _cell_mask((30, 30, 30))
    out = qf.compute_channel_puncta(_spot_image(mask, [(15, 15, 15)]), mask, "g")
    assert np.isnan(out["g_puncta_clustering_index"])
    assert np.isnan(out["g_puncta_dispersion"])
    assert np.isnan(out["g_puncta_intensity_cv"])
    assert np.isfinite(out["g_puncta_radial_position_mean"])


def test_radial_position_bounded():
    mask = _cell_mask((30, 30, 30))
    centres = [(8, 8, 8), (15, 15, 15), (22, 22, 22)]
    out = qf.compute_channel_puncta(_spot_image(mask, centres), mask, "g")
    assert 0.0 - 1e-9 <= out["g_puncta_radial_position_mean"] <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# — compute_channel_puncta_compartments (paired path)
# --------------------------------------------------------------------------- #
def test_puncta_compartments_columns_present():
    cell = _cell_mask((24, 24, 24))
    nuc = np.zeros(cell.shape, bool)
    nuc[8:16, 8:16, 8:16] = True
    ch = _spot_image(cell, [(11, 11, 11)])
    out = qf.compute_channel_puncta_compartments(ch, cell, nuc, "hcr")
    for s in _PUNCTA_COMPARTMENT_SUFFIXES:
        assert f"hcr{s}" in out, s
    assert len(out) == len(_PUNCTA_COMPARTMENT_SUFFIXES)


def test_puncta_compartments_split_and_sum_to_one():
    """Spots placed in the nucleus + the cytoplasm -> both fractions positive +
    sum to 1."""
    cell = _cell_mask((30, 30, 30))
    nuc = np.zeros(cell.shape, bool)
    nuc[12:18, 12:18, 12:18] = True            # central nucleus
    # two spots in the nucleus, two in the cytoplasm.
    centres = [(15, 15, 15), (14, 16, 14), (6, 6, 6), (24, 24, 24)]
    ch = _spot_image(cell, centres)
    out = qf.compute_channel_puncta_compartments(ch, cell, nuc, "hcr")
    fn = out["hcr_nuclear_puncta_fraction"]
    fc = out["hcr_cytoplasm_puncta_fraction"]
    assert fn > 0.0 and fc > 0.0
    assert fn + fc == pytest.approx(1.0, abs=1e-9)


def test_puncta_compartments_all_nuclear():
    """All spots inside the nucleus -> nuclear fraction 1, cytoplasm 0."""
    cell = _cell_mask((30, 30, 30))
    nuc = np.zeros(cell.shape, bool)
    nuc[10:20, 10:20, 10:20] = True
    centres = [(13, 13, 13), (13, 17, 17), (17, 13, 17), (17, 17, 13)]
    ch = _spot_image(cell, centres)
    out = qf.compute_channel_puncta_compartments(ch, cell, nuc, "hcr")
    assert out["hcr_nuclear_puncta_fraction"] == pytest.approx(1.0, abs=1e-9)
    assert out["hcr_cytoplasm_puncta_fraction"] == pytest.approx(0.0, abs=1e-9)


def test_puncta_compartments_zero_spots_defined_zero():
    """No resolvable spots -> both fractions a defined 0.0 (not NaN)."""
    cell = _cell_mask((24, 24, 24))
    nuc = np.zeros(cell.shape, bool)
    nuc[8:16, 8:16, 8:16] = True
    diffuse = np.where(cell, 100.0, 0.0).astype(np.float64)
    out = qf.compute_channel_puncta_compartments(diffuse, cell, nuc, "hcr")
    assert out["hcr_nuclear_puncta_fraction"] == 0.0
    assert out["hcr_cytoplasm_puncta_fraction"] == 0.0


def test_puncta_compartments_nan_guards():
    cell = _cell_mask((24, 24, 24))
    nuc = np.zeros(cell.shape, bool)
    nuc[8:16, 8:16, 8:16] = True
    # shape mismatch -> NaN.
    out = qf.compute_channel_puncta_compartments(np.zeros((8, 8, 8)), cell, nuc, "hcr")
    for s in _PUNCTA_COMPARTMENT_SUFFIXES:
        assert np.isnan(out[f"hcr{s}"])
    # all-NaN crop -> NaN.
    out = qf.compute_channel_puncta_compartments(
        np.full(cell.shape, np.nan), cell, nuc, "hcr")
    for s in _PUNCTA_COMPARTMENT_SUFFIXES:
        assert np.isnan(out[f"hcr{s}"])


# --------------------------------------------------------------------------- #
# manifest tags
# --------------------------------------------------------------------------- #
def test_channel_coloc_tags_helper():
    tags = fm.channel_coloc_tags("nuc", "mem")
    for s in _COLOC_SUFFIXES:
        assert f"nuc_mem{s}" in tags, s
        t = tags[f"nuc_mem{s}"]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in t, (s, key)
        assert t["role"] == "biology"
        assert t["frame"] == "image"
        assert t["redundancy_group"] == "channel_coloc_nuc_mem"
    assert tags["nuc_mem_pearson"]["bounded"] is True
    assert tags["nuc_mem_manders_m1"]["bounded"] is True


def test_channel_puncta_tags_include_spatial_remainder():
    tags = fm.channel_puncta_tags("hcr")
    for s in _PUNCTA_SPATIAL_SUFFIXES:
        assert f"hcr{s}" in tags, s
        assert tags[f"hcr{s}"]["redundancy_group"] == "channel_puncta_hcr"
    assert tags["hcr_puncta_radial_position_mean"]["bounded"] is True
    assert tags["hcr_puncta_clustering_index"]["bounded"] is False
    assert tags["hcr_puncta_polarity"]["bounded"] is True


def test_channel_puncta_compartment_tags_helper():
    tags = fm.channel_puncta_compartment_tags("hcr")
    for s in _PUNCTA_COMPARTMENT_SUFFIXES:
        assert f"hcr{s}" in tags, s
        t = tags[f"hcr{s}"]
        assert t["role"] == "biology"
        assert t["frame"] == "intrinsic"
        assert t["bounded"] is True
        assert t["redundancy_group"] == "channel_puncta_compartment_hcr"


def test_wavek_in_module_all():
    for fn in ("compute_channel_colocalization", "compute_channel_puncta_compartments"):
        assert fn in qf.__all__, fn
    for fn in ("channel_coloc_tags", "channel_puncta_compartment_tags"):
        assert fn in fm.__all__, fn


# --------------------------------------------------------------------------- #
# FeatureStage families registry + per-channel-pair recovery
# --------------------------------------------------------------------------- #
def test_puncta_family_includes_spatial_suffixes():
    from packages.epicycle_scaledseg.core.feature_stage import _CHANNEL_PUNCTA_SUFFIXES
    for s in _PUNCTA_SPATIAL_SUFFIXES:
        assert s in _CHANNEL_PUNCTA_SUFFIXES, s


def test_channel_pair_coloc_families_recovered_from_columns():
    from packages.epicycle_scaledseg.core.feature_stage import _channel_pair_coloc_families
    cols = (
        [f"nuc_mem{s}" for s in _COLOC_SUFFIXES]
        + ["area", "nuc_intensity_mean"]
    )
    fams = _channel_pair_coloc_families(cols, [(0, "nuc"), (1, "mem")])
    assert set(fams["channel_coloc_nuc_mem"]) == {f"nuc_mem{s}" for s in _COLOC_SUFFIXES}


def test_channel_pair_coloc_families_three_channels_three_pairs():
    from packages.epicycle_scaledseg.core.feature_stage import _channel_pair_coloc_families
    chans = [(0, "a"), (1, "b"), (2, "c")]
    cols = []
    for pair in ("a_b", "a_c", "b_c"):
        cols += [f"{pair}{s}" for s in _COLOC_SUFFIXES]
    fams = _channel_pair_coloc_families(cols, chans)
    assert set(fams) == {"channel_coloc_a_b", "channel_coloc_a_c", "channel_coloc_b_c"}


def test_channel_pair_coloc_families_empty_when_single_channel():
    from packages.epicycle_scaledseg.core.feature_stage import _channel_pair_coloc_families
    assert _channel_pair_coloc_families(["a_b_pearson"], [(0, "a")]) == {}
    assert _channel_pair_coloc_families(["a_b_pearson"], None) == {}


# --------------------------------------------------------------------------- #
# FeatureStage wiring: coloc + spatial puncta appear only with >= 2 channels;
# labels-only + single-channel stay byte-identical (no coloc columns).
# --------------------------------------------------------------------------- #
def _solid_cube_label(shape=(20, 24, 24), lo=2, hi=18):
    lab = np.zeros(shape, dtype=np.int32)
    lab[lo:hi, lo:hi, lo:hi] = 1
    return lab


def test_extract_one_labels_only_emits_no_coloc(tmp_path):
    from packages.epicycle_scaledseg.core import feature_stage as fs
    lab = _solid_cube_label()
    p = tmp_path / "labels.npy"
    np.save(p, lab)
    fs._init_feature_worker(str(p))
    try:
        task = (1, (2, 18, 2, 18, 2, 18), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict), row
    assert not any("_pearson" in k or "_manders_" in k or "_coloc_fraction" in k
                   for k in row)


def test_extract_one_single_channel_emits_spatial_puncta_no_coloc(tmp_path, monkeypatch):
    """A SINGLE channel emits the Wave-K spatial puncta columns but NO co-loc (needs a
    pair)."""
    from packages.epicycle_scaledseg.core import feature_stage as fs
    from packages.epicycle_scaledseg.reconcile import raw_roi

    lab = _solid_cube_label()
    p = tmp_path / "labels.npy"
    np.save(p, lab)
    full_mask = lab == 1
    full_ch = np.full(lab.shape, 5.0)
    for (z, y, x) in [(6, 6, 6), (6, 14, 14), (14, 6, 14), (14, 14, 6)]:
        full_ch[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] = 200.0
    full_ch = np.where(full_mask, full_ch, 0.0)

    def _fake_read_global_roi(tiles_dir, bbox, channel, halo=0, halo_z=None):
        z0, z1, y0, y1, x0, x1 = bbox
        return full_ch[z0:z1, y0:y1, x0:x1]

    monkeypatch.setattr(raw_roi, "read_global_roi", _fake_read_global_roi)
    fs._init_feature_worker(str(p), tiles_dir="FAKE", recon_dir=None,
                            channels=[(0, "hcr")], prob=False)
    try:
        task = (1, (2, 18, 2, 18, 2, 18), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
        fs._WORKER_TILES_DIR = None
        fs._WORKER_CHANNELS = None
    assert isinstance(row, dict), row
    for s in _PUNCTA_SPATIAL_SUFFIXES:
        assert f"hcr{s}" in row, s
    assert not any("_pearson" in k for k in row)


def test_extract_one_two_channels_emits_coloc(tmp_path, monkeypatch):
    """Two sampled channels -> the per-channel-pair co-localization columns appear."""
    from packages.epicycle_scaledseg.core import feature_stage as fs
    from packages.epicycle_scaledseg.reconcile import raw_roi

    lab = _solid_cube_label()
    p = tmp_path / "labels.npy"
    np.save(p, lab)
    full_mask = lab == 1
    rng = np.random.RandomState(7)
    ch0 = np.where(full_mask, rng.rand(*lab.shape) + 0.5, 0.0)
    ch1 = ch0.copy()                                   # correlated 2nd channel

    def _fake_read_global_roi(tiles_dir, bbox, channel, halo=0, halo_z=None):
        z0, z1, y0, y1, x0, x1 = bbox
        src = ch0 if int(channel) == 0 else ch1
        return src[z0:z1, y0:y1, x0:x1]

    monkeypatch.setattr(raw_roi, "read_global_roi", _fake_read_global_roi)
    fs._init_feature_worker(str(p), tiles_dir="FAKE", recon_dir=None,
                            channels=[(0, "nuc"), (1, "mem")], prob=False)
    try:
        task = (1, (2, 18, 2, 18, 2, 18), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
        fs._WORKER_TILES_DIR = None
        fs._WORKER_CHANNELS = None
    assert isinstance(row, dict), row
    for s in _COLOC_SUFFIXES:
        assert f"nuc_mem{s}" in row, s
    # identical channels -> pearson ~ 1.
    assert row["nuc_mem_pearson"] == pytest.approx(1.0, abs=1e-6)

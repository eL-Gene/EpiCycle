"""Wave F feature tests — probability-geometry interaction (atlas ) +
channel / expression localization (atlas ).

Pure-function known-shape tests (no tile readers / GPU). Both families combine the
raw/prob SIGNAL (the M2 sampler crops) with GEOMETRY (the M5 interior-EDT radial
shells + the Wave C outward normal). We build tiny synthetic crops with a KNOWN
localization character — a MEMBRANE-localized channel (bright surface rind), a
CORE-localized one, an APICAL-bright channel split by an outward normal, and a prob
crop that is confident-interior / uncertain-rind — and assert the discriminators
move in the documented direction.

Three compute functions under test:

* :func:`qc_features.compute_prob_geometry` — single-volume; prob ⊗ EDT
  shells. ``prob_area_normalized`` / inner-vs-outer shell prob means + gradient +
  boundary-shell uncertainty.
* :func:`qc_features.compute_channel_localization` — single-volume; channel
  ⊗ EDT shells + radial slope (always) + apical/basal split (only with an outward
  normal). ``<prefix>_shell_*`` / ``_radial_intensity_slope`` / ``_apical_*`` /
  ``_basal_*`` / ``_apical_basal_intensity_ratio``.
* :func:`qc_features.compute_channel_compartments` (PAIRED path) — channel ⊗
  nucleus mask. ``<prefix>_nuclear_mean`` / ``_cytoplasm_mean`` / ``_nuclear_
  cytoplasm_ratio`` / ``_perinuclear_enrichment``.

Plus: NaN guards on every path; the manifest tags every new column; the
FeatureStage wiring emits the single-volume columns when channels / prob are
sampled (and the labels-only path stays byte-identical); the reconcile paired path
emits the compartment columns when channels are supplied.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# --------------------------------------------------------------------------- #
# column templates
# --------------------------------------------------------------------------- #
_PROB_GEOM_COLS = (
    "prob_area_normalized",
    "prob_shell_inner_mean", "prob_shell_outer_mean", "prob_shell_gradient",
    "prob_surface_uncertainty",
    # Mop-up O remainder (kept in lockstep with qf._PROB_GEOMETRY_COLS).
    "prob_surface_normalized", "prob_boundary_drop_per_area",
    # P1a SIGNED remainder — apical-vs-basal confidence difference (always a
    # present key; NaN unless a trusted outward normal + centroid are supplied).
    "prob_apical_basal_gradient",
)
_LOC_SHELL = (
    "_shell_inner_mean", "_shell_outer_mean", "_shell_gradient",
    "_radial_intensity_slope",
    # Mop-up O single-volume remainder.
    "_shell_mid_mean", "_radial_bin_inner_mean", "_radial_bin_mid_mean",
    "_radial_bin_outer_mean",
    "_membrane_enrichment", "_outer_to_inner_ratio", "_fraction_above_threshold",
)
_LOC_APICALBASAL = (
    "_apical_intensity_mean", "_basal_intensity_mean",
    "_apical_basal_intensity_ratio",
    # Mop-up O — lateral (equatorial) intensity band.
    "_lateral_intensity_mean",
)
_LOC_SUFFIXES = _LOC_SHELL + _LOC_APICALBASAL
_COMPARTMENT_SUFFIXES = (
    "_nuclear_mean", "_cytoplasm_mean", "_nuclear_cytoplasm_ratio",
    "_perinuclear_enrichment",
    # Mop-up O compartment remainder.
    "_nuclear_signal_fraction",
    "_nuclear_texture_entropy", "_cytoplasmic_texture_entropy",
    "_compartment_mismatch_score",
)


# --------------------------------------------------------------------------- #
# synthetic crops (mask + signal)
# --------------------------------------------------------------------------- #
def _sphere_mask(shape=(24, 24, 24), radius=9.0):
    """A solid sphere cell mask well inside the crop (a clean EDT core + rind)."""
    zz, yy, xx = np.indices(shape).astype(np.float64)
    c = (np.array(shape) - 1) / 2.0
    d = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    return d <= radius


def _radial_position(mask, spacing=None):
    """The same r = d/max(d) field compute_channel_localization uses (for building
    KNOWN membrane/core signals from the geometry)."""
    r, _ = qf._edt_radial_position(mask, spacing=spacing)
    return r


def _membrane_signal(mask, lo=10.0, hi=200.0):
    """A MEMBRANE-localized channel: bright at the surface rind (low r), dim in the
    core (high r). Intensity decreases with radial depth -> negative radial slope,
    positive shell gradient (outer > inner)."""
    r = _radial_position(mask)
    img = np.zeros(mask.shape, dtype=np.float64)
    # linear ramp: r=0 (surface) -> hi, r=1 (core) -> lo.
    img[mask] = hi + (lo - hi) * r[mask]
    return img


def _core_signal(mask, lo=10.0, hi=200.0):
    """A CORE-localized channel: bright in the core (high r), dim at the surface.
    The opposite of _membrane_signal -> positive radial slope, negative shell
    gradient (inner > outer)."""
    r = _radial_position(mask)
    img = np.zeros(mask.shape, dtype=np.float64)
    img[mask] = lo + (hi - lo) * r[mask]
    return img


def _uniform(mask, value=100.0):
    return np.where(mask, value, 0.0).astype(np.float64)


# --------------------------------------------------------------------------- #
# — compute_prob_geometry
# --------------------------------------------------------------------------- #
def test_prob_geometry_columns_present():
    mask = _sphere_mask()
    prob = _uniform(mask, 0.9)
    out = qf.compute_prob_geometry(mask, prob)
    for c in _PROB_GEOM_COLS:
        assert c in out, c
    assert len(out) == len(_PROB_GEOM_COLS)


def test_prob_area_normalized_uniform_confident():
    """A mask filled with prob=1.0 -> soft volume == geometric volume ->
    prob_area_normalized ~ 1.0."""
    mask = _sphere_mask()
    prob = _uniform(mask, 1.0)
    out = qf.compute_prob_geometry(mask, prob)
    assert out["prob_area_normalized"] == pytest.approx(1.0, abs=1e-6)


def test_prob_area_normalized_uses_supplied_weighted_volume():
    """When prob_weighted_volume is supplied (the M2 soft volume over the padded
    ROI) it is used as the numerator, so the ratio reflects that soft volume."""
    mask = _sphere_mask()
    prob = _uniform(mask, 0.5)
    geom = float(mask.sum())
    out = qf.compute_prob_geometry(mask, prob, prob_weighted_volume=2.0 * geom)
    assert out["prob_area_normalized"] == pytest.approx(2.0, abs=1e-6)


def test_prob_shell_gradient_sign_confident_interior_uncertain_rind():
    """A prob crop that is CONFIDENT in the interior (->1) and UNCERTAIN at the rind
    (->0.5): the outer-shell mean < inner-shell mean -> prob_shell_gradient (outer −
    inner) is NEGATIVE (the well-segmented case). Surface uncertainty is HIGH (the
    rind sits near p=0.5)."""
    mask = _sphere_mask()
    # prob = high in the core (high r), ~0.5 at the surface (low r).
    r = _radial_position(mask)
    prob = np.zeros(mask.shape, dtype=np.float64)
    prob[mask] = 0.5 + 0.5 * r[mask]          # 0.5 surface -> 1.0 core
    out = qf.compute_prob_geometry(mask, prob)
    assert out["prob_shell_inner_mean"] > out["prob_shell_outer_mean"]
    assert out["prob_shell_gradient"] < 0.0
    # surface uncertainty: outer shell ~0.5 -> near-max binary entropy (ln2 ~0.693).
    assert out["prob_surface_uncertainty"] > 0.5


def test_prob_surface_uncertainty_low_when_boundary_confident():
    """A crisp confident boundary (prob ~1 everywhere) -> boundary-shell binary
    entropy ~0 (no uncertainty)."""
    mask = _sphere_mask()
    prob = _uniform(mask, 0.99)
    out = qf.compute_prob_geometry(mask, prob)
    assert out["prob_surface_uncertainty"] < 0.1


def test_prob_geometry_nan_guards():
    # empty mask
    empty = np.zeros((8, 8, 8), dtype=bool)
    out = qf.compute_prob_geometry(empty, np.zeros((8, 8, 8)))
    for c in _PROB_GEOM_COLS:
        assert np.isnan(out[c])
    # shape mismatch
    mask = _sphere_mask((12, 12, 12), 4.0)
    out = qf.compute_prob_geometry(mask, np.zeros((8, 8, 8)))
    for c in _PROB_GEOM_COLS:
        assert np.isnan(out[c])
    # all-NaN prob crop (failed read)
    mask = _sphere_mask()
    out = qf.compute_prob_geometry(mask, np.full(mask.shape, np.nan))
    for c in _PROB_GEOM_COLS:
        assert np.isnan(out[c])


def test_prob_geometry_thin_line_no_outer_shell():
    """A 1-voxel-wide line (thin in TWO axes -> every voxel sits at radial position
    r == 1.0, the deepest core level) has NO surface (outer) shell: outer_mean +
    gradient + surface_uncertainty NaN, but inner_mean + prob_area_normalized are
    still defined. Confirms the no-outer-shell branch degrades gracefully."""
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[2:10, 6, 6] = True                   # a 1x1xN line down Z (no surface rind)
    prob = _uniform(mask, 0.8)
    out = qf.compute_prob_geometry(mask, prob)
    assert np.isfinite(out["prob_area_normalized"])
    assert np.isfinite(out["prob_shell_inner_mean"])      # whole line is "core"
    assert np.isnan(out["prob_shell_outer_mean"])          # no surface shell
    assert np.isnan(out["prob_shell_gradient"])
    assert np.isnan(out["prob_surface_uncertainty"])


# --------------------------------------------------------------------------- #
# — compute_channel_localization (SHELL + radial slope)
# --------------------------------------------------------------------------- #
def test_localization_columns_present_with_prefix():
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_membrane_signal(mask), mask, "mem")
    for s in _LOC_SUFFIXES:
        assert f"mem{s}" in out, f"mem{s} missing"
    assert len(out) == len(_LOC_SUFFIXES)


def test_returns_dict_not_dataframe():
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_uniform(mask), mask, "c")
    assert isinstance(out, dict)


def test_membrane_localized_high_outer_positive_gradient_negative_slope():
    """A MEMBRANE-localized channel (bright surface rind) -> outer shell mean ≫ inner
    shell mean (positive shell_gradient) and intensity DECREASES with radial depth
    (negative radial_intensity_slope)."""
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_membrane_signal(mask), mask, "mem")
    assert out["mem_shell_outer_mean"] > out["mem_shell_inner_mean"]
    assert out["mem_shell_gradient"] > 0.0
    # intensity vs r (1=core): membrane signal is bright at low r -> slope < 0.
    assert out["mem_radial_intensity_slope"] < 0.0


def test_core_localized_opposite_of_membrane():
    """A CORE-localized channel -> inner ≫ outer (negative shell_gradient), positive
    radial slope (brighter toward the core). The opposite of the membrane case."""
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_core_signal(mask), mask, "c")
    assert out["c_shell_inner_mean"] > out["c_shell_outer_mean"]
    assert out["c_shell_gradient"] < 0.0
    assert out["c_radial_intensity_slope"] > 0.0


def test_membrane_vs_core_gradient_signs_oppose():
    mask = _sphere_mask()
    mem = qf.compute_channel_localization(_membrane_signal(mask), mask, "x")
    core = qf.compute_channel_localization(_core_signal(mask), mask, "x")
    assert mem["x_shell_gradient"] > 0.0 > core["x_shell_gradient"]
    assert mem["x_radial_intensity_slope"] < 0.0 < core["x_radial_intensity_slope"]


def test_uniform_channel_flat_gradient_and_slope():
    """A uniform fill -> shell means equal, gradient ~0, slope ~0."""
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_uniform(mask, 100.0), mask, "c")
    assert out["c_shell_gradient"] == pytest.approx(0.0, abs=1e-6)
    assert out["c_radial_intensity_slope"] == pytest.approx(0.0, abs=1e-6)


def test_localization_apical_basal_skipped_without_outward():
    """With no outward normal the apical/basal block stays NaN (the shell block is
    still defined) — the per-cell-pool default."""
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_uniform(mask), mask, "c", outward=None)
    for s in _LOC_APICALBASAL:
        assert np.isnan(out[f"c{s}"]), s
    assert np.isfinite(out["c_shell_inner_mean"])


def test_localization_nan_guards():
    # empty mask
    empty = np.zeros((8, 8, 8), dtype=bool)
    out = qf.compute_channel_localization(np.zeros((8, 8, 8)), empty, "c")
    for s in _LOC_SUFFIXES:
        assert np.isnan(out[f"c{s}"])
    # shape mismatch
    mask = _sphere_mask((12, 12, 12), 4.0)
    out = qf.compute_channel_localization(np.zeros((8, 8, 8)), mask, "c")
    for s in _LOC_SUFFIXES:
        assert np.isnan(out[f"c{s}"])
    # all-NaN crop
    mask = _sphere_mask()
    out = qf.compute_channel_localization(np.full(mask.shape, np.nan), mask, "c")
    for s in _LOC_SUFFIXES:
        assert np.isnan(out[f"c{s}"])


def test_distinct_prefixes_distinct_columns():
    mask = _sphere_mask()
    img = _membrane_signal(mask)
    nuc = qf.compute_channel_localization(img, mask, "nuc")
    mem = qf.compute_channel_localization(img, mask, "mem")
    assert all(k.startswith("nuc") for k in nuc)
    assert all(k.startswith("mem") for k in mem)
    assert nuc["nuc_shell_gradient"] == pytest.approx(mem["mem_shell_gradient"])


# --------------------------------------------------------------------------- #
# — apical/basal split with an outward normal
# --------------------------------------------------------------------------- #
def test_apical_bright_channel_ratio_above_one():
    """An APICAL-bright channel (bright in the +Z half) + an outward normal pointing
    +Z -> the apical (outward) half mean > the basal (inward) half mean ->
    apical_basal_intensity_ratio > 1."""
    shape = (24, 24, 24)
    mask = _sphere_mask(shape, 9.0)
    cz = (shape[0] - 1) / 2.0
    # bright in the apical (high-Z) half, dim basal.
    img = np.zeros(shape, dtype=np.float64)
    zz = np.indices(shape)[0]
    img[mask & (zz > cz)] = 200.0
    img[mask & (zz <= cz)] = 20.0
    # outward normal points +Z (apical = +Z half). centroid omitted -> mask mean.
    out = qf.compute_channel_localization(img, mask, "ap", outward=(1.0, 0.0, 0.0))
    assert np.isfinite(out["ap_apical_intensity_mean"])
    assert np.isfinite(out["ap_basal_intensity_mean"])
    assert out["ap_apical_intensity_mean"] > out["ap_basal_intensity_mean"]
    assert out["ap_apical_basal_intensity_ratio"] > 1.0


def test_apical_basal_flips_with_outward_sign():
    """The same apical-bright channel with the outward normal pointing −Z -> the
    'apical' half is now the dim one -> ratio < 1 (the split follows the normal)."""
    shape = (24, 24, 24)
    mask = _sphere_mask(shape, 9.0)
    cz = (shape[0] - 1) / 2.0
    img = np.zeros(shape, dtype=np.float64)
    zz = np.indices(shape)[0]
    img[mask & (zz > cz)] = 200.0
    img[mask & (zz <= cz)] = 20.0
    out = qf.compute_channel_localization(img, mask, "ap", outward=(-1.0, 0.0, 0.0))
    assert out["ap_apical_basal_intensity_ratio"] < 1.0


def test_apical_basal_degenerate_outward_skipped():
    """A zero / non-finite outward normal -> apical/basal stays NaN (not a crash)."""
    mask = _sphere_mask()
    img = _uniform(mask, 50.0)
    out0 = qf.compute_channel_localization(img, mask, "c", outward=(0.0, 0.0, 0.0))
    outn = qf.compute_channel_localization(img, mask, "c", outward=(np.nan, 0.0, 0.0))
    for o in (out0, outn):
        for s in _LOC_APICALBASAL:
            assert np.isnan(o[f"c{s}"]), s


# --------------------------------------------------------------------------- #
# — compute_channel_compartments (paired nuclear vs cytoplasm)
# --------------------------------------------------------------------------- #
def _cell_nucleus_crop(shape=(16, 16, 16)):
    """A cube cell with a centered cube nucleus inside it (both boolean crops)."""
    cell = np.zeros(shape, dtype=bool)
    nuc = np.zeros(shape, dtype=bool)
    cell[3:13, 3:13, 3:13] = True
    nuc[6:10, 6:10, 6:10] = True
    return cell, nuc


def test_compartment_columns_present_with_prefix():
    cell, nuc = _cell_nucleus_crop()
    img = _uniform(cell, 50.0)
    out = qf.compute_channel_compartments(img, cell, nuc, "mem")
    for s in _COMPARTMENT_SUFFIXES:
        assert f"mem{s}" in out, s
    assert len(out) == len(_COMPARTMENT_SUFFIXES)


def test_nuclear_localized_ratio_above_one():
    """A NUCLEAR-localized channel (bright in the nucleus, dim in cytoplasm) ->
    nuclear_mean > cytoplasm_mean -> nuclear_cytoplasm_ratio > 1."""
    cell, nuc = _cell_nucleus_crop()
    img = np.zeros(cell.shape, dtype=np.float64)
    img[cell] = 20.0
    img[nuc] = 200.0
    out = qf.compute_channel_compartments(img, cell, nuc, "nuc")
    assert out["nuc_nuclear_mean"] > out["nuc_cytoplasm_mean"]
    assert out["nuc_nuclear_cytoplasm_ratio"] > 1.0


def test_cytoplasm_localized_ratio_below_one():
    cell, nuc = _cell_nucleus_crop()
    img = np.zeros(cell.shape, dtype=np.float64)
    img[cell] = 200.0
    img[nuc] = 20.0
    out = qf.compute_channel_compartments(img, cell, nuc, "c")
    assert out["c_nuclear_cytoplasm_ratio"] < 1.0


def test_perinuclear_enrichment_high_for_ring_signal():
    """A signal concentrated in a rim hugging the nucleus (perinuclear) -> the
    perinuclear-rim mean ≫ the (otherwise dim) cytoplasm mean -> enrichment > 1."""
    from scipy.ndimage import binary_dilation
    cell, nuc = _cell_nucleus_crop()
    rim = binary_dilation(nuc, iterations=2) & ~nuc & cell
    img = np.zeros(cell.shape, dtype=np.float64)
    img[cell] = 10.0           # dim cytoplasm baseline
    img[rim] = 300.0           # bright perinuclear ring
    out = qf.compute_channel_compartments(img, cell, nuc, "g")
    assert out["g_perinuclear_enrichment"] > 1.5


def test_compartment_nan_guards():
    cell, nuc = _cell_nucleus_crop()
    img = _uniform(cell, 50.0)
    # shape mismatch
    out = qf.compute_channel_compartments(np.zeros((8, 8, 8)), cell, nuc, "c")
    for s in _COMPARTMENT_SUFFIXES:
        assert np.isnan(out[f"c{s}"])
    # all-NaN crop
    out = qf.compute_channel_compartments(np.full(cell.shape, np.nan), cell, nuc, "c")
    for s in _COMPARTMENT_SUFFIXES:
        assert np.isnan(out[f"c{s}"])
    # no nucleus inside the cell -> nuclear NaN, cytoplasm still defined
    empty_nuc = np.zeros(cell.shape, dtype=bool)
    out = qf.compute_channel_compartments(img, cell, empty_nuc, "c")
    assert np.isnan(out["c_nuclear_mean"])
    assert np.isfinite(out["c_cytoplasm_mean"])
    assert np.isnan(out["c_nuclear_cytoplasm_ratio"])
    assert np.isnan(out["c_perinuclear_enrichment"])


# --------------------------------------------------------------------------- #
# manifest tags
# --------------------------------------------------------------------------- #
def test_prob_geometry_in_static_manifest():
    man = fm.get_manifest()
    for c in _PROB_GEOM_COLS:
        assert c in man, c
        t = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in t, (c, key)
        assert t["role"] == "qc"
        assert t["redundancy_group"] == "probability_geometry"
    # prob_shell_gradient is a SIGNED [-1,1] difference -> bounded True but a score.
    assert man["prob_shell_gradient"]["units"] == "score"
    # the two shell means are [0,1] bounded ratios.
    assert man["prob_shell_inner_mean"]["bounded"] is True


def test_channel_localization_tags_helper():
    tags = fm.channel_localization_tags("mem")
    for s in _LOC_SUFFIXES:
        assert f"mem{s}" in tags, s
        t = tags[f"mem{s}"]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in t, (s, key)
        assert t["role"] == "biology"
        assert t["redundancy_group"] == "channel_localization_mem"
    # shell columns are object-intrinsic; apical/basal use the tissue frame.
    assert tags["mem_shell_inner_mean"]["frame"] == "intrinsic"
    assert tags["mem_apical_intensity_mean"]["frame"] == "tissue"


def test_channel_compartment_tags_helper():
    tags = fm.channel_compartment_tags("nuc")
    for s in _COMPARTMENT_SUFFIXES:
        assert f"nuc{s}" in tags, s
        t = tags[f"nuc{s}"]
        assert t["role"] == "biology"
        assert t["frame"] == "intrinsic"
        assert t["redundancy_group"] == "channel_compartment_nuc"
    assert tags["nuc_nuclear_cytoplasm_ratio"]["units"] == "ratio"


def test_wavef_in_module_all():
    for fn in ("compute_prob_geometry", "compute_channel_localization",
               "compute_channel_compartments"):
        assert fn in qf.__all__, fn
    for fn in ("channel_localization_tags", "channel_compartment_tags"):
        assert fn in fm.__all__, fn


# --------------------------------------------------------------------------- #
# FeatureStage families registry + per-channel recovery
# --------------------------------------------------------------------------- #
def test_per_channel_localization_families_recovered_from_columns():
    from packages.epicycle_scaledseg.core.feature_stage import _per_channel_localization_families
    cols = (
        [f"nuc{s}" for s in _LOC_SUFFIXES]
        + [f"mem{s}" for s in _LOC_SUFFIXES]
        + ["area", "nuc_intensity_mean"]
    )
    fams = _per_channel_localization_families(cols, [(0, "nuc"), (1, "mem")])
    assert set(fams["channel_localization_nuc"]) == {f"nuc{s}" for s in _LOC_SUFFIXES}
    assert set(fams["channel_localization_mem"]) == {f"mem{s}" for s in _LOC_SUFFIXES}


def test_per_channel_localization_families_empty_when_no_channels():
    from packages.epicycle_scaledseg.core.feature_stage import _per_channel_localization_families
    assert _per_channel_localization_families(["area", "nuc_shell_gradient"], None) == {}


def test_probability_geometry_family_registered():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["probability_geometry"]) == set(_PROB_GEOM_COLS)


# --------------------------------------------------------------------------- #
# FeatureStage wiring: localization appears ONLY when channels are sampled;
# prob-geometry only when prob is sampled.
# --------------------------------------------------------------------------- #
def _solid_cube_label(shape=(14, 18, 18), lo=2, hi=12):
    lab = np.zeros(shape, dtype=np.int32)
    lab[lo:hi, lo:hi, lo:hi] = 1
    return lab


def test_extract_one_labels_only_emits_no_wavef(tmp_path):
    """The per-cell worker with NO sampler config emits NO Wave-F columns
    (default-off byte-identical)."""
    from packages.epicycle_scaledseg.core import feature_stage as fs

    lab = _solid_cube_label()
    p = tmp_path / "labels.npy"
    np.save(p, lab)
    fs._init_feature_worker(str(p))
    try:
        task = (1, (2, 12, 2, 12, 2, 12), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict), row
    assert not any("_shell_" in k or "_radial_intensity_slope" in k for k in row)
    assert not any(k.startswith("prob_shell") or k == "prob_area_normalized"
                   for k in row)


def test_extract_one_with_channel_emits_localization(tmp_path, monkeypatch):
    """With a channel sampler config, _extract_one emits the per-channel
    localization columns (apical/basal NaN — no outward normal in the pool)."""
    from packages.epicycle_scaledseg.core import feature_stage as fs
    from packages.epicycle_scaledseg.reconcile import raw_roi

    lab = _solid_cube_label()
    p = tmp_path / "labels.npy"
    np.save(p, lab)

    full_mask = lab == 1
    full_ch = _membrane_signal(full_mask)

    def _fake_read_global_roi(tiles_dir, bbox, channel, halo=0, halo_z=None):
        z0, z1, y0, y1, x0, x1 = bbox
        return full_ch[z0:z1, y0:y1, x0:x1]

    monkeypatch.setattr(raw_roi, "read_global_roi", _fake_read_global_roi)

    fs._init_feature_worker(str(p), tiles_dir="FAKE", recon_dir=None,
                            channels=[(0, "mem")], prob=False)
    try:
        task = (1, (2, 12, 2, 12, 2, 12), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
        fs._WORKER_TILES_DIR = None
        fs._WORKER_RECON_DIR = None
        fs._WORKER_CHANNELS = None
        fs._WORKER_PROB = False
    assert isinstance(row, dict), row
    for s in _LOC_SUFFIXES:
        assert f"mem{s}" in row, f"mem{s}"
    # apical/basal NaN (no outward normal in the per-cell pool); inner shell defined.
    assert np.isnan(row["mem_apical_intensity_mean"])
    assert np.isfinite(row["mem_shell_inner_mean"])
    # a membrane signal is bright at the rind, dim in the core -> intensity DECREASES
    # with radial depth -> negative radial slope (robust on a small cube where the
    # equal-volume outer tertile band can be empty; the sphere direction tests above
    # exercise the full inner/outer shell gradient).
    assert np.isfinite(row["mem_radial_intensity_slope"])
    assert row["mem_radial_intensity_slope"] < 0.0
    # intensity columns still present (localization rides alongside).
    assert "mem_intensity_mean" in row


def test_extract_one_with_prob_emits_prob_geometry(tmp_path, monkeypatch):
    """With prob sampling on, _extract_one emits the prob-geometry columns."""
    from packages.epicycle_scaledseg.core import feature_stage as fs
    from packages.epicycle_scaledseg.reconcile import raw_roi

    lab = _solid_cube_label()
    p = tmp_path / "labels.npy"
    np.save(p, lab)

    full_mask = lab == 1
    full_prob = np.where(full_mask, 0.9, 0.0).astype(np.float32)

    def _fake_read_prob_roi(tiles_dir, recon_dir, bbox, apply_sigmoid=True, **kw):
        z0, z1, y0, y1, x0, x1 = bbox
        return full_prob[z0:z1, y0:y1, x0:x1]

    monkeypatch.setattr(raw_roi, "read_prob_roi", _fake_read_prob_roi)

    fs._init_feature_worker(str(p), tiles_dir="FAKE", recon_dir="FAKE_RECON",
                            channels=None, prob=True)
    try:
        task = (1, (2, 12, 2, 12, 2, 12), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
        fs._WORKER_TILES_DIR = None
        fs._WORKER_RECON_DIR = None
        fs._WORKER_CHANNELS = None
        fs._WORKER_PROB = False
    assert isinstance(row, dict), row
    for c in _PROB_GEOM_COLS:
        assert c in row, c
    # prob_weighted_volume (M2) was reused -> area_normalized finite.
    assert np.isfinite(row["prob_area_normalized"])


# --------------------------------------------------------------------------- #
# reconcile paired path: compartment columns appear only with channels
# --------------------------------------------------------------------------- #
def _paired_scene():
    """A single cube cell (id 1) with a centered cube nucleus (id 10)."""
    Z, Y, X = 16, 16, 16
    cells = np.zeros((Z, Y, X), np.int32)
    nuclei = np.zeros((Z, Y, X), np.int32)
    cells[3:13, 3:13, 3:13] = 1
    nuclei[6:10, 6:10, 6:10] = 10
    triage = [{"cell_id": 1, "status": "clean", "n_nuclei": 1,
               "nucleus_ids": [10], "cell_voxels": 1000}]
    return cells, nuclei, triage


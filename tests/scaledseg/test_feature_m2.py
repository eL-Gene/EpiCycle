"""M2 feature tests — sampler wiring: cellprob + raw channels (B1, B2, B3).

Pure-function known-shape tests (no tile readers / GPU). Builds a tiny synthetic
intensity volume (a bright nucleus blob in a dimmer cytoplasm) + a synthetic
probability volume, and calls the 3 ported functions DIRECTLY on a single-label
crop + the intensity/prob crops — so we exercise the math without needing a real
tiles dir (the tile readers are integration-tested separately in
``test_reconcile_raw_roi.py``).

Asserts:

* ``compute_boundary_interior_features`` (B1) — a membrane-bright / interior-dark
  cell has ``boundary_interior_ratio > 1``; a uniform cell has ratio ≈ 1.
* ``compute_channel_intensity`` (B2) — per-channel ``<prefix>_intensity_*`` stats
  match the crop's numpy stats; a brighter channel reads higher mean.
* ``compute_probability_features`` (B3) — ``prob_mean`` ∈ [0,1] after sigmoid; a
  confident foreground reads high; the 12 prob_* columns are present + finite.
* The ENGINE PROB caveat: the engine prob is a per-tile LOGIT — applying sigmoid
  maps it to [0,1] and the prob_* stats become well-behaved.
* The manifest tags every new (static) M2 column + the dynamic per-channel tags.
* The FeatureStage wiring emits the M2 columns when a sampler config is set, and
  the labels-only path (no sampler) is byte-identical.
* ``extract_label_features(..., tiles_dir=None)`` is byte-identical to the prior
  labels-only output (the additive-optional contract).
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


_BI_COLS = (
    "boundary_intensity_mean", "interior_intensity_mean",
    "boundary_interior_ratio", "z_intensity_variation",
)
_PROB_COLS = (
    "prob_mean", "prob_p10", "prob_std",
    "prob_boundary_mean", "prob_interior_mean", "prob_boundary_drop",
    "prob_weighted_volume", "prob_soft_hard_ratio",
    "prob_boundary_entropy", "prob_mask_entropy",
    "prob_edge_gradient", "prob_edge_gradient_std",
)
_CH_SUFFIXES = (
    "_intensity_mean", "_intensity_median", "_intensity_std",
    "_intensity_min", "_intensity_max", "_intensity_sum", "_intensity_cv",
    "_boundary_intensity_mean", "_interior_intensity_mean",
    "_boundary_interior_ratio", "_z_intensity_variation",
)


# --------------------------------------------------------------------------- #
# synthetic shapes + crops
# --------------------------------------------------------------------------- #

def _solid_cube(shape=(12, 12, 12), lo=2, hi=10):
    """A single solid cube label (id==1) — leaves a >=2-voxel boundary shell."""
    lab = np.zeros(shape, dtype=np.int32)
    lab[lo:hi, lo:hi, lo:hi] = 1
    return lab


def _membrane_bright_intensity(lab):
    """Intensity that is BRIGHT on the cell boundary, DARK in the interior — the
    correctly-segmented membrane signature (boundary/interior ratio > 1)."""
    from scipy.ndimage import binary_erosion
    mask = lab == 1
    interior = binary_erosion(mask, iterations=2)
    boundary = mask & ~interior
    img = np.zeros(lab.shape, dtype=np.float32)
    img[boundary] = 200.0
    img[interior] = 20.0
    return img


def _uniform_intensity(lab, value=100.0):
    """Flat intensity over the whole cell (boundary/interior ratio ≈ 1)."""
    img = np.zeros(lab.shape, dtype=np.float32)
    img[lab == 1] = value
    return img


def _bright_nucleus_dim_cyto(lab):
    """A bright nucleus blob (center) inside a dimmer cytoplasm — a second channel
    whose mean over the cell is dominated by the dim cytoplasm but whose max is the
    bright nucleus."""
    mask = lab == 1
    img = np.zeros(lab.shape, dtype=np.float32)
    img[mask] = 30.0                                  # dim cytoplasm
    img[5:8, 5:8, 5:8] = 255.0                        # bright nucleus blob (within the cube)
    img[~mask] = 0.0
    return img


def _single_df(lab):
    return qf.extract_features(lab)


# --------------------------------------------------------------------------- #
# B1 — boundary/interior intensity
# --------------------------------------------------------------------------- #

def test_boundary_interior_columns_present_and_finite():
    lab = _solid_cube()
    df = _single_df(lab)
    qf.compute_boundary_interior_features(lab, _membrane_bright_intensity(lab), df)
    for c in _BI_COLS:
        assert c in df.columns, c
        assert np.isfinite(df.iloc[0][c]), (c, df.iloc[0][c])


def test_membrane_bright_cell_ratio_above_one():
    lab = _solid_cube()
    df = _single_df(lab)
    qf.compute_boundary_interior_features(lab, _membrane_bright_intensity(lab), df)
    r = df.iloc[0]
    # Bright membrane (200) + dark interior (20) -> boundary mean >> interior mean.
    assert r["boundary_intensity_mean"] > r["interior_intensity_mean"]
    assert r["boundary_interior_ratio"] > 2.0


def test_uniform_cell_ratio_near_one():
    lab = _solid_cube()
    df = _single_df(lab)
    qf.compute_boundary_interior_features(lab, _uniform_intensity(lab, 100.0), df)
    r = df.iloc[0]
    assert r["boundary_interior_ratio"] == pytest.approx(1.0, abs=0.05)


def test_boundary_interior_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_boundary_interior_features(
        np.zeros((4, 4, 4), np.int32), np.zeros((4, 4, 4), np.float32), df)
    for c in _BI_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# B2 — per-channel intensity
# --------------------------------------------------------------------------- #

def test_channel_intensity_columns_present_with_prefix():
    lab = _solid_cube()
    df = _single_df(lab)
    qf.compute_channel_intensity(lab, df, _bright_nucleus_dim_cyto(lab), "nuc")
    for s in _CH_SUFFIXES:
        assert f"nuc{s}" in df.columns, f"nuc{s}"


def test_channel_intensity_stats_match_numpy():
    lab = _solid_cube()
    ch = _bright_nucleus_dim_cyto(lab)
    df = _single_df(lab)
    qf.compute_channel_intensity(lab, df, ch, "nuc")
    r = df.iloc[0]
    vals = ch[lab == 1]
    assert r["nuc_intensity_mean"] == pytest.approx(float(vals.mean()), rel=1e-5)
    assert r["nuc_intensity_median"] == pytest.approx(float(np.median(vals)), rel=1e-5)
    assert r["nuc_intensity_max"] == pytest.approx(255.0)
    assert r["nuc_intensity_min"] == pytest.approx(30.0)
    assert r["nuc_intensity_sum"] == pytest.approx(float(vals.sum()), rel=1e-5)
    # CV = std/mean for a non-uniform channel is > 0.
    assert r["nuc_intensity_cv"] > 0.0


def test_two_channels_distinct_prefixes_distinct_values():
    lab = _solid_cube()
    df = _single_df(lab)
    nuc = _bright_nucleus_dim_cyto(lab)       # has a 255 blob
    mem = _uniform_intensity(lab, 50.0)       # flat 50
    qf.compute_channel_intensity(lab, df, nuc, "nuc")
    qf.compute_channel_intensity(lab, df, mem, "mem")
    r = df.iloc[0]
    assert r["nuc_intensity_max"] == pytest.approx(255.0)
    assert r["mem_intensity_max"] == pytest.approx(50.0)
    assert r["mem_intensity_cv"] == pytest.approx(0.0, abs=1e-6)   # flat channel
    assert r["nuc_intensity_mean"] != r["mem_intensity_mean"]


def test_channel_intensity_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_channel_intensity(
        np.zeros((4, 4, 4), np.int32), df, np.zeros((4, 4, 4), np.float32), "nuc")
    for s in _CH_SUFFIXES:
        assert f"nuc{s}" in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# B3 — probability family (engine prob = LOGIT -> sigmoid)
# --------------------------------------------------------------------------- #

def _logit_prob_for(lab, fg_logit=4.0, bg_logit=-4.0):
    """A synthetic per-tile LOGIT volume: high (+) inside the cell, low (-) outside.
    Mirrors the engine's pre-sigmoid probability."""
    logit = np.full(lab.shape, bg_logit, dtype=np.float32)
    logit[lab == 1] = fg_logit
    return logit


def test_probability_columns_present_and_finite():
    lab = _solid_cube()
    df = _single_df(lab)
    from scipy.special import expit
    prob = expit(_logit_prob_for(lab)).astype(np.float32)     # sigmoid -> [0,1]
    qf.compute_probability_features(lab, df, prob)
    for c in _PROB_COLS:
        assert c in df.columns, c
        assert np.isfinite(df.iloc[0][c]), (c, df.iloc[0][c])


def test_prob_mean_in_unit_interval_after_sigmoid():
    lab = _solid_cube()
    df = _single_df(lab)
    from scipy.special import expit
    prob = expit(_logit_prob_for(lab, fg_logit=4.0)).astype(np.float32)
    qf.compute_probability_features(lab, df, prob)
    r = df.iloc[0]
    for c in ("prob_mean", "prob_p10", "prob_boundary_mean", "prob_interior_mean"):
        assert 0.0 <= r[c] <= 1.0, (c, r[c])
    # sigmoid(4) ≈ 0.982 -> a confident foreground reads high.
    assert r["prob_mean"] > 0.9


def test_prob_confident_vs_uncertain_foreground():
    lab = _solid_cube()
    from scipy.special import expit
    df_hi = _single_df(lab)
    qf.compute_probability_features(lab, df_hi, expit(_logit_prob_for(lab, 6.0)).astype(np.float32))
    df_lo = _single_df(lab)
    qf.compute_probability_features(lab, df_lo, expit(_logit_prob_for(lab, 0.2)).astype(np.float32))
    # A more confident foreground reads a higher prob_mean and lower entropy.
    assert df_hi.iloc[0]["prob_mean"] > df_lo.iloc[0]["prob_mean"]
    assert df_hi.iloc[0]["prob_mask_entropy"] < df_lo.iloc[0]["prob_mask_entropy"]


def test_prob_soft_hard_ratio_positive():
    lab = _solid_cube()
    df = _single_df(lab)
    from scipy.special import expit
    prob = expit(_logit_prob_for(lab)).astype(np.float32)
    qf.compute_probability_features(lab, df, prob)
    r = df.iloc[0]
    assert r["prob_weighted_volume"] > 0.0
    assert r["prob_soft_hard_ratio"] > 0.0


def test_prob_float16_logit_handled():
    """The engine prob is float16 on disk; compute_probability_features casts it.
    Pass a sigmoid'd float16 volume and confirm no crash + sane prob_mean."""
    lab = _solid_cube()
    df = _single_df(lab)
    from scipy.special import expit
    prob = expit(_logit_prob_for(lab)).astype(np.float16)
    qf.compute_probability_features(lab, df, prob)
    assert 0.0 <= df.iloc[0]["prob_mean"] <= 1.0


def test_probability_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_probability_features(
        np.zeros((4, 4, 4), np.int32), df, np.zeros((4, 4, 4), np.float32))
    for c in _PROB_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def test_manifest_tags_static_m2_features():
    man = fm.get_manifest()
    # Only the B3 probability family is statically tagged. The UN-PREFIXED B1
    # boundary/interior columns are NOT in the manifest by design (the function
    # that would emit them is never wired into the per-cell flow; the B1 signal
    # arrives PREFIXED via compute_channel_intensity -> channel_intensity_tags).
    for c in _PROB_COLS:
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] in ("image", "tissue", "intrinsic")
        assert isinstance(tags["bounded"], bool)


def test_manifest_omits_dead_unprefixed_boundary_interior():
    """The un-prefixed B1 boundary_interior columns are DELIBERATELY absent from the
    manifest (dead family — compute_boundary_interior_features is never wired into
    the per-cell flow). The prefixed B1 columns ARE tagged via channel_intensity_tags."""
    man = fm.get_manifest()
    for c in _BI_COLS:
        assert c not in man, f"dead un-prefixed {c} should not be tagged"
    # the prefixed B1 ratio IS tagged for a sampled channel.
    assert "mem_boundary_interior_ratio" in fm.channel_intensity_tags("mem")


def test_manifest_prob_bounded_flags():
    man = fm.get_manifest()
    # Confidence stats are bounded [0,1]; weighted_volume / soft_hard_ratio /
    # gradients are unbounded.
    for c in ("prob_mean", "prob_p10", "prob_boundary_mean", "prob_interior_mean",
              "prob_mask_entropy", "prob_boundary_entropy"):
        assert man[c]["bounded"] is True, c
    for c in ("prob_weighted_volume", "prob_soft_hard_ratio",
              "prob_edge_gradient", "prob_edge_gradient_std"):
        assert man[c]["bounded"] is False, c


# FIX 6 — the RAW absolute intensity suffixes are role="covariate" (a staining /
# illumination confound); the SELF-NORMALIZING ratios + _intensity_std stay
# role="biology". Keyed by suffix so the per-suffix split is pinned.
_RAW_INTENSITY_COVARIATE_SUFFIXES = frozenset({
    "_intensity_mean", "_intensity_median", "_intensity_min", "_intensity_max",
    "_intensity_sum", "_boundary_intensity_mean", "_interior_intensity_mean",
    "_z_intensity_variation",
})
_INTENSITY_BIOLOGY_SUFFIXES = frozenset({
    "_intensity_std", "_intensity_cv", "_boundary_interior_ratio",
})


def test_channel_intensity_tags_helper():
    tags = fm.channel_intensity_tags("nuc")
    for s in _CH_SUFFIXES:
        assert f"nuc{s}" in tags, f"nuc{s}"
        t = tags[f"nuc{s}"]
        if s in _RAW_INTENSITY_COVARIATE_SUFFIXES:
            assert t["role"] == "covariate", s   # raw absolute intensity = confound
        elif s in _INTENSITY_BIOLOGY_SUFFIXES:
            assert t["role"] == "biology", s     # self-normalizing ratio / std
        else:
            raise AssertionError(f"un-classified intensity suffix {s!r}")
        assert t["frame"] == "image"
        assert t["redundancy_group"] == "channel_intensity_nuc"


def test_channel_intensity_role_split_covers_all_suffixes():
    """Guard the guard: every emitted intensity suffix is classified covariate XOR biology."""
    tags = fm.channel_intensity_tags("nuc")
    suffixes = {k[len("nuc"):] for k in tags}
    assert suffixes == (_RAW_INTENSITY_COVARIATE_SUFFIXES | _INTENSITY_BIOLOGY_SUFFIXES), suffixes


# --------------------------------------------------------------------------- #
# FeatureStage families registry
# --------------------------------------------------------------------------- #

def test_feature_stage_families_registry_includes_m2():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["probability"]) == set(_PROB_COLS)
    # the un-prefixed B1 "boundary_interior" family was dropped (dead — never
    # produced by the per-cell flow; the B1 signal comes PREFIXED via
    # compute_channel_intensity -> the dynamic channel_intensity_<prefix> families).
    assert "boundary_interior" not in _FEATURE_FAMILIES


def test_per_channel_families_recovered_from_columns():
    from packages.epicycle_scaledseg.core.feature_stage import _per_channel_families
    cols = [f"nuc{s}" for s in _CH_SUFFIXES] + [f"mem{s}" for s in _CH_SUFFIXES] + ["area"]
    fams = _per_channel_families(cols, [(0, "nuc"), (1, "mem")])
    assert set(fams["channel_intensity_nuc"]) == {f"nuc{s}" for s in _CH_SUFFIXES}
    assert set(fams["channel_intensity_mem"]) == {f"mem{s}" for s in _CH_SUFFIXES}


def test_per_channel_families_empty_when_no_channels():
    from packages.epicycle_scaledseg.core.feature_stage import _per_channel_families
    assert _per_channel_families(["area", "nuc_intensity_mean"], None) == {}


# --------------------------------------------------------------------------- #
# Additive-optional contract: labels-only path byte-identical
# --------------------------------------------------------------------------- #

def _toy_labels():
    vol = np.zeros((8, 16, 16), np.int32)
    vol[1:5, 1:6, 1:6] = 1
    vol[1:5, 9:15, 9:15] = 2
    return vol


def test_extract_label_features_labels_only_unchanged(tmp_path):
    """``extract_label_features(..., tiles_dir=None, prob=False)`` is byte-identical
    to a labels-only call WITHOUT the new params (the M2 additive-optional contract)."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df_baseline = extract_label_features(str(p), n_workers=1)            # no M2 args
    df_default = extract_label_features(                                 # M2 args explicitly off
        str(p), n_workers=1, tiles_dir=None, recon_dir=None, channels=None, prob=False)

    # Same columns, same order, same values — no M2 columns leaked in.
    assert list(df_baseline.columns) == list(df_default.columns)
    pd.testing.assert_frame_equal(df_baseline, df_default)
    # And NO M2 columns are present in the labels-only output.
    for c in (*_BI_COLS, *_PROB_COLS):
        assert c not in df_baseline.columns, c
    assert not any(col.startswith(("nuc_", "mem_")) for col in df_baseline.columns)


def test_extract_one_labels_only_emits_no_m2_columns():
    """The per-cell worker, with NO sampler config, emits exactly the pre-M2
    columns (no boundary/interior, no prob, no per-channel)."""
    from packages.epicycle_scaledseg.core import feature_stage as fs

    lab = _solid_cube((12, 16, 16))
    import tempfile, os
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "labels.npy")
    np.save(p, lab)
    fs._init_feature_worker(p)                       # no M2 args
    try:
        task = (1, (2, 10, 2, 10, 2, 10), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict), row
    for c in (*_BI_COLS, *_PROB_COLS):
        assert c not in row, c


def test_extract_one_with_channel_config_emits_b2_columns(monkeypatch):
    """With a channel sampler config set, ``_extract_one`` emits the per-channel
    (B2) columns — exercised WITHOUT a real tiles dir by monkeypatching the reader
    to return a synthetic crop at the requested bbox."""
    from packages.epicycle_scaledseg.core import feature_stage as fs
    from packages.epicycle_scaledseg.reconcile import raw_roi

    lab = _solid_cube((12, 16, 16))
    import tempfile, os
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "labels.npy")
    np.save(p, lab)

    # A fake raw reader: return a bright-nucleus channel crop at the bbox shape.
    full_ch = _bright_nucleus_dim_cyto(lab)

    def _fake_read_global_roi(tiles_dir, bbox, channel, halo=0, halo_z=None):
        z0, z1, y0, y1, x0, x1 = bbox
        return full_ch[z0:z1, y0:y1, x0:x1]

    monkeypatch.setattr(raw_roi, "read_global_roi", _fake_read_global_roi)

    fs._init_feature_worker(
        p, tiles_dir="FAKE_TILES", recon_dir=None,
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
    for s in _CH_SUFFIXES:
        assert f"nuc{s}" in row, f"nuc{s}"
    # The synthetic channel max (255 nucleus) reaches the worker.
    assert row["nuc_intensity_max"] == pytest.approx(255.0)


def test_extract_one_with_prob_config_emits_b3_columns(monkeypatch):
    """With prob sampling on, ``_extract_one`` emits the prob_* (B3) columns —
    the reader is monkeypatched to return a sigmoid'd crop at the bbox."""
    from packages.epicycle_scaledseg.core import feature_stage as fs
    from packages.epicycle_scaledseg.reconcile import raw_roi
    from scipy.special import expit

    lab = _solid_cube((12, 16, 16))
    import tempfile, os
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "labels.npy")
    np.save(p, lab)

    full_prob = expit(_logit_prob_for(lab)).astype(np.float32)

    def _fake_read_prob_roi(tiles_dir, recon_dir, bbox, halo=0, halo_z=None, apply_sigmoid=False):
        z0, z1, y0, y1, x0, x1 = bbox
        crop = full_prob[z0:z1, y0:y1, x0:x1]
        # the engine reader already applied sigmoid (we pre-applied above) — return as-is
        return crop

    monkeypatch.setattr(raw_roi, "read_prob_roi", _fake_read_prob_roi)

    fs._init_feature_worker(
        p, tiles_dir="FAKE_TILES", recon_dir="FAKE_RECON",
        channels=None, prob=True)
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
    for c in _PROB_COLS:
        assert c in row, c
    assert 0.0 <= row["prob_mean"] <= 1.0

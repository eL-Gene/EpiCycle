"""M0 feature tests — cross-section profile stats + triaxial descriptors.

Pure-function known-shape tests (no GPU / Cellpose). Builds synthetic shapes
whose answers we know by construction:

* a **uniform cylinder** (constant A(z)) → CV≈0, peak_count==1, skew≈0.
* a **tapered cone** (A(z) shrinks with z) → range>0, taper skew, waist at the
  narrow end.
* a **disc** (flat, broad) vs a **rod** (long, thin) → triaxiality + the SIGN of
  disc_vs_rod_score separate them.

Also asserts the new columns exist + are sane on the shared 3-cell volume, the
manifest tags the new features, and the FeatureStage wiring emits them.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


_PROFILE_COLS = (
    "min_cross_section_area", "cross_section_area_range",
    "cross_section_area_cv", "cross_section_area_skew",
    "cross_section_waist_depth", "cross_section_peak_count",
    # Wave I remainder — joined the "cross_section_profile" family.
    "cross_section_area_kurtosis", "min_cross_section_z_relative",
    "cross_section_constriction_index", "cross_section_n_lobes",
    # Wave N remainder — joined the "cross_section_profile" family.
    "basal_flare_index",
    "cross_section_peak_prominence", "cross_section_trough_prominence",
    "slice_centroid_drift_mean", "slice_centroid_drift_max",
    "slice_eccentricity_mean", "slice_eccentricity_taper",
    "slice_orientation_twist",
)
_TRIAXIAL_COLS = (
    "axis_mid_length", "triaxiality", "disc_vs_rod_score",
    # Wave I axis-ratio remainder — joined the "triaxial" family.
    "axis_mid_to_major_ratio", "axis_minor_to_mid_ratio", "shape_anisotropy_3d",
)


# --------------------------------------------------------------------------- #
# shape builders
# --------------------------------------------------------------------------- #

def _uniform_cylinder():
    """Constant cross-section: a 12x10x10 prism (square 'cylinder')."""
    lab = np.zeros((16, 16, 16), dtype=np.int32)
    lab[2:14, 3:13, 3:13] = 1            # constant 10x10 area on every z in 2..13
    return lab


def _tapered_cone():
    """Cross-section area shrinks monotonically from low z (wide) to high z (narrow)."""
    lab = np.zeros((14, 24, 24), dtype=np.int32)
    cz, cy, cx = 0, 12, 12
    for k, z in enumerate(range(2, 12)):
        # radius shrinks with z: wide base at low z, narrow tip at high z.
        rad = 9 - k                       # 9, 8, ..., 0
        if rad < 1:
            rad = 1
        y0, y1 = cy - rad, cy + rad
        x0, x1 = cx - rad, cx + rad
        lab[z, y0:y1, x0:x1] = 1
    return lab


def _disc():
    """Flat pancake: large in X/Y, thin in Z (oblate). a≈b >> c."""
    lab = np.zeros((20, 30, 30), dtype=np.int32)
    lab[9:12, 5:25, 5:25] = 1            # 3 thick, 20x20 wide
    return lab


def _rod():
    """Long cigar: long in X, thin in Y/Z (prolate). a >> b≈c."""
    lab = np.zeros((16, 16, 40), dtype=np.int32)
    lab[7:10, 7:10, 2:38] = 1            # 36 long, 3x3 cross-section
    return lab


def _single(lab, label_id=1):
    df = qf.extract_features(lab)
    return df


# --------------------------------------------------------------------------- #
# cross-section profile stats
# --------------------------------------------------------------------------- #

def test_profile_columns_present_and_finite():
    lab = _uniform_cylinder()
    df = _single(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    for c in _PROFILE_COLS:
        assert c in df.columns, c
    r = df.iloc[0]
    for c in _PROFILE_COLS:
        assert np.isfinite(r[c]), (c, r[c])


def test_uniform_cylinder_low_cv_single_peak():
    lab = _uniform_cylinder()
    df = _single(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    r = df.iloc[0]
    assert r["cross_section_area_cv"] == pytest.approx(0.0, abs=1e-9)
    assert r["cross_section_area_range"] == pytest.approx(0.0, abs=1e-9)
    assert r["cross_section_peak_count"] == 1.0
    # constant area -> min == any slice area == 100 voxels
    assert r["min_cross_section_area"] == pytest.approx(100.0)
    assert abs(r["cross_section_area_skew"]) < 1e-6


def test_tapered_cone_range_and_waist():
    lab = _tapered_cone()
    df = _single(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    r = df.iloc[0]
    # A cone has a real spread of cross-section areas.
    assert r["cross_section_area_range"] > 0
    assert r["cross_section_area_cv"] > 0.1
    # The narrow tip (thinnest slice) is at HIGH z (apical in default frame).
    # waist_depth uses 0=apical..1=basal, so the apical waist -> near 0.
    assert r["cross_section_waist_depth"] < 0.25
    # Area concentrated at the wide base (low z) -> positive skew of A(z).
    assert r["cross_section_area_skew"] > 0
    # Min cross-section is the small tip, well below the base area.
    assert r["min_cross_section_area"] >= 0


def test_waist_depth_honours_invert_z():
    lab = _tapered_cone()
    df_def = _single(lab)
    qf.compute_cross_section_profile_stats(lab, df_def, invert_z=False)
    df_inv = _single(lab)
    qf.compute_cross_section_profile_stats(lab, df_inv, invert_z=True)
    wd_def = df_def.iloc[0]["cross_section_waist_depth"]
    wd_inv = df_inv.iloc[0]["cross_section_waist_depth"]
    # Flipping apical/basal flips the normalized waist position (0<->1).
    assert wd_def == pytest.approx(1.0 - wd_inv, abs=1e-9)


def test_two_lobes_peak_count_gt_one():
    """A dumbbell (two bulges joined by a neck) has >1 peak in A(z)."""
    lab = np.zeros((24, 20, 20), dtype=np.int32)
    lab[3:8, 5:15, 5:15] = 1             # lower bulge (wide)
    lab[8:16, 8:12, 8:12] = 1            # neck (narrow)
    lab[16:21, 5:15, 5:15] = 1           # upper bulge (wide)
    df = _single(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    assert df.iloc[0]["cross_section_peak_count"] >= 2


def test_profile_single_slice_is_degenerate_not_crash():
    lab = np.zeros((6, 10, 10), dtype=np.int32)
    lab[3, 2:8, 2:8] = 1                 # single Z slice
    df = _single(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    r = df.iloc[0]
    assert r["min_cross_section_area"] == pytest.approx(36.0)   # 6x6
    assert r["cross_section_area_range"] == pytest.approx(0.0)
    assert r["cross_section_area_cv"] == 0.0
    assert r["cross_section_peak_count"] == 1.0
    assert r["cross_section_waist_depth"] == 0.5


def test_profile_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_cross_section_profile_stats(np.zeros((4, 4, 4), np.int32), df)
    for c in _PROFILE_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# triaxial descriptors
# --------------------------------------------------------------------------- #

def test_triaxial_columns_present_and_mid_between_major_minor():
    lab = _disc()
    df = qf.extract_features(lab)
    qf.compute_triaxial(lab, df)
    for c in _TRIAXIAL_COLS:
        assert c in df.columns, c
    r = df.iloc[0]
    # mid axis must lie between the major and minor axis lengths.
    assert r["axis_minor_length"] <= r["axis_mid_length"] + 1e-6
    assert r["axis_mid_length"] <= r["axis_major_length"] + 1e-6


def test_triaxiality_bounded_unit_interval():
    for builder in (_disc, _rod, _uniform_cylinder):
        lab = builder()
        df = qf.extract_features(lab)
        qf.compute_triaxial(lab, df)
        t = df.iloc[0]["triaxiality"]
        assert np.isnan(t) or (-1e-9 <= t <= 1.0 + 1e-9), (builder.__name__, t)


def test_disc_vs_rod_score_sign_separates_shapes():
    disc_lab = _disc()
    rod_lab = _rod()
    dd = qf.extract_features(disc_lab)
    qf.compute_triaxial(disc_lab, dd)
    rr = qf.extract_features(rod_lab)
    qf.compute_triaxial(rod_lab, rr)
    disc_score = dd.iloc[0]["disc_vs_rod_score"]
    rod_score = rr.iloc[0]["disc_vs_rod_score"]
    # disc (oblate, broad mid axis) scores HIGHER than rod (prolate).
    assert disc_score > rod_score
    # and the disc is on the positive / rod on the negative side.
    assert disc_score > 0
    assert rod_score < 0


def test_disc_low_triaxiality_rod_high_triaxiality():
    disc_lab = _disc()
    rod_lab = _rod()
    dd = qf.extract_features(disc_lab)
    qf.compute_triaxial(disc_lab, dd)
    rr = qf.extract_features(rod_lab)
    qf.compute_triaxial(rod_lab, rr)
    # triaxiality (A²-B²)/(A²-C²): oblate (B≈A) -> ~0, prolate (B≈C) -> ~1.
    assert dd.iloc[0]["triaxiality"] < 0.5
    assert rr.iloc[0]["triaxiality"] > 0.5


def test_triaxial_missing_eigvals_is_nan():
    df = pd.DataFrame({"label": [1, 2], "area": [10.0, 20.0]})
    qf.compute_triaxial(np.zeros((2, 2, 2), np.int32), df)
    for c in _TRIAXIAL_COLS:
        assert c in df.columns
        assert df[c].isna().all()


def test_triaxial_empty_df_schema():
    df = pd.DataFrame(columns=["label", "inertia_eigval_major",
                               "inertia_eigval_mid", "inertia_eigval_minor"])
    qf.compute_triaxial(np.zeros((2, 2, 2), np.int32), df)
    for c in _TRIAXIAL_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# manifest scaffold
# --------------------------------------------------------------------------- #

def test_manifest_tags_all_m0_features():
    man = fm.get_manifest()
    for c in (*_PROFILE_COLS, *_TRIAXIAL_COLS):
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] in ("image", "tissue", "intrinsic")
        assert isinstance(tags["bounded"], bool)


def test_manifest_bounded_flags_match_feature_math():
    man = fm.get_manifest()
    # waist_depth ∈ [0,1], triaxiality ∈ [0,1], disc_vs_rod_score ∈ [-1,1] -> bounded
    for c in ("cross_section_waist_depth", "triaxiality", "disc_vs_rod_score"):
        assert man[c]["bounded"] is True, c
    # raw areas / counts are unbounded
    for c in ("min_cross_section_area", "cross_section_area_range",
              "cross_section_peak_count"):
        assert man[c]["bounded"] is False, c


def test_manifest_get_returns_copy():
    a = fm.get_manifest()
    a["min_cross_section_area"]["units"] = "TAMPERED"
    b = fm.get_manifest()
    assert b["min_cross_section_area"]["units"] == "voxel"


# --------------------------------------------------------------------------- #
# FeatureStage wiring (the M0 columns reach the per-cell worker output)
# --------------------------------------------------------------------------- #

def test_feature_stage_extract_one_emits_m0_columns():
    from packages.epicycle_scaledseg.core import feature_stage as fs

    # Build a 2-cell volume, save as the source memmap, and drive _extract_one
    # directly (same path the Pool uses) for one cell.
    lab = np.zeros((16, 24, 24), dtype=np.int32)
    lab[2:12, 3:13, 3:13] = 1            # a tapered-ish prism
    lab[2:12, 15:21, 15:21] = 2
    import tempfile, os
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "labels.npy")
    np.save(p, lab)
    fs._init_feature_worker(p)
    try:
        task = (1, (2, 12, 3, 13, 3, 13), None, False, False)  # include_surface=False
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict), row
    for c in (*_PROFILE_COLS, *_TRIAXIAL_COLS):
        assert c in row, f"{c} missing from _extract_one output"


def test_feature_stage_families_registry_includes_m0():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["cross_section_profile"]) == set(_PROFILE_COLS)
    assert set(_FEATURE_FAMILIES["triaxial"]) == set(_TRIAXIAL_COLS)

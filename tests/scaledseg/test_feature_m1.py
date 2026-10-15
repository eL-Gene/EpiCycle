"""M1 feature tests — EDT thickness field + geodesic/tortuosity + major-axis
cross-section profile (atlas + ).

Pure-function known-shape tests (no GPU / Cellpose). Builds synthetic shapes
whose answers we know by construction:

* a **thick blob** (cube) → large ``edt_max``; a **thin sheet** (1-voxel-thick)
  → ``edt_max ≈ 0/1``. EDT separates thickness.
* a **straight rod** → ``tortuosity ≈ 1``; a **bent / L-shaped rod** →
  ``tortuosity > 1`` (the in-mask path must follow the bend).
* an **oblique-in-Z rod** → the major-axis (``ma_*``) profile stays uniform while
  the global-Z cross-section is distorted by the tilt (ma_ is tilt-robust).
* degenerate / empty guards return NaN (or the documented degenerate value), not
  a crash.

Also asserts the helper ``_object_edt`` is byte-identical to the historical
transient EDT used inside ``compute_skeleton_features`` (the additive-only
guardrail for the refactor), the manifest tags every new feature, and the
FeatureStage wiring emits all M1 columns.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


_EDT_COLS = (
    "edt_max", "edt_mean", "edt_p90", "edt_cv", "edt_skew",
    # Wave I medial-axis remainder — joined the "edt" family.
    "medial_radius_mean", "medial_radius_cv", "edt_core_fraction",
    # Wave N medial-axis remainder — joined the "edt" family.
    "medial_radius_range",
)
_GEO_COLS = (
    "geodesic_max", "tortuosity",
    # Wave N skeleton main-path / medial-axis geometry.
    "skeleton_main_path_length", "skeleton_main_path_tortuosity",
    "endpoint_distance_ratio", "medial_axis_bend_energy",
    "medial_branch_count",
)
_MA_COLS = (
    "ma_min_cross_section_area", "ma_max_cross_section_area",
    "ma_cross_section_range", "ma_cross_section_taper",
    "ma_cross_section_waist_depth",
    # Wave I remainder — joined the "ma_cross_section" family.
    "ma_cross_section_cv",
    # Wave N remainder — joined the "ma_cross_section" family.
    "ma_max_cross_section_axis_relative", "ma_cross_section_bulge",
    "ma_peak_count", "ma_cross_section_skew",
)
_ALL_M1 = (*_EDT_COLS, *_GEO_COLS, *_MA_COLS)


# --------------------------------------------------------------------------- #
# shape builders
# --------------------------------------------------------------------------- #

def _thick_blob():
    """A solid 14x14x14 cube → a thick interior (large edt_max)."""
    lab = np.zeros((20, 20, 20), dtype=np.int32)
    lab[3:17, 3:17, 3:17] = 1
    return lab


def _thin_sheet():
    """A 1-voxel-thick sheet (broad in X/Y) → no interior depth (edt_max≈0)."""
    lab = np.zeros((10, 24, 24), dtype=np.int32)
    lab[5, 2:22, 2:22] = 1
    return lab


def _straight_rod():
    """A straight 3x3 cross-section rod along X → tortuosity ≈ 1."""
    lab = np.zeros((10, 10, 40), dtype=np.int32)
    lab[4:7, 4:7, 2:38] = 1
    return lab


def _bent_rod():
    """An L-shaped rod (arm along X, arm along Y) → tortuosity > 1."""
    lab = np.zeros((10, 40, 40), dtype=np.int32)
    lab[4:7, 4:7, 4:34] = 1       # arm along X
    lab[4:7, 4:34, 31:34] = 1     # arm along Y (sharing the corner)
    return lab


def _uniform_rod_along_x(shape=(24, 24, 60)):
    """A uniform 5x5 cross-section rod along X (constant cross-section)."""
    lab = np.zeros(shape, dtype=np.int32)
    lab[10:15, 10:15, 6:54] = 1
    return lab


def _waisted_rod_along_x():
    """A rod along X with a genuine WAIST (mid pinch) → ma profile shows a dip."""
    lab = np.zeros((24, 24, 60), dtype=np.int32)
    cy, cz = 12, 12
    for x in range(6, 54):
        d = abs(x - 30)
        r = 2 + (4 if d > 10 else (d // 3))   # radius dips at the center x=30
        lab[cz - r:cz + r + 1, cy - r:cy + r + 1, x] = 1
    return lab


def _single(lab):
    return qf.extract_features(lab)


# --------------------------------------------------------------------------- #
# _object_edt helper — additive-only guardrail (byte-identical to old transient)
# --------------------------------------------------------------------------- #

def test_object_edt_matches_historical_transient():
    """``_object_edt(spacing=None)`` reproduces the pad->edt->strip transient
    that ``compute_skeleton_features`` used inline, byte-for-byte — so the
    skeleton refactor changes NO existing output."""
    from scipy.ndimage import distance_transform_edt

    rng = np.random.RandomState(0)
    for shape in [(12, 10, 10), (8, 8, 8), (5, 20, 7)]:
        m = np.zeros(shape, bool)
        m[1:-1, 1:-1, 1:-1] = True
        m &= rng.rand(*shape) > 0.1
        padded = np.pad(m, 1, mode="constant", constant_values=False)
        dt_old = distance_transform_edt(padded)[1:-1, 1:-1, 1:-1].astype(np.float64)
        dt_new = qf._object_edt(m, spacing=None)
        assert np.array_equal(dt_old, dt_new), shape


def test_object_edt_spacing_scales_with_physical_units():
    """The interior distance is reported in physical units: uniformly coarser
    voxels (spacing 2 on every axis) double the deepest interior distance vs the
    isotropic voxel-unit transform."""
    m = np.zeros((9, 9, 9), bool)
    m[1:8, 1:8, 1:8] = True
    iso = qf._object_edt(m, spacing=None)
    coarse = qf._object_edt(m, spacing=(2.0, 2.0, 2.0))
    assert coarse.max() == pytest.approx(2.0 * iso.max())


def test_skeleton_features_unchanged_after_refactor():
    """Sanity: the skeleton columns are still produced + finite on a rod (the
    refactor only swapped the transient EDT for the helper)."""
    lab = _straight_rod()
    df = _single(lab)
    qf.compute_skeleton_features(lab, df)
    r = df.iloc[0]
    for c in ("skeleton_branches", "skeleton_nodes", "skeleton_length"):
        assert c in df.columns
        assert np.isfinite(r[c])
    assert r["skeleton_length"] > 0


# --------------------------------------------------------------------------- #
# EDT thickness field
# --------------------------------------------------------------------------- #

def test_edt_columns_present_and_finite():
    lab = _thick_blob()
    df = _single(lab)
    qf.compute_edt_features(lab, df)
    for c in _EDT_COLS:
        assert c in df.columns, c
    r = df.iloc[0]
    for c in _EDT_COLS:
        assert np.isfinite(r[c]), (c, r[c])


def test_thick_blob_large_edt_thin_sheet_small():
    blob = _thick_blob()
    sheet = _thin_sheet()
    db = _single(blob); qf.compute_edt_features(blob, db)
    ds = _single(sheet); qf.compute_edt_features(sheet, ds)
    # A 14^3 cube has a deep core (radius ~7); a 1-voxel sheet has none.
    assert db.iloc[0]["edt_max"] > 4.0
    assert ds.iloc[0]["edt_max"] <= 1.0
    assert db.iloc[0]["edt_max"] > ds.iloc[0]["edt_max"]
    assert db.iloc[0]["edt_mean"] > ds.iloc[0]["edt_mean"]


def test_edt_spacing_aware():
    """``spacing`` reports the thickness in physical units: uniformly coarser
    voxels (2 on every axis) double edt_max vs the voxel-unit transform."""
    lab = _thick_blob()
    d_vox = _single(lab); qf.compute_edt_features(lab, d_vox, spacing=None)
    d_phys = _single(lab); qf.compute_edt_features(lab, d_phys, spacing=(2.0, 2.0, 2.0))
    assert d_phys.iloc[0]["edt_max"] == pytest.approx(2.0 * d_vox.iloc[0]["edt_max"])


def test_edt_single_voxel_degenerate_not_crash():
    lab = np.zeros((6, 6, 6), dtype=np.int32)
    lab[3, 3, 3] = 1
    df = _single(lab)
    qf.compute_edt_features(lab, df)
    r = df.iloc[0]
    # A 1-voxel object sits 1 voxel-face from background on every side → its
    # interior EDT is exactly 1.0 everywhere. The distribution-shape stats on a
    # single constant value are 0 (not NaN, not a crash).
    assert r["edt_max"] == pytest.approx(1.0)
    assert r["edt_mean"] == pytest.approx(1.0)
    assert r["edt_p90"] == pytest.approx(1.0)
    assert r["edt_cv"] == pytest.approx(0.0)
    assert r["edt_skew"] == pytest.approx(0.0)


def test_edt_one_voxel_thick_sheet_all_zero_branch():
    """A genuinely 1-voxel-THICK broad sheet: every voxel touches background in
    Z, so the in-Z interior EDT IS 1.0 — but a 1D curve / single-row object
    where every voxel is on the boundary exercises the all-zero-interior guard.
    Here we assert the broad thin sheet's edt_max is small + finite (no crash)."""
    lab = _thin_sheet()
    df = _single(lab)
    qf.compute_edt_features(lab, df)
    r = df.iloc[0]
    assert np.isfinite(r["edt_max"])
    assert r["edt_max"] <= 1.0


def test_edt_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_edt_features(np.zeros((4, 4, 4), np.int32), df)
    for c in _EDT_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# geodesic / tortuosity
# --------------------------------------------------------------------------- #

def test_geodesic_columns_present_and_finite():
    lab = _straight_rod()
    df = _single(lab)
    qf.compute_geodesic_features(lab, df)
    for c in _GEO_COLS:
        assert c in df.columns, c
    r = df.iloc[0]
    assert np.isfinite(r["geodesic_max"])
    assert np.isfinite(r["tortuosity"])


def test_straight_rod_tortuosity_near_one():
    lab = _straight_rod()
    df = _single(lab)
    qf.compute_geodesic_features(lab, df)
    r = df.iloc[0]
    assert r["geodesic_max"] > 20.0          # spans the ~36-long rod
    assert r["tortuosity"] == pytest.approx(1.0, abs=0.15)


def test_bent_rod_tortuosity_above_one():
    straight = _straight_rod()
    bent = _bent_rod()
    ds = _single(straight); qf.compute_geodesic_features(straight, ds)
    db = _single(bent); qf.compute_geodesic_features(bent, db)
    # The L-shape forces the interior path around the corner → longer than the
    # straight-line distance, so tortuosity clearly exceeds 1 and the straight rod.
    assert db.iloc[0]["tortuosity"] > 1.2
    assert db.iloc[0]["tortuosity"] > ds.iloc[0]["tortuosity"]


def test_geodesic_tortuosity_at_least_one():
    """Geodesic path can never be shorter than the straight line → tortuosity≥1."""
    for builder in (_straight_rod, _bent_rod, _uniform_rod_along_x):
        lab = builder()
        df = _single(lab)
        qf.compute_geodesic_features(lab, df)
        t = df.iloc[0]["tortuosity"]
        assert np.isnan(t) or t >= 1.0 - 1e-6, (builder.__name__, t)


def test_geodesic_single_voxel_is_nan():
    lab = np.zeros((6, 6, 6), dtype=np.int32)
    lab[3, 3, 3] = 1
    df = _single(lab)
    qf.compute_geodesic_features(lab, df)
    r = df.iloc[0]
    # No internal path on a 1-voxel object → undefined (NaN), not a crash.
    assert np.isnan(r["geodesic_max"])
    assert np.isnan(r["tortuosity"])


def test_geodesic_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_geodesic_features(np.zeros((4, 4, 4), np.int32), df)
    for c in _GEO_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# major-axis cross-section profile (tilt-robust)
# --------------------------------------------------------------------------- #

def test_ma_columns_present_and_finite():
    lab = _waisted_rod_along_x()
    df = _single(lab)
    qf.compute_ma_cross_section_features(lab, df)
    for c in _MA_COLS:
        assert c in df.columns, c
    r = df.iloc[0]
    for c in _MA_COLS:
        assert np.isfinite(r[c]), (c, r[c])


def test_ma_waist_depth_bounded_unit_interval():
    lab = _waisted_rod_along_x()
    df = _single(lab)
    qf.compute_ma_cross_section_features(lab, df)
    wd = df.iloc[0]["ma_cross_section_waist_depth"]
    assert 0.0 - 1e-9 <= wd <= 1.0 + 1e-9


def test_ma_waisted_rod_has_range_and_min_below_max():
    lab = _waisted_rod_along_x()
    df = _single(lab)
    qf.compute_ma_cross_section_features(lab, df)
    r = df.iloc[0]
    assert r["ma_cross_section_range"] > 0
    assert r["ma_min_cross_section_area"] < r["ma_max_cross_section_area"]
    assert r["ma_cross_section_range"] == pytest.approx(
        r["ma_max_cross_section_area"] - r["ma_min_cross_section_area"], rel=1e-6
    )


def test_ma_stable_under_z_tilt_while_z_based_distorts():
    """A UNIFORM rod, tilted 40° into Z: the major-axis profile stays roughly
    uniform (slabs follow the cell's own axis) while the global-Z cross-section
    CV jumps (the Z slicing is now oblique). This is the whole point of ma_*."""
    from scipy.ndimage import rotate

    base = _uniform_rod_along_x()

    def _feats(lab):
        df = qf.extract_features(lab)
        qf.compute_cross_section_profile_stats(lab, df)   # global-Z
        qf.compute_ma_cross_section_features(lab, df)     # major-axis
        return df.iloc[0]

    aligned = _feats(base)
    rot = rotate(base.astype(float), 40, axes=(0, 2), order=0, reshape=True)
    tilted = _feats((rot > 0.5).astype(np.int32))

    # The rod is genuinely tilted (its major axis moved well off the Z axis).
    assert aligned["major_axis_z_angle"] == pytest.approx(90.0, abs=5.0)
    assert tilted["major_axis_z_angle"] < 70.0

    # Global-Z profile is corrupted by the tilt: CV goes from ~0 to clearly >0.
    assert aligned["cross_section_area_cv"] == pytest.approx(0.0, abs=1e-6)
    assert tilted["cross_section_area_cv"] > 0.1

    # Major-axis profile stays comparably uniform across orientations: the
    # min/max ratio (1.0 == perfectly uniform) stays high in BOTH and does NOT
    # collapse the way the Z-based CV does.
    def _ma_uniformity(r):
        return r["ma_min_cross_section_area"] / max(r["ma_max_cross_section_area"], 1e-9)

    assert _ma_uniformity(aligned) > 0.5
    assert _ma_uniformity(tilted) > 0.5


def test_ma_missing_axis_columns_is_nan():
    df = pd.DataFrame({"label": [1, 2], "area": [10.0, 20.0]})
    qf.compute_ma_cross_section_features(np.zeros((2, 2, 2), np.int32), df)
    for c in _MA_COLS:
        assert c in df.columns
        assert df[c].isna().all()


def test_ma_empty_df_schema():
    df = pd.DataFrame(columns=["label", "major_axis_z", "major_axis_y",
                               "major_axis_x", "centroid-0", "centroid-1",
                               "centroid-2"])
    qf.compute_ma_cross_section_features(np.zeros((2, 2, 2), np.int32), df)
    for c in _MA_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# manifest scaffold
# --------------------------------------------------------------------------- #

def test_manifest_tags_all_m1_features():
    man = fm.get_manifest()
    for c in _ALL_M1:
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] in ("image", "tissue", "intrinsic")
        assert isinstance(tags["bounded"], bool)


def test_manifest_bounded_flags_match_m1_math():
    man = fm.get_manifest()
    # waist_depth ∈ [0,1] → bounded; everything else here is unbounded
    # (edt_max is a length, tortuosity is ≥1 with no upper bound, etc.)
    assert man["ma_cross_section_waist_depth"]["bounded"] is True
    for c in ("edt_max", "edt_mean", "edt_p90", "edt_cv", "edt_skew",
              "geodesic_max", "tortuosity", "ma_min_cross_section_area",
              "ma_max_cross_section_area", "ma_cross_section_range",
              "ma_cross_section_taper"):
        assert man[c]["bounded"] is False, c


# --------------------------------------------------------------------------- #
# FeatureStage wiring (the M1 columns reach the per-cell worker output)
# --------------------------------------------------------------------------- #

def test_feature_stage_extract_one_emits_m1_columns():
    from packages.epicycle_scaledseg.core import feature_stage as fs

    lab = np.zeros((16, 24, 24), dtype=np.int32)
    lab[2:12, 3:13, 3:13] = 1
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
    for c in _ALL_M1:
        assert c in row, f"{c} missing from _extract_one output"


def test_feature_stage_families_registry_includes_m1():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["edt"]) == set(_EDT_COLS)
    assert set(_FEATURE_FAMILIES["geodesic"]) == set(_GEO_COLS)
    assert set(_FEATURE_FAMILIES["ma_cross_section"]) == set(_MA_COLS)

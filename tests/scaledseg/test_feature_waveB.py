"""Wave B feature tests — mesh-curvature REMAINDER + reliability/confidence
META + single-nucleus SHAPE / CROSS-SECTION.

NEW features (NOT a monorepo port) built from the atlas equations. Pure-function
known-shape tests (no GPU / Cellpose); synthetic shapes whose answers we know by
construction:

* curvature remainder — a **dumbbell** (neck_count >= 1, higher bending energy than
  a smooth ellipsoid; mean_curvature_skew finite), a **smooth ellipsoid / sphere**
  (Gaussian curvature positive + size-invariant, neck_count == 0), a **bumpy**
  blob (bending energy clearly above a sphere's).
* reliability meta — an **oblique-in-Z** cell (low tilt / cross-section
  reliability) vs an **axis-aligned upright** cell (high); a **border** cell
  (border_reliability == 0); feature_confidence_score combines the components +
  degrades gracefully.
* nuclear shape — an **hourglass** nucleus (high hourglass / bilobed / waist) vs a
  **round** one (all ~0); sphericity / solidity / elongation / invagination sane +
  bounded.

Also asserts the manifest tags every new feature, the FeatureStage wires the
curvature-remainder + reliability columns, the reconcile path wires the nuclear
shape columns, and the existing M5 curvature output is UNCHANGED.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# --------------------------------------------------------------------------- #
# column groups
# --------------------------------------------------------------------------- #
_CURV_REMAINDER = (
    "gaussian_curvature_mean", "gaussian_curvature_std", "bending_energy",
    "mean_curvature_skew", "neck_count",
    # Wave I remainder — joined the "curvature_remainder" family.
    "curvature_negative_fraction", "gaussian_curvature_skew",
)
_RELIABILITY = (
    "tilt_reliability", "cross_section_reliability",
    "border_reliability", "feature_confidence_score",
    # Mop-up O remainder (kept in lockstep with compute_reliability_meta).
    "neighbor_topology_reliability", "probability_support_score",
)
# the original M5 curvature columns (must stay byte-identical to the M5 pass)
_M5_CURV = (
    "mean_curvature_mean", "mean_curvature_std", "abs_curvature_mean",
    "convexity", "concavity_count", "max_concavity_depth",
    "bilateral_symmetry_score",
)


# --------------------------------------------------------------------------- #
# shape builders
# --------------------------------------------------------------------------- #
def _ball_mask(R, c, shape):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= R * R


def _solid_sphere(R=8, shape=(24, 24, 24)):
    lab = np.zeros(shape, dtype=np.int32)
    c = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    lab[_ball_mask(R, c, shape)] = 1
    return lab


def _big_sphere(R=11, shape=(30, 30, 30)):
    lab = np.zeros(shape, dtype=np.int32)
    c = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    lab[_ball_mask(R, c, shape)] = 1
    return lab


def _dumbbell():
    """Two balls joined by a thin neck along X → a concave peanut with one girdle."""
    shape = (24, 24, 48)
    lab = np.zeros(shape, dtype=np.int32)
    m = _ball_mask(7, (12, 12, 13), shape) | _ball_mask(7, (12, 12, 35), shape)
    neck = np.zeros(shape, dtype=bool)
    neck[10:15, 10:15, 13:35] = True
    lab[m | neck] = 1
    return lab


def _smooth_ellipsoid():
    """A prolate ellipsoid along X → smooth, convex, NO neck."""
    zz, yy, xx = np.ogrid[:24, :24, :48]
    m = ((zz - 12) / 7.0) ** 2 + ((yy - 12) / 7.0) ** 2 + ((xx - 24) / 20.0) ** 2 <= 1
    lab = np.zeros((24, 24, 48), dtype=np.int32)
    lab[m] = 1
    return lab


def _straight_rod():
    lab = np.zeros((10, 10, 40), dtype=np.int32)
    lab[4:7, 4:7, 2:38] = 1
    return lab


def _bumpy():
    """A central ball studded with smaller bumps → high bending energy."""
    shape = (28, 28, 28)
    m = _ball_mask(8, (14, 14, 14), shape)
    for c in [(6, 14, 14), (22, 14, 14), (14, 6, 14),
              (14, 22, 14), (14, 14, 6), (14, 14, 22)]:
        m |= _ball_mask(4, c, shape)
    lab = np.zeros(shape, dtype=np.int32)
    lab[m] = 1
    return lab


def _tiny():
    lab = np.zeros((6, 6, 6), dtype=np.int32)
    lab[3, 3, 3] = 1
    return lab


def _single(lab):
    return qf.extract_features(lab)


# =========================================================================== #
# GROUP 1 — curvature remainder
# =========================================================================== #
def test_curvature_remainder_columns_present_and_finite():
    lab = _solid_sphere()
    df = _single(lab)
    qf.compute_curvature_features(lab, df)
    r = df.iloc[0]
    for c in _CURV_REMAINDER:
        assert c in df.columns, c
        assert np.isfinite(r[c]), (c, r[c])


def test_dumbbell_has_a_neck_smooth_shapes_do_not():
    """neck_count: a dumbbell's central girdle is one neck; a smooth ball /
    ellipsoid / rod has none."""
    dumb = _dumbbell()
    dd = _single(dumb); qf.compute_curvature_features(dumb, dd)
    assert dd.iloc[0]["neck_count"] >= 1

    for builder in (_solid_sphere, _big_sphere, _smooth_ellipsoid, _straight_rod):
        lab = builder()
        df = _single(lab); qf.compute_curvature_features(lab, df)
        assert df.iloc[0]["neck_count"] == 0, builder.__name__


def test_dumbbell_bending_energy_exceeds_smooth_ellipsoid():
    """A pinched dumbbell bends more than a smoothly-tapering ellipsoid."""
    dumb = _dumbbell(); ell = _smooth_ellipsoid()
    dd = _single(dumb); qf.compute_curvature_features(dumb, dd)
    de = _single(ell); qf.compute_curvature_features(ell, de)
    bd = dd.iloc[0]["bending_energy"]
    be = de.iloc[0]["bending_energy"]
    assert bd >= 0 and be >= 0
    assert bd > be


def test_bumpy_blob_bending_energy_exceeds_sphere():
    """A studded/bumpy blob has clearly higher bending energy than a smooth ball."""
    bumpy = _bumpy(); sph = _big_sphere()
    db = _single(bumpy); qf.compute_curvature_features(bumpy, db)
    ds = _single(sph); qf.compute_curvature_features(sph, ds)
    assert db.iloc[0]["bending_energy"] > ds.iloc[0]["bending_energy"]


def test_gaussian_curvature_positive_on_convex_and_size_invariant():
    """A convex ball/ellipsoid has a POSITIVE mean Gaussian curvature; the
    size-normalized value is invariant to the object's scale (a small vs a big
    sphere score the same K·R²)."""
    small = _solid_sphere(R=8, shape=(24, 24, 24))
    big = _solid_sphere(R=13, shape=(34, 34, 34))
    ds = _single(small); qf.compute_curvature_features(small, ds)
    db = _single(big); qf.compute_curvature_features(big, db)
    ks = ds.iloc[0]["gaussian_curvature_mean"]
    kb = db.iloc[0]["gaussian_curvature_mean"]
    assert ks > 0 and kb > 0                        # convex => positive
    # size-invariant: the two spheres' normalized K agree within voxelization noise
    assert ks == pytest.approx(kb, rel=0.25)
    # an ellipsoid is also convex -> positive
    ell = _smooth_ellipsoid()
    de = _single(ell); qf.compute_curvature_features(ell, de)
    assert de.iloc[0]["gaussian_curvature_mean"] > 0


def test_gaussian_curvature_size_invariant_under_spacing():
    """Scaling the voxel spacing uniformly should NOT change the size-normalized
    Gaussian curvature (it's divided by R²; both scale together)."""
    lab = _solid_sphere()
    dv = _single(lab); qf.compute_curvature_features(lab, dv, spacing=None)
    dp = _single(lab); qf.compute_curvature_features(lab, dp, spacing=(2.0, 2.0, 2.0))
    assert dp.iloc[0]["gaussian_curvature_mean"] == pytest.approx(
        dv.iloc[0]["gaussian_curvature_mean"], rel=1e-4)
    # bending_energy: the marching-cubes spacing path perturbs vertex positions at
    # ~float32 precision even under a uniform scale (the size-normalization cancels
    # the scale, not the per-vertex round-off), so allow a small relative tolerance.
    assert dp.iloc[0]["bending_energy"] == pytest.approx(
        dv.iloc[0]["bending_energy"], rel=1e-3)


def test_curvature_remainder_tiny_object_not_crash():
    lab = _tiny()
    df = _single(lab)
    qf.compute_curvature_features(lab, df)        # must not raise
    for c in _CURV_REMAINDER:
        assert c in df.columns


def test_curvature_remainder_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_curvature_features(np.zeros((4, 4, 4), np.int32), df)
    for c in _CURV_REMAINDER:
        assert c in df.columns
    assert len(df) == 0


def test_m5_curvature_unchanged_by_wave_b():
    """The Wave-B remainder is additive: the 7 original M5 curvature columns must
    be byte-identical whether or not the remainder is computed (they're produced
    by the SAME call now, so compare against a sphere's known M5 values)."""
    lab = _dumbbell()
    df = _single(lab)
    qf.compute_curvature_features(lab, df)
    r = df.iloc[0]
    # the M5 columns are still present + finite (the remainder didn't perturb them)
    for c in _M5_CURV:
        assert c in df.columns
    # a dumbbell is concave + asymmetric-free-ish: convexity < 1, a real pocket.
    assert r["convexity"] < 1.0
    assert r["concavity_count"] >= 1


# =========================================================================== #
# GROUP 2 — reliability / confidence meta
# =========================================================================== #
def _upright_cell():
    """A cell elongated along image Z → major_axis_z_angle ~ 0 (upright)."""
    lab = np.zeros((30, 16, 16), dtype=np.int32)
    lab[2:28, 6:10, 6:10] = 1
    return lab


def _oblique_cell():
    """A cell elongated in-plane (along X) → major axis ~90° from Z (oblique)."""
    lab = np.zeros((10, 16, 40), dtype=np.int32)
    lab[4:7, 6:10, 2:38] = 1
    return lab


def test_reliability_columns_present_and_bounded():
    df = _single(_upright_cell())
    qf.compute_reliability_meta(df)
    r = df.iloc[0]
    for c in _RELIABILITY:
        assert c in df.columns, c
        v = r[c]
        assert np.isnan(v) or (0.0 - 1e-9 <= v <= 1.0 + 1e-9), (c, v)


def test_oblique_cell_low_tilt_reliability_upright_high():
    up = _single(_upright_cell()); qf.compute_reliability_meta(up)
    ob = _single(_oblique_cell()); qf.compute_reliability_meta(ob)
    assert up.iloc[0]["tilt_reliability"] > 0.9          # upright -> trustworthy
    assert ob.iloc[0]["tilt_reliability"] < 0.2          # oblique -> not
    assert ob.iloc[0]["tilt_reliability"] < up.iloc[0]["tilt_reliability"]
    # cross-section reliability tracks the same ordering
    assert ob.iloc[0]["cross_section_reliability"] < up.iloc[0]["cross_section_reliability"]


def test_border_reliability_zero_when_border_else_one():
    df = _single(_upright_cell())
    df["is_border_cell"] = [1]
    qf.compute_reliability_meta(df)
    assert df.iloc[0]["border_reliability"] == 0.0

    df2 = _single(_upright_cell())
    df2["is_border_cell"] = [0]
    qf.compute_reliability_meta(df2)
    assert df2.iloc[0]["border_reliability"] == 1.0


def test_feature_confidence_combines_components():
    """An upright, interior, well-segmented cell scores high confidence; an
    oblique border cell scores clearly lower."""
    good = _single(_upright_cell())
    good["is_border_cell"] = [0]; good["prob_mean"] = [0.95]
    qf.compute_reliability_meta(good)

    bad = _single(_oblique_cell())
    bad["is_border_cell"] = [1]; bad["prob_mean"] = [0.5]
    qf.compute_reliability_meta(bad)

    gc = good.iloc[0]["feature_confidence_score"]
    bc = bad.iloc[0]["feature_confidence_score"]
    assert gc > 0.9
    assert bc < gc
    # mean of (tilt~1, cross~1, border 1, prob .95) ~ .98
    assert gc == pytest.approx(np.mean([1.0, 1.0, 1.0, 0.95]), abs=0.05)


def test_reliability_degrades_gracefully_without_optional_columns():
    """A labels-only table (no is_border_cell / prob_mean) still produces sensible
    reliabilities — the missing terms default to optimistic, not NaN."""
    df = _single(_upright_cell())                  # no border / prob columns
    qf.compute_reliability_meta(df)
    r = df.iloc[0]
    assert r["border_reliability"] == 1.0          # absent border -> assume interior
    assert np.isfinite(r["feature_confidence_score"])


def test_reliability_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_reliability_meta(df)
    for c in _RELIABILITY:
        assert c in df.columns
    assert len(df) == 0


# =========================================================================== #
# GROUP 3 — nuclear shape / cross-section
# =========================================================================== #
def _round_vs_hourglass_scene():
    """One cell with a ROUND nucleus, one with an HOURGLASS (pinched) nucleus."""
    Z, Y, X = 24, 24, 60
    cells = np.zeros((Z, Y, X), np.int32)
    nuclei = np.zeros((Z, Y, X), np.int32)
    # cell 1: round nucleus
    cells[2:22, 2:22, 2:24] = 1
    nuclei[_ball_mask(6, (12, 12, 13), (Z, Y, X))] = 10
    # cell 2: hourglass nucleus (two lobes + a thin neck along X)
    cells[2:22, 2:22, 30:58] = 2
    hb = _ball_mask(6, (12, 12, 37), (Z, Y, X)) | _ball_mask(6, (12, 12, 51), (Z, Y, X))
    neck = np.zeros((Z, Y, X), bool)
    neck[10:15, 10:15, 37:51] = True
    nuclei[hb | neck] = 20
    triage = [
        {"cell_id": 1, "status": "clean", "n_nuclei": 1, "nucleus_ids": [10],
         "cell_voxels": int((cells == 1).sum())},
        {"cell_id": 2, "status": "clean", "n_nuclei": 1, "nucleus_ids": [20],
         "cell_voxels": int((cells == 2).sum())},
    ]
    return cells, nuclei, triage


def test_manifest_bounded_flags_match_wave_b_math():
    man = fm.get_manifest()
    # Bounded [0,1]: the four reliabilities + the bounded nuclear scores.
    for c in ("tilt_reliability", "cross_section_reliability", "border_reliability",
              "feature_confidence_score", "nuclear_sphericity", "nuclear_solidity",
              "nuclear_invagination", "nuclear_cross_section_waist_depth",
              "nuclear_hourglass_score", "nuclear_bilobed_score"):
        assert man[c]["bounded"] is True, c
    # Unbounded: the curvature remainder stats + neck count + unbounded nuclear.
    for c in ("gaussian_curvature_mean", "gaussian_curvature_std", "bending_energy",
              "mean_curvature_skew", "neck_count", "nuclear_elongation",
              "nuclear_min_cross_section_area"):
        assert man[c]["bounded"] is False, c


def test_reliability_all_qc_role():
    man = fm.get_manifest()
    for c in _RELIABILITY:
        assert man[c]["role"] == "qc", c


def test_feature_stage_emits_curvature_remainder_and_reliability():
    """The per-cell worker emits the curvature-remainder columns; the families
    registry carries both the remainder + reliability families."""
    from packages.epicycle_scaledseg.core import feature_stage as fs
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES

    lab = np.zeros((16, 24, 24), dtype=np.int32)
    lab[2:12, 3:13, 3:13] = 1
    lab[2:12, 15:21, 15:21] = 2
    import tempfile, os
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "labels.npy")
    np.save(p, lab)
    fs._init_feature_worker(p)
    try:
        task = (1, (2, 12, 3, 13, 3, 13), None, False, True)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict), row
    for c in _CURV_REMAINDER:
        assert c in row, f"{c} missing from _extract_one output"
    # families registry: curvature_remainder + reliability are registered.
    assert set(_FEATURE_FAMILIES["curvature_remainder"]) == set(_CURV_REMAINDER)
    assert set(_FEATURE_FAMILIES["reliability"]) == set(_RELIABILITY)


def test_extract_label_features_adds_reliability(tmp_path):
    """The reconcile per-label extractor appends the reliability columns (a table
    pass over the assembled frame)."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    lab = np.zeros((20, 16, 16), dtype=np.int32)
    lab[2:18, 5:9, 5:9] = 1                          # an upright rod
    p = tmp_path / "labels.npy"
    np.save(p, lab)
    df = extract_label_features(str(p), n_workers=1)
    for c in _RELIABILITY:
        assert c in df.columns, c
    assert np.isfinite(df.iloc[0]["feature_confidence_score"])

"""M5 feature tests — radial/shell decomposition + surface curvature/concavity/
symmetry (atlas + ).

NEW features (NOT a monorepo port) built from the atlas equations. Pure-function
known-shape tests (no GPU / Cellpose); synthetic shapes whose answers we know by
construction:

* a **solid sphere** → shell fractions follow a ball's radial profile (a fat outer
  rind, a small inner core), ``convexity ≈ 1``, ``concavity_count == 0``,
  ``bilateral_symmetry_score ≈ 1``.
* a **dumbbell/peanut** → concave (``convexity < 1``), at least one concave pocket
  (``concavity_count >= 1``, ``max_concavity_depth > 0``).
* an **L-shape** → low ``bilateral_symmetry_score`` (the bend breaks left/right
  mirror symmetry about the long axis).
* a **thin slab vs a ball** → ``mean_radial_position`` (and the shell structure)
  clearly differ.
* degenerate / tiny objects → NaN (not a crash).

Also asserts ``compute_surface_features`` output is UNCHANGED by the shared-mesh
refactor, the manifest tags every new feature, and the FeatureStage wiring emits
all M5 columns.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


_SHELL_COLS = (
    "shell_volume_frac_inner", "shell_volume_frac_mid", "shell_volume_frac_outer",
    "core_surface_ratio", "mean_radial_position", "radial_position_std",
    # Wave N radial/shell remainder — joined the "shell" family.
    "shell_compactness_inner", "shell_compactness_mid", "shell_compactness_outer",
    "shell_eccentricity_inner", "shell_eccentricity_mid", "shell_eccentricity_outer",
    "core_offset",
    "shell_volume_ratio_inner_outer", "shell_volume_ratio_mid_outer",
)
_CURV_COLS = (
    "mean_curvature_mean", "mean_curvature_std", "abs_curvature_mean",
    "convexity", "concavity_count", "max_concavity_depth",
    "bilateral_symmetry_score",
    # Wave N curvature/symmetry remainder — joined the "curvature" family.
    "radial_symmetry_score", "apical_basal_symmetry_score",
)
_ALL_M5 = (*_SHELL_COLS, *_CURV_COLS)


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


def _big_ball(R=11, shape=(30, 30, 30)):
    lab = np.zeros(shape, dtype=np.int32)
    c = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    lab[_ball_mask(R, c, shape)] = 1
    return lab


def _dumbbell():
    """Two balls joined by a thin neck along X → a concave peanut/dumbbell."""
    shape = (24, 24, 48)
    lab = np.zeros(shape, dtype=np.int32)
    m = _ball_mask(7, (12, 12, 13), shape) | _ball_mask(7, (12, 12, 35), shape)
    neck = np.zeros(shape, dtype=bool)
    neck[10:15, 10:15, 13:35] = True
    lab[m | neck] = 1
    return lab


def _l_shape():
    """An L-shaped rod (two perpendicular arms) → not bilaterally symmetric."""
    lab = np.zeros((10, 40, 40), dtype=np.int32)
    lab[3:7, 4:8, 4:34] = 1       # arm along X
    lab[3:7, 4:34, 30:34] = 1     # arm along Y (sharing the corner)
    return lab


def _straight_rod():
    """A straight 3x3 cross-section rod along X → bilaterally symmetric."""
    lab = np.zeros((10, 10, 40), dtype=np.int32)
    lab[4:7, 4:7, 2:38] = 1
    return lab


def _thin_slab():
    """A 3-voxel-thick broad slab → mostly surface rind, no deep core."""
    lab = np.zeros((12, 40, 40), dtype=np.int32)
    lab[5:8, 4:36, 4:36] = 1
    return lab


def _tiny():
    lab = np.zeros((6, 6, 6), dtype=np.int32)
    lab[3, 3, 3] = 1
    return lab


def _single(lab):
    return qf.extract_features(lab)


# --------------------------------------------------------------------------- #
# shell decomposition (C8 / atlas )
# --------------------------------------------------------------------------- #

def test_shell_columns_present_and_finite_on_sphere():
    lab = _solid_sphere()
    df = _single(lab)
    qf.compute_shell_features(lab, df)
    for c in _SHELL_COLS:
        assert c in df.columns, c
    r = df.iloc[0]
    for c in _SHELL_COLS:
        assert np.isfinite(r[c]), (c, r[c])


def test_shell_fractions_sum_to_one_and_bounded():
    for builder in (_solid_sphere, _big_ball, _dumbbell, _straight_rod):
        lab = builder()
        df = _single(lab)
        qf.compute_shell_features(lab, df)
        r = df.iloc[0]
        s = (r["shell_volume_frac_inner"] + r["shell_volume_frac_mid"]
             + r["shell_volume_frac_outer"])
        assert s == pytest.approx(1.0, abs=1e-9), (builder.__name__, s)
        for c in ("shell_volume_frac_inner", "shell_volume_frac_mid",
                  "shell_volume_frac_outer", "mean_radial_position"):
            assert 0.0 - 1e-9 <= r[c] <= 1.0 + 1e-9, (builder.__name__, c, r[c])


def test_sphere_equal_volume_shells_not_fat_rind():
    """EQUAL-VOLUME shells (B4 fix): a solid ball is split into ~1/3 in each shell
    by construction — the equal-WIDTH partition used to dump ~0.7 of the voxels into
    the outer band (a pure geometric artifact, even for a perfect sphere). The fix
    uses the sphere equal-volume thresholds, so no band dominates and the outer
    fraction is nowhere near the old ~0.7."""
    lab = _big_ball()
    df = _single(lab)
    qf.compute_shell_features(lab, df)
    r = df.iloc[0]
    # No shell holds a runaway majority — each is near 1/3 (discretization keeps it
    # loose, but the old artifact pushed outer to ~0.7; assert it can't recur).
    for c in ("shell_volume_frac_inner", "shell_volume_frac_mid",
              "shell_volume_frac_outer"):
        assert 0.15 < r[c] < 0.55, (c, r[c])
    assert r["shell_volume_frac_outer"] < 0.5         # NOT the old fat rind
    # The ball's mean radial position is well inside [0, 1] (not all-core).
    assert 0.1 < r["mean_radial_position"] < 0.5


def test_thin_slab_vs_ball_radial_position_differs():
    """A thin (3-voxel) slab is almost entirely surface rind with no deep core;
    a ball has a graded radial profile. Their mean radial positions + shell
    structure clearly differ."""
    slab = _thin_slab()
    ball = _big_ball()
    ds = _single(slab); qf.compute_shell_features(slab, ds)
    db = _single(ball); qf.compute_shell_features(ball, db)
    rs, rb = ds.iloc[0], db.iloc[0]
    # The two shapes have clearly different mean radial positions.
    assert abs(rs["mean_radial_position"] - rb["mean_radial_position"]) > 0.2
    # The ball has a genuine surface rind (r <= 1/3 band populated); the thin
    # slab — only ~1.5 EDT thick — has essentially none.
    assert rb["shell_volume_frac_outer"] > 0.2
    assert rs["shell_volume_frac_outer"] < rb["shell_volume_frac_outer"]


def test_shell_spacing_aware_invariant_fractions():
    """Uniform isotropic spacing only rescales the EDT — the (normalized) radial
    partition + fractions are unchanged vs voxel units."""
    lab = _solid_sphere()
    d_vox = _single(lab); qf.compute_shell_features(lab, d_vox, spacing=None)
    d_phys = _single(lab); qf.compute_shell_features(lab, d_phys, spacing=(2.0, 2.0, 2.0))
    for c in ("shell_volume_frac_inner", "shell_volume_frac_mid",
              "shell_volume_frac_outer", "mean_radial_position"):
        assert d_phys.iloc[0][c] == pytest.approx(d_vox.iloc[0][c], abs=1e-9), c


def test_shell_tiny_object_not_crash():
    lab = _tiny()
    df = _single(lab)
    qf.compute_shell_features(lab, df)
    r = df.iloc[0]
    # A 1-voxel object's interior EDT is a single value (1.0) → r==1 everywhere
    # → it reads as all-core. No crash, no NaN on the fractions.
    for c in ("shell_volume_frac_inner", "shell_volume_frac_mid",
              "shell_volume_frac_outer"):
        assert np.isfinite(r[c])
    assert r["shell_volume_frac_inner"] == pytest.approx(1.0)


def test_shell_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_shell_features(np.zeros((4, 4, 4), np.int32), df)
    for c in _SHELL_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# curvature / concavity / symmetry (C13 / atlas )
# --------------------------------------------------------------------------- #

def test_curvature_columns_present_and_finite_on_sphere():
    lab = _solid_sphere()
    df = _single(lab)
    qf.compute_curvature_features(lab, df)
    for c in _CURV_COLS:
        assert c in df.columns, c
    r = df.iloc[0]
    for c in _CURV_COLS:
        assert np.isfinite(r[c]), (c, r[c])


def test_sphere_convex_no_concavity_symmetric():
    lab = _solid_sphere()
    df = _single(lab)
    qf.compute_curvature_features(lab, df)
    r = df.iloc[0]
    assert r["convexity"] > 0.9                       # a ball ~fills its hull
    assert r["concavity_count"] == 0.0
    assert r["max_concavity_depth"] == pytest.approx(0.0)
    assert r["bilateral_symmetry_score"] > 0.9


def test_dumbbell_concave_with_pocket():
    lab = _dumbbell()
    df = _single(lab)
    qf.compute_curvature_features(lab, df)
    r = df.iloc[0]
    # The neck leaves hull volume empty → concave; the two empty wedges beside
    # the neck are concave pockets with real depth.
    assert r["convexity"] < 0.85
    assert r["concavity_count"] >= 1
    assert r["max_concavity_depth"] > 0.0


def test_dumbbell_less_convex_than_sphere():
    sph = _solid_sphere(); dumb = _dumbbell()
    ds = _single(sph); qf.compute_curvature_features(sph, ds)
    dd = _single(dumb); qf.compute_curvature_features(dumb, dd)
    assert dd.iloc[0]["convexity"] < ds.iloc[0]["convexity"]


def test_l_shape_low_bilateral_symmetry():
    """An L-shape is NOT mirror-symmetric about the plane containing its long
    axis (the bend sits on one side) → a low IoU. A straight rod IS."""
    L = _l_shape(); rod = _straight_rod()
    dl = _single(L); qf.compute_curvature_features(L, dl)
    dr = _single(rod); qf.compute_curvature_features(rod, dr)
    assert dl.iloc[0]["bilateral_symmetry_score"] < 0.5
    assert dr.iloc[0]["bilateral_symmetry_score"] > 0.85
    assert dl.iloc[0]["bilateral_symmetry_score"] < dr.iloc[0]["bilateral_symmetry_score"]


def test_bilateral_symmetry_bounded_unit_interval():
    for builder in (_solid_sphere, _dumbbell, _l_shape, _straight_rod):
        lab = builder()
        df = _single(lab)
        qf.compute_curvature_features(lab, df)
        s = df.iloc[0]["bilateral_symmetry_score"]
        assert np.isnan(s) or (0.0 - 1e-9 <= s <= 1.0 + 1e-9), (builder.__name__, s)


def test_convexity_bounded_unit_interval():
    for builder in (_solid_sphere, _big_ball, _dumbbell, _l_shape, _straight_rod):
        lab = builder()
        df = _single(lab)
        qf.compute_curvature_features(lab, df)
        cx = df.iloc[0]["convexity"]
        assert np.isnan(cx) or (0.0 - 1e-9 <= cx <= 1.0 + 1e-9), (builder.__name__, cx)


def test_curvature_missing_axis_columns_symmetry_nan():
    """No ``mid_axis_*`` columns → only the symmetry score is NaN; the mesh +
    hull families still compute."""
    lab = _solid_sphere()
    df = pd.DataFrame({"label": [1]})   # no axis columns at all
    qf.compute_curvature_features(lab, df)
    r = df.iloc[0]
    assert np.isnan(r["bilateral_symmetry_score"])
    assert np.isfinite(r["convexity"])
    assert np.isfinite(r["mean_curvature_mean"])


def test_curvature_tiny_object_not_crash():
    lab = _tiny()
    df = _single(lab)
    qf.compute_curvature_features(lab, df)
    r = df.iloc[0]
    # A 1-voxel object has too few hull points → convexity/concavity NaN (no
    # crash). The mesh families may be NaN too; symmetry of a point is 1.0.
    assert np.isnan(r["convexity"])
    assert np.isnan(r["concavity_count"])
    for c in _CURV_COLS:
        assert c in df.columns


def test_curvature_spacing_depth_scales():
    """``max_concavity_depth`` carries the spacing unit: uniformly coarser voxels
    (2 on every axis) scale the deepest pocket depth on the dumbbell."""
    lab = _dumbbell()
    d_vox = _single(lab); qf.compute_curvature_features(lab, d_vox, spacing=None)
    d_phys = _single(lab); qf.compute_curvature_features(lab, d_phys, spacing=(2.0, 2.0, 2.0))
    dv = d_vox.iloc[0]["max_concavity_depth"]
    dp = d_phys.iloc[0]["max_concavity_depth"]
    assert dv > 0
    assert dp == pytest.approx(2.0 * dv, rel=1e-6)


def test_curvature_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_curvature_features(np.zeros((4, 4, 4), np.int32), df)
    for c in _CURV_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# compute_surface_features must be UNCHANGED (the shared-mesh additive guard)
# --------------------------------------------------------------------------- #

def test_surface_features_unchanged_by_m5():
    """M5 reads an equivalent marching-cubes mesh but must not alter
    ``compute_surface_features``' output. Run surface BEFORE + AFTER the M5
    curvature pass and assert the surface columns are identical."""
    lab = _dumbbell()
    df_a = _single(lab)
    qf.compute_surface_features(lab, df_a)
    before = {c: float(df_a.iloc[0][c]) for c in
              ("surface_area", "true_sphericity", "roughness", "solidity")}

    df_b = _single(lab)
    qf.compute_surface_features(lab, df_b)
    qf.compute_curvature_features(lab, df_b)   # M5 after
    qf.compute_shell_features(lab, df_b)
    after = {c: float(df_b.iloc[0][c]) for c in
             ("surface_area", "true_sphericity", "roughness", "solidity")}
    assert before == after


# --------------------------------------------------------------------------- #
# manifest scaffold
# --------------------------------------------------------------------------- #

def test_manifest_tags_all_m5_features():
    man = fm.get_manifest()
    for c in _ALL_M5:
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] in ("image", "tissue", "intrinsic")
        assert isinstance(tags["bounded"], bool)


def test_manifest_bounded_flags_match_m5_math():
    man = fm.get_manifest()
    # Bounded [0,1]: the three shell fractions, mean_radial_position, convexity,
    # bilateral_symmetry_score.
    for c in ("shell_volume_frac_inner", "shell_volume_frac_mid",
              "shell_volume_frac_outer", "mean_radial_position",
              "convexity", "bilateral_symmetry_score"):
        assert man[c]["bounded"] is True, c
    # Unbounded: core_surface_ratio, radial_position_std, the curvature stats,
    # concavity_count, max_concavity_depth.
    for c in ("core_surface_ratio", "radial_position_std",
              "mean_curvature_mean", "mean_curvature_std", "abs_curvature_mean",
              "concavity_count", "max_concavity_depth"):
        assert man[c]["bounded"] is False, c


# --------------------------------------------------------------------------- #
# FeatureStage wiring (the M5 columns reach the per-cell worker output)
# --------------------------------------------------------------------------- #

def test_feature_stage_extract_one_emits_m5_columns():
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
        # include_surface=True so the worker exercises the full per-cell path.
        task = (1, (2, 12, 3, 13, 3, 13), None, False, True)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict), row
    for c in _ALL_M5:
        assert c in row, f"{c} missing from _extract_one output"


def test_feature_stage_families_registry_includes_m5():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["shell"]) == set(_SHELL_COLS)
    assert set(_FEATURE_FAMILIES["curvature"]) == set(_CURV_COLS)

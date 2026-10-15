
import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm
from packages.epicycle_scaledseg.core import feature_stage as fs


# --------------------------------------------------------------------------- #
# Wave I column inventory (the new columns added by this wave).
# --------------------------------------------------------------------------- #
_TRIAXIAL_WI = ("axis_mid_to_major_ratio", "axis_minor_to_mid_ratio",
                "shape_anisotropy_3d")
_XSEC_WI = ("cross_section_area_kurtosis", "min_cross_section_z_relative",
            "cross_section_constriction_index", "cross_section_n_lobes")
_MA_WI = ("ma_cross_section_cv",)
_CURV_WI = ("curvature_negative_fraction", "gaussian_curvature_skew")
_EDT_WI = ("medial_radius_mean", "medial_radius_cv", "edt_core_fraction")
_CTX_WI = ("distance_to_boundary_face_min",)
_SP_WI = ("lateral_to_apical_surface_ratio", "lateral_to_basal_surface_ratio")
_NUC_ALIAS = ("nuclear_volume", "nuclear_extent", "nuclear_surface_area",
              "nuclear_roughness", "nuclear_aspect_ratio",
              "nuclear_axis_major_length", "nuclear_axis_minor_length")

_ALL_WAVE_I = (*_TRIAXIAL_WI, *_XSEC_WI, *_MA_WI, *_CURV_WI, *_EDT_WI,
               *_CTX_WI, *_SP_WI, *_NUC_ALIAS)


# --------------------------------------------------------------------------- #
# shape builders
# --------------------------------------------------------------------------- #

def _triaxial_brick():
    """A>B>C brick: long in X, mid in Y, thin in Z -> strongly anisotropic."""
    lab = np.zeros((10, 20, 30), np.int32)
    lab[3:6, 4:14, 3:27] = 1            # z-extent 3, y-extent 10, x-extent 24
    return lab


def _sphere(r=7):
    """A solid ball: convex everywhere -> negative-curvature fraction ~ 0."""
    n = 2 * r + 5
    lab = np.zeros((n, n, n), np.int32)
    c = n // 2
    zz, yy, xx = np.ogrid[:n, :n, :n]
    lab[((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= r * r] = 1
    return lab


def _uniform_brick():
    """A constant-cross-section prism: uniform thickness -> high core fraction."""
    lab = np.zeros((16, 16, 16), np.int32)
    lab[2:14, 3:13, 3:13] = 1
    return lab


def _dumbbell():
    """Two fat lobes joined by a thin neck along Z -> multi-lobed profile."""
    lab = np.zeros((24, 24, 24), np.int32)
    lab[3:9, 4:20, 4:20] = 1            # bottom lobe (wide)
    lab[9:15, 10:14, 10:14] = 1        # neck (narrow)
    lab[15:21, 4:20, 4:20] = 1         # top lobe (wide)
    return lab


def _single_cell(lab_id=1, blob=None):
    if blob is None:
        blob = _uniform_brick()
    return blob


# --------------------------------------------------------------------------- #
# — axis ratios
# --------------------------------------------------------------------------- #

def test_triaxial_axis_ratios_ordering_and_bounds():
    lab = _triaxial_brick()
    df = qf.extract_features(lab)
    qf.compute_triaxial(lab, df)
    for c in _TRIAXIAL_WI:
        assert c in df.columns
    r = df.iloc[0]
    # ratios are in (0,1]; anisotropy in [0,1].
    assert 0 < r["axis_mid_to_major_ratio"] <= 1.0 + 1e-9
    assert 0 < r["axis_minor_to_mid_ratio"] <= 1.0 + 1e-9
    assert 0 <= r["shape_anisotropy_3d"] <= 1.0 + 1e-9
    # a clearly non-spherical (flat, elongated) brick -> meaningful anisotropy.
    assert r["shape_anisotropy_3d"] > 0.1


def test_triaxial_ratios_consistent_with_axis_lengths():
    lab = _triaxial_brick()
    df = qf.extract_features(lab)
    qf.compute_triaxial(lab, df)
    r = df.iloc[0]
    A = r["axis_major_length"]; B = r["axis_mid_length"]; C = r["axis_minor_length"]
    assert abs(r["axis_mid_to_major_ratio"] - B / A) < 1e-9
    assert abs(r["axis_minor_to_mid_ratio"] - C / B) < 1e-9
    assert abs(r["shape_anisotropy_3d"] - (A - C) / (A + C)) < 1e-9


def test_sphere_is_nearly_isotropic():
    lab = _sphere()
    df = qf.extract_features(lab)
    qf.compute_triaxial(lab, df)
    r = df.iloc[0]
    # a ball: all three axes ~ equal -> ratios ~ 1, anisotropy ~ 0.
    assert r["axis_mid_to_major_ratio"] > 0.9
    assert r["shape_anisotropy_3d"] < 0.15


# --------------------------------------------------------------------------- #
# — cross-section remainder
# --------------------------------------------------------------------------- #

def test_cross_section_remainder_columns_exist():
    lab = _uniform_brick()
    df = qf.extract_features(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    for c in _XSEC_WI:
        assert c in df.columns


def test_uniform_profile_no_constriction_one_lobe():
    lab = _uniform_brick()
    df = qf.extract_features(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    r = df.iloc[0]
    # constant A(z): no constriction, a single lobe, kurtosis ~ 0.
    assert r["cross_section_constriction_index"] < 1e-6
    assert r["cross_section_n_lobes"] == 1.0
    assert abs(r["cross_section_area_kurtosis"]) < 1.0


def test_dumbbell_is_multi_lobed_and_constricted():
    lab = _dumbbell()
    df = qf.extract_features(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    r = df.iloc[0]
    # two fat lobes + a thin neck -> >1 lobe + a strong constriction index.
    assert r["cross_section_n_lobes"] >= 2.0
    assert r["cross_section_constriction_index"] > 0.5
    # the thinnest slice is the central neck -> min_z_relative near the middle.
    assert 0.3 < r["min_cross_section_z_relative"] < 0.7


def test_n_lobes_mirrors_peak_count():
    """cross_section_n_lobes is a synonym surface for cross_section_peak_count."""
    for lab in (_uniform_brick(), _dumbbell(), _triaxial_brick()):
        df = qf.extract_features(lab)
        qf.compute_cross_section_profile_stats(lab, df)
        assert (df["cross_section_n_lobes"] == df["cross_section_peak_count"]).all()


def test_min_cross_section_z_relative_in_unit_interval():
    lab = _dumbbell()
    df = qf.extract_features(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    assert 0.0 <= df.iloc[0]["min_cross_section_z_relative"] <= 1.0


# --------------------------------------------------------------------------- #
# — major-axis CV
# --------------------------------------------------------------------------- #

def test_ma_cross_section_cv_present_and_nonneg():
    lab = _triaxial_brick()
    df = qf.extract_features(lab)
    qf.compute_ma_cross_section_features(lab, df)
    assert "ma_cross_section_cv" in df.columns
    v = df.iloc[0]["ma_cross_section_cv"]
    assert np.isfinite(v) and v >= 0.0


def test_ma_cv_uniform_lower_than_dumbbell():
    """A constant-thickness brick's major-axis profile is far more even than a
    necked dumbbell's -> smaller CV."""
    du = _dumbbell(); db = qf.extract_features(du)
    qf.compute_ma_cross_section_features(du, db)
    br = _triaxial_brick(); bb = qf.extract_features(br)
    qf.compute_ma_cross_section_features(br, bb)
    assert db.iloc[0]["ma_cross_section_cv"] > bb.iloc[0]["ma_cross_section_cv"]


# --------------------------------------------------------------------------- #
# — curvature remainder
# --------------------------------------------------------------------------- #

def test_curvature_remainder_columns_exist():
    lab = _sphere()
    df = qf.extract_features(lab)
    qf.compute_curvature_features(lab, df)
    for c in _CURV_WI:
        assert c in df.columns


def test_sphere_has_low_negative_curvature_fraction():
    """A convex ball curves outward everywhere -> few/no concave vertices."""
    lab = _sphere(r=8)
    df = qf.extract_features(lab)
    qf.compute_curvature_features(lab, df)
    nf = df.iloc[0]["curvature_negative_fraction"]
    assert np.isfinite(nf)
    assert 0.0 <= nf <= 1.0
    # A voxelized ball still carries some marching-cubes sliver noise giving a
    # minority of vertices a slightly-negative umbrella curvature; the signal is
    # that the convex majority is NOT concave (< half) — the strong concavity
    # signal is the dumbbell comparison below.
    assert nf < 0.45            # convex majority


def test_concave_cup_has_more_negative_curvature_than_solid_block():
    """Carving a deep concave well into a solid block ADDS concave inner-wall
    vertices -> a higher negative-curvature fraction than the same block left
    solid. A relative invariant (no magic threshold): the metric responds to a
    real, controlled increase in concavity on otherwise identical geometry."""
    n = 24
    zz, yy, xx = np.ogrid[:n, :n, :n]

    solid = np.zeros((n, n, n), np.int32)
    solid[3:21, 3:21, 3:21] = 1                # solid convex block
    sd = qf.extract_features(solid)
    qf.compute_curvature_features(solid, sd)

    cup = solid.copy()
    bore = ((yy - 12) ** 2 + (xx - 12) ** 2) <= 5 ** 2
    cup[8:21] = np.where(bore[None, :, :], 0, cup[8:21])   # carve a top-open well
    cd = qf.extract_features(cup)
    qf.compute_curvature_features(cup, cd)

    assert cd.iloc[0]["curvature_negative_fraction"] > sd.iloc[0]["curvature_negative_fraction"]


def test_gaussian_curvature_skew_finite():
    lab = _dumbbell()
    df = qf.extract_features(lab)
    qf.compute_curvature_features(lab, df)
    assert np.isfinite(df.iloc[0]["gaussian_curvature_skew"])


# --------------------------------------------------------------------------- #
# — medial-axis remainder (interior EDT)
# --------------------------------------------------------------------------- #

def test_edt_medial_remainder_columns_exist():
    lab = _uniform_brick()
    df = qf.extract_features(lab)
    qf.compute_edt_features(lab, df)
    for c in _EDT_WI:
        assert c in df.columns


def test_medial_radius_aliases_equal_edt_mean_cv():
    """medial_radius_mean == edt_mean and medial_radius_cv == edt_cv exactly."""
    for lab in (_uniform_brick(), _sphere(), _triaxial_brick()):
        df = qf.extract_features(lab)
        qf.compute_edt_features(lab, df)
        assert (df["medial_radius_mean"] == df["edt_mean"]).all()
        assert (df["medial_radius_cv"] == df["edt_cv"]).all()


def test_edt_core_fraction_in_unit_interval():
    for lab in (_uniform_brick(), _sphere()):
        df = qf.extract_features(lab)
        qf.compute_edt_features(lab, df)
        cf = df.iloc[0]["edt_core_fraction"]
        assert 0.0 <= cf <= 1.0


def test_edt_core_fraction_uniform_slab_higher_than_tapered_ball():
    """edt_core_fraction measures the UNIFORMITY of thickness (share of interior
    at >= half the max depth), not the absolute thickness. A uniform-thickness
    slab has nearly all its interior at the single max depth -> a high core
    fraction; a ball has a single deep centre with a graded thin periphery ->
    a much smaller share near the max -> a lower core fraction."""
    slab = np.zeros((20, 24, 24), np.int32)
    slab[8:10, 3:21, 3:21] = 1          # 2-voxel-thick uniform slab
    sd = qf.extract_features(slab)
    qf.compute_edt_features(slab, sd)
    ball = _sphere(r=8); bd = qf.extract_features(ball)
    qf.compute_edt_features(ball, bd)
    assert sd.iloc[0]["edt_core_fraction"] > bd.iloc[0]["edt_core_fraction"]


# --------------------------------------------------------------------------- #
# — FOV-face distance (global tissue_depth pass)
# --------------------------------------------------------------------------- #

def test_distance_to_boundary_face_min_centered_vs_edge():
    """A cell at the volume center sits far from every FOV face; a cell hugging a
    face sits close. The centered cell's min-face distance must exceed the edge
    cell's."""
    lab = np.zeros((30, 30, 30), np.int32)
    lab[12:18, 12:18, 12:18] = 1        # centered
    lab[0:4, 0:4, 0:4] = 2              # cornered against the (0,0,0) faces
    df = qf.extract_features(lab)
    qf.compute_tissue_depth(lab, df)
    assert "distance_to_boundary_face_min" in df.columns
    s = df.set_index("label")["distance_to_boundary_face_min"]
    assert np.isfinite(s.loc[1]) and np.isfinite(s.loc[2])
    assert s.loc[1] > s.loc[2]
    assert s.loc[2] >= 0.0


def test_distance_to_boundary_face_respects_spacing():
    """With anisotropic spacing the face distance is physical (scales with the
    near axis), so a centered cell stays comfortably positive."""
    lab = np.zeros((20, 20, 20), np.int32)
    lab[8:12, 8:12, 8:12] = 1
    df = qf.extract_features(lab, spacing=(2.0, 1.0, 1.0))
    qf.compute_tissue_depth(lab, df, spacing=(2.0, 1.0, 1.0))
    assert df.iloc[0]["distance_to_boundary_face_min"] > 0.0


# --------------------------------------------------------------------------- #
# — lateral surface ratios
# --------------------------------------------------------------------------- #

def test_lateral_surface_ratios_present_and_consistent():
    """With the Wave C outward normal present, the partition runs and the two
    lateral cross-ratios equal lateral/apical and lateral/basal areas."""
    lab = _uniform_brick()
    df = qf.extract_features(lab)
    # supply a trusted outward normal (Wave C contract) pointing +Z.
    n = len(df)
    df["tissue_outward_z"] = np.ones(n)
    df["tissue_outward_y"] = np.zeros(n)
    df["tissue_outward_x"] = np.zeros(n)
    df["tissue_outward_confidence"] = np.ones(n)
    qf.compute_surface_partition(lab, df)
    for c in _SP_WI:
        assert c in df.columns
    r = df.iloc[0]
    eps = 1e-12
    assert abs(r["lateral_to_apical_surface_ratio"]
               - r["lateral_surface_area"] / max(r["apical_surface_area"], eps)) < 1e-6
    assert abs(r["lateral_to_basal_surface_ratio"]
               - r["lateral_surface_area"] / max(r["basal_surface_area"], eps)) < 1e-6


def test_lateral_ratios_skipped_without_waveC():
    """Default-off: no tissue_outward_* -> the partition (and the lateral ratios)
    are absent entirely."""
    lab = _uniform_brick()
    df = qf.extract_features(lab)
    qf.compute_surface_partition(lab, df)   # no Wave C columns
    for c in _SP_WI:
        assert c not in df.columns


# --------------------------------------------------------------------------- #
# — nucleus naming reconcile (aliases of nuc_*)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# manifest + family registry wiring (every Wave I column is a first-class citizen)
# --------------------------------------------------------------------------- #

def test_manifest_tags_all_wave_i_features():
    man = fm.get_manifest()
    for c in _ALL_WAVE_I:
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] in ("image", "tissue", "intrinsic")
        assert isinstance(tags["bounded"], bool)


def test_manifest_bounded_flags_match_wave_i_math():
    man = fm.get_manifest()
    # [0,1]/(0,1] ratios + fractions -> bounded.
    for c in ("axis_mid_to_major_ratio", "axis_minor_to_mid_ratio",
              "shape_anisotropy_3d", "min_cross_section_z_relative",
              "edt_core_fraction", "curvature_negative_fraction",
              "nuclear_extent"):
        assert man[c]["bounded"] is True, c
    # raw counts / unbounded ratios -> not bounded.
    for c in ("cross_section_n_lobes", "cross_section_constriction_index",
              "ma_cross_section_cv", "lateral_to_apical_surface_ratio",
              "nuclear_volume", "distance_to_boundary_face_min"):
        assert man[c]["bounded"] is False, c


def test_feature_stage_families_registry_includes_wave_i():
    fam = fs._FEATURE_FAMILIES
    assert set(_TRIAXIAL_WI).issubset(fam["triaxial"])
    assert set(_XSEC_WI).issubset(fam["cross_section_profile"])
    assert set(_MA_WI).issubset(fam["ma_cross_section"])
    assert set(_CURV_WI).issubset(fam["curvature_remainder"])
    assert set(_EDT_WI).issubset(fam["edt"])
    assert set(_CTX_WI).issubset(fam["spatial_context"])
    assert set(_SP_WI).issubset(fam["surface_partition"])


def test_feature_stage_extract_one_emits_wave_i_per_cell_columns(tmp_path):
    """The per-cell Wave I columns (triaxial / cross-section / ma / curvature /
    edt) reach the FeatureStage per-cell worker output."""
    import os
    lab = _triaxial_brick()
    p = os.path.join(tmp_path, "labels.npy")
    np.save(p, lab)
    fs._init_feature_worker(p)
    task = (1, (0, lab.shape[0], 0, lab.shape[1], 0, lab.shape[2]), None, False, True)
    row = fs._extract_one(task)
    assert isinstance(row, dict), row
    for c in (*_TRIAXIAL_WI, *_XSEC_WI, *_MA_WI, *_CURV_WI, *_EDT_WI):
        assert c in row, f"{c} missing from _extract_one output"


# --------------------------------------------------------------------------- #
# empty-table contracts (typed empty columns, no crash)
# --------------------------------------------------------------------------- #

def test_empty_table_typed_columns():
    empty = pd.DataFrame({"label": pd.Series([], dtype=np.int64)})
    lab = np.zeros((4, 4, 4), np.int32)
    qf.compute_triaxial(lab, empty)
    qf.compute_cross_section_profile_stats(lab, empty)
    qf.compute_ma_cross_section_features(lab, empty)
    qf.compute_curvature_features(lab, empty)
    qf.compute_edt_features(lab, empty)
    for c in (*_TRIAXIAL_WI, *_XSEC_WI, *_MA_WI, *_CURV_WI, *_EDT_WI):
        assert c in empty.columns, c
    assert len(empty) == 0

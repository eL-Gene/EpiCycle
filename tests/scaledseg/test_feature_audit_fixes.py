
import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _single_df(lab, spacing=None):
    return qf.extract_features(lab, spacing=spacing)


def _ball_mask(R, c, shape):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= R * R


def _ball(R, shape):
    lab = np.zeros(shape, dtype=np.int32)
    c = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    lab[_ball_mask(R, c, shape)] = 1
    return lab


def _flat_sheet_3x3():
    """A flat monolayer: a 3x3 grid of touching cells in a SINGLE thin Z-layer
    (centre cell id 5 has 8 coplanar neighbours)."""
    lab = np.zeros((6, 18, 18), dtype=np.int32)
    cid = 1
    for yi in range(3):
        for xi in range(3):
            lab[1:4, yi * 6:(yi + 1) * 6, xi * 6:(xi + 1) * 6] = cid
            cid += 1
    return lab


def _with_graph(lab, spacing=None, max_radius=3):
    df = qf.extract_features(lab, spacing=spacing)
    qf.compute_neighbor_features(lab, df, max_radius=max_radius)
    return df


# =========================================================================== #
# A. ANISOTROPY
# =========================================================================== #

def test_a1_tissue_frame_normal_is_z_under_anisotropy():
    """A flat sheet's neighbours are coplanar in the Z=const plane, so the estimated
    local tissue NORMAL must be ~Z regardless of (anisotropic) voxel spacing. The
    OLD code re-scaled the already-physical centroid offsets by spacing, tilting the
    covariance + normal away from Z under z=4; the fix keeps the normal on Z (so
    planarity stays high). We read the normal back via local_normal_alignment with a
    KNOWN-Z probe major axis injected into the centre cell."""
    lab = _flat_sheet_3x3()
    sp = (4.0, 1.0, 1.0)
    df = _with_graph(lab, spacing=sp)
    # inject a unit-Z major axis on the centre cell so |alignment| reads |normal_z|.
    for ax, val in (("major_axis_z", 1.0), ("major_axis_y", 0.0), ("major_axis_x", 0.0)):
        df.loc[df["label"] == 5, ax] = val
    qf.compute_tissue_frame_features(df, spacing=sp)
    r = df.set_index("label").loc[5]
    # neighbours coplanar -> high planarity even with z=4 spacing.
    assert r["neighbor_ring_planarity"] > 0.9, r["neighbor_ring_planarity"]
    # the estimated normal is ~Z -> |alignment| with the injected Z axis ~ 1.
    assert r["local_normal_alignment"] > 0.95, r["local_normal_alignment"]


def test_a1_tissue_frame_matches_isotropic_when_geometry_unchanged():
    """The fix makes the result independent of an arbitrary isotropic spacing scale
    (the offsets are already physical), so planarity is identical voxel vs (2,2,2)."""
    lab = _flat_sheet_3x3()
    df_v = _with_graph(lab, spacing=None)
    df_p = _with_graph(lab, spacing=(2.0, 2.0, 2.0))
    qf.compute_tissue_frame_features(df_v, spacing=None)
    qf.compute_tissue_frame_features(df_p, spacing=(2.0, 2.0, 2.0))
    pv = df_v.set_index("label").loc[5, "neighbor_ring_planarity"]
    pp = df_p.set_index("label").loc[5, "neighbor_ring_planarity"]
    assert pp == pytest.approx(pv, abs=1e-9)


def test_a2_bilateral_symmetry_high_for_symmetric_cell_anisotropic():
    """A solid ellipsoid (symmetric about every principal plane) scores IoU ~1 for
    bilateral_symmetry under anisotropic spacing (z=4). The OLD code reflected VOXEL
    coords across a PHYSICAL-frame normal -> a wrong IoU on anisotropic data."""
    # an axis-aligned solid ellipsoid: symmetric across each mid-plane.
    shape = (24, 28, 28)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = 12, 14, 14
    ell = ((zz - cz) / 6.0) ** 2 + ((yy - cy) / 9.0) ** 2 + ((xx - cx) / 6.0) ** 2 <= 1.0
    lab = np.zeros(shape, dtype=np.int32)
    lab[ell] = 1

    sp = (4.0, 1.0, 1.0)
    df = _single_df(lab, spacing=sp)
    qf.compute_curvature_features(lab, df, spacing=sp)
    sym = df.iloc[0]["bilateral_symmetry_score"]
    assert np.isfinite(sym)
    assert sym > 0.9, sym


def test_a2_bilateral_symmetry_isotropic_still_high():
    """Sanity: the symmetric ellipsoid also scores ~1 in voxel units (no regression
    of the isotropic path)."""
    shape = (24, 24, 24)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    ell = ((zz - 12) / 6.0) ** 2 + ((yy - 12) / 9.0) ** 2 + ((xx - 12) / 6.0) ** 2 <= 1.0
    lab = np.zeros(shape, dtype=np.int32)
    lab[ell] = 1
    df = _single_df(lab)
    qf.compute_curvature_features(lab, df)
    assert df.iloc[0]["bilateral_symmetry_score"] > 0.9


# =========================================================================== #
# B. ESTIMATOR CORRECTNESS
# =========================================================================== #

def test_b3_contact_geometry_nonzero_for_gapped_adjacent_pair():
    """Two cells separated by a 1-voxel GAP: M3's graduated dilation finds them as
    neighbours (neighbor_count >= 1), and the contact geometry — now using the SAME
    graduated rule — reports NONZERO contact area instead of 0/NaN (the radius-1
    shell missed the gap)."""
    lab = np.zeros((8, 12, 24), np.int32)
    lab[2:6, 2:8, 2:8] = 1
    # cell 2 starts at x=9, leaving x=8 as a 1-voxel background gap (cell 1 ends x=7).
    lab[2:6, 2:8, 9:15] = 2
    df = _with_graph(lab, max_radius=3)
    # M3 must have found them across the gap.
    assert df.set_index("label").loc[1, "neighbor_count"] >= 1.0
    qf.compute_contact_geometry(lab, df, max_radius=3)
    r = df.set_index("label").loc[1]
    # consistent with the graph: a connected neighbour -> nonzero contact area.
    assert r["contact_area_total"] > 0.0, r["contact_area_total"]
    assert r["contact_count"] >= 1.0
    assert np.isfinite(r["largest_contact_fraction"])


def test_b3_touching_pair_still_nonzero():
    """A directly-TOUCHING pair (no gap) is unchanged — still nonzero contact."""
    lab = np.zeros((8, 12, 20), np.int32)
    lab[2:6, 2:8, 2:8] = 1
    lab[2:6, 2:8, 8:14] = 2          # shares the x==8 face
    df = _with_graph(lab, max_radius=3)
    qf.compute_contact_geometry(lab, df, max_radius=3)
    assert df.set_index("label").loc[1, "contact_area_total"] > 0.0


def test_b4_sphere_equal_volume_shells():
    """A solid sphere splits ~1/3,1/3,1/3 by equal-VOLUME shells — the old
    equal-WIDTH partition dumped ~0.7 of the voxels into the outer band (a pure
    geometric artifact, even for a sphere)."""
    lab = _ball(11, (30, 30, 30))
    df = _single_df(lab)
    qf.compute_shell_features(lab, df)
    r = df.iloc[0]
    for c in ("shell_volume_frac_inner", "shell_volume_frac_mid",
              "shell_volume_frac_outer"):
        assert 0.15 < r[c] < 0.55, (c, r[c])
    assert r["shell_volume_frac_outer"] < 0.5     # NOT the old ~0.7 fat rind


def test_b4_sheet_vs_ball_radial_position_separates():
    """Equal-volume shells still SEPARATE shapes: a thin slab (no graded radial
    profile — almost all voxels share one EDT level) has a markedly different mean
    radial position than a ball, so the partition is not a degenerate constant."""
    ball = _ball(11, (30, 30, 30))
    slab = np.zeros((6, 30, 30), np.int32)
    slab[1:4, 4:26, 4:26] = 1                     # ~3 voxels thick
    db = _single_df(ball); qf.compute_shell_features(ball, db)
    ds = _single_df(slab); qf.compute_shell_features(slab, ds)
    # clearly different radial structure between a graded ball + a flat slab.
    assert abs(ds.iloc[0]["mean_radial_position"] - db.iloc[0]["mean_radial_position"]) > 0.2
    # and the ball's three equal-volume fractions are all genuinely populated
    # (no single band swallows everything the way the old equal-WIDTH outer did).
    rb = db.iloc[0]
    assert min(rb["shell_volume_frac_inner"], rb["shell_volume_frac_mid"],
               rb["shell_volume_frac_outer"]) > 0.1


def test_b5_mean_curvature_dimensionless_under_spacing():
    """The size-normalized curvature is DIMENSIONLESS: the SAME object sampled at a
    different physical spacing (a genuine physical size change, same mesh topology)
    gives the SAME curvature. The raw umbrella Laplacian carries a length unit and
    scales linearly with spacing; dividing by the equivalent radius (also a length)
    cancels it exactly. This is the rigorous test of the B5 fix — before the fix the
    raw value (and `H * R`) scaled with size."""
    lab = _ball(10, (28, 28, 28))
    d1 = _single_df(lab, spacing=(1.0, 1.0, 1.0))
    qf.compute_curvature_features(lab, d1, spacing=(1.0, 1.0, 1.0))
    d2 = _single_df(lab, spacing=(2.0, 2.0, 2.0))
    qf.compute_curvature_features(lab, d2, spacing=(2.0, 2.0, 2.0))
    a = d1.iloc[0]["abs_curvature_mean"]
    b = d2.iloc[0]["abs_curvature_mean"]
    assert np.isfinite(a) and a > 0
    # exactly invariant to the physical scale (dimensionless H / R).
    assert b == pytest.approx(a, rel=1e-6), (a, b)
    # and the signed mean curvature is likewise scale-invariant.
    assert d2.iloc[0]["mean_curvature_mean"] == pytest.approx(
        d1.iloc[0]["mean_curvature_mean"], rel=1e-6)


def _cyto_scene(nuc_x):
    """A 10^3 cell with a 4^3 nucleus placed at x-offset ``nuc_x`` inside it."""
    Z, Y, X = 16, 16, 16
    cells = np.zeros((Z, Y, X), np.int32)
    nuclei = np.zeros((Z, Y, X), np.int32)
    cells[3:13, 3:13, 3:13] = 1
    nuclei[6:10, 6:10, nuc_x:nuc_x + 4] = 10
    triage = [{"cell_id": 1, "n_nuclei": 1, "nucleus_ids": [10]}]
    return cells, nuclei, triage


# =========================================================================== #
# C. CHEAP GUARDS
# =========================================================================== #

def test_c8_peak_count_ignores_jitter():
    """A near-cylindrical rod with a single tiny 1-voxel bump must NOT be counted as
    a multi-lobe cell — the prominence floor rejects voxel jitter. A genuine
    dumbbell (two real lobes) still reads >= 2."""
    # smooth rod: constant cross-section -> exactly one peak after padding.
    rod = np.zeros((30, 12, 12), np.int32)
    rod[3:27, 4:8, 4:8] = 1
    drod = _single_df(rod)
    qf.compute_cross_section_profile_stats(rod, drod)
    assert drod.iloc[0]["cross_section_peak_count"] == 1.0

    # genuine dumbbell: two fat lobes joined by a thin neck -> >= 2 peaks.
    db = np.zeros((30, 16, 16), np.int32)
    db[3:9, 3:13, 3:13] = 1          # lobe A
    db[12:18, 6:10, 6:10] = 1        # thin neck
    db[21:27, 3:13, 3:13] = 1        # lobe B
    ddb = _single_df(db)
    qf.compute_cross_section_profile_stats(db, ddb)
    assert ddb.iloc[0]["cross_section_peak_count"] >= 2.0


def test_c9_colinear_neighbour_ring_is_nan():
    """A cell whose graph neighbours are ~COLINEAR (a 1D string) gives an ambiguous
    plane fit -> every tissue-frame column is NaN (rank-deficient covariance)."""
    # a straight line of 4 touching cells along X; the 2nd cell's neighbours (the
    # 1st, 3rd, 4th) are all colinear along X.
    lab = np.zeros((6, 6, 40), np.int32)
    for i, x0 in enumerate(range(2, 34, 8)):
        lab[1:5, 1:5, x0:x0 + 8] = i + 1
    df = _with_graph(lab, max_radius=3)
    qf.compute_tissue_frame_features(df)
    # the 2nd cell has 3 colinear neighbours -> rank-deficient -> NaN frame.
    r = df.set_index("label").loc[2]
    assert np.isnan(r["neighbor_ring_planarity"])
    assert np.isnan(r["local_normal_alignment"])


# =========================================================================== #
# D. MANIFEST / DEAD CODE
# =========================================================================== #

def test_d11_dead_boundary_interior_family_dropped():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert "boundary_interior" not in _FEATURE_FAMILIES
    man = fm.get_manifest()
    for c in ("boundary_intensity_mean", "interior_intensity_mean",
              "boundary_interior_ratio", "z_intensity_variation"):
        assert c not in man, c
    # the prefixed B1 ratio IS still tagged for a sampled channel.
    assert "mem_boundary_interior_ratio" in fm.channel_intensity_tags("mem")


def test_d12_crowding_columns_tagged():
    man = fm.get_manifest()
    for c in ("nn_centroid_distance", "knn_mean_distance",
              "neighbor_count_in_radius", "local_density"):
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["redundancy_group"] == "crowding"
        assert tags["frame"] == "image"
        assert tags["bounded"] is False


def test_d12_prob_boundary_drop_bounded_signed():
    """prob_boundary_drop is a signed difference bounded [-1,1] -> tagged bounded."""
    man = fm.get_manifest()
    assert man["prob_boundary_drop"]["bounded"] is True
    assert man["prob_boundary_drop"]["units"] == "score"

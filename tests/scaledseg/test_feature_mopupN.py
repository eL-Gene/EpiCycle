"""Mop-up N feature tests — the SHAPE / SHELL / CROSS-SECTION / CURVATURE tail.

NEW features built from the EpiCycle atlas equations:
the remaining NOT-rows: cross-section, major-axis cross-section,
radial/shell, geodesic / medial-axis, curvature/symmetry,
and multi-scale. All geometry-only (labels-only), additive, default-off
byte-identical — they ride the SAME per-cell crop pass as M0/M1/M5/Wave-H.

Pure-function known-shape tests (no GPU / Cellpose) on synthetic shapes whose
answers we know by construction:

* an **hourglass** cell -> a deep cross-section waist: high
  ``cross_section_trough_prominence`` + ``cross_section_constriction_index``.
* a **flared cone** -> ``basal_flare_index`` > 1 at the fat base.
* a **prolate** (cigar) vs **oblate** (pancake) ellipsoid -> the triaxial axis
  ratios separate them (already covered in M0; here we sanity-check the new
  major-axis profile bulge/peak + the shell anisotropy on them).
* a **cortex-offset** cell (a dense core pushed to one side) -> ``core_offset`` > 0.
* a **bent rod** -> ``skeleton_main_path_tortuosity`` > 1 (the medial spine
  follows the bend) while a straight rod -> ~1.
* a **Y / fork** -> ``medial_branch_count`` >= 1; a simple rod -> 0.

Also asserts the manifest tags every new column, the FeatureStage wiring emits
them, and the new columns are bounded where the manifest says so.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# Every NEW column this wave adds, grouped by the extended function.
_CROSS_SECTION_NEW = (
    "basal_flare_index",
    "cross_section_peak_prominence", "cross_section_trough_prominence",
    "slice_centroid_drift_mean", "slice_centroid_drift_max",
    "slice_eccentricity_mean", "slice_eccentricity_taper",
    "slice_orientation_twist",
)
_MA_NEW = (
    "ma_max_cross_section_axis_relative", "ma_cross_section_bulge",
    "ma_peak_count", "ma_cross_section_skew",
)
_EDT_NEW = ("medial_radius_range",)
_GEO_NEW = (
    "skeleton_main_path_length", "skeleton_main_path_tortuosity",
    "endpoint_distance_ratio", "medial_axis_bend_energy", "medial_branch_count",
)
_SHELL_NEW = (
    "shell_compactness_inner", "shell_compactness_mid", "shell_compactness_outer",
    "shell_eccentricity_inner", "shell_eccentricity_mid", "shell_eccentricity_outer",
    "core_offset",
    "shell_volume_ratio_inner_outer", "shell_volume_ratio_mid_outer",
)
_CURV_NEW = ("radial_symmetry_score", "apical_basal_symmetry_score")
_MULTISCALE_NEW = (
    "shell_sphericity_mid",
    "shape_axis_q1_fraction", "shape_axis_q4_fraction", "shape_axis_q1_q4_ratio",
)
_ALL_NEW = (
    *_CROSS_SECTION_NEW, *_MA_NEW, *_EDT_NEW, *_GEO_NEW,
    *_SHELL_NEW, *_CURV_NEW, *_MULTISCALE_NEW,
)


# --------------------------------------------------------------------------- #
# shape builders
# --------------------------------------------------------------------------- #

def _ball_mask(R, c, shape):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= R * R


def _tube(points, R, shape):
    """Union of radius-``R`` balls along a polyline -> a thick tube that
    skeletonizes reliably (block prisms can skeletonize to nothing when flat)."""
    lab = np.zeros(shape, dtype=np.int32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    m = np.zeros(shape, dtype=bool)
    for (z, y, x) in points:
        m |= ((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) <= R * R
    lab[m] = 1
    return lab


def _hourglass():
    """A waisted cell along Z: wide at both ends, pinched in the middle."""
    shape = (24, 24, 24)
    lab = np.zeros(shape, dtype=np.int32)
    cy, cx = 12, 12
    for z in range(2, 22):
        # radius dips to a minimum at the centre (z=12), wide at z=2 / z=21.
        rad = 3 + int(round(6 * abs(z - 12) / 10.0))   # 3 at centre, 9 at ends
        yy, xx = np.ogrid[:shape[1], :shape[2]]
        lab[z][((yy - cy) ** 2 + (xx - cx) ** 2) <= rad * rad] = 1
    return lab


def _flared_cone():
    """A cone fat at LOW z (the base), narrow at HIGH z — a basal flare."""
    shape = (16, 28, 28)
    lab = np.zeros(shape, dtype=np.int32)
    cy, cx = 14, 14
    yy, xx = np.ogrid[:shape[1], :shape[2]]
    for k, z in enumerate(range(2, 14)):
        rad = 11 - k                                   # 11 at base .. 0 at tip
        if rad < 1:
            rad = 1
        lab[z][((yy - cy) ** 2 + (xx - cx) ** 2) <= rad * rad] = 1
    return lab


def _uniform_cylinder():
    """Constant cross-section: a square prism along Z (no flare, no waist)."""
    lab = np.zeros((20, 20, 20), dtype=np.int32)
    lab[2:18, 6:14, 6:14] = 1
    return lab


def _prolate():
    """A cigar (prolate spheroid): long along X, narrow in Y/Z."""
    shape = (16, 16, 40)
    lab = np.zeros(shape, dtype=np.int32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    a, b = 17.0, 5.0
    val = ((xx - 20) / a) ** 2 + ((yy - 8) / b) ** 2 + ((zz - 8) / b) ** 2
    lab[val <= 1.0] = 1
    return lab


def _oblate():
    """A pancake (oblate spheroid): broad in X/Y, thin in Z."""
    shape = (12, 36, 36)
    lab = np.zeros(shape, dtype=np.int32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    a, c = 15.0, 3.0
    val = ((xx - 18) / a) ** 2 + ((yy - 18) / a) ** 2 + ((zz - 6) / c) ** 2
    lab[val <= 1.0] = 1
    return lab


def _cortex_offset():
    """A teardrop: a big lobe (which holds the EDT-deepest core) + a thin tail
    that drags the whole-cell centroid away from the core -> the inner-band
    centroid is offset from the cell centroid (core_offset > 0)."""
    shape = (32, 32, 40)
    lab = np.zeros(shape, dtype=np.int32)
    lab[_ball_mask(10, (16, 16, 12), shape)] = 1      # big lobe near low x
    lab[14:18, 14:18, 12:36] = 1                      # thin tail toward high x
    return lab


def _cone_x():
    """A cone whose long (major) axis is X: fat disc at low x, narrow at high x.
    Its major axis lies along X so the apical/basal + axis-quartile families read
    the genuine taper."""
    shape = (24, 44, 44)
    lab = np.zeros(shape, dtype=np.int32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    m = np.zeros(shape, dtype=bool)
    for k, x in enumerate(range(4, 40)):
        rad = max(1.0, 11.0 - 0.28 * k)
        m |= (xx == x) & (((yy - 22) ** 2 + (zz - 12) ** 2) <= rad * rad)
    lab[m] = 1
    return lab


def _round_tube():
    """A fat round tube along X — close to a body of revolution about its long
    (major) axis."""
    return _tube([(10, 20, x) for x in range(4, 36)], 4, (20, 40, 40))


def _flat_plate():
    """A thin off-axis plate — voxels concentrate in a few azimuthal sectors
    about its long axis (low radial symmetry)."""
    lab = np.zeros((8, 40, 40), dtype=np.int32)
    lab[3:5, 16:24, 4:36] = 1
    return lab


def _straight_rod():
    """A straight round tube along X (skeletonizes to a clean spine)."""
    shape = (20, 40, 40)
    return _tube([(10, 20, x) for x in range(4, 36)], 3, shape)


def _bent_rod():
    """A bent round tube: along X then turning along Y — the medial spine must
    follow the bend (skeleton main-path tortuosity > 1)."""
    shape = (20, 40, 40)
    pts = ([(10, 12, x) for x in range(4, 30)]
           + [(10, y, 30) for y in range(12, 34)])
    return _tube(pts, 3, shape)


def _y_fork():
    """A Y-shaped tube: one stem splitting into two diverging arms (a branch
    node in the skeleton)."""
    shape = (16, 44, 44)
    stem = [(8, 22, x) for x in range(4, 22)]
    arm1 = [(8, 22 - (x - 22), x) for x in range(22, 40)]
    arm2 = [(8, 22 + (x - 22), x) for x in range(22, 40)]
    return _tube(stem + arm1 + arm2, 2, shape)


def _tiny():
    lab = np.zeros((6, 6, 6), dtype=np.int32)
    lab[3, 3, 3] = 1
    return lab


def _single(lab):
    return qf.extract_features(lab)


# --------------------------------------------------------------------------- #
# cross-section remainder
# --------------------------------------------------------------------------- #

def test_cross_section_new_columns_present_and_finite():
    lab = _uniform_cylinder()
    df = _single(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    for c in _CROSS_SECTION_NEW:
        assert c in df.columns, c
        assert np.isfinite(df.iloc[0][c]), (c, df.iloc[0][c])


def test_hourglass_has_deep_trough_and_constriction():
    """An hourglass cell has a pronounced central neck -> a large trough
    prominence + a high constriction index; a uniform cylinder ~0."""
    hg = _hourglass()
    cyl = _uniform_cylinder()
    dh = _single(hg); qf.compute_cross_section_profile_stats(hg, dh)
    dc = _single(cyl); qf.compute_cross_section_profile_stats(cyl, dc)
    rh, rc = dh.iloc[0], dc.iloc[0]
    assert rh["cross_section_trough_prominence"] > rc["cross_section_trough_prominence"]
    assert rh["cross_section_trough_prominence"] > 0.2
    assert rh["cross_section_constriction_index"] > rc["cross_section_constriction_index"]


def test_flared_cone_basal_flare_high():
    """A cone fat at the base flares (basal_area >> mean -> basal_flare_index > 1);
    a uniform cylinder is ~1."""
    cone = _flared_cone()
    cyl = _uniform_cylinder()
    dco = _single(cone); qf.compute_cross_section_profile_stats(cone, dco)
    dcy = _single(cyl); qf.compute_cross_section_profile_stats(cyl, dcy)
    assert dco.iloc[0]["basal_flare_index"] > 1.2
    assert dcy.iloc[0]["basal_flare_index"] == pytest.approx(1.0, abs=0.2)


def test_basal_flare_honours_invert_z():
    """With invert_z the basal end is the HIGH-z end. For the cone (fat at LOW z),
    flipping invert_z moves the 'basal' sample to the narrow tip -> flare < 1."""
    cone = _flared_cone()
    d0 = _single(cone); qf.compute_cross_section_profile_stats(cone, d0, invert_z=False)
    d1 = _single(cone); qf.compute_cross_section_profile_stats(cone, d1, invert_z=True)
    assert d0.iloc[0]["basal_flare_index"] > 1.2     # fat base read as basal
    assert d1.iloc[0]["basal_flare_index"] < 1.0     # narrow tip read as basal


def test_slice_eccentricity_round_vs_elongated():
    """A round (square-ish) prism has low per-slice eccentricity; a slab that is
    very wide in X and thin in Y has high per-slice eccentricity."""
    cyl = _uniform_cylinder()
    slab = np.zeros((20, 12, 40), dtype=np.int32)
    slab[2:18, 5:7, 4:36] = 1          # each Z slice is a long thin Y-by-X strip
    dc = _single(cyl); qf.compute_cross_section_profile_stats(cyl, dc)
    ds = _single(slab); qf.compute_cross_section_profile_stats(slab, ds)
    assert ds.iloc[0]["slice_eccentricity_mean"] > dc.iloc[0]["slice_eccentricity_mean"]
    assert ds.iloc[0]["slice_eccentricity_mean"] > 0.8
    assert 0.0 <= dc.iloc[0]["slice_eccentricity_mean"] <= 1.0


def test_slice_centroid_drift_straight_vs_leaning():
    """A straight prism has ~0 centroid drift; a 'leaning tower' whose lateral
    centroid shifts with Z has a large drift."""
    straight = _uniform_cylinder()
    lean = np.zeros((20, 40, 40), dtype=np.int32)
    for z in range(2, 18):
        off = z                                       # centroid marches with z
        lean[z, 4 + off - 2:4 + off + 2, 4 + off - 2:4 + off + 2] = 1
    dsr = _single(straight); qf.compute_cross_section_profile_stats(straight, dsr)
    dln = _single(lean); qf.compute_cross_section_profile_stats(lean, dln)
    assert dln.iloc[0]["slice_centroid_drift_max"] > dsr.iloc[0]["slice_centroid_drift_max"]
    assert dln.iloc[0]["slice_centroid_drift_mean"] > 0.5
    assert dsr.iloc[0]["slice_centroid_drift_mean"] < 0.2


def test_orientation_twist_straight_vs_twisted():
    """A constant-orientation slab has ~0 twist; a slab whose in-plane long axis
    rotates from slice to slice accumulates twist."""
    straight = np.zeros((20, 24, 24), dtype=np.int32)
    straight[2:18, 10:14, 4:20] = 1        # long axis fixed along X
    twisted = np.zeros((20, 28, 28), dtype=np.int32)
    cy, cx = 14, 14
    for z in range(2, 18):
        ang = (z - 2) * (np.pi / 16.0)     # rotate the strip with z
        for r in range(-9, 10):
            y = int(round(cy + r * np.sin(ang)))
            x = int(round(cx + r * np.cos(ang)))
            if 0 <= y < 28 and 0 <= x < 28:
                twisted[z, max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 1
    dst = _single(straight); qf.compute_cross_section_profile_stats(straight, dst)
    dtw = _single(twisted); qf.compute_cross_section_profile_stats(twisted, dtw)
    assert dtw.iloc[0]["slice_orientation_twist"] > dst.iloc[0]["slice_orientation_twist"]
    assert dtw.iloc[0]["slice_orientation_twist"] > 0.5


def test_cross_section_single_slice_degenerate_not_crash():
    lab = np.zeros((6, 8, 8), dtype=np.int32)
    lab[3, 2:6, 2:6] = 1                  # a single present Z slice
    df = _single(lab)
    qf.compute_cross_section_profile_stats(lab, df)
    r = df.iloc[0]
    for c in _CROSS_SECTION_NEW:
        assert c in df.columns
    assert r["basal_flare_index"] == pytest.approx(1.0)
    assert r["slice_orientation_twist"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# major-axis cross-section remainder
# --------------------------------------------------------------------------- #

def test_ma_new_columns_present_and_finite():
    lab = _prolate()
    df = _single(lab)
    qf.compute_ma_cross_section_features(lab, df)
    for c in _MA_NEW:
        assert c in df.columns, c
        assert np.isfinite(df.iloc[0][c]), (c, df.iloc[0][c])


def test_ma_peak_count_hourglass_vs_cylinder():
    """A waisted cell along its own axis shows 2 lobes; a uniform prism 1."""
    hg = _hourglass()
    cyl = _uniform_cylinder()
    dh = _single(hg); qf.compute_ma_cross_section_features(hg, dh)
    dc = _single(cyl); qf.compute_ma_cross_section_features(cyl, dc)
    assert dh.iloc[0]["ma_peak_count"] >= 2
    assert dc.iloc[0]["ma_peak_count"] == pytest.approx(1.0)


def test_ma_max_axis_relative_bounded():
    for builder in (_prolate, _oblate, _hourglass, _uniform_cylinder):
        lab = builder()
        df = _single(lab)
        qf.compute_ma_cross_section_features(lab, df)
        v = df.iloc[0]["ma_max_cross_section_axis_relative"]
        assert np.isnan(v) or (0.0 - 1e-9 <= v <= 1.0 + 1e-9), (builder.__name__, v)


def test_ma_bulge_hourglass_gt_cylinder():
    """A waisted (barreled) profile departs from its linear taper fit -> a larger
    residual bulge than a uniform prism."""
    hg = _hourglass()
    cyl = _uniform_cylinder()
    dh = _single(hg); qf.compute_ma_cross_section_features(hg, dh)
    dc = _single(cyl); qf.compute_ma_cross_section_features(cyl, dc)
    assert dh.iloc[0]["ma_cross_section_bulge"] > dc.iloc[0]["ma_cross_section_bulge"]


# --------------------------------------------------------------------------- #
# EDT + skeleton main-path / medial-axis remainder
# --------------------------------------------------------------------------- #

def test_edt_medial_range_present_and_nonneg():
    lab = _prolate()
    df = _single(lab)
    qf.compute_edt_features(lab, df)
    v = df.iloc[0]["medial_radius_range"]
    assert np.isfinite(v) and v >= 0.0


def test_geo_new_columns_present():
    lab = _straight_rod()
    df = _single(lab)
    qf.compute_geodesic_features(lab, df)
    for c in _GEO_NEW:
        assert c in df.columns, c


def test_bent_rod_main_path_more_tortuous_than_straight():
    """The skeleton main-path of a straight rod is ~straight (tortuosity ~1);
    a bent rod's spine follows the bend (tortuosity > 1)."""
    straight = _straight_rod()
    bent = _bent_rod()
    ds = _single(straight); qf.compute_geodesic_features(straight, ds)
    db = _single(bent); qf.compute_geodesic_features(bent, db)
    ts = ds.iloc[0]["skeleton_main_path_tortuosity"]
    tb = db.iloc[0]["skeleton_main_path_tortuosity"]
    assert np.isfinite(ts) and np.isfinite(tb)
    assert ts == pytest.approx(1.0, abs=0.25)
    assert tb > ts
    assert tb > 1.15
    # endpoint_distance_ratio is the inverse straightness: straight ~1, bent < 1.
    assert db.iloc[0]["endpoint_distance_ratio"] < ds.iloc[0]["endpoint_distance_ratio"]


def test_straight_rod_main_path_length_positive():
    lab = _straight_rod()
    df = _single(lab)
    qf.compute_geodesic_features(lab, df)
    r = df.iloc[0]
    assert r["skeleton_main_path_length"] > 10.0      # spans most of the long rod
    assert r["medial_axis_bend_energy"] >= 0.0


def test_y_fork_has_branch_node_rod_does_not():
    """A Y-shaped object has at least one skeleton branch node; a simple rod 0."""
    fork = _y_fork()
    rod = _straight_rod()
    df = _single(fork); qf.compute_geodesic_features(fork, df)
    dr = _single(rod); qf.compute_geodesic_features(rod, dr)
    assert df.iloc[0]["medial_branch_count"] >= 1
    assert dr.iloc[0]["medial_branch_count"] == pytest.approx(0.0)


def test_geo_main_path_tortuosity_at_least_one():
    for builder in (_straight_rod, _bent_rod, _prolate):
        lab = builder()
        df = _single(lab)
        qf.compute_geodesic_features(lab, df)
        t = df.iloc[0]["skeleton_main_path_tortuosity"]
        assert np.isnan(t) or t >= 1.0 - 1e-6, (builder.__name__, t)


def test_geo_tiny_object_not_crash():
    lab = _tiny()
    df = _single(lab)
    qf.compute_geodesic_features(lab, df)
    for c in _GEO_NEW:
        assert c in df.columns


# --------------------------------------------------------------------------- #
# radial / shell remainder
# --------------------------------------------------------------------------- #

def test_shell_new_columns_present_on_ball():
    lab = np.zeros((28, 28, 28), dtype=np.int32)
    lab[_ball_mask(10, (14, 14, 14), lab.shape)] = 1
    df = _single(lab)
    qf.compute_shell_features(lab, df)
    for c in _SHELL_NEW:
        assert c in df.columns, c


def test_shell_band_eccentricity_bounded():
    for builder in (_prolate, _oblate, _uniform_cylinder):
        lab = builder()
        df = _single(lab)
        qf.compute_shell_features(lab, df)
        for c in ("shell_eccentricity_inner", "shell_eccentricity_mid",
                  "shell_eccentricity_outer"):
            v = df.iloc[0][c]
            assert np.isnan(v) or (0.0 - 1e-9 <= v <= 1.0 + 1e-9), (builder.__name__, c, v)


def test_cortex_offset_core_off_centre():
    """The cortex-offset (teardrop) shape has its EDT-deepest core in the big
    lobe while the thin tail drags the whole-cell centroid away, so the inner
    band centroid is offset from the cell centroid (core_offset > 0). A
    concentric ball has a centred core (~0 offset)."""
    off = _cortex_offset()
    ball = np.zeros((28, 28, 28), dtype=np.int32)
    ball[_ball_mask(10, (14, 14, 14), ball.shape)] = 1
    do = _single(off); qf.compute_shell_features(off, do)
    db = _single(ball); qf.compute_shell_features(ball, db)
    assert np.isfinite(do.iloc[0]["core_offset"])
    assert do.iloc[0]["core_offset"] > db.iloc[0]["core_offset"]
    assert do.iloc[0]["core_offset"] > 0.03


def test_shell_compactness_prolate_bands_low():
    """A prolate cigar's shells are elongated streaks -> lower Bribiesca band
    compactness than a compact ball's."""
    cigar = _prolate()
    ball = np.zeros((28, 28, 28), dtype=np.int32)
    ball[_ball_mask(10, (14, 14, 14), ball.shape)] = 1
    dc = _single(cigar); qf.compute_shell_features(cigar, dc)
    dbz = _single(ball); qf.compute_shell_features(ball, dbz)
    # the inner core compactness of a ball exceeds the cigar's stringy core.
    assert dbz.iloc[0]["shell_compactness_inner"] > dc.iloc[0]["shell_compactness_inner"]


def test_shell_volume_ratios_finite_on_ball():
    lab = np.zeros((28, 28, 28), dtype=np.int32)
    lab[_ball_mask(10, (14, 14, 14), lab.shape)] = 1
    df = _single(lab)
    qf.compute_shell_features(lab, df)
    r = df.iloc[0]
    assert np.isfinite(r["shell_volume_ratio_inner_outer"])
    assert np.isfinite(r["shell_volume_ratio_mid_outer"])
    assert r["shell_volume_ratio_inner_outer"] >= 0.0


def test_shell_remainder_tiny_object_not_crash():
    """A 1-voxel object — the interior EDT collapses, so the shell remainder
    columns degrade to NaN (no crash, all columns present)."""
    lab = _tiny()
    df = _single(lab)
    qf.compute_shell_features(lab, df)
    for c in _SHELL_NEW:
        assert c in df.columns


def test_shell_remainder_thin_sheet_not_crash():
    """A 1-voxel-thick broad SHEET keeps a (2D) interior EDT, so it lands in the
    main branch with all voxels in the inner band — the outer-band shape columns
    are NaN (empty rind) and the inner/outer ratio is NaN (zero rind). No crash."""
    lab = np.zeros((6, 24, 24), dtype=np.int32)
    lab[3, 3:21, 3:21] = 1
    df = _single(lab)
    qf.compute_shell_features(lab, df)
    for c in _SHELL_NEW:
        assert c in df.columns


# --------------------------------------------------------------------------- #
# curvature / symmetry remainder
# --------------------------------------------------------------------------- #

def test_curv_new_columns_present():
    lab = np.zeros((24, 24, 24), dtype=np.int32)
    lab[_ball_mask(8, (12, 12, 12), lab.shape)] = 1
    df = _single(lab)
    qf.compute_curvature_features(lab, df)
    for c in _CURV_NEW:
        assert c in df.columns, c
        assert np.isfinite(df.iloc[0][c]), (c, df.iloc[0][c])


def test_radial_symmetry_cylinder_high_slab_low():
    """A body of revolution (a round tube about its long axis) fills azimuthal
    sectors evenly -> high radial symmetry; a flat off-axis plate concentrates
    voxels in a few sectors -> low."""
    rod = _round_tube()
    slab = _flat_plate()
    dr = _single(rod); qf.compute_curvature_features(rod, dr)
    dsl = _single(slab); qf.compute_curvature_features(slab, dsl)
    sr = dr.iloc[0]["radial_symmetry_score"]
    ss = dsl.iloc[0]["radial_symmetry_score"]
    assert np.isfinite(sr) and np.isfinite(ss)
    assert sr > ss
    assert 0.0 <= sr <= 1.0 and 0.0 <= ss <= 1.0


def test_apical_basal_symmetry_barrel_high_cone_low():
    """A symmetric tube is end-to-end symmetric about the plane ⊥ its long axis
    (high IoU); a cone-along-X (one fat end) is not. Both must have their MAJOR
    axis along the long (X) dimension for the reflection to test the two ends."""
    tube = _round_tube()
    cone = _cone_x()
    dt = _single(tube); qf.compute_curvature_features(tube, dt)
    dco = _single(cone); qf.compute_curvature_features(cone, dco)
    ab_t = dt.iloc[0]["apical_basal_symmetry_score"]
    ab_co = dco.iloc[0]["apical_basal_symmetry_score"]
    assert ab_t > ab_co
    assert ab_t > 0.7


def test_symmetry_scores_bounded():
    for builder in (_straight_rod, _hourglass, _flared_cone, _prolate):
        lab = builder()
        df = _single(lab)
        qf.compute_curvature_features(lab, df)
        for c in _CURV_NEW:
            v = df.iloc[0][c]
            assert np.isnan(v) or (0.0 - 1e-9 <= v <= 1.0 + 1e-9), (builder.__name__, c, v)


def test_curv_symmetry_missing_major_axis_nan():
    """No ``major_axis_*`` columns -> the new symmetry scores are NaN (no crash);
    the rest of the curvature family still computes."""
    lab = np.zeros((24, 24, 24), dtype=np.int32)
    lab[_ball_mask(8, (12, 12, 12), lab.shape)] = 1
    df = pd.DataFrame({"label": [1]})   # no axis columns
    qf.compute_curvature_features(lab, df)
    r = df.iloc[0]
    assert np.isnan(r["radial_symmetry_score"])
    assert np.isnan(r["apical_basal_symmetry_score"])
    assert np.isfinite(r["convexity"])


# --------------------------------------------------------------------------- #
# multi-scale remainder
# --------------------------------------------------------------------------- #

def test_multiscale_new_columns_present():
    lab = np.zeros((28, 28, 28), dtype=np.int32)
    lab[_ball_mask(10, (14, 14, 14), lab.shape)] = 1
    df = _single(lab)
    qf.compute_multiscale_shape(lab, df)
    for c in _MULTISCALE_NEW:
        assert c in df.columns, c


def test_axis_quartile_fractions_bounded_and_balanced_on_symmetric():
    """A symmetric round tube has ~equal first/last axial-quartile volume
    fractions (ratio ~1); a cone-along-X is imbalanced (q1 fat end >> q4 narrow
    end, ratio far from 1)."""
    tube = _round_tube()
    cone = _cone_x()
    dc = _single(tube); qf.compute_multiscale_shape(tube, dc)
    dco = _single(cone); qf.compute_multiscale_shape(cone, dco)
    rc, rco = dc.iloc[0], dco.iloc[0]
    for c in ("shape_axis_q1_fraction", "shape_axis_q4_fraction"):
        assert 0.0 <= rc[c] <= 1.0
        assert 0.0 <= rco[c] <= 1.0
    # symmetric tube: q1 ≈ q4 -> ratio ~1; cone: lopsided -> ratio far from 1.
    assert rc["shape_axis_q1_q4_ratio"] == pytest.approx(1.0, abs=0.4)
    assert abs(np.log(rco["shape_axis_q1_q4_ratio"])) > abs(np.log(rc["shape_axis_q1_q4_ratio"]))


def test_shell_sphericity_mid_present_on_ball():
    lab = np.zeros((30, 30, 30), dtype=np.int32)
    lab[_ball_mask(11, (15, 15, 15), lab.shape)] = 1
    df = _single(lab)
    qf.compute_multiscale_shape(lab, df)
    v = df.iloc[0]["shell_sphericity_mid"]
    assert np.isnan(v) or (0.0 <= v <= 1.0)


# --------------------------------------------------------------------------- #
# manifest + bounded-flag coverage
# --------------------------------------------------------------------------- #

def test_manifest_tags_all_new_columns():
    man = fm.get_manifest()
    for c in _ALL_NEW:
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] in ("image", "tissue", "intrinsic")
        assert isinstance(tags["bounded"], bool)


def test_manifest_bounded_flags_match_math():
    man = fm.get_manifest()
    bounded_true = (
        "cross_section_peak_prominence", "cross_section_trough_prominence",
        "slice_eccentricity_mean",
        "ma_max_cross_section_axis_relative",
        "endpoint_distance_ratio",
        "shell_compactness_inner", "shell_compactness_mid", "shell_compactness_outer",
        "shell_eccentricity_inner", "shell_eccentricity_mid", "shell_eccentricity_outer",
        "radial_symmetry_score", "apical_basal_symmetry_score",
        "shell_sphericity_mid", "shape_axis_q1_fraction", "shape_axis_q4_fraction",
    )
    bounded_false = (
        "basal_flare_index", "slice_centroid_drift_mean", "slice_centroid_drift_max",
        "slice_eccentricity_taper", "slice_orientation_twist",
        "ma_cross_section_bulge", "ma_peak_count", "ma_cross_section_skew",
        "medial_radius_range",
        "skeleton_main_path_length", "skeleton_main_path_tortuosity",
        "medial_axis_bend_energy", "medial_branch_count",
        "core_offset", "shell_volume_ratio_inner_outer", "shell_volume_ratio_mid_outer",
        "shape_axis_q1_q4_ratio",
    )
    for c in bounded_true:
        assert man[c]["bounded"] is True, c
    for c in bounded_false:
        assert man[c]["bounded"] is False, c


# --------------------------------------------------------------------------- #
# FeatureStage wiring — the new columns reach the per-cell worker output
# --------------------------------------------------------------------------- #

def test_feature_stage_extract_one_emits_new_columns():
    from packages.epicycle_scaledseg.core import feature_stage as fs

    lab = np.zeros((20, 28, 28), dtype=np.int32)
    lab[_ball_mask(7, (10, 9, 9), lab.shape)] = 1
    lab[_ball_mask(7, (10, 18, 18), lab.shape)] = 2
    import tempfile, os
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "labels.npy")
    np.save(p, lab)
    fs._init_feature_worker(p)
    try:
        task = (1, (3, 18, 2, 17, 2, 17), None, False, True)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict), row
    for c in _ALL_NEW:
        assert c in row, f"{c} missing from _extract_one output"


def test_feature_stage_families_registry_includes_new():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    for c in _CROSS_SECTION_NEW:
        assert c in _FEATURE_FAMILIES["cross_section_profile"], c
    for c in _MA_NEW:
        assert c in _FEATURE_FAMILIES["ma_cross_section"], c
    for c in _EDT_NEW:
        assert c in _FEATURE_FAMILIES["edt"], c
    for c in _GEO_NEW:
        assert c in _FEATURE_FAMILIES["geodesic"], c
    for c in _SHELL_NEW:
        assert c in _FEATURE_FAMILIES["shell"], c
    for c in _CURV_NEW:
        assert c in _FEATURE_FAMILIES["curvature"], c
    for c in _MULTISCALE_NEW:
        assert c in _FEATURE_FAMILIES["multiscale_shape"], c

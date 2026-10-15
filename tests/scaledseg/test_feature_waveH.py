"""Wave H feature tests — the LAST non-research atlas wave. Three families built on
the existing primitives (no new deps), default-off byte-identical:

* ``ma_apical_basal`` — orients the M1 tilt-robust major-axis cross-section
  profile by the Wave C outward normal and splits it apical-third vs basal-third, so
  a WEDGE cell (wider at the apical end) reads ``ma_apical_area > ma_basal_area``
  when its major axis is oriented toward the tissue surface. A GLOBAL pass (needs the
  outward normal); NaN'd for an untrusted normal; absent when Wave C did not run.
* ``apical_junction`` — the classic epithelial polygon topology from a cell's
  LATERAL contacts (perpendicular to outward — the apical-junctional belt): the
  number of distinct lateral neighbours (~6 in ordered tissue), the total lateral
  contact perimeter, and the cv of the lateral patch sizes. A GLOBAL pass (needs the
  adjacency graph + the outward normal). A synthetic hexagonal-packed centre cell
  reads ``apical_polygon_sides ≈ 6``.
* ``multiscale_shape`` — a MODEST per-cell SHAPE template across the M5 EDT
  shells (inner core / outer rind discrete compactness + the outer-rind surface
  fraction). A sphere (a fat compact core) vs a thin sheet (almost all surface rind)
  separate on the shell shape stats.

Pure-function known-geometry tests (no GPU / Cellpose) + the manifest tags + the
``_FEATURE_FAMILIES`` registry + the default-off contract through
``extract_label_features``.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm
from packages.epicycle_scaledseg.core.qc_features import (
    compute_ma_apical_basal, compute_apical_junction, compute_multiscale_shape,
    MA_APICAL_BASAL_COLUMNS, APICAL_JUNCTION_COLUMNS, MULTISCALE_SHAPE_COLUMNS,
)
from packages.epicycle_scaledseg.reconcile.features import extract_label_features


_MA_COLS = list(MA_APICAL_BASAL_COLUMNS)
_AJ_COLS = list(APICAL_JUNCTION_COLUMNS)
_MS_COLS = list(MULTISCALE_SHAPE_COLUMNS)
# the two GLOBAL Wave-H passes (gated on the Wave C outward normal).
_ALL_WAVEH_GLOBAL = _MA_COLS + _AJ_COLS


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ball(R, c, shape):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= R * R


def _inject_outward(df, vec_zyx, conf=1.0, labels=None):
    """Hand-set the Wave C outward columns on ``df`` (the sign basis the global Wave
    H passes consume). ``vec_zyx`` applied to every row (or just ``labels``)."""
    z, y, x = vec_zyx
    df["tissue_outward_z"] = float(z)
    df["tissue_outward_y"] = float(y)
    df["tissue_outward_x"] = float(x)
    df["tissue_outward_confidence"] = float(conf)
    return df


# --------------------------------------------------------------------------- #
# 1) ma_apical/basal — a WEDGE cell wider at the apical end
# --------------------------------------------------------------------------- #
def _wedge_cell():
    """A cell that flares along +X: at each X plane its Y/Z cross-section grows, so
    the +X end is WIDE and the −X end is NARROW. Major axis ~ +/-X. Orient outward to
    +X -> the wide end is apical -> ma_apical_area > ma_basal_area."""
    Z, Y, X = 24, 24, 30
    lab = np.zeros((Z, Y, X), dtype=np.int32)
    cz, cy = Z // 2, Y // 2
    for xi in range(4, 26):
        # radius grows linearly from ~2 (basal, low x) to ~9 (apical, high x).
        rad = 2.0 + 7.0 * (xi - 4) / (26 - 4)
        zz, yy = np.ogrid[:Z, :Y]
        disk = (zz - cz) ** 2 + (yy - cy) ** 2 <= rad * rad
        lab[:, :, xi][disk] = 1
    return lab


def test_ma_apical_basal_wedge_apical_wider_when_oriented_out():
    """Outward = +X (toward the wide end). The wedge is wider apically, so
    ma_apical_area > ma_basal_area and the ratio > 1."""
    lab = _wedge_cell()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (0.0, 0.0, 1.0))           # +X = apical = the wide end
    compute_ma_apical_basal(lab, df)
    row = df.iloc[0]
    for c in _MA_COLS:
        assert c in df.columns, c
    assert row["ma_apical_area"] > row["ma_basal_area"], (
        row["ma_apical_area"], row["ma_basal_area"])
    assert row["ma_apical_basal_ratio"] > 1.0


def test_ma_apical_basal_sign_flips_with_outward():
    """Flipping the outward normal swaps which end is apical -> the apical/basal areas
    swap and the ratio inverts (the split IS oriented by the outward sign)."""
    lab = _wedge_cell()
    df_pos = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    df_neg = df_pos.copy()
    _inject_outward(df_pos, (0.0, 0.0, 1.0))        # +X apical
    _inject_outward(df_neg, (0.0, 0.0, -1.0))       # −X apical
    compute_ma_apical_basal(lab, df_pos)
    compute_ma_apical_basal(lab, df_neg)
    rp, rn = df_pos.iloc[0], df_neg.iloc[0]
    # apical area under +X outward == basal area under −X outward (same physical end).
    assert rp["ma_apical_area"] == pytest.approx(rn["ma_basal_area"], rel=1e-9)
    assert rp["ma_basal_area"] == pytest.approx(rn["ma_apical_area"], rel=1e-9)
    assert rn["ma_apical_basal_ratio"] < 1.0        # apical (now −X) is the narrow end


def test_ma_apical_basal_nan_untrusted_normal():
    """A zeroed outward confidence -> the apical/basal split is NaN (untrusted frame)."""
    lab = _wedge_cell()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (0.0, 0.0, 1.0), conf=0.0)   # below the confidence floor
    compute_ma_apical_basal(lab, df)
    for c in ("ma_apical_area", "ma_basal_area", "ma_apical_basal_ratio"):
        assert np.isnan(df.iloc[0][c]), c


def test_ma_apical_basal_skips_without_waveC():
    """No tissue_outward_* columns (Wave C did not run) -> SKIP (columns absent)."""
    lab = _wedge_cell()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    compute_ma_apical_basal(lab, df)
    for c in _MA_COLS:
        assert c not in df.columns, c


def test_ma_apical_basal_empty_df_typed_columns():
    df = pd.DataFrame({
        "label": pd.Series([], dtype=np.int64),
        "tissue_outward_z": pd.Series([], dtype=np.float64),
        "tissue_outward_y": pd.Series([], dtype=np.float64),
        "tissue_outward_x": pd.Series([], dtype=np.float64),
        "tissue_outward_confidence": pd.Series([], dtype=np.float64),
        "major_axis_z": pd.Series([], dtype=np.float64),
        "major_axis_y": pd.Series([], dtype=np.float64),
        "major_axis_x": pd.Series([], dtype=np.float64),
        "centroid-0": pd.Series([], dtype=np.float64),
        "centroid-1": pd.Series([], dtype=np.float64),
        "centroid-2": pd.Series([], dtype=np.float64),
    })
    compute_ma_apical_basal(np.zeros((4, 4, 4), np.int32), df)
    for c in _MA_COLS:
        assert c in df.columns, c


# --------------------------------------------------------------------------- #
# 2) apical junction — a hexagonally-packed centre cell -> ~6 lateral sides
# --------------------------------------------------------------------------- #
def _hex_packed_sheet():
    """A single Z-slab of HEXAGONALLY packed cylindrical cells. The centre cell has
    exactly 6 in-plane neighbours; the apical/basal axis is Z (out of the sheet
    plane), so all 6 neighbour contacts are LATERAL -> apical_polygon_sides ≈ 6.
    Returns (labels, centre_label)."""
    Z, Y, X = 9, 64, 64
    lab = np.zeros((Z, Y, X), dtype=np.int32)
    rad = 5.0
    dx = 12.0                       # horizontal centre-to-centre spacing
    dy = dx * np.sqrt(3.0) / 2.0    # row pitch for a hex lattice
    centers = []
    nid = 1
    centre_label = None
    cy0, cx0 = Y / 2.0, X / 2.0
    for row in range(-2, 3):
        yy = cy0 + row * dy
        xoff = (dx / 2.0) if (row % 2) else 0.0
        for col in range(-2, 3):
            xx = cx0 + col * dx + xoff
            if not (rad + 1 <= yy <= Y - rad - 1 and rad + 1 <= xx <= X - rad - 1):
                continue
            yg, xg = np.ogrid[:Y, :X]
            disk2d = ((yg - yy) ** 2 + (xg - xx) ** 2) <= rad * rad   # (Y, X)
            lab[:, disk2d] = nid                                       # extrude over Z
            centers.append((nid, yy, xx))
            if abs(yy - cy0) < dy / 2.0 and abs(xx - cx0) < dx / 4.0:
                centre_label = nid
            nid += 1
    return lab, centre_label


def test_apical_junction_hex_centre_has_six_sides():
    """The centre cell of a hex-packed sheet has 6 lateral (in-plane) neighbours, so
    apical_polygon_sides ≈ 6 (outward = +Z, so the in-plane contacts are lateral)."""
    lab, centre = _hex_packed_sheet()
    assert centre is not None
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))            # outward = +Z (sheet normal)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_apical_junction(lab, df, max_radius=5)
    for c in _AJ_COLS:
        assert c in df.columns, c
    sides = df.set_index("label").loc[centre, "apical_polygon_sides"]
    assert 5 <= sides <= 7, sides                   # ~6 in an ordered hex lattice
    # the junction belt has a real, positive perimeter + a defined irregularity.
    assert df.set_index("label").loc[centre, "apical_junction_length"] > 0.0
    assert np.isfinite(df.set_index("label").loc[centre, "apical_polygon_irregularity"])


def test_apical_junction_in_plane_contacts_are_lateral_not_apical():
    """With outward = +Z the in-plane neighbours are LATERAL, so they ALL count toward
    the polygon — flipping outward to +X (in-plane) would reclassify them. Sanity:
    the centre cell's lateral side count under +Z outward is its full neighbour count."""
    lab, centre = _hex_packed_sheet()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))            # +Z
    qf.compute_neighbor_features(lab, df, max_radius=5)
    nbrs = int(df.set_index("label").loc[centre, "neighbor_count"])
    compute_apical_junction(lab, df, max_radius=5)
    sides = df.set_index("label").loc[centre, "apical_polygon_sides"]
    # every in-plane neighbour is lateral -> sides == the full neighbour count.
    assert sides == pytest.approx(float(nbrs))
    assert nbrs >= 5


def test_apical_junction_nan_without_neighbours():
    """An isolated cell (no graph neighbours) -> NaN apical junction."""
    lab = np.zeros((16, 16, 16), np.int32)
    lab[4:10, 4:10, 4:10] = 1
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_apical_junction(lab, df, max_radius=5)
    assert np.isnan(df.set_index("label").loc[1, "apical_polygon_sides"])


def test_apical_junction_nan_untrusted_normal():
    lab, centre = _hex_packed_sheet()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0), conf=0.0)   # below the floor
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_apical_junction(lab, df, max_radius=5)
    assert np.isnan(df.set_index("label").loc[centre, "apical_polygon_sides"])


def test_apical_junction_skips_without_waveC():
    lab, centre = _hex_packed_sheet()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_apical_junction(lab, df, max_radius=5)   # no outward columns
    for c in _AJ_COLS:
        assert c not in df.columns, c


# --------------------------------------------------------------------------- #
# 3) multi-scale shell SHAPE — sphere (fat core) vs sheet (all rind)
# --------------------------------------------------------------------------- #
def _solid_sphere(R=11):
    shape = (2 * R + 6, 2 * R + 6, 2 * R + 6)
    c = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    lab = np.zeros(shape, dtype=np.int32)
    lab[_ball(R, c, shape)] = 1
    return lab


def _thin_sheet():
    lab = np.zeros((6, 40, 40), dtype=np.int32)
    lab[3, 4:36, 4:36] = 1          # 1-voxel-thick sheet -> no radial core
    return lab


def _thick_sheet(thick=3):
    lab = np.zeros((12, 40, 40), dtype=np.int32)
    lab[4:4 + thick, 4:36, 4:36] = 1
    return lab


def test_multiscale_shape_columns_present_and_bounded():
    lab = _solid_sphere()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    compute_multiscale_shape(lab, df)
    for c in _MS_COLS:
        assert c in df.columns, c
    row = df.iloc[0]
    # the band compactnesses + the surface fraction are bounded ratios in [0,1].
    for c in ("shell_sphericity_inner", "shell_sphericity_outer", "shell_surface_ratio"):
        v = row[c]
        assert np.isnan(v) or (-1e-9 <= v <= 1.0 + 1e-9), (c, v)


def test_multiscale_shape_sphere_has_defined_core():
    """A solid sphere has a real radial core (r>R2 band non-empty), so its inner
    shell compactness is finite + positive, and its outer-rind surface fraction is
    well below 1 (the rind is a fraction of a full surface)."""
    lab = _solid_sphere()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    compute_multiscale_shape(lab, df)
    row = df.iloc[0]
    assert np.isfinite(row["shell_sphericity_inner"])
    assert row["shell_sphericity_inner"] > 0.0
    assert np.isfinite(row["shell_surface_ratio"])
    assert row["shell_surface_ratio"] < 0.95         # a full 3D blob: rind is partial


def test_multiscale_shape_sheet_is_all_rind():
    """A THICK sheet (so a radial field exists) is mostly outer rind: nearly all its
    surface is in the outer band, so shell_surface_ratio is HIGHER than the sphere's
    (a sheet is almost all surface, a sphere has a fat interior)."""
    sphere = _solid_sphere()
    sheet = _thick_sheet(thick=3)
    ds = qf.extract_features(sphere).sort_values("label").reset_index(drop=True)
    dh = qf.extract_features(sheet).sort_values("label").reset_index(drop=True)
    compute_multiscale_shape(sphere, ds)
    compute_multiscale_shape(sheet, dh)
    assert dh.iloc[0]["shell_surface_ratio"] > ds.iloc[0]["shell_surface_ratio"]


def test_multiscale_shape_thin_sheet_is_all_skin_no_outer_band():
    """A 1-voxel-thick sheet has uniform interior depth (every voxel r=1), so the
    OUTER rind band (r<=R1) is EMPTY -> shell_sphericity_outer + the inner/outer
    compactness ratio are NaN (undefined), while shell_surface_ratio == 1.0 (every
    voxel is on the skin). No crash on the degenerate sheet."""
    lab = _thin_sheet()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    compute_multiscale_shape(lab, df)
    row = df.iloc[0]
    assert np.isnan(row["shell_sphericity_outer"])      # no outer band on a 1-vox sheet
    assert np.isnan(row["shell_compactness_ratio"])     # ratio needs both bands
    assert row["shell_surface_ratio"] == pytest.approx(1.0)   # entirely skin
    assert np.isfinite(row["shell_sphericity_inner"])   # the whole sheet is the band


def test_multiscale_shape_empty_df_typed_columns():
    df = pd.DataFrame({"label": pd.Series([], dtype=np.int64)})
    compute_multiscale_shape(np.zeros((4, 4, 4), np.int32), df)
    for c in _MS_COLS:
        assert c in df.columns, c


def test_multiscale_shape_spacing_does_not_crash():
    """Anisotropic spacing affects only the radial partition; the function stays
    well-defined (the face counts are dimensionless)."""
    lab = _solid_sphere(R=9)
    df = qf.extract_features(lab, spacing=(2.0, 1.0, 1.0)).sort_values(
        "label").reset_index(drop=True)
    compute_multiscale_shape(lab, df, spacing=(2.0, 1.0, 1.0))
    assert np.isfinite(df.iloc[0]["shell_surface_ratio"])


# --------------------------------------------------------------------------- #
# 4) opt-in global pass wired into extract_label_features (default-off contract)
# --------------------------------------------------------------------------- #
def _toy_labels():
    """Two touching cells + an isolated one (a tiny tissue with a surface)."""
    vol = np.zeros((10, 18, 36), np.int32)
    vol[2:8, 4:12, 4:12] = 1
    vol[2:8, 4:12, 12:20] = 2
    vol[2:8, 4:12, 26:34] = 3
    return vol


def test_extract_label_features_waveH_global_off_by_default(tmp_path):
    """No Wave H GLOBAL columns (ma_apical_basal / apical_junction) without
    tissue_depth=True; the per-cell multiscale_shape IS always present."""
    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)
    df = extract_label_features(str(p), n_workers=1)
    for c in _ALL_WAVEH_GLOBAL:
        assert c not in df.columns, c
    # multiscale_shape is a per-cell (always-on) family -> present even off the
    # global path.
    for c in _MS_COLS:
        assert c in df.columns, c


def test_extract_label_features_waveH_on_with_tissue_depth(tmp_path):
    """tissue_depth=True (+ adjacency for the junction graph) runs the Wave H global
    passes; the pre-existing columns are byte-identical (Wave H only APPENDS)."""
    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df_off = extract_label_features(str(p), n_workers=1)
    df_on = extract_label_features(
        str(p), n_workers=1, adjacency=True, tissue_depth=True, max_radius=5)

    assert list(df_on["label"].values) == list(df_off["label"].values)
    for c in _ALL_WAVEH_GLOBAL:
        assert c in df_on.columns, c
        assert c not in df_off.columns, c

    # pre-existing columns byte-identical (Wave H global only APPENDS). The per-cell
    # multiscale_shape columns are present in BOTH (always-on), so they are shared.
    shared = [c for c in df_off.columns if c in df_on.columns]
    pd.testing.assert_frame_equal(
        df_on[shared].reset_index(drop=True),
        df_off[shared].reset_index(drop=True),
        check_dtype=False,
    )


def test_ma_apical_basal_runs_without_adjacency(tmp_path):
    """ma_apical_basal needs only the outward normal (no graph) -> present under
    tissue_depth=True even with adjacency=False; apical_junction is present too but
    all-NaN (no graph to find lateral neighbours)."""
    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)
    df = extract_label_features(str(p), n_workers=1, tissue_depth=True)  # adjacency off
    for c in _MA_COLS:
        assert c in df.columns, c
    for c in _AJ_COLS:
        assert c in df.columns, c
    assert df["apical_polygon_sides"].isna().all()   # no adjacency graph


# --------------------------------------------------------------------------- #
# 5) manifest + family registry
# --------------------------------------------------------------------------- #
def test_manifest_tags_all_waveH_features():
    man = fm.get_manifest()
    for c in (*_MA_COLS, *_AJ_COLS, *_MS_COLS):
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert isinstance(tags["bounded"], bool)


def test_manifest_waveH_frames():
    man = fm.get_manifest()
    # the two GLOBAL signed-frame remainders are frame="tissue".
    for c in (*_MA_COLS, *_AJ_COLS):
        assert man[c]["frame"] == "tissue", c
    # the per-cell multiscale shell shape is intrinsic (object-only distance field).
    for c in _MS_COLS:
        assert man[c]["frame"] == "intrinsic", c


def test_manifest_waveH_bounded_flags():
    man = fm.get_manifest()
    for c in ("shell_sphericity_inner", "shell_sphericity_outer", "shell_surface_ratio"):
        assert man[c]["bounded"] is True, c
    for c in ("ma_apical_area", "ma_basal_area", "ma_apical_basal_ratio",
              "ma_apical_taper", "ma_basal_taper",
              "apical_polygon_sides", "apical_junction_length",
              "apical_polygon_irregularity", "shell_compactness_ratio"):
        assert man[c]["bounded"] is False, c


def test_feature_stage_families_registry_includes_waveH():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["ma_apical_basal"]) == set(_MA_COLS)
    assert set(_FEATURE_FAMILIES["apical_junction"]) == set(_AJ_COLS)
    assert set(_FEATURE_FAMILIES["multiscale_shape"]) == set(_MS_COLS)

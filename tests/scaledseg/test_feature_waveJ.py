"""Wave J feature tests — the atlas REMAINDERS that bring those
three families to 100 % of the practical coverage scorecard.

NEW features (NOT a monorepo port) built from the atlas equations. Pure-function
known-geometry tests (no GPU / Cellpose); synthetic arrangements whose answers we
know by construction:

* axis-based neighbour persistence REMAINDER (14 cols extending the Wave A 4):
  a tall Z-major cell with its neighbours clustered at ONE axis end reads a high
  count asymmetry (apical > basal) + a single-span structure (no re-entry); a cell
  with disjoint basal/apical neighbours reads the full basal↔mid↔apical turnover.

* apical/basal/lateral surface partition REMAINDER (14 cols): a columnar BOX
  cell has square apical/basal FOOTPRINTS (aspect ~1, circularity < 1), and a DOME
  cell (domed apical cap, flat basal) reads a higher apical surface ROUGHNESS than
  basal (the dome is far less planar than the flat bottom).

* apical junctional polygon REMAINDER (7 cols): a hexagonal apical FOOTPRINT
  reads ``apical_shape_index`` ≈ 3.7-3.9 + ``apical_junction_vertex_count`` ≈ 6; a
  square footprint reads shape_index = 4 + 4 corners + convexity 1 + angle_cv 0.

* the byte-identical default-off contract (Wave J columns ride the SAME global
  passes as Waves A/D/H — they appear ONLY on the respective ``adjacency=True`` /
  ``tissue_depth=True`` passes; pre-existing columns unchanged) + manifest tags +
  the ``_FEATURE_FAMILIES`` registry.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm
from packages.epicycle_scaledseg.core.qc_features import (
    compute_axis_neighbor_persistence, compute_surface_partition,
    compute_apical_junction, _apical_footprint_polygon, _footprint_metrics,
    AXIS_NEIGHBOR_PERSISTENCE_COLUMNS, SURFACE_PARTITION_COLUMNS,
    APICAL_JUNCTION_COLUMNS,
)


# --------------------------------------------------------------------------- #
# column groups (Wave J = the new tail of each family)
# --------------------------------------------------------------------------- #
_AXIS_J_COLS = (
    "axis_neighbor_persistence_mean", "axis_neighbor_persistence_max",
    "axis_neighbor_persistence_cv",
    "axis_neighbor_birth_median", "axis_neighbor_death_median",
    "axis_neighbor_span_cv",
    "axis_neighbor_events", "axis_neighbor_exchange_rate",
    "axis_neighbor_multispan_fraction", "axis_neighbor_span_count_mean",
    "axis_neighbor_reentry_count",
    "mid_axis_neighbor_count",
    "basal_to_mid_turnover", "mid_to_apical_turnover",
)
_SP_J_COLS = (
    "apical_perimeter", "apical_circularity", "apical_aspect_ratio",
    "apical_compactness",
    "basal_perimeter", "basal_circularity", "basal_aspect_ratio",
    "basal_compactness",
    "apical_curvature_mean", "apical_curvature_std",
    "basal_curvature_mean", "lateral_curvature_mean",
    "apical_surface_roughness", "basal_surface_roughness",
)
_AJ_J_COLS = (
    "apical_junction_vertex_count", "apical_junction_edge_length_cv",
    "apical_vertex_angle_cv",
    "apical_polygon_area", "apical_polygon_perimeter",
    "apical_shape_index", "apical_polygon_convexity",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _inject_outward(df, vec_zyx, conf=1.0):
    z, y, x = vec_zyx
    df["tissue_outward_z"] = float(z)
    df["tissue_outward_y"] = float(y)
    df["tissue_outward_x"] = float(x)
    df["tissue_outward_confidence"] = float(conf)
    return df


def _hex_footprint_labels(rad=14.0, Z=6, pad=8):
    """A hexagonal-prism cell (regular hexagon footprint extruded over Z). The
    apical/basal caps ARE the hexagon."""
    from matplotlib.path import Path
    ang = np.deg2rad(np.arange(0, 360, 60))
    verts2d = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    side = int(2 * rad + 2 * pad)
    cy = cx = side / 2.0
    yy, xx = np.mgrid[:side, :side]
    pts = np.column_stack([xx.ravel() - cx, yy.ravel() - cy])
    mask2d = Path(verts2d).contains_points(pts).reshape(side, side)
    lab = np.zeros((Z, side, side), dtype=np.int32)
    for z in range(1, Z - 1):
        lab[z][mask2d] = 1
    return lab


def _square_footprint_labels(half=12, Z=6, pad=8):
    side = int(2 * half + 2 * pad)
    c = side // 2
    lab = np.zeros((Z, side, side), dtype=np.int32)
    lab[1:Z - 1, c - half:c + half, c - half:c + half] = 1
    return lab


# --------------------------------------------------------------------------- #
# — axis-based neighbour persistence REMAINDER
# --------------------------------------------------------------------------- #
def _tall_cell_with_split_neighbours():
    """A tall Z-major cell (1) with a basal-only neighbour (2, low Z) and an
    apical-only neighbour (3, high Z): disjoint ends -> turnover 1, mid empty."""
    Z, Y, X = 30, 14, 14
    lab = np.zeros((Z, Y, X), dtype=np.int32)
    lab[5:25, 5:9, 5:9] = 1
    lab[5:11, 5:9, 9:11] = 2          # basal end (low Z)
    lab[19:25, 5:9, 9:11] = 3         # apical end (high Z)
    return lab


def _tall_cell_apical_clustered():
    """A tall Z-major cell with BOTH neighbours clustered at the APICAL (high-Z)
    end -> apical count > basal count (asymmetry)."""
    Z, Y, X = 30, 16, 16
    lab = np.zeros((Z, Y, X), dtype=np.int32)
    lab[5:25, 6:10, 6:10] = 1
    lab[18:25, 6:10, 10:12] = 2       # apical, +X side
    lab[18:25, 10:12, 6:10] = 3       # apical, +Y side
    return lab


def test_axis_neighbour_remainder_columns_present():
    lab = _tall_cell_with_split_neighbours()
    df = qf.extract_features(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    compute_axis_neighbor_persistence(lab, df, max_radius=3)
    for c in _AXIS_J_COLS:
        assert c in df.columns, c
    # all 18 columns present (Wave A 4 + Wave J 14).
    for c in AXIS_NEIGHBOR_PERSISTENCE_COLUMNS:
        assert c in df.columns, c


def test_axis_neighbour_split_ends_turnover_and_bins():
    lab = _tall_cell_with_split_neighbours()
    df = qf.extract_features(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    compute_axis_neighbor_persistence(lab, df, max_radius=3)
    by = df.set_index("label")
    assert by.loc[1, "axis_basal_neighbor_count"] == pytest.approx(1.0)
    assert by.loc[1, "axis_apical_neighbor_count"] == pytest.approx(1.0)
    assert by.loc[1, "mid_axis_neighbor_count"] == pytest.approx(0.0)
    # disjoint ends -> full turnover at both bin transitions.
    assert by.loc[1, "axis_neighbor_turnover"] == pytest.approx(1.0)
    assert by.loc[1, "basal_to_mid_turnover"] == pytest.approx(1.0)
    assert by.loc[1, "mid_to_apical_turnover"] == pytest.approx(1.0)


def test_axis_neighbour_count_asymmetry_when_clustered_one_end():
    """Neighbours clustered at the APICAL end -> apical count > basal count."""
    lab = _tall_cell_apical_clustered()
    df = qf.extract_features(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    compute_axis_neighbor_persistence(lab, df, max_radius=3)
    by = df.set_index("label")
    ap = by.loc[1, "axis_apical_neighbor_count"]
    ba = by.loc[1, "axis_basal_neighbor_count"]
    assert ap > ba, (ap, ba)
    assert ap == pytest.approx(2.0)
    assert ba == pytest.approx(0.0)


def test_axis_neighbour_single_span_structure():
    """Each neighbour forms ONE contiguous contact span -> no multispan / re-entry,
    span_count_mean == 1, events == 2*n_neighbours."""
    lab = _tall_cell_with_split_neighbours()
    df = qf.extract_features(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    compute_axis_neighbor_persistence(lab, df, max_radius=3)
    by = df.set_index("label")
    assert by.loc[1, "axis_neighbor_multispan_fraction"] == pytest.approx(0.0)
    assert by.loc[1, "axis_neighbor_span_count_mean"] == pytest.approx(1.0)
    assert by.loc[1, "axis_neighbor_reentry_count"] == pytest.approx(0.0)
    assert by.loc[1, "axis_neighbor_events"] == pytest.approx(4.0)   # 2 nbrs × 2


def test_axis_neighbour_persistence_birth_death_bounded():
    """persistence mean/max + birth/death medians are normalised-t fractions in
    [0,1]; birth <= death."""
    lab = _tall_cell_with_split_neighbours()
    df = qf.extract_features(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    compute_axis_neighbor_persistence(lab, df, max_radius=3)
    by = df.set_index("label")
    for c in ("axis_neighbor_persistence_mean", "axis_neighbor_persistence_max",
              "axis_neighbor_birth_median", "axis_neighbor_death_median"):
        v = by.loc[1, c]
        assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9, (c, v)
    assert by.loc[1, "axis_neighbor_birth_median"] <= by.loc[1, "axis_neighbor_death_median"]
    assert by.loc[1, "axis_neighbor_persistence_max"] >= by.loc[1, "axis_neighbor_persistence_mean"]


def test_axis_neighbour_remainder_nan_without_neighbours():
    """An isolated cell -> NaN persistence stats + 0 bin counts (Wave J)."""
    lab = np.zeros((20, 14, 14), np.int32)
    lab[3:17, 5:9, 5:9] = 1
    df = qf.extract_features(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    compute_axis_neighbor_persistence(lab, df, max_radius=3)
    by = df.set_index("label")
    for c in ("axis_neighbor_persistence_mean", "axis_neighbor_events",
              "axis_neighbor_multispan_fraction", "basal_to_mid_turnover"):
        assert np.isnan(by.loc[1, c]), c
    assert by.loc[1, "mid_axis_neighbor_count"] == pytest.approx(0.0)


def test_axis_neighbour_remainder_missing_axis_cols_nan():
    """Without major_axis_* the whole (extended) family is NaN."""
    df = pd.DataFrame({"label": [1, 2],
                       "centroid-0": [1.0, 2.0], "centroid-1": [1.0, 2.0],
                       "centroid-2": [1.0, 2.0]})
    df.attrs["_adjacency_list"] = [{"label_a": 1, "label_b": 2}]
    compute_axis_neighbor_persistence(np.zeros((4, 4, 4), np.int32), df, max_radius=3)
    for c in AXIS_NEIGHBOR_PERSISTENCE_COLUMNS:
        assert c in df.columns and df[c].isna().all(), c


def test_axis_neighbour_remainder_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    compute_axis_neighbor_persistence(np.zeros((4, 4, 4), np.int32), df, max_radius=3)
    for c in AXIS_NEIGHBOR_PERSISTENCE_COLUMNS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# — surface partition REMAINDER (footprint contour + per-partition shape)
# --------------------------------------------------------------------------- #
def _box_cell():
    """A columnar box cell — flat square apical/basal caps (outward = +Z)."""
    Z, Y, X = 24, 22, 22
    lab = np.zeros((Z, Y, X), np.int32)
    lab[4:20, 6:16, 6:16] = 1
    return lab


def _dome_cell():
    """A dome cell — domed apical (top, high Z), flat basal (bottom). outward=+Z."""
    Z, Y, X = 40, 40, 40
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    cz, cy, cx, R = 10.0, 20.0, 20.0, 14.0
    sphere = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= R * R
    dome = sphere & (zz >= cz)
    lab = np.zeros((Z, Y, X), np.int32)
    lab[dome] = 1
    return lab


def test_surface_partition_remainder_columns_present():
    lab = _box_cell()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))
    compute_surface_partition(lab, df, spacing=(1.0, 1.0, 1.0))
    for c in _SP_J_COLS:
        assert c in df.columns, c
    for c in SURFACE_PARTITION_COLUMNS:
        assert c in df.columns, c


def test_box_footprint_is_square_like():
    """A box cell's apical + basal footprints are squares: aspect ~1, circularity
    in (0,1) (< a circle's 1), apical == basal by symmetry."""
    lab = _box_cell()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))
    compute_surface_partition(lab, df, spacing=(1.0, 1.0, 1.0))
    by = df.set_index("label")
    assert by.loc[1, "apical_aspect_ratio"] == pytest.approx(1.0, abs=0.15)
    assert by.loc[1, "basal_aspect_ratio"] == pytest.approx(1.0, abs=0.15)
    assert 0.6 < by.loc[1, "apical_circularity"] < 1.0
    # symmetric cell -> apical / basal footprints match.
    assert by.loc[1, "apical_perimeter"] == pytest.approx(by.loc[1, "basal_perimeter"], rel=0.05)
    assert by.loc[1, "apical_circularity"] == pytest.approx(by.loc[1, "basal_circularity"], rel=0.05)


def test_dome_apical_rougher_than_basal():
    """A dome cell's APICAL cap (domed) is far less planar than its flat BASAL
    bottom -> apical_surface_roughness > basal_surface_roughness; the per-partition
    curvature columns are present + finite."""
    lab = _dome_cell()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))          # outward +Z = apical (the dome)
    compute_surface_partition(lab, df, spacing=(1.0, 1.0, 1.0))
    by = df.set_index("label")
    ar = by.loc[1, "apical_surface_roughness"]
    br = by.loc[1, "basal_surface_roughness"]
    assert ar > br, (ar, br)
    for c in ("apical_curvature_mean", "apical_curvature_std",
              "basal_curvature_mean", "lateral_curvature_mean"):
        assert np.isfinite(by.loc[1, c]), c


def test_surface_partition_remainder_bounded_circularity():
    """Circularity / compactness are bounded to [0,1]."""
    lab = _box_cell()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))
    compute_surface_partition(lab, df, spacing=(1.0, 1.0, 1.0))
    for c in ("apical_circularity", "apical_compactness",
              "basal_circularity", "basal_compactness"):
        v = df[c].dropna()
        assert (v >= -1e-9).all() and (v <= 1.0 + 1e-9).all(), c


def test_surface_partition_remainder_nan_untrusted_normal():
    lab = _box_cell()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0), conf=0.0)     # below the floor
    compute_surface_partition(lab, df, spacing=(1.0, 1.0, 1.0))
    by = df.set_index("label")
    for c in ("apical_perimeter", "apical_curvature_mean", "apical_surface_roughness"):
        assert np.isnan(by.loc[1, c]), c


def test_surface_partition_remainder_skips_without_waveC():
    lab = _box_cell()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    compute_surface_partition(lab, df, spacing=(1.0, 1.0, 1.0))   # no outward cols
    for c in _SP_J_COLS:
        assert c not in df.columns, c


def test_footprint_metrics_helper_square():
    """The _footprint_metrics helper on a clean square gives aspect ~1."""
    # build a square of 3D vertices in the z=const plane, outward = +Z.
    yy, xx = np.mgrid[0:10, 0:10]
    pts = np.column_stack([np.zeros(yy.size), yy.ravel(), xx.ravel()]).astype(float)
    p, c, ar, cm = _footprint_metrics(pts, np.array([1.0, 0.0, 0.0]))
    assert np.isfinite(p) and p > 0
    assert ar == pytest.approx(1.0, abs=0.1)
    assert 0.6 < c <= 1.0


def test_footprint_metrics_helper_degenerate():
    pts = np.zeros((2, 3))
    out = _footprint_metrics(pts, np.array([1.0, 0.0, 0.0]))
    assert all(np.isnan(x) for x in out)


# --------------------------------------------------------------------------- #
# — apical junctional polygon REMAINDER
# --------------------------------------------------------------------------- #
def test_apical_footprint_polygon_hexagon_shape_index():
    """A hexagonal apical footprint -> shape_index ≈ 3.7-3.9 + vertex_count ≈ 6."""
    lab = _hex_footprint_labels(rad=14.0)
    cap = np.argwhere(lab == 1).astype(np.float64)
    vc, area, perim, si, cx, acv = _apical_footprint_polygon(
        cap, np.array([1.0, 0.0, 0.0]), 1.0)
    assert 5 <= vc <= 7, vc
    assert 3.6 <= si <= 3.95, si           # hexagon ~3.72 (circle 3.545, square 4)
    assert 0.0 < cx <= 1.0
    assert np.isfinite(acv)


def test_apical_footprint_polygon_square_is_regular():
    """A square footprint -> 4 corners, shape_index 4, convexity 1, angle_cv 0."""
    lab = _square_footprint_labels(half=12)
    cap = np.argwhere(lab == 1).astype(np.float64)
    vc, area, perim, si, cx, acv = _apical_footprint_polygon(
        cap, np.array([1.0, 0.0, 0.0]), 1.0)
    assert vc == pytest.approx(4.0)
    assert si == pytest.approx(4.0, abs=0.05)
    assert cx == pytest.approx(1.0, abs=0.02)
    assert acv == pytest.approx(0.0, abs=1e-6)


def test_apical_footprint_polygon_degenerate():
    out = _apical_footprint_polygon(np.zeros((2, 3)), np.array([1.0, 0.0, 0.0]), 1.0)
    assert all(np.isnan(x) for x in out)


def _hex_packed_sheet():
    """A Z-slab of hexagonally-packed cylinders; the centre cell has ~6 in-plane
    (lateral, with outward = +Z) neighbours -> the junction belt is a ~6-gon."""
    Z, Y, X = 9, 64, 64
    lab = np.zeros((Z, Y, X), dtype=np.int32)
    rad, dx = 5.0, 12.0
    dy = dx * np.sqrt(3.0) / 2.0
    nid = 1
    centre = None
    cy0, cx0 = Y / 2.0, X / 2.0
    for row in range(-2, 3):
        yy = cy0 + row * dy
        xoff = (dx / 2.0) if (row % 2) else 0.0
        for col in range(-2, 3):
            xx = cx0 + col * dx + xoff
            if not (rad + 1 <= yy <= Y - rad - 1 and rad + 1 <= xx <= X - rad - 1):
                continue
            yg, xg = np.ogrid[:Y, :X]
            disk = ((yg - yy) ** 2 + (xg - xx) ** 2) <= rad * rad
            lab[:, disk] = nid
            if abs(yy - cy0) < dy / 2.0 and abs(xx - cx0) < dx / 4.0:
                centre = nid
            nid += 1
    return lab, centre


def test_apical_junction_remainder_columns_present():
    lab, centre = _hex_packed_sheet()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_apical_junction(lab, df, max_radius=5)
    for c in _AJ_J_COLS:
        assert c in df.columns, c
    for c in APICAL_JUNCTION_COLUMNS:
        assert c in df.columns, c


def test_apical_junction_belt_vertex_count_matches_sides():
    """The junction vertex count == the lateral-neighbour (polygon side) count, and
    the belt is near-regular (low edge/angle cv)."""
    lab, centre = _hex_packed_sheet()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_apical_junction(lab, df, max_radius=5)
    by = df.set_index("label")
    assert by.loc[centre, "apical_junction_vertex_count"] == pytest.approx(
        by.loc[centre, "apical_polygon_sides"])
    assert by.loc[centre, "apical_junction_vertex_count"] >= 5
    # a regular hex belt -> small edge-length + angle cv.
    assert by.loc[centre, "apical_junction_edge_length_cv"] < 0.4
    assert np.isfinite(by.loc[centre, "apical_vertex_angle_cv"])


def test_apical_junction_footprint_polygon_present():
    """The explicit apical FOOTPRINT polygon (area / perimeter / shape_index /
    convexity) is computed for a real cell (the centre's apical cap is a disc)."""
    lab, centre = _hex_packed_sheet()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_apical_junction(lab, df, max_radius=5)
    by = df.set_index("label")
    assert by.loc[centre, "apical_polygon_area"] > 0.0
    assert by.loc[centre, "apical_polygon_perimeter"] > 0.0
    # a disc cap -> shape_index near the circle value (3.545); always >= ~3.5.
    assert by.loc[centre, "apical_shape_index"] >= 3.4
    assert 0.0 < by.loc[centre, "apical_polygon_convexity"] <= 1.0


def test_apical_junction_remainder_nan_without_neighbours():
    lab = np.zeros((16, 16, 16), np.int32)
    lab[4:10, 4:10, 4:10] = 1
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0))
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_apical_junction(lab, df, max_radius=5)
    by = df.set_index("label")
    assert np.isnan(by.loc[1, "apical_junction_vertex_count"])


def test_apical_junction_remainder_nan_untrusted_normal():
    lab, centre = _hex_packed_sheet()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    _inject_outward(df, (1.0, 0.0, 0.0), conf=0.0)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_apical_junction(lab, df, max_radius=5)
    assert np.isnan(df.set_index("label").loc[centre, "apical_junction_vertex_count"])


def test_apical_junction_remainder_skips_without_waveC():
    lab, centre = _hex_packed_sheet()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_apical_junction(lab, df, max_radius=5)   # no outward columns
    for c in _AJ_J_COLS:
        assert c not in df.columns, c


# --------------------------------------------------------------------------- #
# default-off contract through extract_label_features
# --------------------------------------------------------------------------- #
def test_default_off_no_waveJ_columns_without_passes(tmp_path):
    """A plain extract (adjacency off, tissue_depth off) emits NONE of the Wave J
    GLOBAL columns."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    lab = _box_cell()
    p = tmp_path / "labels.npy"
    np.save(p, lab)
    df = extract_label_features(str(p), spacing=(1.0, 1.0, 1.0))
    for c in (*_AXIS_J_COLS, *_SP_J_COLS, *_AJ_J_COLS):
        assert c not in df.columns, c


# --------------------------------------------------------------------------- #
# manifest + families registry
# --------------------------------------------------------------------------- #
def test_manifest_tags_all_waveJ_features():
    man = fm.get_manifest()
    for c in (*_AXIS_J_COLS, *_SP_J_COLS, *_AJ_J_COLS):
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert isinstance(tags["bounded"], bool)


def test_manifest_waveJ_frames():
    man = fm.get_manifest()
    for c in _AXIS_J_COLS:
        assert man[c]["frame"] == "intrinsic", c     # principal-axis family
    for c in (*_SP_J_COLS, *_AJ_J_COLS):
        assert man[c]["frame"] == "tissue", c         # signed apical/basal frame


def test_manifest_waveJ_bounded_flags():
    man = fm.get_manifest()
    for c in ("apical_circularity", "apical_compactness",
              "basal_circularity", "basal_compactness",
              "apical_polygon_convexity",
              "axis_neighbor_persistence_mean", "axis_neighbor_multispan_fraction",
              "basal_to_mid_turnover", "mid_to_apical_turnover"):
        assert man[c]["bounded"] is True, c
    for c in ("apical_perimeter", "apical_aspect_ratio", "apical_polygon_area",
              "apical_shape_index", "axis_neighbor_events",
              "axis_neighbor_reentry_count", "apical_curvature_mean"):
        assert man[c]["bounded"] is False, c


def test_feature_stage_families_registry_includes_waveJ():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["axis_neighbor_persistence"]) == set(
        AXIS_NEIGHBOR_PERSISTENCE_COLUMNS)
    assert set(_FEATURE_FAMILIES["surface_partition"]) == set(SURFACE_PARTITION_COLUMNS)
    assert set(_FEATURE_FAMILIES["apical_junction"]) == set(APICAL_JUNCTION_COLUMNS)

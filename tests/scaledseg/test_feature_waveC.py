"""Wave C feature tests — tissue-depth ("in/out") + spatial context +
the per-cell OUTWARD normal (atlas ; the C-W tissue-depth primitive).

NEW features (NOT a monorepo port) built from the atlas equations. Pure-function
known-geometry tests (no GPU / Cellpose); synthetic arrangements whose answers we
know by construction:

* a SPHERE OF CELLS (a ball of labels carved into cube cells): the outer-shell
  cells get SMALL ``distance_to_tissue_surface`` and their ``tissue_outward_*``
  points RADIALLY OUT (away from the ball centre); the core cell gets a LARGE depth
  and a LOW ``tissue_outward_confidence`` (the flat-core gradient is ambiguous).
* a FLAT SLAB of cells (the geometry caveat): the top + bottom cells are SHALLOW
  (close to a face), a mid-slab cell is the deepest.
* an ANISOTROPIC-SPACING check: the tissue-EDT respects ``sampling=spacing`` so the
  depth scales with the spacing in the thin direction.
* the byte-identical ``tissue_depth`` default-off contract (Wave C columns appear
  ONLY on ``tissue_depth=True``; the pre-existing morphology columns are unchanged).

Plus the manifest tags, the ``_FEATURE_FAMILIES`` registry, and the empty/no-tissue
guards.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm
from packages.epicycle_scaledseg.core.qc_features import compute_tissue_depth, TISSUE_DEPTH_COLUMNS


# --------------------------------------------------------------------------- #
# column groups
# --------------------------------------------------------------------------- #
_DEPTH_COLS = (
    "distance_to_tissue_surface", "tissue_depth_mean", "tissue_depth_max",
    "normalized_tissue_depth", "at_fov_border",
)
_CONTEXT_COLS = (
    "centroid_z_relative", "centroid_y_relative", "centroid_x_relative",
    "distance_to_tissue_centroid", "radial_tissue_position",
    # Wave I remainder — joined the "spatial_context" family.
    "distance_to_boundary_face_min",
    # Mop-up O remainder — the named per-cell local tissue depth.
    "local_tissue_depth",
)
_OUTWARD_COLS = (
    "tissue_outward_z", "tissue_outward_y", "tissue_outward_x",
    "tissue_outward_confidence",
)
_ALL_WAVEC = (*_DEPTH_COLS, *_CONTEXT_COLS, *_OUTWARD_COLS)


# --------------------------------------------------------------------------- #
# shape builders
# --------------------------------------------------------------------------- #
def _ball(R, c, shape):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= R * R


def _sphere_of_cells(shape=(40, 40, 40), R=18, cube=8):
    """A solid ball partitioned into cube cells (a 'sphere of cells').

    Tile the volume into ``cube``-sized blocks, intersect each block with a
    centred ball of radius ``R``, and keep every block that has enough voxels in
    the ball as a distinct label. Returns (labels, centre, outer_ids, core_id):

    * ``centre`` — the ball centre (z, y, x) voxel coords.
    * ``outer_ids`` — labels whose block centre is in the OUTER shell of the ball
      (block-centre radius > 0.6·R).
    * ``core_id`` — the single label whose block straddles the ball centre.
    """
    lab = np.zeros(shape, dtype=np.int32)
    c = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 2], dtype=np.float64)
    ball = _ball(R, c, shape)
    next_id = 1
    outer_ids, core_id = [], None
    block_info = {}
    for z0 in range(0, shape[0], cube):
        for y0 in range(0, shape[1], cube):
            for x0 in range(0, shape[2], cube):
                block = np.zeros(shape, dtype=bool)
                block[z0:z0 + cube, y0:y0 + cube, x0:x0 + cube] = True
                cell = block & ball
                if cell.sum() < 20:                  # too small a sliver -> skip
                    continue
                lab[cell] = next_id
                bc = np.array([z0 + cube / 2, y0 + cube / 2, x0 + cube / 2])
                block_info[next_id] = bc
                next_id += 1
    for lid, bc in block_info.items():
        r = float(np.linalg.norm(bc - c))
        if r > 0.6 * R:
            outer_ids.append(lid)
        if r < 0.5 * cube:                           # block straddling the centre
            core_id = lid
    return lab, c, outer_ids, core_id


def _flat_slab(shape=(24, 60, 60), slab_z=(4, 20), n_cells_z=4, cube=10):
    """A flat slab tiled into cube cells, with BACKGROUND PADDING above + below in Z
    (the slab occupies only ``slab_z`` of the Z extent) so the top + bottom faces
    are real tissue surfaces. The slab is stacked ``n_cells_z`` cells through Z and
    tiled laterally; lateral cells sit well INSIDE the YX margins so the limiting
    surface for an interior column is the Z top/bottom face, NOT a lateral edge.

    Returns (labels, top_ids, bottom_ids, mid_ids) by Z band so the caller can
    assert the top/bottom bands are SHALLOW (near a Z face) and the middle band is
    deepest.
    """
    lab = np.zeros(shape, dtype=np.int32)
    z0_slab, z1_slab = slab_z
    z_edges = np.linspace(z0_slab, z1_slab, n_cells_z + 1).astype(int)
    next_id = 1
    bands = {b: [] for b in range(n_cells_z)}
    margin = 14                                      # wide lateral margin (>> slab thickness)
    for b in range(n_cells_z):
        z0, z1 = z_edges[b], z_edges[b + 1]
        for y0 in range(margin, shape[1] - margin, cube):
            for x0 in range(margin, shape[2] - margin, cube):
                lab[z0:z1, y0:y0 + cube, x0:x0 + cube] = next_id
                bands[b].append(next_id)
                next_id += 1
    top = bands[0]
    bottom = bands[n_cells_z - 1]
    mid = bands[n_cells_z // 2]
    return lab, top, bottom, mid


def _feats(lab, spacing=None):
    """Run the engine's per-cell extract + (optionally spacing-aware) centroids so
    the spatial-context + outward-normal paths have ``centroid-*`` columns."""
    df = qf.extract_features(lab, spacing=spacing)
    return df.sort_values("label").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1) sphere of cells — outer shallow + radial-out normal, core deep + low conf
# --------------------------------------------------------------------------- #
def test_sphere_outer_cells_are_shallow_core_is_deep():
    lab, c, outer_ids, core_id = _sphere_of_cells()
    assert len(outer_ids) >= 4 and core_id is not None
    df = _feats(lab)
    compute_tissue_depth(lab, df, spacing=None)
    by = df.set_index("label")

    # every Wave C column present + finite/typed.
    for col in _ALL_WAVEC:
        assert col in df.columns, col

    core_depth = by.loc[core_id, "tissue_depth_mean"]
    outer_depths = [by.loc[i, "tissue_depth_mean"] for i in outer_ids]
    # the core cell is DEEPER than every outer-shell cell.
    assert core_depth > max(outer_depths), (core_depth, max(outer_depths))
    # outer cells reach close to a surface; the core's nearest surface is far.
    assert by.loc[core_id, "distance_to_tissue_surface"] > max(
        by.loc[i, "distance_to_tissue_surface"] for i in outer_ids)
    # normalized depth is bounded [0,1]; the core is near the high end.
    nd = by["normalized_tissue_depth"].dropna()
    assert (nd >= -1e-9).all() and (nd <= 1.0 + 1e-9).all()
    assert by.loc[core_id, "normalized_tissue_depth"] > 0.5


def test_sphere_outer_normal_points_radially_out():
    lab, c, outer_ids, core_id = _sphere_of_cells()
    df = _feats(lab)
    compute_tissue_depth(lab, df, spacing=None)
    by = df.set_index("label")

    # for an outer-shell cell the outward normal should agree with the radial
    # direction (centroid - ball centre): a positive, near-1 cosine.
    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    n_good = 0
    for lid in outer_ids:
        outward = np.array([by.loc[lid, "tissue_outward_z"],
                            by.loc[lid, "tissue_outward_y"],
                            by.loc[lid, "tissue_outward_x"]], dtype=float)
        if not np.all(np.isfinite(outward)):
            continue
        # unit vector
        assert abs(np.linalg.norm(outward) - 1.0) < 1e-6
        radial = by.loc[lid, cen_cols].values.astype(float) - c
        rn = np.linalg.norm(radial)
        if rn < 1e-9:
            continue
        cos = float(np.dot(outward, radial / rn))
        assert cos > 0.55, (lid, cos)               # points outward, not inward
        n_good += 1
    assert n_good >= 4


def test_sphere_core_confidence_low_outer_confidence_higher():
    lab, c, outer_ids, core_id = _sphere_of_cells()
    df = _feats(lab)
    compute_tissue_depth(lab, df, spacing=None)
    by = df.set_index("label")
    conf = by["tissue_outward_confidence"].dropna()
    assert (conf >= -1e-9).all() and (conf <= 1.0 + 1e-9).all()
    # the deep core direction is ambiguous (small gradient) -> low confidence vs
    # the mean outer-shell confidence (a sharp, well-defined surface direction).
    core_conf = by.loc[core_id, "tissue_outward_confidence"]
    outer_conf = np.mean([by.loc[i, "tissue_outward_confidence"] for i in outer_ids])
    assert core_conf < outer_conf, (core_conf, outer_conf)


def test_sphere_radial_tissue_position_bounded_outer_larger():
    lab, c, outer_ids, core_id = _sphere_of_cells()
    df = _feats(lab)
    compute_tissue_depth(lab, df, spacing=None)
    by = df.set_index("label")
    rad = by["radial_tissue_position"].dropna()
    assert (rad >= -1e-9).all() and (rad <= 1.0 + 1e-9).all()
    # outer cells are farther from the tissue centroid than the core cell.
    assert np.mean([by.loc[i, "radial_tissue_position"] for i in outer_ids]) > \
        by.loc[core_id, "radial_tissue_position"]
    # the relative centroids live in [0,1].
    for col in ("centroid_z_relative", "centroid_y_relative", "centroid_x_relative"):
        v = by[col].dropna()
        assert (v >= -1e-9).all() and (v <= 1.0 + 1e-9).all()


# --------------------------------------------------------------------------- #
# 2) flat slab — geometry caveat: top/bottom shallow, middle deepest
# --------------------------------------------------------------------------- #
def test_flat_slab_top_bottom_shallow_middle_deep():
    lab, top, bottom, mid = _flat_slab()
    df = _feats(lab)
    compute_tissue_depth(lab, df, spacing=None)
    by = df.set_index("label")

    top_depth = np.mean([by.loc[i, "tissue_depth_mean"] for i in top])
    bot_depth = np.mean([by.loc[i, "tissue_depth_mean"] for i in bottom])
    mid_depth = np.mean([by.loc[i, "tissue_depth_mean"] for i in mid])
    # the mid Z band is deeper than the top + bottom bands (the slab's two faces).
    assert mid_depth > top_depth, (mid_depth, top_depth)
    assert mid_depth > bot_depth, (mid_depth, bot_depth)


def test_flat_slab_interior_cells_not_border_flagged():
    # the padded slab sits well inside the array (Z padding + wide lateral margin),
    # so NO cell touches an array face -> at_fov_border == 0 for all.
    lab, top, bottom, mid = _flat_slab()
    df = _feats(lab)
    compute_tissue_depth(lab, df, spacing=None)
    by = df.set_index("label")
    assert (by["at_fov_border"] == 0.0).all()


def test_at_fov_border_flags_face_touching_cells():
    # a cell that reaches the z==0 array face IS flagged; an interior cell is not.
    lab = np.zeros((16, 16, 16), np.int32)
    lab[0:5, 4:12, 4:12] = 1                          # touches the z==0 face
    lab[8:13, 4:12, 4:12] = 2                         # interior in Z
    df = _feats(lab)
    compute_tissue_depth(lab, df, spacing=None)
    by = df.set_index("label")
    assert by.loc[1, "at_fov_border"] == 1.0
    assert by.loc[2, "at_fov_border"] == 0.0
    assert set(np.unique(by["at_fov_border"].values)).issubset({0.0, 1.0})


# --------------------------------------------------------------------------- #
# 3) anisotropic spacing — the EDT respects sampling=spacing
# --------------------------------------------------------------------------- #
def test_anisotropic_spacing_scales_depth():
    # a block THIN in Z (the limiting / shortest dimension), wide in YX: the
    # max-depth voxel's nearest surface is reached THROUGH Z, so a large Z spacing
    # inflates that physical distance (the field is no longer in voxel units).
    lab = np.zeros((24, 40, 40), np.int32)
    lab[10:14, 6:34, 6:34] = 1                        # 4 thick in Z, 28 wide in YX
    df_iso = _feats(lab)
    compute_tissue_depth(lab, df_iso, spacing=None)
    iso_depth = float(df_iso.set_index("label").loc[1, "tissue_depth_max"])

    df_aniso = _feats(lab, spacing=(3.0, 1.0, 1.0))
    compute_tissue_depth(lab, df_aniso, spacing=(3.0, 1.0, 1.0))
    aniso_depth = float(df_aniso.set_index("label").loc[1, "tissue_depth_max"])
    # large Z spacing inflates the through-Z depth -> aniso max depth > iso.
    assert aniso_depth > iso_depth, (aniso_depth, iso_depth)
    assert aniso_depth > 0 and iso_depth > 0


def test_spacing_makes_distance_to_centroid_physical():
    lab, c, outer_ids, core_id = _sphere_of_cells()
    df_vox = _feats(lab)
    compute_tissue_depth(lab, df_vox, spacing=None)
    df_phys = _feats(lab, spacing=(2.0, 1.0, 1.0))
    compute_tissue_depth(lab, df_phys, spacing=(2.0, 1.0, 1.0))
    # an anisotropic spacing changes the physical distance-to-centroid field
    # relative to the voxel one (Z stretched 2x).
    dv = df_vox.set_index("label")["distance_to_tissue_centroid"]
    dp = df_phys.set_index("label")["distance_to_tissue_centroid"]
    assert not np.allclose(dv.values, dp.values)


# --------------------------------------------------------------------------- #
# 4) guards
# --------------------------------------------------------------------------- #
def test_empty_df_gets_typed_columns():
    df = pd.DataFrame()
    compute_tissue_depth(np.zeros((4, 4, 4), np.int32), df)
    for c in TISSUE_DEPTH_COLUMNS:
        assert c in df.columns, c


def test_no_tissue_all_nan_or_zero():
    df = pd.DataFrame({"label": [1, 2], "centroid-0": [1.0, 2.0],
                       "centroid-1": [1.0, 2.0], "centroid-2": [1.0, 2.0]})
    compute_tissue_depth(np.zeros((6, 6, 6), np.int32), df)
    # no foreground -> depth NaN, border flag 0.
    assert df["tissue_depth_mean"].isna().all()
    assert (df["at_fov_border"] == 0.0).all()


def test_missing_centroid_columns_context_nan_depth_ok():
    """Without centroid columns the spatial-context + outward-normal columns are
    NaN, but the tissue-EDT depth columns (which only need find_objects) are OK."""
    lab = np.zeros((16, 16, 16), np.int32)
    lab[2:8, 2:8, 2:8] = 1
    lab[2:8, 2:8, 9:15] = 2
    df = pd.DataFrame({"label": [1, 2]})              # no centroid-* columns
    compute_tissue_depth(lab, df)
    assert df["tissue_depth_mean"].notna().all()      # depth defined
    assert df["centroid_z_relative"].isna().all()     # context undefined
    assert df["tissue_outward_z"].isna().all()        # outward normal undefined


# --------------------------------------------------------------------------- #
# 5) opt-in global pass wired into extract_label_features
# --------------------------------------------------------------------------- #
def _toy_labels():
    """Two touching cells + an isolated one (a tiny tissue with a surface)."""
    vol = np.zeros((10, 18, 36), np.int32)
    vol[2:8, 4:12, 4:12] = 1
    vol[2:8, 4:12, 12:20] = 2
    vol[2:8, 4:12, 26:34] = 3
    return vol


def test_extract_label_features_tissue_depth_off_no_waveC(tmp_path):
    """``tissue_depth=False`` (default) emits NO Wave C columns."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)
    df = extract_label_features(str(p), n_workers=1)
    for c in _ALL_WAVEC:
        assert c not in df.columns, c


def test_extract_label_features_tissue_depth_on_adds_waveC(tmp_path):
    """``tissue_depth=True`` runs the Wave C pass: the columns appear and the
    pre-existing morphology columns are byte-identical (additive contract)."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df_off = extract_label_features(str(p), n_workers=1)
    df_on = extract_label_features(str(p), n_workers=1, tissue_depth=True)

    assert list(df_on["label"].values) == list(df_off["label"].values)
    for c in _ALL_WAVEC:
        assert c in df_on.columns, c
        assert c not in df_off.columns, c

    # pre-existing columns byte-identical (Wave C only APPENDS).
    shared = [c for c in df_off.columns if c in df_on.columns]
    pd.testing.assert_frame_equal(
        df_on[shared].reset_index(drop=True),
        df_off[shared].reset_index(drop=True),
        check_dtype=False,
    )


def test_extract_label_features_tissue_depth_independent_of_adjacency(tmp_path):
    """Wave C runs with adjacency=False (a whole-tissue field, not graph-derived)
    and ALSO alongside adjacency=True (shared single full_labels load)."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    # tissue_depth alone (adjacency off) still emits Wave C, NO adjacency cols.
    df_td = extract_label_features(str(p), n_workers=1, tissue_depth=True)
    for c in _ALL_WAVEC:
        assert c in df_td.columns, c
    assert "neighbor_count" not in df_td.columns

    # both on -> both column sets present.
    df_both = extract_label_features(
        str(p), n_workers=1, adjacency=True, tissue_depth=True, max_radius=5)
    for c in _ALL_WAVEC:
        assert c in df_both.columns, c
    assert "neighbor_count" in df_both.columns


def test_extract_label_features_tissue_depth_from_array():
    """``tissue_depth=True`` works on an IN-MEMORY array input too."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    vol = _toy_labels()
    df = extract_label_features(vol, n_workers=1, tissue_depth=True)
    for c in _ALL_WAVEC:
        assert c in df.columns, c


# --------------------------------------------------------------------------- #
# 6) manifest + family registry
# --------------------------------------------------------------------------- #
def test_manifest_tags_all_waveC_features():
    man = fm.get_manifest()
    for c in _ALL_WAVEC:
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] == "tissue", c          # whole-tissue frame
        assert isinstance(tags["bounded"], bool)


def test_manifest_waveC_bounded_flags():
    man = fm.get_manifest()
    # bounded [0,1]/{0,1}/[-1,1] columns
    for c in ("normalized_tissue_depth", "at_fov_border",
              "centroid_z_relative", "centroid_y_relative", "centroid_x_relative",
              "radial_tissue_position",
              "tissue_outward_z", "tissue_outward_y", "tissue_outward_x",
              "tissue_outward_confidence"):
        assert man[c]["bounded"] is True, c
    # unbounded length / depth columns
    for c in ("distance_to_tissue_surface", "tissue_depth_mean",
              "tissue_depth_max", "distance_to_tissue_centroid"):
        assert man[c]["bounded"] is False, c


def test_feature_stage_families_registry_includes_waveC():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["tissue_depth"]) == set(_DEPTH_COLS + _OUTWARD_COLS)
    assert set(_FEATURE_FAMILIES["spatial_context"]) == set(_CONTEXT_COLS)

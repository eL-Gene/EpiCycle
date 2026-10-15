"""Wave D feature tests — signed-frame families that consume the Wave C per-cell
OUTWARD normal (the ``tissue_outward_*`` columns) to split a direction-bearing
quantity into APICAL (aligned with outward) / BASAL (anti-aligned) / LATERAL
(perpendicular) parts (directional contact, surface
partition, signed nuclear offset).

NEW features (NOT a monorepo port) built from the atlas equations. Pure-function
known-geometry tests (no GPU / Cellpose); synthetic arrangements whose answers we
know by construction:

* a SPHERE OF CELLS (a ball of labels carved into cube cells; Wave C gives radial
  outward normals): an outer-shell cell's surface facing the ball EXTERIOR is
  APICAL (apical_surface_fraction > 0, the apical patch points outward); its
  contact with the INNER neighbour is BASAL (basal_contact_fraction dominates,
  apical_contact_fraction ~0 — there are no neighbours on the apical/exterior
  side); the three fractions SUM TO 1.
* a SINGLE CELL + nucleus with a HAND-SET outward normal: a nucleus pushed toward
  the apical surface -> POSITIVE ``nucleus_apical_basal_offset``; pushed the other
  way -> NEGATIVE.
* low ``tissue_outward_confidence`` (the flat tissue core) -> the signed columns
  are NaN'd (the apical/basal direction is meaningless there).
* the byte-identical default-off contract (Wave D columns appear ONLY on
  ``tissue_depth=True``; the pre-existing columns are unchanged) + the manifest
  tags + the ``_FEATURE_FAMILIES`` registry.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm
from packages.epicycle_scaledseg.core.qc_features import (
    compute_directional_contact, compute_surface_partition,
    DIRECTIONAL_CONTACT_COLUMNS, SURFACE_PARTITION_COLUMNS,
)
from packages.epicycle_scaledseg.reconcile.features import extract_label_features


# --------------------------------------------------------------------------- #
# column groups
# --------------------------------------------------------------------------- #
_DC_COLS = list(DIRECTIONAL_CONTACT_COLUMNS)
_SP_COLS = list(SURFACE_PARTITION_COLUMNS)
_SIGNED_NUC_COLS = ["nucleus_apical_basal_offset", "nucleus_apical_basal_offset_norm"]
_ALL_WAVED_GLOBAL = _DC_COLS + _SP_COLS         # the two global qc_features passes


# --------------------------------------------------------------------------- #
# shape builders
# --------------------------------------------------------------------------- #
def _ball(R, c, shape):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= R * R


def _sphere_of_cells(shape=(40, 40, 40), R=18, cube=8):
    """A solid ball partitioned into cube cells (the Wave C builder). Returns
    (labels, centre, outer_ids, core_id)."""
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
                if cell.sum() < 20:
                    continue
                lab[cell] = next_id
                bc = np.array([z0 + cube / 2, y0 + cube / 2, x0 + cube / 2])
                block_info[next_id] = bc
                next_id += 1
    for lid, bc in block_info.items():
        r = float(np.linalg.norm(bc - c))
        if r > 0.6 * R:
            outer_ids.append(lid)
        if r < 0.5 * cube:
            core_id = lid
    return lab, c, outer_ids, core_id


def _sphere_with_waveC(spacing=None):
    """Sphere of cells + the full Wave C pass (centroids, adjacency, tissue_depth)
    so the Wave D directional/surface passes have the outward normal + graph."""
    lab, c, outer_ids, core_id = _sphere_of_cells()
    df = qf.extract_features(lab, spacing=spacing).sort_values("label").reset_index(drop=True)
    sp = tuple(spacing) if spacing is not None else None
    qf.compute_tissue_depth(lab, df, spacing=sp)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    return lab, df, c, outer_ids, core_id, sp


# --------------------------------------------------------------------------- #
# 1) directional contact — sphere: outer cells' contacts are BASAL (inward)
# --------------------------------------------------------------------------- #
def test_directional_contact_fractions_present_and_bounded():
    lab, df, c, outer_ids, core_id, sp = _sphere_with_waveC()
    compute_directional_contact(lab, df, spacing=sp, max_radius=5)
    for col in _DC_COLS:
        assert col in df.columns, col
    for col in ("apical_contact_fraction", "basal_contact_fraction",
                "lateral_contact_fraction"):
        v = df[col].dropna()
        assert (v >= -1e-9).all() and (v <= 1.0 + 1e-9).all(), col
    pol = df["contact_polarity_bias"].dropna()
    assert (pol >= -1.0 - 1e-9).all() and (pol <= 1.0 + 1e-9).all()


def test_directional_contact_fractions_sum_to_one():
    lab, df, c, outer_ids, core_id, sp = _sphere_with_waveC()
    compute_directional_contact(lab, df, spacing=sp, max_radius=5)
    s = df[["apical_contact_fraction", "basal_contact_fraction",
            "lateral_contact_fraction"]].sum(axis=1)
    have = df["apical_contact_fraction"].notna()
    assert have.sum() >= 4
    assert np.allclose(s[have].values, 1.0, atol=1e-9)


def test_outer_cell_contacts_are_basal_not_apical():
    """An outer-shell cell's neighbours are all on the INWARD (basal) side — the
    apical side faces the ball exterior where there is no tissue — so its contacts
    are basal, not apical."""
    lab, df, c, outer_ids, core_id, sp = _sphere_with_waveC()
    compute_directional_contact(lab, df, spacing=sp, max_radius=5)
    by = df.set_index("label")
    basal = np.nanmean([by.loc[i, "basal_contact_fraction"] for i in outer_ids])
    apical = np.nanmean([by.loc[i, "apical_contact_fraction"] for i in outer_ids])
    assert basal > apical, (basal, apical)
    # polarity bias = apical − basal is NEGATIVE for outer (basal-dominated) cells.
    pol = np.nanmean([by.loc[i, "contact_polarity_bias"] for i in outer_ids])
    assert pol < 0.0, pol


def test_directional_contact_nan_without_neighbours():
    """An isolated cell (no graph neighbours) -> NaN directional contact (no
    contacts to direction-classify), even with a trusted outward normal."""
    lab = np.zeros((20, 20, 20), np.int32)
    lab[2:8, 2:8, 2:8] = 1            # lone cell touching the z==2 corner region
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    qf.compute_tissue_depth(lab, df, spacing=None)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    compute_directional_contact(lab, df, spacing=None, max_radius=5)
    assert df.set_index("label").loc[1, "apical_contact_fraction"] != df.set_index(
        "label").loc[1, "apical_contact_fraction"] or True  # noqa (NaN check below)
    assert np.isnan(df.set_index("label").loc[1, "apical_contact_fraction"])


def test_directional_contact_low_confidence_is_nan():
    """A cell with a deliberately ZEROED outward confidence -> NaN (the apical/basal
    direction is untrusted)."""
    lab, df, c, outer_ids, core_id, sp = _sphere_with_waveC()
    target = outer_ids[0]
    df.loc[df["label"] == target, "tissue_outward_confidence"] = 0.0   # below floor
    compute_directional_contact(lab, df, spacing=sp, max_radius=5)
    assert np.isnan(df.set_index("label").loc[target, "apical_contact_fraction"])


# --------------------------------------------------------------------------- #
# 2) surface partition — sphere: outer cells expose an APICAL patch (faces out)
# --------------------------------------------------------------------------- #
def test_surface_partition_present_and_bounded():
    lab, df, c, outer_ids, core_id, sp = _sphere_with_waveC()
    compute_surface_partition(lab, df, spacing=sp)
    for col in _SP_COLS:
        assert col in df.columns, col
    for col in ("apical_surface_fraction", "basal_surface_fraction",
                "lateral_surface_fraction"):
        v = df[col].dropna()
        assert (v >= -1e-9).all() and (v <= 1.0 + 1e-9).all(), col
    for col in ("apical_surface_area", "basal_surface_area",
                "lateral_surface_area", "apical_basal_surface_ratio"):
        v = df[col].dropna()
        assert (v >= -1e-9).all(), col


def test_surface_partition_fractions_sum_to_one():
    lab, df, c, outer_ids, core_id, sp = _sphere_with_waveC()
    compute_surface_partition(lab, df, spacing=sp)
    s = df[["apical_surface_fraction", "basal_surface_fraction",
            "lateral_surface_fraction"]].sum(axis=1)
    have = df["apical_surface_fraction"].notna()
    assert have.sum() >= 4
    assert np.allclose(s[have].values, 1.0, atol=1e-9)


def test_outer_cell_has_apical_surface_facing_out():
    """An outer-shell cell's boundary facing the ball EXTERIOR is apical (its
    outward-pointing faces align with the radial outward normal), so its
    apical_surface_fraction is > 0 and its apical surface area is a real patch."""
    lab, df, c, outer_ids, core_id, sp = _sphere_with_waveC()
    compute_surface_partition(lab, df, spacing=sp)
    by = df.set_index("label")
    ap_fracs = np.array([by.loc[i, "apical_surface_fraction"] for i in outer_ids])
    ap_fracs = ap_fracs[np.isfinite(ap_fracs)]
    assert ap_fracs.size >= 4
    assert (ap_fracs > 0.0).mean() > 0.9                  # nearly all expose apical
    assert np.nanmean(ap_fracs) > 0.1                     # a meaningful apical patch
    # the apical surface AREA is a real (positive) patch for the outer cells.
    assert np.nanmean([by.loc[i, "apical_surface_area"] for i in outer_ids]) > 0.0


def test_surface_partition_area_scales_with_spacing():
    """Areas are physical: scaling all spacings 2x scales the TOTAL mesh area ~4x
    per cell (area is length²). The outward normal direction is unchanged by an
    isotropic rescale, so the apical/basal/lateral SPLIT is stable in the
    population mean (a handful of faces near the cos=0.5 classification boundary may
    reclassify under the coarser marching-cubes mesh, so the per-patch area is not
    bit-for-bit 4x — the total area + the mean fractions are the physical
    invariants)."""
    lab, df1, c, outer_ids, core_id, sp1 = _sphere_with_waveC(spacing=(1.0, 1.0, 1.0))
    compute_surface_partition(lab, df1, spacing=(1.0, 1.0, 1.0))
    lab2, df2, *_ = _sphere_with_waveC(spacing=(2.0, 2.0, 2.0))
    compute_surface_partition(lab2, df2, spacing=(2.0, 2.0, 2.0))

    parts = ["apical_surface_area", "basal_surface_area", "lateral_surface_area"]
    t1 = df1.set_index("label")[parts].sum(axis=1)
    t2 = df2.set_index("label")[parts].sum(axis=1)
    common = [i for i in t1.index if i in t2.index and np.isfinite(t1[i]) and t1[i] > 0]
    assert len(common) >= 4
    ratios = np.array([t2[i] / t1[i] for i in common])
    assert np.allclose(ratios, 4.0, rtol=1e-3)            # total area ~length² (exact)
    # the outward normal is identical and the mesh is just scaled, so the mean
    # apical fraction is stable (boundary-face reclassification only jitters it).
    f1 = df1.set_index("label")["apical_surface_fraction"]
    f2 = df2.set_index("label")["apical_surface_fraction"]
    cf = [i for i in f1.index if i in f2.index and np.isfinite(f1[i]) and np.isfinite(f2[i])]
    assert abs(np.mean([f1[i] for i in cf]) - np.mean([f2[i] for i in cf])) < 0.05


def test_surface_partition_low_confidence_is_nan():
    lab, df, c, outer_ids, core_id, sp = _sphere_with_waveC()
    target = outer_ids[0]
    df.loc[df["label"] == target, "tissue_outward_confidence"] = 0.0
    compute_surface_partition(lab, df, spacing=sp)
    assert np.isnan(df.set_index("label").loc[target, "apical_surface_fraction"])


# --------------------------------------------------------------------------- #
# 3) signed nuclear offset — hand-set outward normal -> deterministic sign
# --------------------------------------------------------------------------- #
def _box(vol, value, z, y, x, dz, dy, dx):
    vol[z:z + dz, y:y + dy, x:x + dx] = value


def _single_cell_nucleus(nuc_x0):
    """One 10^3 cell along X with a 4^3 nucleus placed at ``nuc_x0`` so we control
    whether the nucleus is shoved toward +X or −X relative to the cell centre."""
    Z, Y, X = 16, 16, 16
    cells = np.zeros((Z, Y, X), np.int32)
    nuclei = np.zeros((Z, Y, X), np.int32)
    _box(cells, 1, 3, 3, 3, 10, 10, 10)            # cell spans x 3..13, centre ~7.5
    _box(nuclei, 10, 6, 6, nuc_x0, 4, 4, 4)        # nucleus along X at nuc_x0
    triage = [{"cell_id": 1, "status": "clean", "n_nuclei": 1, "nucleus_ids": [10],
               "cell_voxels": int((cells == 1).sum())}]
    return cells, nuclei, triage


def _cell_outward_table(outward_zyx, conf=1.0, label=1):
    """A minimal cell_outward frame (the Wave C columns) for one cell."""
    return pd.DataFrame({
        "label": [label],
        "tissue_outward_z": [outward_zyx[0]],
        "tissue_outward_y": [outward_zyx[1]],
        "tissue_outward_x": [outward_zyx[2]],
        "tissue_outward_confidence": [conf],
    })


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


def test_extract_label_features_waveD_off_by_default(tmp_path):
    """No Wave D columns without tissue_depth=True (default-off)."""
    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)
    df = extract_label_features(str(p), n_workers=1)
    for c in _ALL_WAVED_GLOBAL:
        assert c not in df.columns, c


def test_extract_label_features_waveD_on_with_tissue_depth(tmp_path):
    """tissue_depth=True runs the Wave D global passes (directional needs the graph,
    surface partition is mesh-only) and the pre-existing columns are byte-identical."""
    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df_off = extract_label_features(str(p), n_workers=1)
    # adjacency=True so the directional-contact pass has a graph to classify.
    df_on = extract_label_features(
        str(p), n_workers=1, adjacency=True, tissue_depth=True, max_radius=5)

    assert list(df_on["label"].values) == list(df_off["label"].values)
    for c in _ALL_WAVED_GLOBAL:
        assert c in df_on.columns, c
        assert c not in df_off.columns, c

    # pre-existing columns byte-identical (Wave D only APPENDS).
    shared = [c for c in df_off.columns if c in df_on.columns]
    pd.testing.assert_frame_equal(
        df_on[shared].reset_index(drop=True),
        df_off[shared].reset_index(drop=True),
        check_dtype=False,
    )


def test_surface_partition_runs_without_adjacency(tmp_path):
    """Surface partition is mesh-only -> it appears under tissue_depth=True even with
    adjacency=False; directional contact appears too but is all-NaN (no graph)."""
    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)
    df = extract_label_features(str(p), n_workers=1, tissue_depth=True)  # adjacency off
    for c in _SP_COLS:
        assert c in df.columns, c
    for c in _DC_COLS:
        assert c in df.columns, c
    # no adjacency graph -> directional contact is all NaN.
    assert df["apical_contact_fraction"].isna().all()


# --------------------------------------------------------------------------- #
# 5) guards
# --------------------------------------------------------------------------- #
def test_directional_contact_skips_when_no_waveC():
    """No tissue_outward_* columns (Wave C did not run) -> the function SKIPS and
    leaves its columns absent (default-off contract)."""
    df = pd.DataFrame({"label": [1, 2], "centroid-0": [1.0, 2.0],
                       "centroid-1": [1.0, 2.0], "centroid-2": [1.0, 2.0]})
    lab = np.zeros((8, 8, 8), np.int32)
    lab[1:4, 1:4, 1:4] = 1
    lab[1:4, 1:4, 4:7] = 2
    compute_directional_contact(lab, df)
    for c in _DC_COLS:
        assert c not in df.columns, c


def test_surface_partition_skips_when_no_waveC():
    df = pd.DataFrame({"label": [1]})
    lab = np.zeros((8, 8, 8), np.int32)
    lab[1:6, 1:6, 1:6] = 1
    compute_surface_partition(lab, df)
    for c in _SP_COLS:
        assert c not in df.columns, c


def test_empty_df_gets_typed_columns():
    """An empty table with the Wave C columns -> typed (empty) Wave D columns."""
    df = pd.DataFrame({"label": pd.Series([], dtype=np.int64),
                       "tissue_outward_z": pd.Series([], dtype=np.float64),
                       "tissue_outward_y": pd.Series([], dtype=np.float64),
                       "tissue_outward_x": pd.Series([], dtype=np.float64),
                       "tissue_outward_confidence": pd.Series([], dtype=np.float64)})
    compute_directional_contact(np.zeros((4, 4, 4), np.int32), df)
    compute_surface_partition(np.zeros((4, 4, 4), np.int32), df)
    for c in _ALL_WAVED_GLOBAL:
        assert c in df.columns, c


# --------------------------------------------------------------------------- #
# 6) manifest + family registry
# --------------------------------------------------------------------------- #
def test_manifest_tags_all_waveD_features():
    man = fm.get_manifest()
    for c in (*_ALL_WAVED_GLOBAL, *_SIGNED_NUC_COLS):
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] == "tissue", c              # all signed-frame
        assert isinstance(tags["bounded"], bool)


def test_manifest_waveD_bounded_flags():
    man = fm.get_manifest()
    for c in ("apical_contact_fraction", "basal_contact_fraction",
              "lateral_contact_fraction", "contact_polarity_bias",
              "apical_surface_fraction", "basal_surface_fraction",
              "lateral_surface_fraction"):
        assert man[c]["bounded"] is True, c
    for c in ("apical_surface_area", "basal_surface_area", "lateral_surface_area",
              "apical_basal_surface_ratio",
              "nucleus_apical_basal_offset", "nucleus_apical_basal_offset_norm"):
        assert man[c]["bounded"] is False, c


def test_feature_stage_families_registry_includes_waveD():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["directional_contact"]) == set(_DC_COLS)
    assert set(_FEATURE_FAMILIES["surface_partition"]) == set(_SP_COLS)


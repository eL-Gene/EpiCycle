"""M4 feature tests — contact-patch geometry (C3) + local tissue frame (C4).

Pure-function known-shape tests (no tile readers / GPU). These features ride on
the M3 adjacency graph (``df.attrs["_adjacency_list"]`` + the per-cell centroid /
major-axis columns), so each test first builds the graph via
:func:`qc_features.compute_neighbor_features`, then exercises the two new M4
functions directly, then the OPT-IN global pass wired into ``extract_label_features``.

Synthetic answers we know by construction:

* a **flat monolayer SHEET** of cells (a single Z-layer grid): an interior cell's
  estimated local tissue normal is ~perpendicular to the sheet plane (so a cell
  standing PERPENDICULAR to the sheet has |local_normal_alignment|≈1, a cell lying
  IN-plane ≈0), and ``neighbor_ring_planarity`` is high (the neighbours are coplanar).
* a cell with **one big contact + several small contacts** → ``largest_contact_fraction``
  high and ``contact_balance_index`` low; a cell with **even contacts** → the reverse.
* a neighbour touching in **two disconnected places** → ``multi_patch_neighbor_fraction`` > 0
  and ``contact_count`` exceeds the neighbour count.
* a cell with **< 3 neighbours** → every tissue-frame column is NaN (the plane fit
  is undetermined).

Plus the manifest tags, the ``_FEATURE_FAMILIES`` registry, and the byte-identical
``adjacency`` default-off contract (M4 columns appear ONLY on ``adjacency=True``).
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# M4 built the first 6 contact-geometry columns; Mop-up M (remainder)
# extended the family. The exact-set family test reads the module constant so it
# tracks the growth (the 6 M4 columns asserted explicitly elsewhere stay a subset).
_CONTACT_COLS = qf.CONTACT_GEOMETRY_COLUMNS
_CONTACT_COLS_M4 = (
    "contact_area_total", "contact_area_max", "largest_contact_fraction",
    "contact_count", "multi_patch_neighbor_fraction", "contact_balance_index",
)
# The original 4 + Mop-up O remainder columns that are bounded in [0,1].
_FRAME_COLS_BOUNDED01 = (
    "neighbor_ring_planarity", "local_normal_alignment",
    "neighbor_axis_coherence", "neighbor_ring_anisotropy",
    # Mop-up O — the [0,1]-bounded coherence/order scores.
    "contact_weighted_axis_coherence", "neighbor_azimuth_entropy",
    "local_packing_order", "coordination_anisotropy", "frame_stability",
)
# Mop-up O remainder columns that are NOT bounded in [0,1] (a variance, a radian
# std, and an angle in degrees) -> excluded from the [0,1] bounded check.
_FRAME_COLS_UNBOUNDED01 = (
    "neighbor_axis_dispersion",         # variance, >=0 unbounded
    "neighbor_polar_dispersion",        # std (radians), >=0 unbounded
    "major_axis_vs_local_normal_angle",  # degrees, [0,90] (a bounded ANGLE, not [0,1])
)
_FRAME_COLS = _FRAME_COLS_BOUNDED01 + _FRAME_COLS_UNBOUNDED01


# --------------------------------------------------------------------------- #
# synthetic volumes
# --------------------------------------------------------------------------- #

def _flat_sheet_3x3():
    """A flat monolayer: a 3x3 grid of touching cells in a SINGLE thin Z-layer.

    Cells fill Y/X in a 3x3 tiling, each a 6x6 in-plane block, all on z in [1, 4)
    (thin in Z). The centre cell (id 5) has 8 in-plane neighbours all lying in the
    z≈const plane -> the least-spread direction of the neighbour-centroid offsets
    is the Z (sheet-normal) axis.

    Layout (ids), Y rows x X cols, each cell 6x6:
        1 2 3
        4 5 6
        7 8 9
    """
    lab = np.zeros((6, 18, 18), dtype=np.int32)
    cid = 1
    for yi in range(3):
        for xi in range(3):
            lab[1:4, yi * 6:(yi + 1) * 6, xi * 6:(xi + 1) * 6] = cid
            cid += 1
    return lab


def _sheet_with_perpendicular_center():
    """Same flat 3x3 sheet, but the CENTRE cell is a tall column standing
    perpendicular to the sheet (elongated along Z) — so its major (long) axis is
    ~the sheet normal -> ``local_normal_alignment`` should be ~1.

    The 8 surround cells are thin in-plane blocks (z in [3, 6)); the centre cell
    is a tall thin column (z in [0, 9), small Y/X) at the grid centre so it shares
    a face with each of its 4 edge neighbours.
    """
    lab = np.zeros((10, 18, 18), dtype=np.int32)
    cid = 1
    for yi in range(3):
        for xi in range(3):
            if yi == 1 and xi == 1:
                cid += 1
                continue                       # skip centre; placed below
            lab[3:6, yi * 6:(yi + 1) * 6, xi * 6:(xi + 1) * 6] = cid
            cid += 1
    # Centre cell id 5: a tall column spanning Z, occupying the centre 6x6 in Y/X.
    lab[0:9, 6:12, 6:12] = 5
    return lab


def _sheet_with_inplane_center():
    """Same flat 3x3 sheet, centre cell elongated IN-plane (a long bar along X) so
    its major axis lies IN the sheet -> ``local_normal_alignment`` should be ~0.
    """
    lab = np.zeros((6, 18, 18), dtype=np.int32)
    cid = 1
    for yi in range(3):
        for xi in range(3):
            if yi == 1 and xi == 1:
                cid += 1
                continue
            lab[1:4, yi * 6:(yi + 1) * 6, xi * 6:(xi + 1) * 6] = cid
            cid += 1
    # Centre cell id 5: a long thin bar along X (in-plane), 1 Z-slice thick-ish.
    lab[1:4, 7:11, 6:12] = 5
    return lab


def _dominant_vs_even_contacts():
    """Cell 1 with FOUR neighbours: one shares a BIG face, three share TINY 1x1
    spots -> largest_contact_fraction high, balance low for cell 1.

    Cell 1: a central block (z 2..5, y 4..10, x 4..8) -> the +X face is 4x6=24.
    Cell 2: large neighbour sharing the WHOLE +X face (big contact, ~24 voxels).
    Cells 3,4,5: tiny 1x1x1 pillars each touching cell 1 at a single voxel face on
        the -X / +Y / -Y sides (~1 voxel each).
    """
    lab = np.zeros((10, 18, 18), dtype=np.int32)
    lab[2:6, 4:10, 4:8] = 1               # central block (z 2..5, y 4..9, x 4..7)
    lab[2:6, 4:10, 8:12] = 2              # shares the whole x==8 face (big, 24)
    lab[2:3, 4:5, 3:4] = 3               # single voxel touching the x==4 face (1)
    lab[2:3, 10:11, 5:6] = 4            # single voxel touching the y==10 face (1)
    lab[2:3, 3:4, 5:6] = 5             # single voxel touching the y==4 face (1)
    return lab


def _even_contacts():
    """Cell 1 with two EQUAL-size opposing contacts -> balance high, largest
    fraction ~0.5."""
    lab = np.zeros((8, 16, 16), dtype=np.int32)
    lab[2:6, 6:10, 6:10] = 1
    lab[2:6, 6:10, 10:14] = 2             # equal face on +X
    lab[2:6, 6:10, 2:6] = 3              # equal face on -X
    return lab


def _two_patch_neighbor():
    """A neighbour (id 2) that touches cell 1 in TWO disconnected places -> a
    multi-patch contact. Cell 2 is a 'C' that wraps around with a gap.

    Cell 1: a bar along X. Cell 2: two separate blocks both touching cell 1's +Y
    face but separated along X by a gap (so they are two disconnected patches of
    the SAME neighbour id).
    """
    lab = np.zeros((8, 16, 20), dtype=np.int32)
    lab[2:6, 4:8, 4:16] = 1               # cell 1 bar (y 4..7, x 4..15)
    lab[2:6, 8:11, 4:7] = 2              # patch A on +Y face, low X
    lab[2:6, 8:11, 13:16] = 2            # patch B on +Y face, high X (gap between)
    return lab


def _df_for(lab):
    return qf.extract_features(lab)


def _with_graph(lab, max_radius=3):
    """extract_features + the M3 adjacency graph (so M4 has _adjacency_list)."""
    df = _df_for(lab)
    qf.compute_neighbor_features(lab, df, max_radius=max_radius)
    return df


# --------------------------------------------------------------------------- #
# C3 — contact patch geometry
# --------------------------------------------------------------------------- #

def test_contact_geometry_columns_present_and_typed():
    lab = _dominant_vs_even_contacts()
    df = _with_graph(lab)
    qf.compute_contact_geometry(lab, df)
    for c in _CONTACT_COLS:
        assert c in df.columns, c
    assert len(df) == 5


def test_largest_contact_fraction_high_for_dominant():
    lab = _dominant_vs_even_contacts()
    df = _with_graph(lab)
    qf.compute_contact_geometry(lab, df)
    r = df.set_index("label").loc[1]
    # one big contact dominates -> fraction well above an even split.
    assert r["largest_contact_fraction"] > 0.5
    # four neighbours each contribute >= one patch -> contact_count >= 4.
    assert r["contact_count"] >= 4.0
    assert r["contact_area_total"] > 0.0


def test_largest_contact_fraction_low_for_even():
    lab = _even_contacts()
    df = _with_graph(lab)
    qf.compute_contact_geometry(lab, df)
    r = df.set_index("label").loc[1]
    # two equal contacts -> largest fraction ~0.5, balance high.
    assert r["largest_contact_fraction"] == pytest.approx(0.5, abs=0.1)
    assert r["contact_balance_index"] > 0.7


def test_contact_balance_ordering_dominant_vs_even():
    """The even-contacts cell has a HIGHER balance index than the dominant one."""
    lab_dom = _dominant_vs_even_contacts()
    df_dom = _with_graph(lab_dom)
    qf.compute_contact_geometry(lab_dom, df_dom)
    bal_dom = df_dom.set_index("label").loc[1, "contact_balance_index"]

    lab_even = _even_contacts()
    df_even = _with_graph(lab_even)
    qf.compute_contact_geometry(lab_even, df_even)
    bal_even = df_even.set_index("label").loc[1, "contact_balance_index"]

    assert bal_even > bal_dom


def test_multi_patch_neighbor_detected():
    lab = _two_patch_neighbor()
    df = _with_graph(lab)
    qf.compute_contact_geometry(lab, df)
    r = df.set_index("label").loc[1]
    # the single neighbour (id 2) touches in two disconnected patches.
    assert r["multi_patch_neighbor_fraction"] == pytest.approx(1.0)
    # contact_count counts the two patches -> >= 2 even though neighbour_count==1.
    assert r["contact_count"] >= 2.0
    assert r["neighbor_count"] == 1.0


def test_bounded_contact_fractions():
    lab = _dominant_vs_even_contacts()
    df = _with_graph(lab)
    qf.compute_contact_geometry(lab, df)
    for c in ("largest_contact_fraction", "multi_patch_neighbor_fraction",
              "contact_balance_index"):
        v = df[c].values
        finite = v[np.isfinite(v)]
        assert np.all(finite >= 0.0) and np.all(finite <= 1.0), (c, finite)


def test_contact_geometry_isolated_cell_zero():
    """A cell with no graph neighbours reads zero contact area + NaN fractions."""
    lab = np.zeros((8, 12, 30), np.int32)
    lab[2:6, 2:8, 2:8] = 1
    lab[2:6, 2:8, 8:14] = 2               # touches cell 1
    lab[2:6, 2:8, 24:30] = 3              # isolated
    df = _with_graph(lab, max_radius=3)
    qf.compute_contact_geometry(lab, df)
    r = df.set_index("label").loc[3]
    assert r["contact_area_total"] == 0.0
    assert r["contact_count"] == 0.0
    assert np.isnan(r["largest_contact_fraction"])
    assert np.isnan(r["contact_balance_index"])


def test_contact_geometry_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_contact_geometry(np.zeros((4, 4, 4), np.int32), df)
    for c in _CONTACT_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# C4 — local tissue frame
# --------------------------------------------------------------------------- #

def test_tissue_frame_columns_present():
    lab = _flat_sheet_3x3()
    df = _with_graph(lab)
    qf.compute_tissue_frame_features(df)
    for c in _FRAME_COLS:
        assert c in df.columns, c


def test_sheet_center_high_planarity():
    """The centre cell of a flat sheet has coplanar neighbours -> high planarity."""
    lab = _flat_sheet_3x3()
    df = _with_graph(lab)
    qf.compute_tissue_frame_features(df)
    r = df.set_index("label").loc[5]              # centre cell of the 3x3 grid
    assert r["neighbor_ring_planarity"] > 0.9     # neighbours nearly coplanar


def test_perpendicular_center_normal_alignment_high():
    """A centre cell standing PERPENDICULAR to the sheet (tall Z column) has its
    major axis ~along the estimated sheet normal -> |local_normal_alignment| ~1."""
    lab = _sheet_with_perpendicular_center()
    df = _with_graph(lab, max_radius=3)
    qf.compute_tissue_frame_features(df)
    r = df.set_index("label").loc[5]
    assert r["local_normal_alignment"] > 0.8


def test_inplane_center_normal_alignment_low():
    """A centre cell elongated IN the sheet plane has its major axis ~perpendicular
    to the sheet normal -> |local_normal_alignment| ~0."""
    lab = _sheet_with_inplane_center()
    df = _with_graph(lab)
    qf.compute_tissue_frame_features(df)
    r = df.set_index("label").loc[5]
    assert r["local_normal_alignment"] < 0.4


def test_tissue_frame_bounded():
    lab = _flat_sheet_3x3()
    df = _with_graph(lab)
    qf.compute_tissue_frame_features(df)
    # the [0,1]-bounded coherence/order scores.
    for c in _FRAME_COLS_BOUNDED01:
        v = df[c].values
        finite = v[np.isfinite(v)]
        assert np.all(finite >= 0.0) and np.all(finite <= 1.0 + 1e-9), (c, finite)
    # the unbounded remainder is still >= 0 (a variance / a non-negative dispersion /
    # an angle), and the angle is within [0, 90].
    for c in ("neighbor_axis_dispersion", "neighbor_polar_dispersion"):
        finite = df[c].values[np.isfinite(df[c].values)]
        assert np.all(finite >= 0.0), c
    ang = df["major_axis_vs_local_normal_angle"].values
    ang = ang[np.isfinite(ang)]
    assert np.all((ang >= 0.0) & (ang <= 90.0 + 1e-9))


def test_tissue_frame_few_neighbors_nan():
    """A cell with < 3 neighbours -> every tissue-frame column NaN (plane fit
    undetermined)."""
    lab = np.zeros((8, 12, 24), np.int32)
    lab[2:6, 2:8, 2:8] = 1
    lab[2:6, 2:8, 8:14] = 2               # cell 1 has exactly ONE neighbour
    df = _with_graph(lab, max_radius=3)
    qf.compute_tissue_frame_features(df)
    r = df.set_index("label").loc[1]
    for c in _FRAME_COLS:
        assert np.isnan(r[c]), c


def test_tissue_frame_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_tissue_frame_features(df)
    for c in _FRAME_COLS:
        assert c in df.columns
    assert len(df) == 0


def test_tissue_frame_missing_centroid_cols_nan():
    """No centroid columns -> all frame columns NaN, no crash."""
    df = pd.DataFrame({"label": [1, 2, 3, 4]})
    qf.compute_tissue_frame_features(df, adjacency=[
        {"label_a": 1, "label_b": 2, "contact_voxels": 5},
        {"label_a": 1, "label_b": 3, "contact_voxels": 5},
        {"label_a": 1, "label_b": 4, "contact_voxels": 5},
    ])
    for c in _FRAME_COLS:
        assert df[c].isna().all(), c


# --------------------------------------------------------------------------- #
# manifest + families registry
# --------------------------------------------------------------------------- #

def test_manifest_tags_m4_features():
    man = fm.get_manifest()
    for c in (*_CONTACT_COLS, *_FRAME_COLS):
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] in ("image", "tissue", "intrinsic")
        assert isinstance(tags["bounded"], bool)


def test_manifest_m4_bounded_flags():
    man = fm.get_manifest()
    # the [0,1]-bounded coherence/order scores + the bounded ANGLE ([0,90]).
    for c in ("largest_contact_fraction", "multi_patch_neighbor_fraction",
              "contact_balance_index", *_FRAME_COLS_BOUNDED01,
              "major_axis_vs_local_normal_angle"):
        assert man[c]["bounded"] is True, c
    for c in ("contact_area_total", "contact_area_max", "contact_count",
              # Mop-up O — the variance + radian std are unbounded.
              "neighbor_axis_dispersion", "neighbor_polar_dispersion"):
        assert man[c]["bounded"] is False, c


def test_tissue_frame_frame_tag():
    man = fm.get_manifest()
    for c in _FRAME_COLS:
        assert man[c]["frame"] == "tissue", c


def test_feature_stage_families_registry_includes_m4():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["contact_geometry"]) == set(_CONTACT_COLS)
    # The tissue_frame family = the unsigned columns this test owns PLUS the P1a
    # SIGNED apical/basal remainder (added on the Wave C tissue_depth re-call; see
    # test_feature_p1a.py). Reference the source constant so the two stay in lockstep.
    assert (set(_FEATURE_FAMILIES["tissue_frame"])
            == set(_FRAME_COLS) | set(qf.SIGNED_TISSUE_FRAME_COLUMNS))


# --------------------------------------------------------------------------- #
# opt-in global pass wired into extract_label_features
# --------------------------------------------------------------------------- #

def _toy_sheet_labels():
    """A flat 3x3 sheet that also has an isolated cell — enough to exercise the
    full M4 global pass through extract_label_features."""
    return _flat_sheet_3x3()


def test_extract_label_features_adjacency_off_no_m4(tmp_path):
    """``adjacency=False`` (the default) must NOT add any M4 columns — the additive
    -optional contract carries through to M4."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_sheet_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df_baseline = extract_label_features(str(p), n_workers=1)
    df_default = extract_label_features(str(p), n_workers=1, adjacency=False)

    assert list(df_baseline.columns) == list(df_default.columns)
    pd.testing.assert_frame_equal(df_baseline, df_default)
    for c in (*_CONTACT_COLS, *_FRAME_COLS):
        assert c not in df_baseline.columns, c


def test_extract_label_features_adjacency_on_adds_m4(tmp_path):
    """``adjacency=True`` runs the M4 passes: contact-geometry + tissue-frame
    columns appear, and the per-cell morphology columns are unchanged."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_sheet_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df_off = extract_label_features(str(p), n_workers=1)
    df_on = extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=3)

    assert list(df_on["label"].values) == list(df_off["label"].values)
    for c in (*_CONTACT_COLS, *_FRAME_COLS):
        assert c in df_on.columns, c
        assert c not in df_off.columns, c
    # the pre-existing morphology columns are byte-identical (M4 only APPENDS).
    shared = [c for c in df_off.columns if c in df_on.columns]
    pd.testing.assert_frame_equal(
        df_on[shared].reset_index(drop=True),
        df_off[shared].reset_index(drop=True),
        check_dtype=False,
    )
    # the centre cell of the flat sheet -> high planarity through the full pass.
    r = df_on.set_index("label").loc[5]
    assert r["neighbor_ring_planarity"] > 0.9
    assert r["contact_area_total"] > 0.0


def test_extract_label_features_adjacency_on_from_array():
    """``adjacency=True`` M4 also works for an IN-MEMORY array input."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_sheet_labels()
    df = extract_label_features(vol, n_workers=1, adjacency=True, max_radius=3)
    for c in (*_CONTACT_COLS, *_FRAME_COLS):
        assert c in df.columns, c
    r = df.set_index("label").loc[5]
    assert r["neighbor_ring_planarity"] > 0.9

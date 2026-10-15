"""Wave A feature tests — adjacency-graph-derived families.

Four families that ride on the M3 adjacency graph (``df.attrs["_adjacency_list"]``
+ the per-cell centroid / major-axis / size columns) and run in the SAME global
``adjacency=True`` pass:

* C5  :func:`compute_graph_topology`            (atlas )
* C6  :func:`compute_axis_neighbor_persistence` (atlas )
* C12 :func:`compute_local_comparative`         (atlas )
* C16 :func:`compute_graph_motifs`              (atlas )

The graph-only families (C5/C12/C16) are tested by handing them a synthetic
``_adjacency_list`` + a tiny per-cell ``df`` directly (no label volume needed),
with known-graph answers:

* a **star/hub** (1 central + N leaves, no leaf-leaf edges): ego_density == 0,
  ego_triangle_count == 0, second_ring of a leaf == the other leaves.
* a **triangle/clique** (3 mutually-adjacent cells): ego_density == 1,
  ego_clustering_coefficient == 1, ego_triangle_count == 1 per node.
* a **chain** (1-2-3-4): the middle nodes have second_ring_count == 1
  (the node two hops away), turnover of degree, etc.
* local-comparative on a cell that is **2× its neighbours** -> ratio ≈ 2 and a
  high positive local z-score.

C6 (axis persistence) needs real contact voxels, so it's tested on a small
synthetic label volume (a tall central cell with basal-only + apical-only
neighbours along its Z major axis).

Plus the manifest tags, the ``_FEATURE_FAMILIES`` registry, and the byte-identical
``adjacency`` default-off contract (Wave A columns appear ONLY on ``adjacency=True``).
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# Wave A built the first 4 graph-topology columns; Mop-up M (remainder)
# extended the family. The exact-set family test reads the module constant so it
# tracks the growth (the 4 Wave-A columns asserted explicitly below stay a subset).
_TOPO_COLS = qf.GRAPH_TOPOLOGY_COLUMNS
_TOPO_COLS_WAVEA = ("ego_density", "ego_clustering_coefficient",
                    "second_ring_count", "mean_neighbor_degree")
# Wave A built the first 4 axis-neighbour columns; Wave J (remainder) extended
# the family. The exact-set family test reads the module constant so it tracks the
# growth (the 4 Wave-A columns asserted explicitly below stay a subset).
_AXIS_COLS = qf.AXIS_NEIGHBOR_PERSISTENCE_COLUMNS
_AXIS_COLS_WAVEA = ("axis_neighbor_turnover", "axis_neighbor_persistence_ratio",
                    "axis_basal_neighbor_count", "axis_apical_neighbor_count")
# Wave A built the first 4 motif columns; Mop-up M (remainder) extended it.
_MOTIF_COLS = qf.GRAPH_MOTIF_COLUMNS
_MOTIF_COLS_WAVEA = ("ego_triangle_count", "neighbor_count_cv",
                     "degree_centrality", "neighbor_degree_assortativity")


# --------------------------------------------------------------------------- #
# helpers — build a df + adjacency list directly (no volume)
# --------------------------------------------------------------------------- #

def _edge_list(edges):
    """A minimal M3-style _adjacency_list from undirected (a, b) pairs."""
    return [{"label_a": int(a), "label_b": int(b), "contact_voxels": 10}
            for (a, b) in edges]


def _df_with_graph(labels, edges, extra_cols=None):
    """A per-cell df (one row per label) carrying a synthetic _adjacency_list."""
    df = pd.DataFrame({"label": list(labels)})
    if extra_cols:
        for c, vals in extra_cols.items():
            df[c] = vals
    df.attrs["_adjacency_list"] = _edge_list(edges)
    return df


# Three canonical graphs used across the tests:
def _star(n_leaves=4):
    """Hub = label 1; leaves = 2..n_leaves+1. Only hub-leaf edges (no leaf-leaf)."""
    leaves = list(range(2, 2 + n_leaves))
    edges = [(1, lf) for lf in leaves]
    return [1, *leaves], edges


def _triangle():
    """Three mutually-adjacent cells (a 3-clique): 1-2, 2-3, 1-3."""
    return [1, 2, 3], [(1, 2), (2, 3), (1, 3)]


def _chain():
    """A path 1-2-3-4 (each interior node has degree 2)."""
    return [1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)]


# --------------------------------------------------------------------------- #
# C5 — graph topology
# --------------------------------------------------------------------------- #

def test_graph_topology_columns_present():
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    for c in _TOPO_COLS:
        assert c in df.columns, c
    assert len(df) == 3


def test_ego_density_clique_is_one():
    """A clique: every node's neighbours are fully connected -> ego_density == 1."""
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    for lid in (1, 2, 3):
        # each node has 2 neighbours which ARE connected -> density 1, clustering 1
        assert by.loc[lid, "ego_density"] == pytest.approx(1.0)
        assert by.loc[lid, "ego_clustering_coefficient"] == pytest.approx(1.0)


def test_ego_density_star_is_zero():
    """A star hub: its leaves are mutually UNconnected -> ego_density == 0."""
    labels, edges = _star(n_leaves=4)
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    assert by.loc[1, "ego_density"] == pytest.approx(0.0)
    assert by.loc[1, "ego_clustering_coefficient"] == pytest.approx(0.0)
    # the hub has 4 neighbours, each of degree 1 -> mean_neighbor_degree == 1
    assert by.loc[1, "mean_neighbor_degree"] == pytest.approx(1.0)


def test_second_ring_count_on_star_leaf():
    """A leaf of a 4-star: its only neighbour is the hub, whose other neighbours
    (the other 3 leaves) are the leaf's second ring."""
    labels, edges = _star(n_leaves=4)
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    assert by.loc[2, "second_ring_count"] == pytest.approx(3.0)
    # ego density / clustering undefined for a degree-1 leaf -> NaN
    assert np.isnan(by.loc[2, "ego_density"])
    assert np.isnan(by.loc[2, "ego_clustering_coefficient"])


def test_second_ring_count_on_chain():
    """Chain 1-2-3-4: node 1's first ring is {2}, second ring is {3}
    (node 3 is two hops away, not directly adjacent)."""
    labels, edges = _chain()
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    assert by.loc[1, "second_ring_count"] == pytest.approx(1.0)   # {3}
    assert by.loc[2, "second_ring_count"] == pytest.approx(1.0)   # {4}
    assert by.loc[1, "mean_neighbor_degree"] == pytest.approx(2.0)  # nbr 2 has degree 2


def test_graph_topology_isolated_cell():
    """An isolated cell (no edges) -> second_ring 0, the rest NaN."""
    df = _df_with_graph([1, 2, 3], [(1, 2)])    # 3 is isolated
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    assert by.loc[3, "second_ring_count"] == 0.0
    assert np.isnan(by.loc[3, "mean_neighbor_degree"])
    assert np.isnan(by.loc[3, "ego_density"])


def test_graph_topology_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_graph_topology(df)
    for c in _TOPO_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# C16 — graph motifs
# --------------------------------------------------------------------------- #

def test_graph_motif_columns_present():
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges)
    qf.compute_graph_motifs(df)
    for c in _MOTIF_COLS:
        assert c in df.columns, c


def test_ego_triangle_count_clique():
    """In a 3-clique each node sits in exactly ONE triangle (the edge between its
    two neighbours)."""
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges)
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    for lid in (1, 2, 3):
        assert by.loc[lid, "ego_triangle_count"] == pytest.approx(1.0)


def test_ego_triangle_count_star_is_zero():
    """A star has no triangles (leaves are mutually unconnected)."""
    labels, edges = _star(n_leaves=4)
    df = _df_with_graph(labels, edges)
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    for lid in (1, 2, 3, 4, 5):
        assert by.loc[lid, "ego_triangle_count"] == pytest.approx(0.0)


def test_degree_centrality_bounded_and_hub_is_max():
    """degree_centrality ∈ [0,1]; the star hub is the most-connected -> 1.0."""
    labels, edges = _star(n_leaves=4)
    df = _df_with_graph(labels, edges)
    qf.compute_graph_motifs(df)
    v = df["degree_centrality"].values
    assert np.all(v >= 0.0) and np.all(v <= 1.0)
    by = df.set_index("label")
    assert by.loc[1, "degree_centrality"] == pytest.approx(1.0)   # hub (degree 4)
    assert by.loc[2, "degree_centrality"] == pytest.approx(0.25)  # leaf (degree 1)


def test_neighbor_count_cv_uniform_neighbours():
    """A clique: every node's neighbours have identical degree -> CV == 0."""
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges)
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    for lid in (1, 2, 3):
        assert by.loc[lid, "neighbor_count_cv"] == pytest.approx(0.0)


def test_assortativity_star_disassortative():
    """A star is the canonical DISASSORTATIVE graph: the high-degree hub connects
    only to low-degree leaves -> the hub's per-cell assortativity proxy is < 0."""
    labels, edges = _star(n_leaves=4)
    df = _df_with_graph(labels, edges)
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    assert by.loc[1, "neighbor_degree_assortativity"] < 0.0    # hub vs leaves


def test_graph_motifs_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_graph_motifs(df)
    for c in _MOTIF_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# C12 — local comparative
# --------------------------------------------------------------------------- #

def test_local_comparative_ratio_and_zscore_on_2x_cell():
    """Cell 1 is the hub of a 4-star whose 4 leaves all have area 10; the hub has
    area 20 (2x). area_vs_neighbor_mean_ratio == 2.0 and a large positive
    local z-score."""
    labels, edges = _star(n_leaves=4)
    # hub area 20, leaves area 10
    areas = [20.0, 10.0, 10.0, 10.0, 10.0]
    df = _df_with_graph(labels, edges, extra_cols={"area": areas})
    qf.compute_local_comparative(df, features=("area",))
    by = df.set_index("label")
    assert by.loc[1, "area_vs_neighbor_mean_ratio"] == pytest.approx(2.0)
    # neighbours are identical (all 10) -> MAD == 0; the hub differs -> NaN z by
    # the documented "infinite z is not informative" rule.
    assert np.isnan(by.loc[1, "area_local_zscore"])


def test_local_comparative_zscore_with_spread():
    """Hub neighbours have a SPREAD of areas so MAD > 0 and the z-score is finite
    + positive when the hub is the largest."""
    labels, edges = _star(n_leaves=4)
    areas = [30.0, 8.0, 10.0, 12.0, 14.0]   # hub 30, leaves 8/10/12/14
    df = _df_with_graph(labels, edges, extra_cols={"area": areas})
    qf.compute_local_comparative(df, features=("area",))
    by = df.set_index("label")
    z = by.loc[1, "area_local_zscore"]
    assert np.isfinite(z) and z > 0.0
    # ratio = 30 / mean(8,10,12,14)=11 -> ~2.727
    assert by.loc[1, "area_vs_neighbor_mean_ratio"] == pytest.approx(30.0 / 11.0)


def test_local_comparative_skips_absent_base_feature():
    """A base feature not in the df produces NO columns (silently skipped)."""
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges, extra_cols={"area": [1.0, 2.0, 3.0]})
    qf.compute_local_comparative(df, features=("area", "not_a_column"))
    assert "area_vs_neighbor_mean_ratio" in df.columns
    assert "not_a_column_vs_neighbor_mean_ratio" not in df.columns


def test_local_comparative_isolated_cell_nan():
    """An isolated cell has no neighbours -> ratio + zscore NaN."""
    df = _df_with_graph([1, 2, 3], [(1, 2)], extra_cols={"area": [5.0, 5.0, 5.0]})
    qf.compute_local_comparative(df, features=("area",))
    by = df.set_index("label")
    assert np.isnan(by.loc[3, "area_vs_neighbor_mean_ratio"])
    assert np.isnan(by.loc[3, "area_local_zscore"])


def test_local_comparative_default_feature_set():
    """With ``features=None`` only the base columns PRESENT in df are templated."""
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges, extra_cols={
        "area": [1.0, 2.0, 3.0], "sphericity": [0.5, 0.6, 0.7]})
    qf.compute_local_comparative(df)   # default LOCAL_COMPARATIVE_FEATURES
    # area + sphericity are in the default set AND present -> templated
    assert "area_vs_neighbor_mean_ratio" in df.columns
    assert "sphericity_local_zscore" in df.columns
    # surface_area is in the default set but ABSENT here -> not templated
    assert "surface_area_vs_neighbor_mean_ratio" not in df.columns


# --------------------------------------------------------------------------- #
# C6 — axis neighbour persistence (needs a real label volume)
# --------------------------------------------------------------------------- #

def _tall_axis_cell():
    """A tall central column (id 1, Z-major axis) with neighbours that touch only
    at its BASE (low Z) and only at its TOP (high Z).

    cell 1: tall column x∈[6,10), y∈[2,6), z∈[0,16) -> major axis ≈ +Z
    cell 2: basal neighbour, touches at the x==10 face, z∈[0,4) only (low t)
    cell 3: apical neighbour, touches at the x==6 face, z∈[12,16) only (high t)
    """
    lab = np.zeros((16, 8, 16), dtype=np.int32)
    lab[0:16, 2:6, 6:10] = 1
    lab[0:4, 2:6, 10:14] = 2          # basal (low Z) neighbour
    lab[12:16, 2:6, 2:6] = 3          # apical (high Z) neighbour
    return lab


def test_axis_persistence_columns_present_and_basal_apical_split():
    lab = _tall_axis_cell()
    df = qf.extract_features(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    qf.compute_axis_neighbor_persistence(lab, df, max_radius=3)
    for c in _AXIS_COLS:
        assert c in df.columns, c
    by = df.set_index("label")
    # cell 1 has a basal-only neighbour (2) and an apical-only neighbour (3):
    # exactly one neighbour at each end -> basal==1, apical==1.
    assert by.loc[1, "axis_basal_neighbor_count"] == pytest.approx(1.0)
    assert by.loc[1, "axis_apical_neighbor_count"] == pytest.approx(1.0)
    # the basal and apical neighbour sets are DISJOINT -> turnover == 1.
    assert by.loc[1, "axis_neighbor_turnover"] == pytest.approx(1.0)


def test_axis_persistence_ratios_bounded():
    lab = _tall_axis_cell()
    df = qf.extract_features(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    qf.compute_axis_neighbor_persistence(lab, df, max_radius=3)
    for c in ("axis_neighbor_turnover", "axis_neighbor_persistence_ratio"):
        v = df[c].values
        finite = v[np.isfinite(v)]
        assert np.all(finite >= 0.0) and np.all(finite <= 1.0), (c, finite)


def test_axis_persistence_with_anisotropic_spacing():
    """Anisotropic spacing scales the contact coords into the SAME physical frame
    as the major axis (no frame mismatch) -> the basal/apical split is unchanged
    for a Z-major cell."""
    lab = _tall_axis_cell()
    sp = (2.0, 0.5, 0.5)
    df = qf.extract_features(lab, spacing=sp)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    qf.compute_axis_neighbor_persistence(lab, df, spacing=sp, max_radius=3)
    by = df.set_index("label")
    assert by.loc[1, "axis_basal_neighbor_count"] == pytest.approx(1.0)
    assert by.loc[1, "axis_apical_neighbor_count"] == pytest.approx(1.0)
    assert by.loc[1, "axis_neighbor_turnover"] == pytest.approx(1.0)


def test_axis_persistence_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_axis_neighbor_persistence(
        np.zeros((4, 4, 4), np.int32), df, max_radius=3)
    for c in _AXIS_COLS:
        assert c in df.columns
    assert len(df) == 0


def test_axis_persistence_missing_axis_cols_nan():
    """Without major_axis_* columns the whole family is NaN (guard)."""
    df = pd.DataFrame({"label": [1, 2],
                       "centroid-0": [1.0, 2.0], "centroid-1": [1.0, 2.0],
                       "centroid-2": [1.0, 2.0]})
    df.attrs["_adjacency_list"] = _edge_list([(1, 2)])
    qf.compute_axis_neighbor_persistence(
        np.zeros((4, 4, 4), np.int32), df, max_radius=3)
    for c in _AXIS_COLS:
        assert c in df.columns
        assert df[c].isna().all(), c


# --------------------------------------------------------------------------- #
# manifest + families registry
# --------------------------------------------------------------------------- #

def test_manifest_tags_static_waveA_features():
    man = fm.get_manifest()
    static = (*_TOPO_COLS, *_AXIS_COLS, *_MOTIF_COLS)
    for c in static:
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] in ("image", "tissue", "intrinsic")
        assert isinstance(tags["bounded"], bool)


def test_manifest_waveA_bounded_flags():
    man = fm.get_manifest()
    for c in ("ego_density", "ego_clustering_coefficient",
              "axis_neighbor_turnover", "axis_neighbor_persistence_ratio",
              "degree_centrality"):
        assert man[c]["bounded"] is True, c
    for c in ("second_ring_count", "mean_neighbor_degree",
              "ego_triangle_count", "neighbor_count_cv",
              "neighbor_degree_assortativity",
              "axis_basal_neighbor_count", "axis_apical_neighbor_count"):
        assert man[c]["bounded"] is False, c


def test_manifest_local_comparative_tags_helper():
    tags = fm.local_comparative_tags(("area", "sphericity"))
    for base in ("area", "sphericity"):
        for suffix in ("_vs_neighbor_mean_ratio", "_local_zscore"):
            col = f"{base}{suffix}"
            assert col in tags, col
            assert tags[col]["role"] == "biology"
            assert tags[col]["frame"] == "image"
            assert tags[col]["redundancy_group"] == f"local_comparative_{base}"


def test_feature_stage_families_registry_includes_waveA():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["graph_topology"]) == set(_TOPO_COLS)
    assert set(_FEATURE_FAMILIES["axis_neighbor_persistence"]) == set(_AXIS_COLS)
    assert set(_FEATURE_FAMILIES["graph_motif"]) == set(_MOTIF_COLS)


def test_local_comparative_families_recovered_from_columns():
    from packages.epicycle_scaledseg.core.feature_stage import _local_comparative_families
    cols = ["label", "area", "area_vs_neighbor_mean_ratio", "area_local_zscore",
            "sphericity_vs_neighbor_mean_ratio", "sphericity_local_zscore"]
    fams = _local_comparative_families(cols)
    assert set(fams["local_comparative_area"]) == {
        "area_vs_neighbor_mean_ratio", "area_local_zscore"}
    assert set(fams["local_comparative_sphericity"]) == {
        "sphericity_vs_neighbor_mean_ratio", "sphericity_local_zscore"}


# --------------------------------------------------------------------------- #
# opt-in global pass wired into extract_label_features
# --------------------------------------------------------------------------- #

_WAVEA_STATIC = (*_TOPO_COLS, *_AXIS_COLS, *_MOTIF_COLS)


def _toy_labels():
    """Cells 1 & 2 touch at x==8; cell 3 isolated (gap >> max_radius=5)."""
    vol = np.zeros((8, 16, 36), np.int32)
    vol[1:6, 2:8, 2:8] = 1
    vol[1:6, 2:8, 8:14] = 2          # touches cell 1 at x==8
    vol[1:6, 2:8, 28:34] = 3         # isolated
    return vol


def test_extract_label_features_adjacency_off_no_waveA(tmp_path):
    """``adjacency=False`` (the default) emits NO Wave A columns — the additive-
    optional contract."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)
    df = extract_label_features(str(p), n_workers=1)            # default off
    for c in _WAVEA_STATIC:
        assert c not in df.columns, c
    # local-comparative templated columns absent too
    assert not any(c.endswith("_vs_neighbor_mean_ratio") for c in df.columns)
    assert not any(c.endswith("_local_zscore") for c in df.columns)


def test_extract_label_features_adjacency_on_adds_waveA(tmp_path):
    """``adjacency=True`` runs the Wave A passes: the static columns appear, the
    local-comparative templated columns appear for the present base features, and
    the per-cell morphology columns are UNCHANGED."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df_off = extract_label_features(str(p), n_workers=1)
    df_on = extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)

    assert list(df_on["label"].values) == list(df_off["label"].values)
    for c in _WAVEA_STATIC:
        assert c in df_on.columns, c
        assert c not in df_off.columns, c
    # at least one local-comparative pair appears (area is a default base feature
    # and is always produced by extract_features).
    assert "area_vs_neighbor_mean_ratio" in df_on.columns
    assert "area_local_zscore" in df_on.columns

    # the pre-existing morphology columns are byte-identical (Wave A only APPENDS).
    shared = [c for c in df_off.columns if c in df_on.columns]
    pd.testing.assert_frame_equal(
        df_on[shared].reset_index(drop=True),
        df_off[shared].reset_index(drop=True),
        check_dtype=False,
    )
    # the touching pair (1,2) each have neighbour_count 1; their ego graph has no
    # triangle (only 2 cells), so ego_triangle_count == 0; the isolated cell 3 has
    # second_ring 0.
    by = df_on.set_index("label")
    assert by.loc[1, "neighbor_count"] == 1.0
    assert by.loc[1, "ego_triangle_count"] == 0.0
    assert by.loc[3, "second_ring_count"] == 0.0


def test_extract_label_features_waveA_from_array():
    """``adjacency=True`` Wave A works on an IN-MEMORY array input too."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    df = extract_label_features(vol, n_workers=1, adjacency=True, max_radius=5)
    for c in _WAVEA_STATIC:
        assert c in df.columns, c
    assert "area_vs_neighbor_mean_ratio" in df.columns

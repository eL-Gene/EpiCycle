"""Mop-up M feature tests — the ADJACENCY / GRAPH / CONTACT tail of the atlas,
completing six families to 100 % of their non-research candidate columns:

* Neighbourhood graph topology   — ``compute_graph_topology`` REMAINDER
        (``neighbor_degree_std`` / ``_max``, ``ego_component_count``,
         ``first_to_second_ring_ratio``, ``second_ring_size_ratio``,
         ``second_ring_degree_mean``).
* Graph motif / local network    — ``compute_graph_motifs`` REMAINDER
        (``ego_square_count``, ``ego_cycle_fraction``, ``neighbor_homophily_area``
         / ``_shape``, ``ego_betweenness`` / ``ego_pagerank`` /
         ``ego_eigenvector_centrality``).
* Contact patch geometry         — ``compute_contact_geometry`` REMAINDER
        (``contact_area_mean`` / ``_std``, ``contact_patch_eccentricity_mean`` /
         ``_max``, ``contact_patch_compactness_mean``,
         ``contact_centroid_offset_mean`` / ``_max``, ``contact_side_asymmetry``,
         ``contact_patch_count_mean``).
* Directional contact            — ``compute_directional_contact`` REMAINDER
        (``dominant_contact_axis``, ``contact_hemisphere_balance``,
         ``contact_direction_entropy``, ``contact_sector_count``,
         ``dominant_contact_neighbor_fraction``).
* Local comparative              — ``compute_local_comparative`` REMAINDER
        (``<f>_vs_neighbor_median_ratio``, ``_minus_neighbor_mean``,
         ``_vs_contact_weighted_neighbor_mean``) + 3 more base features.
* Watershed territory / crowding — ``compute_watershed_territory`` REMAINDER
        (``ws_neighbor_count``, ``ws_nearest_gap``, ``territory_z_bias``,
         ``territory_surface_area``).

Pure-function known-shape tests (no tile readers / GPU). Plus: the manifest tags
every new column + the FeatureStage families registry stays exact-set in sync.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# --------------------------------------------------------------------------- #
# graph helpers (no volume) — mirror the Wave A test builders
# --------------------------------------------------------------------------- #
def _edge_list(edges, weight=10):
    return [{"label_a": int(a), "label_b": int(b), "contact_voxels": weight}
            for (a, b) in edges]


def _df_with_graph(labels, edges, extra_cols=None, weight=10):
    df = pd.DataFrame({"label": list(labels)})
    if extra_cols:
        for c, vals in extra_cols.items():
            df[c] = vals
    df.attrs["_adjacency_list"] = _edge_list(edges, weight=weight)
    return df


def _triangle():
    return [1, 2, 3], [(1, 2), (2, 3), (1, 3)]


def _star(n_leaves=4):
    leaves = list(range(2, 2 + n_leaves))
    return [1, *leaves], [(1, lf) for lf in leaves]


def _square_cycle():
    """A 4-cycle 1-2-3-4-1 with NO diagonals (so squares but no triangles)."""
    return [1, 2, 3, 4], [(1, 2), (2, 3), (3, 4), (4, 1)]


# --------------------------------------------------------------------------- #
# — graph topology REMAINDER
# --------------------------------------------------------------------------- #
_TOPO_NEW = ("neighbor_degree_std", "neighbor_degree_max", "ego_component_count",
             "first_to_second_ring_ratio", "second_ring_size_ratio",
             "second_ring_degree_mean")


def test_topology_remainder_columns_present():
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    for c in _TOPO_NEW:
        assert c in df.columns, c


def test_topology_ego_component_count_triangle_is_one():
    """A clique's neighbours are mutually connected -> ONE induced component."""
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    assert by.loc[1, "ego_component_count"] == pytest.approx(1.0)


def test_topology_ego_component_count_star_equals_leaves():
    """A star hub's leaves are mutually UNconnected -> each leaf its own component."""
    labels, edges = _star(n_leaves=4)
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    assert by.loc[1, "ego_component_count"] == pytest.approx(4.0)


def test_topology_neighbor_degree_spread_on_star_hub_is_zero():
    """A 4-star hub: all four leaves have degree 1 -> std 0, max 1."""
    labels, edges = _star(n_leaves=4)
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    assert by.loc[1, "neighbor_degree_std"] == pytest.approx(0.0)
    assert by.loc[1, "neighbor_degree_max"] == pytest.approx(1.0)


def test_topology_degree_spread_nonzero_on_mixed_neighbourhood():
    """Chain 1-2-3-4-5: node 3's neighbours are 2 and 4, both degree 2 -> std 0;
    but node 2's neighbours are 1 (deg 1) + 3 (deg 2) -> std > 0, max 2."""
    labels = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    assert by.loc[2, "neighbor_degree_std"] > 0.0
    assert by.loc[2, "neighbor_degree_max"] == pytest.approx(2.0)


def test_topology_ring_ratios_on_chain():
    """Chain 1-2-3-4: node 1 has k=1, second ring {3} (size 1).
    first_to_second = 1/1 = 1; second_ring_size_ratio = 1/1 = 1;
    second_ring_degree_mean = deg(3) = 2."""
    labels = [1, 2, 3, 4]
    edges = [(1, 2), (2, 3), (3, 4)]
    df = _df_with_graph(labels, edges)
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    assert by.loc[1, "first_to_second_ring_ratio"] == pytest.approx(1.0)
    assert by.loc[1, "second_ring_size_ratio"] == pytest.approx(1.0)
    assert by.loc[1, "second_ring_degree_mean"] == pytest.approx(2.0)


def test_topology_remainder_isolated_cell_nan():
    df = _df_with_graph([1, 2, 3], [(1, 2)])    # 3 isolated
    qf.compute_graph_topology(df)
    by = df.set_index("label")
    for c in _TOPO_NEW:
        assert np.isnan(by.loc[3, c]), c


def test_topology_remainder_empty_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_graph_topology(df)
    for c in _TOPO_NEW:
        assert c in df.columns


# --------------------------------------------------------------------------- #
# — graph motif REMAINDER
# --------------------------------------------------------------------------- #
_MOTIF_NEW = ("ego_square_count", "ego_cycle_fraction",
              "neighbor_homophily_area", "neighbor_homophily_shape",
              "ego_betweenness", "ego_pagerank", "ego_eigenvector_centrality")


def test_motif_remainder_columns_present():
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges, extra_cols={
        "area": [10.0, 10.0, 10.0], "sphericity": [0.5, 0.5, 0.5]})
    qf.compute_graph_motifs(df)
    for c in _MOTIF_NEW:
        assert c in df.columns, c


def test_motif_square_count_on_4cycle():
    """4-cycle 1-2-3-4 (no diagonals): node 1's neighbours {2,4} share common
    neighbour 3 (≠ 1) -> 1 square; no triangle."""
    labels, edges = _square_cycle()
    df = _df_with_graph(labels, edges, extra_cols={"area": [10.0] * 4})
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    assert by.loc[1, "ego_square_count"] == pytest.approx(1.0)
    assert by.loc[1, "ego_triangle_count"] == pytest.approx(0.0)


def test_motif_triangle_has_no_squares():
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges, extra_cols={"area": [10.0] * 3})
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    assert by.loc[1, "ego_square_count"] == pytest.approx(0.0)
    # the single neighbour pair forms a triangle -> cycle fraction == 1.
    assert by.loc[1, "ego_cycle_fraction"] == pytest.approx(1.0)


def test_motif_cycle_fraction_zero_for_star():
    """A 4-star hub: its leaves are mutually unconnected -> no triangle / square
    through any pair -> cycle fraction 0."""
    labels, edges = _star(n_leaves=4)
    df = _df_with_graph(labels, edges, extra_cols={"area": [10.0] * 5})
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    assert by.loc[1, "ego_cycle_fraction"] == pytest.approx(0.0)


def test_motif_homophily_high_for_uniform_neighbours():
    """A hub whose neighbours all have the SAME area -> homophily 1 (cv 0)."""
    labels, edges = _star(n_leaves=4)
    areas = [99.0, 10.0, 10.0, 10.0, 10.0]   # leaves identical
    df = _df_with_graph(labels, edges, extra_cols={"area": areas})
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    assert by.loc[1, "neighbor_homophily_area"] == pytest.approx(1.0)


def test_motif_homophily_lower_for_mixed_neighbours():
    """Spread-out neighbour areas -> homophily < 1."""
    labels, edges = _star(n_leaves=4)
    areas = [50.0, 2.0, 10.0, 30.0, 80.0]    # very mixed leaves
    df = _df_with_graph(labels, edges, extra_cols={"area": areas})
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    h = by.loc[1, "neighbor_homophily_area"]
    assert 0.0 < h < 1.0


def test_motif_homophily_shape_nan_without_shape_column():
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges, extra_cols={"area": [10.0] * 3})  # no shape col
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    assert np.isnan(by.loc[1, "neighbor_homophily_shape"])
    # area homophily still defined (uniform -> 1).
    assert by.loc[1, "neighbor_homophily_area"] == pytest.approx(1.0)


def test_motif_centrality_bounded_and_symmetric_on_triangle():
    """Symmetric triangle: every node has identical pagerank / eigenvector (1/3)
    and zero betweenness (no bridges)."""
    labels, edges = _triangle()
    df = _df_with_graph(labels, edges, extra_cols={"area": [10.0] * 3})
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    for lid in (1, 2, 3):
        assert by.loc[lid, "ego_pagerank"] == pytest.approx(1.0 / 3.0, abs=1e-3)
        assert by.loc[lid, "ego_eigenvector_centrality"] == pytest.approx(
            1.0 / 3.0, abs=1e-3)
        assert by.loc[lid, "ego_betweenness"] == pytest.approx(0.0, abs=1e-9)


def test_motif_betweenness_high_for_star_hub():
    """A star hub bridges all its leaves (each leaf-pair shortest path goes through
    it) -> betweenness 1; the leaves have betweenness 0."""
    labels, edges = _star(n_leaves=4)
    df = _df_with_graph(labels, edges, extra_cols={"area": [10.0] * 5})
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    assert by.loc[1, "ego_betweenness"] == pytest.approx(1.0, abs=1e-6)
    assert by.loc[2, "ego_betweenness"] == pytest.approx(0.0, abs=1e-9)


def test_motif_centrality_isolated_cell_nan():
    df = _df_with_graph([1, 2, 3], [(1, 2)], extra_cols={"area": [10.0] * 3})
    qf.compute_graph_motifs(df)
    by = df.set_index("label")
    assert np.isnan(by.loc[3, "ego_betweenness"])
    assert np.isnan(by.loc[3, "ego_pagerank"])
    assert np.isnan(by.loc[3, "ego_eigenvector_centrality"])


# --------------------------------------------------------------------------- #
# — contact patch geometry REMAINDER (needs a real volume)
# --------------------------------------------------------------------------- #
_CONTACT_NEW = ("contact_area_mean", "contact_area_std",
                "contact_patch_eccentricity_mean", "contact_patch_eccentricity_max",
                "contact_patch_compactness_mean",
                "contact_centroid_offset_mean", "contact_centroid_offset_max",
                "contact_side_asymmetry", "contact_patch_count_mean")


def _two_cells_flat_face():
    """Two cells touching at a single flat X-face (one connected planar contact)."""
    lab = np.zeros((10, 12, 24), np.int32)
    lab[2:8, 2:10, 2:11] = 1
    lab[2:8, 2:10, 11:20] = 2
    return lab


def _cell_with_two_neighbours_both_sides():
    """Cell 1 sandwiched between neighbours on its low-X and high-X sides
    (balanced, two-sided contact)."""
    lab = np.zeros((10, 12, 30), np.int32)
    lab[2:8, 2:10, 10:19] = 1
    lab[2:8, 2:10, 2:10] = 2          # low-X neighbour
    lab[2:8, 2:10, 19:27] = 3         # high-X neighbour
    return lab


def _contact_df(lab, max_radius=3):
    df = qf.extract_features(lab)
    qf.compute_neighbor_features(lab, df, max_radius=max_radius)
    qf.compute_contact_geometry(lab, df, max_radius=max_radius)
    return df


def test_contact_remainder_columns_present():
    df = _contact_df(_two_cells_flat_face())
    for c in _CONTACT_NEW:
        assert c in df.columns, c


def test_contact_single_neighbour_one_patch():
    df = _contact_df(_two_cells_flat_face())
    r = df.set_index("label").loc[1]
    # one neighbour -> std 0; one connected planar patch -> patch_count_mean 1.
    assert r["contact_area_std"] == pytest.approx(0.0)
    assert r["contact_patch_count_mean"] == pytest.approx(1.0)
    assert r["contact_area_mean"] > 0.0


def test_contact_side_asymmetry_one_sided_vs_balanced():
    """A single-neighbour cell has all its contact on one side (asymmetry ~1); a
    cell with neighbours on opposite sides is more balanced (lower asymmetry)."""
    r_one = _contact_df(_two_cells_flat_face()).set_index("label").loc[1]
    r_bal = _contact_df(_cell_with_two_neighbours_both_sides()).set_index("label").loc[1]
    assert r_one["contact_side_asymmetry"] == pytest.approx(1.0, abs=0.05)
    assert r_bal["contact_side_asymmetry"] < r_one["contact_side_asymmetry"]


def test_contact_geometry_bounded_shape_columns():
    df = _contact_df(_cell_with_two_neighbours_both_sides())
    for c in ("contact_patch_eccentricity_mean", "contact_patch_eccentricity_max",
              "contact_patch_compactness_mean", "contact_side_asymmetry"):
        v = df[c].values
        finite = v[np.isfinite(v)]
        assert np.all(finite >= 0.0) and np.all(finite <= 1.0), (c, finite)


def test_contact_area_mean_equals_total_over_neighbours():
    """contact_area_mean × (#contacting neighbours) ≈ contact_area_total."""
    df = _contact_df(_cell_with_two_neighbours_both_sides())
    r = df.set_index("label").loc[1]
    # two neighbours -> mean × 2 == total (areas may differ but the mean identity holds).
    assert r["contact_area_mean"] * 2.0 == pytest.approx(r["contact_area_total"], rel=0.01)


def test_contact_remainder_empty_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_contact_geometry(np.zeros((4, 4, 4), np.int32), df)
    for c in _CONTACT_NEW:
        assert c in df.columns


# --------------------------------------------------------------------------- #
# — directional contact REMAINDER (sphere-of-cells, needs Wave C normal)
# --------------------------------------------------------------------------- #
_DIR_NEW = ("dominant_contact_axis", "contact_hemisphere_balance",
            "contact_direction_entropy", "contact_sector_count",
            "dominant_contact_neighbor_fraction")


def _ball(R, c, shape):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= R * R


def _sphere_of_cells(shape=(40, 40, 40), R=18, cube=8):
    lab = np.zeros(shape, dtype=np.int32)
    c = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 2], dtype=np.float64)
    ball = _ball(R, c, shape)
    next_id = 1
    for z0 in range(0, shape[0], cube):
        for y0 in range(0, shape[1], cube):
            for x0 in range(0, shape[2], cube):
                block = np.zeros(shape, dtype=bool)
                block[z0:z0 + cube, y0:y0 + cube, x0:x0 + cube] = True
                cell = block & ball
                if cell.sum() < 20:
                    continue
                lab[cell] = next_id
                next_id += 1
    return lab


def _sphere_directional():
    """Sphere of cells + Wave C tissue_depth + adjacency, then directional contact."""
    lab = _sphere_of_cells()
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    df["label"] = df["label"].astype(np.int64)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    qf.compute_tissue_depth(lab, df)
    qf.compute_directional_contact(lab, df, max_radius=5)
    return lab, df


def test_directional_remainder_columns_present_and_bounded():
    lab, df = _sphere_directional()
    for c in _DIR_NEW:
        assert c in df.columns, c
    for c in ("dominant_contact_axis", "contact_hemisphere_balance",
              "contact_direction_entropy", "dominant_contact_neighbor_fraction"):
        v = df[c].values
        finite = v[np.isfinite(v)]
        if finite.size:
            assert np.all(finite >= -1e-9) and np.all(finite <= 1.0 + 1e-9), (c, finite)


def test_directional_sector_count_within_range():
    lab, df = _sphere_directional()
    v = df["contact_sector_count"].values
    finite = v[np.isfinite(v)]
    if finite.size:
        assert np.all(finite >= 0.0)
        assert np.all(finite <= qf._CONTACT_AZIMUTH_NSECTORS)


def test_directional_remainder_skips_without_waveC():
    """No tissue_outward_* columns -> the whole family (incl. the new columns) is
    absent (the default-off contract)."""
    lab = _two_cells_flat_face()
    df = qf.extract_features(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    qf.compute_directional_contact(lab, df, max_radius=3)   # no Wave C ran
    for c in _DIR_NEW:
        assert c not in df.columns, c


def test_directional_remainder_empty_schema_with_waveC_attempt():
    df = pd.DataFrame(columns=["label", "tissue_outward_z", "tissue_outward_y",
                               "tissue_outward_x", "tissue_outward_confidence",
                               "centroid-0", "centroid-1", "centroid-2"])
    qf.compute_directional_contact(np.zeros((4, 4, 4), np.int32), df)
    for c in _DIR_NEW:
        assert c in df.columns


# --------------------------------------------------------------------------- #
# — local comparative REMAINDER
# --------------------------------------------------------------------------- #
_LC_NEW = ("_vs_neighbor_median_ratio", "_minus_neighbor_mean",
           "_vs_contact_weighted_neighbor_mean")


def test_local_comparative_remainder_columns_present():
    labels, edges = _star(n_leaves=4)
    areas = [20.0, 10.0, 10.0, 10.0, 10.0]
    df = _df_with_graph(labels, edges, extra_cols={"area": areas})
    qf.compute_local_comparative(df, features=("area",))
    for suf in _LC_NEW:
        assert f"area{suf}" in df.columns, suf


def test_local_comparative_median_ratio_and_difference():
    """Hub area 20, four leaves area 10: median ratio 2.0, minus_mean = 20-10 = 10."""
    labels, edges = _star(n_leaves=4)
    areas = [20.0, 10.0, 10.0, 10.0, 10.0]
    df = _df_with_graph(labels, edges, extra_cols={"area": areas})
    qf.compute_local_comparative(df, features=("area",))
    by = df.set_index("label")
    assert by.loc[1, "area_vs_neighbor_median_ratio"] == pytest.approx(2.0)
    assert by.loc[1, "area_minus_neighbor_mean"] == pytest.approx(10.0)


def test_local_comparative_contact_weighted_mean_weights_dominant_neighbour():
    """Hub area 20; one leaf (id 2) shares a HUGE contact (weight 100, area 4), the
    other three small contacts (weight 1, area 16). The contact-weighted neighbour
    mean is pulled toward the dominant leaf's area (≈4), so the ratio >> the
    unweighted ratio (which uses mean area 13)."""
    labels = [1, 2, 3, 4, 5]
    edges = [(1, 2), (1, 3), (1, 4), (1, 5)]
    adj = [
        {"label_a": 1, "label_b": 2, "contact_voxels": 100},
        {"label_a": 1, "label_b": 3, "contact_voxels": 1},
        {"label_a": 1, "label_b": 4, "contact_voxels": 1},
        {"label_a": 1, "label_b": 5, "contact_voxels": 1},
    ]
    df = pd.DataFrame({"label": labels, "area": [20.0, 4.0, 16.0, 16.0, 16.0]})
    df.attrs["_adjacency_list"] = adj
    qf.compute_local_comparative(df, features=("area",))
    by = df.set_index("label")
    cw = by.loc[1, "area_vs_contact_weighted_neighbor_mean"]
    unw = by.loc[1, "area_vs_neighbor_mean_ratio"]
    # contact-weighted mean ≈ 4 (dominated by leaf 2) -> ratio ≈ 5; unweighted mean
    # = (4+16+16+16)/4 = 13 -> ratio ≈ 1.54. weighted ratio is much larger.
    assert cw > unw
    assert cw == pytest.approx(20.0 / ((100 * 4 + 16 + 16 + 16) / 103.0), rel=0.01)


def test_local_comparative_contact_weighted_falls_back_to_unweighted():
    """When no edge carries a contact weight, the contact-weighted variant degrades
    to the plain mean ratio."""
    labels, edges = _star(n_leaves=4)
    areas = [20.0, 10.0, 10.0, 10.0, 10.0]
    df = _df_with_graph(labels, edges, extra_cols={"area": areas}, weight=0)
    qf.compute_local_comparative(df, features=("area",))
    by = df.set_index("label")
    assert (by.loc[1, "area_vs_contact_weighted_neighbor_mean"]
            == pytest.approx(by.loc[1, "area_vs_neighbor_mean_ratio"]))


def test_local_comparative_remainder_isolated_cell_nan():
    df = _df_with_graph([1, 2, 3], [(1, 2)], extra_cols={"area": [5.0, 5.0, 5.0]})
    qf.compute_local_comparative(df, features=("area",))
    by = df.set_index("label")
    for suf in _LC_NEW:
        assert np.isnan(by.loc[3, f"area{suf}"]), suf


def test_local_comparative_extended_base_feature_set():
    """The default base set now includes axis_minor_length / solidity / extent."""
    assert "axis_minor_length" in qf.LOCAL_COMPARATIVE_FEATURES
    assert "solidity" in qf.LOCAL_COMPARATIVE_FEATURES
    assert "extent" in qf.LOCAL_COMPARATIVE_FEATURES


# --------------------------------------------------------------------------- #
# — watershed territory REMAINDER (crowded vs sparse cells)
# --------------------------------------------------------------------------- #
_WS_NEW = ("ws_neighbor_count", "ws_nearest_gap",
           "territory_z_bias", "territory_surface_area")


def _crowded_and_sparse():
    """Crowded pair (cells 1 & 2 abutting) + an isolated cell 3 far away."""
    lab = np.zeros((20, 20, 40), np.int32)
    lab[6:14, 6:14, 4:11] = 1
    lab[6:14, 6:14, 11:18] = 2      # abuts cell 1
    lab[6:14, 6:14, 28:36] = 3      # isolated
    return lab


def _territory_df(lab, max_distance_voxels=8.0, spacing=None):
    df = qf.extract_features(lab, spacing=spacing)
    df["label"] = df["label"].astype(np.int64)
    qf.compute_watershed_territory(lab, df, spacing=spacing,
                                   max_distance_voxels=max_distance_voxels)
    return df


def test_territory_remainder_columns_present():
    df = _territory_df(_crowded_and_sparse())
    for c in _WS_NEW:
        assert c in df.columns, c


def test_territory_ws_neighbor_count_crowded_vs_isolated():
    """The abutting pair have ws-neighbours; the isolated cell may still reach the
    pair's territory within the cap but has a much larger nearest gap."""
    df = _territory_df(_crowded_and_sparse()).set_index("label")
    # cell 1 touches cell 2 -> at least one ws-neighbour; gap ~0 (abutting).
    assert df.loc[1, "ws_neighbor_count"] >= 1.0
    assert df.loc[1, "ws_nearest_gap"] == pytest.approx(0.0, abs=1.5)
    # cell 3 is isolated; if it has any ws-neighbour the gap is large (>> the pair's).
    if np.isfinite(df.loc[3, "ws_nearest_gap"]):
        assert df.loc[3, "ws_nearest_gap"] > df.loc[1, "ws_nearest_gap"]


def test_territory_surface_area_positive():
    df = _territory_df(_crowded_and_sparse())
    v = df["territory_surface_area"].values
    finite = v[np.isfinite(v)]
    assert finite.size > 0 and np.all(finite > 0.0)


def test_territory_z_bias_bounded():
    df = _territory_df(_crowded_and_sparse())
    v = df["territory_z_bias"].values
    finite = v[np.isfinite(v)]
    if finite.size:
        assert np.all(finite >= -1.0 - 1e-9) and np.all(finite <= 1.0 + 1e-9)


def test_territory_z_bias_detects_asymmetry():
    """A cell pressed against the LOW-Z wall (tissue extends only upward) gets a
    POSITIVE z-bias (its territory sits above it in Z)."""
    lab = np.zeros((30, 16, 16), np.int32)
    # cell sits low in Z; lots of open space above it (high Z) up to the cap.
    lab[1:6, 4:12, 4:12] = 1
    df = _territory_df(lab, max_distance_voxels=12.0)
    z = df.set_index("label").loc[1, "territory_z_bias"]
    assert np.isfinite(z) and z > 0.0


def test_territory_surface_area_scales_with_spacing():
    """Anisotropic spacing scales the surface area (physical units)."""
    lab = _crowded_and_sparse()
    df_v = _territory_df(lab, spacing=None)
    df_p = _territory_df(lab, spacing=(2.0, 1.0, 1.0))
    s_v = df_v.set_index("label").loc[3, "territory_surface_area"]
    s_p = df_p.set_index("label").loc[3, "territory_surface_area"]
    assert s_p > s_v       # physical face areas are larger with 2x Z pitch


def test_territory_remainder_empty_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_watershed_territory(np.zeros((6, 6, 6), np.int32), df)
    for c in _WS_NEW:
        assert c in df.columns


def test_territory_remainder_no_cells_nan():
    df = pd.DataFrame({"label": [1, 2]})
    qf.compute_watershed_territory(np.zeros((6, 6, 6), np.int32), df)
    for c in _WS_NEW:
        assert df[c].isna().all(), c


# --------------------------------------------------------------------------- #
# manifest + FeatureStage family registry (exact-set in lockstep)
# --------------------------------------------------------------------------- #
def test_manifest_tags_all_new_columns():
    man = fm.get_manifest()
    static_new = (*_TOPO_NEW, *_MOTIF_NEW, *_CONTACT_NEW, *_DIR_NEW, *_WS_NEW)
    for c in static_new:
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        t = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in t, (c, key)
        assert t["role"] in ("biology", "qc", "covariate")
        assert t["frame"] in ("image", "tissue", "intrinsic")
        assert isinstance(t["bounded"], bool)


def test_manifest_local_comparative_remainder_tags():
    tags = fm.local_comparative_tags(("area", "sphericity"))
    for base in ("area", "sphericity"):
        for suf in _LC_NEW:
            col = f"{base}{suf}"
            assert col in tags, col
            assert tags[col]["redundancy_group"] == f"local_comparative_{base}"


def test_feature_stage_families_exact_set_after_mopupM():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["graph_topology"]) == set(qf.GRAPH_TOPOLOGY_COLUMNS)
    assert set(_FEATURE_FAMILIES["graph_motif"]) == set(qf.GRAPH_MOTIF_COLUMNS)
    assert set(_FEATURE_FAMILIES["contact_geometry"]) == set(qf.CONTACT_GEOMETRY_COLUMNS)
    assert (set(_FEATURE_FAMILIES["directional_contact"])
            == set(qf.DIRECTIONAL_CONTACT_COLUMNS))
    assert (set(_FEATURE_FAMILIES["watershed_territory"])
            == set(qf.WATERSHED_TERRITORY_COLUMNS))


def test_local_comparative_families_recovered_with_remainder_suffixes():
    from packages.epicycle_scaledseg.core.feature_stage import _local_comparative_families
    cols = ["label"]
    for suf in ("_vs_neighbor_mean_ratio", "_local_zscore", *_LC_NEW):
        cols.append(f"area{suf}")
    fams = _local_comparative_families(cols)
    assert set(fams["local_comparative_area"]) == {f"area{s}" for s in (
        "_vs_neighbor_mean_ratio", "_local_zscore", *_LC_NEW)}


# --------------------------------------------------------------------------- #
# opt-in global pass wiring (default-off byte-identical)
# --------------------------------------------------------------------------- #
def _toy_two_cells(tmp_path):
    vol = np.zeros((8, 16, 30), np.int32)
    vol[1:6, 2:8, 2:9] = 1
    vol[1:6, 2:8, 9:15] = 2          # touches cell 1
    p = tmp_path / "labels.npy"
    np.save(p, vol)
    return str(p)


def test_extract_label_features_adjacency_emits_mopupM(tmp_path):
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    p = _toy_two_cells(tmp_path)
    df_off = extract_label_features(p, n_workers=1)
    df_on = extract_label_features(p, n_workers=1, adjacency=True, max_radius=5)
    # the remainder columns appear only with adjacency.
    for c in (*_TOPO_NEW, *_MOTIF_NEW, *_CONTACT_NEW):
        assert c in df_on.columns, c
        assert c not in df_off.columns, c
    # local-comparative remainder appears for the present base features.
    assert "area_vs_neighbor_median_ratio" in df_on.columns
    # default-off morphology columns byte-identical (additive contract).
    shared = [c for c in df_off.columns if c in df_on.columns]
    pd.testing.assert_frame_equal(
        df_on[shared].reset_index(drop=True),
        df_off[shared].reset_index(drop=True),
        check_dtype=False,
    )


def test_extract_label_features_territory_emits_mopupM(tmp_path):
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    p = _toy_two_cells(tmp_path)
    df_off = extract_label_features(p, n_workers=1)
    df_on = extract_label_features(p, n_workers=1, territory=True)
    for c in _WS_NEW:
        assert c in df_on.columns, c
        assert c not in df_off.columns, c


def test_extract_label_features_tissue_depth_emits_directional_mopupM(tmp_path):
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    p = _toy_two_cells(tmp_path)
    df_on = extract_label_features(p, n_workers=1, adjacency=True,
                                   tissue_depth=True, max_radius=5)
    for c in _DIR_NEW:
        assert c in df_on.columns, c

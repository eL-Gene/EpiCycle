"""Mop-up O feature tests — the CHANNEL / SIGNAL / FRAME / SPATIAL tail.

The LAST implementation wave. NEW features built from the EpiCycle atlas
equations, extending existing ``compute_*`` functions:

* single-volume channel localization remainder (``compute_channel_localization``):
  the 3-band radial-intensity profile + ``membrane_enrichment`` + ``outer_to_inner_ratio``
  + ``fraction_above_threshold`` + the lateral (equatorial) intensity band.
* compartment remainder (``compute_channel_compartments``, PAIRED path):
  nuclear signal fraction + per-compartment texture entropy + compartment mismatch.
* co-localization remainder (``compute_channel_colocalization``):
  A:B mean ratio + boundary-shell Pearson + interior-vs-boundary ratio delta.
* prob-geometry remainder (``compute_prob_geometry``): surface-normalized
  confidence + boundary-drop-per-area.
* local-tissue-frame remainder (``compute_tissue_frame_features``): the
  orientation-coherence battery (dispersion / contact-weighted coherence /
  polar dispersion / azimuth entropy / packing order / coordination anisotropy /
  frame stability / major-vs-normal angle).
* spatial context remainder (``compute_tissue_depth``): ``local_tissue_depth``.
* reliability remainder (``compute_reliability_meta``):
  ``neighbor_topology_reliability`` + ``probability_support_score``.

Pure-function known-shape tests (no GPU / Cellpose) on synthetic crops/fields
whose answers we know by construction: a membrane-bright vs core-bright channel,
a coherent hexagonal neighbour ring vs a clumped one, a tall well-connected cell
vs a thin isolated one. Also asserts the manifest tags every new column, the
default-off byte-identical contract, and the bounded columns stay in range.
"""

import math

import numpy as np
import pandas as pd

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# --------------------------------------------------------------------------- #
# new-column groups (kept in lockstep with the source constants)
# --------------------------------------------------------------------------- #
_LOC_NEW = (
    "_shell_mid_mean", "_radial_bin_inner_mean", "_radial_bin_mid_mean",
    "_radial_bin_outer_mean", "_membrane_enrichment", "_outer_to_inner_ratio",
    "_fraction_above_threshold", "_lateral_intensity_mean",
)
_COMPARTMENT_NEW = (
    "_nuclear_signal_fraction", "_nuclear_texture_entropy",
    "_cytoplasmic_texture_entropy", "_compartment_mismatch_score",
)
_COLOC_NEW = ("_mean_ratio", "_boundary_corr", "_interior_ratio_delta")
_PROB_GEOM_NEW = ("prob_surface_normalized", "prob_boundary_drop_per_area")
_FRAME_NEW = (
    "contact_weighted_axis_coherence", "neighbor_axis_dispersion",
    "major_axis_vs_local_normal_angle", "neighbor_polar_dispersion",
    "neighbor_azimuth_entropy", "local_packing_order",
    "coordination_anisotropy", "frame_stability",
)
_SPATIAL_NEW = ("local_tissue_depth",)
_RELIABILITY_NEW = ("neighbor_topology_reliability", "probability_support_score")


# --------------------------------------------------------------------------- #
# synthetic builders
# --------------------------------------------------------------------------- #
def _sphere_mask(shape=(30, 30, 30), radius=12.0):
    zz, yy, xx = np.indices(shape).astype(np.float64)
    c = (np.array(shape) - 1) / 2.0
    d = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    return d <= radius


def _radial_position(mask):
    r, _ = qf._edt_radial_position(mask, spacing=None)
    return r


def _membrane_signal(mask, lo=10.0, hi=200.0):
    """Bright at the surface rind (low r), dim in the core."""
    r = _radial_position(mask)
    img = np.zeros(mask.shape, dtype=np.float64)
    img[mask] = hi + (lo - hi) * r[mask]
    return img


def _core_signal(mask, lo=10.0, hi=200.0):
    r = _radial_position(mask)
    img = np.zeros(mask.shape, dtype=np.float64)
    img[mask] = lo + (hi - lo) * r[mask]
    return img


def _uniform(mask, value=100.0):
    return np.where(mask, value, 0.0).astype(np.float64)


def _cell_nucleus_crop(shape=(24, 24, 24), cell_r=10.0, nuc_r=5.0):
    cell = _sphere_mask(shape, cell_r)
    nuc = _sphere_mask(shape, nuc_r)
    return cell, nuc


def _hex_ring_df(jitter_to_clump=False, axis_vec=(1.0, 0.0, 0.0)):
    """A center cell + 6 in-plane (YX) neighbours arranged hexagonally (or clumped
    in one wedge when ``jitter_to_clump``), all with the same major axis."""
    cz = 15.0
    rows = [dict(label=1, **{"centroid-0": cz, "centroid-1": 15.0, "centroid-2": 15.0},
                 major_axis_z=axis_vec[0], major_axis_y=axis_vec[1], major_axis_x=axis_vec[2])]
    for k in range(6):
        th = (0.1 * k) if jitter_to_clump else (2.0 * math.pi * k / 6.0)
        rows.append(dict(
            label=k + 2,
            **{"centroid-0": cz, "centroid-1": 15.0 + 6.0 * math.cos(th),
               "centroid-2": 15.0 + 6.0 * math.sin(th)},
            major_axis_z=axis_vec[0], major_axis_y=axis_vec[1], major_axis_x=axis_vec[2]))
    df = pd.DataFrame(rows)
    adj = [{"label_a": 1, "label_b": j, "contact_voxels": 50} for j in range(2, 8)]
    df.attrs["_adjacency_list"] = adj
    return df, adj


# =========================================================================== #
# — compute_channel_localization (single-volume remainder)
# =========================================================================== #
def test_localization_new_columns_present():
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_membrane_signal(mask), mask, "mem")
    for s in _LOC_NEW:
        assert f"mem{s}" in out, s


def test_localization_membrane_signal_enriched_at_rim():
    """A membrane-bright channel -> outer/inner > 1, membrane_enrichment > 1, the
    3-band radial profile DECREASES inner->outer in r (so outer band brightest)."""
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_membrane_signal(mask), mask, "mem")
    assert out["mem_outer_to_inner_ratio"] > 1.0
    assert out["mem_membrane_enrichment"] > 1.0
    # the outer (surface) band is brighter than the inner (core) band.
    assert out["mem_radial_bin_outer_mean"] > out["mem_radial_bin_inner_mean"]


def test_localization_core_signal_opposite():
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_core_signal(mask), mask, "mem")
    assert out["mem_outer_to_inner_ratio"] < 1.0
    assert out["mem_membrane_enrichment"] < 1.0


def test_localization_fraction_above_threshold_bounded():
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_membrane_signal(mask), mask, "mem")
    f = out["mem_fraction_above_threshold"]
    assert 0.0 <= f <= 1.0


def test_localization_fraction_above_threshold_uniform_is_zero():
    """A uniform fill -> nothing is strictly above the mean -> fraction ~0."""
    mask = _sphere_mask()
    out = qf.compute_channel_localization(_uniform(mask, 50.0), mask, "mem")
    assert out["mem_fraction_above_threshold"] == 0.0


def test_localization_lateral_needs_outward_normal():
    """The lateral band (like apical/basal) is NaN without a trusted outward normal,
    finite with one."""
    mask = _sphere_mask()
    out_no = qf.compute_channel_localization(_uniform(mask), mask, "mem")
    assert np.isnan(out_no["mem_lateral_intensity_mean"])
    c = (np.array(mask.shape) - 1) / 2.0
    out_yes = qf.compute_channel_localization(
        _uniform(mask), mask, "mem", spacing=(1, 1, 1),
        outward=(1.0, 0.0, 0.0), centroid=tuple(c))
    assert np.isfinite(out_yes["mem_lateral_intensity_mean"])


# =========================================================================== #
# — compute_channel_compartments (PAIRED remainder)
# =========================================================================== #
def test_compartment_new_columns_present():
    cell, nuc = _cell_nucleus_crop()
    out = qf.compute_channel_compartments(_uniform(cell, 50.0), cell, nuc, "mem")
    for s in _COMPARTMENT_NEW:
        assert f"mem{s}" in out, s


def test_compartment_nuclear_signal_fraction_bounded_and_sensible():
    """A nucleus-bright channel piles its signal mass in the nucleus -> a high
    nuclear signal fraction (but < 1, since the cytoplasm carries a baseline)."""
    cell, nuc = _cell_nucleus_crop()
    img = np.where(nuc, 100.0, np.where(cell, 5.0, 0.0)).astype(np.float64)
    out = qf.compute_channel_compartments(img, cell, nuc, "nuc")
    frac = out["nuc_nuclear_signal_fraction"]
    assert 0.0 <= frac <= 1.0
    assert frac > 0.3  # the bright nucleus carries a large share of the signal mass


def test_compartment_mismatch_score_high_when_restricted():
    """A nucleus-only signal (cytoplasm ~0) -> a high compartment mismatch; a
    uniform signal -> ~0 mismatch."""
    cell, nuc = _cell_nucleus_crop()
    nuc_only = np.where(nuc, 100.0, 0.0).astype(np.float64)
    out_r = qf.compute_channel_compartments(nuc_only, cell, nuc, "nuc")
    out_u = qf.compute_channel_compartments(_uniform(cell, 50.0), cell, nuc, "nuc")
    assert out_r["nuc_compartment_mismatch_score"] > 0.5
    assert out_u["nuc_compartment_mismatch_score"] < 1e-6
    for o in (out_r, out_u):
        assert 0.0 <= o["nuc_compartment_mismatch_score"] <= 1.0


def test_compartment_texture_entropy_finite_for_nonuniform():
    cell, nuc = _cell_nucleus_crop()
    rng = np.random.default_rng(0)
    img = np.where(cell, rng.uniform(0, 100, cell.shape), 0.0)
    out = qf.compute_channel_compartments(img, cell, nuc, "nuc")
    assert np.isfinite(out["nuc_nuclear_texture_entropy"])
    assert np.isfinite(out["nuc_cytoplasmic_texture_entropy"])
    assert out["nuc_nuclear_texture_entropy"] >= 0.0


# =========================================================================== #
# — compute_channel_colocalization (remainder)
# =========================================================================== #
def test_coloc_new_columns_present():
    mask = _sphere_mask()
    out = qf.compute_channel_colocalization(
        _membrane_signal(mask), _core_signal(mask), mask, "a", "b")
    for s in _COLOC_NEW:
        assert f"a_b{s}" in out, s


def test_coloc_mean_ratio_tracks_intensity():
    """A:B mean ratio > 1 when A is brighter than B overall."""
    mask = _sphere_mask()
    a = _uniform(mask, 100.0)
    b = _uniform(mask, 25.0)
    out = qf.compute_channel_colocalization(a, b, mask, "a", "b")
    assert out["a_b_mean_ratio"] > 1.0
    assert abs(out["a_b_mean_ratio"] - 4.0) < 0.1


def test_coloc_boundary_corr_anticorrelated_signals():
    """An anti-correlated channel pair (B = C - A, with A textured) is anti-correlated
    within EVERY shell -> boundary_corr ~ -1 (and finite interior delta). The signals
    must vary WITHIN the boundary shell, so a textured A is used (a pure radial ramp is
    near-constant inside one thin shell -> degenerate Pearson there)."""
    mask = _sphere_mask()
    rng = np.random.default_rng(1)
    a = np.where(mask, rng.uniform(10.0, 200.0, mask.shape), 0.0)
    b = np.where(mask, 210.0 - a, 0.0)            # exact per-voxel anti-correlation
    out = qf.compute_channel_colocalization(a, b, mask, "a", "b")
    assert np.isfinite(out["a_b_boundary_corr"])
    assert out["a_b_boundary_corr"] < -0.9         # ~ -1 anti-correlated
    assert -1.0 <= out["a_b_boundary_corr"] <= 1.0
    assert np.isfinite(out["a_b_interior_ratio_delta"])


# =========================================================================== #
# — compute_prob_geometry (remainder)
# =========================================================================== #
def test_prob_geometry_new_columns_present():
    mask = _sphere_mask()
    out = qf.compute_prob_geometry(mask, _uniform(mask, 0.9))
    for c in _PROB_GEOM_NEW:
        assert c in out, c


def test_prob_geometry_surface_columns_need_surface_area():
    """The surface-conditioned columns are NaN without surface_area, finite with it."""
    mask = _sphere_mask()
    pr = _uniform(mask, 0.9)
    out_no = qf.compute_prob_geometry(mask, pr, prob_weighted_volume=100.0)
    assert np.isnan(out_no["prob_surface_normalized"])
    assert np.isnan(out_no["prob_boundary_drop_per_area"])
    out_yes = qf.compute_prob_geometry(
        mask, pr, prob_weighted_volume=100.0, surface_area=50.0,
        prob_boundary_drop=0.2)
    assert abs(out_yes["prob_surface_normalized"] - 2.0) < 1e-6   # 100 / 50
    assert abs(out_yes["prob_boundary_drop_per_area"] - 0.004) < 1e-6  # 0.2 / 50


# =========================================================================== #
# — compute_tissue_frame_features (orientation-coherence remainder)
# =========================================================================== #
def test_frame_new_columns_present():
    df, adj = _hex_ring_df()
    qf.compute_tissue_frame_features(df, adjacency=adj)
    for c in _FRAME_NEW:
        assert c in df.columns, c


def test_frame_coherent_ring_high_order():
    """A coherent hexagonal ring of aligned neighbours -> high azimuth entropy +
    packing order + contact-weighted coherence, ~0 axis dispersion."""
    df, adj = _hex_ring_df(jitter_to_clump=False)
    qf.compute_tissue_frame_features(df, adjacency=adj)
    r = df.set_index("label").loc[1]
    assert r.neighbor_azimuth_entropy > 0.8
    assert r.local_packing_order > 0.5
    assert r.contact_weighted_axis_coherence > 0.9     # all axes parallel
    assert r.neighbor_axis_dispersion < 1e-6           # no spread in alignment
    assert 0.0 <= r.frame_stability <= 1.0


def test_frame_clumped_ring_lower_order():
    """Neighbours clumped in one azimuthal wedge -> low azimuth entropy + low
    packing order vs the coherent ring."""
    df_c, adj = _hex_ring_df(jitter_to_clump=False)
    df_k, _ = _hex_ring_df(jitter_to_clump=True)
    qf.compute_tissue_frame_features(df_c, adjacency=adj)
    qf.compute_tissue_frame_features(df_k, adjacency=adj)
    rc = df_c.set_index("label").loc[1]
    rk = df_k.set_index("label").loc[1]
    assert rk.neighbor_azimuth_entropy < rc.neighbor_azimuth_entropy
    assert rk.local_packing_order < rc.local_packing_order


def test_frame_bounded_columns_in_range():
    df, adj = _hex_ring_df()
    qf.compute_tissue_frame_features(df, adjacency=adj)
    r = df.set_index("label").loc[1]
    for c in ("contact_weighted_axis_coherence", "neighbor_azimuth_entropy",
              "local_packing_order", "coordination_anisotropy", "frame_stability"):
        v = r[c]
        if np.isfinite(v):
            assert 0.0 <= v <= 1.0, (c, v)
    if np.isfinite(r.major_axis_vs_local_normal_angle):
        assert 0.0 <= r.major_axis_vs_local_normal_angle <= 90.0


def test_frame_few_neighbours_nan():
    """A cell with < 3 neighbours gets NaN for every frame column (the plane fit is
    undetermined) — the new columns inherit the existing guard."""
    df = pd.DataFrame({
        "label": [1, 2],
        "centroid-0": [5.0, 6.0], "centroid-1": [5.0, 6.0], "centroid-2": [5.0, 6.0],
        "major_axis_z": [1.0, 1.0], "major_axis_y": [0.0, 0.0], "major_axis_x": [0.0, 0.0],
    })
    adj = [{"label_a": 1, "label_b": 2, "contact_voxels": 10}]
    df.attrs["_adjacency_list"] = adj
    qf.compute_tissue_frame_features(df, adjacency=adj)
    for c in _FRAME_NEW:
        assert df[c].isna().all(), c


# =========================================================================== #
# — compute_tissue_depth (local_tissue_depth)
# =========================================================================== #
def _toy_tissue():
    """Two touching blocks (a tiny slab with a surface)."""
    lab = np.zeros((16, 32, 32), np.int32)
    lab[3:13, 4:28, 4:16] = 1
    lab[3:13, 4:28, 16:28] = 2
    df = pd.DataFrame({
        "label": [1, 2],
        "centroid-0": [8.0, 8.0], "centroid-1": [16.0, 16.0],
        "centroid-2": [10.0, 22.0],
    })
    return lab, df


def test_local_tissue_depth_present_and_finite():
    lab, df = _toy_tissue()
    qf.compute_tissue_depth(lab, df, spacing=(1, 1, 1))
    assert "local_tissue_depth" in df.columns
    assert df["local_tissue_depth"].notna().all()
    assert (df["local_tissue_depth"] >= 0).all()


def test_local_tissue_depth_defined_without_centroids():
    """local_tissue_depth only needs the cell mask (not centroids), so it is defined
    even when the spatial-context columns are NaN."""
    lab, df = _toy_tissue()
    df_nocen = pd.DataFrame({"label": [1, 2]})
    qf.compute_tissue_depth(lab, df_nocen)
    assert df_nocen["local_tissue_depth"].notna().all()
    assert df_nocen["centroid_z_relative"].isna().all()


def test_local_tissue_depth_in_typed_columns_empty():
    df = pd.DataFrame()
    qf.compute_tissue_depth(np.zeros((4, 4, 4), np.int32), df)
    assert "local_tissue_depth" in df.columns


# =========================================================================== #
# — compute_reliability_meta (remainder)
# =========================================================================== #
def test_reliability_new_columns_present():
    df = pd.DataFrame({"label": [1], "z_span": [5.0]})
    qf.compute_reliability_meta(df)
    for c in _RELIABILITY_NEW:
        assert c in df.columns, c


def test_neighbor_topology_reliability_tall_connected_vs_thin_isolated():
    df = pd.DataFrame({
        "label": [1, 2],
        "z_span": [10.0, 1.0],
        "neighbor_count": [5, 0],
    })
    qf.compute_reliability_meta(df)
    vals = df["neighbor_topology_reliability"].values
    assert vals[0] > vals[1]            # tall + connected > thin + isolated
    assert vals[1] == 0.0               # 1-slice isolated -> 0
    for v in vals:
        assert 0.0 <= v <= 1.0


def test_probability_support_score_confident_vs_uncertain():
    df = pd.DataFrame({
        "label": [1, 2],
        "prob_mean": [0.95, 0.4],
        "prob_boundary_mean": [0.8, 0.3],
        "prob_boundary_entropy": [0.05, 0.6],
    })
    qf.compute_reliability_meta(df)
    vals = df["probability_support_score"].values
    assert vals[0] > vals[1]
    for v in vals:
        assert 0.0 <= v <= 1.0


def test_probability_support_nan_on_labels_only():
    """A labels-only table (no prob columns) -> NaN probability support."""
    df = pd.DataFrame({"label": [1], "z_span": [6.0]})
    qf.compute_reliability_meta(df)
    assert df["probability_support_score"].isna().all()


# =========================================================================== #
# manifest tagging
# =========================================================================== #
def test_manifest_tags_static_new_columns():
    man = fm.get_manifest()
    for c in (*_PROB_GEOM_NEW, *_FRAME_NEW, *_SPATIAL_NEW, *_RELIABILITY_NEW):
        assert c in man, c
        tag = man[c]
        assert set(tag) >= {"units", "bounded", "role", "frame", "redundancy_group"}


def test_manifest_tags_templated_channel_columns():
    loc = fm.channel_localization_tags("mem")
    for s in _LOC_NEW:
        assert f"mem{s}" in loc, s
    comp = fm.channel_compartment_tags("nuc")
    for s in _COMPARTMENT_NEW:
        assert f"nuc{s}" in comp, s
    coloc = fm.channel_coloc_tags("nuc", "mem")
    for s in _COLOC_NEW:
        assert f"nuc_mem{s}" in coloc, s


def test_manifest_bounded_flags_match_value_ranges():
    """Spot-check: the columns the manifest marks bounded stay in [0,1] (or the
    angle in [0,90]) on the synthetic data above."""
    man = fm.get_manifest()
    # frame_stability + neighbor_azimuth_entropy bounded.
    df, adj = _hex_ring_df()
    qf.compute_tissue_frame_features(df, adjacency=adj)
    r = df.set_index("label").loc[1]
    for c in ("frame_stability", "neighbor_azimuth_entropy", "local_packing_order"):
        if man[c]["bounded"] and np.isfinite(r[c]):
            assert 0.0 <= r[c] <= 1.0, (c, r[c])


# =========================================================================== #
# default-off byte-identical contract
# =========================================================================== #
def test_localization_default_off_byte_identical():
    """A run with NO channels sampled never produces the channel-localization
    columns: the per-cell labels-only path is unchanged."""
    mask = _sphere_mask()
    df = qf.extract_features(mask.astype(np.int32))
    cols_before = set(df.columns)
    # localization is ONLY produced when a channel is sampled; on a labels-only
    # extract the new columns are absent.
    for s in _LOC_NEW:
        assert not any(c.endswith(s) for c in cols_before), s


def test_frame_columns_only_with_adjacency():
    """The frame columns ride the global adjacency pass; a bare extract_features
    table never carries them (default-off)."""
    mask = _sphere_mask()
    df = qf.extract_features(mask.astype(np.int32))
    for c in _FRAME_NEW:
        assert c not in df.columns, c


def test_reliability_remainder_always_runs_but_prob_term_optional():
    """compute_reliability_meta ALWAYS adds its columns (like the existing 4); the
    NEW neighbor-topology + prob-support degrade gracefully on a labels-only table
    (topology -> z_span ramp, prob-support -> NaN) rather than being absent."""
    df = pd.DataFrame({"label": [1, 2, 3], "z_span": [1.0, 3.0, 8.0]})
    qf.compute_reliability_meta(df)
    assert "neighbor_topology_reliability" in df.columns
    assert "probability_support_score" in df.columns
    # neighbor-topology degrades to the z_span ramp (no neighbor_count column):
    assert df["neighbor_topology_reliability"].iloc[0] < df["neighbor_topology_reliability"].iloc[2]
    # prob-support is NaN with no prob columns.
    assert df["probability_support_score"].isna().all()

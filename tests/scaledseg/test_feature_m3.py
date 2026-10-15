"""M3 feature tests — adjacency graph: neighbour + topology + composites (B4, B5, B6).

Pure-function known-shape tests (no tile readers / GPU). Builds a small synthetic
label volume of several touching + isolated cells and exercises the two ported
functions directly, then the OPT-IN global pass wired into ``extract_label_features``.

Asserts:

* :func:`compute_neighbor_features` (B4) — ``neighbor_count`` is correct for the
  known adjacencies; an isolated cell reads ``neighbor_count == 0``;
  ``max_contact_fraction`` ∈ [0,1]; ``background_fraction`` ∈ [0,1]; the
  ``df.attrs["_adjacency_list"]`` edge list contains the touching pair (and not
  the isolated cell), de-duplicated (label_a < label_b).
* The z-stratified topology (B5) columns behave — a tall cell touching neighbours
  only at its base reads basal neighbours > apical, and ``neighbor_z_turnover`` /
  ``neighbor_persistence_ratio`` ∈ [0,1].
* :func:`compute_named_composites` (B6) emits merge_signal / fragment_signal /
  boundary_confidence / shape_abnormality, and degrades gracefully (no crash,
  NaN boundary_confidence) when the prob_* columns are absent.
* Empty-df schema for both functions (every column still emitted).
* The manifest tags every new (static) M3 column.
* The ``_FEATURE_FAMILIES`` registry lists the adjacency / topology / composite
  columns.
* The opt-in global pass: ``extract_label_features(adjacency=True)`` adds the 21
  neighbour/topology + 4 composite columns and carries ``_adjacency_list``; and
  ``extract_label_features(adjacency=False)`` is byte-identical to the prior
  (M2 labels-only) output — the additive-optional contract.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


_B4_COLS = (
    "neighbor_count", "nearest_neighbor_distance",
    "max_contact_fraction", "background_fraction",
    "nn_count_r1", "nn_count_r3", "nn_birth_median", "nn_birth_std",
    "nn_accumulation_rate", "nn_accumulation_knee",
    "mean_neighbor_size_ratio", "neighbor_size_cv",
    "contact_weighted_neighbor_size_ratio",
)
_B5_COLS = (
    "basal_neighbor_count", "apical_neighbor_count",
    "neighbor_z_turnover", "neighbor_persistence_ratio",
)
_B6_COLS = ("merge_signal", "fragment_signal",
            "boundary_confidence", "shape_abnormality")
_ADJ_COLS = _B4_COLS + _B5_COLS


# --------------------------------------------------------------------------- #
# synthetic volumes
# --------------------------------------------------------------------------- #

def _two_touching_plus_isolated():
    """Three cells: 1 and 2 TOUCH (share a face); 3 is ISOLATED (far away).

    Layout in a (10, 10, 30) volume (Z, Y, X):
        cell 1: x in [2, 8)   |  cell 2: x in [8, 14)  -> faces meet at x==8
        cell 3: x in [22, 28) -> separated by >> max_radius from 1/2
    All span the full Y and a few Z slices so they are real 3D blocks.
    """
    lab = np.zeros((10, 10, 30), dtype=np.int32)
    lab[2:8, 2:8, 2:8] = 1
    lab[2:8, 2:8, 8:14] = 2          # shares the x==8 face with cell 1
    lab[2:8, 2:8, 22:28] = 3         # isolated
    return lab


def _tall_basal_neighbors():
    """A tall cell (id 1) spanning many Z slices, with neighbours that touch only
    near its BASE (low Z). Used to exercise the z-stratified topology (B5).

    cell 1: a tall column x in [4, 8), z in [0, 12)
    cell 2: a basal neighbour touching at low Z only (z in [0, 4), x in [8, 12))
    cell 3: a basal neighbour touching at low Z only (z in [0, 4), x in [0, 4))
    """
    lab = np.zeros((14, 8, 16), dtype=np.int32)
    lab[0:12, 2:6, 4:8] = 1
    lab[0:4, 2:6, 8:12] = 2          # touches cell 1 at the x==8 face, low Z only
    lab[0:4, 2:6, 0:4] = 3           # touches cell 1 at the x==4 face, low Z only
    return lab


def _df_for(lab):
    """extract_features over a MULTI-label volume -> one row per label (label = id)."""
    return qf.extract_features(lab)


# --------------------------------------------------------------------------- #
# B4 — neighbour / adjacency
# --------------------------------------------------------------------------- #

def test_neighbor_columns_present_and_finite():
    lab = _two_touching_plus_isolated()
    df = _df_for(lab)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    for c in _ADJ_COLS:
        assert c in df.columns, c
    assert df.shape[0] == 3
    # all finite
    for c in _ADJ_COLS:
        assert np.all(np.isfinite(df[c].values)), c


def test_neighbor_count_correct_for_known_adjacency():
    lab = _two_touching_plus_isolated()
    df = _df_for(lab)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    by_label = df.set_index("label")
    # cells 1 and 2 touch -> each has exactly 1 neighbour
    assert by_label.loc[1, "neighbor_count"] == 1.0
    assert by_label.loc[2, "neighbor_count"] == 1.0
    # cell 3 is isolated -> 0 neighbours
    assert by_label.loc[3, "neighbor_count"] == 0.0


def test_isolated_cell_nearest_neighbor_distance_is_max():
    lab = _two_touching_plus_isolated()
    df = _df_for(lab)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    by_label = df.set_index("label")
    # the isolated cell never finds a neighbour -> nn_dist == max_radius + 1
    assert by_label.loc[3, "nearest_neighbor_distance"] == 6.0
    # the touching pair find each other at radius 1 (shared face)
    assert by_label.loc[1, "nearest_neighbor_distance"] == 1.0
    assert by_label.loc[2, "nearest_neighbor_distance"] == 1.0


def test_contact_and_background_fractions_bounded():
    lab = _two_touching_plus_isolated()
    df = _df_for(lab)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    for c in ("max_contact_fraction", "background_fraction"):
        v = df[c].values
        assert np.all(v >= 0.0) and np.all(v <= 1.0), (c, v)
    # the isolated cell sees only background -> background_fraction == 1, contact 0
    by_label = df.set_index("label")
    assert by_label.loc[3, "background_fraction"] == pytest.approx(1.0)
    assert by_label.loc[3, "max_contact_fraction"] == pytest.approx(0.0)


def test_adjacency_list_attr_dedup_and_correct():
    lab = _two_touching_plus_isolated()
    df = _df_for(lab)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    adj = df.attrs.get("_adjacency_list")
    assert isinstance(adj, list)
    # exactly one undirected edge (1,2); the isolated cell 3 has none.
    edges = {(r["label_a"], r["label_b"]) for r in adj}
    assert edges == {(1, 2)}
    for r in adj:
        assert r["label_a"] < r["label_b"]           # de-duplicated direction
        assert r["contact_voxels"] > 0


def test_neighbor_features_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_neighbor_features(np.zeros((4, 4, 4), np.int32), df, max_radius=5)
    for c in _ADJ_COLS:
        assert c in df.columns
    assert len(df) == 0
    assert df.attrs.get("_adjacency_list") == [] or df.attrs.get("_adjacency_list") is None


# --------------------------------------------------------------------------- #
# B5 — z-stratified neighbour topology
# --------------------------------------------------------------------------- #

def test_z_stratified_counts_basal_vs_apical():
    lab = _tall_basal_neighbors()
    df = _df_for(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    r = df.set_index("label").loc[1]
    # cell 1 has 2 neighbours, both touching only at its BASE (low Z).
    assert r["neighbor_count"] == 2.0
    assert r["basal_neighbor_count"] == 2.0
    # apical (top third) should see fewer / none of those basal-only neighbours.
    assert r["apical_neighbor_count"] <= r["basal_neighbor_count"]
    assert r["apical_neighbor_count"] == 0.0


def test_z_topology_ratios_bounded():
    lab = _tall_basal_neighbors()
    df = _df_for(lab)
    qf.compute_neighbor_features(lab, df, max_radius=3)
    for c in ("neighbor_z_turnover", "neighbor_persistence_ratio"):
        v = df[c].values
        assert np.all(v >= 0.0) and np.all(v <= 1.0), (c, v)
    r = df.set_index("label").loc[1]
    # neighbours present basally but not apically -> turnover == 1, persistence 0.
    assert r["neighbor_z_turnover"] == pytest.approx(1.0)
    assert r["neighbor_persistence_ratio"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# B6 — named composites
# --------------------------------------------------------------------------- #

def test_named_composites_columns_present():
    lab = _two_touching_plus_isolated()
    df = _df_for(lab)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    qf.compute_named_composites(df)
    for c in _B6_COLS:
        assert c in df.columns, c
    # merge_signal / fragment_signal are >=0 one-sided scores
    assert np.all(df["merge_signal"].values >= 0.0)
    assert np.all(df["fragment_signal"].values >= 0.0)


def test_named_composites_degrade_without_prob():
    """No prob_* columns -> boundary_confidence is all-NaN, no crash; the other
    composites still compute from the available geometry/adjacency columns."""
    lab = _two_touching_plus_isolated()
    df = _df_for(lab)
    qf.compute_neighbor_features(lab, df, max_radius=5)
    assert not any(c.startswith("prob_") for c in df.columns)
    qf.compute_named_composites(df)
    assert df["boundary_confidence"].isna().all()
    # merge/fragment/shape are produced (finite or zero, never crash)
    for c in ("merge_signal", "fragment_signal", "shape_abnormality"):
        assert c in df.columns


def test_named_composites_empty_df_schema():
    df = pd.DataFrame(columns=["label"])
    qf.compute_named_composites(df)
    for c in _B6_COLS:
        assert c in df.columns
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# manifest + families registry
# --------------------------------------------------------------------------- #

def test_manifest_tags_static_m3_features():
    man = fm.get_manifest()
    for c in (*_ADJ_COLS, *_B6_COLS):
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["role"] in ("biology", "qc", "covariate")
        assert tags["frame"] in ("image", "tissue", "intrinsic")
        assert isinstance(tags["bounded"], bool)


def test_manifest_m3_bounded_flags():
    man = fm.get_manifest()
    for c in ("max_contact_fraction", "background_fraction",
              "neighbor_z_turnover", "neighbor_persistence_ratio"):
        assert man[c]["bounded"] is True, c
    for c in ("neighbor_count", "nearest_neighbor_distance",
              "merge_signal", "fragment_signal"):
        assert man[c]["bounded"] is False, c


def test_feature_stage_families_registry_includes_m3():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["adjacency"]) == set(_B4_COLS)
    assert set(_FEATURE_FAMILIES["adjacency_topology"]) == set(_B5_COLS)
    assert set(_FEATURE_FAMILIES["named_composites"]) == set(_B6_COLS)


# --------------------------------------------------------------------------- #
# opt-in global pass wired into extract_label_features
# --------------------------------------------------------------------------- #

def _toy_labels():
    """Cells 1 & 2 touch at x==8; cell 3 is isolated (gap >> max_radius=5)."""
    vol = np.zeros((8, 16, 36), np.int32)
    vol[1:6, 2:8, 2:8] = 1
    vol[1:6, 2:8, 8:14] = 2          # touches cell 1 at x==8
    vol[1:6, 2:8, 28:34] = 3         # isolated (gap from cell 2 = 14 voxels > 5)
    return vol


def test_extract_label_features_adjacency_off_byte_identical(tmp_path):
    """``extract_label_features(adjacency=False)`` is byte-identical to a default
    call WITHOUT the new params — the M3 additive-optional contract."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df_baseline = extract_label_features(str(p), n_workers=1)                 # no M3 args
    df_default = extract_label_features(                                      # M3 args explicitly off
        str(p), n_workers=1, adjacency=False)

    assert list(df_baseline.columns) == list(df_default.columns)
    pd.testing.assert_frame_equal(df_baseline, df_default)
    # NO adjacency / composite columns leaked into the labels-only output.
    for c in (*_ADJ_COLS, *_B6_COLS):
        assert c not in df_baseline.columns, c


def test_extract_label_features_adjacency_on_adds_columns(tmp_path):
    """``adjacency=True`` runs the global pass: the 21 neighbour/topology + 4
    composite columns appear, the edge list is attached, and the per-cell
    morphology columns from the labels-only run are UNCHANGED."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df_off = extract_label_features(str(p), n_workers=1)
    df_on = extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)

    # same rows / ids
    assert list(df_on["label"].values) == list(df_off["label"].values)
    # the adjacency + composite columns are present on the ON run only
    for c in (*_ADJ_COLS, *_B6_COLS):
        assert c in df_on.columns, c
        assert c not in df_off.columns, c
    # the pre-existing morphology columns are byte-identical (the global pass
    # only APPENDS columns; it must not perturb the per-cell VALUES). The one
    # allowed difference is the `label` column dtype: the adjacency pass casts
    # it float64 -> int64 (compute_neighbor_features indexes find_objects with
    # it); the VALUES are identical, so compare with check_dtype=False.
    shared = [c for c in df_off.columns if c in df_on.columns]
    pd.testing.assert_frame_equal(
        df_on[shared].reset_index(drop=True),
        df_off[shared].reset_index(drop=True),
        check_dtype=False,
    )
    # neighbour count is correct on the assembled table: 1&2 touch, 3 isolated.
    by_label = df_on.set_index("label")
    assert by_label.loc[1, "neighbor_count"] == 1.0
    assert by_label.loc[2, "neighbor_count"] == 1.0
    assert by_label.loc[3, "neighbor_count"] == 0.0
    # edge list carried through.
    edges = {(r["label_a"], r["label_b"]) for r in df_on.attrs.get("_adjacency_list", [])}
    assert edges == {(1, 2)}


def test_extract_label_features_adjacency_on_from_array():
    """``adjacency=True`` works when ``labels`` is an IN-MEMORY array (temp-staged
    to a .npy) — the full-volume load + unlink must not deadlock on Windows."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    df = extract_label_features(vol, n_workers=1, adjacency=True, max_radius=5)
    for c in (*_ADJ_COLS, *_B6_COLS):
        assert c in df.columns, c
    by_label = df.set_index("label")
    assert by_label.loc[1, "neighbor_count"] == 1.0
    assert by_label.loc[3, "neighbor_count"] == 0.0

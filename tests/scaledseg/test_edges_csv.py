"""Tests for :func:`epicycle.core.qc_features.write_edges_csv` (S1a).

The contact graph is built in-memory as ``df.attrs["_adjacency_list"]`` by
:func:`compute_neighbor_features` during the ``adjacency=True`` pass, and the
build-site docstring warns it is "NOT persisted in the CSV — must be saved
separately". ``write_edges_csv`` is that separate persistence step — the
``edges.csv`` artifact (``label_a, label_b, contact_voxels``) the scaling plan
 calls for.

These are pure (no tile readers / GPU): most exercise the writer directly with a
synthetic ``_adjacency_list``; one end-to-end test runs
``extract_label_features(adjacency=True)`` on a tiny touching-cells volume and
confirms the persisted edges match the in-memory graph (the touching pair, the
isolated cell absent).
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf


_EXPECTED_COLS = ["label_a", "label_b", "contact_voxels"]


def _df_with_edges(rows):
    """A per-cell df carrying a synthetic ``_adjacency_list`` attr."""
    df = pd.DataFrame({"label": [1, 2, 3]})
    df.attrs["_adjacency_list"] = rows
    return df


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #
def test_writes_csv_when_adjacency_present(tmp_path):
    rows = [
        {"label_a": 1, "label_b": 2, "contact_voxels": 12},
        {"label_a": 2, "label_b": 3, "contact_voxels": 7},
    ]
    df = _df_with_edges(rows)
    out = tmp_path / "edges.csv"

    wrote = qf.write_edges_csv(df, out)

    assert wrote is True
    assert out.exists()
    back = pd.read_csv(out)
    assert list(back.columns) == _EXPECTED_COLS
    assert len(back) == 2


def test_column_order_is_exact(tmp_path):
    """Even when the dict keys arrive in a different order, the CSV columns are
    a fixed (label_a, label_b, contact_voxels) — a stable schema for the napari
    graph reader + the eventual mirror-port."""
    rows = [{"contact_voxels": 9, "label_b": 5, "label_a": 4}]
    out = tmp_path / "edges.csv"

    qf.write_edges_csv(_df_with_edges(rows), out)

    back = pd.read_csv(out)
    assert list(back.columns) == _EXPECTED_COLS
    assert int(back.loc[0, "label_a"]) == 4
    assert int(back.loc[0, "label_b"]) == 5
    assert int(back.loc[0, "contact_voxels"]) == 9


def test_round_trip_values_preserved(tmp_path):
    rows = [
        {"label_a": 1, "label_b": 4, "contact_voxels": 100},
        {"label_a": 2, "label_b": 9, "contact_voxels": 3},
        {"label_a": 7, "label_b": 8, "contact_voxels": 55},
    ]
    out = tmp_path / "edges.csv"
    qf.write_edges_csv(_df_with_edges(rows), out)

    back = pd.read_csv(out).sort_values(["label_a", "label_b"]).reset_index(drop=True)
    expect = pd.DataFrame(rows, columns=_EXPECTED_COLS)
    expect = expect.sort_values(["label_a", "label_b"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(back, expect, check_dtype=False)


def test_accepts_str_and_path(tmp_path):
    rows = [{"label_a": 1, "label_b": 2, "contact_voxels": 1}]
    p_path = tmp_path / "a" / "edges.csv"
    p_str = str(tmp_path / "b" / "edges.csv")

    assert qf.write_edges_csv(_df_with_edges(rows), p_path) is True
    assert qf.write_edges_csv(_df_with_edges(rows), p_str) is True
    assert p_path.exists()
    assert (tmp_path / "b" / "edges.csv").exists()


def test_creates_parent_dirs(tmp_path):
    rows = [{"label_a": 1, "label_b": 2, "contact_voxels": 1}]
    out = tmp_path / "deep" / "nested" / "run" / "edges.csv"

    assert qf.write_edges_csv(_df_with_edges(rows), out) is True
    assert out.exists()


def test_overwrites_existing(tmp_path):
    out = tmp_path / "edges.csv"
    out.write_text("stale,garbage\n1,2\n")

    rows = [{"label_a": 1, "label_b": 2, "contact_voxels": 42}]
    assert qf.write_edges_csv(_df_with_edges(rows), out) is True

    back = pd.read_csv(out)
    assert list(back.columns) == _EXPECTED_COLS
    assert int(back.loc[0, "contact_voxels"]) == 42


def test_dedup_invariant_label_a_lt_b_preserved(tmp_path):
    """The writer persists the upstream dedup invariant (label_a < label_b) as-is
    — it does not reorder or re-canonicalise (that's compute_neighbor_features'
    job). Every persisted row keeps a < b."""
    rows = [
        {"label_a": 1, "label_b": 2, "contact_voxels": 5},
        {"label_a": 3, "label_b": 10, "contact_voxels": 8},
    ]
    out = tmp_path / "edges.csv"
    qf.write_edges_csv(_df_with_edges(rows), out)

    back = pd.read_csv(out)
    assert (back["label_a"] < back["label_b"]).all()


# --------------------------------------------------------------------------- #
# No-op paths                                                                  #
# --------------------------------------------------------------------------- #
def test_no_op_when_adjacency_list_missing(tmp_path):
    df = pd.DataFrame({"label": [1, 2]})  # no _adjacency_list attr at all
    out = tmp_path / "edges.csv"

    assert qf.write_edges_csv(df, out) is False
    assert not out.exists()


def test_no_op_when_adjacency_list_empty(tmp_path):
    df = _df_with_edges([])  # adjacency ran but found no touching cells
    out = tmp_path / "edges.csv"

    assert qf.write_edges_csv(df, out) is False
    assert not out.exists()


def test_no_op_when_adjacency_list_none(tmp_path):
    df = _df_with_edges(None)
    out = tmp_path / "edges.csv"

    assert qf.write_edges_csv(df, out) is False
    assert not out.exists()


# --------------------------------------------------------------------------- #
# End-to-end: the persisted edges match the in-memory adjacency graph          #
# --------------------------------------------------------------------------- #
def _toy_labels():
    """Cells 1 & 2 touch at x==8; cell 3 is isolated (gap >> max_radius=5).

    Same shape as ``tests/test_feature_m3.py::_toy_labels`` so the e2e edge set
    is the documented {(1, 2)}.
    """
    vol = np.zeros((8, 16, 36), np.int32)
    vol[1:6, 2:8, 2:8] = 1
    vol[1:6, 2:8, 8:14] = 2          # touches cell 1 at x==8
    vol[1:6, 2:8, 28:34] = 3         # isolated (gap from cell 2 = 14 voxels > 5)
    return vol


def test_end_to_end_edges_match_in_memory_graph(tmp_path):
    """``extract_label_features(adjacency=True)`` builds the in-memory graph;
    ``write_edges_csv`` persists it. The persisted edges must equal the in-memory
    ``_adjacency_list`` exactly — the touching pair (1, 2), no self-edges, the
    isolated cell 3 absent."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df = extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)
    out = tmp_path / "edges.csv"
    wrote = qf.write_edges_csv(df, out)

    assert wrote is True
    back = pd.read_csv(out)
    persisted = {(int(r.label_a), int(r.label_b)) for r in back.itertuples()}
    in_memory = {
        (int(e["label_a"]), int(e["label_b"]))
        for e in df.attrs.get("_adjacency_list", [])
    }
    assert persisted == in_memory == {(1, 2)}
    # no self-edges, dedup invariant holds on disk.
    assert (back["label_a"] != back["label_b"]).all()
    assert (back["label_a"] < back["label_b"]).all()
    # contact_voxels is a positive count for a real contact.
    assert (back["contact_voxels"] > 0).all()


def test_end_to_end_no_op_when_adjacency_off(tmp_path):
    """With ``adjacency=False`` there is no ``_adjacency_list`` -> the writer is a
    no-op (nothing to persist), matching the labels-only byte-identical contract."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    df = extract_label_features(str(p), n_workers=1)  # adjacency off (default)
    out = tmp_path / "edges.csv"

    assert qf.write_edges_csv(df, out) is False
    assert not out.exists()

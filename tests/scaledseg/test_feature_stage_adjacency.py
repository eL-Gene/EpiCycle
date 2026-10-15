"""FeatureStage ``adjacency=True`` opt-in (S1 wiring): the global contact-graph
family runs off a memmap of the global labels and the contact graph is persisted
to ``edges.csv`` next to ``cell_features.csv``.

Asserts: edges.csv schema + content; the graph columns land in the table;
``adjacency=False`` is unchanged (no edges.csv, no graph columns); and — the
parity gate — the graph columns are byte-identical to
``reconcile.extract_label_features(adjacency=True)`` on the same volume (both go
through the SHARED ``run_adjacency_graph_passes``)."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core.feature_stage import FeatureStage, _FEATURE_FAMILIES

_PROV_COLS = [
    "dense_id", "seam_status", "n_tiles", "min_match_iou", "voxel_count",
    "z_min", "z_max", "y_min", "y_max", "x_min", "x_max",
    "touches_border", "tiles", "constituents",
]


def _touch_volume():
    """Cells 1 & 2 touch at x==8; cell 3 isolated -> graph edge set {(1, 2)}."""
    vol = np.zeros((8, 16, 36), np.int32)
    vol[1:6, 2:8, 2:8] = 1
    vol[1:6, 2:8, 8:14] = 2
    vol[1:6, 2:8, 28:34] = 3
    return vol


def _write_prov(path, vol):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROV_COLS)
        w.writeheader()
        for lbl in (1, 2, 3):
            zs, ys, xs = np.nonzero(vol == lbl)
            w.writerow({
                "dense_id": lbl, "seam_status": "clean", "n_tiles": 1,
                "min_match_iou": "", "voxel_count": int((vol == lbl).sum()),
                "z_min": int(zs.min()), "z_max": int(zs.max()),
                "y_min": int(ys.min()), "y_max": int(ys.max()),
                "x_min": int(xs.min()), "x_max": int(xs.max()),
                "touches_border": 0, "tiles": "t0", "constituents": "",
            })


def _fixture(tmp: Path):
    stitch = tmp / "stage4_full_stitch"
    stitch.mkdir(parents=True, exist_ok=True)
    vol = _touch_volume()
    np.save(str(stitch / "global_labels_clean.npy"), vol)
    _write_prov(stitch / "label_provenance_clean.csv", vol)
    return stitch, vol


def _td():
    return [Path("x/tiles/t0")]


def test_adjacency_writes_edges_and_graph_columns(tmp_path):
    stitch, _ = _fixture(tmp_path)
    res = FeatureStage(
        input_dir=str(stitch), n_workers=1, adjacency=True).run(_td(), n_gpus=0)
    assert res.n_failed == 0

    edges = stitch / "edges.csv"
    assert edges.exists()
    e = pd.read_csv(edges)
    assert list(e.columns) == ["label_a", "label_b", "contact_voxels"]
    assert {(int(r.label_a), int(r.label_b)) for r in e.itertuples()} == {(1, 2)}
    assert (e["contact_voxels"] > 0).all()

    cf = pd.read_csv(stitch / "cell_features.csv").set_index("label")
    assert "neighbor_count" in cf.columns
    assert cf.loc[1, "neighbor_count"] == 1.0
    assert cf.loc[2, "neighbor_count"] == 1.0
    assert cf.loc[3, "neighbor_count"] == 0.0


def test_adjacency_manifest_records_edges(tmp_path):
    import json
    stitch, _ = _fixture(tmp_path)
    FeatureStage(input_dir=str(stitch), n_workers=1, adjacency=True).run(_td(), n_gpus=0)
    man = json.loads((stitch / "feature_manifest.json").read_text())
    assert man["status"] == "complete"
    assert man["adjacency"] is True
    assert man["n_edges"] == 1
    assert man["edges_csv"] and man["edges_csv"].endswith("edges.csv")


def test_adjacency_off_no_edges_no_graph_columns(tmp_path):
    stitch, _ = _fixture(tmp_path)
    FeatureStage(input_dir=str(stitch), n_workers=1).run(_td(), n_gpus=0)
    assert not (stitch / "edges.csv").exists()
    cf = pd.read_csv(stitch / "cell_features.csv")
    assert "neighbor_count" not in cf.columns
    man = __import__("json").loads((stitch / "feature_manifest.json").read_text())
    assert man["adjacency"] is False
    assert man["n_edges"] == 0


def test_adjacency_parity_with_extract_label_features(tmp_path):
    """The graph columns FeatureStage emits must be byte-identical to
    reconcile.extract_label_features(adjacency=True) on the same volume — both run
    the SHARED run_adjacency_graph_passes, so the contact graph is the same."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    stitch, vol = _fixture(tmp_path)
    FeatureStage(input_dir=str(stitch), n_workers=1, adjacency=True).run(_td(), n_gpus=0)
    cf = (pd.read_csv(stitch / "cell_features.csv")
          .sort_values("label").reset_index(drop=True))

    p = tmp_path / "lab.npy"
    np.save(p, vol)
    ref = (extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)
           .sort_values("label").reset_index(drop=True))

    # Every column in the adjacency family must be present in both + match exactly.
    graph_cols = [c for c in _FEATURE_FAMILIES["adjacency"]
                  if c in cf.columns and c in ref.columns]
    assert "neighbor_count" in graph_cols      # sanity: the family really landed
    for c in graph_cols:
        np.testing.assert_allclose(
            cf[c].to_numpy(dtype=float), ref[c].to_numpy(dtype=float),
            equal_nan=True, err_msg=f"graph column {c} differs FeatureStage vs reconcile")


def test_min_voxels_drops_dangling_edges(tmp_path):
    """With ``min_voxels`` above a tiny cell's voxel count, that cell is dropped from
    cell_features.csv but is still a graph NEIGHBOUR (the graph sees the whole
    memmap volume). edges.csv must NOT reference the dropped node (no dangling
    endpoint), while the surviving cell's ``neighbor_count`` keeps the true physical
    count. (M1 — audit finding.)"""
    import json

    stitch = tmp_path / "stage4_full_stitch"
    stitch.mkdir(parents=True, exist_ok=True)
    vol = np.zeros((8, 16, 20), np.int32)
    vol[1:7, 2:8, 2:8] = 1                # big cell (216 voxels)
    vol[1:3, 2:4, 8:10] = 2               # tiny cell (8 voxels), touches cell 1 at x==8
    np.save(str(stitch / "global_labels_clean.npy"), vol)
    with open(stitch / "label_provenance_clean.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROV_COLS)
        w.writeheader()
        for lbl in (1, 2):
            zs, ys, xs = np.nonzero(vol == lbl)
            w.writerow({
                "dense_id": lbl, "seam_status": "clean", "n_tiles": 1,
                "min_match_iou": "", "voxel_count": int((vol == lbl).sum()),
                "z_min": int(zs.min()), "z_max": int(zs.max()),
                "y_min": int(ys.min()), "y_max": int(ys.max()),
                "x_min": int(xs.min()), "x_max": int(xs.max()),
                "touches_border": 0, "tiles": "t0", "constituents": "",
            })

    # floor=20 > the tiny cell's 8 voxels -> cell 2 dropped from the table.
    FeatureStage(input_dir=str(stitch), n_workers=1, adjacency=True,
                 min_voxels=20).run(_td(), n_gpus=0)

    cf = pd.read_csv(stitch / "cell_features.csv")
    assert set(cf["label"]) == {1}                       # cell 2 dropped from table
    assert cf.set_index("label").loc[1, "neighbor_count"] == 1.0   # physical truth kept

    # The only edge (1, 2) is dangling (node 2 absent) -> filtered out -> no file.
    edges = stitch / "edges.csv"
    if edges.exists():
        e = pd.read_csv(edges)
        assert 2 not in (set(e["label_a"]) | set(e["label_b"]))
    man = json.loads((stitch / "feature_manifest.json").read_text())
    assert man["n_edges"] == 0


def test_adjacency_with_spacing_writes_physical_contact(tmp_path):
    """With voxel_size_um set, the contact-geometry physical columns are produced
    and the graph still matches the reconcile path (spacing threaded identically)."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    stitch, vol = _fixture(tmp_path)
    sp = (2.0, 0.5, 0.5)
    FeatureStage(input_dir=str(stitch), n_workers=1, adjacency=True,
                 voxel_size_um=sp).run(_td(), n_gpus=0)
    cf = (pd.read_csv(stitch / "cell_features.csv")
          .sort_values("label").reset_index(drop=True))

    p = tmp_path / "lab.npy"
    np.save(p, vol)
    ref = (extract_label_features(str(p), n_workers=1, adjacency=True,
                                  max_radius=5, spacing=sp)
           .sort_values("label").reset_index(drop=True))
    graph_cols = [c for c in _FEATURE_FAMILIES["adjacency"]
                  if c in cf.columns and c in ref.columns]
    for c in graph_cols:
        np.testing.assert_allclose(
            cf[c].to_numpy(dtype=float), ref[c].to_numpy(dtype=float),
            equal_nan=True, err_msg=f"graph column {c} differs (spacing)")

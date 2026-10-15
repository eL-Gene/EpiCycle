"""FeatureStage integration tests on a synthetic global volume + provenance.

No GPU / Cellpose / u-Segment3D. Builds a tiny ``global_labels_clean.npy`` +
``label_provenance_clean.csv`` (a cube, a rod, a slab, a border-touching corner
cell, and a zero-voxel ghost row), then exercises dry-run / extract / idempotency /
ghost-skip / border-from-provenance / raw-stitch fallback / worker isolation."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core.feature_stage import FeatureStage
import packages.epicycle_scaledseg.core.feature_stage as fs

_PROV_COLS = [
    "dense_id", "seam_status", "n_tiles", "min_match_iou", "voxel_count",
    "z_min", "z_max", "y_min", "y_max", "x_min", "x_max",
    "touches_border", "tiles", "constituents",
]


def _build_volume():
    vol = np.zeros((24, 40, 40), dtype=np.int32)
    vol[4:12, 5:13, 5:13] = 1            # cube
    vol[4:9, 5:8, 20:32] = 2             # rod (seam cell)
    vol[14:20, 22:30, 22:28] = 3         # slab
    vol[0:6, 0:6, 0:6] = 4               # corner cell (touches Z=0/Y=0/X=0)
    return vol


def _bbox(vol, lbl):
    zs, ys, xs = np.nonzero(vol == lbl)
    return (int(zs.min()), int(zs.max()), int(ys.min()), int(ys.max()),
            int(xs.min()), int(xs.max()), int((vol == lbl).sum()))


def _write_prov(path, vol, *, clean=True, ghost=True):
    seam = {1: "clean", 2: "seam", 3: "clean", 4: "corner"}
    miou = {1: "", 2: "0.42", 3: "", 4: ""}
    border = {1: 0, 2: 0, 3: 0, 4: 1}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROV_COLS)
        w.writeheader()
        for lbl in (1, 2, 3, 4):
            z0, z1, y0, y1, x0, x1, vc = _bbox(vol, lbl)
            w.writerow({
                "dense_id": lbl, "seam_status": seam[lbl], "n_tiles": 1,
                "min_match_iou": miou[lbl], "voxel_count": vc,
                "z_min": z0, "z_max": z1, "y_min": y0, "y_max": y1,
                "x_min": x0, "x_max": x1, "touches_border": border[lbl],
                "tiles": "t0", "constituents": "",
            })
        if ghost:
            w.writerow({c: 0 for c in _PROV_COLS} | {
                "dense_id": 5, "seam_status": "clean", "voxel_count": 0,
                "z_min": -1, "z_max": -1, "y_min": -1, "y_max": -1,
                "x_min": -1, "x_max": -1,
            })


def _build_fixture(tmp: Path, clean=True, ghost=True):
    stitch = tmp / "stage4_full_stitch"
    stitch.mkdir(parents=True, exist_ok=True)
    vol = _build_volume()
    npy = "global_labels_clean.npy" if clean else "global_labels.npy"
    prov = "label_provenance_clean.csv" if clean else "label_provenance.csv"
    np.save(str(stitch / npy), vol)
    _write_prov(stitch / prov, vol, clean=clean, ghost=ghost)
    return stitch, vol


def _stage(stitch, **kw):
    defaults = dict(input_dir=str(stitch), n_workers=1)
    defaults.update(kw)
    return FeatureStage(**defaults)


def _td():
    # A dummy tile dir whose .parent/stage4_full_stitch is unused (input_dir is set).
    return [Path("x/tiles/t0")]


# --------------------------------------------------------------------------- #
# core integration
# --------------------------------------------------------------------------- #

def test_dry_run_writes_no_table(tmp_path):
    stitch, _ = _build_fixture(tmp_path)
    res = _stage(stitch, dry_run=True).run(_td(), n_gpus=0)
    assert res.n_failed == 0
    assert not (stitch / "cell_features.parquet").exists()
    assert not (stitch / "cell_features.csv").exists()
    man = json.loads((stitch / "feature_manifest.json").read_text())
    assert man["status"] == "dry_run"
    assert man["n_cells_requested"] == 4          # ghost excluded


def test_extract_writes_table_and_manifest(tmp_path):
    stitch, vol = _build_fixture(tmp_path)
    res = _stage(stitch, voxel_size_um=(2.0, 0.5, 0.5)).run(_td(), n_gpus=0)
    assert res.n_failed == 0
    assert (stitch / "cell_features.parquet").exists()
    assert (stitch / "cell_features.csv").exists()
    df = pd.read_parquet(stitch / "cell_features.parquet")
    assert sorted(df["label"].tolist()) == [1, 2, 3, 4]   # ghost #5 skipped
    assert df["label"].dtype.kind in "iu"                 # int join key
    man = json.loads((stitch / "feature_manifest.json").read_text())
    assert man["status"] == "complete"
    assert man["n_cells_extracted"] == 4 and man["n_missing"] == 0 and man["n_failed"] == 0
    assert man["used_clean_inputs"] is True
    assert man["voxel_volume_um3"] == pytest.approx(0.5)


def test_area_is_physical_volume(tmp_path):
    stitch, vol = _build_fixture(tmp_path)
    _stage(stitch, voxel_size_um=(2.0, 0.5, 0.5)).run(_td(), n_gpus=0)
    df = pd.read_parquet(stitch / "cell_features.parquet").set_index("label")
    # area (physical) == voxel_count * (2.0*0.5*0.5)
    for lbl in (1, 2, 3, 4):
        vc = int((vol == lbl).sum())
        assert df.loc[lbl, "area"] == pytest.approx(vc * 0.5)
        assert df.loc[lbl, "voxel_count"] == vc


def test_border_flag_from_provenance(tmp_path):
    stitch, _ = _build_fixture(tmp_path)
    _stage(stitch).run(_td(), n_gpus=0)
    df = pd.read_parquet(stitch / "cell_features.parquet").set_index("label")
    # Only the corner cell (#4) is flagged — and it comes from provenance, not the
    # per-cell crop (every cell touches its own crop face).
    assert df.loc[4, "is_border_cell"] == 1
    assert df.loc[1, "is_border_cell"] == 0
    assert df.loc[2, "is_border_cell"] == 0


def test_seam_fields_carried(tmp_path):
    stitch, _ = _build_fixture(tmp_path)
    _stage(stitch).run(_td(), n_gpus=0)
    df = pd.read_parquet(stitch / "cell_features.parquet").set_index("label")
    assert df.loc[2, "seam_status"] == "seam"
    assert df.loc[2, "min_match_iou"] == pytest.approx(0.42)
    assert np.isnan(df.loc[1, "min_match_iou"])     # blank -> NaN


def test_min_voxels_floor(tmp_path):
    stitch, vol = _build_fixture(tmp_path)
    # rod (#2) is the smallest real cell; floor above it drops it.
    rod_vc = int((vol == 2).sum())
    _stage(stitch, min_voxels=rod_vc + 1).run(_td(), n_gpus=0)
    df = pd.read_parquet(stitch / "cell_features.parquet")
    assert 2 not in df["label"].tolist()


def test_crowding_columns_present_and_finite(tmp_path):
    stitch, _ = _build_fixture(tmp_path)
    _stage(stitch).run(_td(), n_gpus=0)
    df = pd.read_parquet(stitch / "cell_features.parquet")
    for c in ("nn_centroid_distance", "knn_mean_distance",
              "neighbor_count_in_radius", "local_density"):
        assert c in df.columns
    assert df["nn_centroid_distance"].notna().all()   # 4 cells -> all have a neighbour


def test_include_surface_false_omits_family(tmp_path):
    stitch, _ = _build_fixture(tmp_path)
    _stage(stitch, include_surface=False).run(_td(), n_gpus=0)
    df = pd.read_parquet(stitch / "cell_features.parquet")
    assert "surface_area" not in df.columns
    assert "discrete_compactness" in df.columns       # the rest still run


def test_idempotent_skip(tmp_path):
    stitch, _ = _build_fixture(tmp_path)
    _stage(stitch).run(_td(), n_gpus=0)
    mtime = (stitch / "cell_features.parquet").stat().st_mtime_ns
    res = _stage(stitch).run(_td(), n_gpus=0)
    assert res.n_failed == 0
    assert (stitch / "cell_features.parquet").stat().st_mtime_ns == mtime


def test_incomplete_manifest_forces_rerun(tmp_path):
    stitch, _ = _build_fixture(tmp_path)
    (stitch / "cell_features.parquet").write_bytes(b"partial")
    (stitch / "feature_manifest.json").write_text(json.dumps({"status": "incomplete"}))
    res = _stage(stitch).run(_td(), n_gpus=0)
    assert res.n_failed == 0
    df = pd.read_parquet(stitch / "cell_features.parquet")   # really re-ran (valid)
    assert sorted(df["label"].tolist()) == [1, 2, 3, 4]


def test_missing_inputs_fail_cleanly(tmp_path):
    stitch = tmp_path / "stage4_full_stitch"
    stitch.mkdir(parents=True)
    res = _stage(stitch).run(_td(), n_gpus=0)
    assert res.n_failed > 0 and res.n_success == 0


def test_validate_inputs(tmp_path):
    stitch, _ = _build_fixture(tmp_path)
    assert _stage(stitch).validate_inputs(_td()) is True
    empty = tmp_path / "empty"
    (empty / "stage4_full_stitch").mkdir(parents=True)
    assert FeatureStage(input_dir=str(empty / "stage4_full_stitch")).validate_inputs(_td()) is False


# --------------------------------------------------------------------------- #
# input resolution: clean preferred, raw fallback
# --------------------------------------------------------------------------- #

def test_raw_stitch_fallback_when_no_clean(tmp_path):
    stitch, _ = _build_fixture(tmp_path, clean=False)   # only raw global_labels.*
    res = _stage(stitch).run(_td(), n_gpus=0)
    assert res.n_failed == 0
    man = json.loads((stitch / "feature_manifest.json").read_text())
    assert man["used_clean_inputs"] is False
    assert man["n_cells_extracted"] == 4


def test_clean_preferred_over_raw(tmp_path):
    # Build BOTH; clean has only labels {1,2}, raw has {1,2,3,4}.
    stitch = tmp_path / "stage4_full_stitch"
    stitch.mkdir(parents=True)
    raw = _build_volume()
    np.save(str(stitch / "global_labels.npy"), raw)
    _write_prov(stitch / "label_provenance.csv", raw, clean=False, ghost=False)
    clean = np.where(np.isin(raw, [1, 2]), raw, 0).astype(np.int32)
    np.save(str(stitch / "global_labels_clean.npy"), clean)
    # Clean provenance carries the RAW geometry (valid bboxes for all 4) while the
    # clean VOLUME only HAS 1,2 -> the stage requests 4, extracts 1,2, and 3,4 are
    # MISSING (the case under test).
    _write_prov(stitch / "label_provenance_clean.csv", raw, ghost=False)
    _stage(stitch).run(_td(), n_gpus=0)
    man = json.loads((stitch / "feature_manifest.json").read_text())
    assert man["used_clean_inputs"] is True
    df = pd.read_parquet(stitch / "cell_features.parquet")
    assert sorted(df["label"].tolist()) == [1, 2]    # only clean's labels extracted
    assert man["n_missing"] == 2                      # 3,4 absent from clean volume


def test_prefer_clean_false_uses_raw(tmp_path):
    stitch, _ = _build_fixture(tmp_path, clean=True)    # clean present
    # also drop a raw pair so prefer_clean=False has something to read
    raw = _build_volume()
    np.save(str(stitch / "global_labels.npy"), raw)
    _write_prov(stitch / "label_provenance.csv", raw, clean=False, ghost=False)
    _stage(stitch, prefer_clean=False).run(_td(), n_gpus=0)
    man = json.loads((stitch / "feature_manifest.json").read_text())
    assert man["used_clean_inputs"] is False


# --------------------------------------------------------------------------- #
# multi-worker (Pool) path
# --------------------------------------------------------------------------- #

def test_multiworker_path_matches_single(tmp_path):
    stitch, _ = _build_fixture(tmp_path)
    _stage(stitch, n_workers=2, voxel_size_um=(2.0, 0.5, 0.5)).run(_td(), n_gpus=0)
    df = pd.read_parquet(stitch / "cell_features.parquet")
    assert sorted(df["label"].tolist()) == [1, 2, 3, 4]
    # deterministic order (sorted by label) regardless of imap_unordered
    assert df["label"].tolist() == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# worker-level isolation (unit)
# --------------------------------------------------------------------------- #

def test_extract_one_missing_when_cell_absent(tmp_path):
    vol = np.zeros((6, 12, 12), dtype=np.int32)
    vol[2:4, 4:8, 4:8] = 7
    np.save(str(tmp_path / "g.npy"), vol)
    fs._init_feature_worker(str(tmp_path / "g.npy"))
    try:
        # ask for a cell id that isn't in the crop -> MISSING
        res = fs._extract_one((99, (0, 6, 0, 12, 0, 12), None, False, True))
    finally:
        fs._WORKER_SRC = None
    assert res == ("MISSING", 99)


def test_extract_one_error_on_unreadable_crop():
    fs._WORKER_SRC = None    # not initialised -> subscript raises -> caught
    res = fs._extract_one((7, (0, 2, 0, 2, 0, 2), None, False, True))
    assert res[0] == "ERROR" and res[1] == 7


def test_extract_one_global_centroid_offset(tmp_path):
    # A single cell placed away from the origin; its returned centroid must be GLOBAL.
    vol = np.zeros((20, 20, 20), dtype=np.int32)
    vol[8:12, 8:12, 8:12] = 3            # centred at (9.5, 9.5, 9.5)
    np.save(str(tmp_path / "g.npy"), vol)
    fs._init_feature_worker(str(tmp_path / "g.npy"))
    try:
        row = fs._extract_one((3, (8, 12, 8, 12, 8, 12), None, False, True))
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict)
    assert row["centroid-0"] == pytest.approx(9.5)
    assert row["centroid-1"] == pytest.approx(9.5)
    assert row["centroid-2"] == pytest.approx(9.5)
    assert int(row["label"]) == 3


def test_extract_one_global_centroid_offset_with_spacing(tmp_path):
    # Same cell, but with anisotropic spacing — the global centroid must be
    # (voxel centroid + integer origin) * spacing (exercises the spacing branch).
    vol = np.zeros((20, 20, 20), dtype=np.int32)
    vol[8:12, 8:12, 8:12] = 3            # voxel centroid (9.5, 9.5, 9.5)
    np.save(str(tmp_path / "g.npy"), vol)
    fs._init_feature_worker(str(tmp_path / "g.npy"))
    try:
        row = fs._extract_one((3, (8, 12, 8, 12, 8, 12), (2.0, 0.5, 0.5), False, True))
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict)
    assert row["centroid-0"] == pytest.approx(9.5 * 2.0)    # 19.0
    assert row["centroid-1"] == pytest.approx(9.5 * 0.5)    # 4.75
    assert row["centroid-2"] == pytest.approx(9.5 * 0.5)    # 4.75

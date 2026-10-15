"""PostprocessStage integration tests on a synthetic global volume + provenance.

No GPU / Cellpose / u-Segment3D. Builds a tiny ``global_labels.npy`` +
``label_provenance.csv`` with one big tissue blob, an adjacent second blob, a tiny
fragment (size-dropped), and a far detached island (island-dropped), then exercises
dry-run / relabel / smooth / idempotency."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from packages.epicycle_scaledseg.core.postprocess_stage import PostprocessStage

_PROV_COLS = [
    "dense_id", "seam_status", "n_tiles", "min_match_iou", "voxel_count",
    "z_min", "z_max", "y_min", "y_max", "x_min", "x_max",
    "touches_border", "tiles", "constituents",
]


def _build_fixture(tmp: Path):
    """Volume (12,40,40) with labels 1 (big), 4 (big, adjacent), 2 (tiny frag),
    3 (far island, >min_voxels so only the island filter can drop it)."""
    vol = np.zeros((12, 40, 40), dtype=np.int32)
    vol[2:10, 5:21, 5:21] = 1     # big tissue blob               -> KEEP
    vol[2:10, 22:34, 5:21] = 4    # second big blob (adjacent)     -> KEEP
    vol[5:6, 21:23, 10:12] = 2    # tiny fragment (4 vox)          -> size-drop
    vol[0:4, 32:40, 32:40] = 3    # far island (256 vox > 200)     -> island-drop
    np.save(str(tmp / "global_labels.npy"), vol)

    def _bbox(v, lbl):
        zs, ys, xs = np.nonzero(v == lbl)
        return (int(zs.min()), int(zs.max()), int(ys.min()), int(ys.max()),
                int(xs.min()), int(xs.max()), int((v == lbl).sum()))

    seam = {1: "seam", 4: "corner", 2: "clean", 3: "clean"}
    with open(tmp / "label_provenance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROV_COLS)
        w.writeheader()
        for lbl in (1, 2, 3, 4):
            z0, z1, y0, y1, x0, x1, vc = _bbox(vol, lbl)
            w.writerow({
                "dense_id": lbl, "seam_status": seam[lbl], "n_tiles": 2,
                "min_match_iou": "0.30", "voxel_count": vc,
                "z_min": z0, "z_max": z1, "y_min": y0, "y_max": y1,
                "x_min": x0, "x_max": x1, "touches_border": 0,
                "tiles": "tile_0_0;tile_0_1", "constituents": "tile_0_0:5;tile_0_1:9",
            })
    return vol


def _stage(tmp, **kw):
    # dilate=0 keeps the far island a separate component on this tiny grid.
    defaults = dict(
        input_dir=str(tmp), output_dir=str(tmp),
        min_voxels=200, island_dilate=0, island_min_component_frac=0.1,
        n_workers=1,
    )
    defaults.update(kw)
    return PostprocessStage(**defaults)


def test_dry_run_writes_no_volume(tmp_path):
    _build_fixture(tmp_path)
    res = _stage(tmp_path, smooth=False, dry_run=True).run([tmp_path], n_gpus=0)
    assert res.n_failed == 0
    assert not (tmp_path / "global_labels_clean.tif").exists()
    assert not (tmp_path / "global_labels_clean.npy").exists()
    man = json.loads((tmp_path / "postprocess_manifest.json").read_text())
    assert man["status"] == "dry_run" and man["dry_run"] is True
    assert man["n_cells_before"] == 4 and man["n_cells_after"] == 2
    assert man["n_dropped_size"] == 1 and man["n_dropped_island"] == 1


def test_relabel_drops_and_renumbers(tmp_path):
    _build_fixture(tmp_path)
    res = _stage(tmp_path, smooth=False).run([tmp_path], n_gpus=0)
    assert res.n_failed == 0
    out = np.load(str(tmp_path / "global_labels_clean.npy"))
    # labels 2 (tiny) and 3 (island) gone; survivors densely renumbered 1..2.
    assert set(np.unique(out).tolist()) == {0, 1, 2}
    assert (tmp_path / "global_labels_clean.tif").exists()
    # clean provenance has exactly the 2 survivors, carrying seam status.
    rows = list(csv.DictReader(open(tmp_path / "label_provenance_clean.csv", newline="")))
    assert len(rows) == 2
    statuses = {r["seam_status"] for r in rows}
    assert statuses == {"seam", "corner"}


def test_npy_tif_byte_identical(tmp_path):
    _build_fixture(tmp_path)
    _stage(tmp_path, smooth=False).run([tmp_path], n_gpus=0)
    import tifffile
    npy = np.load(str(tmp_path / "global_labels_clean.npy"))
    tif = tifffile.imread(str(tmp_path / "global_labels_clean.tif"))
    assert np.array_equal(npy, tif)


def test_smooth_path_runs_and_keeps_survivors(tmp_path):
    _build_fixture(tmp_path)
    res = _stage(tmp_path, smooth=True, smooth_sigma=1.0).run([tmp_path], n_gpus=0)
    assert res.n_failed == 0
    out = np.load(str(tmp_path / "global_labels_clean.npy"))
    uniq = set(np.unique(out).tolist())
    assert uniq.issubset({0, 1, 2})
    assert 1 in uniq and 2 in uniq           # both survivors present after smoothing
    man = json.loads((tmp_path / "postprocess_manifest.json").read_text())
    assert man["status"] == "complete" and man["smooth"] is True


def test_idempotent_skip(tmp_path):
    _build_fixture(tmp_path)
    _stage(tmp_path, smooth=False).run([tmp_path], n_gpus=0)
    mtime = (tmp_path / "global_labels_clean.tif").stat().st_mtime_ns
    # second run should skip (clean tif exists) and not rewrite.
    res = _stage(tmp_path, smooth=False).run([tmp_path], n_gpus=0)
    assert res.n_failed == 0
    assert (tmp_path / "global_labels_clean.tif").stat().st_mtime_ns == mtime


def test_toggles_all_off_is_passthrough(tmp_path):
    _build_fixture(tmp_path)
    _stage(tmp_path, remove_small=False, remove_islands=False, smooth=False).run(
        [tmp_path], n_gpus=0
    )
    out = np.load(str(tmp_path / "global_labels_clean.npy"))
    # nothing dropped -> all 4 labels survive (renumbered densely 1..4).
    assert set(np.unique(out).tolist()) == {0, 1, 2, 3, 4}


def test_missing_provenance_fails_cleanly(tmp_path):
    np.save(str(tmp_path / "global_labels.npy"), np.zeros((4, 8, 8), dtype=np.int32))
    res = _stage(tmp_path, smooth=False).run([tmp_path], n_gpus=0)
    assert res.n_failed > 0 and res.n_success == 0


# --- audit-driven regression tests -------------------------------------------

def _append_prov_row(tmp: Path, **fields):
    row = {c: fields.get(c, 0) for c in _PROV_COLS}
    row.setdefault("seam_status", "clean")
    with open(tmp / "label_provenance.csv", "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_PROV_COLS).writerow(row)


def test_zero_voxel_ghost_dropped_even_with_filters_off(tmp_path):
    # A voxel_count==0 provenance row must NEVER be renumbered into the clean CSV,
    # even with the size filter off (regression: audit C1b ghost rows).
    _build_fixture(tmp_path)
    _append_prov_row(tmp_path, dense_id=99, voxel_count=0, seam_status="seam")
    _stage(tmp_path, remove_small=False, remove_islands=False, smooth=False).run(
        [tmp_path], n_gpus=0
    )
    rows = list(csv.DictReader(open(tmp_path / "label_provenance_clean.csv", newline="")))
    # the 4 real labels survive; the ghost id 99 is gone.
    assert len(rows) == 4
    out = np.load(str(tmp_path / "global_labels_clean.npy"))
    assert set(np.unique(out).tolist()) == {0, 1, 2, 3, 4}


def test_relabel_out_of_csv_id_maps_to_zero(tmp_path):
    # An id present in the VOLUME but absent from the provenance CSV must map to 0,
    # not raise IndexError (regression: audit H1 CSV/volume drift).
    _build_fixture(tmp_path)
    vol = np.load(str(tmp_path / "global_labels.npy"))
    vol[0, 0, 0] = 99            # background corner, id not in the CSV (max_id=4)
    np.save(str(tmp_path / "global_labels.npy"), vol)
    res = _stage(tmp_path, smooth=False).run([tmp_path], n_gpus=0)
    assert res.n_failed == 0
    out = np.load(str(tmp_path / "global_labels_clean.npy"))
    assert out[0, 0, 0] == 0    # unknown id -> background, no crash


def test_incomplete_manifest_forces_rerun(tmp_path):
    # A clean TIFF left by a crashed run (manifest status != complete) must NOT be
    # trusted by the idempotency skip (regression: audit C2 truncated-tif skip).
    _build_fixture(tmp_path)
    (tmp_path / "global_labels_clean.tif").write_bytes(b"partial-garbage")
    (tmp_path / "postprocess_manifest.json").write_text(json.dumps({"status": "incomplete"}))
    res = _stage(tmp_path, smooth=False).run([tmp_path], n_gpus=0)
    assert res.n_failed == 0
    import tifffile
    out = tifffile.imread(str(tmp_path / "global_labels_clean.tif"))
    assert set(np.unique(out).tolist()) == {0, 1, 2}   # really re-ran (valid tif)
    man = json.loads((tmp_path / "postprocess_manifest.json").read_text())
    assert man["status"] == "complete"


def test_smooth_one_falls_back_to_original_when_smoothing_erases_cell(tmp_path, monkeypatch):
    # A survivor whose smoothed mask vanishes must be written UNSMOOTHED, never dropped
    # (regression: audit C1 silent deletion).
    import packages.epicycle_scaledseg.core.postprocess_stage as pp
    vol = np.zeros((6, 12, 12), dtype=np.int32)
    vol[2:4, 4:8, 4:8] = 7
    np.save(str(tmp_path / "g.npy"), vol)
    pp._init_smooth_worker(str(tmp_path / "g.npy"))
    monkeypatch.setattr(pp, "smooth_label_sdf",
                        lambda crop, lbl, **k: (np.zeros_like(crop), 0, ""))  # "erase" it
    try:
        res = pp._smooth_one((7, 1, (0, 6, 0, 12, 0, 12), 1.5, 1.5, True))
    finally:
        pp._WORKER_SRC = None
    assert res is not None and res[0] != "ERROR"
    _new_id, _origin, mask = res
    assert int(mask.sum()) == int((vol == 7).sum())   # fell back to the original footprint


def test_smooth_one_error_on_unreadable_crop():
    # A crop-read failure is isolated to one cell (returns ERROR), not raised
    # (regression: audit H1 worker-exception isolation).
    import packages.epicycle_scaledseg.core.postprocess_stage as pp
    pp._WORKER_SRC = None   # not initialized -> subscripting raises -> caught
    res = pp._smooth_one((7, 1, (0, 2, 0, 2, 0, 2), 1.5, 1.5, True))
    assert res[0] == "ERROR"


# --- large filter / percentile / physical units ------------------------------
# fixture sizes: label 1=2048, 2=4, 3=256, 4=1536 voxels.

def test_large_filter_drops_big_cell(tmp_path):
    _build_fixture(tmp_path)
    res = _stage(tmp_path, smooth=False, remove_small=False, remove_islands=False,
                 remove_large=True, max_voxels=1800).run([tmp_path], n_gpus=0)
    assert res.n_failed == 0
    out = np.load(str(tmp_path / "global_labels_clean.npy"))
    # only label 1 (2048 > 1800) dropped; 2,3,4 survive -> renumbered 1..3.
    assert set(np.unique(out).tolist()) == {0, 1, 2, 3}
    man = json.loads((tmp_path / "postprocess_manifest.json").read_text())
    assert man["n_dropped_large"] == 1 and man["eff_max_voxels"] == 1800


def test_max_percentile_overrides_absolute(tmp_path):
    _build_fixture(tmp_path)
    res = _stage(tmp_path, smooth=False, remove_small=False, remove_islands=False,
                 remove_large=True, max_percentile=75.0).run([tmp_path], n_gpus=0)
    assert res.n_failed == 0
    man = json.loads((tmp_path / "postprocess_manifest.json").read_text())
    assert man["max_percentile"] == 75.0
    assert man["eff_max_voxels"] > 0          # resolved from the run's distribution
    assert man["n_dropped_large"] >= 1


def test_min_percentile_drops_smallest(tmp_path):
    _build_fixture(tmp_path)
    _stage(tmp_path, smooth=False, remove_islands=False, remove_large=False,
           remove_small=True, min_percentile=30.0).run([tmp_path], n_gpus=0)
    man = json.loads((tmp_path / "postprocess_manifest.json").read_text())
    assert man["min_percentile"] == 30.0
    assert man["eff_min_voxels"] > 0
    assert man["n_dropped_size"] >= 1         # the tiny fragment goes


def test_voxel_size_reported_in_manifest(tmp_path):
    _build_fixture(tmp_path)
    _stage(tmp_path, smooth=False, voxel_size_um=(1.0, 0.5, 0.5)).run([tmp_path], n_gpus=0)
    man = json.loads((tmp_path / "postprocess_manifest.json").read_text())
    assert man["voxel_size_um"] == [1.0, 0.5, 0.5]
    assert abs(man["voxel_volume_um3"] - 0.25) < 1e-9   # 1.0 * 0.5 * 0.5

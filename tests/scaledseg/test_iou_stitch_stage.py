"""
TDD suite for the Mode-B ("full") IoU seam-stitch ORCHESTRATOR + WRITER.

The pure algorithm core (``epicycle/core/iou_stitch.py``) is verified separately
(``tests/test_iou_stitch.py``). THIS suite pins the Stage that drives it
(``epicycle/core/iou_stitch_stage.py``): geometry loading, the RAM-bounded
slice-streaming writer (memmap .npy + byte-identical BigTIFF), the
background-0-preserving relabel, the pure ``compose_slice`` factoring, and the
result manifest.

These tests are GPU/Qt/napari-free. They build on the conftest fixtures
(``make_tile_mask``, ``two_overlapping_tiles``, ``four_corner_tiles``) and the
flush-anchored tile geometry from ``compute_tile_grid``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import tifffile

from packages.epicycle_scaledseg.core.iou_stitch_stage import (
    IoUStitchStage,
    compose_slice,
    global_to_local_rect,
)
from packages.epicycle_scaledseg.utils.tile_grid import compute_tile_grid


# ---------------------------------------------------------------------------
# tiling_config.json builders (mirror the real Stage-0 schema)
# ---------------------------------------------------------------------------


def _geom_entry(spec):
    """Build one tile_geometry entry from a conftest TileSpec."""
    return {
        "tile_name": spec.name,
        "origin": [int(spec.y0), int(spec.x0)],
        "extent": [int(spec.patch_size), int(spec.patch_size)],
        "grid_index": [int(spec.row), int(spec.col)],
        "is_edge": True,
        "is_flush": False,
    }


def _write_tiling_config(base_dir: Path, full_size, depth, specs):
    """Write a tiling_config.json next to the tile dirs (the stage auto-finds it)."""
    cfg = {
        "full_size": [int(full_size[0]), int(full_size[1])],
        "depth": int(depth),
        "n_tiles": len(specs),
        "tile_geometry": [_geom_entry(s) for s in specs],
    }
    path = Path(base_dir) / "tiling_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def _tile_dirs(specs):
    return [s.dir for s in specs]


# ---------------------------------------------------------------------------
# global_to_local_rect helper
# ---------------------------------------------------------------------------


def test_global_to_local_rect_basic():
    # global (12,16,12,16) within a tile at origin (12,0) -> local (0,4,12,16)
    assert global_to_local_rect((12, 16, 12, 16), (12, 0)) == (0, 4, 12, 16)


def test_global_to_local_rect_zero_origin():
    assert global_to_local_rect((0, 16, 12, 16), (0, 0)) == (0, 16, 12, 16)


# ===========================================================================
# BLOCKING — end-to-end two-tile straddle
# ===========================================================================


def test_end_to_end_two_tile_straddle(two_overlapping_tiles, tmp_path):
    """The straddle cell collapses to ONE global label spanning both tiles'
    regions; the two singletons stay distinct. Output shape == (depth, H, W)."""
    t0, t1 = two_overlapping_tiles
    base = t0.dir.parent  # the fixture's tiles/ dir
    # Geometry: tile_0_0 origin x0=0 spans X[0,16); tile_0_1 origin x0=12 spans
    # X[12,28). Full width = 28, height = 16.
    H, W, depth = 16, 28, 16
    _write_tiling_config(base, (H, W), depth, [t0, t1])

    out_dir = tmp_path / "stage_full"
    stage = IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1)
    result = stage.run(_tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "logs")

    assert result.n_failed == 0, result.failures
    out = np.load(out_dir / "global_labels.npy")
    assert out.shape == (depth, H, W)

    # The straddle cell (label 3 in both tiles) maps to ONE global id. Sample a
    # voxel that is uniquely owned by tile_0_0's part of the straddle (global
    # X[10,12) is single-coverage tile_0_0) and one from tile_0_1's part
    # (global X[16,18) is single-coverage tile_0_1).
    z = 6
    left_part = out[z, 4:12, 10:12]   # tile_0_0 straddle, single-coverage
    right_part = out[z, 4:12, 16:18]  # tile_0_1 straddle, single-coverage
    assert (left_part > 0).all()
    assert (right_part > 0).all()
    straddle_ids = set(np.unique(left_part)) | set(np.unique(right_part))
    straddle_ids.discard(0)
    assert len(straddle_ids) == 1, f"straddle cell not unified: {straddle_ids}"
    straddle_id = straddle_ids.pop()

    # Singleton 1 lives in tile_0_0 at global X[2,6) (single coverage); singleton
    # 2 in tile_0_1 at global X[22,26). Both distinct from each other and from 3.
    sing1 = set(np.unique(out[z, 4:12, 2:6]))
    sing2 = set(np.unique(out[z, 4:12, 22:26]))
    sing1.discard(0)
    sing2.discard(0)
    assert len(sing1) == 1 and len(sing2) == 1
    s1, s2 = sing1.pop(), sing2.pop()
    assert len({s1, s2, straddle_id}) == 3, "labels collapsed unexpectedly"

    # Exactly 3 foreground labels total.
    all_labels = set(np.unique(out))
    all_labels.discard(0)
    assert all_labels == {1, 2, 3} or len(all_labels) == 3


# ===========================================================================
# Regression — IO/geometry edge cases (caught by audit)
# ===========================================================================


def test_stage_runs_at_depth_3_page_collapse(make_tile_mask, tmp_path):
    """tifffile collapses a depth-3 contiguous volume into ONE 3D page; the
    writer's per-page read must handle that. Regression for a `len(pages) > z`
    guard that returned a 3D array -> broadcast crash in compose_slice."""
    base = tmp_path / "tiles"
    fz, zr = 3, (0, 3)
    t0 = make_tile_mask(name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
                        full_z=fz, z_range=zr,
                        labels={3: (slice(0, 3), slice(4, 12), slice(10, 16))})
    t1 = make_tile_mask(name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
                        full_z=fz, z_range=zr,
                        labels={3: (slice(0, 3), slice(4, 12), slice(0, 6))})
    _write_tiling_config(base, (16, 28), fz, [t0, t1])

    out_dir = tmp_path / "stage_full"
    stage = IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1)
    result = stage.run(_tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "logs")

    assert result.n_failed == 0, result.failures
    out = np.load(out_dir / "global_labels.npy")
    assert out.shape == (3, 16, 28)
    assert (out > 0).any()  # the straddle cell was written


def test_stage_tolerates_missing_tile_mask(make_tile_mask, tmp_path):
    """A tile present in geometry but with NO mask on disk must not crash the
    stitch. Regression: computing cores from the present-only (incomplete)
    lattice stretched a survivor's core past its extent -> broadcast crash. With
    cores from the FULL lattice, the missing tile's region is left as background."""
    base = tmp_path / "tiles"
    t0 = make_tile_mask(name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
                        labels={1: (slice(4, 12), slice(2, 6))})
    t1 = make_tile_mask(name="tile_0_1", row=0, col=1, base_dir=base, origin=(0, 12),
                        labels={2: (slice(4, 12), slice(10, 14))})
    _write_tiling_config(base, (16, 28), 16, [t0, t1])  # geometry lists BOTH
    t1.mask_path.unlink()  # ...but tile_0_1's mask is gone

    out_dir = tmp_path / "stage_full"
    stage = IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1)
    result = stage.run(_tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "logs")

    assert result.n_failed == 0, result.failures
    out = np.load(out_dir / "global_labels.npy")
    assert out.shape == (16, 16, 28)
    # tile_0_0's cell is written; the missing tile_0_1's exclusive region is bg 0.
    assert (out[:, 4:12, 2:6] > 0).any()
    assert (out[:, :, 20:28] == 0).all()


# ===========================================================================
# BLOCKING — memmap .npy and exported .tif are byte-identical arrays
# ===========================================================================


def test_memmap_to_tiff_byte_identical(two_overlapping_tiles, tmp_path):
    t0, t1 = two_overlapping_tiles
    base = t0.dir.parent
    _write_tiling_config(base, (16, 28), 16, [t0, t1])

    out_dir = tmp_path / "stage_full"
    stage = IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1)
    stage.run(_tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "logs")

    npy = np.load(out_dir / "global_labels.npy")
    tif = tifffile.imread(str(out_dir / "global_labels.tif"))
    assert npy.shape == tif.shape
    assert npy.dtype == tif.dtype == np.int32
    assert np.array_equal(npy, tif)


def test_tiff_is_bigtiff(two_overlapping_tiles, tmp_path):
    t0, t1 = two_overlapping_tiles
    base = t0.dir.parent
    _write_tiling_config(base, (16, 28), 16, [t0, t1])

    out_dir = tmp_path / "stage_full"
    stage = IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1)
    stage.run(_tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "logs")

    tif_path = out_dir / "global_labels.tif"
    # BigTIFF magic: byte-order marker then version number 43 (0x2B) instead of
    # the classic-TIFF 42 (0x2A). Read the header directly.
    with open(tif_path, "rb") as f:
        header = f.read(4)
    byte_order = header[:2]
    if byte_order == b"II":
        version = struct.unpack("<H", header[2:4])[0]
    else:
        version = struct.unpack(">H", header[2:4])[0]
    assert version == 43, f"not a BigTIFF (version={version}, expected 43)"
    # And tifffile confirms it explicitly.
    with tifffile.TiffFile(str(tif_path)) as tf:
        assert tf.is_bigtiff


# ===========================================================================
# BLOCKING — the writer preserves background-0 (the relabel trap)
# ===========================================================================


def test_writer_preserves_background(make_tile_mask, tmp_path):
    """A single tile with background voxels: background MUST stay 0 in the output,
    NOT relabeled to a prior-tile id (the ``local + offset`` -> offset trap)."""
    base = tmp_path / "tiles"
    # Single tile, full coverage. A cell in the interior; the rest is background.
    spec = make_tile_mask(
        name="tile_0_0", row=0, col=0, base_dir=base, origin=(0, 0),
        patch_size=16, full_z=8, z_range=(2, 6),
        labels={5: (slice(2, 6), slice(4, 8), slice(4, 8))},
    )
    _write_tiling_config(base, (16, 16), 8, [spec])

    out_dir = tmp_path / "stage_full"
    stage = IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1)
    result = stage.run([spec.dir], n_gpus=0, log_dir=tmp_path / "logs")
    assert result.n_failed == 0, result.failures

    out = np.load(out_dir / "global_labels.npy")
    # The cell is foreground (some positive label).
    cell = out[2:6, 4:8, 4:8]
    assert (cell > 0).all()
    cell_id = int(np.unique(cell[cell > 0])[0])
    # Background everywhere else must be EXACTLY 0 — not cell_id, not an offset.
    bg_mask = np.ones_like(out, dtype=bool)
    bg_mask[2:6, 4:8, 4:8] = False
    assert (out[bg_mask] == 0).all(), "background was relabeled (the offset trap)"


# ===========================================================================
# BLOCKING — pure compose_slice with a synthetic read_page_fn
# ===========================================================================


def test_compose_slice_pure():
    """compose_slice is pure given a read_page_fn — composites disjoint cores,
    preserves background-0, applies the global->dense relabel."""
    full_size = (8, 12)
    # Two tiles: A origin (0,0) extent (8,8) owns global X[0,6); B origin (0,6)
    # extent (8,6) owns global X[6,12). (Midpoint cut of overlap X[6,8) -> 6/7=6.)
    geoms = [
        {"tile_name": "A", "origin": [0, 0], "extent": [8, 8],
         "grid_index": [0, 0], "is_edge": True, "is_flush": False},
        {"tile_name": "B", "origin": [0, 6], "extent": [8, 6],
         "grid_index": [0, 1], "is_edge": True, "is_flush": False},
    ]
    from packages.epicycle_scaledseg.core.iou_stitch import compute_tile_cores
    cores = compute_tile_cores(geoms, full_size)
    offsets = {"A": 0, "B": 10}

    # Synthetic local pages: A has local label 1 across its whole core; B has
    # local label 1 across its whole page. After offset, A's 1 -> global 1,
    # B's 1 -> global 11. The LUT fuses A:1 and B:1 into dense 7 to prove relabel.
    def read_page_fn(tile_name, z, src_inset):
        y0, y1, x0, x1 = src_inset
        h, w = y1 - y0, x1 - x0
        page = np.ones((h, w), dtype=np.int32)
        # leave a background hole in A's core at local (0:2, 0:2)
        if tile_name == "A":
            page[0:2, 0:2] = 0
        return page

    # LUT: dense maps global 1 and global 11 both to 7 (fused); index 0 -> 0.
    max_global = 11
    lut_array = np.zeros(max_global + 1, dtype=np.int32)
    lut_array[1] = 7
    lut_array[11] = 7

    geoms_cores = [(g["tile_name"], cores[g["tile_name"]]["dst_core"],
                    cores[g["tile_name"]]["src_inset"]) for g in geoms]

    out_slice = compose_slice(
        z=0,
        geoms_cores=geoms_cores,
        offsets=offsets,
        lut_array=lut_array,
        full_size=full_size,
        read_page_fn=read_page_fn,
    )
    assert out_slice.shape == full_size
    assert out_slice.dtype == np.int32
    # The background hole in A's core (global (0:2, 0:2)) stays 0.
    assert (out_slice[0:2, 0:2] == 0).all()
    # Everywhere else is the fused dense label 7.
    fg = out_slice.copy()
    fg[0:2, 0:2] = 7  # treat the hole as if filled for the all-7 assertion
    assert (fg == 7).all(), f"unexpected labels: {np.unique(out_slice)}"


def test_compose_slice_disjoint_cores_no_overwrite():
    """Each tile writes ONLY into its own dst_core — no cross-tile overwrite."""
    full_size = (8, 12)
    geoms = [
        {"tile_name": "A", "origin": [0, 0], "extent": [8, 8],
         "grid_index": [0, 0], "is_edge": True, "is_flush": False},
        {"tile_name": "B", "origin": [0, 6], "extent": [8, 6],
         "grid_index": [0, 1], "is_edge": True, "is_flush": False},
    ]
    from packages.epicycle_scaledseg.core.iou_stitch import compute_tile_cores
    cores = compute_tile_cores(geoms, full_size)
    offsets = {"A": 0, "B": 10}

    def read_page_fn(tile_name, z, src_inset):
        y0, y1, x0, x1 = src_inset
        h, w = y1 - y0, x1 - x0
        # A paints local 1 everywhere, B paints local 2 everywhere.
        lab = 1 if tile_name == "A" else 2
        return np.full((h, w), lab, dtype=np.int32)

    max_global = 12
    lut_array = np.zeros(max_global + 1, dtype=np.int32)
    lut_array[1] = 1   # A:1 -> dense 1
    lut_array[12] = 2  # B:2 -> global 12 -> dense 2

    geoms_cores = [(g["tile_name"], cores[g["tile_name"]]["dst_core"],
                    cores[g["tile_name"]]["src_inset"]) for g in geoms]
    out_slice = compose_slice(0, geoms_cores, offsets, lut_array, full_size, read_page_fn)
    # Midpoint cut of the overlap X[6,8) between spans (0,8) and (6,12) is
    # (6+8)//2 = 7, so A owns X[0,7) -> dense 1; B owns X[7,12) -> dense 2.
    # Read the boundary from the cores rather than hard-coding it.
    cut = cores["A"]["dst_core"][3]  # A's dst_core x1
    assert (out_slice[:, 0:cut] == 1).all()
    assert (out_slice[:, cut:12] == 2).all()


# ===========================================================================
# Result manifest
# ===========================================================================


def test_result_manifest_written(two_overlapping_tiles, tmp_path):
    t0, t1 = two_overlapping_tiles
    base = t0.dir.parent
    _write_tiling_config(base, (16, 28), 16, [t0, t1])

    out_dir = tmp_path / "stage_full"
    stage = IoUStitchStage(recon_root=str(base), output_dir=str(out_dir),
                           vmin=1, tau=0.25, one_to_many="max_iou")
    stage.run(_tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "logs")

    manifest = json.loads((out_dir / "result_manifest.json").read_text())
    assert manifest["mode"] == "full"
    assert manifest["full_size"] == [16, 28]
    assert manifest["depth"] == 16
    assert manifest["n_global_labels"] == 3
    assert manifest["n_tiles"] == 2
    assert manifest["tau"] == 0.25
    assert manifest["vmin"] == 1
    assert manifest["one_to_many"] == "max_iou"
    assert manifest["dtype"] == "int32"
    assert "version" in manifest
    # Output paths recorded.
    assert manifest["output_npy"].endswith("global_labels.npy")
    assert manifest["output_tif"].endswith("global_labels.tif")


# ===========================================================================
# BLOCKING — 4-corner end-to-end
# ===========================================================================


def test_four_corner_end_to_end(four_corner_tiles, tmp_path):
    """The corner-only cell (present in all 4 tiles, absent from every 2-way
    edge slab) resolves to ONE label in the composited output."""
    specs = four_corner_tiles
    base = specs[0].dir.parent
    # 2x2 grid, origins up to (12,12), patch 16 -> full extent 28x28.
    H, W, depth = 28, 28, 16
    _write_tiling_config(base, (H, W), depth, specs)

    out_dir = tmp_path / "stage_full"
    stage = IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1)
    result = stage.run(_tile_dirs(specs), n_gpus=0, log_dir=tmp_path / "logs")
    assert result.n_failed == 0, result.failures

    out = np.load(out_dir / "global_labels.npy")
    assert out.shape == (depth, H, W)
    # The shared corner square is global Y[12,16) x X[12,16). It must be ONE label.
    z = 6
    corner = out[z, 12:16, 12:16]
    corner_ids = set(np.unique(corner))
    corner_ids.discard(0)
    assert len(corner_ids) == 1, f"corner cell not unified: {corner_ids}"

    # 4 private singletons stay distinct -> total 5 labels (4 singletons + 1 corner).
    all_labels = set(np.unique(out))
    all_labels.discard(0)
    assert len(all_labels) == 5, f"expected 5 labels, got {sorted(all_labels)}"


# ===========================================================================
# RAM invariant — the writer reads ONE tile page (cropped to its core) per
# tile per slice; never a whole patch page, never a whole volume.
# ===========================================================================


def test_writer_reads_one_cropped_page_per_tile_per_slice(
    two_overlapping_tiles, tmp_path, monkeypatch
):
    """Concrete spy on the writer's page reader: each call decodes exactly one
    Z-page cropped to the tile's src_inset (a full-page-decode or whole-volume
    regression would fail here). The total read count == depth * n_tiles."""
    import packages.epicycle_scaledseg.core.iou_stitch_stage as mod

    t0, t1 = two_overlapping_tiles
    base = t0.dir.parent
    H, W, depth = 16, 28, 16
    _write_tiling_config(base, (H, W), depth, [t0, t1])

    calls = {"n": 0, "bad_shape": 0}
    real_read = mod._read_tile_page

    def spy_read(mask_path, z, src_inset):
        calls["n"] += 1
        page = real_read(mask_path, z, src_inset)
        y0, y1, x0, x1 = src_inset
        # The decoded page must be a single 2D slice cropped to the core rect.
        if page.ndim != 2 or page.shape != (y1 - y0, x1 - x0):
            calls["bad_shape"] += 1
        return page

    monkeypatch.setattr(mod, "_read_tile_page", spy_read)

    out_dir = tmp_path / "stage_full"
    stage = IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1)
    stage.run(_tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "logs")

    assert calls["bad_shape"] == 0, "writer decoded a non-cropped or 3D page"
    # 2 tiles * 16 slices = 32 single-page reads — never a bulk volume load.
    assert calls["n"] == depth * 2


# ===========================================================================
# Seam provenance — the QC worklist (label_provenance.csv + manifest summary)
# ===========================================================================


def test_stage_emits_seam_provenance(two_overlapping_tiles, tmp_path):
    """The stitch writes label_provenance.csv classifying each final cell as
    clean/seam/corner. The straddle cell (spans 2 tiles) -> 'seam'; the two
    singletons -> 'clean'. The manifest carries the seam summary."""
    import csv as _csv
    import json as _json

    t0, t1 = two_overlapping_tiles
    base = t0.dir.parent
    _write_tiling_config(base, (16, 28), 16, [t0, t1])
    out_dir = tmp_path / "stage_full"
    IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1).run(
        _tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "logs")

    prov = out_dir / "label_provenance.csv"
    assert prov.exists()
    rows = list(_csv.DictReader(open(prov)))
    by_status = {}
    for r in rows:
        by_status[r["seam_status"]] = by_status.get(r["seam_status"], 0) + 1
    assert by_status.get("seam", 0) == 1   # the straddle cell crossed the seam
    assert by_status.get("clean", 0) == 2  # the two singletons

    seam_row = next(r for r in rows if r["seam_status"] == "seam")
    assert int(seam_row["n_tiles"]) == 2
    assert "tile_0_0" in seam_row["tiles"] and "tile_0_1" in seam_row["tiles"]
    assert int(seam_row["voxel_count"]) > 0
    assert int(seam_row["z_min"]) >= 0 and int(seam_row["z_max"]) >= int(seam_row["z_min"])

    m = _json.loads((out_dir / "result_manifest.json").read_text())
    assert m["seam_summary"]["seam"] == 1
    assert m["seam_summary"]["clean"] == 2
    assert "border_clipped" in m["seam_summary"]


def test_stage_provenance_corner_cell(four_corner_tiles, tmp_path):
    """A cell present in the 4-way corner is classified 'corner' (n_tiles>=3)."""
    import csv as _csv

    tiles = four_corner_tiles
    specs = list(tiles.values()) if isinstance(tiles, dict) else list(tiles)
    base = specs[0].dir.parent
    # 2x2 grid sharing a corner -> full size 28x28 (origins 0 and 12, patch 16).
    _write_tiling_config(base, (28, 28), specs[0].full_z, specs)
    out_dir = tmp_path / "stage_corner"
    IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1).run(
        [s.dir for s in specs], n_gpus=0, log_dir=tmp_path / "logs")

    rows = list(_csv.DictReader(open(out_dir / "label_provenance.csv")))
    statuses = {r["seam_status"] for r in rows}
    assert "corner" in statuses, f"no corner cell classified: {statuses}"
    corner = next(r for r in rows if r["seam_status"] == "corner")
    assert int(corner["n_tiles"]) >= 3


def test_stage_provenance_bbox_matches_volume_and_min_iou(two_overlapping_tiles, tmp_path):
    """The per-cell ZYX bbox in the CSV exactly bounds the cell in the written
    volume (it's what a QC crop centers on), and the seam cell carries a numeric
    ``min_match_iou`` while clean cells leave it empty."""
    import csv as _csv

    t0, t1 = two_overlapping_tiles
    base = t0.dir.parent
    H, W, depth = 16, 28, 16
    _write_tiling_config(base, (H, W), depth, [t0, t1])
    out_dir = tmp_path / "stage_bbox"
    IoUStitchStage(recon_root=str(base), output_dir=str(out_dir), vmin=1).run(
        _tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "logs")

    vol = np.load(out_dir / "global_labels.npy")
    rows = list(_csv.DictReader(open(out_dir / "label_provenance.csv")))

    for r in rows:
        if int(r["voxel_count"]) == 0:
            continue  # merged/clipped-away phantom — no voxels to bound
        d = int(r["dense_id"])
        zz, yy, xx = np.nonzero(vol == d)
        assert zz.size > 0, f"dense {d} claims {r['voxel_count']} voxels but none found"
        # bbox columns are INCLUSIVE min/max in voxels — must match the volume exactly.
        assert int(r["z_min"]) == int(zz.min()) and int(r["z_max"]) == int(zz.max())
        assert int(r["y_min"]) == int(yy.min()) and int(r["y_max"]) == int(yy.max())
        assert int(r["x_min"]) == int(xx.min()) and int(r["x_max"]) == int(xx.max())
        assert int(r["voxel_count"]) == int(zz.size)

    seam_row = next(r for r in rows if r["seam_status"] == "seam")
    assert seam_row["min_match_iou"] != ""             # seam cell ranked by confidence
    assert 0.0 < float(seam_row["min_match_iou"]) <= 1.0
    for r in rows:
        if r["seam_status"] == "clean":
            assert r["min_match_iou"] == ""             # nothing for a seam to break


def test_stage_emit_provenance_false_is_byte_identical(two_overlapping_tiles, tmp_path):
    """emit_provenance=False drops the CSV + nulls the manifest seam fields, and
    the label volume is byte-identical to the emit=True run (the stats pass is
    read-only — disabling it cannot change the output)."""
    import json as _json

    t0, t1 = two_overlapping_tiles
    base = t0.dir.parent
    H, W, depth = 16, 28, 16
    _write_tiling_config(base, (H, W), depth, [t0, t1])

    out_on = tmp_path / "on"
    out_off = tmp_path / "off"
    IoUStitchStage(recon_root=str(base), output_dir=str(out_on), vmin=1).run(
        _tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "log_on")
    IoUStitchStage(recon_root=str(base), output_dir=str(out_off), vmin=1,
                   emit_provenance=False).run(
        _tile_dirs([t0, t1]), n_gpus=0, log_dir=tmp_path / "log_off")

    a = np.load(out_on / "global_labels.npy")
    b = np.load(out_off / "global_labels.npy")
    assert np.array_equal(a, b), "disabling provenance changed the label volume"

    assert not (out_off / "label_provenance.csv").exists()
    m = _json.loads((out_off / "result_manifest.json").read_text())
    assert m["emit_provenance"] is False
    assert m["seam_summary"] is None
    assert m["label_provenance"] is None
    # The emit=True run still wrote everything.
    assert (out_on / "label_provenance.csv").exists()
    m_on = _json.loads((out_on / "result_manifest.json").read_text())
    assert m_on["seam_summary"] is not None
    assert m_on["provenance_units"] == "voxels"

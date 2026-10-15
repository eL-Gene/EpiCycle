"""
Stack 5 BLOCKING acceptance gates + the assembly-origin coupling fix.

The flush-no-pad design exists so the assembled output matches the input dims
exactly and instances register 1:1 with the input stack (needed for intensity
feature extraction off the reconstructed mask). These tests pin that end-to-end,
GPU-free, by round-tripping synthetic tile masks through stitch_mask_volume.

The highest-risk integration point: a flush tile's true origin is `dim-patch`,
NOT `grid_index*step`, so assembly must read the STORED origin from the manifest
(not regex-derive it). test_assembly_reads_stored_flush_origins locks that.
"""

import json

import numpy as np
import tifffile

from packages.epicycle_scaledseg.core.assembly import (
    AssemblyStage,
    parse_tile_positions,
    stitch_mask_volume,
)
from packages.epicycle_scaledseg.utils.tile_grid import compute_tile_grid
from packages.epicycle_scaledseg.utils.tiling import ZStackTiler


# ---------------------------------------------------------------------------
# BLOCKING gate 1: assembled output dims == input dims (no pad, no crop)
# ---------------------------------------------------------------------------


def test_output_shape_equals_input(make_tile_mask, tmp_path):
    H, W, patch, depth = 20, 20, 16, 5
    grid = compute_tile_grid(H, W, patch, 0.25)
    base = tmp_path / "tiles"
    specs = [
        make_tile_mask(name=g.name, base_dir=base, origin=g.origin,
                       patch_size=patch, full_z=depth, z_range=(1, 4), labels={})
        for g in grid.tiles
    ]
    out = tmp_path / "stitched.tif"
    stitch_mask_volume(
        [(s.y0, s.x0, s.dir) for s in specs], recon_root=base,
        output_path=str(out), full_size=(H, W), tile_size=patch,
        depth=depth, n_workers=1,
    )
    # full_size passed = input (H,W); flush-no-pad => assembled dims == input.
    assert tifffile.imread(str(out)).shape == (depth, H, W)


# ---------------------------------------------------------------------------
# BLOCKING gate 2: a labelled object registers 1:1 at the same input voxel,
# including one inside a FLUSH tile (origin (4,4) != grid_index*step=12).
# ---------------------------------------------------------------------------


def test_instance_registration(make_tile_mask, tmp_path):
    H, W, patch, depth = 20, 20, 16, 8
    z0, z1 = 2, 6
    grid = compute_tile_grid(H, W, patch, 0.25)
    base = tmp_path / "tiles"
    specs = []
    for g in grid.tiles:
        labels = {}
        if g.grid_index == (0, 0):
            # global [0,4)x[0,4) (single-coverage) -> tile-local [0,4)x[0,4)
            labels = {7: (slice(0, 4), slice(0, 4))}
        elif g.grid_index == (1, 1):
            # FLUSH tile, origin (4,4). global [16,20)x[16,20) (single-coverage)
            # -> tile-local [12,16)x[12,16).
            labels = {9: (slice(12, 16), slice(12, 16))}
        specs.append(make_tile_mask(
            name=g.name, base_dir=base, origin=g.origin,
            patch_size=patch, full_z=depth, z_range=(z0, z1), labels=labels,
        ))
    out = tmp_path / "stitched.tif"
    stitch_mask_volume(
        [(s.y0, s.x0, s.dir) for s in specs], recon_root=base,
        output_path=str(out), full_size=(H, W), tile_size=patch,
        depth=depth, n_workers=1,
    )
    result = tifffile.imread(str(out))
    assert result.shape == (depth, H, W)
    # Object 7 registers at its input voxel.
    assert (result[z0:z1, 0:4, 0:4] == 7).all()
    # Object 9 (inside the FLUSH tile) registers at global [16,20) — only
    # correct because the tile's true origin (4,4) was used, not regex 12.
    assert (result[z0:z1, 16:20, 16:20] == 9).all()


# ---------------------------------------------------------------------------
# The assembly-origin coupling fix
# ---------------------------------------------------------------------------


def test_assembly_reads_stored_flush_origins(tmp_path):
    """Assembly must prefer the STORED origin over the regex one for flush tiles.
    For (20,20),16,0.25 the regex gives tile_1_1 -> (12,12) (WRONG); the stored
    geometry gives (4,4) (correct)."""
    out = tmp_path / "tiles"
    out.mkdir()
    grid = compute_tile_grid(20, 20, 16, 0.25)
    cfg = {
        "full_size": [20, 20], "patch_size": 16, "overlap": 0.25, "depth": 8,
        "tiles": [g.name for g in grid.tiles],
        "tile_geometry": [
            {"tile_name": g.name, "origin": list(g.origin)} for g in grid.tiles
        ],
    }
    (out / "tiling_config.json").write_text(json.dumps(cfg))
    tile_dirs = []
    for g in grid.tiles:
        d = out / g.name
        d.mkdir()
        tile_dirs.append(d)

    stage = AssemblyStage(recon_root=str(out), output_dir=str(tmp_path / "asm"))
    stage._load_tiling_config(tile_dirs)

    assert stage._tile_origins["tile_1_1"] == (4, 4)  # stored, correct
    regex = {td.name: (y, x)
             for y, x, td in parse_tile_positions(tile_dirs, tile_size=16, overlap=0.25)}
    assert regex["tile_1_1"] == (12, 12)  # regex MIS-places the flush tile
    assert stage._tile_origins["tile_1_1"] != regex["tile_1_1"]


# ---------------------------------------------------------------------------
# Real end-to-end tiling — split_tiles is RUNNABLE for the first time
# (patchify is gone; flush-anchoring slices real data).
# ---------------------------------------------------------------------------


def test_tiling_then_stitch_is_spatial_identity(tmp_path):
    """THE alignment guarantee: real tiling + the stitch's core-crop geometry is a
    PERFECT spatial inverse — tile an image, un-tile it via the geometry the
    pipeline actually recorded, and recover the input pixel-for-pixel.

    This is what proves "the reconstruction aligns with the input image data": a
    feature at input voxel (z,c,y,x) comes back at the SAME voxel. It is
    independent of segmentation/reconstruction, which are position-PRESERVING
    (they change voxel VALUES per tile, never positions), so geometry is the only
    thing that could misalign — and this pins it. Uses an AWKWARD (H,W) that is
    not a multiple of the patch, with unique per-voxel values so any shift,
    transpose, or off-by-one anywhere in tiling/stitch is caught."""
    import json
    from packages.epicycle_scaledseg.utils.tiling import ZStackTiler
    from packages.epicycle_scaledseg.core.iou_stitch import compute_tile_cores

    Z, C, H, W = 5, 2, 37, 23  # awkward dims, not multiples of patch=16
    # Distinct value at every voxel (uint16 = the real microscopy dtype; 8510
    # voxels < 65535 so all distinct) + signal in every Z slice (so no Z-trim).
    img = (np.arange(Z * C * H * W, dtype=np.uint16).reshape(Z, C, H, W) + 1)
    p = tmp_path / "stack.tif"
    tifffile.imwrite(str(p), img)

    out_dir = tmp_path / "out"
    tiler = ZStackTiler(p, out_dir, target_patch_size=16, overlap=0.25, search_range=0)
    tiler.split_tiles()

    cfg = json.loads((out_dir / "tiling_config.json").read_text())
    geoms = cfg["tile_geometry"]
    cores = compute_tile_cores(geoms, (H, W))

    # Un-tile: each tile writes only its disjoint CORE back to its global position.
    recon = np.zeros((Z, C, H, W), dtype=img.dtype)
    for g in geoms:
        name = g["tile_name"]
        tile_tif = next((out_dir / name).glob(f"{name}_segmentrange_*.tif"))
        tile_img = np.asarray(tifffile.imread(str(tile_tif)))
        assert tile_img.shape[:2] == (Z, C), f"tile {name} read back as {tile_img.shape}"
        sy0, sy1, sx0, sx1 = cores[name]["src_inset"]
        dy0, dy1, dx0, dx1 = cores[name]["dst_core"]
        recon[:, :, dy0:dy1, dx0:dx1] = tile_img[:, :, sy0:sy1, sx0:sx1]

    assert recon.shape == img.shape
    # Pixel-perfect: nothing shifted, transposed, dropped, or duplicated.
    assert np.array_equal(recon, img)


def test_sub_patch_mask_stitches_without_overread(tmp_path):
    """A mask physically SMALLER than tile_size (dim < patch -> single sub-patch
    tile) must stitch without over-reading. The assembly min()-clamp caps the
    write to the real extent. (The fixtures always write patch-sized masks, so
    this genuinely-small case needs a hand-built mask.)"""
    base = tmp_path / "tiles"
    tdir = base / "tile_0_0"
    tdir.mkdir(parents=True)
    arr = np.zeros((6, 10, 10), dtype=np.uint32)  # 10x10 < patch 16
    arr[2:4, 1:5, 1:5] = 7
    tifffile.imwrite(str(tdir / "tile_0_0_mask.tif"), arr, bigtiff=True)

    out = tmp_path / "stitched.tif"
    stitch_mask_volume(
        [(0, 0, tdir)], recon_root=base, output_path=str(out),
        full_size=(10, 10), tile_size=16, depth=6, n_workers=1,
    )
    result = tifffile.imread(str(out))
    assert result.shape == (6, 10, 10)  # == input, not padded up to patch
    assert (result[2:4, 1:5, 1:5] == 7).all()


def test_split_tiles_real_geometry_matches_planner(tmp_path):
    """A real ZStackTiler.split_tiles run on an awkward-dimension stack writes
    tile geometry that matches compute_tile_grid exactly, with full coverage."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 100, (4, 1, 23, 17), dtype=np.uint16)  # ZCYX, awkward H,W
    p = tmp_path / "stack.tif"
    tifffile.imwrite(str(p), arr)

    out = tmp_path / "out"
    tiler = ZStackTiler(p, out, target_patch_size=16, overlap=0.25, search_range=0)
    tile_dirs = tiler.split_tiles()

    grid = compute_tile_grid(23, 17, 16, 0.25)
    assert len(tile_dirs) == len(grid.tiles)

    cfg = json.loads((out / "tiling_config.json").read_text())
    assert cfg["n_tiles"] == len(grid.tiles)
    planner = {tuple(g.origin) for g in grid.tiles}
    written = {tuple(t["origin"]) for t in cfg["tile_geometry"]}
    assert written == planner
    # Full coverage: the max origin+patch reaches the boundary in each axis.
    assert max(o[0] + 16 for o in written) == 23
    assert max(o[1] + 16 for o in written) == 17

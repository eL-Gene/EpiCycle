"""Roundtrip tests for the read-only global-ROI reader (epicycle.reconcile.raw_roi).

Builds a synthetic multi-channel ZCYX stack whose voxel values ENCODE their (channel,
y, x) position, tiles it with the REAL ``ZStackTiler`` (so the on-disk format — ZCYX
tiles, tiling_config geometry, 25% overlap — is the production one), then asserts
``read_global_roi`` reconstructs known global bboxes byte-for-byte. Covers: single-tile,
seam-spanning, halo across a seam, channel indexing, FOV-border clamp, full-Z (no
truncated_range), and the per-tile probability reader. No GPU."""

import json
from pathlib import Path

import numpy as np
import pytest

from packages.epicycle_scaledseg.reconcile.raw_roi import (
    load_geometry, resolve_channel_layout, read_global_roi, read_prob_roi,
)


def _encode_volume(Z=5, C=2, H=40, W=40):
    """value = channel*100_000 + y*1000 + x  (z-independent so any crop is checkable).

    float32 — exact for these integer codes (< 2**24) and ImageJ-TIFF-supported
    (ImageJ rejects int32). Real data is uint16/float32; the reader returns the
    tile's native dtype regardless."""
    vol = np.zeros((Z, C, H, W), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    for c in range(C):
        plane = (c * 100_000 + yy * 1000 + xx).astype(np.float32)
        for z in range(Z):
            vol[z, c] = plane
    return vol


@pytest.fixture
def tiled(tmp_path):
    """Tile a synthetic ZCYX stack with the real tiler; return (tiles_dir, vol)."""
    from packages.epicycle_scaledseg.utils.tiling import ZStackTiler
    import tifffile

    vol = _encode_volume()
    tif = tmp_path / "input.tif"
    tifffile.imwrite(str(tif), vol, imagej=True, metadata={"axes": "ZCYX"})
    tiles_dir = tmp_path / "tiles"
    tiler = ZStackTiler(
        tif_path=str(tif), output_dir=str(tiles_dir),
        target_patch_size=24, overlap=0.25,
        z_threshold_method="default", threshold_value=0.0, channel_index=1,
    )
    tiler.split_tiles()
    return tiles_dir, vol


def test_geometry_and_channel_layout(tiled):
    tiles_dir, vol = tiled
    geo = load_geometry(tiles_dir)
    assert geo["full"] == (vol.shape[2], vol.shape[3])
    assert geo["depth"] == vol.shape[0]
    assert geo["channels"] == vol.shape[1]
    assert len(geo["geoms"]) >= 4              # 24px patch over 40px -> 2x2 grid
    lay = resolve_channel_layout(tiles_dir)
    assert lay["n_channels"] == vol.shape[1]


def test_signal_channel_cellpose_to_array_index(tiled, tmp_path):
    # Cellpose channels=[2,1] (signal=2, 1-based) -> array index 1, NOT 2.
    tiles_dir, vol = tiled
    name = sorted(load_geometry(tiles_dir)["geoms"].keys())[0]
    cfg_path = next((Path(tiles_dir) / name).glob("*_config.json"))
    d = json.loads(cfg_path.read_text())
    d["segmentation_params"] = {"channels": [2, 1]}
    cfg_path.write_text(json.dumps(d))
    lay = resolve_channel_layout(tiles_dir)
    assert lay["signal_channel_cellpose"] == 2
    assert lay["signal_channel"] == 1            # 2 (cellpose) -> 1 (array index)
    # and the resolved index actually reads on a 2-channel tile (no IndexError)
    out = read_global_roi(tiles_dir, (0, 1, 0, 4, 0, 4), channel=lay["signal_channel"])
    assert np.array_equal(out, vol[0:1, 1, 0:4, 0:4])


def test_nucleus_channel_cellpose_to_array_index(tiled):
    # Cellpose CHANNELS=[signal, nucleus]=[2,1] -> nucleus is cellpose 1 -> array index 0.
    tiles_dir, vol = tiled
    name = sorted(load_geometry(tiles_dir)["geoms"].keys())[0]
    cfg_path = next((Path(tiles_dir) / name).glob("*_config.json"))
    d = json.loads(cfg_path.read_text())
    d["segmentation_params"] = {"channels": [2, 1]}
    cfg_path.write_text(json.dumps(d))
    lay = resolve_channel_layout(tiles_dir)
    assert lay["nucleus_channel_cellpose"] == 1
    assert lay["nucleus_channel"] == 0           # 1 (cellpose) -> 0 (array index)
    out = read_global_roi(tiles_dir, (0, 1, 0, 4, 0, 4), channel=lay["nucleus_channel"])
    assert np.array_equal(out, vol[0:1, 0, 0:4, 0:4])


def test_nucleus_channel_zero_means_none(tiled):
    # Cellpose nucleus == 0 means "no nucleus channel used" -> None, NOT array index 0.
    tiles_dir, _ = tiled
    name = sorted(load_geometry(tiles_dir)["geoms"].keys())[0]
    cfg_path = next((Path(tiles_dir) / name).glob("*_config.json"))
    d = json.loads(cfg_path.read_text())
    d["segmentation_params"] = {"channels": [2, 0]}
    cfg_path.write_text(json.dumps(d))
    lay = resolve_channel_layout(tiles_dir)
    assert lay["signal_channel"] == 1
    assert lay["nucleus_channel"] is None


def test_single_tile_roi_roundtrip(tiled):
    tiles_dir, vol = tiled
    # a bbox safely inside the top-left tile (origin 0,0; extent 24)
    bbox = (1, 4, 3, 13, 3, 13)            # z0,z1,y0,y1,x0,x1 (half-open)
    for c in range(vol.shape[1]):
        out = read_global_roi(tiles_dir, bbox, channel=c)
        expected = vol[1:4, c, 3:13, 3:13]
        assert out.shape == expected.shape
        assert np.array_equal(out, expected), f"channel {c} mismatch"


def test_seam_spanning_roi(tiled):
    tiles_dir, vol = tiled
    # a bbox straddling the vertical seam (x around the mid of a 40px volume)
    bbox = (0, 5, 5, 35, 14, 26)
    out = read_global_roi(tiles_dir, bbox, channel=0)
    expected = vol[0:5, 0, 5:35, 14:26]
    assert np.array_equal(out, expected), "seam-spanning crop wrong"


def test_halo_across_seam(tiled):
    tiles_dir, vol = tiled
    # small bbox near the center seam, grown by an XY halo that crosses the seam
    # (halo_z=0 to isolate the XY behavior under test)
    bbox = (2, 3, 18, 22, 18, 22)
    out = read_global_roi(tiles_dir, bbox, channel=1, halo=6, halo_z=0)
    expected = vol[2:3, 1, 12:28, 12:28]
    assert out.shape == expected.shape
    assert np.array_equal(out, expected), "haloed seam crop wrong"


def test_fov_border_clamp(tiled):
    tiles_dir, vol = tiled
    # bbox at the very corner + an XY halo that would run off the volume -> clamps, no crash
    bbox = (0, 2, 0, 6, 0, 6)
    out = read_global_roi(tiles_dir, bbox, channel=0, halo=10, halo_z=0)
    expected = vol[0:2, 0, 0:16, 0:16]      # halo clamped to [0, 16) on the low corner
    assert np.array_equal(out, expected), "FOV-border clamp wrong"


def test_full_z_not_truncated(tiled):
    tiles_dir, vol = tiled
    # the LAST z-slice must be readable (raw tile is full-Z; truncated_range NOT applied)
    Z = vol.shape[0]
    out = read_global_roi(tiles_dir, (Z - 1, Z, 5, 10, 5, 10), channel=0)
    assert np.array_equal(out, vol[Z - 1:Z, 0, 5:10, 5:10])


def test_channel_out_of_range_raises(tiled):
    tiles_dir, vol = tiled
    with pytest.raises(IndexError):
        read_global_roi(tiles_dir, (0, 1, 0, 4, 0, 4), channel=vol.shape[1])


def test_prob_roi_roundtrip(tmp_path, tiled):
    """The prob reader assembles per-tile (Z,Y,X) crops with the same geometry."""
    tiles_dir, vol = tiled
    geo = load_geometry(tiles_dir)
    Z, H, W = geo["depth"], geo["full"][0], geo["full"][1]
    # fabricate a per-tile prob (full-Z (Z,Y,X)) carrying a global-position code so a
    # reassembled crop is verifiable; cropping each tile from one global field.
    field = np.zeros((Z, H, W), dtype=np.float16)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    for z in range(Z):
        field[z] = (yy * 0.001 + xx * 0.0001).astype(np.float16)
    recon = tmp_path / "stage2_reconstruction"
    for name, ((oy, ox), (h, w)) in geo["geoms"].items():
        d = recon / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{name}_probabilities.npy", field[:, oy:oy + h, ox:ox + w])
    out = read_prob_roi(tiles_dir, recon, (1, 3, 10, 30, 10, 30), halo=0)
    assert np.array_equal(out, field[1:3, 10:30, 10:30])


def test_prob_sigmoid_option(tmp_path, tiled):
    tiles_dir, vol = tiled
    geo = load_geometry(tiles_dir)
    Z, H, W = geo["depth"], geo["full"][0], geo["full"][1]
    recon = tmp_path / "stage2_reconstruction"
    for name, ((oy, ox), (h, w)) in geo["geoms"].items():
        d = recon / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{name}_probabilities.npy", np.zeros((Z, h, w), dtype=np.float16))
    out = read_prob_roi(tiles_dir, recon, (0, 1, 0, 8, 0, 8), apply_sigmoid=True)
    assert np.allclose(out, 0.5, atol=1e-3)   # expit(0) == 0.5

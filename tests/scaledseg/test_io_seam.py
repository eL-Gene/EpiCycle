"""
Tests for the volume-IO format-dispatch seam (Stack 7) + the chunked-zarr
probability stitch (the OOM fix that is the seam's first use).

The seam (``epicycle/utils/io.py``) lets the engine write/read volumes as
BigTIFF (default) or raw zarr; the chunked prob stitch
(``epicycle/core/assembly.stitch_prob_volume_zarr``) uses it to stitch without
materializing the whole ``(depth, H, W)`` float32 volume that the numpy
``stitch_prob_volume`` allocates (a guaranteed OOM at 7680²×Z).
"""

import json

import numpy as np
import pytest
import tifffile

from packages.epicycle_scaledseg.utils.io import (
    load_volume,
    open_working_volume,
    save_volume,
)
from packages.epicycle_scaledseg.core.assembly import (
    AssemblyStage,
    stitch_prob_volume,
    stitch_prob_volume_zarr,
)


# ---------------------------------------------------------------------------
# save_volume / load_volume — format dispatch
# ---------------------------------------------------------------------------


def _vol():
    return np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)


def test_tiff_roundtrip(tmp_path):
    v = _vol()
    p = tmp_path / "vol.tif"
    save_volume(v, p)
    assert np.array_equal(load_volume(p), v)
    with tifffile.TiffFile(str(p)) as tf:
        assert tf.is_bigtiff


def test_zarr_directory_roundtrip(tmp_path):
    v = _vol()
    p = tmp_path / "vol.zarr"
    save_volume(v, p)
    out = load_volume(p)
    assert np.array_equal(out, v) and out.dtype == v.dtype


def test_zarr_zip_roundtrip(tmp_path):
    """A .zip path uses a ZipStore (the GPFS-safe single-file backend)."""
    v = _vol()
    p = tmp_path / "vol.zip"
    save_volume(v, p)
    assert np.array_equal(load_volume(p), v)
    assert p.exists() and p.is_file()  # single file, not an inode-storm directory


def test_tiff_and_zarr_produce_identical_arrays(tmp_path):
    """The migration-safety equivalence: same array out of either backend."""
    v = _vol()
    t = tmp_path / "v.tif"
    z = tmp_path / "v.zarr"
    save_volume(v, t)
    save_volume(v, z)
    assert np.array_equal(load_volume(t), load_volume(z))


def test_format_auto_by_suffix(tmp_path):
    v = _vol()
    # .tif -> tiff (multi-page TIFF); .zarr -> zarr (a directory).
    save_volume(v, tmp_path / "a.tif")
    save_volume(v, tmp_path / "b.zarr")
    assert (tmp_path / "a.tif").is_file()
    assert (tmp_path / "b.zarr").is_dir()


def test_explicit_fmt_overrides_suffix(tmp_path):
    v = _vol()
    p = tmp_path / "weird.dat"
    save_volume(v, p, fmt="tiff")
    assert np.array_equal(load_volume(p, fmt="tiff"), v)


def test_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        save_volume(_vol(), tmp_path / "x.dat")  # unknown suffix, fmt=auto
    with pytest.raises(ValueError):
        save_volume(_vol(), tmp_path / "x.tif", fmt="bogus")


# ---------------------------------------------------------------------------
# open_working_volume
# ---------------------------------------------------------------------------


def test_open_working_volume_numpy():
    a = open_working_volume((2, 3, 4), np.float32, backend="numpy")
    assert isinstance(a, np.ndarray) and a.shape == (2, 3, 4)
    assert (a == 0).all()


def test_open_working_volume_zarr_region_accumulate(tmp_path):
    z = open_working_volume((4, 8, 8), np.float32, backend="zarr",
                            store_path=str(tmp_path / "w.zarr"), chunks=(2, 8, 8))
    # zarr region read-modify-write works (the stitch's accumulation primitive).
    z[0:2, 0:4, 0:4] = z[0:2, 0:4, 0:4] + 1.0
    z[0:2, 0:4, 0:4] = z[0:2, 0:4, 0:4] + 1.0
    assert np.allclose(z[0:2, 0:4, 0:4], 2.0)
    assert np.allclose(z[2:4], 0.0)  # untouched stays 0


def test_open_working_volume_zarr_requires_store_path():
    with pytest.raises(ValueError):
        open_working_volume((2, 2, 2), np.float32, backend="zarr")


def test_open_working_volume_bad_backend():
    with pytest.raises(ValueError):
        open_working_volume((2, 2, 2), np.float32, backend="bogus")


# ---------------------------------------------------------------------------
# stitch_prob_volume_zarr == stitch_prob_volume (numpy), RAM-bounded
# ---------------------------------------------------------------------------


def _two_prob_tiles(tmp_path, depth=8, p0=0.8, p1=0.4):
    base = tmp_path / "recon"
    for name, val in [("tile_0_0", p0), ("tile_0_1", p1)]:
        d = base / name
        d.mkdir(parents=True)
        np.save(d / f"{name}_probabilities.npy",
                np.full((depth, 4, 4), val, dtype=np.float32))
    positions = [(0, 0, base / "tile_0_0"), (0, 2, base / "tile_0_1")]
    return base, positions


@pytest.mark.parametrize("mode", ["average", "max"])
def test_stitch_prob_zarr_matches_numpy(tmp_path, mode):
    base, positions = _two_prob_tiles(tmp_path)
    out_npy = tmp_path / "prob_numpy.npy"
    out_zarr = tmp_path / "prob_zarr.zarr"
    stitch_prob_volume(positions, recon_root=base, output_path=str(out_npy),
                       full_size=(4, 6), tile_size=4, depth=8, stitch_mode=mode)
    stitch_prob_volume_zarr(positions, recon_root=base, output_path=str(out_zarr),
                            full_size=(4, 6), tile_size=4, depth=8, stitch_mode=mode,
                            z_chunk=2)
    numpy_result = np.load(str(out_npy))
    zarr_result = load_volume(out_zarr)
    assert zarr_result.shape == numpy_result.shape == (8, 4, 6)
    assert np.allclose(zarr_result, numpy_result)


def test_zarr_chunks_match_tile_grid(tmp_path):
    """The working zarr store's chunks are tile-aligned (Z-slab x tile plane), so
    a tile-region write touches a few chunks, not the whole array."""
    base, positions = _two_prob_tiles(tmp_path, depth=8)
    out = tmp_path / "prob.zarr"
    stitch_prob_volume_zarr(positions, recon_root=base, output_path=str(out),
                            full_size=(4, 6), tile_size=4, depth=8,
                            stitch_mode="average", z_chunk=2)
    import zarr
    z = zarr.open(str(out), mode="r")
    # chunks = (min(z_chunk,depth), min(tile_size,H), min(tile_size,W)) = (2,4,4)
    assert z.chunks == (2, 4, 4)


def test_stitch_prob_zarr_to_zip_roundtrips(tmp_path):
    """A .zip output is a single GPFS-safe file that loads back correctly. The
    accumulation happens in a DirectoryStore work dir (ZipStore is write-once),
    then is stream-copied to the ZipStore — and the work dir is cleaned up."""
    base, positions = _two_prob_tiles(tmp_path, depth=8)
    out_zarr = tmp_path / "ref.zarr"
    out_zip = tmp_path / "prob.zip"
    stitch_prob_volume_zarr(positions, recon_root=base, output_path=str(out_zarr),
                            full_size=(4, 6), tile_size=4, depth=8,
                            stitch_mode="average", z_chunk=2)
    stitch_prob_volume_zarr(positions, recon_root=base, output_path=str(out_zip),
                            full_size=(4, 6), tile_size=4, depth=8,
                            stitch_mode="average", z_chunk=2)
    assert out_zip.is_file()  # single file, not an inode-storm directory
    assert np.allclose(load_volume(out_zip), load_volume(out_zarr))
    # the DirectoryStore work dir was cleaned up.
    assert not (tmp_path / "prob.zip.work.zarr").exists()


def test_stitch_prob_zarr_never_allocates_full_volume(tmp_path, monkeypatch):
    """The zarr path must never np.zeros the whole (depth,H,W) volume — that's
    the OOM it exists to avoid."""
    base, positions = _two_prob_tiles(tmp_path, depth=8)
    full_shape = (8, 4, 6)
    real_zeros = np.zeros

    def guard_zeros(shape, *a, **k):
        shp = tuple(shape) if hasattr(shape, "__len__") else (shape,)
        assert shp != full_shape, f"zarr stitch allocated full volume np.zeros({shp})"
        return real_zeros(shape, *a, **k)

    monkeypatch.setattr(np, "zeros", guard_zeros)
    stitch_prob_volume_zarr(positions, recon_root=base,
                            output_path=str(tmp_path / "p.zarr"),
                            full_size=(4, 6), tile_size=4, depth=8,
                            stitch_mode="average", z_chunk=2)


# ---------------------------------------------------------------------------
# AssemblyStage prob_backend wiring (follow-up #9)
# ---------------------------------------------------------------------------


def test_assembly_prob_backend_param_and_default():
    """Default backend is numpy (byte-identical); zarr is opt-in + in get_params."""
    assert AssemblyStage().prob_backend == "numpy"
    s = AssemblyStage(prob_backend="zarr")
    assert s.prob_backend == "zarr"
    assert s.get_params()["prob_backend"] == "zarr"


def _assembly_prob_scenario(tmp_path, depth=8):
    """Two overlapping prob tiles + a tiling_config, ready for AssemblyStage.run."""
    recon = tmp_path / "recon"
    tile_dirs, geoms = [], []
    for name, x0, val in [("tile_0_0", 0, 0.8), ("tile_0_1", 2, 0.4)]:
        d = recon / name
        d.mkdir(parents=True)
        np.save(d / f"{name}_probabilities.npy",
                np.full((depth, 4, 4), val, dtype=np.float32))
        tile_dirs.append(d)
        geoms.append({"tile_name": name, "origin": [0, x0], "extent": [4, 4],
                      "grid_index": [0, x0 // 2], "is_edge": True, "is_flush": False})
    cfg = {"full_size": [4, 6], "depth": depth, "patch_size": 4, "overlap": 0.25,
           "tiles": [g["tile_name"] for g in geoms], "tile_geometry": geoms}
    (recon / "tiling_config.json").write_text(json.dumps(cfg))
    return recon, tile_dirs


def test_assembly_prob_backend_zarr_matches_numpy(tmp_path):
    """AssemblyStage(prob_backend='zarr') writes full_probabilities.zarr whose
    contents match the numpy backend's full_probabilities.npy — the OOM-safe path
    is wired in + equivalent. (Nothing downstream reads the prob, so the format
    change is safe.)"""
    recon, tile_dirs = _assembly_prob_scenario(tmp_path)

    out_np = tmp_path / "out_np"
    AssemblyStage(recon_root=str(recon), output_dir=str(out_np),
                  stitch_mask=False, stitch_binary=False, stitch_prob=True,
                  prob_stitch_mode="average", prob_backend="numpy"
                  ).run(tile_dirs, n_gpus=0)

    out_z = tmp_path / "out_z"
    AssemblyStage(recon_root=str(recon), output_dir=str(out_z),
                  stitch_mask=False, stitch_binary=False, stitch_prob=True,
                  prob_stitch_mode="average", prob_backend="zarr"
                  ).run(tile_dirs, n_gpus=0)

    assert (out_np / "full_probabilities.npy").exists()
    assert (out_z / "full_probabilities.zarr").exists()
    assert not (out_z / "full_probabilities.npy").exists()  # zarr backend -> no .npy
    npy = np.load(str(out_np / "full_probabilities.npy"))
    zr = load_volume(out_z / "full_probabilities.zarr")
    assert zr.shape == npy.shape == (8, 4, 6)
    assert np.allclose(zr, npy)

"""Brick 4b items 2 + 3 — store CODEC (zstd-3 + shuffle) and METADATA hygiene
(consolidated metadata + nested `dimension_separator`).

Both are NEW-store-only defaults.  The gate in every test here is that an OLD
store (default lz4, flat separator, no `.zmetadata`) still opens and reads back
identically — the codec and the separator live in `.zarray`, and the reader must
never assume either.
"""
from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_largedata.config import (
    ALLOWED_COMPRESSORS,
    DEFAULT_COMPRESSOR,
)
from packages.epicycle_largedata.multiscale import (
    _make_compressor,
    convert_to_ome_zarr,
    open_multiscale,
)


def _write_tif(tmp_path, shape=(70, 200, 200), name="src.tif"):
    import tifffile

    rng = np.random.default_rng(7)
    # smooth-ish data so the codecs actually differ (pure noise is incompressible)
    a = np.cumsum(rng.integers(0, 40, size=shape, dtype=np.uint16), axis=0)
    a = (a % 4000).astype(np.uint16)
    p = tmp_path / name
    tifffile.imwrite(str(p), a)
    return p, a


# --------------------------------------------------------------------------- #
# 2) compressor
# --------------------------------------------------------------------------- #

def test_default_compressor_is_zstd():
    assert DEFAULT_COMPRESSOR == "zstd"
    assert set(ALLOWED_COMPRESSORS) == {"zstd", "lz4", "default"}


def test_make_compressor_zstd_is_blosc_zstd_clevel3_shuffle():
    from numcodecs import Blosc

    c = _make_compressor("zstd")
    assert isinstance(c, Blosc)
    assert c.cname == "zstd"
    assert c.clevel == 3
    assert c.shuffle == Blosc.SHUFFLE


def test_make_compressor_lz4_is_the_alternative():
    c = _make_compressor("lz4")
    assert c.cname == "lz4"


def test_make_compressor_default_is_zarrs_own_stock_codec():
    """"default" reproduces every pre-Brick-4b store: zarr's stock Blosc lz4-5."""
    import zarr.storage

    assert _make_compressor("default") is zarr.storage.default_compressor


def test_make_compressor_none_means_genuinely_uncompressed():
    """The footgun this test exists to pin: zarr treats an explicit
    `compressor=None` as NO codec, not as "choose the default".  Using None as a
    stand-in for the default silently writes a ~4x larger store."""
    assert _make_compressor(None) is None


def test_make_compressor_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        _make_compressor("brotli")


def test_convert_records_the_codec_in_zarray(tmp_path):
    import zarr

    src, _a = _write_tif(tmp_path)
    out = tmp_path / "zstd.zarr"
    man = convert_to_ome_zarr(src, out, max_levels=1)
    assert man["compressor"] == "zstd"
    z = zarr.open_group(str(out), mode="r")["0"]
    assert z.compressor.cname == "zstd"
    assert z.compressor.clevel == 3


def test_lz4_store_still_writes_and_reads_back(tmp_path):
    src, a = _write_tif(tmp_path)
    out = tmp_path / "lz4.zarr"
    convert_to_ome_zarr(src, out, compressor="lz4", max_levels=1)
    levels, _meta = open_multiscale(out)
    np.testing.assert_array_equal(np.asarray(levels[0]), a)


def test_a_legacy_zarr_default_store_still_opens(tmp_path):
    """`compressor="default"` reproduces every store written before Brick 4b."""
    src, a = _write_tif(tmp_path)
    out = tmp_path / "legacy.zarr"
    convert_to_ome_zarr(src, out, compressor="default", max_levels=1)
    levels, _meta = open_multiscale(out)
    np.testing.assert_array_equal(np.asarray(levels[0]), a)


def test_the_codec_never_changes_the_pixels(tmp_path):
    src, a = _write_tif(tmp_path)
    got = []
    for name in ("zstd", "lz4", "default"):
        out = tmp_path / f"px_{name}.zarr"
        convert_to_ome_zarr(src, out, compressor=name, max_levels=2)
        levels, _meta = open_multiscale(out)
        got.append([np.asarray(l) for l in levels])
    for ref, other in zip(got[0], got[1]):
        np.testing.assert_array_equal(ref, other)
    for ref, other in zip(got[0], got[2]):
        np.testing.assert_array_equal(ref, other)


def test_zstd_is_not_larger_than_lz4_or_the_legacy_default(tmp_path):
    import os

    src, _a = _write_tif(tmp_path)
    sizes = {}
    for name in ("zstd", "lz4", "default"):
        out = tmp_path / f"sz_{name}.zarr"
        convert_to_ome_zarr(src, out, compressor=name, max_levels=1)
        sizes[name] = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _d, fs in os.walk(out) for f in fs)
    assert sizes["zstd"] <= sizes["lz4"]
    assert sizes["zstd"] <= sizes["default"]


# --------------------------------------------------------------------------- #
# 3) metadata: consolidated + nested separator
# --------------------------------------------------------------------------- #

def test_new_store_is_consolidated(tmp_path):
    import zarr

    src, a = _write_tif(tmp_path)
    out = tmp_path / "cons.zarr"
    convert_to_ome_zarr(src, out, max_levels=2)
    assert (out / ".zmetadata").is_file()
    # and it OPENS through the consolidated path
    g = zarr.open_consolidated(str(out), mode="r")
    assert "multiscales" in g.attrs
    np.testing.assert_array_equal(np.asarray(g["0"]), a)


def test_consolidated_metadata_includes_the_final_provenance(tmp_path):
    """`epicycle_source` is written LAST (the interrupted-Prepare guard).  If the
    consolidation ran before it, the freshness probe would read a store with no
    provenance and silently re-prepare every time."""
    from packages.epicycle_largedata.multiscale import read_store_source

    src, _a = _write_tif(tmp_path)
    out = tmp_path / "cons2.zarr"
    convert_to_ome_zarr(src, out, max_levels=1)
    got = read_store_source(out)
    assert got is not None
    assert got["path"].endswith("src.tif")


def test_new_store_uses_the_nested_dimension_separator(tmp_path):
    import zarr

    src, _a = _write_tif(tmp_path)
    out = tmp_path / "nested.zarr"
    convert_to_ome_zarr(src, out, max_levels=1)
    z = zarr.open_group(str(out), mode="r")["0"]
    assert z._store._dimension_separator == "/" or (out / "0" / "0").is_dir()


def test_a_flat_legacy_store_still_opens(tmp_path):
    """Old stores are FLAT (`0/0.0.0`).  The reader must not assume nested."""
    import zarr

    a = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    out = tmp_path / "flat.zarr"
    root = zarr.open_group(str(out), mode="w")
    d = root.create_dataset("0", shape=a.shape, chunks=(4, 4, 4),
                            dtype="uint16", dimension_separator=".")
    d[:] = a
    root.attrs["multiscales"] = [{
        "version": "0.4", "name": "flat",
        "axes": [{"name": n, "type": "space"} for n in "zyx"],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
    }]
    levels, _meta = open_multiscale(out)
    np.testing.assert_array_equal(np.asarray(levels[0]), a)
    assert (out / "0" / "0.0.0").is_file()      # genuinely flat on disk

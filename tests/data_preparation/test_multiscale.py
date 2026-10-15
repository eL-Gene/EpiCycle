"""Tests for the TIFF -> multiscale OME-Zarr converter + lazy reader
(``packages.epicycle_largedata.multiscale``, Step-1 large-data Brick 1.2).

The keystone claims under test:

* **Pyramid geometry** — an **anisotropic, size-aware** mean-downsample pyramid:
  each ZYX axis halves *independently*, and only while the result stays
  ``>= min_size`` (a thin-Z stack keeps Z, still downsamples XY); **cubic
  chunks** (the axial-lag fix) and the source dtype preserved.
* **Byte-identity of level 0** — the finest level equals the *canonicalized*
  input exactly (CZYX for 4D, ZYX for 3D), so no data is lost in conversion.
* **Downsampled-level content** — a coarser level equals an independent
  ``rint(coarsen(np.mean, ..., trim_excess=True)).astype(dtype)`` reference
  (Brick 4b item 6 rounds instead of truncating).
* **Valid OME-NGFF v0.4 metadata** — a ``multiscales`` block a standard reader
  (and napari) can consume: axes names+types, one ``datasets`` entry per level,
  per-axis cumulative-factor scales (a downsampled axis increases, a clamped
  axis stays flat), and optional physical ``voxel_size`` + ``micrometer`` unit.
* **RAM boundedness (THE gate)** — conversion of a volume much larger than the
  ceiling never materializes a whole level.  Two paths, two bounds: the
  **memmap** (uncompressed) path is held to a few cubic chunks (tighter than the
  old vol/4); the **compressed** path is held to a small multiple of the Z-slab
  it must buffer.  Measured via ``tracemalloc`` peak (verified to capture
  numpy's C-level allocations in this env) plus a spy that
  ``dask.array.Array.compute`` is never invoked on a whole-volume-sized array.
* **Lazy reader** — ``open_multiscale`` returns lazy dask levels (level 0 first),
  from both a ``.zarr`` DirectoryStore and a ``.zip`` ZipStore, and rejects a
  malformed ``multiscales`` block with a clean ``ValueError``.
* **Axis confirmation** — an ambiguous 4D source is refused without ``image_axes``.
* **Overwrite guard** — refuses to clobber an existing store unless asked.
* **3D stays 3D** — a single-channel ZYX source gets no phantom channel axis.

Fixtures use ``tifffile.imwrite`` of a per-voxel-distinct ``arange`` array, so
any axis mis-order or off-by-one is caught by the byte-identity checks.
"""
from __future__ import annotations

import json
import os
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import tifffile

from packages.epicycle_largedata.multiscale import (
    convert_to_ome_zarr,
    open_multiscale,
    read_store_source,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _distinct(shape, dtype=np.int32):
    """An array whose every voxel is unique (arange), so any mis-order shows."""
    return np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)


def _write_tif(tmp_path, name, arr, axes=None):
    p = tmp_path / name
    kw = {}
    if axes is not None:
        kw["metadata"] = {"axes": axes}
    tifffile.imwrite(str(p), arr, **kw)
    return str(p)


def _write_tif_compressed(tmp_path, name, arr):
    """A DEFLATE-compressed TIFF -- NOT memory-mappable, so
    :func:`open_lazy` takes the dask/zarr (page-chunked) branch."""
    p = tmp_path / name
    tifffile.imwrite(str(p), arr, compression="deflate")
    return str(p)


def _open_group(store_path):
    """Open the written zarr group read-only (store-agnostic)."""
    import zarr

    from packages.epicycle_largedata.multiscale import _zarr_store

    store = _zarr_store(str(store_path), mode="r")
    return zarr.open_group(store=store, mode="r"), store


def _spatial(shape, axes):
    """The ZYX sub-tuple of a shape given canonical axes CZYX/ZYX."""
    return tuple(shape[-3:])


# ---------------------------------------------------------------------------
# 1) pyramid geometry
# ---------------------------------------------------------------------------

class TestConvertBuildsPyramid:
    @pytest.mark.parametrize(
        "shape, axes_meta, canon_axes",
        [
            ((16, 96, 96), None, "ZYX"),          # 3D single-channel
            ((2, 16, 96, 96), "CZYX", "CZYX"),    # 4D channel-first
        ],
    )
    def test_pyramid_shapes_chunks_dtype(
        self, tmp_path, shape, axes_meta, canon_axes
    ):
        arr = _distinct(shape, dtype=np.int32)
        src = _write_tif(tmp_path, "src.tif", arr, axes=axes_meta)
        out = tmp_path / "out.zarr"

        chunk, ds, min_size = 32, 2, 48
        man = convert_to_ome_zarr(
            src, out, chunk=chunk, downsample=ds, min_size=min_size
        )

        # canonical axes + dtype preserved
        assert man["axes"] == canon_axes
        assert man["dtype"] == "int32"
        assert man["downsample"] == ds
        # Brick 4b: level 0's chunks are CLAMPED to the level-0 extent, so a
        # requested edge larger than an axis (here Z=16 < 32) is clipped -- zarr
        # would otherwise store a padded chunk and the progressive reader would
        # fetch more than the level holds.
        l0 = tuple(man["level_shapes"][0])
        assert man["chunks"] == tuple(
            min(c, s) for c, s in zip(
                ((1, chunk, chunk, chunk) if canon_axes == "CZYX"
                 else (chunk, chunk, chunk)), l0))

        shapes = [tuple(s) for s in man["level_shapes"]]
        # level 0 is the full canonical shape
        expected_l0 = shape if canon_axes == "ZYX" else shape
        assert shapes[0] == expected_l0

        # anisotropic per-level rule: each ZYX axis halves iff the result stays
        # >= min_size, else it is clamped (unchanged); channel (if any) constant
        for prev, nxt in zip(shapes, shapes[1:]):
            pz, nz = _spatial(prev, canon_axes), _spatial(nxt, canon_axes)
            for a in range(3):
                if pz[a] // ds >= min_size:
                    assert nz[a] == pz[a] // ds   # axis qualified -> halved
                else:
                    assert nz[a] == pz[a]         # axis clamped -> unchanged
            if canon_axes == "CZYX":
                assert nxt[0] == prev[0]  # channel count unchanged

        # stop rule: NO axis can be halved further without dropping below
        # min_size (i.e. the pyramid is complete), unless max_levels capped
        # (which we didn't set here)
        last_zyx = _spatial(shapes[-1], canon_axes)
        assert max(d // ds for d in last_zyx) < min_size

        assert man["n_levels"] == len(shapes)
        assert man["n_levels"] >= 2  # this fixture halves at least once

        # on-disk chunks are cubic (clamped to dim); channel chunked at 1
        group, store = _open_group(out)
        try:
            for i in range(man["n_levels"]):
                z = group[str(i)]
                lshape = shapes[i]
                if canon_axes == "CZYX":
                    assert z.chunks[0] == 1
                    for axsize, ch in zip(lshape[1:], z.chunks[1:]):
                        assert ch == min(chunk, axsize)
                else:
                    for axsize, ch in zip(lshape, z.chunks):
                        assert ch == min(chunk, axsize)
                assert str(z.dtype) == "int32"
        finally:
            if hasattr(store, "close"):
                store.close()

    def test_max_levels_caps_pyramid(self, tmp_path):
        arr = _distinct((32, 128, 128), dtype=np.uint16)
        src = _write_tif(tmp_path, "cap.tif", arr)
        out = tmp_path / "cap.zarr"
        man = convert_to_ome_zarr(
            src, out, chunk=16, downsample=2, min_size=8, max_levels=2
        )
        assert man["n_levels"] == 2


# ---------------------------------------------------------------------------
# 2) byte-identity of level 0
# ---------------------------------------------------------------------------

class TestLevel0ByteIdentical:
    def test_3d_level0_identical(self, tmp_path):
        arr = _distinct((12, 40, 40), dtype=np.int32)
        src = _write_tif(tmp_path, "z.tif", arr)
        out = tmp_path / "z.zarr"
        convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        levels, _meta = open_multiscale(out)
        got = levels[0].compute()
        assert got.shape == arr.shape
        assert np.array_equal(got, arr)

    def test_4d_czyx_level0_identical(self, tmp_path):
        arr = _distinct((2, 12, 40, 40), dtype=np.int32)
        src = _write_tif(tmp_path, "czyx.tif", arr, axes="CZYX")
        out = tmp_path / "czyx.zarr"
        convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        levels, _meta = open_multiscale(out)
        got = levels[0].compute()
        assert got.shape == (2, 12, 40, 40)
        assert np.array_equal(got, arr)  # CZYX canonicalization is a no-op

    def test_4d_zcyx_canonicalized_to_czyx(self, tmp_path):
        """A ZCYX on-disk source (image_axes override) must land as CZYX in the
        store — level 0 equals moveaxis(src, 1, 0)."""
        zcyx = _distinct((12, 2, 40, 40), dtype=np.int32)   # (Z, C, Y, X)
        src = _write_tif(tmp_path, "zcyx.tif", zcyx)
        out = tmp_path / "zcyx.zarr"
        man = convert_to_ome_zarr(
            src, out, chunk=16, downsample=2, min_size=8, image_axes="ZCYX"
        )
        assert man["axes"] == "CZYX"
        assert tuple(man["level_shapes"][0]) == (2, 12, 40, 40)

        levels, _meta = open_multiscale(out)
        assert np.array_equal(levels[0].compute(), np.moveaxis(zcyx, 1, 0))


# ---------------------------------------------------------------------------
# 3) OME-NGFF metadata validity
# ---------------------------------------------------------------------------

class TestOmeNgffMetadata:
    def _read_zattrs_file(self, out):
        """For a DirectoryStore the multiscales live literally in ``.zattrs``."""
        return json.loads((Path(out) / ".zattrs").read_text())

    def test_czyx_metadata_shape(self, tmp_path):
        arr = _distinct((2, 16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "m.tif", arr, axes="CZYX")
        out = tmp_path / "m.zarr"
        man = convert_to_ome_zarr(src, out, chunk=32, downsample=2, min_size=48)

        zattrs = self._read_zattrs_file(out)
        assert "multiscales" in zattrs
        ms = zattrs["multiscales"]
        assert isinstance(ms, list) and len(ms) == 1
        block = ms[0]
        assert block["version"] == "0.4"

        # axes: names + types, channel first
        axes = block["axes"]
        assert [a["name"] for a in axes] == ["c", "z", "y", "x"]
        assert [a["type"] for a in axes] == ["channel", "space", "space", "space"]

        # one dataset per level, paths "0".."N-1"
        dsets = block["datasets"]
        assert len(dsets) == man["n_levels"]
        assert [d["path"] for d in dsets] == [str(i) for i in range(man["n_levels"])]

        # per-level scale: channel stays 1; each ZYX scale = the per-axis
        # CUMULATIVE downsample factor (anisotropic -> a clamped axis stays flat,
        # a downsampled axis increases), so scales are non-DEcreasing per axis
        scales = [d["coordinateTransformations"][0]["scale"] for d in dsets]
        for s in scales:
            assert len(s) == 4
            assert s[0] == 1
        for prev, nxt in zip(scales, scales[1:]):
            for a in range(1, 4):
                assert nxt[a] >= prev[a]     # non-decreasing (clamped = flat)
        # scale[axis] == level0_size / level_size (the cumulative factor); with
        # voxel_size=None the scale is in dimensionless voxel units
        l0 = _spatial(tuple(man["level_shapes"][0]), "CZYX")
        for lvl, s in enumerate(scales):
            lvl_zyx = _spatial(tuple(man["level_shapes"][lvl]), "CZYX")
            for a in range(3):
                assert s[a + 1] == float(l0[a] // lvl_zyx[a])
        # the pyramid actually downsamples SOMETHING (here Y/X halve; Z clamps)
        assert any(scales[-1][a] > scales[0][a] for a in range(1, 4))

    def test_zyx_metadata_shape(self, tmp_path):
        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "m3.tif", arr)
        out = tmp_path / "m3.zarr"
        man = convert_to_ome_zarr(src, out, chunk=32, downsample=2, min_size=48)

        zattrs = self._read_zattrs_file(out)
        block = zattrs["multiscales"][0]
        axes = block["axes"]
        assert [a["name"] for a in axes] == ["z", "y", "x"]
        assert all(a["type"] == "space" for a in axes)
        dsets = block["datasets"]
        assert len(dsets) == man["n_levels"]
        for d in dsets:
            assert len(d["coordinateTransformations"][0]["scale"]) == 3

    def test_result_manifest_written_alongside(self, tmp_path):
        arr = _distinct((12, 40, 40), dtype=np.uint16)
        src = _write_tif(tmp_path, "r.tif", arr)
        out = tmp_path / "r.zarr"
        man = convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        side = Path(out).parent / "result_manifest.json"
        assert side.exists()
        on_disk = json.loads(side.read_text())
        assert on_disk["axes"] == man["axes"]
        assert on_disk["n_levels"] == man["n_levels"]
        assert on_disk["dtype"] == man["dtype"]

    def test_manifest_records_source_provenance(self, tmp_path):
        """The manifest (returned + sidecar) must record ``source_path`` (the
        abspath of the input), ``source_mtime`` (its getmtime at convert time)
        and ``source_size`` (its getsize) — the provenance the Step-1 "already
        prepared?" / Smart-Load check reads to tell a fresh store from a stale
        one (an mtime-preserving replacement still changes the size)."""
        arr = _distinct((10, 40, 40), dtype=np.uint16)
        src = _write_tif(tmp_path, "prov.tif", arr)
        out = tmp_path / "prov.zarr"
        man = convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        # Returned manifest carries correct provenance.
        assert man["source_path"] == os.path.abspath(src)
        assert man["source_mtime"] == pytest.approx(os.path.getmtime(src))
        assert man["source_size"] == os.path.getsize(src)

        # And it round-trips to the on-disk sidecar.
        side = json.loads(
            (Path(out).parent / "result_manifest.json").read_text())
        assert side["source_path"] == os.path.abspath(src)
        assert side["source_mtime"] == pytest.approx(os.path.getmtime(src))
        assert side["source_size"] == os.path.getsize(src)

    def test_store_zattrs_records_per_store_source_provenance(self, tmp_path):
        """PER-STORE provenance: each store's OWN ``.zattrs`` carries an
        ``epicycle_source`` block ``{"path", "source_mtime"}`` so multiple
        stores sharing one parent dir are each self-describing (the Smart-Load
        per-source fix — the shared ``result_manifest.json`` is last-writer-
        wins).  ``open_multiscale`` surfaces it as ``meta["source"]``."""
        arr = _distinct((10, 40, 40), dtype=np.uint16)
        src = _write_tif(tmp_path, "prov2.tif", arr)
        out = tmp_path / "prov2.zarr"
        convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        # The store's OWN .zattrs carries per-store provenance.
        zattrs = self._read_zattrs_file(out)
        assert zattrs["epicycle_source"]["path"] == os.path.abspath(src)
        assert zattrs["epicycle_source"]["source_mtime"] == pytest.approx(
            os.path.getmtime(src))
        assert zattrs["epicycle_source"]["source_size"] == os.path.getsize(src)
        # schema v2: the TRUE on-disk order (a 3D source is ZYX) + voxel (unset
        # here) live in the per-store attrs so a Prepare-only source's real
        # order is durable — the store's NGFF `multiscales.axes` is CANONICAL.
        assert zattrs["epicycle_source"]["source_axes"] == "ZYX"
        assert zattrs["epicycle_source"]["voxel_size"] is None

        # open_multiscale surfaces it as meta["source"].
        _levels, meta = open_multiscale(out)
        assert meta["source"]["path"] == os.path.abspath(src)
        assert meta["source"]["source_mtime"] == pytest.approx(
            os.path.getmtime(src))
        assert meta["source"]["source_size"] == os.path.getsize(src)
        assert meta["source"]["source_axes"] == "ZYX"
        assert meta["source"]["voxel_size"] is None

    def test_store_records_source_axes_and_voxel_for_4d(self, tmp_path):
        """A 4D ZCYX source records its DISTINCT on-disk order (ZCYX, NOT the
        canonical CZYX the store's NGFF axes carry) + the voxel param (A2a)."""
        arr = _distinct((8, 2, 24, 24), dtype=np.uint16)   # Z C Y X on disk
        src = _write_tif(tmp_path, "zcyx.tif", arr)
        out = tmp_path / "zcyx.zarr"
        convert_to_ome_zarr(
            src, out, chunk=8, downsample=2, min_size=4,
            image_axes="ZCYX", voxel_size=(2.0, 0.5, 0.5))

        prov = read_store_source(out)
        assert prov["source_axes"] == "ZCYX"        # honest on-disk order
        assert prov["voxel_size"] == [2.0, 0.5, 0.5]

        # The store's NGFF axes remain CANONICAL (CZYX) — the two are distinct.
        _levels, meta = open_multiscale(out)
        canon = [a["name"] for a in meta["axes"]]
        assert canon == ["c", "z", "y", "x"]

    def test_read_store_source_reads_attrs_and_never_raises(self, tmp_path):
        """``read_store_source`` returns the per-store ``epicycle_source`` dict
        via a LIGHT attrs-only read, and NEVER raises — a missing store or a
        corrupt/attrs-less store returns ``None``."""
        arr = _distinct((10, 40, 40), dtype=np.uint16)
        src = _write_tif(tmp_path, "rss.tif", arr)
        out = tmp_path / "rss.zarr"
        convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        prov = read_store_source(out)
        assert isinstance(prov, dict)
        assert prov["path"] == os.path.abspath(src)
        assert prov["source_mtime"] == pytest.approx(os.path.getmtime(src))
        assert prov["source_size"] == os.path.getsize(src)

        # Missing store → None (never raises).
        assert read_store_source(tmp_path / "does_not_exist.zarr") is None

        # Corrupt attrs → None (never raises).
        (Path(out) / ".zattrs").write_text("{ not valid json ")
        assert read_store_source(out) is None


# ---------------------------------------------------------------------------
# 4) THE peak-RAM gate
# ---------------------------------------------------------------------------

def test_peak_ram_below_volume(tmp_path, monkeypatch):
    """Convert a volume much larger than the ceiling; assert conversion never
    materializes a whole level.

    Two independent, non-vibes guards:

    * ``tracemalloc`` peak < a TIGHT ceiling: a small multiple of the finest-level
      cubic-chunk working set (``chunk**3 * itemsize``) plus a fixed overhead
      headroom -- deliberately far below the old ``volume/4`` gate.  This matters
      under the anisotropic pyramid: a thin-Z stack keeps Z, so level 1 here is
      ``(48, 512, 512)`` = 25 MB = *exactly* ``volume/4`` -- the old gate could
      not tell a whole-level materialization from the legit stream.  The tight
      ceiling (~16 MB) sits above the measured streaming peak (~5 MB, ~10x
      ``chunk**3``) with >3x margin yet below level 1 (25 MB), so materializing
      ANY level >= level 1 fails.  In this env numpy's data allocations ARE
      reported to tracemalloc (numpy >= 1.22 routes ``PyDataMem_*`` through it --
      verified: a 32 MB ``np.zeros`` traces as ~33 MB), so it is not a
      Python-object-overhead artifact.
    * a spy on ``dask.array.Array.compute`` -- asserts it is never invoked on an
      array whose element count equals the full volume (the exact ".compute() a
      whole level" failure the gate targets).  ``da.to_zarr`` streams via
      ``da.store`` and does not route through ``Array.compute``, so the legit
      path never trips it.

    WARMUP (why): the FIRST conversion in a process pays a large one-time cost
    (lazy ``import dask``/``zarr``, dask graph-machinery caches, code-path JIT)
    that ``tracemalloc`` counts if it lands inside the timed region -- ~50 MB of
    it, none of which is volume data.  That is not what this gate measures.  So
    heavy imports + the full convert code path are warmed on a *tiny* throwaway
    volume BEFORE the timed region; the measured peak is then the genuine
    *streaming* allocation of the big convert (~4-5 MB here, a >5x margin under
    the ceiling).  A whole-level materialization would still add ~100 MB and
    fail loudly.
    """
    import dask          # noqa: F401  (warm heavy imports before the timed region)
    import dask.array as da
    import zarr          # noqa: F401

    Z, Y, X = 48, 1024, 1024
    chunk = 64
    itemsize = np.dtype(np.uint16).itemsize
    vol_elems = Z * Y * X
    vol_bytes = vol_elems * itemsize                       # ~100 MB
    chunk_ws = chunk ** 3 * itemsize                       # one cubic chunk, ~0.5 MB
    # TIGHT ceiling (FIX 4).  Genuine streaming peak measured ~5 MB (~10x a cubic
    # chunk: a handful of in-flight chunks + the dask task graph for the ~256-chunk
    # level 0).  16 cubic chunks of headroom for the streaming set + a fixed 8 MB
    # for graph/writer bookkeeping (which scales with chunk COUNT, not chunk size)
    # -> ~16 MB: >3x the measured peak (robust, not flaky) yet below level 1
    # (48*512*512*2 = 25 MB = old vol/4), so materializing any whole level >=
    # level 1 fails.  (The coarsest level, (48,256,256)=6 MB, is the size of the
    # legit working set and is intentionally NOT separable from it.)
    ceiling = 16 * chunk_ws + 8 * 1024 * 1024              # ~16.4 MB

    arr = np.arange(vol_elems, dtype=np.uint16).reshape(Z, Y, X)
    src = _write_tif(tmp_path, "big.tif", arr)             # uncompressed -> memmap
    del arr
    out = tmp_path / "big.zarr"

    # warm the whole convert code path (imports, proxy, coarsen, to_zarr,
    # synchronous scheduler machinery) on a negligible volume -- excludes the
    # one-time overhead from the measurement below without touching the big convert.
    warm = _write_tif(tmp_path, "warm.tif", _distinct((4, 8, 8), np.uint16))
    convert_to_ome_zarr(warm, tmp_path / "warm.zarr", chunk=4, downsample=2, min_size=2)

    seen_sizes = []
    orig_compute = da.Array.compute

    def _spy_compute(self, *a, **k):
        seen_sizes.append(int(self.size))
        return orig_compute(self, *a, **k)

    monkeypatch.setattr(da.Array, "compute", _spy_compute, raising=True)

    tracemalloc.start()
    try:
        convert_to_ome_zarr(src, out, chunk=chunk, downsample=2, min_size=256)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < ceiling, (
        f"peak traced RAM {peak/1e6:.1f} MB is not < tight ceiling "
        f"{ceiling/1e6:.1f} MB (cubic chunk {chunk_ws/1e6:.2f} MB, volume "
        f"{vol_bytes/1e6:.1f} MB) -- a whole level was likely materialized"
    )
    assert (max(seen_sizes) if seen_sizes else 0) < vol_elems, (
        f"Array.compute() was called on a whole-volume-sized array: "
        f"max size {max(seen_sizes)} >= volume {vol_elems}"
    )


# ---------------------------------------------------------------------------
# 5) lazy reader
# ---------------------------------------------------------------------------

def test_open_multiscale_levels_are_lazy(tmp_path):
    import dask.array as da

    arr = _distinct((16, 96, 96), dtype=np.uint16)
    src = _write_tif(tmp_path, "lz.tif", arr)
    out = tmp_path / "lz.zarr"
    man = convert_to_ome_zarr(src, out, chunk=32, downsample=2, min_size=24)

    levels, meta = open_multiscale(out)
    assert isinstance(levels, list)
    assert len(levels) == man["n_levels"]
    assert meta["n_levels"] == man["n_levels"]

    for lvl, expected in zip(levels, man["level_shapes"]):
        assert isinstance(lvl, da.Array)   # lazy: NOT a numpy array
        assert hasattr(lvl, "compute")
        assert tuple(lvl.shape) == tuple(expected)

    # level 0 is finest (largest); descending resolution order
    zyx = [tuple(s)[-3:] for s in man["level_shapes"]]
    assert zyx == sorted(zyx, key=lambda t: -max(t))


# ---------------------------------------------------------------------------
# 6) ZipStore round-trip
# ---------------------------------------------------------------------------

def test_zip_store_roundtrip(tmp_path):
    arr = _distinct((12, 40, 40), dtype=np.int32)
    src = _write_tif(tmp_path, "zip_src.tif", arr)
    out = tmp_path / "store.zip"

    man = convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)
    assert Path(out).exists() and Path(out).is_file()   # a single .zip file

    levels, meta = open_multiscale(out)
    assert meta["n_levels"] == man["n_levels"]
    assert np.array_equal(levels[0].compute(), arr)     # base level intact


# ---------------------------------------------------------------------------
# 7) overwrite guard
# ---------------------------------------------------------------------------

class TestOverwriteGuard:
    def test_refuses_existing_dir_store(self, tmp_path):
        arr = _distinct((8, 32, 32), dtype=np.uint16)
        src = _write_tif(tmp_path, "o.tif", arr)
        out = tmp_path / "o.zarr"
        convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        with pytest.raises(FileExistsError):
            convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        # overwrite=True succeeds
        man = convert_to_ome_zarr(
            src, out, chunk=16, downsample=2, min_size=8, overwrite=True
        )
        assert man["n_levels"] >= 1
        levels, _ = open_multiscale(out)
        assert np.array_equal(levels[0].compute(), arr)

    def test_refuses_existing_zip_store(self, tmp_path):
        arr = _distinct((8, 32, 32), dtype=np.uint16)
        src = _write_tif(tmp_path, "oz.tif", arr)
        out = tmp_path / "oz.zip"
        convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)
        with pytest.raises(FileExistsError):
            convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)
        convert_to_ome_zarr(
            src, out, chunk=16, downsample=2, min_size=8, overwrite=True
        )


# ---------------------------------------------------------------------------
# 8) 3D stays 3D
# ---------------------------------------------------------------------------

def test_3d_source_stays_3d(tmp_path):
    arr = _distinct((14, 50, 50), dtype=np.uint16)
    src = _write_tif(tmp_path, "d3.tif", arr)
    out = tmp_path / "d3.zarr"
    man = convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=12)

    assert man["axes"] == "ZYX"
    assert len(man["chunks"]) == 3
    for s in man["level_shapes"]:
        assert len(s) == 3   # never a phantom leading channel axis

    levels, meta = open_multiscale(out)
    assert levels[0].ndim == 3
    assert [a["name"] for a in meta["axes"]] == ["z", "y", "x"]
    assert np.array_equal(levels[0].compute(), arr)


# ---------------------------------------------------------------------------
# 9) anisotropic, size-aware pyramid (FIX 1)
# ---------------------------------------------------------------------------

class TestAnisotropicPyramid:
    def test_thin_z_keeps_z_downsamples_xy(self, tmp_path):
        """A thin-Z wide-XY stack: Z is already below min_size so it must NOT
        downsample (it would vanish); XY still halves to min_size.  The old
        isotropic policy stopped the whole pyramid the moment Z couldn't halve,
        giving such a stack no overview at all."""
        arr = _distinct((4, 512, 512), dtype=np.uint16)
        src = _write_tif(tmp_path, "thin.tif", arr)
        out = tmp_path / "thin.zarr"
        man = convert_to_ome_zarr(src, out, chunk=64, downsample=2, min_size=256)

        shapes = [tuple(s) for s in man["level_shapes"]]
        assert shapes == [(4, 512, 512), (4, 256, 256)]
        assert all(s[0] == 4 for s in shapes)           # Z constant everywhere
        assert [s[1] for s in shapes] == [512, 256]     # XY halves once

        _levels, meta = open_multiscale(out)
        # scales are the per-axis CUMULATIVE factor: Z clamped -> 1, XY -> 1,2
        assert meta["scales"] == [[1.0, 1.0, 1.0], [1.0, 2.0, 2.0]]

    def test_single_z_plane_still_builds_xy_overview(self, tmp_path):
        """A single-Z-plane source still gets an XY pyramid (NOT n_levels=1)."""
        arr = _distinct((1, 512, 512), dtype=np.uint16)
        src = _write_tif(tmp_path, "single.tif", arr)
        out = tmp_path / "single.zarr"
        man = convert_to_ome_zarr(src, out, chunk=64, downsample=2, min_size=256)

        assert [tuple(s) for s in man["level_shapes"]] == [
            (1, 512, 512), (1, 256, 256)
        ]
        assert man["n_levels"] == 2

    def test_wound_like_z_clamped_xy_pyramid(self, tmp_path):
        """The wound-like (thin-ish Z, wide XY) case: Z stays, XY halves twice
        to min_size (cumulative XY factors 1, 2, 4)."""
        arr = _distinct((7, 1024, 1024), dtype=np.uint16)  # small-Z stand-in
        src = _write_tif(tmp_path, "wound.tif", arr)
        out = tmp_path / "wound.zarr"
        man = convert_to_ome_zarr(src, out, chunk=64, downsample=2, min_size=256)

        assert [tuple(s) for s in man["level_shapes"]] == [
            (7, 1024, 1024), (7, 512, 512), (7, 256, 256)
        ]
        _levels, meta = open_multiscale(out)
        assert meta["scales"] == [
            [1.0, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 4.0, 4.0]
        ]


# ---------------------------------------------------------------------------
# 10) downsampled-level content (an independent coarsen reference)
# ---------------------------------------------------------------------------

def test_downsampled_level1_content(tmp_path):
    """Level 1 equals an INDEPENDENT ``coarsen(np.mean).astype(dtype)`` of the
    canonical source -- the byte-identity tests only ever checked level 0."""
    import dask.array as da

    # A RANDOM field, not the arange `_distinct` this used to use: every 2x2x2
    # block mean of an arange is already an exact integer, so rint == truncate
    # everywhere and the test could not tell them apart (measured: 0 differing
    # voxels).  This fixture differs in ~3.6k voxels, so a truncation regression
    # actually fails it.
    rng = np.random.default_rng(5)
    arr = rng.integers(0, 30000, size=(16, 64, 64)).astype(np.int32)
    src = _write_tif(tmp_path, "content.tif", arr)
    out = tmp_path / "content.zarr"
    man = convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)
    assert man["n_levels"] >= 2
    assert tuple(man["level_shapes"][1]) == (8, 32, 32)  # all three axes halve

    levels, _meta = open_multiscale(out)
    got = levels[1].compute()

    # Brick 4b item 6: mean then ROUND (np.rint) then cast -- a bare .astype
    # truncates and biases every coarse level ~0.5 LSB dim.
    ref = (
        np.rint(da.coarsen(np.mean, da.from_array(arr, chunks=arr.shape),
                           {0: 2, 1: 2, 2: 2}, trim_excess=True))
        .astype(np.int32)
        .compute()
    )
    assert got.shape == ref.shape == (8, 32, 32)
    assert np.array_equal(got, ref)     # mean -> rint -> int, bit-for-bit


# ---------------------------------------------------------------------------
# 11) per-axis NGFF scales + physical voxel spacing (FIX 2)
# ---------------------------------------------------------------------------

class TestVoxelSize:
    def test_physical_scales_and_micrometer_unit(self, tmp_path):
        arr = _distinct((2, 16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "vox.tif", arr, axes="CZYX")
        out = tmp_path / "vox.zarr"
        vz, vy, vx = 3.0, 0.65, 0.65
        man = convert_to_ome_zarr(
            src, out, chunk=32, downsample=2, min_size=48,
            voxel_size=(vz, vy, vx),
        )
        # spacing recorded in the manifest (+ sidecar)
        assert tuple(man["voxel_size"]) == (vz, vy, vx)
        side = json.loads((Path(out).parent / "result_manifest.json").read_text())
        assert list(side["voxel_size"]) == [vz, vy, vx]

        _levels, meta = open_multiscale(out)
        scales = meta["scales"]
        # 2 levels: Z clamps (16//2=8 < 48), YX halve once
        assert scales[0] == pytest.approx([1.0, vz, vy, vx])
        assert scales[1] == pytest.approx([1.0, vz, vy * 2, vx * 2])

        # SPACE axes carry the micrometer unit; the channel axis never does
        axes = meta["axes"]
        assert axes[0] == {"name": "c", "type": "channel"}
        for a in axes[1:]:
            assert a["type"] == "space" and a["unit"] == "micrometer"

    def test_no_voxel_size_has_no_unit_and_none_manifest(self, tmp_path):
        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "novox.tif", arr)
        out = tmp_path / "novox.zarr"
        man = convert_to_ome_zarr(src, out, chunk=32, downsample=2, min_size=48)

        assert man["voxel_size"] is None
        _levels, meta = open_multiscale(out)
        for a in meta["axes"]:
            assert "unit" not in a

    def test_voxel_size_validation(self, tmp_path):
        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "badvox.tif", arr)
        out = tmp_path / "badvox.zarr"
        with pytest.raises(ValueError):
            convert_to_ome_zarr(src, out, chunk=32, voxel_size=(1.0, 1.0))
        with pytest.raises(ValueError):
            convert_to_ome_zarr(src, out, chunk=32, voxel_size=(1.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 12) compressed-source peak RAM: a Z-slab bound, NOT the vol/4 gate (FIX 3)
# ---------------------------------------------------------------------------

def test_peak_ram_compressed_source_bounded(tmp_path):
    """A DEFLATE-compressed source cannot be memmapped, so it is opened as a
    dask array chunked by TIFF *pages* (thin-Z ``(1, Y, X)``) and rechunked
    cubic.  Assembling cubic chunks out of thin pages must buffer a **Z-slab**
    (~``chunk * Y * X``) -- bounded (flat in total Z) but larger than the memmap
    path's few-cubic-chunk working set.  So the tight ``vol/4`` gate does NOT
    apply here; the honest bound is a small multiple of the slab.
    """
    import dask          # noqa: F401  (warm heavy imports before the timed region)
    import dask.array as da  # noqa: F401
    import zarr          # noqa: F401

    Z, Y, X = 384, 256, 256
    chunk = 64
    itemsize = np.dtype(np.uint16).itemsize
    vol_bytes = Z * Y * X * itemsize                   # ~50 MB
    slab_bytes = chunk * Y * X * itemsize              # ~8.4 MB (Z-independent)
    # slab << volume BY CONSTRUCTION (chunk << Z): a genuinely tight ceiling,
    # not merely "< the whole thing".
    assert slab_bytes * 4 < vol_bytes                  # chunk/Z < 1/4

    arr = np.arange(Z * Y * X, dtype=np.uint16).reshape(Z, Y, X)
    src = _write_tif_compressed(tmp_path, "cmp.tif", arr)
    del arr
    out = tmp_path / "cmp.zarr"

    # warm the whole (compressed) convert code path off the timed region -- same
    # rationale as test_peak_ram_below_volume.
    warm = _write_tif_compressed(tmp_path, "cmpwarm.tif",
                                 _distinct((8, 16, 16), np.uint16))
    convert_to_ome_zarr(warm, tmp_path / "cmpwarm.zarr",
                        chunk=8, downsample=2, min_size=4)

    tracemalloc.start()
    try:
        convert_to_ome_zarr(src, out, chunk=chunk, downsample=2, min_size=128)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # measured peak here ~20 MB ~= 2.5 slabs.  Ceiling = 4 slabs (~33 MB):
    # >1.5x the measured peak (robust) yet still well BELOW the whole volume
    # (~50 MB), so passing proves boundedness rather than "< the whole thing".
    ceiling = 4 * slab_bytes
    assert peak < ceiling, (
        f"compressed-path peak {peak/1e6:.1f} MB exceeded {ceiling/1e6:.1f} MB "
        f"(4x the {slab_bytes/1e6:.1f} MB Z-slab; volume {vol_bytes/1e6:.1f} MB)"
    )
    # ...and the peak is deliberately NOT below vol/4 (the tight memmap gate):
    # the compressed slab genuinely exceeds vol/4, which is the whole reason this
    # path gets a looser, slab-based bound.  (Env-pinned lower bound; a dask that
    # buffered dramatically less would be a real improvement worth revisiting.)
    assert peak > vol_bytes / 4


# ---------------------------------------------------------------------------
# 13) lazy-proxy invariant (FIX 5) + ambiguous-axis confirmation (FIX 5)
# ---------------------------------------------------------------------------

def test_lazyproxy_never_exposes_copy():
    """``da.from_array`` runs ``x.copy()`` on any arraylike exposing ``.copy``,
    and ``memmap.copy()`` reads the whole volume -- the proxy must NOT expose it."""
    from packages.epicycle_largedata.multiscale import _LazyReadProxy

    proxy = _LazyReadProxy(np.zeros((2, 2, 2)))
    assert not hasattr(proxy, "copy")


class TestAxisConfirmation:
    def test_ambiguous_4d_requires_image_axes(self, tmp_path):
        """A 4D TIFF with no clean axes metadata (tifffile writes 'QQYX') ->
        resolve_axes is not confident -> convert refuses without image_axes."""
        arr = _distinct((2, 12, 40, 40), dtype=np.int32)
        src = _write_tif(tmp_path, "ambig.tif", arr)   # NO axes metadata
        out = tmp_path / "ambig.zarr"
        with pytest.raises(ValueError, match="image_axes"):
            convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        # confirming the order lets it proceed
        man = convert_to_ome_zarr(
            src, out, chunk=16, downsample=2, min_size=8, image_axes="CZYX"
        )
        assert man["axes"] == "CZYX"

    def test_confident_4d_proceeds_without_image_axes(self, tmp_path):
        arr = _distinct((2, 12, 40, 40), dtype=np.int32)
        src = _write_tif(tmp_path, "conf.tif", arr, axes="CZYX")  # clean metadata
        out = tmp_path / "conf.zarr"
        man = convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)
        assert man["axes"] == "CZYX"

    def test_3d_proceeds_without_image_axes(self, tmp_path):
        arr = _distinct((12, 40, 40), dtype=np.int32)   # unambiguous ZYX
        src = _write_tif(tmp_path, "d3ok.tif", arr)
        out = tmp_path / "d3ok.zarr"
        man = convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)
        assert man["axes"] == "ZYX"

    def test_image_axes_zyx_on_4d_source_raises(self, tmp_path):
        """image_axes='ZYX' is caller misuse for a 4D source (must be CZYX/ZCYX);
        refuse it rather than silently build a malformed 5D array."""
        arr = _distinct((2, 12, 40, 40), dtype=np.int32)
        src = _write_tif(tmp_path, "misuse.tif", arr, axes="CZYX")
        out = tmp_path / "misuse.zarr"
        with pytest.raises(ValueError, match="image_axes must be"):
            convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8,
                                image_axes="ZYX")

    def test_min_size_below_one_raises(self, tmp_path):
        arr = _distinct((12, 40, 40), dtype=np.int32)
        src = _write_tif(tmp_path, "ms0.tif", arr)
        out = tmp_path / "ms0.zarr"
        with pytest.raises(ValueError, match="min_size"):
            convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=0)


# ---------------------------------------------------------------------------
# 14) reader rejects a malformed multiscales block (FIX 6)
# ---------------------------------------------------------------------------

class TestMalformedMultiscales:
    def _write_bad_store(self, tmp_path, name, multiscales):
        import zarr

        from packages.epicycle_largedata.multiscale import _zarr_store

        out = tmp_path / name
        store = _zarr_store(str(out), mode="w")
        root = zarr.open_group(store=store, mode="w")
        root.attrs["multiscales"] = multiscales
        if hasattr(store, "close"):
            store.close()
        return out

    def test_missing_datasets_raises_valueerror(self, tmp_path):
        out = self._write_bad_store(
            tmp_path, "bad1.zarr", [{"version": "0.4", "name": "bad"}]
        )
        with pytest.raises(ValueError, match="malformed"):
            open_multiscale(out)

    def test_missing_scale_transform_raises_valueerror(self, tmp_path):
        out = self._write_bad_store(
            tmp_path, "bad2.zarr",
            [{"version": "0.4", "datasets": [{"path": "0"}]}],
        )
        with pytest.raises(ValueError, match="malformed"):
            open_multiscale(out)

    def test_absent_multiscales_raises_valueerror(self, tmp_path):
        import zarr

        from packages.epicycle_largedata.multiscale import _zarr_store

        out = tmp_path / "empty.zarr"
        store = _zarr_store(str(out), mode="w")
        zarr.open_group(store=store, mode="w")  # no 'multiscales' attr at all
        if hasattr(store, "close"):
            store.close()
        with pytest.raises(ValueError, match="multiscales"):
            open_multiscale(out)


# ---------------------------------------------------------------------------
# 15) write-phase optimizations (OPT 1 read-once + OPT 2 bounded parallelism)
# ---------------------------------------------------------------------------

def _l0_chunk_count(level0_shape, chunk):
    """Number of cubic chunks in level 0 = product of ceil(dim/chunk) over the
    SPATIAL (ZYX) axes (the channel axis is chunked at 1, one chunk per channel;
    for a 3D ZYX source there is no channel axis)."""
    import math

    zyx = tuple(level0_shape[-3:])
    n = math.prod(math.ceil(s / chunk) for s in zyx)
    if len(level0_shape) == 4:
        n *= level0_shape[0]        # one chunk-column per channel (chunk[0]==1)
    return n


class TestWritePhaseOptimizations:
    def test_levels_equal_independent_coarsen_chain(self, tmp_path):
        """OPT 1 EQUIVALENCE: the store built the NEW way (level 0 from source,
        each coarser level coarsened from the previous level READ BACK from the
        store) is byte-identical to the straightforward reference — a plain
        chained ``coarsen(np.mean, ..., trim_excess=True).astype(dtype)`` off the
        canonical source — level for level, incl. exact dtype, shapes, and the
        NGFF cumulative-factor scales.  A small ANISOTROPIC uint16 volume: Z is
        already below min_size (kept), XY halves twice."""
        import dask.array as da

        chunk, ds, min_size = 16, 2, 24
        # RANDOM, not `arange % 60000`: with the modulo field every block mean
        # was already exact, so rint == truncate and the chain equivalence could
        # not detect a rounding regression (measured: 0 differing voxels).  This
        # one differs in ~6.1k voxels across the chain.
        rng = np.random.default_rng(5)
        arr = rng.integers(0, 60000, size=(7, 96, 96)).astype(np.uint16)
        src = _write_tif(tmp_path, "equiv.tif", arr)
        out = tmp_path / "equiv.zarr"
        man = convert_to_ome_zarr(
            src, out, chunk=chunk, downsample=ds, min_size=min_size)

        shapes = [tuple(s) for s in man["level_shapes"]]
        assert shapes == [(7, 96, 96), (7, 48, 48), (7, 24, 24)]  # anisotropic

        # INDEPENDENT reference: a plain chained mean-coarsen off the canonical
        # source, per-axis factor derived purely from the shape ratios (not from
        # any converter internal).  Each level is cast back to the dtype BEFORE
        # feeding the next coarsen — exactly the pyramid's definition.
        ref_levels = [arr.copy()]                       # level 0 == source
        cur = da.from_array(arr, chunks=(chunk, chunk, chunk))
        for i in range(1, len(shapes)):
            f = [shapes[i - 1][a] // shapes[i][a] for a in range(3)]
            nxt = (np.rint(da.coarsen(np.mean, cur,
                                      {0: f[0], 1: f[1], 2: f[2]},
                                      trim_excess=True))
                   .astype(np.uint16)
                   .rechunk((chunk, chunk, chunk)))
            ref_levels.append(nxt.compute())
            cur = nxt

        levels, meta = open_multiscale(out)
        assert len(levels) == len(ref_levels) == man["n_levels"]
        for i, (lvl, ref) in enumerate(zip(levels, ref_levels)):
            got = lvl.compute()
            assert got.dtype == np.uint16 == ref.dtype, f"dtype drift at level {i}"
            assert got.shape == ref.shape == shapes[i], f"shape drift at level {i}"
            assert np.array_equal(got, ref), f"value drift at level {i}"

        # NGFF cumulative-factor scales: Z clamped -> 1, XY -> 1,2,4
        assert meta["scales"] == [
            [1.0, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 4.0, 4.0]
        ]

    def test_source_read_exactly_once(self, tmp_path, monkeypatch):
        """OPT 1 SOURCE-READ-ONCE: the on-disk source is consumed exactly ONCE
        (for level 0); every coarser level reads the previous level back from the
        STORE, not the source.

        Two spies:
        * a read-counter on ``_LazyReadProxy.__getitem__`` (the memmap source
          handle): total source-chunk reads stays ~= level 0's chunk count and
          does NOT scale with level count (the old per-level chain re-read the
          whole source once PER LEVEL -> ~N x that count).
        * ``da.from_zarr`` component reads during the convert are exactly the
          previous levels ``"0".."N-2"`` — proof the coarser levels come from the
          store.
        """
        import dask.array as da

        from packages.epicycle_largedata import multiscale as ms

        chunk = 16
        # RANDOM, not `arange % 60000`: with the modulo field every block mean
        # was already exact, so rint == truncate and the chain equivalence could
        # not detect a rounding regression (measured: 0 differing voxels).  This
        # one differs in ~6.1k voxels across the chain.
        rng = np.random.default_rng(5)
        arr = rng.integers(0, 60000, size=(7, 96, 96)).astype(np.uint16)
        src = _write_tif(tmp_path, "once.tif", arr)   # uncompressed -> memmap
        out = tmp_path / "once.zarr"

        reads = {"n": 0}
        orig_getitem = ms._LazyReadProxy.__getitem__

        def _counting_getitem(self, key):
            reads["n"] += 1
            return orig_getitem(self, key)

        monkeypatch.setattr(ms._LazyReadProxy, "__getitem__", _counting_getitem)

        comps = []
        orig_from_zarr = da.from_zarr

        def _spy_from_zarr(*a, **k):
            comps.append(k.get("component"))
            return orig_from_zarr(*a, **k)

        monkeypatch.setattr(da, "from_zarr", _spy_from_zarr)

        # num_workers=1 -> synchronous scheduler: each cubic chunk is read once,
        # so the source-read count is deterministic.
        man = convert_to_ome_zarr(
            src, out, chunk=chunk, downsample=2, min_size=24, num_workers=1)

        n_l0 = _l0_chunk_count(tuple(man["level_shapes"][0]), chunk)
        # source read exactly once: >= its chunk count (each level-0 chunk read
        # to write it) and STRICTLY LESS than twice it (the discriminator — the
        # old per-level chain would re-read the source ~N times => ~N x n_l0).
        assert n_l0 <= reads["n"] < 2 * n_l0, (
            f"source read {reads['n']} chunks; expected ~{n_l0} (read-once), "
            f"NOT ~{n_l0 * man['n_levels']} (the old per-level re-read)"
        )
        # coarser levels read the PREVIOUS level back from the store.
        assert comps == [str(i) for i in range(man["n_levels"] - 1)]

    def test_peak_ram_bounded_workers_1_and_4(self, tmp_path):
        """OPT 2 RAM-BOUNDED: convert a volume many x the cubic chunk at
        ``num_workers=1`` AND ``num_workers=4``; peak traced RAM stays a small
        multiple of the chunk working set for BOTH — it never scales with the
        volume, and the 4-worker peak is at most ~4x the 1-worker peak."""
        import dask          # noqa: F401  (warm imports before the timed region)
        import dask.array as da  # noqa: F401
        import zarr          # noqa: F401

        Z, Y, X = 48, 1024, 1024
        chunk = 64
        itemsize = np.dtype(np.uint16).itemsize
        chunk_ws = chunk ** 3 * itemsize                    # one cubic chunk ~0.5 MB
        level1_bytes = 48 * 512 * 512 * itemsize            # ~25 MB (Z clamps)
        # Tight, volume-INDEPENDENT ceiling: 16 cubic chunks of streaming set +
        # 8 MB graph/writer bookkeeping (scales with chunk COUNT, not size).
        # ~16 MB — below level 1 (25 MB), so materializing any whole level fails,
        # for BOTH schedulers.
        ceiling = 16 * chunk_ws + 8 * 1024 * 1024           # ~16.4 MB

        arr = np.arange(Z * Y * X, dtype=np.uint16).reshape(Z, Y, X)
        src = _write_tif(tmp_path, "ram.tif", arr)          # uncompressed -> memmap
        del arr

        # warm the full convert path for BOTH schedulers off the timed region
        # (imports, proxy, coarsen, to_zarr, from_zarr, both schedulers).
        warm = _write_tif(tmp_path, "ramwarm.tif", _distinct((4, 8, 8), np.uint16))
        convert_to_ome_zarr(warm, tmp_path / "w1.zarr",
                            chunk=4, downsample=2, min_size=2, num_workers=1)
        convert_to_ome_zarr(warm, tmp_path / "w4.zarr",
                            chunk=4, downsample=2, min_size=2, num_workers=4)

        def _peak(nw, name):
            tracemalloc.start()
            try:
                convert_to_ome_zarr(src, tmp_path / name, chunk=chunk,
                                    downsample=2, min_size=256, num_workers=nw)
                _cur, pk = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            return pk

        peak1 = _peak(1, "ram1.zarr")
        peak4 = _peak(4, "ram4.zarr")

        assert peak1 < ceiling, (
            f"workers=1 peak {peak1/1e6:.1f} MB not < {ceiling/1e6:.1f} MB "
            f"(cubic chunk {chunk_ws/1e6:.2f} MB) — a whole level materialized?"
        )
        assert peak4 < ceiling, (
            f"workers=4 peak {peak4/1e6:.1f} MB not < {ceiling/1e6:.1f} MB "
            f"— threading must stay bounded, below level 1 ({level1_bytes/1e6:.0f} MB)"
        )
        # 4-worker peak is bounded ~4x the 1-worker peak (task's "≈4x at most"),
        # NOT the whole volume — proves parallelism does not unbound RAM.
        assert peak4 <= 4 * peak1 + 4 * chunk_ws, (
            f"workers=4 peak {peak4/1e6:.1f} MB exceeds ~4x workers=1 "
            f"{peak1/1e6:.1f} MB — parallelism is not bounded"
        )

    def test_workers_1_and_4_byte_identical(self, tmp_path):
        """OPT 2 determinism: threading changes no bytes — every level of a
        ``num_workers=4`` store equals the ``num_workers=1`` store."""
        arr = (np.arange(8 * 128 * 128) % 60000).astype(np.uint16).reshape(8, 128, 128)
        src = _write_tif(tmp_path, "det.tif", arr)
        out1 = tmp_path / "det1.zarr"
        out4 = tmp_path / "det4.zarr"

        man1 = convert_to_ome_zarr(
            src, out1, chunk=32, downsample=2, min_size=32, num_workers=1)
        man4 = convert_to_ome_zarr(
            src, out4, chunk=32, downsample=2, min_size=32, num_workers=4)

        assert man1["n_levels"] == man4["n_levels"] >= 2
        assert man1["level_shapes"] == man4["level_shapes"]

        levels1, _ = open_multiscale(out1)
        levels4, _ = open_multiscale(out4)
        for i, (a, b) in enumerate(zip(levels1, levels4)):
            va, vb = a.compute(), b.compute()
            assert va.dtype == vb.dtype
            assert np.array_equal(va, vb), f"workers=1 vs 4 differ at level {i}"


# ---------------------------------------------------------------------------
# 12) interrupted convert must not leave a store the freshness probe accepts
#
# ---------------------------------------------------------------------------

class TestInterruptedConvertLeavesNoFreshMarker:
    """An interrupted Prepare (crash / disk full / cancel) must NOT leave a
    store carrying the ``epicycle_source`` provenance marker: Step-1's
    ``_fresh_prepared_store_for`` treats that marker as "this store is a good,
    complete conversion of that source", and the missing chunks then read back
    as ZEROS forever.  The marker is therefore written only AFTER every level
    has been written, and a failed convert removes its partial store."""

    @staticmethod
    def _fail_on_level(monkeypatch, level_index):
        """Make ``da.to_zarr`` raise when writing component ``level_index``."""
        import dask.array as da

        real = da.to_zarr

        def flaky(arr, url=None, component=None, **kw):
            if str(component) == str(level_index):
                raise RuntimeError("simulated disk-full mid-convert")
            return real(arr, url=url, component=component, **kw)

        monkeypatch.setattr(da, "to_zarr", flaky)

    def test_failure_mid_levels_leaves_no_marker_and_no_store(
            self, tmp_path, monkeypatch):
        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "interrupted.tif", arr)
        out = tmp_path / "interrupted.zarr"

        self._fail_on_level(monkeypatch, 1)     # dies writing the 2nd level

        with pytest.raises(RuntimeError):
            convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        # Whatever survived, it must NOT look like a complete prepared store.
        assert read_store_source(out) is None
        # Best-effort cleanup: the partial store is gone.
        assert not Path(out).exists()

    def test_failure_on_level0_leaves_no_marker(self, tmp_path, monkeypatch):
        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "interrupted0.tif", arr)
        out = tmp_path / "interrupted0.zarr"

        self._fail_on_level(monkeypatch, 0)

        with pytest.raises(RuntimeError):
            convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)

        assert read_store_source(out) is None

    def test_success_path_still_writes_the_marker(self, tmp_path):
        """The happy path is unchanged: a complete convert is recognisable."""
        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "complete.tif", arr)
        out = tmp_path / "complete.zarr"

        manifest = convert_to_ome_zarr(src, out, chunk=16, downsample=2,
                                       min_size=8)

        prov = read_store_source(out)
        assert prov is not None
        assert prov["path"] == os.path.abspath(src)
        assert prov["source_axes"] == "ZYX"
        # and the data is all there
        levels, _meta = open_multiscale(out)
        assert len(levels) == manifest["n_levels"]
        np.testing.assert_array_equal(np.asarray(levels[0]), arr)


class TestConvertCancel:
    """Brick 1b #2 — a real in-loop cancel.  Prepare's Cancel used to only
    DISCARD the result: the converter kept writing every remaining pyramid
    level to disk.  ``should_cancel`` is polled between level writes and raises
    ``ConvertCancelled``, so the existing ``finally`` removes the partial store
    and no ``epicycle_source`` marker is ever written."""

    def test_cancel_before_any_write_stops_and_leaves_no_store(self, tmp_path):
        from packages.epicycle_largedata.exceptions import ConvertCancelled

        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "cancel0.tif", arr)
        out = tmp_path / "cancel0.zarr"

        with pytest.raises(ConvertCancelled):
            convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8,
                                should_cancel=lambda: True)

        assert read_store_source(out) is None
        assert not Path(out).exists()

    def test_cancel_between_levels_stops_at_the_next_check(self, tmp_path):
        """Cancel raised after the level-0 write: the coarser levels are never
        written, the marker is absent and the partial store is removed."""
        from packages.epicycle_largedata.exceptions import ConvertCancelled

        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "cancel1.tif", arr)
        out = tmp_path / "cancel1.zarr"

        polls = {"n": 0}

        def should_cancel():
            polls["n"] += 1
            return polls["n"] > 1          # let level 0 through, then stop

        with pytest.raises(ConvertCancelled):
            convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8,
                                should_cancel=should_cancel)

        assert polls["n"] >= 2             # it really was polled in the loop
        assert read_store_source(out) is None
        assert not Path(out).exists()

    def test_cancel_is_a_baseexception(self):
        """Mirrors ``PreprocessingCancelled``: derived from ``BaseException`` so
        no ``except Exception`` guard on the way out can swallow it into a
        "conversion failed" or, worse, a silent success."""
        from packages.epicycle_largedata.exceptions import ConvertCancelled

        assert issubclass(ConvertCancelled, BaseException)
        assert not issubclass(ConvertCancelled, Exception)

    def test_a_raising_probe_is_treated_as_not_cancelled(self, tmp_path):
        """A broken probe must not masquerade as a user cancel (which would
        silently delete a healthy store), nor abort a healthy convert."""
        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "boomprobe.tif", arr)
        out = tmp_path / "boomprobe.zarr"
        polls = {"n": 0}

        def boom():
            polls["n"] += 1
            raise RuntimeError("probe blew up")

        convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8,
                            should_cancel=boom)

        assert polls["n"] >= 2                     # it really was polled
        assert read_store_source(out) is not None  # ... and the store completed
        levels, _meta = open_multiscale(out)
        np.testing.assert_array_equal(np.asarray(levels[0]), arr)

    def test_no_should_cancel_is_unchanged(self, tmp_path):
        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "nocancel.tif", arr)
        out = tmp_path / "nocancel.zarr"
        convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8)
        assert read_store_source(out) is not None

    def test_should_cancel_returning_false_is_unchanged(self, tmp_path):
        arr = _distinct((16, 96, 96), dtype=np.uint16)
        src = _write_tif(tmp_path, "falsecancel.tif", arr)
        out = tmp_path / "falsecancel.zarr"
        convert_to_ome_zarr(src, out, chunk=16, downsample=2, min_size=8,
                            should_cancel=lambda: False)
        prov = read_store_source(out)
        assert prov is not None
        levels, _meta = open_multiscale(out)
        np.testing.assert_array_equal(np.asarray(levels[0]), arr)

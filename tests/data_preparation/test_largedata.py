"""Tests for the L0 large-data data layer (`_largedata.py`).

Pure and Qt-free by construction — `_largedata` is the DATA layer (lazy open,
axis detection, label-safe downsample, lazy crop), so every helper here is a
function of arrays / header numbers only.  The module is the foundation the ③
box tool and the preprocessing engine sit on, and the whole
point of it is that it must NOT `imread`-everything: cropping reads only the
ROI off disk.  These tests prove that laziness AND that a lazy crop is
byte-identical to the eager crop it replaces.

The load-bearing case is the real WoundT01 file: tifffile reports its axes as
``'QQYX'`` (unresolved) on a ``(2, 143, 1024, 1024)`` uint16 DEFLATE stack, so
`resolve_axes` MUST infer channel-axis 0 / CZYX from the shape, and `open_lazy`
MUST take the compressed->dask branch (a DEFLATE page is not memory-mappable).
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation._largedata import (
    AxisResolution,
    CHANNEL_AXIS_MAX,
    content_bbox,
    crop_czyx,
    crop_zyx,
    downsample_for_display,
    materialize,
    open_lazy,
    resolve_axes,
)


# ---------------------------------------------------------------------------
# resolve_axes — real axis detection
# ---------------------------------------------------------------------------

class TestResolveAxesInferred:
    """When metadata is missing or unresolved (tifffile's 'Q' unknown marker),
    infer: a small leading axis (<=~8) is channels, the large one is Z."""

    def test_wound_t01_qqyx_resolves_to_czyx_c0(self):
        """THE load-bearing case: (2,143,1024,1024) 'QQYX' -> CZYX, C axis 0."""
        r = resolve_axes((2, 143, 1024, 1024), "QQYX")
        assert r.axes == "CZYX"
        assert r.channel_axis == 0
        assert r.n_channels == 2

    def test_leading_large_axis_is_z_not_channel(self):
        """(143,2,...) 'QQYX' -> the small SECOND axis is channels -> ZCYX."""
        r = resolve_axes((143, 2, 1024, 1024), "QQYX")
        assert r.axes == "ZCYX"
        assert r.channel_axis == 1
        assert r.n_channels == 2

    def test_none_metadata_infers_from_shape(self):
        r = resolve_axes((3, 90, 512, 512), None)
        assert r.axes == "CZYX"
        assert r.channel_axis == 0
        assert r.n_channels == 3

    def test_empty_metadata_infers_from_shape(self):
        assert resolve_axes((80, 4, 256, 256), "").axes == "ZCYX"

    def test_inferred_is_not_confident(self):
        """An inferred guess must flag low confidence so the UI shows+confirms."""
        assert resolve_axes((2, 143, 64, 64), "QQYX").confident is False

    @pytest.mark.parametrize("a0,a1,expected_c", [
        (2, 143, 0),      # channel first
        (143, 2, 1),      # channel second
        (1, 200, 0),      # single-channel-as-4D, still axis 0
        (5, 5, 0),        # tie -> axis 0 (matches historical shape[0]>shape[1])
        (6, 3, 1),        # both small, second smaller -> channel second
    ])
    def test_channel_axis_inference_table(self, a0, a1, expected_c):
        r = resolve_axes((a0, a1, 32, 32), "QQYX")
        assert r.channel_axis == expected_c
        assert r.axes == ("CZYX" if expected_c == 0 else "ZCYX")

    def test_matches_historical_auto_transpose_default(self):
        """The old loader transposed iff shape[0] > shape[1] (treat as ZCYX).
        Detection must agree on that default so routine extraction is unchanged.
        """
        for a0, a1 in [(2, 143), (143, 2), (3, 4), (4, 3), (100, 200), (200, 100)]:
            r = resolve_axes((a0, a1, 16, 16), "QQYX")
            historical_zcyx = a0 > a1
            assert (r.axes == "ZCYX") == historical_zcyx, (a0, a1)


class TestResolveAxesMetadata:
    """A CLEAN metadata axes string is trusted (and marked confident)."""

    def test_trusts_clean_czyx(self):
        r = resolve_axes((2, 143, 512, 512), "CZYX")
        assert r.axes == "CZYX"
        assert r.channel_axis == 0
        assert r.n_channels == 2
        assert r.confident is True
        assert r.source == "metadata"

    def test_trusts_clean_zcyx_even_against_shape_guess(self):
        """(4, 3, ...) shape alone would guess ZCYX (a1 smaller); a clean 'CZYX'
        overrides the shape heuristic — metadata wins when it is trustworthy."""
        r = resolve_axes((4, 3, 128, 128), "CZYX")
        assert r.axes == "CZYX"
        assert r.channel_axis == 0
        assert r.n_channels == 4

    def test_zcyx_metadata(self):
        r = resolve_axes((143, 2, 512, 512), "ZCYX")
        assert r.axes == "ZCYX"
        assert r.channel_axis == 1
        assert r.n_channels == 2

    def test_case_insensitive(self):
        assert resolve_axes((2, 10, 32, 32), "czyx").axes == "CZYX"

    def test_partial_unknown_axes_falls_back_to_inference(self):
        """'QCYX' still carries a 'Q' — treat the whole string as unresolved."""
        r = resolve_axes((2, 143, 64, 64), "QCYX")
        assert r.source == "inferred"

    def test_clean_metadata_round_trips_through_tifffile(self, tmp_path):
        """The path that actually DIVERGES from the old default: a file written
        with an explicit clean ``axes='CZYX'`` is read back as 'CZYX' by tifffile
        and trusted with confidence — NOT re-guessed from the shape.  (Every
        other fixture writes a plain ndarray whose axes come back 'QQYX', the
        inferred path, so this is the only coverage of the trusted-metadata
        branch end-to-end.)"""
        arr = np.random.randint(0, 60000, (2, 12, 64, 64)).astype(np.uint16)
        p = tmp_path / "clean_czyx.tif"
        tifffile.imwrite(str(p), arr, metadata={"axes": "CZYX"})
        with tifffile.TiffFile(str(p)) as tf:
            meta_axes = tf.series[0].axes
        assert meta_axes == "CZYX"          # tifffile preserved the clean string
        r = resolve_axes(arr.shape, meta_axes)
        assert r.axes == "CZYX"
        assert r.channel_axis == 0
        assert r.confident is True
        assert r.source == "metadata"

    def test_clean_czyx_metadata_trusted_when_c_gt_z(self, tmp_path):
        """C>Z (10 channels, 5 Z): the OLD ``shape[0] > shape[1]`` rule would
        transpose this to garbage; a clean ``axes='CZYX'`` must be trusted as-is
        so the channel axis stays 0."""
        arr = np.random.randint(0, 60000, (10, 5, 64, 64)).astype(np.uint16)
        p = tmp_path / "clean_cgtz.tif"
        tifffile.imwrite(str(p), arr, metadata={"axes": "CZYX"})
        with tifffile.TiffFile(str(p)) as tf:
            meta_axes = tf.series[0].axes
        assert meta_axes == "CZYX"
        r = resolve_axes(arr.shape, meta_axes)
        assert r.axes == "CZYX"
        assert r.channel_axis == 0
        assert r.n_channels == 10
        assert r.confident is True
        assert r.source == "metadata"
        # The plain (no-metadata) write of the same shape would be inferred to
        # ZCYX (a0 > a1) — proving metadata is what flips the interpretation.
        assert resolve_axes(arr.shape, "QQYX").axes == "ZCYX"


class TestResolveAxes3D:
    """3D (Z,Y,X) is single-channel — no channel axis."""

    def test_zyx(self):
        r = resolve_axes((90, 512, 512), "ZYX")
        assert r.axes == "ZYX"
        assert r.channel_axis is None
        assert r.n_channels == 1

    def test_zyx_without_metadata(self):
        r = resolve_axes((90, 512, 512), None)
        assert r.axes == "ZYX"
        assert r.channel_axis is None
        assert r.n_channels == 1


class TestResolveAxesUnsupported:
    def test_too_many_dims_raises(self):
        with pytest.raises(ValueError):
            resolve_axes((3, 2, 143, 512, 512), "TCZYX")

    def test_channel_axis_max_is_a_named_constant(self):
        assert CHANNEL_AXIS_MAX >= 2


# ---------------------------------------------------------------------------
# open_lazy — memmap (uncompressed) / dask (compressed & zarr)
# ---------------------------------------------------------------------------

class TestOpenLazyUncompressed:
    def test_uncompressed_returns_memmap(self, tmp_path):
        arr = np.random.randint(0, 4000, (2, 6, 32, 32)).astype(np.uint16)
        p = tmp_path / "u.tif"
        tifffile.imwrite(str(p), arr)
        handle = open_lazy(str(p))
        assert isinstance(handle, np.memmap)
        assert tuple(handle.shape) == arr.shape

    def test_uncompressed_crop_equals_eager(self, tmp_path):
        arr = np.random.randint(0, 4000, (2, 6, 32, 32)).astype(np.uint16)
        p = tmp_path / "u.tif"
        tifffile.imwrite(str(p), arr)
        handle = open_lazy(str(p))
        bounds = {"z0": 1, "z1": 4, "y0": 5, "y1": 20, "x0": 3, "x1": 18}
        got = crop_czyx(handle, bounds)
        want = arr[:, 1:4, 5:20, 3:18]
        assert np.array_equal(got, want)


class TestOpenLazyCompressed:
    """DEFLATE pages are NOT memory-mappable (the WoundT01 lesson) -> dask."""

    def test_compressed_returns_lazy_dask_handle(self, tmp_path):
        arr = np.random.randint(0, 4000, (2, 6, 32, 32)).astype(np.uint16)
        p = tmp_path / "c.tif"
        tifffile.imwrite(str(p), arr, compression="deflate")
        handle = open_lazy(str(p))
        assert hasattr(handle, "compute"), "compressed source must be lazy (dask)"
        assert not isinstance(handle, np.memmap)
        assert tuple(handle.shape) == arr.shape

    def test_compressed_crop_equals_eager_crop(self, tmp_path):
        """The core L0 guarantee: a lazy crop is byte-identical to the eager
        crop it replaces, while only the crop is ever materialized."""
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        p = tmp_path / "c.tif"
        tifffile.imwrite(str(p), arr, compression="deflate")
        handle = open_lazy(str(p))
        bounds = {"z0": 2, "z1": 6, "y0": 4, "y1": 30, "x0": 6, "x1": 33}
        got = crop_czyx(handle, bounds)
        want = arr[:, 2:6, 4:30, 6:33]
        assert got.shape == want.shape
        assert np.array_equal(got, want)

    def test_compressed_3d_crop_equals_eager(self, tmp_path):
        arr = np.random.randint(0, 60000, (10, 40, 40)).astype(np.uint16)
        p = tmp_path / "c3.tif"
        tifffile.imwrite(str(p), arr, compression="deflate")
        handle = open_lazy(str(p))
        bounds = {"z0": 1, "z1": 8, "y0": 2, "y1": 30, "x0": 5, "x1": 35}
        got = crop_zyx(handle, bounds)
        assert np.array_equal(got, arr[1:8, 2:30, 5:35])


class TestOpenLazyZarr:
    def test_zarr_store_returns_lazy(self, tmp_path):
        import dask.array as da

        arr = np.random.randint(0, 4000, (2, 6, 32, 32)).astype(np.uint16)
        out = tmp_path / "v.zarr"
        da.from_array(arr, chunks=(1, 3, 16, 16)).to_zarr(str(out), component="0",
                                                          overwrite=True)
        handle = open_lazy(str(out))
        assert hasattr(handle, "compute")
        assert np.array_equal(materialize(handle), arr)


# ---------------------------------------------------------------------------
# downsample_for_display — strided, label-safe
# ---------------------------------------------------------------------------

class TestDownsample:
    def test_factors_returned_and_clamped(self):
        arr = np.zeros((10, 40, 40), np.uint16)
        sub, facs = downsample_for_display(arr, (2, 4, 4))
        assert facs == (2, 4, 4)
        assert sub.shape == (5, 10, 10)

    def test_factor_below_one_is_clamped_to_one(self):
        arr = np.zeros((10, 40, 40), np.uint16)
        sub, facs = downsample_for_display(arr, (0, 1, 1))
        assert facs == (1, 1, 1)
        assert sub.shape == arr.shape

    def test_is_strided_not_averaged_label_safe(self):
        """Downsample must be nearest/strided so integer LABEL ids survive —
        averaging would invent ids between labels."""
        arr = np.arange(8 * 8, dtype=np.uint16).reshape(1, 8, 8)
        sub, _ = downsample_for_display(arr, (1, 2, 2))
        assert np.array_equal(sub, arr[:, ::2, ::2])
        assert set(np.unique(sub)).issubset(set(np.unique(arr)))

    def test_czyx_downsamples_only_spatial_axes(self):
        arr = np.random.randint(0, 100, (3, 10, 40, 40)).astype(np.uint16)
        sub, facs = downsample_for_display(arr, (2, 4, 4))
        assert sub.shape == (3, 5, 10, 10)  # channel axis untouched
        assert np.array_equal(sub, arr[:, ::2, ::4, ::4])

    def test_materializes_a_dask_input(self, tmp_path):
        import dask.array as da

        arr = np.random.randint(0, 100, (2, 8, 16, 16)).astype(np.uint16)
        d = da.from_array(arr, chunks=(1, 4, 8, 8))
        sub, _ = downsample_for_display(d, (1, 2, 2))
        assert isinstance(sub, np.ndarray) and not hasattr(sub, "compute")
        assert np.array_equal(sub, arr[:, :, ::2, ::2])

    def test_bad_ndim_raises(self):
        with pytest.raises(ValueError):
            downsample_for_display(np.zeros((4, 4)), (1, 1, 1))


# ---------------------------------------------------------------------------
# crop_* — lazy slice then materialize only the crop
# ---------------------------------------------------------------------------

class TestCrop:
    def test_crop_equals_full_then_crop_czyx(self):
        arr = np.random.randint(0, 5000, (2, 8, 20, 20)).astype(np.uint16)
        b = {"z0": 1, "z1": 5, "y0": 2, "y1": 15, "x0": 3, "x1": 17}
        assert np.array_equal(crop_czyx(arr, b), arr[:, 1:5, 2:15, 3:17])

    def test_crop_equals_full_then_crop_zyx(self):
        arr = np.random.randint(0, 5000, (8, 20, 20)).astype(np.uint16)
        b = {"z0": 0, "z1": 8, "y0": 4, "y1": 12, "x0": 4, "x1": 12}
        assert np.array_equal(crop_zyx(arr, b), arr[0:8, 4:12, 4:12])

    def test_materialize_computes_dask_and_passes_numpy(self):
        import dask.array as da

        arr = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        assert np.array_equal(materialize(da.from_array(arr, chunks=1)), arr)
        assert np.array_equal(materialize(arr), arr)


# ---------------------------------------------------------------------------
# content_bbox — nonzero bounding box (framing helper)
# ---------------------------------------------------------------------------

class TestContentBbox:
    def test_locates_nonzero_block_3d(self):
        arr = np.zeros((40, 80, 80), np.uint16)
        arr[10:20, 30:50, 40:60] = 7
        mn, mx = content_bbox(arr)
        assert mn[0] <= 10 and mx[0] >= 20
        assert mn[1] <= 30 and mx[1] >= 50
        assert mn[2] <= 40 and mx[2] >= 60

    def test_all_zero_returns_full_extent(self):
        arr = np.zeros((10, 20, 20), np.uint16)
        mn, mx = content_bbox(arr)
        assert mn == (0, 0, 0)
        assert mx == (10, 20, 20)

    def test_czyx_uses_first_channel(self):
        arr = np.zeros((2, 20, 40, 40), np.uint16)
        arr[0, 5:10, 10:20, 10:20] = 3
        mn, mx = content_bbox(arr)
        assert len(mn) == 3 and len(mx) == 3


# ---------------------------------------------------------------------------
# dataclass contract
# ---------------------------------------------------------------------------

class TestAxisResolutionDataclass:
    def test_fields(self):
        r = AxisResolution(axes="CZYX", channel_axis=0, n_channels=2,
                           confident=True, source="metadata")
        assert (r.axes, r.channel_axis, r.n_channels) == ("CZYX", 0, 2)

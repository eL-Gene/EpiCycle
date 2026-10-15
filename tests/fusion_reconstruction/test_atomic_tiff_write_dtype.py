"""Atomic TIFF write `dtype=` refactor — Phase A #3 (perf push 2026-05-09).

Pre-fix call sites (controller.py:3009, 3232, 6939) used:

    _atomic_tiff_write(path, layer.data.astype(np.uint32))

The ``.astype(np.uint32)`` allocated a full second copy of the volume
(1.68 GB at 1024³) which lived until the tifffile.imwrite call returned.

The fix moves the dtype responsibility into ``_atomic_tiff_write`` itself
via an optional ``dtype=`` kwarg.  Internally:

    np.asarray(data).astype(dtype, copy=False)

— a no-op when the input array already has the target dtype (zero
allocation), exactly one cast otherwise.  Same semantics as before,
centralized.

These tests pin the new API contract.  No Qt — pure tempfile + tifffile.
"""

import os

import numpy as np
import pytest
import tifffile


@pytest.fixture
def tmp_tif(tmp_path):
    return tmp_path / "test_output.tif"


# ---------------------------------------------------------------------------
# dtype= kwarg performs the conversion
# ---------------------------------------------------------------------------


class TestDtypeKwargConverts:

    def test_int32_to_uint32(self, tmp_tif):
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )
        data = np.array([[1, 2], [3, 4]], dtype=np.int32)
        _atomic_tiff_write(str(tmp_tif), data, dtype=np.uint32)

        loaded = tifffile.imread(str(tmp_tif))
        assert loaded.dtype == np.uint32
        np.testing.assert_array_equal(
            loaded, np.array([[1, 2], [3, 4]], dtype=np.uint32),
        )

    def test_int64_to_uint32(self, tmp_tif):
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )
        data = np.array([[10, 20], [30, 40]], dtype=np.int64)
        _atomic_tiff_write(str(tmp_tif), data, dtype=np.uint32)

        loaded = tifffile.imread(str(tmp_tif))
        assert loaded.dtype == np.uint32
        np.testing.assert_array_equal(
            loaded, np.array([[10, 20], [30, 40]], dtype=np.uint32),
        )


# ---------------------------------------------------------------------------
# No-copy invariant when dtype already matches
# ---------------------------------------------------------------------------


class TestNoCopyWhenAlreadyMatching:
    """When the input array already has the target dtype, the function
    must NOT allocate an intermediate copy.  ``astype(dtype, copy=False)``
    delivers this — verified via id() identity through a
    tifffile.imwrite spy."""

    def test_uint32_input_with_uint32_dtype_no_copy(self, tmp_tif):
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )
        data = np.array([[1, 2], [3, 4]], dtype=np.uint32)
        original_id = id(data)

        captured = {}
        original_imwrite = tifffile.imwrite

        def spy_imwrite(path, arr, **kwargs):
            captured["arr_id"] = id(arr)
            captured["dtype"] = arr.dtype
            return original_imwrite(path, arr, **kwargs)

        tifffile.imwrite = spy_imwrite
        try:
            _atomic_tiff_write(str(tmp_tif), data, dtype=np.uint32)
        finally:
            tifffile.imwrite = original_imwrite

        assert captured["arr_id"] == original_id, (
            "Function allocated a copy even though input dtype matched target"
        )
        assert captured["dtype"] == np.uint32


# ---------------------------------------------------------------------------
# Backwards compat — positional API still works
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """When no ``dtype`` kwarg is given, behavior matches pre-fix:
    writes the array's existing dtype to disk."""

    def test_no_dtype_kwarg_writes_native_dtype(self, tmp_tif):
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )
        data = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        _atomic_tiff_write(str(tmp_tif), data)

        loaded = tifffile.imread(str(tmp_tif))
        assert loaded.dtype == np.uint16

    def test_existing_call_pattern_still_works(self, tmp_tif):
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )
        for dtype in [np.uint8, np.int16, np.uint32, np.int64]:
            data = np.array([[1, 2], [3, 4]], dtype=dtype)
            _atomic_tiff_write(str(tmp_tif), data)
            loaded = tifffile.imread(str(tmp_tif))
            assert loaded.dtype == dtype


# ---------------------------------------------------------------------------
# Atomicity preserved
# ---------------------------------------------------------------------------


class TestAtomicityPreserved:
    """A failure during the write must leave the original file
    untouched — atomicity property must survive the refactor."""

    def test_no_partial_file_on_imwrite_failure(self, tmp_tif):
        from napari_epicycle.fusion_reconstruction.controller import (
            _atomic_tiff_write,
        )
        original = np.array([[99, 99], [99, 99]], dtype=np.uint32)
        tifffile.imwrite(str(tmp_tif), original)

        original_imwrite = tifffile.imwrite

        def fail_imwrite(path, arr, **kwargs):
            raise RuntimeError("simulated write failure")

        tifffile.imwrite = fail_imwrite
        try:
            with pytest.raises(RuntimeError):
                _atomic_tiff_write(
                    str(tmp_tif),
                    np.array([[1, 2], [3, 4]], dtype=np.int32),
                    dtype=np.uint32,
                )
        finally:
            tifffile.imwrite = original_imwrite

        loaded = tifffile.imread(str(tmp_tif))
        np.testing.assert_array_equal(loaded, original)
        leftover_tifs = list(tmp_tif.parent.glob("*.tif"))
        assert leftover_tifs == [tmp_tif], (
            f"Leftover temp files: {leftover_tifs}"
        )

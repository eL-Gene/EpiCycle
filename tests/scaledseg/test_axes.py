"""
Tests for the axis resolver + canonicalization (Stack 4).

Mirrors the monorepo's ``test_resolve_patch_axes.py`` (the pure-resolver cases),
adjusted for the engine's two deliberate divergences: the fallback default is
ZCYX (not CZYX), and the labels-shape tier is dropped (no labels at tiling).
Adds ``canonicalize_to_zcyx`` content-fidelity tests and ``ZStackTiler``
integration tests proving CZYX/ZYX inputs now tile correctly.
"""

import numpy as np
import pytest
import tifffile

from packages.epicycle_scaledseg.utils.axes import (
    canonicalize_to_zcyx,
    detect_axes_order,
    resolve_input_axes,
)


# ---------------------------------------------------------------------------
# detect_axes_order (size heuristic)
# ---------------------------------------------------------------------------


def test_detect_3d_is_zyx():
    assert detect_axes_order((10, 64, 64)) == "ZYX"


def test_detect_channel_first_is_czyx():
    assert detect_axes_order((2, 50, 64, 64)) == "CZYX"  # axis-0 smallest


def test_detect_z_first_is_zcyx():
    assert detect_axes_order((50, 2, 64, 64)) == "ZCYX"  # axis-1 smallest


def test_detect_single_channel_first_is_czyx():
    assert detect_axes_order((1, 266, 512, 512)) == "CZYX"


def test_detect_single_channel_second_is_zcyx():
    assert detect_axes_order((266, 1, 512, 512)) == "ZCYX"


def test_detect_non_4d_defaults_czyx():
    assert detect_axes_order((2, 3, 4, 5, 6)) == "CZYX"  # not 3D/4D


# ---------------------------------------------------------------------------
# resolve_input_axes
# ---------------------------------------------------------------------------


def test_resolve_3d_returns_zyx():
    assert resolve_input_axes(np.zeros((10, 64, 64))) == "ZYX"


def test_resolve_heuristic_czyx_no_path():
    assert resolve_input_axes(np.zeros((2, 50, 64, 64))) == "CZYX"


def test_resolve_heuristic_zcyx_no_path():
    assert resolve_input_axes(np.zeros((50, 2, 64, 64))) == "ZCYX"


def test_resolve_metadata_wins_over_heuristic(tmp_path, monkeypatch):
    """Metadata (CZYX) overrides a shape the heuristic would call ZCYX."""
    import packages.epicycle_scaledseg.utils.axes as axes_mod

    class _FakeSeries:
        axes = "CZYX"

    class _FakeTiff:
        series = [_FakeSeries()]
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    # Shape (50,2,...) would be ZCYX by heuristic; metadata must win -> CZYX.
    monkeypatch.setattr(axes_mod, "tifffile", tifffile, raising=False)
    monkeypatch.setattr(tifffile, "TiffFile", lambda *a, **k: _FakeTiff())
    got = resolve_input_axes(np.zeros((50, 2, 8, 8)), image_path="x.tif")
    assert got == "CZYX"


def test_resolve_thin_z_metadata_beats_size_heuristic(monkeypatch):
    """Thin-Z ZCYX (Z=2,C=3): the size heuristic MIS-calls it CZYX (axis-0 is
    the smallest), so metadata is the only thing that rescues it to ZCYX. This
    path is load-bearing precisely because the engine dropped the monorepo's
    labels-shape tier — metadata is now the ONLY automatic thin-Z disambiguator
    (else the user must pass --input-axes)."""
    class _FakeSeries:
        axes = "ZCYX"

    class _FakeTiff:
        series = [_FakeSeries()]
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    arr = np.zeros((2, 3, 8, 8))  # genuinely ZCYX, but thin Z=2 < C=3
    assert detect_axes_order(arr.shape) == "CZYX"  # heuristic is WRONG here
    monkeypatch.setattr(tifffile, "TiffFile", lambda *a, **k: _FakeTiff())
    assert resolve_input_axes(arr, image_path="x.tif") == "ZCYX"  # metadata rescues


def test_resolve_metadata_ignored_when_unrecognized(monkeypatch):
    """Unrecognized metadata axes falls through to the heuristic."""
    class _FakeSeries:
        axes = "QSYX"

    class _FakeTiff:
        series = [_FakeSeries()]
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(tifffile, "TiffFile", lambda *a, **k: _FakeTiff())
    # (2,50,...) heuristic -> CZYX; bad metadata ignored.
    assert resolve_input_axes(np.zeros((2, 50, 8, 8)), image_path="x.tif") == "CZYX"


def test_resolve_bad_path_falls_through_to_heuristic():
    """An unreadable image_path must not raise — fall to the heuristic."""
    got = resolve_input_axes(np.zeros((50, 2, 8, 8)), image_path="/no/such.tif")
    assert got == "ZCYX"


def test_resolve_fallback_default_is_zcyx():
    """Garbage input (no ndim/shape) -> fallback, default ZCYX (the engine
    canonical), NOT the monorepo's CZYX."""
    assert resolve_input_axes(None) == "ZCYX"


def test_resolve_fallback_honored_when_given():
    assert resolve_input_axes(None, fallback="CZYX") == "CZYX"


def test_resolve_fallback_normalized_when_garbage():
    assert resolve_input_axes(None, fallback="junk") == "ZCYX"


@pytest.mark.parametrize("bad", [None, "a string", 123, np.zeros(5), np.zeros((4, 4))])
def test_resolve_never_raises(bad):
    out = resolve_input_axes(bad)
    assert out in ("ZYX", "ZCYX", "CZYX")


# ---------------------------------------------------------------------------
# canonicalize_to_zcyx
# ---------------------------------------------------------------------------


def test_canonicalize_zyx_inserts_channel():
    arr = np.arange(5 * 8 * 8).reshape(5, 8, 8)
    out = canonicalize_to_zcyx(arr, "ZYX")
    assert out.shape == (5, 1, 8, 8)
    assert np.array_equal(out[:, 0], arr)  # content preserved


def test_canonicalize_czyx_moveaxis_content_fidelity():
    # (C=2, Z=5, Y, X); after canonicalize -> (Z=5, C=2, Y, X), out[:,c]==in[c].
    arr = np.arange(2 * 5 * 8 * 8).reshape(2, 5, 8, 8)
    out = canonicalize_to_zcyx(arr, "CZYX")
    assert out.shape == (5, 2, 8, 8)
    for c in range(2):
        assert np.array_equal(out[:, c], arr[c])


def test_canonicalize_zcyx_is_noop():
    arr = np.zeros((5, 2, 8, 8))
    out = canonicalize_to_zcyx(arr, "ZCYX")
    assert out is arr  # no copy


def test_canonicalize_unsupported_raises():
    with pytest.raises(ValueError):
        canonicalize_to_zcyx(np.zeros((5, 2, 8, 8)), "XYZC")


# ---------------------------------------------------------------------------
# ZStackTiler integration (the bug-first fix)
# ---------------------------------------------------------------------------


def _write(tmp_path, shape, name, **imwrite_kw):
    arr = np.zeros(shape, dtype=np.uint16)
    p = tmp_path / name
    tifffile.imwrite(str(p), arr, **imwrite_kw)
    return p


def test_tiling_resolves_czyx(tmp_path):
    """CZYX (C=2,Z=5,H=8,W=8) is canonicalized -> tiler sees Z=5, C=2.
    This is the fix for the Stack-2 CZYX bug-lock."""
    from packages.epicycle_scaledseg.utils.tiling import ZStackTiler

    p = _write(tmp_path, (2, 5, 8, 8), "czyx.tif")
    tiler = ZStackTiler(p, tmp_path / "out", target_patch_size=8,
                        overlap=0.25, search_range=0)
    assert (tiler.Z, tiler.C, tiler.H, tiler.W) == (5, 2, 8, 8)
    assert tiler.input_axes == "CZYX"


def test_tiling_zcyx_unchanged(tmp_path):
    from packages.epicycle_scaledseg.utils.tiling import ZStackTiler

    p = _write(tmp_path, (5, 2, 8, 8), "zcyx.tif")
    tiler = ZStackTiler(p, tmp_path / "out", target_patch_size=8,
                        overlap=0.25, search_range=0)
    assert (tiler.Z, tiler.C) == (5, 2)
    assert tiler.input_axes == "ZCYX"


def test_tiling_handles_3d_zyx(tmp_path):
    """A 3D grayscale stack used to CRASH the 4-tuple unpack; now it
    canonicalizes to C=1 and tiles. channel_index is clamped to a valid axis."""
    from packages.epicycle_scaledseg.utils.tiling import ZStackTiler

    p = _write(tmp_path, (5, 8, 8), "zyx.tif")
    tiler = ZStackTiler(p, tmp_path / "out", target_patch_size=8,
                        overlap=0.25, search_range=0)
    assert (tiler.Z, tiler.C, tiler.H, tiler.W) == (5, 1, 8, 8)
    assert tiler.input_axes == "ZYX"
    assert tiler.channel_index <= tiler.C - 1  # clamped, no out-of-bounds


def test_tiling_input_axes_override(tmp_path):
    """An explicit input_axes overrides auto-resolution (the escape hatch for
    irreducibly-ambiguous thin-Z stacks)."""
    from packages.epicycle_scaledseg.utils.tiling import ZStackTiler

    # Shape (5,2,8,8) auto-resolves to ZCYX; force CZYX -> moveaxis -> (2,5,8,8).
    p = _write(tmp_path, (5, 2, 8, 8), "amb.tif")
    tiler = ZStackTiler(p, tmp_path / "out", target_patch_size=8,
                        overlap=0.25, search_range=0, input_axes="CZYX")
    assert (tiler.Z, tiler.C) == (2, 5)
    assert tiler.input_axes == "CZYX"

"""Tests for the axis canonicalizers in ``packages.epicycle_largedata.axes``
(ported from the scaled-seg engine, added in Step-1 large-data Brick 1).

Both directions are checked for SHAPE and CONTENT FIDELITY (the reordered array
must hold the same planes the input did), plus the no-op / error cases, and that
they are reachable via BOTH the package path and the ``_largedata`` shim.
"""
import numpy as np
import pytest

from packages.epicycle_largedata.axes import (
    canonicalize_to_czyx,
    canonicalize_to_zcyx,
)


def _distinct(shape):
    """An array whose every voxel is unique, so any mis-permutation is caught."""
    return np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)


class TestCanonicalizeToCZYX:
    """Target = channel-first (C, Z, Y, X) — Step 1's display/extract convention."""

    def test_czyx_is_noop(self):
        a = _distinct((2, 4, 5, 6))            # (C, Z, Y, X)
        out = canonicalize_to_czyx(a, "CZYX")
        assert out is a
        assert out.shape == (2, 4, 5, 6)

    def test_zcyx_moves_channel_first(self):
        a = _distinct((4, 2, 5, 6))            # (Z, C, Y, X)
        out = canonicalize_to_czyx(a, "ZCYX")
        assert out.shape == (2, 4, 5, 6)       # (C, Z, Y, X)
        # content fidelity: out[c, z] is the plane a[z, c]
        for c in range(2):
            for z in range(4):
                assert np.array_equal(out[c, z], a[z, c])

    def test_zyx_inserts_leading_channel(self):
        a = _distinct((4, 5, 6))               # (Z, Y, X)
        out = canonicalize_to_czyx(a, "ZYX")
        assert out.shape == (1, 4, 5, 6)       # (C=1, Z, Y, X)
        assert np.array_equal(out[0], a)

    def test_case_insensitive(self):
        a = _distinct((4, 2, 5, 6))
        assert canonicalize_to_czyx(a, "zcyx").shape == (2, 4, 5, 6)

    def test_bad_axes_raises(self):
        with pytest.raises(ValueError):
            canonicalize_to_czyx(_distinct((2, 4, 5, 6)), "TCYX")


class TestCanonicalizeToZCYX:
    """Target = Z-first (Z, C, Y, X) — the scaled-seg engine's convention."""

    def test_zcyx_is_noop(self):
        a = _distinct((4, 2, 5, 6))            # (Z, C, Y, X)
        out = canonicalize_to_zcyx(a, "ZCYX")
        assert out is a

    def test_czyx_moves_z_first(self):
        a = _distinct((2, 4, 5, 6))            # (C, Z, Y, X)
        out = canonicalize_to_zcyx(a, "CZYX")
        assert out.shape == (4, 2, 5, 6)       # (Z, C, Y, X)
        for z in range(4):
            for c in range(2):
                assert np.array_equal(out[z, c], a[c, z])

    def test_zyx_inserts_channel_after_z(self):
        a = _distinct((4, 5, 6))               # (Z, Y, X)
        out = canonicalize_to_zcyx(a, "ZYX")
        assert out.shape == (4, 1, 5, 6)       # (Z, C=1, Y, X)
        assert np.array_equal(out[:, 0], a)

    def test_case_insensitive(self):
        a = _distinct((2, 4, 5, 6))            # (C, Z, Y, X)
        assert canonicalize_to_zcyx(a, "czyx").shape == (4, 2, 5, 6)

    def test_bad_axes_raises(self):
        with pytest.raises(ValueError):
            canonicalize_to_zcyx(_distinct((2, 4, 5, 6)), "QQYX")


def test_czyx_zcyx_are_inverses_on_4d():
    """Round-trip: ZCYX -> CZYX -> ZCYX returns the original layout + content."""
    a = _distinct((4, 2, 5, 6))                # (Z, C, Y, X)
    there = canonicalize_to_czyx(a, "ZCYX")    # (C, Z, Y, X)
    back = canonicalize_to_zcyx(there, "CZYX")  # (Z, C, Y, X)
    assert back.shape == a.shape
    assert np.array_equal(back, a)


def test_reachable_via_largedata_shim():
    """The controller/box-tool import path (`_largedata`) re-exports both."""
    from napari_epicycle.data_preparation._largedata import (
        canonicalize_to_czyx as shim_czyx,
        canonicalize_to_zcyx as shim_zcyx,
    )
    a = _distinct((4, 2, 5, 6))
    assert np.array_equal(shim_czyx(a, "ZCYX"), canonicalize_to_czyx(a, "ZCYX"))
    assert shim_zcyx is canonicalize_to_zcyx

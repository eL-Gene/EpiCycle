"""Tests for the multi-channel intensity-split fix (Add-tab P1·0).

`_preview_split` used to slice a 4D image ``(Z,Y,X,C)`` / ``(C,Z,Y,X)`` with
the 3D ``_split_roi`` and pass the result straight to ``sobel`` →
``seeded_watershed_split``. A 4D ``image_roi`` makes ``sobel`` return a 4D
elevation map, which then mismatches the 3D ``mask`` inside
``watershed(elevation, markers, mask)`` — the split silently breaks on any
multi-channel membrane image.

The fix routes every image-ROI accessor through
``Curation3DAnnotator._squeeze_image_channel`` (channel 0), so the ROI slice
is always 3D. These tests pin both halves of the contract:

1. ``_squeeze_image_channel`` collapses 4D → 3D and passes 2D/3D through.
2. A channel-squeezed 4D image drives a valid intensity-mode watershed split,
   whereas a raw 4D ``image_roi`` does not.

All Qt-free: ``_squeeze_image_channel`` is a ``@staticmethod`` (no instance),
and ``seeded_watershed_split`` is pure NumPy/scipy.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.split import seeded_watershed_split


def _make_cube_label(shape, label_id, zr, yr, xr):
    vol = np.zeros(shape, dtype=np.int32)
    vol[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]] = label_id
    return vol


class TestSqueezeImageChannel:
    """Curation3DAnnotator._squeeze_image_channel — the single channel-collapse."""

    def _sq(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        return Curation3DAnnotator._squeeze_image_channel

    def test_3d_passthrough(self):
        sq = self._sq()
        a = np.zeros((4, 5, 6))
        assert sq(a).shape == (4, 5, 6)
        assert sq(a) is a  # no copy for the common 3D case

    def test_2d_passthrough(self):
        sq = self._sq()
        a = np.zeros((5, 6))
        assert sq(a).shape == (5, 6)

    def test_zyxc_rgb_drops_trailing_channel(self):
        """(Z, Y, X, C) with C in {3,4} → channel 0 via ``[..., 0]``."""
        sq = self._sq()
        a = np.zeros((4, 5, 6, 3))
        a[..., 0] = 7.0
        a[..., 1] = 99.0  # other channels must be ignored
        out = sq(a)
        assert out.shape == (4, 5, 6)
        assert np.all(out == 7.0)

        a4 = np.zeros((4, 5, 6, 4))
        assert sq(a4).shape == (4, 5, 6)

    def test_czyx_leading_channel_takes_first(self):
        """(C, Z, Y, X) (C not in {3,4}) → ``[0]`` leading channel."""
        sq = self._sq()
        a = np.zeros((2, 4, 5, 6))
        a[0] = 3.0
        a[1] = 50.0
        out = sq(a)
        assert out.shape == (4, 5, 6)
        assert np.all(out == 3.0)

    def test_squeeze_then_roi_slice_is_3d(self):
        """The squeeze MUST precede the 3D ROI slice (the actual bug surface)."""
        sq = self._sq()
        roi = (slice(0, 3), slice(0, 4), slice(0, 5))
        # 4D image, squeeze first, then slice with a 3-tuple ROI → 3D.
        img4 = np.random.rand(6, 8, 10, 3).astype(np.float32)
        assert sq(img4)[roi].shape == (3, 4, 5)
        # Bug repro: slicing the raw 4D array with a 3-tuple keeps the channel.
        assert img4[roi].ndim == 4


class TestIntensitySplitMultichannel:
    """End-to-end: squeezed 4D image drives a valid watershed split."""

    def _setup(self):
        snapshot = _make_cube_label((6, 12, 12), 3, (1, 5), (2, 10), (2, 10))
        current = snapshot.copy()
        current[3, 4, 4] = 10
        current[3, 8, 8] = 11
        return snapshot, current

    def test_squeezed_4d_image_splits(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        snapshot, current = self._setup()
        # Multi-channel membrane image (Z, Y, X, C).
        img4 = np.random.rand(6, 12, 12, 3).astype(np.float32)
        image_roi = Curation3DAnnotator._squeeze_image_channel(img4)
        assert image_roi.shape == (6, 12, 12)

        result, new_ids, msg = seeded_watershed_split(
            snapshot, current, 3, method="intensity", image_roi=image_roi
        )
        assert set(new_ids) == {10, 11}
        assert "Split into 2" in msg

    def test_raw_4d_image_breaks_watershed(self):
        """Document the pre-fix failure: a 4D image_roi must NOT reach watershed."""
        snapshot, current = self._setup()
        img4 = np.random.rand(6, 12, 12, 3).astype(np.float32)
        # sobel(4D) → 4D elevation vs 3D mask → watershed raises.
        with pytest.raises(Exception):
            seeded_watershed_split(
                snapshot, current, 3, method="intensity", image_roi=img4
            )

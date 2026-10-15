"""Downsample helper module — Phase D1 (perf push 2026-05-09).

Pure-function helpers for the Step 3 downsample-on-load feature:

  - ``compute_factor(volume_shape, max_bytes, dtype_size)``: smallest
    integer factor that brings the volume below the byte cap (isotropic).
  - ``downsample_image(image, factor)``: smooth average-pool downsample
    via ``skimage.transform.downscale_local_mean``.
  - ``downsample_labels_majority(labels, factor)``: block-wise majority
    pooling.  Preserves label IDs (no averaging across labels).  Output
    label set is a subset of input.
  - ``upsample_labels_nn(labels, factor, target_shape=None)``: nearest-
    neighbor upsample via ``np.repeat``.  Optional target_shape crops
    output to the original full-res shape (handles non-divisible
    inputs).

These tests pin pure-function correctness only — no integration with
controller / state_manager / patch load yet (that's D2).
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# compute_factor
# ---------------------------------------------------------------------------


class TestComputeFactor:
    """Compute smallest integer downsample factor for the given byte cap."""

    def test_small_volume_returns_1(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            compute_factor,
        )
        # 256×256×100 × 2 bytes = 13 MB — well under 256 MB cap.
        assert compute_factor((100, 256, 256)) == 1

    def test_at_threshold_returns_1(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            compute_factor,
        )
        # 512×512×200 × 2 = 105 MB — under 256 MB.
        assert compute_factor((200, 512, 512)) == 1

    def test_user_canonical_1024_returns_2(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            compute_factor,
        )
        # 1024×1024×400 × 2 = 838 MB — needs f=2 → 105 MB.
        assert compute_factor((400, 1024, 1024)) == 2

    def test_huge_volume_returns_3_or_more(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            compute_factor,
        )
        # 2048×2048×800 × 2 = 6.7 GB — needs f=3 → 248 MB OR f=4 → 105 MB.
        f = compute_factor((800, 2048, 2048))
        assert f >= 3
        # Verify result fits the cap.
        downsampled_bytes = 1
        for dim in (800, 2048, 2048):
            downsampled_bytes *= -(-dim // f)  # ceil-div
        downsampled_bytes *= 2
        assert downsampled_bytes <= 256 * 1024 * 1024

    def test_custom_cap(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            compute_factor,
        )
        # 1024×1024×400 with a 64 MB cap (smaller than default 256 MB)
        # forces a larger factor.
        f_default = compute_factor((400, 1024, 1024))
        f_strict = compute_factor((400, 1024, 1024), max_bytes=64 * 1024 * 1024)
        assert f_strict >= f_default

    def test_custom_dtype_size_uint32(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            compute_factor,
        )
        # int16 = 2 bytes; uint32 = 4 bytes — uint32 needs larger factor.
        f_int16 = compute_factor((400, 1024, 1024), dtype_size=2)
        f_uint32 = compute_factor((400, 1024, 1024), dtype_size=4)
        assert f_uint32 >= f_int16

    def test_factor_capped_to_avoid_degenerate(self):
        """Even if the user passes an absurdly small cap, the factor
        should be bounded so we don't downsample to 1×1×1."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            compute_factor,
        )
        # Tiny volume + impossibly small cap — factor still capped.
        f = compute_factor((10, 10, 10), max_bytes=1)
        # min(shape)=10, cap should be at most 10 // 2 = 5.
        assert 1 <= f <= 5


# ---------------------------------------------------------------------------
# downsample_image
# ---------------------------------------------------------------------------


class TestDownsampleImage:
    """3D image average-pool downsampling."""

    def test_factor_1_returns_copy_unchanged(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image,
        )
        img = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
        out = downsample_image(img, 1)
        np.testing.assert_array_equal(out, img)

    def test_factor_2_halves_each_dim(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image,
        )
        img = np.zeros((10, 20, 30), dtype=np.float32)
        out = downsample_image(img, 2)
        assert out.shape == (5, 10, 15)

    def test_uniform_input_uniform_output(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image,
        )
        img = np.full((10, 20, 30), 42.0, dtype=np.float32)
        out = downsample_image(img, 2)
        np.testing.assert_allclose(out, 42.0)

    def test_two_value_input_averaged(self):
        """A simple 2×2×2 block of 1s and 0s averages to 0.5 — verifies
        the smoothing semantics."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image,
        )
        img = np.zeros((2, 2, 2), dtype=np.float32)
        img[0, 0, 0] = 1.0
        out = downsample_image(img, 2)
        assert out.shape == (1, 1, 1)
        np.testing.assert_allclose(out[0, 0, 0], 1.0 / 8.0)

    def test_4d_channel_first_preserves_channel_axis(self):
        """For 4D ``(C, Z, Y, X)`` input, the channel axis is NOT
        downsampled — only spatial axes."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image,
        )
        # 3 channels, spatial 4×8×8
        img = np.zeros((3, 4, 8, 8), dtype=np.float32)
        out = downsample_image(img, 2)
        # Expected: channel preserved (3), spatial halved (2, 4, 4)
        assert out.shape == (3, 2, 4, 4)

    def test_4d_per_channel_independent(self):
        """Each channel downsamples independently; values from one
        channel don't bleed into another's average."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image,
        )
        img = np.zeros((2, 2, 2, 2), dtype=np.float32)
        img[0, 0, 0, 0] = 8.0  # only one voxel in channel 0
        img[1, 1, 1, 1] = 8.0  # only one voxel in channel 1, opposite corner
        out = downsample_image(img, 2)
        # After downsample to (2, 1, 1, 1), each channel's single block
        # should average its 8 spatial voxels independently:
        # Channel 0: 8.0 / 8 = 1.0; channel 1: 8.0 / 8 = 1.0
        np.testing.assert_allclose(out[0, 0, 0, 0], 1.0)
        np.testing.assert_allclose(out[1, 0, 0, 0], 1.0)

    def test_non_divisible_z_matches_label_shape(self):
        """REGRESSION TEST for D2.2 fix (2026-05-10).

        At odd Z dimensions (e.g., 461 with factor=2), skimage's
        downscale_local_mean pads to next multiple → output Z=231 (ceil),
        while downsample_labels_majority uses floor-div → output Z=230.
        That 1-voxel shape mismatch silently aborts the annotator's
        subviewer load via _normalize_labels_volume's shape check.

        This test pins the contract: paired image + labels arrays at the
        SAME input shape produce the SAME output shape after downsample.
        """
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image, downsample_labels_majority,
        )
        # Realistic case from user's 461-slice 512×512 patches.
        image = np.zeros((461, 512, 512), dtype=np.float32)
        labels = np.zeros((461, 512, 512), dtype=np.int32)
        ds_img = downsample_image(image, 2)
        ds_lbl = downsample_labels_majority(labels, 2)
        assert ds_img.shape == ds_lbl.shape, (
            f"Image {ds_img.shape} ≠ Labels {ds_lbl.shape} — "
            f"silent subviewer-load abort guaranteed at this mismatch"
        )
        assert ds_img.shape == (230, 256, 256)

    def test_non_divisible_z_matches_label_shape_4d_image(self):
        """Same shape-match contract for 4D channel-first image input."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image, downsample_labels_majority,
        )
        image = np.zeros((2, 461, 512, 512), dtype=np.float32)
        labels = np.zeros((461, 512, 512), dtype=np.int32)
        ds_img = downsample_image(image, 2)
        ds_lbl = downsample_labels_majority(labels, 2)
        # Spatial axes (skip channel axis 0 of image)
        assert ds_img.shape[1:] == ds_lbl.shape, (
            f"Image spatial {ds_img.shape[1:]} ≠ Labels {ds_lbl.shape}"
        )
        assert ds_img.shape == (2, 230, 256, 256)

    def test_invalid_ndim_raises(self):
        """2D and 5D input should raise — only 3D and 4D supported."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image,
        )
        with pytest.raises(ValueError, match="3D.*or 4D"):
            downsample_image(np.zeros((4, 4)), 2)
        with pytest.raises(ValueError, match="3D.*or 4D"):
            downsample_image(np.zeros((2, 2, 2, 2, 2)), 2)

    # -- Single-channel image cases (most common for grayscale data) --

    def test_3d_single_channel_grayscale(self):
        """Pure 3D grayscale image (Z, Y, X) — no channel axis at all.
        Most common single-channel format; pin the contract explicitly."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image, downsample_labels_majority,
        )
        # Use the user's odd-Z case to also exercise the floor-div crop.
        image = np.random.default_rng(0).random(
            (461, 512, 512), dtype=np.float32
        )
        labels = np.zeros((461, 512, 512), dtype=np.int32)
        ds_img = downsample_image(image, 2)
        ds_lbl = downsample_labels_majority(labels, 2)
        # Spatial axes match between image and labels.
        assert ds_img.shape == ds_lbl.shape
        assert ds_img.shape == (230, 256, 256)
        # Dtype preserved.
        assert ds_img.dtype == np.float32

    def test_4d_single_channel_image(self):
        """4D channel-first image with C=1 (single-channel volumetric).
        Falls through the multi-channel code path; pin explicitly."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image, downsample_labels_majority,
        )
        image = np.random.default_rng(0).random(
            (1, 461, 512, 512), dtype=np.float32
        )
        labels = np.zeros((461, 512, 512), dtype=np.int32)
        ds_img = downsample_image(image, 2)
        ds_lbl = downsample_labels_majority(labels, 2)
        # Channel preserved at C=1; spatial halved.
        assert ds_img.shape == (1, 230, 256, 256)
        # Image spatial axes match labels.
        assert ds_img.shape[1:] == ds_lbl.shape

    def test_3d_single_channel_uniform_value_preserved(self):
        """For grayscale 3D, uniform input → uniform output (same value)."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_image,
        )
        image = np.full((10, 20, 30), 0.42, dtype=np.float32)
        out = downsample_image(image, 2)
        assert out.shape == (5, 10, 15)
        np.testing.assert_allclose(out, 0.42)


# ---------------------------------------------------------------------------
# downsample_labels_majority
# ---------------------------------------------------------------------------


class TestDownsampleLabelsMajority:
    """Block-wise majority pool for label volumes."""

    def test_factor_1_returns_copy(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_labels_majority,
        )
        labels = np.arange(8 * 16 * 16, dtype=np.int32).reshape(8, 16, 16)
        out = downsample_labels_majority(labels, 1)
        np.testing.assert_array_equal(out, labels)
        # Must be a copy, not a view (caller may mutate).
        out[0, 0, 0] = -1
        assert labels[0, 0, 0] != -1

    def test_factor_2_halves_each_dim(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_labels_majority,
        )
        labels = np.zeros((10, 20, 30), dtype=np.int32)
        out = downsample_labels_majority(labels, 2)
        assert out.shape == (5, 10, 15)

    def test_uniform_label_preserved(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_labels_majority,
        )
        labels = np.full((10, 20, 30), 7, dtype=np.int32)
        out = downsample_labels_majority(labels, 2)
        assert (out == 7).all()

    def test_majority_label_wins(self):
        """In a 2×2×2 block where label 5 occupies 5 voxels and label 9
        occupies 3 voxels, majority pool returns 5."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_labels_majority,
        )
        block = np.array(
            [[[5, 5], [5, 5]],
             [[5, 9], [9, 9]]],
            dtype=np.int32,
        )
        out = downsample_labels_majority(block, 2)
        assert out.shape == (1, 1, 1)
        assert out[0, 0, 0] == 5

    def test_output_label_set_is_subset_of_input(self):
        """No new labels invented during downsampling."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_labels_majority,
        )
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 5, size=(10, 20, 30), dtype=np.int32)
        out = downsample_labels_majority(labels, 2)
        in_set = set(np.unique(labels).tolist())
        out_set = set(np.unique(out).tolist())
        assert out_set <= in_set, (
            f"Output introduced labels not in input: {out_set - in_set}"
        )

    def test_dtype_preserved(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_labels_majority,
        )
        for dtype in [np.int32, np.uint32, np.int16, np.uint16]:
            labels = np.zeros((10, 10, 10), dtype=dtype)
            out = downsample_labels_majority(labels, 2)
            assert out.dtype == dtype, f"dtype changed for {dtype}: {out.dtype}"

    def test_non_divisible_shape_cropped(self):
        """If shape isn't divisible by factor, the trailing remainder
        is dropped (output uses floor-div shape)."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_labels_majority,
        )
        # 11×21×31 with factor=2 → output should be 5×10×15.
        labels = np.zeros((11, 21, 31), dtype=np.int32)
        out = downsample_labels_majority(labels, 2)
        assert out.shape == (5, 10, 15)


# ---------------------------------------------------------------------------
# upsample_labels_nn
# ---------------------------------------------------------------------------


class TestUpsampleLabelsNN:
    """Nearest-neighbor upsample for restoring full-res mask on save."""

    def test_factor_1_returns_copy(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            upsample_labels_nn,
        )
        labels = np.arange(8 * 16 * 16, dtype=np.int32).reshape(8, 16, 16)
        out = upsample_labels_nn(labels, 1)
        np.testing.assert_array_equal(out, labels)

    def test_factor_2_doubles_each_dim(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            upsample_labels_nn,
        )
        labels = np.zeros((5, 10, 15), dtype=np.int32)
        out = upsample_labels_nn(labels, 2)
        assert out.shape == (10, 20, 30)

    def test_block_uniform_after_upsample(self):
        """Each f×f×f block in the upsampled output holds the same
        label as the corresponding input voxel."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            upsample_labels_nn,
        )
        labels = np.array(
            [[[1, 2], [3, 4]]],  # shape (1, 2, 2)
            dtype=np.int32,
        )
        out = upsample_labels_nn(labels, 2)
        assert out.shape == (2, 4, 4)
        # The (0:2, 0:2, 0:2) sub-block should all be label 1.
        assert (out[0:2, 0:2, 0:2] == 1).all()
        assert (out[0:2, 0:2, 2:4] == 2).all()
        assert (out[0:2, 2:4, 0:2] == 3).all()
        assert (out[0:2, 2:4, 2:4] == 4).all()

    def test_target_shape_crops_excess(self):
        """target_shape smaller than np.repeat result → crop."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            upsample_labels_nn,
        )
        labels = np.zeros((5, 10, 15), dtype=np.int32)
        # Upsample at f=2 produces 10×20×30, but ask for 9×19×29.
        out = upsample_labels_nn(labels, 2, target_shape=(9, 19, 29))
        assert out.shape == (9, 19, 29)

    def test_target_shape_pads_when_undershoot(self):
        """REGRESSION (D2.3, 2026-05-10): target_shape LARGER than
        np.repeat result must PAD to exactly match, not return short.

        Real-world case: original Z=461, downsample f=2 → 230, upsample
        f=2 → 460.  target_shape=461 must produce 461 (1 voxel of edge-
        padding), not 460 (silent data loss in trailing slice).
        """
        from napari_epicycle.fusion_reconstruction.downsample import (
            upsample_labels_nn,
        )
        labels = np.zeros((230, 256, 256), dtype=np.int32)
        # np.repeat at f=2 → (460, 512, 512); target asks for (461, 512, 512).
        out = upsample_labels_nn(labels, 2, target_shape=(461, 512, 512))
        assert out.shape == (461, 512, 512)

    def test_target_shape_pad_uses_edge_duplicate_for_labels(self):
        """Padding must use ``mode='edge'`` (duplicate last slice),
        NOT zero-pad — labels at the trailing voxel inherit the last
        block's label, not synthetic background."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            upsample_labels_nn,
        )
        # Single-axis case for clarity.  3D shape (3,) at f=2 → (6,);
        # target (7,) needs 1 padded voxel.  Set the input so the last
        # voxel has a known non-zero label.
        labels = np.array([[[1, 2, 3]]], dtype=np.int32)  # shape (1, 1, 3)
        out = upsample_labels_nn(labels, 2, target_shape=(2, 2, 7))
        assert out.shape == (2, 2, 7)
        # First 6 voxels of last axis: NN-upsampled — [1, 1, 2, 2, 3, 3]
        # 7th voxel: edge-duplicate of last → 3 (NOT 0)
        np.testing.assert_array_equal(out[0, 0, :], [1, 1, 2, 2, 3, 3, 3])

    def test_target_shape_crops_and_pads_mixed_axes(self):
        """target_shape that overshoots in one axis and undershoots in
        another should crop one + pad the other in a single call."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            upsample_labels_nn,
        )
        labels = np.zeros((5, 5, 5), dtype=np.int32)
        # np.repeat at f=2 → (10, 10, 10); target overshoots Z (=9 < 10),
        # exact in Y (=10), undershoots X (=11 > 10).
        out = upsample_labels_nn(labels, 2, target_shape=(9, 10, 11))
        assert out.shape == (9, 10, 11)

    def test_dtype_preserved(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            upsample_labels_nn,
        )
        for dtype in [np.int32, np.uint32, np.int16, np.uint16]:
            labels = np.zeros((5, 5, 5), dtype=dtype)
            out = upsample_labels_nn(labels, 2)
            assert out.dtype == dtype, f"dtype changed for {dtype}: {out.dtype}"


# ---------------------------------------------------------------------------
# Round-trip: downsample → upsample
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Downsampling then upsampling at the same factor preserves the
    majority structure block-wise (each f³ voxel block in the upsampled
    output holds the majority label of the original f³ block)."""

    def test_uniform_volume_round_trip_identity(self):
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_labels_majority, upsample_labels_nn,
        )
        labels = np.full((10, 20, 30), 7, dtype=np.int32)
        downsampled = downsample_labels_majority(labels, 2)
        roundtrip = upsample_labels_nn(downsampled, 2,
                                       target_shape=labels.shape)
        np.testing.assert_array_equal(roundtrip, labels)

    def test_block_uniform_round_trip(self):
        """If the input is already block-uniform (each 2×2×2 block holds
        a single label), downsample → upsample round-trips losslessly."""
        from napari_epicycle.fusion_reconstruction.downsample import (
            downsample_labels_majority, upsample_labels_nn,
        )
        # Build a 4×4×4 volume from 2×2×2 uniform blocks.
        labels = np.zeros((4, 4, 4), dtype=np.int32)
        labels[0:2, 0:2, 0:2] = 1
        labels[0:2, 0:2, 2:4] = 2
        labels[2:4, 0:2, 0:2] = 3
        # ... rest stays 0
        downsampled = downsample_labels_majority(labels, 2)
        roundtrip = upsample_labels_nn(downsampled, 2,
                                       target_shape=labels.shape)
        np.testing.assert_array_equal(roundtrip, labels)

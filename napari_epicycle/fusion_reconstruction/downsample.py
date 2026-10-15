"""Downsample helpers for the Step 3 downsample-on-load feature.

Phase D1 (perf push 2026-05-09).  Pure functions; no Qt, no integration
with controller / state_manager / patch load yet — those land in D2.

The Step 3 downsample feature reduces the in-memory curation volume by an
isotropic integer factor when the original is too big for the GPU /
makes interactive ops sluggish.  The canonical full-resolution mask is
preserved on disk:

  - On patch load (D2): read full-res TIFF, downsample image + labels,
    set the downsampled arrays as ``layer.data``.  Curation, refinement,
    QC extraction, drawing all operate on the downsampled volume.
  - On save (D4): NN-upsample the curated downsampled mask back to the
    original full-res shape, write the full-res TIFF to disk.  Anything
    downstream (Step 4, training pair extractor, future classifier)
    reads the canonical full-res mask from disk.

Functions:

  - ``compute_factor(volume_shape, max_bytes, dtype_size)`` — smallest
    integer factor that brings the in-memory volume below the cap.
  - ``downsample_image(image, factor)`` — average-pool downsample
    (``skimage.transform.downscale_local_mean``) for image data.
  - ``downsample_labels_majority(labels, factor)`` — block-wise
    majority pooling for label data (preserves label IDs; output set
    is a subset of input).
  - ``upsample_labels_nn(labels, factor, target_shape)`` — nearest-
    neighbor upsample via ``np.repeat``.  Optional ``target_shape``
    crops the output (handles non-divisible original shapes).

See test_downsample.py for the contract pinned in tests.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------


def compute_factor(
    volume_shape: Tuple[int, ...],
    max_bytes: int = 256 * 1024 * 1024,
    dtype_size: int = 2,
) -> int:
    """Return smallest integer factor f >= 1 such that the downsampled
    volume fits in ``max_bytes`` bytes (assuming ``dtype_size`` bytes
    per voxel).

    Isotropic factor — same f applied to every dimension.

    Capped at ``min(shape) // 2`` to avoid degenerate downsamples
    (e.g. a 10-voxel-thin axis would otherwise collapse to 1 voxel).
    For volumes already smaller than the cap, returns 1.

    Args:
        volume_shape: e.g. ``(Z, Y, X)`` for a 3D volume.
        max_bytes: target byte cap for the downsampled in-memory volume.
            Default 256 MB matches the Step 3 downsample-on-load policy.
        dtype_size: bytes per voxel (2 for int16/uint16, 4 for int32/uint32).
            Default 2 matches the canonical labels dtype after downsample.

    Returns:
        Integer factor >= 1.
    """
    if not volume_shape or max_bytes <= 0:
        return 1

    max_factor = max(1, min(volume_shape) // 2)

    for f in range(1, max_factor + 1):
        # ceil-div per dim — captures the actual downsampled extent
        downsampled_voxels = 1
        for dim in volume_shape:
            downsampled_voxels *= -(-dim // f)  # ceil-div
        if downsampled_voxels * dtype_size <= max_bytes:
            return f

    # Even at max_factor we don't fit — return max_factor anyway (best
    # effort).  Caller may decide to error / warn separately.
    return max_factor


# ---------------------------------------------------------------------------
# Image downsample
# ---------------------------------------------------------------------------


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool downsample a 3D or 4D image by integer factor.

    Wraps ``skimage.transform.downscale_local_mean``.  Smooth — averages
    each f×f×f spatial block to a single output voxel.

    For 3D input ``(Z, Y, X)``: downsamples all 3 axes isotropically.
    For 4D input ``(C, Z, Y, X)``: downsamples spatial axes (Z, Y, X)
    only; channel axis is preserved at full size.  This matches
    napari's channel-first convention (``add_image(stack, channel_axis=0)``).

    Args:
        image: ndarray of any numeric dtype.  Preserves dtype on output.
            Must be 3D or 4D.
        factor: integer downsample factor >= 1.

    Returns:
        Downsampled image.  Shape is ``ceil(dim/f)`` per spatial
        dimension by skimage convention (pads with edge value if shape
        isn't divisible).  Channel axis (if present) preserved at
        original size.
    """
    if factor <= 1:
        return image.copy()

    from skimage.transform import downscale_local_mean

    # D2.2 (perf push 2026-05-10): floor-div crop BEFORE downscale_local_mean.
    # skimage's downscale_local_mean pads non-divisible spatial dims to the
    # next multiple of `factor` (effectively ceil-div with edge padding) —
    # but downsample_labels_majority uses floor-div + crop trailing remainder.
    # Without this matching crop, image + labels from the same odd-Z volume
    # come out with shapes that differ by 1 in any non-divisible axis,
    # which trips _normalize_labels_volume's image/labels shape match check
    # in the annotator's _load_selected_layers — silently aborting the
    # subviewer load.  Symptom: black subviewers on first downsample load.
    if image.ndim == 4:
        # Channel-first: leave axis 0 alone, downsample (Z, Y, X)
        new_z = (image.shape[1] // factor) * factor
        new_y = (image.shape[2] // factor) * factor
        new_x = (image.shape[3] // factor) * factor
        cropped = image[:, :new_z, :new_y, :new_x]
        block_size = (1, factor, factor, factor)
    elif image.ndim == 3:
        new_z = (image.shape[0] // factor) * factor
        new_y = (image.shape[1] // factor) * factor
        new_x = (image.shape[2] // factor) * factor
        cropped = image[:new_z, :new_y, :new_x]
        block_size = (factor, factor, factor)
    else:
        raise ValueError(
            f"downsample_image expects 3D (Z,Y,X) or 4D (C,Z,Y,X) input, "
            f"got shape {image.shape}"
        )

    out = downscale_local_mean(cropped, block_size)
    return out.astype(image.dtype, copy=False)


# ---------------------------------------------------------------------------
# Label downsample (majority pooling)
# ---------------------------------------------------------------------------


def downsample_labels_majority(labels: np.ndarray, factor: int) -> np.ndarray:
    """Block-wise approximate majority pool for a 3D label array.

    For each f×f×f block in ``labels``, the output voxel is the value at
    the median position of the sorted block — a fast vectorized
    approximation of strict mode.

    For block size N = factor³ and a value V occurring with strict
    majority (count > N/2 in the block), V is GUARANTEED to be at
    position N//2 of the sorted block.  Returned in those cases.

    For tied / no-strict-majority cases (e.g., 4-2-2 or 3-3-2 splits),
    the median-position value is one of the labels present in the block
    but not necessarily THE most common.  Acceptable for downsample-
    for-display: the result still preserves label IDs (never invents
    new ones), output label set ⊆ input set, and very-thin labels
    (count = 1 in a 2×2×2 block) reliably get pooled away.

    Pre-2026-05-09 implementation used ``scipy.stats.mode`` which gave
    truly-correct mode but ran in Python per row — measured 981 s on a
    1024×1024×400 int32 volume at f=2, vs sub-second for this version.

    Args:
        labels: 3D integer array of label IDs.
        factor: integer downsample factor >= 1.

    Returns:
        Downsampled label array with shape
        ``(Z // factor, Y // factor, X // factor)``.  Trailing remainder
        (when shape isn't divisible by factor) is dropped.
    """
    if factor <= 1:
        return labels.copy()

    if labels.ndim != 3:
        raise ValueError(
            f"downsample_labels_majority expects 3D input, got shape {labels.shape}"
        )

    z, y, x = labels.shape
    new_z = (z // factor) * factor
    new_y = (y // factor) * factor
    new_x = x // factor * factor
    cropped = labels[:new_z, :new_y, :new_x]

    out_shape = (new_z // factor, new_y // factor, new_x // factor)

    # Reshape so each block of factor**3 voxels is contiguous along the
    # last axis: shape becomes (n_blocks, factor**3).
    blocks = cropped.reshape(
        out_shape[0], factor,
        out_shape[1], factor,
        out_shape[2], factor,
    ).transpose(0, 2, 4, 1, 3, 5).reshape(-1, factor ** 3)

    # Vectorized via np.partition — selects the value at the given
    # position as if the row were fully sorted, but cheaper.  For
    # block size N, position N//2 is guaranteed to hold the strict
    # majority value when one exists; for ties, it holds A label
    # present in the block (no invented labels).
    median_idx = factor ** 3 // 2
    modes = np.partition(blocks, median_idx, axis=1)[:, median_idx]

    return modes.reshape(out_shape).astype(labels.dtype, copy=False)


# ---------------------------------------------------------------------------
# Label upsample (nearest-neighbor)
# ---------------------------------------------------------------------------


def upsample_labels_nn(
    labels: np.ndarray,
    factor: int,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Nearest-neighbor upsample a 3D label array by integer factor.

    Each input voxel becomes an f×f×f block of identical values in the
    output.  Implemented via ``np.repeat`` (no interpolation).

    When ``target_shape`` is provided, the output is FORCED to exactly
    that shape:
      - Axes where ``np.repeat`` produces too many voxels: cropped.
      - Axes where ``np.repeat`` produces too few voxels (non-divisible
        original shape — e.g., original Z=461, downsample f=2 → 230,
        upsample → 460, target=461 needs 1 extra): padded with
        ``np.pad(..., mode='edge')`` which duplicates the last slice.

    Edge-duplicate is the correct behavior for labels — the trailing
    voxel inherits the last block's label, not background-zero.  Used
    by the save-lift path so the on-disk TIFF EXACTLY matches the
    canonical full-res shape that was loaded (no data loss for non-
    divisible dimensions).

    Args:
        labels: 3D integer array of label IDs.
        factor: integer upsample factor >= 1.
        target_shape: optional ``(Z, Y, X)`` to force-fit output to.

    Returns:
        Upsampled label array.  Dtype preserved.  Shape exactly equals
        ``target_shape`` when provided, else ``labels.shape * factor``.
    """
    if factor <= 1:
        out = labels.copy()
    else:
        # np.repeat per axis is the canonical NN upsample.  Each call
        # returns a contiguous array — no fancy indexing.
        out = labels
        for axis in range(labels.ndim):
            out = np.repeat(out, factor, axis=axis)

    if target_shape is not None:
        # Pass 1: crop axes where np.repeat overshot target.
        crop_slices = tuple(
            slice(0, min(out.shape[i], target_shape[i]))
            for i in range(out.ndim)
        )
        out = out[crop_slices]
        # Pass 2: pad axes where np.repeat undershot target (non-
        # divisible original shape).  Edge mode duplicates the
        # trailing slice/row/column rather than padding zeros, so the
        # last in-memory voxel's label propagates into the missing
        # full-res tail (correct for labels — no synthetic background).
        pad_widths = tuple(
            (0, max(0, target_shape[i] - out.shape[i]))
            for i in range(out.ndim)
        )
        if any(p[1] > 0 for p in pad_widths):
            out = np.pad(out, pad_widths, mode="edge")

    return out

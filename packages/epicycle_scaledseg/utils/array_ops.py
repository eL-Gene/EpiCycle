"""
Array operations for EpiCycle ScaledSegmentation.

Functions for manipulating Z-stack arrays: axis permutations, channel
splitting/combining, orthoview generation, and RGB conversion.
"""

import os
import numpy as np
from typing import List, Dict


def recover_original_dimensions(
    region_stack: np.ndarray,
    z_range: tuple,
    original_z_dim: int,
) -> np.ndarray:
    """
    Embed a truncated subvolume back into a full-Z zero-initialized volume.

    Parameters
    ----------
    region_stack : array of shape (Zt, ...)
    z_range : (start_z, end_z)
    original_z_dim : total Z slices of the original stack

    Returns
    -------
    Array of shape (original_z_dim, ...) with region_stack inserted at z_range.
    """
    recovered = np.zeros(
        (original_z_dim, *region_stack.shape[1:]), dtype=region_stack.dtype
    )
    start_z, end_z = z_range
    recovered[start_z:end_z] = region_stack
    return recovered


def split_channels(stack: np.ndarray) -> List[np.ndarray]:
    """
    Split a multi-channel stack into per-channel arrays.

    Parameters
    ----------
    stack : (Z, C, X, Y)

    Returns
    -------
    List of arrays, each (Z, X, Y).
    """
    return [stack[:, c, :, :] for c in range(stack.shape[1])]


def permute_axes(zstack: np.ndarray, target_axes: str = 'xy') -> np.ndarray:
    """
    Reorder axes of a (Z, X, Y) stack to match an orthogonal view.

    Parameters
    ----------
    zstack : (Z, X, Y)
    target_axes : 'xy', 'xz', or 'yz'

    Returns
    -------
    Reoriented array.
    """
    if target_axes == 'xz':
        return zstack.transpose(1, 0, 2)   # (X, Z, Y)
    elif target_axes == 'yz':
        return zstack.transpose(2, 0, 1)   # (Y, Z, X)
    return zstack  # 'xy' — no change


def combine_channels(channel_list: List[np.ndarray]) -> np.ndarray:
    """
    Stack per-channel arrays into a single 4-D array.

    Parameters
    ----------
    channel_list : list of (Z, X, Y) arrays

    Returns
    -------
    (Z, C, X, Y) uint8 array.
    """
    return np.stack(channel_list, axis=1).astype(np.uint8)


def get_orthoview(stack: np.ndarray, ortho_view: str) -> np.ndarray:
    """
    Generate a multi-channel stack for a specified orthogonal view.

    Parameters
    ----------
    stack : (Z, C, X, Y)
    ortho_view : 'xy', 'xz', or 'yz'

    Returns
    -------
    (Z', C, X', Y') oriented stack.
    """
    return combine_channels(
        [permute_axes(ch, ortho_view) for ch in split_channels(stack)]
    )


def convert_to_rgb_stack(zstack: np.ndarray) -> np.ndarray:
    """
    Convert a 2-channel stack to RGB format for CellPose / visualisation.

    Maps channel 0 (DAPI) → R, channel 1 (signal) → G, zeros → B.
    Each slice is independently normalised to [0, 255].

    Parameters
    ----------
    zstack : (Z, 2, X, Y)

    Returns
    -------
    (Z, X, Y, 3) uint8 array.
    """
    Z, C, X, Y = zstack.shape
    if C != 2:
        raise ValueError(
            f"Input stack must have exactly 2 channels, got {C}."
        )

    rgb_images = []
    for z in range(Z):
        ch1 = zstack[z, 0].astype(np.float64)
        ch2 = zstack[z, 1].astype(np.float64)

        r1 = ch1.max() - ch1.min()
        r2 = ch2.max() - ch2.min()

        ch1 = (ch1 - ch1.min()) / r1 * 255 if r1 > 0 else np.zeros_like(ch1)
        ch2 = (ch2 - ch2.min()) / r2 * 255 if r2 > 0 else np.zeros_like(ch2)

        rgb = np.stack([ch1, ch2, np.zeros_like(ch1)], axis=-1).astype(np.uint8)
        rgb_images.append(rgb)

    return np.array(rgb_images)


def get_tile_zranges_from_filenames(
    tile_dir: str, extension: str = ".tif"
) -> Dict[str, tuple]:
    """
    Parse Z-ranges from filenames matching ``*_startZ_endZ.ext``.

    Parameters
    ----------
    tile_dir : directory containing tile files
    extension : file extension filter

    Returns
    -------
    Dict mapping filename → (start_z, end_z).
    """
    zrange_dict = {}
    for fname in os.listdir(tile_dir):
        if fname.endswith(extension):
            try:
                parts = fname.split('.')[0].split('_')
                zrange_dict[fname] = tuple(map(int, parts[-2:]))
            except Exception:
                pass
    return zrange_dict


def per_view_sigmas(sigma_perp: float, sigma_inplane: float):
    """Per-view anisotropic Gaussian sigmas for pre-smoothing.

    Each per-view array is ``(Z, Y, X)``. The axis PERPENDICULAR to a view's
    slicing plane gets ``sigma_perp`` (high, to suppress slice-to-slice
    discontinuities from independent 2D CellPose predictions); the two in-plane
    axes get ``sigma_inplane`` (low). Perpendicular axis per view:

      * XY view -> Z is perpendicular -> ``(perp, in, in)``
      * XZ view -> Y is perpendicular -> ``(in, perp, in)``
      * YZ view -> X is perpendicular -> ``(in, in, perp)``

    Returns ``(sigma_xy, sigma_xz, sigma_yz)``. Mirrors the monorepo
    (packages/epicycle_reconstruction/logic.py:404-406); extracted to a pure
    helper so the perpendicular-axis assignment is unit-testable without a GPU.

    Parameters
    ----------
    sigma_perp : Gaussian sigma along each view's perpendicular axis.
    sigma_inplane : Gaussian sigma along each view's two in-plane axes.

    Returns
    -------
    tuple of three (Z, Y, X) sigma tuples: (sigma_xy, sigma_xz, sigma_yz).
    """
    s_perp, s_in = sigma_perp, sigma_inplane
    sigma_xy = (s_perp, s_in, s_in)   # XY view: perpendicular = Z
    sigma_xz = (s_in, s_perp, s_in)   # XZ view: perpendicular = Y
    sigma_yz = (s_in, s_in, s_perp)   # YZ view: perpendicular = X
    return sigma_xy, sigma_xz, sigma_yz


def smooth_flows_3d(
    flows: np.ndarray,
    probs: np.ndarray = None,
    sigma: tuple = (1.0, 1.0, 1.0),
    mask_threshold: float = None,
    soft_weighting: bool = False,
) -> np.ndarray:
    """
    Gaussian-smooth 3-D flow fields, optionally masked/weighted by probabilities.

    Parameters
    ----------
    flows : (C, Z, Y, X) gradient field
    probs : (Z, Y, X) probability map (optional)
    sigma : per-axis Gaussian sigma
    mask_threshold : if set, zero out flows where sigmoid(probs) < threshold
    soft_weighting : multiply flows by sigmoid(probs)

    Returns
    -------
    Smoothed flow array of same shape.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.special import expit

    flows_smooth = np.zeros_like(flows)
    for c in range(flows.shape[0]):
        flows_smooth[c] = gaussian_filter(flows[c], sigma=sigma)

    if probs is not None:
        prob_sigmoid = expit(probs)
        if mask_threshold is not None:
            mask = (prob_sigmoid >= mask_threshold).astype(np.float32)
            flows_smooth *= mask[np.newaxis, ...]
        if soft_weighting:
            flows_smooth *= prob_sigmoid[np.newaxis, ...]

    return flows_smooth


def smooth_probs_3d(
    probs: np.ndarray,
    sigma: tuple = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Gaussian-smooth a 3-D probability volume with anisotropic sigma.

    Designed for per-view pre-smoothing before USegment3D aggregation.
    Use a high sigma along the axis perpendicular to the slicing plane
    to suppress slice-to-slice discontinuities from independent 2D
    CellPose predictions.

    Parameters
    ----------
    probs : (Z, Y, X) probability map
    sigma : per-axis Gaussian sigma (Z, Y, X)

    Returns
    -------
    Smoothed probability array of same shape.
    """
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(probs, sigma=sigma)

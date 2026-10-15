"""
Downsampling utilities for large 3D volumes.

Provides max-pool downsampling (preserves foreground edges in binary masks)
and nearest-neighbor downsampling (preserves label IDs in segmentation masks).
"""

import numpy as np


def maxpool_downsample_3d(vol: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsample a 3D binary volume using max-pooling.

    Max-pooling preserves any foreground voxel within each pooling window,
    preventing thin structures from being lost during downsampling.

    Parameters
    ----------
    vol : 3D array (Z, Y, X), typically uint8 or bool
    factor : downsampling factor (default 2 = half resolution)

    Returns
    -------
    Downsampled array with shape (Z//factor, Y//factor, X//factor).
    """
    Z, Y, X = vol.shape

    # Trim to be divisible by factor
    Z_t = (Z // factor) * factor
    Y_t = (Y // factor) * factor
    X_t = (X // factor) * factor
    trimmed = vol[:Z_t, :Y_t, :X_t]

    # Reshape into pooling blocks and take max
    reshaped = trimmed.reshape(
        Z_t // factor, factor,
        Y_t // factor, factor,
        X_t // factor, factor,
    )
    return reshaped.max(axis=(1, 3, 5))


def nearest_downsample_3d(vol: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsample a 3D volume using nearest-neighbor sampling.

    Appropriate for label masks where interpolation between label IDs
    is not meaningful.

    Parameters
    ----------
    vol : 3D array (Z, Y, X)
    factor : downsampling factor (default 2 = half resolution)

    Returns
    -------
    Downsampled array with shape (Z//factor, Y//factor, X//factor).
    """
    return vol[::factor, ::factor, ::factor].copy()

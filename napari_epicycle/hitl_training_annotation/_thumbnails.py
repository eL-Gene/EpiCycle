"""Phase 8e/8f/8c Tier 3 — shared thumbnail rendering for Import dialogs.

Renders small QIcons of image + mask pairs for the pair-selection tables.
Used by both ``_import_pairs_dialog`` (cross-run cherry-pick) and
``_import_external_pairs_dialog`` (external pair import).

Design:
* TIFF read via ``tifffile`` (header-only metadata access is fast; full
  decode of a 1024² page is ~10-20ms on local SSD).
* Downsample with ``scipy.ndimage.zoom`` if available, else slice-stride
  (NumPy-only fallback).
* For multi-channel images: pick channel 0 + render as grayscale.
* For masks: random colormap per label so labels are visually distinct.
* QImage assembled from numpy bytes — no on-disk PNG roundtrip.

Performance budget:
* 50 pairs @ 10-20ms each → ~0.5-1.0s — acceptable on dialog open
* 200+ pairs → cache (added 2026-05-25 Group G) makes rescans free;
  cold-cache initial render is ~2-4s for 200 pairs

Cache (added 2026-05-25 Group G):
* Module-level LRU bounded at 512 entries (~36MB max).
* Key includes the file mtime_ns + size + channel_axis so:
  - Replacing a file on disk invalidates its entry on next call
  - Toggling the channel-axis radio in the dialog correctly forces a
    fresh decode (each axis produces a different thumbnail)
* Reused freely across dialog opens within one napari session; lost
  on app restart (persistent disk cache deferred).

Backward-compatible: a missing file or unreadable TIFF returns a small
placeholder icon, never raises.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QImage, QPixmap


_DEFAULT_THUMB_SIZE = 48  # px (square)

# Cache size: 512 entries × ~9KB/icon (48×48 RGB) ≈ 4.5MB.  Bounded so
# repeated rescans of large folders don't grow memory indefinitely.  At
# steady state a single folder of 200 pairs (img + mask) uses 400
# entries; 512 covers that plus 100 entries of churn-room.
_THUMBNAIL_CACHE_SIZE = 512


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def make_image_thumbnail(
    img_path: str,
    size: int = _DEFAULT_THUMB_SIZE,
    channel_axis: str = "first",
) -> Optional[QIcon]:
    """Render a thumbnail QIcon for an image TIFF.

    Args:
        img_path: path to TIFF.
        size: thumbnail edge in pixels (square).
        channel_axis: ``"first"``, ``"last"``, or ``"none"``.  Determines
            how a 3D source array is interpreted before picking channel 0.

    Returns:
        QIcon on success, placeholder icon on read failure.  Defensive —
        never raises.

    Cache: keyed by ``(path, mtime_ns, size, channel_axis)``.  Same
    file + same parameters returns the cached icon instantly.  File
    mutation (mtime change) invalidates the entry on next call.
    """
    try:
        st = os.stat(img_path)
        mtime_ns = st.st_mtime_ns
    except OSError:
        return _placeholder_icon(size)
    return _cached_image_thumbnail(
        str(img_path), int(mtime_ns), int(size), str(channel_axis),
    )


def make_mask_thumbnail(
    mask_path: str,
    size: int = _DEFAULT_THUMB_SIZE,
) -> Optional[QIcon]:
    """Render a thumbnail QIcon for a label-mask TIFF.

    Masks are colored with a perceptually-distinct palette (one color
    per instance label).  Background (label=0) renders as black.

    Args:
        mask_path: path to TIFF.
        size: thumbnail edge in pixels (square).

    Returns:
        QIcon on success, placeholder icon on read failure.

    Cache: keyed by ``(path, mtime_ns, size)``.
    """
    try:
        st = os.stat(mask_path)
        mtime_ns = st.st_mtime_ns
    except OSError:
        return _placeholder_icon(size)
    return _cached_mask_thumbnail(
        str(mask_path), int(mtime_ns), int(size),
    )


def clear_thumbnail_cache() -> None:
    """Drop all cached thumbnails.

    Intended for test isolation (test fixtures call this between tests
    so cache state doesn't leak across cases).  Production code shouldn't
    need this — LRU eviction handles steady-state growth + mtime keys
    handle file-mutation invalidation.
    """
    _cached_image_thumbnail.cache_clear()
    _cached_mask_thumbnail.cache_clear()


def thumbnail_cache_info() -> Tuple[int, int, int]:
    """Return (hits, misses, currsize) for diagnostic / test purposes.

    Returns the sum across the image + mask caches.
    """
    img_info = _cached_image_thumbnail.cache_info()
    mask_info = _cached_mask_thumbnail.cache_info()
    return (
        img_info.hits + mask_info.hits,
        img_info.misses + mask_info.misses,
        img_info.currsize + mask_info.currsize,
    )


# ----------------------------------------------------------------------------
# Cache layer
# ----------------------------------------------------------------------------

@lru_cache(maxsize=_THUMBNAIL_CACHE_SIZE)
def _cached_image_thumbnail(
    path_str: str,
    mtime_ns: int,
    size: int,
    channel_axis: str,
) -> QIcon:
    """LRU-cached impl.  Args MUST be hashable (str, int, str)."""
    arr = _read_2d_array_for_thumbnail(path_str, channel_axis=channel_axis)
    if arr is None:
        return _placeholder_icon(size)
    # Normalize to uint8 grayscale for display
    arr8 = _normalize_to_uint8(arr)
    # Downsample to target size
    arr_thumb = _downsample(arr8, size)
    return _icon_from_grayscale(arr_thumb)


@lru_cache(maxsize=_THUMBNAIL_CACHE_SIZE)
def _cached_mask_thumbnail(
    path_str: str,
    mtime_ns: int,
    size: int,
) -> QIcon:
    """LRU-cached impl.  Args MUST be hashable (str, int, int)."""
    arr = _read_2d_array_for_thumbnail(path_str, channel_axis="none")
    if arr is None:
        return _placeholder_icon(size)
    arr_thumb = _downsample_labels(arr, size)
    return _icon_from_labels(arr_thumb)


# ----------------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------------

def _read_2d_array_for_thumbnail(
    path: str, channel_axis: str = "first"
) -> Optional[np.ndarray]:
    """Read a TIFF and return a 2D array suitable for thumbnailing.

    For multi-channel arrays, picks channel 0.  For 4D+ arrays, picks
    the first slice along the leading axes.
    """
    try:
        import tifffile
        arr = tifffile.imread(path)
    except Exception:
        return None
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim < 2:
        return None
    if arr.ndim == 2:
        return arr
    # 3D: depends on channel_axis
    if arr.ndim == 3:
        if channel_axis == "first":
            return arr[0]
        elif channel_axis == "last":
            return arr[..., 0]
        else:  # "none" but ndim=3 — could be (Z, H, W) volumetric.  Pick middle slice.
            return arr[arr.shape[0] // 2]
    # 4D+: pick first slice along leading axes, then first channel
    while arr.ndim > 3:
        arr = arr[0]
    # Now 3D — apply channel_axis logic
    if channel_axis == "first":
        return arr[0]
    elif channel_axis == "last":
        return arr[..., 0]
    return arr[arr.shape[0] // 2]


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip + normalize 2D array to uint8 [0, 255]."""
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)
    # Robust percentile clip for display (handles outliers gracefully)
    lo = float(np.percentile(arr_f, 1))
    hi = float(np.percentile(arr_f, 99))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (arr_f - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _downsample(arr: np.ndarray, target_size: int) -> np.ndarray:
    """Downsample 2D array to target_size × target_size using stride slicing.

    Stride-slicing is fast + good enough for tiny thumbnails.  scipy zoom
    would be slightly higher quality but adds a dep+ ~3x slower.
    """
    h, w = arr.shape[:2]
    if h <= target_size and w <= target_size:
        return arr.copy()
    step_h = max(1, h // target_size)
    step_w = max(1, w // target_size)
    return arr[::step_h, ::step_w]


def _downsample_labels(arr: np.ndarray, target_size: int) -> np.ndarray:
    """Downsample label array preserving label values (no interpolation).

    Same as ``_downsample`` (stride-slicing already preserves discrete
    label values since no averaging happens).
    """
    return _downsample(arr, target_size)


def _icon_from_grayscale(arr: np.ndarray) -> QIcon:
    """Build a grayscale QIcon from a uint8 2D array."""
    h, w = arr.shape[:2]
    # QImage expects bytes with row stride; np arrays may not be contiguous
    arr_c = np.ascontiguousarray(arr.astype(np.uint8))
    qimg = QImage(arr_c.tobytes(), w, h, w, QImage.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qimg.copy())  # .copy() to detach from numpy buffer
    return QIcon(pixmap)


def _icon_from_labels(arr: np.ndarray) -> QIcon:
    """Build a colored QIcon from a label array (each instance → color)."""
    h, w = arr.shape[:2]
    labels = arr.astype(np.int64)
    max_label = int(labels.max()) if labels.size else 0
    if max_label == 0:
        # All background — render as black square
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        # Build a colormap: stable random colors keyed by label id
        # (so same label always → same color across thumbnails)
        rng = np.random.RandomState(42)
        lut = rng.randint(40, 255, size=(max_label + 1, 3), dtype=np.uint8)
        lut[0] = (0, 0, 0)  # background = black
        rgb = lut[labels]
    # QImage needs row stride; ensure contiguous
    rgb_c = np.ascontiguousarray(rgb)
    qimg = QImage(rgb_c.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg.copy())
    return QIcon(pixmap)


def _placeholder_icon(size: int) -> QIcon:
    """Solid-gray placeholder for unreadable / missing TIFFs."""
    arr = np.full((size, size), 90, dtype=np.uint8)
    return _icon_from_grayscale(arr)


__all__ = [
    "make_image_thumbnail",
    "make_mask_thumbnail",
    "clear_thumbnail_cache",
    "thumbnail_cache_info",
]

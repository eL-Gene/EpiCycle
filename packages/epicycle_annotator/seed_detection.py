"""EDT-based automatic seed detection for assisted split.

Pure algorithm module — no Qt, no napari.  All heavy dependencies
(scipy, skimage) are imported lazily inside functions.

Algorithm overview (detect_seed_peaks):
  1. ``mask = (volume_roi == label_id)`` — binary mask of the target label
  2. ``edt = distance_transform_edt(mask)`` — distance from boundary
  3. ``peaks = peak_local_max(edt, min_distance=, threshold_abs=)``
  4. Sort peaks by EDT value descending, assign sequential seed_ids from 1
  5. Generate seed marker volume via ``peaks_to_seed_volume()``
"""

from __future__ import annotations

import numpy as np


def detect_seed_peaks(
    volume_roi: np.ndarray,
    label_id: int,
    *,
    min_distance: int = 5,
    min_edt_value: float = 0.0,
) -> tuple[list[dict], np.ndarray]:
    """Detect seed peaks from EDT of a label mask.

    Parameters
    ----------
    volume_roi : np.ndarray
        3D label volume (or ROI crop).
    label_id : int
        The label ID to detect seeds for.
    min_distance : int
        Minimum distance between peaks (voxels).
    min_edt_value : float
        Minimum EDT value for a peak to be kept.

    Returns
    -------
    peaks : list[dict]
        Each dict: ``{coord: (z,y,x), edt_value: float, seed_id: int}``.
        Sorted by edt_value descending.  seed_ids are sequential from 1.
    seed_markers : np.ndarray
        Same shape as *volume_roi*, with small spheres painted at each
        peak location using the peak's seed_id.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.feature import peak_local_max

    mask = volume_roi == label_id
    if not np.any(mask):
        return [], np.zeros(volume_roi.shape, dtype=np.int32)

    edt = distance_transform_edt(mask).astype(np.float64)

    # Find local maxima in the EDT.
    coords = peak_local_max(
        edt,
        min_distance=min_distance,
        threshold_abs=max(min_edt_value, 1e-6),
        exclude_border=False,
    )

    if len(coords) == 0:
        return [], np.zeros(volume_roi.shape, dtype=np.int32)

    # Build peak dicts with EDT values.
    raw_peaks = []
    for coord in coords:
        z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
        raw_peaks.append({
            "coord": (z, y, x),
            "edt_value": float(edt[z, y, x]),
        })

    # Sort by EDT value descending (largest = deepest interior = best seed).
    raw_peaks.sort(key=lambda p: p["edt_value"], reverse=True)

    # Filter by min_edt_value.
    if min_edt_value > 0:
        raw_peaks = [p for p in raw_peaks if p["edt_value"] >= min_edt_value]

    # Assign sequential seed_ids from 1.
    peaks = []
    for i, p in enumerate(raw_peaks):
        peaks.append({
            "coord": p["coord"],
            "edt_value": p["edt_value"],
            "seed_id": i + 1,
        })

    # Generate seed marker volume.
    markers = peaks_to_seed_volume(volume_roi.shape, peaks)

    return peaks, markers


def threshold_peaks(
    peaks: list[dict],
    min_edt_value: float,
) -> list[dict]:
    """Filter peaks by EDT height.  Keep seed_ids stable (no reassignment).

    Parameters
    ----------
    peaks : list[dict]
        Peaks from ``detect_seed_peaks``.
    min_edt_value : float
        Minimum EDT value.  Peaks below this are removed.

    Returns
    -------
    list[dict]
        Filtered peaks with original seed_ids preserved.
    """
    return [p for p in peaks if p["edt_value"] >= min_edt_value]


def peaks_to_seed_volume(
    volume_shape: tuple[int, ...],
    peaks: list[dict],
    *,
    sphere_radius: int = 2,
) -> np.ndarray:
    """Paint small spheres at each peak location with peak's seed_id.

    Parameters
    ----------
    volume_shape : tuple
        Shape of the output volume.
    peaks : list[dict]
        Each dict must have ``coord`` (z,y,x) and ``seed_id`` (int).
    sphere_radius : int
        Radius of the seed marker spheres (voxels).

    Returns
    -------
    np.ndarray
        Int32 volume with seed markers.
    """
    vol = np.zeros(volume_shape, dtype=np.int32)
    if not peaks:
        return vol

    ndim = len(volume_shape)
    for p in peaks:
        cz, cy, cx = p["coord"]
        sid = p["seed_id"]

        # Compute bounding box clipped to volume.
        z_lo = max(0, cz - sphere_radius)
        z_hi = min(volume_shape[0], cz + sphere_radius + 1)
        y_lo = max(0, cy - sphere_radius)
        y_hi = min(volume_shape[1], cy + sphere_radius + 1)
        x_lo = max(0, cx - sphere_radius)
        x_hi = min(volume_shape[2], cx + sphere_radius + 1)

        # Create local coordinate grids.
        zz, yy, xx = np.ogrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        sphere_mask = dist2 <= sphere_radius ** 2
        vol[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][sphere_mask] = sid

    return vol

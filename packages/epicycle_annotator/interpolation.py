"""SDF-based contour interpolation for 3D label curation.

Ported from napari-multiview-annotator's InterpolationEngine. Uses Signed
Distance Field (SDF) linear interpolation to fill intermediate slices between
user-annotated slices.  Works across XY, XZ, and YZ orientations.

Algorithm:
  1. Extract binary masks from two user-annotated anchor slices.
     The mask includes BOTH user-drawn corrections AND existing label
     data on that slice (reconstruction prior).
  2. Compute SDF for each mask: EDT(outside) - EDT(inside).
  3. Linearly blend SDFs: sdf_interp = (1 - alpha) * sdf_start + alpha * sdf_end.
  4. Threshold: mask_interp = sdf_interp < 0.
  5. Write interpolated voxels using safe-write guard: only overwrite
     background (0) or this label's own voxels — never neighbors.
  6. Post-process: binary_fill_holes + binary_closing on the label's ROI.

Phase K5.1.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
)


def compute_sdf(binary_mask: np.ndarray) -> np.ndarray:
    """Compute signed distance field from a 2D binary mask.

    Returns negative values inside the mask, positive outside.
    """
    return distance_transform_edt(~binary_mask) - distance_transform_edt(binary_mask)


def interpolate_slices_sdf(
    mask_start: np.ndarray,
    mask_end: np.ndarray,
    n_intermediate: int,
) -> list[np.ndarray]:
    """Interpolate *n_intermediate* binary masks between start and end via SDF.

    Returns a list of *n_intermediate* boolean 2D masks.
    """
    if n_intermediate <= 0:
        return []
    sdf_start = compute_sdf(mask_start)
    sdf_end = compute_sdf(mask_end)
    results: list[np.ndarray] = []
    for k in range(1, n_intermediate + 1):
        alpha = k / (n_intermediate + 1)
        sdf_interp = (1.0 - alpha) * sdf_start + alpha * sdf_end
        results.append(sdf_interp < 0)
    return results


def get_slice(volume, orientation, idx):
    """Return a 2D slice from *volume* at *idx* along *orientation*.

    A VIEW, not a copy: writing into it writes into *volume*, which is what
    ``interpolate_label_in_volume`` relies on.  ``None`` for an orientation
    that is not one of "XY" / "XZ" / "YZ".

    Public since REFINE (Brick B2): the serial-2D Active contour
    reads its planes through it, so a module next door does not import a
    private name.  ``_get_slice`` stays, as a one-line delegate.
    """
    if orientation == "XY":
        return volume[idx, :, :]
    elif orientation == "XZ":
        return volume[:, idx, :]
    elif orientation == "YZ":
        return volume[:, :, idx]
    return None


def _get_slice(volume, orientation, idx):
    """The pre-B2 private name -- a delegate to :func:`get_slice`."""
    return get_slice(volume, orientation, idx)


def interpolate_label_in_volume(
    volume: np.ndarray,
    label_id: int,
    edited_slice_indices: list[int],
    orientation: str,
) -> int:
    """Interpolate a single label between edited slices along one orientation.

    Interpolates only between USER-EDITED slices.  The mask at each anchor
    slice includes both user-drawn corrections AND existing label data
    (reconstruction prior), so the SDF blends the full shape — not just
    the user's brush strokes.

    Modifies *volume* in-place.  Returns the number of intermediate slices
    that were filled.
    """
    if len(edited_slice_indices) < 2:
        return 0

    sorted_slices = sorted(set(edited_slice_indices))
    total_interpolated = 0

    for i in range(len(sorted_slices) - 1):
        start_idx = sorted_slices[i]
        end_idx = sorted_slices[i + 1]
        gap = end_idx - start_idx - 1
        if gap <= 0:
            continue

        # Extract 2D binary masks at the annotated slices.
        # These include ALL label voxels (user-drawn + reconstruction).
        sl_start = _get_slice(volume, orientation, start_idx)
        sl_end = _get_slice(volume, orientation, end_idx)
        if sl_start is None or sl_end is None:
            continue

        mask_start = sl_start == label_id
        mask_end = sl_end == label_id

        if not mask_start.any() or not mask_end.any():
            continue

        interpolated = interpolate_slices_sdf(mask_start, mask_end, gap)

        for k, interp_mask in enumerate(interpolated):
            idx = start_idx + k + 1
            target = _get_slice(volume, orientation, idx)
            if target is None:
                continue
            # Only write to background (0) or voxels already belonging to
            # this label — never overwrite neighboring cells.
            safe = interp_mask & ((target == 0) | (target == label_id))
            target[safe] = label_id
            total_interpolated += 1

    return total_interpolated


def postprocess_label(volume: np.ndarray, label_id: int) -> None:
    """Apply hole-filling and binary closing to smooth interpolated results.

    Operates on the bounding-box ROI of the label (not the full volume)
    for performance.  Modifies *volume* in-place.
    """
    coords = np.argwhere(volume == label_id)
    if coords.size == 0:
        return
    # Compute tight bounding box with 1-voxel margin.
    mins = np.maximum(coords.min(axis=0) - 1, 0)
    maxs = np.minimum(coords.max(axis=0) + 2, volume.shape)
    roi = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))
    crop = volume[roi]
    mask = crop == label_id
    structure = generate_binary_structure(3, 2)
    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, structure=structure)
    # Only write to background or self — never overwrite neighboring labels.
    safe = mask & ((crop == 0) | (crop == label_id))
    crop[safe] = label_id


def interpolate_label_all_orientations(
    volume: np.ndarray,
    label_id: int,
    edited_slices: dict,
    post_process: bool = True,
) -> int:
    """Interpolate a label across all orientations that have >= 2 edited slices.

    Interpolation runs between USER-EDITED slices only.  The SDF at each
    anchor slice captures the full label shape (user corrections + existing
    reconstruction data), so the result blends smoothly.

    Parameters
    ----------
    volume : np.ndarray
        3D labels array (Z, Y, X), modified in-place.
    label_id : int
        The label to interpolate.
    edited_slices : dict
        ``{"XY": {label_id: [s1, s2, ...]}, "XZ": {...}, "YZ": {...}}``
        — matches ``LabelSyncManager.edited_slices[layer_name]`` format.
        Label ID keys may be int or numpy integer types.
    post_process : bool
        Apply ``binary_fill_holes`` + ``binary_closing`` after interpolation.

    Returns
    -------
    int
        Total number of intermediate slices filled across all orientations.
    """
    total = 0
    for orientation in ("XY", "XZ", "YZ"):
        orient_dict = edited_slices.get(orientation, {})
        # Handle both int and numpy int keys.
        slices_for_label: list[int] = []
        for key, indices in orient_dict.items():
            if int(key) == int(label_id):
                slices_for_label = [int(s) for s in indices]
                break
        if len(slices_for_label) >= 2:
            total += interpolate_label_in_volume(
                volume, int(label_id), slices_for_label, orientation
            )

    if post_process and total > 0:
        postprocess_label(volume, int(label_id))

    return total

"""Seeded watershed split for 3D label curation.

Pure algorithm module — no Qt, no napari.  All heavy dependencies
(scipy, skimage) are imported lazily inside functions.

Algorithm overview:
  1. ``mask = (snapshot_roi == target_label_id)`` — original label footprint
  2. ``seeds`` = unique IDs painted by user within the mask (not target, not 0)
  3. ``elevation`` = ``-distance_transform_edt(mask)`` [mask mode]
                   = ``sobel(image_roi)`` [intensity mode]
  4. ``result = watershed(elevation, markers, mask=mask)``
  5. Safe-write into volume (never overwrite neighbouring labels)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Seed validation
# ---------------------------------------------------------------------------

def validate_split_seeds(
    snapshot_roi: np.ndarray,
    current_roi: np.ndarray,
    target_label_id: int,
) -> tuple[bool, list[int], str]:
    """Check whether the user has painted enough seeds for a split.

    Parameters
    ----------
    snapshot_roi : np.ndarray
        Snapshot of the label volume ROI taken *before* seed painting.
    current_roi : np.ndarray
        Current state of the label volume ROI (contains user-painted seeds).
    target_label_id : int
        The label ID being split.

    Returns
    -------
    valid : bool
        ``True`` if >= 2 unique seed IDs found within the target mask.
    seed_ids : list[int]
        Sorted list of detected seed label IDs.
    message : str
        Human-readable status / error message.
    """
    mask = snapshot_roi == target_label_id
    if not np.any(mask):
        return False, [], f"Label {target_label_id} not found in snapshot."

    # Seeds are IDs in current_roi that are *within* the original mask
    # and are neither the target label nor background (0).
    ids_in_mask = np.unique(current_roi[mask])
    seed_ids = sorted(
        int(i) for i in ids_in_mask if i != 0 and i != target_label_id
    )

    if len(seed_ids) < 2:
        return (
            False,
            seed_ids,
            f"Need >= 2 seed regions, found {len(seed_ids)}. "
            "Paint more seeds, then try again.",
        )

    return True, seed_ids, f"Found {len(seed_ids)} seeds: {seed_ids}."


# ---------------------------------------------------------------------------
# Seeded watershed split
# ---------------------------------------------------------------------------

def seeded_watershed_split(
    snapshot_roi: np.ndarray,
    current_roi: np.ndarray,
    target_label_id: int,
    *,
    image_roi: np.ndarray | None = None,
    method: str = "mask",
) -> tuple[np.ndarray, list[int], str]:
    """Run seeded watershed to split a label.

    Parameters
    ----------
    snapshot_roi : np.ndarray
        Snapshot of the label volume ROI before seed painting.
    current_roi : np.ndarray
        Current ROI with user-painted seed IDs.
    target_label_id : int
        Original label ID being split.
    image_roi : np.ndarray, optional
        Image intensity data within the ROI.  Required for ``method="intensity"``.
    method : str
        ``"mask"`` — EDT (distance transform) elevation map.
        ``"intensity"`` — Sobel gradient elevation map.

    Returns
    -------
    result : np.ndarray
        Same shape as *snapshot_roi*.  Within the target mask, voxels are
        relabelled with the seed IDs assigned by the watershed.  Outside
        the mask, values are 0.
    new_ids : list[int]
        Sorted list of label IDs present in the result.
    message : str
        Human-readable result description.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed

    mask = snapshot_roi == target_label_id
    if not np.any(mask):
        return (
            np.zeros_like(snapshot_roi),
            [],
            f"Label {target_label_id} not found in snapshot.",
        )

    # --- Validate seeds ---
    valid, seed_ids, msg = validate_split_seeds(
        snapshot_roi, current_roi, target_label_id
    )
    if not valid:
        return np.zeros_like(snapshot_roi), seed_ids, msg

    # --- Build marker array ---
    markers = np.zeros(current_roi.shape, dtype=np.int32)
    for sid in seed_ids:
        markers[current_roi == sid] = sid
    # Ensure markers are only inside the mask.
    markers[~mask] = 0

    # --- Compute elevation map ---
    if method == "intensity" and image_roi is not None:
        from skimage.filters import sobel

        elevation = sobel(image_roi.astype(np.float32))
    else:
        # Default: mask-based distance transform.
        elevation = -distance_transform_edt(mask).astype(np.float32)

    # --- Watershed ---
    labels = watershed(elevation, markers=markers, mask=mask)

    # Build result: non-zero only within original mask.
    result = np.zeros_like(snapshot_roi)
    result[mask] = labels[mask]

    new_ids = sorted(int(i) for i in np.unique(result) if i != 0)
    n_voxels = int(np.sum(mask))
    return (
        result,
        new_ids,
        f"Split into {len(new_ids)} regions ({n_voxels} voxels).",
    )


# ---------------------------------------------------------------------------
# Safe-write helper
# ---------------------------------------------------------------------------

def apply_split_to_volume(
    volume: np.ndarray,
    roi: tuple,
    split_result: np.ndarray,
    target_label_id: int,
) -> int:
    """Write watershed split result into the full volume.

    Uses a **safe-write guard**: only writes where ``volume == target_label_id``
    so neighbouring labels are never overwritten.

    Parameters
    ----------
    volume : np.ndarray
        Full 3D labels volume, modified **in-place**.
    roi : tuple of slice
        ROI slices corresponding to *split_result*.
    split_result : np.ndarray
        Output from :func:`seeded_watershed_split`.
    target_label_id : int
        Original label being split.

    Returns
    -------
    int
        Number of voxels modified.
    """
    volume_roi = volume[roi]
    write_mask = (volume_roi == target_label_id) & (split_result > 0)
    n_modified = int(np.sum(write_mask))
    if n_modified > 0:
        volume_roi[write_mask] = split_result[write_mask]
    return n_modified


# ---------------------------------------------------------------------------
# EDT overlay
# ---------------------------------------------------------------------------

def compute_edt_overlay(
    volume: np.ndarray,
    target_label_id: int,
    *,
    margin: int = 3,
) -> tuple[np.ndarray, tuple[slice, ...], float] | tuple[None, None, None]:
    """Compute the Euclidean Distance Transform for a single label.

    Crops to the label's bounding box first so the EDT runs on a small
    region instead of the full volume.

    Parameters
    ----------
    volume : np.ndarray
        Full labels volume.
    target_label_id : int
        The label to compute EDT for.
    margin : int
        Padding around the bounding box (voxels).

    Returns
    -------
    (edt_crop, roi_slices, edt_max) or (None, None, None)
        ``edt_crop`` is a float32 array covering only the label's bounding
        box.  ``roi_slices`` is a tuple of slices for positioning the crop
        in the full volume coordinate space.  ``edt_max`` is the global
        maximum EDT value (used to pin napari contrast_limits).
    """
    from scipy.ndimage import distance_transform_edt

    coords = np.argwhere(volume == target_label_id)
    if coords.size == 0:
        return None, None, None

    # Tight bounding box + margin.
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    roi = tuple(
        slice(max(0, int(lo) - margin), min(s, int(hi) + margin + 1))
        for lo, hi, s in zip(mins, maxs, volume.shape)
    )

    crop = volume[roi]
    mask = crop == target_label_id
    edt = distance_transform_edt(mask).astype(np.float32)
    edt_max = float(edt.max()) if edt.size > 0 else 0.0
    return edt, roi, edt_max

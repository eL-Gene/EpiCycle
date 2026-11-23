"""SDF-based label smoothing for 3D curation.

Pure algorithm module — no Qt, no napari.  All heavy dependencies
(scipy, skimage) are imported lazily inside functions.

Algorithm overview (smooth_label_sdf):
  1. ``mask = (volume_roi == label_id)`` — binary mask of the target label
  2. ``sdf = EDT(mask) - EDT(~mask)`` — signed distance field (positive inside)
  3. ``sdf_smooth = gaussian_filter(sdf, sigma=(sigma_z, sigma_xy, sigma_xy))``
  4. ``smooth_mask = sdf_smooth > 0`` — threshold to get smoothed binary mask
  5. Topology preservation: if smoothing disconnected the label, dilate skeleton
  6. Safe-write into volume: never overwrite neighbouring labels
"""

from __future__ import annotations

import numpy as np


def smooth_label_sdf(
    volume_roi: np.ndarray,
    label_id: int,
    *,
    sigma_z: float = 3.0,
    sigma_xy: float = 1.5,
    preserve_topology: bool = True,
) -> tuple[np.ndarray, int, str]:
    """Smooth a single label using SDF Gaussian filtering.

    Parameters
    ----------
    volume_roi : np.ndarray
        3D label volume (or ROI crop).  Only ``label_id`` voxels are smoothed;
        all other labels are ignored.
    label_id : int
        The label ID to smooth.
    sigma_z : float
        Gaussian sigma along Z axis (typically higher to reduce staircase).
    sigma_xy : float
        Gaussian sigma along X and Y axes.
    preserve_topology : bool
        If True and smoothing would disconnect the label, dilate the skeleton
        to maintain connectivity.

    Returns
    -------
    result_roi : np.ndarray
        Volume of same shape as ``volume_roi`` containing only ``label_id``
        (smoothed) and 0 elsewhere.  Does NOT contain other labels.
    n_modified : int
        Number of voxels that changed (added + removed).
    message : str
        Human-readable status message.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, label as nd_label

    mask = volume_roi == label_id
    if not np.any(mask):
        return np.zeros_like(volume_roi), 0, f"Label {label_id} not found in ROI."

    # Signed distance field: positive inside, negative outside.
    sdf = distance_transform_edt(mask).astype(np.float64)
    sdf -= distance_transform_edt(~mask).astype(np.float64)

    # Gaussian smoothing with anisotropic sigma.
    sdf_smooth = gaussian_filter(sdf, sigma=(sigma_z, sigma_xy, sigma_xy))

    # Threshold at zero to get the smoothed mask.
    smooth_mask = sdf_smooth > 0

    # Topology preservation: check if smoothing increased component count.
    if preserve_topology:
        _, n_orig = nd_label(mask)
        _, n_smooth = nd_label(smooth_mask)
        if n_smooth > n_orig:
            # Smoothing disconnected the label — dilate skeleton to reconnect.
            from skimage.morphology import binary_dilation
            try:
                from skimage.morphology import skeletonize_3d
                skeleton = skeletonize_3d(mask.astype(np.uint8)) > 0
            except ImportError:
                # skeletonize_3d removed in scikit-image >= 0.21;
                # use skeletonize with method='lee' instead.
                from skimage.morphology import skeletonize
                skeleton = skeletonize(mask, method='lee') > 0
            smooth_mask = smooth_mask | binary_dilation(skeleton)

    # Post-smooth cleanup: keep only the largest connected component.
    # Smoothing can create tiny satellite fragments from threshold noise.
    labeled_arr, n_components = nd_label(smooth_mask)
    if n_components > 1:
        # Find the largest component by voxel count
        component_sizes = np.bincount(labeled_arr.ravel())
        component_sizes[0] = 0  # ignore background
        largest = component_sizes.argmax()
        smooth_mask = labeled_arr == largest

    result = np.zeros_like(volume_roi)
    result[smooth_mask] = label_id

    n_modified = int(np.count_nonzero(smooth_mask != mask))
    if n_modified == 0:
        return result, 0, f"Label {label_id} is already smooth."

    frag_msg = f" ({n_components - 1} fragments removed)" if n_components > 1 else ""
    return result, n_modified, f"Smoothed label {label_id}: {n_modified} voxels changed.{frag_msg}"


def apply_smooth_to_volume(
    volume: np.ndarray,
    roi: tuple[slice, ...],
    smooth_result: np.ndarray,
    label_id: int,
) -> int:
    """Apply a smoothed label result back into the full volume.

    Uses the safe-write pattern: clears old footprint, then writes only
    into empty (zero) voxels to avoid overwriting neighbours.

    Parameters
    ----------
    volume : np.ndarray
        The full 3D label volume (modified in-place).
    roi : tuple of slices
        ROI slices indexing into ``volume``.
    smooth_result : np.ndarray
        The result from ``smooth_label_sdf()`` (same shape as ``volume[roi]``).
    label_id : int
        The label being smoothed.

    Returns
    -------
    n_modified : int
        Number of voxels that changed in the volume.
    """
    target_area = volume[roi]
    old_mask = target_area == label_id

    # Clear old footprint.
    target_area[old_mask] = 0

    # Write new footprint — safe-write: only into empty voxels.
    new_mask = (smooth_result == label_id) & (target_area == 0)
    target_area[new_mask] = label_id

    return int(np.count_nonzero(old_mask != new_mask))


def smooth_split_fragments(
    refine_data: np.ndarray,
    roi: tuple[slice, ...],
    fragment_ids: list[int],
    *,
    sigma_z: float = 3.0,
    sigma_xy: float = 1.5,
    preserve_topology: bool = True,
    keep_object: bool = False,
    object_mask: np.ndarray | None = None,
    marker_labels: np.ndarray | None = None,
) -> tuple[int, str]:
    """Smooth all split fragments within the refine layer ROI.

    Processes each fragment independently.  Safe-write between fragments
    prevents overlap.

    Parameters
    ----------
    refine_data : np.ndarray
        The full refine layer data array (modified in-place).
    roi : tuple of slices
        ROI slices covering the split area.
    fragment_ids : list of int
        Label IDs of the split fragments to smooth.
    sigma_z : float
        Gaussian sigma along Z axis.
    sigma_xy : float
        Gaussian sigma along X and Y axes.
    preserve_topology : bool
        If True, maintain connectivity of each fragment.
    keep_object : bool
        If True, the pieces together keep exactly the voxels they had:
        a piece that would smooth away is left as it is, growth outside
        the object is dropped, and holes are filled from the pieces
        through the object (only zeros inside the object mask are ever
        filled).  If False (the default) each piece is smoothed on its
        own.
    object_mask : np.ndarray or None
        With ``keep_object``, the voxels that make up the object (same
        shape as the ROI).  Default None: the voxels the pieces hold
        when this is called.

    marker_labels : np.ndarray or None
        With ``keep_object``, an array (ROI shape) holding each painted
        marker under its piece id.  Those voxels always stay with their
        own piece.  Default None: no voxel is protected.

    Returns
    -------
    total_modified : int
        Total voxels changed across all fragments.
    message : str
        Human-readable summary.
    """
    total_modified = 0
    roi_data = refine_data[roi]
    ids = [int(f) for f in fragment_ids]
    # With keep_object the object's voxel set is fixed before anything
    # moves: smoothing reshapes the cuts between the pieces, it never
    # adds to the object or takes from it.
    union = None
    before = None
    if keep_object:
        union = (
            np.asarray(object_mask, dtype=bool)
            if object_mask is not None else np.isin(roi_data, ids)
        )
        before = roi_data.copy()

    for fid in ids:
        if not np.any(roi_data == fid):
            continue

        smooth_result, n_mod, _msg = smooth_label_sdf(
            roi_data, fid,
            sigma_z=sigma_z, sigma_xy=sigma_xy,
            preserve_topology=preserve_topology,
        )
        if n_mod == 0:
            continue
        if keep_object and not np.any((smooth_result == fid) & union):
            # Smoothing would erase this piece (a thin slab), or move it
            # wholly outside the object: leave it as it is.
            continue

        # Apply in-place: clear old, write new (safe-write within ROI).
        old_mask = roi_data == fid
        roi_data[old_mask] = 0
        new_mask = (smooth_result == fid) & (roi_data == 0)
        roi_data[new_mask] = fid
        total_modified += n_mod

    if total_modified == 0:
        return 0, "Fragments already smooth."

    if keep_object:
        from skimage.segmentation import watershed

        # Growth outside the object goes; every hole inside it is
        # filled from the neighbouring pieces, flooding THROUGH the
        # object so a piece never reaches across empty space.
        roi_data[~union & np.isin(roi_data, ids)] = 0
        if marker_labels is not None:
            # A painted marker always belongs to its own piece.
            mk = np.isin(marker_labels, ids) & union
            roi_data[mk] = marker_labels[mk]
        owned = np.isin(roi_data, ids) & union
        if np.any(union & (roi_data == 0)) and np.any(owned):
            filled = watershed(
                np.zeros(roi_data.shape, dtype=np.uint8),
                markers=np.where(owned, roi_data, 0),
                mask=union & ((roi_data == 0) | owned),
            )
            fill = union & (roi_data == 0) & (filled > 0)
            roi_data[fill] = filled[fill]
        total_modified = int(np.count_nonzero(before != roi_data))
        if total_modified == 0:
            return 0, "Fragments already smooth."
        return (
            total_modified,
            f"Smoothed {len(ids)} fragments: {total_modified} voxels moved "
            f"between pieces (the object keeps all its voxels).",
        )
    return total_modified, f"Smoothed {len(ids)} fragments: {total_modified} voxels changed."

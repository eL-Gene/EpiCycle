"""Interactive neighbor-discovery helpers for the curation UI.

Lives separately from ``qc_features.py`` (which is for batch QC
quantification) because this module's purpose is interactive single-
label queries — the kind a UI needs when a user clicks "Merge Into..."
on a label and we need to surface adjacent label IDs as merge candidates.

The algorithm (graduated dilation) is the same as the one used in
``qc_features.compute_neighbor_features``; this module provides a
single-label, returns-actual-IDs variant suitable for UI consumption.


"""
from __future__ import annotations

import numpy as np


def get_neighbor_ids(
    labels_3d: np.ndarray,
    label_id: int,
    max_radius: int = 3,
) -> list[dict]:
    """Discover neighbor labels of *label_id* via graduated dilation.

    Iteratively dilates *label_id*'s mask from radius 1 up to
    *max_radius*, recording which other label IDs are first
    encountered at each step. This is the same pattern used by
    ``qc_features.compute_neighbor_features`` for batch QC, but
    returns the actual neighbor IDs instead of aggregate stats —
    suitable for the merge dialog's adjacency table.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        label_id: The label whose neighbors to discover.
        max_radius: Maximum dilation radius in voxels. Default 3 for
            interactive use (covers directly-touching plus 2-voxel
            gaps). The QC pipeline uses ``max_radius=5`` for broader
            characterization; for an interactive merge picker, 3 is
            usually enough — fragments separated by more than 2-3
            voxels are rarely meaningful merge candidates.

    Returns:
        A list of dicts, each describing one neighbor::

            {"label_id": int, "contact_voxels": int, "first_radius": int}

        Sort key: ``(first_radius asc, contact_voxels desc)``.
        Touching labels (first_radius=1) come first; within each
        radius level, larger contact area floats to the top.

        Empty list if *label_id* has no neighbors within *max_radius*
        voxels (truly isolated label, or label not present in the
        volume).

    Raises:
        ValueError: if max_radius < 1.
    """
    if max_radius < 1:
        raise ValueError(f"max_radius must be >= 1, got {max_radius}")

    from scipy.ndimage import binary_dilation, find_objects

    # Locate the label's bounding box. find_objects returns one slice
    # tuple per integer label_id (1-indexed) up to the max value in
    # the array. None for missing labels.
    slices = find_objects(labels_3d.astype(np.int32))
    if label_id < 1 or label_id > len(slices):
        return []
    obj_slice = slices[label_id - 1]
    if obj_slice is None:
        return []

    # Expand bounding box by max_radius so dilation has room to grow.
    expanded = tuple(
        slice(
            max(s.start - max_radius, 0),
            min(s.stop + max_radius, dim),
        )
        for s, dim in zip(obj_slice, labels_3d.shape)
    )
    label_crop = labels_3d[expanded]
    mask_crop = label_crop == label_id

    # Graduated dilation: at each step, look at the *new* voxels
    # added by this round of dilation and see what labels they hit.
    all_neighbors: dict[int, int] = {}   # neighbor_id -> total contact voxels
    first_radius: dict[int, int] = {}    # neighbor_id -> radius first seen
    prev_dilated = mask_crop.copy()

    for r in range(1, max_radius + 1):
        dilated = binary_dilation(prev_dilated)
        new_ring = dilated & ~prev_dilated
        ring_ids = label_crop[new_ring]

        if ring_ids.size > 0:
            unique, counts = np.unique(ring_ids, return_counts=True)
            for uid, cnt in zip(unique, counts):
                uid_int = int(uid)
                if uid_int == 0 or uid_int == int(label_id):
                    continue  # skip background and self
                all_neighbors[uid_int] = (
                    all_neighbors.get(uid_int, 0) + int(cnt)
                )
                if uid_int not in first_radius:
                    first_radius[uid_int] = r

        prev_dilated = dilated

    # Build sorted result.
    # Sort key: (first_radius asc, -contact_voxels) — touching first,
    # larger contact area within each radius level floats to top.
    neighbors = [
        {
            "label_id": nid,
            "contact_voxels": all_neighbors[nid],
            "first_radius": first_radius[nid],
        }
        for nid in all_neighbors
    ]
    neighbors.sort(
        key=lambda d: (d["first_radius"], -d["contact_voxels"])
    )
    return neighbors

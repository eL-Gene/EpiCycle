"""Bounded (ROI-scoped) area + centroid statistics for curation accepts.

Pure numpy helpers — no napari, no Qt.  They exist so the Step-3 accept path
can stop doing full-volume ``data == label_id`` + ``np.argwhere`` scans for
every accepted label / de-novo piece.

Numerical equivalence with the full-volume path
-----------------------------------------------
Both helpers return ``area`` as an exact voxel count and ``centroid`` as
``coordinate_sum / area`` in **float64**.  Every coordinate is a small
non-negative integer and every partial sum stays far below ``2**53``, so the
sums are represented EXACTLY in float64 regardless of accumulation order.
The bounded results are therefore bit-identical to
``np.argwhere(volume == label_id).mean(axis=0)`` — not merely close.

Scope / semantics (pinned by tests)
-----------------------------------
``label_stats_in_roi`` is deliberately **ROI-scoped**: voxels of the same id
that live OUTSIDE the ROI are not counted.  Callers must only use it when the
ROI provably contains the label's whole extent (Step-3 interpolation / smooth
accepts pass the ``_snapshot_label_roi`` ROI, which is the label's bounding box
plus a margin, and the op mutates only inside it).  Merge, whose ROI is the
union bbox of the *absorbed* labels and can miss part of the keep label, must
NOT use it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "label_escapes_roi",
    "label_stats_in_roi",
    "piece_stats_from_write",
]


def _origin(roi_origin, ndim: int):
    if roi_origin is None:
        return (0,) * ndim
    return tuple(int(v) for v in roi_origin)



def label_escapes_roi(sub_volume, label_id, roi_slices, full_shape) -> bool:
    """True when ``label_id`` touches an UNCLIPPED face of the ROI crop.

    A containment tripwire for the ROI-scoped stats path.  An ROI face that
    coincides with the volume boundary is CLIPPED — the label cannot continue
    past it, so touching it is fine.  Any other face touched means voxels may
    lie outside the crop and the caller must fall back to the whole volume.

    Cost is six 2-D face comparisons, negligible next to the crop itself.
    Callers use it as a safety net: the interpolation / smooth ROI provably
    contains the label today, and this turns any future op that breaks that
    into slow-but-correct rather than silently-wrong.
    """
    sub = np.asarray(sub_volume)
    for axis, sl in enumerate(roi_slices):
        if axis >= sub.ndim or sub.shape[axis] == 0:
            continue
        start = int(sl.start or 0)
        stop = int(sl.stop) if sl.stop is not None else int(full_shape[axis])
        faces = []
        if start > 0:
            faces.append(0)                      # low face is not the volume edge
        if stop < int(full_shape[axis]):
            faces.append(sub.shape[axis] - 1)    # high face is not the volume edge
        for index in faces:
            key = [slice(None)] * sub.ndim
            key[axis] = index
            if np.any(sub[tuple(key)] == int(label_id)):
                return True
    return False


def label_stats_in_roi(sub_volume, label_id, roi_origin=None):
    """Area + centroid of ``label_id`` inside an already-cropped sub-volume.

    Parameters
    ----------
    sub_volume : np.ndarray
        The ROI crop of the labels volume (any integer dtype, any ndim).
    label_id : int
        The label to measure.
    roi_origin : sequence of int, optional
        Start coordinate of the crop in the full volume; added back so the
        centroid is expressed in FULL-volume coordinates.  Defaults to zeros.

    Returns
    -------
    (area, centroid) or None
        ``None`` when the label has no voxels in the crop — matching the
        ``if not np.any(vol_mask): return`` early-out of the full-volume path.
        ``centroid`` is a float64 ndarray of length ``sub_volume.ndim``.
    """
    sub = np.asarray(sub_volume)
    mask = sub == int(label_id)
    area = int(np.count_nonzero(mask))
    if area == 0:
        return None
    origin = _origin(roi_origin, sub.ndim)
    coords = np.argwhere(mask)
    # Exact integer sums promoted to float64 (see module docstring).
    sums = coords.sum(axis=0, dtype=np.int64).astype(np.float64)
    sums += np.asarray(origin, dtype=np.float64) * float(area)
    return area, sums / float(area)


def piece_stats_from_write(result_roi, write_mask, roi_origin=None):
    """Per-piece area + centroid for a de-novo Add commit, in one bounded pass.

    ``result_roi`` is the candidate label image over the ROI and ``write_mask``
    the boolean mask of voxels actually written into the shared volume
    (background-only safe-write).  Only written voxels are counted, which is
    exactly what a post-commit full-volume scan of each fresh piece id would
    find — Add ids come from ``max(volume) + 1`` and so exist nowhere else.

    Returns
    -------
    dict
        ``{piece_id: (area, centroid_ndarray)}`` for every non-zero written id.
    """
    result = np.asarray(result_roi)
    mask = np.asarray(write_mask, dtype=bool)
    ndim = result.ndim
    origin = np.asarray(_origin(roi_origin, ndim), dtype=np.float64)

    values = result[mask]
    if values.size == 0:
        return {}
    coords = np.argwhere(mask)  # C-order, aligned with `values`
    # Dense-remap the ids before bincount: label ids can be arbitrarily large
    # (Add uses ``max(volume) + 1``), and binning by raw id would allocate an
    # array sized by the largest id rather than by the piece count.
    unique_ids, inverse = np.unique(values.astype(np.int64, copy=False),
                                    return_inverse=True)
    n_bins = unique_ids.size
    counts = np.bincount(inverse, minlength=n_bins)
    sums = np.empty((n_bins, ndim), dtype=np.float64)
    for axis in range(ndim):
        sums[:, axis] = np.bincount(
            inverse, weights=coords[:, axis].astype(np.float64),
            minlength=n_bins,
        )

    out = {}
    for slot in range(n_bins):
        pid = int(unique_ids[slot])
        if pid == 0:
            continue
        area = int(counts[slot])
        if area == 0:
            continue
        # (local_sum + area*origin) / area — one division, matching the
        # full-volume `argwhere(...).mean(axis=0)` exactly.
        centroid = (sums[slot] + origin * float(area)) / float(area)
        out[pid] = (area, centroid)
    return out

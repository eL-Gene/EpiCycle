"""Pure post-stitch cleanup logic — size + detached-island filtering + dense relabel.

These functions decide WHICH global cells to drop and how to renumber the
survivors, working ENTIRELY from the per-cell provenance rows (``dense_id``,
``voxel_count``, and the ZYX bbox) — they never read the multi-GB label volume.
:class:`~epicycle.core.postprocess_stage.PostprocessStage` applies the resulting
keep-set / relabel LUT (plus the optional per-instance smooth) during its single
streaming write.

Two filters, each independently toggleable by the stage:

* **size** (:func:`select_dropped_by_size`) — drop cells below a voxel-count floor
  (the scattered "confetti" of noise/debris masks).
* **detached islands** (:func:`select_dropped_islands`) — drop cells that fall in a
  small connected component of the low-res cell-occupancy grid (debris physically
  separated from the tissue mass, e.g. masks on a fiber/scratch off in the void).

The island test is the clever cheap one: it rasterizes each surviving cell's bbox
into a downsampled occupancy grid, connected-components it, keeps every component
at least ``min_component_frac`` of the largest (so multi-lobe tissue is safe), and
drops cells whose center lands in a small island — all from the CSV, no volume read.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# Required numeric fields on a provenance row (the rest are passthrough strings).
_INT_FIELDS = (
    "dense_id", "voxel_count",
    "z_min", "z_max", "y_min", "y_max", "x_min", "x_max",
)


def _ri(row: dict, key: str) -> int:
    """Robust int read off a provenance row (CSV values arrive as strings)."""
    v = row.get(key, 0)
    if v == "" or v is None:
        return 0
    return int(float(v))


def select_dropped_by_size(rows: Sequence[dict], min_voxels: int) -> Set[int]:
    """Dense ids of cells with ``voxel_count < min_voxels``.

    ``min_voxels <= 0`` disables the filter (returns the empty set).
    """
    if not min_voxels or int(min_voxels) <= 0:
        return set()
    thr = int(min_voxels)
    return {_ri(r, "dense_id") for r in rows if _ri(r, "voxel_count") < thr}


def select_dropped_by_max_size(rows: Sequence[dict], max_voxels: int) -> Set[int]:
    """Dense ids of cells with ``voxel_count > max_voxels`` (extremely large masks).

    ``max_voxels <= 0`` disables the filter (returns the empty set). This is the
    symmetric twin of :func:`select_dropped_by_size`; it removes giant outliers
    (e.g. a background blob or runaway merge). Use with care + a dry run — a
    genuinely large cell and an over-merged blob look the same by volume alone.
    """
    if not max_voxels or int(max_voxels) <= 0:
        return set()
    thr = int(max_voxels)
    return {_ri(r, "dense_id") for r in rows if _ri(r, "voxel_count") > thr}


def select_dropped_islands(
    rows: Sequence[dict],
    full_shape: Tuple[int, int, int],
    *,
    downsample: int = 8,
    min_component_frac: float = 0.01,
    dilate: int = 1,
    exclude: Optional[Set[int]] = None,
) -> Set[int]:
    """Dense ids of cells sitting in detached debris islands (CSV-only decision).

    Parameters
    ----------
    rows : provenance rows (need ``dense_id``, ``voxel_count``, ZYX bbox).
    full_shape : ``(Z, H, W)`` of the global volume — sets the occupancy grid extent.
    downsample : occupancy grid stride (full-res voxels per grid cell). Bigger =
        coarser/cheaper and bridges sub-cell gaps more aggressively.
    min_component_frac : keep every connected component whose low-res voxel count is
        at least this fraction of the LARGEST component; drop cells in smaller ones.
        Multi-lobe tissue (several large blobs) is preserved.
    dilate : binary-dilation iterations on the occupancy grid before labeling, to
        bridge small gaps between adjacent cells/lobes (0 disables).
    exclude : dense ids already dropped (e.g. by the size filter) — skipped here.

    Returns
    -------
    Set of dense ids to drop as islands. Empty when there is 0/1 component.
    """
    from scipy import ndimage as ndi

    exclude = exclude or set()
    Z, H, W = (int(full_shape[0]), int(full_shape[1]), int(full_shape[2]))
    ds = max(1, int(downsample))
    gz = (Z + ds - 1) // ds
    gy = (H + ds - 1) // ds
    gx = (W + ds - 1) // ds

    survivors = [
        r for r in rows
        if _ri(r, "dense_id") not in exclude and _ri(r, "voxel_count") > 0
    ]
    if not survivors:
        return set()

    occ = np.zeros((gz, gy, gx), dtype=bool)
    for r in survivors:
        z0, z1 = _ri(r, "z_min") // ds, _ri(r, "z_max") // ds
        y0, y1 = _ri(r, "y_min") // ds, _ri(r, "y_max") // ds
        x0, x1 = _ri(r, "x_min") // ds, _ri(r, "x_max") // ds
        occ[
            min(z0, gz - 1):min(z1 + 1, gz),
            min(y0, gy - 1):min(y1 + 1, gy),
            min(x0, gx - 1):min(x1 + 1, gx),
        ] = True

    if dilate and int(dilate) > 0:
        occ = ndi.binary_dilation(occ, iterations=int(dilate))

    lab, n = ndi.label(occ)
    if n <= 1:
        return set()  # everything is one blob — nothing to drop as an island

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    largest = int(sizes.max())
    thr = max(1, int(min_component_frac * largest))
    keep_components = {int(c) for c in np.nonzero(sizes >= thr)[0].tolist() if c != 0}

    dropped: Set[int] = set()
    for r in survivors:
        zc = min(((_ri(r, "z_min") + _ri(r, "z_max")) // 2) // ds, gz - 1)
        yc = min(((_ri(r, "y_min") + _ri(r, "y_max")) // 2) // ds, gy - 1)
        xc = min(((_ri(r, "x_min") + _ri(r, "x_max")) // 2) // ds, gx - 1)
        if int(lab[zc, yc, xc]) not in keep_components:
            dropped.add(_ri(r, "dense_id"))
    return dropped


def build_relabel_lut(all_ids: Iterable[int], dropped_ids: Set[int]) -> Dict[int, int]:
    """Map surviving dense ids -> a dense 1..K renumbering (dropped/background -> 0)."""
    survivors = sorted(int(i) for i in all_ids if int(i) not in dropped_ids)
    return {old: new for new, old in enumerate(survivors, start=1)}


def lut_to_array(lut: Dict[int, int], max_id: int) -> np.ndarray:
    """Dense int32 LUT array indexed by old dense id (0 -> 0; dropped -> 0)."""
    arr = np.zeros(int(max_id) + 1, dtype=np.int32)
    for old, new in lut.items():
        if 0 <= int(old) <= int(max_id):
            arr[int(old)] = int(new)
    return arr


__all__ = [
    "select_dropped_by_size",
    "select_dropped_by_max_size",
    "select_dropped_islands",
    "build_relabel_lut",
    "lut_to_array",
]

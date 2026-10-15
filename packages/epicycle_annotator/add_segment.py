"""De-novo region segmentation for 3D label curation (the "Add" tool).

Pure algorithm module — no Qt, no napari.  All heavy dependencies
(scipy, skimage) are imported lazily inside functions.

The "Add" tool segments a *missed* cell/nucleus from scratch inside a
user-drawn box: threshold an image channel, separate touching pieces with a
distance-watershed, and emit new labels **clipped to background** — existing
labels are never overwritten (Add only ever fills gaps).

Algorithm overview (``add_segment_region``):
  1. optional CLAHE on ``image_roi`` (contrast enhance; works in [0,1])
  2. ``binary = proc >= threshold``         (Otsu or explicit value)
  3. ``fg = binary & (existing_roi == 0)``   — segment ONLY background
  4. if ``separate``: ``peak_local_max(EDT(fg))`` seeds → ``watershed(-EDT)``
     else: connected components
  5. drop pieces ``< min_size`` voxels
  6. relabel survivors sequentially from ``next_label``

Output invariant: ``out`` is nonzero ONLY where ``existing_roi == 0`` (the
clipping rule holds *by construction*, step 3), so ``out`` and any existing
label never share a voxel.

Reuses the same primitives as ``split.py`` (``distance_transform_edt``,
``watershed``) and ``seed_detection.py`` (``peak_local_max``).
"""

from __future__ import annotations

import numpy as np


def _normalize01(image: np.ndarray) -> np.ndarray:
    """Min-max scale a float image to [0, 1].  Constant images map to all-zeros."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float64)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float64)
    return ((image - lo) / (hi - lo)).astype(np.float64)


def _resolve_threshold(proc: np.ndarray, threshold) -> float | None:
    """Resolve ``threshold`` (``"otsu"`` or a number) to a scalar cut value.

    Returns ``None`` for a degenerate image (no finite range → nothing to
    threshold), which the caller treats as "no foreground".
    """
    if isinstance(threshold, str):
        method = threshold.lower()
        if method != "otsu":
            raise ValueError(f"unknown threshold method {threshold!r}")
        finite = proc[np.isfinite(proc)]
        if finite.size == 0 or float(finite.min()) == float(finite.max()):
            return None
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(proc))
    return float(threshold)


def add_threshold_space(image_roi, *, clahe=False, clahe_clip=0.01):
    """The image values the threshold is compared against.

    Raw image when ``clahe`` is off; min-max normalized to [0, 1] then CLAHE
    when on.  Shared by :func:`add_segment_region` (internally) and the Add
    tab's live in-box histogram so the draggable threshold line sits in exactly
    the value space the watershed will threshold — they never disagree.
    """
    image = np.asarray(image_roi, dtype=np.float64)
    if not clahe:
        return image
    from skimage.exposure import equalize_adapthist

    norm = _normalize01(image)
    if not np.any(norm):
        return norm
    return equalize_adapthist(norm, clip_limit=clahe_clip)


def add_segment_region(
    image_roi: np.ndarray,
    existing_roi: np.ndarray,
    *,
    clahe: bool = False,
    clahe_clip: float = 0.01,
    threshold="otsu",
    min_size: int = 0,
    separate: bool = True,
    min_distance: int = 3,
    next_label: int,
    spacing=None,
) -> np.ndarray:
    """Segment new label(s) from an image channel inside a box, clipped to gaps.

    Parameters
    ----------
    image_roi : np.ndarray
        Single-channel image cropped to the box (2D or 3D).  Multichannel
        images MUST be channel-collapsed by the caller first.
    existing_roi : np.ndarray
        Current label volume cropped to the same box.  Foreground is segmented
        ONLY where this is 0, so new pieces never overwrite a neighbour.
    clahe : bool
        Apply CLAHE (``equalize_adapthist``) before thresholding.  When on, the
        image is normalized to [0, 1] and ``threshold`` is interpreted in that
        space; when off, ``threshold`` is in the raw image's value space.
    clahe_clip : float
        CLAHE clip limit (skimage default 0.01).
    threshold : {"otsu"} or float
        ``"otsu"`` auto-thresholds; a number cuts at that value (``proc >= t``).
    min_size : int
        Drop connected pieces smaller than this many voxels (0 = keep all).
    separate : bool
        If True, distance-watershed splits touching pieces; else connected
        components (one label per blob).
    min_distance : int
        Minimum voxel separation between watershed seeds (``peak_local_max``).
    next_label : int
        First label ID to assign (typically ``max(volume) + 1``).  Survivors
        are numbered ``next_label, next_label + 1, ...``.
    spacing : sequence of float, optional
        Per-axis voxel spacing for the EDT (anisotropy-aware watershed).

    Returns
    -------
    np.ndarray
        ``int32`` array, same shape as *existing_roi*, nonzero ONLY in
        background (``existing_roi == 0``), relabeled from *next_label*.
    """
    image = np.asarray(image_roi, dtype=np.float64)
    existing = np.asarray(existing_roi)
    if image.shape != existing.shape:
        raise ValueError(
            f"image_roi shape {image.shape} != existing_roi shape "
            f"{existing.shape}"
        )

    out = np.zeros(existing.shape, dtype=np.int32)
    if image.size == 0:
        return out

    # --- 1. optional CLAHE (requires [0,1]) ---
    proc = add_threshold_space(image, clahe=clahe, clahe_clip=clahe_clip)
    if clahe and not np.any(proc):  # constant image → nothing to enhance/segment
        return out

    # --- 2. threshold ---
    thr = _resolve_threshold(proc, threshold)
    if thr is None:
        return out
    binary = proc >= thr

    # --- 3. background-masked foreground (the clipping invariant) ---
    fg = binary & (existing == 0)
    if not np.any(fg):
        return out

    # --- 4. separate touching pieces ---
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import label as cc_label

    if separate:
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        from scipy.ndimage import gaussian_filter

        edt = distance_transform_edt(fg, sampling=spacing).astype(np.float64)
        # Lightly smooth the EDT before peak detection so flat interior
        # plateaus collapse to a single maximum per blob — otherwise
        # peak_local_max returns many co-equal peaks across a plateau and the
        # watershed over-segments one cell into a dozen slivers.
        edt_smooth = gaussian_filter(edt, sigma=1.0)
        peaks = peak_local_max(
            edt_smooth,
            min_distance=max(1, int(min_distance)),
            exclude_border=False,
            labels=fg,
        )
        if len(peaks) >= 2:
            markers = np.zeros(fg.shape, dtype=np.int32)
            for i, coord in enumerate(peaks):
                markers[tuple(int(c) for c in coord)] = i + 1
            pieces = watershed(-edt, markers=markers, mask=fg)
        else:
            # 0 or 1 interior peak → no meaningful split; one piece per blob.
            pieces, _ = cc_label(fg)
    else:
        pieces, _ = cc_label(fg)

    # --- 5 + 6. drop small pieces, relabel from next_label ---
    new_id = int(next_label)
    for val in np.unique(pieces):
        if val == 0:
            continue
        piece = pieces == val
        if min_size and int(piece.sum()) < int(min_size):
            continue
        out[piece] = new_id
        new_id += 1

    return out


def edge_flag_ids(labels, margin: int) -> set:
    """IDs of pieces reaching the box EDGE ZONE -- within *margin* voxels of
    any of the 6 walls -- i.e. objects the box likely TRUNCATED.

    Flags on wall PROXIMITY (any voxel in the boundary shell), not centroid.
    A LONE piece is never flagged (a single tightly-boxed object stays
    ticked; grow the box if it is clipped).  Moved verbatim from
    ``Curation3DAnnotator._add_edge_flags`` (Add rebuild).
    """
    if getattr(labels, "ndim", 0) != 3:
        return set()
    ids = [int(v) for v in np.unique(labels) if v != 0]
    if len(ids) <= 1:
        return set()  # single boxed object stays checked
    m = max(1, int(margin))
    flagged = set()
    for v in ids:
        mask = labels == v
        if (
            mask[:m].any() or mask[-m:].any()
            or mask[:, :m].any() or mask[:, -m:].any()
            or mask[:, :, :m].any() or mask[:, :, -m:].any()
        ):
            flagged.add(v)
    return flagged


def smooth_pieces(result, sigma: float = 1.5):
    """Isotropic SDF surface-smooth every piece in *result*.

    Routed through ``run_batch_recipe`` -- the SAME backend as Batch Refine
    -- with ``empty="erase"``: a piece its smoothing empties is dropped, as
    it always was.  Moved verbatim from
    ``Curation3DAnnotator._smooth_add_result`` (Add rebuild).
    """
    from .batch_refine import RefineStep, run_batch_recipe

    piece_ids = [int(v) for v in np.unique(result) if v != 0]
    if not piece_ids:
        return result
    recipe = [
        RefineStep(
            method="sdf",
            params={"sigma_z": float(sigma), "sigma_xy": float(sigma)},
        )
    ]
    try:
        res = run_batch_recipe(
            result, piece_ids, recipe, n_workers=1, empty="erase"
        )
        return res.working_volume
    except Exception as exc:
        print(f"[ADD] smooth failed: {exc!r}")
        return result

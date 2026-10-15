"""Wall (elevation) construction -- the surface the solver floods, high = wall.

Lifted from ``resegment_label``'s elevation branch chain
(in the membrane sandbox) into one function with an explicit enum, so the
elevation stage is independently testable and the ``ReshapeSpec.walls`` value
maps 1:1 onto a branch.

One deliberate addition to the ported set: ``"none"``.  In the sandbox there
was no flat option -- "geometry only" meant ``distance``.  Here ``none`` is a
genuinely flat elevation (a pure seed-midline / Voronoi partition) and
``distance`` is the -EDT, so no value silently aliases another.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .preproc import axis_sigma, clahe_stack, norm, preprocess_membrane


def build_elevation(walls, elev_img, region, *, spacing=(1, 1, 1), clip=0.01,
                    smooth=1.0, ridge=True):
    """Return ``(elevation, clahe, ridge_img)`` for one ``walls`` kind.

    ``elev_img`` is the (already preprocessed) wall channel crop, or ``None``
    for the geometry-only kinds.  ``region`` is the boolean target footprint.
    """
    clahe = ridge_img = None

    if walls == "gradient":
        # boundaries = intensity EDGES -> gradient magnitude. CLAHE FIRST so FAINT
        # membrane is locally boosted before differentiating (else it's missed ->
        # cells stay merged). The CLAHE'd base is exposed as the _clahe preview.
        base = clahe_stack(elev_img, clip) if clip else norm(elev_img)
        clahe = base
        elev = norm(ndi.gaussian_gradient_magnitude(
            base.astype(np.float32), axis_sigma(max(smooth, 1.0), spacing)))

    elif walls == "distance":
        # No intensity boundary -> split by SHAPE. EDT of the LABEL FOOTPRINT itself
        # (the merged blob is already binary -- no Otsu needed); negate so cell
        # centers are low basins and the thin NECK between sub-cells is a high ridge
        # -> watershed cuts at the neck.
        d = ndi.distance_transform_edt(region, sampling=spacing).astype(np.float64)
        elev = norm(-d)

    elif walls == "skeleton":
        # Medial-axis distance (Omnipose-style, lightweight). The ridge sits in the
        # inter-cell VALLEY (points far from EVERY cell's skeleton). Beats raw EDT for
        # SIDE-BY-SIDE elongated merges (no single distance peak). Does NOT help
        # STACKED merges -- there the interface lies ON the merged skeleton.
        from skimage.morphology import skeletonize
        try:
            skel = np.asarray(skeletonize(region.astype(np.uint8)), bool)
        except Exception:
            skel = np.zeros_like(region, bool)
        if not skel.any():
            skel = region & ~ndi.binary_erosion(region)   # fallback: boundary shell
        elev = norm(ndi.distance_transform_edt(~skel, sampling=spacing).astype(np.float64))

    elif walls == "intensity":
        # raw membrane INTENSITY as the wall (CLAHE'd, no ridge) -- bright = wall
        base = clahe_stack(elev_img, clip) if clip else norm(elev_img)
        clahe = base
        elev = norm(base)

    elif walls in ("frangi", "meijering"):
        # alternative membrane RIDGE filters (skimage) -- crisper than Sato on some
        # membranes. frangi = vesselness, meijering = neuriteness (thin/faint ridges).
        from skimage import filters as _skf
        base = clahe_stack(elev_img, clip) if clip else norm(elev_img)
        clahe = base
        _fn = _skf.frangi if walls == "frangi" else _skf.meijering
        try:
            ridge_img = norm(np.asarray(_fn(base, black_ridges=False), np.float32))
        except Exception:
            ridge_img = norm(base)
        elev = ridge_img

    elif walls == "hybrid":
        # membrane INTENSITY ridge BLENDED with the label SHAPE (-EDT): where the wall
        # is faint the shape neck backstops it. 0.6*membrane + 0.4*shape.
        mem, clahe, ridge_img = preprocess_membrane(elev_img, clip, smooth, ridge, spacing)
        d = ndi.distance_transform_edt(region, sampling=spacing).astype(np.float64)
        elev = norm(0.6 * norm(mem) + 0.4 * norm(-d))

    elif walls == "probability":
        # The Refine tab's tenth kind (REFINE).  `elev_img` here is
        # the reconstruction's probability crop, not a channel: high where the
        # fusion was CONFIDENT this was one object, so the boundary is where
        # it was least sure.  One line, purely additive -- no existing kind's
        # arithmetic moves, so the golden suite is bit-identical.
        elev = norm(1.0 - norm(elev_img))

    elif walls == "none":
        # No wall evidence at all -- a flat elevation makes watershed a pure
        # seed-midline (Voronoi) partition.  Distinct from "distance", which is
        # the -EDT; nothing here silently aliases anything else.
        elev = np.zeros(region.shape, np.float32)

    else:  # "membrane": bright rim IS the wall -> intensity + Sato ridge
        elev, clahe, ridge_img = preprocess_membrane(elev_img, clip, smooth, ridge, spacing)

    return np.asarray(elev, np.float32), clahe, ridge_img


def raise_neighbour_walls(elev, label_crop, target_id):
    """Neighbouring existing labels raise a hard wall.

    Ported from the membrane sandbox.  NOTE: inert while the solve is
    masked to the target footprint (neighbours sit outside it) -- it becomes
    live for the add-shaped / grow presets in R2.
    """
    others = (label_crop != 0) & (label_crop != target_id)
    if not others.any():
        return elev
    shell = others & ~ndi.binary_erosion(others)
    return np.maximum(elev, shell.astype(np.float32))

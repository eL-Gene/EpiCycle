"""Seed detectors -- ported **verbatim** from the membrane sandbox.

Provenance: ``seeds_from_nuclei``, ``seeds_membrane_minima``,
``seeds_auto``, ``seeds_blobs_log``, ``seeds_blobs_dog``,
``seeds_hmaxima``, ``seeds_threshold_cc``.

Two deliberate changes to the ports:

* ``seeds_auto`` -> :func:`seeds_distance_maxima`.  It **moves, but it is never
  an automatic fallback**.  The sandbox silently fell through
  to it whenever a chosen seed source found nothing, which can shatter a
  columnar cell into stacked Z-discs.  In this package it is one explicit
  ``ReshapeSpec.seeds`` value (``"distance"``) and nothing else routes to it.
* ``min_distance`` and ``max_sigma`` are taken in **physical units** and
  converted here.  ``peak_local_max`` cannot take a per-axis
  value, so the conversion is against ``min(spacing)`` and
  :func:`physical_to_voxel_scalar` says so rather than pretending otherwise.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

from .preproc import axis_sigma, norm, safe_otsu


def physical_to_voxel_scalar(distance, spacing):
    """Convert a PHYSICAL distance to a single scalar voxel count.

    ``peak_local_max``'s ``min_distance`` is one scalar for every axis, so an
    anisotropic conversion is impossible there.  We convert against the FINEST
    axis (``min(spacing)``), which makes the constraint conservative in the
    coarse axis rather than silently over-permissive.  Callers must surface
    that caveat in the tooltip.
    """
    sp = np.asarray(spacing, float)
    fine = float(sp.min()) if sp.size else 1.0
    if fine <= 0:
        fine = 1.0
    return max(1, int(round(float(distance) / fine)))


def seeds_from_labels(label_crop, region=None):
    """A trusted second label layer: each overlapping instance = one seed.

    ``label_crop`` is the guide layer already cropped to the ROI.  Seeds keep
    the guide's own instance ids inside the ROI (they are private to the ROI --
 mints final ids at write time), so a Preview and a commit see the same
    seed ids.
    """
    seeds = np.asarray(label_crop, np.int32).copy()
    if region is not None:
        seeds = np.where(region, seeds, 0)
    return seeds


def seeds_from_nuclei(nuc, thresh=None, smooth=2.0, min_distance=5, spacing=(1, 1, 1)):
    """One seed per nucleus via peak detection on the *smoothed* nuclei intensity.

    NOTE: this is peak_local_max (spatial-peak detection), NOT h-maxima -- it has no
    prominence gating, so on CLAHE'd/noisy nuclei it can over-seed; raise ``smooth``
    or ``min_distance`` if so. Robust to punctate (gaussian consolidates), touching
    (min_distance separates), dim (0.5*Otsu gate). Gaussian is per-axis (spacing-aware);
    ``min_distance`` is a scalar voxel count (peak_local_max limitation -- see
    :func:`physical_to_voxel_scalar`).
    """
    if nuc is None:
        return None
    from skimage.feature import peak_local_max
    ns = (ndi.gaussian_filter(nuc.astype(np.float32), axis_sigma(smooth, spacing))
          if smooth else nuc.astype(np.float32))
    t = thresh if thresh is not None else safe_otsu(ns)
    fg = ns >= 0.5 * float(t)
    peaks = peak_local_max(ns, min_distance=max(1, int(min_distance)),
                           labels=fg, exclude_border=False)
    seeds = np.zeros(nuc.shape, np.int32)
    for i, c in enumerate(peaks, 1):
        seeds[tuple(int(v) for v in c)] = i
    return seeds


def seeds_membrane_minima(mem, h=0.05, smooth=2.0, region=None, spacing=(1, 1, 1)):
    """Membrane-only seeding done right: cell interiors are the DIM basins of the
    membrane image, so seed from its regional **h-minima**. One seed per
    membrane-bounded interior -- works even for thin/flat cells (unlike the
    distance approach, which misses them).

    Feed the **raw / lightly-smoothed** membrane here (NOT the ridge-enhanced
    elevation -- CLAHE + ridge destroy the interior basins). If ``region`` is given
    (the label footprint), everything outside it is treated as wall so the interior
    basins CLOSE -- essential for cells open at the apical/basal face, whose
    interiors would otherwise drain out and not register as minima. ``h`` = minima
    depth in [0,1] (higher = fewer, deeper basins = fewer seeds).
    """
    m = np.asarray(mem, np.float32).copy()
    if region is not None:
        m[~region] = 1.0                       # outside footprint = wall (closes basins)
    ms = ndi.gaussian_filter(m, axis_sigma(smooth, spacing)) if smooth else m
    seeds, _ = ndi.label(morphology.h_minima(ms, float(h)))
    if region is not None:
        seeds = np.where(region, seeds, 0)     # keep only interior minima
    return seeds


def seeds_distance_maxima(mem, spacing=(1, 1, 1), h=0.15, region=None):
    """h-maxima of the interior distance transform (was ``seeds_auto``).

    Geometry-blind -- a fat cell gets one big seed, a thin cell none, and a
    columnar cell can shatter into stacked Z-discs.  Prefer
    :func:`seeds_membrane_minima` for membrane-only data.

    **This is never a fallback.**  It runs only when the user explicitly picks
    ``ReshapeSpec.seeds == "distance"``.

    ``region`` (the label footprint) was added 2026-09-20, from a live
    measurement (UX-round item 1).  This was the ONE detector of
    the six that did not take it, and the engine feeds it the WALL
    elevation over the whole ROI crop -- so on a branched object the
    background, being one big blob, owns every regional maximum of the
    distance transform and the object owns none.  Measured on object 139
    of ``CELLS-figure/patch_6``: the object is 7.6 % of its own bounding
    box, ``elev < otsu`` covers 99.2 % of it, the normalised distance
    reaches 0.494 inside against 1.000 outside, and the engine answered
    ``found=11 in_region=0`` at prominence 0.15, 0.05 and 0.02 alike.
    The source worked on compact objects and silently found nothing on
    the shape you most want to cut.

    The fix confines the INTERIOR to the footprint, before the distance
    transform and before the h-maxima -- masking only the output would
    not help, because the maxima would still all be in the background
    and masking would leave nothing.

    The Otsu threshold is deliberately still read off the WHOLE array.
    Re-reading it inside the footprint was tried and measured: on a
    solid box with ``walls="skeleton"`` it drops 0.432 -> 0.354, which
    carves the skeleton ridge out of the interior and turns one central
    marker into **20** scattered round the medial axis
    (``test_shape_markers_need_a_shape_wall``).  Confining the interior
    alone leaves every compact case bit-for-bit where it was -- box +
    skeleton 1, box + distance 1, box + no wall 0 -- and changes only
    the case that was broken.

    With ``region=None`` every line is the one that was here before, so
    the golden comparison against ``seeds_auto`` is untouched.

    **What ``region=`` did to the meaning of ``h``.**  ``norm`` rescales
    by the 1st and 99.5th percentiles of whatever it is handed.  Confined
    to the footprint, the distance transform's range is the OBJECT's own
    thickness span instead of the whole crop's, so ``h`` is now a
    fraction of that -- roughly five times more sensitive on an object
    filling 5 % of its bounding box than it was before.  The same knob
    means the same thing, measured against a smaller ruler.

    **And below about 0.5 % fill it stops meaning anything.**  Those
    percentiles are taken over the WHOLE array including the outside
    zeros, so once the object is small enough the 99.5th percentile is
    itself a background zero, the interior saturates at 1.0, and there is
    no gradient left for ``h`` to bite on -- every ``h`` gives the same
    answer.  A HOLLOW object hits the same wall for a different reason
    and much sooner, which is what ``engine.SHAPE_SEED_CAP`` refuses
    (audit M7): every point of a shell is an equal maximum, so the finder
    returns one marker per surface bump -- 68 on a 6.9 %-fill spherical
    shell, identical at every ``h``.

    **Which detectors take ``region=``:** all of them except
    :func:`seeds_from_nuclei`, which finds peaks in a nuclei channel and
    is filtered by the engine instead.  This one was the last to get it.
    """
    mem = np.asarray(mem)
    interior = mem < safe_otsu(mem)
    if region is not None:
        interior = interior & region
    dist = ndi.distance_transform_edt(interior, sampling=spacing).astype(np.float64)
    dist = norm(ndi.gaussian_filter(dist, axis_sigma(1.0, spacing)))
    seeds, _ = ndi.label(morphology.h_maxima(dist, h))
    if region is not None:
        seeds = np.where(region, seeds, 0)
    return seeds


def seeds_blobs_log(img, threshold=0.05, max_sigma=6.0, region=None, spacing=(1, 1, 1)):
    """Laplacian-of-Gaussian blob detection (the 'curvature / Hessian' family) --
    finds blob CENTERS by their curvature signature. More robust than plain
    intensity peaks when blobs TOUCH or vary in brightness (clumped nuclei).
    One seed per detected blob. ``threshold`` LOWER = more blobs; ``max_sigma`` ~
    the largest blob radius (in min-voxel units). Per-axis sigmas (spacing-aware)
    so an anisotropic nucleus is detected at the right scale in Z."""
    from skimage.feature import blob_log
    im = norm(np.asarray(img, np.float32))
    try:
        blobs = blob_log(im, min_sigma=axis_sigma(1.0, spacing),
                         max_sigma=axis_sigma(float(max_sigma), spacing),
                         num_sigma=5, threshold=float(threshold))
    except Exception:
        blobs = np.empty((0, im.ndim + im.ndim))
    seeds = np.zeros(img.shape, np.int32)
    for i, b in enumerate(blobs, 1):
        seeds[int(b[0]), int(b[1]), int(b[2])] = i
    if region is not None:
        seeds = np.where(region, seeds, 0)
    return seeds


def seeds_blobs_dog(img, threshold=0.05, max_sigma=6.0, region=None, spacing=(1, 1, 1)):
    """Difference-of-Gaussians blob detection -- a faster approximation of LoG, same
    'curvature' family. One seed per blob; per-axis (spacing-aware) sigmas."""
    from skimage.feature import blob_dog
    im = norm(np.asarray(img, np.float32))
    try:
        blobs = blob_dog(im, min_sigma=axis_sigma(1.0, spacing),
                         max_sigma=axis_sigma(float(max_sigma), spacing),
                         threshold=float(threshold))
    except Exception:
        blobs = np.empty((0, im.ndim * 2))
    seeds = np.zeros(img.shape, np.int32)
    for i, b in enumerate(blobs, 1):
        seeds[int(b[0]), int(b[1]), int(b[2])] = i
    if region is not None:
        seeds = np.where(region, seeds, 0)
    return seeds


def seeds_hmaxima(img, h=0.15, smooth=2.0, region=None, spacing=(1, 1, 1)):
    """Extended-maxima (h-maxima) on a FILLED signal (nuclei/cytoplasm). Prominence-
    gated -- keeps only peaks that stand out by depth >= h, so it's far more robust
    than plain peak detection on textured / CLAHE'd nuclei. One seed per regional
    maximum. LOWER h = more seeds."""
    im = norm(np.asarray(img, np.float32))
    if smooth:
        im = ndi.gaussian_filter(im, axis_sigma(smooth, spacing))
    seeds, _ = ndi.label(morphology.h_maxima(im, max(float(h), 1e-3)))  # h>0 (h<=0 raises)
    if region is not None:
        seeds = np.where(region, seeds, 0)
    return seeds


def seeds_threshold_cc(img, thresh=None, smooth=2.0, region=None, spacing=(1, 1, 1)):
    """Threshold + connected-components centroids on a FILLED signal -- the simplest
    seeding for well-SEPARATED bright blobs: smooth -> Otsu -> label components ->
    one seed at each centroid. (Touching blobs merge into one; use blobs / h-maxima /
    distance for those.)"""
    im = norm(np.asarray(img, np.float32))
    if smooth:
        im = ndi.gaussian_filter(im, axis_sigma(smooth, spacing))
    t = thresh if thresh is not None else safe_otsu(im)
    lab, n = ndi.label(im >= float(t))
    seeds = np.zeros(img.shape, np.int32)
    if n:
        for i, c in enumerate(ndi.center_of_mass(np.ones_like(lab), lab, range(1, n + 1)), 1):
            vox = tuple(int(round(v)) for v in c)
            if lab[vox] == 0:                       # centroid outside a concave blob -> snap in
                pts = np.argwhere(lab == i)
                vox = tuple(pts[len(pts) // 2])
            seeds[vox] = i
    if region is not None:
        seeds = np.where(region, seeds, 0)
    return seeds

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

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from skimage.util import img_as_float

from .preproc import axis_sigma, clahe_stack, norm, preprocess_membrane


def build_elevation(walls, elev_img, region, *, spacing=(1, 1, 1), clip=0.01,
                    smooth=1.0, ridge=True, cancel_event=None, n_workers=1):
    """Return ``(elevation, clahe, ridge_img)`` for one ``walls`` kind.

    ``elev_img`` is the (already preprocessed) wall channel crop, or ``None``
    for the geometry-only kinds.  ``region`` is the boolean target footprint.

    ``cancel_event`` (an object with ``is_set()``, or ``None``) is polled by
    the two ridge kinds (``frangi`` / ``meijering``) between their scales
    and passes; once set, ``ReshapeCancelled`` is raised.  The other kinds
    are quick and do not poll it.

    ``n_workers`` (the two ridge kinds only): threads for their passes.
    The default, 1, builds on the calling thread with no pool -- right for
    callers that already run one object per thread (Batch Refine's chain);
    the engine's wall stage asks for :func:`ridge_worker_count`.
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
        # Built by `ridge_filter` below: skimage's arithmetic, the same
        # array, with a Stop between scales and the passes on threads.
        from .engine import ReshapeCancelled
        base = clahe_stack(elev_img, clip) if clip else norm(elev_img)
        clahe = base
        try:
            ridge_img = norm(np.asarray(
                ridge_filter(base, walls, black_ridges=False,
                             cancel_event=cancel_event, n_workers=n_workers),
                np.float32))
        except ReshapeCancelled:
            raise                          # a Stop is not a filter failure
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


# --------------------------------------------------------------------------- #
# the ridge kinds (frangi / meijering): skimage's arithmetic, in our own loop
# --------------------------------------------------------------------------- #
# The loop below reproduces scikit-image's ``frangi`` and ``meijering`` (and
# the Hessian / eigenvalue helpers they call) operation for operation, so the
# picture is the same array.  It differs only in how the work is scheduled: a
# Stop is polled between scales, between the Gaussian passes and between the
# eigenvalue slabs, and the passes and the slabs run on a small thread pool
# (the Gaussian filter and LAPACK release the GIL).  Every call, argument and
# kernel width is skimage's, so the values are the same, not just close.
#
# Adapted from scikit-image (skimage/filters/ridges.py,
# skimage/feature/corner.py and the float-type rule of
# skimage/_shared/utils.py), Copyright: 2009-2022 the scikit-image team,
# BSD-3-Clause licence; the licence text is in NOTICE-scikit-image.txt next
# to this file.  Checked line by line against RIDGE_SKIMAGE_VERSION; a test
# compares the two and pins that version.

#: The scikit-image release this loop was checked against.
RIDGE_SKIMAGE_VERSION = "0.25.2"

#: At most this many threads build one ridge picture for the engine's wall
#: stage (and never more than the cores this process may use minus two):
#: the window has to stay responsive and the rest of the engine shares the
#: machine.  Eight took a whole-patch picture
#: from about 124 s to under a minute on a 20-core machine.
RIDGE_MAX_WORKERS = 8

#: The scales, as skimage's default ``sigmas=range(1, 10, 2)``.
RIDGE_SIGMAS = (1, 3, 5, 7, 9)

#: z-slices per eigenvalue slab (one pool task, one Stop poll).
_EIG_SLAB = 8

#: The Gaussian kernel half-width, in sigmas, of the sigma-1 scale; skimage
#: uses 100 there (and 8 above sigma 1).  A width of 8 would save about a
#: sixth of the build, but it is not the same arithmetic: on a whole patch
#: (float32) 34 of 36 million sigma-1 Hessian values moved, by at most
#: 1.4e-20, and on float64 input the ridge picture itself moved.  Kept at
#: skimage's 100 so the picture is identical by construction.
RIDGE_SIGMA1_TRUNCATE = 100


def _ridge_truncate(sigma_tuple):
    """skimage's rule: 8 when every sigma exceeds 1, else the sigma-1 width."""
    return 8 if all(s > 1 for s in sigma_tuple) else RIDGE_SIGMA1_TRUNCATE


#: Name prefix of the ridge pool's threads (a test checks none outlive a call).
RIDGE_THREAD_PREFIX = "epicycle-ridge"

# skimage's float-type rule (``_supported_float_type``, a private helper),
# kept here so a moved or renamed private name can never turn the ridge
# picture into the fall-back one: float32 / float64 kept, float16 -> float32,
# long double -> float64, every other real type -> float64; complex refused.
_RIDGE_FLOAT_TYPES = {
    np.dtype(np.float32).char: np.float32,
    np.dtype(np.float64).char: np.float64,
    np.dtype(np.float16).char: np.float32,
    "g": np.float64,
}


def _ridge_float_type(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        raise ValueError("complex valued input is not supported")
    return np.dtype(_RIDGE_FLOAT_TYPES.get(dtype.char, np.float64))


def ridge_worker_count(n_workers=None) -> int:
    """``n_workers`` when given, else ``min(RIDGE_MAX_WORKERS, usable
    cores - 2)``, at least 1 (see :mod:`..pool_workers`)."""
    if n_workers:
        return max(1, int(n_workers))
    from ..pool_workers import pool_worker_count

    return pool_worker_count(RIDGE_MAX_WORKERS)


def _poll(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        from .engine import ReshapeCancelled   # engine imports this module

        raise ReshapeCancelled()


def _run_all(pool, fn, jobs, cancel_event):
    """``[fn(*job) for job in jobs]``, on ``pool`` when there is one; a Stop
    is polled before each job starts."""
    def one(job):
        _poll(cancel_event)
        return fn(*job)

    if pool is None:
        return [one(job) for job in jobs]
    return list(pool.map(one, jobs))


def _ridge_hessian(image, sigma, pool, cancel_event):
    """skimage ``hessian_matrix(image, sigma, mode='reflect', cval=0,
    use_gaussian_derivatives=True)`` for a 3-D float image: two successive
    first-order Gaussian derivatives, elements in (0,0) (0,1) (0,2) (1,1)
    (1,2) (2,2) order."""
    from itertools import combinations_with_replacement

    ndim = image.ndim
    sigma_tuple = (sigma,) * ndim
    truncate = _ridge_truncate(sigma_tuple)
    sigma_scaled = tuple((1 / math.sqrt(2)) * s for s in sigma_tuple)

    def gauss(img, order):
        return ndi.gaussian_filter(img, sigma=sigma_scaled, mode="reflect",
                                   cval=0, truncate=truncate, order=order)

    orders = tuple([0] * d + [1] + [0] * (ndim - d - 1) for d in range(ndim))
    gradients = _run_all(pool, gauss,
                         [(image, orders[d]) for d in range(ndim)],
                         cancel_event)
    return _run_all(pool, gauss,
                    [(gradients[a], orders[b]) for a, b in
                     combinations_with_replacement(range(ndim), 2)],
                    cancel_event)


def _ridge_eigvals(H, pool, cancel_event):
    """skimage ``hessian_matrix_eigvals(H)`` (3-D): ``eigvalsh`` per voxel,
    decreasing, eigenvalue axis first -- computed in z-slabs, which gives
    the same values (``eigvalsh`` treats each 3x3 matrix on its own)."""
    from itertools import combinations_with_replacement

    nz = H[0].shape[0]
    pairs = list(combinations_with_replacement(range(3), 2))

    def slab(z0, z1):
        sub = [h[z0:z1] for h in H]
        m = np.zeros(sub[0].shape + (3, 3), dtype=sub[0].dtype)
        for idx, (row, col) in enumerate(pairs):
            m[..., row, col] = sub[idx]
            m[..., col, row] = sub[idx]
        e = np.linalg.eigvalsh(m)[..., ::-1]
        return np.transpose(e, (3, 0, 1, 2))

    spans = [(z, min(nz, z + _EIG_SLAB)) for z in range(0, nz, _EIG_SLAB)]
    return np.concatenate(_run_all(pool, slab, spans, cancel_event), axis=1)


def _frangi_scales(image, H_of, alpha=0.5, beta=0.5):
    """skimage ``frangi`` (3-D, ``gamma=None``) over the scales."""
    gamma = None
    filtered_max = np.zeros_like(image)
    for sigma in RIDGE_SIGMAS:
        eigvals = H_of(sigma)
        eigvals = np.take_along_axis(eigvals, abs(eigvals).argsort(0), 0)
        lambda1 = eigvals[0]
        lambda2, lambda3 = np.maximum(eigvals[1:], 1e-10)
        r_a = lambda2 / lambda3
        r_b = abs(lambda1) / np.sqrt(lambda2 * lambda3)
        s = np.sqrt((eigvals ** 2).sum(0))
        if gamma is None:
            gamma = s.max() / 2
            if gamma == 0:
                gamma = 1
        vals = 1.0 - np.exp(-(r_a ** 2) / (2 * alpha ** 2), dtype=image.dtype)
        vals *= np.exp(-(r_b ** 2) / (2 * beta ** 2), dtype=image.dtype)
        vals *= 1.0 - np.exp(-(s ** 2) / (2 * gamma ** 2), dtype=image.dtype)
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max


def _meijering_scales(image, H_of):
    """skimage ``meijering`` (``alpha=None``) over the scales."""
    from scipy import linalg

    alpha = 1 / (image.ndim + 1)
    mtx = linalg.circulant([1, *[alpha] * (image.ndim - 1)]).astype(image.dtype)
    filtered_max = np.zeros_like(image)
    for sigma in RIDGE_SIGMAS:
        eigvals = H_of(sigma)
        vals = np.tensordot(mtx, eigvals, 1)
        vals = np.take_along_axis(vals, abs(vals).argmax(0)[None], 0).squeeze(0)
        vals = np.maximum(vals, 0)
        max_val = vals.max()
        if max_val > 0:
            vals /= max_val
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max


def ridge_filter(image, kind, *, black_ridges=False, cancel_event=None,
                 n_workers=1):
    """skimage's ``frangi`` / ``meijering`` at the default scales, the same
    array, with a Stop polled between scales and passes.

    ``kind`` is ``"frangi"`` or ``"meijering"``.  ``cancel_event`` (an
    object with ``is_set()``, or ``None``) raises ``ReshapeCancelled`` at
    the next poll once set.  ``n_workers``: threads for the passes and the
    eigenvalue slabs; 1 (the default) runs on the calling thread with no
    pool, ``None`` means :func:`ridge_worker_count`.  A picture that is not
    3-D goes to skimage's own call.
    """
    if kind not in ("frangi", "meijering"):
        raise ValueError("ridge_filter: unknown kind {!r}".format(kind))
    if np.ndim(image) != 3:
        from skimage import filters as _skf

        fn = _skf.frangi if kind == "frangi" else _skf.meijering
        return fn(image, black_ridges=black_ridges)
    _poll(cancel_event)
    image = np.asarray(image)
    image = image.astype(_ridge_float_type(image.dtype), copy=False)
    if not black_ridges:
        image = -image
    # what skimage's Hessian helper does to its input before filtering
    image = img_as_float(image)
    image = image.astype(_ridge_float_type(image.dtype), copy=False)

    workers = ridge_worker_count(n_workers)
    pool = (ThreadPoolExecutor(max_workers=workers,
                               thread_name_prefix=RIDGE_THREAD_PREFIX)
            if workers > 1 else None)
    try:
        def H_of(sigma):
            _poll(cancel_event)
            H = _ridge_hessian(image, sigma, pool, cancel_event)
            _poll(cancel_event)
            return _ridge_eigvals(H, pool, cancel_event)

        if kind == "frangi":
            return _frangi_scales(image, H_of)
        return _meijering_scales(image, H_of)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

"""Preprocessing suite -- ported **verbatim** from the membrane sandbox.

Pure: numpy / scipy / skimage only.  **No Qt, no napari, no printing.**

Provenance: ``norm:16``, ``_axsigma:23``, ``_safe_otsu:31``,
``clahe_stack:44``, ``zdepth_correct:54``, ``background_subtract:68``,
``denoise_median:80``, ``apply_preproc:86``, ``preprocess_membrane:111`` --
all moved unchanged apart from the two leading-underscore names becoming
public (``axis_sigma`` / ``safe_otsu``) so tests and sibling modules can reach
them without touching a private.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters


def norm(a):
    """Percentile-normalize to [0, 1]."""
    a = np.asarray(a, np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def axis_sigma(sigma, spacing):
    """Per-axis voxel sigma for an isotropic PHYSICAL blur ``sigma`` (in min-voxel
    units). On (2,1,1) data, Z gets half the voxel-sigma of XY -> equal physical
    blur. Keeps gaussians/gradients/blobs from being Z-distorted."""
    sp = np.asarray(spacing, float)
    return tuple(float(sigma) * float(sp.min()) / s for s in sp)


def safe_otsu(img, default=0.5):
    """threshold_otsu guarded against flat / single-valued crops (which raise)."""
    a = np.asarray(img)
    if a.size == 0:
        return default
    if a.min() == a.max():
        return float(a.min())
    try:
        return float(filters.threshold_otsu(a))
    except Exception:
        return float(np.mean(a))


def clahe_stack(vol, clip=0.01):
    """Per-Z-slice CLAHE -- local contrast, robust to depth attenuation."""
    if not clip:
        return norm(vol)
    return np.stack(
        [exposure.equalize_adapthist(norm(s), clip_limit=clip) for s in vol]
    ).astype(np.float32)


# ---- standard preprocessing suite (pure; reusable) ------------------------- #
def zdepth_correct(vol):
    """Bleach-style correction ALONG Z: scale each z-slice so its mean matches the
    brightest slice -- flattens depth-dependent dimming (attenuation) so DEEP nuclei
    become as detectable as shallow ones. (ImageJ 'Simple Ratio' bleach correction,
    applied over Z instead of time.)

    **Optional, and worth thinking about before combining it with the arbiter**:
    it matches every z-slice's mean to the brightest slice's, so a
    membrane occupying a single z-plane -- which IS the brightest slice -- is
    absorbed into the baseline while the interior is lifted to meet it.  It
    flattens depth attenuation and a thin z-normal wall alike.  The engine emits
    a caution when both are on.
    """
    a = norm(np.asarray(vol, np.float32))
    means = np.array([a[z].mean() for z in range(a.shape[0])], np.float64)
    ref = means.max() if means.size else 1.0
    for z in range(a.shape[0]):
        if means[z] > 1e-6:
            a[z] = np.clip(a[z] * (ref / means[z]), 0, 1)
    return a


def background_subtract(vol, radius=15):
    """Remove slow, uneven background/haze via a per-slice WHITE TOP-HAT (= image -
    morphological opening). Same idea as ImageJ's rolling-ball: an opening with a disk
    of ``radius`` estimates the smooth background (features smaller than the disk are
    erased), and subtracting it keeps the small bright things (nuclei). ``radius`` =
    the largest feature to KEEP as foreground (set > a nucleus, < the background scale)."""
    from skimage.morphology import white_tophat, disk
    a = norm(np.asarray(vol, np.float32))
    se = disk(int(max(1, radius)))
    return norm(np.stack([white_tophat(a[z], se) for z in range(a.shape[0])]))


def denoise_median(vol, size=2):
    """Median filter -- despeckle. Run it BEFORE CLAHE so CLAHE doesn't amplify noise.
    NOTE: size=1 is a no-op (median of one voxel = itself); the default is 2."""
    return ndi.median_filter(np.asarray(vol, np.float32), size=int(max(1, size)))


def apply_preproc(img, spec, *, gamma=0.5, clip=0.02, bg_radius=15, denoise_size=2):
    """Apply a preprocessing CHAIN (a '+'-joined spec string, e.g. 'zcorrect+gamma+clahe')
    on a crop, IN a fixed sensible order: zcorrect -> bgsub -> gamma -> log -> denoise ->
    clahe.  Each token is optional; unknown/'none'/empty spec = passthrough. Tokens are
    split on '+' + matched as a SET (no substring false-triggers, and a stray 'none' in a
    compound spec is simply ignored)."""
    if img is None or not spec or spec == "none":
        return img
    tok = set(str(spec).split("+"))
    out = np.asarray(img, np.float32)
    if "zcorrect" in tok:
        out = zdepth_correct(out)
    if "bgsub" in tok:
        out = background_subtract(out, bg_radius)
    if "gamma" in tok:
        out = norm(out) ** max(float(gamma), 1e-3)
    if "log" in tok:
        out = norm(np.log1p(norm(out)))
    if "denoise" in tok:
        out = denoise_median(out, denoise_size)
    if "clahe" in tok:
        out = clahe_stack(out, clip)
    return out


def strip_clahe(chain):
    """Drop the ``clahe`` token from a wall chain.

    The wall stage keeps its OWN CLAHE (the ``clip`` knob inside
    :func:`preprocess_membrane` / the gradient branch), so allowing ``clahe``
    in the wall chain would double-apply it.  Making that an engine invariant
    rather than a widget-discipline assumption is ported verbatim from
    ``resegment_label``'s wall-preproc block.
    """
    if chain in (None, "none", ""):
        return chain
    return "+".join(t for t in str(chain).split("+") if t != "clahe")


def preprocess_membrane(cad, clip=0.01, smooth=1.0, ridge=True, spacing=(1, 1, 1)):
    """cadherin -> (elevation, clahe, ridge). Elevation is high at walls.

    CLAHE (attenuation-robust) -> gaussian (per-axis, spacing-aware) -> optional
    Sato tubeness. Returns the intermediates so the caller can show them as layers.
    """
    clahe = clahe_stack(cad, clip)
    m = ndi.gaussian_filter(clahe, axis_sigma(smooth, spacing)) if smooth else clahe.copy()
    ridge_img = None
    if ridge:
        ridge_img = filters.sato(m, sigmas=range(1, 3), black_ridges=False).astype(np.float32)
        m = norm(0.5 * norm(m) + 0.5 * norm(ridge_img))
    else:
        m = norm(m)
    return m, clahe, ridge_img

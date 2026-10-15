"""Label refinement algorithms for 3D curation.

Pure algorithm module — no Qt, no napari.  All heavy dependencies
(scipy, skimage) are imported lazily inside functions.

Methods (all share the ``(result_roi, n_modified, message)`` return contract):

- **Active contour** (``contour_refine``, REFINE) — ONE method, two
  modes: *Snap to edges* (``morphgac_refine`` on the shared wall picture,
  with the balloon gate passed explicitly) and *Inside vs outside
  brightness* (``morphacwe_refine`` restricted to a band around the
  outline).  3D, or serial 2D over a sample of slices with an SDF rebuild.
  A two-sided limit in microns and a neighbour fence on both.
- **Dilate/Erode** — fast morphological expansion/contraction.
- **Guided Grow** — probability- and/or intensity-guided region growing,
  ideal for closing open surfaces on basal/periderm cells.
- **Fill Holes** — automatic ``binary_fill_holes``, per-axis (3D/XY/XZ/YZ).
- **Flood Fill** — fill connected background region at seed coordinate,
  per-axis, with max-size safety limit.

All operate on 3D ROI crops (~80³) and typically complete in <5 seconds.
Return contract matches ``smooth_label_sdf()`` from ``smooth.py``:
``(result_roi, n_modified, message)``.
"""

from __future__ import annotations

import numpy as np


def _make_axis_structure(axis: str) -> np.ndarray | None:
    """Build a planar structuring element for axis-constrained dilation/erosion.

    Parameters
    ----------
    axis : str
        ``"3D"`` (default, full 3D), ``"XY"``, ``"XZ"``, or ``"YZ"``.

    Returns
    -------
    np.ndarray or None
        3×3×3 bool array with activity confined to one plane,
        or ``None`` for default 3D (scipy chooses its own structure).
    """
    if axis == "3D":
        return None

    from scipy.ndimage import generate_binary_structure

    cross_2d = generate_binary_structure(2, 1)  # 3×3 cross
    s = np.zeros((3, 3, 3), dtype=bool)
    if axis == "XY":
        s[1] = cross_2d          # z=1 plane
    elif axis == "XZ":
        s[:, 1, :] = cross_2d    # y=1 plane
    elif axis == "YZ":
        s[:, :, 1] = cross_2d    # x=1 plane
    else:
        return None  # fallback to default
    return s


def build_edge_stopping(
    elevation: np.ndarray,
    *,
    sharpness: float = 100.0,
) -> np.ndarray:
    """``gimage`` for a morphological geodesic active contour, from a WALL
    ELEVATION (REFINE).

    This used to run ``inverse_gaussian_gradient`` on a raw channel with a
    hard-coded ``alpha=100`` and an isotropic ``sigma=2.0`` -- 6 um through
    depth against 2 um across on (3, 1, 1) data -- and then optionally blend
    in a second gradient.  It now takes the SHARED wall elevation that
    :func:`packages.epicycle_annotator.reshape.walls.build_elevation` makes,
    already normalised to [0, 1] and already spacing-aware, and is nothing
    but the transfer function onto it:

    .. math:: g = 1 / (1 + k E^2)

    **Why this form and not ``1 - E``** (in order of weight):

    1. the balloon gate needs the two ends pushed apart.  skimage gates on
       ``gimage > threshold / |balloon|``; with ``k = 100`` flat background
       sits at ~1.0 and a real wall falls to ~0.01, so an explicit gate
       anywhere in (0, 1) separates "wall" from "not wall" cleanly.  With
       ``1 - E`` a faint wall sits at 0.7, in the middle of the gate's own
       range, and the gate would silently double as a wall-strength
       threshold;
    2. it is the family the code already used, so the old ``alpha`` knob
       becomes *Edge sharpness* with its order-100 default intact;
    3. it never reaches exactly 0, so a contour can always leave a spurious
       edge given enough steps (a morphological GAC that reaches 0 is frozen
       there);
    4. it has a knee, so *Edge sharpness* has something to do.

    The old "Image weight" ``alpha`` is **retired**: it was a measured no-op
    without a probability file, and the intent it expressed is now said
    honestly by choosing ``probability`` or ``hybrid`` as the wall kind.

    Parameters
    ----------
    elevation : np.ndarray
        The wall picture -- high = wall, nominally in [0, 1].
    sharpness : float
        ``k`` above (the panel's *Edge sharpness*).  Higher = a sharper knee
        between "wall" and "not wall".

    Returns
    -------
    gimage : np.ndarray
        float32, same shape, values in (0, 1].
    """
    e = np.asarray(elevation, np.float32)
    k = float(max(sharpness, 0.0))
    return (1.0 / (1.0 + k * e * e)).astype(np.float32)


def two_sided_limit(
    original_mask: np.ndarray,
    result_mask: np.ndarray,
    *,
    limit_um: float,
    spacing=(1.0, 1.0, 1.0),
):
    """Clip a moved outline to ``limit_um`` **on both sides**.

    The limit every contour shipped with was one-sided::

        evolved & (edt(~original) <= clamp)

    ``edt(~original)`` is zero everywhere *inside* the original footprint, so
    that expression is trivially true across the whole of it: it capped
    GROWTH and never restrained COLLAPSE -- the one failure mode that
    actually happened (the 2026-09-21 audit deleted objects to 0 while
    reporting success).  The second transform is the one that did not exist.

    Returns
    -------
    (clipped, hit_limit, mean_move_um, max_move_um)
        ``clipped`` is a fresh boolean array; ``hit_limit`` says whether the
        clip actually bit, and the two distances are measured off the same
        two transforms so the panel's read-out and the clip cannot disagree.
    """
    from scipy.ndimage import distance_transform_edt

    original_mask = np.asarray(original_mask, bool)
    result_mask = np.asarray(result_mask, bool)
    sampling = tuple(float(s) if float(s) > 0 else 1.0 for s in spacing)
    limit = float(limit_um)

    d_out = distance_transform_edt(~original_mask, sampling=sampling)
    d_in = distance_transform_edt(original_mask, sampling=sampling)

    grew = result_mask & ~original_mask
    shrank = original_mask & ~result_mask

    if limit > 0:
        too_far_out = grew & (d_out > limit)
        too_far_in = shrank & (d_in > limit)
        clipped = (result_mask & ~too_far_out) | too_far_in
        hit = bool(too_far_out.any() or too_far_in.any())
    else:
        clipped = result_mask.copy()
        hit = False

    grew = clipped & ~original_mask
    shrank = original_mask & ~clipped
    moves = []
    if grew.any():
        moves.append(d_out[grew])
    if shrank.any():
        moves.append(d_in[shrank])
    if moves:
        allm = np.concatenate([m.ravel() for m in moves])
        mean_um, max_um = float(allm.mean()), float(allm.max())
    else:
        mean_um = max_um = 0.0
    return clipped, hit, mean_um, max_um


def morphgac_refine(
    volume_roi: np.ndarray,
    label_id: int,
    gimage: np.ndarray,
    *,
    iterations: int = 100,
    smoothing: int = 2,
    balloon: float = 0.0,
    threshold: float | str = "auto",
    neighbor_constraint: bool = True,
    distance_clamp: int = 8,
) -> tuple[np.ndarray, int, str]:
    """Refine a single label using morphological geodesic active contours.

    Parameters
    ----------
    volume_roi : np.ndarray
        3D label volume (ROI crop).
    label_id : int
        Which label to refine.
    gimage : np.ndarray
        Edge-stopping function from ``build_edge_stopping()``.
    iterations : int
        MorphGAC iteration count (10–500).
    smoothing : int
        Morphological smoothing passes per iteration (1–4).
    balloon : float
        Expansion/contraction force (−1.0 to 1.0, 0.0 = neutral).
    threshold : float or 'auto'
        Edge threshold for MorphGAC.
    neighbor_constraint : bool
        If True, never expand into voxels occupied by other labels.
    distance_clamp : int
        Max voxels the contour can move from original boundary (0 = unlimited).

    Returns
    -------
    result_roi : np.ndarray
        Volume of same shape containing only ``label_id`` and 0.
    n_modified : int
        Number of voxels that changed.
    message : str
        Human-readable status message.
    """
    from skimage.segmentation import morphological_geodesic_active_contour

    mask = volume_roi == label_id
    if not np.any(mask):
        return np.zeros_like(volume_roi), 0, f"Label {label_id} not found in ROI."

    init_ls = mask.astype(np.int8)

    evolved = morphological_geodesic_active_contour(
        gimage, iterations,
        init_level_set=init_ls,
        smoothing=smoothing,
        balloon=balloon,
        threshold=threshold,
    )
    evolved_mask = evolved > 0.5

    # Neighbor constraint: never invade voxels occupied by other labels.
    if neighbor_constraint:
        other_labels = (volume_roi != 0) & (volume_roi != label_id)
        evolved_mask = evolved_mask & ~other_labels

    # Distance clamp: limit how far the contour can move from the original.
    if distance_clamp > 0:
        from scipy.ndimage import distance_transform_edt

        dist_from_original = distance_transform_edt(~mask)
        evolved_mask = evolved_mask & (dist_from_original <= distance_clamp)

    result = np.zeros_like(volume_roi)
    result[evolved_mask] = label_id

    n_modified = int(np.count_nonzero(evolved_mask != mask))
    if n_modified == 0:
        return result, 0, f"Label {label_id} is already well-shaped (MorphGAC)."

    return result, n_modified, (
        f"MorphGAC refined label {label_id}: {n_modified} voxels changed."
    )


def morphacwe_refine(
    volume_roi: np.ndarray,
    label_id: int,
    image_roi: np.ndarray,
    *,
    iterations: int = 100,
    smoothing: int = 2,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    neighbor_constraint: bool = True,
    distance_clamp: int = 8,
    band_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, str]:
    """Refine a single label using morphological Chan-Vese (ACWE).

    Region-based active contour — works on raw intensity, no edge
    preprocessing needed.

    Parameters
    ----------
    volume_roi : np.ndarray
        3D label volume (ROI crop).
    label_id : int
        Which label to refine.
    image_roi : np.ndarray
        3D float array (raw membrane image cropped to ROI).
    iterations : int
        MorphACWE iteration count (10–500).
    smoothing : int
        Morphological smoothing passes per iteration (1–4).
    lambda1 : float
        Weight for "inside" region term (higher = stronger pull to match interior).
    lambda2 : float
        Weight for "outside" region term (higher = stronger push away from exterior).
    neighbor_constraint : bool
        If True, never expand into voxels occupied by other labels.
    distance_clamp : int
        Max voxels the contour can move from original boundary (0 = unlimited).
    band_mask : np.ndarray or None
        When given, the two populations Chan-Vese
        compares are measured **inside this band only**, and only voxels in
        the band may change.  On membrane data the whole interior looks like
        background, so an unbanded run calls the bright rim "inside" and
        deletes the object (the 2026-09-21 audit: 3 cases of 4); a band a
        shell thick around the outline does not.  ``None`` keeps skimage's
        own whole-image behaviour, unchanged.

    Returns
    -------
    result_roi : np.ndarray
        Volume of same shape containing only ``label_id`` and 0.
    n_modified : int
        Number of voxels that changed.
    message : str
        Human-readable status message.
    """
    from skimage.segmentation import morphological_chan_vese

    mask = volume_roi == label_id
    if not np.any(mask):
        return np.zeros_like(volume_roi), 0, f"Label {label_id} not found in ROI."

    init_ls = mask.astype(np.int8)

    # Normalize image for ACWE.
    img = image_roi.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    if band_mask is None:
        evolved = morphological_chan_vese(
            img, iterations,
            init_level_set=init_ls,
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2,
        )
    else:
        evolved = _chan_vese_in_band(
            img, init_ls, np.asarray(band_mask, bool),
            iterations=iterations, smoothing=smoothing,
            lambda1=lambda1, lambda2=lambda2,
        )
    evolved_mask = evolved > 0.5

    # Neighbor constraint: never invade voxels occupied by other labels.
    if neighbor_constraint:
        other_labels = (volume_roi != 0) & (volume_roi != label_id)
        evolved_mask = evolved_mask & ~other_labels

    # Distance clamp: limit how far the contour can move from the original.
    if distance_clamp > 0:
        from scipy.ndimage import distance_transform_edt

        dist_from_original = distance_transform_edt(~mask)
        evolved_mask = evolved_mask & (dist_from_original <= distance_clamp)

    result = np.zeros_like(volume_roi)
    result[evolved_mask] = label_id

    n_modified = int(np.count_nonzero(evolved_mask != mask))
    if n_modified == 0:
        return result, 0, f"Label {label_id} is already well-shaped (MorphACWE)."

    return result, n_modified, (
        f"MorphACWE refined label {label_id}: {n_modified} voxels changed."
    )



# --------------------------------------------------------------------------- #
# Active contour -- ONE method, two modes (REFINE, Brick B2)
# --------------------------------------------------------------------------- #

#: The two modes, read BY INDEX off the panel's mode picker: 0 = *Snap to
#: edges* (a geodesic active contour on the shared wall picture), 1 =
#: *Inside vs outside brightness* (Chan-Vese on the prepared channel,
#: compared inside a band around the outline).
CONTOUR_MODES = ("snap", "brightness")

#: The orientations the contour runs in.  "3D" evolves the whole ROI at
#: once; the three planes trace a SAMPLE of slices and fill the rest in.
CONTOUR_ORIENTATIONS = ("3D", "XY", "XZ", "YZ")

#: The slice-sampling picker, read BY INDEX.
SLICE_SAMPLINGS = ("share", "count", "every")

#: A share of the slices is floored at this many.
SLICE_MIN_COUNT = 5

#: ...and every sampling is capped at this many.  100 % of a 300-slice
#: object at ~0.14 s a plane is 42 s on a worker whose Stop cannot
#: interrupt a plane -- a job nobody chose.  A module constant, so the
#: status can say "the maximum" when it bites.
SLICE_MAX_COUNT = 60

#: The stacking axis of each plane orientation, in (z, y, x).
_PLANE_AXIS = {"XY": 0, "XZ": 1, "YZ": 2}


class ContourResult(tuple):
    """``(result_roi, n_modified, message)`` -- the shared return contract,
    so every existing unpack keeps working -- with the facts the panel needs
    riding on ``.details`` (why nothing moved, whether the limit bit, how
    many slices were traced).  A plain tuple subclass: no new type to
    import on the other side, and ``a, b, c = result`` is unchanged.
    """

    def __new__(cls, result_roi, n_modified, message, **details):
        obj = tuple.__new__(cls, (result_roi, int(n_modified), str(message)))
        obj.details = dict(details)
        return obj


def push_acts_below(balloon: float, gate: float,
                    sharpness: float = 100.0):
    """The wall-picture value below which *Push outward* acts, or ``None``
    when it acts nowhere at all.

    skimage lets the balloon act only where ``gimage > gate / |balloon|``;
    with ``gimage = 1 / (1 + k E^2)`` that is exactly
    ``E < sqrt((|b| / g - 1) / k)`` -- monotone in both knobs, both ends
    meaningful.  ``gimage <= 1``, so a push that is not bigger than the gate
    acts NOWHERE, which is the case the panel's live line says out loud
    before the user presses Preview.  ``inf`` means everywhere (a gate of 0).
    """
    b = abs(float(balloon))
    g = float(gate)
    if b == 0.0:
        return None
    if g <= 0.0:
        return float("inf")
    if b <= g:
        return None
    k = float(sharpness)
    if k <= 0.0:
        return float("inf")
    return float(np.sqrt((b / g - 1.0) / k))


def _plane_spacing(spacing, orientation):
    """The 2-vector of spacing for one plane orientation."""
    sp = tuple(float(s) for s in spacing)
    ax = _PLANE_AXIS[orientation]
    return tuple(s for i, s in enumerate(sp) if i != ax)


def _band_around(mask, band_um, spacing):
    """Voxels within ``band_um`` of the outline, on either side, in microns."""
    from scipy.ndimage import distance_transform_edt

    sampling = tuple(float(s) if float(s) > 0 else 1.0 for s in spacing)
    mask = np.asarray(mask, bool)
    d_out = distance_transform_edt(~mask, sampling=sampling)
    d_in = distance_transform_edt(mask, sampling=sampling)
    # Each transform is ZERO on the other side of the outline, so each
    # half must be taken on its OWN side -- `(d_out <= b) | (d_in <= b)` is
    # True everywhere, and the "band" was the whole crop (B2 audit C-H1).
    return ((~mask & (d_out <= float(band_um)))
            | (mask & (d_in <= float(band_um))))


def _chan_vese_in_band(img, init_ls, band, *, iterations, smoothing,
                       lambda1, lambda2):
    """Morphological Chan-Vese with both populations measured INSIDE
    ``band`` and every change confined to it.

    skimage's own loop, line for line, except for the two masks: the
    means ``c0`` / ``c1`` are taken over the band only, and the attachment
    and smoothing steps may not change a voxel outside it.  The smoothing
    operators are skimage's public ``sup_inf`` / ``inf_sup`` in the same
    alternating order its private ``_curvop`` cycles through.
    """
    from skimage.segmentation.morphsnakes import inf_sup, sup_inf

    u = (np.asarray(init_ls) > 0).astype(np.int8)
    u0 = u.copy()
    frozen = ~band
    curv = (lambda v: sup_inf(inf_sup(v)), lambda v: inf_sup(sup_inf(v)))
    k = 0
    for _ in range(int(iterations)):
        inside = (u > 0) & band
        outside = (u == 0) & band
        c1 = float(img[inside].mean()) if inside.any() else 0.0
        c0 = float(img[outside].mean()) if outside.any() else 0.0
        du = np.gradient(u)
        abs_du = np.abs(np.asarray(du)).sum(0)
        aux = abs_du * (lambda1 * (img - c1) ** 2 - lambda2 * (img - c0) ** 2)
        u[(aux < 0) & band] = 1
        u[(aux > 0) & band] = 0
        for _s in range(int(smoothing)):
            u = curv[k % 2](u)
            k += 1
        u[frozen] = u0[frozen]
    return u


def _normalised(image):
    img = np.asarray(image, np.float64)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img)


def _evolve(volume_roi, label_id, *, mode, gimage, image, spacing,
            band_um, iterations, smoothing, balloon, gate, lambda1,
            lambda2, neighbor_constraint):
    """One contour run on a 2D plane or a 3D ROI; returns the boolean mask.

    The two 2024 bodies do the work -- ``morphgac_refine`` with the gate
    passed EXPLICITLY (never ``"auto"``) and ``morphacwe_refine`` with its
    band -- each with ``distance_clamp=0``, because the one-sided clamp was
    a measured no-op inside the footprint and the two-sided limit replaces
    it outside them.
    """
    if mode == "snap":
        res, _n, _m = morphgac_refine(
            volume_roi, label_id, gimage,
            iterations=int(iterations), smoothing=int(smoothing),
            balloon=float(balloon), threshold=float(gate),
            neighbor_constraint=neighbor_constraint, distance_clamp=0,
        )
    else:
        band = _band_around(volume_roi == label_id, band_um, spacing)
        res, _n, _m = morphacwe_refine(
            volume_roi, label_id, image,
            iterations=int(iterations), smoothing=int(smoothing),
            lambda1=float(lambda1), lambda2=float(lambda2),
            neighbor_constraint=neighbor_constraint, distance_clamp=0,
            band_mask=band,
        )
    return np.asarray(res) == label_id


def sample_slice_indices(first: int, last: int, *, sampling="share",
                         share_pct: float = 25.0, n_slices: int = 12,
                         min_count: int = SLICE_MIN_COUNT,
                         max_count: int = SLICE_MAX_COUNT):
    """``(indices, capped)`` -- which slices the serial-2D contour traces.

    Spread EVENLY across ``first..last`` (inclusive), and the first and
    last are ALWAYS in the list -- otherwise the rebuild extrapolates past
    the ends and the object grows a cap.  ``capped`` says the
    ``max_count`` cap bit, so the status can say "the maximum".

    ``sampling`` is ``"share"`` (``share_pct`` of the extent, floored at
    ``min_count``), ``"count"`` (``n_slices``) or ``"every"``; an int is
    read as an index into :data:`SLICE_SAMPLINGS`.
    """
    if isinstance(sampling, (int, np.integer)):
        sampling = SLICE_SAMPLINGS[int(sampling)]
    first, last = int(first), int(last)
    if last < first:
        first, last = last, first
    n_total = last - first + 1
    if sampling == "share":
        n = int(round(float(share_pct) / 100.0 * n_total))
        n = max(n, int(min_count))
    elif sampling == "count":
        n = int(n_slices)
    elif sampling == "every":
        n = n_total
    else:
        raise ValueError("unknown slice sampling {!r}".format(sampling))
    capped = n > int(max_count) and n_total > int(max_count)
    n = min(n, int(max_count), n_total)
    n = max(n, min(2, n_total))
    idx = np.unique(np.round(np.linspace(first, last, n)).astype(int))
    return [int(i) for i in idx], bool(capped)


def contour_refine(
    volume_roi: np.ndarray,
    label_id: int,
    *,
    mode,
    elevation: np.ndarray | None = None,
    image_roi: np.ndarray | None = None,
    spacing=(1.0, 1.0, 1.0),
    limit_um: float = 3.0,
    iterations: int = 100,
    smoothing: int = 2,
    balloon: float = 0.0,
    gate: float = 0.30,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    band_um: float | None = None,
    neighbor_constraint: bool = True,
    sharpness: float = 100.0,
    orientation: str = "3D",
    sampling="share",
    share_pct: float = 25.0,
    n_slices: int = 12,
    rebuild_sigma: float = 1.5,
    rebuild_um: float | None = None,
    size_budget: int | None = None,
    progress=None,
    cancel_event=None,
) -> ContourResult:
    """Slide one object's outline onto the image (REFINE).

    **Snap to edges** (``mode="snap"`` or 0) runs a morphological geodesic
    active contour on ``build_edge_stopping(elevation, sharpness)`` -- the
    shared wall picture, not a raw channel -- with the balloon gate passed
    EXPLICITLY as ``gate``.  A push that is not bigger than the gate acts
    nowhere (:func:`push_acts_below`), so that case returns at once with
    nothing moved and ``details["reason"] = "push_below_gate"`` -- the
    user's push would otherwise be silently ignored while the edges alone
    moved the outline, and the result would not be the thing they asked
    for.

    **Inside vs outside brightness** (``"brightness"`` or 1) runs
    Chan-Vese on ``image_roi`` (the PREPARED channel), with both
    populations measured inside a band ``band_um`` (default ``limit_um``)
    either side of the outline.

    Both then apply the neighbour FENCE (``(volume == 0) | (volume ==
    label_id)``) and the TWO-SIDED limit in microns.

    ``orientation`` "XY" / "XZ" / "YZ" traces a SAMPLE of slices
    (:func:`sample_slice_indices`), each read through
    :func:`packages.epicycle_annotator.interpolation.get_slice` with the
    limit applied in-plane, then fills the gaps with the annotator's own
    ``interpolate_slices_sdf``, blurs the signed distance through depth by
    ``rebuild_sigma`` voxels and re-thresholds -- and the fence and the
    limit are applied again to the rebuilt 3D shape.  ``progress(stage, i,
    n)`` is called once per plane (``"contour"``, 1-based) and once for the
    rebuild; ``cancel_event`` is checked BETWEEN planes, and a set event
    returns the object unchanged with ``details["reason"] = "stopped"``.

    ``rebuild_um``, when given, is the rebuild's blur in MICRONS and wins
    over ``rebuild_sigma`` (voxels): it is divided by the spacing of the
    stacking axis, so the same number joins the same physical depth on
    (3, 1, 1) data as on isotropic data (B2 audit C-L3).

    A traced plane the contour EMPTIES is not used: the object's own plane
    stands in for it, it is not counted as traced, and ``details["dropped"]``
    says how many (B2 audit C-L2) -- one bad plane would otherwise punch a
    hole the SDF fill then spreads across two gaps.

    Returns a :class:`ContourResult` -- ``(result_roi, n_modified,
    message)`` -- whose ``.details`` carries ``reason`` (None, "missing",
    "push_below_gate", "limit", "fence", "stopped", "too_large"),
    ``hit_limit``,
    ``mean_move_um``, ``max_move_um``, ``traced``, ``total``, ``capped``,
    ``plane_seconds`` and ``rebuild_seconds``.
    """
    import time as _time

    from scipy.ndimage import distance_transform_edt, gaussian_filter1d

    if isinstance(mode, (int, np.integer)):
        mode = CONTOUR_MODES[int(mode)]
    if mode not in CONTOUR_MODES:
        raise ValueError("unknown contour mode {!r}".format(mode))
    if orientation not in CONTOUR_ORIENTATIONS:
        raise ValueError("unknown orientation {!r}".format(orientation))
    volume_roi = np.asarray(volume_roi)
    spacing = tuple(float(s) if float(s) > 0 else 1.0 for s in spacing)
    limit_um = float(limit_um)
    band = float(limit_um if band_um is None else band_um)
    label_id = int(label_id)

    base = dict(reason=None, hit_limit=False, mean_move_um=0.0,
                max_move_um=0.0, traced=0, total=0, capped=False, dropped=0,
                plane_seconds=[], rebuild_seconds=0.0, mode=mode,
                orientation=orientation, gate=float(gate),
                balloon=float(balloon), limit_um=limit_um)

    mask = volume_roi == label_id
    unchanged = np.where(mask, label_id, 0).astype(volume_roi.dtype)
    if not mask.any():
        return ContourResult(np.zeros_like(volume_roi), 0,
                             "Object {} not found in the region.".format(
                                 label_id), **dict(base, reason="missing"))

    if size_budget is not None and volume_roi.size > int(size_budget):
        return ContourResult(unchanged, 0, (
            "This object's region is too large to trace here ({:,} voxels)."
            .format(int(volume_roi.size))), **dict(base, reason="too_large"))

    gimage = image = None
    if mode == "snap":
        if elevation is None:
            raise ValueError("Snap to edges needs a wall picture (elevation).")
        if np.shape(elevation) != volume_roi.shape:
            raise ValueError("the wall picture does not match the region")
        if balloon != 0 and push_acts_below(balloon, gate, sharpness) is None:
            return ContourResult(unchanged, 0, (
                "The push ({:.2f}) is not bigger than the Edge gate ({:.2f}), "
                "so it acts nowhere.".format(float(balloon), float(gate))),
                **dict(base, reason="push_below_gate"))
        gimage = build_edge_stopping(elevation, sharpness=sharpness)
    else:
        if image_roi is None:
            raise ValueError("Inside vs outside brightness needs a channel.")
        if np.shape(image_roi) != volume_roi.shape:
            raise ValueError("the channel does not match the region")
        image = _normalised(image_roi)

    fence = (volume_roi == 0) | mask
    evolve_kw = dict(mode=mode, band_um=band, iterations=iterations,
                     smoothing=smoothing, balloon=balloon, gate=gate,
                     lambda1=lambda1, lambda2=lambda2,
                     neighbor_constraint=False)
    traced_msg = ""
    dropped = 0

    if orientation == "3D":
        if progress is not None:
            progress("contour", 0, 0)
        t0 = _time.perf_counter()
        new = _evolve(volume_roi, label_id, gimage=gimage, image=image,
                      spacing=spacing, **evolve_kw)
        base["plane_seconds"] = [float(_time.perf_counter() - t0)]
    else:
        from .interpolation import get_slice, interpolate_slices_sdf

        ax = _PLANE_AXIS[orientation]
        along = np.any(mask, axis=tuple(i for i in range(3) if i != ax))
        present = np.where(along)[0]
        first, last = int(present[0]), int(present[-1])
        picks, capped = sample_slice_indices(
            first, last, sampling=sampling, share_pct=share_pct,
            n_slices=n_slices)
        total = last - first + 1
        pspace = _plane_spacing(spacing, orientation)
        traced = {}
        seconds = []
        for i, idx in enumerate(picks, start=1):
            if cancel_event is not None and cancel_event.is_set():
                return ContourResult(unchanged, 0, "Stopped.", **dict(
                    base, reason="stopped", traced=i - 1, total=total,
                    plane_seconds=seconds))
            if progress is not None:
                progress("contour", i, len(picks))
            t0 = _time.perf_counter()
            vplane = np.ascontiguousarray(
                get_slice(volume_roi, orientation, idx))
            orig2 = vplane == label_id
            if not orig2.any():
                traced[idx] = orig2
                seconds.append(float(_time.perf_counter() - t0))
                continue
            g2 = (None if gimage is None else np.ascontiguousarray(
                get_slice(gimage, orientation, idx)))
            i2 = (None if image is None else np.ascontiguousarray(
                get_slice(image, orientation, idx)))
            m2 = _evolve(vplane, label_id, gimage=g2, image=i2,
                         spacing=pspace, **evolve_kw)
            if neighbor_constraint:
                m2 = m2 & np.ascontiguousarray(
                    get_slice(fence, orientation, idx))
            m2, _h, _a, _b = two_sided_limit(orig2, m2, limit_um=limit_um,
                                             spacing=pspace)
            if not m2.any():
                # C-L2: an emptied plane is not evidence of the shape.
                dropped += 1
                m2 = orig2
            traced[idx] = m2
            seconds.append(float(_time.perf_counter() - t0))
        if cancel_event is not None and cancel_event.is_set():
            return ContourResult(unchanged, 0, "Stopped.", **dict(
                base, reason="stopped", traced=len(picks), total=total,
                plane_seconds=seconds))

        # The rebuild: the traced planes, the SDF fill between
        # each consecutive pair, a blur of the signed distance THROUGH
        # DEPTH, and a re-threshold -- confined to the object's own extent,
        # so the ends cannot grow a cap.
        if progress is not None:
            progress("rebuild", 0, 0)
        t0 = _time.perf_counter()
        new = np.zeros(mask.shape, bool)

        def _put(idx, plane):
            sl = [slice(None)] * 3
            sl[ax] = int(idx)
            new[tuple(sl)] = plane

        for idx, plane in traced.items():
            _put(idx, plane)
        for a, b in zip(picks[:-1], picks[1:]):
            gap = b - a - 1
            if gap <= 0:
                continue
            ma, mb = traced[a], traced[b]
            if ma.any() and mb.any():
                fills = interpolate_slices_sdf(ma, mb, gap)
            else:
                # One side empty: there is nothing to interpolate towards,
                # so the planes in between are left as the object had them.
                fills = [np.ascontiguousarray(get_slice(mask, orientation, k))
                         for k in range(a + 1, b)]
            for k, plane in enumerate(fills, start=a + 1):
                _put(k, plane)
        sigma_vox = float(rebuild_sigma)
        if rebuild_um is not None:
            sigma_vox = float(rebuild_um) / float(spacing[ax])
        base["rebuild_sigma_voxels"] = sigma_vox
        if sigma_vox > 0 and new.any():
            sdf = (distance_transform_edt(~new).astype(np.float32)
                   - distance_transform_edt(new).astype(np.float32))
            sdf = gaussian_filter1d(sdf, sigma_vox, axis=ax)
            new = sdf < 0
        keep = np.zeros(mask.shape[ax], bool)
        keep[first:last + 1] = True
        shape = [1, 1, 1]
        shape[ax] = mask.shape[ax]
        new &= keep.reshape(shape)
        base["rebuild_seconds"] = float(_time.perf_counter() - t0)
        base["plane_seconds"] = seconds
        n_traced = len(picks) - dropped
        base.update(traced=n_traced, total=total, capped=capped,
                    dropped=dropped)
        # The three parts ADD UP (B3 audit LOW): traced + kept (a traced
        # slice that came out empty stands as it was) + filled in == the
        # object's slices.  It said "Traced 9 of 52 ... filled in the other
        # 39", and 9 + 39 is not 52.
        n_filled = total - len(picks)
        if dropped:
            one = dropped == 1
            kept = ("{} more came out empty and {} kept as {} {}".format(
                dropped, "was" if one else "were", "it" if one else "they",
                "was" if one else "were"))
            if capped:
                traced_msg = ("Traced {} of {} slices (the maximum); {}; the "
                              "rest were filled in.".format(
                                  n_traced, total, kept))
            else:
                traced_msg = ("Traced {} of {} slices; {}; the other {} were "
                              "filled in.".format(
                                  n_traced, total, kept, n_filled))
        elif capped:
            traced_msg = ("Traced {} of {} slices (the maximum); the rest "
                          "were filled in.".format(n_traced, total))
        else:
            traced_msg = ("Traced {} of {} slices and filled in the other "
                          "{}.".format(n_traced, total, n_filled))
    base["traced_text"] = traced_msg

    # The fence and the two-sided limit, on the 3D shape.
    # Whether the fence took anything is remembered, so "nothing moved
    # because every move was into a neighbour" gets its OWN sentence
    # (B2 audit UX-M3).
    fence_bit = bool(neighbor_constraint and (new & ~fence).any())
    if neighbor_constraint:
        new &= fence
    new, hit, mean_um, max_um = two_sided_limit(
        mask, new, limit_um=limit_um, spacing=spacing)
    if neighbor_constraint:
        new &= fence
    base.update(hit_limit=bool(hit), mean_move_um=mean_um,
                max_move_um=max_um)

    result = np.where(new, label_id, 0).astype(volume_roi.dtype)
    n_mod = int(np.count_nonzero(new != mask))
    if n_mod == 0:
        base["reason"] = ("limit" if hit
                          else "fence" if fence_bit else None)
        return ContourResult(result, 0, (
            "Nothing moved." + (" " + traced_msg if traced_msg else "")),
            **base)
    n_out = int(np.count_nonzero(new & ~mask))
    n_in = int(np.count_nonzero(mask & ~new))
    msg = ("the outline moved on {:,} voxels - {:,} out, {:,} in, on "
           "average {:.1f} um and at most {:.1f} um.".format(
               n_mod, n_out, n_in, mean_um, max_um))
    return ContourResult(result, n_mod, msg, **base)



def contour_with_walls(
    volume_roi: np.ndarray,
    label_id: int,
    *,
    mode,
    wall_kind: str = "membrane",
    channel_roi: np.ndarray | None = None,
    prob_roi: np.ndarray | None = None,
    prep_chain: str = "none",
    blur_um: float = 2.0,
    spacing=(1.0, 1.0, 1.0),
    progress=None,
    cancel_event=None,
    **contour_kw,
) -> ContourResult:
    """Prepare the channel, build the wall picture, then run the contour --
    ALL OF IT on the worker (B2 audit C-M3).

    The wall picture used to be built on the main thread before the worker
    started: 1.48 s of frozen GUI on the owner's patch, before the busy
    strip had even appeared.  The orchestrator now hands over only the
    CROPPED inputs (the channel crop, the probability crop) and the
    decisions (wall kind, prep chain, blur), and this function does the
    arithmetic: ``apply_preproc`` + ``walls.build_elevation`` exactly as
    the main-thread path did, then :func:`contour_refine`.

    The picture comes back in ``details["elevation"]`` for the walls tick
    ("Show what the tool sees"); it is ``None`` in the brightness mode,
    which builds none.  ``progress("elevation", 0, 0)`` /
    ``("prepare", 0, 0)`` is emitted first, so the strip names the stage.
    """
    from .reshape.preproc import apply_preproc
    from .reshape.walls import build_elevation

    if isinstance(mode, (int, np.integer)):
        mode = CONTOUR_MODES[int(mode)]
    spacing = tuple(float(s) if float(s) > 0 else 1.0 for s in spacing)
    region = np.asarray(volume_roi) == int(label_id)
    elevation = image = None
    if mode == "snap":
        if progress is not None:
            progress("elevation", 0, 0)
        kind = str(wall_kind)
        if kind == "probability":
            if prob_roi is None:
                raise ValueError("this wall type needs the probability map")
            elevation, _c, _r = build_elevation(
                "probability", prob_roi, region, spacing=spacing)
        elif kind in ("distance", "skeleton"):
            elevation, _c, _r = build_elevation(
                kind, None, region, spacing=spacing)
        else:
            if channel_roi is None:
                raise ValueError("this wall type needs a channel")
            prepared = apply_preproc(channel_roi, prep_chain or "none")
            # `smooth` is in MIN-VOXEL units (axis_sigma), so the
            # micron blur is divided by the finest spacing here.
            elevation, _c, _r = build_elevation(
                kind, prepared, region, spacing=spacing,
                smooth=float(blur_um) / max(float(min(spacing)), 1e-6))
    else:
        if progress is not None:
            progress("prepare", 0, 0)
        if channel_roi is None:
            raise ValueError("Inside vs outside brightness needs a channel.")
        image = apply_preproc(channel_roi, prep_chain or "none")
    if cancel_event is not None and cancel_event.is_set():
        mask = region
        res = ContourResult(np.where(mask, int(label_id), 0).astype(
            np.asarray(volume_roi).dtype), 0, "Stopped.", reason="stopped")
        res.details["elevation"] = None
        return res
    res = contour_refine(
        volume_roi, label_id, mode=mode, elevation=elevation,
        image_roi=image, spacing=spacing, progress=progress,
        cancel_event=cancel_event, **contour_kw)
    res.details["elevation"] = elevation
    return res


def dilate_erode_label(
    volume_roi: np.ndarray,
    label_id: int,
    *,
    radius: int = 2,
    axis: str = "3D",
    neighbor_constraint: bool = True,
) -> tuple[np.ndarray, int, str]:
    """Morphological dilation (radius > 0) or erosion (radius < 0).

    Fast, predictable expansion/contraction of a single label.

    Parameters
    ----------
    volume_roi : np.ndarray
        3D label volume (ROI crop).
    label_id : int
        Which label to dilate or erode.
    radius : int
        Positive = dilate (expand), negative = erode (shrink), 0 = no-op.
    axis : str
        ``"3D"`` for full 3D, or ``"XY"``/``"XZ"``/``"YZ"`` for planar.
    neighbor_constraint : bool
        If True, dilation never expands into voxels occupied by other labels.

    Returns
    -------
    result_roi : np.ndarray
        Volume of same shape containing only ``label_id`` and 0.
    n_modified : int
        Number of voxels that changed.
    message : str
        Human-readable status message.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    mask = volume_roi == label_id
    if not np.any(mask):
        return np.zeros_like(volume_roi), 0, f"Label {label_id} not found in ROI."

    if radius == 0:
        result = np.zeros_like(volume_roi)
        result[mask] = label_id
        return result, 0, f"Label {label_id}: radius=0, no change."

    struct = _make_axis_structure(axis)

    if radius > 0:
        new_mask = binary_dilation(mask, structure=struct, iterations=radius)
        if neighbor_constraint:
            other_labels = (volume_roi != 0) & (volume_roi != label_id)
            new_mask = new_mask & ~other_labels
        action = "Dilated"
    else:
        new_mask = binary_erosion(mask, structure=struct, iterations=abs(radius))
        action = "Eroded"

    result = np.zeros_like(volume_roi)
    result[new_mask] = label_id

    n_modified = int(np.count_nonzero(new_mask != mask))
    if n_modified == 0:
        return result, 0, f"Label {label_id}: no voxels changed ({action.lower()})."

    return result, n_modified, (
        f"{action} label {label_id} by {abs(radius)}: {n_modified} voxels changed."
    )


def guided_grow(
    volume_roi: np.ndarray,
    label_id: int,
    *,
    image_roi: np.ndarray | None = None,
    prob_roi: np.ndarray | None = None,
    max_radius: int = 5,
    prob_threshold: float = 0.5,
    intensity_percentile: tuple[float, float] = (10.0, 90.0),
    axis: str = "3D",
    neighbor_constraint: bool = True,
) -> tuple[np.ndarray, int, str]:
    """Grow a label into eligible voxels guided by probability and/or intensity.

    Iterative bounded region-growing: each iteration expands by 1 voxel,
    but only into voxels that satisfy all active criteria (probability
    above threshold, intensity matching the label's interior distribution).

    Ideal for closing open surfaces on basal/periderm cells where the
    probability volume indicates cell interior but the membrane signal
    is absent.

    Parameters
    ----------
    volume_roi : np.ndarray
        3D label volume (ROI crop).
    label_id : int
        Which label to grow.
    image_roi : np.ndarray or None
        Raw membrane image (ROI crop).  When provided, growth is restricted
        to voxels whose intensity falls within the label's interior
        percentile range.

        **UNREACHABLE FROM THE UI since REFINE** -- the Refine tab
        never passes it.  Measured 2026-09-21: the object's own p10-p90
        interior band admitted 82.0 % of the region on a membrane channel
        (the probability criterion admits 18.6 %) and leaked straight
        through an open top, Dice 0.914 -> 0.508.  The parameter and its
        code stay so a caller that still passes it keeps working; read the
        measurement before re-wiring it to a control.
    prob_roi : np.ndarray or None
        Probability volume (ROI crop).  When provided, growth is restricted
        to voxels where probability >= ``prob_threshold``.
    max_radius : int
        Maximum number of dilation iterations (caps how far growth extends).
    prob_threshold : float
        Minimum probability for a voxel to be eligible (0.0–1.0).
    intensity_percentile : tuple of (lo, hi)
        Percentile range of the label's interior intensity distribution
        used to define eligible voxels (default: 10th–90th).
    axis : str
        ``"3D"`` for full 3D, or ``"XY"``/``"XZ"``/``"YZ"`` for planar.
    neighbor_constraint : bool
        If True, never grow into voxels occupied by other labels.

    Returns
    -------
    result_roi : np.ndarray
        Volume of same shape containing only ``label_id`` and 0.
    n_modified : int
        Number of voxels added.
    message : str
        Human-readable status message.
    """
    from scipy.ndimage import binary_dilation

    mask = volume_roi == label_id
    if not np.any(mask):
        return np.zeros_like(volume_roi), 0, f"Label {label_id} not found in ROI."

    if image_roi is None and prob_roi is None:
        return (
            np.zeros_like(volume_roi),
            0,
            "Guided grow requires an image and/or probability volume.",
        )

    # Build eligibility mask — must satisfy ALL active criteria.
    eligible = np.ones(volume_roi.shape, dtype=bool)

    # Only grow into background (empty) voxels.
    eligible &= volume_roi == 0

    # Neighbor constraint.
    if neighbor_constraint:
        other_labels = (volume_roi != 0) & (volume_roi != label_id)
        eligible &= ~other_labels

    # Probability criterion.
    if prob_roi is not None:
        prob = prob_roi.astype(np.float64)
        p_min, p_max = prob.min(), prob.max()
        if p_max > p_min:
            prob = (prob - p_min) / (p_max - p_min)
        eligible &= prob >= prob_threshold

    # Intensity criterion — match the label's interior distribution.
    if image_roi is not None:
        img = image_roi.astype(np.float64)
        interior_vals = img[mask]
        if interior_vals.size > 0:
            lo = np.percentile(interior_vals, intensity_percentile[0])
            hi = np.percentile(interior_vals, intensity_percentile[1])
            eligible &= (img >= lo) & (img <= hi)

    # Iterative bounded region-growing.
    struct = _make_axis_structure(axis)
    grown = mask.copy()
    for _ in range(max_radius):
        expanded = binary_dilation(grown, structure=struct) & eligible & ~grown
        if not np.any(expanded):
            break
        grown |= expanded

    result = np.zeros_like(volume_roi)
    result[grown] = label_id

    n_added = int(np.count_nonzero(grown & ~mask))
    if n_added == 0:
        return result, 0, f"Label {label_id}: no eligible voxels to grow into."

    sources = []
    if prob_roi is not None:
        sources.append("prob")
    if image_roi is not None:
        sources.append("intensity")
    source_str = "+".join(sources)

    return result, n_added, (
        f"Guided grow ({source_str}) label {label_id}: {n_added} voxels added."
    )


def fill_holes_label(
    volume_roi: np.ndarray,
    label_id: int,
    *,
    axis: str = "3D",
) -> tuple[np.ndarray, int, str]:
    """Fill internal holes/cavities within a label.

    Parameters
    ----------
    volume_roi : np.ndarray
        3D label volume (ROI crop).
    label_id : int
        Which label to fill holes in.
    axis : str
        ``"3D"`` fills all enclosed cavities in the full volume.
        ``"XY"``/``"XZ"``/``"YZ"`` fills holes slice-by-slice along
        the chosen axis.

    Returns
    -------
    result_roi : np.ndarray
        Volume of same shape containing only ``label_id`` and 0.
    n_modified : int
        Number of voxels that were filled.
    message : str
        Human-readable status message.
    """
    from scipy.ndimage import binary_fill_holes

    mask = volume_roi == label_id
    if not np.any(mask):
        return np.zeros_like(volume_roi), 0, f"Label {label_id} not found in ROI."

    other = (volume_roi != 0) & (volume_roi != label_id)

    if axis == "3D":
        filled = binary_fill_holes(mask)
    else:
        filled = mask.copy()
        ax = {"XY": 0, "XZ": 1, "YZ": 2}.get(axis, 0)
        for i in range(filled.shape[ax]):
            slc = [slice(None)] * 3
            slc[ax] = i
            filled[tuple(slc)] = binary_fill_holes(mask[tuple(slc)])

    # Only newly filled background voxels — never overwrite other labels.
    new_mask = filled & ~mask & ~other

    result = np.zeros_like(volume_roi)
    result[mask | new_mask] = label_id
    n_modified = int(np.count_nonzero(new_mask))

    if n_modified == 0:
        return result, 0, f"Label {label_id}: no holes to fill."

    return result, n_modified, (
        f"Fill holes ({axis}) label {label_id}: {n_modified} voxels filled."
    )


#: Flood Fill's two refusals that are about WHERE the crosshair is (B3 audit,
#: HIGH).  They are module constants because the orchestrator shows them as
#: the state-9 sentence verbatim, and the guide quotes them.
FLOOD_SEED_OUTSIDE = (
    "Nothing moved. The crosshair is outside the region around this object. "
    "Put the crosshair on or next to this object."
)
FLOOD_NO_CONTACT = (
    "Nothing moved. The empty space at the crosshair does not touch this "
    "object."
)
FLOOD_NOT_ENCLOSED = (
    "Nothing moved. The empty space at the crosshair runs out past the "
    "region around this object, so it is open background, not a gap in it."
)


def flood_fill_label(
    volume_roi: np.ndarray,
    label_id: int,
    seed_coord: tuple[int, int, int],
    *,
    axis: str = "3D",
    max_fill_size: int = 50000,
    walls=None,
) -> tuple[np.ndarray, int, str]:
    """Flood-fill connected background region at seed coordinate.

    Only fills background (0) voxels connected to the seed.
    Never overwrites existing labels.

    Parameters
    ----------
    volume_roi : np.ndarray
        3D label volume (ROI crop).
    label_id : int
        Label ID to paint into the filled region.
    seed_coord : tuple of (z, y, x)
        Seed coordinate in the ROI volume.
    axis : str
        ``"3D"`` fills the full connected background region.
        ``"XY"``/``"XZ"``/``"YZ"`` fills only the 2D connected region
        on the seed's slice along that axis.
    max_fill_size : int
        Safety limit (voxels).  If the connected region exceeds this,
        the fill is skipped.  The SECOND guard since the B3 audit: the
        first two are below.
    walls : sequence of 6 bools, optional
        ``(z_lo, z_hi, y_lo, y_hi, x_lo, x_hi)`` -- which faces of the ROI
        are an INTERIOR wall (the crop stopped there) rather than the edge of
        the whole volume.  A region that reaches an interior wall runs on
        past the region around the object: it is open background, not a gap
        in the object, and is refused (``FLOOD_NOT_ENCLOSED``).  ``None``
        (the default) checks no face, so a caller that passes the whole
        volume behaves as before.

    Two more refusals, both measured in the B3 audit (a stray click on
    background inside the object's box filled a 1,584-voxel slab):

    * the region must TOUCH the object (26-connectivity) --
      ``FLOOD_NO_CONTACT``;
    * a seed outside the ROI says where to put the crosshair --
      ``FLOOD_SEED_OUTSIDE``.

    Returns
    -------
    result_roi : np.ndarray
        Volume of same shape containing only ``label_id`` and 0.
    n_modified : int
        Number of voxels filled.
    message : str
        Human-readable status message.
    """
    from scipy.ndimage import label as ndlabel

    mask = volume_roi == label_id

    # Validate seed bounds.
    sz, sy, sx = seed_coord
    shape = volume_roi.shape
    if not (0 <= sz < shape[0] and 0 <= sy < shape[1] and 0 <= sx < shape[2]):
        result = np.zeros_like(volume_roi)
        result[mask] = label_id
        return result, 0, FLOOD_SEED_OUTSIDE

    # Seed must be on background.
    if volume_roi[sz, sy, sx] != 0:
        result = np.zeros_like(volume_roi)
        result[mask] = label_id
        return result, 0, (
            f"Seed {seed_coord} is not empty (background). "
            "Position crosshairs on the void you want to fill."
        )

    if axis == "3D":
        bg_mask = volume_roi == 0
        labeled, _ = ndlabel(bg_mask)
        component_id = labeled[sz, sy, sx]
        component = labeled == component_id
    else:
        ax = {"XY": 0, "XZ": 1, "YZ": 2}.get(axis, 0)
        # Extract 2D slice at the seed's axis position.
        slc_3d = [slice(None)] * 3
        seed_list = [sz, sy, sx]
        slc_3d[ax] = seed_list[ax]
        slice_2d = volume_roi[tuple(slc_3d)]
        bg_2d = slice_2d == 0

        labeled_2d, _ = ndlabel(bg_2d)
        # 2D seed coordinate (drop the axis dimension).
        seed_2d = tuple(s for i, s in enumerate(seed_list) if i != ax)
        component_id = labeled_2d[seed_2d]
        component_2d = labeled_2d == component_id

        # Map back to 3D.
        component = np.zeros(volume_roi.shape, dtype=bool)
        component[tuple(slc_3d)] = component_2d

    unchanged = np.zeros_like(volume_roi)
    unchanged[mask] = label_id

    # Guard 1 -- it must touch the object, in any of the 26 directions.
    from scipy.ndimage import binary_dilation as _dil

    touch = _dil(mask, structure=np.ones((3, 3, 3), bool))
    if not np.any(component & touch):
        return unchanged, 0, FLOOD_NO_CONTACT

    # Guard 2 -- it must not run out through an interior wall of the ROI.
    if walls is not None:
        faces = (
            (0, 0), (0, -1), (1, 0), (1, -1), (2, 0), (2, -1),
        )
        for inside, (ax_, end) in zip(walls, faces):
            if not inside:
                continue
            sl = [slice(None)] * 3
            sl[ax_] = end
            if np.any(component[tuple(sl)]):
                return unchanged, 0, FLOOD_NOT_ENCLOSED

    comp_size = int(np.count_nonzero(component))
    if comp_size > max_fill_size:
        result = np.zeros_like(volume_roi)
        result[mask] = label_id
        return result, 0, (
            f"Connected region ({comp_size} voxels) exceeds max_fill_size "
            f"({max_fill_size}). Region too large — increase limit or use "
            "per-slice mode."
        )

    result = np.zeros_like(volume_roi)
    result[mask | component] = label_id
    n_modified = comp_size

    if n_modified == 0:
        return result, 0, f"Label {label_id}: no background to fill at seed."

    return result, n_modified, (
        f"Flood fill ({axis}) label {label_id}: {n_modified} voxels filled."
    )

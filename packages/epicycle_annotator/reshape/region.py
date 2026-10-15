"""The Add region target -- "Reshape on empty space" (Add rebuild).

Inside a box, markers stop each other, walls come from the membrane, and new
objects are written only where the volume is 0.  Existing objects never lose
or gain a voxel: this module RETURNS a box-sized partition holding ids
``1..N`` on empty voxels only; the Add tab's ``_accept_add`` is the only
writer and the Add worker mints the final ids.

The algorithm, each stage polling ``cancel_event``:

1. **Budget** -- a box over ``params["box_budget"]`` voxels is refused
   before anything is cropped.
2. **Walls** -- ``engine._build_wall_stage`` unchanged, on the WHOLE box
   (the solve domain is the whole box, never ``existing == 0``).
3. **Dark** -- with ``outside="faces+dark"``, every image channel of the
   patch (``dark_channels``), normalised with the PATCH's own percentiles,
   max-projected, blurred by :data:`DARK_BLUR_UM` and cut at
   ``dark_otsu_factor x`` the patch Otsu.
4. **New-object markers** ``1..N`` -- run cores (``labels``), sliver-
   filtered nuclei pieces (``nuclei_masks``) or click balls (``paint``);
   automatic markers mostly in the dark are dropped, clicks never.
5. **Kept neighbours** -- every kept id in the box, CORED, as a marker.
6. **Outside** -- the six box faces, plus the eroded dark mask.
7. **Cut** -- one seeded watershed over the whole box.
8. **Keep only new objects, one piece each** (``keep_marker_components``).
9. **Notes** -- faint-wall pairs and the unfilled share of the tissue.

Unlike the label path, a cancel here RAISES :class:`~.engine.ReshapeCancelled`;
the Add worker turns it into its own ``cancelled`` answer.

Pure: numpy / scipy / skimage, no Qt, no napari.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .engine import (
    ReshapeCancelled,
    SeedReport,
    _build_wall_stage,
)
from .notes import MessageSink
from .preproc import safe_otsu
from .solvers import solve
from .spec import REGION_SEEDS, ReshapeResult, ReshapeSpec

__all__ = [
    "ADD_ENGINE_NO_MARKERS",
    "ADD_ENGINE_TOO_BIG",
    "DARK_BLUR_UM",
    "DARK_SAMPLE_VOXELS",
    "NUCLEI_SLIVER_UM",
    "REGION_SEEDS",
    "REGION_STAGES",
    "dark_mask",
    "drop_slivers",
    "eroded_cores",
    "faint_wall_pairs",
    "find_region_seeds",
    "keep_marker_components",
    "patch_dark_threshold",
    "rasterise_points",
    "reshape_empty_region",
    "too_big_sentence",
    "unfilled_fraction",
]

# --------------------------------------------------------------------------- #
# constants (module-level, so no UI can reach them)
# --------------------------------------------------------------------------- #
#: The physical sigma of the dark mask's blur, per axis.
DARK_BLUR_UM = 2.0

#: The most voxels the box-info worker reads to take the patch threshold;
#: the caller picks a stride that brings the patch under it.
DARK_SAMPLE_VOXELS = 1_000_000

#: A nuclei piece is a marker only if it reaches at least this far from
#: every kept voxel (Wound: 103-294 false pieces per box without it).
NUCLEI_SLIVER_UM = 1.0

#: Share of an automatic marker's (post-clip) voxels in the dark at which it
#: is dropped.
DARK_DROP_SHARE = 0.5

#: The marker sources the region target reads -- ONE tuple, defined in
#: ``spec.py`` (which refuses any other source on ``target="region"``) and
#: re-exported here.
# REGION_SEEDS is imported from .spec above.

#: The ``progress`` stage names of :func:`reshape_empty_region`, in order.
#: Each is emitted as its stage STARTS, so a present-tense strip ("Filling
#: the empty space...") names the work under way.  The Add worker maps them
#: to ``STAGE_TEXT``.
REGION_STAGES = ("add_box", "elevation", "add_markers", "add_cut")

# -- the engine's sentences ---------------------------- #
_TOO_BIG_TEMPLATE = (
    "This box is too large to fill in one go — the limit is {} "
    "voxels. (Nothing was changed.)"
)


def _voxel_count_words(n):
    """``4_000_000`` -> ``"4 million"``; ``2_500_000`` -> ``"2.5 million"``;
    under a million, digits with thousands separators (``"750,000"``)."""
    n = int(n)
    if n >= 1_000_000:
        return "{:g} million".format(n / 1_000_000)
    return "{:,}".format(n)


def too_big_sentence(budget):
    """The too-big refusal for a box budget of ``budget`` voxels."""
    return _TOO_BIG_TEMPLATE.format(_voxel_count_words(budget))


#: The refusal at the default budget (``spec.ADD_BOX_BUDGET``); B2 imports
#: it by name.  A refusal under any other budget is ``too_big_sentence``.
ADD_ENGINE_TOO_BIG = too_big_sentence(4_000_000)
ADD_ENGINE_NO_MARKERS = (
    "No marker was left to grow from — every marker fell in dark space, "
    "over kept objects, or was too small to use. (Nothing was changed.)"
)

#: Log-only diagnostic (not a user sentence; never in ``messages``).
_LOG_NEIGHBOUR_NO_CORE = (
    "[ADD-ENGINE] {} kept object(s) in the box have no core after the "
    "erosion; they hold back nothing."
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise ReshapeCancelled()


class _RegionProgress:
    """``(stage, i, n)`` over :data:`REGION_STAGES`; a raising callback is
    swallowed, as the engine's own ``_Progress`` does."""

    __slots__ = ("_cb", "_i")

    def __init__(self, callback):
        self._cb = callback
        self._i = 0

    def __call__(self, stage):
        self._i += 1
        if self._cb is None:
            return
        try:
            self._cb(stage, self._i, len(REGION_STAGES))
        except Exception:
            pass


def _box_shape(roi):
    if roi is None:
        raise ValueError("target='region' needs roi=<the box, a tuple of 3 slices>.")
    shape = []
    for axis, sl in enumerate(roi):
        if not isinstance(sl, slice) or sl.stop is None or sl.step not in (None, 1):
            raise ValueError(
                "target='region' needs roi= as three slices with explicit "
                "stops and no step; got {!r}.".format(roi))
        start = int(sl.start or 0)
        if start < 0 or int(sl.stop) < 0:
            raise ValueError(
                "the region roi {!r} starts before the volume on axis {} "
                "(a box never wraps round).".format(roi, axis))
        shape.append(int(sl.stop) - start)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("the region roi {!r} is not a 3D box.".format(roi))
    return tuple(shape)


def _region_crop(array, roi, shape, *, what):
    """The box of ``array``, read by SLICING FIRST and then copying.

    An array already the box's shape is taken as the box crop (the Add
    worker's usual input).  Anything else is the full volume: the box must
    lie inside it (a box past its edge is refused, naming the axis), and
    only ``array[roi]`` is ever read -- never ``np.asarray`` of the whole of
    a lazy or multiscale layer.  Always a private copy.
    """
    ashape = getattr(array, "shape", None)
    if ashape is None:
        array = np.asarray(array)
        ashape = array.shape
    ashape = tuple(int(n) for n in ashape)
    if ashape == tuple(shape):
        key = (slice(None),) * len(ashape)
    else:
        if len(ashape) != len(roi):
            raise ValueError(
                "{} is {}-dimensional; the region target takes 3D "
                "single-channel arrays.".format(what, len(ashape)))
        for axis, (sl, n) in enumerate(zip(roi, ashape)):
            if int(sl.stop) > n:
                raise ValueError(
                    "the box {!r} runs past the edge of {} on axis {} "
                    "(stop {} > size {}).".format(roi, what, axis,
                                                   int(sl.stop), n))
        key = tuple(roi)
    try:
        cropped = np.array(array[key], copy=True)
    except Exception as exc:
        raise ValueError("{} could not be cropped to the box: {!r}".format(
            what, exc))
    if cropped.shape != tuple(shape):
        raise ValueError(
            "{} has shape {} which matches neither the box {} nor a volume "
            "holding it.".format(what, ashape, tuple(shape)))
    return cropped


def _region_piece_stats(partition, roi):
    """``engine._piece_stats`` in ONE ``find_objects`` pass (audit M-A1).

    Same rows, same values, same order: per id, the voxels of its bounding
    box are scanned in the same row-major order, and the box start is added
    to the integer coordinates BEFORE the mean, so every float is the one
    the engine computes.  The engine's version scans the whole box once per
    id (O(N x V)); this one reads each id's bounding box once.
    """
    part = np.asarray(partition)
    ids = _ids_of(part)
    if ids.size == 0:
        return []
    offset = np.array([sl.start or 0 for sl in roi], float)
    if int(ids[-1]) > 4 * int(ids.size) + 1024:
        # find_objects allocates one slot per id up to the max
        lab, _n = _relabel_ascending(part)
    else:
        lab = part
        ids = None
    rows = []
    for idx, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        pid = int(idx if ids is None else ids[idx - 1])
        start = np.array([s.start for s in sl], np.int64)
        coords = np.argwhere(lab[sl] == idx) + start
        centroid = coords.mean(axis=0) + offset
        lo = coords.min(axis=0) + offset.astype(int)
        hi = coords.max(axis=0) + offset.astype(int)
        rows.append({
            "id": pid,
            "voxels": int(coords.shape[0]),
            "centroid_zyx": tuple(float(v) for v in centroid),
            "bbox": (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]),
                     int(lo[2]), int(hi[2])),
        })
    return rows


def _spacing(spacing):
    return tuple(float(s) for s in spacing)


def _blur_sigma(blur_um, spacing):
    return tuple(float(blur_um) / float(s) for s in spacing)


def _relabel_ascending(markers):
    """Renumber the non-zero ids of ``markers`` to ``1..N`` in ascending
    original-id order.  Returns ``(relabelled int32, N)``."""
    ids = np.unique(markers)
    ids = ids[ids != 0]
    out = np.zeros(markers.shape, np.int32)
    if ids.size == 0:
        return out, 0
    nz = markers != 0
    out[nz] = (np.searchsorted(ids, markers[nz]) + 1).astype(np.int32)
    return out, int(ids.size)


def _ids_of(array):
    ids = np.unique(array)
    return ids[ids != 0]


# --------------------------------------------------------------------------- #
# the pure pieces
# --------------------------------------------------------------------------- #
def _normalised_max(channels, lo, hi):
    out = None
    for name, arr in channels.items():
        if name not in lo or name not in hi:
            raise ValueError(
                "the dark threshold has no normalisation for channel {!r} "
                "(it has {}).".format(name, ", ".join(sorted(lo)) or "none"))
        a = np.asarray(arr, np.float32)
        span = max(float(hi[name]) - float(lo[name]), 1e-6)
        a = np.clip((a - float(lo[name])) / span, 0.0, 1.0)
        out = a if out is None else np.maximum(out, a)
    if out is None:
        raise ValueError("the dark rule needs at least one image channel.")
    return out


def patch_dark_threshold(channels, *, spacing, stride, blur_um=DARK_BLUR_UM):
    """The patch's dark threshold ``{"t", "lo", "hi"}`` from a strided sample.

    ``channels`` maps a layer name to that channel of the WHOLE patch
    (anything sliceable -- only ``[::stride]`` of it is read).  Each channel's
    ``lo`` / ``hi`` are its 1st / 99.5th percentiles over the sample; the
    normalised channels are max-projected, blurred with sigma
    ``blur_um`` at the SAMPLE's spacing (``spacing x stride``, so the blur
    stays ``blur_um`` in physical units), and ``t`` is the Otsu of that.
    """
    stride = max(1, int(stride))
    spacing = _spacing(spacing)
    samples = {}
    lo, hi = {}, {}
    for name, arr in dict(channels or {}).items():
        a = np.array(arr[::stride, ::stride, ::stride], np.float32, copy=True)
        if a.ndim != 3:
            raise ValueError(
                "channel {!r} is {}-dimensional; the dark rule takes 3D "
                "single-channel arrays.".format(name, a.ndim))
        samples[name] = a
        lo[name] = float(np.percentile(a, 1))
        hi[name] = float(np.percentile(a, 99.5))
    img = _normalised_max(samples, lo, hi)
    sample_spacing = tuple(s * stride for s in spacing)
    blurred = ndi.gaussian_filter(img, _blur_sigma(blur_um, sample_spacing))
    return {"t": float(safe_otsu(blurred)), "lo": lo, "hi": hi}


def dark_mask(channels_roi, *, dark_threshold, factor, spacing,
              blur_um=DARK_BLUR_UM):
    """Dark space in the box: the blurred, patch-normalised max projection of
    ``channels_roi`` (``{layer name: box crop}``) below ``factor x t``."""
    img = _normalised_max(dict(channels_roi or {}), dark_threshold["lo"],
                          dark_threshold["hi"])
    blurred = ndi.gaussian_filter(img, _blur_sigma(blur_um, _spacing(spacing)))
    return blurred < float(factor) * float(dark_threshold["t"])


def eroded_cores(label_crop, *, erode_um, spacing, cancel_event=None):
    """Shrink every object by ``erode_um`` (EDT inside each object, true
    spacing).  Returns ``(cores, n_emptied)`` -- ``cores`` keeps the original
    ids; ``n_emptied`` counts objects the erosion emptied (no fallback).

    The crop's own edge is NOT a boundary: an object cut by the box keeps
    its core up to the box face.
    """
    original = np.asarray(label_crop)
    # compact first: `find_objects` allocates one slot per id up to the max,
    # and a run's ids can be large (the batch brick's stray-huge-id lesson)
    lab, _n = _relabel_ascending(original)
    ids = _ids_of(original)
    cores = np.zeros(lab.shape, np.int32)
    erode_um = float(erode_um)
    spacing = _spacing(spacing)
    n_emptied = 0
    slices = ndi.find_objects(lab) if _n else []
    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        _check_cancel(cancel_event)
        grown = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1))
                      for s, n in zip(sl, lab.shape))
        mask = lab[grown] == idx
        if erode_um <= 0.0:
            core = mask
        elif mask.all():
            core = mask       # no boundary in reach inside the crop
        else:
            core = ndi.distance_transform_edt(mask, sampling=spacing) > erode_um
        if not core.any():
            n_emptied += 1
            continue
        view = cores[grown]
        view[core] = idx
    out = np.zeros(original.shape, original.dtype)
    nz = cores != 0
    out[nz] = ids[cores[nz] - 1]
    return out, n_emptied


def drop_slivers(seed_crop, existing_roi, *, min_depth_um, spacing,
                 cancel_event=None):
    """Keep a nuclei piece only if it reaches ``min_depth_um`` from every kept
    voxel.  Returns ``(kept, n_dropped)``; pieces are clipped to
    ``existing == 0`` first and keep their ids."""
    seeds = np.asarray(seed_crop)
    existing = np.asarray(existing_roi)
    empty = existing == 0
    seeds = np.where(empty, seeds, 0)
    ids = _ids_of(seeds)
    if ids.size == 0 or empty.all():
        return seeds, 0
    _check_cancel(cancel_event)
    depth = ndi.distance_transform_edt(empty, sampling=_spacing(spacing))
    reach = ndi.maximum(depth, labels=seeds, index=ids)
    drop = ids[np.asarray(reach) < float(min_depth_um) - 1e-9]
    if drop.size:
        seeds = np.where(np.isin(seeds, drop), 0, seeds)
    return seeds, int(drop.size)


def rasterise_points(points, shape, *, radius_um, spacing):
    """Balls of physical radius ``radius_um`` at box-local ZYX ``points``,
    numbered ``1..`` in placement order (an earlier ball keeps a shared
    voxel).  The centre voxel is always in its ball."""
    shape = tuple(int(n) for n in shape)
    out = np.zeros(shape, np.int32)
    sp = np.asarray(_spacing(spacing), float)
    r = float(radius_um)
    half = [int(np.ceil(r / s)) for s in sp]
    for i, point in enumerate(points or (), start=1):
        p = np.asarray(point, float)[-3:]
        c = np.rint(p).astype(int)
        box = tuple(slice(max(0, c[k] - half[k]), min(shape[k], c[k] + half[k] + 1))
                    for k in range(3))
        if any(b.start >= b.stop for b in box):
            continue
        grids = np.ogrid[box]
        d2 = sum(((g - p[k]) * sp[k]) ** 2 for k, g in enumerate(grids))
        ball = d2 <= r * r + 1e-9
        if all(0 <= c[k] < shape[k] for k in range(3)):
            ball[tuple(c[k] - box[k].start for k in range(3))] = True
        view = out[box]
        view[ball & (view == 0)] = i
    return out


def keep_marker_components(result, markers, *, cancel_event=None):
    """Per id, keep only the 6-connected component(s) holding a voxel of that
    id's marker; zero the rest.  Returns ``(result, n_dropped_pieces)``."""
    res = np.array(result, copy=True)
    mk = np.asarray(markers)
    n_dropped = 0
    structure = ndi.generate_binary_structure(3, 1)
    slices = ndi.find_objects(res) if res.size and res.max() > 0 else []
    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        _check_cancel(cancel_event)
        mask = res[sl] == idx
        comp, n = ndi.label(mask, structure=structure)
        if n <= 1 and (mk[sl][mask] == idx).any():
            continue
        keep = np.unique(comp[mask & (mk[sl] == idx)])
        keep = keep[keep != 0]
        drop = mask & ~np.isin(comp, keep)
        n_dropped += int(n - keep.size)
        view = res[sl]
        view[drop] = 0
    return res, n_dropped


def faint_wall_pairs(result, elevation, *, percentile):
    """Touching new-object pairs on a faint wall, weakest first.

    One pass over the three axis-shifted neighbour pairs: for every touching
    ``(a, b)`` the elevation of BOTH interface voxels is collected; a pair
    is flagged when its median is below ``np.percentile(elevation[result >
    0], percentile)``.  Returns ``[(a, b, median), ...]`` sorted by median.
    """
    res = np.asarray(result)
    elev = np.asarray(elevation, np.float32)
    inside = res > 0
    if not inside.any():
        return []
    threshold = float(np.percentile(elev[inside], float(percentile)))
    keys, vals = [], []
    top = int(res.max()) + 1
    for axis in range(res.ndim):
        a = np.moveaxis(res, axis, 0)
        e = np.moveaxis(elev, axis, 0)
        a0, a1 = a[:-1], a[1:]
        touch = (a0 > 0) & (a1 > 0) & (a0 != a1)
        if not touch.any():
            continue
        x, y = a0[touch].astype(np.int64), a1[touch].astype(np.int64)
        k = np.minimum(x, y) * top + np.maximum(x, y)
        keys.append(np.concatenate([k, k]))
        vals.append(np.concatenate([e[:-1][touch], e[1:][touch]]))
    if not keys:
        return []
    keys = np.concatenate(keys)
    vals = np.concatenate(vals)
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]
    uniq, start = np.unique(keys, return_index=True)
    bounds = list(start) + [keys.size]
    out = []
    for j, key in enumerate(uniq):
        med = float(np.median(vals[bounds[j]:bounds[j + 1]]))
        if med < threshold:
            out.append((int(key // top), int(key % top), med))
    out.sort(key=lambda t: (t[2], t[0], t[1]))
    return out


def unfilled_fraction(result, existing_roi, dark_raw):
    """|empty, not dark, not filled| / |empty, not dark| -- ``None`` without a
    dark mask (``outside="faces"``) or when there is no such tissue."""
    if dark_raw is None:
        return None
    tissue = (np.asarray(existing_roi) == 0) & ~np.asarray(dark_raw, bool)
    denominator = int(np.count_nonzero(tissue))
    if denominator == 0:
        return None
    unfilled = int(np.count_nonzero(tissue & (np.asarray(result) == 0)))
    return unfilled / denominator


# --------------------------------------------------------------------------- #
# the shared stages 1, 3, 4
# --------------------------------------------------------------------------- #
class _Refused(Exception):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason
        self.message = message


def _check_spec(spec):
    if not isinstance(spec, ReshapeSpec):
        raise TypeError("spec must be a ReshapeSpec, got {!r}".format(type(spec)))
    if spec.target != "region":
        raise ValueError("reshape/region.py takes target='region' only "
                         "(got {!r}).".format(spec.target))
    if spec.seeds not in REGION_SEEDS:
        raise ValueError(
            "seeds={!r} is not a marker source of the region target. "
            "Allowed: {}.".format(spec.seeds, ", ".join(REGION_SEEDS)))


def _budget(spec, shape):
    n = int(np.prod(shape, dtype=np.int64))
    budget = int(spec.params["box_budget"])
    if n > budget:
        raise _Refused("too-big", too_big_sentence(budget))
    return n


def _dark_stage(spec, *, roi, shape, dark_channels, dark_threshold, spacing):
    """``dark_raw`` (bool, box-sized) or ``None`` with ``outside="faces"``."""
    if spec.params["outside"] != "faces+dark":
        return None
    if dark_threshold is None:
        raise ValueError(
            "outside='faces+dark' needs dark_threshold= (the patch threshold "
            "from patch_dark_threshold).")
    if not dark_channels:
        raise ValueError(
            "outside='faces+dark' needs dark_channels= (every image channel "
            "of the patch, cropped to the box).")
    crops = {name: _region_crop(arr, roi, shape,
                                what="dark channel {!r}".format(name))
             for name, arr in dict(dark_channels).items()}
    return dark_mask(crops, dark_threshold=dark_threshold,
                     factor=spec.params["dark_otsu_factor"], spacing=spacing)


def _drop_dark(markers, dark_raw):
    """Drop automatic markers with >= 50 % of their voxels in the dark."""
    if dark_raw is None:
        return markers, 0
    ids = _ids_of(markers)
    if ids.size == 0:
        return markers, 0
    total = ndi.sum(np.ones(markers.shape, np.float64), labels=markers, index=ids)
    in_dark = ndi.sum(dark_raw.astype(np.float64), labels=markers, index=ids)
    share = np.asarray(in_dark) / np.maximum(np.asarray(total), 1.0)
    drop = ids[share >= DARK_DROP_SHARE]
    if drop.size:
        markers = np.where(np.isin(markers, drop), 0, markers)
    return markers, int(drop.size)


def _marker_stage(spec, *, roi, shape, existing, seed_layer, seed_points,
                  dark_raw, spacing, cancel_event):
    """Stage 4: the new-object markers ``1..N`` and their counts."""
    p = spec.params
    empty = existing == 0
    counts = {"n_dropped_thin": 0, "n_markers_over_kept": 0,
              "n_nuclei_slivers": 0, "n_dropped_dark": 0}
    if spec.seeds == "paint":
        balls = rasterise_points(seed_points, shape,
                                 radius_um=p["click_radius_um"], spacing=spacing)
        n_found = int(len(seed_points or ()))
        balls = np.where(empty, balls, 0)
        markers, n = _relabel_ascending(balls)   # placement order == id order
        return markers, n, n_found, counts, "clicked markers"

    if seed_layer is None:
        raise ValueError(
            "seeds={!r} needs seed_layer=<array> (another run's labels, or "
            "the bright-object pieces).".format(spec.seeds))
    crop = _region_crop(seed_layer, roi, shape, what="the seed layer")
    n_found = int(_ids_of(crop).size)

    if spec.seeds == "labels":
        cores, _n_thin = eroded_cores(crop, erode_um=p["core_erode_um"],
                                      spacing=spacing, cancel_event=cancel_event)
        # an emptied object counts as "too thin" only if it
        # had at least one voxel on empty space before the erosion
        emptied = np.setdiff1d(_ids_of(crop), _ids_of(cores))
        on_empty = _ids_of(np.where(empty, crop, 0))
        counts["n_dropped_thin"] = int(np.intersect1d(emptied, on_empty).size)
        before = _ids_of(cores)
        cores = np.where(empty, cores, 0)
        counts["n_markers_over_kept"] = int(before.size - _ids_of(cores).size)
        source = "another run's objects"
    else:  # nuclei_masks
        before = _ids_of(np.where(empty, crop, 0))
        counts["n_markers_over_kept"] = int(n_found - before.size)
        cores, n_sliv = drop_slivers(crop, existing, min_depth_um=NUCLEI_SLIVER_UM,
                                     spacing=spacing, cancel_event=cancel_event)
        counts["n_nuclei_slivers"] = n_sliv
        source = "bright objects (nuclei)"
    _check_cancel(cancel_event)
    cores, counts["n_dropped_dark"] = _drop_dark(cores, dark_raw)
    markers, n = _relabel_ascending(cores)
    return markers, n, n_found, counts, source


def _prepare(labels, spec, roi, spacing, cancel_event):
    _check_spec(spec)
    shape = _box_shape(roi)
    _check_cancel(cancel_event)
    _budget(spec, shape)
    existing = _region_crop(labels, roi, shape, what="the labels")
    return shape, existing


def _wall_images(spec, images, roi, shape, supplied_elevation):
    """``images`` with the wall channel already cut to the box, so the
    engine's ``_resolve_channel`` (which reads its array WHOLE) only ever
    sees the crop.  Other entries are passed through unread."""
    images = dict(images or {})
    name = spec.params.get("wall_channel")
    if supplied_elevation is None and name and name in images:
        images[name] = _region_crop(images[name], roi, shape,
                                    what="the wall channel {!r}".format(name))
    return images


# --------------------------------------------------------------------------- #
# the entry points
# --------------------------------------------------------------------------- #
def find_region_seeds(labels, images, spec, *, roi, dark_channels=None,
                      seed_layer=None, seed_points=None, dark_threshold=None,
                      spacing=(1, 1, 1), cancel_event=None):
    """Stages 1, 3 and 4 only: how many markers the box would grow from.

    Same crop, same arguments, same cached dark threshold as
    :func:`reshape_empty_region`, so ``n_in_region`` equals that Preview's
    ``n_seeds``.  ``n_found`` = markers before drops.  ``images`` is unused
    (no automatic source reads the walls) and accepted for symmetry.

    ``report.markers`` stays EMPTY: the Add box-info line needs the two
    counts only, and per-marker statistics cost most of a Find on a packed
    box (audit M-A1).  Array inputs follow :func:`reshape_empty_region`.
    """
    spacing = _spacing(spacing)
    sink = MessageSink(None)
    report = SeedReport(roi=roi)
    try:
        shape, existing = _prepare(labels, spec, roi, spacing, cancel_event)
        dark_raw = _dark_stage(spec, roi=roi, shape=shape,
                               dark_channels=dark_channels,
                               dark_threshold=dark_threshold, spacing=spacing)
        _check_cancel(cancel_event)
        markers, n, n_found, _counts, source = _marker_stage(
            spec, roi=roi, shape=shape, existing=existing, seed_layer=seed_layer,
            seed_points=seed_points, dark_raw=dark_raw, spacing=spacing,
            cancel_event=cancel_event)
        report.seed_source = source
        report.seeds = markers
        report.n_found = n_found
        report.n_in_region = n
        if n < 1:
            raise _Refused("no-markers", ADD_ENGINE_NO_MARKERS)
    except _Refused as refusal:
        report.status = "refused"
        report.refusal_reason = refusal.reason
        sink.emit(refusal.message)
    report.messages = list(sink.messages)
    return report


def reshape_empty_region(labels, images, spec, *, roi, dark_channels=None,
                         seed_layer=None, seed_points=None, dark_threshold=None,
                         spacing=(1, 1, 1), cancel_event=None, log=None,
                         progress=None, elevation=None,
                         keep_intermediates=False) -> ReshapeResult:
    """Fill the empty space of the box ``roi`` with new objects.

    **Pass box crops** for ``labels``, ``seed_layer``, the wall channel and
    every ``dark_channels`` entry; ``roi`` is the box in VOLUME
    coordinates (the stats are offset by it).  A full-volume array is
    sliced to the box BEFORE it is read (never ``np.asarray`` of the whole
    of a lazy or multiscale layer); an array the box's own shape is taken
    as the crop.  A box starting below 0 or ending past a full volume's
    edge is a ``ValueError`` naming the axis.  ``images`` holds the wall
    channel under ``params["wall_channel"]`` (a KEY, e.g. ``"membrane"``);
    ``dark_channels`` holds every image channel of the patch by layer name.

    Returns a :class:`ReshapeResult` whose ``result`` is **int32**, the box
    shape, holding ids ``1..N`` (N = ``n_seeds``, some possibly absent)
    on ``existing == 0`` only -- the Add worker mints the final ids -- or a
    refusal carrying :func:`too_big_sentence` (``ADD_ENGINE_TOO_BIG`` at the
    default budget) / :data:`ADD_ENGINE_NO_MARKERS`.
    ``unfilled_tissue_fraction`` measures against the dark mask AFTER the
    clearance around clicks, so a cleared ball counts as tissue.  Progress
    names (:data:`REGION_STAGES`) are emitted as each stage STARTS; a cancel
    is polled between and inside stages and RAISES
    :class:`~.engine.ReshapeCancelled`.
    """
    spacing = _spacing(spacing)
    sink = MessageSink(log)
    step = _RegionProgress(progress)
    p = spec.params if isinstance(spec, ReshapeSpec) else {}
    out = ReshapeResult(vox_volume=float(np.prod(spacing)), roi=roi,
                        dark_threshold=dark_threshold)
    try:
        # ---- 1. budget --------------------------------------------------- #
        step("add_box")
        shape, existing = _prepare(labels, spec, roi, spacing, cancel_event)
        _check_cancel(cancel_event)

        # ---- 2. walls, on the whole box ---------------------------------- #
        step("elevation")
        whole = np.ones(shape, bool)
        elev, clahe, ridge_img, _img = _build_wall_stage(
            spec, images=_wall_images(spec, images, roi, shape, elevation),
            roi=roi, region=whole, spacing=spacing, params=p, sink=sink,
            supplied_elevation=elevation)
        if keep_intermediates:
            kept_side = {"clahe": clahe, "ridge": ridge_img}
        del clahe, ridge_img, _img      # not held through the cut (L-A5)
        _check_cancel(cancel_event)

        # ---- 3. dark ------------------------------------------------------ #
        step("add_markers")
        dark_raw = _dark_stage(spec, roi=roi, shape=shape,
                               dark_channels=dark_channels,
                               dark_threshold=dark_threshold, spacing=spacing)
        _check_cancel(cancel_event)

        # ---- 4. new-object markers --------------------------------------- #
        new, n, n_found, counts, source = _marker_stage(
            spec, roi=roi, shape=shape, existing=existing, seed_layer=seed_layer,
            seed_points=seed_points, dark_raw=dark_raw, spacing=spacing,
            cancel_event=cancel_event)
        for key, value in counts.items():
            setattr(out, key, value)
        out.seed_source = source
        out.n_seeds_found = n_found
        out.n_seeds = n
        out.n_seeds_dropped = max(0, n_found - n)
        if n < 1:
            raise _Refused("no-markers", ADD_ENGINE_NO_MARKERS)
        _check_cancel(cancel_event)

        # ---- 5. kept neighbours, cored ----------------------------------- #
        cores, n_no_core = eroded_cores(existing, erode_um=p["core_erode_um"],
                                        spacing=spacing,
                                        cancel_event=cancel_event)
        if n_no_core and log is not None:
            try:
                log(_LOG_NEIGHBOUR_NO_CORE.format(n_no_core))
            except Exception:
                pass
        markers = new.copy()
        nb_ids = _ids_of(cores)
        if nb_ids.size:
            nz = cores != 0
            markers[nz] = (np.searchsorted(nb_ids, cores[nz]) + n + 2).astype(np.int32)

        # ---- 6. the outside marker --------------------------------------- #
        outside = np.zeros(shape, bool)
        outside[0, :, :] = outside[-1, :, :] = True
        outside[:, 0, :] = outside[:, -1, :] = True
        outside[:, :, 0] = outside[:, :, -1] = True
        if dark_raw is not None:
            if spec.seeds == "paint" and seed_points:
                clear = rasterise_points(seed_points, shape,
                                         radius_um=p["dark_clearance_um"],
                                         spacing=spacing) > 0
                dark_raw = dark_raw & ~clear
            reach = float(p["dark_clearance_um"])
            _check_cancel(cancel_event)
            if reach > 0:
                from .arbiter import physical_ball
                dark_out = ndi.binary_erosion(dark_raw,
                                              structure=physical_ball(reach, spacing))
            else:
                dark_out = dark_raw
            outside |= dark_out
        outside &= (existing == 0) & (new == 0)
        markers[outside & (markers == 0)] = n + 1
        _check_cancel(cancel_event)

        # ---- 7. the cut --------------------------------------------------- #
        step("add_cut")
        _check_cancel(cancel_event)
        partition = solve(elev, markers, whole, solver="watershed",
                          compactness=p["compactness"], spacing=spacing,
                          sink=sink)
        _check_cancel(cancel_event)

        # ---- 8. new objects only, one piece each -------------------------- #
        result = np.where((partition >= 1) & (partition <= n) & (existing == 0),
                          partition, 0).astype(np.int32)
        del partition
        result, out.n_fragments_dropped = keep_marker_components(
            result, new, cancel_event=cancel_event)
        _check_cancel(cancel_event)

        # ---- 9. notes ------------------------------------------------------ #
        out.faint_pairs = faint_wall_pairs(
            result, elev, percentile=p["arbiter_low_evidence_percentile"])
        _check_cancel(cancel_event)
        # measured against dark_raw AFTER the click clearance: a cleared
        # ball around a click counts as tissue
        out.unfilled_tissue_fraction = unfilled_fraction(result, existing, dark_raw)
        _check_cancel(cancel_event)

        # ---- 10. the answer ------------------------------------------------ #
        ids = _ids_of(result)
        sizes = np.bincount(result.ravel(), minlength=n + 1)[1:]
        out.result = result
        out.seeds = new
        out.elevation = elev
        out.n_pieces = int(ids.size)
        out.piece_sizes = sorted((int(sizes[i - 1]) for i in ids), reverse=True)
        out.piece_stats = _region_piece_stats(result, roi)
        if keep_intermediates:
            out.intermediates = dict(kept_side, dark_raw=dark_raw,
                                     markers=markers)
        out.status = "ok"
    except _Refused as refusal:
        out.status = "refused"
        out.refusal_reason = refusal.reason
        sink.emit(refusal.message)
    out.messages = list(sink.messages)
    return out

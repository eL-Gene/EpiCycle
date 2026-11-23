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

**Three ways to grow** (``params["growth"]``): ``"flood"`` -- the stages
above, the default, unchanged -- ``"compete"``, the competition with a
growth limit (``compete.py``), and ``"fill_with_background"``, one seeded
watershed with a background marker.

Under ``"compete"``: stage 4 also drops run markers that are mostly over
kept objects (``drop_sliver_markers``, on the unclipped cores); stage 5
seeds every kept object as a competitor (``kept_competitors``) instead of
its core; stage 6 is skipped (no outside marker -- the competition only
claims voxels of its *space*: the empty voxels, cut to the tissue outline
when one is chosen, else minus the eroded dark); stage 7 runs
``compete_tiled`` (tiles in all three axes on ONE elevation and ONE marker
volume, so tiles equal one call), zeroes the competitors' claims and, with
``include_membrane``, hands the membrane at a new cell's edge to it.

Under ``"fill_with_background"`` (what fills a whole patch on Cells):
stages 1-4 and the space as under the competition (the same sliver drop,
the same space from one shared helper), the same kept competitors, plus
ONE background marker: every space voxel holding no source that lies
farther from its nearest marker than that marker's own distance ``r`` --
``max(BACKGROUND_FROM_MARKER_UM, BACKGROUND_NN_SHARE x the distance from
its centroid to the nearest other marker's centroid)``, in um; a lone
marker takes the floor.  That distance is where the background STARTS,
not how far a cell grows: on flat terrain the cut lands about half-way.
The seed is geometric (the space, the markers, the spacing; never an
image channel) and is measured in z-slabs with a halo wider than every
``r``, which gives exactly the whole-volume answer and a Stop between
slabs.  The slabs hold the distance map's memory (about 49 bytes per
voxel of a slab and its halo) to the larger of the slab budget and about
four halos of planes; one marker far from the rest makes the halo, and so
a slab, as large as the whole patch.  Then one watershed over the space
(6-connected, no line, compactness 0) under the flood's stage name
``add_cut``; the competitors' and the background's claims are zeroed,
the membrane step runs when ticked, the voxels the kept objects were
seeded on are emptied again (a kept object keeps its side of a gap), and
one piece per marker is kept.  Measured, the whole path peaks at about
59 bytes per voxel of the patch with the membrane step off and about 70
with it on, besides the caller's own inputs.  The watershed itself cannot
be stopped; a Stop pressed during it lands when it returns, before
anything is handed back.

``whole_patch=True`` lifts the box budget for the competition (grown in
tiles) and for the fill with a background, which has its own ceiling,
:data:`WHOLE_FILL_MAX_VOXELS` (reason ``"too-big-whole"``).  The flood
never takes a whole patch.

**The tissue outline** (``params["outline"]`` is the MODE, both growth
rules): ``"shrink"`` (``outline.tissue_outline``) or ``"cleared"``.  The
outline itself is the run argument ``outline=`` -- the finished outline
the caller built ONCE over the whole patch (or its box crop); the engine
crops it and never builds one for a box alone: ``"shrink"`` with no
``outline=`` is refused (``"outline-per-box"``) unless the run IS the
whole patch (``whole_patch=True``), where it is built here from the dark
channels.  Choosing one turns the dark plumbing on whatever
``params["outside"]`` says.  Hook A: an automatic marker with >= 70 % of
its voxels outside is dropped (``n_outside_outline``, apart from the dark
drop, which runs first); when that leaves no marker the refusal is
``"no-markers-outline"``.  Hook B (the flood only): the empty space
outside the outline joins the outside marker.  Hook D (both growth rules;
it only ever shrinks the answer): new objects only inside it.  The
unfilled share is then measured against the outline.  It never writes and
never touches a kept object.

Unlike the label path, a cancel here RAISES :class:`~.engine.ReshapeCancelled`;
the Add worker turns it into its own ``cancelled`` answer.

Pure: numpy / scipy / skimage, no Qt, no napari.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# the module, not its names: `include_membrane` is also a spec knob
from . import compete as _compete
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
    "ADD_ENGINE_NO_MARKERS_OUTLINE",
    "ADD_ENGINE_OUTLINE_PER_BOX",
    "ADD_ENGINE_TOO_BIG",
    "ADD_ENGINE_TOO_BIG_WHOLE",
    "BACKGROUND_FROM_MARKER_UM",
    "BACKGROUND_NN_SHARE",
    "BACKGROUND_SLAB_VOXELS",
    "ENGINE_REFUSALS",
    "DARK_BLUR_UM",
    "DARK_SAMPLE_VOXELS",
    "NUCLEI_SLIVER_UM",
    "REGION_SEEDS",
    "REGION_STAGES",
    "REGION_STAGES_COMPETE",
    "compete_is_warm",
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
    "too_big_whole_sentence",
    "unfilled_fraction",
    "warm_up_compete",
    "WHOLE_FILL_MAX_VOXELS",
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

#: Under ``growth="fill_with_background"``: the least distance (um) from a
#: marker at which the background seed may start.  It is where the seed
#: STARTS, not how far a cell grows (on flat terrain the cut lands about
#: half-way between the marker and the seed).
BACKGROUND_FROM_MARKER_UM = 6.0

#: ... and the share of the distance from a marker's centroid to the
#: nearest other marker's centroid at which its background seed starts,
#: when that is farther than :data:`BACKGROUND_FROM_MARKER_UM` -- so a
#: sparse layer of large cells is not cut short by a fixed distance.
BACKGROUND_NN_SHARE = 0.75

#: The largest whole patch (voxels) the fill with a background takes: the
#: fill peaks at about 59 bytes per voxel of the patch with the membrane
#: step off and about 70 with it on (besides the caller's own inputs: about
#: 6 GB and 7 GB at this ceiling), and its watershed cannot be stopped once
#: it runs (the watershed step alone took about two minutes at 73 million
#: voxels; the whole Preview takes longer).
WHOLE_FILL_MAX_VOXELS = 100_000_000

#: Voxels per z-slab, halo included, when the background seed is measured;
#: the distance map holds about 49 bytes per voxel of a slab.  A slab is
#: never thinner than two halos, so the distance maps together cover at
#: most about three times the patch; with one marker far from the rest the
#: halo -- and a slab -- can reach the whole patch.
BACKGROUND_SLAB_VOXELS = 16_000_000

#: The growth rules that drop run markers mostly over kept objects.
_SLIVER_DROP_GROWTH = ("compete", "fill_with_background")

#: The largest id the solvers take.
_INT32_MAX = int(np.iinfo(np.int32).max)

#: The marker sources the region target reads -- ONE tuple, defined in
#: ``spec.py`` (which refuses any other source on ``target="region"``) and
#: re-exported here.
# REGION_SEEDS is imported from .spec above.

#: The ``progress`` stage names of :func:`reshape_empty_region`, in order.
#: Each is emitted as its stage STARTS, so a present-tense strip ("Filling
#: the empty space...") names the work under way.  The Add worker maps them
#: to ``STAGE_TEXT``.
REGION_STAGES = ("add_box", "elevation", "add_markers", "add_cut")

#: The same four stages under ``growth="compete"``: the last one grows.
REGION_STAGES_COMPETE = ("add_box", "elevation", "add_markers", "add_grow")

#: The competition reports each finished tile as ``(PIECE_STAGE, i, k)``.
PIECE_STAGE = "add_piece"

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


_TOO_BIG_WHOLE_TEMPLATE = (
    "This patch is too large to fill as a whole — the limit is {} "
    "voxels; fill it box by box instead. (Nothing was changed.)"
)


def too_big_whole_sentence(limit):
    """The refusal of a whole-patch fill over ``limit`` voxels (reason
    ``"too-big-whole"``)."""
    return _TOO_BIG_WHOLE_TEMPLATE.format(_voxel_count_words(limit))


#: The refusal at the default budget (``spec.ADD_BOX_BUDGET``); B2 imports
#: it by name.  A refusal under any other budget is ``too_big_sentence``.
ADD_ENGINE_TOO_BIG = too_big_sentence(4_000_000)
ADD_ENGINE_NO_MARKERS = (
    "No marker was left to grow from — every marker fell in dark space, "
    "over kept objects, or was too small to use. (Nothing was changed.)"
)
#: Reason ``"no-markers-outline"``: the tissue outline dropped the last
#: markers (at least one marker was left until the outline drop).
ADD_ENGINE_NO_MARKERS_OUTLINE = (
    "No marker was left to grow from — the markers that were left all lay "
    "outside the tissue outline. (Nothing was changed.)"
)
#: Reason ``"outline-per-box"``: a shrink outline was asked for one box
#: with no outline handed in.  The outline is built once over the whole
#: patch and cropped; one built over a box alone differs at the box faces.
ADD_ENGINE_OUTLINE_PER_BOX = (
    "The tissue outline is built over the whole patch, never over one box "
    "alone, and none was handed in. (Nothing was changed.)"
)

#: Reason ``"too-big-whole"`` at the shipped ceiling; a refusal under any
#: other ceiling is :func:`too_big_whole_sentence`.
ADD_ENGINE_TOO_BIG_WHOLE = too_big_whole_sentence(WHOLE_FILL_MAX_VOXELS)

#: Every engine refusal's sentence by its reason code, except ``"too-big"``
#: (its sentence names the budget: :func:`too_big_sentence`).
ENGINE_REFUSALS = {
    "no-markers": ADD_ENGINE_NO_MARKERS,
    "no-markers-outline": ADD_ENGINE_NO_MARKERS_OUTLINE,
    "outline-per-box": ADD_ENGINE_OUTLINE_PER_BOX,
    "too-big-whole": ADD_ENGINE_TOO_BIG_WHOLE,
}

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
    """``(stage, i, n)`` over :data:`REGION_STAGES` (or ``names``); a raising
    callback is swallowed, as the engine's own ``_Progress`` does -- except
    that with ``propagate_cancel`` a :class:`~.engine.ReshapeCancelled` it
    raises stops the run (the fill with a background: its watershed cannot
    be stopped once it starts, so a Stop at its stage must land before)."""

    __slots__ = ("_cb", "_i", "_n", "_propagate")

    def __init__(self, callback, names=REGION_STAGES, propagate_cancel=False):
        self._cb = callback
        self._i = 0
        self._n = len(names)
        self._propagate = bool(propagate_cancel)

    def __call__(self, stage):
        self._i += 1
        if self._cb is None:
            return
        try:
            self._cb(stage, self._i, self._n)
        except ReshapeCancelled:
            if self._propagate:
                raise
        except Exception:
            pass


class _PieceProgress:
    """``compete_tiled``'s ``progress(i, k)`` -> ``callback(PIECE_STAGE, i,
    k)``; remembers ``k``.  A Stop raised by the callback propagates; any
    other error in it is swallowed (a broken bar never stops the work)."""

    __slots__ = ("_cb", "k")

    def __init__(self, callback):
        self._cb = callback
        self.k = 0

    def __call__(self, i, k):
        self.k = int(k)
        if self._cb is None:
            return
        try:
            self._cb(PIECE_STAGE, int(i), int(k))
        except ReshapeCancelled:
            raise
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


def _unfilled_in_outline(result, existing_roi, outline):
    """|empty, inside the outline, not filled| / |empty, inside the outline|
    -- ``None`` when the outline holds no empty voxel."""
    tissue = (np.asarray(existing_roi) == 0) & np.asarray(outline, bool)
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
    if spec.params.get("whole_patch"):
        # the competition grows a whole patch in tiles, so its size is never
        # refused here; the fill with a background is ONE watershed over
        # the whole patch, held whole in memory and not stoppable while it
        # runs, so it has a ceiling (checked before anything is read)
        if spec.params.get("growth") == "fill_with_background":
            limit = int(WHOLE_FILL_MAX_VOXELS)
            if n > limit:
                raise _Refused("too-big-whole", too_big_whole_sentence(limit))
        return n
    budget = int(spec.params["box_budget"])
    if n > budget:
        raise _Refused("too-big", too_big_sentence(budget))
    return n


def _effective_outside(params):
    """``params["outside"]``, except that a tissue outline turns the dark
    plumbing on whatever the dark tick says (the outline is decoupled from
    the tick; the dark drop and, for the flood, the dark outside marker
    then always run)."""
    if params.get("outline", "none") != "none":
        return "faces+dark"
    return params["outside"]


def _dark_stage(spec, *, roi, shape, dark_channels, dark_threshold, spacing,
                keep_crops=False):
    """``(dark_raw, crops)``: ``dark_raw`` (bool, box-sized) or ``None``
    with ``outside="faces"`` and no tissue outline; ``crops`` -- the dark
    channels cut to the box, returned only with ``keep_crops`` (the shrink
    outline built here reads the same crops: each channel is cut ONCE),
    else ``None``."""
    if _effective_outside(spec.params) != "faces+dark":
        return None, None
    why = ("outside='faces+dark'" if spec.params["outside"] == "faces+dark"
           else "a tissue outline")
    if dark_threshold is None:
        raise ValueError(
            "{} needs dark_threshold= (the patch threshold "
            "from patch_dark_threshold).".format(why))
    if not dark_channels:
        raise ValueError(
            "{} needs dark_channels= (every image channel "
            "of the patch, cropped to the box).".format(why))
    crops = {name: _region_crop(arr, roi, shape,
                                what="dark channel {!r}".format(name))
             for name, arr in dict(dark_channels).items()}
    dark_raw = dark_mask(crops, dark_threshold=dark_threshold,
                         factor=spec.params["dark_otsu_factor"], spacing=spacing)
    return dark_raw, (crops if keep_crops else None)


def _drop_dark(markers, dark_raw):
    """Drop automatic markers with >= 50 % of their voxels in the dark.

    Counts with ``np.bincount`` over COMPACTED ids (the ids' own values are
    never an index), so a whole patch or an id of 2**31 - 1 costs only the
    marker voxels -- a float64 ``ndi.sum`` over the volume cost 56 s and
    4.2 GiB at 73 M voxels.  The decisions are the ones that sum made."""
    if dark_raw is None:
        return markers, 0
    nz = markers != 0
    if not nz.any():
        return markers, 0
    ids, inv = np.unique(markers[nz], return_inverse=True)
    inv = inv.reshape(-1)
    total = np.bincount(inv, minlength=ids.size).astype(np.float64)
    in_dark = np.bincount(inv[np.asarray(dark_raw, bool)[nz]],
                          minlength=ids.size).astype(np.float64)
    share = in_dark / np.maximum(total, 1.0)
    drop = ids[share >= DARK_DROP_SHARE]
    if drop.size:
        markers = np.where(np.isin(markers, drop), 0, markers)
    return markers, int(drop.size)


def _drop_automatic(markers, *, dark_raw, outline):
    """The automatic markers' two drops, in order and counted apart:
    >= 50 % of a marker in the dark (``n_dropped_dark``), THEN >= 70 % of a
    survivor outside the tissue outline (``n_outside_outline``).  A marker
    both dark and outside is dropped once and counted as dark; a marker
    30 % dark and 30 % outside is kept (each rule on its own).  Returns
    ``(markers, n_dark, n_outside)``."""
    markers, n_dark = _drop_dark(markers, dark_raw)
    if outline is None:
        return markers, n_dark, 0
    from .outline import markers_outside

    drop = markers_outside(markers, outline)
    if drop.size:
        markers = np.where(np.isin(markers, drop), 0, markers)
    return markers, n_dark, int(drop.size)


def _check_outline_arg(spec, outline):
    """The run argument ``outline=`` against the spec's outline MODE, in
    plain words (a ``ValueError``: a caller's mistake, raised before any
    work).  A volume with no outline chosen, ``"cleared"`` without its
    volume, a volume that is not 3D, or one that is not bool / integer
    0-1 (a float is refused -- a probability map is no outline)."""
    mode = spec.params.get("outline", "none")
    if outline is None:
        if mode == "cleared":
            raise ValueError(
                "params['outline']='cleared' needs outline= (the outline "
                "of the objects cleared on this patch, built by the "
                "caller).")
        return
    if mode == "none":
        raise ValueError(
            "outline= was given but no outline is chosen "
            "(params['outline']='none'); pick 'shrink' or 'cleared', or "
            "leave the outline out.")
    shape = getattr(outline, "shape", None)
    if shape is None or len(shape) != 3:
        raise ValueError(
            "outline= must be a 3D volume (the whole patch or the box "
            "crop), got shape {!r}.".format(shape))
    dtype = np.dtype(getattr(outline, "dtype", np.float64))
    if dtype.kind not in "biu":
        raise ValueError(
            "outline= must be bool, or integers 0 and 1; got {} (a float "
            "map is not an outline -- threshold it first).".format(dtype))


def _outline_refusal(spec, outline):
    """A box always receives the patch-wide outline cropped.  A shrink
    outline with nothing handed in is built here only when the run IS the
    whole patch; for a box it is refused -- one built over the box alone
    differs from the patch's at every box face."""
    p = spec.params
    if (p.get("outline", "none") == "shrink" and outline is None
            and not p.get("whole_patch")):
        raise _Refused("outline-per-box", ADD_ENGINE_OUTLINE_PER_BOX)


def _outline_stage(spec, *, outline, roi, shape, crops, dark_threshold,
                   spacing, cancel_event):
    """The tissue outline cut to the box (bool), or ``None`` for
    ``outline="none"``.  ``outline`` (the run argument: the whole patch or
    the box crop) is cropped and used as handed in; ``"shrink"`` without
    one (the whole patch only, see :func:`_outline_refusal`) is built here
    from ``crops`` -- the dark channels already cut for the dark stage --
    at the patch threshold, Dark below and Bridge gaps."""
    p = spec.params
    mode = p.get("outline", "none")
    if mode == "none":
        return None
    if outline is not None:
        box = _region_crop(outline, roi, shape, what="the tissue outline")
        if box.dtype != bool:
            if box.size and (box.min() < 0 or box.max() > 1):
                raise ValueError(
                    "outline= must hold only 0 and 1; it holds values from "
                    "{} to {} in the box.".format(box.min(), box.max()))
            box = box.astype(bool)
        return box
    from .outline import tissue_outline

    return tissue_outline(crops, spacing, dark_factor=p["dark_otsu_factor"],
                          bridge_um=p["bridge_um"],
                          dark_threshold=dark_threshold, cancel=cancel_event)


def _no_markers(counts):
    """The refusal when no marker is left: the outline's own reason when it
    dropped the last ones, else the general one."""
    if counts.get("n_outside_outline", 0) > 0:
        return _Refused("no-markers-outline", ADD_ENGINE_NO_MARKERS_OUTLINE)
    return _Refused("no-markers", ADD_ENGINE_NO_MARKERS)


def _dark_and_outline(spec, *, outline, roi, shape, dark_channels,
                      dark_threshold, spacing, cancel_event):
    """Stages 3 and 3b: ``(dark_raw, outline_box)``; the dark channels are
    cut to the box once and shared."""
    build_here = (spec.params.get("outline", "none") == "shrink"
                  and outline is None)
    dark_raw, crops = _dark_stage(spec, roi=roi, shape=shape,
                                  dark_channels=dark_channels,
                                  dark_threshold=dark_threshold,
                                  spacing=spacing, keep_crops=build_here)
    _check_cancel(cancel_event)
    # None for "none": no work, no poll
    box = _outline_stage(spec, outline=outline, roi=roi, shape=shape,
                         crops=crops, dark_threshold=dark_threshold,
                         spacing=spacing, cancel_event=cancel_event)
    return dark_raw, box


def _dark_outside(spec, dark_raw, *, seed_points, shape, spacing, cancel_event,
                  faces_are_dark=False):
    """The dark part of the outside: ``dark_raw`` with the click clearance
    taken out, eroded by the clearance reach.  Returns ``(dark_out,
    dark_raw)`` -- ``dark_raw`` after the clearance, which the unfilled
    share reads.  ``faces_are_dark``: the box faces are no edge of the dark
    (the erosion pads with dark), so the dark reaches the faces -- the
    competition needs that, having no face marker; the flood keeps its
    faces in the outside marker anyway."""
    p = spec.params
    if spec.seeds == "paint" and seed_points:
        clear = rasterise_points(seed_points, shape,
                                 radius_um=p["dark_clearance_um"],
                                 spacing=spacing) > 0
        dark_raw = dark_raw & ~clear
    reach = float(p["dark_clearance_um"])
    _check_cancel(cancel_event)
    if reach > 0:
        from .arbiter import physical_ball
        if faces_are_dark:
            dark_out = ndi.binary_erosion(
                dark_raw, structure=physical_ball(reach, spacing),
                border_value=1)
        else:
            dark_out = ndi.binary_erosion(dark_raw,
                                          structure=physical_ball(reach, spacing))
    else:
        dark_out = dark_raw
    return dark_out, dark_raw


def _marker_stage(spec, *, roi, shape, existing, seed_layer, seed_points,
                  dark_raw, spacing, cancel_event, outline=None):
    """Stage 4: the new-object markers ``1..N`` and their counts.

    The order: the cores; under the competition and the fill with a
    background the sliver drop on the UNCLIPPED cores (run objects only);
    the clip to empty space; the dark drop; the outline drop (hook A); the
    ascending relabel."""
    p = spec.params
    empty = existing == 0
    counts = {"n_dropped_thin": 0, "n_markers_over_kept": 0,
              "n_nuclei_slivers": 0, "n_dropped_dark": 0,
              "n_outside_outline": 0, "n_slivers": 0}
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
        clipped = np.where(empty, cores, 0)
        counts["n_markers_over_kept"] = int(before.size - _ids_of(clipped).size)
        if p.get("growth") in _SLIVER_DROP_GROWTH:
            # a core mostly over kept objects is a sliver of a kept cell's
            # own nucleus; judged on the UNCLIPPED core, then clipped.  A
            # core wholly over kept objects stays counted "over kept".
            kept, n_sliv = _compete.drop_sliver_markers(cores, empty)
            counts["n_slivers"] = int(n_sliv) - counts["n_markers_over_kept"]
            clipped = np.where(empty, kept, 0)
            del kept
        cores = clipped
        del clipped
        source = "another run's objects"
    else:  # nuclei_masks
        before = _ids_of(np.where(empty, crop, 0))
        counts["n_markers_over_kept"] = int(n_found - before.size)
        cores, n_sliv = drop_slivers(crop, existing, min_depth_um=NUCLEI_SLIVER_UM,
                                     spacing=spacing, cancel_event=cancel_event)
        counts["n_nuclei_slivers"] = n_sliv
        source = "bright objects (nuclei)"
    _check_cancel(cancel_event)
    cores, counts["n_dropped_dark"], counts["n_outside_outline"] = _drop_automatic(
        cores, dark_raw=dark_raw, outline=outline)
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
# stages 5-8: the three ways to grow
# --------------------------------------------------------------------------- #
def _grow_by_flood(spec, *, elev, existing, new, n, outline, dark_raw,
                   seed_points, shape, spacing, step, log, sink, cancel_event):
    """Stages 5-8 of the flood (the default), with hooks B and D when an
    outline is chosen.  Returns ``(result, markers, dark_raw, n_frag)``."""
    p = spec.params
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
    # the box faces are outside (a box is a cut); a whole patch filled the
    # way a box is keeps its faces open (``faces_outside=False``: they are
    # the edge of the data), and its dark is not eroded back from them
    faces = bool(p.get("faces_outside", True))
    outside = np.zeros(shape, bool)
    if faces:
        outside[0, :, :] = outside[-1, :, :] = True
        outside[:, 0, :] = outside[:, -1, :] = True
        outside[:, :, 0] = outside[:, :, -1] = True
    if dark_raw is not None:
        dark_out, dark_raw = _dark_outside(spec, dark_raw,
                                           seed_points=seed_points, shape=shape,
                                           spacing=spacing,
                                           cancel_event=cancel_event,
                                           faces_are_dark=not faces)
        outside |= dark_out
    if outline is not None:
        # hook B: outside the outline is outside -- a hard edge, not eroded
        outside |= ~outline
    outside &= (existing == 0) & (new == 0)
    markers[outside & (markers == 0)] = n + 1
    _check_cancel(cancel_event)

    # ---- 7. the cut --------------------------------------------------- #
    step("add_cut")
    _check_cancel(cancel_event)
    partition = solve(elev, markers, np.ones(shape, bool), solver="watershed",
                      compactness=p["compactness"], spacing=spacing,
                      sink=sink)
    _check_cancel(cancel_event)

    # ---- 8. new objects only, one piece each -------------------------- #
    keep = (partition >= 1) & (partition <= n) & (existing == 0)
    if outline is not None:
        keep &= outline                  # hook D: new objects inside it only
    result = np.where(keep, partition, 0).astype(np.int32)
    del partition, keep
    result, n_frag = keep_marker_components(result, new,
                                            cancel_event=cancel_event)
    _check_cancel(cancel_event)
    return result, markers, dark_raw, n_frag


def _growth_space(spec, *, existing, outline, dark_raw, seed_points, shape,
                  spacing, cancel_event):
    """Where the competition and the fill with a background may claim
    voxels: the empty voxels inside the tissue outline; with no outline and
    the dark rule on, the empty voxels minus the eroded dark (the box faces
    count as dark); else every empty voxel.

    Returns ``(space, dark_raw)`` -- ``dark_raw`` after the clearance
    around clicks when the dark rule shaped the space (the unfilled share
    reads it), else as handed in."""
    empty = existing == 0
    if outline is not None:
        space = empty & outline          # hook D
    elif dark_raw is not None:
        dark_out, dark_raw = _dark_outside(spec, dark_raw,
                                           seed_points=seed_points, shape=shape,
                                           spacing=spacing,
                                           cancel_event=cancel_event,
                                           faces_are_dark=True)
        space = empty & ~dark_out
        del dark_out
    else:
        space = empty
    return space, dark_raw


def _grow_by_competition(spec, *, elev, existing, new, n, outline, dark_raw,
                         seed_points, shape, spacing, tile_budget, step,
                         progress, cancel_event):
    """Stages 5-8 of the competition.  No outside marker: the competition
    claims only its space -- the empty voxels inside the outline, or (no
    outline) the empty voxels minus the eroded dark, or every empty voxel.
    Returns ``(result, sources, dark_raw, n_frag, n_tiles)``."""
    p = spec.params
    # ---- 6'. the space (no outside marker) ----------------------------- #
    space, dark_raw = _growth_space(spec, existing=existing, outline=outline,
                                    dark_raw=dark_raw, seed_points=seed_points,
                                    shape=shape, spacing=spacing,
                                    cancel_event=cancel_event)
    _check_cancel(cancel_event)

    # ---- 5'. kept objects compete from the empty voxels next to them --- #
    sources, n_new = _compete.kept_competitors(new, existing, space)
    if n_new != n:          # pragma: no cover - `new` is relabelled 1..n
        raise AssertionError("the markers are not numbered 1..{}".format(n))
    _check_cancel(cancel_event)

    # ---- 7'. the growth, tile by tile --------------------------------- #
    step("add_grow")
    pieces = _PieceProgress(progress)
    claims = _compete.compete_tiled(
        elev, sources, space, spacing, wall_cost=float(p["wall_cost"]),
        growth_limit=float(p["growth_limit"]),
        budget=(_compete.COMPETE_TILE_VOXELS if tile_budget is None
                else int(tile_budget)),
        cancel=cancel_event, progress=pieces)
    _mark_warm("grow")
    claims[claims > n] = 0               # a kept object never gains a voxel
    _check_cancel(cancel_event)
    if p["include_membrane"]:
        claims = _compete.include_membrane(
            claims, elevation=elev, existing=existing, space=space,
            spacing_um=spacing)
        _mark_warm("membrane")
        _check_cancel(cancel_event)

    # ---- 8'. new objects only, one piece each ------------------------- #
    result = np.where((claims >= 1) & (claims <= n) & space, claims,
                      0).astype(np.int32)
    del claims
    result, n_frag = keep_marker_components(result, new,
                                            cancel_event=cancel_event)
    _check_cancel(cancel_event)
    return result, sources, dark_raw, n_frag, pieces.k


# --------------------------------------------------------------------------- #
# the background marker of the fill with a background
# --------------------------------------------------------------------------- #
def _checked_spacing(spacing):
    """The voxel spacing as three finite positive floats (um), else a
    ``ValueError`` in plain words: every distance of the background seed is
    measured in microns."""
    try:
        sp = tuple(float(s) for s in spacing)
    except (TypeError, ValueError):
        sp = ()
    if len(sp) != 3 or not all(math.isfinite(s) and s > 0.0 for s in sp):
        raise ValueError(
            "the background seed needs the voxel spacing as three positive "
            "numbers in um (z, y, x); got {!r}.".format(spacing))
    return sp


def _background_radii(new, spacing):
    """Per marker id, the distance (um) past which its background seed
    starts: ``max(BACKGROUND_FROM_MARKER_UM, BACKGROUND_NN_SHARE x NN)``,
    NN = the distance from the marker's centroid (its mean voxel index x
    the spacing) to the nearest other marker's centroid.  A lone marker
    takes the floor.  Returns a float64 array indexed by id (index 0, and
    any id with no voxel, hold 0.0).  Reads only the marker voxels."""
    sp = _checked_spacing(spacing)
    mk = np.asarray(new)
    n = int(mk.max()) if mk.size else 0
    radii = np.zeros(n + 1, np.float64)
    if n <= 0:
        return radii
    flat = mk.reshape(-1)
    where = np.flatnonzero(flat)
    ids = flat[where].astype(np.int64)
    count = np.bincount(ids, minlength=n + 1).astype(np.float64)
    present = np.flatnonzero(count[1:] > 0) + 1
    coords = np.unravel_index(where, mk.shape)
    del where
    centroids = np.empty((present.size, 3), np.float64)
    for axis in range(3):
        total = np.bincount(ids, weights=coords[axis].astype(np.float64),
                            minlength=n + 1)
        centroids[:, axis] = total[present] / count[present] * sp[axis]
    floor = float(BACKGROUND_FROM_MARKER_UM)
    if present.size < 2:
        radii[present] = floor
        return radii
    dist, _idx = cKDTree(centroids).query(centroids, k=2)
    radii[present] = np.maximum(floor, float(BACKGROUND_NN_SHARE) * dist[:, 1])
    return radii


def _background_id(n, n_kept):
    """The background's id: after the ``n`` new markers and the ``n_kept``
    kept competitors.  Refused in plain words past the solver's int32."""
    bg = int(n) + int(n_kept) + 1
    if bg > _INT32_MAX:
        raise ValueError(
            "too many ids for the fill: {} new markers, {} kept objects and "
            "the background need more ids than the solver takes "
            "({}).".format(int(n), int(n_kept), _INT32_MAX))
    return bg


def _background_seeds(new, space, sources, spacing, *, cancel_event=None,
                      slab_voxels=None):
    """The background seed: every ``space`` voxel holding no source
    (``sources == 0``) whose distance to its nearest marker voxel of
    ``new`` exceeds that marker's distance (:func:`_background_radii`).

    Geometric: it reads only the space, the markers, the sources and the
    spacing -- never an image channel.  Measured in z-slabs with a halo of
    ``ceil(max r / dz) + 1`` slices: a voxel whose nearest marker lies past
    the halo is farther than every ``r``, so the slabs give the
    whole-volume answer.  A slab is about ``slab_voxels`` voxels, halo
    included (``None`` = :data:`BACKGROUND_SLAB_VOXELS`), but never thinner
    than two halos, so the distance maps together cover at most about three
    times the volume.  Memory: the distance map holds about 49 bytes per
    voxel of a slab and its halo -- bounded by the larger of the slab budget
    and about four halos of planes, which reaches the whole volume when one
    marker is far from the rest (its ``r``, and the halo, are then large).
    ``cancel_event`` is polled before every slab (raises
    :class:`~.engine.ReshapeCancelled`).  Returns a bool volume; all False
    when ``new`` holds no marker."""
    sp = _checked_spacing(spacing)
    mk = np.asarray(new)
    shape = mk.shape
    if len(shape) != 3:
        raise ValueError("the markers must be a 3D volume, got shape "
                         "{}.".format(shape))
    space = np.asarray(space)
    sources = np.asarray(sources)
    if space.shape != shape or sources.shape != shape:
        raise ValueError(
            "the space {} and the sources {} must have the markers' shape "
            "{}.".format(space.shape, sources.shape, shape))
    space = space.astype(bool, copy=False)
    out = np.zeros(shape, bool)
    radii = _background_radii(mk, sp)
    if radii.size < 2:
        return out
    r_max = float(radii.max())
    nz = int(shape[0])
    plane = int(shape[1]) * int(shape[2])
    halo = int(math.ceil(r_max / sp[0])) + 1
    budget = int(BACKGROUND_SLAB_VOXELS if slab_voxels is None else slab_voxels)
    # never thinner than two halos: a slab of one slice under a wide halo
    # would measure a distance map over 2 x halo + 1 slices per slice
    thick = max(budget // max(plane, 1) - 2 * halo, 2 * halo, 1)
    for z0 in range(0, nz, thick):
        _check_cancel(cancel_event)
        z1 = min(nz, z0 + thick)
        cand = space[z0:z1] & (sources[z0:z1] == 0)
        if not cand.any():
            continue
        a0, a1 = max(0, z0 - halo), min(nz, z1 + halo)
        block = mk[a0:a1]
        free = block == 0
        if free.all():
            # no marker within the halo: farther than every r
            out[z0:z1] = cand
            continue
        dist, ind = ndi.distance_transform_edt(free, sampling=sp,
                                               return_indices=True)
        del free
        lo, hi = z0 - a0, z1 - a0
        nearest = block[ind[0, lo:hi], ind[1, lo:hi], ind[2, lo:hi]]
        far = dist[lo:hi] > radii[nearest]
        del dist, ind, nearest
        out[z0:z1] = cand & far
    return out


def _grow_by_fill_with_background(spec, *, elev, existing, new, n, outline,
                                  dark_raw, seed_points, shape, spacing, step,
                                  sink, cancel_event):
    """Stages 5-8 of the fill with a background.  The space and the kept
    competitors as under the competition; one background marker
    (:func:`_background_seeds`, id ``n + K + 1``); one watershed over the
    space, 6-connected, no watershed line, compactness 0; every claim above
    ``n`` zeroed; the membrane step when ticked; the voxels the kept objects
    were seeded on emptied again (the membrane step must not hand a kept
    object's side of a gap to a new cell: on a coarse z a new voxel beside
    it can be nearer than the kept object below it); one piece per marker.
    Returns ``(result, sources, dark_raw, n_frag)``."""
    p = spec.params
    _check_cancel(cancel_event)
    # ---- 6'. the space (shared with the competition) ------------------- #
    space, dark_raw = _growth_space(spec, existing=existing, outline=outline,
                                    dark_raw=dark_raw, seed_points=seed_points,
                                    shape=shape, spacing=spacing,
                                    cancel_event=cancel_event)
    _check_cancel(cancel_event)

    # ---- 5'. kept objects compete from the empty voxels next to them --- #
    sources, n_new = _compete.kept_competitors(new, existing, space)
    if n_new != n:          # pragma: no cover - `new` is relabelled 1..n
        raise AssertionError("the markers are not numbered 1..{}".format(n))
    # K counts every kept object, not only those seeded on the space: the
    # largest competitor id in `sources` falls short of n + K when the
    # last-ranked kept object has no empty voxel of the space beside it
    kept = existing != 0
    n_kept = int(np.unique(existing[kept]).size) if kept.any() else 0
    del kept
    background = _background_id(n, n_kept)
    _check_cancel(cancel_event)

    # ---- 5''. the background, far from every marker -------------------- #
    seed = _background_seeds(new, space, sources, spacing,
                             cancel_event=cancel_event)
    sources[seed] = background
    del seed
    _check_cancel(cancel_event)

    # ---- 7''. one watershed over the space (cannot stop once it runs) -- #
    step("add_cut")
    _check_cancel(cancel_event)
    claims = solve(elev, sources, space, solver="watershed", compactness=0.0,
                   spacing=spacing, sink=sink)
    _check_cancel(cancel_event)
    claims = np.asarray(claims, np.int32)
    claims[claims > n] = 0       # no kept object or background voxel kept
    _check_cancel(cancel_event)
    if p["include_membrane"]:
        claims = _compete.include_membrane(
            claims, elevation=elev, existing=existing, space=space,
            spacing_um=spacing)
        _mark_warm("membrane")
        _check_cancel(cancel_event)
    # a kept object's seam stays empty, whatever the membrane step did
    claims[(sources > n) & (sources < background)] = 0

    # ---- 8''. new objects only, one piece each ------------------------- #
    result = np.where((claims >= 1) & (claims <= n) & space, claims,
                      0).astype(np.int32)
    del claims
    result, n_frag = keep_marker_components(result, new,
                                            cancel_event=cancel_event)
    _check_cancel(cancel_event)
    return result, sources, dark_raw, n_frag


# --------------------------------------------------------------------------- #
# the solver's one-time set-up
# --------------------------------------------------------------------------- #
_WARM = [False]

#: The solver loops a real competition has run in this process; both ->
#: warm (the membrane loop runs only with ``include_membrane``).
_WARM_PARTS = set()

#: The one-time sentence of :func:`warm_up_compete` is announced only when
#: the set-up is still running after this many seconds (a compile; loading
#: the on-disk cache takes 1-2.5 s and is not announced).
WARM_ANNOUNCE_AFTER_S = 3.0


def _mark_warm(part):
    _WARM_PARTS.add(part)
    if {"grow", "membrane"} <= _WARM_PARTS:
        _WARM[0] = True


def compete_is_warm():
    """True once the solver's loops are set up in this process: by
    :func:`warm_up_compete`, or by a real competition that ran both loops
    (the growth and the membrane step)."""
    return bool(_WARM[0])


def _warm_body():
    shape = (4, 8, 8)
    unit = (1.0, 1.0, 1.0)
    everywhere = np.ones(shape, bool)
    markers = np.zeros(shape, np.int32)
    markers[2, 4, 4] = 1
    claims, _cost = _compete.compete(np.zeros(shape, np.float32), markers,
                                     everywhere, unit, wall_cost=0.0,
                                     growth_limit=1.0)
    # bright everywhere and unclaimed beyond 1 um: the membrane loop runs
    _compete.include_membrane(claims, elevation=np.full(shape, 0.5, np.float32),
                              existing=np.zeros(shape, np.int32),
                              space=everywhere, spacing_um=unit)


def warm_up_compete(*, announce=None):
    """Set the competition's solver up on a tiny volume, before the first
    real use, on the worker thread.

    The solver's two inner loops are compiled by numba on their first call:
    about 10 s the first time on a machine (then cached on disk beside
    ``compete.py``; a later process loads the cache in 1-2.5 s).  The set-up
    runs on a helper thread while this one waits; ``announce()`` is called
    -- on THIS thread, at most once -- only if the set-up is still running
    after :data:`WARM_ANNOUNCE_AFTER_S` (a real compile, measured; a cache
    load stays silent).  An error in ``announce`` is swallowed.  Returns
    the seconds it took (0.0 when already warm).  Raises ``ImportError``
    (``compete.NUMBA_MISSING``) or ``RuntimeError``
    (``compete.NUMBA_CACHE_FAILED``) in plain words.

    ``compete.have_numba()`` asks without compiling, but it IMPORTS numba
    (about 1.3 s the first time in a process): call it on a worker too,
    never on the main thread.
    """
    if _WARM[0]:
        return 0.0
    t0 = time.perf_counter()
    failed = []

    def body():
        try:
            _warm_body()
        except BaseException as exc:   # handed back to the caller below
            failed.append(exc)

    helper = threading.Thread(target=body, name="compete-warm-up",
                              daemon=True)
    helper.start()
    helper.join(float(WARM_ANNOUNCE_AFTER_S))
    if helper.is_alive() and announce is not None:
        try:
            announce()
        except Exception:
            pass
    helper.join()
    if failed:
        raise failed[0]
    _WARM_PARTS.update(("grow", "membrane"))
    _WARM[0] = True
    return time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# the entry points
# --------------------------------------------------------------------------- #
def find_region_seeds(labels, images, spec, *, roi, dark_channels=None,
                      seed_layer=None, seed_points=None, dark_threshold=None,
                      spacing=(1, 1, 1), cancel_event=None, outline=None):
    """Stages 1, 3 and 4 only: how many markers the box would grow from.

    Same crop, same arguments (``outline=`` included), same cached dark
    threshold as :func:`reshape_empty_region`, so ``n_in_region`` equals
    that Preview's ``n_seeds``.  ``n_found`` = markers before drops.
    ``images`` is unused (no automatic source reads the walls) and accepted
    for symmetry.

    ``report.markers`` stays EMPTY: the Add box-info line needs the two
    counts only, and per-marker statistics cost most of a Find on a packed
    box (audit M-A1).  Array inputs follow :func:`reshape_empty_region`.
    """
    spacing = _spacing(spacing)
    sink = MessageSink(None)
    report = SeedReport(roi=roi)
    _check_spec(spec)
    _check_outline_arg(spec, outline)
    try:
        shape, existing = _prepare(labels, spec, roi, spacing, cancel_event)
        _outline_refusal(spec, outline)
        dark_raw, outline_box = _dark_and_outline(
            spec, outline=outline, roi=roi, shape=shape,
            dark_channels=dark_channels, dark_threshold=dark_threshold,
            spacing=spacing, cancel_event=cancel_event)
        markers, n, n_found, counts, source = _marker_stage(
            spec, roi=roi, shape=shape, existing=existing, seed_layer=seed_layer,
            seed_points=seed_points, dark_raw=dark_raw, spacing=spacing,
            cancel_event=cancel_event, outline=outline_box)
        report.seed_source = source
        report.seeds = markers
        report.n_found = n_found
        report.n_in_region = n
        report.n_outside_outline = int(counts.get("n_outside_outline", 0))
        if n < 1:
            raise _no_markers(counts)
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
                         keep_intermediates=False,
                         tile_budget=None, outline=None) -> ReshapeResult:
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
    default budget, reason ``"too-big"``) or the sentence of
    :data:`ENGINE_REFUSALS` under its reason (``"no-markers"``,
    ``"no-markers-outline"``, ``"outline-per-box"``, ``"too-big-whole"``).
    ``unfilled_tissue_fraction`` measures against the dark mask AFTER the
    clearance around clicks, so a cleared ball counts as tissue.  Progress
    names (:data:`REGION_STAGES`) are emitted as each stage STARTS; a cancel
    is polled between and inside stages and RAISES
    :class:`~.engine.ReshapeCancelled`.

    Under ``params["growth"]="compete"`` (see the module notes) the last
    stage is named ``add_grow`` (:data:`REGION_STAGES_COMPETE`) and every
    finished tile is reported as ``("add_piece", i, k)``; ``tile_budget``
    sets the voxels per tile, halo included (``None`` =
    ``compete.COMPETE_TILE_VOXELS``).  On ``whole_patch=True`` hand the
    elevation in through ``elevation=``, built ONCE over the whole patch:
    the tiles then crop that one volume and equal a single call.

    Under ``params["growth"]="fill_with_background"`` (see the module
    notes) the stages are :data:`REGION_STAGES` (the last, ``add_cut``, is
    the one watershed, which cannot be stopped once it runs).  A
    ``ReshapeCancelled`` raised by ``progress`` at ANY of the four stages
    stops the run there -- unlike the other growth rules, which swallow
    anything the callback raises; any other error the callback raises is
    still swallowed.  ``n_tiles`` is 0.  The spacing must be three finite
    positive numbers (a ``ValueError`` before any stage).
    ``whole_patch=True`` lifts the box budget up to
    :data:`WHOLE_FILL_MAX_VOXELS`; above it the refusal is
    :func:`too_big_whole_sentence` (reason ``"too-big-whole"``), given
    before anything is read.

    ``outline=`` -- the tissue outline for ``params["outline"]`` (bool, or
    integers 0 / 1): the finished outline built ONCE over the whole patch
    (or its box crop), cropped here and used as is.  Required for
    ``"cleared"``; for ``"shrink"`` it may be left out only on the whole
    patch (built here), else the run is refused (``"outline-per-box"``).
    With a tissue outline, ``unfilled_tissue_fraction`` is measured against
    it.  The first competition in a fresh install compiles its solver
    (about 10 s, once): :func:`warm_up_compete` does that up front.
    """
    spacing = _spacing(spacing)
    _check_spec(spec)
    _check_outline_arg(spec, outline)
    sink = MessageSink(log)
    p = spec.params if isinstance(spec, ReshapeSpec) else {}
    compete = p.get("growth") == "compete"
    fill = p.get("growth") == "fill_with_background"
    if fill:
        # every distance of the fill is in microns: an unusable spacing is
        # refused before any stage reads or builds anything
        _checked_spacing(spacing)
    step = _RegionProgress(progress,
                           REGION_STAGES_COMPETE if compete else REGION_STAGES,
                           propagate_cancel=fill)
    out = ReshapeResult(vox_volume=float(np.prod(spacing)), roi=roi,
                        dark_threshold=dark_threshold)
    try:
        # ---- 1. budget --------------------------------------------------- #
        step("add_box")
        shape, existing = _prepare(labels, spec, roi, spacing, cancel_event)
        _outline_refusal(spec, outline)
        _check_cancel(cancel_event)

        # ---- 2. walls, on the whole box ---------------------------------- #
        step("elevation")
        whole = np.ones(shape, bool)
        elev, clahe, ridge_img, _img = _build_wall_stage(
            spec, images=_wall_images(spec, images, roi, shape, elevation),
            roi=roi, region=whole, spacing=spacing, params=p, sink=sink,
            supplied_elevation=elevation, cancel_event=cancel_event)
        if keep_intermediates:
            kept_side = {"clahe": clahe, "ridge": ridge_img}
        del clahe, ridge_img, _img      # not held through the cut (L-A5)
        _check_cancel(cancel_event)

        # ---- 3. dark ------------------------------------------------------ #
        step("add_markers")
        # ---- 3b. the tissue outline (None for "none": no work, no poll) - #
        dark_raw, outline = _dark_and_outline(
            spec, outline=outline, roi=roi, shape=shape,
            dark_channels=dark_channels, dark_threshold=dark_threshold,
            spacing=spacing, cancel_event=cancel_event)

        # ---- 4. new-object markers --------------------------------------- #
        new, n, n_found, counts, source = _marker_stage(
            spec, roi=roi, shape=shape, existing=existing, seed_layer=seed_layer,
            seed_points=seed_points, dark_raw=dark_raw, spacing=spacing,
            cancel_event=cancel_event, outline=outline)
        for key, value in counts.items():
            setattr(out, key, value)
        out.seed_source = source
        out.n_seeds_found = n_found
        out.n_seeds = n
        out.n_seeds_dropped = max(0, n_found - n)
        if n < 1:
            # the walls were built already: handed back with the refusal, so
            # a caller that caches a whole patch's walls need not build them
            # again on the retry
            out.elevation = elev
            raise _no_markers(counts)
        _check_cancel(cancel_event)

        if compete:
            # ---- 5-8 under the competition -------------------------------- #
            result, markers, dark_raw, out.n_fragments_dropped, out.n_tiles = (
                _grow_by_competition(
                    spec, elev=elev, existing=existing, new=new, n=n,
                    outline=outline, dark_raw=dark_raw, seed_points=seed_points,
                    shape=shape, spacing=spacing, tile_budget=tile_budget,
                    step=step, progress=progress, cancel_event=cancel_event))
        elif fill:
            # ---- 5-8 under the fill with a background ---------------------- #
            result, markers, dark_raw, out.n_fragments_dropped = (
                _grow_by_fill_with_background(
                    spec, elev=elev, existing=existing, new=new, n=n,
                    outline=outline, dark_raw=dark_raw, seed_points=seed_points,
                    shape=shape, spacing=spacing, step=step, sink=sink,
                    cancel_event=cancel_event))
            out.n_tiles = 0
        else:
            result, markers, dark_raw, out.n_fragments_dropped = _grow_by_flood(
                spec, elev=elev, existing=existing, new=new, n=n,
                outline=outline, dark_raw=dark_raw, seed_points=seed_points,
                shape=shape, spacing=spacing, step=step, log=log, sink=sink,
                cancel_event=cancel_event)
            out.n_tiles = 0

        # ---- 9. notes ------------------------------------------------------ #
        out.faint_pairs = faint_wall_pairs(
            result, elev, percentile=p["arbiter_low_evidence_percentile"])
        _check_cancel(cancel_event)
        # measured against dark_raw AFTER the click clearance: a cleared
        # ball around a click counts as tissue; with an outline, against it
        if outline is not None:
            out.unfilled_tissue_fraction = _unfilled_in_outline(
                result, existing, outline)
        else:
            out.unfilled_tissue_fraction = unfilled_fraction(result, existing,
                                                             dark_raw)
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

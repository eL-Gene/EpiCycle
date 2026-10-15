"""``reshape_region`` -- one seeded partition of a bounded region.

This is the original membrane-sandbox ``resegment_label`` **rewritten**, not
adapted.  The sandbox hardwired a two-channel vocabulary
(``chans = {"cadherin": ..., "nuclei": ...}``) that the plugin does not have (it
has N named Image layers), and it fell through to ``seeds_auto`` whenever a seed
source found nothing.  Both are gone:

* channels are a caller-supplied ``images`` mapping keyed by layer name; an
  unknown name is a ``ValueError``;
* the ``_alias`` back-compat dict is dropped -- :class:`~.spec.ReshapeSpec`
  validates every slot and raises on an unknown value;
* **zero seeds is a refusal, never a fallback**.  There is no path
  from any seed source to an unseeded watershed;
* the eight ``print`` sites become :class:`~.notes.MessageSink` entries;
* the arbiter runs after the solve and before the refine stage;
* ``cancel_event`` is polled between stages;
* a pre-flight size guard refuses an over-budget job with the numbers named.

**R0 implements the split shape only** (``target="label"``).  The add shape
(``target="region"``, background-only inside a box) and the ``layer_label``
shape are new code with new tests, and land in R2 with the Add tab's pickers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..batch_refine import compute_label_roi
from . import seeds as seed_ops
from .arbiter import arbitrate
from .notes import MessageSink
from .postprocess import merge_small_pieces, merge_to_count, refine_result_guided
from .preproc import apply_preproc, strip_clahe
from .solvers import solve
from .spec import (
    GEOMETRY_ONLY_WALLS,
    IMAGE_DERIVED_SEEDS,
    IMAGE_DERIVED_WALLS,
    ReshapeResult,
    ReshapeSpec,
)
from .walls import build_elevation, raise_neighbour_walls


class ReshapeCancelled(Exception):
    """Raised internally when ``cancel_event`` is set between stages."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise ReshapeCancelled()


#: The stages a caller sees through ``progress``.  Fixed and ordered, so a
#: progress bar can size itself from the first callback.
STAGES = ("roi", "walls", "seeds", "guard", "solve", "arbiter", "refine",
          "stats")


class _Progress:
    """Reports ``(stage_name, i, n)``, once per stage, in order.

    A callback that raises must never take down a segmentation, so every
    call is guarded -- the same rule as the ``log`` sink.
    """

    __slots__ = ("_cb", "_n", "_i")

    def __init__(self, callback):
        self._cb = callback
        self._n = len(STAGES)
        self._i = 0

    def __call__(self, stage):
        self._i += 1
        if self._cb is None:
            return
        try:
            self._cb(stage, self._i, self._n)
        except Exception:
            pass


def remap_flags(flags, id_map):
    """Translate arbiter flags from interim piece ids to FINAL label ids.

    Pieces absent from ``id_map`` were not written and drop out.  Two
    interim pieces can legitimately map to the same final id (an unticked
    piece folded back into the target), in which case the flag lands on
    that id.
    """
    out = {}
    for piece_id, flag in (flags or {}).items():
        final = (id_map or {}).get(piece_id)
        if final is not None:
            out[final] = flag
    return out


def _piece_stats(partition, roi):
    """Per-piece ``id`` / ``voxels`` / ``centroid_zyx`` / ``bbox``, offset
    into VOLUME coordinates so a caller never has to redo the arithmetic
    (and never forgets to)."""
    offset = np.array([sl.start or 0 for sl in roi], float)
    rows = []
    for pid in sorted(set(int(v) for v in np.unique(partition)) - {0}):
        coords = np.argwhere(partition == pid)
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


def _resolve_channel(images, name, *, what, param_key):
    """Fetch one named single-channel 3D array from the caller's mapping."""
    images = images or {}
    if not name:
        raise ValueError(
            "{} needs a channel: set params[{!r}] to one of the available "
            "image names ({}).".format(
                what, param_key, ", ".join(sorted(images)) or "none supplied",
            )
        )
    if name not in images:
        raise ValueError(
            "{} names image {!r}, which was not supplied. Available: {}.".format(
                what, name, ", ".join(sorted(images)) or "none"
            )
        )
    array = np.asarray(images[name])
    if array.ndim != 3:
        raise ValueError(
            "Image {!r} is {}-dimensional; the reshape engine takes 3D "
            "single-channel arrays only (collapse channels in the caller).".format(
                name, array.ndim
            )
        )
    return array


def _crop_to_roi(array, roi, region_shape, *, what):
    """Accept either a full-volume array (cropped here) or an ROI-sized one."""
    array = np.asarray(array)
    if array.shape == region_shape:
        return array
    try:
        cropped = array[roi]
    except Exception as exc:
        raise ValueError("{} could not be cropped to the ROI: {!r}".format(what, exc))
    if cropped.shape != region_shape:
        raise ValueError(
            "{} has shape {} which matches neither the volume ROI {} nor the "
            "ROI crop.".format(what, array.shape, region_shape)
        )
    return cropped


def _count_ids(array):
    return int((np.unique(array) != 0).sum())


# --------------------------------------------------------------------------- #
# the wall stage
# --------------------------------------------------------------------------- #
def _build_wall_stage(spec, *, images, roi, region, spacing, params, sink,
                      supplied_elevation=None):
    """Build the elevation the solver floods.  ``(elevation, clahe, ridge, img)``.

    Lifted OUT of :func:`reshape_region` unchanged so that
    :func:`find_seeds` can reuse it verbatim.  That is not tidiness: the
    ``distance`` seed detector reads this elevation (see ``_build_seeds``),
    so a Find and a Preview that built the wall differently would place
    different markers, and the user's cherry-pick ticks would then select
    the wrong ones.
    """
    p = params
    if supplied_elevation is not None:
        supplied_elevation = np.asarray(supplied_elevation, np.float32)
        if supplied_elevation.shape != region.shape:
            raise ValueError(
                "the supplied elevation has shape {} but this ROI is "
                "{}; pass the roi= it was built for.".format(
                    supplied_elevation.shape, region.shape))
    elev_img = None
    if spec.walls not in GEOMETRY_ONLY_WALLS and supplied_elevation is None:
        raw = _resolve_channel(images, p["wall_channel"],
                               what="The wall source",
                               param_key="wall_channel")
        elev_img = _crop_to_roi(raw, roi, region.shape, what="The wall channel")
        chain = strip_clahe(p["wall_preproc"])
        if spec.arbiter and "zcorrect" in str(chain or "").split("+"):
            # Not forbidden -- the user may genuinely need it on attenuated
            # data -- but they should know what it costs the arbiter.
            sink.emit(
                "Depth correction is on -- a wall lying in a single "
                "z-slice may be flattened; check stacked objects."
            )
        # The wall keeps its OWN CLAHE (the `clip` knob inside the wall
        # builders), so stripping `clahe` from the chain here makes
        # "no double-CLAHE" an engine invariant rather than a widget
        # discipline assumption (ported from the membrane sandbox).
        if chain not in ("none", None, ""):
            elev_img = apply_preproc(elev_img, chain, gamma=p["wall_gamma"],
                                     bg_radius=p["wall_bg"],
                                     denoise_size=p["wall_denoise"])
    if supplied_elevation is not None:
        return supplied_elevation, None, None, elev_img
    elevation, clahe, ridge_img = build_elevation(
        spec.walls, elev_img, region, spacing=spacing,
        clip=p["clip"], smooth=p["smooth"], ridge=p["ridge"],
    )
    return elevation, clahe, ridge_img, elev_img


# --------------------------------------------------------------------------- #
# the seed stage
# --------------------------------------------------------------------------- #
def _build_seeds(spec, *, images, roi, region, elevation, spacing, seed_layer):
    """Return ``(seeds_full, source_label)`` -- seeds BEFORE masking to the target."""
    p = spec.params
    kind = spec.seeds

    if kind in ("paint", "labels"):
        if seed_layer is None:
            raise ValueError(
                "seeds={!r} needs a seed volume: pass seed_layer=<array> (the "
                "painted seeds, or the trusted guide labels).".format(kind)
            )
        crop = _crop_to_roi(seed_layer, roi, region.shape, what="The seed layer")
        seeds_full = seed_ops.seeds_from_labels(crop)
        return seeds_full, ("painted seeds" if kind == "paint" else "a trusted mask")

    if kind == "distance":
        # `region=` matters here more than anywhere else: the elevation
        # this reads is the WALL stage's, over the whole crop, and
        # without the footprint every regional maximum of the distance
        # transform belongs to the background on a branched object
        # (measured).
        return (
            seed_ops.seeds_distance_maxima(elevation, spacing, p["h"],
                                           region=region),
            "distance maxima (geometry only)",
        )

    # every remaining kind reads an image channel
    raw = _resolve_channel(images, p["seed_channel"], what="The seed source",
                           param_key="seed_channel")
    crop = _crop_to_roi(raw, roi, region.shape, what="The seed channel")
    img = apply_preproc(crop, p["seed_preproc"], gamma=p["seed_gamma"],
                        clip=p["seed_clip"], bg_radius=p["seed_bg"],
                        denoise_size=p["seed_denoise"])
    min_distance = seed_ops.physical_to_voxel_scalar(p["min_distance_um"], spacing)
    max_sigma = seed_ops.physical_to_voxel_scalar(p["max_sigma_um"], spacing)
    channel = p["seed_channel"]

    if kind == "peaks":
        return (
            seed_ops.seeds_from_nuclei(img, thresh=p["seed_thresh"],
                                       smooth=p["seed_smooth"],
                                       min_distance=min_distance, spacing=spacing),
            "intensity peaks in {}".format(channel),
        )
    if kind == "hmax":
        return (
            seed_ops.seeds_hmaxima(img, h=p["h"], smooth=p["seed_smooth"],
                                   region=region, spacing=spacing),
            "prominence-gated peaks (h-maxima) in {}".format(channel),
        )
    if kind == "hmin":
        return (
            seed_ops.seeds_membrane_minima(img, p["h"], smooth=p["seed_smooth"],
                                           region=region, spacing=spacing),
            "interior basins (h-minima) in {}".format(channel),
        )
    if kind == "blobs_log":
        return (
            seed_ops.seeds_blobs_log(img, threshold=p["h"],
                                     max_sigma=max(2.0, float(max_sigma)),
                                     region=region, spacing=spacing),
            "LoG blobs in {}".format(channel),
        )
    if kind == "blobs_dog":
        return (
            seed_ops.seeds_blobs_dog(img, threshold=p["h"],
                                     max_sigma=max(2.0, float(max_sigma)),
                                     region=region, spacing=spacing),
            "DoG blobs in {}".format(channel),
        )
    if kind == "threshcc":
        return (
            seed_ops.seeds_threshold_cc(img, thresh=p["seed_thresh"],
                                        smooth=p["seed_smooth"], region=region,
                                        spacing=spacing),
            "threshold components in {}".format(channel),
        )
    # unreachable: ReshapeSpec validated the enum
    raise ValueError("unhandled seed kind {!r}".format(kind))


# --------------------------------------------------------------------------- #
# the marker-only entry point (panel Brick 2a)
# --------------------------------------------------------------------------- #
@dataclass
class SeedReport:
    """What a "Find markers" answers with.

    A refusal is an ANSWER here too: zero markers comes back as
    ``status="refused"`` carrying a plain-language sentence, never an
    exception and never a fallback detector.
    """

    status: str = "ok"                 # "ok" | "refused"
    refusal_reason: str | None = None

    roi: tuple | None = None
    #: the markers MASKED TO THE CELL and otherwise **unfiltered** --
    #: ``keep_seed_ids`` narrows the solve, not the picker, or a marker
    #: could never be ticked back on.
    seeds: Any = None

    n_found: int = 0                   # before masking to the cell
    n_in_region: int = 0               # what the picker lists

    seed_source: str = ""
    #: per-marker ``id`` / ``voxels`` / ``centroid_zyx`` / ``bbox``, in
    #: VOLUME coordinates (the ROI offset is already applied).
    markers: list = field(default_factory=list)
    messages: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def find_seeds(labels, images, spec, *, target_id=None, roi=None, margin=4,
               spacing=(1, 1, 1), seed_layer=None, seed_points=None,
               dark_threshold=None, dark_channels=None, cancel_event=None):
    """Run ONLY the seed stage of :func:`reshape_region` and report it.

    Composes ``compute_label_roi`` (only when ``roi is None``) ->
    ``_crop_to_roi`` -> ``_build_wall_stage`` -> ``_build_seeds``, in the
    same order and with the same arguments the full engine uses, so that::

        find_seeds(..., roi=R).seeds == reshape_region(
            ..., roi=R, keep_seed_ids=None).seeds

    for every detector and every spacing.  **Pass the Preview's ``roi``.**
    The detectors normalise and threshold over the CROP and number the
    markers by enumeration order, so two different crops give different
    markers with different ids -- and the ticks would then select the
    wrong ones.

    ``target="region"`` (the Add box, Add rebuild) is dispatched
    to :func:`.region.find_region_seeds` before anything else; there
    ``roi`` is required and ``target_id`` unused.  ``seed_points``,
    ``dark_threshold``, ``dark_channels`` and ``cancel_event`` are read by
    the region target only; the label path ignores them.
    """
    if not isinstance(spec, ReshapeSpec):
        raise TypeError("spec must be a ReshapeSpec, got {!r}".format(type(spec)))
    if spec.target == "region":
        if target_id is not None:
            raise ValueError(
                "target='region' takes no target_id (the box is roi=); got "
                "target_id={!r}.".format(target_id))
        from . import region as _region
        return _region.find_region_seeds(
            labels, images, spec, roi=roi, dark_channels=dark_channels,
            seed_layer=seed_layer, seed_points=seed_points,
            dark_threshold=dark_threshold, spacing=spacing,
            cancel_event=cancel_event,
        )
    if spec.target != "label":
        raise NotImplementedError(
            "ReshapeSpec.target={!r} has no marker stage of its own.".format(
                spec.target)
        )
    if target_id is None:
        raise ValueError("target='label' needs target_id=<the label to look in>.")

    sink = MessageSink(None)
    p = spec.params
    spacing = tuple(float(s) for s in spacing)
    labels = np.asarray(labels)
    images = dict(images or {})
    target_id = int(target_id)
    report = SeedReport()

    def _refuse(reason, message):
        report.status = "refused"
        report.refusal_reason = reason
        sink.emit(message)
        report.messages = list(sink.messages)
        return report

    if roi is None:
        roi = compute_label_roi(labels, target_id, int(margin))
    if roi is None:
        return _refuse(
            "no-target",
            "Label {} is not in this volume -- nothing to look in.".format(
                target_id),
        )
    report.roi = roi
    label_crop = labels[roi]
    region = label_crop == target_id
    if not region.any():
        return _refuse(
            "no-target",
            "Label {} has no voxels in this region -- nothing to look "
            "in.".format(target_id),
        )

    elevation, _clahe, _ridge, _elev_img = _build_wall_stage(
        spec, images=images, roi=roi, region=region, spacing=spacing,
        params=p, sink=sink,
    )
    seeds_full, seed_source = _build_seeds(
        spec, images=images, roi=roi, region=region, elevation=elevation,
        spacing=spacing, seed_layer=seed_layer,
    )
    report.seed_source = seed_source

    n_found = _count_ids(seeds_full)
    seeds = np.where(region, seeds_full, 0)
    n_in_region = _count_ids(seeds)
    report.seeds = seeds
    report.n_found = n_found
    report.n_in_region = n_in_region
    if n_found > n_in_region:
        sink.emit(
            "{} of {} markers lie outside this object -- ignored.".format(
                n_found - n_in_region, n_found)
        )
    report.markers = _piece_stats(seeds, roi)

    if n_in_region < 1:
        return _refuse(
            "no-seeds",
            "No markers found in this object -- lower the prominence, change "
            "the marker source, or paint one. (Nothing was changed.)",
        )
    report.messages = list(sink.messages)
    return report


# --------------------------------------------------------------------------- #
# the public entry point
# --------------------------------------------------------------------------- #
#: How many markers the SHAPE finder may return before the engine calls
#: the shape unreadable and refuses (audit M7, 2026-09-20).
#:
#: `seeds_distance_maxima` finds h-maxima of the interior distance
#: transform.  On a solid object that is a handful of lobes.  On a
#: HOLLOW one -- a cell imaged only at its membrane, a nucleus whose
#: interior washed out -- every point of the shell is an equal local
#: maximum, so it returns one marker per bump of the discretised
#: surface, and the SAME number at every `h`, because there is no
#: thickness variation left for `h` to bite on.  Measured on a closed
#: spherical shell, r 12-14 in a 40-cube (6.9 % fill): 68 markers with
#: `walls="skeleton"`, 98 with `walls="distance"`, `status=ok`, one
#: piece per marker.  The user asked to split one object in two and got
#: it shattered, with nothing on screen saying why.
#:
#: 32 because it is far above anything a real reshape asks for -- the
#: measured solid shapes return 1 to 3, and "I expect exactly n" tops
#: out in the low tens -- and far below the 42 the mildest shell in the
#: sweep produced.  It bounds a failure, it does not ration a feature.
#: Only the shape finder is capped: a guide mask or an image detector
#: may legitimately report many markers.
SHAPE_SEED_CAP = 32


def reshape_region(labels, images, spec, *, target_id=None, roi=None,
                   seed_layer=None, spacing=(1, 1, 1), margin=4,
                   keep_seed_ids=None, cancel_event=None, log=None,
                   progress=None, elevation=None,
                   keep_intermediates=False, seed_points=None,
                   dark_threshold=None, dark_channels=None):
    """Re-segment ONE label into pieces, staying inside its footprint.

    Parameters
    ----------
    labels : ndarray
        The labels volume **from memory**, never re-read from disk.
    images : mapping[str, ndarray]
        Named single-channel 3D image arrays.  Multichannel volumes must be
        channel-collapsed by the caller.
    spec : ReshapeSpec
        The validated recipe.
    target_id : int
        The label to re-segment.  Required for ``target="label"``.
    roi : tuple[slice] | None
        Override the computed bounding box (e.g. to reuse a Preview's ROI).
    seed_layer : ndarray | None
        The painted seeds (``seeds="paint"``) or the trusted guide labels
        (``seeds="labels"``), full-volume or already ROI-sized.
    keep_seed_ids : iterable[int] | None
        Cherry-pick: keep only these seed ids.  Detection is deterministic, so a
        Preview and a commit see the same seed ids.
    cancel_event : threading.Event | None
        Polled **between** stages.  Three stages cannot be interrupted once
        entered (``random_walker``, ``diffuse_labels3D``, ``guidedfilter``), so
        cancel means "abandon the result", not "stop the computation".
    log : callable | None
        Mirrors every message live; all messages also land in
        ``result.messages``.

    Returns
    -------
    ReshapeResult
        ``status="ok"``, ``"refused"`` (a plain-language answer: no target, no
        seeds, over budget) or ``"cancelled"``.  **Never a fallback.**

    ``target="region"`` (the Add box, Add rebuild) is dispatched to
    :func:`.region.reshape_empty_region` BEFORE the label path, which stays
    textually unchanged below.  ``seed_points``, ``dark_threshold`` and
    ``dark_channels`` are read by the region target only.  Unlike the label
    path, the region target lets :class:`ReshapeCancelled` propagate.
    """
    if not isinstance(spec, ReshapeSpec):
        raise TypeError("spec must be a ReshapeSpec, got {!r}".format(type(spec)))
    if spec.target == "region":
        # label-path keywords the region target would silently ignore
        # (``margin`` keeps its default and is unused: the box is ``roi``)
        if target_id is not None:
            raise ValueError(
                "target='region' takes no target_id (the box is roi=); got "
                "target_id={!r}.".format(target_id))
        if keep_seed_ids is not None:
            raise ValueError(
                "target='region' takes no keep_seed_ids: every marker it is "
                "given grows (drop a click or a source instead).")
        from . import region as _region
        return _region.reshape_empty_region(
            labels, images, spec, roi=roi, dark_channels=dark_channels,
            seed_layer=seed_layer, seed_points=seed_points,
            dark_threshold=dark_threshold, spacing=spacing,
            cancel_event=cancel_event, log=log, progress=progress,
            elevation=elevation, keep_intermediates=keep_intermediates,
        )
    if spec.target != "label":
        raise NotImplementedError(
            "ReshapeSpec.target={!r} is not implemented in R0. The engine ships "
            "the split shape (target='label') first; the add shape lands with "
            "the Add tab's pickers.".format(spec.target)
        )
    if target_id is None:
        raise ValueError("target='label' needs target_id=<the label to reshape>.")

    sink = MessageSink(log)
    step = _Progress(progress)
    supplied_elevation = elevation
    p = spec.params
    spacing = tuple(float(s) for s in spacing)
    labels = np.asarray(labels)
    images = dict(images or {})
    target_id = int(target_id)

    result_obj = ReshapeResult(vox_volume=float(np.prod(spacing)))

    def _refuse(reason, message):
        step("refused")
        result_obj.status = "refused"
        result_obj.refusal_reason = reason
        sink.emit(message)
        result_obj.messages = list(sink.messages)
        return result_obj

    try:
        # ---- 0. the ROI ---------------------------------------------------- #
        if roi is None:
            roi = compute_label_roi(labels, target_id, int(margin))
        if roi is None:
            return _refuse(
                "no-target",
                "Label {} is not in this volume -- nothing to reshape.".format(target_id),
            )
        step("roi")
        result_obj.roi = roi
        label_crop = labels[roi]
        region = label_crop == target_id
        result_obj.region = region
        if not region.any():
            return _refuse(
                "no-target",
                "Label {} has no voxels in this region -- nothing to "
                "reshape.".format(target_id),
            )
        _check_cancel(cancel_event)

        # ---- 1. the wall (elevation) --------------------------------------- #
        # A caller that already built the wall for a Preview passes it back
        # for the Accept rather than paying for it twice -- the wall build
        # is the engine's most expensive stage, and rebuilding it also risks
        # a Preview and a commit disagreeing.
        elevation, clahe, ridge_img, elev_img = _build_wall_stage(
            spec, images=images, roi=roi, region=region, spacing=spacing,
            params=p, sink=sink, supplied_elevation=supplied_elevation,
        )
        # One of the sandbox's two dropped summary
        # prints.  Without it "the cut went nowhere" and "there was nothing
        # to cut on" read identically on screen -- a flat range says the
        # second out loud.  Measured INSIDE the cell, because the elevation
        # outside it is not what the solver floods.
        #
        # Only for a wall READ OUT OF A CHANNEL: for a shape wall the
        # number is a fact about the distance transform, and the result
        # slot has room for a sentence that answers a question.
        if spec.walls in IMAGE_DERIVED_WALLS:
            sink.emit(_wall_contrast_line(elevation, region))
        step("walls")
        _check_cancel(cancel_event)

        # ---- 2. the seeds --------------------------------------------------- #
        seeds_full, seed_source = _build_seeds(
            spec, images=images, roi=roi, region=region, elevation=elevation,
            spacing=spacing, seed_layer=seed_layer,
        )
        step("seeds")
        result_obj.seed_source = seed_source
        n_found = _count_ids(seeds_full)
        seeds = np.where(region, seeds_full, 0)   # only seeds inside the target
        n_inside = _count_ids(seeds)
        n_dropped = max(0, n_found - n_inside)
        result_obj.n_seeds_found = n_found
        result_obj.n_seeds_dropped = n_dropped
        if n_dropped:
            sink.emit(
                "{} of {} markers lie outside this object -- ignored.".format(
                    n_dropped, n_found
                )
            )

        # cherry-pick: keep only the seed IDs the user ticked (from Preview).
        if keep_seed_ids is not None:
            keep = sorted(set(int(v) for v in keep_seed_ids))
            seeds = np.where(np.isin(seeds, keep) if keep else False, seeds, 0)

        n_seeds = _count_ids(seeds)
        result_obj.seeds = seeds
        result_obj.n_seeds = n_seeds
        # A hollow or very thin object shatters the SHAPE finder, and a
        # silent 68-piece cut is worse than a refusal (audit M7).  Only
        # `distance`: a mask or an image detector may legitimately hand
        # over many markers -- that is the user's own label image
        # talking, not a geometry artefact.
        if spec.seeds == "distance" and n_seeds > SHAPE_SEED_CAP:
            return _refuse(
                "shape-seeds-exploded",
                "The shape finder found {} markers; this object looks "
                "hollow or very thin. The shape finder needs a solid "
                "body to measure. Use an image channel for markers "
                "instead. (Nothing was changed.)".format(n_seeds),
            )
        if n_seeds < 1:
            return _refuse(
                "no-seeds",
                "No markers found in this object -- lower the prominence, "
                "change where the markers come from, or paint one. "
                "(Nothing was changed.)",
            )
        _check_cancel(cancel_event)

        # ---- 3. the pre-flight size guard ---------------------------------- #
        region_voxels = int(region.sum())
        budget = int(p["size_budget"])
        product = region_voxels * n_seeds
        if budget > 0 and product > budget:
            return _refuse(
                "too-big",
                "This object is too large to reshape in one go: {:,} voxels x {} "
                "markers = {:,}, over the {:,} budget. Reshape a smaller object, "
                "or use fewer markers.".format(
                    region_voxels, n_seeds, product, budget),
            )

        step("guard")

        # ---- 4. neighbour walls + solve ------------------------------------ #
        elev_walls = raise_neighbour_walls(elevation, label_crop, target_id)
        result_obj.elevation = elev_walls
        _check_cancel(cancel_event)
        partition = solve(elev_walls, seeds, region, solver=spec.solver,
                          beta=p["rw_beta"], compactness=p["compactness"],
                          spacing=spacing, sink=sink)
        step("solve")
        _check_cancel(cancel_event)

        # ---- 5. the arbiter (before refine) ------------------------ #
        flags: dict = {}
        if spec.arbiter:
            partition, report, flags = arbitrate(
                partition, elev_walls, region, spacing=spacing,
                percentile=p["arbiter_percentile"],
                low_evidence_percentile=p["arbiter_low_evidence_percentile"],
                band_um=p["arbiter_band_um"], sink=sink,
            )
            result_obj.arbiter = report
        result_obj.flags = flags
        step("arbiter")
        _check_cancel(cancel_event)

        # ---- 6. refine ------------------------------------------------------ #
        seed_labels = set(int(v) for v in np.unique(seeds)) - {0}
        if "guided" in spec.refine:
            guide = elev_img if elev_img is not None else elev_walls
            partition = refine_result_guided(partition, region, guide,
                                             p["refine_radius"], p["refine_eps"],
                                             sink=sink)
            _check_cancel(cancel_event)
        if "merge_small" in spec.refine and p["min_size"] and int(p["min_size"]) > 0:
            # Absorb sub-min_size slivers HERE (not dropped at write-back -> no
            # holes), so pieces reported == pieces committed. `protect` means a
            # genuine seed-backed cell is never absorbed.
            partition = merge_small_pieces(partition, int(p["min_size"]),
                                           protect=seed_labels)
        n_before = len(set(np.unique(partition)) - {0})
        # OUTSIDE the branch below, deliberately (UX round 6.2 / audit
        # M3): recorded even when the merge stage never fires, because
        # that is the case the diagnostic exists for.
        result_obj.n_pieces_before_merge = n_before
        target_count = p["target_count"]
        if ("merge_to_count" in spec.refine and target_count
                and int(target_count) >= 1 and n_before > int(target_count)):
            partition = merge_to_count(partition, int(target_count))
            result_obj.merged_to_count = True
            sink.emit("Merged {} pieces down to the expected {}.".format(
                n_before, int(target_count)))
        step("refine")
        _check_cancel(cancel_event)

        # ---- 7. stats -------------------------------------------------------- #
        piece_ids = sorted(set(int(v) for v in np.unique(partition)) - {0})
        n_pieces = len(piece_ids)
        if (target_count and int(target_count) >= 1
                and n_pieces < int(target_count)):
            sink.emit(
                "Wanted {} pieces but only got {} -- too FEW markers (lower "
                "the prominence or the minimum gap, or change where the "
                "markers come from).".format(int(target_count), n_pieces)
            )
        if n_pieces == 1:
            sink.emit("1 piece -- boundary refined." if spec.refine
                      else "1 piece -- the object was kept whole.")

        piece_sizes = sorted((int((partition == pid).sum()) for pid in piece_ids),
                             reverse=True)
        result_obj.result = partition
        result_obj.n_pieces = n_pieces
        result_obj.piece_sizes = piece_sizes
        result_obj.coverage = (float(sum(piece_sizes)) / region_voxels
                               if region_voxels else 0.0)
        # flags only ever describe surviving pieces
        result_obj.flags = {k: v for k, v in result_obj.flags.items()
                            if k in set(piece_ids)}
        if keep_intermediates:
            result_obj.intermediates = {
                "clahe": clahe, "ridge": ridge_img, "elevation_raw": elevation,
            }
        result_obj.piece_stats = _piece_stats(partition, roi)
        step("stats")
        result_obj.messages = list(sink.messages)
        return result_obj

    except ReshapeCancelled:
        result_obj.status = "cancelled"
        result_obj.refusal_reason = "cancelled"
        sink.emit("Stopped waiting -- the result was abandoned.")
        result_obj.messages = list(sink.messages)
        return result_obj


def _wall_contrast_line(elevation, region) -> str:
    """"Wall contrast 0.12-0.81." -- the range the solver actually sees.

    Two decimals, ASCII hyphen, always one sentence: it is read in a
    Qt label and mirrored to a console that may be cp1252 (the
    2026-09-07 hazard class).
    """
    try:
        vals = np.asarray(elevation)[np.asarray(region, dtype=bool)]
        if vals.size:
            return "Wall contrast {:.2f}-{:.2f}.".format(
                float(vals.min()), float(vals.max()))
    except Exception:
        pass
    return "Wall contrast could not be measured."


# --------------------------------------------------------------------------- #
# the write-back
# --------------------------------------------------------------------------- #
def apply_reshape_to_volume(volume, roi, result, *, allowed_ids, id_map,
                            flags=None, return_flags=False):
    """Write ``result`` into ``volume[roi]``.  Returns the voxel count written.

    The safe-write guard, generalised::

        write_mask = np.isin(volume_roi, allowed_ids) & (result > 0)

    ``allowed_ids`` is ``{target_label_id}`` for a split-shaped preset and
    ``{0}`` for an add-shaped one.  **A neighbour label is never overwritten, in
    any preset, ever** -- and the pure layer never depends on a pre-punched hole
    the way the third form in the codebase does.

    ``id_map`` maps every interim piece id to its final label id.  An **unticked**
    piece is expressed here by mapping it back to the surviving target id, which
    is why unticking a piece on a split-shaped preset returns those voxels to the
    cell instead of deleting tissue.
    """
    def _answer(written):
        if not return_flags:
            return written
        return written, remap_flags(flags, id_map)

    volume_roi = volume[roi]
    result = np.asarray(result)
    if volume_roi.shape != result.shape:
        raise ValueError(
            "result shape {} does not match volume[roi] shape {}".format(
                result.shape, volume_roi.shape
            )
        )
    piece_ids = sorted(set(int(v) for v in np.unique(result)) - {0})
    missing = [pid for pid in piece_ids if pid not in id_map]
    if missing:
        raise ValueError(
            "id_map is missing final ids for piece(s) {} -- every piece must be "
            "mapped (map an unticked piece back to the target id).".format(missing)
        )

    allowed = sorted(set(int(v) for v in allowed_ids))
    write_mask = np.isin(volume_roi, allowed) & (result > 0)
    if not write_mask.any():
        return _answer(0)
    lookup = np.zeros(max(piece_ids) + 1, dtype=volume.dtype)
    for pid in piece_ids:
        lookup[pid] = id_map[pid]
    volume_roi[write_mask] = lookup[result[write_mask]]
    return _answer(int(write_mask.sum()))

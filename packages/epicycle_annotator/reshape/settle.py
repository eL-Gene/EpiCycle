"""Settle one object's outline onto the wall signal.

The Refine tab's seventh method, and the only one that reads the same wall
elevation the Reshape engine builds.  It is a **seeded partition of a thin
physical band around one outline**, which is the Reshape shape again with the
band standing in for the whole region -- so it lives beside the engine and
borrows its vocabulary, its solvers and its size guard.

**Pure: no Qt, no napari**, numpy / scipy / skimage only, with u-Segment3D as
a lazy optional import (``segment3D`` is imported *inside* the function, per
:mod:`~.optional_deps`).

What it does, in the order it does it
-------------------------------------
1. ``cur = labels_roi == label_id``; a missing object is a **refusal**, not an
   exception.
2. The band is **physical**: ``dist_out = edt(~cur, sampling=spacing)`` and
   ``dist_in = edt(cur, sampling=spacing)``, and the band is the shell of
   width ``band_um`` on either side of the outline --
   ``(~cur & (dist_out <= band_um)) | (cur & (dist_in <= band_um))``.

   .. note::
 writes that union without the two side masks.  Taken
      literally ``(dist_out <= b) | (dist_in <= b)`` is the whole array --
      ``dist_out`` is 0 everywhere *inside* the object and ``dist_in`` is 0
      everywhere *outside* it -- so each half is masked to its own side
      here.  The set this produces is the one every other sentence in 2.7
      describes ("only voxels in the band may change", "the core is
      ``cur & (dist_in > band_um)``"), and it is what the markers below are
      the complement of.

3. The **markers**: ``1`` = this object's core (``cur & ~band``); ``2..k`` =
   each neighbouring label's core, one id each; ``k+1`` = the background
   core.  Seeding the neighbours is what makes the gap between two objects
   be divided **on the wall signal** rather than swallowed whole by whichever
   object is settling (question 1).
4. The **solve**, on the same three solvers as Reshape.
5. The **accept rule**: only voxels in the band may change, a swap is
   accepted only between **background and this object**, and **no other label
   ever loses a voxel**.  Then only the connected component(s) touching the
   original footprint are kept.
6. The **empty-object refusal**: a result with nothing left is refused here
   *and* refused again by :func:`apply_settle_to_volume`, which returns -1 and
   writes nothing.

Everything it measured is on the result, because the panel's status line is
made of those numbers and not of adjectives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .optional_deps import (
    USEGMENT3D_MISSING,
    have_usegment3d,
    import_diffuse_labels3D,
)
from .preproc import norm
from .solvers import solve as _solve_partition
from .spec import DEFAULT_SIZE_BUDGET, SOLVERS


@dataclass
class SettleResult:
    """Everything the panel needs to say what happened, and nothing else.

    ``result_roi`` carries **only** ``label_id`` and 0 -- the same shape the
    rest of the Refine tab's pure functions return, so the orchestrator's
    write-back is the one it already has.
    """

    result_roi: np.ndarray
    n_modified: int = 0
    n_grown: int = 0
    n_shrunk: int = 0
    n_border_background: int = 0
    n_border_held: int = 0
    n_band_voxels: int = 0
    n_neighbour_markers: int = 0
    mean_move_um: float = 0.0
    max_move_um: float = 0.0
    hit_band_limit: bool = False
    status: str = "ok"
    messages: list = field(default_factory=list)
    #: The wall picture the solve read, when :func:`settle_with_walls` built
    #: it on the worker -- the walls tick ("Show what the tool sees") shows
    #: it.  ``None`` from a bare :func:`settle_label` call, which was handed
    #: its picture and has nothing new to hand back.
    elevation: np.ndarray | None = None

    @property
    def border_background_fraction(self) -> float:
        """The number the staged status line leads with.

        Counted over **faces**, not voxels, so the same measurement can be
        taken independently off the volume (the B1 driver does exactly
        that).  0.0 when the object has no border at all.
        """
        total = int(self.n_border_background) + int(self.n_border_held)
        if total <= 0:
            return 0.0
        return float(self.n_border_background) / float(total)


def roi_margin_voxels(band_um: float, spacing=(1.0, 1.0, 1.0)) -> int:
    """The SCALAR ``margin=`` a caller hands ``_snapshot_label_roi``.

    ``_snapshot_label_roi`` takes one number for all three axes and is
    shared with Reshape's Start, so it is not given a per-axis keyword
    (audit C-H2).  The largest padding any axis needs is
    ``max(ceil(band_um / s))`` -- never too small, and at worst a few
    voxels too generous on the fine axes, which costs a slightly bigger
    crop and nothing else.
    """
    band = float(max(band_um, 0.0))
    out = 1
    for s in tuple(spacing) or (1.0,):
        s = float(s) if float(s) > 0 else 1.0
        out = max(out, int(math.ceil(band / s)))
    return int(out)


def _border_face_counts(cur: np.ndarray, labels_roi: np.ndarray):
    """``(faces_to_background, faces_to_another_label)`` over 6-connectivity.

    Faces that leave the ROI are not counted -- the ROI is padded by the
    band, so a face on its wall would be an artefact of the crop.
    """
    to_bg = 0
    to_other = 0
    for axis in range(cur.ndim):
        a = [slice(None)] * cur.ndim
        b = [slice(None)] * cur.ndim
        a[axis] = slice(0, -1)
        b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        for lo, hi in ((a, b), (b, a)):
            face = cur[lo] & ~cur[hi]
            if not face.any():
                continue
            outside = labels_roi[hi][face]
            to_bg += int(np.count_nonzero(outside == 0))
            to_other += int(np.count_nonzero(outside != 0))
    return int(to_bg), int(to_other)


def _refused(labels_roi, label_id, messages, **counts) -> SettleResult:
    """A refusal is an ANSWER: the object comes back untouched."""
    out = np.zeros_like(labels_roi)
    out[labels_roi == label_id] = label_id
    return SettleResult(
        result_roi=out, n_modified=0, status="refused",
        messages=list(messages), **counts,
    )


def _diffuse(dense, elevation, region, *, clamp, n_iter, sink_messages):
    """``label_diffusion``, with Settle's own ``clamp`` and ``n_iter``.

    :mod:`~.solvers` pins both at their Reshape values and is explicitly
    out of scope for this brick, so the one solver whose two
    knobs this tool exposes is called here instead.  The drop-guard is the
    same one ``solvers.solve`` applies: a diffusion that makes a marker
    vanish keeps the watershed answer.
    """
    diffuse_labels3D = import_diffuse_labels3D()
    guide = norm(elevation).astype(np.float32)
    refined = diffuse_labels3D(
        dense.astype(np.int32), guide,
        clamp=float(clamp), n_iter=int(n_iter), alpha=0.8,
        affinity_type="heat",
    )
    out = np.where(region, np.asarray(refined), 0).astype(np.int32)
    dropped = ((set(int(v) for v in np.unique(dense)) - {0})
               - set(int(v) for v in np.unique(out)))
    if dropped:
        sink_messages.append(
            "Settling made region(s) {} disappear -- kept the sharp "
            "answer instead.".format(sorted(dropped))
        )
        return dense
    return out


def settle_label(
    labels_roi: np.ndarray,
    elevation: np.ndarray,
    label_id: int,
    *,
    band_um: float = 2.0,
    spacing=(1.0, 1.0, 1.0),
    solver: str = "watershed",
    beta: float = 130.0,
    compactness: float = 0.0,
    n_iter: int = 10,
    clamp: float = 0.5,
    size_budget: int = DEFAULT_SIZE_BUDGET,
) -> SettleResult:
    """Move one object's outline onto the wall signal, inside a physical band.

    Parameters
    ----------
    labels_roi
        The label ROI **as it is in the volume** -- neighbours included, since
        they are what competes for the gap.  Never modified.
    elevation
        The wall picture, same shape, high = wall, already normalised to
        [0, 1] by :func:`~.walls.build_elevation`.
    band_um
        How far the outline may move, in microns, on either side.
    spacing
        ``(z, y, x)`` voxel size in microns.

    Returns
    -------
    SettleResult
        ``status`` is ``"ok"`` (something moved), ``"unchanged"`` (nothing
        could move, and the caller says why) or ``"refused"`` (an answer
        with a sentence, never an exception).
    """
    labels_roi = np.asarray(labels_roi)
    label_id = int(label_id)
    messages: list = []

    elev = np.asarray(elevation, np.float32) if elevation is not None else None
    if elev is None or elev.shape != labels_roi.shape:
        messages.append(
            "The wall picture does not match this object's region, so "
            "nothing was changed."
        )
        return _refused(labels_roi, label_id, messages)

    if str(solver) not in SOLVERS:
        messages.append(
            "'{}' is not a solver this tool has, so nothing was "
            "changed.".format(solver)
        )
        return _refused(labels_roi, label_id, messages)

    cur = labels_roi == label_id
    if not cur.any():
        messages.append(
            "Object {} is not in this region any more, so nothing was "
            "changed.".format(label_id)
        )
        return _refused(labels_roi, label_id, messages)

    # ---- the band, in microns -------------------------------------- #
    from scipy import ndimage as ndi

    sampling = tuple(float(s) if float(s) > 0 else 1.0 for s in spacing)
    band_um = float(max(band_um, 0.0))
    dist_out = ndi.distance_transform_edt(~cur, sampling=sampling)
    dist_in = ndi.distance_transform_edt(cur, sampling=sampling)
    band = ((~cur) & (dist_out <= band_um)) | (cur & (dist_in <= band_um))

    n_border_bg, n_border_held = _border_face_counts(cur, labels_roi)

    others = np.unique(labels_roi[(labels_roi != 0) & (labels_roi != label_id)])
    n_neighbours = int(others.size)

    counts = dict(
        n_border_background=n_border_bg,
        n_border_held=n_border_held,
        n_band_voxels=int(band.sum()),
        n_neighbour_markers=n_neighbours,
    )

    # ---- the size guard, BEFORE any dense allocation ---------------- #
    region_voxels = int(labels_roi.size)
    if region_voxels * (n_neighbours + 2) > int(size_budget):
        messages.append(
            "This object's region is too large to settle ({} voxels against "
            "{} neighbour(s)).  Lower 'Edge may move', or pick a smaller "
            "object.".format(region_voxels, n_neighbours)
        )
        return _refused(labels_roi, label_id, messages, **counts)

    if str(solver) == "label_diffusion" and not have_usegment3d():
        messages.append(
            USEGMENT3D_MISSING.format(
                what="settling with label diffusion",
                fallback="nothing was changed",
            )
        )
        return _refused(labels_roi, label_id, messages, **counts)

    # ---- the markers: this object's core, every neighbour's, the
    #      background's.  The band itself is unseeded -- it is what is
    #      being decided.
    markers = np.zeros(labels_roi.shape, np.int32)
    markers[cur & ~band] = 1
    next_id = 2
    for other in others:
        core = (labels_roi == int(other)) & ~band
        if core.any():
            markers[core] = next_id
            next_id += 1
    bg_core = (labels_roi == 0) & ~band
    if bg_core.any():
        markers[bg_core] = next_id
        bg_marker = next_id
        next_id += 1
    else:
        bg_marker = 0

    if not (markers == 1).any():
        # The only way the object can be emptied: the band swallows it
        # whole, so there is no core to hold it and nothing would be left.
        # (Once a core exists it is marker 1 and always wins itself, which
        # is why the post-solve guard below is belt-and-braces.)
        messages.append(
            "'Edge may move' ({:.1f} um) is bigger than object {}, so "
            "nothing would be left of it -- nothing was changed.  Lower it "
            "and try again.".format(band_um, label_id)
        )
        return _refused(labels_roi, label_id, messages, **counts)
    if bg_marker == 0:
        messages.append(
            "There is no background left around this object at that "
            "distance, so nothing could move."
        )
        return _refused(labels_roi, label_id, messages, **counts)

    # ---- the solve --------------------------------------------------- #
    region = np.ones(labels_roi.shape, bool)
    try:
        if str(solver) == "label_diffusion":
            from skimage.segmentation import watershed

            dense = watershed(elev, markers=markers, mask=region,
                              compactness=float(compactness))
            solved = _diffuse(dense, elev, region, clamp=clamp,
                              n_iter=n_iter, sink_messages=messages)
        else:
            solved = _solve_partition(
                elev, markers, region, solver=str(solver),
                beta=float(beta), compactness=float(compactness),
                spacing=sampling,
            )
    except Exception as exc:  # noqa: BLE001 - a refusal, never a crash
        messages.append(
            "The settle could not be solved ({!r}), so nothing was "
            "changed.".format(exc)
        )
        return _refused(labels_roi, label_id, messages, **counts)

    won = np.asarray(solved) == 1

    # ---- the accept rule --------------------------------------------- #
    grow = band & ~cur & (labels_roi == 0) & won
    # A shrink is only accepted where the BACKGROUND marker won the voxel,
    # not merely where this object lost it.  Without that clause a voxel
    # claimed by a NEIGHBOUR's core would be released to background, which
    # moves a shared border -- and the status line promises the opposite
    # ("Settle only moves edges that face background" state 9).
    # It is also the half of "background <-> this object" that is easy to
    # drop: "no other label loses a voxel" is about growth, this is about
    # shrinkage.
    shrink = band & cur & (np.asarray(solved) == bg_marker)
    result_mask = (cur | grow) & ~shrink

    if not result_mask.any():
        messages.append(
            "Settling would leave nothing of object {} -- nothing would be "
            "left, so nothing was changed.  Lower 'Edge may move'."
            .format(label_id)
        )
        return _refused(labels_roi, label_id, messages, **counts)

    # Keep only the piece(s) still touching the original footprint, so a
    # shrink cannot strand an island of the object on the far side of a
    # wall.  Every component that touches `cur` is kept: an object that
    # arrived here already in two pieces must not silently lose one.
    lab, n_comp = ndi.label(result_mask)
    if n_comp > 1:
        keep = set(int(v) for v in np.unique(lab[cur & result_mask])) - {0}
        if keep:
            result_mask = np.isin(lab, sorted(keep))
            grow = grow & result_mask
            shrink = shrink | (cur & ~result_mask)

    if not result_mask.any():
        messages.append(
            "Settling would leave nothing of object {} -- nothing would be "
            "left, so nothing was changed.".format(label_id)
        )
        return _refused(labels_roi, label_id, messages, **counts)

    result = np.zeros_like(labels_roi)
    result[result_mask] = label_id

    n_grown = int(grow.sum())
    n_shrunk = int((cur & ~result_mask).sum())
    n_modified = n_grown + n_shrunk

    moves = []
    if n_grown:
        moves.append(dist_out[grow])
    if n_shrunk:
        moves.append(dist_in[cur & ~result_mask])
    if moves:
        allmoves = np.concatenate([m.ravel() for m in moves])
        mean_move = float(allmoves.mean())
        max_move = float(allmoves.max())
    else:
        mean_move = max_move = 0.0

    status = "ok" if n_modified else "unchanged"
    if not n_modified:
        messages.append(
            "Nothing moved: this object's outline already touches its "
            "neighbours everywhere, and Settle only moves edges that face "
            "background."
            if n_border_bg == 0 else
            "Nothing moved: the wall signal already runs along this "
            "object's outline."
        )

    return SettleResult(
        result_roi=result,
        n_modified=n_modified,
        n_grown=n_grown,
        n_shrunk=n_shrunk,
        mean_move_um=mean_move,
        max_move_um=max_move,
        hit_band_limit=bool(n_modified and max_move >= band_um - 1e-6),
        status=status,
        messages=messages,
        **counts,
    )


def settle_with_walls(
    labels_roi: np.ndarray,
    label_id: int,
    *,
    wall_kind: str = "membrane",
    channel_roi: np.ndarray | None = None,
    prob_roi: np.ndarray | None = None,
    prep_chain: str = "none",
    spacing=(1.0, 1.0, 1.0),
    progress=None,
    **settle_kw,
) -> SettleResult:
    """Build the wall picture, then settle -- ALL OF IT on the worker.

    The Settle twin of ``refine.contour_with_walls`` (REFINE B3; the contour
    moved first, B2 audit C-M3).  The picture used to be built on the main
    thread before the worker started: 1.47 s of frozen GUI on the owner's
    patch, before the busy strip had even appeared.  The orchestrator now
    hands over only the CROPPED inputs (the channel crop, the probability
    crop) and the decisions (wall kind, prep chain), and this function does
    the arithmetic the main-thread path did, line for line:
    ``apply_preproc`` + ``walls.build_elevation`` at its default smoothing,
    then :func:`settle_label` with the same ``spacing``.

    The picture comes back on ``SettleResult.elevation`` for the walls tick.
    ``progress("elevation", 0, 0)`` is emitted first and
    ``progress("settle", 0, 0)`` before the solve, so the strip names the
    stage.  A missing input raises -- the orchestrator checks both before
    it launches, so reaching one here is a programming error, and the
    worker turns it into its ``error`` signal.
    """
    from .preproc import apply_preproc
    from .walls import build_elevation

    region = np.asarray(labels_roi) == int(label_id)
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
        elevation, _c, _r = build_elevation(
            kind, prepared, region, spacing=spacing)
    if progress is not None:
        progress("settle", 0, 0)
    res = settle_label(labels_roi, elevation, label_id, spacing=spacing,
                       **settle_kw)
    res.elevation = elevation
    return res


def apply_settle_to_volume(
    volume: np.ndarray,
    roi,
    result_roi: np.ndarray,
    label_id: int,
) -> int:
    """Write a settled result back, or **refuse and write nothing**.

    Returns the number of voxels changed, or **-1** when the result would
    leave nothing of the object.  The refusal is here as
    well as in :func:`settle_label` because this is the last gate before the
    volume changes, and R1c -- "an empty step result erases the label" -- is
    exactly the failure that happens when only the caller checks.

    Background-only write, like every other curation write-back: a voxel that
    belongs to another label is never taken.
    """
    label_id = int(label_id)
    result_roi = np.asarray(result_roi)
    keep = result_roi == label_id
    if not keep.any():
        return -1

    sub = volume[roi]
    before = sub == label_id
    writable = (sub == 0) | before
    after = keep & writable
    if not after.any():
        return -1

    sub[before & ~after] = 0
    sub[after] = label_id
    volume[roi] = sub
    return int(np.count_nonzero(before != after))

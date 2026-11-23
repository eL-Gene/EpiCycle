"""The markers' search on a whole patch, one click at a time.

Pure algorithm module -- no Qt, no napari.  Heavy dependencies (scipy,
skimage) are imported lazily inside functions.

:func:`add_segment.segment_at_markers` answers a list of clicks in one go:
it thresholds the box, labels its bright pieces, and floods every clicked
piece from its markers.  On a whole patch that costs seconds per click.
This module splits the same search in two:

* :func:`prepare_markers_box` -- once per stage: the threshold space (Local
  contrast or not), the bright voxels, their 6-connected pieces and each
  piece's box.  The stage is read-only; it depends on the image, the kept
  objects, Local contrast and its clip, the threshold, the click radius, the
  split mode, the markers' Spacing, Smallest object and the voxel spacing,
  and on nothing else.
* :func:`solve_markers` -- per click: which piece each click claims, and the
  objects of the pieces whose markers changed.  A piece is solved again
  only when its own markers changed: a click added to it, removed from it,
  moved within or into or out of it, or a marker reshaped because an
  earlier click's ball (which keeps the voxels two balls share) came or
  went.  Every other piece keeps its objects.  A click's claim is looked up
  by its point and the earlier balls its box meets, so only new or
  reshaped claims are computed; every object's smoothed shape is kept (the
  first solve keeps them all), so only the shapes whose voxels changed are
  smoothed again.  What stays costly is the answer itself: a click on a
  very large piece (about 100,000 voxels) floods and smooths that piece,
  about 1.5 s on a 91 x 240 x 240 patch.

What is exact, in plain words:

* With the split off and ONE marker in each clicked piece the answer is the
  one :func:`segment_at_markers` gives (then numbered by ``ids`` and
  smoothed as :func:`add_segment.smooth_pieces` smooths it), voxel for
  voxel: one marker floods its whole piece, whatever the other pieces hold.
* Otherwise -- two markers or more in one piece, or a split mode that cuts
  pieces at their waists -- each clicked piece is solved on its own markers
  only (its clicks' balls as the whole click list draws them, an earlier
  ball keeping a shared voxel).  That equals :func:`segment_at_markers` run
  on that piece's clicks alone, but it can differ from ONE run over every
  click by a few voxels inside such a piece.  The cause is the flood (the
  ``skimage`` watershed): it starts every seed with the same age, so
  between two seeds of equal height the one that goes first is set by the
  layout of its queue, and that layout depends on every other seed in the
  same call -- the seeds of the other pieces too.  A run over all clicks
  and a run over one piece's clicks can therefore hand a tie voxel to
  different objects.  Measured on synthetic touching nuclei (14 clicks,
  four or five pieces holding two or three of them): 36 click lists in 80
  differed somewhere, by 1 to 79 voxels; a split mode differed by up to
  about 120 voxels in a piece.  The answer here never depends on the order
  the clicks were made in, only on the click list: the same list always
  gives the same voxels.  On the Add page this search runs for the Nuclei
  markers on a whole patch with "Update as I click" on, and the one-go
  search with it off -- so turning the tick on or off can move those few
  tie voxels inside a nucleus holding several seeds.
* The smoothing is exact in every case.  Each object's smoothed shape
  depends only on its own unsmoothed voxels, and the shapes are written in
  ascending id order, each into empty voxels only, so a voxel's final id
  depends only on its unsmoothed id and on which smoothed shapes reach it.
  A change is therefore redone over the reach of every object whose
  unsmoothed voxels changed (its old and its new box, grown by the
  smoothing's margin), with every object whose own grown box meets it,
  and nowhere else.

Several seeds, one object, as :func:`segment_at_markers` makes it with
``groups``: ``groups`` gives each click a group (default: each click its
own).  The seeds of one group share one id and are ONE marker label -- the
union of their balls inside the group's piece -- so they never compete for
the voxels between them, and each later seed also climbs to its own hill.
A group lives in the bright piece of its earliest seed that claims one; a
later seed of the group on another piece is refused (``"apart"``) and the
search runs as if it had never been placed.  So is a later seed none of
whose marker voxels ends in its group's object (another object kept the
way between them): every piece is flooded, the lost seeds are refused, the
claims are made again without them and the pieces whose claims changed are
flooded again, until no seed is lost -- the re-run
:func:`segment_at_markers` makes, by its own rule
(:func:`add_segment._lost_seeds`).  A cut piece's group takes the basins
between its seeds as there too (the same
:func:`add_segment._split_clicked_pieces`).  With the split off and ONE
group in each clicked piece the answer is exact, as for one marker.

No tissue outline: the whole-patch Nuclei search never has one (a whole
patch is cut by nothing), and nothing here clips to one;
:func:`prepare_markers_box` refuses an ``outline``.

The caller owns what is drawn: a solve never writes into the arrays of the
solution it was given (it copies them), nor into the stage, and a solution
and its arrays cannot be changed after it is returned.
"""

from __future__ import annotations

import time
from collections import ChainMap
from dataclasses import dataclass, field

import numpy as np

from .add_segment import (
    MARKER_SPLIT_MODES,
    _claim_piece,
    _gain_tops,
    _guard_markers,
    _lost_seeds,
    _marker_crop_margin,
    _marker_foreground,
    _marker_half_widths,
    _poll_stop,
    _split_clicked_pieces,
)


@dataclass(frozen=True, eq=False)
class MarkerStage:
    """The prepared box: read-only arrays and the settings they were made
    with.  Built by :func:`prepare_markers_box`; never changed after."""

    shape: tuple
    spacing: tuple
    radius_um: float
    split: str
    n_vox: object
    half: object
    min_size: int
    empty: np.ndarray
    fg: np.ndarray
    comp: np.ndarray
    piece_boxes: tuple
    margin: tuple
    ball_half: tuple


@dataclass(frozen=True, eq=False)
class _PieceResult:
    """One clicked piece's objects: ``local`` numbers the piece's marker
    labels ``1..m`` (one per id, in the order of each id's first seed) over
    the box ``crop``; ``ids`` the final id of each, ``small`` the local
    numbers Smallest object dropped; ``lost`` the positions (in the
    piece's claims) of the later seeds none of whose marker voxels ended in
    their object -- a piece with a lost seed is never the answer, only a
    step towards it."""

    crop: tuple
    local: np.ndarray
    ids: tuple
    small: frozenset
    lost: tuple = ()


@dataclass(frozen=True, eq=False)
class MarkerSolution:
    """One answer of :func:`solve_markers` (frozen; its arrays are
    read-only).

    ``raw`` -- the objects before the smoothing, numbered by the click ids;
    ``result`` -- what is drawn (``raw`` smoothed, or ``raw`` itself when
    the smoothing is off); ``why`` -- ``{g: "kept" | "dim" | "small"}`` for
    every group that made nothing, ``g`` numbering the groups ``1..`` in the
    order each first appears (with no groups: the click number);
    ``seed_why`` -- ``{k: "kept" | "dim" | "apart"}`` for every click
    (1-based) that added nothing to its group; both exactly as
    :func:`segment_at_markers` reports them.  ``click_ids`` / ``groups`` --
    the id and the group of each click, as handed in.  ``sizes`` -- ``{id:
    voxels}`` of ``result``; ``raw_ids`` -- the ids present in ``raw``;
    :attr:`ids` -- the ids present in ``result``; :attr:`emptied_ids` --
    the ids the smoothing emptied.  The other fields are the solve's own
    caches, which the next solve reuses as they are."""

    stage: MarkerStage
    points: tuple
    groups: tuple
    click_ids: tuple
    smooth_sigma: object
    raw: np.ndarray
    result: np.ndarray
    why: dict
    seed_why: dict
    sizes: dict
    raw_ids: tuple
    _signatures: dict = field(repr=False)
    _pieces: dict = field(repr=False)
    _boxes: dict = field(repr=False)
    _smoothed: dict = field(repr=False)
    _claim_cache: tuple = field(repr=False, default=None)
    #: ``{(piece, signature): result}`` of the floods that lost a seed,
    #: met on the way to this answer
    _tried: dict = field(repr=False, default=None)
    #: the smoothing failed (``result`` is ``raw``): the next solve
    #: smooths everything again
    smooth_failed: bool = False

    @property
    def ids(self) -> tuple:
        """The ids present in ``result``, ascending."""
        return tuple(sorted(self.sizes))

    @property
    def emptied_ids(self) -> tuple:
        """The ids in ``raw`` the smoothing left with no voxel."""
        return tuple(sorted(set(self.raw_ids) - set(self.sizes)))


@dataclass(frozen=True)
class MarkerDiff:
    """What changed between the solution handed in and the new one.

    ``full`` -- True when there was nothing to compare with (no previous
    solution, or one made on another stage or with another smoothing): draw
    ``solution.result`` whole; ``patches`` is then empty.  Otherwise
    ``changed_ids`` lists every id whose drawn voxels changed (an object
    that appeared, went, or moved, and every neighbour the smoothing
    reshaped) and ``patches`` is a tuple of ``(box, patch)``, one per
    separate changed place (boxes never overlap; empty when nothing
    changed): ``patch`` is ``result[box]`` (a copy), so writing every patch
    over the drawn array gives the new answer.  ``raw_changed_ids``: the
    ids whose unsmoothed voxels changed; ``n_pieces_solved``: how many
    pieces were flooded again; ``ms``: the time of each part of the solve,
    in milliseconds."""

    full: bool
    changed_ids: tuple
    patches: tuple
    raw_changed_ids: tuple
    n_pieces_solved: int
    ms: dict


def prepare_markers_box(image_roi, existing_roi, *, radius_um, spacing,
                        clahe=False, clahe_clip=0.01, threshold="otsu",
                        min_size=0, split="off", peak_spacing_um=None,
                        cancel_event=None, outline=None) -> MarkerStage:
    """The stage of the markers' search, computed once: the bright voxels
    of the box (the threshold and the Local contrast exactly as
    :func:`segment_at_markers` computes them), their 6-connected pieces and
    each piece's box.  The arguments mean what they mean there.
    ``outline`` must be ``None``: nothing here cuts to a tissue outline, so
    a caller that has one must not come here."""
    if outline is not None:
        raise ValueError("the markers' incremental search takes no tissue "
                         "outline")
    if split not in MARKER_SPLIT_MODES:
        raise ValueError(f"split must be one of {MARKER_SPLIT_MODES}, "
                         f"got {split!r}")
    from scipy import ndimage as ndi

    image = np.asarray(image_roi, dtype=np.float64)
    existing = np.asarray(existing_roi)
    if image.shape != existing.shape:
        raise ValueError(
            f"image_roi shape {image.shape} != existing_roi shape "
            f"{existing.shape}"
        )
    if image.ndim != 3:
        raise ValueError("markers need a 3D box")
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    spacing = tuple(float(v) for v in spacing)
    n_vox, half = None, None
    if split != "off":
        n_vox, half = _marker_half_widths(peak_spacing_um, spacing)
    _poll_stop(cancel_event)
    empty, fg = _marker_foreground(image, existing, clahe=clahe,
                                   clahe_clip=clahe_clip,
                                   threshold=threshold)
    _poll_stop(cancel_event)
    comp, _n = ndi.label(fg)
    boxes = tuple(ndi.find_objects(comp))
    for a in (empty, fg, comp):
        a.flags.writeable = False
    r = float(radius_um)
    return MarkerStage(
        shape=tuple(int(n) for n in existing.shape),
        spacing=spacing,
        radius_um=r,
        split=split,
        n_vox=n_vox,
        half=half,
        min_size=int(min_size),
        empty=empty,
        fg=fg,
        comp=comp,
        piece_boxes=boxes,
        margin=_marker_crop_margin(spacing),
        ball_half=tuple(int(np.ceil(r / s)) for s in spacing),
    )


def _raw_ball(stage, point):
    """``(box, ball)`` of one click, before any earlier ball takes its
    voxels -- the ball :func:`reshape.region.rasterise_points` draws, over
    its own box (``None`` when the box lies outside the stage)."""
    shape = stage.shape
    sp = stage.spacing
    r = stage.radius_um
    half = stage.ball_half
    p = np.asarray(point, float)[-3:]
    c = np.rint(p).astype(int)
    box = tuple(slice(max(0, c[k] - half[k]),
                      min(shape[k], c[k] + half[k] + 1)) for k in range(3))
    if any(b.start >= b.stop for b in box):
        return None
    grids = np.ogrid[box]
    d2 = sum(((g - p[k]) * sp[k]) ** 2 for k, g in enumerate(grids))
    ball = d2 <= r * r + 1e-9
    if all(0 <= c[k] < shape[k] for k in range(3)):
        ball[tuple(c[k] - box[k].start for k in range(3))] = True
    return box, ball


def _overlap(a, b):
    """The part of box ``b`` inside box ``a``, in ``a``'s and ``b``'s own
    coordinates, or ``None`` when they do not meet."""
    lo = [max(x.start, y.start) for x, y in zip(a, b)]
    hi = [min(x.stop, y.stop) for x, y in zip(a, b)]
    if any(l >= h for l, h in zip(lo, hi)):
        return None
    in_a = tuple(slice(l - x.start, h - x.start)
                 for l, h, x in zip(lo, hi, a))
    in_b = tuple(slice(l - y.start, h - y.start)
                 for l, h, y in zip(lo, hi, b))
    return in_a, in_b


def _point_key(point):
    """What a click's ball and claim depend on: its last three coordinates
    as numbers."""
    return tuple(float(v) for v in tuple(point)[-3:])


def _claim_one(stage, key, ball_box, earlier):
    """The claim of one click (see :func:`_claims`) from its own ball and
    the balls of the earlier clicks whose boxes meet its box."""
    box, ball = ball_box
    ball = ball.copy()
    for ebox, eball in earlier:
        ov = _overlap(box, ebox)
        if ov is not None:
            ball[ov[0]] &= ~eball[ov[1]]
    if not ball.any():
        return (0, "kept", None, None, None, None)
    own = tuple(int(round(v)) for v in key)
    if not (all(0 <= own[a] < stage.shape[a] for a in range(3))
            and ball[tuple(own[a] - box[a].start for a in range(3))]):
        own = None
    c, reason = _claim_piece(ball, box, own, stage.empty, stage.fg,
                             stage.comp)
    if reason is not None:
        return (0, reason, None, None, None, None)
    marker = ball & (stage.comp[box] == c)
    if own is not None and not stage.fg[own]:
        own = None
    vox = np.argwhere(marker) + np.asarray([b.start for b in box])
    return (c, None, box, marker, own, vox.tobytes())


_KEPT = (0, "kept", None, None, None, None)


def _claims(stage, points, cache=None):
    """Per click (in order): ``(piece, why, box, marker, own, voxels)`` --
    the piece it claims (0 when it claims none), ``"kept"`` / ``"dim"`` /
    ``None``, and for a claiming click its marker (its ball inside the
    piece) over ``box``, its own voxel when it names the piece (else
    ``None``) and the marker's voxels as bytes.  The balls are
    :func:`segment_at_markers`' balls: an earlier ball keeps a shared voxel.

    A click's claim depends only on its own point and on the balls of the
    earlier clicks whose boxes meet its box, so it is looked up by exactly
    that (``cache``, from the last solve on this stage) and computed only
    when new.  The earlier clicks are found through a grid of cells one
    ball box wide: two boxes can meet only when their centres lie in the
    same or neighbouring cells.  Returns ``(claims, cache)``."""
    old_balls, old_claims = cache if cache is not None else ({}, {})
    cell = tuple(2 * h + 1 for h in stage.ball_half)
    keys = [_point_key(p) for p in points]
    balls, claims, grid, out = {}, {}, {}, []
    near = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
            for c in (-1, 0, 1)]
    for key in keys:
        if key not in balls:
            balls[key] = (old_balls[key] if key in old_balls
                          else _raw_ball(stage, key))
    for key in keys:
        rb = balls[key]
        if rb is None:
            out.append(_KEPT)
            continue
        centre = tuple(int(np.rint(v)) // n for v, n in zip(key, cell))
        earlier = set()
        for d in near:
            for other in grid.get(tuple(x + y for x, y in zip(centre, d)),
                                  ()):
                if _overlap(rb[0], balls[other][0]) is not None:
                    earlier.add(other)
        ck = (key, frozenset(earlier))
        if ck not in claims:
            claims[ck] = (old_claims[ck] if ck in old_claims else
                          _claim_one(stage, key, rb,
                                     [balls[e] for e in sorted(earlier)]))
        out.append(claims[ck])
        grid.setdefault(centre, []).append(key)
    return out, (balls, claims)


def _signature(entries):
    """What a piece's objects depend on: its claims in order (final id, the
    marker's voxels, the click's own voxel)."""
    return tuple((int(ident), vox, own)
                 for _k, ident, _box, _marker, own, vox in entries)


def _solve_piece(stage, c, entries, cancel_event):
    """The objects of piece ``c`` from its own claims ``entries`` (``(k,
    id, box, marker, own, voxels)`` in placement order), as
    :func:`segment_at_markers` gives them for these clicks alone.  The seeds
    sharing an id are ONE marker label (numbered by its first seed); the
    first seed's own voxel starts its climb, and each later seed climbs
    too, from its own voxel (or its ball's highest voxel)."""
    from scipy import ndimage as ndi

    pb = stage.piece_boxes[c - 1]
    crop = tuple(slice(max(0, s.start - m), min(n, s.stop + m))
                 for s, m, n in zip(pb, stage.margin, stage.shape))
    origin = np.asarray([s.start for s in crop])
    shape = tuple(s.stop - s.start for s in crop)
    labels = {}
    markers = np.zeros(shape, np.int32)
    claimed, clicked, extra = {}, {}, {}
    later = []           # the later seeds as (position, label, voxels)
    for pos, (_k, ident, box, marker, own, _vox) in enumerate(entries):
        head = int(ident) not in labels
        j = labels.setdefault(int(ident), len(labels) + 1)
        vox = np.argwhere(marker) + np.asarray([b.start for b in box]) - origin
        markers[tuple(vox.T)] = j
        local_own = (None if own is None
                     else tuple(int(v) for v in np.asarray(own) - origin))
        if head:
            claimed[j] = c
            if local_own is not None:
                clicked[j] = local_own
        else:
            extra.setdefault(j, []).append((local_own, vox))
            later.append((pos, j, vox))
    extra = extra or None
    m = len(labels)
    fg = stage.fg[crop]
    comp = stage.comp[crop]
    if stage.split != "off":
        local = _split_clicked_pieces(fg, comp, markers, claimed, clicked, m,
                                      stage.spacing, stage.split,
                                      stage.n_vox, stage.half, cancel_event,
                                      extra=extra)
    else:
        # the markers' own flood, as segment_at_markers floods with the
        # split off
        mask = fg & (comp == c)
        edt = ndi.distance_transform_edt(mask, sampling=stage.spacing)
        _poll_stop(cancel_event)
        seeds = np.array(markers, dtype=np.int32, copy=True)
        found, anchors = _gain_tops(
            edt, comp, seeds, claimed, ndi.find_objects(seeds, max_label=m),
            clicked, cancel_event, extra=extra)
        gained = {}
        for k, vox in found.items():
            new = vox[seeds[tuple(vox.T)] == 0]
            if len(new):
                seeds[tuple(new.T)] = k
                gained[k] = new
        _poll_stop(cancel_event)
        local = _guard_markers(edt, seeds, mask, gained, anchors,
                               set(range(1, m + 1)), cancel_event)
    local = np.array(local, dtype=np.int32, copy=True)
    lost = tuple(sorted(_lost_seeds(local, later)))
    small = set()
    if stage.min_size:
        sizes = np.bincount(local.ravel(), minlength=m + 1)
        for j in range(1, m + 1):
            if int(sizes[j]) < int(stage.min_size):
                local[local == j] = 0
                small.add(j)
    local.flags.writeable = False
    return _PieceResult(crop=crop, local=local, ids=tuple(labels),
                        small=frozenset(small), lost=lost)


def _footprints(rec):
    """``{id: (box, mask)}`` of the objects of one piece, in the stage's
    coordinates (an id with no voxel left has no key)."""
    from scipy import ndimage as ndi

    out = {}
    m = len(rec.ids)
    if not m:
        return out
    origin = [s.start for s in rec.crop]
    for j, sl in enumerate(ndi.find_objects(rec.local, max_label=m),
                           start=1):
        if sl is None:
            continue
        box = tuple(slice(s.start + o, s.stop + o)
                    for s, o in zip(sl, origin))
        out[rec.ids[j - 1]] = (box, rec.local[sl] == j)
    return out


def _paste(raw, rec, *, clear):
    """Write (or, with ``clear``, erase) one piece's objects into ``raw``."""
    view = raw[rec.crop]
    on = rec.local != 0
    if clear:
        view[on] = 0
        return
    lut = np.zeros(len(rec.ids) + 1, raw.dtype)
    lut[1:] = rec.ids
    view[on] = lut[rec.local[on]]


def _smooth_recipe(sigma):
    from .batch_refine import RefineStep

    return [RefineStep(method="sdf",
                       params={"sigma_z": float(sigma),
                               "sigma_xy": float(sigma)})]


def _smooth_one(raw, ident, margin, recipe, bbox):
    """``(roi, mask)``: one object's smoothed shape over its grown box, as
    the batch smoothing computes it."""
    from .batch_refine import _compute_one

    _lid, roi, res, _n = _compute_one(raw, ident, margin, recipe, bbox=bbox)
    return roi, None if roi is None else res == ident


def _smooth_all(raw, boxes, sigma, cancel_event):
    """``(result, smoothed, failed)``: every object of ``raw`` smoothed, as
    :func:`add_segment.smooth_pieces` smooths them -- each shape computed
    on its own (on :func:`add_segment.smooth_worker_count` threads), then
    written in ascending id order, the old voxels cleared and the new ones
    written into empty voxels only -- with each object's smoothed shape
    kept for the next solve.  A shape that fails leaves everything
    unsmoothed, as there (``failed``)."""
    from concurrent.futures import ThreadPoolExecutor

    from .add_segment import smooth_worker_count
    from .batch_refine import _recipe_margin

    order = sorted(boxes)
    if not order:
        return raw, {}, False
    recipe = _smooth_recipe(sigma)
    margin = _recipe_margin(recipe, raw.shape)
    smoothed = {}
    try:
        with ThreadPoolExecutor(max_workers=smooth_worker_count()) as pool:
            futs = {i: pool.submit(_smooth_one, raw, i, margin, recipe,
                                   boxes[i]) for i in order}
            for i in order:
                if cancel_event is not None and cancel_event.is_set():
                    for f in futs.values():
                        f.cancel()
                    break
                smoothed[i] = futs[i].result()
    except Exception as exc:
        print(f"[ADD] smooth failed: {exc!r}")
        return raw, {}, True
    _poll_stop(cancel_event)
    work = raw.copy()
    for i in order:
        roi, smask = smoothed[i]
        if roi is None:
            continue
        t = work[roi]
        t[t == i] = 0
        t[smask & (t == 0)] = i
    return work, smoothed, False


def _grow(box, margin, shape):
    return tuple(slice(max(0, s.start - m), min(n, s.stop + m))
                 for s, m, n in zip(box, margin, shape))


def _result_dtype(ids):
    top = max([int(i) for i in ids] or [0])
    return np.int32 if top <= np.iinfo(np.int32).max else np.int64


def _meets(a, b):
    return all(x.start < y.stop and y.start < x.stop for x, y in zip(a, b))


def _merge_boxes(boxes):
    """The boxes, those that share a voxel joined into their bounding box
    until none does: the same voxels covered, none covered twice."""
    out = [tuple(b) for b in boxes]
    joined = True
    while joined:
        joined = False
        merged = []
        for b in out:
            for n, o in enumerate(merged):
                if _meets(b, o):
                    merged[n] = tuple(slice(min(x.start, y.start),
                                            max(x.stop, y.stop))
                                      for x, y in zip(b, o))
                    joined = True
                    break
            else:
                merged.append(b)
        out = merged
    return out


#: ids up to this are counted with one ``np.bincount``, larger by sorting
_BINCOUNT_MAX_ID = 1 << 24


def _count_ids(values):
    """``{id: voxels}`` of the non-zero values of ``values``."""
    v = np.asarray(values).ravel()
    if not v.size:
        return {}
    top = int(v.max())
    if 0 <= int(v.min()) and top <= _BINCOUNT_MAX_ID:
        counts = np.bincount(v.astype(np.int64, copy=False))
        nz = np.flatnonzero(counts)
        return {int(i): int(counts[i]) for i in nz if i}
    vals, counts = np.unique(v, return_counts=True)
    return {int(a): int(b) for a, b in zip(vals, counts) if a}


def _click_ids(n, ids, groups):
    """``(ids, groups)`` per click, checked: one id per group, distinct
    between groups, > 0.  Default groups: each click its own; default ids:
    ``1..`` per group in the order of each group's first click."""
    if groups is None:
        groups = list(range(n))
    groups = list(groups)
    if len(groups) != n:
        raise ValueError(f"{len(groups)} groups for {n} points")
    if ids is None:
        rank = {}
        for g in groups:
            rank.setdefault(g, len(rank) + 1)
        ids = [rank[g] for g in groups]
    ids = [int(i) for i in ids]
    if len(ids) != n:
        raise ValueError(f"{len(ids)} ids for {n} points")
    by_group = {}
    for g, i in zip(groups, ids):
        if by_group.setdefault(g, i) != i:
            raise ValueError(f"group {g!r} has two ids")
    if len(set(by_group.values())) != len(by_group):
        raise ValueError("two groups share an id")
    if any(i <= 0 for i in ids):
        raise ValueError("ids must be > 0")
    return ids, groups


def solve_markers(stage, points, ids=None, *, groups=None,
                  smooth_sigma=None, previous=None, cancel_event=None):
    """The objects of the clicks ``points`` on ``stage``.

    ``points`` -- box-local ZYX, the ticked markers in placement order;
    ``ids`` -- the final id of each click (default ``1..``, the numbering
    of :func:`segment_at_markers` when there are no groups); ``groups`` --
    a group per click (default each click its own): the clicks of one group
    carry the same id and make one object; ``smooth_sigma`` -- the
    smoothing's sigma (``None``: no smoothing).  ``previous`` -- the last
    solution on this stage: only the pieces whose markers changed are
    flooded again and only the smoothing's reach around them is redrawn.
    Without it, or when it was made on another stage or with another
    smoothing, every clicked piece is solved.

    Returns ``(solution, diff)`` (:class:`MarkerSolution`,
    :class:`MarkerDiff`).  Stop (``cancel_event``) is polled between the
    pieces and inside each flood; a Stop raises ``ReshapeCancelled``.
    """
    t0 = time.perf_counter()
    ms = {}
    pts = [tuple(p) for p in (points or ())]
    ids, groups = _click_ids(len(pts), ids, groups)
    sigma = None if smooth_sigma is None else float(smooth_sigma)
    same_stage = previous is not None and previous.stage is stage
    full = (not same_stage or previous.smooth_sigma != sigma
            or previous.smooth_failed)
    dtype = _result_dtype(ids)
    if not full and previous.raw.dtype != dtype:
        full = True
    _poll_stop(cancel_event)

    # --- which piece each click claims, and each piece's objects ---
    first = {}
    gidx = [first.setdefault(g, len(first) + 1) for g in groups]
    # the claims met in this solve (with and without the refused seeds),
    # looked up first here, then in the last solve's
    old_balls, old_claims = (previous._claim_cache if same_stage
                             and previous._claim_cache is not None
                             else ({}, {}))
    acc_balls, acc_claims = {}, {}
    lookup = (ChainMap(acc_balls, old_balls), ChainMap(acc_claims, old_claims))
    old_sigs = {} if full else previous._signatures
    old_pieces = {} if full else previous._pieces
    old_tried = (previous._tried or {}) if not full else {}
    tried = {}           # the floods that lost a seed, kept for the next
    met = {}             # every flood of this solve, by its claims
    n_solved = 0
    t_claims = t_flood = 0.0
    skip = set()
    while True:
        # the refused seeds (apart, lost) are as if never placed
        tc = time.perf_counter()
        apart = set(skip)
        while True:
            # a group's piece is its earliest claiming seed's; a later seed
            # on another piece is refused and the claims are made again
            # without it (its ball may have kept a voxel a later ball needs)
            live = [k for k in range(1, len(pts) + 1) if k not in apart]
            got, cache = _claims(stage, [pts[k - 1] for k in live], lookup)
            acc_balls.update(cache[0])
            acc_claims.update(cache[1])
            claim_of = dict(zip(live, got))
            head, new = {}, set()
            for k in live:
                if claim_of[k][1] is not None:
                    continue
                g = gidx[k - 1]
                if g not in head:
                    head[g] = k
                elif claim_of[k][0] != claim_of[head[g]][0]:
                    new.add(k)
            if not new:
                break
            apart |= new
        seed_why = {k: "apart" for k in apart}
        by_piece = {}
        for k in live:
            c, reason, box, marker, own, vox = claim_of[k]
            if reason is not None:
                seed_why[k] = reason
                continue
            by_piece.setdefault(c, []).append(
                (k, ids[k - 1], box, marker, own, vox))
        sigs = {c: _signature(e) for c, e in by_piece.items()}
        t_claims += time.perf_counter() - tc

        # each claimed piece's objects, flooded again only when its claims
        # are new (a flood that lost a seed is kept too: the next solve
        # meets the same claims before it refuses that seed again)
        tf = time.perf_counter()
        pieces = {}
        lost = set()
        for c in sorted(by_piece):
            _poll_stop(cancel_event)
            key = (c, sigs[c])
            if key in met:
                rec = met[key]
            elif key in old_tried:
                rec = old_tried[key]
            elif c in old_pieces and old_sigs.get(c) == sigs[c]:
                rec = old_pieces[c]
            else:
                rec = _solve_piece(stage, c, by_piece[c], cancel_event)
                n_solved += 1
            met[key] = rec
            if rec.lost:
                tried[key] = rec
                lost.update(by_piece[c][n][0] for n in rec.lost)
            pieces[c] = rec
        t_flood += time.perf_counter() - tf
        if not lost:
            break
        # the lost seeds are refused, and everything is made again as if
        # they had never been placed, as segment_at_markers makes it again
        skip = apart | lost
    seed_why = dict(sorted(seed_why.items()))
    why = {}
    for k in range(len(pts), 0, -1):
        g = gidx[k - 1]
        if g not in head and k in seed_why:
            why[g] = seed_why[k]          # the first seed's reason wins
    group_of_id = {ids[k - 1]: gidx[k - 1] for k in range(1, len(pts) + 1)}
    ms["claims"] = t_claims * 1000.0

    # --- the pieces whose objects changed ---
    t1 = time.perf_counter()
    touched = sorted(c for c in set(sigs) | set(old_sigs)
                     if sigs.get(c) != old_sigs.get(c))
    for c in by_piece:
        rec = pieces[c]
        for j in rec.small:
            why[group_of_id[rec.ids[j - 1]]] = "small"
    why = dict(sorted(why.items()))
    ms["flood"] = (t_flood + time.perf_counter() - t1) * 1000.0

    # --- the unsmoothed answer ---
    t2 = time.perf_counter()
    if full:
        raw = np.zeros(stage.shape, dtype)
        for rec in pieces.values():
            _paste(raw, rec, clear=False)
        boxes = {}
        for rec in pieces.values():
            boxes.update({i: b for i, (b, _m) in _footprints(rec).items()})
        raw_changed = sorted(boxes)
        old_boxes = {}
    else:
        raw = previous.raw.copy()
        old_boxes = previous._boxes
        boxes = dict(old_boxes)
        old_fp, new_fp = {}, {}
        for c in touched:
            if c in old_pieces:
                _paste(raw, old_pieces[c], clear=True)
                old_fp.update(_footprints(old_pieces[c]))
        for c in touched:
            if c in pieces:
                _paste(raw, pieces[c], clear=False)
                new_fp.update(_footprints(pieces[c]))
        raw_changed = []
        for i in sorted(set(old_fp) | set(new_fp)):
            a, b = old_fp.get(i), new_fp.get(i)
            if (a is None or b is None or a[0] != b[0]
                    or not np.array_equal(a[1], b[1])):
                raw_changed.append(i)
        for i in raw_changed:
            boxes.pop(i, None)
            if i in new_fp:
                boxes[i] = new_fp[i][0]
    ms["raw"] = (time.perf_counter() - t2) * 1000.0

    # --- the smoothing ---
    t3 = time.perf_counter()
    smoothed = {}
    failed = False
    if sigma is None:
        result = raw
        regions = _merge_boxes(
            [old_boxes[i] for i in raw_changed if i in old_boxes]
            + [boxes[i] for i in raw_changed if i in boxes])
    elif full:
        result, smoothed, failed = _smooth_all(raw, boxes, sigma,
                                               cancel_event)
        regions = []
    else:
        from .batch_refine import _recipe_margin

        recipe = _smooth_recipe(sigma)
        margin = _recipe_margin(recipe, stage.shape)
        changed = set(raw_changed)
        smoothed = {i: v for i, v in previous._smoothed.items()
                    if i not in changed and i in boxes}
        result = previous.result.copy()
        regions = _merge_boxes(
            [_grow(old_boxes[i], margin, stage.shape)
             for i in raw_changed if i in old_boxes]
            + [_grow(boxes[i], margin, stage.shape)
               for i in raw_changed if i in boxes])
        order = sorted(boxes)
        if order:
            rois = [_grow(boxes[i], margin, stage.shape) for i in order]
            lo = np.asarray([[s.start for s in r] for r in rois])
            hi = np.asarray([[s.stop for s in r] for r in rois])
        for region in regions:
            _poll_stop(cancel_event)
            work = raw[region].copy()
            if order:
                rlo = np.asarray([s.start for s in region])
                rhi = np.asarray([s.stop for s in region])
                meet = np.flatnonzero(np.all((lo < rhi) & (hi > rlo), axis=1))
            else:
                meet = ()
            for n in meet:
                ident = order[int(n)]
                if ident not in smoothed:
                    try:
                        smoothed[ident] = _smooth_one(raw, ident, margin,
                                                      recipe, boxes[ident])
                    except Exception as exc:
                        print(f"[ADD] smooth failed: {exc!r}")
                        failed = True
                        break
                roi, smask = smoothed[ident]
                if roi is None:
                    continue
                ov = _overlap(region, roi)
                if ov is None:
                    continue
                w = work[ov[0]]
                w[w == ident] = 0
                w[smask[ov[1]] & (w == 0)] = ident
            if failed:
                break
            result[region] = work
        if failed:
            # as the smoothing of a whole run fails: nothing is smoothed
            result, smoothed, regions = raw, {}, []
            full = True
    ms["smooth"] = (time.perf_counter() - t3) * 1000.0

    # --- what the caller redraws, and the sizes ---
    t4 = time.perf_counter()
    patches = []
    if full:
        sizes = _count_ids(result)
        diff_ids = tuple(sorted(sizes))
    else:
        before = previous.result
        sizes = dict(previous.sizes)
        changed_ids = set()
        for region in regions:          # disjoint: nothing counted twice
            d = before[region] != result[region]
            if not d.any():
                continue
            gone = _count_ids(before[region][d])
            came = _count_ids(result[region][d])
            for i, n in gone.items():
                sizes[i] = sizes.get(i, 0) - n
            for i, n in came.items():
                sizes[i] = sizes.get(i, 0) + n
            changed_ids.update(gone)
            changed_ids.update(came)
            where = np.argwhere(d)
            start = np.asarray([s.start for s in region])
            lo_d = where.min(axis=0) + start
            hi_d = where.max(axis=0) + start + 1
            box = tuple(slice(int(a), int(b)) for a, b in zip(lo_d, hi_d))
            patches.append((box, result[box].copy()))
        sizes = {i: n for i, n in sizes.items() if n > 0}
        diff_ids = tuple(sorted(changed_ids))
    ms["diff"] = (time.perf_counter() - t4) * 1000.0
    ms["total"] = (time.perf_counter() - t0) * 1000.0

    raw.flags.writeable = False
    result.flags.writeable = False
    solution = MarkerSolution(
        stage=stage, points=tuple(pts), groups=tuple(groups),
        click_ids=tuple(ids), smooth_sigma=sigma, raw=raw, result=result,
        why=why, seed_why=seed_why, sizes=dict(sorted(sizes.items())),
        raw_ids=tuple(sorted(boxes)), _signatures=sigs, _pieces=pieces,
        _boxes=boxes, _smoothed=smoothed,
        _claim_cache=(acc_balls, acc_claims),
        _tried=tried, smooth_failed=failed)
    diff = MarkerDiff(full=full, changed_ids=diff_ids, patches=tuple(patches),
                      raw_changed_ids=tuple(raw_changed),
                      n_pieces_solved=n_solved,
                      ms=ms)
    return solution, diff

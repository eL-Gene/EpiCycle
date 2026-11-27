"""Cells ▸ Click markers, one click at a time: the fill to the walls redone
only where a click can change it.

The fill to the walls (``growth="flood"`` of :mod:`.region`) is one seeded
watershed over the box: the click balls, the kept objects' cores and the
outside (the box faces, the eroded dark, everything outside the tissue
outline) flood the wall picture together, and every voxel goes to the
marker whose cheapest climb reaches it first.  A new click can only take
voxels from its neighbours; a removed click can only hand its voxels back
to them.  So instead of flooding the whole box again, this module floods a
WINDOW around the change, with every voxel just outside the window frozen
as a marker of the answer it already had, at the water level it was
reached at.  The window then has to prove itself, and grows on a side (and
floods again) when, on that side:

* an edge voxel's answer or water level came out different from before
  (the change reaches out of the window);
* a frozen voxel would now be reached lower than it was, or at its own
  level from another label (it would change, or a tie would decide it);
* a voxel anywhere in the window is decided by a tie -- its lowest
  neighbours share one level but carry two labels, so the flood's order
  of arrival picks -- and that order runs back, along one level, to the
  frozen voxels of that side.  The frozen voxels enter the window's flood
  as markers, first among their level, so an order that depends on them
  is not the full flood's.

A window that grows to the whole box is the full flood itself; once the
rounds of one click and the next window would pass a quarter of the box
(:data:`WINDOW_WHOLE_SHARE`), the next round is the whole box, so a click
never floods more than a quarter of the box on top of one full flood.

The flood is a copy of scikit-image's watershed (the same heap, the same
order of neighbours, the same first-come rule), so the whole-box flood here
equals ``skimage.segmentation.watershed`` voxel for voxel; it also keeps the
water level each voxel was reached at, which the frozen edge needs.

**The guard: where ties could decide, the full flood answers.**  The
watershed breaks a tie (two labels reaching a voxel at exactly one level)
by order of arrival.  A window reproduces that order when it comes from
distinct levels, or from the frozen edge (checked above).  It does not when
the order comes from two markers of exactly equal height (the heap hands
those out in an order set by every marker in the box) or from a tie one
level lower.  Both need flat spots: touching voxels with exactly the same
wall value.  Two markers on one flat patch at the lowest value are
flooded whole at the click that pairs them; a later click elsewhere is
NOT flooded whole for them (it was, until 2026-11-25: one such pair made
every later click flood the whole box), although the full flood's order
among equal markers can shift with any marker added or removed -- so on
their shared patch the two can swap voxels against a later full Preview.
Measured (the whole-patch session test): with the box faces or the dark
outside as markers the swap was never seen (0 of 528 steps each); on a
whole patch with the dark tick off, 3 of 264 steps, at most 127 voxels,
inside the pair's own cells.  So:

* only the wall pictures of :data:`CLICK_FILL_WALLS` (membrane, gradient)
  are redone by click; the plain intensity and the ridge filters leave
  large flat areas (half a Frangi box is exactly 0) and keep the full run;
* the flat-spot rule (:func:`flat_survey`): a box keeps the full run
  (:data:`FULL_RUN`, nothing shown) when its touching equal-valued voxel
  pairs at a value strictly between its lowest and highest wall value
  number more than one per 100,000 voxels where cells can grow
  (:data:`FLAT_PAIRS_PER_VOXEL`), or when any one flat spot they join holds
  more than :data:`FLAT_SPOT_MAX_VOXELS` voxels -- sparser, smaller spots
  are coincidences of the float values (9 pairs in a 5.2 million voxel
  patch), left to the windows and counted on the stage
  (:attr:`CellsClickFill.n_flat_pairs`);
* the flat patches AT the lowest value (a normalised picture puts its
  darkest 1 % at exactly 0) are normal; a click whose ball lies on one that
  also holds another click, the outside or a kept core floods the whole
  box that time, and so does a window in which a tie leads back, along one
  level, to two markers;
* flat patches at the highest value (the walls' clip at 1) are left alone:
  every front arrives there from a lower level.

**What this does not prove.**  At a coincidental flat spot the rule lets
through, a window can hand a tied voxel to another label than the full
flood would, when the tie's order comes from a tie one level lower outside
the window: the answer can then differ from the full run at those few
voxels.  The parity tests measure it (random click sessions on membrane
boxes with 1 to 20 injected flat pairs per 100,000 voxels, against the
full flood of the same clicks; the measured ceiling is pinned there).
Boxes with no flat spot between the clips equal the full run voxel for
voxel, and so do the clip plateaus.  Not proven, and not seen: two markers
of equal height whose fronts meet on a wall voxel between two separate
lowest patches outside the window, or two coincidentally equal values that
are not touching.

The Cells Preview reaches this through ``add_preview._cells_by_click``,
which keeps the stage between Previews and falls back to the full run on
any trouble; :func:`warm_kernel` compiles the flood once per process
first.

What it keeps per box and settings (:func:`prepare`): the wall picture, the
dark mask and the outline, the kept objects' cores, the outside marker, the
flood of the outside alone (every water level), and from then on the
current flood, its water levels, the raw cells and each cell's smoothed
shape.  :meth:`CellsClickFill.step` takes the full list of clicks and
returns what changed; :meth:`CellsClickFill.finish` smooths only the cells
whose raw shape changed, re-writes their neighbours over the smoothing's
reach in ascending id order (as ``smooth_pieces`` does), and clips to the
outline.

Only the fill to the walls with click markers is done here.  The spread
from each marker (``"compete"``) and the fill with a background marker keep
the full run: the spread's growth limit and its membrane step, and the
background marker's distance to the NEAREST other marker, couple the cells
in ways this window cannot bound.

Pure: numpy / scipy, plus numba for the flood (imported on first use).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .engine import ReshapeCancelled, _build_wall_stage
from .notes import MessageSink
from .region import (
    _Refused,
    _check_outline_arg,
    _check_spec,
    _dark_and_outline,
    _outline_refusal,
    _prepare,
    _spacing,
    _wall_images,
    eroded_cores,
    rasterise_points,
)
from .spec import ReshapeSpec

__all__ = [
    "CLICK_FILL_WALLS",
    "CellsClickFill",
    "FULL_RUN",
    "FinishResult",
    "StepResult",
    "flood",
    "is_click_fill",
    "kernel_is_warm",
    "prepare",
    "warm_kernel",
]

#: The label every marker that is not a click carries inside the flood (the
#: kept cores and the outside: none of them is ever a new cell, so one
#: label is enough -- the flood's choices never read a label's value).
OTHER = -1

#: The wall pictures a box may be redone click by click: their values are
#: continuous between the clip at 0 and the clip at 1.  The others (the
#: plain intensity and the two ridge filters) leave large areas at exactly
#: one value, where the flood breaks ties by an order a window cannot
#: reproduce; they keep the full run.
CLICK_FILL_WALLS = ("membrane", "gradient")

#: ``prepare``'s answer when the box is not redone click by click (the
#: caller runs the full Preview; this is not a refusal and shows nothing).
FULL_RUN = "full-run"

#: How far (voxels) the first window reaches past the change, per side.
WINDOW_START_VOXELS = 6

#: The least a side grows by when its edge fails a check (else by half the
#: window's extent on that axis).
WINDOW_GROW_VOXELS = 8

#: The voxels a click may flood in windows before the whole box is flooded
#: instead, as a share of the box: once the rounds so far and the next
#: window would pass it, the next round is the whole box (the full flood,
#: which needs no proof) -- so a click floods at most this share on top of
#: one full flood.  A quarter measured best on closed bricks (the worst click
#: 13.4 s at a half, 11.0 s at a quarter; smaller shares were slower).
WINDOW_WHOLE_SHARE = 0.25

#: The flat-spot rule's density: a box whose touching equal-valued voxel
#: pairs (at a value strictly between its lowest and highest wall value,
#: where cells can grow) number more than this share of the voxels where
#: cells can grow keeps the full run -- one per 100,000.  A ridge filter's
#: plateau or a coarsely quantised picture is far denser; the coincidences
#: of a real patch were 9 pairs in 5.2 million voxels.
FLAT_PAIRS_PER_VOXEL = 1e-5

#: The flat-spot rule's size: one flat spot (equal-valued voxels joined by
#: such pairs) larger than this keeps the full run, however sparse.
#: Coincidental equal floats come in pairs; a plateau is a run.
FLAT_SPOT_MAX_VOXELS = 3


# --------------------------------------------------------------------------- #
# the flood (scikit-image's watershed, with the water levels)
# --------------------------------------------------------------------------- #
def _ws_kernel(e, lab, lev, mark, nz, ny, nx):
    """scikit-image's ``watershed_raveled`` for compactness 0, no line,
    6-connectivity and a mask of every voxel, on a raveled C-order box.

    ``lab`` holds the markers (non-zero) and is filled in place; ``lev``
    receives the level each voxel was labelled at (a marker: its own
    value).  The heap is a copy of scikit-image's: entries ``(value, age)``,
    the markers pushed in raster order at age 0, a child pushed with
    ``max(own value, parent's)`` and the next age, labelled when pushed.

    ``mark`` (uint8 bits, in place): a voxel pushed at its parent's own
    level inherits the parent's bits, any other voxel gets 0 -- so a bit
    set on some markers follows every voxel whose order of arrival traces
    back to them along one level."""
    n = nz * ny * nx
    hv = np.empty(n, np.float64)
    ha = np.empty(n, np.int64)
    hi = np.empty(n, np.int64)
    size = 0
    plane = ny * nx
    for i in range(n):
        if lab[i] != 0:
            j = size
            hv[j] = e[i]
            ha[j] = 0
            hi[j] = i
            size += 1
            while j > 0:
                p = (j + 1) // 2 - 1
                if hv[j] < hv[p] or (hv[j] == hv[p] and ha[j] < ha[p]):
                    hv[p], hv[j] = hv[j], hv[p]
                    ha[p], ha[j] = ha[j], ha[p]
                    hi[p], hi[j] = hi[j], hi[p]
                    j = p
                else:
                    break
            lev[i] = e[i]
    age = 1
    while size > 0:
        v = hv[0]
        i = hi[0]
        size -= 1
        if size > 0:
            hv[0] = hv[size]
            ha[0] = ha[size]
            hi[0] = hi[size]
            j = 0
            while True:
                left = 2 * j + 1
                right = left + 1
                s = j
                if left < size:
                    if hv[left] < hv[j] or (hv[left] == hv[j]
                                            and ha[left] < ha[j]):
                        s = left
                    if right < size and (
                            hv[right] < hv[s] or (hv[right] == hv[s]
                                                  and ha[right] < ha[s])):
                        s = right
                else:
                    break
                if s == j:
                    break
                hv[s], hv[j] = hv[j], hv[s]
                ha[s], ha[j] = ha[j], ha[s]
                hi[s], hi[j] = hi[j], hi[s]
                j = s
        z = i // plane
        y = (i // nx) % ny
        x = i % nx
        # scikit-image's order: -z, -y, -x, +x, +y, +z
        for k in range(6):
            if k == 0:
                if z == 0:
                    continue
                nb = i - plane
            elif k == 1:
                if y == 0:
                    continue
                nb = i - nx
            elif k == 2:
                if x == 0:
                    continue
                nb = i - 1
            elif k == 3:
                if x == nx - 1:
                    continue
                nb = i + 1
            elif k == 4:
                if y == ny - 1:
                    continue
                nb = i + nx
            else:
                if z == nz - 1:
                    continue
                nb = i + plane
            if lab[nb] != 0:
                continue
            age += 1
            c = np.float64(e[nb])
            if c < v:
                c = v
            j = size
            hv[j] = c
            ha[j] = age
            hi[j] = nb
            size += 1
            while j > 0:
                p = (j + 1) // 2 - 1
                if hv[j] < hv[p] or (hv[j] == hv[p] and ha[j] < ha[p]):
                    hv[p], hv[j] = hv[j], hv[p]
                    ha[p], ha[j] = ha[j], ha[p]
                    hi[p], hi[j] = hi[j], hi[p]
                    j = p
                else:
                    break
            lab[nb] = lab[i]
            lev[nb] = c
            if c == v:
                mark[nb] = mark[i]
            else:
                mark[nb] = 0


_KERNEL = {}

#: True once :func:`warm_kernel` compiled the flood in this process.
_WARM = False


def _kernel():
    """The compiled flood; numba is imported here, on first use (the same
    sentences as the spread's solver when it cannot be set up)."""
    if not _KERNEL:
        from .compete import NUMBA_CACHE_FAILED, NUMBA_MISSING

        try:
            import numba
        except Exception as exc:  # ImportError, or a broken install
            raise ImportError(NUMBA_MISSING) from exc
        try:
            _KERNEL["ws"] = numba.njit(cache=True, nogil=True)(_ws_kernel)
        except RuntimeError as exc:
            raise RuntimeError(NUMBA_CACHE_FAILED) from exc
    return _KERNEL["ws"]


def warm_kernel():
    """Compile the flood once (one flood of a 2x2x2 box); ``True`` when it
    runs, ``False`` when numba is missing or cannot keep its compiled code
    (the page then keeps the full run).  The first call in a fresh install
    compiles for a few seconds; later processes load the cached code."""
    global _WARM
    try:
        flood(np.zeros((2, 2, 2), np.float32), np.ones((2, 2, 2), np.int32))
    except (ImportError, RuntimeError):
        return False
    _WARM = True
    return True


def kernel_is_warm():
    """True once :func:`warm_kernel` succeeded in this process."""
    return _WARM


def flood(elevation, markers):
    """``(labels int32, levels float32)``: scikit-image's watershed of
    ``elevation`` from ``markers`` (compactness 0, 6-connected, every voxel
    in the mask) -- the same labels -- and the level each voxel was reached
    at.  ``markers`` is not changed; a voxel no marker reaches keeps 0 and
    level ``inf``."""
    elev = np.ascontiguousarray(elevation, np.float32)
    lab = np.array(markers, np.int32, copy=True, order="C")
    if lab.shape != elev.shape or lab.ndim != 3:
        raise ValueError("the elevation is {} and the markers {}; both must "
                         "be the same 3D shape".format(elev.shape, lab.shape))
    lev = np.full(lab.shape, np.inf, np.float32)
    mark = np.zeros(lab.shape, np.uint8)
    nz, ny, nx = lab.shape
    if lab.size:
        _kernel()(elev.ravel(), lab.ravel(), lev.ravel(), mark.ravel(),
                  nz, ny, nx)
    return lab, lev


def _flood_marked(elevation, markers, mark):
    """:func:`flood` that also carries ``mark`` (uint8, in place)."""
    elev = np.ascontiguousarray(elevation, np.float32)
    lab = np.array(markers, np.int32, copy=True, order="C")
    lev = np.full(lab.shape, np.inf, np.float32)
    nz, ny, nx = lab.shape
    _kernel()(elev.ravel(), lab.ravel(), lev.ravel(), mark.ravel(), nz, ny, nx)
    return lab, lev


#: The mark every marker inside the window carries (the six sides use bits
#: 1-32): a tie whose two labels both lead back to markers along one level
#: is decided by the markers' order in the heap.
_MARKER_BIT = 64

#: The failure that sends a click to the whole box.
_MARKERS_TIE = ("markers", "tie")


def _tied_marks(lab, lev, mark, inner, root):
    """``(bits, markers_tie)``: the OR of ``mark`` over the tied best
    neighbours of every non-marker voxel of ``inner`` -- a voxel is tied
    when its lowest-level neighbours carry two different labels (the flood
    then decides by order of arrival) -- and whether some tie has two
    labels that both lead back, along one level, to markers."""
    sub = tuple(slice(s.start - 1 if s.start > 0 else s.start,
                      s.stop + 1 if s.stop < n else s.stop)
                for s, n in zip(inner, lab.shape))
    L = lab[sub]
    F = lev[sub]
    M = mark[sub]
    pad_lo = [s.start - t.start for s, t in zip(inner, sub)]
    core = tuple(slice(p, p + (s.stop - s.start)) for p, s in zip(pad_lo, inner))
    Fp = np.pad(F, 1, constant_values=np.inf)
    Lp = np.pad(L, 1, constant_values=0)
    Mp = np.pad(M, 1, constant_values=0)
    shape = F.shape
    views = []
    for axis in range(3):
        for d in (-1, 1):
            sl = [slice(1, 1 + n) for n in shape]
            sl[axis] = slice(1 + d, 1 + d + shape[axis])
            views.append(tuple(sl))
    best = np.full(shape, np.inf, np.float32)
    for v in views:
        np.minimum(best, Fp[v], out=best)
    first = np.zeros(shape, np.int32)
    tied = np.zeros(shape, bool)
    bits = np.zeros(shape, np.uint8)
    first_m = np.zeros(shape, np.int32)       # a marker-led label
    both_m = np.zeros(shape, bool)
    for v in views:
        at = Fp[v] == best
        lv = Lp[v]
        tied |= at & (first != 0) & (lv != first)
        first = np.where(at & (first == 0), lv, first)
        bits |= np.where(at, Mp[v], 0).astype(np.uint8)
        led = at & ((Mp[v] & _MARKER_BIT) != 0)
        both_m |= led & (first_m != 0) & (lv != first_m)
        first_m = np.where(led & (first_m == 0), lv, first_m)
    tied = tied[core] & (root == 0)
    if not tied.any():
        return 0, False
    side_bits = int(np.bitwise_or.reduce(bits[core][tied])) & ~_MARKER_BIT
    return side_bits, bool(np.any(both_m[core] & tied))


# --------------------------------------------------------------------------- #
# small box helpers
# --------------------------------------------------------------------------- #
def _bbox_of(mask, offset=(0, 0, 0)):
    """Tight ``((lo, hi) x 3)`` of a bool array, shifted by ``offset``, or
    ``None`` when nothing is set."""
    if not mask.any():
        return None
    out = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        hit = np.flatnonzero(mask.any(axis=other))
        out.append((int(hit[0]) + offset[axis], int(hit[-1]) + 1 + offset[axis]))
    return tuple(out)


def _union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return tuple((min(p[0], q[0]), max(p[1], q[1])) for p, q in zip(a, b))


def _grow_box(box, by, shape):
    return tuple((max(0, lo - b), min(n, hi + b))
                 for (lo, hi), b, n in zip(box, by, shape))


def _voxels(box):
    return int(np.prod([hi - lo for lo, hi in box]))


def _slices(box):
    return tuple(slice(lo, hi) for lo, hi in box)


def _meets(a, b):
    return all(p[0] < q[1] and q[0] < p[1] for p, q in zip(a, b))


def _ball_box(point, radius_um, spacing, shape):
    """The box ``rasterise_points`` scans for one point (clipped), or
    ``None`` when it is empty."""
    c = np.rint(np.asarray(point, float)[-3:]).astype(int)
    half = [int(np.ceil(float(radius_um) / float(s))) for s in spacing]
    box = tuple((max(0, int(c[k]) - half[k]), min(int(shape[k]),
                                                 int(c[k]) + half[k] + 1))
                for k in range(3))
    if any(lo >= hi for lo, hi in box):
        return None
    return box


def _raster_in(points, box, radius_um, spacing):
    """``rasterise_points`` restricted to ``box``: the same numbers (1 = the
    first point of ``points``) on every voxel of the box.  Exact: each
    voxel's owner is the first ball holding it, and the distance arithmetic
    is the same (integer points shifted by integers)."""
    start = [lo for lo, _ in box]
    shape = tuple(hi - lo for lo, hi in box)
    local = [tuple(float(p[k]) - start[k] for k in range(3)) for p in points]
    return rasterise_points(local, shape, radius_um=radius_um,
                            spacing=spacing)


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise ReshapeCancelled()


# --------------------------------------------------------------------------- #
# the answers
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    """What one :meth:`CellsClickFill.step` changed.

    ``status`` -- ``"ok"``, or ``"refused"`` with ``refusal_reason``
    ``"no-markers"`` when no click has a voxel on empty space (the full run's
    refusal; the flood is kept up to date anyway).  ``changed_ids`` -- the
    reserved ids whose RAW cell changed (new, gone or reshaped); ``window``
    -- the box (slices, box coordinates) the last flood covered, ``None``
    when nothing had to be flooded; ``n_floods`` -- how many floods it took
    (1 when the first window held); ``voxels_flooded`` -- the voxels of every
    round together; ``absorbed_ids`` -- reserved ids whose ball has no voxel
    (taken by an earlier ball or a kept object), in placement order;
    ``n_fragments_dropped`` -- over the whole box: the pieces of a cell's
    flood that do not hold its ball, left out (the full run's count)."""

    status: str = "ok"
    refusal_reason: str | None = None
    changed_ids: list = field(default_factory=list)
    window: tuple | None = None
    n_floods: int = 0
    window_voxels: int = 0
    voxels_flooded: int = 0
    absorbed_ids: list = field(default_factory=list)
    n_fragments_dropped: int = 0
    seconds: float = 0.0


@dataclass
class FinishResult:
    """What :meth:`CellsClickFill.finish` hands the page, the same fields as
    the full Preview's finish.  ``result`` is a read-only view of the
    stage's own array (box shape, the final ids): copy it to keep it -- the
    next finish updates it in place.  ``changed_ids`` -- the final ids whose
    voxels changed since the previous finish; ``diff_box`` -- slices
    holding every voxel that changed (``None``: none)."""

    result: np.ndarray | None = None
    ids: list = field(default_factory=list)
    sizes: dict = field(default_factory=dict)
    edge_ids: set = field(default_factory=set)
    small_ids: set = field(default_factory=set)
    n_below_min_size: int = 0
    n_emptied_by_smooth: int = 0
    changed_ids: list = field(default_factory=list)
    diff_box: tuple | None = None
    seconds: float = 0.0


@dataclass
class _Slot:
    point: tuple
    rid: int
    order: int
    pbox: tuple | None = None     # the flood's region
    rbox: tuple | None = None     # the raw cell
    size: int = 0                 # raw voxels
    nball: int = 0                # ball voxels on empty space
    n_frag: int = 0               # flood pieces left out (no ball in them)
    version: int = 0              # bumped when the raw cell changes


@dataclass
class _Smoothed:
    version: int
    roi: tuple                    # ((lo, hi) x 3), box coordinates
    mask: np.ndarray              # bool, the roi's shape


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #
class CellsClickFill:
    """The prepared box for Cells ▸ Click markers ▸ Fill to the walls.

    Build it with :func:`prepare`.  Then, after every change of the click
    list, call :meth:`step` with the WHOLE list (box-local ZYX points and
    their reserved ids, in placement order) and :meth:`finish` with the
    page's settings.  The answer equals the full Preview
    (``reshape_empty_region`` and the Preview's finish) for the same
    clicks, up to the tie described in the module notes.  One object serves
    one box with one set of settings; it is not thread-safe (one worker at
    a time).
    """

    #: the flat-spot reason :func:`prepare` let through when asked to
    #: (``allow_flat``); ``None``: the box had none the rule keeps whole
    flat_override = None

    def __init__(self, *, spec, roi, spacing, shape, existing, elevation,
                 dark_raw, outline, cores, messages, low=None,
                 other_patches=frozenset(), n_flat_pairs=0):
        p = spec.params
        self.spec = spec
        self.roi = roi
        self.spacing = spacing
        self.shape = shape
        self.messages = list(messages)
        self._existing = existing
        self._empty = existing == 0
        self._elev = np.ascontiguousarray(elevation, np.float32)
        self._outline = outline
        self._dark_raw = dark_raw
        self._cores = cores
        self._click_r = float(p["click_radius_um"])
        self._clear_r = float(p["dark_clearance_um"])
        # the box faces in the outside marker (False: a whole patch filled
        # the way a box is -- its faces open, its dark not eroded back from
        # them), as the full run's flood
        self._faces = bool(p.get("faces_outside", True))
        self._ero = None
        self._ero_half = (0, 0, 0)
        if dark_raw is not None and self._clear_r > 0:
            from .arbiter import physical_ball

            self._ero = physical_ball(self._clear_r, spacing)
            self._ero_half = tuple(int(n) // 2 for n in self._ero.shape)
        # no clicks yet: the outside alone
        self._dark_cleared = None if dark_raw is None else dark_raw.copy()
        self._dark_out = None
        if dark_raw is not None:
            self._dark_out = (ndi.binary_erosion(
                dark_raw, structure=self._ero, border_value=int(not self._faces))
                if self._ero is not None else dark_raw.copy())
        self._ball = np.zeros(shape, np.int32)       # slot per ball voxel
        self._root = np.zeros(shape, np.int32)
        whole = tuple((0, n) for n in shape)
        self._root[...] = self._roots_in(whole)
        self._part, self._level = flood(self._elev, self._root)
        self._raw = np.zeros(shape, np.int32)        # slot per raw voxel
        self._slots = {}                             # slot -> _Slot
        self._order = []                             # slots, placement order
        self._next_slot = 1
        self._smoothed = {}                          # slot -> _Smoothed
        self._final = None
        self._final_ids = {}                         # rid -> size
        self._finish_key = None
        self._dirty_slots = set()                    # raw changed since finish
        self._gone_rois = []                         # rois of removed cells
        self._broken = False                         # a step / finish cut short
        # the flat patches at the lowest wall value (see flat_survey)
        self._low = (np.zeros(shape, np.int32) if low is None
                     else np.asarray(low, np.int32))
        self._low_other = frozenset(other_patches)
        self._low_ties = 0          # steps flooded whole for a marker tie
        #: the coincidental flat pairs the flat-spot rule let through (the
        #: console line says them)
        self.n_flat_pairs = int(n_flat_pairs)

    @property
    def existing(self):
        """The kept objects the box was prepared with (the stage's own
        copy: read it, never write it)."""
        return self._existing

    @property
    def elevation(self):
        """The wall picture the box floods (float32, the stage's own: read
        it, never write it)."""
        return self._elev

    # ---- the markers ------------------------------------------------------ #
    def _outside_in(self, box):
        sl = _slices(box)
        out = np.zeros(tuple(hi - lo for lo, hi in box), bool)
        for axis in range(3 if self._faces else 0):
            lo, hi = box[axis]
            idx = [slice(None)] * 3
            if lo == 0:
                idx[axis] = 0
                out[tuple(idx)] = True
            if hi == self.shape[axis]:
                idx[axis] = -1
                out[tuple(idx)] = True
        if self._dark_out is not None:
            out |= self._dark_out[sl]
        if self._outline is not None:
            out |= ~self._outline[sl]
        out &= self._empty[sl] & (self._ball[sl] == 0)
        return out

    def _roots_in(self, box):
        sl = _slices(box)
        root = np.where(self._ball[sl] != 0, self._ball[sl], 0).astype(np.int32)
        other = (self._cores[sl] != 0) | self._outside_in(box)
        root[other & (root == 0)] = OTHER
        return root

    @property
    def broken(self):
        """True once a step or a finish was cut short (a Stop or an error):
        the stage is half-updated, and every later step or finish raises --
        prepare the box again."""
        return self._broken

    def _check_whole(self):
        if self._broken:
            raise RuntimeError(STAGE_BROKEN)

    # ---- one change of the click list ------------------------------------ #
    def step(self, points, ids, *, cancel_event=None):
        """Bring the fill up to date with the click list ``points`` (box-
        local ZYX, placement order) reserving ``ids`` (one per point).
        Returns a :class:`StepResult`.  A cancel raises
        :class:`~.engine.ReshapeCancelled`; a cancel or any error leaves
        the stage :attr:`broken` (every later call raises: prepare again)."""
        self._check_whole()
        pts = [tuple(int(v) for v in np.asarray(p)[-3:]) for p in points or ()]
        ids = [int(i) for i in (ids or ())]
        if len(pts) != len(ids):
            raise ValueError("{} points but {} ids".format(len(pts), len(ids)))
        if len(set(ids)) != len(ids):
            raise ValueError("the reserved ids repeat: {}".format(ids))
        if any(i <= 0 for i in ids):
            raise ValueError("a reserved id must be positive: {}".format(ids))
        # broken until the step ends: the balls, the roots, the dark, the
        # slots and the order change before the flood can be stopped
        self._broken = True
        out = self._step(pts, ids, cancel_event)
        self._broken = False
        return out

    def _step(self, pts, ids, cancel_event):
        t0 = time.perf_counter()
        out = StepResult()
        # ---- which clicks survive, in the same order --------------------- #
        old = [(self._slots[s].point, self._slots[s].rid, s) for s in self._order]
        where = {(pt, rid): k for k, (pt, rid, _s) in enumerate(old)}
        keep, last = {}, -1
        for pt, rid in zip(pts, ids):
            k = where.get((pt, rid))
            if k is not None and k > last:
                keep[(pt, rid)] = old[k][2]
                last = k
        removed = [s for (_pt, _rid, s) in old if s not in keep.values()]
        order, added = [], []
        for pt, rid in zip(pts, ids):
            s = keep.get((pt, rid))
            if s is None:
                s = self._next_slot
                self._next_slot += 1
                self._slots[s] = _Slot(point=pt, rid=rid, order=0)
                added.append(s)
            order.append(s)
        for k, s in enumerate(order):
            self._slots[s].order = k
        changed_pts = ([self._slots[s].point for s in removed]
                       + [self._slots[s].point for s in added])
        # ---- the balls and the cleared dark, where they can change ------- #
        reach = max(self._click_r, self._clear_r if self._dark_raw is not None
                    else 0.0)
        region = None
        for pt in changed_pts:
            b = _ball_box(pt, reach, self.spacing, self.shape)
            region = _union(region, b)
        touched = set(removed) | set(added)
        d0 = None
        if region is not None:
            region = _grow_box(region, [h + 1 for h in self._ero_half],
                               self.shape)
            d0, touched_more = self._update_markers(region, order)
            touched |= touched_more
        self._order = order
        gone_rids = [self._slots[s].rid for s in removed
                     if self._slots[s].rbox is not None]
        # ---- the window: the change, the regions it touches -------------- #
        window = d0
        for s in touched:
            if s in self._slots:
                window = _union(window, self._slots[s].pbox)
        if window is None:
            # nothing on the volume moved (no click came or went, or a ball
            # with no voxel did)
            self._forget(removed)
            return self._done(out, order, gone_rids, {}, t0)
        window = _grow_box(window, (WINDOW_START_VOXELS,) * 3, self.shape)
        box_n = int(np.prod(self.shape))
        whole = tuple((0, n) for n in self.shape)
        if (_voxels(window) > WINDOW_WHOLE_SHARE * box_n
                or self._clicks_share_a_low_patch(order, added)):
            window = whole
        before = {}
        while True:
            _check_cancel(cancel_event)
            out.n_floods += 1
            out.voxels_flooded += _voxels(window)
            lab, lev, fails = self._flood_window(window)
            if _MARKERS_TIE in fails and window != whole:
                # two markers of equal height decide a voxel: their order
                # is the whole box's, so flood the whole box
                self._low_ties += 1
                window = whole
                continue
            fails.discard(_MARKERS_TIE)
            if not fails:
                break
            grow = []
            for axis in range(3):
                lo, hi = window[axis]
                by = max(WINDOW_GROW_VOXELS, (hi - lo) // 2)
                grow.append((by if (axis, 0) in fails else 0,
                             by if (axis, 1) in fails else 0))
            window = tuple((max(0, lo - g0), min(n, hi + g1))
                           for (lo, hi), (g0, g1), n
                           in zip(window, grow, self.shape))
            if (out.voxels_flooded + _voxels(window)
                    > WINDOW_WHOLE_SHARE * box_n):
                # more rounds would cost more than one flood of the box
                window = whole
        sl = _slices(window)
        old_part = self._part[sl]
        diff = lab != old_part
        moved = set(int(v) for v in np.unique(old_part[diff]) if v > 0)
        moved |= set(int(v) for v in np.unique(lab[diff]) if v > 0)
        self._part[sl] = lab
        self._level[sl] = lev
        out.window = sl
        out.window_voxels = int(lab.size)
        # ---- the raw cells of every slot the flood moved ------------------ #
        for s in sorted(moved | touched):
            if s not in self._slots:
                continue
            before[s] = self._slots[s].version
            self._redo_raw(s, window, out)
        self._forget(removed)
        return self._done(out, order, gone_rids, before, t0)

    def _done(self, out, order, gone_rids, before, t0):
        out.changed_ids = sorted(
            set(self._slots[s].rid for s in before
                if s in self._slots and self._slots[s].version != before[s])
            | set(gone_rids))
        out.absorbed_ids = [self._slots[s].rid for s in order
                            if not self._ball_has(s)]
        out.n_fragments_dropped = sum(self._slots[s].n_frag for s in order)
        if not any(self._ball_has(s) for s in order):
            out.status = "refused"
            out.refusal_reason = "no-markers"
        out.seconds = time.perf_counter() - t0
        return out

    def _clicks_share_a_low_patch(self, order, changed=None):
        """True when a click's ball lies on a flat patch at the lowest wall
        value that also holds another click or a marker that is not a click:
        markers of equal height, whose order of leaving the flood depends on
        every marker in the box -- only the whole box floods them as the
        full run does.  ``changed`` (the slots this step added): only a
        tie that involves one of them floods the whole box.  A pair that
        shared a patch before this step keeps the split the flood that
        paired them drew; a later click elsewhere CAN shift the full
        flood's order among equal markers, so against a later full
        Preview the two may swap voxels on their shared patch (measured:
        never with the faces or the dark outside as markers; on a whole
        patch with the dark tick off 3 of 264 steps, at most 127 voxels --
        the module notes).  Before this, one such pair anywhere made EVERY
        later click flood the whole box for the rest of the session (8 s a
        click on a 5 M patch).  ``None``: any pair, as before."""
        if not self._low.any():
            return False
        chg = None if changed is None else set(changed)
        seen = {}
        for s in order:
            pt = self._slots[s].point
            b = _ball_box(pt, self._click_r, self.spacing, self.shape)
            if b is None:
                continue
            sl = _slices(b)
            on = self._low[sl][self._ball[sl] == s]
            for patch in np.unique(on[on > 0]):
                if int(patch) in self._low_other:
                    if chg is None or s in chg:
                        self._low_ties += 1
                        return True
                    continue
                first = seen.setdefault(int(patch), s)
                if first != s and (chg is None or s in chg or first in chg):
                    self._low_ties += 1
                    return True
        return False

    def _ball_has(self, s):
        return self._slots[s].nball > 0

    def _update_markers(self, region, order):
        """Re-draw the balls, the cleared dark and the outside inside
        ``region``; returns ``(bbox of changed roots, slots whose ball
        changed)``."""
        sl = _slices(region)
        pts = [self._slots[s].point for s in order]
        meet = [k for k, pt in enumerate(pts)
                if (b := _ball_box(pt, self._click_r, self.spacing,
                                   self.shape)) is not None
                and _meets(b, region)]
        balls = _raster_in([pts[k] for k in meet], region, self._click_r,
                           self.spacing)
        lut = np.zeros(len(meet) + 1, np.int32)
        for j, k in enumerate(meet, start=1):
            lut[j] = order[k]
        balls = lut[balls]
        balls[~self._empty[sl]] = 0
        old_ball = self._ball[sl]
        changed_ball = balls != old_ball
        touched = set(int(v) for v in np.unique(old_ball[changed_ball]) if v)
        touched |= set(int(v) for v in np.unique(balls[changed_ball]) if v)
        for s in touched:
            if s in self._slots:
                self._slots[s].nball += (int(np.count_nonzero(balls == s))
                                         - int(np.count_nonzero(old_ball == s)))
        self._ball[sl] = balls
        if self._dark_raw is not None:
            clear_box = _grow_box(region, [0, 0, 0], self.shape)
            csl = _slices(clear_box)
            cmeet = [pt for pt in pts
                     if (b := _ball_box(pt, self._clear_r, self.spacing,
                                        self.shape)) is not None
                     and _meets(b, clear_box)]
            if cmeet:
                clear = _raster_in(cmeet, clear_box, self._clear_r,
                                   self.spacing) > 0
            else:
                clear = np.zeros(tuple(hi - lo for lo, hi in clear_box), bool)
            self._dark_cleared[csl] = self._dark_raw[csl] & ~clear
            self._redo_dark_out(region)
        new_root = self._roots_in(region)
        diff = new_root != self._root[sl]
        start = tuple(lo for lo, _ in region)
        d0 = _bbox_of(diff, start)
        self._root[sl] = new_root
        return d0, touched

    def _redo_dark_out(self, region):
        """``dark_out`` inside ``region`` from ``dark_cleared``: the erosion
        of a crop one structure wider (the crop's own edge never reaches the
        region, and at a box face it is the box's own edge)."""
        if self._ero is None:
            self._dark_out[_slices(region)] = self._dark_cleared[_slices(region)]
            return
        crop = _grow_box(region, self._ero_half, self.shape)
        # the crop's own edge inside the box never reaches the region; at
        # a box face it is the box's edge: a cut (border 0), or with the
        # faces open the dark going on (border 1), as the full run erodes
        er = ndi.binary_erosion(self._dark_cleared[_slices(crop)],
                                structure=self._ero,
                                border_value=int(not self._faces))
        inner = tuple(slice(r0 - c0, r1 - c0)
                      for (r0, r1), (c0, _c1) in zip(region, crop))
        self._dark_out[_slices(region)] = er[inner]

    def _flood_window(self, window):
        """Flood ``window`` with the voxels around it frozen; returns
        ``(labels, levels, failing sides)``.  A side ``(axis, 0|1)`` fails
        when, next to it:

        * an edge voxel's label or level changed (the change reaches out);
        * a frozen voxel would now be reached lower than it was, or at its
          own level from a different label (it would change, or be decided
          by a tie);
        * anywhere in the window, a voxel whose two best neighbours tie
          (same level, different labels) is decided by an order of arrival
          that runs, along one level, back to the frozen voxels of that side
          (they enter the flood as markers, so their order is not the full
          flood's)."""
        ring = tuple((lo - 1 if lo > 0 else lo, hi + 1 if hi < n else hi)
                     for (lo, hi), n in zip(window, self.shape))
        rsl = _slices(ring)
        inner = tuple(slice(w0 - r0, w1 - r0)
                      for (w0, w1), (r0, _r1) in zip(window, ring))
        e = np.array(self._elev[rsl], np.float32, copy=True)
        old_lab = self._part[rsl]
        old_lev = self._level[rsl]
        lab = np.array(old_lab, np.int32, copy=True)
        is_ring = np.ones(lab.shape, bool)
        is_ring[inner] = False
        e[is_ring] = old_lev[is_ring]
        root_w = self._root[_slices(window)]
        lab[inner] = root_w
        mark = np.zeros(lab.shape, np.uint8)
        mark[inner] = np.where(root_w != 0, _MARKER_BIT, 0).astype(np.uint8)
        sides = []
        for axis in range(3):
            for side in (0, 1):
                w0, w1 = window[axis]
                r0, _r1 = ring[axis]
                if side == 0 and w0 > 0:
                    wi, ri = w0 - r0, w0 - r0 - 1
                elif side == 1 and w1 < self.shape[axis]:
                    wi, ri = w1 - 1 - r0, w1 - r0
                else:
                    continue
                bit = 1 << (2 * axis + side)
                idx_w = list(inner)
                idx_r = list(inner)
                idx_w[axis] = wi
                idx_r[axis] = ri
                sides.append((axis, side, bit, tuple(idx_w), tuple(idx_r)))
                mark[tuple(idx_r)] |= bit
        lab_out, lev_out = _flood_marked(e, lab, mark)
        fails = set()
        if sides:
            fails |= self._frozen_changes(window, ring, lab_out, lev_out,
                                          sides)
        bits, markers_tie = _tied_marks(lab_out, lev_out, mark, inner, root_w)
        for axis, side, bit, _w, _r in sides:
            if bits & bit:
                fails.add((axis, side))
        if markers_tie:
            fails.add(_MARKERS_TIE)
        return lab_out[inner], lev_out[inner], fails

    def _frozen_changes(self, window, ring, lab, lev, sides):
        """The sides whose frozen voxels would not keep their answer: each
        frozen voxel next to the window is decided again from its six
        neighbours (the window's new answers, the rest as they were); a side
        fails when one would be reached at another level, or when its lowest
        neighbours carry another label (it would change, or tie)."""
        far = tuple((max(0, lo - 2), min(n, hi + 2))
                    for (lo, hi), n in zip(window, self.shape))
        fsl = _slices(far)
        L = np.array(self._part[fsl], np.int32, copy=True)
        F = np.array(self._level[fsl], np.float32, copy=True)
        at = tuple(slice(r0 - f0, r1 - f0)
                   for (r0, r1), (f0, _f1) in zip(ring, far))
        L[at] = lab
        F[at] = lev
        E = self._elev[fsl]
        R = self._root[fsl]
        Fp = np.pad(F, 1, constant_values=np.inf)
        Lp = np.pad(L, 1, constant_values=0)
        win = tuple(slice(w0 - f0 + 1, w1 - f0 + 1)
                    for (w0, w1), (f0, _f1) in zip(window, far))
        fails = set()
        for axis, side, _bit, _iw, _ir in sides:
            w0, w1 = window[axis]
            plane = (w0 - 1 if side == 0 else w1) - far[axis][0]
            idx = list(win)
            idx[axis] = plane + 1
            best = np.full([s.stop - s.start for k, s in enumerate(idx)
                            if k != axis], np.inf, np.float32)
            views = []
            for ax2 in range(3):
                for d in (-1, 1):
                    v = list(idx)
                    if ax2 == axis:
                        v[ax2] = plane + 1 + d
                    else:
                        v[ax2] = slice(idx[ax2].start + d, idx[ax2].stop + d)
                    views.append(tuple(v))
            for v in views:
                np.minimum(best, Fp[v], out=best)
            own = tuple(slice(s.start - 1, s.stop - 1) if k != axis else plane
                        for k, s in enumerate(idx))
            lr, fr = L[own], F[own]
            moved = np.maximum(E[own], best) != fr
            for v in views:
                moved |= (Fp[v] == best) & (Lp[v] != lr)
            if np.any(moved & (R[own] == 0)):
                fails.add((axis, side))
        return fails

    def _redo_raw(self, s, window, out):
        """The slot's raw cell: its flood region on empty space (and inside
        the outline), keeping only the piece(s) that hold its ball -- the
        full run's ``keep_marker_components`` for one id."""
        slot = self._slots[s]
        start = tuple(lo for lo, _ in window)
        sl_w = _slices(window)
        in_w = _bbox_of(self._part[sl_w] == s, start)
        box = _union(slot.pbox, in_w)
        if box is not None:
            sl = _slices(box)
            mask = self._part[sl] == s
            slot.pbox = _bbox_of(mask, tuple(lo for lo, _ in box))
        else:
            slot.pbox = None
        new_box, new_mask = None, None
        slot.n_frag = 0
        if slot.pbox is not None:
            sl = _slices(slot.pbox)
            mask = (self._part[sl] == s) & self._empty[sl]
            if self._outline is not None:
                mask &= self._outline[sl]
            if mask.any():
                comp, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
                on_ball = mask & (self._ball[sl] == s)
                if not (n <= 1 and on_ball.any()):
                    keep_ids = np.unique(comp[on_ball])
                    keep_ids = keep_ids[keep_ids != 0]
                    slot.n_frag = int(n - keep_ids.size)
                    mask = np.isin(comp, keep_ids) & mask
                if mask.any():
                    new_box = _bbox_of(mask, tuple(lo for lo, _ in slot.pbox))
                    inner = tuple(slice(a0 - b0, a1 - b0) for (a0, a1), (b0, _b1)
                                  in zip(new_box, slot.pbox))
                    new_mask = mask[inner]
        old_box = slot.rbox
        same = (old_box == new_box)
        if same and new_box is not None:
            same = np.array_equal(self._raw[_slices(new_box)] == s, new_mask)
        if same:
            return
        if old_box is not None:
            v = self._raw[_slices(old_box)]
            v[v == s] = 0
        if new_box is not None:
            v = self._raw[_slices(new_box)]
            v[new_mask] = s
            slot.size = int(np.count_nonzero(new_mask))
        else:
            slot.size = 0
        slot.rbox = new_box
        slot.version += 1
        self._dirty_slots.add(s)

    def _forget(self, removed):
        """Drop removed slots (their flood region is gone by now); what the
        last finish drew of them is re-written by the next."""
        for s in removed:
            slot = self._slots.pop(s, None)
            if slot is None:
                continue
            if slot.rbox is not None:
                v = self._raw[_slices(slot.rbox)]
                v[v == s] = 0
            sm = self._smoothed.pop(s, None)
            if sm is not None:
                self._gone_rois.append(sm.roi)
            self._dirty_slots.discard(s)

    # ---- the finish ------------------------------------------------------- #
    def finish(self, *, min_size=0, auto_smooth=True, smooth_sigma=1.5,
               clear_border=True, edge_margin=3, whole_patch=False,
               cancel_event=None):
        """The Preview's finish on the current raw cells, redone only where
        it can change: the size floor (``min_size``), the edge flags on the
        raw cells, the smoothing of every cell whose raw shape changed, the
        neighbours over the smoothing's reach re-written in ascending id
        order, the clip to the outline.  Returns a :class:`FinishResult`
        whose ``result`` is a read-only view of the stage's own array.
        ``cancel_event`` is read between cells; a Stop raises
        :class:`~.engine.ReshapeCancelled` and, like any error, leaves the
        stage :attr:`broken`."""
        self._check_whole()
        self._broken = True
        out = self._finish(min_size=min_size, auto_smooth=auto_smooth,
                           smooth_sigma=smooth_sigma,
                           clear_border=clear_border, edge_margin=edge_margin,
                           whole_patch=whole_patch, cancel_event=cancel_event)
        self._broken = False
        return out

    def _finish(self, *, min_size, auto_smooth, smooth_sigma, clear_border,
                edge_margin, whole_patch, cancel_event):
        from ..add_preview import small_piece_ids
        from ..add_segment import BOX_FACES
        from ..batch_refine import (
            RefineStep,
            _recipe_margin,
            apply_recipe_to_label,
            compute_label_roi,
        )

        t0 = time.perf_counter()
        out = FinishResult()
        key = (int(min_size or 0), bool(auto_smooth), float(smooth_sigma))
        live = [s for s in self._order
                if self._slots[s].size > 0
                and self._slots[s].size >= int(min_size or 0)]
        out.n_below_min_size = sum(
            1 for s in self._order
            if 0 < self._slots[s].size < int(min_size or 0))
        rids = [self._slots[s].rid for s in live]
        # the full run's id dtype follows every click whose ball survived
        top = max([self._slots[s].rid for s in self._order
                   if self._ball_has(s)] or [0])
        dtype = np.int32 if top <= np.iinfo(np.int32).max else np.int64
        recipe = [RefineStep(method="sdf",
                             params={"sigma_z": float(smooth_sigma),
                                     "sigma_xy": float(smooth_sigma)})]
        margin = _recipe_margin(recipe, self.shape)
        shape_only = np.broadcast_to(np.int8(0), self.shape)
        fresh = (self._final is None or key != self._finish_key
                 or self._final.dtype != dtype)
        if fresh:
            self._smoothed = {}
            self._final = np.zeros(self.shape, dtype)
            self._final_ids = {}
            dirty = set(live)
            gone = []
        else:
            dirty = {s for s in self._dirty_slots if s in self._slots}
            gone = list(self._gone_rois)
        live_set = set(live)
        # a slot that crossed the size floor either way is dirty too
        for s in list(self._smoothed):
            if s not in live_set:
                dirty.add(s)
        for s in live:
            if s not in self._smoothed:
                dirty.add(s)
        boxes = list(gone)
        for s in sorted(dirty):
            _check_cancel(cancel_event)
            sm = self._smoothed.get(s)
            if sm is not None:
                boxes.append(sm.roi)
            if s not in live_set or s not in self._slots:
                self._smoothed.pop(s, None)
                continue
            slot = self._slots[s]
            if auto_smooth:
                roi = compute_label_roi(shape_only, 1, margin,
                                        bbox=_slices(slot.rbox))
                crop = (self._raw[roi] == s).astype(np.uint8)
                res, _n = apply_recipe_to_label(crop, 1,
                                                (slice(None),) * 3, recipe)
                mask = res == 1
                roi_box = tuple((r.start, r.stop) for r in roi)
            else:
                roi_box = slot.rbox
                mask = self._raw[_slices(roi_box)] == s
            self._smoothed[s] = _Smoothed(version=slot.version, roi=roi_box,
                                          mask=mask)
            boxes.append(roi_box)
            boxes.append(slot.rbox)
        self._dirty_slots = set()
        self._gone_rois = []
        # ---- re-write every dirty box, ascending ids --------------------- #
        by_rid = sorted(live, key=lambda s: self._slots[s].rid)
        lut = np.zeros(self._next_slot + 1, dtype)
        for s in live:
            lut[s] = self._slots[s].rid
        changed = set()
        diff_box = None
        for box in boxes:
            if box is None:
                continue
            _check_cancel(cancel_event)
            sl = _slices(box)
            start = tuple(lo for lo, _ in box)
            work = lut[self._raw[sl]]
            for s in by_rid:
                sm = self._smoothed[s]
                if not _meets(sm.roi, box):
                    continue
                rid = self._slots[s].rid
                ov = tuple((max(a0, b0), min(a1, b1))
                           for (a0, a1), (b0, b1) in zip(sm.roi, box))
                w = work[tuple(slice(a - s0, b - s0)
                               for (a, b), s0 in zip(ov, start))]
                m = sm.mask[tuple(slice(a - r0, b - r0)
                                  for (a, b), (r0, _r1) in zip(ov, sm.roi))]
                w[w == rid] = 0
                w[m & (w == 0)] = rid
            if self._outline is not None:
                work[~self._outline[sl]] = 0
            prev = self._final[sl]
            d = prev != work
            if d.any():
                changed |= set(int(v) for v in np.unique(prev[d]) if v)
                changed |= set(int(v) for v in np.unique(work[d]) if v)
                diff_box = _union(diff_box, _bbox_of(d, start))
                self._final[sl] = work
        # ---- sizes of every id a re-write changed ------------------------ #
        slot_of = {self._slots[s].rid: s for s in live}
        for rid in changed:
            s = slot_of.get(rid)
            n = 0
            if s is not None:
                box = _union(self._smoothed[s].roi, self._slots[s].rbox)
                n = int(np.count_nonzero(self._final[_slices(box)] == rid))
            if n:
                self._final_ids[rid] = n
            else:
                self._final_ids.pop(rid, None)
        self._finish_key = key
        # ---- the edge flags, on the raw cells ---------------------------- #
        if clear_border and len(live) > 1:
            m = max(1, int(edge_margin))
            open_set = set(BOX_FACES) if whole_patch else set()
            for s in live:
                box = self._slots[s].rbox
                for axis, (lo, hi) in enumerate(box):
                    low, high = BOX_FACES[2 * axis], BOX_FACES[2 * axis + 1]
                    if ((low not in open_set and lo < m) or
                            (high not in open_set
                             and hi > self.shape[axis] - m)):
                        out.edge_ids.add(self._slots[s].rid)
                        break
        after = set(self._final_ids)
        if auto_smooth:
            out.n_emptied_by_smooth = len(set(rids) - after)
            out.edge_ids &= after
        out.ids = sorted(after)
        out.sizes = {i: self._final_ids[i] for i in out.ids}
        out.small_ids = small_piece_ids(out.sizes)
        view = self._final.view()
        view.setflags(write=False)
        out.result = view
        out.changed_ids = sorted(changed)
        out.diff_box = None if diff_box is None else _slices(diff_box)
        out.seconds = time.perf_counter() - t0
        return out

    # ---- for tests and the page ------------------------------------------ #
    def raw_result(self):
        """The raw cells in reserved ids (a fresh array): the full run's
        engine result after the reserved-id mapping, before the size floor
        and the smoothing."""
        lut = {s: self._slots[s].rid for s in self._order}
        top = max(lut.values()) if lut else 0
        dtype = np.int32 if top <= np.iinfo(np.int32).max else np.int64
        out = np.zeros(self.shape, dtype)
        for s, rid in lut.items():
            box = self._slots[s].rbox
            if box is None:
                continue
            v = out[_slices(box)]
            v[self._raw[_slices(box)] == s] = rid
        return out


def prepare(labels, images, spec, *, roi, dark_channels=None,
            dark_threshold=None, spacing=(1, 1, 1), elevation=None,
            outline=None, cancel_event=None, allow_flat=False):
    """Prepare the box once per box and settings: the same inputs as
    :func:`.region.reshape_empty_region` (box crops, the wall picture in
    ``elevation=`` when the page holds it, the patch's outline in
    ``outline=``), with no clicks.

    ``spec`` must be the region target with ``seeds="paint"``,
    ``params["growth"] == "flood"`` and compactness 0 (a ``ValueError``
    otherwise; :func:`is_click_fill` says beforehand).  Returns
    ``(stage, None)`` -- a :class:`CellsClickFill` -- or ``(None,
    (reason, sentence))``: the full run's own refusals before any marker
    (``"too-big"``, ``"outline-per-box"``), or :data:`FULL_RUN` when the
    box has flat spots the rule keeps whole (:func:`flat_survey`) -- then
    the caller runs the
    full Preview and shows nothing (the sentence is for the log).

    ``allow_flat`` -- the user asked for the click fill whatever the flat
    spots: the stage is returned all the same, its ``flat_override`` holding
    the survey's reason (``None`` when there was none) and
    ``n_flat_pairs`` the count as always.  A tie there may then be broken
    differently from the full run, by a few voxels."""
    spacing = _spacing(spacing)
    _check_spec(spec)
    if spec.seeds != "paint":
        raise ValueError("the click fill takes seeds='paint' only (got "
                         "{!r}).".format(spec.seeds))
    p = spec.params
    if p.get("growth", "flood") != "flood":
        raise ValueError(
            "the click fill takes growth='flood' only (got {!r}); the other "
            "growth rules keep the full run.".format(p.get("growth")))
    if spec.walls not in CLICK_FILL_WALLS:
        raise ValueError(
            "the click fill takes the walls {} only (got {!r}); the others "
            "keep the full run.".format(", ".join(CLICK_FILL_WALLS),
                                        spec.walls))
    if float(p.get("compactness", 0.0) or 0.0) != 0.0:
        raise ValueError(
            "the click fill floods with compactness 0 only (got {!r}); a "
            "compact watershed keeps the full run.".format(p.get("compactness")))
    _check_outline_arg(spec, outline)
    sink = MessageSink(None)
    try:
        shape, existing = _prepare(labels, spec, roi, spacing, cancel_event)
        _outline_refusal(spec, outline)
    except _Refused as refusal:
        return None, (refusal.reason, refusal.message)
    whole = np.ones(shape, bool)
    elev, _clahe, _ridge, _img = _build_wall_stage(
        spec, images=_wall_images(spec, images, roi, shape, elevation),
        roi=roi, region=whole, spacing=spacing, params=p, sink=sink,
        supplied_elevation=elevation, cancel_event=cancel_event)
    del _clahe, _ridge, _img, whole
    _check_cancel(cancel_event)
    dark_raw, outline_box = _dark_and_outline(
        spec, outline=outline, roi=roi, shape=shape,
        dark_channels=dark_channels, dark_threshold=dark_threshold,
        spacing=spacing, cancel_event=cancel_event)
    _check_cancel(cancel_event)
    cores, _n = eroded_cores(existing, erode_um=p["core_erode_um"],
                             spacing=spacing, cancel_event=cancel_event)
    _check_cancel(cancel_event)
    why, low, other_patches, n_flat = flat_survey(
        elev, existing, cores, dark_raw=dark_raw, outline=outline_box,
        clearance_um=p["dark_clearance_um"], spacing=spacing,
        faces=bool(p.get("faces_outside", True)))
    if why is not None and not allow_flat:
        return None, (FULL_RUN, FLAT_SENTENCE.format(why=why))
    _check_cancel(cancel_event)
    stage = CellsClickFill(spec=spec, roi=roi, spacing=spacing, shape=shape,
                           existing=existing, elevation=elev,
                           dark_raw=dark_raw, outline=outline_box,
                           cores=cores, messages=sink.messages, low=low,
                           other_patches=other_patches, n_flat_pairs=n_flat)
    stage.flat_override = why
    return stage, None


def is_click_fill(spec) -> bool:
    """True when ``spec`` is one this module redoes by window: the region
    target, click markers, the fill to the walls, a wall picture of
    :data:`CLICK_FILL_WALLS`.  (A box of such a spec may still be sent to
    the full run by :func:`prepare`, for flat spots.)"""
    return (isinstance(spec, ReshapeSpec) and spec.target == "region"
            and spec.seeds == "paint" and spec.walls in CLICK_FILL_WALLS
            and spec.params.get("growth", "flood") == "flood"
            and float(spec.params.get("compactness", 0.0) or 0.0) == 0.0)



#: The log line when a box goes to the full run for its flat spots.
FLAT_SENTENCE = ("[ADD] click-by-click fill not used on this box: {why}; "
                 "the fill runs whole")

#: The log line of a step or finish on a stage a Stop or an error cut short.
STAGE_BROKEN = ("the click fill was interrupted on this box and is half "
                "updated; prepare the box again")


def _outside_alone(shape, existing, dark_raw, outline, clearance_um, spacing,
                   faces=True):
    """The outside marker with no click: the box faces (``faces``), the
    eroded dark, outside the outline -- on empty voxels."""
    out = np.zeros(shape, bool)
    for axis in range(3 if faces else 0):
        idx = [slice(None)] * 3
        idx[axis] = 0
        out[tuple(idx)] = True
        idx[axis] = -1
        out[tuple(idx)] = True
    if dark_raw is not None:
        if clearance_um > 0:
            from .arbiter import physical_ball

            out |= ndi.binary_erosion(dark_raw, structure=physical_ball(
                clearance_um, spacing), border_value=int(not faces))
        else:
            out |= dark_raw
    if outline is not None:
        out |= ~outline
    return out & (np.asarray(existing) == 0)


def flat_survey(elevation, existing, cores, *, dark_raw, outline,
                clearance_um, spacing, faces=True):
    """``(why, low_patches, other_patches, n_flat_pairs)``: why the box must
    keep the full run (``None``: it need not); the flat patches at the
    box's lowest wall value, numbered (int32, 0 elsewhere); the numbers of
    those that hold a marker that is not a click (the outside, a kept
    object's core); the touching equal-valued pairs between the clips.
    The middle two serve the per-click check.

    The flood breaks a tie (two fronts reaching a voxel at exactly one
    level) by order of arrival, and a window reproduces that order only
    when it comes from distinct levels.  Flat spots -- face-touching voxels
    with exactly the same wall value -- are where it may not:

    * flat spots at a value between the box's lowest and highest (where
      cells can grow, widened by the dark clearance and one voxel): a tie
      there can take its order from a tie one level lower.  The box keeps
      the full run when such pairs number more than
      :data:`FLAT_PAIRS_PER_VOXEL` of the voxels where cells can grow, or
      when one spot they join holds more than :data:`FLAT_SPOT_MAX_VOXELS`
      voxels; fewer, smaller ones are coincidences, let through and
      counted;
    * a flat patch at the lowest value holding two markers of different
      labels -- two clicks, or a click and the outside or a kept core:
      markers of equal height leave the flood in an order set by every
      marker in the box.  Clicks are not known here, so the click that
      pairs them is checked (:meth:`CellsClickFill.step` then floods the
      whole box, once; see the module notes for what a later click may
      leave).

    The lowest and highest values are the wall picture's clips (a
    normalised picture puts its darkest 1 % at exactly 0 and its brightest
    0.5 % at exactly 1): flat patches there are the normal state.  At the
    top every front arrives from a lower level; at the bottom only markers
    on one patch tie (above)."""
    e = np.asarray(elevation, np.float32)
    if not e.size:
        return None, np.zeros(e.shape, np.int32), frozenset(), 0
    outside = _outside_alone(e.shape, existing, dark_raw, outline,
                             clearance_um, spacing, faces=faces)
    grow = (np.asarray(existing) == 0) & ~outside
    reach = int(np.ceil(float(clearance_um) / min(spacing))) + 1
    grow = ndi.binary_dilation(grow, iterations=reach)
    n_grow = int(np.count_nonzero(grow))
    limit = FLAT_PAIRS_PER_VOXEL * n_grow
    lo, hi = float(e.min()), float(e.max())
    mid = grow & (e > lo) & (e < hi)
    n_mid = 0
    pairs = []
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis] = slice(0, -1)
        b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        hit = (e[a] == e[b]) & mid[a] & mid[b]
        n = int(np.count_nonzero(hit))
        n_mid += n
        if n and n_mid <= limit:
            at = np.nonzero(hit)
            nxt = list(at)
            nxt[axis] = at[axis] + 1
            pairs.append((np.ravel_multi_index(at, e.shape),
                          np.ravel_multi_index(tuple(nxt), e.shape)))
    low, _n = ndi.label(e == lo)
    low = low.astype(np.int32, copy=False)
    others = outside | (np.asarray(cores) != 0)
    other_patches = frozenset(int(v) for v in np.unique(low[others & (low > 0)]))
    if n_mid > limit:
        return ("{} pairs of touching voxels share one wall value between "
                "the lowest and the highest ({:.1f} per 100,000 voxels where "
                "cells can grow; over {:g} keeps the full run)".format(
                    n_mid, 1e5 * n_mid / max(1, n_grow),
                    1e5 * FLAT_PAIRS_PER_VOXEL)), low, other_patches, n_mid
    biggest = _largest_flat_spot(pairs)
    if biggest > FLAT_SPOT_MAX_VOXELS:
        return ("{} touching voxels share one wall value between the lowest "
                "and the highest (a flat spot over {} voxels keeps the full "
                "run)".format(biggest, FLAT_SPOT_MAX_VOXELS),
                low, other_patches, n_mid)
    return None, low, other_patches, n_mid


def _largest_flat_spot(pairs):
    """The voxels of the largest flat spot joined by ``pairs`` (a list of
    ``(flat indices a, flat indices b)``), 0 with none."""
    if not pairs:
        return 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    a = np.concatenate([p[0] for p in pairs])
    b = np.concatenate([p[1] for p in pairs])
    nodes, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    k = a.size
    graph = coo_matrix((np.ones(k, np.int8), (inv[:k], inv[k:])),
                       shape=(nodes.size, nodes.size))
    _n, comp = connected_components(graph, directed=False)
    return int(np.bincount(comp).max())

"""Nuclei inside each cell of another run -- the Add tab's "Inside each cell"
recipe ("nuclei first, then owners").

Pure algorithm module -- no Qt, no napari.  scipy and skimage are imported
lazily inside functions.

Given a nuclear channel and a cells labelling of the same voxels (another
run of the patch), it finds every nucleus by its own edge, gives each one
whole to the cell that holds most of it, and then applies Add's rules
against the objects already in the run being edited.  Every length is in
micrometres and is converted per axis from ``spacing_um`` (z, y, x), so
the recipe behaves the same at any voxel size, anisotropic included.

The steps (:func:`nuclei_in_cells`):

1. Smooth: ``g`` = Gaussian 0.5 um, ``g2`` = Gaussian 1.0 um.
2. Patch levels (:func:`patch_levels`): ``otsu`` = Otsu of ``g`` at a voxel
   stride of 2; ``background_z`` = per z slice, the median of ``g`` inside
   the cells where ``g < otsu``, from slices with at least 500 um^2 of such
   area, interpolated across z and smoothed 1.5 um along z.
3. Centres: a scale-normalised Laplacian of Gaussian in physical units
   (each second derivative divided by its axis spacing squared) at
   0.5 r and 0.7 r, the larger of the two; local maxima where
   ``g >= 0.6 otsu``, then any centre closer than 0.6 r (largest per-axis
   distance, in um) to a stronger one is dropped; only centres where the
   filter is positive are kept.
4. Own edge: inside ``g >= 0.35 otsu`` each centre floods its basin of
   ``-g2``; its peak is the brightest ``g`` within 1 um; its edge level is
   ``background + edge * (peak - background)``, never below 0.35 otsu; a
   centre whose peak rises less than 0.25 otsu over the background is
   refused; the nucleus is the basin voxels at or above the level that
   connect to the centre.  Nuclei under the size floor are dropped.
5. Owners: each nucleus goes whole to the cell holding most of it, and is
   kept only when at least half of it lies inside the cells.
6a. A nucleus with at least 30 % in a second cell that would otherwise be
   empty is split at the border between the two cells.
6b. Two in a cell: inside the cell and its nuclei padded by 1 um, the
   nuclei flood ``-g2``; a pair merges when the brightest pass between
   them is at least ``gap`` x the dimmer one's peak; pairs kept apart are
   reported.
6c. Rescue: each still-empty cell runs step 4 inside that cell only, from
   its own centres (a centre in its 1 um border shell belongs to a
   neighbour), the cell border a hard wall.
7. Tidy: each nucleus is cut at its owner's border (more than a quarter
   lost flags it "cut at the cell edge"); then per nucleus: holes filled,
   an opening of radius round(0.5 um / spacing) per axis (an axis that
   rounds to 0 is left out; no opening when every axis does), the largest
   piece kept, pieces under the size floor dropped.  When cut at the
   border, the tidy stays inside the owner: a filled hole that is another
   cell (or no cell) is cut out again.
8. Second-nucleus guard: in a cell with two or more nuclei, any nucleus
   other than the largest that is under ``second_min_um3`` is dropped
   (also when it touches the patch face).

Then Add's rules, when ``target`` (the run being edited) is given: a cell
that already owns a kept object (by majority) is skipped (on a box the
caller can measure that over the whole patch, :func:`owning_cells`, and
pass it as ``owning``: a box sees only its crop of a cell); a found nucleus
that one kept object covers by at least a quarter is dropped; only empty
target voxels are written; a trimmed nucleus keeps its largest piece and is
dropped below the size floor, and the second-nucleus guard runs once more
(what these drop counts in ``n_all_taken``).

Stop: ``cancel_event`` is polled between the six filter passes, between
steps, between the two id compactions and every :data:`CANCEL_EVERY`
items inside the three loops (cells, empty cells, nuclei); a
Stop RAISES :class:`~.reshape.engine.ReshapeCancelled`, as the region
engine does, and the Add worker answers it as ``cancelled``.  The step-4
flood is one call and cannot be stopped part-way; its stage text says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DeriveResult", "PatchLevels", "nuclei_in_cells", "owning_cells",
    "patch_levels",
    "NUC_RADIUS_UM", "NUC_RADIUS_RANGE", "NUC_EDGE", "NUC_EDGE_RANGE",
    "TWO_GAP", "TWO_GAP_RANGE", "SECOND_MIN_UM3", "MIN_NUCLEUS_UM3",
    "CUT_FLAG_SHARE", "KEPT_SHARE", "CANCEL_EVERY", "DERIVE_STAGES",
    "DERIVE_LOOP_CELLS_WITH", "DERIVE_LOOP_EMPTY_CELLS", "DERIVE_LOOP_NUCLEI",
]

# --------------------------------------------------------------------------- #
# the user's three knobs (defaults and ranges) and the fixed recipe constants
# --------------------------------------------------------------------------- #
#: Nucleus radius, um: the size of the bright spots the finder starts from
NUC_RADIUS_UM = 2.5
NUC_RADIUS_RANGE = (1.0, 10.0)
#: Nucleus edge: the share of the way from the background up to the
#: nucleus's own peak where its edge is drawn
NUC_EDGE = 0.25
NUC_EDGE_RANGE = (0.05, 0.90)
#: Gap that splits two nuclei: two nuclei of one cell stay apart only when
#: the brightest pass between them is below this share of the dimmer peak
TWO_GAP = 0.70
TWO_GAP_RANGE = (0.30, 0.95)
#: a second (third...) nucleus of one cell under this volume is dropped
SECOND_MIN_UM3 = 40.0
#: the size floor of every nucleus
MIN_NUCLEUS_UM3 = 12.5
#: a nucleus that loses more than this share to its owner's border is
#: flagged "cut at the cell edge"
CUT_FLAG_SHARE = 0.25
#: a found nucleus that one kept object covers by at least this share is
#: dropped
KEPT_SHARE = 0.25
#: Stop is polled (and progress sent) every this many items in the loops
CANCEL_EVERY = 200

_SMOOTH_UM = 0.5             # step 1, g
_SMOOTH2_UM = 1.0            # step 1, g2
_LEVEL_STRIDE = 2            # step 2, the Otsu sample (voxels)
_BZ_AREA_UM2 = 500.0         # step 2, least cytoplasm area per slice
_BZ_ZSIGMA_UM = 1.5          # step 2, smoothing along z
_LOG_SCALES = (0.5, 0.7)     # step 3, x radius
_CENTRE_SPACING = 0.6        # step 3, x radius
_GATE = 0.6                  # step 3, x otsu
_FLOOR = 0.35                # step 4, x otsu
_MIN_CONTRAST = 0.25         # step 4, x otsu
_PEAK_WINDOW_UM = 1.0        # step 4
_IN_CELLS_MIN = 0.5          # step 5
_CROSSING_SHARE = 0.10       # step 6a, a nucleus counted as crossing
_SPLIT_SHARE = 0.3           # step 6a
_PAD_UM = 1.0                # step 6b
_BORDER_UM = 1.0             # step 6c
_OPEN_UM = 0.5               # step 7
#: ids up to this are compacted with one lookup table; larger ids are
#: compacted by sorting the non-zero values (a 2**31 - 1 id must not make
#: a 16 GiB table)
_LUT_MAX_ID = 1 << 22
#: ids are carried as int64 here: a uint64 id at or above this is refused
_MAX_ID = 1 << 63

# --------------------------------------------------------------------------- #
# the stage texts (plain words; ``progress(text, i, n)``)
# --------------------------------------------------------------------------- #
DERIVE_STAGE_READ = "Reading the cells and the objects already here..."
DERIVE_STAGE_SMOOTH = "Smoothing the nuclear channel..."
DERIVE_STAGE_LEVELS = "Measuring the patch's brightness..."
DERIVE_STAGE_CENTRES = "Finding nuclei..."
DERIVE_STAGE_EDGES = ("Drawing each nucleus's edge (Stop takes effect "
                      "after this step)...")
DERIVE_STAGE_OWNERS = "Giving each nucleus to a cell..."
DERIVE_STAGE_SPLIT = "Splitting nuclei shared by two cells..."
DERIVE_STAGE_TWO = "Checking cells that hold two nuclei..."
DERIVE_STAGE_RESCUE = "Looking again inside empty cells..."
DERIVE_STAGE_TIDY = "Tidying the nuclei..."
DERIVE_STAGE_KEPT = "Leaving out what is already here..."
#: the three loops, each saying what ``k`` counts (step 6b: cells that
#: hold a nucleus; 6c: cells still empty; 7: nuclei)
DERIVE_LOOP_CELLS_WITH = "Cell {i} of {k} with a nucleus · {stage}"
DERIVE_LOOP_EMPTY_CELLS = "Empty cell {i} of {k} · {stage}"
DERIVE_LOOP_NUCLEI = "Nucleus {i} of {k} · {stage}"

DERIVE_STAGES = (
    DERIVE_STAGE_READ,
    DERIVE_STAGE_SMOOTH, DERIVE_STAGE_LEVELS, DERIVE_STAGE_CENTRES,
    DERIVE_STAGE_EDGES, DERIVE_STAGE_OWNERS, DERIVE_STAGE_SPLIT,
    DERIVE_STAGE_TWO, DERIVE_STAGE_RESCUE, DERIVE_STAGE_TIDY,
    DERIVE_STAGE_KEPT,
)


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PatchLevels:
    """Step 2's two levels.  ``background_z`` has one value per z slice of
    the array it is used with (crop it with :meth:`crop_z` for a box)."""

    otsu: float
    background_z: np.ndarray

    def crop_z(self, start: int, stop: int) -> "PatchLevels":
        return PatchLevels(float(self.otsu),
                           np.asarray(self.background_z[start:stop],
                                      np.float32).copy())


@dataclass
class DeriveResult:
    """One run of :func:`nuclei_in_cells`.

    ``labels`` holds ids ``1..N`` (int32, compact), ordered by owner cell id,
    then by size (largest first).  Cell ids are in the caller's numbering.
    """

    labels: np.ndarray
    owner: dict = field(default_factory=dict)          # {id: cell id}
    two_in_cell: dict = field(default_factory=dict)    # {cell id: [ids]}
    cut_at_edge: set = field(default_factory=set)      # {ids}
    #: {id: share of the nucleus its owner's border cut off} (ids that
    #: lost any; the flag above is this share over CUT_FLAG_SHARE)
    cut_share: dict = field(default_factory=dict)
    n_cells: int = 0
    n_no_nucleus: int = 0       # cells (not skipped) left with no nucleus
    n_outside_cells: int = 0    # nuclei with less than half inside cells
    n_cell_has_one: int = 0     # cells skipped: they own a kept object
    n_split_border: int = 0     # nuclei split between two cells
    n_cut_at_edge: int = 0      # nuclei flagged "cut at the cell edge"
    n_all_taken: int = 0        # found nuclei left out by the kept objects
                                # (a second nucleus the trim put under the
                                # guard volume included)
    n_found: int = 0            # nuclei the recipe found before Add's rules
    n_guard_removed: int = 0    # second nuclei under the guard volume
    n_rescued: int = 0          # nuclei found by the rescue step
    n_centres: int = 0          # step 3's centres
    otsu: float | None = None
    seconds: dict = field(default_factory=dict)

    @property
    def n_nuclei(self) -> int:
        return len(self.owner)

    @property
    def n_two_in_cell(self) -> int:
        return len(self.two_in_cell)

    @property
    def n_one_nucleus(self) -> int:
        """Cells (not skipped) given exactly one nucleus."""
        return (self.n_cells - self.n_cell_has_one - self.n_no_nucleus
                - self.n_two_in_cell)

    @property
    def all_taken(self) -> bool:
        """Nuclei were found, but every one sits on a kept object."""
        return self.n_found > 0 and self.n_nuclei == 0

    def counts(self) -> dict:
        return {k: int(getattr(self, k)) for k in (
            "n_cells", "n_no_nucleus", "n_outside_cells", "n_cell_has_one",
            "n_split_border", "n_cut_at_edge", "n_all_taken", "n_found",
            "n_guard_removed", "n_rescued", "n_centres")}

    def log_line(self) -> str:
        return ("[ADD-DERIVE] nuclei={} cells={} one={} two_in_cell={} "
                "counts={} otsu={} seconds={}".format(
                    self.n_nuclei, self.n_cells, self.n_one_nucleus,
                    self.n_two_in_cell, self.counts(),
                    None if self.otsu is None else round(self.otsu, 4),
                    self.seconds))


# --------------------------------------------------------------------------- #
# cancel and progress
# --------------------------------------------------------------------------- #
def _stop_requested(cancel_event) -> bool:
    if cancel_event is None:
        return False
    is_set = getattr(cancel_event, "is_set", None)
    if is_set is not None:
        return bool(is_set())
    return bool(cancel_event())


class _Run:
    """Stop polling and progress for one call."""

    def __init__(self, cancel_event, progress):
        self._cancel = cancel_event
        self._cb = progress
        self._i = 0
        self._stage = ""

    def poll(self):
        if _stop_requested(self._cancel):
            from .reshape.engine import ReshapeCancelled

            raise ReshapeCancelled()

    def _emit(self, text):
        if self._cb is None:
            return
        try:
            self._cb(text, self._i, len(DERIVE_STAGES))
        except Exception:
            pass

    def stage(self, text):
        """A stage STARTS: poll Stop, then report it."""
        self.poll()
        self._i += 1
        self._stage = text
        self._emit(text)

    def cell(self, i, k, template):
        """Inside a loop: every CANCEL_EVERY items, poll and report with
        ``template`` (one of the ``DERIVE_LOOP_*`` texts)."""
        if i % CANCEL_EVERY:
            return
        self.poll()
        self._emit(template.format(i=i, k=k, stage=self._stage))


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _check_labels(name, arr, shape):
    a = np.asarray(arr)
    if a.shape != shape:
        raise ValueError("{} has shape {} but the nuclear channel has shape "
                         "{}.".format(name, a.shape, shape))
    if a.dtype == bool or not np.issubdtype(a.dtype, np.integer):
        raise TypeError("{} must hold integer ids, not {}.".format(name, a.dtype))
    if np.issubdtype(a.dtype, np.signedinteger) and a.size and int(a.min()) < 0:
        raise ValueError("{} holds a negative id ({}).".format(name, int(a.min())))
    if a.dtype.kind == "u" and a.dtype.itemsize >= 8 and a.size:
        top = int(a.max())
        if top >= _MAX_ID:
            raise ValueError(
                "{} holds the id {}; ids of 2**63 or more cannot be used -- "
                "renumber that run first.".format(name, top))
    return a


def _check_channel(nuclear):
    """The nuclear channel as float32 (no copy when it already is)."""
    nuc = np.asarray(nuclear)
    if nuc.ndim != 3:
        raise ValueError("the nuclear channel must be 3D (z, y, x); it has "
                         "{} dimensions.".format(nuc.ndim))
    if 0 in nuc.shape:
        raise ValueError("the nuclear channel is empty (shape {}).".format(nuc.shape))
    if nuc.dtype == bool or not (np.issubdtype(nuc.dtype, np.integer)
                                 or np.issubdtype(nuc.dtype, np.floating)):
        raise TypeError("the nuclear channel must be numeric, not {}.".format(nuc.dtype))
    nuc = np.asarray(nuc, dtype=np.float32)
    if not np.isfinite(nuc).all():
        raise ValueError("the nuclear channel holds NaN or infinite values.")
    return nuc


def _check_spacing(spacing_um):
    msg = ("spacing_um must be three positive numbers (z, y, x) in um, "
           "not {!r}.".format(spacing_um))
    if isinstance(spacing_um, (str, bytes)):
        raise TypeError(msg)
    try:
        sp = tuple(float(s) for s in spacing_um)
    except (TypeError, ValueError):
        raise TypeError(msg) from None
    if len(sp) != 3 or not all(math.isfinite(s) and s > 0 for s in sp):
        raise ValueError(msg)
    return sp


def _number(name, value, need):
    """``float(value)`` with a plain TypeError for a non-number."""
    if isinstance(value, (str, bytes)) or value is None:
        raise TypeError("{} must be a number {}, not {!r}.".format(name, need, value))
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError("{} must be a number {}, not {!r}.".format(
            name, need, value)) from None


def _check_inputs(nuclear, cells, target, spacing_um, radius_um, edge, gap,
                  second_min_um3, min_um3):
    nuc = _check_channel(nuclear)
    cells = _check_labels("cells", cells, nuc.shape)
    if target is not None:
        target = _check_labels("target", target, nuc.shape)
    sp = _check_spacing(spacing_um)

    def _in(name, value, lo, hi):
        need = "between {} and {}".format(lo, hi)
        v = _number(name, value, need)
        if not (lo <= v <= hi):
            raise ValueError("{} must be {}, not {}.".format(name, need, value))
        return v

    radius_um = _in("radius_um", radius_um, *NUC_RADIUS_RANGE)
    edge = _in("edge", edge, *NUC_EDGE_RANGE)
    gap = _in("gap", gap, *TWO_GAP_RANGE)
    min_um3 = _number("min_um3", min_um3, "of zero or more")
    if not (math.isfinite(min_um3) and min_um3 >= 0):
        raise ValueError("min_um3 must be zero or more, not {}.".format(min_um3))
    if second_min_um3 is not None:
        second_min_um3 = _number("second_min_um3", second_min_um3,
                                 "of zero or more (or None)")
        if not (math.isfinite(second_min_um3) and second_min_um3 >= 0):
            raise ValueError("second_min_um3 must be zero or more (or None), "
                             "not {}.".format(second_min_um3))
    return nuc, cells, target, sp, radius_um, edge, gap, second_min_um3, min_um3


def _check_otsu(po):
    """The nuclear threshold must be a positive level: every gate below is a
    share of it."""
    if not (math.isfinite(po) and po > 0):
        raise ValueError(
            "The nuclear channel has no bright level above zero (its "
            "threshold is {:g}), so no nucleus can be found -- check that "
            "this is the nuclear channel.".format(po))


# --------------------------------------------------------------------------- #
# unit helpers
# --------------------------------------------------------------------------- #
def _sig(um, sp):
    return tuple(float(um) / s for s in sp)


def _halfw(um, sp):
    """A box half-width per axis: round(um / spacing), at least 1."""
    return tuple(max(1, int(round(um / s))) for s in sp)


def _l1_ball_um(r_um, sp):
    """{d : sum_a |d_a| * sp_a <= r_um}."""
    h = [int(np.floor(r_um / s + 1e-9)) for s in sp]
    zz, yy, xx = np.meshgrid(*[np.arange(-k, k + 1) for k in h], indexing="ij")
    return (np.abs(zz) * sp[0] + np.abs(yy) * sp[1]
            + np.abs(xx) * sp[2]) <= r_um + 1e-9


def _opening_footprint(sp):
    """The tidy opening: an L1 ellipsoid of radius round(0.5 um / spacing)
    per axis; an axis that rounds to 0 is left out; None when every axis
    does (no opening)."""
    radii = [int(round(_OPEN_UM / s)) for s in sp]
    if not any(radii):
        return None
    grids = np.meshgrid(*[np.arange(-k, k + 1) for k in radii], indexing="ij")
    acc = np.zeros(grids[0].shape)
    for g, k in zip(grids, radii):
        if k:
            acc = acc + np.abs(g) / k
    return acc <= 1 + 1e-9


def _compact(labels):
    """Ids -> ``1..N`` in ascending id order.  Returns ``(int32, ids)`` with
    ``ids[k - 1]`` the original id of ``k``.  Small ids go through one
    lookup table; huge ids through a sort of the non-zero values."""
    lab = np.asarray(labels)
    top = int(lab.max()) if lab.size else 0
    if top == 0:
        return np.zeros(lab.shape, np.int32), np.zeros(0, np.int64)
    if top <= _LUT_MAX_ID:
        flat = lab.ravel()
        if not np.can_cast(flat.dtype, np.intp):
            flat = flat.astype(np.intp)    # uint64; every value <= top
        present = np.bincount(flat, minlength=top + 1) > 0
        present[0] = False
        ids = np.flatnonzero(present)
        lut = np.zeros(top + 1, np.int32)
        lut[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
        return lut[lab], ids.astype(np.int64)
    nz = lab != 0
    vals = lab[nz]
    ids = np.unique(vals)
    out = np.zeros(lab.shape, np.int32)
    out[nz] = (np.searchsorted(ids, vals) + 1).astype(np.int32)
    return out, ids.astype(np.int64)


def _overlap_table(a, b):
    """``{a id: {b id: voxels}}`` for ``a > 0`` (``b`` may be 0); both small
    non-negative ints."""
    m = a > 0
    av = a[m].astype(np.int64)
    bv = b[m].astype(np.int64)
    M = int(b.max()) + 1 if b.size else 1
    k, n = np.unique(av * M + bv, return_counts=True)
    t = {}
    for x, y, c in zip((k // M).tolist(), (k % M).tolist(), n.tolist()):
        t.setdefault(x, {})[y] = c
    return t


def _per_cell_count(n_cells, owner):
    cnt = {c: 0 for c in range(1, n_cells + 1)}
    for _x, c in owner.items():
        if c in cnt:
            cnt[c] += 1
    return cnt


def _lut_of(own, n):
    lut = np.zeros(n + 1, np.int64)
    for k, v in own.items():
        lut[k] = v
    return lut


# --------------------------------------------------------------------------- #
# step 2: the patch levels
# --------------------------------------------------------------------------- #
def _background_z(g, cells, po, sp):
    from scipy import ndimage as ndi

    Z = g.shape[0]
    b = np.full(Z, np.nan)
    min_vox = _BZ_AREA_UM2 / (sp[1] * sp[2])
    for z in range(Z):
        v = g[z][(cells[z] > 0) & (g[z] < po)]
        if v.size >= min_vox:
            b[z] = np.median(v)
    ok = np.isfinite(b)
    if ok.any():
        zz = np.arange(Z)
        b = np.interp(zz, zz[ok], b[ok])
    else:
        # no slice has enough cytoplasm (a small box): one level for all
        pooled = g[(cells > 0) & (g < po)]
        if pooled.size:
            b[:] = float(np.median(pooled))
        else:
            inside = g[cells > 0]
            b[:] = float(inside.min()) if inside.size else float(g.min())
    return ndi.gaussian_filter1d(b, _BZ_ZSIGMA_UM / sp[0],
                                 mode="nearest").astype(np.float32)


def _otsu(g):
    from skimage.filters import threshold_otsu

    s = _LEVEL_STRIDE
    sample = g[::s, ::s, ::s]
    if float(sample.min()) == float(sample.max()):
        return float(sample.min())
    return float(threshold_otsu(sample))


def patch_levels(nuclear, cells, spacing_um) -> PatchLevels:
    """Step 2 on the whole patch (the box-info worker caches it per patch;
    crop it to a box with :meth:`PatchLevels.crop_z`).  A channel with no
    bright level above zero is refused HERE, in plain words, as the recipe
    itself would refuse it -- so the page says so before Preview, not as a
    failed Preview."""
    from scipy import ndimage as ndi

    nuc = _check_channel(nuclear)
    cells = _check_labels("cells", cells, nuc.shape)
    sp = _check_spacing(spacing_um)
    g = ndi.gaussian_filter(nuc, _sig(_SMOOTH_UM, sp))
    po = _otsu(g)
    _check_otsu(float(po))
    return PatchLevels(po, _background_z(g, cells, po, sp))


# --------------------------------------------------------------------------- #
# step 3: centres
# --------------------------------------------------------------------------- #
def _log_max(nuc, radius_um, sp, run):
    """max over the two scales of the physical scale-normalised LoG:
    -s^2 * sum_a d2/dx_a^2, each derivative in um (divided by spacing_a^2).
    Stop is polled before each of the six derivative passes."""
    from scipy import ndimage as ndi

    best = None
    for frac in _LOG_SCALES:
        s_um = frac * radius_um
        sv = _sig(s_um, sp)
        out = None
        for a in range(3):
            run.poll()
            order = [0, 0, 0]
            order[a] = 2
            d = ndi.gaussian_filter(nuc, sv, order=order)
            d *= np.float32(1.0 / sp[a] ** 2)
            if out is None:
                out = d
            else:
                out += d
            del d
        out *= np.float32(-(s_um ** 2))
        if best is None:
            best = out
        else:
            np.maximum(best, out, out=best)
        del out
    return best


def _bbox(mask):
    sl = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        hit = np.flatnonzero(mask.any(axis=other))
        sl.append(slice(int(hit[0]), int(hit[-1]) + 1))
    return tuple(sl)


def _thin_points(phys, d):
    """Greedy, strongest first: drop every point closer than ``d`` (largest
    per-axis distance) to a point already kept.  Returns the keep mask."""
    n = len(phys)
    rejected = np.zeros(n, bool)
    if n == 0:
        return ~rejected
    from scipy.spatial import cKDTree

    near = cKDTree(phys).query_ball_point(phys, r=d, p=np.inf)
    for i in range(n):
        if rejected[i]:
            continue
        cand = [j for j in near[i] if j != i]
        if not cand:
            continue
        cand = np.asarray(cand, np.intp)
        dist = np.abs(phys[cand] - phys[i]).max(axis=1)
        rejected[cand[dist < d]] = True
    return ~rejected


def _centres(L, cand, sp, md_um):
    """Local maxima of ``L`` inside ``cand`` (box half-width
    max(1, floor(md / sp)) per axis), strongest first, thinned to ``md_um``
    apart."""
    from scipy import ndimage as ndi

    if not cand.any():
        return np.zeros((0, 3), np.intp)
    threshold = L.min()
    roi = _bbox(cand)
    lm = cand[roi]
    img = L[roi].copy()
    img[~lm] = np.finfo(L.dtype).min
    # at least one voxel per axis: a window of 0 would make every voxel a
    # maximum (a small radius at 1 um voxels found hundreds of centres)
    hw = [max(1, int(np.floor(md_um / s + 1e-9))) for s in sp]
    imax = ndi.maximum_filter(img, size=tuple(2 * h + 1 for h in hw),
                              mode="nearest")
    out = img == imax
    del imax
    out &= img > threshold
    coord = np.nonzero(out)
    del out
    inten = img[coord]
    order = np.argsort(-inten, kind="stable")
    coord = np.transpose(coord)[order]
    if len(coord):
        keep = _thin_points(coord * np.asarray(sp, float), md_um)
        coord = coord[keep]
    return (coord + np.array([s.start for s in roi])).astype(np.intp)


# --------------------------------------------------------------------------- #
# step 4: each centre's own edge
# --------------------------------------------------------------------------- #
def _seed_peaks(g, pts, hw):
    out = np.zeros(len(pts), np.float32)
    for i, (z, y, x) in enumerate(pts):
        out[i] = g[max(z - hw[0], 0):z + hw[0] + 1,
                   max(y - hw[1], 0):y + hw[1] + 1,
                   max(x - hw[2], 0):x + hw[2] + 1].max()
    return out


def _own_edges(pts, g, g2, bz, mask, *, alpha, floor_abs, min_contrast, hw):
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    N = len(pts)
    out = np.zeros(g.shape, np.int32)
    if N == 0:
        return out
    mk = np.zeros(g.shape, np.int32)
    mk[tuple(pts.T)] = np.arange(1, N + 1)
    mk[~mask] = 0
    W = watershed(-g2, mk, mask=mask)
    del mk
    peak = np.zeros(N + 1, np.float32)
    peak[1:] = _seed_peaks(g, pts, hw)
    bk = np.zeros(N + 1, np.float32)
    bk[1:] = bz[pts[:, 0]]
    lev = bk + alpha * (peak - bk)
    lev = np.maximum(lev, floor_abs)
    ok = (peak - bk) >= min_contrast
    lev[0] = np.inf
    lev[~ok] = np.inf
    fg = (W > 0) & (g >= lev[W])
    lab = np.where(fg, W, 0)
    del W
    cc, _n = ndi.label(fg)
    del fg
    seed_cc = np.zeros(N + 1, np.int64)
    seed_cc[1:] = cc[tuple(pts.T)]
    good = (seed_cc[lab] == cc) & (lab > 0)
    out[good] = lab[good]
    return out


# --------------------------------------------------------------------------- #
# step 5 / 6
# --------------------------------------------------------------------------- #
def _assign(G, cells):
    owner = {}
    info = {}
    for x, row in _overlap_table(G, cells).items():
        tot = sum(row.values())
        cs = sorted(((v, k) for k, v in row.items() if k), reverse=True)
        if not cs or sum(v for v, _ in cs) < _IN_CELLS_MIN * tot:
            continue
        owner[x] = cs[0][1]
        info[x] = (cs[0][0] / tot, (cs[1][0] / tot) if len(cs) > 1 else 0.0)
    return owner, info


def _pair_passes(W, h):
    """``{(a, b): pass}`` = the max over face-adjacent voxel pairs (p in a,
    q in b) of ``min(h[p], h[q])``."""
    out = {}
    M = int(W.max()) + 1
    for ax in range(W.ndim):
        s0 = [slice(None)] * W.ndim
        s1 = [slice(None)] * W.ndim
        s0[ax] = slice(None, -1)
        s1[ax] = slice(1, None)
        a = W[tuple(s0)]
        b = W[tuple(s1)]
        m = (a > 0) & (b > 0) & (a != b)
        if not m.any():
            continue
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        v = np.minimum(h[tuple(s0)][m], h[tuple(s1)][m])
        key = lo * M + hi
        order = np.lexsort((v, key))
        key, v = key[order], v[order]
        last = np.r_[key[1:] != key[:-1], True]
        for k, val in zip(key[last].tolist(), v[last].tolist()):
            kk = (k // M, k % M)
            out[kk] = max(out.get(kk, -np.inf), float(val))
    return out


def _two_in_a_cell(nucl, owner, cells, h, *, ratio, pad, run):
    """Step 6b.  Returns ``(relabelled nuclei, owner, n kept apart)``."""
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    by_cell = {}
    for x, c in owner.items():
        by_cell.setdefault(c, []).append(x)
    lut = np.arange(int(nucl.max()) + 1, dtype=np.int32)
    objs = ndi.find_objects(nucl)
    cobjs = ndi.find_objects(cells)
    k = len(by_cell)
    for i, (c, xs) in enumerate(by_cell.items(), start=1):
        run.cell(i, k, DERIVE_LOOP_CELLS_WITH)
        if len(xs) < 2:
            continue
        sls = [cobjs[c - 1]] + [objs[x - 1] for x in xs]
        sl = tuple(slice(max(min(s[d].start for s in sls) - pad[d], 0),
                         min(max(s[d].stop for s in sls) + pad[d], nucl.shape[d]))
                   for d in range(3))
        sub = nucl[sl]
        mk = np.zeros(sub.shape, np.int32)
        for x in xs:
            mk[sub == x] = x
        dom = (cells[sl] == c) | (mk > 0)
        hs = h[sl]
        W = watershed(-hs, mk, mask=dom)
        sad = _pair_passes(W, hs)
        pk = {x: float(hs[mk == x].max()) for x in xs}
        parent = {x: x for x in xs}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for (a, b), s in sorted(sad.items(), key=lambda kv: -kv[1]):
            if a in pk and b in pk and s >= ratio * min(pk[a], pk[b]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
        for x in xs:
            lut[x] = find(x)
    new = lut[nucl]
    new_owner = {int(lut[x]): c for x, c in owner.items()}
    return new, new_owner


def _relabel_owner(G, own):
    lab, ids = _compact(G)
    lut = np.zeros(int(G.max()) + 1, np.int32)
    lut[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    return lab, {int(lut[k]): v for k, v in own.items()
                 if k < lut.size and lut[k] > 0}


def _rescue(pts, g, g2, cells, cobjs, po, bz, *, cells_list, forbid, sp, edge,
            minv, next_id, run):
    """Step 6c: the edge rule inside each still-empty cell only."""
    from scipy import ndimage as ndi

    pcell = cells[tuple(pts.T)] if len(pts) else np.zeros(0, np.int32)
    shell_fp = _l1_ball_um(_BORDER_UM, sp)
    hw = _halfw(_PEAK_WINDOW_UM, sp)
    out = {}
    nxt = next_id
    k = len(cells_list)
    for i, c in enumerate(cells_list, start=1):
        run.cell(i, k, DERIVE_LOOP_EMPTY_CELLS)
        sl = cobjs[c - 1]
        if sl is None:
            continue
        P = pts[pcell == c]
        if len(P) == 0:
            continue
        cell = cells[sl] == c
        if forbid is not None:
            P = P[~forbid[tuple(P.T)]]
            if len(P) == 0:
                continue
        loc = P - np.array([s.start for s in sl])
        inner = ndi.binary_erosion(cell, structure=shell_fp, border_value=0)
        ins = inner[tuple(loc.T)]
        if ins.any():
            loc = loc[ins]
        mask = cell & (g[sl] >= _FLOOR * po)
        if forbid is not None:
            mask &= ~forbid[sl]
        lab = _own_edges(loc, g[sl], g2[sl], bz[sl[0]], mask, alpha=edge,
                         floor_abs=_FLOOR * po, min_contrast=_MIN_CONTRAST * po,
                         hw=hw)
        if not ins.any():
            sz = np.bincount(lab.ravel())
            sz[0] = 0
            if sz.max() == 0:
                continue
            kk = int(sz.argmax())
            if (inner & (lab == kk)).sum() < 0.5 * sz[kk]:
                continue
            lab = np.where(lab == kk, 1, 0)
        for kk in np.unique(lab[lab > 0]):
            m = lab == kk
            if m.sum() >= minv:
                out[nxt] = (c, sl, m)
                nxt += 1
    return out


def _largest_piece(m):
    from scipy import ndimage as ndi

    cc, n = ndi.label(m)
    if n > 1:
        m = cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)
    return m


def _tidy(lab, fp, minv, run, *, cells=None, owner_lut=None):
    """Per nucleus: fill holes, open (when ``fp``), keep the largest piece,
    drop it under ``minv`` voxels.  With ``cells`` and ``owner_lut`` (the
    cut mode) every written voxel stays inside the nucleus's owner cell:
    a filled hole that is another cell, or no cell, is cut out again (then
    the largest piece and the floor once more)."""
    from scipy import ndimage as ndi

    out = np.zeros_like(lab)
    pads = [max(1, k // 2) for k in fp.shape] if fp is not None else [1, 1, 1]
    slices = ndi.find_objects(lab)
    k = len(slices)
    for i, sl in enumerate(slices, start=1):
        run.cell(i, k, DERIVE_LOOP_NUCLEI)
        if sl is None:
            continue
        sl = tuple(slice(max(s.start - pd, 0), s.stop + pd)
                   for s, pd in zip(sl, pads))
        m = lab[sl] == i
        m = ndi.binary_fill_holes(m)
        if fp is not None:
            m2 = ndi.binary_opening(m, fp)
            if m2.sum() >= minv:
                m = m2
        m = _largest_piece(m)
        if cells is not None:
            inside = m & (cells[sl] == owner_lut[i])
            if not np.array_equal(inside, m):
                m = _largest_piece(inside)
        if m.sum() < minv:
            continue
        o = out[sl]
        o[m & ((lab[sl] == i) | (lab[sl] == 0)) & (o == 0)] = i
    return out


def _guard(G, own, guard_um3, sp):
    """In every cell with 2+ nuclei, drop any nucleus other than the cell's
    largest that is under ``guard_um3``."""
    vox = float(np.prod(sp))
    vol = np.bincount(G.ravel()) * vox
    by = {}
    for x, c in own.items():
        by.setdefault(c, []).append(x)
    removed = set()
    for _c, xs in by.items():
        if len(xs) < 2:
            continue
        big = max(xs, key=lambda x: vol[x])
        for x in xs:
            if x != big and vol[x] < guard_um3:
                removed.add(x)
    if removed:
        lut = np.arange(int(G.max()) + 1, dtype=G.dtype)
        lut[list(removed)] = 0
        G = lut[G]
        own = {k: v for k, v in own.items() if k not in removed}
    return G, own, len(removed)


# --------------------------------------------------------------------------- #
# Add's rules against the kept objects
# --------------------------------------------------------------------------- #
def _majority_owners(t, c, n_t=None, n_c=None):
    """The cells that own a kept object: each kept object's owner is the
    cell holding most of it, when at least half of it lies inside the cells
    (ties -> the higher cell id).

    ``t`` / ``c``: the kept object id and the cell id (0 = no cell) of
    every kept voxel, 1-D.  With ``n_t`` / ``n_c`` (each id's maximum + 1,
    both small) the ids are used as they are; without, they are compacted
    first (any ids, uint32 and uint64 included).  Returns the owning cells
    in the numbering of ``c``."""
    if n_t is None or n_c is None:
        tu, t = np.unique(t, return_inverse=True)
        cu, c = np.unique(c, return_inverse=True)
        n_t, n_c = int(tu.size), int(cu.size)
        cell_of = cu.astype(np.int64)
    else:
        cell_of = None
    if n_t == 0:
        return set()
    keys, n = np.unique(np.asarray(t).astype(np.int64) * n_c
                        + np.asarray(c).astype(np.int64), return_counts=True)
    e = keys // n_c
    k = keys % n_c
    tot = np.bincount(e, weights=n, minlength=n_t)
    inside = (k > 0) if cell_of is None else (cell_of[k] > 0)
    ins = np.bincount(e[inside], weights=n[inside], minlength=n_t)
    if not inside.any():
        return set()
    ei, ki, ni = e[inside], k[inside], n[inside]
    order = np.lexsort((ki, ni, ei))
    es = ei[order]
    last = np.r_[es[1:] != es[:-1], True]
    best_e = es[last]
    best_k = ki[order][last]
    hold = ins[best_e] >= _IN_CELLS_MIN * tot[best_e]
    if cell_of is not None:
        best_k = cell_of[best_k]
    return {int(x) for x in best_k[hold]}


def _member_mask(arr, ids):
    """``arr`` holds one of ``ids`` (sorted positive int64), voxel by
    voxel: one lookup table when ``arr``'s ids are small, else a sort."""
    if ids.size == 0 or arr.size == 0:
        return np.zeros(arr.shape, bool)
    top = int(arr.max())
    if top <= _LUT_MAX_ID:
        lut = np.zeros(top + 1, bool)
        lut[ids[ids <= top]] = True
        a = arr if np.can_cast(arr.dtype, np.intp) else arr.astype(np.intp)
        return lut[a]
    return np.isin(arr, ids)


def owning_cells(cells, target, *, among=None) -> set:
    """The cells of ``cells`` (the caller's ids) that already own a kept
    object of ``target`` -- the rule :func:`nuclei_in_cells` skips a cell
    by: the cell holding most of the object, when at least half of the
    object lies inside the cells (ties -> the higher cell id).

    Measure it over the whole patch when the recipe runs on a box: a box
    sees only its crop of a cell, so a kept nucleus just outside the box
    would not count.  ``among`` (cell ids): only the objects touching one
    of these cells are measured -- whole, wherever they lie -- and only
    these cells are returned."""
    cells = np.asarray(cells)
    target = np.asarray(target)
    if cells.shape != target.shape:
        raise ValueError("cells is {} but target is {}".format(
            cells.shape, target.shape))
    m = target != 0
    if not m.any():
        return set()
    t = target[m]
    c = cells[m]
    if among is None:
        return _majority_owners(t, c)
    want = np.unique(np.asarray(among, np.int64).ravel())
    want = want[want > 0]
    if want.size == 0:
        return set()
    # the kept objects touching those cells, then each of them whole
    objs = np.unique(t[_member_mask(c, want)])
    if objs.size == 0:
        return set()
    sel = _member_mask(t, objs.astype(np.int64))
    owning = _majority_owners(t[sel], c[sel])
    return owning & {int(x) for x in want}


def _kept_rules(G, own, cells, target_c, minv, owning=None):
    """Returns ``(written labels, owner, owning cells)``.  ``owning`` (the
    compact ids of the cells that own a kept object, measured by the
    caller over a larger array): used instead of the ones measured here."""
    m = target_c > 0
    if owning is None:
        if not m.any():
            return G, own, set()
        K1 = int(target_c.max()) + 1
        C1 = int(cells.max()) + 1
        owning = _majority_owners(target_c[m], cells[m], K1, C1)
    else:
        owning = set(owning)
        if not m.any() and not owning:
            return G, own, set()
    K1 = int(target_c.max()) + 1
    # a found nucleus that one kept object covers by KEPT_SHARE or more
    size = np.bincount(G.ravel())
    both = (G > 0) & m
    drop = set()
    if both.any():
        keys, n = np.unique(G[both].astype(np.int64) * K1
                            + target_c[both].astype(np.int64),
                            return_counts=True)
        x = keys // K1
        best = np.zeros(size.size, np.int64)
        np.maximum.at(best, x, n)
        drop = {int(i) for i in np.flatnonzero(best >= KEPT_SHARE * size)
                if i > 0}
    keep = np.zeros(size.size, bool)
    for xid, c in own.items():
        if c not in owning and xid not in drop:
            keep[xid] = True
    out = np.where(keep[G] & (target_c == 0), G, 0).astype(np.int32)
    sz = np.bincount(out.ravel(), minlength=size.size)
    small = sz < minv
    small[0] = False
    out[small[out]] = 0
    present = sz >= minv
    own = {xid: c for xid, c in own.items() if keep[xid] and present[xid]}
    return out, own, owning


def _pieces_after_trim(G, own, before, minv):
    """Every nucleus that lost voxels to the kept objects keeps its largest
    piece, and is dropped under ``minv`` voxels."""
    from scipy import ndimage as ndi

    after = np.bincount(G.ravel(), minlength=before.size)
    trimmed = [x for x in own if 0 < after[x] < before[x]]
    if not trimmed:
        return G, own
    objs = ndi.find_objects(G)
    own = dict(own)
    for x in trimmed:
        sl = objs[x - 1]
        sub = G[sl]
        m = sub == x
        keep = _largest_piece(m)
        if keep.sum() < minv:
            sub[m] = 0
            del own[x]
        else:
            sub[m & ~keep] = 0
    return G, own


# --------------------------------------------------------------------------- #
# the recipe
# --------------------------------------------------------------------------- #
def nuclei_in_cells(nuclear, cells, target, spacing_um, *,
                    radius_um=NUC_RADIUS_UM, edge=NUC_EDGE, gap=TWO_GAP,
                    second_min_um3=SECOND_MIN_UM3, min_um3=MIN_NUCLEUS_UM3,
                    cut_at_border=True, levels=None, owning=None,
                    cancel_event=None, progress=None) -> DeriveResult:
    """Find the nucleus inside each cell of ``cells``.

    Parameters
    ----------
    nuclear : (Z, Y, X) array
        The nuclear channel (cast to float32 here; never modified).
    cells : (Z, Y, X) integer array
        The other run's cells (any ids, uint32 included; never modified).
    target : (Z, Y, X) integer array or None
        The run being edited.  When given, only its empty voxels are
        written, a cell that already owns a kept object is skipped, and a
        found nucleus a kept object covers by a quarter or more is dropped.
    spacing_um : (z, y, x) um per voxel, required (there is no default: a
        wrong spacing scales every length below).  Every length is in um.
    radius_um, edge, gap : the three knobs (see the module constants).
    second_min_um3 : the second-nucleus guard volume; None or 0 turns it off.
    min_um3 : the size floor of every nucleus.
    cut_at_border : cut each nucleus at its owner's border (the default);
        False keeps it whole (and nothing is flagged cut at the edge).
    levels : :class:`PatchLevels` measured on the whole patch, cropped to
        this array's z range; None measures them on this array.
    owning : the cells (ids of ``cells``) that already own a kept object,
        measured over the whole patch (:func:`owning_cells`), used instead
        of the ones ``target`` shows here -- a box sees only its crop of a
        cell.  None: measured on ``target``.  Used only with ``target``.
    cancel_event : an object with ``is_set()`` or a callable returning a
        bool; a Stop raises ``ReshapeCancelled``.
    progress : ``progress(text, i, n)``, ``text`` in plain words.
    """
    import time

    from scipy import ndimage as ndi

    (nuc, cells, target, sp, radius_um, edge, gap, second_min_um3,
     min_um3) = _check_inputs(nuclear, cells, target, spacing_um, radius_um,
                              edge, gap, second_min_um3, min_um3)
    if levels is not None:
        if not isinstance(levels, PatchLevels):
            raise TypeError("levels must be a PatchLevels, not {!r}.".format(
                type(levels)))
        if np.asarray(levels.background_z).shape != (nuc.shape[0],):
            raise ValueError(
                "levels.background_z has {} values for {} z slices; crop it "
                "to the box with crop_z.".format(
                    np.asarray(levels.background_z).size, nuc.shape[0]))
    run = _Run(cancel_event, progress)
    secs = {}
    t_last = [time.perf_counter()]

    def tick(name):
        now = time.perf_counter()
        secs[name] = round(now - t_last[0], 3)
        t_last[0] = now

    if levels is not None:
        _check_otsu(float(levels.otsu))
    run.stage(DERIVE_STAGE_READ)
    cells_c, cell_ids = _compact(cells)
    n_cells = int(cell_ids.size)
    run.poll()
    target_c = _compact(target)[0] if target is not None else None
    tick("compact")
    if n_cells == 0:
        return DeriveResult(labels=np.zeros(nuc.shape, np.int32), seconds=secs)
    minv = min_um3 / float(np.prod(sp))

    # 1-2 smoothing and patch levels
    run.stage(DERIVE_STAGE_SMOOTH)
    g = ndi.gaussian_filter(nuc, _sig(_SMOOTH_UM, sp))
    run.poll()
    g2 = ndi.gaussian_filter(nuc, _sig(_SMOOTH2_UM, sp))
    run.stage(DERIVE_STAGE_LEVELS)
    if levels is None:
        po = _otsu(g)
        _check_otsu(po)
        bz = _background_z(g, cells_c, po, sp)
    else:
        po = float(levels.otsu)
        bz = np.asarray(levels.background_z, np.float32)
    tick("1-2 smooth + levels")

    # 3 centres
    run.stage(DERIVE_STAGE_CENTRES)
    L = _log_max(nuc, radius_um, sp, run)
    run.poll()
    pts = _centres(L, g >= _GATE * po, sp, _CENTRE_SPACING * radius_um)
    if len(pts):
        pts = pts[L[tuple(pts.T)] > 0]
    del L
    tick("3 centres")

    # 4 own edge
    run.stage(DERIVE_STAGE_EDGES)
    lab = _own_edges(pts, g, g2, bz, g >= _FLOOR * po, alpha=edge,
                     floor_abs=_FLOOR * po, min_contrast=_MIN_CONTRAST * po,
                     hw=_halfw(_PEAK_WINDOW_UM, sp))
    sz = np.bincount(lab.ravel())
    small = sz < minv
    small[0] = False
    lab[small[lab]] = 0
    G = _compact(lab)[0]
    del lab
    tick("4 own edge")

    # 5 owners
    run.stage(DERIVE_STAGE_OWNERS)
    n_step4 = int(G.max())
    own, info = _assign(G, cells_c)
    n_outside = n_step4 - len(own)
    lut = _lut_of(own, n_step4)
    G = np.where(lut[G] > 0, G, 0).astype(np.int32)
    crossing = {x: v for x, v in info.items() if v[1] >= _CROSSING_SHARE}
    tick("5 owners")

    # 6a split at the border
    run.stage(DERIVE_STAGE_SPLIT)
    n_split = 0
    cnt = _per_cell_count(n_cells, own)
    nxt = int(G.max()) + 1
    tab = _overlap_table(G, cells_c) if crossing else {}
    gobjs = ndi.find_objects(G) if crossing else []
    for x, (_s1, s2) in sorted(crossing.items()):
        if s2 < _SPLIT_SHARE:
            continue
        row = {k: v for k, v in tab[x].items() if k}
        second = sorted(row, key=row.get, reverse=True)[1]
        if cnt.get(second, 1) == 0:
            sl = gobjs[x - 1]
            m = (G[sl] == x) & (cells_c[sl] == second)
            G[sl][m] = nxt
            own[nxt] = second
            cnt[second] = 1
            nxt += 1
            n_split += 1
    tick("6a split")

    # 6b two in a cell
    run.stage(DERIVE_STAGE_TWO)
    if own:
        G, own = _two_in_a_cell(G, own, cells_c, g2, ratio=gap,
                                pad=_halfw(_PAD_UM, sp), run=run)
        G, own = _relabel_owner(G, own)
    tick("6b two in a cell")

    # 6c rescue
    run.stage(DERIVE_STAGE_RESCUE)
    cnt = _per_cell_count(n_cells, own)
    empty = [c for c, k in cnt.items() if k == 0]
    cobjs = ndi.find_objects(cells_c)
    res = _rescue(pts, g, g2, cells_c, cobjs, po, bz, cells_list=empty,
                  forbid=G > 0, sp=sp, edge=edge, minv=minv,
                  next_id=int(G.max()) + 1, run=run)
    for kk, (c, sl, m) in res.items():
        G[sl][m] = kk
        own[kk] = c
    n_rescued = len(res)
    del g, g2, res
    tick("6c rescue")

    # 7 tidy: cut at the owner's border, then per nucleus
    run.stage(DERIVE_STAGE_TIDY)
    cut = set()
    if cut_at_border:
        before = np.bincount(G.ravel())
        lutc = _lut_of(own, int(G.max()))
        G = np.where(cells_c == lutc[G], G, 0).astype(np.int32)
        after = np.bincount(G.ravel(), minlength=before.size)
        lost = np.where(before > 0, 1 - after / np.maximum(before, 1), 0)
        cut = {int(k) for k in np.nonzero(lost > CUT_FLAG_SHARE)[0] if k > 0}
    run.poll()
    if cut_at_border:
        G = _tidy(G, _opening_footprint(sp), minv, run, cells=cells_c,
                  owner_lut=lutc)
    else:
        G = _tidy(G, _opening_footprint(sp), minv, run)
    present = np.bincount(G.ravel()) > 0
    own = {k: v for k, v in own.items() if k < present.size and present[k]}
    n_guard = 0
    if second_min_um3:
        G, own, n_guard = _guard(G, own, second_min_um3, sp)
    cut &= set(own)
    n_found = len(own)
    tick("7 tidy")

    # Add's rules against the kept objects
    run.stage(DERIVE_STAGE_KEPT)
    owning_given, owning = owning, set()
    if target_c is not None:
        before_kept = np.bincount(G.ravel())
        own_c = None
        if owning_given is not None:
            want = np.asarray(sorted(int(x) for x in owning_given),
                              np.int64)
            pos = np.searchsorted(cell_ids, want)
            hit = pos < cell_ids.size
            hit[hit] = cell_ids[pos[hit]] == want[hit]
            own_c = {int(p) + 1 for p in pos[hit]}
        G, own, owning = _kept_rules(G, own, cells_c, target_c, minv,
                                     owning=own_c)
        run.poll()
        # a trimmed nucleus may fall apart or shrink: the largest piece, the
        # floor and the second-nucleus guard once more
        G, own = _pieces_after_trim(G, own, before_kept, minv)
        if second_min_um3:
            G, own, _n = _guard(G, own, second_min_um3, sp)
        cut &= set(own)
    run.poll()

    # final ids: by owner cell id, then size (largest first)
    size = np.bincount(G.ravel())
    order = sorted(own, key=lambda x: (int(cell_ids[own[x] - 1]), -int(size[x]), x))
    lutf = np.zeros(int(G.max()) + 1, np.int32)
    for new_id, x in enumerate(order, start=1):
        lutf[x] = new_id
    labels = lutf[G]
    owner = {int(lutf[x]): int(cell_ids[own[x] - 1]) for x in order}
    by = {}
    for nid in sorted(owner):
        by.setdefault(owner[nid], []).append(nid)
    two = {c: ids for c, ids in by.items() if len(ids) >= 2}
    cut_final = {int(lutf[x]) for x in cut}
    cut_share = {}
    if cut_at_border:
        cut_share = {int(lutf[x]): float(lost[x]) for x in order
                     if x < lost.size and lost[x] > 0}
    has = {own[x] for x in order}
    n_no = sum(1 for c in range(1, n_cells + 1)
               if c not in owning and c not in has)
    tick("kept + ids")
    return DeriveResult(
        labels=labels, owner=owner, two_in_cell=two, cut_at_edge=cut_final,
        cut_share=cut_share,
        n_cells=n_cells, n_no_nucleus=n_no, n_outside_cells=n_outside,
        n_cell_has_one=len(owning), n_split_border=n_split,
        n_cut_at_edge=len(cut_final), n_all_taken=n_found - len(owner),
        n_found=n_found, n_guard_removed=n_guard, n_rescued=n_rescued,
        n_centres=int(len(pts)), otsu=po, seconds=secs)

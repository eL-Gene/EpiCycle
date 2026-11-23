"""The pairing check between two runs of one patch.

A patch can carry several runs: cells segmented from the membrane channel,
nuclei from the DNA channel.  This module answers, for every object of the
ACTIVE run, which objects of the OTHER run it shares voxels with, how many
of them, and how well -- so that "a cell with no nucleus, or two" and "a
nucleus across two cells" can be found during triage, in either direction.
It is a check, not an edit, and not a biology measurement: one match
means the two runs agree, not that the object is right.

The rule, in plain words:

* Two objects A (active run) and P (other run) are LINKED when the voxels
  they share are at least ``link_floor`` of the smaller one, the overlap
  coefficient ``O = |A and P| / min(|A|, |P|)``.  ``O`` is the same number
  seen from either side.  Links do not exclude each other: one nucleus may
  link two cells, and one cell many nuclei.
* Each run has a size floor in voxels (:func:`run_floor`): a tenth of the
  median size of its objects that touch no face of the volume.  An object
  below its run's floor never links, on either side; an active object below
  the floor is marked "too small to judge".
* Per active object: ``match_count`` (its links); ``match_id`` (the
  linked match with the most SHARED VOXELS, ties to the higher ``O``,
  then the lower id; 0 when there is none); ``match_overlap`` (that
  match's ``O``); ``match_shared`` (the largest link count among my
  matches: "my nucleus is also in another cell"); ``match_minor_ratio``
  (the second-largest shared count over the largest, among my links).

The best match is chosen by shared voxels, not by ``O``: a fragment lying
wholly inside a cell has ``O = 1`` and would beat the real nucleus beside it.
The minor ratio is a ratio of shared counts, not of the matched objects'
sizes, so it reads the same from both sides: a 60 / 40 straddle gives 0.67 whether the
nucleus or the cells are active.

Counting is done a slab of z-planes at a time.  Sixty-four-bit pair keys are
built only for the voxels where BOTH runs are labelled within one slab, so
no whole-volume 64-bit copy is ever made.  The ids of BOTH runs are read
the way the Reshape cut reads them (:func:`as_cut_ids`, an int32 view), so
a negative id in an int16 file, or an id past 2**31 in a wider one, is
keyed the same on every path, and the active ids match the label column of
the feature table (which is built from an int32 cast too).

numpy only: no Qt, no napari, no pandas.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

#: z-planes counted at a time (the memory held is one slab, never a copy)
SLAB_PLANES = 16

#: size floor: this share of the median interior object volume ...
FLOOR_FACTOR = 0.1
#: ... over at least this many interior objects ...
FLOOR_MIN_COUNT = 20
#: ... else this many voxels
FLOOR_FALLBACK = 20

# The status words, shown in the legend.  Order = the legend's order.
STATUS_NONE = "no match"
STATUS_ONE = "one match"
STATUS_SEVERAL = "several matches"
STATUS_SHARED = "shared match"
STATUS_TOO_SMALL = "too small to judge"
STATUS_NOT_MEASURED = "not measured"
STATUS_WORDS = (
    STATUS_NONE, STATUS_ONE, STATUS_SEVERAL, STATUS_SHARED,
    STATUS_TOO_SMALL, STATUS_NOT_MEASURED,
)

#: The colour (RGB 0-255) of each word, in the legend's order, shared by the
#: QC plot and the triage histogram.  The common case (one match) is a
#: quiet steel blue so the exceptions carry the colour: pink for no match,
#: gold for several, cyan for a shared match; a dim grey for an object
#: under the size floor and a light grey for a row whose values were
#: blanked or whose label is gone.
MATCH_COLOURS = {
    STATUS_NONE: (236, 96, 140),        # pink
    STATUS_ONE: (120, 150, 175),        # steel blue
    STATUS_SEVERAL: (246, 190, 70),     # gold
    STATUS_SHARED: (80, 200, 215),      # cyan
    STATUS_TOO_SMALL: (105, 105, 110),      # dim grey
    STATUS_NOT_MEASURED: (190, 190, 190),   # light grey
}
#: The words a legend lists even at a count of 0 (the greys only when a
#: row reads them).
MATCH_ALWAYS = (STATUS_NONE, STATUS_ONE, STATUS_SEVERAL, STATUS_SHARED)

#: the per-object columns of :class:`PairTable`, in table order
COLUMNS = (
    "match_count", "match_id", "match_overlap", "match_shared",
    "match_minor_ratio",
)

# Pair keys pack the active id in the high 32 bits and the other id (an
# int32, shifted to be non-negative) in the low 32 bits.
_SHIFT = 32
_LOW = (1 << 32) - 1
_OFFSET = 1 << 31
# ids counted with bincount when they are non-negative and below this
_BINCOUNT_MAX = 1 << 24


class PairingCancelled(Exception):
    """The cancel event was set between two slabs."""


def as_cut_ids(a):
    """Label ids as the Reshape cut reads them: int32-safe dtypes unchanged,
    wider dtypes wrap the way an int32 cast does."""
    a = np.asarray(a)
    if a.dtype == np.int32 or np.can_cast(a.dtype, np.int32):
        return a
    return np.asarray(a, np.int32)


def _real(name, value):
    """``value`` as a float when it is a real, finite, non-bool number."""
    is_bool = isinstance(value, (bool, np.bool_))
    if is_bool or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return v


def _plain(v: float):
    """A plain Python int when ``v`` is whole, else a plain float."""
    return int(v) if float(v).is_integer() else float(v)


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OverlapCounts:
    """Every object of both runs and every pair that shares voxels.

    ``active_ids`` / ``active_voxels``: the active run's nonzero ids (as
    :func:`as_cut_ids` reads them, ascending) and their sizes in voxels;
    ``other_ids`` / ``other_voxels`` the same for the other run;
    ``pair_active`` / ``pair_other`` / ``pair_shared``: one entry per pair
    of objects that share at least one voxel, with the shared count.
    """

    active_ids: np.ndarray
    active_voxels: np.ndarray
    other_ids: np.ndarray
    other_voxels: np.ndarray
    pair_active: np.ndarray
    pair_other: np.ndarray
    pair_shared: np.ndarray

    def pair_index(self):
        """``(ia, ip)``: each pair's position in ``active_ids`` and
        ``other_ids``."""
        ia = np.searchsorted(self.active_ids, self.pair_active)
        ip = np.searchsorted(self.other_ids, self.pair_other)
        return ia, ip

    def overlap(self) -> np.ndarray:
        """The overlap coefficient of each pair (float64)."""
        if self.pair_shared.size == 0:
            return np.zeros(0, np.float64)
        ia, ip = self.pair_index()
        smaller = np.minimum(self.active_voxels[ia], self.other_voxels[ip])
        return self.pair_shared.astype(np.float64) / smaller


def _count_ids(part):
    """``(ids, counts)`` of the nonzero values of one slab."""
    flat = part.reshape(-1)
    if flat.size == 0:
        return flat[:0], np.zeros(0, np.int64)
    if part.dtype.kind in "ui" and part.dtype != np.uint64:
        lo, hi = int(flat.min()), int(flat.max())
        if lo >= 0 and hi < _BINCOUNT_MAX:
            c = np.bincount(flat, minlength=hi + 1)
            ids = np.flatnonzero(c)
            ids = ids[ids != 0]
            return ids.astype(part.dtype), c[ids].astype(np.int64)
    ids, c = np.unique(flat, return_counts=True)
    keep = ids != 0
    return ids[keep], c[keep].astype(np.int64)


def _sum_by_id(ids_list, counts_list, dtype):
    """Sum the per-slab counts of equal ids; ids ascending in ``dtype``."""
    if not ids_list:
        return np.zeros(0, dtype), np.zeros(0, np.int64)
    ids = np.concatenate(ids_list)
    counts = np.concatenate(counts_list)
    u, inv = np.unique(ids, return_inverse=True)
    total = np.zeros(u.size, np.int64)
    np.add.at(total, inv.reshape(-1), counts)
    return u.astype(dtype), total


def _unique_sorted_inplace(k):
    """``(unique, counts)`` of a 1-D int64 array, sorting it in place."""
    k.sort()
    edge = np.empty(k.size, bool)
    edge[0] = True
    np.not_equal(k[1:], k[:-1], out=edge[1:])
    starts = np.flatnonzero(edge)
    counts = np.diff(np.append(starts, k.size)).astype(np.int64)
    return k[starts], counts


def _slab_pairs(av, pv):
    """Distinct ``(active id, other id, count)`` over one slab's overlap
    voxels.  Both hold int32-safe ids; ``av`` keeps its dtype."""
    if av.size == 0:
        return av[:0], pv[:0].astype(np.int32), np.zeros(0, np.int64)
    a_lo, a_hi = int(av.min()), int(av.max())
    if a_lo >= 0 and a_hi < (1 << 31):
        k = av.astype(np.int64)
        k <<= _SHIFT
        low = pv.astype(np.int64)
        low += _OFFSET
        k |= low
        del low
        u, c = _unique_sorted_inplace(k)
        del k
        a_ids = (u >> _SHIFT).astype(av.dtype)
        p_ids = ((u & _LOW) - _OFFSET).astype(np.int32)
        return a_ids, p_ids, c
    # negative active ids cannot be packed: number each side within the
    # slab first
    ua, ia = np.unique(av, return_inverse=True)
    up, ip = np.unique(pv, return_inverse=True)
    k = ia.reshape(-1).astype(np.int64)
    k *= up.size
    k += ip.reshape(-1)
    u, c = _unique_sorted_inplace(k)
    return ua[u // up.size], up[u % up.size].astype(np.int32), c


def _sum_pairs(a_list, p_list, c_list, a_dtype):
    """Sum the per-slab pair counts of equal pairs; sorted by (active, other)."""
    if not a_list:
        return (np.zeros(0, a_dtype), np.zeros(0, np.int32),
                np.zeros(0, np.int64))
    a = np.concatenate(a_list)
    p = np.concatenate(p_list).astype(np.int32)
    c = np.concatenate(c_list)
    if a.size == 0:
        return a.astype(a_dtype), p, c
    order = np.lexsort((p, a))
    a, p, c = a[order], p[order], c[order]
    edge = np.empty(a.size, bool)
    edge[0] = True
    edge[1:] = (a[1:] != a[:-1]) | (p[1:] != p[:-1])
    starts = np.flatnonzero(edge)
    total = np.add.reduceat(c, starts)
    return a[starts].astype(a_dtype), p[starts], total.astype(np.int64)


def count_overlaps(active, other, *, slab=SLAB_PLANES, cancel_event=None):
    """Count every object of both runs and every pair that shares voxels.

    Both volumes must have the same shape; they are read, never changed.
    The work is done ``slab`` z-planes at a time: the sizes of each slab's
    objects, and the pairs found on the voxels labelled in BOTH runs.  Both
    runs' ids are read the way the Reshape cut reads them.  A set
    ``cancel_event`` raises :class:`PairingCancelled` between slabs.
    """
    active = np.asarray(active)
    other = np.asarray(other)
    if active.shape != other.shape:
        raise ValueError(
            f"the two runs' labels differ in shape: {active.shape} and "
            f"{other.shape}")
    if active.dtype.kind not in "ui" or other.dtype.kind not in "ui":
        raise ValueError(
            f"labels must hold integer ids, got {active.dtype} and "
            f"{other.dtype}")
    a_dtype = as_cut_ids(np.zeros(1, active.dtype)).dtype
    o_dtype = as_cut_ids(np.zeros(1, other.dtype)).dtype
    step = max(1, int(slab))
    planes = int(active.shape[0]) if active.ndim else 0
    a_ids, a_cnt, o_ids, o_cnt = [], [], [], []
    pa, pp, pc = [], [], []
    for z0 in range(0, planes, step):
        if cancel_event is not None and cancel_event.is_set():
            raise PairingCancelled()
        a = as_cut_ids(active[z0:z0 + step])
        p = as_cut_ids(other[z0:z0 + step])
        i, c = _count_ids(a)
        a_ids.append(i)
        a_cnt.append(c)
        i, c = _count_ids(p)
        o_ids.append(i)
        o_cnt.append(c)
        both = a != 0
        both &= p != 0
        if both.any():
            x, y, c = _slab_pairs(a[both], p[both])
            pa.append(x)
            pp.append(y)
            pc.append(c)
        del both
    if cancel_event is not None and cancel_event.is_set():
        raise PairingCancelled()
    a_ids, a_vox = _sum_by_id(a_ids, a_cnt, a_dtype)
    o_ids, o_vox = _sum_by_id(o_ids, o_cnt, o_dtype)
    pair_a, pair_p, pair_c = _sum_pairs(pa, pp, pc, a_dtype)
    return OverlapCounts(
        active_ids=a_ids, active_voxels=a_vox,
        other_ids=o_ids, other_voxels=o_vox,
        pair_active=pair_a, pair_other=pair_p, pair_shared=pair_c,
    )


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairTable:
    """The pairing check per active object, as aligned arrays.

    ``label`` holds the active run's ids as :func:`as_cut_ids` reads them
    (the volume's dtype when it is int32-safe, else int32 with wider ids
    wrapped the way the feature table's int32 cast wraps them), ascending,
    one entry per nonzero id, so the table joins the feature table on the
    label value.  ``voxels`` is each object's size in voxels.
    ``too_small`` marks the objects below the active run's floor: they never
    link, read ``match_count`` 0, ``match_id`` 0, ``match_shared`` 0,
    and NaN for ``match_overlap`` and ``match_minor_ratio`` (not
    judged).  An object above the floor with no link reads ``match_id`` 0
    and ``match_overlap`` 0.  ``match_minor_ratio`` is NaN with fewer
    than two links.  The floors used are kept with the table as plain
    Python numbers.
    """

    label: np.ndarray
    voxels: np.ndarray
    match_count: np.ndarray
    match_id: np.ndarray
    match_overlap: np.ndarray
    match_shared: np.ndarray
    match_minor_ratio: np.ndarray
    too_small: np.ndarray
    link_floor: float
    floor_active: int | float
    floor_other: int | float

    def __len__(self) -> int:
        return int(self.label.size)

    def columns(self) -> dict:
        """``{column name: array}`` for the five match columns."""
        return {name: getattr(self, name) for name in COLUMNS}


def _check_floor(name, value):
    """A size floor as a plain Python number: real, finite, zero or more."""
    v = _real(name, value)
    if v < 0:
        raise ValueError(f"{name} must be zero or more voxels, got {value!r}")
    return _plain(v)


def pair_runs(active, other, *, link_floor, floor_active, floor_other,
              slab=SLAB_PLANES, cancel_event=None) -> PairTable:
    """Pair every object of the active run with the other run's objects.

    ``active`` and ``other`` are the two runs' label volumes of the same
    patch (same shape; neither is changed).  ``link_floor`` is the share of
    the smaller object two objects must share to be linked (0 < floor <= 1);
    ``floor_active`` / ``floor_other`` are the runs' size floors in voxels:
    an object smaller than its run's floor never links.  The link test is
    done in exact integer arithmetic, so an overlap of exactly the floor
    links.  A set ``cancel_event`` raises :class:`PairingCancelled`.
    """
    lf = _real("link_floor", link_floor)
    if not (0.0 < lf <= 1.0):
        raise ValueError(f"link_floor must be in (0, 1], got {link_floor!r}")
    floor_active = _check_floor("floor_active", floor_active)
    floor_other = _check_floor("floor_other", floor_other)
    counts = count_overlaps(active, other, slab=slab,
                            cancel_event=cancel_event)
    n = counts.active_ids.size
    small_a = counts.active_voxels < floor_active
    small_p = counts.other_voxels < floor_other

    count = np.zeros(n, np.int64)
    best = np.zeros(n, np.int64)
    overlap = np.where(small_a, np.nan, 0.0)
    shared_max = np.zeros(n, np.int64)
    minor = np.full(n, np.nan)

    if counts.pair_shared.size:
        ia, ip = counts.pair_index()
        sh = counts.pair_shared
        smaller = np.minimum(counts.active_voxels[ia],
                             counts.other_voxels[ip])
        frac = Fraction(lf).limit_denominator(1_000_000)
        linked = (sh * frac.denominator >= frac.numerator * smaller)
        linked &= ~small_a[ia]
        linked &= ~small_p[ip]
        ia, ip, sh, smaller = ia[linked], ip[linked], sh[linked], smaller[linked]
        if ia.size:
            degree = np.bincount(ip, minlength=counts.other_ids.size)
            count = np.bincount(ia, minlength=n).astype(np.int64)
            np.maximum.at(shared_max, ia, degree[ip])
            pid = counts.other_ids[ip].astype(np.int64)
            # per active object: most shared voxels first, then the higher
            # O (= the smaller match at equal shared), then the lower id
            order = np.lexsort((pid, smaller, -sh, ia))
            ia_o = ia[order]
            first = np.flatnonzero(np.r_[True, ia_o[1:] != ia_o[:-1]])
            top = order[first]
            rows = ia[top]
            best[rows] = pid[top]
            overlap[rows] = sh[top] / smaller[top]
            sizes = np.diff(np.r_[first, ia_o.size])
            two = sizes >= 2
            second = order[first[two] + 1]
            minor[ia[top[two]]] = sh[second] / sh[top[two]]

    return PairTable(
        label=counts.active_ids,
        voxels=counts.active_voxels,
        match_count=count,
        match_id=best,
        match_overlap=overlap,
        match_shared=shared_max,
        match_minor_ratio=minor,
        too_small=small_a.astype(bool),
        link_floor=lf,
        floor_active=floor_active,
        floor_other=floor_other,
    )


# --------------------------------------------------------------------------- #
# the floors
# --------------------------------------------------------------------------- #
def object_volumes_and_border(labels, *, slab=SLAB_PLANES, cancel_event=None):
    """``(ids, volumes_vx, touches_border)`` of one run's objects.

    ``ids`` are the nonzero ids in the volume's own dtype, ascending;
    ``volumes_vx`` their sizes in voxels; ``touches_border`` is True when an
    object's bounding box touches any face of the volume (the same as the
    object having a voxel on that face).  Sizes are counted a slab of
    z-planes at a time; the faces are read as planes.
    """
    labels = np.asarray(labels)
    step = max(1, int(slab))
    planes = int(labels.shape[0]) if labels.ndim else 0
    ids_l, cnt_l = [], []
    for z0 in range(0, planes, step):
        if cancel_event is not None and cancel_event.is_set():
            raise PairingCancelled()
        i, c = _count_ids(labels[z0:z0 + step])
        ids_l.append(i)
        cnt_l.append(c)
    ids, vols = _sum_by_id(ids_l, cnt_l, labels.dtype)
    on_face = []
    if labels.size:
        for axis in range(labels.ndim):
            for end in (0, labels.shape[axis] - 1):
                face = np.take(labels, end, axis=axis)
                on_face.append(np.unique(face))
    if on_face:
        face_ids = np.unique(np.concatenate(on_face))
        border = np.isin(ids, face_ids)
    else:
        border = np.zeros(ids.size, bool)
    return ids, vols, border


def run_floor(volumes_vx, border_mask, *, factor=FLOOR_FACTOR,
              min_count=FLOOR_MIN_COUNT, fallback=FLOOR_FALLBACK) -> int:
    """One run's size floor in voxels.

    ``factor`` of the median volume of the run's INTERIOR objects (those
    whose bounding box touches no face of the volume), rounded up.  With
    fewer than ``min_count`` interior objects the median is not trusted and
    the floor is ``fallback`` voxels (rounded up).  Volumes are in voxels,
    never physical.  The reconstruction's "remove specks" value is not used:
    it acts on the joined mask before the cut, not on each object's size.
    """
    f = _real("factor", factor)
    n_min = _real("min_count", min_count)
    fb = _real("fallback", fallback)
    for name, v in (("factor", f), ("min_count", n_min), ("fallback", fb)):
        if v < 0:
            raise ValueError(f"{name} must be zero or more, got {v!r}")
    vols = np.asarray(volumes_vx, dtype=np.int64).reshape(-1)
    border = np.asarray(border_mask, dtype=bool).reshape(-1)
    if vols.shape != border.shape:
        raise ValueError(
            f"{vols.size} volumes but {border.size} border flags")
    interior = vols[~border]
    if interior.size < n_min:
        return int(math.ceil(fb))
    s = np.sort(interior)
    m = s.size // 2
    if s.size % 2:
        med = Fraction(int(s[m]))
    else:
        med = Fraction(int(s[m - 1]) + int(s[m]), 2)
    share = Fraction(f).limit_denominator(1_000_000) * med
    return int(math.ceil(share))


# --------------------------------------------------------------------------- #
# the status
# --------------------------------------------------------------------------- #
def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def match_status(count, shared, too_small, measured=True) -> str:
    """The match word for one object, with a fixed precedence.

    "too small to judge" first, then "not measured" (the values were
    blanked, or are missing), then "no match" (no link), "several
    matches" (two links or more; such an object may also share one),
    "shared match" (one link, and that match links another object
    too), and "one match".
    """
    if not _is_missing(too_small) and bool(too_small):
        return STATUS_TOO_SMALL
    if not measured or _is_missing(count) or _is_missing(shared):
        return STATUS_NOT_MEASURED
    count = int(count)
    if count <= 0:
        return STATUS_NONE
    if count >= 2:
        return STATUS_SEVERAL
    if int(shared) >= 2:
        return STATUS_SHARED
    return STATUS_ONE


def status_from_columns(count, shared, overlap) -> np.ndarray:
    """The match word per object, read back from the saved columns.

    The words and their precedence are those of :func:`match_status`.
    "too small to judge" is stored as a count of 0 with a NaN overlap (a
    measured object with no link reads overlap 0.0); "not measured" is a
    NaN count (the row was blanked).  Returns an object array of words, one
    per row.
    """
    c = np.asarray(count, dtype=np.float64).reshape(-1)
    sh = np.asarray(shared, dtype=np.float64).reshape(-1)
    ov = np.asarray(overlap, dtype=np.float64).reshape(-1)
    if not (c.shape == sh.shape == ov.shape):
        raise ValueError(
            f"columns differ in length: {c.size}, {sh.size} and {ov.size}")
    out = np.full(c.size, STATUS_ONE, dtype=object)
    out[sh >= 2] = STATUS_SHARED
    out[c >= 2] = STATUS_SEVERAL
    out[c <= 0] = STATUS_NONE
    out[np.isnan(c) | np.isnan(sh)] = STATUS_NOT_MEASURED
    out[(c == 0) & np.isnan(ov)] = STATUS_TOO_SMALL
    return out

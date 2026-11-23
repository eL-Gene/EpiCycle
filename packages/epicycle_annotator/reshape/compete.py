"""Cells from markers: a competition with a growth limit.

Every marker spreads into the free space along its cheapest path.  A step
between two neighbouring voxels costs its physical length in um times the
mean of the two voxels' prices, where a voxel's price per um is
``1 + wall_cost x elevation`` (the elevation is the 0-1 wall map, bright on
a membrane).  Each voxel goes to the marker that reaches it most cheaply,
and only while that total stays within ``growth_limit``: a cell crosses
plain cytoplasm cheaply, stops on a membrane, and leaves space no marker
can afford empty.  The growth limit is therefore a budget of path cost, not
a distance.

* **Ties go to the lower id.**  The solver keeps one heap keyed
  ``(cost, id, voxel)``: at an exactly equal cost the lower id wins, so the
  answer does not depend on the order voxels are visited -- the same in a
  tile as in one call, whatever the ids happen to be.
* **Voxels outside ``space`` are impassable.**  The caller passes
  ``existing == 0`` (and the tissue outline, when one is chosen) as the
  space, so no path runs through a kept object.
* **Kept objects may compete.**  :func:`kept_competitors` seeds every kept
  object on the empty voxels next to it, under ids above the new markers;
  the caller zeroes their claims afterwards, so a new cell does not take a
  kept neighbour's side of a gap.
* **Membrane at the edge.**  :func:`include_membrane` hands an unclaimed
  bright voxel within ``within_um`` to the nearest object when that object
  is new, so neighbouring new cells share the membrane between them.  The
  caller then keeps one piece per marker (``keep_marker_components``).
* **Tiles.**  :func:`plan_tiles` cuts a large volume into tiles in all three
  axes with a halo wider than any affordable path; :func:`compete_tiled`
  runs them one by one on crops of ONE elevation and ONE marker volume and
  equals a single call voxel for voxel.

Memory: the solver holds a float32 cost, an int32 id and a done flag per
voxel (9 B) plus 16 B per waiting heap entry; no float64 volume is made.

Stop: a cancel is polled before and after every solve and, while the
compiled solver runs, a small watcher thread turns it into a flag the solver
reads every ~1 M heap pops, so Stop answers within about a second even
inside one large tile.  The flag only aborts; it never changes an answer.

Pure: numpy / scipy, plus numba for the two inner loops (imported on first
use, with a plain-word error when it is missing or cannot compile).
"""

from __future__ import annotations

import math
import threading

import numpy as np
from scipy import ndimage as ndi

from ._common import check_cancel as _check_cancel
from ._common import spacing_um as _spacing
from .engine import ReshapeCancelled

__all__ = [
    "COMPETE_TILE_VOXELS",
    "GROWTH_LIMIT",
    "MEMBRANE_LEVEL",
    "MEMBRANE_WITHIN_UM",
    "NEIGHBOURS",
    "NUMBA_CACHE_FAILED",
    "NUMBA_MISSING",
    "SLIVER_SHARE",
    "STOP_CHECK_POPS",
    "TILE_HALO_EXTRA_UM",
    "TILE_HALO_EXTRA_VOXELS",
    "WALL_COST",
    "compete",
    "compete_tiled",
    "drop_sliver_markers",
    "have_numba",
    "include_membrane",
    "kept_competitors",
    "plan_tiles",
    "tile_halo",
]

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: The price of the brightest wall, per um, on top of the 1 every um costs.
WALL_COST = 40.0

#: The budget of path cost a marker may spend (cost units, not um).  At
#: 28, plain cytoplasm (about 4.75 per um on a typical wall picture) is
#: used up about 6 um from the marker, while a bright membrane costs nearly
#: the whole budget per um, so a cell stops in its wall.  Measured against
#: a model's labelling of one patch, 28 matched its cells better than 14
#: (median overlap 0.72 against 0.66) at the price of more cells spilling
#: a little past their walls.
GROWTH_LIMIT = 28.0

#: The neighbourhoods a path may step through.
NEIGHBOURS = (6, 18, 26)

#: An unclaimed voxel at least this bright on the wall map counts as
#: membrane for :func:`include_membrane`.
MEMBRANE_LEVEL = 0.25

#: How far (um) a new object reaches to take the membrane at its edge.
MEMBRANE_WITHIN_UM = 1.0

#: A marker with less than this share of its voxels on empty space is a
#: sliver of a kept object's own nucleus and is dropped.
SLIVER_SHARE = 0.5

#: Voxels per tile, halo included.
COMPETE_TILE_VOXELS = 16_000_000

#: The halo beyond the growth limit: one um for the membrane step, plus two
#: voxels.
TILE_HALO_EXTRA_UM = 1.0
TILE_HALO_EXTRA_VOXELS = 2

NUMBA_MISSING = (
    "The numba package is not installed, so cells cannot spread from their "
    "markers. Install numba (it normally comes with Cellpose).")

NUMBA_CACHE_FAILED = (
    "numba could not set up the cell-growth solver: it found no folder to "
    "keep the compiled solver in (EpiCycle's own folder may be read-only). "
    "Set the NUMBA_CACHE_DIR environment variable to a folder you can write "
    "to and restart.")

#: The solver reads the Stop flag once every this many heap pops (2**20).
STOP_CHECK_POPS = 1 << 20

#: How often (s) the watcher thread looks at the cancel event.
_STOP_POLL_S = 0.05

#: ``pops & mask == 0`` marks a Stop check (a module value, so a test can
#: make it check often).
_STOP_CHECK_MASK = STOP_CHECK_POPS - 1

_MAX_ID = np.iinfo(np.int32).max


# --------------------------------------------------------------------------- #
# the inner loops (compiled by numba on first use)
# --------------------------------------------------------------------------- #
def _grow_kernel(nz, ny, nx, space, elev, init_v, init_l, dz, dy, dx,
                 step_len, wall_cost, limit, stop, check_mask):
    """Multi-source Dijkstra with a cut-off.

    ``init_v`` / ``init_l`` are the marker voxels and their ids, already
    sorted by ``(id, voxel)`` -- a sorted array is a valid heap, all keys
    being 0.  Returns ``(labels int32, cost float32, aborted)`` over the
    raveled volume; unclaimed voxels hold 0 / inf.  ``stop[0]`` is read
    once every ``check_mask + 1`` pops; when it is non-zero the solve stops
    and ``aborted`` is True (the partial answer is then meaningless).
    """
    n = nz * ny * nx
    plane = ny * nx
    dist = np.full(n, np.inf, dtype=np.float32)
    lab = np.zeros(n, dtype=np.int32)
    done = np.zeros(n, dtype=np.bool_)
    m = init_v.shape[0]
    no = dz.shape[0]
    cap = m + 64 * no + 4096
    hk = np.empty(cap, dtype=np.float32)
    hl = np.empty(cap, dtype=np.int32)
    hv = np.empty(cap, dtype=np.int64)
    for t in range(m):
        v = init_v[t]
        hk[t] = 0.0
        hl[t] = init_l[t]
        hv[t] = v
        dist[v] = 0.0
        lab[v] = init_l[t]
    size = m
    pops = 0
    aborted = False
    while size > 0:
        pops += 1
        if (pops & check_mask) == 0 and stop[0] != 0:
            aborted = True
            break
        k0 = hk[0]
        l0 = hl[0]
        v0 = hv[0]
        size -= 1
        if size > 0:
            kk = hk[size]
            ll = hl[size]
            vv = hv[size]
            pos = 0
            while True:
                c = 2 * pos + 1
                if c >= size:
                    break
                r = c + 1
                if r < size:
                    if hk[r] < hk[c] or (hk[r] == hk[c] and (
                            hl[r] < hl[c] or (hl[r] == hl[c] and hv[r] < hv[c]))):
                        c = r
                if hk[c] < kk or (hk[c] == kk and (
                        hl[c] < ll or (hl[c] == ll and hv[c] < vv))):
                    hk[pos] = hk[c]
                    hl[pos] = hl[c]
                    hv[pos] = hv[c]
                    pos = c
                else:
                    break
            hk[pos] = kk
            hl[pos] = ll
            hv[pos] = vv
        if done[v0]:
            continue
        done[v0] = True
        if size + no > cap:
            cap2 = 2 * cap
            hk2 = np.empty(cap2, dtype=np.float32)
            hl2 = np.empty(cap2, dtype=np.int32)
            hv2 = np.empty(cap2, dtype=np.int64)
            hk2[:size] = hk[:size]
            hl2[:size] = hl[:size]
            hv2[:size] = hv[:size]
            hk = hk2
            hl = hl2
            hv = hv2
            cap = cap2
        z = v0 // plane
        rem = v0 - z * plane
        y = rem // nx
        x = rem - y * nx
        ci = 1.0 + wall_cost * np.float64(elev[v0])
        base = np.float64(k0)
        for k in range(no):
            zz = z + dz[k]
            yy = y + dy[k]
            xx = x + dx[k]
            if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            j = (zz * ny + yy) * nx + xx
            if done[j] or not space[j]:
                continue
            cj = 1.0 + wall_cost * np.float64(elev[j])
            nd = np.float32(base + step_len[k] * 0.5 * (ci + cj))
            if nd > limit:
                continue
            dj = dist[j]
            if nd < dj or (nd == dj and l0 < lab[j]):
                dist[j] = nd
                lab[j] = l0
                pos = size
                size += 1
                while pos > 0:
                    p = (pos - 1) // 2
                    if nd < hk[p] or (nd == hk[p] and (
                            l0 < hl[p] or (l0 == hl[p] and j < hv[p]))):
                        hk[pos] = hk[p]
                        hl[pos] = hl[p]
                        hv[pos] = hv[p]
                        pos = p
                    else:
                        break
                hk[pos] = nd
                hl[pos] = l0
                hv[pos] = j
    return lab, dist, aborted


def _membrane_kernel(nz, ny, nx, cand, labels, kept, dz, dy, dx, group_end,
                     out):
    """For each candidate voxel, look outwards ring by ring (offsets grouped
    by equal physical distance): the first ring holding a claimed voxel
    decides -- a kept voxel there leaves it empty, else the lowest new id
    there takes it.  Reads ``labels`` / ``kept``, writes ``out``."""
    plane = ny * nx
    ng = group_end.shape[0]
    for t in range(cand.shape[0]):
        v = cand[t]
        z = v // plane
        rem = v - z * plane
        y = rem // nx
        x = rem - y * nx
        start = 0
        for g in range(ng):
            stop = group_end[g]
            found_kept = False
            best = 0
            for k in range(start, stop):
                zz = z + dz[k]
                yy = y + dy[k]
                xx = x + dx[k]
                if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                    continue
                j = (zz * ny + yy) * nx + xx
                if kept[j]:
                    found_kept = True
                else:
                    lj = labels[j]
                    if lj > 0 and (best == 0 or lj < best):
                        best = lj
            start = stop
            if found_kept:
                break
            if best > 0:
                out[v] = best
                break


_KERNELS = {}


def _kernels():
    """The compiled inner loops; numba is imported here, on first use.

    Raises ``ImportError(NUMBA_MISSING)`` when numba is missing or broken
    and ``RuntimeError(NUMBA_CACHE_FAILED)`` when numba finds no folder to
    cache the compiled code in.  Nothing is kept unless both loops are set
    up, so a later call tries again.
    """
    if not _KERNELS:
        try:
            import numba
        except Exception as exc:  # ImportError, or a broken install
            raise ImportError(NUMBA_MISSING) from exc
        try:
            jit = numba.njit(cache=True, nogil=True)
            built = {"grow": jit(_grow_kernel),
                     "membrane": jit(_membrane_kernel)}
        except RuntimeError as exc:  # "cannot cache function ...: no locator"
            raise RuntimeError(NUMBA_CACHE_FAILED) from exc
        _KERNELS.update(built)
    return _KERNELS


def have_numba():
    """True when the solver's inner loops can be set up (numba imports and
    has somewhere to cache them).  Cheap after the first call; nothing is
    compiled here -- the first solve compiles (about 10 s, once per
    install)."""
    try:
        _kernels()
    except (ImportError, RuntimeError):
        return False
    return True


# --------------------------------------------------------------------------- #
# input checks
# --------------------------------------------------------------------------- #
def _volume(a, what, shape=None):
    a = np.asarray(a)
    if a.ndim != 3:
        raise ValueError("{} must be a 3D volume, got {} dimensions.".format(
            what, a.ndim))
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("{} has shape {}, the elevation {}.".format(
            what, a.shape, tuple(shape)))
    return a


def _ids_int32(markers, what="markers"):
    """``markers`` as C-contiguous int32 (a copy only when needed); ids must
    be integers in ``0 .. 2**31 - 1``."""
    m = np.asarray(markers)
    if m.dtype == np.bool_ or not np.issubdtype(m.dtype, np.integer):
        raise ValueError("{} must hold integer ids, got dtype {}.".format(
            what, m.dtype))
    if m.size:
        lo, hi = int(m.min()), int(m.max())
        if lo < 0:
            raise ValueError("{} holds a negative id ({}).".format(what, lo))
        if hi > _MAX_ID:
            raise ValueError("{} holds id {}, above the largest the solver "
                             "takes ({}).".format(what, hi, _MAX_ID))
    return np.ascontiguousarray(m, dtype=np.int32)


def _elevation(elevation):
    e = _volume(elevation, "elevation")
    if not np.issubdtype(e.dtype, np.floating):
        raise ValueError("elevation must be a float wall map (0-1), got dtype "
                         "{}.".format(e.dtype))
    e = np.ascontiguousarray(e, dtype=np.float32)
    if e.size:
        lo = float(e.min())
        if not lo >= 0.0 or not math.isfinite(float(e.max())):
            raise ValueError("elevation must be finite and >= 0 (a 0-1 wall "
                             "map); its minimum is {!r}.".format(lo))
    return e


def _space(space, shape):
    s = _volume(space, "space", shape)
    if s.dtype != np.bool_:
        raise ValueError("space must be a boolean volume, got dtype "
                         "{}.".format(s.dtype))
    return np.ascontiguousarray(s)


def _price(wall_cost, growth_limit):
    w = float(wall_cost)
    g = float(growth_limit)
    if not (math.isfinite(w) and w >= 0.0):
        raise ValueError("wall_cost must be a finite number >= 0, got "
                         "{!r}.".format(wall_cost))
    if not (math.isfinite(g) and g > 0.0):
        raise ValueError("growth_limit must be a finite number > 0, got "
                         "{!r}.".format(growth_limit))
    return w, g


def _offsets(spacing, neighbours):
    if neighbours not in NEIGHBOURS:
        raise ValueError("neighbours must be one of {}, got {!r}.".format(
            ", ".join(str(n) for n in NEIGHBOURS), neighbours))
    rows = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                order = abs(dz) + abs(dy) + abs(dx)
                if order == 0:
                    continue
                if neighbours == 6 and order != 1:
                    continue
                if neighbours == 18 and order == 3:
                    continue
                rows.append((dz, dy, dx))
    d = np.array(rows, np.int64)
    length = np.sqrt(((d * np.asarray(spacing, np.float64)) ** 2).sum(axis=1))
    return (np.ascontiguousarray(d[:, 0]), np.ascontiguousarray(d[:, 1]),
            np.ascontiguousarray(d[:, 2]), np.ascontiguousarray(length))


def _integer_volume(a, what, shape):
    v = _volume(a, what, shape)
    if v.dtype == np.bool_ or not np.issubdtype(v.dtype, np.integer):
        raise ValueError("{} must hold integer ids, got dtype {}.".format(
            what, v.dtype))
    return v


class _StopWatch:
    """While the compiled solver runs (without the GIL), a daemon thread
    copies ``cancel.is_set()`` into ``flag[0]``, which the solver reads every
    :data:`STOP_CHECK_POPS` pops.  Without a ``cancel`` it does nothing."""

    def __init__(self, cancel):
        self.flag = np.zeros(1, np.uint8)
        self._cancel = cancel
        self._done = threading.Event()
        self._thread = None

    def _run(self):
        while not self._done.wait(_STOP_POLL_S):
            if self._cancel.is_set():
                self.flag[0] = 1
                return

    def __enter__(self):
        if self._cancel is not None:
            self._thread = threading.Thread(
                target=self._run, name="compete-stop-watch", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        if self._thread is not None:
            self._thread.join()
        return False


# --------------------------------------------------------------------------- #
# the competition
# --------------------------------------------------------------------------- #
def compete(elevation, markers, space, spacing_um, *, wall_cost=WALL_COST,
            growth_limit=GROWTH_LIMIT, neighbours=26, cancel=None):
    """Grow every marker into ``space`` by the cheapest path, within a budget.

    ``elevation`` -- float 0-1 wall map; ``markers`` -- integer ids (0 = no
    marker; ``uint32`` is fine up to ``2**31 - 1``); ``space`` -- bool, where
    paths may run and voxels may be claimed (marker voxels outside it are
    ignored); ``spacing_um`` -- (z, y, x).  A step costs its length in um x
    the mean of ``1 + wall_cost x elevation`` at its two ends; a voxel is
    claimed only while its total is ``<= growth_limit``; at an exactly equal
    total the lower id wins.

    Returns ``(labels int32, cost float32)``: the winning id (0 = unclaimed)
    and the total path cost (``inf`` where unclaimed).  ``cancel`` (an object
    with ``is_set()``) is polled before and after the solve, and inside it
    about once a second (see the module notes); a set cancel raises
    :class:`~.engine.ReshapeCancelled`.
    """
    e = _elevation(elevation)
    shape = e.shape
    mk = _ids_int32(_volume(markers, "markers", shape))
    sp_mask = _space(space, shape)
    sp = _spacing(spacing_um)
    w, g = _price(wall_cost, growth_limit)
    dz, dy, dx, length = _offsets(sp, int(neighbours))
    _check_cancel(cancel)
    grow = _kernels()["grow"]
    flat_mk = mk.reshape(-1)
    flat_space = sp_mask.reshape(-1)
    init_v = np.flatnonzero((flat_mk > 0) & flat_space)
    init_l = flat_mk[init_v]
    order = np.lexsort((init_v, init_l))
    init_v = np.ascontiguousarray(init_v[order], dtype=np.int64)
    init_l = np.ascontiguousarray(init_l[order], dtype=np.int32)
    del order
    with _StopWatch(cancel) as watch:
        lab, cost, aborted = grow(
            int(shape[0]), int(shape[1]), int(shape[2]), flat_space,
            e.reshape(-1), init_v, init_l, dz, dy, dx, length, w, g,
            watch.flag, _STOP_CHECK_MASK)
    if aborted:
        raise ReshapeCancelled()
    _check_cancel(cancel)
    return lab.reshape(shape), cost.reshape(shape)


def kept_competitors(markers, existing, space):
    """Seed every kept object as a competitor next to the new markers.

    Each kept object of ``existing`` is seeded on the voxels of ``space``
    6-adjacent to it that hold no marker; competitor ids follow the new
    markers (``n_new + 1 ..`` in ascending kept id; a voxel next to two kept
    objects goes to the higher kept id).  Returns ``(sources int32, n_new)``
    with ``n_new = markers.max()``.  After :func:`compete` the caller zeroes
    every claim above ``n_new``: a kept object never gains a voxel, it only
    keeps a new cell off its side of a gap.
    """
    mk = _ids_int32(_volume(markers, "markers"))
    ex = _volume(existing, "existing", mk.shape)
    sp_mask = _space(space, mk.shape)
    n_new = int(mk.max()) if mk.size else 0
    sources = mk.copy()
    nz = ex != 0
    if not nz.any():
        return sources, n_new
    kept_ids = np.unique(ex[nz])
    if n_new + int(kept_ids.size) > _MAX_ID:
        raise ValueError("too many ids for the solver: {} new markers and {} "
                         "kept objects.".format(n_new, kept_ids.size))
    rank = np.zeros(mk.shape, np.int32)
    rank[nz] = (np.searchsorted(kept_ids, ex[nz]) + 1).astype(np.int32)
    grown = ndi.grey_dilation(rank, footprint=ndi.generate_binary_structure(3, 1))
    del rank
    ring = (~nz) & (grown > 0) & sp_mask & (mk == 0)
    sources[ring] = grown[ring] + n_new
    return sources, n_new


def drop_sliver_markers(markers, empty, share=SLIVER_SHARE):
    """Drop every marker with less than ``share`` of its voxels on ``empty``.

    A marker mostly over kept objects is a sliver of a kept cell's own
    nucleus: grown, it makes a junk cell.  Returns ``(kept, n_dropped)``;
    ``kept`` keeps the surviving ids and dtype and is NOT clipped to
    ``empty`` (the caller does that, as before).  Exactly ``share`` stays.
    """
    mk = _volume(markers, "markers")
    if mk.dtype == np.bool_ or not np.issubdtype(mk.dtype, np.integer):
        raise ValueError("markers must hold integer ids, got dtype "
                         "{}.".format(mk.dtype))
    em = _volume(empty, "empty", mk.shape)
    if em.dtype != np.bool_:
        raise ValueError("empty must be a boolean volume, got dtype "
                         "{}.".format(em.dtype))
    share = float(share)
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must be in [0, 1], got {!r}.".format(share))
    out = np.array(mk, copy=True)
    nz = mk != 0
    if not nz.any():
        return out, 0
    _ids, inv = np.unique(mk[nz], return_inverse=True)
    inv = inv.reshape(-1)
    total = np.bincount(inv)
    on_empty = np.bincount(inv[em[nz]], minlength=total.size)
    drop = on_empty < share * total
    n_dropped = int(drop.sum())
    if n_dropped:
        vals = out[nz]
        vals[drop[inv]] = 0
        out[nz] = vals
    return out, n_dropped


# --------------------------------------------------------------------------- #
# the membrane at the edge
# --------------------------------------------------------------------------- #
def _shell_offsets(spacing, within_um):
    half = [int(math.floor(within_um / s + 1e-9)) for s in spacing]
    rows = []
    for dz in range(-half[0], half[0] + 1):
        for dy in range(-half[1], half[1] + 1):
            for dx in range(-half[2], half[2] + 1):
                if dz == dy == dx == 0:
                    continue
                d2 = ((dz * spacing[0]) ** 2 + (dy * spacing[1]) ** 2
                      + (dx * spacing[2]) ** 2)
                if d2 <= within_um * within_um + 1e-9:
                    rows.append((d2, dz, dy, dx))
    rows.sort()
    ends = []
    for i in range(1, len(rows)):
        if rows[i][0] - rows[i - 1][0] > 1e-9:
            ends.append(i)
    if rows:
        ends.append(len(rows))
    d = np.array([r[1:] for r in rows], np.int64).reshape(-1, 3)
    return (np.ascontiguousarray(d[:, 0]), np.ascontiguousarray(d[:, 1]),
            np.ascontiguousarray(d[:, 2]), np.array(ends, np.int64))


def include_membrane(labels, *, elevation, existing, space, spacing_um,
                     level=MEMBRANE_LEVEL, within_um=MEMBRANE_WITHIN_UM):
    """Hand the membrane at a new object's edge to that object.

    ``labels`` holds the NEW objects only (ids > 0; kept competitors already
    zeroed); ``existing`` the kept objects (integer ids); ``elevation`` the
    same float 0-1 wall map :func:`compete` ran on; ``space`` the same
    boolean space (pass ``np.ones(shape, bool)`` for no limit).  An
    unclaimed voxel (0 in both, inside ``space``) with
    ``elevation >= level`` goes to the nearest claimed voxel's object within
    ``within_um`` (physical distance), and only when that object is new.  At
    an equal distance a kept object wins (the voxel stays empty) and, among
    new objects, the lower id.  Returns a new int32 volume; the inputs are
    not changed.
    """
    lab = _ids_int32(_volume(labels, "labels"))
    shape = lab.shape
    e = _elevation(elevation)
    if e.shape != shape:
        raise ValueError("elevation has shape {}, the labels {}.".format(
            e.shape, shape))
    ex = _integer_volume(existing, "existing", shape)
    sp_mask = _space(space, shape)
    sp = _spacing(spacing_um)
    level = float(level)
    if not math.isfinite(level):
        raise ValueError("level must be a finite number, got {!r}.".format(level))
    within_um = float(within_um)
    if not (math.isfinite(within_um) and within_um >= 0.0):
        raise ValueError("within_um must be a finite number >= 0, got "
                         "{!r}.".format(within_um))
    out = lab.copy()
    kept = np.ascontiguousarray(ex != 0)
    cand = (lab == 0) & ~kept & (e >= level) & sp_mask
    dz, dy, dx, ends = _shell_offsets(sp, within_um)
    if ends.size == 0 or not cand.any():
        return out
    cand_idx = np.flatnonzero(cand)
    del cand
    _kernels()["membrane"](int(shape[0]), int(shape[1]), int(shape[2]),
                           cand_idx, lab.reshape(-1), kept.reshape(-1),
                           dz, dy, dx, ends, out.reshape(-1))
    return out


# --------------------------------------------------------------------------- #
# tiles
# --------------------------------------------------------------------------- #
def tile_halo(spacing_um, growth_limit):
    """The halo in voxels, per axis (z, y, x):
    ``ceil((growth_limit + 1 um) / spacing_axis) + 2``.

    Every step costs at least its length, so no affordable path is longer
    than ``growth_limit`` um; a path that long moves at most
    ``growth_limit / spacing_axis`` voxels along an axis, so none leaves a
    tile's halo.  On a coarse z the z halo is that much thinner (z 2 um,
    xy 0.5 um, limit 14: 10 slices, 32 voxels in y and x).
    """
    sp = _spacing(spacing_um)
    _w, g = _price(0.0, growth_limit)
    return tuple(int(math.ceil((g + TILE_HALO_EXTRA_UM) / s)) + TILE_HALO_EXTRA_VOXELS
                 for s in sp)


def _axis_options(n, halo):
    """Per distinct core size along an axis of ``n`` voxels:
    ``(core, worst crop, summed crop lengths)``, fewest tiles first."""
    seen = set()
    out = []
    for k in range(1, n + 1):
        c = -(-n // k)
        if c in seen:
            continue
        seen.add(c)
        starts = range(0, n, c)
        lens = [min(n, s + c + halo) - max(0, s - halo) for s in starts]
        out.append((c, max(lens), sum(lens)))
    return out


def plan_tiles(shape, spacing_um, growth_limit, budget=COMPETE_TILE_VOXELS):
    """Tiles in all three axes for a volume of ``shape``.

    The halo is :func:`tile_halo` (per axis); the budget per tile (halo
    included) is ``max(budget, 4 hz x 4 hy x 4 hx)``.  Among every regular
    grid whose tiles all fit, the one that processes the fewest voxels in
    total is chosen.  Returns a list of ``(core, halo)`` -- each a tuple of
    three slices; the cores partition the volume, row-major order.
    """
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("shape must be three positive sizes, got "
                         "{!r}.".format(shape))
    halos = tile_halo(spacing_um, growth_limit)
    cap = max(int(budget), int(np.prod([4 * h for h in halos], dtype=np.int64)))
    opts = [_axis_options(n, h) for n, h in zip(shape, halos)]
    best = None
    for cz, wz, sz in opts[0]:
        for cy, wy, sy in opts[1]:
            if wz * wy > cap:
                continue
            for cx, wx, sx in opts[2]:
                if wz * wy * wx > cap:
                    continue
                proc = sz * sy * sx
                if best is None or proc < best[0]:
                    best = (proc, (cz, cy, cx))
                break       # more tiles along x only cost more
    core = best[1]
    tiles = []
    for z0 in range(0, shape[0], core[0]):
        for y0 in range(0, shape[1], core[1]):
            for x0 in range(0, shape[2], core[2]):
                c = tuple(slice(a, min(a + k, n)) for a, k, n in
                          zip((z0, y0, x0), core, shape))
                h = tuple(slice(max(0, s.start - hw), min(n, s.stop + hw))
                          for s, hw, n in zip(c, halos, shape))
                tiles.append((c, h))
    return tiles


def compete_tiled(elevation, markers, space, spacing_um, *,
                  wall_cost=WALL_COST, growth_limit=GROWTH_LIMIT, neighbours=26,
                  budget=COMPETE_TILE_VOXELS, cancel=None, progress=None,
                  return_cost=False):
    """:func:`compete` tile by tile -- the same answer, voxel for voxel.

    Each tile runs on crops of the ONE ``elevation`` and ONE ``markers``
    volume (never rebuilt per tile) and writes its core only; a plan of one
    tile is a single :func:`compete` call.  ``cancel`` is polled before
    every tile and inside each solve (and raises
    :class:`~.engine.ReshapeCancelled`); ``progress(i, k)`` is called after
    tile ``i`` of ``k`` -- an exception it raises (a cancel included)
    propagates and stops the run.

    Returns the labels (int32), or ``(labels, cost)`` with
    ``return_cost=True`` (the cost volume costs 4 B per voxel more).
    """
    e = _elevation(elevation)
    shape = e.shape
    mk = _ids_int32(_volume(markers, "markers", shape))
    sp_mask = _space(space, shape)
    sp = _spacing(spacing_um)
    _price(wall_cost, growth_limit)
    _offsets(sp, int(neighbours))
    tiles = plan_tiles(shape, sp, growth_limit, budget)
    kwargs = dict(wall_cost=wall_cost, growth_limit=growth_limit,
                  neighbours=neighbours, cancel=cancel)
    if len(tiles) == 1:
        _check_cancel(cancel)
        labels, cost = compete(e, mk, sp_mask, sp, **kwargs)
        if progress is not None:
            progress(1, 1)
        _check_cancel(cancel)
        return (labels, cost) if return_cost else labels
    labels = np.zeros(shape, np.int32)
    cost = np.full(shape, np.inf, np.float32) if return_cost else None
    for i, (core, halo) in enumerate(tiles, start=1):
        _check_cancel(cancel)
        lab, cst = compete(e[halo], mk[halo], sp_mask[halo], sp, **kwargs)
        inner = tuple(slice(c.start - h.start, c.stop - h.start)
                      for c, h in zip(core, halo))
        labels[core] = lab[inner]
        if return_cost:
            cost[core] = cst[inner]
        del lab, cst
        if progress is not None:
            progress(i, len(tiles))
    _check_cancel(cancel)
    return (labels, cost) if return_cost else labels

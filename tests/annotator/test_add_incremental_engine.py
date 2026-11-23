"""The markers' search made incremental: prepare once, solve per click.

``add_incremental.solve_markers`` against the one-go answer
(``add_segment.segment_at_markers``, numbered by the click ids, then
``add_segment.smooth_pieces``) on synthetic touching nuclei, over random
click edits (add, remove, move), with the split off and in both split modes,
the smoothing on and off, Local contrast on and off.

What is exact:

* the split off with one marker per clicked piece: equal to the one-go
  answer, voxel for voxel;
* otherwise each clicked piece is solved on its own markers: equal to
  ``segment_at_markers`` run on that piece's clicks alone (a run over every
  click can hand a few tie voxels inside a piece with two seeds or more to a
  different object -- the flood's queue order depends on every seed in the
  call); measured and printed, never asserted equal;
* the smoothing, in every case.

Qt-free.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from packages.epicycle_annotator.add_incremental import (
    _claims,
    _raw_ball,
    prepare_markers_box,
    solve_markers,
)
from packages.epicycle_annotator.add_preview import _mint_to
from packages.epicycle_annotator.add_segment import (
    _claim_piece,
    add_threshold_space,
    segment_at_markers,
    smooth_pieces,
)
from packages.epicycle_annotator.reshape.region import rasterise_points

RADIUS = 1.0
PEAK_UM = 2.0
SIGMA = 1.5


# --------------------------------------------------------------- fixtures
def nuclei(shape, spacing, seed, *, r_um=(2.2, 3.4), density=0.35,
           kept=True):
    """A nuclear stain of touching nuclei: about a third come as a pair
    pressed into each other; noise; and (``kept``) a few kept objects, one
    over a nucleus."""
    rng = np.random.default_rng(seed)
    sp = np.asarray(spacing, float)
    ext = np.asarray(shape) * sp
    img = rng.normal(0.1, 0.03, shape).astype(np.float32)
    n = max(4, int(np.prod(ext) * density / (4 / 3 * np.pi * r_um[1] ** 3)))
    centres = []
    for i in range(n):
        r = rng.uniform(*r_um)
        if centres and i % 3 == 1:
            c0, r0 = centres[-1]
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            c = c0 + d * (r0 + r) * 0.8
        else:
            c = rng.uniform(0, ext)
        centres.append((c, r))
    for c, r in centres:
        cv = c / sp
        h = [int(np.ceil(r / s)) + 1 for s in sp]
        sl = tuple(slice(max(0, int(cv[a]) - h[a]),
                         min(shape[a], int(cv[a]) + h[a] + 2))
                   for a in range(3))
        if any(s.start >= s.stop for s in sl):
            continue
        g = np.ogrid[sl]
        d2 = sum(((g[a] - cv[a]) * sp[a]) ** 2 for a in range(3))
        img[sl][d2 <= r * r] += 0.8
    ex = np.zeros(shape, np.int32)
    if kept:
        z, y, x = (s // 4 for s in shape)
        ex[z:z + 3, y:y + 6, x:x + 6] = 5
        c, r = centres[0]
        cv = (c / sp).astype(int)
        lo = np.maximum(cv - 2, 0)
        hi = np.minimum(cv + 3, shape)
        ex[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 9
    return np.clip(img, 0, None), ex


def pieces_of(img, ex, clahe):
    """The bright pieces, computed here (not by the engine): Otsu in the
    Local contrast space, kept voxels out, 6-connected."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    proc = add_threshold_space(img, clahe=clahe)
    fg = (proc >= threshold_otsu(proc)) & (ex == 0)
    comp, _n = ndi.label(fg)
    return fg, comp


def one_go(img, ex, pts, ids, *, spacing, clahe, split, min_size, sigma):
    """The answer of one run over every click: segment_at_markers, numbered
    by ``ids``, smoothed."""
    pieces, why = segment_at_markers(
        img, ex, pts, radius_um=RADIUS, spacing=spacing, clahe=clahe,
        min_size=min_size, split=split, peak_spacing_um=PEAK_UM)
    mapping = {k: ids[k - 1] for k in range(1, len(pts) + 1) if k not in why}
    raw = (_mint_to(pieces, mapping) if mapping
           else np.zeros(pieces.shape, np.int32))
    result = smooth_pieces(raw, sigma) if sigma is not None else raw
    return raw, result, why


def per_piece(img, ex, pts, ids, comp, *, spacing, clahe, split, min_size,
              sigma):
    """Each clicked piece solved on its own clicks (every click here lies on
    the stain, so it claims the piece under it, and no two clicks of
    different pieces share a ball voxel): segment_at_markers on that
    piece's clicks alone, kept inside the piece, numbered by ``ids``, then
    smoothed."""
    raw = np.zeros(comp.shape, np.int32)
    why = {}
    owner = [int(comp[tuple(p)]) for p in pts]
    for c in sorted(set(owner)):
        idx = [i for i in range(len(pts)) if owner[i] == c]
        sub = [pts[i] for i in idx]
        alone, w = segment_at_markers(
            img, ex, sub, radius_um=RADIUS, spacing=spacing, clahe=clahe,
            min_size=min_size, split=split, peak_spacing_um=PEAK_UM)
        lut = np.zeros(len(sub) + 1, np.int32)
        for j, i in enumerate(idx, start=1):
            lut[j] = ids[i]
            if j in w:
                why[i + 1] = w[j]
        inside = comp == c
        raw[inside] = lut[alone[inside]]
    result = smooth_pieces(raw, sigma) if sigma is not None else raw
    return raw, result, dict(sorted(why.items()))


# ------------------------------------------------------------ click edits
class Clicks:
    """A click list with stable ids (each click keeps its id through
    edits; ids are not in placement order, so the smoothing's id order is
    exercised) and random edits that keep the fixture's promise."""

    def __init__(self, rng, comp, spacing, *, one_per_piece, max_per_piece=3):
        self.rng = rng
        self.comp = comp
        self.half = [int(np.ceil(RADIUS / s)) for s in spacing]
        self.one = one_per_piece
        self.max_per = max_per_piece
        self.pts, self.ids = [], []
        self.next_id = 40
        sizes = np.bincount(comp.ravel())
        self.big = [c for c in range(1, len(sizes)) if sizes[c] >= 60]
        self.extra = []      # off-stain / kept clicks (one-per-piece runs)

    def _owner(self, p):
        return int(self.comp[tuple(p)])

    def _counts(self, skip=None):
        n = {}
        for i, p in enumerate(self.pts):
            if i != skip and self._owner(p):
                n[self._owner(p)] = n.get(self._owner(p), 0) + 1
        return n

    def _far_from_others(self, p, c, skip=None):
        """No ball voxel shared with a click of another piece."""
        for i, q in enumerate(self.pts):
            if i == skip or self._owner(q) in (c, 0):
                continue
            if all(abs(p[a] - q[a]) <= 2 * self.half[a] for a in range(3)):
                return False
        return True

    def _pick(self, skip=None):
        counts = self._counts(skip)
        if self.one:
            pool = [c for c in self.big if c not in counts]
        else:
            pool = [c for c in self.big if counts.get(c, 0) < self.max_per]
            # favour pieces already clicked: two or more seeds per piece
            again = [c for c in pool if c in counts]
            if again and self.rng.random() < 0.6:
                pool = again
        for _try in range(40):
            if not pool:
                return None
            c = int(self.rng.choice(pool))
            vox = np.argwhere(self.comp == c)
            p = tuple(int(v) for v in vox[self.rng.integers(len(vox))])
            if self._far_from_others(p, c, skip):
                return p
        return None

    def add(self):
        p = self._pick()
        if p is not None:
            self.pts.append(p)
            self.ids.append(self.next_id)
            self.next_id += int(self.rng.integers(1, 7))

    def remove(self):
        if self.pts:
            i = int(self.rng.integers(len(self.pts)))
            del self.pts[i]
            del self.ids[i]

    def move(self):
        if self.pts:
            i = int(self.rng.integers(len(self.pts)))
            p = self._pick(skip=i)
            if p is not None:
                self.pts[i] = p

    def edit(self):
        r = self.rng.random()
        if r < 0.45 or len(self.pts) < 2:
            self.add()
        elif r < 0.7:
            self.remove()
        else:
            self.move()


def check_diff(prev, sol, diff):
    """The diff turns the previous drawing into the new one, names every
    id whose drawn voxels changed, and nothing else; its patches never
    overlap and hold every changed voxel."""
    before = prev.result
    after = sol.result
    changed = before != after
    want = set(np.unique(before[changed]).tolist())
    want |= set(np.unique(after[changed]).tolist())
    want.discard(0)
    assert set(diff.changed_ids) == want
    if not changed.any():
        assert diff.patches == ()
        return
    drawn = before.copy()
    cover = np.zeros(before.shape, np.int32)
    for box, patch in diff.patches:
        drawn[box] = patch
        cover[box] += 1
    np.testing.assert_array_equal(drawn, after)
    assert cover.max() == 1
    assert not (changed & (cover == 0)).any()


def check_counts(sol):
    """The sizes, the ids and the raw ids kept by the solve are the ones a
    count over the whole arrays gives."""
    vals, counts = np.unique(sol.result, return_counts=True)
    want = {int(v): int(n) for v, n in zip(vals, counts) if v}
    assert sol.sizes == want
    assert sol.ids == tuple(sorted(want))
    raw_ids = tuple(int(v) for v in np.unique(sol.raw) if v)
    assert sol.raw_ids == raw_ids
    assert sol.emptied_ids == tuple(sorted(set(raw_ids) - set(want)))


def run_sequence(img, ex, spacing, *, clahe, split, sigma, one_per_piece,
                 seed, n_start=5, n_edits=6, min_size=0, reference):
    fg, comp = pieces_of(img, ex, clahe)
    rng = np.random.default_rng(seed)
    clicks = Clicks(rng, comp, spacing, one_per_piece=one_per_piece)
    for _ in range(n_start):
        clicks.add()
    stage = prepare_markers_box(
        img, ex, radius_um=RADIUS, spacing=spacing, clahe=clahe,
        min_size=min_size, split=split, peak_spacing_um=PEAK_UM)
    np.testing.assert_array_equal(stage.fg, fg)
    prev = None
    n_multi = 0
    for step in range(n_edits + 1):
        if step:
            clicks.edit()
        pts, ids = list(clicks.pts), list(clicks.ids)
        keep = None
        if prev is not None:
            keep = (prev.raw.copy(), prev.result.copy())
        sol, diff = solve_markers(stage, pts, ids, smooth_sigma=sigma,
                                  previous=prev)
        check_counts(sol)
        if prev is not None:
            # the solve never writes into what the caller holds
            np.testing.assert_array_equal(prev.raw, keep[0])
            np.testing.assert_array_equal(prev.result, keep[1])
            assert not diff.full
            check_diff(prev, sol, diff)
        raw_ref, res_ref, why_ref = reference(
            img, ex, pts, ids, comp, spacing=spacing, clahe=clahe,
            split=split, min_size=min_size, sigma=sigma)
        np.testing.assert_array_equal(sol.raw, raw_ref)
        np.testing.assert_array_equal(sol.result, res_ref)
        assert sol.why == why_ref
        # the same list from scratch gives the same voxels
        fresh, fdiff = solve_markers(stage, pts, ids, smooth_sigma=sigma)
        assert fdiff.full
        np.testing.assert_array_equal(fresh.result, sol.result)
        owners = [int(comp[tuple(p)]) for p in pts]
        n_multi += sum(1 for c in set(owners) if owners.count(c) > 1)
        prev = sol
    return n_multi


def _one_go_ref(img, ex, pts, ids, comp, **kw):
    return one_go(img, ex, pts, ids, **kw)


# ---------------------------------------------------------------- pinned
def test_the_engine_s_balls_are_rasterise_points_balls():
    """An earlier ball keeps a shared voxel; half voxels round as numpy
    rounds; a centre outside the box draws only the part inside."""
    rng = np.random.default_rng(3)
    shape = (9, 14, 13)
    for spacing in ((1.0, 1.0, 1.0), (2.0, 0.5, 0.5), (0.7, 0.3, 0.3)):
        stage = prepare_markers_box(
            np.zeros(shape), np.zeros(shape, np.int32), radius_um=1.3,
            spacing=spacing)
        for _ in range(20):
            pts = [tuple(rng.integers(-2, n + 2) + rng.choice([0, 0.5, 0.25])
                         for n in shape) for _ in range(6)]
            want = rasterise_points(pts, shape, radius_um=1.3,
                                    spacing=spacing)
            got = np.zeros(shape, np.int32)
            for k, p in enumerate(pts, start=1):
                rb = _raw_ball(stage, p)
                if rb is None:
                    continue
                box, ball = rb
                view = got[box]
                view[ball & (view == 0)] = k
            np.testing.assert_array_equal(got, want)


def test_the_stage_is_read_only():
    img, ex = nuclei((16, 40, 40), (1.0, 1.0, 1.0), 0)
    stage = prepare_markers_box(img, ex, radius_um=RADIUS,
                                spacing=(1.0, 1.0, 1.0))
    for a in (stage.fg, stage.comp, stage.empty):
        with pytest.raises(ValueError):
            a[0, 0, 0] = 1


# ------------------------------------------- exact: split off, one per piece
@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0),
                                     (1.0, 0.4, 0.4)])
@pytest.mark.parametrize("sigma", [None, SIGMA])
@pytest.mark.parametrize("clahe", [False, True])
def test_off_one_marker_per_piece_equals_the_one_go_answer(spacing, sigma,
                                                           clahe):
    shape = (26, 60, 60) if spacing[1] == 1.0 else (26, 110, 110)
    img, ex = nuclei(shape, spacing, 11)
    run_sequence(img, ex, spacing, clahe=clahe, split="off", sigma=sigma,
                 one_per_piece=True, seed=5, reference=_one_go_ref)


def test_off_one_marker_per_piece_with_small_objects_dropped():
    spacing = (1.0, 1.0, 1.0)
    img, ex = nuclei((26, 60, 60), spacing, 12)
    run_sequence(img, ex, spacing, clahe=False, split="off", sigma=SIGMA,
                 one_per_piece=True, seed=6, min_size=90,
                 reference=_one_go_ref)


def test_off_kept_dim_near_miss_and_shadowed_balls_equal_the_one_go_answer():
    """Clicks on a kept object, off the stain, a near miss beside a piece,
    and two clicks of two pieces whose balls share voxels (the earlier keeps
    them): removing the earlier click reshapes the later one's marker."""
    spacing = (1.0, 1.0, 1.0)
    shape = (12, 30, 40)
    img = np.full(shape, 0.1, np.float32)
    img[3:9, 5:14, 4:13] = 1.0          # piece A
    img[3:9, 15:24, 4:13] = 1.0         # piece B, one dark row from A
    img[3:9, 5:14, 25:34] = 1.0         # piece C (a near miss beside it)
    ex = np.zeros(shape, np.int32)
    ex[3:9, 20:28, 30:38] = 7           # a kept object
    img[3:9, 20:28, 30:38] = 1.0
    pts = [(6, 13, 8),                  # A, by the gap
           (6, 15, 8),                  # B: its own voxel is in A's ball
           (6, 24, 34),                 # inside the kept object
           (6, 2, 20),                  # off the stain
           (6, 9, 23)]                  # near miss beside C
    ids = [3, 11, 2, 8, 6]
    stage = prepare_markers_box(img, ex, radius_um=2.5, spacing=spacing)
    prev = None
    for p, i in [(pts, ids), (pts[1:], ids[1:]), (pts[2:], ids[2:]),
                 (pts, ids)]:
        sol, diff = solve_markers(stage, p, i, smooth_sigma=SIGMA,
                                  previous=prev)
        check_counts(sol)
        pieces, why = segment_at_markers(img, ex, p, radius_um=2.5,
                                         spacing=spacing)
        mapping = {k: i[k - 1] for k in range(1, len(p) + 1)
                   if k not in why}
        raw = _mint_to(pieces, mapping)
        np.testing.assert_array_equal(sol.raw, raw)
        np.testing.assert_array_equal(sol.result, smooth_pieces(raw, SIGMA))
        assert sol.why == why
        if prev is not None:
            check_diff(prev, sol, diff)
        prev = sol
    assert prev.why == {3: "kept", 4: "dim"}


# ------------------------------- per piece: several seeds, the split modes
@pytest.mark.parametrize("split,sigma,clahe,spacing", [
    ("off", SIGMA, False, (1.0, 1.0, 1.0)),
    ("off", None, True, (2.0, 1.0, 1.0)),
    ("single", SIGMA, True, (1.0, 1.0, 1.0)),
    ("single", None, False, (2.0, 1.0, 1.0)),
    ("always", SIGMA, False, (2.0, 1.0, 1.0)),
    ("always", None, True, (1.0, 1.0, 1.0)),
    ("always", SIGMA, True, (1.0, 0.4, 0.4)),
    ("single", SIGMA, False, (1.0, 0.4, 0.4)),
])
def test_each_piece_solved_on_its_own_markers(split, sigma, clahe, spacing):
    shape = (26, 60, 60) if spacing[1] == 1.0 else (26, 110, 110)
    img, ex = nuclei(shape, spacing, 21)
    n_multi = run_sequence(img, ex, spacing, clahe=clahe, split=split,
                           sigma=sigma, one_per_piece=False, seed=8,
                           reference=per_piece)
    assert n_multi > 0, "the sequence never put two clicks in one piece"


def test_one_marker_pieces_equal_the_one_go_answer_beside_several_seed_ones():
    """With the split off, a piece holding one marker comes out as one run
    over every click gives it even when other pieces hold several; the
    several-seed pieces may differ by tie voxels (printed)."""
    spacing = (1.0, 1.0, 1.0)
    img, ex = nuclei((26, 60, 60), spacing, 31)
    _fg, comp = pieces_of(img, ex, False)
    stage = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing)
    rng = np.random.default_rng(2)
    total_diff = n_multi = n_multi_diff = 0
    for _trial in range(4):
        clicks = Clicks(rng, comp, spacing, one_per_piece=False)
        for _ in range(12):
            clicks.add()
        pts, ids = clicks.pts, clicks.ids
        sol, _d = solve_markers(stage, pts, ids)
        raw, _r, _w = one_go(img, ex, pts, ids, spacing=spacing, clahe=False,
                             split="off", min_size=0, sigma=None)
        owners = [int(comp[tuple(p)]) for p in pts]
        for c in set(owners):
            inside = comp == c
            d = int(np.count_nonzero(sol.raw[inside] != raw[inside]))
            if owners.count(c) == 1:
                assert d == 0
            else:
                n_multi += 1
                n_multi_diff += bool(d)
                total_diff += d
    print(f"[ADD-INC] several-seed pieces: {n_multi}, differing from one "
          f"run over every click: {n_multi_diff}, voxels: {total_diff}")


def test_one_run_s_answer_for_a_piece_depends_on_the_other_pieces_clicks():
    """Why the answer is per piece: one run over every click can give a
    piece holding several seeds a different answer from a run over that
    piece's clicks alone -- the flood starts every seed with the same age,
    so which of two equal-height seeds goes first is set by the layout of
    its queue, which every seed in the call shapes.  The engine gives the
    piece's own answer, whatever the other pieces hold.  (A witness found by
    search; if a later skimage breaks ties by position, this test fails and
    the narrower claim can be widened.)"""
    spacing = (1.0, 1.0, 1.0)
    img, ex = nuclei((26, 60, 60), spacing, 6)
    _fg, comp = pieces_of(img, ex, False)
    stage = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing)
    rng = np.random.default_rng(6)
    clicks = Clicks(rng, comp, spacing, one_per_piece=False)
    for _ in range(14):
        clicks.add()
    pts, ids = clicks.pts, clicks.ids
    sol, _d = solve_markers(stage, pts, ids)
    raw_all, _r, _w = one_go(img, ex, pts, ids, spacing=spacing, clahe=False,
                             split="off", min_size=0, sigma=None)
    owners = [int(comp[tuple(p)]) for p in pts]
    coupled = 0
    for c in sorted(set(owners)):
        inside = comp == c
        idx = [i for i in range(len(pts)) if owners[i] == c]
        alone, _r2, _w2 = one_go(img, ex, [pts[i] for i in idx],
                                 [ids[i] for i in idx], spacing=spacing,
                                 clahe=False, split="off", min_size=0,
                                 sigma=None)
        np.testing.assert_array_equal(sol.raw[inside], alone[inside])
        if not np.array_equal(raw_all[inside], alone[inside]):
            assert len(idx) > 1
            coupled += 1
    assert coupled >= 1


# --------------------------------------------------------------- contract
def test_a_previous_solution_on_another_stage_or_smoothing_solves_all():
    spacing = (1.0, 1.0, 1.0)
    img, ex = nuclei((16, 40, 40), spacing, 4)
    _fg, comp = pieces_of(img, ex, False)
    a = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing)
    b = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing)
    vox = np.argwhere(comp > 0)
    pts = [tuple(int(v) for v in vox[0])]
    sa, _ = solve_markers(a, pts)
    _sb, db = solve_markers(b, pts, previous=sa)
    assert db.full
    _sc, dc = solve_markers(a, pts, smooth_sigma=SIGMA, previous=sa)
    assert dc.full
    _sd, dd = solve_markers(a, pts, previous=sa)
    assert not dd.full and dd.changed_ids == () and dd.patches == ()


def test_no_clicks_and_back():
    spacing = (1.0, 1.0, 1.0)
    img, ex = nuclei((16, 40, 40), spacing, 4)
    _fg, comp = pieces_of(img, ex, False)
    stage = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing)
    empty, d0 = solve_markers(stage, [], smooth_sigma=SIGMA)
    assert d0.full and not empty.result.any() and empty.why == {}
    vox = np.argwhere(comp > 0)
    pts = [tuple(int(v) for v in vox[len(vox) // 2])]
    one, d1 = solve_markers(stage, pts, [17], smooth_sigma=SIGMA,
                            previous=empty)
    check_diff(empty, one, d1)
    check_counts(one)
    assert 17 in d1.changed_ids
    none, d2 = solve_markers(stage, [], [], smooth_sigma=SIGMA, previous=one)
    check_diff(one, none, d2)
    check_counts(none)
    assert not none.result.any() and none.sizes == {}


def test_a_remove_and_an_add_far_apart_come_back_as_two_small_patches():
    spacing = (1.0, 1.0, 1.0)
    shape = (12, 30, 90)
    img = np.full(shape, 0.1, np.float32)
    img[3:9, 5:14, 4:13] = 1.0
    img[3:9, 15:24, 70:80] = 1.0
    ex = np.zeros(shape, np.int32)
    stage = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing)
    a, _ = solve_markers(stage, [(6, 9, 8)], [4], smooth_sigma=SIGMA)
    b, diff = solve_markers(stage, [(6, 19, 75)], [5], smooth_sigma=SIGMA,
                            previous=a)
    check_diff(a, b, diff)
    check_counts(b)
    assert len(diff.patches) == 2
    vox = sum(int(np.prod(p.shape)) for _box, p in diff.patches)
    assert vox < np.prod(shape) / 4


def test_the_solution_and_its_arrays_cannot_change():
    import dataclasses

    spacing = (1.0, 1.0, 1.0)
    img, ex = nuclei((16, 40, 40), spacing, 4)
    _fg, comp = pieces_of(img, ex, False)
    stage = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing)
    pts = [tuple(int(v) for v in np.argwhere(comp > 0)[0])]
    for sigma in (None, SIGMA):
        sol, _ = solve_markers(stage, pts, smooth_sigma=sigma)
        for a in (sol.raw, sol.result):
            with pytest.raises(ValueError):
                a[0, 0, 0] = 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            sol.why = {}
        nxt, _ = solve_markers(stage, [], smooth_sigma=sigma, previous=sol)
        assert not nxt.result.any()


def test_a_tissue_outline_is_refused():
    with pytest.raises(ValueError):
        prepare_markers_box(np.zeros((4, 5, 6)),
                            np.zeros((4, 5, 6), np.int32), radius_um=1.0,
                            spacing=(1, 1, 1),
                            outline=np.ones((4, 5, 6), bool))


# ------------------------------------------------------------------ groups
def _two_touching(spacing=(1.0, 1.0, 1.0)):
    """Two round nuclei pressed into each other: one bright piece."""
    shape = (16, 30, 44)
    z, y, x = np.indices(shape)
    img = np.full(shape, 0.1, np.float32)
    for cx in (14, 27):
        img[((z - 8) ** 2 + (y - 15) ** 2 + (x - cx) ** 2) <= 49] = 1.0
    return img, np.zeros(shape, np.int32)


@pytest.mark.parametrize("sigma", [None, SIGMA])
def test_two_seeds_in_one_piece_under_one_group_are_one_whole_object(sigma):
    """With the split off, two seeds of one group on one piece make one
    object covering the whole piece (segment_at_markers has no groups yet:
    the pin is the off split's whole-piece answer)."""
    img, ex = _two_touching()
    _fg, comp = pieces_of(img, ex, False)
    stage = prepare_markers_box(img, ex, radius_um=RADIUS,
                                spacing=(1.0, 1.0, 1.0))
    pts = [(8, 15, 12), (8, 15, 29)]
    assert comp[pts[0]] == comp[pts[1]] != 0
    two, _ = solve_markers(stage, pts, [7, 9], smooth_sigma=sigma)
    assert set(np.unique(two.raw[comp > 0])) == {7, 9}
    one, diff = solve_markers(stage, pts, [7, 7], groups=[0, 0],
                              smooth_sigma=sigma, previous=two)
    np.testing.assert_array_equal(one.raw, np.where(comp > 0, 7, 0))
    assert one.why == {}
    check_diff(two, one, diff)
    check_counts(one)
    # the same object as one seed alone gives
    alone, _ = solve_markers(stage, pts[:1], [7], smooth_sigma=sigma)
    np.testing.assert_array_equal(one.result, alone.result)


def test_a_group_seed_on_another_piece_is_apart_as_if_never_placed():
    spacing = (1.0, 1.0, 1.0)
    shape = (12, 30, 40)
    img = np.full(shape, 0.1, np.float32)
    img[3:9, 5:14, 4:13] = 1.0
    img[3:9, 18:27, 4:13] = 1.0
    ex = np.zeros(shape, np.int32)
    stage = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing)
    pts = [(6, 9, 8), (6, 22, 8), (6, 2, 30)]
    sol, _ = solve_markers(stage, pts, [3, 3, 3], groups=["a", "a", "a"])
    assert sol.seed_why == {2: "apart", 3: "dim"}
    assert sol.why == {}
    assert sol.ids == (3,)
    assert sol.raw[6, 22, 8] == 0 and sol.raw[6, 9, 8] == 3
    pieces, why, seed_why = segment_at_markers(
        img, ex, pts, radius_um=RADIUS, spacing=spacing,
        groups=["a", "a", "a"])
    assert seed_why == sol.seed_why and why == sol.why
    np.testing.assert_array_equal(sol.raw, pieces * 3)


def test_group_ids_are_checked():
    stage = prepare_markers_box(np.zeros((4, 5, 6)),
                                np.zeros((4, 5, 6), np.int32),
                                radius_um=1.0, spacing=(1, 1, 1))
    pts = [(1, 1, 1), (2, 2, 2), (2, 3, 3)]
    with pytest.raises(ValueError):       # one group, two ids
        solve_markers(stage, pts, [1, 2, 3], groups=[0, 0, 1])
    with pytest.raises(ValueError):       # two groups, one id
        solve_markers(stage, pts, [1, 1, 1], groups=[0, 0, 1])
    with pytest.raises(ValueError):
        solve_markers(stage, pts, [1, 1, 2], groups=[0, 0])
    sol, _ = solve_markers(stage, pts, groups=[5, 5, 8])
    assert sol.click_ids == (1, 1, 2)


def test_ids_must_match_the_points():
    stage = prepare_markers_box(np.zeros((4, 5, 6)),
                                np.zeros((4, 5, 6), np.int32),
                                radius_um=1.0, spacing=(1, 1, 1))
    with pytest.raises(ValueError):
        solve_markers(stage, [(1, 1, 1)], [1, 2])
    with pytest.raises(ValueError):
        solve_markers(stage, [(1, 1, 1), (2, 2, 2)], [3, 3])
    with pytest.raises(ValueError):
        prepare_markers_box(np.zeros((4, 5, 6)), np.zeros((4, 5, 6)),
                            radius_um=1.0, spacing=(1, 1, 1), split="twice")


# ------------------------------------------------------------------ timing
@pytest.mark.parametrize("split", ["off", "always"])
def test_one_click_on_a_whole_patch_is_fast_after_the_first(split):
    """91 x 240 x 240, the smoothing on: one more click, one click removed
    and one click moved after the first solve.  The budget is 300 ms a
    click; the measured times are printed."""
    spacing = (1.0, 0.4, 0.4)
    shape = (91, 240, 240)
    img, ex = nuclei(shape, spacing, 0, r_um=(2.5, 4.0), density=0.12,
                     kept=False)
    _fg, comp = pieces_of(img, ex, False)
    t = time.perf_counter()
    stage = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing,
                                split=split, peak_spacing_um=PEAK_UM)
    t_prep = (time.perf_counter() - t) * 1000.0
    rng = np.random.default_rng(9)
    clicks = Clicks(rng, comp, spacing, one_per_piece=True)
    for _ in range(25):
        clicks.add()
    t = time.perf_counter()
    sol, _d = solve_markers(stage, clicks.pts, clicks.ids,
                            smooth_sigma=SIGMA)
    t_first = (time.perf_counter() - t) * 1000.0
    times = {}
    for name, edit in (("remove", clicks.remove), ("add", clicks.add),
                       ("move", clicks.move), ("add again", clicks.add)):
        edit()
        t = time.perf_counter()
        new, diff = solve_markers(stage, clicks.pts, clicks.ids,
                                  smooth_sigma=SIGMA, previous=sol)
        times[name] = (time.perf_counter() - t) * 1000.0
        assert not diff.full
        print(f"[ADD-INC] {split} {name}: {times[name]:.0f} ms "
              f"({', '.join(f'{k} {v:.0f}' for k, v in diff.ms.items())})")
        sol = new
    print(f"[ADD-INC] {split} prepare {t_prep:.0f} ms, first solve "
          f"{t_first:.0f} ms ({len(clicks.pts)} clicks)")
    assert max(times.values()) < 300.0, times


# ------------------------------------------------- claims, many clicks
def claims_every_pair(stage, pts):
    """Each click's claim the way segment_at_markers makes it: every ball
    drawn over the whole box (an earlier ball keeps a shared voxel), then
    the piece the ball claims."""
    from scipy import ndimage as ndi

    balls = rasterise_points(pts, stage.shape, radius_um=stage.radius_um,
                             spacing=stage.spacing)
    slices = ndi.find_objects(balls, max_label=len(pts))
    out = []
    for k in range(1, len(pts) + 1):
        sl = slices[k - 1]
        if sl is None:
            out.append((0, "kept", None, None))
            continue
        ball = balls[sl] == k
        own = tuple(int(round(float(v))) for v in pts[k - 1][-3:])
        if not (all(0 <= own[a] < stage.shape[a] for a in range(3))
                and balls[own] == k):
            own = None
        c, reason = _claim_piece(ball, sl, own, stage.empty, stage.fg,
                                 stage.comp)
        if reason is not None:
            out.append((0, reason, None, None))
            continue
        vox = np.argwhere(ball & (stage.comp[sl] == c))
        vox = vox + np.asarray([v.start for v in sl])
        if own is not None and not stage.fg[own]:
            own = None
        out.append((c, None, vox.tobytes(), own))
    return out


def _as_reference(claims):
    return [(c, why, vox, own) for c, why, _b, _m, own, vox in claims]


@pytest.mark.parametrize("n", [50, 200, 800])
def test_the_claims_equal_every_pair_s_through_edits(n):
    """Crowded clicks (balls overlapping, some on the same voxel), then
    edits that add, remove, move and reorder: the claims kept from the
    last solve equal the every-pair claims each time."""
    spacing = (1.0, 0.5, 0.5)
    img, ex = nuclei((20, 60, 60), spacing, 3)
    stage = prepare_markers_box(img, ex, radius_um=1.5, spacing=spacing)
    rng = np.random.default_rng(n)
    pts = [tuple(int(rng.integers(-1, m + 1)) for m in stage.shape)
           for _ in range(n)]
    pts += pts[:5]                                    # the same voxels again
    got, cache = _claims(stage, pts)
    assert _as_reference(got) == claims_every_pair(stage, pts)
    for step in range(6):
        pts = list(pts)
        if step % 3 == 0:
            pts.append(tuple(int(rng.integers(0, m)) for m in stage.shape))
        elif step % 3 == 1:
            del pts[int(rng.integers(len(pts)))]
        else:
            i, j = (int(v) for v in rng.integers(len(pts), size=2))
            pts[i] = tuple(int(rng.integers(0, m)) for m in stage.shape)
            pts[i], pts[j] = pts[j], pts[i]
        got, cache = _claims(stage, pts, cache)
        assert _as_reference(got) == claims_every_pair(stage, pts)


def _lattice_stage():
    """91 x 240 x 240: a lattice of 2,000-odd small bright blocks, one
    piece each."""
    shape = (91, 240, 240)
    img = np.zeros(shape, np.float32)
    cents = []
    for z in range(3, 88, 8):
        for y in range(4, 236, 12):
            for x in range(4, 236, 12):
                img[z:z + 4, y:y + 8, x:x + 8] = 1.0
                cents.append((z + 2, y + 4, x + 4))
    stage = prepare_markers_box(img, np.zeros(shape, np.int32),
                                radius_um=1.0, spacing=(1.0, 0.4, 0.4),
                                threshold=0.5)
    return stage, cents


def test_one_more_click_among_800_is_fast():
    """The claims are not redone for every click on every solve: one more
    click among 800 on a whole patch (smoothing off) stays under 300 ms."""
    stage, cents = _lattice_stage()
    np.random.default_rng(0).shuffle(cents)
    times = {}
    for n in (100, 800):
        pts = [tuple(int(v) for v in c) for c in cents[:n]]
        sol, _ = solve_markers(stage, pts)
        t = time.perf_counter()
        new, diff = solve_markers(stage, pts + [cents[n]], previous=sol)
        times[n] = (time.perf_counter() - t) * 1000.0
        assert not diff.full and diff.n_pieces_solved == 1
        assert new.sizes[n + 1] > 0
        print(f"[ADD-INC] one more click among {n}: {times[n]:.0f} ms "
              f"({', '.join(f'{k} {v:.0f}' for k, v in diff.ms.items())})")
    assert max(times.values()) < 300.0, times


def test_a_click_on_the_largest_piece_of_a_dense_stain_measured():
    """Measured, not budgeted: on a dense stain (touching nuclei joined
    into pieces of about 100,000 voxels) the flood of the one piece a click
    lands on is the cost; the smoothing reuses every untouched shape.  The
    times are printed; the pin is a ceiling well above them."""
    spacing = (1.0, 0.4, 0.4)
    shape = (91, 240, 240)
    img, ex = nuclei(shape, spacing, 0, r_um=(2.5, 4.0), density=0.35,
                     kept=False)
    _fg, comp = pieces_of(img, ex, False)
    sizes = np.bincount(comp.ravel())
    sizes[0] = 0
    stage = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing)
    rng = np.random.default_rng(9)
    clicks = Clicks(rng, comp, spacing, one_per_piece=True)
    for _ in range(25):
        clicks.add()
    sol, _ = solve_markers(stage, clicks.pts, clicks.ids,
                           smooth_sigma=SIGMA)
    big = int(np.argmax(sizes))
    vox = np.argwhere(comp == big)
    pts, ids = list(clicks.pts), list(clicks.ids)
    times = []
    for frac, ident in ((2, 9001), (3, 9002)):
        pts.append(tuple(int(v) for v in vox[len(vox) // frac]))
        ids.append(ident)
        t = time.perf_counter()
        sol, diff = solve_markers(stage, pts, ids, smooth_sigma=SIGMA,
                                  previous=sol)
        times.append((time.perf_counter() - t) * 1000.0)
        assert not diff.full
        print(f"[ADD-INC] click on the largest piece ({int(sizes[big])} "
              f"voxels): {times[-1]:.0f} ms "
              f"({', '.join(f'{k} {v:.0f}' for k, v in diff.ms.items())})")
    assert max(times) < 3000.0, times


# ------------------------------------------------------ smoothing failure
def test_a_failed_smoothing_leaves_everything_unsmoothed_then_recovers(
        monkeypatch):
    """As the one-go smoothing: a shape that fails leaves the whole answer
    unsmoothed; the solve says so, and the next solve smooths everything
    again."""
    from packages.epicycle_annotator import batch_refine

    spacing = (1.0, 1.0, 1.0)
    img, ex = nuclei((26, 60, 60), spacing, 11)
    _fg, comp = pieces_of(img, ex, False)
    stage = prepare_markers_box(img, ex, radius_um=RADIUS, spacing=spacing)
    rng = np.random.default_rng(4)
    clicks = Clicks(rng, comp, spacing, one_per_piece=True)
    for _ in range(5):
        clicks.add()
    real = batch_refine._compute_one

    def broken(snapshot, label_id, *a, **kw):
        raise RuntimeError("no smoothing today")

    # the full solve, as smooth_pieces
    monkeypatch.setattr(batch_refine, "_compute_one", broken)
    sol, diff = solve_markers(stage, clicks.pts, clicks.ids,
                              smooth_sigma=SIGMA)
    assert sol.smooth_failed and diff.full
    np.testing.assert_array_equal(sol.result, sol.raw)
    np.testing.assert_array_equal(
        sol.result, smooth_pieces(np.array(sol.raw), SIGMA))
    check_counts(sol)
    # it recovers: the next solve smooths everything
    monkeypatch.setattr(batch_refine, "_compute_one", real)
    clicks.add()
    good, diff = solve_markers(stage, clicks.pts, clicks.ids,
                               smooth_sigma=SIGMA, previous=sol)
    assert diff.full and not good.smooth_failed
    np.testing.assert_array_equal(good.result,
                                  smooth_pieces(np.array(good.raw), SIGMA))
    # an incremental solve that fails falls back the same way
    monkeypatch.setattr(batch_refine, "_compute_one", broken)
    clicks.add()
    bad, diff = solve_markers(stage, clicks.pts, clicks.ids,
                              smooth_sigma=SIGMA, previous=good)
    assert bad.smooth_failed and diff.full
    np.testing.assert_array_equal(bad.result, bad.raw)
    check_counts(bad)
    monkeypatch.setattr(batch_refine, "_compute_one", real)
    clicks.remove()
    again, diff = solve_markers(stage, clicks.pts, clicks.ids,
                                smooth_sigma=SIGMA, previous=bad)
    assert diff.full and not again.smooth_failed
    np.testing.assert_array_equal(again.result,
                                  smooth_pieces(np.array(again.raw), SIGMA))


# ------------------------------------------------ grouped click edits
class GroupClicks:
    """Clicks in groups (each group one id), with edits that add a new
    group, add a seed to a group, remove a seed and move a seed.  With
    ``exact`` every group holds its own piece and an occasional seed is
    put on a piece no group uses (refused as apart); otherwise several
    groups may share a piece and no seed is apart."""

    def __init__(self, rng, comp, spacing, *, exact):
        self.rng = rng
        self.comp = comp
        self.half = [int(np.ceil(RADIUS / s)) for s in spacing]
        self.exact = exact
        self.pts, self.keys = [], []
        self.ids = {}
        self.home = {}           # group key -> its piece
        self.next = 0
        sizes = np.bincount(comp.ravel())
        self.big = [c for c in range(1, len(sizes)) if sizes[c] >= 60]
        self.used = set()        # pieces ever given to a group or a stray

    def _owner(self, p):
        return int(self.comp[tuple(p)])

    def _far(self, p, c, skip=None):
        for i, q in enumerate(self.pts):
            if i == skip or self._owner(q) == c:
                continue
            if all(abs(p[a] - q[a]) <= 2 * self.half[a] for a in range(3)):
                return False
        return True

    def _point_in(self, c, skip=None):
        vox = np.argwhere(self.comp == c)
        for _try in range(40):
            p = tuple(int(v) for v in vox[self.rng.integers(len(vox))])
            if self._far(p, c, skip):
                return p
        return None

    def _fresh_piece(self):
        pool = [c for c in self.big if c not in self.used]
        return int(self.rng.choice(pool)) if pool else None

    def new_group(self):
        if self.exact:
            c = self._fresh_piece()
        else:
            n = {}
            for g, c in self.home.items():
                if g in self.keys:
                    n[c] = n.get(c, 0) + 1
            pool = [c for c in self.big if n.get(c, 0) < 2]
            again = [c for c in pool if c in n]
            if again and self.rng.random() < 0.6:
                pool = again
            c = int(self.rng.choice(pool)) if pool else None
        if c is None:
            return
        p = self._point_in(c)
        if p is None:
            return
        g = f"g{self.next}"
        self.next += 1
        self.used.add(c)
        self.home[g] = c
        self.ids[g] = 30 + 7 * self.next
        self.pts.append(p)
        self.keys.append(g)

    def add_seed(self):
        live = sorted(set(self.keys))
        if not live:
            return self.new_group()
        g = live[int(self.rng.integers(len(live)))]
        if self.exact and self.rng.random() < 0.25:
            c = self._fresh_piece()          # apart: a piece nobody uses
            if c is None:
                return
            self.used.add(c)
        else:
            c = self.home[g]
        p = self._point_in(c)
        if p is not None:
            self.pts.append(p)
            self.keys.append(g)

    def remove(self):
        if self.pts:
            i = int(self.rng.integers(len(self.pts)))
            del self.pts[i]
            del self.keys[i]

    def move(self):
        if self.pts:
            i = int(self.rng.integers(len(self.pts)))
            p = self._point_in(self._owner(self.pts[i]), skip=i)
            if p is not None:
                self.pts[i] = p

    def edit(self):
        r = self.rng.random()
        if r < 0.3 or len(self.pts) < 2:
            self.new_group()
        elif r < 0.6:
            self.add_seed()
        elif r < 0.8:
            self.remove()
        else:
            self.move()

    def click_ids(self):
        return [self.ids[g] for g in self.keys]


def one_go_grouped(img, ex, pts, keys, ids, *, spacing, clahe, split,
                   min_size, sigma, threshold="otsu"):
    """segment_at_markers with the groups, numbered by the groups' ids,
    smoothed."""
    pieces, why, seed_why = segment_at_markers(
        img, ex, pts, radius_um=RADIUS, spacing=spacing, clahe=clahe,
        min_size=min_size, split=split, peak_spacing_um=PEAK_UM,
        groups=keys, threshold=threshold)
    gids = list(dict.fromkeys(ids))
    mapping = {g: gids[g - 1] for g in range(1, len(gids) + 1)
               if g not in why}
    raw = (_mint_to(pieces, mapping) if mapping
           else np.zeros(pieces.shape, np.int32))
    result = smooth_pieces(raw, sigma) if sigma is not None else raw
    return raw, result, why, seed_why


def per_piece_grouped(img, ex, pts, keys, ids, comp, *, spacing, clahe,
                      split, min_size, sigma, threshold="otsu"):
    """Each clicked piece solved on its own groups' clicks (every seed
    lies on its group's piece)."""
    raw = np.zeros(comp.shape, np.int32)
    first = {}
    gidx = [first.setdefault(g, len(first) + 1) for g in keys]
    why, seed_why = {}, {}
    owner = [int(comp[tuple(p)]) for p in pts]
    for c in sorted(set(owner)):
        idx = [i for i in range(len(pts)) if owner[i] == c]
        sub_keys = [keys[i] for i in idx]
        alone, w, sw = segment_at_markers(
            img, ex, [pts[i] for i in idx], radius_um=RADIUS,
            spacing=spacing, clahe=clahe, min_size=min_size, split=split,
            peak_spacing_um=PEAK_UM, groups=sub_keys, threshold=threshold)
        local = list(dict.fromkeys(sub_keys))
        lut = np.zeros(len(local) + 1, np.int32)
        for j, g in enumerate(local, start=1):
            lut[j] = ids[keys.index(g)]
            if j in w:
                why[first[g]] = w[j]
        for j, i in enumerate(idx, start=1):
            if j in sw:
                seed_why[i + 1] = sw[j]
        inside = comp == c
        raw[inside] = lut[alone[inside]]
    result = smooth_pieces(raw, sigma) if sigma is not None else raw
    return raw, result, dict(sorted(why.items())), dict(sorted(
        seed_why.items()))


def run_grouped(img, ex, spacing, *, clahe, split, sigma, exact, seed,
                n_start=6, n_edits=8, min_size=0):
    _fg, comp = pieces_of(img, ex, clahe)
    rng = np.random.default_rng(seed)
    clicks = GroupClicks(rng, comp, spacing, exact=exact)
    for _ in range(n_start):
        clicks.edit()
    stage = prepare_markers_box(
        img, ex, radius_um=RADIUS, spacing=spacing, clahe=clahe,
        min_size=min_size, split=split, peak_spacing_um=PEAK_UM)
    prev = None
    n_grouped = n_apart = 0
    for step in range(n_edits + 1):
        if step:
            clicks.edit()
        pts, keys, ids = list(clicks.pts), list(clicks.keys), \
            clicks.click_ids()
        sol, diff = solve_markers(stage, pts, ids, groups=keys,
                                  smooth_sigma=sigma, previous=prev)
        check_counts(sol)
        if prev is not None:
            check_diff(prev, sol, diff)
        if exact:
            raw, res, why, seed_why = one_go_grouped(
                img, ex, pts, keys, ids, spacing=spacing, clahe=clahe,
                split=split, min_size=min_size, sigma=sigma)
        else:
            raw, res, why, seed_why = per_piece_grouped(
                img, ex, pts, keys, ids, comp, spacing=spacing, clahe=clahe,
                split=split, min_size=min_size, sigma=sigma)
        np.testing.assert_array_equal(sol.raw, raw)
        np.testing.assert_array_equal(sol.result, res)
        assert sol.why == why
        assert sol.seed_why == seed_why
        n_grouped += sum(1 for g in set(keys) if keys.count(g) > 1)
        n_apart += sum(1 for w in seed_why.values() if w == "apart")
        prev = sol
    return n_grouped, n_apart


@pytest.mark.parametrize("spacing,sigma,clahe", [
    ((1.0, 1.0, 1.0), SIGMA, False),
    ((2.0, 1.0, 1.0), None, True),
    ((1.0, 0.4, 0.4), SIGMA, True),
])
def test_off_one_group_per_piece_equals_the_one_go_grouped_answer(
        spacing, sigma, clahe):
    shape = (26, 60, 60) if spacing[1] == 1.0 else (26, 110, 110)
    img, ex = nuclei(shape, spacing, 13)
    n_grouped, n_apart = run_grouped(img, ex, spacing, clahe=clahe,
                                     split="off", sigma=sigma, exact=True,
                                     seed=3)
    assert n_grouped > 0 and n_apart > 0


@pytest.mark.parametrize("split,sigma,spacing", [
    ("off", SIGMA, (1.0, 1.0, 1.0)),
    ("single", None, (2.0, 1.0, 1.0)),
    ("always", SIGMA, (1.0, 1.0, 1.0)),
    ("always", None, (2.0, 1.0, 1.0)),
])
def test_grouped_pieces_solved_on_their_own_groups(split, sigma, spacing):
    img, ex = nuclei((26, 60, 60), spacing, 23)
    n_grouped, _n = run_grouped(img, ex, spacing, clahe=False, split=split,
                                sigma=sigma, exact=False, seed=7)
    assert n_grouped > 0


# ------------------------------------- lobes in a row: the lost-seed rule
#: lobes in a row (r = 10, a neck between each two): three rows in one box,
#: each row one 6-connected piece; the rows lie far apart
LOBE_X = (12, 30, 48, 66)
LOBE_ROWS = ((15, 3), (45, 4), (75, 4))          # (y, lobes in the row)
LOBE_SHAPE = (30, 90, 80)


def lobes():
    zz, yy, xx = np.indices(LOBE_SHAPE)
    fg = np.zeros(LOBE_SHAPE, bool)
    for y, n in LOBE_ROWS:
        for x in LOBE_X[:n]:
            fg |= (zz - 15) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= 100
    return fg.astype(np.float64), np.zeros(LOBE_SHAPE, np.int32)


def lobe(row, i):
    return (15, LOBE_ROWS[row][0], LOBE_X[i])


#: seeds of one group on the two ends of a row of three and of four
ENDS = [(lobe(0, 0), "a"), (lobe(0, 2), "a"),
        (lobe(1, 0), "b"), (lobe(1, 3), "b")]
#: another object on lobe 3, one group on lobes 1 and 2
BLOCKED = [(lobe(2, 2), "c"), (lobe(2, 0), "d"), (lobe(2, 1), "d")]
#: one group on lobes 1 and 4, another object on lobe 2 between them: the
#: seed on lobe 4 is cut off, refused as never placed
CUT_OFF = [(lobe(2, 0), "e"), (lobe(2, 3), "e"), (lobe(2, 1), "f")]


class LobeClicks:
    """Grouped clicks on the rows, each group on one row, with random
    edits: a seed added to a group, a new group, a click removed or moved
    within its row."""

    def __init__(self, rng, start):
        self.rng = rng
        self.pts = [p for p, _g in start]
        self.keys = [g for _p, g in start]
        rows = {y: r for r, (y, _n) in enumerate(LOBE_ROWS)}
        self.row = {g: rows[p[1]] for p, g in start}
        self.ids = {g: 20 + 9 * n
                    for n, g in enumerate(dict.fromkeys(self.keys))}
        self.next = 0

    def _point(self, row):
        for _try in range(20):
            x = LOBE_X[int(self.rng.integers(LOBE_ROWS[row][1]))]
            d = self.rng.integers(-5, 6, size=3)
            p = (15 + int(d[0]), LOBE_ROWS[row][0] + int(d[1]),
                 x + int(d[2]))
            if p not in self.pts:
                return p
        return None

    def add(self, p, g):
        self.pts.append(p)
        self.keys.append(g)

    def pop(self):
        del self.pts[-1]
        del self.keys[-1]

    def edit(self):
        r = self.rng.random()
        live = sorted(set(self.keys))
        if r < 0.35 and live:
            g = live[int(self.rng.integers(len(live)))]
            p = self._point(self.row[g])
            if p is not None:
                self.add(p, g)
        elif r < 0.55 or not self.pts:
            row = int(self.rng.integers(len(LOBE_ROWS)))
            p = self._point(row)
            if p is not None:
                g = f"n{self.next}"
                self.next += 1
                self.row[g] = row
                self.ids[g] = 100 + 3 * self.next
                self.add(p, g)
        elif r < 0.8:
            i = int(self.rng.integers(len(self.pts)))
            del self.pts[i]
            del self.keys[i]
        else:
            i = int(self.rng.integers(len(self.pts)))
            p = self._point(self.row[self.keys[i]])
            if p is not None:
                self.pts[i] = p

    def click_ids(self):
        return [self.ids[g] for g in self.keys]


def _lobe_step(stage, img, ex, comp, clicks, *, split, sigma, prev):
    """One solve against the reference: each piece on its own groups'
    clicks; with the split off and one group per piece, the one run over
    every click too.  The incremental answer equals a fresh one."""
    pts, keys, ids = list(clicks.pts), list(clicks.keys), clicks.click_ids()
    sol, diff = solve_markers(stage, pts, ids, groups=keys,
                              smooth_sigma=sigma, previous=prev)
    check_counts(sol)
    if prev is not None:
        assert not diff.full
        check_diff(prev, sol, diff)
    kw = dict(spacing=(1.0, 1.0, 1.0), clahe=False, split=split,
              min_size=0, sigma=sigma, threshold=0.5)
    raw, res, why, seed_why = per_piece_grouped(img, ex, pts, keys, ids,
                                                comp, **kw)
    np.testing.assert_array_equal(sol.raw, raw)
    np.testing.assert_array_equal(sol.result, res)
    assert sol.why == why
    assert sol.seed_why == seed_why
    owners = {}
    for p, g in zip(pts, keys):
        owners.setdefault(int(comp[p]), set()).add(g)
    if split == "off" and all(len(v) == 1 for v in owners.values()):
        raw1, res1, why1, sw1 = one_go_grouped(img, ex, pts, keys, ids, **kw)
        np.testing.assert_array_equal(sol.raw, raw1)
        np.testing.assert_array_equal(sol.result, res1)
        assert (sol.why, sol.seed_why) == (why1, sw1)
    fresh, fdiff = solve_markers(stage, pts, ids, groups=keys,
                                 smooth_sigma=sigma)
    assert fdiff.full
    np.testing.assert_array_equal(fresh.raw, sol.raw)
    np.testing.assert_array_equal(fresh.result, sol.result)
    assert (fresh.why, fresh.seed_why) == (sol.why, sol.seed_why)
    return sol


def _n_apart(sol):
    return sum(1 for w in sol.seed_why.values() if w == "apart")


@pytest.mark.parametrize("sigma", [None, SIGMA])
@pytest.mark.parametrize("split", ["off", "single", "always"])
def test_grouped_lobes_follow_the_lost_seed_rule_through_edits(split, sigma):
    """Seeds on the ends of a row of three and of four lobes, a group
    beside another object's lobe, a seed cut off by another object (refused
    as never placed): the engine equals segment_at_markers on each piece's
    clicks, through the refusal going and coming back and random edits."""
    from scipy import ndimage as ndi

    img, ex = lobes()
    comp, n = ndi.label(img > 0.5)
    assert n == len(LOBE_ROWS)
    stage = prepare_markers_box(img, ex, radius_um=RADIUS,
                                spacing=(1.0, 1.0, 1.0), threshold=0.5,
                                split=split, peak_spacing_um=PEAK_UM)
    n_apart = 0
    for n, start in enumerate((CUT_OFF, BLOCKED)):
        rng = np.random.default_rng(31 + n + 7 * (sigma is None))
        clicks = LobeClicks(rng, ENDS + start)
        sol = _lobe_step(stage, img, ex, comp, clicks, split=split,
                         sigma=sigma, prev=None)
        a, b = clicks.ids["a"], clicks.ids["b"]
        if split != "off":
            # nothing between a group's seeds is left out
            assert np.array_equal(sol.raw == a, comp == comp[lobe(0, 0)])
            assert np.array_equal(sol.raw == b, comp == comp[lobe(1, 0)])
        if start is CUT_OFF:
            cut = clicks.pts.index(lobe(2, 3)) + 1
            assert sol.seed_why == {cut: "apart"}
            n_apart += 1
            # the object in the way goes, then comes back, twice
            for _again in range(2):
                clicks.pop()
                sol = _lobe_step(stage, img, ex, comp, clicks, split=split,
                                 sigma=sigma, prev=sol)
                assert sol.seed_why == {}
                clicks.add(lobe(2, 1), "f")
                sol = _lobe_step(stage, img, ex, comp, clicks, split=split,
                                 sigma=sigma, prev=sol)
                assert sol.seed_why == {cut: "apart"}
        else:
            assert sol.seed_why == {}
            assert sol.raw[lobe(2, 2)] == clicks.ids["c"]
            d = clicks.ids["d"]
            assert sol.raw[lobe(2, 0)] == sol.raw[lobe(2, 1)] == d
        for _step in range(6):
            clicks.edit()
            sol = _lobe_step(stage, img, ex, comp, clicks, split=split,
                             sigma=sigma, prev=sol)
            n_apart += _n_apart(sol)
    assert n_apart > 0


@pytest.mark.parametrize("split", ["off", "always"])
def test_a_refused_lost_seed_costs_nothing_on_the_next_solve(monkeypatch,
                                                             split):
    """The claims and the floods met on the way to a refusal are kept: the
    same click list again, or one more click on another row, claims and
    floods only what is new."""
    from packages.epicycle_annotator import add_incremental as AI

    img, ex = lobes()
    stage = prepare_markers_box(img, ex, radius_um=RADIUS,
                                spacing=(1.0, 1.0, 1.0), threshold=0.5,
                                split=split, peak_spacing_um=PEAK_UM)
    pts = [p for p, _g in CUT_OFF]
    keys = [g for _p, g in CUT_OFF]
    sol, diff = solve_markers(stage, pts, [5, 5, 6], groups=keys,
                              smooth_sigma=SIGMA)
    assert sol.seed_why == {2: "apart"}
    assert diff.n_pieces_solved == 2         # with the seed, then without
    n_claims = []
    real = AI._claim_one

    def counting(*a, **kw):
        n_claims.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(AI, "_claim_one", counting)
    again, diff = solve_markers(stage, pts, [5, 5, 6], groups=keys,
                                smooth_sigma=SIGMA, previous=sol)
    assert (diff.n_pieces_solved, len(n_claims)) == (0, 0)
    assert diff.changed_ids == () and diff.patches == ()
    np.testing.assert_array_equal(again.result, sol.result)
    assert again.seed_why == sol.seed_why
    more, diff = solve_markers(stage, pts + [lobe(0, 1)], [5, 5, 6, 7],
                               groups=keys + ["g"], smooth_sigma=SIGMA,
                               previous=again)
    assert (diff.n_pieces_solved, len(n_claims)) == (1, 1)
    assert more.seed_why == {2: "apart"}
    fresh, _d = solve_markers(stage, pts + [lobe(0, 1)], [5, 5, 6, 7],
                              groups=keys + ["g"], smooth_sigma=SIGMA)
    np.testing.assert_array_equal(more.result, fresh.result)

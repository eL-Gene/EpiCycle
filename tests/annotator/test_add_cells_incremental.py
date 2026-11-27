"""Cells ▸ Click markers ▸ Fill to the walls, one click at a time.

``reshape/incremental.py`` floods only a window around each change of the
click list, with the voxels around the window frozen, and grows the window
until its edge proves nothing outside it would change.  The finish smooths
only the cells whose raw shape changed and re-writes their neighbours over
the smoothing's reach in ascending id order.  Every answer here is compared
with the full Preview (``run_add_preview`` -> ``reshape_empty_region`` and
the Preview's finish) for the same clicks, array for array.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import incremental as INC
from packages.epicycle_annotator.reshape import region as R
from packages.epicycle_annotator.reshape.outline import tissue_outline

SP = (1.0, 0.5, 0.5)


# --------------------------------------------------------------------------- #
# a synthetic patch: touching cells drawn as bright shells in a tissue slab
# --------------------------------------------------------------------------- #
def _patch(shape, *, n_cells, seed, open_caps=2, n_kept=2):
    """``(membrane, kept labels, cell map)``: Voronoi cells whose walls are
    bright, inside a slab of faint tissue over dark space; the slab's top
    and bottom are walls too, except over ``open_caps`` cells (an open
    cap); ``n_kept`` cells are kept objects (ids 500, 501, ...)."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    z0, z1 = int(nz * 0.2), int(nz * 0.8)
    pts = np.column_stack([rng.uniform(z0 + 2, z1 - 2, n_cells),
                           rng.uniform(2, ny - 2, n_cells),
                           rng.uniform(2, nx - 2, n_cells)])
    zz, yy, xx = np.indices(shape)
    d, lab = cKDTree(pts).query(
        np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()]), k=2)
    cell = lab[:, 0].reshape(shape) + 1
    gap = (d[:, 1] - d[:, 0]).reshape(shape)
    slab = (zz >= z0) & (zz < z1)
    wall = (gap < 1.2) & slab
    wall |= slab & ((zz == z0) | (zz == z1 - 1))
    for c in rng.choice(n_cells, size=open_caps, replace=False):
        wall &= ~((cell == c + 1) & (zz == z1 - 1))
    img = np.where(slab, 0.25, 0.02).astype(np.float32)
    img[wall] = 1.0
    img = ndi.gaussian_filter(img, 0.7) + rng.normal(
        0, 0.03, shape).astype(np.float32)
    img = np.clip(img, 0, None).astype(np.float32)
    kept = np.zeros(shape, np.int32)
    for j, c in enumerate(rng.choice(n_cells, size=n_kept, replace=False)):
        kept[(cell == c + 1) & slab & ~wall] = 500 + j
    cells = np.where(slab & ~wall, cell, 0)
    return img, kept, cells


def _brick_patch(shape, *, seed, cell=(10, 14, 14)):
    """A cheap large patch for the timing pins: a jittered brick wall of
    cells (walls every ``cell`` voxels) in a tissue slab."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.25, 0.03, shape).astype(np.float32)
    nz = shape[0]
    z0, z1 = int(nz * 0.1), int(nz * 0.9)
    img[:z0] = 0.02
    img[z1:] = 0.02
    for axis, step in enumerate(cell):
        n = shape[axis]
        for k in range(step // 2, n, step):
            idx = [slice(None)] * 3
            idx[axis] = slice(k, k + 1)
            img[tuple(idx)] = 1.0
    img[:z0] = 0.02
    img[z1:] = 0.02
    img = np.clip(ndi.gaussian_filter(img, 0.7), 0, None).astype(np.float32)
    return img


def _cells_params(*, outline="none", dark=True, smooth=True, min_size=0,
                  wall_kind="membrane"):
    return {"source": AP.SOURCE_CLICKS, "dark_outside": dark,
            "outline": outline, "auto_smooth": smooth, "smooth_sigma": 1.5,
            "click_radius_um": 1.0, "box_budget": 10 ** 9,
            "min_size": min_size, "clear_border": True, "edge_margin": 3,
            "wall_kind": wall_kind}


class _Case:
    """One patch, one box inside it, one set of settings: the full Preview
    and the prepared click fill side by side.  ``whole`` -- the whole patch
    filled the way a box is (its faces open)."""

    def __init__(self, *, seed, shape=(30, 52, 52), pad=2, n_cells=12,
                 n_kept=2, whole=False, **settings):
        img, kept, cells = _patch(shape, n_cells=n_cells, seed=seed,
                                  n_kept=n_kept)
        if whole:
            pad = 0
        self.pad = pad
        self.roi = tuple(slice(pad, n - pad) for n in shape)
        self.params = _cells_params(**settings)
        if whole:
            self.params.update(whole_patch=True, box_rule=True)
        self.existing = kept[self.roi].copy()
        self.cells = cells[self.roi]
        self.wall = img[self.roi].copy()
        self.darks = {"mem": self.wall}
        chans = {"mem": img}
        self.dt = R.patch_dark_threshold(chans, spacing=SP, stride=1)
        self.outline = None
        if self.params["outline"] == "shrink":
            self.outline = tissue_outline(chans, SP, dark_factor=0.5,
                                          bridge_um=10.0,
                                          dark_threshold=self.dt)[self.roi]
        # the page keeps the box's wall picture from its first Preview
        first = self.full([self.any_point(np.random.default_rng(0))], [1],
                          elevation=None)
        self.elevation = first.elevation
        spec = AP.cells_spec(self.params)
        self.stage = None
        self.flat = None
        if INC.is_click_fill(spec):
            self.stage, refusal = INC.prepare(
                self.existing, {"membrane": self.wall}, spec, roi=self.roi,
                dark_channels=(self.darks
                               if spec.params["outside"] == "faces+dark"
                               else None),
                dark_threshold=self.dt, spacing=SP, elevation=self.elevation,
                outline=self.outline)
            if refusal is not None:
                assert refusal[0] == INC.FULL_RUN, refusal
                self.flat = refusal[1]
        self.holder = {}
        self.key = ("box", seed, tuple(sorted(self.params.items())))
        INC.warm_kernel()

    def any_point(self, rng):
        free = np.argwhere((self.cells > 0) & (self.existing == 0))
        return tuple(int(v) for v in free[rng.integers(len(free))])

    def full(self, points, ids, *, elevation="kept", **kw):
        elev = self.elevation if elevation == "kept" else elevation
        return AP.run_add_preview(
            preset="cells", roi=self.roi, existing_roi=self.existing,
            next_label=1, spacing=SP, params=self.params, wall_roi=self.wall,
            dark_rois=self.darks,
            click_points=[tuple(c + self.pad for c in p) for p in points],
            reserved_ids=list(ids), dark_threshold=self.dt,
            outline_roi=self.outline, elevation_roi=elev, **kw)

    def by_click(self, points, ids, **kw):
        """The page's path: the same call with the stage kept in a holder."""
        kw.setdefault("stage_holder", self.holder)
        kw.setdefault("stage_key", self.key)
        return self.full(points, ids, **kw)

    def step(self, points, ids):
        if self.stage is None:
            return None, None
        st = self.stage.step(points, ids)
        p = self.params
        fin = self.stage.finish(min_size=p["min_size"],
                                auto_smooth=p["auto_smooth"],
                                smooth_sigma=p["smooth_sigma"],
                                clear_border=p["clear_border"],
                                edge_margin=p["edge_margin"],
                                whole_patch=bool(p.get("whole_patch", False)))
        return st, fin


def _same(full, st, fin):
    """The incremental answer equals the full Preview's."""
    if full.status != "ok":
        # the full run refused (no marker left, or every cell filtered)
        return st.status != "ok" or not fin.ids
    return (st.status == "ok"
            and fin.result.dtype == full.result.dtype
            and np.array_equal(fin.result, full.result)
            and fin.ids == sorted(full.ids)
            and fin.sizes == full.sizes
            and fin.edge_ids == set(full.edge_ids)
            and fin.small_ids == set(full.small_ids)
            and fin.n_emptied_by_smooth == full.n_emptied_by_smooth
            and fin.n_below_min_size == full.n_below_min_size)


#: the bridge's answer fields compared with the full Preview's (the faint
#: pairs and the unfilled share are not measured click by click)
_FIELDS = ("status", "refusal", "refusal_reason", "ids", "sizes", "edge_ids",
           "small_ids", "n_emptied_by_smooth", "n_below_min_size",
           "absorbed_ids", "n_fragments_dropped", "n_dropped_dark",
           "n_dropped_thin", "n_nuclei_slivers", "n_markers_over_kept",
           "n_outside_outline", "n_slivers", "n_tiles", "whole_patch",
           "source", "messages")


def _same_result(full, inc):
    """The bridge's ``AddPreviewResult`` against the full Preview's, field
    for field; returns the names that differ (empty: the same)."""
    bad = [k for k in _FIELDS if getattr(full, k) != getattr(inc, k)]
    if (full.result is None) != (inc.result is None):
        bad.append("result")
    elif full.result is not None and (
            full.result.dtype != inc.result.dtype
            or not np.array_equal(full.result, inc.result)):
        bad.append("result")
    if (full.elevation is None) != (inc.elevation is None) or (
            full.elevation is not None
            and not np.array_equal(full.elevation, inc.elevation)):
        bad.append("elevation")
    if inc.status == "ok" and inc.by_click:
        if inc.faint_pairs or inc.unfilled_tissue_fraction is not None:
            bad.append("notes")
        if "faint=n/a (by click)" not in inc.log_line():
            bad.append("log_line")
    return bad


def _random_session(case, rng, n_ops):
    """Random clicks: add (mostly), remove one, re-add a point with a new
    id; after each change the incremental answer is compared with a full
    Preview.  Returns the number of compared steps."""
    points, ids, nid, compared = [], [], 1, 0
    for _ in range(n_ops):
        r = rng.random()
        if points and r < 0.3:
            k = int(rng.integers(len(points)))
            points.pop(k)
            ids.pop(k)
        elif points and r < 0.4:
            points.append(points[int(rng.integers(len(points)))])
            ids.append(nid)
            nid += 1
        else:
            points.append(case.any_point(rng))
            ids.append(nid)
            nid += 1
        st, fin = case.step(points, ids)
        full = case.full(points, ids)
        inc = case.by_click(points, ids)
        assert not _same_result(full, inc), _same_result(full, inc)
        if case.holder.get("stage") is not None:
            assert inc.result is None or not np.shares_memory(
                inc.result, case.holder["stage"]._final)
        compared += 1
        if st is None:
            continue
        assert _same(full, st, fin), (
            "differs after {} clicks: {} voxels".format(
                len(points),
                -1 if full.result is None or fin.result is None
                else int(np.count_nonzero(full.result != fin.result))))
    return compared


# --------------------------------------------------------------------------- #
# the flood is scikit-image's watershed
# --------------------------------------------------------------------------- #
def test_the_flood_equals_scikit_image_watershed_on_ties():
    from skimage.segmentation import watershed

    rng = np.random.default_rng(1)
    for t in range(120):
        shape = tuple(int(v) for v in rng.integers(3, 13, 3))
        levels = int(rng.integers(1, 5))
        elev = (rng.integers(0, levels, shape).astype(np.float32)
                / max(levels - 1, 1))
        if t % 3 == 0:
            elev = ndi.gaussian_filter(rng.random(shape).astype(np.float32), 1)
        markers = np.zeros(shape, np.int32)
        k = int(rng.integers(1, 7))
        idx = rng.integers(0, markers.size, 3 * k)
        markers.ravel()[idx] = rng.integers(1, k + 1, idx.size)
        markers.ravel()[idx[:2]] = -1
        lab, lev = INC.flood(elev, markers)
        assert np.array_equal(lab, watershed(elev, markers)), t
        # a marker sits at its own height; every other voxel at least there
        assert np.all(lev[markers != 0] == elev[markers != 0])
        assert np.all(lev >= elev)


def test_the_flood_levels_are_the_lowest_climb():
    from skimage.morphology import reconstruction

    rng = np.random.default_rng(2)
    elev = rng.random((14, 17, 19)).astype(np.float32)
    markers = np.zeros(elev.shape, np.int32)
    markers.ravel()[rng.integers(0, elev.size, 12)] = 1
    _lab, lev = INC.flood(elev, markers)
    seed = np.where(markers != 0, elev, elev.max()).astype(np.float64)
    low = reconstruction(seed, elev.astype(np.float64), method="erosion",
                         footprint=ndi.generate_binary_structure(3, 1))
    assert np.array_equal(lev.astype(np.float64), low)


# --------------------------------------------------------------------------- #
# parity with the full Preview
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("outline", ["none", "shrink"])
@pytest.mark.parametrize("dark", [True, False])
@pytest.mark.parametrize("smooth", [True, False])
def test_click_by_click_equals_the_full_preview(outline, dark, smooth):
    n = 0
    for seed in (0, 3):
        case = _Case(seed=seed, outline=outline, dark=dark, smooth=smooth)
        n += _random_session(case, np.random.default_rng(seed + 100), 9)
    assert n == 18


def test_parity_with_no_kept_neighbour_and_a_size_floor():
    case = _Case(seed=5, n_kept=0, min_size=150)
    _random_session(case, np.random.default_rng(7), 10)


def test_parity_with_kept_neighbours_everywhere():
    case = _Case(seed=4, n_kept=6, n_cells=14)
    _random_session(case, np.random.default_rng(8), 10)


def test_removing_every_click_is_the_no_marker_refusal_then_recovers():
    case = _Case(seed=2)
    rng = np.random.default_rng(3)
    a, b = case.any_point(rng), case.any_point(rng)
    st, fin = case.step([a, b], [1, 2])
    assert _same(case.full([a, b], [1, 2]), st, fin)
    st, fin = case.step([], [])
    assert st.status == "refused" and st.refusal_reason == "no-markers"
    assert fin.ids == [] and not fin.result.any()
    st, fin = case.step([b], [7])
    assert _same(case.full([b], [7]), st, fin)


def test_a_re_base_of_every_id_keeps_parity():
    case = _Case(seed=6)
    rng = np.random.default_rng(4)
    pts = [case.any_point(rng) for _ in range(4)]
    st, fin = case.step(pts, [10, 11, 12, 13])
    assert _same(case.full(pts, [10, 11, 12, 13]), st, fin)
    st, fin = case.step(pts, [20, 21, 22, 23])
    assert _same(case.full(pts, [20, 21, 22, 23]), st, fin)
    # ids out of placement order: the smoothing writes the lower id first
    st, fin = case.step(pts, [33, 31, 34, 30])
    assert _same(case.full(pts, [33, 31, 34, 30]), st, fin)


def test_the_raw_cells_equal_the_engine_with_the_reserved_ids():
    case = _Case(seed=1, smooth=False)
    rng = np.random.default_rng(9)
    pts = [case.any_point(rng) for _ in range(5)]
    ids = [4, 9, 2, 7, 5]
    st, _fin = case.step(pts, ids)
    res = R.reshape_empty_region(
        case.existing, {"membrane": case.wall}, AP.cells_spec(case.params),
        roi=case.roi, dark_channels=case.darks, seed_points=pts,
        dark_threshold=case.dt, spacing=SP, elevation=case.elevation,
        outline=case.outline)
    mapping, _absorbed = AP._reserved_mapping(pts, ids, case.existing,
                                              radius_um=1.0, spacing=SP)
    assert np.array_equal(case.stage.raw_result(),
                          AP._mint_to(res.result, mapping))
    assert sorted(st.changed_ids) == sorted(
        int(v) for v in np.unique(case.stage.raw_result()) if v)


def test_one_more_click_floods_a_window_not_the_box():
    case = _Case(seed=0, shape=(30, 80, 80), n_cells=24)
    rng = np.random.default_rng(11)
    pts, ids = [], []
    windows = []
    for k in range(6):
        pts.append(case.any_point(rng))
        ids.append(k + 1)
        st, fin = case.step(pts, ids)
        windows.append(st.window_voxels / case.existing.size)
        assert _same(case.full(pts, ids), st, fin)
    print("window share per click:", [round(w, 3) for w in windows])
    assert min(windows) < 0.5


def test_the_diff_box_holds_every_changed_voxel():
    case = _Case(seed=2)
    rng = np.random.default_rng(12)
    pts, ids = [], []
    before = None
    for k in range(5):
        pts.append(case.any_point(rng))
        ids.append(k + 1)
        _st, fin = case.step(pts, ids)
        now = fin.result.copy()
        if before is not None:
            changed = now != before
            if fin.diff_box is None:
                assert not changed.any()
            else:
                outside = changed.copy()
                outside[fin.diff_box] = False
                assert not outside.any()
                assert set(fin.changed_ids) == (
                    set(int(v) for v in np.unique(before[changed]) if v)
                    | set(int(v) for v in np.unique(now[changed]) if v))
        before = now


# --------------------------------------------------------------------------- #
# what prepare refuses
# --------------------------------------------------------------------------- #
def test_prepare_takes_the_click_fill_only():
    case = _Case(seed=0)
    for kw in ({"growth": "compete"},):
        p = dict(case.params, **kw)
        with pytest.raises(ValueError):
            INC.prepare(case.existing, {"membrane": case.wall},
                        AP.cells_spec(p), roi=case.roi,
                        dark_channels=case.darks, dark_threshold=case.dt,
                        spacing=SP, elevation=case.elevation)
    assert INC.is_click_fill(AP.cells_spec(case.params))
    assert not INC.is_click_fill(AP.cells_spec(dict(case.params,
                                                    growth="compete")))
    assert not INC.is_click_fill(AP.cells_spec(dict(case.params,
                                                    whole_patch=True)))
    p = dict(case.params, source=AP.SOURCE_NUCLEI)
    assert not INC.is_click_fill(AP.cells_spec(p))
    with pytest.raises(ValueError):
        INC.prepare(case.existing, {"membrane": case.wall}, AP.cells_spec(p),
                    roi=case.roi, dark_channels=case.darks,
                    dark_threshold=case.dt, spacing=SP,
                    elevation=case.elevation)


def test_prepare_gives_the_full_runs_refusals():
    case = _Case(seed=0)
    p = dict(case.params, box_budget=1000)
    stage, refusal = INC.prepare(
        case.existing, {"membrane": case.wall}, AP.cells_spec(p),
        roi=case.roi, dark_channels=case.darks, dark_threshold=case.dt,
        spacing=SP, elevation=case.elevation)
    assert stage is None and refusal[0] == "too-big"
    p = dict(case.params, outline="shrink")
    stage, refusal = INC.prepare(
        case.existing, {"membrane": case.wall}, AP.cells_spec(p),
        roi=case.roi, dark_channels=case.darks, dark_threshold=case.dt,
        spacing=SP, elevation=case.elevation, outline=None)
    assert stage is None and refusal[0] == "outline-per-box"


def test_the_spec_helper_is_the_preview_spec():
    """``cells_spec`` builds exactly the spec the Cells Preview always
    built (the extraction moved code, it did not change it)."""
    from packages.epicycle_annotator.reshape import ReshapeSpec
    from packages.epicycle_annotator.reshape.spec import DEFAULT_PARAMS

    def old(params):
        source = params.get("source", AP.SOURCE_CLICKS)
        outline = params.get("outline", "none") or "none"
        outside = ("faces+dark" if params.get("dark_outside", True)
                   or outline != "none" else "faces")
        budget = int(params.get("box_budget", AP.ADD_BOX_BUDGET))
        return ReshapeSpec(
            target="region", seeds=source,
            walls=params.get("wall_kind", "membrane"), solver="watershed",
            params={
                "wall_channel": "membrane",
                "wall_preproc": params.get("wall_chain", "zcorrect") or "none",
                "ridge": False, "outside": outside,
                "dark_otsu_factor": float(params.get("dark_otsu_factor", 0.5)),
                "dark_clearance_um": float(params.get("dark_clearance_um", 2.0)),
                "core_erode_um": float(params.get("core_erode_um", 1.0)),
                "click_radius_um": float(params.get("click_radius_um", 1.0)),
                "box_budget": budget, "growth": AP.cells_growth(params),
                "wall_cost": float(params.get("wall_cost",
                                              DEFAULT_PARAMS["wall_cost"])),
                "growth_limit": float(params.get(
                    "growth_limit", DEFAULT_PARAMS["growth_limit"])),
                "include_membrane": bool(params.get(
                    "include_membrane", DEFAULT_PARAMS["include_membrane"])),
                "outline": outline,
                "bridge_um": float(params.get("bridge_um",
                                              DEFAULT_PARAMS["bridge_um"])),
                "whole_patch": bool(params.get("whole_patch", False)),
            })

    for params in ({}, {"outline": "shrink", "dark_outside": False},
                   {"outline": None, "growth": "compete", "box_budget": 5},
                   {"whole_patch": True, "wall_chain": "", "source": "labels"},
                   {"click_radius_um": 3, "dark_clearance_um": 0}):
        assert AP.cells_spec(params) == old(params)


# --------------------------------------------------------------------------- #
# the timing pins (measured and printed; 300 ms per click is the target)
# --------------------------------------------------------------------------- #
def _no_guard(elevation, existing, cores, **_kw):
    """``flat_survey`` with nothing found: the window engine itself, for the
    timing and the budget pins on closed bricks (whose identical walls the
    real guard sends to the full run)."""
    return None, np.zeros(np.shape(elevation), np.int32), frozenset(), 0


def _timed_clicks(img, existing, *, n_clicks, seed, guard=True):
    shape = img.shape
    roi = tuple(slice(0, n) for n in shape)
    params = _cells_params(dark=True)
    chans = {"mem": img}
    stride = max(1, int(round((img.size / R.DARK_SAMPLE_VOXELS) ** (1 / 3))))
    dt = R.patch_dark_threshold(chans, spacing=SP, stride=stride)
    from packages.epicycle_annotator.reshape.walls import build_elevation

    elev, _c, _r = build_elevation("membrane", img, np.ones(shape, bool),
                                   spacing=SP, clip=0.01, smooth=1.0,
                                   ridge=False)
    t = time.perf_counter()
    real = INC.flat_survey
    if not guard:
        INC.flat_survey = _no_guard
    try:
        stage, refusal = INC.prepare(existing, {"membrane": img},
                                     AP.cells_spec(params), roi=roi,
                                     dark_channels=chans, dark_threshold=dt,
                                     spacing=SP, elevation=elev)
    finally:
        INC.flat_survey = real
    t_prepare = time.perf_counter() - t
    if refusal is not None:
        assert refusal[0] == INC.FULL_RUN
        print("\n[ADD-INC-PERF] guard: {}".format(refusal[1]))
        return t_prepare, None
    rng = np.random.default_rng(seed)
    nz = shape[0]
    pts, ids, per_click = [], [], []
    for k in range(n_clicks):
        pts.append((int(rng.integers(int(nz * 0.3), int(nz * 0.7))),
                    int(rng.integers(5, shape[1] - 5)),
                    int(rng.integers(5, shape[2] - 5))))
        ids.append(k + 1)
        t = time.perf_counter()
        st = stage.step(pts, ids)
        fin = stage.finish(min_size=0, auto_smooth=True, smooth_sigma=1.5)
        per_click.append((time.perf_counter() - t, st.n_floods,
                          st.window_voxels, st.seconds, fin.seconds,
                          len(st.changed_ids), len(fin.changed_ids),
                          st.voxels_flooded))
    return t_prepare, per_click


@pytest.mark.parametrize("kind", ["shells", "bricks"])
@pytest.mark.parametrize("shape,label", [((65, 120, 70), "M box"),
                                         ((147, 240, 240), "whole patch")])
def test_timing_pin_one_click(shape, label, kind):
    if kind == "shells":
        # Voronoi cells of about 4000 voxels with bright walls
        n_cells = max(8, int(0.6 * np.prod(shape) / 4000))
        img, existing, _cells = _patch(shape, n_cells=n_cells, seed=1,
                                       open_caps=n_cells // 10, n_kept=0)
    else:
        # closed bricks of identical walls: the most ties and the longest
        # chains of cells one click can take over
        img = _brick_patch(shape, seed=1)
        existing = np.zeros(shape, np.int32)
    label = "{} ({})".format(label, kind)
    INC.flood(np.zeros((2, 2, 2), np.float32),
              np.ones((2, 2, 2), np.int32))       # compile outside the clock
    t_prep, per = _timed_clicks(img, existing, n_clicks=6, seed=2)
    if per is None:
        # the guard keeps the box on the full run: time the engine alone
        label += " [guard bypassed: the page runs this box whole]"
        t_prep, per = _timed_clicks(img, existing, n_clicks=6, seed=2,
                                    guard=False)
    ms = [round(1e3 * p[0], 1) for p in per]
    print("\n[ADD-INC-PERF] {} {}: prepare {:.0f} ms; per click {} ms "
          "(step / finish ms: {}); floods {}; window voxels {}; raw cells "
          "changed {}; drawn cells changed {}; voxels flooded {}".format(
              label, shape, 1e3 * t_prep, ms,
              [(round(1e3 * p[3]), round(1e3 * p[4])) for p in per],
              [p[1] for p in per], [p[2] for p in per],
              [p[5] for p in per], [p[6] for p in per],
              [p[7] for p in per]))
    # a measured pin, not a gate: the clicks after the first must not be
    # slower than a full flood of the box would be
    full = time.perf_counter()
    INC.flood(img, np.pad(np.zeros(np.subtract(shape, 2), np.int32), 1,
                          constant_values=-1))
    full = time.perf_counter() - full
    print("[ADD-INC-PERF] {}: one full flood of the box {:.0f} ms".format(
        label, 1e3 * full))
    assert np.median([p[0] for p in per[1:]]) < max(full, 0.3) * 3


# --------------------------------------------------------------------------- #
# the bridge inside the Preview: reuse, fall back, never a half-done stage
# --------------------------------------------------------------------------- #
def _count_prepares(monkeypatch):
    calls = []
    real = INC.prepare

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(INC, "prepare", spy)
    return calls


def test_the_stage_is_reused_only_for_the_same_key_and_voxels(monkeypatch):
    case = _Case(seed=1)
    calls = _count_prepares(monkeypatch)
    rng = np.random.default_rng(5)
    pts = [case.any_point(rng)]
    case.by_click(pts, [1])
    pts.append(case.any_point(rng))
    inc = case.by_click(pts, [1, 2])
    assert len(calls) == 1
    assert not _same_result(case.full(pts, [1, 2]), inc)
    # another key: a fresh stage
    inc = case.by_click(pts, [1, 2], stage_key=("another",))
    assert len(calls) == 2 and not _same_result(case.full(pts, [1, 2]), inc)
    # the same key but other kept voxels: a fresh stage too
    case.existing = case.existing.copy()
    free = np.argwhere((case.cells > 0) & (case.existing == 0))
    case.existing[tuple(free[-1])] = 900
    inc = case.by_click(pts, [1, 2], stage_key=("another",))
    assert len(calls) == 3 and not _same_result(case.full(pts, [1, 2]), inc)


def test_without_a_holder_or_switched_off_it_is_the_full_run(monkeypatch):
    case = _Case(seed=2)
    calls = _count_prepares(monkeypatch)
    pts = [case.any_point(np.random.default_rng(1))]
    case.full(pts, [1])                                 # no holder
    monkeypatch.setattr(AP, "_INCREMENTAL_OFF", True)
    case.by_click(pts, [1])
    # nothing prepared; the holder only remembers why, for the key (the
    # page then stops running a whole patch as you click)
    assert calls == [] and "stage" not in case.holder
    assert case.holder == {"key": case.key, "fallback":
                           "full run (switched off for this session)"}


def test_until_the_flood_is_compiled_it_is_the_full_run(monkeypatch):
    case = _Case(seed=2)
    calls = _count_prepares(monkeypatch)
    monkeypatch.setattr(INC, "_WARM", False)
    pts = [case.any_point(np.random.default_rng(1))]
    inc = case.by_click(pts, [1])
    assert calls == [] and inc.status == "ok"
    assert "faint=n/a" not in inc.log_line()


def test_a_missing_numba_falls_back_to_the_full_run(monkeypatch, capsys):
    import sys

    case = _Case(seed=2)
    rng = np.random.default_rng(2)
    pts = [case.any_point(rng), case.any_point(rng)]
    full = case.full(pts, [1, 2])
    assert full.status == "ok"
    monkeypatch.setattr(INC, "_KERNEL", {})
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setattr(AP, "_INCREMENTAL_OFF", False)
    capsys.readouterr()
    inc = case.by_click(pts, [1, 2])
    assert inc.status == "ok" and inc.refusal is None
    assert np.array_equal(inc.result, full.result)
    assert AP._INCREMENTAL_OFF is True
    assert case.holder.get("stage") is None
    assert capsys.readouterr().out.count("[ADD] click-by-click fill") == 1
    # from then on, no try at all
    calls = _count_prepares(monkeypatch)
    case.by_click(pts, [1, 2])
    assert calls == []


def test_a_cancelled_step_drops_the_stage_and_the_next_run_prepares_again(
        monkeypatch):
    import threading

    from packages.epicycle_annotator.reshape.engine import ReshapeCancelled

    case = _Case(seed=4)
    rng = np.random.default_rng(3)
    pts = [case.any_point(rng)]
    case.by_click(pts, [1])
    first = case.holder["stage"]
    real = INC.CellsClickFill.step
    stop = threading.Event()

    def cancelled(self, *a, **k):
        stop.set()                      # the Stop lands mid step
        raise ReshapeCancelled()

    monkeypatch.setattr(INC.CellsClickFill, "step", cancelled)
    pts.append(case.any_point(rng))
    out = case.by_click(pts, [1, 2], cancel_event=stop)
    assert out.status == "cancelled"
    assert case.holder.get("stage") is None
    monkeypatch.setattr(INC.CellsClickFill, "step", real)
    calls = _count_prepares(monkeypatch)
    inc = case.by_click(pts, [1, 2])
    assert len(calls) == 1 and case.holder["stage"] is not first
    assert not _same_result(case.full(pts, [1, 2]), inc)


def test_an_error_mid_finish_drops_the_stage(monkeypatch):
    case = _Case(seed=4)
    rng = np.random.default_rng(3)
    pts = [case.any_point(rng)]
    case.by_click(pts, [1])

    def boom(self, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(INC.CellsClickFill, "finish", boom)
    pts.append(case.any_point(rng))
    inc = case.by_click(pts, [1, 2])
    assert case.holder.get("stage") is None
    assert not AP._INCREMENTAL_OFF
    assert inc.status == "ok" and "faint=n/a" not in inc.log_line()
    assert np.array_equal(inc.result, case.full(pts, [1, 2]).result)


def test_the_refusals_match_the_full_run():
    case = _Case(seed=5)
    rng = np.random.default_rng(4)
    # the only click deep inside a kept object: no marker on empty space
    depth = ndi.distance_transform_edt(case.existing != 0, sampling=SP)
    pts = [tuple(int(v) for v in np.unravel_index(np.argmax(depth),
                                                   depth.shape))]
    full = case.full(pts, [1])
    assert full.refusal_reason == "no-markers"
    assert not _same_result(full, case.by_click(pts, [1]))
    # a size floor no cell passes: every cell filtered
    case.params = dict(case.params, min_size=10 ** 7)
    case.key = ("floor",)
    pts = [case.any_point(rng)]
    full = case.full(pts, [1])
    assert full.status == "refused"
    assert not _same_result(full, case.by_click(pts, [1]))
    # the box budget
    case.params = dict(case.params, min_size=0, box_budget=1000)
    case.key = ("budget",)
    full = case.full(pts, [1])
    assert full.refusal_reason == "too-big"
    assert not _same_result(full, case.by_click(pts, [1]))


def test_the_absorbed_ids_are_the_reserved_mappings():
    case = _Case(seed=6, n_kept=4)
    rng = np.random.default_rng(6)
    depth = ndi.distance_transform_edt(case.existing != 0, sampling=SP)
    deep = tuple(int(v) for v in np.unravel_index(np.argmax(depth),
                                                  depth.shape))
    pts = [case.any_point(rng), deep, case.any_point(rng)]
    pts.append(pts[0])                      # its ball is the first's
    ids = [3, 4, 5, 6]
    inc = case.by_click(pts, ids)
    _mapping, absorbed = AP._reserved_mapping(pts, ids, case.existing,
                                              radius_um=1.0, spacing=SP)
    assert inc.absorbed_ids == absorbed and 4 in absorbed and 6 in absorbed
    assert not _same_result(case.full(pts, ids), inc)


def test_the_flood_budget_on_closed_bricks():
    """Each round floods a window; once the rounds so far and the next
    window would pass half the box the whole box is flooded, so a click
    never floods more than half the box on top of one full flood."""
    shape = (65, 120, 70)
    img = _brick_patch(shape, seed=1)
    existing = np.zeros(shape, np.int32)
    INC.warm_kernel()
    _t, per = _timed_clicks(img, existing, n_clicks=6, seed=2)
    assert per is None             # identical walls: the guard's full run
    _t, per = _timed_clicks(img, existing, n_clicks=6, seed=2, guard=False)
    box = int(np.prod(shape))
    assert INC.WINDOW_WHOLE_SHARE == 0.25
    for p in per:
        assert p[7] <= INC.WINDOW_WHOLE_SHARE * box + box, (p[7], box)



# --------------------------------------------------------------------------- #
# the guard: wall pictures with flat areas keep the full run
# --------------------------------------------------------------------------- #
FLAT_WALLS = ("intensity", "frangi", "meijering")


def test_only_membrane_and_gradient_walls_are_redone_by_click():
    for wk in ("membrane", "gradient"):
        assert INC.is_click_fill(AP.cells_spec(_cells_params(wall_kind=wk)))
    for wk in FLAT_WALLS:
        spec = AP.cells_spec(_cells_params(wall_kind=wk))
        assert not INC.is_click_fill(spec)
        with pytest.raises(ValueError):
            INC.prepare(np.zeros((4, 4, 4), np.int32),
                        {"membrane": np.zeros((4, 4, 4), np.float32)}, spec,
                        roi=(slice(0, 4),) * 3, spacing=SP,
                        dark_threshold={"t": 0.5, "lo": {}, "hi": {}})


@pytest.mark.parametrize("wall_kind", FLAT_WALLS)
def test_flat_walls_fall_back_and_equal_the_full_run(wall_kind, monkeypatch):
    """The stress that found the ties (8 seeds, outline none / shrink):
    through the Preview the answer is the full run's, every step."""
    calls = _count_prepares(monkeypatch)
    for seed in range(8):
        for cfg in (dict(outline="none", dark=True),
                    dict(outline="shrink", dark=False)):
            case = _Case(seed=seed, wall_kind=wall_kind, **cfg)
            assert case.stage is None
            rng = np.random.default_rng(seed + 700)
            points, ids, nid = [], [], 1
            for _k in range(5):
                if points and rng.random() < 0.3:
                    j = int(rng.integers(len(points)))
                    points.pop(j)
                    ids.pop(j)
                else:
                    points.append(case.any_point(rng))
                    ids.append(nid)
                    nid += 1
                full = case.full(points, ids)
                inc = case.by_click(points, ids)
                assert np.array_equal(full.result, inc.result) if (
                    full.result is not None) else inc.result is None
                assert full.status == inc.status
                assert "faint=n/a" not in inc.log_line()
    assert calls == []


#: how often the guard sent a box or a click to the full flood, per wall
FALLBACKS = {}


@pytest.mark.parametrize("wall_kind", ["membrane", "gradient"])
def test_sixty_four_sessions_on_membrane_and_gradient(wall_kind):
    """32 random sessions per wall (16 seeds, outline none / shrink, 8
    changes each): the stage and the Preview path equal the full run on
    every step; the guard's work is counted and printed."""
    boxes = full_boxes = clicks = whole = 0
    for seed in range(16):
        for cfg in (dict(outline="none", dark=True),
                    dict(outline="shrink", dark=False)):
            case = _Case(seed=seed, wall_kind=wall_kind, **cfg)
            boxes += 1
            full_boxes += case.stage is None
            clicks += _random_session(case, np.random.default_rng(seed + 50),
                                      8)
            if case.stage is not None:
                whole += case.stage._low_ties
    FALLBACKS[wall_kind] = (full_boxes, boxes, whole, clicks)
    print("\n[ADD-INC-GUARD] {}: {} of {} boxes keep the full run (flat "
          "spots); {} of {} clicks flooded the whole box for a marker "
          "tie".format(wall_kind, full_boxes, boxes, whole, clicks))
    assert boxes == 32


class _ElevCase:
    """A box handed its wall picture directly (no wall building): the
    clip plateaus of a normalised picture drawn by hand."""

    def __init__(self, elev, *, smooth=False, radius=1.0):
        self.elev = np.asarray(elev, np.float32)
        shape = self.elev.shape
        self.existing = np.zeros(shape, np.int32)
        self.roi = tuple(slice(0, n) for n in shape)
        self.params = dict(_cells_params(dark=False, smooth=smooth),
                           click_radius_um=radius)
        self.holder = {}
        INC.warm_kernel()

    def run(self, pts, ids, **kw):
        return AP.run_add_preview(
            preset="cells", roi=self.roi, existing_roi=self.existing,
            next_label=1, spacing=(1.0, 1.0, 1.0), params=self.params,
            wall_roi=self.elev, click_points=list(pts), reserved_ids=list(ids),
            elevation_roi=self.elev, **kw)


def _dim_cell(seed):
    """Walls of exactly 1.0 (the top clip) and one cell whose inside is
    exactly 0.0 (the bottom clip); the rest distinct values."""
    rng = np.random.default_rng(seed)
    shape = (24, 48, 48)
    e = ndi.gaussian_filter(rng.random(shape), 1.0).astype(np.float32) * 0.5
    e += 0.05
    e[:, ::12, :] = 1.0
    e[:, :, ::12] = 1.0
    e[::8, :, :] = 1.0
    cy, cx = 1 + int(rng.integers(0, 3)), 1 + int(rng.integers(0, 3))
    dim = (slice(9, 16), slice(12 * cy + 1, 12 * cy + 12),
           slice(12 * cx + 1, 12 * cx + 12))
    e[dim] = 0.0
    inside = [(11, 12 * cy + 3, 12 * cx + 3), (13, 12 * cy + 9, 12 * cx + 9)]
    return e, inside


def test_the_clip_plateaus_the_dim_cell_equals_the_full_run():
    n_whole = 0
    for seed in range(6):
        e, inside = _dim_cell(seed)
        case = _ElevCase(e, smooth=bool(seed % 2))
        rng = np.random.default_rng(seed)
        pts, ids, nid = list(inside), [1, 2], 3
        for k in range(8):
            if k and k % 3 == 2 and len(pts) > 2:
                j = int(rng.integers(2, len(pts)))
                pts.pop(j)
                ids.pop(j)
            elif k:
                pts.append(tuple(int(v) for v in (
                    rng.integers(1, e.shape[0] - 1),
                    rng.integers(1, e.shape[1] - 1),
                    rng.integers(1, e.shape[2] - 1))))
                ids.append(nid)
                nid += 1
            full = case.run(pts, ids)
            inc = case.run(pts, ids, stage_holder=case.holder,
                           stage_key=("dim", seed))
            assert not [k for k in _same_result(full, inc)
                        if k not in ("messages",)], _same_result(full, inc)
        stage = case.holder.get("stage")
        n_whole += 0 if stage is None else stage._low_ties
    print("\n[ADD-INC-GUARD] dim cell: {} clicks flooded the whole box "
          "(two clicks on the 0.0 patch)".format(n_whole))
    assert n_whole > 0


def _two_level(seed=0):
    """A tie at 0.5 whose order comes from the 0.1 level below it."""
    shape = (9, 9, 60)
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    e = (0.6 + 0.39 * rng.permutation(n) / n).reshape(shape).astype(
        np.float32)
    line = e[4, 4]
    line[2] = 0.02
    line[3:17] = 0.1
    line[17:20] = 0.5
    line[20:22] = 0.1
    line[22] = 0.05
    return e


def test_the_two_level_tie_keeps_the_full_run():
    e = _two_level()
    case = _ElevCase(e, radius=0.5)
    a, b = (4, 4, 2), (4, 4, 22)
    for pts, ids in (([a], [1]), ([a, b], [1, 2]), ([b, a], [2, 1])):
        full = case.run(pts, ids)
        inc = case.run(pts, ids, stage_holder=case.holder, stage_key=("t",))
        assert np.array_equal(full.result, inc.result)
    assert case.holder.get("full") and case.holder.get("stage") is None
    spec = AP.cells_spec(case.params)
    stage, refusal = INC.prepare(case.existing, {"membrane": e}, spec,
                                 roi=case.roi, spacing=(1.0, 1.0, 1.0),
                                 elevation=e)
    assert stage is None and refusal[0] == INC.FULL_RUN
    assert "between the lowest and the highest" in refusal[1]


# --------------------------------------------------------------------------- #
# one console line for every Cells click run, a prepared box or none
# --------------------------------------------------------------------------- #
def test_a_click_run_with_no_prepared_box_says_why():
    case = _Case(seed=2)
    pts = [case.any_point(np.random.default_rng(1))]
    for extra, why in (({"growth": AP.GROW_SPREAD}, "Spread"),
                       ({"wall_kind": "intensity"}, "walls not redone"),
                       ({}, "no prepared box here: over the live limit")):
        params = dict(case.params, **extra)
        res = AP.run_add_preview(
            preset="cells", roi=case.roi, existing_roi=case.existing,
            next_label=1, spacing=SP, params=params, wall_roi=case.wall,
            dark_rois=case.darks,
            click_points=[tuple(c + case.pad for c in p) for p in pts],
            reserved_ids=[1], dark_threshold=case.dt,
            outline_roi=case.outline)
        assert res.click_how == "full run ({})".format(why), res.click_how
        assert res.click_line().startswith(
            "[ADD] cells by click: full run ({}), window ".format(why))
    # the whole patch's fill with a background (the tick off)
    whole = _Case(seed=2, whole=True)
    params = dict(whole.params)
    params.pop("box_rule")
    res = AP.run_add_preview(
        preset="cells", roi=whole.roi, existing_roi=whole.existing,
        next_label=1, spacing=SP, params=params, wall_roi=whole.wall,
        dark_rois=whole.darks, click_points=pts, reserved_ids=[1],
        dark_threshold=whole.dt, outline_roi=whole.outline)
    assert res.click_how == (
        "full run (the whole patch's fill with a background)")


def test_a_spread_only_knob_does_not_change_the_fill_spec():
    base = {"source": AP.SOURCE_CLICKS}
    for extra in ({"wall_cost": 7.0}, {"growth_limit": 9.0},
                  {"include_membrane": False}):
        assert AP.cells_spec(dict(base, **extra)) == AP.cells_spec(base)
        live = dict(base, whole_patch=True, box_rule=True)
        assert AP.cells_spec(dict(live, **extra)) == AP.cells_spec(live)
    spread = dict(base, growth=AP.GROW_SPREAD)
    assert AP.cells_spec(dict(spread, wall_cost=7.0)) != AP.cells_spec(spread)
    patch_ = dict(base, whole_patch=True)
    assert (AP.cells_spec(dict(patch_, include_membrane=False))
            != AP.cells_spec(patch_))
    assert AP.cells_spec(dict(patch_, wall_cost=7.0)) == AP.cells_spec(patch_)


# --------------------------------------------------------------------------- #
# a whole patch filled the way a box is: its faces are open
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("outline,dark", [("none", True), ("shrink", True),
                                          ("none", False)])
def test_faces_open_click_by_click_equals_the_full_preview(outline, dark):
    compared = 0
    for seed in range(12):
        case = _Case(seed=seed, whole=True, outline=outline, dark=dark)
        spec = AP.cells_spec(case.params)
        assert spec.params["faces_outside"] is False
        if case.stage is None:
            continue
        compared += _random_session(case, np.random.default_rng(seed), 6)
    assert compared >= 24


def test_open_faces_let_a_cell_reach_the_patch_edge():
    """The same clicks on the same patch, raw cells: with the faces in the
    outside marker no cell holds a face voxel; with the faces open cells
    reach them."""
    seed = next(s for s in range(2, 20)
                if _Case(seed=s, whole=True).stage is not None)
    case = _Case(seed=seed, whole=True)
    rng = np.random.default_rng(3)
    shape = case.existing.shape
    pts = []
    while len(pts) < 6:
        # a click's own ball never on a face (clicks are markers anywhere)
        pt = case.any_point(rng)
        if all(3 <= v < n - 3 for v, n in zip(pt, shape)):
            pts.append(pt)
    ids = list(range(1, 7))
    shut = dict(case.params)
    shut.pop("box_rule")
    shut["whole_patch"] = False
    edge = np.zeros(case.existing.shape, bool)
    edge[:, 0, :] = edge[:, -1, :] = edge[:, :, 0] = edge[:, :, -1] = True
    edge[0] = edge[-1] = True
    on_edge = []
    for params in (case.params, shut):
        spec = AP.cells_spec(params)
        stage, refusal = INC.prepare(
            case.existing, {"membrane": case.wall}, spec, roi=case.roi,
            dark_channels=case.darks, dark_threshold=case.dt, spacing=SP,
            elevation=case.elevation, outline=case.outline)
        assert refusal is None, refusal
        stage.step(pts, ids)
        on_edge.append(int(np.count_nonzero(stage.raw_result()[edge])))
    assert on_edge[0] > 0 and on_edge[1] == 0, on_edge


# --------------------------------------------------------------------------- #
# the flat-spot rule is proportional: coincidental pairs pass, plateaus not
# --------------------------------------------------------------------------- #
def _survey(e, existing=None):
    existing = np.zeros(e.shape, np.int32) if existing is None else existing
    return INC.flat_survey(e, existing, existing, dark_raw=None,
                           outline=None, clearance_um=0.0,
                           spacing=(1.0, 1.0, 1.0))


def _distinct(shape, seed=0):
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    return (0.05 + 0.9 * rng.permutation(n) / n).reshape(shape).astype(
        np.float32)


def _inject_pairs(e, n_pairs, rng, *, inside=None):
    """``n_pairs`` touching voxel pairs set to one value (strictly between
    the clips), each a spot of exactly two voxels, apart from each other;
    returns the pairs."""
    lo, hi = float(e.min()), float(e.max())
    shape = e.shape
    done, used = [], set()
    tries = 0
    while len(done) < n_pairs and tries < 100 * n_pairs + 1000:
        tries += 1
        axis = int(rng.integers(3))
        a = [int(rng.integers(1, n - 2)) for n in shape]
        b = list(a)
        b[axis] += 1
        a, b = tuple(a), tuple(b)
        if inside is not None and not (inside[a] and inside[b]):
            continue
        if not (lo < e[a] < hi):
            continue
        near = set()
        for q0 in (a, b):
            for ax in range(3):
                for d in (-1, 1):
                    q = list(q0)
                    q[ax] += d
                    near.add(tuple(q))
        near -= {a, b}
        if near & used or a in used or b in used:
            continue
        if any(e[q] == e[a] for q in near):
            continue
        e[b] = e[a]
        used |= near | {a, b}
        done.append((a, b))
    assert len(done) == n_pairs, (len(done), n_pairs)
    return done


def test_the_flat_spot_rule_lets_sparse_coincidences_through():
    # 310,000 voxels, the growth space a few hundred fewer: the limit is 3
    shape = (31, 100, 100)
    e = _distinct(shape)
    why, _low, _o, n = _survey(e)
    assert why is None and n == 0
    _inject_pairs(e, 3, np.random.default_rng(1))
    why, _low, _o, n = _survey(e)
    assert why is None and n == 3
    _inject_pairs(e, 1, np.random.default_rng(2))
    why, _low, _o, n = _survey(e)
    assert n == 4 and why is not None
    assert "between the lowest and the highest" in why
    assert "per 100,000" in why


def test_one_flat_spot_over_three_voxels_keeps_the_full_run():
    shape = (40, 100, 100)                 # the limit is 4 pairs
    e = _distinct(shape)
    e[20, 50, 50:53] = e[20, 50, 50]       # a run of three: two pairs
    why, _low, _o, n = _survey(e)
    assert why is None and n == 2
    e[20, 50, 53] = e[20, 50, 50]          # a run of four: three pairs
    why, _low, _o, n = _survey(e)
    assert n == 3 and why is not None and "flat spot over 3" in why


def test_a_flat_spot_at_the_clips_is_not_counted():
    e = _distinct((20, 40, 40))
    e[5:8, 5:8, 5:8] = e.min()
    e[10:13, 10:13, 10:13] = e.max()
    why, _low, _o, n = _survey(e)
    assert why is None and n == 0


def test_the_two_level_tie_keeps_its_verdict():
    why, _l, _o, n = _survey(_two_level())
    assert why is not None and n > 3


def test_the_console_line_says_the_flat_pairs(monkeypatch):
    case = next(c for c in (_Case(seed=s) for s in range(2, 12))
                if c.stage is not None)
    a = case.any_point(np.random.default_rng(1))
    real = INC.CellsClickFill.__init__

    def init(self, *args, **kw):
        kw["n_flat_pairs"] = 9
        real(self, *args, **kw)

    monkeypatch.setattr(INC.CellsClickFill, "__init__", init)
    res = case.by_click([a], [1])
    assert res.by_click and res.click_how == "stage prepared, 9 flat pairs"
    assert res.click_line().startswith(
        "[ADD] cells by click: stage prepared, 9 flat pairs, window ")
    res = case.by_click([a, case.any_point(np.random.default_rng(2))],
                        [1, 2])
    assert res.click_how == "stage reused, 9 flat pairs"


#: The measured ceiling: the most voxels one click's windowed answer
#: differed from the full flood of the same clicks, over every session of
#: ``test_the_windowed_answer_stays_near_the_full_flood_with_flat_pairs``.
FLAT_PAIR_DIFF_CEILING = 8          # measured: 0 (160 clicks, 2026-11-22)


def test_a_low_patch_pair_floods_whole_once_not_at_every_later_click(
        monkeypatch):
    """Two clicks whose balls share a flat patch at the lowest wall value
    flood the whole box (their order is the full flood's).  A later click
    far from them is a window again -- before, that pair made every later
    click flood the whole box for the rest of the session (8 s a click on a
    5 M patch, in the owner's hands).  The answers equal the full flood."""
    from packages.epicycle_annotator.reshape.walls import build_elevation

    INC.warm_kernel()
    shape = (36, 96, 96)
    spec = AP.cells_spec(_cells_params(dark=True))
    img, kept, cells = _patch(shape, n_cells=40, seed=107, n_kept=2)
    roi = tuple(slice(0, n) for n in shape)
    elev = build_elevation("membrane", img, np.ones(shape, bool),
                           spacing=SP, clip=0.01, smooth=1.0,
                           ridge=False)[0].astype(np.float32)
    chans = {"mem": img}
    dt = R.patch_dark_threshold(chans, spacing=SP, stride=1)
    kw = dict(roi=roi, dark_channels=chans, dark_threshold=dt, spacing=SP,
              elevation=elev)
    win, refusal = INC.prepare(kept, {"membrane": img}, spec, **kw)
    assert refusal is None, refusal
    full, _r = INC.prepare(kept, {"membrane": img}, spec, **kw)
    # two clicks on one of the stage's own flat patches at the lowest wall
    # value (the darkest 1 % of a normalised picture), at least five voxels
    # apart, in free space and not on a patch the outside or a kept core
    # holds; a third click far away in another cell
    free_mask = (cells > 0) & (kept == 0)
    low = win._low
    a = b = None
    pids, counts = np.unique(low[(low > 0) & free_mask], return_counts=True)
    for pid, _c in sorted(zip(pids, counts), key=lambda t: -t[1]):
        if int(pid) in win._low_other:
            continue
        vox = np.argwhere((low == pid) & free_mask)
        for p in vox:
            d = np.abs(vox - p).sum(axis=1)
            j = int(np.argmax(d))
            if d[j] >= 5:
                a = tuple(int(v) for v in p)
                b = tuple(int(v) for v in vox[j])
                break
        if a is not None:
            break
    assert a is not None, "no low patch wide enough in this picture"
    free = np.argwhere(free_mask)

    def _off_every_low_patch(p):
        bb = INC._ball_box(tuple(int(v) for v in p), win._click_r,
                           win.spacing, win.shape)
        return bb is not None and not low[INC._slices(bb)].any()

    far = next(tuple(int(v) for v in p) for p in free[::-1]
               if cells[tuple(p)] != cells[a]
               and abs(int(p[1]) - a[1]) + abs(int(p[2]) - a[2]) > 40
               and _off_every_low_patch(p))
    whole = int(np.prod(shape))
    pts, ids = [a], [1]
    win.step(pts, ids)
    pts, ids = [a, b], [1, 2]
    st = win.step(pts, ids)
    assert st.window_voxels == whole, "the shared low patch: the whole box"
    pts, ids = [a, b, far], [1, 2, 3]
    st = win.step(pts, ids)
    assert st.window_voxels < whole // 4, \
        "a click far from the pair is a window again"
    with monkeypatch.context() as m:
        m.setattr(INC, "WINDOW_WHOLE_SHARE", 0.0)
        full.step(pts, ids)
    assert np.array_equal(win.raw_result(), full.raw_result())


@pytest.mark.parametrize("mode", ["box", "whole_dark", "whole_nodark"])
def test_a_low_patch_pair_then_far_clicks_against_the_full_flood(
        mode, monkeypatch):
    """Random sessions that start with two clicks on one of the stage's own
    lowest-value flat patches, then clicks and removals elsewhere; the
    windowed raw cells against a full flood of the same clicks after every
    step.  The pair is flooded whole once, not at every later click; the
    full flood's order among equal markers can shift with a far click, so
    the two may swap voxels on their patch.  Pinned: with the box faces or
    the dark outside as markers no step differs (measured 0 of 528 each);
    on a whole patch with the dark tick off a few steps differ by a cell's
    worth at most (measured 3 of 264, 127 voxels).  Printed: the counts."""
    from packages.epicycle_annotator.reshape.walls import build_elevation

    INC.warm_kernel()
    shape = (36, 96, 96)
    dark = mode != "whole_nodark"
    pp = _cells_params(dark=dark)
    if mode != "box":
        pp.update(whole_patch=True, box_rule=True)
    spec = AP.cells_spec(pp)
    n_steps = n_diff = worst = 0
    for seed in range(100, 106):
        img, kept, cells = _patch(shape, n_cells=40, seed=seed, n_kept=2)
        roi = tuple(slice(0, n) for n in shape)
        elev = build_elevation("membrane", img, np.ones(shape, bool),
                               spacing=SP, clip=0.01, smooth=1.0,
                               ridge=False)[0].astype(np.float32)
        chans = {"mem": img}
        dt = R.patch_dark_threshold(chans, spacing=SP, stride=1)
        kw = dict(roi=roi, dark_channels=chans if dark else None,
                  dark_threshold=dt, spacing=SP, elevation=elev)
        win, refusal = INC.prepare(kept, {"membrane": img}, spec, **kw)
        if refusal is not None:
            continue
        full, _r = INC.prepare(kept, {"membrane": img}, spec, **kw)
        free_mask = (cells > 0) & (kept == 0)
        low = win._low
        a = b = None
        pids, counts = np.unique(low[(low > 0) & free_mask],
                                 return_counts=True)
        for pid, _c in sorted(zip(pids, counts), key=lambda t: -t[1]):
            if int(pid) in win._low_other:
                continue
            vox = np.argwhere((low == pid) & free_mask)
            for p in vox:
                d = np.abs(vox - p).sum(axis=1)
                j = int(np.argmax(d))
                if d[j] >= 5:
                    a = tuple(int(v) for v in p)
                    b = tuple(int(v) for v in vox[j])
                    break
            if a is not None:
                break
        if a is None:
            continue
        rng = np.random.default_rng(7 * seed)
        free = np.argwhere(free_mask)
        pts, ids, nid = [a, b], [1, 2], 3
        win.step(pts, ids)
        with monkeypatch.context() as m:
            m.setattr(INC, "WINDOW_WHOLE_SHARE", 0.0)
            full.step(pts, ids)
        for _k in range(10):
            if len(pts) > 2 and rng.random() < 0.3:
                j = int(rng.integers(2, len(pts)))
                pts.pop(j)
                ids.pop(j)
            else:
                pts.append(tuple(int(v) for v in
                                 free[rng.integers(len(free))]))
                ids.append(nid)
                nid += 1
            win.step(pts, ids)
            with monkeypatch.context() as m:
                m.setattr(INC, "WINDOW_WHOLE_SHARE", 0.0)
                full.step(pts, ids)
            d = int(np.count_nonzero(win.raw_result() != full.raw_result()))
            n_steps += 1
            n_diff += d > 0
            worst = max(worst, d)
    print("\n[ADD-INC-LOWPAIR] {}: {} of {} steps differed from the full "
          "flood, the most voxels {}".format(mode, n_diff, n_steps, worst))
    assert n_steps >= 30
    if mode == "whole_nodark":
        assert n_diff <= n_steps // 10, (n_diff, n_steps)
        assert worst <= 400, worst
    else:
        assert n_diff == 0, (n_diff, worst)


def test_the_windowed_answer_stays_near_the_full_flood_with_flat_pairs(
        monkeypatch):
    """Membrane boxes with 1 to 20 coincidental flat pairs per 100,000
    voxels injected into their wall picture, inside the cells (the density
    raised over the rule's own limit, so the windows meet every one):
    random click sessions, the windowed raw cells against a full flood of
    the same clicks after every change.  Printed: the voxels that
    differed."""
    from packages.epicycle_annotator.reshape.walls import build_elevation

    monkeypatch.setattr(INC, "FLAT_PAIRS_PER_VOXEL", 1e-3)
    INC.warm_kernel()
    shape = (36, 96, 96)
    spec = AP.cells_spec(_cells_params(dark=True))
    worst, per_density, n_clicks, n_diff_clicks = 0, {}, 0, 0
    for density in (1, 5, 10, 20):
        top = 0
        for seed in range(4):
            img, kept, cells = _patch(shape, n_cells=40, seed=100 + seed,
                                      n_kept=2)
            roi = tuple(slice(0, n) for n in shape)
            elev = build_elevation("membrane", img, np.ones(shape, bool),
                                   spacing=SP, clip=0.01, smooth=1.0,
                                   ridge=False)[0].astype(np.float32)
            rng = np.random.default_rng(1000 * density + seed)
            n_pairs = max(1, int(round(density * elev.size / 1e5)))
            _inject_pairs(elev, n_pairs, rng,
                          inside=(cells > 0) & (kept == 0))
            chans = {"mem": img}
            dt = R.patch_dark_threshold(chans, spacing=SP, stride=1)
            kw = dict(roi=roi, dark_channels=chans, dark_threshold=dt,
                      spacing=SP, elevation=elev)
            win, refusal = INC.prepare(kept, {"membrane": img}, spec, **kw)
            assert refusal is None, refusal
            assert win.n_flat_pairs >= n_pairs
            full, _r = INC.prepare(kept, {"membrane": img}, spec, **kw)
            free = np.argwhere((cells > 0) & (kept == 0))
            pts, ids, nid = [], [], 1
            for _k in range(10):
                if pts and rng.random() < 0.25:
                    j = int(rng.integers(len(pts)))
                    pts.pop(j)
                    ids.pop(j)
                else:
                    pts.append(tuple(int(v) for v in
                                     free[rng.integers(len(free))]))
                    ids.append(nid)
                    nid += 1
                win.step(pts, ids)
                with monkeypatch.context() as m:
                    m.setattr(INC, "WINDOW_WHOLE_SHARE", 0.0)
                    full.step(pts, ids)
                d = int(np.count_nonzero(win.raw_result()
                                         != full.raw_result()))
                n_clicks += 1
                n_diff_clicks += d > 0
                top = max(top, d)
        per_density[density] = top
        worst = max(worst, top)
    print("\n[ADD-INC-FLAT] the most voxels one click differed from the full "
          "flood, per injected flat pairs per 100,000 voxels: {}; {} of {} "
          "clicks differed at all".format(per_density, n_diff_clicks,
                                          n_clicks))
    assert worst <= FLAT_PAIR_DIFF_CEILING


# --------------------------------------------------------------------------- #
# a Stop mid step or mid finish breaks the stage for good
# --------------------------------------------------------------------------- #
def _three_cells(case):
    cells = [c for c in np.unique(case.cells)
             if c and ((case.cells == c) & (case.existing == 0)).sum() > 300]

    def pt_in(cell):
        f = np.argwhere((case.cells == cell) & (case.existing == 0))
        return tuple(int(v) for v in f[len(f) // 2])

    return pt_in(cells[0]), pt_in(cells[1]), pt_in(cells[2])


def test_a_stop_mid_step_breaks_the_stage_and_a_retry_raises():
    import threading

    from packages.epicycle_annotator.reshape.engine import ReshapeCancelled

    case = _Case(seed=2)
    a, b, _c = _three_cells(case)
    st, fin = case.step([a], [1])
    stop = threading.Event()
    stop.set()
    with pytest.raises(ReshapeCancelled):
        case.stage.step([a, b], [1, 2], cancel_event=stop)
    assert case.stage.broken
    with pytest.raises(RuntimeError):
        case.stage.step([a, b], [1, 2])
    with pytest.raises(RuntimeError):
        case.stage.finish()


def test_a_stop_mid_step_through_the_preview_prepares_again(monkeypatch):
    import threading

    case = _Case(seed=2)
    a, b, c = _three_cells(case)
    case.by_click([a], [1])
    first = case.holder["stage"]
    stop = threading.Event()
    real = INC.CellsClickFill._step

    def stopped_after_the_markers(self, pts, ids, cancel_event):
        stop.set()                      # the Stop lands inside the step
        return real(self, pts, ids, cancel_event)

    monkeypatch.setattr(INC.CellsClickFill, "_step",
                        stopped_after_the_markers)
    out = case.by_click([a, b], [1, 2], cancel_event=stop)
    assert out.status == "cancelled" and first.broken
    assert case.holder.get("stage") is None
    monkeypatch.setattr(INC.CellsClickFill, "_step", real)
    calls = _count_prepares(monkeypatch)
    for pts, ids in (([a, b], [1, 2]), ([a, b, c], [1, 2, 3])):
        inc = case.by_click(pts, ids)
        assert not _same_result(case.full(pts, ids), inc)
    assert len(calls) == 1 and case.holder["stage"] is not first


def test_a_stop_mid_finish_breaks_the_stage():
    import threading

    from packages.epicycle_annotator.reshape.engine import ReshapeCancelled

    case = _Case(seed=2)
    a, b, _c = _three_cells(case)
    case.stage.step([a, b], [1, 2])
    stop = threading.Event()
    stop.set()
    with pytest.raises(ReshapeCancelled):
        case.stage.finish(cancel_event=stop)
    assert case.stage.broken
    with pytest.raises(RuntimeError):
        case.stage.finish()


def test_the_finish_hands_back_a_read_only_view():
    case = _Case(seed=2)
    rng = np.random.default_rng(1)
    _st, fin = case.step([case.any_point(rng)], [1])
    assert not fin.result.flags.writeable
    with pytest.raises(ValueError):
        fin.result[0, 0, 0] = 1
    inc = case.by_click([case.any_point(np.random.default_rng(1))], [1])
    assert inc.result is None or inc.result.flags.writeable


# --------------------------------------------------------------------------- #
# click markers sharing an id (the seeds of one cell) go through the stage:
# every run of the refused-seed loop is one step of the same stage
# --------------------------------------------------------------------------- #
def _count_steps(monkeypatch):
    calls = []
    real = INC.CellsClickFill.step

    def spy(self, points, ids, **k):
        calls.append(list(ids))
        return real(self, points, ids, **k)

    monkeypatch.setattr(INC.CellsClickFill, "step", spy)
    return calls


@pytest.mark.parametrize("seed", [1, 4])
def test_seeds_of_one_cell_through_the_stage_equal_the_full_run(seed,
                                                                monkeypatch):
    case = _Case(seed=seed)
    if case.flat is not None or case.stage is None:
        pytest.skip("this box keeps the full run")
    steps = _count_steps(monkeypatch)
    rng = np.random.default_rng(seed + 20)
    points, ids, nid, grouped_ok = [], [], 1, 0
    for _ in range(9):
        if points and rng.random() < 0.45:
            ids.append(ids[int(rng.integers(len(ids)))])   # one more seed
        else:
            ids.append(nid)
            nid += 1
        points.append(case.any_point(rng))
        full = case.full(points, ids)
        inc = case.by_click(points, ids)
        assert not _same_result(full, inc), _same_result(full, inc)
        assert inc.seed_why == full.seed_why
        assert inc.apart_ids == full.apart_ids
        if len(set(ids)) < len(ids) and full.status == "ok":
            assert inc.by_click
            grouped_ok += 1
    assert grouped_ok and steps


def _two_cells():
    """Two walled cells with open space between them (as the extend
    gesture's tissue): a second seed in the other cell is refused."""
    shape = (20, 30, 30)
    rng = np.random.default_rng(0)
    mem = (np.full(shape, 0.05, np.float32)
           + rng.normal(0, 0.005, shape).astype(np.float32))
    for x0, x1 in [(1, 14), (16, 29)]:
        mem[3:17, 5:25, x0:x1] = 1.0
        mem[5:15, 7:23, x0 + 2:x1 - 2] = (
            0.05 + rng.normal(0, 0.005, (10, 16, x1 - x0 - 4))
        ).astype(np.float32)
    return mem


def _two_cells_run(points, ids, holder=None):
    mem = _two_cells()
    kw = {}
    if holder is not None:
        kw = dict(stage_holder=holder, stage_key=("two cells",))
    return AP.run_add_preview(
        preset="cells", roi=tuple(slice(0, n) for n in mem.shape),
        existing_roi=np.zeros(mem.shape, np.int32), next_label=500,
        spacing=(1.0, 1.0, 1.0),
        params=dict(source=AP.SOURCE_CLICKS, wall_kind="membrane",
                    wall_chain="none", dark_outside=False, core_erode_um=1.0,
                    click_radius_um=1.0, min_size=0, clear_border=True,
                    edge_margin=3, auto_smooth=False, smooth_sigma=1.5),
        wall_roi=mem, dark_rois={"membrane": mem}, click_points=list(points),
        reserved_ids=list(ids), **kw)


def test_a_refused_seed_is_one_more_step_on_the_same_stage(monkeypatch):
    """A(#500) in one cell, B(#500) and C(#501) in the other: the loop runs
    the stage twice (with B, then without it), prepares once, and answers
    as the full run does -- B refused, C's cell grown as if B was never
    placed."""
    INC.warm_kernel()
    A, B, C = (10, 15, 7), (10, 10, 22), (10, 20, 22)
    holder = {}
    first = _two_cells_run([A, C], [500, 501], holder)
    if holder.get("stage") is None:
        pytest.skip("this box keeps the full run")
    prepares = _count_prepares(monkeypatch)
    steps = _count_steps(monkeypatch)
    full = _two_cells_run([A, B, C], [500, 500, 501])
    inc = _two_cells_run([A, B, C], [500, 500, 501], holder)
    assert full.ok and inc.ok, (full.refusal, inc.refusal)
    assert inc.by_click and not full.by_click
    assert inc.seed_why == full.seed_why == [None, "apart", None]
    assert inc.apart_ids == full.apart_ids == [500]
    assert np.array_equal(inc.result, full.result)
    assert np.array_equal(inc.result, first.result)
    assert inc.ids == full.ids and inc.sizes == full.sizes
    assert prepares == []
    # the seeds are named by their place in the click list: the re-run
    # without B is the same stage with fewer points
    assert steps == [[1, 2, 3], [1, 3]]
    assert holder.get("stage") is not None
    assert "faint=n/a (by click)" in inc.log_line()


def test_a_refused_seed_stays_out_of_the_next_clicks(monkeypatch, capsys):
    """After B (#500) was refused, a further click D in A's cell runs ONE
    step without B (the refusal is remembered on the stage) and answers as
    the full run does.  Handing B in again at every click put a change at
    its place each time, and the click fill's one window grew to span every
    refused seed (the whole patch, seconds a click, in the owner's hands).
    A removed marker (the old list no longer a prefix) judges every seed
    again."""
    INC.warm_kernel()
    A, B, C, D = (10, 15, 7), (10, 10, 22), (10, 20, 22), (10, 8, 10)
    holder = {}
    _two_cells_run([A, C], [500, 501], holder)
    if holder.get("stage") is None:
        pytest.skip("this box keeps the full run")
    _two_cells_run([A, B, C], [500, 500, 501], holder)
    assert holder.get("apart") and holder["apart"][1] == {(B, 500)}
    steps = _count_steps(monkeypatch)
    capsys.readouterr()
    full = _two_cells_run([A, B, C, D], [500, 500, 501, 502])
    inc = _two_cells_run([A, B, C, D], [500, 500, 501, 502], holder)
    assert full.ok and inc.ok, (full.refusal, inc.refusal)
    assert inc.by_click
    assert steps == [[1, 3, 4]], "one step, B left out from the start"
    assert "refused last time stay out: [500]" in capsys.readouterr().out
    assert inc.seed_why == full.seed_why == [None, "apart", None, None]
    assert inc.apart_ids == full.apart_ids == [500]
    assert np.array_equal(inc.result, full.result)
    assert inc.ids == full.ids and inc.sizes == full.sizes
    assert holder["apart"][1] == {(B, 500)}
    # a marker removed: the list is no longer the old one plus new
    # clicks, so every seed is judged again (B refused once more)
    steps = _count_steps(monkeypatch)
    inc2 = _two_cells_run([A, B, D], [500, 500, 502], holder)
    assert inc2.ok and inc2.apart_ids == [500]
    assert len(steps) == 2 and steps[0] == [1, 2, 3]


def test_a_remembered_refusal_is_placement_time_judgement(monkeypatch):
    """The corner the memo decides differently from one run of every
    seed at once (pinned so a change is seen): B (#500) refused in the other
    cell, then t a second seed of #501 placed BEYOND B as seen from C.  One
    run with every seed present refuses B and t together (B's cell stood
    between t and C's), and never judges t again; click by click B was never
    there, so t touches and is kept.  The memo gives placement-time
    judgement (a seed refused when placed stays refused), the stated rule
    ("as if that seed had never been placed"); the one-go difference is
    recorded for the owner."""
    INC.warm_kernel()
    A, B, C, t = (10, 15, 7), (10, 10, 22), (10, 20, 22), (10, 8, 22)
    holder = {}
    _two_cells_run([A, C], [500, 501], holder)
    if holder.get("stage") is None:
        pytest.skip("this box keeps the full run")
    _two_cells_run([A, B, C], [500, 500, 501], holder)
    full = _two_cells_run([A, B, C, t], [500, 500, 501, 501])
    inc = _two_cells_run([A, B, C, t], [500, 500, 501, 501], holder)
    assert full.ok and inc.ok
    assert inc.seed_why == [None, "apart", None, None]
    assert inc.apart_ids == [500]
    assert full.seed_why == [None, "apart", None, "apart"], \
        "the one-go run refuses t too (recorded); if this changed, the " \
        "two paths may agree now -- update the rule's words"
    # the memo is valid for the stage that judged it only: a new stage
    # object (the labels changed under the same key) judges every seed
    old = holder["stage"]
    holder["apart"] = (holder["apart"][0], holder["apart"][1], object())
    steps = _count_steps(monkeypatch)
    again = _two_cells_run([A, B, C, t], [500, 500, 501, 501], holder)
    assert again.ok and holder["stage"] is old
    assert len(steps) >= 2 and steps[0] == [1, 2, 3, 4], "B judged again"


def test_a_grouped_run_falls_back_whole_when_the_stage_fails(monkeypatch,
                                                             capsys):
    INC.warm_kernel()
    A, B, C = (10, 15, 7), (10, 10, 22), (10, 20, 22)
    holder = {}
    _two_cells_run([A, C], [500, 501], holder)
    if holder.get("stage") is None:
        pytest.skip("this box keeps the full run")

    def boom(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(INC.CellsClickFill, "step", boom)
    capsys.readouterr()
    full = _two_cells_run([A, B, C], [500, 500, 501])
    inc = _two_cells_run([A, B, C], [500, 500, 501], holder)
    assert inc.ok and not inc.by_click
    assert np.array_equal(inc.result, full.result)
    assert inc.seed_why == full.seed_why and inc.apart_ids == full.apart_ids
    assert holder.get("stage") is None
    assert capsys.readouterr().out.count("click-by-click fill failed") == 1

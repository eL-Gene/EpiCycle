"""The region target's third way to grow cells -- ``growth="fill_with_background"``.

A whole patch on Cells is filled by ONE seeded watershed over the growth
space, with the new markers, the kept objects as competitors (on the empty
voxels next to them) and one more marker: the background.  The background
seed is every space voxel that holds no source and lies farther from its
nearest marker than that marker's own distance ``r``: ``r = max(6 um,
0.75 x the distance from the marker's centroid to the nearest other
centroid)``; a lone marker takes 6 um.  The distance is where the
background seed STARTS, not how far a cell grows: on flat terrain the cut
lands about half-way.  The seed is geometric (the space, the markers and
the spacing; never an image channel) and is computed in z-slabs with a
halo, which gives exactly the whole-volume answer.

Pinned here: the seed rule against a brute-force reference; slabs equal one
call; the flat-terrain cut near the midline; kept objects keep their gap;
the outline is a hard edge; ids and pieces; the honeycomb and the sparse
layer; Stop before the stage, between slabs and from the progress callback;
Spread's growth space unchanged by the lift (digests produced by the code
before it); the spacing in every micron; the seed reads no channel; the
id guard; dense markers make no seed; the lone marker; the whole-patch
size ceiling; the engine defaults.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import hashlib
import inspect
import threading

import numpy as np
import pytest
from scipy import ndimage as ndi

from packages.epicycle_annotator.reshape import ReshapeCancelled, ReshapeSpec
from packages.epicycle_annotator.reshape import compete as C
from packages.epicycle_annotator.reshape import region as R
from packages.epicycle_annotator.reshape import spec as S

from .test_reshape_region import _tissue

ISO05 = (0.5, 0.5, 0.5)
ANISO = (1.0, 0.5, 0.5)
SPACINGS = [ISO05, ANISO]
FILL = "fill_with_background"
SIX = ndi.generate_binary_structure(3, 1)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _box(shape):
    return tuple(slice(0, int(n)) for n in shape)


def _fill_spec(seeds="paint", outside="faces", **params):
    base = {"wall_channel": "membrane", "ridge": False, "outside": outside,
            "growth": FILL, "include_membrane": False}
    base.update(params)
    return ReshapeSpec(target="region", seeds=seeds, walls="membrane",
                       params=base)


def _fill(labels, elev, spec, *, spacing, seed_points=None, seed_layer=None,
          **kw):
    return R.reshape_empty_region(
        labels, {}, spec, roi=_box(labels.shape), seed_points=seed_points,
        seed_layer=seed_layer, spacing=spacing, elevation=elev, **kw)


def _nz_for(spacing, um):
    """Slices for ``um`` microns of depth at this spacing."""
    return int(round(um / spacing[0]))


def _clicks(shape, points, spacing, radius_um=1.0):
    return R.rasterise_points(points, shape, radius_um=radius_um,
                              spacing=spacing)


def _bright_dark_args(shape):
    """A dark channel that is never dark (for a run with an outline, which
    turns the dark plumbing on)."""
    ch = {"c": np.ones(shape, np.float32)}
    threshold = {"t": 0.0, "lo": {"c": 0.0}, "hi": {"c": 1.0}}
    return ch, threshold


def _reference_seeds(new, space, sources, spacing):
    """Brute force: every voxel against every marker voxel.  Returns
    ``(seeds, ambiguous, nearest_id, dmin, r)``; ``ambiguous`` marks voxels
    whose tied nearest markers would decide differently."""
    sp = np.asarray(spacing, float)
    coords = np.argwhere(new != 0)
    lab = new[new != 0]
    ids = np.unique(lab)
    cents = np.array([coords[lab == i].mean(axis=0) * sp for i in ids])
    if ids.size == 1:
        radius = {int(ids[0]): 6.0}
    else:
        d = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        radius = {int(i): max(6.0, 0.75 * float(nn[k]))
                  for k, i in enumerate(ids)}
    pts = np.argwhere(np.ones(new.shape, bool))
    phys_m = coords * sp
    seeds = np.zeros(new.size, bool)
    amb = np.zeros(new.size, bool)
    near_id = np.zeros(new.size, np.int64)
    dmin_all = np.zeros(new.size, float)
    for a in range(0, pts.shape[0], 4096):
        p = pts[a:a + 4096] * sp
        dist = np.sqrt(((p[:, None, :] - phys_m[None, :, :]) ** 2).sum(-1))
        dmin = dist.min(axis=1)
        dmin_all[a:a + 4096] = dmin
        for k in range(p.shape[0]):
            tied = np.unique(lab[dist[k] <= dmin[k] + 1e-9])
            decisions = {bool(dmin[k] > radius[int(t)] + 1e-9) for t in tied}
            near_id[a + k] = int(tied[0])
            if len(decisions) > 1:
                amb[a + k] = True
            seeds[a + k] = decisions.pop()
    shape = new.shape
    seeds = seeds.reshape(shape) & space & (sources == 0)
    return (seeds, amb.reshape(shape), near_id.reshape(shape),
            dmin_all.reshape(shape), radius)


def _digest(a):
    if a is None:
        return None
    a = np.ascontiguousarray(a)
    if a.dtype == bool:
        payload = np.packbits(a).tobytes()
    else:
        payload = a.astype(np.int64).tobytes()
    return hashlib.sha256(repr(a.shape).encode() + payload).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# the engine's defaults; the new growth value
# --------------------------------------------------------------------------- #
def test_the_engine_defaults():
    assert R.BACKGROUND_FROM_MARKER_UM == 6.0
    assert R.BACKGROUND_NN_SHARE == 0.75
    assert R.WHOLE_FILL_MAX_VOXELS == 100_000_000
    assert C.GROWTH_LIMIT == S.DEFAULT_PARAMS["growth_limit"] == 28.0
    assert C.WALL_COST == S.DEFAULT_PARAMS["wall_cost"] == 40.0
    assert FILL in S.GROWTH_VALUES
    assert S.GROWTH_VALUES[:2] == ("flood", "compete")


def test_the_whole_patch_takes_spread_or_the_fill_never_the_flood():
    assert _fill_spec(whole_patch=True).params["whole_patch"] is True
    assert _fill_spec(growth="compete", whole_patch=True).params["growth"] == "compete"
    with pytest.raises(ValueError, match="whole_patch"):
        _fill_spec(growth="flood", whole_patch=True)
    # the fill is legal on a box too
    assert _fill_spec().params["growth"] == FILL
    with pytest.raises(ValueError, match="flood, compete, fill_with_background"):
        _fill_spec(growth="spread")


# --------------------------------------------------------------------------- #
# the seed rule
# --------------------------------------------------------------------------- #
def _seed_world(spacing):
    shape = (_nz_for(spacing, 6.0), 40, 40)
    zc = shape[0] // 2
    # two markers 3 um apart (r = 6) and one far from both (r > 6)
    new = _clicks(shape, [(zc, 10, 5), (zc, 10, 11), (zc, 30, 30)], spacing)
    existing = np.zeros(shape, np.int32)
    existing[:, 32:38, 2:8] = 9            # a kept object, with its ring
    space = existing == 0
    space[:, :2, :] = False                # some voxels outside the space
    sources, n = C.kept_competitors(new, existing, space)
    assert n == 3
    return new, space, sources


@pytest.mark.parametrize("spacing", SPACINGS)
def test_seed_rule(spacing):
    new, space, sources = _seed_world(spacing)
    got = R._background_seeds(new, space, sources, spacing)
    want, amb, near, dmin, radius = _reference_seeds(new, space, sources, spacing)
    assert got.dtype == bool and got.shape == new.shape
    # equal wherever the nearest marker is not a tie that would decide
    # both ways
    assert not ((want != got) & ~amb).any()
    # never a marker, a competitor, or outside the space
    assert not (got & (new != 0)).any()
    assert not (got & (sources != 0)).any()
    assert not (got & ~space).any()
    assert got.any()
    # the per-marker distance matters: past 6 um from the far marker but
    # within its own r, no seed
    far = 3
    assert radius[far] > 6.5
    between = (near == far) & (dmin > 6.0 + 1e-6) & (dmin <= radius[far]) & space & (sources == 0)
    assert between.any()
    assert not got[between].any()


@pytest.mark.parametrize("spacing", SPACINGS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_slabs_equal_one_call(spacing, seed, monkeypatch):
    rng = np.random.default_rng(seed)
    shape = (int(round(24.0 / spacing[0])), 30, 30)
    pts = [tuple(int(rng.integers(0, n)) for n in shape) for _ in range(9)]
    new = _clicks(shape, pts, spacing)
    space = rng.random(shape) > 0.1
    sources = new.copy()
    one = R._background_seeds(new, space, sources, spacing, slab_voxels=10 ** 9)
    calls = []
    real = R.ndi.distance_transform_edt

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(R.ndi, "distance_transform_edt", counting)
    slabs = R._background_seeds(new, space, sources, spacing, slab_voxels=1)
    assert len(calls) >= 1
    assert np.array_equal(one, slabs)
    assert one.any() and not one.all()


def _spy_edt(monkeypatch):
    """Record the voxels of every distance map the seed measures."""
    sizes = []
    real = R.ndi.distance_transform_edt

    def spy(free, *a, **k):
        sizes.append(int(np.asarray(free).size))
        return real(free, *a, **k)

    monkeypatch.setattr(R.ndi, "distance_transform_edt", spy)
    return sizes


def _slab_scene(name, spacing):
    """Deterministic scenes for the slabs: ``"gap"`` -- markers only near
    the bottom of a tall volume, so the slabs near the top see no marker
    within their halo; ``"far"`` -- a pair of markers near the bottom and
    one marker far up the column, whose own distance is far past the
    6 um floor (the halo must follow it)."""
    nz = int(round(100.0 / spacing[0]))
    shape = (nz, 30, 30)
    if name == "gap":
        pts = [(2, 10, 10), (2, 20, 20)]
    else:
        pts = [(2, 10, 10), (2, 10, 16), (int(round(30.0 / spacing[0])), 15, 15)]
    return _clicks(shape, pts, spacing)


@pytest.mark.parametrize("spacing", SPACINGS)
@pytest.mark.parametrize("scene", ["gap", "far"])
def test_slabs_equal_one_call_on_set_scenes(spacing, scene, monkeypatch):
    new = _slab_scene(scene, spacing)
    space = np.ones(new.shape, bool)
    one = R._background_seeds(new, space, new.copy(), spacing,
                              slab_voxels=10 ** 9)
    if scene == "far":
        assert R._background_radii(new, spacing).max() > 15.0
    sizes = _spy_edt(monkeypatch)
    slabs = R._background_seeds(new, space, new.copy(), spacing, slab_voxels=1)
    assert len(sizes) >= 1
    assert np.array_equal(one, slabs)
    assert one.any() and not one.all()
    # measured in slabs (fewer voxels than the volume per distance map),
    # and never much more work than one call
    assert max(sizes) < new.size
    assert sum(sizes) <= 3 * new.size


@pytest.mark.parametrize("spacing", SPACINGS)
def test_one_far_marker_does_not_multiply_the_slab_work(spacing, monkeypatch):
    """One marker far from the rest makes its distance -- and the halo --
    large; on a wide plane the slabs must not shrink to one slice each,
    every one measuring a distance map over most of the volume."""
    nz = int(round(20.0 / spacing[0]))
    shape = (nz, 64, 64)
    new = _clicks(shape, [(2, 4, 4), (2, 4, 10), (nz - 3, 60, 60)], spacing)
    space = np.ones(shape, bool)
    one = R._background_seeds(new, space, new.copy(), spacing,
                              slab_voxels=10 ** 12)
    sizes = _spy_edt(monkeypatch)
    slabs = R._background_seeds(new, space, new.copy(), spacing,
                                slab_voxels=8 * shape[1] * shape[2])
    assert sum(sizes) <= 3 * new.size, (len(sizes), sum(sizes) / new.size)
    assert np.array_equal(one, slabs)


# --------------------------------------------------------------------------- #
# the distance is where the seed STARTS, not a reach
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spacing", SPACINGS)
def test_flat_terrain_cut_at_midline(spacing):
    n_xy = 49
    shape = (int(round(24.0 / spacing[0])) | 1, n_xy, n_xy)
    c = tuple(n // 2 for n in shape)
    labels = np.zeros(shape, np.int32)
    elev = np.zeros(shape, np.float32)
    out = _fill(labels, elev, _fill_spec(click_radius_um=0.5), spacing=spacing,
                seed_points=[c])
    assert out.ok, out.messages
    res, seeds = out.result, out.seeds
    big_r = R.BACKGROUND_FROM_MARKER_UM
    for axis in range(3):
        for sign in (1, -1):
            line_res = []
            line_seed = []
            k = 0
            while True:
                idx = list(c)
                idx[axis] = c[axis] + sign * k
                if not 0 <= idx[axis] < shape[axis]:
                    break
                line_res.append(int(res[tuple(idx)]))
                line_seed.append(int(seeds[tuple(idx)]))
                k += 1
            edge = max(i for i, v in enumerate(line_seed) if v == 1)
            last = max(i for i, v in enumerate(line_res) if v == 1)
            # the first voxel past the cell is empty: the background took it
            assert line_res[last + 1] == 0
            cut_um = (last + 0.5 - edge) * spacing[axis]
            assert abs(cut_um - big_r / 2.0) <= 1.0, (axis, sign, cut_um)
            assert cut_um < 0.75 * big_r


# --------------------------------------------------------------------------- #
# kept objects, the outline, ids and pieces
# --------------------------------------------------------------------------- #
def _ring(existing):
    kept = existing != 0
    return ndi.binary_dilation(kept, structure=SIX) & ~kept


COARSE_Z = (2.0, 0.5, 0.5)


@pytest.mark.parametrize("spacing", SPACINGS + [COARSE_Z])
@pytest.mark.parametrize("membrane", [False, True])
def test_kept_object_keeps_its_gap(spacing, membrane):
    """A kept object keeps the empty voxels next to it, with the membrane
    step on or off -- also on a coarse z, where a new cell's voxel beside
    a seam voxel is nearer than the kept object below it."""
    shape = (_nz_for(spacing, 12.0), 40, 40)
    nz = shape[0]
    zc = nz // 2
    z_lo, z_hi = nz // 3, 2 * nz // 3
    labels = np.zeros(shape, np.int32)
    labels[z_lo:z_hi, 10:20, 25:35] = 7    # with faces in z, x and y
    before = labels.copy()
    elev = np.zeros(shape, np.float32)
    elev[:, :, 22] = 0.6                   # a bright wall in the gap
    elev[z_hi] = 0.3                       # a faint membrane on its top
    elev[z_lo - 1] = 0.3                   # ... and under it
    out = _fill(labels, elev, _fill_spec(include_membrane=membrane),
                spacing=spacing,
                seed_points=[(zc, 15, 12), (zc, 32, 10), (z_hi, 15, 38)])
    assert out.ok, out.messages
    assert np.array_equal(labels, before)
    assert not (out.result[labels != 0] != 0).any()
    ring = _ring(labels) & (out.seeds == 0)
    assert ring.any()
    assert not (out.result[ring] != 0).any(), int((out.result[ring] != 0).sum())


@pytest.mark.parametrize("spacing", SPACINGS)
def test_outline_is_a_hard_edge(spacing):
    shape = (_nz_for(spacing, 6.0), 40, 40)
    zc = shape[0] // 2
    labels = np.zeros(shape, np.int32)
    labels[:, 5:10, 25:30] = 3             # kept, outside the outline
    labels[:, 30:34, 5:9] = 4              # kept, inside it
    before = labels.copy()
    outline = np.zeros(shape, bool)
    outline[:, :, :20] = True
    ch, threshold = _bright_dark_args(shape)
    out = _fill(labels, np.zeros(shape, np.float32),
                _fill_spec(outline="cleared", include_membrane=True),
                spacing=spacing, seed_points=[(zc, 10, 5), (zc, 20, 15)],
                outline=outline, dark_channels=ch, dark_threshold=threshold)
    assert out.ok, out.messages
    assert np.array_equal(labels, before)
    assert not (out.result[~outline] != 0).any()
    assert not (out.result[labels != 0] != 0).any()
    assert (out.result[outline & (labels == 0)] != 0).any()


def _tissue_fill(spacing, membrane, *, outside="faces+dark", keep=(101, 106),
                 **kw):
    mem, nuc, cells = _tissue()
    ch = {"membrane": mem, "nuclei": nuc}
    labels = np.where(np.isin(cells, keep), cells, 0).astype(np.int32)
    spec = _fill_spec(seeds="labels", outside=outside,
                      include_membrane=membrane)
    out = R.reshape_empty_region(
        labels, {"membrane": mem}, spec, roi=_box(labels.shape),
        dark_channels=ch, seed_layer=cells,
        dark_threshold=R.patch_dark_threshold(ch, spacing=spacing, stride=1),
        spacing=spacing, **kw)
    assert out.ok, out.messages
    return out, labels


def _one_piece_each(out):
    ids = [int(v) for v in np.unique(out.result) if v]
    for i in ids:
        comp, k = ndi.label(out.result == i, structure=SIX)
        assert k == 1, (i, k)
        assert (out.seeds[comp == 1] == i).any(), i
    return ids


@pytest.mark.parametrize("spacing", SPACINGS)
def test_ids_and_pieces(spacing):
    out, labels = _tissue_fill(spacing, True)
    off, _labels = _tissue_fill(spacing, False)
    n = out.n_seeds
    assert n >= 4 and off.n_seeds == n
    for res in (out, off):
        ids = _one_piece_each(res)
        assert ids and set(ids) <= set(range(1, n + 1))
        assert not (res.result[labels != 0] != 0).any()
        assert res.n_tiles == 0
    # the membrane step runs: it only adds, and only bright voxels
    on_px, off_px = out.result != 0, off.result != 0
    assert not (off_px & ~on_px).any()
    added = on_px & ~off_px
    assert added.any()
    assert (out.elevation[added] >= C.MEMBRANE_LEVEL).all()
    assert np.array_equal(out.result[off_px], off.result[off_px])


def _reference_after_the_seeds(out, labels, spacing, membrane):
    """Everything after the seeds, rebuilt from public pieces: one
    watershed (skimage's defaults: 6-connected, no line, compactness 0)
    over the space (the empty voxels: no dark rule, no outline), every
    claim above n zeroed, the membrane step, the kept seam emptied, one
    piece per marker."""
    from skimage.segmentation import watershed

    sources = out.intermediates["markers"]
    n = out.n_seeds
    background = n + int(np.unique(labels[labels != 0]).size) + 1
    space = labels == 0
    claims = watershed(out.elevation, markers=sources, mask=space).astype(np.int32)
    claims[claims > n] = 0
    if membrane:
        claims = C.include_membrane(claims, elevation=out.elevation,
                                    existing=labels, space=space,
                                    spacing_um=spacing)
    claims[(sources > n) & (sources < background)] = 0
    res = np.where((claims >= 1) & (claims <= n) & space, claims, 0).astype(np.int32)
    return R.keep_marker_components(res, out.seeds)


@pytest.mark.parametrize("spacing", SPACINGS + [COARSE_Z])
@pytest.mark.parametrize("membrane", [False, True])
@pytest.mark.parametrize("keep", [(101, 106), (101, 105, 103)])
def test_everything_after_the_seeds_is_the_reference(spacing, membrane, keep):
    out, labels = _tissue_fill(spacing, membrane, outside="faces", keep=keep,
                               keep_intermediates=True)
    want, n_frag = _reference_after_the_seeds(out, labels, spacing, membrane)
    assert (out.intermediates["markers"] > out.n_seeds).any()
    assert np.array_equal(out.result, want)
    assert out.n_fragments_dropped == n_frag


@pytest.mark.parametrize("spacing", SPACINGS)
def test_the_membrane_step_changes_the_answer_beside_kept_objects(spacing):
    on, _l = _tissue_fill(spacing, True, keep=(101, 105, 103))
    off, _l = _tissue_fill(spacing, False, keep=(101, 105, 103))
    assert not np.array_equal(on.result, off.result)


@pytest.mark.parametrize("spacing", SPACINGS)
def test_a_detached_piece_is_dropped(spacing):
    """A bright voxel one voxel past a cell's edge (an empty voxel between)
    is handed to the cell by the membrane step but touches it by no face:
    it is dropped, and counted."""
    shape = (int(round(24.0 / spacing[0])) | 1, 49, 49)
    c = tuple(n // 2 for n in shape)
    labels = np.zeros(shape, np.int32)
    elev = np.zeros(shape, np.float32)
    first = _fill(labels, elev, _fill_spec(click_radius_um=0.5,
                                           include_membrane=True),
                  spacing=spacing, seed_points=[c])
    assert first.ok
    line = first.result[c[0], c[1], c[2]:]
    last = int(np.flatnonzero(line == 1).max())
    lone = (c[0], c[1], c[2] + last + 2)
    assert first.result[c[0], c[1], c[2] + last + 1] == 0
    elev[lone] = 0.5
    out = _fill(labels, elev, _fill_spec(click_radius_um=0.5,
                                         include_membrane=True),
                spacing=spacing, seed_points=[c])
    assert out.ok
    assert out.n_fragments_dropped >= 1
    assert out.result[lone] == 0
    _one_piece_each(out)


def test_the_cut_is_one_plain_watershed(monkeypatch):
    seen = []
    real = R.solve

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **k)

    monkeypatch.setattr(R, "solve", spy)
    labels, elev, pts = (np.zeros((12, 40, 40), np.int32),
                         np.zeros((12, 40, 40), np.float32),
                         [(6, 10, 10), (6, 30, 30)])
    assert _fill(labels, elev, _fill_spec(compactness=0.5), spacing=ISO05,
                 seed_points=pts).ok
    assert len(seen) == 1
    assert seen[0]["solver"] == "watershed"
    assert seen[0]["compactness"] == 0.0


# --------------------------------------------------------------------------- #
# the honeycomb and the sparse layer
# --------------------------------------------------------------------------- #
def _honeycomb(pitch_um, spacing, n_cells=3, depth_um=6.0):
    pitch = int(round(pitch_um / spacing[1]))
    n_xy = pitch * n_cells + 1
    shape = (_nz_for(spacing, depth_um), n_xy, n_xy)
    elev = np.zeros(shape, np.float32)
    for k in range(n_cells + 1):
        elev[:, k * pitch, :] = 1.0
        elev[:, :, k * pitch] = 1.0
    centres, interiors = {}, {}
    zc = shape[0] // 2
    for a in range(n_cells):
        for b in range(n_cells):
            centres[(a, b)] = (zc, a * pitch + pitch // 2, b * pitch + pitch // 2)
            m = np.zeros(shape, bool)
            m[:, a * pitch + 1:(a + 1) * pitch, b * pitch + 1:(b + 1) * pitch] = True
            interiors[(a, b)] = m
    return shape, elev, centres, interiors


@pytest.mark.parametrize("spacing", SPACINGS)
def test_honeycomb(spacing):
    shape, elev, centres, interiors = _honeycomb(8.0, spacing)
    unmarked = (1, 1)
    keys = [k for k in sorted(centres) if k != unmarked]
    out = _fill(np.zeros(shape, np.int32), elev, _fill_spec(), spacing=spacing,
                seed_points=[centres[k] for k in keys])
    assert out.ok, out.messages
    for i, key in enumerate(keys, start=1):
        assert (out.result[interiors[key]] == i).all(), key
    assert not (out.result[interiors[unmarked]] != 0).any()


def _own_fill(pitch_um, spacing):
    shape, elev, centres, interiors = _honeycomb(pitch_um, spacing)
    keys = sorted(centres)
    out = _fill(np.zeros(shape, np.int32), elev, _fill_spec(), spacing=spacing,
                seed_points=[centres[k] for k in keys])
    assert out.ok, out.messages
    return min(float((out.result[interiors[k]] == i).mean())
               for i, k in enumerate(keys, start=1))


@pytest.mark.parametrize("spacing", SPACINGS)
@pytest.mark.parametrize("pitch_um", [15.0, 20.0])
def test_sparse_layer(spacing, pitch_um, monkeypatch):
    assert _own_fill(pitch_um, spacing) >= 0.9
    # teeth: a fixed 6 um start would starve these cells
    monkeypatch.setattr(R, "BACKGROUND_NN_SHARE", 0.0)
    assert _own_fill(pitch_um, spacing) < 0.9


# --------------------------------------------------------------------------- #
# Stop
# --------------------------------------------------------------------------- #
def _stop_world(spacing=ISO05):
    shape = (_nz_for(spacing, 6.0), 40, 40)
    zc = shape[0] // 2
    return (np.zeros(shape, np.int32), np.zeros(shape, np.float32),
            [(zc, 10, 10), (zc, 30, 30)])


def _count_solve(monkeypatch):
    calls = []
    real = R.solve

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(R, "solve", counting)
    return calls


def test_stop_before_the_stage(monkeypatch):
    labels, elev, pts = _stop_world()
    before = labels.copy()
    solves = _count_solve(monkeypatch)
    ev = threading.Event()
    ev.set()
    with pytest.raises(ReshapeCancelled):
        _fill(labels, elev, _fill_spec(), spacing=ISO05, seed_points=pts,
              cancel_event=ev)
    ev2 = threading.Event()

    def progress(stage, i, n):
        if stage == "add_markers":
            ev2.set()

    with pytest.raises(ReshapeCancelled):
        _fill(labels, elev, _fill_spec(), spacing=ISO05, seed_points=pts,
              cancel_event=ev2, progress=progress)
    assert solves == []
    assert np.array_equal(labels, before)


def test_stop_between_slabs(monkeypatch):
    # a tall column, markers near its bottom: several slabs, each at
    # least two halos thick
    shape = (120, 30, 30)
    labels = np.zeros(shape, np.int32)
    elev = np.zeros(shape, np.float32)
    pts = [(2, 10, 10), (2, 20, 20)]
    solves = _count_solve(monkeypatch)
    monkeypatch.setattr(R, "BACKGROUND_SLAB_VOXELS", 1)
    ev = threading.Event()
    calls = []
    real = R.ndi.distance_transform_edt

    def edt(*a, **k):
        calls.append(1)
        ev.set()
        return real(*a, **k)

    monkeypatch.setattr(R.ndi, "distance_transform_edt", edt)
    with pytest.raises(ReshapeCancelled):
        _fill(labels, elev, _fill_spec(), spacing=ISO05, seed_points=pts,
              cancel_event=ev)
    assert len(calls) == 1
    assert solves == []
    # the helper on its own polls between slabs too
    calls.clear()
    ev.clear()
    space = np.ones(labels.shape, bool)
    new = _clicks(labels.shape, pts, ISO05)
    with pytest.raises(ReshapeCancelled):
        R._background_seeds(new, space, new.copy(), ISO05, cancel_event=ev,
                            slab_voxels=1)
    assert len(calls) == 1


def test_stop_raised_by_the_progress_callback_at_the_cut(monkeypatch):
    labels, elev, pts = _stop_world()
    solves = _count_solve(monkeypatch)

    def progress(stage, i, n):
        if stage == "add_cut":
            raise ReshapeCancelled()

    with pytest.raises(ReshapeCancelled):
        _fill(labels, elev, _fill_spec(), spacing=ISO05, seed_points=pts,
              progress=progress)
    assert solves == []
    # a Stop pressed while the cut is announced lands before the watershed
    ev = threading.Event()

    def progress_set(stage, i, n):
        if stage == "add_cut":
            ev.set()

    with pytest.raises(ReshapeCancelled):
        _fill(labels, elev, _fill_spec(), spacing=ISO05, seed_points=pts,
              progress=progress_set, cancel_event=ev)
    assert solves == []


def test_a_broken_progress_bar_does_not_stop_the_fill():
    labels, elev, pts = _stop_world()
    seen = []

    def progress(stage, i, n):
        seen.append((stage, i, n))
        raise RuntimeError("a broken bar")

    out = _fill(labels, elev, _fill_spec(), spacing=ISO05, seed_points=pts,
                progress=progress)
    assert out.ok
    assert [s[0] for s in seen] == list(R.REGION_STAGES)
    assert seen[-1] == ("add_cut", 4, 4)


# --------------------------------------------------------------------------- #
# Spread's growth space is the one the code made before the lift
# --------------------------------------------------------------------------- #
#: (case, spacing) -> digests of Spread's space, its dark mask after the
#: click clearance, and its result; produced by the code before the growth
#: space was shared (growth limit given explicitly, so the default moving
#: changes none of them).
SPREAD_BEFORE = {
    ("dark", (1.0, 1.0, 1.0)): {"space": "f14e872dd72da9b9",
                                "dark_raw": "1efdbb2c15c834f0",
                                "result": "b3e22abf2ef34feb"},
    ("dark", ANISO): {"space": "0afc44495e356c85",
                      "dark_raw": "e47486553dc46286",
                      "result": "9e8b5c0a3ec1efac"},
    ("outline", (1.0, 1.0, 1.0)): {"space": "d865716d4ffc84be",
                                   "dark_raw": "1efdbb2c15c834f0",
                                   "result": "4aa08315062d1bcb"},
    ("outline", ANISO): {"space": "d865716d4ffc84be",
                         "dark_raw": "e47486553dc46286",
                         "result": "849b008ef3ac75e5"},
    ("faces", (1.0, 1.0, 1.0)): {"space": "9a82af725a2ed874",
                                 "dark_raw": None,
                                 "result": "7fa4ae85e519860f"},
    ("faces", ANISO): {"space": "9a82af725a2ed874",
                       "dark_raw": None,
                       "result": "15bd9ece4909c093"},
    ("clicks", (1.0, 1.0, 1.0)): {"space": "349567b2edd9228d",
                                  "dark_raw": "98bbd627abbe0f19",
                                  "result": "0de1f301e1c219e1"},
    ("clicks", ANISO): {"space": "730ee78961055024",
                        "dark_raw": "fe40afd1f0151d79",
                        "result": "5e23c9a0afad9b91"},
}


def _spread_run(case, spacing, monkeypatch=None, *, growth="compete"):
    mem, nuc, cells = _tissue()
    ch = {"membrane": mem, "nuclei": nuc}
    labels = np.where(np.isin(cells, (101, 105)), cells, 0).astype(np.int32)
    params = {"wall_channel": "membrane", "ridge": False,
              "outside": "faces" if case == "faces" else "faces+dark",
              "growth": growth, "growth_limit": 14.0, "wall_cost": 40.0,
              "include_membrane": True}
    kw = {"seed_layer": cells}
    seeds = "labels"
    outline = None
    if case == "outline":
        params["outline"] = "cleared"
        outline = np.zeros(cells.shape, bool)
        outline[:, :20, :] = True
    if case == "clicks":
        seeds = "paint"
        kw = {"seed_points": [(6, 6, 5), (6, 30, 15), (6, 28, 33)]}
    spec = ReshapeSpec(target="region", seeds=seeds, walls="membrane",
                       params=params)
    captured = {}
    if monkeypatch is not None:
        real = R._compete.kept_competitors

        def capture(markers, existing, space):
            captured["space"] = np.array(space, copy=True)
            return real(markers, existing, space)

        monkeypatch.setattr(R._compete, "kept_competitors", capture)
    out = R.reshape_empty_region(
        labels, {"membrane": mem}, spec, roi=_box(labels.shape),
        dark_channels=ch,
        dark_threshold=R.patch_dark_threshold(ch, spacing=spacing, stride=1),
        spacing=spacing, outline=outline, keep_intermediates=True, **kw)
    assert out.ok, out.messages
    return out, captured


@pytest.mark.parametrize("case, spacing", sorted(SPREAD_BEFORE, key=repr))
def test_growth_space_unchanged(case, spacing, monkeypatch):
    want = SPREAD_BEFORE[(case, spacing)]
    assert want is not None, "pin missing"
    returned = {}
    real_space = R._growth_space

    def wrap(*a, **k):
        space, dark_raw = real_space(*a, **k)
        returned["space"] = space.copy()
        returned["dark_raw"] = None if dark_raw is None else dark_raw.copy()
        return space, dark_raw

    monkeypatch.setattr(R, "_growth_space", wrap)
    out, captured = _spread_run(case, spacing, monkeypatch)
    got = {"space": _digest(captured["space"]),
           "dark_raw": _digest(out.intermediates["dark_raw"]),
           "result": _digest(out.result)}
    assert got == want
    # the shared helper hands back exactly those two
    assert _digest(returned["space"]) == want["space"]
    assert _digest(returned["dark_raw"]) == want["dark_raw"]


@pytest.mark.parametrize("case", ["dark", "outline", "faces", "clicks"])
def test_the_fill_grows_in_spreads_space(case, monkeypatch):
    """Same inputs, same space: the fill and Spread share the helper."""
    spread, cap_s = _spread_run(case, ANISO, monkeypatch)
    monkeypatch.undo()
    fill, cap_f = _spread_run(case, ANISO, monkeypatch, growth=FILL)
    assert np.array_equal(cap_s["space"], cap_f["space"])
    assert _digest(fill.intermediates["dark_raw"]) == _digest(
        spread.intermediates["dark_raw"])
    assert not (fill.result[~cap_f["space"]] != 0).any()


# --------------------------------------------------------------------------- #
# every micron uses the spacing
# --------------------------------------------------------------------------- #
def test_the_radii_follow_the_spacing():
    shape = (20, 10, 70)
    new = np.zeros(shape, np.int32)
    new[10, 5, 5] = 1
    new[10, 5, 35] = 2                     # 30 voxels apart along x
    r = R._background_radii(new, (0.5, 0.5, 0.5))
    assert r[1] == pytest.approx(0.75 * 15.0) and r[2] == pytest.approx(11.25)
    r = R._background_radii(new, (0.5, 0.5, 0.25))
    assert r[1] == pytest.approx(6.0)      # 7.5 um apart: the floor
    r = R._background_radii(new, (0.5, 0.5, 1.0))
    assert r[1] == pytest.approx(22.5)
    along_z = np.zeros(shape, np.int32)
    along_z[2, 5, 5] = 1
    along_z[18, 5, 5] = 2                  # 16 slices apart
    assert R._background_radii(along_z, (1.0, 0.5, 0.5))[1] == pytest.approx(12.0)
    assert R._background_radii(along_z, (0.5, 0.5, 0.5))[1] == pytest.approx(6.0)
    # the centroid is the marker's mean voxel, in microns
    blob = np.zeros(shape, np.int32)
    blob[10, 5, 0:3] = 1                   # centroid x = 1
    blob[10, 5, 31] = 2                    # 30 voxels from it
    assert R._background_radii(blob, (0.5, 0.5, 0.5))[1] == pytest.approx(11.25)


@pytest.mark.parametrize("bad", [(0.0, 0.5, 0.5), (0.5, -1.0, 0.5),
                                 (0.5, float("nan"), 0.5), (0.5, 0.5)])
def test_an_unusable_spacing_is_refused(bad):
    new = np.zeros((4, 4, 4), np.int32)
    new[1, 1, 1] = 1
    with pytest.raises(ValueError, match="spacing"):
        R._background_seeds(new, np.ones(new.shape, bool), new, bad)


# --------------------------------------------------------------------------- #
# the seed reads no channel
# --------------------------------------------------------------------------- #
def test_seed_reads_no_channel(monkeypatch):
    params = inspect.signature(R._background_seeds).parameters
    assert not {"images", "channels", "dark_channels", "elevation", "elev",
                "seed_layer"} & set(params)
    shape = (12, 40, 40)
    labels = np.zeros(shape, np.int32)
    outline = np.zeros(shape, bool)
    outline[:, 2:38, 2:38] = True          # the space is fixed by it
    pts = [(6, 10, 10), (6, 30, 12), (6, 20, 32)]
    rng = np.random.default_rng(3)
    one = {"a": rng.random(shape).astype(np.float32)}
    two = dict(one, b=rng.random(shape).astype(np.float32) * 3)
    seen = []
    real = R._background_seeds

    def wrap(*a, **k):
        out = real(*a, **k)
        seen.append(out.copy())
        return out

    monkeypatch.setattr(R, "_background_seeds", wrap)
    for chans in (one, two):
        out = _fill(labels, rng.random(shape).astype(np.float32) * 0.1,
                    _fill_spec(outline="cleared"), spacing=ISO05,
                    seed_points=pts, outline=outline, dark_channels=chans,
                    dark_threshold=R.patch_dark_threshold(chans, spacing=ISO05,
                                                          stride=1))
        assert out.ok, out.messages
    assert len(seen) == 2
    assert np.array_equal(seen[0], seen[1])
    assert seen[0].any()


# --------------------------------------------------------------------------- #
# the id guard, dense markers, the lone marker
# --------------------------------------------------------------------------- #
def test_the_background_id_follows_the_kept_competitors():
    assert R._background_id(5, 3) == 9
    assert R._background_id(5, 0) == 6
    assert R._background_id(2 ** 31 - 3, 1) == 2 ** 31 - 1
    with pytest.raises(ValueError, match="too many ids"):
        R._background_id(2 ** 31 - 2, 1)


@pytest.mark.parametrize("spacing", SPACINGS)
def test_no_seed_when_markers_are_dense(spacing):
    shape = (_nz_for(spacing, 6.0), 20, 20)
    step = int(round(2.5 / spacing[1]))
    zc = shape[0] // 2
    pts = [(zc, y, x) for y in range(2, 20, step) for x in range(2, 20, step)]
    labels = np.zeros(shape, np.int32)
    new = _clicks(shape, pts, spacing)
    space = np.ones(shape, bool)
    assert not R._background_seeds(new, space, new.copy(), spacing).any()
    out = _fill(labels, np.zeros(shape, np.float32), _fill_spec(),
                spacing=spacing, seed_points=pts)
    assert out.ok
    assert (out.result != 0).all()


@pytest.mark.parametrize("spacing", SPACINGS)
def test_a_lone_marker_takes_six_microns(spacing):
    shape = (int(round(20.0 / spacing[0])), 40, 40)
    new = np.zeros(shape, np.int32)
    c = (shape[0] // 2, 20, 20)
    new[c] = 1
    r = R._background_radii(new, spacing)
    assert r[1] == pytest.approx(6.0)
    seeds = R._background_seeds(new, np.ones(shape, bool), new.copy(), spacing)
    dist = ndi.distance_transform_edt(new == 0, sampling=spacing)
    assert np.array_equal(seeds, dist > 6.0)


# --------------------------------------------------------------------------- #
# the whole-patch size ceiling
# --------------------------------------------------------------------------- #
class _Unreadable:
    """A volume that must never be read."""

    def __init__(self, shape):
        self.shape = shape
        self.dtype = np.dtype(np.int32)

    def __getitem__(self, key):
        raise AssertionError("read before the size check")

    def __array__(self, *a, **k):
        raise AssertionError("read before the size check")


def test_the_whole_fill_ceiling_refuses_before_any_read(monkeypatch):
    shape = (6, 20, 20)
    n = 6 * 20 * 20
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", n - 1)
    spec = _fill_spec(whole_patch=True)
    out = R.reshape_empty_region(
        _Unreadable(shape), {}, spec, roi=_box(shape),
        seed_points=[(3, 10, 10)], spacing=ISO05,
        elevation=np.zeros(shape, np.float32))
    assert out.status == "refused"
    assert out.refusal_reason == "too-big-whole"
    assert out.messages == [R.too_big_whole_sentence(n - 1)]
    seeds = R.find_region_seeds(_Unreadable(shape), {}, spec, roi=_box(shape),
                                seed_points=[(3, 10, 10)], spacing=ISO05)
    assert seeds.status == "refused" and seeds.refusal_reason == "too-big-whole"
    # at the ceiling it runs; Spread's whole patch has no such ceiling
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", n)
    labels = np.zeros(shape, np.int32)
    assert _fill(labels, np.zeros(shape, np.float32), spec, spacing=ISO05,
                 seed_points=[(3, 10, 10)]).ok
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", 10)
    spread = _fill(labels, np.zeros(shape, np.float32),
                   _fill_spec(growth="compete", whole_patch=True),
                   spacing=ISO05, seed_points=[(3, 10, 10)])
    assert spread.ok


@pytest.mark.parametrize("bad", [(0.0, 0.5, 0.5), (0.5, float("nan"), 0.5),
                                 (0.5, 0.5)])
def test_an_unusable_spacing_is_refused_before_any_stage(bad):
    shape = (6, 20, 20)
    seen = []
    with pytest.raises(ValueError, match="spacing"):
        R.reshape_empty_region(
            _Unreadable(shape), {}, _fill_spec(), roi=_box(shape),
            seed_points=[(3, 10, 10)], spacing=bad,
            elevation=np.zeros(shape, np.float32),
            progress=lambda *a: seen.append(a))
    assert seen == []


def test_the_whole_fill_sentence():
    assert "100 million" in R.too_big_whole_sentence(100_000_000)
    assert R.ENGINE_REFUSALS["too-big-whole"] == R.too_big_whole_sentence(
        R.WHOLE_FILL_MAX_VOXELS)
    assert "(Nothing was changed.)" in R.too_big_whole_sentence(10)


def test_a_box_fill_keeps_the_box_budget():
    shape = (6, 20, 20)
    out = _fill(np.zeros(shape, np.int32), np.zeros(shape, np.float32),
                _fill_spec(box_budget=100), spacing=ISO05,
                seed_points=[(3, 10, 10)])
    assert out.status == "refused" and out.refusal_reason == "too-big"


# --------------------------------------------------------------------------- #
# the dispatch: stages, the sliver drop, the box-info count
# --------------------------------------------------------------------------- #
def test_the_fill_reports_the_floods_stages_and_no_tiles():
    labels, elev, pts = _stop_world()
    seen = []
    out = _fill(labels, elev, _fill_spec(), spacing=ISO05, seed_points=pts,
                progress=lambda *a: seen.append(a))
    assert out.ok
    assert [a[0] for a in seen] == list(R.REGION_STAGES)
    assert out.n_tiles == 0
    assert out.result.dtype == np.int32


def _sliver_world():
    shape = (12, 30, 30)
    labels = np.zeros(shape, np.int32)
    labels[:, 10:20, 10:20] = 5
    run = np.zeros(shape, np.int32)
    run[:, 10:20, 4:22] = 9                # its core lies mostly over kept
    run[:, 2:8, 2:8] = 11                  # a plain marker
    return labels, run


@pytest.mark.parametrize("growth, dropped", [("flood", 0), ("compete", 1),
                                             (FILL, 1)])
def test_the_sliver_drop_runs_for_the_fill_and_the_box_count_agrees(
        growth, dropped):
    labels, run = _sliver_world()
    spec = _fill_spec(seeds="labels", growth=growth)
    out = R.reshape_empty_region(
        labels, {}, spec, roi=_box(labels.shape), seed_layer=run,
        spacing=ISO05, elevation=np.zeros(labels.shape, np.float32))
    assert out.ok, out.messages
    assert out.n_slivers == dropped
    info = R.find_region_seeds(labels, {}, spec, roi=_box(labels.shape),
                               seed_layer=run, spacing=ISO05)
    assert info.n_in_region == out.n_seeds == 2 - dropped

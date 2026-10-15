"""The Add region target -- ``reshape/region.py``.

"Reshape on empty space": inside a box, markers stop each other, walls come
from the membrane, and new objects are written only where the volume is 0.
Every test that touches the leak or the dark rule runs with BOTH
``outside="faces+dark"`` and ``outside="faces"``.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from scipy import ndimage as ndi

from packages.epicycle_annotator.reshape import ReshapeCancelled, ReshapeSpec
from packages.epicycle_annotator.reshape import region as R
from packages.epicycle_annotator.reshape.engine import find_seeds, reshape_region

ISO = (1.0, 1.0, 1.0)
OUTSIDE = ("faces+dark", "faces")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _spec(seeds="labels", outside="faces+dark", **params):
    base = {"wall_channel": "membrane", "ridge": False, "outside": outside}
    base.update(params)
    return ReshapeSpec(target="region", seeds=seeds, walls="membrane",
                       params=base)


def _box(shape):
    return tuple(slice(0, int(n)) for n in shape)


def _tissue(shape=(12, 40, 40), *, open_row=True, seed=0):
    """Columns of cells in y < 24, open space (dark) in y >= 24.

    Membrane walls at x = 10, 20, 30 and y = 12 (full z).  With
    ``open_row`` the tissue has NO wall at its y = 24 edge, so the second
    row of cells is open to the space -- the case the dark rule exists for.
    Nuclei are bright blobs at the cell centres.  Returns
    ``(membrane, nuclei, cells)`` where ``cells`` labels each cell interior
    (ids 101, 102, ...) in the tissue.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    mem = np.full(shape, 0.02, np.float32)
    nuc = np.full(shape, 0.02, np.float32)
    tissue = np.zeros(shape, bool)
    tissue[:, :24, :] = True
    # dim membrane inside the cells AND in the open space, as on real data:
    # both read the same on the wall channel, so only the markers
    # (and the dark rule) decide who takes the space.  The cytoplasm is
    # lifted on the nuclei channel, which keeps the interiors out of the
    # dark mask.
    mem[:] = 0.05
    nuc[tissue] = 0.3
    for x in (10, 20, 30):
        mem[:, :24, x] = 1.0
    mem[:, 12, :] = 1.0
    if not open_row:
        mem[:, 23, :] = 1.0
    cells = np.zeros(shape, np.int32)
    cid = 101
    z, y, x = np.indices(shape)
    for y0, y1 in ((0, 12), (13, 24)):
        for x0, x1 in ((0, 10), (11, 20), (21, 30), (31, 40)):
            cells[:, y0:y1, x0:x1] = cid
            cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
            blob = (z - nz / 2.0) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= 9
            nuc[blob] = 1.0
            cid += 1
    cells[:, :, (10, 20, 30)] = 0
    cells[:, 12, :] = 0
    mem += rng.normal(0, 0.01, shape).astype(np.float32)
    nuc += rng.normal(0, 0.01, shape).astype(np.float32)
    return np.clip(mem, 0, None), np.clip(nuc, 0, None), cells


def _threshold(channels, spacing=ISO):
    return R.patch_dark_threshold(channels, spacing=spacing, stride=1)


def _run(spec, labels, channels, *, seed_layer=None, seed_points=None,
         elevation=None, dark_channels=None, spacing=ISO, **kw):
    dark = channels if dark_channels is None else dark_channels
    return R.reshape_empty_region(
        labels, {"membrane": channels["membrane"]}, spec,
        roi=_box(labels.shape), dark_channels=dark, seed_layer=seed_layer,
        seed_points=seed_points, dark_threshold=_threshold(dark, spacing),
        spacing=spacing, elevation=elevation, **kw)


def _run_cores(cells):
    """A 'run' guide: the cells themselves, as another run would hold them."""
    return cells.copy()


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def test_reshape_region_dispatches_the_region_target():
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    ch = {"membrane": mem, "nuclei": nuc}
    out = reshape_region(labels, {"membrane": mem}, _spec(), roi=_box(labels.shape),
                         seed_layer=cells, dark_channels=ch,
                         dark_threshold=_threshold(ch), spacing=ISO)
    assert out.ok
    assert out.n_seeds == 8


def test_find_seeds_dispatches_the_region_target_without_a_target_id():
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    ch = {"membrane": mem, "nuclei": nuc}
    rep = find_seeds(labels, {"membrane": mem}, _spec(), roi=_box(labels.shape),
                     seed_layer=cells, dark_channels=ch,
                     dark_threshold=_threshold(ch), spacing=ISO)
    assert rep.ok
    assert rep.n_in_region == 8


def test_the_region_target_needs_a_roi():
    mem, nuc, cells = _tissue()
    with pytest.raises(ValueError, match="roi"):
        R.reshape_empty_region(np.zeros_like(cells), {"membrane": mem}, _spec(),
                               roi=None, seed_layer=cells, spacing=ISO,
                               dark_channels={"membrane": mem},
                               dark_threshold=_threshold({"membrane": mem}))


@pytest.mark.parametrize("seeds", ["peaks", "distance", "hmin"])
def test_marker_sources_the_region_target_has_not_got_are_refused_loudly(seeds):
    """Since the B1 fix pass (audit L-A3) the refusal comes from the spec
    itself, so it is loud at construction, not only at run time."""
    mem, nuc, cells = _tissue()
    with pytest.raises(ValueError, match="region"):
        spec = ReshapeSpec(target="region", seeds=seeds, walls="membrane",
                           params={"wall_channel": "membrane", "outside": "faces"})
        R.reshape_empty_region(np.zeros_like(cells), {"membrane": mem}, spec,
                               roi=_box(cells.shape), spacing=ISO)


def test_faces_dark_without_a_dark_threshold_is_a_value_error():
    mem, nuc, cells = _tissue()
    with pytest.raises(ValueError, match="dark_threshold"):
        R.reshape_empty_region(np.zeros_like(cells), {"membrane": mem}, _spec(),
                               roi=_box(cells.shape), seed_layer=cells,
                               dark_channels={"membrane": mem}, spacing=ISO)


# --------------------------------------------------------------------------- #
# the leak, with and without the dark rule
# --------------------------------------------------------------------------- #
def _leak(out, labels):
    res = out.result
    total = int((res > 0).sum())
    return float((res[:, 24:, :] > 0).sum()) / max(total, 1)


def test_the_dark_rule_stops_the_leak_into_open_space():
    # 24 deep in z, so the z faces do not do the dark rule's job for it
    mem, nuc, cells = _tissue(shape=(24, 40, 40), open_row=True)
    labels = np.zeros_like(cells)
    ch = {"membrane": mem, "nuclei": nuc}
    dark = _run(_spec(outside="faces+dark"), labels, ch, seed_layer=cells)
    faces = _run(_spec(outside="faces"), labels, ch, seed_layer=cells)
    assert dark.ok and faces.ok
    leak_dark, leak_faces = _leak(dark, labels), _leak(faces, labels)
    assert leak_dark < 0.05, leak_dark
    assert leak_faces > leak_dark + 0.05, (leak_faces, leak_dark)


@pytest.mark.parametrize("outside", OUTSIDE)
def test_a_walled_tissue_fills_its_cells_either_way(outside):
    mem, nuc, cells = _tissue(open_row=False)
    labels = np.zeros_like(cells)
    out = _run(_spec(outside=outside), labels, {"membrane": mem, "nuclei": nuc},
               seed_layer=cells)
    assert out.ok
    assert out.n_pieces == 8
    assert _leak(out, labels) < 0.05


# --------------------------------------------------------------------------- #
# ids, kept objects, writes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("outside", OUTSIDE)
def test_only_ids_one_to_n_and_only_on_empty_space(outside):
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    labels[:, 0:12, 0:10] = 99           # one cell already kept
    out = _run(_spec(outside=outside), labels, {"membrane": mem, "nuclei": nuc},
               seed_layer=cells)
    assert out.ok
    ids = set(int(v) for v in np.unique(out.result)) - {0}
    assert ids == set(range(1, out.n_seeds + 1))
    assert not (out.result[labels != 0]).any()
    assert 99 not in ids


@pytest.mark.parametrize("outside", OUTSIDE)
def test_run_ids_are_relabelled_in_ascending_order(outside):
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    run = np.where(cells > 0, cells * 10 + 5000, 0).astype(np.int32)
    out = _run(_spec(outside=outside), labels, {"membrane": mem, "nuclei": nuc},
               seed_layer=run)
    assert out.ok
    ids = set(int(v) for v in np.unique(out.result)) - {0}
    assert ids == set(range(1, 9))
    assert not (ids & set(int(v) for v in np.unique(run)))
    # ascending original id -> 1..N: marker 1 sits in the cell with the
    # lowest run id (101 -> 6010)
    first = np.argwhere(out.seeds == 1)[0]
    assert run[tuple(first)] == 6010


@pytest.mark.parametrize("outside", OUTSIDE)
def test_a_neighbour_core_stops_the_fill_and_never_appears(outside):
    shape = (10, 20, 40)
    labels = np.zeros(shape, np.int32)
    labels[:, :, 18:23] = 50              # a thick kept wall across the box
    elev = np.zeros(shape, np.float32)    # flat: nothing but markers decide
    guide = np.zeros(shape, np.int32)
    guide[3:7, 8:12, 4:8] = 9             # one run object, left of the wall
    bright = {"membrane": np.ones(shape, np.float32)}
    out = _run(_spec(outside=outside, core_erode_um=1.0), labels, bright,
               seed_layer=guide, elevation=elev, keep_intermediates=True)
    assert out.ok
    assert 50 not in set(np.unique(out.result))
    assert not (out.result[:, :, 23:] > 0).any()
    assert not (out.result[labels != 0]).any()
    # the wall's CORE is the marker that did it (id N + 2), so the fill
    # never crossed the wall and no piece had to be cut off afterwards
    markers = out.intermediates["markers"]
    assert (markers[:, :, 19:22] == out.n_seeds + 2).all()
    assert out.n_fragments_dropped == 0


@pytest.mark.parametrize("outside", OUTSIDE)
def test_an_object_its_erosion_empties_is_dropped_and_counted(outside):
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    run = cells.copy()
    run[run == 108] = 0
    run[6, 18, 35] = 108                  # a one-voxel object: no core at 1 um
    out = _run(_spec(outside=outside, core_erode_um=1.0), labels,
               {"membrane": mem, "nuclei": nuc}, seed_layer=run)
    assert out.ok
    assert out.n_dropped_thin == 1
    assert out.n_seeds == 7
    # never revived: nothing is written on the thin object's own voxel as
    # a marker of its own (8 ids would mean a fallback revived it)
    assert set(np.unique(out.result)) - {0} == set(range(1, 8))


@pytest.mark.parametrize("outside", OUTSIDE)
def test_a_thin_object_wholly_over_kept_space_is_not_counted_thin(outside):
    """B2.0 addendum: ``n_dropped_thin`` counts only run
    objects that had at least one voxel on EMPTY space before the erosion.
    A thin run object sitting wholly over a kept object is not "cut too thin
    to use" -- it was never going to be a marker."""
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    labels[cells == 101] = 7                 # a kept object
    run = cells.copy()
    run[run == 108] = 0
    run[6, 18, 35] = 108                     # thin, on empty space -> counted
    run[run == 101] = 0
    run[6, 5, 5] = 101                       # thin, wholly over kept -> not
    out = _run(_spec(outside=outside, core_erode_um=1.0), labels,
               {"membrane": mem, "nuclei": nuc}, seed_layer=run)
    assert out.ok
    assert out.n_dropped_thin == 1


@pytest.mark.parametrize("outside", OUTSIDE)
def test_an_object_wholly_over_kept_space_is_counted_over_kept(outside):
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    labels[cells == 101] = 7
    out = _run(_spec(outside=outside), labels, {"membrane": mem, "nuclei": nuc},
               seed_layer=cells)
    assert out.ok
    assert out.n_markers_over_kept == 1
    assert out.n_seeds == 7


@pytest.mark.parametrize("outside", OUTSIDE)
def test_a_marker_beside_a_thin_kept_sheet_keeps_only_its_own_piece(outside):
    """Invariant 5: the basin floods through a 1-voxel kept sheet (no core);
    zeroing the kept voxels cuts it in two; only the marker's piece stays."""
    shape = (10, 20, 41)
    labels = np.zeros(shape, np.int32)
    labels[:, :, 20] = 9                   # 1 voxel thick: its core is empty
    elev = np.ones(shape, np.float32)
    elev[2:8, 2:18, 2:20] = 0.0            # left compartment, low
    elev[2:8, 2:18, 21:39] = 0.0           # right compartment, low
    elev[2:8, 2:18, 20] = 0.0              # the sheet is low where it
    #                                        joins the two compartments
    guide = np.zeros(shape, np.int32)
    guide[3:7, 7:13, 6:14] = 3             # thick enough to keep a core
    bright = {"membrane": np.ones(shape, np.float32)}
    out = _run(_spec(outside=outside, core_erode_um=1.0), labels, bright,
               seed_layer=guide, elevation=elev)
    assert out.ok
    assert out.n_fragments_dropped == 1
    assert not (out.result[:, :, 21:] > 0).any()
    lab, n = ndi.label(out.result == 1)
    assert n == 1


# --------------------------------------------------------------------------- #
# nuclei masks and the sliver filter
# --------------------------------------------------------------------------- #
def _sliver_case(spacing_x):
    """A kept plane at x = 10 and a one-voxel-thick nuclei piece against it
    (x = 11), plus one fat piece far away.  The sliver reaches exactly one
    voxel (= ``spacing_x`` um) from the kept voxels."""
    shape = (8, 16, 30)
    labels = np.zeros(shape, np.int32)
    labels[:, :, 10] = 4
    pieces = np.zeros(shape, np.int32)
    pieces[2:6, 4:12, 11] = 900           # the sliver
    pieces[2:6, 4:12, 20:26] = 901        # a real nucleus
    return labels, pieces, (spacing_x, spacing_x, spacing_x)


@pytest.mark.parametrize("spacing_x,kept", [(0.9, False), (1.0, True), (1.1, True)])
def test_the_sliver_filter_boundary(spacing_x, kept):
    labels, pieces, spacing = _sliver_case(spacing_x)
    out, n = R.drop_slivers(pieces, labels, min_depth_um=R.NUCLEI_SLIVER_UM,
                            spacing=spacing)
    assert (900 in set(np.unique(out))) is kept
    assert 901 in set(np.unique(out))
    assert n == (0 if kept else 1)


def test_the_sliver_constant_is_one_micron():
    assert R.NUCLEI_SLIVER_UM == 1.0


@pytest.mark.parametrize("outside", OUTSIDE)
def test_nuclei_masks_skip_the_sliver_in_the_engine(outside):
    labels, pieces, spacing = _sliver_case(0.9)
    bright = {"membrane": np.ones(labels.shape, np.float32)}
    out = _run(_spec(seeds="nuclei_masks", outside=outside), labels, bright,
               seed_layer=pieces, spacing=spacing,
               elevation=np.zeros(labels.shape, np.float32))
    assert out.ok
    assert out.n_nuclei_slivers == 1
    assert out.n_seeds == 1


def test_drop_slivers_with_no_kept_voxel_keeps_everything():
    pieces = np.zeros((6, 6, 6), np.int32)
    pieces[1:3, 1:3, 1:3] = 5
    out, n = R.drop_slivers(pieces, np.zeros_like(pieces), min_depth_um=1.0,
                            spacing=ISO)
    assert n == 0 and (out == pieces).all()


# --------------------------------------------------------------------------- #
# the dark rule
# --------------------------------------------------------------------------- #
def _dark_case():
    """A 20x30x30 box, the half x >= 15 dark in every channel."""
    shape = (20, 30, 30)
    img = np.full(shape, 0.8, np.float32)
    img[:, :, 15:] = 0.0
    labels = np.zeros(shape, np.int32)
    return shape, {"membrane": img}, labels


@pytest.mark.parametrize("outside", OUTSIDE)
def test_the_dark_test_runs_after_the_kept_voxel_clip(outside):
    """One run object 40 voxels long: its first 20 over a kept object (on
    the bright side), its last 20 on empty space of which ~15 are dark.
    Before the clip ~37 % dark (kept); after it ~75 % (dropped)."""
    shape = (20, 30, 60)
    img = np.full(shape, 0.8, np.float32)
    img[:, :, 30:] = 0.0
    ch = {"membrane": img}
    labels = np.zeros(shape, np.int32)
    guide = np.zeros(shape, np.int32)
    guide[8:12, 12:17, 5:45] = 77
    labels[8:12, 12:17, 5:25] = 5
    guide[2:6, 2:6, 2:6] = 78              # a healthy marker, bright, empty
    spec = _spec(outside=outside, core_erode_um=0.0)
    out = _run(spec, labels, ch, seed_layer=guide,
               elevation=np.zeros(shape, np.float32))
    assert out.ok
    if outside == "faces+dark":
        assert out.n_dropped_dark == 1
        assert out.n_seeds == 1
    else:
        assert out.n_dropped_dark == 0
        assert out.n_seeds == 2


@pytest.mark.parametrize("outside", OUTSIDE)
def test_a_click_in_the_dark_survives(outside):
    shape, ch, labels = _dark_case()
    spec = _spec(seeds="paint", outside=outside)
    out = _run(spec, labels, ch, seed_points=[(10, 15, 24)])
    assert out.ok
    assert out.n_seeds == 1
    assert out.n_dropped_dark == 0
    assert out.result[10, 15, 24] == 1


@pytest.mark.parametrize("outside", OUTSIDE)
def test_every_automatic_marker_in_the_dark_refuses_with_its_constant(outside):
    shape, ch, labels = _dark_case()
    guide = np.zeros(shape, np.int32)
    guide[8:12, 12:18, 20:26] = 3          # wholly in the dark half
    out = _run(_spec(outside=outside, core_erode_um=0.0), labels, ch,
               seed_layer=guide)
    if outside == "faces+dark":
        assert out.status == "refused"
        assert out.refusal_reason == "no-markers"
        assert R.ADD_ENGINE_NO_MARKERS in out.messages
        assert out.result is None
    else:
        assert out.ok and out.n_seeds == 1


@pytest.mark.parametrize("outside", OUTSIDE)
def test_two_dark_channels_fill_a_nucleus_bright_interior(outside):
    """A membrane-only dark mask calls a cell's dim interior dark; with the
    nuclei channel in ``dark_channels`` the interior is not dark and the
    run marker survives (grill E-M4)."""
    shape = (16, 30, 30)
    mem = np.full(shape, 0.02, np.float32)
    nuc = np.full(shape, 0.02, np.float32)
    mem[:, :, 0:14] = 0.8                  # bright tissue on the left
    # the cell: x 16..26 -- dim membrane inside, bright nucleus
    mem[:, 4:26, 15] = mem[:, 4:26, 27] = 1.0
    mem[:, 4, 15:28] = mem[:, 25, 15:28] = 1.0
    nuc[2:14, 6:24, 17:26] = 1.0
    labels = np.zeros(shape, np.int32)
    guide = np.zeros(shape, np.int32)
    guide[4:12, 8:22, 18:25] = 11
    spec = _spec(outside=outside, core_erode_um=1.0)
    both = {"membrane": mem, "nuclei": nuc}
    only = {"membrane": mem}
    out_both = _run(spec, labels, both, seed_layer=guide)
    out_only = _run(spec, labels, both, seed_layer=guide, dark_channels=only)
    assert out_both.ok and out_both.n_seeds == 1
    assert (out_both.result[4:12, 8:22, 18:25] == 1).all()
    if outside == "faces+dark":
        assert out_only.status == "refused"
        assert out_only.refusal_reason == "no-markers"
    else:
        assert out_only.ok and out_only.n_seeds == 1


@pytest.mark.parametrize("outside", OUTSIDE)
def test_the_dark_threshold_is_carried_on_the_result(outside):
    mem, nuc, cells = _tissue()
    ch = {"membrane": mem, "nuclei": nuc}
    out = _run(_spec(outside=outside), np.zeros_like(cells), ch, seed_layer=cells)
    assert out.dark_threshold == _threshold(ch)


def test_patch_dark_threshold_holds_across_strides():
    rng = np.random.default_rng(3)
    shape = (48, 96, 96)
    field_ = ndi.gaussian_filter(rng.normal(size=shape), 6).astype(np.float32)
    field_ = (field_ - field_.min()) / (field_.max() - field_.min())
    field_[:, :, 60:] *= 0.1              # a dark region
    nuc = np.roll(field_, 17, axis=1)
    ch = {"membrane": field_, "nuclei": nuc}
    spacing = (1.0, 0.5, 0.5)
    ts = [R.patch_dark_threshold(ch, spacing=spacing, stride=s)["t"]
          for s in (1, 2, 4)]
    for t in ts[1:]:
        assert abs(t - ts[0]) <= 0.05 * ts[0], ts
    out = R.patch_dark_threshold(ch, spacing=spacing, stride=2)
    assert set(out) == {"t", "lo", "hi"}
    assert set(out["lo"]) == set(out["hi"]) == {"membrane", "nuclei"}


def test_dark_mask_uses_the_patch_normalisation_and_the_factor():
    img = np.zeros((10, 10, 10), np.float32)
    img[:, :, 5:] = 1.0
    thr = {"t": 0.4, "lo": {"a": 0.0}, "hi": {"a": 1.0}}
    d = R.dark_mask({"a": img}, dark_threshold=thr, factor=0.5, spacing=ISO,
                    blur_um=0.01)
    assert d[:, :, :5].all() and not d[:, :, 5:].any()
    with pytest.raises(ValueError):
        R.dark_mask({"b": img}, dark_threshold=thr, factor=0.5, spacing=ISO)


def test_the_dark_constants():
    assert R.DARK_BLUR_UM == 2.0
    assert R.DARK_SAMPLE_VOXELS == 1_000_000


# --------------------------------------------------------------------------- #
# the notes
# --------------------------------------------------------------------------- #
def _two_halves(wall_value):
    rng = np.random.default_rng(1)
    shape = (8, 10, 20)
    result = np.zeros(shape, np.int32)
    result[:, :, :10] = 1
    result[:, :, 10:] = 2
    elev = rng.uniform(0.3, 0.7, shape).astype(np.float32)
    elev[:, :, 9:11] = wall_value
    return result, elev


def test_a_faint_wall_is_flagged():
    result, elev = _two_halves(0.05)
    pairs = R.faint_wall_pairs(result, elev, percentile=90.0)
    assert [(a, b) for a, b, _m in pairs] == [(1, 2)]
    assert pairs[0][2] == pytest.approx(0.05)


def test_a_strong_wall_is_not_flagged():
    result, elev = _two_halves(1.0)
    assert R.faint_wall_pairs(result, elev, percentile=90.0) == []


def test_faint_pairs_are_sorted_weakest_first():
    shape = (6, 6, 30)
    result = np.zeros(shape, np.int32)
    result[:, :, 0:10] = 1
    result[:, :, 10:20] = 2
    result[:, :, 20:30] = 3
    elev = np.full(shape, 0.5, np.float32)
    elev[:, :, 3:7] = 0.9                  # lift the percentile
    elev[:, :, 9:11] = 0.2                 # 1|2 fainter
    elev[:, :, 19:21] = 0.3                # 2|3 faint
    pairs = R.faint_wall_pairs(result, elev, percentile=90.0)
    assert [(a, b) for a, b, _m in pairs] == [(1, 2), (2, 3)]


def test_unfilled_fraction_on_a_known_partition():
    existing = np.zeros((4, 4, 10), np.int32)
    existing[:, :, 0:2] = 5                     # kept: never counted
    dark = np.zeros(existing.shape, bool)
    dark[:, :, 8:] = True                       # dark: never counted
    result = np.zeros(existing.shape, np.int32)
    result[:, :, 2:5] = 1                       # 3 of the 6 tissue columns
    assert R.unfilled_fraction(result, existing, dark) == pytest.approx(0.5)
    assert R.unfilled_fraction(result, existing, None) is None
    assert R.unfilled_fraction(result, existing,
                               np.ones(existing.shape, bool)) is None


@pytest.mark.parametrize("outside", OUTSIDE)
def test_unfilled_fraction_on_the_result(outside):
    mem, nuc, cells = _tissue(open_row=False)
    out = _run(_spec(outside=outside), np.zeros_like(cells),
               {"membrane": mem, "nuclei": nuc}, seed_layer=cells)
    if outside == "faces":
        assert out.unfilled_tissue_fraction is None
    else:
        assert 0.0 <= out.unfilled_tissue_fraction < 0.3


# --------------------------------------------------------------------------- #
# clicks
# --------------------------------------------------------------------------- #
def test_rasterise_points_balls_are_numbered_in_placement_order():
    out = R.rasterise_points([(5, 5, 5), (5, 5, 15)], (11, 11, 21),
                             radius_um=2.0, spacing=ISO)
    assert out.dtype == np.int32
    assert out[5, 5, 5] == 1 and out[5, 5, 15] == 2
    assert out[5, 5, 7] == 1 and out[5, 5, 8] == 0      # radius 2 um
    assert int((out == 1).sum()) == int((out == 2).sum())


def test_rasterise_points_respects_anisotropy_and_keeps_the_centre():
    out = R.rasterise_points([(3, 5, 5)], (7, 11, 11), radius_um=2.0,
                             spacing=(5.0, 1.0, 1.0))
    assert out[3, 5, 5] == 1
    assert not out[2].any() and not out[4].any()        # 5 um > 2 um
    out = R.rasterise_points([(3, 5, 5)], (7, 11, 11), radius_um=0.1,
                             spacing=ISO)
    assert int(out.sum()) == 1


@pytest.mark.parametrize("outside", OUTSIDE)
def test_clicks_fill_their_cells(outside):
    mem, nuc, cells = _tissue(open_row=False)
    pts = [(6, 6, 5), (6, 6, 15), (6, 18, 25)]
    out = _run(_spec(seeds="paint", outside=outside), np.zeros_like(cells),
               {"membrane": mem, "nuclei": nuc}, seed_points=pts)
    assert out.ok and out.n_seeds == 3
    for i, p in enumerate(pts, start=1):
        assert out.result[p] == i
    # the first click's object stays in its own cell
    assert (cells[out.result == 1] <= 101).all()


# --------------------------------------------------------------------------- #
# refusals, budget, cancel, find == preview
# --------------------------------------------------------------------------- #
def test_the_budget_refusal_at_four_million_and_one():
    roi = (slice(0, 1), slice(0, 1), slice(0, 4_000_001))
    tiny = np.zeros((1, 1, 1), np.int32)     # never cropped: refused first
    out = R.reshape_empty_region(tiny, {}, _spec(), roi=roi, spacing=ISO)
    assert out.status == "refused"
    assert out.refusal_reason == "too-big"
    assert R.ADD_ENGINE_TOO_BIG in out.messages
    rep = R.find_region_seeds(tiny, {}, _spec(), roi=roi, spacing=ISO)
    assert rep.status == "refused" and R.ADD_ENGINE_TOO_BIG in rep.messages


def test_a_box_at_exactly_the_budget_is_not_refused():
    mem, nuc, cells = _tissue()
    n = int(np.prod(cells.shape))
    out = _run(_spec(box_budget=n), np.zeros_like(cells),
               {"membrane": mem, "nuclei": nuc}, seed_layer=cells)
    assert out.ok
    out = _run(_spec(box_budget=n - 1), np.zeros_like(cells),
               {"membrane": mem, "nuclei": nuc}, seed_layer=cells)
    assert out.refusal_reason == "too-big"


def test_the_engine_sentences_are_the_spec_constants():
    assert R.ADD_ENGINE_TOO_BIG == (
        "This box is too large to fill in one go — the limit is 4 million "
        "voxels. (Nothing was changed.)")
    assert R.ADD_ENGINE_NO_MARKERS == (
        "No marker was left to grow from — every marker fell in dark space, "
        "over kept objects, or was too small to use. (Nothing was changed.)")


@pytest.mark.parametrize("outside", OUTSIDE)
@pytest.mark.parametrize("source", ["labels", "nuclei_masks", "paint"])
def test_find_region_seeds_counts_what_the_preview_uses(outside, source):
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    labels[cells == 101] = 7
    ch = {"membrane": mem, "nuclei": nuc}
    kw = {}
    if source == "paint":
        kw["seed_points"] = [(6, 6, 15), (6, 18, 25), (6, 6, 5)]
    else:
        kw["seed_layer"] = cells if source == "labels" else np.where(
            labels == 0, cells + 800, 0)
    spec = _spec(seeds=source, outside=outside)
    rep = R.find_region_seeds(labels, {"membrane": mem}, spec,
                              roi=_box(cells.shape), dark_channels=ch,
                              dark_threshold=_threshold(ch), spacing=ISO, **kw)
    out = _run(spec, labels, ch, **kw)
    assert rep.ok and out.ok
    assert rep.n_in_region == out.n_seeds
    assert rep.n_found == out.n_seeds_found
    assert (rep.seeds == out.seeds).all()


def test_a_cancel_already_set_raises():
    mem, nuc, cells = _tissue()
    ev = threading.Event()
    ev.set()
    with pytest.raises(ReshapeCancelled):
        _run(_spec(), np.zeros_like(cells), {"membrane": mem, "nuclei": nuc},
             seed_layer=cells, cancel_event=ev)
    with pytest.raises(ReshapeCancelled):
        R.find_region_seeds(np.zeros_like(cells), {"membrane": mem}, _spec(),
                            roi=_box(cells.shape), seed_layer=cells,
                            dark_channels={"membrane": mem},
                            dark_threshold=_threshold({"membrane": mem}),
                            spacing=ISO, cancel_event=ev)


@pytest.mark.parametrize("stage", R.REGION_STAGES)
def test_a_cancel_set_mid_run_raises(stage, monkeypatch):
    """A cancel set as a stage is announced raises before the NEXT stage is
    announced -- and, for the cut, before the watershed runs at all."""
    mem, nuc, cells = _tissue()
    ev = threading.Event()
    seen = []
    solved = []
    real_solve = R.solve

    def solve(*a, **k):
        solved.append(1)
        return real_solve(*a, **k)

    monkeypatch.setattr(R, "solve", solve)

    def progress(name, i, n):
        seen.append((name, i, n))
        if name == stage:
            ev.set()

    with pytest.raises(ReshapeCancelled):
        _run(_spec(), np.zeros_like(cells), {"membrane": mem, "nuclei": nuc},
             seed_layer=cells, cancel_event=ev, progress=progress)
    assert seen[-1][0] == stage
    assert not solved


def test_progress_reports_the_region_stages_in_order():
    mem, nuc, cells = _tissue()
    seen = []
    out = _run(_spec(), np.zeros_like(cells), {"membrane": mem, "nuclei": nuc},
               seed_layer=cells, progress=lambda *a: seen.append(a))
    assert out.ok
    assert [s[0] for s in seen] == list(R.REGION_STAGES)
    assert [s[1] for s in seen] == list(range(1, len(R.REGION_STAGES) + 1))
    assert all(s[2] == len(R.REGION_STAGES) for s in seen)


# --------------------------------------------------------------------------- #
# the pure helpers
# --------------------------------------------------------------------------- #
def test_eroded_cores_keep_ids_and_count_the_emptied():
    lab = np.zeros((12, 12, 30), np.int32)
    lab[2:10, 2:10, 2:10] = 5
    lab[5, 5, 20] = 6                     # emptied
    cores, n = R.eroded_cores(lab, erode_um=1.0, spacing=ISO)
    assert n == 1
    assert set(np.unique(cores)) == {0, 5}
    assert cores[3:9, 3:9, 3:9].all() and not cores[2, 5, 5]
    same, n0 = R.eroded_cores(lab, erode_um=0.0, spacing=ISO)
    assert n0 == 0 and (same == lab).all()


def test_eroded_cores_do_not_treat_the_crop_edge_as_a_boundary():
    lab = np.zeros((6, 6, 6), np.int32)
    lab[:, :, 0:3] = 2                     # touches the crop on five faces
    cores, _ = R.eroded_cores(lab, erode_um=1.0, spacing=ISO)
    assert cores[:, :, 0:2].all() and not cores[:, :, 2].any()


def test_keep_marker_components_drops_orphan_pieces():
    res = np.zeros((4, 4, 12), np.int32)
    res[:, :, 0:4] = 1
    res[:, :, 6:9] = 1                     # an orphan piece of 1
    res[:, :, 10:12] = 2
    markers = np.zeros_like(res)
    markers[1, 1, 1] = 1
    markers[1, 1, 11] = 2
    out, n = R.keep_marker_components(res, markers)
    assert n == 1
    assert not out[:, :, 6:9].any()
    assert (out[:, :, 0:4] == 1).all() and (out[:, :, 10:12] == 2).all()


def test_region_imports_no_napari_or_qt():
    import subprocess
    import sys

    code = ("import sys; import packages.epicycle_annotator.reshape.region; "
            "assert 'napari' not in sys.modules and 'qtpy' not in sys.modules")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# =========================================================================== #
# B1 fix pass (two audits of 538f359e..4942feed)
# =========================================================================== #
class _CountingEvent:
    """``is_set()`` answers True from its ``k``-th call on."""

    def __init__(self, k):
        self.k = int(k)
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls >= self.k

    def set(self):
        self.k = 0


# -- cancel: the tail, and inside the per-object loops (M-A2, L-A6) --------- #
def _cells_run(**kw):
    mem, nuc, cells = _tissue()
    return _run(_spec(), np.zeros_like(cells), {"membrane": mem, "nuclei": nuc},
                seed_layer=cells, **kw)


def _set_after(monkeypatch, name, ev):
    real = getattr(R, name)

    def wrapped(*a, **k):
        out = real(*a, **k)
        ev.set()
        return out

    monkeypatch.setattr(R, name, wrapped)


@pytest.mark.parametrize("name", ["keep_marker_components", "faint_wall_pairs",
                                  "unfilled_fraction"])
def test_a_cancel_during_the_tail_raises(monkeypatch, name):
    """A Stop after the cut used to come back ``ok`` (audit M-A2): nothing
    polled after ``keep_marker_components``."""
    ev = threading.Event()
    _set_after(monkeypatch, name, ev)
    with pytest.raises(ReshapeCancelled):
        _cells_run(cancel_event=ev)


def test_a_cancel_after_the_neighbour_cores_raises_before_the_erosion(monkeypatch):
    ev = threading.Event()
    calls = {"cores": 0, "erosion": 0}
    real_cores = R.eroded_cores
    real_erosion = R.ndi.binary_erosion

    def cores(*a, **k):
        out = real_cores(*a, **k)
        calls["cores"] += 1
        if calls["cores"] == 2:            # the neighbours' call
            ev.set()
        return out

    def erosion(*a, **k):
        calls["erosion"] += 1
        return real_erosion(*a, **k)

    monkeypatch.setattr(R, "eroded_cores", cores)
    monkeypatch.setattr(R.ndi, "binary_erosion", erosion)
    with pytest.raises(ReshapeCancelled):
        _cells_run(cancel_event=ev)
    assert calls["cores"] == 2
    assert calls["erosion"] == 0


def test_a_cancel_inside_the_eroded_cores_loop_raises():
    lab = np.zeros((10, 10, 40), np.int32)
    for i, x0 in enumerate((1, 11, 21, 31), start=1):
        lab[2:8, 2:8, x0:x0 + 7] = i
    with pytest.raises(ReshapeCancelled):
        R.eroded_cores(lab, erode_um=1.0, spacing=ISO,
                       cancel_event=_CountingEvent(2))
    # and without a cancel all four come back
    cores, n = R.eroded_cores(lab, erode_um=1.0, spacing=ISO,
                              cancel_event=_CountingEvent(10 ** 6))
    assert n == 0 and set(np.unique(cores)) == {0, 1, 2, 3, 4}


def test_a_cancel_inside_the_keep_marker_components_loop_raises():
    res = np.zeros((4, 4, 30), np.int32)
    markers = np.zeros_like(res)
    for i, x0 in enumerate((0, 8, 16, 24), start=1):
        res[:, :, x0:x0 + 5] = i
        markers[1, 1, x0 + 1] = i
    with pytest.raises(ReshapeCancelled):
        R.keep_marker_components(res, markers, cancel_event=_CountingEvent(2))


# -- progress: named at the START of each stage (M-A3) ---------------------- #
def test_each_stage_is_announced_before_its_work_starts(monkeypatch):
    done = []
    for name in ("_prepare", "_build_wall_stage", "_dark_stage", "_marker_stage",
                 "solve"):
        real = getattr(R, name)

        def wrapped(*a, _real=real, _name=name, **k):
            out = _real(*a, **k)
            done.append(_name)
            return out

        monkeypatch.setattr(R, name, wrapped)
    at = {}

    def progress(stage, i, n):
        at[stage] = list(done)

    out = _cells_run(progress=progress)
    assert out.ok
    assert at["add_box"] == []
    assert at["elevation"] == ["_prepare"]
    assert at["add_markers"] == ["_prepare", "_build_wall_stage"]
    assert at["add_cut"] == ["_prepare", "_build_wall_stage", "_dark_stage",
                             "_marker_stage"]


# -- memory: the wall stage's side products are let go (L-A5) --------------- #
def test_the_wall_stage_side_products_are_released_before_the_cut(monkeypatch):
    import gc
    import weakref

    real_wall = R._build_wall_stage
    refs = []

    class _Tracked(np.ndarray):
        pass

    def wall(*a, **k):
        elev, clahe, ridge, img = real_wall(*a, **k)
        side = [np.zeros((2, 2, 2), np.float32).view(_Tracked) for _ in range(3)]
        refs.extend(weakref.ref(x) for x in side)
        return (elev, *side)

    alive = []
    real_solve = R.solve

    def solve(*a, **k):
        gc.collect()
        alive.append([r() is not None for r in refs])
        return real_solve(*a, **k)

    monkeypatch.setattr(R, "_build_wall_stage", wall)
    monkeypatch.setattr(R, "solve", solve)
    assert _cells_run().ok
    assert alive == [[False, False, False]]
    # ... unless the caller asked to see them
    refs.clear()
    alive.clear()
    out = _cells_run(keep_intermediates=True)
    # clahe and ridge are kept for the caller; the prepared image never was
    assert alive == [[True, True, False]]
    assert out.intermediates["clahe"] is not None


# -- speed: per-object stats in one pass (M-A1) ------------------------------ #
def test_the_region_piece_stats_equal_the_engine_ones():
    from packages.epicycle_annotator.reshape.engine import _piece_stats

    rng = np.random.default_rng(5)
    part = np.zeros((14, 30, 33), np.int32)
    for i in range(1, 21):
        z, y, x = rng.integers(0, 10), rng.integers(0, 25), rng.integers(0, 28)
        part[z:z + rng.integers(1, 5), y:y + rng.integers(1, 6),
             x:x + rng.integers(1, 6)] = i
    part[part == 7] = 0                          # a gap in the ids
    part[0, 0, 0] = 5                            # a disjoint voxel of 5
    roi = (slice(3, 17), slice(10, 40), slice(0, 33))
    assert R._region_piece_stats(part, roi) == _piece_stats(part, roi)
    assert R._region_piece_stats(np.zeros((3, 3, 3), np.int32),
                                 _box((3, 3, 3))) == []


def test_find_region_seeds_computes_no_per_marker_stats(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("per-marker stats computed")

    monkeypatch.setattr(R, "_region_piece_stats", boom)
    mem, nuc, cells = _tissue()
    ch = {"membrane": mem, "nuclei": nuc}
    rep = R.find_region_seeds(np.zeros_like(cells), {"membrane": mem}, _spec(),
                              roi=_box(cells.shape), seed_layer=cells,
                              dark_channels=ch, dark_threshold=_threshold(ch),
                              spacing=ISO)
    assert rep.ok and rep.n_found == 8 and rep.n_in_region == 8


# -- inputs: slice before reading (M-A4 / LOW-B10) --------------------------- #
class _Lazy:
    """Sliceable like a lazy / multiscale layer, but reading it WHOLE fails."""

    def __init__(self, array):
        self._a = array
        self.shape = array.shape
        self.ndim = array.ndim
        self.dtype = array.dtype

    def __array__(self, *a, **k):
        raise AssertionError("the whole array was read")

    def __getitem__(self, key):
        return self._a[key]


@pytest.mark.parametrize("source", ["labels", "paint"])
def test_a_lazy_full_volume_is_sliced_before_it_is_read(source):
    mem, nuc, cells = _tissue(shape=(12, 40, 40))
    big = (20, 60, 70)
    roi = (slice(4, 16), slice(10, 50), slice(20, 60))

    def pad(a):
        out = np.zeros(big, a.dtype)
        out[roi] = a
        return out

    labels = np.zeros_like(cells)
    labels[cells == 101] = 7
    ch = {"membrane": mem, "nuclei": nuc}
    thr = _threshold(ch)
    spec = _spec(seeds=source)
    kw = ({"seed_points": [(6, 18, 25), (6, 6, 15)]} if source == "paint"
          else {})
    plain = R.reshape_empty_region(
        pad(labels), {"membrane": pad(mem)}, spec, roi=roi,
        dark_channels={k: pad(v) for k, v in ch.items()}, dark_threshold=thr,
        seed_layer=None if source == "paint" else pad(cells), spacing=ISO, **kw)
    lazy = R.reshape_empty_region(
        _Lazy(pad(labels)), {"membrane": _Lazy(pad(mem)), "nuclei": _Lazy(pad(nuc))},
        spec, roi=roi, dark_channels={k: _Lazy(pad(v)) for k, v in ch.items()},
        dark_threshold=thr,
        seed_layer=None if source == "paint" else _Lazy(pad(cells)),
        spacing=ISO, **kw)
    assert plain.ok and lazy.ok
    assert (plain.result == lazy.result).all()
    # the same answer as when the caller hands in box crops
    crop = R.reshape_empty_region(
        labels, {"membrane": mem}, spec, roi=roi, dark_channels=ch,
        dark_threshold=thr, seed_layer=None if source == "paint" else cells,
        spacing=ISO, **kw)
    assert (crop.result == plain.result).all()
    assert crop.piece_stats == plain.piece_stats
    rep = R.find_region_seeds(
        _Lazy(pad(labels)), {}, spec, roi=roi,
        dark_channels={k: _Lazy(pad(v)) for k, v in ch.items()},
        dark_threshold=thr,
        seed_layer=None if source == "paint" else _Lazy(pad(cells)),
        spacing=ISO, **kw)
    assert rep.ok and rep.n_in_region == plain.n_seeds


def test_a_box_crop_is_copied_not_aliased():
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    ch = {"membrane": mem, "nuclei": nuc}
    before = labels.copy()
    out = _run(_spec(), labels, ch, seed_layer=cells)
    assert out.ok and (labels == before).all()


# -- the box itself (L-A2) --------------------------------------------------- #
def test_a_negative_box_start_is_refused_naming_the_axis():
    mem, nuc, cells = _tissue()
    roi = (slice(0, 12), slice(-2, 38), slice(0, 40))
    with pytest.raises(ValueError, match="axis 1"):
        R.reshape_empty_region(np.zeros_like(cells), {"membrane": mem}, _spec(),
                               roi=roi, seed_layer=cells, spacing=ISO,
                               dark_channels={"membrane": mem},
                               dark_threshold=_threshold({"membrane": mem}))


def test_a_box_past_the_volume_edge_is_refused_naming_the_axis():
    mem, nuc, cells = _tissue()            # (12, 40, 40)
    roi = (slice(0, 12), slice(0, 30), slice(20, 45))
    with pytest.raises(ValueError, match="axis 2"):
        R.reshape_empty_region(np.zeros_like(cells), {"membrane": mem}, _spec(),
                               roi=roi, seed_layer=cells, spacing=ISO,
                               dark_channels={"membrane": mem},
                               dark_threshold=_threshold({"membrane": mem}))
    with pytest.raises(ValueError, match="axis 2"):
        R.find_region_seeds(np.zeros_like(cells), {}, _spec(), roi=roi,
                            seed_layer=cells, spacing=ISO,
                            dark_channels={"membrane": mem},
                            dark_threshold=_threshold({"membrane": mem}))


# -- the region target's own keywords (L-A3 / LOW-B8) ------------------------ #
def test_label_path_keywords_are_refused_on_the_region_target():
    mem, nuc, cells = _tissue()
    ch = {"membrane": mem, "nuclei": nuc}
    kw = dict(roi=_box(cells.shape), seed_layer=cells, dark_channels=ch,
              dark_threshold=_threshold(ch), spacing=ISO)
    with pytest.raises(ValueError, match="keep_seed_ids"):
        reshape_region(np.zeros_like(cells), {"membrane": mem}, _spec(),
                       keep_seed_ids=[1], **kw)
    with pytest.raises(ValueError, match="target_id"):
        reshape_region(np.zeros_like(cells), {"membrane": mem}, _spec(),
                       target_id=3, **kw)
    with pytest.raises(ValueError, match="target_id"):
        find_seeds(np.zeros_like(cells), {"membrane": mem}, _spec(),
                   target_id=3, **kw)


def test_region_seeds_is_one_tuple_shared_with_the_spec():
    from packages.epicycle_annotator.reshape import spec as S

    assert R.REGION_SEEDS is S.REGION_SEEDS
    assert R.REGION_SEEDS == ("labels", "nuclei_masks", "paint")


# -- the too-big sentence follows the budget (L-A4) -------------------------- #
def test_the_too_big_sentence_follows_the_budget():
    assert R.too_big_sentence(4_000_000) == R.ADD_ENGINE_TOO_BIG
    assert "4 million" in R.too_big_sentence(4_000_000)
    assert "2.5 million" in R.too_big_sentence(2_500_000)
    assert "4 million" not in R.too_big_sentence(2_500_000)
    assert "750,000 voxels" in R.too_big_sentence(750_000)


def test_the_refusal_names_the_budget_it_used():
    mem, nuc, cells = _tissue()
    n = int(np.prod(cells.shape))
    out = _run(_spec(box_budget=n - 1), np.zeros_like(cells),
               {"membrane": mem, "nuclei": nuc}, seed_layer=cells)
    assert out.refusal_reason == "too-big"
    assert out.messages == [R.too_big_sentence(n - 1)]
    assert R.ADD_ENGINE_TOO_BIG not in out.messages


def test_the_result_is_int32():
    out = _cells_run()
    assert out.ok and out.result.dtype == np.int32


# -- test discrimination: the dark rule (MED-B1..B4, LOW-B5, LOW-B7) --------- #
def test_the_dark_factor_decides():
    """0.3 against t = 0.4: dark at factor 1.0 (0.3 < 0.4), not at 0.5
    (0.3 >= 0.2).  A dark mask that ignored the factor fails one side."""
    img = np.full((8, 8, 8), 0.3, np.float32)
    thr = {"t": 0.4, "lo": {"a": 0.0}, "hi": {"a": 1.0}}
    assert R.dark_mask({"a": img}, dark_threshold=thr, factor=1.0,
                       spacing=ISO).all()
    assert not R.dark_mask({"a": img}, dark_threshold=thr, factor=0.5,
                           spacing=ISO).any()


def test_the_dark_blur_is_two_microns_at_the_true_spacing():
    rng = np.random.default_rng(11)
    img = rng.uniform(0, 1, (20, 40, 40)).astype(np.float32)
    thr = {"t": 0.5, "lo": {"a": 0.0}, "hi": {"a": 1.0}}
    got = R.dark_mask({"a": img}, dark_threshold=thr, factor=1.0,
                      spacing=(1.0, 0.5, 0.5))
    want = ndi.gaussian_filter(img, (2.0, 4.0, 4.0)) < 0.5
    iso = ndi.gaussian_filter(img, (2.0, 2.0, 2.0)) < 0.5
    assert (got == want).all()
    assert (want != iso).any()             # the case can tell them apart


def test_a_marker_exactly_half_in_the_dark_is_dropped():
    markers = np.zeros((1, 1, 10), np.int32)
    markers[0, 0, 0:4] = 1                 # 2 of 4 dark: dropped
    markers[0, 0, 5:8] = 2                 # 1 of 3 dark: kept
    dark = np.zeros(markers.shape, bool)
    dark[0, 0, 2:4] = True
    dark[0, 0, 7] = True
    out, n = R._drop_dark(markers, dark)
    assert n == 1
    assert set(np.unique(out)) == {0, 2}


def test_the_normalisation_takes_the_first_and_the_99_5th_percentile():
    rng = np.random.default_rng(2)
    a = rng.uniform(0.0, 1.0, (20, 20, 20)).astype(np.float32)
    flat = a.reshape(-1)
    hot = rng.choice(flat.size, size=flat.size // 200, replace=False)  # 0.5 %
    flat[hot] = 100.0
    cold = rng.choice(np.setdiff1d(np.arange(flat.size), hot),
                      size=flat.size // 200, replace=False)
    flat[cold] = -100.0
    out = R.patch_dark_threshold({"a": a}, spacing=ISO, stride=1)
    assert out["hi"]["a"] == pytest.approx(float(np.percentile(a, 99.5)))
    assert out["lo"]["a"] == pytest.approx(float(np.percentile(a, 1)))
    assert out["hi"]["a"] < 2.0 < float(a.max())
    assert out["lo"]["a"] > -1.0 > float(a.min())


def _dark_outside_case(**params):
    """The dark half x >= 15; one bright run marker; flat elevation."""
    shape, ch, labels = _dark_case()
    guide = np.zeros(shape, np.int32)
    guide[8:12, 12:18, 4:8] = 3
    spec = _spec(outside="faces+dark", core_erode_um=0.0, **params)
    out = _run(spec, labels, ch, seed_layer=guide,
               elevation=np.zeros(shape, np.float32), keep_intermediates=True)
    assert out.ok
    return out


def _off_faces(shape):
    inner = np.zeros(shape, bool)
    inner[1:-1, 1:-1, 1:-1] = True
    return inner


def test_the_outside_marker_keeps_the_clearance_from_the_dark_edge():
    out = _dark_outside_case(dark_clearance_um=2.0)
    n = out.n_seeds
    dark = out.intermediates["dark_raw"]
    outside = out.intermediates["markers"] == n + 1
    depth = ndi.distance_transform_edt(dark)       # to the nearest non-dark
    near = dark & (depth <= 2.0) & _off_faces(dark.shape)
    far = dark & (depth > 3.0) & _off_faces(dark.shape)
    assert near.any() and far.any()
    assert not outside[near].any()
    assert outside[far].any()


def test_without_a_clearance_all_the_dark_is_outside():
    out = _dark_outside_case(dark_clearance_um=0.0)
    n = out.n_seeds
    dark = out.intermediates["dark_raw"]
    markers = out.intermediates["markers"]
    assert dark.any()
    assert (markers[dark] == n + 1).all()


def test_a_click_in_the_dark_clears_the_outside_marker_around_it():
    """The click ball (0.5 um: one voxel) is smaller than the clearance
    (3 um), so it is the CLEARANCE that keeps the outside marker away; the
    object then grows well past its one-voxel ball."""
    shape, ch, labels = _dark_case()
    click = (10, 15, 24)
    spec = _spec(seeds="paint", outside="faces+dark", click_radius_um=0.5,
                 dark_clearance_um=3.0)
    out = _run(spec, labels, ch, seed_points=[click],
               elevation=np.zeros(shape, np.float32), keep_intermediates=True)
    assert out.ok and out.n_seeds == 1
    assert int((out.seeds == 1).sum()) == 1
    zz, yy, xx = np.indices(shape)
    dist = np.sqrt((zz - click[0]) ** 2 + (yy - click[1]) ** 2
                   + (xx - click[2]) ** 2)
    outside = out.intermediates["markers"] == 2
    assert outside.any()
    assert not outside[dist <= 3.0].any()
    assert int((out.result == 1).sum()) > 20


# -- test discrimination: 6-connectivity (LOW-B6) ---------------------------- #
def test_a_piece_touching_its_marker_only_diagonally_is_dropped():
    res = np.zeros((3, 6, 6), np.int32)
    res[1, 1:3, 1:3] = 1                   # the marker's piece
    res[1, 3, 3] = 1                       # touches it on an edge diagonal only
    markers = np.zeros_like(res)
    markers[1, 1, 1] = 1
    out, n = R.keep_marker_components(res, markers)
    assert n == 1
    assert out[1, 3, 3] == 0 and (out[1, 1:3, 1:3] == 1).all()

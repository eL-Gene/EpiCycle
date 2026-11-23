"""The region target's second way to grow cells -- ``growth="compete"``.

``reshape_empty_region`` dispatches to the competition with a growth limit
(``reshape/compete.py``) when the spec says ``growth="compete"``; the flood
stays the engine default.  Pinned here: no outside marker, the space
(empty, cut to the tissue outline when one is chosen), kept objects never
written and kept competitors keeping nothing, the order of the marker
decisions and their separate counts, an outline turning the dark plumbing
on whatever the tick says, the membrane step on and off, the flood with an
outline (hooks A, B and D only), the whole-patch exemption of the box
budget, the new spec knobs and their plain-word refusals, the outline as
a run argument (never a param; a box always gets the patch-wide outline
cropped), the outline's own no-marker reason, the warm-up, and the Add
worker handing the knobs over.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import ReshapeSpec
from packages.epicycle_annotator.reshape import compete as C
from packages.epicycle_annotator.reshape import region as R

from .test_reshape_region import _box, _threshold, _tissue

ISO = (1.0, 1.0, 1.0)


def _spec(seeds="labels", outside="faces+dark", **params):
    base = {"wall_channel": "membrane", "ridge": False, "outside": outside}
    base.update(params)
    return ReshapeSpec(target="region", seeds=seeds, walls="membrane",
                       params=base)


def _compete_spec(seeds="labels", outside="faces+dark", **params):
    base = {"growth": "compete"}
    base.update(params)
    return _spec(seeds, outside, **base)


def _run(spec, labels, channels, *, seed_layer=None, seed_points=None,
         spacing=ISO, roi=None, **kw):
    return R.reshape_empty_region(
        labels, {"membrane": channels["membrane"]}, spec,
        roi=_box(labels.shape) if roi is None else roi,
        dark_channels=channels, seed_layer=seed_layer,
        seed_points=seed_points, dark_threshold=_threshold(channels, spacing),
        spacing=spacing, **kw)


def _world(shape=(12, 40, 40), seed=0):
    mem, nuc, cells = _tissue(shape=shape, seed=seed)
    return {"membrane": mem, "nuclei": nuc}, cells


def _ring(existing):
    """The empty voxels 6-adjacent to a kept object."""
    kept = existing != 0
    grown = ndi.binary_dilation(kept, structure=ndi.generate_binary_structure(3, 1))
    return grown & ~kept


# --------------------------------------------------------------------------- #
# the spec
# --------------------------------------------------------------------------- #
def test_the_new_knobs_and_their_defaults():
    p = _spec().params
    assert p["growth"] == "flood"
    assert p["wall_cost"] == C.WALL_COST == 40.0
    assert p["growth_limit"] == C.GROWTH_LIMIT == 28.0
    assert p["include_membrane"] is True
    assert p["outline"] == "none"
    assert p["bridge_um"] == 10.0
    assert "outline_mask" not in p
    assert p["whole_patch"] is False


@pytest.mark.parametrize("key, bad", [
    ("wall_cost", -1.0), ("wall_cost", 100.5), ("growth_limit", 1.9),
    ("growth_limit", 40.1), ("bridge_um", -0.1), ("bridge_um", 30.5),
])
def test_the_new_ranges_are_enforced(key, bad):
    with pytest.raises(ValueError, match=key):
        _compete_spec(**{key: bad})


@pytest.mark.parametrize("key, good", [
    ("wall_cost", 0.0), ("wall_cost", 100.0), ("growth_limit", 2.0),
    ("growth_limit", 40.0), ("bridge_um", 0.0), ("bridge_um", 30.0),
])
def test_the_range_ends_are_allowed(key, good):
    assert _compete_spec(**{key: good}).params[key] == good


def test_unknown_growth_and_outline_values_are_refused_naming_the_options():
    with pytest.raises(ValueError, match="flood, compete"):
        _spec(growth="spread")
    with pytest.raises(ValueError, match="none, shrink, cleared"):
        _spec(outline="tissue")


def test_the_outline_volume_is_not_a_param():
    """The outline itself is a run argument: a spec carrying it is refused
    as an unknown key, and a spec with an outline MODE serialises and
    compares again."""
    with pytest.raises(ValueError, match="outline_mask"):
        _compete_spec(outline="cleared", outline_mask=np.ones((2, 2, 2), bool))
    for mode in ("shrink", "cleared"):
        spec = _compete_spec(outline=mode)
        assert ReshapeSpec.from_json(spec.to_json()) == spec
        assert spec == _compete_spec(outline=mode)
        assert spec != _compete_spec(outline="none")


def _both_entry_points(spec, *, outline):
    ch, cells = _world()
    labels = np.zeros_like(cells)
    kw = dict(roi=_box(cells.shape), dark_channels=ch, seed_layer=cells,
              dark_threshold=_threshold(ch), spacing=ISO, outline=outline)
    yield lambda: R.reshape_empty_region(
        labels, {"membrane": ch["membrane"]}, spec, **kw)
    yield lambda: R.find_region_seeds(labels, {}, spec, **kw)


def test_the_cleared_outline_needs_its_volume():
    for call in _both_entry_points(_compete_spec(outline="cleared"),
                                   outline=None):
        with pytest.raises(ValueError, match="'cleared' needs outline="):
            call()


def test_an_outline_with_no_mode_chosen_is_refused():
    ch, cells = _world()
    for call in _both_entry_points(_compete_spec(),
                                   outline=np.ones(cells.shape, bool)):
        with pytest.raises(ValueError, match="no outline is chosen"):
            call()


def test_the_outline_must_be_a_3d_volume():
    for call in _both_entry_points(_compete_spec(outline="cleared"),
                                   outline=np.ones((2, 2), bool)):
        with pytest.raises(ValueError, match="3D"):
            call()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_float_outline_is_refused(dtype):
    ch, cells = _world()
    for call in _both_entry_points(_compete_spec(outline="cleared"),
                                   outline=np.ones(cells.shape, dtype)):
        with pytest.raises(ValueError, match="bool, or integers 0 and 1"):
            call()


def test_an_integer_outline_of_0_and_1_is_the_bool_one():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    outline = np.zeros(cells.shape, bool)
    outline[:, :24, :] = True
    spec = _compete_spec(outline="cleared")
    a = _run(spec, labels, ch, seed_layer=cells, outline=outline)
    b = _run(spec, labels, ch, seed_layer=cells,
             outline=outline.astype(np.uint8))
    assert a.ok and b.ok
    assert np.array_equal(a.result, b.result)
    bad = outline.astype(np.int32) * 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        _run(spec, labels, ch, seed_layer=cells, outline=bad)


def test_the_whole_patch_needs_the_competition():
    with pytest.raises(ValueError, match="whole_patch"):
        _spec(whole_patch=True)
    assert _compete_spec(whole_patch=True).params["whole_patch"] is True


@pytest.mark.parametrize("key", ["include_membrane", "whole_patch"])
def test_the_ticks_must_be_true_or_false(key):
    with pytest.raises(ValueError, match=key):
        _compete_spec(**{key: "yes"})


# --------------------------------------------------------------------------- #
# the competition through reshape_empty_region
# --------------------------------------------------------------------------- #
def test_compete_has_no_outside_marker():
    """The flood keeps the six box faces for its outside marker; the
    competition has none, so new cells reach the box faces."""
    ch, cells = _world()
    labels = np.zeros_like(cells)
    flood = _run(_spec(outside="faces"), labels, ch, seed_layer=cells)
    comp = _run(_compete_spec(outside="faces"), labels, ch, seed_layer=cells)
    assert flood.ok and comp.ok
    faces = np.zeros(cells.shape, bool)
    faces[0] = faces[-1] = True
    faces[:, 0] = faces[:, -1] = True
    faces[:, :, 0] = faces[:, :, -1] = True
    off_marker = faces & (flood.seeds == 0)
    assert not (flood.result[off_marker] > 0).any()
    assert (comp.result[off_marker] > 0).sum() > 100


def test_compete_writes_only_ids_one_to_n_on_empty_space():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    labels[:, :12, :10] = 7          # a kept object over one cell
    comp = _run(_compete_spec(), labels, ch, seed_layer=cells)
    assert comp.ok
    ids = np.unique(comp.result)
    ids = ids[ids != 0]
    assert ids.min() >= 1 and ids.max() <= comp.n_seeds
    assert not (comp.result[labels != 0] > 0).any()
    assert comp.result.dtype == np.int32
    assert comp.n_tiles == 1


@pytest.mark.parametrize("membrane", [True, False])
def test_kept_objects_outside_the_outline_are_untouched(membrane):
    """I5: the outline limits where NEW objects grow; a kept object outside
    it is never written, and nothing new appears outside it."""
    ch, cells = _world()
    labels = np.zeros_like(cells)
    labels[:, 30:36, 5:15] = 9       # kept, in the open space
    labels[:, :12, 31:40] = 4        # kept, inside the tissue
    outline = np.zeros(cells.shape, bool)
    outline[:, :26, :] = True        # the tissue; the kept 9 is outside
    before = labels.copy()
    out = _run(_compete_spec(outline="cleared", include_membrane=membrane),
               labels, ch, seed_layer=cells, outline=outline)
    assert out.ok
    assert np.array_equal(labels, before)
    assert not (out.result[labels != 0] > 0).any()
    assert not (out.result[~outline] > 0).any()


@pytest.mark.parametrize("membrane", [True, False])
def test_kept_competitors_keep_nothing(membrane):
    """Every kept object competes from the empty voxels next to it, and its
    claims are zeroed: no new cell takes its side of the gap, and no claim
    above n_seeds survives."""
    ch, cells = _world()
    labels = np.zeros_like(cells)
    labels[:, :12, 11:20] = 5        # a kept cell between two new ones
    out = _run(_compete_spec(include_membrane=membrane, wall_cost=0.0,
                             growth_limit=40.0), labels, ch, seed_layer=cells)
    assert out.ok
    assert int(out.result.max()) <= out.n_seeds
    # the ring next to the kept cell stays empty (it was the kept cell's)
    ring = _ring(labels)
    assert not (out.result[ring] > 0).any()


def test_without_kept_competitors_a_new_cell_would_take_the_ring():
    """The control for the test above: the same box with the kept cell's
    voxels merely blocked (no competitor) lets new cells reach its ring."""
    ch, cells = _world()
    labels = np.zeros_like(cells)
    labels[:, :12, 11:20] = 5
    space = labels == 0
    new, _n = R._relabel_ascending(np.where(space, cells, 0))
    lab, _cost = C.compete(np.zeros(cells.shape, np.float32), new, space, ISO,
                           wall_cost=0.0, growth_limit=40.0)
    assert (lab[_ring(labels)] > 0).any()


def test_the_space_is_empty_minus_the_eroded_dark_without_an_outline():
    ch, cells = _world(shape=(24, 40, 40))
    labels = np.zeros_like(cells)
    dark = _run(_compete_spec(wall_cost=0.0, growth_limit=40.0), labels, ch,
                seed_layer=cells)
    faces = _run(_compete_spec(outside="faces", wall_cost=0.0, growth_limit=40.0),
                 labels, ch, seed_layer=cells)
    assert dark.ok and faces.ok
    # the open space (y >= 26, well inside the dark) is left alone with the
    # dark rule and filled without it
    assert (dark.result[:, 30:, :] > 0).sum() == 0
    assert (faces.result[:, 30:, :] > 0).sum() > 1000


def test_include_membrane_adds_only_bright_voxels_to_new_cells():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    on = _run(_compete_spec(include_membrane=True), labels, ch, seed_layer=cells)
    off = _run(_compete_spec(include_membrane=False), labels, ch, seed_layer=cells)
    assert on.ok and off.ok
    grown = (on.result > 0) & (off.result == 0)
    assert grown.sum() > 0
    assert (on.result[off.result > 0] == off.result[off.result > 0]).all()
    assert (np.asarray(on.elevation)[grown] >= C.MEMBRANE_LEVEL).all()


def test_the_elevation_is_the_same_under_either_growth_and_any_outline():
    """Hook C (the rim band) is flood-only and not built: no growth rule
    and no outline changes the elevation the cut reads."""
    ch, cells = _world()
    labels = np.zeros_like(cells)
    outline = np.zeros(cells.shape, bool)
    outline[:, :26, :] = True
    a = _run(_spec(), labels, ch, seed_layer=cells)
    b = _run(_spec(outline="cleared"), labels, ch, seed_layer=cells,
             outline=outline)
    c = _run(_compete_spec(outline="cleared"), labels, ch, seed_layer=cells,
             outline=outline)
    assert np.array_equal(a.elevation, b.elevation)
    assert np.array_equal(a.elevation, c.elevation)


# --------------------------------------------------------------------------- #
# the marker decisions
# --------------------------------------------------------------------------- #
def _markers_10():
    """Ten-voxel markers 1..4 along x, each in its own row."""
    m = np.zeros((1, 4, 10), np.int32)
    for i in range(4):
        m[0, i, :] = i + 1
    return m


def test_the_dark_drop_and_the_outline_drop_are_separate_and_in_order():
    """Marker 1: 50 % dark -> dark.  Marker 2: 70 % outside -> outline.
    Marker 3: 30 % dark AND 30 % outside (disjoint) -> KEPT (each rule on
    its own; the studies' union at 0.5 would have dropped it).  Marker 4:
    60 % dark and 80 % outside -> dropped ONCE, counted as dark (the dark
    drop runs first)."""
    m = _markers_10()
    dark = np.zeros(m.shape, bool)
    outline = np.ones(m.shape, bool)
    dark[0, 0, :5] = True                         # 1: 50 % dark
    outline[0, 1, :7] = False                     # 2: 70 % outside
    dark[0, 2, :3] = True                         # 3: 30 % dark ...
    outline[0, 2, 7:] = False                     # ... and 30 % outside
    dark[0, 3, :6] = True                         # 4: 60 % dark
    outline[0, 3, :8] = False                     # ... and 80 % outside
    kept, n_dark, n_out = R._drop_automatic(m, dark_raw=dark, outline=outline)
    assert (n_dark, n_out) == (2, 1)
    assert sorted(int(v) for v in np.unique(kept) if v) == [3]


def test_just_under_each_share_is_kept():
    m = _markers_10()
    dark = np.zeros(m.shape, bool)
    outline = np.ones(m.shape, bool)
    dark[0, 0, :4] = True                         # 40 % dark
    outline[0, 1, :6] = False                     # 60 % outside
    kept, n_dark, n_out = R._drop_automatic(m, dark_raw=dark, outline=outline)
    assert (n_dark, n_out) == (0, 0)
    assert np.array_equal(kept, m)


def test_no_dark_and_no_outline_drop_nothing():
    m = _markers_10()
    kept, n_dark, n_out = R._drop_automatic(m, dark_raw=None, outline=None)
    assert (n_dark, n_out) == (0, 0) and kept is m


def test_the_dark_drop_counts_huge_ids():
    """I14: the dark drop compacts its ids (np.bincount), so an id of
    2**31 - 1 costs nothing and decides as before."""
    m = np.zeros((1, 2, 4), np.int64)
    m[0, 0, :] = 2 ** 31 - 1
    m[0, 1, :] = 3
    dark = np.zeros(m.shape, bool)
    dark[0, 0, :2] = True                          # exactly half -> dropped
    dark[0, 1, :1] = True
    out, n = R._drop_dark(m, dark)
    assert n == 1
    assert out.dtype == m.dtype
    assert sorted(int(v) for v in np.unique(out) if v) == [3]


def test_the_dark_drop_agrees_with_the_sum_it_replaced():
    rng = np.random.default_rng(3)
    m = rng.integers(0, 40, (6, 20, 20)).astype(np.int32)
    dark = rng.random(m.shape) < 0.5
    ids = np.unique(m)
    ids = ids[ids != 0]
    total = ndi.sum(np.ones(m.shape), labels=m, index=ids)
    in_dark = ndi.sum(dark.astype(np.float64), labels=m, index=ids)
    expect = ids[np.asarray(in_dark) / np.maximum(np.asarray(total), 1.0) >= 0.5]
    out, n = R._drop_dark(m, dark)
    assert n == expect.size
    assert np.array_equal(out, np.where(np.isin(m, expect), 0, m))


def test_a_sliver_marker_is_dropped_under_the_competition_only():
    """Marker order under compete: the cores, then the sliver drop on the
    UNCLIPPED cores, then the clip.  A run object 40 % over a kept object
    is a sliver (dropped, counted); the flood clips it and keeps it.  A run
    object wholly over a kept object stays counted 'over kept', not as a
    sliver."""
    ch, cells = _world()
    labels = np.zeros_like(cells)
    # cell 101 spans y 0-11, x 0-9; its core is y 1-10, x 1-8 (erode 1 um)
    labels[:, :12, :6] = 50          # 60 % of 101's core is kept
    labels[:, :12, 11:20] = 51       # cell 102 wholly over a kept object
    flood = _run(_spec(), labels, ch, seed_layer=cells)
    comp = _run(_compete_spec(), labels, ch, seed_layer=cells)
    assert flood.ok and comp.ok
    assert flood.n_slivers == 0
    assert (flood.n_markers_over_kept, comp.n_markers_over_kept) == (1, 1)
    assert comp.n_slivers == 1
    assert comp.n_seeds == flood.n_seeds - 1


def test_the_outline_drop_reaches_the_result_and_the_seed_count():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    outline = np.ones(cells.shape, bool)
    outline[:, :12, :10] = False     # cell 101 wholly outside
    for spec in (_spec(outline="cleared"), _compete_spec(outline="cleared")):
        out = _run(spec, labels, ch, seed_layer=cells, outline=outline)
        rep = R.find_region_seeds(
            labels, {"membrane": ch["membrane"]}, spec, roi=_box(cells.shape),
            dark_channels=ch, seed_layer=cells, dark_threshold=_threshold(ch),
            spacing=ISO, outline=outline)
        assert out.ok
        assert out.n_outside_outline == 1
        assert rep.n_in_region == out.n_seeds == 7


def test_clicks_are_never_dropped_by_the_outline():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    outline = np.ones(cells.shape, bool)
    outline[:, :12, :10] = False
    clicks = [(6, 6, 5), (6, 6, 15)]
    out = _run(_compete_spec(seeds="paint", outline="cleared"),
               labels, ch, seed_points=clicks, outline=outline)
    assert out.ok
    assert out.n_outside_outline == 0 and out.n_seeds == 2


# --------------------------------------------------------------------------- #
# an outline forces the dark plumbing
# --------------------------------------------------------------------------- #
def test_an_outline_turns_the_dark_plumbing_on_with_the_tick_off():
    """With outside='faces' a marker in the open dark space survives; the
    same spec with an outline drops it as dark and measures the unfilled
    share against the outline."""
    ch, cells = _world(shape=(24, 40, 40))
    cells = cells.copy()
    cells[8:16, 30:36, 14:20] = 300            # a run object in the dark
    labels = np.zeros_like(cells)
    outline = np.zeros(cells.shape, bool)
    outline[:, :24, :] = True
    plain = _run(_compete_spec(outside="faces"), labels, ch, seed_layer=cells)
    with_outline = _run(_compete_spec(outside="faces", outline="cleared"),
                        labels, ch, seed_layer=cells, outline=outline)
    assert plain.ok and with_outline.ok
    assert plain.n_dropped_dark == 0 and plain.unfilled_tissue_fraction is None
    assert with_outline.n_dropped_dark == 1
    assert with_outline.n_outside_outline == 0      # counted once, as dark
    assert with_outline.unfilled_tissue_fraction is not None


def test_an_outline_with_the_tick_off_still_needs_the_dark_inputs():
    ch, cells = _world()
    outline = np.ones(cells.shape, bool)
    spec = _compete_spec(outside="faces", outline="cleared")
    with pytest.raises(ValueError, match="dark_threshold"):
        R.reshape_empty_region(np.zeros_like(cells), {"membrane": ch["membrane"]},
                               spec, roi=_box(cells.shape), seed_layer=cells,
                               dark_channels=ch, spacing=ISO, outline=outline)


def test_the_shrink_wrap_is_built_from_the_dark_channels_on_the_whole_patch():
    """Only the whole patch builds its own shrink outline (the run IS the
    patch); it is the outline the caller would have built and handed in."""
    from packages.epicycle_annotator.reshape import outline as O

    ch, cells = _world(shape=(24, 40, 40))
    labels = np.zeros_like(cells)
    built = O.tissue_outline(ch, ISO, dark_factor=0.5, bridge_um=10.0,
                             dark_threshold=_threshold(ch))
    spec = _compete_spec(outline="shrink", whole_patch=True)
    a = _run(spec, labels, ch, seed_layer=cells)
    b = _run(spec, labels, ch, seed_layer=cells, outline=built)
    assert a.ok and b.ok
    assert np.array_equal(a.result, b.result)
    assert not (a.result[~built] > 0).any()


def test_the_dark_channels_are_cut_to_the_box_once(monkeypatch):
    """The shrink outline built here reads the dark stage's own crops."""
    ch, cells = _world(shape=(24, 40, 40))
    seen = []
    real = R._region_crop

    def spy(array, roi, shape, *, what):
        seen.append(what)
        return real(array, roi, shape, what=what)

    monkeypatch.setattr(R, "_region_crop", spy)
    out = _run(_compete_spec(outline="shrink", whole_patch=True),
               np.zeros_like(cells), ch, seed_layer=cells)
    assert out.ok
    for name in ch:
        assert seen.count("dark channel {!r}".format(name)) == 1


def _surface_box(cells):
    """A thin box (6 voxels in y) at the tissue's surface: y 21-27 crosses
    the stain's edge near y = 24 in the (24, 40, 40) world."""
    return (slice(0, cells.shape[0]), slice(21, 27), slice(0, cells.shape[2]))


def test_a_box_never_builds_its_own_shrink_outline():
    """A box with ``outline="shrink"`` and no outline handed in is refused
    in plain words with its own reason -- under either growth rule, and in
    the box-info count too -- instead of building a per-box outline that
    differs from the patch's at the box faces."""
    ch, cells = _world(shape=(24, 40, 40))
    labels = np.zeros_like(cells)
    roi = _surface_box(cells)
    for spec in (_spec(outline="shrink"), _compete_spec(outline="shrink")):
        out = _run(spec, labels, ch, seed_layer=cells, roi=roi)
        rep = R.find_region_seeds(
            labels, {}, spec, roi=roi, dark_channels=ch, seed_layer=cells,
            dark_threshold=_threshold(ch), spacing=ISO)
        assert out.status == "refused"
        assert out.refusal_reason == rep.refusal_reason == "outline-per-box"
        assert out.messages == [R.ADD_ENGINE_OUTLINE_PER_BOX]
        assert R.ENGINE_REFUSALS["outline-per-box"] == R.ADD_ENGINE_OUTLINE_PER_BOX
        assert "whole patch" in R.ADD_ENGINE_OUTLINE_PER_BOX


def test_a_box_uses_the_cropped_patch_outline_as_is():
    """The patch-wide outline, cropped, is the box's outline voxel for
    voxel -- the full patch volume handed in and its crop handed in give
    the same answer, and nothing new lands outside the crop."""
    from packages.epicycle_annotator.reshape import outline as O

    ch, cells = _world(shape=(24, 40, 40))
    labels = np.zeros_like(cells)
    patch = O.tissue_outline(ch, ISO, dark_factor=0.5, bridge_um=10.0,
                             dark_threshold=_threshold(ch))
    roi = _surface_box(cells)
    spec = _compete_spec(outline="shrink")
    whole = _run(spec, labels, ch, seed_layer=cells, roi=roi, outline=patch,
                 keep_intermediates=True)
    crop = _run(spec, labels, ch, seed_layer=cells, roi=roi,
                outline=patch[roi])
    assert whole.ok and crop.ok
    assert np.array_equal(whole.result, crop.result)
    assert not (whole.result[~patch[roi]] > 0).any()
    # used AS IS: a box-built outline would differ at the box faces
    boxed = O.tissue_outline({k: v[roi] for k, v in ch.items()}, ISO,
                             dark_factor=0.5, bridge_um=10.0,
                             dark_threshold=_threshold(ch))
    assert not np.array_equal(boxed, patch[roi])
    assert whole.unfilled_tissue_fraction == pytest.approx(
        R._unfilled_in_outline(whole.result, labels[roi], patch[roi]))


def test_the_outline_dropping_every_marker_has_its_own_reason():
    """When the outline drops the last markers the refusal is
    ``no-markers-outline`` (C2 gives it a sentence); when nothing reached
    the outline drop it stays ``no-markers``."""
    ch, cells = _world()
    labels = np.zeros_like(cells)
    nothing = np.zeros(cells.shape, bool)
    for spec in (_spec(outline="cleared"), _compete_spec(outline="cleared")):
        out = _run(spec, labels, ch, seed_layer=cells, outline=nothing)
        rep = R.find_region_seeds(
            labels, {}, spec, roi=_box(cells.shape), dark_channels=ch,
            seed_layer=cells, dark_threshold=_threshold(ch), spacing=ISO,
            outline=nothing)
        assert out.status == "refused"
        assert out.refusal_reason == rep.refusal_reason == "no-markers-outline"
        assert out.n_outside_outline >= 1
        assert out.messages == [R.ADD_ENGINE_NO_MARKERS_OUTLINE]
    # the one marker dropped as dark first: the outline never saw one
    dark_only = np.zeros_like(cells)
    dark_only[4:8, 30:36, 14:20] = 300           # in the open dark space
    out = _run(_compete_spec(outline="cleared"), labels, ch,
               seed_layer=dark_only, outline=np.ones(cells.shape, bool))
    assert out.status == "refused"
    assert (out.n_dropped_dark, out.n_outside_outline) == (1, 0)
    assert out.refusal_reason == "no-markers"
    assert out.messages == [R.ADD_ENGINE_NO_MARKERS]


def test_run_cells_maps_the_outline_refusal_through():
    mem, nuc, cells = _tissue()
    chans = {"membrane": mem, "nuclei": nuc}
    out = AP._run_cells(
        AP.AddPreviewResult(roi=_box(cells.shape), preset="cells"),
        roi=_box(cells.shape), existing_roi=np.zeros_like(cells),
        next_label=1000, spacing=ISO,
        params={"source": "labels", "growth": "compete", "outline": "cleared",
                "auto_smooth": False, "clear_border": False},
        nuclei_roi=None, wall_roi=mem, dark_rois=chans,
        run_guide_roi=cells.copy(), click_points=(),
        dark_threshold=_threshold(chans), id_max=None, cancel_event=None,
        progress=None, outline_roi=np.zeros(cells.shape, bool))
    assert out.status == "refused"
    assert out.refusal_reason == "no-markers-outline"
    # the page says it in its own words (what to change -- on the outline
    # of the objects you cleared, only another outline); the engine's
    # sentence goes to the log
    assert out.refusal == AP.ADD_S1_NO_MARKERS_OUTLINE_CLEARED
    assert any(R.ADD_ENGINE_NO_MARKERS_OUTLINE in m for m in out.messages)


def test_the_background_is_never_a_piece_of_the_outline():
    """At min_piece_um3=0 every piece is kept and the
    background is not (its size is -1, below any minimum)."""
    from packages.epicycle_annotator.reshape import outline as O

    mask = np.zeros((4, 10, 10), bool)
    mask[1:3, 2:5, 2:5] = True
    mask[0, 8, 8] = True                       # a one-voxel piece
    assert np.array_equal(O._keep_big(mask, ISO, 0.0), mask)
    assert np.array_equal(O._keep_big(mask, ISO, 2.0),
                          mask & ~(np.indices(mask.shape)[0] == 0))
    assert not O._keep_big(np.zeros_like(mask), ISO, 0.0).any()


def test_the_unfilled_share_is_measured_against_the_outline():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    outline = np.zeros(cells.shape, bool)
    outline[:, :24, :] = True
    out = _run(_compete_spec(outline="cleared"), labels, ch, seed_layer=cells,
               outline=outline)
    tissue = (labels == 0) & outline
    expect = (tissue & (out.result == 0)).sum() / tissue.sum()
    assert out.unfilled_tissue_fraction == pytest.approx(expect)


def test_a_whole_volume_outline_is_cut_to_the_box():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    outline = np.zeros(cells.shape, bool)
    outline[:, :20, :] = True
    roi = (slice(0, 12), slice(0, 24), slice(0, 40))
    crop = {k: v[roi] for k, v in ch.items()}
    whole = R.reshape_empty_region(
        labels, {"membrane": ch["membrane"]},
        _compete_spec(outline="cleared"), roi=roi,
        dark_channels=ch, seed_layer=cells, dark_threshold=_threshold(ch),
        spacing=ISO, outline=outline)
    boxed = R.reshape_empty_region(
        labels[roi], {"membrane": ch["membrane"][roi]},
        _compete_spec(outline="cleared"), roi=roi,
        dark_channels=crop, seed_layer=cells[roi],
        dark_threshold=_threshold(ch), spacing=ISO, outline=outline[roi])
    assert whole.ok and boxed.ok
    assert np.array_equal(whole.result, boxed.result)


# --------------------------------------------------------------------------- #
# the flood with an outline: hooks A, B and D only
# --------------------------------------------------------------------------- #
def test_the_flood_without_an_outline_is_unchanged_by_the_new_knobs():
    """Every new knob at a non-default value but the growth rule and the
    outline: the flood's answer is byte for byte the default one."""
    ch, cells = _world()
    labels = np.zeros_like(cells)
    labels[:, :12, :10] = 3
    a = _run(_spec(), labels, ch, seed_layer=cells)
    b = _run(_spec(growth="flood", wall_cost=7.0, growth_limit=3.0,
                   include_membrane=False, bridge_um=22.0), labels, ch,
             seed_layer=cells)
    assert np.array_equal(a.result, b.result)
    assert np.array_equal(a.seeds, b.seeds)
    assert a.n_tiles == b.n_tiles == 0


def test_the_flood_with_an_outline_stays_inside_it():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    outline = np.ones(cells.shape, bool)
    outline[:, :12, :10] = False     # cell 101 wholly outside: dropped (A)
    outline[:, 13:24, 36:40] = False  # part of cell 108 outside: cut (B, D)
    out = _run(_spec(outline="cleared"), labels, ch, seed_layer=cells,
               outline=outline)
    plain = _run(_spec(), labels, ch, seed_layer=cells)
    assert out.ok
    assert out.n_outside_outline == 1
    assert not (out.result[~outline] > 0).any()
    assert (plain.result[~outline] > 0).sum() > 100
    assert out.n_tiles == 0


def test_hook_b_makes_the_outside_of_the_outline_an_outside_marker():
    """Hook B: with an outline the flood's outside marker also covers the
    empty space outside it (not eroded), so a cell beside the outline's
    edge is held back by it, not only clipped afterwards (hook D)."""
    ch, cells = _world()
    labels = np.zeros_like(cells)
    outline = np.ones(cells.shape, bool)
    outline[:, :, 36:] = False
    out = _run(_spec(outside="faces", outline="cleared"),
               labels, ch, seed_layer=cells, keep_intermediates=True,
               outline=outline)
    markers = out.intermediates["markers"]
    n = out.n_seeds
    empty_out = (~outline) & (labels == 0) & (out.seeds == 0)
    assert (markers[empty_out] == n + 1).all()


# --------------------------------------------------------------------------- #
# the whole patch: the budget exemption
# --------------------------------------------------------------------------- #
def test_the_whole_patch_skips_the_box_budget():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    budget = int(cells.size) - 1
    over = _run(_compete_spec(box_budget=budget), labels, ch, seed_layer=cells)
    whole = _run(_compete_spec(box_budget=budget, whole_patch=True), labels, ch,
                 seed_layer=cells)
    assert over.status == "refused" and over.refusal_reason == "too-big"
    assert whole.ok


def test_find_region_seeds_skips_the_budget_on_the_whole_patch():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    budget = int(cells.size) - 1
    kw = dict(roi=_box(cells.shape), dark_channels=ch, seed_layer=cells,
              dark_threshold=_threshold(ch), spacing=ISO)
    over = R.find_region_seeds(labels, {}, _compete_spec(box_budget=budget), **kw)
    whole = R.find_region_seeds(
        labels, {}, _compete_spec(box_budget=budget, whole_patch=True), **kw)
    assert over.status == "refused"
    assert whole.ok and whole.n_in_region == 8


def _box_info(params):
    mem, nuc, cells = _tissue()
    chans = {"membrane": mem, "nuclei": nuc}
    p = {"preset": "cells", "dark_outside": True, "box_budget": int(cells.size) - 1}
    p.update(params)
    return AP.add_box_info(
        existing_roi=np.zeros_like(cells), dark_sources=chans, dark_rois=chans,
        spacing=ISO, stride=1, dark_cache=None, source="labels",
        run_guide_roi=cells.copy(), nuclei_roi=nuc, params=p)


def test_the_box_info_is_not_over_the_limit_on_the_whole_patch():
    boxed = _box_info({})
    whole = _box_info({"whole_patch": True, "growth": "compete"})
    assert boxed.over_limit is True and boxed.run_count is None
    assert whole.over_limit is False
    assert whole.run_count == (8, 8)


def test_the_box_info_takes_every_cells_whole_patch():
    """A Cells whole patch is filled to the walls whatever the hidden Grow
    by says, so the box-info worker lifts the box limit for it with either
    pick and counts the run's markers; it never raises."""
    for growth in ("flood", "compete"):
        info = _box_info({"whole_patch": True, "growth": growth})
        assert info.status == "ok"
        assert info.over_limit is False, growth
        assert info.run_count == (8, 8), growth


def test_the_box_info_never_raises_on_a_refused_knob():
    """A knob the spec refuses (an unknown growth rule, a cleared outline
    with no volume) leaves the run count out instead of raising."""
    for params in ({"growth": "spread", "box_budget": 10 ** 9},
                   {"outline": "cleared", "box_budget": 10 ** 9}):
        info = _box_info(params)
        assert info.status == "ok"
        assert info.run_count is None


def test_the_box_info_count_follows_the_outline():
    """The run count drops the markers the Preview's outline drops."""
    mem, nuc, cells = _tissue()
    outline = np.ones(cells.shape, bool)
    outline[:, :12, :10] = False     # cell 101 wholly outside
    chans = {"membrane": mem, "nuclei": nuc}
    info = AP.add_box_info(
        existing_roi=np.zeros_like(cells), dark_sources=chans, dark_rois=chans,
        spacing=ISO, stride=1, dark_cache=None, source="labels",
        run_guide_roi=cells.copy(), nuclei_roi=nuc,
        params={"preset": "cells", "dark_outside": False, "growth": "compete",
                "outline": "cleared"}, outline=outline)
    plain = _box_info({"box_budget": 10 ** 9})
    assert plain.run_count == (8, 8)
    assert info.run_count == (8, 7)
    shrink = _box_info({"box_budget": 10 ** 9, "outline": "shrink"})
    assert shrink.status == "ok" and shrink.run_count is None



# --------------------------------------------------------------------------- #
# the Add worker hands the knobs over
# --------------------------------------------------------------------------- #
def _preview(params, **kw):
    mem, nuc, cells = _tissue()
    chans = {"membrane": mem, "nuclei": nuc}
    p = {"source": "labels", "dark_outside": True, "auto_smooth": False,
         "clear_border": False}
    p.update(params)
    return AP.run_add_preview(
        preset="cells", roi=_box(cells.shape), existing_roi=np.zeros_like(cells),
        next_label=1000, spacing=ISO, params=p, wall_roi=mem, dark_rois=chans,
        run_guide_roi=cells.copy(), dark_threshold=_threshold(chans), **kw)


def test_run_cells_hands_the_growth_knobs_to_the_engine(monkeypatch):
    seen = {}
    real = R.reshape_empty_region

    def spy(*a, **k):
        seen["spec"] = a[2]
        seen["kw"] = k
        return real(*a, **k)

    monkeypatch.setattr(R, "reshape_empty_region", spy)
    out = _preview({"growth": "compete", "wall_cost": 25.0, "growth_limit": 9.0,
                    "include_membrane": False})
    assert out.ok
    p = seen["spec"].params
    assert (p["growth"], p["wall_cost"], p["growth_limit"],
            p["include_membrane"]) == ("compete", 25.0, 9.0, False)
    assert p["outline"] == "none" and p["whole_patch"] is False
    assert out.n_slivers == 0 and out.n_outside_outline == 0
    assert out.n_tiles == 1


def test_run_cells_at_defaults_is_the_flood(monkeypatch):
    seen = {}
    real = R.reshape_empty_region

    def spy(*a, **k):
        seen["spec"] = a[2]
        return real(*a, **k)

    monkeypatch.setattr(R, "reshape_empty_region", spy)
    out = _preview({})
    assert out.ok
    assert seen["spec"].params["growth"] == "flood"
    assert seen["spec"].params["outline"] == "none"
    assert out.n_tiles == 0


def test_run_cells_with_an_outline_forces_the_dark_inputs(monkeypatch):
    seen = {}
    real = R.reshape_empty_region

    def spy(*a, **k):
        seen["spec"] = a[2]
        seen["kw"] = k
        return real(*a, **k)

    monkeypatch.setattr(R, "reshape_empty_region", spy)
    mem, nuc, cells = _tissue()
    outline = np.ones(cells.shape, bool)
    out = AP._run_cells(
        AP.AddPreviewResult(roi=_box(cells.shape), preset="cells"),
        roi=_box(cells.shape), existing_roi=np.zeros_like(cells),
        next_label=1000, spacing=ISO,
        params={"source": "labels", "dark_outside": False, "growth": "compete",
                "outline": "cleared", "auto_smooth": False,
                "clear_border": False},
        nuclei_roi=None, wall_roi=mem,
        dark_rois={"membrane": mem, "nuclei": nuc},
        run_guide_roi=cells.copy(), click_points=(),
        dark_threshold=_threshold({"membrane": mem, "nuclei": nuc}),
        id_max=None, cancel_event=None, progress=None,
        outline_roi=outline)
    assert out.ok
    assert seen["spec"].params["outside"] == "faces+dark"
    assert seen["kw"]["dark_channels"] is not None


# --------------------------------------------------------------------------- #
# the warm-up
# --------------------------------------------------------------------------- #
def _cold(monkeypatch):
    monkeypatch.setattr(R, "_WARM", [False])
    monkeypatch.setattr(R, "_WARM_PARTS", set())


def test_the_warm_up_announces_once_and_only_a_slow_set_up(monkeypatch):
    """The sentence is announced only when the set-up is still running
    after WARM_ANNOUNCE_AFTER_S (a compile), once, on the calling thread;
    a second warm-up is free and silent."""
    import threading
    import time

    _cold(monkeypatch)
    real = R._warm_body
    monkeypatch.setattr(R, "WARM_ANNOUNCE_AFTER_S", 0.05)
    monkeypatch.setattr(R, "_warm_body", lambda: (time.sleep(0.4), real()))
    calls = []
    took = R.warm_up_compete(
        announce=lambda: calls.append(threading.current_thread()))
    assert calls == [threading.current_thread()]
    assert took >= 0.4
    assert R.compete_is_warm()
    assert R.warm_up_compete(announce=lambda: calls.append("again")) == 0.0
    assert len(calls) == 1


def test_a_fast_set_up_is_not_announced(monkeypatch):
    _cold(monkeypatch)
    monkeypatch.setattr(R, "WARM_ANNOUNCE_AFTER_S", 30.0)
    calls = []
    R.warm_up_compete(announce=lambda: calls.append(1))
    assert calls == []
    assert R.compete_is_warm()


def test_a_failed_set_up_raises_and_stays_cold(monkeypatch):
    _cold(monkeypatch)

    def broken():
        raise ImportError(C.NUMBA_MISSING)

    monkeypatch.setattr(R, "_warm_body", broken)
    with pytest.raises(ImportError):
        R.warm_up_compete()
    assert not R.compete_is_warm()


def test_the_first_real_solve_warms_the_solver(monkeypatch):
    """A real competition that ran both loops leaves the process
    warm (no second set-up); without the membrane step it is not yet."""
    ch, cells = _world()
    labels = np.zeros_like(cells)
    _cold(monkeypatch)
    assert _run(_compete_spec(include_membrane=False), labels, ch,
                seed_layer=cells).ok
    assert not R.compete_is_warm()
    assert _run(_compete_spec(), labels, ch, seed_layer=cells).ok
    assert R.compete_is_warm()
    calls = []
    assert R.warm_up_compete(announce=lambda: calls.append(1)) == 0.0
    assert calls == []

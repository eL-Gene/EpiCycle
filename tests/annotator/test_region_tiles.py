"""The engine's competition in tiles equals one call.

The Add page no longer sends a whole patch this way (a Cells whole patch is
filled to the walls with a background marker); the engine keeps the tiled
path, and the page's Spread on a box runs through it.

Through ``reshape_empty_region(growth="compete", whole_patch=True)``: the
elevation is built (or handed in) ONCE for the whole patch and every
marker decision is made once, so cutting the growth into tiles changes
nothing -- the answer is the one-tile answer voxel for voxel; every voxel
is in exactly one tile core; every id is one object; and a Stop between
tiles writes nothing.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from scipy import ndimage as ndi

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import ReshapeCancelled, ReshapeSpec
from packages.epicycle_annotator.reshape import compete as C
from packages.epicycle_annotator.reshape import region as R

from .test_region_flood_golden import tissue

ISO = (1.0, 1.0, 1.0)
ANISO = (2.0, 1.0, 1.0)

#: A small growth limit keeps the halo small, so the 26 x 60 x 60 tissue
#: needs several tiles at a tiny tile budget.
LIMIT = 4.0


def _spec(**params):
    base = {"wall_channel": "membrane", "ridge": False, "outside": "faces+dark",
            "growth": "compete", "whole_patch": True, "growth_limit": LIMIT,
            "wall_cost": 40.0, "box_budget": 1000}
    base.update(params)
    return ReshapeSpec(target="region", seeds="labels", walls="membrane",
                       params=base)


def _world(seed=0):
    nuc, mem, labels, run, _pieces, _clicks = tissue(seed)
    return {"nuc": nuc, "mem": mem}, labels, run


def _run(spacing=ISO, *, tile_budget=None, elevation=None, outline=None,
         seed=0, **kw):
    ch, labels, run = _world(seed)
    threshold = R.patch_dark_threshold(ch, spacing=spacing, stride=1)
    params = {}
    if outline is not None:
        params = {"outline": "cleared"}
    return R.reshape_empty_region(
        labels, {"membrane": ch["mem"]}, _spec(**params),
        roi=tuple(slice(0, n) for n in labels.shape), dark_channels=ch,
        seed_layer=run, dark_threshold=threshold, spacing=spacing,
        elevation=elevation, tile_budget=tile_budget, outline=outline, **kw)


@pytest.mark.parametrize("spacing", [ISO, ANISO])
@pytest.mark.parametrize("with_outline", [False, True])
def test_tiles_equal_one_call_voxel_for_voxel(spacing, with_outline):
    outline = None
    if with_outline:
        _ch, labels, _run_ = _world()
        outline = np.zeros(labels.shape, bool)
        outline[:, :40, :] = True
    one = _run(spacing, outline=outline)
    assert one.ok and one.n_tiles == 1
    # the patch-wide elevation, built once, handed in
    tiled = _run(spacing, tile_budget=1, elevation=one.elevation, outline=outline)
    assert tiled.ok
    assert tiled.n_tiles > 1
    assert tiled.n_tiles == len(C.plan_tiles(one.result.shape, spacing, LIMIT, 1))
    assert np.array_equal(tiled.result, one.result)
    assert np.array_equal(tiled.seeds, one.seeds)
    assert tiled.n_seeds == one.n_seeds and tiled.n_pieces == one.n_pieces
    assert tiled.unfilled_tissue_fraction == one.unfilled_tissue_fraction


def test_each_voxel_is_in_exactly_one_tile_core():
    shape = (26, 60, 60)
    plan = C.plan_tiles(shape, ISO, LIMIT, 1)
    assert len(plan) > 1
    hits = np.zeros(shape, np.int32)
    for core, _halo in plan:
        hits[core] += 1
    assert (hits == 1).all()


def test_each_id_is_one_object_and_only_new_ids():
    out = _run(tile_budget=1)
    assert out.ok
    res = out.result
    _ch, labels, _r = _world()
    assert not (res[labels != 0] > 0).any()
    ids = [int(v) for v in np.unique(res) if v]
    assert len(ids) >= 8
    assert max(ids) <= out.n_seeds
    structure = ndi.generate_binary_structure(3, 1)
    for i in ids:
        _lab, n = ndi.label(res == i, structure=structure)
        assert n == 1, i


def test_the_pieces_are_reported_one_by_one():
    seen = []
    out = _run(tile_budget=1, progress=lambda *a: seen.append(a))
    assert out.ok
    pieces = [a for a in seen if a[0] == "add_piece"]
    k = out.n_tiles
    assert [a[1] for a in pieces] == list(range(1, k + 1))
    assert all(a[2] == k for a in pieces)
    stages = [a[0] for a in seen if a[0] != "add_piece"]
    assert stages == list(R.REGION_STAGES_COMPETE)


def test_a_stop_between_tiles_raises_and_writes_nothing():
    """I8: a Stop set after the second tile raises before the third; the
    labels handed in are untouched."""
    ch, labels, run = _world()
    before = labels.copy()
    ev = threading.Event()
    calls = []

    def progress(stage, i, n):
        if stage == "add_piece":
            calls.append(i)
            if i == 2:
                ev.set()

    with pytest.raises(ReshapeCancelled):
        R.reshape_empty_region(
            labels, {"membrane": ch["mem"]}, _spec(),
            roi=tuple(slice(0, n) for n in labels.shape), dark_channels=ch,
            seed_layer=run,
            dark_threshold=R.patch_dark_threshold(ch, spacing=ISO, stride=1),
            spacing=ISO, tile_budget=1, cancel_event=ev, progress=progress)
    assert calls == [1, 2]
    assert np.array_equal(labels, before)


def test_a_stop_raised_by_the_progress_callback_propagates():
    def progress(stage, i, n):
        if stage == "add_piece" and i == 1:
            raise ReshapeCancelled()

    with pytest.raises(ReshapeCancelled):
        _run(tile_budget=1, progress=progress)


def test_a_broken_progress_callback_does_not_stop_the_tiles():
    def progress(stage, i, n):
        raise RuntimeError("a broken bar")

    assert _run(tile_budget=1, progress=progress).ok


def test_a_stop_in_the_add_worker_answers_cancelled():
    """The Add worker sends Spread in tiles from a box (a Cells whole patch
    is filled to the walls instead): a Stop at a tile is its cancelled
    answer."""
    ch, labels, run = _world()
    ev = threading.Event()

    def progress(stage, i, n):
        if stage == "add_piece":
            ev.set()

    out = AP.run_add_preview(
        preset="cells", roi=tuple(slice(0, n) for n in labels.shape),
        existing_roi=labels, next_label=1000, spacing=ISO,
        params={"source": "labels", "dark_outside": True, "growth": "compete",
                "growth_limit": LIMIT, "auto_smooth": False,
                "clear_border": False},
        wall_roi=ch["mem"], dark_rois=ch, run_guide_roi=run,
        dark_threshold=R.patch_dark_threshold(ch, spacing=ISO, stride=1),
        cancel_event=ev, progress=progress)
    assert out.status == "cancelled"
    assert out.result is None

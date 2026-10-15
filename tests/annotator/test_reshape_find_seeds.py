"""``find_seeds`` -- the marker-only half of the engine (Brick 2a).

The panel needs to SHOW the markers before it cuts anything, so the seed
stage got its own entry point.  The one thing that must be true of it is
**equivalence**: the markers a Find shows have to be the markers a Preview
then uses, or the cherry-pick ticks select the wrong things.

So every test here compares ``find_seeds(..., roi=R).seeds`` against
``reshape_region(..., roi=R, keep_seed_ids=None).seeds`` -- the same crop,
the same detector, the same spacing -- rather than asserting anything about
what a detector "should" find.

Pure numpy; no Qt, no napari.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.reshape import (
    ReshapeSpec,
    SeedReport,
    find_seeds,
    reshape_region,
)

from .reshape_fixtures import two_columns

#: The six image-reading detectors, in the panel's combo order.
IMAGE_DETECTORS = ("peaks", "hmax", "blobs_log", "blobs_dog", "threshcc", "hmin")

SPACINGS = ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0))


def _case():
    """One walled two-column cell plus a nuclei-ish channel."""
    labels, guide, membrane, region = two_columns(0.30)
    nuclei = np.zeros_like(membrane)
    # two bright blobs, one per column, well clear of the wall plane
    nuclei[4:8, 12:17, 12:20] = 1.0
    nuclei[4:8, 23:28, 12:20] = 1.0
    nuclei = nuclei * region
    return labels, guide, membrane, nuclei


def _spec(seeds, *, walls="membrane", **params):
    p = {"wall_channel": "membrane"}
    if seeds in IMAGE_DETECTORS:
        p["seed_channel"] = "nuclei"
    p.update(params)
    return ReshapeSpec(target="label", seeds=seeds, walls=walls, params=p)


def _roi_of(labels, target_id=7, margin=4):
    from packages.epicycle_annotator.batch_refine import compute_label_roi

    return compute_label_roi(labels, target_id, margin)


# --------------------------------------------------------------------------- #
# THE equivalence lock
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spacing", SPACINGS)
@pytest.mark.parametrize("detector", IMAGE_DETECTORS)
def test_find_seeds_matches_the_full_engine_for_every_image_detector(
        detector, spacing):
    labels, _guide, membrane, nuclei = _case()
    images = {"membrane": membrane, "nuclei": nuclei}
    spec = _spec(detector)
    roi = _roi_of(labels)

    found = find_seeds(labels, images, spec, target_id=7, roi=roi,
                       spacing=spacing)
    solved = reshape_region(labels, images, spec, target_id=7, roi=roi,
                            spacing=spacing, keep_seed_ids=None)

    assert found.seeds is not None
    assert np.array_equal(found.seeds, solved.seeds), (
        "the Find and the Preview would place different markers for {!r} at "
        "spacing {}".format(detector, spacing)
    )


@pytest.mark.parametrize("spacing", SPACINGS)
def test_find_seeds_matches_the_full_engine_for_the_shape_detector(spacing):
    """``distance`` reads the WALL elevation, so it is the one detector that
    would silently diverge if the two entry points built the wall
    differently (engine.py's seed stage takes ``elevation=``)."""
    labels, _guide, membrane, _nuclei = _case()
    images = {"membrane": membrane}
    spec = _spec("distance")
    roi = _roi_of(labels)

    found = find_seeds(labels, images, spec, target_id=7, roi=roi,
                       spacing=spacing)
    solved = reshape_region(labels, images, spec, target_id=7, roi=roi,
                            spacing=spacing, keep_seed_ids=None)

    assert np.array_equal(found.seeds, solved.seeds)


def test_a_trusted_mask_gives_the_same_markers_too():
    labels, guide, membrane, _nuclei = _case()
    spec = _spec("labels")
    roi = _roi_of(labels)

    found = find_seeds(labels, {"membrane": membrane}, spec, target_id=7,
                       roi=roi, seed_layer=guide)
    solved = reshape_region(labels, {"membrane": membrane}, spec, target_id=7,
                            roi=roi, seed_layer=guide, keep_seed_ids=None)

    assert np.array_equal(found.seeds, solved.seeds)


def test_the_cherry_pick_filter_is_NOT_applied_here():
    """``keep_seed_ids`` narrows what the SOLVE uses; the picker has to see
    everything that was found, or a marker can never be ticked back on."""
    labels, guide, membrane, _nuclei = _case()
    spec = _spec("labels")
    roi = _roi_of(labels)

    found = find_seeds(labels, {"membrane": membrane}, spec, target_id=7,
                       roi=roi, seed_layer=guide)
    narrowed = reshape_region(labels, {"membrane": membrane}, spec,
                              target_id=7, roi=roi, seed_layer=guide,
                              keep_seed_ids=[1])

    assert found.n_in_region == 2
    assert int((np.unique(narrowed.seeds) != 0).sum()) == 1


# --------------------------------------------------------------------------- #
# what the report carries
# --------------------------------------------------------------------------- #
def test_the_report_names_every_marker_with_its_size_and_centre():
    labels, guide, membrane, _nuclei = _case()
    roi = _roi_of(labels)

    found = find_seeds(labels, {"membrane": membrane}, _spec("labels"),
                       target_id=7, roi=roi, seed_layer=guide)

    assert isinstance(found, SeedReport)
    assert found.status == "ok"
    assert found.roi == roi
    assert [m["id"] for m in found.markers] == [1, 2]
    for marker in found.markers:
        assert marker["voxels"] > 0
        assert len(marker["centroid_zyx"]) == 3
    assert found.seed_source, "the report says where the markers came from"


def test_the_centroids_are_in_VOLUME_coordinates():
    """The ROI offset is applied once, here, so no caller has to redo it."""
    labels, guide, membrane, _nuclei = _case()
    roi = _roi_of(labels)

    found = find_seeds(labels, {"membrane": membrane}, _spec("labels"),
                       target_id=7, roi=roi, seed_layer=guide)

    ys = [m["centroid_zyx"][1] for m in found.markers]
    truth = [float(np.argwhere(guide == sid)[:, 1].mean())
             for sid in (1, 2)]
    assert ys == pytest.approx(truth, abs=1e-6)


def test_markers_outside_the_cell_are_reported_and_dropped():
    labels, guide, membrane, _nuclei = _case()
    guide = guide.copy()
    # inside the ROI crop, outside the cell's footprint
    guide[0, 4, 6] = 44
    roi = _roi_of(labels)

    found = find_seeds(labels, {"membrane": membrane}, _spec("labels"),
                       target_id=7, roi=roi, seed_layer=guide)

    assert found.n_found == 3
    assert found.n_in_region == 2
    assert any("outside" in m for m in found.messages)


def test_a_roi_is_computed_when_none_is_given():
    labels, guide, membrane, _nuclei = _case()
    found = find_seeds(labels, {"membrane": membrane}, _spec("labels"),
                       target_id=7, seed_layer=guide, margin=4)
    assert found.roi == _roi_of(labels, margin=4)


# --------------------------------------------------------------------------- #
# refusals are ANSWERS
# --------------------------------------------------------------------------- #
def test_zero_markers_is_a_refusal_with_a_plain_sentence():
    labels, _guide, membrane, _nuclei = _case()
    empty = np.zeros_like(labels)

    found = find_seeds(labels, {"membrane": membrane}, _spec("labels"),
                       target_id=7, roi=_roi_of(labels), seed_layer=empty)

    assert found.status == "refused"
    assert found.refusal_reason == "no-seeds"
    assert found.messages and found.messages[-1].endswith(")")
    assert found.n_in_region == 0


def test_a_label_that_is_not_here_refuses_rather_than_raising():
    labels, _guide, membrane, _nuclei = _case()
    found = find_seeds(labels, {"membrane": membrane}, _spec("labels"),
                       target_id=999, seed_layer=labels)
    assert found.status == "refused"
    assert found.refusal_reason == "no-target"


def test_an_image_detector_with_no_channel_named_raises_the_engines_error():
    """Same ValueError the solve would raise -- the panel refuses before it
    gets here, but the seam must not invent a fallback channel."""
    labels, _guide, membrane, nuclei = _case()
    spec = ReshapeSpec(target="label", seeds="peaks", walls="membrane",
                       params={"wall_channel": "membrane"})
    with pytest.raises(ValueError):
        find_seeds(labels, {"membrane": membrane, "nuclei": nuclei}, spec,
                   target_id=7, roi=_roi_of(labels))


# --------------------------------------------------------------------------- #
# the documented coupling: `distance` markers READ the wall
# --------------------------------------------------------------------------- #
def _box(shape=(10, 20, 20), label_id=5):
    vol = np.zeros(shape, np.int32)
    vol[2:8, 4:16, 4:16] = label_id
    return vol


@pytest.mark.parametrize("walls,expected", [
    ("none", 0),
    ("skeleton", 1),
    ("distance", 1),
])
def test_shape_markers_need_a_shape_wall(walls, expected):
    """Rule 6, measured.  The engine feeds the WALL elevation to
    the ``distance`` detector, so a flat (``none``) elevation has no maxima
    at all and the panel must not offer that pair."""
    vol = _box()
    spec = ReshapeSpec(target="label", seeds="distance", walls=walls)
    found = find_seeds(vol, {}, spec, target_id=5, spacing=(1, 1, 1))
    assert found.n_in_region == expected
    if expected == 0:
        assert found.status == "refused"


def test_the_none_wall_pairing_refuses_in_the_solve_too():
    """Belt and braces: the panel disables the pair, and if it ever slipped
    through the engine still answers rather than cutting blind."""
    vol = _box()
    spec = ReshapeSpec(target="label", seeds="distance", walls="none")
    res = reshape_region(vol, {}, spec, target_id=5)
    assert res.status == "refused"
    assert res.refusal_reason == "no-seeds"


# --------------------------------------------------------------------------- #
# housekeeping
# --------------------------------------------------------------------------- #
def test_find_seeds_never_mutates_its_inputs():
    labels, guide, membrane, nuclei = _case()
    before = (labels.copy(), guide.copy(), membrane.copy(), nuclei.copy())
    find_seeds(labels, {"membrane": membrane, "nuclei": nuclei},
               _spec("peaks"), target_id=7, roi=_roi_of(labels))
    assert np.array_equal(labels, before[0])
    assert np.array_equal(guide, before[1])
    assert np.array_equal(membrane, before[2])
    assert np.array_equal(nuclei, before[3])


def test_a_spec_that_is_not_a_ReshapeSpec_is_a_TypeError():
    with pytest.raises(TypeError):
        find_seeds(_box(), {}, {"seeds": "distance"}, target_id=5)


def test_the_seam_is_exported_eagerly_from_the_package():
    import packages.epicycle_annotator.reshape as R

    assert "find_seeds" in R.__all__
    assert "SeedReport" in R.__all__
    assert R.find_seeds is find_seeds

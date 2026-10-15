"""R1c in the batch path — an empty result must NOT erase the object.

Pure pytest — no Qt / napari (except the one Add byte-identity case, which
imports the annotator module for its unbound ``_smooth_add_result``).

invariants 2 and 3.

``apply_smooth_to_volume`` clears the old footprint before it writes the new
one, so an object whose result has no voxel left to write used to be
DELETED and reported as a change.  Two ways in:

* the step empties the crop (a negative dilate radius larger than a thin
  object's half-thickness), and
* every voxel of a non-empty result sits on a LOWER id already written this
  batch (lower-id-wins at write time).

With the default ``empty="skip"`` such an object is not written, is not in
``touched_ids``, adds nothing to the counts or the union ROI, and is listed
in ``BatchResult.emptied_ids``.  ``empty="erase"`` is the old loop verbatim,
kept for Add's own smoothing (a candidate piece its smoothing empties is
dropped, as it always was).
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import batch_refine as br
from packages.epicycle_annotator.batch_refine import (
    RefineStep,
    apply_recipe_to_label,
    run_batch_recipe,
)
from packages.epicycle_annotator.smooth import apply_smooth_to_volume


# --------------------------------------------------------------------------- #
# The frozen pre-R1c apply loop (d1cdc6bb) — the reference for "erase".
# --------------------------------------------------------------------------- #
def _old_run(volume, ids, recipe):
    """``run_batch_recipe`` as it was at ``d1cdc6bb``, serial, no cancel --
    except ``n_total``, now the actually-changed count (the geometry-path
    count fix; the old loop summed pre-contention per-object counts)."""
    snapshot = volume
    working = volume.copy()
    margin = br._recipe_margin(recipe, volume.shape)
    results = []
    for lid in sorted(int(i) for i in ids):
        roi = br.compute_label_roi(snapshot, lid, margin)
        if roi is None:
            results.append((lid, None, None, 0))
            continue
        result_roi, n_changed = br.apply_recipe_to_label(snapshot, lid, roi, recipe)
        results.append((lid, roi, result_roi, n_changed))
    touched, n_total, u0, u1 = [], 0, None, None
    for lid, roi, result_roi, n_changed in results:
        if roi is None:
            continue
        applied = apply_smooth_to_volume(working, roi, result_roi, lid)
        if n_changed > 0 or applied > 0:
            touched.append(lid)
            n_total += n_changed
            s = [x.start for x in roi]
            e = [x.stop for x in roi]
            if u0 is None:
                u0, u1 = s, e
            else:
                u0 = [min(a, b) for a, b in zip(u0, s)]
                u1 = [max(a, b) for a, b in zip(u1, e)]
    union = tuple(slice(a, b) for a, b in zip(u0, u1)) if u0 is not None else None
    # The staged count is the voxels that actually changed over the union
    # (after lower-id-wins), not the old loop's sum of per-object counts.
    if union is not None:
        n_total = int(np.count_nonzero(working[union] != volume[union]))
    n_processed = len([r for r in results if r[1] is not None])
    return working, touched, union, n_processed, n_total


def _thin_plate_and_block():
    """Label 1 = a one-voxel-thick plate; label 2 = a solid block far away."""
    vol = np.zeros((24, 24, 24), dtype=np.int32)
    vol[5, 4:9, 4:9] = 1          # 25 voxels, one voxel thick
    vol[14:20, 14:20, 14:20] = 2  # 216 voxels, survives a 1-voxel erosion
    return vol


ERODE_3 = [RefineStep("dilate_erode", {"radius": -3, "axis": "3D"})]


# --------------------------------------------------------------------------- #
# 1. A step that empties the crop: nothing written, listed.
# --------------------------------------------------------------------------- #
def test_step_emptied_object_is_not_written_and_is_listed():
    vol = _thin_plate_and_block()
    res = run_batch_recipe(vol, [1], ERODE_3, n_workers=1)

    assert np.array_equal(res.working_volume, vol), "an emptied object must be left as it was"
    assert res.emptied_ids == [1]
    assert res.touched_ids == []
    assert res.n_changed_total == 0
    assert res.union_roi is None


def test_emptied_object_beside_a_changed_one():
    vol = _thin_plate_and_block()
    res = run_batch_recipe(vol, [1, 2], [RefineStep("dilate_erode", {"radius": -1})], n_workers=1)

    # Label 1 (one voxel thick) empties; label 2 shrinks but survives.
    assert res.emptied_ids == [1]
    assert res.touched_ids == [2]
    out = res.working_volume
    assert np.array_equal(out == 1, vol == 1), "label 1 must keep its footprint"
    assert 0 < np.count_nonzero(out == 2) < np.count_nonzero(vol == 2)
    # The union ROI covers only what was written — label 2's crop.
    z0 = res.union_roi[0].start
    assert z0 > 5, "label 1's crop must not reach the union ROI"


@pytest.mark.parametrize("n_workers", [1, 4])
def test_no_id_present_before_is_absent_after(n_workers):
    vol = _thin_plate_and_block()
    ids = [1, 2]
    res = run_batch_recipe(vol, ids, ERODE_3, n_workers=n_workers)
    after = set(np.unique(res.working_volume).tolist())
    for lid in ids:
        assert lid in after, f"label {lid} erased"


# --------------------------------------------------------------------------- #
# 2. Write-time emptying: a lower id already took every voxel of the result.
# --------------------------------------------------------------------------- #
def _write_time_fixture(monkeypatch):
    """Label 1 grows into region R; label 2's result lies ONLY in R.

    The compute is stubbed so the geometry is exact: label 1 is written
    first (ascending ids) and claims R, so label 2's non-empty result has no
    voxel left it may write.
    """
    vol = np.zeros((12, 12, 12), dtype=np.int32)
    vol[2:5, 2:5, 2:5] = 1
    vol[8:10, 8:10, 8:10] = 2
    region = (slice(5, 7), slice(2, 5), slice(2, 5))  # background, beside label 1

    def fake(snapshot, label_id, roi, recipe):
        cur = np.zeros(snapshot[roi].shape, dtype=snapshot.dtype)
        full = np.zeros(snapshot.shape, dtype=bool)
        if label_id == 1:
            full[snapshot == 1] = True
            full[region] = True
        else:
            full[region] = True  # every voxel of 2's result is in R
        cur[full[roi]] = label_id
        n = int(np.count_nonzero((cur == label_id) != (snapshot[roi] == label_id)))
        return cur, n

    monkeypatch.setattr(br, "apply_recipe_to_label", fake)
    return vol, region


def test_write_time_emptied_object_keeps_its_footprint(monkeypatch):
    vol, region = _write_time_fixture(monkeypatch)
    # Wide margin so label 2's crop reaches R.
    recipe = [RefineStep("dilate_erode", {"radius": 8})]
    res = run_batch_recipe(vol, [1, 2], recipe, n_workers=1)

    out = res.working_volume
    assert np.all(out[region] == 1), "the lower id wins R"
    assert np.array_equal(out == 2, vol == 2), "label 2 keeps its footprint"
    assert res.emptied_ids == [2]
    assert res.touched_ids == [1]


def test_write_time_emptying_erase_mode_is_the_old_behaviour(monkeypatch):
    vol, _region = _write_time_fixture(monkeypatch)
    recipe = [RefineStep("dilate_erode", {"radius": 8})]
    res = run_batch_recipe(vol, [1, 2], recipe, n_workers=1, empty="erase")
    working, touched, union, n_proc, n_total = _old_run(vol, [1, 2], recipe)

    assert not np.any(working == 2), "fixture: the old loop erased label 2"
    assert np.array_equal(res.working_volume, working)
    assert res.touched_ids == touched
    assert res.union_roi == union
    assert res.n_processed == n_proc
    assert res.n_changed_total == n_total
    assert res.emptied_ids == []


# --------------------------------------------------------------------------- #
# 3. empty="erase" reproduces the old loop byte-for-byte.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "recipe",
    [
        ERODE_3,
        [RefineStep("dilate_erode", {"radius": -1})],
        [RefineStep("sdf", {"sigma_z": 1.5, "sigma_xy": 1.5})],
        [RefineStep("dilate_erode", {"radius": 2}), RefineStep("fill_holes", {})],
    ],
)
@pytest.mark.parametrize("n_workers", [1, 3])
def test_erase_mode_is_byte_identical_to_the_old_loop(recipe, n_workers):
    vol = _thin_plate_and_block()
    vol[20:23, 2:5, 2:5] = 3
    ids = [1, 2, 3]
    res = run_batch_recipe(vol, ids, recipe, n_workers=n_workers, empty="erase")
    working, touched, union, n_proc, n_total = _old_run(vol, ids, recipe)

    assert np.array_equal(res.working_volume, working)
    assert res.touched_ids == touched
    assert res.union_roi == union
    assert res.n_processed == n_proc
    assert res.n_changed_total == n_total


def test_skip_mode_equals_the_old_loop_when_nothing_empties():
    vol = _thin_plate_and_block()
    vol[1:4, 14:20, 14:20] = 3
    ids = [2, 3]
    recipe = [RefineStep("sdf", {"sigma_z": 1.0, "sigma_xy": 1.0}), RefineStep("dilate_erode", {"radius": 1})]
    res = run_batch_recipe(vol, ids, recipe, n_workers=2)
    working, touched, union, n_proc, n_total = _old_run(vol, ids, recipe)

    assert res.emptied_ids == []
    assert np.array_equal(res.working_volume, working)
    assert res.touched_ids == touched
    assert res.union_roi == union
    assert res.n_processed == n_proc
    assert res.n_changed_total == n_total


def test_empty_keyword_is_validated():
    vol = _thin_plate_and_block()
    with pytest.raises(ValueError):
        run_batch_recipe(vol, [1], ERODE_3, n_workers=1, empty="nope")


def test_emptied_ids_defaults_to_empty():
    res = br.BatchResult(
        working_volume=np.zeros((2, 2, 2)), touched_ids=[], union_roi=None,
        n_processed=0, n_changed_total=0, cancelled=False,
    )
    assert res.emptied_ids == []


# --------------------------------------------------------------------------- #
# 4. Add's own smoothing keeps erasing (D13): byte-identical to d1cdc6bb.
# --------------------------------------------------------------------------- #
def test_add_smoothing_passes_erase_and_is_byte_identical():
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    # A single-voxel piece (1) that SDF smoothing empties, plus a solid one (2).
    result = np.zeros((20, 20, 20), dtype=np.int32)
    result[3, 3, 3] = 1
    result[8:16, 8:16, 8:16] = 2
    sigma = 1.5

    got = Curation3DAnnotator._smooth_add_result(None, result.copy(), sigma=sigma)

    recipe = [RefineStep("sdf", {"sigma_z": sigma, "sigma_xy": sigma})]
    expected, *_ = _old_run(result, [1, 2], recipe)
    assert not np.any(expected == 1), "fixture: the old loop drops the emptied piece"
    assert np.array_equal(got, expected)

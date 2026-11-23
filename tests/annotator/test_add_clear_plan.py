"""Clear objects: which objects a Clear takes (the pure counting step).

* "Objects mostly inside this box": an object is taken when MORE than half
  of its voxels are inside the box; it is then taken whole, and the plan
  counts how many of the taken objects reach outside the box;
* "Every object in the patch": every object;
* the union box of the taken objects (whole-volume coordinates);
* uint32 and int32 volumes, and an id of 2**31 - 1 (ids are compacted
  before they are counted, so a huge id costs nothing);
* a cancel between the steps answers ``cancelled``.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

BOX = (slice(0, 10), slice(0, 10), slice(0, 10))


def _vol(dtype=np.int32):
    return np.zeros((10, 20, 10), dtype)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_more_than_half_inside_is_taken_and_half_or_less_is_not(dtype):
    v = np.zeros((10, 160, 10), dtype)
    v[1, 49:100, 0] = 1       # 51 voxels inside y < 100 ...
    v[1, 100:149, 0] = 1      # ... 49 outside -> taken
    v[2, 51:100, 0] = 2       # 49 inside
    v[2, 100:151, 0] = 2      # 51 outside -> not taken
    v[3, 50:100, 0] = 3       # exactly half -> not taken (MORE than half)
    v[3, 100:150, 0] = 3
    v[4, 0:5, 0:5] = 4        # wholly inside -> taken
    box = (slice(0, 10), slice(0, 100), slice(0, 10))
    plan = AP.clear_plan(labels=v, mode="box", box=box)
    assert plan.status == "ok"
    assert plan.ids == (1, 4)
    assert plan.n_outside == 1           # object 1 reaches out, 4 does not
    assert plan.n_in_patch == 4


def test_the_union_box_covers_the_taken_objects_whole():
    v = _vol()
    v[2, 1:4, 1:4] = 5            # inside
    v[7, 6:14, 8] = 9             # 4 inside (y 6..9), 4 outside -> no
    v[7, 5:14, 9] = 11            # 5 inside, 4 outside -> taken, reaches y 13
    v[0, 15:18, 0] = 12           # outside entirely
    plan = AP.clear_plan(labels=v, mode="box", box=BOX)
    assert plan.ids == (5, 11)
    assert plan.n_outside == 1
    assert plan.bbox == (slice(2, 8), slice(1, 14), slice(1, 10))
    assert plan.n_voxels == 9 + 9


def test_patch_mode_takes_every_object():
    v = _vol(np.uint32)
    v[0, 0, 0] = 3
    v[9, 19, 9] = 7
    v[5, 15, 2] = 4
    plan = AP.clear_plan(labels=v, mode="patch")
    assert plan.ids == (3, 4, 7)
    assert plan.n_outside == 0
    assert plan.bbox == (slice(0, 10), slice(0, 20), slice(0, 10))
    assert plan.n_in_patch == 3


def test_an_empty_box_and_an_empty_patch():
    v = _vol()
    v[0, 15:20, 0] = 2                 # all outside the box
    plan = AP.clear_plan(labels=v, mode="box", box=BOX)
    assert plan.status == "ok" and plan.ids == () and plan.bbox is None
    assert plan.n_in_patch == 1
    empty = AP.clear_plan(labels=_vol(), mode="patch")
    assert empty.ids == () and empty.bbox is None and empty.n_in_patch == 0


def test_a_huge_id_is_counted_without_a_huge_array():
    v = _vol()
    huge = 2 ** 31 - 1
    v[1, 1:3, 1] = huge
    v[2, 2, 2] = 6
    plan = AP.clear_plan(labels=v, mode="box", box=BOX)
    assert plan.ids == (6, huge)
    assert plan.bbox == (slice(1, 3), slice(1, 3), slice(1, 3))


def test_the_plan_never_writes_the_volume():
    v = _vol()
    v[1, 1, 1] = 3
    before = v.copy()
    AP.clear_plan(labels=v, mode="patch")
    AP.clear_plan(labels=v, mode="box", box=BOX)
    np.testing.assert_array_equal(v, before)


def test_a_cancel_answers_cancelled():
    v = _vol()
    v[1, 1, 1] = 3
    ev = threading.Event()
    ev.set()
    plan = AP.clear_plan(labels=v, mode="box", box=BOX, cancel_event=ev)
    assert plan.status == "cancelled"
    assert plan.ids == ()


def test_a_box_mode_without_a_box_is_refused_loudly():
    with pytest.raises(ValueError):
        AP.clear_plan(labels=_vol(), mode="box", box=None)
    with pytest.raises(ValueError):
        AP.clear_plan(labels=_vol(), mode="half", box=BOX)


# --------------------------------------------------------------------------- #
# the two counting paths, and ids of 0 or below
# --------------------------------------------------------------------------- #
def _mixed(dtype=np.int32):
    """Blocks of 2 x 4 x 2 voxels, ids 3, 6, 9, ... (gaps between ids):
    some wholly inside the box, some across its faces, some outside."""
    v = np.zeros((10, 20, 10), dtype)
    i = 3
    for z in range(0, 10, 2):
        for y in range(0, 20, 4):
            for x in range(0, 10, 2):
                v[z:z + 2, y:y + 4, x:x + 2] = i
                i += 3
    return v


@pytest.mark.parametrize("mode", ["box", "patch"])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.uint64])
def test_counting_by_id_and_by_rank_give_the_same_plan(mode, dtype,
                                                       monkeypatch):
    v = _mixed(dtype)
    box = (slice(1, 8), slice(2, 15), slice(0, 5))
    direct = AP.clear_plan(labels=v, mode=mode, box=box)
    monkeypatch.setattr(AP, "CLEAR_DIRECT_MAX_ID", 0)   # force the ranks
    ranked = AP.clear_plan(labels=v, mode=mode, box=box)
    assert direct.ids and direct == ranked
    if mode == "box":
        assert direct.n_outside > 0 and len(direct.ids) < direct.n_in_patch


def test_a_huge_id_takes_the_rank_path_and_a_small_one_the_direct(
        monkeypatch):
    calls = []
    real = np.unique

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(AP.np, "unique", spy)
    v = _vol()
    v[1, 1, 1] = 3
    AP.clear_plan(labels=v, mode="patch")
    assert calls == [], "a small largest id is counted with no sort"
    v[2, 2, 2] = AP.CLEAR_DIRECT_MAX_ID + 1
    plan = AP.clear_plan(labels=v, mode="patch")
    assert calls, "a huge id is compacted first"
    assert plan.ids == (3, AP.CLEAR_DIRECT_MAX_ID + 1)


def test_ids_of_zero_or_below_are_never_taken():
    v = _vol()
    v[1, 1:4, 1] = -5               # a negative id: background, not taken
    v[2, 2, 2] = 6
    for mode in ("patch", "box"):
        plan = AP.clear_plan(labels=v, mode=mode, box=BOX)
        assert plan.ids == (6,)
        assert plan.n_in_patch == 1
        assert plan.bbox == (slice(2, 3), slice(2, 3), slice(2, 3))

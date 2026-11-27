"""Look for nuclei ▸ Inside each cell of another run, wired into the Add page.

The real orchestrator on the Qt-free harness (``add_harness``), the real
recipe (``add_derive.nuclei_in_cells``) on a synthetic three-cell scene drawn
in microns; the panel half on the module-scoped ``AdvancedFeaturesWidget``.

Pinned:

* index 0 (Anywhere) sends exactly what it always sent;
* a box preview passes the PATCH's levels cropped to the box's z range,
  never the box's own (they differ), measured once on the whole session
  array on the box-info worker;
* new ids are fresh: above the volume, the QC-table floor and the Clear
  floor;
* the chips "two in a cell" / "cut at edge", their sort, their
  tooltips, the S1 tails, the REVIEW chips; neither chip unticks a row;
* Accept writes only empty voxels; the other run's file and array are
  never written; a stale read of it is dropped and a preset switch resets
  it;
* the 1.0 um default spacing runs (the amber note says it); the refusals:
  no cell in the box, every nucleus already taken, nothing found, a run
  that cannot be read;
* the wrong-run and no-run help lines.
"""

from __future__ import annotations

import hashlib
import types

import numpy as np
import pytest

from napari_epicycle.annotator import reshape_worker as RW
from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.add_derive import patch_levels

from .add_harness import FakeAdv, FakeThread, Harness, install_fakes
from .test_add_derive_nuclei import ISO, straddle_scene, three_cells_scene


class InsideAdv(FakeAdv):
    """The panel with Look for nuclei ▸ Inside each cell of run "cells"."""

    def __init__(self):
        super().__init__()
        self.look = (AP.LOOK_INSIDE, "cells")
        self.params.update(look_for=AP.LOOK_INSIDE, nuc_radius_um=2.5,
                           nuc_edge=0.25, two_gap=0.70)

    def get_add_look_for(self):
        return self.look


def _harness(monkeypatch, *, box=None, scene=None, spacing=ISO,
             source="qc_settings", vol=None, cache_levels=True):
    install_fakes(monkeypatch)
    img, cells = scene if scene is not None else three_cells_scene()
    if vol is None:
        vol = np.zeros(cells.shape, np.int32)
    h = Harness(vol, images={"nuc": img}, spacing=spacing)
    h.adv = InsideAdv()
    h.ctx["spacing_source"] = source
    h.place_box(box or tuple(slice(0, n) for n in cells.shape))
    h._add_run_guide = cells
    h._add_levels_cache = {}
    if cache_levels:
        h._add_levels_cache[h._add_levels_key()] = patch_levels(
            img, cells, spacing)
    h._add_info = AP.AddBoxInfo(empty_share=1.0,
                                inside_count=AP.count_ids(cells[h._add_roi]))
    return h, img, cells


def _run(h):
    h._preview_add()
    assert h._add_busy, h._add_transient
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    return h._add_preview_res


# --------------------------------------------------------------------------- #
# index 0 is today's finder
# --------------------------------------------------------------------------- #
def test_anywhere_sends_what_it_always_sent(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    h.adv = FakeAdv()                         # Look for nuclei at index 0
    kw, why = h._add_gather_preview()
    assert why is None
    assert "look_for" not in kw["params"]
    assert "patch_levels" not in kw and "run_guide_roi" not in kw
    assert set(kw["params"]) == {
        "min_size", "clear_border", "edge_margin", "auto_smooth",
        "smooth_sigma", "clahe", "clahe_clip", "threshold", "separate",
        "min_distance"}


def test_the_engine_reads_no_inside_key_on_anywhere():
    """``look_for`` absent or "anywhere" both run the brightness finder."""
    from .test_add_nuclei_golden import nuclei_fixture

    img, labels = nuclei_fixture(0)
    roi = tuple(slice(0, n) for n in img.shape)
    base = dict(clear_border=True, edge_margin=3, auto_smooth=False)
    a = AP.run_add_preview(preset="nuclei", roi=roi, existing_roi=labels,
                           next_label=100, spacing=(1.0, 1.0, 1.0),
                           params=base, image_roi=img)
    b = AP.run_add_preview(preset="nuclei", roi=roi, existing_roi=labels,
                           next_label=100, spacing=(1.0, 1.0, 1.0),
                           params=dict(base, look_for=AP.LOOK_ANYWHERE),
                           image_roi=img)
    assert a.ok and b.ok
    assert np.array_equal(a.result, b.result)
    assert a.look_for == b.look_for == AP.LOOK_ANYWHERE


# --------------------------------------------------------------------------- #
# the patch's levels (M4)
# --------------------------------------------------------------------------- #
def test_a_box_gets_the_patch_levels_cropped_never_its_own(monkeypatch):
    box = (slice(6, 22), slice(0, 40), slice(30, 96))
    h, img, cells = _harness(monkeypatch, box=box)
    kw, why = h._add_gather_preview()
    assert why is None
    whole = h._add_cached_levels()
    got = kw["patch_levels"]
    assert got.otsu == whole.otsu
    assert np.array_equal(got.background_z, whole.background_z[6:22])
    own = patch_levels(img[box], cells[box], ISO)
    assert (own.otsu != whole.otsu
            or not np.array_equal(own.background_z, got.background_z)), (
        "the box's own levels must differ from the patch's, or this pin "
        "says nothing")


def test_the_levels_are_measured_once_on_the_whole_session_array(monkeypatch):
    box = (slice(6, 22), slice(0, 40), slice(30, 96))
    h, img, cells = _harness(monkeypatch, box=box, cache_levels=False)
    kw = h._add_gather_info()
    nuclear, guide = kw["levels_sources"]
    assert nuclear.shape == img.shape and guide is cells
    assert kw["run_guide_roi"].shape == tuple(s.stop - s.start for s in box)
    assert kw["image_roi"] is None            # no histogram on Inside
    info = AP.add_box_info(**kw)
    assert info.inside_count == 2              # cells 2 and 3 reach the box
    want = patch_levels(img, cells, ISO)
    assert info.patch_levels.otsu == want.otsu
    # landing caches them; the next request measures nothing
    h._add_info_worker = None
    w = types.SimpleNamespace(_epicycle_token=h._add_info_token,
                              _epicycle_patch="P", _epicycle_epoch=1,
                              _epicycle_roi=tuple(h._add_roi),
                              _epicycle_dark_key=None,
                              _epicycle_brightness_key=None,
                              _epicycle_levels_key=h._add_levels_key())
    h._on_add_info_finished(types.SimpleNamespace(result=info), worker=w)
    assert h._add_cached_levels() is info.patch_levels
    assert "levels_sources" not in h._add_gather_info()


def test_a_preview_waits_for_the_levels(monkeypatch):
    h, _img, _cells = _harness(monkeypatch, cache_levels=False)
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_S1_MEASURING
    h._preview_add()
    assert not h._add_busy


# --------------------------------------------------------------------------- #
# ids, chips, tails, REVIEW
# --------------------------------------------------------------------------- #
def test_fresh_ids_above_both_floors(monkeypatch):
    vol = np.zeros((28, 40, 96), np.int32)
    vol[0, 0, 0] = 10                          # the volume's max
    h, _img, _cells = _harness(monkeypatch, vol=vol)
    h._get_label_id_floor_cb = lambda: 50      # the QC table's floor
    h._add_clear_floor = 70                    # a Clear removed id 70
    res = _run(h)
    assert res.ok, res.refusal
    assert min(res.ids) == 71
    assert res.ids == list(range(71, 71 + len(res.ids)))
    assert set(res.owner) == set(res.ids)


def test_chips_sort_tips_and_the_found_tail(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    res = _run(h)
    pair = res.two_in_cell[2]
    assert len(pair) == 2 and all(res.flags[i] == ("two",) for i in pair)
    order = h.adv.last("candidate_ids")
    assert order[:2] == sorted(pair), "two in a cell sorts first"
    tips = h.adv.last("candidate_tips")
    assert tips[pair[0]] == AP.ADD_TIP_TWO.format(cell=2, k=2)
    assert h.adv.last("candidate_flags") == {i: ("two",) for i in pair}
    h.adv.checked = list(res.ids)
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.found_line(len(res.ids), n_two=1)
    assert AP.ADD_S1_TWO_TAIL_ONE in h.adv.last("status_text")


def test_a_cut_nucleus_carries_its_chip_and_tip(monkeypatch):
    h, _img, _cells = _harness(monkeypatch, scene=straddle_scene(1.0))
    res = _run(h)
    assert res.ok and res.cut_at_edge
    nid = next(iter(res.cut_at_edge))
    assert res.flags[nid] == ("cut",)
    assert h.adv.last("candidate_tips")[nid] == AP.ADD_TIP_CUT.format(cell=1)


def test_review_keeps_the_chips_and_says_the_tail(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    res = _run(h)
    pair = res.two_in_cell[2]
    h.adv.checked = list(res.ids)
    h._on_add_committed_cb = None
    h._accept_add()
    assert h._add_review
    chips = h.adv.last("piece_chips")
    assert {i: chips[i] for i in pair} == {i: [AP.ADD_CHIP_TWO] for i in pair}
    assert set(chips) == set(pair)
    assert h.adv.last("piece_tips")[pair[0]] == AP.ADD_TIP_TWO.format(
        cell=2, k=2)
    h._add_refresh_ui()
    assert AP.ADD_S1_ADDED_TWO_TAIL_ONE in h.adv.last("status_text")


def test_one_of_a_pair_accepted_keeps_no_two_chip(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    res = _run(h)
    pair = res.two_in_cell[2]
    h.adv.checked = [i for i in res.ids if i != pair[1]]
    h._on_add_committed_cb = None
    h._accept_add()
    assert not (h.adv.last("piece_chips") or {})
    h._add_refresh_ui()
    assert "two nuclei" not in h.adv.last("status_text")


def test_review_groups_two_nuclei_by_run_and_cell(monkeypatch):
    """Two boxes that read two runs: cell 5 of run A and cell 5 of run B are
    two cells, so one nucleus in each is not "two in a cell"."""
    h, _img, _cells = _harness(monkeypatch)
    h._add_session = [
        {"ids": [101], "sizes": {101: 50}, "flags": {101: ("two",)},
         "owners": {101: 5}, "owner_run": "A"},
        {"ids": [202], "sizes": {202: 60}, "flags": {202: ("two",)},
         "owners": {202: 5}, "owner_run": "B"},
    ]
    assert h._add_review_chips([101, 202]) == {"piece_chips": {},
                                               "piece_tips": {}}
    h._add_piece_ids = [101, 202]
    h._add_review = True
    assert h._add_s1_state()["review_two"] == 0
    # the same run: one cell with two nuclei, said with the cell's number
    h._add_session[1]["owner_run"] = "A"
    got = h._add_review_chips([101, 202])
    assert got["piece_chips"] == {101: [AP.ADD_CHIP_TWO],
                                  202: [AP.ADD_CHIP_TWO]}
    assert got["piece_tips"][101] == AP.ADD_TIP_TWO.format(cell=5, k=2)
    assert h._add_s1_state()["review_two"] == 1


def test_accept_records_whose_cells_they_are(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    res = _run(h)
    h.adv.checked = list(res.ids)
    h._on_add_committed_cb = None
    h._accept_add()
    assert h._add_session[-1]["owner_run"] == "cells"


def test_the_other_run_and_a_whole_patch_volume_are_not_copied(monkeypatch):
    """The other run is never written, so the gathers hand its box as a
    view; on a whole patch the box-info worker reads the live volume
    itself (an ordinary box still gets a copy, and so does a Preview)."""
    box = (slice(6, 22), slice(0, 40), slice(30, 96))
    h, _img, cells = _harness(monkeypatch, box=box)
    shared = h.sync_manager.shared_labels["L"]
    kw = h._add_gather_info()
    assert np.shares_memory(kw["run_guide_roi"], cells)
    assert not np.shares_memory(kw["existing_roi"], shared)
    pkw, why = h._add_gather_preview()
    assert why is None
    assert np.shares_memory(pkw["run_guide_roi"], cells)
    assert not np.shares_memory(pkw["existing_roi"], shared)
    h.place_box(tuple(slice(0, n) for n in cells.shape))
    h._add_whole_patch = True
    kw = h._add_gather_info()
    assert np.shares_memory(kw["existing_roi"], shared)
    pkw, why = h._add_gather_preview()
    assert why is None
    assert not np.shares_memory(pkw["existing_roi"], shared), (
        "the Preview's write rule reads its own snapshot")


def test_a_box_preview_sees_the_whole_patch_for_kept_nuclei(monkeypatch):
    """On a box the Inside preview also gets the other run and the edited
    run over the WHOLE patch (so a cell's kept nucleus outside the box
    counts), both as views (the worker only reads them).  A whole patch
    passes neither (its crop IS the patch)."""
    box = (slice(6, 22), slice(0, 40), slice(30, 96))
    h, _img, cells = _harness(monkeypatch, box=box)
    shared = h.sync_manager.shared_labels["L"]
    pkw, why = h._add_gather_preview()
    assert why is None
    assert pkw["run_guide_patch"] is cells
    # a view, not a copy: the worker only reads it, and a copy of a large
    # patch on the main thread costs seconds
    assert pkw["existing_patch"] is shared
    h.place_box(tuple(slice(0, n) for n in cells.shape))
    h._add_whole_patch = True
    pkw, why = h._add_gather_preview()
    assert why is None
    assert "run_guide_patch" not in pkw and "existing_patch" not in pkw


def test_a_cells_box_over_the_limit_cuts_no_run_or_channel(monkeypatch):
    h, img, cells = _harness(monkeypatch)
    h.adv = FakeAdv()
    h.adv.preset = "cells"
    h.adv.source = ("labels", "cells")
    h.adv.wall = {"channel": "nuc", "chain": "none",
                  "wall_kind": "membrane"}
    h._add_run_guide = cells
    h.adv.params["box_budget"] = 10 ** 9
    kw = h._add_gather_info()
    assert kw["run_guide_roi"] is not None and kw["dark_rois"]
    h.adv.params["box_budget"] = 1000           # the box is over it
    kw = h._add_gather_info()
    assert kw["run_guide_roi"] is None and kw["dark_rois"] is None
    info = AP.add_box_info(**kw)
    assert info.over_limit and info.run_count is None


def test_accept_writes_only_empty_voxels(monkeypatch):
    vol = np.zeros((28, 40, 96), np.int32)
    vol[10:18, 15:25, 36:46] = 9               # a kept object over a nucleus
    h, _img, _cells = _harness(monkeypatch, vol=vol)
    before = vol.copy()
    res = _run(h)
    assert res.ok
    h.adv.checked = list(res.ids)
    h._on_add_committed_cb = None
    h._accept_add()
    after = h.sync_manager.shared_labels["L"]
    assert np.array_equal(after[before > 0], before[before > 0])
    new = (after != before)
    assert np.all(before[new] == 0)
    assert set(np.unique(after[new])) <= set(res.ids)


def test_accept_on_a_whole_patch_takes_the_workers_ids_and_says_so_first(
        monkeypatch):
    """Accept reads the ids the worker measured (no np.unique of the whole
    result on the main thread -- 2.4 s at 73 Mvox), paints "Adding…" before
    the write, and writes by the same rule: only empty voxels."""
    h, _img, _cells = _harness(monkeypatch)
    h._add_whole_patch = True
    res = _run(h)
    assert res.ok and len(res.ids) >= 2
    result = h._add_preview_result
    before = h.sync_manager.shared_labels["L"].copy()
    shapes = []
    real_unique = np.unique

    def spy(a, *args, **kw):
        shapes.append(np.shape(a))
        return real_unique(a, *args, **kw)

    monkeypatch.setattr(np, "unique", spy)
    h.adv.checked = list(res.ids)
    h._on_add_committed_cb = None
    n_push = len(h.adv.pushes)
    h._accept_add()
    monkeypatch.setattr(np, "unique", real_unique)
    assert result.shape not in shapes, "np.unique ran over the whole result"
    texts = [kw["status_text"] for kw in h.adv.pushes[n_push:]
             if "status_text" in kw]
    assert texts[0] == AP.adding_line(len(res.ids))
    after = h.sync_manager.shared_labels["L"]
    assert set(int(v) for v in real_unique(after)) - {0} == set(
        int(i) for i in res.ids)
    assert h._add_session[-1]["ids"] == sorted(int(i) for i in res.ids)
    new = after != before
    assert np.all(before[new] == 0)
    assert np.array_equal(after[new], result[new])


def test_the_adding_line_is_singular_safe():
    assert AP.adding_line(1) == AP.ADD_S1_ADDING_ONE == "Adding 1 object…"
    assert AP.adding_line(3) == "Adding 3 objects…"


# --------------------------------------------------------------------------- #
# the other run: read once, never written, dropped when stale
# --------------------------------------------------------------------------- #
class _FakeGuideWorker(RW.GuideLoadWorker):
    def moveToThread(self, _th):      # noqa: N802 - Qt name
        return None


def test_the_other_run_is_never_written(monkeypatch, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    img, cells = three_cells_scene()
    path = tmp_path / "labels_curated.tif"
    tifffile.imwrite(str(path), cells.astype(np.uint16))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    h, _img, _cells = _harness(monkeypatch, scene=(img, cells))
    h._add_run_guide = None
    monkeypatch.setattr(RW, "GuideLoadWorker", _FakeGuideWorker)
    h._add_run_sources = {"cells": types.SimpleNamespace(
        path=str(path), run_id="cells", display_name="cells",
        target="cells", which_file="labels_curated.tif")}
    h._add_load_run_guide("cells")
    assert h._add_run_state == "loading"
    w = h._add_guide_worker
    h.sender = w
    FakeThread.instances[-1].run()             # read on the "worker"
    h.sender = None
    guide = h._add_run_guide
    assert guide is not None and np.array_equal(guide, cells)
    kept = guide.copy()
    # the read dropped the box info (it is measured again with the run)
    assert h._add_info is None
    h._add_info = AP.AddBoxInfo(empty_share=1.0, inside_count=3)
    res = _run(h)
    h.adv.checked = list(res.ids)
    h._on_add_committed_cb = None
    h._accept_add()
    assert np.array_equal(guide, kept), "the run's array was written"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    # it is never a layer
    assert not any(getattr(lay, "name", "") == "cells"
                   for lay in h.viewer.layers)


def test_a_stale_read_is_dropped_and_a_preset_switch_resets(monkeypatch,
                                                           tmp_path):
    tifffile = pytest.importorskip("tifffile")
    img, cells = three_cells_scene()
    path = tmp_path / "labels_3d.tif"
    tifffile.imwrite(str(path), cells.astype(np.uint16))
    h, _img, _cells = _harness(monkeypatch, scene=(img, cells))
    h._add_run_guide = None
    monkeypatch.setattr(RW, "GuideLoadWorker", _FakeGuideWorker)
    h._add_run_sources = {"cells": types.SimpleNamespace(
        path=str(path), run_id="cells", display_name="cells", target="",
        which_file="labels_3d.tif")}
    h._add_load_run_guide("cells")
    old = h._add_guide_worker
    old_thread = FakeThread.instances[-1]
    h.adv.preset = "cells"
    h._on_add_preset_changed("cells")
    assert h._add_run_guide is None and h._add_run_state is None
    h.sender = old
    old_thread.run()                           # the old read lands late
    assert h._add_run_guide is None, "a stale read must be dropped"


def test_a_look_for_change_reads_the_picked_run(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    seen = []
    h._add_load_run_guide = lambda run_id: seen.append(run_id)
    h._on_add_look_for_changed((AP.LOOK_INSIDE, "other"))
    assert seen == ["other"]
    assert h._add_run_guide is None
    h._on_add_look_for_changed((AP.LOOK_ANYWHERE, None))
    assert seen == ["other"]


def test_a_box_start_reads_the_picked_run_again(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    h._reject_add()
    assert h._add_run_guide is None
    seen = []
    h._add_load_run_guide = lambda run_id: seen.append(run_id)
    h._start_add()
    assert seen == ["cells"]


def _sources(path, run_id="cells", target="cells"):
    return {run_id: types.SimpleNamespace(
        path=str(path), run_id=run_id, display_name=run_id, target=target,
        which_file="labels_curated.tif")}


def _land_info(h):
    """The box-info worker runs and its thread finishes (a dirty request
    relaunches once, as a real thread's finish does)."""
    for _ in range(3):
        th = getattr(h, "_add_info_thread", None)
        if th is None:
            return
        th.run()
        th.end()


def test_a_pick_with_no_box_reads_nothing_and_whole_patch_reads_it(
        monkeypatch, tmp_path):
    """The sequence that stuck the page on "Reading that run's objects…":
    pick Inside with no box, then Whole patch.  The pick reads nothing; the
    box start reads the run; it lands; the levels and the count land;
    Preview is available."""
    tifffile = pytest.importorskip("tifffile")
    img, cells = three_cells_scene()
    path = tmp_path / "labels_curated.tif"
    tifffile.imwrite(str(path), cells.astype(np.uint16))
    h, _img, _cells = _harness(monkeypatch, scene=(img, cells),
                               cache_levels=False)
    monkeypatch.setattr(RW, "GuideLoadWorker", _FakeGuideWorker)
    h._add_run_sources = _sources(path)
    # the box start lists the runs again (the context push): the same one
    from napari_epicycle.annotator import reshape_guide as RG

    src = h._add_run_sources["cells"]
    src.kind = "run"
    monkeypatch.setattr(RG, "list_guide_sources", lambda *a, **k: [src])
    h._reject_add()                                   # idle: no box
    assert not h._add_running
    n0 = len(FakeThread.instances)
    h._on_add_look_for_changed((AP.LOOK_INSIDE, "cells"))
    assert len(FakeThread.instances) == n0, "an idle pick must read nothing"
    assert h._add_run_state is None and h._add_run_guide is None
    assert h.adv.last("status_text") == AP.ADD_S1_IDLE
    h._start_add_whole()
    assert h._add_running and h._add_whole_patch
    assert h._add_run_state == "loading"
    assert h.adv.last("status_text") == AP.ADD_S1_RUN_LOADING
    assert h.adv.last("preview_enabled") is False
    gw, gth = h._add_guide_worker, h._add_guide_thread
    assert gw is not None and gth is not None
    h.sender = gw
    gth.run()                                         # the read lands
    h.sender = None
    assert h._add_run_state is None
    assert np.array_equal(h._add_run_guide, cells)
    _land_info(h)                                     # count + patch levels
    assert h._add_cached_levels() is not None
    assert h._add_info is not None and h._add_info.inside_count == 3
    assert h.adv.last("preview_enabled") is True
    kw, why = h._add_gather_preview()
    assert why is None and kw["params"]["look_for"] == AP.LOOK_INSIDE


@pytest.mark.parametrize("how", ["read", "refused", "error"])
def test_a_read_that_lands_with_no_box_is_not_read_again_later(
        monkeypatch, how):
    """A read that lands after the box went (the token unchanged) leaves the
    run "not read", never "loading" or "refused": the next box start reads
    it again."""
    h, _img, cells = _harness(monkeypatch)
    h._add_run_guide = None
    h._add_run_state = "loading"
    h._add_running = False
    if how == "read":
        h._on_add_guide_loaded(cells)
    elif how == "refused":
        h._on_add_guide_refused("That file is not a label mask.")
    else:
        h._on_add_guide_error("Traceback ...")
    assert h._add_run_state is None and h._add_run_guide is None
    seen = []
    h._add_load_run_guide = lambda run_id: seen.append(run_id)
    h._start_add()
    assert seen == ["cells"]


# --------------------------------------------------------------------------- #
# the refusals
# --------------------------------------------------------------------------- #
def test_the_default_spacing_runs_with_the_note(monkeypatch):
    """A spacing never set is no refusal: Inside each cell runs on the
    1.0 um fallback and the page asks for the amber default-spacing note."""
    img, cells = three_cells_scene(sp=(1.0, 1.0, 1.0))
    h, _img, _cells = _harness(monkeypatch, spacing=(1.0, 1.0, 1.0),
                               source="unknown", scene=(img, cells))
    h._add_refresh_ui()
    assert h.adv.last("preview_enabled") is True
    assert h.adv.last("default_spacing") is True
    assert not h._add_s1_state().get("spacing_unknown")
    res = _run(h)
    assert res is not None and res.status == "ok", (
        getattr(res, "refusal", None), h._add_transient)


def test_a_known_one_micron_spacing_runs(monkeypatch):
    """Half size of 0.5 um voxels is 1.0 um per voxel, and known."""
    h, _img, _cells = _harness(monkeypatch)
    h.ctx.update(spacing=(1.0, 1.0, 1.0), spacing_source="qc_settings")
    h._add_levels_cache[h._add_levels_key()] = patch_levels(
        _img, _cells, (1.0, 1.0, 1.0))
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert tuple(kw["spacing"]) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("source", ["qc_settings", "patch"])
def test_a_one_micron_voxel_from_an_explicit_source_runs(monkeypatch,
                                                         source):
    """1.0 um entered in QC Settings (the context then says "qc_settings")
    or set on the patch ("patch": Set spacing..., Step 1, the file's tags)
    runs Inside each cell with no note; the never-set fallback ("unknown")
    runs too, with the amber default-spacing note."""
    img, cells = three_cells_scene()
    h, _img, _cells = _harness(monkeypatch, spacing=(1.0, 1.0, 1.0),
                               source=source, scene=(img, cells),
                               cache_levels=False)
    h._add_levels_cache[h._add_levels_key()] = patch_levels(
        img, cells, (1.0, 1.0, 1.0))
    h._add_refresh_ui()
    assert h.adv.last("preview_enabled") is True
    assert h.adv.last("default_spacing") is False
    res = _run(h)
    assert res is not None and res.status == "ok", res.refusal
    h.ctx["spacing_source"] = "unknown"
    h._add_preview_res = None
    h._add_preview_result = None
    h._add_refresh_ui()
    assert h.adv.last("preview_enabled") is True
    assert h.adv.last("default_spacing") is True


def test_no_cell_of_that_run_in_the_box(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    h._add_info.inside_count = 0
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_S1_INSIDE_NO_CELLS


def test_a_refused_run_is_said_under_look_for(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    h._add_run_refused("That file is not a label mask.")
    assert h.adv.last("status_text") == AP.ADD_S1_INSIDE_UNUSABLE


def test_a_run_still_reading_blocks_the_preview(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    h._add_run_guide = None
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_S1_RUN_LOADING


def test_every_nucleus_taken_and_nothing_found():
    img, cells = three_cells_scene()
    roi = tuple(slice(0, n) for n in img.shape)
    lev = patch_levels(img, cells, ISO)
    taken = AP.run_add_preview(
        preset="nuclei", roi=roi,
        existing_roi=np.where(cells > 0, 900, 0).astype(np.int32),
        next_label=1000, spacing=ISO, params={"look_for": AP.LOOK_INSIDE},
        image_roi=img, run_guide_roi=cells, patch_levels=lev)
    assert taken.refusal == AP.ADD_S1_INSIDE_ALL_TAKEN
    flat = np.full(img.shape, 10.0, np.float32)
    nothing = AP.run_add_preview(
        preset="nuclei", roi=roi, existing_roi=np.zeros_like(cells),
        next_label=1, spacing=ISO, params={"look_for": AP.LOOK_INSIDE},
        image_roi=flat, run_guide_roi=cells,
        patch_levels=patch_levels(img, cells, ISO))
    assert nothing.refusal == AP.ADD_S1_INSIDE_NOTHING
    none = AP.run_add_preview(
        preset="nuclei", roi=roi, existing_roi=np.zeros_like(cells),
        next_label=1, spacing=ISO, params={"look_for": AP.LOOK_INSIDE},
        image_roi=img, run_guide_roi=np.zeros_like(cells), patch_levels=lev)
    assert none.refusal == AP.ADD_S1_INSIDE_NO_CELLS


def _mark_run(h, target):
    h._add_run_sources = {"cells": types.SimpleNamespace(
        path="", run_id="cells", display_name="cells", target=target,
        which_file="labels_curated.tif")}


def test_a_run_marked_nuclei_never_advises_clearing_the_box(monkeypatch):
    """In the cells run the only other run is the nuclei run: the objects
    here are probably the curated cells, so neither the amber nor the
    all-taken line says "Clear objects first"."""
    h, _img, cells = _harness(monkeypatch)
    h._add_info.empty_share = 0.1
    _mark_run(h, "cells")
    h._add_refresh_ui()
    assert h.adv.last("derive_note") == AP.ADD_AMBER_BOX_MOSTLY_FULL.format(
        p=10)
    _mark_run(h, "Nuclei")
    assert h._add_picked_run_is_nuclei()
    h._add_refresh_ui()
    note = h.adv.last("derive_note")
    assert note == AP.ADD_AMBER_BOX_MOSTLY_FULL_NUCLEI_RUN.format(p=10)
    assert "Clear objects" not in note
    # every nucleus found sits on an object already here (the recipe's
    # answer, as test_every_nucleus_taken_and_nothing_found pins it)
    from packages.epicycle_annotator import add_derive as AD

    taken = types.SimpleNamespace(n_cells=3, all_taken=True, n_nuclei=0,
                                  log_line=lambda: "[DERIVE] all taken")
    monkeypatch.setattr(AD, "nuclei_in_cells", lambda *a, **k: taken)
    h._add_info.empty_share = 0.5
    _run(h)
    assert h._add_transient == AP.ADD_S1_INSIDE_ALL_TAKEN_NUCLEI_RUN
    assert "Clear objects" not in h.adv.last("status_text")
    # a cells run keeps the full advice
    _mark_run(h, "cells")
    _run(h)
    assert h._add_transient == AP.ADD_S1_INSIDE_ALL_TAKEN


# --------------------------------------------------------------------------- #
# a channel the patch's levels refuse (never the run's refusal)
# --------------------------------------------------------------------------- #
def _channel_harness(monkeypatch, channel="zero"):
    install_fakes(monkeypatch)
    img, cells = three_cells_scene()
    vol = np.zeros(cells.shape, np.int32)
    h = Harness(vol, images={"nuc": img, "zero": np.zeros_like(img)},
                spacing=ISO)
    h.adv = InsideAdv()
    h.adv.params["channel"] = channel
    h.ctx["spacing_source"] = "qc_settings"
    h.place_box(tuple(slice(0, n) for n in cells.shape))
    h._add_run_guide = cells
    h._add_levels_cache = {}
    h._add_levels_errors = {}
    return h, img, cells


def _land_box_info(h, kw, key):
    info = AP.add_box_info(**kw)
    w = types.SimpleNamespace(
        _epicycle_token=h._add_info_token, _epicycle_patch="P",
        _epicycle_epoch=1, _epicycle_roi=tuple(h._add_roi),
        _epicycle_dark_key=None, _epicycle_brightness_key=None,
        _epicycle_levels_key=key)
    h._on_add_info_finished(types.SimpleNamespace(result=info), worker=w)
    return info


def test_the_patch_levels_refuse_a_channel_with_nothing_above_zero():
    img, cells = three_cells_scene()
    with pytest.raises(ValueError, match="no bright level above zero"):
        patch_levels(np.zeros_like(img), cells, ISO)


def test_a_zero_channel_is_said_as_the_channel_not_the_run(monkeypatch):
    h, _img, _cells = _channel_harness(monkeypatch)
    kw = h._add_gather_info()
    assert "levels_sources" in kw
    info = _land_box_info(h, kw, h._add_levels_key())
    assert info.levels_error and info.patch_levels is None
    assert h._add_run_state is None, "the run is not refused"
    assert h._add_run_guide is not None
    assert h._add_levels_error() == AP.plain_reason(info.levels_error)
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_S1_INSIDE_CHANNEL_UNUSABLE
    assert h.adv.last("preview_enabled") is False
    assert h.adv.last("levels_note") == h._add_levels_error()
    # the same key is not measured again on every request
    assert "levels_sources" not in h._add_gather_info()


def test_a_refusal_for_another_channel_lands_as_nothing(monkeypatch):
    """The answer for the channel picked BEFORE a change lands after it:
    it is kept for its own key only, so the new channel is not refused."""
    h, _img, _cells = _channel_harness(monkeypatch)
    kw = h._add_gather_info()
    old_key = h._add_levels_key()
    h.adv.params["channel"] = "nuc"
    h._on_add_channels_changed()
    _land_box_info(h, kw, old_key)
    assert h._add_levels_error() is None
    assert not h._add_levels_errors
    h._add_refresh_ui()
    assert h.adv.last("status_text") != AP.ADD_S1_INSIDE_CHANNEL_UNUSABLE
    assert h.adv.last("levels_note") == ""


def test_a_channel_change_clears_the_refusal_and_measures_again(monkeypatch):
    h, _img, _cells = _channel_harness(monkeypatch)
    _land_box_info(h, h._add_gather_info(), h._add_levels_key())
    assert h._add_levels_error() is not None
    h.adv.params["channel"] = "nuc"
    h._on_add_channels_changed()
    assert h._add_levels_errors == {}
    kw = h._add_gather_info()
    assert "levels_sources" in kw, "the new channel is measured"
    _land_box_info(h, kw, h._add_levels_key())
    assert h._add_cached_levels() is not None
    h._add_refresh_ui()
    assert h.adv.last("status_text") != AP.ADD_S1_INSIDE_CHANNEL_UNUSABLE
    # back to the refused channel: measured again, refused again
    h.adv.params["channel"] = "zero"
    h._on_add_channels_changed()
    assert "levels_sources" in h._add_gather_info()


def test_a_recipe_refusal_at_preview_is_said_in_plain_words():
    """A ValueError of the recipe (here the threshold of zero) is a
    refusal with its own words, never a failed Preview."""
    from packages.epicycle_annotator.add_derive import PatchLevels

    img, cells = three_cells_scene()
    roi = tuple(slice(0, n) for n in img.shape)
    lev = patch_levels(img, cells, ISO)
    bad = PatchLevels(0.0, lev.background_z)
    out = AP.run_add_preview(
        preset="nuclei", roi=roi, existing_roi=np.zeros_like(cells),
        next_label=1, spacing=ISO,
        params={"look_for": AP.LOOK_INSIDE},
        image_roi=img, run_guide_roi=cells, patch_levels=bad)
    assert out.status == "refused"
    assert "no bright level above zero" in out.refusal
    assert "\n" not in out.refusal


def test_the_engine_runs_at_the_default_spacing():
    """The engine has no spacing refusal: the 1.0 um default runs, and a
    missing spacing falls back to 1.0 um on every axis."""
    img, cells = three_cells_scene(sp=(1.0, 1.0, 1.0))
    roi = tuple(slice(0, n) for n in img.shape)
    outs = []
    for spacing in ((1.0, 1.0, 1.0), None):
        lev = patch_levels(img, cells, (1.0, 1.0, 1.0))
        out = AP.run_add_preview(
            preset="nuclei", roi=roi, existing_roi=np.zeros_like(cells),
            next_label=1, spacing=spacing,
            params={"look_for": AP.LOOK_INSIDE}, image_roi=img,
            run_guide_roi=cells, patch_levels=lev)
        assert out.status == "ok", (spacing, out.refusal)
        outs.append(out)
    assert outs[0].ids == outs[1].ids


def test_the_walls_tick_says_why_on_inside(monkeypatch):
    h, _img, _cells = _harness(monkeypatch)
    _run(h)
    assert h.adv.last("walls_toggle_enabled") is False
    assert h.adv.last("walls_toggle_reason") == AP.ADD_WALLS_INSIDE


def test_the_counts_note_says_what_to_act_on_first():
    counts = {k: 3 for k, _m, _o in AP._COUNT_PARTS}
    parts = AP.counts_note(counts).split(AP.ADD_NOTE_COUNTS_SEP)
    assert parts[:5] == [
        AP.ADD_NOTE_ABSORBED.format(n=3),
        AP.ADD_NOTE_CELL_HAS_ONE.format(n=3),
        AP.ADD_NOTE_NO_NUCLEUS.format(n=3),
        AP.ADD_NOTE_SPLIT_BORDER.format(n=3),
        AP.ADD_NOTE_OUTSIDE_CELLS.format(n=3)]


# --------------------------------------------------------------------------- #
# the panel half
# --------------------------------------------------------------------------- #
from .add_widget_fixture import qapp, reset_add, widget  # noqa: E402,F401


def _items(nuclei=False, raw=False):
    text = AP.look_inside_text("default", "nuclei" if nuclei else "cells",
                               curated=not raw)
    return [(text, True, (AP.LOOK_INSIDE, "default"), nuclei)]


def test_with_no_other_run_the_item_is_grey_and_its_tooltip_says_how(widget):
    """The how-to lives in the grey item's tooltip: never on the page under
    Anywhere (on a single-run workspace it sat on every Nuclei page)."""
    from qtpy.QtCore import Qt

    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[
        (AP.ADD_LOOK_NO_RUN, False, (AP.LOOK_INSIDE, None), False)])
    c = widget.add_look_for_combo
    assert c.currentIndex() == 0 and c.itemText(0) == AP.ADD_LOOK_ANYWHERE
    assert c.itemText(1) == AP.ADD_LOOK_NO_RUN
    assert not c.model().item(1).isEnabled()
    assert c.itemData(1, Qt.ToolTipRole) == AP.ADD_HELP_INSIDE_NO_RUN
    for box in (False, True):
        widget.update_add_ui_state(box_active=box)
        assert widget._add_inside_help.isHidden(), box
    widget.update_add_ui_state(box_active=False)


def test_inside_with_no_run_current_says_how_on_the_page(widget):
    """Only an item that vanished could leave Inside current with no run:
    then, and only then, the page says how to get one."""
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[
        (AP.ADD_LOOK_NO_RUN, True, (AP.LOOK_INSIDE, None), False)])
    c = widget.add_look_for_combo
    c.setCurrentIndex(1)
    try:
        assert widget.get_add_look_for() == (AP.LOOK_INSIDE, None)
        assert not widget._add_inside_help.isHidden()
        assert widget._add_inside_help.text() == AP.ADD_HELP_INSIDE_NO_RUN
    finally:
        c.setCurrentIndex(0)


def test_the_combo_tooltip_is_the_current_items_whole_text(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    reset_add(widget)
    widget.update_add_ui_state(look_for_items=_items(raw=True))
    c = widget.add_look_for_combo
    assert c.toolTip() == ADD_HELP_LINES["add_look_for_combo"]
    c.setCurrentIndex(1)
    try:
        assert c.toolTip() == c.itemText(1)
        assert c.toolTip().endswith(AP.ADD_ITEM_RAW_SUFFIX)
    finally:
        c.setCurrentIndex(0)
    assert c.toolTip() == ADD_HELP_LINES["add_look_for_combo"]


def test_the_advanced_summary_shows_the_shares_with_two_decimals(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=_items())
    widget.add_look_for_combo.setCurrentIndex(1)
    try:
        title = str(widget._add_advanced_toggle.property(
            "_epicycle_chevron_title"))
        assert title.startswith("Advanced: " + AP.ADD_SUM_ADVANCED_INSIDE.format(
            r="2.5", e="0.25", g="0.70")), title
    finally:
        widget.add_look_for_combo.setCurrentIndex(0)


def test_the_count_says_patch_on_the_whole_patch(widget):
    reset_add(widget)
    widget.update_add_ui_state(box_active=True, whole_patch=True,
                               inside_count=812)
    try:
        assert widget._add_inside_count.text() == (
            "812 cells of that run in the patch.")
        widget.update_add_ui_state(whole_patch=False, inside_count=812)
        assert widget._add_inside_count.text() == (
            "812 cells of that run in the box.")
    finally:
        widget.update_add_ui_state(box_active=False, whole_patch=False)


def test_a_refused_channel_is_said_under_look_for_nuclei(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=_items(), box_active=True)
    widget.add_look_for_combo.setCurrentIndex(1)
    try:
        widget.update_add_ui_state(levels_note="No bright level.")
        assert widget._add_inside_help.text() == "No bright level."
        widget.update_add_ui_state(levels_note="")
        assert widget._add_inside_help.text() == AP.ADD_HELP_INSIDE
    finally:
        widget.add_look_for_combo.setCurrentIndex(0)
        widget.update_add_ui_state(box_active=False, levels_note="")
    assert widget._add_inside_help.isHidden()


def test_inside_shows_the_amber_spacing_note_at_the_default(widget):
    """Inside each cell runs at the never-set spacing: the amber
    default-spacing note shows, on a box and on the whole patch, before a
    result -- as on Cells."""
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=_items(), box_active=True,
                               default_spacing=True)
    widget.add_look_for_combo.setCurrentIndex(1)
    try:
        assert widget.add_look_inside()
        assert not widget.add_spacing_note.isHidden()
        assert widget.add_spacing_note.text() == (
            AP.ADD_NOTE_DEFAULT_SPACING_BOXED)
        widget.update_add_ui_state(whole_patch=True)
        assert not widget.add_spacing_note.isHidden()
        widget.update_add_ui_state(default_spacing=False)
        assert widget.add_spacing_note.isHidden()
    finally:
        widget.add_look_for_combo.setCurrentIndex(0)
        widget.update_add_ui_state(box_active=False, whole_patch=False,
                                   default_spacing=False)


def test_a_run_is_listed_never_preselected_and_read_by_index(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=_items())
    c = widget.add_look_for_combo
    assert c.currentIndex() == 0
    assert c.itemText(1) == "Inside each cell of run ▸ default · cells"
    assert widget.get_add_look_for() == (AP.LOOK_ANYWHERE, None)
    c.setCurrentIndex(1)
    try:
        assert widget.get_add_look_for() == (AP.LOOK_INSIDE, "default")
        assert widget._add_inside_help.text() == AP.ADD_HELP_INSIDE
    finally:
        c.setCurrentIndex(0)


def test_a_run_marked_nuclei_asks_for_a_cells_run(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=_items(nuclei=True))
    widget.add_look_for_combo.setCurrentIndex(1)
    try:
        assert widget._add_inside_help.text() == (
            AP.ADD_HELP_INSIDE_PICKED_NUCLEI)
    finally:
        widget.add_look_for_combo.setCurrentIndex(0)


def test_inside_hides_the_threshold_and_swaps_the_advanced_rows(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=_items(), box_active=True)
    widget.add_look_for_combo.setCurrentIndex(1)
    try:
        for w in (widget.add_clahe_cb, widget._add_threshold_row,
                  widget._add_threshold_help):
            assert w.isHidden()
        assert not widget._add_inside_count.isHidden()
        assert widget._add_anywhere_adv.isHidden()
        assert not widget._add_inside_adv.isHidden()
        p = widget.get_add_params()
        assert p["look_for"] == AP.LOOK_INSIDE
        assert (p["nuc_radius_um"], p["nuc_edge"], p["two_gap"]) == (
            2.5, 0.25, 0.70)
        assert p["auto_smooth"] is False, "Inside starts with smoothing off"
        widget.update_add_ui_state(inside_count=())
        assert widget._add_inside_count.text() == AP.ADD_INSIDE_COUNT_PENDING
        widget.update_add_ui_state(inside_count=1)
        assert widget._add_inside_count.text() == AP.ADD_INSIDE_COUNT_ONE
    finally:
        widget.add_look_for_combo.setCurrentIndex(0)
        widget.update_add_ui_state(box_active=False)
    assert widget.get_add_params()["auto_smooth"] is True
    assert "look_for" not in widget.get_add_params()
    assert not widget._add_threshold_row.isHidden()


def test_a_smoothing_the_user_set_is_kept_on_inside(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=_items())
    widget.add_auto_smooth_cb.setChecked(False)
    widget.add_auto_smooth_cb.setChecked(True)          # touched: on
    widget.add_look_for_combo.setCurrentIndex(1)
    try:
        assert widget.add_auto_smooth_cb.isChecked()
    finally:
        widget.add_look_for_combo.setCurrentIndex(0)
        reset_add(widget)


def test_the_check_this_chips_never_untick_a_row(widget):
    reset_add(widget)
    widget.update_add_ui_state(
        candidate_flags={5: ("two",), 6: ("two", "cut"), 7: ("cut",)},
        candidate_tips={5: "tip five"}, candidate_small_ids=set())
    widget.update_add_ui_state(candidate_ids=[5, 6, 7, 8],
                               candidate_sizes={i: 10 for i in (5, 6, 7, 8)},
                               candidate_edge_ids={8})
    from qtpy.QtCore import Qt

    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_CHIPS_ROLE,
    )

    lst = widget.add_candidate_list
    rows = {lst.item(i).data(Qt.UserRole): lst.item(i) for i in range(4)}
    assert rows[5].data(ADD_CHIPS_ROLE) == [AP.ADD_CHIP_TWO]
    assert rows[6].data(ADD_CHIPS_ROLE) == [AP.ADD_CHIP_TWO, AP.ADD_CHIP_CUT]
    assert rows[7].data(ADD_CHIPS_ROLE) == [AP.ADD_CHIP_CUT]
    for i in (5, 6, 7):
        assert rows[i].checkState() == Qt.Checked
    assert rows[8].checkState() == Qt.Unchecked
    assert "tip five" in rows[5].toolTip()
    widget.update_add_ui_state(candidate_flags={}, candidate_tips={},
                               candidate_ids=[], candidate_sizes={},
                               candidate_edge_ids=set())


def test_review_rows_carry_their_chips(widget):
    from qtpy.QtCore import Qt

    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_CHIPS_ROLE,
    )

    widget.update_add_ui_state(piece_ids=[11, 12], piece_sizes={11: 5, 12: 6},
                               piece_chips={11: [AP.ADD_CHIP_TWO]},
                               piece_tips={11: "the tip"})
    lst = widget.add_pieces_list
    rows = {lst.item(i).data(Qt.UserRole): lst.item(i) for i in range(2)}
    assert rows[11].data(ADD_CHIPS_ROLE) == [AP.ADD_CHIP_TWO]
    assert rows[12].data(ADD_CHIPS_ROLE) == []
    assert "the tip" in rows[11].toolTip()
    widget.update_add_ui_state(piece_ids=[])


def test_the_look_for_combo_is_sized_like_the_markers_combo(widget):
    from qtpy.QtWidgets import QComboBox

    c = widget.add_look_for_combo
    m = widget.add_marker_source_combo
    assert c.sizeAdjustPolicy() == m.sizeAdjustPolicy() == (
        QComboBox.AdjustToMinimumContentsLengthWithIcon)
    # the panel's width pass lowers every combo's minimum alike
    assert c.minimumContentsLength() == m.minimumContentsLength() <= 10
    assert c.sizePolicy().horizontalPolicy() == (
        m.sizePolicy().horizontalPolicy())


def test_the_dim_markers_part_comes_sixth():
    """Nuclei ▸ At the markers I place: the markers that made nothing are
    said right after the outside-cells part."""
    counts = {k: 3 for k, _m, _o in AP._COUNT_PARTS}
    parts = AP.counts_note(counts).split(AP.ADD_NOTE_COUNTS_SEP)
    assert parts[5] == AP.ADD_NOTE_MARKERS_DIM.format(n=3)

"""One size rule for Cells on any region -- a box of any size or the whole
patch.

"Update as I click" runs Cells ▸ Click markers live up to
``ADD_LIVE_CELLS_MAX_VOXELS`` (20 million voxels, the slow-patch
threshold); Preview runs up to ``ADD_BOX_BUDGET`` (100 million, the whole
fill's own ceiling); above either limit S1 names it.  A box keeps its
closed faces (its edge cuts through tissue; only a whole patch's faces are
the end of the data).  A live click the click fill cannot take still runs
the full fill on a box or a whole patch of at most
``ADD_LIVE_FULL_RUN_MAX_VOXELS`` (1 million, a fast full run), and later
clicks keep running; above that size the click is refused rather than
flooding the region whole (that flood cannot be stopped), later clicks
wait for Preview until the settings change, and a box needs the walls and
the Grow by the click fill redoes.  The engine reads a box's channels itself (the
worker crops them), as it does on a whole patch.  S, M and L and the start
box keep their sizes: on Cells they reach at most
``ADD_CELLS_PRESET_MAX_VOXELS``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import region as R
from packages.epicycle_annotator.reshape import spec as S

from .add_harness import FakeThread, Harness, install_fakes
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_cells_box import CELL_A, CELL_B, _fire, _finish, _id_at
from .test_add_live_cells_incremental import ci, co  # noqa: F401

MILLION = 1_000_000


# --------------------------------------------------------------------------- #
# the numbers
# --------------------------------------------------------------------------- #
def test_one_live_limit_and_one_preview_limit_for_any_region():
    assert AP.ADD_LIVE_CELLS_MAX_VOXELS == 20 * MILLION
    # the whole patch reads the same limit: no second, whole-only one
    assert not hasattr(AP, "ADD_LIVE_CELLS_WHOLE_MAX_VOXELS")
    assert S.ADD_BOX_BUDGET == 100 * MILLION
    assert R.WHOLE_FILL_MAX_VOXELS == S.ADD_BOX_BUDGET
    assert AP.ADD_BOX_BUDGET == S.ADD_BOX_BUDGET
    assert S.DEFAULT_PARAMS["box_budget"] == S.ADD_BOX_BUDGET
    assert R.ADD_ENGINE_TOO_BIG == R.too_big_sentence(S.ADD_BOX_BUDGET)
    # S, M and L keep their sizes: a preset size, not a limit
    assert AP.ADD_CELLS_PRESET_MAX_VOXELS == 4 * MILLION


def test_the_widget_has_no_literal_budget():
    from napari_epicycle.annotator import curation_3d_widget as CW

    src = inspect.getsource(CW)
    assert "4_000_000" not in src
    assert "ADD_LIVE_CELLS_WHOLE_MAX_VOXELS" not in src


# --------------------------------------------------------------------------- #
# S1, by size
# --------------------------------------------------------------------------- #
def _state(shape, **kw):
    s = {"has_patch": True, "has_box": True, "preset": "cells",
         "source": AP.SOURCE_CLICKS, "wall_channel": "mem",
         "box_shape": shape, "box_budget": AP.ADD_BOX_BUDGET,
         "whole_patch": False, "n_clicks": 1, "n_ticked": 1,
         "place_markers_on": True, "live_on": True, "empty_share": 1.0,
         "live_whole_ok": True}
    s.update(kw)
    return s


#: 5.0, 25.0 and 150.0 million voxels
FIVE = (50, 100, 1000)
TWENTY_FIVE = (250, 100, 1000)
HUNDRED_FIFTY = (1500, 100, 1000)


def test_a_5_million_box_is_live_and_previewable():
    s = _state(FIVE)
    assert AP.preview_block(s) is None
    assert not AP.s1_sentence(s).startswith("Press Preview")


def test_a_25_million_box_waits_for_preview_and_names_20_million():
    s = _state(TWENTY_FIVE)
    assert AP.preview_block(s) is None, "Preview still runs it"
    line = AP.s1_sentence(s)
    assert line == AP.ADD_S1_LIVE_OVER_GATE_BOX.format(n="20 million")
    assert line == ("Press Preview to see the cells — this box is over 20 "
                    "million voxels.")


def test_a_25_million_patch_says_the_same_with_patch():
    s = _state(TWENTY_FIVE, whole_patch=True, live_whole_ok=False)
    line = AP.s1_sentence(s)
    assert line == AP.ADD_S1_LIVE_OVER_GATE_PATCH_SIZE.format(n="20 million")
    assert line == ("Press Preview to see the cells — this patch is over 20 "
                    "million voxels.")
    # the box and the patch: one family, the region the only difference
    assert AP.ADD_S1_LIVE_OVER_GATE_PATCH_SIZE == (
        AP.ADD_S1_LIVE_OVER_GATE_BOX.replace("this box", "this patch"))
    # under the limit, a patch the click fill cannot take keeps its own line
    small = _state(FIVE, whole_patch=True, live_whole_ok=False)
    assert AP.s1_sentence(small) == AP.ADD_S1_LIVE_OVER_GATE_PATCH


def test_at_the_live_limit_a_box_is_still_live():
    s = _state((200, 100, 1000))                # 20,000,000
    assert not AP.s1_sentence(s).startswith("Press Preview")


def test_a_150_million_box_refuses_preview_and_names_100_million():
    s = _state(HUNDRED_FIFTY)
    block = AP.preview_block(s)
    assert block == AP.over_limit_line(HUNDRED_FIFTY)
    assert block == ("This box is 150.0 million voxels; the limit for cells "
                     "is 100 million. Shrink it with Numbers… or pick S, M "
                     "or L.")
    # the whole patch names its ceiling the same way
    assert AP.whole_too_big_line(150 * MILLION) == (
        "This patch is 150.0 million voxels; a whole patch can be filled up "
        "to 100 million. Press S, M or L for a box.")


def test_a_box_over_the_preview_limit_never_reads_as_the_limit():
    assert AP.millions(100 * MILLION + 1) == "100.1"
    assert AP.millions(100 * MILLION) == "100.0"
    assert "100.0 million voxels" not in AP.over_limit_line(
        (1, 1, 100 * MILLION + 1))


def test_the_ready_line_and_the_clamp_sentence_name_the_preview_limit():
    assert AP.box_line((100, 100, 100), preset="cells", empty_share=0.42) == (
        "Box 100 × 100 × 100 voxels (1.0 million; the limit for cells is 100 "
        "million) · 42 % of it is empty space")
    assert AP.ADD_S1_LEFT_WHOLE_CLAMPED == (
        "The box is no longer the whole patch; for cells it was shrunk to the "
        "100 million voxel limit.")


def test_a_box_the_click_fill_stepped_aside_from_says_so():
    s = _state(FIVE, live_box_ok=False)
    assert AP.s1_sentence(s) == AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX
    assert AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX == (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH.replace("this patch", "this box"))
    # the tick off: the ready line
    assert not AP.s1_sentence(_state(FIVE, live_box_ok=False,
                                     live_on=False)).startswith(
        "Press Preview")


def test_a_flat_spot_step_aside_at_the_default_spacing_names_it():
    """The tissue outline's smallest piece is a volume in µm³: at the
    never-set 1.0 µm spacing debris in the dark can count as tissue and
    bring flat spots the true spacing would not -- the step aside says the
    spacing may change it."""
    box = _state(FIVE, live_box_ok=False, flat_spots=True,
                 default_spacing=True)
    assert AP.s1_sentence(box) == AP.ADD_S1_LIVE_STEPPED_ASIDE_SPACING_BOX
    assert AP.ADD_S1_LIVE_STEPPED_ASIDE_SPACING_BOX == (
        "Press Preview to see the cells — at the default 1.0 µm spacing "
        "this box is not updated as you click; Set spacing… in the Session "
        "list may change that.")
    patch = _state(FIVE, whole_patch=True, live_whole_ok=False,
                   flat_spots=True, default_spacing=True)
    assert AP.s1_sentence(patch) == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_SPACING_PATCH)
    # one family, the region the only difference
    assert AP.ADD_S1_LIVE_STEPPED_ASIDE_SPACING_PATCH == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_SPACING_BOX.replace("this box",
                                                         "this patch"))
    # a set spacing: the bare line
    assert AP.s1_sentence(dict(box, default_spacing=False)) == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX)
    assert AP.s1_sentence(dict(patch, default_spacing=False)) == (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH)
    # another step aside (walls, a failure) at the default: the bare line
    assert AP.s1_sentence(dict(box, flat_spots=False)) == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX)
    assert AP.s1_sentence(dict(patch, flat_spots=False)) == (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH)
    # the size row keeps precedence
    assert AP.s1_sentence(_state(TWENTY_FIVE, live_box_ok=False,
                                 flat_spots=True, default_spacing=True)) == (
        AP.ADD_S1_LIVE_OVER_GATE_BOX.format(n="20 million"))
    # run as you click: no step-aside line at all
    assert not AP.s1_sentence(_state(FIVE, flat_spots=True,
                                     default_spacing=True)).startswith(
        "Press Preview")


# --------------------------------------------------------------------------- #
# the gate on the page
# --------------------------------------------------------------------------- #
def _voxels(monkeypatch, hh, n):
    monkeypatch.setattr(type(hh), "_add_roi_voxels", lambda self: int(n))


def test_the_full_run_size_is_one_million():
    assert AP.ADD_LIVE_FULL_RUN_MAX_VOXELS == 1 * MILLION


def test_a_5_million_cells_box_is_live_with_the_click_fill_walls(
        ci, monkeypatch):
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    _voxels(monkeypatch, ci, 5 * MILLION)
    assert ci._add_live_possible()
    assert ci._add_box_elev_key() is not None, "its walls are kept"
    # over the full-run size a click needs the walls and the Grow by the
    # click fill redoes: a ridge wall or Spread waits for Preview
    ci.adv.wall = dict(ci.adv.wall, wall_kind="frangi")
    assert not ci._add_live_possible()
    assert AP.s1_sentence(ci._add_s1_state()) == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX)
    ci.adv.wall = dict(ci.adv.wall, wall_kind="membrane")
    ci.adv.params["growth"] = AP.GROW_SPREAD
    assert not ci._add_live_possible()
    ci.adv.params["growth"] = AP.GROW_FLOOD
    assert ci._add_live_possible()


@pytest.mark.parametrize("n", [660_000, 1 * MILLION])
def test_a_box_at_or_under_the_full_run_size_is_live_with_any_walls(
        ci, monkeypatch, n):
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    _voxels(monkeypatch, ci, n)
    for kind in ("membrane", "frangi", "meijering", "intensity"):
        ci.adv.wall = dict(ci.adv.wall, wall_kind=kind)
        assert ci._add_live_possible(), kind
        assert ci._add_s1_state()["live_box_ok"] is True
    ci.adv.params["growth"] = AP.GROW_SPREAD
    assert ci._add_live_possible()


def test_a_25_million_cells_box_is_not(ci, monkeypatch):
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    _voxels(monkeypatch, ci, 25 * MILLION)
    assert not ci._add_live_possible()
    assert ci._add_box_elev_key() is None
    assert ci._add_cells_stage_key() is None


# --------------------------------------------------------------------------- #
# the click fill's stage, by size and by the tick
# --------------------------------------------------------------------------- #
def test_a_5_million_box_with_the_tick_off_keeps_no_stage(co, monkeypatch):
    """Over the full-run size a box prepares the click fill only for clicks
    run as you click -- as the whole patch does: with the tick off no stage
    key, so nothing is prepared for clicks nobody runs live."""
    co._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    _voxels(monkeypatch, co, 5 * MILLION)
    assert not co._add_live_on()
    assert co._add_box_elev_key() is not None, "its walls are still kept"
    assert co._add_cells_stage_key() is None
    assert co._add_cells_sig() is None
    co._add_live_enabled = True
    assert co._add_cells_stage_key() is not None


@pytest.mark.parametrize("n", [660_000, 1 * MILLION])
def test_a_box_at_or_under_the_full_run_size_keeps_a_stage_tick_or_not(
        co, monkeypatch, n):
    co._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    _voxels(monkeypatch, co, n)
    assert not co._add_live_on()
    assert co._add_cells_stage_key() is not None


def test_a_large_box_preview_with_the_tick_off_runs_the_full_fill(
        co, monkeypatch):
    """The fixture's box stands for a 5 million one: a plain Preview with
    the tick off hands no stage, runs the full fill and keeps nothing."""
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    floods = _Floods(monkeypatch)
    co._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert co._add_roi_voxels() > AP.ADD_LIVE_FULL_RUN_MAX_VOXELS
    co._preview_add()
    kw = co.launched[-1][0]
    assert kw.get("stage_holder") is None and kw.get("stage_key") is None
    assert "live" not in kw["params"]
    _finish(co)
    assert floods.n == 1
    assert co.prepares.n == 0, "the click fill was prepared"
    res = co._add_preview_res
    assert res.ok and res.click_how.startswith("full run (")
    assert getattr(co, "_add_cells_stage", None) is None
    # the tick on: the same box hands its stage again
    co._add_live_enabled = True
    assert co._add_cells_stage_key() is not None


# --------------------------------------------------------------------------- #
# a live box click the click fill cannot take floods nothing
# --------------------------------------------------------------------------- #
class _Floods:
    def __init__(self, monkeypatch):
        self.n = 0
        real = R.reshape_empty_region

        def spy(*a, **k):
            self.n += 1
            return real(*a, **k)

        monkeypatch.setattr(R, "reshape_empty_region", spy)


def _answers(monkeypatch):
    got = []
    real = AP.run_add_preview

    def spy(**kw):
        res = real(**kw)
        got.append(res)
        return res

    monkeypatch.setattr(AP, "run_add_preview", spy)
    return got


def test_a_small_box_click_that_falls_back_floods_and_stays_live(
        ci, monkeypatch):
    """The fixture's box (10,816 voxels; a 0.66 million box alike) is at or
    under the full-run size: the click the click fill cannot take runs the
    full fill once, and the next click runs again."""
    monkeypatch.setattr(AP, "_click_fill_possible", lambda spec: False)
    floods = _Floods(monkeypatch)
    assert ci._add_roi_voxels() <= AP.ADD_LIVE_FULL_RUN_MAX_VOXELS
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    assert ci.launched[-1][0]["params"]["live"]
    _finish(ci)
    assert floods.n == 1
    res = ci._add_preview_res
    assert res.ok and res.click_how.startswith("full run (")
    assert (ci._add_preview_result == _id_at(ci, CELL_A)).any()
    assert ci._add_live_possible()
    assert ci._add_s1_state()["live_box_ok"] is True
    assert not AP.s1_sentence(ci._add_s1_state()).startswith("Press Preview")
    assert ci._add_marker_click(CELL_B)
    assert ci._add_live_timer.isActive()
    _fire(ci)
    _finish(ci)
    assert floods.n == 2
    assert (ci._add_preview_result == _id_at(ci, CELL_B)).any()


def test_a_large_box_click_that_falls_back_is_refused_and_pauses(
        ci, monkeypatch, capsys):
    """Over the full-run size (the fixture's box stands for a 5 million
    one) the click the click fill cannot take is refused, not flooded, and
    later clicks wait for Preview."""
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    monkeypatch.setattr(AP, "_click_fill_possible", lambda spec: False)
    floods = _Floods(monkeypatch)
    got = _answers(monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    kw = ci.launched[-1][0]
    assert kw["params"]["live"] and kw.get("stage_holder") is not None
    assert "box_rule" not in kw["params"], "a box keeps its closed faces"
    capsys.readouterr()
    _finish(ci)
    assert floods.n == 0, "the click flooded the box whole"
    res = got[-1]
    assert res.status == "refused"
    assert res.refusal_reason == AP.NOT_LIVE_HERE
    assert res.refusal is None and res.result is None
    assert ci._add_preview_result is None and ci._add_transient is None
    out = capsys.readouterr().out
    assert "[ADD] live run on the box stepped aside - press Preview" in out
    assert [tuple(m["zyx"]) for m in ci._add_click_markers] == [CELL_A]
    # S1 row 13.5 says it at once, and later clicks are gated
    assert ci._add_s1_state()["live_box_ok"] is False
    assert AP.s1_sentence(ci._add_s1_state()) == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX)
    assert not ci._add_live_possible()
    n = len(FakeThread.instances)
    assert ci._add_marker_click(CELL_B)
    assert not ci._add_live_timer.isActive(), "a click scheduled a run"
    assert len(FakeThread.instances) == n
    # a Preview fills the box with the plain flood, once
    ci._preview_add()
    assert len(FakeThread.instances) == n + 1
    kw = ci.launched[-1][0]
    assert "live" not in kw["params"]
    _finish(ci)
    assert floods.n == 1
    res = ci._add_preview_res
    assert res.ok and res.click_how.startswith("full run (")
    # a key change lets the box run as you click again
    ci._add_volume_epoch = int(getattr(ci, "_add_volume_epoch", 0)) + 1
    ci._add_prune_caches()
    assert ci._add_live_possible()


@pytest.mark.parametrize("default", [True, False])
def test_a_large_box_that_steps_aside_for_flat_spots_names_the_spacing(
        ci, monkeypatch, default):
    """A live click on a box over the full-run size whose click fill finds
    flat spots steps aside; at the never-set spacing S1 names the spacing,
    at a set one it keeps the bare line."""
    from packages.epicycle_annotator.reshape import incremental as INC

    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    monkeypatch.setattr(
        INC, "prepare",
        lambda *a, **k: (None, (INC.FULL_RUN, "[ADD] flat spots")))
    monkeypatch.setattr(type(ci), "_add_default_spacing",
                        lambda self: bool(default))
    floods = _Floods(monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    _finish(ci)
    assert floods.n == 0, "the click flooded the box whole"
    assert ci._add_cells_click_flat()
    state = ci._add_s1_state()
    assert state["live_box_ok"] is False
    assert state["flat_spots"] is True
    assert state["default_spacing"] is default
    assert AP.s1_sentence(state) == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_SPACING_BOX if default
        else AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX)


def test_a_failure_step_aside_at_the_default_spacing_keeps_the_bare_line(
        ci, monkeypatch):
    """A step aside that is not the flat spots (here the click fill
    switched off) is not about the spacing: the bare line."""
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    monkeypatch.setattr(AP, "_click_fill_possible", lambda spec: False)
    monkeypatch.setattr(type(ci), "_add_default_spacing", lambda self: True)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    _finish(ci)
    assert ci._add_cells_click_fell_back()
    assert not ci._add_cells_click_flat()
    assert AP.s1_sentence(ci._add_s1_state()) == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX)


def test_a_manual_box_preview_that_falls_back_still_floods(ci, monkeypatch):
    monkeypatch.setattr(AP, "_click_fill_possible", lambda spec: False)
    floods = _Floods(monkeypatch)
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    ci._preview_add()
    assert "live" not in ci.launched[-1][0]["params"]
    _finish(ci)
    assert floods.n == 1
    assert ci._add_preview_res.ok


# --------------------------------------------------------------------------- #
# the worker crops a box's channels
# --------------------------------------------------------------------------- #
def _crops(ci, kw):
    """``kw`` with every channel cut to the box, as the page used to."""
    roi = tuple(kw["roi"])
    out = dict(kw)
    out["wall_roi"] = np.array(kw["wall_roi"][roi], np.float32, copy=True)
    if kw.get("dark_rois"):
        out["dark_rois"] = {n: np.array(a[roi], np.float32, copy=True)
                            for n, a in kw["dark_rois"].items()}
    if kw.get("nuclei_roi") is not None:
        out["nuclei_roi"] = np.array(kw["nuclei_roi"][roi], np.float64,
                                     copy=True)
    return {k: v for k, v in out.items()
            if k not in ("cancel_event", "stage_holder", "stage_key")}


def test_a_box_hands_the_engine_its_channels_not_crops(ci):
    ci.adv.params["dark_outside"] = True
    names = ("mem", "mem2")
    ci._add_dark_cache = {ci._add_dark_key(): R.patch_dark_threshold(
        {n: ci._refine_channel_source(n) for n in names},
        spacing=ci._refine_spacing(), stride=1)}
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    kw, why = ci._add_gather_preview()
    assert why is None, why
    assert kw["wall_roi"] is ci._refine_channel_source("mem")
    assert set(kw["dark_rois"]) == {"mem", "mem2"}
    for name, arr in kw["dark_rois"].items():
        assert arr is ci._refine_channel_source(name), name
    assert kw["existing_roi"].shape == tuple(
        sl.stop - sl.start for sl in ci._add_roi)
    # the worker's crop answers as the page's crop did
    a = AP.run_add_preview(**{k: v for k, v in kw.items()
                              if k not in ("cancel_event", "stage_holder",
                                           "stage_key")})
    b = AP.run_add_preview(**_crops(ci, kw))
    assert a.ok and b.ok
    assert np.array_equal(a.result, b.result)


def test_a_nuclei_masks_box_is_cropped_on_the_worker(ci):
    ci.adv.source = ("nuclei_masks", None)
    ci.adv.nuclei_channel = "mem2"
    kw, why = ci._add_gather_preview()
    assert why is None, why
    assert kw["nuclei_roi"] is ci._refine_channel_source("mem2")
    a = AP.run_add_preview(**{k: v for k, v in kw.items()
                              if k not in ("cancel_event", "stage_holder",
                                           "stage_key")})
    b = AP.run_add_preview(**_crops(ci, kw))
    assert a.status == b.status
    if a.result is not None:
        assert np.array_equal(a.result, b.result)


# --------------------------------------------------------------------------- #
# S / M / L keep their sizes
# --------------------------------------------------------------------------- #
def _harness(monkeypatch, shape):
    install_fakes(monkeypatch)
    hh = Harness(np.zeros(shape, np.uint8), images={},
                 spacing=(0.5, 0.5, 0.5))
    z, y, x = shape
    hh.place_box((slice(0, z // 4), slice(0, y // 4), slice(0, x // 4)))
    hh.adv.preset = "cells"
    return hh


def _n(box):
    return int(box.sz) * int(box.sy) * int(box.sx)


def test_l_on_a_5_2_million_patch_is_unchanged(monkeypatch):
    hh = _harness(monkeypatch, (52, 316, 316))       # 5.19 million
    hh._set_add_box_size_fraction(0.5)
    assert (hh._add_box.sz, hh._add_box.sy, hh._add_box.sx) == (26, 158, 158)
    assert _n(hh._add_box) == 649_064
    assert hh._add_transient is None


def test_l_on_a_large_patch_stays_at_the_preset_size(monkeypatch):
    hh = _harness(monkeypatch, (300, 400, 400))      # 48 million
    hh._set_add_box_size_fraction(0.5)
    n = _n(hh._add_box)
    assert 0.9 * AP.ADD_CELLS_PRESET_MAX_VOXELS < n <= (
        AP.ADD_CELLS_PRESET_MAX_VOXELS)
    # a size, not a refusal: Preview is not refused for its size
    assert not str(AP.preview_block(hh._add_s1_state()) or "").startswith(
        "This box is")


def test_a_numbers_box_over_the_preset_size_is_kept(monkeypatch):
    hh = _harness(monkeypatch, (300, 400, 400))
    b = hh._add_box
    hh._on_add_box_changed(b.cz, b.cy, b.cx, 100, 200, 300)
    assert _n(hh._add_box) == 6 * MILLION
    assert not str(AP.preview_block(hh._add_s1_state()) or "").startswith(
        "This box is")


@pytest.mark.parametrize("frac", [0.125, 0.25, 0.5])
def test_nuclei_presets_are_never_capped(monkeypatch, frac):
    hh = _harness(monkeypatch, (300, 400, 400))
    hh.adv.preset = "nuclei"
    hh._set_add_box_size_fraction(frac)
    assert _n(hh._add_box) == (round(300 * frac) * round(400 * frac)
                               * round(400 * frac))


def test_a_uint16_wall_channel_answers_as_its_float_crop(ci):
    """The page's crop was float32; the worker now reads the layer's own
    dtype -- the engine normalises it to float32 first, so the answer is
    the same."""
    from napari.layers import Image

    mem = ci._refine_channel_source("mem")
    ci.viewer.layers.append(Image((mem * 1000).astype(np.uint16),
                                  name="mem16"))
    ci.adv.wall = dict(ci.adv.wall, channel="mem16")
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    kw, why = ci._add_gather_preview()
    assert why is None, why
    assert kw["wall_roi"].dtype == np.uint16
    a = AP.run_add_preview(**{k: v for k, v in kw.items()
                              if k not in ("cancel_event", "stage_holder",
                                           "stage_key")})
    b = AP.run_add_preview(**_crops(ci, kw))
    assert a.ok and b.ok
    assert np.array_equal(a.result, b.result)

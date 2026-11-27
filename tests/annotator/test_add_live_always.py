"""Always update as I click: the live restrictions the user can switch off,
and Grow by back on a Cells whole patch.

A tick under Advanced, shown only with "Update as I click" on and Cells ▸
Click markers, off by default and never remembered between napari
sessions.  With it on:

* the click fill keeps running where its flat spots would have sent the
  region to Preview (a few tied voxels may land differently from a
  Preview, which still fills exactly);
* where the click fill cannot run at all (walls it does not redo, Spread,
  a region over the live limit, a click fill that failed) the region is
  filled whole a moment after the clicks settle, as a Preview would fill
  it, on a live run: nothing greys, S1 says it is updating, Stop works,
  clicks during it queue one more run.

With it off everything is as before.  Grow by shows on a Cells whole
patch: Spread there grows the patch by Spread, Fill keeps the fill with a
background marker (or the box rule while filled as you click).
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import incremental as INC
from packages.epicycle_annotator.reshape import region as R

from .add_harness import FakeThread
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_cells_box import CELL_A, CELL_B, _fire, _finish, _id_at
from .test_add_live_cells_incremental import ci, co  # noqa: F401
from .test_add_live_cells_whole import _whole

MILLION = 1_000_000


@pytest.fixture
def wl(h, monkeypatch):  # noqa: F811
    return _whole(h, monkeypatch, live=True)


def _timer(hh):
    return getattr(hh, "_add_live_timer", None)


def _runs():
    return len(FakeThread.instances)


class _Floods:
    def __init__(self, monkeypatch):
        self.n = 0
        real = R.reshape_empty_region

        def spy(*a, **k):
            self.n += 1
            return real(*a, **k)

        monkeypatch.setattr(R, "reshape_empty_region", spy)


def _flat(monkeypatch, n_pairs=12):
    """Every survey reports flat spots (the step-aside's cause)."""
    real = INC.flat_survey

    def survey(*a, **k):
        _why, low, others, _n = real(*a, **k)
        return "a test's flat spots", low, others, int(n_pairs)

    monkeypatch.setattr(INC, "flat_survey", survey)


def _always(hh, on=True):
    hh._on_add_live_always_toggled(bool(on))


# --------------------------------------------------------------------------- #
# the tick on the page
# --------------------------------------------------------------------------- #
def _cells_clicks(w, whole=False):
    reset_add(w)
    w.set_add_preset("cells")
    w.update_add_ui_state(box_active=True, whole_patch=whole)
    w.update_add_ui_state(marker_sources=[
        (AP.ADD_ITEM_RUN.format(name="r2"), True, ("labels", "r2")),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))])
    keys = list(w._add_source_keys)
    w.add_marker_source_combo.setCurrentIndex(keys.index(("paint", None)))
    return keys


def _undo(w):
    w.add_live_cb.setChecked(True)
    w.add_live_always_cb.setChecked(False)
    w.add_grow_combo.setCurrentIndex(0)
    w.set_add_preset("nuclei")
    w.add_look_for_combo.setCurrentIndex(0)
    w.update_add_ui_state(box_active=False, whole_patch=False)
    reset_add(w)


def test_the_tick_is_off_and_shows_only_where_it_acts(widget):  # noqa: F811
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    w = widget
    cb = w.add_live_always_cb
    assert cb.text() == AP.ADD_LIVE_ALWAYS_LABEL == "Always update as I click"
    assert not cb.isChecked(), "off by default"
    assert cb.toolTip() == ADD_HELP_LINES["add_live_always_cb"] == (
        AP.ADD_HELP_LIVE_ALWAYS)
    assert AP.ADD_HELP_LIVE_ALWAYS == (
        "Runs the quick fill even where its flat spots could place a few "
        "tied voxels differently from Preview, and where the quick fill "
        "cannot run at all, redoes the region a moment after you stop "
        "clicking. That may take minutes on a large region or a slow Wall "
        "type; the first line says how long.")
    keys = _cells_clicks(w)
    try:
        assert not w.add_live_cb.isHidden()
        assert not cb.isHidden()
        # the live tick off: the tick does nothing, it hides
        w.add_live_cb.setChecked(False)
        assert cb.isHidden()
        w.add_live_cb.setChecked(True)
        assert not cb.isHidden()
        # another source: no click runs live
        w.add_marker_source_combo.setCurrentIndex(
            keys.index(("labels", "r2")))
        assert cb.isHidden()
        w.add_marker_source_combo.setCurrentIndex(
            keys.index(("paint", None)))
        assert not cb.isHidden()
        # a whole patch with Click markers: shown too
        w.update_add_ui_state(whole_patch=True)
        assert not cb.isHidden()
        # Nuclei, the markers included: hidden
        w.set_add_preset("nuclei")
        assert cb.isHidden()
        w.add_look_for_combo.setCurrentIndex(
            list(w._add_look_keys).index((AP.LOOK_MARKERS, None)))
        assert not w.add_live_cb.isHidden()
        assert cb.isHidden()
    finally:
        _undo(w)


def test_the_tick_is_wired_and_not_remembered(h):  # noqa: F811
    """The orchestrator's copy follows the tick; nothing about it is kept
    in the settings the live tick is kept in."""
    import inspect

    from napari_epicycle.annotator import curation_3d_widget as CW

    src = inspect.getsource(CW.Curation3DAnnotator._on_add_live_always_toggled)
    assert "_add_save_setting" not in src
    assert '"live_always"' not in inspect.getsource(
        CW.Curation3DAnnotator._add_restore_live_tick)
    assert not getattr(h, "_add_live_always_enabled", False)


# --------------------------------------------------------------------------- #
# the tick off: today's bytes
# --------------------------------------------------------------------------- #
def test_the_tick_off_keeps_the_key_the_gate_and_the_params(ci, monkeypatch):
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    key = ci._add_cells_stage_key()
    assert key is not None and "always" not in key
    kw, why = ci._add_gather_preview(live=True)
    assert why is None, why
    assert "live_always" not in kw["params"]
    assert kw["params"]["live"] is True
    ci._add_let_go_launch_cells()
    ci.adv.wall = dict(ci.adv.wall, wall_kind="frangi")
    assert not ci._add_live_possible()
    assert AP.s1_sentence(ci._add_s1_state()) == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX)
    assert ci._add_s1_state()["live_always"] is False
    # the tick ticked while Cells is not on Click markers does nothing
    ci._add_live_always_enabled = True
    ci.adv.source = ("labels", "r2")
    assert not ci._add_live_always()
    ci.adv.source = ("paint", None)
    assert ci._add_live_always()
    # nor with the live tick off
    ci._add_live_enabled = False
    assert not ci._add_live_always()


def test_the_tick_never_changes_the_stage_key(ci):
    """A good prepared box is good either way (a 5 million box takes
    about 10 s to prepare): the flip lets go only a stage it made
    meaningless, not every stage."""
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    off = ci._add_cells_stage_key()
    _always(ci)
    assert ci._add_cells_stage_key() == off
    _always(ci, False)
    assert ci._add_cells_stage_key() == off


# --------------------------------------------------------------------------- #
# flat spots: the click fill keeps running
# --------------------------------------------------------------------------- #
def test_prepare_keeps_the_stage_despite_flat_spots_when_allowed(ci,
                                                                 monkeypatch):
    _flat(monkeypatch)
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    kw, why = ci._add_gather_preview()
    assert why is None, why
    spec = AP.cells_spec(kw["params"])
    args = dict(roi=kw["roi"], spacing=kw["spacing"])
    existing = kw["existing_roi"]
    wall = np.asarray(kw["wall_roi"][kw["roi"]], np.float32)
    ci._add_let_go_launch_cells()
    stage, refusal = INC.prepare(existing, {"membrane": wall}, spec, **args)
    assert stage is None and refusal[0] == INC.FULL_RUN
    stage, refusal = INC.prepare(existing, {"membrane": wall}, spec,
                                 allow_flat=True, **args)
    assert refusal is None and stage is not None
    assert stage.n_flat_pairs == 12
    assert stage.flat_override == "a test's flat spots"


def test_flat_spots_keep_the_patch_live_with_the_tick_on(wl, monkeypatch,
                                                          capsys):
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    _flat(monkeypatch)
    floods = _Floods(monkeypatch)
    _always(wl)
    assert wl._add_marker_click(CELL_A)
    _fire(wl)
    kw = wl.launched[-1]
    assert kw["params"]["live"] and kw["params"]["live_always"]
    capsys.readouterr()
    _finish(wl)
    out = capsys.readouterr().out
    assert floods.n == 0, "the click flooded the whole patch"
    res = wl._add_preview_res
    assert res is not None and res.ok and res.by_click, wl._add_transient
    assert res.click_how.startswith("stage prepared, 12 flat pairs")
    assert ("[ADD] click fill kept despite 12 flat pairs (Always update is "
            "on)") in out
    kept = wl._add_cells_stage
    assert kept and kept.get("stage") is not None
    assert not kept.get("full") and not kept.get("fallback")
    assert not wl._add_cells_click_fell_back()
    assert wl._add_live_possible()
    assert not AP.s1_sentence(wl._add_s1_state()).startswith("Press Preview")
    # the next click reuses the stage, still by click
    assert wl._add_marker_click(CELL_B)
    _fire(wl)
    _finish(wl)
    res = wl._add_preview_res
    assert res.by_click and res.click_how.startswith("stage reused")
    assert len(wl.spies.prepares) == 1
    assert floods.n == 0
    assert (wl._add_preview_result == _id_at(wl, CELL_B)).any()


def test_a_preview_over_a_kept_flat_stage_fills_exactly(wl, monkeypatch):
    """Preview stays the reference: over a stage kept despite its flat
    spots it runs the full fill (the stage is kept for the next click)."""
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    _flat(monkeypatch)
    floods = _Floods(monkeypatch)
    _always(wl)
    assert wl._add_marker_click(CELL_A)
    _fire(wl)
    _finish(wl)
    stage = wl._add_cells_stage["stage"]
    assert not wl._add_cells_drawn_current(), (
        "a Preview must not take the windowed answer as the exact one")
    wl._preview_add()
    assert "live_always" not in wl.launched[-1]["params"]
    _finish(wl)
    assert floods.n == 1
    res = wl._add_preview_res
    assert res.ok and not res.by_click
    assert res.click_how == "full run (flat spots)"
    assert wl._add_cells_stage["stage"] is stage
    # the exact answer drawn: a second Preview runs nothing
    assert wl._add_cells_drawn_current()


def test_the_tick_off_mid_session_steps_aside_again(wl, monkeypatch, capsys):
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    _flat(monkeypatch)
    floods = _Floods(monkeypatch)
    _always(wl)
    assert wl._add_marker_click(CELL_A)
    _fire(wl)
    _finish(wl)
    assert wl._add_cells_stage["stage"] is not None
    capsys.readouterr()
    _always(wl, False)
    assert wl._add_cells_stage is None, "the stage kept despite flat spots"
    assert "[ADD] cells click stage dropped" in capsys.readouterr().out
    assert wl._add_marker_click(CELL_B)
    _fire(wl)
    _finish(wl)
    assert floods.n == 0
    assert wl._add_preview_result is None
    assert wl._add_cells_stage.get("full")
    assert not wl._add_live_possible()
    assert AP.s1_sentence(wl._add_s1_state()) == (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH)
    # ... and on again: the kept flat record no longer stops a click
    _always(wl)
    assert wl._add_live_possible()
    assert not wl._add_cells_click_fell_back()


# --------------------------------------------------------------------------- #
# where the click fill cannot run: the slow update
# --------------------------------------------------------------------------- #
def test_ridge_walls_on_a_large_box_update_after_the_clicks_settle(
        ci, monkeypatch):
    # the fixture's box stands for a 5 million one
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    floods = _Floods(monkeypatch)
    ci.adv.wall = dict(ci.adv.wall, wall_kind="frangi")
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert not ci._add_live_possible()
    _always(ci)
    assert ci._add_live_possible()
    assert ci._add_live_slow()
    assert AP.ADD_LIVE_SLOW_DEBOUNCE_MS == 700
    ci._add_click_markers = []
    assert ci._add_marker_click(CELL_A)
    assert _timer(ci).isActive()
    assert _timer(ci).starts[-1] == AP.ADD_LIVE_SLOW_DEBOUNCE_MS
    n = _runs()
    _fire(ci)
    assert _runs() == n + 1
    kw = ci.launched[-1][0]
    assert "live" not in kw["params"], "the region is filled whole"
    assert kw["params"]["live_always"] is True
    assert ci._add_live_run and not ci._add_busy
    state = ci._add_s1_state()
    assert AP.s1_sentence(state) == AP.ADD_S1_LIVE_UPDATING == (
        "Updating the region — this takes a few seconds here.")
    assert AP.s1_kind(state) == "busy"
    assert ci.adv.last("stop_enabled") is True, "Stop stays live"
    # a second click during the run queues one more and cancels the run
    # it made stale (here before its fill began)
    assert ci._add_marker_click(CELL_B)
    assert ci._add_live_dirty
    assert ci._add_worker.cancel_event.is_set()
    assert _runs() == n + 1
    _finish(ci)
    assert floods.n == 0, "the cancelled run filled nothing"
    # the queued run follows, for both clicks
    assert _runs() == n + 2, "the queued click ran"
    assert ci._add_live_run
    assert AP.s1_sentence(ci._add_s1_state()) == AP.ADD_S1_LIVE_UPDATING
    _finish(ci)
    assert floods.n == 1
    assert (ci._add_preview_result == _id_at(ci, CELL_A)).any()
    assert (ci._add_preview_result == _id_at(ci, CELL_B)).any()
    assert not ci._add_live_run
    assert AP.s1_sentence(ci._add_s1_state()) != AP.ADD_S1_LIVE_UPDATING


def test_the_slow_update_names_its_time_and_logs_it(ci, monkeypatch,
                                                   capsys):
    """S1 says how long the region takes, estimated from its voxels, its
    walls and Grow by (a 4.2 million box with Vessel ridges walls was
    measured at 58.6 s); the console says how long it really took."""
    import re

    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    ci.adv.wall = dict(ci.adv.wall, wall_kind="frangi")
    monkeypatch.setattr(type(ci), "_add_roi_voxels",
                        lambda self: 4_200_000)
    _always(ci)
    assert ci._add_live_slow()
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    assert ci._add_live_run and ci._add_live_slow_run
    state = ci._add_s1_state()
    want = AP.live_update_seconds(4_200_000, "frangi", AP.GROW_FLOOD)
    assert state["live_updating_seconds"] == pytest.approx(want)
    assert AP.s1_sentence(state) == (
        "Updating the region — about a minute here.")
    capsys.readouterr()
    _finish(ci)
    out = capsys.readouterr().out
    assert re.search(r"\[ADD\] live update of the region took \d+\.\d s "
                     r"\(estimated \d+ s\)", out), out
    # done: no time is left in S1's state
    assert ci._add_s1_state().get("live_updating_seconds") is None


def test_a_slow_update_can_be_stopped(ci, monkeypatch):
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    ci.adv.wall = dict(ci.adv.wall, wall_kind="frangi")
    _always(ci)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    assert ci._add_live_run
    ci._stop_add_preview()
    assert ci._add_stopping
    _finish(ci)
    assert not ci._add_live_run and not ci._add_stopping
    assert ci._add_preview_result is None
    assert ci._add_transient == AP.ADD_S1_STOPPED
    assert not ci._add_live_dirty


def test_a_box_over_the_live_limit_takes_the_slow_update(ci, monkeypatch):
    """A box over 20 million voxels (the fixture's box stands for 25
    million) updates a moment after the clicks with the tick on, up to the
    Preview limit; over that limit nothing runs."""
    n = ci._add_roi_voxels()
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_MAX_VOXELS", n - 1)
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert not ci._add_live_possible()
    assert AP.s1_sentence(ci._add_s1_state()).startswith("Press Preview")
    _always(ci)
    assert ci._add_live_possible() and ci._add_live_slow()
    assert not AP.s1_sentence(ci._add_s1_state()).startswith("Press Preview")
    floods = _Floods(monkeypatch)
    ci._add_click_markers = []
    assert ci._add_marker_click(CELL_A)
    assert _timer(ci).starts[-1] == AP.ADD_LIVE_SLOW_DEBOUNCE_MS
    _fire(ci)
    assert "live" not in ci.launched[-1][0]["params"]
    _finish(ci)
    assert floods.n == 1
    assert (ci._add_preview_result == _id_at(ci, CELL_A)).any()
    # over the Preview limit: not even slowly
    monkeypatch.setattr(type(ci), "_add_roi_voxels",
                        lambda self: AP.ADD_BOX_BUDGET + 1)
    assert not ci._add_live_possible()


def test_a_25_million_box_is_live_slowly_by_size(ci, monkeypatch):
    monkeypatch.setattr(type(ci), "_add_roi_voxels",
                        lambda self: 25 * MILLION)
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert not ci._add_live_possible()
    _always(ci)
    assert ci._add_live_possible() and ci._add_live_slow()


def test_a_click_fill_that_failed_takes_the_slow_update(wl, monkeypatch):
    """The click fill switched off (a failure that is not the flat spots):
    with the tick on the patch is filled whole a moment after the clicks."""
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    monkeypatch.setattr(AP, "_INCREMENTAL_OFF", True)
    floods = _Floods(monkeypatch)
    _always(wl)
    assert wl._add_marker_click(CELL_A)
    # the first click tries the click fill (nothing is known yet)
    assert _timer(wl).starts[-1] == AP.ADD_LIVE_DEBOUNCE_MS
    _fire(wl)
    _finish(wl)
    assert floods.n == 0
    # the step aside runs the slow update at once (it holds the record)
    assert _runs() == 2
    kw = wl.launched[-1]
    assert "live" not in kw["params"]
    assert kw["stage_holder"].get("fallback")
    assert AP.s1_sentence(wl._add_s1_state()) == AP.ADD_S1_LIVE_UPDATING
    _finish(wl)
    assert floods.n == 1
    assert (wl._add_preview_result == _id_at(wl, CELL_A)).any()
    assert wl._add_cells_stage.get("fallback")
    assert wl._add_live_slow()
    assert not AP.s1_sentence(wl._add_s1_state()).startswith("Press Preview")
    # the next click waits the slow quiet
    assert wl._add_marker_click(CELL_B)
    assert _timer(wl).starts[-1] == AP.ADD_LIVE_SLOW_DEBOUNCE_MS


def test_with_the_tick_on_no_row_13_5_sentence_shows():
    s = {"has_patch": True, "has_box": True, "preset": "cells",
         "source": AP.SOURCE_CLICKS, "wall_channel": "mem",
         "box_shape": (250, 100, 1000), "box_budget": AP.ADD_BOX_BUDGET,
         "whole_patch": False, "n_clicks": 1, "n_ticked": 1,
         "place_markers_on": True, "live_on": True, "empty_share": 1.0,
         "live_whole_ok": False, "live_box_ok": False, "fill_pending": True,
         "flat_spots": True, "default_spacing": True}
    assert AP.s1_sentence(s).startswith("Press Preview")
    s["live_always"] = True
    assert not AP.s1_sentence(s).startswith("Press Preview")
    # the click fill still being compiled is a wait with the tick on too
    assert AP.s1_sentence(s) == AP.ADD_S1_LIVE_PREPARING
    assert AP.s1_sentence(dict(s, fill_pending=False)) != (
        AP.ADD_S1_LIVE_PREPARING)
    assert not AP.s1_sentence(dict(s, whole_patch=True)).startswith(
        "Press Preview")
    s["live_updating"] = True
    assert AP.s1_sentence(s) == AP.ADD_S1_LIVE_UPDATING
    assert AP.s1_kind(s) == "busy"


# --------------------------------------------------------------------------- #
# Grow by on a whole patch
# --------------------------------------------------------------------------- #
def test_cells_growth_honours_spread_on_a_whole_patch():
    assert AP.cells_growth({"whole_patch": True,
                            "growth": AP.GROW_SPREAD}) == AP.GROW_SPREAD
    assert AP.cells_growth({"whole_patch": True,
                            "growth": AP.GROW_FLOOD}) == AP.GROW_PATCH
    assert AP.cells_growth({"whole_patch": True}) == AP.GROW_PATCH
    # the box rule (filled as you click) is the flood only
    assert AP.cells_growth({"whole_patch": True, "box_rule": True,
                            "growth": AP.GROW_FLOOD}) == AP.GROW_FLOOD
    spec = AP.cells_spec({"whole_patch": True, "growth": AP.GROW_SPREAD,
                          "source": AP.SOURCE_CLICKS, "wall_cost": 7.0})
    assert spec.params["growth"] == AP.GROW_SPREAD
    assert spec.params["whole_patch"] is True
    assert spec.params["wall_cost"] == 7.0
    # the engine never caps the competition on a whole patch (it grows in
    # tiles); the page's ceiling is the whole fill's
    assert R._budget(spec, (500, 500, 500)) == 125 * MILLION
    assert R.WHOLE_FILL_MAX_VOXELS == AP.ADD_BOX_BUDGET


def test_a_whole_patch_preview_grows_by_the_pick(wl):
    from packages.epicycle_annotator.reshape.compete import (
        COMPETE_TILE_VOXELS,
    )

    assert COMPETE_TILE_VOXELS == 16 * MILLION
    wl._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    wl.adv.params["growth"] = AP.GROW_SPREAD
    assert not wl._add_cells_live_whole_rule()
    assert wl._add_cells_stage_key() is None
    kw, why = wl._add_gather_preview()
    assert why is None, why
    assert "box_rule" not in kw["params"]
    assert AP.cells_growth(kw["params"]) == AP.GROW_SPREAD
    wl._add_let_go_launch_cells()
    wl.adv.params["growth"] = AP.GROW_FLOOD
    wl._add_live_enabled = False
    kw, why = wl._add_gather_preview()
    assert why is None, why
    assert AP.cells_growth(kw["params"]) == AP.GROW_PATCH


def test_spread_on_a_live_whole_patch_waits_with_the_tick_off(wl):
    wl._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert wl._add_live_possible()
    wl.adv.params["growth"] = AP.GROW_SPREAD
    assert not wl._add_live_possible()
    assert not wl._add_cells_live_whole()
    assert AP.s1_sentence(wl._add_s1_state()) == (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH)
    # the tick on: the slow update, grown by Spread
    _always(wl)
    assert wl._add_live_possible() and wl._add_live_slow()
    kw, why = wl._add_gather_preview(live=True)
    assert why is None, why
    assert "live" not in kw["params"]
    assert AP.cells_growth(kw["params"]) == AP.GROW_SPREAD


def test_grow_by_shows_on_a_whole_patch(widget):  # noqa: F811
    w = widget
    reset_add(w)
    w.set_add_preset("cells")
    w.update_add_ui_state(box_active=True, whole_patch=True, numba_ok=True)
    w._add_advanced_toggle.setChecked(True)
    try:
        assert not w._add_grow_row.isHidden()
        g = w.add_grow_combo
        for i in range(g.count()):
            assert g.model().item(i).isEnabled(), i
        assert g.itemText(0) == AP.ADD_GROW_FLOOD
        assert g.itemText(1) == AP.ADD_GROW_SPREAD
        # Fill: Spread's rows hidden, the fill's line
        g.setCurrentIndex(AP.GROW_KEYS.index(AP.GROW_FLOOD))
        assert w._add_wall_cost_row.isHidden()
        assert w._add_growth_limit_row.isHidden()
        assert w._add_whole_fill_label.text() == AP.ADD_HELP_WHOLE_FILL
        assert not w.add_include_membrane_cb.isHidden()
        # Spread: its rows and the membrane tick, Spread's line, its summary
        g.setCurrentIndex(AP.GROW_KEYS.index(AP.GROW_SPREAD))
        assert not w._add_wall_cost_row.isHidden()
        assert not w._add_growth_limit_row.isHidden()
        assert not w.add_include_membrane_cb.isHidden()
        assert w._add_whole_fill_label.text() == (
            AP.ADD_HELP_WHOLE_FILL_SPREAD)
        title = str(w._add_advanced_toggle.property(
            "_epicycle_chevron_title"))
        assert "spreading" in title
        # filled as you click (Fill, the box rule): its line
        g.setCurrentIndex(AP.GROW_KEYS.index(AP.GROW_FLOOD))
        w.update_add_ui_state(live_whole_rule=True)
        assert w._add_whole_fill_label.text() == AP.ADD_HELP_WHOLE_FILL_LIVE
    finally:
        w.update_add_ui_state(live_whole_rule=False)
        w._add_advanced_toggle.setChecked(False)
        _undo(w)


def test_the_whole_fill_help_names_each_pick():
    assert AP.whole_fill_help(False, spread=True) == (
        AP.ADD_HELP_WHOLE_FILL_SPREAD)
    assert AP.whole_fill_help(False) == AP.ADD_HELP_WHOLE_FILL
    assert AP.whole_fill_help(True, True) == AP.ADD_HELP_WHOLE_FILL_LIVE
    assert AP.whole_fill_help(True, False) == AP.ADD_HELP_WHOLE_FILL_PREVIEW
    assert "Spread" in AP.ADD_HELP_WHOLE_FILL_SPREAD

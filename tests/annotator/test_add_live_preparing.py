"""Update as I click on a Cells region over a million voxels while the
click fill is still being compiled.

The click fill's flood is compiled once per process, on the solver-check
worker started at the first box (a few seconds).  A live click on a region
over ``ADD_LIVE_FULL_RUN_MAX_VOXELS`` that lands before it is ready is not
a fallback: nothing is remembered against the region, the run is refused
as "preparing" (S1 says the clicks will be drawn in a moment, the markers
stay), and when the check finishes the live run fires once and draws.  A
region the click fill genuinely cannot take (flat spots, switched off)
still steps aside, and S1 and the Advanced whole-patch line both say so.
A large whole patch filled as you click says in amber that its first
update prepares it once.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.annotator import curation_3d_widget as CW
from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import incremental as INC

from .add_harness import FakeThread
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_cells_box import CELL_A, CELL_B, _fire, _finish, _id_at
from .test_add_live_cells_incremental import ci  # noqa: F401
from .test_add_live_cells_whole import (
    _answers,
    _flat_everywhere,
    _Floods,
    _over_the_full_run_size,
    wl,  # noqa: F401
)


def _cold(monkeypatch):
    """The flood not compiled yet, the solver check still running."""
    monkeypatch.setattr(INC, "_WARM", False)
    monkeypatch.setattr(AP, "_CLICK_FILL_SETTLED", False)
    monkeypatch.setitem(CW._ADD_NUMBA, "ok", None)


def _compiled(monkeypatch):
    """The solver check's answer: the flood compiled (the fixture warmed
    it for real), the module's answer kept."""
    monkeypatch.setattr(INC, "_WARM", True)
    monkeypatch.setattr(AP, "_CLICK_FILL_SETTLED", True)
    monkeypatch.setitem(CW._ADD_NUMBA, "ok", True)


def _runs():
    return len(FakeThread.instances)


def _waits(hh, res, capsys_out, region):
    """A live click on a cold region: refused as preparing, no fallback
    kept, the markers kept, S1 the preparing sentence."""
    assert res.status == "refused"
    assert res.refusal_reason == AP.LIVE_PREPARING
    assert res.refusal is None and res.result is None
    assert hh._add_preview_result is None and hh._add_transient is None
    kept = getattr(hh, "_add_cells_stage", None) or {}
    assert not kept.get("fallback") and not kept.get("full")
    assert not hh._add_cells_click_fell_back()
    assert hh._add_live_possible(), "live paused for good"
    state = hh._add_s1_state()
    assert state["fill_pending"] is True
    assert AP.s1_sentence(state) == AP.ADD_S1_LIVE_PREPARING
    assert AP.s1_kind(state) == "busy"
    assert "[ADD] live run waits for the fill to compile" in capsys_out
    assert "stepped aside" not in capsys_out
    assert "[ADD] cells by click:" not in capsys_out
    assert "Preview" not in AP.s1_sentence(state)
    del region


# --------------------------------------------------------------------------- #
# the sentence
# --------------------------------------------------------------------------- #
def test_the_preparing_sentence():
    assert AP.ADD_S1_LIVE_PREPARING == (
        "Preparing the fill — your clicks will be drawn in a moment.")
    assert len(AP.ADD_S1_LIVE_PREPARING) <= 80
    assert "numba" not in AP.ADD_S1_LIVE_PREPARING.lower()
    assert AP.LIVE_PREPARING != AP.NOT_LIVE_HERE


def test_s1_says_preparing_only_where_a_click_would_wait():
    base = {"has_patch": True, "has_box": True, "preset": "cells",
            "source": AP.SOURCE_CLICKS, "wall_channel": "mem",
            "box_shape": (50, 100, 1000), "box_budget": AP.ADD_BOX_BUDGET,
            "whole_patch": False, "n_clicks": 1, "n_ticked": 1,
            "place_markers_on": True, "live_on": True, "empty_share": 1.0,
            "live_whole_ok": True, "live_box_ok": True,
            "fill_pending": True}
    assert AP.s1_sentence(base) == AP.ADD_S1_LIVE_PREPARING
    # Preview is never greyed by it
    assert AP.preview_block(base) is None
    # a region that stepped aside says so instead
    assert AP.s1_sentence(dict(base, live_box_ok=False)) == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX)
    assert AP.s1_sentence(dict(base, whole_patch=True,
                               live_whole_ok=False)) == (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH)
    # no marker to draw: the marker line comes first
    assert AP.s1_sentence(dict(base, n_clicks=0, n_ticked=0)) == (
        AP.ADD_S1_NO_MARKERS_ON)
    # the tick off: the ready line
    assert not AP.s1_sentence(dict(base, live_on=False)).startswith(
        "Preparing")


# --------------------------------------------------------------------------- #
# a box over a million voxels
# --------------------------------------------------------------------------- #
def test_a_cold_click_on_a_large_box_waits_then_draws_once(ci, monkeypatch,
                                                           capsys):
    _over_the_full_run_size(monkeypatch)
    _cold(monkeypatch)
    floods = _Floods(monkeypatch)
    got = _answers(monkeypatch)
    assert ci._add_marker_click(CELL_A)
    # before the run lands S1 already says the clicks wait
    assert AP.s1_sentence(ci._add_s1_state()) == AP.ADD_S1_LIVE_PREPARING
    _fire(ci)
    assert ci.launched[-1][0]["params"]["live"]
    capsys.readouterr()
    _finish(ci)
    assert floods.n == 0, "the click flooded the box whole"
    assert ci.prepares.n == 0
    _waits(ci, got[-1], capsys.readouterr().out, "box")
    assert [tuple(m["zyx"]) for m in ci._add_click_markers] == [CELL_A]
    # a second click during the compile is placed and waits too
    assert ci._add_marker_click(CELL_B)
    _fire(ci)
    _finish(ci)
    assert got[-1].refusal_reason == AP.LIVE_PREPARING
    n = _runs()
    # the check finishes: one live run, both clicks drawn by the click fill
    _compiled(monkeypatch)
    ci._on_add_numba_finished(None)
    assert ci._add_live_timer.isActive()
    _fire(ci)
    assert _runs() == n + 1
    _finish(ci)
    res = ci._add_preview_res
    assert res is not None and res.ok and res.by_click
    assert (ci._add_preview_result == _id_at(ci, CELL_A)).any()
    assert (ci._add_preview_result == _id_at(ci, CELL_B)).any()
    assert floods.n == 0
    assert not ci._add_live_timer.isActive(), "a second run is owed"
    assert ci._add_s1_state()["fill_pending"] is False


def test_the_check_finishing_with_nothing_waiting_runs_nothing(ci,
                                                               monkeypatch):
    _over_the_full_run_size(monkeypatch)
    _compiled(monkeypatch)
    n = _runs()
    # no marker: nothing to draw
    ci._on_add_numba_finished(None)
    assert not ci._add_live_timer_active()
    # a result already drawn: kept, not run again
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    _finish(ci)
    drawn = ci._add_preview_res
    assert drawn.ok and drawn.by_click
    m = _runs()
    ci._on_add_numba_finished(None)
    assert not ci._add_live_timer_active()
    assert _runs() == m and ci._add_preview_res is drawn
    assert m == n + 1


def test_a_cold_manual_preview_on_a_large_box_keeps_live_on(ci, monkeypatch):
    """A Preview pressed during the compile runs the full fill, and is not
    remembered as a fallback: the next click runs as you click."""
    _over_the_full_run_size(monkeypatch)
    _cold(monkeypatch)
    floods = _Floods(monkeypatch)
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    ci._preview_add()
    assert "live" not in ci.launched[-1][0]["params"]
    _finish(ci)
    assert floods.n == 1
    res = ci._add_preview_res
    assert res.ok and res.click_how == (
        "full run (the flood is not compiled yet)")
    assert not ci._add_cells_click_fell_back()
    assert ci._add_live_possible()
    assert ci._add_s1_state()["live_box_ok"] is True


def test_a_flat_box_still_steps_aside(ci, monkeypatch, capsys):
    _over_the_full_run_size(monkeypatch)
    _compiled(monkeypatch)
    _flat_everywhere(monkeypatch)
    got = _answers(monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    capsys.readouterr()
    _finish(ci)
    assert got[-1].refusal_reason == AP.NOT_LIVE_HERE
    out = capsys.readouterr().out
    assert "[ADD] live run on the box stepped aside - press Preview" in out
    assert "waits for the fill to compile" not in out
    state = ci._add_s1_state()
    assert state["fill_pending"] is False
    assert AP.s1_sentence(state) == AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX


def test_a_switched_off_click_fill_is_not_preparing(ci, monkeypatch):
    """The click fill switched off for the process (numba could not run
    it) is a real fallback, never a wait."""
    _over_the_full_run_size(monkeypatch)
    _cold(monkeypatch)
    monkeypatch.setattr(AP, "_INCREMENTAL_OFF", True)
    got = _answers(monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    _finish(ci)
    assert got[-1].refusal_reason == AP.NOT_LIVE_HERE
    assert ci._add_s1_state()["fill_pending"] is False
    assert AP.s1_sentence(ci._add_s1_state()) == (
        AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX)


def test_a_check_that_could_not_compile_is_not_preparing(monkeypatch):
    """Once the solver check has answered, a flood still not compiled is a
    real fallback (the click fill cannot run here), never a wait."""
    spec = AP.cells_spec({"source": AP.SOURCE_CLICKS})
    _cold(monkeypatch)
    monkeypatch.setattr(AP, "_INCREMENTAL_OFF", False)
    assert AP.click_fill_compiling()
    assert AP._click_fill_off_why(spec) == "the flood is not compiled yet"
    monkeypatch.setattr(INC, "warm_kernel", lambda: False)
    from packages.epicycle_annotator.reshape import compete as CP

    monkeypatch.setattr(CP, "_kernels", lambda: {})
    assert AP.numba_answer() == (AP.NUMBA_OK, "")
    assert AP._CLICK_FILL_SETTLED is True
    assert not AP.click_fill_compiling()
    assert AP._click_fill_off_why(spec) == "the flood could not be compiled"


def test_the_check_settles_the_click_fill_when_numba_is_missing(
        monkeypatch):
    from packages.epicycle_annotator.reshape import compete as CP

    _cold(monkeypatch)

    def broken():
        raise ImportError("gone")

    monkeypatch.setattr(CP, "_kernels", broken)
    assert AP.numba_answer() == (AP.NUMBA_MISSING_KEY, "gone")
    assert not AP.click_fill_compiling()


def test_a_small_box_never_waits(ci, monkeypatch):
    """At or under a million voxels a cold click runs the full fill, as
    before: no wait, no preparing sentence."""
    _cold(monkeypatch)
    floods = _Floods(monkeypatch)
    assert ci._add_roi_voxels() <= AP.ADD_LIVE_FULL_RUN_MAX_VOXELS
    assert ci._add_s1_state()["fill_pending"] is False
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    _finish(ci)
    assert floods.n == 1
    assert ci._add_preview_res.ok


# --------------------------------------------------------------------------- #
# a whole patch over a million voxels, and the Advanced line
# --------------------------------------------------------------------------- #
def _help_line(hh):
    """The Advanced whole-patch line as the page's last push picks it."""
    return AP.whole_fill_help(bool(hh.adv.last("live_whole_rule")),
                              hh.adv.last("live_whole_ok"))


def test_a_cold_click_on_a_large_patch_waits_then_draws(wl, monkeypatch,
                                                        capsys):
    _over_the_full_run_size(monkeypatch)
    _cold(monkeypatch)
    floods = _Floods(monkeypatch)
    got = _answers(monkeypatch)
    assert wl._add_marker_click(CELL_A)
    _fire(wl)
    capsys.readouterr()
    _finish(wl)
    assert floods.n == 0
    _waits(wl, got[-1], capsys.readouterr().out, "whole patch")
    assert wl._add_cells_live_whole()
    assert wl._add_s1_state()["live_whole_ok"] is True
    # the Advanced line still says it is filled as you click
    assert _help_line(wl) == AP.ADD_HELP_WHOLE_FILL_LIVE
    _compiled(monkeypatch)
    wl._on_add_numba_finished(None)
    _fire(wl)
    _finish(wl)
    res = wl._add_preview_res
    assert res is not None and res.ok and res.by_click
    assert (wl._add_preview_result == _id_at(wl, CELL_A)).any()
    assert floods.n == 0


def test_a_flat_patch_steps_aside_and_the_advanced_line_agrees(
        wl, monkeypatch):
    _over_the_full_run_size(monkeypatch)
    _compiled(monkeypatch)
    _flat_everywhere(monkeypatch)
    wl._add_refresh_ui()
    assert _help_line(wl) == AP.ADD_HELP_WHOLE_FILL_LIVE
    assert wl._add_marker_click(CELL_A)
    _fire(wl)
    _finish(wl)
    state = wl._add_s1_state()
    assert AP.s1_sentence(state) == AP.ADD_S1_LIVE_OVER_GATE_PATCH
    assert wl.adv.last("live_whole_ok") is False
    line = _help_line(wl)
    assert line == AP.ADD_HELP_WHOLE_FILL_PREVIEW
    assert not line.startswith("Filled as you click")
    # a Preview still fills it by the box rule: far tissue fills too
    assert "fills too" in line


def test_the_whole_fill_help_picks_by_the_rule_and_live():
    assert AP.whole_fill_help(False, True) == AP.ADD_HELP_WHOLE_FILL
    assert AP.whole_fill_help(False, False) == AP.ADD_HELP_WHOLE_FILL
    assert AP.whole_fill_help(True, True) == AP.ADD_HELP_WHOLE_FILL_LIVE
    # not pushed: the live words (the rule alone, as before)
    assert AP.whole_fill_help(True, None) == AP.ADD_HELP_WHOLE_FILL_LIVE
    assert AP.whole_fill_help(True, False) == AP.ADD_HELP_WHOLE_FILL_PREVIEW
    assert AP.ADD_HELP_WHOLE_FILL_PREVIEW == (
        "Filled on Preview, the way a box is: tissue far from every marker "
        "fills too. To keep it empty, untick Update as I click and press "
        "Preview.")
    # no longer than the two lines it stands beside
    assert len(AP.ADD_HELP_WHOLE_FILL_PREVIEW) <= max(
        len(AP.ADD_HELP_WHOLE_FILL), len(AP.ADD_HELP_WHOLE_FILL_LIVE))


# --------------------------------------------------------------------------- #
# the amber note of a large patch filled as you click
# --------------------------------------------------------------------------- #
def test_a_large_live_patch_says_its_first_update_prepares_it(
        wl, monkeypatch):
    n = int(np.prod(wl.sync_manager.shared_labels["L"].shape))
    live = AP.ADD_NOTE_PATCH_SLOW_LIVE_CELLS.format(m=AP.millions(n))
    assert AP.ADD_SLOW_LIVE_PATCH_VOXELS == 5_000_000
    assert AP.ADD_SLOW_LIVE_PATCH_VOXELS < AP.ADD_LIVE_CELLS_MAX_VOXELS
    # the fixture's patch is small: no note
    assert wl._add_derive_note_text() != live
    monkeypatch.setattr(AP, "ADD_SLOW_LIVE_PATCH_VOXELS", 10)
    assert wl._add_derive_note_text() == live
    # once the patch is prepared (a result drawn, then dropped by a
    # setting), the note does not come back for the same key
    assert wl._add_marker_click(CELL_A)
    _fire(wl)
    _finish(wl)
    assert wl._add_preview_result is not None
    assert wl._add_derive_note_text() == ""
    wl.adv.params["min_size"] = 3
    wl._invalidate_add_preview(live_schedule=False)
    assert wl._add_preview_result is None
    assert wl._add_cells_stage and wl._add_cells_stage.get("stage")
    assert wl._add_derive_note_text() != live
    # the tick off: never the live note
    wl._add_live_enabled = False
    assert wl._add_derive_note_text() != live


@pytest.mark.parametrize("n", [5_000_000, 5_000_001, 20_000_000])
def test_the_live_note_spans_5_to_20_million(n):
    assert (AP.ADD_SLOW_LIVE_PATCH_VOXELS < n
            <= AP.ADD_LIVE_CELLS_MAX_VOXELS) == (n > 5_000_000)

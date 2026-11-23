"""Update as I click: Cells ▸ Click markers on a whole patch.

With the tick on and walls the click fill redoes (Membrane sheets,
Intensity edges), a whole patch is filled as you click by the BOX rule:
every marker fills to the walls, bounded by the tissue outline and the
dark rule, with no background marker -- through the click fill's stage,
prepared once (its wall picture is the whole patch's, kept) and redone
only where a click changed it.  A Preview with the tick on goes the same
way, and does nothing when the drawn result already answers the markers.
With the tick off a whole patch keeps the fill with a background marker,
byte for byte.  Other walls still wait for Preview, and S1 says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import incremental as INC
from packages.epicycle_annotator.reshape import region as R

from .add_harness import FakeThread
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_cells_box import CELL_A, CELL_B, _fire, _finish, _id_at
from .test_add_live_cells_incremental import _cells, _lines
from .test_add_live_cells_incremental import ci  # noqa: F401
from .test_add_live_nuclei_box import _FakeTimer


class _Spies:
    def __init__(self, monkeypatch):
        self.prepares = []
        real = INC.prepare

        def spy(*a, **k):
            self.prepares.append(k)
            return real(*a, **k)

        monkeypatch.setattr(INC, "prepare", spy)


def _launches(hh):
    got = []
    real = hh._launch_add_worker

    def spy(kwargs, *a, **k):
        got.append(dict(kwargs))
        return real(kwargs, *a, **k)

    hh._launch_add_worker = spy
    return got


def _whole(hh, monkeypatch, *, live):
    import qtpy.QtCore

    assert INC.warm_kernel(), "numba is needed for the click fill"
    monkeypatch.setattr(AP, "_INCREMENTAL_OFF", False)
    _FakeTimer.made = []
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    _cells(hh, live=live)
    vol = hh.sync_manager.shared_labels["L"]
    hh._add_whole_patch = True
    hh.place_box(tuple(slice(0, s) for s in vol.shape))
    hh.ctx["spacing_source"] = "qc_settings"
    hh.spies = _Spies(monkeypatch)
    hh.launched = _launches(hh)
    return hh


@pytest.fixture
def wl(h, monkeypatch):  # noqa: F811
    return _whole(h, monkeypatch, live=True)


@pytest.fixture
def wo(h, monkeypatch):  # noqa: F811
    return _whole(h, monkeypatch, live=False)


def _click_and_draw(hh, at):
    assert hh._add_marker_click(at)
    _fire(hh)
    _finish(hh)
    assert hh._add_preview_result is not None, hh._add_transient
    return hh._add_preview_res


def _runs():
    return len(FakeThread.instances)


# --------------------------------------------------------------------------- #
# the growth rule
# --------------------------------------------------------------------------- #
def test_the_growth_rule_of_a_live_whole_patch_is_the_box_rule():
    assert AP.cells_growth({"whole_patch": True, "box_rule": True}) == (
        AP.GROW_FLOOD)
    # whatever Grow by says: a whole patch is filled to the walls
    assert AP.cells_growth({"whole_patch": True, "box_rule": True,
                            "growth": AP.GROW_SPREAD}) == AP.GROW_FLOOD
    assert AP.cells_growth({"whole_patch": True}) == AP.GROW_PATCH
    assert AP.cells_growth({"whole_patch": True,
                            "growth": AP.GROW_SPREAD}) == AP.GROW_PATCH
    assert AP.cells_growth({"growth": AP.GROW_SPREAD}) == AP.GROW_SPREAD
    assert AP.cells_growth({}) == AP.GROW_FLOOD
    spec = AP.cells_spec({"whole_patch": True, "box_rule": True,
                          "source": AP.SOURCE_CLICKS})
    # the whole patch runs as one box: the flood takes no whole patch, and
    # the whole fill's ceiling is its budget
    assert INC.is_click_fill(spec) and not spec.params["whole_patch"]
    assert spec.params["box_budget"] == R.WHOLE_FILL_MAX_VOXELS
    assert spec.params["growth"] == AP.GROW_FLOOD


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_a_whole_patch_runs_live_with_membrane_or_edges_walls(wl):
    assert not wl._add_live_possible()          # no marker yet
    wl._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    for kind in INC.CLICK_FILL_WALLS:
        wl.adv.wall = dict(wl.adv.wall, wall_kind=kind)
        assert wl._add_live_possible(), kind
        assert wl._add_cells_live_whole(), kind
    for kind in ("frangi", "meijering", "intensity", "distance"):
        wl.adv.wall = dict(wl.adv.wall, wall_kind=kind)
        assert not wl._add_live_possible(), kind
        assert not wl._add_cells_live_whole(), kind
    wl.adv.wall = dict(wl.adv.wall, wall_kind="membrane")
    wl.adv.source = ("labels", "r2")
    assert not wl._add_live_possible()
    wl.adv.source = ("paint", None)


def test_with_the_tick_off_a_whole_patch_is_not_live(wo):
    wo._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert not wo._add_live_possible()
    assert not wo._add_cells_live_whole()
    assert wo._add_cells_stage_key() is None


def _patch_state(**kw):
    s = {"has_patch": True, "has_box": True, "preset": "cells",
         "source": AP.SOURCE_CLICKS, "wall_channel": "mem",
         "box_shape": (20, 30, 30), "whole_patch": True, "n_clicks": 1,
         "n_ticked": 1, "place_markers_on": True, "live_on": True,
         "empty_share": 1.0}
    s.update(kw)
    return s


def test_s1_row_13_5_holds_only_for_the_walls_still_gated(wl):
    assert AP.ADD_S1_LIVE_OVER_GATE_PATCH == (
        "Press Preview to see the cells — this patch is not updated as you "
        "click.")
    assert AP.s1_sentence(_patch_state()) == AP.ADD_S1_LIVE_OVER_GATE_PATCH
    ok = _patch_state(live_whole_ok=True)
    assert AP.s1_sentence(ok) != AP.ADD_S1_LIVE_OVER_GATE_PATCH
    assert "not updated as you click" not in AP.s1_sentence(ok)
    # on the page: Membrane sheets runs live, Vessel ridges waits
    wl._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert AP.s1_sentence(wl._add_s1_state()) != (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH)
    wl.adv.wall = dict(wl.adv.wall, wall_kind="frangi")
    assert AP.s1_sentence(wl._add_s1_state()) == (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH)


# --------------------------------------------------------------------------- #
# the rule switch, and the tick-off path byte for byte
# --------------------------------------------------------------------------- #
def test_a_live_click_fills_the_whole_patch_by_the_box_rule(wl):
    res = _click_and_draw(wl, CELL_A)
    kw = wl.launched[-1]
    assert AP.cells_growth(kw["params"]) == AP.GROW_FLOOD
    assert kw["params"]["whole_patch"] and kw["params"]["box_rule"]
    assert "stage_holder" in kw
    assert res.by_click and res.click_how == "stage prepared"
    assert res.whole_patch
    # the same run with no stage is the full flood: the same answer
    full = AP.run_add_preview(**{k: v for k, v in kw.items()
                                 if k not in ("cancel_event", "stage_holder",
                                              "stage_key")})
    assert full.ok and not full.by_click
    assert np.array_equal(res.result, full.result)
    assert res.ids == full.ids and res.sizes == full.sizes


def test_with_the_tick_off_a_whole_patch_runs_as_before(wo):
    wo._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True},
                             {"zyx": CELL_B, "id": 9, "ticked": True}]
    wo._preview_add()
    _finish(wo)
    res = wo._add_preview_res
    assert res is not None and res.ok, wo._add_transient
    kw = {k: v for k, v in wo.launched[-1].items() if k != "cancel_event"}
    assert "stage_holder" not in kw and "stage_key" not in kw
    assert "box_rule" not in kw["params"]
    assert AP.cells_growth(kw["params"]) == AP.GROW_PATCH
    # no prepared patch, and the console line says why the run was whole
    assert not res.by_click
    assert res.click_how == (
        "full run (the whole patch's fill with a background)")
    assert wo.spies.prepares == []
    again = AP.run_add_preview(**kw)
    assert np.array_equal(again.result, res.result)
    assert again.ids == res.ids and again.sizes == res.sizes


# --------------------------------------------------------------------------- #
# the stage, and the whole patch's wall picture
# --------------------------------------------------------------------------- #
def test_a_second_click_reuses_the_stage(wl, capsys):
    _click_and_draw(wl, CELL_A)
    stage = wl._add_cells_stage["stage"]
    capsys.readouterr()
    res = _click_and_draw(wl, CELL_B)
    assert res.by_click and res.click_how == "stage reused"
    assert len(wl.spies.prepares) == 1
    assert wl._add_cells_stage["stage"] is stage
    lines = _lines(capsys.readouterr().out)
    assert len(lines) == 1 and "stage reused" in lines[0]
    for at in (CELL_A, CELL_B):
        assert (wl._add_preview_result == _id_at(wl, at)).any()


# --------------------------------------------------------------------------- #
# the faces of a whole patch are the edge of the data, not a cut
# --------------------------------------------------------------------------- #
def test_a_live_whole_patch_fills_to_its_faces(wl):
    """The membrane column of CELL_A is open in z: filled as you click it
    reaches both z faces, as the tick-off fill does -- not a cell cut short
    halfway to each face by a face marker."""
    res = _click_and_draw(wl, CELL_A)
    nz = wl.sync_manager.shared_labels["L"].shape[0]
    cid = _id_at(wl, CELL_A)
    zs = np.nonzero((wl._add_preview_result == cid).any(axis=(1, 2)))[0]
    assert (int(zs.min()), int(zs.max())) == (0, nz - 1), zs
    assert not res.edge_ids
    kw = wl.launched[-1]
    assert kw["params"]["box_rule"]
    spec = AP.cells_spec(kw["params"])
    assert spec.params["faces_outside"] is False
    # the tick-off fill of the same marker: the same z range
    off = dict(kw["params"])
    off.pop("box_rule")
    one = AP.run_add_preview(**dict(
        {k: v for k, v in kw.items()
         if k not in ("cancel_event", "stage_holder", "stage_key")},
        params=off))
    assert one.ok and AP.cells_growth(off) == AP.GROW_PATCH
    zs_off = np.nonzero((one.result == cid).any(axis=(1, 2)))[0]
    assert (int(zs_off.min()), int(zs_off.max())) == (0, nz - 1)
    # a box keeps its faces in the outside marker
    assert AP.cells_spec({"source": AP.SOURCE_CLICKS}).params[
        "faces_outside"] is True
    assert AP.cells_spec({"source": AP.SOURCE_CLICKS, "whole_patch": True}
                         ).params["faces_outside"] is True


# --------------------------------------------------------------------------- #
# a patch the click fill cannot take is not run by a click
# --------------------------------------------------------------------------- #
def _flat_everywhere(monkeypatch):
    real = INC.flat_survey

    def survey(*a, **k):
        _why, low, others, n = real(*a, **k)
        return "a test's flat spots", low, others, n

    monkeypatch.setattr(INC, "flat_survey", survey)


class _Floods:
    """Every full flood of the engine (the whole-patch fill a click must
    never start where the click fill steps aside)."""

    def __init__(self, monkeypatch):
        self.n = 0
        real = R.reshape_empty_region

        def spy(*a, **k):
            self.n += 1
            return real(*a, **k)

        monkeypatch.setattr(R, "reshape_empty_region", spy)


def _answers(monkeypatch):
    """Every answer the worker got from run_add_preview."""
    got = []
    real = AP.run_add_preview

    def spy(**kw):
        res = real(**kw)
        got.append(res)
        return res

    monkeypatch.setattr(AP, "run_add_preview", spy)
    return got


def _stepped_aside(wl, res, how):
    """A live run the click fill could not take: no flood, nothing drawn,
    the markers kept, S1's gate at once."""
    assert res.status == "refused"
    assert res.refusal_reason == AP.NOT_LIVE_HERE == "not-live-here"
    assert res.refusal is None and res.result is None
    assert res.click_how == how
    assert wl._add_preview_result is None and wl._add_preview_res is None
    assert wl._add_transient is None
    assert not wl._add_cells_live_whole()
    assert wl._add_s1_state()["live_whole_ok"] is False
    assert AP.s1_sentence(wl._add_s1_state()) == (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH)


def test_a_flat_patch_is_not_flooded_by_the_first_click(wl, monkeypatch,
                                                         capsys):
    _flat_everywhere(monkeypatch)
    floods = _Floods(monkeypatch)
    got = _answers(monkeypatch)
    assert wl._add_marker_click(CELL_A)
    _fire(wl)
    kw = wl.launched[-1]
    assert kw["params"]["live"] and kw["params"]["box_rule"]
    capsys.readouterr()
    _finish(wl)
    assert floods.n == 0, "the first click flooded the whole patch"
    assert len(wl.spies.prepares) == 1
    _stepped_aside(wl, got[-1], "not run as you click (flat spots)")
    assert [tuple(m["zyx"]) for m in wl._add_click_markers] == [CELL_A]
    assert wl._add_cells_stage.get("full")
    out = capsys.readouterr().out
    assert "[ADD] cells by click: not run as you click (flat spots)" in out
    # the next click schedules nothing
    n = _runs()
    assert wl._add_marker_click(CELL_B)
    assert not wl._add_live_timer.isActive(), "a click scheduled a run"
    assert _runs() == n
    assert AP.s1_sentence(wl._add_s1_state()) == (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH)
    # a Preview fills it whole by the box rule, once, as a manual run
    wl._preview_add()
    assert _runs() == n + 1 and wl._add_busy
    kw = wl.launched[-1]
    assert kw["params"]["box_rule"] and kw["stage_key"] is not None
    assert "live" not in kw["params"]
    assert not wl._add_live_run
    _finish(wl)
    assert floods.n == 1
    res = wl._add_preview_res
    assert res.ok and res.click_how == "full run (flat spots)"
    assert res.growth == AP.GROW_FLOOD
    # a key change lets the patch run as you click again
    wl._add_volume_epoch = int(getattr(wl, "_add_volume_epoch", 0)) + 1
    wl._add_prune_caches()
    assert wl._add_cells_live_whole()


def test_a_cold_or_failed_click_fill_floods_nothing_on_a_click(
        wl, monkeypatch):
    monkeypatch.setattr(AP, "_INCREMENTAL_OFF", True)
    floods = _Floods(monkeypatch)
    got = _answers(monkeypatch)
    assert wl._add_marker_click(CELL_A)
    _fire(wl)
    _finish(wl)
    assert floods.n == 0
    _stepped_aside(wl, got[-1],
                   "not run as you click (switched off for this session)")
    assert wl._add_cells_stage.get("fallback")
    n = _runs()
    assert wl._add_marker_click(CELL_B)
    assert not wl._add_live_timer.isActive()
    assert _runs() == n
    wl._preview_add()
    _finish(wl)
    assert floods.n == 1
    res = wl._add_preview_res
    assert res.ok and res.click_how.startswith("full run (")


def test_seeds_of_one_cell_on_a_flat_patch_flood_nothing(wl, monkeypatch):
    """Markers sharing an id (the seeds of one cell) take the stage step
    by step: where it steps aside a click floods nothing either."""
    _flat_everywhere(monkeypatch)
    floods = _Floods(monkeypatch)
    got = _answers(monkeypatch)
    wl._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True},
                             {"zyx": (11, 21, 21), "id": 8, "ticked": True}]
    assert wl._add_live_schedule()
    _fire(wl)
    _finish(wl)
    assert floods.n == 0
    _stepped_aside(wl, got[-1], "not run as you click (flat spots)")
    assert len(wl._add_click_markers) == 2


def test_a_manual_preview_on_a_flat_patch_still_floods_it(wl, monkeypatch):
    """The step-aside refusal is a live run's only: a Preview pressed with
    the tick on and nothing run yet fills the patch whole by the box rule."""
    _flat_everywhere(monkeypatch)
    floods = _Floods(monkeypatch)
    wl._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    wl._preview_add()
    assert not wl._add_live_run
    _finish(wl)
    assert floods.n == 1
    res = wl._add_preview_res
    assert res.ok and res.click_how == "full run (flat spots)"
    assert "live" not in wl.launched[-1]["params"]


# --------------------------------------------------------------------------- #
# the tick coming on over a result another rule drew
# --------------------------------------------------------------------------- #
def test_the_tick_on_redoes_a_background_rule_result(wo):
    wo._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    wo._preview_add()
    _finish(wo)
    drawn = wo._add_preview_res
    assert drawn.ok and drawn.growth == AP.GROW_PATCH
    n = _runs()
    wo._on_add_live_toggled(True)
    # zeroed at once, the markers kept, one live run scheduled
    assert wo._add_preview_result is None and wo._add_preview_res is None
    assert [tuple(m["zyx"]) for m in wo._add_click_markers] == [CELL_A]
    assert wo._add_live_timer.isActive()
    _fire(wo)
    assert _runs() == n + 1 and wo._add_live_run
    kw = wo.launched[-1]
    assert kw["params"]["box_rule"] and kw["params"]["live"]
    _finish(wo)
    res = wo._add_preview_res
    assert res.ok and res.by_click and res.growth == AP.GROW_FLOOD
    assert (wo._add_preview_result == 8).any()


def test_the_tick_on_keeps_a_box_rule_result(wl):
    """A Preview with the tick on drew by the box rule: the tick off then
    on keeps it, and nothing runs."""
    wl._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    wl._preview_add()
    _finish(wl)
    drawn = wl._add_preview_res
    assert drawn.ok and drawn.growth == AP.GROW_FLOOD
    wl._on_add_live_toggled(False)
    assert wl._add_preview_res is drawn
    n = _runs()
    wl._on_add_live_toggled(True)
    assert wl._add_preview_res is drawn
    assert wl._add_preview_result is not None
    t = getattr(wl, "_add_live_timer", None)
    assert t is None or not t.isActive()
    assert _runs() == n


def test_the_tick_on_keeps_a_box_result(ci):
    """On a box the rule does not change with the tick: a Preview drawn
    with the tick off stays when it comes on."""
    ci._add_live_enabled = False
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    ci._preview_add()
    _finish(ci)
    drawn = ci._add_preview_res
    assert drawn.ok and drawn.growth == AP.GROW_FLOOD
    n = _runs()
    ci._on_add_live_toggled(True)
    assert ci._add_preview_res is drawn
    t = getattr(ci, "_add_live_timer", None)
    assert t is None or not t.isActive()
    assert _runs() == n


def test_over_the_live_whole_ceiling_a_patch_waits_for_preview(
        wl, monkeypatch):
    n = int(np.prod(wl.sync_manager.shared_labels["L"].shape))
    assert AP.ADD_LIVE_CELLS_WHOLE_MAX_VOXELS == 20_000_000
    wl._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_WHOLE_MAX_VOXELS", n)
    assert wl._add_cells_live_whole() and wl._add_live_possible()
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_WHOLE_MAX_VOXELS", n - 1)
    assert not wl._add_cells_live_whole_rule()
    assert not wl._add_live_possible()
    assert wl._add_cells_stage_key() is None
    assert AP.s1_sentence(wl._add_s1_state()) == (
        AP.ADD_S1_LIVE_OVER_GATE_PATCH)
    # a Preview then fills it with a background marker, as the tick off
    assert "box_rule" not in wl._add_cells_engine_params()


def test_a_spread_only_knob_never_prepares_again(wl):
    _click_and_draw(wl, CELL_A)
    stage = wl._add_cells_stage["stage"]
    for k, v in (("wall_cost", 7.0), ("growth_limit", 9.0),
                 ("include_membrane", False)):
        wl.adv.params[k] = v
    res = _click_and_draw(wl, CELL_B)
    assert res.click_how == "stage reused"
    assert wl._add_cells_stage["stage"] is stage
    assert len(wl.spies.prepares) == 1


def test_a_membrane_walls_preview_on_a_live_whole_patch_refuses_a_click(
        wl):
    wl._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert wl._add_live_possible()
    wl._preview_add()
    assert wl._add_busy, wl._add_transient
    assert not wl._add_marker_click(CELL_B)
    assert wl._add_transient == AP.ADD_S1_BUSY_REFUSED


def test_the_first_click_names_the_prepare_on_the_strip(wl):
    from napari_epicycle.annotator.reshape_worker import stage_text

    said = []
    real = wl._push_add_ui

    def push(**kw):
        if kw.get("measuring"):
            said.append(kw["measuring"])
        return real(**kw)

    wl._push_add_ui = push
    _click_and_draw(wl, CELL_A)
    assert stage_text("add_prepare_patch") == "Preparing the patch..."
    assert stage_text("add_prepare") == "Preparing the box..."
    assert stage_text("add_prepare_patch") in said
    assert stage_text("elevation") in said
    assert said.index(stage_text("add_prepare_patch")) < said.index(
        stage_text("elevation"))
    said.clear()
    _click_and_draw(wl, CELL_B)
    assert stage_text("add_prepare_patch") not in said


def test_the_launch_holder_is_let_go_on_every_exit(wl):
    """A Stop just after the run finished, the session's reset, the patch
    going away: the prepared patch a run took is let go each time."""
    _click_and_draw(wl, CELL_A)
    assert wl._add_marker_click(CELL_B)
    _fire(wl)
    holder = wl._add_launch_cells_holder
    assert holder and holder.get("stage") is not None
    # Stop pressed just as the worker finished: its answer is the Stop
    wl._add_stopping = True
    th = FakeThread.instances[-1]
    th.run()
    assert wl._add_launch_cells_holder is None and holder == {}
    th.end()
    wl._reset_add_state()
    wl._add_forget_clears()
    assert wl._add_launch_cells_holder is None
    assert wl._add_cells_stage is None
    # an error landing, an abandon, a dropped stage: the same
    wl._add_is_stale = lambda _token: False
    for exit_ in (lambda: wl._on_add_error("boom"),
                  wl._add_abandon_workers,
                  lambda: wl._add_drop_cells_stage("test")):
        h = {"stage": object(), "key": 1}
        wl._add_launch_cells_holder = h
        exit_()
        assert wl._add_launch_cells_holder is None and h == {}


def test_the_whole_patch_wall_picture_is_handed_to_prepare(wl):
    _click_and_draw(wl, CELL_A)
    assert wl.launched[0]["elevation_roi"] is None, "the first run builds it"
    cache = wl._add_elev_cache
    assert cache is not None and cache["key"] == wl._add_elev_key()
    # a write to the volume prepares the box again: with the kept picture
    wl._add_volume_epoch = int(getattr(wl, "_add_volume_epoch", 0)) + 1
    wl._add_prune_caches()
    assert wl._add_cells_stage is None
    _click_and_draw(wl, CELL_B)
    assert wl.launched[-1]["elevation_roi"] is cache["elev"]
    assert len(wl.spies.prepares) == 2
    assert wl.spies.prepares[-1]["elevation"] is cache["elev"]


# --------------------------------------------------------------------------- #
# a Preview while the drawn result is current runs nothing
# --------------------------------------------------------------------------- #
def test_a_preview_while_the_drawn_result_is_current_runs_nothing(wl,
                                                                  capsys):
    res = _click_and_draw(wl, CELL_A)
    assert wl._add_cells_drawn_current()
    n = _runs()
    capsys.readouterr()
    wl._preview_add()
    assert _runs() == n, "Preview ran again"
    assert "the drawn result is current" in capsys.readouterr().out
    assert wl._add_preview_res is res
    # Smallest object changes the answer: a Preview runs again
    wl.adv.params["min_size"] = 3
    assert not wl._add_cells_drawn_current()
    wl._add_live_timer.stop()
    wl._preview_add()
    assert _runs() == n + 1
    _finish(wl)
    assert wl._add_cells_drawn_current()
    assert len(wl.spies.prepares) == 1


def test_a_box_preview_while_the_drawn_result_is_current_runs_nothing(
        ci, capsys):
    res = _click_and_draw(ci, CELL_A)
    assert ci._add_cells_stage and ci._add_cells_drawn_current()
    n = _runs()
    ci._add_live_timer.stop()
    capsys.readouterr()
    ci._preview_add()
    assert _runs() == n, "Preview ran again"
    assert "the drawn result is current" in capsys.readouterr().out
    assert ci._add_preview_res is res
    # the smoothing changes the answer: a Preview runs again, on the stage
    ci.adv.params["smooth_sigma"] = 2.0
    assert not ci._add_cells_drawn_current()
    ci._preview_add()
    assert _runs() == n + 1 and ci._add_busy
    _finish(ci)
    assert ci._add_preview_res is not res
    assert ci._add_preview_res.click_how == "stage reused"
    assert ci._add_cells_drawn_current()


def test_a_box_preview_runs_when_its_stage_is_gone(ci):
    res = _click_and_draw(ci, CELL_A)
    assert ci._add_cells_drawn_current()
    # the prepared box let go (as a Clear or Close box does): the next run
    # would prepare again, so a Preview runs
    ci._add_drop_cells_stage("test")
    assert not ci._add_cells_drawn_current()
    n = _runs()
    ci._add_live_timer.stop()
    ci._preview_add()
    assert _runs() == n + 1
    _finish(ci)
    assert ci._add_preview_res is not res


def test_a_box_over_the_live_limit_preview_always_runs(ci, monkeypatch):
    res = _click_and_draw(ci, CELL_A)
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_MAX_VOXELS",
                        ci._add_roi_voxels() - 1)
    assert ci._add_cells_stage_key() is None
    assert not ci._add_cells_drawn_current()
    n = _runs()
    ci._add_live_timer.stop()
    ci._preview_add()
    assert _runs() == n + 1
    _finish(ci)
    assert ci._add_preview_res is not res


# --------------------------------------------------------------------------- #
# the slow note, and the fill's ceiling
# --------------------------------------------------------------------------- #
def test_a_large_whole_patch_with_the_tick_on_says_the_first_update(
        wl, monkeypatch):
    monkeypatch.setattr(AP, "ADD_SLOW_PATCH_VOXELS", 10)
    n = int(np.prod(wl.sync_manager.shared_labels["L"].shape))
    live = AP.ADD_NOTE_PATCH_SLOW_LIVE_CELLS.format(m=AP.millions(n))
    assert "prepares it once" in live
    assert wl._add_derive_note_text() == live
    # the tick off: the Preview's wait, as before
    wl._add_live_enabled = False
    assert wl._add_derive_note_text() == AP.patch_slow_line(
        n, "membrane", picture_kept=False)
    # the tick on but walls that wait for Preview
    wl._add_live_enabled = True
    wl.adv.wall = dict(wl.adv.wall, wall_kind="intensity")
    assert wl._add_derive_note_text() != live


def test_over_the_fill_ceiling_a_live_click_runs_nothing(wl, monkeypatch):
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", 1000)
    n = int(np.prod(wl.sync_manager.shared_labels["L"].shape))
    wl._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert AP.s1_sentence(wl._add_s1_state()) == AP.whole_too_big_line(n)
    kw, why = wl._add_gather_preview(live=True)
    assert kw is None and why == AP.whole_too_big_line(n)


# --------------------------------------------------------------------------- #
# leaving the live whole patch lets the stage go
# --------------------------------------------------------------------------- #
def test_the_tick_off_or_other_walls_drop_the_stage(wl):
    _click_and_draw(wl, CELL_A)
    assert wl._add_cells_stage is not None
    wl.adv.wall = dict(wl.adv.wall, wall_kind="intensity")
    wl._add_prune_caches()
    assert wl._add_cells_stage is None
    wl.adv.wall = dict(wl.adv.wall, wall_kind="membrane")
    _click_and_draw(wl, CELL_B)
    assert wl._add_cells_stage is not None
    wl._on_add_live_toggled(False)
    assert wl._add_cells_stage is None

"""Update as I click: Cells ▸ Click markers, on a box.

A Cells box at or under ``ADD_LIVE_CELLS_MAX_VOXELS`` runs as you click,
with the Nuclei rules: one run in flight, a change meanwhile marks it dirty
and exactly one more run follows -- never a cancel (a watershed that has
started cannot stop).  The box's wall picture is kept per box and wall
settings, so every click after the first skips its build.  Over the limit,
and on a whole patch, the balls still appear but the cells wait for
Preview, and S1 says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.annotator import reshape_worker as RW
from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread
from .add_widget_fixture import qapp, widget  # noqa: F401
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_nuclei_box import LiveAdv, _FakeTimer

CLICKS = (AP.SOURCE_CLICKS, None)
#: two cells inside the click fixture's box (2:18, 2:28, 2:28), clear of
#: its kept object 7 at [5:9, 5:9, 5:9]
CELL_A = (11, 20, 20)
CELL_B = (11, 12, 22)
BOX = (slice(2, 18), slice(2, 28), slice(2, 28))


def _membrane(shape):
    """Bright sheets every 8 voxels in y and x: closed walls the cells
    fill up to."""
    img = np.zeros(shape, np.float32)
    img[:, ::8, :] = 1.0
    img[:, :, ::8] = 1.0
    return img


def _cells(hh, *, live):
    from napari.layers import Image

    adv = LiveAdv()
    adv.preset = "cells"
    adv.source = CLICKS
    adv.params["dark_outside"] = False
    adv.params["click_radius_um"] = 1.0
    adv.wall = dict(adv.wall, channel="mem")
    hh.adv = adv
    vol = hh.sync_manager.shared_labels["L"]
    hh.viewer.layers.append(Image(_membrane(vol.shape), name="mem"))
    hh._add_info = AP.AddBoxInfo(empty_share=1.0)
    hh._add_live_enabled = bool(live)
    return hh


@pytest.fixture
def lc(h, monkeypatch):  # noqa: F811
    import qtpy.QtCore

    _FakeTimer.made = []
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    return _cells(h, live=True)


def _timer(hh):
    return getattr(hh, "_add_live_timer", None)


def _active(hh):
    t = _timer(hh)
    return t is not None and t.isActive()


def _fire(hh):
    assert _active(hh), "no live run is scheduled"
    _timer(hh).fire()


def _finish(hh):
    th = FakeThread.instances[-1]
    th.run()
    th.end()


def _click_and_draw(hh, at):
    assert hh._add_marker_click(at)
    _fire(hh)
    _finish(hh)
    assert hh._add_preview_result is not None, hh._add_transient


def _id_at(hh, at):
    return next(int(m["id"]) for m in hh._add_click_markers
                if tuple(m["zyx"]) == tuple(at))


def _spy_launch(hh):
    """Every worker launch: (kwargs, live)."""
    seen = []
    real = hh._launch_add_worker

    def spy(kwargs, live=False):
        seen.append((kwargs, bool(live)))
        return real(kwargs, live=live)

    hh._launch_add_worker = spy
    return seen


def _box_key(hh):
    return hh._add_elev_key() + (
        tuple((sl.start, sl.stop) for sl in hh._add_roi),)


# --------------------------------------------------------------------------- #
# the constants
# --------------------------------------------------------------------------- #
def test_the_cells_gate_and_its_sentence():
    assert AP.ADD_LIVE_CELLS_MAX_VOXELS == 1_000_000
    assert AP.ADD_S1_LIVE_OVER_GATE_BOX == (
        "Press Preview to see the cells — this box is over {n} voxels.")
    # the limit in millions, as the other Cells size lines say it
    assert AP.voxel_limit_words(1_000_000) == "1 million"
    assert AP.voxel_limit_words(4_000_000) == "4 million"
    assert AP.voxel_limit_words(1_500_000) == "1.5 million"
    assert AP.voxel_limit_words(1000) == "1,000"
    assert AP.ADD_S1_LIVE_OVER_GATE_BOX.format(
        n=AP.voxel_limit_words(AP.ADD_LIVE_CELLS_MAX_VOXELS)) == (
        "Press Preview to see the cells — this box is over 1 million "
        "voxels.")


# --------------------------------------------------------------------------- #
# the tick on the page
# --------------------------------------------------------------------------- #
def test_the_tick_shows_on_cells_click_markers_only(widget):  # noqa: F811
    from .add_widget_fixture import reset_add

    reset_add(widget)
    try:
        widget.update_add_ui_state(box_active=True)
        widget.set_add_preset("cells")
        widget.update_add_ui_state(marker_sources=[
            (AP.ADD_ITEM_RUN.format(name="r2"), True, ("labels", "r2")),
            (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
            (AP.ADD_ITEM_CLICKS, True, ("paint", None))])
        skeys = list(widget._add_source_keys)
        widget.add_marker_source_combo.setCurrentIndex(
            skeys.index(("paint", None)))
        assert not widget._add_click_rows.isHidden()
        assert not widget.add_live_cb.isHidden()
        # another run's objects, or nuclei masks: nothing runs as you click
        widget.add_marker_source_combo.setCurrentIndex(
            skeys.index(("labels", "r2")))
        assert widget.add_live_cb.isHidden()
        widget.add_marker_source_combo.setCurrentIndex(
            skeys.index(("nuclei_masks", None)))
        assert widget.add_live_cb.isHidden()
    finally:
        widget.set_add_preset("nuclei")
        widget.add_look_for_combo.setCurrentIndex(0)
        widget.update_add_ui_state(box_active=False)
        reset_add(widget)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_live_is_possible_on_a_small_cells_box(lc):
    lc._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert lc._add_live_possible()


def test_live_is_not_possible_over_the_gate(lc, monkeypatch):
    lc._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    n = int(np.prod([sl.stop - sl.start for sl in lc._add_roi]))
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_MAX_VOXELS", n)
    assert lc._add_live_possible(), "at the gate: live"
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_MAX_VOXELS", n - 1)
    assert not lc._add_live_possible()


def test_live_is_not_possible_on_a_whole_patch_or_another_source(lc):
    lc._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    lc._add_whole_patch = True
    # a whole patch runs as you click only with walls the click fill
    # redoes (Membrane sheets, Intensity edges)
    assert lc._add_live_possible()
    lc.adv.wall = dict(lc.adv.wall, wall_kind="intensity")
    assert not lc._add_live_possible()
    lc.adv.wall = dict(lc.adv.wall, wall_kind="membrane")
    lc._add_whole_patch = False
    for src in (("labels", "r2"), ("nuclei_masks", None)):
        lc.adv.source = src
        assert not lc._add_live_possible()
    lc.adv.source = CLICKS
    lc._add_click_markers[0]["ticked"] = False
    assert not lc._add_live_possible(), "no ticked marker"


def test_over_the_gate_a_click_runs_nothing(lc, monkeypatch):
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_MAX_VOXELS", 1000)
    assert lc._add_marker_click(CELL_A)
    assert not _active(lc)
    assert FakeThread.instances == []


# --------------------------------------------------------------------------- #
# one click, one run, never busy
# --------------------------------------------------------------------------- #
def test_one_click_makes_one_live_run_and_never_greys(lc):
    pushes0 = len(lc.adv.pushes)
    assert lc._add_marker_click(CELL_A)
    assert _active(lc)
    assert FakeThread.instances == []
    _fire(lc)
    assert len(FakeThread.instances) == 1
    assert lc._add_busy is False
    _finish(lc)
    assert lc._add_busy is False
    assert lc._add_preview_result is not None, lc._add_transient
    assert (lc._add_preview_result == _id_at(lc, CELL_A)).any()
    assert len(FakeThread.instances) == 1
    assert not any(kw.get("busy") is True
                   for kw in lc.adv.pushes[pushes0:])


def test_a_click_mid_run_is_one_relaunch_and_never_a_cancel(lc, monkeypatch):
    cancels = []
    monkeypatch.setattr(RW.RefineWorker, "cancel",
                        lambda self: cancels.append(self), raising=False)
    assert lc._add_marker_click(CELL_A)
    _fire(lc)
    th = FakeThread.instances[-1]
    ev = lc._add_worker._cancel_event
    assert lc._add_marker_click(CELL_B)        # not refused: not busy
    assert lc._add_live_dirty
    assert not _active(lc)
    assert len(FakeThread.instances) == 1
    th.run()
    assert lc._add_preview_result is None      # launched on A only: stale
    th.end()
    assert len(FakeThread.instances) == 2      # exactly one relaunch
    _finish(lc)
    assert len(FakeThread.instances) == 2
    res = lc._add_preview_result
    assert res is not None
    assert (res == _id_at(lc, CELL_A)).any()
    assert (res == _id_at(lc, CELL_B)).any()
    assert cancels == []
    assert not ev.is_set()


# --------------------------------------------------------------------------- #
# parity with a manual Preview
# --------------------------------------------------------------------------- #
def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (a is b) or (isinstance(a, np.ndarray)
                            and isinstance(b, np.ndarray)
                            and np.array_equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def test_a_live_run_gathers_what_a_manual_preview_gathers(lc):
    seen = _spy_launch(lc)
    _click_and_draw(lc, CELL_A)
    assert lc._add_marker_click(CELL_B)
    _fire(lc)
    live_kw, live = seen[-1]
    assert live is True
    _finish(lc)
    manual_kw, why = lc._add_gather_preview(live=False)
    assert why is None, why
    live_kw = {k: v for k, v in live_kw.items() if k != "cancel_event"}
    assert live_kw.keys() == manual_kw.keys()
    bad = [k for k in live_kw if not _same(live_kw[k], manual_kw[k])]
    assert bad == []


def test_the_live_result_equals_a_manual_preview(lc):
    _click_and_draw(lc, CELL_A)
    assert lc._add_marker_click(CELL_B)
    _fire(lc)
    _finish(lc)
    live = np.array(lc._add_preview_result, copy=True)
    lc._preview_add()
    assert lc._add_busy                          # the manual way: busy
    _finish(lc)
    assert np.array_equal(live, lc._add_preview_result)


# --------------------------------------------------------------------------- #
# the box's wall picture, kept per box and wall settings
# --------------------------------------------------------------------------- #
def test_the_first_run_keeps_the_box_wall_picture_the_next_reuses_it(lc):
    seen = _spy_launch(lc)
    _click_and_draw(lc, CELL_A)
    assert seen[0][0]["elevation_roi"] is None, "the first run builds it"
    cache = lc._add_box_elev_cache
    assert cache is not None and cache["key"] == _box_key(lc)
    assert cache["elev"] is lc._add_preview_res.elevation
    assert cache["elev"].shape == lc._add_preview_result.shape
    assert getattr(lc, "_add_elev_cache", None) is None, (
        "the whole patch's slot is apart")
    assert lc._add_marker_click(CELL_B)
    _fire(lc)
    assert seen[-1][0]["elevation_roi"] is cache["elev"]
    _finish(lc)
    assert lc._add_box_elev_cache["elev"] is cache["elev"]


def test_another_wall_preparation_builds_it_again(lc):
    seen = _spy_launch(lc)
    _click_and_draw(lc, CELL_A)
    lc.adv.wall = dict(lc.adv.wall, chain="zcorrect")
    kw, why = lc._add_gather_preview()
    assert why is None, why
    assert kw["elevation_roi"] is None
    lc._add_prune_caches()
    assert lc._add_box_elev_cache is None
    assert seen


def _land_box_info(hh):
    """The box-info worker measures the box and lands its answer."""
    hh._request_add_box_info()
    th = hh._add_info_thread
    assert th is not None
    th.run()
    th.end()
    assert hh._add_info is not None


def test_a_box_move_drops_the_box_wall_picture(lc):
    _click_and_draw(lc, CELL_A)
    assert lc._add_box_elev_cache is not None
    old = lc._add_box_elev_cache["key"]
    # a box move: the box changes and its measurement is asked again
    lc.place_box((slice(2, 18), slice(2, 27), slice(2, 28)))
    lc._add_drop_info()
    assert _box_key(lc) != old
    kw, why = lc._add_gather_preview()
    assert kw is None and why, "the box is being measured"
    _land_box_info(lc)
    kw, why = lc._add_gather_preview()
    assert why is None, why
    assert kw["elevation_roi"] is None, "the moved box builds its own"
    lc._add_refresh_ui()                         # the refresh prunes
    assert lc._add_box_elev_cache is None


def test_a_fresh_build_equals_the_cached_picture_result(lc):
    seen = _spy_launch(lc)
    _click_and_draw(lc, CELL_A)
    cache = lc._add_box_elev_cache
    assert cache is not None
    # the second click's run reuses the kept picture
    assert lc._add_marker_click(CELL_B)
    _fire(lc)
    assert seen[-1][0]["elevation_roi"] is cache["elev"]
    _finish(lc)
    cached = np.array(lc._add_preview_result, copy=True)
    # the picture forgotten: a manual Preview builds it again
    lc._add_box_elev_cache = None
    lc._preview_add()
    assert seen[-1][0]["elevation_roi"] is None
    _finish(lc)
    assert np.array_equal(cached, lc._add_preview_result)
    again = lc._add_box_elev_cache
    assert again is not None and again["key"] == _box_key(lc)
    assert np.array_equal(again["elev"], cache["elev"])


def test_the_preset_leaving_cells_drops_the_box_wall_picture(lc):
    _click_and_draw(lc, CELL_A)
    assert lc._add_box_elev_cache is not None
    lc.adv.preset = "nuclei"
    lc._add_prune_caches()
    assert lc._add_box_elev_cache is None


def test_the_whole_patch_slot_is_never_touched_by_a_box(lc):
    sentinel = np.zeros(3, np.float32)
    lc._add_elev_cache = {"key": lc._add_elev_key(), "elev": sentinel}
    _click_and_draw(lc, CELL_A)
    assert lc._add_elev_cache["elev"] is sentinel
    lc._add_prune_caches()
    assert lc._add_elev_cache["elev"] is sentinel
    assert lc._add_box_elev_cache is not None


def test_over_the_gate_a_manual_preview_keeps_no_box_wall_picture(
        lc, monkeypatch):
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_MAX_VOXELS", 1000)
    lc._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    lc._preview_add()
    _finish(lc)
    assert lc._add_preview_result is not None, lc._add_transient
    assert getattr(lc, "_add_box_elev_cache", None) is None


def test_the_box_wall_picture_is_forgotten_with_the_clears_and_reset(lc):
    _click_and_draw(lc, CELL_A)
    assert lc._add_box_elev_cache is not None
    lc._add_forget_clears()
    assert lc._add_box_elev_cache is None
    _click_and_draw(lc, CELL_B)
    assert lc._add_box_elev_cache is not None
    lc._reset_add_state()
    assert lc._add_box_elev_cache is None


# --------------------------------------------------------------------------- #
# a stale result never applies; its wall picture is kept when the walls are
# unchanged
# --------------------------------------------------------------------------- #
def test_a_stale_cells_result_by_settings_keeps_its_walls(lc):
    seen = _spy_launch(lc)
    assert lc._add_marker_click(CELL_A)
    _fire(lc)
    th = FakeThread.instances[-1]
    # a setting changed while it ran (not the walls)
    lc._invalidate_add_preview()
    assert lc._add_live_dirty
    th.run()
    assert lc._add_preview_result is None       # stale: not applied
    cache = lc._add_box_elev_cache
    assert cache is not None and cache["key"] == _box_key(lc)
    th.end()
    assert len(FakeThread.instances) == 2       # the one run owed
    assert seen[-1][0]["elevation_roi"] is cache["elev"]
    _finish(lc)
    assert lc._add_preview_result is not None
    assert lc._add_box_elev_cache["elev"] is cache["elev"]


def test_a_click_during_the_first_run_keeps_its_walls_for_the_relaunch(lc):
    seen = _spy_launch(lc)
    assert lc._add_marker_click(CELL_A)
    _fire(lc)
    assert seen[-1][0]["elevation_roi"] is None   # the first run builds
    th = FakeThread.instances[-1]
    assert lc._add_marker_click(CELL_B)           # the markers changed
    assert lc._add_live_dirty
    th.run()
    assert lc._add_preview_result is None         # stale: not applied
    cache = lc._add_box_elev_cache
    assert cache is not None and cache["key"] == _box_key(lc)
    th.end()
    assert len(FakeThread.instances) == 2
    assert seen[-1][0]["elevation_roi"] is cache["elev"]
    _finish(lc)
    res = lc._add_preview_result
    assert res is not None
    assert (res == _id_at(lc, CELL_A)).any()
    assert (res == _id_at(lc, CELL_B)).any()


def test_a_stale_cells_result_by_the_walls_neither_applies_nor_keeps(lc):
    seen = _spy_launch(lc)
    assert lc._add_marker_click(CELL_A)
    _fire(lc)
    th = FakeThread.instances[-1]
    # the walls' preparation changed while it ran: its picture is no
    # longer this box's
    lc.adv.wall = dict(lc.adv.wall, chain="zcorrect")
    lc._invalidate_add_preview()
    assert lc._add_live_dirty
    th.run()
    assert lc._add_preview_result is None
    assert getattr(lc, "_add_box_elev_cache", None) is None
    th.end()
    assert len(FakeThread.instances) == 2       # the one run owed
    assert seen[-1][0]["elevation_roi"] is None  # built again
    _finish(lc)
    assert lc._add_preview_result is not None
    assert lc._add_box_elev_cache["key"] == _box_key(lc)


# --------------------------------------------------------------------------- #
# the busy refusal of a manual Cells run
# --------------------------------------------------------------------------- #
def test_a_manual_box_run_still_refuses_a_click_busy(lc):
    lc._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    lc._preview_add()
    assert lc._add_busy
    assert not lc._add_marker_click(CELL_B)
    assert lc._add_transient == AP.ADD_S1_BUSY_REFUSED


def test_a_whole_patch_preview_refuses_a_click_busy(lc):
    # walls a whole patch is not filled as you click with
    lc.adv.wall = dict(lc.adv.wall, wall_kind="intensity")
    vol = lc.sync_manager.shared_labels["L"]
    lc._add_whole_patch = True
    lc.place_box(tuple(slice(0, s) for s in vol.shape))
    lc.ctx["spacing_source"] = "qc_settings"
    lc._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert not lc._add_live_possible()
    lc._preview_add()
    assert lc._add_busy, lc._add_transient
    assert not lc._add_marker_click(CELL_B)
    assert lc._add_transient == AP.ADD_S1_BUSY_REFUSED
    assert not _active(lc)


# --------------------------------------------------------------------------- #
# S1 row 13.5
# --------------------------------------------------------------------------- #
def _state(**kw):
    s = {"has_patch": True, "has_box": True, "preset": "cells",
         "source": AP.SOURCE_CLICKS, "wall_channel": "mem",
         "box_shape": (100, 100, 101), "whole_patch": False, "n_clicks": 1,
         "n_ticked": 1, "place_markers_on": True, "live_on": True,
         "empty_share": 1.0}
    s.update(kw)
    return s


def test_row_13_5_over_the_gate_and_on_a_whole_patch():
    over = AP.ADD_S1_LIVE_OVER_GATE_BOX.format(n="1 million")
    assert AP.s1_sentence(_state()) == over
    assert AP.preview_block(_state()) is None   # Preview is the way
    assert AP.s1_kind(_state()) == "ready"
    whole = _state(whole_patch=True)
    assert AP.s1_sentence(whole) == AP.ADD_S1_LIVE_OVER_GATE_PATCH
    assert AP.s1_kind(whole) == "ready"
    gate = (AP.ADD_S1_LIVE_OVER_GATE_PATCH, over)
    # at the gate, the tick off, another source: the ready line
    for other in (_state(box_shape=(100, 100, 100)), _state(live_on=False),
                  _state(source="labels", n_run_markers=3,
                         run_id="r2")):
        assert AP.s1_sentence(other) not in gate
        assert "not updated as you click" not in AP.s1_sentence(other)
    # the rows above it win
    assert AP.s1_sentence(_state(transient="x")) == "x"
    assert AP.s1_sentence(_state(busy=True)) == AP.ADD_S1_WORKING
    assert AP.s1_sentence(_state(n_clicks=0)) == AP.ADD_S1_NO_MARKERS_ON


def test_row_13_5_on_the_harness(lc, monkeypatch):
    lc._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    s = lc._add_s1_state()
    assert AP.s1_sentence(s) not in (AP.ADD_S1_LIVE_OVER_GATE_PATCH,)
    assert not AP.s1_sentence(s).startswith("Press Preview")
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_MAX_VOXELS", 1000)
    assert AP.s1_sentence(lc._add_s1_state()) == (
        AP.ADD_S1_LIVE_OVER_GATE_BOX.format(n="1,000"))


# --------------------------------------------------------------------------- #
# the box measured again: the click waiting on it runs
# --------------------------------------------------------------------------- #
def test_a_box_measurement_landing_runs_the_click_that_waited(lc):
    assert lc._add_marker_click(CELL_A)
    # the box is being measured (a box move, a channel)
    lc._add_drop_info()
    _fire(lc)
    assert FakeThread.instances == [] or lc._add_thread is None
    assert lc._add_preview_result is None
    assert not _active(lc)
    lc._request_add_box_info()
    th = lc._add_info_thread
    assert th is not None
    th.run()
    th.end()
    assert lc._add_info is not None
    assert _active(lc), "the click that waited on the box runs now"
    _fire(lc)
    _finish(lc)
    assert lc._add_preview_result is not None, lc._add_transient


# --------------------------------------------------------------------------- #
# fix round: the Accept-cancelled sentence outlives the box measurement
# --------------------------------------------------------------------------- #
def test_a_preset_switch_keeps_the_accept_cancelled_sentence(lc):
    lc._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    # an Accept waiting on the other preset's run
    lc._add_accept_pending = True
    lc.adv.preset = "cells"
    lc._on_add_preset_changed("cells")
    assert not lc._add_accept_pending
    assert lc._add_transient == AP.ADD_S1_ACCEPT_WAIT_OFF
    assert lc._add_info is None                  # the box is measured again
    # the quiet ends: the live run is refused while the box is measured
    _fire(lc)
    assert FakeThread.instances == []
    assert lc._add_transient == AP.ADD_S1_ACCEPT_WAIT_OFF
    assert AP.s1_sentence(lc._add_s1_state()) == AP.ADD_S1_ACCEPT_WAIT_OFF


# --------------------------------------------------------------------------- #
# fix round: the growth engine's warm-up on a live run
# --------------------------------------------------------------------------- #
def test_a_live_run_says_the_warm_up_in_s1(lc):
    from napari_epicycle.annotator.reshape_worker import stage_text

    assert lc._add_marker_click(CELL_A)
    _fire(lc)
    assert lc._add_live_run and not lc._add_busy
    lc._on_add_progress("add_warm", 0, 0)
    assert lc._add_warm_text == AP.ADD_S1_WARMING
    assert AP.s1_sentence(lc._add_s1_state()) == AP.ADD_S1_WARMING
    assert lc.adv.last("status_text") == AP.ADD_S1_WARMING
    # the strip's line still names the stage
    assert lc.adv.last("measuring") == stage_text("add_warm")
    # the next stage: the warm-up is over
    lc._on_add_progress("add_box", 1, 4)
    assert lc._add_warm_text is None
    assert AP.s1_sentence(lc._add_s1_state()) != AP.ADD_S1_WARMING
    assert lc.adv.last("measuring") == stage_text("add_box")
    _finish(lc)
    assert lc._add_preview_result is not None, lc._add_transient

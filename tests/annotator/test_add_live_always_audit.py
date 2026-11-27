"""Always update as I click: the edges an independent read of the tick found.

* Stop stays live through a slow update's progress lines (the page's strip
  greyed it on every line that named a stage);
* a change during a slow update cancels the run it makes stale, and one
  more update follows;
* a slow update's result answers a Preview with the same inputs;
* the tick no longer changes the click fill's prepared-box key: only a
  stage it made meaningless goes when it flips;
* the click fill still being compiled is a wait with the tick on too;
* on a whole patch under Spread one amber line says what open background
  does.
"""

from __future__ import annotations

import types

import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_cells_box import CELL_A, CELL_B, _fire, _finish, _id_at
from .test_add_live_cells_incremental import ci  # noqa: F401
from .test_add_live_preparing import _cold


def _always(hh, on=True):
    hh._on_add_live_always_toggled(bool(on))


def _runs():
    return len(FakeThread.instances)


def _slow_box(hh, monkeypatch):
    """A box the quick fill cannot take (ridge walls over the full-run
    size), the tick on."""
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    hh.adv.wall = dict(hh.adv.wall, wall_kind="frangi")
    _always(hh)
    assert hh._add_live_slow()


# --------------------------------------------------------------------------- #
# Stop through the progress lines
# --------------------------------------------------------------------------- #
_STRIP = ("measuring", "stop_enabled", "busy", "stage_text")


def test_stop_stays_live_through_a_slow_update_on_the_real_page(
        ci, monkeypatch, widget):  # noqa: F811
    _slow_box(ci, monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    assert ci._add_live_slow_in_flight()
    n = len(ci.adv.pushes)
    for stage in ("add_box", "elevation", "add_cut"):
        ci._on_add_progress(stage)
    pushes = [p for p in ci.adv.pushes[n:] if "measuring" in p]
    assert pushes, "the progress lines reach the page"
    assert all(p.get("stop_enabled") is True for p in pushes), pushes
    # the same pushes on the real page: Stop is never greyed by them
    w = widget
    reset_add(w)
    try:
        w.update_add_ui_state(busy=False, stop_enabled=True)
        assert w.add_stop_btn.isEnabled()
        for p in pushes:
            w.update_add_ui_state(**{k: v for k, v in p.items()
                                     if k in _STRIP})
            assert w.add_stop_btn.isEnabled(), p
        # (a line with no word on Stop greys it: the page's own rule)
        w.update_add_ui_state(measuring="Reading the box…")
        assert not w.add_stop_btn.isEnabled()
    finally:
        w.update_add_ui_state(busy=False, measuring="", stop_enabled=False)
        reset_add(w)


def test_a_quick_live_run_s_progress_says_nothing_of_stop(ci):
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    assert ci._add_live_run and not ci._add_live_slow_in_flight()
    n = len(ci.adv.pushes)
    ci._on_add_progress("add_box")
    pushes = [p for p in ci.adv.pushes[n:] if "measuring" in p]
    assert pushes and all("stop_enabled" not in p for p in pushes)


# --------------------------------------------------------------------------- #
# a change during a slow update
# --------------------------------------------------------------------------- #
def test_a_click_during_a_slow_update_cancels_it(ci, monkeypatch, capsys):
    _slow_box(ci, monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    worker = ci._add_worker
    assert not worker.cancel_event.is_set()
    n = _runs()
    capsys.readouterr()
    assert ci._add_marker_click(CELL_B)
    assert worker.cancel_event.is_set(), "the stale run is cancelled"
    assert ci._add_live_dirty
    assert "slow update cancelled" in capsys.readouterr().out
    # its end draws nothing and says nothing went wrong; one more update
    _finish(ci)
    assert ci._add_transient not in (AP.ADD_S1_STOPPED,
                                     AP.ADD_S1_STOPPED_PATCH)
    assert _runs() == n + 1, "the queued update ran"
    _finish(ci)
    assert (ci._add_preview_result == _id_at(ci, CELL_A)).any()
    assert (ci._add_preview_result == _id_at(ci, CELL_B)).any()


def test_a_click_after_the_walls_keeps_them_for_the_run_owed(
        ci, monkeypatch):
    """A click lands after the cancelled slow update built its wall picture
    (ridge walls: the slow part): the run owed is handed that picture, not
    asked to build it again."""
    from packages.epicycle_annotator.reshape import region as R

    _slow_box(ci, monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    worker = ci._add_worker
    real = R.reshape_empty_region
    handed = []

    def wrapped(*a, progress=None, elevation=None, **k):
        handed.append(elevation)
        first = len(handed) == 1

        def spy(stage, i=0, n=0):
            if first and stage == "add_markers":
                # the walls are built: the user clicks again
                assert ci._add_marker_click(CELL_B)
            if progress is not None:
                progress(stage, i, n)
        return real(*a, progress=spy, elevation=elevation, **k)

    monkeypatch.setattr(R, "reshape_empty_region", wrapped)
    n = _runs()
    _finish(ci)
    assert worker.cancel_event.is_set(), "the click cancelled the run"
    assert _runs() == n + 1, "the update owed was queued"
    assert ci._add_box_elev_cache is not None, "the walls were kept"
    kept = ci._add_box_elev_cache["elev"]
    _finish(ci)
    assert handed[0] is None
    assert handed[-1] is kept, "the run owed got the walls"
    assert (ci._add_preview_result == _id_at(ci, CELL_B)).any()


def test_the_tick_going_off_during_a_slow_update_cancels_it(ci, monkeypatch):
    """"Always update as I click" turned off while the slow update runs:
    nothing would draw its answer, so it is cancelled at once, and no
    other run follows (the region waits for Preview)."""
    _slow_box(ci, monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    worker = ci._add_worker
    assert ci._add_live_slow_in_flight()
    n = _runs()
    _always(ci, False)
    assert worker.cancel_event.is_set(), "the slow update is cancelled"
    _finish(ci)
    assert _runs() == n, "no run follows the tick going off"
    assert ci._add_preview_result is None
    assert not ci._add_busy


def test_a_click_during_a_quick_live_run_does_not_cancel_it(ci):
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    worker = ci._add_worker
    assert ci._add_marker_click(CELL_B)
    assert not worker.cancel_event.is_set()


# --------------------------------------------------------------------------- #
# Preview after a slow update
# --------------------------------------------------------------------------- #
def test_preview_after_a_slow_update_runs_nothing(ci, monkeypatch, capsys):
    _slow_box(ci, monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    _finish(ci)
    assert ci._add_preview_result is not None
    n = _runs()
    capsys.readouterr()
    ci._preview_add()
    assert _runs() == n, "the slow update already answered these inputs"
    assert "current" in capsys.readouterr().out
    # another click: the inputs moved, a Preview runs again
    assert ci._add_marker_click(CELL_B)
    ci._add_live_timer.stop()
    ci._preview_add()
    assert _runs() == n + 1


def test_preview_during_a_slow_update_is_answered_by_it(ci, monkeypatch):
    _slow_box(ci, monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    n = _runs()
    ci._preview_add()
    assert ci._add_manual_owed
    _finish(ci)
    assert _runs() == n, "the identical fill does not run twice"
    assert ci._add_preview_result is not None


# --------------------------------------------------------------------------- #
# the tick and the prepared box
# --------------------------------------------------------------------------- #
def test_the_stage_key_does_not_carry_the_tick(ci):
    key = ci._add_cells_stage_key()
    assert key is not None
    _always(ci)
    assert ci._add_cells_stage_key() == key
    assert "always" not in key


@pytest.mark.parametrize("kept,on,goes", [
    # a good stage survives either flip
    (dict(stage=types.SimpleNamespace(flat_override=None)), True, False),
    (dict(stage=types.SimpleNamespace(flat_override=None)), False, False),
    # a stage kept despite flat spots means nothing with the tick off
    (dict(stage=types.SimpleNamespace(flat_override="flat")), False, True),
    # a full run the flat spots chose means nothing with the tick on
    (dict(stage=None, full=True), True, True),
])
def test_only_a_stage_the_flip_made_meaningless_goes(ci, kept, on, goes):
    if not on:
        _always(ci)
    key = ci._add_cells_stage_key()
    ci._add_cells_stage = dict(kept, key=key)
    _always(ci, on)
    assert (ci._add_cells_stage is None) is goes


def test_the_flip_makes_a_run_in_flight_stale(ci):
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    before = int(getattr(ci, "_add_settings_epoch", 0))
    _always(ci)
    assert ci._add_settings_epoch == before + 1
    assert ci._add_live_dirty, "one more run follows under the new rule"


# --------------------------------------------------------------------------- #
# the compile wait with the tick on
# --------------------------------------------------------------------------- #
def test_a_compiling_fill_is_a_wait_with_the_tick_on(ci, monkeypatch):
    from .test_add_live_cells_whole import _over_the_full_run_size

    _over_the_full_run_size(monkeypatch)
    _cold(monkeypatch)
    _always(ci)
    assert ci._add_cells_fill_pending()
    assert not ci._add_live_slow(), "no slow fill during the compile"
    assert ci._add_marker_click(CELL_A)
    assert AP.s1_sentence(ci._add_s1_state()) == AP.ADD_S1_LIVE_PREPARING
    assert ci._add_live_timer.starts[-1] == AP.ADD_LIVE_DEBOUNCE_MS
    _fire(ci)
    assert ci.launched[-1][0]["params"].get("live") is True
    # the line under the tick stays hidden with the tick on
    assert AP.live_tick_note(ci._add_s1_state()) == ""


# --------------------------------------------------------------------------- #
# one amber line under Spread on a whole patch
# --------------------------------------------------------------------------- #
def test_spread_on_a_whole_patch_says_one_thing_about_open_background(
        widget):  # noqa: F811
    w = widget
    reset_add(w)
    try:
        w.set_add_preset("cells")
        w.update_add_ui_state(box_active=True, whole_patch=True,
                              derive_note="")
        w.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        w.add_dark_outside_cb.setChecked(False)
        w.add_grow_combo.setCurrentIndex(AP.GROW_KEYS.index(AP.GROW_SPREAD))
        w._add_advanced_toggle.setChecked(True)
        assert not w.add_dark_off_note.isHidden()
        assert w.add_dark_off_note.text() == AP.ADD_NOTE_SPREAD_OPEN
        assert AP.ADD_NOTE_SPREAD_OPEN not in w._add_advanced_help.text()
        # Fill: the dark-off leak note, as before
        w.add_grow_combo.setCurrentIndex(AP.GROW_KEYS.index(AP.GROW_FLOOD))
        assert w.add_dark_off_note.text() == AP.ADD_NOTE_DARK_OFF_LEAK_WHOLE
        # a box under Spread keeps both (the box's leak is to its faces)
        w.update_add_ui_state(whole_patch=False)
        w.add_grow_combo.setCurrentIndex(AP.GROW_KEYS.index(AP.GROW_SPREAD))
        assert w.add_dark_off_note.text() == AP.ADD_NOTE_DARK_OFF_LEAK
        assert AP.ADD_NOTE_SPREAD_OPEN in w._add_advanced_help.text()
    finally:
        w._add_advanced_toggle.setChecked(False)
        w.add_grow_combo.setCurrentIndex(0)
        w.add_dark_outside_cb.setChecked(True)
        w.set_add_preset("nuclei")
        w.update_add_ui_state(box_active=False, whole_patch=False)
        reset_add(w)


# --------------------------------------------------------------------------- #
# the words
# --------------------------------------------------------------------------- #
def test_the_whole_fill_line_with_always_update_on():
    assert AP.whole_fill_help(True, False, always=True) == (
        AP.ADD_HELP_WHOLE_FILL_ALWAYS)
    assert "Always update as I click" in AP.ADD_HELP_WHOLE_FILL_ALWAYS
    # the tick off: as before
    assert AP.whole_fill_help(True, False) == AP.ADD_HELP_WHOLE_FILL_PREVIEW
    assert AP.whole_fill_help(True, True, always=True) == (
        AP.ADD_HELP_WHOLE_FILL_LIVE)
    assert AP.whole_fill_help(False, None, spread=True, always=True) == (
        AP.ADD_HELP_WHOLE_FILL_SPREAD)


def test_the_one_channel_note_on_a_whole_patch_names_controls_there():
    """Grow by and Wall cost are on a whole patch now: the note names them
    there, as on a box, and no longer sends the user to a box."""
    for spread in (False, True):
        for shrink in (False, True):
            whole = AP.one_channel_note(spread=spread, shrink=shrink,
                                        whole=True)
            assert whole == AP.one_channel_note(spread=spread,
                                                shrink=shrink)
            assert "S, M or L" not in whole
            assert "Wall cost" in whole
    assert not hasattr(AP, "ADD_AMBER_ONE_CHANNEL_WHOLE")

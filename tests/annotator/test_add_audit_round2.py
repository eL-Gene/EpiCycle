"""The second adversarial round on the live gates (four reviewers, one
lens each): a slow update is cut short by ANY change that switches live off
(the Update tick, a preset or source change), not only by the Always tick;
a marker placed while Stop winds down is drawn once Stopped is said; the
labels' write epoch moves when the main Undo takes back an Accept or a
Clear (so the kept nuclear seeds and the prepared patch go); the line under
the tick stands down where Preview is grey for the size or for numba; the
over-limit line names the tick that lifts it; the finder's wait figure is
told apart from the whole Anywhere Preview's; a box slow note follows Grow
by; the Always help says minutes; the spacing note shows on a Nuclei whole
patch with a Tissue outline; a parked preview thread lets go of its inputs
when it ends."""
from __future__ import annotations

import types

import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_always_audit import _runs, _slow_box
from .test_add_live_cells_box import CELL_A, CELL_B, _finish, _fire
from .test_add_live_cells_incremental import ci  # noqa: F401
from .test_add_live_tick_note import TWENTY_FIVE, _state
from .test_add_region_neutral import _land_info, _run, hm  # noqa: F401

HUGE = (600, 500, 500)          # 150 million voxels: over the Preview budget


# --------------------------------------------------------------------------- #
# a slow update is cut short by every change that switches live off
# --------------------------------------------------------------------------- #
def test_the_update_tick_going_off_cancels_a_slow_update(ci, monkeypatch):
    _slow_box(ci, monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    worker = ci._add_worker
    assert ci._add_live_slow_in_flight()
    n = _runs()
    ci._on_add_live_toggled(False)
    assert worker.cancel_event.is_set(), "the slow update is cancelled"
    _finish(ci)
    assert _runs() == n, "no run follows the tick going off"
    assert not ci._add_live_slow_in_flight()
    assert ci._add_preview_result is None


def test_a_change_that_makes_live_impossible_cancels_a_slow_update(
        ci, monkeypatch):
    """The preset or the markers' source changed: the schedule finds live
    impossible and cuts the stale run short instead of leaving it."""
    _slow_box(ci, monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    worker = ci._add_worker
    assert ci._add_live_slow_in_flight()
    monkeypatch.setattr(ci, "_add_live_possible", lambda: False)
    assert ci._add_live_schedule() is False
    assert worker.cancel_event.is_set()
    assert not ci._add_live_dirty
    n = _runs()
    _finish(ci)
    assert _runs() == n


def test_a_click_before_stop_is_dropped_with_the_run(ci, monkeypatch):
    """Stop is the answer: a click that came in during the slow update
    (before Stop) does not start another run after Stopped."""
    _slow_box(ci, monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    assert ci._add_live_slow_in_flight()
    assert ci._add_marker_click(CELL_B)          # during the run
    assert ci._add_live_dirty
    ci._stop_add_preview()
    assert not ci._add_live_dirty
    n = _runs()
    _finish(ci)
    assert _runs() == n, "no run follows Stop"
    assert not ci._add_live_timer_active()
    assert ci._add_preview_result is None
    assert ci._add_transient in (AP.ADD_S1_STOPPED, AP.ADD_S1_STOPPED_PATCH)


def test_a_marker_placed_while_stopping_is_drawn_after_the_stop(
        ci, monkeypatch):
    _slow_box(ci, monkeypatch)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    assert ci._add_live_slow_in_flight()
    ci._stop_add_preview()
    assert ci._add_stopping
    assert ci._add_marker_click(CELL_B)
    n = _runs()
    _finish(ci)                 # Stopped is said; the click's run follows
    assert not ci._add_stopping
    if ci._add_live_timer_active():
        _fire(ci)
    assert _runs() == n + 1, "the click placed while stopping is not lost"
    assert ci._add_transient not in (AP.ADD_S1_STOPPED,
                                     AP.ADD_S1_STOPPED_PATCH), \
        "the run that follows says its own S1"
    _finish(ci)
    assert ci._add_preview_result is not None


# --------------------------------------------------------------------------- #
# the labels' write epoch moves on an Undo
# --------------------------------------------------------------------------- #
def test_an_undo_of_an_accept_or_a_clear_moves_the_write_epoch(hm):
    e0 = int(getattr(hm, "_add_volume_epoch", 0))
    hm._add_after_accept_undone(5)
    assert hm._add_volume_epoch == e0 + 1
    hm._add_after_clear_undone([5])
    assert hm._add_volume_epoch == e0 + 2


def test_the_kept_seeds_go_when_the_main_undo_takes_back_an_accept(hm):
    hm._start_add_whole()
    _land_info(hm)
    _run(hm)
    assert hm._add_nuclei_seeds_cache

    def _handed():
        kw, why = hm._add_gather_preview()
        assert why is None, why
        return kw.get("nuclei_seeds") is not None

    assert _handed()
    hm._add_after_accept_undone(1)        # the labels changed under the page
    assert not _handed()


# --------------------------------------------------------------------------- #
# the line under the tick
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("whole", [False, True])
def test_the_note_stands_down_where_preview_is_grey_for_the_size(whole):
    s = _state(HUGE, whole_patch=whole, live_whole_ok=False,
               live_box_ok=False)
    assert AP.preview_block(s), "Preview is grey over the budget"
    assert AP.live_tick_note(s) == ""
    # under the budget the line is back
    s = _state(TWENTY_FIVE, whole_patch=whole, live_whole_ok=False,
               live_box_ok=False)
    assert AP.live_tick_note(s) == AP.ADD_LIVE_NOTE_OVER.format(
        n="20 million")


def test_the_note_stands_down_without_numba_on_a_whole_patch():
    s = _state(TWENTY_FIVE, whole_patch=True, live_whole_ok=False)
    s["numba_ok"] = False
    assert AP.preview_block(s)
    assert AP.live_tick_note(s) == ""


def test_the_over_limit_note_names_the_tick_that_lifts_it():
    note = AP.ADD_LIVE_NOTE_OVER.format(n="20 million")
    assert AP.ADD_LIVE_ALWAYS_LABEL in note
    assert "Preview" not in note
    assert len(note) <= 60


# --------------------------------------------------------------------------- #
# the wait figures and the words
# --------------------------------------------------------------------------- #
def test_the_finder_and_the_anywhere_figures_are_told_apart():
    assert AP.FINDER_SECONDS_PER_MILLION < AP.ANYWHERE_SECONDS_PER_MILLION
    assert AP.finder_minutes(100) == pytest.approx(
        100 * AP.FINDER_SECONDS_PER_MILLION / 60)
    assert AP.anywhere_minutes(100) == pytest.approx(
        100 * AP.ANYWHERE_SECONDS_PER_MILLION / 60)
    n = 50_000_000
    assert AP.patch_slow_line(n) != AP.patch_slow_line(n, find_nuclei=True)
    assert AP.wait_phrase(AP.whole_fill_minutes(50, None)
                          + AP.finder_minutes(50)) in AP.patch_slow_line(
        n, find_nuclei=True)


def test_the_box_slow_note_follows_grow_by():
    n = 25_000_000
    flood = AP.patch_slow_line(n, box=True, growth=AP.GROW_FLOOD)
    spread = AP.patch_slow_line(n, box=True, growth=AP.GROW_SPREAD)
    assert flood != spread
    # a cold box Preview's own rate, not the warm live update's
    assert AP.BOX_FILL_SECONDS_PER_MILLION > AP.LIVE_FLOOD_SECONDS_PER_MILLION
    assert AP.wait_phrase(
        n / 1e6 * AP.BOX_FILL_SECONDS_PER_MILLION / 60) in flood
    # a ridge wall type adds its picture unless it is kept
    ridge = AP.patch_slow_line(n, "frangi", box=True, growth=AP.GROW_FLOOD)
    kept = AP.patch_slow_line(n, "frangi", box=True, growth=AP.GROW_FLOOD,
                              picture_kept=True)
    assert kept == flood and ridge != flood
    assert AP.wait_phrase(
        AP.live_update_seconds(n, None, AP.GROW_SPREAD) / 60) in spread
    # no growth named: the whole patch's fill figure, as before
    assert AP.patch_slow_line(n) == AP.patch_slow_line(n, growth=None)


def test_the_always_help_says_minutes():
    assert "minutes" in AP.ADD_HELP_LIVE_ALWAYS
    assert "a minute" not in AP.ADD_HELP_LIVE_ALWAYS


# --------------------------------------------------------------------------- #
# the spacing note on a Nuclei whole patch with an outline
# --------------------------------------------------------------------------- #
def test_the_spacing_note_shows_on_a_nuclei_whole_patch_with_an_outline(
        widget):
    reset_add(widget)
    widget.set_add_preset("nuclei", silent=True)
    try:
        widget.update_add_ui_state(box_active=True, whole_patch=True,
                                   default_spacing=True)
        oc = widget.add_nuc_outline_combo
        oc.setCurrentIndex(AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        widget.update_add_ui_state(default_spacing=True)
        shown = not widget.add_spacing_note.isHidden()
        moved = (AP.ADD_NOTE_DEFAULT_SPACING in
                 widget._add_advanced_help.text())
        assert shown or moved, "the outline's microns run at the default " \
            "spacing on a whole patch as on a box: the note says so"
        # as on a box
        widget.update_add_ui_state(whole_patch=False, default_spacing=True)
        assert (not widget.add_spacing_note.isHidden()
                or AP.ADD_NOTE_DEFAULT_SPACING in
                widget._add_advanced_help.text())
    finally:
        widget.update_add_ui_state(box_active=False, whole_patch=False,
                                   default_spacing=False)
        reset_add(widget)


# --------------------------------------------------------------------------- #
# a parked thread lets go of its inputs when it ends
# --------------------------------------------------------------------------- #
def test_a_parked_preview_thread_is_released_when_it_ends(h):  # noqa: F811
    from napari_epicycle.annotator import curation_3d_widget as C3W

    class _Th:
        def __init__(self, running):
            self._running = running
            self.finished = types.SimpleNamespace(connect=lambda *_a: None)

        def isRunning(self):
            return self._running

        def wait(self, _ms):
            return not self._running

    big = {"existing_roi": object(), "outline": object()}
    done = types.SimpleNamespace(kind="add", _kwargs=dict(big))
    live = types.SimpleNamespace(kind="add", _kwargs=dict(big))
    other = types.SimpleNamespace(kind="add_info", _kwargs=dict(big))
    th_done, th_live, th_other = _Th(False), _Th(True), _Th(False)
    h._refine_parked_threads = [(th_done, done), (th_live, live),
                                (th_other, other)]
    C3W._ADD_PARKED_THREADS[:] = [(th_done, done), (th_live, live)]
    h._add_parked_watched = [th_done, th_live]
    try:
        h._add_release_parked()
        assert done._kwargs == {}, "the ended thread's inputs are let go"
        assert live._kwargs == big, "a running thread keeps its inputs"
        assert other._kwargs == {}, "an info or clear thread too"
        assert [w for _t, w in h._refine_parked_threads] == [live]
        assert [w for _t, w in C3W._ADD_PARKED_THREADS] == [live]
        assert h._add_parked_watched == [th_live]
        # the module-level slot sweeps without any page (the page that
        # parked the thread may be gone)
        th_live._running = False
        C3W._on_parked_thread_finished()
        assert live._kwargs == {}
        assert C3W._ADD_PARKED_THREADS == []
    finally:
        C3W._ADD_PARKED_THREADS[:] = []
        h._refine_parked_threads = []
        h._add_parked_watched = []

"""Update as I click: Nuclei ▸ At the markers I place, on a box.

With the Advanced tick "Update as I click" on, every marker change (and
every setting change) schedules one run after a short quiet; a change while
a run is in flight marks it dirty and one more run follows.  A live run is
neither busy nor greying: only the strip's line moves.  A marker change
keeps the drawn result until the next one replaces it; a setting change
zeros and hides it.  Accept waits for the last change to be drawn.  A whole
patch is updated as you click too (``test_add_live_whole_patch.py``).  With
the tick off every path is today's.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.annotator.reshape_worker import stage_text
from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeAdv, FakeSignal, FakeThread
from .add_widget_fixture import qapp, widget  # noqa: F401
from .test_add_click_markers import h  # noqa: F401

MARKERS = (AP.LOOK_MARKERS, None)
#: two nuclei inside the click fixture's box (2:18, 2:28, 2:28), clear of
#: its kept object 7 at [5:9, 5:9, 5:9] and of the box's edge zone
NUC_A = (12, 20, 20)
NUC_B = (12, 10, 20)
DIM = (14, 24, 8)


class _FakeTimer:
    """A single-shot timer the test fires by hand."""

    made: list = []

    def __init__(self, parent=None):
        self.parent = parent
        self.timeout = FakeSignal()
        self.starts = []
        self.active = False
        self.single = None
        _FakeTimer.made.append(self)

    def setSingleShot(self, v):            # noqa: N802 - Qt name
        self.single = bool(v)

    def start(self, ms=None):
        self.starts.append(ms)
        self.active = True

    def stop(self):
        self.active = False

    def isActive(self):                    # noqa: N802 - Qt name
        return self.active

    def fire(self):
        self.active = False
        self.timeout.emit()


class LiveAdv(FakeAdv):
    """The fake panel, plus the candidate list's ticks as the real list
    keeps them: an id the orchestrator names in ``candidate_checked`` takes
    that tick, else the row starts unticked at the box edge or small."""

    def update_add_ui_state(self, **kw):
        super().update_add_ui_state(**kw)
        if "candidate_ids" in kw:
            ids = [int(i) for i in kw.get("candidate_ids") or []]
            keep = kw.get("candidate_checked")
            if keep is not None:
                keep = {int(i) for i in keep}
                self.checked = [i for i in ids if i in keep]
            else:
                off = (set(kw.get("candidate_edge_ids") or ())
                       | set(kw.get("candidate_small_ids") or ()))
                self.checked = [i for i in ids if i not in off]


def _image(shape, centres):
    zz, yy, xx = np.indices(shape)
    img = np.zeros(shape, np.float64)
    for cz, cy, cx in centres:
        img[((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= 16] = 1.0
    return img


def _nuclei(hh, *, live):
    from napari.layers import Image

    adv = LiveAdv()
    adv.preset = "nuclei"
    adv.look = MARKERS
    adv.params["click_radius_um"] = 1.0
    hh.adv = adv
    vol = hh.sync_manager.shared_labels["L"]
    hh.viewer.layers.append(Image(_image(vol.shape, (NUC_A, NUC_B)),
                                  name="nuc"))
    hh._add_threshold_value = 0.5
    hh._add_live_enabled = bool(live)
    hidden = []
    real_hide = hh._add_hide_candidate

    def spy_hide():
        hidden.append(True)
        return real_hide()

    hh._add_hide_candidate = spy_hide
    hh.hidden = hidden
    return hh


@pytest.fixture
def lv(h, monkeypatch):  # noqa: F811
    import qtpy.QtCore

    _FakeTimer.made = []
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    return _nuclei(h, live=True)


@pytest.fixture
def off(h, monkeypatch):  # noqa: F811
    import qtpy.QtCore

    _FakeTimer.made = []
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    return _nuclei(h, live=False)


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
    """A click, the quiet, the run, its finish: the result on screen."""
    assert hh._add_marker_click(at)
    _fire(hh)
    _finish(hh)
    assert hh._add_preview_result is not None, hh._add_transient


def _id_at(hh, at):
    return next(int(m["id"]) for m in hh._add_click_markers
                if tuple(m["zyx"]) == tuple(at))


# --------------------------------------------------------------------------- #
# (a) the tick
# --------------------------------------------------------------------------- #
def test_the_constants_say_what_the_tick_does():
    assert AP.ADD_LIVE_DEBOUNCE_MS == 150
    assert AP.ADD_LIVE_LABEL == "Update as I click"
    assert AP.ADD_HELP_LIVE_TICK == (
        "Finds the objects again after every click and setting change, so "
        "you see the result without pressing Preview. Preview still works.")
    assert AP.ADD_S1_LIVE_OVER_GATE_PATCH == (
        "Press Preview to see the cells — this patch is not updated as you "
        "click.")
    assert AP.ADD_S1_ACCEPT_WAITING == (
        "Accepting once the last click is drawn.")
    assert AP.ADD_S1_ACCEPT_WAIT_CLICK == (
        "Wait — Accept is finishing; place more markers after it.")
    assert AP.ADD_S1_ACCEPT_WAIT_OFF == (
        "Accept was cancelled — the result changed before it was drawn; "
        "press Accept again.")
    assert AP.ADD_ACCEPT_WAITING_BTN == "Accepting…"
    assert AP.ADD_S1_CLEAR_WAIT_LIVE == (
        "Wait — the last click is still being drawn; Clear once it appears.")


def test_the_tick_is_on_by_default_and_carries_its_help(widget):  # noqa: F811
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    cb = widget.add_live_cb
    assert cb.text() == AP.ADD_LIVE_LABEL
    assert cb.isChecked()
    assert cb.toolTip() == AP.ADD_HELP_LIVE_TICK
    assert ADD_HELP_LINES["add_live_cb"] == AP.ADD_HELP_LIVE_TICK


def test_the_tick_shows_only_with_the_markers(widget):  # noqa: F811
    from .add_widget_fixture import reset_add

    reset_add(widget)
    try:
        widget.update_add_ui_state(box_active=True)
        keys = list(widget._add_look_keys)
        widget.add_look_for_combo.setCurrentIndex(keys.index(MARKERS))
        assert not widget.add_live_cb.isHidden()
        widget.add_look_for_combo.setCurrentIndex(0)
        assert widget.add_live_cb.isHidden()
        # Cells ▸ Click markers: shown too (a small box runs as you click)
        widget.set_add_preset("cells")
        widget.update_add_ui_state(marker_sources=[
            (AP.ADD_ITEM_CLICKS, True, ("paint", None))])
        skeys = list(widget._add_source_keys)
        widget.add_marker_source_combo.setCurrentIndex(
            skeys.index(("paint", None)))
        assert not widget._add_click_rows.isHidden()
        assert not widget.add_live_cb.isHidden()
    finally:
        widget.set_add_preset("nuclei")
        widget.add_look_for_combo.setCurrentIndex(0)
        widget.update_add_ui_state(box_active=False)
        reset_add(widget)


def test_the_accept_button_reads_accepting_while_it_waits(widget):  # noqa: F811
    from .add_widget_fixture import reset_add

    reset_add(widget)
    try:
        widget.update_add_ui_state(box_active=True, candidate_ids=[5, 6],
                                   candidate_sizes={5: 10, 6: 10},
                                   candidate_edge_ids=set())
        assert widget.add_accept_btn.text() == "Accept (2)"
        assert widget.add_accept_btn.isEnabled()
        widget.update_add_ui_state(accept_waiting=True)
        assert widget.add_accept_btn.text() == AP.ADD_ACCEPT_WAITING_BTN
        assert not widget.add_accept_btn.isEnabled()
        # a live fill while it waits keeps the word and the grey
        widget.update_add_ui_state(candidate_ids=[5, 6, 7],
                                   candidate_sizes={5: 10, 6: 10, 7: 10},
                                   candidate_edge_ids=set())
        assert widget.add_accept_btn.text() == AP.ADD_ACCEPT_WAITING_BTN
        assert not widget.add_accept_btn.isEnabled()
        widget.update_add_ui_state(accept_waiting=False)
        assert widget.add_accept_btn.text() == "Accept (3)"
        assert widget.add_accept_btn.isEnabled()
    finally:
        widget.update_add_ui_state(accept_waiting=False, candidate_ids=[],
                                   candidate_sizes={},
                                   candidate_edge_ids=set(),
                                   box_active=False)
        reset_add(widget)


class _Tick:
    def __init__(self, on):
        self.on = on
        self.blocked = []

    def isChecked(self):                    # noqa: N802 - Qt name
        return self.on

    def setChecked(self, v):                # noqa: N802 - Qt name
        self.on = bool(v)

    def blockSignals(self, v):              # noqa: N802 - Qt name
        self.blocked.append(bool(v))


class _RestoreAdv(FakeAdv):
    def __init__(self):
        super().__init__()
        self.add_live_cb = _Tick(True)
        self.add_dark_outside_cb = _Tick(True)

    def set_add_preset(self, preset, silent=False):
        self.preset = preset

    def _apply_add_visibility(self):
        pass


def test_the_tick_is_kept_per_workspace_and_restored(h):  # noqa: F811
    h.adv = _RestoreAdv()
    h._add_live_enabled = True
    h._on_add_live_toggled(False)
    assert h._add_live_enabled is False
    assert h._add_session_memory["live_update"] is False
    h._add_live_enabled = True
    h.adv.add_live_cb.on = True
    h._add_restore_preset()
    assert h.adv.add_live_cb.on is False
    assert h._add_live_enabled is False
    # restored with its signals blocked (no save, no run)
    assert h.adv.add_live_cb.blocked == [True, False]
    # a string from the settings file reads as a bool
    h._add_session_memory["live_update"] = "true"
    h._add_restore_preset()
    assert h._add_live_enabled is True and h.adv.add_live_cb.on is True
    h._add_session_memory["live_update"] = "false"
    h._add_restore_preset()
    assert h._add_live_enabled is False


def test_the_harness_defaults_to_live_off(h):  # noqa: F811
    assert not h._add_live_on()


# --------------------------------------------------------------------------- #
# (b) (l) a click schedules; the run is neither busy nor greying
# --------------------------------------------------------------------------- #
def test_a_click_starts_the_quiet_and_the_run_is_not_busy(lv):
    lv._add_place_markers_on = True
    pushes0 = len(lv.adv.pushes)
    assert lv._add_marker_click(NUC_A)
    t = _timer(lv)
    assert t is not None and t.single is True
    assert t.starts[-1] == AP.ADD_LIVE_DEBOUNCE_MS
    assert FakeThread.instances == []          # nothing runs in the quiet
    _fire(lv)
    assert len(FakeThread.instances) == 1
    assert lv._add_busy is False
    assert lv.adv.last("measuring") == stage_text("add_box")
    assert lv.adv.last("busy") is False
    assert lv.adv.last("stop_enabled") is False
    # S1 never reads busy
    assert AP.s1_sentence(lv._add_s1_state()) not in (AP.ADD_S1_WORKING,
                                                      AP.ADD_S1_WORKING_PATCH)
    _finish(lv)
    # the page never greyed: no push named busy=True from click to finish
    assert not any(kw.get("busy") is True
                   for kw in lv.adv.pushes[pushes0:])
    assert lv._add_preview_result is not None
    assert lv.adv.last("measuring") == ""


# --------------------------------------------------------------------------- #
# (c) the live result is the manual Preview's
# --------------------------------------------------------------------------- #
def test_the_live_result_equals_a_manual_preview(lv):
    _click_and_draw(lv, NUC_A)
    a = _id_at(lv, NUC_A)
    assert a in lv.adv.last("candidate_ids")
    live = np.array(lv._add_preview_result, copy=True)
    assert (live == a).any()
    lv._preview_add()
    assert lv._add_busy                         # the manual way: busy
    assert not _active(lv)                      # a launch ends the quiet
    _finish(lv)
    assert np.array_equal(live, lv._add_preview_result)


# --------------------------------------------------------------------------- #
# (d) five clicks, one run; a click mid-run: dirty, one relaunch; stale
# --------------------------------------------------------------------------- #
def test_five_fast_clicks_make_one_run(lv):
    for at in (NUC_A, NUC_B, DIM, (14, 24, 14), (10, 24, 22)):
        assert lv._add_marker_click(at)
    assert len(_timer(lv).starts) == 5
    assert FakeThread.instances == []
    _fire(lv)
    assert len(FakeThread.instances) == 1
    _finish(lv)
    assert len(FakeThread.instances) == 1
    assert not _active(lv) and not lv._add_live_dirty


def test_a_click_mid_run_is_never_lost(lv):
    assert lv._add_marker_click(NUC_A)
    _fire(lv)
    th = FakeThread.instances[-1]
    assert lv._add_marker_click(NUC_B)          # not refused: not busy
    assert lv._add_live_dirty
    assert not _active(lv)                      # dirty, not a second quiet
    assert len(FakeThread.instances) == 1
    th.run()
    # launched on the first click only: stale, discarded
    assert lv._add_preview_result is None
    th.end()
    assert len(FakeThread.instances) == 2       # exactly one relaunch
    assert not lv._add_live_dirty
    _finish(lv)
    assert len(FakeThread.instances) == 2
    res = lv._add_preview_result
    assert res is not None
    assert (res == _id_at(lv, NUC_A)).any()
    assert (res == _id_at(lv, NUC_B)).any()


def test_a_setting_change_mid_run_makes_it_stale(lv):
    assert lv._add_marker_click(NUC_A)
    _fire(lv)
    th = FakeThread.instances[-1]
    lv._on_add_threshold_changed(0.4)
    assert lv._add_live_dirty
    th.run()
    assert lv._add_preview_result is None       # settings changed: stale
    th.end()
    assert len(FakeThread.instances) == 2
    _finish(lv)
    assert lv._add_preview_result is not None


# --------------------------------------------------------------------------- #
# (e) a marker change keeps the drawn result; a setting change zeros it
# --------------------------------------------------------------------------- #
def test_a_marker_change_keeps_the_drawn_result(lv):
    _click_and_draw(lv, NUC_A)
    drawn = lv._add_preview_result
    lv.hidden.clear()
    n_cand = sum(1 for kw in lv.adv.pushes if "candidate_ids" in kw)
    assert lv._add_marker_click(NUC_B)
    assert lv._add_preview_result is drawn
    assert lv.hidden == []
    assert sum(1 for kw in lv.adv.pushes if "candidate_ids" in kw) == n_cand
    assert _active(lv)


def test_a_setting_change_zeros_hides_and_schedules(lv):
    _click_and_draw(lv, NUC_A)
    lv.hidden.clear()
    lv._on_add_threshold_changed(0.4)
    assert lv._add_preview_result is None
    assert lv.hidden == [True]
    assert lv.adv.last("candidate_ids") == []
    assert _active(lv)


def test_the_accept_ticks_stick_by_id(lv):
    """Both nuclei reach the edge zone here: a lone object stays ticked,
    two start unticked.  A, ticked while alone, keeps its tick; B is new
    and takes its default."""
    _click_and_draw(lv, NUC_A)
    a = _id_at(lv, NUC_A)
    assert lv.adv.checked == [a]                 # lone: ticked
    assert lv._add_marker_click(NUC_B)
    _fire(lv)
    _finish(lv)
    b = _id_at(lv, NUC_B)
    res = lv._add_preview_res
    assert {a, b} <= set(res.edge_ids)           # the defaults: both off
    assert set(lv.adv.last("candidate_checked")) == {a}
    assert lv.adv.checked == [a]
    # the user unticks A: it stays off through the next result
    lv.adv.checked = []
    lv._on_add_threshold_changed(0.5)            # a setting: zeros ...
    assert lv._add_preview_result is None
    _fire(lv)
    _finish(lv)
    # ... and with nothing on screen every row takes its default again
    assert set(lv.adv.last("candidate_checked")) == set()
    lv.adv.checked = [b]                         # the user ticks B
    assert lv._add_marker_click(DIM)             # a marker: keeps the rows
    _fire(lv)
    _finish(lv)
    assert set(lv.adv.last("candidate_checked")) == {b}


# --------------------------------------------------------------------------- #
# (f) Accept waits for the last change
# --------------------------------------------------------------------------- #
def _accepted_ids(hh):
    vol = hh.sync_manager.shared_labels["L"]
    return {int(v) for v in np.unique(vol)} - {0, 7}


def test_accept_in_the_quiet_waits_then_commits_the_last_click(lv):
    lv.adv.params["clear_border"] = False         # both rows start ticked
    _click_and_draw(lv, NUC_A)
    assert lv._add_marker_click(NUC_B)
    assert _active(lv)
    lv._accept_add()
    assert lv._add_accept_pending
    assert lv._add_transient == AP.ADD_S1_ACCEPT_WAITING
    assert _accepted_ids(lv) == set()            # nothing written yet
    # no new marker while it waits
    assert not lv._add_marker_click(DIM)
    assert lv._add_transient == AP.ADD_S1_ACCEPT_WAIT_CLICK
    assert len(lv._add_click_markers) == 2
    _fire(lv)
    _finish(lv)
    assert not lv._add_accept_pending
    assert _accepted_ids(lv) == {_id_at_gone(lv, NUC_A), _id_at_gone(lv,
                                                                     NUC_B)}


def _id_at_gone(hh, at):
    """The id the shared volume holds at ``at`` (the markers are gone after
    an Accept)."""
    return int(hh.sync_manager.shared_labels["L"][at])


def test_a_refused_run_cancels_the_wait_and_zeros_the_box(lv):
    _click_and_draw(lv, NUC_A)
    lv.hidden.clear()
    assert lv._add_marker_click(NUC_B)
    lv._accept_add()
    assert lv._add_accept_pending
    lv.adv.params["channel"] = "gone"            # the channel cannot be read
    _fire(lv)
    assert not lv._add_accept_pending
    assert lv._add_preview_result is None        # (g) the model goes ...
    assert lv.hidden == [True]                   # ... with the drawn box
    assert lv.adv.last("candidate_ids") == []
    # the wait ends with its own sentence (the console names the reason)
    assert lv._add_transient == AP.ADD_S1_ACCEPT_WAIT_OFF
    assert _accepted_ids(lv) == set()
    assert not _active(lv)                       # a refusal never loops


def test_discard_cancels_the_wait_and_stops_the_quiet(lv):
    _click_and_draw(lv, NUC_A)
    assert lv._add_marker_click(NUC_B)
    lv._accept_add()
    assert lv._add_accept_pending
    lv._add_secondary()                          # Discard (a result is up)
    assert not lv._add_accept_pending
    assert lv._add_transient == AP.ADD_S1_ACCEPT_WAIT_OFF
    assert not _active(lv) and not lv._add_live_dirty
    assert lv._add_preview_result is None


# --------------------------------------------------------------------------- #
# (h) Preview during a live run; the busy refusal of a manual run
# --------------------------------------------------------------------------- #
def test_preview_during_a_live_run_marks_it_dirty(lv):
    assert lv._add_marker_click(NUC_A)
    _fire(lv)
    lv._add_transient = None
    lv._preview_add()
    assert lv._add_live_dirty
    assert len(FakeThread.instances) == 1
    assert lv._add_transient is None
    assert lv._add_busy is False
    FakeThread.instances[-1].run()
    FakeThread.instances[-1].end()
    assert len(FakeThread.instances) == 2


def test_a_manual_run_still_refuses_a_click_busy(lv):
    lv._add_click_markers = [{"zyx": NUC_A, "id": 8, "ticked": True}]
    lv._preview_add()
    assert lv._add_busy
    assert not lv._add_marker_click(NUC_B)
    assert lv._add_transient == AP.ADD_S1_BUSY_REFUSED


# --------------------------------------------------------------------------- #
# (i) the whole patch: the markers run as you click there too
# --------------------------------------------------------------------------- #
def test_a_whole_patch_runs_live_and_s1_names_no_gate(lv):
    lv._add_whole_patch = True
    assert lv._add_marker_click(NUC_A)
    assert _active(lv)
    assert "not updated as you click" not in AP.s1_sentence(
        lv._add_s1_state())


def _whole_state(**kw):
    s = {"has_patch": True, "has_box": True, "preset": "nuclei",
         "look_for": AP.LOOK_MARKERS, "channel": "nuc",
         "box_shape": (20, 30, 30), "whole_patch": True, "n_clicks": 1,
         "n_ticked": 1, "place_markers_on": True, "live_on": True,
         "source": AP.SOURCE_CLICKS}
    s.update(kw)
    return s


def test_the_gate_sentence_sits_at_row_13_5_for_cells_only():
    s = _whole_state()
    # Nuclei ▸ the markers on a whole patch run as you click: no gate
    assert "not updated as you click" not in AP.s1_sentence(s)
    assert AP.preview_block(s) is None
    # Cells ▸ Click markers on a whole patch: still Preview
    cells = _whole_state(preset="cells", wall_channel="mem")
    assert AP.s1_sentence(cells) == AP.ADD_S1_LIVE_OVER_GATE_PATCH
    # the rows above it win: a transient, busy, candidates, a block
    assert AP.s1_sentence(_whole_state(transient="x")) == "x"
    assert AP.s1_sentence(_whole_state(busy=True)) == AP.ADD_S1_WORKING_PATCH
    assert AP.s1_sentence(_whole_state(n_clicks=0)) == AP.ADD_S1_NO_MARKERS_ON
    assert AP.s1_sentence(_whole_state(candidates=(2, 0, 0, 0, 2))) == (
        AP.found_line(2))
    # live off, or a box: the ready line
    for other in (_whole_state(live_on=False),
                  _whole_state(whole_patch=False)):
        assert AP.s1_sentence(other) != AP.ADD_S1_LIVE_OVER_GATE_PATCH
        assert "not updated as you click" not in AP.s1_sentence(other)
    # the row-13.5 line itself is not a warning: Preview is live there
    assert AP.s1_kind(cells) == "ready"


# --------------------------------------------------------------------------- #
# (j) the quiet ends at every exit
# --------------------------------------------------------------------------- #
def _scheduled(hh):
    assert hh._add_marker_click(NUC_A)
    assert _active(hh)


def test_the_tick_off_stops_the_quiet(lv):
    _scheduled(lv)
    lv._on_add_live_toggled(False)
    assert not _active(lv)
    assert not lv._add_live_on()


def test_reset_stops_the_quiet(lv):
    _scheduled(lv)
    lv._reset_add_state()
    assert not _active(lv)


def test_close_box_stops_the_quiet(lv):
    _scheduled(lv)
    lv._add_secondary()                          # no result, no session
    assert not _active(lv)
    assert not lv._add_running


def test_close_box_back_to_the_review_stops_the_quiet(lv):
    _scheduled(lv)
    lv._add_session = [{"ids": [99], "sizes": {99: 1}}]
    lv._add_resume_review()
    assert not _active(lv)


def test_the_patch_switch_teardown_stops_the_quiet(lv):
    _scheduled(lv)
    lv._reject_add()
    assert not _active(lv)


def test_accept_finally_stops_the_quiet(lv, monkeypatch):
    _click_and_draw(lv, NUC_A)
    lv._add_live_timer.start(AP.ADD_LIVE_DEBOUNCE_MS)
    # the wait is proven above; here only the finally's stop
    monkeypatch.setattr(type(lv), "_add_live_unsettled",
                        lambda self: False, raising=False)
    lv._accept_add()
    assert lv._add_review
    assert not _active(lv)


# --------------------------------------------------------------------------- #
# (k) the tick off: today's flow
# --------------------------------------------------------------------------- #
def test_live_off_a_click_drops_the_result_as_today(off):
    off._add_click_markers = [{"zyx": NUC_A, "id": 8, "ticked": True}]
    off._preview_add()
    _finish(off)
    assert off._add_preview_result is not None
    off.hidden.clear()
    assert off._add_marker_click(NUC_B)
    assert off._add_preview_result is None
    assert off.hidden == [True]
    assert off.adv.last("candidate_ids") == []
    assert _timer(off) is None                  # no timer is ever made
    off._on_add_threshold_changed(0.4)
    assert _timer(off) is None


# --------------------------------------------------------------------------- #
# fix round: Clear and a live run
# --------------------------------------------------------------------------- #
def _spy_clear(hh):
    calls = []
    hh._launch_add_clear_plan = lambda mode: calls.append(mode)
    hh.staged_kinds = lambda: set()
    return calls


@pytest.mark.parametrize("when", ["quiet", "in flight", "dirty"])
def test_clear_is_refused_while_a_live_run_is_owed(lv, when):
    calls = _spy_clear(lv)
    assert lv._add_marker_click(NUC_A)
    if when == "in flight":
        _fire(lv)
    if when == "dirty":
        # dirty alone: no quiet, no thread (a parked thread holds it)
        _timer(lv).stop()
        lv._add_live_dirty = True
        assert lv._add_thread is None and not _active(lv)
    lv._on_add_clear_requested("box")
    assert calls == []
    assert lv._add_transient == AP.ADD_S1_CLEAR_WAIT_LIVE


def test_clear_while_accept_waits_keeps_the_accept_sentence(lv):
    calls = _spy_clear(lv)
    _waiting(lv)
    assert lv._add_transient == AP.ADD_S1_ACCEPT_WAITING
    lv._on_add_clear_requested("box")
    assert calls == []
    assert lv._add_transient == AP.ADD_S1_ACCEPT_WAITING
    assert lv._add_accept_pending


def test_a_clear_sentence_outlives_the_update_it_owed(lv):
    assert lv._add_marker_click(NUC_A)
    _timer(lv).stop()
    lv._add_clear_phase = "clearing"
    lv._add_busy = True
    lv._add_live_input_landed("the Auto threshold")
    assert lv._add_live_dirty
    lv._add_clear_done("Cleared 3 objects.")
    assert lv._add_transient == "Cleared 3 objects."
    _finish(lv)                                  # the owed update lands
    assert lv._add_preview_result is not None
    assert lv._add_transient == "Cleared 3 objects."
    # ... for that one landing only
    assert lv._add_marker_click(NUC_B)
    _fire(lv)
    _finish(lv)
    assert lv._add_transient is None


def test_clear_runs_once_the_live_run_is_drawn_and_gone(lv):
    calls = _spy_clear(lv)
    assert lv._add_marker_click(NUC_A)
    _fire(lv)
    _finish(lv)
    lv._add_secondary()                          # Discard the result
    lv._on_add_clear_requested("box")
    assert calls == ["box"]


def test_a_live_finish_never_writes_busy(lv):
    assert lv._add_marker_click(NUC_A)
    _fire(lv)
    th = FakeThread.instances[-1]
    # a Clear holding the page busy (reached however): the live run's end
    # must not let the page go
    lv._add_clear_phase = "counting"
    lv._add_busy = True
    th.run()
    th.end()
    assert lv._add_busy is True
    assert lv._add_clear_phase == "counting"


def test_a_change_during_a_clear_is_drawn_after_it(lv):
    assert lv._add_marker_click(NUC_A)
    _timer(lv).stop()
    lv._add_clear_phase = "clearing"
    lv._add_busy = True
    # clicks are refused busy during a Clear; an input that lands while the
    # Clear holds the page owes one run
    assert not lv._add_marker_click(NUC_B)
    lv._add_live_input_landed("the Auto threshold")
    assert lv._add_live_dirty and FakeThread.instances == []
    epoch = getattr(lv, "_add_settings_epoch", 0)
    lv._add_clear_done(None)
    assert lv._add_settings_epoch > epoch       # the volume changed
    assert len(FakeThread.instances) == 1
    assert not lv._add_live_dirty
    _finish(lv)
    assert lv._add_preview_result is not None


# --------------------------------------------------------------------------- #
# fix round: a waiting Accept never outlives its run
# --------------------------------------------------------------------------- #
def _waiting(hh):
    _click_and_draw(hh, NUC_A)
    assert hh._add_marker_click(NUC_B)
    hh._accept_add()
    assert hh._add_accept_pending
    assert hh.adv.last("accept_waiting") is True


def _cancelled(hh):
    assert not hh._add_accept_pending
    assert hh._add_transient == AP.ADD_S1_ACCEPT_WAIT_OFF
    assert hh.adv.last("accept_waiting") is False


def test_accept_goes_on_only_after_the_live_thread_ends(lv):
    lv.adv.params["clear_border"] = False
    _waiting(lv)
    _fire(lv)
    th = FakeThread.instances[-1]
    th.run()
    assert lv._add_accept_pending                # drawn, thread not ended
    assert _accepted_ids(lv) == set()
    th.end()
    assert not lv._add_accept_pending
    assert len(_accepted_ids(lv)) == 2
    assert lv._add_review


def test_a_preset_change_cancels_the_wait(lv):
    _waiting(lv)
    lv.adv.preset = "cells"
    lv._on_add_preset_changed("cells")
    _cancelled(lv)


def test_a_look_for_change_cancels_the_wait(lv):
    _waiting(lv)
    lv.adv.look = (AP.LOOK_ANYWHERE, None)
    lv._on_add_look_for_changed(lv.adv.look)
    _cancelled(lv)


def test_a_wait_cancelled_mid_run_is_never_resumed_by_preview(lv):
    _waiting(lv)
    _fire(lv)
    th = FakeThread.instances[-1]
    lv.adv.look = (AP.LOOK_ANYWHERE, None)
    lv._on_add_look_for_changed(lv.adv.look)
    th.run()
    th.end()
    _cancelled(lv)
    lv.adv.look = MARKERS
    lv._on_add_look_for_changed(lv.adv.look)
    lv._preview_add()                            # a manual Preview
    _finish(lv)
    assert lv._add_preview_result is not None
    assert _accepted_ids(lv) == set()            # a Preview never accepts
    assert not lv._add_review


def test_the_tick_off_cancels_the_wait(lv):
    _waiting(lv)
    lv._on_add_live_toggled(False)
    _cancelled(lv)


def test_stop_cancels_the_wait(lv):
    _waiting(lv)
    lv._add_stopping = True
    lv._on_add_stopped()
    _cancelled(lv)


def test_the_cancel_paints_its_sentence(lv):
    lv._add_accept_pending = True
    lv._add_live_cancel_accept("a test")
    _cancelled(lv)


def test_a_refused_run_from_the_worker_zeros_and_cancels(lv):
    _click_and_draw(lv, NUC_A)
    lv.hidden.clear()
    # only a marker on dim stain is left ticked: the engine refuses
    assert lv._add_marker_click(DIM)
    for m in lv._add_click_markers:
        if tuple(m["zyx"]) != DIM:
            m["ticked"] = False
    lv._accept_add()
    assert lv._add_accept_pending
    _fire(lv)
    th = FakeThread.instances[-1]
    th.run()
    assert lv._add_preview_res is None or lv._add_preview_result is None
    th.end()
    assert lv._add_preview_result is None
    assert lv.hidden == [True]
    assert lv.adv.last("candidate_ids") == []
    _cancelled(lv)


def test_a_failing_live_run_zeros_and_says_so(lv):
    _click_and_draw(lv, NUC_A)
    lv.hidden.clear()
    assert lv._add_marker_click(NUC_B)
    _fire(lv)
    lv._on_add_error("boom")
    assert lv._add_preview_result is None
    assert lv.hidden == [True]
    assert lv._add_transient == AP.ADD_S1_FAILED
    assert lv._add_busy is False


def test_a_failed_live_launch_zeros_the_box(lv):
    _click_and_draw(lv, NUC_A)
    lv.hidden.clear()
    assert lv._add_marker_click(NUC_B)
    FakeThread.fail_start = True
    try:
        _fire(lv)
    finally:
        FakeThread.fail_start = False
    assert lv._add_preview_result is None
    assert lv.hidden == [True]
    assert lv._add_transient == AP.ADD_S1_FAILED
    assert lv._add_thread is None and not lv._add_busy


# --------------------------------------------------------------------------- #
# fix round: dirty never stranded, a manual run never lost
# --------------------------------------------------------------------------- #
def test_several_clicks_mid_run_make_one_relaunch(lv):
    assert lv._add_marker_click(NUC_A)
    _fire(lv)
    th = FakeThread.instances[-1]
    for at in (NUC_B, DIM, (14, 24, 14)):
        assert lv._add_marker_click(at)
    assert len(FakeThread.instances) == 1 and lv._add_live_dirty
    th.run()
    th.end()
    assert len(FakeThread.instances) == 2
    _finish(lv)
    assert len(FakeThread.instances) == 2


@pytest.mark.parametrize("order", ["off then preview", "preview then off"])
def test_a_preview_owed_by_a_live_run_happens_with_the_tick_off(lv, order):
    assert lv._add_marker_click(NUC_A)
    _fire(lv)
    th = FakeThread.instances[-1]
    if order == "off then preview":
        lv._on_add_live_toggled(False)
        lv._preview_add()
    else:
        lv._preview_add()
        lv._on_add_live_toggled(False)
    th.run()
    th.end()
    assert len(FakeThread.instances) == 2        # the manual run
    assert lv._add_busy                          # ... the manual way
    _finish(lv)
    assert lv._add_preview_result is not None


def test_stop_drops_a_run_owed(lv):
    assert lv._add_marker_click(NUC_A)
    _timer(lv).stop()
    lv._preview_add()                            # a manual run
    assert lv._add_busy
    lv._add_live_dirty = True                    # owed (an input landed)
    lv._stop_add_preview()
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert not lv._add_live_dirty
    assert len(FakeThread.instances) == 1


class _Parked:
    def __init__(self):
        self.running = True

    def isRunning(self):                    # noqa: N802 - Qt name
        return self.running


class _ParkedWorker:
    kind = "add"


def test_a_parked_thread_does_not_strand_the_run(lv):
    pt = _Parked()
    lv._refine_parked_threads = [(pt, _ParkedWorker())]
    assert lv._add_marker_click(NUC_A)
    _fire(lv)
    assert FakeThread.instances == [] and lv._add_live_dirty
    pt.running = False
    lv._on_add_parked_finished()
    assert len(FakeThread.instances) == 1
    assert not lv._add_live_dirty


# --------------------------------------------------------------------------- #
# fix round: the user's unticks survive an invalidation
# --------------------------------------------------------------------------- #
def test_an_untick_survives_an_auto_landing(lv):
    _click_and_draw(lv, NUC_A)
    a = _id_at(lv, NUC_A)
    assert lv.adv.checked == [a]
    lv.adv.checked = []                          # the user unticks A
    lv._sync_add_candidate_overlay()
    assert a in lv._add_unticked_ids
    lv._add_threshold_value = 0.45               # an Auto lands
    lv._add_live_input_landed("the Auto threshold")
    assert lv._add_preview_result is None
    _fire(lv)
    _finish(lv)
    assert a in lv._add_preview_res.ids
    assert set(lv.adv.last("candidate_checked")) == set()


def test_a_freed_id_forgets_its_untick(lv):
    lv._add_unticked_ids = {8}
    lv._add_forget_marks([8])
    assert 8 not in lv._add_unticked_ids


def test_the_input_landing_schedules(lv):
    assert lv._add_marker_click(NUC_A)
    _timer(lv).stop()
    lv._add_live_input_landed("the tissue outline")
    assert _active(lv)


# --------------------------------------------------------------------------- #
# fix round: the whole patch start restores the tick too
# --------------------------------------------------------------------------- #
def test_the_whole_patch_start_restores_the_tick(monkeypatch):
    from .add_harness import Harness, blank, install_fakes

    install_fakes(monkeypatch)
    hh = Harness(blank())
    hh.adv = _RestoreAdv()
    hh._add_session_memory = {"live_update": False}
    hh._add_live_enabled = True
    hh._start_add_whole()
    assert hh._add_whole_patch
    assert hh._add_live_enabled is False
    assert hh.adv.add_live_cb.on is False


def test_a_live_error_while_accept_waits_cancels_and_zeros(lv):
    _waiting(lv)
    lv.hidden.clear()
    _fire(lv)
    lv._on_add_error("boom")
    assert lv._add_preview_result is None
    assert lv.hidden == [True]
    _cancelled(lv)
    assert _accepted_ids(lv) == set()


def test_a_rebase_carries_the_users_unticks(lv):
    lv.adv.params["clear_border"] = False         # every row starts ticked
    _click_and_draw(lv, NUC_A)
    a = _id_at(lv, NUC_A)
    lv.adv.checked = []                          # the user unticks A
    lv._sync_add_candidate_overlay()
    assert lv._add_unticked_ids == {a}
    # the df floor moves: the next run re-bases every reserved id
    lv._get_label_id_floor_cb = lambda: 50
    assert lv._add_marker_click(NUC_B)
    _fire(lv)
    _finish(lv)
    a2 = _id_at(lv, NUC_A)
    d = a2 - a
    assert d > 0
    assert lv._add_unticked_ids == {a2}
    assert a2 in lv._add_preview_res.ids
    b2 = _id_at(lv, NUC_B)
    assert set(lv.adv.last("candidate_checked")) == {b2}
    assert lv._add_last_checked == {b2}

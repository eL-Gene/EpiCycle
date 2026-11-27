"""Every wait on the Add page says what it is doing.

A wait on the main thread freezes the app: no timer runs and no queued
repaint happens until it ends.  So its words -- S1 and the strip's grey
line -- are set and painted BEFORE the heavy call starts, and the bar
cannot animate: it stays still, or, on a whole patch where the page owns
the steps, it steps between them and is never full while frozen.  A wait on
a worker (the app is live) gets the moving bar, including the measuring
of the box, the outline and a run read.  The words never stick: every wait
ends with the page's own state, also when its heavy call raises.

Pinned here:

* painted before -- one ordered event log: the panel's paint and the spied
  heavy call append to the same list;
* the steps of the whole-patch waits, never full; one-step waits leave the
  bar alone;
* the measuring bar runs only while S1 says measuring and a reader is alive
  or due; the controls stay live and Stop grey;
* every wait ends idle, also when its heavy call raises; every caller of the
  ball drawing ends idle; a box Preview end and a box Clear paint once, a
  later click never;
* the ridge notes of Cells: their rules and their place in the one amber
  slot;
* the panel's paint: words and bar set directly, the layouts activated and
  repainted, never ``processEvents``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from napari_epicycle.annotator.reshape_worker import stage_text
from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.orientation import transpose_data
from packages.epicycle_annotator.reshape import region as R

from .add_harness import FakeThread, Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_click_markers import _Layers, _SubMgr, _VM
from .test_add_outline_wiring import _CellsAdv, _land_info, _world
from .test_reshape_region import _threshold


class _Boom(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# the event log
# --------------------------------------------------------------------------- #
def _wire_log(h):
    """The panel's paint and every push append to ``h.log``, in order."""
    h.log = []
    log = h.log
    # what each paint asked of Stop (None: left as it was), in paint order
    h.paint_stops = []
    stops = h.paint_stops

    def paint_add_wait(*, status_text, status_kind="busy", stage_text="",
                       step=None, of=None, stop_enabled=None):
        log.append(("paint", str(status_text), str(stage_text), step, of))
        stops.append(stop_enabled)

    real = h.adv.update_add_ui_state

    def update(**kw):
        log.append(("push", kw))
        real(**kw)

    h.adv.paint_add_wait = paint_add_wait
    h.adv.update_add_ui_state = update


def _paints(h):
    return [e for e in h.log if e[0] == "paint"]


def _at(h, entry):
    return h.log.index(entry)


def _first_paint(h, words):
    for i, e in enumerate(h.log):
        if e[0] == "paint" and e[1] == words:
            return i
    raise AssertionError("{!r} was never painted".format(words))


def _ends_idle(h, words):
    """After the wait's last paint the page pushed its own state: a full
    refresh (it carries ``busy``, so the strip is put back) whose S1 is not
    the wait's words."""
    last = max(i for i, e in enumerate(h.log) if e[0] == "paint")
    after = [e[1] for e in h.log[last + 1:]
             if e[0] == "push" and "busy" in e[1]]
    assert after, "the wait's words were never replaced"
    assert after[-1].get("status_text") != words, "the words stuck"


# --------------------------------------------------------------------------- #
# the harnesses
# --------------------------------------------------------------------------- #
class _Layer:
    def __init__(self):
        self.visible = False


def _cells(monkeypatch):
    install_fakes(monkeypatch)
    ch, cells = _world()
    h = Harness(np.zeros(cells.shape, np.int32), images=ch)
    h.adv = _CellsAdv()
    h.adv.params["outline"] = "none"
    h.ctx["spacing_source"] = "qc_settings"
    h._add_run_guide = cells
    h._add_dark_cache = {h._add_dark_key(): _threshold(ch)}
    h.temp = []
    h._add_temp_layer = _record_temp(h)
    h._add_confirm = _yes
    _wire_log(h)
    return h


def _record_temp(h):
    def _temp(key, array, **kw):
        h.log.append(("heavy", "temp_layer"))
        if getattr(h, "raise_temp", False):
            raise _Boom("temp layer")
        h.temp.append(key)
        return _Layer()
    return _temp


def _yes(_text):
    return True


BOX = (slice(0, 12), slice(0, 20), slice(0, 20))


def _measure(h):
    h._request_add_box_info()
    _land_info(h)
    h._add_refresh_ui()


def _preview(h):
    h._preview_add()
    th = h._add_thread
    th.run()
    th.end()
    return h._add_preview_res


def _box_result(h):
    h.place_box(BOX)
    _measure(h)
    res = _preview(h)
    assert res is not None and res.ok, getattr(res, "refusal", None)
    # the harness draws nothing: a layer as the real one would leave
    h._add_candidate_layer = _Layer()
    return res


def _whole_result(h):
    h._start_add_whole()
    _land_info(h)
    res = _preview(h)
    assert res is not None and res.ok, getattr(res, "refusal", None)
    h._add_candidate_layer = _Layer()
    return res


def _clicks(monkeypatch):
    """The click-markers world: a real Labels ball layer, three side views
    that hold views of it."""
    from napari.layers import Labels

    install_fakes(monkeypatch)
    vol = np.zeros((20, 30, 30), np.int32)
    vol[5:9, 5:9, 5:9] = 7
    h = Harness(vol, spacing=(0.5, 0.5, 0.5))
    h.viewer.layers = _LogLayers(h)
    h.viewer_model_xy, h.viewer_model_xz, h.viewer_model_yz = (
        _VM(), _VM(), _VM())
    h.subviewer_manager = _SubMgr(h)
    h.drawing_handler = None
    for orient, vm in (("XY", h.viewer_model_xy), ("XZ", h.viewer_model_xz),
                       ("YZ", h.viewer_model_yz)):
        c = Labels(transpose_data(vol, orient), name=f"L_{orient}")
        vm.layers.append(c)
        h.subviewer_manager.copies[("L", orient)] = c
    h.adv.preset = "cells"
    h.place_box((slice(2, 18), slice(2, 28), slice(2, 28)))
    h._add_place_markers_on = True
    h._add_double_click_s = _no_double_click
    _wire_log(h)
    return h


def _no_double_click():
    return 0.0


class _LogLayers(_Layers):
    """The main viewer's layer list: an append is the ball layer's heavy
    call (the layer is built and drawn)."""

    def __init__(self, h):
        super().__init__()
        self.h = h

    def append(self, layer):
        log = getattr(self.h, "log", None)
        if log is not None:
            log.append(("heavy", "append"))
        if getattr(self.h, "raise_append", False):
            raise _Boom("append")
        super().append(layer)


A, B, C = (12, 20, 20), (12, 10, 20), (12, 20, 10)


# --------------------------------------------------------------------------- #
# the waits, each a function: (words, stage, run)
# --------------------------------------------------------------------------- #
def _wait_accept(monkeypatch, raising):
    h = _cells(monkeypatch)
    res = _whole_result(h)
    h.adv.checked = list(res.ids)

    def commit(ids, **_kw):
        h.log.append(("heavy", "commit"))

    h._on_add_committed_cb = commit
    if raising:
        def sync(**_k):
            raise _Boom("sync")
        h._sync_all_views_after_operation = sync
    del h.log[:]
    return h, AP.adding_line(len(res.ids)), "add_accept", "commit", \
        h._accept_add


def _wait_accept_box(monkeypatch, raising):
    """A box Accept: the write, the commit and the teardown freeze the app
    for seconds on a large box too, so it says so first (a still bar: no
    steps on a box)."""
    h = _cells(monkeypatch)
    res = _box_result(h)
    h.adv.checked = list(res.ids)

    def commit(ids, **_kw):
        h.log.append(("heavy", "commit"))

    h._on_add_committed_cb = commit
    if raising:
        def sync(**_k):
            raise _Boom("sync")
        h._sync_all_views_after_operation = sync
    del h.log[:]
    return h, AP.adding_line(len(res.ids)), "add_accept", "commit", \
        h._accept_add


class _Catch:
    def __init__(self):
        self.outcomes = []

    def __call__(self, outcome):
        self.outcomes.append(outcome)


def _finish_by_hand(h, *, whole):
    """Run the preview, catch the worker's answer, and hand it to the real
    finish handler by hand (an exception raised in a Qt slot would abort the
    process)."""
    if whole:
        h._start_add_whole()
        _land_info(h)
    else:
        h.place_box(BOX)
        _measure(h)
    catch = _Catch()
    h._on_add_finished = catch
    h._preview_add()
    th = h._add_thread
    th.run()
    del h._on_add_finished
    assert catch.outcomes
    return th, catch.outcomes[0]


def _wait_preview_end(monkeypatch, raising, whole=False):
    h = _cells(monkeypatch)
    th, outcome = _finish_by_hand(h, whole=whole)

    def render(result, roi=None, *a, **k):
        h.log.append(("heavy", "render"))
        if raising:
            raise _Boom("render")

    h._render_add_candidate = render
    del h.log[:]

    def run():
        try:
            h._on_add_finished(outcome)
        finally:
            th.end()

    return h, AP.ADD_S1_SHOWING, "add_draw", "render", run


def _clear_world(monkeypatch, raising, *, whole=False):
    from .test_add_clear import _harness as _clear_harness
    from .test_add_clear import _volume

    install_fakes(monkeypatch)
    vol = _volume()
    h = _clear_harness(vol)
    if whole:
        h.place_box(tuple(slice(0, n) for n in vol.shape))
        h._add_whole_patch = True
    _wire_log(h)

    def commit(ids, **kw):
        h.log.append(("heavy", "commit"))
        h.commits.append((list(ids), kw))
        if raising:
            raise _Boom("commit")

    h._on_discard_garbage_cb = commit

    def sync(**_k):
        h.log.append(("heavy", "sync"))

    h._sync_all_views_after_operation = sync
    return h


def _wait_clear(monkeypatch, raising, whole=False):
    h = _clear_world(monkeypatch, raising, whole=whole)
    mode = "patch" if whole else "box"

    def run():
        h._on_add_clear_requested(mode)
        th = FakeThread.instances[-1]
        th.run()
        th.end()

    # the box holds objects 3, 4 and 5; the patch those and 6 and 9
    n = 5 if whole else 3
    return h, AP.ADD_S1_CLEARING.format(n=n), "add_clear", "sync", run


def _wait_undo_all(monkeypatch, raising):
    h = _cells(monkeypatch)
    h._add_session = [{"ids": [10, 11], "sizes": {10: 5, 11: 6}},
                      {"ids": [20], "sizes": {20: 7}}]
    h._add_piece_ids = [10, 11, 20]
    h._add_review = True
    h._add_running = False

    def undo_session(n):
        h.log.append(("heavy", "undo"))
        if raising:
            raise _Boom("undo")
        return n

    h._on_add_undo_session_cb = undo_session
    del h.log[:]
    return h, AP.undoing_line(3), "add_undo_all", "undo", h._undo_add_all


def _spy_remove(h, raising):
    def remove():
        h.log.append(("heavy", "remove"))
        h._add_candidate_layer = None
        if raising:
            raise _Boom("remove")
    h._remove_add_candidate = remove


def _wait_discard(monkeypatch, raising):
    h = _cells(monkeypatch)
    _box_result(h)
    _spy_remove(h, raising)
    del h.log[:]
    return h, AP.ADD_S1_REMOVING_RESULT, "add_discard", "remove", \
        h._discard_add_candidates


def _spy_hide(h, raising):
    """A preset switch zeros and hides the drawn result (the layers are
    kept for the box): that is its heavy step now, not a removal."""
    def hide():
        h.log.append(("heavy", "hide"))
        if raising:
            raise _Boom("hide")
    h._add_hide_candidate = hide


def _wait_preset_switch(monkeypatch, raising):
    h = _cells(monkeypatch)
    _box_result(h)
    _spy_hide(h, raising)
    del h.log[:]

    def run():
        h.adv.preset = "nuclei"
        h._on_add_preset_changed("nuclei")

    return h, AP.ADD_S1_REMOVING_RESULT, "add_discard", "hide", run


def _wait_walls(monkeypatch, raising):
    h = _cells(monkeypatch)
    res = _box_result(h)
    assert res.elevation is not None
    h.raise_temp = raising
    del h.log[:]

    def run():
        h._on_add_show_walls(True)

    return h, AP.ADD_S1_DRAWING_WALLS, "add_walls", "temp_layer", run


def _wait_first_click(monkeypatch, raising):
    h = _clicks(monkeypatch)
    h.raise_append = raising
    del h.log[:]

    def run():
        h._add_marker_click(A)

    return h, AP.ADD_S1_DRAWING_MARKERS, "add_balls", "append", run


def _wait_gather(monkeypatch, raising):
    h = _cells(monkeypatch)
    h._start_add_whole()
    _land_info(h)
    real = h._add_gather_preview

    def gather():
        h.log.append(("heavy", "gather"))
        if raising:
            raise _Boom("gather")
        return real()

    h._add_gather_preview = gather
    del h.log[:]
    return h, AP.ADD_S1_GATHERING, "add_box_patch", "gather", h._preview_add


_WAITS = {
    "accept": _wait_accept,
    "accept_box": _wait_accept_box,
    "preview_end": _wait_preview_end,
    "clear": _wait_clear,
    "undo_all": _wait_undo_all,
    "discard": _wait_discard,
    "preset_switch": _wait_preset_switch,
    "walls": _wait_walls,
    "first_click": _wait_first_click,
    "gather": _wait_gather,
}

#: the heavy calls that raise straight out of the wait (the others are
#: caught inside it, which is the code's own guard)
_RAISES_OUT = {"accept", "accept_box", "preview_end", "discard",
               "preset_switch", "walls", "gather"}


# --------------------------------------------------------------------------- #
# painted before
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_WAITS))
def test_painted_before(monkeypatch, name):
    h, words, stage, heavy, run = _WAITS[name](monkeypatch, False)
    run()
    i = _first_paint(h, words)
    paint = h.log[i]
    assert paint[2] == stage_text(stage), "the grey line names the wait"
    assert paint[2] and paint[2] != stage, "a sentence, not a key"
    assert i < _at(h, ("heavy", heavy)), "painted only after the heavy call"


# --------------------------------------------------------------------------- #
# the steps of a whole patch, the still bar of one-step waits
# --------------------------------------------------------------------------- #
def test_steps_never_full(monkeypatch):
    # Accept: mask, write + stats, commit, teardown, sync, review
    h, words, _s, _heavy, run = _wait_accept(monkeypatch, False)
    run()
    steps = [(e[3], e[4]) for e in _paints(h) if e[1] == words]
    assert steps == [(k, 6) for k in range(6)]
    # each step is painted before its part runs: the commit is part 3
    assert _at(h, ("paint", words, stage_text("add_accept"), 2, 6)) < _at(
        h, ("heavy", "commit")) < _at(
        h, ("paint", words, stage_text("add_accept"), 3, 6))


def test_steps_never_full_preview_end(monkeypatch):
    h, words, _s, _heavy, run = _wait_preview_end(monkeypatch, False,
                                                  whole=True)
    run()
    steps = [(e[3], e[4]) for e in _paints(h) if e[1] == words]
    assert steps == [(0, 3), (1, 3)]
    # 0 before the render, 1 after it (before the list)
    assert _at(h, ("paint", words, stage_text("add_draw"), 0, 3)) < _at(
        h, ("heavy", "render")) < _at(
        h, ("paint", words, stage_text("add_draw"), 1, 3))


def test_steps_never_full_clear(monkeypatch):
    h, words, _s, _heavy, run = _wait_clear(monkeypatch, False, whole=True)
    run()
    steps = [(e[3], e[4]) for e in _paints(h) if e[1] == words]
    assert steps == [(0, 3), (1, 3), (2, 3)]
    line = stage_text("add_clear")
    assert _at(h, ("paint", words, line, 1, 3)) < _at(
        h, ("heavy", "sync")) < _at(h, ("paint", words, line, 2, 3)) < _at(
        h, ("heavy", "commit"))


@pytest.mark.parametrize("name", ["accept_box", "preview_end", "clear",
                                  "undo_all", "discard", "preset_switch",
                                  "walls", "first_click"])
def test_one_step_waits_leave_the_bar_still(monkeypatch, name):
    h, words, _s, _heavy, run = _WAITS[name](monkeypatch, False)
    run()
    for e in _paints(h):
        if e[1] == words:
            assert e[3] is None and e[4] is None, e


def test_the_panel_paint_leaves_the_bar_alone_without_steps(widget, qapp):
    reset_add(widget)
    widget.update_add_ui_state(busy=False, measuring="")
    bar = widget.add_busy_bar
    before = (bar.minimum(), bar.maximum(), bar.value())
    widget.paint_add_wait(status_text=AP.ADD_S1_REMOVING_RESULT,
                          stage_text=stage_text("add_discard"))
    assert (bar.minimum(), bar.maximum(), bar.value()) == before
    widget.paint_add_wait(status_text=AP.adding_line(3),
                          stage_text=stage_text("add_accept"), step=2, of=6)
    assert (bar.minimum(), bar.maximum(), bar.value()) == (0, 6, 2)
    assert bar.isEnabled()
    # never full while frozen, whatever the caller asks
    widget.paint_add_wait(status_text=AP.adding_line(3),
                          stage_text=stage_text("add_accept"), step=6, of=6)
    assert bar.value() < bar.maximum()
    widget.update_add_ui_state(busy=False, measuring="")
    assert bar.maximum() > 0 and bar.value() == 0 and not bar.isEnabled()


def test_the_panel_paint_sets_the_words_and_paints_now(widget, qapp,
                                                       monkeypatch):
    from qtpy.QtWidgets import QApplication

    reset_add(widget)
    widget.resize(340, 350)
    widget.show()
    QApplication.processEvents()
    widget.focus_tab("Add")
    QApplication.processEvents()
    calls = []

    def no_events(*_a, **_k):
        calls.append("processEvents")

    monkeypatch.setattr(QApplication, "processEvents", no_events)
    pushed = []
    real_update = widget.update_add_ui_state
    monkeypatch.setattr(widget, "update_add_ui_state",
                        lambda **kw: pushed.append(kw) or real_update(**kw))
    s1 = widget._add_s1_container
    s4 = widget._add_s4_container
    page = s1.parentWidget()
    for what, w in (("s1", s1), ("page", page), ("s4", s4)):
        lay = w.layout()
        monkeypatch.setattr(lay, "activate",
                            _rec(calls, what + ".activate", lay.activate))
    for what, w in (("page", page), ("s4", s4)):
        monkeypatch.setattr(w, "repaint",
                            _rec(calls, what + ".repaint", w.repaint))
    widget.paint_add_wait(status_text=AP.ADD_S1_REMOVING_RESULT,
                          stage_text=stage_text("add_discard"))
    assert widget.add_status_label.text() == AP.ADD_S1_REMOVING_RESULT
    assert widget._add_stage_full_text == stage_text("add_discard")
    assert widget.add_stage_label.toolTip() == stage_text("add_discard")
    assert widget.add_stage_label.isEnabled()
    assert "processEvents" not in calls, "never processEvents"
    assert pushed == [], "no full page update per paint"
    # S1's row, then the page that holds it, laid out; the page painted
    # (S1 and what moved below it); the strip laid out and painted
    assert calls.index("s1.activate") < calls.index("page.activate") < (
        calls.index("page.repaint"))
    assert calls.index("s4.activate") < calls.index("s4.repaint")
    widget.hide()


def test_a_longer_wait_line_is_painted_at_its_full_height(widget, qapp):
    """Words that need more lines than S1 had get them before the paint:
    the freeze lets no layout request through, so the page's layout is
    run by hand (the S1 row's own layout cannot make the row taller)."""
    from qtpy.QtWidgets import QApplication

    reset_add(widget)
    widget.resize(340, 700)
    widget.show()
    QApplication.processEvents()
    widget.focus_tab("Add")
    QApplication.processEvents()
    lb = widget.add_status_label
    long = " ".join([AP.ADD_S1_REMOVING_RESULT] * 6)
    widget.paint_add_wait(status_text=long,
                          stage_text=stage_text("add_discard"))
    need = lb.heightForWidth(lb.width())
    assert need > lb.fontMetrics().lineSpacing() * 2
    assert lb.height() >= need, "painted at the old height"
    widget.update_add_ui_state(busy=False, measuring="")
    widget.hide()


def _rec(calls, what, real):
    def f(*a, **k):
        calls.append(what)
        return real(*a, **k)
    return f


# --------------------------------------------------------------------------- #
# the measuring bar
# --------------------------------------------------------------------------- #
def test_measuring_bar_is_honest(monkeypatch):
    h = _cells(monkeypatch)
    h.place_box(BOX)
    h._request_add_box_info()
    assert h._add_info_thread is not None
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_S1_MEASURING
    assert h.adv.last("measuring") == stage_text("add_info")
    assert h.adv.last("busy") is False
    _land_info(h)
    h._add_refresh_ui()
    assert h._add_info_thread is None
    assert h.adv.last("measuring") == ""
    # S1 says measuring with no reader alive or due: no moving bar
    h._add_info = None
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_S1_MEASURING
    assert h.adv.last("measuring") == ""


def test_measuring_bar_stops_when_the_reader_ends(monkeypatch):
    """The answer lands while its thread is still alive: the thread's end
    puts the bar back."""
    h = _cells(monkeypatch)
    h.place_box(BOX)
    h._request_add_box_info()
    th = h._add_info_thread
    th.run()                              # the answer lands
    # S1 still says measuring (say the outline is still out) while the
    # thread is alive: the bar moves
    h._add_info = None
    h._add_refresh_ui()
    assert h.adv.last("measuring") == stage_text("add_info")
    th.end()                              # then the thread ends
    assert h._add_info_thread is None
    assert h.adv.last("measuring") == ""


def test_measuring_a_whole_patch_and_a_run_read(monkeypatch):
    h = _cells(monkeypatch)
    h._start_add_whole()
    assert h._add_info_thread is not None
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_S1_MEASURING_PATCH
    assert h.adv.last("measuring") == stage_text("add_info_patch")
    _land_info(h)
    # a run being read
    h._add_run_guide = None
    h._add_run_state = "loading"
    h._add_guide_thread = object()
    h._add_info = None
    h._add_info_thread = None
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_S1_RUN_LOADING
    assert h.adv.last("measuring") == stage_text("add_run")
    h._add_guide_thread = None


def test_measuring_stage_names_the_reader():
    assert AP.measuring_stage({"transient": AP.ADD_S1_MEASURING}) == "add_info"
    assert AP.measuring_stage(
        {"transient": AP.ADD_S1_MEASURING_PATCH}) == "add_info_patch"
    assert AP.measuring_stage(
        {"transient": AP.ADD_OUTLINE_PENDING}) == "add_outline"
    assert AP.measuring_stage(
        {"transient": AP.ADD_S1_RUN_LOADING}) == "add_run"
    assert AP.measuring_stage({"transient": AP.ADD_S1_TAKEN}) is None
    assert AP.measuring_stage({"has_patch": False}) is None


def test_the_panel_measuring_look(widget, qapp):
    reset_add(widget)
    # no box: no Clear objects, so no idle hint on the line
    widget.update_add_ui_state(busy=False, measuring="", box_active=False)
    widget._add_reject_wanted = True
    widget.update_add_ui_state(busy=False,
                               measuring=stage_text("add_info"))
    bar = widget.add_busy_bar
    assert bar.maximum() == 0 and bar.isEnabled(), "the moving bar"
    assert widget._add_stage_full_text == stage_text("add_info")
    assert widget.add_stage_label.isEnabled()
    assert not widget.add_stop_btn.isEnabled(), "Stop stays grey"
    assert widget._add_s2_container.isEnabled(), "the controls stay live"
    assert widget._add_s3_container.isEnabled()
    assert widget.add_reject_btn.isEnabled()
    # busy wins over measuring: the worker's own line
    widget.update_add_ui_state(busy=True, stage_text="Reading the box...",
                               measuring=stage_text("add_info"))
    assert widget._add_stage_full_text == "Reading the box..."
    assert not widget._add_s2_container.isEnabled()
    widget.update_add_ui_state(busy=False, measuring="")
    assert bar.maximum() > 0 and not bar.isEnabled()
    assert widget._add_stage_full_text == ""
    assert not widget.add_stage_label.isEnabled()


def test_the_idle_line_is_one_helpers_words(widget, qapp, monkeypatch):
    """The strip's idle line comes from one place, so a hint can live there
    and the measuring words take its place while they last."""
    reset_add(widget)
    # no box: no Clear objects, so no Clear hint on the line
    widget.update_add_ui_state(box_active=False)
    monkeypatch.setattr(widget, "_add_idle_stage_text", _hint)
    widget.update_add_ui_state(busy=False, measuring="")
    assert widget._add_stage_full_text == "a hint"
    widget.update_add_ui_state(busy=False, measuring=stage_text("add_info"))
    assert widget._add_stage_full_text == stage_text("add_info")
    widget.update_add_ui_state(busy=False, measuring="")
    assert widget._add_stage_full_text == "a hint"
    monkeypatch.undo()
    widget.update_add_ui_state(busy=False, measuring="")
    assert widget._add_stage_full_text == ""


def _hint():
    return "a hint"


# --------------------------------------------------------------------------- #
# the words never stick
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_WAITS))
@pytest.mark.parametrize("raising", [False, True])
def test_every_wait_ends_idle(monkeypatch, name, raising):
    h, words, _stage, heavy, run = _WAITS[name](monkeypatch, raising)
    if raising and name in _RAISES_OUT:
        with pytest.raises(_Boom):
            run()
    else:
        run()
    assert ("heavy", heavy) in h.log, "the heavy call ran"
    _ends_idle(h, words)
    assert not any(e[0] == "paint" for e in h.log[-1:]), "a paint came last"


def test_a_whole_patch_accept_ends_idle_when_the_commit_raises(monkeypatch):
    h, words, _stage, _heavy, run = _wait_accept(monkeypatch, False)

    def commit(ids, **_kw):
        h.log.append(("heavy", "commit"))
        raise _Boom("commit")

    h._on_add_committed_cb = commit
    run()
    _ends_idle(h, words)
    assert h.adv.last("status_text") == AP.ADD_S1_COMMIT_FAILED
    assert h.adv.last("busy") is False


def _boom(*_a, **_k):
    raise _Boom("a later step")


def test_a_whole_patch_accept_ends_idle_when_the_mask_raises(monkeypatch):
    """The span between the Accept's paint and its guarded write (the ticked
    ids picked out, the mask, the ids that land) raising: the words are
    replaced all the same."""
    h, _words, _stage, _heavy, run = _wait_accept(monkeypatch, False)
    ids = sorted(int(i) for i in h._add_preview_res.ids)
    assert len(ids) >= 2
    h.adv.checked = ids[:-1]            # some unticked: the ids are picked
    monkeypatch.setattr(np, "isin", _boom)
    with pytest.raises(_Boom):
        run()
    _ends_idle(h, AP.adding_line(len(ids) - 1))


def test_a_preset_switch_ends_idle_when_a_later_step_raises(monkeypatch):
    h, words, _stage, _heavy, run = _wait_preset_switch(monkeypatch, False)
    h._add_drop_info = _boom
    with pytest.raises(_Boom):
        run()
    _ends_idle(h, words)


def test_undo_all_ends_idle_when_a_later_step_raises(monkeypatch):
    h, words, _stage, _heavy, run = _wait_undo_all(monkeypatch, False)
    h._reset_add_state = _boom
    with pytest.raises(_Boom):
        run()
    _ends_idle(h, words)


@pytest.mark.parametrize("whole", [False, True])
def test_stop_after_the_answer_landed_changes_nothing(monkeypatch, whole):
    """The worker's answer is on screen while its thread is still quitting:
    Stop is grey, and a Stop click queued during the main-thread wait
    neither says Stopped nor drops the candidates."""
    h = _cells(monkeypatch)
    th, outcome = _finish_by_hand(h, whole=whole)
    h._add_candidate_layer = _Layer()
    h._render_add_candidate = _no_render
    h._on_add_finished(outcome)
    assert h._add_preview_result is not None
    last = [e[1] for e in h.log if e[0] == "push" and "busy" in e[1]][-1]
    assert not last.get("stop_enabled"), last.get("stop_enabled")
    h._stop_add_preview()
    th.end()
    assert h._add_transient not in (AP.ADD_S1_STOPPED,
                                    AP.ADD_S1_STOPPED_PATCH), h._add_transient
    assert h._add_preview_result is not None, "the candidates stay"
    last = [e[1] for e in h.log if e[0] == "push" and "busy" in e[1]][-1]
    assert not last.get("stop_enabled")
    assert last.get("status_text") not in (AP.ADD_S1_STOPPED,
                                           AP.ADD_S1_STOPPED_PATCH)


def _no_render(*_a, **_k):
    return None


def _ball_ops():
    def click(h):
        h._add_marker_click(B)

    def ctrl_click(h):
        h._add_marker_click(A, ctrl=True)

    def undo_last(h):
        h._add_click_undo()

    def remove_all(h):
        h._add_click_clear()

    def tick(h):
        h._on_add_click_tick(int(h._add_click_markers[0]["id"]), False)

    def box_move_drop(h):
        h._on_add_box_changed(12, 20, 12, 10, 10, 10)

    def rebase(h):
        from napari.layers import Image

        h._get_label_id_floor_cb = _floor_30
        h._add_info = AP.AddBoxInfo(empty_share=1.0)
        h.adv.params["dark_outside"] = False
        h.adv.wall["channel"] = "mem"
        h.viewer.layers.append(Image(np.random.rand(20, 30, 30),
                                     name="mem"))
        del h.log[:]
        h._add_gather_preview()

    def radius(h):
        # a Ball change: the redraw waits for the arrow steps to stop (a
        # timer, fired here by hand)
        import qtpy.QtCore

        from .test_add_page_usability import _FakeTimer

        real = qtpy.QtCore.QTimer
        qtpy.QtCore.QTimer = _FakeTimer
        try:
            h.adv.params["click_radius_um"] = 3.0
            h._on_add_advanced_changed()
        finally:
            qtpy.QtCore.QTimer = real
        t = h._add_ball_timer
        assert isinstance(t, _FakeTimer) and t.active
        t.fire()

    return {"click": click, "ctrl_click": ctrl_click, "undo_last": undo_last,
            "remove_all": remove_all, "tick": tick,
            "box_move_drop": box_move_drop, "rebase": rebase,
            "radius": radius}


def _floor_30():
    return 30


@pytest.mark.parametrize("op", sorted(_ball_ops()))
def test_ball_callers_end_idle(monkeypatch, op):
    h = _clicks(monkeypatch)
    h._add_marker_click(A)
    h._add_marker_click(C)
    del h.log[:]
    _ball_ops()[op](h)
    pushes = [e[1] for e in h.log if e[0] == "push" and "busy" in e[1]]
    assert pushes, "no refresh after the balls were drawn"
    assert pushes[-1]["busy"] is False
    last_paint = max((i for i, e in enumerate(h.log) if e[0] == "paint"),
                     default=-1)
    last_push = max(i for i, e in enumerate(h.log)
                    if e[0] == "push" and "busy" in e[1])
    assert last_push > last_paint


def test_box_paths_paint_as_before(monkeypatch):
    # a box Preview end: one paint
    h, words, _s, _heavy, run = _wait_preview_end(monkeypatch, False)
    run()
    assert [e[1] for e in _paints(h)] == [words]
    # a box Clear: one paint
    h, words, _s, _heavy, run = _wait_clear(monkeypatch, False)
    run()
    assert [e[1] for e in _paints(h)] == [words]
    # the first click paints, the later ones never
    h = _clicks(monkeypatch)
    h._add_marker_click(A)
    assert [e[1] for e in _paints(h)] == [AP.ADD_S1_DRAWING_MARKERS]
    del h.log[:]
    h._add_marker_click(B)
    h._add_marker_click(C)
    assert _paints(h) == []


# --------------------------------------------------------------------------- #
# the ridge notes
# --------------------------------------------------------------------------- #
def _ridge(h, kind="frangi"):
    h.adv.wall = dict(h.adv.wall, wall_kind=kind)


def _whole_n(h):
    return int(np.prod([sl.stop - sl.start for sl in h._add_roi]))


def test_ridge_notes(monkeypatch):
    h = _cells(monkeypatch)
    # a whole patch, a ridge wall type, the picture not kept yet
    h._start_add_whole()
    n = _whole_n(h)
    _ridge(h, "frangi")
    assert h._add_derive_note_text() == AP.ridges_whole_line(
        "Vessel ridges", n, "frangi")
    _ridge(h, "meijering")
    assert h._add_derive_note_text() == AP.ridges_whole_line(
        "Thin-faint ridges", n, "meijering")
    # kept for this key: no note
    h._add_elev_cache = {"key": h._add_elev_key(), "elev": np.zeros(1)}
    assert h._add_derive_note_text() == ""
    # kept for another key (another wall type): the note again
    _ridge(h, "frangi")
    assert h._add_derive_note_text() == AP.ridges_whole_line(
        "Vessel ridges", n, "frangi")
    # not a ridge type: no note
    _ridge(h, "membrane")
    assert h._add_derive_note_text() == ""
    # over the fill's ceiling S1 refuses: no note
    _ridge(h, "frangi")
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", 10)
    assert h._add_derive_note_text() == ""
    monkeypatch.undo()


def test_ridge_notes_on_a_box(monkeypatch):
    h = _cells(monkeypatch)
    h.place_box(BOX)
    n = int(np.prod([sl.stop - sl.start for sl in BOX]))
    _ridge(h, "frangi")
    assert AP.ADD_RIDGE_BOX_VOXELS == 2_000_000
    assert n < AP.ADD_RIDGE_BOX_VOXELS
    assert h._add_derive_note_text() == ""
    monkeypatch.setattr(AP, "ADD_RIDGE_BOX_VOXELS", n)
    # at or under the keep limit: built once, then kept for the box --
    # said until it is kept
    assert n <= AP.ADD_LIVE_CELLS_MAX_VOXELS
    once = AP.ADD_NOTE_RIDGES_BOX_ONCE.format(kind="Vessel ridges")
    assert h._add_derive_note_text() == once
    # a kept whole-patch picture changes nothing on a box (another key)
    h._add_elev_cache = {"key": h._add_elev_key(), "elev": np.zeros(1)}
    assert h._add_derive_note_text() == once
    # the box's own picture kept for this key: no note
    h._add_box_elev_cache = {"key": h._add_box_elev_key(),
                             "elev": np.zeros(1)}
    assert h._add_derive_note_text() == ""
    # kept for another wall type: the note again
    _ridge(h, "meijering")
    assert h._add_derive_note_text() == AP.ADD_NOTE_RIDGES_BOX_ONCE.format(
        kind="Thin-faint ridges")
    _ridge(h, "frangi")
    # over the keep limit (the size the box's cache keys on): built again
    # at every Preview, kept picture or not
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_MAX_VOXELS", n - 1)
    assert h._add_box_elev_key() is None
    assert h._add_derive_note_text() == AP.ADD_NOTE_RIDGES_BOX.format(
        kind="Vessel ridges")
    _ridge(h, "gradient")
    assert h._add_derive_note_text() == ""
    # over a result: the amber slot is empty
    _ridge(h, "frangi")
    h._add_preview_result = np.zeros((1, 1, 1), np.int32)
    assert h._add_derive_note_text() == ""


def test_ridge_note_priority(monkeypatch):
    h = _cells(monkeypatch)
    h._start_add_whole()
    _ridge(h, "frangi")
    ridge = AP.ridges_whole_line("Vessel ridges", _whole_n(h), "frangi")
    # before the slow patch
    monkeypatch.setattr(AP, "ADD_SLOW_PATCH_VOXELS", 10)
    assert h._add_derive_note_text() == ridge
    # after the outline drops
    h.adv.params["outline"] = "shrink"
    h._add_info = AP.AddBoxInfo(empty_share=1.0, run_count=(10, 10),
                                n_outside_outline=9)
    drops = h._add_cells_note_text()
    assert drops != ridge and drops
    h.adv.params["outline"] = "none"
    h._add_info = None
    # after one channel
    lay = h.viewer.layers.pop()
    assert h._add_cells_note_text() != ridge
    h.viewer.layers.append(lay)
    assert h._add_cells_note_text() == ridge
    # after a half-size session
    h.ctx["downsample_factor"] = 2
    assert h._add_cells_note_text() == AP.ADD_NOTE_HALFSIZE_BOXED


def test_the_ridge_notes_say_no_engine_word():
    for text in (AP.ADD_NOTE_RIDGES_WHOLE, AP.ADD_NOTE_RIDGES_WHOLE_MEMORY,
                 AP.ADD_NOTE_RIDGES_BOX, AP.ADD_NOTE_RIDGES_BOX_ONCE):
        for word in ("frangi", "meijering", "voxel", "elevation"):
            assert word not in text.lower()
        assert "{kind}" in text
        assert "Membrane sheets" in text


# --------------------------------------------------------------------------- #
# how long a whole patch takes, in the notes' words
# --------------------------------------------------------------------------- #
#: measured whole-patch Cells fills with Membrane sheets, in minutes
_SHEETS_MEASURED = {6: 28 / 60, 24: 2.2, 48: 5.1, 96: 12.8}
#: measured ridge wall pictures, in minutes
_PICTURE_MEASURED = {6: 1.0, 12: 2.0, 24: 4.2}


@pytest.mark.parametrize("m", sorted(_SHEETS_MEASURED))
def test_the_sheets_wait_follows_the_measurements(m):
    got = AP.whole_fill_minutes(m, "membrane")
    assert abs(got - _SHEETS_MEASURED[m]) < 0.5, (m, got)
    # every wall type but the two ridges waits like Membrane sheets
    for kind in (None, "", "distance", "hybrid", "gradient", "intensity"):
        assert AP.whole_fill_minutes(m, kind) == got


@pytest.mark.parametrize("m", sorted(_PICTURE_MEASURED))
def test_the_ridge_picture_follows_the_measurements(m):
    got = AP.ridge_picture_minutes(m)
    assert abs(got - _PICTURE_MEASURED[m]) < 0.5, (m, got)


@pytest.mark.parametrize("kind", ["frangi", "meijering"])
def test_a_ridge_wait_adds_its_wall_picture(kind):
    # measured: 6.1 minutes in all at 24 million voxels
    assert abs(AP.whole_fill_minutes(24, kind) - 6.1) < 0.5
    for m in (6, 24, 96):
        assert AP.whole_fill_minutes(m, kind) == pytest.approx(
            AP.whole_fill_minutes(m, "membrane") + AP.ridge_picture_minutes(m))
        # a picture already kept is not built again
        assert AP.whole_fill_minutes(m, kind, picture_kept=True) == (
            AP.whole_fill_minutes(m, "membrane"))


def test_the_wait_words_round_to_whole_minutes():
    for minutes in (0.0, 0.48, 0.74):
        assert AP.wait_words(minutes) == "under a minute"
    for minutes in (0.75, 1.0, 1.49):
        assert AP.wait_words(minutes) == "a minute"
    assert AP.wait_words(1.5) == "2 minutes"
    assert AP.wait_words(2.18) == "2 minutes"
    assert AP.wait_words(2.5) == "3 minutes"
    assert AP.wait_words(12.74) == "13 minutes"


def test_the_ridge_memory_follows_the_measurement():
    # measured: 4.4 GB at 24 million voxels
    assert abs(AP.ridge_memory_gb(24) - 4.4) < 0.6
    assert 18 <= AP.ridge_memory_gb(100) <= 21


def test_the_slow_patch_note_says_how_long():
    sheets = AP.patch_slow_line(24_000_000, "membrane")
    assert sheets == (
        "This patch is 24.0 million voxels; a Preview may take about 2 "
        "minutes, longer on a slower computer. Stop acts between steps.")
    ridges = AP.patch_slow_line(24_000_000, "frangi")
    assert "about 6 minutes" in ridges
    assert ridges != sheets
    # no wall type (the Nuclei preset): the Membrane sheets figure
    assert AP.patch_slow_line(24_000_000) == sheets
    assert "about 5 minutes" in AP.patch_slow_line(48_000_000, "membrane")
    assert "about 13 minutes" in AP.patch_slow_line(96_000_000, "membrane")
    # a ridge picture already kept: as fast as Membrane sheets
    assert AP.patch_slow_line(24_000_000, "frangi", picture_kept=True) == (
        sheets)
    assert "a few minutes" not in AP.ADD_NOTE_PATCH_SLOW_STEPS
    assert "{t}" in AP.ADD_NOTE_PATCH_SLOW_STEPS


def test_the_ridge_note_says_how_long_and_how_much_memory():
    # the WHOLE Preview's wait, the wall picture included
    six = AP.ridges_whole_line("Vessel ridges", 6_000_000, "frangi")
    assert six == (
        "Vessel ridges is slow on a whole patch: a Preview takes about a "
        "minute here, most of it building the wall picture. Membrane sheets "
        "is much faster.")
    t24 = AP.ridges_whole_line("Vessel ridges", 24_000_000, "frangi")
    assert t24 == (
        "Vessel ridges is slow on a whole patch: a Preview takes about 6 "
        "minutes here, most of it building the wall picture. Membrane "
        "sheets is much faster.")
    assert "memory" not in t24 and "GB" not in t24
    t96 = AP.ridges_whole_line("Thin-faint ridges", 96_000_000, "meijering")
    assert t96 == (
        "Thin-faint ridges is slow on a whole patch: a Preview takes about "
        "29 minutes here, most of it building the wall picture, and needs "
        "about 19 GB of memory. Membrane sheets is much faster.")
    # the figure is the ridge total, never the picture alone
    for n, kind in ((24_000_000, "frangi"), (73_000_000, "meijering")):
        want = AP.wait_words(AP.whole_fill_minutes(n / 1e6, kind))
        assert "about {} here".format(want) in AP.ridges_whole_line(
            "Vessel ridges", n, kind)
    # the memory words from 8 GB on
    assert "memory" not in AP.ridges_whole_line("Vessel ridges", 39_000_000,
                                                "frangi")
    assert "about 8 GB" in AP.ridges_whole_line("Vessel ridges", 40_000_000,
                                                "frangi")
    for tpl in (AP.ADD_NOTE_RIDGES_WHOLE, AP.ADD_NOTE_RIDGES_WHOLE_MEMORY):
        assert "a minute or more" not in tpl
        assert "longer on a big patch" not in tpl
        assert "to build here" not in tpl
        assert "{t}" in tpl


@pytest.mark.parametrize("n, want", [(500_000, "under a minute"),
                                     (3_000_000, "under a minute")])
def test_a_small_ridge_patch_says_under_a_minute(n, want):
    # 0.5 M: about 0.1 minutes in all; 3 M: about 0.74 minutes in all
    line = AP.ridges_whole_line("Vessel ridges", n, "frangi")
    assert "a Preview takes about {} here".format(want) not in line
    assert "a Preview takes {} here".format(want) in line


def test_the_ridge_note_reads_the_patch_and_the_wall_type(monkeypatch):
    h = _cells(monkeypatch)
    h._start_add_whole()
    n = int(np.prod(h.sync_manager.shared_labels["L"].shape))
    assert n == _whole_n(h)
    seen = []
    real = AP.ridges_whole_line

    def _spy(kind, n_, wall_kind):
        seen.append((kind, n_, wall_kind))
        return real(kind, n_, wall_kind)

    monkeypatch.setattr(AP, "ridges_whole_line", _spy)
    _ridge(h, "frangi")
    assert h._add_derive_note_text() == real("Vessel ridges", n, "frangi")
    assert seen[-1] == ("Vessel ridges", n, "frangi")
    _ridge(h, "meijering")
    assert h._add_derive_note_text() == real("Thin-faint ridges", n,
                                             "meijering")
    assert seen[-1] == ("Thin-faint ridges", n, "meijering")


def test_the_slow_patch_note_reads_the_wall_type(monkeypatch):
    h = _cells(monkeypatch)
    h._start_add_whole()
    n = _whole_n(h)
    monkeypatch.setattr(AP, "ADD_SLOW_PATCH_VOXELS", 10)
    seen = []
    real = AP.patch_slow_line

    def _spy(n_, wall_kind=None, *, picture_kept=False, find_nuclei=False,
             growth=None):
        seen.append((n_, wall_kind, picture_kept, find_nuclei, growth))
        return real(n_, wall_kind, picture_kept=picture_kept,
                    find_nuclei=find_nuclei, growth=growth)

    monkeypatch.setattr(AP, "patch_slow_line", _spy)
    _ridge(h, "membrane")
    assert h._add_derive_note_text() == real(n, "membrane")
    assert seen[-1] == (n, "membrane", False, False, None)
    # a ridge type whose picture is kept: the ridge note is gone, and the
    # slow-patch note counts no picture to build
    _ridge(h, "frangi")
    h._add_elev_cache = {"key": h._add_elev_key(), "elev": np.zeros(1)}
    assert h._add_derive_note_text() == real(n, "frangi", picture_kept=True)
    assert seen[-1] == (n, "frangi", True, False, None)


def _box_of_millions(z):
    """A box of ``z`` x 100 x 1000 voxels (``z / 10`` million); only its
    size is read here."""
    return (slice(0, int(z)), slice(0, 100), slice(0, 1000))


def test_a_large_cells_box_says_its_preview_is_slow(monkeypatch):
    """A box follows the whole patch's size rule: over 20 million voxels a
    Preview fills it in one go, and the amber slot says so before it is
    pressed, as on a whole patch."""
    h = _cells(monkeypatch)
    _ridge(h, "membrane")
    assert AP.ADD_SLOW_PATCH_VOXELS == 20_000_000
    h.place_box(_box_of_millions(250))                  # 25.0 million
    line = h._add_derive_note_text()
    # a box grows by Grow by's pick: the flood's figure (Spread has its own)
    assert line == AP.patch_slow_line(25_000_000, "membrane", box=True,
                                      growth=AP.GROW_FLOOD)
    assert line == (
        "This box is 25.0 million voxels; a Preview may take about a "
        "minute, longer on a slower computer. Stop acts between steps.")
    # one family with the whole patch's line, the region the only difference
    assert AP.ADD_NOTE_BOX_SLOW_STEPS == (
        AP.ADD_NOTE_PATCH_SLOW_STEPS.replace("This patch", "This box"))
    # a 5 million box: no slow line
    h.place_box(_box_of_millions(50))
    assert h._add_derive_note_text() == ""
    # at the slow size exactly: none either
    h.place_box(_box_of_millions(200))
    assert h._add_derive_note_text() == ""
    # over the Preview limit S1 refuses the box: no slow line
    h.place_box(_box_of_millions(1500))
    assert h._add_derive_note_text() == ""
    # a ridge wall type: its own note comes first
    h.place_box(_box_of_millions(250))
    _ridge(h, "frangi")
    assert h._add_derive_note_text() == AP.ADD_NOTE_RIDGES_BOX.format(
        kind="Vessel ridges")


@pytest.mark.parametrize("n", [24_000_000, 500_000_000])
def test_the_nuclei_slow_patch_note_gives_no_figure(n):
    # the Nuclei whole patch was never timed and has no size ceiling: its
    # note names no wait in numbers
    m = AP.millions(n)
    line = AP.nuclei_patch_slow_line(n)
    assert line == (
        "This patch is {} million voxels; a Preview may take several "
        "minutes or longer. Stop acts between steps.".format(m))
    assert not re.search(r"\d", line.replace(m, "", 1))
    # the Cells note still carries its computed minutes
    assert re.search(r"about \d+ minutes", AP.patch_slow_line(n, "membrane"))


# --------------------------------------------------------------------------- #
# the words
# --------------------------------------------------------------------------- #
def test_the_wait_words():
    assert AP.ADD_S1_UNDOING_ALL == "Taking back {n} objects…"
    assert AP.ADD_S1_UNDOING_ALL_ONE == "Taking back 1 object…"
    assert AP.undoing_line(1) == AP.ADD_S1_UNDOING_ALL_ONE
    assert AP.undoing_line(5) == "Taking back 5 objects…"
    assert AP.ADD_S1_REMOVING_RESULT == "Taking the result off the screen…"
    assert AP.ADD_S1_DRAWING_WALLS == "Drawing the wall picture…"
    assert AP.ADD_S1_DRAWING_MARKERS == "Drawing the markers…"
    for text in (AP.ADD_S1_UNDOING_ALL, AP.ADD_S1_REMOVING_RESULT,
                 AP.ADD_S1_DRAWING_WALLS, AP.ADD_S1_DRAWING_MARKERS):
        assert "voxel" not in text


def test_the_new_stage_words():
    want = {
        "add_accept": "Writing the new objects...",
        "add_draw": "Drawing the new objects...",
        "add_clear": "Clearing the objects...",
        "add_undo_all": "Taking the objects back...",
        "add_discard": "Taking the result off the screen...",
        "add_walls": "Drawing the wall picture...",
        "add_balls": "Drawing the markers...",
        "add_info_patch": "Measuring the patch...",
        "add_run": "Reading the run's objects...",
    }
    for key, words in want.items():
        assert stage_text(key) == words, key


# --------------------------------------------------------------------------- #
# the live driver's four gaps
# --------------------------------------------------------------------------- #
def test_a_box_accept_paints_its_words_before_the_write(monkeypatch):
    """A box Accept froze the app for seconds with no words: it says it is
    adding before the write, with a still bar, and the words go after."""
    h, words, stage, heavy, run = _wait_accept_box(monkeypatch, False)
    run()
    paints = [e for e in _paints(h) if e[1] == words]
    assert paints, "a box Accept painted nothing"
    assert paints[0][2] == stage_text(stage)
    assert all(e[3] is None and e[4] is None for e in paints), "a still bar"
    assert _first_paint(h, words) < _at(h, ("heavy", heavy))
    _ends_idle(h, words)


def test_a_box_accept_ends_idle_when_the_mask_raises(monkeypatch):
    h, _words, _stage, _heavy, run = _wait_accept_box(monkeypatch, False)
    ids = sorted(int(i) for i in h._add_preview_res.ids)
    assert len(ids) >= 2
    h.adv.checked = ids[:-1]            # some unticked: the ids are picked
    monkeypatch.setattr(np, "isin", _boom)
    with pytest.raises(_Boom):
        run()
    words = AP.adding_line(len(ids) - 1)
    assert _first_paint(h, words) >= 0
    _ends_idle(h, words)


@pytest.mark.parametrize("whole", [False, True])
def test_the_preview_end_paint_asks_stop_grey(monkeypatch, whole):
    h, words, _s, _heavy, run = _wait_preview_end(monkeypatch, False,
                                                  whole=whole)
    del h.paint_stops[:]                  # in step with the cleared log
    run()
    i = [k for k, e in enumerate(_paints(h)) if e[1] == words]
    assert i, "never painted"
    assert h.paint_stops[i[0]] is False, "Stop is drawn live in the freeze"


@pytest.mark.parametrize("whole", [False, True])
def test_stop_is_grey_in_the_painted_frame_of_the_preview_end(
        monkeypatch, widget, qapp, whole):
    """On the real page: at the instant the strip is painted for "Showing
    the result…", Stop is grey (its row is in the painted strip)."""
    reset_add(widget)
    h, words, _s, _heavy, run = _wait_preview_end(monkeypatch, False,
                                                  whole=whole)
    # the preview ran: the page's Stop is live
    widget.update_add_ui_state(busy=True, stop_enabled=True)
    assert widget.add_stop_btn.isEnabled()
    assert widget._add_s4_container.isAncestorOf(widget.add_stop_btn)
    seen = []
    s4 = widget._add_s4_container
    real = s4.repaint

    def rec(*a, **k):
        seen.append((widget.add_status_label.text(),
                     widget.add_stop_btn.isEnabled()))
        return real(*a, **k)

    monkeypatch.setattr(s4, "repaint", rec)
    h.adv.paint_add_wait = widget.paint_add_wait
    run()
    painted = [on for text, on in seen if text == words]
    assert painted, "the strip was never painted for the wait"
    assert not any(painted), "Stop drawn live in the frozen frame"
    widget.update_add_ui_state(busy=False, measuring="")


def test_a_new_preview_lights_stop_again(widget, qapp):
    reset_add(widget)
    widget.paint_add_wait(status_text=AP.ADD_S1_SHOWING,
                          stage_text=stage_text("add_draw"),
                          stop_enabled=False)
    assert not widget.add_stop_btn.isEnabled()
    widget.update_add_ui_state(busy=True, stop_enabled=True)
    assert widget.add_stop_btn.isEnabled()
    # a paint that says nothing of Stop leaves it as it is
    widget.paint_add_wait(status_text=AP.ADD_S1_GATHERING,
                          stage_text=stage_text("add_box_patch"))
    assert widget.add_stop_btn.isEnabled()
    widget.update_add_ui_state(busy=False, measuring="")
    assert not widget.add_stop_btn.isEnabled()


def test_a_wall_type_change_shows_the_measuring_words(monkeypatch):
    """Whole patch, run markers, Wall type -> Vessel ridges: the patch is
    measured again on the worker; the strip says so and its bar moves for
    as long as the reader runs, then the idle line returns."""
    import qtpy.QtCore

    from .test_add_page_usability import _FakeTimer

    h = _cells(monkeypatch)
    h._start_add_whole()
    _land_info(h)
    h._add_refresh_ui()
    assert h.adv.last("measuring") == ""
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    # the real debounce (the harness stubs it)
    from .add_harness import C

    h._schedule_add_histogram = C._schedule_add_histogram.__get__(h)
    _ridge(h, "frangi")
    h._on_add_advanced_changed()
    t = h._add_box_timer
    assert isinstance(t, _FakeTimer) and t.active
    t.fire()                                # the debounced request
    assert h._add_info_thread is not None, "the reader runs"
    assert h.adv.last("status_text") == AP.ADD_S1_MEASURING_PATCH
    assert h.adv.last("measuring") == stage_text("add_info_patch")
    _land_info(h)
    assert h._add_info_thread is None
    assert h.adv.last("measuring") == ""


def test_a_clear_measures_the_box_again_with_words(monkeypatch):
    """A box Clear asks the box info again: a refresh comes after the reader
    started, so the strip names it while S1 says what was cleared."""
    h, _words, _s, _heavy, run = _wait_clear(monkeypatch, False)
    run()
    assert h._add_info_thread is not None, "the box is measured again"
    assert h.adv.last("measuring") == stage_text("add_info")


def test_the_measuring_words_follow_the_reader_not_s1(monkeypatch):
    """Click markers, none placed: S1 asks for a marker (it outranks the
    measuring sentence), and the strip still says what is being read."""
    h = _clicks(monkeypatch)
    h.adv.wall["channel"] = "mem"
    h._add_info_thread = None
    h._add_guide_thread = None
    h._add_refresh_ui()
    s1 = h.adv.last("status_text")
    assert s1 in (AP.ADD_S1_NO_MARKERS, AP.ADD_S1_NO_MARKERS_ON), s1
    assert h.adv.last("measuring") == ""
    h._add_info_thread = object()           # a box-info read is running
    try:
        h._add_refresh_ui()
        assert h.adv.last("status_text") == s1, "S1 is not changed"
        assert h.adv.last("measuring") == stage_text("add_info")
    finally:
        h._add_info_thread = None
    h._add_guide_thread = object()          # a run read is running
    try:
        h._add_refresh_ui()
        assert h.adv.last("status_text") == s1
        assert h.adv.last("measuring") == stage_text("add_run")
    finally:
        h._add_guide_thread = None
    h._add_refresh_ui()
    assert h.adv.last("measuring") == ""


def test_measuring_stage_follows_the_reader():
    no_marker = {"has_box": True, "preset": "cells", "wall_channel": "mem",
                 "source": AP.SOURCE_CLICKS, "n_clicks": 0}
    assert AP.measuring_stage(no_marker) is None
    assert AP.measuring_stage(no_marker, reader="info") == "add_info"
    # a whole patch grows from another run's objects (clicks place nothing
    # there, and that whole patch is measured by nothing)
    whole = dict(no_marker, whole_patch=True, box_shape=(4, 4, 4),
                 source=AP.SOURCE_RUN, run_id="r1")
    assert AP.s1_sentence(whole) != AP.ADD_S1_MEASURING_PATCH
    assert AP.measuring_stage(whole, reader="info") == "add_info_patch"
    assert AP.measuring_stage(dict(no_marker, outline_pending=True),
                              reader="info") == "add_outline"
    assert AP.measuring_stage(no_marker, reader="run") == "add_run"
    # S1 naming a read still names it
    assert AP.measuring_stage({"transient": AP.ADD_S1_MEASURING},
                              reader="run") == "add_info"


# --------------------------------------------------------------------------- #
# review findings on the reader's words
# --------------------------------------------------------------------------- #
def test_no_reader_words_in_the_review_after_close_box(monkeypatch):
    """"+ Add another", a run read in flight, Close box: the review shows no
    reader's words and no moving bar, and still none once the read lands
    (a read landing with no box up refreshes nothing)."""
    h = _cells(monkeypatch)
    res = _box_result(h)
    h.adv.checked = list(res.ids)
    h._on_add_committed_cb = _no_render
    h._accept_add()
    h._add_another()
    h._add_guide_thread = object()           # Markers ▸ a run being read
    h._add_guide_worker = None
    h._add_secondary()                       # Close box: back to the review
    assert not h._add_running
    assert h.adv.last("measuring") == ""
    assert h.adv.last("busy") is False
    h._add_guide_thread = None
    h.sender = None
    h._on_add_guide_loaded(np.zeros((2, 2, 2), np.int32))
    assert h.adv.last("measuring") == ""


def test_a_whole_patch_refused_for_measuring_names_no_reader():
    """A Cells whole patch with a blank From is measured by nothing (the
    Auto lines say so): the strip names no read.  The nuclei masks and
    Nuclei ▸ Anywhere are measured on a whole patch as on a box."""
    base = {"has_box": True, "whole_patch": True, "box_shape": (4, 4, 4)}
    cells = dict(base, preset="cells", wall_channel="mem",
                 nuclei_channel="nuc", source=AP.SOURCE_RUN, run_id=None,
                 n_clicks=0)
    assert AP.preview_block(cells) == AP.ADD_S1_WHOLE_NO_RUN
    assert AP.measuring_stage(cells, reader="info") is None
    picked = dict(cells, has_other_run=True)
    assert AP.preview_block(picked) == AP.ADD_S1_WHOLE_PICK_RUN
    assert AP.measuring_stage(picked, reader="info") is None
    masks = dict(cells, source=AP.SOURCE_NUCLEI, info_pending=True)
    assert AP.measuring_stage(masks, reader="info") == "add_info_patch"
    anywhere = dict(base, preset="nuclei", channel="nuc",
                    look_for=AP.LOOK_ANYWHERE)
    assert AP.preview_block(anywhere) is None
    assert AP.measuring_stage(anywhere, reader="info") == "add_info_patch"
    # a whole patch that is measured still names its read: another run's
    # objects, Click markers (a marker to place outranks the measuring
    # line) and the Nuclei markers
    run = dict(base, preset="cells", wall_channel="mem",
               source=AP.SOURCE_RUN, run_id="r1", info_pending=True)
    assert AP.measuring_stage(run, reader="info") == "add_info_patch"
    clicks = dict(base, preset="cells", wall_channel="mem",
                  source=AP.SOURCE_CLICKS, n_clicks=0)
    assert AP.preview_block(clicks) == AP.ADD_S1_NO_MARKERS
    assert AP.measuring_stage(clicks, reader="info") == "add_info_patch"
    markers = dict(base, preset="nuclei", channel="nuc",
                   look_for=AP.LOOK_MARKERS, n_clicks=0)
    assert AP.preview_block(markers) == AP.ADD_S1_NO_MARKERS
    assert AP.measuring_stage(markers, reader="info") == "add_info_patch"

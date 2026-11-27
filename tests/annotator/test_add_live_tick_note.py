"""The reason on the tick's own row: a grey line under "Update as I click".

Cells ▸ Click markers with "Update as I click" on, where a click cannot be
run as you click here (S1 row 13.5: over the live limit; walls or a Grow by
the quick fill does not redo; a step aside for flat spots, at the default
spacing or not; the fill still being prepared), a short grey line directly
under the tick says why, in at most 60 characters, naming the control that
changes it by its word on the page.  It is decided by the same state keys,
through one function beside S1's, so the tick row and S1 cannot disagree.
Hidden while clicks run live, with the tick off, and always while "Always
update as I click" is on.
"""

from __future__ import annotations

import itertools

import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_cells_box import CELL_A, _fire, _finish
from .test_add_live_cells_incremental import ci  # noqa: F401

MILLION = 1_000_000
#: 5.0 and 25.0 million voxels
FIVE = (50, 100, 1000)
TWENTY_FIVE = (250, 100, 1000)


def _state(shape=FIVE, **kw):
    s = {"has_patch": True, "has_box": True, "preset": "cells",
         "source": AP.SOURCE_CLICKS, "wall_channel": "mem",
         "box_shape": shape, "box_budget": AP.ADD_BOX_BUDGET,
         "whole_patch": False, "n_clicks": 1, "n_ticked": 1,
         "place_markers_on": True, "live_on": True, "empty_share": 1.0,
         "live_whole_ok": True}
    s.update(kw)
    return s


NOTES = ("ADD_LIVE_NOTE_OVER", "ADD_LIVE_NOTE_WALLS", "ADD_LIVE_NOTE_GROW",
         "ADD_LIVE_NOTE_FLAT", "ADD_LIVE_NOTE_SPACING", "ADD_LIVE_NOTE_OFF",
         "ADD_LIVE_NOTE_PREPARING")


def _all_notes():
    out = []
    for k in NOTES:
        v = getattr(AP, k)
        if "{n}" in v:
            v = v.format(n=AP.voxel_limit_words(AP.ADD_LIVE_CELLS_MAX_VOXELS))
        out.append(v)
    return out


# --------------------------------------------------------------------------- #
# every row-13.5 state -> its note
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("whole", [False, True])
def test_over_the_live_limit_names_the_limit(whole):
    s = _state(TWENTY_FIVE, whole_patch=whole, live_whole_ok=False,
               live_box_ok=False)
    note = AP.live_tick_note(s)
    assert note == AP.ADD_LIVE_NOTE_OVER.format(n="20 million")
    assert note == "Not live over 20 million — tick Always update as I click."
    # the same number S1 names, and the same remedy
    assert "20 million voxels" in AP.s1_sentence(s)
    assert "Preview" in AP.s1_sentence(s)


@pytest.mark.parametrize("whole", [False, True])
def test_walls_the_quick_fill_does_not_redo_name_wall_type(whole):
    s = _state(whole_patch=whole, live_whole_ok=False, live_box_ok=False,
               live_aside="walls")
    assert AP.live_tick_note(s) == AP.ADD_LIVE_NOTE_WALLS
    assert AP.ADD_LIVE_NOTE_WALLS == (
        "Not live here: Wall type — tick Always update as I click.")
    assert AP.s1_sentence(s).startswith("Press Preview")


@pytest.mark.parametrize("whole", [False, True])
def test_spread_names_grow_by(whole):
    s = _state(whole_patch=whole, live_whole_ok=False, live_box_ok=False,
               live_aside="grow")
    assert AP.live_tick_note(s) == AP.ADD_LIVE_NOTE_GROW
    assert AP.ADD_LIVE_NOTE_GROW == (
        "Not live here: Grow by — tick Always update as I click.")
    assert AP.ADD_GROW_LABEL in AP.ADD_LIVE_NOTE_GROW


@pytest.mark.parametrize("whole", [False, True])
def test_flat_spots_at_a_set_spacing_name_always_update(whole):
    s = _state(whole_patch=whole, live_whole_ok=False, live_box_ok=False,
               flat_spots=True, default_spacing=False)
    assert AP.live_tick_note(s) == AP.ADD_LIVE_NOTE_FLAT
    assert AP.ADD_LIVE_NOTE_FLAT == (
        "Not live here: flat spots — tick Always update as I click.")


@pytest.mark.parametrize("whole", [False, True])
def test_flat_spots_at_the_default_spacing_name_set_spacing(whole):
    s = _state(whole_patch=whole, live_whole_ok=False, live_box_ok=False,
               flat_spots=True, default_spacing=True)
    note = AP.live_tick_note(s)
    assert note == AP.ADD_LIVE_NOTE_SPACING
    assert note == "Not live here: default spacing — Set spacing… may help."
    # the tick row and S1 send the user to the same control
    assert "Set spacing…" in note
    assert "Set spacing…" in AP.s1_sentence(s)


@pytest.mark.parametrize("whole", [False, True])
def test_another_step_aside_has_its_own_note(whole):
    """A quick fill that could not take the region for another reason (it
    failed, or is switched off): the note names Preview, as S1 does."""
    s = _state(whole_patch=whole, live_whole_ok=False, live_box_ok=False)
    assert AP.live_tick_note(s) == AP.ADD_LIVE_NOTE_OFF
    assert AP.ADD_LIVE_NOTE_OFF == (
        "Not live here: the quick fill could not run — press Preview.")


def test_the_fill_being_prepared():
    s = _state(fill_pending=True)
    assert AP.s1_sentence(s) == AP.ADD_S1_LIVE_PREPARING
    assert AP.live_tick_note(s) == AP.ADD_LIVE_NOTE_PREPARING
    assert AP.ADD_LIVE_NOTE_PREPARING == "Preparing the fill…"


# --------------------------------------------------------------------------- #
# hidden
# --------------------------------------------------------------------------- #
def test_live_running_has_no_note():
    assert AP.live_tick_note(_state()) == ""
    assert AP.live_tick_note(_state(whole_patch=True)) == ""
    # at the live limit a box is still live
    assert AP.live_tick_note(_state((200, 100, 1000))) == ""


@pytest.mark.parametrize("kw", [
    dict(shape=TWENTY_FIVE), dict(live_box_ok=False, live_aside="walls"),
    dict(live_box_ok=False, flat_spots=True, default_spacing=True),
    dict(fill_pending=True)])
def test_the_tick_off_has_no_note(kw):
    s = _state(**kw)
    assert AP.live_tick_note(s), "on: a note"
    assert AP.live_tick_note(dict(s, live_on=False)) == ""


@pytest.mark.parametrize("kw", [
    dict(shape=TWENTY_FIVE), dict(live_box_ok=False, live_aside="walls"),
    dict(live_box_ok=False, live_aside="grow"),
    dict(live_box_ok=False, flat_spots=True),
    dict(live_box_ok=False, flat_spots=True, default_spacing=True),
    dict(live_box_ok=False), dict(fill_pending=True),
    dict(whole_patch=True, live_whole_ok=False)])
def test_always_update_on_has_no_note(kw):
    s = _state(**kw)
    assert AP.live_tick_note(dict(s, live_always=True)) == ""


def test_only_cells_click_markers_have_a_note():
    s = _state(TWENTY_FIVE)
    assert AP.live_tick_note(dict(s, preset="nuclei",
                                  look_for=AP.LOOK_MARKERS)) == ""
    assert AP.live_tick_note(dict(s, source=AP.SOURCE_RUN)) == ""


# --------------------------------------------------------------------------- #
# one decision for the two rows
# --------------------------------------------------------------------------- #
def _row_13_5_sentences():
    n = AP.voxel_limit_words(AP.ADD_LIVE_CELLS_MAX_VOXELS)
    return {AP.ADD_S1_LIVE_OVER_GATE_BOX.format(n=n),
            AP.ADD_S1_LIVE_OVER_GATE_PATCH_SIZE.format(n=n),
            AP.ADD_S1_LIVE_OVER_GATE_PATCH, AP.ADD_S1_LIVE_STEPPED_ASIDE_BOX,
            AP.ADD_S1_LIVE_STEPPED_ASIDE_SPACING_BOX,
            AP.ADD_S1_LIVE_STEPPED_ASIDE_SPACING_PATCH,
            AP.ADD_S1_LIVE_PREPARING}


def test_the_note_shows_exactly_when_s1_shows_row_13_5():
    rows = _row_13_5_sentences()
    n = 0
    for (shape, whole, whole_ok, box_ok, aside, flat, default, pending,
         live, always) in itertools.product(
            (FIVE, TWENTY_FIVE), (False, True), (False, True),
            (False, True), (None, "walls", "grow"), (False, True),
            (False, True), (False, True), (False, True), (False, True)):
        s = _state(shape, whole_patch=whole, live_whole_ok=whole_ok,
                   live_box_ok=box_ok, live_aside=aside, flat_spots=flat,
                   default_spacing=default, fill_pending=pending,
                   live_on=live, live_always=always)
        # (with "Always update as I click" on S1 may still say the compile
        # wait; the line under the tick never shows then)
        shown = AP.s1_sentence(s) in rows and not always
        note = AP.live_tick_note(s)
        assert bool(note) == shown, s
        n += shown
        if not shown:
            continue
        # the spacing names one control on both rows, the size one number
        if "Set spacing…" in AP.s1_sentence(s):
            assert note == AP.ADD_LIVE_NOTE_SPACING, s
        if AP.s1_sentence(s) == AP.ADD_S1_LIVE_PREPARING:
            assert note == AP.ADD_LIVE_NOTE_PREPARING, s
        if "over 20 million" in AP.s1_sentence(s):
            assert note == AP.ADD_LIVE_NOTE_OVER.format(n="20 million"), s
    assert n > 50
    # S1 saying something else first hides the line: a transient, busy, an
    # update running, the review, a result on screen
    shown_state = None
    for (shape, whole, whole_ok, box_ok, aside, flat, default, pending,
         live) in itertools.product(
            (FIVE, TWENTY_FIVE), (False, True), (False, True),
            (False, True), (None, "walls", "grow"), (False, True),
            (False, True), (False, True), (False, True)):
        s = _state(shape, whole_patch=whole, live_whole_ok=whole_ok,
                   live_box_ok=box_ok, live_aside=aside, flat_spots=flat,
                   default_spacing=default, fill_pending=pending,
                   live_on=live)
        if AP.live_tick_note(s):
            shown_state = s
            break
    assert shown_state is not None
    for extra in (dict(transient=AP.ADD_S1_STOPPED), dict(busy=True),
                  dict(live_updating=True), dict(review=(3, 100, 0)),
                  dict(candidates=(4, 0, 0, 0, 4))):
        s = dict(shown_state, **extra)
        assert AP.s1_sentence(s) not in rows, extra
        assert AP.live_tick_note(s) == "", extra


# --------------------------------------------------------------------------- #
# the words
# --------------------------------------------------------------------------- #
def test_every_note_is_at_most_60_characters():
    for note in _all_notes():
        assert 0 < len(note) <= 60, (len(note), note)


def test_the_notes_name_controls_by_their_page_words(widget):  # noqa: F811
    from qtpy.QtWidgets import QLabel

    labels = {lab.text() for lab in widget.findChildren(QLabel)}
    assert "Wall type" in labels
    assert AP.ADD_GROW_LABEL in labels
    assert "Wall type" in AP.ADD_LIVE_NOTE_WALLS
    for k in ("ADD_LIVE_NOTE_WALLS", "ADD_LIVE_NOTE_GROW",
              "ADD_LIVE_NOTE_FLAT"):
        assert AP.ADD_LIVE_ALWAYS_LABEL in getattr(AP, k), k
    assert widget.add_live_always_cb.text() == AP.ADD_LIVE_ALWAYS_LABEL
    assert AP.ADD_LIVE_ALWAYS_LABEL in AP.ADD_LIVE_NOTE_OVER
    assert "Preview" in AP.ADD_LIVE_NOTE_OFF


def test_the_notes_name_no_engine_word():
    for note in _all_notes():
        for word in ("flood", "watershed", "ridge", "numba", "compil",
                     "stage", "kernel", "fallback"):
            assert word not in note.lower(), (word, note)


# --------------------------------------------------------------------------- #
# the line on the page
# --------------------------------------------------------------------------- #
def _cells_clicks(w):
    reset_add(w)
    w.set_add_preset("cells")
    w.update_add_ui_state(box_active=True, whole_patch=False)
    w.update_add_ui_state(marker_sources=[
        (AP.ADD_ITEM_RUN.format(name="r2"), True, ("labels", "r2")),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))])
    keys = list(w._add_source_keys)
    w.add_marker_source_combo.setCurrentIndex(keys.index(("paint", None)))
    return keys


def _undo(w):
    w.update_add_ui_state(live_tick_note="")
    w.add_live_cb.setChecked(True)
    w.add_live_always_cb.setChecked(False)
    w.set_add_preset("nuclei")
    w.add_look_for_combo.setCurrentIndex(0)
    w.update_add_ui_state(box_active=False, whole_patch=False)
    reset_add(w)


def test_the_line_sits_directly_under_the_tick(widget):  # noqa: F811
    from napari_epicycle.annotator.advanced_features_widget import (
        _HELPER_LINE_STYLE,
    )

    w = widget
    note = w._add_live_note
    lay = w.add_live_cb.parentWidget().layout()
    i_tick = lay.indexOf(w.add_live_cb)
    assert i_tick >= 0
    assert lay.indexOf(note) == i_tick + 1
    assert lay.indexOf(w.add_live_always_cb) == i_tick + 2
    assert note.styleSheet() == _HELPER_LINE_STYLE, "grey, not amber"
    assert note.wordWrap()
    assert note.isHidden(), "hidden on a fresh page"
    # the tick's tooltip is unchanged
    assert w.add_live_cb.toolTip() == AP.ADD_HELP_LIVE_TICK


def test_the_line_shows_the_pushed_reason_and_hides(widget):  # noqa: F811
    w = widget
    note = w._add_live_note
    _cells_clicks(w)
    try:
        w.update_add_ui_state(live_tick_note=AP.ADD_LIVE_NOTE_WALLS)
        assert not note.isHidden()
        assert note.text() == AP.ADD_LIVE_NOTE_WALLS
        # the tick off: no reason to give
        w.add_live_cb.setChecked(False)
        assert note.isHidden()
        w.add_live_cb.setChecked(True)
        assert not note.isHidden()
        # "Always update as I click" on: hidden, whatever was pushed
        w.add_live_always_cb.setChecked(True)
        assert note.isHidden()
        w.add_live_always_cb.setChecked(False)
        assert not note.isHidden()
        # live runs: the orchestrator pushes "" and the line goes
        w.update_add_ui_state(live_tick_note="")
        assert note.isHidden()
        w.update_add_ui_state(live_tick_note=AP.ADD_LIVE_NOTE_PREPARING)
        assert not note.isHidden()
        # another source: the tick is not there, nor is its line
        w.add_marker_source_combo.setCurrentIndex(0)
        assert note.isHidden()
    finally:
        _undo(w)


def test_nuclei_markers_never_show_the_line(widget):  # noqa: F811
    w = widget
    reset_add(w)
    try:
        w.set_add_preset("nuclei")
        w.update_add_ui_state(box_active=True)
        c = w.add_look_for_combo
        c.setCurrentIndex(c.count() - 1)
        assert w.add_look_markers()
        assert not w.add_live_cb.isHidden(), "the tick shows on the markers"
        w.update_add_ui_state(live_tick_note=AP.ADD_LIVE_NOTE_OVER.format(
            n="20 million"))
        assert w._add_live_note.isHidden()
    finally:
        _undo(w)


# --------------------------------------------------------------------------- #
# the orchestrator pushes it from S1's own state
# --------------------------------------------------------------------------- #
def test_the_page_gets_the_note_s1_s_state_decides(ci, monkeypatch):
    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    ci._add_refresh_ui()
    assert ci.adv.last("live_tick_note") == ""
    # walls the quick fill does not redo, on a box over the full-run size
    ci.adv.wall = dict(ci.adv.wall, wall_kind="frangi")
    ci._add_refresh_ui()
    state = ci._add_s1_state()
    assert state["live_aside"] == "walls"
    assert ci.adv.last("live_tick_note") == AP.ADD_LIVE_NOTE_WALLS
    assert ci.adv.last("live_tick_note") == AP.live_tick_note(state)
    # "Always update as I click" lifts it
    ci._on_add_live_always_toggled(True)
    ci._add_refresh_ui()
    assert ci.adv.last("live_tick_note") == ""
    ci._on_add_live_always_toggled(False)
    # Grow by on Spread, the walls redone
    ci.adv.wall = dict(ci.adv.wall, wall_kind="membrane")
    ci.adv.params["growth"] = AP.GROW_SPREAD
    ci._add_refresh_ui()
    assert ci._add_s1_state()["live_aside"] == "grow"
    assert ci.adv.last("live_tick_note") == AP.ADD_LIVE_NOTE_GROW
    ci.adv.params["growth"] = AP.GROW_FLOOD
    ci._add_refresh_ui()
    assert ci._add_s1_state()["live_aside"] is None
    assert ci.adv.last("live_tick_note") == ""


def test_a_flat_step_aside_reaches_the_page(ci, monkeypatch):
    from packages.epicycle_annotator.reshape import incremental as INC

    monkeypatch.setattr(AP, "ADD_LIVE_FULL_RUN_MAX_VOXELS", 1000)
    monkeypatch.setattr(
        INC, "prepare",
        lambda *a, **k: (None, (INC.FULL_RUN, "[ADD] flat spots")))
    monkeypatch.setattr(type(ci), "_add_default_spacing", lambda self: True)
    assert ci._add_marker_click(CELL_A)
    _fire(ci)
    _finish(ci)
    assert ci._add_cells_click_flat()
    assert ci.adv.last("live_tick_note") == AP.ADD_LIVE_NOTE_SPACING
    monkeypatch.setattr(type(ci), "_add_default_spacing", lambda self: False)
    ci._add_refresh_ui()
    assert ci.adv.last("live_tick_note") == AP.ADD_LIVE_NOTE_FLAT


# --------------------------------------------------------------------------- #
# findable rows: the closed chevrons summarise
# --------------------------------------------------------------------------- #
def test_the_closed_markers_and_walls_titles_summarise(widget):  # noqa: F811
    w = widget
    _cells_clicks(w)
    try:
        w.update_add_ui_state(click_count=3, click_markers=[
            (65520 + i, (1, 2, 3), True) for i in range(3)])
        w.add_markers_btn.setChecked(False)
        w.add_walls_btn.setChecked(False)
        title = str(w.add_markers_btn.property("_epicycle_chevron_title"))
        assert title == "Markers: clicks · 3 placed"
        # the shrink-wrap outline is named in the closed Walls title
        w.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        title = str(w.add_walls_btn.property("_epicycle_chevron_title"))
        assert title.startswith(AP.ADD_SUM_WALLS + ": ")
        assert title.endswith(AP.ADD_SUM_OUTLINE_SHRINK)
        # titles stay summaries: no live reason is folded into them
        w.update_add_ui_state(live_tick_note=AP.ADD_LIVE_NOTE_WALLS)
        assert "live" not in str(
            w.add_walls_btn.property("_epicycle_chevron_title")).lower()
    finally:
        w.update_add_ui_state(click_count=0, click_markers=[])
        w.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        _undo(w)

"""The Markers source picked on Cells stays picked for the next box or
whole patch on the same patch.

The end of a box (Close box, Done, the main Undo emptying the review) used
to forget that the user had picked the source, and the next Place box put
Markers back to the preset's default (nuclei masks).  Pressing Whole patch
after it, Preview refused with "Cells fills a whole patch only from Markers
▸ Objects from another run ..." although the run had been picked a minute
before.

Pinned here:

* after Close box, Done or an emptied review, the next Place box and the
  next Whole patch start with the run still picked, and read it again;
* a preset round trip after a box ended keeps the pick (as it does while a
  box is up);
* a box on another patch or run puts Markers back to the default --
  whether the switch found Add idle, with a box up or in the review -- and
  reads no run; the switch in place on the same patch (after an Extract)
  keeps the pick;
* the page: the end of a box keeps the source as the user's pick, a full
  forget does not.
"""

from __future__ import annotations

import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_d_fixpass import (
    _assert_the_run_is_read_and_counted,
    _cells_page_with_a_run,
)

RUN = ("labels", "cells")
DEFAULT = (AP.SOURCE_NUCLEI, None)


def _end(hh, how):
    """End the box the way the page does."""
    if how == "close":
        hh._add_secondary()                 # Close box: nothing added
    elif how == "done":
        hh._add_running = False             # the review after an Accept
        hh._add_review = True
        hh._add_primary()                   # Done
    else:                                   # the main Undo was emptied
        hh._add_running = False
        hh._add_review = True
        hh._add_after_undo_emptied()
    assert not hh._add_running and not hh._add_review


def _start(hh, how):
    if how == "box":
        hh._start_add()                     # Place box
    else:
        hh._start_add_whole()               # Whole patch with no box
    assert hh._add_running


@pytest.mark.parametrize("start", ["box", "whole"])
@pytest.mark.parametrize("end", ["close", "done", "undo_emptied"])
def test_the_run_picked_under_markers_survives_the_end_of_a_box(
        widget, monkeypatch, tmp_path, end, start):
    hh, cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    try:
        hh._start_add()
        _assert_the_run_is_read_and_counted(hh, widget, cells)
        _end(hh, end)
        assert widget.get_add_marker_source() == RUN
        _start(hh, start)
        assert widget.get_add_marker_source() == RUN, "the pick is kept"
        # the run is read again for this box (the last one's copy went
        # with it) and counted
        _assert_the_run_is_read_and_counted(hh, widget, cells)
        assert hh._add_transient != AP.ADD_S1_WHOLE_PICK_RUN
        assert widget.add_status_label.text() != AP.ADD_S1_WHOLE_PICK_RUN
    finally:
        hh._reject_add()
        widget.set_add_marker_sources([])
        reset_add(widget)


def test_a_preset_round_trip_after_a_box_keeps_the_pick(
        widget, monkeypatch, tmp_path):
    hh, _cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    monkeypatch.setattr(widget, "_on_add_preset_cb",
                        hh._on_add_preset_changed, raising=False)
    try:
        hh._start_add()
        _end(hh, "close")
        widget.add_preset_combo.setCurrentIndex(0)          # Nuclei
        widget.add_preset_combo.setCurrentIndex(1)          # Cells
        assert widget.get_add_marker_source() == RUN
        hh._start_add()
        assert widget.get_add_marker_source() == RUN
        assert hh._add_run_state == "loading"
    finally:
        hh._reject_add()
        widget.set_add_marker_sources([])
        reset_add(widget)


@pytest.mark.parametrize("moved", ["patch", "run"])
@pytest.mark.parametrize("state", ["idle", "box", "review"])
@pytest.mark.parametrize("start", ["box", "whole"])
def test_another_patch_or_run_puts_markers_back_to_the_default(
        widget, monkeypatch, tmp_path, state, moved, start):
    hh, _cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    try:
        hh._start_add()
        if state == "idle":
            _end(hh, "close")
        elif state == "review":
            hh._add_running = False
            hh._add_review = True
        assert widget.get_add_marker_source() == RUN
        hh._teardown_add_for_switch()
        assert not hh._add_running and not hh._add_review
        if moved == "patch":
            hh.ctx["patch_name"] = "Q"
        else:
            hh.ctx["run_id"] = "another"
        _start(hh, start)
        if start == "whole":
            # a Cells whole patch lists only runs and never picks one: the
            # combo is blank, and the default comes back with the box
            assert widget.get_add_marker_source() == ("labels", None)
        else:
            assert widget.get_add_marker_source() == DEFAULT
        assert "add_marker_source_combo" not in widget._add_touched
        assert getattr(hh, "_add_guide_worker", None) is None
        assert hh._add_run_state is None
        if start == "whole":
            hh._set_add_box_size_fraction(1.0 / 8.0)       # S
            assert widget.get_add_marker_source() == DEFAULT
            assert "add_marker_source_combo" not in widget._add_touched
            assert getattr(hh, "_add_guide_worker", None) is None
    finally:
        hh._reject_add()
        widget.set_add_marker_sources([])
        reset_add(widget)


@pytest.mark.parametrize("state", ["idle", "review"])
def test_the_switch_in_place_on_the_same_patch_keeps_the_pick(
        widget, monkeypatch, tmp_path, state):
    """After an Extract the switch is re-run on the same patch: the pick
    is not the other patch's, so it stays."""
    hh, cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    try:
        hh._start_add()
        if state == "idle":
            _end(hh, "close")
        else:
            hh._add_running = False
            hh._add_review = True
        hh._teardown_add_for_switch()
        hh._start_add()
        assert widget.get_add_marker_source() == RUN
        _assert_the_run_is_read_and_counted(hh, widget, cells)
    finally:
        hh._reject_add()
        widget.set_add_marker_sources([])
        reset_add(widget)


def test_the_page_keeps_the_source_pick_only_when_asked(widget):
    reset_add(widget)
    widget.set_add_preset("cells")
    widget.set_add_marker_sources([
        (AP.ADD_ITEM_RUN.format(name="cells"), True, RUN),
        (AP.ADD_ITEM_NUCLEI, True, DEFAULT),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))])
    try:
        assert widget.get_add_marker_source() == DEFAULT
        widget.add_marker_source_combo.setCurrentIndex(0)    # the user
        widget.add_min_size_sb.setValue(7)                   # another key
        assert {"add_marker_source_combo",
                "add_min_size_sb"} <= widget._add_touched
        widget.clear_add_touched(keep_source=True)
        assert widget._add_touched == {"add_marker_source_combo"}
        widget.clear_add_touched()
        assert widget._add_touched == set()
        widget.add_marker_source_combo.setCurrentIndex(0)
        widget.forget_add_source()
        assert widget.get_add_marker_source() == DEFAULT
        assert "add_marker_source_combo" not in widget._add_touched
    finally:
        widget.set_add_marker_sources([])
        reset_add(widget)

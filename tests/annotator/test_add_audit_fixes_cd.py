"""The combined fix pass: the A5 UX audit (group C) and the owner's live
test of 2026-09-27 20:33 (group D).

* C-L1 -- the same-spot refusal on an UNTICKED marker (its ball is not on
  screen) says where it is hiding.
* C-L2 -- the ticked tail is " · k of n ticked" (no repeated "markers").
* C-L3 -- a double click's second press lands in the marker its first press
  just placed: refused quietly, within Qt's double-click interval of THAT
  marker's placement, not with the amber same-spot sentence.
* D-1 -- "Treat dark space as outside" OFF on Cells: the amber
  ADD_NOTE_DARK_OFF_LEAK under S1, the spacing note's rules; stacked with
  the spacing note it put 7f over the ceiling, so it wins the slot and the
  spacing note moves into the Advanced help slot.
(C-MED, the idle ceiling measures, is in test_panel_skeleton.py.)
"""

from __future__ import annotations

import types

import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_click_markers import A, B, _ids, _place
from .test_add_click_markers import h  # noqa: F401  (the marker fixture)


# --------------------------------------------------------------------------- #
# C-L1
# --------------------------------------------------------------------------- #
def test_a_click_in_an_unticked_marker_says_where_it_hides(h):  # noqa: F811
    _place(h, A)
    h._on_add_click_tick(8, False)
    assert not h._add_marker_click((12, 21, 20))
    assert h._add_transient == AP.ADD_S1_CLICK_SAME_SPOT_UNTICKED.format(id=8)
    assert h.adv.last("status_kind") == "warn"
    assert _ids(h) == [8]
    h._on_add_click_tick(8, True)
    assert not h._add_marker_click((12, 21, 20))
    assert h._add_transient == AP.ADD_S1_CLICK_SAME_SPOT.format(id=8)


def test_the_unticked_sentence():
    assert AP.ADD_S1_CLICK_SAME_SPOT_UNTICKED.format(id=12) == (
        "That spot is inside unticked marker 12 — tick it in the list to "
        "see it, or click a little further away.")


# --------------------------------------------------------------------------- #
# C-L2
# --------------------------------------------------------------------------- #
def test_the_ticked_tail_drops_the_repeated_word():
    assert AP.ADD_S1_TICKED_TAIL == " · {k} of {n} ticked"
    line = AP.box_line((2, 3, 4), preset="cells", n_clicks=7, n_ticked=5)
    assert line.endswith("· 7 markers placed · 5 of 7 ticked")
    assert line.count("markers") == 1


# --------------------------------------------------------------------------- #
# C-L3
# --------------------------------------------------------------------------- #
class _Clock:
    def __init__(self):
        self.t = 100.0

    def monotonic(self):
        return self.t


def test_a_double_clicks_second_press_is_quiet(h, monkeypatch):  # noqa: F811
    from napari_epicycle.annotator import curation_3d_widget as CW

    clock = _Clock()
    monkeypatch.setattr(CW, "_time", clock)
    h._add_double_click_s = lambda: 0.5
    _place(h, A)
    before = h._add_transient
    clock.t += 0.2                                   # inside the interval
    assert not h._add_marker_click(A)
    assert h._add_transient == before, "no amber same-spot sentence"
    assert _ids(h) == [8], "one marker"


def test_a_slow_second_click_is_the_same_spot_refusal(h, monkeypatch):  # noqa: F811
    from napari_epicycle.annotator import curation_3d_widget as CW

    clock = _Clock()
    monkeypatch.setattr(CW, "_time", clock)
    h._add_double_click_s = lambda: 0.5
    _place(h, A)
    clock.t += 0.8                                   # after the interval
    assert not h._add_marker_click(A)
    assert h._add_transient == AP.ADD_S1_CLICK_SAME_SPOT.format(id=8)


def test_only_the_just_placed_marker_is_quiet(h, monkeypatch):  # noqa: F811
    """A fast click into an OLDER marker is still refused out loud."""
    from napari_epicycle.annotator import curation_3d_widget as CW

    clock = _Clock()
    monkeypatch.setattr(CW, "_time", clock)
    h._add_double_click_s = lambda: 0.5
    _place(h, A, B)                                  # B is the last one
    clock.t += 0.1
    assert not h._add_marker_click(A)
    assert h._add_transient == AP.ADD_S1_CLICK_SAME_SPOT.format(id=8)


def test_the_interval_is_qts(qapp):  # noqa: F811
    from qtpy.QtWidgets import QApplication

    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    got = Curation3DAnnotator._add_double_click_s(types.SimpleNamespace())
    assert got == pytest.approx(QApplication.doubleClickInterval() / 1000.0)


# --------------------------------------------------------------------------- #
# D-1
# --------------------------------------------------------------------------- #
def _cells(widget, **kw):
    reset_add(widget)
    widget.set_add_preset("cells", silent=True)
    widget.update_add_ui_state(**kw)


def test_the_dark_off_note_shows_on_cells_with_the_tick_off(widget):
    from napari_epicycle.ui_shell.theme import PANEL_WARN_STYLE

    _cells(widget, box_active=True, default_spacing=False)
    note = widget.add_dark_off_note
    try:
        assert widget.add_dark_outside_cb.isChecked()
        assert note.isHidden(), "the tick is on"
        widget.add_dark_outside_cb.setChecked(False)
        assert not note.isHidden()
        assert note.text() == AP.ADD_NOTE_DARK_OFF_LEAK
        assert note.styleSheet() == PANEL_WARN_STYLE
        assert note.wordWrap()
        # under S1, beside the spacing note
        lay = widget._add_s1_container.parentWidget().layout()
        assert lay.indexOf(note) == lay.indexOf(widget.add_spacing_note) + 1
        # not over candidates / REVIEW
        widget.update_add_ui_state(candidate_ids=[5],
                                   candidate_sizes={5: 10},
                                   candidate_edge_ids=set())
        assert note.isHidden()
        widget.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                                   candidate_edge_ids=set())
        assert not note.isHidden()
        widget.update_add_ui_state(review_active=True)
        assert note.isHidden()
        widget.update_add_ui_state(review_active=False)
        # Nuclei: never
        widget.set_add_preset("nuclei", silent=True)
        assert note.isHidden()
    finally:
        widget.add_dark_outside_cb.setChecked(True)
        widget.update_add_ui_state(box_active=False)
        reset_add(widget)


def test_the_dark_off_note_wins_and_the_spacing_note_moves(widget):
    _cells(widget, box_active=True, default_spacing=True)
    try:
        assert not widget.add_spacing_note.isHidden()
        widget.add_dark_outside_cb.setChecked(False)
        assert not widget.add_dark_off_note.isHidden()
        assert widget.add_spacing_note.isHidden(), "never stacked"
        adv = widget._add_advanced_help.text()
        assert adv.startswith(AP.ADD_DARK_OFF)
        assert AP.ADD_NOTE_DEFAULT_SPACING_BOXED in adv
        widget.add_dark_outside_cb.setChecked(True)
        assert not widget.add_spacing_note.isHidden()
        assert AP.ADD_NOTE_DEFAULT_SPACING_BOXED not in (
            widget._add_advanced_help.text())
    finally:
        widget.add_dark_outside_cb.setChecked(True)
        widget.update_add_ui_state(box_active=False, default_spacing=False)
        reset_add(widget)


def test_the_dark_off_note_says_objects_not_cells():
    assert "cell" not in AP.ADD_NOTE_DARK_OFF_LEAK.lower()
    assert "Advanced" in AP.ADD_NOTE_DARK_OFF_LEAK

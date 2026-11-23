"""The Add page redesign (owner-approved 2026-11-14).

* R1 -- Whole patch is a box size: one row ``[Place box] [Whole patch]`` with
  no box, ``S · M · L · Whole · Numbers… · ⌖`` with one; never grey; the
  Numbers… window has no Whole patch button.  Pressing it always makes the
  box the patch; a choice that cannot fill a whole patch greys Preview and
  S1 says why.  Only S, M, L or a Numbers… slider make it a box again -- a
  marker source, Look for nuclei or a preset change keep it whole.
* R3 -- REVIEW's footer: **Done** (keeps the objects) + **Undo all** (red);
  the card's three buttons share one row.
* R4 -- the secondary has one word per meaning: Close box (a box is up, and
  grey at idle), Discard (over a result), Undo all (REVIEW).  Close box after
  "+ Add another" goes back to the review, the session kept.  Grey while
  busy: Stop is the one exit while running.
* R5 -- no "Accept and refine ▸"; the footer line explains Accept and Discard.
* R6 -- hide, don't grey: the Edge rows on a whole patch, each marker
  source's own Advanced row, the Advanced chevron with no box.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeAdv, Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401


# --------------------------------------------------------------------------- #
# the sentences
# --------------------------------------------------------------------------- #
def test_the_idle_lines():
    assert AP.ADD_S1_IDLE == (
        "Place a box over the space to fill, or press Whole patch.")
    assert AP.ADD_FOOT_IDLE == "Place a box to start."
    assert not hasattr(AP, "ADD_S1_IDLE_WHOLE")
    # no box: both presets say the same line (Whole patch is never grey)
    for state in ({"preset": "nuclei"}, {"preset": "cells"},
                  {"preset": "cells", "source": AP.SOURCE_CLICKS},
                  {"preset": "nuclei", "look_for": AP.LOOK_INSIDE}):
        assert AP.preview_block(dict(state, has_box=False)) == AP.ADD_S1_IDLE
        assert AP.s1_kind(dict(state, has_box=False)) == "idle"


def test_the_footer_lines():
    assert AP.ADD_FOOT_BOX == "Close box ends this box; nothing is added."
    assert AP.ADD_FOOT_CANDIDATES == (
        "Accept keeps the ticked objects. Discard throws this result away; "
        "the box stays.")
    assert AP.foot_review(721) == (
        "Done keeps the 721 objects. Undo all takes back every Accept since "
        "the first box.")
    assert AP.foot_review(1).startswith("Done keeps the 1 object. ")
    assert "Close box" in AP.ADD_FOOT_BOX_CLEARED
    assert "Cancel" not in AP.ADD_FOOT_BOX_CLEARED
    # after "+ Add another": Close box goes back to the review, so the line
    # never says Undo all is gone
    for n in (1, 12):
        line = AP.foot_box_session(n)
        assert line.startswith("Close box ")
        assert "no longer" not in line
    assert "12 objects" in AP.foot_box_session(12)
    assert "1 object " in AP.foot_box_session(1)


def test_no_add_sentence_names_cancel_any_more():
    """The box's secondary is Close box now: no Add sentence tells the user
    to press a Cancel the page no longer shows."""
    bad = [k for k, v in vars(AP).items()
           if k.startswith("ADD_") and isinstance(v, str)
           and "Cancel" in v]
    assert not bad, bad


def test_the_retired_names_are_gone():
    for name in ("ADD_ACCEPT_REFINE_BTN", "ADD_ACCEPT_REFINE_TIP",
                 "ADD_WHOLE_NEEDS_RUN", "ADD_WHOLE_ONLY_INSIDE",
                 "ADD_WHOLE_HELP", "ADD_S1_LEFT_WHOLE_ANYWHERE",
                 "ADD_S1_LEFT_WHOLE_OVER_LIMIT", "ADD_S1_LEFT_WHOLE_SOURCE",
                 "ADD_S1_LEFT_WHOLE_SHRUNK_TAIL", "ADD_FOOT_REVIEW"):
        assert not hasattr(AP, name), name


def test_a_whole_patch_that_cannot_be_filled_greys_preview_and_says_why():
    base = {"has_box": True, "whole_patch": True, "box_shape": (60, 900, 900)}
    # Cells needs Markers ▸ Objects from run or Click markers: pick one, or
    # (no other run on this patch) Click markers or a box
    src = AP.SOURCE_NUCLEI
    assert AP.preview_block(dict(base, preset="cells", source=src,
                                 has_other_run=True)) == (
        AP.ADD_S1_WHOLE_PICK_RUN)
    assert AP.preview_block(dict(base, preset="cells", source=src)) == (
        AP.ADD_S1_WHOLE_NO_RUN)
    # Click markers are not refused for the source: the next row speaks
    # (no Walls channel in this state)
    for other in (True, False):
        why = AP.preview_block(dict(base, preset="cells",
                                    source=AP.SOURCE_CLICKS,
                                    has_other_run=other, n_clicks=2))
        assert why == AP.ADD_S1_PICK_WALL_CHANNEL
    # Nuclei ▸ Anywhere runs on boxes
    assert AP.ADD_S1_WHOLE_ANYWHERE == (
        "Anywhere in the box runs on boxes — pick Inside each cell, or S, M "
        "or L.")
    assert AP.preview_block(dict(base, preset="nuclei", channel="nuc",
                                 look_for=AP.LOOK_ANYWHERE)) == (
        AP.ADD_S1_WHOLE_ANYWHERE)
    # ... and not on an ordinary box
    assert AP.preview_block(dict(base, whole_patch=False, preset="nuclei",
                                 channel="nuc", box_shape=(10, 10, 10),
                                 look_for=AP.LOOK_ANYWHERE,
                                 empty_share=0.5)) is None


# --------------------------------------------------------------------------- #
# the orchestrator (the Qt-free harness)
# --------------------------------------------------------------------------- #
@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((12, 20, 20), np.int32),
                 images={"nuc": np.zeros((12, 20, 20), np.float32)},
                 spacing=(0.5, 0.25, 0.25))
    hh.adv = FakeAdv()
    hh._get_advanced_features_widget = lambda: hh.adv
    hh._add_restore_preset = lambda: None
    return hh


def _whole_roi(h):
    return tuple(slice(0, n) for n in h.sync_manager.shared_labels["L"].shape)


def test_whole_on_cells_with_click_markers_waits_for_a_marker(h):
    """Click markers take a whole patch: the patch is made and Preview
    waits for a marker (Place markers is off in this harness)."""
    h.adv.preset = "cells"
    h.adv.source = ("paint", None)
    h._start_add_whole()
    assert h._add_running and h._add_whole_patch
    assert h._add_roi == _whole_roi(h)
    assert h.adv.last("preview_enabled") is False
    assert h._add_place_markers_on is False
    assert h.adv.last("status_text") == AP.ADD_S1_NO_MARKERS


def test_whole_on_cells_with_nuclei_masks_makes_the_patch_and_refuses(h):
    """Nuclei masks cannot fill a whole patch, and this harness lists no
    other run of the patch: Click markers or a box is the way."""
    h.adv.preset = "cells"
    h.adv.source = ("nuclei_masks", None)
    h._start_add_whole()
    assert h._add_running and h._add_whole_patch
    assert h._add_roi == _whole_roi(h)
    assert h.adv.last("preview_enabled") is False
    assert h.adv.last("status_text") == AP.ADD_S1_WHOLE_NO_RUN


def test_whole_on_anywhere_makes_the_patch_and_preview_refuses(h):
    h.adv.look = (AP.LOOK_ANYWHERE, None)
    h._start_add_whole()
    assert h._add_running and h._add_whole_patch
    assert h._add_roi == _whole_roi(h)
    assert h.adv.last("preview_enabled") is False
    assert h.adv.last("status_text") == AP.ADD_S1_WHOLE_ANYWHERE


def test_whole_where_spread_cannot_run_makes_the_patch_and_says_why(
        h, monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as CW

    monkeypatch.setitem(CW._ADD_NUMBA, "ok", False)
    h.adv.preset = "cells"
    h.adv.source = ("labels", "cells")
    h._start_add_whole()
    assert h._add_running and h._add_whole_patch
    assert h.adv.last("preview_enabled") is False
    assert h.adv.last("status_text") == AP.ADD_WHOLE_NEEDS_NUMBA


def test_another_marker_source_keeps_the_whole_patch(h):
    h.adv.preset = "cells"
    h.adv.source = ("labels", "cells")
    h._start_add_whole()
    box = h._add_box
    # the start lists the patch's runs (none in this harness): one more
    h._add_run_sources = {"cells": object()}
    h.adv.source = ("nuclei_masks", None)
    h._on_add_source_changed(h.adv.source)
    assert h._add_whole_patch and h._add_box == box
    assert h.adv.last("status_text") == AP.ADD_S1_WHOLE_PICK_RUN
    assert h.adv.last("preview_enabled") is False
    # Click markers keep the whole patch too, and ask for a marker
    h.adv.source = ("paint", None)
    h._on_add_source_changed(h.adv.source)
    assert h._add_whole_patch and h._add_box == box
    # a source change turns Place markers off: S1 asks for a marker
    assert h._add_place_markers_on is False
    assert h.adv.last("status_text") == AP.ADD_S1_NO_MARKERS
    assert h.adv.last("preview_enabled") is False


def test_anywhere_keeps_the_whole_patch(h):
    h.adv.look = (AP.LOOK_INSIDE, "cells")
    h._start_add_whole()
    box = h._add_box
    h.adv.look = (AP.LOOK_ANYWHERE, None)
    h._on_add_look_for_changed(h.adv.look)
    assert h._add_whole_patch and h._add_box == box
    assert h.adv.last("status_text") == AP.ADD_S1_WHOLE_ANYWHERE


@pytest.mark.parametrize("preset", ["cells", "nuclei"])
def test_a_preset_change_keeps_the_whole_patch(h, preset):
    h.adv.preset = "nuclei" if preset == "cells" else "cells"
    h._start_add_whole()
    box = h._add_box
    h.adv.params["box_budget"] = 1000           # the patch is over it
    h.adv.preset = preset
    h._on_add_preset_changed(preset)
    assert h._add_whole_patch and h._add_box == box


def test_only_s_m_l_or_a_slider_make_it_a_box_again(h):
    h._start_add_whole()
    h._set_add_box_size_fraction(0.25)
    assert not h._add_whole_patch
    h._start_add_whole()
    b = h._add_box
    h._on_add_box_changed(b.cz, b.cy, b.cx, 4, 8, 8)
    assert not h._add_whole_patch


def test_add_another_keeps_a_whole_patch_whatever_the_choices(h):
    h.adv.look = (AP.LOOK_ANYWHERE, None)
    h._start_add_whole()
    was = h._add_box
    h._add_running = False
    h._start_add(reuse_box=True)
    assert h._add_box == was and h._add_whole_patch


# --- R3 / R4: the footer's two buttons in the orchestrator ---------------- #
def _to_review(h):
    """Accept one object -> REVIEW."""
    h.place_box((slice(2, 10), slice(2, 18), slice(2, 18)))
    res = np.zeros((8, 16, 16), np.int32)
    res[2:5, 4:8, 4:8] = 40
    h._add_preview_result = res
    h._add_preview_roi = tuple(h._add_roi)
    h.adv.checked = [40]
    h._on_add_committed_cb = None
    h._accept_add()
    assert h._add_review and len(h._add_session) == 1


def test_done_is_the_primary_in_review_and_keeps_the_objects(h):
    _to_review(h)
    vol = h.sync_manager.shared_labels["L"]
    assert (vol == 40).any()
    h._add_primary()                            # Done
    assert not h._add_review and not h._add_running
    assert h._add_session == []
    assert (vol == 40).any(), "Done keeps the objects"
    assert h.adv.last("review_active") is False


def test_the_primary_over_a_result_is_accept(h):
    h.place_box((slice(2, 10), slice(2, 18), slice(2, 18)))
    res = np.zeros((8, 16, 16), np.int32)
    res[2:5, 4:8, 4:8] = 40
    h._add_preview_result = res
    h._add_preview_roi = tuple(h._add_roi)
    h.adv.checked = [40]
    h._on_add_committed_cb = None
    h._add_primary()
    assert h._add_review


def test_undo_all_is_the_secondary_in_review(h):
    _to_review(h)
    called = []
    h._undo_add_all = lambda: called.append(1)
    h._add_secondary()
    assert called == [1]


def test_close_box_after_add_another_goes_back_to_the_review(h):
    _to_review(h)
    h._add_another()
    assert h._add_running and not h._add_review
    assert h.adv.last("footer_text") == AP.foot_box_session(1)
    locks = list(h.locks)
    h._add_secondary()                          # Close box
    assert h._add_review, "back to the review"
    assert not h._add_running
    assert len(h._add_session) == 1, "the session is kept"
    assert h.adv.last("review_active") is True
    assert h.adv.last("piece_ids") == [40]
    assert h.locks[len(locks):] == [False], "the lock is let go once"


def test_close_box_without_a_session_ends_it(h):
    h.place_box((slice(2, 10), slice(2, 18), slice(2, 18)))
    h._add_secondary()                          # Close box
    assert not h._add_running and not h._add_review


def test_the_secondary_does_nothing_while_busy(h):
    h.place_box((slice(2, 10), slice(2, 18), slice(2, 18)))
    h._add_busy = True
    h._add_secondary()
    assert h._add_running, "Stop is the one exit while running"


def test_accept_and_refine_is_retired():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    assert not hasattr(Curation3DAnnotator, "_accept_add_and_refine")


# --------------------------------------------------------------------------- #
# the panel
# --------------------------------------------------------------------------- #
def _cells(widget, source_index=None):
    widget.set_add_preset("cells")
    widget.set_add_marker_sources([
        (AP.ADD_ITEM_RUN.format(name="cells"), True, ("labels", "cells")),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))])
    if source_index is not None:
        widget.add_marker_source_combo.setCurrentIndex(source_index)


def _done(widget):
    widget.update_add_ui_state(box_active=False, whole_patch=False,
                               review_active=False, busy=False,
                               numba_ok=True, numba_why="",
                               candidate_ids=[], candidate_sizes={},
                               candidate_edge_ids=set(), piece_ids=[],
                               reject_enabled=False, footer_text="")
    widget.set_add_marker_sources([])
    reset_add(widget)


def test_idle_shows_place_box_and_whole_patch_side_by_side(widget):
    reset_add(widget)
    widget.update_add_ui_state(box_active=False)
    try:
        assert widget.add_start_btn.text() == "Place box"
        assert widget.add_whole_btn.text() == AP.ADD_WHOLE_BTN
        for b in (widget.add_start_btn, widget.add_whole_btn):
            assert not b.isHidden() and b.isEnabled()
        assert (widget.add_start_btn.parentWidget()
                is widget.add_whole_btn.parentWidget()
                is widget._add_idle_row)
        assert widget._add_box_container.isHidden()
        assert widget.add_whole_btn.toolTip() == AP.ADD_WHOLE_TIP
    finally:
        _done(widget)


@pytest.mark.parametrize("how", ["anywhere", "cells-clicks", "no-spread"])
def test_whole_patch_is_never_grey(widget, how):
    reset_add(widget)
    try:
        if how != "anywhere":
            _cells(widget, 2)
        if how == "no-spread":
            widget.update_add_ui_state(numba_ok=False, numba_why="missing")
        widget.update_add_ui_state(box_active=False)
        assert widget.add_whole_btn.isEnabled()
        assert widget.add_whole_btn.toolTip() == AP.ADD_WHOLE_TIP
        widget.update_add_ui_state(box_active=True)
        assert widget.add_size_whole_btn.isEnabled()
        assert not widget.add_size_whole_btn.isHidden()
        assert widget.add_whole_btn.isHidden()
    finally:
        _done(widget)


def test_the_box_row_is_s_m_l_whole_numbers_and_the_glyph(widget, qapp):
    from qtpy.QtWidgets import QPushButton

    reset_add(widget)
    widget.update_add_ui_state(box_active=True)
    qapp.processEvents()
    try:
        row = widget._add_box_container
        texts = [b.text() for b in row.findChildren(QPushButton)
                 if not b.isHidden()]
        assert texts == ["S", "M", "L", "Whole", "Numbers…", "⌖"]
        assert widget.add_recenter_btn.text() == "⌖"
        assert "Recenter" in widget.add_recenter_btn.toolTip()
        # never wider than the row was (offscreen 269 px with "⌖ Recenter")
        assert row.minimumSizeHint().width() <= 269
    finally:
        _done(widget)


def test_the_box_row_whole_and_the_idle_button_fire_whole_patch(widget):
    reset_add(widget)
    fired = []
    widget.set_add_callbacks(on_whole_patch=lambda: fired.append(1))
    try:
        widget.update_add_ui_state(box_active=False)
        widget.add_whole_btn.click()
        widget.update_add_ui_state(box_active=True)
        widget.add_size_whole_btn.click()
    finally:
        _done(widget)
    assert fired == [1, 1]


def test_the_numbers_window_has_no_whole_patch(widget):
    from qtpy.QtWidgets import QPushButton

    assert not hasattr(widget, "add_whole_numbers_btn")
    assert widget._add_numbers_dialog.findChildren(QPushButton) == []


def test_review_footer_is_done_and_undo_all(widget):
    reset_add(widget)
    ids = list(range(400, 404))
    widget.update_add_ui_state(box_active=True, review_active=True,
                               piece_ids=ids,
                               piece_sizes={i: 10 for i in ids})
    try:
        assert widget.add_accept_btn.text() == "Done"
        assert widget.add_accept_btn.isEnabled()
        assert widget.add_reject_btn.text() == "Undo all"
        assert widget.add_reject_btn.isEnabled()
        from napari_epicycle.annotator.advanced_features_widget import (
            _REFINEMENT_ACCEPT_STYLE,
            _REFINEMENT_REJECT_STYLE,
        )

        assert widget.add_accept_btn.styleSheet() == _REFINEMENT_ACCEPT_STYLE
        assert widget.add_reject_btn.styleSheet() == _REFINEMENT_REJECT_STYLE
        assert widget._add_footer_summary.text() == AP.foot_review(4)
        # Undo all left the card; its three buttons share one row
        assert not hasattr(widget, "add_undo_all_btn")
        row = widget.add_again_btn.parentWidget()
        assert row is widget.add_batch_refine_btn.parentWidget()
        assert row is widget.add_refine_piece_btn.parentWidget()
        assert row is widget._add_review_btn_row
    finally:
        _done(widget)
    assert widget.add_accept_btn.text() == "Accept"


def test_the_secondary_words_and_lines(widget):
    reset_add(widget)
    try:
        # idle: grey, the word it takes once a box is up
        widget.update_add_ui_state(box_active=False, reject_enabled=False)
        assert widget.add_reject_btn.text() == "Close box"
        assert not widget.add_reject_btn.isEnabled()
        assert widget._add_footer_summary.text() == AP.ADD_FOOT_IDLE
        widget.update_add_ui_state(box_active=True, reject_enabled=True)
        assert widget.add_reject_btn.text() == "Close box"
        assert widget._add_footer_summary.text() == AP.ADD_FOOT_BOX
        # over a result: Discard, and the line is on screen (no third button)
        ids = [1, 2]
        widget.update_add_ui_state(candidate_ids=ids,
                                   candidate_sizes={1: 5, 2: 5},
                                   candidate_edge_ids=set(),
                                   footer_text=AP.ADD_FOOT_CANDIDATES)
        assert widget.add_reject_btn.text() == "Discard"
        assert not widget._add_footer_summary.isHidden()
        assert widget._add_footer_summary.text() == AP.ADD_FOOT_CANDIDATES
        assert not hasattr(widget, "add_accept_refine_btn")
        widget.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                                   candidate_edge_ids=set(), footer_text="")
        assert widget.add_reject_btn.text() == "Close box"
    finally:
        _done(widget)


def test_the_secondary_is_grey_while_busy(widget):
    reset_add(widget)
    widget.update_add_ui_state(box_active=True, reject_enabled=True)
    try:
        widget.update_add_ui_state(busy=True)
        assert not widget.add_reject_btn.isEnabled()
        widget.update_add_ui_state(busy=False)
        assert widget.add_reject_btn.isEnabled()
    finally:
        _done(widget)


def test_edge_rows_hide_on_a_whole_patch(widget):
    reset_add(widget)
    widget.update_add_ui_state(box_active=True)
    try:
        assert not widget.add_clear_border_cb.isHidden()
        assert not widget._add_edge_zone_row.isHidden()
        widget.update_add_ui_state(whole_patch=True)
        assert widget.add_clear_border_cb.isHidden()
        assert widget._add_edge_zone_row.isHidden()
        assert not hasattr(AP, "ADD_EDGE_WHOLE"), "hidden, not grey"
    finally:
        _done(widget)


@pytest.mark.parametrize("index,ball,shrink", [(0, False, True),
                                               (1, False, False),
                                               (2, True, False)])
def test_each_marker_source_shows_its_own_advanced_row(widget, index, ball,
                                                      shrink):
    reset_add(widget)
    try:
        _cells(widget, index)
        widget.update_add_ui_state(box_active=True)
        # the ball sits beside Place markers in the Markers card now, so
        # it shows and hides with the click rows, never in Advanced
        assert widget._add_click_radius_row.parentWidget() is (
            widget._add_click_rows)
        assert widget._add_click_rows.isHidden() is not ball
        assert widget._add_core_erode_row.isHidden() is not shrink
        title = widget._add_advanced_toggle.text()
        assert "ball" not in title.lower()
        assert ("shrink " in title) is shrink
    finally:
        _done(widget)


@pytest.mark.parametrize("preset", ["nuclei", "cells"])
def test_advanced_waits_for_a_box(widget, preset):
    reset_add(widget)
    try:
        if preset == "cells":
            _cells(widget)
        widget.update_add_ui_state(box_active=False)
        assert widget._add_advanced_toggle.isHidden()
        assert widget._add_advanced_container.isHidden()
        widget.update_add_ui_state(box_active=True)
        assert not widget._add_advanced_toggle.isHidden()
    finally:
        _done(widget)


def test_the_footer_does_not_move_and_fits_the_live_width(widget, qapp):
    """Every state's two buttons on one row within the 312 px live width,
    the footer the same height in each (its line has a fixed slot)."""
    reset_add(widget)
    widget.resize(340, 350)
    widget.show()
    qapp.processEvents()
    try:
        widget.focus_tab("Add")
        footer = widget._panel_slots["Add"]["footer"]
        heights, words = [], []
        ids = list(range(400, 412))
        for push in (dict(box_active=False, reject_enabled=False),
                     dict(box_active=True, reject_enabled=True),
                     dict(candidate_ids=ids,
                          candidate_sizes={i: 1234 for i in ids},
                          candidate_edge_ids=set(),
                          footer_text=AP.ADD_FOOT_CANDIDATES),
                     dict(candidate_ids=[], candidate_sizes={},
                          candidate_edge_ids=set(), review_active=True,
                          piece_ids=ids,
                          piece_sizes={i: 1234 for i in ids})):
            widget.update_add_ui_state(**push)
            qapp.processEvents()
            words.append((widget.add_accept_btn.text(),
                          widget.add_reject_btn.text()))
            heights.append(footer.height())
            assert footer.minimumSizeHint().width() <= 312, words[-1]
        assert words == [("Accept", "Close box"), ("Accept", "Close box"),
                         ("Accept (12)", "Discard"), ("Done", "Undo all")]
        assert len(set(heights)) == 1, heights
    finally:
        _done(widget)
        widget.hide()


# --------------------------------------------------------------------------- #
# review A: a Clear after "+ Add another", the whole patch's read, the
# REVIEW list, the words
# --------------------------------------------------------------------------- #
@pytest.fixture
def fakes(monkeypatch):
    install_fakes(monkeypatch)


def _session_then_clear(h):
    """An earlier Accept (object 77), then a Clear in this box -- the box
    a "+ Add another" opened."""
    from .test_add_clear import _clear

    h._add_session = [{"ids": [77], "accept_ids": [77], "sizes": {77: 1}}]
    h._add_piece_ids = [77]
    _clear(h)
    assert h._add_session_clear_seqs


def test_a_clear_in_a_session_box_says_close_box_goes_back(fakes):
    from .test_add_clear import _harness, _volume

    h = _harness(_volume())
    _session_then_clear(h)
    # Close box goes back to the review (not "removes the box"), and the
    # Clear stays
    assert h.adv.last("footer_text") == AP.foot_box_session(1, cleared=True)
    assert "Close box goes back" in AP.foot_box_session(1, cleared=True)
    h._add_refresh_ui()
    assert h.adv.last("footer_text") == AP.foot_box_session(1, cleared=True)


def test_a_clear_above_the_accepts_is_said_in_review_and_the_confirm(fakes):
    from .test_add_clear import _harness, _volume

    h = _harness(_volume())
    _session_then_clear(h)
    h._add_secondary()                          # Close box -> REVIEW
    assert h._add_review
    assert h.adv.last("footer_text") == AP.foot_review(1, blocked=True)
    seen = []
    h._add_confirm = lambda text: (seen.append(text), False)[1]
    h._undo_add_all()
    assert seen and AP.ADD_CONFIRM_UNDO_ALL_BLOCKED in seen[0]
    assert h._add_review, "kept: the answer was no"


def test_a_clear_before_the_first_accept_blocks_nothing(fakes):
    from .test_add_clear import _clear, _harness, _volume

    h = _harness(_volume())
    _clear(h)                                   # the first box, no Accept
    assert not h._add_clear_above_accepts()
    assert h.adv.last("footer_text") == AP.ADD_FOOT_BOX_CLEARED


def test_the_whole_patch_on_anywhere_reads_no_brightness_crop(h):
    """Preview is refused there anyway: no float copy of the whole patch on
    the main thread at every box-info request."""
    h.adv.look = (AP.LOOK_ANYWHERE, None)
    h._start_add_whole()
    kw = h._add_gather_info()
    assert kw["image_roi"] is None
    h._set_add_box_size_fraction(0.25)          # a box: the crop is read
    assert h._add_gather_info()["image_roi"] is not None


def test_a_cells_whole_patch_without_a_run_reads_no_nuclei(h):
    h.adv.preset = "cells"
    h.adv.source = ("nuclei_masks", None)
    h._start_add_whole()
    assert "nuclei_source" not in h._add_gather_info()


def test_review_lists_every_object_of_the_session(h):
    """An Accept in a later box lists the earlier boxes' objects too."""
    h.place_box((slice(2, 10), slice(2, 18), slice(2, 18)))
    res = np.zeros((8, 16, 16), np.int32)
    res[1:3, 1:4, 1:4] = 40
    res[5:7, 9:12, 9:12] = 41
    h._add_preview_result = res
    h._add_preview_roi = tuple(h._add_roi)
    h.adv.checked = [40, 41]
    h._on_add_committed_cb = None
    h._add_session = [{"ids": [7], "sizes": {7: 5}}]   # an earlier box
    h._accept_add()
    assert h.adv.last("piece_ids") == [7, 40, 41]


def test_two_boxes_then_close_box_lists_both(h):
    _to_review(h)                               # object 40
    h._add_another()
    res = np.zeros((8, 16, 16), np.int32)
    res[5:7, 9:12, 9:12] = 41
    h._add_preview_result = res
    h._add_preview_roi = tuple(h._add_roi)
    h.adv.checked = [41]
    h._accept_add()
    assert h.adv.last("piece_ids") == [40, 41]
    h._add_another()
    h._add_secondary()                          # Close box -> REVIEW
    assert h._add_review
    assert h.adv.last("piece_ids") == [40, 41]
    assert len(h._add_session) == 2


def test_the_review_line_with_nothing_left():
    assert AP.foot_review(0) == AP.ADD_FOOT_DONE_NONE
    assert "0 objects" not in AP.foot_review(0)


def test_the_idle_line_without_spread_on_cells_offers_no_whole_patch():
    s = {"has_box": False, "preset": "cells", "numba_ok": False}
    assert AP.preview_block(s) == AP.ADD_S1_IDLE_BOX
    assert "Whole patch" not in AP.ADD_S1_IDLE_BOX
    assert AP.s1_kind(s) == "idle"
    # Nuclei and Cells that can spread keep the Whole patch line
    assert AP.preview_block(dict(s, numba_ok=True)) == AP.ADD_S1_IDLE
    assert AP.preview_block(dict(s, preset="nuclei")) == AP.ADD_S1_IDLE


def test_every_add_footer_sentence_fits_its_two_line_slot(widget, qapp):
    """The new lines at their longest (a five-digit count) fit the footer's
    fixed two-line slot at the dock's width."""
    reset_add(widget)
    widget.resize(340, 350)
    widget.show()
    qapp.processEvents()
    try:
        widget.focus_tab("Add")
        qapp.processEvents()
        lbl = widget._add_footer_summary
        # at most the 312 px live width (offscreen the label can come up
        # wider than it is live)
        width, reserved = min(lbl.width() or 312, 312), lbl.height()
        texts = [AP.ADD_FOOT_IDLE, AP.ADD_FOOT_BOX, AP.ADD_FOOT_BOX_CLEARED,
                 AP.ADD_FOOT_CANDIDATES, AP.foot_review(0)]
        for n in (1, 12345):
            texts += [AP.foot_box_session(n), AP.foot_review(n),
                      AP.foot_box_session(n, cleared=True),
                      AP.foot_review(n, blocked=True)]
        was = lbl.text()
        try:
            for text in texts:
                # the real label, at its own width and style
                lbl.setText(text)
                need = lbl.heightForWidth(width)
                assert need <= reserved, (text, need, width, reserved)
        finally:
            lbl.setText(was)
    finally:
        widget.hide()


def test_the_secondarys_tooltip_follows_its_state(widget):
    """It never names a word the button does not show: the tooltip is the
    footer line in every state but REVIEW (Undo all keeps its own)."""
    reset_add(widget)
    try:
        widget.update_add_ui_state(box_active=True, reject_enabled=True)
        assert widget.add_reject_btn.toolTip() == AP.ADD_FOOT_BOX
        widget.update_add_ui_state(footer_text=AP.foot_box_session(3))
        assert widget.add_reject_btn.toolTip() == AP.foot_box_session(3)
        widget.update_add_ui_state(candidate_ids=[1],
                                   candidate_sizes={1: 5},
                                   candidate_edge_ids=set())
        assert widget.add_reject_btn.toolTip() == AP.ADD_FOOT_CANDIDATES
        widget.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                                   candidate_edge_ids=set(),
                                   review_active=True, piece_ids=[1],
                                   piece_sizes={1: 5})
        assert widget.add_reject_btn.toolTip() == AP.ADD_UNDO_ALL_TIP
    finally:
        _done(widget)

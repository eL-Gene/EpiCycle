"""A4 -- "Accept and refine ▸".

One footer button while candidates are up: Accept exactly as Accept does,
then Batch Refine on "Added this session (n)" exactly as "Refine all ▸"
does -- the two shipped paths, one after the other.  Refused with
``ADD_BATCH_BUSY`` like the other doors; a refused Accept hands nothing off.
The footer is outside the scroll area, so it must not grow: MEASURED, the
three buttons need 381 px on one row (319 at 3 px padding) against 312, and
a second footer line shrank the viewport 245 -> 220 px and put 7a+ / 7d
over 3.1 -- so the button takes the footer summary's fixed two-line SLOT
while candidates are up, and the footer never moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401


# --------------------------------------------------------------------------- #
# the panel
# --------------------------------------------------------------------------- #
def _candidates(w, n=12):
    ids = list(range(400, 400 + n))
    w.update_add_ui_state(box_active=True, review_active=False)
    w.update_add_ui_state(candidate_ids=ids,
                          candidate_sizes={i: 1234 for i in ids},
                          candidate_edge_ids=set())
    return ids


def _clear(w):
    w.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                          candidate_edge_ids=set(), box_active=False)


def test_the_button_sits_in_the_footer_summary_slot(widget):
    from qtpy.QtWidgets import QPushButton

    btn = widget.add_accept_refine_btn
    assert isinstance(btn, QPushButton)
    assert btn.text() == AP.ADD_ACCEPT_REFINE_BTN == "Accept and refine ▸"
    footer = widget._panel_slots["Add"]["footer"]
    slot = widget._add_footer_slot
    assert slot.parentWidget() is footer
    assert btn.parentWidget() is slot
    assert widget._add_footer_summary.parentWidget() is slot


def test_it_shows_only_with_candidates_and_follows_accept(widget):
    reset_add(widget)
    _clear(widget)
    assert widget.add_accept_refine_btn.isHidden(), "box state: hidden"
    assert not widget._add_footer_summary.isHidden()
    _candidates(widget)
    assert not widget.add_accept_refine_btn.isHidden()
    assert widget._add_footer_summary.isHidden(), "the button has the slot"
    assert widget.add_accept_btn.toolTip() == AP.ADD_FOOT_CANDIDATES, (
        "the slot's sentence rides the Accept tooltip")
    assert widget.add_accept_refine_btn.isEnabled()
    widget._set_all_add_checks(False)
    assert not widget.add_accept_btn.isEnabled()
    assert not widget.add_accept_refine_btn.isEnabled(), "exactly as Accept"
    widget._set_all_add_checks(True)
    assert widget.add_accept_refine_btn.isEnabled()
    widget.update_add_ui_state(review_active=True)
    assert widget.add_accept_refine_btn.isHidden(), "REVIEW: hidden"
    widget.update_add_ui_state(review_active=False)
    _clear(widget)
    assert widget.add_accept_refine_btn.isHidden()


def test_its_help_line(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    assert ADD_HELP_LINES["add_accept_refine_btn"] == AP.ADD_ACCEPT_REFINE_TIP
    assert widget.add_accept_refine_btn.toolTip() == AP.ADD_ACCEPT_REFINE_TIP


def test_it_is_wired_through_the_frozen_setter(widget):
    got = []
    widget.set_add_callbacks(on_accept_refine=lambda *_a: got.append(1))
    _candidates(widget)
    widget.add_accept_refine_btn.click()
    assert got == [1]
    _clear(widget)


def test_the_footer_does_not_move_and_fits_the_live_width(widget, qapp):
    """Measured with "Accept (12)" and "Discard"."""
    widget.resize(340, 350)
    widget.show()
    qapp.processEvents()
    try:
        widget.focus_tab("Add")
        _clear(widget)
        widget.update_add_ui_state(box_active=True)
        qapp.processEvents()
        footer = widget._panel_slots["Add"]["footer"]
        h0 = footer.height()
        _candidates(widget, 12)
        qapp.processEvents()
        assert widget.add_accept_btn.text() == "Accept (12)"
        assert widget.add_reject_btn.text() == "Discard"
        need = footer.minimumSizeHint().width()
        print("[ADD-FOOTER] {} px wide, {} -> {} px high (live width "
              "312)".format(need, h0, footer.height()))
        assert need <= 312, need
        assert footer.height() == h0, "the footer never moves"
        btn = widget.add_accept_refine_btn
        assert btn.geometry().bottom() < widget._add_footer_slot.height(), (
            "the button fits the reserved slot")
    finally:
        _clear(widget)
        widget.hide()


# --------------------------------------------------------------------------- #
# the orchestrator
# --------------------------------------------------------------------------- #
@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((20, 30, 30), np.int32), images={})
    hh.place_box((slice(2, 18), slice(2, 28), slice(2, 28)))
    res = np.zeros((16, 26, 26), np.int32)
    res[2:6, 2:6, 2:6] = 40
    res[8:12, 15:19, 15:19] = 41
    hh._add_preview_result = res
    hh._add_preview_roi = tuple(hh._add_roi)
    hh.adv.checked = [40, 41]
    hh._on_add_committed_cb = None
    hh._add_session = [{"ids": [7], "sizes": {7: 5}}]    # an earlier box
    hh._batch_running = False
    hh._batch_staged = False
    hh.opened = []

    def _open(explicit_targets=None, default_target=None):
        hh.opened.append((list(explicit_targets), default_target))
        return None

    hh._open_batch_refine_dialog = _open
    return hh


def test_it_accepts_then_hands_the_whole_session_off(h):
    h._accept_add_and_refine()
    assert h._add_review, "Accept ran and reached REVIEW"
    assert h.sync_manager.shared_labels["L"].max() == 41, "written"
    assert h.opened == [([7, 40, 41], "selected")], (
        "Batch Refine on Added this session, exactly as Refine all")


def test_a_running_batch_refuses_before_accept(h):
    from packages.epicycle_annotator.batch_refine import ADD_BATCH_BUSY

    h._batch_running = True
    h._accept_add_and_refine()
    assert not h._add_review and h._add_preview_result is not None
    assert h.sync_manager.shared_labels["L"].max() == 0, "nothing written"
    assert h.opened == []
    assert h.adv.last("status_text") == ADD_BATCH_BUSY


def test_a_refused_accept_hands_nothing_off(h):
    h.adv.checked = []                        # TICK_ONE
    h._accept_add_and_refine()
    assert not h._add_review
    assert h.opened == []
    assert h._add_transient == AP.ADD_S1_TICK_ONE


def test_a_failed_commit_hands_nothing_off(h):
    def boom(*_a, **_k):
        raise RuntimeError("x")

    h._on_add_committed_cb = boom
    h._accept_add_and_refine()
    assert h.opened == [], "the session did not grow: nothing to refine"

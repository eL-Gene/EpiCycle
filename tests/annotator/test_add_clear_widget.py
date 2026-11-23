"""The Clear objects button on the Add page (the panel side).

* one menu button in the pinned row beside Preview, Qt's own arrow (no
  arrow character in the text), two items;
* shown only while a box is placed, no candidates are up and not in the
  review; disabled while busy (the count is busy too);
* each item fires the orchestrator's callback with its mode.
"""

from __future__ import annotations

from packages.epicycle_annotator import add_preview as AP

from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401


def _visible(w):
    return not w.add_clear_btn.isHidden()


def test_the_button_sits_in_the_pinned_row_before_preview(widget):
    btn = widget.add_clear_btn
    assert btn.text() == AP.ADD_CLEAR_BTN
    assert "▾" not in btn.text() and "▸" not in btn.text()
    row = widget.add_preview_btn.parentWidget().layout()
    assert row.indexOf(btn) == 0
    assert row.indexOf(widget.add_preview_btn) == 1
    items = [a.text() for a in btn.menu().actions()]
    assert items == [AP.ADD_CLEAR_ITEM_BOX, AP.ADD_CLEAR_ITEM_PATCH]
    assert btn.toolTip() == AP.ADD_CLEAR_TIP


def test_shown_with_a_box_only_and_never_over_candidates_or_review(widget):
    reset_add(widget)
    widget.update_add_ui_state(box_active=False, review_active=False)
    assert not _visible(widget)
    widget.update_add_ui_state(box_active=True)
    assert _visible(widget)
    widget.update_add_ui_state(candidate_ids=[5, 6],
                               candidate_sizes={5: 10, 6: 12},
                               candidate_edge_ids=set())
    assert not _visible(widget)
    widget.update_add_ui_state(candidate_ids=[])
    assert _visible(widget)
    widget.update_add_ui_state(review_active=True)
    assert not _visible(widget)
    widget.update_add_ui_state(review_active=False, box_active=False)
    assert not _visible(widget)


def test_disabled_while_busy(widget):
    reset_add(widget)
    widget.update_add_ui_state(box_active=True, review_active=False)
    widget.update_add_ui_state(busy=True, stop_enabled=False)
    assert not widget.add_clear_btn.isEnabled()
    widget.update_add_ui_state(busy=False)
    assert widget.add_clear_btn.isEnabled()
    widget.update_add_ui_state(box_active=False)


def test_each_item_fires_its_mode(widget):
    got = []
    widget.set_add_callbacks(on_clear=got.append)
    widget.add_clear_box_action.trigger()
    widget.add_clear_patch_action.trigger()
    assert got == ["box", "patch"]
    # a second wiring replaces the callback; nothing fires twice
    got2 = []
    widget.set_add_callbacks(on_clear=got2.append)
    widget.add_clear_patch_action.trigger()
    assert got == ["box", "patch"] and got2 == ["patch"]

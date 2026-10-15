"""The candidate / REVIEW rows.

"Object {id}", the chips edge / small in one line, the size right-aligned
mono; chips elide right-to-left before the id or the size are touched; the
lists show ``ADD_LIST_ROWS`` rows and scroll.
"""

from __future__ import annotations

import pytest

from packages.epicycle_annotator.add_preview import ADD_LIST_ROWS

from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401


def test_the_widest_row_keeps_the_id_and_the_size_at_316px(widget):
    from qtpy.QtGui import QFont, QFontMetrics

    from napari_epicycle.annotator.advanced_features_widget import (
        AddCandidateDelegate,
        add_row_layout,
    )

    font = widget.add_candidate_list.font()
    fm = QFontMetrics(font)
    mono = QFont(font)
    mono.setFamily("Consolas")
    fmm = QFontMetrics(mono)
    name_w = fm.horizontalAdvance("Object 65535") + 8
    tail_w = fmm.horizontalAdvance("1,234,567") + 10
    chips = ["edge", "small"]
    chip_ws = [fm.horizontalAdvance(c) + AddCandidateDelegate.CHIP_PAD
               for c in chips]
    # the text rect of a 316 px list, less the check box and the swatch
    text_w = 316 - 2 * 20 - 8
    n = add_row_layout(text_w, name_w, chip_ws, tail_w)
    used = name_w + tail_w + sum(w + 6 for w in chip_ws[:n])
    assert name_w + tail_w <= text_w, "the id and the size must both fit"
    assert used <= text_w + 6
    assert 0 <= n <= 2


def test_chips_elide_right_to_left_before_the_id():
    from napari_epicycle.annotator.advanced_features_widget import (
        add_row_layout,
    )

    assert add_row_layout(200, 80, [30, 30], 60) == 1
    assert add_row_layout(300, 80, [30, 30], 60) == 2
    assert add_row_layout(140, 80, [30, 30], 60) == 0


def test_rows_read_object_id_and_flagged_start_unticked(widget):
    reset_add(widget)
    widget.update_add_ui_state(candidate_small_ids={7})
    widget.update_add_ui_state(candidate_ids=[5, 7, 9],
                               candidate_sizes={5: 10, 7: 2, 9: 1234},
                               candidate_edge_ids={5})
    texts = [widget.add_candidate_list.item(i).text()
             for i in range(widget.add_candidate_list.count())]
    assert texts == ["Object 5  ·  10", "Object 7  ·  2",
                     "Object 9  ·  1,234"]
    assert widget.get_checked_add_pieces() == [9]
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_CHIPS_ROLE,
    )

    assert widget.add_candidate_list.item(0).data(ADD_CHIPS_ROLE) == ["edge"]
    assert widget.add_candidate_list.item(1).data(ADD_CHIPS_ROLE) == ["small"]


def test_the_review_list_shows_five_rows_and_scrolls_at_fifty(widget):
    reset_add(widget)
    ids = list(range(100, 150))
    widget.update_add_ui_state(review_active=True, piece_ids=ids,
                               piece_sizes={i: 1000 + i for i in ids})
    lst = widget.add_pieces_list
    widget.resize(340, 700)
    widget.focus_tab("Add")
    widget.show()
    from qtpy.QtWidgets import QApplication

    QApplication.processEvents()
    lst.doItemsLayout()
    QApplication.processEvents()
    rows = lst.height() // max(1, lst.sizeHintForRow(0))
    assert rows == ADD_LIST_ROWS
    assert lst.verticalScrollBar().maximum() > 0
    assert lst.item(0).text() == "Object 100  ·  1,100"
    widget.update_add_ui_state(review_active=False)
    widget.hide()


def test_the_delegate_paints(widget):
    reset_add(widget)
    widget.update_add_ui_state(candidate_small_ids={2})
    widget.update_add_ui_state(candidate_ids=[1, 2],
                               candidate_sizes={1: 5000, 2: 3},
                               candidate_edge_ids={1})
    widget.resize(340, 700)
    pm = widget.add_candidate_list.grab()
    assert not pm.isNull()


# --------------------------------------------------------------------------- #
# A5 -- the click-marker list
# --------------------------------------------------------------------------- #
def _cells_clicks(widget):
    from packages.epicycle_annotator import add_preview as AP

    reset_add(widget)
    widget.set_add_preset("cells", silent=True)
    widget.update_add_ui_state(marker_sources=[
        (AP.ADD_ITEM_NO_RUN, False, ("labels", None)),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))], box_active=True)
    widget.add_marker_source_combo.setCurrentIndex(2)
    widget.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                               candidate_edge_ids=set(), review_active=False)


def test_the_widest_marker_row_keeps_the_id_and_the_position_at_312px(widget):
    """"Object 65535" and "1234 / 1234 / 1234" both fit
    the 312 px live viewport's list row, with the check box and swatch."""
    from qtpy.QtGui import QFont, QFontMetrics

    font = widget.add_click_list.font()
    fm = QFontMetrics(font)
    mono = QFont(font)
    mono.setFamily("Consolas")
    fmm = QFontMetrics(mono)
    name_w = fm.horizontalAdvance("Object 65535") + 8
    tail_w = fmm.horizontalAdvance("1234 / 1234 / 1234") + 10
    # the Markers card is indented once inside the 312 px viewport
    text_w = 312 - 16 - 2 * 20 - 8
    assert name_w + tail_w <= text_w, (name_w, tail_w, text_w)


def test_marker_rows_read_id_and_position_and_carry_the_id(widget):
    from qtpy.QtCore import Qt

    from packages.epicycle_annotator import add_preview as AP

    _cells_clicks(widget)
    widget.update_add_ui_state(click_count=3, click_markers=[
        (8, (12, 20, 20), True), (9, (12, 10, 20), False),
        (10, (12, 20, 10), True)])
    lst = widget.add_click_list
    assert [lst.item(i).text() for i in range(lst.count())] == [
        AP.marker_row_text(8, (12, 20, 20)),
        AP.marker_row_text(9, (12, 10, 20)),
        AP.marker_row_text(10, (12, 20, 10))]
    assert [lst.item(i).data(Qt.UserRole) for i in range(3)] == [8, 9, 10]
    assert [lst.item(i).checkState() == Qt.Checked for i in range(3)] == [
        True, False, True]
    assert widget.add_click_count_label.text() == "3 placed · 2 ticked"
    assert widget.add_markers_btn.property("_epicycle_chevron_title") == (
        "Markers: clicks · 3 placed · 2 ticked")
    widget.update_add_ui_state(click_count=0, click_markers=[])


def test_a_user_tick_reports_the_id_once(widget):
    from qtpy.QtCore import Qt

    _cells_clicks(widget)
    seen = []
    widget.set_add_callbacks(on_click_tick=lambda i, t: seen.append((i, t)))
    widget.update_add_ui_state(click_count=2, click_markers=[
        (8, (1, 2, 3), True), (9, (4, 5, 6), True)])
    widget.add_click_list.item(1).setCheckState(Qt.Unchecked)
    assert seen == [(9, False)], "one report per tick (the restyle is quiet)"
    # a push from the orchestrator reports nothing
    widget.update_add_ui_state(click_count=2, click_markers=[
        (8, (1, 2, 3), True), (9, (4, 5, 6), False)])
    assert seen == [(9, False)]
    widget.update_add_ui_state(click_count=0, click_markers=[])


def test_the_marker_list_is_shown_only_for_clicks_before_a_result(widget):
    from qtpy.QtWidgets import QApplication

    _cells_clicks(widget)
    widget.resize(340, 900)
    widget.focus_tab("Add")
    widget.show()
    QApplication.processEvents()
    lst = widget.add_click_list
    widget.update_add_ui_state(click_count=0, click_markers=[])
    assert lst.isHidden(), "no marker: no list"
    widget.update_add_ui_state(click_count=1,
                               click_markers=[(8, (1, 2, 3), True)])
    assert not lst.isHidden()
    widget.update_add_ui_state(candidate_ids=[8], candidate_sizes={8: 10},
                               candidate_edge_ids=set())
    assert lst.isHidden(), "S5's list is the list over candidates (H2)"
    widget.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                               candidate_edge_ids=set())
    assert not lst.isHidden(), "back after Discard, ticks and all"
    assert lst.count() == 1
    widget.add_marker_source_combo.setCurrentIndex(1)
    assert lst.isHidden(), "the clicks source only"
    widget.add_marker_source_combo.setCurrentIndex(2)
    widget.set_add_preset("nuclei", silent=True)
    assert lst.isHidden() or not lst.isVisibleTo(widget)
    widget.update_add_ui_state(click_count=0, click_markers=[])
    widget.hide()


def test_the_marker_list_shows_three_rows_and_scrolls(widget):
    from qtpy.QtWidgets import QApplication

    from packages.epicycle_annotator.add_preview import ADD_MARKER_LIST_ROWS

    _cells_clicks(widget)
    widget.resize(340, 900)
    widget.focus_tab("Add")
    widget.show()
    widget.update_add_ui_state(click_count=12, click_markers=[
        (100 + i, (i, i, i), True) for i in range(12)])
    QApplication.processEvents()
    lst = widget.add_click_list
    lst.doItemsLayout()
    QApplication.processEvents()
    assert lst.height() // max(1, lst.sizeHintForRow(0)) == ADD_MARKER_LIST_ROWS
    assert lst.verticalScrollBar().maximum() > 0
    widget.update_add_ui_state(click_count=0, click_markers=[])
    widget.hide()


@pytest.mark.parametrize("attr", ["add_click_list", "add_candidate_list"])
def test_both_lists_never_take_the_focus_or_a_selection(widget, attr):
    """Ticking a row must not take the keyboard off the
    side view -- Shift+arrow keeps nudging the box."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QAbstractItemView

    lst = getattr(widget, attr)
    assert lst.focusPolicy() == Qt.NoFocus
    assert lst.selectionMode() == QAbstractItemView.NoSelection

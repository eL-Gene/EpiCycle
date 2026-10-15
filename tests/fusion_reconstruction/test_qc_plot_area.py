"""The QC plot area host.

The source table (one assertion per source), the lock
(each control proven LIVE first, the Step 4 R1-1 lesson), the reset door,
the tab count, the card helpers, the see-line, the status kinds and the
divider's floors.
"""

from __future__ import annotations

import pytest

from tests.fusion_reconstruction import _qc_plot_snap as S


def _area():
    S.app()
    from napari_epicycle.fusion_reconstruction._qc_plot_area import QCPlotArea

    a = QCPlotArea()
    a.resize(640, 760)
    a.show()
    _pump()
    return a


def _pump(n=3):
    from qtpy.QtWidgets import QApplication

    for _ in range(n):
        QApplication.processEvents()


@pytest.fixture
def area():
    a = _area()
    yield a
    a.hide()
    a.deleteLater()
    _pump()


@pytest.fixture
def T():
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T

    return T


def _load(a, df=None):
    """What the controller's load path does: the plot, the histogram and
    the area's table."""
    df = S.five_tier_df() if df is None else df
    a.qc_plot_widget.set_features(df)
    a.panel.set_data(df)
    a.set_dataframe(df)
    a.set_enabled_state(True, False, False)
    _pump()
    return df


def _select(a, ids):
    a.qc_plot_widget.highlight_labels(list(ids))
    _pump()


def _active_ids(df):
    return [int(x) for x in df.loc[df["label_status"] == "active", "label"]]


# ---------------------------------------------------------------------------
# Construction: the empty area
# ---------------------------------------------------------------------------

def test_a_new_area_is_the_no_patch_state(area, T):
    assert area.see_label.text() == T.SEE_NO_PATCH
    assert area.current_view() == "histogram"
    assert area.status_label.text() == ""
    assert area.qc_plot_widget._wb_empty_sentence == T.PLOT_NO_PATCH
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        HIST_EMPTY,
    )
    assert area.panel.empty_sentence() == HIST_EMPTY
    assert not area.assign_btn.isEnabled()
    assert not area.undo_btn.isEnabled()
    assert not area.tier_combo.isEnabled()
    assert area.colours_in_viewer_cb.text() == T.COLOURS_IN_VIEWER
    assert area.colours_in_viewer_cb.isChecked()
    # the plot's own Clear lives in the action row now
    assert area.clear_btn.parentWidget() is area.action_row
    assert area.clear_btn.text() == T.ACTION_CLEAR
    assert not area.clear_btn.isEnabled()
    # the toolbar is greyed with nothing to plot, Ghosts ticked
    pw = area.qc_plot_widget
    assert not pw._wb_seg.isEnabled()
    assert not pw._history_cb.isEnabled()
    assert pw._history_cb.isChecked()


def test_the_words_come_from_the_text_module(area, T):
    assert area.tab_histogram.text() == T.TAB_HISTOGRAM
    assert area.tab_selection.text() == T.TAB_SELECTION
    assert area.assign_btn.text() == T.ACTION_ASSIGN
    assert area.undo_btn.text() == T.ACTION_UNDO
    assert area.assign_label.text() == T.ACTION_ASSIGN_LABEL_NONE
    H = T.STEP3_PLOT_HELP
    assert area.tab_histogram.toolTip() == H["Histogram tab"]
    assert area.tab_selection.toolTip() == H["Selection tab"]
    assert area.assign_btn.toolTip() == H["Assign"]
    assert area.undo_btn.toolTip() == H["Undo assign"]
    assert area.clear_btn.toolTip() == H["Clear"]
    assert area.colours_in_viewer_cb.toolTip() == H["Colours in viewer"]
    assert area.panel._feature_combo.toolTip() == H["Triage by"]
    assert area.panel._color_combo.toolTip() == H["Colour bars by"]
    assert area.panel._reset_btn.toolTip() == H["Reset lines"]
    assert area.panel._gate_cb.toolTip() == H["Gate"]
    assert area._splitter.handle(1).toolTip() == H["Divider"]


def test_the_plot_is_the_step3_workbench(area):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP3_SCHEME,
    )
    from napari_epicycle.ui_shell.theme import PANEL_GOLD

    pw = area.qc_plot_widget
    assert pw._layout_mode == "workbench"
    assert pw._scheme is STEP3_SCHEME
    assert pw._palette.accent == PANEL_GOLD
    assert pw._history_cb.text() == "Ghosts"


# ---------------------------------------------------------------------------
# The source table: one assertion per source
# ---------------------------------------------------------------------------

def test_lasso_shows_selection(area):
    df = _load(area)
    _select(area, _active_ids(df)[:5])
    area.selection_changed("lasso")
    assert area.current_view() == "selection"


def test_plot_source_resolved_as_lasso_shows_selection(area):
    df = _load(area)
    _select(area, _active_ids(df)[:5])          # reads as "lasso"
    area.selection_changed("plot")
    assert area.current_view() == "selection"


def test_legend_shows_selection(area, T):
    _load(area)
    area.qc_plot_widget._toggle_category_selection("qc_tier", "good")
    _pump()
    area.selection_changed("plot")
    assert area.current_view() == "selection"
    assert area.card.message()[0].startswith("Legend: good")


def test_kept_across_a_plot_type_switch_shows_selection(area, T):
    df = _load(area)
    _select(area, _active_ids(df)[:4])
    area.selection_changed("echo")               # no switch yet
    assert area.current_view() == "histogram"
    pw = area.qc_plot_widget
    pw._wb_type_buttons["Violin"].click()        # keeps the selection
    _pump()
    assert area.current_view() == "selection"
    assert area.card.message()[0] == T.CARD_MSG_KEPT.format(n="4")


def test_clear_shows_histogram_and_clears_the_status(area, T):
    df = _load(area)
    _select(area, _active_ids(df)[:5])
    area.selection_changed("lasso")
    area.qc_plot_widget.set_status("Viewer: 5 labels shown alone", "info")
    area.qc_plot_widget.highlight_labels([])
    area.selection_changed("clear")
    assert area.current_view() == "histogram"
    assert area.qc_plot_widget.status() == ("", "info")
    assert area.status_label.text() == T.STATUS_IDLE_HINT


def test_plot_source_with_nothing_selected_is_a_clear(area):
    df = _load(area)
    _select(area, _active_ids(df)[:5])
    area.selection_changed("lasso")
    area.qc_plot_widget.highlight_labels([])
    area.selection_changed("plot")
    assert area.current_view() == "histogram"


@pytest.mark.parametrize("source", ["click", "echo", "gate", "refresh"])
def test_no_switch_from_histogram(area, source):
    df = _load(area)
    _select(area, _active_ids(df)[:5])
    area.selection_changed(source)
    assert area.current_view() == "histogram"


@pytest.mark.parametrize("source", ["click", "echo", "gate", "refresh"])
def test_no_switch_from_selection(area, source):
    df = _load(area)
    _select(area, _active_ids(df)[:5])
    area.selection_changed("lasso")
    _select(area, _active_ids(df)[5:8])
    area.selection_changed(source)
    assert area.current_view() == "selection"


def test_an_unknown_source_is_refused(area):
    with pytest.raises(ValueError):
        area.selection_changed("wobble")


def test_the_tabs_can_be_clicked_at_any_time(area):
    _load(area)
    area.tab_selection.click()
    assert area.current_view() == "selection"
    assert area.panel.isHidden() and not area.card.isHidden()
    area.tab_histogram.click()
    assert area.current_view() == "histogram"
    assert not area.panel.isHidden() and area._card_page.isHidden()


def test_a_non_gate_selection_unticks_gate_silently(area):
    df = _load(area)
    p = area.panel
    p._feature_combo.setCurrentText("area")
    _pump()
    heard = []
    p.gated_drag_finished.connect(lambda ids: heard.append(ids))
    p._gate_cb.setChecked(True)
    _pump()
    assert p._gate_cb.isChecked()
    area.selection_changed("gate")
    assert p._gate_cb.isChecked(), "the gate's own push keeps the gate"
    heard.clear()
    _select(area, _active_ids(df)[:3])
    area.selection_changed("lasso")
    assert not p._gate_cb.isChecked()
    assert heard == [], "the untick emits nothing"


def test_the_gate_message_names_the_feature_and_the_band(area, T):
    _load(area)
    p = area.panel
    p._feature_combo.setCurrentText("area")
    _pump()
    p._gate_cb.setChecked(True)
    _pump()
    ids = sorted(p._compute_gated_label_ids())
    area.qc_plot_widget.highlight_labels(ids)
    area.selection_changed("gate")
    msg = area.card.message()[0]
    assert msg.startswith("Gate: ") and "with area between" in msg


# ---------------------------------------------------------------------------
# The card, the tab count, the see-line, the status
# ---------------------------------------------------------------------------

def test_the_tab_count_and_the_card_follow_the_live_selection(area, T):
    df = _load(area)
    ids = _active_ids(df)[:3]
    _select(area, ids)
    area.selection_changed("lasso")
    assert area.tab_selection.text() == T.TAB_SELECTION_COUNT.format(n="3")
    assert area.card.count_label.text() == T.CARD_SELECTED.format(n="3")
    assert area.assign_label.text() == T.ACTION_ASSIGN_LABEL.format(n="3")
    assert area.card.message()[0] == T.CARD_MSG_LASSO.format(n="3")
    assert area.card.has_selection()
    area.qc_plot_widget.highlight_labels([])
    area.selection_changed("clear")
    assert area.tab_selection.text() == T.TAB_SELECTION
    assert area.assign_label.text() == T.ACTION_ASSIGN_LABEL_NONE
    assert not area.card.has_selection()
    assert area.card.none_label.text() == T.CARD_NONE


def test_a_click_names_its_label(area, T):
    df = _load(area)
    lid = _active_ids(df)[2]
    _select(area, [lid])
    area.selection_changed("click")
    assert area.card.message()[0] == T.CARD_MSG_CLICK.format(id=lid)


def test_an_echo_says_where_it_came_from(area, T):
    df = _load(area)
    _select(area, _active_ids(df)[:2])
    area.selection_changed("echo")
    assert area.card.message()[0] == T.CARD_MSG_ECHO.format(n="2")


def test_a_refresh_keeps_the_last_message(area, T):
    df = _load(area)
    _select(area, _active_ids(df)[:2])
    area.selection_changed("echo")
    area.refresh_card()
    area.selection_changed("refresh")
    assert area.card.message()[0] == T.CARD_MSG_ECHO.format(n="2")


def test_a_warn_message_is_a_block(area, T):
    from napari_epicycle.ui_shell.theme import PANEL_WARN_STYLE

    area.set_card_message("Nothing was assigned.", "warn")
    assert area.card.message() == ("Nothing was assigned.", "warn")
    assert area.card.message_label.styleSheet() == PANEL_WARN_STYLE
    with pytest.raises(ValueError):
        area.set_card_message("x", "loud")


def test_the_see_line_counts_labels_and_ghosts(area, T):
    df = _load(area)
    n_ghost = int(df["label_status"].isin(["deleted", "superseded"]).sum())
    n_lab = len(df) - n_ghost
    text = area.see_label.text()
    # labels · selected · ghosts: the least important part last (MED-4)
    assert text == T.SEE_LINE.format(n=f"{n_lab:,}", sel=T.SEE_NONE_SELECTED,
                                     g=f"{n_ghost:,}")
    _select(area, _active_ids(df)[:4])
    area.selection_changed("lasso")
    assert area.see_label.text() == T.SEE_LINE.format(
        n=f"{n_lab:,}", sel=T.SEE_SELECTED.format(n="4"), g=f"{n_ghost:,}")
    # Ghosts off: the see-line still counts them (they are in the table)
    area.qc_plot_widget._history_cb.setChecked(False)
    assert area.see_label.text().endswith(T.SEE_GHOSTS.format(
        n=f"{n_ghost:,}"))


def test_the_status_slot_kinds(area, T):
    from napari_epicycle.fusion_reconstruction import _qc_plot_area as A

    _load(area)
    pw = area.qc_plot_widget
    assert area.status_label.text() == T.STATUS_IDLE_HINT
    pw.set_status("Viewer: 3 labels shown alone", "info")
    assert area.status_label.text() == "Viewer: 3 labels shown alone"
    assert A.STATUS_INFO_COLOUR in area.status_label.styleSheet()
    pw.set_status("careful", "warn")
    assert A.STATUS_WARN_COLOUR in area.status_label.styleSheet()
    pw.set_status(T.STATUS_ISOLATION_FAILED, "error")
    assert A.STATUS_ERROR_COLOUR in area.status_label.styleSheet()
    assert area.status_shown() == (T.STATUS_ISOLATION_FAILED, "error")
    pw.set_status("", "info")
    assert area.status_label.text() == T.STATUS_IDLE_HINT


def test_the_status_slot_takes_at_most_forty_percent(area):
    from napari_epicycle.fusion_reconstruction import _qc_plot_area as A

    area.resize(900, 760)
    _pump()
    assert area.status_label.maximumWidth() == int(
        area.width() * A.STATUS_MAX_SHARE)


def test_ghosts_off_takes_the_histograms_ghost_stack_away(area):
    _load(area)
    area.qc_plot_widget._history_cb.setChecked(False)
    _pump()
    assert area.panel._ghosts_visible is False
    area.qc_plot_widget._history_cb.setChecked(True)
    assert area.panel._ghosts_visible is True


def test_the_toolbar_comes_back_with_a_table(area):
    _load(area)
    pw = area.qc_plot_widget
    assert pw._wb_seg.isEnabled()
    assert pw._history_cb.isEnabled()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_selection_card_parts_lists_five_tiers_and_unassigned_only_when_used():
    from napari_epicycle.fusion_reconstruction._qc_plot_area import (
        selection_card_parts,
    )

    parts = selection_card_parts(["good", "good", "bad"])
    assert [p[0] for p in parts] == ["curated", "good", "review", "bad",
                                     "garbage"]
    assert [p[1] for p in parts] == [0, 2, 0, 1, 0]
    parts = selection_card_parts(["good", "", "whatever", None])
    assert parts[-1][0] == "unassigned" and parts[-1][1] == 3


def test_totals_line_counts_active_tiers_and_ghosts(T):
    from napari_epicycle.fusion_reconstruction._qc_plot_area import totals_line

    df = S.five_tier_df()
    line = totals_line(df)
    n_ghost = int(df["label_status"].isin(["deleted", "superseded"]).sum())
    assert line.startswith("All labels: curated ")
    assert line.endswith(f", plus {n_ghost} ghosts")
    assert "unassigned" in line                      # some tiers are ""
    assert totals_line(None) == ""
    one = df[df["label_status"] != "superseded"].copy()
    one.loc[one["label_status"] == "deleted", "label_status"] = "active"
    one.iloc[0, one.columns.get_loc("label_status")] = "deleted"
    assert totals_line(one).endswith(", plus 1 ghost")
    none = df.assign(label_status="active")
    assert ", plus" not in totals_line(none)


def test_see_line_text_reads_the_counts(T):
    from napari_epicycle.fusion_reconstruction._qc_plot_area import (
        see_line_text,
    )

    assert see_line_text({"labels": 842, "ghosts": 837, "selected": 403,
                          "colour_by": "qc_score"}) == (
        "842 labels · 403 selected · 837 ghosts")
    assert see_line_text({"labels": 3000, "ghosts": 1, "selected": 0,
                          "colour_by": "qc_score"}) == (
        "3,000 labels · none selected · 1 ghost")
    assert see_line_text({"labels": 1, "ghosts": 0, "selected": 1,
                          "colour_by": "area"}) == (
        "1 label · 1 selected · 0 ghosts")
    assert not hasattr(T, "SEE_COLOURED_BY")


def test_card_message_per_kind(T):
    from napari_epicycle.fusion_reconstruction._qc_plot_area import (
        card_message,
    )

    assert card_message("lasso", [1, 2]) == "Lasso: 2 labels."
    assert card_message("lasso", [1]) == "Lasso: 1 label."
    assert card_message("legend", [1, 2], tiers=["review"]) == (
        "Legend: review, 2 labels.")
    assert card_message("click", [7]) == "Click: label 7."
    assert card_message("kept", [1, 2, 3]) == (
        "Kept from the last plot: 3 labels.")
    assert card_message("gate", [1, 2], gate=("area", 1020.0, 3400.0)) == (
        "Gate: 2 labels with area between 1,020 and 3,400.")
    assert card_message("lasso", []) == ""
    assert card_message("refresh", [1]) == ""


# ---------------------------------------------------------------------------
# The lock: every control proven LIVE first
# ---------------------------------------------------------------------------

def test_set_locked_greys_the_histogram_and_the_action_row(area):
    df = _load(area)
    _select(area, _active_ids(df)[:3])
    area.selection_changed("lasso")
    area.set_enabled_state(True, True, True)
    p = area.panel
    # proven live first
    live = [p._feature_combo, p._color_combo, p._reset_btn, p._gate_cb,
            area.tier_combo, area.assign_btn, area.undo_btn, area.clear_btn]
    assert all(w.isEnabled() for w in live), [w.isEnabled() for w in live]
    assert all(ln.movable for ln in p._tier_lines)
    assert p._gate_region.movable
    keep = [area.tab_histogram, area.tab_selection, area._splitter,
            area.colours_in_viewer_cb]
    assert all(w.isEnabled() for w in keep)

    area.set_locked(True)
    assert not any(w.isEnabled() for w in live), [w.isEnabled() for w in live]
    assert not any(ln.movable for ln in p._tier_lines)
    assert not p._gate_region.movable
    assert all(w.isEnabled() for w in keep)
    # a selection change while locked: the Clear proxy re-enables the
    # button itself, the container keeps it grey
    _select(area, _active_ids(df)[3:6])
    area.selection_changed("lasso")
    assert not area.clear_btn.isEnabled()
    assert not area.assign_btn.isEnabled()

    area.set_locked(False)
    assert all(w.isEnabled() for w in live), [w.isEnabled() for w in live]
    assert all(ln.movable for ln in p._tier_lines)


def test_the_panel_lock_survives_a_redraw(area):
    df = _load(area)
    area.set_locked(True)
    area.panel.set_data(df.copy())          # a new df redraws
    assert not area.panel._feature_combo.isEnabled()
    assert area.panel.is_locked()


def test_set_enabled_state_gates(area):
    _load(area)
    area.set_enabled_state(True, False, False)
    assert area.tier_combo.isEnabled()
    assert not area.assign_btn.isEnabled()
    assert not area.undo_btn.isEnabled()
    area.set_enabled_state(True, False, True)
    assert area.undo_btn.isEnabled()
    area.set_enabled_state(False, True, True)
    assert not area.tier_combo.isEnabled()
    assert not area.assign_btn.isEnabled()
    assert not area.undo_btn.isEnabled()


# ---------------------------------------------------------------------------
# The reset door
# ---------------------------------------------------------------------------

def test_reset_empties_everything(area, T):
    df = _load(area)
    _select(area, _active_ids(df)[:5])
    area.selection_changed("lasso")
    area.qc_plot_widget.set_status("Assigned 5 labels to good.", "info")
    area.reset(T.PLOT_NO_FEATURES, T.SEE_NO_FEATURES)
    _pump()
    pw = area.qc_plot_widget
    assert pw._df is None
    assert pw.selected_label_ids() == []
    assert pw.status() == ("", "info")
    assert pw._wb_empty_sentence == T.PLOT_NO_FEATURES
    assert area.see_label.text() == T.SEE_NO_FEATURES
    assert area.status_label.text() == ""
    assert area.current_view() == "histogram"
    assert area.panel.empty_sentence() is not None
    assert area.panel._df is None
    assert not area.card.has_selection()
    assert area.dataframe() is None
    assert not area.assign_btn.isEnabled()
    assert not area.tier_combo.isEnabled()
    assert pw._wb_gradient.isHidden()
    assert not pw._wb_seg.isEnabled()


def test_reset_shows_the_loading_sentence(area, T):
    _load(area)
    area.reset(T.PLOT_NO_PATCH, T.SEE_LOADING.format(patch="p1"))
    assert area.see_label.text() == "Loading p1…"


# ---------------------------------------------------------------------------
# The divider
# ---------------------------------------------------------------------------

def test_the_floors_are_derived(area):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        WB_PLOT_MIN_H,
    )

    tabs = area.tab_histogram.sizeHint().height()
    note = area.panel._note_label
    assert note.isHidden()                  # no data: no note
    expect = (tabs + area._lower.layout().spacing()
              + area.panel.minimumSizeHint().height()
              + note.fontMetrics().height() + area.panel.layout().spacing())
    assert area.lower_minimum_px() == expect
    assert area._lower.minimumHeight() == expect
    w = area._pinned.width()
    assert area.pinned_minimum_px(w) == (area.pinned_chrome_height(w)
                                         + WB_PLOT_MIN_H)
    assert area._pinned.minimumHeight() == area.pinned_minimum_px(w)


def test_the_floor_keeps_room_for_the_note_in_both_states(area):
    before = area.lower_minimum_px()
    _load(area)                              # qc_score: the note shows
    assert not area.panel._note_label.isHidden()
    assert area.lower_minimum_px() == before


def test_a_narrow_area_counts_the_wrapped_toolbar(area):
    wide = area.pinned_chrome_height(900)
    narrow = area.pinned_chrome_height(260)
    assert narrow > wide


def test_reset_divider_gives_the_plot_270_and_the_lower_area_its_floor(area):
    from napari_epicycle.fusion_reconstruction import _qc_plot_area as A

    area.resize(640, 1200)
    _pump()
    area.reset_divider()
    top, bottom = area._splitter.sizes()
    assert top == area.pinned_chrome_height() + A.PLOT_DEFAULT_PX
    assert bottom >= area.lower_minimum_px() + A.LOWER_SLACK_PX
    area.resize(640, 640)
    _pump()
    area.reset_divider()
    top, bottom = area._splitter.sizes()
    assert bottom >= area.lower_minimum_px()
    assert top >= area._pinned.minimumHeight()


def test_a_double_click_on_the_divider_resets_it(area):
    area.resize(640, 1200)
    _pump()
    area._splitter.setSizes([300, 700])
    area._divider_user_moved = True
    area._splitter.handle(1).reset_requested.emit()
    from napari_epicycle.fusion_reconstruction import _qc_plot_area as A

    assert area._splitter.sizes()[0] == (area.pinned_chrome_height()
                                         + A.PLOT_DEFAULT_PX)
    assert not area._divider_user_moved


def test_the_plot_keeps_its_height_when_the_chrome_grows(area):
    area.resize(640, 1200)
    _pump()
    area.reset_divider()
    top0, bottom0 = area._splitter.sizes()
    area._last_chrome -= 20       # as if the chrome grew by 20 px
    area.keep_plot_height()
    top1, bottom1 = area._splitter.sizes()
    assert top1 == top0 + 20 and bottom1 == bottom0 - 20


def test_the_divider_timers_belong_to_the_area(area):
    assert area._keep_timer.parent() is area
    assert area._min_timer.parent() is area



# ---------------------------------------------------------------------------
# The P2 audits (2026-10-06)
# ---------------------------------------------------------------------------

def test_the_selection_tab_count_has_its_separator(area, T):
    df = _load(area)
    _select(area, _active_ids(df)[:3])
    area.selection_changed("lasso")
    assert area.tab_selection.text() == "Selection · 3"


def test_the_assign_label_dims_when_locked(area):
    assert ":disabled" in area.assign_label.styleSheet()


def test_the_legend_carries_the_ghosts_help(area, T):
    assert (area.qc_plot_widget._legend_label.toolTip()
            == T.STEP3_PLOT_HELP["Ghosts legend"])


def test_no_idle_hint_while_locked(area, T):
    _load(area)
    assert area.status_label.text() == T.STATUS_IDLE_HINT
    area.set_locked(True)
    assert area.status_label.text() == ""
    area.set_locked(False)
    assert area.status_label.text() == T.STATUS_IDLE_HINT


def _receipt(area, T):
    area.set_receipt(T.STATUS_UNDONE)
    assert area.status_label.text() == T.STATUS_UNDONE


def test_a_receipt_survives_a_refresh_of_the_same_selection(area, T):
    df = _load(area)
    _select(area, _active_ids(df)[:3])
    area.selection_changed("lasso")
    _receipt(area, T)
    area.selection_changed("refresh")
    area.refresh_card()
    assert area.status_label.text() == T.STATUS_UNDONE


def test_a_receipt_clears_on_a_new_selection(area, T):
    df = _load(area)
    _receipt(area, T)
    _select(area, _active_ids(df)[:3])
    area.selection_changed("lasso")
    assert area.status_label.text() != T.STATUS_UNDONE


def test_a_receipt_clears_on_a_plot_type_change(area, T):
    _load(area)
    _receipt(area, T)
    area.qc_plot_widget._wb_type_buttons["Violin"].click()
    # the receipt is gone; on a violin the idle hint names a violin click
    assert area.status_label.text() == T.STATUS_IDLE_HINT_VIOLIN


def test_a_receipt_clears_on_a_colour_by_change(area, T):
    _load(area)
    _receipt(area, T)
    area.qc_plot_widget.set_colour_by("area")
    assert area.status_label.text() == T.STATUS_IDLE_HINT


def test_a_receipt_clears_on_a_ghosts_toggle(area, T):
    _load(area)
    _receipt(area, T)
    area.qc_plot_widget._history_cb.setChecked(False)
    assert area.status_label.text() == T.STATUS_IDLE_HINT


def test_a_viewer_status_is_not_a_receipt(area, T):
    df = _load(area)
    area.qc_plot_widget.set_status("Viewer: 3 labels shown alone", "info")
    area.qc_plot_widget.set_colour_by("area")
    assert area.status_label.text() == "Viewer: 3 labels shown alone"


def test_the_area_asks_for_the_plots_default_height(area):
    # HIGH-2: the section must give the plot 270 px, not its 150 floor
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction import _qc_plot_area as A
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        WB_PLOT_MIN_H,
    )

    _load(area)
    bare = QWidget.minimumSizeHint(area).height()
    assert area.minimumSizeHint().height() == (
        bare + (A.PLOT_DEFAULT_PX - WB_PLOT_MIN_H) + A.LOWER_SLACK_PX)
    # at exactly that height a reset gives the plot its 270 px
    area.resize(area.width(), area.minimumSizeHint().height())
    _pump()
    area.reset_divider()
    top, bottom = area._splitter.sizes()
    assert top == area.pinned_chrome_height() + A.PLOT_DEFAULT_PX
    assert bottom >= area.lower_minimum_px() + A.LOWER_SLACK_PX
    # the divider's own floor is still the 150 px one
    assert area._pinned.minimumHeight() == area.pinned_minimum_px()


def test_a_shrinking_chrome_grows_a_short_plot(area):
    area.resize(640, 1200)
    _pump()
    area.reset_divider()
    chrome = area.pinned_chrome_height()
    area._splitter.setSizes([chrome + 200, 1200])        # plot 200 px
    area._divider_user_moved = True                      # (a drag)
    top0 = area._splitter.sizes()[0]
    area._last_chrome = chrome + 20                      # the chrome shrank
    area.keep_plot_height()
    assert area._splitter.sizes()[0] == top0             # the plot grew


def test_a_shrinking_chrome_never_grows_the_plot_past_270(area):
    from napari_epicycle.fusion_reconstruction import _qc_plot_area as A

    area.resize(640, 1200)
    _pump()
    area.reset_divider()
    chrome = area.pinned_chrome_height()
    area._splitter.setSizes([chrome + 290, 1200])   # plot was 260 px
    area._last_chrome = chrome + 30                  # it would be 290
    area.keep_plot_height()
    assert area._splitter.sizes()[0] == chrome + A.PLOT_DEFAULT_PX


def test_a_chrome_change_without_a_signal_keeps_the_plot_height(area):
    # live (P2 audit re-run): the load grows the chrome (Scale / Colours,
    # the gradient legend) through set_features, which emits no Colour by
    # change -- the plot lost 40 px.  A layout request on the plot widget
    # is what every chrome change has in common.
    area.resize(640, 1200)
    _pump()
    area.reset_divider()
    plot0 = area._splitter.sizes()[0] - area.pinned_chrome_height()
    pw = area.qc_plot_widget
    pw._wb_scale_field.setVisible(True)
    pw._wb_cmap_field.setVisible(True)
    pw._wb_gradient.show()
    _pump(6)
    plot1 = area._splitter.sizes()[0] - area.pinned_chrome_height()
    assert plot1 == plot0


def test_at_its_minimum_the_plot_gets_270_after_the_chrome_grows(area):
    # the section gives the area exactly its minimum; when the chrome grows
    # the minimum grows with it, and the plot -- not the lower area -- must
    # end at its default
    from napari_epicycle.fusion_reconstruction import _qc_plot_area as A

    area.resize(640, area.minimumSizeHint().height())
    _pump()
    area.reset_divider()
    pw = area.qc_plot_widget
    pw._wb_scale_field.setVisible(True)
    pw._wb_cmap_field.setVisible(True)
    pw._wb_gradient.show()
    _pump(6)
    area.resize(640, area.minimumSizeHint().height())    # the parent's answer
    _pump(6)
    top = area._splitter.sizes()[0]
    assert top - area.pinned_chrome_height() == A.PLOT_DEFAULT_PX

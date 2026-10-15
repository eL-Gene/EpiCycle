"""The selection card, wired.

Line 1 (SEL_NONE / SEL_ONE / SEL_COUNT), the tier-mix bar, the chips,
Mark good / Mark bad / Unmark / Undo with their messages, the totals line
in Step 4's words, the grey "unassigned" legend entry, and the lock line.
"""
from __future__ import annotations

import pytest

from napari_epicycle.segmentation import _workbench_text as T
from napari_epicycle.segmentation.analysis_widget import (
    selection_card_parts,
    tier_bucket,
    totals_line,
)

from ._workbench_ctrl import pump, select, table, wired


# ── the pure parts ──

def test_parts_with_nothing_selected():
    parts = selection_card_parts([])
    assert parts == {"count_text": T.SEL_NONE, "mix": [], "chips_html": ""}


def test_parts_for_one_cell_name_its_label():
    parts = selection_card_parts(["good"], first_id=17)
    assert parts["count_text"] == T.SEL_ONE.format(id=17)
    assert parts["mix"] == [("#50C878", 1)]


def test_parts_count_and_mix_in_card_order():
    parts = selection_card_parts(["bad", "", "good", "garbage", None,
                                  "review"] + ["good"] * 1000)
    assert parts["count_text"] == T.SEL_COUNT.format(m="1,006")
    # good, bad (garbage counts as bad), unmarked (empty, None, review)
    assert parts["mix"] == [("#50C878", 1001), ("#FF7828", 2),
                            ("#8b909a", 3)]
    html = parts["chips_html"]
    assert html.index("good") < html.index("bad") < html.index("unmarked")
    assert "1,001" in html


def test_bucket_rule():
    assert [tier_bucket(t) for t in ("good", "bad", "garbage", "", None,
                                     "review", "curated")] == [
        "good", "bad", "bad", "unmarked", "unmarked", "unmarked", "unmarked"]


def test_totals_are_step4_words():
    line = totals_line(["good", "bad", "", ""])
    assert line == T.TOTALS_STEP4.format(g="1", b="1", u="2")
    assert "review" not in line and "unmarked" in line


# ── wired to a real Workbench ──

@pytest.fixture
def cw(tmp_path):
    c, wb = wired(tmp_path, table(40, tiers=["good"] * 10 + [""] * 30))
    yield c, wb
    wb.deleteLater()
    pump()


def test_the_card_starts_empty_with_its_buttons_grey(cw):
    c, wb = cw
    assert wb.qc_selection_label.text() == T.SEL_NONE
    assert wb.sel_mix_bar.isHidden()
    for b in (wb.mark_good_btn, wb.mark_bad_btn, wb.unmark_btn,
              wb.undo_mark_btn):
        assert not b.isEnabled()


def test_a_lasso_fills_line_one_the_bar_and_the_chips(cw):
    c, wb = cw
    wb.qc_plot_widget._selected_indices = {0, 1}   # as a lasso leaves it
    select(c, [1, 2, 11, 12, 13])
    assert wb.qc_selection_label.text() == T.SEL_COUNT.format(m="5")
    assert not wb.sel_mix_bar.isHidden()
    assert wb.sel_mix_bar._shares == [("#50C878", 2), ("#8b909a", 3)]
    assert "unmarked" in wb.sel_chips_label.text()
    assert wb.sel_msg_label.text() == T.LASSO_DONE.format(m="5")
    assert wb.mark_good_btn.isEnabled() and wb.unmark_btn.isEnabled()
    assert not wb.undo_mark_btn.isEnabled()


def test_one_cell_names_its_label(cw):
    c, wb = cw
    select(c, [7])
    assert wb.qc_selection_label.text() == T.SEL_ONE.format(id=7)
    assert wb.sel_msg_label.text() == ""     # a click leaves no message


def test_mark_bad_then_undo(cw):
    c, wb = cw
    select(c, [11, 12, 13])
    wb.mark_bad_btn.click()
    df = c._qc_features_df
    assert list(df.loc[df.label.isin([11, 12, 13]), "qc_tier"]) == ["bad"] * 3
    assert wb.sel_msg_label.text() == T.MARKED.format(m="3", tier="bad")
    assert wb.tier_summary_label.text() == T.TOTALS_STEP4.format(
        g="10", b="3", u="27")
    # the selection is kept, so the card shows the new mix at once
    assert wb.qc_selection_label.text() == T.SEL_COUNT.format(m="3")
    assert wb.sel_mix_bar._shares == [("#FF7828", 3)]
    assert wb.undo_mark_btn.isEnabled()
    wb.undo_mark_btn.click()
    assert list(df.loc[df.label.isin([11, 12, 13]), "qc_tier"]) == [""] * 3
    assert wb.sel_msg_label.text() == T.UNDONE.format(m="3")
    assert not wb.undo_mark_btn.isEnabled()


def test_unmark_returns_cells_to_unassigned_and_undo_restores(cw):
    c, wb = cw
    select(c, [1, 2])
    wb.unmark_btn.click()
    df = c._qc_features_df
    assert list(df.loc[df.label.isin([1, 2]), "qc_tier"]) == ["", ""]
    assert wb.sel_msg_label.text() == T.UNMARKED.format(m="2")
    assert c._tiers_dirty
    wb.undo_mark_btn.click()
    assert list(df.loc[df.label.isin([1, 2]), "qc_tier"]) == ["good", "good"]


def test_undo_is_a_history_popped_last_first(cw):
    c, wb = cw
    select(c, [20])
    c._mark_good()
    c._mark_bad()
    c._undo_mark()
    assert c._qc_features_df.set_index("label").loc[20, "qc_tier"] == "good"
    c._undo_mark()
    assert c._qc_features_df.set_index("label").loc[20, "qc_tier"] == ""
    c._undo_mark()
    assert wb.sel_msg_label.text() == T.UNDO_EMPTY


def test_a_mark_with_nothing_selected_says_so(cw):
    c, wb = cw
    c._mark_good()
    assert wb.sel_msg_label.text() == T.MARK_NEEDS_SEL


def test_save_or_load_ends_the_history(cw):
    c, wb = cw
    select(c, [3])
    c._mark_bad()
    assert wb.undo_mark_btn.isEnabled()
    c._has_unsaved_changes = False        # what Save and a load do
    assert c._mark_history == []
    assert not wb.undo_mark_btn.isEnabled()


def test_a_legend_pick_says_cat_picked_and_keeps_it(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw._color_combo.setCurrentText("QC tier")
    pw._toggle_category_selection("qc_tier", "good")
    pump()
    assert wb.sel_msg_label.text() == T.CAT_PICKED.format(cat="good", m="10")
    assert wb.qc_selection_label.text() == T.SEL_COUNT.format(m="10")


def test_the_grey_unassigned_entry_selects_the_unmarked_cells(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw._color_combo.setCurrentText("QC tier")
    pw._redraw()
    assert "unmarked" in pw._legend_label.text()
    assert "rgb(139,144,154)" in pw._legend_label.text()
    pw._toggle_category_selection("qc_tier", "unmarked")
    pump()
    assert sorted(c._selected_label_ids) == list(range(11, 41))
    assert wb.sel_msg_label.text() == T.CAT_PICKED.format(cat="unmarked",
                                                         m="30")


def test_a_gate_keeps_its_gate_line(cw):
    c, wb = cw
    wb.show_card_message(T.GATE_LINE.format(m="2", feat="x", a="0", b="1"),
                         "ok")
    c._on_eda_gate_selection([1, 2])
    assert wb.sel_msg_label.text().startswith("Gate:")
    assert wb.qc_selection_label.text() == T.SEL_COUNT.format(m="2")


def test_the_lock_line_greys_marks_and_a_key_does_nothing(cw):
    c, wb = cw
    select(c, [11])
    c._set_step4_flag("_measure_running", True)
    assert wb.sel_msg_label.text() == T.WB_LOCKED.format(job="Measure")
    assert not wb.mark_good_btn.isEnabled()
    assert not wb.compute_umap_btn.isEnabled()
    c._mark_good()                       # the slot checks the lock itself
    assert c._qc_features_df.set_index("label").loc[11, "qc_tier"] == ""
    c._set_step4_flag("_measure_running", False)
    assert wb.mark_good_btn.isEnabled()
    assert wb.sel_msg_label.text() == ""


def test_a_grey_key_button_also_dims_its_word(cw):
    """The key buttons' word and letter are child labels with their own
    colours: disabled, they must read grey too (B3b shots)."""
    c, wb = cw
    assert not wb.mark_good_btn.isEnabled()
    assert "#6f6f6f" in wb.mark_good_btn._epicycle_word.styleSheet()
    select(c, [1])
    assert "#e6f7ec" in wb.mark_good_btn._epicycle_word.styleSheet()


def test_totals_count_every_cell_before_the_first_mark(tmp_path):
    c, wb = wired(tmp_path, table(9).drop(columns=["qc_tier"]))
    c._update_tier_summary()
    assert wb.tier_summary_label.text() == T.TOTALS_STEP4.format(
        g="0", b="0", u="9")
    wb.deleteLater()
    pump()


def test_one_cell_reads_singular(cw):
    c, wb = cw
    select(c, [3])
    c._mark_bad()
    assert wb.sel_msg_label.text() == T.MARKED_ONE.format(tier="bad")
    c._undo_mark()
    assert wb.sel_msg_label.text() == T.UNDONE_ONE
    c._unmark()
    assert wb.sel_msg_label.text() == T.UNMARKED_ONE

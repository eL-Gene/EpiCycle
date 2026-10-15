"""Audit X1-1 / E1-5 (HIGH): a legend pick highlights its cells, and keeps
the highlight after a mark; the plot and the card share ONE bucket rule.

The DRAWN mask (``_get_selected_category_mask``, what the scatter dims
and rings by) is tested, not ``_selected_categories``.
"""
from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.segmentation import _workbench_text as T

from ._workbench_ctrl import pump, select, table, wired

TIERS = ["good"] * 5 + ["bad"] * 3 + ["garbage"] + ["review"] + [""] * 10


@pytest.fixture
def cw(tmp_path):
    c, wb = wired(tmp_path, table(20, tiers=TIERS))
    wb.qc_plot_widget._color_combo.setCurrentText("QC tier")
    wb.qc_plot_widget._redraw()     # the combo redraws on `activated`
    pump()
    yield c, wb
    wb.deleteLater()
    pump()


def _drawn(pw):
    m = pw._get_selected_category_mask()
    return set(np.where(m)[0]) if m is not None else set()


def test_the_unmarked_pick_is_drawn_and_counts_match(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw._toggle_category_selection("qc_tier", "unmarked")
    pump()
    # review + the ten empty rows: one rule, 11 cells, everywhere
    assert _drawn(pw) == set(range(9, 20))
    assert len(c._selected_label_ids) == 11
    assert wb.sel_msg_label.text() == T.CAT_PICKED.format(cat="unmarked",
                                                         m="11")


def test_bad_takes_garbage_in_the_plot_and_the_card(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw._toggle_category_selection("qc_tier", "bad")
    pump()
    assert _drawn(pw) == {5, 6, 7, 8}
    assert sorted(c._selected_label_ids) == [6, 7, 8, 9]
    assert wb.sel_msg_label.text() == T.CAT_PICKED.format(cat="bad", m="4")


def test_the_highlight_survives_a_mark(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw._toggle_category_selection("qc_tier", "good")
    pump()
    before = _drawn(pw)
    assert before == set(range(5))
    wb.mark_bad_btn.click()
    assert _drawn(pw) == before          # frozen to the rows it picked
    assert sorted(c._selected_label_ids) == [1, 2, 3, 4, 5]
    wb.undo_mark_btn.click()
    assert _drawn(pw) == before


def test_the_legend_has_a_title_counts_and_one_word(cw):
    _, wb = cw
    html = wb.qc_plot_widget._legend_label.text()
    assert html.startswith('<span style="color:#888888;">QC tier:</span>')
    assert "unmarked" in html and "unassigned" not in html
    assert ">11</span>" in html and ">4</span>" in html


def test_groups_read_g1_not_c0(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    c._qc_features_df["cluster_id"] = np.arange(20) % 2
    pw._df = c._qc_features_df
    if pw._color_combo.findText("Cluster") < 0:
        pw._color_combo.addItem("Cluster")
    pw._color_combo.setCurrentText("Cluster")
    pw._redraw()
    html = pw._legend_label.text()
    assert "G1" in html and "G2" in html and "C0" not in html


def test_a_plot_switch_keeps_the_selection(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw.plot_type_combo.setCurrentText("Scatter")
    pw._toggle_category_selection("qc_tier", "good")
    pump()
    pw.plot_type_combo.setCurrentText("PCA")
    pump()
    assert len(c._selected_label_ids) == 5
    assert pw.selected_label_ids() and len(pw.selected_label_ids()) == 5
    assert _drawn(pw) == set(range(5))

"""Step 3 plot pass, brick P0: the triage histogram is honest.

Nothing on screen may come from made-up data: before a patch loads the
panel shows one sentence, a feature with no finite values says so, Gate
never emits on a data push, the ghost stack can be hidden, and a note
line says what dragging a tier line does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _mod():
    from napari_epicycle.fusion_reconstruction import triage_histogram_panel
    return triage_histogram_panel


def _panel():
    return _mod().TriageHistogramPanel()


def _df(n=60, *, tiers=None, ghosts=0, seed=0):
    rng = np.random.default_rng(seed)
    if tiers is None:
        tiers = rng.choice(["good", "review", "bad", "garbage", "curated"], n)
    status = np.array(["active"] * n, dtype=object)
    status[:ghosts] = "deleted"
    return pd.DataFrame({
        "label": np.arange(1, n + 1),
        "qc_score": rng.uniform(0.0, 5.0, n),
        "qc_tier": np.asarray(tiers, dtype=object),
        "area": np.linspace(50.0, 500.0, n),
        "label_status": status,
    })


def _recorder(panel):
    seen = []
    for name in ("gated_selection_changed", "gated_drag_finished",
                 "thresholds_changed", "tier_drag_finished"):
        getattr(panel, name).connect(
            lambda payload, _n=name: seen.append((_n, payload)))
    return seen


# ---------------------------------------------------------------------------
# Empty states
# ---------------------------------------------------------------------------


def _assert_empty(panel, sentence):
    assert panel._bar_items == [], "no bars may be drawn in an empty state"
    assert all(not line.isVisible() for line in panel._tier_lines)
    assert not panel._gate_region.isVisible()
    assert panel.empty_sentence() == sentence
    assert panel._empty_text.isVisible()
    assert panel._empty_text.textItem.toPlainText() == sentence
    item = panel._plot_widget.getPlotItem()
    assert not item.getAxis("bottom").isVisible()
    assert not item.getAxis("left").isVisible()
    assert item.getAxis("bottom").labelText == ""


def test_a_fresh_panel_shows_the_empty_sentence(qapp):
    m = _mod()
    panel = _panel()
    assert m.HIST_EMPTY == "No QC data yet."
    _assert_empty(panel, m.HIST_EMPTY)
    assert panel._feature_combo.count() == 0, "Triage by starts empty"


def test_set_data_none_after_data_clears_everything(qapp):
    m = _mod()
    panel = _panel()
    panel.set_data(_df())
    assert panel._bar_items, "a real df draws bars"
    assert panel.empty_sentence() is None
    assert not panel._empty_text.isVisible()
    panel.set_data(None)
    _assert_empty(panel, m.HIST_EMPTY)


def test_data_brings_the_axes_back(qapp):
    panel = _panel()
    panel.set_data(_df())
    item = panel._plot_widget.getPlotItem()
    assert item.getAxis("bottom").isVisible()
    assert item.getAxis("left").isVisible()
    assert item.getAxis("bottom").labelText == "qc_score"


def test_a_feature_with_no_finite_values_says_so(qapp):
    m = _mod()
    panel = _panel()
    df = _df()
    df["empty_feat"] = np.nan
    panel.set_data(df)
    panel._feature_combo.setCurrentText("empty_feat")
    assert panel._feature_combo.currentText() == "empty_feat"
    sentence = m.HIST_NO_VALUES.format(feature="empty_feat")
    assert sentence == "No values for empty_feat on this patch."
    _assert_empty(panel, sentence)
    # Back to a feature with values: the sentence goes, bars come back.
    panel._feature_combo.setCurrentText("area")
    assert panel.empty_sentence() is None
    assert panel._bar_items


# ---------------------------------------------------------------------------
# Gate rules
# ---------------------------------------------------------------------------


def _gated(panel):
    panel._feature_combo.setCurrentText("area")
    panel._gate_cb.setChecked(True)
    assert panel._gate_region.isVisible()


def test_untick_gate_silently_emits_nothing(qapp):
    panel = _panel()
    panel.set_data(_df())
    _gated(panel)
    seen = _recorder(panel)
    panel.untick_gate_silently()
    assert not panel._gate_cb.isChecked()
    assert not panel._gate_region.isVisible()
    assert seen == []


def test_set_data_none_turns_gate_off_without_emitting(qapp):
    panel = _panel()
    panel.set_data(_df())
    _gated(panel)
    seen = _recorder(panel)
    panel.set_data(None)
    assert not panel._gate_cb.isChecked()
    assert not panel._gate_region.isVisible()
    assert seen == []


def test_set_data_new_df_turns_gate_off_without_emitting(qapp):
    panel = _panel()
    panel.set_data(_df(seed=1))
    _gated(panel)
    seen = _recorder(panel)
    panel.set_data(_df(seed=2))
    assert not panel._gate_cb.isChecked()
    assert not panel._gate_region.isVisible()
    assert seen == []


def test_set_data_same_df_keeps_the_gate(qapp):
    panel = _panel()
    df = _df()
    panel.set_data(df)
    _gated(panel)
    seen = _recorder(panel)
    panel.set_data(df)
    assert panel._gate_cb.isChecked()
    assert panel._gate_region.isVisible()
    assert [s for s in seen if s[0].startswith("gated")] == []


def test_gate_checkbox_hidden_with_no_data(qapp):
    panel = _panel()
    panel.set_data(_df())
    panel._feature_combo.setCurrentText("area")
    assert not panel._gate_cb.isHidden()
    panel.set_data(None)
    assert panel._gate_cb.isHidden()


def test_gate_visibility_follows_a_feature_change_made_by_a_data_push(qapp):
    """A new df without the chosen feature falls back to qc_score under
    blockSignals; the Gate tick must follow (it belongs to non-qc_score)."""
    panel = _panel()
    panel.set_data(_df())
    panel._feature_combo.setCurrentText("area")
    assert not panel._gate_cb.isHidden()
    df2 = _df(seed=3).drop(columns=["area"])
    panel.set_data(df2)
    assert panel._feature_combo.currentText() == "qc_score"
    assert panel._gate_cb.isHidden()


# ---------------------------------------------------------------------------
# Ghosts
# ---------------------------------------------------------------------------


def _bar_total(panel):
    total = 0.0
    for bar in panel._bar_items:
        total += float(np.sum(bar.opts["height"]))
    return total


def _bar_colours(panel):
    out = set()
    for bar in panel._bar_items:
        c = bar.opts["brush"].color()
        out.add((c.red(), c.green(), c.blue()))
    return out


def test_ghost_stack_drawn_by_default_in_ghost_grey(qapp):
    m = _mod()
    assert m.GHOST_GREY == (140, 140, 160)
    panel = _panel()
    panel.set_data(_df(n=60, ghosts=10))
    assert m.GHOST_GREY in _bar_colours(panel)
    assert _bar_total(panel) == 60


def test_set_ghosts_visible_false_drops_the_ghost_stack(qapp):
    m = _mod()
    panel = _panel()
    panel.set_data(_df(n=60, ghosts=10))
    panel.set_ghosts_visible(False)
    assert m.GHOST_GREY not in _bar_colours(panel)
    assert _bar_total(panel) == 50
    panel.set_ghosts_visible(True)
    assert m.GHOST_GREY in _bar_colours(panel)
    assert _bar_total(panel) == 60


def test_hidden_ghosts_also_leave_the_gradient_bars(qapp):
    panel = _panel()
    panel.set_data(_df(n=60, ghosts=10))
    panel._color_combo.setCurrentText(_mod().HIST_VALUE_ITEM)
    assert _bar_total(panel) == 60
    panel.set_ghosts_visible(False)
    assert _bar_total(panel) == 50


# (Ghosts toggle keeps the x-range, bin edges and tier lines: see
# test_triage_histogram_audit_fold.py::test_ghosts_toggle_keeps_x_range_and_bin_edges.)


# ---------------------------------------------------------------------------
# The note line
# ---------------------------------------------------------------------------


def test_note_says_what_a_drag_does_on_qc_score(qapp):
    m = _mod()
    assert m.HIST_NOTE_DRAG == (
        "Dragging a line re-sorts every label that is not curated, "
        "hand assignments included."
    )
    panel = _panel()
    panel.set_data(_df())
    assert not panel._note_label.isHidden()
    assert panel._note_label.full_text() == m.HIST_NOTE_DRAG
    assert panel._note_label.toolTip() == m.HIST_NOTE_DRAG


def test_note_says_all_curated_when_every_active_label_is(qapp):
    m = _mod()
    assert m.HIST_NOTE_ALL_CURATED == (
        "All {n} labels are curated; the lines change nothing."
    )
    panel = _panel()
    # 40 rows: 30 active all curated, 10 ghosts (not counted).
    tiers = ["garbage"] * 10 + ["curated"] * 30
    panel.set_data(_df(n=40, tiers=tiers, ghosts=10))
    assert panel._note_label.full_text() == m.HIST_NOTE_ALL_CURATED.format(n=30)
    assert not panel._note_label.isHidden()


def test_note_hidden_for_any_other_feature(qapp):
    panel = _panel()
    panel.set_data(_df())
    panel._feature_combo.setCurrentText("area")
    assert panel._note_label.isHidden()
    assert panel._note_label.full_text() == ""
    assert panel._note_label.text() == ""


def test_note_hidden_with_no_data(qapp):
    panel = _panel()
    assert panel._note_label.isHidden()
    panel.set_data(_df())
    panel.set_data(None)
    assert panel._note_label.isHidden()


def test_note_hidden_with_no_active_rows(qapp):
    panel = _panel()
    panel.set_data(_df(n=20, ghosts=20))
    assert panel._note_label.isHidden()


def test_note_elided_to_one_line_with_the_full_sentence_as_tooltip(qapp):
    m = _mod()
    panel = _panel()
    panel.resize(240, 300)
    panel.show()
    try:
        panel.set_data(_df())
        qapp.processEvents()
        label = panel._note_label
        assert label.full_text() == m.HIST_NOTE_DRAG
        assert label.toolTip() == m.HIST_NOTE_DRAG
        shown = label.text()
        assert shown != m.HIST_NOTE_DRAG and shown.endswith("…"), shown
        assert label.fontMetrics().horizontalAdvance(shown) <= label.width()
        assert not label.wordWrap()
        # Wider: the full sentence comes back (re-elided on resize).
        panel.resize(2000, 300)
        qapp.processEvents()
        assert label.text() == m.HIST_NOTE_DRAG
    finally:
        panel.hide()

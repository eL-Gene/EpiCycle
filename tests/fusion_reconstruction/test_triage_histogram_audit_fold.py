"""Step 3 plot pass, brick P0 -- the audit fold (2026-10-06).

1. categorical stacks line up with their rows when ghosts are shown;
2. Gate re-seats on a switch between two gateable features;
3. "Colour bars by" offers the categorical columns + one "value" entry;
5. Reset lines and Gate share one slot; the empty state disables the combos;
6. bins come from the FULL set, so Ghosts off is a pure subtraction;
7. mouse off in the empty state, a dark halo under each tier line, every
   on-screen word a named constant.
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


def _rows(area, cluster, status):
    n = len(area)
    return pd.DataFrame({
        "label": np.arange(1, n + 1),
        "qc_score": np.linspace(0.0, 1.0, n),
        "qc_tier": ["good"] * n,
        "area": np.asarray(area, dtype=float),
        "cluster_id": np.asarray(cluster, dtype=int),
        "label_status": list(status),
    })


def _heights_by_colour(panel):
    """{(r, g, b): per-bin heights summed over that colour's bars}."""
    out = {}
    for bar in panel._bar_items:
        c = bar.opts["brush"].color()
        key = (c.red(), c.green(), c.blue())
        h = np.asarray(bar.opts["height"], dtype=float)
        x0 = np.atleast_1d(np.asarray(bar.opts["x0"], dtype=float))
        out.setdefault(key, []).append((x0, np.atleast_1d(h)))
    return out


def _per_bin(panel, edges_x0, colour_filter):
    total = np.zeros(len(edges_x0))
    for key, parts in _heights_by_colour(panel).items():
        if not colour_filter(key):
            continue
        for x0, h in parts:
            for xv, hv in zip(x0, h):
                total[int(np.argmin(np.abs(edges_x0 - xv)))] += hv
    return total


def _bin_x0(panel):
    """The x0 array of the full-width (stacked) bars."""
    for bar in panel._bar_items:
        x0 = np.atleast_1d(np.asarray(bar.opts["x0"], dtype=float))
        if len(x0) > 1:
            return x0
    raise AssertionError("no stacked bar")


# ---------------------------------------------------------------------------
# 1. categorical stacks with ghosts
# ---------------------------------------------------------------------------


def test_categorical_stack_lines_up_with_ghost_rows_first(qapp):
    m = _mod()
    df = _rows(area=[0, 0, 100, 100, 100], cluster=[7, 7, 1, 2, 3],
               status=["deleted", "deleted", "active", "active", "active"])
    panel = _panel()
    panel.set_data(df)
    panel._feature_combo.setCurrentText("area")
    panel._color_combo.setCurrentText("cluster_id")
    x0 = _bin_x0(panel)
    active = _per_bin(panel, x0, lambda k: k != m.GHOST_GREY)
    ghost = _per_bin(panel, x0, lambda k: k == m.GHOST_GREY)
    assert active[-1] == 3 and active.sum() == 3, active
    assert ghost[0] == 2 and ghost.sum() == 2, ghost


def test_categorical_stack_hundred_ghosts_at_zero(qapp):
    m = _mod()
    area = [0.0] * 100 + [10.0] * 100
    status = ["superseded"] * 100 + ["active"] * 100
    df = _rows(area=area, cluster=[1] * 200, status=status)
    panel = _panel()
    panel.set_data(df)
    panel._feature_combo.setCurrentText("area")
    panel._color_combo.setCurrentText("cluster_id")
    x0 = _bin_x0(panel)
    active = _per_bin(panel, x0, lambda k: k != m.GHOST_GREY)
    ghost = _per_bin(panel, x0, lambda k: k == m.GHOST_GREY)
    assert active[-1] == 100 and active.sum() == 100, active
    assert ghost[0] == 100 and ghost.sum() == 100, ghost
    # Ghosts hidden: the grey stack goes, the active stack stays put.
    panel.set_ghosts_visible(False)
    x0b = _bin_x0(panel)
    assert np.allclose(x0, x0b)
    active = _per_bin(panel, x0b, lambda k: k != m.GHOST_GREY)
    ghost = _per_bin(panel, x0b, lambda k: k == m.GHOST_GREY)
    assert active[-1] == 100 and active.sum() == 100
    assert ghost.sum() == 0


# ---------------------------------------------------------------------------
# 2. Gate re-seats on a gateable feature switch
# ---------------------------------------------------------------------------


def _two_feature_df(n=40):
    return pd.DataFrame({
        "label": np.arange(1, n + 1),
        "qc_score": np.linspace(0.0, 1.0, n),
        "qc_tier": ["good"] * n,
        "area": np.linspace(50.0, 500.0, n),
        "volume": np.linspace(1000.0, 9000.0, n)[::-1].copy(),
    })


def test_gate_reseats_and_emits_on_a_switch_between_gateable_features(qapp):
    df = _two_feature_df()
    panel = _panel()
    panel.set_data(df)
    panel._feature_combo.setCurrentText("area")
    panel._gate_cb.setChecked(True)
    light, heavy = [], []
    panel.gated_selection_changed.connect(lambda s: light.append(set(s)))
    panel.gated_drag_finished.connect(lambda s: heavy.append(set(s)))
    panel._feature_combo.setCurrentText("volume")
    lo = float(np.percentile(df["volume"], 10))
    hi = float(np.percentile(df["volume"], 90))
    low, high = panel._gate_region.getRegion()
    assert abs(low - lo) < 1e-6 and abs(high - hi) < 1e-6
    expected = {int(v) for v in df.loc[
        (df["volume"] >= lo) & (df["volume"] <= hi), "label"]}
    assert expected
    assert light and light[-1] == expected
    assert heavy and heavy[-1] == expected
    assert panel._gate_cb.isChecked()
    assert panel._gate_region.isVisible()


def test_gate_early_return_hides_the_band(qapp):
    """A feature with fewer than two finite values cannot be gated: the
    band must not be left on screen."""
    df = _two_feature_df()
    df["sparse"] = np.nan
    df.loc[0, "sparse"] = 3.0
    panel = _panel()
    panel.set_data(df)
    panel._feature_combo.setCurrentText("sparse")
    panel._gate_cb.setChecked(True)
    assert not panel._gate_region.isVisible()


# ---------------------------------------------------------------------------
# 3. "Colour bars by"
# ---------------------------------------------------------------------------


def test_colour_bars_by_offers_categoricals_and_one_value_entry(qapp):
    m = _mod()
    df = _two_feature_df()
    df["cluster_id"] = 0
    df["is_border_cell"] = False
    panel = _panel()
    panel.set_data(df)
    items = [panel._color_combo.itemText(i)
             for i in range(panel._color_combo.count())]
    assert items == ["qc_tier", "cluster_id", "is_border_cell",
                     m.HIST_VALUE_ITEM]
    for numeric in ("qc_score", "area", "volume", "label"):
        assert numeric not in items
    idx = items.index(m.HIST_VALUE_ITEM)
    from qtpy.QtCore import Qt
    tip = panel._color_combo.itemData(idx, Qt.ItemDataRole.ToolTipRole)
    assert tip == m.HIST_HELP[m.HIST_VALUE_ITEM]


def test_value_entry_draws_the_gradient(qapp):
    m = _mod()
    panel = _panel()
    panel.set_data(_two_feature_df())
    panel._color_combo.setCurrentText(m.HIST_VALUE_ITEM)
    assert panel._color_combo.currentText() == m.HIST_VALUE_ITEM
    # The gradient draws one bar item per non-empty bin, coloured by
    # position: many items, many colours, none of them a tier colour.
    assert len(panel._bar_items) > 3
    colours = set(_heights_by_colour(panel))
    assert len(colours) > 3
    assert m._TIER_COLORS_RGB["good"] not in colours


def test_colour_bars_by_tooltip_says_what_value_does(qapp):
    m = _mod()
    tip = m.HIST_HELP[m.HIST_COLOUR_BARS_BY]
    assert "value" in tip and "Triage by" in tip


# ---------------------------------------------------------------------------
# 5. one slot for Reset lines / Gate; the empty state
# ---------------------------------------------------------------------------


def test_empty_state_hides_reset_and_gate_and_disables_combos(qapp):
    panel = _panel()
    assert panel._reset_btn.isHidden()
    assert panel._gate_cb.isHidden()
    assert not panel._feature_combo.isEnabled()
    assert not panel._color_combo.isEnabled()


def test_qc_score_shows_reset_only(qapp):
    panel = _panel()
    panel.set_data(_two_feature_df())
    assert not panel._reset_btn.isHidden()
    assert panel._gate_cb.isHidden()
    assert panel._feature_combo.isEnabled()
    assert panel._color_combo.isEnabled()


def test_other_feature_shows_gate_only(qapp):
    panel = _panel()
    panel.set_data(_two_feature_df())
    panel._feature_combo.setCurrentText("area")
    assert panel._reset_btn.isHidden()
    assert not panel._gate_cb.isHidden()


def test_no_values_state_hides_both_but_keeps_the_combos_live(qapp):
    """The way out of an empty feature is the Triage by combo itself."""
    df = _two_feature_df()
    df["empty_feat"] = np.nan
    panel = _panel()
    panel.set_data(df)
    panel._feature_combo.setCurrentText("empty_feat")
    assert panel._reset_btn.isHidden()
    assert panel._gate_cb.isHidden()
    assert panel._feature_combo.isEnabled()


def test_back_to_no_data_hides_reset_again(qapp):
    panel = _panel()
    panel.set_data(_two_feature_df())
    panel.set_data(None)
    assert panel._reset_btn.isHidden()
    assert not panel._feature_combo.isEnabled()


# ---------------------------------------------------------------------------
# 6. bins over the full set
# ---------------------------------------------------------------------------


def test_ghosts_toggle_keeps_x_range_and_bin_edges(qapp):
    area = list(np.linspace(0.0, 20.0, 30)) + list(np.linspace(50.0, 80.0, 30))
    status = ["deleted"] * 30 + ["active"] * 30
    df = _rows(area=area, cluster=[1] * 60, status=status)
    df["qc_score"] = np.asarray(area) / 10.0
    panel = _panel()
    panel.set_data(df)
    edges_on = _bin_x0(panel)
    range_on = panel._plot_widget.getViewBox().viewRange()[0]
    lines_on = panel.get_thresholds()
    panel.set_ghosts_visible(False)
    edges_off = _bin_x0(panel)
    range_off = panel._plot_widget.getViewBox().viewRange()[0]
    assert np.allclose(edges_on, edges_off)
    assert np.allclose(range_on, range_off)
    assert panel.get_thresholds() == lines_on


# ---------------------------------------------------------------------------
# 7. polish
# ---------------------------------------------------------------------------


def test_mouse_off_in_the_empty_state_and_back_with_data(qapp):
    panel = _panel()
    vb = panel._plot_widget.getViewBox()
    assert list(vb.state["mouseEnabled"]) == [False, False]
    panel.set_data(_two_feature_df())
    assert list(vb.state["mouseEnabled"]) == [True, False]
    panel.set_data(None)
    assert list(vb.state["mouseEnabled"]) == [False, False]


def test_a_dark_halo_under_each_tier_line(qapp):
    m = _mod()
    panel = _panel()
    assert len(panel._tier_halos) == 3
    panel.set_data(_two_feature_df())
    items = panel._plot_widget.getPlotItem().items
    for line, halo in zip(panel._tier_lines, panel._tier_halos):
        assert halo in items
        c = halo.pen.color()
        assert (c.red(), c.green(), c.blue()) == m.HALO_RGB == (30, 30, 36)
        assert halo.pen.widthF() > line.pen.widthF()
        assert halo.zValue() == line.zValue() - 1
        assert not halo.movable
        assert halo.isVisible() == line.isVisible() is True
        assert halo.value() == pytest.approx(line.value())
    # A drag moves the halo with its line.
    panel._tier_lines[1].setValue(0.42)
    assert panel._tier_halos[1].value() == pytest.approx(0.42)
    # Another feature hides both.
    panel._feature_combo.setCurrentText("area")
    for halo in panel._tier_halos:
        assert not halo.isVisible()
    panel.set_data(None)
    for halo in panel._tier_halos:
        assert not halo.isVisible()


def test_reset_lines_moves_the_halos_too(qapp):
    panel = _panel()
    panel.set_data(_two_feature_df())
    for line in panel._tier_lines:
        line.setValue(0.99)
    panel._reset_btn.click()
    for line, halo in zip(panel._tier_lines, panel._tier_halos):
        assert halo.value() == pytest.approx(line.value())


def test_every_on_screen_word_is_a_named_constant(qapp):
    from qtpy.QtWidgets import QLabel
    m = _mod()
    assert m.HIST_TRIAGE_BY == "Triage by"
    assert m.HIST_COLOUR_BARS_BY == "Colour bars by"
    assert m.HIST_RESET_LINES == "Reset lines"
    assert m.HIST_GATE == "Gate"
    assert m.HIST_COUNT_AXIS == "count"
    assert m.HIST_VALUE_ITEM == "value"
    assert set(m.HIST_HELP) == {m.HIST_TRIAGE_BY, m.HIST_COLOUR_BARS_BY,
                                m.HIST_RESET_LINES, m.HIST_GATE,
                                m.HIST_VALUE_ITEM}
    panel = _panel()
    labels = {lab.text() for lab in panel.findChildren(QLabel)}
    assert {m.HIST_TRIAGE_BY, m.HIST_COLOUR_BARS_BY} <= labels
    assert panel._reset_btn.text() == m.HIST_RESET_LINES
    assert panel._gate_cb.text() == m.HIST_GATE
    panel.set_data(_two_feature_df())
    left = panel._plot_widget.getPlotItem().getAxis("left")
    assert left.labelText == m.HIST_COUNT_AXIS


def test_gate_tooltip_trimmed(qapp):
    m = _mod()
    assert "Not offered" not in m.HIST_HELP[m.HIST_GATE]


def test_stale_docstrings_fixed():
    import inspect
    m = _mod()
    cls = m.TriageHistogramPanel
    assert "outlive" not in inspect.getdoc(cls._create_tier_lines)
    assert "at line 638" not in inspect.getdoc(cls._attach_tier_lines)

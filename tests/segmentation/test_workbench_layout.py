"""14: the Workbench layout (brick B3a).

* 578 px: nothing scrolls sideways; the pinned part (without the plot) is
  within WORKBENCH_PINNED_PX, idle AND busy, with every optional line on.
* 1000 px: the ONE feature panel moves to the 420 px left column (a
  reparent, never a rebuild) with hysteresis; the Features tab hides.
* The divider: 9 px, its help line, a double click resets the panes.
* Every help line of the mockup's H table reaches a control's tooltip.

Heights are measured through the layout's height-for-width at the width
the user gets, not ``sizeHint()`` at the content's own.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_APP = []

#: The dock widths.
NARROW_W = 578
WIDE_W = 1000


#: Offscreen Qt has NO font database of its own: every glyph comes out
#: ~11 px wide, about 1.6x Segoe UI, so a width or height measured with it
#: says nothing about the dock the user gets.  The CEILINGS are therefore
#: measured in a SUBPROCESS pointed at the system fonts with napari's
#: Windows face (the mockup's ceilings were measured in Segoe UI too); the
#: in-process tests below are font-agnostic and never change the app's
#: font, so their result does not depend on which test file ran first.
_FONT_DIR = "C:/Windows/Fonts"
_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    a = QApplication.instance() or QApplication([])
    _APP.append(a)
    font = a.font()
    yield a
    a.setFont(font)   # whatever a test did, the next file gets it back


def _df(n=240, seed=0):
    from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES

    rng = np.random.RandomState(seed)
    cols = {"label": np.arange(1, n + 1)}
    for f in list(PCA_FEATURES)[:18] + ["membrane_intensity_mean",
                                        "nuclei_intensity_mean",
                                        "a_feature_with_a_very_long_name_x"]:
        cols[f] = rng.lognormal(size=n)
    cols["qc_tier"] = rng.choice(["good", "bad", "unassigned"], size=n)
    cols["is_border_cell"] = rng.rand(n) < 0.1
    return pd.DataFrame(cols)


def _pump(app, k=3):
    for _ in range(k):
        app.processEvents()


@pytest.fixture
def wb(app):
    from napari_epicycle.segmentation.analysis_widget import AnalysisControlsWidget

    w = AnalysisControlsWidget()
    w.resize(NARROW_W, 900)
    w.show()
    df = _df()
    w.set_feature_data(df)
    w.qc_plot_widget.set_features(df)
    _pump(app)
    yield w
    w.hide()
    w.deleteLater()
    _pump(app)


def _every_optional_line(wb, app, *, busy: bool):
    """Push every control the real pushes push: a
    selection with every chip, a message line, the totals, a feature
    colour (Scale and Colours on), and the busy strip on or off."""
    from napari_epicycle.segmentation import _workbench_text as T

    pw = wb.qc_plot_widget
    pw.plot_type_combo.setCurrentText("Scatter")
    pw._color_combo.setCurrentText(pw._feature_x_combo.itemText(0))
    chips = " &nbsp; ".join(
        f'<span style="color:{c}">●</span> {t} <b>{n}</b>'
        for t, c, n in (("good", "#50C878", 1204), ("review", "#FFC83C", 312),
                        ("bad", "#FF7828", 88), ("unassigned", "#8b909a", 1396)))
    wb.show_selection(count_text=T.SEL_COUNT.format(m="3,000"),
                      mix=[("#50C878", 1204), ("#FFC83C", 312),
                           ("#FF7828", 88), ("#8b909a", 1396)],
                      chips_html=chips)
    wb.show_card_message(T.GATE_LINE.format(
        m="1,204", feat="membrane_boundary_intensity_mean", a="0.25",
        b="12.50"), "ok")
    wb.tier_summary_label.setText(T.TOTALS_STEP4.format(g=1204, b=88, u=1708))
    if busy:
        wb.set_busy_strip(T.BUSY.format(map="UMAP", s=45))
    else:
        wb.set_busy_strip(None)
    _pump(app)
    assert not pw._wb_scale_field.isHidden()
    assert not wb.sel_chips_label.isHidden()
    assert not wb.sel_msg_label.isHidden()
    assert wb.busy_strip.isHidden() is (not busy)


def _measured_chrome(wb, width):
    """An independent measure: pin the plot to 100 px, ask the pinned
    pane's layout for its height-for-width, take the 100 px off."""
    pw = wb.qc_plot_widget
    plot = pw._plot_widget
    lay = wb._pinned.layout()
    prev = (plot.minimumHeight(), plot.maximumHeight())

    def measure():
        plot.updateGeometry()
        pw.layout().invalidate()
        pw.updateGeometry()
        lay.invalidate()
        return int(lay.totalHeightForWidth(width))

    try:
        plot.setFixedHeight(100)
        total = measure()
    finally:
        plot.setMinimumHeight(prev[0])
        plot.setMaximumHeight(prev[1])
        measure()
    return total - 100


_SUBPROCESS = """
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
sys.path.insert(0, TEST_DIR)
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
import test_workbench_layout as t
from napari_epicycle.segmentation.analysis_widget import AnalysisControlsWidget
out = {"font": app.font().family()}
for width in (578, 1000):
    w = AnalysisControlsWidget()
    w.resize(width, 900)
    w.show()
    df = t._df()
    w.set_feature_data(df)
    w.qc_plot_widget.set_features(df)
    t._pump(app, 6)
    column = w._splitter.width()
    got = t._busy_and_idle(w, app, column)
    got["column"] = column
    got["wide"] = w.is_wide()
    got["min_width"] = w.minimumSizeHint().width()
    out[str(width)] = got
    w.hide()
    w.deleteLater()
    t._pump(app)
print("WBJSON" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def ceilings():
    """The pinned part's height at 578 and 1000 px, measured in a fresh
    process with real fonts (W1-1)."""
    code = (f"FONT_DIR = {_FONT_DIR!r}\n"
            f"TEST_DIR = {str(Path(__file__).parent)!r}\n" + _SUBPROCESS)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                          env=env, capture_output=True, text=True,
                          timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("WBJSON")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("WBJSON"):])


def _busy_and_idle(wb, app, width):
    out = {}
    for busy in (True, False):
        _every_optional_line(wb, app, busy=busy)
        got = _measured_chrome(wb, width)
        # the widget's own sum (used by the divider reset) agrees
        assert abs(wb.pinned_chrome_height(width) - got) <= 2, (
            wb.pinned_chrome_height(width), got)
        out["busy" if busy else "idle"] = got
    return out


# --------------------------------------------------------------------------
# widths
# --------------------------------------------------------------------------

def test_the_dock_fits_578(wb):
    assert wb.minimumSizeHint().width() <= NARROW_W
    assert not wb.is_wide()


@pytest.mark.parametrize("tab", [0, 1, 2])
def test_no_tab_scrolls_sideways_at_578(wb, app, tab):
    wb._tabs.setCurrentIndex(tab)
    _pump(app)
    scroll = wb._tabs.currentWidget()
    content = scroll.widget()
    assert content.minimumSizeHint().width() <= scroll.viewport().width() + 1
    assert scroll.horizontalScrollBar().maximum() == 0


def test_the_pinned_part_fits_the_column(wb):
    assert wb._pinned.minimumSizeHint().width() <= NARROW_W


# --------------------------------------------------------------------------
# heights (WORKBENCH_PINNED_PX, idle AND busy, the max asserted)
# --------------------------------------------------------------------------

def test_pinned_ceiling_at_578(ceilings):
    from napari_epicycle.segmentation.analysis_widget import WORKBENCH_PINNED_PX

    got = ceilings["578"]
    print(f"[WB-CEILING] 578 px ({ceilings['font']}): busy {got['busy']} px, "
          f"idle {got['idle']} px (ceiling {WORKBENCH_PINNED_PX})")
    assert not got["wide"]
    assert got["busy"] > got["idle"], "the busy strip adds height"
    assert max(got["busy"], got["idle"]) <= WORKBENCH_PINNED_PX
    assert got["min_width"] <= NARROW_W


def test_pinned_ceiling_at_1000(ceilings):
    from napari_epicycle.segmentation.analysis_widget import (
        WORKBENCH_LEFT_COL_W, WORKBENCH_PINNED_PX, WORKBENCH_WIDE_OFF_PX)

    got = ceilings["1000"]
    print(f"[WB-CEILING] 1000 px (right column {got['column']} px, "
          f"{ceilings['font']}): busy {got['busy']} px, idle {got['idle']} px "
          f"(ceiling {WORKBENCH_PINNED_PX})")
    assert got["wide"]
    assert got["column"] >= WIDE_W - WORKBENCH_LEFT_COL_W - 2
    assert max(got["busy"], got["idle"]) <= WORKBENCH_PINNED_PX
    # W1-4: the split-out dock can shrink below its own hysteresis edge
    assert got["min_width"] < WORKBENCH_WIDE_OFF_PX


def test_the_chrome_sum_matches_a_pinned_measure(wb, app):
    """In process and font-agnostic: the widget's own sum (used by the
    divider) agrees with the pinning measure, busy and idle."""
    got = _busy_and_idle(wb, app, NARROW_W)
    assert got["busy"] > got["idle"]


def test_the_wide_dock_can_shrink_below_its_edge(wb, app):
    from napari_epicycle.segmentation.analysis_widget import (
        WORKBENCH_WIDE_OFF_PX)

    wb.resize(WIDE_W, 900)
    _pump(app)
    assert wb.is_wide()
    assert wb.minimumSizeHint().width() < WORKBENCH_WIDE_OFF_PX


def test_the_plot_keeps_its_minimum_under_the_ceiling(wb, app):
    """The pinned pane can always show the plot at its 150 px floor."""
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import WB_PLOT_MIN_H

    _every_optional_line(wb, app, busy=True)
    assert wb.qc_plot_widget._plot_widget.minimumHeight() == WB_PLOT_MIN_H


# --------------------------------------------------------------------------
# the 1000 px move (Appendix B R6)
# --------------------------------------------------------------------------

def test_the_feature_panel_moves_with_hysteresis(wb, app):
    from napari_epicycle.segmentation.analysis_widget import (
        WORKBENCH_WIDE_OFF_PX, WORKBENCH_WIDE_ON_PX)

    panel = wb._feature_panel
    table = wb.feature_table
    assert wb._features_scroll.widget() is panel
    wb.resize(WORKBENCH_WIDE_ON_PX - 1, 900)
    _pump(app)
    assert not wb.is_wide()
    wb.resize(WIDE_W, 900)
    _pump(app)
    assert wb.is_wide()
    assert wb._left_scroll.widget() is panel      # the SAME instance
    assert wb.feature_table is table
    assert not wb._left_scroll.isHidden()
    assert not wb._tabs.isTabVisible(0)
    assert wb._tabs.currentIndex() != 0
    # inside the band: no flip either way
    wb.resize(WORKBENCH_WIDE_OFF_PX, 900)
    _pump(app)
    assert wb.is_wide()
    wb.resize(WORKBENCH_WIDE_OFF_PX - 1, 900)
    _pump(app)
    assert not wb.is_wide()
    assert wb._features_scroll.widget() is panel
    assert wb._left_scroll.isHidden()
    assert wb._tabs.isTabVisible(0)


def test_the_move_keeps_state(wb, app):
    """Ticks, rescales and the open detail survive the reparent."""
    feats = wb.get_selected_features()
    first = feats[0]
    wb._show_feature_detail(first)
    wb.resize(WIDE_W, 900)
    _pump(app)
    wb.resize(NARROW_W, 900)
    _pump(app)
    assert wb.get_selected_features() == feats
    assert not wb._detail_panel.isHidden()
    assert wb.feature_table.rowCount() > len(feats)


def test_the_wide_dock_is_not_wider_than_1000(wb, app):
    wb.resize(WIDE_W, 900)
    _pump(app)
    assert wb.minimumSizeHint().width() <= WIDE_W


# --------------------------------------------------------------------------
# the divider
# --------------------------------------------------------------------------

def test_the_divider(wb, app):
    from qtpy.QtCore import QEvent, QPointF, Qt
    from qtpy.QtGui import QMouseEvent

    from napari_epicycle.segmentation._workbench_text import WORKBENCH_HELP_LINES
    from napari_epicycle.segmentation.analysis_widget import (
        WORKBENCH_HANDLE_PX, WORKBENCH_PLOT_DEFAULT_PX)

    sp = wb._splitter
    assert sp.handleWidth() == WORKBENCH_HANDLE_PX == 9
    handle = sp.handle(1)
    assert handle.toolTip() == WORKBENCH_HELP_LINES["Divider"]
    total = sum(sp.sizes())
    sp.setSizes([total - 150, 150])
    _pump(app)
    moved = sp.sizes()
    pos = QPointF(handle.width() / 2, handle.height() / 2)
    ev = QMouseEvent(QEvent.MouseButtonDblClick, pos, pos, Qt.LeftButton,
                     Qt.LeftButton, Qt.NoModifier)
    app.sendEvent(handle, ev)
    _pump(app)
    after = sp.sizes()
    assert after != moved
    plot_h = wb.qc_plot_widget._plot_widget.height()
    assert abs(plot_h - WORKBENCH_PLOT_DEFAULT_PX) <= 12, plot_h


def test_the_plot_is_clamped(wb, app):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        WB_PLOT_MAX_H, WB_PLOT_MIN_H)

    sp = wb._splitter
    total = sum(sp.sizes())
    sp.setSizes([total, 0])
    _pump(app)
    assert wb.qc_plot_widget._plot_widget.height() <= WB_PLOT_MAX_H
    sp.setSizes([0, total])
    _pump(app)
    assert wb.qc_plot_widget._plot_widget.height() >= WB_PLOT_MIN_H


# --------------------------------------------------------------------------
# tabs, card, strip, surfaces
# --------------------------------------------------------------------------

def test_three_tabs(wb):
    from napari_epicycle.segmentation import _workbench_text as T

    names = [wb._tabs.tabText(i) for i in range(wb._tabs.count())]
    assert names == [T.TAB_FEATURES, T.TAB_MAP, T.TAB_GROUPS]


def test_the_card_replaces_the_triage_tab(wb):
    from napari_epicycle.segmentation import _workbench_text as T

    assert wb.qc_selection_label.text() == T.SEL_NONE
    assert wb.clear_selection_btn is wb.qc_plot_widget._wb_clear_btn
    assert wb.clear_selection_btn.text() == "✕ Clear"
    for b in (wb.mark_good_btn, wb.mark_bad_btn, wb.unmark_btn,
              wb.undo_mark_btn):
        assert b.parent() is wb._sel_card
    assert "255, 120, 40" in wb.mark_bad_btn.styleSheet()   # the bad orange


def test_card_buttons_emit(wb):
    seen = []
    wb.unmark_requested.connect(lambda: seen.append("u"))
    wb.undo_mark_requested.connect(lambda: seen.append("z"))
    wb.mark_good_requested.connect(lambda: seen.append("g"))
    wb.mark_bad_requested.connect(lambda: seen.append("b"))
    wb.stop_map_requested.connect(lambda: seen.append("s"))
    # B3b: the mark buttons need a selection, Undo a history
    wb.set_mark_state(has_selection=True, can_undo=True)
    for b in (wb.mark_good_btn, wb.mark_bad_btn, wb.unmark_btn,
              wb.undo_mark_btn, wb.busy_stop_btn):
        b.click()
    assert seen == ["g", "b", "u", "z", "s"]


def test_the_busy_strip_is_hidden_until_asked(wb, app):
    assert wb.busy_strip.isHidden()
    wb.set_busy_strip("x")
    assert not wb.busy_strip.isHidden()
    wb.set_busy_strip(None)
    assert wb.busy_strip.isHidden()


def test_no_navy_surfaces(wb):
    from qtpy.QtWidgets import QWidget

    for w in [wb] + wb.findChildren(QWidget):
        sheet = w.styleSheet().lower()
        assert "#1e1e2e" not in sheet and "#333348" not in sheet \
            and "#1a1a2e" not in sheet, type(w).__name__


def test_the_status_line_under_the_tabs_is_gone(wb):
    """status_label (the controller still writes it) is the Map tab's
    result line now."""
    page = wb._map_scroll.widget()
    assert wb.status_label.parent() is page


def test_the_see_line_counts(wb, app):
    wb.qc_plot_widget.highlight_labels([1, 2, 3])
    _pump(app)
    assert "3 selected" in wb.see_label.text()
    wb.qc_plot_widget._clear_selection()
    _pump(app)
    assert "none selected" in wb.see_label.text()


def test_pc_pickers_ride_the_axes_row(wb, app):
    pw = wb.qc_plot_widget
    pw.plot_type_combo.setCurrentText("PCA")
    _pump(app)
    assert not wb._pc_axes_field.isHidden()
    pw.plot_type_combo.setCurrentText("UMAP")
    _pump(app)
    assert wb._pc_axes_field.isHidden()


def test_groups_shows_only_the_knobs_a_method_uses(wb):
    combo = wb._cluster_algo_combo
    expect = {"leiden": ("_cluster_resolution_spin", "_cluster_knn_spin"),
              "hdbscan": ("_cluster_min_size_spin",),
              "kmeans": ("_cluster_k_spin",), "gmm": ("_cluster_k_spin",)}
    all_knobs = {"_cluster_resolution_spin", "_cluster_knn_spin",
                 "_cluster_min_size_spin", "_cluster_k_spin"}
    for i in range(combo.count()):
        combo.setCurrentIndex(i)
        key = combo.currentData()
        shown = {k for k in all_knobs if not getattr(wb, k).isHidden()}
        assert shown == set(expect[key]), key
        assert wb._cluster_seed_spin.isEnabled() is (key != "hdbscan")
        assert wb._seed_off_label.isHidden() is (key != "hdbscan")


def test_k_and_min_size_are_one_value(wb):
    wb._cluster_min_size_spin.setValue(9)
    assert wb._cluster_k_spin.value() == 9
    wb._cluster_k_spin.setValue(4)
    assert wb._cluster_min_size_spin.value() == 4


def test_the_controller_keys_are_unchanged(wb):
    """The combos' DATA is what the controller reads (S4c:3136, :3269)."""
    got = []
    wb.compute_clusters_requested.connect(got.append)
    wb.compute_umap_requested.connect(got.append)
    wb._cluster_algo_combo.setCurrentIndex(2)
    wb._on_compute_clusters()
    wb._on_compute_umap()
    assert got[0]["algorithm"] == "kmeans"
    assert got[0]["input_space"] == "PCA"
    assert got[1]["input_space"] == "PCA"
    assert "Robust" in wb._scaling_combo.itemText(1)


# --------------------------------------------------------------------------
# the Features tab
# --------------------------------------------------------------------------

def test_feature_columns(wb):
    from napari_epicycle.segmentation import _workbench_text as T

    t = wb.feature_table
    heads = [t.horizontalHeaderItem(i).text() for i in range(t.columnCount())]
    assert heads == list(T.FEATURE_COLUMNS)
    assert heads[wb.COL_NORM] == "Bell"
    assert wb.reset_defaults_btn.text() == "Defaults"
    assert wb.auto_transform_btn.text() == "Auto rescale"
    assert wb._sync_colors_cb.text() == "Colour plot by this"


def test_category_headers_and_channel_groups(wb):
    texts = []
    t = wb.feature_table
    for r in range(t.rowCount()):
        it = t.item(r, wb.COL_CHECK)
        if it is not None and str(it.data(256) or "").startswith("__category__"):
            texts.append(it.text())
    assert any(x.startswith("▾ Membrane channel ") for x in texts), texts
    assert any(x.startswith("▾ Nuclei channel ") for x in texts), texts
    assert all(" used" in x and " of " in x for x in texts)


def test_category_header_counts_follow_ticks(wb):
    wb._deselect_all()
    t = wb.feature_table
    heads = [t.item(r, wb.COL_CHECK).text() for r in range(t.rowCount())
             if str(t.item(r, wb.COL_CHECK).data(256) or "")
             .startswith("__category__")]
    assert all("  0 of " in h for h in heads), heads


def test_bell_column_is_a_dot(wb):
    from qtpy.QtCore import Qt

    t = wb.feature_table
    dots = {t.item(r, wb.COL_NORM).text() for r in range(t.rowCount())
            if t.item(r, wb.COL_CHECK) is not None
            and bool(t.item(r, wb.COL_CHECK).flags()
                     & Qt.ItemFlag.ItemIsUserCheckable)}
    assert dots <= {"●", "○"} and dots


def test_skew_look():
    from napari_epicycle.segmentation.analysis_widget import (
        _ACCENT, _AMBER, skew_look)

    assert skew_look(2.0) == ("↗ 2.0", _AMBER)
    assert skew_look(-1.5) == ("↘ -1.5", _AMBER)
    assert skew_look(0.2) == ("→ 0.2", _ACCENT)
    assert skew_look(0.7)[1] == "#bbbbbb"
    assert skew_look(float("nan"))[0] == "—"


# --------------------------------------------------------------------------
# help lines reach their controls
# --------------------------------------------------------------------------

def test_every_help_line_reaches_a_tooltip(wb):
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.segmentation._workbench_text import WORKBENCH_HELP_LINES

    tips = set()
    for w in [wb] + wb.findChildren(QWidget):
        tips.update(w.toolTip().split("\n"))
    t = wb.feature_table
    for c in range(t.columnCount()):
        tips.add(t.horizontalHeaderItem(c).toolTip())
    missing = [k for k, v in WORKBENCH_HELP_LINES.items() if v not in tips]
    assert not missing, missing


# --------------------------------------------------------------------------
# the audit fixes (V1-*, W1-*)
# --------------------------------------------------------------------------

def _plot_h(wb):
    return wb.qc_plot_widget._plot_widget.height()


def _ready(wb, app):
    """The dock at 578 x 900 with its first divider reset done."""
    _pump(app, 6)
    assert wb._divider_was_reset
    return _plot_h(wb)


def test_the_busy_strip_keeps_the_plot_height(wb, app):
    """V1-2: a chrome line that appears grows the top pane, the tabs give."""
    from napari_epicycle.segmentation import _workbench_text as T

    before = _ready(wb, app)
    tabs_before = wb._tabs.height()
    wb.set_busy_strip(T.BUSY.format(map="UMAP", s=45))
    _pump(app, 6)
    assert abs(_plot_h(wb) - before) <= 1, (before, _plot_h(wb))
    assert wb._tabs.height() < tabs_before
    wb.set_busy_strip(None)
    _pump(app, 6)
    assert abs(_plot_h(wb) - before) <= 1
    assert abs(wb._tabs.height() - tabs_before) <= 1


def test_a_card_message_keeps_the_plot_height(wb, app):
    before = _ready(wb, app)
    wb.show_card_message("x " * 80, "ok")
    _pump(app, 6)
    assert abs(_plot_h(wb) - before) <= 1


def test_the_tabs_keep_their_minimum(wb, app):
    from napari_epicycle.segmentation.analysis_widget import WORKBENCH_TABS_MIN_PX

    _ready(wb, app)
    total = sum(wb._splitter.sizes())
    wb._splitter.setSizes([total - WORKBENCH_TABS_MIN_PX,
                           WORKBENCH_TABS_MIN_PX])
    wb._on_splitter_moved()
    _pump(app)
    wb.set_busy_strip("x")
    _pump(app, 6)
    assert wb._splitter.sizes()[1] >= WORKBENCH_TABS_MIN_PX - 1


def test_the_wide_layout_gives_the_plot_340(wb, app):
    from napari_epicycle.segmentation.analysis_widget import (
        WORKBENCH_PLOT_DEFAULT_PX, WORKBENCH_PLOT_DEFAULT_WIDE_PX)

    assert abs(_ready(wb, app) - WORKBENCH_PLOT_DEFAULT_PX) <= 12
    wb.resize(WIDE_W, 1000)
    _pump(app, 6)
    assert wb.is_wide()
    assert wb.plot_default_px() == WORKBENCH_PLOT_DEFAULT_WIDE_PX == 340
    assert abs(_plot_h(wb) - WORKBENCH_PLOT_DEFAULT_WIDE_PX) <= 12, _plot_h(wb)


def test_a_dragged_divider_is_not_reset_by_a_width_change(wb, app):
    _ready(wb, app)
    total = sum(wb._splitter.sizes())
    wb._splitter.setSizes([total - 200, 200])
    wb._on_splitter_moved()
    _pump(app)
    moved = wb._splitter.sizes()
    wb.resize(WIDE_W, 900)
    _pump(app, 6)
    assert wb._splitter.sizes()[0] == moved[0]


def test_the_width_change_is_applied_a_loop_turn_later(wb, app):
    """W1-7: the re-parent never runs inside Qt's resize dispatch."""
    from qtpy.QtGui import QResizeEvent
    from qtpy.QtCore import QSize

    wb.resizeEvent(QResizeEvent(QSize(WIDE_W, 900), QSize(NARROW_W, 900)))
    assert not wb.is_wide()                 # not yet
    wb._pending_width = WIDE_W
    wb._apply_pending_width()
    assert wb.is_wide()


def test_clear_is_always_shown_and_greyed_without_a_selection(wb, app):
    btn = wb.clear_selection_btn
    assert not btn.isHidden()
    assert not btn.isEnabled()
    wb.qc_plot_widget.highlight_labels([1, 2, 3])
    assert btn.isEnabled() and not btn.isHidden()
    wb.qc_plot_widget._clear_selection()
    assert not btn.isEnabled() and not btn.isHidden()


def test_no_empty_band_at_the_card_foot(wb):
    assert wb.tier_summary_label.isHidden()
    wb.tier_summary_label.setText("All cells: good 1 · bad 2 · unmarked 3")
    assert not wb.tier_summary_label.isHidden()
    wb.tier_summary_label.setText("")
    assert wb.tier_summary_label.isHidden()


def test_the_key_letters_are_dimmed(wb):
    assert wb.mark_good_btn._epicycle_key.text() == "G"
    assert wb.mark_bad_btn._epicycle_key.text() == "B"
    # B3b: live buttons dim their letter to 0.55; grey ones further
    wb.set_mark_state(has_selection=True, can_undo=False)
    assert "0.55" in wb.mark_good_btn._epicycle_key.styleSheet()
    assert wb.mark_good_btn._epicycle_word.text() == "Mark good"
    # the whole word fits (it was cut to "Mark go" once themed)
    for btn in (wb.mark_good_btn, wb.mark_bad_btn):
        word = btn._epicycle_word
        assert word.width() >= word.sizeHint().width(), (
            word.text(), word.width(), word.sizeHint().width())


def test_colour_plot_by_this_is_a_button(wb):
    from qtpy.QtWidgets import QPushButton

    assert isinstance(wb._sync_colors_cb, QPushButton)
    assert wb._sync_colors_cb.isCheckable()


def test_the_feature_header_is_left_aligned(wb):
    from qtpy.QtCore import Qt

    item = wb.feature_table.horizontalHeaderItem(wb.COL_NAME)
    assert Qt.AlignmentFlag(item.textAlignment()) & Qt.AlignmentFlag.AlignLeft


def test_the_gate_line_goes_with_the_gate(wb, app):
    feat = wb.get_selected_features()[0]
    wb._show_feature_detail(feat)
    wb._gate_btn.setChecked(True)
    wb._on_gate_changed()
    assert not wb.sel_msg_label.isHidden()
    wb._gate_btn.setChecked(False)
    assert wb.sel_msg_label.isHidden()
    wb._gate_btn.setChecked(True)
    wb._on_gate_changed()
    assert not wb.sel_msg_label.isHidden()
    wb._close_detail_panel()
    assert wb.sel_msg_label.isHidden()


def _header_row(wb, category):
    return wb._category_header_rows[category][0]


def test_the_caret_folds_a_category(wb):
    cat = next(iter(wb._category_header_rows))
    rows = wb._category_rows[cat]
    assert rows and not any(wb.feature_table.isRowHidden(r) for r in rows)
    ticks = wb.get_selected_features()
    wb.on_category_header_clicked(cat, ctrl=False)
    assert all(wb.feature_table.isRowHidden(r) for r in rows)
    head = wb.feature_table.item(_header_row(wb, cat), wb.COL_CHECK).text()
    assert head.startswith("▸ ")
    assert wb.get_selected_features() == ticks        # folding ticks nothing
    wb.on_category_header_clicked(cat, ctrl=False)
    assert not any(wb.feature_table.isRowHidden(r) for r in rows)
    assert wb.feature_table.item(_header_row(wb, cat),
                                 wb.COL_CHECK).text().startswith("▾ ")


def test_ctrl_click_ticks_the_whole_category(wb, monkeypatch):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    from napari_epicycle.segmentation import _workbench_text as T

    cat = next(iter(wb._category_header_rows))
    feats = [wb.feature_table.item(r, wb.COL_CHECK).data(Qt.UserRole)
             for r in wb._category_rows[cat]]
    monkeypatch.setattr(QApplication, "keyboardModifiers",
                        staticmethod(lambda: Qt.KeyboardModifier.ControlModifier))
    before = set(feats) <= set(wb.get_selected_features())
    wb._on_cell_clicked(_header_row(wb, cat), wb.COL_NAME)
    on = set(wb.get_selected_features()) & set(feats)
    # all ticked -> none; otherwise -> all
    assert on == (set() if before else set(feats))
    wb._on_cell_clicked(_header_row(wb, cat), wb.COL_NAME)
    on2 = set(wb.get_selected_features()) & set(feats)
    assert on2 == (set(feats) if not on else set())
    assert not any(wb.feature_table.isRowHidden(r)
                   for r in wb._category_rows[cat])   # ctrl never folds
    tip = wb.feature_table.item(_header_row(wb, cat), wb.COL_CHECK).toolTip()
    assert tip == T.CATEGORY_HEADER_TIP


def test_the_shape_is_the_rescaled_values_and_always_green(wb):
    """V1-3: a log-skewed column's sparkline is of its LOG, green even when
    the feature is not ticked."""
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QColor

    from napari_epicycle.segmentation.analysis_widget import _ACCENT
    from napari_epicycle.segmentation.feature_stats import (
        apply_transform, compute_histogram, resolve_auto_method)

    t = wb.feature_table
    row = next(r for r in range(t.rowCount())
               if t.item(r, wb.COL_CHECK) is not None
               and t.item(r, wb.COL_CHECK).data(Qt.UserRole) not in
               (None, "") and not str(t.item(r, wb.COL_CHECK)
                                      .data(Qt.UserRole)).startswith("__")
               and t.item(r, wb.COL_CHECK).checkState() == Qt.Unchecked)
    feat = t.item(row, wb.COL_CHECK).data(Qt.UserRole)
    raw = wb._raw_data[feat]
    assert resolve_auto_method(raw, feat) == "log"      # lognormal data
    want = compute_histogram(apply_transform(raw, "auto", feature_name=feat))
    assert list(wb._spark_hist[feat]["counts"]) == list(want["counts"])
    assert list(want["counts"]) != list(compute_histogram(raw)["counts"])
    img = t.cellWidget(row, wb.COL_SPARK).pixmap().toImage()
    colours = {img.pixelColor(x, y).name() for x in range(img.width())
               for y in range(img.height())}
    assert QColor(_ACCENT).name() in colours


def test_rescale_reads_what_auto_picked(wb):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QComboBox

    t = wb.feature_table
    combos = [t.cellWidget(r, wb.COL_XFRM) for r in range(t.rowCount())]
    combos = [c for c in combos if isinstance(c, QComboBox)]
    assert any(c.currentText() == "auto · log" for c in combos)
    c = combos[0]
    assert c.currentData() == "auto"
    words = [c.itemText(i) for i in range(c.count())]
    assert "none" in words and "square root" in words
    # the data, not the words, is what a restore finds
    wb.restore_transforms({c.property("feature_name"): "rank"})
    assert c.currentData() == "rank"
    del Qt


# --------------------------------------------------------------------------
# W2 (owner live test 2026-10-05): the divider at its top stop
# --------------------------------------------------------------------------

def _pw_rect(pw, widget):
    """``widget``'s rectangle in the plot widget's coordinates."""
    from qtpy.QtCore import QPoint, QRect
    top_left = widget.mapTo(pw, QPoint(0, 0))
    return QRect(top_left, widget.size())


def test_the_divider_at_its_top_stop_never_overlaps_the_plot(wb, app):
    """W2: with a feature colour the axes row wraps to two rows at this
    width; the top pane's minimum must count them, so at the divider's top
    stop the legend sits under the plot and the axes row above it."""
    pw = wb.qc_plot_widget
    pw.plot_type_combo.setCurrentText("Scatter")
    pw._color_combo.setCurrentText(pw._feature_x_combo.itemText(0))
    pw._redraw()
    _pump(app, 6)
    width = wb._pinned.width()
    assert pw.axes_flow.heightForWidth(width) > \
        pw.axes_flow.minimumSize().height(), "the axes row must wrap here"
    assert not pw._wb_gradient.isHidden()
    sp = wb._splitter
    total = sum(sp.sizes())
    sp.setSizes([1, total - 1])              # dragged up as far as it goes
    wb._on_splitter_moved()
    _pump(app, 8)
    plot = _pw_rect(pw, pw._plot_widget)
    legend = _pw_rect(pw, pw._wb_gradient)
    assert legend.top() >= plot.y() + plot.height(), (sp.sizes(), plot, legend)
    lowest = max(_pw_rect(pw, pw.axes_flow.itemAt(i).widget()).bottom()
                 for i in range(pw.axes_flow.count())
                 if not pw.axes_flow.itemAt(i).widget().isHidden())
    assert lowest < plot.y(), (lowest, plot)
    assert plot.height() >= pw._plot_widget.minimumHeight()

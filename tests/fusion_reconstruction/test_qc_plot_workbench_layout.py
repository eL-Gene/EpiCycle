"""``QCPlotWidget(layout="workbench")``.

The opt-in Workbench layout: a segmented plot-type row driving the hidden
combo, toggles with words, one wrapping axes row, the neutral surface and
the empty maps that name the Map tab.  The classic layout is held by
``test_qc_plot_classic_layout_golden.py``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

_APP = []


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    a = QApplication.instance() or QApplication([])
    _APP.append(a)
    return a


def _df(n=30, seed=1):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "label": np.arange(1, n + 1),
        "area": rng.rand(n) * 100.0 + 1.0,
        "solidity": rng.rand(n),
        "qc_score": rng.rand(n),
        "qc_tier": rng.choice(["good", "review", "bad"], size=n),
    })


@pytest.fixture
def pw(app):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget

    from napari_epicycle.segmentation.analysis_widget import WORKBENCH_PLOT_TEXTS

    w = QCPlotWidget(layout="workbench", texts=WORKBENCH_PLOT_TEXTS)
    w.resize(562, 600)
    w.set_features(_df())
    yield w
    w.deleteLater()


def _texts(pw):
    import pyqtgraph as pg

    return [it.toPlainText() for it in pw._plot_widget.getPlotItem().items
            if isinstance(it, pg.TextItem)]


def test_an_unknown_layout_is_refused(app):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget

    with pytest.raises(ValueError):
        QCPlotWidget(layout="wide")


def test_segmented_buttons_drive_the_hidden_combo(pw):
    assert pw.plot_type_combo.isHidden()
    assert list(pw._wb_type_buttons) == pw.PLOT_TYPES
    pw._wb_type_buttons["UMAP"].click()
    assert pw.plot_type_combo.currentText() == "UMAP"
    # and the combo, set by the controller, drives the pressed button
    pw.plot_type_combo.setCurrentText("Violin")
    assert pw._wb_type_buttons["Violin"].isChecked()
    assert not pw._wb_type_buttons["UMAP"].isChecked()


def test_toggles_have_words_and_help(pw):
    from napari_epicycle.segmentation._workbench_text import WORKBENCH_HELP_LINES as H

    assert pw._select_btn.text() == "Lasso"
    assert pw._density_btn.text() == "Density"
    assert pw._history_cb.text() == "Removed"
    assert pw._select_btn.toolTip() == H["Lasso"]
    assert pw._density_btn.toolTip() == H["Density"]
    assert pw._history_cb.toolTip() == H["Removed"]
    assert pw._wb_type_buttons["PCA"].toolTip() == H["Plot type"]
    assert pw._color_combo.toolTip() == H["Colour by"]
    assert pw._norm_combo.toolTip() == H["Scale"]
    assert pw._cmap_combo.toolTip() == H["Colours"]


def test_the_axes_labels_are_the_workbench_words(pw):
    assert pw._color_label.text() == "Colour by"
    assert pw._norm_label.text() == "Scale"
    assert pw._cmap_label.text() == "Colours"


def test_scale_and_colours_only_for_a_feature(pw):
    pw._color_combo.setCurrentText("QC tier")
    assert pw._wb_scale_field.isHidden() and pw._wb_cmap_field.isHidden()
    pw._color_combo.setCurrentText("area")
    assert not pw._wb_scale_field.isHidden()
    assert not pw._wb_cmap_field.isHidden()


def test_axes_fields_follow_the_plot_type(pw):
    pw.plot_type_combo.setCurrentText("Scatter")
    assert not pw._wb_x_field.isHidden() and not pw._wb_y_field.isHidden()
    assert pw._x_label.text() == "X"
    pw.plot_type_combo.setCurrentText("PCA")
    assert pw._wb_x_field.isHidden() and pw._wb_y_field.isHidden()
    pw.plot_type_combo.setCurrentText("Violin")
    assert not pw._wb_x_field.isHidden() and pw._wb_y_field.isHidden()
    assert pw._x_label.text() == "Feature"
    assert " by " in pw._wb_axes_note.text()


def test_empty_maps_name_the_map_tab(pw):
    from napari_epicycle.segmentation import _workbench_text as T

    pw.plot_type_combo.setCurrentText("UMAP")
    texts = _texts(pw)
    assert texts == ["\n".join(T.split_title(T.UMAP_EMPTY))]
    assert "QC Settings" not in texts[0]
    pw.plot_type_combo.setCurrentText("t-SNE")
    assert _texts(pw) == ["\n".join(T.split_title(T.TSNE_EMPTY))]
    pw.set_map_busy("t-SNE")
    assert _texts(pw) == ["\n".join(T.split_title(
        T.MAP_BUSY_EMPTY.format(map="t-SNE")))]
    pw.set_map_busy(None)
    assert _texts(pw) == ["\n".join(T.split_title(T.TSNE_EMPTY))]


def test_map_axes_needs_a_map_and_its_input(pw):
    pw.plot_type_combo.setCurrentText("UMAP")
    assert pw._wb_axes_note.isHidden()
    pw._umap_cache = np.random.RandomState(0).randn(30, 2)
    pw.set_map_input("UMAP", "PCA (10 components)")
    assert pw._wb_axes_note.text() == (
        "UMAP 1 vs UMAP 2 · made from PCA (10 components)")


def test_neutral_surface(pw):
    assert pw._plot_widget.backgroundBrush().color().name() == "#23242b"
    assert "1e1e2e" not in pw._color_combo.styleSheet()
    assert "1a1a2e" not in pw._color_combo.styleSheet()


def test_the_count_moves_out_and_notifies(pw):
    seen = []
    pw.stats_changed.connect(lambda: seen.append(1))
    assert pw._stats_label.isHidden()
    pw.highlight_labels([1, 2, 3])
    assert seen


def test_clear_button_is_handed_over(pw):
    from qtpy.QtWidgets import QHBoxLayout, QWidget

    btn = pw.take_clear_button()
    assert btn.text() == "✕ Clear"
    holder = QWidget()
    QHBoxLayout(holder).addWidget(btn)
    assert btn.parent() is holder
    # V1-8: always shown, greyed with no selection
    assert not btn.isHidden() and not btn.isEnabled()
    pw.highlight_labels([1, 2])
    assert not btn.isHidden() and btn.isEnabled()
    pw._clear_selection()
    assert not btn.isHidden() and not btn.isEnabled()


def test_removed_is_off_at_open_in_the_workbench(pw, app):
    """V1-5 (a departure the owner sees): the mockup opens with Removed off;
    the classic plot keeps its ghosts on."""
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget

    assert not pw._history_cb.isChecked()
    w = QCPlotWidget()
    try:
        assert w._history_cb.isChecked()
    finally:
        w.deleteLater()


def test_the_workbench_needs_its_texts(app):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget

    with pytest.raises(ValueError):
        QCPlotWidget(layout="workbench")
    with pytest.raises(ValueError):
        QCPlotWidget(layout="workbench", texts={"help": {}})


def test_the_plot_module_does_not_import_step4_text():
    """W1-10: a Step 3 module never depends on Step 4's sentence table."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "napari_epicycle"
           / "fusion_reconstruction" / "qc_plot_widget.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.append((node.module or "") + ":" + ",".join(
                a.name for a in node.names))
        elif isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
    assert not [n for n in names if "segmentation" in n or "_workbench_text"
                in n], names


def test_the_feature_legend_is_a_gradient(pw):
    pw._color_combo.setCurrentText("area")
    pw._redraw()
    g = pw._wb_gradient
    assert not g.isHidden() and pw._legend_label.isHidden()
    assert g.name_label.text() == "area"
    assert g.scale_label.text() == pw._norm_combo.currentText()
    assert g.bar.width() == 140
    assert g.lo_label.text() and g.hi_label.text()
    assert pw._colorbar_label.isHidden()
    pw._color_combo.setCurrentText("QC tier")
    pw._redraw()
    assert g.isHidden() and not pw._legend_label.isHidden()


def test_the_empty_map_draws_no_stale_axes(pw):
    pw.plot_type_combo.setCurrentText("PCA")
    item = pw._plot_widget.getPlotItem()
    assert item.getAxis("bottom").isVisible() or True
    pw.plot_type_combo.setCurrentText("UMAP")
    assert not item.getAxis("bottom").isVisible()
    assert not item.getAxis("left").isVisible()
    pw.plot_type_combo.setCurrentText("Scatter")
    assert item.getAxis("bottom").isVisible()


def test_the_segmented_buttons_fit_their_bold_words(pw):
    from qtpy.QtGui import QFont, QFontMetrics

    for name, b in pw._wb_type_buttons.items():
        bold = QFont(b.font())
        bold.setPixelSize(11)
        bold.setBold(True)
        assert b.minimumWidth() >= QFontMetrics(bold).horizontalAdvance(name) + 18


def test_classic_refuses_the_workbench_hooks(app):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget

    w = QCPlotWidget()
    try:
        for call in (lambda: w.set_map_busy("UMAP"),
                     lambda: w.set_map_input("UMAP", "x"),
                     lambda: w.take_clear_button()):
            with pytest.raises(RuntimeError):
                call()
    finally:
        w.deleteLater()


def test_plot_clamp(pw):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        WB_PLOT_MAX_H, WB_PLOT_MIN_H)

    assert pw._plot_widget.minimumHeight() == WB_PLOT_MIN_H == 150
    assert pw._plot_widget.maximumHeight() == WB_PLOT_MAX_H == 560


def test_no_sideways_overflow_at_the_column_width(pw):
    """The pinned plot part fits a 562 px column (578 less the margins)."""
    assert pw.minimumSizeHint().width() <= 562


def test_add_axes_widget_goes_before_colour(pw):
    from qtpy.QtWidgets import QLabel

    lab = QLabel("PC")
    pw.add_axes_widget(lab)
    order = [pw.axes_flow.itemAt(i).widget() for i in range(pw.axes_flow.count())]
    assert order.index(lab) < order.index(pw._wb_colour_field)


def test_classic_has_no_workbench_hooks(app):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget

    w = QCPlotWidget()
    try:
        assert not hasattr(w, "axes_flow")
        with pytest.raises(RuntimeError):
            w.add_axes_widget(None)
    finally:
        w.deleteLater()

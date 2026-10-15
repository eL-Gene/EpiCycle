"""LossPlotWidget: additive val API, palette look, Step 2 defaults (Brick C)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")

PAL = {"accent": "#d9a441", "accent_rgb": (217, 164, 65), "ok_text": "#cccccc"}


@pytest.fixture(scope="module")
def _app():
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def widget(_app):
    from napari_epicycle.hitl_training_annotation.ui.loss_plot_widget import (
        LossPlotWidget)
    w = LossPlotWidget()
    yield w
    w.deleteLater()


def _line_items(w):
    return [i for i in w._plot_widget.plotItem.items
            if hasattr(i, "getData") and i is not w._vline]


def _dashed(item):
    from qtpy.QtCore import Qt
    return item.opts["pen"] is not None and item.opts["pen"].style() == Qt.DashLine


def test_old_positional_calls_work(widget):
    widget.add_round_curve("lateral_0", [1.0, 0.8, 0.6])
    widget.begin_live_round("lateral_1")
    widget.append_live_point(0, 0.9)
    widget.append_live_point(1, 0.7)
    assert widget._live_vx == [] and widget._vals() == {}
    assert not any(_dashed(i) for i in _line_items(widget))


def test_default_look_unchanged(widget):
    from napari_epicycle.hitl_training_annotation.ui import loss_plot_widget as m
    assert widget._palette is None
    assert widget._plot_widget.backgroundBrush().color().name() == m._BG_COLOR
    ax = widget._plot_widget.getAxis("bottom")
    assert ax.pen().color().getRgb()[:3] == (60, 60, 60)
    assert ax.labelText == ""
    widget.begin_live_round("lateral_0")
    widget.append_live_point(0, 0.5)
    assert "#999" not in widget._legend_label.styleSheet()
    assert "(live)" in widget._legend_label.text()


def test_palette_look(widget):
    widget.set_palette(PAL)
    assert widget._plot_widget.backgroundBrush().color().name() == "#23242b"
    ax = widget._plot_widget.getAxis("left")
    assert ax.pen().color().getRgb()[:3] == (50, 50, 60)
    # no axis titles on the palette look: the legend says what the lines are
    assert ax.labelText == ""
    assert widget._plot_widget.getAxis("bottom").labelText == ""
    assert "font-size: 11px" in widget.mode_combo.styleSheet()
    assert "#999999" in widget._legend_label.styleSheet()
    widget.set_palette(None)
    assert widget._plot_widget.getAxis("bottom").labelText == ""


def test_live_val_only_when_given(widget):
    widget.set_palette(PAL)
    widget.begin_live_round("lateral_0/lateral")
    widget.append_live_point(0, 0.9)
    widget.append_live_point(1, 0.8, val_loss=0.85)
    widget.append_live_point(2, 0.7, None)
    widget.append_live_point(3, 0.6, 0.65)
    assert widget._live_vx == [1.0, 3.0]
    assert widget._live_vy == [0.85, 0.65]
    assert widget._live_val_item is not None
    x, y = widget._live_val_item.getData()
    assert list(x) == [1.0, 3.0]
    txt = widget._legend_label.text()
    assert "train" in txt and "0.60" in txt and "validation" in txt and "0.65" in txt


def test_round_val_sparse_dashed_markers(widget):
    widget.set_palette(PAL)
    widget.add_round_curve("b1/lateral", [1.0, 0.9, 0.8, 0.7],
                           val_losses=[(1, 0.95), (3, 0.75)])
    widget.add_round_curve("b1/axial", [1.0, 0.9])  # no val
    dashed = [i for i in _line_items(widget) if _dashed(i)]
    assert len(dashed) == 1
    x, y = dashed[0].getData()
    assert list(x) == [1.0, 3.0] and list(y) == [0.95, 0.75]
    assert dashed[0].opts["symbol"] == "o"


def test_no_val_no_dashed(widget):
    widget.set_palette(PAL)
    widget.add_round_curve("b1/lateral", [1.0, 0.9])
    assert not any(_dashed(i) for i in _line_items(widget))
    assert "validation" not in widget._legend_label.text()


def test_hue_per_model_and_the_newest_bundle_at_full_strength(widget):
    """M4: lateral in the accent, axial in the second hue (#9fb4d8), a
    bundle's models share nothing but the strength; only OLDER bundles fade."""
    widget.set_palette(PAL)
    widget.add_round_curve("b1/lateral", [1, 2])
    widget.add_round_curve("b1/axial", [1, 2])
    widget.add_round_curve("b2/lateral", [1, 2], val_losses=[(1, 0.5)])
    widget.add_round_curve("b2/axial", [1, 2])
    c = widget._round_color
    assert c("b1/lateral", 0) == c("b2/lateral", 2) == (217, 164, 65)
    assert c("b1/axial", 1) == c("b2/axial", 3) == (159, 180, 216)
    assert widget._history_width("b1/lateral") == 1.6
    assert widget._history_width("b1/axial") == 1.4
    alphas = {}
    for i in _line_items(widget):
        pen = i.opts["pen"]
        if pen is not None and not _dashed(i):
            alphas.setdefault(pen.color().getRgb()[:3], set()).add(
                pen.color().alpha())
    # the newest bundle (b2) is at 255, the older (b1) fades
    assert 255 in alphas[(217, 164, 65)] and 100 in alphas[(217, 164, 65)]
    assert 255 in alphas[(159, 180, 216)] and 100 in alphas[(159, 180, 216)]


def test_stats_label_in_panel_words(widget):
    widget.set_palette(PAL)
    widget.begin_live_round("refined_03/lateral", n_epochs=100)
    widget.append_live_point(5, 0.9)
    assert widget._stats_label.text() == "refined_03 · lateral · live"
    widget.finish_live_round()
    widget.add_round_curve("refined_03/lateral", [1.0, 0.9])
    widget.add_round_curve("refined_03/axial", [1.0, 0.8])
    assert widget._stats_label.text() == "refined_03 · both models"
    assert "round" not in widget._stats_label.text()


def test_live_x_axis_is_fixed_to_the_epochs(widget):
    widget.set_palette(PAL)
    widget.begin_live_round("b/lateral", n_epochs=100)
    widget.append_live_point(0, 1.0)
    (lo, hi), _ = widget._plot_widget.plotItem.getViewBox().viewRange()
    assert lo <= 0 and hi >= 100 and hi < 110
    # ~3 % clear of both axes: the epoch-0 point does not sit on the axis
    assert lo <= -2.5 and hi >= 102.5
    widget.finish_live_round()
    assert widget._live_n_epochs is None


def test_legend_uses_line_glyphs_and_two_decimals(widget):
    widget.set_palette(PAL)
    widget.begin_live_round("b/lateral")
    widget.append_live_point(5, 0.4123, val_loss=0.4412)
    txt = widget._legend_label.text()
    assert "—" in txt and "- -" in txt and "0.41" in txt and "0.44" in txt
    assert "0.4123" not in txt


def test_grid_alpha_is_28(widget):
    from napari_epicycle.hitl_training_annotation.ui import loss_plot_widget as m
    assert m._PAL_GRID_ALPHA == 28


def test_clear_all_clears_val(widget):
    widget.add_round_curve("b1/lateral", [1, 2], val_losses=[(1, 0.5)])
    widget.begin_live_round("x")
    widget.append_live_point(0, 1.0, 0.9)
    widget.clear_all()
    assert widget._vals() == {} and widget._live_vx == [] and widget._live_vy == []


def test_palette_hides_the_mode_and_round_combos_keeps_the_stats(widget):
    """Step 3 (a palette) has no mode / round combo: Overlay is the only
    behaviour; the stats line stays.  Step 2 (no palette) keeps both."""
    assert not widget.mode_combo.isHidden()
    widget.set_palette(PAL)
    assert widget.mode_combo.isHidden()
    assert widget._round_selector.isHidden()
    assert not widget._stats_label.isHidden()
    assert widget._mode() == widget.MODES[0]
    widget.set_palette(None)
    assert not widget.mode_combo.isHidden()

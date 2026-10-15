"""The readers the
plot area host needs -- the ghosts toggle, the see-line counts, Colour by
with its colours and scale in one redraw, and what kind of selection is
held.  Workbench only; Step 4 calls none of them.
"""

from __future__ import annotations

import types

import pytest

from tests.fusion_reconstruction import _qc_plot_snap as S


def _wb():
    S.app()
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP3_SCHEME,
        QCPlotWidget,
    )
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

    w = QCPlotWidget(layout="workbench", texts=STEP3_PLOT_TEXTS,
                     palette=STEP3_PLOT_PALETTE, tier_scheme=STEP3_SCHEME)
    w.resize(562, 600)
    return w


@pytest.fixture
def wb():
    w = _wb()
    yield w
    w.deleteLater()


@pytest.fixture
def classic():
    S.app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    w = QCPlotWidget()
    yield w
    w.deleteLater()


def click_at(pw, x, y):
    """Drive the plot's own click handler at data point (x, y)."""
    from qtpy.QtCore import QPointF
    from qtpy.QtWidgets import QApplication

    pw.show()
    QApplication.processEvents()
    pos = pw._plot_widget.plotItem.vb.mapViewToScene(QPointF(x, y))
    pw._on_mouse_clicked(types.SimpleNamespace(scenePos=lambda: pos))


# --- 1. the ghosts toggle -------------------------------------------------------


def test_ghosts_toggled_and_ghosts_shown(wb):
    seen = []
    wb.ghosts_toggled.connect(seen.append)
    assert wb.ghosts_shown() is True
    wb._history_cb.setChecked(False)
    assert wb.ghosts_shown() is False
    wb._history_cb.setChecked(True)
    assert seen == [False, True]
    assert "ghosts_toggled" in vars(wb)


def test_the_ghosts_readers_are_workbench_only(classic):
    assert not hasattr(classic, "ghosts_toggled")
    with pytest.raises(RuntimeError):
        classic.ghosts_shown()


# --- 2. the see-line counts -------------------------------------------------------


def test_see_counts_with_no_table(wb):
    assert wb.see_counts() == {"labels": 0, "ghosts": 0, "selected": 0,
                               "colour_by": "QC status"}


def test_see_counts_follow_the_table_and_the_selection(wb):
    wb.set_features(S.five_tier_df())
    c = wb.see_counts()
    assert (c["labels"], c["ghosts"], c["selected"]) == (55, 5, 0)
    assert c["colour_by"] == wb._color_combo.currentText()
    wb.highlight_labels([1, 2, 3])
    assert wb.see_counts()["selected"] == 3
    wb._toggle_category_selection("qc_tier", "good")
    assert wb.see_counts()["selected"] == len(wb.selected_label_ids()) > 0
    wb.set_colour_by("area")
    assert wb.see_counts()["colour_by"] == "area"


def test_colour_by_changed_is_the_combo_signal(wb):
    wb.set_features(S.five_tier_df())
    seen = []
    wb.colour_by_changed.connect(seen.append)
    wb.set_colour_by("QC tier")
    wb.set_colour_by("area")
    assert seen == ["QC tier", "area"]
    assert "colour_by_changed" in vars(wb)


def test_see_counts_is_workbench_only(classic):
    with pytest.raises(RuntimeError):
        classic.see_counts()
    assert not hasattr(classic, "colour_by_changed")


# --- 3. Colour by with colours and scale, one redraw -----------------------------


def test_set_colour_by_with_colours_and_scale_redraws_once(wb):
    wb.set_features(S.five_tier_df())
    emits = []
    wb.viewer_color_requested.connect(lambda *a: emits.append(a))
    assert wb.set_colour_by("area", colours="Plasma", scale="Log")
    assert emits == [("area", "Plasma", "Log")]
    assert wb._cmap_combo.currentText() == "Plasma"
    assert wb._norm_combo.currentText() == "Log"
    assert wb._wb_gradient.scale_label.text() == "Log"


def test_set_colour_by_refuses_unknown_colours_or_scale(wb):
    wb.set_features(S.five_tier_df())
    before = (wb._color_combo.currentText(), wb._cmap_combo.currentText(),
              wb._norm_combo.currentText())
    assert wb.set_colour_by("area", colours="Rainbow") is False
    assert wb.set_colour_by("area", scale="Cubic") is False
    assert wb.set_colour_by("no such feature", colours="Plasma") is False
    assert (wb._color_combo.currentText(), wb._cmap_combo.currentText(),
            wb._norm_combo.currentText()) == before


def test_set_colour_by_without_keywords_keeps_colours_and_scale(wb):
    wb.set_features(S.five_tier_df())
    wb._cmap_combo.setCurrentText("Magma")
    wb._norm_combo.setCurrentText("Linear")
    emits = []
    wb.viewer_color_requested.connect(lambda *a: emits.append(a))
    assert wb.set_colour_by("solidity")
    assert emits == [("solidity", "Magma", "Linear")]


# --- 6. what kind of selection is held -------------------------------------------


def test_selection_kind(wb):
    df = S.five_tier_df()
    wb.set_features(df)
    assert wb.selection_kind() == ""
    wb._toggle_category_selection("qc_tier", "bad")
    assert wb.selection_kind() == "legend"
    wb._clear_selection()
    assert wb.selection_kind() == ""
    wb.highlight_labels([1, 2, 3])          # a lasso and an echo look alike
    assert wb.selection_kind() == "lasso"
    wb.plot_type_combo.setCurrentText("PCA")
    assert wb.selection_kind() == "kept"
    wb.plot_type_combo.setCurrentText("Scatter")
    wb._clear_selection()
    row = df.iloc[1]
    click_at(wb, float(row["area"]), float(row["solidity"]))
    assert wb.selection_kind() == "click"
    assert wb.selected_label_ids() == [int(row["label"])]


def test_selection_kind_is_workbench_only(classic):
    with pytest.raises(RuntimeError):
        classic.selection_kind()

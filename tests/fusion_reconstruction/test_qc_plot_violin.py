"""The Step 3 QC plot's Violin view (fix/step3-plot-violin, 2026-10-06).

The owner felt "a slight UI hiccup" on the Violin after the plot pass; the
live driver (``drive_step3_plot_pass.py --phase violin``) found it:

* right after a load Colour by is qc_score, a feature, so the violin fell
  back to grouping by "QC status" -- a column no Step 3 table has -- and
  drew NOTHING, under a qc_score gradient legend and the last plot's axis
  titles (shot ``p08v_violin_qc_score``);
* the violins counted the ghosts: on a curated patch every active label is
  curated, yet good / review / bad / garbage violins were drawn from the
  ghosts while the legend said 0, and clicking them selected nothing;
* the Lasso button vanished on Violin (the toolbar shifted) and, left on
  from a map, it swallowed every violin click;
* a violin pick on Border cell read "Legend: , 153 labels.";
* the bottom title overprinted the middle group's word ("review / qc tier").

Each test failed on ``cba8c555``.  Step 4 keeps its own violin (its scheme
leaves ``violin_active_groups`` off) -- the last tests hold that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.fusion_reconstruction import _qc_plot_snap as S


def _pump(n=4):
    from qtpy.QtWidgets import QApplication

    for _ in range(n):
        QApplication.processEvents()


@pytest.fixture
def area():
    S.app()
    from napari_epicycle.fusion_reconstruction._qc_plot_area import QCPlotArea

    a = QCPlotArea()
    a.resize(640, 900)
    a.show()
    _pump()
    yield a
    a.hide()
    a.deleteLater()
    _pump()


@pytest.fixture
def T():
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T

    return T


def curated_df(n=60):
    """Wound_patch_1's shape: every ACTIVE label curated, the ghosts
    carrying the old tiers (good / review / bad / garbage)."""
    df = S.five_tier_df(n)
    ghost = np.arange(n) % 2 == 1
    df["label_status"] = np.where(ghost, "superseded", "active")
    old = ["good", "review", "bad", "garbage"]
    df["qc_tier"] = ["curated" if not g else old[i % 4]
                     for i, g in enumerate(ghost)]
    return df


def _load(a, df):
    pw = a.qc_plot_widget
    pw.set_features(df)
    a.panel.set_data(df)
    a.set_dataframe(df)
    a.set_enabled_state(True, False, False)
    pw._feature_x_combo.setCurrentText("area")
    _pump()
    return pw


def _violin(pw):
    pw._wb_type_buttons["Violin"].click()
    _pump()


def _n_violins(pw):
    import pyqtgraph as pg

    return sum(isinstance(it, pg.FillBetweenItem)
               for it in pw._plot_widget.getPlotItem().items)


def _ticks(pw):
    lv = getattr(pw._plot_widget.getAxis("bottom"), "_tickLevels", None)
    return [t for _, t in lv[0]] if lv else None


def _active(df):
    return (df["label_status"] == "active").to_numpy()


# ---------------------------------------------------------------------------
# What the violin groups by
# ---------------------------------------------------------------------------

def test_a_feature_colour_groups_the_violins_by_qc_tier(area):
    """The default after a load (qc_score) drew no violin at all."""
    df = S.five_tier_df()
    pw = _load(area, df)
    pw.set_colour_by("qc_score")
    _violin(pw)
    assert _n_violins(pw) > 0
    assert pw._wb_axes_note.text() == "area by QC tier"


def test_qc_status_without_its_column_groups_by_qc_tier(area):
    df = S.five_tier_df()
    assert "qc_status" not in df.columns
    pw = _load(area, df)
    pw.set_colour_by("QC status")
    _violin(pw)
    assert _n_violins(pw) > 0
    assert pw._wb_axes_note.text() == "area by QC tier"


def test_a_categorical_colour_still_names_the_groups(area):
    pw = _load(area, S.five_tier_df())
    pw.set_colour_by("Border cell")
    _violin(pw)
    assert pw._wb_axes_note.text() == "area by Border cell"
    assert _ticks(pw) == ["Interior", "Border"]


def test_on_violin_the_legend_is_the_groups_over_the_viewer_gradient(area):
    """The groups' legend for the violins, and -- with a feature Colour
    by -- the gradient under it, the key of the viewer's colours (audit)."""
    pw = _load(area, S.five_tier_df())
    pw.set_colour_by("qc_score")
    _violin(pw)
    assert not pw._legend_label.isHidden()
    assert "QC tier:" in pw._legend_label.text()
    assert not pw._wb_gradient.isHidden()
    assert pw._wb_gradient.name_label.text() == "qc_score"
    pw.set_colour_by("QC tier")                   # a categorical: no gradient
    _pump()
    assert pw._wb_gradient.isHidden()
    pw.set_colour_by("qc_score")
    pw._wb_type_buttons["PCA"].click()            # off Violin: gradient alone
    _pump()
    assert not pw._wb_gradient.isHidden()
    assert pw._legend_label.isHidden()


# ---------------------------------------------------------------------------
# Ghosts are no group
# ---------------------------------------------------------------------------

def test_the_violins_are_drawn_over_active_labels_only(area):
    df = curated_df()
    pw = _load(area, df)
    pw.set_colour_by("QC tier")
    _violin(pw)
    assert _ticks(pw) == ["curated"]
    assert _n_violins(pw) == 1


def test_the_ghosts_toggle_does_not_change_the_violins(area):
    df = curated_df()
    pw = _load(area, df)
    pw.set_colour_by("QC tier")
    _violin(pw)
    before = (_ticks(pw), _n_violins(pw))
    pw._history_cb.setChecked(False)
    _pump()
    assert (_ticks(pw), _n_violins(pw)) == before


def test_a_violin_click_selects_exactly_its_group(area):
    df = S.five_tier_df()
    pw = _load(area, df)
    pw.set_colour_by("QC tier")
    _violin(pw)
    ticks = _ticks(pw)
    active = _active(df)
    for i, word in enumerate(ticks):
        pw._clear_selection()
        _pump()
        pw._handle_violin_click(float(i))
        _pump()
        ids = set(pw.selected_label_ids())
        bucket = word if word != "unassigned" else ""
        want = set(df.loc[active & (df["qc_tier"] == bucket), "label"])
        assert ids == want, word
        assert ids, word


def test_a_click_beside_every_violin_picks_nothing(area):
    """It picked the nearest group however far away the click was."""
    df = curated_df()
    pw = _load(area, df)
    pw.set_colour_by("QC tier")
    _violin(pw)
    assert _ticks(pw) == ["curated"]
    pw._handle_violin_click(2.0)
    _pump()
    assert pw.selected_label_ids() == []
    pw._handle_violin_click(0.3)
    _pump()
    assert pw.selected_label_ids()


def test_a_violin_with_one_finite_value_is_still_a_group(area):
    df = S.five_tier_df()
    act = df.index[df["label_status"] == "active"]
    df.loc[act, "qc_tier"] = "good"
    df.loc[act[0], "qc_tier"] = "bad"             # a group of one
    pw = _load(area, df)
    pw.set_colour_by("QC tier")
    _violin(pw)
    assert _ticks(pw) == ["good", "bad"]


def test_no_values_for_the_feature_says_so(area, T):
    df = S.five_tier_df()
    df["area"] = np.nan
    pw = _load(area, df)
    pw.set_colour_by("QC tier")
    _violin(pw)
    assert _n_violins(pw) == 0
    assert any(T.VIOLIN_NO_VALUES.split("|")[0].format(feat="area") in t
               for t in S.plot_texts(pw))


def test_an_all_nan_feature_takes_no_click_and_shows_no_hint(area, T):
    df = S.five_tier_df()
    df["area"] = np.nan
    pw = _load(area, df)
    pw.set_colour_by("QC tier")
    _violin(pw)
    for x in (0.0, 1.0, 2.0):
        pw._handle_violin_click(x)
        _pump()
        assert pw.selected_label_ids() == []
    area.refresh_see_line()
    assert area.status_label.text() == ""


def test_a_group_with_no_value_is_not_clickable(area):
    df = S.five_tier_df()
    act = (df["label_status"] == "active") & (df["qc_tier"] == "good")
    df.loc[act, "area"] = np.nan                  # "good" draws nothing
    pw = _load(area, df)
    pw.set_colour_by("QC tier")
    _violin(pw)
    ticks = _ticks(pw)
    i = ticks.index("good")
    pw._handle_violin_click(float(i))
    _pump()
    assert pw.selected_label_ids() == []
    j = ticks.index("bad")
    pw._handle_violin_click(float(j))
    _pump()
    assert pw.selected_label_ids()


def test_the_violin_has_no_bottom_title(area):
    """The title overprinted the middle group's word ("review / qc tier");
    the axes note already says "area by QC tier"."""
    pw = _load(area, S.five_tier_df())
    pw.set_colour_by("QC tier")
    _violin(pw)
    ax = pw._plot_widget.getAxis("bottom")
    assert not ax.label.isVisible() or not ax.labelText
    pw._wb_type_buttons["Scatter"].click()
    _pump()
    assert ax.labelText == "area"                  # other plots keep theirs


# ---------------------------------------------------------------------------
# The toolbar on Violin (the area)
# ---------------------------------------------------------------------------

def test_lasso_and_density_stay_in_place_but_greyed_on_violin(area):
    pw = _load(area, S.five_tier_df())
    _violin(pw)
    assert not pw._select_btn.isHidden()           # the toolbar does not move
    assert not pw._select_btn.isEnabled()
    assert not pw._density_btn.isEnabled()
    assert pw._history_cb.isEnabled()              # Ghosts: the histogram
    pw._wb_type_buttons["PCA"].click()
    _pump()
    assert pw._select_btn.isEnabled()
    assert pw._density_btn.isEnabled()


def test_a_lasso_left_on_is_off_on_violin_and_back_after(area):
    pw = _load(area, S.five_tier_df())
    pw._wb_type_buttons["PCA"].click()
    _pump()
    pw._select_btn.setChecked(True)
    _violin(pw)
    assert not pw._select_btn.isChecked()
    pw._wb_type_buttons["UMAP"].click()
    _pump()
    assert pw._select_btn.isChecked()


def test_a_lasso_never_on_stays_off_after_violin(area):
    pw = _load(area, S.five_tier_df())
    pw._wb_type_buttons["PCA"].click()
    _pump()
    assert not pw._select_btn.isChecked()
    _violin(pw)
    pw._wb_type_buttons["UMAP"].click()
    _pump()
    assert not pw._select_btn.isChecked()
    _violin(pw)
    pw._wb_type_buttons["Scatter"].click()
    _pump()
    assert not pw._select_btn.isChecked()


def test_a_violin_click_works_with_the_lasso_left_on(area):
    from qtpy.QtCore import QPointF, Qt

    df = S.five_tier_df()
    pw = _load(area, df)
    pw.set_colour_by("QC tier")
    pw._wb_type_buttons["PCA"].click()
    _pump()
    pw._select_btn.setChecked(True)
    _violin(pw)
    pw._select_btn.setChecked(True)                # even forced back on

    class Click:
        def __init__(self, pos):
            self._pos = pos

        def scenePos(self):
            return self._pos

        def button(self):
            return Qt.LeftButton

    vb = pw._plot_widget.plotItem.vb
    (_x0, _x1), (y0, y1) = vb.viewRange()
    pw._on_mouse_clicked(Click(vb.mapViewToScene(
        QPointF(0.0, y0 + 0.3 * (y1 - y0)))))
    _pump()
    assert pw.selected_label_ids()


def test_the_idle_hint_on_violin_names_a_violin_click(area, T):
    pw = _load(area, S.five_tier_df())
    _violin(pw)
    assert area.status_label.text() == T.STATUS_IDLE_HINT_VIOLIN
    pw._wb_type_buttons["Scatter"].click()
    _pump()
    assert area.status_label.text() == T.STATUS_IDLE_HINT


# ---------------------------------------------------------------------------
# The card's source line
# ---------------------------------------------------------------------------

def test_a_violin_pick_says_violin_and_the_group(area, T):
    df = S.five_tier_df()
    pw = _load(area, df)
    pw.set_colour_by("Border cell")
    _violin(pw)
    pw._handle_violin_click(1.0)
    _pump()
    area.selection_changed("plot")
    n = len(pw.selected_label_ids())
    assert area.current_view() == "selection"
    assert area.card.message()[0] == T.CARD_MSG_VIOLIN.format(
        group="Border", n=n)


def test_a_legend_pick_on_a_non_tier_column_names_it(area, T):
    """It read "Legend: , 153 labels." on Border cell."""
    pw = _load(area, S.five_tier_df())
    pw.set_colour_by("Border cell")
    pw._wb_type_buttons["Scatter"].click()
    _pump()
    pw._toggle_category_selection("is_border_cell", "1")
    _pump()
    area.selection_changed("plot")
    n = len(pw.selected_label_ids())
    assert area.card.message()[0] == T.CARD_MSG_LEGEND.format(
        tier="Border", n=n)


def test_a_tier_legend_pick_is_unchanged(area):
    pw = _load(area, S.five_tier_df())
    pw._toggle_category_selection("qc_tier", "good")
    _pump()
    area.selection_changed("plot")
    assert area.card.message()[0].startswith("Legend: good, ")


# ---------------------------------------------------------------------------
# The divider: the plot keeps its height across the Violin switch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("colour", ["QC tier", "qc_score"])
def test_the_plot_canvas_keeps_its_height_to_violin_and_back(area, colour):
    pw = _load(area, S.five_tier_df())
    pw.set_colour_by(colour)
    pw._wb_type_buttons["PCA"].click()
    _pump(8)
    area.reset_divider()
    _pump(8)
    h0 = pw._plot_widget.height()
    _violin(pw)
    _pump(8)
    assert pw._plot_widget.height() == h0
    pw._wb_type_buttons["PCA"].click()
    _pump(8)
    assert pw._plot_widget.height() == h0


# ---------------------------------------------------------------------------
# Step 4 keeps its violin
# ---------------------------------------------------------------------------

def test_step4_scheme_leaves_the_violin_as_it_was():
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP3_SCHEME, STEP4_SCHEME,
    )

    assert STEP3_SCHEME.violin_active_groups is True
    assert STEP4_SCHEME.violin_active_groups is False


def test_step4_workbench_violin_still_falls_back_to_qc_status():
    S.app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )
    from napari_epicycle.segmentation.analysis_widget import (
        WORKBENCH_PLOT_TEXTS,
    )

    pw = QCPlotWidget(layout="workbench", texts=WORKBENCH_PLOT_TEXTS)
    try:
        pw.set_features(S.five_tier_df())
        pw._feature_x_combo.setCurrentText("area")
        pw.set_colour_by("qc_score")
        pw.plot_type_combo.setCurrentText("Violin")
        _pump()
        assert pw._wb_axes_note.text() == "area by QC status"
        assert pw._select_btn.isHidden()           # Step 4: as before
    finally:
        pw.deleteLater()
        _pump()

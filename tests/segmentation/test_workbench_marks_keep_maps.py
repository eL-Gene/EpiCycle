"""7 / I-17: a mark never clears a map.

Mark, Unmark and Undo keep the PCA, UMAP and t-SNE caches of the real
Workbench plot, keep the selection and the zoom, and colour the unmarked
cells grey in the Workbench only (Step 3's classic plot is untouched).
"""
from __future__ import annotations

import numpy as np
import pytest

from ._workbench_ctrl import pump, select, table, wired


@pytest.fixture
def cw(tmp_path):
    c, wb = wired(tmp_path, table(30, tiers=[""] * 30))
    pw = wb.qc_plot_widget
    n = len(c._qc_features_df)
    pw._pca_cache = np.ones((n, 2))
    pw._umap_cache = np.full((n, 2), 2.0)
    pw._tsne_cache = np.full((n, 2), 3.0)
    pw._pc_projected = np.ones((n, 4))
    yield c, wb
    wb.deleteLater()
    pump()


def _caches(pw):
    return (pw._pca_cache, pw._umap_cache, pw._tsne_cache, pw._pc_projected)


def test_mark_unmark_undo_keep_every_map(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    before = _caches(pw)
    pw.plot_type_combo.setCurrentText("UMAP")
    select(c, [1, 2, 3])
    for act in (c._mark_good, c._mark_bad, c._unmark, c._undo_mark,
                c._undo_mark):
        act()
        after = _caches(pw)
        assert all(a is b for a, b in zip(before, after)), act.__name__
    assert c._tiers_dirty and not c._labels_dirty            # I-7
    assert pw.plot_type_combo.currentText() == "UMAP"


def test_a_mark_keeps_the_plot_selection(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw._color_combo.setCurrentText("QC tier")
    pw._toggle_category_selection("qc_tier", "unmarked")
    pump()
    assert len(c._selected_label_ids) == 30
    c._mark_bad()
    assert pw._selected_categories            # not cleared by the mark
    assert len(c._selected_label_ids) == 30


def test_a_mark_does_not_reset_the_zoom(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw.plot_type_combo.setCurrentText("Scatter")
    vb = pw._plot_widget.getViewBox()
    vb.setRange(xRange=(0.1, 0.2), yRange=(0.1, 0.2), padding=0)
    r0 = vb.viewRange()
    select(c, [4])
    c._mark_good()
    r1 = vb.viewRange()
    assert np.allclose(r0, r1)


def test_the_unassigned_grey_is_workbench_only(cw, tmp_path):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        _DEFAULT_COLOR, QCPlotWidget,
    )

    c, wb = cw
    pw = wb.qc_plot_widget
    pw._color_combo.setCurrentText("QC tier")
    colours = pw._get_point_colors_for(c._qc_features_df)
    assert np.allclose(colours[0, :3], np.array((139, 144, 154)) / 255)
    classic = QCPlotWidget()
    classic.set_features(c._qc_features_df)
    classic._color_combo.setCurrentText("QC tier")
    cl = classic._get_point_colors_for(c._qc_features_df)
    assert np.allclose(cl[0, :3], np.array(_DEFAULT_COLOR) / 255)
    classic._redraw()
    assert "unassigned" not in classic._legend_label.text()
    with pytest.raises(RuntimeError):
        classic.refresh_marks(c._qc_features_df)
    classic.deleteLater()


def test_qc_tier_is_offered_before_the_first_mark(tmp_path):
    """A measured stack with no mark has no qc_tier column: the Workbench
    still offers "QC tier" (every cell grey) and its unassigned entry
    selects every cell (found by the B3b shots on patch_4)."""
    df = table(12).drop(columns=["qc_tier"])
    c, wb = wired(tmp_path, df)
    pw = wb.qc_plot_widget
    assert pw._color_combo.findText("QC tier") >= 0
    pw._color_combo.setCurrentText("QC tier")
    colours = pw._get_point_colors_for(df)
    assert np.allclose(colours[:, :3], np.array((139, 144, 154)) / 255)
    pw._redraw()
    assert "unmarked" in pw._legend_label.text()
    pw._toggle_category_selection("qc_tier", "unmarked")
    pump()
    assert sorted(c._selected_label_ids) == list(range(1, 13))
    select(c, [3])
    c._mark_bad()
    assert c._qc_features_df.set_index("label").loc[3, "qc_tier"] == "bad"
    wb.deleteLater()
    pump()

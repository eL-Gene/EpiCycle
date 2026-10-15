"""The PCA results card is one NAMED table of every
ticked feature against every component (no 20-feature cap), sorted by
|PC1|, with each component's share of the variation as a bar in its
column header; the scree plot is gone."""

from __future__ import annotations

import os

import numpy as np
import pytest

_APP = []


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    a = QApplication.instance() or QApplication([])
    _APP.append(a)
    return a


@pytest.fixture
def wb(app):
    from napari_epicycle.segmentation.analysis_widget import AnalysisControlsWidget

    w = AnalysisControlsWidget()
    yield w
    w.deleteLater()


def _result(n_feat=30, n_pcs=12, seed=4, residual=False):
    rng = np.random.RandomState(seed)
    ev = np.sort(rng.rand(n_pcs))[::-1]
    ev = ev / ev.sum() * 0.9
    r = {"loadings": rng.randn(n_pcs, n_feat) * 0.3,
         "explained_variance_ratio": ev,
         "feature_names": [f"feature_{i:02d}" for i in range(n_feat)],
         "projected": rng.randn(50, n_pcs)}
    if residual:
        r["regress_cols_used"] = ["area"]
    return r


def test_every_feature_is_a_named_row(wb):
    res = _result(n_feat=30)
    wb.show_pca_result(res)
    rows = wb.loadings_rows()
    assert len(rows) == 30                       # no 20-feature cap
    assert sorted(rows) == sorted(res["feature_names"])
    assert wb.loadings_table.columnCount() == 1 + 12
    assert not wb._pca_results_panel.isHidden()


def test_sorted_by_abs_pc1(wb):
    res = _result()
    wb.show_pca_result(res)
    names = res["feature_names"]
    pc1 = {n: abs(res["loadings"][0, j]) for j, n in enumerate(names)}
    got = [pc1[n] for n in wb.loadings_rows()]
    assert got == sorted(got, reverse=True)


def test_cells_are_signed_numbers_tinted_by_sign(wb):
    from napari_epicycle.segmentation.analysis_widget import loading_tint

    res = _result()
    wb.show_pca_result(res)
    t = wb.loadings_table
    names = res["feature_names"]
    for r in range(t.rowCount()):
        j = names.index(t.item(r, 0).text())
        for i in range(3):
            v = res["loadings"][i, j]
            cell = t.item(r, 1 + i)
            assert cell.text() == f"{v:+.2f}"
            assert cell.background().color() == loading_tint(v)
    assert loading_tint(0.5).green() > loading_tint(0.5).red()
    assert loading_tint(-0.5).red() > loading_tint(-0.5).green()


def test_the_header_bars_are_shares_of_pc1(wb):
    res = _result()
    wb.show_pca_result(res)
    hdr = wb._loadings_header
    ev = res["explained_variance_ratio"]
    assert hdr.bar_height(0) == hdr.BAR_MAX_PX == 28
    for i in range(len(ev)):
        assert hdr.bar_height(i) == max(1, round(ev[i] / ev[0] * 28))
    assert wb.loadings_table.horizontalHeader() is hdr


def test_the_scree_plot_is_gone(wb):
    assert not hasattr(wb, "_scree_plot")
    assert not hasattr(wb, "_heatmap_plot")


def test_the_result_line_and_caption(wb):
    from napari_epicycle.segmentation import _workbench_text as T

    res = _result()
    wb.show_pca_result(res)
    v = res["explained_variance_ratio"][:2].sum() * 100
    assert wb._pca_done_label.text() == T.PCA_DONE.format(f=30, v=f"{v:.0f}")
    assert wb._pca_caption_label.text() == T.PCA_TABLE_CAPTION


def test_a_new_pca_clears_the_stale_line(wb):
    wb.show_pca_result(_result())
    wb._mark_embeddings_stale()
    assert not wb._pca_stale_label.isHidden()
    wb.show_pca_result(_result(seed=5))
    assert wb._pca_stale_label.isHidden()


def test_residual_pca_keeps_todays_input_space(wb):
    """The first Start-from item's DATA mirrors today's text exactly."""
    wb.show_pca_result(_result(residual=True))
    assert wb._umap_input_combo.itemData(0) == "Residual PCA"
    wb.show_pca_result(_result())
    assert wb._umap_input_combo.itemData(0) == "PCA"
    assert wb._umap_input_combo.itemText(0) == "PCA components"


def test_the_table_fits_its_card_and_scrolls_sideways_itself(wb, app):
    """Wide tables scroll inside the card; the Map tab never does."""
    wb.resize(578, 900)
    wb.show()
    wb.show_pca_result(_result(n_feat=40, n_pcs=15))
    for _ in range(3):
        app.processEvents()
    wb._tabs.setCurrentIndex(1)
    for _ in range(3):
        app.processEvents()
    scroll = wb._map_scroll
    assert scroll.horizontalScrollBar().maximum() == 0
    assert wb.loadings_table.minimumSizeHint().width() <= 578
    wb.hide()


def test_regress_out_features_left_out_are_said(wb):
    """W1-12 (spec addition): after a residual PCA, PCA_EXCLUDED says how
    many ticked regress-out features were left out of its input."""
    from napari_epicycle.segmentation import _workbench_text as T

    feats = sorted(wb._regress_checkboxes)
    wb._feature_names = feats + ["solidity", "extent"]
    wb._selected_features = set(wb._feature_names)
    wb._pca_method_combo.setCurrentText("Residual PCA")
    sent = []
    wb.compute_pca_requested.connect(sent.append)
    wb._on_compute_pca()
    assert sent and set(sent[0]["features"]) == {"solidity", "extent"}
    wb.show_pca_result(_result(n_feat=2, residual=True))
    assert wb._pca_excluded_label.text() == T.PCA_EXCLUDED.format(k=len(feats))
    assert not wb._pca_excluded_label.isHidden()
    # a plain PCA shows no such line
    wb.show_pca_result(_result(n_feat=2))
    assert wb._pca_excluded_label.isHidden()

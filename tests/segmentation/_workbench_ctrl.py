"""A bare Step 4 controller wired to a REAL Workbench (brick B3b tests).

``wired(tmp_path)`` builds the controller the B0 way (``_step4_fakes``)
and a real ``AnalysisControlsWidget``, connects the signals the way
``_open_analysis_dock`` connects them, and feeds both one table.  No
napari viewer: ``_highlight_selected_in_viewer`` returns without a layer.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ._step4_fakes import controller


def table(n=60, seed=0, tiers=None):
    from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES

    rng = np.random.RandomState(seed)
    cols = {"label": np.arange(1, n + 1)}
    for f in list(PCA_FEATURES)[:8]:
        cols[f] = rng.lognormal(size=n)
    cols["qc_tier"] = (list(tiers) if tiers is not None
                       else [""] * n)
    cols["is_border_cell"] = np.zeros(n, bool)
    return pd.DataFrame(cols)


_APP = []


def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    a = QApplication.instance() or QApplication([])
    if not _APP:
        _APP.append(a)     # never let the application be collected
    return a


def pump(k=3):
    a = app()
    for _ in range(k):
        a.processEvents()


def wired(tmp_path, df=None, *, stack="s.tif"):
    """(controller, workbench) sharing one table, signals connected."""
    app()
    from napari_epicycle.segmentation.analysis_widget import (
        AnalysisControlsWidget,
    )

    c = controller(tmp_path)
    c._current_stack_name = stack
    c._labels_volume = np.zeros((2, 2, 2), np.uint32)
    c._qc_features_df = table() if df is None else df
    c.state_manager.set_stack_status(stack, "qc_extracted")
    c._open_analysis_dock = lambda: None
    wb = AnalysisControlsWidget()
    c._analysis_widget = wb
    wb.mark_good_requested.connect(c._mark_good)
    wb.mark_bad_requested.connect(c._mark_bad)
    wb.unmark_requested.connect(c._unmark)
    wb.undo_mark_requested.connect(c._undo_mark)
    if hasattr(c, "_on_stop_map"):
        wb.stop_map_requested.connect(c._on_stop_map)
    wb.compute_pca_requested.connect(c._on_compute_pca)
    wb.compute_umap_requested.connect(c._on_compute_umap)
    wb.compute_tsne_requested.connect(c._on_compute_tsne)
    wb.compute_clusters_requested.connect(c._on_compute_clusters)
    wb.labels_gated.connect(c._on_eda_gate_selection)
    pw = wb.qc_plot_widget
    pw.labels_selected.connect(c._on_lasso_selection)
    pw.category_selected.connect(c._on_category_selected)
    wb.set_feature_data(c._qc_features_df)
    pw.set_features(c._qc_features_df)
    c._refresh_selection_card()
    return c, wb


def select(c, ids):
    """A selection as the plot's lasso would make it."""
    c._on_lasso_selection(list(ids))

"""Owner live test (2026-10-05): the Workbench dock opened again.

Diagnosed by the live driver ``drive_wb_collapse.py`` on a copy of the
owner's data:

* W1 -- napari's ``QtViewerDockWidget._maybe_add_vertical_stretch``
  appends ``addStretch(next(counter))`` to the docked widget's root layout;
  ``counter`` is module-wide, so a Workbench built after any other
  stretched dock (or after napari's X) got a stretch >= 1 that competed
  with its content area (stretch 1): the content was squashed into the top
  half of the dock.  The Workbench is docked with
  ``add_vertical_stretch=False``.
* W3 -- ③'s Workbench button on an already-loaded stack reset the
  Workbench to its defaults (ticks, PCA on the default features, the saved
  UMAP gone).  The button now only shows a live Workbench; a Workbench it
  must build is filled from the stack's saved EDA state, as a load is.
* W5 -- a Workbench built by the button wrote no totals line.
* W6 -- the empty dock napari's X leaves was purged on every load and never
  went: its deferred delete was posted at a scope the purge could not flush.

A real napari ``QtViewerDockWidget`` in a real ``QMainWindow`` (the window's
``add_dock_widget`` / ``remove_dock_widget`` are napari's own, minus the
viewer), a real ``ViewerModel`` for the layers.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import Qt, QTimer

from ._step4_panel import close, make_workspace, panel, pump

napari = pytest.importorskip("napari")

TITLE = "EDA Workbench"


class _Window:
    """napari's ``Window`` as far as Step 4 uses it: ``_qt_window``,
    ``add_dock_widget`` (napari's real ``QtViewerDockWidget``, recording the
    keywords) and ``remove_dock_widget`` (napari's body)."""

    def __init__(self, main):
        from qtpy.QtWidgets import QWidget
        self._qt_window = main
        self._dummy_qt_viewer = QWidget(main)    # a parent for the title bar
        self._dummy_qt_viewer.hide()
        self.calls = []

    def add_dock_widget(self, widget, **kw):
        from napari._qt.widgets.qt_viewer_dock_widget import (
            QtViewerDockWidget,
        )
        self.calls.append(dict(kw))
        dock = QtViewerDockWidget(
            self._dummy_qt_viewer, widget, name=kw.get("name", ""),
            area=kw.get("area", "right"),
            add_vertical_stretch=kw.get("add_vertical_stretch", True))
        self._qt_window.addDockWidget(dock.qt_area, dock)
        dock.show()
        return dock

    def remove_dock_widget(self, widget, menu=None):
        from qtpy.QtWidgets import QDockWidget
        dw = widget
        if not isinstance(widget, QDockWidget):
            for d in self._qt_window.findChildren(QDockWidget):
                if d.widget() is widget:
                    dw = d
                    break
            else:
                raise LookupError("no dock holds this widget")
        if dw.widget():
            dw.widget().setParent(None)
        self._qt_window.removeDockWidget(dw)
        dw.deleteLater()


class _Viewer:
    """A ``ViewerModel`` with a ``window`` (``ViewerModel`` has none)."""

    def __init__(self, model, window):
        self.__dict__["_model"] = model
        self.__dict__["window"] = window

    def __getattr__(self, name):
        return getattr(self.__dict__["_model"], name)


def _df(n=40, seed=0):
    import pandas as pd

    from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES

    rng = np.random.RandomState(seed)
    cols = {"label": np.arange(1, n + 1)}
    for f in list(PCA_FEATURES) + ["extra_a", "extra_b"]:
        cols[f] = rng.lognormal(size=n)
    cols["qc_score"] = rng.rand(n)
    cols["qc_tier"] = (["good"] * 5 + ["bad"] * 3 + [""] * (n - 8))
    cols["is_border_cell"] = np.zeros(n, bool)
    cols["label_status"] = ["active"] * n
    return pd.DataFrame(cols)


@pytest.fixture
def setup(qapp, tmp_path, monkeypatch):
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QMainWindow

    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_pca_full, save_eda_state, save_patch_csv,
    )
    from napari_epicycle.ui_shell.layer_lifecycle_manager import (
        LayerLifecycleManager,
    )
    ws = make_workspace(tmp_path, patches=("P1.tif",), channels=2)
    w = panel(qapp, ws)
    c = w.controller
    main = QMainWindow()
    main.resize(1400, 1000)
    main.show()
    window = _Window(main)
    model = ViewerModel()
    c.viewer = _Viewer(model, window)
    c.layer_lifecycle_manager = LayerLifecycleManager(model)
    run = ws / "artifacts" / "step4" / "runs" / "r1"
    run.mkdir(parents=True)
    n = 40
    lab = np.zeros((6, 24, 24), np.uint32)
    for i in range(n):
        lab[i % 6, (i // 6) * 3:(i // 6) * 3 + 2, 0:2] = i + 1
    tifffile.imwrite(str(run / "labels_3d.tif"), lab)
    df = _df(n)
    csv = ws / "artifacts" / "step4" / "qc" / "P1_features.csv"
    csv.parent.mkdir(parents=True)
    save_patch_csv(df, str(csv))
    c.state_manager.set_stack_status(
        "P1.tif", "qc_extracted", labels_path=str(run / "labels_3d.tif"),
        qc_features_csv=str(csv))
    # The SAVED Workbench: five ticked features (not the default set),
    # their PCA, and a UMAP.
    saved = ["extra_a", "extra_b"] + list(df.columns[1:4])
    pca = compute_pca_full(df, n_components=4, mode="qc",
                           exclude_border=True, features=saved)
    umap = np.random.RandomState(1).rand(n, 2)
    save_eda_state(
        c._eda_dir(), "P1.tif", pca_result=pca,
        pca_scatter_cache=pca["projected"][:, :2].copy(), umap_cache=umap,
        df=df, ui_state={"selected_features": saved, "transforms": {},
                         "preset_name": "Custom", "pca_mode": "qc",
                         "scaling": "qc", "winsorize_enabled": False,
                         "winsorize_value": 0.01, "cluster_seed": 0})
    c._refresh_stacks()
    c._step4_active = True
    pump(qapp)
    launched = []
    monkeypatch.setattr(c, "_launch_eda",
                        lambda kind, params, *a, **k: launched.append(kind))
    yield {"w": w, "c": c, "main": main, "window": window, "lab": lab,
           "saved": saved, "umap": umap, "launched": launched, "df": df}
    c.viewer = None
    close(w)
    main.close()
    main.deleteLater()
    pump(qapp)


def _load(qapp, s):
    c = s["c"]
    c._apply_loaded("P1.tif", {"labels": s["lab"].copy(), "n_cells": 40,
                               "backup": "", "raw": None, "source_path": ""})
    pump(qapp)
    return c


def _docks(main):
    from qtpy.QtWidgets import QDockWidget
    return [d for d in main.findChildren(QDockWidget)
            if str(d.windowTitle()) == TITLE]


def _x(qapp, s):
    """napari's title-bar X: ``destroyOnClose`` -> ``remove_dock_widget``,
    at the top of the event loop."""
    dock = s["c"]._find_eda_dock()
    assert dock is not None
    s["window"].remove_dock_widget(dock)
    pump(qapp)


def _state(c):
    aw = c._analysis_widget
    pw = aw.qc_plot_widget
    pca = aw._pca_result
    return {"ticks": list(aw.get_selected_features()),
            "pca_features": list(pca["feature_names"]) if pca else None,
            "umap": None if pw._umap_cache is None else np.array(
                pw._umap_cache)}


def _same_as_saved(st, s):
    assert sorted(st["ticks"]) == sorted(s["saved"])
    assert sorted(st["pca_features"]) == sorted(s["saved"])
    assert st["umap"] is not None
    assert np.allclose(st["umap"], s["umap"])


# --------------------------------------------------------------------------
# W1
# --------------------------------------------------------------------------

def test_the_workbench_is_docked_without_napari_stretch(qapp, setup):
    c = _load(qapp, setup)
    calls = [k for k in setup["window"].calls if k.get("name") == TITLE]
    assert calls, setup["window"].calls
    assert all(k.get("add_vertical_stretch") is False for k in calls)


def test_a_later_workbench_gets_no_competing_stretch(qapp, setup):
    """napari's module-wide counter already past 0 (another stretched dock
    opened earlier, or napari's X): the Workbench's root layout gets no
    spacer with a stretch, and its content area fills the widget under the
    "what you see" row."""
    from napari._qt.widgets import qt_viewer_dock_widget as q
    for _ in range(3):
        next(q.counter)
    c = _load(qapp, setup)
    pump(qapp)
    aw = c._analysis_widget
    root = aw.layout()
    for i in range(root.count()):
        item = root.itemAt(i)
        if item.spacerItem() is not None:
            assert root.stretch(i) == 0, ("napari stretch", i, root.stretch(i))
    main_area = aw._splitter.parentWidget()
    see = root.itemAt(0).widget()
    assert main_area.height() >= aw.height() - see.height() - 1, (
        main_area.height(), aw.height(), see.height())


# --------------------------------------------------------------------------
# W3 / W5
# --------------------------------------------------------------------------

def test_a_load_restores_the_saved_workbench(qapp, setup):
    c = _load(qapp, setup)
    _same_as_saved(_state(c), setup)


def test_the_button_twice_changes_nothing(qapp, setup):
    c = _load(qapp, setup)
    aw = c._analysis_widget
    before = _state(c)
    umap_obj = aw.qc_plot_widget._umap_cache
    setup["launched"].clear()
    c.ui.open_analysis_btn.click()
    pump(qapp)
    c.ui.open_analysis_btn.click()
    pump(qapp)
    assert c._analysis_widget is aw                 # the same Workbench
    after = _state(c)
    assert after["ticks"] == before["ticks"]
    assert after["pca_features"] == before["pca_features"]
    assert aw.qc_plot_widget._umap_cache is umap_obj
    assert setup["launched"] == []
    _same_as_saved(after, setup)


def test_x_then_the_button_brings_the_saved_workbench_back(qapp, setup):
    c = _load(qapp, setup)
    old = c._analysis_widget
    _x(qapp, setup)
    c.ui.open_analysis_btn.click()
    pump(qapp)
    assert c._analysis_widget is not old            # built again
    _same_as_saved(_state(c), setup)
    assert setup["launched"] == []


def test_a_rebuilt_workbench_writes_its_totals_line(qapp, setup):
    c = _load(qapp, setup)
    _x(qapp, setup)
    c.ui.open_analysis_btn.click()
    pump(qapp)
    text = c._analysis_widget.tier_summary_label.text()
    assert text and "5" in text and "3" in text, text


def test_after_a_step_round_trip_the_button_keeps_the_workbench(qapp, setup,
                                                              monkeypatch):
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: False)
    c = _load(qapp, setup)
    aw = c._analysis_widget
    c.on_step_exit()
    c.on_step_enter()
    pump(qapp)
    c.ui.open_analysis_btn.click()
    pump(qapp)
    assert c._analysis_widget is aw
    assert not c._find_eda_dock().isHidden()
    _same_as_saved(_state(c), setup)


# --------------------------------------------------------------------------
# W6
# --------------------------------------------------------------------------

def test_x_then_a_load_leaves_exactly_one_workbench_dock(qapp, setup,
                                                        capsys):
    """The load's finish runs inside an event delivery (a worker's signal),
    a deeper scope than napari's X posted its deferred delete at."""
    c = _load(qapp, setup)
    _x(qapp, setup)
    QTimer.singleShot(0, lambda: _load(qapp, setup))
    pump(qapp, 6)
    assert len(_docks(setup["main"])) == 1, [
        d.widget() for d in _docks(setup["main"])]
    capsys.readouterr()
    QTimer.singleShot(0, lambda: _load(qapp, setup))
    pump(qapp, 6)
    assert len(_docks(setup["main"])) == 1
    assert "purging a stale" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# Audit of W1-W6 (2026-10-05): MED-1, LOW-2, LOW-3
# --------------------------------------------------------------------------

def _measure_style_selection(c, feats):
    """What a Measure with a non-default feature set leaves behind: the
    dialog's selection remembered, the saved EDA state deleted, no Save."""
    c._invalidate_eda_cache_on_disk()
    c._last_pca_feature_selection = list(feats)
    c._last_pca_mode = "qc"
    c._last_pca_preset_name = "Custom"


def _ticks_match_pca(c):
    aw = c._analysis_widget
    pca = aw._pca_result
    assert pca is not None
    assert sorted(aw.get_selected_features()) == sorted(pca["feature_names"])


def test_med1_rebuild_with_nothing_saved_computes_the_pca_on_the_ticks(
        qapp, setup):
    c = _load(qapp, setup)
    feats = ["extra_a", "extra_b", setup["df"].columns[1]]
    _measure_style_selection(c, feats)
    _x(qapp, setup)
    c.ui.open_analysis_btn.click()
    pump(qapp)
    assert sorted(c._analysis_widget.get_selected_features()) == sorted(feats)
    _ticks_match_pca(c)


def test_med1_the_open_dock_path_computes_the_pca_on_the_ticks(qapp, setup):
    c = _load(qapp, setup)
    feats = ["extra_a", "extra_b", setup["df"].columns[1]]
    _measure_style_selection(c, feats)
    c._cached_eda_state = None
    c._open_analysis_dock()                 # the live dock's re-populate
    pump(qapp)
    _ticks_match_pca(c)


def test_low2_a_rebuild_that_lost_maps_says_so(qapp, setup):
    from napari_epicycle.segmentation import _workbench_text as WT
    c = _load(qapp, setup)
    pw = c._analysis_widget.qc_plot_widget
    pw._tsne_cache = np.random.RandomState(5).rand(40, 2)   # never saved
    _x(qapp, setup)
    c.ui.open_analysis_btn.click()
    pump(qapp)
    assert c._analysis_widget.sel_msg_label.text() == \
        WT.WB_REBUILT_FROM_SAVED


def test_low2_a_rebuild_that_lost_nothing_says_nothing(qapp, setup):
    from napari_epicycle.segmentation import _workbench_text as WT
    c = _load(qapp, setup)
    _x(qapp, setup)
    c.ui.open_analysis_btn.click()
    pump(qapp)
    assert c._analysis_widget.sel_msg_label.text() != \
        WT.WB_REBUILT_FROM_SAVED


def test_low3_a_rebuilt_workbench_keeps_the_undo_history(qapp, setup):
    c = _load(qapp, setup)
    c._mark_history = [{"ids": [1], "old": [""], "new": ["good"]}]
    _x(qapp, setup)
    c.ui.open_analysis_btn.click()
    pump(qapp)
    assert c._analysis_widget.undo_mark_btn.isEnabled()


def test_low3_a_map_running_at_the_x_shows_on_the_rebuilt_workbench(
        qapp, setup):
    import time
    c = _load(qapp, setup)
    c._eda_running = True
    c._eda_stopping = False
    c._eda_abandoned = False
    c._eda_launch = {"kind": "umap", "token": ("P1.tif",), "params": {},
                     "stack": "P1.tif", "n": 40, "t0": time.monotonic(),
                     "plot": "PCA"}
    try:
        _x(qapp, setup)
        c.ui.open_analysis_btn.click()
        pump(qapp)
        aw = c._analysis_widget
        assert not aw.busy_strip.isHidden()
        assert aw.busy_stop_btn.isEnabled()
        assert not aw.compute_umap_btn.isEnabled()        # BUSY_ONE
    finally:
        c._eda_running = False
        c._eda_launch = None

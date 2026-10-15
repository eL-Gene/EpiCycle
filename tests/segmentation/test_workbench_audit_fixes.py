"""The B3b audits' MED / LOW fixes (E1-1..E1-4, E1-7, X1-2..X1-5, X1-8)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from napari_epicycle.segmentation import _eda_worker as EW
from napari_epicycle.segmentation import _workbench_text as T

from ._workbench_ctrl import app, pump, select, table, wired


def _wait(pred, timeout=60.0):
    a = app()
    t0 = time.monotonic()
    while not pred():
        a.processEvents()
        time.sleep(0.01)
        if time.monotonic() - t0 > timeout:
            raise AssertionError("timed out")
    for _ in range(5):
        a.processEvents()


class _Gate:
    def __init__(self, monkeypatch, kind, result=None):
        self.release = threading.Event()
        self.entered = threading.Event()
        real = EW._RUNNERS[kind]

        def runner(params, inputs):
            self.entered.set()
            self.release.wait(30)
            return (result or real)(params, inputs)

        monkeypatch.setitem(EW._RUNNERS, kind, runner)


def _fake_umap(params, inputs):
    return {"projected": np.full((len(inputs["df"]), 2), 7.0),
            "input": T.INPUT_FEATURES, "log": ""}


def _umap_params(features=("x",)):
    return {"features": list(features), "transforms": {}, "scaling": "eda",
            "winsorize": None, "n_neighbors": 5, "min_dist": 0.1,
            "input_space": "Features", "pca_dims": 3, "metric": "euclidean"}


@pytest.fixture
def cw(tmp_path):
    c, wb = wired(tmp_path, table(40, tiers=[""] * 40))
    c.ui.log_output.lines.clear()
    yield c, wb
    th = c._eda_thread
    if th is not None:
        _wait(lambda: th.isFinished())
    wb.deleteLater()
    pump()


# ── E1-1 ──

def test_a_scaling_change_during_a_pca_lands_stale(cw, monkeypatch):
    c, wb = cw
    feats = [f for f in c._qc_features_df.columns
             if f not in ("label", "qc_tier", "is_border_cell")]
    wb.restore_selected_features(feats)
    gate = _Gate(monkeypatch, "pca")
    wb.compute_pca_btn.click()
    assert gate.entered.wait(10)
    wb._winsorize_cb.setChecked(True)             # changed mid-run
    gate.release.set()
    _wait(lambda: not c._eda_running)
    assert c._last_pca_projected is not None
    assert not wb._pca_stale_label.isHidden()


def test_an_unchanged_pca_lands_fresh(cw):
    c, wb = cw
    feats = [f for f in c._qc_features_df.columns
             if f not in ("label", "qc_tier", "is_border_cell")]
    wb.restore_selected_features(feats)
    wb.compute_pca_btn.click()
    _wait(lambda: not c._eda_running)
    assert wb._pca_stale_label.isHidden()


# ── E1-2 ──

def test_shutdown_parks_a_running_map_and_no_handler_runs(cw, monkeypatch):
    c, wb = cw
    gate = _Gate(monkeypatch, "umap", _fake_umap)
    c._on_compute_umap(_umap_params())
    assert gate.entered.wait(10)
    thread = c._eda_thread
    c.shutdown()
    assert c._eda_thread is None and not c._eda_running
    assert any(t is thread for t, _w in EW._LIVE_EDA_THREADS)
    gate.release.set()
    _wait(lambda: thread.isFinished())
    assert wb.qc_plot_widget._umap_cache is None      # nothing landed
    assert wb.status_label.text() == ""


# ── X1-2 ──

def _land_umap(c):
    n = len(c._qc_features_df)
    c._eda_launch = {"kind": "umap", "token": 1, "params": {},
                     "stack": c._current_stack_name, "n": n,
                     "plot": "Scatter"}
    c._on_eda_finished({"kind": "umap", "token": 1,
                        "projected": np.zeros((n, 2)), "input": "x",
                        "log": ""})


def test_a_landing_map_takes_the_plot_it_was_launched_from(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw.plot_type_combo.setCurrentText("Scatter")
    _land_umap(c)
    assert pw.plot_type_combo.currentText() == "UMAP"


def test_a_landing_map_leaves_a_moved_user_alone(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw.plot_type_combo.setCurrentText("Violin")     # moved since launch
    _land_umap(c)
    assert pw.plot_type_combo.currentText() == "Violin"
    assert wb.status_label.text() == T.MAP_DONE_PICK.format(
        map="UMAP", input="x")                    # LOW-5: where it is


def test_a_landing_map_leaves_a_selection_alone(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw.plot_type_combo.setCurrentText("Scatter")
    select(c, [1, 2])
    _land_umap(c)
    assert pw.plot_type_combo.currentText() == "Scatter"


# ── X1-3 / E1-3 ──

def test_stopped_stays_on_the_strip_then_goes(cw, monkeypatch):
    c, wb = cw
    wb.FLASH_MS = 200
    gate = _Gate(monkeypatch, "umap", _fake_umap)
    c._on_compute_umap(_umap_params())
    assert gate.entered.wait(10)
    wb.busy_stop_btn.click()
    gate.release.set()
    _wait(lambda: not c._eda_running)
    assert not wb.busy_strip.isHidden()
    assert wb.busy_label.text() == T.STOPPED.format(map="UMAP")
    assert wb.busy_stop_btn.isHidden()
    _wait(lambda: wb.busy_strip.isHidden(), timeout=5)


def test_a_change_of_cells_after_stop_leaves_no_stopped(cw, monkeypatch):
    c, wb = cw
    pw = wb.qc_plot_widget
    gate = _Gate(monkeypatch, "umap", _fake_umap)
    c._on_compute_umap(_umap_params())
    assert gate.entered.wait(10)
    wb.busy_stop_btn.click()
    c._workbench_cells_changed("remove bad cells")
    assert c._eda_abandoned
    kept = np.full((40, 2), 3.0)
    pw._umap_cache = kept                # E1-7: a cache seeded afterwards
    gate.release.set()
    _wait(lambda: not c._eda_running)
    assert wb.status_label.text() == ""
    assert wb.busy_strip.isHidden()
    assert pw._umap_cache is kept        # the landing left it alone


# ── X1-4 ──

def test_a_grey_key_says_why_and_does_nothing_else(cw):
    c, wb = cw
    sc = {s.key().toString(): s for s in wb._shortcuts}
    sc["G"].activated.emit()
    assert wb.sel_msg_label.text() == T.MARK_NEEDS_SEL
    sc["Ctrl+Z"].activated.emit()
    assert wb.sel_msg_label.text() == T.UNDO_EMPTY
    assert c._mark_history == []
    wb.set_workbench_lock("Save")
    msg = wb.sel_msg_label.text()
    sc["B"].activated.emit()
    assert wb.sel_msg_label.text() == msg          # the lock line stays


# ── X1-5 ──

def test_busy_text_three_branches():
    assert T.busy_text("umap", 3000, None) == T.BUSY_WARMUP.format(map="UMAP")
    assert T.busy_text("tsne", 3000, None) == T.BUSY.format(map="t-SNE",
                                                            s=20)
    assert T.busy_text("umap", 3000, 0.01) == T.BUSY.format(map="UMAP",
                                                            s=30)
    assert T.busy_text("umap", 165, 0.01) == T.BUSY_FEW.format(map="UMAP")
    assert T.busy_text("pca", 165, None) == T.BUSY_FEW.format(map="PCA")


def test_the_second_umap_quotes_the_measured_rate(cw, monkeypatch):
    c, wb = cw
    monkeypatch.setitem(EW._RUNNERS, "umap", _fake_umap)
    gate = _Gate(monkeypatch, "umap", _fake_umap)
    c._on_compute_umap(_umap_params())
    assert gate.entered.wait(10)
    assert wb.busy_label.text() == T.BUSY_WARMUP.format(map="UMAP")
    gate.release.set()
    _wait(lambda: not c._eda_running)
    assert "umap" not in c._eda_rates        # the warm-up is not a rate
    gate2 = _Gate(monkeypatch, "umap", _fake_umap)
    c._on_compute_umap(_umap_params())
    assert gate2.entered.wait(10)
    time.sleep(0.3)                          # a measurable run
    gate2.release.set()
    _wait(lambda: not c._eda_running)
    rate = c._eda_rates["umap"]              # measured, not hand-set
    assert rate > 0
    gate3 = _Gate(monkeypatch, "umap", _fake_umap)
    c._on_compute_umap(_umap_params())
    assert gate3.entered.wait(10)
    assert wb.busy_label.text() == T.busy_text("umap", 40, rate,
                                               warmed=True)
    gate3.release.set()
    _wait(lambda: not c._eda_running)


# ── E1-4 ──

def test_a_map_from_no_features_is_refused_before_a_worker(cw):
    c, wb = cw
    c._on_compute_umap(_umap_params(features=()))
    assert not c._eda_running and c._eda_thread is None
    assert wb._map_error_label.text() == T.MAP_NEEDS_FEATURES.format(
        map="UMAP")
    c._on_compute_clusters(dict(_umap_params(features=()),
                                algorithm="kmeans"))
    assert not c._eda_running
    assert wb.groups_result_label.text() == T.MAP_NEEDS_FEATURES.format(
        map="Groups")


# ── X1-8 ──

def test_lock_words_are_the_specs():
    assert set(T.LOCK_JOB_WORDS.values()) <= {
        "Predict", "Reconstruct", "Measure", "Smooth", "Save", "Load",
        "Export"}
    assert T.LOCK_JOB_WORDS["_preview_in_flight"] == "Load"
    assert T.LOCK_JOB_WORDS["_mesh_running"] == "Export"

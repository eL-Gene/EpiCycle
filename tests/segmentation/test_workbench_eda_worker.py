"""13 / I-17: maps on a worker (brick B3b).

* ``_EdaWorker.run`` emits exactly one terminal signal from a ``finally``,
  and both reach ``thread.quit`` (a REAL QThread).
* The launch greys Compute / Cluster only (BUSY_ONE); marks stay live and a
  mark during a map never invalidates it: both land.
* Stop is a STATE: no event-loop pump, the answer is dropped, the last map
  is kept, STOPPED at the thread's end, a new map refused until then.
* A stale token (another stack) is discarded; Remove bad cells abandons a
  running map (LOG_MAP_ABANDONED).
"""
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


# ── the worker alone ──

def _pca_params(df):
    feats = [c for c in df.columns if c not in ("label", "qc_tier",
                                               "is_border_cell")]
    return {"features": feats, "transforms": {}, "n_components": 3,
            "pc_x": 0, "pc_y": 1, "scaling": "eda", "winsorize": None,
            "mode": "eda"}


def test_run_emits_finished_once_from_a_real_thread():
    from qtpy.QtCore import QThread

    app()
    df = table(40)
    w = EW.EdaWorker("pca", _pca_params(df), {"df": df}, token=("t",))
    th = QThread()
    w.moveToThread(th)
    got = {"finished": [], "error": []}

    class Sink:     # a plain slot object, kept alive for the test
        def fin(self, o):
            got["finished"].append(o)

        def err(self, o):
            got["error"].append(o)

    sink = Sink()
    w.finished.connect(sink.fin)
    w.error.connect(sink.err)
    w.finished.connect(th.quit)
    w.error.connect(th.quit)
    th.started.connect(w.run)
    th.start()
    _wait(lambda: th.isFinished())
    assert len(got["finished"]) == 1 and got["error"] == []
    out = got["finished"][0]
    assert out["token"] == ("t",) and out["kind"] == "pca"
    assert out["result"]["projected"].shape == (40, 3)


def test_a_failure_emits_error_and_the_thread_quits():
    from qtpy.QtCore import QThread

    app()
    w = EW.EdaWorker("umap", {"input_space": "Features", "features": []},
                     {"df": table(10)}, token=1)
    th = QThread()
    w.moveToThread(th)
    got = []

    class Sink:
        def err(self, o):
            got.append(o)

    sink = Sink()
    w.error.connect(sink.err)
    w.error.connect(th.quit)
    w.finished.connect(th.quit)
    th.started.connect(w.run)
    th.start()
    _wait(lambda: th.isFinished())
    assert got and got[0]["too_few"] is True


def test_run_emits_even_when_the_runner_raises_baseexception(monkeypatch):
    class Boom(BaseException):
        pass

    def boom(*a):
        raise Boom("x")

    monkeypatch.setitem(EW._RUNNERS, "pca", boom)
    w = EW.EdaWorker("pca", {}, {}, token=2)
    got = []
    w.error.connect(got.append)
    w.run()
    assert got and "Boom" in got[0]["message"]


def test_residual_pca_start_from_uses_the_stored_projection():
    proj = np.arange(20.0).reshape(10, 2)
    X, words = EW.map_input({"input_space": "Residual PCA"},
                            {"pca_projected": proj, "pca_residual": True,
                             "df": table(10)})
    assert X is not None and X.shape == (10, 2)
    assert words == T.INPUT_RESIDUAL_PCA.format(n=2)


# ── wired: a controller, a real Workbench, a real QThread ──

@pytest.fixture
def cw(tmp_path):
    c, wb = wired(tmp_path, table(60, tiers=[""] * 60))
    c.ui.log_output.lines.clear()
    yield c, wb
    th = c._eda_thread
    if th is not None:
        _wait(lambda: th.isFinished())
    wb.deleteLater()
    pump()


class _Gate:
    """Holds a map's computation until the test releases it."""

    def __init__(self, monkeypatch, kind, result):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.result = result

        def runner(params, inputs):
            self.entered.set()
            self.release.wait(30)
            return self.result(params, inputs)

        monkeypatch.setitem(EW._RUNNERS, kind, runner)


def _fake_umap(params, inputs):
    n = len(inputs["df"])
    return {"projected": np.full((n, 2), 7.0), "input": T.INPUT_FEATURES,
            "log": "[analysis] UMAP fake"}


def _umap_params():
    return {"features": ["x"], "transforms": {}, "scaling": "eda",
            "winsorize": None, "n_neighbors": 5, "min_dist": 0.1,
            "input_space": "Features", "pca_dims": 3, "metric": "euclidean"}


def test_a_umap_greys_compute_only_and_a_mark_lands_too(cw, monkeypatch):
    c, wb = cw
    gate = _Gate(monkeypatch, "umap", _fake_umap)
    pw = wb.qc_plot_widget
    pw._pca_cache = np.ones((60, 2))
    c._on_compute_umap(_umap_params())
    assert gate.entered.wait(10)
    assert c._eda_running
    assert not c._step4_is_nav_locked() if hasattr(c, "_step4_is_nav_locked") \
        else True
    assert not wb.compute_umap_btn.isEnabled()
    assert not wb.compute_clusters_btn.isEnabled()
    assert wb._map_why_label.text() == T.BUSY_ONE.format(map="UMAP")
    assert wb.busy_label.text().startswith("Computing UMAP")
    # a mark during the map: live, applied, and the map still lands
    select(c, [1, 2, 3])
    assert wb.mark_bad_btn.isEnabled()
    wb.mark_bad_btn.click()
    assert list(c._qc_features_df.loc[:2, "qc_tier"]) == ["bad"] * 3
    gate.release.set()
    _wait(lambda: not c._eda_running)
    assert np.allclose(pw._umap_cache, 7.0)                    # I-17
    assert pw._pca_cache is not None
    # a selection existed, so the map did not take the plot (X1-2)
    assert wb.status_label.text() == T.MAP_DONE_PICK.format(
        map="UMAP", input=T.INPUT_FEATURES)
    assert wb.busy_strip.isHidden()
    assert wb.compute_umap_btn.isEnabled()
    assert c._qc_features_df.loc[0, "qc_tier"] == "bad"


def test_a_second_map_is_refused_with_busy_one(cw, monkeypatch):
    c, wb = cw
    gate = _Gate(monkeypatch, "umap", _fake_umap)
    c._on_compute_umap(_umap_params())
    assert gate.entered.wait(10)
    thread = c._eda_thread
    c._on_compute_tsne(dict(_umap_params(), perplexity=5))
    assert c._eda_thread is thread                 # nothing new started
    assert wb._map_why_label.text() == T.BUSY_ONE.format(map="UMAP")
    gate.release.set()
    _wait(lambda: not c._eda_running)


def test_stop_is_a_state_and_keeps_the_old_map(cw, monkeypatch):
    c, wb = cw
    pw = wb.qc_plot_widget
    old = np.full((60, 2), 3.0)
    pw._umap_cache = old
    gate = _Gate(monkeypatch, "umap", _fake_umap)
    c._on_compute_umap(_umap_params())
    assert gate.entered.wait(10)
    # Stop must not pump the event loop
    import qtpy.QtWidgets as QW
    pumped = []
    with monkeypatch.context() as m:
        m.setattr(QW.QApplication, "processEvents",
                  staticmethod(lambda *a: pumped.append(1)))
        wb.busy_stop_btn.click()
    assert pumped == []
    assert c._eda_stopping and c._eda_running
    assert wb.busy_label.text() == T.STOPPING.format(map="UMAP")
    assert not wb.busy_stop_btn.isEnabled()
    # a new map waits for the thread
    thread = c._eda_thread
    c._on_compute_umap(_umap_params())
    assert c._eda_thread is thread
    gate.release.set()
    _wait(lambda: not c._eda_running)
    assert pw._umap_cache is old                           # the last map
    assert wb.status_label.text() == T.STOPPED.format(map="UMAP")
    assert not c._eda_stopping


def test_a_stale_token_is_discarded(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    c._eda_launch = {"kind": "umap", "token": ("s.tif", (), (), 9),
                     "params": {}, "stack": "s.tif", "n": 60}
    c._on_eda_finished({"kind": "umap", "token": ("s.tif", (), (), 8),
                        "projected": np.zeros((60, 2)), "input": "x"})
    assert pw._umap_cache is None
    # the right token for a stack that is no longer loaded
    c._current_stack_name = "other.tif"
    c._on_eda_finished({"kind": "umap", "token": ("s.tif", (), (), 9),
                        "projected": np.zeros((60, 2)), "input": "x"})
    assert pw._umap_cache is None


def test_remove_bad_cells_abandons_a_running_map(cw, monkeypatch):
    import napari_epicycle.ui_shell.theme as theme

    c, wb = cw
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)
    gate = _Gate(monkeypatch, "umap", _fake_umap)
    vol = np.zeros((2, 8, 8), np.uint32)
    vol.flat[:60] = np.arange(1, 61)
    c._labels_volume = vol
    select(c, [5])
    c._mark_bad()
    c._on_compute_umap(_umap_params())
    assert gate.entered.wait(10)
    c._delete_bad_labels()
    assert c._eda_abandoned
    assert any(T.LOG_MAP_ABANDONED.format(map="UMAP") in line
               for line in c.ui.log_output.lines)
    assert c._mark_history == []
    assert wb.qc_selection_label.text() == T.SEL_NONE
    gate.release.set()
    _wait(lambda: not c._eda_running)
    assert wb.qc_plot_widget._umap_cache is None     # the answer dropped
    assert wb.status_label.text() == ""              # no STOPPED line


def test_a_pca_lands_its_loadings_and_maps_old(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw._umap_cache = np.ones((60, 2))
    params = _pca_params(c._qc_features_df)
    wb.restore_selected_features(params["features"])
    c._on_compute_pca(params)
    _wait(lambda: not c._eda_running)
    assert c._last_pca_projected.shape == (60, 3)
    assert wb.loadings_table.rowCount() == len(params["features"])
    assert not wb._maps_old_label.isHidden()
    assert pw._umap_cache is not None              # not cleared
    # only the axes change: no new worker
    c._on_compute_pca(dict(params, pc_x=2, pc_y=0))
    assert not c._eda_running
    assert np.allclose(pw._pca_cache[:, 0], c._last_pca_projected[:, 2])


def test_pca_with_one_feature_says_too_few(cw):
    c, wb = cw
    c._on_compute_pca({"features": ["x"]})
    assert wb._map_error_label.text() == T.PCA_TOO_FEW
    assert not c._eda_running


def test_a_failed_map_says_so_and_logs_why(cw, monkeypatch):
    c, wb = cw

    def boom(params, inputs):
        raise RuntimeError("umap broke")

    monkeypatch.setitem(EW._RUNNERS, "umap", boom)
    c._on_compute_umap(_umap_params())
    _wait(lambda: not c._eda_running)
    assert wb._map_error_label.text() == T.MAP_FAILED.format(map="UMAP")
    assert any("umap broke" in line for line in c.ui.log_output.lines)

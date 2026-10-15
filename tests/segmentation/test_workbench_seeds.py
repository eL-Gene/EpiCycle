"""Leiden, K-means and GMM take a seed.

* The same seed gives the same groups (Leiden, K-means, GMM) on a
  synthetic table, through the pure functions AND through the worker.
* Step 3's calls (no seed) are unchanged: K-means / GMM keep 42.
* HDBSCAN has no random start: the Seed box is grey with SEED_OFF beside it.
* The seed rides in the Cluster params and is saved in the EDA ui_state.
"""
from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction import qc_features as QF
from napari_epicycle.segmentation import _eda_worker as EW
from napari_epicycle.segmentation import _workbench_text as T

from ._workbench_ctrl import pump, table, wired


def _blobs(n=120, seed=3):
    rng = np.random.RandomState(seed)
    centres = np.array([[0, 0, 0], [6, 0, 0], [0, 6, 0], [0, 0, 6]], float)
    X = np.vstack([c + rng.normal(size=(n // 4, 3)) for c in centres])
    return X


def test_kmeans_and_gmm_default_to_42_as_before():
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    X = _blobs()
    ref_k = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)
    ref_g = GaussianMixture(n_components=4, random_state=42).fit_predict(X)
    assert np.array_equal(QF._cluster_kmeans(X, 4), ref_k)
    assert np.array_equal(QF._cluster_gmm(X, 4), ref_g)


@pytest.mark.parametrize("fn", ["kmeans", "gmm"])
def test_the_same_seed_gives_the_same_groups(fn):
    X = _blobs() + np.random.RandomState(9).normal(scale=2.5, size=(120, 3))
    f = {"kmeans": QF._cluster_kmeans, "gmm": QF._cluster_gmm}[fn]
    a = f(X, 5, seed=7)
    b = f(X, 5, seed=7)
    assert np.array_equal(a, b)


def test_leiden_is_repeatable_with_a_seed():
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")
    X = _blobs() + np.random.RandomState(1).normal(scale=1.5, size=(120, 3))
    a = QF._cluster_leiden(X, 1.0, 10, seed=11)
    b = QF._cluster_leiden(X, 1.0, 10, seed=11)
    assert np.array_equal(a, b)


def test_leiden_without_a_seed_is_step3s_call(monkeypatch):
    """No seed -> no ``seed=`` keyword reaches leidenalg (Step 3)."""
    pytest.importorskip("leidenalg")
    import leidenalg

    seen = {}
    real = leidenalg.find_partition

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(leidenalg, "find_partition", spy)
    QF._cluster_leiden(_blobs(), 1.0, 10)
    assert "seed" not in seen
    QF._cluster_leiden(_blobs(), 1.0, 10, seed=5)
    assert seen["seed"] == 5


@pytest.mark.parametrize("algo", ["leiden", "kmeans", "gmm"])
def test_the_worker_repeats_with_the_same_seed(algo):
    if algo == "leiden":
        pytest.importorskip("leidenalg")
    df = table(80, seed=4)
    feats = [c for c in df.columns
             if c not in ("label", "qc_tier", "is_border_cell")]
    params = {"algorithm": algo, "features": feats, "transforms": {},
              "scaling": "eda", "winsorize": None, "resolution": 1.0,
              "n_neighbors": 10, "n_clusters": 4, "n_components": 5,
              "min_cluster_size": 5, "input_space": "Features",
              "metric": "euclidean", "seed": 123}
    a = EW.run_cluster(params, {"df": df})
    b = EW.run_cluster(params, {"df": df.copy()})
    assert np.array_equal(a["labels"], b["labels"])
    assert a["seed"] == 123 and a["algorithm"] == algo


@pytest.fixture
def cw(tmp_path):
    c, wb = wired(tmp_path)
    yield c, wb
    wb.deleteLater()
    pump()


def test_hdbscan_greys_the_seed_with_its_reason(cw):
    _, wb = cw
    combo = wb._cluster_algo_combo
    combo.setCurrentIndex(combo.findData("hdbscan"))
    assert not wb._cluster_seed_spin.isEnabled()
    assert not wb._seed_off_label.isHidden()
    assert wb._seed_off_label.text() == T.SEED_OFF
    combo.setCurrentIndex(combo.findData("kmeans"))
    assert wb._cluster_seed_spin.isEnabled()
    assert wb._seed_off_label.isHidden()


def test_the_seed_rides_in_the_cluster_params(cw):
    _, wb = cw
    got = []
    wb.compute_clusters_requested.disconnect()
    wb.compute_clusters_requested.connect(got.append)
    wb._cluster_seed_spin.setValue(777)
    wb.compute_clusters_btn.click()
    assert got and got[0]["seed"] == 777


def test_the_seed_is_saved_in_ui_state_and_restored(cw, tmp_path):
    c, wb = cw
    wb._cluster_seed_spin.setValue(4242)
    c._last_pca_projected = np.zeros((len(c._qc_features_df), 2))
    wb.qc_plot_widget._umap_cache = np.zeros((len(c._qc_features_df), 2))
    c._save_eda_state()
    from napari_epicycle.fusion_reconstruction.qc_features import (
        load_eda_state,
    )
    got = load_eda_state(c._eda_dir(), c._current_stack_name,
                         df=c._qc_features_df)
    assert got is not None and got["ui_state"]["cluster_seed"] == 4242
    wb._cluster_seed_spin.setValue(1)
    wb.set_cluster_seed(got["ui_state"]["cluster_seed"])
    assert wb._cluster_seed_spin.value() == 4242


def test_cluster_done_names_method_and_seed(cw):
    c, wb = cw
    n = len(c._qc_features_df)
    c._eda_launch = {"kind": "cluster", "token": 1, "params": {},
                     "stack": c._current_stack_name, "n": n}
    c._on_eda_finished({"kind": "cluster", "token": 1,
                        "labels": np.arange(n) % 3, "algorithm": "kmeans",
                        "n_groups": 3, "noise": 0, "seed": 9,
                        "input": T.INPUT_FEATURES, "log": ""})
    assert wb.groups_result_label.text() == T.CLUSTER_DONE.format(
        k=3, method="K-means", seed=9)
    assert not wb.groups_legend.isHidden()
    assert "G3" in wb._groups_legend_label.text()

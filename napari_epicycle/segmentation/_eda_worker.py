"""The Workbench's maps on a worker.

PCA, UMAP, t-SNE and the grouping used to run on the main thread inside
the controller's ``_on_compute_*`` handlers.  Their heavy bodies moved
here VERBATIM into pure functions over COPIES (the table, the stored PCA
projection, the UMAP), so ``_EdaWorker`` can run one of them on its own
``QThread`` while the user keeps marking cells.

Nothing here touches a widget, a layer or the controller: the worker
gets its inputs at launch and hands back an outcome dict; the controller
decides on the main thread whether it still belongs on screen.

Every ``run()`` ends in a ``try/finally`` that emits exactly one of
``finished`` or ``error`` (a worker that returns without
emitting strands its thread), and both reach ``thread.quit``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from qtpy.QtCore import QObject, Signal

from . import _workbench_text as T

#: The kinds of map job.
EDA_KINDS = ("pca", "umap", "tsne", "cluster")

#: The ``input_space`` values that mean "the stored PCA".  After a
#: residual PCA the Start-from combos carry "Residual PCA" (B3a); before
#: B3b that fell through to the features path, silently (the recorded
#: quirk).  Both are the stored projection now.
PCA_INPUTS = ("PCA", "Residual PCA")


#: Map threads still running at napari's close, parked so Python never
#: destroys a running QThread (the thumbnail pattern, audit E1-2).
_LIVE_EDA_THREADS: set = set()


def stop_eda_thread(thread, worker=None, wait_ms: int = 3000) -> bool:
    """``shutdown()``: quit and wait at most ``wait_ms``; a fit that cannot
    stop in time is parked in ``_LIVE_EDA_THREADS``.  True if stopped.
    The caller disconnects its own slots first."""
    global _LIVE_EDA_THREADS
    _LIVE_EDA_THREADS = {pair for pair in _LIVE_EDA_THREADS
                         if _running(pair[0])}
    if thread is None:
        return True
    try:
        if not thread.isRunning():
            return True
        thread.quit()
        ok = bool(thread.wait(int(wait_ms)))
    except RuntimeError:
        return True
    if not ok:
        _LIVE_EDA_THREADS.add((thread, worker))
        print("[EDA] shutdown: a map is still running -- thread parked")
    return ok


def _running(thread) -> bool:
    try:
        return bool(thread.isRunning())
    except RuntimeError:
        return False


class EdaTooFew(Exception):
    """Too few cells (or features) for this map: MAP_TOO_FEW_CELLS."""


# ---------------------------------------------------------------------------
# Pure computations (moved verbatim from the controller's handlers)
# ---------------------------------------------------------------------------

def feature_matrix(df, params) -> tuple:
    """The rescaled, standardised feature matrix the UMAP, t-SNE and
    grouping "Features" paths build (verbatim from the controller):
    per-feature transforms, a median fill of non-finite values, scaling
    fitted on the interior cells, the optional clip, then NaN -> 0.

    Returns ``(X, n_features)``; ``X`` is ``None`` when no feature is
    usable.
    """
    from napari_epicycle.segmentation.feature_stats import apply_transform

    features = params.get("features") or []
    transforms = params.get("transforms")
    scaling = params.get("scaling", "eda")
    winsorize_val = params.get("winsorize")
    cols = [c for c in features if c in df.columns]
    if not cols:
        return None, 0
    X = df[cols].values.astype(np.float64)
    if transforms:
        for j, col in enumerate(cols):
            method = transforms.get(col, "auto")
            X[:, j] = apply_transform(X[:, j], method, feature_name=col)
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            finite = X[~bad, j]
            X[bad, j] = np.median(finite) if finite.size else 0.0
    exclude = None
    if "is_border_cell" in df.columns:
        exclude = df["is_border_cell"].values.astype(bool)
    fit_mask = ~exclude if exclude is not None and exclude.any() else None
    fit_X = X[fit_mask] if fit_mask is not None else X
    if scaling == "qc":
        for j in range(X.shape[1]):
            med = np.median(fit_X[:, j])
            mad = np.median(np.abs(fit_X[:, j] - med))
            scale = mad * 1.4826
            if scale < 1e-12:
                X[:, j] = 0.0
            else:
                X[:, j] = (X[:, j] - med) / scale
    else:
        mean = fit_X.mean(axis=0)
        std = fit_X.std(axis=0)
        std[std < 1e-12] = 1.0
        X = (X - mean) / std
    if winsorize_val is not None:
        X = np.clip(X, -winsorize_val, winsorize_val)
    return np.nan_to_num(X, nan=0.0), len(cols)


def pca_input_words(pca_projected, residual: bool) -> str:
    n = int(pca_projected.shape[1])
    return (T.INPUT_RESIDUAL_PCA if residual else T.INPUT_PCA).format(n=n)


def map_input(params, inputs) -> tuple:
    """``(X, words)`` for UMAP / t-SNE: the stored PCA when "Start from"
    is PCA and one exists, else the rescaled features (the controller's
    rule, with the Residual PCA quirk fixed)."""
    proj = inputs.get("pca_projected")
    if params.get("input_space", "PCA") in PCA_INPUTS and proj is not None:
        return (np.asarray(proj, dtype=np.float64),
                pca_input_words(proj, bool(inputs.get("pca_residual"))))
    X, _n = feature_matrix(inputs["df"], params)
    if X is None:
        raise EdaTooFew("no features")
    return X, T.INPUT_FEATURES


def run_pca(params, inputs) -> dict:
    """Standard or residual PCA (``compute_pca_full`` /
    ``compute_residual_pca``), verbatim from ``_on_compute_pca``."""
    df = inputs["df"]
    features = params.get("features")
    transforms = params.get("transforms")
    n_components = params.get("n_components", 10)
    if params.get("method", "standard") == "residual":
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_residual_pca,
        )
        result = compute_residual_pca(
            df, n_components=n_components, features=features,
            regress_out=params.get("regress_out") or None,
            exclude_border=True, transforms=transforms,
            scaling=params.get("scaling", "eda"),
            winsorize=params.get("winsorize"),
        )
        log = (f"[analysis] Residual PCA, {n_components} components, "
               f"{len(features or [])} features, "
               f"regressed out: {result.get('regress_cols_used', [])}")
    else:
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_pca_full,
        )
        mode = params.get("mode", "eda")
        winsorize_val = params.get("winsorize")
        result = compute_pca_full(
            df, n_components=n_components, features=features,
            exclude_border=True, mode=mode, transforms=transforms,
            winsorize=winsorize_val,
        )
        scaling_label = "robust" if mode == "qc" else "standard"
        wl = f", winsorize ±{winsorize_val}" if winsorize_val else ""
        log = (f"[analysis] PCA ({scaling_label}{wl}), {n_components} "
               f"components, {len(features or [])} features")
    return {"result": result, "log": log}


def run_umap(params, inputs) -> dict:
    """UMAP, verbatim from ``_on_compute_umap`` (``random_state=42``)."""
    import umap as _umap

    X, words = map_input(params, inputs)
    n_samples = X.shape[0]
    X = np.nan_to_num(X, nan=0.0)
    k = min(params.get("n_neighbors", 15), n_samples - 1)
    if n_samples < 3 or k < 2:
        raise EdaTooFew(n_samples)
    min_dist = params.get("min_dist", 0.1)
    reducer = _umap.UMAP(n_components=2, n_neighbors=k, min_dist=min_dist,
                         metric=params.get("metric", "euclidean"),
                         random_state=42)
    projected = reducer.fit_transform(X)
    log = (f"[analysis] UMAP: n_neighbors={params.get('n_neighbors', 15)}, "
           f"min_dist={min_dist}, input={words}")
    return {"projected": np.asarray(projected), "input": words, "log": log}


def run_tsne(params, inputs) -> dict:
    """t-SNE, verbatim from ``_on_compute_tsne`` (``random_state=42``)."""
    from sklearn.manifold import TSNE

    X, words = map_input(params, inputs)
    if X.shape[0] < 3:
        raise EdaTooFew(X.shape[0])
    perplexity = min(params.get("perplexity", 30), len(X) - 1)
    projected = TSNE(n_components=2, perplexity=perplexity,
                     random_state=42).fit_transform(X)
    log = f"[analysis] t-SNE: perplexity={perplexity}, input={words}"
    return {"projected": np.asarray(projected), "input": words, "log": log}


def run_cluster(params, inputs) -> dict:
    """The grouping, verbatim from ``_on_compute_clusters``, plus the
    seed: Leiden / K-means / GMM take it, HDBSCAN has none."""
    from napari_epicycle.fusion_reconstruction.qc_features import (
        _cluster_gmm, _cluster_hdbscan, _cluster_kmeans, _cluster_leiden,
    )

    algo = params.get("algorithm", "leiden")
    n_components = params.get("n_components", 5)
    input_space = params.get("input_space", "PCA")
    proj = inputs.get("pca_projected")
    if input_space in PCA_INPUTS and proj is not None:
        X = np.asarray(proj)[:, :n_components]
        words = pca_input_words(np.asarray(proj)[:, :n_components],
                                bool(inputs.get("pca_residual")))
    elif input_space == "UMAP":
        X = inputs.get("umap")
        if X is None:
            raise EdaTooFew("no UMAP")   # the launch refuses this first
        words = T.INPUT_UMAP
    else:
        X, _n = feature_matrix(inputs["df"], params)
        if X is None:
            raise EdaTooFew("no features")
        words = T.INPUT_FEATURES
    if len(X) < 2:
        raise EdaTooFew(len(X))
    seed = params.get("seed", 42)
    k = params.get("n_clusters", 5)
    min_size = params.get("min_cluster_size", 15)
    used = algo
    if algo == "leiden":
        try:
            labels = _cluster_leiden(X, params.get("resolution", 1.0),
                                     params.get("n_neighbors", 15),
                                     metric=params.get("metric", "euclidean"),
                                     seed=seed)
        except ImportError:
            labels = _cluster_hdbscan(X, min_size)
            used = "hdbscan"
    elif algo == "hdbscan":
        labels = _cluster_hdbscan(X, min_size)
    elif algo == "kmeans":
        labels = _cluster_kmeans(X, k, seed=seed)
    elif algo == "gmm":
        labels = _cluster_gmm(X, k, seed=seed)
    else:
        labels = np.full(len(X), -1)
    labels = np.asarray(labels)
    noise = int((labels == -1).sum())
    n_groups = int(len(set(labels.tolist()) - {-1}))
    log = (f"[analysis] {used} clustering: {n_groups} clusters, seed="
           f"{seed if used != 'hdbscan' else 'n/a'}, input={words}")
    return {"labels": labels, "algorithm": used, "n_groups": n_groups,
            "noise": noise, "seed": seed, "input": words, "log": log}


_RUNNERS = {"pca": run_pca, "umap": run_umap, "tsne": run_tsne,
            "cluster": run_cluster}


def run_eda(kind: str, params: dict, inputs: dict) -> dict:
    """One map job, pure.  Raises on failure."""
    return _RUNNERS[kind](params, inputs)


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

class EdaWorker(QObject):
    """One Workbench map on its own QThread.

    ``finished(dict)`` carries ``{"kind", "token", **outcome}``;
    ``error(dict)`` carries ``{"kind", "token", "message", "too_few"}``.
    Exactly one of them is emitted, from a ``finally``.
    """

    finished = Signal(object)
    error = Signal(object)

    def __init__(self, kind: str, params: dict, inputs: dict, token):
        super().__init__()
        if kind not in EDA_KINDS:
            raise ValueError(f"unknown map kind {kind!r}")
        self.kind = kind
        self.params = params
        self.inputs = inputs
        self.token = token

    def run(self):
        outcome: Optional[Dict[str, Any]] = None
        failure: Optional[Dict[str, Any]] = None
        try:
            outcome = run_eda(self.kind, self.params, self.inputs)
        except EdaTooFew as exc:
            failure = {"message": f"too few: {exc}", "too_few": True}
        except BaseException as exc:      # noqa: BLE001 -- always answer
            failure = {"message": f"{type(exc).__name__}: {exc}",
                       "too_few": False}
        finally:
            head = {"kind": self.kind, "token": self.token}
            if outcome is not None and failure is None:
                self.finished.emit({**head, **outcome})
            else:
                self.error.emit({**head, **(failure or {
                    "message": "no result", "too_few": False})})

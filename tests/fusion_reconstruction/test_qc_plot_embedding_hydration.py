"""Phase A.5 — embedding-cache hydration in QCPlotWidget.

Exercises the real ``_hydrate_embedding_caches`` method through a
Qt-free harness, mirroring the pattern in ``test_plot_identity_real``.
The helper reads ``UMAP_1``/``UMAP_2`` and ``tSNE_1``/``tSNE_2``
columns out of ``self._df`` and writes them into ``self._umap_cache``
/ ``self._tsne_cache`` so the existing ``_draw_umap``/``_draw_tsne``
rendering paths work without on-demand background computation.
"""

import types

import numpy as np
import pandas as pd


class _HydrateHarness:
    """Mirror of the QCPlotWidget attributes touched by hydration."""

    def __init__(self):
        self._df = None
        self._umap_cache = None
        self._tsne_cache = None


def _bind_hydrate(harness):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )
    real = QCPlotWidget._hydrate_embedding_caches
    harness._hydrate_embedding_caches = types.MethodType(real, harness)


def _make_df(n=12, *, with_umap=False, with_tsne=False, seed=0):
    rng = np.random.RandomState(seed)
    cols = {
        "label": np.arange(1, n + 1),
        "area": rng.rand(n) * 100.0,
        "qc_score": rng.rand(n),
    }
    if with_umap:
        cols["UMAP_1"] = rng.randn(n)
        cols["UMAP_2"] = rng.randn(n)
    if with_tsne:
        cols["tSNE_1"] = rng.randn(n)
        cols["tSNE_2"] = rng.randn(n)
    return pd.DataFrame(cols)


class TestHydrateEmbeddingCaches:
    def test_no_columns_leaves_caches_none(self):
        h = _HydrateHarness()
        _bind_hydrate(h)
        h._df = _make_df(with_umap=False, with_tsne=False)
        h._hydrate_embedding_caches()
        assert h._umap_cache is None
        assert h._tsne_cache is None

    def test_umap_columns_populate_umap_cache(self):
        h = _HydrateHarness()
        _bind_hydrate(h)
        df = _make_df(n=8, with_umap=True)
        h._df = df
        h._hydrate_embedding_caches()
        assert h._umap_cache is not None
        assert h._umap_cache.shape == (8, 2)
        np.testing.assert_allclose(
            h._umap_cache[:, 0], df["UMAP_1"].values,
        )
        np.testing.assert_allclose(
            h._umap_cache[:, 1], df["UMAP_2"].values,
        )
        assert h._tsne_cache is None  # only one set provided

    def test_tsne_columns_populate_tsne_cache(self):
        h = _HydrateHarness()
        _bind_hydrate(h)
        df = _make_df(n=10, with_tsne=True)
        h._df = df
        h._hydrate_embedding_caches()
        assert h._tsne_cache is not None
        assert h._tsne_cache.shape == (10, 2)
        np.testing.assert_allclose(
            h._tsne_cache[:, 0], df["tSNE_1"].values,
        )
        np.testing.assert_allclose(
            h._tsne_cache[:, 1], df["tSNE_2"].values,
        )
        assert h._umap_cache is None

    def test_both_columns_populate_both_caches(self):
        h = _HydrateHarness()
        _bind_hydrate(h)
        df = _make_df(n=6, with_umap=True, with_tsne=True)
        h._df = df
        h._hydrate_embedding_caches()
        assert h._umap_cache is not None
        assert h._tsne_cache is not None
        assert h._umap_cache.shape == (6, 2)
        assert h._tsne_cache.shape == (6, 2)

    def test_partial_umap_columns_does_not_hydrate(self):
        """Both UMAP_1 AND UMAP_2 must be present; one is not enough."""
        h = _HydrateHarness()
        _bind_hydrate(h)
        df = _make_df(n=5)
        df["UMAP_1"] = np.zeros(5)  # only one of the pair
        h._df = df
        h._hydrate_embedding_caches()
        assert h._umap_cache is None

    def test_dtype_is_float64(self):
        """Persisted columns may be float32 from CSV — ensure cast."""
        h = _HydrateHarness()
        _bind_hydrate(h)
        df = _make_df(n=4, with_umap=True)
        df["UMAP_1"] = df["UMAP_1"].astype(np.float32)
        df["UMAP_2"] = df["UMAP_2"].astype(np.float32)
        h._df = df
        h._hydrate_embedding_caches()
        assert h._umap_cache.dtype == np.float64

    def test_none_df_is_noop(self):
        h = _HydrateHarness()
        _bind_hydrate(h)
        h._df = None
        h._hydrate_embedding_caches()
        assert h._umap_cache is None
        assert h._tsne_cache is None

    def test_empty_df_is_noop(self):
        h = _HydrateHarness()
        _bind_hydrate(h)
        h._df = pd.DataFrame(
            columns=["label", "UMAP_1", "UMAP_2", "tSNE_1", "tSNE_2"],
        )
        h._hydrate_embedding_caches()
        assert h._umap_cache is None
        assert h._tsne_cache is None

    def test_hydrate_overwrites_stale_cache(self):
        """When new df has columns, an older cache from a previous df
        must be replaced, not silently kept alongside fresh data."""
        h = _HydrateHarness()
        _bind_hydrate(h)
        h._umap_cache = np.zeros((99, 2))  # stale
        df = _make_df(n=5, with_umap=True, seed=7)
        h._df = df
        h._hydrate_embedding_caches()
        assert h._umap_cache.shape == (5, 2)
        np.testing.assert_allclose(
            h._umap_cache[:, 0], df["UMAP_1"].values,
        )

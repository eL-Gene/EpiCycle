"""Tests for the EDA recompute buttons (2026-05-17).

Verifies `_recompute_pca`, `_recompute_umap`, `_recompute_tsne` and
their shared helpers `_active_mask_and_df` /
`_persist_and_refresh_after_recompute`.

Each recompute method must:
  1. Bail cleanly when df is None / empty / has no active rows.
  2. Filter to ``label_status == "active"`` rows before running the
     compute (ghost rows have NaN in feature columns when n_components
     changes between extracts — same rationale as `_run_clustering`).
  3. Write output columns back to the main df at active row positions
     only; ghost rows retain whatever value (or NaN) they had.
  4. Refresh `ui.qc_plot_widget` with the updated df.
  5. Persist the df to CSV via `save_patch_csv`.

2026-06-06: the recompute ops were WORKER-IFIED — they now build a pure
``_compute`` closure + a main-thread ``_apply`` closure and hand them to
``_launch_async_compute`` (an _AsyncComputeWorker on a QThread).  To keep
these tests synchronous + Qt-free, ``_bind_recompute_methods`` stubs
``_launch_async_compute`` to run the compute closure inline and immediately
invoke the apply closure — exercising the SAME real compute + real
write-back logic the worker path runs, just on the calling thread.  A
``_busy_show`` / ``_busy_hide`` no-op stub is also bound so the methods don't
need real widgets.
"""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Bare-instance controller fixture
# ---------------------------------------------------------------------------


def _make_bare_controller(df=None, working_dir=None, qc_settings=None):
    """Build a Step-3 controller stub with real recompute methods bound."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl._qc_features_df = df
    ctrl.working_dir = str(working_dir) if working_dir is not None else None
    ctrl._append_log = lambda msg: ctrl._log.append(msg)
    ctrl._log = []
    ctrl._qc_settings = qc_settings if qc_settings is not None else {}
    ctrl._selected_curation_patch = lambda: None
    ctrl._selected_patch_name = lambda: None
    ctrl._qc_view_patch_name = "test_patch.tif"

    plot_widget = MagicMock()
    ctrl.ui = types.SimpleNamespace(qc_plot_widget=plot_widget)
    ctrl._mock_plot_widget = plot_widget
    # Busy-bar helpers are no-ops in these Qt-free tests.
    ctrl._busy_show = lambda *a, **k: None
    ctrl._busy_hide = lambda *a, **k: None
    return ctrl


def _sync_launch_async_compute(ctrl):
    """Return a synchronous stand-in for ``_launch_async_compute`` that runs
    the compute closure inline + immediately applies the result — exercising
    the real compute + write-back logic without a QThread / Qt event loop."""
    def _stub(op_tag, compute_fn, on_done):
        result = compute_fn()
        on_done(result)
    return _stub


def _bind_recompute_methods(ctrl):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    for name in (
        "_recompute_pca",
        "_recompute_umap",
        "_recompute_tsne",
        "_active_mask_and_df",
        "_persist_and_refresh_after_recompute",
    ):
        real = getattr(FusionReconstructionController, name)
        ctrl.__dict__[name] = types.MethodType(real, ctrl)
    # Worker-ification (2026-06-06): drive the launch synchronously so the
    # compute + apply closures run inline on the calling thread.
    ctrl.__dict__["_launch_async_compute"] = _sync_launch_async_compute(ctrl)


def _make_df(n=10, n_ghosts=2, n_features=4):
    """Realistic df: ``n`` active + ``n_ghosts`` ghost rows, ``n_features``
    feature columns + label / label_status / area."""
    rng = np.random.default_rng(seed=42)
    total = n + n_ghosts
    df = pd.DataFrame({
        "label": list(range(1, total + 1)),
        "label_status": ["active"] * n + ["deleted"] * n_ghosts,
        "area": rng.uniform(100, 1000, total),
    })
    for i in range(n_features):
        df[f"feat_{i}"] = rng.normal(size=total)
    # Ghost rows get NaN in features (simulates an earlier extract with a
    # different feature set, which is the real-world failure mode).
    for i in range(n_features):
        df.loc[df["label_status"] != "active", f"feat_{i}"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Worker-ification (2026-06-06): each recompute op references
# _launch_async_compute + the _AsyncComputeWorker path; the busy bar marquee
# animates off the main thread.  Source-introspection (no Qt).
# ---------------------------------------------------------------------------


class TestRecomputeWorkerified:

    def test_each_op_launches_async_compute(self):
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        for name in ("_recompute_pca", "_recompute_umap", "_recompute_tsne",
                     "_run_clustering"):
            src = inspect.getsource(getattr(FusionReconstructionController, name))
            assert "_launch_async_compute(" in src, (
                f"{name} must launch via _launch_async_compute (worker-ified)"
            )

    def test_launch_helper_has_done_and_error_handlers(self):
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._launch_async_compute
        )
        assert "_AsyncComputeWorker(" in src
        assert "done.connect" in src and "error.connect" in src
        assert "_busy_hide()" in src
        assert "finally" in src


# ---------------------------------------------------------------------------
# _active_mask_and_df helper
# ---------------------------------------------------------------------------


class TestActiveMaskAndDfHelper:
    """The gate that every recompute method shares."""

    def test_none_df_returns_none(self, tmp_path):
        ctrl = _make_bare_controller(df=None, working_dir=tmp_path)
        _bind_recompute_methods(ctrl)
        assert ctrl._active_mask_and_df("TEST-RC") is None

    def test_empty_df_returns_none(self, tmp_path):
        ctrl = _make_bare_controller(df=pd.DataFrame(), working_dir=tmp_path)
        _bind_recompute_methods(ctrl)
        assert ctrl._active_mask_and_df("TEST-RC") is None

    def test_no_active_rows_returns_none(self, tmp_path):
        df = pd.DataFrame({
            "label": [1, 2, 3],
            "label_status": ["deleted", "deleted", "superseded"],
            "area": [100, 200, 300],
        })
        ctrl = _make_bare_controller(df=df, working_dir=tmp_path)
        _bind_recompute_methods(ctrl)
        assert ctrl._active_mask_and_df("TEST-RC") is None

    def test_missing_label_status_column_treats_all_active(self, tmp_path):
        df = pd.DataFrame({"label": [1, 2, 3], "area": [100, 200, 300]})
        ctrl = _make_bare_controller(df=df, working_dir=tmp_path)
        _bind_recompute_methods(ctrl)
        result = ctrl._active_mask_and_df("TEST-RC")
        assert result is not None
        df_full, mask, n_active, n_ghost = result
        assert n_active == 3
        assert n_ghost == 0

    def test_returns_mask_filtering_to_active(self, tmp_path):
        df = _make_df(n=5, n_ghosts=3, n_features=2)
        ctrl = _make_bare_controller(df=df, working_dir=tmp_path)
        _bind_recompute_methods(ctrl)
        result = ctrl._active_mask_and_df("TEST-RC")
        assert result is not None
        df_full, mask, n_active, n_ghost = result
        assert n_active == 5
        assert n_ghost == 3
        assert mask.sum() == 5
        assert (df_full.loc[mask, "label_status"] == "active").all()


# ---------------------------------------------------------------------------
# _recompute_pca
# ---------------------------------------------------------------------------


class TestRecomputePCA:

    def test_pca_writes_pc_columns_at_active_rows_only(self, tmp_path):
        df = _make_df(n=20, n_ghosts=3, n_features=4)
        qc_settings = {
            "n_cluster_components": 3,
            "pca_features": ["feat_0", "feat_1", "feat_2", "feat_3"],
            "exclude_border": False,
        }
        ctrl = _make_bare_controller(
            df=df, working_dir=tmp_path, qc_settings=qc_settings
        )
        _bind_recompute_methods(ctrl)
        ctrl._recompute_pca()

        # PC columns exist
        for i in range(1, 4):
            assert f"PC{i}" in ctrl._qc_features_df.columns

        # Active rows have non-null PC values; ghost rows retain NaN
        active = ctrl._qc_features_df["label_status"] == "active"
        for i in range(1, 4):
            col = f"PC{i}"
            assert ctrl._qc_features_df.loc[active, col].notna().all()
            assert ctrl._qc_features_df.loc[~active, col].isna().all()

    def test_pca_persists_csv_to_disk(self, tmp_path):
        df = _make_df(n=10, n_ghosts=2, n_features=3)
        qc_settings = {
            "n_cluster_components": 2,
            "pca_features": ["feat_0", "feat_1", "feat_2"],
            "exclude_border": False,
        }
        ctrl = _make_bare_controller(
            df=df, working_dir=tmp_path, qc_settings=qc_settings
        )
        _bind_recompute_methods(ctrl)
        ctrl._recompute_pca()

        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "test_patch_features.csv"
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert "PC1" in loaded.columns
        assert "PC2" in loaded.columns

    def test_pca_refreshes_plot_widget(self, tmp_path):
        df = _make_df(n=10, n_ghosts=2, n_features=3)
        qc_settings = {
            "n_cluster_components": 2,
            "pca_features": ["feat_0", "feat_1", "feat_2"],
        }
        ctrl = _make_bare_controller(
            df=df, working_dir=tmp_path, qc_settings=qc_settings
        )
        _bind_recompute_methods(ctrl)
        ctrl._recompute_pca()
        ctrl._mock_plot_widget.set_features.assert_called_once()

    def test_pca_skips_when_df_none(self, tmp_path):
        ctrl = _make_bare_controller(df=None, working_dir=tmp_path)
        _bind_recompute_methods(ctrl)
        ctrl._recompute_pca()
        ctrl._mock_plot_widget.set_features.assert_not_called()

    def test_pca_ghost_rows_unchanged(self, tmp_path):
        """Ghost rows that already have stale PC columns are left alone."""
        df = _make_df(n=10, n_ghosts=2, n_features=3)
        df["PC1"] = 99.0  # Pretend an earlier run left stale values.
        df["PC2"] = -99.0
        qc_settings = {
            "n_cluster_components": 2,
            "pca_features": ["feat_0", "feat_1", "feat_2"],
        }
        ctrl = _make_bare_controller(
            df=df, working_dir=tmp_path, qc_settings=qc_settings
        )
        _bind_recompute_methods(ctrl)
        ctrl._recompute_pca()

        ghost = ctrl._qc_features_df["label_status"] != "active"
        assert (ctrl._qc_features_df.loc[ghost, "PC1"] == 99.0).all()
        assert (ctrl._qc_features_df.loc[ghost, "PC2"] == -99.0).all()


# ---------------------------------------------------------------------------
# _recompute_umap
# ---------------------------------------------------------------------------


class TestRecomputeUMAP:

    def test_umap_writes_columns_at_active_rows(self, tmp_path):
        umap = pytest.importorskip("umap")
        df = _make_df(n=30, n_ghosts=3, n_features=4)
        qc_settings = {
            "n_cluster_components": 3,
            "pca_features": ["feat_0", "feat_1", "feat_2", "feat_3"],
            "leiden_n_neighbors": 5,
            "umap_min_dist": 0.1,
            "metric": "euclidean",
        }
        ctrl = _make_bare_controller(
            df=df, working_dir=tmp_path, qc_settings=qc_settings
        )
        _bind_recompute_methods(ctrl)
        ctrl._recompute_umap()

        out = ctrl._qc_features_df
        assert "UMAP_1" in out.columns
        assert "UMAP_2" in out.columns
        active = out["label_status"] == "active"
        assert out.loc[active, "UMAP_1"].notna().all()
        assert out.loc[~active, "UMAP_1"].isna().all()

    def test_umap_skips_when_df_none(self, tmp_path):
        ctrl = _make_bare_controller(df=None, working_dir=tmp_path)
        _bind_recompute_methods(ctrl)
        ctrl._recompute_umap()
        ctrl._mock_plot_widget.set_features.assert_not_called()


# ---------------------------------------------------------------------------
# _recompute_tsne
# ---------------------------------------------------------------------------


class TestRecomputeTSNE:

    def test_tsne_writes_columns_at_active_rows(self, tmp_path):
        df = _make_df(n=30, n_ghosts=3, n_features=4)
        qc_settings = {
            "n_cluster_components": 3,
            "pca_features": ["feat_0", "feat_1", "feat_2", "feat_3"],
            "tsne_perplexity": 5.0,  # perplexity must be < n_samples
            "metric": "euclidean",
        }
        ctrl = _make_bare_controller(
            df=df, working_dir=tmp_path, qc_settings=qc_settings
        )
        _bind_recompute_methods(ctrl)
        ctrl._recompute_tsne()

        out = ctrl._qc_features_df
        assert "tSNE_1" in out.columns
        assert "tSNE_2" in out.columns
        active = out["label_status"] == "active"
        assert out.loc[active, "tSNE_1"].notna().all()
        assert out.loc[~active, "tSNE_1"].isna().all()

    def test_tsne_skips_when_df_none(self, tmp_path):
        ctrl = _make_bare_controller(df=None, working_dir=tmp_path)
        _bind_recompute_methods(ctrl)
        ctrl._recompute_tsne()
        ctrl._mock_plot_widget.set_features.assert_not_called()

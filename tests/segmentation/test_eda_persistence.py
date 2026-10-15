"""Tests for EDA state persistence (PCA, UMAP, t-SNE cache save/load)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction.qc_features import (
    _label_hash,
    load_eda_state,
    save_eda_state,
)


def _make_df(n=50, seed=42):
    """Create a minimal features DataFrame."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "label": np.arange(1, n + 1),
        "area": rng.rand(n) * 1000,
        "sphericity": rng.rand(n),
        "qc_score": rng.rand(n),
    })


def _make_pca_result(n=50, n_components=5, n_features=3):
    rng = np.random.RandomState(0)
    return {
        "projected": rng.randn(n, n_components),
        "loadings": rng.randn(n_components, n_features),
        "explained_variance_ratio": np.sort(rng.rand(n_components))[::-1],
        "feature_names": [f"f{i}" for i in range(n_features)],
        "regress_cols_used": None,
    }


# ---------------------------------------------------------------------------
# Label hash
# ---------------------------------------------------------------------------

class TestLabelHash:
    def test_deterministic(self):
        df = _make_df()
        assert _label_hash(df) == _label_hash(df)

    def test_order_independent(self):
        df1 = _make_df()
        df2 = df1.sample(frac=1, random_state=99).reset_index(drop=True)
        assert _label_hash(df1) == _label_hash(df2)

    def test_different_labels(self):
        df1 = _make_df(n=10)
        df2 = _make_df(n=10)
        df2["label"] = np.arange(100, 110)
        assert _label_hash(df1) != _label_hash(df2)


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------

class TestEdaRoundtrip:
    def test_pca_only(self, tmp_path):
        df = _make_df()
        pca = _make_pca_result(n=len(df))

        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)
        result = load_eda_state(tmp_path, "s1", df=df)

        assert result is not None
        np.testing.assert_array_almost_equal(
            result["pca_result"]["projected"], pca["projected"]
        )
        np.testing.assert_array_almost_equal(
            result["pca_result"]["loadings"], pca["loadings"]
        )
        np.testing.assert_array_almost_equal(
            result["pca_result"]["explained_variance_ratio"],
            pca["explained_variance_ratio"],
        )
        assert result["pca_result"]["feature_names"] == pca["feature_names"]
        assert result["umap_cache"] is None
        assert result["tsne_cache"] is None

    def test_full_state(self, tmp_path):
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        umap = np.random.randn(len(df), 2)
        tsne = np.random.randn(len(df), 2)

        save_eda_state(
            tmp_path, "s1",
            pca_result=pca, umap_cache=umap, tsne_cache=tsne, df=df,
        )
        result = load_eda_state(tmp_path, "s1", df=df)

        assert result is not None
        np.testing.assert_array_almost_equal(
            result["umap_cache"], umap
        )
        np.testing.assert_array_almost_equal(
            result["tsne_cache"], tsne
        )

    def test_residual_pca(self, tmp_path):
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        pca["regress_cols_used"] = ["area", "sphericity"]

        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)
        result = load_eda_state(tmp_path, "s1", df=df)

        assert result is not None
        assert result["meta"]["pca_method"] == "residual"
        assert result["pca_result"]["regress_cols_used"] == ["area", "sphericity"]

    def test_params_roundtrip(self, tmp_path):
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        params = {"umap_neighbors": 15, "tsne_perplexity": 30}

        save_eda_state(
            tmp_path, "s1", pca_result=pca, df=df, params=params,
        )
        result = load_eda_state(tmp_path, "s1", df=df)

        assert result["params"] == params


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_row_count_mismatch(self, tmp_path):
        df = _make_df(n=50)
        pca = _make_pca_result(n=50)
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)

        # Load with different row count
        df2 = _make_df(n=40)
        assert load_eda_state(tmp_path, "s1", df=df2) is None

    def test_label_change(self, tmp_path):
        df = _make_df(n=50)
        pca = _make_pca_result(n=50)
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)

        # Same row count but different labels
        df2 = df.copy()
        df2.loc[0, "label"] = 9999
        assert load_eda_state(tmp_path, "s1", df=df2) is None

    def test_no_df_skips_staleness(self, tmp_path):
        """When df is None, skip staleness check (load always succeeds)."""
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)

        result = load_eda_state(tmp_path, "s1", df=None)
        assert result is not None


# ---------------------------------------------------------------------------
# Missing / corrupt files
# ---------------------------------------------------------------------------

class TestEdgeCase:
    def test_missing_files(self, tmp_path):
        assert load_eda_state(tmp_path, "nonexistent") is None

    def test_missing_npz_only(self, tmp_path):
        json_path = tmp_path / "s1_eda.json"
        json_path.write_text("{}")
        assert load_eda_state(tmp_path, "s1") is None

    def test_corrupt_npz(self, tmp_path):
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)

        # Corrupt the NPZ
        npz_path = tmp_path / "s1_eda.npz"
        npz_path.write_bytes(b"garbage data")
        assert load_eda_state(tmp_path, "s1", df=df) is None

    def test_corrupt_json(self, tmp_path):
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)

        # Corrupt the JSON
        json_path = tmp_path / "s1_eda.json"
        json_path.write_text("not valid json {{{")
        assert load_eda_state(tmp_path, "s1", df=df) is None

    def test_schema_version_mismatch(self, tmp_path):
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)

        # Change schema version
        json_path = tmp_path / "s1_eda.json"
        meta = json.loads(json_path.read_text())
        meta["schema_version"] = 999
        json_path.write_text(json.dumps(meta))
        assert load_eda_state(tmp_path, "s1", df=df) is None

    def test_nothing_to_save(self, tmp_path):
        """save_eda_state with no arrays is a no-op."""
        save_eda_state(tmp_path, "s1")
        assert not (tmp_path / "s1_eda.npz").exists()

    def test_overwrite(self, tmp_path):
        """Second save overwrites first."""
        df = _make_df()
        pca1 = _make_pca_result(n=len(df))
        save_eda_state(tmp_path, "s1", pca_result=pca1, df=df)

        pca2 = _make_pca_result(n=len(df), n_components=3)
        save_eda_state(tmp_path, "s1", pca_result=pca2, df=df)

        result = load_eda_state(tmp_path, "s1", df=df)
        assert result["pca_result"]["projected"].shape[1] == 3


# ---------------------------------------------------------------------------
# Cluster ID in CSV roundtrip
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# UI state persistence (schema v3)
# ---------------------------------------------------------------------------

class TestUIStatePersistence:
    """Verify ui_state roundtrip in the JSON sidecar."""

    def test_ui_state_roundtrip(self, tmp_path):
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        ui_state = {
            "selected_features": ["area", "sphericity"],
            "transforms": {"area": "log", "sphericity": "raw", "qc_score": "auto"},
            "preset_name": "Custom (based on EpiCycle QC)",
            "pca_mode": "qc",
        }
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df, ui_state=ui_state)
        result = load_eda_state(tmp_path, "s1", df=df)

        assert result is not None
        assert result["ui_state"] == ui_state

    def test_v2_loads_without_ui_state(self, tmp_path):
        """v2 files load fine, ui_state is None."""
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)

        # Manually downgrade to v2
        json_path = tmp_path / "s1_eda.json"
        meta = json.loads(json_path.read_text())
        meta["schema_version"] = 2
        meta.pop("ui_state", None)
        json_path.write_text(json.dumps(meta))

        result = load_eda_state(tmp_path, "s1", df=df)
        assert result is not None
        assert result["ui_state"] is None
        assert result["pca_result"] is not None

    def test_v3_schema_version(self, tmp_path):
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)

        json_path = tmp_path / "s1_eda.json"
        meta = json.loads(json_path.read_text())
        assert meta["schema_version"] == 3

    def test_no_ui_state_gives_none(self, tmp_path):
        """When ui_state not passed, result has ui_state=None."""
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)

        result = load_eda_state(tmp_path, "s1", df=df)
        assert result is not None
        assert result["ui_state"] is None

    def test_partial_ui_state(self, tmp_path):
        """ui_state with only some keys is fine."""
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        ui_state = {"selected_features": ["area"]}
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df, ui_state=ui_state)

        result = load_eda_state(tmp_path, "s1", df=df)
        assert result["ui_state"]["selected_features"] == ["area"]
        assert result["ui_state"].get("transforms") is None

    def test_schema_999_still_rejected(self, tmp_path):
        """Unknown schema versions are still rejected."""
        df = _make_df()
        pca = _make_pca_result(n=len(df))
        save_eda_state(tmp_path, "s1", pca_result=pca, df=df)

        json_path = tmp_path / "s1_eda.json"
        meta = json.loads(json_path.read_text())
        meta["schema_version"] = 999
        json_path.write_text(json.dumps(meta))
        assert load_eda_state(tmp_path, "s1", df=df) is None


class TestClusterPersistence:
    def test_cluster_id_survives_csv(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            load_patch_csv,
            save_patch_csv,
        )

        df = _make_df()
        df["cluster_id"] = np.random.randint(0, 5, len(df))
        csv_path = tmp_path / "features.csv"

        save_patch_csv(df, csv_path)
        loaded = load_patch_csv(csv_path)

        assert "cluster_id" in loaded.columns
        np.testing.assert_array_equal(
            loaded["cluster_id"].values, df["cluster_id"].values
        )

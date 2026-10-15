"""Tests for QC feature extraction module."""

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction.qc_features import (
    BOUNDED_FEATURES,
    FEATURE_COLUMNS,
    LOG_TRANSFORM_FEATURES,
    NUMERIC_FEATURES,
    PCA_FEATURES,
    _detect_skewed_columns,
    _robust_standardize,
    apply_threshold,
    compute_boundary_interior_features,
    compute_composite_score,
    compute_neighbor_features,
    compute_pca,
    compute_probability_features,
    compute_skeleton_features,
    compute_surface_features,
    extract_features,
    extract_single_label_features,
    load_patch_csv,
    merge_patch_csvs,
    save_patch_csv,
    spacing_label,
)


def _make_labels(n_labels=5, shape=(32, 48, 48)):
    """Create a small 3D label volume with n_labels distinct cubic regions."""
    labels = np.zeros(shape, dtype=np.int32)
    z, y, x = shape
    # Place non-overlapping cubes with extent in all 3 dimensions
    cube_size = min(z // 3, y // n_labels - 2, x // 3)
    cube_size = max(cube_size, 4)
    step_y = y // n_labels
    for i in range(n_labels):
        label_id = i + 1
        y0 = i * step_y + 1
        y1 = min(y0 + cube_size, y - 1)
        z0, z1 = 2, min(2 + cube_size, z - 1)
        x0, x1 = 2, min(2 + cube_size, x - 1)
        labels[z0:z1, y0:y1, x0:x1] = label_id
    return labels


class TestExtractFeatures:
    def test_basic_extraction(self):
        labels = _make_labels(n_labels=3)
        df = extract_features(labels)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "label" in df.columns
        assert "area" in df.columns
        assert "sphericity" in df.columns
        assert "aspect_ratio" in df.columns
        assert "bbox_volume" in df.columns
        assert "compactness" in df.columns

    def test_empty_labels(self):
        labels = np.zeros((8, 8, 8), dtype=np.int32)
        df = extract_features(labels)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert set(FEATURE_COLUMNS).issubset(set(df.columns))

    def test_single_label(self):
        labels = np.zeros((8, 8, 8), dtype=np.int32)
        labels[1:7, 1:7, 1:7] = 1
        df = extract_features(labels)
        assert len(df) == 1
        assert df.iloc[0]["label"] == 1
        assert df.iloc[0]["area"] > 0
        assert df.iloc[0]["compactness"] > 0

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="3D"):
            extract_features(np.zeros((10, 10), dtype=np.int32))

    def test_derived_features_are_finite(self):
        labels = _make_labels(n_labels=5)
        df = extract_features(labels)
        for col in ("sphericity", "aspect_ratio", "compactness", "bbox_volume"):
            assert np.all(np.isfinite(df[col].values))


class TestApplyThreshold:
    def test_min_max_threshold(self):
        labels = _make_labels(n_labels=5)
        df = extract_features(labels)
        apply_threshold(df, "area", min_val=10, max_val=10000)
        assert "qc_status" in df.columns
        assert set(df["qc_status"].unique()).issubset({"pass", "flagged"})

    def test_no_lower_bound(self):
        df = pd.DataFrame({"area": [1, 2, 3, 4, 5], "label": range(5)})
        apply_threshold(df, "area", max_val=3)
        assert list(df["qc_status"]) == ["pass", "pass", "pass", "flagged", "flagged"]

    def test_no_upper_bound(self):
        df = pd.DataFrame({"area": [1, 2, 3, 4, 5], "label": range(5)})
        apply_threshold(df, "area", min_val=3)
        assert list(df["qc_status"]) == ["flagged", "flagged", "pass", "pass", "pass"]

    def test_invalid_feature(self):
        df = pd.DataFrame({"area": [1, 2]})
        with pytest.raises(ValueError, match="not in DataFrame"):
            apply_threshold(df, "nonexistent")


class TestCSVIO:
    def test_save_load_roundtrip(self, tmp_path):
        labels = _make_labels(n_labels=3)
        df = extract_features(labels)
        csv_path = tmp_path / "test_features.csv"
        save_patch_csv(df, csv_path)
        loaded = load_patch_csv(csv_path)
        assert len(loaded) == len(df)
        # load_patch_csv adds backward-compat columns (feature_state,
        # label_status) if missing — verify all original columns present.
        for col in df.columns:
            assert col in loaded.columns

    def test_merge_patch_csvs(self, tmp_path):
        for name in ("patch001", "patch002"):
            labels = _make_labels(n_labels=3)
            df = extract_features(labels)
            save_patch_csv(df, tmp_path / f"{name}_features.csv")

        merged = merge_patch_csvs([
            tmp_path / "patch001_features.csv",
            tmp_path / "patch002_features.csv",
        ])
        assert len(merged) == 6
        assert "patch_name" in merged.columns
        assert set(merged["patch_name"].unique()) == {"patch001", "patch002"}


class TestComputePCA:
    def test_basic_pca(self):
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        projected = compute_pca(df, n_components=2)
        assert projected.shape == (10, 2)
        assert np.all(np.isfinite(projected))

    def test_few_samples(self):
        labels = _make_labels(n_labels=1)
        df = extract_features(labels)
        projected = compute_pca(df, n_components=2)
        assert projected.shape == (1, 2)

    def test_empty_df(self):
        df = pd.DataFrame(columns=FEATURE_COLUMNS)
        projected = compute_pca(df, n_components=2)
        assert projected.shape == (0, 2)


class TestComputeResidualPCA:
    """Phase 5: Residual PCA — regress out size, reveal shape variation."""

    def test_basic_residual_pca(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_residual_pca,
        )
        labels = _make_labels(n_labels=15)
        df = extract_features(labels)
        result = compute_residual_pca(df, n_components=2)
        assert result["projected"].shape == (15, 2)
        assert np.all(np.isfinite(result["projected"]))
        assert result["explained_variance_ratio"].shape == (2,)
        assert all(0 <= v <= 1 for v in result["explained_variance_ratio"])
        assert len(result["regress_cols_used"]) > 0
        assert "loadings" in result
        assert "feature_names" in result

    def test_returns_explained_variance(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_residual_pca,
        )
        labels = _make_labels(n_labels=20)
        df = extract_features(labels)
        result = compute_residual_pca(df, n_components=2)
        var_ratio = result["explained_variance_ratio"]
        assert var_ratio[0] >= var_ratio[1]

    def test_regress_out_reduces_size_correlation(self):
        """After regressing out size, PC1 should correlate less with area."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_pca,
            compute_residual_pca,
        )
        from scipy.stats import spearmanr

        rng = np.random.default_rng(42)
        n = 100
        size = rng.exponential(100, n)
        df = pd.DataFrame({
            "area": size,
            "equivalent_diameter": size ** (1 / 3) + rng.normal(0, 0.5, n),
            "surface_area": size ** (2 / 3) + rng.normal(0, 1, n),
            "bbox_volume": size * 1.5 + rng.normal(0, 5, n),
            "axis_major_length": size ** (1 / 3) * 2 + rng.normal(0, 0.3, n),
            "axis_minor_length": size ** (1 / 3) + rng.normal(0, 0.2, n),
            "aspect_ratio": rng.normal(1.5, 0.3, n),
            "compactness": rng.normal(0.7, 0.1, n),
            "true_sphericity": rng.normal(0.8, 0.05, n),
            "solidity": rng.normal(0.9, 0.05, n),
            "extent": rng.normal(0.6, 0.1, n),
            "inertia_elongation": rng.normal(2, 0.5, n),
            "major_axis_z_angle": rng.normal(45, 15, n),
            "intensity_mean": rng.normal(100, 20, n),
            "boundary_interior_ratio": rng.normal(1.2, 0.3, n),
            "neighbor_count": rng.poisson(5, n).astype(float),
            "nearest_neighbor_distance": rng.exponential(3, n),
            "skeleton_branches": rng.poisson(2, n).astype(float),
            "prob_mean": rng.beta(5, 2, n),
            "prob_boundary_drop": rng.normal(0.3, 0.1, n),
            "prob_boundary_entropy": rng.beta(2, 5, n),
        })

        std_proj = compute_pca(df, n_components=2, mode="eda")
        res_result = compute_residual_pca(df, n_components=2)
        res_proj = res_result["projected"]

        area = df["area"].values
        std_corr = abs(spearmanr(area, std_proj[:, 0]).statistic)
        res_corr = abs(spearmanr(area, res_proj[:, 0]).statistic)
        assert res_corr < std_corr

    def test_custom_regress_out(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_residual_pca,
        )
        labels = _make_labels(n_labels=15)
        df = extract_features(labels)
        result = compute_residual_pca(df, n_components=2, regress_out=["area"])
        assert result["projected"].shape == (15, 2)
        assert result["regress_cols_used"] == ["area"]

    def test_no_regress_features_available(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_residual_pca,
        )
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        result = compute_residual_pca(
            df, n_components=2, regress_out=["nonexistent_col"],
        )
        assert result["projected"].shape == (10, 2)
        assert result["regress_cols_used"] == []

    def test_few_samples(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_residual_pca,
        )
        labels = _make_labels(n_labels=2)
        df = extract_features(labels)
        result = compute_residual_pca(df, n_components=2)
        assert result["projected"].shape == (2, 2)

    def test_border_cell_exclusion(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_residual_pca,
        )
        rng = np.random.default_rng(42)
        n = 30
        size = rng.exponential(100, n)
        df = pd.DataFrame({
            "area": size,
            "equivalent_diameter": size ** (1/3) + rng.normal(0, 0.5, n),
            "surface_area": size ** (2/3) + rng.normal(0, 1, n),
            "bbox_volume": size * 1.5 + rng.normal(0, 5, n),
            "axis_major_length": size ** (1/3) * 2 + rng.normal(0, 0.3, n),
            "axis_minor_length": size ** (1/3) + rng.normal(0, 0.2, n),
            "aspect_ratio": rng.normal(1.5, 0.3, n),
            "compactness": rng.normal(0.7, 0.1, n),
            "true_sphericity": rng.normal(0.8, 0.05, n),
            "solidity": rng.normal(0.9, 0.05, n),
            "inertia_elongation": rng.normal(2, 0.5, n),
            "is_border_cell": [True] * 10 + [False] * 20,
        })
        df.loc[:9, "area"] *= 3

        r_excl = compute_residual_pca(df, n_components=2, exclude_border=True)
        r_incl = compute_residual_pca(df, n_components=2, exclude_border=False)
        assert r_excl["projected"].shape == (n, 2)
        assert r_incl["projected"].shape == (n, 2)
        assert not np.allclose(r_excl["projected"], r_incl["projected"])

    def test_pca_full_returns_loadings(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_pca_full,
        )
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({
            "area": rng.exponential(100, n),
            "aspect_ratio": rng.normal(1.5, 0.3, n),
            "compactness": rng.normal(0.7, 0.1, n),
            "true_sphericity": rng.normal(0.8, 0.05, n),
            "solidity": rng.normal(0.9, 0.05, n),
        })
        result = compute_pca_full(df, n_components=3, features=list(df.columns))
        assert result["projected"].shape == (n, 3)
        assert result["loadings"].shape == (3, 5)
        assert result["explained_variance_ratio"].shape == (3,)
        assert sum(result["explained_variance_ratio"]) <= 1.0 + 1e-9
        assert len(result["feature_names"]) == 5


class TestNonlinearEmbeddings:
    """Phase A.4: ``compute_umap_embedding`` / ``compute_tsne_embedding``.

    Both writers run the underlying transform and persist 2D
    coordinates as new DataFrame columns so they ride into the
    per-patch CSV and can be consumed by the scatter plot without
    on-demand recomputation.
    """

    def _toy_df(self, n=20, seed=0):
        rng = np.random.RandomState(seed)
        return pd.DataFrame(
            {f: rng.randn(n) for f in PCA_FEATURES[:6]},
        )

    def test_umap_embedding_writes_columns(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_umap_embedding,
        )

        df = self._toy_df()
        out = compute_umap_embedding(df, n_neighbors=8, min_dist=0.1)
        assert out is df  # in-place
        assert "UMAP_1" in df.columns
        assert "UMAP_2" in df.columns
        assert df["UMAP_1"].notna().all()
        assert df["UMAP_2"].notna().all()
        assert len(df) == 20

    def test_tsne_embedding_writes_columns(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_tsne_embedding,
        )

        df = self._toy_df()
        out = compute_tsne_embedding(df, perplexity=5)
        assert out is df
        assert "tSNE_1" in df.columns
        assert "tSNE_2" in df.columns
        assert df["tSNE_1"].notna().all()
        assert df["tSNE_2"].notna().all()

    def test_pca_dims_pre_reduction_keeps_shape(self):
        """pca_dims pre-reduces to that many PCs before embedding,
        but the output shape must still be (n_samples, 2)."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_umap_embedding, compute_tsne_embedding,
        )

        df = self._toy_df(n=15)
        compute_umap_embedding(df, n_neighbors=5, pca_dims=3)
        compute_tsne_embedding(df, perplexity=5, pca_dims=3)
        assert df[["UMAP_1", "UMAP_2"]].shape == (15, 2)
        assert df[["tSNE_1", "tSNE_2"]].shape == (15, 2)

    def test_tiny_frame_returns_zeros_no_crash(self):
        """n=1 < 2 should fall back gracefully (zeros, no exception)."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_umap_embedding, compute_tsne_embedding,
        )

        df = pd.DataFrame({f: [1.0] for f in PCA_FEATURES[:4]})
        compute_umap_embedding(df, n_neighbors=2)
        compute_tsne_embedding(df, perplexity=2)
        assert df[["UMAP_1", "UMAP_2"]].shape == (1, 2)
        assert df[["tSNE_1", "tSNE_2"]].shape == (1, 2)

    def test_columns_persist_through_csv_roundtrip(self, tmp_path):
        """UMAP_*/tSNE_* columns must round-trip via save/load_patch_csv."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_umap_embedding, compute_tsne_embedding,
        )

        df = self._toy_df(n=12)
        compute_umap_embedding(df, n_neighbors=5)
        compute_tsne_embedding(df, perplexity=4)

        path = tmp_path / "patch.csv"
        save_patch_csv(df, path)
        loaded = load_patch_csv(path)
        assert {"UMAP_1", "UMAP_2", "tSNE_1", "tSNE_2"} <= set(loaded.columns)
        np.testing.assert_allclose(
            loaded["UMAP_1"].values, df["UMAP_1"].values, rtol=1e-6,
        )
        np.testing.assert_allclose(
            loaded["tSNE_1"].values, df["tSNE_1"].values, rtol=1e-6,
        )


class TestMorphologyFeatures:
    """CP1: Tests for ellipsoid solidity, z-span, elongation, fill ratio, orientation."""

    def test_ellipsoid_solidity_present_and_reasonable(self):
        labels = _make_labels(n_labels=5)
        df = extract_features(labels)
        assert "ellipsoid_solidity" in df.columns
        # Cubes should have finite, positive ellipsoid solidity
        vals = df["ellipsoid_solidity"].values
        assert np.all(np.isfinite(vals))
        assert np.all(vals > 0)

    def test_z_span_matches_bbox(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[3:13, 2:8, 2:8] = 1  # z_span = 10
        labels[1:5, 12:18, 12:18] = 2  # z_span = 4
        df = extract_features(labels)
        row1 = df[df["label"] == 1].iloc[0]
        row2 = df[df["label"] == 2].iloc[0]
        assert row1["z_span"] == 10.0
        assert row2["z_span"] == 4.0

    def test_z_elongation_correct(self):
        labels = np.zeros((30, 10, 10), dtype=np.int32)
        labels[2:28, 2:8, 2:8] = 1  # tall along Z: z=26, y=6, x=6
        df = extract_features(labels)
        row = df.iloc[0]
        # z_elongation = z_span / max(y_span, x_span) = 26/6 ~ 4.33
        assert row["z_elongation"] > 3.0

    def test_xy_elongation_correct(self):
        labels = np.zeros((10, 40, 10), dtype=np.int32)
        labels[2:8, 2:38, 2:8] = 1  # wide along Y: y=36, x=6
        df = extract_features(labels)
        row = df.iloc[0]
        # xy_elongation = max(y,x) / min(y,x) = 36/6 = 6
        assert row["xy_elongation"] > 5.0

    def test_fill_ratio_solid_equals_one(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[1:9, 1:9, 1:9] = 1  # solid cube, no holes
        df = extract_features(labels)
        np.testing.assert_allclose(df.iloc[0]["fill_ratio"], 1.0, atol=1e-6)

    def test_major_axis_z_angle_in_range(self):
        labels = _make_labels(n_labels=5)
        df = extract_features(labels)
        assert "major_axis_z_angle" in df.columns
        vals = df["major_axis_z_angle"].values
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 90.0)

    def test_inertia_elongation_and_flatness_valid(self):
        labels = _make_labels(n_labels=5)
        df = extract_features(labels)
        for col in ("inertia_elongation", "inertia_flatness"):
            assert col in df.columns
            vals = df[col].values
            assert np.all(np.isfinite(vals))
            assert np.all(vals >= 0.0)

    def test_eigenvalue_and_tensor_columns_dropped(self):
        labels = _make_labels(n_labels=3)
        df = extract_features(labels)
        for col in df.columns:
            assert not col.startswith("inertia_tensor_eigvals-"), \
                f"Intermediate column {col} should be dropped"
            assert not col.startswith("inertia_tensor-"), \
                f"Intermediate column {col} should be dropped"


class TestSurfaceFeatures:
    """CP1b: Tests for surface area, true sphericity, roughness, fast solidity."""

    def test_cube_roughness_greater_than_one(self):
        labels = np.zeros((16, 16, 16), dtype=np.int32)
        labels[2:14, 2:14, 2:14] = 1  # cube — rougher than a sphere
        df = extract_features(labels)
        compute_surface_features(labels, df)
        assert "roughness" in df.columns
        # A cube has more surface area than equiv sphere -> roughness > 1
        assert df.iloc[0]["roughness"] > 1.0

    def test_sphere_has_higher_sphericity_than_rod(self):
        # Sphere-ish (cube is the closest easy shape)
        labels_a = np.zeros((20, 20, 20), dtype=np.int32)
        labels_a[3:17, 3:17, 3:17] = 1
        df_a = extract_features(labels_a)
        compute_surface_features(labels_a, df_a)

        # Rod-shaped (elongated along Z)
        labels_b = np.zeros((40, 12, 12), dtype=np.int32)
        labels_b[2:38, 2:10, 2:10] = 1
        df_b = extract_features(labels_b)
        compute_surface_features(labels_b, df_b)

        assert df_a.iloc[0]["true_sphericity"] > df_b.iloc[0]["true_sphericity"]

    def test_solidity_reasonable(self):
        labels = _make_labels(n_labels=5)
        df = extract_features(labels)
        compute_surface_features(labels, df)
        assert "solidity" in df.columns
        vals = df["solidity"].values
        assert np.all(np.isfinite(vals))
        # Solid convex-ish shapes should have solidity between 0 and ~2
        # (can slightly exceed 1 due to subsampling approximation)
        assert np.all(vals > 0)
        assert np.all(vals < 3.0)


class TestIntensityFeatures:
    """CP2: Tests for intensity feature extraction via regionprops."""

    def test_intensity_present_when_given(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        intensity = np.random.RandomState(0).rand(10, 10, 10).astype(np.float32)
        df = extract_features(labels, intensity_image=intensity)
        for col in ("intensity_mean", "intensity_std", "intensity_min", "intensity_max"):
            assert col in df.columns

    def test_intensity_absent_when_not_given(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        df = extract_features(labels)
        for col in ("intensity_mean", "intensity_std", "intensity_min", "intensity_max"):
            assert col not in df.columns

    def test_intensity_values_correct(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        # Uniform intensity of 5.0 inside label
        intensity = np.zeros((10, 10, 10), dtype=np.float32)
        intensity[2:8, 2:8, 2:8] = 5.0
        df = extract_features(labels, intensity_image=intensity)
        np.testing.assert_allclose(df.iloc[0]["intensity_mean"], 5.0, atol=1e-5)
        np.testing.assert_allclose(df.iloc[0]["intensity_std"], 0.0, atol=1e-5)
        np.testing.assert_allclose(df.iloc[0]["intensity_min"], 5.0, atol=1e-5)
        np.testing.assert_allclose(df.iloc[0]["intensity_max"], 5.0, atol=1e-5)


class TestBoundaryInteriorFeatures:
    """CP3: Tests for boundary/interior membrane intensity features."""

    def test_bright_boundary_dark_interior_high_ratio(self):
        labels = np.zeros((16, 16, 16), dtype=np.int32)
        labels[2:14, 2:14, 2:14] = 1
        # Bright boundary (membrane), dark interior
        intensity = np.zeros((16, 16, 16), dtype=np.float32)
        # Set everything in the label to low value (interior)
        intensity[2:14, 2:14, 2:14] = 1.0
        # Set the boundary shell to high value (membrane)
        intensity[2, 2:14, 2:14] = 10.0  # z faces
        intensity[13, 2:14, 2:14] = 10.0
        intensity[2:14, 2, 2:14] = 10.0  # y faces
        intensity[2:14, 13, 2:14] = 10.0
        intensity[2:14, 2:14, 2] = 10.0  # x faces
        intensity[2:14, 2:14, 13] = 10.0
        df = extract_features(labels, intensity_image=intensity)
        compute_boundary_interior_features(labels, intensity, df)
        assert df.iloc[0]["boundary_interior_ratio"] > 1.5

    def test_tiny_label_no_crash(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[4:6, 4:6, 4:6] = 1  # 2x2x2 — erosion eliminates interior
        intensity = np.ones((10, 10, 10), dtype=np.float32) * 3.0
        df = extract_features(labels, intensity_image=intensity)
        compute_boundary_interior_features(labels, intensity, df)
        # Should not crash, values should be finite
        for col in ("boundary_intensity_mean", "interior_intensity_mean",
                     "boundary_interior_ratio", "z_intensity_variation"):
            assert col in df.columns
            assert np.all(np.isfinite(df[col].values))

    def test_z_intensity_variation_present(self):
        labels = np.zeros((20, 10, 10), dtype=np.int32)
        labels[2:18, 2:8, 2:8] = 1
        # Vary intensity along Z to create variation
        intensity = np.zeros((20, 10, 10), dtype=np.float32)
        for z in range(2, 18):
            intensity[z, 2:8, 2:8] = float(z)  # increasing along Z
        df = extract_features(labels, intensity_image=intensity)
        compute_boundary_interior_features(labels, intensity, df)
        assert df.iloc[0]["z_intensity_variation"] > 0


class TestProbabilityFeatures:
    """Tests for probability consensus features from reconstruction probability volume."""

    _PROB_COLS = (
        "prob_mean", "prob_p10", "prob_std",
        "prob_boundary_mean", "prob_interior_mean", "prob_boundary_drop",
        "prob_weighted_volume", "prob_soft_hard_ratio",
        "prob_boundary_entropy", "prob_mask_entropy",
        "prob_edge_gradient", "prob_edge_gradient_std",
    )

    def test_all_columns_present_and_finite(self):
        labels = _make_labels(n_labels=3)
        prob = np.random.RandomState(42).rand(*labels.shape).astype(np.float32)
        df = extract_features(labels)
        compute_probability_features(labels, df, prob)
        for col in self._PROB_COLS:
            assert col in df.columns, f"Missing column: {col}"
            assert np.all(np.isfinite(df[col].values)), f"Non-finite in: {col}"

    def test_uniform_high_prob(self):
        """Uniform high probability → predictable values."""
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        prob = np.ones((10, 10, 10), dtype=np.float32) * 0.9
        df = extract_features(labels)
        compute_probability_features(labels, df, prob)
        row = df.iloc[0]
        np.testing.assert_allclose(row["prob_mean"], 0.9, atol=1e-5)
        np.testing.assert_allclose(row["prob_p10"], 0.9, atol=1e-5)
        np.testing.assert_allclose(row["prob_std"], 0.0, atol=1e-5)
        # Boundary drop ~ 0 (uniform)
        np.testing.assert_allclose(row["prob_boundary_drop"], 0.0, atol=1e-5)
        # Binary entropy at p=0.9 is small
        assert row["prob_mask_entropy"] < 0.4

    def test_boundary_drop_positive_when_interior_confident(self):
        """High interior, low boundary → positive boundary drop."""
        labels = np.zeros((16, 16, 16), dtype=np.int32)
        labels[2:14, 2:14, 2:14] = 1
        prob = np.zeros((16, 16, 16), dtype=np.float32)
        # High interior
        prob[2:14, 2:14, 2:14] = 0.9
        # Low boundary (faces)
        prob[2, 2:14, 2:14] = 0.2
        prob[13, 2:14, 2:14] = 0.2
        prob[2:14, 2, 2:14] = 0.2
        prob[2:14, 13, 2:14] = 0.2
        prob[2:14, 2:14, 2] = 0.2
        prob[2:14, 2:14, 13] = 0.2
        df = extract_features(labels)
        compute_probability_features(labels, df, prob)
        assert df.iloc[0]["prob_boundary_drop"] > 0.3

    def test_overseg_detection_low_ratio(self):
        """Mask extends into low-probability region → soft/hard ratio < 1."""
        labels = np.zeros((16, 16, 16), dtype=np.int32)
        labels[2:14, 2:14, 2:14] = 1  # large mask
        prob = np.zeros((16, 16, 16), dtype=np.float32)
        # Only half the mask has high probability
        prob[2:8, 2:14, 2:14] = 0.9
        prob[8:14, 2:14, 2:14] = 0.05  # low-P region inside mask
        df = extract_features(labels)
        compute_probability_features(labels, df, prob)
        # Mean should be middling
        assert df.iloc[0]["prob_mean"] < 0.6

    def test_float16_input(self):
        """Probability volume stored as float16 should work correctly."""
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        prob = (np.ones((10, 10, 10)) * 0.7).astype(np.float16)
        df = extract_features(labels)
        compute_probability_features(labels, df, prob)
        np.testing.assert_allclose(df.iloc[0]["prob_mean"], 0.7, atol=0.01)

    def test_empty_df(self):
        """No labels → empty DataFrame with columns present."""
        labels = np.zeros((8, 8, 8), dtype=np.int32)
        prob = np.ones((8, 8, 8), dtype=np.float32) * 0.5
        df = extract_features(labels)
        compute_probability_features(labels, df, prob)
        assert len(df) == 0
        for col in self._PROB_COLS:
            assert col in df.columns

    def test_single_voxel_no_crash(self):
        """Single-voxel label should not crash (erosion fallback)."""
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[4, 4, 4] = 1
        prob = np.ones((10, 10, 10), dtype=np.float32) * 0.6
        df = extract_features(labels)
        compute_probability_features(labels, df, prob)
        np.testing.assert_allclose(df.iloc[0]["prob_mean"], 0.6, atol=1e-5)
        # Boundary should fall back to mean for tiny labels
        np.testing.assert_allclose(
            df.iloc[0]["prob_boundary_mean"], 0.6, atol=1e-5
        )

    def test_boundary_entropy_peaks_at_half(self):
        """Boundary prob ≈ 0.5 → maximum binary entropy."""
        labels = np.zeros((16, 16, 16), dtype=np.int32)
        labels[2:14, 2:14, 2:14] = 1
        prob = np.zeros((16, 16, 16), dtype=np.float32)
        prob[2:14, 2:14, 2:14] = 0.5  # max entropy everywhere
        df = extract_features(labels)
        compute_probability_features(labels, df, prob)
        # Binary entropy at p=0.5 is ln(2) ≈ 0.693
        assert df.iloc[0]["prob_boundary_entropy"] > 0.6

    def test_multiple_labels(self):
        """Multiple labels — all values in valid ranges."""
        labels = _make_labels(n_labels=5)
        prob = np.random.RandomState(99).rand(*labels.shape).astype(np.float32)
        df = extract_features(labels)
        compute_probability_features(labels, df, prob)
        assert len(df) == 5
        assert np.all(df["prob_mean"].values >= 0)
        assert np.all(df["prob_mean"].values <= 1)
        assert np.all(df["prob_p10"].values >= 0)
        assert np.all(np.isfinite(df["prob_edge_gradient"].values))


class TestNeighborFeatures:
    """CP4: Tests for neighbor/adjacency features (graduated dilation)."""

    def test_adjacent_labels_detected(self):
        labels = np.zeros((10, 10, 20), dtype=np.int32)
        labels[2:8, 2:8, 2:10] = 1
        labels[2:8, 2:8, 10:18] = 2  # adjacent to label 1 along X
        df = extract_features(labels)
        compute_neighbor_features(labels, df)
        # Each should have at least 1 neighbor
        assert df[df["label"] == 1].iloc[0]["neighbor_count"] >= 1
        assert df[df["label"] == 2].iloc[0]["neighbor_count"] >= 1
        # Touching labels: nearest neighbor distance should be 1
        assert df[df["label"] == 1].iloc[0]["nearest_neighbor_distance"] == 1

    def test_isolated_label_zero_neighbors(self):
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[8:12, 8:12, 8:12] = 1  # isolated in center
        df = extract_features(labels)
        compute_neighbor_features(labels, df)
        assert df.iloc[0]["neighbor_count"] == 0
        # No neighbor found: distance should be max_radius + 1
        assert df.iloc[0]["nearest_neighbor_distance"] == 6  # default max_radius=5

    def test_background_fraction_correct(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1  # surrounded by background
        df = extract_features(labels)
        compute_neighbor_features(labels, df)
        # Entirely surrounded by background
        assert df.iloc[0]["background_fraction"] > 0.5

    def test_gap_bridged_by_graduated_dilation(self):
        """Non-touching labels with a 3-voxel gap are detected as neighbors."""
        labels = np.zeros((10, 10, 24), dtype=np.int32)
        labels[2:8, 2:8, 2:9] = 1
        labels[2:8, 2:8, 12:20] = 2  # 3-voxel gap (indices 9,10,11)
        df = extract_features(labels)
        # With max_radius=1 the old approach would miss this
        compute_neighbor_features(labels, df, max_radius=1)
        assert df[df["label"] == 1].iloc[0]["neighbor_count"] == 0
        # With default max_radius=5 the gap is bridged
        df2 = extract_features(labels)
        compute_neighbor_features(labels, df2, max_radius=5)
        assert df2[df2["label"] == 1].iloc[0]["neighbor_count"] >= 1
        # Distance should reflect the gap width (~3)
        dist = df2[df2["label"] == 1].iloc[0]["nearest_neighbor_distance"]
        assert 2 <= dist <= 4

    def test_nearest_neighbor_distance_ordering(self):
        """Closer neighbor has smaller distance than farther neighbor."""
        labels = np.zeros((10, 10, 30), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1    # left cell
        labels[2:8, 2:8, 9:15] = 2   # 1-voxel gap from label 1
        labels[2:8, 2:8, 19:25] = 3  # 4-voxel gap from label 2
        df = extract_features(labels)
        compute_neighbor_features(labels, df, max_radius=5)
        # Label 1's nearest neighbor is label 2 at distance 1
        d1 = df[df["label"] == 1].iloc[0]["nearest_neighbor_distance"]
        # Label 3's nearest neighbor is label 2 at distance 4
        d3 = df[df["label"] == 3].iloc[0]["nearest_neighbor_distance"]
        assert d1 < d3


class TestSkeletonFeatures:
    """CP4: Tests for skeleton topology features."""

    def test_simple_cube_few_branches(self):
        labels = np.zeros((16, 16, 16), dtype=np.int32)
        labels[2:14, 2:14, 2:14] = 1  # solid cube
        df = extract_features(labels)
        compute_skeleton_features(labels, df)
        assert "skeleton_branches" in df.columns
        assert "skeleton_length" in df.columns
        # A cube should have a simple skeleton with few branches
        assert df.iloc[0]["skeleton_branches"] >= 1
        assert df.iloc[0]["skeleton_length"] > 0

    def test_elongated_shape_more_skeleton_length(self):
        # Compact cube (8x8x8 interior)
        labels_a = np.zeros((12, 12, 12), dtype=np.int32)
        labels_a[2:10, 2:10, 2:10] = 1
        df_a = extract_features(labels_a)
        compute_skeleton_features(labels_a, df_a)

        # Elongated rod — wide enough for meaningful skeleton (10 voxels cross-section)
        labels_b = np.zeros((60, 16, 16), dtype=np.int32)
        labels_b[2:58, 3:13, 3:13] = 1
        df_b = extract_features(labels_b)
        compute_skeleton_features(labels_b, df_b)

        # Rod should have more skeleton length than compact cube
        assert df_b.iloc[0]["skeleton_length"] > df_a.iloc[0]["skeleton_length"]


class TestDiscreteCompactness:
    """Tests for Bribiesca discrete compactness."""

    def test_column_added(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_discrete_compactness,
        )
        labels = _make_labels(n_labels=3)
        df = extract_features(labels)
        compute_discrete_compactness(labels, df)
        assert "discrete_compactness" in df.columns
        assert len(df) == 3

    def test_perfect_cube_returns_one(self):
        """C_d for a cube of side L should approach 1.0."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_discrete_compactness,
        )
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[3:13, 3:13, 3:13] = 1   # 10x10x10 cube => 1000 voxels
        df = extract_features(labels)
        compute_discrete_compactness(labels, df)
        cd = float(df.iloc[0]["discrete_compactness"])
        # For a 10^3 cube: n=1000, A_v = 6 * 100 = 600
        # C_d = (1000 - 100) / (1000 - 100) = 1.0 exactly
        assert abs(cd - 1.0) < 1e-9, f"Cube C_d should be 1.0, got {cd}"

    def test_sphere_below_cube(self):
        """A sphere should have lower discrete_compactness than a cube of similar volume."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_discrete_compactness,
        )
        # Build a sphere mask (label 1) and a cube mask (label 2) with similar volume
        shape = (40, 40, 40)
        zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
        sphere_mask = ((zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) <= 49  # r=7
        cube_mask = np.zeros(shape, dtype=bool)
        cube_mask[25:35, 25:35, 25:35] = True   # 10^3 = 1000 voxels
        labels = np.zeros(shape, dtype=np.int32)
        labels[sphere_mask] = 1
        labels[cube_mask] = 2

        df = extract_features(labels)
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_discrete_compactness,
        )
        compute_discrete_compactness(labels, df)
        cd_sphere = float(df[df["label"] == 1].iloc[0]["discrete_compactness"])
        cd_cube = float(df[df["label"] == 2].iloc[0]["discrete_compactness"])
        # Cube = exactly 1.0; sphere is < 1.0
        # (for small voxelized spheres, discretization makes them more
        # cube-like than smooth-surface theory predicts; converges to ~0.85
        # only at much larger radii)
        assert cd_cube > cd_sphere, (
            f"Cube ({cd_cube}) should be more compact than sphere ({cd_sphere})"
        )
        assert 0.7 <= cd_sphere < 1.0, (
            f"Voxelized sphere C_d should be in (0.7, 1.0), got {cd_sphere}"
        )

    def test_thin_rod_lower_than_cube(self):
        """A thin rod has many external faces relative to volume → lower C_d
        than a cube of similar volume."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_discrete_compactness,
        )
        labels = np.zeros((20, 20, 50), dtype=np.int32)
        labels[10, 10, 5:45] = 1                   # 1x1x40 thin rod, 40 voxels
        labels[2:6, 2:6, 2:6] = 2                  # 4x4x4 small cube, 64 voxels
        df = extract_features(labels)
        compute_discrete_compactness(labels, df)
        cd_rod = float(df[df["label"] == 1].iloc[0]["discrete_compactness"])
        cd_cube = float(df[df["label"] == 2].iloc[0]["discrete_compactness"])
        assert cd_rod < cd_cube, f"Rod ({cd_rod}) should be < cube ({cd_cube})"
        assert cd_cube == 1.0, f"4x4x4 cube should give exactly 1.0, got {cd_cube}"
        # Thin rod is around 0.4-0.5 for this aspect ratio
        assert 0.0 < cd_rod < 0.6, f"Thin rod C_d should be < 0.6, got {cd_rod}"

    def test_single_voxel_returns_zero(self):
        """A single voxel is the degenerate case — C_d defined as 0."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_discrete_compactness,
        )
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[5, 5, 5] = 1
        df = extract_features(labels)
        compute_discrete_compactness(labels, df)
        cd = float(df.iloc[0]["discrete_compactness"])
        assert cd == 0.0


class TestCompositeScore:
    """CP5: Tests for composite QC outlier score."""

    def test_score_present(self):
        labels = _make_labels(n_labels=5)
        df = extract_features(labels)
        compute_composite_score(df)
        assert "qc_score" in df.columns
        assert len(df) == 5
        assert np.all(np.isfinite(df["qc_score"].values))

    def test_zscore_method_non_negative(self):
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        compute_composite_score(df, method="zscore")
        # Mean absolute z-score is always >= 0
        assert np.all(df["qc_score"].values >= 0)


class TestVoxelSpacing:
    def test_extract_with_spacing_scales_area(self):
        labels = _make_labels(n_labels=3)
        df_voxel = extract_features(labels)
        df_phys = extract_features(labels, spacing=(2.0, 1.0, 1.0))
        # With spacing (2,1,1), physical volume = voxel_count * 2*1*1 = 2x
        ratio = df_phys["area"].values / df_voxel["area"].values
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-6)

    def test_extract_no_spacing_unchanged(self):
        labels = _make_labels(n_labels=3)
        df1 = extract_features(labels)
        df2 = extract_features(labels, spacing=None)
        pd.testing.assert_frame_equal(df1, df2)

    def test_extract_isotropic_spacing(self):
        labels = _make_labels(n_labels=3)
        df_voxel = extract_features(labels)
        df_iso = extract_features(labels, spacing=(0.5, 0.5, 0.5))
        # Volume scales as spacing^3 = 0.125x
        ratio = df_iso["area"].values / df_voxel["area"].values
        np.testing.assert_allclose(ratio, 0.125, rtol=1e-6)

    def test_spacing_label_none(self):
        assert spacing_label(None) == "voxel units"

    def test_spacing_label_tuple(self):
        result = spacing_label((0.3, 0.1, 0.1))
        assert "0.3" in result
        assert "0.1" in result
        assert "µm" in result


class TestCentroidGridStorage:
    """Centroids must be stored in GRID (voxel-index) coords, NOT physical.

    regionprops(spacing=) scales area/volume/centroid into physical units, but
    the centroid is only ever used to point the camera at a cell (grid coords).
    So the centroid columns must be spacing-INDEPENDENT, while area/volume stay
    physical.  See the navigator auto-pan "corner" bug at non-unit voxel sizes.
    """

    def test_centroid_grid_isotropic_spacing(self):
        # The user's case: voxel size 0.26 on all axes.
        labels = _make_labels(n_labels=3)
        df_voxel = extract_features(labels)                       # grid centroid
        df_phys = extract_features(labels, spacing=(0.26, 0.26, 0.26))
        for ax in ("centroid-0", "centroid-1", "centroid-2"):
            np.testing.assert_allclose(
                df_phys[ax].values, df_voxel[ax].values, rtol=1e-6,
                err_msg=f"{ax} must be spacing-independent (grid), got physical",
            )

    def test_centroid_grid_anisotropic_spacing(self):
        labels = _make_labels(n_labels=3)
        df_voxel = extract_features(labels)
        df_aniso = extract_features(labels, spacing=(4.0, 0.25, 0.25))
        for ax in ("centroid-0", "centroid-1", "centroid-2"):
            np.testing.assert_allclose(
                df_aniso[ax].values, df_voxel[ax].values, rtol=1e-6,
            )

    def test_centroid_spacing_one_byte_identical(self):
        labels = _make_labels(n_labels=3)
        df_voxel = extract_features(labels)
        df_one = extract_features(labels, spacing=(1.0, 1.0, 1.0))
        for ax in ("centroid-0", "centroid-1", "centroid-2"):
            np.testing.assert_allclose(
                df_one[ax].values, df_voxel[ax].values, rtol=1e-9,
            )

    def test_area_still_physical_with_spacing(self):
        # The fix must NOT touch area/volume — those stay physical.
        labels = _make_labels(n_labels=3)
        df_voxel = extract_features(labels)
        df_phys = extract_features(labels, spacing=(2.0, 1.0, 1.0))
        ratio = df_phys["area"].values / df_voxel["area"].values
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-6)

    def test_centroid_spacing_zero_no_inf(self):
        # spacing=0 is NOT reachable via the clamped [0.001, 100] dialog, but a
        # degenerate auto-detected layer scale could supply it.  The guard skips
        # the divide so the (already-zero) centroid stays finite — no 0/0=NaN
        # that the navigator's int() would crash on.
        labels = _make_labels(n_labels=3)
        df = extract_features(labels, spacing=(0.0, 1.0, 1.0))
        assert np.all(np.isfinite(df["centroid-0"].values))

    def test_single_label_centroid_grid_with_spacing(self):
        # extract_single_label_features must also store grid centroids.
        labels = _make_labels(n_labels=3)
        df_no = extract_single_label_features(labels, 1)
        df_sp = extract_single_label_features(labels, 1, spacing=(0.26, 0.26, 0.26))
        for ax in ("centroid-0", "centroid-1", "centroid-2"):
            np.testing.assert_allclose(
                df_sp[ax].values, df_no[ax].values, rtol=1e-6,
            )


class TestPCAFeaturesCurated:
    """K8c: Tests for curated PCA feature set."""

    def test_pca_features_is_subset_of_numeric(self):
        """All PCA features should be valid NUMERIC_FEATURES entries."""
        for feat in PCA_FEATURES:
            assert feat in NUMERIC_FEATURES, f"{feat} not in NUMERIC_FEATURES"

    def test_pca_features_excludes_qc_score(self):
        """qc_score must never be in PCA_FEATURES (circular dependency)."""
        assert "qc_score" not in PCA_FEATURES

    def test_pca_features_excludes_redundant_pairs(self):
        """PCA_FEATURES must not contain mathematically-redundant pairs.

        compactness == extent (both V/V_bbox).
        roughness == 1/true_sphericity (perfect inverse).
        Including either pair double-weights that dimension in PCA.
        """
        assert not ("compactness" in PCA_FEATURES and "extent" in PCA_FEATURES), (
            "compactness and extent compute the same value (V/V_bbox)"
        )
        assert not ("roughness" in PCA_FEATURES and "true_sphericity" in PCA_FEATURES), (
            "roughness is the perfect inverse of true_sphericity"
        )

    def test_pca_features_count(self):
        """Curated set should have ~16 features."""
        assert 10 <= len(PCA_FEATURES) <= 20

    def test_pca_with_curated_features(self):
        """PCA using curated defaults produces valid output."""
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        projected = compute_pca(df, n_components=2)  # defaults to PCA_FEATURES
        assert projected.shape == (10, 2)
        assert np.all(np.isfinite(projected))

    def test_pca_with_custom_features(self):
        """PCA with explicit feature list uses only those columns."""
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        custom = ["area", "extent"]
        projected = compute_pca(df, n_components=2, features=custom)
        assert projected.shape == (10, 2)
        assert np.all(np.isfinite(projected))

    def test_composite_score_uses_curated_by_default(self):
        """compute_composite_score should use PCA_FEATURES by default."""
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        compute_composite_score(df)
        assert "qc_score" in df.columns
        assert np.all(np.isfinite(df["qc_score"].values))

    def test_composite_score_with_custom_features(self):
        """compute_composite_score respects explicit feature list."""
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        compute_composite_score(df, features=["area", "extent"])
        assert "qc_score" in df.columns
        assert np.all(np.isfinite(df["qc_score"].values))

    def test_composite_score_excludes_qc_score_from_input(self):
        """Even if qc_score is in the feature list, it should be excluded."""
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        # Pre-populate qc_score
        df["qc_score"] = 999.0
        compute_composite_score(df, features=["area", "extent", "qc_score"])
        # Should have been recomputed, not 999.0
        assert not np.all(df["qc_score"].values == 999.0)


class TestRobustPreprocessing:
    """K8d: Tests for robust PCA preprocessing (log1p + median/MAD)."""

    def test_log_transform_features_exist(self):
        """LOG_TRANSFORM_FEATURES should all be valid NUMERIC_FEATURES."""
        for feat in LOG_TRANSFORM_FEATURES:
            assert feat in NUMERIC_FEATURES, f"{feat} not in NUMERIC_FEATURES"

    def test_robust_standardize_centers_data(self):
        """After robust standardization, median of each column should be ~0."""
        X = np.random.RandomState(42).randn(100, 3) * 10 + 50
        cols = ["col_a", "col_b", "col_c"]
        X_std = _robust_standardize(X.copy(), cols)
        medians = np.median(X_std, axis=0)
        np.testing.assert_allclose(medians, 0.0, atol=1e-10)

    def test_robust_standardize_log_transforms_skewed(self):
        """Features in LOG_TRANSFORM_FEATURES get log1p applied."""
        X = np.array([[100, 0.5], [1000, 0.8], [10000, 0.3]])
        cols = ["area", "extent"]  # area is in LOG_TRANSFORM, extent is not
        X_std = _robust_standardize(X.copy(), cols)
        # After log1p + centering, the range of column 0 (area) should be
        # much more compressed than raw values would suggest
        assert X_std.shape == (3, 2)
        assert np.all(np.isfinite(X_std))

    def test_robust_standardize_constant_column(self):
        """Constant columns should not cause division by zero."""
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        cols = ["const_feat", "varying_feat"]
        X_std = _robust_standardize(X.copy(), cols)
        assert np.all(np.isfinite(X_std))
        # Constant column should be all zeros after centering
        np.testing.assert_allclose(X_std[:, 0], 0.0, atol=1e-10)

    def test_skew_detection_finds_right_skewed(self):
        """Highly right-skewed column should be detected."""
        rng = np.random.RandomState(42)
        X = np.column_stack([
            rng.exponential(scale=1000, size=500),  # skewed
            rng.randn(500),  # normal
        ])
        cols = ["skewed_feat", "normal_feat"]
        indices = _detect_skewed_columns(X, cols, threshold=1.0)
        assert 0 in indices  # skewed column detected
        assert 1 not in indices  # normal column not flagged

    def test_skew_detection_skips_bounded(self):
        """Bounded features should not be flagged even if skewed."""
        rng = np.random.RandomState(7)
        X = np.column_stack([
            rng.exponential(scale=100, size=500),  # skewed + unbounded
            rng.beta(0.5, 5, size=500),  # skewed but bounded
        ])
        cols = ["area", "extent"]  # extent is in BOUNDED_FEATURES
        indices = _detect_skewed_columns(X, cols, threshold=1.0)
        assert 0 in indices
        assert 1 not in indices

    def test_skew_detection_handles_channel_features(self):
        """Per-channel intensity features should be auto-detected."""
        rng = np.random.RandomState(99)
        X = np.column_stack([
            rng.exponential(scale=500, size=500),  # nuclei_intensity_sum
            rng.randn(500) + 50,  # nuclei_intensity_mean (normal-ish)
        ])
        cols = ["nuclei_intensity_sum", "nuclei_intensity_mean"]
        indices = _detect_skewed_columns(X, cols, threshold=1.0)
        assert 0 in indices  # sum is skewed

    def test_robust_standardize_auto_detects_skewness(self):
        """Auto-detection should log-transform skewed columns."""
        rng = np.random.RandomState(42)
        X = np.column_stack([
            rng.exponential(scale=1000, size=100),  # skewed
            rng.randn(100) * 10 + 50,  # normal
        ])
        cols = ["area", "some_normal"]
        X_std = _robust_standardize(X.copy(), cols)
        assert np.all(np.isfinite(X_std))
        # After standardization, both columns should have median ~0
        medians = np.median(X_std, axis=0)
        np.testing.assert_allclose(medians, 0.0, atol=1e-10)

    def test_left_skewed_not_flagged(self):
        """Left-skewed features should NOT be flagged (log1p only fixes right)."""
        rng = np.random.RandomState(42)
        # Create left-skewed data (negate exponential)
        X = np.column_stack([
            -rng.exponential(scale=1000, size=500),  # left-skewed
        ])
        cols = ["left_skewed"]
        indices = _detect_skewed_columns(X, cols, threshold=1.0)
        assert len(indices) == 0  # should NOT be flagged

    def test_prob_soft_hard_ratio_not_bounded(self):
        """prob_soft_hard_ratio is unbounded (V_soft/|M| can exceed 1)."""
        assert "prob_soft_hard_ratio" not in BOUNDED_FEATURES

    def test_bounded_features_not_empty(self):
        """BOUNDED_FEATURES should contain known bounded features."""
        assert "extent" in BOUNDED_FEATURES
        assert "compactness" in BOUNDED_FEATURES
        assert "solidity" in BOUNDED_FEATURES
        # area should NOT be bounded
        assert "area" not in BOUNDED_FEATURES

    def test_legacy_log_transform_features_still_exists(self):
        """LOG_TRANSFORM_FEATURES kept for backward compatibility."""
        assert "area" in LOG_TRANSFORM_FEATURES

    def test_pca_uses_robust_standardize(self):
        """PCA output should be finite with skewed data (log1p handles it)."""
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        # Make area extremely skewed
        df.loc[df.index[0], "area"] = 1_000_000
        projected = compute_pca(df, n_components=2)
        assert projected.shape == (10, 2)
        assert np.all(np.isfinite(projected))

    def test_composite_zscore_uses_robust_standardize(self):
        """Composite z-score should use robust (median/MAD) standardization."""
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        # Inject an extreme outlier
        df.loc[df.index[0], "area"] = 1_000_000
        compute_composite_score(df, method="zscore")
        # The outlier should have a high score
        scores = df["qc_score"].values
        assert scores[0] > np.median(scores)
        assert np.all(np.isfinite(scores))

    def test_composite_pca_uses_robust_standardize(self):
        """Composite PCA method should produce finite scores with skewed data."""
        labels = _make_labels(n_labels=10)
        df = extract_features(labels)
        df.loc[df.index[0], "area"] = 1_000_000
        compute_composite_score(df, method="pca")
        assert np.all(np.isfinite(df["qc_score"].values))


class TestClustering:
    """Tests for multi-algorithm clustering (compute_clusters)."""

    def test_hdbscan_adds_cluster_id(self):
        from napari_epicycle.fusion_reconstruction.qc_features import compute_clusters
        labels = _make_labels(n_labels=20, shape=(64, 96, 96))
        df = extract_features(labels)
        compute_clusters(df, algorithm="hdbscan", min_cluster_size=5)
        assert "cluster_id" in df.columns
        assert len(df) == 20

    def test_hdbscan_small_dataset_all_noise(self):
        from napari_epicycle.fusion_reconstruction.qc_features import compute_clusters
        labels = _make_labels(n_labels=3)
        df = extract_features(labels)
        compute_clusters(df, algorithm="hdbscan", min_cluster_size=50)
        assert "cluster_id" in df.columns
        assert (df["cluster_id"] == -1).all()

    def test_kmeans_adds_cluster_id(self):
        from napari_epicycle.fusion_reconstruction.qc_features import compute_clusters
        labels = _make_labels(n_labels=20, shape=(64, 96, 96))
        df = extract_features(labels)
        compute_clusters(df, algorithm="kmeans", n_clusters=3)
        assert "cluster_id" in df.columns
        assert (df["cluster_id"] >= 0).all()

    def test_gmm_adds_cluster_id(self):
        from napari_epicycle.fusion_reconstruction.qc_features import compute_clusters
        labels = _make_labels(n_labels=20, shape=(64, 96, 96))
        df = extract_features(labels)
        compute_clusters(df, algorithm="gmm", n_clusters=3)
        assert "cluster_id" in df.columns
        assert (df["cluster_id"] >= 0).all()

    def test_leiden_adds_cluster_id(self):
        from napari_epicycle.fusion_reconstruction.qc_features import compute_clusters
        labels = _make_labels(n_labels=20, shape=(64, 96, 96))
        df = extract_features(labels)
        compute_clusters(df, algorithm="leiden", resolution=1.0, n_neighbors=5)
        assert "cluster_id" in df.columns
        # Leiden assigns every point — no -1 noise
        assert (df["cluster_id"] >= 0).all()

    def test_custom_features(self):
        from napari_epicycle.fusion_reconstruction.qc_features import compute_clusters
        labels = _make_labels(n_labels=20, shape=(64, 96, 96))
        df = extract_features(labels)
        compute_clusters(df, algorithm="kmeans", n_clusters=2, features=["area", "extent"])
        assert "cluster_id" in df.columns

    def test_backward_compat_wrapper(self):
        from napari_epicycle.fusion_reconstruction.qc_features import compute_hdbscan_clusters
        labels = _make_labels(n_labels=20, shape=(64, 96, 96))
        df = extract_features(labels)
        compute_hdbscan_clusters(df, min_cluster_size=5)
        assert "cluster_id" in df.columns


class TestAssignQCTiers:
    """Tests for QC tier assignment from thresholds."""

    def test_tier_assignment_basic(self):
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers
        df = pd.DataFrame({
            "label": [1, 2, 3, 4, 5],
            "qc_score": [0.5, 1.5, 2.5, 3.5, 4.5],
        })
        assign_qc_tiers(df, (1.0, 2.0, 3.0))
        assert "qc_tier" in df.columns
        assert df.iloc[0]["qc_tier"] == "good"
        assert df.iloc[1]["qc_tier"] == "review"
        assert df.iloc[2]["qc_tier"] == "bad"
        assert df.iloc[3]["qc_tier"] == "garbage"
        assert df.iloc[4]["qc_tier"] == "garbage"

    def test_tier_assignment_sorts_thresholds(self):
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers
        df = pd.DataFrame({
            "label": [1, 2],
            "qc_score": [0.5, 5.0],
        })
        # Reversed thresholds should be sorted internally
        assign_qc_tiers(df, (3.0, 1.0, 2.0))
        assert df.iloc[0]["qc_tier"] == "good"
        assert df.iloc[1]["qc_tier"] == "garbage"

    def test_tier_assignment_no_qc_score(self):
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers
        df = pd.DataFrame({"label": [1, 2, 3]})
        assign_qc_tiers(df, (1.0, 2.0, 3.0))
        assert "qc_tier" in df.columns
        assert (df["qc_tier"] == "good").all()  # all zeros ≤ 1.0

    def test_tier_assignment_empty_df(self):
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers
        df = pd.DataFrame({"label": [], "qc_score": []})
        assign_qc_tiers(df, (1.0, 2.0, 3.0))
        assert "qc_tier" in df.columns
        assert len(df) == 0


class TestTierPersistence:
    """Tests for QC tier persistence across CSV save/load and merge."""

    def test_csv_roundtrip_preserves_qc_tier(self, tmp_path):
        """Saved qc_tier column should survive CSV load."""
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers
        labels = _make_labels(n_labels=5)
        df = extract_features(labels)
        compute_composite_score(df)
        assign_qc_tiers(df, (1.0, 2.0, 3.0))
        # Manually override one tier
        df.loc[df.index[0], "qc_tier"] = "garbage"
        csv_path = tmp_path / "test_features.csv"
        save_patch_csv(df, csv_path)
        loaded = load_patch_csv(csv_path)
        assert "qc_tier" in loaded.columns
        assert loaded.iloc[0]["qc_tier"] == "garbage"

    def test_merge_preserves_qc_tier(self, tmp_path):
        """Merged CSVs should preserve per-patch qc_tier columns."""
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers
        for name in ("patch001", "patch002"):
            labels = _make_labels(n_labels=3)
            df = extract_features(labels)
            compute_composite_score(df)
            assign_qc_tiers(df, (1.0, 2.0, 3.0))
            save_patch_csv(df, tmp_path / f"{name}_features.csv")
        merged = merge_patch_csvs([
            tmp_path / "patch001_features.csv",
            tmp_path / "patch002_features.csv",
        ])
        assert "qc_tier" in merged.columns
        assert "patch_name" in merged.columns
        assert merged["qc_tier"].notna().all()

    def test_merge_mixed_tier_presence(self, tmp_path):
        """Merge with one CSV having qc_tier and one without."""
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers
        # Patch with tiers
        labels1 = _make_labels(n_labels=3)
        df1 = extract_features(labels1)
        compute_composite_score(df1)
        assign_qc_tiers(df1, (1.0, 2.0, 3.0))
        save_patch_csv(df1, tmp_path / "patch001_features.csv")
        # Patch without tiers (fresh extraction)
        labels2 = _make_labels(n_labels=3)
        df2 = extract_features(labels2)
        compute_composite_score(df2)
        save_patch_csv(df2, tmp_path / "patch002_features.csv")
        merged = merge_patch_csvs([
            tmp_path / "patch001_features.csv",
            tmp_path / "patch002_features.csv",
        ])
        # patch001 rows should have tier, patch002 rows should be NaN
        p1_rows = merged[merged["patch_name"] == "patch001"]
        p2_rows = merged[merged["patch_name"] == "patch002"]
        assert p1_rows["qc_tier"].notna().all()
        assert p2_rows["qc_tier"].isna().all()


# ---------------------------------------------------------------------------
# K13.1 — Curated tier preservation
# ---------------------------------------------------------------------------
class TestAssignQCTiersPreservesCurated:
    """Curated labels must survive ``assign_qc_tiers()`` calls."""

    def test_curated_preserved_on_reassign(self):
        """A single curated row survives a full tier reassignment."""
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers

        df = pd.DataFrame({
            "label": [1, 2, 3, 4],
            "qc_score": [0.5, 1.5, 2.5, 3.5],
            "qc_tier": ["curated", "review", "bad", "garbage"],
        })
        assign_qc_tiers(df, (1.0, 2.0, 3.0))
        # Label 1 had curated — must stay curated despite score mapping to "good"
        assert df.loc[df["label"] == 1, "qc_tier"].iloc[0] == "curated"
        # Other labels reassigned normally
        assert df.loc[df["label"] == 2, "qc_tier"].iloc[0] == "review"
        assert df.loc[df["label"] == 3, "qc_tier"].iloc[0] == "bad"
        assert df.loc[df["label"] == 4, "qc_tier"].iloc[0] == "garbage"

    def test_curated_preserved_multiple(self):
        """Multiple curated rows all preserved with new thresholds."""
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers

        df = pd.DataFrame({
            "label": [1, 2, 3, 4, 5],
            "qc_score": [0.1, 0.9, 1.5, 2.5, 4.0],
            "qc_tier": ["curated", "good", "curated", "bad", "curated"],
        })
        assign_qc_tiers(df, (0.5, 1.2, 3.0))
        curated_labels = df.loc[df["qc_tier"] == "curated", "label"].tolist()
        assert sorted(curated_labels) == [1, 3, 5]

    def test_no_existing_tier_column(self):
        """First-time assignment works normally (no curated to preserve)."""
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers

        df = pd.DataFrame({
            "label": [1, 2, 3],
            "qc_score": [0.5, 1.5, 3.5],
        })
        # Should not raise even though there is no qc_tier column yet
        assign_qc_tiers(df, (1.0, 2.0, 3.0))
        assert df.iloc[0]["qc_tier"] == "good"
        assert df.iloc[1]["qc_tier"] == "review"
        assert df.iloc[2]["qc_tier"] == "garbage"


# ---------------------------------------------------------------------------
# K13.1 — Single-label QC feature extraction
# ---------------------------------------------------------------------------
class TestExtractSingleLabelFeatures:
    """Tests for ``extract_single_label_features()`` — crop + extract for one label."""

    def test_single_label_basic(self):
        """Returns 1-row DataFrame with correct label and area > 0."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_label_features,
        )

        labels = _make_labels(n_labels=3)
        df = extract_single_label_features(labels, label_id=2)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["label"] == 2
        assert df.iloc[0]["area"] > 0

    def test_single_label_nonexistent(self):
        """Returns empty DataFrame for a label not present in volume."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_label_features,
        )

        labels = _make_labels(n_labels=3)
        df = extract_single_label_features(labels, label_id=999)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_single_label_centroids_match_full(self):
        """Centroids from single-label extraction match full extraction."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_label_features,
        )

        labels = _make_labels(n_labels=5)
        # Full extraction for reference
        df_full = extract_features(labels)
        ref_row = df_full[df_full["label"] == 3].iloc[0]
        # Single-label extraction
        df_single = extract_single_label_features(labels, label_id=3)
        single_row = df_single.iloc[0]
        # Centroids should match (adjusted back to full-volume coordinates)
        for ax in ("centroid-0", "centroid-1", "centroid-2"):
            assert ax in ref_row.index, f"Missing {ax} in full extraction"
            assert ax in single_row.index, f"Missing {ax} in single extraction"
            np.testing.assert_allclose(
                single_row[ax], ref_row[ax], atol=0.5,
                err_msg=f"{ax} mismatch",
            )

    def test_single_label_offset_correctness(self):
        """Centroids and bboxes are in full-volume coordinates, not crop-local."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_label_features,
        )

        # Place a label far from origin so crop offset is significant
        labels = np.zeros((64, 64, 64), dtype=np.int32)
        labels[40:50, 30:40, 20:30] = 7  # offset from origin
        df = extract_single_label_features(labels, label_id=7)
        assert len(df) == 1
        row = df.iloc[0]

        # Centroid must be within the label's voxel range (not near zero)
        assert row["centroid-0"] >= 40 and row["centroid-0"] < 50, (
            f"centroid-0 = {row['centroid-0']}, expected in [40, 50)"
        )
        assert row["centroid-1"] >= 30 and row["centroid-1"] < 40, (
            f"centroid-1 = {row['centroid-1']}, expected in [30, 40)"
        )
        assert row["centroid-2"] >= 20 and row["centroid-2"] < 30, (
            f"centroid-2 = {row['centroid-2']}, expected in [20, 30)"
        )

        # Bounding box must enclose the label region
        assert row["bbox-0"] <= 40, f"bbox-0 = {row['bbox-0']}, expected <= 40"
        assert row["bbox-3"] >= 50, f"bbox-3 = {row['bbox-3']}, expected >= 50"
        assert row["bbox-1"] <= 30, f"bbox-1 = {row['bbox-1']}, expected <= 30"
        assert row["bbox-4"] >= 40, f"bbox-4 = {row['bbox-4']}, expected >= 40"
        assert row["bbox-2"] <= 20, f"bbox-2 = {row['bbox-2']}, expected <= 20"
        assert row["bbox-5"] >= 30, f"bbox-5 = {row['bbox-5']}, expected >= 30"

    def test_single_label_with_intensity(self):
        """intensity_mean present when intensity image provided."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_label_features,
        )

        labels = _make_labels(n_labels=3)
        intensity = np.random.RandomState(42).rand(*labels.shape).astype(np.float32)
        df = extract_single_label_features(
            labels, label_id=1, intensity_image=intensity,
        )
        assert len(df) == 1
        assert "intensity_mean" in df.columns
        assert df.iloc[0]["intensity_mean"] > 0

    def test_single_label_has_surface_features(self):
        """surface_area and true_sphericity present in result."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_label_features,
        )

        labels = _make_labels(n_labels=3)
        df = extract_single_label_features(labels, label_id=2)
        assert "surface_area" in df.columns
        assert "true_sphericity" in df.columns

    def test_single_label_has_skeleton_features(self):
        """skeleton_branches present in result."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_label_features,
        )

        labels = _make_labels(n_labels=3)
        df = extract_single_label_features(labels, label_id=2)
        assert "skeleton_branches" in df.columns

    def test_single_label_neighbor_features_nan(self):
        """Neighbor features should be NaN for isolated single-label volume."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_label_features,
        )

        labels = _make_labels(n_labels=3)
        df = extract_single_label_features(labels, label_id=2)
        # Neighbor features are meaningless in cropped isolation
        for col in ("n_neighbors", "neighbor_dist_mean"):
            if col in df.columns:
                assert np.isnan(df.iloc[0][col]) or df.iloc[0][col] == 0

    def test_single_label_no_composite_score(self):
        """qc_score should NOT be computed (z-score on 1 row is meaningless)."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_label_features,
        )

        labels = _make_labels(n_labels=3)
        df = extract_single_label_features(labels, label_id=2)
        # qc_score column should either be absent or 0.0
        if "qc_score" in df.columns:
            assert df.iloc[0]["qc_score"] == 0.0

    def test_rejects_2d(self):
        """ValueError for 2D input."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_label_features,
        )

        labels_2d = np.zeros((48, 48), dtype=np.int32)
        labels_2d[10:20, 10:20] = 1
        with pytest.raises(ValueError, match="3D"):
            extract_single_label_features(labels_2d, label_id=1)


# ---------------------------------------------------------------------------
# K13.1 — Curated tier CSV persistence
# ---------------------------------------------------------------------------
class TestCuratedTierCSVPersistence:
    """Curated tier survives CSV roundtrip and merge."""

    def test_csv_roundtrip_preserves_curated_tier(self, tmp_path):
        """Save + load preserves 'curated' as a qc_tier value."""
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers

        labels = _make_labels(n_labels=5)
        df = extract_features(labels)
        compute_composite_score(df)
        assign_qc_tiers(df, (1.0, 2.0, 3.0))
        # Set two labels to curated
        df.loc[df.index[0], "qc_tier"] = "curated"
        df.loc[df.index[2], "qc_tier"] = "curated"
        csv_path = tmp_path / "curated_test.csv"
        save_patch_csv(df, csv_path)
        loaded = load_patch_csv(csv_path)
        curated_count = (loaded["qc_tier"] == "curated").sum()
        assert curated_count == 2

    def test_curated_survives_merge(self, tmp_path):
        """merge_patch_csvs preserves curated rows from individual patches."""
        from napari_epicycle.fusion_reconstruction.qc_features import assign_qc_tiers

        # Patch 1: has a curated label
        labels1 = _make_labels(n_labels=3)
        df1 = extract_features(labels1)
        compute_composite_score(df1)
        assign_qc_tiers(df1, (1.0, 2.0, 3.0))
        df1.loc[df1.index[0], "qc_tier"] = "curated"
        save_patch_csv(df1, tmp_path / "patch001_features.csv")

        # Patch 2: no curated labels
        labels2 = _make_labels(n_labels=3)
        df2 = extract_features(labels2)
        compute_composite_score(df2)
        assign_qc_tiers(df2, (1.0, 2.0, 3.0))
        save_patch_csv(df2, tmp_path / "patch002_features.csv")

        merged = merge_patch_csvs([
            tmp_path / "patch001_features.csv",
            tmp_path / "patch002_features.csv",
        ])
        curated_rows = merged[merged["qc_tier"] == "curated"]
        assert len(curated_rows) == 1
        assert curated_rows.iloc[0]["patch_name"] == "patch001"


# ---------------------------------------------------------------------------
# Phase 1: compute_channel_intensity + shared erosion helper
# ---------------------------------------------------------------------------


class TestExtractSingleChannel:
    """Tests for the extract_single_channel helper (moved from controller)."""

    def test_3d_passthrough(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_channel,
        )
        src = np.random.rand(10, 20, 20).astype(np.float32)
        result = extract_single_channel(src, (10, 20, 20))
        assert result is src

    def test_3d_shape_mismatch_returns_none(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_channel,
        )
        src = np.random.rand(10, 20, 20).astype(np.float32)
        result = extract_single_channel(src, (10, 30, 30))
        assert result is None

    def test_czyx_extracts_correct_channel(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_channel,
        )
        # CZYX: shape (3, 10, 20, 20)
        src = np.zeros((3, 10, 20, 20), dtype=np.float32)
        src[1] = 42.0  # channel 1
        # seg_channel=1 → array index 1 (0-based)
        result = extract_single_channel(src, (10, 20, 20), seg_channel=1)
        assert result is not None
        assert result.shape == (10, 20, 20)
        assert result.mean() == pytest.approx(42.0)

    def test_zcyx_extracts_correct_channel(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_single_channel,
        )
        # ZCYX: shape (10, 3, 20, 20)
        src = np.zeros((10, 3, 20, 20), dtype=np.float32)
        src[:, 0] = 99.0  # channel 0
        # seg_channel=0 (gray) → arr_idx=0
        result = extract_single_channel(src, (10, 20, 20), seg_channel=0)
        assert result is not None
        assert result.shape == (10, 20, 20)
        assert result.mean() == pytest.approx(99.0)

    def test_import_alias_still_works(self):
        """Verify Step 3 controller's import alias is functional."""
        from napari_epicycle.fusion_reconstruction.controller import (
            _extract_single_channel,
        )
        src = np.random.rand(10, 20, 20).astype(np.float32)
        result = _extract_single_channel(src, (10, 20, 20))
        assert result is src


class TestComputeBorderCells:
    """Tests for border cell detection."""

    def test_interior_cell_not_border(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_border_cells,
        )
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[5:15, 5:15, 5:15] = 1  # fully interior
        df = extract_features(labels)
        compute_border_cells(labels, df)
        assert df.iloc[0]["is_border_cell"] == 0

    def test_cell_touching_z0_is_border(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_border_cells,
        )
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[0:10, 5:15, 5:15] = 1  # touches z=0
        df = extract_features(labels)
        compute_border_cells(labels, df)
        assert df.iloc[0]["is_border_cell"] == 1

    def test_cell_touching_x_max_is_border(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_border_cells,
        )
        labels = np.zeros((20, 20, 20), dtype=np.int32)
        labels[5:15, 5:15, 15:20] = 1  # touches x=19
        df = extract_features(labels)
        compute_border_cells(labels, df)
        assert df.iloc[0]["is_border_cell"] == 1

    def test_mixed_border_and_interior(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_border_cells,
        )
        labels = np.zeros((20, 30, 30), dtype=np.int32)
        labels[0:5, 5:10, 5:10] = 1   # border (touches z=0)
        labels[8:14, 12:18, 12:18] = 2  # interior
        df = extract_features(labels)
        compute_border_cells(labels, df)
        border_vals = dict(zip(df["label"].values, df["is_border_cell"].values))
        assert border_vals[1] == 1
        assert border_vals[2] == 0

    def test_empty_volume(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_border_cells,
        )
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        df = extract_features(labels)
        compute_border_cells(labels, df)
        assert "is_border_cell" in df.columns


class TestComputeChannelIntensity:
    """Tests for per-channel intensity feature extraction."""

    _EXPECTED_SUFFIXES = (
        "_intensity_mean", "_intensity_median", "_intensity_std",
        "_intensity_min", "_intensity_max", "_intensity_sum",
        "_intensity_cv",
        "_boundary_intensity_mean", "_interior_intensity_mean",
        "_boundary_interior_ratio", "_z_intensity_variation",
    )

    def test_adds_11_columns_with_prefix(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_channel_intensity,
        )
        labels = _make_labels(n_labels=2)
        channel = np.random.RandomState(7).rand(*labels.shape).astype(np.float32)
        df = extract_features(labels)
        compute_channel_intensity(labels, df, channel, prefix="nuclei")
        for suffix in self._EXPECTED_SUFFIXES:
            col = f"nuclei{suffix}"
            assert col in df.columns, f"Missing column: {col}"

    def test_prefix_naming(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_channel_intensity,
        )
        labels = _make_labels(n_labels=1)
        channel = np.ones(labels.shape, dtype=np.float32)
        df = extract_features(labels)
        compute_channel_intensity(labels, df, channel, prefix="gfp")
        assert "gfp_intensity_mean" in df.columns
        assert "gfp_boundary_interior_ratio" in df.columns

    def test_cv_is_std_over_mean(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_channel_intensity,
        )
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[2:8, 2:8, 2:8] = 1
        # Uniform intensity → std=0, cv=0
        channel = np.ones((10, 10, 10), dtype=np.float32) * 5.0
        df = extract_features(labels)
        compute_channel_intensity(labels, df, channel, prefix="ch0")
        assert df.iloc[0]["ch0_intensity_cv"] == pytest.approx(0.0, abs=1e-6)

    def test_values_are_finite(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_channel_intensity,
        )
        labels = _make_labels(n_labels=3)
        channel = np.random.RandomState(42).rand(*labels.shape).astype(np.float32)
        df = extract_features(labels)
        compute_channel_intensity(labels, df, channel, prefix="test")
        for suffix in self._EXPECTED_SUFFIXES:
            col = f"test{suffix}"
            assert np.all(np.isfinite(df[col].values)), f"Non-finite in: {col}"

    def test_empty_df_returns_empty_columns(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_channel_intensity,
        )
        labels = np.zeros((5, 5, 5), dtype=np.int32)  # no labels
        channel = np.ones((5, 5, 5), dtype=np.float32)
        df = extract_features(labels)
        compute_channel_intensity(labels, df, channel, prefix="ch0")
        for suffix in self._EXPECTED_SUFFIXES:
            assert f"ch0{suffix}" in df.columns

    def test_tiny_label_no_crash(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_channel_intensity,
        )
        labels = np.zeros((8, 8, 8), dtype=np.int32)
        labels[3:5, 3:5, 3:5] = 1  # 2x2x2 — erosion eliminates interior
        channel = np.ones((8, 8, 8), dtype=np.float32) * 2.0
        df = extract_features(labels)
        compute_channel_intensity(labels, df, channel, prefix="tiny")
        assert np.all(np.isfinite(df["tiny_intensity_mean"].values))

    def test_multiple_channels_independent(self):
        """Two channels with different data produce different features."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_channel_intensity,
        )
        labels = _make_labels(n_labels=2)
        ch_bright = np.ones(labels.shape, dtype=np.float32) * 100.0
        ch_dim = np.ones(labels.shape, dtype=np.float32) * 1.0
        df = extract_features(labels)
        compute_channel_intensity(labels, df, ch_bright, prefix="bright")
        compute_channel_intensity(labels, df, ch_dim, prefix="dim")
        assert df.iloc[0]["bright_intensity_mean"] > df.iloc[0]["dim_intensity_mean"]


class TestRefactoredBoundaryInterior:
    """Regression guard: refactored compute_boundary_interior_features
    must produce identical results to the original implementation."""

    def test_bright_boundary_still_high_ratio(self):
        """Same test as TestBoundaryInteriorFeatures but verifies refactor."""
        labels = np.zeros((16, 16, 16), dtype=np.int32)
        labels[2:14, 2:14, 2:14] = 1
        intensity = np.zeros((16, 16, 16), dtype=np.float32)
        intensity[2:14, 2:14, 2:14] = 1.0
        intensity[2, 2:14, 2:14] = 10.0
        intensity[13, 2:14, 2:14] = 10.0
        intensity[2:14, 2, 2:14] = 10.0
        intensity[2:14, 13, 2:14] = 10.0
        intensity[2:14, 2:14, 2] = 10.0
        intensity[2:14, 2:14, 13] = 10.0
        df = extract_features(labels, intensity_image=intensity)
        compute_boundary_interior_features(labels, intensity, df)
        assert df.iloc[0]["boundary_interior_ratio"] > 1.5

    def test_z_variation_still_works(self):
        labels = np.zeros((20, 10, 10), dtype=np.int32)
        labels[2:18, 2:8, 2:8] = 1
        intensity = np.zeros((20, 10, 10), dtype=np.float32)
        for z in range(2, 18):
            intensity[z, 2:8, 2:8] = float(z)
        df = extract_features(labels, intensity_image=intensity)
        compute_boundary_interior_features(labels, intensity, df)
        assert df.iloc[0]["z_intensity_variation"] > 0

    def test_all_four_columns_present(self):
        labels = _make_labels(n_labels=3)
        intensity = np.random.RandomState(99).rand(*labels.shape).astype(np.float32)
        df = extract_features(labels, intensity_image=intensity)
        compute_boundary_interior_features(labels, intensity, df)
        for col in ("boundary_intensity_mean", "interior_intensity_mean",
                     "boundary_interior_ratio", "z_intensity_variation"):
            assert col in df.columns
            assert np.all(np.isfinite(df[col].values))

"""Pure-function tests for the lifted Wave-1 QC features + cKDTree crowding.

No GPU / Cellpose. Builds known synthetic shapes (cube / rod / hollow box) and
checks the geometric invariants, the deliberate divergences from the monorepo
source (no ``compactness`` / ``feature_state`` columns), and the new
``compute_centroid_spacing``. A final parity spot-check compares the engine
outputs to the monorepo ``qc_features`` value-for-value on a shared volume
(skipped if the monorepo is not importable, so the engine suite stays standalone).
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _three_cell_volume():
    """A cube (1), a rod (2), and an L-bend slab (3) in one (20,30,30) volume."""
    lab = np.zeros((20, 30, 30), dtype=np.int32)
    lab[5:11, 5:11, 5:11] = 1            # 6x6x6 cube
    lab[5:9, 5:7, 15:25] = 2             # rod (long in X)
    lab[12:16, 18:26, 18:22] = 3         # slab
    lab[12:14, 18:20, 18:26] = 3         # L extension
    return lab


def _full_pipeline(lab, spacing=None):
    df = qf.extract_features(lab, spacing=spacing)
    qf.compute_surface_features(lab, df, spacing=spacing)
    qf.compute_discrete_compactness(lab, df)
    qf.compute_skeleton_features(lab, df)
    qf.compute_cross_section_features(lab, df)
    qf.compute_border_cells(lab, df)
    qf.compute_centroid_spacing(df)
    return df


# --------------------------------------------------------------------------- #
# extract_features
# --------------------------------------------------------------------------- #

def test_extract_features_basic_columns_and_dropped():
    lab = _three_cell_volume()
    df = qf.extract_features(lab)
    assert len(df) == 3
    # core columns present
    for c in ("label", "area", "extent", "sphericity", "aspect_ratio",
              "bbox_volume", "axis_major_length", "ellipsoid_solidity"):
        assert c in df.columns, c
    # deliberate divergences from the monorepo source
    assert "compactness" not in df.columns
    assert "feature_state" not in df.columns


def test_extract_features_empty_volume_returns_schema():
    df = qf.extract_features(np.zeros((4, 4, 4), dtype=np.int32))
    assert len(df) == 0
    assert "compactness" not in df.columns
    assert list(df.columns) == qf.FEATURE_COLUMNS


def test_extract_features_rejects_non_3d():
    with pytest.raises(ValueError):
        qf.extract_features(np.zeros((4, 4), dtype=np.int32))


def test_cube_geometry():
    lab = np.zeros((16, 16, 16), dtype=np.int32)
    lab[4:12, 4:12, 4:12] = 1            # 8x8x8 solid cube
    df = qf.extract_features(lab)
    r = df.iloc[0]
    assert r["area"] == 512                 # 8^3 voxels
    assert r["extent"] == pytest.approx(1.0)         # fills its bbox
    assert r["aspect_ratio"] == pytest.approx(1.0)   # cubic
    assert r["z_span"] == pytest.approx(8.0)


def test_spacing_scales_volume_physically():
    lab = np.zeros((16, 16, 16), dtype=np.int32)
    lab[4:12, 4:12, 4:12] = 1            # 512 voxels
    df = qf.extract_features(lab, spacing=(2.0, 0.5, 0.5))
    # area in physical units = voxels * (2.0 * 0.5 * 0.5) = 512 * 0.5
    assert df.iloc[0]["area"] == pytest.approx(256.0)


def test_rod_is_elongated():
    lab = np.zeros((10, 10, 30), dtype=np.int32)
    lab[4:6, 4:6, 2:28] = 1              # long thin rod in X
    df = qf.extract_features(lab)
    assert df.iloc[0]["aspect_ratio"] > 5.0
    assert df.iloc[0]["xy_elongation"] > 5.0


# --------------------------------------------------------------------------- #
# surface / discrete-compactness / skeleton / cross-section / border
# --------------------------------------------------------------------------- #

def test_surface_features_bounded_sphericity():
    lab = np.zeros((16, 16, 16), dtype=np.int32)
    lab[4:12, 4:12, 4:12] = 1
    df = qf.extract_features(lab)
    qf.compute_surface_features(lab, df)
    r = df.iloc[0]
    assert r["surface_area"] > 0
    assert 0.0 < r["true_sphericity"] <= 1.05      # ~1 for a compact blob
    assert r["roughness"] > 0


def test_discrete_compactness_cube_near_one():
    lab = np.zeros((16, 16, 16), dtype=np.int32)
    lab[3:13, 3:13, 3:13] = 1           # 10^3 cube
    df = qf.extract_features(lab)
    qf.compute_discrete_compactness(lab, df)
    cd = df.iloc[0]["discrete_compactness"]
    assert 0.95 <= cd <= 1.0            # a cube is the discrete-compactness max


def test_discrete_compactness_bounded_zero_one():
    df = _full_pipeline(_three_cell_volume())
    cd = df["discrete_compactness"].values
    assert np.all(cd >= 0.0) and np.all(cd <= 1.0 + 1e-9)


def test_skeleton_length_nonzero_for_rod():
    lab = np.zeros((10, 10, 40), dtype=np.int32)
    lab[4:7, 4:7, 2:38] = 1
    df = qf.extract_features(lab)
    qf.compute_skeleton_features(lab, df)
    assert df.iloc[0]["skeleton_length"] > 0


def test_cross_section_columns_present():
    df = qf.extract_features(_three_cell_volume())
    qf.compute_cross_section_features(_three_cell_volume(), df)
    for c in ("apical_area", "basal_area", "apical_basal_area_ratio",
              "max_cross_section_z_relative", "cross_section_taper",
              "cross_section_bulge"):
        assert c in df.columns


def test_border_cell_flag():
    lab = np.zeros((12, 12, 12), dtype=np.int32)
    lab[0:3, 0:3, 0:3] = 1              # touches Z=0, Y=0, X=0 faces
    lab[5:8, 5:8, 5:8] = 2              # interior
    df = qf.extract_features(lab)
    qf.compute_border_cells(lab, df)
    by_label = dict(zip(df["label"], df["is_border_cell"]))
    assert by_label[1] == 1
    assert by_label[2] == 0


# --------------------------------------------------------------------------- #
# compute_centroid_spacing (the new Wave-1 crowding feature)
# --------------------------------------------------------------------------- #

def test_centroid_spacing_two_cells_distance():
    # Two centroids 10 voxels apart in X.
    df = pd.DataFrame({
        "label": [1, 2],
        "centroid-0": [0.0, 0.0],
        "centroid-1": [0.0, 0.0],
        "centroid-2": [0.0, 10.0],
    })
    qf.compute_centroid_spacing(df, k_neighbors=6)
    assert df["nn_centroid_distance"].tolist() == pytest.approx([10.0, 10.0])
    # knn_mean with only 1 neighbour available == the nn distance
    assert df["knn_mean_distance"].tolist() == pytest.approx([10.0, 10.0])


def test_centroid_spacing_single_cell_is_nan():
    df = pd.DataFrame({"label": [1], "centroid-0": [1.0], "centroid-1": [2.0],
                       "centroid-2": [3.0]})
    qf.compute_centroid_spacing(df)
    assert np.isnan(df.iloc[0]["nn_centroid_distance"])
    assert df.iloc[0]["neighbor_count_in_radius"] == 0.0


def test_centroid_spacing_empty():
    df = pd.DataFrame(columns=["label", "centroid-0", "centroid-1", "centroid-2"])
    qf.compute_centroid_spacing(df)
    for c in ("nn_centroid_distance", "knn_mean_distance",
              "neighbor_count_in_radius", "local_density"):
        assert c in df.columns
    assert len(df) == 0


def test_centroid_spacing_missing_centroids_nan():
    df = pd.DataFrame({"label": [1, 2], "area": [10.0, 20.0]})
    qf.compute_centroid_spacing(df)
    assert df["nn_centroid_distance"].isna().all()


def test_centroid_spacing_radius_count():
    # 3 cells in a line at x=0,5,100. With density_radius=10 the middle cell sees
    # both? no — only the one at distance 5; the far one at 95 is outside.
    df = pd.DataFrame({
        "label": [1, 2, 3],
        "centroid-0": [0.0, 0.0, 0.0],
        "centroid-1": [0.0, 0.0, 0.0],
        "centroid-2": [0.0, 5.0, 100.0],
    })
    qf.compute_centroid_spacing(df, density_radius=10.0)
    counts = dict(zip(df["label"], df["neighbor_count_in_radius"]))
    assert counts[1] == 1.0     # sees cell 2 (dist 5)
    assert counts[2] == 1.0     # sees cell 1 (dist 5); cell 3 is 95 away
    assert counts[3] == 0.0     # alone within r=10


# --------------------------------------------------------------------------- #
# monorepo parity spot-check (the guardrail) — skipped if monorepo absent
# --------------------------------------------------------------------------- #

def _monorepo_qc():
    """Load the plugin's ``qc_features.py`` BY FILE PATH (it is numpy/pandas/scipy
    only), so importing it never pulls napari / Qt (31 s, and a Qt dependency in
    a napari-free suite). Skips when the plugin source is absent (standalone)."""
    import importlib.util
    import sys
    from pathlib import Path
    path = (Path(__file__).resolve().parents[2] / "napari_epicycle"
            / "fusion_reconstruction" / "qc_features.py")
    if not path.is_file():
        pytest.skip("plugin qc_features.py not present — engine suite stays standalone")
    name = "_plugin_qc_features_by_path"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("spacing", [None, (2.0, 0.5, 0.5)])
def test_parity_extract_features(spacing):
    mono = _monorepo_qc()
    lab = _three_cell_volume()
    eng = qf.extract_features(lab, spacing=spacing)
    ref = mono.extract_features(lab, spacing=spacing)
    # Compare every shared column except the engine's deliberate drops.
    shared = (set(eng.columns) & set(ref.columns)) - {"compactness", "feature_state"}
    if spacing is not None:
        # Centroid units differ by design: the engine emits PHYSICAL units when a
        # spacing is given, the plugin keeps voxel units (both are internally
        # consistent). Everything else must match.
        shared = {c for c in shared if not c.startswith("centroid-")}
    assert "area" in shared and "ellipsoid_solidity" in shared
    for c in sorted(shared):
        if eng[c].dtype.kind in "fiu" and ref[c].dtype.kind in "fiu":
            np.testing.assert_allclose(
                eng[c].values, ref[c].values, rtol=1e-9, atol=1e-9,
                err_msg=f"parity mismatch in extract_features['{c}']",
            )


def test_parity_surface_and_skeleton_and_discrete():
    mono = _monorepo_qc()
    lab = _three_cell_volume()

    eng = qf.extract_features(lab)
    ref = mono.extract_features(lab)
    qf.compute_surface_features(lab, eng)
    mono.compute_surface_features(lab, ref)
    qf.compute_discrete_compactness(lab, eng)
    mono.compute_discrete_compactness(lab, ref)
    qf.compute_skeleton_features(lab, eng)
    mono.compute_skeleton_features(lab, ref)
    qf.compute_cross_section_features(lab, eng)
    mono.compute_cross_section_features(lab, ref)

    for c in ("surface_area", "true_sphericity", "roughness", "solidity",
              "discrete_compactness", "skeleton_branches", "skeleton_nodes",
              "skeleton_length", "apical_basal_area_ratio", "cross_section_taper"):
        np.testing.assert_allclose(
            eng[c].values, ref[c].values, rtol=1e-9, atol=1e-9,
            err_msg=f"parity mismatch in '{c}'",
        )

"""Tests for 3D mesh export (marching cubes → OBJ)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction.qc_features import (
    export_all_cell_meshes,
    export_cell_mesh_obj,
)


def _make_sphere_volume(label_id=1, radius=8, shape=(30, 30, 30)):
    """Create a volume with a single spherical label."""
    vol = np.zeros(shape, dtype=np.int32)
    center = np.array(shape) // 2
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        (zz - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (xx - center[2]) ** 2
    )
    vol[dist <= radius] = label_id
    return vol


def _make_multi_label_volume():
    """Create a volume with 3 non-overlapping spheres."""
    vol = np.zeros((40, 40, 80), dtype=np.int32)
    for lid, cx in [(1, 15), (2, 40), (3, 65)]:
        zz, yy, xx = np.ogrid[:40, :40, :80]
        dist = np.sqrt((zz - 20) ** 2 + (yy - 20) ** 2 + (xx - cx) ** 2)
        vol[dist <= 8] = lid
    return vol


class TestExportCellMeshObj:
    def test_basic_sphere(self, tmp_path):
        vol = _make_sphere_volume()
        obj_path = tmp_path / "cell_1.obj"
        result = export_cell_mesh_obj(vol, 1, obj_path)

        assert result is not None
        assert result["n_vertices"] > 10
        assert result["n_faces"] > 10
        assert obj_path.exists()

    def test_obj_file_format(self, tmp_path):
        vol = _make_sphere_volume()
        obj_path = tmp_path / "cell_1.obj"
        export_cell_mesh_obj(vol, 1, obj_path)

        content = obj_path.read_text()
        lines = content.strip().split("\n")
        # Should have comment lines, vertex lines, face lines
        v_lines = [l for l in lines if l.startswith("v ")]
        f_lines = [l for l in lines if l.startswith("f ")]
        assert len(v_lines) > 0
        assert len(f_lines) > 0
        # Vertex format: v x y z
        parts = v_lines[0].split()
        assert len(parts) == 4
        float(parts[1])  # should not raise
        # Face format: f i j k (1-indexed)
        parts = f_lines[0].split()
        assert len(parts) == 4
        assert int(parts[1]) >= 1  # 1-indexed

    def test_with_spacing(self, tmp_path):
        vol = _make_sphere_volume()
        obj_no_sp = tmp_path / "no_spacing.obj"
        obj_sp = tmp_path / "with_spacing.obj"

        r1 = export_cell_mesh_obj(vol, 1, obj_no_sp)
        r2 = export_cell_mesh_obj(vol, 1, obj_sp, spacing=(2.0, 0.5, 0.5))

        # Centroid should differ due to spacing
        assert r1["centroid_z"] != r2["centroid_z"] or r1["centroid_y"] != r2["centroid_y"]

    def test_missing_label_returns_none(self, tmp_path):
        vol = _make_sphere_volume(label_id=1)
        result = export_cell_mesh_obj(vol, 99, tmp_path / "cell_99.obj")
        assert result is None

    def test_tiny_label_returns_none(self, tmp_path):
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        vol[5, 5, 5] = 1  # single voxel
        result = export_cell_mesh_obj(vol, 1, tmp_path / "cell_1.obj")
        assert result is None

    def test_creates_parent_dirs(self, tmp_path):
        vol = _make_sphere_volume()
        deep_path = tmp_path / "a" / "b" / "c" / "cell_1.obj"
        result = export_cell_mesh_obj(vol, 1, deep_path)
        assert result is not None
        assert deep_path.exists()


class TestExportAllCellMeshes:
    def test_multi_label(self, tmp_path):
        vol = _make_multi_label_volume()
        manifest = export_all_cell_meshes(
            vol, tmp_path / "meshes", "test_stack",
        )

        assert len(manifest) == 3
        assert set(manifest["label"]) == {1, 2, 3}
        assert all(manifest["stack_name"] == "test_stack")
        assert all(Path(p).exists() for p in manifest["mesh_path"])

    def test_manifest_csv_written(self, tmp_path):
        vol = _make_multi_label_volume()
        export_all_cell_meshes(vol, tmp_path / "meshes", "s1")

        csv_path = tmp_path / "meshes" / "mesh_manifest.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "label" in df.columns
        assert "stack_name" in df.columns
        assert "mesh_path" in df.columns
        assert "n_vertices" in df.columns
        assert "n_faces" in df.columns
        assert "centroid_z" in df.columns

    def test_specific_label_ids(self, tmp_path):
        vol = _make_multi_label_volume()
        manifest = export_all_cell_meshes(
            vol, tmp_path / "meshes", "s1", label_ids=[1, 3],
        )
        assert len(manifest) == 2
        assert set(manifest["label"]) == {1, 3}

    def test_with_spacing(self, tmp_path):
        vol = _make_multi_label_volume()
        manifest = export_all_cell_meshes(
            vol, tmp_path / "meshes", "s1", spacing=(2.0, 0.5, 0.5),
        )
        assert len(manifest) == 3
        # Centroids should reflect spacing
        assert all(manifest["centroid_z"] > 0)

    def test_progress_callback(self, tmp_path):
        vol = _make_multi_label_volume()
        progress = []
        export_all_cell_meshes(
            vol, tmp_path / "meshes", "s1",
            progress_cb=lambda cur, tot: progress.append((cur, tot)),
        )
        assert len(progress) == 3
        assert progress[-1] == (3, 3)

    def test_empty_volume(self, tmp_path):
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        manifest = export_all_cell_meshes(
            vol, tmp_path / "meshes", "s1",
        )
        assert len(manifest) == 0

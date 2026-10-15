"""Integration tests for Step 1 end-to-end workflows.

Tests the logic chain in realistic multi-step sequences.
No Qt/napari — tests the pure logic chain only.
"""
import json
import os
import tempfile

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation.logic import (
    extract_patches_from_shapes,
    find_next_patch_index,
    scan_tiff_folder,
    split_channels,
    transpose_to_czyx,
)


def _box(y0, x0, y1, x1):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------

class TestEndToEndExtraction:
    """Full workflow: scan → load → extract → export."""

    def test_full_workflow(self):
        """Simulate: scan folder, load file, extract patches, write to workspace."""
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as ws:
            # 1. Create source TIFF
            shape = (2, 20, 64, 64)
            data = np.random.randint(0, 255, shape, dtype=np.uint8)
            tifffile.imwrite(os.path.join(src, "sample_001.tiff"), data)

            # 2. Scan folder
            files = scan_tiff_folder(src)
            assert len(files) == 1
            assert files[0]["shape"] == shape

            # 3. Load and split
            loaded = tifffile.imread(files[0]["path"])
            channels = split_channels(loaded, "kidney_ant")
            assert len(channels) == 2

            # 4. Extract patches with display name prefix
            boxes = [_box(0, 0, 32, 32), _box(32, 32, 64, 64)]
            name_prefix = "kidney_ant_patch"
            start_index = find_next_patch_index(ws, name_prefix)
            patches = extract_patches_from_shapes(
                loaded, boxes,
                name_prefix=name_prefix, start_index=start_index,
            )
            assert "kidney_ant_patch_1" in patches
            assert "kidney_ant_patch_2" in patches

            # 5. Export to workspace via the real persistence helper.
            from napari_epicycle.data_preparation._export import persist_patch
            from napari_epicycle.data_preparation._paths import step1_patches_dir

            for pname, pdata in patches.items():
                persist_patch(
                    ws, pname, pdata,
                    channel_names=["kidney_ant_ch0", "kidney_ant_ch1"],
                    original_shape=pdata.shape,
                    cropped_shape=pdata.shape,
                    selected_z_range=[0, pdata.shape[1]],
                    bounding_box="not stored yet",
                    source_path=src, source_display_name="kidney_ant",
                    timestamp="t",
                )

            # Verify files exist in the canonical layout.
            pdir = step1_patches_dir(ws)
            assert (pdir / "kidney_ant_patch_1.tiff").exists()
            assert (pdir / "kidney_ant_patch_1_metadata.json").exists()

    def test_cumulative_extraction_across_sessions(self):
        """Extract + persist 2 patches (new layout). Then extract 2 more —
        numbering continues from the registry (3, 4), no reset."""
        from napari_epicycle.data_preparation._export import persist_patch

        with tempfile.TemporaryDirectory() as ws:
            img = np.zeros((2, 4, 32, 32), dtype=np.uint8)
            prefix = "sample_patch"

            # Session 1: extract + persist 2 patches into artifacts/step1/patches.
            patches_s1 = extract_patches_from_shapes(
                img, [_box(0, 0, 16, 16), _box(16, 0, 32, 16)],
                name_prefix=prefix, start_index=1,
            )
            assert set(patches_s1.keys()) == {"sample_patch_1", "sample_patch_2"}
            for name, pdata in patches_s1.items():
                persist_patch(
                    ws, name, pdata, channel_names=["a", "b"],
                    original_shape=pdata.shape, cropped_shape=pdata.shape,
                    selected_z_range=[0, pdata.shape[1]], bounding_box="x",
                    source_path="", timestamp="t",
                )

            # Session 2: index continues from the registry — 3 and 4.
            start = find_next_patch_index(ws, prefix)
            assert start == 3
            patches_s2 = extract_patches_from_shapes(
                img, [_box(0, 16, 16, 32), _box(16, 16, 32, 32)],
                name_prefix=prefix, start_index=start,
            )
            assert set(patches_s2.keys()) == {"sample_patch_3", "sample_patch_4"}

    def test_multichannel_roundtrip(self):
        """persist_patch writes a multichannel patch Step 2 can read back (real path)."""
        from napari_epicycle.data_preparation._export import persist_patch

        with tempfile.TemporaryDirectory() as ws:
            data = np.random.randint(0, 255, (3, 10, 32, 32), dtype=np.uint8)
            res = persist_patch(
                ws, "test_patch_1", data,
                channel_names=["nuclei", "membrane", "marker"],
                original_shape=data.shape, cropped_shape=data.shape,
                selected_z_range=[0, 10],
                bounding_box=[[0, 0], [0, 32], [32, 32], [32, 0]],
                source_path="D:/s", timestamp="t",
            )
            # Step 2 read-back from the canonical location.
            loaded = tifffile.imread(res["tiff_path"])
            assert loaded.ndim == 4 and loaded.shape[0] == 3
            loaded_meta = json.loads(open(res["sidecar_path"]).read())
            assert loaded_meta["patch_name"] == "test_patch_1"
            assert len(loaded_meta["channel_names"]) == loaded.shape[0]


# ---------------------------------------------------------------------------
# Downstream contract
# ---------------------------------------------------------------------------

class TestDownstreamContract:
    """Verify exported artifacts match what Step 2 expects."""

    def test_tiff_is_4d_czyx(self):
        with tempfile.TemporaryDirectory() as d:
            data = np.zeros((2, 5, 16, 16), dtype=np.uint8)
            path = os.path.join(d, "p1.tiff")
            tifffile.imwrite(path, data, metadata={"axes": "CZYX"})
            loaded = tifffile.imread(path)
            assert loaded.ndim == 4

    def test_transpose_workflow_produces_czyx(self):
        """ZCYX data → transpose → CZYX → extract → verify 4D."""
        zcyx = np.random.rand(10, 2, 32, 32).astype(np.float32)
        czyx = transpose_to_czyx(zcyx)
        assert czyx.shape == (2, 10, 32, 32)
        patches = extract_patches_from_shapes(czyx, [_box(0, 0, 16, 16)])
        patch = patches["patch_1"]
        assert patch.ndim == 4
        assert patch.shape[0] == 2  # channels first

"""Tests for Step 1 data preparation logic.

Pure algorithm tests — no Qt, no napari required.
"""
import os
import tempfile

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation.logic import (
    detect_sample_range_all_channels,
    detect_sample_z_range_by_intensity,
    extract_patches_from_shapes,
    find_next_patch_index,
    scan_tiff_folder,
    split_channels,
    transpose_to_czyx,
)


# ---------------------------------------------------------------------------
# extract_patches_from_shapes
# ---------------------------------------------------------------------------

class TestExtractPatches:
    """Tests for the core patch extraction function."""

    def _box(self, y0, x0, y1, x1):
        """Helper: create a (4, 2) rectangle in (Y, X) corner order."""
        return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)

    def test_basic_extraction_4d(self):
        img = np.random.rand(2, 10, 64, 64)
        box = self._box(16, 16, 48, 48)
        patches = extract_patches_from_shapes(img, [box])
        assert "patch_1" in patches
        assert patches["patch_1"].shape == (2, 10, 32, 32)

    def test_basic_extraction_3d(self):
        img = np.random.rand(10, 64, 64)
        box = self._box(0, 0, 32, 32)
        patches = extract_patches_from_shapes(img, [box])
        assert "patch_1" in patches
        assert patches["patch_1"].shape == (10, 32, 32)

    def test_boundary_clipping_pads_zeros(self):
        img = np.ones((2, 5, 32, 32))
        box = self._box(16, 16, 48, 48)  # extends past 32x32
        patches = extract_patches_from_shapes(img, [box])
        patch = patches["patch_1"]
        assert patch.shape == (2, 5, 32, 32)
        # Bottom-right corner should be zero-padded
        assert np.all(patch[:, :, 16:, :] == 0)

    def test_custom_naming_prefix(self):
        img = np.zeros((1, 2, 8, 8))
        boxes = [self._box(0, 0, 4, 4), self._box(4, 4, 8, 8)]
        patches = extract_patches_from_shapes(img, boxes, name_prefix="kidney")
        assert "kidney_1" in patches
        assert "kidney_2" in patches
        assert "patch_1" not in patches

    def test_custom_start_index(self):
        img = np.zeros((1, 2, 8, 8))
        boxes = [self._box(0, 0, 4, 4)]
        patches = extract_patches_from_shapes(img, boxes, start_index=5)
        assert "patch_5" in patches
        assert "patch_1" not in patches

    def test_default_naming_backward_compat(self):
        img = np.zeros((2, 4, 16, 16))
        boxes = [self._box(0, 0, 8, 8), self._box(8, 8, 16, 16)]
        patches = extract_patches_from_shapes(img, boxes)
        assert list(patches.keys()) == ["patch_1", "patch_2"]

    def test_multiple_boxes(self):
        img = np.zeros((1, 2, 32, 32))
        boxes = [self._box(0, 0, 8, 8), self._box(8, 0, 16, 8), self._box(0, 8, 8, 16)]
        patches = extract_patches_from_shapes(img, boxes)
        assert len(patches) == 3

    def test_empty_bounding_boxes(self):
        img = np.zeros((2, 4, 16, 16))
        patches = extract_patches_from_shapes(img, [])
        assert patches == {}


class TestExtractPatchesZRange:
    """Per-box Z crop (box tool's editable-Z, spec item B).

    The load-bearing invariant: a full-Z box (or the default ``z_ranges=None``)
    extracts BYTE-IDENTICALLY to the pre-feature path; a shrunk-Z box extracts
    exactly its Z slab.
    """

    def _box(self, y0, x0, y1, x1):
        return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)

    def test_z_ranges_none_is_byte_identical(self):
        img = np.arange(2 * 10 * 32 * 32).reshape(2, 10, 32, 32).astype(np.int32)
        box = self._box(4, 4, 20, 20)
        ref = extract_patches_from_shapes(img, [box])
        got = extract_patches_from_shapes(img, [box], z_ranges=None)
        assert np.array_equal(got["patch_1"], ref["patch_1"])

    def test_full_z_range_is_byte_identical(self):
        img = np.arange(2 * 10 * 32 * 32).reshape(2, 10, 32, 32).astype(np.int32)
        box = self._box(4, 4, 20, 20)
        ref = extract_patches_from_shapes(img, [box])
        got = extract_patches_from_shapes(img, [box], z_ranges=[[0, 10]])
        assert got["patch_1"].shape == ref["patch_1"].shape
        assert np.array_equal(got["patch_1"], ref["patch_1"])

    def test_z_range_crops_exact_slab_4d(self):
        img = np.arange(2 * 10 * 32 * 32).reshape(2, 10, 32, 32).astype(np.int32)
        box = self._box(4, 4, 20, 20)
        got = extract_patches_from_shapes(img, [box], z_ranges=[[3, 7]])
        patch = got["patch_1"]
        assert patch.shape == (2, 4, 16, 16)          # Z == 7-3 == 4
        assert np.array_equal(patch, img[:, 3:7, 4:20, 4:20])

    def test_z_range_crops_exact_slab_3d(self):
        img = np.arange(10 * 32 * 32).reshape(10, 32, 32).astype(np.int32)
        box = self._box(0, 0, 16, 16)
        got = extract_patches_from_shapes(img, [box], z_ranges=[[2, 5]])
        patch = got["patch_1"]
        assert patch.shape == (3, 16, 16)             # Z == 5-2 == 3
        assert np.array_equal(patch, img[2:5, 0:16, 0:16])

    def test_none_entry_keeps_full_z(self):
        img = np.arange(1 * 8 * 16 * 16).reshape(1, 8, 16, 16).astype(np.int32)
        box = self._box(0, 0, 8, 8)
        got = extract_patches_from_shapes(img, [box], z_ranges=[None])
        assert got["patch_1"].shape == (1, 8, 8, 8)   # full Z

    def test_per_box_z_ranges_independent(self):
        img = np.arange(1 * 12 * 24 * 24).reshape(1, 12, 24, 24).astype(np.int32)
        boxes = [self._box(0, 0, 8, 8), self._box(8, 8, 16, 16)]
        got = extract_patches_from_shapes(img, boxes, z_ranges=[[0, 12], [4, 9]])
        assert got["patch_1"].shape == (1, 12, 8, 8)  # full Z
        assert got["patch_2"].shape == (1, 5, 8, 8)   # slab 4..9
        assert np.array_equal(got["patch_2"], img[:, 4:9, 8:16, 8:16])

    def test_z_range_clamped_to_volume(self):
        img = np.arange(1 * 6 * 16 * 16).reshape(1, 6, 16, 16).astype(np.int32)
        box = self._box(0, 0, 8, 8)
        # Over-range end clamps to Z, negative start clamps to 0.
        got = extract_patches_from_shapes(img, [box], z_ranges=[[-3, 99]])
        assert got["patch_1"].shape == (1, 6, 8, 8)


# ---------------------------------------------------------------------------
# scan_tiff_folder
# ---------------------------------------------------------------------------

class TestScanTiffFolder:
    """Tests for TIFF folder scanning with header-only reads."""

    def test_discovers_tiff_files(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.tiff", "b.tif", "c.tiff"]:
                tifffile.imwrite(os.path.join(d, name), np.zeros((2, 10, 32, 32)))
            result = scan_tiff_folder(d)
            assert len(result) == 3
            assert all("path" in r and "filename" in r and "stem" in r and "shape" in r for r in result)

    def test_ignores_non_tiff(self):
        with tempfile.TemporaryDirectory() as d:
            tifffile.imwrite(os.path.join(d, "good.tiff"), np.zeros((4, 4, 4)))
            with open(os.path.join(d, "bad.png"), "w") as f:
                f.write("not a tiff")
            with open(os.path.join(d, "data.json"), "w") as f:
                f.write("{}")
            result = scan_tiff_folder(d)
            assert len(result) == 1
            assert result[0]["filename"] == "good.tiff"

    def test_reads_shapes_from_headers(self):
        with tempfile.TemporaryDirectory() as d:
            shape = (2, 20, 64, 64)
            tifffile.imwrite(os.path.join(d, "sample.tiff"), np.zeros(shape))
            result = scan_tiff_folder(d)
            assert result[0]["shape"] == shape

    def test_nbytes_no_int32_overflow_on_huge_shape(self, monkeypatch):
        """Regression: a >2^31-voxel header must yield the TRUE decompressed
        nbytes.  ``np.prod`` accumulates in the platform int (int32 on Windows),
        so a ~18e9-voxel volume overflowed ('overflow encountered in scalar
        multiply') to a wrapped/garbage nbytes that then mis-drove
        ``plan_load_route``'s >RAM decision.  ``math.prod`` (arbitrary precision)
        is used instead.  Mocked header (can't write an 18e9-voxel TIFF)."""
        import math

        from napari_epicycle.data_preparation import logic

        big_shape = (2, 20000, 1024, 1024)   # 41.9e9 voxels >> 2**31
        itemsize = 2

        class _FakeSeries:
            shape = big_shape
            dtype = np.dtype("uint16")
            axes = "CZYX"

        class _FakePage:
            compression = "NONE"

        class _FakeTiff:
            series = [_FakeSeries()]
            pages = [_FakePage()]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "huge.tif"), "wb").close()
            monkeypatch.setattr(
                logic.tifffile, "TiffFile", lambda *a, **k: _FakeTiff())
            result = scan_tiff_folder(d)

        assert len(result) == 1
        entry = result[0]
        assert entry["shape"] == big_shape
        assert entry["nbytes"] == math.prod(big_shape) * itemsize
        assert entry["nbytes"] > 2 ** 31     # wrong (wrapped) if int32-overflowed

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as d:
            result = scan_tiff_folder(d)
            assert result == []

    def test_sorts_by_filename(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["z_last.tiff", "a_first.tiff", "m_middle.tiff"]:
                tifffile.imwrite(os.path.join(d, name), np.zeros((4, 4, 4)))
            result = scan_tiff_folder(d)
            names = [r["filename"] for r in result]
            assert names == ["a_first.tiff", "m_middle.tiff", "z_last.tiff"]

    def test_nonexistent_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            scan_tiff_folder("/nonexistent/path/that/does/not/exist")


# ---------------------------------------------------------------------------
# transpose_to_czyx
# ---------------------------------------------------------------------------

class TestTransposeToCzyx:
    """Tests for axis transposition."""

    def test_4d_swap(self):
        data = np.zeros((10, 2, 64, 64))  # ZCYX
        result = transpose_to_czyx(data)
        assert result.shape == (2, 10, 64, 64)

    def test_3d_swap(self):
        data = np.zeros((10, 64, 64))
        result = transpose_to_czyx(data)
        assert result.shape == (64, 10, 64)

    def test_2d_passthrough(self):
        data = np.zeros((64, 64))
        result = transpose_to_czyx(data)
        assert result.shape == (64, 64)

    def test_preserves_data(self):
        data = np.arange(24).reshape(2, 3, 4)
        result = transpose_to_czyx(data)
        assert result[0, 0, 0] == data[0, 0, 0]
        assert result[1, 0, 0] == data[0, 1, 0]


# ---------------------------------------------------------------------------
# split_channels
# ---------------------------------------------------------------------------

class TestSplitChannels:
    """Tests for channel splitting."""

    def test_4d_splits(self):
        data = np.zeros((3, 10, 64, 64))
        result = split_channels(data, "sample")
        assert len(result) == 3
        assert result[0][0] == "sample_ch0"
        assert result[1][0] == "sample_ch1"
        assert result[2][0] == "sample_ch2"
        assert result[0][1].shape == (10, 64, 64)

    def test_3d_single_channel(self):
        data = np.zeros((10, 64, 64))
        result = split_channels(data, "mono")
        assert len(result) == 1
        assert result[0][0] == "mono"
        assert result[0][1].shape == (10, 64, 64)

    def test_raises_on_2d(self):
        data = np.zeros((64, 64))
        with pytest.raises(ValueError, match="2D"):
            split_channels(data, "bad")

    def test_raises_on_5d(self):
        data = np.zeros((1, 2, 3, 4, 5))
        with pytest.raises(ValueError, match="5D"):
            split_channels(data, "bad")


# ---------------------------------------------------------------------------
# find_next_patch_index
# ---------------------------------------------------------------------------

class TestFindNextPatchIndex:
    """Tests for cumulative patch index calculation."""

    def test_empty_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            assert find_next_patch_index(d, "patch") == 1

    def test_existing_patches(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["patch_1.tiff", "patch_2.tiff"]:
                open(os.path.join(d, name), "w").close()
            assert find_next_patch_index(d, "patch") == 3

    def test_gap_in_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["img_1.tiff", "img_3.tiff"]:
                open(os.path.join(d, name), "w").close()
            assert find_next_patch_index(d, "img") == 4

    def test_different_prefix_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["other_1.tiff", "other_2.tiff"]:
                open(os.path.join(d, name), "w").close()
            assert find_next_patch_index(d, "mydata") == 1

    def test_nonexistent_dir(self):
        assert find_next_patch_index("/nonexistent/dir", "patch") == 1

    def test_with_display_name_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["kidney_anterior_patch_1.tiff", "kidney_anterior_patch_2.tiff"]:
                open(os.path.join(d, name), "w").close()
            assert find_next_patch_index(d, "kidney_anterior_patch") == 3

    def test_mixed_prefixes(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["kidney_patch_1.tiff", "kidney_patch_2.tiff",
                         "liver_patch_1.tiff"]:
                open(os.path.join(d, name), "w").close()
            assert find_next_patch_index(d, "kidney_patch") == 3
            assert find_next_patch_index(d, "liver_patch") == 2


class TestFindNextPatchIndexLayout:
    """Union of registry + patches-dir + legacy-root sources (new layout)."""

    def test_from_registry(self):
        from napari_epicycle.data_preparation._state import (
            empty_state, register_patch, save_state,
        )
        with tempfile.TemporaryDirectory() as ws:
            st = empty_state()
            for n in (1, 2):
                register_patch(st, patch_name=f"kidney_patch_{n}",
                               relative_path=f"x_{n}.tiff",
                               sidecar_path=f"x_{n}.json", source_id="",
                               timestamp="t")
            save_state(ws, st)
            assert find_next_patch_index(ws, "kidney_patch") == 3

    def test_from_patches_dir(self):
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        with tempfile.TemporaryDirectory() as ws:
            pdir = step1_patches_dir(ws)
            os.makedirs(pdir)
            for n in (1, 2, 3):
                open(pdir / f"kidney_patch_{n}.tiff", "w").close()
            assert find_next_patch_index(ws, "kidney_patch") == 4

    def test_legacy_root_still_counts(self):
        with tempfile.TemporaryDirectory() as ws:
            for n in (1, 2):
                open(os.path.join(ws, f"kidney_patch_{n}.tiff"), "w").close()
            assert find_next_patch_index(ws, "kidney_patch") == 3

    def test_union_takes_max_across_sources(self):
        """Registry says 2, legacy root has 5 → next is 6 (no reset, no clobber)."""
        from napari_epicycle.data_preparation._state import (
            empty_state, register_patch, save_state,
        )
        with tempfile.TemporaryDirectory() as ws:
            st = empty_state()
            register_patch(st, patch_name="kidney_patch_2", relative_path="b.tiff",
                           sidecar_path="b.json", source_id="", timestamp="t")
            save_state(ws, st)
            open(os.path.join(ws, "kidney_patch_5.tiff"), "w").close()
            assert find_next_patch_index(ws, "kidney_patch") == 6

    def test_persist_then_index_is_cumulative(self):
        """End-to-end: persisting a patch advances the index via the registry."""
        from napari_epicycle.data_preparation._export import persist_patch
        with tempfile.TemporaryDirectory() as ws:
            persist_patch(
                ws, "kidney_patch_1", np.zeros((1, 2, 4, 4), dtype=np.uint8),
                channel_names=["c"], original_shape=(1, 2, 4, 4),
                cropped_shape=(1, 2, 4, 4), selected_z_range=[0, 2],
                bounding_box="x", source_path="", timestamp="t",
            )
            assert find_next_patch_index(ws, "kidney_patch") == 2

    def test_registry_match_is_case_insensitive(self):
        from napari_epicycle.data_preparation._state import (
            empty_state, register_patch, save_state,
        )
        with tempfile.TemporaryDirectory() as ws:
            st = empty_state()
            register_patch(st, patch_name="Kidney_Patch_1", relative_path="x.tiff",
                           sidecar_path="x.json", source_id="", timestamp="t")
            save_state(ws, st)
            # Querying with different case still finds it (parity with dir scan).
            assert find_next_patch_index(ws, "kidney_patch") == 2

    def test_count_patches_distinct_handles_gaps(self):
        from napari_epicycle.data_preparation._state import (
            empty_state, register_patch, save_state,
        )
        from napari_epicycle.data_preparation.logic import count_patches_for_prefix
        with tempfile.TemporaryDirectory() as ws:
            st = empty_state()
            for n in (1, 2, 5):  # gap at 3,4
                register_patch(st, patch_name=f"k_patch_{n}", relative_path=f"x{n}.tiff",
                               sidecar_path=f"x{n}.json", source_id="", timestamp="t")
            save_state(ws, st)
            # True count is 3 even though the max index is 5.
            assert count_patches_for_prefix(ws, "k_patch") == 3
            assert find_next_patch_index(ws, "k_patch") == 6

    def test_count_dedups_registry_and_dir(self):
        """A patch present in BOTH registry and patches dir counts once."""
        from napari_epicycle.data_preparation._state import (
            empty_state, register_patch, save_state,
        )
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        from napari_epicycle.data_preparation.logic import count_patches_for_prefix
        with tempfile.TemporaryDirectory() as ws:
            st = empty_state()
            register_patch(st, patch_name="k_patch_1",
                           relative_path="artifacts/step1/patches/k_patch_1.tiff",
                           sidecar_path="x.json", source_id="", timestamp="t")
            save_state(ws, st)
            pdir = step1_patches_dir(ws)
            os.makedirs(pdir)
            open(pdir / "k_patch_1.tiff", "w").close()  # same index, two sources
            assert count_patches_for_prefix(ws, "k_patch") == 1


class TestSidecarScanDirAndStateHoist:
    def test_scan_dir_prefers_patches_dir(self):
        from napari_epicycle.data_preparation.logic import step1_sidecar_scan_dir
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        with tempfile.TemporaryDirectory() as ws:
            pd = step1_patches_dir(ws)
            os.makedirs(pd)
            assert step1_sidecar_scan_dir(ws) == str(pd)

    def test_scan_dir_falls_back_to_root(self):
        from napari_epicycle.data_preparation.logic import step1_sidecar_scan_dir
        with tempfile.TemporaryDirectory() as ws:
            assert step1_sidecar_scan_dir(ws) == str(ws)

    def test_count_uses_preloaded_state_without_rereading_disk(self):
        from napari_epicycle.data_preparation._state import empty_state, register_patch
        from napari_epicycle.data_preparation.logic import count_patches_for_prefix
        with tempfile.TemporaryDirectory() as ws:
            st = empty_state()
            for n in (1, 2):
                register_patch(st, patch_name=f"k_patch_{n}", relative_path=f"x{n}.tiff",
                               sidecar_path=f"x{n}.json", source_id="", timestamp="t")
            # No step1_state.json on disk — the count must come from the passed state.
            assert count_patches_for_prefix(ws, "k_patch", state=st) == 2


# ---------------------------------------------------------------------------
# Z-range detection
# ---------------------------------------------------------------------------

class TestZRangeDetection:
    """Tests for intensity-based Z-range detection."""

    def test_basic_detection(self):
        stack = np.zeros((20, 32, 32))
        stack[5:15] = 1.0  # signal in slices 5-14
        start, end = detect_sample_z_range_by_intensity(
            stack, hard_threshold=0.1
        )
        assert start <= 5
        assert end >= 14

    def test_all_channels_union(self):
        data = np.zeros((2, 30, 16, 16))
        data[0, 5:10] = 1.0   # channel 0: slices 5-9
        data[1, 20:25] = 1.0  # channel 1: slices 20-24
        z_start, z_end = detect_sample_range_all_channels(data)
        assert z_start <= 5
        assert z_end >= 24

    def test_no_signal_raises(self):
        data = np.zeros((2, 10, 16, 16))
        with pytest.raises(ValueError):
            detect_sample_range_all_channels(data)


# ---------------------------------------------------------------------------
# Commit 1 — unified naming authority
#   split_indexed_name / patch_name_taken / allocate_unique_patch_name
# ---------------------------------------------------------------------------

class TestSplitIndexedName:
    def test_indexed(self):
        from napari_epicycle.data_preparation.logic import split_indexed_name
        assert split_indexed_name("kidney_patch_3") == ("kidney_patch", 3)

    def test_underscored_prefix_is_greedy(self):
        from napari_epicycle.data_preparation.logic import split_indexed_name
        assert split_indexed_name("roi_2_patch_3") == ("roi_2_patch", 3)

    def test_non_indexed(self):
        from napari_epicycle.data_preparation.logic import split_indexed_name
        assert split_indexed_name("p") == ("p", None)

    def test_leading_underscore_number_is_non_indexed(self):
        from napari_epicycle.data_preparation.logic import split_indexed_name
        assert split_indexed_name("_5") == ("_5", None)

    def test_trailing_underscore_is_non_indexed(self):
        from napari_epicycle.data_preparation.logic import split_indexed_name
        assert split_indexed_name("kidney_patch_") == ("kidney_patch_", None)


class TestPatchNameTaken:
    def test_free_name(self):
        from napari_epicycle.data_preparation.logic import patch_name_taken
        with tempfile.TemporaryDirectory() as ws:
            assert patch_name_taken(ws, "kidney_patch_1") is False

    def test_registry_hit(self):
        from napari_epicycle.data_preparation.logic import patch_name_taken
        from napari_epicycle.data_preparation._state import (
            empty_state, register_patch, save_state)
        with tempfile.TemporaryDirectory() as ws:
            st = empty_state()
            register_patch(st, patch_name="k_1", relative_path="a.tiff",
                           sidecar_path="a.json", source_id="", timestamp="t")
            save_state(ws, st)
            assert patch_name_taken(ws, "k_1") is True

    def test_disk_tiff(self):
        from napari_epicycle.data_preparation.logic import patch_name_taken
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        with tempfile.TemporaryDirectory() as ws:
            pdir = step1_patches_dir(ws); os.makedirs(pdir)
            open(pdir / "k_1.tiff", "w").close()
            assert patch_name_taken(ws, "k_1") is True

    def test_disk_tif_single_f(self):
        """B5: a legacy .tif (single-f) must also register as taken."""
        from napari_epicycle.data_preparation.logic import patch_name_taken
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        with tempfile.TemporaryDirectory() as ws:
            pdir = step1_patches_dir(ws); os.makedirs(pdir)
            open(pdir / "k_1.tif", "w").close()
            assert patch_name_taken(ws, "k_1") is True

    def test_sidecar_only(self):
        from napari_epicycle.data_preparation.logic import patch_name_taken
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        with tempfile.TemporaryDirectory() as ws:
            pdir = step1_patches_dir(ws); os.makedirs(pdir)
            open(pdir / "k_1_metadata.json", "w").close()
            assert patch_name_taken(ws, "k_1") is True

    def test_case_insensitive_on_disk(self):
        """ext4+NTFS parity: a case-variant on disk is still 'taken'."""
        from napari_epicycle.data_preparation.logic import patch_name_taken
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        with tempfile.TemporaryDirectory() as ws:
            pdir = step1_patches_dir(ws); os.makedirs(pdir)
            open(pdir / "Kidney_1.tiff", "w").close()
            assert patch_name_taken(ws, "kidney_1") is True

    def test_legacy_root_counts(self):
        from napari_epicycle.data_preparation.logic import patch_name_taken
        with tempfile.TemporaryDirectory() as ws:
            open(os.path.join(ws, "k_1.tiff"), "w").close()
            assert patch_name_taken(ws, "k_1") is True


class TestAllocateUniquePatchName:
    def _seed(self, ws, indices, ext="tiff"):
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        pdir = step1_patches_dir(ws); os.makedirs(pdir, exist_ok=True)
        for n in indices:
            open(pdir / f"kidney_patch_{n}.{ext}", "w").close()
        return pdir

    def test_free_passthrough(self):
        from napari_epicycle.data_preparation.logic import allocate_unique_patch_name
        with tempfile.TemporaryDirectory() as ws:
            assert allocate_unique_patch_name(ws, "kidney_patch_1") == "kidney_patch_1"

    def test_indexed_collision_is_indexer_visible(self):
        from napari_epicycle.data_preparation.logic import (
            allocate_unique_patch_name, find_next_patch_index)
        with tempfile.TemporaryDirectory() as ws:
            pdir = self._seed(ws, (1, 2, 3))
            name = allocate_unique_patch_name(ws, "kidney_patch_3")
            assert name == "kidney_patch_4"          # NOT nested kidney_patch_3_2
            open(pdir / f"{name}.tiff", "w").close()
            assert find_next_patch_index(ws, "kidney_patch") == 5  # visible to indexer

    def test_orphan_sidecar_at_maxplus1_advances_no_hang(self):
        from napari_epicycle.data_preparation.logic import allocate_unique_patch_name
        with tempfile.TemporaryDirectory() as ws:
            pdir = self._seed(ws, (1, 2, 3))
            # orphan sidecar occupies index 4 (find_next scans TIFFs only) -> skip to 5
            open(pdir / "kidney_patch_4_metadata.json", "w").close()
            assert allocate_unique_patch_name(ws, "kidney_patch_3") == "kidney_patch_5"

    def test_non_indexed_yields_underscore_2(self):
        from napari_epicycle.data_preparation.logic import allocate_unique_patch_name
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        with tempfile.TemporaryDirectory() as ws:
            pdir = step1_patches_dir(ws); os.makedirs(pdir)
            open(pdir / "p.tiff", "w").close()
            assert allocate_unique_patch_name(ws, "p") == "p_2"

    def test_tif_sibling_forces_bump(self):
        from napari_epicycle.data_preparation.logic import allocate_unique_patch_name
        with tempfile.TemporaryDirectory() as ws:
            self._seed(ws, (1,), ext="tif")  # legacy single-f .tif at index 1
            assert allocate_unique_patch_name(ws, "kidney_patch_1") == "kidney_patch_2"


# ---------------------------------------------------------------------------
# Commit 2 — ROI validation (drop boxes that would extract an all-zero patch
# and still burn a monotonic index)
# ---------------------------------------------------------------------------

class TestFilterExtractableBoxes:
    def _box(self, y0, x0, y1, x1):
        return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)

    def test_valid_box_kept(self):
        from napari_epicycle.data_preparation.logic import filter_extractable_boxes
        valid, dropped = filter_extractable_boxes([self._box(0, 0, 16, 16)], (2, 5, 32, 32))
        assert len(valid) == 1
        assert dropped == 0

    def test_zero_area_box_dropped(self):
        from napari_epicycle.data_preparation.logic import filter_extractable_boxes
        valid, dropped = filter_extractable_boxes([self._box(5, 5, 5, 5)], (2, 5, 32, 32))
        assert valid == []
        assert dropped == 1

    def test_fully_outside_box_dropped(self):
        from napari_epicycle.data_preparation.logic import filter_extractable_boxes
        valid, dropped = filter_extractable_boxes([self._box(40, 40, 60, 60)], (2, 5, 32, 32))
        assert valid == []
        assert dropped == 1

    def test_partial_overlap_kept(self):
        # extends past the 32x32 image but overlaps -> real data + padding, keep it.
        from napari_epicycle.data_preparation.logic import filter_extractable_boxes
        valid, dropped = filter_extractable_boxes([self._box(16, 16, 48, 48)], (2, 5, 32, 32))
        assert len(valid) == 1
        assert dropped == 0

    def test_mixed_batch(self):
        from napari_epicycle.data_preparation.logic import filter_extractable_boxes
        boxes = [self._box(0, 0, 16, 16), self._box(5, 5, 5, 5), self._box(8, 8, 24, 24)]
        valid, dropped = filter_extractable_boxes(boxes, (2, 5, 32, 32))
        assert len(valid) == 2
        assert dropped == 1

    def test_3d_image_shape(self):
        from napari_epicycle.data_preparation.logic import filter_extractable_boxes
        valid, dropped = filter_extractable_boxes([self._box(0, 0, 16, 16)], (5, 32, 32))
        assert len(valid) == 1
        assert dropped == 0

    def test_empty_input(self):
        from napari_epicycle.data_preparation.logic import filter_extractable_boxes
        assert filter_extractable_boxes([], (2, 5, 32, 32)) == ([], 0)


# ---------------------------------------------------------------------------
# 5-eye audit fix (F1) — truly per-file patch count by source_id, so two files
# sharing a display-name don't cross-count and a rename doesn't undercount.
# ---------------------------------------------------------------------------

class TestCountPatchesForSource:
    def _state_with(self, rows):
        from napari_epicycle.data_preparation._state import empty_state
        st = empty_state()
        st["patches"] = [{"patch_name": n, "source_id": s} for n, s in rows]
        return st

    def test_counts_only_matching_source_id(self):
        from napari_epicycle.data_preparation.logic import count_patches_for_source
        st = self._state_with([
            ("kidney_patch_1", "src_0"), ("kidney_patch_2", "src_0"),
            ("liver_patch_1", "src_1"),
        ])
        assert count_patches_for_source(st, "src_0") == 2
        assert count_patches_for_source(st, "src_1") == 1

    def test_rename_between_sessions_counts_all(self):
        """One file, two prefixes (renamed mid-way) -> both count under its source."""
        from napari_epicycle.data_preparation.logic import count_patches_for_source
        st = self._state_with([
            ("kidney_patch_1", "src_0"), ("kidney_patch_2", "src_0"),
            ("kidney_dorsal_patch_1", "src_0"),  # after rename, same file/source
        ])
        assert count_patches_for_source(st, "src_0") == 3

    def test_same_name_files_do_not_cross_count(self):
        """Two files, same display-name/prefix, but distinct source_ids."""
        from napari_epicycle.data_preparation.logic import count_patches_for_source
        st = self._state_with([
            ("kidney_patch_1", "src_0"), ("kidney_patch_1", "src_1"),
        ])
        assert count_patches_for_source(st, "src_0") == 1
        assert count_patches_for_source(st, "src_1") == 1

    def test_empty_source_id_returns_zero(self):
        from napari_epicycle.data_preparation.logic import count_patches_for_source
        st = self._state_with([("legacy_patch_1", "")])
        assert count_patches_for_source(st, "") == 0


# ---------------------------------------------------------------------------
# Re-extract fix — committed-box-signature exclusion so already-exported /
# restored boxes aren't re-extracted (escalating patch numbers).
# ---------------------------------------------------------------------------

class TestBoxSignature:
    def _box(self, y0, x0, y1, x1):
        return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)

    def test_floored_aabb(self):
        from napari_epicycle.data_preparation.logic import box_signature
        assert box_signature(self._box(5.2, 3.9, 20.7, 18.1)) == (5, 20, 3, 18)

    def test_winding_invariant(self):
        """Same rectangle in a different corner order -> same signature (napari
        may reorder corners when re-adding a restored box)."""
        from napari_epicycle.data_preparation.logic import box_signature
        cw = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
        reordered = np.array([[10, 10], [10, 0], [0, 0], [0, 10]], dtype=float)
        assert box_signature(cw) == box_signature(reordered)

    def test_json_roundtrip_matches(self):
        """A box stored to the sidecar (list) + restored via np.array matches."""
        import json
        from napari_epicycle.data_preparation.logic import box_signature
        b = self._box(5, 3, 20, 18)
        restored = np.array(json.loads(json.dumps(b.tolist())), dtype=float)
        assert box_signature(b) == box_signature(restored)

    def test_accepts_plain_list(self):
        from napari_epicycle.data_preparation.logic import box_signature
        assert box_signature([[5, 3], [5, 18], [20, 18], [20, 3]]) == (5, 20, 3, 18)


class TestSelectUncommittedBoxes:
    def _box(self, y0, x0, y1, x1):
        return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)

    def test_filters_committed(self):
        from napari_epicycle.data_preparation.logic import (
            box_signature, select_uncommitted_boxes)
        a, b = self._box(0, 0, 10, 10), self._box(20, 20, 30, 30)
        result = select_uncommitted_boxes([a, b], {box_signature(a)})
        assert len(result) == 1
        assert result[0][0] == 1  # original index of the surviving box b

    def test_all_new_when_no_committed(self):
        from napari_epicycle.data_preparation.logic import select_uncommitted_boxes
        boxes = [self._box(0, 0, 10, 10), self._box(20, 20, 30, 30)]
        assert len(select_uncommitted_boxes(boxes, set())) == 2

    def test_original_indices_preserved_after_gaps(self):
        from napari_epicycle.data_preparation.logic import (
            box_signature, select_uncommitted_boxes)
        a, b, c = self._box(0, 0, 5, 5), self._box(10, 10, 15, 15), self._box(20, 20, 25, 25)
        result = select_uncommitted_boxes([a, b, c], {box_signature(b)})
        assert [i for i, _ in result] == [0, 2]


class TestBoxSignatureZAware:
    """The box's editable Z now folds into its identity: two boxes with the
    SAME XY footprint but DIFFERENT Z slabs must be distinct signatures, else
    narrowing (or re-drawing at a different depth) a same-XY box is silently
    dropped as "already captured" and its Z-slab patch never extracts."""

    def _box(self, y0, x0, y1, x1):
        return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)

    def test_z_range_none_stays_xy_only(self):
        """Back-compat: no z_range -> the unchanged (y0,y1,x0,x1) 4-tuple."""
        from napari_epicycle.data_preparation.logic import box_signature
        b = self._box(5, 3, 20, 18)
        assert box_signature(b) == (5, 20, 3, 18)
        assert box_signature(b, None) == (5, 20, 3, 18)

    def test_z_range_folds_into_identity(self):
        from napari_epicycle.data_preparation.logic import box_signature
        b = self._box(0, 0, 10, 10)
        assert box_signature(b, [0, 100]) == (0, 10, 0, 10, 0, 100)
        # Same XY footprint, different Z slab -> DIFFERENT signature.
        assert box_signature(b, [0, 50]) != box_signature(b, [0, 100])
        # Same XY + same Z -> identical signature (still dedups).
        assert box_signature(b, [10, 40]) == box_signature(b, [10, 40])

    def test_narrowed_z_same_xy_not_dropped_but_same_z_is(self):
        """The reported defect: export a full-Z box (commits its sig), then a
        same-XY box with a NARROWED Z must survive the uncommitted filter (its
        slab still needs extracting) while a same-XY SAME-Z box is deduped."""
        from napari_epicycle.data_preparation.logic import (
            box_signature, select_uncommitted_boxes)
        b = self._box(0, 0, 10, 10)
        committed = {box_signature(b, [0, 100])}          # full-Z box exported
        # Full-Z re-draw -> deduped (nothing survives).
        assert select_uncommitted_boxes([b], committed, [[0, 100]]) == []
        # Same-XY NARROWED-Z box -> NOT dropped (the slab must extract).
        survived = select_uncommitted_boxes([b], committed, [[0, 50]])
        assert len(survived) == 1 and survived[0][0] == 0

    def test_z_ranges_threaded_per_box_and_none_is_xy_only(self):
        from napari_epicycle.data_preparation.logic import (
            box_signature, select_uncommitted_boxes)
        a, b = self._box(0, 0, 10, 10), self._box(20, 20, 30, 30)
        # a committed at [0,100]; b uncommitted.  Per-box z_ranges aligned.
        committed = {box_signature(a, [0, 100])}
        result = select_uncommitted_boxes([a, b], committed, [[0, 100], [0, 100]])
        assert [i for i, _ in result] == [1]        # only b survives
        # z_ranges=None keeps the legacy XY-only behaviour intact.
        legacy = select_uncommitted_boxes([a, b], {box_signature(a)}, None)
        assert [i for i, _ in legacy] == [1]

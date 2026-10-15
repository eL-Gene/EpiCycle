"""Tests for global indirect reconstruction stage and downsampling utilities."""

import json
import numpy as np
import pytest
from pathlib import Path

from packages.epicycle_scaledseg.core.global_reconstruction import GlobalReconstructionStage
from packages.epicycle_scaledseg.utils.downsampling import maxpool_downsample_3d, nearest_downsample_3d


class TestGlobalReconstructionStageInit:
    def test_defaults(self):
        stage = GlobalReconstructionStage(assembly_dir="/fake/assembly")
        assert stage.name == "global_reconstruction"
        assert stage.assembly_dir == "/fake/assembly"
        assert stage.downsample_factor == 2
        assert stage.min_size == 1000
        assert stage.use_precomputed_binary is True
        assert stage.indirect_params is None
        assert stage.n_cpu == 64
        assert stage.tile_shape is None
        assert stage.output_dir is None

    def test_custom_params(self):
        stage = GlobalReconstructionStage(
            assembly_dir="/data/stage3",
            output_dir="/data/stage4",
            downsample_factor=4,
            min_size=500,
            use_precomputed_binary=False,
            n_cpu=32,
            tile_shape=(100, 512, 512),
        )
        assert stage.downsample_factor == 4
        assert stage.min_size == 500
        assert stage.use_precomputed_binary is False
        assert stage.n_cpu == 32
        assert stage.tile_shape == (100, 512, 512)


class TestGlobalReconstructionValidation:
    def test_validate_missing_assembly_dir(self, tmp_path):
        stage = GlobalReconstructionStage(assembly_dir=str(tmp_path / "nonexistent"))
        assert stage.validate_inputs([]) is False

    def test_validate_missing_mask(self, tmp_path):
        adir = tmp_path / "assembly"
        adir.mkdir()
        stage = GlobalReconstructionStage(assembly_dir=str(adir))
        assert stage.validate_inputs([]) is False

    def test_validate_success(self, tmp_path):
        import tifffile
        adir = tmp_path / "assembly"
        adir.mkdir()
        # Create a small fake mask
        fake = np.zeros((4, 8, 8), dtype=np.uint16)
        tifffile.imwrite(str(adir / "full_mask.tif"), fake)
        stage = GlobalReconstructionStage(assembly_dir=str(adir))
        assert stage.validate_inputs([]) is True

    def test_validate_none_assembly_dir(self):
        stage = GlobalReconstructionStage(assembly_dir=None)
        assert stage.validate_inputs([]) is False


class TestGlobalReconstructionSkipCompleted:
    def test_skips_if_output_exists(self, tmp_path):
        import tifffile
        adir = tmp_path / "assembly"
        adir.mkdir()
        tifffile.imwrite(str(adir / "full_mask.tif"), np.zeros((4, 8, 8), dtype=np.uint16))

        out_dir = tmp_path / "stage4"
        out_dir.mkdir()
        tifffile.imwrite(str(out_dir / "global_segmentation.tif"), np.zeros((2, 4, 4), dtype=np.uint16))

        stage = GlobalReconstructionStage(assembly_dir=str(adir), output_dir=str(out_dir))
        result = stage.run([], n_gpus=1)
        assert result.n_success == 1


class TestMaxpoolDownsample:
    def test_basic_shape(self):
        vol = np.ones((8, 8, 8), dtype=np.uint8)
        result = maxpool_downsample_3d(vol, factor=2)
        assert result.shape == (4, 4, 4)

    def test_preserves_foreground(self):
        vol = np.zeros((8, 8, 8), dtype=np.uint8)
        vol[0, 0, 0] = 1  # single voxel
        result = maxpool_downsample_3d(vol, factor=2)
        assert result[0, 0, 0] == 1

    def test_factor_4(self):
        vol = np.ones((16, 16, 16), dtype=np.uint8)
        result = maxpool_downsample_3d(vol, factor=4)
        assert result.shape == (4, 4, 4)

    def test_non_divisible_trims(self):
        vol = np.ones((9, 9, 9), dtype=np.uint8)
        result = maxpool_downsample_3d(vol, factor=2)
        assert result.shape == (4, 4, 4)

    def test_max_operation(self):
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        vol[1, 1, 1] = 255
        result = maxpool_downsample_3d(vol, factor=2)
        assert result[0, 0, 0] == 255


class TestNearestDownsample:
    def test_basic_shape(self):
        vol = np.ones((8, 8, 8), dtype=np.uint16)
        result = nearest_downsample_3d(vol, factor=2)
        assert result.shape == (4, 4, 4)

    def test_preserves_labels(self):
        vol = np.zeros((8, 8, 8), dtype=np.uint16)
        vol[0, 0, 0] = 42
        vol[2, 2, 2] = 99
        result = nearest_downsample_3d(vol, factor=2)
        assert result[0, 0, 0] == 42
        assert result[1, 1, 1] == 99

    def test_returns_copy(self):
        vol = np.ones((4, 4, 4), dtype=np.uint16)
        result = nearest_downsample_3d(vol, factor=2)
        result[0, 0, 0] = 999
        assert vol[0, 0, 0] == 1  # original unchanged

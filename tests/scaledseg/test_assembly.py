"""Tests for epicycle.core.assembly module."""

import pytest
import numpy as np
from pathlib import Path

from packages.epicycle_scaledseg.core.assembly import (
    parse_tile_positions,
    downsample_binary_maxpool,
    AssemblyStage,
    stitch_mask_volume,
    stitch_prob_volume,
    stitch_binary_volume,
)


class TestParseTilePositions:
    def test_basic_grid(self, tmp_path):
        """tile_0_0, tile_0_1, tile_1_0 with 512px, 25% overlap."""
        names = ["tile_0_0_seg", "tile_0_1_seg", "tile_1_0_seg"]
        dirs = []
        for n in names:
            d = tmp_path / n
            d.mkdir()
            dirs.append(d)

        positions = parse_tile_positions(dirs, tile_size=512, overlap=0.25)
        step = int(512 * 0.75)  # 384

        assert len(positions) == 3
        # sorted by (y, x)
        assert positions[0] == (0, 0, dirs[0])
        assert positions[1] == (0, step, dirs[1])
        assert positions[2] == (step, 0, dirs[2])

    def test_ignores_non_matching_names(self, tmp_path):
        d1 = tmp_path / "tile_0_0_seg"
        d1.mkdir()
        d2 = tmp_path / "logs"
        d2.mkdir()
        positions = parse_tile_positions([d1, d2])
        assert len(positions) == 1

    def test_custom_overlap(self, tmp_path):
        d = tmp_path / "tile_2_3_foo"
        d.mkdir()
        positions = parse_tile_positions([d], tile_size=100, overlap=0.5)
        step = 50
        assert positions[0][:2] == (2 * step, 3 * step)


class TestDownsampleBinaryMaxpool:
    def test_factor_2(self):
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        vol[0, 0, 0] = 1  # single voxel
        result = downsample_binary_maxpool(vol, factor=2)
        assert result.shape == (2, 2, 2)
        assert result[0, 0, 0] == 1
        assert result.sum() == 1

    def test_factor_1_noop(self):
        vol = np.ones((4, 4, 4), dtype=np.uint8)
        result = downsample_binary_maxpool(vol, factor=1)
        assert np.array_equal(result, vol)

    def test_too_large_factor_raises(self):
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            downsample_binary_maxpool(vol, factor=8)


class TestAssemblyStageInit:
    def test_defaults(self):
        stage = AssemblyStage()
        assert stage.name == "assembly"
        assert stage.full_size is None
        assert stage.tile_size is None
        assert stage.overlap is None
        assert stage.depth is None
        assert stage.tiling_config_path is None
        assert stage.stitch_mask is True
        assert stage.stitch_prob is True
        assert stage.stitch_binary is True
        assert stage.prob_stitch_mode == 'max'
        assert stage.binary_format == 'npy'

    def test_custom_params(self):
        stage = AssemblyStage(
            full_size=(1024, 1024),
            tile_size=256,
            overlap=0.5,
            depth=100,
            stitch_prob=False,
            prob_stitch_mode='max',
        )
        assert stage.full_size == (1024, 1024)
        assert stage.tile_size == 256
        assert stage.overlap == 0.5
        assert stage.depth == 100
        assert stage.stitch_prob is False
        assert stage.prob_stitch_mode == 'max'

    def test_repr(self):
        stage = AssemblyStage(full_size=(7680, 7680), tile_size=512, overlap=0.25, depth=1886)
        r = repr(stage)
        assert "assembly" in r.lower()
        assert "7680" in r


class TestAssemblyAutoInfer:
    def test_loads_tiling_config(self, tmp_path):
        """AssemblyStage should auto-infer params from tiling_config.json."""
        import json
        tile_dir = tmp_path / "tile_0_0"
        tile_dir.mkdir()
        config = {
            "full_size": [4096, 4096],
            "patch_size": 1024,
            "overlap": 0.25,
            "depth": 500,
            "tiles": ["tile_0_0"],
        }
        (tmp_path / "tiling_config.json").write_text(json.dumps(config))

        stage = AssemblyStage()
        stage._load_tiling_config([tile_dir])
        assert stage.full_size == (4096, 4096)
        assert stage.tile_size == 1024
        assert stage.overlap == 0.25
        assert stage.depth == 500

    def test_explicit_params_not_overridden(self, tmp_path):
        """Explicit params should not be overridden by tiling config."""
        import json
        tile_dir = tmp_path / "tile_0_0"
        tile_dir.mkdir()
        config = {
            "full_size": [4096, 4096],
            "patch_size": 1024,
            "overlap": 0.25,
            "depth": 500,
        }
        (tmp_path / "tiling_config.json").write_text(json.dumps(config))

        stage = AssemblyStage(full_size=(2048, 2048), depth=100)
        stage._load_tiling_config([tile_dir])
        assert stage.full_size == (2048, 2048)
        assert stage.depth == 100
        # These should be filled from config
        assert stage.tile_size == 1024
        assert stage.overlap == 0.25

    def test_no_config_leaves_none(self, tmp_path):
        """Without tiling_config.json, params stay None."""
        tile_dir = tmp_path / "tile_0_0"
        tile_dir.mkdir()
        stage = AssemblyStage()
        stage._load_tiling_config([tile_dir])
        assert stage.full_size is None

    def test_run_fails_gracefully_without_params(self, tmp_path):
        """run() should return failures if params can't be inferred."""
        tile_dir = tmp_path / "tile_0_0"
        tile_dir.mkdir()
        stage = AssemblyStage()
        result = stage.run([tile_dir], n_gpus=1)
        assert result.n_failed > 0


class TestAssemblyStageValidation:
    def test_validates_recon_root_exists(self, tmp_path):
        stage = AssemblyStage()
        tile = tmp_path / "tile_0_0"
        tile.mkdir()
        # No stage2_reconstruction dir → fails
        assert stage.validate_inputs([tile]) is False

    def test_validates_at_least_one_mask(self, tmp_path):
        stage = AssemblyStage()
        tile = tmp_path / "tile_0_0"
        tile.mkdir()
        recon = tmp_path / "stage2_reconstruction" / "tile_0_0"
        recon.mkdir(parents=True)
        # No mask file → fails
        assert stage.validate_inputs([tile]) is False

    def test_passes_with_mask(self, tmp_path):
        stage = AssemblyStage()
        tile = tmp_path / "tile_0_0"
        tile.mkdir()
        recon = tmp_path / "stage2_reconstruction" / "tile_0_0"
        recon.mkdir(parents=True)
        (recon / "tile_0_0_mask.tif").touch()
        assert stage.validate_inputs([tile]) is True


class TestStitchMaskVolume:
    def test_stitches_single_tile(self, tmp_path):
        """Create a tiny 2x2x2 tile mask and stitch it."""
        import tifffile

        tile = tmp_path / "tile_0_0"
        tile.mkdir()
        recon = tmp_path / "recon" / "tile_0_0"
        recon.mkdir(parents=True)

        # Create a 2-slice, 4x4 mask — write each Z as a separate page
        mask_path = str(recon / "tile_0_0_mask.tif")
        with tifffile.TiffWriter(mask_path, bigtiff=True) as tw:
            s0 = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint32)
            s1 = np.array([[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint32)
            tw.write(s0)
            tw.write(s1)

        positions = [(0, 0, tile)]
        out_path = str(tmp_path / "stitched.tif")

        stitch_mask_volume(
            positions, recon.parent, out_path,
            full_size=(4, 4), tile_size=4, depth=2, n_workers=1,
        )

        result = tifffile.imread(out_path)
        assert result.shape == (2, 4, 4)
        assert result[0, 0, 0] == 1
        assert result[1, 0, 0] == 2


class TestStitchProbVolume:
    def test_single_tile_average(self, tmp_path):
        tile = tmp_path / "tile_0_0"
        tile.mkdir()
        recon = tmp_path / "recon" / "tile_0_0"
        recon.mkdir(parents=True)

        prob = np.ones((2, 4, 4), dtype=np.float32) * 0.5
        np.save(str(recon / "tile_0_0_probabilities.npy"), prob)

        positions = [(0, 0, tile)]
        out_path = str(tmp_path / "prob.npy")

        stitch_prob_volume(
            positions, recon.parent, out_path,
            full_size=(4, 4), tile_size=4, depth=2,
        )

        result = np.load(out_path)
        assert result.shape == (2, 4, 4)
        np.testing.assert_allclose(result, 0.5, atol=1e-5)


class TestStitchBinaryVolume:
    def test_single_tile_npy(self, tmp_path):
        tile = tmp_path / "tile_0_0"
        tile.mkdir()
        recon = tmp_path / "recon" / "tile_0_0"
        recon.mkdir(parents=True)

        binary = np.zeros((2, 4, 4), dtype=np.uint8)
        binary[0, 0, 0] = 1
        binary[1, 3, 3] = 1
        np.save(str(recon / "tile_0_0_binary.npy"), binary)

        positions = [(0, 0, tile)]
        out_path = str(tmp_path / "binary.npy")

        stitch_binary_volume(
            positions, recon.parent, out_path,
            full_size=(4, 4), tile_size=4, depth=2,
            save_format='npy',
        )

        result = np.load(out_path)
        assert result.shape == (2, 4, 4)
        assert result[0, 0, 0] == 1
        assert result[1, 3, 3] == 1
        assert result.sum() == 2

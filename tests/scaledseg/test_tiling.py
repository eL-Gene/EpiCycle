"""Tests for tiling stage and ZStackTiler."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from packages.epicycle_scaledseg.core.tiling import TilingStage


class TestTilingStageInit:
    def test_defaults(self, tmp_path):
        tif = tmp_path / "stack.tif"
        tif.touch()
        stage = TilingStage(input_tif=str(tif))
        assert stage.name == "tiling"
        assert stage.input_tif == str(tif.resolve())
        assert stage.patch_size == 1024
        assert stage.overlap == 0.25
        assert stage.z_threshold_method == "hard"
        assert stage.threshold_value == 0.07
        assert stage.channel_index == 1
        assert stage.auto_correct is False
        assert stage.output_dir is None

    def test_custom_params(self, tmp_path):
        tif = tmp_path / "vol.tif"
        tif.touch()
        tiles = tmp_path / "tiles"
        tiles.mkdir()
        stage = TilingStage(
            input_tif=str(tif),
            output_dir=str(tiles),
            patch_size=512,
            overlap=0.5,
            z_threshold_method="percentile",
            threshold_value=0.1,
            channel_index=0,
            auto_correct=True,
        )
        assert stage.patch_size == 512
        assert stage.overlap == 0.5
        assert stage.output_dir == str(tiles.resolve())
        assert stage.auto_correct is True


class TestTilingStageValidation:
    def test_validate_missing_tif(self, tmp_path):
        stage = TilingStage(input_tif=str(tmp_path / "nonexistent.tif"))
        assert stage.validate_inputs([]) is False

    def test_validate_existing_tif(self, tmp_path):
        tif = tmp_path / "stack.tif"
        tif.write_bytes(b"fake")
        stage = TilingStage(input_tif=str(tif))
        assert stage.validate_inputs([]) is True


class TestTilingStageSkipCompleted:
    def test_skips_if_tiling_config_exists(self, tmp_path):
        tif = tmp_path / "stack.tif"
        tif.write_bytes(b"fake")

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()
        (tiles_dir / "tile_0_0").mkdir()
        (tiles_dir / "tile_0_1").mkdir()

        cfg = {"tiles": ["tile_0_0", "tile_0_1"]}
        with open(tiles_dir / "tiling_config.json", "w") as f:
            json.dump(cfg, f)

        stage = TilingStage(input_tif=str(tif), output_dir=str(tiles_dir))
        result = stage.run([], n_gpus=1)

        assert result.n_success == 2
        assert result.tile_dirs is not None
        assert len(result.tile_dirs) == 2


class TestTilingStageGetTileDirs:
    def test_empty_before_run(self):
        stage = TilingStage(input_tif="/fake.tif")
        assert stage.get_tile_dirs() == []


class TestStageResultTileDirs:
    def test_tile_dirs_default_none(self):
        from packages.epicycle_scaledseg.core.pipeline import StageResult
        sr = StageResult()
        assert sr.tile_dirs is None

    def test_tile_dirs_settable(self):
        from packages.epicycle_scaledseg.core.pipeline import StageResult
        sr = StageResult()
        sr.tile_dirs = [Path("/a"), Path("/b")]
        assert len(sr.tile_dirs) == 2

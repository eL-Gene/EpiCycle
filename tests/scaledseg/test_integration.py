"""Integration tests for the full EpiCycle pipeline.

Tests the pipeline orchestration, checkpointing, resume, CLI argument
parsing, and status reporting — all using DummyStages so no GPU or
heavy dependencies are needed.
"""

import json
import pytest
from pathlib import Path

from packages.epicycle_scaledseg.core.pipeline import (
    Stage,
    StageResult,
    PipelineResult,
    EpiCyclePipeline,
)


# --- Dummy stages for integration testing ---

class PassStage(Stage):
    """Stage that marks all tiles as success."""

    def __init__(self, name="pass_stage"):
        super().__init__(name)
        self.ran = False

    def validate_inputs(self, tile_dirs):
        return True

    def run(self, tile_dirs, n_gpus, log_dir=None):
        self.ran = True
        result = StageResult()
        for td in tile_dirs:
            result.add_success(td.name)
        return result


class PartialFailStage(Stage):
    """Stage that fails specific tiles."""

    def __init__(self, name="partial", fail_names=None):
        super().__init__(name)
        self.fail_names = set(fail_names or [])

    def validate_inputs(self, tile_dirs):
        return True

    def run(self, tile_dirs, n_gpus, log_dir=None):
        result = StageResult()
        for td in tile_dirs:
            if td.name in self.fail_names:
                result.add_failure(td.name, "test failure")
            else:
                result.add_success(td.name)
        return result


# --- Pipeline integration tests ---

class TestPipelineCheckpoint:
    def _make_tiles(self, tmp_path, n=3):
        tiles = []
        for i in range(n):
            t = tmp_path / f"tile_{i}"
            t.mkdir()
            tiles.append(str(t))
        return tiles

    def test_checkpoint_created_on_full_success(self, tmp_path):
        tiles = self._make_tiles(tmp_path)
        p = EpiCyclePipeline(str(tmp_path), n_gpus=1, log_dir=str(tmp_path / "logs"))
        p.add_stage(PassStage("seg"))
        p.run(tiles)

        # The checkpoint is mode-scoped (default mode "coarse" ->
        # pipeline_checkpoint_coarse.json); reference the actual path.
        ckpt = p._checkpoint_path
        assert ckpt.exists()
        data = json.loads(ckpt.read_text())
        assert "seg" in data["completed_stages"]

    def test_checkpoint_not_created_on_partial_failure(self, tmp_path):
        tiles = self._make_tiles(tmp_path)
        p = EpiCyclePipeline(str(tmp_path), n_gpus=1, log_dir=str(tmp_path / "logs"))
        p.add_stage(PartialFailStage("seg", fail_names={"tile_1"}))
        p.run(tiles)

        ckpt = p._checkpoint_path
        if ckpt.exists():
            data = json.loads(ckpt.read_text())
            assert "seg" not in data.get("completed_stages", [])

    def test_resume_skips_completed_stage(self, tmp_path):
        tiles = self._make_tiles(tmp_path)
        log_dir = tmp_path / "logs"

        # First run: seg passes
        p1 = EpiCyclePipeline(str(tmp_path), n_gpus=1, log_dir=str(log_dir))
        seg1 = PassStage("seg")
        p1.add_stage(seg1)
        p1.run(tiles)
        assert seg1.ran is True

        # Second run with resume: seg should be skipped
        p2 = EpiCyclePipeline(str(tmp_path), n_gpus=1, log_dir=str(log_dir))
        seg2 = PassStage("seg")
        recon2 = PassStage("recon")
        p2.add_stage(seg2)
        p2.add_stage(recon2)
        result = p2.run(tiles, resume=True)

        assert seg2.ran is False  # skipped via checkpoint
        assert recon2.ran is True
        assert result.n_success == 3

    def test_resume_false_reruns_everything(self, tmp_path):
        tiles = self._make_tiles(tmp_path)
        log_dir = tmp_path / "logs"

        # First run
        p1 = EpiCyclePipeline(str(tmp_path), n_gpus=1, log_dir=str(log_dir))
        p1.add_stage(PassStage("seg"))
        p1.run(tiles)

        # Second run without resume
        p2 = EpiCyclePipeline(str(tmp_path), n_gpus=1, log_dir=str(log_dir))
        seg2 = PassStage("seg")
        p2.add_stage(seg2)
        p2.run(tiles, resume=False)

        assert seg2.ran is True  # ran again because resume=False


class TestPipelinePropagation:
    def test_two_stages_all_pass(self, tmp_path):
        tiles = []
        for i in range(3):
            t = tmp_path / f"tile_{i}"
            t.mkdir()
            tiles.append(str(t))

        p = EpiCyclePipeline(str(tmp_path), n_gpus=2, log_dir=str(tmp_path / "logs"))
        p.add_stage(PassStage("seg"))
        p.add_stage(PassStage("recon"))
        result = p.run(tiles)

        assert result.n_total == 3
        assert result.n_success == 3
        assert "seg" in result.stage_results
        assert "recon" in result.stage_results

    def test_failed_tiles_not_propagated(self, tmp_path):
        tiles = []
        for i in range(3):
            t = tmp_path / f"tile_{i}"
            t.mkdir()
            tiles.append(str(t))

        p = EpiCyclePipeline(str(tmp_path), n_gpus=1, log_dir=str(tmp_path / "logs"))
        p.add_stage(PartialFailStage("seg", fail_names={"tile_1"}))
        p.add_stage(PassStage("recon"))
        result = p.run(tiles)

        assert result.stage_results["seg"].n_success == 2
        assert result.stage_results["seg"].n_failed == 1
        # recon only sees the 2 tiles that passed seg
        assert result.stage_results["recon"].n_total == 2
        assert result.stage_results["recon"].n_success == 2

    def test_all_fail_stops_pipeline(self, tmp_path):
        tiles = []
        for i in range(2):
            t = tmp_path / f"tile_{i}"
            t.mkdir()
            tiles.append(str(t))

        p = EpiCyclePipeline(str(tmp_path), n_gpus=1, log_dir=str(tmp_path / "logs"))
        p.add_stage(PartialFailStage("seg", fail_names={"tile_0", "tile_1"}))
        recon = PassStage("recon")
        p.add_stage(recon)
        result = p.run(tiles)

        assert recon.ran is False  # pipeline stopped
        assert "recon" not in result.stage_results


class TestCLIParsing:
    def test_main_import(self):
        from packages.epicycle_scaledseg.cli.main import main
        assert callable(main)

    def test_status_command_on_empty_dir(self, tmp_path, capsys):
        """Status command should not crash on an empty work dir."""
        from packages.epicycle_scaledseg.cli.main import _cmd_status
        import argparse
        args = argparse.Namespace(
            work_dir=str(tmp_path),
            seg_root=None,
            recon_root=None,
        )
        _cmd_status(args)
        captured = capsys.readouterr()
        assert "Total tiles: 0" in captured.out

    def test_status_with_tiles(self, tmp_path, capsys):
        """Status command reports correct counts."""
        from packages.epicycle_scaledseg.cli.main import _cmd_status
        import argparse

        # Create 3 tile dirs
        for i in range(3):
            (tmp_path / f"tile_{i}").mkdir()

        # Mark tile_0 seg done
        seg_dir = tmp_path / "stage1_segmentation" / "tile_0"
        seg_dir.mkdir(parents=True)
        (seg_dir / "SEGMENTATION_DONE").touch()

        # Mark tile_0 recon done
        recon_dir = tmp_path / "stage2_reconstruction" / "tile_0"
        recon_dir.mkdir(parents=True)
        (recon_dir / "tile_0_mask.tif").touch()

        args = argparse.Namespace(
            work_dir=str(tmp_path),
            seg_root=None,
            recon_root=None,
        )
        _cmd_status(args)
        captured = capsys.readouterr()
        assert "Total tiles: 3" in captured.out
        assert "Complete:  1/3" in captured.out

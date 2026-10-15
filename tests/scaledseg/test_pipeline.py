"""Tests for epicycle.core.pipeline module."""

import pytest
from pathlib import Path
from typing import List, Optional

from packages.epicycle_scaledseg.core.pipeline import (
    Stage,
    StageResult,
    PipelineResult,
    EpiCyclePipeline,
)


# --- Concrete test stage ---

class DummyStage(Stage):
    """A stage that always succeeds for testing."""

    def __init__(self, name="dummy", fail_tiles=None):
        super().__init__(name)
        self.fail_tiles = fail_tiles or []

    def validate_inputs(self, tile_dirs):
        return True

    def run(self, tile_dirs, n_gpus, log_dir=None):
        result = StageResult()
        for td in tile_dirs:
            if td.name in self.fail_tiles:
                result.add_failure(td.name, "intentional failure")
            else:
                result.add_success(td.name)
        return result


class FailValidationStage(Stage):
    def __init__(self):
        super().__init__("fail_validation")

    def validate_inputs(self, tile_dirs):
        return False

    def run(self, tile_dirs, n_gpus, log_dir=None):
        return StageResult()


# --- StageResult tests ---

class TestStageResult:
    def test_counts(self):
        r = StageResult()
        r.add_success("tile_0")
        r.add_success("tile_1")
        r.add_failure("tile_2", "oom")
        assert r.n_success == 2
        assert r.n_failed == 1
        assert r.n_total == 3

    def test_summary(self):
        r = StageResult()
        r.add_success("a")
        r.elapsed_seconds = 5.0
        assert "1/1" in r.summary()
        assert "5.0s" in r.summary()


class TestPipelineResult:
    def test_aggregation(self):
        pr = PipelineResult()
        s1 = StageResult()
        s1.add_success("a")
        s1.add_success("b")
        pr.add_stage_result("seg", s1)

        s2 = StageResult()
        s2.add_success("a")
        s2.add_failure("b", "err")
        pr.add_stage_result("recon", s2)

        assert pr.n_total == 2
        assert pr.n_success == 1  # last stage


# --- Pipeline tests ---

class TestEpiCyclePipeline:
    def test_empty_pipeline(self, tmp_path):
        p = EpiCyclePipeline(work_dir=str(tmp_path), n_gpus=1, log_dir=str(tmp_path / "logs"))
        result = p.run([])
        assert result.n_total == 0

    def test_single_stage_all_pass(self, tmp_path):
        t1 = tmp_path / "tile_0"
        t2 = tmp_path / "tile_1"
        t1.mkdir()
        t2.mkdir()

        p = EpiCyclePipeline(work_dir=str(tmp_path), n_gpus=1, log_dir=str(tmp_path / "logs"))
        p.add_stage(DummyStage())
        result = p.run([str(t1), str(t2)])
        assert result.n_success == 2

    def test_failure_stops_propagation(self, tmp_path):
        t1 = tmp_path / "tile_0"
        t2 = tmp_path / "tile_1"
        t1.mkdir()
        t2.mkdir()

        p = EpiCyclePipeline(work_dir=str(tmp_path), n_gpus=1, log_dir=str(tmp_path / "logs"))
        p.add_stage(DummyStage(name="seg", fail_tiles=["tile_1"]))
        p.add_stage(DummyStage(name="recon"))
        result = p.run([str(t1), str(t2)])

        # seg: 1 pass, 1 fail
        assert result.stage_results["seg"].n_success == 1
        assert result.stage_results["seg"].n_failed == 1
        # recon only sees tile_0
        assert result.stage_results["recon"].n_success == 1
        assert result.stage_results["recon"].n_total == 1

    def test_validation_failure(self, tmp_path):
        t1 = tmp_path / "tile_0"
        t1.mkdir()

        p = EpiCyclePipeline(work_dir=str(tmp_path), n_gpus=1, log_dir=str(tmp_path / "logs"))
        p.add_stage(FailValidationStage())
        result = p.run([str(t1)])
        assert result.stage_results["fail_validation"].n_failed == 1

    def test_repr(self, tmp_path):
        p = EpiCyclePipeline(work_dir=str(tmp_path), n_gpus=8, log_dir=str(tmp_path / "logs"))
        p.add_stage(DummyStage("seg"))
        assert "n_gpus=8" in repr(p)
        assert "seg" in repr(p)


# --- Mode-scoped checkpoint (Mode A coarse vs Mode B full) ---


class TestModeScopedCheckpoint:
    def test_default_mode_coarse(self, tmp_path):
        """Default mode is 'coarse' for back-compat."""
        p = EpiCyclePipeline(work_dir=str(tmp_path), n_gpus=1, log_dir=str(tmp_path / "logs"))
        assert p.mode == "coarse"

    def test_mode_scoped_checkpoint_path(self, tmp_path):
        """A coarse pipeline and a full pipeline use DIFFERENT checkpoint files."""
        coarse = EpiCyclePipeline(
            work_dir=str(tmp_path), n_gpus=1,
            log_dir=str(tmp_path / "logs"), mode="coarse",
        )
        full = EpiCyclePipeline(
            work_dir=str(tmp_path), n_gpus=1,
            log_dir=str(tmp_path / "logs"), mode="full",
        )
        assert coarse._checkpoint_path != full._checkpoint_path
        assert coarse._checkpoint_path.name == "pipeline_checkpoint_coarse.json"
        assert full._checkpoint_path.name == "pipeline_checkpoint_full.json"

    def test_coarse_checkpoint_does_not_skip_full(self, tmp_path):
        """A coarse run that completed a stage must NOT make a full run skip it.

        The mode-scoped checkpoint is the safety: a stale coarse checkpoint and a
        full run live in separate files, so resume can't cross-contaminate.
        """
        t1 = tmp_path / "tile_0"
        t1.mkdir()

        # Coarse run completes a stage named "seg".
        coarse = EpiCyclePipeline(
            work_dir=str(tmp_path), n_gpus=1,
            log_dir=str(tmp_path / "logs"), mode="coarse",
        )
        coarse.add_stage(DummyStage("seg"))
        coarse.run([str(t1)])
        assert coarse._checkpoint_path.exists()

        # A full run resuming sees NO completed stages (its checkpoint is empty).
        full = EpiCyclePipeline(
            work_dir=str(tmp_path), n_gpus=1,
            log_dir=str(tmp_path / "logs"), mode="full",
        )
        ckpt = full._load_checkpoint()
        assert "seg" not in set(ckpt.get("completed_stages", []))


# --- CLI argument parsing ---


class TestCliModeArg:
    def test_cli_mode_default_coarse(self):
        """The --mode argparse default is 'coarse'."""
        from packages.epicycle_scaledseg.cli.main import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "run", "--work-dir", ".",
            "--lateral-model", "x.pth", "--axial-model", "y.pth",
        ])
        assert ns.mode == "coarse"

    def test_cli_mode_full_accepted(self):
        from packages.epicycle_scaledseg.cli.main import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "run", "--work-dir", ".",
            "--lateral-model", "x.pth", "--axial-model", "y.pth",
            "--mode", "full",
        ])
        assert ns.mode == "full"

    def test_cli_iou_arg_defaults(self):
        """The --iou-* tuning args default to the documented values."""
        from packages.epicycle_scaledseg.cli.main import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "run", "--work-dir", ".",
            "--lateral-model", "x.pth", "--axial-model", "y.pth",
        ])
        assert ns.iou_tau == 0.25
        assert ns.iou_vmin == 10
        assert ns.iou_z_block == 64
        assert ns.iou_one_to_many == "max_iou"

    def test_cli_iou_stitch_in_stages_choices(self):
        """'iou_stitch' is an accepted --stages value."""
        from packages.epicycle_scaledseg.cli.main import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "run", "--work-dir", ".",
            "--lateral-model", "x.pth", "--axial-model", "y.pth",
            "--stages", "iou_stitch",
        ])
        assert "iou_stitch" in ns.stages

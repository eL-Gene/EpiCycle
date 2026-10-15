"""
Core pipeline abstractions for EpiCycle ScaledSegmentation.

Defines the Stage ABC, StageResult, PipelineResult, and the main
EpiCyclePipeline orchestrator.
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from packages.epicycle_scaledseg.utils.logging_utils import setup_logger, setup_stage_logger


def _filter_tile_paths(paths: "List[Path]") -> "List[Path]":
    """Remove hidden/system directories (e.g. .ipynb_checkpoints) from tile lists."""
    return [p for p in paths if not p.name.startswith(".")]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Collects per-tile success/failure info for a single pipeline stage."""

    successes: List[str] = field(default_factory=list)
    failures: List[Dict[str, str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    tile_dirs: Optional[List[Any]] = None  # set by TilingStage to update downstream tile paths

    def add_success(self, tile_name: str):
        self.successes.append(tile_name)

    def add_failure(self, tile_name: str, error: str):
        self.failures.append({"tile": tile_name, "error": error})

    @property
    def n_success(self) -> int:
        return len(self.successes)

    @property
    def n_failed(self) -> int:
        return len(self.failures)

    @property
    def n_total(self) -> int:
        return self.n_success + self.n_failed

    def summary(self) -> str:
        return (
            f"{self.n_success}/{self.n_total} succeeded "
            f"({self.elapsed_seconds:.1f}s)"
        )


@dataclass
class PipelineResult:
    """Aggregates StageResults across the full pipeline."""

    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def add_stage_result(self, stage_name: str, result: StageResult):
        self.stage_results[stage_name] = result

    @property
    def n_success(self) -> int:
        if not self.stage_results:
            return 0
        # success = tiles that passed ALL stages
        last = list(self.stage_results.values())[-1]
        return last.n_success

    @property
    def n_total(self) -> int:
        if not self.stage_results:
            return 0
        first = list(self.stage_results.values())[0]
        return first.n_total

    def summary(self) -> str:
        lines = [f"Pipeline: {self.n_success}/{self.n_total} tiles "
                 f"({self.elapsed_seconds:.1f}s total)"]
        for name, res in self.stage_results.items():
            lines.append(f"  {name}: {res.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage ABC
# ---------------------------------------------------------------------------

class Stage(ABC):
    """
    Abstract base class for pipeline stages.

    Subclasses must implement ``run`` and ``validate_inputs``.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(
        self,
        tile_dirs: List[Path],
        n_gpus: int,
        log_dir: Optional[Path] = None,
    ) -> StageResult:
        """Execute the stage on the given tile directories."""

    @abstractmethod
    def validate_inputs(self, tile_dirs: List[Path]) -> bool:
        """Return True if all required inputs are present."""

    def restore_outputs(self, log_dir: Optional[Path] = None) -> Optional[List[Path]]:
        """Re-derive, from disk, the tile directories this stage produced.

        Called when the stage is skipped on ``--resume``. Return ``None`` (default)
        if the stage creates no new tile directories; a stage that does must return
        them, or raise ``FileNotFoundError`` if its outputs are gone (the pipeline
        then reruns it).
        """
        return None

    def get_params(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of this stage's configuration."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EpiCyclePipeline:
    """
    Main pipeline orchestrator.

    Manages stages, resources, and sequential execution across tiles.

    Parameters
    ----------
    work_dir : root directory for data (tile directories live here)
    n_gpus : number of GPUs available on the node
    log_dir : directory for log files (created if missing)
    """

    def __init__(
        self,
        work_dir: str,
        n_gpus: int = 8,
        log_dir: str = "logs",
        mode: str = "coarse",
    ):
        self.work_dir = Path(work_dir).resolve()
        self.n_gpus = n_gpus
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Run mode: "coarse" (Mode A — downsample-then-re-segment assembly) or
        # "full" (Mode B — full-resolution IoU seam-stitch). The checkpoint is
        # mode-SCOPED so a stale coarse checkpoint can never make a full run skip
        # its work (and vice-versa) — the two resume universes are separate files.
        self.mode = mode
        self.stages: List[Stage] = []
        self.logger = logging.getLogger("epicycle.pipeline")

    def add_stage(self, stage: Stage):
        """Append a stage to the pipeline."""
        self.stages.append(stage)

    @property
    def _checkpoint_path(self) -> Path:
        return self.log_dir / f"pipeline_checkpoint_{self.mode}.json"

    def _load_checkpoint(self) -> dict:
        """Load checkpoint state from disk."""
        if self._checkpoint_path.exists():
            with open(self._checkpoint_path) as f:
                return json.load(f)
        return {"completed_stages": [], "failed_tiles": {}}

    def _save_checkpoint(self, checkpoint: dict):
        """Persist checkpoint state to disk."""
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def run(self, tile_dirs: List[str], resume: bool = False) -> PipelineResult:
        """
        Execute all stages in order.

        Parameters
        ----------
        tile_dirs : list of tile directory paths
        resume : if True, skip stages that fully completed in a previous run

        Returns
        -------
        PipelineResult with per-stage outcomes.
        """
        tile_paths = _filter_tile_paths([Path(d).resolve() for d in tile_dirs])
        result = PipelineResult()
        t0 = time.time()

        checkpoint = self._load_checkpoint() if resume else {
            "completed_stages": [], "failed_tiles": {},
        }
        completed_stages = set(checkpoint.get("completed_stages", []))

        # --- Save run parameters ---
        run_params = {
            "n_gpus": self.n_gpus,
            "mode": self.mode,
            "n_tiles": len(tile_paths),
            "stages": {s.name: s.get_params() for s in self.stages},
        }
        params_path = self.log_dir / "run_params.json"
        with open(params_path, "w") as f:
            json.dump(run_params, f, indent=2, default=str)

        # --- Set up main pipeline logger to file ---
        setup_stage_logger("pipeline", str(self.log_dir))

        self.logger.info(
            f"Pipeline starting: {len(self.stages)} stage(s), "
            f"{len(tile_paths)} tile(s), {self.n_gpus} GPU(s)"
            + (f" (resuming, {len(completed_stages)} stage(s) done)" if completed_stages else "")
        )
        self.logger.info(f"Run parameters saved to {params_path}")

        for stage in self.stages:
            self.logger.info(f"=== Stage: {stage.name} ===")

            # A stage with a ``feature_set`` (FeatureStage) that completed under ANOTHER
            # feature_set is not complete for this run (checkpoints that predate the gate
            # recorded no feature set: they were all "full").
            fs_now = getattr(stage, "feature_set", None)
            fs_done = checkpoint.get("feature_sets", {}).get(stage.name, "full")
            if stage.name in completed_stages and fs_now is not None and fs_done != fs_now:
                self.logger.info(
                    f"Stage '{stage.name}' completed under feature_set={fs_done!r}, this run "
                    f"is {fs_now!r} -- rerunning")
                completed_stages.discard(stage.name)
                checkpoint["completed_stages"] = list(completed_stages)
                self._save_checkpoint(checkpoint)

            # Skip fully-completed stages on resume -- but a stage that creates the
            # tile directories (tiling) must hand them on, so re-derive them from disk.
            restored = None
            if stage.name in completed_stages:
                try:
                    restored = stage.restore_outputs(self.log_dir)
                except FileNotFoundError as e:
                    self.logger.info(
                        f"Stage '{stage.name}' outputs missing ({e}) -- rerunning")
                    completed_stages.discard(stage.name)
                    checkpoint["completed_stages"] = list(completed_stages)
                    self._save_checkpoint(checkpoint)
            if stage.name in completed_stages:
                if restored is not None:
                    tile_paths = _filter_tile_paths([Path(d) for d in restored])
                self.logger.info(f"Stage '{stage.name}' already complete (checkpoint) — skipping")
                # Still need a StageResult so propagation works
                sr = StageResult()
                for td in tile_paths:
                    sr.add_success(td.name)
                result.add_stage_result(stage.name, sr)
                continue

            # Set up per-stage file logger
            stage_logger = setup_stage_logger(stage.name, str(self.log_dir))
            stage_logger.info(f"Stage '{stage.name}' starting with {len(tile_paths)} tile(s)")
            stage_logger.info(f"Parameters: {json.dumps(stage.get_params(), default=str)}")

            if not stage.validate_inputs(tile_paths):
                self.logger.error(
                    f"Input validation failed for stage '{stage.name}'"
                )
                sr = StageResult()
                for td in tile_paths:
                    sr.add_failure(td.name, "input validation failed")
                result.add_stage_result(stage.name, sr)
                break

            stage_t0 = time.time()
            sr = stage.run(tile_paths, self.n_gpus, log_dir=self.log_dir)
            sr.elapsed_seconds = time.time() - stage_t0

            result.add_stage_result(stage.name, sr)
            self.logger.info(f"Stage '{stage.name}': {sr.summary()}")
            stage_logger.info(f"Stage '{stage.name}' finished: {sr.summary()}")
            if sr.failures:
                for fail in sr.failures:
                    stage_logger.error(f"  Failed: {fail['tile']} — {fail['error']}")

            # If this stage produced new tile directories, use them downstream
            if sr.tile_dirs is not None:
                tile_paths = _filter_tile_paths([Path(d) for d in sr.tile_dirs])
                self.logger.info(
                    f"Stage '{stage.name}' produced {len(tile_paths)} tile directories"
                )

            # Record checkpoint if all tiles passed
            if sr.n_failed == 0:
                completed_stages.add(stage.name)
                checkpoint["completed_stages"] = sorted(completed_stages)
                if fs_now is not None:
                    checkpoint.setdefault("feature_sets", {})[stage.name] = fs_now
                self._save_checkpoint(checkpoint)

            # Propagate only successful tiles to the next stage
            tile_paths = [
                td for td in tile_paths if td.name in sr.successes
            ]
            if not tile_paths:
                self.logger.warning("No tiles remaining — stopping pipeline.")
                break

        result.elapsed_seconds = time.time() - t0
        self.logger.info(result.summary())
        return result

    def __repr__(self) -> str:
        stage_names = [s.name for s in self.stages]
        return (
            f"EpiCyclePipeline(work_dir={self.work_dir}, "
            f"n_gpus={self.n_gpus}, stages={stage_names})"
        )

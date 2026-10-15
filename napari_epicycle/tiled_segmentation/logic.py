from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tifffile

from napari_epicycle.domain import ScaledSegmentationJobSpec


class ScaledSegmentationBridgeService:
    def _new_job_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _infer_3d_shape(self, arr: np.ndarray) -> Tuple[int, int, int]:
        if arr.ndim == 2:
            return (1, int(arr.shape[0]), int(arr.shape[1]))
        if arr.ndim == 3:
            return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
        if arr.ndim == 4:
            if arr.shape[0] <= 4 and arr.shape[-1] > 4:
                return (int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]))
            if arr.shape[-1] <= 4 and arr.shape[0] > 4:
                return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
            return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
        return (16, 64, 64)

    def create_submission_bundle(self, *, working_dir: str, spec: ScaledSegmentationJobSpec) -> Dict[str, str]:
        spec.validate()
        job_id = self._new_job_id()
        out_dir = Path(working_dir) / "artifacts" / "scaled_segmentation" / "jobs" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        spec_path = out_dir / "job_spec.json"
        spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

        script_path = out_dir / "submit.slurm"
        script_path.write_text(self._render_slurm_script(spec=spec, spec_path=spec_path), encoding="utf-8")

        return {
            "job_id": job_id,
            "job_dir": str(out_dir),
            "spec_path": str(spec_path),
            "submit_script_path": str(script_path),
            "mode": "slurm",
        }

    def run_local_placeholder(
        self,
        *,
        working_dir: str,
        spec: ScaledSegmentationJobSpec,
        source_volume_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec.validate()
        job_id = self._new_job_id()
        out_dir = Path(working_dir) / "artifacts" / "scaled_segmentation" / "jobs" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = self._build_placeholder_labels(source_volume_path)
        labels_path = out_dir / "labels_3d.tif"
        tifffile.imwrite(str(labels_path), labels.astype(np.int32))

        manifest = {
            "job_id": job_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "local_placeholder",
            "source_volume_path": source_volume_path or "",
            "labels_path": str(labels_path),
            "labels_shape": [int(x) for x in labels.shape],
            "spec": spec.to_dict(),
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "job_id": job_id,
            "job_dir": str(out_dir),
            "manifest_path": str(manifest_path),
            "labels_path": str(labels_path),
            "mode": "local_placeholder",
        }

    def _build_placeholder_labels(self, source_volume_path: Optional[str]) -> np.ndarray:
        if not source_volume_path:
            return np.zeros((16, 64, 64), dtype=np.int32)
        path = Path(source_volume_path)
        if not path.exists():
            return np.zeros((16, 64, 64), dtype=np.int32)
        if path.suffix.lower() == ".npy":
            arr = np.asarray(np.load(path))
        else:
            arr = np.asarray(tifffile.imread(str(path)))
        zyx = self._infer_3d_shape(arr)
        return np.zeros(zyx, dtype=np.int32)

    def _render_slurm_script(self, *, spec: ScaledSegmentationJobSpec, spec_path: Path) -> str:
        params = dict(spec.scheduler_params or {})
        job_name = params.get("job_name", f"epicycle_scaledseg_{spec.round_id}")
        partition = params.get("partition", "gpu")
        gpus = int(params.get("gpus", 1))
        cpus = int(params.get("cpus_per_task", 8))
        mem = params.get("mem", "32G")
        time_limit = params.get("time", "08:00:00")
        output_log = params.get("output_log", "slurm-%j.out")

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --gres=gpu:{gpus}",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={output_log}",
            "",
            "set -euo pipefail",
            'echo "[EpiCycle] Starting scaled segmentation job"',
            f'echo "[EpiCycle] Spec: {spec_path}"',
            # This generated script is not used: the plugin's Step 5 panel is hidden and
            # `epicycle_scaledseg.run_job` does not exist. The engine runs from
            # tutorials/segmentation/01_run_whole_tissue_segmentation.ipynb or `epicycle-scaledseg run`.
            "python -m epicycle_scaledseg.run_job --spec \"${1:-" + str(spec_path) + "}\"",
        ]
        return "\n".join(lines) + "\n"

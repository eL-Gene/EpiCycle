from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tifffile

from packages.epicycle_domain import StackSegmentationJobSpec


class StackSegmentationService:
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

    def _load_shape(self, source_volume_path: Optional[str]) -> Tuple[int, int, int]:
        if not source_volume_path:
            return (16, 64, 64)
        path = Path(source_volume_path)
        if not path.exists():
            return (16, 64, 64)
        if path.suffix.lower() == ".npy":
            arr = np.asarray(np.load(path))
        else:
            arr = np.asarray(tifffile.imread(str(path)))
        return self._infer_3d_shape(arr)

    def run_placeholder(
        self,
        *,
        working_dir: str,
        spec: StackSegmentationJobSpec,
        source_volume_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec.validate()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path(working_dir) / "artifacts" / "segmentation" / "stack_runs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        shape = self._load_shape(source_volume_path)
        labels = np.zeros(shape, dtype=np.int32)
        labels_path = out_dir / "labels_3d.tif"
        tifffile.imwrite(str(labels_path), labels)

        manifest = {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "local_placeholder",
            "labels_path": str(labels_path),
            "labels_shape": [int(x) for x in shape],
            "spec": spec.to_dict(),
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "labels_path": str(labels_path),
        }

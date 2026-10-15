from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from packages.epicycle_domain import StackSegmentationJobSpec
from packages.epicycle_workflows.stack_segmentation import StackSegmentationService


def test_stack_segmentation_placeholder_run_outputs_manifest_and_labels(tmp_path):
    volume = tmp_path / "stack.tif"
    tifffile.imwrite(str(volume), np.random.default_rng(2).integers(0, 255, size=(8, 32, 24), dtype=np.uint8))

    model_pack = tmp_path / "final_model_pack.json"
    model_pack.write_text(json.dumps({"pack_id": "final_model_pack_x"}), encoding="utf-8")

    spec = StackSegmentationJobSpec(
        schema_version="1.0",
        input_volume_uri=str(volume),
        model_pack_uri=str(model_pack),
        output_target=str(tmp_path / "artifacts" / "segmentation" / "stack_runs"),
        run_mode="local",
    )
    svc = StackSegmentationService()
    result = svc.run_placeholder(working_dir=str(tmp_path), spec=spec, source_volume_path=str(volume))

    manifest_path = Path(result["manifest_path"])
    labels_path = Path(result["labels_path"])
    assert manifest_path.exists()
    assert labels_path.exists()

    labels = tifffile.imread(str(labels_path))
    assert labels.shape == (8, 32, 24)
    assert labels.dtype == np.int32

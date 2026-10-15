from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from napari_epicycle.domain import ScaledSegmentationJobSpec
from napari_epicycle.tiled_segmentation.logic import ScaledSegmentationBridgeService


def _spec(run_mode: str) -> ScaledSegmentationJobSpec:
    return ScaledSegmentationJobSpec(
        schema_version="1.0",
        round_id="round_03",
        input_volume_uri="stacks/stack_03.tif",
        model_artifact_uri="artifacts/models/model_xy_r03.json",
        output_target="artifacts/scaled_segmentation",
        tile_shape_zyx=[16, 128, 128],
        overlap_zyx=[2, 16, 16],
        run_mode=run_mode,
        scheduler_params={"partition": "gpu", "gpus": 1, "cpus_per_task": 4},
    )


def test_create_submission_bundle_writes_spec_and_slurm(tmp_path):
    svc = ScaledSegmentationBridgeService()
    result = svc.create_submission_bundle(working_dir=str(tmp_path), spec=_spec("slurm"))

    spec_path = Path(result["spec_path"])
    submit_path = Path(result["submit_script_path"])
    assert spec_path.exists()
    assert submit_path.exists()
    assert submit_path.read_text(encoding="utf-8").startswith("#!/bin/bash")
    assert "epicycle_scaledseg.run_job" in submit_path.read_text(encoding="utf-8")

    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    assert payload["run_mode"] == "slurm"
    assert payload["tile_shape_zyx"] == [16, 128, 128]


def test_run_local_placeholder_emits_manifest_and_labels(tmp_path):
    source = tmp_path / "stack.tif"
    tifffile.imwrite(str(source), np.random.default_rng(3).integers(0, 255, size=(6, 40, 50), dtype=np.uint8))

    spec = _spec("local")
    spec = ScaledSegmentationJobSpec(
        schema_version=spec.schema_version,
        round_id=spec.round_id,
        input_volume_uri=str(source),
        model_artifact_uri=spec.model_artifact_uri,
        output_target=spec.output_target,
        tile_shape_zyx=spec.tile_shape_zyx,
        overlap_zyx=spec.overlap_zyx,
        run_mode=spec.run_mode,
        scheduler_params=spec.scheduler_params,
    )
    svc = ScaledSegmentationBridgeService()
    result = svc.run_local_placeholder(working_dir=str(tmp_path), spec=spec, source_volume_path=str(source))

    manifest_path = Path(result["manifest_path"])
    labels_path = Path(result["labels_path"])
    assert manifest_path.exists()
    assert labels_path.exists()

    labels = tifffile.imread(str(labels_path))
    assert labels.shape == (6, 40, 50)
    assert labels.dtype == np.int32

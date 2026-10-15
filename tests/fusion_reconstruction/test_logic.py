from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from napari_epicycle.domain import ReconstructionJobSpec
from napari_epicycle.fusion_reconstruction.logic import ReconstructionPipelineService


def _spec(output_target: str) -> ReconstructionJobSpec:
    return ReconstructionJobSpec(
        schema_version="1.0",
        input_predictions=["predictions/round_01/xy.tif"],
        fusion_params={"dummy": True},
        output_target=output_target,
        run_mode="local",
    )


def test_placeholder_reconstruction_emits_manifest_and_labels(tmp_path):
    src = tmp_path / "stack.tif"
    tifffile.imwrite(str(src), np.random.default_rng(7).integers(0, 255, size=(5, 24, 32), dtype=np.uint8))

    svc = ReconstructionPipelineService()
    result = svc.run_placeholder(
        working_dir=str(tmp_path),
        spec=_spec(str(tmp_path / "artifacts" / "step3" / "runs")),
        source_stack_path=str(src),
    )

    manifest_path = Path(result["manifest_path"])
    labels_path = Path(result["labels_path"])
    assert manifest_path.exists()
    assert labels_path.exists()

    labels = tifffile.imread(str(labels_path))
    assert labels.shape == (5, 24, 32)
    assert labels.dtype == np.int32

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "placeholder"
    assert payload["spec"]["run_mode"] == "local"
    assert payload["labels_shape"] == [5, 24, 32]


def test_placeholder_reconstruction_handles_zyxc_input(tmp_path):
    src = tmp_path / "stack_zyxc.tif"
    data = np.random.default_rng(8).integers(0, 255, size=(7, 18, 20, 2), dtype=np.uint8)
    tifffile.imwrite(str(src), data)

    svc = ReconstructionPipelineService()
    result = svc.run_placeholder(
        working_dir=str(tmp_path),
        spec=_spec(str(tmp_path / "artifacts" / "step3" / "runs")),
        source_stack_path=str(src),
    )

    labels = tifffile.imread(result["labels_path"])
    assert labels.shape == (7, 18, 20)

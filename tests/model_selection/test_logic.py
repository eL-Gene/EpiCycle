from __future__ import annotations

import json

from packages.epicycle_domain import ModelArtifact
from packages.epicycle_workflows.final_model import FinalModelRegistryService


def test_final_model_registry_lists_models_and_registers_pack(tmp_path):
    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    m1 = ModelArtifact(
        schema_version="1.0",
        model_id="model_spatial_r03",
        view_type="spatial",
        checkpoint_uri="models/r03/spatial.ckpt",
        training_data_ref="round_03",
    )
    m2 = ModelArtifact(
        schema_version="1.0",
        model_id="model_axial_r03",
        view_type="axial",
        checkpoint_uri="models/r03/axial.ckpt",
        training_data_ref="round_03",
    )
    (models_dir / "m1.json").write_text(json.dumps(m1.to_dict(), indent=2), encoding="utf-8")
    (models_dir / "m2.json").write_text(json.dumps(m2.to_dict(), indent=2), encoding="utf-8")

    bench_dir = tmp_path / "artifacts" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "round_benchmark_summary.json").write_text(
        json.dumps({"num_rounds": 3, "best_round": "round_03", "best_final_loss": 0.12}, indent=2),
        encoding="utf-8",
    )

    svc = FinalModelRegistryService()
    models = svc.list_workspace_models(str(tmp_path))
    assert len(models) == 2

    summary = svc.load_benchmark_summary(str(tmp_path))
    assert summary is not None
    assert summary["best_round"] == "round_03"

    result = svc.register_final_model_pack(
        working_dir=str(tmp_path),
        lateral_model_id="model_spatial_r03",
        axial_model_id="model_axial_r03",
        imported_model_uris=["C:/models/external_model.json"],
        notes="approved",
    )
    assert result["pack_id"].startswith("final_model_pack_")
    assert (tmp_path / "artifacts" / "model_registry" / "final_model_pack.json").exists()

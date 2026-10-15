from napari_epicycle.domain import (
    InferenceArtifact,
    ModelArtifact,
    ReconstructionJobSpec,
    ScaledSegmentationJobSpec,
    StackSegmentationJobSpec,
    RoundMetadata,
    SelectedIndicesRecord,
    SliceAnnotationRecord,
)
from napari_epicycle.io_registry import FileArtifactRegistry


def _sample(round_id: str = "round_01") -> SliceAnnotationRecord:
    return SliceAnnotationRecord(
        schema_version="1.0",
        dataset_id="dataset_a",
        stack_id="stack_01",
        view="XY",
        slice_index=12,
        channel_map={"membrane": 0},
        img_uri="training_data/round_01/spatial/stack_01_XY_z0012_img.tif",
        mask_uri="training_data/round_01/spatial/stack_01_XY_z0012_mask.tif",
        round_id=round_id,
    )


def test_round_metadata_registry_roundtrip(tmp_path):
    registry = FileArtifactRegistry()
    metadata = RoundMetadata(
        schema_version="1.0",
        round_id="round_01",
        created_at="2026-02-20T00:00:00+00:00",
        config_snapshot={"epochs": 100},
        samples=[_sample()],
    )

    out_path = registry.save_round_metadata(str(tmp_path), metadata)
    assert out_path.exists()

    loaded = registry.load_round_metadata(str(tmp_path), "round_01")
    assert loaded is not None
    assert loaded.round_id == "round_01"
    assert loaded.samples[0].stack_id == "stack_01"


def test_model_and_inference_registry_roundtrip(tmp_path):
    registry = FileArtifactRegistry()

    model = ModelArtifact(
        schema_version="1.0",
        model_id="model_xy_r01",
        view_type="spatial",
        checkpoint_uri="models/model_xy_r01.pt",
        training_data_ref="round_01",
        metrics={"ap50": 0.8},
    )
    model_path = registry.save_model_artifact(str(tmp_path), model)
    assert model_path.exists()
    loaded_model = registry.load_model_artifact(str(tmp_path), "model_xy_r01")
    assert loaded_model is not None
    assert loaded_model.training_data_ref == "round_01"

    inference = InferenceArtifact(
        schema_version="1.0",
        model_id="model_xy_r01",
        dataset_id="dataset_a",
        view_type="XY",
        prediction_uri="predictions/r01/xy",
    )
    inf_path = registry.save_inference_artifact(str(tmp_path), inference)
    assert inf_path.exists()
    loaded_inf = registry.load_inference_artifact(str(tmp_path), "model_xy_r01", "XY")
    assert loaded_inf is not None
    assert loaded_inf.prediction_uri == "predictions/r01/xy"


def test_reconstruction_job_spec_is_persisted(tmp_path):
    registry = FileArtifactRegistry()
    spec = ReconstructionJobSpec(
        schema_version="1.0",
        input_predictions=["predictions/r01/xy", "predictions/r01/xz"],
        fusion_params={"method": "uSegment3D"},
        output_target="reconstructions/r01",
        run_mode="local",
    )
    spec_path = registry.save_reconstruction_job_spec(str(tmp_path), spec)
    assert spec_path.exists()


def test_scaled_segmentation_job_spec_is_persisted(tmp_path):
    registry = FileArtifactRegistry()
    spec = ScaledSegmentationJobSpec(
        schema_version="1.0",
        round_id="round_05",
        input_volume_uri="stacks/stack_01.tif",
        model_artifact_uri="artifacts/models/model_xy_r05.json",
        output_target="artifacts/scaled_segmentation",
        tile_shape_zyx=[32, 256, 256],
        overlap_zyx=[4, 32, 32],
        run_mode="slurm",
    )
    spec_path = registry.save_scaled_segmentation_job_spec(str(tmp_path), spec)
    assert spec_path.exists()


def test_selected_indices_roundtrip(tmp_path):
    registry = FileArtifactRegistry()
    record = SelectedIndicesRecord(
        schema_version="1.0",
        round_id="round_01",
        selections={"XY": {"stack01": [12, 25]}, "XZ": {"stack01": [30]}, "YZ": {}},
    )
    path = registry.save_selected_indices(str(tmp_path), record)
    assert path.exists()
    loaded = registry.load_selected_indices(str(tmp_path), "round_01")
    assert loaded is not None
    assert loaded.selections["XY"]["stack01"] == [12, 25]


def test_stack_segmentation_job_spec_is_persisted(tmp_path):
    registry = FileArtifactRegistry()
    spec = StackSegmentationJobSpec(
        schema_version="1.0",
        input_volume_uri="stacks/stack_01.tif",
        model_pack_uri="artifacts/model_registry/final_model_pack.json",
        output_target="artifacts/segmentation/stack_runs",
        run_mode="local",
    )
    spec_path = registry.save_stack_segmentation_job_spec(str(tmp_path), spec)
    assert spec_path.exists()

import pytest

from napari_epicycle.domain import (
    CurationSessionRecord,
    InferenceArtifact,
    ModelArtifact,
    ReconstructionJobSpec,
    ScaledSegmentationJobSpec,
    StackSegmentationJobSpec,
    RoundMetadata,
    SelectedIndicesRecord,
    SliceAnnotationRecord,
)


def test_slice_annotation_roundtrip():
    record = SliceAnnotationRecord(
        schema_version="1.0",
        dataset_id="dataset_a",
        stack_id="stack_01",
        view="XY",
        slice_index=12,
        channel_map={"membrane": 0, "nuclei": 1},
        img_uri="training/round_01/spatial/stack_01_XY_z0012_img.tif",
        mask_uri="training/round_01/spatial/stack_01_XY_z0012_mask.tif",
        round_id="round_01",
    )
    payload = record.to_dict()
    restored = SliceAnnotationRecord.from_dict(payload)
    assert restored == record


def test_slice_annotation_invalid_view_raises():
    with pytest.raises(ValueError, match="Unsupported view"):
        SliceAnnotationRecord(
            schema_version="1.0",
            dataset_id="dataset_a",
            stack_id="stack_01",
            view="ZZ",
            slice_index=1,
            channel_map={},
            img_uri="x",
            mask_uri="y",
            round_id="round_01",
        ).validate()


def test_round_metadata_serialization_with_samples():
    sample = SliceAnnotationRecord(
        schema_version="1.0",
        dataset_id="dataset_a",
        stack_id="stack_01",
        view="XZ",
        slice_index=3,
        channel_map={},
        img_uri="img.tif",
        mask_uri="mask.tif",
        round_id="round_02",
    )
    metadata = RoundMetadata(
        schema_version="1.0",
        round_id="round_02",
        created_at="2026-02-20T00:00:00+00:00",
        config_snapshot={"epochs": 100},
        samples=[sample],
    )
    restored = RoundMetadata.from_dict(metadata.to_dict())
    assert restored.round_id == "round_02"
    assert len(restored.samples) == 1
    assert restored.samples[0].view == "XZ"


def test_model_artifact_validates_view_type():
    artifact = ModelArtifact(
        schema_version="1.0",
        model_id="model_axial_r02",
        view_type="axial",
        checkpoint_uri="models/round_02/axial.pt",
        training_data_ref="round_02",
        metrics={"ap50": 0.73},
    )
    assert artifact.to_dict()["metrics"]["ap50"] == 0.73

    with pytest.raises(ValueError, match="Unsupported view_type"):
        ModelArtifact(
            schema_version="1.0",
            model_id="bad",
            view_type="3d",
            checkpoint_uri="x",
            training_data_ref="round_00",
        ).validate()


def test_inference_and_reconstruction_spec_validation():
    inference = InferenceArtifact(
        schema_version="1.0",
        model_id="model_xy_r03",
        dataset_id="dataset_a",
        view_type="XY",
        prediction_uri="predictions/r03/xy",
    )
    assert inference.to_dict()["dataset_id"] == "dataset_a"

    spec = ReconstructionJobSpec(
        schema_version="1.0",
        input_predictions=["predictions/r03/xy", "predictions/r03/xz"],
        fusion_params={"method": "uSegment3D"},
        output_target="reconstructions/r03",
        run_mode="local",
    )
    assert spec.to_dict()["run_mode"] == "local"

    with pytest.raises(ValueError, match="at least one prediction path"):
        ReconstructionJobSpec(
            schema_version="1.0",
            input_predictions=[],
            fusion_params={},
            output_target="x",
            run_mode="local",
        ).validate()


def test_curation_session_requires_workflow_chain():
    session = CurationSessionRecord(
        schema_version="1.0",
        source_artifact="reconstructions/r03/labels.tif",
        workflows_used=["histogram_roi", "scribble"],
        handoff_chain=["histogram_roi->scribble"],
        final_label_id_map={"17": 17},
    )
    assert session.to_dict()["workflows_used"] == ["histogram_roi", "scribble"]

    with pytest.raises(ValueError, match="must not be empty"):
        CurationSessionRecord(
            schema_version="1.0",
            source_artifact="x",
            workflows_used=[],
            handoff_chain=[],
        ).validate()


def test_scaled_segmentation_job_spec_validation_and_roundtrip():
    spec = ScaledSegmentationJobSpec(
        schema_version="1.0",
        round_id="round_04",
        input_volume_uri="stacks/stack_01.tif",
        model_artifact_uri="artifacts/models/model_xy_r04.json",
        output_target="artifacts/scaled_segmentation",
        tile_shape_zyx=[32, 256, 256],
        overlap_zyx=[4, 32, 32],
        run_mode="slurm",
        scheduler_params={"partition": "gpu", "gpus": 1},
    )
    restored = ScaledSegmentationJobSpec.from_dict(spec.to_dict())
    assert restored.run_mode == "slurm"
    assert restored.tile_shape_zyx == [32, 256, 256]

    with pytest.raises(ValueError, match="tile_shape_zyx"):
        ScaledSegmentationJobSpec(
            schema_version="1.0",
            round_id="round_04",
            input_volume_uri="x.tif",
            model_artifact_uri="m.json",
            output_target="out",
            tile_shape_zyx=[0, 128, 128],
            overlap_zyx=[0, 8, 8],
            run_mode="local",
        ).validate()


def test_selected_indices_record_roundtrip():
    record = SelectedIndicesRecord(
        schema_version="1.0",
        round_id="round_02",
        selections={
            "XY": {"stack01": [12, 25, 25]},
            "XZ": {"stack01": [30]},
            "YZ": {"stack02": [9]},
        },
    )
    restored = SelectedIndicesRecord.from_dict(record.to_dict())
    assert restored.selections["XY"]["stack01"] == [12, 25]
    assert restored.selections["YZ"]["stack02"] == [9]


def test_stack_segmentation_job_spec_validation_and_roundtrip():
    spec = StackSegmentationJobSpec(
        schema_version="1.0",
        input_volume_uri="stacks/stack_01.tif",
        model_pack_uri="artifacts/model_registry/final_model_pack.json",
        output_target="artifacts/segmentation/stack_runs",
        run_mode="local",
        params={"run_name": "roi01_apply"},
    )
    restored = StackSegmentationJobSpec.from_dict(spec.to_dict())
    assert restored.run_mode == "local"
    assert restored.params["run_name"] == "roi01_apply"

    with pytest.raises(ValueError, match="Unsupported run_mode"):
        StackSegmentationJobSpec(
            schema_version="1.0",
            input_volume_uri="x.tif",
            model_pack_uri="m.json",
            output_target="out",
            run_mode="slurm",
        ).validate()

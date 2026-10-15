import json

from napari_epicycle.io_registry import HITLPersistenceService


def test_hitl_persistence_save_and_load_roundtrip(tmp_path):
    service = HITLPersistenceService()
    state = {
        "current_round": 2,
        "training_data_by_round": {
            2: [
                {
                    "dataset_id": "dataset_a",
                    "stack_id": "stack_03",
                    "view": "xy",
                    "slice_index": 20,
                    "channel_map": {"membrane": 0},
                    "img_uri": "training_data/round_02/spatial/stack_03_XY_z0020_img.tif",
                    "mask_uri": "training_data/round_02/spatial/stack_03_XY_z0020_mask.tif",
                }
            ]
        },
        "losses_by_round": {},
        "model_checkpoint_paths": {},
        "current_config": {"epochs": 100},
        "selected_view": "XY",
    }

    state_path, metadata_path = service.save_hitl_state(
        working_dir=str(tmp_path),
        state_dict=state,
        current_round=2,
        current_config=state["current_config"],
        training_data_by_round=state["training_data_by_round"],
    )
    assert state_path.exists()
    assert metadata_path.exists()
    assert (tmp_path / "training_data" / "round_02" / "selected_indices.json").exists()

    loaded_state, loaded_metadata = service.load_hitl_state(str(tmp_path))
    assert loaded_state is not None
    assert loaded_state["current_round"] == 2
    assert loaded_metadata is not None
    assert loaded_metadata.round_id == "round_02"
    assert loaded_metadata.samples[0].view == "XY"


def test_hitl_persistence_loads_legacy_round_metadata_and_normalizes(tmp_path):
    service = HITLPersistenceService()

    # Save a state that points to round_01
    service.state_store.save_state(
        str(tmp_path),
        {
            "current_round": 1,
            "current_config": {},
        },
    )

    # Write legacy round metadata payload (without schema_version)
    legacy_round_dir = tmp_path / "training_data" / "round_01"
    legacy_round_dir.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        "round": 1,
        "date": "2026-01-28",
        "samples": [
            {
                "stack_id": "stack01",
                "view": "xy",
                "slice_index": 12,
                "img_path": "training_data/round_01/spatial/stack01_XY_z0012_img.tif",
                "mask_path": "training_data/round_01/spatial/stack01_XY_z0012_mask.tif",
            }
        ],
    }
    legacy_path = legacy_round_dir / "round_metadata.json"
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    _, metadata = service.load_hitl_state(str(tmp_path))
    assert metadata is not None
    assert metadata.round_id == "round_01"
    assert metadata.samples[0].img_uri.endswith("_img.tif")

    # Ensure file got normalized to canonical schema (contains schema_version now)
    normalized = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert normalized["schema_version"] == "1.0"

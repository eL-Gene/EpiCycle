from napari_epicycle.domain import round_metadata_from_legacy


def test_legacy_round_metadata_conversion_with_round_int_and_paths():
    legacy_payload = {
        "round": 1,
        "date": "2026-01-28",
        "mode": "both",
        "channels": [2, 1],
        "normalize": True,
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.4,
        "training_config": {"epochs": 500, "learning_rate": 0.01},
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

    metadata = round_metadata_from_legacy(legacy_payload)
    assert metadata.round_id == "round_01"
    assert len(metadata.samples) == 1
    assert metadata.samples[0].view == "XY"
    assert metadata.samples[0].img_uri.endswith("_img.tif")
    assert metadata.config_snapshot["flow_threshold"] == 0.4


def test_legacy_round_metadata_conversion_with_round_id_and_uris():
    payload = {
        "round_id": "round_03",
        "created_at": "2026-02-20T00:00:00+00:00",
        "config_snapshot": {"epochs": 300},
        "samples": [
            {
                "dataset_id": "dataset_a",
                "stack_id": "stack02",
                "view": "YZ",
                "slice_index": 9,
                "img_uri": "x_img.tif",
                "mask_uri": "x_mask.tif",
                "channel_map": {"membrane": 0},
            }
        ],
    }
    metadata = round_metadata_from_legacy(payload)
    assert metadata.round_id == "round_03"
    assert metadata.samples[0].dataset_id == "dataset_a"
    assert metadata.config_snapshot == {"epochs": 300}

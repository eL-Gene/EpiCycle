import json

from napari_epicycle.io_registry import migrate_training_data_tree


def test_migration_converts_legacy_round_metadata_and_writes_selected_indices(tmp_path):
    round_dir = tmp_path / "training_data" / "round_01"
    round_dir.mkdir(parents=True, exist_ok=True)
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
            },
            {
                "stack_id": "stack01",
                "view": "xy",
                "slice_index": 25,
                "img_path": "training_data/round_01/spatial/stack01_XY_z0025_img.tif",
                "mask_path": "training_data/round_01/spatial/stack01_XY_z0025_mask.tif",
            },
        ],
    }
    (round_dir / "round_metadata.json").write_text(json.dumps(legacy_payload, indent=2), encoding="utf-8")

    summaries = migrate_training_data_tree(str(tmp_path))
    assert len(summaries) == 1
    assert summaries[0]["round_id"] == "round_01"

    canonical = json.loads((round_dir / "round_metadata.json").read_text(encoding="utf-8"))
    assert canonical["schema_version"] == "1.0"
    assert canonical["round_id"] == "round_01"

    selected = json.loads((round_dir / "selected_indices.json").read_text(encoding="utf-8"))
    assert selected["schema_version"] == "1.0"
    assert selected["selections"]["XY"]["stack01"] == [12, 25]

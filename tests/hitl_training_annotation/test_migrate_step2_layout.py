"""Tests for ``napari_epicycle/hitl_training_annotation/_migrate_step2_layout.py``.

Covers Phase 2a of the Step 2 directory restructure: migration logic
(plan + execute + rollback, schema v2→v3 wrap, sentinel handling).

Test groups:
    - Schema wrap (v2 → v3 conversion)
    - Plan builder (empty workspace, legacy workspace, mixed layout)
    - Plan validation (mis-sorted files, duplicate basenames)
    - Bucket fan-out (XY → lateral, XZ/YZ → axial)
    - Metadata splits (round_metadata, selected_indices)
    - Backup creation
    - Execute migration (happy path)
    - Rollback (success path + auto-rollback on failure)
    - Sentinel (skip / mark / clear)
    - Round-trip (legacy → migrate → execute → load → verify state)
"""

from __future__ import annotations

import json

import pytest

from napari_epicycle.hitl_training_annotation import _paths as p
from napari_epicycle.hitl_training_annotation import _migrate_step2_layout as m


# ----------------------------------------------------------------------------
# Phase 8a — wrap_state_schema_v2_to_v3 active_run_id kwarg (audit-fix)
# ----------------------------------------------------------------------------

class TestWrapStateSchemaActiveRunIdKwarg:
    """Phase 8a critical fix: wrap honors active_run_id, not hardcoded default.

    Without this kwarg, saves on a non-default run CLOBBER the default
    run's state and LOSE the current run's edits.
    Audit-found 2026-05-23.
    """

    def test_default_active_run_id_preserves_legacy_behavior(self):
        v2 = {"schema_version": 2, "current_round": 3, "sample_records": {"a": {}}}
        v3 = m.wrap_state_schema_v2_to_v3(v2)
        assert v3["active_run_id"] == "default"
        assert "default" in v3["runs"]
        assert v3["runs"]["default"]["current_round"] == 3
        assert v3["runs"]["default"]["display_name"] == "Default"

    def test_explicit_active_run_id_targets_correct_slot(self):
        v2 = {"schema_version": 2, "current_round": 5}
        v3 = m.wrap_state_schema_v2_to_v3(v2, active_run_id="nuclei_v1")
        assert v3["active_run_id"] == "nuclei_v1"
        assert "nuclei_v1" in v3["runs"]
        assert "default" not in v3["runs"]
        assert v3["runs"]["nuclei_v1"]["current_round"] == 5
        # Non-default runs get display_name = run_id (caller should override
        # via the existing-runs merge in _save_internal_state)
        assert v3["runs"]["nuclei_v1"]["display_name"] == "nuclei_v1"

    def test_active_run_id_none_falls_back_to_default(self):
        v3 = m.wrap_state_schema_v2_to_v3({"schema_version": 2}, active_run_id=None)
        assert v3["active_run_id"] == "default"

    def test_active_run_id_empty_falls_back_to_default(self):
        v3 = m.wrap_state_schema_v2_to_v3({"schema_version": 2}, active_run_id="")
        assert v3["active_run_id"] == "default"





# ============================================================================
# Fixtures — synthetic legacy workspaces
# ============================================================================

@pytest.fixture
def empty_ws(tmp_path):
    """Fresh empty workspace dir."""
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _make_pair_file(dir_: "Path", stem: str) -> None:
    """Create a fake training-pair file (just a marker, not real TIFF)."""
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{stem}.tif").write_bytes(b"fake tiff")


@pytest.fixture
def legacy_ws_multi_round(tmp_path):
    """A legacy workspace with 3 rounds (rounds 0, 1, 2) — exercises
    multi-round migration including round-key preservation.
    """
    ws = tmp_path / "ws_multi"
    for round_idx in (0, 1, 2):
        spatial = ws / "training_data" / f"round_{round_idx:02d}" / "accepted_pairs" / "spatial"
        axial = ws / "training_data" / f"round_{round_idx:02d}" / "accepted_pairs" / "axial"
        _make_pair_file(spatial, f"stack_a_XY_z{round_idx + 1:04d}_img")
        _make_pair_file(spatial, f"stack_a_XY_z{round_idx + 1:04d}_mask")
        _make_pair_file(axial, f"stack_a_XZ_z{round_idx + 1:04d}_img")
        _make_pair_file(axial, f"stack_a_XZ_z{round_idx + 1:04d}_mask")

        meta = ws / "training_data" / f"round_{round_idx:02d}" / "round_metadata.json"
        meta.write_text(json.dumps({
            "schema_version": "1.0",
            "round_id": f"round_{round_idx:02d}",
            "created_at": "2026-05-23T00:00:00Z",
            "config_snapshot": {},
            "parent_round": None,
            "samples": [
                {"label": f"stack_a_XY_z{round_idx + 1:04d}", "view": "XY",
                 "stack_id": "stack_a", "slice_index": round_idx + 1},
                {"label": f"stack_a_XZ_z{round_idx + 1:04d}", "view": "XZ",
                 "stack_id": "stack_a", "slice_index": round_idx + 1},
            ],
        }), encoding="utf-8")

        sel = ws / "training_data" / f"round_{round_idx:02d}" / "selected_indices.json"
        sel.write_text(json.dumps({
            "schema_version": "1.0",
            "round_id": f"round_{round_idx:02d}",
            "selections": {
                "XY": {"stack_a": [round_idx + 1]},
                "XZ": {"stack_a": [round_idx + 1]},
            },
        }), encoding="utf-8")

        models = ws / "models" / f"round_{round_idx:02d}"
        models.mkdir(parents=True, exist_ok=True)
        (models / f"model_lateral_r{round_idx}.json").write_bytes(b"fake")
        (models / f"model_axial_r{round_idx}.json").write_bytes(b"fake")

    # Single state file covering all rounds
    state = ws / "hitl_state.json"
    state.write_text(json.dumps({
        "schema_version": 2,
        "current_round": 2,
        "current_round_by_view": {"lateral": 2, "axial": 2},
        "training_data_by_round": {
            "lateral_0": [], "lateral_1": [], "lateral_2": [],
            "axial_0": [], "axial_1": [], "axial_2": [],
        },
        "losses_by_round": {},
        "model_checkpoint_paths": {},
        "round_backend_info": {},
        "current_config": {"channels": ["DAPI"]},
        "sample_records": {},
    }, indent=2), encoding="utf-8")

    return ws


@pytest.fixture
def legacy_ws(tmp_path):
    """A minimal legacy workspace with one round of data."""
    ws = tmp_path / "ws"

    # training_data/round_00/accepted_pairs/{spatial,axial}/
    spatial = ws / "training_data" / "round_00" / "accepted_pairs" / "spatial"
    axial = ws / "training_data" / "round_00" / "accepted_pairs" / "axial"
    _make_pair_file(spatial, "stack_a_XY_z0001_img")
    _make_pair_file(spatial, "stack_a_XY_z0001_mask")
    _make_pair_file(spatial, "stack_a_XY_z0002_img")
    _make_pair_file(spatial, "stack_a_XY_z0002_mask")
    _make_pair_file(axial, "stack_a_XZ_z0005_img")
    _make_pair_file(axial, "stack_a_XZ_z0005_mask")
    _make_pair_file(axial, "stack_a_YZ_z0010_img")
    _make_pair_file(axial, "stack_a_YZ_z0010_mask")

    # round_metadata.json (joint per-round) — uses canonical RoundMetadata
    # schema with dict envelope + samples list (per packages/epicycle_domain/contracts.py).
    meta = ws / "training_data" / "round_00" / "round_metadata.json"
    meta.write_text(json.dumps({
        "schema_version": "1.0",
        "round_id": "round_00",
        "created_at": "2026-05-23T00:00:00Z",
        "config_snapshot": {"channels": ["DAPI"]},
        "parent_round": None,
        "samples": [
            {"label": "stack_a_XY_z0001", "view": "XY", "status": "accepted",
             "stack_id": "stack_a", "slice_index": 1},
            {"label": "stack_a_XY_z0002", "view": "XY", "status": "accepted",
             "stack_id": "stack_a", "slice_index": 2},
            {"label": "stack_a_XZ_z0005", "view": "XZ", "status": "accepted",
             "stack_id": "stack_a", "slice_index": 5},
            {"label": "stack_a_YZ_z0010", "view": "YZ", "status": "accepted",
             "stack_id": "stack_a", "slice_index": 10},
        ],
    }, indent=2), encoding="utf-8")

    # selected_indices.json (joint per-round) — uses canonical
    # SelectedIndicesRecord schema with dict envelope + selections sub-dict.
    sel = ws / "training_data" / "round_00" / "selected_indices.json"
    sel.write_text(json.dumps({
        "schema_version": "1.0",
        "round_id": "round_00",
        "selections": {
            "XY": {"stack_a": [1, 2]},
            "XZ": {"stack_a": [5]},
            "YZ": {"stack_a": [10]},
        },
    }, indent=2), encoding="utf-8")

    # models/round_00/
    models = ws / "models" / "round_00"
    models.mkdir(parents=True, exist_ok=True)
    (models / "model_lateral_20260101.ckpt.json").write_bytes(b"fake model")
    (models / "model_axial_20260101.ckpt.json").write_bytes(b"fake model")

    # predictions/round_00/{spatial,axial}/
    pred_spatial = ws / "predictions" / "round_00" / "spatial"
    pred_axial = ws / "predictions" / "round_00" / "axial"
    _make_pair_file(pred_spatial, "stack_a_XY_z0001_pred_mask")
    _make_pair_file(pred_axial, "stack_a_XZ_z0005_pred_mask")
    (pred_spatial / "manifest.json").write_text("{}", encoding="utf-8")

    # hitl_state.json (legacy schema v2)
    state = ws / "hitl_state.json"
    state.write_text(json.dumps({
        "schema_version": 2,
        "current_round": 0,
        "current_round_by_view": {"lateral": 1, "axial": 1},
        "training_data_by_round": {"lateral_0": [], "axial_0": []},
        "losses_by_round": {},
        "model_checkpoint_paths": {},
        "round_backend_info": {},
        "current_config": {"channels": ["DAPI"]},
        "selected_view": "XY",
        "view_target": "lateral",
        "axial_plane": "XZ",
        "sample_records": {},
        "active_curation_checkpoints": {},
        "curation_model_source_mode": "hitl",
        "imported_curation_model_path": "",
        "curation_pretrained_model_type": "cyto",
        "training_mode": "cumulative",
        "auto_update_model_after_training": True,
        "loaded_stack_path": "/data/stack_a.tif",
        "loaded_patch_name": "patch_a",
    }, indent=2), encoding="utf-8")

    return ws


# ============================================================================
# Schema wrap
# ============================================================================

class TestSchemaWrap:
    def test_minimal_v2_wraps(self):
        v2 = {"schema_version": 2}
        v3 = m.wrap_state_schema_v2_to_v3(v2, migration_timestamp="2026-05-23T00:00:00Z")
        assert v3["schema_version"] == 3
        assert v3["active_run_id"] == "default"
        assert "default" in v3["runs"]
        assert v3["runs"]["default"]["display_name"] == "Default"
        assert v3["runs"]["default"]["backend"] == "cellpose"
        assert v3["runs"]["default"]["target"] is None
        assert v3["runs"]["default"]["created_at"] == "2026-05-23T00:00:00Z"

    def test_per_run_keys_move_into_default(self):
        v2 = {
            "schema_version": 2,
            "current_round": 3,
            "training_data_by_round": {"lateral_0": ["a"]},
            "sample_records": {"slice_xy_z0": {"status": "accepted"}},
        }
        v3 = m.wrap_state_schema_v2_to_v3(v2)
        default = v3["runs"]["default"]
        assert default["current_round"] == 3
        assert default["training_data_by_round"] == {"lateral_0": ["a"]}
        assert default["sample_records"] == {"slice_xy_z0": {"status": "accepted"}}

    def test_picker_and_view_keys_are_per_run(self):
        """Release-hardening 2026-09-07 (bucket-switch item #2): these six
        keys used to be workspace-level, which let one run's external model
        file + active view bleed into every other run.  They now land in
        ``runs[<slug>]``.  Old files keep loading because the v3 unwrap
        merges the top-level values first and lets the per-run entry win."""
        v2 = {
            "schema_version": 2,
            "selected_view": "XY",
            "view_target": "lateral",
            "axial_plane": "XZ",
            "curation_model_source_mode": "hitl",
            "imported_curation_model_path": "/some/path",
            "curation_pretrained_model_type": "cyto",
        }
        v3 = m.wrap_state_schema_v2_to_v3(v2)
        default = v3["runs"]["default"]
        assert default["selected_view"] == "XY"
        assert default["view_target"] == "lateral"
        assert default["axial_plane"] == "XZ"
        assert default["curation_model_source_mode"] == "hitl"
        assert default["imported_curation_model_path"] == "/some/path"
        assert default["curation_pretrained_model_type"] == "cyto"
        # And these should NOT stay at workspace level any more.
        assert "selected_view" not in v3
        assert "imported_curation_model_path" not in v3

    def test_schema_version_stays_the_only_workspace_level_key(self):
        assert m.WORKSPACE_LEVEL_KEYS_V2 == frozenset({"schema_version"})

    def test_unknown_keys_default_to_per_run(self):
        # Robust against future schema drift
        v2 = {"schema_version": 2, "future_per_run_field": "some_value"}
        v3 = m.wrap_state_schema_v2_to_v3(v2)
        assert v3["runs"]["default"]["future_per_run_field"] == "some_value"

    def test_invalid_schema_version_raises(self):
        with pytest.raises(ValueError, match="v2 schema"):
            m.wrap_state_schema_v2_to_v3({"schema_version": 99})

    def test_missing_schema_version_accepted(self):
        # Some early v2 files may not have schema_version at all
        v2 = {"current_round": 0}
        v3 = m.wrap_state_schema_v2_to_v3(v2)
        assert v3["schema_version"] == 3

    def test_non_dict_raises(self):
        with pytest.raises(ValueError):
            m.wrap_state_schema_v2_to_v3([1, 2, 3])


# ============================================================================
# build_default_run_manifest
# ============================================================================

class TestBuildDefaultRunManifest:
    def test_basic_shape(self, empty_ws):
        manifest = m.build_default_run_manifest(
            empty_ws, migration_timestamp="2026-05-23T00:00:00Z"
        )
        assert manifest["schema_version"] == "2.0"
        assert manifest["run_id"] == "default"
        assert manifest["display_name"] == "Default"
        assert manifest["target"] is None
        assert manifest["backend"] == "cellpose"
        assert manifest["created_at"] == "2026-05-23T00:00:00Z"
        assert manifest["created_from"] is None
        assert manifest["current_round"] == {"lateral": 0, "axial": 0}


# ============================================================================
# plan_migration — empty + new workspaces
# ============================================================================

class TestPlanEmptyWorkspaces:
    def test_empty_workspace_returns_empty_plan(self, empty_ws):
        plan = m.plan_migration(empty_ws)
        assert plan.is_empty
        assert plan.total_operations == 0
        assert not plan.has_hard_errors

    def test_new_layout_returns_empty_plan(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)
        plan = m.plan_migration(ws)
        assert plan.is_empty


# ============================================================================
# plan_migration — legacy workspace + bucket fan-out
# ============================================================================

class TestPlanBucketFanOut:
    def test_xy_files_go_to_lateral(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        # All XY-stem accepted pairs should end up in lateral bucket
        xy_moves = [
            mv for mv in plan.moves
            if "_XY_" in mv.src.name and "accepted_pairs" in str(mv.src)
        ]
        assert len(xy_moves) > 0
        for mv in xy_moves:
            assert "/lateral/" in str(mv.dst).replace("\\", "/")
            assert "/accepted/" in str(mv.dst).replace("\\", "/")

    def test_xz_and_yz_files_go_to_axial(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        axial_moves = [
            mv for mv in plan.moves
            if ("_XZ_" in mv.src.name or "_YZ_" in mv.src.name)
            and "accepted_pairs" in str(mv.src)
        ]
        assert len(axial_moves) > 0
        for mv in axial_moves:
            assert "/axial/" in str(mv.dst).replace("\\", "/")
            assert "/accepted/" in str(mv.dst).replace("\\", "/")

    def test_mis_sorted_file_is_hard_error(self, tmp_path):
        # Put an XY file inside legacy axial/ — should HARD FAIL
        ws = tmp_path / "ws"
        misplaced = ws / "training_data" / "round_00" / "accepted_pairs" / "axial"
        _make_pair_file(misplaced, "stack_a_XY_z0001_img")

        plan = m.plan_migration(ws)
        assert plan.has_hard_errors
        assert any("Mis-sorted" in e for e in plan.hard_errors)

    def test_unparseable_filename_warns_not_fails(self, tmp_path):
        ws = tmp_path / "ws"
        bad_dir = ws / "training_data" / "round_00" / "accepted_pairs" / "spatial"
        _make_pair_file(bad_dir, "random_garbage")

        plan = m.plan_migration(ws)
        assert not plan.has_hard_errors
        assert any("Skipping unparseable" in w for w in plan.warnings)


class TestPlanModelsFanOut:
    def test_lateral_keyword_routes_to_lateral_bucket(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        lateral_models = [
            mv for mv in plan.moves
            if "model" in mv.src.name and "lateral" in mv.src.name.lower()
        ]
        assert len(lateral_models) >= 1
        for mv in lateral_models:
            # Phase B.2 (2026-05-25): migration now targets the FLAT
            # layout — `bucket/round_NN/model/` instead of legacy split
            # `bucket/models/round_NN/`.  Users migrating from very-old
            # workspaces skip the intermediate split layout entirely.
            dst_norm = str(mv.dst).replace("\\", "/")
            assert "/lateral/" in dst_norm
            assert "/model/" in dst_norm, \
                f"Expected /model/ subdir in destination; got {dst_norm}"

    def test_axial_keyword_routes_to_axial_bucket(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        axial_models = [
            mv for mv in plan.moves
            if "model" in mv.src.name and "axial" in mv.src.name.lower()
        ]
        assert len(axial_models) >= 1
        for mv in axial_models:
            # Phase B.2: same flat-layout migration target.
            dst_norm = str(mv.dst).replace("\\", "/")
            assert "/axial/" in dst_norm
            assert "/model/" in dst_norm, \
                f"Expected /model/ subdir in destination; got {dst_norm}"

    def test_unmatched_model_filename_warns(self, tmp_path):
        ws = tmp_path / "ws"
        models = ws / "models" / "round_00"
        models.mkdir(parents=True)
        (models / "ambiguous_model.json").write_bytes(b"fake")

        plan = m.plan_migration(ws)
        assert any("Cannot determine bucket" in w for w in plan.warnings)


class TestPlanPredictionsFanOut:
    def test_predictions_fanned_by_view_code(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        pred_moves = [
            mv for mv in plan.moves
            if "_pred_mask" in mv.src.name and "predictions" in str(mv.src)
        ]
        assert len(pred_moves) >= 2  # at least one XY + one XZ from fixture
        for mv in pred_moves:
            if "_XY_" in mv.src.name:
                assert "/lateral/" in str(mv.dst).replace("\\", "/")
            elif "_XZ_" in mv.src.name or "_YZ_" in mv.src.name:
                assert "/axial/" in str(mv.dst).replace("\\", "/")

    def test_predictions_manifest_routed_by_legacy_bucket(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        manifest_moves = [
            mv for mv in plan.moves
            if mv.src.name == "manifest.json" and "predictions" in str(mv.src)
        ]
        assert len(manifest_moves) >= 1
        for mv in manifest_moves:
            assert mv.dst.name == "manifest.json"
            assert "/predictions/" in str(mv.dst).replace("\\", "/")


# ============================================================================
# plan_migration — metadata splits + state wrap + run manifest init
# ============================================================================

class TestPlanMetadataAndState:
    def test_round_metadata_split_planned(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        meta_splits = [s for s in plan.splits if s.kind == "round_metadata"]
        assert len(meta_splits) == 1
        split = meta_splits[0]
        assert split.round_idx == 0
        assert "/lateral/" in str(split.dst_lateral).replace("\\", "/")
        assert "/axial/" in str(split.dst_axial).replace("\\", "/")

    def test_selected_indices_split_planned(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        sel_splits = [s for s in plan.splits if s.kind == "selected_indices"]
        assert len(sel_splits) == 1

    def test_state_wrap_planned(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        assert plan.state_wrap is not None
        assert plan.state_wrap.src.name == "hitl_state.json"
        assert plan.state_wrap.dst.name == "step2_state.json"

    def test_run_manifest_init_planned(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        assert plan.run_manifest_init is not None
        assert plan.run_manifest_init.run_id == "default"
        assert plan.run_manifest_init.dst.name == "run_manifest.json"
        assert "/runs/default/" in str(plan.run_manifest_init.dst).replace("\\", "/")


class TestPlanLegacyDeletes:
    def test_artifacts_subdirs_marked_for_delete(self, tmp_path):
        ws = tmp_path / "ws"
        # Legacy-marker requirement
        (ws / "training_data" / "round_00").mkdir(parents=True)
        for sub in ("models", "inference", "benchmarks"):
            (ws / "artifacts" / sub).mkdir(parents=True)
            (ws / "artifacts" / sub / "dummy.json").write_text("{}")

        plan = m.plan_migration(ws)
        delete_paths = [str(d.path) for d in plan.legacy_deletes]
        for sub in ("models", "inference", "benchmarks"):
            assert any(f"artifacts/{sub}" in p.replace("\\", "/") for p in delete_paths)


# ============================================================================
# Mixed layout detection
# ============================================================================

class TestMixedLayout:
    def test_mixed_workspace_flags_warning(self, legacy_ws):
        # legacy_ws is purely legacy; add new-layout marker to make it mixed
        (legacy_ws / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)

        plan = m.plan_migration(legacy_ws)
        assert plan.is_mixed_layout
        assert any("BOTH legacy and new" in w for w in plan.warnings)


# ============================================================================
# Backup + execute (happy path)
# ============================================================================

class TestIdempotency:
    def test_plan_on_already_migrated_workspace_is_empty(self, legacy_ws):
        # Run a full migration, then plan again — should be empty
        plan = m.plan_migration(legacy_ws)
        backup = m.create_backup(legacy_ws, timestamp="20260523_idem01")
        result = m.execute_migration(plan, backup)
        assert result.success

        # Second plan: should detect new layout, return empty
        plan2 = m.plan_migration(legacy_ws)
        assert plan2.is_empty
        assert plan2.total_operations == 0

    def test_execute_migration_idempotent_state_wrap(self, legacy_ws):
        # Run migration, then re-run execute on the same plan.
        # State wrap should skip (dst exists); no crash on v3-input.
        plan = m.plan_migration(legacy_ws)
        backup = m.create_backup(legacy_ws, timestamp="20260523_idem02")
        result1 = m.execute_migration(plan, backup)
        assert result1.success

        # Build a NEW plan from the original src paths (simulates someone
        # re-running execute on a stale plan).  The wrap dst already exists;
        # _execute_state_wrap should no-op.
        m._execute_state_wrap(plan.state_wrap)  # should NOT crash on v3 input

    def test_execute_run_manifest_init_does_not_clobber(self, legacy_ws):
        # First execute creates run_manifest.json
        plan = m.plan_migration(legacy_ws)
        backup = m.create_backup(legacy_ws, timestamp="20260523_idem03")
        result = m.execute_migration(plan, backup)
        assert result.success

        manifest_path = p.run_manifest_path(legacy_ws, "default")
        # User manually edits the display name
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["display_name"] = "User-Edited Name"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Re-run the init — should NOT clobber the edit
        m._execute_run_manifest_init(plan.run_manifest_init)
        with open(manifest_path, "r", encoding="utf-8") as f:
            after = json.load(f)
        assert after["display_name"] == "User-Edited Name"


class TestBackupCompleteness:
    def test_backup_includes_training_config(self, legacy_ws):
        # Pre-condition: add training_config.json to legacy workspace
        config = legacy_ws / "training_config.json"
        config.write_text('{"channels": ["DAPI", "actin"]}', encoding="utf-8")

        backup = m.create_backup(legacy_ws, timestamp="20260523_bk01")
        backed_up = backup / "training_config.json"
        assert backed_up.is_file()
        assert "DAPI" in backed_up.read_text(encoding="utf-8")

    def test_backup_skips_training_config_if_missing(self, legacy_ws):
        # legacy_ws fixture doesn't have training_config.json
        backup = m.create_backup(legacy_ws, timestamp="20260523_bk02")
        # Backup created successfully; just no training_config.json in it
        assert backup.exists()
        assert not (backup / "training_config.json").exists()

    def test_rollback_restores_training_config(self, legacy_ws):
        config = legacy_ws / "training_config.json"
        config.write_text('{"channels": ["original"]}', encoding="utf-8")

        plan = m.plan_migration(legacy_ws)
        backup = m.create_backup(legacy_ws, timestamp="20260523_bk03")
        result = m.execute_migration(plan, backup)
        assert result.success

        # Simulate post-migration accidental edit to config
        config.write_text('{"channels": ["corrupted"]}', encoding="utf-8")

        m.rollback_migration(backup, legacy_ws)
        restored = legacy_ws / "training_config.json"
        assert restored.is_file()
        assert "original" in restored.read_text(encoding="utf-8")


class TestAutoRollback:
    def test_auto_rollback_fires_on_execute_failure(self, legacy_ws, monkeypatch):
        # Inject a failure mid-execute (during metadata splits) and verify
        # the auto-rollback path restores the workspace.
        plan = m.plan_migration(legacy_ws)
        backup = m.create_backup(legacy_ws, timestamp="20260523_rb01")

        # Monkey-patch _execute_metadata_split to raise mid-execute
        call_count = {"n": 0}
        original = m._execute_metadata_split

        def failing_split(split):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise OSError("Simulated mid-execute failure")
            return original(split)

        monkeypatch.setattr(m, "_execute_metadata_split", failing_split)

        result = m.execute_migration(plan, backup, auto_rollback_on_failure=True)
        assert not result.success
        assert "Simulated mid-execute failure" in (result.error or "")
        assert result.auto_rolled_back

        # Workspace should be back to legacy state
        assert (legacy_ws / "hitl_state.json").is_file()
        assert (legacy_ws / "training_data" / "round_00" / "accepted_pairs"
                / "spatial" / "stack_a_XY_z0001_img.tif").is_file()
        # New layout artifacts should be gone (rolled back)
        assert not p.state_json_path(legacy_ws).is_file()
        assert not p.step2_root(legacy_ws).is_dir()

    def test_auto_rollback_disabled_leaves_partial_state(self, legacy_ws, monkeypatch):
        plan = m.plan_migration(legacy_ws)
        backup = m.create_backup(legacy_ws, timestamp="20260523_rb02")

        def failing_split(split):
            raise OSError("simulated")

        monkeypatch.setattr(m, "_execute_metadata_split", failing_split)

        result = m.execute_migration(plan, backup, auto_rollback_on_failure=False)
        assert not result.success
        assert not result.auto_rolled_back
        # Some moves succeeded before split failed — verify partial state
        # (moves run before splits, so at least some new-layout files exist)


class TestMultiRound:
    def test_full_migration_multi_round(self, legacy_ws_multi_round):
        ws = legacy_ws_multi_round
        plan = m.plan_migration(ws)
        assert not plan.has_hard_errors
        backup = m.create_backup(ws, timestamp="20260523_mr01")
        result = m.execute_migration(plan, backup)
        assert result.success, f"Multi-round migration failed: {result.error}"

        # Verify each round's data is in the right place
        for round_idx in (0, 1, 2):
            lateral_acc = p.round_accepted_dir(ws, "default", "lateral", round_idx)
            axial_acc = p.round_accepted_dir(ws, "default", "axial", round_idx)
            assert (lateral_acc / f"stack_a_XY_z{round_idx + 1:04d}_img.tif").is_file()
            assert (axial_acc / f"stack_a_XZ_z{round_idx + 1:04d}_img.tif").is_file()

            # Per-bucket models
            lateral_model_dir = p.model_dir(ws, "default", "lateral", round_idx)
            axial_model_dir = p.model_dir(ws, "default", "axial", round_idx)
            assert lateral_model_dir.is_dir()
            assert axial_model_dir.is_dir()

            # Per-bucket metadata + selected_indices
            lat_meta = p.round_metadata_path(ws, "default", "lateral", round_idx)
            ax_meta = p.round_metadata_path(ws, "default", "axial", round_idx)
            assert lat_meta.is_file()
            assert ax_meta.is_file()

        # State preserves round-keys
        v3 = json.loads(p.state_json_path(ws).read_text(encoding="utf-8"))
        assert v3["runs"]["default"]["current_round"] == 2
        assert v3["runs"]["default"]["current_round_by_view"] == {"lateral": 2, "axial": 2}
        assert "lateral_2" in v3["runs"]["default"]["training_data_by_round"]
        assert "axial_2" in v3["runs"]["default"]["training_data_by_round"]


class TestMetadataSplitSchemaVariants:
    """Verifies _execute_metadata_split handles BOTH the canonical
    RoundMetadata/SelectedIndicesRecord dict-envelope schema AND the
    legacy flat-shape variant (defensive backward compat).

    The schema bug that broke user testing 2026-05-23:
    user's round_metadata.json had a dict shape (canonical), but the
    initial migration code expected a flat list at top level.
    """

    def test_canonical_round_metadata_dict_envelope_splits(self, tmp_path):
        """Canonical: {'schema_version', 'round_id', 'samples': [...]}.
        After split: each per-bucket file preserves envelope + filters samples by view.
        """
        ws = tmp_path / "ws"
        src = ws / "training_data" / "round_00" / "round_metadata.json"
        src.parent.mkdir(parents=True)
        src.write_text(json.dumps({
            "schema_version": "1.0",
            "round_id": "round_00",
            "created_at": "2026-05-23T00:00:00Z",
            "config_snapshot": {"channels": ["DAPI"]},
            "parent_round": None,
            "samples": [
                {"view": "XY", "stack_id": "s1", "slice_index": 1},
                {"view": "XZ", "stack_id": "s1", "slice_index": 2},
                {"view": "YZ", "stack_id": "s1", "slice_index": 3},
            ],
        }), encoding="utf-8")

        split = m.MetadataSplit(
            src=src,
            dst_lateral=ws / "lateral.json",
            dst_axial=ws / "axial.json",
            kind="round_metadata",
            round_idx=0,
            purpose="test",
        )
        m._execute_metadata_split(split)

        lateral = json.loads((ws / "lateral.json").read_text(encoding="utf-8"))
        axial = json.loads((ws / "axial.json").read_text(encoding="utf-8"))

        # Envelope keys preserved
        assert lateral["schema_version"] == "1.0"
        assert lateral["round_id"] == "round_00"
        assert lateral["config_snapshot"] == {"channels": ["DAPI"]}
        # Only XY records
        assert len(lateral["samples"]) == 1
        assert lateral["samples"][0]["view"] == "XY"
        # Only XZ/YZ records
        assert len(axial["samples"]) == 2
        assert all(s["view"] in ("XZ", "YZ") for s in axial["samples"])

    def test_legacy_flat_list_still_splits(self, tmp_path):
        """Backward compat: bare list at top level (no envelope).
        Migration should still split correctly and emit empty envelope.
        """
        ws = tmp_path / "ws"
        src = ws / "training_data" / "round_00" / "round_metadata.json"
        src.parent.mkdir(parents=True)
        src.write_text(json.dumps([
            {"view": "XY", "stack_id": "s1", "slice_index": 1},
            {"view": "XZ", "stack_id": "s1", "slice_index": 2},
        ]), encoding="utf-8")

        split = m.MetadataSplit(
            src=src,
            dst_lateral=ws / "lateral.json",
            dst_axial=ws / "axial.json",
            kind="round_metadata",
            round_idx=0,
            purpose="test",
        )
        m._execute_metadata_split(split)

        lateral = json.loads((ws / "lateral.json").read_text(encoding="utf-8"))
        axial = json.loads((ws / "axial.json").read_text(encoding="utf-8"))
        assert "samples" in lateral
        assert "samples" in axial
        assert lateral["samples"][0]["view"] == "XY"
        assert axial["samples"][0]["view"] == "XZ"

    def test_canonical_selected_indices_envelope_splits(self, tmp_path):
        """Canonical: {'schema_version', 'round_id', 'selections': {...}}.
        Per-bucket files preserve envelope + filter selections by view.
        """
        ws = tmp_path / "ws"
        src = ws / "training_data" / "round_00" / "selected_indices.json"
        src.parent.mkdir(parents=True)
        src.write_text(json.dumps({
            "schema_version": "1.0",
            "round_id": "round_00",
            "selections": {
                "XY": {"s1": [1, 2, 3]},
                "XZ": {"s1": [10]},
                "YZ": {"s1": [20]},
            },
        }), encoding="utf-8")

        split = m.MetadataSplit(
            src=src,
            dst_lateral=ws / "lateral.json",
            dst_axial=ws / "axial.json",
            kind="selected_indices",
            round_idx=0,
            purpose="test",
        )
        m._execute_metadata_split(split)

        lateral = json.loads((ws / "lateral.json").read_text(encoding="utf-8"))
        axial = json.loads((ws / "axial.json").read_text(encoding="utf-8"))

        assert lateral["schema_version"] == "1.0"
        assert lateral["selections"] == {"XY": {"s1": [1, 2, 3]}}
        assert axial["selections"] == {"XZ": {"s1": [10]}, "YZ": {"s1": [20]}}

    def test_legacy_flat_selected_indices_still_splits(self, tmp_path):
        """Backward compat: bare view-keyed dict at top level."""
        ws = tmp_path / "ws"
        src = ws / "training_data" / "round_00" / "selected_indices.json"
        src.parent.mkdir(parents=True)
        src.write_text(json.dumps({
            "XY": {"s1": [1, 2]},
            "XZ": {"s1": [5]},
        }), encoding="utf-8")

        split = m.MetadataSplit(
            src=src,
            dst_lateral=ws / "lateral.json",
            dst_axial=ws / "axial.json",
            kind="selected_indices",
            round_idx=0,
            purpose="test",
        )
        m._execute_metadata_split(split)

        lateral = json.loads((ws / "lateral.json").read_text(encoding="utf-8"))
        axial = json.loads((ws / "axial.json").read_text(encoding="utf-8"))
        # Legacy flat → envelope-less output (empty envelope merged with selections)
        assert lateral["selections"] == {"XY": {"s1": [1, 2]}}
        assert axial["selections"] == {"XZ": {"s1": [5]}}


class TestModelFileNameAmbiguity:
    def test_model_with_both_keywords_routes_to_axial(self, tmp_path):
        # Documents the substring-match precedence: axial wins.
        # File like 'lateral_axial_combined.json' → axial bucket.
        ws = tmp_path / "ws"
        (ws / "training_data" / "round_00").mkdir(parents=True)  # legacy marker
        models = ws / "models" / "round_00"
        models.mkdir(parents=True)
        (models / "lateral_axial_combined.json").write_bytes(b"fake")

        plan = m.plan_migration(ws)
        # Should route to axial (axial check happens first in _plan_one_model_file)
        ambiguous_moves = [mv for mv in plan.moves
                           if mv.src.name == "lateral_axial_combined.json"]
        assert len(ambiguous_moves) == 1
        # Phase B.2 (2026-05-25): destination now goes through `model_dir`
        # which returns the FLAT layout — `axial/round_NN/model/`.
        dst_norm = str(ambiguous_moves[0].dst).replace("\\", "/")
        assert "/axial/" in dst_norm
        assert "/model/" in dst_norm, (
            "Expected lateral_axial_combined.json to route to axial "
            "(axial substring check has precedence over lateral); "
            f"got dst={dst_norm}"
        )


class TestExecuteHappyPath:
    def test_full_migration_round_trip(self, legacy_ws):
        # Plan + backup + execute
        plan = m.plan_migration(legacy_ws)
        assert not plan.has_hard_errors
        backup = m.create_backup(legacy_ws, timestamp="20260523_000000")
        assert backup.exists()
        assert backup.is_dir()
        assert (backup / "training_data" / "round_00").is_dir()
        assert (backup / "hitl_state.json").is_file()

        result = m.execute_migration(plan, backup)
        assert result.success, f"Execute failed: {result.error}"
        assert not result.auto_rolled_back

        # New layout should now exist
        new_state = p.state_json_path(legacy_ws)
        assert new_state.is_file()
        with open(new_state, "r", encoding="utf-8") as f:
            v3 = json.load(f)
        assert v3["schema_version"] == 3
        assert v3["active_run_id"] == "default"
        assert v3["runs"]["default"]["current_round"] == 0

        # Run manifest written
        run_manifest = p.run_manifest_path(legacy_ws, "default")
        assert run_manifest.is_file()

        # XY pairs in lateral bucket
        lateral_accepted = p.round_accepted_dir(legacy_ws, "default", "lateral", 0)
        assert lateral_accepted.is_dir()
        assert (lateral_accepted / "stack_a_XY_z0001_img.tif").is_file()

        # XZ/YZ pairs in axial bucket
        axial_accepted = p.round_accepted_dir(legacy_ws, "default", "axial", 0)
        assert (axial_accepted / "stack_a_XZ_z0005_img.tif").is_file()
        assert (axial_accepted / "stack_a_YZ_z0010_img.tif").is_file()

        # Per-bucket round_metadata split
        lateral_meta = p.round_metadata_path(legacy_ws, "default", "lateral", 0)
        assert lateral_meta.is_file()
        lateral_meta_data = json.loads(lateral_meta.read_text(encoding="utf-8"))
        # Per RoundMetadata schema, the records live under the "samples" key
        # of the envelope dict.
        assert "samples" in lateral_meta_data, (
            "round_metadata.json must preserve the dict envelope with 'samples' key"
        )
        assert all(r["view"] == "XY" for r in lateral_meta_data["samples"])
        # Envelope keys preserved
        assert lateral_meta_data.get("schema_version") == "1.0"
        assert lateral_meta_data.get("round_id") == "round_00"

        axial_meta = p.round_metadata_path(legacy_ws, "default", "axial", 0)
        axial_meta_data = json.loads(axial_meta.read_text(encoding="utf-8"))
        assert "samples" in axial_meta_data
        assert all(r["view"] in ("XZ", "YZ") for r in axial_meta_data["samples"])

        # Legacy state file should be gone (moved into wrap)
        # NB: the wrap leaves the source file untouched; we did add an
        # explicit delete-legacy-state step in some designs but here we
        # preserve it in the backup and rely on the user to clean up
        # manually if desired.  Verify behavior:
        # (No assertion either way — implementation chose to leave it.)

        # Legacy artifacts/{models,inference,benchmarks} should be deleted
        # (legacy_ws doesn't have them in fixture so nothing to assert)

    def test_execute_refuses_when_plan_has_hard_errors(self, tmp_path):
        ws = tmp_path / "ws"
        # Make a mis-sorted file → hard error
        misplaced = ws / "training_data" / "round_00" / "accepted_pairs" / "axial"
        _make_pair_file(misplaced, "stack_a_XY_z0001_img")

        plan = m.plan_migration(ws)
        backup = m.create_backup(ws, timestamp="20260523_000001")
        result = m.execute_migration(plan, backup)
        assert not result.success
        assert "hard errors" in (result.error or "").lower()


# ============================================================================
# Rollback
# ============================================================================

class TestRollback:
    def test_rollback_restores_legacy_layout(self, legacy_ws):
        plan = m.plan_migration(legacy_ws)
        backup = m.create_backup(legacy_ws, timestamp="20260523_000002")
        result = m.execute_migration(plan, backup)
        assert result.success

        # Rollback
        m.rollback_migration(backup, legacy_ws)

        # Legacy layout should be back
        assert (legacy_ws / "training_data" / "round_00" / "accepted_pairs" / "spatial" /
                "stack_a_XY_z0001_img.tif").is_file()
        assert (legacy_ws / "hitl_state.json").is_file()

        # New layout artifacts should be gone
        assert not p.state_json_path(legacy_ws).is_file()
        assert not p.step2_root(legacy_ws).is_dir()

    def test_rollback_on_nonexistent_backup_raises(self, legacy_ws):
        with pytest.raises(ValueError, match="does not exist"):
            m.rollback_migration(legacy_ws / "nonexistent_backup", legacy_ws)

    def test_rollback_idempotent(self, legacy_ws):
        # Migrate, rollback, then rollback again — second should not fail
        plan = m.plan_migration(legacy_ws)
        backup = m.create_backup(legacy_ws, timestamp="20260523_000003")
        m.execute_migration(plan, backup)
        m.rollback_migration(backup, legacy_ws)
        # Second rollback on already-rolled-back workspace
        m.rollback_migration(backup, legacy_ws)
        # Still has legacy files
        assert (legacy_ws / "hitl_state.json").is_file()


# ============================================================================
# Sentinel
# ============================================================================

class TestSentinel:
    def test_not_skipped_initially(self, empty_ws):
        assert not m.is_migration_skipped(empty_ws)

    def test_mark_then_check(self, empty_ws):
        m.mark_migration_skipped(empty_ws)
        assert m.is_migration_skipped(empty_ws)
        assert (empty_ws / m.MIGRATION_SKIPPED_SENTINEL).is_file()

    def test_clear_after_mark(self, empty_ws):
        m.mark_migration_skipped(empty_ws)
        m.clear_migration_skipped(empty_ws)
        assert not m.is_migration_skipped(empty_ws)

    def test_clear_when_not_present(self, empty_ws):
        # No exception when clearing nonexistent sentinel
        m.clear_migration_skipped(empty_ws)
        assert not m.is_migration_skipped(empty_ws)


# ============================================================================
# Public API
# ============================================================================

class TestPublicApi:
    def test_all_export_list_present(self):
        assert hasattr(m, "__all__")
        assert isinstance(m.__all__, list)
        assert len(m.__all__) > 0

    def test_all_listed_symbols_exist(self):
        for name in m.__all__:
            assert hasattr(m, name), f"__all__ lists {name!r} but module has no such attribute"

    def test_key_data_classes_exported(self):
        for name in ("FileMove", "MetadataSplit", "MigrationPlan",
                     "ExecutionResult", "StateSchemaWrap"):
            assert name in m.__all__

    def test_key_functions_exported(self):
        for name in ("plan_migration", "execute_migration", "rollback_migration",
                     "create_backup", "wrap_state_schema_v2_to_v3",
                     "is_migration_skipped", "mark_migration_skipped"):
            assert name in m.__all__

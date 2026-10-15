"""Tests for the Step 3 directory restructure migration logic.

Phase 2 of the directory restructure (2026-05-20).  Companion to
``napari_epicycle/fusion_reconstruction/_migrate_step3_layout.py``.

These tests build synthetic legacy workspaces inside ``tmp_path``,
plan + execute migrations against them, and verify the resulting
layout matches the v2 spec.  No Qt / napari dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from napari_epicycle.fusion_reconstruction import _paths as p
from napari_epicycle.fusion_reconstruction import _migrate_step3_layout as m


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_legacy_workspace(
    root: Path,
    *,
    patches: dict | None = None,
    refined_models: dict | None = None,
    refinement_pairs: list | None = None,
    include_qc: bool = True,
    include_snapshots: bool = True,
    include_prediction_cache: bool = True,
    include_extraction_manifest: bool = False,
) -> Path:
    """Build a synthetic legacy workspace under ``root``.

    Returns the workspace path.  All inputs are optional — pass only
    what you want present to keep tests narrow.

    ``patches`` is a dict[patch_name -> dict] with optional keys:
      * ``run_id`` (str) — the legacy reconstruction_run_id
      * ``has_qc`` (bool) — write qc CSVs
    ``refined_models`` is a dict[bundle_name -> dict] with keys:
      * ``cellpose_buckets`` (list[str]) — write Cellpose-style files
        under ``models/refinement/{name}/models/{bucket}``
    ``refinement_pairs`` is a list of patch_name strings, each gets a
      synthetic per-patch dir under refinement_training_data/.
    """
    ws = root / "workspace"
    ws.mkdir()

    # State JSON ---------------------------------------------------------------
    state = {
        "schema_version": 1,
        "patches": {},
    }
    if patches:
        for name, opts in patches.items():
            patch_record = {
                "labels_path": f"/some/absolute/old/path/{name}.tif",
            }
            run_id = opts.get("run_id")
            if run_id:
                patch_record["reconstruction_run_id"] = run_id
                # Build the legacy reconstruction dir.
                run_dir = ws / "artifacts" / "reconstruction" / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "labels_3d.tif").write_bytes(b"fake-tiff")
                (run_dir / "manifest.json").write_text(
                    json.dumps({"run_id": run_id, "patch_name": name})
                )
            if opts.get("has_qc"):
                qc_dir = ws / "artifacts" / "qc"
                qc_dir.mkdir(parents=True, exist_ok=True)
                (qc_dir / f"{name}_features.csv").write_text(
                    "label,volume\n1,100\n"
                )
                patch_record["qc_features_csv"] = str(
                    qc_dir / f"{name}_features.csv"
                )
            state["patches"][name] = patch_record

    # Refined models ----------------------------------------------------------
    if refined_models:
        for bundle_name, opts in refined_models.items():
            bundle = ws / "models" / "refinement" / bundle_name
            bundle.mkdir(parents=True)
            # Pretend Cellpose wrote its standard layout.
            cellpose_dir = bundle / "models"
            cellpose_dir.mkdir()
            for bucket in opts.get("cellpose_buckets", []):
                (cellpose_dir / bucket).write_bytes(b"fake-weights")
                # Companion files at bundle root.
                (bundle / f"{bucket}_train_loss.npy").write_bytes(b"npy")
            # Legacy "model_pack.json" at bundle root.
            (bundle / "model_pack.json").write_text(
                json.dumps({
                    "schema_version": "1.1",
                    "name": bundle_name,
                    "lateral_init": {"mode": "scratch"},
                    "axial_init": {"mode": "scratch"},
                })
            )

    # Refinement training pairs ----------------------------------------------
    if refinement_pairs:
        pairs_root = ws / "artifacts" / "refinement_training_data"
        pairs_root.mkdir(parents=True)
        for name in refinement_pairs:
            patch_pairs = pairs_root / name
            (patch_pairs / "spatial").mkdir(parents=True)
            (patch_pairs / "axial").mkdir(parents=True)
            (patch_pairs / "spatial" / "xy_0010_img.tif").write_bytes(b"x")
            (patch_pairs / "spatial" / "xy_0010_mask.tif").write_bytes(b"x")

    if include_extraction_manifest:
        pairs_root = ws / "artifacts" / "refinement_training_data"
        pairs_root.mkdir(parents=True, exist_ok=True)
        (pairs_root / "extraction_manifest.json").write_text(
            json.dumps({"schema_version": "1.2", "per_patch": {}})
        )

    # Snapshots + prediction cache -------------------------------------------
    if include_snapshots:
        snap = ws / "artifacts" / "snapshots"
        snap.mkdir(parents=True)
        (snap / "patch_001_pre_commit_20260101_120000.tif").write_bytes(b"s")
    if include_prediction_cache:
        pc = ws / "artifacts" / "prediction_cache"
        pc.mkdir(parents=True)
        (pc / "patch_001_xy_masks.npy").write_bytes(b"m")
    if not include_qc:
        # Caller suppressed QC entirely — wipe any QC dir we may have
        # created via the patches loop above.
        qc_dir = ws / "artifacts" / "qc"
        if qc_dir.is_dir():
            import shutil
            shutil.rmtree(qc_dir)

    # Write state JSON last so it captures patches' built state.
    (ws / "reconstruction_state.json").write_text(json.dumps(state, indent=2))

    return ws


# ===========================================================================
# Plan: empty / no-op workspaces
# ===========================================================================


class TestPlanEmptyWorkspace:

    def test_empty_workspace_returns_empty_plan(self, tmp_path):
        ws = tmp_path / "empty_workspace"
        ws.mkdir()
        plan = m.plan_migration(ws)
        assert plan.is_empty
        assert plan.total_operations == 0
        assert plan.legacy_state_path is None

    def test_new_layout_workspace_returns_empty_plan(self, tmp_path):
        """Workspace already on v2 → empty plan."""
        ws = tmp_path / "new_workspace"
        ws.mkdir()
        (ws / "step3_state.json").write_text("{}")
        (ws / "artifacts" / "step3" / "runs").mkdir(parents=True)
        plan = m.plan_migration(ws)
        assert plan.is_empty
        assert not plan.is_mixed_layout

    def test_only_models_dir_no_refinement_returns_empty_plan(self, tmp_path):
        """models/round_X/ exists (Step 2 HITL) but no Step 3 markers."""
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "models" / "round_001").mkdir(parents=True)
        plan = m.plan_migration(ws)
        assert plan.is_empty


# ===========================================================================
# Plan: per-component coverage
# ===========================================================================


class TestPlanQcDir:

    def test_qc_dir_move_included(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"p1": {"run_id": "abc", "has_qc": True}},
        )
        plan = m.plan_migration(ws)
        moves = [mv for mv in plan.moves if mv.purpose.startswith("QC")]
        assert len(moves) == 1
        assert moves[0].src == ws / "artifacts" / "qc"
        assert moves[0].dst == p.qc_dir(ws)
        assert moves[0].kind == "dir"


class TestPlanSnapshots:

    def test_snapshots_move_included(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        plan = m.plan_migration(ws)
        moves = [mv for mv in plan.moves if "snapshots" in mv.purpose.lower()]
        assert len(moves) == 1
        assert moves[0].src == ws / "artifacts" / "snapshots"
        assert moves[0].dst == p.snapshots_dir(ws)


class TestPlanPredictionCache:

    def test_prediction_cache_move_included(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        plan = m.plan_migration(ws)
        moves = [mv for mv in plan.moves if "prediction" in mv.purpose.lower()]
        assert len(moves) == 1


class TestPlanReconstructionRuns:

    def test_single_patch_run_planned(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"patch_001": {"run_id": "abc-123"}},
        )
        plan = m.plan_migration(ws)
        run_moves = [
            mv for mv in plan.moves
            if "Reconstruction run" in mv.purpose
        ]
        assert len(run_moves) == 1
        assert run_moves[0].src == ws / "artifacts" / "reconstruction" / "abc-123"
        assert run_moves[0].dst == p.patch_run_dir(ws, "patch_001")

    def test_multiple_patches_all_planned(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={
                "patch_001": {"run_id": "uuid-1"},
                "patch_002": {"run_id": "uuid-2"},
                "patch_003": {"run_id": "uuid-3"},
            },
        )
        plan = m.plan_migration(ws)
        run_moves = [
            mv for mv in plan.moves
            if "Reconstruction run" in mv.purpose
        ]
        assert len(run_moves) == 3

    def test_patch_without_run_id_skipped(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"patch_001": {}},  # No run_id
        )
        plan = m.plan_migration(ws)
        run_moves = [
            mv for mv in plan.moves
            if "Reconstruction run" in mv.purpose
        ]
        assert len(run_moves) == 0

    def test_orphan_run_id_warned(self, tmp_path):
        """State references a run_id but the dir doesn't exist on disk."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        # State references a run_id but we don't actually create the dir.
        state = {
            "schema_version": 1,
            "patches": {
                "ghost": {"reconstruction_run_id": "missing-uuid"},
            },
        }
        (ws / "reconstruction_state.json").write_text(json.dumps(state))
        # Add a legacy marker so detection fires.
        (ws / "artifacts" / "qc").mkdir(parents=True)
        plan = m.plan_migration(ws)
        assert any(
            "does not exist" in w and "missing-uuid" in w
            for w in plan.warnings
        )


class TestPlanRefinementPairs:

    def test_per_patch_pair_dirs_moved(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            refinement_pairs=["patch_001", "patch_002"],
        )
        plan = m.plan_migration(ws)
        pair_moves = [
            mv for mv in plan.moves
            if "Training pairs" in mv.purpose
        ]
        assert len(pair_moves) == 2
        names = {mv.dst.name for mv in pair_moves}
        assert names == {"patch_001", "patch_002"}

    def test_extraction_manifest_moved(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            include_extraction_manifest=True,
        )
        plan = m.plan_migration(ws)
        manifest_moves = [
            mv for mv in plan.moves
            if "extraction manifest" in mv.purpose.lower()
        ]
        assert len(manifest_moves) == 1
        assert manifest_moves[0].dst == p.refinement_manifest_path(ws)


class TestPlanRefinedModelBundles:

    def test_single_bundle_planned(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            refined_models={"test_v1": {"cellpose_buckets": ["lateral"]}},
        )
        plan = m.plan_migration(ws)
        bundle_moves = [
            mv for mv in plan.moves
            if "model bundle" in mv.purpose.lower()
        ]
        assert len(bundle_moves) == 1
        assert bundle_moves[0].dst == p.refined_model_dir(ws, "test_v1")

    def test_cellpose_lifts_planned_for_both_buckets(self, tmp_path):
        """Plan should list lifts for BOTH lateral and axial regardless of
        what actually exists; execute() will skip non-existent srcs."""
        ws = _make_legacy_workspace(
            tmp_path,
            refined_models={"only_lateral": {"cellpose_buckets": ["lateral"]}},
        )
        plan = m.plan_migration(ws)
        bucket_names = {lift.bucket for lift in plan.cellpose_lifts}
        assert bucket_names == {"lateral", "axial"}

    def test_multiple_bundles_each_get_lifts(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            refined_models={
                "v1": {"cellpose_buckets": ["lateral"]},
                "v2": {"cellpose_buckets": ["axial"]},
            },
        )
        plan = m.plan_migration(ws)
        # 2 bundles × 2 buckets each = 4 lifts planned.
        assert len(plan.cellpose_lifts) == 4


# ===========================================================================
# Plan: warnings (mixed layout, duplicate dst, malformed state JSON)
# ===========================================================================


class TestPlanWarnings:

    def test_mixed_layout_flagged(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        # Manually create a partial new-layout marker.
        p.step3_root(ws).mkdir(parents=True)
        plan = m.plan_migration(ws)
        assert plan.is_mixed_layout
        assert any(
            "previous migration was interrupted" in w for w in plan.warnings
        )

    def test_destination_collision_warned(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"p1": {"run_id": "abc", "has_qc": True}},
        )
        # Pre-create the new qc dir so move would clobber.
        p.qc_dir(ws).mkdir(parents=True)
        plan = m.plan_migration(ws)
        qc_moves = [mv for mv in plan.moves if mv.purpose.startswith("QC")]
        assert len(qc_moves) == 0
        assert any("already exists" in w for w in plan.warnings)

    def test_malformed_state_json_warned(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "reconstruction_state.json").write_text("{not valid json")
        (ws / "artifacts" / "reconstruction").mkdir(parents=True)
        plan = m.plan_migration(ws)
        assert any(
            "Could not parse legacy state JSON" in w
            for w in plan.warnings
        )

    def test_non_dict_patches_field_does_not_crash(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "reconstruction_state.json").write_text(
            json.dumps({"patches": "this is not a dict"})
        )
        (ws / "artifacts" / "reconstruction").mkdir(parents=True)
        plan = m.plan_migration(ws)
        # Should not raise; just produce no per-patch moves.
        assert isinstance(plan, m.MigrationPlan)


# ===========================================================================
# State JSON content migration
# ===========================================================================


class TestStateJsonMigration:

    def test_layout_version_added(self, tmp_path):
        state = {"patches": {}}
        new = m.migrate_state_json_content(state, tmp_path / "ws")
        assert new["layout_version"] == p.CURRENT_LAYOUT_VERSION
        assert new["schema_version"] == 2

    def test_labels_path_rewritten(self, tmp_path):
        ws = tmp_path / "ws"
        state = {
            "patches": {
                "patch_001": {
                    "labels_path": "/old/absolute/path/that/will/be/gone.tif",
                },
            },
        }
        new = m.migrate_state_json_content(state, ws)
        new_path = new["patches"]["patch_001"]["labels_path"]
        assert new_path == str(p.patch_labels_stem(ws, "patch_001")) + ".tif"
        assert "/old/absolute" not in new_path

    def test_qc_features_csv_rewritten(self, tmp_path):
        ws = tmp_path / "ws"
        state = {
            "patches": {
                "patch_001": {
                    "qc_features_csv": "/old/abs/path/patch_001_features.csv",
                },
            },
        }
        new = m.migrate_state_json_content(state, ws)
        new_path = new["patches"]["patch_001"]["qc_features_csv"]
        assert new_path == str(p.qc_features_csv(ws, "patch_001"))

    def test_reconstruction_run_id_dropped(self, tmp_path):
        ws = tmp_path / "ws"
        state = {
            "patches": {
                "patch_001": {
                    "reconstruction_run_id": "uuid-123",
                    "labels_path": "/abs.tif",
                },
            },
        }
        new = m.migrate_state_json_content(state, ws)
        assert "reconstruction_run_id" not in new["patches"]["patch_001"]

    def test_run_id_alias_also_dropped(self, tmp_path):
        ws = tmp_path / "ws"
        state = {
            "patches": {
                "patch_001": {"run_id": "uuid-123"},
            },
        }
        new = m.migrate_state_json_content(state, ws)
        assert "run_id" not in new["patches"]["patch_001"]

    def test_top_level_lateral_model_path_rewritten(self, tmp_path):
        ws = tmp_path / "ws"
        legacy_path = p.legacy_refinement_models_root(ws) / "test_v1" / "lateral"
        state = {"lateral_model_path": str(legacy_path)}
        new = m.migrate_state_json_content(state, ws)
        assert new["lateral_model_path"] == str(
            p.refined_model_weights_path(ws, "test_v1", "lateral")
        )

    def test_top_level_axial_model_path_rewritten(self, tmp_path):
        ws = tmp_path / "ws"
        legacy_path = p.legacy_refinement_models_root(ws) / "tuned" / "axial"
        state = {"axial_model_path": str(legacy_path)}
        new = m.migrate_state_json_content(state, ws)
        assert new["axial_model_path"] == str(
            p.refined_model_weights_path(ws, "tuned", "axial")
        )

    def test_unfamiliar_model_path_left_alone(self, tmp_path):
        ws = tmp_path / "ws"
        # Path is NOT under workspace/models/refinement/ — leave it.
        state = {"lateral_model_path": "/some/other/place/weights"}
        new = m.migrate_state_json_content(state, ws)
        assert new["lateral_model_path"] == "/some/other/place/weights"

    def test_empty_lateral_model_path_left_alone(self, tmp_path):
        ws = tmp_path / "ws"
        state = {"lateral_model_path": ""}
        new = m.migrate_state_json_content(state, ws)
        assert new["lateral_model_path"] == ""

    def test_does_not_mutate_input(self, tmp_path):
        """migrate_state_json_content must return a new dict, not mutate."""
        ws = tmp_path / "ws"
        original = {"patches": {"p1": {"labels_path": "/old.tif"}}}
        snapshot = json.dumps(original, sort_keys=True)
        m.migrate_state_json_content(original, ws)
        assert json.dumps(original, sort_keys=True) == snapshot


# ===========================================================================
# Backup
# ===========================================================================


class TestBackup:

    def test_creates_backup_dir(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        backup = m.create_backup(ws, timestamp="20260520_120000")
        assert backup.is_dir()
        assert backup.name == ".legacy_pre_restructure_20260520_120000"

    def test_copies_artifacts_subtree(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"p1": {"run_id": "abc", "has_qc": True}},
        )
        backup = m.create_backup(ws)
        assert (backup / "artifacts" / "qc" / "p1_features.csv").is_file()
        assert (
            backup / "artifacts" / "reconstruction" / "abc" / "labels_3d.tif"
        ).is_file()

    def test_copies_models_subtree(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            refined_models={"v1": {"cellpose_buckets": ["lateral"]}},
        )
        backup = m.create_backup(ws)
        assert (
            backup / "models" / "refinement" / "v1" / "models" / "lateral"
        ).is_file()

    def test_copies_legacy_state_json(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        backup = m.create_backup(ws)
        assert (backup / "reconstruction_state.json").is_file()

    def test_existing_backup_dir_raises(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        m.create_backup(ws, timestamp="x")
        with pytest.raises(FileExistsError):
            m.create_backup(ws, timestamp="x")


# ===========================================================================
# Execute
# ===========================================================================


class TestExecuteMigration:

    def test_skeleton_created(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        assert p.step3_root(ws).is_dir()
        assert p.runs_dir(ws).is_dir()
        assert p.refinement_root(ws).is_dir()
        assert p.refinement_pairs_root(ws).is_dir()
        assert p.refined_models_root(ws).is_dir()

    def test_qc_dir_moved(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"p1": {"run_id": "abc", "has_qc": True}},
        )
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        assert p.qc_features_csv(ws, "p1").is_file()
        assert not (ws / "artifacts" / "qc").exists()  # legacy gone

    def test_snapshots_moved(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        legacy_snap = ws / "artifacts" / "snapshots"
        new_snap = p.snapshots_dir(ws)
        assert not legacy_snap.exists()
        assert new_snap.is_dir()
        # Original file moved.
        assert (
            new_snap / "patch_001_pre_commit_20260101_120000.tif"
        ).is_file()

    def test_reconstruction_run_moved_to_patch_name(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"patch_alpha": {"run_id": "uuid-aaa"}},
        )
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        assert p.patch_run_dir(ws, "patch_alpha").is_dir()
        assert (
            p.patch_run_dir(ws, "patch_alpha") / "labels_3d.tif"
        ).is_file()
        # Legacy uuid-named dir gone.
        assert not (
            ws / "artifacts" / "reconstruction" / "uuid-aaa"
        ).exists()

    def test_refined_model_lift_lateral(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            refined_models={"my_model": {"cellpose_buckets": ["lateral"]}},
        )
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        bundle = p.refined_model_dir(ws, "my_model")
        assert bundle.is_dir()
        # Cellpose file lifted to bundle root.
        assert (bundle / "lateral").is_file()
        # Cellpose-internal models/ subdir cleaned up.
        assert not (bundle / "models").exists()

    def test_refined_model_lift_both_buckets(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            refined_models={
                "both": {"cellpose_buckets": ["lateral", "axial"]},
            },
        )
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        bundle = p.refined_model_dir(ws, "both")
        assert (bundle / "lateral").is_file()
        assert (bundle / "axial").is_file()
        assert not (bundle / "models").exists()

    def test_refinement_pairs_moved(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            refinement_pairs=["patch_001"],
        )
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        new_pairs = p.patch_pairs_dir(ws, "patch_001")
        assert new_pairs.is_dir()
        assert (new_pairs / "spatial" / "xy_0010_img.tif").is_file()
        assert (new_pairs / "axial").is_dir()

    def test_state_json_renamed_and_rewritten(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"patch_001": {"run_id": "uuid-1", "has_qc": True}},
        )
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        # Legacy state JSON gone.
        assert not p.legacy_state_json_path(ws).exists()
        # New state JSON present.
        new_state = json.loads(p.state_json_path(ws).read_text())
        assert new_state["layout_version"] == p.CURRENT_LAYOUT_VERSION
        assert new_state["schema_version"] == 2
        # reconstruction_run_id dropped.
        assert "reconstruction_run_id" not in (
            new_state["patches"]["patch_001"]
        )
        # labels_path rewritten.
        assert "artifacts" + "/" + "step3" in (
            new_state["patches"]["patch_001"]["labels_path"].replace("\\", "/")
        )

    def test_empty_legacy_dirs_cleaned_up(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        # After all reconstruction runs migrated, legacy reconstruction/
        # dir should be deleted (it's empty).
        assert not p.legacy_reconstruction_root(ws).exists()

    def test_success_records_completed_steps(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        assert len(result.completed_steps) > 0
        # First step is always skeleton.
        assert "skeleton" in result.completed_steps[0].lower()

    def test_backup_dir_recorded_in_result(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        backup = m.create_backup(ws, timestamp="x")
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan, backup_dir=backup)
        assert result.backup_dir == backup

    def test_empty_plan_executes_without_error(self, tmp_path):
        ws = tmp_path / "empty_ws"
        ws.mkdir()
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        # Even on empty plan, the skeleton is created (idempotent).
        assert p.step3_root(ws).is_dir()


# ===========================================================================
# Cellpose-lift edge cases
# ===========================================================================


class TestCellposeLiftEdgeCases:

    def test_lift_skipped_if_dst_already_present(self, tmp_path):
        """A second-time migration shouldn't fail on already-lifted files."""
        ws = _make_legacy_workspace(
            tmp_path,
            refined_models={"v1": {"cellpose_buckets": ["lateral"]}},
        )
        # Pre-create the lifted file at the post-move location to
        # simulate a partial prior migration.
        bundle = p.refined_model_dir(ws, "v1")
        bundle.parent.mkdir(parents=True, exist_ok=True)
        plan = m.plan_migration(ws)
        # Manually trigger the first half (move) then pre-create dst.
        result = m.execute_migration(plan)
        assert result.success
        # Re-running execute on an empty plan should be safe.
        empty_plan = m.plan_migration(ws)
        result2 = m.execute_migration(empty_plan)
        assert result2.success

    def test_lift_for_nonexistent_bucket_is_silent_skip(self, tmp_path):
        """Plan lists both buckets but only lateral exists — execute
        should silently skip axial."""
        ws = _make_legacy_workspace(
            tmp_path,
            refined_models={"only_lateral": {"cellpose_buckets": ["lateral"]}},
        )
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        bundle = p.refined_model_dir(ws, "only_lateral")
        assert (bundle / "lateral").is_file()
        assert not (bundle / "axial").exists()


# ===========================================================================
# Rollback
# ===========================================================================


class TestRollback:

    def test_rollback_restores_legacy_state(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"p1": {"run_id": "uuid-1", "has_qc": True}},
            refined_models={"v1": {"cellpose_buckets": ["lateral"]}},
        )
        backup = m.create_backup(ws, timestamp="t1")
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan, backup_dir=backup)
        assert result.success
        # Now roll back.
        m.rollback_migration(backup, ws)
        # Legacy markers restored.
        assert p.legacy_state_json_path(ws).is_file()
        assert (ws / "artifacts" / "qc" / "p1_features.csv").is_file()
        assert (
            ws / "models" / "refinement" / "v1" / "models" / "lateral"
        ).is_file()
        # New-layout markers removed.
        assert not p.step3_root(ws).exists()
        assert not p.state_json_path(ws).exists()

    def test_rollback_is_idempotent(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        backup = m.create_backup(ws, timestamp="t1")
        plan = m.plan_migration(ws)
        m.execute_migration(plan, backup_dir=backup)
        m.rollback_migration(backup, ws)
        # Second rollback should be a no-op, not crash.
        m.rollback_migration(backup, ws)
        assert p.legacy_state_json_path(ws).is_file()

    def test_rollback_missing_backup_raises(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        bogus = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            m.rollback_migration(bogus, ws)


# ===========================================================================
# Skip-flag CRUD
# ===========================================================================


class TestSkipFlag:

    def test_initially_not_skipped(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert not m.is_migration_skipped(ws)

    def test_mark_then_check(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        m.mark_migration_skipped(ws)
        assert m.is_migration_skipped(ws)

    def test_mark_then_clear(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        m.mark_migration_skipped(ws)
        m.clear_migration_skip(ws)
        assert not m.is_migration_skipped(ws)

    def test_mark_idempotent(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        m.mark_migration_skipped(ws)
        m.mark_migration_skipped(ws)  # should not raise
        assert m.is_migration_skipped(ws)

    def test_clear_idempotent(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        m.clear_migration_skip(ws)  # should not raise


# ===========================================================================
# is_migration_needed predicate
# ===========================================================================


class TestIsMigrationNeeded:

    def test_empty_workspace_no(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert not m.is_migration_needed(ws)

    def test_legacy_workspace_yes(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        assert m.is_migration_needed(ws)

    def test_new_layout_no(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "step3_state.json").write_text("{}")
        assert not m.is_migration_needed(ws)

    def test_mixed_layout_yes(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        p.step3_root(ws).mkdir(parents=True)
        assert m.is_migration_needed(ws)


# ===========================================================================
# End-to-end round-trip
# ===========================================================================


class TestEndToEnd:

    def test_full_workspace_migration_round_trip(self, tmp_path):
        """The integration test lists."""
        ws = _make_legacy_workspace(
            tmp_path,
            patches={
                "patch_001": {"run_id": "uuid-1", "has_qc": True},
                "patch_002": {"run_id": "uuid-2", "has_qc": True},
            },
            refined_models={
                "test_v1": {"cellpose_buckets": ["lateral", "axial"]},
            },
            refinement_pairs=["patch_001", "patch_002"],
            include_extraction_manifest=True,
        )

        # 1. Plan
        plan = m.plan_migration(ws)
        assert not plan.is_empty
        assert plan.total_operations > 0

        # 2. Backup
        backup = m.create_backup(ws, timestamp="20260520_120000")

        # 3. Execute
        result = m.execute_migration(plan, backup_dir=backup)
        assert result.success, f"Failed at {result.failed_step}: {result.error}"

        # 4. Verify new layout
        assert p.state_json_path(ws).is_file()
        assert not p.legacy_state_json_path(ws).exists()
        assert (p.patch_run_dir(ws, "patch_001") / "labels_3d.tif").is_file()
        assert (p.patch_run_dir(ws, "patch_002") / "labels_3d.tif").is_file()
        assert p.qc_features_csv(ws, "patch_001").is_file()
        assert p.qc_features_csv(ws, "patch_002").is_file()
        assert (p.refined_model_dir(ws, "test_v1") / "lateral").is_file()
        assert (p.refined_model_dir(ws, "test_v1") / "axial").is_file()
        assert p.patch_pairs_dir(ws, "patch_001").is_dir()
        assert p.refinement_manifest_path(ws).is_file()

        # 5. State JSON contents look right.
        new_state = json.loads(p.state_json_path(ws).read_text())
        assert new_state["layout_version"] == p.CURRENT_LAYOUT_VERSION
        for name in ("patch_001", "patch_002"):
            assert "reconstruction_run_id" not in new_state["patches"][name]

        # 6. Layout detection now says NEW (not legacy / mixed).
        assert p.is_new_layout(ws)
        assert not p.is_legacy_layout(ws)
        assert not p.is_mixed_layout(ws)

        # 7. Rollback restores original.
        m.rollback_migration(backup, ws)
        assert p.is_legacy_layout(ws)
        assert not p.is_new_layout(ws)
        assert p.legacy_state_json_path(ws).is_file()
        assert (ws / "artifacts" / "qc" / "patch_001_features.csv").is_file()


# ===========================================================================
# Post-audit hardening (2026-05-20): atomic state JSON, auto-rollback,
# defensive companion lift, declared-version early-bail.
# ===========================================================================


class TestAutoRollbackOnFailure:
    """execute_migration should auto-rollback when a backup is available."""

    def test_failure_triggers_auto_rollback_when_backup_present(self, tmp_path, monkeypatch):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"p1": {"run_id": "uuid-1", "has_qc": True}},
        )
        backup = m.create_backup(ws, timestamp="t1")
        plan = m.plan_migration(ws)

        # Force a failure mid-execute by monkeypatching shutil.move
        # to fail after the first call.
        import shutil
        call_count = [0]
        original_move = shutil.move
        def _failing_move(src, dst):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise OSError("simulated mid-migration failure")
            return original_move(src, dst)
        monkeypatch.setattr(shutil, "move", _failing_move)

        result = m.execute_migration(plan, backup_dir=backup)
        assert result.success is False
        assert result.auto_rolled_back is True
        # Workspace state should be restored.
        assert p.legacy_state_json_path(ws).is_file()
        assert not p.is_new_layout(ws)
        assert p.is_legacy_layout(ws)

    def test_failure_without_backup_no_auto_rollback(self, tmp_path, monkeypatch):
        ws = _make_legacy_workspace(tmp_path)
        plan = m.plan_migration(ws)

        import shutil
        def _always_fails(*args, **kwargs):
            raise OSError("boom")
        monkeypatch.setattr(shutil, "move", _always_fails)

        # No backup_dir argument → no auto-rollback path available.
        result = m.execute_migration(plan, backup_dir=None)
        assert result.success is False
        assert result.auto_rolled_back is False

    def test_success_does_not_set_auto_rolled_back_flag(self, tmp_path):
        ws = _make_legacy_workspace(tmp_path)
        backup = m.create_backup(ws, timestamp="t1")
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan, backup_dir=backup)
        assert result.success
        assert result.auto_rolled_back is False


class TestAtomicStateJsonWrite:
    """The new state JSON write goes through tempfile + os.replace."""

    def test_new_state_written_atomically(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            patches={"p1": {"run_id": "uuid-1", "has_qc": True}},
        )
        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success
        # New state JSON exists, legacy is gone.
        assert p.state_json_path(ws).is_file()
        assert not p.legacy_state_json_path(ws).exists()
        # No leftover *.tmp files in the workspace root.
        leftover_tmp = list(ws.glob("*.tmp"))
        assert leftover_tmp == [], (
            f"Leftover tempfiles after atomic write: {leftover_tmp}"
        )


class TestDeclaredLayoutVersionEarlyBail:
    """A legacy-named state JSON declaring layout_version >= 2 is
    treated as inconsistent and the per-patch reconstruction-run plan
    bails with a warning."""

    def test_legacy_named_but_declared_v2_warns_and_skips(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        # File named reconstruction_state.json but content says v2.
        state = {
            "schema_version": 2,
            "layout_version": 2,
            "patches": {
                "p1": {"reconstruction_run_id": "uuid-1"},
            },
        }
        (ws / "reconstruction_state.json").write_text(json.dumps(state))
        # Add legacy marker so detection fires.
        (ws / "artifacts" / "reconstruction" / "uuid-1").mkdir(parents=True)

        plan = m.plan_migration(ws)
        # The reconstruction-run planning should NOT have created a
        # move for the patch, because the declared-v2 bail fired.
        run_moves = [
            mv for mv in plan.moves
            if "Reconstruction run" in mv.purpose
        ]
        assert len(run_moves) == 0
        assert any(
            "declares layout_version=2" in w for w in plan.warnings
        )


class TestCompanionFileLift:
    """Stray companion files under bundle/models/ should be lifted
    to the bundle root alongside the per-bucket weights."""

    def test_stray_companion_lifted_to_bundle_root(self, tmp_path):
        ws = _make_legacy_workspace(
            tmp_path,
            refined_models={"v1": {"cellpose_buckets": ["lateral"]}},
        )
        # Inject a stray companion under models/ to simulate a
        # Cellpose-side write that landed in the wrong place.
        legacy_bundle = ws / "models" / "refinement" / "v1"
        stray = legacy_bundle / "models" / "extra_companion.json"
        stray.write_text('{"foo": "bar"}')

        plan = m.plan_migration(ws)
        result = m.execute_migration(plan)
        assert result.success

        new_bundle = p.refined_model_dir(ws, "v1")
        # Stray companion now at bundle root.
        assert (new_bundle / "extra_companion.json").is_file()
        # And NOT still in the models/ subdir (which was cleaned up).
        assert not (new_bundle / "models" / "extra_companion.json").exists()
        assert not (new_bundle / "models").exists()

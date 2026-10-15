"""Tests for the Step 3 multi-run state + layout migration (Phase 0).

Pure-Python — no Qt / napari.  Covers:

* :func:`wrap_step3_state_to_v3` / :func:`unwrap_step3_state_v3`
  round-trip (from flat-v1 AND layout-migrated flat-v2 inputs) +
  active_run_id honoring + validation.
* :func:`merge_forward_runs` no-clobber merge.
* plan / execute / rollback of the physical
  ``runs/{patch}`` → ``runs/default/reconstructions/{patch}`` reshape
  + the per-patch ``reconstruction_run_id`` drop, incl. auto-rollback
  on partial failure + the dst-orphan guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from napari_epicycle.fusion_reconstruction import _migrate_step3_runs as m
from napari_epicycle.fusion_reconstruction import _paths


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants():
    assert m.DEFAULT_RUN_ID == "default"
    assert m.MIGRATION_SKIPPED_SENTINEL == ".step3_runs_migration_skipped"


# ---------------------------------------------------------------------------
# wrap / unwrap (pure)
# ---------------------------------------------------------------------------


class TestWrapUnwrap:

    def _v1(self):
        return {
            "schema_version": 1,
            "patches": {
                "patch_a.tif": {"status": "curating"},
                "patch_b.tif": {"status": "finalized"},
            },
            "active_patch": "patch_a.tif",
            "loaded_stack_dir": "/stacks",
            "downsample_mode": True,
            "refinement_extraction_config": {"lateral": {"min_size": 10}},
            "last_updated": "2026-04-01T00:00:00+00:00",
            "model_source_mode": "import",
            "lateral_model_path": "/m/lateral",
            "axial_model_path": "/m/axial",
        }

    def _v2_flat(self):
        """The layout-migrated FLAT-v2 shape produced by
        ``_migrate_step3_layout`` — schema_version 2, layout_version 2,
        ``patches`` still TOP-LEVEL (not yet runs-nested)."""
        return {
            "schema_version": 2,
            "layout_version": 2,
            "patches": {
                "patch_a.tif": {"status": "curating"},
                "patch_b.tif": {"status": "finalized"},
            },
            "active_patch": "patch_a.tif",
            "loaded_stack_dir": "/stacks",
            "downsample_mode": False,
            "last_updated": "2026-05-20T00:00:00+00:00",
            "model_source_mode": "auto",
        }

    def test_wrap_basic_shape(self):
        v3 = m.wrap_step3_state_to_v3(self._v1())
        assert v3["schema_version"] == 3
        assert v3["active_run_id"] == "default"
        assert "default" in v3["runs"]
        # patches + model fields under the run
        run = v3["runs"]["default"]
        assert run["patches"]["patch_a.tif"]["status"] == "curating"
        assert run["model_source_mode"] == "import"
        assert run["lateral_model_path"] == "/m/lateral"
        # run-manifest defaults seeded
        assert run["display_name"] == "Default"
        assert run["target"] is None
        assert run["created_from"] is None

    def test_wrap_output_schema_is_3(self):
        """The runs-envelope MUST be schema_version 3, NOT 2 (2 is taken
        by _migrate_step3_layout's flat shape)."""
        assert m.wrap_step3_state_to_v3(self._v1())["schema_version"] == 3
        assert m.wrap_step3_state_to_v3(self._v2_flat())["schema_version"] == 3

    def test_wrap_workspace_keys_stay_top_level(self):
        v3 = m.wrap_step3_state_to_v3(self._v1())
        assert v3["active_patch"] == "patch_a.tif"
        assert v3["loaded_stack_dir"] == "/stacks"
        assert v3["downsample_mode"] is True
        assert v3["refinement_extraction_config"] == {"lateral": {"min_size": 10}}
        assert v3["last_updated"] == "2026-04-01T00:00:00+00:00"
        # NOT under the run
        run = v3["runs"]["default"]
        assert "patches" in run
        assert "active_patch" not in run
        assert "downsample_mode" not in run

    def test_round_trip_v1_equals_original_minus_schema_bump(self):
        v1 = self._v1()
        v3 = m.wrap_step3_state_to_v3(v1)
        flat = m.unwrap_step3_state_v3(v3, "default")
        # schema_version is bumped to 3 on the way out; ignore it for the
        # equality, compare everything else verbatim.
        expected = dict(v1)
        expected["schema_version"] = 3
        assert flat == expected

    def test_round_trip_flat_v2_lossless_layout_version_top_level(self):
        """The layout-migrated flat-v2 (schema 2, top-level patches +
        layout_version) wraps to a v3 envelope with patches under
        runs[default] and layout_version still TOP-LEVEL, and unwraps
        losslessly (modulo the schema bump to 3)."""
        v2_flat = self._v2_flat()
        v3 = m.wrap_step3_state_to_v3(v2_flat)
        # patches moved under the run
        assert "patches" not in v3
        assert v3["runs"]["default"]["patches"]["patch_a.tif"]["status"] == "curating"
        # layout_version stays top-level (NOT buried under runs[default])
        assert v3["layout_version"] == 2
        assert "layout_version" not in v3["runs"]["default"]
        # round-trip lossless (schema bumped 2 -> 3)
        flat = m.unwrap_step3_state_v3(v3, "default")
        expected = dict(v2_flat)
        expected["schema_version"] = 3
        assert flat == expected

    def test_wrap_honors_active_run_id(self):
        v3 = m.wrap_step3_state_to_v3(self._v1(), active_run_id="cells")
        assert v3["active_run_id"] == "cells"
        assert "cells" in v3["runs"]
        assert "default" not in v3["runs"]
        # non-default display_name defaults to the slug
        assert v3["runs"]["cells"]["display_name"] == "cells"
        assert v3["runs"]["cells"]["patches"]["patch_a.tif"]["status"] == "curating"

    def test_wrap_rejects_dict(self):
        with pytest.raises(ValueError):
            m.wrap_step3_state_to_v3(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_wrap_rejects_unexpected_schema(self):
        with pytest.raises(ValueError):
            m.wrap_step3_state_to_v3({"schema_version": 99})

    def test_wrap_rejects_already_v3_envelope(self):
        """schema_version >= 3 is already a runs-envelope — reject it."""
        with pytest.raises(ValueError):
            m.wrap_step3_state_to_v3({"schema_version": 3, "runs": {}})

    def test_wrap_accepts_absent_schema(self):
        v3 = m.wrap_step3_state_to_v3({"patches": {"x": {"status": "predicted"}}})
        assert v3["schema_version"] == 3
        assert v3["runs"]["default"]["patches"]["x"]["status"] == "predicted"

    def test_unwrap_fallback_to_default_run(self):
        """Unwrap on a missing active_run_id falls back to runs[default]."""
        v3 = {
            "schema_version": 3,
            "active_run_id": "default",
            "runs": {"default": {"patches": {"d.tif": {"status": "accepted"}}}},
            "active_patch": "d.tif",
        }
        flat = m.unwrap_step3_state_v3(v3, "nonexistent_run")
        assert flat["patches"]["d.tif"]["status"] == "accepted"

    def test_unwrap_drops_manifest_only_keys(self):
        v3 = {
            "schema_version": 3,
            "active_run_id": "default",
            "runs": {
                "default": {
                    "display_name": "Default",
                    "target": "cells",
                    "created_from": "modelX",
                    "patches": {},
                }
            },
        }
        flat = m.unwrap_step3_state_v3(v3, "default")
        assert "display_name" not in flat
        assert "target" not in flat
        assert "created_from" not in flat
        assert "patches" in flat


# ---------------------------------------------------------------------------
# merge_forward_runs (no-clobber)
# ---------------------------------------------------------------------------


class TestMergeForwardRuns:

    def _existing_two_runs(self):
        return {
            "schema_version": 3,
            "active_run_id": "cells",
            "runs": {
                "cells": {
                    "display_name": "Cells",
                    "target": "cells",
                    "created_from": "modelA",
                    "patches": {"c.tif": {"status": "curating"}},
                },
                "nuclei": {
                    "display_name": "Nuclei",
                    "target": "nuclei",
                    "created_from": "modelB",
                    "patches": {"n.tif": {"status": "finalized"}},
                },
            },
        }

    def test_other_run_survives_and_active_manifest_preserved(self):
        existing = self._existing_two_runs()
        # Fresh envelope for the active 'cells' run (e.g. after a save).
        # Note: a plain per-patch save would NOT re-supply the manifest
        # fields — merge_forward must restore them from disk.
        new_env = m.wrap_step3_state_to_v3(
            {
                "schema_version": 1,
                "patches": {"c.tif": {"status": "accepted"}},
            },
            active_run_id="cells",
        )
        merged = m.merge_forward_runs(new_env, existing, "cells")

        # Active run carries the NEW patch state.
        assert merged["runs"]["cells"]["patches"]["c.tif"]["status"] == "accepted"
        # OTHER run survives intact (patches + manifest fields).
        assert merged["runs"]["nuclei"]["patches"]["n.tif"]["status"] == "finalized"
        assert merged["runs"]["nuclei"]["display_name"] == "Nuclei"
        assert merged["runs"]["nuclei"]["created_from"] == "modelB"
        # Active run's manifest fields restored from disk (NOT defaulted).
        assert merged["runs"]["cells"]["display_name"] == "Cells"
        assert merged["runs"]["cells"]["target"] == "cells"
        assert merged["runs"]["cells"]["created_from"] == "modelA"

    def test_none_existing_returns_new_envelope_unchanged(self):
        new_env = m.wrap_step3_state_to_v3(
            {"patches": {"x.tif": {"status": "predicted"}}},
            active_run_id="cells",
        )
        merged = m.merge_forward_runs(new_env, None, "cells")
        assert set(merged["runs"].keys()) == {"cells"}
        assert merged["runs"]["cells"]["patches"]["x.tif"]["status"] == "predicted"

    def test_corrupt_existing_runs_handled_gracefully(self):
        new_env = m.wrap_step3_state_to_v3(
            {"patches": {}}, active_run_id="cells",
        )
        # existing 'runs' is not a dict — must not raise.
        merged = m.merge_forward_runs(new_env, {"runs": "garbage"}, "cells")
        assert set(merged["runs"].keys()) == {"cells"}
        # Non-dict existing-on-disk entirely.
        merged2 = m.merge_forward_runs(
            m.wrap_step3_state_to_v3({"patches": {}}, active_run_id="cells"),
            ["not", "a", "dict"],
            "cells",
        )
        assert set(merged2["runs"].keys()) == {"cells"}


# ---------------------------------------------------------------------------
# Pure state-content rewrite
# ---------------------------------------------------------------------------


class TestStateContentRewrite:

    def test_drops_reconstruction_run_id_flat(self):
        state = {
            "schema_version": 1,
            "patches": {
                "a.tif": {"status": "reconstructed", "reconstruction_run_id": "X"},
                "b.tif": {"status": "predicted", "run_id": "Y"},
            },
        }
        out = m.migrate_state_json_content(state)
        assert "reconstruction_run_id" not in out["patches"]["a.tif"]
        assert "run_id" not in out["patches"]["b.tif"]
        # status preserved
        assert out["patches"]["a.tif"]["status"] == "reconstructed"
        # input not mutated
        assert "reconstruction_run_id" in state["patches"]["a.tif"]

    def test_drops_reconstruction_run_id_nested(self):
        state = {
            "schema_version": 2,
            "runs": {
                "default": {
                    "patches": {"a.tif": {"reconstruction_run_id": "X"}},
                }
            },
        }
        out = m.migrate_state_json_content(state)
        assert "reconstruction_run_id" not in out["runs"]["default"]["patches"]["a.tif"]


# ---------------------------------------------------------------------------
# Physical migration: plan / execute / rollback
# ---------------------------------------------------------------------------


def _seed_flat_workspace(ws: Path) -> None:
    """Create a flat (pre-multi-run) Step 3 workspace with a couple of
    per-patch reconstruction dirs + a v1 state file."""
    runs = _paths.runs_dir(ws)
    for patch in ("patch_001", "patch_002"):
        d = runs / patch
        d.mkdir(parents=True, exist_ok=True)
        (d / "labels_3d.tif").write_bytes(b"FAKE_TIFF")
        (d / "manifest.json").write_text(
            json.dumps({"run_id": "20260101T000000Z", "source_stack_path": patch}),
            encoding="utf-8",
        )
    state = {
        "schema_version": 1,
        "patches": {
            "patch_001": {"status": "reconstructed", "reconstruction_run_id": "X"},
            "patch_002": {"status": "predicted"},
        },
        "active_patch": "patch_001",
    }
    _paths.state_json_path(ws).write_text(json.dumps(state), encoding="utf-8")


class TestPhysicalMigration:

    def test_plan_enumerates_patch_moves(self, tmp_path):
        _seed_flat_workspace(tmp_path)
        plan = m.plan_migration(tmp_path)
        assert len(plan.moves) == 2
        srcs = {mv.src.name for mv in plan.moves}
        assert srcs == {"patch_001", "patch_002"}
        recon = _paths.reconstructions_dir(tmp_path, m.DEFAULT_RUN_ID)
        for mv in plan.moves:
            assert mv.dst.parent == recon
        assert plan.state_rewrite_path == _paths.state_json_path(tmp_path)
        assert not plan.is_empty

    def test_plan_empty_on_clean_workspace(self, tmp_path):
        plan = m.plan_migration(tmp_path)
        assert plan.is_empty

    def test_execute_moves_dirs_and_rewrites_state(self, tmp_path):
        _seed_flat_workspace(tmp_path)
        plan = m.plan_migration(tmp_path)
        backup = m.create_backup(tmp_path)
        result = m.execute_migration(plan, backup)

        assert result.success, result.error
        recon = _paths.reconstructions_dir(tmp_path, m.DEFAULT_RUN_ID)
        # Patches moved to runs/default/reconstructions/{patch}
        for patch in ("patch_001", "patch_002"):
            assert (recon / patch / "labels_3d.tif").is_file()
            # original flat location gone
            assert not (_paths.runs_dir(tmp_path) / patch).is_dir() or (
                # the run-context dir is named 'default' and lives under runs/
                patch == m.DEFAULT_RUN_ID
            )
        # state rewrite dropped reconstruction_run_id
        state = json.loads(
            _paths.state_json_path(tmp_path).read_text(encoding="utf-8")
        )
        assert "reconstruction_run_id" not in state["patches"]["patch_001"]
        assert state["patches"]["patch_001"]["status"] == "reconstructed"

    def test_execute_is_idempotent(self, tmp_path):
        _seed_flat_workspace(tmp_path)
        backup = m.create_backup(tmp_path)
        m.execute_migration(m.plan_migration(tmp_path), backup)
        # Re-plan + re-execute: nothing left to move, no crash.
        plan2 = m.plan_migration(tmp_path)
        assert len(plan2.moves) == 0
        result2 = m.execute_migration(plan2, backup)
        assert result2.success

    def test_rollback_restores_flat_layout(self, tmp_path):
        _seed_flat_workspace(tmp_path)
        plan = m.plan_migration(tmp_path)
        backup = m.create_backup(tmp_path)
        m.execute_migration(plan, backup)

        # Sanity: migrated.
        recon = _paths.reconstructions_dir(tmp_path, m.DEFAULT_RUN_ID)
        assert (recon / "patch_001" / "labels_3d.tif").is_file()

        m.rollback_migration(backup, tmp_path)

        # Flat layout restored, nested gone.
        assert (_paths.runs_dir(tmp_path) / "patch_001" / "labels_3d.tif").is_file()
        assert (_paths.runs_dir(tmp_path) / "patch_002" / "labels_3d.tif").is_file()
        assert not (recon / "patch_001").exists()
        # state restored with the original reconstruction_run_id present
        state = json.loads(
            _paths.state_json_path(tmp_path).read_text(encoding="utf-8")
        )
        assert state["patches"]["patch_001"]["reconstruction_run_id"] == "X"

    def test_rollback_missing_backup_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            m.rollback_migration(tmp_path / "nope", tmp_path)

    def test_is_migration_needed(self, tmp_path):
        assert m.is_migration_needed(tmp_path) is False
        _seed_flat_workspace(tmp_path)
        assert m.is_migration_needed(tmp_path) is True
        # After migration, the only child of runs/ is the 'default' run dir.
        backup = m.create_backup(tmp_path)
        m.execute_migration(m.plan_migration(tmp_path), backup)
        assert m.is_migration_needed(tmp_path) is False

    def test_auto_rollback_on_partial_failure(self, tmp_path, monkeypatch):
        """A failure mid-execute (here: the state rewrite raises AFTER the
        dir moves applied) auto-rolls-back the whole workspace from the
        backup — flat layout + original state restored."""
        _seed_flat_workspace(tmp_path)
        plan = m.plan_migration(tmp_path)
        backup = m.create_backup(tmp_path)

        # Force the state-rewrite step (step 2, runs after the moves) to
        # blow up so we exercise the partial-failure / rollback path.
        def _boom(_path):
            raise RuntimeError("simulated state-rewrite failure")

        monkeypatch.setattr(m, "_execute_state_rewrite", _boom)

        result = m.execute_migration(plan, backup)

        assert result.success is False
        assert result.auto_rolled_back is True
        assert "state JSON rewrite" in (result.failed_step or "")

        # Flat layout restored (moves rolled back), nested gone.
        assert (_paths.runs_dir(tmp_path) / "patch_001" / "labels_3d.tif").is_file()
        assert (_paths.runs_dir(tmp_path) / "patch_002" / "labels_3d.tif").is_file()
        recon = _paths.reconstructions_dir(tmp_path, m.DEFAULT_RUN_ID)
        assert not (recon / "patch_001").exists()
        # Original state restored verbatim (reconstruction_run_id present).
        state = json.loads(
            _paths.state_json_path(tmp_path).read_text(encoding="utf-8")
        )
        assert state["patches"]["patch_001"]["reconstruction_run_id"] == "X"

    def test_dst_orphan_no_data_loss_and_no_infinite_needed(self, tmp_path):
        """src ``runs/{patch}/`` AND a pre-existing dst
        ``runs/default/reconstructions/{patch}/`` both present: execute
        must NOT clobber/lose data, must warn, and a subsequent
        is_migration_needed must NOT loop true forever."""
        # Seed a src patch dir under runs/.
        src = _paths.runs_dir(tmp_path) / "patch_dup"
        src.mkdir(parents=True, exist_ok=True)
        (src / "labels_3d.tif").write_bytes(b"SRC_DATA")
        # Pre-create the destination (already-migrated) with its own data.
        dst = _paths.reconstructions_dir(tmp_path, m.DEFAULT_RUN_ID) / "patch_dup"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "labels_3d.tif").write_bytes(b"DST_DATA")

        # plan_migration recognises the dst-orphan: no move planned, warning recorded.
        plan = m.plan_migration(tmp_path)
        assert all(mv.src != src for mv in plan.moves)
        assert any("patch_dup" in w for w in plan.warnings)

        backup = m.create_backup(tmp_path)
        result = m.execute_migration(plan, backup)
        assert result.success

        # No data loss: BOTH src and dst still present with original bytes.
        assert (src / "labels_3d.tif").read_bytes() == b"SRC_DATA"
        assert (dst / "labels_3d.tif").read_bytes() == b"DST_DATA"

        # The orphan must NOT make migration appear perpetually needed.
        assert m.is_migration_needed(tmp_path) is False


# ---------------------------------------------------------------------------
# Phase 7: sibling sub-tree moves (qc / curation / prediction_cache /
# refinement) + the default_run_nested workspace-level key.
# ---------------------------------------------------------------------------


def _seed_flat_workspace_with_siblings(ws: Path) -> None:
    """Flat Step-3 workspace with per-patch recon dirs AND the four sibling
    sub-trees (qc / curation / prediction_cache / refinement) populated."""
    _seed_flat_workspace(ws)
    # qc
    qc = _paths.qc_dir(ws, None)
    qc.mkdir(parents=True, exist_ok=True)
    (qc / "patch_001_features.csv").write_text("id\n1\n", encoding="utf-8")
    # curation
    cur = _paths.curation_root(ws, None) / "patch_001"
    cur.mkdir(parents=True, exist_ok=True)
    (cur / "tiers.csv").write_text("label,tier\n1,good\n", encoding="utf-8")
    # prediction_cache
    pc = _paths.prediction_cache_dir(ws, None)
    pc.mkdir(parents=True, exist_ok=True)
    (pc / "patch_001_xy_flows.npy").write_bytes(b"FLOWS")
    # refinement
    ref = _paths.refinement_root(ws, None)
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "manifest.json").write_text("{}", encoding="utf-8")


class TestPhase7SiblingMoves:

    def test_default_run_nested_in_workspace_keys(self):
        assert "default_run_nested" in m.WORKSPACE_LEVEL_KEYS_V1

    def test_plan_includes_four_sibling_moves(self, tmp_path):
        _seed_flat_workspace_with_siblings(tmp_path)
        plan = m.plan_migration(tmp_path)
        labels = {mv.dst.name for mv in plan.moves if mv.kind == "dir"}
        # 2 patch dirs (under reconstructions/) + 4 sibling dirs.
        assert {"qc", "curation", "prediction_cache", "refinement"} <= labels
        # The siblings target runs/default/{label}
        run_root = _paths.run_root(tmp_path, m.DEFAULT_RUN_ID)
        for label in ("qc", "curation", "prediction_cache", "refinement"):
            assert any(
                mv.dst == run_root / label for mv in plan.moves
            ), f"missing sibling move for {label}"

    def test_shared_trees_not_moved(self, tmp_path):
        """snapshots / training_archive / batch_refine_recipes.json stay put."""
        _seed_flat_workspace_with_siblings(tmp_path)
        # Seed the workspace-shared trees too.
        snaps = _paths.snapshots_dir(tmp_path)
        snaps.mkdir(parents=True, exist_ok=True)
        (snaps / "snap1.tif").write_bytes(b"S")
        ta = _paths.training_archive_root(tmp_path)
        ta.mkdir(parents=True, exist_ok=True)
        recipes = _paths.batch_refine_recipes_path(tmp_path)
        recipes.write_text("{}", encoding="utf-8")

        plan = m.plan_migration(tmp_path)
        moved_srcs = {mv.src for mv in plan.moves}
        assert snaps not in moved_srcs
        assert ta not in moved_srcs
        # No move touches the recipes file or the shared dirs.
        for mv in plan.moves:
            assert "snapshots" not in mv.dst.parts[-2:]
            assert "training_archive" not in mv.dst.parts[-2:]

    def test_full_execute_moves_all_subtrees(self, tmp_path):
        _seed_flat_workspace_with_siblings(tmp_path)
        backup = m.create_backup(tmp_path)
        result = m.execute_migration(m.plan_migration(tmp_path), backup)
        assert result.success, result.error

        run_root = _paths.run_root(tmp_path, m.DEFAULT_RUN_ID)
        # Reconstructions nested.
        assert (run_root / "reconstructions" / "patch_001" / "labels_3d.tif").is_file()
        # Siblings nested.
        assert (run_root / "qc" / "patch_001_features.csv").is_file()
        assert (run_root / "curation" / "patch_001" / "tiers.csv").is_file()
        assert (run_root / "prediction_cache" / "patch_001_xy_flows.npy").is_file()
        assert (run_root / "refinement" / "manifest.json").is_file()
        # Flat siblings gone.
        assert not _paths.qc_dir(tmp_path, None).exists()
        assert not _paths.curation_root(tmp_path, None).exists()
        assert not _paths.prediction_cache_dir(tmp_path, None).exists()
        assert not _paths.refinement_root(tmp_path, None).exists()
        # Resolved via the per-run helpers (run_id="default") matches.
        assert _paths.qc_features_csv(
            tmp_path, "patch_001", m.DEFAULT_RUN_ID
        ).is_file()

    def test_rollback_restores_all_subtrees(self, tmp_path):
        _seed_flat_workspace_with_siblings(tmp_path)
        backup = m.create_backup(tmp_path)
        m.execute_migration(m.plan_migration(tmp_path), backup)
        m.rollback_migration(backup, tmp_path)
        # Flat layout restored for every sub-tree.
        assert _paths.qc_dir(tmp_path, None).is_dir()
        assert (_paths.qc_dir(tmp_path, None) / "patch_001_features.csv").is_file()
        assert (_paths.curation_root(tmp_path, None) / "patch_001" / "tiers.csv").is_file()
        assert (_paths.prediction_cache_dir(tmp_path, None) / "patch_001_xy_flows.npy").is_file()
        assert (_paths.refinement_root(tmp_path, None) / "manifest.json").is_file()
        # Nested gone.
        assert not _paths.run_root(tmp_path, m.DEFAULT_RUN_ID).exists()

    def test_is_migration_needed_for_siblings_only(self, tmp_path):
        """A workspace with ONLY a flat sibling sub-tree (no runs/ patch
        dirs) still reports migration needed."""
        qc = _paths.qc_dir(tmp_path, None)
        qc.mkdir(parents=True, exist_ok=True)
        (qc / "x_features.csv").write_text("a\n", encoding="utf-8")
        assert m.is_migration_needed(tmp_path) is True

    def test_idempotent_full_migration(self, tmp_path):
        _seed_flat_workspace_with_siblings(tmp_path)
        backup = m.create_backup(tmp_path)
        m.execute_migration(m.plan_migration(tmp_path), backup)
        assert m.is_migration_needed(tmp_path) is False
        plan2 = m.plan_migration(tmp_path)
        assert len(plan2.moves) == 0
        assert m.execute_migration(plan2, backup).success

    def test_sibling_dst_orphan_no_clobber(self, tmp_path):
        """A flat qc/ AND a pre-existing runs/default/qc/ both present:
        execute leaves both in place + warns, no infinite-needed loop."""
        qc_flat = _paths.qc_dir(tmp_path, None)
        qc_flat.mkdir(parents=True, exist_ok=True)
        (qc_flat / "a.csv").write_bytes(b"FLAT")
        qc_nested = _paths.qc_dir(tmp_path, m.DEFAULT_RUN_ID)
        qc_nested.mkdir(parents=True, exist_ok=True)
        (qc_nested / "a.csv").write_bytes(b"NESTED")

        plan = m.plan_migration(tmp_path)
        assert all(mv.src != qc_flat for mv in plan.moves)
        assert any("qc" in w for w in plan.warnings)
        backup = m.create_backup(tmp_path)
        assert m.execute_migration(plan, backup).success
        assert (qc_flat / "a.csv").read_bytes() == b"FLAT"
        assert (qc_nested / "a.csv").read_bytes() == b"NESTED"
        assert m.is_migration_needed(tmp_path) is False


# ---------------------------------------------------------------------------
# HIGH-1: named-run context dirs must NEVER be swept into the default run.
# ---------------------------------------------------------------------------


def _seed_named_run_context(ws: Path, run_id: str = "cells") -> Path:
    """Create a NAMED-run context dir (``runs/{run_id}/run_manifest.json`` +
    a nested per-patch reconstruction) — this is a run CONTEXT, NOT a
    default-run per-patch dir, and must NOT be moved by the migration.

    Returns the run-context dir path.
    """
    run_root = _paths.run_root(ws, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "run_id": run_id,
                    "display_name": run_id}),
        encoding="utf-8",
    )
    nested = _paths.reconstructions_dir(ws, run_id) / "patch_001"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "labels_3d.tif").write_bytes(b"NAMED_RUN_DATA")
    return run_root


class TestNamedRunPreservation:
    """HIGH-1 regression: a workspace that already has NAMED runs must not
    have its run-context dirs (``runs/{slug}/``) swept into
    ``runs/default/reconstructions/{slug}/`` by the default-run convert."""

    def test_plan_skips_named_run_context_dir(self, tmp_path):
        # A default-run patch dir + a NAMED-run context dir side-by-side
        # under runs/.
        _seed_flat_workspace(tmp_path)  # runs/patch_001, runs/patch_002
        run_root = _seed_named_run_context(tmp_path, "cells")

        plan = m.plan_migration(tmp_path)

        moved_srcs = {mv.src for mv in plan.moves}
        # The named-run context dir is NEVER a move source.
        assert run_root not in moved_srcs, (
            "named-run context dir 'cells' was scheduled for a move — "
            "it would orphan the whole named run"
        )
        # No move destination lands the named run under the default run.
        recon_default = _paths.reconstructions_dir(tmp_path, m.DEFAULT_RUN_ID)
        assert (recon_default / "cells") not in {mv.dst for mv in plan.moves}
        # The real default-run patch dirs ARE still planned for a move.
        moved_names = {mv.src.name for mv in plan.moves}
        assert {"patch_001", "patch_002"} <= moved_names

    def test_execute_leaves_named_run_intact(self, tmp_path):
        _seed_flat_workspace(tmp_path)
        run_root = _seed_named_run_context(tmp_path, "cells")

        backup = m.create_backup(tmp_path)
        result = m.execute_migration(m.plan_migration(tmp_path), backup)
        assert result.success, result.error

        # The named run is untouched: context dir + manifest + nested data.
        assert run_root.is_dir()
        assert (run_root / "run_manifest.json").is_file()
        nested = _paths.reconstructions_dir(tmp_path, "cells") / "patch_001"
        assert nested.is_file() is False  # it's a dir
        assert (nested / "labels_3d.tif").read_bytes() == b"NAMED_RUN_DATA"

        # It was NOT swept under the default run.
        recon_default = _paths.reconstructions_dir(tmp_path, m.DEFAULT_RUN_ID)
        assert not (recon_default / "cells").exists()
        # The real default-run patches DID migrate.
        assert (recon_default / "patch_001" / "labels_3d.tif").is_file()
        assert (recon_default / "patch_002" / "labels_3d.tif").is_file()

    def test_is_migration_needed_ignores_named_run(self, tmp_path):
        """A workspace whose ONLY non-default child of runs/ is a named-run
        context dir reports NO pending default-run migration."""
        _seed_named_run_context(tmp_path, "cells")
        # No flat default-run patch dirs, no flat siblings → nothing pending.
        assert m.is_migration_needed(tmp_path) is False


# ---------------------------------------------------------------------------
# Skip sentinel
# ---------------------------------------------------------------------------


class TestSkipSentinel:

    def test_skip_lifecycle(self, tmp_path):
        assert m.is_migration_skipped(tmp_path) is False
        m.mark_migration_skipped(tmp_path)
        assert m.is_migration_skipped(tmp_path) is True
        m.clear_migration_skip(tmp_path)
        assert m.is_migration_skipped(tmp_path) is False


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------


def test_all_exports_defined():
    for name in m.__all__:
        assert hasattr(m, name), f"__all__ lists {name!r} but it doesn't exist"

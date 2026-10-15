"""Tests for the Step 1 layout migration (_migrate_step1_layout).

Real filesystem, no Qt. Covers detection, plan, execute, idempotency,
corrupt/orphan sidecars, conflicts, backup+rollback, registry continuity,
and skip persistence.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation import _migrate_step1_layout as mig
from napari_epicycle.data_preparation._paths import (
    step1_patches_dir,
    step1_state_path,
)
from napari_epicycle.data_preparation._state import load_state
from napari_epicycle.data_preparation.logic import find_next_patch_index


def _write_legacy_patch(ws, name, *, sidecar=True, bad_sidecar=False, ext=".tiff"):
    """Create a legacy flat patch (TIFF + sidecar) at the workspace root."""
    data = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    tifffile.imwrite(os.path.join(ws, f"{name}{ext}"), data, metadata={"axes": "CZYX"})
    if sidecar:
        path = os.path.join(ws, f"{name}_metadata.json")
        if bad_sidecar:
            open(path, "w").write("{not valid json")
        else:
            json.dump(
                {
                    "patch_name": name,
                    "channel_names": ["a", "b"],
                    "original_shape": [2, 3, 8, 8],
                    "cropped_shape": [2, 3, 8, 8],
                    "selected_z_range": [0, 3],
                    "bounding_box": [[0, 0], [0, 8], [8, 8], [8, 0]],
                },
                open(path, "w"),
            )


# ---------------------------------------------------------------------------
# Detection / plan
# ---------------------------------------------------------------------------
class TestDetection:
    def test_needed_for_legacy_root_patches(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            _write_legacy_patch(ws, "kidney_patch_2")
            assert mig.is_migration_needed(ws) is True
            plan = mig.plan_migration(ws)
            assert {p.patch_name for p in plan.patches} == {
                "kidney_patch_1", "kidney_patch_2"}

    def test_not_needed_for_clean_workspace(self):
        with tempfile.TemporaryDirectory() as ws:
            assert mig.is_migration_needed(ws) is False

    def test_orphan_sidecar_skipped(self):
        with tempfile.TemporaryDirectory() as ws:
            # sidecar with no TIFF
            json.dump({"patch_name": "p"}, open(
                os.path.join(ws, "p_metadata.json"), "w"))
            plan = mig.plan_migration(ws)
            assert plan.is_empty

    def test_bare_tiff_without_sidecar_ignored(self):
        with tempfile.TemporaryDirectory() as ws:
            tifffile.imwrite(os.path.join(ws, "source.tiff"),
                             np.zeros((4, 4), dtype=np.uint8))
            assert mig.is_migration_needed(ws) is False


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
class TestExecute:
    def test_moves_pair_into_patches_dir(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            plan = mig.plan_migration(ws)
            res = mig.execute_migration(plan)
            pdir = step1_patches_dir(ws)
            assert (pdir / "kidney_patch_1.tiff").exists()
            assert (pdir / "kidney_patch_1_metadata.json").exists()
            # gone from root
            assert not os.path.exists(os.path.join(ws, "kidney_patch_1.tiff"))
            assert not os.path.exists(os.path.join(ws, "kidney_patch_1_metadata.json"))
            assert res.moved == 1
            assert res.registered == 1

    def test_builds_registry(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            _write_legacy_patch(ws, "kidney_patch_2")
            mig.execute_migration(mig.plan_migration(ws))
            st = load_state(ws)
            names = {p["patch_name"] for p in st["patches"]}
            assert names == {"kidney_patch_1", "kidney_patch_2"}
            entry = next(p for p in st["patches"] if p["patch_name"] == "kidney_patch_1")
            assert entry["relative_path"] == "artifacts/step1/patches/kidney_patch_1.tiff"
            assert entry["source_id"] == ""  # provenance unknown for legacy

    def test_idempotent_second_run_noop(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            mig.execute_migration(mig.plan_migration(ws))
            # Second plan finds nothing.
            assert mig.is_migration_needed(ws) is False
            plan2 = mig.plan_migration(ws)
            assert plan2.is_empty

    def test_corrupt_sidecar_still_migrates(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1", bad_sidecar=True)
            plan = mig.plan_migration(ws)
            assert plan.patches[0].sidecar_readable is False
            mig.execute_migration(plan)
            pdir = step1_patches_dir(ws)
            assert (pdir / "kidney_patch_1.tiff").exists()
            assert (pdir / "kidney_patch_1_metadata.json").exists()
            assert any(p["patch_name"] == "kidney_patch_1"
                       for p in load_state(ws)["patches"])

    def test_tif_extension_preserved(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1", ext=".tif")
            mig.execute_migration(mig.plan_migration(ws))
            pdir = step1_patches_dir(ws)
            assert (pdir / "kidney_patch_1.tif").exists()
            entry = load_state(ws)["patches"][0]
            assert entry["relative_path"].endswith("kidney_patch_1.tif")

    def test_conflict_target_exists_is_skipped(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            # Pre-create a target in the patches dir.
            pdir = step1_patches_dir(ws)
            os.makedirs(pdir)
            open(pdir / "kidney_patch_1.tiff", "w").write("EXISTING")
            plan = mig.plan_migration(ws)
            assert plan.is_empty
            assert "kidney_patch_1" in plan.conflicts
            # Root original untouched; target not overwritten.
            assert os.path.exists(os.path.join(ws, "kidney_patch_1.tiff"))
            assert (pdir / "kidney_patch_1.tiff").read_text() == "EXISTING"

    def test_preserves_pre_existing_state(self):
        """Migrating into a workspace that already has new-layout patches keeps them."""
        with tempfile.TemporaryDirectory() as ws:
            from napari_epicycle.data_preparation._export import persist_patch
            persist_patch(ws, "kidney_patch_1", np.zeros((1, 2, 4, 4), np.uint8),
                          channel_names=["c"], original_shape=(1, 2, 4, 4),
                          cropped_shape=(1, 2, 4, 4), selected_z_range=[0, 2],
                          bounding_box="x", source_path="D:/s", timestamp="t")
            _write_legacy_patch(ws, "legacy_patch_1")
            mig.execute_migration(mig.plan_migration(ws))
            names = {p["patch_name"] for p in load_state(ws)["patches"]}
            assert names == {"kidney_patch_1", "legacy_patch_1"}


# ---------------------------------------------------------------------------
# Cumulative index continuity after migration
# ---------------------------------------------------------------------------
class TestIndexContinuity:
    def test_index_continues_after_migration(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            _write_legacy_patch(ws, "kidney_patch_2")
            mig.execute_migration(mig.plan_migration(ws))
            # Registry now drives the index → next is 3 (no reset, no clobber).
            assert find_next_patch_index(ws, "kidney_patch") == 3


# ---------------------------------------------------------------------------
# Backup + rollback
# ---------------------------------------------------------------------------
class TestRollback:
    def test_rollback_restores_root_and_removes_state(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            plan = mig.plan_migration(ws)
            res = mig.execute_migration(plan)  # make_backup=True default
            assert res.backup_dir is not None
            # After execute: moved out of root.
            assert not os.path.exists(os.path.join(ws, "kidney_patch_1.tiff"))

            mig.rollback_migration(plan, res.backup_dir)
            # Originals restored at root.
            assert os.path.exists(os.path.join(ws, "kidney_patch_1.tiff"))
            assert os.path.exists(os.path.join(ws, "kidney_patch_1_metadata.json"))
            # Migrated copies removed from patches dir.
            pdir = step1_patches_dir(ws)
            assert not (pdir / "kidney_patch_1.tiff").exists()
            # State file (which migration created) removed, since none pre-existed.
            assert not step1_state_path(ws).exists()

    def test_rollback_restores_pre_existing_state(self):
        with tempfile.TemporaryDirectory() as ws:
            from napari_epicycle.data_preparation._export import persist_patch
            persist_patch(ws, "kidney_patch_1", np.zeros((1, 2, 4, 4), np.uint8),
                          channel_names=["c"], original_shape=(1, 2, 4, 4),
                          cropped_shape=(1, 2, 4, 4), selected_z_range=[0, 2],
                          bounding_box="x", source_path="D:/s", timestamp="t")
            _write_legacy_patch(ws, "legacy_patch_1")
            plan = mig.plan_migration(ws)
            res = mig.execute_migration(plan)
            mig.rollback_migration(plan, res.backup_dir)
            # Pre-existing registry restored (only the persisted patch).
            names = {p["patch_name"] for p in load_state(ws)["patches"]}
            assert names == {"kidney_patch_1"}
            # Legacy original is back at root.
            assert os.path.exists(os.path.join(ws, "legacy_patch_1.tiff"))


# ---------------------------------------------------------------------------
# Skip persistence
# ---------------------------------------------------------------------------
class TestSkipPersistence:
    def test_mark_and_check_and_clear(self):
        with tempfile.TemporaryDirectory() as ws:
            assert mig.is_migration_skipped(ws) is False
            mig.mark_migration_skipped(ws)
            assert mig.is_migration_skipped(ws) is True
            mig.clear_migration_skipped(ws)
            assert mig.is_migration_skipped(ws) is False

    def test_skip_marker_does_not_look_like_a_patch(self):
        with tempfile.TemporaryDirectory() as ws:
            mig.mark_migration_skipped(ws)
            # The marker must not be mistaken for a legacy patch.
            assert mig.is_migration_needed(ws) is False


class TestShouldOfferMigration:
    def test_offers_when_needed_and_not_skipped(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            assert mig.should_offer_migration(ws) is True

    def test_not_offered_when_skipped(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            mig.mark_migration_skipped(ws)
            assert mig.should_offer_migration(ws) is False

    def test_not_offered_when_nothing_to_migrate(self):
        with tempfile.TemporaryDirectory() as ws:
            assert mig.should_offer_migration(ws) is False

    def test_not_offered_after_migration_completes(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            mig.execute_migration(mig.plan_migration(ws))
            assert mig.should_offer_migration(ws) is False


# ---------------------------------------------------------------------------
# Data-safety hardening (from the Task-6 adversarial audits, 2026-06-29)
# ---------------------------------------------------------------------------
class TestDataSafetyHardening:
    def test_ambiguous_tif_and_tiff_is_conflict(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1", ext=".tiff")
            tifffile.imwrite(os.path.join(ws, "kidney_patch_1.tif"),
                             np.zeros((2, 3, 8, 8), np.uint8), metadata={"axes": "CZYX"})
            plan = mig.plan_migration(ws)
            assert plan.is_empty
            assert "kidney_patch_1" in plan.conflicts
            # Neither variant moved (don't guess).
            assert os.path.exists(os.path.join(ws, "kidney_patch_1.tiff"))
            assert os.path.exists(os.path.join(ws, "kidney_patch_1.tif"))

    def test_empty_stem_metadata_skipped(self):
        with tempfile.TemporaryDirectory() as ws:
            json.dump({"patch_name": "x"}, open(os.path.join(ws, "_metadata.json"), "w"))
            open(os.path.join(ws, ".tiff"), "wb").close()
            plan = mig.plan_migration(ws)
            assert plan.is_empty and plan.conflicts == []

    def test_non_patch_sidecar_skipped(self):
        with tempfile.TemporaryDirectory() as ws:
            # Readable JSON that is NOT a patch sidecar (from some other tool).
            json.dump({"foo": "bar"}, open(os.path.join(ws, "image_metadata.json"), "w"))
            tifffile.imwrite(os.path.join(ws, "image.tiff"),
                             np.zeros((4, 4), np.uint8))
            assert mig.is_migration_needed(ws) is False

    def test_empty_plan_execute_is_noop(self):
        with tempfile.TemporaryDirectory() as ws:
            res = mig.execute_migration(mig.plan_migration(ws))
            assert res.moved == 0
            assert res.backup_dir is None
            assert not any(n.startswith(".step1_migration_backup_")
                           for n in os.listdir(ws))

    def test_backup_dir_has_unique_name(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            b = mig.create_backup(mig.plan_migration(ws))
            assert b.name.startswith(".step1_migration_backup_")
            assert (b / "root_files" / "kidney_patch_1.tiff").exists()

    def test_stale_plan_reexecute_skips_not_crash(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            plan = mig.plan_migration(ws)
            mig.execute_migration(plan)
            # Re-running the now-stale plan must not crash (source already moved).
            res2 = mig.execute_migration(plan, make_backup=False)
            assert res2.moved == 0
            assert "kidney_patch_1" in res2.skipped

    def test_remigration_does_not_duplicate_registry(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            mig.execute_migration(mig.plan_migration(ws))
            # Manually copy the migrated files back to root, then re-plan/execute.
            pdir = step1_patches_dir(ws)
            shutil.copy2(pdir / "kidney_patch_1.tiff",
                         os.path.join(ws, "kidney_patch_1.tiff"))
            shutil.copy2(pdir / "kidney_patch_1_metadata.json",
                         os.path.join(ws, "kidney_patch_1_metadata.json"))
            mig.execute_migration(mig.plan_migration(ws))
            names = [p["patch_name"] for p in load_state(ws)["patches"]]
            assert names.count("kidney_patch_1") == 1  # register_patch dedups

    def test_timestamp_is_injectable(self):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            mig.execute_migration(mig.plan_migration(ws),
                                  timestamp="2026-06-29T00:00:00Z")
            assert load_state(ws)["patches"][0]["extracted_at"] == "2026-06-29T00:00:00Z"

    def test_partial_failure_auto_rolls_back(self, monkeypatch):
        with tempfile.TemporaryDirectory() as ws:
            _write_legacy_patch(ws, "kidney_patch_1")
            _write_legacy_patch(ws, "kidney_patch_2")
            plan = mig.plan_migration(ws)
            real_move = shutil.move
            calls = {"n": 0}

            def flaky_move(src, dst):
                calls["n"] += 1
                if calls["n"] == 3:  # fail on the 2nd patch's TIFF move
                    raise OSError("simulated disk full")
                return real_move(src, dst)

            monkeypatch.setattr(mig.shutil, "move", flaky_move)
            with pytest.raises(OSError):
                mig.execute_migration(plan)

            # Auto-rollback restored BOTH originals; nothing half-migrated.
            assert os.path.exists(os.path.join(ws, "kidney_patch_1.tiff"))
            assert os.path.exists(os.path.join(ws, "kidney_patch_1_metadata.json"))
            assert os.path.exists(os.path.join(ws, "kidney_patch_2.tiff"))
            assert os.path.exists(os.path.join(ws, "kidney_patch_2_metadata.json"))
            assert not step1_state_path(ws).exists()  # created state rolled back

"""Phase 7: controller-orchestrated "Convert to unified layout" migration.

Exercises ``FusionReconstructionController._do_convert_layout`` end-to-end on
a tmp workspace WITHOUT a Qt display — the method is fully getattr/try-guarded
so a near-bare ``__new__`` instance drives the real migration + flag flip +
state reload.  Verifies:

* a successful convert returns True, flips ``default_run_nested``, physically
  moves the default run's data into ``runs/default/...``, and the resolver
  now points at the nested tree;
* an already-unified workspace is a no-op (returns False);
* an in-flight Step-3 op BLOCKS the convert (data + flag untouched).
"""

from __future__ import annotations

import json
from pathlib import Path

from napari_epicycle.fusion_reconstruction import _migrate_step3_runs as m
from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
    _active_run_path_id,
)
from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
)


def _seed_flat_workspace_with_siblings(ws: Path) -> None:
    runs = _paths.runs_dir(ws)
    for patch in ("patch_001", "patch_002"):
        d = runs / patch
        d.mkdir(parents=True, exist_ok=True)
        (d / "labels_3d.tif").write_bytes(b"FAKE_TIFF")
        (d / "manifest.json").write_text(
            json.dumps({"run_id": "R1", "source_stack_path": patch}),
            encoding="utf-8",
        )
    qc = _paths.qc_dir(ws, None)
    qc.mkdir(parents=True, exist_ok=True)
    (qc / "patch_001_features.csv").write_text("id\n1\n", encoding="utf-8")
    cur = _paths.curation_root(ws, None) / "patch_001"
    cur.mkdir(parents=True, exist_ok=True)
    (cur / "tiers.csv").write_text("label,tier\n1,good\n", encoding="utf-8")
    pc = _paths.prediction_cache_dir(ws, None)
    pc.mkdir(parents=True, exist_ok=True)
    (pc / "patch_001_xy_flows.npy").write_bytes(b"FLOWS")
    ref = _paths.refinement_root(ws, None)
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "manifest.json").write_text("{}", encoding="utf-8")
    state = {
        "schema_version": 1,
        "patches": {
            "patch_001": {"status": "reconstructed"},
            "patch_002": {"status": "predicted"},
        },
        "active_patch": "patch_001",
    }
    _paths.state_json_path(ws).write_text(json.dumps(state), encoding="utf-8")


def _bare_controller(ws: Path) -> FusionReconstructionController:
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c.working_dir = str(ws)
    c.viewer = None
    # Clean unsaved-changes guard (no Qt modals).
    c._save_in_flight = False
    c._tiers_dirty = False
    c._curation_dirty = False
    # Avoid Qt dialogs on the abort path.
    c._show_run_error = lambda *a, **k: None
    sm = ReconstructionStateManager()
    sm.load(str(ws))
    c.state_manager = sm
    return c


def test_convert_moves_data_and_flips_flag(tmp_path):
    _seed_flat_workspace_with_siblings(tmp_path)
    c = _bare_controller(tmp_path)
    assert c.state_manager.get_default_run_nested() is False

    ok = c._do_convert_layout()
    assert ok is True

    # Flag flipped + persisted (reload picks it up).
    assert c.state_manager.get_default_run_nested() is True
    reloaded = ReconstructionStateManager()
    reloaded.load(str(tmp_path))
    assert reloaded.get_default_run_nested() is True

    # Data physically moved into runs/default/.
    run_root = _paths.run_root(tmp_path, m.DEFAULT_RUN_ID)
    assert (run_root / "reconstructions" / "patch_001" / "labels_3d.tif").is_file()
    assert (run_root / "qc" / "patch_001_features.csv").is_file()
    assert (run_root / "curation" / "patch_001" / "tiers.csv").is_file()
    assert (run_root / "prediction_cache" / "patch_001_xy_flows.npy").is_file()
    assert (run_root / "refinement" / "manifest.json").is_file()
    # Flat siblings gone.
    assert not _paths.qc_dir(tmp_path, None).exists()

    # Resolver now points the default run at the nested "default" tree.
    assert _active_run_path_id(c) == m.DEFAULT_RUN_ID

    # A backup was taken.
    backups = list(tmp_path.glob(f"{m.BACKUP_DIR_PREFIX}*"))
    assert backups, "expected a pre-convert backup directory"


def test_convert_finds_nested_recon_after_convert(tmp_path):
    """After convert the scanner resolves the nested recon dir → the patch
    status round-trips as reconstructed."""
    _seed_flat_workspace_with_siblings(tmp_path)
    c = _bare_controller(tmp_path)
    assert c._do_convert_layout() is True
    # Re-scan via a fresh manager (flag now True on disk) → nested read.
    sm = ReconstructionStateManager()
    sm.load(str(tmp_path))
    sm.scan_existing_artifacts(str(tmp_path))
    assert sm.get_patch_status("patch_001") == "reconstructed"


def test_convert_noop_when_already_unified(tmp_path):
    _seed_flat_workspace_with_siblings(tmp_path)
    c = _bare_controller(tmp_path)
    c.state_manager.set_default_run_nested(True)
    # Already unified → no-op, returns False, no migration attempted.
    assert c._do_convert_layout() is False
    # No backup created (we aborted before create_backup).
    assert not list(tmp_path.glob(f"{m.BACKUP_DIR_PREFIX}*"))


def test_convert_blocked_when_op_in_flight(tmp_path):
    _seed_flat_workspace_with_siblings(tmp_path)
    c = _bare_controller(tmp_path)
    c._reconstruction_running = True  # a Step-3 block flag

    assert c._do_convert_layout() is False
    # Flag NOT flipped, data NOT moved, no backup.
    assert c.state_manager.get_default_run_nested() is False
    assert (_paths.runs_dir(tmp_path) / "patch_001" / "labels_3d.tif").is_file()
    assert not list(tmp_path.glob(f"{m.BACKUP_DIR_PREFIX}*"))


def test_convert_preserves_named_run(tmp_path):
    """HIGH-1 at the controller level: a workspace that already has a NAMED
    run must keep that run's context dir intact after a default-run convert
    (it must NOT be swept into runs/default/reconstructions/)."""
    _seed_flat_workspace_with_siblings(tmp_path)
    # Seed a named-run context dir alongside the default-run data.
    run_root = _paths.run_root(tmp_path, "cells")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "run_id": "cells"}),
        encoding="utf-8",
    )
    nested = _paths.reconstructions_dir(tmp_path, "cells") / "patch_001"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "labels_3d.tif").write_bytes(b"NAMED")

    c = _bare_controller(tmp_path)
    assert c._do_convert_layout() is True

    # Named run intact + NOT swept under the default run.
    assert (run_root / "run_manifest.json").is_file()
    assert (nested / "labels_3d.tif").read_bytes() == b"NAMED"
    default_recon = _paths.reconstructions_dir(tmp_path, m.DEFAULT_RUN_ID)
    assert not (default_recon / "cells").exists()
    # The real default-run patches still migrated.
    assert (default_recon / "patch_001" / "labels_3d.tif").is_file()


def test_convert_surfaces_backup_path_on_failure(tmp_path, monkeypatch):
    """MED-1: when execute_migration reports failure, the error modal body
    must surface the BACKUP PATH so the user can recover."""
    _seed_flat_workspace_with_siblings(tmp_path)
    c = _bare_controller(tmp_path)

    captured = {}
    c._show_run_error = lambda title, body: captured.update(
        {"title": title, "body": body}
    )

    # Force the data move to report a rolled-back failure.
    from napari_epicycle.fusion_reconstruction import _migrate_step3_runs as mig

    def _fake_execute(plan, backup, auto_rollback_on_failure=True):
        return mig.ExecutionResult(
            success=False,
            failed_step="step 1 — reconstruction dir moves",
            error="RuntimeError: simulated move failure",
            backup_dir=Path(backup),
            auto_rolled_back=True,
        )

    monkeypatch.setattr(mig, "execute_migration", _fake_execute)

    assert c._do_convert_layout() is False
    assert "body" in captured, "error modal was not shown"
    body = captured["body"]
    # The backup path must be in the message.
    backups = list(tmp_path.glob(f"{m.BACKUP_DIR_PREFIX}*"))
    assert backups, "expected a backup dir to have been created"
    assert str(backups[0]) in body
    assert "rolled back" in body.lower()
    # Flag NOT flipped (failure aborts before the flip).
    assert c.state_manager.get_default_run_nested() is False


def test_convert_loud_message_on_rollback_also_failed(tmp_path, monkeypatch):
    """MED-1: when BOTH the move and the rollback fail, the message is LOUD
    and still surfaces the backup path."""
    _seed_flat_workspace_with_siblings(tmp_path)
    c = _bare_controller(tmp_path)
    captured = {}
    c._show_run_error = lambda title, body: captured.update({"body": body})

    from napari_epicycle.fusion_reconstruction import _migrate_step3_runs as mig

    def _fake_execute(plan, backup, auto_rollback_on_failure=True):
        return mig.ExecutionResult(
            success=False,
            error=("RuntimeError: move failed (rollback ALSO FAILED: "
                   "OSError: disk full)"),
            backup_dir=Path(backup),
            auto_rolled_back=False,
        )

    monkeypatch.setattr(mig, "execute_migration", _fake_execute)

    assert c._do_convert_layout() is False
    body = captured.get("body", "")
    assert "Migration AND rollback failed" in body
    backups = list(tmp_path.glob(f"{m.BACKUP_DIR_PREFIX}*"))
    assert str(backups[0]) in body

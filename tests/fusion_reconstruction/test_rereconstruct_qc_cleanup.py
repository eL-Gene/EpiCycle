"""Tests for stale QC cleanup on re-reconstruction."""

import json
from pathlib import Path

import pytest


class _FakeStateManager:
    """Minimal state manager fake for testing QC cleanup."""

    def __init__(self):
        self._state = {"patches": {}}
        self._saved = False

    def set_patch_status(self, patch_name, status, **metadata):
        if patch_name not in self._state["patches"]:
            self._state["patches"][patch_name] = {}
        rec = self._state["patches"][patch_name]
        rec["status"] = status
        rec.update(metadata)

    def schedule_save(self):
        self._saved = True

    def get_patch_status(self, patch_name):
        rec = self._state["patches"].get(patch_name)
        return rec.get("status", "unprocessed") if rec else "unprocessed"


def _make_controller(tmp_path):
    """Build a minimal controller with just enough wiring for QC cleanup."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.working_dir = str(tmp_path)
    ctrl.state_manager = _FakeStateManager()
    ctrl._qc_features_df = None
    ctrl._log_lines = []

    class _FakeUI:
        class _FakeList:
            def selectedItems(self):
                return []

        loaded_patch_list = _FakeList()

    ctrl.ui = _FakeUI()

    def _append_log(msg):
        ctrl._log_lines.append(msg)

    ctrl._append_log = _append_log

    return ctrl


def _create_qc_csv(tmp_path, patch_stem, content="label,qc_tier\n1,good\n2,bad\n"):
    qc_dir = tmp_path / "artifacts" / "step3" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    csv_path = qc_dir / f"{patch_stem}_features.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


class TestCleanupStaleQc:

    def test_deletes_qc_csv(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        csv_path = _create_qc_csv(tmp_path, "patch_001")
        assert csv_path.exists()

        ctrl._cleanup_stale_qc("patch_001.tif")

        assert not csv_path.exists()

    def test_clears_qc_state_metadata(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl.state_manager._state["patches"]["patch_001.tif"] = {
            "status": "reconstructed",
            "qc_features_csv": "artifacts/qc/patch_001_features.csv",
            "qc_ran_at": "2026-03-29T12:00:00Z",
            "qc_total_labels": 42,
            "qc_triage": {"1": "good", "2": "bad"},
            "reconstruction_run_id": "20260329T120000Z",
        }

        ctrl._cleanup_stale_qc("patch_001.tif")

        record = ctrl.state_manager._state["patches"]["patch_001.tif"]
        assert "qc_features_csv" not in record
        assert "qc_ran_at" not in record
        assert "qc_total_labels" not in record
        assert "qc_triage" not in record
        # Non-QC metadata preserved
        assert record["reconstruction_run_id"] == "20260329T120000Z"

    def test_schedules_state_save(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl.state_manager._state["patches"]["p.tif"] = {
            "status": "reconstructed",
            "qc_triage": {"1": "good"},
        }

        ctrl._cleanup_stale_qc("p.tif")

        assert ctrl.state_manager._saved is True

    def test_clears_in_memory_dataframe(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._qc_features_df = "some_dataframe_placeholder"

        ctrl._cleanup_stale_qc("patch_001.tif")

        assert ctrl._qc_features_df is None

    def test_noop_when_no_qc_exists(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        # No CSV, no state metadata — should not error
        ctrl._cleanup_stale_qc("patch_001.tif")

        assert len(ctrl._log_lines) == 0

    def test_logs_removal(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        _create_qc_csv(tmp_path, "patch_001")

        ctrl._cleanup_stale_qc("patch_001.tif")

        assert any("Removed stale QC" in line for line in ctrl._log_lines)

    def test_noop_when_no_working_dir(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl.working_dir = ""
        _create_qc_csv(tmp_path, "patch_001")

        ctrl._cleanup_stale_qc("patch_001.tif")
        # CSV should still exist since cleanup was skipped
        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "patch_001_features.csv"
        assert csv_path.exists()

    def test_preserves_non_qc_state_fields(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl.state_manager._state["patches"]["patch.tif"] = {
            "status": "curating",
            "reconstructed_at": "2026-03-29T10:00:00Z",
            "reconstruction_run_id": "run123",
            "qc_features_csv": "old.csv",
            "qc_triage": {"1": "good"},
        }

        ctrl._cleanup_stale_qc("patch.tif")

        record = ctrl.state_manager._state["patches"]["patch.tif"]
        assert record["status"] == "curating"
        assert record["reconstructed_at"] == "2026-03-29T10:00:00Z"
        assert record["reconstruction_run_id"] == "run123"
        assert "qc_features_csv" not in record
        assert "qc_triage" not in record

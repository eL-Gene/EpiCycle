"""Tests for backward transitions and selective prediction.

Covers:
- State manager: reset_to_unprocessed, reset_to_predicted, loaded_stack_dir
- Controller: selective prediction, re-prediction, reset to predicted
"""

import json
import pytest
from pathlib import Path

from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
)


# ---------------------------------------------------------------------------
# State Manager Tests
# ---------------------------------------------------------------------------

class TestResetToUnprocessed:
    """Test state_manager.reset_to_unprocessed()."""

    def test_resets_status(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["p1.tiff"])
        sm.set_patch_status("p1.tiff", "predicted", predicted_at="2026-01-01")
        assert sm.get_patch_status("p1.tiff") == "predicted"

        sm.reset_to_unprocessed("p1.tiff")
        assert sm.get_patch_status("p1.tiff") == "unprocessed"

    def test_clears_prediction_metadata(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["p1.tiff"])
        sm.set_patch_status(
            "p1.tiff", "reconstructed",
            predicted_at="2026-01-01",
            reconstructed_at="2026-01-02",
            reconstruction_run_id="run_123",
        )

        sm.reset_to_unprocessed("p1.tiff")
        record = sm._state["patches"]["p1.tiff"]
        assert "predicted_at" not in record
        assert "reconstructed_at" not in record
        assert "reconstruction_run_id" not in record

    def test_noop_for_unknown_patch(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.reset_to_unprocessed("nonexistent")  # should not raise


class TestResetToPredicted:
    """Test state_manager.reset_to_predicted()."""

    def test_resets_reconstructed_to_predicted(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["p1.tiff"])
        sm.set_patch_status("p1.tiff", "reconstructed",
                            predicted_at="2026-01-01",
                            reconstructed_at="2026-01-02")

        sm.reset_to_predicted("p1.tiff")
        assert sm.get_patch_status("p1.tiff") == "predicted"

    def test_preserves_predicted_at(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["p1.tiff"])
        sm.set_patch_status("p1.tiff", "accepted",
                            predicted_at="2026-01-01",
                            reconstructed_at="2026-01-02",
                            accepted_at="2026-01-03")

        sm.reset_to_predicted("p1.tiff")
        record = sm._state["patches"]["p1.tiff"]
        assert record.get("predicted_at") == "2026-01-01"
        assert "reconstructed_at" not in record
        assert "accepted_at" not in record

    def test_clears_finalized_metadata(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["p1.tiff"])
        sm.set_patch_status("p1.tiff", "finalized",
                            finalized_at="2026-01-04")

        sm.reset_to_predicted("p1.tiff")
        record = sm._state["patches"]["p1.tiff"]
        assert "finalized_at" not in record


class TestStackDirPersistence:
    """Test loaded_stack_dir persistence in state manager."""

    def test_persist_and_restore(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.loaded_stack_dir = "/some/path"
        sm.save()  # explicit save (schedule_save uses QTimer)

        # Reload
        sm2 = ReconstructionStateManager()
        sm2.load(str(tmp_path))
        assert sm2.loaded_stack_dir == "/some/path"

    def test_default_is_none(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        assert sm.loaded_stack_dir is None

    def test_backward_compatible_with_old_state(self, tmp_path):
        """State files without loaded_stack_dir should work.

        Post-2026-05-20 cutover: state JSON renamed to step3_state.json
        (was reconstruction_state.json).  Test verifies the "old state
        shape without loaded_stack_dir" still works in the new file.
        """
        state_file = tmp_path / "step3_state.json"
        state_file.write_text(json.dumps({
            "schema_version": 1,
            "patches": {"p1.tiff": {"status": "predicted"}},
            "active_patch": None,
            "last_updated": None,
        }))

        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        assert sm.loaded_stack_dir is None
        assert sm.get_patch_status("p1.tiff") == "predicted"


# ---------------------------------------------------------------------------
# Controller Tests (using fakes)
# ---------------------------------------------------------------------------

class FakeStateManager:
    """Minimal state manager for controller tests."""

    def __init__(self, statuses=None):
        self._statuses = statuses or {}
        self._loaded_stack_dir = None

    def get_all_patch_statuses(self):
        return dict(self._statuses)

    def get_patch_status(self, name):
        return self._statuses.get(name, "unprocessed")

    def set_patch_status(self, name, status, **kw):
        self._statuses[name] = status

    def reset_to_unprocessed(self, name):
        self._statuses[name] = "unprocessed"

    def reset_to_predicted(self, name):
        self._statuses[name] = "predicted"

    def get_status_counts(self):
        counts = {}
        for s in self._statuses.values():
            counts[s] = counts.get(s, 0) + 1
        return counts

    def register_patches(self, names):
        for n in names:
            if n not in self._statuses:
                self._statuses[n] = "unprocessed"

    @property
    def loaded_stack_dir(self):
        return self._loaded_stack_dir

    @loaded_stack_dir.setter
    def loaded_stack_dir(self, v):
        self._loaded_stack_dir = v


class TrackingStateManager:
    def __init__(self, saved_stack_dir=None):
        self._loaded_stack_dir = saved_stack_dir
        self.calls = []

    def load(self, workspace):
        self.calls.append(("load", workspace))

    def scan_existing_artifacts(self, workspace):
        self.calls.append(("scan", workspace))

    def register_patches(self, names):
        self.calls.append(("register", list(names)))

    @property
    def loaded_stack_dir(self):
        return self._loaded_stack_dir

    @loaded_stack_dir.setter
    def loaded_stack_dir(self, v):
        self._loaded_stack_dir = v


class TestResetPatchesToPredicted:
    """Test controller._reset_patches_to_predicted()."""

    def test_resets_reconstructed_patches(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from unittest.mock import MagicMock

        ctrl = object.__new__(FusionReconstructionController)
        ctrl.state_manager = FakeStateManager({
            "p1.tiff": "reconstructed",
            "p2.tiff": "curating",
            "p3.tiff": "accepted",
        })
        ctrl._edited_patches = {"p2.tiff"}
        ctrl._predicting_patches = set()
        ctrl._reconstructing_patches = set()
        ctrl.ui = MagicMock()
        ctrl.ui.log_output = MagicMock()
        ctrl.ui.refresh_refinement_patch_list = MagicMock()
        ctrl.working_dir = "/fake"

        count = ctrl._reset_patches_to_predicted(
            ["p1.tiff", "p2.tiff", "p3.tiff"]
        )

        assert count == 3
        assert ctrl.state_manager.get_patch_status("p1.tiff") == "predicted"
        assert ctrl.state_manager.get_patch_status("p2.tiff") == "predicted"
        assert ctrl.state_manager.get_patch_status("p3.tiff") == "predicted"
        assert "p2.tiff" not in ctrl._edited_patches

    def test_skips_already_predicted(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from unittest.mock import MagicMock

        ctrl = object.__new__(FusionReconstructionController)
        ctrl.state_manager = FakeStateManager({
            "p1.tiff": "predicted",
            "p2.tiff": "unprocessed",
        })
        ctrl._edited_patches = set()
        ctrl._predicting_patches = set()
        ctrl._reconstructing_patches = set()
        ctrl.ui = MagicMock()
        ctrl.ui.log_output = MagicMock()
        ctrl.ui.refresh_refinement_patch_list = MagicMock()
        ctrl.working_dir = "/fake"

        count = ctrl._reset_patches_to_predicted(["p1.tiff", "p2.tiff"])
        assert count == 0  # both already at or before predicted


class TestClearPredictionCache:
    """Test controller._clear_prediction_cache()."""

    def test_removes_cache_files(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        ctrl = object.__new__(FusionReconstructionController)
        ctrl.working_dir = str(tmp_path)

        cache_dir = tmp_path / "artifacts" / "step3" / "prediction_cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "patch_1_xy_flows.npy").touch()
        (cache_dir / "patch_1_xz_flows.npy").touch()
        (cache_dir / "patch_2_xy_flows.npy").touch()

        ctrl._clear_prediction_cache("patch_1.tiff")

        remaining = list(cache_dir.iterdir())
        assert len(remaining) == 1
        assert remaining[0].name == "patch_2_xy_flows.npy"

    def test_noop_if_no_cache_dir(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        ctrl = object.__new__(FusionReconstructionController)
        ctrl.working_dir = str(tmp_path)
        # No cache dir exists — should not raise
        ctrl._clear_prediction_cache("patch_1.tiff")


class TestWorkspaceRestoreOrder:
    def test_browse_workdir_defers_artifact_scan_until_stack_dir_restore(
        self, tmp_path, monkeypatch
    ):
        from napari_epicycle.fusion_reconstruction import controller as controller_mod
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        workspace = tmp_path / "ws"
        stack_dir = workspace / "stacks"
        stack_dir.mkdir(parents=True)

        ctrl = object.__new__(FusionReconstructionController)
        ctrl.state_manager = TrackingStateManager(str(stack_dir))
        ctrl.ui = type("UI", (), {})()
        ctrl.ui.set_workdir_path = type(
            "Label", (), {"setText": lambda self, text: None}
        )()
        ctrl.working_dir = ""
        ctrl._load_stack_dir = (
            lambda path: ctrl.state_manager.calls.append(("load_stack_dir", path))
        )
        ctrl._refresh_patch_state_list = lambda: None
        ctrl._discover_workspace_models = lambda: None
        ctrl._refresh_patch_list_colors = lambda: None
        ctrl._refresh_summary_dashboard = lambda: None
        ctrl._populate_refinement_source_combo = lambda: None
        ctrl._refresh_refinement_patch_list = lambda: None
        ctrl._append_log = lambda text: None

        monkeypatch.setattr(
            controller_mod.QFileDialog,
            "getExistingDirectory",
            lambda *args, **kwargs: str(workspace),
        )

        ctrl._browse_workdir()

        assert ctrl.state_manager.calls == [
            ("load", str(workspace)),
            ("load_stack_dir", str(stack_dir)),
        ]

    def test_load_stack_dir_registers_patches_before_scanning_artifacts(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from unittest.mock import MagicMock

        stack_dir = tmp_path / "stacks"
        stack_dir.mkdir()
        (stack_dir / "patch_1.tif").write_bytes(b"")

        ctrl = object.__new__(FusionReconstructionController)
        ctrl.working_dir = str(tmp_path)
        ctrl.loaded_stack_dir = ""
        ctrl.state_manager = TrackingStateManager()
        ctrl.ui = MagicMock()
        ctrl._refresh_patch_list_colors = lambda: None
        ctrl._refresh_summary_dashboard = lambda: None
        ctrl._refresh_patch_state_list = lambda: None
        ctrl._append_log = lambda text: None

        ctrl._load_stack_dir(str(stack_dir))

        assert ctrl.state_manager.calls == [
            ("register", ["patch_1.tif"]),
            ("scan", str(tmp_path)),
        ]

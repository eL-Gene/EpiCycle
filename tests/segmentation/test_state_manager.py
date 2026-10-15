"""Tests for Step 4 StackStateManager."""

from __future__ import annotations

import json

import pytest


class TestStackStateManager:
    """Core state manager tests (Qt-free — no debounced saves)."""

    def _make_manager(self):
        from napari_epicycle.segmentation.state_manager import StackStateManager
        mgr = StackStateManager()
        return mgr

    def test_empty_state_on_init(self):
        mgr = self._make_manager()
        assert mgr.get_all_stack_statuses() == {}

    def test_load_empty_workspace(self, tmp_path):
        mgr = self._make_manager()
        state = mgr.load(str(tmp_path))
        assert isinstance(state, dict)
        assert state["schema_version"] == 2

    def test_set_and_get_status(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        mgr.set_stack_status("stack_001", "segmented")
        assert mgr.get_stack_status("stack_001") == "segmented"

    def test_default_status_is_unprocessed(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        assert mgr.get_stack_status("nonexistent") == "unprocessed"

    def test_invalid_status_raises(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid status"):
            mgr.set_stack_status("s1", "bogus_status")

    def test_all_valid_statuses_accepted(self, tmp_path):
        from napari_epicycle.segmentation.state_manager import VALID_STATUSES
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        for status in VALID_STATUSES:
            mgr.set_stack_status(f"stack_{status}", status)
            assert mgr.get_stack_status(f"stack_{status}") == status

    def test_register_stacks_no_overwrite(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        mgr.set_stack_status("s1", "reconstructed")
        mgr.register_stacks(["s1", "s2"])
        assert mgr.get_stack_status("s1") == "reconstructed"  # not overwritten
        assert mgr.get_stack_status("s2") == "unprocessed"

    def test_get_status_counts(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        mgr.set_stack_status("a", "segmented")
        mgr.set_stack_status("b", "segmented")
        mgr.set_stack_status("c", "reconstructed")
        counts = mgr.get_status_counts()
        assert counts["segmented"] == 2
        assert counts["reconstructed"] == 1
        assert counts["unprocessed"] == 0

    def test_get_all_stack_statuses(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        mgr.set_stack_status("x", "finalized")
        mgr.set_stack_status("y", "smoothed")
        result = mgr.get_all_stack_statuses()
        assert result == {"x": "finalized", "y": "smoothed"}

    def test_active_stack_property(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        assert mgr.active_stack is None
        mgr.active_stack = "my_stack"
        assert mgr.active_stack == "my_stack"

    def test_metadata_stored(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        mgr.set_stack_status("s1", "qc_extracted",
                             qc_features_csv="/path/to/features.csv",
                             labels_path="/path/to/labels.tif")
        record = mgr._state["stacks"]["s1"]
        assert record["qc_features_csv"] == "/path/to/features.csv"
        assert record["labels_path"] == "/path/to/labels.tif"

    def test_save_load_roundtrip(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        mgr.set_stack_status("s1", "smoothed", labels_path="/a/b.tif")
        mgr.active_stack = "s1"
        mgr.save()

        mgr2 = self._make_manager()
        state = mgr2.load(str(tmp_path))
        assert mgr2.get_stack_status("s1") == "smoothed"
        assert mgr2.active_stack == "s1"
        assert state["stacks"]["s1"]["labels_path"] == "/a/b.tif"

    def test_corrupt_json_is_set_aside_not_overwritten(self, tmp_path):
        """F3: the old test locked the bug in (a silent empty load that the
        next save overwrote).  The file is renamed aside and kept."""
        state_file = tmp_path / "step4_state.json"   # the root (dec. 55)
        state_file.write_text("{ broken json !!!", encoding="utf-8")
        mgr = self._make_manager()
        state = mgr.load(str(tmp_path))
        assert state["schema_version"] == 2
        assert state["stacks"] == {}
        kept = list(state_file.parent.glob("step4_state.json.corrupt-*"))
        assert len(kept) == 1
        assert kept[0].read_text(encoding="utf-8") == "{ broken json !!!"
        assert mgr.take_notices()

    def test_corrupt_legacy_json_is_left_alone(self, tmp_path):
        legacy = tmp_path / "stack_segmentation_state.json"
        legacy.write_text("{ broken json !!!", encoding="utf-8")
        mgr = self._make_manager()
        state = mgr.load(str(tmp_path))
        assert state["stacks"] == {}
        assert legacy.read_text(encoding="utf-8") == "{ broken json !!!"

    def test_reset_to_reconstructed(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        mgr.set_stack_status("s1", "smoothed",
                             qc_features_csv="/old.csv",
                             qc_tier_map={"1": "good"})
        mgr.reset_to_reconstructed("s1")
        assert mgr.get_stack_status("s1") == "reconstructed"
        record = mgr._state["stacks"]["s1"]
        assert "qc_features_csv" not in record
        assert "qc_tier_map" not in record

    def test_reset_to_unprocessed(self, tmp_path):
        mgr = self._make_manager()
        mgr.load(str(tmp_path))
        mgr.set_stack_status("s1", "finalized", labels_path="/a.tif")
        mgr.reset_to_unprocessed("s1")
        assert mgr.get_stack_status("s1") == "unprocessed"

    def test_state_file_name(self):
        from napari_epicycle.segmentation.state_manager import StackStateManager
        assert StackStateManager.STATE_FILENAME == "step4_state.json"
        assert (StackStateManager.LEGACY_STATE_FILENAME
                == "stack_segmentation_state.json")

    def test_valid_statuses_match_theme(self):
        from napari_epicycle.segmentation.state_manager import VALID_STATUSES
        from napari_epicycle.ui_shell.theme import STACK_STATUS_COLORS
        for status in VALID_STATUSES:
            assert status in STACK_STATUS_COLORS, (
                f"Status '{status}' missing from STACK_STATUS_COLORS"
            )
        assert "failed" in VALID_STATUSES

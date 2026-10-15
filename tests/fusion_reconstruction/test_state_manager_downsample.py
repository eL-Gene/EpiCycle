"""Workspace-level downsample_mode + per-patch downsample_factor —
state_manager additions for D2.

Pinned contract:
  - workspace state has ``downsample_mode: bool`` (default False)
  - getter / setter / round-trip through save+load
  - per-patch ``downsample_factor: int`` (default 1) via metadata kwargs
  - backwards-compat: old state files (no downsample_mode field) load
    with default False — no schema-version bump needed
"""

import json
from pathlib import Path

import pytest


def _make_manager(tmp_path):
    from napari_epicycle.fusion_reconstruction.state_manager import (
        ReconstructionStateManager,
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    return mgr


# ---------------------------------------------------------------------------
# Workspace-level downsample_mode
# ---------------------------------------------------------------------------


class TestDownsampleMode:
    """Workspace-level toggle for the downsample-on-load feature."""

    def test_default_false(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.get_downsample_mode() is False

    def test_setter_updates_value(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.set_downsample_mode(True)
        assert mgr.get_downsample_mode() is True
        mgr.set_downsample_mode(False)
        assert mgr.get_downsample_mode() is False

    def test_round_trip_through_disk(self, tmp_path):
        """After save+reload, the toggle value persists."""
        from napari_epicycle.fusion_reconstruction.state_manager import (
            ReconstructionStateManager,
        )
        mgr = _make_manager(tmp_path)
        mgr.set_downsample_mode(True)
        mgr.save()  # bypass QTimer debounce — no event loop in unit tests

        mgr2 = ReconstructionStateManager()
        mgr2.load(str(tmp_path))
        assert mgr2.get_downsample_mode() is True

    def test_old_state_file_loads_with_default(self, tmp_path):
        """A workspace state JSON that predates this field still loads
        cleanly with downsample_mode = False (no schema bump)."""
        from napari_epicycle.fusion_reconstruction.state_manager import (
            ReconstructionStateManager,
        )
        # Write a pre-D2 state file
        state = {
            "schema_version": 1,
            "patches": {
                "patch_a": {"status": "curating"},
            },
            "active_patch": "patch_a",
            "loaded_stack_dir": None,
            "last_updated": "2026-04-01T00:00:00+00:00",
        }
        (tmp_path / "step3_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )

        mgr = ReconstructionStateManager()
        mgr.load(str(tmp_path))
        assert mgr.get_downsample_mode() is False
        # And the patch data still loads correctly
        assert mgr.get_patch_status("patch_a") == "curating"


# ---------------------------------------------------------------------------
# Per-patch downsample_factor
# ---------------------------------------------------------------------------


class TestPatchDownsampleFactor:
    """Per-patch factor recorded at load time, read at save time."""

    def test_default_factor_is_1(self, tmp_path):
        """Patch with no factor recorded returns 1 (full-res)."""
        mgr = _make_manager(tmp_path)
        mgr.set_patch_status("patch_a", "predicted")
        assert mgr.get_patch_downsample_factor("patch_a") == 1

    def test_unknown_patch_returns_1(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.get_patch_downsample_factor("nonexistent") == 1

    def test_factor_recorded_via_set_patch_status(self, tmp_path):
        """The existing set_patch_status accepts metadata kwargs —
        verify downsample_factor=2 round-trips."""
        mgr = _make_manager(tmp_path)
        mgr.set_patch_status("patch_a", "curating", downsample_factor=2)
        assert mgr.get_patch_downsample_factor("patch_a") == 2

    def test_factor_round_trip_through_disk(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.state_manager import (
            ReconstructionStateManager,
        )
        mgr = _make_manager(tmp_path)
        mgr.set_patch_status("patch_a", "curating", downsample_factor=2)
        mgr.save()  # bypass QTimer debounce — no event loop in unit tests

        mgr2 = ReconstructionStateManager()
        mgr2.load(str(tmp_path))
        assert mgr2.get_patch_downsample_factor("patch_a") == 2

    def test_factor_can_be_updated(self, tmp_path):
        """Subsequent loads can record a different factor (e.g., user
        flips workspace toggle, reloads patch — load step records the
        new factor)."""
        mgr = _make_manager(tmp_path)
        mgr.set_patch_status("patch_a", "curating", downsample_factor=2)
        mgr.set_patch_status("patch_a", "curating", downsample_factor=1)
        assert mgr.get_patch_downsample_factor("patch_a") == 1


# ---------------------------------------------------------------------------
# Update-metadata-only convenience method
# ---------------------------------------------------------------------------


class TestUpdatePatchMetadata:
    """Convenience method to update metadata fields without changing
    the patch status (load step needs this to record factor without
    re-asserting status)."""

    def test_updates_metadata_preserving_status(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.set_patch_status("patch_a", "accepted")
        mgr.update_patch_metadata("patch_a", downsample_factor=2)
        assert mgr.get_patch_status("patch_a") == "accepted"
        assert mgr.get_patch_downsample_factor("patch_a") == 2

    def test_creates_record_if_missing(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.update_patch_metadata("patch_a", downsample_factor=2)
        # Status defaults to unprocessed for new patches.
        assert mgr.get_patch_status("patch_a") == "unprocessed"
        assert mgr.get_patch_downsample_factor("patch_a") == 2

    def test_round_trip_through_disk(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.state_manager import (
            ReconstructionStateManager,
        )
        mgr = _make_manager(tmp_path)
        mgr.set_patch_status("patch_a", "accepted")
        mgr.update_patch_metadata("patch_a", downsample_factor=2)
        mgr.save()  # bypass QTimer debounce — no event loop in unit tests

        mgr2 = ReconstructionStateManager()
        mgr2.load(str(tmp_path))
        assert mgr2.get_patch_status("patch_a") == "accepted"
        assert mgr2.get_patch_downsample_factor("patch_a") == 2


# ---------------------------------------------------------------------------
# D2.5 — Lossy-downsample-load warning per-patch dismiss flag
# ---------------------------------------------------------------------------


class TestPatchDownsampleWarningShown:
    """Per-patch flag set when user opts Continue on the lossy-load
    warning modal.  Prevents repeat nagging on subsequent loads of
    the same patch."""

    def test_default_false(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.set_patch_status("patch_a", "accepted")
        assert mgr.get_patch_downsample_warning_shown("patch_a") is False

    def test_unknown_patch_returns_false(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.get_patch_downsample_warning_shown("nonexistent") is False

    def test_set_via_update_metadata(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.set_patch_status("patch_a", "accepted")
        mgr.update_patch_metadata("patch_a", downsample_warning_shown=True)
        assert mgr.get_patch_downsample_warning_shown("patch_a") is True

    def test_round_trip_through_disk(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.state_manager import (
            ReconstructionStateManager,
        )
        mgr = _make_manager(tmp_path)
        mgr.set_patch_status("patch_a", "accepted")
        mgr.update_patch_metadata("patch_a", downsample_warning_shown=True)
        mgr.save()

        mgr2 = ReconstructionStateManager()
        mgr2.load(str(tmp_path))
        assert mgr2.get_patch_downsample_warning_shown("patch_a") is True

    def test_per_patch_isolation(self, tmp_path):
        """Setting warning_shown on patch_a does NOT affect patch_b."""
        mgr = _make_manager(tmp_path)
        mgr.set_patch_status("patch_a", "accepted")
        mgr.set_patch_status("patch_b", "accepted")
        mgr.update_patch_metadata("patch_a", downsample_warning_shown=True)
        assert mgr.get_patch_downsample_warning_shown("patch_a") is True
        assert mgr.get_patch_downsample_warning_shown("patch_b") is False

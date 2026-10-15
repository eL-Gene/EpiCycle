"""Tests for Session section's Last activity + Resume helpers (2026-05-18).

Covers:
  - state_manager.get_most_recent_activity()
  - state_manager.get_resume_patch() priority logic
  - controller._humanize_iso_timestamp()
"""
from __future__ import annotations

import types
import time
from datetime import datetime, timedelta, timezone

import pytest


def _tick():
    """Brief sleep so consecutive set_patch_status calls produce
    distinct ISO timestamps.  ISO format has microsecond precision
    so 1ms is more than enough."""
    time.sleep(0.001)

from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
)


# ---------------------------------------------------------------------------
# get_most_recent_activity
# ---------------------------------------------------------------------------


class TestGetMostRecentActivity:

    def test_empty_workspace_returns_none(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        assert sm.get_most_recent_activity() is None

    def test_single_patch_returns_timestamp_and_name(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["patch_a.tif"])
        sm.set_patch_status("patch_a.tif", "predicted")
        result = sm.get_most_recent_activity()
        assert result is not None
        ts, name = result
        assert name == "patch_a.tif"
        assert ts  # ISO timestamp string

    def test_multiple_patches_returns_most_recent(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["a.tif", "b.tif", "c.tif"])
        # Set in sequence — ISO timestamps are monotonic so last-write
        # is the most-recent.  _tick() ensures distinct microseconds.
        sm.set_patch_status("a.tif", "predicted")
        _tick()
        sm.set_patch_status("b.tif", "predicted")
        _tick()
        sm.set_patch_status("c.tif", "predicted")
        result = sm.get_most_recent_activity()
        assert result is not None
        ts, name = result
        assert name == "c.tif"

    def test_unregistered_patches_skipped(self, tmp_path):
        """Patches with no updated_at field are skipped."""
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        # register_patches creates records WITHOUT updated_at.
        sm.register_patches(["a.tif"])
        # No set_patch_status call — no updated_at.
        assert sm.get_most_recent_activity() is None


# ---------------------------------------------------------------------------
# get_resume_patch
# ---------------------------------------------------------------------------


class TestGetResumePatch:

    def test_empty_workspace_returns_none(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        assert sm.get_resume_patch() is None

    def test_unprocessed_only_returns_none(self, tmp_path):
        """Unprocessed patches are not resume targets."""
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["a.tif"])
        assert sm.get_resume_patch() is None

    def test_predicted_only_returns_none(self, tmp_path):
        """Predicted patches are not resume targets either —
        nothing in-progress to resume."""
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["a.tif"])
        sm.set_patch_status("a.tif", "predicted")
        assert sm.get_resume_patch() is None

    def test_finalized_only_returns_none(self, tmp_path):
        """Finalized patches are done — not resume targets."""
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["a.tif"])
        sm.set_patch_status("a.tif", "predicted")
        sm.set_patch_status("a.tif", "reconstructed")
        sm.set_patch_status("a.tif", "accepted")
        sm.set_patch_status("a.tif", "finalized")
        assert sm.get_resume_patch() is None

    def test_curating_takes_priority(self, tmp_path):
        """Even if accepted/reconstructed patches were touched later,
        curating wins because it represents active in-progress work."""
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["a.tif", "b.tif", "c.tif"])
        sm.set_patch_status("a.tif", "curating")
        sm.set_patch_status("b.tif", "predicted")
        sm.set_patch_status("b.tif", "reconstructed")
        sm.set_patch_status("b.tif", "accepted")  # touched AFTER a
        sm.set_patch_status("c.tif", "predicted")
        sm.set_patch_status("c.tif", "reconstructed")  # also after a
        result = sm.get_resume_patch()
        assert result == ("a.tif", "curating")

    def test_accepted_when_no_curating(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["a.tif", "b.tif"])
        sm.set_patch_status("a.tif", "predicted")
        sm.set_patch_status("a.tif", "reconstructed")
        sm.set_patch_status("a.tif", "accepted")
        sm.set_patch_status("b.tif", "predicted")
        sm.set_patch_status("b.tif", "reconstructed")
        result = sm.get_resume_patch()
        assert result == ("a.tif", "accepted")

    def test_reconstructed_when_no_curating_or_accepted(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["a.tif", "b.tif"])
        sm.set_patch_status("a.tif", "predicted")
        sm.set_patch_status("a.tif", "reconstructed")
        sm.set_patch_status("b.tif", "predicted")
        result = sm.get_resume_patch()
        assert result == ("a.tif", "reconstructed")

    def test_most_recent_within_same_priority(self, tmp_path):
        """Multiple patches at the same priority status — pick most recently touched."""
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.register_patches(["a.tif", "b.tif", "c.tif"])
        sm.set_patch_status("a.tif", "curating")
        _tick()
        sm.set_patch_status("b.tif", "curating")
        _tick()
        sm.set_patch_status("c.tif", "curating")  # most recent
        result = sm.get_resume_patch()
        assert result == ("c.tif", "curating")


# ---------------------------------------------------------------------------
# _humanize_iso_timestamp — controller helper
# ---------------------------------------------------------------------------


def _make_controller_stub():
    """Bare controller instance for testing _humanize_iso_timestamp."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    return ctrl


class TestHumanizeIsoTimestamp:

    def test_just_now(self):
        ctrl = _make_controller_stub()
        now = datetime.now(timezone.utc)
        assert ctrl._humanize_iso_timestamp(now.isoformat()) == "just now"

    def test_minutes_ago(self):
        ctrl = _make_controller_stub()
        ts = datetime.now(timezone.utc) - timedelta(minutes=5)
        result = ctrl._humanize_iso_timestamp(ts.isoformat())
        assert "minute" in result
        assert "5" in result

    def test_one_minute_singular(self):
        ctrl = _make_controller_stub()
        ts = datetime.now(timezone.utc) - timedelta(minutes=1, seconds=5)
        result = ctrl._humanize_iso_timestamp(ts.isoformat())
        # Should say "1 minute ago" — singular, no "s"
        assert "1 minute ago" == result

    def test_hours_ago(self):
        ctrl = _make_controller_stub()
        ts = datetime.now(timezone.utc) - timedelta(hours=3)
        result = ctrl._humanize_iso_timestamp(ts.isoformat())
        assert "hour" in result
        assert "3" in result

    def test_days_ago(self):
        ctrl = _make_controller_stub()
        ts = datetime.now(timezone.utc) - timedelta(days=2)
        result = ctrl._humanize_iso_timestamp(ts.isoformat())
        assert "day" in result
        assert "2" in result

    def test_old_shows_date(self):
        """Older than a week — show absolute date."""
        ctrl = _make_controller_stub()
        ts = datetime.now(timezone.utc) - timedelta(days=30)
        result = ctrl._humanize_iso_timestamp(ts.isoformat())
        # Format: 'YYYY-MM-DD HH:MM UTC'
        assert "UTC" in result
        assert "-" in result

    def test_malformed_falls_back_to_raw(self):
        ctrl = _make_controller_stub()
        bogus = "not an iso timestamp"
        assert ctrl._humanize_iso_timestamp(bogus) == bogus

"""Tests for the Mode=Both refinement training race fix (2026-05-21).

Pre-fix race:
  1. Lateral worker emits `finished` → two queued callbacks fire:
     `_on_refinement_finished` (controller) AND `thread.quit()`.
  2. `_on_refinement_finished` synchronously calls
     `_launch_next_refinement()` → creates AXIAL thread, assigns
     `self._refinement_thread = axial_thread`.
  3. Queued `lateral_thread.quit()` runs → lateral exits, emits
     `finished` → `_cleanup_refinement_thread` reads
     `self._refinement_thread` (which is now AXIAL) → calls
     `.deleteLater()` on it → axial QThread destroyed mid-training.

Fix:
  * Cleanup callback now closure-captures the specific thread:
    `thread.finished.connect(lambda t=thread:
        self._cleanup_specific_refinement_thread(t))`
  * `_cleanup_specific_refinement_thread(thread)` only nulls
    `self._refinement_thread` if it still matches `thread` (identity).
  * Axial launch is deferred via `QTimer.singleShot(0, ...)` so the
    lateral thread cleanup completes BEFORE the axial thread is
    assigned to the slot.

These tests exercise the cleanup identity logic with pure Python fakes
— no Qt event loop, no real threads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakeThread:
    """Stand-in for QThread.  Tracks deleteLater calls."""

    def __init__(self, name: str):
        self.name = name
        self.delete_called = False

    def deleteLater(self):
        self.delete_called = True

    def __repr__(self):
        return f"_FakeThread({self.name!r})"


def _make_controller():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = SimpleNamespace(
        _refinement_thread=None,
        _refinement_worker=None,
    )
    for name in (
        "_cleanup_refinement_thread",
        "_cleanup_specific_refinement_thread",
    ):
        method = getattr(FusionReconstructionController, name)
        setattr(ctrl, name, method.__get__(ctrl))
    return ctrl


# ---------------------------------------------------------------------------
# Identity-guarded cleanup
# ---------------------------------------------------------------------------


class TestSpecificThreadCleanup:

    def test_deletes_the_specific_thread(self):
        ctrl = _make_controller()
        t = _FakeThread("only")
        ctrl._refinement_thread = t
        ctrl._cleanup_specific_refinement_thread(t)
        assert t.delete_called
        # Slot cleared because it matched.
        assert ctrl._refinement_thread is None
        assert ctrl._refinement_worker is None

    def test_does_not_clobber_newer_thread_reference(self):
        """The lateral cleanup must NOT null the slot if the slot
        has already been overwritten by axial (the Mode=Both race)."""
        ctrl = _make_controller()
        lateral = _FakeThread("lateral")
        axial = _FakeThread("axial")
        axial_worker = SimpleNamespace(_name="axial_worker")
        # Simulate the race: axial is now in the slot when lateral's
        # cleanup callback finally fires.
        ctrl._refinement_thread = axial
        ctrl._refinement_worker = axial_worker
        ctrl._cleanup_specific_refinement_thread(lateral)
        # Lateral got deleted (it's safe — it's already exited).
        assert lateral.delete_called
        # CRITICAL: axial slot is preserved — the still-running thread
        # is not orphaned.
        assert ctrl._refinement_thread is axial
        assert ctrl._refinement_worker is axial_worker
        # Axial is NOT touched (would be a use-after-free in real Qt).
        assert not axial.delete_called

    def test_none_thread_is_safe(self):
        """Passing None should be a no-op, not a crash."""
        ctrl = _make_controller()
        ctrl._cleanup_specific_refinement_thread(None)
        # Slot stays None (matches by identity in the trivial sense
        # — actually `is` check skips it because slot is None).
        assert ctrl._refinement_thread is None

    def test_deletelater_runtime_error_is_swallowed(self):
        """If Qt has already deleted the C++ side, deleteLater() raises
        RuntimeError; cleanup must not crash."""

        class _DeadThread:
            def deleteLater(self):
                raise RuntimeError("wrapped C++ object deleted")

        ctrl = _make_controller()
        dead = _DeadThread()
        ctrl._refinement_thread = dead
        # Must not raise.
        ctrl._cleanup_specific_refinement_thread(dead)
        assert ctrl._refinement_thread is None


# ---------------------------------------------------------------------------
# Mode=Both race simulation (the actual bug repro)
# ---------------------------------------------------------------------------


class TestModeBothRaceSimulation:
    """Simulate the timing of the lateral→axial transition in Mode=Both,
    interleaving cleanup callbacks with new-thread assignment.  Pre-fix
    these sequences would orphan the axial thread; post-fix they
    preserve it."""

    def test_lateral_cleanup_fires_after_axial_assigned(self):
        ctrl = _make_controller()
        lateral = _FakeThread("lateral")
        axial = _FakeThread("axial")
        axial_worker = SimpleNamespace(_name="axial_worker")

        # Step 1: lateral thread starts and is assigned.
        ctrl._refinement_thread = lateral

        # Step 2: lateral worker finishes.  Controller's finished handler
        # would synchronously launch axial; with the fix this is now
        # deferred via QTimer.singleShot.  Simulating the deferred call:
        # the slot first becomes None (lateral cleanup runs), THEN axial
        # is assigned.
        ctrl._cleanup_specific_refinement_thread(lateral)
        assert ctrl._refinement_thread is None  # post-cleanup
        assert lateral.delete_called

        # Step 3: deferred `_launch_next_refinement` runs.
        ctrl._refinement_thread = axial
        ctrl._refinement_worker = axial_worker

        # Step 4: axial training is in progress; no stale callbacks fire.
        # Final state: axial slot intact, lateral cleaned up.
        assert ctrl._refinement_thread is axial
        assert not axial.delete_called

    def test_pre_fix_race_no_longer_orphans_axial(self):
        """Even if a stale lateral cleanup somehow fires AFTER axial is
        assigned (e.g., events delivered out of order), the identity
        check protects the axial slot.

        This is the test that would FAIL pre-fix — the lateral cleanup
        would null the slot and call deleteLater on axial.
        """
        ctrl = _make_controller()
        lateral = _FakeThread("lateral")
        axial = _FakeThread("axial")

        # Pretend the race happened: axial got assigned BEFORE lateral
        # cleanup fired.
        ctrl._refinement_thread = axial

        # Now the stale lateral-cleanup callback fires.
        ctrl._cleanup_specific_refinement_thread(lateral)

        # Lateral's stale ref got deleted (safe), axial slot preserved.
        assert lateral.delete_called
        assert ctrl._refinement_thread is axial
        assert not axial.delete_called


# ---------------------------------------------------------------------------
# Backwards-compat: legacy no-arg cleanup still works
# ---------------------------------------------------------------------------


class TestLegacyNoArgCleanup:
    """The bare `_cleanup_refinement_thread()` is retained for callers
    that connected with the old no-arg signature.  It MUST still
    function (not regress) but is no longer the recommended path."""

    def test_legacy_cleanup_clears_slot(self):
        ctrl = _make_controller()
        t = _FakeThread("only")
        ctrl._refinement_thread = t
        ctrl._cleanup_refinement_thread()
        assert t.delete_called
        assert ctrl._refinement_thread is None
        assert ctrl._refinement_worker is None

    def test_legacy_cleanup_handles_none_slot(self):
        ctrl = _make_controller()
        # No thread assigned.
        ctrl._cleanup_refinement_thread()
        assert ctrl._refinement_thread is None

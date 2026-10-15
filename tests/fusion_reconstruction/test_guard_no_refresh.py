"""Pins the 2026-06-06 perf fix: `_refresh_curation_ui()` was REMOVED from
`FusionReconstructionController._guard_unsaved_changes()` while the
`_curation_undo_stack.clear()` call was KEPT.

Rationale (from the in-source comment): the guard is called only from
`_switch_to_curation_patch`, which immediately hard-clears surfaces + loads
the new patch (rebuilding the UI for it).  Refreshing the OLD,
about-to-be-discarded patch's navigator + QC plot was pure waste — over
~1900 rows it took ~40s, freezing the post-extract patch switch.

These tests PROVE the removal empirically by:
  1. Binding + invoking the REAL method on a minimal harness, spying on
     `_refresh_curation_ui` (must NOT be called) and the undo-stack `.clear`
     (must still be called), and asserting the happy-path returns True.
  2. Source-introspecting the bound method to confirm the call site is gone
     but the undo-stack clear remains.
"""

import inspect
import types
from unittest.mock import MagicMock

import pytest


# Mirror test_curation_save_integrity.py's harness style: a minimal fake-state
# object that we bind REAL controller methods onto.
class _ControllerHarness:
    """Wraps the real `_guard_unsaved_changes` with minimal fake state."""

    def __init__(self):
        # Happy-path flags: nothing in flight, nothing dirty.  This routes the
        # guard straight past all three QMessageBox branches to the tail
        # (undo-stack clear → return True) — the exact region the perf fix
        # touched.
        self._save_in_flight = False
        self._tiers_dirty = False
        self._curation_dirty = False
        self._qc_view_patch_name = "patch_a.tif"

        # Undo stack: spy on .clear() — the fix KEEPS this call.
        self._curation_undo_stack = MagicMock()
        self._curation_undo_stack.clear = MagicMock()

        # Spy on the UI refresh — the fix REMOVES this call.  Counter via
        # MagicMock so an accidental re-introduction is caught.
        self._refresh_curation_ui = MagicMock()

        # Defensive UI stub (not exercised on the happy path, but harmless).
        self.ui = MagicMock()


def _bind_real_methods(harness, method_names):
    """Import + bind real controller methods onto the harness (same pattern
    as test_curation_save_integrity.py)."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    for method_name in method_names:
        real_method = getattr(FusionReconstructionController, method_name)
        harness.__dict__[method_name] = types.MethodType(real_method, harness)


@pytest.fixture
def harness():
    h = _ControllerHarness()
    _bind_real_methods(h, ["_guard_unsaved_changes"])
    return h


# ---------------------------------------------------------------------------
# Behavioral proof: run the real method, observe the side effects.
# ---------------------------------------------------------------------------

class TestGuardNoRefresh:
    def test_happy_path_returns_true(self, harness):
        """With nothing in-flight/dirty, the guard permits the switch."""
        assert harness._guard_unsaved_changes() is True

    def test_refresh_curation_ui_not_called(self, harness):
        """PERF FIX: the guard must NOT refresh the old patch's UI."""
        harness._guard_unsaved_changes()
        harness._refresh_curation_ui.assert_not_called()

    def test_undo_stack_cleared(self, harness):
        """The undo-stack clear (context invalidation) must still happen."""
        harness._guard_unsaved_changes()
        harness._curation_undo_stack.clear.assert_called_once()

    def test_all_three_invariants_together(self, harness):
        """Single call: returns True, no UI refresh, undo stack cleared."""
        result = harness._guard_unsaved_changes()
        assert result is True
        harness._refresh_curation_ui.assert_not_called()
        harness._curation_undo_stack.clear.assert_called_once()


# ---------------------------------------------------------------------------
# Source-introspection proof: pin the exact code change.
# ---------------------------------------------------------------------------

class TestGuardSource:
    def _source(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        return inspect.getsource(
            FusionReconstructionController._guard_unsaved_changes
        )

    def test_no_refresh_curation_ui_call(self):
        """The method body must no longer CALL _refresh_curation_ui.

        Note: the explanatory comment mentions the name, so we assert there
        is no `self._refresh_curation_ui(` call specifically rather than just
        the bare substring."""
        src = self._source()
        assert "self._refresh_curation_ui(" not in src, (
            "_guard_unsaved_changes must not call self._refresh_curation_ui()"
        )

    def test_undo_stack_clear_present(self):
        """The undo-stack clear must remain in the method body."""
        src = self._source()
        assert "_curation_undo_stack.clear" in src, (
            "_guard_unsaved_changes must still clear the curation undo stack"
        )

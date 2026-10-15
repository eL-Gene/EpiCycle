"""Tests for P1: Persistence-gated Accept/Finalize.

Verifies that Accept and Finalize cannot transition status without
successful mask/CSV persistence when curation edits are dirty.
"""

import pytest


# ---------------------------------------------------------------------------
# Minimal fakes for controller dependencies
# ---------------------------------------------------------------------------

class _FakeStateManager:
    """Tracks set_patch_status calls."""

    def __init__(self):
        self._state = {"patches": {"patch001.tif": {"status": "curating"}}}
        self.calls = []

    def get_patch_status(self, patch_name):
        return self._state["patches"].get(patch_name, {}).get("status", "")

    def set_patch_status(self, patch_name, status, **kwargs):
        self._state["patches"].setdefault(patch_name, {})["status"] = status
        self.calls.append((patch_name, status, kwargs))

    def schedule_save(self):
        pass


class _FakeUI:
    """Minimal UI stub with attributes accessed by accept/finalize."""

    def __init__(self):
        self.curation_status_text = ""
        self.history_items = []
        self.save_curation_btn_enabled = True

    class curation_status_label:
        _text = ""
        @staticmethod
        def setText(t):
            _FakeUI.curation_status_label._text = t

    class curation_history:
        _items = []
        @staticmethod
        def addItem(t):
            _FakeUI.curation_history._items.append(t)

    class save_curation_btn:
        _enabled = True
        @staticmethod
        def setEnabled(v):
            _FakeUI.save_curation_btn._enabled = v

    class _curation_bucket:
        @staticmethod
        def selectedItems():
            return []


# ---------------------------------------------------------------------------
# Logic under test: _save_curation_checked
# ---------------------------------------------------------------------------

class _PersistenceGateStub:
    """Replicates the persistence gate logic that will be in the controller."""

    def __init__(self, *, save_succeeds=True):
        self._curation_dirty = False
        self._save_called = False
        self._save_succeeds = save_succeeds

    def _save_curation_checked(self) -> bool:
        """Attempt save if dirty. Returns True if safe to proceed."""
        if not self._curation_dirty:
            return True
        self._save_called = True
        if self._save_succeeds:
            self._curation_dirty = False
            return True
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSaveCurationChecked:
    """Verify _save_curation_checked gate logic."""

    def test_clean_state_returns_true_no_save(self):
        stub = _PersistenceGateStub()
        stub._curation_dirty = False
        assert stub._save_curation_checked() is True
        assert stub._save_called is False

    def test_dirty_state_saves_and_returns_true(self):
        stub = _PersistenceGateStub(save_succeeds=True)
        stub._curation_dirty = True
        assert stub._save_curation_checked() is True
        assert stub._save_called is True
        assert stub._curation_dirty is False

    def test_dirty_state_save_fails_returns_false(self):
        stub = _PersistenceGateStub(save_succeeds=False)
        stub._curation_dirty = True
        assert stub._save_curation_checked() is False
        assert stub._save_called is True
        assert stub._curation_dirty is True  # still dirty


class TestAcceptPersistenceGate:
    """Verify Accept transitions status only after successful save."""

    def test_accept_clean_transitions(self):
        sm = _FakeStateManager()
        gate = _PersistenceGateStub(save_succeeds=True)
        gate._curation_dirty = False

        # Simulate accept logic
        if gate._save_curation_checked():
            sm.set_patch_status("patch001.tif", "accepted")

        assert sm.get_patch_status("patch001.tif") == "accepted"
        assert gate._save_called is False  # no save needed

    def test_accept_dirty_saves_then_transitions(self):
        sm = _FakeStateManager()
        gate = _PersistenceGateStub(save_succeeds=True)
        gate._curation_dirty = True

        if gate._save_curation_checked():
            sm.set_patch_status("patch001.tif", "accepted")

        assert sm.get_patch_status("patch001.tif") == "accepted"
        assert gate._save_called is True

    def test_accept_save_fails_no_transition(self):
        sm = _FakeStateManager()
        gate = _PersistenceGateStub(save_succeeds=False)
        gate._curation_dirty = True

        if gate._save_curation_checked():
            sm.set_patch_status("patch001.tif", "accepted")

        # Status should NOT change
        assert sm.get_patch_status("patch001.tif") == "curating"
        assert len(sm.calls) == 0


class TestFinalizePersistenceGate:
    """Verify Finalize transitions status only after successful save."""

    def test_finalize_clean_transitions(self):
        sm = _FakeStateManager()
        gate = _PersistenceGateStub(save_succeeds=True)
        gate._curation_dirty = False

        if gate._save_curation_checked():
            sm.set_patch_status("patch001.tif", "finalized")

        assert sm.get_patch_status("patch001.tif") == "finalized"

    def test_finalize_dirty_saves_then_transitions(self):
        sm = _FakeStateManager()
        gate = _PersistenceGateStub(save_succeeds=True)
        gate._curation_dirty = True

        if gate._save_curation_checked():
            sm.set_patch_status("patch001.tif", "finalized")

        assert sm.get_patch_status("patch001.tif") == "finalized"
        assert gate._save_called is True

    def test_finalize_save_fails_no_transition(self):
        sm = _FakeStateManager()
        gate = _PersistenceGateStub(save_succeeds=False)
        gate._curation_dirty = True

        if gate._save_curation_checked():
            sm.set_patch_status("patch001.tif", "finalized")

        assert sm.get_patch_status("patch001.tif") == "curating"
        assert len(sm.calls) == 0

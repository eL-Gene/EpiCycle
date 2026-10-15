"""Tests for the Stage 2 pretrained section in the registry dialog (2026-05-21).

Stage 2 adds Cellpose pretrained models to the registry dialog so
they appear alongside refined bundles in a single picker.  The
controller's `_registry_action_use` now branches on the new `_kind`
tag (refined vs pretrained) to set bucket paths appropriately:

* `kind=refined`  → copy `lateral_weights` + `axial_weights` from
  the bundle manifest into the path widgets; persist mode=import.
* `kind=pretrained` → set BOTH bucket paths to the pretrained NAME
  (e.g. "cyto3"); persist mode=pretrained; sync pretrained_combo.

Pure-Python fakes — no Qt event loop, no real dialogs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_epicycle.fusion_reconstruction import _paths


class _FakeCombo:
    def __init__(self, initial: str = ""):
        self._text = initial
        self.set_history = []
        self.block_history = []  # records (block_state) calls
    def currentText(self) -> str:
        return self._text
    def setCurrentText(self, t: str) -> None:
        self._text = t
        self.set_history.append(t)
    def blockSignals(self, state: bool) -> bool:
        self.block_history.append(state)
        return False  # mimic QObject.blockSignals' return value
    def addItem(self, s): pass
    def itemText(self, i): return ""
    def count(self): return 0


class _FakeLineEdit:
    def __init__(self):
        self._text = ""
    def text(self): return self._text
    def setText(self, t): self._text = t


def _make_controller(working_dir: Path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    logs: list = []
    ui = SimpleNamespace()
    ui.lateral_model_path = _FakeLineEdit()
    ui.axial_model_path = _FakeLineEdit()
    ui.model_source_combo = _FakeCombo(initial="Auto from HITL")
    ui.pretrained_combo = _FakeCombo()
    ui.refine_init_combo = _FakeCombo()
    ui.refinement_source_combo = _FakeCombo()
    ui.refinement_source_combo_axial = _FakeCombo()
    ui.refine_pack_status_label = _FakeLineEdit()
    ui.switch_to_section = lambda _i: None

    ctrl = SimpleNamespace(
        working_dir=str(working_dir),
        ui=ui,
        _append_log=logs.append,
        _logs=logs,
    )
    for name in (
        "_registry_action_use",
        "_refresh_model_status",
        "_persist_current_model_selection",
        "_read_bundle_training_channels",
        "_try_sync_prediction_channels_from_bundle",
    ):
        method = getattr(FusionReconstructionController, name, None)
        if method is not None:
            setattr(ctrl, name, method.__get__(ctrl))

    class _SM:
        def __init__(self):
            self._calls = []
        def set_model_selection(self, **kw):
            self._calls.append(kw)
    ctrl.state_manager = _SM()
    return ctrl


# ---------------------------------------------------------------------------
# Pretrained row behavior — `_kind=pretrained` branch in controller
# ---------------------------------------------------------------------------


class TestUseActionPretrained:

    def test_pretrained_sets_both_paths_to_name(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "pretrained",
            "name": "cyto3",
            "description": "General epithelial",
        })
        # Both bucket paths set to the bare pretrained name.
        assert ctrl.ui.lateral_model_path.text() == "cyto3"
        assert ctrl.ui.axial_model_path.text() == "cyto3"

    def test_pretrained_flips_source_to_pretrained(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "pretrained",
            "name": "nuclei",
            "description": "Nuclei",
        })
        # Source dropdown flips to "Pretrained" (NOT "Import") so the
        # pretrained widget becomes visible.
        assert ctrl.ui.model_source_combo.currentText() == "Pretrained"

    def test_pretrained_syncs_pretrained_combo(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "pretrained",
            "name": "cyto3",
        })
        # The pretrained_combo should show the chosen name.
        assert ctrl.ui.pretrained_combo.currentText() == "cyto3"

    def test_pretrained_persists_mode_pretrained(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "pretrained",
            "name": "cyto3",
        })
        # state_manager.set_model_selection should be called with
        # mode="pretrained" (not "import").
        calls = ctrl.state_manager._calls
        assert any(c.get("mode") == "pretrained" for c in calls), (
            f"Expected mode=pretrained call, got: {calls}"
        )


# ---------------------------------------------------------------------------
# Refined row behavior — back-compat with Stage 1 + new _kind tag
# ---------------------------------------------------------------------------


class TestUseActionRefined:

    def test_refined_with_kind_tag(self, tmp_path):
        """When the dialog sends _kind=refined, behavior matches the
        legacy untagged refined case."""
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "refined",
            "name": "my_model_v1",
            "lateral_weights": str(tmp_path / "my_model_v1" / "lateral"),
            "axial_weights": str(tmp_path / "my_model_v1" / "axial"),
        })
        assert "my_model_v1" in ctrl.ui.lateral_model_path.text()
        assert "my_model_v1" in ctrl.ui.axial_model_path.text()
        # Source flips to Import for refined.
        assert ctrl.ui.model_source_combo.currentText() == "Import"

    def test_legacy_call_without_kind_defaults_to_refined(self, tmp_path):
        """Callers that pre-date Stage 2 don't include _kind.  The
        handler defaults to 'refined' so back-compat is preserved."""
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "name": "legacy_caller",
            "lateral_weights": "/some/path/lateral",
            "axial_weights": "/some/path/axial",
        })
        # Refined behavior even without _kind.
        assert ctrl.ui.lateral_model_path.text() == "/some/path/lateral"
        assert ctrl.ui.model_source_combo.currentText() == "Import"

    def test_refined_persists_mode_import(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "refined",
            "name": "v1",
            "lateral_weights": "/p/lat",
            "axial_weights": "/p/ax",
        })
        calls = ctrl.state_manager._calls
        assert any(c.get("mode") == "import" for c in calls), (
            f"Expected mode=import call, got: {calls}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_pretrained_with_no_pretrained_combo_does_not_crash(self, tmp_path):
        """If the UI fake lacks pretrained_combo, still set the paths."""
        ctrl = _make_controller(tmp_path)
        delattr(ctrl.ui, "pretrained_combo")
        ctrl._registry_action_use({
            "_kind": "pretrained",
            "name": "cyto3",
        })
        # Paths still set.
        assert ctrl.ui.lateral_model_path.text() == "cyto3"
        assert ctrl.ui.model_source_combo.currentText() == "Pretrained"

    def test_pretrained_constant_has_modular_shape(self):
        """The PRETRAINED_MODELS constant is an IMMUTABLE tuple of
        (name, desc) tuples — adding a new entry is a one-line change.
        Verify shape + immutability so future additions don't
        accidentally break it (and so callers can't mutate the
        module-level catalog from outside).

        Audit fix (2026-05-21): switched from list to tuple per the
        pre-commit audit — a mutable module-level constant could be
        mutated by callers and persist across dialog instances.
        """
        from napari_epicycle.fusion_reconstruction.refinement_registry_dialog import (
            PRETRAINED_MODELS,
        )
        assert isinstance(PRETRAINED_MODELS, tuple)
        assert len(PRETRAINED_MODELS) >= 1
        for entry in PRETRAINED_MODELS:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            name, desc = entry
            assert isinstance(name, str) and name
            assert isinstance(desc, str)

    def test_default_pretrained_contains_cyto3(self):
        """cyto3 is the recommended default — must always be in the
        catalog unless explicitly removed."""
        from napari_epicycle.fusion_reconstruction.refinement_registry_dialog import (
            PRETRAINED_MODELS,
        )
        names = [n for n, _d in PRETRAINED_MODELS]
        assert "cyto3" in names


# ---------------------------------------------------------------------------
# Audit fix: double-persist prevention via blockSignals
# ---------------------------------------------------------------------------


class TestBlockSignalsPreventsDoublePresist:
    """Audit caught: setting pretrained_combo.setCurrentText fires
    `currentTextChanged` → triggers `_on_pretrained_changed` → calls
    `_persist_current_model_selection(mode="pretrained")` a SECOND
    time (after the explicit call in `_registry_action_use`).

    Fix wraps the combo updates in `blockSignals(True/False)` so the
    signal can't fire and re-trigger the persist.

    These tests verify blockSignals is called around the
    setCurrentText calls."""

    def test_model_source_combo_signals_blocked_around_set(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "pretrained",
            "name": "cyto3",
        })
        # blockSignals(True) called before setCurrentText, then
        # blockSignals(False) after.  Both should appear in history.
        history = ctrl.ui.model_source_combo.block_history
        assert True in history, (
            f"Expected blockSignals(True) call, got: {history}"
        )
        assert False in history, (
            f"Expected blockSignals(False) call, got: {history}"
        )

    def test_pretrained_combo_signals_blocked_around_set(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "pretrained",
            "name": "cyto3",
        })
        history = ctrl.ui.pretrained_combo.block_history
        assert True in history
        assert False in history

    def test_persist_called_exactly_once_for_pretrained(self, tmp_path):
        """The persist call should happen ONCE per registry-use action,
        not twice.  Audit-fix verification."""
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "pretrained",
            "name": "cyto3",
        })
        calls = ctrl.state_manager._calls
        pretrained_calls = [c for c in calls if c.get("mode") == "pretrained"]
        assert len(pretrained_calls) == 1, (
            f"Expected exactly 1 mode=pretrained persist call, got "
            f"{len(pretrained_calls)}: {pretrained_calls}"
        )

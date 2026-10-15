"""Tests for the Stage 1 visual-bug fix (2026-05-21).

Pre-fix: `_registry_action_use` and `_use_for_re_prediction` both
populated `lateral_model_path` / `axial_model_path` LineEdits but
DID NOT flip `model_source_combo` to "Import".  Those LineEdits
are children of `_import_widget` which is hidden when source is
"Auto from HITL".  User saw "nothing happened" even though paths
were set.

Fix: both action handlers now call
`self.ui.model_source_combo.setCurrentText("Import")` to make the
path widgets visible.

These tests use a fake combo to verify the flip happens.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_epicycle.fusion_reconstruction import _paths


class _FakeCombo:
    """Stand-in for QComboBox with currentText + setCurrentText only."""
    def __init__(self, initial: str = ""):
        self._text = initial
        self.set_history = []
    def currentText(self) -> str:
        return self._text
    def setCurrentText(self, t: str) -> None:
        self._text = t
        self.set_history.append(t)
    # The fake registry-set-init also calls addItem on the source combo.
    def addItem(self, s):
        pass
    def itemText(self, i):
        return ""
    def count(self):
        return 0


class _FakeLineEdit:
    def __init__(self):
        self._text = ""
    def text(self):
        return self._text
    def setText(self, t):
        self._text = t


def _make_controller(working_dir: Path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    logs: list = []
    ui = SimpleNamespace()
    ui.lateral_model_path = _FakeLineEdit()
    ui.axial_model_path = _FakeLineEdit()
    ui.model_source_combo = _FakeCombo(initial="Auto from HITL")
    ui.refine_init_combo = _FakeCombo()
    ui.refinement_source_combo = _FakeCombo()
    ui.refinement_source_combo_axial = _FakeCombo()
    ui.refine_pack_status_label = _FakeLineEdit()
    ui.refinement_status_label = _FakeLineEdit()
    ui.switch_to_section = lambda _i: None

    ctrl = SimpleNamespace(
        working_dir=str(working_dir),
        ui=ui,
        _append_log=logs.append,
        _logs=logs,
        # the look brick routes the panel's sentences through this
        _set_refinement_status=lambda text, kind="neutral": ui.refinement_status_label.setText(text),
    )
    for name in (
        "_scan_refinement_model_registry",
        "_read_bundle_manifest_normalized",
        "_convert_legacy_manifest_to_v2",
        "_read_bundle_training_channels",
        "_try_sync_prediction_channels_from_bundle",
        "_registry_action_use",
        "_refresh_model_status",
        "_persist_current_model_selection",
    ):
        method = getattr(FusionReconstructionController, name, None)
        if method is not None:
            setattr(ctrl, name, method.__get__(ctrl))

    # State manager stub for the persist call.
    class _SM:
        def __init__(self):
            self._calls = []
        def set_model_selection(self, **kw):
            self._calls.append(kw)
    ctrl.state_manager = _SM()
    return ctrl


def _write_bundle(working_dir: Path, name: str, manifest: dict) -> Path:
    bundle_dir = _paths.refined_model_dir(working_dir, name)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return bundle_dir


# ---------------------------------------------------------------------------
# The actual fix
# ---------------------------------------------------------------------------


class TestSourceComboFlippedToImport:

    def test_use_action_flips_source_to_import(self, tmp_path):
        _write_bundle(tmp_path, "v1", {
            "schema_version": "2.0",
            "name": "v1",
            "lateral": {"weights_path": "lateral"},
            "axial": {"weights_path": "axial"},
        })
        ctrl = _make_controller(tmp_path)
        # Initial state: source is "Auto from HITL" (import widget hidden).
        assert ctrl.ui.model_source_combo.currentText() == "Auto from HITL"

        rows = ctrl._scan_refinement_model_registry()
        ctrl._registry_action_use(rows[0])

        # Post-fix: source dropdown flipped to "Import" so the
        # newly-populated path LineEdits are now visible.
        assert ctrl.ui.model_source_combo.currentText() == "Import"
        # The flip actually happened (set_history records each setCurrentText).
        assert "Import" in ctrl.ui.model_source_combo.set_history

    def test_use_action_path_widgets_populated(self, tmp_path):
        """Sanity: the original behavior (path setText) still works
        alongside the new flip."""
        _write_bundle(tmp_path, "v1", {
            "schema_version": "2.0",
            "name": "v1",
            "lateral": {"weights_path": "lateral"},
            "axial": {"weights_path": "axial"},
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        ctrl._registry_action_use(rows[0])
        assert "v1" in ctrl.ui.lateral_model_path.text()
        assert "v1" in ctrl.ui.axial_model_path.text()

    def test_lateral_only_bundle_still_flips_source(self, tmp_path):
        """Even for partial bundles (lateral only, no axial),
        the source dropdown still flips so the populated lateral
        path is visible."""
        _write_bundle(tmp_path, "lat_only", {
            "schema_version": "2.0",
            "name": "lat_only",
            "lateral": {"weights_path": "lateral"},
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        ctrl._registry_action_use(rows[0])
        assert ctrl.ui.model_source_combo.currentText() == "Import"
        assert ctrl.ui.axial_model_path.text() == ""

    def test_missing_combo_does_not_crash(self, tmp_path):
        """If the UI fake doesn't have model_source_combo (test fakes
        that haven't been updated), the action handler must not
        crash."""
        _write_bundle(tmp_path, "v1", {
            "schema_version": "2.0",
            "name": "v1",
            "lateral": {"weights_path": "lateral"},
        })
        ctrl = _make_controller(tmp_path)
        delattr(ctrl.ui, "model_source_combo")
        rows = ctrl._scan_refinement_model_registry()
        # Must not raise.
        ctrl._registry_action_use(rows[0])
        # Paths still populated.
        assert "v1" in ctrl.ui.lateral_model_path.text()

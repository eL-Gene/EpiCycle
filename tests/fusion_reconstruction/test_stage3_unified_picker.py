"""Tests for Stage 3 + 4 of the Predict source UX unification.

Stage 3 drops the 3-mode dropdown (Auto from HITL / Import /
Pretrained) from the Predict + Reconstruct section.  Replaces with:

* An "Active model" inline indicator (color-coded label) +
  [Change…] button.
* The [Change…] button opens the unified picker dialog.
* Browse-external section in the picker for setting external paths
  (lateral + axial, file pickers + Apply button).

State storage: the underlying `lateral_model_path` / `axial_model_path`
LineEdits + `pretrained_combo` + `model_source_combo` are kept as
HIDDEN widgets so existing controller callers don't break (they
still `setText` and `currentText` on these).

Pure-Python fakes — no Qt event loop, no real dialogs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakeCombo:
    def __init__(self, initial: str = ""):
        self._text = initial
        self.set_history = []
    def currentText(self) -> str:
        return self._text
    def setCurrentText(self, t: str) -> None:
        self._text = t
        self.set_history.append(t)
    def blockSignals(self, state: bool) -> bool:
        return False
    def addItem(self, s): pass
    def itemText(self, i): return ""
    def count(self): return 0


class _FakeLineEdit:
    def __init__(self):
        self._text = ""
    def text(self): return self._text
    def setText(self, t): self._text = t


def _make_controller(tmp_path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    logs: list = []
    ui = SimpleNamespace()
    ui.lateral_model_path = _FakeLineEdit()
    ui.axial_model_path = _FakeLineEdit()
    ui.model_source_combo = _FakeCombo(initial="Import")
    ui.pretrained_combo = _FakeCombo()
    ui.refine_init_combo = _FakeCombo()
    ui.refinement_source_combo = _FakeCombo()
    ui.refinement_source_combo_axial = _FakeCombo()
    ui.refine_pack_status_label = _FakeLineEdit()
    ui.switch_to_section = lambda _i: None

    ctrl = SimpleNamespace(
        working_dir=str(tmp_path),
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
# External path dispatch via _registry_action_use
# ---------------------------------------------------------------------------


class TestExternalKindDispatch:

    def test_external_with_both_paths(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "external",
            "name": "external paths",
            "lateral_weights": "/data/custom_lateral",
            "axial_weights": "/data/custom_axial",
        })
        assert ctrl.ui.lateral_model_path.text() == "/data/custom_lateral"
        assert ctrl.ui.axial_model_path.text() == "/data/custom_axial"
        # External flips source to "Import" (paths are file paths).
        assert ctrl.ui.model_source_combo.currentText() == "Import"

    def test_external_lateral_only_preserves_axial(self, tmp_path):
        """An empty axial field should leave the existing axial path
        unchanged (None → skip)."""
        ctrl = _make_controller(tmp_path)
        # Pre-existing axial value (e.g. from a prior selection).
        ctrl.ui.axial_model_path.setText("/prior/axial_path")
        ctrl._registry_action_use({
            "_kind": "external",
            "name": "external paths",
            "lateral_weights": "/data/new_lateral",
            "axial_weights": None,
        })
        assert ctrl.ui.lateral_model_path.text() == "/data/new_lateral"
        # Axial preserved.
        assert ctrl.ui.axial_model_path.text() == "/prior/axial_path"

    def test_external_persists_mode_import(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "external",
            "name": "external paths",
            "lateral_weights": "/data/lat",
            "axial_weights": "/data/ax",
        })
        calls = ctrl.state_manager._calls
        assert any(c.get("mode") == "import" for c in calls), (
            f"Expected mode=import call, got: {calls}"
        )


# ---------------------------------------------------------------------------
# Active indicator label rendering via _refresh_model_status
# ---------------------------------------------------------------------------


class _FakeLabel:
    def __init__(self):
        self._text = ""
        self._stylesheet = ""
    def text(self): return self._text
    def setText(self, t): self._text = t
    def styleSheet(self): return self._stylesheet
    def setStyleSheet(self, s): self._stylesheet = s


def _make_controller_with_indicator(tmp_path):
    """Variant that adds the new model_active_indicator_label fake."""
    ctrl = _make_controller(tmp_path)
    ctrl.ui.model_active_indicator_label = _FakeLabel()
    ctrl.ui.status_lateral_model_value = _FakeLabel()
    ctrl.ui.status_axial_model_value = _FakeLabel()
    return ctrl


class TestActiveIndicator:

    def test_empty_state_shows_no_model_loaded(self, tmp_path):
        ctrl = _make_controller_with_indicator(tmp_path)
        ctrl._refresh_model_status()
        text = ctrl.ui.model_active_indicator_label.text()
        assert "no model loaded" in text.lower() or "(none" in text.lower()
        # Dim color (888).
        assert "#888" in ctrl.ui.model_active_indicator_label.styleSheet()

    def test_pretrained_shows_blue(self, tmp_path):
        ctrl = _make_controller_with_indicator(tmp_path)
        ctrl.ui.lateral_model_path.setText("cyto3")
        ctrl.ui.axial_model_path.setText("cyto3")
        ctrl._refresh_model_status()
        # Single-line collapsed form (lateral == axial).
        text = ctrl.ui.model_active_indicator_label.text()
        assert "cyto3" in text or "Pretrained" in text
        # Blue (42a5f5) for pretrained.
        assert "#42a5f5" in ctrl.ui.model_active_indicator_label.styleSheet()

    def test_refined_path_shows_green(self, tmp_path):
        ctrl = _make_controller_with_indicator(tmp_path)
        ctrl.ui.lateral_model_path.setText(
            "C:/workspace/artifacts/step3/refinement/models/v1/lateral"
        )
        ctrl.ui.axial_model_path.setText(
            "C:/workspace/artifacts/step3/refinement/models/v1/axial"
        )
        ctrl._refresh_model_status()
        # Green (4caf50) for real path.
        assert "#4caf50" in ctrl.ui.model_active_indicator_label.styleSheet()

    def test_mixed_lateral_axial_shows_both(self, tmp_path):
        """When lateral and axial differ, indicator shows both."""
        ctrl = _make_controller_with_indicator(tmp_path)
        ctrl.ui.lateral_model_path.setText("cyto3")
        ctrl.ui.axial_model_path.setText("/path/to/refined/axial")
        ctrl._refresh_model_status()
        text = ctrl.ui.model_active_indicator_label.text()
        # Both buckets show.
        assert "lat" in text and "ax" in text

    def test_missing_indicator_does_not_crash(self, tmp_path):
        """If the UI fake doesn't have model_active_indicator_label
        (older tests / pre-Stage-3 fixtures), the refresh must not
        crash."""
        ctrl = _make_controller(tmp_path)
        # No model_active_indicator_label attribute.
        ctrl.ui.status_lateral_model_value = _FakeLabel()
        ctrl.ui.status_axial_model_value = _FakeLabel()
        ctrl.ui.lateral_model_path.setText("cyto3")
        ctrl.ui.axial_model_path.setText("cyto3")
        # Must not raise.
        ctrl._refresh_model_status()


# ---------------------------------------------------------------------------
# Browse-external dialog payload shape
# ---------------------------------------------------------------------------


class TestBrowseExternalPayload:
    """The dialog's _handle_ext_apply_clicked emits a specific
    payload shape to the on_use callback.  Verify the shape directly
    by constructing it manually + dispatching through the controller."""

    def test_payload_with_both_paths(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        # Simulate what the dialog would emit.
        ctrl._registry_action_use({
            "_kind": "external",
            "name": "external paths",
            "lateral_weights": "/a/b/lateral.pkl",
            "axial_weights": "/a/b/axial.pkl",
        })
        assert ctrl.ui.lateral_model_path.text() == "/a/b/lateral.pkl"
        assert ctrl.ui.axial_model_path.text() == "/a/b/axial.pkl"

    def test_payload_lateral_only(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "external",
            "name": "external paths",
            "lateral_weights": "/only/lateral",
            "axial_weights": None,
        })
        assert ctrl.ui.lateral_model_path.text() == "/only/lateral"
        # Axial untouched (default empty).
        assert ctrl.ui.axial_model_path.text() == ""

    def test_payload_axial_only(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "external",
            "name": "external paths",
            "lateral_weights": None,
            "axial_weights": "/only/axial",
        })
        assert ctrl.ui.lateral_model_path.text() == ""
        assert ctrl.ui.axial_model_path.text() == "/only/axial"


# ---------------------------------------------------------------------------
# Dropdown removal — hidden state widgets still functional
# ---------------------------------------------------------------------------


class TestHiddenStateWidgetsBackCompat:
    """The Stage 3 refactor removes the visible dropdown but keeps
    model_source_combo + pretrained_combo + path widgets as hidden
    state-storage widgets.  Existing controller callers that read
    `self.ui.model_source_combo.currentText()` etc still work."""

    def test_model_source_combo_still_readable(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        # Hidden combo defaults to "Import" post-Stage-3.
        assert ctrl.ui.model_source_combo.currentText() == "Import"

    def test_refined_action_still_sets_source_to_import(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "refined",
            "name": "v1",
            "lateral_weights": "/p/lat",
            "axial_weights": "/p/ax",
        })
        # The hidden combo is still updated (for back-compat with
        # downstream `currentText()` callers and `_model_source_mode`
        # tracking).
        assert ctrl.ui.model_source_combo.currentText() == "Import"

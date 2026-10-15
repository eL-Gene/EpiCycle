"""Tests for the 2026-05-21 channel-sync fix.

Bug being fixed: training and prediction channel widgets are
independent — user could train with [2, 1] (green/red) and predict
with [1, 0] (red/none).  Model would see totally different input
than it learned on → "horrible reconstruction quality" symptom.

Fix: when the user picks a refined bundle via the registry dialog
or the Use-for-Re-Prediction button, the bundle's training channels
(stored in `{bucket}_training_config.json` by Cellpose's
`train_seg` post-train save) are AUTO-APPLIED to the Predict
combos.

Pure-Python fakes — no Qt event loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_epicycle.fusion_reconstruction import _paths


class _FakeCombo:
    def __init__(self, initial_idx: int = 0):
        self._idx = initial_idx
        self._text = ""
        self.idx_history = []
    def currentIndex(self) -> int:
        return self._idx
    def setCurrentIndex(self, i: int) -> None:
        self._idx = int(i)
        self.idx_history.append(int(i))
    def currentText(self) -> str:
        return self._text
    def setCurrentText(self, t: str) -> None:
        self._text = t
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
    ui.chan1_combo = _FakeCombo(initial_idx=2)  # default green
    ui.chan2_combo = _FakeCombo(initial_idx=1)  # default red
    ui.model_source_combo = _FakeCombo()
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
        "_read_bundle_training_channels",
        "_try_sync_prediction_channels_from_bundle",
        "_registry_action_use",
        "_refresh_model_status",
        "_persist_current_model_selection",
        "_scan_refinement_model_registry",
        "_read_bundle_manifest_normalized",
        "_convert_legacy_manifest_to_v2",
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


def _write_training_config(
    bundle_dir: Path, bucket: str, channels: list,
) -> Path:
    """Write a minimal training_config.json with the given channels."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_name": bucket,
        "hyperparameters": {
            "channels": channels,
            "n_epochs": 30,
            "learning_rate": 0.01,
            "batch_size": 16,
        },
    }
    path = bundle_dir / f"{bucket}_training_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# _read_bundle_training_channels
# ---------------------------------------------------------------------------


class TestReadBundleTrainingChannels:

    def test_reads_lateral_config(self, tmp_path):
        bundle = tmp_path / "v1"
        _write_training_config(bundle, "lateral", [2, 1])
        ctrl = _make_controller(tmp_path)
        ch = ctrl._read_bundle_training_channels(bundle)
        assert ch == [2, 1]

    def test_falls_back_to_axial_if_lateral_missing(self, tmp_path):
        bundle = tmp_path / "ax_only"
        _write_training_config(bundle, "axial", [3, 0])
        ctrl = _make_controller(tmp_path)
        ch = ctrl._read_bundle_training_channels(bundle)
        assert ch == [3, 0]

    def test_returns_none_when_no_config(self, tmp_path):
        bundle = tmp_path / "empty"
        bundle.mkdir()
        ctrl = _make_controller(tmp_path)
        assert ctrl._read_bundle_training_channels(bundle) is None

    def test_corrupt_json_returns_none(self, tmp_path):
        bundle = tmp_path / "corrupt"
        bundle.mkdir()
        (bundle / "lateral_training_config.json").write_text("not json")
        ctrl = _make_controller(tmp_path)
        assert ctrl._read_bundle_training_channels(bundle) is None

    def test_missing_hyperparameters_returns_none(self, tmp_path):
        bundle = tmp_path / "no_hp"
        bundle.mkdir()
        (bundle / "lateral_training_config.json").write_text(
            json.dumps({"model_name": "lateral"})
        )
        ctrl = _make_controller(tmp_path)
        assert ctrl._read_bundle_training_channels(bundle) is None

    def test_malformed_channels_returns_none(self, tmp_path):
        """channels must be a 2-element list of ints; anything else
        is rejected (don't blindly trust corrupt data)."""
        bundle = tmp_path / "bad_ch"
        bundle.mkdir()
        for bad_value in (None, [], [1], [1, 2, 3], "two_one", [1, "x"]):
            (bundle / "lateral_training_config.json").write_text(
                json.dumps({
                    "hyperparameters": {"channels": bad_value},
                })
            )
            ctrl = _make_controller(tmp_path)
            assert ctrl._read_bundle_training_channels(bundle) is None, (
                f"Expected None for malformed channels={bad_value!r}"
            )


# ---------------------------------------------------------------------------
# _try_sync_prediction_channels_from_bundle
# ---------------------------------------------------------------------------


class TestSyncPredictionChannelsFromBundle:

    def test_sync_applies_channels_to_combos(self, tmp_path):
        bundle = tmp_path / "v1"
        _write_training_config(bundle, "lateral", [2, 1])
        ctrl = _make_controller(tmp_path)
        # Pre-sync: chan1=2 (default green), chan2=1 (default red).
        # Set to something different so we can see the sync happen.
        ctrl.ui.chan1_combo.setCurrentIndex(0)
        ctrl.ui.chan2_combo.setCurrentIndex(0)
        ctrl._try_sync_prediction_channels_from_bundle(bundle)
        # Post-sync: chan1=2, chan2=1.
        assert ctrl.ui.chan1_combo.currentIndex() == 2
        assert ctrl.ui.chan2_combo.currentIndex() == 1

    def test_sync_logs_with_channel_sync_prefix(self, tmp_path):
        bundle = tmp_path / "v1"
        _write_training_config(bundle, "lateral", [3, 0])
        ctrl = _make_controller(tmp_path)
        ctrl._try_sync_prediction_channels_from_bundle(bundle)
        assert any("[CHANNEL-SYNC]" in log for log in ctrl._logs)

    def test_no_config_logs_unchanged(self, tmp_path):
        bundle = tmp_path / "no_config"
        bundle.mkdir()
        ctrl = _make_controller(tmp_path)
        # Pre-sync values.
        pre_c1 = ctrl.ui.chan1_combo.currentIndex()
        pre_c2 = ctrl.ui.chan2_combo.currentIndex()
        ctrl._try_sync_prediction_channels_from_bundle(bundle)
        # Channels unchanged.
        assert ctrl.ui.chan1_combo.currentIndex() == pre_c1
        assert ctrl.ui.chan2_combo.currentIndex() == pre_c2
        # Log mentions the no-config case.
        assert any(
            "no training_config.json" in log for log in ctrl._logs
        )

    def test_missing_chan_combos_does_not_crash(self, tmp_path):
        bundle = tmp_path / "v1"
        _write_training_config(bundle, "lateral", [2, 1])
        ctrl = _make_controller(tmp_path)
        delattr(ctrl.ui, "chan1_combo")
        delattr(ctrl.ui, "chan2_combo")
        # Must not raise.
        ctrl._try_sync_prediction_channels_from_bundle(bundle)


# ---------------------------------------------------------------------------
# Integration: _registry_action_use auto-syncs for refined kind
# ---------------------------------------------------------------------------


class TestRegistryActionUseAutoSync:

    def test_refined_use_auto_syncs_channels(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "v1")
        _write_training_config(bundle, "lateral", [2, 1])
        # Also write a manifest so it's a complete bundle.
        manifest = {
            "schema_version": "2.0",
            "name": "v1",
            "lateral": {"weights_path": "lateral"},
            "axial": {"weights_path": "axial"},
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        ctrl = _make_controller(tmp_path)
        # Pre-state: chan1=0, chan2=0 (different from training).
        ctrl.ui.chan1_combo.setCurrentIndex(0)
        ctrl.ui.chan2_combo.setCurrentIndex(0)

        rows = ctrl._scan_refinement_model_registry()
        assert len(rows) == 1
        ctrl._registry_action_use({**rows[0], "_kind": "refined"})

        # Post-Use: chan1+chan2 synced to [2, 1].
        assert ctrl.ui.chan1_combo.currentIndex() == 2
        assert ctrl.ui.chan2_combo.currentIndex() == 1

    def test_pretrained_use_does_not_sync_channels(self, tmp_path):
        """Pretrained models have no training_config.json.  Channels
        should be untouched."""
        ctrl = _make_controller(tmp_path)
        # Pre-state: chan1=0, chan2=0.
        ctrl.ui.chan1_combo.setCurrentIndex(0)
        ctrl.ui.chan2_combo.setCurrentIndex(0)

        ctrl._registry_action_use({
            "_kind": "pretrained",
            "name": "cyto3",
        })

        # Channels unchanged — pretrained branch doesn't call sync.
        assert ctrl.ui.chan1_combo.currentIndex() == 0
        assert ctrl.ui.chan2_combo.currentIndex() == 0

    def test_external_use_does_not_sync_channels(self, tmp_path):
        """External paths don't have a known bundle structure with
        training_config.json — skip sync."""
        ctrl = _make_controller(tmp_path)
        ctrl.ui.chan1_combo.setCurrentIndex(0)
        ctrl.ui.chan2_combo.setCurrentIndex(0)

        ctrl._registry_action_use({
            "_kind": "external",
            "name": "external paths",
            "lateral_weights": "/external/lateral",
            "axial_weights": "/external/axial",
        })

        assert ctrl.ui.chan1_combo.currentIndex() == 0
        assert ctrl.ui.chan2_combo.currentIndex() == 0


# ---------------------------------------------------------------------------
# Scanner row dict now carries trained_channels
# ---------------------------------------------------------------------------


class TestScannerRowsHaveTrainedChannels:

    def test_row_includes_trained_channels(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "v1")
        _write_training_config(bundle, "lateral", [2, 1])
        (bundle / "manifest.json").write_text(json.dumps({
            "schema_version": "2.0",
            "lateral": {"weights_path": "lateral"},
        }))
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        assert len(rows) == 1
        assert rows[0]["trained_channels"] == [2, 1]

    def test_row_trained_channels_none_when_no_config(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "legacy_v1")
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text(json.dumps({
            "schema_version": "2.0",
            "lateral": {"weights_path": "lateral"},
        }))
        # No training_config.json — legacy bundle.
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        assert rows[0]["trained_channels"] is None

"""Tests for the Tier 1c refined-models registry scanner + action handlers.

The dialog itself (Qt-dependent) is exercised manually; this module
covers the pure-Python parts that the dialog consumes:

* ``_scan_refinement_model_registry`` — reads bundle manifests into row dicts
* ``_registry_action_use`` — Use-for-Re-Prediction routing
* ``_registry_action_open_folder`` — folder open (skip OS launch in tests)
* ``_registry_action_set_init`` — Set-as-init source routing
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from napari_epicycle.fusion_reconstruction import _paths


def _make_controller(working_dir: Path):
    """Minimal fake controller with the methods bound from the real class."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    logs: list = []

    class _LineEdit:
        def __init__(self):
            self._text = ""
        def text(self):
            return self._text
        def setText(self, t):
            self._text = t

    class _Combo:
        def __init__(self):
            self._items = []
            self._current = ""
        def addItem(self, s):
            self._items.append(s)
        def itemText(self, i):
            return self._items[i] if 0 <= i < len(self._items) else ""
        def count(self):
            return len(self._items)
        def setCurrentText(self, t):
            self._current = t
        def currentText(self):
            return self._current

    ui = SimpleNamespace()
    ui.lateral_model_path = _LineEdit()
    ui.axial_model_path = _LineEdit()
    ui.refine_init_combo = _Combo()
    ui.refinement_source_combo = _Combo()
    ui.refinement_source_combo_axial = _Combo()

    ctrl = SimpleNamespace(
        working_dir=str(working_dir),
        ui=ui,
        _append_log=logs.append,
        _logs=logs,
    )
    for name in (
        "_scan_refinement_model_registry",
        "_read_bundle_manifest_normalized",
        "_convert_legacy_manifest_to_v2",
        "_read_bundle_training_channels",
        "_try_sync_prediction_channels_from_bundle",
        "_registry_action_use",
        "_registry_action_open_folder",
        "_registry_action_set_init",
    ):
        method = getattr(FusionReconstructionController, name)
        setattr(ctrl, name, method.__get__(ctrl))
    return ctrl


def _write_bundle(
    working_dir: Path, name: str, manifest: Dict[str, Any],
) -> Path:
    """Create a refined-model bundle dir with the given manifest."""
    bundle_dir = _paths.refined_model_dir(working_dir, name)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return bundle_dir


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class TestScannerEmptyAndMissingPaths:

    def test_no_working_dir_returns_empty(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl.working_dir = ""
        assert ctrl._scan_refinement_model_registry() == []

    def test_models_root_missing_returns_empty(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        assert ctrl._scan_refinement_model_registry() == []

    def test_models_root_empty_returns_empty(self, tmp_path):
        _paths.refined_models_root(tmp_path).mkdir(parents=True)
        ctrl = _make_controller(tmp_path)
        assert ctrl._scan_refinement_model_registry() == []

    def test_bundle_without_manifest_skipped(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "no_manifest")
        bundle.mkdir(parents=True)
        ctrl = _make_controller(tmp_path)
        assert ctrl._scan_refinement_model_registry() == []

    def test_corrupt_manifest_logs_and_skips(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "corrupt")
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text("NOT JSON")
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        assert rows == []
        assert any("manifest.json corrupt" in log for log in ctrl._logs)


class TestScannerRowShape:

    def test_lateral_only_bundle(self, tmp_path):
        _write_bundle(tmp_path, "lat_v1", {
            "schema_version": "2.0",
            "name": "lat_v1",
            "created_at": "2026-05-20T10:00:00Z",
            "lateral": {
                "backend": "cellpose",
                "weights_path": "lateral",
                "n_train_pairs": 100,
                "init": {"mode": "cellpose", "path": "cyto3"},
            },
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "lat_v1"
        assert row["mode"] == "Lateral"
        assert row["lateral_pairs"] == 100
        assert row["axial_pairs"] is None
        assert row["lateral_weights"].endswith("lat_v1") or row["lateral_weights"].endswith("lateral")
        # The weights_path is relative; the resolved path includes the bundle.
        assert "lat_v1" in row["lateral_weights"]
        assert row["axial_weights"] is None

    def test_both_buckets_bundle(self, tmp_path):
        _write_bundle(tmp_path, "both_v1", {
            "schema_version": "2.0",
            "name": "both_v1",
            "created_at": "2026-05-20T11:00:00Z",
            "lateral": {
                "weights_path": "lateral",
                "n_train_pairs": 50,
                "init": {"mode": "scratch"},
            },
            "axial": {
                "weights_path": "axial",
                "n_train_pairs": 80,
                "init": {"mode": "custom"},
            },
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        assert len(rows) == 1
        row = rows[0]
        assert row["mode"] == "Both"
        assert row["lateral_pairs"] == 50
        assert row["axial_pairs"] == 80
        assert "lat=scratch" in row["init_summary"]
        assert "ax=custom" in row["init_summary"]

    def test_axial_only_bundle(self, tmp_path):
        _write_bundle(tmp_path, "ax_v1", {
            "schema_version": "2.0",
            "name": "ax_v1",
            "axial": {"weights_path": "axial"},
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        assert rows[0]["mode"] == "Axial"
        assert rows[0]["lateral_pairs"] is None
        assert rows[0]["axial_weights"] is not None


class TestScannerOrdering:

    def test_sorted_newest_first(self, tmp_path):
        _write_bundle(tmp_path, "old", {
            "schema_version": "2.0", "name": "old",
            "created_at": "2026-04-01T00:00:00Z",
            "lateral": {"weights_path": "lateral"},
        })
        _write_bundle(tmp_path, "new", {
            "schema_version": "2.0", "name": "new",
            "created_at": "2026-05-20T00:00:00Z",
            "lateral": {"weights_path": "lateral"},
        })
        _write_bundle(tmp_path, "middle", {
            "schema_version": "2.0", "name": "middle",
            "created_at": "2026-05-01T00:00:00Z",
            "lateral": {"weights_path": "lateral"},
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        names = [r["name"] for r in rows]
        assert names == ["new", "middle", "old"]


class TestActionUse:

    def test_sets_lateral_and_axial_paths(self, tmp_path):
        _write_bundle(tmp_path, "v1", {
            "schema_version": "2.0", "name": "v1",
            "lateral": {"weights_path": "lateral"},
            "axial": {"weights_path": "axial"},
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        ctrl._registry_action_use(rows[0])
        assert "v1" in ctrl.ui.lateral_model_path.text()
        assert ctrl.ui.lateral_model_path.text().endswith("lateral")
        assert ctrl.ui.axial_model_path.text().endswith("axial")

    def test_only_populates_present_buckets(self, tmp_path):
        _write_bundle(tmp_path, "lat_only", {
            "schema_version": "2.0", "name": "lat_only",
            "lateral": {"weights_path": "lateral"},
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        ctrl._registry_action_use(rows[0])
        assert "lat_only" in ctrl.ui.lateral_model_path.text()
        # Axial path widget should be untouched (empty default).
        assert ctrl.ui.axial_model_path.text() == ""


class TestActionSetInit:

    def test_flips_init_combo_and_populates_sources(self, tmp_path):
        _write_bundle(tmp_path, "init_v1", {
            "schema_version": "2.0", "name": "init_v1",
            "lateral": {"weights_path": "lateral"},
            "axial": {"weights_path": "axial"},
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        ctrl._registry_action_set_init(rows[0])
        assert ctrl.ui.refine_init_combo.currentText() == "From prior refinement"
        assert "init_v1" in ctrl.ui.refinement_source_combo.currentText()
        assert "init_v1" in ctrl.ui.refinement_source_combo_axial.currentText()

    def test_lateral_only_populates_lateral_combo_only(self, tmp_path):
        _write_bundle(tmp_path, "lat_only_init", {
            "schema_version": "2.0", "name": "lat_only_init",
            "lateral": {"weights_path": "lateral"},
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        ctrl._registry_action_set_init(rows[0])
        assert "lat_only_init" in ctrl.ui.refinement_source_combo.currentText()
        # Axial combo not set (axial bucket isn't in bundle).
        assert ctrl.ui.refinement_source_combo_axial.currentText() == ""


class TestActionOpenFolder:

    def test_no_bundle_dir_logs_warning(self, tmp_path, monkeypatch):
        ctrl = _make_controller(tmp_path)
        ctrl._registry_action_open_folder({"name": "phantom"})
        assert any("no bundle_dir" in log for log in ctrl._logs)

    def test_calls_os_open_command(self, tmp_path, monkeypatch):
        """Verify the open command is dispatched (don't actually open)."""
        _write_bundle(tmp_path, "openme", {
            "schema_version": "2.0", "name": "openme",
            "lateral": {"weights_path": "lateral"},
        })
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        called = []
        # Monkeypatch the platform launcher to avoid actually opening.
        import os
        import subprocess
        monkeypatch.setattr(
            os, "startfile",
            lambda p: called.append(("startfile", p)),
            raising=False,
        )
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: called.append(("run", a)),
        )
        ctrl._registry_action_open_folder(rows[0])
        assert len(called) == 1
        # Whichever launcher fires depends on platform; just verify
        # something fired.

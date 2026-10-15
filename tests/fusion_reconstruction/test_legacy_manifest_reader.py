"""Tests for the Tier 2b schema 1.0/1.1 → 2.0 in-memory reader.

Legacy refined-model bundles were written by the pre-restructure
``_register_model_pack`` path as ``model_pack.json`` (schema 1.0)
with absolute file-system paths in ``lateral_model`` / ``axial_model``.
The new registry dialog and downstream code expects schema 2.0 shape
(``manifest["lateral"]["weights_path"]`` relative-to-bundle).

The reader normalizes both forms so the registry treats legacy
bundles as first-class citizens (just tagged ``from_legacy_schema``
for UI display).  No on-disk migration — pure read-side.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
        "_read_bundle_manifest_normalized",
        "_convert_legacy_manifest_to_v2",
        "_scan_refinement_model_registry",
        "_read_bundle_training_channels",
    ):
        method = getattr(FusionReconstructionController, name)
        setattr(ctrl, name, method.__get__(ctrl))
    return ctrl


# ---------------------------------------------------------------------------
# Native schema-2.0 path (manifest.json present)
# ---------------------------------------------------------------------------


class TestNativeManifest:

    def test_returns_v2_as_is(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "v2_bundle")
        bundle.mkdir(parents=True)
        manifest = {
            "schema_version": "2.0",
            "name": "v2_bundle",
            "lateral": {"weights_path": "lateral", "n_train_pairs": 42},
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert result == manifest
        # No legacy marker in native bundles.
        assert "_from_legacy_schema" not in result

    def test_corrupt_v2_logs_and_returns_none(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "corrupt_v2")
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text("NOT JSON")
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert result is None
        assert any("manifest.json corrupt" in log for log in ctrl._logs)

    def test_v2_takes_precedence_over_legacy(self, tmp_path):
        """If both manifest.json AND model_pack.json exist, v2 wins."""
        bundle = _paths.refined_model_dir(tmp_path, "both_files")
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text(json.dumps({
            "schema_version": "2.0",
            "lateral": {"weights_path": "modern"},
        }))
        (bundle / "model_pack.json").write_text(json.dumps({
            "schema_version": "1.0",
            "lateral_model": str(bundle / "legacy"),
        }))
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert result["lateral"]["weights_path"] == "modern"
        assert "_from_legacy_schema" not in result


# ---------------------------------------------------------------------------
# Legacy schema-1.0 conversion
# ---------------------------------------------------------------------------


class TestLegacyConversion:

    def test_lateral_only_absolute_path(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "legacy_lat")
        bundle.mkdir(parents=True)
        # Schema 1.0 wrote absolute paths.
        weights_file = bundle / "lateral_model"
        weights_file.write_bytes(b"weights")
        (bundle / "model_pack.json").write_text(json.dumps({
            "schema_version": "1.0",
            "mode": "Fine-tune Lateral",
            "source_workspace": "/old/workspace",
            "lateral_model": str(weights_file),
        }))
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert result is not None
        assert result["schema_version"] == "2.0"
        assert result["_from_legacy_schema"] == "1.0"
        assert result["_legacy_mode"] == "Fine-tune Lateral"
        assert "lateral" in result
        assert result["lateral"]["weights_path"] == "lateral_model"
        assert result["lateral"]["init"]["mode"] == "?"
        assert result["lateral"]["n_train_pairs"] is None
        # No axial bucket.
        assert "axial" not in result

    def test_both_buckets(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "legacy_both")
        bundle.mkdir(parents=True)
        lat = bundle / "lateral_model"
        ax = bundle / "axial_model"
        lat.write_bytes(b"l")
        ax.write_bytes(b"a")
        (bundle / "model_pack.json").write_text(json.dumps({
            "schema_version": "1.0",
            "mode": "Fine-tune Both",
            "lateral_model": str(lat),
            "axial_model": str(ax),
        }))
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert "lateral" in result and "axial" in result
        assert result["lateral"]["weights_path"] == "lateral_model"
        assert result["axial"]["weights_path"] == "axial_model"
        assert result["axial"]["backend"] == "cellpose"

    def test_relative_path_preserved(self, tmp_path):
        """Some legacy paths may already be relative — they should
        round-trip identically (no .resolve() drift)."""
        bundle = _paths.refined_model_dir(tmp_path, "legacy_rel")
        bundle.mkdir(parents=True)
        weights = bundle / "lateral_model"
        weights.write_bytes(b"w")
        # Store the legacy value as relative to bundle (just the filename).
        (bundle / "model_pack.json").write_text(json.dumps({
            "schema_version": "1.0",
            "lateral_model": "lateral_model",
        }))
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        # Falls back to filename — same outcome.
        assert result["lateral"]["weights_path"] == "lateral_model"

    def test_path_outside_bundle_falls_back_to_filename(self, tmp_path):
        """If the legacy absolute path doesn't live under bundle_dir
        (e.g., bundle was copied elsewhere), the converter falls back
        to the filename component."""
        bundle = _paths.refined_model_dir(tmp_path, "moved_bundle")
        bundle.mkdir(parents=True)
        other_root = tmp_path / "elsewhere"
        other_root.mkdir()
        outside = other_root / "lateral_model"
        outside.write_bytes(b"w")
        (bundle / "model_pack.json").write_text(json.dumps({
            "schema_version": "1.0",
            "lateral_model": str(outside),
        }))
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert result["lateral"]["weights_path"] == "lateral_model"

    def test_schema_1_1_carried_through(self, tmp_path):
        """Schema marker in the legacy file is preserved in the
        ``_from_legacy_schema`` marker."""
        bundle = _paths.refined_model_dir(tmp_path, "v1_1_bundle")
        bundle.mkdir(parents=True)
        (bundle / "model_pack.json").write_text(json.dumps({
            "schema_version": "1.1",
            "lateral_model": "lateral",
        }))
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert result["_from_legacy_schema"] == "1.1"

    def test_missing_mode_omits_legacy_mode_marker(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "no_mode")
        bundle.mkdir(parents=True)
        (bundle / "model_pack.json").write_text(json.dumps({
            "schema_version": "1.0",
            "lateral_model": "lateral",
        }))
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert "_legacy_mode" not in result


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:

    def test_no_manifest_files_returns_none(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "empty_bundle")
        bundle.mkdir(parents=True)
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert result is None

    def test_corrupt_legacy_logs_and_returns_none(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "corrupt_legacy")
        bundle.mkdir(parents=True)
        (bundle / "model_pack.json").write_text("{ not valid json")
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert result is None
        assert any("model_pack.json corrupt" in log for log in ctrl._logs)

    def test_legacy_with_no_buckets_returns_minimal_dict(self, tmp_path):
        """Legacy file present but no lateral_model / axial_model keys —
        returns a valid normalized dict with neither bucket populated."""
        bundle = _paths.refined_model_dir(tmp_path, "legacy_empty")
        bundle.mkdir(parents=True)
        (bundle / "model_pack.json").write_text(json.dumps({
            "schema_version": "1.0",
        }))
        ctrl = _make_controller(tmp_path)
        result = ctrl._read_bundle_manifest_normalized(bundle)
        assert result is not None
        assert "lateral" not in result
        assert "axial" not in result
        assert result["schema_version"] == "2.0"
        assert result["_from_legacy_schema"] == "1.0"


# ---------------------------------------------------------------------------
# Scanner integration with legacy bundles
# ---------------------------------------------------------------------------


class TestScannerIntegrationWithLegacy:
    """Verify the scanner picks up legacy bundles and tags them
    correctly via the new ``from_legacy_schema`` row field."""

    def test_legacy_bundle_appears_in_rows(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "legacy_in_scan")
        bundle.mkdir(parents=True)
        (bundle / "model_pack.json").write_text(json.dumps({
            "schema_version": "1.0",
            "mode": "Fine-tune Lateral",
            "created_at": "2026-04-01T00:00:00Z",
            "lateral_model": "lateral_model",
        }))
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "legacy_in_scan"
        assert row["mode"] == "Lateral"
        assert row["from_legacy_schema"] == "1.0"
        # Normalized schema in the row is the converted version.
        assert row["schema_version"] == "2.0"

    def test_native_bundle_has_empty_legacy_field(self, tmp_path):
        bundle = _paths.refined_model_dir(tmp_path, "native_bundle")
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text(json.dumps({
            "schema_version": "2.0",
            "name": "native_bundle",
            "lateral": {"weights_path": "lateral"},
        }))
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        assert len(rows) == 1
        assert rows[0]["from_legacy_schema"] == ""

    def test_mixed_legacy_and_native_bundles(self, tmp_path):
        """Workspaces in the wild may have a mix.  Scanner returns
        both, sorted newest-first by ``created_at``."""
        # Older legacy bundle.
        legacy = _paths.refined_model_dir(tmp_path, "legacy_older")
        legacy.mkdir(parents=True)
        (legacy / "model_pack.json").write_text(json.dumps({
            "schema_version": "1.0",
            "created_at": "2026-03-01T00:00:00Z",
            "lateral_model": "lateral_model",
        }))
        # Newer native bundle.
        native = _paths.refined_model_dir(tmp_path, "native_newer")
        native.mkdir(parents=True)
        (native / "manifest.json").write_text(json.dumps({
            "schema_version": "2.0",
            "created_at": "2026-05-01T00:00:00Z",
            "lateral": {"weights_path": "lateral"},
        }))
        ctrl = _make_controller(tmp_path)
        rows = ctrl._scan_refinement_model_registry()
        assert len(rows) == 2
        # Newest first.
        assert rows[0]["name"] == "native_newer"
        assert rows[0]["from_legacy_schema"] == ""
        assert rows[1]["name"] == "legacy_older"
        assert rows[1]["from_legacy_schema"] == "1.0"

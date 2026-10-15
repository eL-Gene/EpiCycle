"""Tests for ``FusionReconstructionController._resolve_patch_payload``.

Post-2026-05-20 layout cutover: this helper is the canonical
patch_name-based manifest-payload resolver, replacing the legacy
``record["reconstruction_run_id"]`` → manifest-scan lookup that broke
after migration drops the field.

The helper is used by every read site that previously asked
"what's the labels_path for this patch?" — view reconstruction,
open 3D curation, QC config builder, get_active_* helpers, etc.
Audit 2026-05-20 added these tests after the second-round audit
flagged the helper had zero coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest


class _FakeStateManager:
    """Minimal state manager — empty state, mirrors the interface
    that _resolve_patch_payload doesn't touch."""

    def __init__(self):
        self._state: Dict[str, Any] = {"patches": {}}


def _make_controller_for_resolve(working_dir: Path):
    """Build a minimal controller-like object exposing just what
    _resolve_patch_payload needs: working_dir + _manifest_cache.

    Imports the real method off the controller class so we test the
    actual production code, not a re-implementation.
    """
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = SimpleNamespace(
        working_dir=str(working_dir),
        _manifest_cache={},
        state_manager=_FakeStateManager(),
    )
    # Bind the unbound method to our SimpleNamespace.
    ctrl._resolve_patch_payload = (
        FusionReconstructionController._resolve_patch_payload.__get__(ctrl)
    )
    return ctrl


def _write_patch_manifest(workspace: Path, patch_name: str,
                          labels_path_in_manifest: str = "",
                          source_stack_path: str = "",
                          seg_config: dict | None = None,
                          create_labels_file: bool = False) -> Path:
    """Build a synthetic per-patch run dir with manifest.json."""
    run_dir = workspace / "artifacts" / "step3" / "runs" / patch_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if create_labels_file:
        (run_dir / "labels_3d.tif").write_bytes(b"fake-tiff")
    manifest = {
        "run_id": "20260520T120000Z",
        "labels_path": labels_path_in_manifest or str(
            run_dir / "labels_3d.tif"
        ),
        "source_stack_path": source_stack_path,
    }
    if seg_config is not None:
        manifest["seg_config"] = seg_config
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class TestResolvePatchPayload:

    def test_empty_patch_name_returns_empty(self, tmp_path):
        ctrl = _make_controller_for_resolve(tmp_path)
        assert ctrl._resolve_patch_payload("") == {}

    def test_no_working_dir_returns_empty(self, tmp_path):
        ctrl = _make_controller_for_resolve(tmp_path)
        ctrl.working_dir = ""
        assert ctrl._resolve_patch_payload("patch_001.tif") == {}

    def test_missing_manifest_returns_empty(self, tmp_path):
        ctrl = _make_controller_for_resolve(tmp_path)
        # No manifest file written.
        assert ctrl._resolve_patch_payload("patch_001.tif") == {}

    def test_returns_parsed_manifest(self, tmp_path):
        _write_patch_manifest(
            tmp_path, "patch_001.tif",
            source_stack_path="/some/source/patch_001.tif",
            create_labels_file=True,
        )
        ctrl = _make_controller_for_resolve(tmp_path)
        payload = ctrl._resolve_patch_payload("patch_001.tif")
        assert payload["run_id"] == "20260520T120000Z"
        assert payload["source_stack_path"] == "/some/source/patch_001.tif"

    def test_labels_path_overridden_to_new_layout(self, tmp_path):
        """Pre-migration manifests have stale labels_path pointing at
        the legacy reconstruction dir.  _resolve_patch_payload must
        override it with the post-cutover canonical location."""
        _write_patch_manifest(
            tmp_path, "patch_001.tif",
            labels_path_in_manifest="/old/legacy/artifacts/reconstruction/uuid-abc/labels_3d.tif",
            create_labels_file=True,
        )
        ctrl = _make_controller_for_resolve(tmp_path)
        payload = ctrl._resolve_patch_payload("patch_001.tif")
        expected = str(
            tmp_path / "artifacts" / "step3" / "runs"
            / "patch_001.tif" / "labels_3d.tif"
        )
        assert payload["labels_path"] == expected

    def test_labels_path_NOT_overridden_when_new_file_missing(self, tmp_path):
        """If the new-layout labels file doesn't exist on disk, keep
        the manifest's recorded labels_path so callers can see the
        configured path even if the file was deleted."""
        _write_patch_manifest(
            tmp_path, "patch_001.tif",
            labels_path_in_manifest="/some/path/labels.tif",
            create_labels_file=False,
        )
        ctrl = _make_controller_for_resolve(tmp_path)
        payload = ctrl._resolve_patch_payload("patch_001.tif")
        assert payload["labels_path"] == "/some/path/labels.tif"

    def test_corrupt_manifest_returns_empty(self, tmp_path):
        """Corrupt JSON should not crash; return empty dict."""
        run_dir = tmp_path / "artifacts" / "step3" / "runs" / "patch_001.tif"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text("NOT VALID JSON", encoding="utf-8")
        ctrl = _make_controller_for_resolve(tmp_path)
        assert ctrl._resolve_patch_payload("patch_001.tif") == {}

    def test_cache_hit_on_repeat_call(self, tmp_path):
        _write_patch_manifest(
            tmp_path, "patch_001.tif", create_labels_file=True,
        )
        ctrl = _make_controller_for_resolve(tmp_path)
        ctrl._resolve_patch_payload("patch_001.tif")
        cache_key = "patch:patch_001.tif"
        assert cache_key in ctrl._manifest_cache
        # Second call should return the cached entry even if we
        # corrupt the on-disk manifest mid-test.
        manifest_path = (
            tmp_path / "artifacts" / "step3" / "runs"
            / "patch_001.tif" / "manifest.json"
        )
        manifest_path.write_text("CORRUPT NOW", encoding="utf-8")
        payload = ctrl._resolve_patch_payload("patch_001.tif")
        # Cached payload survived the on-disk corruption.
        assert payload != {}

    def test_cache_key_prefix_isolation_from_run_id_entries(self, tmp_path):
        """Verify the patch: prefix doesn't collide with legacy
        run_id entries in the same cache."""
        _write_patch_manifest(
            tmp_path, "20260520T120000Z", create_labels_file=True,
        )
        ctrl = _make_controller_for_resolve(tmp_path)
        # Pre-populate the cache with a legacy run_id-keyed entry
        # whose KEY happens to match the patch_name.
        ctrl._manifest_cache["20260520T120000Z"] = {"legacy": "value"}
        # New helper writes under "patch:" prefix → no collision.
        payload = ctrl._resolve_patch_payload("20260520T120000Z")
        assert payload.get("legacy") != "value"
        assert "patch:20260520T120000Z" in ctrl._manifest_cache
        # Legacy entry still intact.
        assert ctrl._manifest_cache["20260520T120000Z"] == {"legacy": "value"}

    def test_seg_config_passed_through(self, tmp_path):
        _write_patch_manifest(
            tmp_path, "patch_001.tif",
            seg_config={"channels": [2, 0], "diameter": 30.0},
            create_labels_file=True,
        )
        ctrl = _make_controller_for_resolve(tmp_path)
        payload = ctrl._resolve_patch_payload("patch_001.tif")
        sc = payload.get("seg_config", {})
        assert sc.get("channels") == [2, 0]


class TestPatchLayerKey:
    """Coverage for the sibling helper that derives viewer-layer names."""

    def _ctrl(self, tmp_path):
        return _make_controller_for_resolve(tmp_path)

    def test_typical_patch_name(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        c = self._ctrl(tmp_path)
        c._patch_layer_key = (
            FusionReconstructionController._patch_layer_key.__get__(c)
        )
        assert c._patch_layer_key("patch_001.tif") == "patch_001"

    def test_no_extension(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        c = self._ctrl(tmp_path)
        c._patch_layer_key = (
            FusionReconstructionController._patch_layer_key.__get__(c)
        )
        assert c._patch_layer_key("patch_001") == "patch_001"

    def test_empty_returns_empty(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        c = self._ctrl(tmp_path)
        c._patch_layer_key = (
            FusionReconstructionController._patch_layer_key.__get__(c)
        )
        assert c._patch_layer_key("") == ""

    def test_strips_only_last_suffix(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        c = self._ctrl(tmp_path)
        c._patch_layer_key = (
            FusionReconstructionController._patch_layer_key.__get__(c)
        )
        # Path.stem strips only the last suffix.
        assert c._patch_layer_key("patch.foo.bar.tif") == "patch.foo.bar"

from __future__ import annotations

import json
import time
from pathlib import Path

from napari_epicycle.fusion_reconstruction.io import ReconstructionRunStore


def test_run_store_lists_manifests_newest_first(tmp_path):
    run_old = tmp_path / "artifacts" / "step3" / "runs" / "20260101T000000Z"
    run_new = tmp_path / "artifacts" / "step3" / "runs" / "20260101T000100Z"
    run_old.mkdir(parents=True, exist_ok=True)
    run_new.mkdir(parents=True, exist_ok=True)

    old_manifest = run_old / "manifest.json"
    new_manifest = run_new / "manifest.json"
    old_manifest.write_text(json.dumps({"run_id": "old"}), encoding="utf-8")
    time.sleep(0.01)
    new_manifest.write_text(json.dumps({"run_id": "new"}), encoding="utf-8")

    store = ReconstructionRunStore()
    manifests = store.list_manifests(str(tmp_path))
    assert Path(manifests[0]).parent.name == "20260101T000100Z"
    assert Path(manifests[1]).parent.name == "20260101T000000Z"


def test_run_store_lists_manifests_under_reconstructions_root(tmp_path):
    """Phase 1b-ii: with reconstructions_root, glob under the nested run dir."""
    recon_root = (
        tmp_path / "artifacts" / "step3" / "runs" / "cells" / "reconstructions"
    )
    p_a = recon_root / "patch_000"
    p_b = recon_root / "patch_001"
    p_a.mkdir(parents=True, exist_ok=True)
    p_b.mkdir(parents=True, exist_ok=True)
    (p_a / "manifest.json").write_text(json.dumps({"run_id": "a"}), encoding="utf-8")
    time.sleep(0.01)
    (p_b / "manifest.json").write_text(json.dumps({"run_id": "b"}), encoding="utf-8")

    store = ReconstructionRunStore()
    manifests = store.list_manifests(
        str(tmp_path), reconstructions_root=str(recon_root)
    )
    assert len(manifests) == 2
    # newest first
    assert Path(manifests[0]).parent.name == "patch_001"
    assert Path(manifests[1]).parent.name == "patch_000"
    # The flat default would have found nothing (no manifests under runs/* directly)
    assert store.list_manifests(str(tmp_path)) == []


def test_run_store_reconstructions_root_none_byte_identical(tmp_path):
    """reconstructions_root=None == today's flat runs/* glob."""
    flat = tmp_path / "artifacts" / "step3" / "runs" / "patch_000"
    flat.mkdir(parents=True, exist_ok=True)
    (flat / "manifest.json").write_text(json.dumps({"run_id": "x"}), encoding="utf-8")

    store = ReconstructionRunStore()
    default = store.list_manifests(str(tmp_path))
    # passing the equivalent flat root explicitly yields the same result
    explicit = store.list_manifests(
        str(tmp_path),
        reconstructions_root=str(tmp_path / "artifacts" / "step3" / "runs"),
    )
    assert default == explicit
    assert Path(default[0]).parent.name == "patch_000"


def test_run_store_load_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"run_id": "abc123", "mode": "placeholder"}), encoding="utf-8")

    store = ReconstructionRunStore()
    payload = store.load_manifest(str(path))
    assert payload["run_id"] == "abc123"
    assert payload["mode"] == "placeholder"

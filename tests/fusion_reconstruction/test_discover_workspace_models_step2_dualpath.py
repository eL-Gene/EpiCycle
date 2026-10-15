"""Tests for the Step 2 dual-path scanner extension to
``_discover_workspace_models()`` in
``napari_epicycle/fusion_reconstruction/controller.py``.

As a Phase 4 prerequisite, Step 3's discovery
function must walk the NEW Step 2 layout at
``artifacts/step2/runs/*/{lateral,axial}/models/round_*/`` in
addition to the legacy ``models/round_*/`` location.  Files in the
new layout typically have NO orientation keyword in their filename
(Cellpose ``model_name="weights"`` convention); the orientation comes
from the bucket parent directory.

These tests verify the scanner correctly:
* Discovers files in the new bucket-first layout
* Uses parent-dir as orientation source when filename lacks keyword
* Continues to find files in the legacy layout
* Combines candidates from both, picks newest per orientation
* Doesn't crash if Step 2 paths module is unavailable
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


# Fixture: simple workspace with both layouts populated
#
# Phase B.2.1 (2026-05-25 audit 3 HIGH-1 fix): rewrote the local
# `_scan_for_models` mirror to match post-B.2 production dual-path
# behavior.  Pre-fix the mirror only scanned the legacy split layout
# (`bucket/models/round_NN/`) and the fixtures planted there too —
# tests passed vacuously while production was updated to walk the
# new flat layout.  Same vacuous-pass pattern as the 2026-05-20
# Phase 5 cutover incident.
#
# Post-fix fixtures plant data at BOTH layouts as needed; mirror
# walks dual-path matching post-B.2 production code at
# `napari_epicycle/fusion_reconstruction/controller.py:1746-1785`.
@pytest.fixture
def hybrid_ws(tmp_path):
    """Workspace with files in BOTH layouts populated under DIFFERENT
    rounds (round_00 in legacy split, round_01 in flat).  Verifies
    dual-path scanner surfaces both."""
    ws = tmp_path / "hybrid"
    # Workspace-level legacy: models/round_00/{lateral,axial}_model
    legacy = ws / "models" / "round_00"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "model_lateral_round_001.ckpt").write_bytes(b"legacy lateral")
    (legacy / "model_axial_round_001.ckpt").write_bytes(b"legacy axial")

    # Step 2 LEGACY SPLIT round_00: artifacts/step2/runs/default/{bucket}/models/round_00/
    for bucket in ("lateral", "axial"):
        split_dir = ws / "artifacts" / "step2" / "runs" / "default" / bucket / "models" / "round_00"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / f"default_{bucket}").write_bytes(f"split {bucket}".encode())
        (split_dir / f"default_{bucket}_train_loss.npy").write_bytes(b"loss")
        (split_dir / "manifest.json").write_text("{}")  # ignored

    # Step 2 NEW FLAT round_01: artifacts/step2/runs/default/{bucket}/round_01/model/
    for bucket in ("lateral", "axial"):
        flat_dir = ws / "artifacts" / "step2" / "runs" / "default" / bucket / "round_01" / "model"
        flat_dir.mkdir(parents=True, exist_ok=True)
        (flat_dir / f"default_{bucket}_round_01").write_bytes(f"flat {bucket}".encode())
        (flat_dir / f"default_{bucket}_round_01_train_loss.npy").write_bytes(b"loss")
        (flat_dir / "manifest.json").write_text("{}")  # ignored

    return ws


@pytest.fixture
def new_only_ws(tmp_path):
    """Workspace with ONLY the NEW FLAT layout (Phase B.2 fresh user)."""
    ws = tmp_path / "new_only"
    for bucket in ("lateral", "axial"):
        flat_dir = ws / "artifacts" / "step2" / "runs" / "default" / bucket / "round_00" / "model"
        flat_dir.mkdir(parents=True, exist_ok=True)
        (flat_dir / f"default_{bucket}_round_00").write_bytes(f"{bucket} weights".encode())
        (flat_dir / f"default_{bucket}_round_00_train_loss.npy").write_bytes(b"loss")
    return ws


@pytest.fixture
def legacy_split_only_ws(tmp_path):
    """Phase B.2.1 NEW: Workspace with ONLY the LEGACY SPLIT layout
    (post-Phase-6c, pre-Phase-B.2 user who hasn't migrated yet)."""
    ws = tmp_path / "legacy_split_only"
    for bucket in ("lateral", "axial"):
        split_dir = ws / "artifacts" / "step2" / "runs" / "default" / bucket / "models" / "round_00"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / f"default_{bucket}").write_bytes(f"split {bucket}".encode())
        (split_dir / f"default_{bucket}_train_loss.npy").write_bytes(b"loss")
    return ws


@pytest.fixture
def multi_run_ws(tmp_path):
    """Multiple runs all on the NEW FLAT layout (post-B.2)."""
    ws = tmp_path / "multi_run"
    for run_id in ("default", "nuclei_v1"):
        for bucket in ("lateral", "axial"):
            flat_dir = ws / "artifacts" / "step2" / "runs" / run_id / bucket / "round_00" / "model"
            flat_dir.mkdir(parents=True, exist_ok=True)
            (flat_dir / f"{run_id}_{bucket}_round_00").write_bytes(
                f"{run_id} {bucket}".encode()
            )
    return ws


@pytest.fixture
def same_round_both_layouts_ws(tmp_path):
    """Phase B.2.1 NEW: Same round_NN in BOTH layouts (partial migration
    edge case).  Dual-path scanner must prefer NEW flat over legacy split
    to avoid double-counting."""
    ws = tmp_path / "duplicate_round"
    for bucket in ("lateral", "axial"):
        # LEGACY split — same round_00
        split_dir = ws / "artifacts" / "step2" / "runs" / "default" / bucket / "models" / "round_00"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / f"default_{bucket}").write_bytes(f"OLD split {bucket}".encode())
        # NEW flat — same round_00
        flat_dir = ws / "artifacts" / "step2" / "runs" / "default" / bucket / "round_00" / "model"
        flat_dir.mkdir(parents=True, exist_ok=True)
        (flat_dir / f"default_{bucket}_round_00").write_bytes(f"NEW flat {bucket}".encode())
    return ws


def _scan_for_models(working_dir):
    """Mirror of `_discover_workspace_models` in
    `napari_epicycle/fusion_reconstruction/controller.py:1722-1810`.

    Phase B.2 dual-path: walks BOTH `{bucket}/round_NN/model/` (new
    flat) AND `{bucket}/models/round_NN/` (legacy split).  When a
    round_NN exists in BOTH, prefers the new-layout entry.

    This is a TEST MIRROR — production controller's method does the
    same logic but on a Qt-instantiated controller (requires Qt + napari).
    To keep these unit tests Qt-free, the mirror replicates production
    1:1.  IF YOU CHANGE THE PRODUCTION SCANNER, update this mirror to
    match — drift = vacuous-pass tests.

    Returns (latest_lateral, latest_axial) tuple of file paths (or
    empty strings if no candidate found).
    """
    from napari_epicycle.fusion_reconstruction import _paths

    step2_root = _paths.legacy_models_root(working_dir)
    step3_refinement_root = _paths.refined_models_root(working_dir)

    subdirs_to_scan = []
    if step2_root.exists():
        for sub in step2_root.iterdir():
            if sub.is_dir() and sub.name.startswith("round_"):
                subdirs_to_scan.append((sub, None))
    if step3_refinement_root.exists():
        for sub in step3_refinement_root.iterdir():
            if sub.is_dir():
                subdirs_to_scan.append((sub, None))

    try:
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        step2_runs_root = _step2_paths.step2_runs_root(working_dir)
        if step2_runs_root.is_dir():
            for run_dir in step2_runs_root.iterdir():
                if not run_dir.is_dir():
                    continue
                for bucket in ("lateral", "axial"):
                    bucket_dir = run_dir / bucket
                    if not bucket_dir.is_dir():
                        continue
                    seen_round_ids = set()
                    # NEW FLAT layout — bucket/round_NN/model/
                    for round_dir in bucket_dir.iterdir():
                        if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
                            continue
                        model_sub = round_dir / "model"
                        if model_sub.is_dir():
                            subdirs_to_scan.append((model_sub, bucket))
                            seen_round_ids.add(round_dir.name)
                    # LEGACY split layout — bucket/models/round_NN/
                    legacy_models = bucket_dir / "models"
                    if legacy_models.is_dir():
                        for round_dir in legacy_models.iterdir():
                            if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
                                continue
                            if round_dir.name in seen_round_ids:
                                continue
                            subdirs_to_scan.append((round_dir, bucket))
    except Exception:
        pass

    candidates = []
    for sub, orientation_hint in subdirs_to_scan:
        for f in sub.rglob("*"):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if any(skip in name_lower for skip in
                   ("loss", "stat", "diam", ".log", ".json")):
                continue
            file_orientation = orientation_hint
            if file_orientation is None:
                if "lateral" in name_lower or "xy" in name_lower:
                    file_orientation = "lateral"
                elif "axial" in name_lower or "xz" in name_lower or "yz" in name_lower:
                    file_orientation = "axial"
                else:
                    continue
            try:
                candidates.append((f.stat().st_mtime, f, file_orientation))
            except OSError:
                continue
    candidates.sort(key=lambda x: x[0], reverse=True)

    latest_lateral = ""
    latest_axial = ""
    for _, f, orientation in candidates:
        if not latest_lateral and orientation == "lateral":
            latest_lateral = str(f)
        if not latest_axial and orientation == "axial":
            latest_axial = str(f)
        if latest_lateral and latest_axial:
            break
    return latest_lateral, latest_axial


# ============================================================================
# Scanner behavior tests
# ============================================================================

class TestNewLayoutScanning:
    def test_new_only_layout_discovers_lateral(self, new_only_ws):
        lat, ax = _scan_for_models(new_only_ws)
        assert lat != ""
        # Phase B.2: file under `lateral/round_NN/model/` (flat layout)
        assert "/lateral/" in lat.replace("\\", "/")
        assert "/model/" in lat.replace("\\", "/")

    def test_new_only_layout_discovers_axial(self, new_only_ws):
        lat, ax = _scan_for_models(new_only_ws)
        assert ax != ""
        assert "/axial/" in ax.replace("\\", "/")
        assert "/model/" in ax.replace("\\", "/")

    def test_orientation_from_parent_dir_not_filename(self, new_only_ws):
        # Phase B.2: weights filename now `default_{bucket}_round_00`
        # which DOES contain the bucket keyword, but the scanner uses
        # the parent-dir hint regardless.  Even if filename were
        # ambiguous, parent dir would route it.
        lat, ax = _scan_for_models(new_only_ws)
        assert Path(lat).name == "default_lateral_round_00"
        assert Path(ax).name == "default_axial_round_00"

    def test_loss_companions_excluded(self, new_only_ws):
        lat, ax = _scan_for_models(new_only_ws)
        assert "loss" not in Path(lat).name.lower()
        assert "loss" not in Path(ax).name.lower()

    def test_manifest_json_excluded(self, hybrid_ws):
        lat, ax = _scan_for_models(hybrid_ws)
        assert not lat.endswith(".json")
        assert not ax.endswith(".json")


class TestLegacySplitLayoutScanning:
    """Phase B.2.1 NEW: legacy-split-only workspace coverage (was missing
    per audit 3 HIGH).  Verifies dual-path scanner still surfaces pre-
    Phase-B.2 split-layout models for un-migrated users."""

    def test_legacy_split_discovers_lateral(self, legacy_split_only_ws):
        lat, ax = _scan_for_models(legacy_split_only_ws)
        assert lat != ""
        # Legacy split — file under `lateral/models/round_NN/`
        lat_norm = lat.replace("\\", "/")
        assert "/lateral/models/" in lat_norm

    def test_legacy_split_discovers_axial(self, legacy_split_only_ws):
        lat, ax = _scan_for_models(legacy_split_only_ws)
        assert ax != ""
        ax_norm = ax.replace("\\", "/")
        assert "/axial/models/" in ax_norm

    def test_legacy_split_filename_is_pre_b2_naming(self, legacy_split_only_ws):
        # Pre-B.2 weights file: `default_{bucket}` (no round suffix)
        lat, ax = _scan_for_models(legacy_split_only_ws)
        assert Path(lat).name == "default_lateral"
        assert Path(ax).name == "default_axial"


class TestHybridLayoutScanning:
    def test_picks_newer_file_across_layouts(self, hybrid_ws):
        # hybrid_ws has SPLIT round_00 + FLAT round_01.  Make round_01
        # files newer; picker should take them.
        new_lat = (hybrid_ws / "artifacts" / "step2" / "runs"
                   / "default" / "lateral" / "round_01" / "model"
                   / "default_lateral_round_01")
        new_ax = (hybrid_ws / "artifacts" / "step2" / "runs"
                  / "default" / "axial" / "round_01" / "model"
                  / "default_axial_round_01")
        future = time.time() + 1000
        import os
        os.utime(new_lat, (future, future))
        os.utime(new_ax, (future, future))

        lat, ax = _scan_for_models(hybrid_ws)
        # Flat layout file under `bucket/round_01/model/`
        assert "/round_01/model/" in lat.replace("\\", "/")
        assert "/round_01/model/" in ax.replace("\\", "/")

    def test_falls_back_to_legacy_when_new_is_older(self, hybrid_ws):
        # hybrid_ws has SPLIT round_00 + FLAT round_01.  Make SPLIT
        # files newer; picker should fall back to legacy split.
        legacy_lat = (hybrid_ws / "artifacts" / "step2" / "runs"
                      / "default" / "lateral" / "models" / "round_00"
                      / "default_lateral")
        legacy_ax = (hybrid_ws / "artifacts" / "step2" / "runs"
                     / "default" / "axial" / "models" / "round_00"
                     / "default_axial")
        future = time.time() + 1000
        import os
        os.utime(legacy_lat, (future, future))
        os.utime(legacy_ax, (future, future))

        lat, ax = _scan_for_models(hybrid_ws)
        # Legacy split: file under `bucket/models/round_NN/`
        assert "/models/round_" in lat.replace("\\", "/")
        assert "/models/round_" in ax.replace("\\", "/")


class TestSameRoundBothLayoutsDedup:
    """Phase B.2.1 NEW: when same round_NN exists in BOTH layouts
    (partial migration), the new-layout entry must win regardless of
    mtime (legacy is treated as stale)."""

    def test_new_wins_on_partial_migration(self, same_round_both_layouts_ws):
        # Touch LEGACY to be newer to verify new-layout wins despite mtime
        ws = same_round_both_layouts_ws
        for bucket in ("lateral", "axial"):
            legacy_file = (ws / "artifacts" / "step2" / "runs" / "default"
                           / bucket / "models" / "round_00" / f"default_{bucket}")
            future = time.time() + 1000
            import os
            os.utime(legacy_file, (future, future))

        lat, ax = _scan_for_models(ws)
        # Even though legacy is mtime-newer, NEW layout wins because
        # dedup excludes legacy round_00 once seen in new layout
        assert "/round_00/model/" in lat.replace("\\", "/"), \
            f"Expected new-layout flat path; got {lat}"
        assert "/round_00/model/" in ax.replace("\\", "/")


class TestMultiRunScanning:
    def test_multi_run_walks_all_runs(self, multi_run_ws):
        lat, ax = _scan_for_models(multi_run_ws)
        assert lat != ""
        assert ax != ""
        for path in (lat, ax):
            normalized = path.replace("\\", "/")
            assert "/runs/" in normalized
            # Phase B.2: file under `bucket/round_NN/model/`
            assert "/round_00/model/" in normalized


class TestEmptyOrMissing:
    def test_empty_workspace_returns_empty(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        lat, ax = _scan_for_models(ws)
        assert lat == ""
        assert ax == ""

    def test_legacy_only_with_no_round_dirs(self, tmp_path):
        ws = tmp_path / "no_rounds"
        (ws / "models").mkdir(parents=True)
        lat, ax = _scan_for_models(ws)
        assert lat == ""
        assert ax == ""

    def test_one_bucket_only_other_empty(self, tmp_path):
        # Flat layout but only lateral models exist
        ws = tmp_path / "lateral_only"
        flat_dir = ws / "artifacts" / "step2" / "runs" / "default" / "lateral" / "round_00" / "model"
        flat_dir.mkdir(parents=True, exist_ok=True)
        (flat_dir / "default_lateral_round_00").write_bytes(b"lat only")
        lat, ax = _scan_for_models(ws)
        assert lat != ""
        assert ax == ""


class TestStep2ImportMissing:
    """If the Step 2 paths module is somehow missing, scanner should
    not crash — just fall back to legacy + Step 3 refinement."""

    def test_legacy_still_discovered_if_step2_missing(self, hybrid_ws, monkeypatch):
        # Simulate ImportError on _step2_paths import
        import builtins
        real_import = builtins.__import__

        def selective_import(name, *args, **kwargs):
            if "hitl_training_annotation._paths" in name:
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", selective_import)
        lat, ax = _scan_for_models(hybrid_ws)
        # Legacy files should still be found
        assert lat != ""
        assert ax != ""

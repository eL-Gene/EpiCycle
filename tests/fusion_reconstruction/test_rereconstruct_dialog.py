"""Tests for the Tier 2a re-reconstruction enumeration helper.

The themed dialog itself is Qt-heavy and exercised manually; this
module covers the pure-Python part that drives its content:
``_enumerate_rereconstruct_targets`` reads disk + builds a summary
of what gets DELETED vs PRESERVED, per affected patch.

Tests confirm:
* Only existing artifacts are flagged
* Per-patch counts sum correctly
* No cross-patch leakage (one patch's artifacts don't show up in
  another patch's summary)
* Empty / missing workspace handled gracefully
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_epicycle.fusion_reconstruction import _paths


def _make_controller(working_dir: Path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = SimpleNamespace(working_dir=str(working_dir))
    ctrl._enumerate_rereconstruct_targets = (
        FusionReconstructionController._enumerate_rereconstruct_targets.__get__(ctrl)
    )
    return ctrl


def _touch(path: Path, content: bytes = b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _make_full_artifact_set(working_dir: Path, patch_name: str):
    """Create every artifact kind on disk for a single patch."""
    patch_stem = Path(patch_name).stem
    _touch(_paths.patch_run_dir(working_dir, patch_name) / "labels_3d.tif")
    _touch(_paths.qc_features_csv(working_dir, patch_stem))
    _touch(_paths.qc_features_raw_csv(working_dir, patch_stem))
    _touch(_paths.qc_adjacency_raw_csv(working_dir, patch_stem))
    _touch(_paths.qc_adjacency_curated_csv(working_dir, patch_stem))
    # curation/ + refinement/pairs/ are dirs
    curation_dir = _paths.patch_curation_dir(working_dir, patch_stem)
    curation_dir.mkdir(parents=True, exist_ok=True)
    (curation_dir / "labels_curated.tif").write_bytes(b"x")
    pairs_dir = _paths.patch_pairs_dir(working_dir, patch_stem)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    (pairs_dir / "spatial").mkdir()


# ---------------------------------------------------------------------------
# Empty / no-op cases
# ---------------------------------------------------------------------------


class TestEmptyState:

    def test_no_working_dir(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl.working_dir = ""
        summary = ctrl._enumerate_rereconstruct_targets(["patch_001.tif"])
        assert summary == {"per_patch": {}, "n_total_files": 0, "n_patches": 0}

    def test_no_artifacts_for_patch(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        summary = ctrl._enumerate_rereconstruct_targets(["patch_001.tif"])
        # All artifact-kinds report False for the patch.
        per_patch = summary["per_patch"]
        assert "patch_001" in per_patch
        kinds = per_patch["patch_001"]
        assert all(v is False for v in kinds.values())
        assert summary["n_total_files"] == 0
        assert summary["n_patches"] == 1

    def test_empty_patch_names_returns_empty_per_patch(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        summary = ctrl._enumerate_rereconstruct_targets([])
        assert summary["per_patch"] == {}
        assert summary["n_patches"] == 0


# ---------------------------------------------------------------------------
# Full + partial detection
# ---------------------------------------------------------------------------


class TestArtifactDetection:

    def test_all_kinds_present(self, tmp_path):
        _make_full_artifact_set(tmp_path, "patch_001.tif")
        ctrl = _make_controller(tmp_path)
        summary = ctrl._enumerate_rereconstruct_targets(["patch_001.tif"])
        kinds = summary["per_patch"]["patch_001"]
        assert kinds["labels_3d.tif"] is True
        assert kinds["qc_features.csv"] is True
        assert kinds["qc_features_raw.csv"] is True
        assert kinds["qc_adjacency_raw.csv"] is True
        assert kinds["qc_adjacency_curated.csv"] is True
        assert kinds["curation/"] is True
        assert kinds["refinement/pairs/"] is True
        # 7 artifacts total for this patch.
        assert summary["n_total_files"] == 7

    def test_partial_artifacts_only_flagged_existing(self, tmp_path):
        """Only labels_3d.tif + qc_features.csv exist for this patch.
        Everything else should be False."""
        _touch(
            _paths.patch_run_dir(tmp_path, "patch_001.tif")
            / "labels_3d.tif"
        )
        _touch(_paths.qc_features_csv(tmp_path, "patch_001"))
        ctrl = _make_controller(tmp_path)
        summary = ctrl._enumerate_rereconstruct_targets(["patch_001.tif"])
        kinds = summary["per_patch"]["patch_001"]
        assert kinds["labels_3d.tif"] is True
        assert kinds["qc_features.csv"] is True
        assert kinds["qc_features_raw.csv"] is False
        assert kinds["qc_adjacency_raw.csv"] is False
        assert kinds["qc_adjacency_curated.csv"] is False
        assert kinds["curation/"] is False
        assert kinds["refinement/pairs/"] is False
        assert summary["n_total_files"] == 2


# ---------------------------------------------------------------------------
# Multi-patch isolation
# ---------------------------------------------------------------------------


class TestMultiPatch:

    def test_patches_isolated(self, tmp_path):
        """One patch has full artifacts, another has none — verify
        no cross-contamination in the summary."""
        _make_full_artifact_set(tmp_path, "patch_full.tif")
        # patch_empty.tif has no on-disk artifacts.
        ctrl = _make_controller(tmp_path)
        summary = ctrl._enumerate_rereconstruct_targets([
            "patch_full.tif", "patch_empty.tif",
        ])
        assert summary["n_patches"] == 2
        # Each patch has its OWN summary.
        full_kinds = summary["per_patch"]["patch_full"]
        empty_kinds = summary["per_patch"]["patch_empty"]
        assert all(v for v in full_kinds.values())
        assert not any(v for v in empty_kinds.values())
        # Total file count = full patch's 7, empty's 0.
        assert summary["n_total_files"] == 7

    def test_unaffected_patches_excluded(self, tmp_path):
        """Even if patch_unaffected has full artifacts on disk, it
        doesn't appear in the summary because it's not in the
        affected-patches list."""
        _make_full_artifact_set(tmp_path, "patch_in_scope.tif")
        _make_full_artifact_set(tmp_path, "patch_unaffected.tif")
        ctrl = _make_controller(tmp_path)
        summary = ctrl._enumerate_rereconstruct_targets([
            "patch_in_scope.tif",
        ])
        assert summary["n_patches"] == 1
        assert "patch_in_scope" in summary["per_patch"]
        assert "patch_unaffected" not in summary["per_patch"]
        # Total file count only sums the in-scope patch.
        assert summary["n_total_files"] == 7

    def test_three_patches_partial_each(self, tmp_path):
        """Three patches with different subsets of artifacts —
        verify the file count aggregates correctly."""
        # Patch A: only labels_3d
        _touch(
            _paths.patch_run_dir(tmp_path, "A.tif") / "labels_3d.tif"
        )
        # Patch B: features + adjacency_raw (2 files)
        _touch(_paths.qc_features_csv(tmp_path, "B"))
        _touch(_paths.qc_adjacency_raw_csv(tmp_path, "B"))
        # Patch C: full set (7 files)
        _make_full_artifact_set(tmp_path, "C.tif")
        ctrl = _make_controller(tmp_path)
        summary = ctrl._enumerate_rereconstruct_targets([
            "A.tif", "B.tif", "C.tif",
        ])
        assert summary["n_patches"] == 3
        assert summary["n_total_files"] == 1 + 2 + 7


# ---------------------------------------------------------------------------
# Cross-step preservation — refined models, snapshots, training_archive
# ---------------------------------------------------------------------------


class TestPreservedArtifactsNotEnumerated:
    """The enumeration helper should NEVER include preserved artifacts
    (refined-model bundles, snapshots, training_archive) in its
    'will be deleted' summary, even though they live alongside the
    affected patches' artifacts in the workspace."""

    def test_refined_models_not_in_summary(self, tmp_path):
        _make_full_artifact_set(tmp_path, "patch_001.tif")
        # Create a refined-model bundle that should NOT appear.
        bundle = _paths.refined_model_dir(tmp_path, "my_model_v1")
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text("{}")
        (bundle / "lateral").write_bytes(b"weights")

        ctrl = _make_controller(tmp_path)
        summary = ctrl._enumerate_rereconstruct_targets(["patch_001.tif"])
        # Per-patch summary only enumerates DERIVED artifacts.
        # Refined-model bundles don't appear in the per-patch summary.
        kinds = summary["per_patch"]["patch_001"]
        assert "refinement/models/" not in kinds
        # Refined model file still on disk.
        assert (bundle / "lateral").exists()

    def test_snapshots_not_in_summary(self, tmp_path):
        _make_full_artifact_set(tmp_path, "patch_001.tif")
        # Create snapshot artifacts (preserved by design).
        snap = _paths.snapshots_dir(tmp_path)
        snap.mkdir(parents=True)
        (snap / "patch_001_pre_commit_20260101T000000Z.tif").write_bytes(b"s")

        ctrl = _make_controller(tmp_path)
        summary = ctrl._enumerate_rereconstruct_targets(["patch_001.tif"])
        kinds = summary["per_patch"]["patch_001"]
        assert "snapshots/" not in kinds
        # Snapshot still on disk.
        assert (snap / "patch_001_pre_commit_20260101T000000Z.tif").exists()

    def test_training_archive_not_in_summary(self, tmp_path):
        """``training_archive/`` is reserved for classifier training data
        and must NEVER be enumerated as a deletion target, even though
        it lives under ``artifacts/step3/`` alongside per-patch artifacts.
        """
        _make_full_artifact_set(tmp_path, "patch_001.tif")
        archive = _paths.patch_training_archive_dir(tmp_path, "patch_001")
        archive.mkdir(parents=True)
        (archive / "metadata.json").write_text("{}")

        ctrl = _make_controller(tmp_path)
        summary = ctrl._enumerate_rereconstruct_targets(["patch_001.tif"])
        kinds = summary["per_patch"]["patch_001"]
        assert "training_archive/" not in kinds
        # Archive still on disk.
        assert (archive / "metadata.json").exists()

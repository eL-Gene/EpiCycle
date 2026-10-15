"""Tests for the centralized Step 3 path constants module.

Phase 1 of the directory restructure (2026-05-20) ships
``napari_epicycle/fusion_reconstruction/_paths.py`` as a non-breaking
foundation.  Every path helper is a pure function — no side effects,
no filesystem mutation — so these tests are pure-Python with no Qt /
napari dependency.

Test categories:

1. Each helper returns a ``pathlib.Path``.
2. Each helper's returned path has the structurally-correct components.
3. Path stems (no extension) for image artifacts.
4. Layout detection across legacy / new / mixed / empty states.
5. Validators (bucket name).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from napari_epicycle.fusion_reconstruction import _paths as p


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:

    def test_state_filenames(self):
        assert p.STEP3_STATE_FILENAME == "step3_state.json"
        assert p.LEGACY_STATE_FILENAME == "reconstruction_state.json"

    def test_subdir_name(self):
        assert p.STEP3_ARTIFACTS_SUBDIR == "step3"

    def test_layout_version_ordering(self):
        assert p.LAYOUT_VERSION_UNKNOWN == 0
        assert p.LAYOUT_VERSION_LEGACY == 1
        assert p.LAYOUT_VERSION_NEW == 2
        assert p.LAYOUT_VERSION_MIXED == 3
        assert p.CURRENT_LAYOUT_VERSION == p.LAYOUT_VERSION_NEW


# ---------------------------------------------------------------------------
# Root + state JSON
# ---------------------------------------------------------------------------


class TestWorkspaceRoot:

    def test_workspace_root_returns_path(self, tmp_path):
        result = p.workspace_root(tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path

    def test_workspace_root_accepts_str(self, tmp_path):
        result = p.workspace_root(str(tmp_path))
        assert isinstance(result, Path)
        assert result == tmp_path

    def test_state_json_path(self, tmp_path):
        assert p.state_json_path(tmp_path) == tmp_path / "step3_state.json"

    def test_legacy_state_json_path(self, tmp_path):
        assert p.legacy_state_json_path(tmp_path) == (
            tmp_path / "reconstruction_state.json"
        )

    def test_step3_root(self, tmp_path):
        assert p.step3_root(tmp_path) == tmp_path / "artifacts" / "step3"

    def test_step3_artifacts_root_is_alias(self, tmp_path):
        assert p.step3_artifacts_root(tmp_path) == p.step3_root(tmp_path)


# ---------------------------------------------------------------------------
# Reconstruction runs
# ---------------------------------------------------------------------------


class TestRunsPaths:

    def test_runs_dir(self, tmp_path):
        assert p.runs_dir(tmp_path) == tmp_path / "artifacts" / "step3" / "runs"

    def test_patch_run_dir(self, tmp_path):
        result = p.patch_run_dir(tmp_path, "patch_001")
        assert result == (
            tmp_path / "artifacts" / "step3" / "runs" / "patch_001"
        )

    def test_patch_labels_stem_has_no_extension(self, tmp_path):
        result = p.patch_labels_stem(tmp_path, "patch_001")
        assert result.suffix == ""
        assert result.name == "labels_3d"
        assert result.parent == p.patch_run_dir(tmp_path, "patch_001")

    def test_patch_run_manifest_path(self, tmp_path):
        result = p.patch_run_manifest_path(tmp_path, "patch_001")
        assert result.name == "manifest.json"
        assert result.parent == p.patch_run_dir(tmp_path, "patch_001")


# ---------------------------------------------------------------------------
# QC + adjacency
# ---------------------------------------------------------------------------


class TestQcPaths:

    def test_qc_dir(self, tmp_path):
        assert p.qc_dir(tmp_path) == tmp_path / "artifacts" / "step3" / "qc"

    def test_qc_features_csv(self, tmp_path):
        assert p.qc_features_csv(tmp_path, "patch_001").name == (
            "patch_001_features.csv"
        )

    def test_qc_features_raw_csv(self, tmp_path):
        assert p.qc_features_raw_csv(tmp_path, "patch_001").name == (
            "patch_001_features_raw.csv"
        )

    def test_qc_adjacency_raw_csv(self, tmp_path):
        assert p.qc_adjacency_raw_csv(tmp_path, "patch_001").name == (
            "patch_001_adjacency_raw.csv"
        )

    def test_qc_adjacency_curated_csv(self, tmp_path):
        assert p.qc_adjacency_curated_csv(tmp_path, "patch_001").name == (
            "patch_001_adjacency_curated.csv"
        )

    def test_qc_files_share_parent(self, tmp_path):
        """All QC files for a patch live in the same qc_dir."""
        parent = p.qc_dir(tmp_path)
        for getter in (
            p.qc_features_csv, p.qc_features_raw_csv,
            p.qc_adjacency_raw_csv, p.qc_adjacency_curated_csv,
        ):
            assert getter(tmp_path, "patch_001").parent == parent


# ---------------------------------------------------------------------------
# Snapshots + prediction cache
# ---------------------------------------------------------------------------


class TestSnapshotsAndPredictionCache:

    def test_snapshots_dir(self, tmp_path):
        assert p.snapshots_dir(tmp_path) == (
            tmp_path / "artifacts" / "step3" / "snapshots"
        )

    def test_prediction_cache_dir(self, tmp_path):
        assert p.prediction_cache_dir(tmp_path) == (
            tmp_path / "artifacts" / "step3" / "prediction_cache"
        )


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------


class TestCurationPaths:

    def test_curation_root(self, tmp_path):
        assert p.curation_root(tmp_path) == (
            tmp_path / "artifacts" / "step3" / "curation"
        )

    def test_patch_curation_dir(self, tmp_path):
        assert p.patch_curation_dir(tmp_path, "patch_001") == (
            p.curation_root(tmp_path) / "patch_001"
        )

    def test_curation_labels_stem_no_extension(self, tmp_path):
        result = p.patch_curation_labels_stem(tmp_path, "patch_001")
        assert result.suffix == ""
        assert result.name == "labels_curated"

    def test_curation_labels_bak_stem(self, tmp_path):
        result = p.patch_curation_labels_bak_stem(tmp_path, "patch_001")
        assert result.name == "labels_curated.bak"

    def test_curation_tiers_csv(self, tmp_path):
        assert p.patch_curation_tiers_csv(tmp_path, "patch_001").name == (
            "tiers.csv"
        )


# ---------------------------------------------------------------------------
# Refinement: training pairs
# ---------------------------------------------------------------------------


class TestRefinementPairs:

    def test_refinement_root(self, tmp_path):
        assert p.refinement_root(tmp_path) == (
            tmp_path / "artifacts" / "step3" / "refinement"
        )

    def test_refinement_manifest_path(self, tmp_path):
        result = p.refinement_manifest_path(tmp_path)
        assert result.name == "manifest.json"
        assert result.parent == p.refinement_root(tmp_path)

    def test_refinement_pairs_root(self, tmp_path):
        assert p.refinement_pairs_root(tmp_path) == (
            p.refinement_root(tmp_path) / "pairs"
        )

    def test_patch_pairs_dir(self, tmp_path):
        assert p.patch_pairs_dir(tmp_path, "patch_001") == (
            p.refinement_pairs_root(tmp_path) / "patch_001"
        )

    def test_patch_pairs_spatial_dir(self, tmp_path):
        assert p.patch_pairs_spatial_dir(tmp_path, "patch_001").name == (
            "spatial"
        )
        assert p.patch_pairs_spatial_dir(tmp_path, "patch_001").parent == (
            p.patch_pairs_dir(tmp_path, "patch_001")
        )

    def test_patch_pairs_axial_dir(self, tmp_path):
        assert p.patch_pairs_axial_dir(tmp_path, "patch_001").name == "axial"


# ---------------------------------------------------------------------------
# Refinement: model bundles
# ---------------------------------------------------------------------------


class TestRefinedModelBundles:

    def test_refined_models_root(self, tmp_path):
        assert p.refined_models_root(tmp_path) == (
            p.refinement_root(tmp_path) / "models"
        )

    def test_refined_model_dir(self, tmp_path):
        assert p.refined_model_dir(tmp_path, "test_v1") == (
            p.refined_models_root(tmp_path) / "test_v1"
        )

    def test_refined_model_manifest_path(self, tmp_path):
        result = p.refined_model_manifest_path(tmp_path, "test_v1")
        assert result.name == "manifest.json"
        assert result.parent == p.refined_model_dir(tmp_path, "test_v1")

    def test_refined_model_weights_path_lateral(self, tmp_path):
        """Lifted layout: weights file is at bundle/{bucket} (post-lift)."""
        result = p.refined_model_weights_path(tmp_path, "test_v1", "lateral")
        assert result.name == "lateral"
        assert result.parent == p.refined_model_dir(tmp_path, "test_v1")

    def test_refined_model_weights_path_axial(self, tmp_path):
        result = p.refined_model_weights_path(tmp_path, "test_v1", "axial")
        assert result.name == "axial"

    def test_refined_model_weights_path_invalid_bucket(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid bucket"):
            p.refined_model_weights_path(tmp_path, "test_v1", "diagonal")

    def test_refined_model_train_loss_path(self, tmp_path):
        result = p.refined_model_train_loss_path(tmp_path, "test_v1", "lateral")
        assert result.name == "lateral_train_loss.npy"

    def test_refined_model_val_loss_path(self, tmp_path):
        result = p.refined_model_val_loss_path(tmp_path, "test_v1", "axial")
        assert result.name == "axial_val_loss.npy"

    def test_refined_model_training_config_path(self, tmp_path):
        result = p.refined_model_training_config_path(
            tmp_path, "test_v1", "lateral",
        )
        assert result.name == "lateral_training_config.json"

    def test_refined_model_cellpose_intermediate_path(self, tmp_path):
        """Pre-lift path: under bundle/models/{bucket} as Cellpose writes it."""
        result = p.refined_model_cellpose_intermediate_path(
            tmp_path, "test_v1", "lateral",
        )
        assert result == (
            p.refined_model_dir(tmp_path, "test_v1") / "models" / "lateral"
        )

    def test_invalid_bucket_raises(self, tmp_path):
        for getter in (
            p.refined_model_weights_path,
            p.refined_model_train_loss_path,
            p.refined_model_val_loss_path,
            p.refined_model_training_config_path,
            p.refined_model_cellpose_intermediate_path,
        ):
            with pytest.raises(ValueError):
                getter(tmp_path, "test_v1", "BOGUS")


# ---------------------------------------------------------------------------
# Training archive
# ---------------------------------------------------------------------------


class TestTrainingArchive:

    def test_training_archive_root(self, tmp_path):
        assert p.training_archive_root(tmp_path) == (
            tmp_path / "artifacts" / "step3" / "training_archive"
        )

    def test_patch_training_archive_dir(self, tmp_path):
        assert p.patch_training_archive_dir(tmp_path, "patch_001") == (
            p.training_archive_root(tmp_path) / "patch_001"
        )

    def test_training_archive_stems_no_extension(self, tmp_path):
        for getter in (
            p.training_archive_raw_labels_stem,
            p.training_archive_curated_labels_stem,
            p.training_archive_source_stack_stem,
        ):
            assert getter(tmp_path, "patch_001").suffix == ""

    def test_training_archive_metadata_path(self, tmp_path):
        result = p.training_archive_metadata_path(tmp_path, "patch_001")
        assert result.name == "metadata.json"


# ---------------------------------------------------------------------------
# Legacy path constants
# ---------------------------------------------------------------------------


class TestLegacyPaths:

    def test_legacy_artifacts_root(self, tmp_path):
        assert p.legacy_artifacts_root(tmp_path) == tmp_path / "artifacts"

    def test_legacy_reconstruction_root(self, tmp_path):
        assert p.legacy_reconstruction_root(tmp_path) == (
            tmp_path / "artifacts" / "reconstruction"
        )

    def test_legacy_qc_dir(self, tmp_path):
        assert p.legacy_qc_dir(tmp_path) == tmp_path / "artifacts" / "qc"

    def test_legacy_snapshots_dir(self, tmp_path):
        assert p.legacy_snapshots_dir(tmp_path) == (
            tmp_path / "artifacts" / "snapshots"
        )

    def test_legacy_prediction_cache_dir(self, tmp_path):
        assert p.legacy_prediction_cache_dir(tmp_path) == (
            tmp_path / "artifacts" / "prediction_cache"
        )

    def test_legacy_refinement_training_data(self, tmp_path):
        assert p.legacy_refinement_training_data(tmp_path) == (
            tmp_path / "artifacts" / "refinement_training_data"
        )

    def test_legacy_models_root(self, tmp_path):
        assert p.legacy_models_root(tmp_path) == tmp_path / "models"

    def test_legacy_refinement_models_root(self, tmp_path):
        assert p.legacy_refinement_models_root(tmp_path) == (
            tmp_path / "models" / "refinement"
        )


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------


class TestLayoutDetection:

    def test_empty_workspace_returns_unknown(self, tmp_path):
        assert p.detect_layout_version(tmp_path) == p.LAYOUT_VERSION_UNKNOWN
        assert not p.is_legacy_layout(tmp_path)
        assert not p.is_new_layout(tmp_path)
        assert not p.is_mixed_layout(tmp_path)

    def test_legacy_workspace_reconstruction_state(self, tmp_path):
        """Workspace with only reconstruction_state.json → legacy."""
        (tmp_path / "reconstruction_state.json").write_text("{}")
        assert p.detect_layout_version(tmp_path) == p.LAYOUT_VERSION_LEGACY
        assert p.is_legacy_layout(tmp_path)
        assert not p.is_new_layout(tmp_path)
        assert not p.is_mixed_layout(tmp_path)

    def test_legacy_workspace_reconstruction_dir(self, tmp_path):
        """Workspace with only artifacts/reconstruction/ → legacy."""
        (tmp_path / "artifacts" / "reconstruction").mkdir(parents=True)
        assert p.detect_layout_version(tmp_path) == p.LAYOUT_VERSION_LEGACY

    def test_legacy_workspace_refinement_training_data(self, tmp_path):
        """Workspace with only refinement_training_data/ → legacy."""
        (tmp_path / "artifacts" / "refinement_training_data").mkdir(parents=True)
        assert p.detect_layout_version(tmp_path) == p.LAYOUT_VERSION_LEGACY

    def test_legacy_workspace_refinement_models(self, tmp_path):
        """Workspace with only models/refinement/ → legacy."""
        (tmp_path / "models" / "refinement").mkdir(parents=True)
        assert p.detect_layout_version(tmp_path) == p.LAYOUT_VERSION_LEGACY

    def test_new_workspace_state_json(self, tmp_path):
        """Workspace with only step3_state.json → new."""
        (tmp_path / "step3_state.json").write_text("{}")
        assert p.detect_layout_version(tmp_path) == p.LAYOUT_VERSION_NEW
        assert p.is_new_layout(tmp_path)
        assert not p.is_legacy_layout(tmp_path)
        assert not p.is_mixed_layout(tmp_path)

    def test_new_workspace_step3_dir(self, tmp_path):
        """Workspace with only artifacts/step3/ → new."""
        (tmp_path / "artifacts" / "step3").mkdir(parents=True)
        assert p.detect_layout_version(tmp_path) == p.LAYOUT_VERSION_NEW

    def test_mixed_workspace(self, tmp_path):
        """Workspace with BOTH legacy AND new markers → MIXED.

        Previously detect_layout_version used NEW as the tie-breaker
        which caused predicate disagreement (detect=NEW but
        is_new_layout=False).  Now MIXED is a distinct value and the
        predicates agree.
        """
        (tmp_path / "reconstruction_state.json").write_text("{}")
        (tmp_path / "step3_state.json").write_text("{}")
        assert p.detect_layout_version(tmp_path) == p.LAYOUT_VERSION_MIXED
        assert p.is_mixed_layout(tmp_path)
        assert not p.is_new_layout(tmp_path)
        assert not p.is_legacy_layout(tmp_path)

    def test_models_subdir_alone_does_not_imply_legacy(self, tmp_path):
        """A workspace with just `models/` but no `models/refinement/`
        should NOT be classified legacy — that's Step 2 HITL output."""
        (tmp_path / "models" / "round_001").mkdir(parents=True)
        # No refinement/, no reconstruction/, no state JSON.
        assert p.detect_layout_version(tmp_path) == p.LAYOUT_VERSION_UNKNOWN


# ---------------------------------------------------------------------------
# Bucket validator
# ---------------------------------------------------------------------------


class TestBucketValidator:

    def test_lateral_accepted(self, tmp_path):
        # Should not raise.
        p.refined_model_weights_path(tmp_path, "test", "lateral")

    def test_axial_accepted(self, tmp_path):
        p.refined_model_weights_path(tmp_path, "test", "axial")

    @pytest.mark.parametrize("invalid", [
        "Lateral",   # case-sensitive
        "Axial",
        "xy",
        "xz",
        "yz",
        "spatial",
        "",
        "both",
    ])
    def test_invalid_bucket_names_rejected(self, tmp_path, invalid):
        with pytest.raises(ValueError):
            p.refined_model_weights_path(tmp_path, "test", invalid)


# ---------------------------------------------------------------------------
# Cross-helper invariants
# ---------------------------------------------------------------------------


class TestPathInvariants:
    """Round-trip / cross-helper consistency checks."""

    def test_all_step3_paths_under_step3_root(self, tmp_path):
        """Every new-layout helper must produce a path under step3/."""
        s3 = p.step3_root(tmp_path)
        cases = [
            p.runs_dir(tmp_path),
            p.patch_run_dir(tmp_path, "patch_001"),
            p.patch_labels_stem(tmp_path, "patch_001"),
            p.qc_dir(tmp_path),
            p.qc_features_csv(tmp_path, "patch_001"),
            p.snapshots_dir(tmp_path),
            p.prediction_cache_dir(tmp_path),
            p.curation_root(tmp_path),
            p.patch_curation_dir(tmp_path, "patch_001"),
            p.refinement_root(tmp_path),
            p.refinement_pairs_root(tmp_path),
            p.refined_models_root(tmp_path),
            p.training_archive_root(tmp_path),
        ]
        for path in cases:
            assert s3 in path.parents or path == s3, (
                f"{path} not under step3_root {s3}"
            )

    def test_legacy_paths_never_under_step3_subdir(self, tmp_path):
        """Legacy helpers must NEVER produce paths under artifacts/step3/."""
        s3 = p.step3_root(tmp_path)
        cases = [
            p.legacy_reconstruction_root(tmp_path),
            p.legacy_qc_dir(tmp_path),
            p.legacy_snapshots_dir(tmp_path),
            p.legacy_prediction_cache_dir(tmp_path),
            p.legacy_refinement_training_data(tmp_path),
            p.legacy_refinement_models_root(tmp_path),
        ]
        for path in cases:
            assert s3 not in path.parents and path != s3, (
                f"Legacy path {path} should not be under step3 root"
            )

    def test_patch_name_preserved_in_paths(self, tmp_path):
        """Patch name appears as a literal path component."""
        name = "patch_with_underscores_42"
        assert p.patch_run_dir(tmp_path, name).name == name
        assert p.patch_curation_dir(tmp_path, name).name == name
        assert p.patch_pairs_dir(tmp_path, name).name == name
        assert p.patch_training_archive_dir(tmp_path, name).name == name

    def test_model_name_preserved_in_paths(self, tmp_path):
        """Model name appears as a literal path component."""
        name = "tuned_v3"
        assert p.refined_model_dir(tmp_path, name).name == name


# ---------------------------------------------------------------------------
# Multi-run run_id parameterization (Phase 0 — additive, non-breaking)
# ---------------------------------------------------------------------------


class TestReconstructionsDir:
    """The run_id-aware successor to runs_dir."""

    def test_none_is_byte_identical_to_runs_dir(self, tmp_path):
        assert p.reconstructions_dir(tmp_path, None) == p.runs_dir(tmp_path)
        assert p.reconstructions_dir(tmp_path) == p.runs_dir(tmp_path)
        # literal "runs" segment under step3_root
        assert p.reconstructions_dir(tmp_path) == (
            tmp_path / "artifacts" / "step3" / "runs"
        )

    def test_run_id_nests_under_run_context(self, tmp_path):
        assert p.reconstructions_dir(tmp_path, "cells") == (
            tmp_path / "artifacts" / "step3" / "runs" / "cells" / "reconstructions"
        )


class TestRunIdParameterization:
    """Every per-run helper: run_id=None is byte-identical to the flat
    layout; a run slug nests under runs/{slug}/..."""

    # (helper, *extra_args) — extra_args are the positional args BEFORE run_id
    PER_RUN_CASES = [
        ("qc_dir", ()),
        ("qc_features_csv", ("patch_001",)),
        ("qc_features_raw_csv", ("patch_001",)),
        ("qc_adjacency_raw_csv", ("patch_001",)),
        ("qc_adjacency_curated_csv", ("patch_001",)),
        ("prediction_cache_dir", ()),
        ("curation_root", ()),
        ("patch_curation_dir", ("patch_001",)),
        ("patch_curation_labels_stem", ("patch_001",)),
        ("patch_curation_labels_bak_stem", ("patch_001",)),
        ("patch_curation_tiers_csv", ("patch_001",)),
        ("refinement_root", ()),
        ("refinement_manifest_path", ()),
        ("refinement_pairs_root", ()),
        ("patch_pairs_dir", ("patch_001",)),
        ("patch_pairs_spatial_dir", ("patch_001",)),
        ("patch_pairs_axial_dir", ("patch_001",)),
        ("refined_models_root", ()),
        ("refined_model_dir", ("model_v1",)),
        ("refined_model_manifest_path", ("model_v1",)),
        ("patch_run_dir", ("patch_001",)),
        ("patch_labels_stem", ("patch_001",)),
        ("patch_run_manifest_path", ("patch_001",)),
    ]

    # Refined-model helpers that take a (model_name, bucket) signature.
    BUCKET_CASES = [
        ("refined_model_weights_path", ("model_v1", "lateral")),
        ("refined_model_train_loss_path", ("model_v1", "lateral")),
        ("refined_model_val_loss_path", ("model_v1", "axial")),
        ("refined_model_training_config_path", ("model_v1", "lateral")),
        ("refined_model_cellpose_intermediate_path", ("model_v1", "axial")),
    ]

    # LITERAL expected flat (run_id=None) path components, relative to the
    # workspace root.  Comparing against these (rather than merely
    # ``helper(ws) == helper(ws, run_id=None)``, which is tautological)
    # actually pins each helper's flat layout.  (Audit LOW.)
    EXPECTED_FLAT_PARTS = {
        "qc_dir": ("artifacts", "step3", "qc"),
        "qc_features_csv": ("artifacts", "step3", "qc", "patch_001_features.csv"),
        "qc_features_raw_csv": ("artifacts", "step3", "qc", "patch_001_features_raw.csv"),
        "qc_adjacency_raw_csv": ("artifacts", "step3", "qc", "patch_001_adjacency_raw.csv"),
        "qc_adjacency_curated_csv": ("artifacts", "step3", "qc", "patch_001_adjacency_curated.csv"),
        "prediction_cache_dir": ("artifacts", "step3", "prediction_cache"),
        "curation_root": ("artifacts", "step3", "curation"),
        "patch_curation_dir": ("artifacts", "step3", "curation", "patch_001"),
        "patch_curation_labels_stem": ("artifacts", "step3", "curation", "patch_001", "labels_curated"),
        "patch_curation_labels_bak_stem": ("artifacts", "step3", "curation", "patch_001", "labels_curated.bak"),
        "patch_curation_tiers_csv": ("artifacts", "step3", "curation", "patch_001", "tiers.csv"),
        "refinement_root": ("artifacts", "step3", "refinement"),
        "refinement_manifest_path": ("artifacts", "step3", "refinement", "manifest.json"),
        "refinement_pairs_root": ("artifacts", "step3", "refinement", "pairs"),
        "patch_pairs_dir": ("artifacts", "step3", "refinement", "pairs", "patch_001"),
        "patch_pairs_spatial_dir": ("artifacts", "step3", "refinement", "pairs", "patch_001", "spatial"),
        "patch_pairs_axial_dir": ("artifacts", "step3", "refinement", "pairs", "patch_001", "axial"),
        "refined_models_root": ("artifacts", "step3", "refinement", "models"),
        "refined_model_dir": ("artifacts", "step3", "refinement", "models", "model_v1"),
        "refined_model_manifest_path": ("artifacts", "step3", "refinement", "models", "model_v1", "manifest.json"),
        "patch_run_dir": ("artifacts", "step3", "runs", "patch_001"),
        "patch_labels_stem": ("artifacts", "step3", "runs", "patch_001", "labels_3d"),
        "patch_run_manifest_path": ("artifacts", "step3", "runs", "patch_001", "manifest.json"),
    }

    EXPECTED_BUCKET_PARTS = {
        "refined_model_weights_path": ("artifacts", "step3", "refinement", "models", "model_v1", "lateral"),
        "refined_model_train_loss_path": ("artifacts", "step3", "refinement", "models", "model_v1", "lateral_train_loss.npy"),
        "refined_model_val_loss_path": ("artifacts", "step3", "refinement", "models", "model_v1", "axial_val_loss.npy"),
        "refined_model_training_config_path": ("artifacts", "step3", "refinement", "models", "model_v1", "lateral_training_config.json"),
        "refined_model_cellpose_intermediate_path": ("artifacts", "step3", "refinement", "models", "model_v1", "models", "axial"),
    }

    @pytest.mark.parametrize("name,extra", PER_RUN_CASES)
    def test_none_is_byte_identical_to_default(self, tmp_path, name, extra):
        """helper(ws, *extra, run_id=None) == the LITERAL flat path."""
        fn = getattr(p, name)
        flat_default = fn(tmp_path, *extra)
        flat_explicit_none = fn(tmp_path, *extra, run_id=None)
        assert flat_default == flat_explicit_none
        # Compare to the LITERAL expected flat path (not a tautology).
        expected = tmp_path.joinpath(*self.EXPECTED_FLAT_PARTS[name])
        assert flat_default == expected, (
            f"{name}(run_id=None) = {flat_default}, expected {expected}"
        )
        # The flat path must be under step3_root and NOT under runs/<slug>/.
        s3 = p.step3_root(tmp_path)
        assert s3 in flat_default.parents or flat_default == s3
        assert "runs" not in flat_default.relative_to(s3).parts[:1] or (
            # runs/ is allowed ONLY for the reconstruction helpers
            name in ("patch_run_dir", "patch_labels_stem", "patch_run_manifest_path")
        )

    @pytest.mark.parametrize("name,extra", PER_RUN_CASES)
    def test_run_id_nests_under_run_slug(self, tmp_path, name, extra):
        fn = getattr(p, name)
        nested = fn(tmp_path, *extra, run_id="cells")
        run_ctx = tmp_path / "artifacts" / "step3" / "runs" / "cells"
        assert run_ctx in nested.parents, (
            f"{name}(run_id='cells') = {nested} not under {run_ctx}"
        )

    @pytest.mark.parametrize("name,args", BUCKET_CASES)
    def test_bucket_helpers_none_is_byte_identical(self, tmp_path, name, args):
        fn = getattr(p, name)
        assert fn(tmp_path, *args) == fn(tmp_path, *args, run_id=None)
        # Compare to the LITERAL expected flat path (not a tautology).
        expected = tmp_path.joinpath(*self.EXPECTED_BUCKET_PARTS[name])
        assert fn(tmp_path, *args) == expected, (
            f"{name}(run_id=None) = {fn(tmp_path, *args)}, expected {expected}"
        )

    @pytest.mark.parametrize("name,args", BUCKET_CASES)
    def test_bucket_helpers_run_id_nests(self, tmp_path, name, args):
        fn = getattr(p, name)
        nested = fn(tmp_path, *args, run_id="cells")
        run_ctx = tmp_path / "artifacts" / "step3" / "runs" / "cells"
        assert run_ctx in nested.parents

    def test_patch_run_dir_run_id_routes_through_reconstructions(self, tmp_path):
        """run_id != None routes patch_run_dir through reconstructions_dir."""
        nested = p.patch_run_dir(tmp_path, "patch_001", run_id="cells")
        assert nested == (
            tmp_path / "artifacts" / "step3" / "runs" / "cells"
            / "reconstructions" / "patch_001"
        )

    def test_qc_dir_run_id(self, tmp_path):
        assert p.qc_dir(tmp_path, run_id="cells") == (
            tmp_path / "artifacts" / "step3" / "runs" / "cells" / "qc"
        )

    def test_refinement_root_run_id(self, tmp_path):
        assert p.refinement_root(tmp_path, run_id="nuclei") == (
            tmp_path / "artifacts" / "step3" / "runs" / "nuclei" / "refinement"
        )


# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------


class TestPublicAPI:

    def test_all_exports_are_defined(self):
        """Every name in __all__ must actually exist in the module."""
        for name in p.__all__:
            assert hasattr(p, name), (
                f"_paths.__all__ lists {name!r} but it doesn't exist"
            )

    def test_all_helpers_callable_or_constant(self):
        """Every export is either a callable or a primitive constant."""
        for name in p.__all__:
            obj = getattr(p, name)
            assert callable(obj) or isinstance(obj, (int, str, bytes)), (
                f"{name} is neither callable nor a primitive constant"
            )


# ---------------------------------------------------------------------------
# Run-context helpers (named runs) + validate_run_id  (Phase 2)
# ---------------------------------------------------------------------------


class TestRunContextHelpers:

    def test_run_id_max_length_constant(self):
        assert p.RUN_ID_MAX_LENGTH == 64

    def test_run_manifest_filename_constant(self):
        assert p.RUN_MANIFEST_FILENAME == "run_manifest.json"

    def test_step3_runs_root_is_runs_dir_alias(self, tmp_path):
        assert p.step3_runs_root(tmp_path) == p.runs_dir(tmp_path)
        assert p.step3_runs_root(tmp_path) == (
            tmp_path / "artifacts" / "step3" / "runs"
        )

    def test_run_root(self, tmp_path):
        assert p.run_root(tmp_path, "cells") == (
            tmp_path / "artifacts" / "step3" / "runs" / "cells"
        )

    def test_run_manifest_path(self, tmp_path):
        result = p.run_manifest_path(tmp_path, "cells")
        assert result == p.run_root(tmp_path, "cells") / "run_manifest.json"
        assert result.name == "run_manifest.json"

    def test_run_root_accepts_str(self, tmp_path):
        from pathlib import Path
        assert p.run_root(str(tmp_path), "cells") == Path(
            str(tmp_path)
        ) / "artifacts" / "step3" / "runs" / "cells"


class TestValidateRunId:

    def test_accepts_simple(self):
        for ok in ("cells", "nuclei", "run_01", "Run-2", "a", "X9"):
            p.validate_run_id(ok)  # no raise

    def test_accepts_max_length(self):
        p.validate_run_id("a" * 64)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            p.validate_run_id("")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            p.validate_run_id("a" * 65)

    def test_rejects_leading_underscore_or_dash(self):
        for bad in ("_run", "-run"):
            with pytest.raises(ValueError):
                p.validate_run_id(bad)

    def test_rejects_path_traversal(self):
        for bad in ("../evil", "a/b", "a\b", "..", "."):
            with pytest.raises(ValueError):
                p.validate_run_id(bad)

    def test_rejects_special_chars(self):
        for bad in ("a b", "a.b", "a*b", "a@b"):
            with pytest.raises(ValueError):
                p.validate_run_id(bad)

    def test_rejects_non_str(self):
        with pytest.raises(ValueError):
            p.validate_run_id(123)  # type: ignore[arg-type]

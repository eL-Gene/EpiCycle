"""Tests for the Phase 6 post-train lift + schema 2.0 manifest writer.

Closes the original wrong-path bug: Cellpose's auto-inserted
``models/`` subdir gets lifted to the bundle root post-train, and
a backend-agnostic schema 2.0 manifest is written.

Pure-Python, no Qt / napari / Cellpose dependency — simulates
Cellpose's post-train state by creating the files Cellpose would
have written and then exercises the lift + manifest helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from napari_epicycle.fusion_reconstruction._refinement_bundle import (
    SCHEMA_VERSION,
    VALID_BUCKETS,
    finalize_refinement_pass,
    lift_cellpose_output,
    write_or_merge_bundle_manifest,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _simulate_cellpose_output(bundle_dir: Path, bucket: str,
                              *, with_companions: bool = True,
                              weights_content: bytes = b"fake-weights"):
    """Build the on-disk state Cellpose would have left post-train.

    Cellpose writes to ``save_path/models/{model_name}``.  If we pass
    Cellpose ``save_path = bundle_dir`` and ``model_name = bucket``,
    the post-train state is::

        bundle_dir/models/{bucket}                  ← weights file
        bundle_dir/{bucket}_train_loss.npy          ← companion (at root)
        bundle_dir/{bucket}_val_loss.npy            ← companion (at root)
        bundle_dir/{bucket}_training_config.json    ← companion (at root)

    NB: companions are at the BUNDLE ROOT in current Cellpose
    versions (verified by reading the trainer code).  The lift's
    "stray companion" sweep handles the case where a future Cellpose
    version might drop them in ``models/`` instead.
    """
    cellpose_subdir = bundle_dir / "models"
    cellpose_subdir.mkdir(parents=True, exist_ok=True)
    (cellpose_subdir / bucket).write_bytes(weights_content)
    if with_companions:
        (bundle_dir / f"{bucket}_train_loss.npy").write_bytes(b"\x00" * 8)
        (bundle_dir / f"{bucket}_val_loss.npy").write_bytes(b"\x00" * 8)
        (bundle_dir / f"{bucket}_training_config.json").write_text(
            json.dumps({"n_epochs": 10}),
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:

    def test_schema_version(self):
        assert SCHEMA_VERSION == "2.0"

    def test_valid_buckets(self):
        assert "lateral" in VALID_BUCKETS
        assert "axial" in VALID_BUCKETS
        assert len(VALID_BUCKETS) == 2


# ---------------------------------------------------------------------------
# lift_cellpose_output
# ---------------------------------------------------------------------------


class TestLiftCellposeOutput:

    def test_lifts_main_weights_to_bundle_root(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        _simulate_cellpose_output(bundle, "lateral", with_companions=False)
        result = lift_cellpose_output(bundle, "lateral", log_fn=lambda _: None)
        assert result["weights_lifted"] is True
        assert (bundle / "lateral").is_file()
        assert not (bundle / "models" / "lateral").exists()

    def test_removes_empty_models_subdir(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        _simulate_cellpose_output(bundle, "lateral", with_companions=False)
        result = lift_cellpose_output(bundle, "lateral", log_fn=lambda _: None)
        assert result["models_subdir_removed"] is True
        assert not (bundle / "models").exists()

    def test_lifts_stray_companions_inside_models_subdir(self, tmp_path):
        """Future Cellpose versions might drop companions inside
        models/ instead of at bundle root — lift should sweep them up."""
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        cellpose_subdir = bundle / "models"
        cellpose_subdir.mkdir()
        (cellpose_subdir / "lateral").write_bytes(b"weights")
        # Stray companion INSIDE models/.
        (cellpose_subdir / "stray_extra.json").write_text("{}")
        result = lift_cellpose_output(bundle, "lateral", log_fn=lambda _: None)
        assert result["weights_lifted"] is True
        # Stray got lifted too.
        assert (bundle / "stray_extra.json").is_file()
        assert not (bundle / "models").exists()

    def test_invalid_bucket_raises(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        with pytest.raises(ValueError, match="Invalid bucket"):
            lift_cellpose_output(bundle, "spatial", log_fn=lambda _: None)

    def test_missing_bundle_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            lift_cellpose_output(
                tmp_path / "does_not_exist", "lateral",
                log_fn=lambda _: None,
            )

    def test_idempotent_when_already_lifted(self, tmp_path):
        """Re-running lift on an already-flat bundle is a clean no-op."""
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        _simulate_cellpose_output(bundle, "lateral", with_companions=False)
        lift_cellpose_output(bundle, "lateral", log_fn=lambda _: None)
        # Re-run.
        result = lift_cellpose_output(bundle, "lateral", log_fn=lambda _: None)
        assert result["weights_lifted"] is True  # post-condition holds
        assert result["skipped_reason"] == "already_lifted"

    def test_refuses_to_overwrite_existing_dst(self, tmp_path):
        """If both src AND dst exist, don't clobber."""
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        # Pre-existing weights at bundle root (e.g. user's manual stash).
        (bundle / "lateral").write_bytes(b"PRE_EXISTING")
        # Cellpose also wrote (unusual, but defensive).
        cellpose_subdir = bundle / "models"
        cellpose_subdir.mkdir()
        (cellpose_subdir / "lateral").write_bytes(b"CELLPOSE_NEW")
        result = lift_cellpose_output(bundle, "lateral", log_fn=lambda _: None)
        assert result["weights_lifted"] is False
        assert result["skipped_reason"] == "dst_exists"
        # Pre-existing kept; cellpose-side untouched for manual resolution.
        assert (bundle / "lateral").read_bytes() == b"PRE_EXISTING"
        assert (bundle / "models" / "lateral").read_bytes() == b"CELLPOSE_NEW"

    def test_companions_already_at_root_are_unchanged(self, tmp_path):
        """If Cellpose put companions at bundle root (current behavior),
        they're unchanged by the lift — only models/<bucket> moves."""
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        _simulate_cellpose_output(bundle, "lateral")
        train_loss = bundle / "lateral_train_loss.npy"
        original_size = train_loss.stat().st_size
        lift_cellpose_output(bundle, "lateral", log_fn=lambda _: None)
        # Companion still at bundle root.
        assert train_loss.is_file()
        assert train_loss.stat().st_size == original_size

    def test_no_weights_no_lift_no_crash(self, tmp_path):
        """Training failed before save → empty bundle.  Lift should
        return without raising, with src_missing reason."""
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        (bundle / "models").mkdir()  # empty models/ subdir
        result = lift_cellpose_output(bundle, "lateral", log_fn=lambda _: None)
        assert result["weights_lifted"] is False
        assert result["skipped_reason"] == "src_missing"


# ---------------------------------------------------------------------------
# write_or_merge_bundle_manifest
# ---------------------------------------------------------------------------


class TestWriteOrMergeBundleManifest:

    def test_writes_basic_manifest(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        manifest = write_or_merge_bundle_manifest(
            bundle, "my_model", "lateral",
            final_train_loss=0.5,
            n_train_pairs=100,
            n_epochs=10,
        )
        assert manifest["schema_version"] == "2.0"
        assert manifest["name"] == "my_model"
        assert manifest["lateral"]["backend"] == "cellpose"
        assert manifest["lateral"]["weights_path"] == "lateral"
        assert manifest["lateral"]["final_loss"]["train"] == 0.5

    def test_writes_to_disk_atomically(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        write_or_merge_bundle_manifest(bundle, "my_model", "lateral")
        path = bundle / "manifest.json"
        assert path.is_file()
        # No leftover *.tmp files.
        leftover = list(bundle.glob("*.tmp"))
        assert leftover == []

    def test_second_pass_merges_bucket_in_place(self, tmp_path):
        """Mode=Both: first lateral pass writes lateral entry, second
        axial pass adds axial WITHOUT clobbering lateral."""
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        # First pass (lateral).
        write_or_merge_bundle_manifest(
            bundle, "my_model", "lateral",
            final_train_loss=0.7,
        )
        # Second pass (axial).
        manifest = write_or_merge_bundle_manifest(
            bundle, "my_model", "axial",
            final_train_loss=0.3,
        )
        # Both buckets present.
        assert "lateral" in manifest
        assert "axial" in manifest
        # Lateral preserved from first write.
        assert manifest["lateral"]["final_loss"]["train"] == 0.7
        # Axial fresh from second write.
        assert manifest["axial"]["final_loss"]["train"] == 0.3

    def test_preserves_created_at_across_merge(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        m1 = write_or_merge_bundle_manifest(bundle, "my_model", "lateral")
        first_created = m1["created_at"]
        m2 = write_or_merge_bundle_manifest(bundle, "my_model", "axial")
        assert m2["created_at"] == first_created
        # last_updated should be set (may equal first_created if same second).
        assert "last_updated" in m2

    def test_re_writing_same_bucket_replaces_entry(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        write_or_merge_bundle_manifest(
            bundle, "my_model", "lateral",
            final_train_loss=0.7,
        )
        manifest = write_or_merge_bundle_manifest(
            bundle, "my_model", "lateral",
            final_train_loss=0.4,
        )
        assert manifest["lateral"]["final_loss"]["train"] == 0.4

    def test_invalid_bucket_raises(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        with pytest.raises(ValueError, match="Invalid bucket"):
            write_or_merge_bundle_manifest(bundle, "my_model", "BOGUS")

    def test_init_section_recorded(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        manifest = write_or_merge_bundle_manifest(
            bundle, "my_model", "lateral",
            init_mode="custom", init_path="/abs/path/to/init.pth",
            init_combo_text="Use refined: my_prev_model / lateral",
        )
        init = manifest["lateral"]["init"]
        assert init["mode"] == "custom"
        assert init["path"] == "/abs/path/to/init.pth"
        assert init["init_combo_text"] == "Use refined: my_prev_model / lateral"

    def test_corrupt_existing_manifest_does_not_block_write(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        (bundle / "manifest.json").write_text("NOT VALID JSON")
        manifest = write_or_merge_bundle_manifest(
            bundle, "my_model", "lateral",
        )
        # Wrote a fresh manifest (no merge from corrupt content).
        assert manifest["schema_version"] == "2.0"
        assert "lateral" in manifest


# ---------------------------------------------------------------------------
# finalize_refinement_pass (combined)
# ---------------------------------------------------------------------------


class TestFinalizeRefinementPass:

    def test_combined_lift_and_manifest_write(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        _simulate_cellpose_output(bundle, "lateral")
        result = finalize_refinement_pass(
            bundle, "my_model", "lateral",
            log_fn=lambda _: None,
            final_train_loss=0.5,
            n_train_pairs=50,
            n_epochs=10,
        )
        # Lift happened.
        assert result["lift"]["weights_lifted"] is True
        assert (bundle / "lateral").is_file()
        # Manifest was written.
        assert "manifest" in result
        assert result["manifest"]["lateral"]["final_loss"]["train"] == 0.5
        # Manifest on disk.
        assert (bundle / "manifest.json").is_file()

    def test_mode_both_first_lateral_then_axial(self, tmp_path):
        """Simulates Mode=Both: two sequential finalize calls produce
        a bundle with BOTH buckets coexisting."""
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        # Lateral pass first.
        _simulate_cellpose_output(bundle, "lateral")
        finalize_refinement_pass(
            bundle, "my_model", "lateral",
            log_fn=lambda _: None,
            final_train_loss=0.7,
        )
        # Axial pass second.
        _simulate_cellpose_output(bundle, "axial")
        finalize_refinement_pass(
            bundle, "my_model", "axial",
            log_fn=lambda _: None,
            final_train_loss=0.3,
        )
        # Both weights at bundle root.
        assert (bundle / "lateral").is_file()
        assert (bundle / "axial").is_file()
        # Manifest has both buckets.
        manifest = json.loads(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )
        assert "lateral" in manifest
        assert "axial" in manifest
        assert manifest["lateral"]["final_loss"]["train"] == 0.7
        assert manifest["axial"]["final_loss"]["train"] == 0.3

    def test_log_fn_receives_perview_cellpose_io_prefix(self, tmp_path):
        """Verify the print-discipline prefix from """
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        _simulate_cellpose_output(bundle, "lateral")
        log_lines = []
        finalize_refinement_pass(
            bundle, "my_model", "lateral",
            log_fn=log_lines.append,
        )
        # At least one log line per design's print discipline.
        prefixed = [l for l in log_lines if "[PERVIEW-CELLPOSE-IO]" in l]
        assert len(prefixed) >= 3  # start, lift, manifest write at minimum


# ---------------------------------------------------------------------------
# Integration with _paths.py
# ---------------------------------------------------------------------------


class TestRetrainSafety:
    """Audit finding 2026-05-20: re-training the same bucket on an
    existing bundle MUST result in the new weights at bundle/{bucket},
    not stranded at bundle/models/{bucket} due to lift's "refuse to
    overwrite" policy.

    Worker-side fix: unlink the stale dst (and companions) BEFORE
    finalize_refinement_pass runs.  This test asserts the helper's
    contract: when src + dst both exist and dst was pre-unlinked,
    the lift succeeds.
    """

    def test_retrain_with_pre_cleared_dst_succeeds(self, tmp_path):
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        # First training: lift normally.
        _simulate_cellpose_output(bundle, "lateral")
        first_result = finalize_refinement_pass(
            bundle, "my_model", "lateral",
            log_fn=lambda _: None,
        )
        assert first_result["lift"]["weights_lifted"] is True
        first_weights = (bundle / "lateral").read_bytes()

        # Simulate re-train: Cellpose writes fresh weights to
        # bundle/models/lateral.  Without cleanup, dst exists and
        # lift refuses — that's the bug.  WITH cleanup (worker
        # responsibility), dst is cleared first, lift succeeds.
        _simulate_cellpose_output(
            bundle, "lateral",
            weights_content=b"NEW_WEIGHTS_FROM_RETRAIN",
        )
        # Worker-side cleanup that we now do in
        # _RefinementTrainingWorker.run before finalize.
        for stale in (
            bundle / "lateral",
            bundle / "lateral_train_loss.npy",
            bundle / "lateral_val_loss.npy",
            bundle / "lateral_training_config.json",
        ):
            if stale.is_file():
                stale.unlink()

        # Now finalize.
        second_result = finalize_refinement_pass(
            bundle, "my_model", "lateral",
            log_fn=lambda _: None,
        )
        assert second_result["lift"]["weights_lifted"] is True
        new_weights = (bundle / "lateral").read_bytes()
        assert new_weights == b"NEW_WEIGHTS_FROM_RETRAIN"
        assert new_weights != first_weights
        # No leftover models/ subdir.
        assert not (bundle / "models").exists()

    def test_retrain_without_cleanup_strands_new_weights(self, tmp_path):
        """Negative regression: confirms the bug shape if the
        worker-side cleanup ever gets dropped.  WITHOUT pre-clearing
        dst, lift refuses → new weights stranded in models/."""
        bundle = tmp_path / "my_model"
        bundle.mkdir()
        # First training.
        _simulate_cellpose_output(bundle, "lateral")
        finalize_refinement_pass(
            bundle, "my_model", "lateral",
            log_fn=lambda _: None,
        )
        # Second training, NO cleanup.
        _simulate_cellpose_output(
            bundle, "lateral",
            weights_content=b"STRANDED_RETRAIN",
        )
        result = finalize_refinement_pass(
            bundle, "my_model", "lateral",
            log_fn=lambda _: None,
        )
        # Lift refused; bug shape confirmed.
        assert result["lift"]["weights_lifted"] is False
        assert result["lift"]["skipped_reason"] == "dst_exists"
        # New weights stranded.
        assert (bundle / "models" / "lateral").is_file()
        assert (bundle / "models" / "lateral").read_bytes() == b"STRANDED_RETRAIN"


class TestIntegrationWithPaths:

    def test_lift_resolves_via_paths_helpers(self, tmp_path):
        """The lift writes to where _paths.refined_model_weights_path
        expects to find the weights."""
        from napari_epicycle.fusion_reconstruction import _paths
        bundle = _paths.refined_model_dir(tmp_path, "my_model")
        bundle.mkdir(parents=True)
        _simulate_cellpose_output(bundle, "lateral")
        finalize_refinement_pass(
            bundle, "my_model", "lateral", log_fn=lambda _: None,
        )
        # Reader path matches lift's destination.
        expected = _paths.refined_model_weights_path(
            tmp_path, "my_model", "lateral",
        )
        assert expected.is_file()

    def test_manifest_path_matches_paths_helper(self, tmp_path):
        from napari_epicycle.fusion_reconstruction import _paths
        bundle = _paths.refined_model_dir(tmp_path, "my_model")
        bundle.mkdir(parents=True)
        write_or_merge_bundle_manifest(bundle, "my_model", "lateral")
        expected_manifest = _paths.refined_model_manifest_path(
            tmp_path, "my_model",
        )
        assert expected_manifest.is_file()

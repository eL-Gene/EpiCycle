"""Tests for the shared Cellpose post-train lift module at
``packages/epicycle_training/cellpose_lift.py``.

This module was extracted from ``napari_epicycle/fusion_reconstruction/
_refinement_bundle.py`` in Phase 6 of the Step 2 directory restructure
so Step 2 and Step 3 share one implementation.

Tests verify:
* The shared lift function works with any ``weights_filename`` (not
  just "lateral"/"axial" — Step 2 uses "weights")
* Step 3's existing ``_refinement_bundle.py::lift_cellpose_output``
  wrapper still validates bucket + delegates correctly
* All edge cases (already-lifted, src-missing, dst-exists, companion
  collision) preserved from the original implementation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.epicycle_training.cellpose_lift import lift_cellpose_output


# ============================================================================
# Fixture: a bundle dir with Cellpose's auto-inserted models/ subdir
# ============================================================================

def _make_cellpose_output(bundle_dir: Path, weights_filename: str,
                          *, companions: list = None) -> None:
    """Simulate what Cellpose writes after train_seg.

    Creates ``bundle_dir/models/{weights_filename}`` + optional
    companion files in the same subdir.
    """
    cellpose_subdir = bundle_dir / "models"
    cellpose_subdir.mkdir(parents=True, exist_ok=True)
    (cellpose_subdir / weights_filename).write_bytes(b"fake weights")
    for companion_name in (companions or []):
        (cellpose_subdir / companion_name).write_bytes(b"fake companion")


# ============================================================================
# Basic lift behavior (any filename)
# ============================================================================

class TestLiftBasic:
    def test_lifts_weights_with_step2_filename(self, tmp_path):
        """Step 2 uses model_name='weights'."""
        bundle = tmp_path / "step2_bundle"
        bundle.mkdir()
        _make_cellpose_output(bundle, "weights")

        result = lift_cellpose_output(bundle, "weights", log_fn=lambda _: None)
        assert result["weights_lifted"] is True
        assert result["skipped_reason"] is None
        assert (bundle / "weights").is_file()
        assert not (bundle / "models" / "weights").exists()
        assert result["models_subdir_removed"] is True

    def test_lifts_weights_with_step3_lateral_filename(self, tmp_path):
        """Step 3 uses model_name='lateral' or 'axial'."""
        bundle = tmp_path / "step3_bundle"
        bundle.mkdir()
        _make_cellpose_output(bundle, "lateral")

        result = lift_cellpose_output(bundle, "lateral", log_fn=lambda _: None)
        assert result["weights_lifted"] is True
        assert (bundle / "lateral").is_file()

    def test_lifts_weights_with_arbitrary_filename(self, tmp_path):
        """Shared module accepts any weights_filename (no bucket validation)."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        _make_cellpose_output(bundle, "my_custom_model_v1")

        result = lift_cellpose_output(
            bundle, "my_custom_model_v1", log_fn=lambda _: None
        )
        assert result["weights_lifted"] is True
        assert (bundle / "my_custom_model_v1").is_file()


# ============================================================================
# Companion file lifts
# ============================================================================

class TestCompanionLifts:
    def test_lifts_loss_arrays(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        _make_cellpose_output(
            bundle, "weights",
            companions=["weights_train_loss.npy", "weights_val_loss.npy"],
        )

        result = lift_cellpose_output(bundle, "weights", log_fn=lambda _: None)
        assert (bundle / "weights_train_loss.npy").is_file()
        assert (bundle / "weights_val_loss.npy").is_file()
        assert len(result["companions_lifted"]) >= 2

    def test_lifts_training_config(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        _make_cellpose_output(
            bundle, "weights",
            companions=["weights_training_config.json"],
        )

        result = lift_cellpose_output(bundle, "weights", log_fn=lambda _: None)
        assert (bundle / "weights_training_config.json").is_file()

    def test_lifts_unknown_companions_defensively(self, tmp_path):
        """If Cellpose adds new companion files in future versions,
        the lift should move them too (catch-all glob)."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        _make_cellpose_output(
            bundle, "weights",
            companions=["future_cellpose_artifact.dat", "weights_diam.npy"],
        )

        result = lift_cellpose_output(bundle, "weights", log_fn=lambda _: None)
        assert (bundle / "future_cellpose_artifact.dat").is_file()
        assert (bundle / "weights_diam.npy").is_file()


# ============================================================================
# Idempotency
# ============================================================================

class TestIdempotency:
    def test_already_lifted_is_noop(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        _make_cellpose_output(bundle, "weights")

        # First lift
        result1 = lift_cellpose_output(bundle, "weights", log_fn=lambda _: None)
        assert result1["weights_lifted"] is True

        # Second lift on already-lifted state
        result2 = lift_cellpose_output(bundle, "weights", log_fn=lambda _: None)
        assert result2["weights_lifted"] is True  # post-condition holds
        assert result2["skipped_reason"] == "already_lifted"

    def test_dst_exists_refuses_to_overwrite(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        _make_cellpose_output(bundle, "weights")
        # Pre-existing dst weights — different content
        (bundle / "weights").write_bytes(b"pre-existing different content")

        result = lift_cellpose_output(bundle, "weights", log_fn=lambda _: None)
        assert result["skipped_reason"] == "dst_exists"
        # Original dst preserved
        assert (bundle / "weights").read_bytes() == b"pre-existing different content"
        # Src kept for manual resolution
        assert (bundle / "models" / "weights").exists()

    def test_src_missing_no_dst(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        # No Cellpose output at all (training failed before save)
        (bundle / "models").mkdir(parents=True)

        result = lift_cellpose_output(bundle, "weights", log_fn=lambda _: None)
        assert result["skipped_reason"] == "src_missing"
        assert result["weights_lifted"] is False


# ============================================================================
# Error handling
# ============================================================================

class TestErrors:
    def test_empty_weights_filename_raises(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        with pytest.raises(ValueError, match="weights_filename"):
            lift_cellpose_output(bundle, "", log_fn=lambda _: None)

    def test_non_str_weights_filename_raises(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        with pytest.raises(ValueError, match="weights_filename"):
            lift_cellpose_output(bundle, 42, log_fn=lambda _: None)

    def test_nonexistent_save_path_raises(self, tmp_path):
        bundle = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="save_path"):
            lift_cellpose_output(bundle, "weights", log_fn=lambda _: None)


# ============================================================================
# Step 3 wrapper backward-compat — verifies _refinement_bundle still validates
# bucket names
# ============================================================================

class TestStep3WrapperBackwardCompat:
    def test_step3_wrapper_accepts_lateral(self, tmp_path):
        from napari_epicycle.fusion_reconstruction._refinement_bundle import (
            lift_cellpose_output as step3_lift,
        )
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        _make_cellpose_output(bundle, "lateral")
        result = step3_lift(bundle, "lateral", log_fn=lambda _: None)
        assert result["weights_lifted"] is True

    def test_step3_wrapper_accepts_axial(self, tmp_path):
        from napari_epicycle.fusion_reconstruction._refinement_bundle import (
            lift_cellpose_output as step3_lift,
        )
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        _make_cellpose_output(bundle, "axial")
        result = step3_lift(bundle, "axial", log_fn=lambda _: None)
        assert result["weights_lifted"] is True

    def test_step3_wrapper_rejects_invalid_bucket(self, tmp_path):
        from napari_epicycle.fusion_reconstruction._refinement_bundle import (
            lift_cellpose_output as step3_lift,
        )
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        # Step 3 still validates the bucket name even though shared module doesn't
        with pytest.raises(ValueError, match="Invalid bucket"):
            step3_lift(bundle, "weights", log_fn=lambda _: None)
        with pytest.raises(ValueError, match="Invalid bucket"):
            step3_lift(bundle, "spatial", log_fn=lambda _: None)


# ============================================================================
# Log function override
# ============================================================================

class TestFullCellposeOutputSimulation:
    """End-to-end simulation of what Cellpose actually writes after
    train_seg, then lift, and verify the canonical bundle layout.

    This is what happens in production:
    1. Cellpose train_seg called with save_path=bundle_dir, model_name=NAME
    2. Cellpose writes:
       - bundle_dir/models/NAME            (weights, no ext)
       - bundle_dir/models/NAME_train_loss.npy   (Cellpose-written)
       - bundle_dir/models/NAME_val_loss.npy     (cellpose_trainer.py line 237)
       - bundle_dir/models/NAME_training_config.json (cellpose_trainer.py line 264)
    3. lift_cellpose_output called → moves all files up to bundle_dir/
       and deletes empty bundle_dir/models/

    Verifies the post-lift layout matches what manifest readers expect.
    """

    def test_full_cellpose_train_seg_output_lift(self, tmp_path):
        bundle = tmp_path / "step2_round_dir"
        bundle.mkdir()

        # Simulate exactly what a real Cellpose training run leaves on disk
        # for a Step 2 round (Cellpose passed model_name="weights")
        cellpose_subdir = bundle / "models"
        cellpose_subdir.mkdir()

        # Main weights (no extension)
        (cellpose_subdir / "weights").write_bytes(b"cellpose-trained weights bytes")

        # Loss arrays (Cellpose convention: _train_loss.npy + _val_loss.npy
        # per cellpose_trainer.py line 235-237)
        (cellpose_subdir / "weights_train_loss.npy").write_bytes(
            b"\x93NUMPY\x01\x00\x00fake train loss arr"
        )
        (cellpose_subdir / "weights_val_loss.npy").write_bytes(
            b"\x93NUMPY\x01\x00\x00fake val loss arr"
        )

        # Training config JSON (cellpose_trainer.py line 264-266)
        import json as _json
        config = {
            "model_name": "weights",
            "initialization": "scratch",
            "hyperparameters": {"n_epochs": 100, "learning_rate": 0.01},
        }
        (cellpose_subdir / "weights_training_config.json").write_text(
            _json.dumps(config, indent=2), encoding="utf-8"
        )

        # Run the lift
        result = lift_cellpose_output(
            bundle, "weights", log_fn=lambda _: None
        )

        # Verify ALL files lifted to bundle root
        assert (bundle / "weights").is_file()
        assert (bundle / "weights").read_bytes() == b"cellpose-trained weights bytes"
        assert (bundle / "weights_train_loss.npy").is_file()
        assert (bundle / "weights_val_loss.npy").is_file()
        assert (bundle / "weights_training_config.json").is_file()

        # Verify training config content preserved
        loaded = _json.loads(
            (bundle / "weights_training_config.json").read_text(encoding="utf-8")
        )
        assert loaded["hyperparameters"]["n_epochs"] == 100

        # Verify models/ subdir cleaned up
        assert not cellpose_subdir.exists()
        assert result["models_subdir_removed"] is True

        # Verify result dict integrity
        assert result["weights_lifted"] is True
        assert result["skipped_reason"] is None
        assert len(result["companions_lifted"]) >= 3  # 3+ companion files moved

    def test_step3_style_full_output_lift(self, tmp_path):
        """Step 3 convention: bucket name IS the model_name, so files
        are named lateral / lateral_train_loss.npy / etc."""
        bundle = tmp_path / "step3_refinement_bundle"
        bundle.mkdir()
        cellpose_subdir = bundle / "models"
        cellpose_subdir.mkdir()

        (cellpose_subdir / "lateral").write_bytes(b"step3 lateral weights")
        (cellpose_subdir / "lateral_train_loss.npy").write_bytes(b"loss")
        (cellpose_subdir / "lateral_val_loss.npy").write_bytes(b"vloss")

        result = lift_cellpose_output(
            bundle, "lateral", log_fn=lambda _: None
        )

        # All 3 files in bundle root, models/ gone
        assert (bundle / "lateral").read_bytes() == b"step3 lateral weights"
        assert (bundle / "lateral_train_loss.npy").is_file()
        assert (bundle / "lateral_val_loss.npy").is_file()
        assert not cellpose_subdir.exists()
        assert result["weights_lifted"] is True


class TestLogFunctionOverride:
    def test_custom_log_fn_receives_messages(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        _make_cellpose_output(bundle, "weights")

        captured = []
        result = lift_cellpose_output(
            bundle, "weights", log_fn=captured.append
        )
        assert any("[PERVIEW-CELLPOSE-IO]" in m for m in captured)
        assert any("lifted to" in m for m in captured)

    def test_default_log_fn_is_print(self, tmp_path, capsys):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        _make_cellpose_output(bundle, "weights")
        lift_cellpose_output(bundle, "weights")  # no log_fn — defaults to print
        captured = capsys.readouterr()
        assert "[PERVIEW-CELLPOSE-IO]" in captured.out

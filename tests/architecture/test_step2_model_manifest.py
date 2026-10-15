"""Tests for the Step 2 per-bucket-round model manifest writer/reader
at ``packages/epicycle_training/step2_model_manifest.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.epicycle_training.step2_model_manifest import (
    SCHEMA_VERSION,
    VALID_BUCKETS,
    VALID_INIT_MODES,
    build_step2_model_manifest,
    write_step2_model_manifest,
    read_step2_model_manifest,
)


# ============================================================================
# build_step2_model_manifest
# ============================================================================

class TestBuildManifest:
    def test_minimal_scratch_init(self):
        m = build_step2_model_manifest(
            run_id="default",
            bucket="lateral",
            round_number=0,
            source_workspace="/some/ws",
        )
        # Phase B.3 (2026-05-25): schema bumped to 2.1.  Additive,
        # backward-compatible — 2.0 readers ignore unknown keys.
        assert m["schema_version"] == "2.1"
        assert m["run_id"] == "default"
        assert m["bucket"] == "lateral"
        assert m["round_number"] == 0
        assert m["backend"] == "cellpose"  # default
        assert m["init"]["mode"] == "scratch"  # default
        assert m["init"]["path"] is None
        # All inherited_from_* are None for scratch
        assert m["inherited_from_run"] is None
        assert m["inherited_from_bucket"] is None
        assert m["inherited_from_step3_bundle"] is None

    def test_file_pointer_naming_follows_weights_filename(self):
        m = build_step2_model_manifest(
            run_id="r", bucket="lateral", round_number=0,
            source_workspace="/ws", weights_filename="weights",
        )
        assert m["weights_filename"] == "weights"
        assert m["weights_path"] == "weights"
        assert m["loss_train_path"] == "weights_train_loss.npy"
        assert m["training_config_path"] == "weights_training_config.json"
        # Release-hardening 1b#7 (2026-09-07): the val-loss pointer is only
        # advertised when a validation loss EXISTS.  Step 2 passes no
        # `val_data`, so `_val_loss.npy` is never written and a manifest
        # naming it sent readers to FileNotFoundError.  The naming
        # convention itself (audit C-4 fix) is unchanged — see below.
        assert m["loss_val_path"] is None

        m_val = build_step2_model_manifest(
            run_id="r", bucket="lateral", round_number=0,
            source_workspace="/ws", weights_filename="weights",
            final_loss_val=0.42,
        )
        assert m_val["loss_val_path"] == "weights_val_loss.npy"  # Cellpose's actual filename

    def test_step3_style_weights_filename(self):
        """If a caller passes lateral/axial as weights_filename, file
        paths use that name (Step 3 convention)."""
        m = build_step2_model_manifest(
            run_id="r", bucket="lateral", round_number=0,
            source_workspace="/ws", weights_filename="lateral",
        )
        assert m["weights_path"] == "lateral"
        assert m["loss_train_path"] == "lateral_train_loss.npy"

    def test_full_metadata(self):
        m = build_step2_model_manifest(
            run_id="nuclei_v1", bucket="lateral", round_number=3,
            source_workspace="/data/ws",
            weights_filename="weights",
            init_mode="inherited",
            init_path="../round_02/weights",
            init_combo_text="From lateral round 2",
            inherited_from_run="default",
            final_loss_train=0.0847,
            final_loss_val=0.1123,
            n_train_pairs=156, n_val_pairs=17,
            n_epochs=100, batch_size=8,
            channels_used=["DAPI", "actin"],
            channels_snapshot=["DAPI", "actin"],
            backend="cellpose",
        )
        assert m["init"]["mode"] == "inherited"
        assert m["init"]["init_combo_text"] == "From lateral round 2"
        assert m["inherited_from_run"] == "default"
        assert m["final_loss"]["train"] == 0.0847
        assert m["final_loss"]["val"] == 0.1123
        assert m["training"]["n_train_pairs"] == 156
        assert m["training"]["channels_used"] == ["DAPI", "actin"]
        assert m["channels_snapshot"] == ["DAPI", "actin"]


# ============================================================================
# Validation
# ============================================================================

class TestValidation:
    def test_invalid_bucket_raises(self):
        with pytest.raises(ValueError, match="bucket"):
            build_step2_model_manifest(
                run_id="r", bucket="spatial",  # invalid: legacy name
                round_number=0, source_workspace="/ws",
            )

    def test_invalid_init_mode_raises(self):
        with pytest.raises(ValueError, match="init.mode"):
            build_step2_model_manifest(
                run_id="r", bucket="lateral", round_number=0,
                source_workspace="/ws", init_mode="bogus",
            )

    def test_negative_round_number_raises(self):
        with pytest.raises(ValueError, match="round_number"):
            build_step2_model_manifest(
                run_id="r", bucket="lateral", round_number=-1,
                source_workspace="/ws",
            )

    def test_non_int_round_number_raises(self):
        with pytest.raises(ValueError, match="round_number"):
            build_step2_model_manifest(
                run_id="r", bucket="lateral", round_number="3",
                source_workspace="/ws",
            )


# ============================================================================
# Mutually-exclusive inheritance groups
# ============================================================================

class TestMutuallyExclusiveInheritance:
    def test_cross_run_and_cross_bucket_both_set_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_step2_model_manifest(
                run_id="r", bucket="lateral", round_number=0,
                source_workspace="/ws",
                inherited_from_run="other_run",
                inherited_from_bucket="axial",  # different group
            )

    def test_cross_run_and_step3_both_set_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_step2_model_manifest(
                run_id="r", bucket="lateral", round_number=0,
                source_workspace="/ws",
                inherited_from_run="other",
                inherited_from_step3_bundle="my_refined",
                inherited_from_step3_bucket="lateral",
            )

    def test_step3_pair_one_without_other_raises(self):
        # bundle without bucket
        with pytest.raises(ValueError, match="set together as a pair"):
            build_step2_model_manifest(
                run_id="r", bucket="lateral", round_number=0,
                source_workspace="/ws",
                inherited_from_step3_bundle="my_refined",
                inherited_from_step3_bucket=None,
            )
        # bucket without bundle
        with pytest.raises(ValueError, match="set together as a pair"):
            build_step2_model_manifest(
                run_id="r", bucket="lateral", round_number=0,
                source_workspace="/ws",
                inherited_from_step3_bundle=None,
                inherited_from_step3_bucket="lateral",
            )

    def test_external_pair_one_without_other_raises(self):
        with pytest.raises(ValueError, match="set together as a pair"):
            build_step2_model_manifest(
                run_id="r", bucket="lateral", round_number=0,
                source_workspace="/ws",
                inherited_from_external_path="/some/path",
                inherited_from_external_label=None,
            )

    def test_step3_pair_set_together_valid(self):
        m = build_step2_model_manifest(
            run_id="r", bucket="lateral", round_number=0,
            source_workspace="/ws",
            init_mode="step3_refined",
            inherited_from_step3_bundle="my_refined_model",
            inherited_from_step3_bucket="lateral",
        )
        assert m["inherited_from_step3_bundle"] == "my_refined_model"
        assert m["inherited_from_step3_bucket"] == "lateral"

    def test_external_pair_set_together_valid(self):
        m = build_step2_model_manifest(
            run_id="r", bucket="lateral", round_number=0,
            source_workspace="/ws",
            init_mode="browse",
            inherited_from_external_path="/colleague/model",
            inherited_from_external_label="colleague nuclei 2026-04",
        )
        assert m["inherited_from_external_path"] == "/colleague/model"
        assert m["inherited_from_external_label"] == "colleague nuclei 2026-04"


# ============================================================================
# write + read round-trip
# ============================================================================

class TestWriteAndRead:
    def test_round_trip(self, tmp_path):
        m = build_step2_model_manifest(
            run_id="default", bucket="lateral", round_number=2,
            source_workspace=str(tmp_path), weights_filename="weights",
            final_loss_train=0.1, n_train_pairs=42,
        )
        manifest_path = tmp_path / "manifest.json"
        write_step2_model_manifest(manifest_path, m)
        assert manifest_path.is_file()

        loaded = read_step2_model_manifest(manifest_path)
        assert loaded["run_id"] == "default"
        assert loaded["bucket"] == "lateral"
        assert loaded["round_number"] == 2
        assert loaded["weights_path"] == "weights"
        assert loaded["final_loss"]["train"] == 0.1
        assert loaded["training"]["n_train_pairs"] == 42

    def test_write_creates_parent_dirs(self, tmp_path):
        m = build_step2_model_manifest(
            run_id="r", bucket="lateral", round_number=0,
            source_workspace="/ws",
        )
        nested_path = tmp_path / "a" / "b" / "c" / "manifest.json"
        write_step2_model_manifest(nested_path, m)
        assert nested_path.is_file()

    def test_write_overwrite_false_raises_if_exists(self, tmp_path):
        m = build_step2_model_manifest(
            run_id="r", bucket="lateral", round_number=0,
            source_workspace="/ws",
        )
        manifest_path = tmp_path / "manifest.json"
        write_step2_model_manifest(manifest_path, m)
        # Second write with overwrite=False
        with pytest.raises(FileExistsError):
            write_step2_model_manifest(manifest_path, m, overwrite=False)

    def test_write_overwrite_true_replaces(self, tmp_path):
        m1 = build_step2_model_manifest(
            run_id="r", bucket="lateral", round_number=0,
            source_workspace="/ws", n_train_pairs=10,
        )
        m2 = build_step2_model_manifest(
            run_id="r", bucket="lateral", round_number=0,
            source_workspace="/ws", n_train_pairs=99,
        )
        manifest_path = tmp_path / "manifest.json"
        write_step2_model_manifest(manifest_path, m1)
        write_step2_model_manifest(manifest_path, m2, overwrite=True)
        loaded = read_step2_model_manifest(manifest_path)
        assert loaded["training"]["n_train_pairs"] == 99

    def test_write_is_atomic_no_tmp_left(self, tmp_path):
        """Tmp file used during write should be cleaned up."""
        m = build_step2_model_manifest(
            run_id="r", bucket="lateral", round_number=0,
            source_workspace="/ws",
        )
        manifest_path = tmp_path / "manifest.json"
        write_step2_model_manifest(manifest_path, m)
        tmps = list(tmp_path.glob("*.tmp"))
        assert tmps == []


# ============================================================================
# read error paths
# ============================================================================

class TestReadErrors:
    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_step2_model_manifest(tmp_path / "nope.json")

    def test_malformed_json_raises(self, tmp_path):
        bad = tmp_path / "manifest.json"
        bad.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            read_step2_model_manifest(bad)

    def test_non_object_json_raises(self, tmp_path):
        bad = tmp_path / "manifest.json"
        bad.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            read_step2_model_manifest(bad)

    def test_wrong_schema_version_raises(self, tmp_path):
        bad = tmp_path / "manifest.json"
        bad.write_text(json.dumps({
            "schema_version": "1.0",  # wrong
            "run_id": "r",
            "bucket": "lateral",
            "round_number": 0,
            "backend": "cellpose",
            "weights_filename": "weights",
            "weights_path": "weights",
            "init": {"mode": "scratch"},
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            read_step2_model_manifest(bad)

    def test_missing_required_field_raises(self, tmp_path):
        bad = tmp_path / "manifest.json"
        bad.write_text(json.dumps({
            "schema_version": "2.0",
            # missing run_id, bucket, etc.
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required fields"):
            read_step2_model_manifest(bad)


# ============================================================================
# Constants
# ============================================================================

class TestConstants:
    def test_schema_version(self):
        # Phase B.3 bumped 2.0 → 2.1 (additive reproducibility fields).
        assert SCHEMA_VERSION == "2.1"

    def test_accepted_schema_versions_includes_both(self):
        # Readers tolerate both during the 2.0 → 2.1 migration window
        from packages.epicycle_training.step2_model_manifest import (
            ACCEPTED_SCHEMA_VERSIONS,
        )
        assert "2.0" in ACCEPTED_SCHEMA_VERSIONS
        assert "2.1" in ACCEPTED_SCHEMA_VERSIONS

    def test_valid_buckets(self):
        assert VALID_BUCKETS == ("lateral", "axial")

    def test_valid_init_modes_complete(self):
        # All 6 init modes
        for mode in ("scratch", "inherited", "cross_bucket",
                     "step3_refined", "browse", "pretrained"):
            assert mode in VALID_INIT_MODES


class TestH2InheritedFromRunBucketAndRound:
    """Commit 2 audit H2 fix (2026-05-26): schema 2.0 manifest gained
    `inherited_from_run_bucket` + `inherited_from_run_round_id` so
    'inherited from run X bucket Y round Z' provenance is recoverable
    from the manifest (was previously only in human-readable
    `init_combo_text`)."""

    def test_inherited_with_bucket_and_round_populates_fields(self, tmp_path):
        m = build_step2_model_manifest(
            run_id="target_run", bucket="lateral", round_number=0,
            source_workspace=str(tmp_path),
            init_mode="inherited",
            inherited_from_run="src_run",
            inherited_from_run_bucket="axial",
            inherited_from_run_round_id="round_03",
        )
        assert m["inherited_from_run"] == "src_run"
        assert m["inherited_from_run_bucket"] == "axial"
        assert m["inherited_from_run_round_id"] == "round_03"

    def test_inherited_without_bucket_or_round_is_legal(self, tmp_path):
        """Fork mode populates only inherited_from_run (no specific
        round known) — must not violate the schema mutex check."""
        m = build_step2_model_manifest(
            run_id="forked", bucket="lateral", round_number=0,
            source_workspace=str(tmp_path),
            init_mode="inherited",
            inherited_from_run="src_run",
        )
        assert m["inherited_from_run"] == "src_run"
        assert m["inherited_from_run_bucket"] is None
        assert m["inherited_from_run_round_id"] is None

    def test_mutex_check_rejects_cross_group(self, tmp_path):
        """The within_workspace_cross_run group (run+bucket+round) is
        mutex with cross_step.  Setting fields from both must raise."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_step2_model_manifest(
                run_id="x", bucket="lateral", round_number=0,
                source_workspace=str(tmp_path),
                inherited_from_run="src_run",
                inherited_from_run_bucket="axial",
                inherited_from_step3_bundle="b1",
                inherited_from_step3_bucket="lateral",
            )

    def test_default_fields_are_none(self, tmp_path):
        m = build_step2_model_manifest(
            run_id="x", bucket="lateral", round_number=0,
            source_workspace=str(tmp_path),
            init_mode="scratch",
        )
        assert m["inherited_from_run_bucket"] is None
        assert m["inherited_from_run_round_id"] is None

    def test_roundtrip_via_write_read(self, tmp_path):
        m = build_step2_model_manifest(
            run_id="target", bucket="axial", round_number=1,
            source_workspace=str(tmp_path),
            init_mode="inherited",
            inherited_from_run="src",
            inherited_from_run_bucket="lateral",
            inherited_from_run_round_id="round_05",
        )
        mp = tmp_path / "manifest.json"
        write_step2_model_manifest(mp, m)
        loaded = read_step2_model_manifest(mp)
        assert loaded["inherited_from_run"] == "src"
        assert loaded["inherited_from_run_bucket"] == "lateral"
        assert loaded["inherited_from_run_round_id"] == "round_05"

    def test_bucket_without_run_rejected(self, tmp_path):
        """Commit 4 audit HIGH co-pop: bucket without run is orphan
        provenance — must raise."""
        with pytest.raises(ValueError, match="require inherited_from_run"):
            build_step2_model_manifest(
                run_id="x", bucket="lateral", round_number=0,
                source_workspace=str(tmp_path),
                inherited_from_run_bucket="axial",
            )

    def test_round_id_without_run_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="require inherited_from_run"):
            build_step2_model_manifest(
                run_id="x", bucket="lateral", round_number=0,
                source_workspace=str(tmp_path),
                inherited_from_run_round_id="round_03",
            )

    def test_forward_compat_read_pre_fix_manifest(self, tmp_path):
        """Commit 4 audit Audit-3 LOW L1: reading a schema-2.0 manifest
        authored BEFORE the H2 new fields existed must succeed with
        the new fields defaulting to None via .get().  Simulates a
        manifest written by the pre-H2 build_step2_model_manifest."""
        import json
        # Hand-craft a minimal manifest WITHOUT the new H2 fields
        pre_h2_manifest = {
            "schema_version": "2.0",
            "run_id": "legacy_target", "bucket": "lateral", "round_number": 0,
            "backend": "cellpose", "created_at": "2026-05-25T12:00:00Z",
            "source_workspace": str(tmp_path),
            "weights_filename": "legacy_target_lateral",
            "weights_path": "legacy_target_lateral",
            "loss_train_path": "legacy_target_lateral_train_loss.npy",
            "loss_val_path": "legacy_target_lateral_val_loss.npy",
            "training_config_path": "legacy_target_lateral_training_config.json",
            "init": {"mode": "inherited", "path": None, "init_combo_text": ""},
            "inherited_from_run": "src_run",  # OLD field present
            # NO inherited_from_run_bucket / inherited_from_run_round_id
            "inherited_from_bucket": None,
            "inherited_from_step3_bundle": None,
            "inherited_from_step3_bucket": None,
            "inherited_from_external_path": None,
            "inherited_from_external_label": None,
            "final_loss": {"train": None, "val": None},
            "training": {
                "n_train_pairs": 0, "n_val_pairs": 0,
                "n_epochs": 0, "batch_size": 0,
                "channels_used": [],
            },
            "channels_snapshot": [],
            "labels_format": "tiff",
        }
        mp = tmp_path / "manifest.json"
        mp.write_text(json.dumps(pre_h2_manifest), encoding="utf-8")
        # Read via the official reader — schema check must pass
        loaded = read_step2_model_manifest(mp)
        # Old field still present
        assert loaded["inherited_from_run"] == "src_run"
        # New fields default to None via .get() — never raise on missing
        assert loaded.get("inherited_from_run_bucket") is None
        assert loaded.get("inherited_from_run_round_id") is None


# ============================================================================
# Phase B.3 (2026-05-25) — schema 2.1 additive fields
# ============================================================================

class TestB3Schema21ReproducibilityFields:
    """Schema 2.1 adds training-pair identity + random_seed + env +
    timing fields.  All optional; build still produces a valid manifest
    when none are passed (defaults to []/None)."""

    def test_default_schema_21_has_empty_reproducibility(self, tmp_path):
        m = build_step2_model_manifest(
            run_id="default", bucket="lateral", round_number=0,
            source_workspace=str(tmp_path), init_mode="scratch",
        )
        # Schema 2.1 fields present even when caller doesn't pass them
        t = m["training"]
        assert t["train_pair_labels"] == []
        assert t["train_pair_uris"] == []
        assert t["accepted_dir_checksum"] is None
        assert t["random_seed"] is None
        # Timing + environment sub-dicts present, all-None
        assert m["timing"]["started_at"] is None
        assert m["timing"]["finished_at"] is None
        assert m["timing"]["duration_seconds"] is None
        assert m["environment"]["cellpose_version"] is None
        assert m["environment"]["torch_version"] is None
        assert m["environment"]["device"] is None

    def test_populated_schema_21_fields_roundtrip(self, tmp_path):
        from packages.epicycle_training.step2_model_manifest import (
            read_step2_model_manifest, write_step2_model_manifest,
        )
        m = build_step2_model_manifest(
            run_id="myrun", bucket="lateral", round_number=2,
            source_workspace=str(tmp_path),
            weights_filename="myrun_lateral_round_02",
            init_mode="scratch",
            train_pair_labels=["stkA_XY_0042", "stkB_XY_0103"],
            train_pair_uris=[
                {"slice_label": "stkA_XY_0042",
                 "img_uri": "/ws/.../stkA_XY_z0042_img.tif",
                 "mask_uri": "/ws/.../stkA_XY_z0042_mask.tif"},
            ],
            accepted_dir_checksum="sha256:abc123",
            random_seed=1337,
            timing_started_at="2026-05-25T14:00:00Z",
            timing_finished_at="2026-05-25T14:24:33Z",
            timing_duration_seconds=1473.0,
            env_cellpose_version="3.0.10",
            env_torch_version="2.1.0+cu118",
            env_python_version="3.10.13",
            env_device="NVIDIA RTX 3090",
            env_platform="Windows-11-10.0.26200",
        )
        mp = tmp_path / "manifest.json"
        write_step2_model_manifest(mp, m)
        loaded = read_step2_model_manifest(mp)
        assert loaded["training"]["train_pair_labels"] == [
            "stkA_XY_0042", "stkB_XY_0103"
        ]
        assert loaded["training"]["train_pair_uris"][0]["slice_label"] == "stkA_XY_0042"
        assert loaded["training"]["accepted_dir_checksum"] == "sha256:abc123"
        assert loaded["training"]["random_seed"] == 1337
        assert loaded["timing"]["duration_seconds"] == 1473.0
        assert loaded["environment"]["cellpose_version"] == "3.0.10"
        assert loaded["environment"]["device"] == "NVIDIA RTX 3090"

    def test_schema_20_manifest_still_readable(self, tmp_path):
        """Forward compat: a manifest emitted pre-Phase-B.3 (schema 2.0)
        must read cleanly via the post-B.3 reader.  Validation accepts
        BOTH versions via `ACCEPTED_SCHEMA_VERSIONS`."""
        import json
        from packages.epicycle_training.step2_model_manifest import (
            read_step2_model_manifest,
        )
        # Hand-craft a minimal schema 2.0 manifest (NO B.3 fields)
        pre_b3 = {
            "schema_version": "2.0",
            "run_id": "old_run", "bucket": "lateral", "round_number": 0,
            "backend": "cellpose", "created_at": "2026-05-25T10:00:00Z",
            "source_workspace": str(tmp_path),
            "weights_filename": "old_run_lateral",
            "weights_path": "old_run_lateral",
            "loss_train_path": "old_run_lateral_train_loss.npy",
            "loss_val_path": "old_run_lateral_val_loss.npy",
            "training_config_path": "old_run_lateral_training_config.json",
            "init": {"mode": "scratch", "path": None, "init_combo_text": ""},
            "inherited_from_run": None,
            "inherited_from_bucket": None,
            "inherited_from_step3_bundle": None,
            "inherited_from_step3_bucket": None,
            "inherited_from_external_path": None,
            "inherited_from_external_label": None,
            "final_loss": {"train": None, "val": None},
            "training": {
                "n_train_pairs": 5, "n_val_pairs": 0,
                "n_epochs": 100, "batch_size": 8,
                "channels_used": [],
            },
            "channels_snapshot": [],
            "labels_format": "tiff",
        }
        mp = tmp_path / "manifest.json"
        mp.write_text(json.dumps(pre_b3), encoding="utf-8")
        # Reader accepts schema 2.0 → no exception
        loaded = read_step2_model_manifest(mp)
        assert loaded["schema_version"] == "2.0"
        # Schema 2.1 fields missing — readers should `.get()` default
        assert loaded["training"].get("train_pair_labels") is None
        assert loaded.get("timing") is None
        assert loaded.get("environment") is None

    def test_schema_validation_rejects_unknown_version(self, tmp_path):
        import json, pytest
        from packages.epicycle_training.step2_model_manifest import (
            read_step2_model_manifest,
        )
        # Future schema 3.0 — should be rejected by the validator
        future = {
            "schema_version": "3.0",
            "run_id": "x", "bucket": "lateral", "round_number": 0,
            "backend": "cellpose",
            "weights_filename": "x", "weights_path": "x",
            "init": {},
        }
        mp = tmp_path / "manifest.json"
        mp.write_text(json.dumps(future), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            read_step2_model_manifest(mp)

"""Tests for ``napari_epicycle/hitl_training_annotation/_paths.py``.

Covers Phase 1 of the Step 2 directory restructure: the path-helper
module.

Test groups:
    - State JSON paths + workspace roots
    - Run-level paths (run_root, run_manifest_path)
    - Bucket paths (bucket_root, bucket_rounds_root, bucket_models_root)
    - Per-round data paths (round_dir, accepted, staged, predictions, etc.)
    - Per-round model paths (model_dir, weights, manifest, loss, training_config)
    - Slug + parsing helpers (validate_run_id, parse_slice_label, bucket_for_view)
    - Legacy aliases (training_data, models, predictions, artifacts subtrees)
    - Layout detection (UNKNOWN / LEGACY / NEW / MIXED)
    - Constants (DEFAULT_RUN_ID, VALID_BUCKETS, VALID_VIEWS, filenames)
"""

from __future__ import annotations

import pytest

from napari_epicycle.hitl_training_annotation import _paths


@pytest.fixture
def ws(tmp_path):
    """Fresh workspace dir (does not exist yet on disk)."""
    return tmp_path / "workspace"


# ============================================================================
# State JSON + workspace root
# ============================================================================

class TestStateAndRoots:
    def test_workspace_root(self, ws):
        assert _paths.workspace_root(ws) == ws

    def test_state_json_path(self, ws):
        assert _paths.state_json_path(ws) == ws / "step2_state.json"

    def test_legacy_state_json_path(self, ws):
        assert _paths.legacy_state_json_path(ws) == ws / "hitl_state.json"

    def test_step2_root(self, ws):
        assert _paths.step2_root(ws) == ws / "artifacts" / "step2"

    def test_step2_runs_root(self, ws):
        assert _paths.step2_runs_root(ws) == ws / "artifacts" / "step2" / "runs"

    def test_accepts_str_workspace(self, tmp_path):
        ws_str = str(tmp_path / "workspace")
        assert _paths.state_json_path(ws_str) == _paths.state_json_path(tmp_path / "workspace")


# ============================================================================
# Run-level paths
# ============================================================================

class TestRunPaths:
    def test_run_root_default(self, ws):
        expected = ws / "artifacts" / "step2" / "runs" / "default"
        assert _paths.run_root(ws, "default") == expected

    def test_run_root_custom_slug(self, ws):
        expected = ws / "artifacts" / "step2" / "runs" / "nuclei_v1"
        assert _paths.run_root(ws, "nuclei_v1") == expected

    def test_run_root_with_dash(self, ws):
        expected = ws / "artifacts" / "step2" / "runs" / "pilot-2026"
        assert _paths.run_root(ws, "pilot-2026") == expected

    def test_run_manifest_path(self, ws):
        expected = ws / "artifacts" / "step2" / "runs" / "default" / "run_manifest.json"
        assert _paths.run_manifest_path(ws, "default") == expected

    def test_run_root_invalid_slug_raises(self, ws):
        with pytest.raises(ValueError):
            _paths.run_root(ws, "../escape")

    def test_run_root_empty_slug_raises(self, ws):
        with pytest.raises(ValueError):
            _paths.run_root(ws, "")


# ============================================================================
# Bucket paths
# ============================================================================

class TestBucketPaths:
    def test_bucket_root_lateral(self, ws):
        expected = ws / "artifacts" / "step2" / "runs" / "default" / "lateral"
        assert _paths.bucket_root(ws, "default", "lateral") == expected

    def test_bucket_root_axial(self, ws):
        expected = ws / "artifacts" / "step2" / "runs" / "default" / "axial"
        assert _paths.bucket_root(ws, "default", "axial") == expected

    def test_bucket_root_rejects_spatial(self, ws):
        # 'spatial' is the LEGACY name; new code must use 'lateral'.
        with pytest.raises(ValueError, match="spatial"):
            _paths.bucket_root(ws, "default", "spatial")

    def test_bucket_root_rejects_unknown(self, ws):
        with pytest.raises(ValueError):
            _paths.bucket_root(ws, "default", "horizontal")

    def test_bucket_root_rejects_capitalization(self, ws):
        # bucket names are lowercase only
        with pytest.raises(ValueError):
            _paths.bucket_root(ws, "default", "Lateral")

    def test_bucket_rounds_root(self, ws):
        expected = ws / "artifacts" / "step2" / "runs" / "default" / "lateral" / "rounds"
        assert _paths.bucket_rounds_root(ws, "default", "lateral") == expected

    def test_bucket_models_root(self, ws):
        expected = ws / "artifacts" / "step2" / "runs" / "default" / "lateral" / "models"
        assert _paths.bucket_models_root(ws, "default", "lateral") == expected


# ============================================================================
# Per-round data paths
# ============================================================================

class TestRoundDataPaths:
    def test_round_dir(self, ws):
        """Phase B.2: flat layout — round_dir at `{bucket}/round_NN/`
        (no `rounds/` segment).  Legacy `bucket/rounds/round_NN/`
        path is now reachable via `legacy_split_round_dir`."""
        expected = (
            ws / "artifacts" / "step2" / "runs" / "default"
            / "lateral" / "round_03"
        )
        assert _paths.round_dir(ws, "default", "lateral", 3) == expected

    def test_legacy_split_round_dir(self, ws):
        """Pre-Phase-B.2 split-layout path retained for dual-path readers."""
        expected = (
            ws / "artifacts" / "step2" / "runs" / "default"
            / "lateral" / "rounds" / "round_03"
        )
        assert _paths.legacy_split_round_dir(ws, "default", "lateral", 3) == expected

    def test_round_dir_zero_padded_to_two_digits(self, ws):
        assert _paths.round_dir(ws, "default", "lateral", 0).name == "round_00"
        assert _paths.round_dir(ws, "default", "lateral", 9).name == "round_09"
        assert _paths.round_dir(ws, "default", "lateral", 99).name == "round_99"

    def test_round_dir_above_99(self, ws):
        # f"{100:02d}" expands to "100", not "00"; verify no clipping
        assert _paths.round_dir(ws, "default", "lateral", 100).name == "round_100"

    def test_round_dir_negative_raises(self, ws):
        with pytest.raises(ValueError):
            _paths.round_dir(ws, "default", "lateral", -1)

    def test_round_dir_non_int_raises(self, ws):
        with pytest.raises(ValueError):
            _paths.round_dir(ws, "default", "lateral", "3")

    def test_round_dir_bool_rejected(self, ws):
        # bool is int subclass in Python; explicitly reject for clarity
        with pytest.raises(ValueError):
            _paths.round_dir(ws, "default", "lateral", True)

    def test_round_accepted_dir(self, ws):
        d = _paths.round_accepted_dir(ws, "default", "lateral", 0)
        assert d.name == "accepted"
        assert d.parent.name == "round_00"

    def test_round_staged_dir(self, ws):
        d = _paths.round_staged_dir(ws, "default", "lateral", 0)
        assert d.name == "staged"

    def test_round_predictions_dir(self, ws):
        d = _paths.round_predictions_dir(ws, "default", "lateral", 0)
        assert d.name == "predictions"

    def test_round_predictions_manifest_path(self, ws):
        p = _paths.round_predictions_manifest_path(ws, "default", "lateral", 0)
        assert p.name == "manifest.json"
        assert p.parent.name == "predictions"

    def test_round_metadata_path(self, ws):
        p = _paths.round_metadata_path(ws, "default", "lateral", 0)
        assert p.name == "round_metadata.json"

    def test_round_pairs_manifest_path(self, ws):
        p = _paths.round_pairs_manifest_path(ws, "default", "lateral", 0)
        assert p.name == "pairs_manifest.json"

    def test_round_selected_indices_path(self, ws):
        p = _paths.round_selected_indices_path(ws, "default", "lateral", 0)
        assert p.name == "selected_indices.json"


# ============================================================================
# Per-round model paths
# ============================================================================

class TestModelPaths:
    def test_model_dir_under_round_dir(self, ws):
        """Phase B.2: flat layout — model_dir at `{round_dir}/model/`.
        Co-located with accepted/ + predictions/ inside the round dir.
        Pre-B.2 was `{bucket}/models/round_NN/` — now reachable via
        `legacy_split_model_dir`."""
        d = _paths.model_dir(ws, "default", "lateral", 3)
        assert d.name == "model"
        assert d.parent.name == "round_03"
        assert d.parent.parent.name == "lateral"

    def test_legacy_split_model_dir(self, ws):
        """Pre-Phase-B.2 split-layout path retained for dual-path readers."""
        d = _paths.legacy_split_model_dir(ws, "default", "lateral", 3)
        assert d.name == "round_03"
        assert d.parent.name == "models"
        assert d.parent.parent.name == "lateral"

    def test_model_manifest_history_dir(self, ws):
        """Phase B.2 helper for the C-5 manifest archive subdir."""
        d = _paths.model_manifest_history_dir(ws, "default", "lateral", 3)
        assert d.name == "manifest_history"
        assert d.parent == _paths.model_dir(ws, "default", "lateral", 3)

    def test_model_weights_path(self, ws):
        # The actual Cellpose weights file, no extension
        p = _paths.model_weights_path(ws, "default", "lateral", 0)
        assert p.name == "weights"

    def test_model_manifest_path(self, ws):
        # Schema 2.0 manifest at the per-bucket-round level
        p = _paths.model_manifest_path(ws, "default", "lateral", 0)
        assert p.name == "manifest.json"

    def test_model_loss_path_train(self, ws):
        # Cellpose-written: weights_train_loss.npy
        p = _paths.model_loss_path(ws, "default", "lateral", 0, "train")
        assert p.name == "weights_train_loss.npy"

    def test_model_loss_path_val(self, ws):
        # 2026-05-26 audit C-4 fix: Cellpose writes `_val_loss.npy`
        # (NOT `_validation_loss.npy`).  See cellpose_trainer.py:237.
        # Step 3's bundle writer uses the same `_val_loss.npy`.
        p = _paths.model_loss_path(ws, "default", "lateral", 0, "val")
        assert p.name == "weights_val_loss.npy"

    def test_model_loss_path_invalid_kind_raises(self, ws):
        with pytest.raises(ValueError, match="kind"):
            _paths.model_loss_path(ws, "default", "lateral", 0, "test")

    def test_model_loss_path_empty_kind_raises(self, ws):
        with pytest.raises(ValueError):
            _paths.model_loss_path(ws, "default", "lateral", 0, "")

    def test_model_training_config_path(self, ws):
        p = _paths.model_training_config_path(ws, "default", "lateral", 0)
        assert p.name == "weights_training_config.json"

    def test_model_dir_is_child_of_round_dir(self, ws):
        """Phase B.2: model_dir is now a CHILD of round_dir (not a sibling).
        This is the structural co-location that makes each round dir
        self-contained.  Pre-B.2, they were siblings under separate
        `rounds/` and `models/` subtrees."""
        data_dir = _paths.round_dir(ws, "default", "lateral", 3)
        model_dir = _paths.model_dir(ws, "default", "lateral", 3)
        assert model_dir.parent == data_dir, \
            f"Expected model_dir under round_dir; model={model_dir}, round={data_dir}"
        assert model_dir.name == "model"
        assert data_dir.name == "round_03"


# ============================================================================
# validate_run_id
# ============================================================================

class TestValidateRunId:
    def test_default_valid(self):
        _paths.validate_run_id("default")  # no exception

    def test_alphanumeric_valid(self):
        _paths.validate_run_id("nuclei_v1")
        _paths.validate_run_id("membrane")
        _paths.validate_run_id("a")
        _paths.validate_run_id("Z")
        _paths.validate_run_id("123")

    def test_dash_valid(self):
        _paths.validate_run_id("pilot-2026")
        _paths.validate_run_id("a-b-c")

    def test_underscore_in_middle_valid(self):
        _paths.validate_run_id("nuclei_cellpose_v1")

    def test_starts_with_underscore_invalid(self):
        with pytest.raises(ValueError):
            _paths.validate_run_id("_hidden")

    def test_starts_with_dash_invalid(self):
        with pytest.raises(ValueError):
            _paths.validate_run_id("-test")

    def test_empty_invalid(self):
        with pytest.raises(ValueError, match="empty"):
            _paths.validate_run_id("")

    def test_too_long_invalid(self):
        long = "a" * 65
        with pytest.raises(ValueError, match="64"):
            _paths.validate_run_id(long)

    def test_exactly_64_chars_valid(self):
        # boundary: max length is 64
        _paths.validate_run_id("a" * 64)

    def test_path_traversal_invalid(self):
        with pytest.raises(ValueError):
            _paths.validate_run_id("../escape")
        with pytest.raises(ValueError):
            _paths.validate_run_id("a/b")
        with pytest.raises(ValueError):
            _paths.validate_run_id("a\\b")

    def test_dot_invalid(self):
        with pytest.raises(ValueError):
            _paths.validate_run_id("a.b")

    def test_space_invalid(self):
        with pytest.raises(ValueError):
            _paths.validate_run_id("a b")

    def test_special_chars_invalid(self):
        for invalid in ("a@b", "a#b", "a:b", "a;b", "a*b"):
            with pytest.raises(ValueError):
                _paths.validate_run_id(invalid)

    def test_non_str_invalid(self):
        with pytest.raises(ValueError, match="str"):
            _paths.validate_run_id(123)
        with pytest.raises(ValueError):
            _paths.validate_run_id(None)


# ============================================================================
# parse_slice_label
# ============================================================================

class TestParseSliceLabel:
    def test_simple_label(self):
        assert _paths.parse_slice_label("SAMPLE01_XY_z0042") == ("SAMPLE01", "XY", 42)

    def test_underscored_stack_id(self):
        # Single underscore in stack_id (closes STEP2_KNOWN_BUGS )
        assert _paths.parse_slice_label("my_patch_v2_XY_z0001") == ("my_patch_v2", "XY", 1)

    def test_deeply_underscored_stack_id(self):
        assert _paths.parse_slice_label("a_b_c_d_e_XZ_z0010") == ("a_b_c_d_e", "XZ", 10)

    def test_spaces_in_stack_id(self):
        # Adversarial: spaces in stack name (uncommon but should not crash)
        assert _paths.parse_slice_label("weird name with spaces_YZ_z0007") == (
            "weird name with spaces", "YZ", 7
        )

    def test_all_three_views(self):
        for view in ("XY", "XZ", "YZ"):
            stack, parsed_view, idx = _paths.parse_slice_label(f"s_{view}_z0001")
            assert parsed_view == view

    def test_with_img_suffix(self):
        # filename stem may include _img suffix
        assert _paths.parse_slice_label("SAMPLE01_XY_z0042_img") == ("SAMPLE01", "XY", 42)

    def test_with_mask_suffix(self):
        assert _paths.parse_slice_label("SAMPLE01_XY_z0042_mask") == ("SAMPLE01", "XY", 42)

    def test_with_pred_mask_suffix(self):
        # NOTE: _pred_mask contains _mask, so order matters in stripping
        assert _paths.parse_slice_label("SAMPLE01_XY_z0042_pred_mask") == ("SAMPLE01", "XY", 42)

    def test_no_zero_padding_still_works(self):
        # zero-padding is a writer convention; the parser accepts any digit count
        assert _paths.parse_slice_label("s_XY_z42") == ("s", "XY", 42)

    def test_invalid_view_raises(self):
        with pytest.raises(ValueError, match="pattern"):
            _paths.parse_slice_label("SAMPLE01_XX_z0042")

    def test_lowercase_view_invalid(self):
        with pytest.raises(ValueError):
            _paths.parse_slice_label("SAMPLE01_xy_z0042")

    def test_no_z_prefix_raises(self):
        with pytest.raises(ValueError):
            _paths.parse_slice_label("SAMPLE01_XY_42")

    def test_non_digit_idx_raises(self):
        with pytest.raises(ValueError):
            _paths.parse_slice_label("SAMPLE01_XY_z00abc")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _paths.parse_slice_label("")

    def test_non_str_raises(self):
        with pytest.raises(ValueError):
            _paths.parse_slice_label(None)
        with pytest.raises(ValueError):
            _paths.parse_slice_label(42)

    def test_no_stack_id_raises(self):
        # "_XY_z0042" has empty stack_id, regex .+ requires non-empty
        with pytest.raises(ValueError):
            _paths.parse_slice_label("_XY_z0042")


# ============================================================================
# bucket_for_view
# ============================================================================

class TestBucketForView:
    def test_xy_to_lateral(self):
        assert _paths.bucket_for_view("XY") == "lateral"

    def test_xz_to_axial(self):
        assert _paths.bucket_for_view("XZ") == "axial"

    def test_yz_to_axial(self):
        assert _paths.bucket_for_view("YZ") == "axial"

    def test_invalid_view_raises(self):
        with pytest.raises(ValueError):
            _paths.bucket_for_view("XX")

    def test_lowercase_invalid(self):
        with pytest.raises(ValueError):
            _paths.bucket_for_view("xy")


# ============================================================================
# Legacy path aliases
# ============================================================================

class TestLegacyPaths:
    def test_legacy_training_data_root(self, ws):
        assert _paths.legacy_training_data_root(ws) == ws / "training_data"

    def test_legacy_round_dir(self, ws):
        assert _paths.legacy_round_dir(ws, 0) == ws / "training_data" / "round_00"

    def test_legacy_round_dir_zero_padded(self, ws):
        assert _paths.legacy_round_dir(ws, 7).name == "round_07"

    def test_legacy_accepted_pairs_spatial(self, ws):
        expected = ws / "training_data" / "round_00" / "accepted_pairs" / "spatial"
        assert _paths.legacy_round_accepted_pairs_dir(ws, 0, "spatial") == expected

    def test_legacy_accepted_pairs_axial(self, ws):
        expected = ws / "training_data" / "round_00" / "accepted_pairs" / "axial"
        assert _paths.legacy_round_accepted_pairs_dir(ws, 0, "axial") == expected

    def test_legacy_accepted_pairs_rejects_lateral(self, ws):
        # 'lateral' is the NEW name; not valid for the legacy helper
        with pytest.raises(ValueError, match="legacy_bucket"):
            _paths.legacy_round_accepted_pairs_dir(ws, 0, "lateral")

    def test_legacy_staged_pairs_spatial(self, ws):
        expected = ws / "training_data" / "round_00" / "staged_pairs" / "spatial"
        assert _paths.legacy_round_staged_pairs_dir(ws, 0, "spatial") == expected

    def test_legacy_models_root(self, ws):
        assert _paths.legacy_models_root(ws) == ws / "models"

    def test_legacy_round_models_dir(self, ws):
        assert _paths.legacy_round_models_dir(ws, 5) == ws / "models" / "round_05"

    def test_legacy_predictions_root(self, ws):
        assert _paths.legacy_predictions_root(ws) == ws / "predictions"

    def test_legacy_round_predictions_dir(self, ws):
        assert _paths.legacy_round_predictions_dir(ws, 2) == ws / "predictions" / "round_02"

    def test_legacy_artifacts_models_root(self, ws):
        assert _paths.legacy_artifacts_models_root(ws) == ws / "artifacts" / "models"

    def test_legacy_artifacts_inference_root(self, ws):
        assert _paths.legacy_artifacts_inference_root(ws) == ws / "artifacts" / "inference"

    def test_legacy_artifacts_benchmarks_root(self, ws):
        assert _paths.legacy_artifacts_benchmarks_root(ws) == ws / "artifacts" / "benchmarks"


# ============================================================================
# Layout detection
# ============================================================================

class TestLayoutDetection:
    def test_unknown_empty_workspace(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        assert _paths.detect_layout_version(ws) == _paths.LAYOUT_VERSION_UNKNOWN
        assert not _paths.is_mixed_layout(ws)

    def test_unknown_nonexistent_workspace(self, tmp_path):
        ws = tmp_path / "nonexistent"
        # No .mkdir(); workspace doesn't exist
        assert _paths.detect_layout_version(ws) == _paths.LAYOUT_VERSION_UNKNOWN

    def test_legacy_training_data_present(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "training_data" / "round_00").mkdir(parents=True)
        assert _paths.detect_layout_version(ws) == _paths.LAYOUT_VERSION_LEGACY

    def test_legacy_models_present(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "models" / "round_00").mkdir(parents=True)
        assert _paths.detect_layout_version(ws) == _paths.LAYOUT_VERSION_LEGACY

    def test_legacy_models_without_round_dirs_not_detected(self, tmp_path):
        # If models/ exists but no round_* subdirs, don't consider it legacy.
        # (e.g. user created an empty models/ for some other reason)
        ws = tmp_path / "ws"
        (ws / "models").mkdir(parents=True)
        assert _paths.detect_layout_version(ws) == _paths.LAYOUT_VERSION_UNKNOWN

    def test_new_layout_detected(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)
        assert _paths.detect_layout_version(ws) == _paths.LAYOUT_VERSION_NEW

    def test_mixed_layout_detected(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "training_data" / "round_00").mkdir(parents=True)
        (ws / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)
        assert _paths.detect_layout_version(ws) == _paths.LAYOUT_VERSION_MIXED
        assert _paths.is_mixed_layout(ws)

    def test_mixed_via_models_plus_new(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "models" / "round_00").mkdir(parents=True)
        (ws / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)
        assert _paths.detect_layout_version(ws) == _paths.LAYOUT_VERSION_MIXED

    def test_is_mixed_layout_true_on_mixed(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "training_data" / "round_00").mkdir(parents=True)
        (ws / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)
        assert _paths.is_mixed_layout(ws) is True

    def test_is_mixed_layout_false_on_new_only(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)
        assert _paths.is_mixed_layout(ws) is False

    def test_is_mixed_layout_false_on_legacy_only(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "training_data" / "round_00").mkdir(parents=True)
        assert _paths.is_mixed_layout(ws) is False

    def test_is_mixed_layout_false_on_unknown(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        assert _paths.is_mixed_layout(ws) is False


# ============================================================================
# Unified state loader (v3-first / v2-fallback)
# ============================================================================

class TestLoadStep2StateFlat:
    def test_returns_none_when_no_state_file(self, ws):
        ws.mkdir(parents=True, exist_ok=True)
        assert _paths.load_step2_state_flat(ws) is None

    def test_reads_legacy_v2_state(self, ws):
        import json
        ws.mkdir(parents=True, exist_ok=True)
        legacy = ws / "hitl_state.json"
        legacy.write_text(json.dumps({
            "schema_version": 2,
            "current_round": 3,
            "sample_records": {"slice_xy_z0": {"status": "accepted"}},
            "selected_view": "XY",
        }), encoding="utf-8")

        state = _paths.load_step2_state_flat(ws)
        assert state is not None
        assert state["schema_version"] == 2
        assert state["current_round"] == 3
        assert state["sample_records"] == {"slice_xy_z0": {"status": "accepted"}}
        assert state["selected_view"] == "XY"

    def test_reads_v3_state_and_unwraps_default_run(self, ws):
        import json
        ws.mkdir(parents=True, exist_ok=True)
        v3 = ws / "step2_state.json"
        v3.write_text(json.dumps({
            "schema_version": 3,
            "active_run_id": "default",
            "selected_view": "XY",          # workspace-level
            "view_target": "lateral",       # workspace-level
            "runs": {
                "default": {
                    "display_name": "Default",      # run-manifest-only
                    "target": None,                  # run-manifest-only
                    "backend": "cellpose",           # run-manifest-only
                    "current_round": 5,              # per-run -> should flatten
                    "sample_records": {"slice_a": {"status": "accepted"}},
                },
            },
        }), encoding="utf-8")

        state = _paths.load_step2_state_flat(ws)
        assert state is not None
        assert state["schema_version"] == 3
        # Workspace-level keys preserved
        assert state["selected_view"] == "XY"
        assert state["view_target"] == "lateral"
        # Per-run keys flattened from runs.default
        assert state["current_round"] == 5
        assert state["sample_records"] == {"slice_a": {"status": "accepted"}}
        # Run-manifest-only keys NOT leaked into flat
        assert "display_name" not in state
        assert "target" not in state
        assert "backend" not in state

    def test_prefers_v3_when_both_exist(self, ws):
        # Mixed-layout workspace: both files present.  v3 wins.
        import json
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "hitl_state.json").write_text(json.dumps({
            "schema_version": 2, "current_round": 99,
        }), encoding="utf-8")
        (ws / "step2_state.json").write_text(json.dumps({
            "schema_version": 3, "active_run_id": "default",
            "runs": {"default": {"current_round": 5}},
        }), encoding="utf-8")
        state = _paths.load_step2_state_flat(ws)
        assert state["schema_version"] == 3
        assert state["current_round"] == 5  # v3 wins, NOT 99

    def test_v3_with_missing_active_run_falls_back_to_default(self, ws):
        import json
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "step2_state.json").write_text(json.dumps({
            "schema_version": 3,
            # active_run_id missing
            "runs": {"default": {"current_round": 2}},
        }), encoding="utf-8")
        state = _paths.load_step2_state_flat(ws)
        assert state["current_round"] == 2

    def test_v3_with_active_run_pointing_at_nonexistent_falls_back(self, ws):
        import json
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "step2_state.json").write_text(json.dumps({
            "schema_version": 3,
            "active_run_id": "nonexistent_run",
            "runs": {"default": {"current_round": 7}},
        }), encoding="utf-8")
        state = _paths.load_step2_state_flat(ws, active_run_id="nonexistent_run")
        assert state["current_round"] == 7  # fell back to default

    def test_corrupt_v3_falls_back_to_v2(self, ws):
        import json
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "step2_state.json").write_text("{ not json", encoding="utf-8")
        (ws / "hitl_state.json").write_text(json.dumps({
            "schema_version": 2, "current_round": 4,
        }), encoding="utf-8")
        state = _paths.load_step2_state_flat(ws)
        assert state is not None
        assert state["current_round"] == 4

    def test_is_new_layout_state_true_after_migration(self, ws):
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "step2_state.json").write_text("{}", encoding="utf-8")
        assert _paths.is_new_layout_state(ws) is True

    def test_is_new_layout_state_false_when_only_legacy(self, ws):
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "hitl_state.json").write_text("{}", encoding="utf-8")
        assert _paths.is_new_layout_state(ws) is False

    def test_is_new_layout_state_false_when_empty(self, ws):
        ws.mkdir(parents=True, exist_ok=True)
        assert _paths.is_new_layout_state(ws) is False


# ============================================================================
# Public API surface (__all__)
# ============================================================================

class TestPublicApi:
    def test_all_export_list_present(self):
        assert hasattr(_paths, "__all__")
        assert isinstance(_paths.__all__, list)
        assert len(_paths.__all__) > 0

    def test_all_listed_symbols_exist(self):
        # Every name in __all__ must be a real attribute of the module
        for name in _paths.__all__:
            assert hasattr(_paths, name), (
                f"__all__ lists {name!r} but module has no such attribute"
            )

    def test_private_helpers_not_exported(self):
        # Internal helpers should NOT be in __all__
        for private in ("_ws", "_validate_bucket", "_round_segment",
                        "_RUN_ID_PATTERN", "_SLICE_LABEL_RE", "_WS"):
            assert private not in _paths.__all__, (
                f"{private!r} is internal and should not be in __all__"
            )

    def test_parse_slice_label_is_public(self):
        # Explicit decision: public, not private (see module docstring)
        assert "parse_slice_label" in _paths.__all__
        assert "_parse_slice_label" not in _paths.__all__


# ============================================================================
# Constants
# ============================================================================

class TestConstants:
    def test_default_run_id(self):
        assert _paths.DEFAULT_RUN_ID == "default"

    def test_valid_buckets(self):
        # Order matters for migration UI consistency
        assert _paths.VALID_BUCKETS == ("lateral", "axial")

    def test_valid_views(self):
        assert _paths.VALID_VIEWS == ("XY", "XZ", "YZ")

    def test_state_filename(self):
        assert _paths.STEP2_STATE_FILENAME == "step2_state.json"

    def test_legacy_state_filename(self):
        assert _paths.LEGACY_STATE_FILENAME == "hitl_state.json"

    def test_step2_artifacts_subdir(self):
        assert _paths.STEP2_ARTIFACTS_SUBDIR == "step2"

    def test_run_id_max_length(self):
        assert _paths.RUN_ID_MAX_LENGTH == 64

    def test_weights_filename_constants(self):
        # Cellpose-driven filenames (no rename in lift); verify they
        # match what Cellpose actually writes with model_name="weights".
        # 2026-05-26 audit C-4 fix: Cellpose writes _val_loss.npy
        # (see cellpose_trainer.py:237), not _validation_loss.npy.
        assert _paths.WEIGHTS_FILENAME == "weights"
        assert _paths.TRAIN_LOSS_FILENAME == "weights_train_loss.npy"
        assert _paths.VAL_LOSS_FILENAME == "weights_val_loss.npy"
        assert _paths.TRAINING_CONFIG_FILENAME == "weights_training_config.json"

    def test_layout_versions_distinct(self):
        # Sentinels must be unique
        versions = {
            _paths.LAYOUT_VERSION_UNKNOWN,
            _paths.LAYOUT_VERSION_LEGACY,
            _paths.LAYOUT_VERSION_NEW,
            _paths.LAYOUT_VERSION_MIXED,
        }
        assert len(versions) == 4

    def test_current_layout_is_new(self):
        assert _paths.CURRENT_LAYOUT_VERSION == _paths.LAYOUT_VERSION_NEW

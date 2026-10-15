"""Tests for ``_import_pairs`` — Phase 8d cherry-pick + copy logic.

Pure-Python (no Qt) tests for the data layer.  Dialog UX is manual.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
from napari_epicycle.hitl_training_annotation._import_pairs import (
    ImportError_,
    copy_pairs_to_target_round,
    list_source_pairs,
    list_source_run_rounds,
)
from napari_epicycle.hitl_training_annotation._run_management import (
    create_run_on_disk,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _plant_pair(workspace, run_id, bucket, round_idx, label, img_content=b"img", mask_content=b"mask"):
    """Helper: create an accepted pair on disk."""
    accepted = _step2_paths.round_accepted_dir(workspace, run_id, bucket, round_idx)
    accepted.mkdir(parents=True, exist_ok=True)
    (accepted / f"{label}_img.tif").write_bytes(img_content)
    (accepted / f"{label}_mask.tif").write_bytes(mask_content)


# ----------------------------------------------------------------------------
# list_source_pairs
# ----------------------------------------------------------------------------

class TestListSourcePairs:

    def test_empty_dir_returns_empty(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        assert list_source_pairs(tmp_path, "src", "lateral", 0) == []

    def test_finds_paired_files(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        _plant_pair(tmp_path, "src", "lateral", 0, "patch_1_XY_z000")
        _plant_pair(tmp_path, "src", "lateral", 0, "patch_1_XY_z001")

        pairs = list_source_pairs(tmp_path, "src", "lateral", 0)
        assert len(pairs) == 2
        labels = {p["label"] for p in pairs}
        assert labels == {"patch_1_XY_z000", "patch_1_XY_z001"}

    def test_skips_orphan_img_without_mask(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        accepted = _step2_paths.round_accepted_dir(tmp_path, "src", "lateral", 0)
        accepted.mkdir(parents=True)
        (accepted / "orphan_img.tif").write_bytes(b"x")
        # no matching _mask.tif

        pairs = list_source_pairs(tmp_path, "src", "lateral", 0)
        assert pairs == []

    def test_skips_non_img_files(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        accepted = _step2_paths.round_accepted_dir(tmp_path, "src", "lateral", 0)
        accepted.mkdir(parents=True)
        (accepted / "readme.txt").write_bytes(b"hi")
        (accepted / "patch_img.tif").write_bytes(b"x")
        (accepted / "patch_mask.tif").write_bytes(b"y")

        pairs = list_source_pairs(tmp_path, "src", "lateral", 0)
        assert len(pairs) == 1
        assert pairs[0]["label"] == "patch"

    def test_returns_size_and_mtime(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        _plant_pair(tmp_path, "src", "lateral", 0, "p1",
                    img_content=b"x" * 1000, mask_content=b"y" * 500)
        pairs = list_source_pairs(tmp_path, "src", "lateral", 0)
        assert pairs[0]["size_bytes"] == 1500
        assert pairs[0]["img_mtime_iso"] is not None

    def test_bad_bucket_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="source_bucket"):
            list_source_pairs(tmp_path, "src", "diagonal", 0)


# ----------------------------------------------------------------------------
# Status lookup regression — label format canonicalization (2026-05-25 fix)
# ----------------------------------------------------------------------------

class TestStatusLookupCanonicalization:
    """Regression tests for the user-reported bug:

    "import pairs still shows accepted status on thing I wanna import
    which I already stamped as finalized"

    Root cause: sample_records keys are stored in non-padded format
    ('patch_XY_42') by several controller paths (`f"{patch}_{view}_{idx}"`),
    but disk filenames are written in z-prefixed padded format
    ('patch_XY_z0042') by `_persist_curated_training_pair`.  The old
    string-keyed status lookup missed every cross-format pair, falling
    back to default "accepted" — silently demoting finalized status in
    the cherry-pick dialog.

    Fix canonicalizes both sides to `(stack_id, view, idx)` tuples.
    """

    def _write_state_json(self, workspace, run_id, sample_records_dict):
        """Helper: write a v3 state JSON with custom sample_records."""
        from napari_epicycle.hitl_training_annotation._migrate_step2_layout import (
            wrap_state_schema_v2_to_v3,
        )
        v2 = {
            "schema_version": 2,
            "current_round": 0,
            "current_round_by_view": {"lateral": 0, "axial": 0},
            "training_data_by_round": {},
            "losses_by_round": {},
            "model_checkpoint_paths": {},
            "round_backend_info": {},
            "current_config": {},
            "selected_view": "XY",
            "view_target": "lateral",
            "axial_plane": "XZ",
            "sample_records": sample_records_dict,
            "active_curation_checkpoints": {"lateral": "", "axial": ""},
            "curation_model_source_mode": "auto",
            "imported_curation_model_path": "",
            "curation_pretrained_model_type": "cyto3",
            "training_mode": "cumulative",
            "auto_update_model_after_training": True,
            "loaded_stack_path": "",
            "loaded_patch_name": "",
        }
        v3 = wrap_state_schema_v2_to_v3(v2, active_run_id=run_id)
        state_path = _step2_paths.state_json_path(workspace)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(v3, f, indent=2)

    def test_disk_z_prefixed_label_matches_sample_records_legacy_key(self, tmp_path):
        """The smoking gun: sample_records has 'patch_XY_42' as the key
        with status='staged'; disk has 'patch_XY_z0042_img.tif'.  The
        lookup MUST return 'staged', not the default 'accepted'."""
        create_run_on_disk(tmp_path, "src", display_name="Src")
        # Disk: z-prefixed padded filename
        _plant_pair(tmp_path, "src", "lateral", 0, "patch_XY_z0042")
        # State JSON: legacy non-padded key with finalized status
        self._write_state_json(tmp_path, "src", {
            "patch_XY_42": {"status": "staged"},
        })
        pairs = list_source_pairs(tmp_path, "src", "lateral", 0)
        assert len(pairs) == 1
        assert pairs[0]["status"] == "staged", (
            f"Cross-format lookup failed: expected 'staged' but got "
            f"{pairs[0]['status']!r}.  This is the user-reported bug — "
            f"sample_records uses non-padded keys, disk uses z-prefixed."
        )

    def test_disk_legacy_label_matches_sample_records_z_prefixed_key(self, tmp_path):
        """Reverse: disk in legacy format, sample_records in z-prefixed.
        Less likely in practice but verifies the canonicalize is
        symmetric."""
        create_run_on_disk(tmp_path, "src", display_name="Src")
        _plant_pair(tmp_path, "src", "lateral", 0, "patch_XY_42")
        self._write_state_json(tmp_path, "src", {
            "patch_XY_z0042": {"status": "committed"},
        })
        pairs = list_source_pairs(tmp_path, "src", "lateral", 0)
        assert len(pairs) == 1
        assert pairs[0]["status"] == "committed"

    def test_same_format_both_sides_works(self, tmp_path):
        """Verify the fix didn't break the same-format case."""
        create_run_on_disk(tmp_path, "src", display_name="Src")
        _plant_pair(tmp_path, "src", "lateral", 0, "patch_XY_z0042")
        self._write_state_json(tmp_path, "src", {
            "patch_XY_z0042": {"status": "staged"},
        })
        pairs = list_source_pairs(tmp_path, "src", "lateral", 0)
        assert pairs[0]["status"] == "staged"

    def test_missing_status_defaults_to_accepted(self, tmp_path):
        """When sample_records has no entry for the disk label (e.g.
        a stray pair file from outside the normal flow), default to
        'accepted' — pair is at-least-accepted because it's on disk."""
        create_run_on_disk(tmp_path, "src", display_name="Src")
        _plant_pair(tmp_path, "src", "lateral", 0, "orphan_XY_z0001")
        self._write_state_json(tmp_path, "src", {})  # empty records
        pairs = list_source_pairs(tmp_path, "src", "lateral", 0)
        assert pairs[0]["status"] == "accepted"

    def test_canonicalize_handles_underscored_stack_id(self, tmp_path):
        """Stack IDs with underscores (common — patch names from
        original TIFF filenames).  Canonicalization must parse
        right-to-left so 'my_patch_v2_XY_z0042' -> ('my_patch_v2', 'XY', 42)."""
        from napari_epicycle.hitl_training_annotation._import_pairs import (
            _canonicalize_slice_label,
        )
        assert _canonicalize_slice_label("my_patch_v2_XY_z0042") == ("my_patch_v2", "XY", 42)
        assert _canonicalize_slice_label("my_patch_v2_XY_42") == ("my_patch_v2", "XY", 42)
        assert _canonicalize_slice_label("a_b_c_d_e_XZ_z0010") == ("a_b_c_d_e", "XZ", 10)

    def test_canonicalize_rejects_invalid(self, tmp_path):
        """Bad view, bad index, too few parts — all return None."""
        from napari_epicycle.hitl_training_annotation._import_pairs import (
            _canonicalize_slice_label,
        )
        assert _canonicalize_slice_label("only_two_parts") is None
        assert _canonicalize_slice_label("patch_BAD_z0042") is None  # bad view
        assert _canonicalize_slice_label("patch_XY_notanint") is None
        assert _canonicalize_slice_label("") is None
        assert _canonicalize_slice_label(None) is None

    def test_canonicalize_strips_filename_suffixes(self, tmp_path):
        """If caller passes a filename stem with _img / _mask / _pred_mask,
        the suffix is stripped before parsing."""
        from napari_epicycle.hitl_training_annotation._import_pairs import (
            _canonicalize_slice_label,
        )
        assert _canonicalize_slice_label("patch_XY_z0042_img") == ("patch", "XY", 42)
        assert _canonicalize_slice_label("patch_XY_z0042_mask") == ("patch", "XY", 42)
        assert _canonicalize_slice_label("patch_XY_z0042_pred_mask") == ("patch", "XY", 42)


# ----------------------------------------------------------------------------
# list_source_run_rounds
# ----------------------------------------------------------------------------

class TestListSourceRunRounds:

    def test_empty_run_returns_empty(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        assert list_source_run_rounds(tmp_path, "src") == []

    def test_finds_multiple_rounds_and_buckets(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        _plant_pair(tmp_path, "src", "lateral", 0, "p1")
        _plant_pair(tmp_path, "src", "lateral", 0, "p2")
        _plant_pair(tmp_path, "src", "lateral", 1, "p3")
        _plant_pair(tmp_path, "src", "axial", 0, "p4")

        rounds = list_source_run_rounds(tmp_path, "src")
        assert len(rounds) == 3  # 2 lat rounds + 1 ax round
        # Confirm pair counts
        rounds_dict = {(r["bucket"], r["round_idx"]): r["pair_count"] for r in rounds}
        assert rounds_dict[("lateral", 0)] == 2
        assert rounds_dict[("lateral", 1)] == 1
        assert rounds_dict[("axial", 0)] == 1

    def test_skips_empty_rounds(self, tmp_path):
        """A round dir exists but has no pairs → not in the list."""
        create_run_on_disk(tmp_path, "src", display_name="Src")
        # Create round_00 dir but no pairs
        accepted = _step2_paths.round_accepted_dir(tmp_path, "src", "lateral", 0)
        accepted.mkdir(parents=True)
        # Plant a pair in round_01 to confirm scanner works
        _plant_pair(tmp_path, "src", "lateral", 1, "p1")

        rounds = list_source_run_rounds(tmp_path, "src")
        assert len(rounds) == 1
        assert rounds[0]["round_idx"] == 1


# ----------------------------------------------------------------------------
# copy_pairs_to_target_round
# ----------------------------------------------------------------------------

class TestCopyPairsToTargetRound:

    def test_happy_path_copies_pairs(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        create_run_on_disk(tmp_path, "dst", display_name="Dst")
        _plant_pair(tmp_path, "src", "lateral", 0, "p1", img_content=b"img1")
        _plant_pair(tmp_path, "src", "lateral", 0, "p2", img_content=b"img2")

        result = copy_pairs_to_target_round(
            tmp_path,
            source_run_id="src", source_bucket="lateral", source_round_idx=0,
            target_run_id="dst", target_bucket="lateral", target_round_idx=0,
            selected_labels=["p1", "p2"],
        )
        assert result["copied"] == ["p1", "p2"]
        assert result["skipped"] == []
        assert result["errors"] == []

        # Verify files copied
        target_accepted = _step2_paths.round_accepted_dir(tmp_path, "dst", "lateral", 0)
        assert (target_accepted / "p1_img.tif").read_bytes() == b"img1"
        assert (target_accepted / "p2_img.tif").read_bytes() == b"img2"

    def test_provenance_written_to_target_metadata(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        create_run_on_disk(tmp_path, "dst", display_name="Dst")
        _plant_pair(tmp_path, "src", "lateral", 3, "patch_1_XY_z100")

        copy_pairs_to_target_round(
            tmp_path,
            source_run_id="src", source_bucket="lateral", source_round_idx=3,
            target_run_id="dst", target_bucket="lateral", target_round_idx=0,
            selected_labels=["patch_1_XY_z100"],
        )

        meta_path = _step2_paths.round_metadata_path(tmp_path, "dst", "lateral", 0)
        assert meta_path.is_file()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["schema_version"] == "1.1"
        samples = meta["samples"]
        assert len(samples) == 1
        s = samples[0]
        assert s["label_id"] == "patch_1_XY_z100"
        assert s["status"] == "accepted"
        prov = s["inherited_from_pair"]
        assert prov["source_run_id"] == "src"
        assert prov["source_bucket"] == "lateral"
        assert prov["source_round_id"] == "round_03"
        assert prov["source_label"] == "patch_1_XY_z100"

    def test_skips_collisions_by_default(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        create_run_on_disk(tmp_path, "dst", display_name="Dst")
        _plant_pair(tmp_path, "src", "lateral", 0, "p1", img_content=b"src_img")
        _plant_pair(tmp_path, "dst", "lateral", 0, "p1", img_content=b"dst_img")  # collision

        result = copy_pairs_to_target_round(
            tmp_path,
            source_run_id="src", source_bucket="lateral", source_round_idx=0,
            target_run_id="dst", target_bucket="lateral", target_round_idx=0,
            selected_labels=["p1"],
        )
        assert result["copied"] == []
        assert result["skipped"] == ["p1"]
        # Original target file should be untouched
        target_img = _step2_paths.round_accepted_dir(
            tmp_path, "dst", "lateral", 0
        ) / "p1_img.tif"
        assert target_img.read_bytes() == b"dst_img"

    def test_overwrite_collisions_flag(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        create_run_on_disk(tmp_path, "dst", display_name="Dst")
        _plant_pair(tmp_path, "src", "lateral", 0, "p1", img_content=b"src_img")
        _plant_pair(tmp_path, "dst", "lateral", 0, "p1", img_content=b"dst_img")

        result = copy_pairs_to_target_round(
            tmp_path,
            source_run_id="src", source_bucket="lateral", source_round_idx=0,
            target_run_id="dst", target_bucket="lateral", target_round_idx=0,
            selected_labels=["p1"],
            overwrite_collisions=True,
        )
        assert result["copied"] == ["p1"]
        target_img = _step2_paths.round_accepted_dir(
            tmp_path, "dst", "lateral", 0
        ) / "p1_img.tif"
        assert target_img.read_bytes() == b"src_img"

    def test_missing_label_in_source_reports_error(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        create_run_on_disk(tmp_path, "dst", display_name="Dst")
        _plant_pair(tmp_path, "src", "lateral", 0, "real_pair")

        result = copy_pairs_to_target_round(
            tmp_path,
            source_run_id="src", source_bucket="lateral", source_round_idx=0,
            target_run_id="dst", target_bucket="lateral", target_round_idx=0,
            selected_labels=["real_pair", "ghost_pair"],
        )
        assert result["copied"] == ["real_pair"]
        assert len(result["errors"]) == 1
        assert result["errors"][0][0] == "ghost_pair"
        assert "not found" in result["errors"][0][1]

    def test_same_source_target_rejected(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        with pytest.raises(ValueError, match="source and target are the same"):
            copy_pairs_to_target_round(
                tmp_path,
                source_run_id="src", source_bucket="lateral", source_round_idx=0,
                target_run_id="src", target_bucket="lateral", target_round_idx=0,
                selected_labels=[],
            )

    def test_missing_source_dir_raises(self, tmp_path):
        create_run_on_disk(tmp_path, "dst", display_name="Dst")
        with pytest.raises(ImportError_, match="source dir does not exist"):
            copy_pairs_to_target_round(
                tmp_path,
                source_run_id="ghost", source_bucket="lateral",
                source_round_idx=0,
                target_run_id="dst", target_bucket="lateral",
                target_round_idx=0,
                selected_labels=["x"],
            )

    def test_cross_bucket_copy_works(self, tmp_path):
        """Copy lateral pairs into axial bucket (valid op)."""
        create_run_on_disk(tmp_path, "src", display_name="Src")
        create_run_on_disk(tmp_path, "dst", display_name="Dst")
        _plant_pair(tmp_path, "src", "lateral", 0, "p1")

        result = copy_pairs_to_target_round(
            tmp_path,
            source_run_id="src", source_bucket="lateral", source_round_idx=0,
            target_run_id="dst", target_bucket="axial", target_round_idx=0,
            selected_labels=["p1"],
        )
        assert result["copied"] == ["p1"]

    def test_empty_selected_labels_writes_no_metadata(self, tmp_path):
        create_run_on_disk(tmp_path, "src", display_name="Src")
        create_run_on_disk(tmp_path, "dst", display_name="Dst")
        _plant_pair(tmp_path, "src", "lateral", 0, "p1")

        result = copy_pairs_to_target_round(
            tmp_path,
            source_run_id="src", source_bucket="lateral", source_round_idx=0,
            target_run_id="dst", target_bucket="lateral", target_round_idx=0,
            selected_labels=[],
        )
        assert result["copied"] == []
        meta_path = _step2_paths.round_metadata_path(tmp_path, "dst", "lateral", 0)
        assert not meta_path.exists(), "no metadata should be written when no pairs copied"

    def test_provenance_appends_to_existing_metadata(self, tmp_path):
        """If target round_metadata.json already has samples, append don't replace."""
        create_run_on_disk(tmp_path, "src", display_name="Src")
        create_run_on_disk(tmp_path, "dst", display_name="Dst")
        _plant_pair(tmp_path, "src", "lateral", 0, "imported_p1")

        # Pre-populate dst metadata with a native sample
        meta_path = _step2_paths.round_metadata_path(tmp_path, "dst", "lateral", 0)
        meta_path.parent.mkdir(parents=True)
        meta_path.write_text(json.dumps({
            "schema_version": "1.0",
            "round_id": "round_00",
            "samples": [{"label_id": "native_p1", "status": "accepted"}],
        }), encoding="utf-8")

        copy_pairs_to_target_round(
            tmp_path,
            source_run_id="src", source_bucket="lateral", source_round_idx=0,
            target_run_id="dst", target_bucket="lateral", target_round_idx=0,
            selected_labels=["imported_p1"],
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = {s["label_id"] for s in meta["samples"]}
        assert labels == {"native_p1", "imported_p1"}
        assert meta["schema_version"] == "1.1"  # bumped on import

"""Tests for Phase 8f — external pair import (pure data layer).

Qt dialog is manual-tested.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
from napari_epicycle.hitl_training_annotation._import_external_pairs import (
    ExternalImportError,
    copy_external_pairs_to_target_round,
    enumerate_pairs,
    suggest_pair_convention,
    validate_pair,
)
from napari_epicycle.hitl_training_annotation._run_management import (
    create_run_on_disk,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _make_pair_files(folder, basename, img_suffix="_img.tif", mask_suffix="_mask.tif",
                      img_shape=(64, 64), mask_shape=(64, 64)):
    """Plant a valid image + mask pair in folder."""
    folder.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, size=img_shape, dtype=np.uint8)
    mask = np.zeros(mask_shape, dtype=np.uint16)
    mask[10:30, 10:30] = 1
    mask[40:60, 40:60] = 2
    tifffile.imwrite(str(folder / f"{basename}{img_suffix}"), img)
    tifffile.imwrite(str(folder / f"{basename}{mask_suffix}"), mask)


# ----------------------------------------------------------------------------
# enumerate_pairs
# ----------------------------------------------------------------------------

class TestEnumeratePairs:

    def test_empty_folder_returns_empty(self, tmp_path):
        folder = tmp_path / "src"
        folder.mkdir()
        pairs, oi, om = enumerate_pairs(folder, "_img.tif", "_mask.tif")
        assert pairs == []
        assert oi == []
        assert om == []

    def test_happy_pair_detection(self, tmp_path):
        folder = tmp_path / "src"
        _make_pair_files(folder, "patch_1")
        _make_pair_files(folder, "patch_2")
        _make_pair_files(folder, "patch_3")
        pairs, oi, om = enumerate_pairs(folder, "_img.tif", "_mask.tif")
        assert len(pairs) == 3
        assert oi == []
        assert om == []
        basenames = [p["basename"] for p in pairs]
        assert basenames == ["patch_1", "patch_2", "patch_3"]

    def test_orphan_image(self, tmp_path):
        folder = tmp_path / "src"
        folder.mkdir()
        # img only, no mask
        img = np.zeros((10, 10), dtype=np.uint8)
        tifffile.imwrite(str(folder / "lonely_img.tif"), img)
        pairs, oi, om = enumerate_pairs(folder, "_img.tif", "_mask.tif")
        assert pairs == []
        assert oi == ["lonely_img.tif"]
        assert om == []

    def test_orphan_mask(self, tmp_path):
        folder = tmp_path / "src"
        folder.mkdir()
        mask = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(str(folder / "lonely_mask.tif"), mask)
        pairs, oi, om = enumerate_pairs(folder, "_img.tif", "_mask.tif")
        assert pairs == []
        assert oi == []
        assert om == ["lonely_mask.tif"]

    def test_custom_suffix(self, tmp_path):
        folder = tmp_path / "src"
        _make_pair_files(folder, "patch_1",
                          img_suffix="_raw.tif", mask_suffix="_gt.tif")
        pairs, oi, om = enumerate_pairs(folder, "_raw.tif", "_gt.tif")
        assert len(pairs) == 1
        assert pairs[0]["basename"] == "patch_1"

    def test_empty_suffix_rejected(self, tmp_path):
        folder = tmp_path / "src"
        folder.mkdir()
        with pytest.raises(ValueError, match="img_suffix"):
            enumerate_pairs(folder, "", "_mask.tif")
        with pytest.raises(ValueError, match="mask_suffix"):
            enumerate_pairs(folder, "_img.tif", "")

    def test_identical_suffix_rejected(self, tmp_path):
        folder = tmp_path / "src"
        folder.mkdir()
        with pytest.raises(ValueError, match="must differ"):
            enumerate_pairs(folder, "_x.tif", "_x.tif")

    def test_overlapping_suffix_pairs_correctly(self, tmp_path):
        """If img_suf is shorter (e.g. '.tif') + mask_suf is longer
        ('_mask.tif'), the longer one wins on a file that could match
        both — prevents double-counting."""
        folder = tmp_path / "src"
        folder.mkdir()
        img = np.zeros((10, 10), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint16)
        # patch_a.tif (image) + patch_a_mask.tif (mask)
        tifffile.imwrite(str(folder / "patch_a.tif"), img)
        tifffile.imwrite(str(folder / "patch_a_mask.tif"), mask)
        pairs, oi, om = enumerate_pairs(folder, ".tif", "_mask.tif")
        assert len(pairs) == 1
        assert pairs[0]["basename"] == "patch_a"

    def test_missing_folder_raises(self, tmp_path):
        with pytest.raises(ExternalImportError, match="not found"):
            enumerate_pairs(tmp_path / "ghost", "_img.tif", "_mask.tif")

    def test_returns_file_sizes(self, tmp_path):
        folder = tmp_path / "src"
        _make_pair_files(folder, "patch_1")
        pairs, _, _ = enumerate_pairs(folder, "_img.tif", "_mask.tif")
        assert pairs[0]["img_size"] > 0
        assert pairs[0]["mask_size"] > 0


# ----------------------------------------------------------------------------
# suggest_pair_convention
# ----------------------------------------------------------------------------

class TestSuggestPairConvention:

    def test_epicycle_convention(self, tmp_path):
        folder = tmp_path / "src"
        for n in range(3):
            _make_pair_files(folder, f"p{n}")
        img_suf, mask_suf = suggest_pair_convention(folder)
        assert img_suf == "_img.tif"
        assert mask_suf == "_mask.tif"

    def test_alternate_convention(self, tmp_path):
        folder = tmp_path / "src"
        for n in range(3):
            _make_pair_files(folder, f"p{n}",
                              img_suffix="_image.tif", mask_suffix="_label.tif")
        img_suf, mask_suf = suggest_pair_convention(folder)
        assert img_suf == "_image.tif"
        assert mask_suf == "_label.tif"

    def test_missing_folder_returns_default(self, tmp_path):
        img_suf, mask_suf = suggest_pair_convention(tmp_path / "ghost")
        assert img_suf == "_img.tif"
        assert mask_suf == "_mask.tif"

    def test_empty_folder_returns_default(self, tmp_path):
        folder = tmp_path / "src"
        folder.mkdir()
        img_suf, mask_suf = suggest_pair_convention(folder)
        assert img_suf == "_img.tif"
        assert mask_suf == "_mask.tif"


# ----------------------------------------------------------------------------
# validate_pair
# ----------------------------------------------------------------------------

class TestValidatePair:

    def test_valid_pair(self, tmp_path):
        folder = tmp_path / "src"
        _make_pair_files(folder, "patch_1")
        ok, msg = validate_pair(folder / "patch_1_img.tif",
                                  folder / "patch_1_mask.tif")
        assert ok is True

    def test_missing_image_rejected(self, tmp_path):
        folder = tmp_path / "src"
        folder.mkdir()
        mask = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(str(folder / "x_mask.tif"), mask)
        ok, msg = validate_pair(folder / "x_img.tif", folder / "x_mask.tif")
        assert ok is False
        assert "image file missing" in msg

    def test_missing_mask_rejected(self, tmp_path):
        folder = tmp_path / "src"
        folder.mkdir()
        img = np.zeros((10, 10), dtype=np.uint8)
        tifffile.imwrite(str(folder / "x_img.tif"), img)
        ok, msg = validate_pair(folder / "x_img.tif", folder / "x_mask.tif")
        assert ok is False
        assert "mask file missing" in msg

    def test_shape_mismatch_rejected(self, tmp_path):
        folder = tmp_path / "src"
        folder.mkdir()
        img = np.zeros((64, 64), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint16)  # different spatial shape
        tifffile.imwrite(str(folder / "x_img.tif"), img)
        tifffile.imwrite(str(folder / "x_mask.tif"), mask)
        ok, msg = validate_pair(folder / "x_img.tif", folder / "x_mask.tif")
        assert ok is False
        assert "spatial shapes differ" in msg

    def test_3d_mask_rejected(self, tmp_path):
        folder = tmp_path / "src"
        folder.mkdir()
        img = np.zeros((64, 64), dtype=np.uint8)
        mask = np.zeros((3, 64, 64), dtype=np.uint16)  # 3D, not allowed for masks
        tifffile.imwrite(str(folder / "x_img.tif"), img)
        tifffile.imwrite(str(folder / "x_mask.tif"), mask)
        ok, msg = validate_pair(folder / "x_img.tif", folder / "x_mask.tif")
        assert ok is False


# ----------------------------------------------------------------------------
# copy_external_pairs_to_target_round
# ----------------------------------------------------------------------------

class TestCopyExternalPairs:

    def test_happy_path(self, tmp_path):
        # Source folder with 2 external pairs
        src_folder = tmp_path / "external"
        _make_pair_files(src_folder, "ext_patch_a")
        _make_pair_files(src_folder, "ext_patch_b")
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")

        # Target run
        create_run_on_disk(tmp_path, "default", display_name="Default")

        outcome = copy_external_pairs_to_target_round(
            tmp_path,
            target_run_id="default",
            target_bucket="lateral",
            target_round_idx=0,
            pair_records=pairs,
            relabel_basename=True,
        )
        assert len(outcome["copied"]) == 2
        assert outcome["skipped"] == []
        assert outcome["errors"] == []

        # Files actually on disk in target
        target_accepted = _step2_paths.round_accepted_dir(
            tmp_path, "default", "lateral", 0
        )
        copied_files = [p.name for p in target_accepted.iterdir()]
        # 4 files (2 pairs × img + mask)
        assert len(copied_files) == 4

    def test_provenance_written_to_metadata(self, tmp_path):
        src_folder = tmp_path / "external"
        _make_pair_files(src_folder, "ext_a")
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        create_run_on_disk(tmp_path, "default", display_name="Default")

        copy_external_pairs_to_target_round(
            tmp_path,
            target_run_id="default", target_bucket="lateral",
            target_round_idx=0,
            pair_records=pairs,
            relabel_basename=True,
        )
        meta_path = _step2_paths.round_metadata_path(
            tmp_path, "default", "lateral", 0
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["schema_version"] == "1.1"
        sample = meta["samples"][0]
        ext = sample["inherited_from_external"]
        assert "source_path" in ext
        assert "source_filename" in ext
        assert ext["source_basename"] == "ext_a"
        assert sample["status"] == "accepted"

    def test_relabel_produces_parseable_labels(self, tmp_path):
        """With relabel_basename=True, labels match the {stack}_{view}_z{idx}
        convention so parse_slice_label works."""
        src_folder = tmp_path / "external"
        _make_pair_files(src_folder, "weird name with spaces!")
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        create_run_on_disk(tmp_path, "default", display_name="Default")

        outcome = copy_external_pairs_to_target_round(
            tmp_path,
            target_run_id="default", target_bucket="lateral",
            target_round_idx=0,
            pair_records=pairs,
            relabel_basename=True,
        )
        # The label should be parseable
        label = outcome["copied"][0]
        stack, view, idx = _step2_paths.parse_slice_label(label)
        assert view == "XY"  # lateral bucket → XY view
        assert idx == 0

    def test_skip_collisions_by_default(self, tmp_path):
        src_folder = tmp_path / "external"
        _make_pair_files(src_folder, "ext_a")
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        create_run_on_disk(tmp_path, "default", display_name="Default")

        # First import — succeeds
        first = copy_external_pairs_to_target_round(
            tmp_path, target_run_id="default", target_bucket="lateral",
            target_round_idx=0, pair_records=pairs, relabel_basename=True,
        )
        assert len(first["copied"]) == 1

        # Second import of the same pairs — should skip
        second = copy_external_pairs_to_target_round(
            tmp_path, target_run_id="default", target_bucket="lateral",
            target_round_idx=0, pair_records=pairs, relabel_basename=True,
        )
        assert second["copied"] == []
        assert len(second["skipped"]) == 1

    def test_overwrite_collisions_flag(self, tmp_path):
        src_folder = tmp_path / "external"
        _make_pair_files(src_folder, "ext_a")
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        create_run_on_disk(tmp_path, "default", display_name="Default")

        copy_external_pairs_to_target_round(
            tmp_path, target_run_id="default", target_bucket="lateral",
            target_round_idx=0, pair_records=pairs, relabel_basename=True,
        )
        # Now overwrite
        outcome = copy_external_pairs_to_target_round(
            tmp_path, target_run_id="default", target_bucket="lateral",
            target_round_idx=0, pair_records=pairs, relabel_basename=True,
            overwrite_collisions=True,
        )
        assert len(outcome["copied"]) == 1
        assert outcome["skipped"] == []

    def test_axial_bucket_view_defaults_to_xz(self, tmp_path):
        src_folder = tmp_path / "external"
        _make_pair_files(src_folder, "ext_a")
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        create_run_on_disk(tmp_path, "default", display_name="Default")

        outcome = copy_external_pairs_to_target_round(
            tmp_path, target_run_id="default", target_bucket="axial",
            target_round_idx=0, pair_records=pairs, relabel_basename=True,
        )
        label = outcome["copied"][0]
        _, view, _ = _step2_paths.parse_slice_label(label)
        assert view == "XZ"  # axial bucket → XZ default

    def test_bad_bucket_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="target_bucket"):
            copy_external_pairs_to_target_round(
                tmp_path, target_run_id="x", target_bucket="diagonal",
                target_round_idx=0, pair_records=[], relabel_basename=True,
            )

    def test_sanitization_does_not_collide_when_idx_differs(self, tmp_path):
        """Audit-clarification 2026-05-24: with relabel_basename=True,
        two basenames that sanitize equivalently (e.g., 'foo bar' +
        'foo-bar' → both stem to 'foo_bar') do NOT collide because the
        loop-index suffix (_z{idx:04d}) disambiguates them.  Both copy
        cleanly + get unique target labels.  The pre-loop dedup pass
        in copy_external_pairs_to_target_round catches the actual
        worst case (identical target labels) which only manifests with
        relabel_basename=False + duplicate basenames in pair_records."""
        src_folder = tmp_path / "external"
        _make_pair_files(src_folder, "foo bar")
        _make_pair_files(src_folder, "foo-bar")
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        assert len(pairs) == 2

        create_run_on_disk(tmp_path, "default", display_name="Default")
        outcome = copy_external_pairs_to_target_round(
            tmp_path,
            target_run_id="default", target_bucket="lateral",
            target_round_idx=0,
            pair_records=pairs,
            relabel_basename=True,
        )
        # Both copy — idx makes the labels unique
        assert len(outcome["copied"]) == 2
        assert outcome["errors"] == []
        # The two target labels differ only in their _z idx
        labels = sorted(outcome["copied"])
        assert labels == ["ext_foo_bar_XY_z0000", "ext_foo_bar_XY_z0001"]

    def test_dedup_catches_truly_identical_target_labels(self, tmp_path):
        """When relabel_basename=False AND pair_records has duplicate
        basenames (e.g. from a malformed enumerate_pairs result or a
        hand-built records list), the pre-loop dedup must catch this
        and report it via errors — never silently overwrite."""
        src_folder = tmp_path / "external"
        _make_pair_files(src_folder, "p1")
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        # Hand-duplicate the record so two pair_records have the same basename
        duplicated = pairs + [dict(pairs[0])]
        create_run_on_disk(tmp_path, "default", display_name="Default")
        outcome = copy_external_pairs_to_target_round(
            tmp_path,
            target_run_id="default", target_bucket="lateral",
            target_round_idx=0,
            pair_records=duplicated,
            relabel_basename=False,  # use raw basename → no idx → collision
        )
        # First copies; second flagged as collision
        assert len(outcome["copied"]) == 1
        assert len(outcome["errors"]) == 1
        assert "collision" in outcome["errors"][0][1].lower()

    def test_first_axis_with_channel_last_shape_blocked(self, tmp_path):
        """Audit-fix 2026-05-24 (J CRITICAL): if user declares 'Channels
        first' but the image is actually (H, W, C) channel-last, the
        import must REJECT the pair — never silently produce corrupt
        training data."""
        src_folder = tmp_path / "external"
        src_folder.mkdir()
        # Channel-last image: (64, 64, 3) — small channel dim at end
        img = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint16)
        mask[10:30, 10:30] = 1
        tifffile.imwrite(str(src_folder / "p1_img.tif"), img)
        tifffile.imwrite(str(src_folder / "p1_mask.tif"), mask)
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        create_run_on_disk(tmp_path, "default", display_name="Default")

        outcome = copy_external_pairs_to_target_round(
            tmp_path,
            target_run_id="default", target_bucket="lateral",
            target_round_idx=0,
            pair_records=pairs,
            relabel_basename=True,
            source_channel_axis="first",  # WRONG — should be "last"
        )
        assert outcome["copied"] == []
        assert len(outcome["errors"]) == 1
        msg = outcome["errors"][0][1]
        assert "Channels first" in msg
        assert "LAST axis" in msg

    def test_last_axis_with_channel_first_shape_blocked(self, tmp_path):
        """Mirror of above: declare 'last' but data is (C, H, W)."""
        src_folder = tmp_path / "external"
        src_folder.mkdir()
        img = np.random.randint(0, 255, size=(2, 64, 64), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint16)
        tifffile.imwrite(str(src_folder / "p1_img.tif"), img)
        tifffile.imwrite(str(src_folder / "p1_mask.tif"), mask)
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        create_run_on_disk(tmp_path, "default", display_name="Default")

        outcome = copy_external_pairs_to_target_round(
            tmp_path,
            target_run_id="default", target_bucket="lateral",
            target_round_idx=0,
            pair_records=pairs,
            relabel_basename=True,
            source_channel_axis="last",  # WRONG — should be "first"
        )
        assert outcome["copied"] == []
        assert len(outcome["errors"]) == 1
        assert "FIRST axis" in outcome["errors"][0][1]

    def test_none_axis_with_3d_image_blocked(self, tmp_path):
        """Declaring 2D / grayscale but image is 3D blocks import."""
        src_folder = tmp_path / "external"
        src_folder.mkdir()
        img = np.zeros((2, 64, 64), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint16)
        tifffile.imwrite(str(src_folder / "p1_img.tif"), img)
        tifffile.imwrite(str(src_folder / "p1_mask.tif"), mask)
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        create_run_on_disk(tmp_path, "default", display_name="Default")

        outcome = copy_external_pairs_to_target_round(
            tmp_path,
            target_run_id="default", target_bucket="lateral",
            target_round_idx=0,
            pair_records=pairs,
            relabel_basename=True,
            source_channel_axis="none",  # WRONG — 3D, not 2D
        )
        assert outcome["copied"] == []
        assert "3 dims" in outcome["errors"][0][1]

    def test_volumetric_4d_blocked(self, tmp_path):
        """4D / volumetric input is rejected outright — Cellpose 2D
        can't ingest it.  Audit-fix 2026-05-24 (D CRITICAL)."""
        src_folder = tmp_path / "external"
        src_folder.mkdir()
        img = np.zeros((5, 64, 64, 3), dtype=np.uint8)  # (Z, H, W, C)
        mask = np.zeros((64, 64), dtype=np.uint16)
        tifffile.imwrite(str(src_folder / "p1_img.tif"), img)
        tifffile.imwrite(str(src_folder / "p1_mask.tif"), mask)
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        create_run_on_disk(tmp_path, "default", display_name="Default")

        outcome = copy_external_pairs_to_target_round(
            tmp_path,
            target_run_id="default", target_bucket="lateral",
            target_round_idx=0,
            pair_records=pairs,
            relabel_basename=True,
            source_channel_axis="last",  # would-be 4D transpose
        )
        assert outcome["copied"] == []
        # Either shape-validator catches it first, or transpose helper does
        assert len(outcome["errors"]) == 1

    def test_first_axis_with_correct_3d_shape_succeeds(self, tmp_path):
        """Positive case: declared axis matches shape → copy proceeds."""
        src_folder = tmp_path / "external"
        src_folder.mkdir()
        img = np.random.randint(0, 255, size=(2, 64, 64), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint16)
        tifffile.imwrite(str(src_folder / "p1_img.tif"), img)
        tifffile.imwrite(str(src_folder / "p1_mask.tif"), mask)
        pairs, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        create_run_on_disk(tmp_path, "default", display_name="Default")

        outcome = copy_external_pairs_to_target_round(
            tmp_path,
            target_run_id="default", target_bucket="lateral",
            target_round_idx=0,
            pair_records=pairs,
            relabel_basename=True,
            source_channel_axis="first",
        )
        assert len(outcome["copied"]) == 1
        assert outcome["errors"] == []

    def test_partial_copy_failure_reported_per_pair(self, tmp_path):
        """If one pair has missing source files, others still copy + the
        bad one is reported in errors."""
        src_folder = tmp_path / "external"
        _make_pair_files(src_folder, "good_pair")
        # Make a bad record pointing at non-existent files
        bad_record = {
            "basename": "bad",
            "img_path": str(src_folder / "nonexistent_img.tif"),
            "mask_path": str(src_folder / "nonexistent_mask.tif"),
        }
        good_records, _, _ = enumerate_pairs(src_folder, "_img.tif", "_mask.tif")
        all_records = good_records + [bad_record]

        create_run_on_disk(tmp_path, "default", display_name="Default")
        outcome = copy_external_pairs_to_target_round(
            tmp_path, target_run_id="default", target_bucket="lateral",
            target_round_idx=0, pair_records=all_records,
            relabel_basename=True,
        )
        assert len(outcome["copied"]) == 1  # good_pair succeeded
        assert len(outcome["errors"]) == 1  # bad failed

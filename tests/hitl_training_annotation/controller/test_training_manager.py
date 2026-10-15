import numpy as np

from napari_epicycle.hitl_training_annotation.controller.training_manager import TrainingManager


class DummySegmentationManager:
    def __init__(self):
        self._inputs = {}
        self._masks = {}

    def set(self, slice_label, inp, mask):
        self._inputs[slice_label] = inp
        self._masks[slice_label] = mask

    def get_input_stack(self, slice_label):
        return self._inputs.get(slice_label)

    def get_masks(self, slice_label):
        return self._masks.get(slice_label)


def test_prepare_round_dataset_writes_samples(tmp_path):
    mgr = TrainingManager()
    seg = DummySegmentationManager()
    seg.set("stack01_XY_12", np.random.rand(2, 16, 16).astype(np.float32), np.ones((16, 16), dtype=np.uint16))
    seg.set("stack01_XY_12", np.random.rand(2, 16, 16).astype(np.float32), np.ones((16, 16), dtype=np.uint16))
    seg.set("stack02_XZ_5", np.random.rand(2, 16, 16).astype(np.float32), np.ones((16, 16), dtype=np.uint16))

    samples = mgr.prepare_round_dataset(
        segmentation_manager=seg,
        slice_labels=["stack01_XY_12", "stack02_XZ_5"],
        working_dir=str(tmp_path),
        round_id="round_00",
        dataset_id="dataset_a",
    )

    assert len(samples) == 2
    # Phase A (2026-05-25): fallback write path now lands in `accepted/`
    # (canonical curated-pair location) instead of legacy `spatial/`
    # and `axial/` subdirs.  Both XY and XZ slices land in the same
    # `accepted/` dir; view tag lives in filename + sample record.
    assert all("accepted" in s["img_uri"] for s in samples), \
        f"Expected all samples in 'accepted/' subdir; got: {[s['img_uri'] for s in samples]}"
    assert any("_XY_" in s["img_uri"] for s in samples), \
        "Expected at least one XY (lateral) sample"
    assert any("_XZ_" in s["img_uri"] for s in samples), \
        "Expected at least one XZ (axial) sample"


def test_prepare_round_dataset_uses_round_data_root_override(tmp_path):
    """When round_data_root is specified, it takes precedence over the
    legacy {working_dir}/training_data/{round_id} default.

    Step 2 callers pass this for the new bucket-first layout so working
    copies don't accumulate in legacy training_data/ post-migration.
    """
    mgr = TrainingManager()
    seg = DummySegmentationManager()
    seg.set("stack01_XY_12", np.random.rand(2, 16, 16).astype(np.float32),
            np.ones((16, 16), dtype=np.uint16))

    # Custom override path (mimics _paths.round_dir(...) output for
    # the new bucket-first layout)
    custom_root = tmp_path / "artifacts" / "step2" / "runs" / "default" \
                  / "lateral" / "rounds" / "round_00"

    samples = mgr.prepare_round_dataset(
        segmentation_manager=seg,
        slice_labels=["stack01_XY_12"],
        working_dir=str(tmp_path),
        round_id="round_00",
        dataset_id="dataset_a",
        round_data_root=custom_root,
    )

    assert len(samples) == 1
    # The sample's img_uri should be under the custom root, NOT under
    # the legacy training_data/ path
    assert str(custom_root).replace("\\", "/") in samples[0]["img_uri"].replace("\\", "/")
    assert "training_data" not in samples[0]["img_uri"]


def test_prepare_round_dataset_round_data_root_none_uses_legacy(tmp_path):
    """When round_data_root is None (default), uses legacy path."""
    mgr = TrainingManager()
    seg = DummySegmentationManager()
    seg.set("stack01_XY_12", np.random.rand(2, 16, 16).astype(np.float32),
            np.ones((16, 16), dtype=np.uint16))

    samples = mgr.prepare_round_dataset(
        segmentation_manager=seg,
        slice_labels=["stack01_XY_12"],
        working_dir=str(tmp_path),
        round_id="round_00",
        dataset_id="dataset_a",
        round_data_root=None,
    )

    assert len(samples) == 1
    # Should be under legacy training_data/round_00/
    img_uri_normalized = samples[0]["img_uri"].replace("\\", "/")
    assert "training_data/round_00" in img_uri_normalized


# ============================================================================
# Phase A audit follow-up tests (2026-05-25) — recommended by audit 2/5.
# Verifies the Phase A invariant ("no `spatial/`/`axial/` cruft subdirs")
# via NEGATIVE assertions (the prior tests only positively asserted the
# `accepted/` path).  Without these, Phase B could silently regress to
# re-creating the cruft and tests would still pass.
# ============================================================================

def test_prepare_round_dataset_does_not_create_spatial_axial_dirs(tmp_path):
    """Phase A negative invariant: empty cruft dirs MUST NOT be created."""
    mgr = TrainingManager()
    seg = DummySegmentationManager()
    seg.set("stack01_XY_12", np.random.rand(2, 16, 16).astype(np.float32),
            np.ones((16, 16), dtype=np.uint16))
    seg.set("stack02_XZ_5", np.random.rand(2, 16, 16).astype(np.float32),
            np.ones((16, 16), dtype=np.uint16))

    mgr.prepare_round_dataset(
        segmentation_manager=seg,
        slice_labels=["stack01_XY_12", "stack02_XZ_5"],
        working_dir=str(tmp_path),
        round_id="round_00",
        dataset_id="dataset_a",
    )

    legacy_round = tmp_path / "training_data" / "round_00"
    assert not (legacy_round / "spatial").exists(), \
        "Phase A invariant broken: legacy spatial/ subdir was created"
    assert not (legacy_round / "axial").exists(), \
        "Phase A invariant broken: legacy axial/ subdir was created"
    # And the fallback DID write to accepted/
    assert (legacy_round / "accepted").is_dir(), \
        "Fallback should have created accepted/"


def test_prepare_round_dataset_lazy_accepted_when_all_persisted(tmp_path):
    """When every slice has a valid `persisted_pairs` URI on disk,
    the fallback path never fires and `accepted/` is NOT mkdir'd.

    Validates the lazy-init guard at training_manager.py
    `if _accepted_dir is None`.
    """
    import tifffile
    mgr = TrainingManager()
    seg = DummySegmentationManager()

    # Plant a pair on disk OUTSIDE the round_dir + supply its URIs
    # via persisted_pairs.  prepare_round_dataset should reuse them
    # without touching the fallback path.
    ext_dir = tmp_path / "external_pairs"
    ext_dir.mkdir()
    img_path = ext_dir / "stack01_XY_z0012_img.tif"
    mask_path = ext_dir / "stack01_XY_z0012_mask.tif"
    tifffile.imwrite(str(img_path), np.random.rand(2, 16, 16).astype(np.float32))
    tifffile.imwrite(str(mask_path), np.ones((16, 16), dtype=np.uint16))

    samples = mgr.prepare_round_dataset(
        segmentation_manager=seg,  # empty — must not be queried
        slice_labels=["stack01_XY_12"],
        working_dir=str(tmp_path),
        round_id="round_00",
        dataset_id="dataset_a",
        persisted_pairs={
            "stack01_XY_12": {
                "img_uri": str(img_path),
                "mask_uri": str(mask_path),
            }
        },
    )

    assert len(samples) == 1
    legacy_round = tmp_path / "training_data" / "round_00"
    # accepted/ NOT created — lazy guard worked
    assert not (legacy_round / "accepted").exists(), \
        "Lazy guard broken: accepted/ created when no fallback fired"
    # Reused URI points at the EXTERNAL planted file
    assert samples[0]["img_uri"] == str(img_path)


def test_prepare_round_dataset_mixed_persisted_and_fallback(tmp_path):
    """One slice with valid persisted URI + one without → both end up
    in samples; accepted/ created and contains ONLY the fallback file.
    """
    import tifffile
    mgr = TrainingManager()
    seg = DummySegmentationManager()

    # Slice A: has persisted URI on disk (external location)
    ext_dir = tmp_path / "external_pairs"
    ext_dir.mkdir()
    a_img = ext_dir / "stack01_XY_z0012_img.tif"
    a_mask = ext_dir / "stack01_XY_z0012_mask.tif"
    tifffile.imwrite(str(a_img), np.random.rand(2, 16, 16).astype(np.float32))
    tifffile.imwrite(str(a_mask), np.ones((16, 16), dtype=np.uint16))

    # Slice B: only in segmentation_manager — forces fallback
    seg.set("stack02_XY_99",
            np.random.rand(2, 16, 16).astype(np.float32),
            np.ones((16, 16), dtype=np.uint16))

    samples = mgr.prepare_round_dataset(
        segmentation_manager=seg,
        slice_labels=["stack01_XY_12", "stack02_XY_99"],
        working_dir=str(tmp_path),
        round_id="round_00",
        dataset_id="dataset_a",
        persisted_pairs={
            "stack01_XY_12": {
                "img_uri": str(a_img),
                "mask_uri": str(a_mask),
            }
        },
    )

    assert len(samples) == 2
    accepted = tmp_path / "training_data" / "round_00" / "accepted"
    assert accepted.is_dir()
    # ONLY the fallback (stack02) is in accepted/; stack01's URI
    # points at the external dir
    files_in_accepted = sorted(p.name for p in accepted.iterdir())
    assert files_in_accepted == [
        "stack02_XY_z0099_img.tif", "stack02_XY_z0099_mask.tif"
    ], f"Unexpected files in accepted/: {files_in_accepted}"


def test_prepare_round_dataset_no_slices_no_accepted_dir(tmp_path):
    """slice_labels=[] + prior_pairs=[valid] → only the prior pair URIs
    are surfaced in records; accepted/ is NOT mkdir'd (no fallback fires).
    """
    import tifffile
    mgr = TrainingManager()
    seg = DummySegmentationManager()

    # Plant a prior pair on disk
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    p_img = prior_dir / "stack01_XY_z0001_img.tif"
    p_mask = prior_dir / "stack01_XY_z0001_mask.tif"
    tifffile.imwrite(str(p_img), np.random.rand(2, 16, 16).astype(np.float32))
    tifffile.imwrite(str(p_mask), np.ones((16, 16), dtype=np.uint16))

    samples = mgr.prepare_round_dataset(
        segmentation_manager=seg,
        slice_labels=[],
        working_dir=str(tmp_path),
        round_id="round_00",
        dataset_id="dataset_a",
        prior_pairs=[{
            "img_uri": str(p_img),
            "mask_uri": str(p_mask),
            "view_group": "spatial",
        }],
    )

    assert len(samples) == 1
    legacy_round = tmp_path / "training_data" / "round_00"
    assert not (legacy_round / "accepted").exists(), \
        "Lazy guard broken: accepted/ created with no current-round slices"


def test_prepare_round_dataset_round_data_root_no_spatial_axial(tmp_path):
    """When round_data_root is provided (new bucket-first layout),
    Phase A invariant still holds: NO spatial/, NO axial/ subdirs."""
    mgr = TrainingManager()
    seg = DummySegmentationManager()
    seg.set("stack01_XY_12",
            np.random.rand(2, 16, 16).astype(np.float32),
            np.ones((16, 16), dtype=np.uint16))

    custom_root = tmp_path / "artifacts" / "step2" / "runs" / "default" \
                  / "lateral" / "rounds" / "round_00"

    mgr.prepare_round_dataset(
        segmentation_manager=seg,
        slice_labels=["stack01_XY_12"],
        working_dir=str(tmp_path),
        round_id="round_00",
        dataset_id="dataset_a",
        round_data_root=custom_root,
    )

    assert not (custom_root / "spatial").exists()
    assert not (custom_root / "axial").exists()
    assert (custom_root / "accepted").is_dir()


# ============================================================================
# 2026-05-26 — regression: z-prefixed slice labels from cherry-pick /
# external import must parse correctly in prepare_round_dataset.
#
# Bug: SLICE_LABEL_RE rejected ``..._XY_z0055`` (z-prefixed disk-derived
# label).  Native curation writes sample_records keys as
# ``..._XY_55`` (non-prefixed); cherry-pick import derives keys from
# ``{label}_img.tif`` filenames which carry the z-prefix.  Training a
# round that included any imported pair crashed with
# ``Unrecognized slice label format: ..._XY_z0055``.
#
# Fix: ``SLICE_LABEL_RE`` regex made ``z?`` optional so both formats
# parse to the same ``(stack_id, view, int(idx))``.
# ============================================================================

def test_parse_slice_label_accepts_native_non_prefixed_format():
    """Native-curation labels parse as before — no regression."""
    from packages.epicycle_training.training_manager import TrainingManager
    mgr = TrainingManager()
    out = mgr._parse_slice_label("stack_patch_2_XY_62")
    assert out == {"stack_id": "stack_patch_2", "view": "XY", "slice_index": 62}


def test_parse_slice_label_accepts_z_prefixed_import_format():
    """REGRESSION: z-prefixed disk-derived labels (from cherry-pick /
    external import) must also parse.  Before the fix this raised
    ValueError ``Unrecognized slice label format``."""
    from packages.epicycle_training.training_manager import TrainingManager
    mgr = TrainingManager()
    out = mgr._parse_slice_label("stack_patch_2_XY_z0055")
    assert out == {"stack_id": "stack_patch_2", "view": "XY", "slice_index": 55}


def test_parse_slice_label_xz_yz_both_formats():
    """Z-prefix optional across all three views — neither was
    excluded by the pre-fix regex; verify both formats × all views."""
    from packages.epicycle_training.training_manager import TrainingManager
    mgr = TrainingManager()
    for view in ("XY", "XZ", "YZ"):
        # Non-prefixed
        out_a = mgr._parse_slice_label(f"stk_{view}_7")
        assert out_a == {"stack_id": "stk", "view": view, "slice_index": 7}
        # Z-prefixed + zero-padded
        out_b = mgr._parse_slice_label(f"stk_{view}_z0007")
        assert out_b == {"stack_id": "stk", "view": view, "slice_index": 7}


def test_parse_slice_label_still_rejects_garbage():
    """Negative — malformed labels still raise so callers can detect
    typos / wrong-format strings instead of silently fabricating
    stack_id="".  Regression guard for the ``z?`` widening."""
    import pytest
    from packages.epicycle_training.training_manager import TrainingManager
    mgr = TrainingManager()
    with pytest.raises(ValueError, match="Unrecognized slice label format"):
        mgr._parse_slice_label("not_a_label")
    with pytest.raises(ValueError, match="Unrecognized slice label format"):
        mgr._parse_slice_label("stk_AB_12")  # invalid view "AB"


def test_prepare_round_dataset_with_mixed_label_formats(tmp_path):
    """End-to-end regression: a slice_labels list mixing native (no z)
    and imported (z-prefixed) labels must NOT raise.  Mirrors the
    real-world cherry-pick-then-train flow that triggered the bug
    report on 2026-05-26."""
    import tifffile
    mgr = TrainingManager()
    seg = DummySegmentationManager()
    seg.set("stk_XY_62",
            np.random.rand(2, 16, 16).astype(np.float32),
            np.ones((16, 16), dtype=np.uint16))
    # Imported pair (disk-derived, z-prefixed key).  Plant the on-disk
    # files so prepare_round_dataset can stat them via persisted_pairs.
    ext_dir = tmp_path / "imported"
    ext_dir.mkdir()
    ext_img = ext_dir / "stk_XY_z0055_img.tif"
    ext_mask = ext_dir / "stk_XY_z0055_mask.tif"
    tifffile.imwrite(str(ext_img), np.random.rand(2, 16, 16).astype(np.float32))
    tifffile.imwrite(str(ext_mask), np.ones((16, 16), dtype=np.uint16))

    samples = mgr.prepare_round_dataset(
        segmentation_manager=seg,
        slice_labels=["stk_XY_62", "stk_XY_z0055"],
        working_dir=str(tmp_path),
        round_id="round_00",
        dataset_id="dataset_a",
        persisted_pairs={
            "stk_XY_z0055": {
                "img_uri": str(ext_img),
                "mask_uri": str(ext_mask),
            }
        },
    )
    assert len(samples) == 2
    indices = sorted(s["slice_index"] for s in samples)
    assert indices == [55, 62]


def test_run_training_placeholder_writes_checkpoint(tmp_path):
    mgr = TrainingManager()
    result = mgr.run_training(
        round_id="round_00",
        config={"model_name": "epi", "model_type": "cyto3", "epochs": 150},
        working_dir=str(tmp_path),
        training_samples=[{"id": 1}],
        view_type="spatial",
    )

    assert result["checkpoint_path"].endswith(".ckpt.json")
    assert len(result["losses"]) >= 5
    assert result["view_type"] == "spatial"


def test_split_samples_by_view_strategy():
    mgr = TrainingManager()
    grouped = mgr.split_samples_by_view_strategy(
        [
            {"view": "XY", "slice_index": 1},
            {"view": "XZ", "slice_index": 2},
            {"view": "YZ", "slice_index": 3},
        ]
    )
    assert len(grouped["spatial"]) == 1
    assert len(grouped["axial"]) == 2


def test_run_inference_placeholder_writes_manifest(tmp_path):
    mgr = TrainingManager()
    sample_mask = tmp_path / "sample_mask.tif"
    import tifffile

    tifffile.imwrite(str(sample_mask), np.ones((16, 16), dtype=np.uint16))

    result = mgr.run_inference(
        round_id="round_00",
        view_type="spatial",
        model_checkpoint_path="dummy.ckpt",
        config={"device": "CPU"},
        working_dir=str(tmp_path),
        inference_samples=[
            {
                "stack_id": "stack01",
                "view": "XY",
                "slice_index": 12,
                "mask_uri": str(sample_mask),
            }
        ],
    )
    assert result["prediction_uri"].endswith("manifest.json")

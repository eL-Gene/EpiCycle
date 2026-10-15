"""Tests for training pair extraction module."""

import json

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
    extract_training_pairs_from_3d,
)


def _make_volume(shape=(16, 32, 32)):
    """Create a test intensity volume and label volume."""
    rng = np.random.RandomState(42)
    image = rng.randint(0, 255, shape, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.int32)
    # Create 8 labeled regions spread across the volume
    z, y, x = shape
    label_id = 1
    for zi in range(0, z, z // 2):
        for yi in range(0, y, y // 4):
            ze, ye = min(zi + z // 4, z), min(yi + y // 8, y)
            labels[zi:ze, yi:ye, 2:-2] = label_id
            label_id += 1
    return image, labels


class TestExtractTrainingPairs:
    def test_basic_extraction(self, tmp_path):
        image, labels = _make_volume()
        result = extract_training_pairs_from_3d(
            image, labels, tmp_path / "pairs",
            min_labels_per_slice=1,
        )
        assert result["total"] > 0
        assert result["spatial"] >= 0
        assert result["axial"] >= 0
        assert result["spatial"] + result["axial"] == result["total"]

    def test_spatial_dir_created(self, tmp_path):
        image, labels = _make_volume()
        extract_training_pairs_from_3d(
            image, labels, tmp_path / "pairs",
            min_labels_per_slice=1,
        )
        assert (tmp_path / "pairs" / "spatial").is_dir()
        assert (tmp_path / "pairs" / "axial").is_dir()

    def test_min_labels_filter(self, tmp_path):
        image, labels = _make_volume()
        # With very high threshold, few slices should qualify
        result_high = extract_training_pairs_from_3d(
            image, labels, tmp_path / "high",
            min_labels_per_slice=100,
        )
        result_low = extract_training_pairs_from_3d(
            image, labels, tmp_path / "low",
            min_labels_per_slice=1,
        )
        assert result_high["total"] <= result_low["total"]

    def test_selective_views(self, tmp_path):
        image, labels = _make_volume()
        result = extract_training_pairs_from_3d(
            image, labels, tmp_path / "pairs",
            min_labels_per_slice=1,
            views=("xy",),
        )
        assert result["axial"] == 0
        assert result["spatial"] >= 0

    def test_file_naming(self, tmp_path):
        image, labels = _make_volume()
        extract_training_pairs_from_3d(
            image, labels, tmp_path / "pairs",
            min_labels_per_slice=1,
            views=("xy",),
        )
        spatial_dir = tmp_path / "pairs" / "spatial"
        img_files = sorted(spatial_dir.glob("*_img.tif"))
        mask_files = sorted(spatial_dir.glob("*_mask.tif"))
        assert len(img_files) > 0
        assert len(img_files) == len(mask_files)
        # Check naming pattern
        assert img_files[0].name.startswith("xy_")

    def test_multichannel_image(self, tmp_path):
        """4D input (Z, C, Y, X) should preserve all channels in output."""
        import tifffile
        image = np.random.randint(0, 255, (16, 2, 32, 32), dtype=np.uint8)
        labels = np.zeros((16, 32, 32), dtype=np.int32)
        labels[2:14, 4:28, 4:28] = 1
        labels[2:14, 4:14, 4:14] = 2
        labels[2:14, 20:28, 20:28] = 3
        result = extract_training_pairs_from_3d(
            image, labels, tmp_path / "pairs",
            min_labels_per_slice=1,
            views=("xy",),
        )
        assert result["total"] > 0
        # Verify saved images have 2 channels (C, Y, X)
        img_files = list((tmp_path / "pairs" / "spatial").glob("*_img.tif"))
        assert len(img_files) > 0
        saved = tifffile.imread(str(img_files[0]))
        assert saved.ndim == 3  # (C, Y, X)
        assert saved.shape[0] == 2  # 2 channels preserved

    def test_empty_labels(self, tmp_path):
        image = np.zeros((8, 8, 8), dtype=np.uint8)
        labels = np.zeros((8, 8, 8), dtype=np.int32)
        result = extract_training_pairs_from_3d(
            image, labels, tmp_path / "pairs",
            min_labels_per_slice=1,
        )
        assert result["total"] == 0

    def test_mask_dtype_int32(self, tmp_path):
        """Ensure saved masks use int32 dtype."""
        import tifffile
        image, labels = _make_volume()
        extract_training_pairs_from_3d(
            image, labels, tmp_path / "pairs",
            min_labels_per_slice=1,
            views=("xy",),
        )
        mask_files = list((tmp_path / "pairs" / "spatial").glob("*_mask.tif"))
        if mask_files:
            mask = tifffile.imread(str(mask_files[0]))
            assert mask.dtype == np.int32


# ---------------------------------------------------------------------------
# Workspace-level extraction tests (2026-05-19 — feature/refinement-
# extraction-safety branch).  Cover the rmtree-replacement + manifest-
# merge behavior in extract_from_workspace.
# ---------------------------------------------------------------------------


def _make_synthetic_workspace(tmp_path, patch_stems):
    """Build a synthetic workspace with N reconstructed patches.

    Returns (workspace_dir, image_dir) ready to pass to
    extract_from_workspace.  Each patch gets:
      - A reconstruction manifest at
        {ws}/artifacts/reconstruction/run_{stem}/manifest.json
      - A small synthetic labels TIFF + image TIFF at
        {ws}/labels/{stem}.tif and {ws}/images/{stem}.tif
    """
    import tifffile

    workspace_dir = tmp_path / "ws"
    recon_root = workspace_dir / "artifacts" / "step3" / "runs"
    labels_dir = workspace_dir / "labels"
    image_dir = workspace_dir / "images"
    for d in (recon_root, labels_dir, image_dir):
        d.mkdir(parents=True, exist_ok=True)

    image, labels = _make_volume()

    for stem in patch_stems:
        labels_path = labels_dir / f"{stem}.tif"
        image_path = image_dir / f"{stem}.tif"
        tifffile.imwrite(str(labels_path), labels.astype(np.int32))
        tifffile.imwrite(str(image_path), image.astype(np.uint8))

        run_dir = recon_root / f"run_{stem}"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": f"run_{stem}",
            "labels_path": str(labels_path),
            "source_stack_path": str(image_path),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

    return workspace_dir, image_dir


def _read_extraction_manifest(workspace_dir):
    """Load the merged extraction manifest from a workspace.

    2026-05-20 audit fix (F1): path is now refinement/manifest.json
    via the canonical helper.  Pre-fix the manifest was incorrectly
    written inside pairs/ as extraction_manifest.json.
    """
    from napari_epicycle.fusion_reconstruction import _paths
    path = _paths.refinement_manifest_path(workspace_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class TestExtractFromWorkspaceManifestMerge:
    """The rmtree → per-patch-merge behavior change."""

    def test_first_run_creates_schema_1_2_manifest(self, tmp_path):
        """Schema 1.2 (2026-05-19 redesign): per-view settings stored.

        Renamed from test_first_run_creates_schema_1_1_manifest when
        the schema bumped to 1.2 — per-patch entries now carry
        lateral_extraction_settings, lateral_mask_cleaning,
        axial_extraction_settings, axial_mask_cleaning in addition to
        the legacy combined fields.
        """
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(tmp_path, ["patchA"])
        extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1,
        )
        manifest = _read_extraction_manifest(ws)
        assert manifest is not None
        assert manifest["schema_version"] == "1.2"
        assert "patchA" in manifest["per_patch"]
        # Per-view fields present at both top level and per-patch level.
        assert "lateral_extraction_settings" in manifest
        assert "axial_mask_cleaning" in manifest
        entry = manifest["per_patch"]["patchA"]
        assert "lateral_extraction_settings" in entry
        assert "axial_mask_cleaning" in entry
        # Legacy combined fields still present for backward-compat reads.
        assert "extraction_settings" in entry
        assert "mask_cleaning" in entry

    def test_per_patch_source_mtime_persisted(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(tmp_path, ["patchA"])
        labels_path = ws / "labels" / "patchA.tif"
        expected_mtime = labels_path.stat().st_mtime
        extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1,
        )
        manifest = _read_extraction_manifest(ws)
        entry = manifest["per_patch"]["patchA"]
        assert "source_mtime" in entry
        assert abs(entry["source_mtime"] - expected_mtime) < 1.0
        assert entry["source_path"] == str(labels_path)

    def test_per_patch_settings_snapshot(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(tmp_path, ["patchA"])
        extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1,
            remove_boundaries=True, min_size=50, smoothing_disk_size=2,
        )
        manifest = _read_extraction_manifest(ws)
        cleaning = manifest["per_patch"]["patchA"]["mask_cleaning"]
        assert cleaning["remove_boundaries"] is True
        assert cleaning["min_size"] == 50
        assert cleaning["smoothing_disk_size"] == 2

    def test_re_extract_one_patch_preserves_others(self, tmp_path):
        """The flagship test for the rmtree footgun fix."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(
            tmp_path, ["patchA", "patchB", "patchC"],
        )
        # First run: extract all three.
        extract_from_workspace(
            ws, patch_names=["patchA", "patchB", "patchC"],
            image_dir=img_dir, min_labels_per_slice=1,
        )
        output_root = ws / "artifacts" / "step3" / "refinement" / "pairs"
        # Snapshot patch B + C files for later equality check.
        b_files_before = sorted(
            p.name for p in (output_root / "patchB" / "spatial").glob("*")
        )
        c_files_before = sorted(
            p.name for p in (output_root / "patchC" / "spatial").glob("*")
        )
        assert len(b_files_before) > 0
        assert len(c_files_before) > 0

        # Second run: only re-extract patch A.
        extract_from_workspace(
            ws, patch_names=["patchA"],
            image_dir=img_dir, min_labels_per_slice=1,
        )

        # The flagship assertion: patches B and C must still be on disk.
        assert (output_root / "patchB" / "spatial").is_dir()
        assert (output_root / "patchC" / "spatial").is_dir()
        b_files_after = sorted(
            p.name for p in (output_root / "patchB" / "spatial").glob("*")
        )
        c_files_after = sorted(
            p.name for p in (output_root / "patchC" / "spatial").glob("*")
        )
        assert b_files_before == b_files_after
        assert c_files_before == c_files_after

    def test_manifest_merge_includes_all_patches(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(
            tmp_path, ["patchA", "patchB", "patchC"],
        )
        extract_from_workspace(
            ws, patch_names=["patchA", "patchB"],
            image_dir=img_dir, min_labels_per_slice=1,
        )
        # Stage 2: re-extract only patchA + extract patchC.
        extract_from_workspace(
            ws, patch_names=["patchA", "patchC"],
            image_dir=img_dir, min_labels_per_slice=1,
        )
        manifest = _read_extraction_manifest(ws)
        assert set(manifest["per_patch"].keys()) == {
            "patchA", "patchB", "patchC",
        }
        assert manifest["patches"] == ["patchA", "patchB", "patchC"]

    def test_merged_totals_sum_across_all_patches(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(tmp_path, ["patchA", "patchB"])
        extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1,
        )
        manifest_after_A = _read_extraction_manifest(ws)
        a_spatial = manifest_after_A["per_patch"]["patchA"]["spatial"]
        a_axial = manifest_after_A["per_patch"]["patchA"]["axial"]

        extract_from_workspace(
            ws, patch_names=["patchB"], image_dir=img_dir,
            min_labels_per_slice=1,
        )
        manifest_after_B = _read_extraction_manifest(ws)
        b_spatial = manifest_after_B["per_patch"]["patchB"]["spatial"]
        b_axial = manifest_after_B["per_patch"]["patchB"]["axial"]

        # Merged totals must reflect both patches, not just the latest run.
        assert manifest_after_B["totals"]["spatial"] == a_spatial + b_spatial
        assert manifest_after_B["totals"]["axial"] == a_axial + b_axial
        assert manifest_after_B["totals"]["total"] == (
            a_spatial + a_axial + b_spatial + b_axial
        )

    def test_stale_manifest_entry_pruned(self, tmp_path):
        """A manifest entry referencing a no-longer-existing dir is dropped."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        import shutil
        ws, img_dir = _make_synthetic_workspace(tmp_path, ["patchA", "patchB"])
        extract_from_workspace(
            ws, patch_names=["patchA", "patchB"], image_dir=img_dir,
            min_labels_per_slice=1,
        )
        # Simulate the user manually deleting patchB's dir.
        output_root = ws / "artifacts" / "step3" / "refinement" / "pairs"
        shutil.rmtree(output_root / "patchB")

        # Next extraction (of unrelated patchA) should prune patchB from
        # the manifest because its dir is gone.
        extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1,
        )
        manifest = _read_extraction_manifest(ws)
        assert "patchB" not in manifest["per_patch"]
        assert "patchA" in manifest["per_patch"]

    def test_per_patch_settings_independent_across_runs(self, tmp_path):
        """A re-extract with new settings updates ONLY the touched patch."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(tmp_path, ["patchA", "patchB"])
        # Stage 1: extract both with smoothing_disk_size=1.
        extract_from_workspace(
            ws, patch_names=["patchA", "patchB"], image_dir=img_dir,
            min_labels_per_slice=1, smoothing_disk_size=1,
        )
        # Stage 2: re-extract patchA with smoothing_disk_size=3.
        extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1, smoothing_disk_size=3,
        )
        manifest = _read_extraction_manifest(ws)
        a_disk = manifest["per_patch"]["patchA"]["mask_cleaning"][
            "smoothing_disk_size"
        ]
        b_disk = manifest["per_patch"]["patchB"]["mask_cleaning"][
            "smoothing_disk_size"
        ]
        assert a_disk == 3
        assert b_disk == 1


class TestExtractFromWorkspaceAxesDetection:
    """Per-patch axis detection (2026-06-06 mixed-layout fix).

    The extract path took a SINGLE global ``axes`` for every patch.  When a
    patch's real 4D layout (C/Z order) differed from that global value, the
    4D slicing indexed the channel axis (size ~2) with a Z index (up to
    hundreds) → IndexError, which the user saw as "failed to extract".
    ``extract_from_workspace`` now detects the layout PER PATCH from the real
    image shape (``detect_axes_order``) and overrides the global, so both
    ZCYX and CZYX patches extract correctly regardless of the global combo.
    """

    def _make_4d_workspace(self, tmp_path, stem, layout):
        """1-patch workspace whose source image is 4D in *layout*
        ('ZCYX' → (Z,C,Y,X) or 'CZYX' → (C,Z,Y,X)).  Labels are 3D ZYX."""
        import tifffile
        from napari_epicycle.fusion_reconstruction import _paths

        ws = tmp_path / "ws"
        recon_root = _paths.runs_dir(ws)
        labels_dir = ws / "labels"
        image_dir = ws / "images"
        for d in (recon_root, labels_dir, image_dir):
            d.mkdir(parents=True, exist_ok=True)

        _img3d, labels = _make_volume((16, 32, 32))  # labels: (Z, Y, X)
        z, y, x = labels.shape
        rng = np.random.RandomState(7)
        zcyx = rng.randint(0, 255, (z, 2, y, x), dtype=np.uint8)
        if layout == "ZCYX":
            image_4d = zcyx
        elif layout == "CZYX":
            image_4d = np.transpose(zcyx, (1, 0, 2, 3))  # (C, Z, Y, X)
        else:
            raise ValueError(layout)

        labels_path = labels_dir / f"{stem}.tif"
        image_path = image_dir / f"{stem}.tif"
        tifffile.imwrite(str(labels_path), labels.astype(np.int32))
        tifffile.imwrite(str(image_path), image_4d)

        run_dir = recon_root / f"run_{stem}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({
                "run_id": f"run_{stem}",
                "labels_path": str(labels_path),
                "source_stack_path": str(image_path),
            }),
            encoding="utf-8",
        )
        return ws, image_dir

    @pytest.mark.parametrize(
        "layout,wrong_global",
        [("CZYX", "ZCYX"), ("ZCYX", "CZYX")],
    )
    def test_per_patch_axes_override_wrong_global(
        self, tmp_path, layout, wrong_global,
    ):
        """A patch in *layout* must extract correctly even when the global
        ``axes`` says the OPPOSITE — per-patch detection overrides it.
        Before the fix this raised IndexError ('failed to extract')."""
        import tifffile
        from napari_epicycle.fusion_reconstruction import _paths
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = self._make_4d_workspace(tmp_path, "patchA", layout)

        # Deliberately pass the WRONG global axes — the fix must ignore it
        # in favor of the per-patch detected layout.
        result = extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1, views=("xy",), axes=wrong_global,
        )
        # Extraction succeeded (no IndexError) and produced spatial pairs.
        assert result["spatial"] > 0
        # Saved xy images are (C, Y, X) with both channels preserved —
        # proof the 4D was normalized to ZCYX before slicing.
        spatial_dir = _paths.refinement_pairs_root(ws) / "patchA" / "spatial"
        img_files = sorted(spatial_dir.glob("*_img.tif"))
        assert img_files
        saved = tifffile.imread(str(img_files[0]))
        assert saved.ndim == 3
        assert saved.shape[0] == 2  # 2 channels, channel-first

    def test_thin_z_zcyx_beats_size_heuristic(self, tmp_path):
        """A genuinely-ZCYX patch with thin Z (Z <= C) is mis-guessed by the
        pure size heuristic (which calls it CZYX), but the deterministic
        labels-shape match resolves it correctly.  Saved xy images must keep
        all 3 channels — proof no spurious C/Z transpose happened."""
        import tifffile
        from napari_epicycle.fusion_reconstruction import _paths
        from napari_epicycle.fusion_reconstruction.qc_features import (
            detect_axes_order,
        )
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws = tmp_path / "ws"
        recon_root = _paths.runs_dir(ws)
        labels_dir = ws / "labels"
        image_dir = ws / "images"
        for d in (recon_root, labels_dir, image_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Thin-Z labels (Z=2), with labels in BOTH z-slices.
        labels = np.zeros((2, 32, 32), dtype=np.int32)
        labels[:, 4:14, 4:14] = 1
        labels[:, 18:28, 18:28] = 2
        # Genuinely ZCYX image: (Z=2, C=3, Y=32, X=32).
        rng = np.random.RandomState(11)
        image_4d = rng.randint(0, 255, (2, 3, 32, 32), dtype=np.uint8)
        # Precondition: the size-only heuristic gets THIS shape wrong.
        assert detect_axes_order(image_4d.shape) == "CZYX"  # wrong — it's ZCYX

        labels_path = labels_dir / "patchT.tif"
        image_path = image_dir / "patchT.tif"
        tifffile.imwrite(str(labels_path), labels.astype(np.int32))
        tifffile.imwrite(str(image_path), image_4d)
        run_dir = recon_root / "run_patchT"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({
                "run_id": "run_patchT",
                "labels_path": str(labels_path),
                "source_stack_path": str(image_path),
            }),
            encoding="utf-8",
        )

        result = extract_from_workspace(
            ws, patch_names=["patchT"], image_dir=image_dir,
            min_labels_per_slice=1, views=("xy",), axes="CZYX",  # wrong global too
        )
        assert result["spatial"] > 0
        spatial_dir = _paths.refinement_pairs_root(ws) / "patchT" / "spatial"
        img_files = sorted(spatial_dir.glob("*_img.tif"))
        assert img_files
        saved = tifffile.imread(str(img_files[0]))
        # Correct (ZCYX, no transpose) → (C=3, Y, X).  A heuristic-driven
        # CZYX transpose would instead have produced (C=2, ...).
        assert saved.ndim == 3
        assert saved.shape[0] == 3

    def test_3d_single_channel_unaffected(self, tmp_path):
        """3D (ZYX) single-channel input has no C/Z ambiguity — detection
        is skipped and extraction works regardless of the global axes."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(tmp_path, ["patchA"])
        result = extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1, axes="CZYX",  # wrong-but-irrelevant for 3D
        )
        assert result["spatial"] + result["axial"] > 0

    def test_3d_single_channel_spatial_pair_is_2d(self, tmp_path):
        """3D (ZYX) grayscale: the saved spatial (xy) image pair is plain 2D
        (Y, X) — no spurious channel axis.  Strengthens the count-only
        `test_3d_single_channel_unaffected` with a shape assertion."""
        import tifffile
        from napari_epicycle.fusion_reconstruction import _paths
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(tmp_path, ["patchA"])
        result = extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1, views=("xy",), axes="CZYX",
        )
        assert result["spatial"] > 0
        spatial_dir = _paths.refinement_pairs_root(ws) / "patchA" / "spatial"
        img_files = sorted(spatial_dir.glob("*_img.tif"))
        assert img_files
        saved = tifffile.imread(str(img_files[0]))
        # 3D source → xy slice is (Y, X), a plain 2D grayscale image.
        assert saved.ndim == 2
        assert saved.shape == (32, 32)

    def _make_sc4d_workspace(self, tmp_path, stem, layout):
        """1-patch workspace whose source image is 4D SINGLE-channel (C == 1)
        in *layout* ('ZCYX' → (Z,1,Y,X) or 'CZYX' → (1,Z,Y,X)).  Labels 3D ZYX.

        Plain TIFF write (no axes metadata) → resolve_patch_axes falls to the
        size heuristic, which correctly picks the singleton (C=1) dim.
        """
        import tifffile
        from napari_epicycle.fusion_reconstruction import _paths

        ws = tmp_path / "ws"
        recon_root = _paths.runs_dir(ws)
        labels_dir = ws / "labels"
        image_dir = ws / "images"
        for d in (recon_root, labels_dir, image_dir):
            d.mkdir(parents=True, exist_ok=True)

        _img3d, labels = _make_volume((16, 32, 32))  # labels: (Z, Y, X)
        z, y, x = labels.shape
        rng = np.random.RandomState(13)
        if layout == "ZCYX":
            image_4d = rng.randint(0, 255, (z, 1, y, x), dtype=np.uint8)
        elif layout == "CZYX":
            image_4d = rng.randint(0, 255, (1, z, y, x), dtype=np.uint8)
        else:
            raise ValueError(layout)

        labels_path = labels_dir / f"{stem}.tif"
        image_path = image_dir / f"{stem}.tif"
        tifffile.imwrite(str(labels_path), labels.astype(np.int32))
        tifffile.imwrite(str(image_path), image_4d)

        run_dir = recon_root / f"run_{stem}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({
                "run_id": f"run_{stem}",
                "labels_path": str(labels_path),
                "source_stack_path": str(image_path),
            }),
            encoding="utf-8",
        )
        return ws, image_dir

    @pytest.mark.parametrize("layout", ["ZCYX", "CZYX"])
    def test_single_channel_4d_extracts_with_one_channel(self, tmp_path, layout):
        """A 4D SINGLE-channel (C == 1) patch extracts successfully and the
        saved xy image keeps exactly one channel — (1, Y, X).  Covers both
        (Z, 1, Y, X) (ZCYX) and (1, Z, Y, X) (CZYX, normalized internally).
        Pre-existing 4D coverage only used C in {2, 3}; this pins C == 1."""
        import tifffile
        from napari_epicycle.fusion_reconstruction import _paths
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = self._make_sc4d_workspace(tmp_path, "patchSC", layout)
        result = extract_from_workspace(
            ws, patch_names=["patchSC"], image_dir=img_dir,
            min_labels_per_slice=1, views=("xy",),
            axes="CZYX",  # global value is irrelevant — resolved per patch
        )
        assert result["spatial"] > 0
        spatial_dir = _paths.refinement_pairs_root(ws) / "patchSC" / "spatial"
        img_files = sorted(spatial_dir.glob("*_img.tif"))
        assert img_files
        saved = tifffile.imread(str(img_files[0]))
        # 4D source normalized to (Z, 1, Y, X) → xy slice is (C=1, Y, X).
        assert saved.ndim == 3
        assert saved.shape == (1, 32, 32)


# ---------------------------------------------------------------------------
# Uniform-stride sampling (2026-05-19 — feature/refinement-extraction-safety)
# ---------------------------------------------------------------------------
#
# `_sample_with_min_distance` was changed from greedy-from-bottom to
# uniform-stride to fix a real bug: the old behavior silently dropped
# the top of every stack when the slice cap was small (it always
# started picking at the lowest valid index and stopped as soon as the
# cap filled).  These tests pin down the new behavior.


class TestUniformStrideSampling:

    def test_cap_doesnt_bite_returns_all(self):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            _sample_with_min_distance,
        )
        indices = list(range(20))
        # Asking for more than available → return all.
        picks = _sample_with_min_distance(indices, num_samples=50, min_distance=1)
        assert picks == indices

    def test_cap_bites_spans_full_range(self):
        """The flagship regression test for the greedy-from-bottom bug.

        With 200 valid indices and a cap of 50, the picks must cover
        the full range — first pick near the bottom, last pick near
        the top.  Pre-fix the picks would be indices[0..49] and the
        top 75% of the stack was silently ignored.
        """
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            _sample_with_min_distance,
        )
        indices = list(range(200))
        picks = _sample_with_min_distance(
            indices, num_samples=50, min_distance=1,
        )
        assert len(picks) == 50
        # First pick should be in the bottom 10% (i.e., index < 20),
        # last pick should be in the top 10% (i.e., index > 180).
        assert picks[0] < 20
        assert picks[-1] > 180

    def test_picks_are_evenly_spaced(self):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            _sample_with_min_distance,
        )
        indices = list(range(200))
        picks = _sample_with_min_distance(
            indices, num_samples=50, min_distance=1,
        )
        # Gaps between consecutive picks should all be ~4 (200 / 50).
        gaps = [picks[i + 1] - picks[i] for i in range(len(picks) - 1)]
        assert all(3 <= g <= 5 for g in gaps), (
            f"Uneven gaps: min={min(gaps)} max={max(gaps)}"
        )

    def test_single_pick_lands_near_middle(self):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            _sample_with_min_distance,
        )
        indices = list(range(100))
        picks = _sample_with_min_distance(
            indices, num_samples=1, min_distance=1,
        )
        # Single pick should land in the middle of the range.
        assert len(picks) == 1
        assert 30 <= picks[0] <= 70

    def test_two_picks_quartile_halves(self):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            _sample_with_min_distance,
        )
        indices = list(range(100))
        picks = _sample_with_min_distance(
            indices, num_samples=2, min_distance=1,
        )
        # Two picks should land roughly at 25% and 75% markers.
        assert len(picks) == 2
        assert 10 <= picks[0] <= 40
        assert 60 <= picks[1] <= 90

    def test_min_distance_floor_drops_overdense_picks(self):
        """When natural stride < min_distance, over-dense picks
        are dropped (returns fewer than num_samples)."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            _sample_with_min_distance,
        )
        # 100 indices, ask for 80 with min_distance=5.
        # Stride would be ~1.25, but min_distance=5 forces ≤ 20 picks.
        indices = list(range(100))
        picks = _sample_with_min_distance(
            indices, num_samples=80, min_distance=5,
        )
        # The filter applied to the uniform-stride picks must
        # produce a list whose adjacent gaps are all >= 5.
        gaps = [picks[i + 1] - picks[i] for i in range(len(picks) - 1)]
        assert all(g >= 5 for g in gaps)

    def test_min_distance_one_is_noop(self):
        """min_distance=1 means no filter — return the stride picks."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            _sample_with_min_distance,
        )
        indices = list(range(50))
        picks = _sample_with_min_distance(
            indices, num_samples=10, min_distance=1,
        )
        assert len(picks) == 10

    def test_empty_input_returns_empty(self):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            _sample_with_min_distance,
        )
        assert _sample_with_min_distance([], num_samples=5, min_distance=1) == []

    def test_zero_quota_returns_empty(self):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            _sample_with_min_distance,
        )
        indices = list(range(20))
        assert _sample_with_min_distance(indices, num_samples=0, min_distance=1) == []

    def test_picks_come_from_input_set(self):
        """Sanity: every returned pick must be a member of the input."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            _sample_with_min_distance,
        )
        # Non-contiguous input (simulating slices filtered by min_labels).
        indices = [10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175]
        picks = _sample_with_min_distance(
            indices, num_samples=4, min_distance=1,
        )
        assert all(p in indices for p in picks)
        # Picks should span the input range.
        assert min(picks) < indices[len(indices) // 4]
        assert max(picks) > indices[3 * len(indices) // 4]


# ---------------------------------------------------------------------------
# Per-orientation extraction params (2026-05-19, feature/refinement-
# extraction-redesign sub-commit 1).
# ---------------------------------------------------------------------------


class TestPerOrientationExtraction:
    """The new lateral_params / axial_params override semantics."""

    def test_no_overrides_matches_scalar_baseline(self, tmp_path):
        """When lateral_params/axial_params are None, behavior must
        match the pre-redesign scalar-only path exactly."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_training_pairs_from_3d,
        )
        image, labels = _make_volume(shape=(20, 32, 32))
        baseline = extract_training_pairs_from_3d(
            image, labels, tmp_path / "baseline",
            min_labels_per_slice=1, views=("xy",),
            remove_boundaries=True, min_size=20, smoothing_disk_size=1,
        )
        with_none = extract_training_pairs_from_3d(
            image, labels, tmp_path / "with_none",
            min_labels_per_slice=1, views=("xy",),
            remove_boundaries=True, min_size=20, smoothing_disk_size=1,
            lateral_params=None, axial_params=None,
        )
        assert baseline["spatial"] == with_none["spatial"]
        assert baseline["axial"] == with_none["axial"]

    def test_lateral_override_does_not_affect_axial(self, tmp_path):
        """Override lateral's min_labels_per_slice — axial counts
        should stay at the scalar value."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_training_pairs_from_3d,
        )
        image, labels = _make_volume(shape=(20, 32, 32))
        # Lateral gets a very high min_labels (likely drops everything),
        # axial inherits the lenient scalar.
        result = extract_training_pairs_from_3d(
            image, labels, tmp_path / "split",
            min_labels_per_slice=1, views=("xy", "xz", "yz"),
            lateral_params={"min_labels_per_slice": 100},
            axial_params=None,
        )
        assert result["spatial"] == 0  # all xy slices fail the override
        assert result["axial"] > 0     # xz + yz still use min_labels=1

    def test_axial_override_does_not_affect_lateral(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_training_pairs_from_3d,
        )
        image, labels = _make_volume(shape=(20, 32, 32))
        result = extract_training_pairs_from_3d(
            image, labels, tmp_path / "split2",
            min_labels_per_slice=1, views=("xy", "xz", "yz"),
            lateral_params=None,
            axial_params={"min_labels_per_slice": 100},
        )
        assert result["spatial"] > 0   # xy still has lenient min_labels
        assert result["axial"] == 0    # xz + yz drop everything

    def test_partial_override_inherits_unspecified_keys(self, tmp_path):
        """A lateral_params dict with ONE key only should override that
        single key and inherit the rest from scalars."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_training_pairs_from_3d,
        )
        image, labels = _make_volume(shape=(20, 32, 32))
        # Override only edge_buffer for lateral.  min_labels still
        # comes from the scalar (1, lenient).
        result = extract_training_pairs_from_3d(
            image, labels, tmp_path / "partial",
            min_labels_per_slice=1, views=("xy",),
            edge_buffer=0,
            lateral_params={"edge_buffer": 6},  # only edge_buffer set
        )
        # With edge_buffer=6 in a 20-slice volume, the candidate window
        # shrinks to indices [6, 14) → fewer slices vs. edge_buffer=0.
        baseline = extract_training_pairs_from_3d(
            image, labels, tmp_path / "partial_base",
            min_labels_per_slice=1, views=("xy",),
            edge_buffer=0,
        )
        assert result["spatial"] < baseline["spatial"]

    def test_per_view_manifest_fields_match_passed_params(self, tmp_path):
        """The manifest's per-view snapshots should reflect the
        lateral_params / axial_params actually used."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(tmp_path, ["patchA"])
        extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1, smoothing_disk_size=1,
            lateral_params={"smoothing_disk_size": 3},
            axial_params={"smoothing_disk_size": 5},
        )
        manifest = _read_extraction_manifest(ws)
        entry = manifest["per_patch"]["patchA"]
        assert entry["lateral_mask_cleaning"]["smoothing_disk_size"] == 3
        assert entry["axial_mask_cleaning"]["smoothing_disk_size"] == 5
        # Legacy combined field reflects the SCALAR (not the per-view
        # override) — kept that way intentionally for the schema-1.1
        # backward-compat read path.
        assert entry["mask_cleaning"]["smoothing_disk_size"] == 1

    def test_prescan_count_matches_extraction_with_per_view_params(self, tmp_path):
        """Progress-bar accuracy: the pre-scan's predicted slice
        total must equal the actual extracted slice total — even when
        per-view params differ from the scalar defaults.

        This is the regression test for sub-commit 9: prior to the
        fix, the pre-scan used `edge_buffer` / `min_labels_per_slice`
        / `slices_per_patch` scalars for ALL views, while extraction
        used per-view params.  Any per-orientation differentiation
        caused the progress bar to be wrong.
        """
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        ws, img_dir = _make_synthetic_workspace(tmp_path, ["patchA"])

        # Track progress callbacks — final count must == global_total
        # (i.e., progress reaches exactly 100%).
        progress_calls = []

        def _track(current, total):
            progress_calls.append((current, total))

        # Diverge per-view params from scalars to exercise the bug.
        result = extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            # Scalar values intentionally lenient.
            min_labels_per_slice=1, edge_buffer=0, slices_per_patch=0,
            # Per-view values strict — extraction yields FEWER slices
            # than the scalar-based pre-scan would predict.
            lateral_params={
                "min_labels_per_slice": 3, "edge_buffer": 2,
            },
            axial_params={
                "min_labels_per_slice": 3, "edge_buffer": 4,
            },
            progress_cb=_track,
        )

        actual_total = result["spatial"] + result["axial"]
        assert len(progress_calls) == actual_total, (
            f"progress callbacks ({len(progress_calls)}) must equal "
            f"actual extracted total ({actual_total}) — the bug was "
            f"that the pre-scan total used scalar params, so the "
            f"final progress callback's `total` arg would not match "
            f"the actual extracted count."
        )
        # Final callback's `total` (the prescan global_total) must
        # equal the count of fired callbacks (== actual extracted).
        final_total = progress_calls[-1][1]
        assert final_total == actual_total, (
            f"final progress total ({final_total}) != actual "
            f"extracted ({actual_total}) — progress bar would not "
            f"reach 100%."
        )


# ---------------------------------------------------------------------------
# Phase 1b-i: multi-run run_id threading through extract_from_workspace.
# ---------------------------------------------------------------------------


def _make_run_scoped_workspace(tmp_path, patch_stems, run_id):
    """Build a synthetic workspace with reconstructed patches under a run.

    Mirrors :func:`_make_synthetic_workspace` but places the per-patch
    reconstruction manifests under the run-aware location resolved by
    ``_paths.reconstructions_dir(ws, run_id)`` (flat for ``run_id=None``,
    ``runs/{run_id}/reconstructions/`` for a named run).
    """
    import tifffile

    from napari_epicycle.fusion_reconstruction import _paths

    workspace_dir = tmp_path / "ws"
    recon_root = _paths.reconstructions_dir(workspace_dir, run_id)
    labels_dir = workspace_dir / "labels"
    image_dir = workspace_dir / "images"
    for d in (recon_root, labels_dir, image_dir):
        d.mkdir(parents=True, exist_ok=True)

    image, labels = _make_volume()

    for stem in patch_stems:
        labels_path = labels_dir / f"{stem}.tif"
        image_path = image_dir / f"{stem}.tif"
        tifffile.imwrite(str(labels_path), labels.astype(np.int32))
        tifffile.imwrite(str(image_path), image.astype(np.uint8))

        run_dir = recon_root / f"run_{stem}"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": f"run_{stem}",
            "labels_path": str(labels_path),
            "source_stack_path": str(image_path),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

    return workspace_dir, image_dir


class TestExtractFromWorkspaceRunIdRouting:
    """run_id threads to the named run's tree; None stays flat (default)."""

    def test_named_run_routes_to_run_subtree(self, tmp_path):
        from napari_epicycle.fusion_reconstruction import _paths
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )

        ws, img_dir = _make_run_scoped_workspace(
            tmp_path, ["patchA"], run_id="cells",
        )
        result = extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1, run_id="cells",
        )
        assert result["patches_processed"] == 1
        # Pairs + manifest land under runs/cells/refinement/...
        run_pairs = _paths.patch_pairs_dir(ws, "patchA", run_id="cells")
        assert run_pairs.is_dir()
        run_manifest = _paths.refinement_manifest_path(ws, run_id="cells")
        assert run_manifest.exists()
        # The FLAT tree must NOT have been written.
        flat_pairs = _paths.patch_pairs_dir(ws, "patchA", run_id=None)
        assert not flat_pairs.exists()
        flat_manifest = _paths.refinement_manifest_path(ws, run_id=None)
        assert not flat_manifest.exists()

    def test_none_run_id_stays_flat(self, tmp_path):
        from napari_epicycle.fusion_reconstruction import _paths
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )

        ws, img_dir = _make_run_scoped_workspace(
            tmp_path, ["patchA"], run_id=None,
        )
        result = extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1, run_id=None,
        )
        assert result["patches_processed"] == 1
        # Flat tree written; no runs/ run-context subtree created.
        flat_pairs = _paths.patch_pairs_dir(ws, "patchA", run_id=None)
        assert flat_pairs.is_dir()
        cells_pairs = _paths.patch_pairs_dir(ws, "patchA", run_id="cells")
        assert not cells_pairs.exists()

    def test_default_omitted_run_id_matches_none(self, tmp_path):
        """Omitting run_id (legacy callers) is byte-identical to None."""
        from napari_epicycle.fusion_reconstruction import _paths
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )

        ws, img_dir = _make_run_scoped_workspace(
            tmp_path, ["patchA"], run_id=None,
        )
        result = extract_from_workspace(
            ws, patch_names=["patchA"], image_dir=img_dir,
            min_labels_per_slice=1,
        )
        assert result["patches_processed"] == 1
        assert _paths.patch_pairs_dir(ws, "patchA", run_id=None).is_dir()

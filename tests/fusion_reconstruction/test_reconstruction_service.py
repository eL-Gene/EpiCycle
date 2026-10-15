"""Tests for ReconstructionPipelineService and ReconstructionResult."""

import json
from pathlib import Path

import numpy as np
import tifffile

from packages.epicycle_reconstruction.config import ReconstructionConfig
from packages.epicycle_reconstruction.logic import (
    ReconstructionPipelineService,
    ReconstructionResult,
)
from napari_epicycle.domain import ReconstructionJobSpec


# ---------------------------------------------------------------------------
# ReconstructionResult
# ---------------------------------------------------------------------------

def test_result_save_mask_only(tmp_path):
    mask = np.ones((4, 8, 8), dtype=np.uint32)
    result = ReconstructionResult(mask_3d=mask)
    result.save(tmp_path, prefix="test")
    assert (tmp_path / "test_mask.tif").exists()
    loaded = tifffile.imread(str(tmp_path / "test_mask.tif"))
    assert loaded.shape == (4, 8, 8)


def test_result_save_all_artifacts(tmp_path):
    mask = np.ones((4, 8, 8), dtype=np.uint32)
    prob = np.random.rand(4, 8, 8).astype(np.float32)
    binary = (mask > 0).astype(np.uint8)
    flows = np.random.rand(2, 4, 8, 8).astype(np.float32)
    result = ReconstructionResult(
        mask_3d=mask, prob_3d=prob, binary_3d=binary, flows_3d=flows
    )
    result.save(tmp_path, prefix="full")
    assert (tmp_path / "full_mask.tif").exists()
    assert (tmp_path / "full_probabilities.npy").exists()
    assert (tmp_path / "full_binary.npy").exists()
    assert (tmp_path / "full_flows3D.npy").exists()


# ---------------------------------------------------------------------------
# Placeholder regression
# ---------------------------------------------------------------------------

def _spec(output_target: str) -> ReconstructionJobSpec:
    return ReconstructionJobSpec(
        schema_version="1.0",
        input_predictions=["predictions/round_01/xy.tif"],
        fusion_params={"dummy": True},
        output_target=output_target,
        run_mode="local",
    )


def test_placeholder_still_works(tmp_path):
    """Regression: placeholder mode must not break after adding run_real()."""
    src = tmp_path / "stack.tif"
    tifffile.imwrite(str(src), np.zeros((5, 24, 32), dtype=np.uint8))
    svc = ReconstructionPipelineService()
    result = svc.run_placeholder(
        working_dir=str(tmp_path),
        spec=_spec(str(tmp_path / "artifacts" / "step3" / "runs")),
        source_stack_path=str(src),
    )
    assert result["mode"] == "placeholder"
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["mode"] == "placeholder"
    labels = tifffile.imread(result["labels_path"])
    assert labels.shape == (5, 24, 32)
    assert labels.max() == 0


# ---------------------------------------------------------------------------
# Phase 1b-ii: reconstructions_root run-scoping (placeholder shares the
# same out_dir-routing logic as run_real, so it exercises the new branch
# without needing GPU / cellpose / segment3D).
# ---------------------------------------------------------------------------

def test_placeholder_reconstructions_root_nested(tmp_path):
    """reconstructions_root → out_dir = root / patch_name (named run)."""
    src = tmp_path / "stack.tif"
    tifffile.imwrite(str(src), np.zeros((5, 24, 32), dtype=np.uint8))
    svc = ReconstructionPipelineService()
    recon_root = tmp_path / "artifacts" / "step3" / "runs" / "cells" / "reconstructions"
    result = svc.run_placeholder(
        working_dir=str(tmp_path),
        spec=_spec(str(recon_root)),
        source_stack_path=str(src),
        patch_name="patch_000",
        reconstructions_root=str(recon_root),
    )
    expected = recon_root / "patch_000"
    assert Path(result["labels_path"]).parent == expected
    assert Path(result["manifest_path"]).parent == expected
    assert (expected / "labels_3d.tif").exists()
    assert (expected / "manifest.json").exists()


def test_placeholder_reconstructions_root_none_byte_identical(tmp_path):
    """reconstructions_root=None keeps the flat runs/{patch_name} dir."""
    src = tmp_path / "stack.tif"
    tifffile.imwrite(str(src), np.zeros((5, 24, 32), dtype=np.uint8))
    svc = ReconstructionPipelineService()
    result = svc.run_placeholder(
        working_dir=str(tmp_path),
        spec=_spec(str(tmp_path / "artifacts" / "step3" / "runs")),
        source_stack_path=str(src),
        patch_name="patch_000",
    )
    flat = tmp_path / "artifacts" / "step3" / "runs" / "patch_000"
    assert Path(result["labels_path"]).parent == flat
    assert (flat / "labels_3d.tif").exists()


def test_placeholder_reconstructions_root_precedence_over_patch_name(tmp_path):
    """When set, reconstructions_root wins; default run root == flat runs/."""
    src = tmp_path / "stack.tif"
    tifffile.imwrite(str(src), np.zeros((5, 24, 32), dtype=np.uint8))
    svc = ReconstructionPipelineService()
    # Default run: controller passes the flat runs/ dir as the root, so the
    # output must be byte-identical to the patch_name branch.
    flat_root = tmp_path / "artifacts" / "step3" / "runs"
    result = svc.run_placeholder(
        working_dir=str(tmp_path),
        spec=_spec(str(flat_root)),
        source_stack_path=str(src),
        patch_name="patch_000",
        reconstructions_root=str(flat_root),
    )
    assert Path(result["labels_path"]).parent == flat_root / "patch_000"


# ---------------------------------------------------------------------------
# Config defaults and dict merging
# ---------------------------------------------------------------------------

def test_config_dict_merging():
    """User-specified params should override defaults, not replace them."""
    cfg = ReconstructionConfig(
        combine_cell_gradients={"ksize": 7, "alpha": 0.1, "smooth_sigma": 0.5},
    )
    assert cfg.combine_cell_gradients["ksize"] == 7
    assert cfg.combine_cell_gradients["alpha"] == 0.1


# ---------------------------------------------------------------------------
# Shim export verification
# ---------------------------------------------------------------------------

def test_shim_exports_new_symbols():
    from napari_epicycle.fusion_reconstruction.logic import ReconstructionPipelineService as Shim
    from packages.epicycle_reconstruction.logic import ReconstructionPipelineService as Canon
    assert Shim is Canon

    from napari_epicycle.fusion_reconstruction.logic import ReconstructionResult as ShimResult
    from packages.epicycle_reconstruction.logic import ReconstructionResult as CanonResult
    assert ShimResult is CanonResult

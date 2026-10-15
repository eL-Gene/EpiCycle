"""Dummy data pipeline for end-to-end UI testing without GPU backends.

Creates synthetic prediction caches and reconstruction outputs that match
the exact on-disk format produced by real Cellpose + reconstruction runs.
Used by integration tests and for manual UI validation.

Phase K4.1.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from . import _paths


def make_synthetic_labels(
    shape: Tuple[int, int, int] = (10, 16, 16),
    n_labels: int = 5,
    *,
    include_small_artifacts: bool = True,
) -> np.ndarray:
    """Create a 3D label volume with known cubes and optionally tiny artifacts.

    Parameters
    ----------
    shape : tuple
        (Z, Y, X) dimensions.
    n_labels : int
        Number of main label cubes to place.
    include_small_artifacts : bool
        If True, adds 2 tiny labels (1-2 voxels each) for auto-discard testing.

    Returns
    -------
    np.ndarray
        int32 labels volume with label IDs 1..n_labels (+ artifact IDs).
    """
    labels = np.zeros(shape, dtype=np.int32)
    z, y, x = shape
    if n_labels <= 0:
        return labels
    cube_size = max(3, min(z // 3, y // n_labels - 1, x // 3))
    step_y = max(1, y // n_labels)

    for i in range(n_labels):
        label_id = i + 1
        y0 = i * step_y + 1
        y1 = min(y0 + cube_size, y - 1)
        z0, z1 = 1, min(1 + cube_size, z - 1)
        x0, x1 = 1, min(1 + cube_size, x - 1)
        labels[z0:z1, y0:y1, x0:x1] = label_id

    if include_small_artifacts and n_labels > 0:
        art1_id = n_labels + 1
        labels[0, 0, 0] = art1_id
        art2_id = n_labels + 2
        if z > 1 and x > 1:
            labels[z - 1, y - 1, x - 1] = art2_id
            labels[z - 1, y - 1, x - 2] = art2_id

    return labels


def create_dummy_prediction_cache(
    workspace: str,
    patch_name: str,
    shape: Tuple[int, int, int] = (10, 16, 16),
    run_id: Optional[str] = None,
) -> Dict[str, Tuple[str, str]]:
    """Create fake prediction cache .npy files on disk.

    Returns dict mapping view -> (flows_path, probs_path).

    ``run_id`` (Option A multi-run path-id): None → flat layout
    (byte-identical to today); a slug → ``runs/{run_id}/...``.
    """
    cache_dir = _paths.prediction_cache_dir(workspace, run_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(patch_name).stem
    z, y, x = shape
    result = {}
    for view_name in ("xy", "xz", "yz"):
        if view_name == "xy":
            flows = np.random.randn(2, z, y, x).astype(np.float16)
            probs = np.random.rand(z, y, x).astype(np.float16)
        elif view_name == "xz":
            flows = np.random.randn(2, y, z, x).astype(np.float16)
            probs = np.random.rand(y, z, x).astype(np.float16)
        else:
            flows = np.random.randn(2, x, z, y).astype(np.float16)
            probs = np.random.rand(x, z, y).astype(np.float16)

        flows_path = cache_dir / f"{stem}_{view_name}_flows.npy"
        probs_path = cache_dir / f"{stem}_{view_name}_probs.npy"
        np.save(str(flows_path), flows)
        np.save(str(probs_path), probs)
        result[view_name] = (str(flows_path), str(probs_path))
    return result


def create_dummy_reconstruction(
    workspace: str,
    patch_name: str,
    source_stack_path: str = "",
    shape: Tuple[int, int, int] = (10, 16, 16),
    n_labels: int = 5,
    include_small_artifacts: bool = True,
    run_id: Optional[str] = None,
) -> Dict[str, str]:
    """Create a reconstruction manifest + labels TIFF with synthetic data.

    Returns dict with keys: run_id, labels_path, manifest_path, mode.
    """
    import tifffile

    # NOTE: ``run_id`` (param) is the multi-run PATH-id (None → flat).
    # The manifest's ``run_id`` is a DIFFERENT concept — a per-recon
    # timestamp identifier recorded in manifest content for backward-compat
    # readers — kept in a distinct local to avoid shadowing the path-id.
    manifest_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Post-2026-05-20 cutover: per-patch dir, not run_id-keyed.
    out_dir = _paths.patch_run_dir(workspace, patch_name, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = make_synthetic_labels(
        shape, n_labels, include_small_artifacts=include_small_artifacts
    )
    labels_path = out_dir / "labels_3d.tif"
    tifffile.imwrite(str(labels_path), labels)

    manifest = {
        "run_id": manifest_run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dummy",
        "source_stack_path": source_stack_path or "",
        "labels_path": str(labels_path),
        "labels_shape": [int(d) for d in labels.shape],
        "num_labels": int(labels.max()),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "run_id": manifest_run_id,
        "labels_path": str(labels_path),
        "manifest_path": str(manifest_path),
        "mode": "dummy",
    }


def populate_dummy_workspace(
    workspace: str,
    stack_dir: str,
    shape: Tuple[int, int, int] = (10, 16, 16),
    n_labels: int = 5,
    n_patches: int = 2,
    run_id: Optional[str] = None,
) -> Dict[str, Dict]:
    """Create a full dummy workspace: stack TIFFs, prediction caches, reconstructions.

    Creates .tif stacks, prediction caches, and reconstruction outputs per patch.
    Returns {patch_name: {run_id, labels_path, manifest_path, cache_paths}}.
    """
    import tifffile

    stack_path = Path(stack_dir)
    stack_path.mkdir(parents=True, exist_ok=True)

    results = {}
    for i in range(n_patches):
        patch_name = f"dummy_patch_{i:03d}.tif"
        stack = np.random.randint(0, 255, shape, dtype=np.uint8)
        tifffile.imwrite(str(stack_path / patch_name), stack)

        cache_paths = create_dummy_prediction_cache(
            workspace, patch_name, shape, run_id=run_id
        )

        # Small sleep to ensure unique manifest_run_id timestamps (the per-
        # reconstruction job id in the manifest — distinct from the run_id
        # path-id param added in Phase 1b-i).
        time.sleep(0.01)

        recon = create_dummy_reconstruction(
            workspace,
            patch_name,
            source_stack_path=str(stack_path / patch_name),
            shape=shape,
            n_labels=n_labels,
            run_id=run_id,
        )
        recon["cache_paths"] = cache_paths
        results[patch_name] = recon

    return results

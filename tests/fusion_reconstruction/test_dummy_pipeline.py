"""Tests for the dummy data pipeline (Phase K4.1+).

Verifies synthetic data creation, state manager integration,
QC feature extraction on dummy labels, and auto-discard logic.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from napari_epicycle.fusion_reconstruction.dummy_pipeline import (
    create_dummy_prediction_cache,
    create_dummy_reconstruction,
    make_synthetic_labels,
    populate_dummy_workspace,
)
from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
)


# -------------------------------------------------------------------
# K4.1: Dummy pipeline
# -------------------------------------------------------------------


def test_make_synthetic_labels_shape_and_count():
    labels = make_synthetic_labels(shape=(10, 16, 16), n_labels=5, include_small_artifacts=False)
    assert labels.shape == (10, 16, 16)
    assert labels.dtype == np.int32
    unique = set(np.unique(labels)) - {0}
    assert len(unique) == 5
    assert unique == {1, 2, 3, 4, 5}


def test_make_synthetic_labels_with_artifacts():
    labels = make_synthetic_labels(shape=(10, 16, 16), n_labels=5, include_small_artifacts=True)
    unique = set(np.unique(labels)) - {0}
    # 5 main labels + 2 artifacts
    assert len(unique) == 7
    assert {6, 7}.issubset(unique)
    # Artifact 1: single voxel at (0,0,0)
    assert labels[0, 0, 0] == 6
    # Artifact 2: two voxels at corners
    assert labels[9, 15, 15] == 7
    assert labels[9, 15, 14] == 7


def test_make_synthetic_labels_no_labels():
    labels = make_synthetic_labels(shape=(10, 16, 16), n_labels=0, include_small_artifacts=False)
    assert labels.max() == 0


def test_create_dummy_prediction_cache_files_exist(tmp_path):
    result = create_dummy_prediction_cache(str(tmp_path), "patch001.tif", shape=(8, 12, 12))
    assert set(result.keys()) == {"xy", "xz", "yz"}
    for view, (flows_p, probs_p) in result.items():
        assert Path(flows_p).exists()
        assert Path(probs_p).exists()
        flows = np.load(flows_p)
        probs = np.load(probs_p)
        assert flows.ndim == 4
        assert flows.shape[0] == 2
        assert probs.ndim == 3


def test_create_dummy_reconstruction_manifest(tmp_path):
    result = create_dummy_reconstruction(
        str(tmp_path), "patch001.tif",
        source_stack_path="/fake/path.tif",
        shape=(8, 12, 12), n_labels=3,
    )
    assert "run_id" in result
    assert result["mode"] == "dummy"
    assert Path(result["labels_path"]).exists()
    assert Path(result["manifest_path"]).exists()

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["run_id"] == result["run_id"]
    assert manifest["mode"] == "dummy"
    assert manifest["num_labels"] >= 3  # may include artifacts
    assert manifest["labels_shape"] == [8, 12, 12]

    labels = tifffile.imread(result["labels_path"])
    assert labels.shape == (8, 12, 12)


def test_populate_dummy_workspace_full(tmp_path):
    workspace = str(tmp_path / "ws")
    stack_dir = str(tmp_path / "stacks")
    results = populate_dummy_workspace(workspace, stack_dir, n_patches=3)
    assert len(results) == 3
    for patch_name, info in results.items():
        assert patch_name.endswith(".tif")
        assert Path(info["labels_path"]).exists()
        assert Path(info["manifest_path"]).exists()
        assert "cache_paths" in info
        # Stack file should exist
        assert (Path(stack_dir) / patch_name).exists()


def test_state_manager_scan_discovers_dummy_artifacts(tmp_path):
    workspace = str(tmp_path / "ws")
    stack_dir = str(tmp_path / "stacks")
    results = populate_dummy_workspace(workspace, stack_dir, n_patches=2)

    sm = ReconstructionStateManager()
    sm.load(workspace)
    sm.register_patches(list(results.keys()))
    sm.scan_existing_artifacts(workspace)

    for patch_name in results:
        status = sm.get_patch_status(patch_name)
        # scan_existing_artifacts should detect prediction cache + reconstruction
        assert status in ("predicted", "reconstructed"), (
            f"Expected predicted or reconstructed for {patch_name}, got {status}"
        )


def test_qc_extract_on_dummy_labels(tmp_path):
    from napari_epicycle.fusion_reconstruction.qc_features import extract_features

    labels = make_synthetic_labels(shape=(10, 16, 16), n_labels=5, include_small_artifacts=True)
    df = extract_features(labels)
    assert len(df) == 7  # 5 main + 2 artifacts
    assert "area" in df.columns
    assert "centroid-0" in df.columns
    assert "centroid-1" in df.columns
    assert "centroid-2" in df.columns
    # Artifacts should have small area
    art_rows = df[df["label"].isin([6, 7])]
    assert all(art_rows["area"] <= 2)
    # Main labels should have larger area
    main_rows = df[df["label"].isin([1, 2, 3, 4, 5])]
    assert all(main_rows["area"] > 2)


def test_dummy_full_chain(tmp_path):
    """Full chain: populate -> state scan -> QC extract -> threshold -> verify flagged."""
    from napari_epicycle.fusion_reconstruction.qc_features import (
        apply_threshold,
        extract_features,
    )

    workspace = str(tmp_path / "ws")
    stack_dir = str(tmp_path / "stacks")
    results = populate_dummy_workspace(workspace, stack_dir, n_patches=1)

    patch_name = list(results.keys())[0]
    labels_path = results[patch_name]["labels_path"]
    labels = tifffile.imread(labels_path)

    df = extract_features(labels)
    assert len(df) == 7  # 5 main + 2 artifacts

    # Threshold on area: flag labels smaller than 5 voxels
    apply_threshold(df, "area", 5, None)
    flagged = df[df["qc_status"] == "flagged"]
    passed = df[df["qc_status"] == "pass"]

    # Artifacts (1-2 voxels) should be flagged, main cubes should pass
    flagged_ids = set(flagged["label"].values)
    assert {6, 7}.issubset(flagged_ids)
    assert all(pid in passed["label"].values for pid in [1, 2, 3, 4, 5])


# -------------------------------------------------------------------
# K4.2: Auto-discard
# -------------------------------------------------------------------


def test_auto_discard_below_threshold(tmp_path):
    """Verify that labels below volume threshold can be identified for auto-discard."""
    from napari_epicycle.fusion_reconstruction.qc_features import (
        apply_threshold,
        extract_features,
    )

    labels = make_synthetic_labels(shape=(10, 16, 16), n_labels=5, include_small_artifacts=True)
    df = extract_features(labels)

    min_volume = 5
    small_mask = df["area"] < min_volume
    auto_discard_ids = set(int(row["label"]) for _, row in df[small_mask].iterrows())

    # Both artifacts should be below the threshold
    assert 6 in auto_discard_ids
    assert 7 in auto_discard_ids
    # Main labels should NOT be in auto_discard
    for lid in [1, 2, 3, 4, 5]:
        assert lid not in auto_discard_ids

    # Simulate zeroing out
    zeroed = labels.copy()
    mask = np.isin(zeroed, list(auto_discard_ids))
    zeroed[mask] = 0
    remaining = set(np.unique(zeroed)) - {0}
    assert remaining == {1, 2, 3, 4, 5}


# -------------------------------------------------------------------
# K4.3: Flagged centroids
# -------------------------------------------------------------------


def test_flagged_centroids_correct(tmp_path):
    """Verify centroid extraction matches regionprops for flagged labels."""
    from napari_epicycle.fusion_reconstruction.qc_features import (
        apply_threshold,
        extract_features,
    )

    labels = make_synthetic_labels(shape=(10, 16, 16), n_labels=5, include_small_artifacts=True)
    df = extract_features(labels)
    apply_threshold(df, "area", 5, None)

    flagged = df[df["qc_status"] == "flagged"]
    centroids = {}
    for _, row in flagged.iterrows():
        lid = int(row["label"])
        centroids[lid] = (
            int(row["centroid-0"]),
            int(row["centroid-1"]),
            int(row["centroid-2"]),
        )

    # Artifact 1 is at (0,0,0) — single voxel, centroid should be (0,0,0)
    assert 6 in centroids
    assert centroids[6] == (0, 0, 0)

    # Artifact 2 is at (9,15,15) and (9,15,14) — centroid Z=9, Y=15, X~14.5 -> int(14)
    assert 7 in centroids
    assert centroids[7][0] == 9
    assert centroids[7][1] == 15
    assert centroids[7][2] in (14, 15)  # float centroid rounded


# -------------------------------------------------------------------
# K4.4: Focused refinement data computation
# -------------------------------------------------------------------


def test_focused_data_computation():
    """Verify flagged IDs and centroids computed correctly from QC DataFrame."""
    from napari_epicycle.fusion_reconstruction.qc_features import (
        apply_threshold,
        extract_features,
    )

    labels = make_synthetic_labels(shape=(10, 16, 16), n_labels=3, include_small_artifacts=True)
    df = extract_features(labels)
    apply_threshold(df, "area", 5, None)

    flagged = df[df["qc_status"] == "flagged"]
    flagged_ids = []
    flagged_centroids = {}
    for _, row in flagged.iterrows():
        lid = int(row["label"])
        flagged_ids.append(lid)
        flagged_centroids[lid] = (
            int(row.get("centroid-0", 0)),
            int(row.get("centroid-1", 0)),
            int(row.get("centroid-2", 0)),
        )

    assert len(flagged_ids) >= 2  # at least the 2 artifacts
    assert all(lid in flagged_centroids for lid in flagged_ids)
    for lid in flagged_ids:
        z, y, x = flagged_centroids[lid]
        assert isinstance(z, int) and isinstance(y, int) and isinstance(x, int)


def test_nav_index_cycling():
    """Verify prev/next navigation index wraps around correctly."""
    n = 5
    index = -1

    # Next: -1 -> 0 -> 1 -> 2 -> 3 -> 4 -> 0
    for expected in [0, 1, 2, 3, 4, 0]:
        index = (index + 1) % n
        assert index == expected

    # Prev: 0 -> 4 -> 3 -> 2 -> 1 -> 0
    for expected in [4, 3, 2, 1, 0]:
        index = (index - 1) % n
        assert index == expected


# ---------------------------------------------------------------------------
# Phase 1b-i: run_id threading through dummy_pipeline helpers.
# ---------------------------------------------------------------------------


def test_dummy_prediction_cache_named_run_routes_to_run_subtree(tmp_path):
    from napari_epicycle.fusion_reconstruction import _paths

    ws = str(tmp_path / "ws")
    create_dummy_prediction_cache(ws, "patch001.tif", shape=(8, 12, 12), run_id="cells")
    assert _paths.prediction_cache_dir(ws, "cells").is_dir()
    # Flat default-run cache untouched.
    assert not _paths.prediction_cache_dir(ws, None).exists()


def test_dummy_reconstruction_named_run_routes_to_run_subtree(tmp_path):
    from napari_epicycle.fusion_reconstruction import _paths

    ws = str(tmp_path / "ws")
    create_dummy_reconstruction(ws, "patch001.tif", run_id="cells")
    assert _paths.patch_run_dir(ws, "patch001.tif", run_id="cells").is_dir()
    assert not _paths.patch_run_dir(ws, "patch001.tif", run_id=None).exists()


def test_populate_dummy_workspace_none_run_id_stays_flat(tmp_path):
    """Omitting run_id (default) is byte-identical to today's flat layout."""
    from napari_epicycle.fusion_reconstruction import _paths

    workspace = str(tmp_path / "ws")
    stack_dir = str(tmp_path / "stacks")
    results = populate_dummy_workspace(workspace, stack_dir, n_patches=1)
    patch_name = next(iter(results))
    assert _paths.patch_run_dir(workspace, patch_name, run_id=None).is_dir()
    # No named-run subtree created.
    assert not _paths.runs_dir(workspace).joinpath("cells").exists()


def test_populate_dummy_workspace_named_run_routes_and_scans(tmp_path):
    from napari_epicycle.fusion_reconstruction import _paths

    workspace = str(tmp_path / "ws")
    stack_dir = str(tmp_path / "stacks")
    results = populate_dummy_workspace(
        workspace, stack_dir, n_patches=2, run_id="cells",
    )

    # Artifacts land under the named run.
    for patch_name in results:
        assert _paths.patch_run_dir(workspace, patch_name, run_id="cells").is_dir()

    # The state manager active on that run discovers them.
    sm = ReconstructionStateManager()
    sm.load(workspace)
    sm.active_run_id = "cells"
    sm.register_patches(list(results.keys()))
    sm.scan_existing_artifacts(workspace)
    for patch_name in results:
        assert sm.get_patch_status(patch_name) in ("predicted", "reconstructed")

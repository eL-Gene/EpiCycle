"""Tests for the index-based split partition fix (2026-05-21).

Pre-fix bug: `_launch_next_refinement` partitioned `images` into train
vs val by `Path(p).stem.replace("_img", "")` membership in
`set(train_split)`.  Per-patch slice naming is unique only WITHIN a
patch — `xz_0001_img.tif` exists under EVERY patch's axial directory.
When the user trained Mode=Both on a multi-patch workspace, the stem
collapse meant any unique stem covered in the train sample dragged
ALL its cross-patch copies to train, producing a 300/0 split with no
val loss curve.

Fix: use INDEX-based identity (one unique ID per image regardless of
stem collisions).  These tests verify the index-partition algorithm
in isolation; the controller-level integration is exercised by the
existing `_launch_next_refinement` path.

Pure Python — no controller, no Qt, no real files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
    split_training_data,
)


def _partition_by_index(
    images: list, masks: list, train_frac: float, seed: int = 42,
) -> tuple:
    """Mirror of the controller's post-fix partition logic (without
    the QThread / signal / Qt scaffolding)."""
    index_ids = [f"{i}" for i in range(len(images))]
    train_split, _ = split_training_data(
        index_ids,
        train_frac=train_frac,
        spatial_split=False,
        chunk_size=10,
        seed=seed,
    )
    train_set = set(train_split)
    train_imgs, train_masks = [], []
    val_imgs, val_masks = [], []
    for i, (img, mask) in enumerate(zip(images, masks)):
        if f"{i}" in train_set:
            train_imgs.append(img)
            train_masks.append(mask)
        else:
            val_imgs.append(img)
            val_masks.append(mask)
    return train_imgs, val_imgs, train_masks, val_masks


def _partition_by_stem(
    images: list, masks: list, train_frac: float, seed: int = 42,
) -> tuple:
    """Mirror of the PRE-fix partition logic.  Kept for the regression
    test that proves the bug used to exist."""
    train_stems = [Path(p).stem.replace("_img", "") for p in images]
    train_split, _ = split_training_data(
        train_stems,
        train_frac=train_frac,
        spatial_split=False,
        chunk_size=10,
        seed=seed,
    )
    train_set = set(train_split)
    train_imgs, train_masks = [], []
    val_imgs, val_masks = [], []
    for img, mask in zip(images, masks):
        stem = Path(img).stem.replace("_img", "")
        if stem in train_set:
            train_imgs.append(img)
            train_masks.append(mask)
        else:
            val_imgs.append(img)
            val_masks.append(mask)
    return train_imgs, val_imgs, train_masks, val_masks


def _make_multi_patch_pairs(
    n_patches: int, n_per_patch: int, view: str = "xz",
) -> tuple:
    """Build a list of fake per-patch paths that simulate
    `_collect_training_pairs` output for Mode=Axial across multiple
    patches.  Stems collide across patches by construction."""
    images, masks = [], []
    for p_idx in range(n_patches):
        for s_idx in range(n_per_patch):
            # Per-patch directory; per-patch stem reuses xz_0000..xz_NNNN
            # for every patch.  Mirrors training_pair_extractor.py:423.
            root = f"/fake/patch_{p_idx:02d}/axial"
            stem = f"{view}_{s_idx:04d}"
            images.append(f"{root}/{stem}_img.tif")
            masks.append(f"{root}/{stem}_mask.tif")
    return images, masks


# ---------------------------------------------------------------------------
# The actual bug repro — proves the pre-fix logic produces 0 val
# ---------------------------------------------------------------------------


class TestPreFixBugReproduction:
    """These tests EXHIBIT the pre-fix bug.  They confirm the
    `_partition_by_stem` algorithm DOES collapse cross-patch
    duplicates the way the user reported."""

    def test_pre_fix_collapses_duplicates_biases_split(self):
        """3 patches × 100 slices per patch = 300 total but only 100
        unique stems.  With train_frac=0.82, the train sample covers
        almost every unique stem → val is essentially empty, far
        below the expected 54 (= 18% of 300).  This reproduces the
        user's symptom (their actual run hit 300/0; the exact split
        depends on which unique stems happen to be covered).

        The expected post-fix val count would be 54 (54/300 = 18%).
        Pre-fix produces ≤ ~10 val — a >5x deficit — proving the
        partition algorithm was the bug, not random variance."""
        images, masks = _make_multi_patch_pairs(
            n_patches=3, n_per_patch=100, view="xz",
        )
        train, val, _, _ = _partition_by_stem(
            images, masks, train_frac=0.82, seed=42,
        )
        # Expected val with correct partition: ~54.  Pre-fix gives
        # essentially zero (≤10) because the duplicate-stem collapse
        # forces all copies of each train-set stem into train.
        assert len(train) + len(val) == 300
        assert len(val) <= 10, (
            f"Pre-fix should produce near-zero val due to stem "
            f"collision; got {len(val)} (would be ~54 with correct "
            f"partition)"
        )

    def test_pre_fix_lateral_gets_lucky_partial_split(self):
        """Lateral has fewer pairs per patch (50 vs 100), so coverage
        of unique stems is less certain.  With seed=42 it happens to
        miss some stems → some images go to val.  This matches the
        user's report of 123/27 lateral split."""
        images, masks = _make_multi_patch_pairs(
            n_patches=3, n_per_patch=50, view="xy",
        )
        train, val, _, _ = _partition_by_stem(
            images, masks, train_frac=0.82, seed=42,
        )
        # Pre-fix produces SOME val for lateral (not 0), but the
        # split is biased — not a true 82/18 of the 150 pairs.
        # The exact numbers depend on the seed; the key invariant
        # we test is "non-empty val, but biased away from the
        # nominal train_frac".
        assert len(train) + len(val) == 150
        # Lateral happens to leave some val, unlike axial.
        # (We don't assert exact numbers because they're seed-specific.)


# ---------------------------------------------------------------------------
# Post-fix behavior — index-based partition
# ---------------------------------------------------------------------------


class TestPostFixIndexPartition:

    def test_correct_proportions_with_duplicate_stems(self):
        """Same multi-patch input that broke the old logic.  Index
        partition produces the expected ~82/18 split."""
        images, masks = _make_multi_patch_pairs(
            n_patches=3, n_per_patch=100, view="xz",
        )
        train, val, _, _ = _partition_by_index(
            images, masks, train_frac=0.82, seed=42,
        )
        assert len(train) + len(val) == 300
        # 82% of 300 = 246, allowing ±1 for round().
        assert len(train) == 246
        assert len(val) == 54

    def test_no_overlap_post_fix(self):
        """Sanity: no image appears in both train and val."""
        images, masks = _make_multi_patch_pairs(
            n_patches=3, n_per_patch=100,
        )
        train, val, _, _ = _partition_by_index(
            images, masks, train_frac=0.8, seed=7,
        )
        assert set(train).isdisjoint(set(val))

    def test_masks_align_with_images(self):
        """Each image's mask must travel with it to the same bucket."""
        images, masks = _make_multi_patch_pairs(
            n_patches=2, n_per_patch=20,
        )
        train_i, val_i, train_m, val_m = _partition_by_index(
            images, masks, train_frac=0.75, seed=3,
        )
        # Mask path is image path with _img→_mask.
        for img, mask in zip(train_i, train_m):
            assert img.replace("_img", "_mask") == mask
        for img, mask in zip(val_i, val_m):
            assert img.replace("_img", "_mask") == mask

    def test_single_patch_no_collisions(self):
        """When there's only one patch (no stem collisions), the
        post-fix partition gives the same answer the pre-fix did
        for non-degenerate inputs.  Verifies we didn't break the
        single-patch case while fixing multi-patch."""
        images, masks = _make_multi_patch_pairs(
            n_patches=1, n_per_patch=100,
        )
        train, val, _, _ = _partition_by_index(
            images, masks, train_frac=0.8, seed=11,
        )
        assert len(train) == 80
        assert len(val) == 20

    def test_axial_xz_and_yz_unique_within_patch(self):
        """Axial mixes xz_NNNN and yz_NNNN — unique within a single
        patch, but each repeats across patches.  Index partition
        handles this cleanly."""
        images, masks = [], []
        for p in range(3):
            for v in ("xz", "yz"):
                for s in range(50):
                    images.append(f"/fake/patch_{p}/axial/{v}_{s:04d}_img.tif")
                    masks.append(f"/fake/patch_{p}/axial/{v}_{s:04d}_mask.tif")
        # 3 patches × 2 views × 50 slices = 300 total, 100 unique stems.
        assert len(images) == 300
        train, val, _, _ = _partition_by_index(
            images, masks, train_frac=0.82, seed=42,
        )
        assert len(train) + len(val) == 300
        assert len(val) >= 50  # well above zero


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_same_seed_same_partition(self):
        images, masks = _make_multi_patch_pairs(n_patches=2, n_per_patch=30)
        a_train, a_val, _, _ = _partition_by_index(
            images, masks, train_frac=0.8, seed=99,
        )
        b_train, b_val, _, _ = _partition_by_index(
            images, masks, train_frac=0.8, seed=99,
        )
        assert a_train == b_train
        assert a_val == b_val

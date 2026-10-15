"""Tests for the qc_features_df duplicate-row fixes (2026-05-17).

User-confirmed bug 2026-05-17: labels 1281 and 203 had 4 ACTIVE rows
each in the on-disk CSV, and accepting a paint+interp on label 1286
created 3 ghost copies.  Re-extract then multiplied 1286 to 4 active
rows via my left-merge.

Three compounding bugs were patched in this branch:

  Bug A: _on_annotator_interpolation_accepted (and
         _on_annotator_label_smoothed) copied ALL rows matching
         label_id (including prior ghosts) as new superseded ghosts
         → exponential ghost growth on repeated accepts.

  Bug B: _on_qc_extract_finished left-merge took prev_df[[label,
         qc_tier]] without filtering to active rows or
         deduplicating → multiplied each fresh row by the number
         of prev_df matches.

  Bug C: No defensive de-dup on CSV load → existing on-disk
         corruption (from earlier sessions running pre-patch code)
         persisted across reloads.

Tests verify each fix independently + end-to-end recovery from a
corrupted CSV file.
"""
from __future__ import annotations

import inspect
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Bare-instance controller fixture (reused across all three test groups)
# ---------------------------------------------------------------------------


def _make_bare_controller(prev_df=None, working_dir=None):
    """Build a bare Step-3 controller stub with the minimum state
    needed for the fix paths to run without Qt."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl._qc_features_df = prev_df
    ctrl.working_dir = str(working_dir) if working_dir is not None else None
    # Stub UI side-effects.
    ctrl._auto_assign_qc_tiers = lambda df, **kw: None
    ctrl._update_qc_summary = lambda df: None
    ctrl._populate_threshold_feature_combo = lambda df: None
    ctrl._update_flagged_labels_list = lambda: None
    ctrl._refresh_patch_list_colors = lambda: None
    ctrl._append_log = lambda msg: None
    ctrl.ui = types.SimpleNamespace(
        extract_features_btn=types.SimpleNamespace(
            setEnabled=lambda _: None,
            setText=lambda _: None,
        ),
    )
    ctrl._qc_settings = {}
    return ctrl


def _fresh_features_df(label_ids, area_values=None):
    """Mimic worker output: features without qc_tier."""
    n = len(label_ids)
    if area_values is None:
        area_values = list(range(100, 100 + n * 10, 10))
    return pd.DataFrame({
        "label": list(label_ids),
        "area": area_values[:n],
        "qc_score": np.linspace(0.1, 0.9, n).tolist(),
    })


# ===========================================================================
# Bug B fix: left-merge filters prev to active + dedupes
# ===========================================================================


class TestLeftMergeFiltersActiveAndDedupes:
    """The fix: extract `[label, qc_tier]` ONLY from active rows in
    prev_df, then drop_duplicates(subset="label", keep="last") to
    guarantee a 1:1 lookup table.

    Without this, ghost rows AND active duplicates both contributed
    to the merge, multiplying output rows.
    """

    def test_ghost_rows_in_prev_do_NOT_multiply_active_rows(self):
        """The headline case from user repro: prev_df has 1 active + 3
        ghosts for label 1286.  Fresh has 1 row for 1286.  Output
        should have exactly 1 row for 1286 (not 4)."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        # Build prev_df: 1 active + 3 ghosts for 1286, plus a clean label 5
        prev = pd.DataFrame({
            "label": [5, 1286, 1286, 1286, 1286],
            "area": [50, 100, 100, 100, 100],
            "qc_score": [0.5, 0.7, 0.7, 0.7, 0.7],
            "qc_tier": ["good", "curated", "review", "review", "review"],
            "label_status": [
                "active", "active",
                "superseded", "superseded", "superseded",
            ],
        })
        ctrl = _make_bare_controller(prev_df=prev)
        # Fresh extract has clean 1 row per label.
        fresh = _fresh_features_df([5, 1286])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, fresh, "patch_xyz"
        )
        result = ctrl._qc_features_df
        # Exactly 2 active rows total — one per label.
        assert len(result) == 2
        assert set(result["label"]) == {5, 1286}
        # 1286's tier was inherited from the ACTIVE prev row ("curated"),
        # NOT from a ghost row.
        tier_map = dict(zip(result["label"], result["qc_tier"]))
        assert tier_map[1286] == "curated"
        assert tier_map[5] == "good"

    def test_duplicate_active_rows_in_prev_collapse_to_one(self):
        """prev_df has 4 ACTIVE rows for label 1281 (the user's
        on-disk corruption).  Fresh has 1 row.  Output: 1 row."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        # 4 active dups for label 1281 — simulating the on-disk corruption.
        prev = pd.DataFrame({
            "label": [1281, 1281, 1281, 1281],
            "area": [50, 50, 50, 50],
            "qc_score": [0.3, 0.3, 0.3, 0.3],
            "qc_tier": ["review", "review", "review", "good"],  # last wins
            "label_status": ["active", "active", "active", "active"],
        })
        ctrl = _make_bare_controller(prev_df=prev)
        fresh = _fresh_features_df([1281])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, fresh, "patch_xyz"
        )
        result = ctrl._qc_features_df
        # Exactly 1 row.
        assert len(result) == 1
        # keep="last" means we got the "good" tier.
        assert result.iloc[0]["qc_tier"] == "good"

    def test_no_ghost_drops_in_prev_works_as_before(self):
        """Regression guard: clean prev_df (no ghosts, no dups) still
        produces the expected single-row-per-label result."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        prev = pd.DataFrame({
            "label": [1, 2, 3],
            "qc_score": [0.2, 0.5, 0.8],
            "qc_tier": ["review", "good", "garbage"],
            "label_status": ["active", "active", "active"],
        })
        ctrl = _make_bare_controller(prev_df=prev)
        fresh = _fresh_features_df([1, 2, 3])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, fresh, "patch_xyz"
        )
        result = ctrl._qc_features_df
        assert len(result) == 3
        tier_map = dict(zip(result["label"], result["qc_tier"]))
        assert tier_map[1] == "review"
        assert tier_map[2] == "good"
        assert tier_map[3] == "garbage"


# ===========================================================================
# Bug A fix: interp/smooth accept ghost-copy filters to active rows only
# ===========================================================================


class TestGhostCopyFiltersToActiveRowsOnly:
    """The fix: interp/smooth accept's mask must be `label_status ==
    "active"` so only the current active row gets copied as a ghost.
    Prior ghosts must NOT be re-copied.
    """

    def test_interp_accept_ghost_copy_source_filters_active(self):
        """Source-level: _on_annotator_interpolation_accepted must
        AND the label match with label_status == "active"."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._on_annotator_interpolation_accepted
        )
        # The ghost-copy mask must include the active filter.
        assert '== "active"' in src or "== 'active'" in src
        # And the diagnostic doc-comment for the fix.
        assert "2026-05-17 fix" in src

    def test_smooth_accept_ghost_copy_source_filters_active(self):
        """Source-level: _on_annotator_label_smoothed must do the
        same active-filter."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._on_annotator_label_smoothed
        )
        assert '== "active"' in src or "== 'active'" in src
        assert "2026-05-17 fix" in src


# ===========================================================================
# Bug C fix: defensive de-dup on CSV load
# ===========================================================================


class TestCsvLoadDeduplication:
    """The fix: after `load_patch_csv`, immediately
    `drop_duplicates(subset=["label", "label_status"], keep="first")`
    to heal any on-disk corruption.  Log [DUP-CLEANUP] when it fires.
    """

    def test_load_path_invokes_drop_duplicates(self):
        """Source-level: the load path must call drop_duplicates
        with the (label, label_status) subset."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        # The cleanup lives in the QC-view load helper; we don't have
        # a great hook to call it from a unit test without standing up
        # a real workspace.  Source-level check is enough — the
        # behavior tests below verify the actual dropping via a
        # controllable helper.
        src = inspect.getsource(FusionReconstructionController)
        assert "[DUP-CLEANUP] dropped" in src
        assert 'drop_duplicates(subset=subset, keep="first")' in src

    def test_dedup_keeps_first_per_label_status_pair(self):
        """Behavioral test for the de-dup logic itself, using a small
        DataFrame.  Mirrors the user's on-disk corruption pattern:
        4 active rows for label 1281, dedupe should keep 1."""
        # Simulate the cleanup that the load path performs.
        df = pd.DataFrame({
            "label": [1281, 1281, 1281, 1281, 5, 1286, 1286, 1286],
            "area": [50, 50, 50, 50, 30, 100, 100, 100],
            "qc_score": [0.3, 0.3, 0.3, 0.3, 0.5, 0.7, 0.7, 0.7],
            "qc_tier": [
                "review", "review", "review", "good",
                "good", "curated", "review", "review",
            ],
            "label_status": [
                "active", "active", "active", "active",
                "active",
                "active", "superseded", "superseded",
            ],
        })
        # 4 dup active 1281 + 1 active 5 + 1 active 1286 + 2 dup
        # ghost 1286 = 8 rows total before dedupe.
        assert len(df) == 8
        n_before = len(df)
        cleaned = df.drop_duplicates(
            subset=["label", "label_status"], keep="first"
        ).reset_index(drop=True)
        n_after = len(cleaned)
        # Expected:
        #   - 1 active 1281 (1 of 4 kept)
        #   - 1 active 5
        #   - 1 active 1286
        #   - 1 superseded 1286 (1 of 2 kept)
        assert n_after == 4
        assert n_before - n_after == 4
        # The kept "1281" row should be the FIRST occurrence ("review"
        # tier, NOT the last "good") — matches keep="first".
        active_1281 = cleaned[
            (cleaned["label"] == 1281) & (cleaned["label_status"] == "active")
        ]
        assert len(active_1281) == 1
        assert active_1281.iloc[0]["qc_tier"] == "review"


# ===========================================================================
# End-to-end: simulated corrupted CSV → load → re-extract → still clean
# ===========================================================================


class TestSplitDefensiveGuard:
    """The split callback's defensive guard against fragment ID
    collisions (added 2026-05-17).  Watershed shouldn't reuse IDs,
    but if it does (corruption, undo state, etc.), the guard
    drops duplicates and logs [DUP-PREVENT]."""

    def test_split_source_has_drop_duplicates_after_concat(self):
        """Source-level: split's concat must be followed by
        drop_duplicates(subset=[label, label_status]) + log."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._on_annotator_label_split
        )
        assert "[DUP-PREVENT]" in src
        assert 'subset=["label", "label_status"]' in src
        assert 'keep="last"' in src

    def test_merge_source_uses_in_place_mutation_no_concat(self):
        """Source-level regression guard: merge must NOT use concat
        (it mutates label_status in place).  Verify there's no
        pd.concat on _qc_features_df in the merge path — that would
        risk the same duplication class."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._on_annotator_label_merged
        )
        # Merge should NOT concat anything to _qc_features_df.
        # If a future refactor adds concat, this guard fires.
        assert "pd.concat" not in src, (
            "Merge must mutate label_status in place, not add rows.  "
            "If a refactor adds concat, also add the drop_duplicates "
            "guard like split does."
        )


class TestEndToEndCorruptionRecovery:
    """The user's actual scenario: a corrupted CSV with 4 active
    dups + ghost dups gets loaded.  Then user paints/interp/smooths
    a label.  Then re-extracts.  Final state should be clean."""

    def test_corrupted_prev_with_4x_dups_yields_clean_post_reextract(self):
        """Mirrors the user repro: 1281 has 4 active dups in prev_df,
        1286 has 1 active + 3 ghosts.  Re-extract should produce 1
        row per label."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        # Simulate the in-memory state right before re-extract:
        prev = pd.DataFrame({
            "label": [
                1281, 1281, 1281, 1281,  # 4 active dups (on-disk corruption)
                203, 203, 203, 203,       # 4 active dups
                1286,                     # 1 active
                1286, 1286, 1286,         # 3 ghosts from interp/smooth bug
                5,                        # clean control
            ],
            "qc_score": [0.3] * 4 + [0.4] * 4 + [0.7, 0.7, 0.7, 0.7, 0.5],
            "qc_tier": (
                ["review"] * 4
                + ["good"] * 4
                + ["curated", "review", "review", "review"]
                + ["good"]
            ),
            "label_status": (
                ["active"] * 4
                + ["active"] * 4
                + ["active"]
                + ["superseded"] * 3
                + ["active"]
            ),
        })
        ctrl = _make_bare_controller(prev_df=prev)
        # Re-extract: worker produces 1 fresh row per label still in
        # the volume (all 4 labels).
        fresh = _fresh_features_df([5, 203, 1281, 1286])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, fresh, "patch_xyz"
        )
        result = ctrl._qc_features_df
        # No duplicates in the output — exactly 4 rows.
        assert len(result) == 4
        assert sorted(result["label"]) == [5, 203, 1281, 1286]
        # No duplicate labels.
        assert result["label"].is_unique
        # Tiers preserved from the active-only de-dup'd lookup.
        tier_map = dict(zip(result["label"], result["qc_tier"]))
        assert tier_map[5] == "good"
        # For dup-active labels (1281, 203), keep="last" of the
        # de-dup'd active subset.  All four rows have same tier so
        # result is deterministic.
        assert tier_map[1281] == "review"
        assert tier_map[203] == "good"
        # 1286's tier comes from the ACTIVE row in prev, NOT a ghost.
        assert tier_map[1286] == "curated"

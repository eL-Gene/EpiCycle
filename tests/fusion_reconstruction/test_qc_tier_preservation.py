"""Tests for qc_tier preservation across re-extract (2026-05-16 fix).

Background
----------
Before this fix, ``Controller._on_qc_extract_finished`` would discard
the user's manual qc_tier assignments on ACTIVE labels every time
features were re-extracted:

  1. Worker writes fresh CSV (no qc_tier column) — overwrites disk.
  2. UI receives fresh df with no qc_tier.
  3. Ghost concat brings qc_tier into df.columns via ghost rows.
  4. ``_auto_assign_qc_tiers`` returns early because qc_tier is now
     in columns → active rows stay NaN.

The user's triage work — review/bad/garbage assignments via
threshold gating, lasso, manual tier buttons — was lost.

This is bad for two reasons:
  (a) The user has to redo triage after every re-extract.
  (b) The planned classifier-training use treats every
      manual tier decision as labeled training data.  Discarding
      them on re-extract throws away the substrate the future
      garbage / QC classifier would be trained on.

Fix: left-merge old qc_tier onto fresh df by label_id BEFORE the
existing ghost concat.  Every active label whose ID was in the
previous DataFrame inherits its qc_tier.  De novo labels (new in
this session, no prior row) end up with NaN qc_tier — correct
"untriaged" state.

Tests use bare-instance pattern.  No Qt required for these tests
because the merge logic itself is pure pandas.  UI updates inside
``_on_qc_extract_finished`` are stubbed.
"""
from __future__ import annotations

import inspect
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Test fixture builder
# ---------------------------------------------------------------------------


def _make_bare_controller(prev_df=None, working_dir=None):
    """Build a bare Step-3 controller stub with the minimum state for
    ``_on_qc_extract_finished`` to run without Qt.

    Parameters
    ----------
    prev_df : pd.DataFrame | None
        Previous in-memory ``_qc_features_df`` (None for first-extract).
    working_dir : str | Path | None
        If set, the auto-save block in ``_on_qc_extract_finished`` will
        write the merged df to ``<working_dir>/artifacts/qc/<stem>_features.csv``.
        Default None means the auto-save silently fails (test exercises
        in-memory behavior only).
    """
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl._qc_features_df = prev_df
    ctrl.working_dir = str(working_dir) if working_dir is not None else None
    # Stubs for all UI / side-effect methods invoked downstream of
    # the merge step.  They must be no-ops so the merge logic runs
    # to completion without touching Qt.
    ctrl._auto_assign_qc_tiers = lambda df, **kw: None
    ctrl._update_qc_summary = lambda df: None
    ctrl._populate_threshold_feature_combo = lambda df: None
    ctrl._update_flagged_labels_list = lambda: None
    ctrl._refresh_patch_list_colors = lambda: None
    ctrl._append_log = lambda msg: None
    # ``ui`` namespace stubs — _on_qc_extract_finished touches a few.
    ctrl.ui = types.SimpleNamespace(
        extract_features_btn=types.SimpleNamespace(
            setEnabled=lambda _: None,
            setText=lambda _: None,
        ),
    )
    ctrl._qc_settings = {}
    return ctrl


def _fresh_features_df(label_ids, area_values=None):
    """Mimic what the worker emits: features without qc_tier."""
    n = len(label_ids)
    if area_values is None:
        area_values = list(range(100, 100 + n * 10, 10))
    return pd.DataFrame({
        "label": list(label_ids),
        "area": area_values[:n],
        "qc_score": np.linspace(0.1, 0.9, n).tolist(),
    })


# ===========================================================================
# Core left-merge behavior
# ===========================================================================


class TestQcTierPreservation:

    def test_first_extract_no_prior_df_proceeds_unchanged(self):
        """When there's no prior DataFrame (first-time extract), the
        merge step is skipped and the fresh df flows through as-is."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        ctrl = _make_bare_controller(prev_df=None)
        df = _fresh_features_df([1, 2, 3])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_001"
        )
        result = ctrl._qc_features_df
        assert list(result["label"]) == [1, 2, 3]
        # No qc_tier column added by the merge step (auto-assign is
        # stubbed to no-op here so we can verify the merge alone).
        assert "qc_tier" not in result.columns or result["qc_tier"].isna().all()

    def test_re_extract_preserves_qc_tier_for_matching_labels(self):
        """The headline test: a previous df with qc_tier values for
        labels 1/2/3 should carry those values onto the fresh re-
        extract df for the same labels."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        prev = pd.DataFrame({
            "label": [1, 2, 3],
            "area": [50, 60, 70],          # old features
            "qc_score": [0.2, 0.5, 0.8],
            "qc_tier": ["review", "good", "garbage"],
            "label_status": ["active", "active", "active"],
        })
        ctrl = _make_bare_controller(prev_df=prev)
        # Fresh extract: same 3 labels, new feature values, no qc_tier.
        df = _fresh_features_df([1, 2, 3], area_values=[100, 110, 120])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_001"
        )
        result = ctrl._qc_features_df
        # Features are FRESH (new area values from worker).
        assert list(result.sort_values("label")["area"]) == [100, 110, 120]
        # qc_tier values are PRESERVED from prev_df.
        tier_map = dict(zip(result["label"], result["qc_tier"]))
        assert tier_map[1] == "review"
        assert tier_map[2] == "good"
        assert tier_map[3] == "garbage"

    def test_de_novo_labels_get_nan_qc_tier(self):
        """Labels new in this session (in fresh df but NOT in prev_df)
        should get NaN qc_tier — they're untriaged."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        prev = pd.DataFrame({
            "label": [1, 2],
            "qc_score": [0.3, 0.6],
            "qc_tier": ["review", "good"],
            "label_status": ["active", "active"],
        })
        ctrl = _make_bare_controller(prev_df=prev)
        # Fresh extract finds labels 1, 2, AND a new de novo 99.
        df = _fresh_features_df([1, 2, 99])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_001"
        )
        result = ctrl._qc_features_df
        tier_map = dict(zip(result["label"], result["qc_tier"]))
        assert tier_map[1] == "review"
        assert tier_map[2] == "good"
        # De novo label 99 has NaN qc_tier — pd.isna handles it.
        assert pd.isna(tier_map[99])

    def test_ghosts_DROPPED_on_reextract(self):
        """2026-05-17 design decision: ghosts are dropped on re-extract
        (no longer concat'd onto the fresh df).

        Rationale: ghosts cause NaN bugs when feature schemas drift
        between extracts.  Training-data substrate for the classifier
        path comes from raw-archive + pre-commit snapshots, NOT from
        in-flight ghost rows.
        """
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        prev = pd.DataFrame({
            "label": [1, 2, 5],
            "area": [50, 60, 70],
            "qc_score": [0.2, 0.5, 0.1],
            "qc_tier": ["good", "review", "garbage"],
            "label_status": ["active", "active", "superseded"],
        })
        ctrl = _make_bare_controller(prev_df=prev)
        # Fresh extract: labels 1, 2 still in volume; label 5 was
        # absorbed by merge (label_status="superseded" in prev_df).
        df = _fresh_features_df([1, 2])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_001"
        )
        result = ctrl._qc_features_df
        # Only active labels survive.
        assert set(result["label"]) == {1, 2}
        # No ghost rows.
        if "label_status" in result.columns:
            assert (result["label_status"] == "active").all()

    def test_merge_does_not_duplicate_labels(self):
        """Sanity: the left-merge must not duplicate rows.  Even if
        prev_df had a label appearing twice (defensive corner case),
        the merge must not emit two rows for that label in the fresh
        df."""
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
        df = _fresh_features_df([1, 2, 3])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_001"
        )
        result = ctrl._qc_features_df
        assert len(result) == 3
        assert sorted(result["label"]) == [1, 2, 3]

    def test_prev_df_without_qc_tier_column_handled_safely(self):
        """If prev_df has no qc_tier column (e.g., very early
        extract before auto-assign ran, or test fixture), the merge
        step must skip cleanly without crashing."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        prev = pd.DataFrame({
            "label": [1, 2],
            "qc_score": [0.3, 0.6],
            "label_status": ["active", "active"],
            # NOTE: no qc_tier column
        })
        ctrl = _make_bare_controller(prev_df=prev)
        df = _fresh_features_df([1, 2, 3])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_001"
        )
        # No crash; df assigned.
        assert ctrl._qc_features_df is not None
        assert len(ctrl._qc_features_df) == 3

    def test_prev_df_without_label_column_handled_safely(self):
        """Defensive: if prev_df lacks a 'label' column (corrupt
        state), the merge skips cleanly."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        prev = pd.DataFrame({
            "qc_tier": ["good", "review"],
            "label_status": ["active", "active"],
            # NOTE: no label column
        })
        ctrl = _make_bare_controller(prev_df=prev)
        df = _fresh_features_df([1, 2])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_001"
        )
        # No crash.
        assert ctrl._qc_features_df is not None

    def test_combined_scenario_active_preserved_plus_de_novo_drops_ghosts(self):
        """End-to-end: prev_df has active rows with tiers, a
        superseded ghost row, AND the fresh extract adds a de novo
        label.  Active tiers preserved; de novo gets NaN; ghosts
        DROPPED (2026-05-17 design decision)."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        prev = pd.DataFrame({
            "label": [1, 2, 3, 7],
            "area": [100, 200, 50, 80],
            "qc_score": [0.2, 0.5, 0.8, 0.1],
            "qc_tier": ["good", "review", "garbage", "good"],
            "label_status": ["active", "active", "active", "superseded"],
        })
        ctrl = _make_bare_controller(prev_df=prev)
        # Fresh extract: 1, 2, 3 still in volume + de novo 99.
        # Label 7 was previously superseded — DROPPED on re-extract.
        df = _fresh_features_df([1, 2, 3, 99])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_001"
        )
        result = ctrl._qc_features_df
        # 4 active labels, no ghosts.
        assert len(result) == 4
        assert set(result["label"]) == {1, 2, 3, 99}
        # Active tiers preserved; de novo gets NaN.
        tier_map = dict(zip(result["label"], result["qc_tier"]))
        assert tier_map[1] == "good"
        assert tier_map[2] == "review"
        assert tier_map[3] == "garbage"
        assert pd.isna(tier_map[99])
        # Label 7 is gone — no row at all (ghost dropped).
        assert 7 not in result["label"].values


# ===========================================================================
# Auto-save to disk after merge (2026-05-17 fix)
# ===========================================================================


class TestAutoSaveAfterMerge:
    """After the left-merge runs, the merged df must be written back to
    disk so the on-disk CSV reflects the in-memory state.  Without this,
    quitting napari between re-extract and the next manual save loses
    the user's triage decisions — exactly the failure the user hit on
    2026-05-16."""

    def test_disk_csv_has_preserved_tiers_after_reextract(self, tmp_path):
        """The headline test: after re-extract, the on-disk CSV must
        contain the qc_tier values from the prev_df.  Setup mirrors a
        real Step-3 workflow."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        prev = pd.DataFrame({
            "label": [1, 2, 3],
            "area": [50, 60, 70],
            "qc_score": [0.2, 0.5, 0.8],
            "qc_tier": ["review", "good", "garbage"],
            "label_status": ["active", "active", "active"],
        })
        ctrl = _make_bare_controller(prev_df=prev, working_dir=tmp_path)
        df = _fresh_features_df([1, 2, 3], area_values=[100, 110, 120])
        # Patch name must have a stem so the auto-save path resolves.
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_xyz"
        )
        # Verify disk CSV exists and has preserved tiers.
        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "patch_xyz_features.csv"
        assert csv_path.exists(), (
            "Auto-save must write the merged df to disk at "
            "<working_dir>/artifacts/qc/<stem>_features.csv."
        )
        disk_df = pd.read_csv(csv_path)
        assert "qc_tier" in disk_df.columns
        tier_map = dict(zip(disk_df["label"], disk_df["qc_tier"]))
        assert tier_map[1] == "review"
        assert tier_map[2] == "good"
        assert tier_map[3] == "garbage"
        # Fresh feature values must be on disk too (not the old ones).
        area_map = dict(zip(disk_df["label"], disk_df["area"]))
        assert area_map[1] == 100
        assert area_map[2] == 110
        assert area_map[3] == 120

    def test_disk_csv_DROPS_ghost_rows(self, tmp_path):
        """2026-05-17 design decision: ghosts are dropped on
        re-extract.  The disk CSV after re-extract contains only
        active rows."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        prev = pd.DataFrame({
            "label": [1, 2, 7],
            "area": [50, 60, 80],
            "qc_score": [0.2, 0.5, 0.1],
            "qc_tier": ["good", "review", "garbage"],
            "label_status": ["active", "active", "superseded"],
        })
        ctrl = _make_bare_controller(prev_df=prev, working_dir=tmp_path)
        df = _fresh_features_df([1, 2])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_xyz"
        )
        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "patch_xyz_features.csv"
        disk_df = pd.read_csv(csv_path)
        # 2 active rows, 0 ghosts.
        assert len(disk_df) == 2
        assert set(disk_df["label"]) == {1, 2}
        # No ghost rows survived to disk.
        assert 7 not in disk_df["label"].values

    def test_no_working_dir_silently_skips_auto_save(self, tmp_path):
        """If working_dir is None (defensive — should never happen in
        production but tests / weird states might trigger), the
        auto-save must fail silently without raising."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        prev = pd.DataFrame({
            "label": [1, 2],
            "qc_score": [0.3, 0.6],
            "qc_tier": ["review", "good"],
            "label_status": ["active", "active"],
        })
        ctrl = _make_bare_controller(prev_df=prev, working_dir=None)
        df = _fresh_features_df([1, 2])
        # Must NOT raise.
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_xyz"
        )
        # In-memory state still has tiers.
        result = ctrl._qc_features_df
        tier_map = dict(zip(result["label"], result["qc_tier"]))
        assert tier_map[1] == "review"
        assert tier_map[2] == "good"

    def test_first_extract_no_prev_df_skips_auto_save(self, tmp_path):
        """First-time extract has no prev_df.  The auto-save block is
        gated on ``prev_df is not None`` because there's nothing to
        preserve — the worker's own write at line 728 already wrote
        the fresh features to disk."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        ctrl = _make_bare_controller(prev_df=None, working_dir=tmp_path)
        df = _fresh_features_df([1, 2, 3])
        FusionReconstructionController._on_qc_extract_finished(
            ctrl, df, "patch_xyz"
        )
        # Auto-save was skipped — CSV should NOT have been written by
        # this code path.  (The worker would write it in production,
        # but we're testing _on_qc_extract_finished in isolation.)
        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "patch_xyz_features.csv"
        assert not csv_path.exists()


# ===========================================================================
# Source-level: documents the fix
# ===========================================================================


class TestSourceWiring:

    def test_extract_finished_references_re_extract_diagnostic(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._on_qc_extract_finished
        )
        assert "[RE-EXTRACT]" in src

    def test_extract_finished_does_left_merge_on_label(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._on_qc_extract_finished
        )
        # Confirm the merge by label_id is present and explicitly left.
        assert "df.merge(" in src
        assert 'on="label"' in src
        assert 'how="left"' in src

    def test_extract_finished_captures_prev_df_before_reassign(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._on_qc_extract_finished
        )
        # prev_df capture must come BEFORE self._qc_features_df = df
        prev_idx = src.find("prev_df = self._qc_features_df")
        assign_idx = src.find("self._qc_features_df = df")
        assert prev_idx != -1
        assert assign_idx != -1
        assert prev_idx < assign_idx

    def test_classifier_vision_reference_in_docstring(self):
        """The merge keeps every tier's label because each tier is
        training signal for a future classifier; the source comment
        must say so, so future contributors don't strip the merge as
        'unnecessary'."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._on_qc_extract_finished
        )
        assert "every tier IS labeled" in src

    def test_extract_finished_calls_save_patch_csv_on_reextract(self):
        """Auto-save block must invoke save_patch_csv after the merge
        so the disk CSV catches up with the in-memory merged state."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._on_qc_extract_finished
        )
        assert "save_patch_csv" in src

    def test_auto_save_block_is_best_effort(self):
        """Auto-save failure must not propagate — in-memory state is
        the authority.  Source must wrap save_patch_csv in try/except."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._on_qc_extract_finished
        )
        # The auto-save block must wrap save_patch_csv in try/except.
        # Naive substring check: find the auto-save block by its
        # diagnostic prefix, then ensure "except" appears nearby.
        auto_save_idx = src.find("[RE-EXTRACT] Auto-saved")
        assert auto_save_idx != -1, (
            "Auto-save diagnostic [RE-EXTRACT] Auto-saved must be "
            "present so manual testers can verify the write fired."
        )
        # Look within a reasonable window after the diagnostic for
        # the except clause.
        block = src[max(0, auto_save_idx - 600):auto_save_idx + 600]
        assert "except Exception" in block
        assert "Auto-save to disk FAILED" in block

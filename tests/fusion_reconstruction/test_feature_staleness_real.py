"""WP2 + WP3 + WP5: Real-path tests for feature staleness.

Tests that exercise actual production code:
- ``merge_patch_csvs()`` backward-compat backfill (WP2)
- ``load_patch_csv()`` backward-compat backfill
- ``extract_features()`` produces ``feature_state`` column
- ``_commit_curation_action()`` action-aware staleness (WP3)
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Tests: WP2 — merge_patch_csvs backward compat
# ---------------------------------------------------------------------------

class TestMergePatchCsvsBackfill:
    """Verify merge_patch_csvs adds feature_state for old CSVs."""

    def test_old_csvs_get_feature_state(self, tmp_path):
        """CSVs without feature_state column get 'fresh' after merge."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            merge_patch_csvs,
        )

        # Create two old-style CSVs without feature_state
        for name in ["patchA_features.csv", "patchB_features.csv"]:
            df = pd.DataFrame({
                "label": [1, 2, 3],
                "area": [100, 200, 300],
                "qc_tier": ["good", "review", "bad"],
            })
            df.to_csv(tmp_path / name, index=False)

        merged = merge_patch_csvs([
            tmp_path / "patchA_features.csv",
            tmp_path / "patchB_features.csv",
        ])

        assert "feature_state" in merged.columns
        assert (merged["feature_state"] == "fresh").all()
        assert len(merged) == 6
        assert set(merged["patch_name"]) == {"patchA", "patchB"}

    def test_new_csvs_preserve_feature_state(self, tmp_path):
        """CSVs with feature_state preserve existing values after merge."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            merge_patch_csvs,
        )

        df = pd.DataFrame({
            "label": [1, 2],
            "area": [100, 200],
            "feature_state": ["fresh", "stale"],
        })
        df.to_csv(tmp_path / "patch_features.csv", index=False)

        merged = merge_patch_csvs([tmp_path / "patch_features.csv"])

        assert merged.iloc[0]["feature_state"] == "fresh"
        assert merged.iloc[1]["feature_state"] == "stale"

    def test_mixed_old_and_new_csvs(self, tmp_path):
        """Mix of old (no column) and new (with column) CSVs works."""
        from napari_epicycle.fusion_reconstruction.qc_features import (
            merge_patch_csvs,
        )

        # Old CSV (no feature_state)
        pd.DataFrame({
            "label": [1], "area": [100],
        }).to_csv(tmp_path / "old_features.csv", index=False)

        # New CSV (with feature_state)
        pd.DataFrame({
            "label": [2], "area": [200], "feature_state": ["stale"],
        }).to_csv(tmp_path / "new_features.csv", index=False)

        merged = merge_patch_csvs([
            tmp_path / "old_features.csv",
            tmp_path / "new_features.csv",
        ])

        assert "feature_state" in merged.columns
        assert merged[merged["patch_name"] == "old"]["feature_state"].iloc[0] == "fresh"
        assert merged[merged["patch_name"] == "new"]["feature_state"].iloc[0] == "stale"


# ---------------------------------------------------------------------------
# Tests: load_patch_csv backward compat
# ---------------------------------------------------------------------------

class TestLoadPatchCsvBackfill:

    def test_old_csv_gets_fresh(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            load_patch_csv,
        )

        df = pd.DataFrame({"label": [1, 2], "area": [100, 200]})
        csv_path = tmp_path / "features.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_patch_csv(csv_path)
        assert "feature_state" in loaded.columns
        assert (loaded["feature_state"] == "fresh").all()

    def test_new_csv_preserves_state(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            load_patch_csv,
        )

        df = pd.DataFrame({
            "label": [1, 2],
            "area": [100, 200],
            "feature_state": ["stale", "fresh"],
        })
        csv_path = tmp_path / "features.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_patch_csv(csv_path)
        assert loaded.iloc[0]["feature_state"] == "stale"
        assert loaded.iloc[1]["feature_state"] == "fresh"


# ---------------------------------------------------------------------------
# Tests: extract_features produces feature_state
# ---------------------------------------------------------------------------

class TestExtractFeaturesState:

    def test_extract_features_has_feature_state(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            extract_features,
        )

        labels = np.zeros((4, 8, 8), dtype=np.uint32)
        labels[1:3, 2:6, 2:6] = 1

        df = extract_features(labels)
        assert "feature_state" in df.columns
        assert (df["feature_state"] == "fresh").all()


# ---------------------------------------------------------------------------
# Tests: WP3 — action-aware staleness
# ---------------------------------------------------------------------------

class TestActionAwareStaleness:
    """_commit_curation_action marks stale only for mask-mutating actions."""

    def _make_harness(self):
        """Minimal harness to call _commit_curation_action."""
        from unittest.mock import MagicMock
        import types
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        harness = MagicMock()
        harness._qc_features_df = pd.DataFrame({
            "label": [1, 2, 3],
            "area": [100, 200, 300],
            "feature_state": ["fresh", "fresh", "fresh"],
            "qc_tier": ["good", "review", "bad"],
        })
        harness._curation_dirty = False
        harness._curation_undo_stack = MagicMock()
        harness._MASK_MUTATING_ACTIONS = FusionReconstructionController._MASK_MUTATING_ACTIONS

        # Bind real method
        real_method = FusionReconstructionController._commit_curation_action
        harness._commit_curation_action = types.MethodType(real_method, harness)

        return harness

    def test_delete_marks_stale(self):
        h = self._make_harness()
        h._commit_curation_action("delete", 2, "Delete label 2")
        assert h._qc_features_df[h._qc_features_df["label"] == 2]["feature_state"].iloc[0] == "stale"
        # Others stay fresh
        assert h._qc_features_df[h._qc_features_df["label"] == 1]["feature_state"].iloc[0] == "fresh"

    def test_split_marks_stale(self):
        h = self._make_harness()
        h._commit_curation_action("split", 1, "Split label 1")
        assert h._qc_features_df[h._qc_features_df["label"] == 1]["feature_state"].iloc[0] == "stale"

    def test_interpolation_marks_stale(self):
        h = self._make_harness()
        h._commit_curation_action("interpolation", 3, "Interpolate label 3")
        assert h._qc_features_df[h._qc_features_df["label"] == 3]["feature_state"].iloc[0] == "stale"

    def test_smooth_marks_stale(self):
        h = self._make_harness()
        h._commit_curation_action("smooth", 1, "Smooth label 1")
        assert h._qc_features_df[h._qc_features_df["label"] == 1]["feature_state"].iloc[0] == "stale"

    def test_mark_good_does_NOT_mark_stale(self):
        h = self._make_harness()
        h._commit_curation_action("mark_good", 2, "Mark good label 2")
        # Tier-only action — should stay fresh
        assert h._qc_features_df[h._qc_features_df["label"] == 2]["feature_state"].iloc[0] == "fresh"

    def test_explicit_marks_stale_override(self):
        """marks_stale=True forces staleness even for non-mask actions."""
        h = self._make_harness()
        h._commit_curation_action("mark_good", 2, "Force stale", marks_stale=True)
        assert h._qc_features_df[h._qc_features_df["label"] == 2]["feature_state"].iloc[0] == "stale"

    def test_explicit_marks_stale_false_override(self):
        """marks_stale=False prevents staleness even for mask actions."""
        h = self._make_harness()
        h._commit_curation_action("delete", 2, "No stale", marks_stale=False)
        assert h._qc_features_df[h._qc_features_df["label"] == 2]["feature_state"].iloc[0] == "fresh"


# ---------------------------------------------------------------------------
# Tests: R2-WP3 — Split-created rows must be stale
# ---------------------------------------------------------------------------

class TestSplitCreatedRowsStaleness:
    """After split, EVERY new label row must have feature_state='stale'."""

    def _make_split_harness(self):
        """Harness with real _on_annotator_label_split bound."""
        from unittest.mock import MagicMock
        import types
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        harness = MagicMock()
        harness._qc_features_df = pd.DataFrame({
            "label": [1, 2, 3],
            "area": [100, 200, 300],
            "feature_state": ["fresh", "fresh", "fresh"],
            "qc_tier": ["good", "review", "bad"],
            "centroid-0": [10.0, 20.0, 30.0],
            "centroid-1": [10.0, 20.0, 30.0],
            "centroid-2": [10.0, 20.0, 30.0],
        })
        harness._curation_dirty = False
        harness._curation_undo_stack = MagicMock()
        harness._MASK_MUTATING_ACTIONS = FusionReconstructionController._MASK_MUTATING_ACTIONS

        # Fake labels layer with split result
        labels = np.zeros((4, 8, 8), dtype=np.uint32)
        labels[1, 2:5, 2:4] = 10  # new label 10
        labels[1, 2:5, 4:6] = 11  # new label 11
        layer = MagicMock()
        layer.data = labels
        harness._find_active_labels_layer = MagicMock(return_value=layer)
        # D2.4 (perf push 2026-05-10): refinement-op centroid updates
        # consult _current_downsample_factor.  Stub to return 1 so the
        # MagicMock's auto-mock doesn't break the int comparison.
        harness._current_downsample_factor = MagicMock(return_value=1)

        # Bind real methods
        for name in ("_on_annotator_label_split", "_commit_curation_action"):
            real = getattr(FusionReconstructionController, name)
            harness.__dict__[name] = types.MethodType(real, harness)

        return harness

    def test_split_new_rows_are_stale(self):
        """Each new label from split must have feature_state='stale'."""
        h = self._make_split_harness()
        h._on_annotator_label_split(1, [10, 11])

        df = h._qc_features_df
        for new_id in [10, 11]:
            rows = df[df["label"] == new_id]
            assert len(rows) == 1, f"Expected exactly 1 row for label {new_id}"
            assert rows.iloc[0]["feature_state"] == "stale", (
                f"Label {new_id} should be stale after split, got "
                f"'{rows.iloc[0]['feature_state']}'"
            )

    def test_split_marks_original_superseded(self):
        """Original label row should be marked superseded after split."""
        h = self._make_split_harness()
        h._on_annotator_label_split(1, [10, 11])

        df = h._qc_features_df
        orig = df[df["label"] == 1]
        assert len(orig) == 1, "Original label row should still exist as ghost"
        assert orig.iloc[0]["label_status"] == "superseded"

    def test_split_non_target_stays_fresh(self):
        """Labels not involved in split keep their original feature_state."""
        h = self._make_split_harness()
        h._on_annotator_label_split(1, [10, 11])

        df = h._qc_features_df
        assert df[df["label"] == 2].iloc[0]["feature_state"] == "fresh"
        assert df[df["label"] == 3].iloc[0]["feature_state"] == "fresh"

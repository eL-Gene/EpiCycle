"""Tests for QC Provenance DataFrame — ghost rows (deleted/superseded labels).

Validates that curation actions mark rows instead of deleting them, ghost rows
are preserved across re-extraction, and plot/navigator filtering works correctly.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(labels, *, tier="review", status="active"):
    """Create a minimal QC DataFrame."""
    rows = []
    for lid in labels:
        rows.append({
            "label": lid,
            "area": 100 + lid,
            "centroid-0": 10.0,
            "centroid-1": 20.0,
            "centroid-2": 30.0,
            "qc_tier": tier,
            "qc_score": float(lid),
            "feature_state": "fresh",
            "label_status": status,
        })
    return pd.DataFrame(rows)


def _make_df_mixed(active_ids, ghost_ids, ghost_status="deleted"):
    """Create a DF with both active and ghost rows."""
    active = _make_df(active_ids, status="active")
    ghosts = _make_df(ghost_ids, status=ghost_status)
    return pd.concat([active, ghosts], ignore_index=True)


# ---------------------------------------------------------------------------
# Tests: qc_features.active_rows()
# ---------------------------------------------------------------------------

class TestActiveRows:
    """Tests for the active_rows() filter function."""

    def test_all_active(self):
        from napari_epicycle.fusion_reconstruction.qc_features import active_rows
        df = _make_df([1, 2, 3])
        result = active_rows(df)
        assert len(result) == 3

    def test_filters_deleted(self):
        from napari_epicycle.fusion_reconstruction.qc_features import active_rows
        df = _make_df_mixed([1, 2], [3, 4], "deleted")
        result = active_rows(df)
        assert len(result) == 2
        assert set(result["label"].values) == {1, 2}

    def test_filters_superseded(self):
        from napari_epicycle.fusion_reconstruction.qc_features import active_rows
        df = _make_df_mixed([1], [2], "superseded")
        result = active_rows(df)
        assert len(result) == 1
        assert result.iloc[0]["label"] == 1

    def test_no_label_status_column(self):
        from napari_epicycle.fusion_reconstruction.qc_features import active_rows
        df = pd.DataFrame({"label": [1, 2], "area": [10, 20]})
        result = active_rows(df)
        assert len(result) == 2  # all rows returned

    def test_empty_df(self):
        from napari_epicycle.fusion_reconstruction.qc_features import active_rows
        df = _make_df([])
        result = active_rows(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: load_patch_csv backward compat
# ---------------------------------------------------------------------------

class TestLoadPatchCSV:
    """Ensure label_status column is added on load if missing."""

    def test_adds_label_status_column(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            load_patch_csv, save_patch_csv,
        )
        # Save a DF without label_status
        df = pd.DataFrame({
            "label": [1, 2],
            "area": [100, 200],
            "feature_state": ["fresh", "fresh"],
        })
        csv_path = tmp_path / "test_features.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_patch_csv(csv_path)
        assert "label_status" in loaded.columns
        assert (loaded["label_status"] == "active").all()

    def test_preserves_existing_label_status(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            load_patch_csv,
        )
        df = pd.DataFrame({
            "label": [1, 2],
            "area": [100, 200],
            "feature_state": ["fresh", "fresh"],
            "label_status": ["active", "deleted"],
        })
        csv_path = tmp_path / "test_features.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_patch_csv(csv_path)
        assert loaded.loc[0, "label_status"] == "active"
        assert loaded.loc[1, "label_status"] == "deleted"


# ---------------------------------------------------------------------------
# Tests: merge_patch_csvs backward compat
# ---------------------------------------------------------------------------

class TestMergePatchCSVs:
    def test_adds_label_status_on_merge(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.qc_features import merge_patch_csvs
        df1 = pd.DataFrame({
            "label": [1, 2],
            "area": [100, 200],
        })
        df2 = pd.DataFrame({
            "label": [3],
            "area": [300],
            "label_status": ["deleted"],
        })
        p1 = tmp_path / "patch1_features.csv"
        p2 = tmp_path / "patch2_features.csv"
        df1.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)

        merged = merge_patch_csvs([p1, p2])
        assert "label_status" in merged.columns
        # patch1 rows should get default "active"
        patch1_rows = merged[merged["patch_name"] == "patch1"]
        assert (patch1_rows["label_status"] == "active").all()
        # patch2 row keeps "deleted"
        patch2_rows = merged[merged["patch_name"] == "patch2"]
        assert patch2_rows.iloc[0]["label_status"] == "deleted"


# ---------------------------------------------------------------------------
# Minimal controller harness for curation callback tests
# ---------------------------------------------------------------------------

class _FakeUI:
    """Minimal fake UI matching controller expectations."""
    def __init__(self):
        self.qc_summary_label = MagicMock()
        self.save_curation_btn = MagicMock()
        self.undo_curation_btn = MagicMock()
        self._flagged_labels_list = MagicMock()
        self.qc_plot_widget = MagicMock()
        self.qc_plot_widget._color_combo = MagicMock()
        self.qc_plot_widget._color_combo.count = MagicMock(return_value=0)
        self.qc_plot_widget._cmap_combo = MagicMock()
        self.qc_plot_widget._cmap_combo.count = MagicMock(return_value=0)
        self.undo_tier_btn = MagicMock()
        self.extract_features_btn = MagicMock()


class _CurationCallbackHarness:
    """Wraps controller curation callbacks with minimal state."""

    _MASK_MUTATING_ACTIONS = {"delete", "split", "merge", "smooth",
                              "interpolation", "discard_garbage", "carver",
                              "fill_holes", "flood_fill"}
    # Mirror the controller's tier whitelist used by _on_annotator_mark_tier.
    _VALID_TIERS = frozenset({"curated", "good", "review", "bad", "garbage"})

    def __init__(self):
        self._qc_features_df = None
        self._curation_dirty = False
        self._curation_undo_stack = MagicMock()
        self._curation_undo_stack.__bool__ = MagicMock(return_value=False)
        self._tier_undo_stack = MagicMock()
        self._tier_undo_stack.clear = MagicMock()
        self._committing_triage = False
        self.ui = _FakeUI()
        self.viewer = MagicMock()
        self._flagged_centroids = {}
        self._log = []

    def _append_log(self, msg):
        self._log.append(msg)

    def _find_active_labels_layer(self):
        return None

    def _update_flagged_labels_list(self):
        pass

    def _update_qc_summary(self, df):
        pass

    def _refresh_curation_ui(self):
        pass

    def _refresh_patch_list_colors(self):
        pass

    def _update_label_basic_stats(self, label_id, *, roi_slices=None):
        pass

    # _is_population_mode harness stub removed 2026-05-03
    # (drop-population-mode Commit 5/6) — production method gone.

    def _commit_curation_action(self, action_type, label_id, description,
                                *, roi_slices=None, old_roi_data=None,
                                df_updater=None, extra=None,
                                marks_stale=None):
        """Simplified version: just run the updater and mark dirty."""
        if df_updater is not None:
            df_updater()
        self._curation_dirty = True


# Import the real controller class for method binding.
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as _Ctrl,
)

# Bind real controller methods as unbound functions (avoid double-self).
_BOUND_METHODS = [
    "_on_annotator_label_deleted",
    "_on_annotator_discard_garbage",
    "_on_annotator_label_split",
    "_on_annotator_label_merged",
    "_on_annotator_interpolation_accepted",
    "_on_annotator_label_smoothed",
    "_on_annotator_label_accepted",
    "_on_annotator_mark_tier",
    "_on_qc_extract_finished",
    "_build_flagged_label_items",
    "_commit_triage_single_patch",
    # 2026-06-07: _commit_triage_single_patch is now a thin synchronous
    # orchestrator (prep → pure persist → finalize) so the async button path
    # and the tests share one code path.  Bind the prep + finalize helpers it
    # delegates to, or the provenance test would AttributeError.
    "_commit_triage_prepare",
    "_on_commit_triage_finished",
    "_commit_triage_partial_failure_refresh",
    "_maybe_lift_for_save",
    "_scaled_centroid_from_row",  # D2.1 helper used by _build_flagged_label_items
    "_current_downsample_factor",  # D2.4 helper used by _scaled_centroid_from_row
    # 2026-05-17 diagnostic: every curation callback ends by calling
    # this read-only logger; harness must have it bound or tests
    # raise AttributeError.
    "_scan_qc_df_for_duplicates",
    # 2026-06-05 (integration/step3-patch-perf): _on_qc_extract_finished now
    # hides the generalized busy bar on every terminal path; bind the
    # self-contained helpers so the provenance tests don't AttributeError.
    "_busy_show",
    "_busy_progress",
    "_busy_hide",
]

for _name in _BOUND_METHODS:
    _fn = getattr(_Ctrl, _name)
    setattr(_CurationCallbackHarness, _name, _fn)


# ---------------------------------------------------------------------------
# Tests: Curation callbacks produce ghost rows
# ---------------------------------------------------------------------------

class TestDeleteMarksGhost:
    """_on_annotator_label_deleted should mark rows as deleted, not drop them."""

    def test_delete_marks_deleted(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2, 3])
        h._on_annotator_label_deleted(2)

        assert len(h._qc_features_df) == 3  # no rows dropped
        row2 = h._qc_features_df[h._qc_features_df["label"] == 2].iloc[0]
        assert row2["label_status"] == "deleted"

    def test_delete_preserves_active(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2, 3])
        h._on_annotator_label_deleted(2)

        active = h._qc_features_df[h._qc_features_df["label_status"] == "active"]
        assert set(active["label"].values) == {1, 3}

    def test_delete_nonexistent_label(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2])
        h._on_annotator_label_deleted(99)
        # No crash, no change
        assert len(h._qc_features_df) == 2
        assert (h._qc_features_df["label_status"] == "active").all()


class TestDiscardGarbageMarksGhost:
    """_on_annotator_discard_garbage should mark rows as deleted."""

    def test_discard_marks_deleted(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2, 3, 4])
        h._on_annotator_discard_garbage([2, 4])

        assert len(h._qc_features_df) == 4  # no rows dropped
        deleted = h._qc_features_df[h._qc_features_df["label_status"] == "deleted"]
        assert set(deleted["label"].values) == {2, 4}


class TestSplitCreatesGhost:
    """_on_annotator_label_split should mark original as superseded."""

    def test_split_marks_superseded(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2, 3])
        h._on_annotator_label_split(2, [20, 21])

        # Original label 2 should be superseded
        orig = h._qc_features_df[
            (h._qc_features_df["label"] == 2) &
            (h._qc_features_df["label_status"] == "superseded")
        ]
        assert len(orig) == 1

        # New labels should be active
        for new_id in [20, 21]:
            new_rows = h._qc_features_df[
                (h._qc_features_df["label"] == new_id) &
                (h._qc_features_df["label_status"] == "active")
            ]
            assert len(new_rows) == 1
            assert new_rows.iloc[0]["qc_tier"] == "curated"

    def test_split_total_rows(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2, 3])
        h._on_annotator_label_split(2, [20, 21])
        # 3 original + 2 new = 5 rows (original stays as ghost)
        assert len(h._qc_features_df) == 5


class TestMergeCreatesGhost:
    """_on_annotator_label_merged should mark absorbed as superseded.

    The callback now accepts either an int (legacy single-label path)
    or a list (new batch path from the floating MergeDialog).
    """

    def test_merge_marks_superseded_legacy_int_path(self):
        """Backward-compat: passing absorb_ids as a single int still works."""
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2, 3])
        h._on_annotator_label_merged(keep_id=1, absorb_ids=2)

        assert len(h._qc_features_df) == 3  # no rows dropped
        absorbed = h._qc_features_df[h._qc_features_df["label"] == 2].iloc[0]
        assert absorbed["label_status"] == "superseded"

        # Keep label stays active
        kept = h._qc_features_df[h._qc_features_df["label"] == 1].iloc[0]
        assert kept["label_status"] == "active"

    def test_merge_marks_superseded_batch_path(self):
        """New batch path: list of absorb_ids merges all into keep."""
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2, 3, 4])
        h._on_annotator_label_merged(keep_id=1, absorb_ids=[2, 3])

        # All absorbed rows marked superseded.
        for absorb_id in (2, 3):
            absorbed = h._qc_features_df[
                h._qc_features_df["label"] == absorb_id
            ].iloc[0]
            assert absorbed["label_status"] == "superseded"

        # Keep label stays active.
        kept = h._qc_features_df[h._qc_features_df["label"] == 1].iloc[0]
        assert kept["label_status"] == "active"

        # Untouched label 4 stays active.
        other = h._qc_features_df[h._qc_features_df["label"] == 4].iloc[0]
        assert other["label_status"] == "active"

    def test_merge_sets_keep_tier_to_curated(self):
        """Keep label's qc_tier is set to 'curated' on accept (per spec)."""
        h = _CurationCallbackHarness()
        df = _make_df([1, 2, 3])
        df["qc_tier"] = "review"  # all start as review
        h._qc_features_df = df
        h._on_annotator_label_merged(keep_id=1, absorb_ids=[2])

        kept = h._qc_features_df[
            (h._qc_features_df["label"] == 1)
            & (h._qc_features_df["label_status"] == "active")
        ].iloc[0]
        assert kept["qc_tier"] == "curated"

    def test_merge_inherits_border_flag_when_any_absorbed_is_border(self):
        """If any absorbed label was a border cell, the keep label
        inherits is_border=True. Border cells are filtered from
        interactive views; preserving the flag prevents the merged
        result from accidentally re-entering them."""
        import numpy as np
        h = _CurationCallbackHarness()
        df = _make_df([1, 2, 3])
        df["is_border"] = [False, True, False]  # absorbed=2 is border
        h._qc_features_df = df
        h._on_annotator_label_merged(keep_id=1, absorb_ids=[2])

        kept = h._qc_features_df[
            (h._qc_features_df["label"] == 1)
            & (h._qc_features_df["label_status"] == "active")
        ].iloc[0]
        assert bool(kept["is_border"]) is True

    def test_merge_does_not_set_border_flag_when_no_absorbed_is_border(self):
        h = _CurationCallbackHarness()
        df = _make_df([1, 2, 3])
        df["is_border"] = [False, False, False]
        h._qc_features_df = df
        h._on_annotator_label_merged(keep_id=1, absorb_ids=[2])

        kept = h._qc_features_df[
            (h._qc_features_df["label"] == 1)
            & (h._qc_features_df["label_status"] == "active")
        ].iloc[0]
        assert bool(kept["is_border"]) is False

    def test_merge_empty_absorb_list_is_noop(self):
        """Defensive: empty absorb_ids list returns without error."""
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2, 3])
        h._on_annotator_label_merged(keep_id=1, absorb_ids=[])

        # All rows still active.
        for label in (1, 2, 3):
            row = h._qc_features_df[h._qc_features_df["label"] == label].iloc[0]
            assert row["label_status"] == "active"


class TestInterpolationCreatesGhost:
    """_on_annotator_interpolation_accepted should snapshot pre-curation row."""

    def test_interpolation_creates_superseded(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2, 3], tier="bad")
        h._on_annotator_interpolation_accepted(2)

        # Should have superseded ghost row for label 2
        ghosts = h._qc_features_df[
            (h._qc_features_df["label"] == 2) &
            (h._qc_features_df["label_status"] == "superseded")
        ]
        assert len(ghosts) == 1
        assert ghosts.iloc[0]["qc_tier"] == "bad"  # original tier preserved

        # Active row should be curated
        active = h._qc_features_df[
            (h._qc_features_df["label"] == 2) &
            (h._qc_features_df["label_status"] == "active")
        ]
        assert len(active) == 1
        assert active.iloc[0]["qc_tier"] == "curated"

    def test_interpolation_total_rows(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2])
        h._on_annotator_interpolation_accepted(2)
        # 2 original + 1 ghost = 3 rows
        assert len(h._qc_features_df) == 3


class TestSmoothCreatesGhost:
    """_on_annotator_label_smoothed should snapshot pre-curation row."""

    def test_smooth_creates_superseded(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2], tier="review")
        h._on_annotator_label_smoothed(1)

        ghosts = h._qc_features_df[
            (h._qc_features_df["label"] == 1) &
            (h._qc_features_df["label_status"] == "superseded")
        ]
        assert len(ghosts) == 1
        assert ghosts.iloc[0]["qc_tier"] == "review"

        active = h._qc_features_df[
            (h._qc_features_df["label"] == 1) &
            (h._qc_features_df["label_status"] == "active")
        ]
        assert len(active) == 1
        assert active.iloc[0]["qc_tier"] == "curated"


class TestMarkGoodActiveOnly:
    """_on_annotator_label_accepted should only update active rows."""

    def test_mark_good_skips_ghost(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df_mixed([1, 2], [2], "superseded")
        # Label 2 exists as both active and superseded
        h._on_annotator_label_accepted(2)

        # Active row should be "good"
        active = h._qc_features_df[
            (h._qc_features_df["label"] == 2) &
            (h._qc_features_df["label_status"] == "active")
        ]
        assert active.iloc[0]["qc_tier"] == "good"

        # Ghost row should keep original tier
        ghost = h._qc_features_df[
            (h._qc_features_df["label"] == 2) &
            (h._qc_features_df["label_status"] == "superseded")
        ]
        assert ghost.iloc[0]["qc_tier"] == "review"


# ---------------------------------------------------------------------------
# Tests: Navigator / flagged list filtering
# ---------------------------------------------------------------------------

class TestBuildFlaggedLabelItemsActiveOnly:
    """_build_flagged_label_items should exclude ghost rows."""

    def test_excludes_ghosts(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df_mixed([1, 2], [3], "deleted")
        items = h._build_flagged_label_items()
        label_ids = {item["label_id"] for item in items}
        assert 3 not in label_ids
        assert {1, 2} == label_ids


# ---------------------------------------------------------------------------
# Tests: Re-extraction preserves ghosts
# ---------------------------------------------------------------------------

class TestReExtractionDropsGhosts:
    """_on_qc_extract_finished DROPS ghost rows on re-extract.

    2026-05-17 design decision (user-confirmed): re-extract gives a
    clean active-only df.  The training-data
    substrate is preserved by raw-archive + pre-commit snapshots,
    not by in-flight ghost rows.
    """

    def test_ghosts_DROPPED(self):
        h = _CurationCallbackHarness()
        # Existing DF with ghosts
        h._qc_features_df = _make_df_mixed([1, 2], [3, 4], "deleted")
        h._auto_assign_qc_tiers = lambda df, **kw: None
        h._update_qc_summary = lambda df: None
        h._populate_threshold_feature_combo = lambda df: None
        h._update_flagged_labels_list = lambda: None

        # Fresh extraction (new labels from current volume)
        fresh_df = _make_df([5, 6, 7])

        # Bind real method
        h._on_qc_extract_finished = _Ctrl._on_qc_extract_finished.__get__(h)
        h._on_qc_extract_finished(fresh_df, "test_patch")

        # Only fresh active rows survive; ghosts dropped.
        result = h._qc_features_df
        assert len(result) == 3

        from napari_epicycle.fusion_reconstruction.qc_features import active_rows
        active = active_rows(result)
        assert set(active["label"].values) == {5, 6, 7}

        # Ghost rows must NOT be present.
        if "label_status" in result.columns:
            assert (result["label_status"] == "active").all()
        assert 3 not in result["label"].values
        assert 4 not in result["label"].values

    def test_first_extraction_no_ghosts(self):
        h = _CurationCallbackHarness()
        h._qc_features_df = None  # no previous data
        h._auto_assign_qc_tiers = lambda df, **kw: None
        h._update_qc_summary = lambda df: None
        h._populate_threshold_feature_combo = lambda df: None
        h._update_flagged_labels_list = lambda: None

        fresh_df = _make_df([1, 2, 3])
        h._on_qc_extract_finished = _Ctrl._on_qc_extract_finished.__get__(h)
        h._on_qc_extract_finished(fresh_df, "test_patch")

        result = h._qc_features_df
        assert len(result) == 3
        assert "label_status" in result.columns
        assert (result["label_status"] == "active").all()


# ---------------------------------------------------------------------------
# Tests: CSV round-trip
# ---------------------------------------------------------------------------

class TestCSVRoundTrip:
    """label_status column survives save + load."""

    def test_roundtrip_preserves_status(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            save_patch_csv, load_patch_csv,
        )
        df = _make_df_mixed([1, 2], [3], "superseded")
        csv_path = tmp_path / "test_features.csv"
        save_patch_csv(df, csv_path)

        loaded = load_patch_csv(csv_path)
        assert len(loaded) == 3
        active = loaded[loaded["label_status"] == "active"]
        assert set(active["label"].values) == {1, 2}
        ghosts = loaded[loaded["label_status"] == "superseded"]
        assert set(ghosts["label"].values) == {3}


# ---------------------------------------------------------------------------
# Tests: QC plot ghost mask
# ---------------------------------------------------------------------------

class TestPlotGhostMask:
    """QCPlotWidget._get_ghost_mask separates active from ghost points."""

    def _make_widget(self):
        """Create minimal plot widget mock with the ghost mask method."""
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget
        # We can't instantiate the full widget without Qt, so test the method
        # logic directly.
        return QCPlotWidget.__dict__["_get_ghost_mask"]

    def test_no_label_status_column(self):
        get_ghost_mask = self._make_widget()
        widget = MagicMock()
        widget._df = pd.DataFrame({"label": [1, 2, 3]})
        result = get_ghost_mask(widget)
        assert result is None

    def test_all_active(self):
        get_ghost_mask = self._make_widget()
        widget = MagicMock()
        widget._df = _make_df([1, 2, 3])
        result = get_ghost_mask(widget)
        assert result is None  # no ghosts

    def test_mixed_active_and_ghost(self):
        get_ghost_mask = self._make_widget()
        widget = MagicMock()
        widget._df = _make_df_mixed([1, 2], [3], "deleted")
        result = get_ghost_mask(widget)
        assert result is not None
        assert result.sum() == 1  # 1 ghost
        assert (~result).sum() == 2  # 2 active

    def test_superseded_counted_as_ghost(self):
        get_ghost_mask = self._make_widget()
        widget = MagicMock()
        widget._df = _make_df_mixed([1], [2, 3], "superseded")
        result = get_ghost_mask(widget)
        assert result is not None
        assert result.sum() == 2


# ---------------------------------------------------------------------------
# Tests: Ghost scatter visibility (WP-D)
# ---------------------------------------------------------------------------


class TestGhostScatterVisibility:
    """WP-D: Ghost points must be clearly visible — correct size, color, pen."""

    def test_ghost_scatter_size_matches_active_default(self):
        """Ghost point size must be 7, matching the non-selected active default."""
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget
        assert QCPlotWidget._GHOST_COLOR is not None
        # The _add_ghost_scatter method uses size=7 (verified via source inspection).
        # We test via a minimal mock that captures the ScatterPlotItem kwargs.
        import types
        calls = []
        widget = types.SimpleNamespace()
        widget._GHOST_COLOR = QCPlotWidget._GHOST_COLOR
        widget._plot_widget = types.SimpleNamespace(
            addItem=lambda item: calls.append(item)
        )
        add_ghost = QCPlotWidget.__dict__["_add_ghost_scatter"]
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        mask = np.array([True, True])
        add_ghost(widget, x, y, mask)
        assert len(calls) == 1
        scatter = calls[0]
        # ScatterPlotItem stores opts — check size
        opts = scatter.opts
        assert opts["size"] == 10, f"Ghost size should be 10, got {opts['size']}"

    def test_ghost_scatter_color_alpha_above_threshold(self):
        """Ghost color alpha must be >= 100 for clear visibility."""
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget
        gc = QCPlotWidget._GHOST_COLOR
        assert len(gc) == 4
        assert gc[3] >= 100, f"Ghost alpha {gc[3]} is too low (< 100)"

    def test_ghost_scatter_pen_visible(self):
        """Ghost pen outline must have alpha >= 80 and width >= 0.8."""
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import QCPlotWidget
        import types
        calls = []
        widget = types.SimpleNamespace()
        widget._GHOST_COLOR = QCPlotWidget._GHOST_COLOR
        widget._plot_widget = types.SimpleNamespace(
            addItem=lambda item: calls.append(item)
        )
        add_ghost = QCPlotWidget.__dict__["_add_ghost_scatter"]
        x = np.array([1.0])
        y = np.array([2.0])
        mask = np.array([True])
        add_ghost(widget, x, y, mask)
        scatter = calls[0]
        pen = scatter.opts["pen"]
        # pen is a QPen — check color alpha and width
        assert pen.color().alpha() >= 80, f"Pen alpha {pen.color().alpha()} too low"
        assert pen.widthF() >= 0.8, f"Pen width {pen.widthF()} too thin"


# ---------------------------------------------------------------------------
# Tests: Commit triage marks ghost (not drops)
# ---------------------------------------------------------------------------

class TestCommitTriageGhost:
    """_commit_triage_single_patch marks garbage as deleted, not drops rows."""

    def test_garbage_marked_deleted(self, tmp_path):
        h = _CurationCallbackHarness()
        h._qc_features_df = _make_df([1, 2, 3, 4])
        h._qc_features_df.loc[
            h._qc_features_df["label"].isin([3, 4]), "qc_tier"
        ] = "garbage"
        h.working_dir = str(tmp_path)
        h.state_manager = MagicMock()
        h._selected_curation_patch = lambda: "patch001.tif"
        h._selected_patch_name = lambda: "patch001.tif"
        h._zero_labels_in_viewer = lambda ids: None
        h._get_active_labels_path = lambda: None  # skip mask save
        h._find_active_labels_layer = lambda: None

        # Bind and call the real method
        h._commit_triage_single_patch = _Ctrl._commit_triage_single_patch.__get__(h)

        # We need to mock save_patch_csv to avoid file I/O
        import napari_epicycle.fusion_reconstruction.qc_features as qcf
        original_save = qcf.save_patch_csv
        qcf.save_patch_csv = lambda df, path: None  # no-op
        try:
            h._commit_triage_single_patch()
        finally:
            qcf.save_patch_csv = original_save

        # Garbage rows should be marked deleted, not removed
        result = h._qc_features_df
        assert len(result) == 4  # all rows still present
        garbage_rows = result[result["label"].isin([3, 4])]
        assert (garbage_rows["label_status"] == "deleted").all()

        # Active rows unchanged
        active_rows = result[result["label"].isin([1, 2])]
        assert (active_rows["label_status"] == "active").all()


# ---------------------------------------------------------------------------
# Tests: Viewer color sync must exclude ghost rows
# ---------------------------------------------------------------------------

def _make_color_df(active_ids, ghost_ids, *, ghost_status="deleted"):
    """Create a DF with numeric features suitable for color sync tests."""
    rows = []
    for lid in active_ids:
        rows.append({
            "label": lid,
            "area": float(100 + lid),
            "volume": float(50 + lid),
            "centroid-0": 10.0,
            "centroid-1": 20.0,
            "centroid-2": 30.0,
            "qc_tier": "review",
            "qc_status": "flagged",
            "qc_score": float(lid),
            "feature_state": "fresh",
            "label_status": "active",
        })
    for lid in ghost_ids:
        rows.append({
            "label": lid,
            "area": float(100 + lid),
            "volume": float(50 + lid),
            "centroid-0": 10.0,
            "centroid-1": 20.0,
            "centroid-2": 30.0,
            "qc_tier": "bad",
            "qc_status": "flagged",
            "qc_score": float(lid),
            "feature_state": "fresh",
            "label_status": ghost_status,
        })
    return pd.DataFrame(rows)


class _ColorSyncHarness:
    """Minimal harness for the viewer colour sync (Step 3 plot pass P3: the
    ONE colour path, ``_apply_plot_colours_to_viewer``, which folded
    ``_update_viewer_qc_colors`` and ``_update_viewer_categorical_colors``
    in; ``_update_viewer_feature_colors`` is its thin wrapper).

    The plot is REAL (the Step 3 Workbench plot): the viewer takes the
    colours the plot draws, so the ghost rule below is the plot's own.
    """

    _MASK_MUTATING_ACTIONS = {"delete", "split", "merge", "smooth",
                              "interpolation", "discard_garbage", "carver",
                              "fill_holes", "flood_fill"}

    def __init__(self, df):
        import os
        import types

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from qtpy.QtWidgets import QApplication

        self._qapp = QApplication.instance() or QApplication([])
        from napari_epicycle.fusion_reconstruction._qc_plot_text import (
            STEP3_PLOT_TEXTS,
        )
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            STEP3_SCHEME,
            QCPlotWidget,
        )
        from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

        self.pw = QCPlotWidget(layout="workbench", texts=STEP3_PLOT_TEXTS,
                               palette=STEP3_PLOT_PALETTE,
                               tier_scheme=STEP3_SCHEME)
        self._qc_features_df = df
        self._isolated_label_ids = None
        self._isolated_df_indices = None
        cb = MagicMock()
        # "Colours in viewer" ticked so the one path does not return early
        cb.isChecked.return_value = True
        self.ui = types.SimpleNamespace(qc_plot_widget=self.pw,
                                        sync_viewer_colors_cb=cb)
        # Mock layer for single-patch mode
        self._mock_layer = MagicMock()

        # WP-2: ColormapAuthority required by color sync methods
        from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapAuthority
        self._colormap_authority = ColormapAuthority()
        self.pw.set_features(df)
        self.pw.plot_type_combo.setCurrentText("Scatter")

    def _find_all_patch_labels_layers(self):
        return [self._mock_layer]

    def _find_active_labels_layer(self):
        return self._mock_layer

    def colour_by(self, name):
        """Pick ``name`` in the plot's Colour by and push the colours the
        way the plot's own redraw does (one call of the one path)."""
        assert self.pw.set_colour_by(name), name
        self._last_qc_colormap = None       # every call writes afresh
        self._update_viewer_feature_colors(name, "Viridis", "Linear")


# Bind the real colour path onto the harness.
for _color_method in (
    "_apply_plot_colours_to_viewer",
    "_plot_hover_properties",
    "_update_viewer_feature_colors",
    "_suppress_sync_cascade",
    "_propagate_colormap_to_annotator_subviewers",
):
    setattr(_ColorSyncHarness, _color_method, getattr(_Ctrl, _color_method))


def _captured_color_dict(h, colour_by):
    from unittest.mock import patch as _patch
    captured = {}

    class _FakeCmap:
        def __init__(self, color_dict=None):
            captured["color_dict"] = dict(color_dict or {})

    with _patch("napari.utils.colormaps.DirectLabelColormap", _FakeCmap):
        h.colour_by(colour_by)
    assert "color_dict" in captured, (
        "the one colour path wrote no colormap -- the test would pass "
        "vacuously")
    return captured["color_dict"]


class TestViewerQCColorsActiveOnly:
    """Colour by "QC status" (was ``_update_viewer_qc_colors``) must only
    colour active labels."""

    def test_ghost_labels_not_in_colormap(self):
        """Ghost label IDs should not appear in the DirectLabelColormap."""
        h = _ColorSyncHarness(
            _make_color_df([1, 2, 3], [10, 20], ghost_status="deleted"))
        cd = _captured_color_dict(h, "QC status")
        # Active labels should be present
        for lid in [1, 2, 3]:
            assert lid in cd, f"Active label {lid} missing from colormap"
        # Ghost labels should NOT be present
        for lid in [10, 20]:
            assert lid not in cd, f"Ghost label {lid} should not be in colormap"

    def test_superseded_labels_not_in_colormap(self):
        h = _ColorSyncHarness(
            _make_color_df([1, 2], [5], ghost_status="superseded"))
        cd = _captured_color_dict(h, "QC status")
        assert 1 in cd and 2 in cd
        assert 5 not in cd


class TestViewerFeatureColorsActiveOnly:
    """Colour by a feature must exclude ghost rows."""

    def test_ghost_labels_not_in_colormap(self):
        h = _ColorSyncHarness(_make_color_df([1, 2, 3], [10, 20]))
        cd = _captured_color_dict(h, "area")
        for lid in [1, 2, 3]:
            assert lid in cd, f"Active label {lid} missing from colormap"
        for lid in [10, 20]:
            assert lid not in cd, f"Ghost label {lid} should not be in colormap"

    def test_ghost_values_dont_affect_normalization(self):
        """Ghost feature values must not skew normalization of active labels."""
        # Active labels: area 101, 102 (range 1)
        # Ghost label: area 99999 (would wildly skew normalization)
        df = _make_color_df([1, 2], [99])
        df.loc[df["label"] == 99, "area"] = 99999.0
        h = _ColorSyncHarness(df)
        h.pw._norm_combo.setCurrentText("Linear")
        cd = _captured_color_dict(h, "area")
        # If normalization is correct, the two active labels sit at the two
        # ends of the colour scale and get DIFFERENT colours; a ghost-skewed
        # range would squash both near 0.
        assert 1 in cd and 2 in cd
        color1 = cd[1]
        color2 = cd[2]
        assert not np.allclose(color1[:3], color2[:3], atol=0.05), (
            f"Active labels got nearly identical colors ({color1[:3]} vs {color2[:3]})"
            " — ghost values likely skewed normalization"
        )

    def test_props_array_excludes_ghost_labels(self):
        """Properties array for hover tooltips should not use ghost label IDs."""
        # Ghost label 999 would cause a huge props_array if labels.max() includes it
        df = _make_color_df([1, 2], [999])
        df.loc[df["label"] == 999, "area"] = 50000.0
        h = _ColorSyncHarness(df)
        _captured_color_dict(h, "area")
        props = h._mock_layer.properties
        assert isinstance(props, dict) and "area" in props, props
        arr = props["area"]
        # Sized by the max active label (2), not the ghost label (999)
        assert len(arr) == 3, (
            f"Properties array has {len(arr)} entries — ghost label 999 "
            "likely inflated the array size"
        )


class TestViewerCategoricalColorsActiveOnly:
    """Colour by "QC tier" / "Cluster" (was
    ``_update_viewer_categorical_colors``) must exclude ghost rows."""

    def test_ghost_labels_not_in_tier_colormap(self):
        h = _ColorSyncHarness(_make_color_df([1, 2, 3], [10, 20]))
        cd = _captured_color_dict(h, "QC tier")
        for lid in [1, 2, 3]:
            assert lid in cd, f"Active label {lid} missing from colormap"
        for lid in [10, 20]:
            assert lid not in cd, f"Ghost label {lid} should not be in colormap"

    def test_ghost_labels_not_in_cluster_colormap(self):
        df = _make_color_df([1, 2, 3], [10, 20])
        df["cluster_id"] = [0, 1, 0, 1, 0]  # add cluster column
        h = _ColorSyncHarness(df)
        cd = _captured_color_dict(h, "Cluster")
        for lid in [1, 2, 3]:
            assert lid in cd, f"Active label {lid} missing from colormap"
        for lid in [10, 20]:
            assert lid not in cd, f"Ghost label {lid} should not be in colormap"

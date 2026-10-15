"""Tests for P0: Population single-click identity hardening.

Verifies that ``get_selected_df_indices()`` returns the correct row index
for single-click, lasso, and category selection modes — without requiring
a full QApplication or pyqtgraph.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal stub that replicates the state logic of QCPlotWidget without Qt
# ---------------------------------------------------------------------------

class _PlotStateStub:
    """Replicates the selection state logic of QCPlotWidget."""

    def __init__(self):
        self._scatter_label_ids = []
        self._scatter_to_df_index = []
        self._clicked_df_index = None
        self._selected_indices = set()
        self._selected_category_df_indices = []

    def load_df(self, df, *, ghost_mask=None):
        """Load DataFrame, optionally filtering ghost rows.

        *ghost_mask* — boolean array; True = ghost (excluded from scatter).
        """
        if ghost_mask is not None:
            active_df = df[~ghost_mask]
        else:
            active_df = df
        self._scatter_label_ids = active_df["label"].values.tolist()
        self._scatter_to_df_index = list(active_df.index)
        self._clicked_df_index = None
        self._selected_indices.clear()
        self._selected_category_df_indices = []

    def simulate_click(self, nearest_idx):
        """Simulate single-click on scatter point at position nearest_idx."""
        self._clicked_df_index = nearest_idx

    def simulate_lasso(self, indices):
        """Simulate lasso selection of given positional indices."""
        self._clicked_df_index = None
        self._selected_indices = set(indices)

    def simulate_category(self, df_indices):
        """Simulate category selection of given DataFrame row indices."""
        self._selected_category_df_indices = list(df_indices)

    def clear_selection(self):
        self._selected_indices.clear()
        self._clicked_df_index = None
        self._selected_category_df_indices = []

    def get_selected_df_indices(self):
        """Mirrors QCPlotWidget.get_selected_df_indices()."""
        mapping = self._scatter_to_df_index
        if self._selected_indices:
            if mapping:
                return sorted(
                    mapping[i] for i in self._selected_indices
                    if i < len(mapping)
                )
            return sorted(self._selected_indices)
        if self._selected_category_df_indices:
            return self._selected_category_df_indices
        if self._clicked_df_index is not None:
            if mapping and self._clicked_df_index < len(mapping):
                return [mapping[self._clicked_df_index]]
            return [self._clicked_df_index]
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_population_df():
    """Two patches with duplicate label IDs (1, 2, 3 in each)."""
    rows = []
    for patch in ["patchA", "patchB"]:
        for lid in [1, 2, 3]:
            rows.append({
                "label": lid,
                "patch_name": patch,
                "area": 100 + lid,
                "centroid-0": lid * 10,
                "centroid-1": lid * 20,
                "centroid-2": lid * 30,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleClickDfIndex:
    """Verify single-click stores and returns correct df index."""

    def test_click_returns_index(self):
        stub = _PlotStateStub()
        df = _make_population_df()
        stub.load_df(df)
        stub.simulate_click(4)  # row 4 = patchB label 2
        assert stub.get_selected_df_indices() == [4]

    def test_click_returns_first_row(self):
        stub = _PlotStateStub()
        df = _make_population_df()
        stub.load_df(df)
        stub.simulate_click(0)  # row 0 = patchA label 1
        assert stub.get_selected_df_indices() == [0]

    def test_click_returns_last_row(self):
        stub = _PlotStateStub()
        df = _make_population_df()
        stub.load_df(df)
        stub.simulate_click(5)  # row 5 = patchB label 3
        assert stub.get_selected_df_indices() == [5]


class TestLassoOverridesSingleClick:
    """Verify lasso selection takes priority over single-click."""

    def test_lasso_overrides_click(self):
        stub = _PlotStateStub()
        df = _make_population_df()
        stub.load_df(df)
        stub.simulate_click(2)
        assert stub.get_selected_df_indices() == [2]
        # Lasso overrides
        stub.simulate_lasso([0, 1, 3])
        assert stub.get_selected_df_indices() == [0, 1, 3]

    def test_clear_after_lasso_returns_empty(self):
        stub = _PlotStateStub()
        df = _make_population_df()
        stub.load_df(df)
        stub.simulate_lasso([0, 1])
        stub.clear_selection()
        assert stub.get_selected_df_indices() == []


class TestCategoryOverridesSingleClick:
    """Verify category selection takes priority over single-click."""

    def test_category_overrides_click(self):
        stub = _PlotStateStub()
        df = _make_population_df()
        stub.load_df(df)
        stub.simulate_click(3)
        assert stub.get_selected_df_indices() == [3]
        # Category overrides
        stub.simulate_category([0, 2, 4])
        assert stub.get_selected_df_indices() == [0, 2, 4]


class TestNewDataClearsClick:
    """Verify loading new data clears clicked index."""

    def test_load_clears_click(self):
        stub = _PlotStateStub()
        df = _make_population_df()
        stub.load_df(df)
        stub.simulate_click(2)
        assert stub.get_selected_df_indices() == [2]
        # Load new data
        stub.load_df(df)
        assert stub.get_selected_df_indices() == []

    def test_empty_without_any_selection(self):
        stub = _PlotStateStub()
        df = _make_population_df()
        stub.load_df(df)
        assert stub.get_selected_df_indices() == []


class TestPopulationIdentityResolution:
    """Verify that the clicked df index maps to the correct (patch, label) pair."""

    def test_duplicate_label_ids_resolve_correctly(self):
        """Label 2 exists in both patches — click on each should resolve differently."""
        df = _make_population_df()
        stub = _PlotStateStub()
        stub.load_df(df)

        # Click on row 1 (patchA, label 2)
        stub.simulate_click(1)
        idx = stub.get_selected_df_indices()[0]
        row = df.iloc[idx]
        assert row["label"] == 2
        assert row["patch_name"] == "patchA"

        # Click on row 4 (patchB, label 2)
        stub.simulate_click(4)
        idx = stub.get_selected_df_indices()[0]
        row = df.iloc[idx]
        assert row["label"] == 2
        assert row["patch_name"] == "patchB"

    def test_all_rows_addressable(self):
        """Every row in the population DataFrame should be individually addressable."""
        df = _make_population_df()
        stub = _PlotStateStub()
        stub.load_df(df)

        for i in range(len(df)):
            stub.simulate_click(i)
            indices = stub.get_selected_df_indices()
            assert indices == [i], f"Row {i}: expected [{i}], got {indices}"
            row = df.iloc[indices[0]]
            assert row["label"] == df.iloc[i]["label"]
            assert row["patch_name"] == df.iloc[i]["patch_name"]


# ---------------------------------------------------------------------------
# Ghost-row tests — verify lasso/click on filtered scatter maps back to
# the correct full-DataFrame row indices
# ---------------------------------------------------------------------------

def _make_df_with_ghosts():
    """6-row DataFrame: rows 1 and 3 are deleted (ghost).

    Full df:
        0: label=10, active
        1: label=20, deleted    ← ghost
        2: label=30, active
        3: label=40, deleted    ← ghost
        4: label=50, active
        5: label=60, active

    Scatter (active only) positions 0..3 map to df rows [0, 2, 4, 5].
    """
    df = pd.DataFrame({
        "label": [10, 20, 30, 40, 50, 60],
        "label_status": ["active", "deleted", "active", "deleted", "active", "active"],
        "area": [100, 200, 300, 400, 500, 600],
    })
    ghost_mask = np.array([False, True, False, True, False, False])
    return df, ghost_mask


class TestGhostRowMapping:
    """Verify scatter-position → df-index mapping when ghost rows exist."""

    def test_click_maps_through_ghosts(self):
        """Single click at scatter position 2 → df row 4 (label=50)."""
        df, ghost_mask = _make_df_with_ghosts()
        stub = _PlotStateStub()
        stub.load_df(df, ghost_mask=ghost_mask)

        # Scatter pos 0 → df row 0 (label 10)
        stub.simulate_click(0)
        assert stub.get_selected_df_indices() == [0]
        assert df.iloc[0]["label"] == 10

        # Scatter pos 1 → df row 2 (label 30, skipping ghost at row 1)
        stub.simulate_click(1)
        assert stub.get_selected_df_indices() == [2]
        assert df.iloc[2]["label"] == 30

        # Scatter pos 2 → df row 4 (label 50, skipping ghosts at rows 1,3)
        stub.simulate_click(2)
        assert stub.get_selected_df_indices() == [4]
        assert df.iloc[4]["label"] == 50

        # Scatter pos 3 → df row 5 (label 60)
        stub.simulate_click(3)
        assert stub.get_selected_df_indices() == [5]
        assert df.iloc[5]["label"] == 60

    def test_lasso_maps_through_ghosts(self):
        """Lasso selecting scatter positions [0, 2] → df rows [0, 4]."""
        df, ghost_mask = _make_df_with_ghosts()
        stub = _PlotStateStub()
        stub.load_df(df, ghost_mask=ghost_mask)

        stub.simulate_lasso([0, 2])
        result = stub.get_selected_df_indices()
        assert result == [0, 4]
        assert df.iloc[0]["label"] == 10
        assert df.iloc[4]["label"] == 50

    def test_lasso_all_active_maps_correctly(self):
        """Lasso all 4 active scatter points → df rows [0, 2, 4, 5]."""
        df, ghost_mask = _make_df_with_ghosts()
        stub = _PlotStateStub()
        stub.load_df(df, ghost_mask=ghost_mask)

        stub.simulate_lasso([0, 1, 2, 3])
        result = stub.get_selected_df_indices()
        assert result == [0, 2, 4, 5]

    def test_category_indices_unchanged_with_ghosts(self):
        """Category selection uses real df indices — not affected by mapping."""
        df, ghost_mask = _make_df_with_ghosts()
        stub = _PlotStateStub()
        stub.load_df(df, ghost_mask=ghost_mask)

        # Category selection gives real df indices directly
        stub.simulate_category([0, 4])
        assert stub.get_selected_df_indices() == [0, 4]

    def test_no_ghosts_identity_mapping(self):
        """Without ghosts, scatter pos == df index (identity mapping)."""
        df, _ = _make_df_with_ghosts()
        stub = _PlotStateStub()
        stub.load_df(df)  # no ghost_mask → all rows active

        stub.simulate_click(3)
        assert stub.get_selected_df_indices() == [3]

        stub.simulate_lasso([1, 4])
        assert stub.get_selected_df_indices() == [1, 4]

    def test_click_then_lasso_with_ghosts(self):
        """Click sets index, lasso overrides — both map through ghosts."""
        df, ghost_mask = _make_df_with_ghosts()
        stub = _PlotStateStub()
        stub.load_df(df, ghost_mask=ghost_mask)

        stub.simulate_click(1)  # scatter pos 1 → df row 2
        assert stub.get_selected_df_indices() == [2]

        stub.simulate_lasso([0, 3])  # scatter [0,3] → df [0,5]
        assert stub.get_selected_df_indices() == [0, 5]

    def test_population_with_ghosts(self):
        """Ghost rows in population mode still map correctly."""
        rows = []
        for patch in ["patchA", "patchB"]:
            for lid, status in [(1, "active"), (2, "deleted"), (3, "active")]:
                rows.append({
                    "label": lid, "patch_name": patch,
                    "label_status": status, "area": 100,
                })
        df = pd.DataFrame(rows)
        ghost_mask = (df["label_status"] == "deleted").values

        stub = _PlotStateStub()
        stub.load_df(df, ghost_mask=ghost_mask)
        # Active rows: 0(A,1), 2(A,3), 3(B,1), 5(B,3) → scatter pos 0,1,2,3
        assert stub._scatter_label_ids == [1, 3, 1, 3]

        # Click scatter pos 2 → df row 3 (patchB, label 1)
        stub.simulate_click(2)
        idx = stub.get_selected_df_indices()[0]
        assert df.iloc[idx]["patch_name"] == "patchB"
        assert df.iloc[idx]["label"] == 1

        # Lasso scatter [1,3] → df [2,5] (patchA label 3, patchB label 3)
        stub.simulate_lasso([1, 3])
        result = stub.get_selected_df_indices()
        assert result == [2, 5]
        assert df.iloc[2]["patch_name"] == "patchA"
        assert df.iloc[5]["patch_name"] == "patchB"


# ---------------------------------------------------------------------------
# Selection preservation tests — verify update_data preserves lasso selection
# ---------------------------------------------------------------------------

class _UpdateDataStub(_PlotStateStub):
    """Extended stub that simulates update_data selection preservation."""

    def update_data(self, df, *, ghost_mask=None):
        """Simulate update_data: preserve label IDs across re-render."""
        # Convert position-based selection to label IDs
        preserved_label_ids = None
        if self._selected_indices and self._scatter_label_ids:
            preserved_label_ids = set()
            for i in self._selected_indices:
                if 0 <= i < len(self._scatter_label_ids):
                    preserved_label_ids.add(int(self._scatter_label_ids[i]))

        # Re-load data (rebuilds scatter_label_ids)
        self._selected_indices.clear()
        if ghost_mask is not None:
            active_df = df[~ghost_mask]
        else:
            active_df = df
        self._scatter_label_ids = active_df["label"].values.tolist()
        self._scatter_to_df_index = list(active_df.index)

        # Re-map preserved label IDs to new scatter positions
        if preserved_label_ids and self._scatter_label_ids:
            new_indices = set()
            for i, lid in enumerate(self._scatter_label_ids):
                if int(lid) in preserved_label_ids:
                    new_indices.add(i)
            self._selected_indices = new_indices


class TestSelectionPreservation:
    """Verify lasso selection survives DataFrame updates (label deletion)."""

    def test_selection_preserved_after_deletion(self):
        """Lasso selects labels 30,50. Delete label 20 → selection preserved."""
        df, ghost_mask = _make_df_with_ghosts()
        stub = _UpdateDataStub()
        stub.load_df(df, ghost_mask=ghost_mask)
        # Active: [10, 30, 50, 60] at scatter pos [0, 1, 2, 3]

        # Lasso select scatter pos 1,2 → labels 30, 50
        stub.simulate_lasso([1, 2])
        assert stub.get_selected_df_indices() == [2, 4]  # df rows for 30, 50

        # Now delete label 10 (add it to ghosts)
        new_ghost = np.array([True, True, False, True, False, False])
        stub.update_data(df, ghost_mask=new_ghost)
        # Active: [30, 50, 60] at scatter pos [0, 1, 2]

        # Labels 30 and 50 should still be selected
        assert 30 in [stub._scatter_label_ids[i] for i in stub._selected_indices]
        assert 50 in [stub._scatter_label_ids[i] for i in stub._selected_indices]
        assert stub.get_selected_df_indices() == [2, 4]  # same df rows

    def test_deleted_label_removed_from_selection(self):
        """If a selected label is deleted, it drops out of the selection."""
        df, ghost_mask = _make_df_with_ghosts()
        stub = _UpdateDataStub()
        stub.load_df(df, ghost_mask=ghost_mask)
        # Active: [10, 30, 50, 60] at scatter pos [0, 1, 2, 3]

        # Lasso select all: labels 10, 30, 50, 60
        stub.simulate_lasso([0, 1, 2, 3])

        # Delete label 30 (make it a ghost)
        new_ghost = np.array([False, True, True, True, False, False])
        stub.update_data(df, ghost_mask=new_ghost)
        # Active: [10, 50, 60] at scatter pos [0, 1, 2]

        selected_labels = {stub._scatter_label_ids[i] for i in stub._selected_indices}
        assert 30 not in selected_labels
        assert selected_labels == {10, 50, 60}

    def test_no_selection_no_crash(self):
        """update_data with no selection doesn't crash."""
        df, ghost_mask = _make_df_with_ghosts()
        stub = _UpdateDataStub()
        stub.load_df(df, ghost_mask=ghost_mask)

        new_ghost = np.array([True, True, False, True, False, False])
        stub.update_data(df, ghost_mask=new_ghost)
        assert stub._selected_indices == set()
        assert stub.get_selected_df_indices() == []

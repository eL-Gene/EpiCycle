"""WP4 + WP5: Real-path tests for plot identity and click safety.

Exercises the actual ``QCPlotWidget.get_selected_df_indices()`` method
and verifies that ``_clicked_df_index`` is properly cleared by
``highlight_labels()`` and ``_toggle_category_selection()``.

NOTE: These tests require pyqtgraph importable but do NOT need a
QApplication — they instantiate the widget class methods directly via
bound-method injection on a minimal harness that has the right attributes.
"""

import pandas as pd
import numpy as np
import pytest


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
# Minimal harness that mirrors QCPlotWidget selection state
# ---------------------------------------------------------------------------

class _PlotWidgetHarness:
    """Mirrors QCPlotWidget selection attributes for method binding."""

    def __init__(self):
        self._scatter_label_ids = []
        self._scatter_to_df_index = []
        self._clicked_df_index = None
        self._selected_indices = set()
        self._selected_category_df_indices = []
        self._selected_categories = set()
        self._df = None
        self._lasso_points = []

    def load(self, df, *, ghost_mask=None):
        self._df = df
        if ghost_mask is not None:
            active_df = df[~ghost_mask]
        else:
            active_df = df
        self._scatter_label_ids = active_df["label"].values.tolist()
        self._scatter_to_df_index = list(active_df.index)
        self._clicked_df_index = None
        self._selected_indices.clear()
        self._selected_category_df_indices = []


def _bind_real_get_selected(harness):
    """Bind the real get_selected_df_indices from QCPlotWidget."""
    import types
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )
    real_method = QCPlotWidget.get_selected_df_indices
    harness.get_selected_df_indices = types.MethodType(real_method, harness)


# ---------------------------------------------------------------------------
# Tests: get_selected_df_indices priority
# ---------------------------------------------------------------------------

class TestGetSelectedDfIndicesReal:
    """Test the real get_selected_df_indices method's priority logic."""

    def test_clicked_index_returned(self):
        h = _PlotWidgetHarness()
        _bind_real_get_selected(h)
        h.load(_make_population_df())
        h._clicked_df_index = 4
        assert h.get_selected_df_indices() == [4]

    def test_lasso_overrides_click(self):
        h = _PlotWidgetHarness()
        _bind_real_get_selected(h)
        h.load(_make_population_df())
        h._clicked_df_index = 2
        h._selected_indices = {0, 3, 5}
        # Lasso takes priority
        assert h.get_selected_df_indices() == [0, 3, 5]

    def test_category_overrides_click(self):
        h = _PlotWidgetHarness()
        _bind_real_get_selected(h)
        h.load(_make_population_df())
        h._clicked_df_index = 1
        h._selected_category_df_indices = [0, 2, 4]
        # Category takes priority
        assert h.get_selected_df_indices() == [0, 2, 4]

    def test_empty_when_no_selection(self):
        h = _PlotWidgetHarness()
        _bind_real_get_selected(h)
        h.load(_make_population_df())
        assert h.get_selected_df_indices() == []


# ---------------------------------------------------------------------------
# Tests: _clicked_df_index cleared by programmatic selections
# ---------------------------------------------------------------------------

class TestClickedIndexClearedByProgrammatic:
    """Verify highlight_labels and category toggle clear _clicked_df_index."""

    def test_highlight_labels_clears_click(self):
        """highlight_labels() must clear _clicked_df_index (behavioral)."""
        from unittest.mock import MagicMock
        import types
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            QCPlotWidget,
        )

        harness = MagicMock()
        harness._clicked_df_index = 5
        harness._scatter_label_ids = [1, 2, 3]
        harness._selected_indices = set()
        harness._selected_categories = set()
        harness._selected_category_df_indices = []

        bound = types.MethodType(QCPlotWidget.highlight_labels, harness)
        bound([1, 2])

        assert harness._clicked_df_index is None, (
            "highlight_labels must clear _clicked_df_index"
        )

    def test_toggle_category_clears_click(self):
        """_toggle_category_selection() must clear _clicked_df_index (behavioral)."""
        from unittest.mock import MagicMock
        import types
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            QCPlotWidget,
        )

        harness = MagicMock()
        harness._clicked_df_index = 7
        harness._df = _make_population_df()
        harness._selected_categories = set()
        harness._selected_indices = set()
        harness._lasso_points = []

        bound = types.MethodType(
            QCPlotWidget._toggle_category_selection, harness
        )
        bound("patch_name", "patchA")

        assert harness._clicked_df_index is None, (
            "_toggle_category_selection must clear _clicked_df_index"
        )


# ---------------------------------------------------------------------------
# Tests: R2-WP4 — Population ambiguous single-click dropped (behavioral)
# ---------------------------------------------------------------------------

# TestPopulationAmbiguousClickGuard class removed 2026-05-03
# (drop-population-mode Commit 3/6).  Population mode is gone, so the
# ambiguous-click guard (which only fired in population scope) no
# longer exists.  test_single_patch_click_proceeds was the lone non-
# population test in this class — its assertion (single-click proceeds
# normally) is unchanged behavior post-removal and is implicitly
# covered by the broader plot identity tests.


# ---------------------------------------------------------------------------
# Tests: Ghost-row mapping through real get_selected_df_indices
# ---------------------------------------------------------------------------

class TestGhostRowMappingReal:
    """Verify scatter→df index mapping with real method when ghosts exist."""

    def test_click_maps_through_ghosts(self):
        """Click at scatter pos 1 → df row 2 (skipping ghost at row 1)."""
        h = _PlotWidgetHarness()
        _bind_real_get_selected(h)
        df = pd.DataFrame({
            "label": [10, 20, 30, 40],
            "label_status": ["active", "deleted", "active", "active"],
        })
        ghost = np.array([False, True, False, False])
        h.load(df, ghost_mask=ghost)

        h._clicked_df_index = 1  # scatter pos 1 → df row 2
        assert h.get_selected_df_indices() == [2]

    def test_lasso_maps_through_ghosts(self):
        """Lasso scatter [0,2] → df rows [0,3] (skipping ghost at row 1)."""
        h = _PlotWidgetHarness()
        _bind_real_get_selected(h)
        df = pd.DataFrame({
            "label": [10, 20, 30, 40],
            "label_status": ["active", "deleted", "active", "active"],
        })
        ghost = np.array([False, True, False, False])
        h.load(df, ghost_mask=ghost)

        h._selected_indices = {0, 2}
        assert h.get_selected_df_indices() == [0, 3]

    def test_no_ghosts_identity(self):
        """Without ghosts, mapping is identity."""
        h = _PlotWidgetHarness()
        _bind_real_get_selected(h)
        h.load(_make_population_df())

        h._clicked_df_index = 3
        assert h.get_selected_df_indices() == [3]

        h._selected_indices = {1, 4}
        assert h.get_selected_df_indices() == [1, 4]

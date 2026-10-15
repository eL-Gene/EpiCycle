"""Tier-line drag handler split — Phase A #1 (perf push 2026-05-09).

The continuous handler ``_on_tier_thresholds_changed`` previously called
``_update_flagged_labels_list()`` on every drag pixel.  With the 3D
annotator + Label Navigator open, that triggered:

  1. clear + iterrows-rebuild side panel QListWidget
  2. clear ``_flagged_centroids`` dict
  3. ``_push_flagged_to_annotator()`` — full QTableWidget rebuild on
     the floating Label Navigator (with Color/Tier delegates)
  4. ``_push_to_triage_histogram(df)`` — left histogram bar re-stack

At ~5k+ active labels, items 1+3 cost 50–200 ms PER DRAG PIXEL.

The fix splits the work by **visual feedback need**:
  - Continuous handler keeps the work the user watches LIVE during
    drag (right plot recolor + left histogram bar restack + chip
    counts).  Item 4 above stays on the continuous path.
  - Release handler picks up the heavy rebuilds (1, 3) once per drag.

Items 1 + 3 = 50–200 ms heavy rebuilds, deferred to release.
Item 4 = 10–30 ms panel redraw, kept live.

These tests pin the contract.  Lightweight fakes — no Qt.
"""

from unittest.mock import MagicMock, patch

import pandas as pd


def _make_controller():
    """Return a controller built via ``object.__new__`` with the fields
    the tier-drag handlers touch.  No Qt event loop, no real workspace.
    """
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = object.__new__(FusionReconstructionController)

    ctrl._qc_features_df = pd.DataFrame({
        "label": [1, 2, 3, 4, 5],
        "qc_score": [0.1, 0.3, 0.5, 0.7, 0.9],
        "qc_tier": ["good"] * 5,
        "centroid-0": [0, 1, 2, 3, 4],
        "centroid-1": [0, 1, 2, 3, 4],
        "centroid-2": [0, 1, 2, 3, 4],
        "area": [100] * 5,
    })
    ctrl._committing_triage = False
    ctrl._patch_tier_thresholds = {}
    ctrl._qc_view_patch_name = "patch_a"

    pw = MagicMock()
    pw._redraw = MagicMock()
    pw._df = None
    ctrl.ui = MagicMock()
    ctrl.ui.qc_plot_widget = pw

    # Stub the heavy methods so we can assert call counts.
    ctrl._update_qc_summary = MagicMock()
    ctrl._update_flagged_labels_list = MagicMock()
    ctrl._push_to_triage_histogram = MagicMock()

    return ctrl


# ---------------------------------------------------------------------------
# Continuous handler — what stays light during drag
# ---------------------------------------------------------------------------


class TestContinuousHandlerStaysLight:
    """``_on_tier_thresholds_changed`` (fires every drag pixel) MUST
    NOT trigger the heavy side-panel/navigator rebuild.  That work is
    deferred to ``_on_tier_drag_finished``."""

    def test_does_not_call_update_flagged_labels_list(self):
        ctrl = _make_controller()
        ctrl._on_tier_thresholds_changed((0.2, 0.4, 0.6))
        ctrl._update_flagged_labels_list.assert_not_called()

    def test_does_not_call_update_flagged_across_many_drag_pixels(self):
        """Worst-case: simulating 100 drag pixels.  Heavy rebuild stays
        deferred even across the entire drag."""
        ctrl = _make_controller()
        for i in range(100):
            t1 = 0.2 + 0.001 * i
            ctrl._on_tier_thresholds_changed((t1, t1 + 0.2, t1 + 0.4))
        ctrl._update_flagged_labels_list.assert_not_called()


class TestContinuousHandlerKeepsLiveFeedback:
    """``_on_tier_thresholds_changed`` MUST keep the user's live
    visual feedback running during drag — right plot recolor, LEFT
    histogram bar re-stack, qc_summary chip counts."""

    def test_lightweight_redraw_runs_every_drag_pixel(self):
        ctrl = _make_controller()
        ctrl._on_tier_thresholds_changed((0.2, 0.4, 0.6))
        ctrl.ui.qc_plot_widget._redraw.assert_called_once_with(_lightweight=True)

    def test_qc_summary_updated_every_drag_pixel(self):
        ctrl = _make_controller()
        ctrl._on_tier_thresholds_changed((0.2, 0.4, 0.6))
        ctrl._update_qc_summary.assert_called_once()

    def test_triage_histogram_pushed_every_drag_pixel(self):
        """Left histogram bar restacking is the user's primary visual
        feedback for 'where will tiers land if I let go here'.  It
        MUST fire on the continuous path."""
        ctrl = _make_controller()
        ctrl._on_tier_thresholds_changed((0.2, 0.4, 0.6))
        ctrl._push_to_triage_histogram.assert_called_once_with(
            ctrl._qc_features_df
        )

    def test_assign_qc_tiers_runs_every_drag_pixel(self):
        ctrl = _make_controller()
        ctrl._on_tier_thresholds_changed((0.05, 0.4, 0.95))
        # qc_tier column reassigned in-place.
        assert "qc_tier" in ctrl._qc_features_df.columns


# ---------------------------------------------------------------------------
# Release handler — what fires on drag-finished
# ---------------------------------------------------------------------------


class TestReleaseHandlerPicksUpDeferredWork:
    """``_on_tier_drag_finished`` (fires once on release) MUST trigger
    the heavy side-panel + navigator rebuild that the continuous
    handler skipped."""

    def test_release_handler_calls_update_flagged_labels_list(self):
        ctrl = _make_controller()
        ctrl._on_tier_drag_finished((0.2, 0.4, 0.6))
        ctrl._update_flagged_labels_list.assert_called_once()

    def test_release_handler_still_does_full_redraw(self):
        """Full redraw with autoRange + viewer color sync — existing
        behavior from before the deferral split.  Must survive."""
        ctrl = _make_controller()
        ctrl._on_tier_drag_finished((0.2, 0.4, 0.6))
        ctrl.ui.qc_plot_widget._redraw.assert_called_once_with()


# ---------------------------------------------------------------------------
# Drag lifecycle property
# ---------------------------------------------------------------------------


class TestDragLifecycleProperty:
    """Across a full drag (many continuous fires + one release):
    - heavy rebuild fires EXACTLY once (at release)
    - panel redraw + lightweight redraw fire EVERY pixel + once on release
    """

    def test_full_lifecycle_one_heavy_rebuild(self):
        ctrl = _make_controller()
        for i in range(50):
            t1 = 0.2 + 0.005 * i
            ctrl._on_tier_thresholds_changed((t1, t1 + 0.2, t1 + 0.4))
        ctrl._on_tier_drag_finished((0.45, 0.65, 0.85))
        assert ctrl._update_flagged_labels_list.call_count == 1, (
            "Heavy rebuild should fire exactly once at release; "
            f"fired {ctrl._update_flagged_labels_list.call_count} times"
        )

    def test_full_lifecycle_panel_redraws_on_every_drag_pixel(self):
        """Left histogram bar restacking fires per drag pixel for
        live feedback."""
        ctrl = _make_controller()
        for i in range(50):
            t1 = 0.2 + 0.005 * i
            ctrl._on_tier_thresholds_changed((t1, t1 + 0.2, t1 + 0.4))
        # 50 drag pixels = 50 panel pushes (release doesn't add one
        # because release calls _update_flagged_labels_list which is
        # mocked here, so the inner _push_to_triage_histogram doesn't
        # fire from there).
        assert ctrl._push_to_triage_histogram.call_count == 50, (
            "Panel redraw should fire on every drag pixel; "
            f"fired {ctrl._push_to_triage_histogram.call_count} times "
            f"during 50 continuous handler invocations"
        )


# ---------------------------------------------------------------------------
# Early-return guards preserved
# ---------------------------------------------------------------------------


class TestEarlyReturnsStillSkipBoth:
    """Existing early-return guards (committing-triage, df-is-None,
    df-empty) must skip BOTH the continuous and the release work."""

    def test_continuous_skips_when_committing(self):
        ctrl = _make_controller()
        ctrl._committing_triage = True
        ctrl._on_tier_thresholds_changed((0.2, 0.4, 0.6))
        ctrl._update_qc_summary.assert_not_called()
        ctrl._update_flagged_labels_list.assert_not_called()
        ctrl._push_to_triage_histogram.assert_not_called()

    def test_continuous_skips_when_df_is_none(self):
        ctrl = _make_controller()
        ctrl._qc_features_df = None
        ctrl._on_tier_thresholds_changed((0.2, 0.4, 0.6))
        ctrl._update_qc_summary.assert_not_called()
        ctrl._update_flagged_labels_list.assert_not_called()
        ctrl._push_to_triage_histogram.assert_not_called()

    def test_continuous_skips_when_df_empty(self):
        ctrl = _make_controller()
        ctrl._qc_features_df = pd.DataFrame()
        ctrl._on_tier_thresholds_changed((0.2, 0.4, 0.6))
        ctrl._update_qc_summary.assert_not_called()
        ctrl._update_flagged_labels_list.assert_not_called()
        ctrl._push_to_triage_histogram.assert_not_called()

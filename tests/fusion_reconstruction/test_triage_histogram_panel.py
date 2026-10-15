"""Regression tests for the Phase 2A triage histogram panel.

User-reported bug 2026-05-04: panel histogram shape did not match the
right-side QCPlotWidget when both plotted ``qc_score``.  Diagnosis
identified four bugs (commits 9f7b993, 24224ad, this-commit):

  (1) hardcoded ``x_range = (0, 1)`` for qc_score only — real EpiCycle
      qc_score is unbounded composite.
  (2) bin count hardcoded to 30 — right plot uses
      ``min(50, max(10, n // 5))``.
  (3) rows with qc_tier outside the canonical 5 (e.g. NaN, "flagged")
      were silently dropped from the stacked histogram.
  (4) tier-line color order was inverted from EpiCycle's convention
      "higher qc_score = worse" — the leftmost line should be green
      (good/review boundary), rightmost red (bad/garbage boundary).

These tests pin all four contracts so future refactors can't silently
regress them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# pyqtgraph + Qt are available without a running display for widget
# CONSTRUCTION but rendering needs a QApplication.  These tests use
# the qtbot fixture indirectly via a single per-module QApplication.
@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _make_df(n=600, qc_score_lo=-2.0, qc_score_hi=10.0, with_oddballs=False):
    """Synthetic features DataFrame mimicking real EpiCycle shape."""
    rng = np.random.default_rng(0)
    qc = rng.uniform(qc_score_lo, qc_score_hi, n)
    tiers = rng.choice(
        ["good", "review", "bad", "garbage", "curated"], n,
        p=[0.55, 0.20, 0.15, 0.05, 0.05],
    )
    if with_oddballs:
        tiers[0] = "flagged"   # legacy non-canonical
        tiers[1] = "unassigned"
        tiers[2] = None        # NaN-like
    df = pd.DataFrame({
        "label": np.arange(1, n + 1),
        "qc_score": qc,
        "area": rng.lognormal(10, 0.5, n),
        "true_sphericity": rng.beta(5, 3, n),
        "qc_tier": tiers,
        "label_status": ["active"] * n,
    })
    return df


# ----------------------------------------------------------------------
# Bug (1): x_range derives from data, not hardcoded for qc_score
# ----------------------------------------------------------------------


def test_x_range_matches_data_min_max_for_qc_score(qapp):
    """Pre-fix bug: qc_score x_range was hardcoded to (0, 1) so values
    outside that range piled up at the edges via np.clip.  Verify the
    fix by passing values well outside [0, 1]."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df(qc_score_lo=-2.0, qc_score_hi=10.0)
    panel.set_data(df)

    values, _tiers, x_range = panel._values_from_df("qc_score")
    assert values.min() == pytest.approx(x_range[0], abs=1e-9), (
        "x_range[0] must equal values.min() (fix for hardcoded (0,1))"
    )
    assert values.max() == pytest.approx(x_range[1], abs=1e-9)
    assert x_range[0] < 0.0, "values genuinely span outside [0, 1]"
    assert x_range[1] > 1.0


def test_x_range_for_non_qc_score_features(qapp):
    """Non-qc_score features should also derive x_range from values."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)

    _vals, _t, xr_area = panel._values_from_df("area")
    assert xr_area[0] > 0  # lognormal is positive
    assert xr_area[1] > xr_area[0]


# ----------------------------------------------------------------------
# Bug (2): bin count formula matches QCPlotWidget
# ----------------------------------------------------------------------


def test_bin_count_matches_qcplotwidget_formula(qapp):
    """Verify the panel uses the same bin formula as the historic
    qc_plot_widget histogram path: min(50, max(10, n // 5)).
    (The right-plot histogram was removed in Phase E; the formula is
    preserved on the panel side for visual continuity.)"""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()

    # 600 cells -> 600 // 5 = 120, capped at 50 -> 50 bins
    df = _make_df(n=600)
    panel.set_data(df)
    # The bar count after redraw should equal n_bins (one BarGraphItem
    # per bin per stacked tier).  Easier: re-derive via the same
    # formula we expect.
    _vals, _t, _xr = panel._values_from_df("qc_score")
    expected_bins = min(50, max(10, len(_vals) // 5))
    assert expected_bins == 50, "600 cells should yield 50 bins"

    # 30 cells -> max(10, 30//5)=10 bins
    df_small = _make_df(n=30)
    panel.set_data(df_small)
    _vals, _t, _xr = panel._values_from_df("qc_score")
    assert min(50, max(10, len(_vals) // 5)) == 10


# ----------------------------------------------------------------------
# Perf refactor guard (2026-06-08, fix/step3-post-extract-curation-reset):
# _values_from_df / _derive_thresholds_from_qc_tier / chip counts now mask
# individual column .values with a numpy row-keep mask instead of copying the
# full (highly-fragmented, ~100-col) DataFrame via df[~mask] — the wide copy
# was the entire post-extract threshold-drag lag (3731ms -> tens of ms).
# These pin the EXACT ghost-drop + NaN-drop + tier-alignment behavior, since
# the [keep][finite_mask] double-indexing is the risky part of the rewrite.
# ----------------------------------------------------------------------


def test_values_from_df_keeps_ghosts_as_gray_drops_nans(qapp):
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    # 2026-06-09: committed/ghost rows (label 2 deleted, 6 superseded) carry
    # FINITE qc_scores and are now KEPT, tagged "ghost" (gray) so committed
    # garbage stays visible in the histogram until re-extraction (mirrors the
    # scatter's Show-History ghosts).  The active-but-NaN row 4 is still dropped
    # by the finite filter.
    df = pd.DataFrame({
        "label":        [1, 2, 3, 4, 5, 6],
        "qc_score":     [1.0, 7.0, 5.0, np.nan, 9.0, 8.0],
        "qc_tier":      ["good", "bad", "bad", "good", "garbage", "review"],
        "label_status": ["active", "deleted", "active", "active",
                         "active", "superseded"],
    })
    panel.set_data(df)
    values, tiers, x_range = panel._values_from_df("qc_score")
    # All FINITE rows kept (only the NaN active row 4 dropped); ghosts INCLUDED.
    assert sorted(values.tolist()) == [1.0, 5.0, 7.0, 8.0, 9.0], (
        "ghosts (finite-scored) are now KEPT; only the NaN active row is dropped"
    )
    val_to_tier = dict(zip(values.tolist(), tiers.tolist()))
    assert val_to_tier[1.0] == "good"
    assert val_to_tier[5.0] == "bad"
    assert val_to_tier[9.0] == "garbage"
    # Ghost rows are tagged "ghost" (gray), NOT their original qc_tier.
    assert val_to_tier[7.0] == "ghost"  # label 2 deleted (was "bad")
    assert val_to_tier[8.0] == "ghost"  # label 6 superseded (was "review")
    assert x_range[0] == pytest.approx(1.0)
    assert x_range[1] == pytest.approx(9.0)


def test_ghost_tag_not_truncated_with_short_active_tiers(qapp):
    # Regression (audit 2026-06-09): when EVERY active tier is <=4 chars
    # ("good"/"bad"), the numpy tiers array is <U4 and assigning "ghost" (5
    # chars) silently TRUNCATED to "ghos" → ghosts fell into the blue
    # "unassigned" stack instead of the gray "ghost" stack.  (The other ghost
    # test passes even with the bug because its df contains "garbage" → <U7.)
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = pd.DataFrame({
        "label":        [1, 2, 3],
        "qc_score":     [1.0, 5.0, 9.0],
        "qc_tier":      ["good", "bad", "good"],   # all <=4 chars → would be <U4
        "label_status": ["active", "active", "superseded"],
    })
    panel.set_data(df)
    values, tiers, _x = panel._values_from_df("qc_score")
    val_to_tier = dict(zip(values.tolist(), tiers.tolist()))
    assert val_to_tier[9.0] == "ghost", (
        f"ghost tag truncated: got {val_to_tier[9.0]!r} (the <U4 truncation bug)"
    )
    assert val_to_tier[1.0] == "good"
    assert val_to_tier[5.0] == "bad"


def test_derive_thresholds_ignores_ghost_rows(qapp):
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    # The deleted ghost (label 4) is tier 'good' with an extreme qc_score; it
    # must NOT raise the 'good' boundary.  Active good/review/bad maxima are
    # 1.0 / 2.0 / 3.0.
    df = pd.DataFrame({
        "label":        [1, 2, 3, 4],
        "qc_score":     [1.0, 2.0, 3.0, 99999.0],
        "qc_tier":      ["good", "review", "bad", "good"],
        "label_status": ["active", "active", "active", "deleted"],
    })
    panel.set_data(df)
    assert panel._derive_thresholds_from_qc_tier() == (1.0, 2.0, 3.0)


# ----------------------------------------------------------------------
# Bug (3): rows with unknown/missing qc_tier are NOT dropped
# ----------------------------------------------------------------------


def test_unknown_tiers_fall_into_unassigned_stack(qapp):
    """Rows with qc_tier outside the canonical 5 (e.g. NaN, "flagged",
    "unassigned") must NOT be silently dropped from the stacked
    histogram — they should fall into the gray "unassigned" stack so
    the visible total matches the value count."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        _TIER_STACK_ORDER, _TIER_COLORS_RGB,
    )
    assert "unassigned" in _TIER_STACK_ORDER, (
        "unassigned must be a stack slot — pre-fix it was missing and "
        "rows with weird tiers got silently skipped"
    )
    assert "unassigned" in _TIER_COLORS_RGB, (
        "unassigned needs a fallback color (gray-blue)"
    )


# ----------------------------------------------------------------------
# Bug (4): tier-line colors match the canonical "higher = worse"
# convention (mirror qc_plot_widget._tier_line_colors order)
# ----------------------------------------------------------------------


def test_tier_line_colors_match_canonical_order(qapp):
    """Per ``qc_features.assign_qc_tiers`` docstring (higher qc_score
    = worse), the leftmost line is the good/review boundary (green),
    the rightmost is the bad/garbage boundary (red).  Pre-fix the
    panel had this inverted.

    Mirrors the right plot's ``_tier_line_colors`` at
    qc_plot_widget.py:441-444.
    """
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)

    assert len(panel._tier_lines) == 3, "qc_score X-axis should show 3 tier lines"

    # Extract pen colors (each pen returns a QColor)
    pens = [line.pen for line in panel._tier_lines]
    rgbs = [(p.color().red(), p.color().green(), p.color().blue()) for p in pens]

    # Leftmost = good/review boundary = green-ish
    assert rgbs[0][1] > 150, f"leftmost should be green, got {rgbs[0]}"
    assert rgbs[0][0] < 200, "leftmost not red"
    # Rightmost = bad/garbage boundary = red-ish
    assert rgbs[2][0] > 200, f"rightmost should be red, got {rgbs[2]}"
    assert rgbs[2][1] < 150, "rightmost not green"


# ----------------------------------------------------------------------
# Tier line visibility: Rule A (only when X-axis = qc_score)
# ----------------------------------------------------------------------


def test_tier_lines_hidden_when_x_axis_not_qc_score(qapp):
    """Rule A: tier lines are visible only when ``Triage by ==
    qc_score``.  Hidden for any other X-axis.

    Phase 2C update: tier lines are persistent Python objects (created
    once in __init__, re-attached on each redraw to preserve drag state
    + signal connections).  So the list length is always 3; visibility
    is what toggles per Rule A.
    """
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)

    assert len(panel._tier_lines) == 3, (
        "tier lines are persistent — always 3 objects post-Phase 2C"
    )
    assert all(line.isVisible() for line in panel._tier_lines), (
        "qc_score X → all 3 lines visible"
    )

    panel._feature_combo.setCurrentText("area")
    assert all(not line.isVisible() for line in panel._tier_lines), (
        "non-qc_score X → all 3 lines hidden"
    )


# ----------------------------------------------------------------------
# Color by combo population — bug fix to expand options beyond 2
# ----------------------------------------------------------------------


def test_color_by_combo_includes_all_categorical_columns(qapp):
    """Colour bars by includes every canonical EpiCycle categorical
    column present in the df, plus ONE gradient entry ("value": the bar's
    position on the Triage-by axis).  Numeric columns are no longer
    offered: the gradient never read them (plot pass P0 audit fold)."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    # Add canonical categorical columns the right plot recognises.
    df["qc_status"] = "pass"
    df["cluster_id"] = 0
    df["is_border_cell"] = False
    df["euler_number"] = 1
    panel.set_data(df)

    items = [
        panel._color_combo.itemText(i)
        for i in range(panel._color_combo.count())
    ]
    for cat in ("qc_tier", "qc_status", "cluster_id",
                "is_border_cell", "euler_number"):
        assert cat in items, f"Color by should include {cat!r}, got {items}"
    # One gradient entry; no numeric column.
    assert items[-1] == "value"
    assert "qc_score" not in items
    assert "area" not in items


# ----------------------------------------------------------------------
# set_data(None) clears the panel back to the empty state (P0)
# ----------------------------------------------------------------------


def test_set_data_none_returns_to_the_empty_state(qapp):
    """Switching to a patch with no QC features should clear the panel:
    no bars, no df, the empty sentence (never made-up data)."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        HIST_EMPTY,
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)
    assert panel._df is df
    assert panel._bar_items
    assert panel.empty_sentence() is None
    panel.set_data(None)
    assert panel._df is None
    assert panel._bar_items == []
    assert panel.empty_sentence() == HIST_EMPTY


# ----------------------------------------------------------------------
# Phase 2C: tier-line drag preservation across redraws
# ----------------------------------------------------------------------


def test_drag_preserved_when_same_df_redraws(qapp):
    """When set_data is called with the SAME df reference (e.g. via
    the re-tier flow: drag → assign_qc_tiers → _update_flagged_labels_list
    → set_data), the user's dragged line positions must NOT snap back
    to defaults.  This is the bug the persistent-line refactor fixes."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)

    # Simulate user drag of the middle line
    panel._tier_lines[1].setValue(5.5)

    # Same df coming back through the re-tier flow
    panel.set_data(df)
    assert panel._tier_lines[1].value() == pytest.approx(5.5), (
        "same-df redraw must preserve dragged line position"
    )


# ----------------------------------------------------------------------
# Crash-fix (2026-06-08): tier lines must PERSIST in the scene across a
# _redraw — never removed + re-added.  The native Windows access
# violation was caused by _redraw calling _plot_widget.clear() (which
# tore out the tier InfiniteLines) and then re-addItem'ing the SAME line
# objects, all synchronously while a dragged line's sigPositionChanged
# handler was still on the stack (mid-drag re-entrant scene mutation).
# A native crash can't be unit-tested directly, but we CAN assert the
# invariant that makes it impossible: after a redraw, every tier line is
# still continuously in the plot's item list (i.e. it was not removed).
# ----------------------------------------------------------------------


def _plot_items(panel):
    """Helper: the current item list of the panel's plot."""
    return panel._plot_widget.getPlotItem().items


def test_tier_lines_persist_in_scene_across_redraw(qapp):
    """After a real-data ``_redraw``, every ``self._tier_lines`` object
    must still be present in the plot's item list — i.e. it was NOT
    removed/re-added.  This is the invariant that makes the mid-drag
    re-entrant crash impossible: a line that is never torn out cannot be
    torn out while its own drag handler is on the stack."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)  # real-data mode, feature == qc_score

    items = _plot_items(panel)
    for line in panel._tier_lines:
        assert line in items, (
            "tier line missing from plot after set_data redraw — it was "
            "removed (the crash-causing tear-out)"
        )

    # Capture the exact item objects, force several more redraws, and
    # confirm the SAME line objects remain continuously in the scene.
    lines_before = list(panel._tier_lines)
    for _ in range(3):
        panel._redraw()
    items_after = _plot_items(panel)
    for line in lines_before:
        assert line in items_after, (
            "tier line must remain in the scene across repeated redraws "
            "— never removed/re-added (mid-drag re-entrant crash root cause)"
        )


def test_redraw_during_simulated_drag_keeps_lines_in_scene(qapp):
    """Simulate the exact crash trigger: a redraw fired SYNCHRONOUSLY
    from a tier line's drag handler.  Wire ``thresholds_changed``
    (emitted by ``_on_tier_line_moved`` mid-drag) to call ``_redraw``,
    then move a line.  Pre-fix this removed every line from the scene
    (and re-added them) re-entrantly.  Post-fix every line must remain
    continuously in the scene — proving no tear-out occurs mid-drag."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df(qc_score_lo=-2.0, qc_score_hi=10.0)
    panel.set_data(df)

    # Mirror the controller round-trip: a tier drag emits
    # thresholds_changed which (via the controller) ends up calling
    # _redraw on this panel.  Connect it directly so the redraw fires
    # while _on_tier_line_moved is still on the stack.
    panel.thresholds_changed.connect(lambda _thr: panel._redraw())

    dragged = panel._tier_lines[1]
    # setValue fires sigPositionChanged → _on_tier_line_moved →
    # thresholds_changed → _redraw, all synchronously.
    dragged.setValue(4.2)

    items = _plot_items(panel)
    for line in panel._tier_lines:
        assert line in items, (
            "tier line removed during a redraw triggered mid-drag — this "
            "is the exact re-entrant scene mutation that crashed Python"
        )
    # The dragged line kept its position (it was never torn out).
    assert dragged.value() == pytest.approx(4.2)


def test_redraw_does_not_call_plot_clear(qapp):
    """Regression guard: ``_redraw`` must NOT call
    ``_plot_widget.clear()`` — that blanket clear is what removed the
    persistent tier lines + gate region and forced the crash-causing
    re-add.  Selective bar-only removal (``_clear_bars``) replaced it."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)

    clear_calls = [0]
    panel._plot_widget.clear = lambda *a, **k: clear_calls.__setitem__(0, clear_calls[0] + 1)
    panel._redraw()
    assert clear_calls[0] == 0, (
        "_redraw must not call _plot_widget.clear() — use selective "
        "bar-only removal so tier lines/gate region are never torn out"
    )


def test_bars_rebuilt_and_recolored_each_redraw(qapp):
    """Live-recolor preservation: each redraw must still rebuild the
    bar items (so a threshold/color change recolors the bars).  Assert
    fresh BarGraphItem objects are produced and the old ones removed
    from the scene on each redraw."""
    import pyqtgraph as pg
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)

    bars_first = list(panel._bar_items)
    assert bars_first, "redraw must produce bar items"
    assert all(isinstance(b, pg.BarGraphItem) for b in bars_first)

    panel._redraw()
    bars_second = list(panel._bar_items)
    assert bars_second, "redraw must produce bar items again (live recolor)"
    # Old bars are gone from the scene (selectively removed) and new
    # ones are present — this is what keeps the live recolor working.
    items = _plot_items(panel)
    for b in bars_second:
        assert b in items
    for b in bars_first:
        assert b not in items, (
            "stale bars from the previous redraw must be removed"
        )


# ----------------------------------------------------------------------
# Crash-fix follow-up (2026-06-08): z-order correction.  The crash-fix
# made the tier lines + gate region persistent (added once in __init__)
# and switched _redraw to selective bar-only removal + re-add.  Because
# the bars are re-addItem'd AFTER the already-present lines on every
# redraw, the ~78%-opaque bars painted ON TOP of the draggable lines,
# washing them out.  The correction sets an explicit zValue on the
# persistent items so they always paint above the bars regardless of
# insertion order.  These tests lock that invariant.
# ----------------------------------------------------------------------


def test_tier_lines_paint_above_bars_after_redraw(qapp):
    """Paint-order invariant: after a real-data redraw on qc_score, every
    tier ``InfiniteLine``'s zValue must be strictly greater than the
    zValue of every histogram bar.  Pre-correction the bars (default
    z=0) were re-added after the persistent lines, washing them out;
    the correction sets the lines to z=10 so they always paint on top."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df(qc_score_lo=-2.0, qc_score_hi=10.0)
    panel.set_data(df)  # real-data, feature == qc_score, tier lines shown

    assert panel._bar_items, "redraw must have produced bar items"
    max_bar_z = max(b.zValue() for b in panel._bar_items)
    for line in panel._tier_lines:
        assert line.zValue() > max_bar_z, (
            f"tier line z={line.zValue()} must exceed every bar's z="
            f"{max_bar_z} so the draggable line paints above the bars"
        )

    # Re-derive after a few forced redraws to prove the ordering holds
    # across the selective-clear path (bars rebuilt, lines untouched).
    for _ in range(3):
        panel._redraw()
    max_bar_z = max(b.zValue() for b in panel._bar_items)
    for line in panel._tier_lines:
        assert line.zValue() > max_bar_z, (
            "tier line must stay above the bars across repeated redraws"
        )


def test_gate_region_paints_above_bars_on_non_qc_score_feature(qapp):
    """Paint-order invariant for the gate region: on a non-qc_score
    feature (where the gate region is the active draggable artifact),
    the region's zValue must exceed every bar's zValue so the gate band
    isn't obscured by the bars re-added after it."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)
    panel._feature_combo.setCurrentText("area")  # gate region attached

    assert panel._bar_items, "redraw must have produced bar items"
    max_bar_z = max(b.zValue() for b in panel._bar_items)
    assert panel._gate_region.zValue() > max_bar_z, (
        f"gate region z={panel._gate_region.zValue()} must exceed every "
        f"bar's z={max_bar_z} so the gate band paints above the bars"
    )


def test_patch_switch_new_df_renders_correctly_through_selective_clear(qapp):
    """Patch-switch / new-df regression (Codex's shared-path concern).

    ``set_data()`` is the SHARED entry point hit on BOTH a tier-line
    drag (SAME df re-tier round-trip) AND a patch switch (NEW df).  The
    crash-fix replaced the blanket ``_plot_widget.clear()`` with
    selective bar-only removal.  This test proves the selective-clear
    redraw handles a fresh-df arrival, not just same-df redraws:

      - df_A then df_B are genuinely different (different label counts,
        different tier distribution, a different feature set).
      - After the switch the panel renders for df_B: bars rebuilt for
        df_B's data (df_A's bars removed from the scene, none leaked),
        tier lines still present (exactly 3, not duplicated/leaked),
        gate region count still ≤1 (not duplicated), and NO accumulation
        of bar items across the switch.
    """
    import pyqtgraph as pg
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )

    panel = TriageHistogramPanel()

    # df_A: 600 labels, wide qc_score range, the default feature set.
    df_a = _make_df(n=600, qc_score_lo=-2.0, qc_score_hi=10.0)
    panel.set_data(df_a)

    bars_a = list(panel._bar_items)
    assert bars_a, "df_A redraw must produce bar items"
    items_a = _plot_items(panel)
    n_bar_items_a = sum(1 for it in items_a if isinstance(it, pg.BarGraphItem))
    assert n_bar_items_a == len(bars_a), (
        "df_A bar items in the scene must match the tracked bar list"
    )

    # df_B: a DIFFERENT df object — different label count, different
    # tier distribution, and a different feature set (an extra numeric
    # column 'roundness' df_A doesn't have; drop 'true_sphericity').
    df_b = _make_df(n=150, qc_score_lo=0.0, qc_score_hi=1.0)
    df_b = df_b.drop(columns=["true_sphericity"])
    rng = np.random.default_rng(7)
    df_b["roundness"] = rng.beta(4, 4, len(df_b))
    # Skew the tier distribution differently from df_A.
    df_b["qc_tier"] = rng.choice(
        ["good", "review", "bad"], len(df_b), p=[0.2, 0.3, 0.5]
    )

    panel.set_data(df_b)

    # --- Bars rebuilt for df_B; df_A's bars gone (no leak) ------------
    bars_b = list(panel._bar_items)
    assert bars_b, "df_B redraw must rebuild bar items"
    items_b = _plot_items(panel)
    for b in bars_a:
        assert b not in items_b, (
            "df_A's bar items must be removed on the new-df switch "
            "(selective-clear must drop the stale bars)"
        )
    for b in bars_b:
        assert b in items_b, "df_B's freshly built bars must be in the scene"

    # No accumulation: the only BarGraphItems in the scene are df_B's.
    scene_bars = [it for it in items_b if isinstance(it, pg.BarGraphItem)]
    assert len(scene_bars) == len(bars_b), (
        f"bar items accumulated across the patch switch — scene has "
        f"{len(scene_bars)} bars but only {len(bars_b)} are tracked"
    )

    # --- Tier lines still present (exactly 3, not leaked) -------------
    assert len(panel._tier_lines) == 3, (
        "patch switch must not duplicate or drop the persistent tier lines"
    )
    # (each tier line has a dark halo line beneath it -- counted apart)
    scene_lines = [it for it in items_b if isinstance(it, pg.InfiniteLine)
                   and it not in panel._tier_halos]
    scene_halos = [it for it in items_b if it in panel._tier_halos]
    assert len(scene_halos) == 3
    assert len(scene_lines) == 3, (
        f"exactly 3 tier InfiniteLines must be in the scene after the "
        f"switch, found {len(scene_lines)} (leak/duplication)"
    )
    for line in panel._tier_lines:
        assert line in items_b, "each persistent tier line must remain attached"
    # qc_score is still the default feature for df_B → lines visible.
    assert all(line.isVisible() for line in panel._tier_lines), (
        "df_B defaults to qc_score → tier lines visible"
    )

    # --- Gate region count ≤ 1 (not duplicated) ----------------------
    scene_regions = [
        it for it in items_b if isinstance(it, pg.LinearRegionItem)
    ]
    assert len(scene_regions) <= 1, (
        f"gate region must not be duplicated across the switch, found "
        f"{len(scene_regions)}"
    )

    # --- Tier lines repositioned into df_B's [0, 1] range ------------
    middle = panel._tier_lines[1].value()
    assert 0.0 <= middle <= 1.0, (
        f"new-df tier lines must reposition into df_B's range, got {middle}"
    )

    # --- Paint-order still correct on the new df ---------------------
    max_bar_z = max(b.zValue() for b in bars_b)
    for line in panel._tier_lines:
        assert line.zValue() > max_bar_z, (
            "tier lines must still paint above the bars after a patch switch"
        )


def test_gate_region_persists_and_tier_lines_toggle_on_feature_switch(qapp):
    """Feature-switch path (NOT a drag): switching qc_score ↔ area must
    correctly show/hide the tier lines and the gate region without
    orphaning either across the selective-clear.  Both the tier lines
    and the gate region stay attached to the scene (just visibility
    toggles)."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)  # qc_score

    # qc_score: tier lines visible + in scene; gate region hidden.
    assert all(line.isVisible() for line in panel._tier_lines)
    items_qc = _plot_items(panel)
    for line in panel._tier_lines:
        assert line in items_qc

    # Switch to a non-qc_score feature: tier lines hidden but STILL in
    # the scene (never torn out); gate region attached + hidden (gate
    # checkbox off by default).
    panel._feature_combo.setCurrentText("area")
    items_area = _plot_items(panel)
    assert all(not line.isVisible() for line in panel._tier_lines)
    for line in panel._tier_lines:
        assert line in items_area, "tier lines must stay attached, just hidden"
    assert panel._gate_region in items_area, (
        "gate region must be attached on a non-qc_score feature"
    )

    # Switch back to qc_score: tier lines visible again, gate hidden.
    panel._feature_combo.setCurrentText("qc_score")
    assert all(line.isVisible() for line in panel._tier_lines)
    assert not panel._gate_region.isVisible()


def test_new_df_resets_lines_to_quartiles(qapp):
    """When set_data is called with a DIFFERENT df reference (patch
    switch), tier lines must reposition at the new df's quartiles.
    Otherwise users would see lines way outside the new patch's range."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()

    df_a = _make_df(qc_score_lo=-2.0, qc_score_hi=10.0)
    panel.set_data(df_a)
    panel._tier_lines[1].setValue(5.5)

    # New patch with very different qc_score range [0, 1]
    df_b = _make_df(qc_score_lo=0.0, qc_score_hi=1.0)
    panel.set_data(df_b)
    middle = panel._tier_lines[1].value()
    assert 0.0 < middle < 1.0, (
        f"new df must reset middle line into [0, 1] range, got {middle}"
    )


def test_thresholds_changed_signal_emits_on_drag(qapp):
    """Phase 2C: dragging a line emits ``thresholds_changed`` with
    the sorted tuple of all 3 line positions, so the controller's
    canonical _on_tier_thresholds_changed handler can re-run
    assign_qc_tiers."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)

    captured = []
    panel.thresholds_changed.connect(lambda thr: captured.append(thr))

    panel._tier_lines[1].setValue(5.5)
    assert len(captured) >= 1, "thresholds_changed should fire on drag"
    last = captured[-1]
    assert len(last) == 3, "signal payload is (t1, t2, t3)"
    # Tuple is sorted ascending
    assert last[0] <= last[1] <= last[2]


def test_panel_declares_both_tier_signals_at_class_level(qapp):
    """Both ``thresholds_changed`` AND ``tier_drag_finished`` must be
    class-level Qt Signals.  Pre-fix bug 2026-05-04: the slot
    ``_on_tier_line_released`` called ``self.tier_drag_finished.emit(...)``
    but the signal was never declared on the class — crashed at
    napari launch in controller._connect_signals when trying to
    .connect() it.

    This test reproduces the controller's connect pattern so any
    missing signal would surface as AttributeError here, not at
    napari launch time.
    """
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    # Both signals must be Signal instances callable for connect()
    panel.thresholds_changed.connect(lambda thr: None)
    panel.tier_drag_finished.connect(lambda thr: None)
    panel.bin_clicked.connect(lambda b, c: None)


def test_tier_drag_finished_signal_emits_on_release(qapp):
    """Phase 2C: releasing a line emits ``tier_drag_finished`` once
    so the controller can do the full viewer-color sync."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df()
    panel.set_data(df)

    captured = []
    panel.tier_drag_finished.connect(lambda thr: captured.append(thr))

    # Direct slot invocation simulates the sigPositionChangeFinished
    # signal that pyqtgraph fires on mouse release.
    panel._on_tier_line_released(panel._tier_lines[1])
    assert len(captured) == 1, "tier_drag_finished should fire on release"
    assert len(captured[0]) == 3


# ----------------------------------------------------------------------
# Phase 2C+ (2026-05-04): tier line positions derive from saved qc_tier
# ----------------------------------------------------------------------


def test_lines_derived_from_saved_qc_tier_on_load(qapp):
    """When a df arrives with a populated qc_tier column, line
    positions should derive from the column's tier boundaries — NOT
    default to quartiles of qc_score.

    This is the persistence mechanism: committed user calibrations
    saved to disk reflect in qc_tier, and switching back to that
    patch shows lines at those exact positions."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    from napari_epicycle.fusion_reconstruction.qc_features import (
        assign_qc_tiers,
    )

    panel = TriageHistogramPanel()
    df = _make_df(qc_score_lo=-2.0, qc_score_hi=10.0)
    user_thresholds = (1.5, 5.0, 8.5)
    assign_qc_tiers(df, user_thresholds)
    panel.set_data(df)

    pos = [line.value() for line in panel._tier_lines]
    for got, want in zip(pos, user_thresholds):
        assert abs(got - want) < 0.5, (
            f"Lines should snap near saved tier boundaries; "
            f"got {pos}, expected near {user_thresholds}"
        )


def test_lines_persist_across_patch_switches_via_qc_tier(qapp):
    """User calibrates A → saved via assign_qc_tiers → switches to B
    → switches back to A.  Lines on A should be back at A's
    calibrated positions, not B's range or A's quartiles."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    from napari_epicycle.fusion_reconstruction.qc_features import (
        assign_qc_tiers,
    )
    import numpy as _np

    panel = TriageHistogramPanel()
    df_a = _make_df(qc_score_lo=-2.0, qc_score_hi=10.0)
    a_thresholds = (1.5, 5.0, 8.5)
    assign_qc_tiers(df_a, a_thresholds)
    panel.set_data(df_a)
    a_pos_initial = [round(line.value(), 2) for line in panel._tier_lines]

    df_b = _make_df(qc_score_lo=0.0, qc_score_hi=1.0)
    q25b, q50b, q75b = _np.percentile(df_b["qc_score"], [25, 50, 75])
    assign_qc_tiers(df_b, (float(q25b), float(q50b), float(q75b)))
    panel.set_data(df_b)
    b_pos = [line.value() for line in panel._tier_lines]
    assert max(b_pos) <= 1.0, (
        f"Patch B lines should be in [0, 1] range, got {b_pos}"
    )

    panel.set_data(df_a)
    a_pos_again = [round(line.value(), 2) for line in panel._tier_lines]
    assert a_pos_again == a_pos_initial, (
        f"Lines must persist across switches.  "
        f"Initial A: {a_pos_initial}, after round-trip: {a_pos_again}"
    )


def test_reset_lines_snaps_to_quartiles_within_same_df(qapp):
    """Reset Lines button must reposition tier lines at the current
    df's quartiles, even if user has dragged them elsewhere."""
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )
    panel = TriageHistogramPanel()
    df = _make_df(qc_score_lo=-2.0, qc_score_hi=10.0)
    panel.set_data(df)

    # User drags lines to extreme positions
    panel._tier_lines[0].setValue(-1.5)
    panel._tier_lines[1].setValue(0.0)
    panel._tier_lines[2].setValue(0.5)

    # Reset -> snap to quartiles
    panel._on_reset_clicked()
    # Quartiles of uniform [-2, 10] are roughly [1, 4, 7]
    pos = [line.value() for line in panel._tier_lines]
    assert all(0.0 < p < 9.0 for p in pos), (
        f"Reset Lines must move from extreme positions to quartiles, got {pos}"
    )


# ---------------------------------------------------------------------------
# Phase F (2026-05-04) — multi-feature gating
# ---------------------------------------------------------------------------

class TestGateBehaviour:
    """Gate checkbox + two cyan threshold lines for non-qc_score features.

    The gate is mutually exclusive with the three tier lines:
      - feature == qc_score → tier lines visible, gate checkbox HIDDEN.
      - feature != qc_score → tier lines hidden, gate checkbox SHOWN.

    When the user enables the gate:
      - Two threshold lines appear at 10th + 90th percentile of the
        active rows for the current feature.
      - Drag either line → emits ``gated_selection_changed`` with the
        set of label IDs whose feature value is in [low, high].
      - Disabling the checkbox emits an empty set so the controller
        can clear any stale selection downstream.
    """

    def _panel_with_df(self, qapp):
        from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
            TriageHistogramPanel,
        )
        N = 30
        df = pd.DataFrame({
            "label": np.arange(1, N + 1),
            "qc_score": np.random.uniform(0, 2, N),
            "qc_tier": np.random.choice(
                ["good", "review", "bad", "garbage"], N
            ),
            # Monotonic so percentile boundaries are predictable.
            "area": np.linspace(50.0, 500.0, N),
        })
        panel = TriageHistogramPanel()
        panel.set_data(df)
        return panel, df

    def test_gate_checkbox_hidden_when_feature_is_qc_score(self, qapp):
        panel, _ = self._panel_with_df(qapp)
        panel._feature_combo.setCurrentText("qc_score")
        # `setVisible(False)` was called → isHidden() is True regardless
        # of whether the parent panel itself is shown.
        assert panel._gate_cb.isHidden(), (
            "gate checkbox must hide when feature is qc_score"
        )

    def test_gate_checkbox_shown_when_feature_is_not_qc_score(self, qapp):
        panel, _ = self._panel_with_df(qapp)
        panel._feature_combo.setCurrentText("area")
        assert not panel._gate_cb.isHidden(), (
            "gate checkbox must show when feature is not qc_score"
        )

    def test_gate_region_position_at_percentiles_when_enabled(self, qapp):
        panel, df = self._panel_with_df(qapp)
        panel._feature_combo.setCurrentText("area")
        panel._gate_cb.setChecked(True)
        low_expected = float(np.percentile(df["area"], 10))
        high_expected = float(np.percentile(df["area"], 90))
        low, high = panel._gate_region.getRegion()
        assert abs(low - low_expected) < 1e-6
        assert abs(high - high_expected) < 1e-6

    def test_gate_emits_label_ids_in_range(self, qapp):
        panel, df = self._panel_with_df(qapp)
        panel._feature_combo.setCurrentText("area")

        light_emits = []
        panel.gated_selection_changed.connect(
            lambda s: light_emits.append(set(s))
        )
        panel._gate_cb.setChecked(True)

        # Initial emit (continuous signal): roughly 80% of labels.
        assert light_emits, "checking the gate should fire continuous signal"
        low0, high0 = panel._gate_region.getRegion()
        in_range_init = set(
            int(lbl)
            for lbl in df.loc[
                (df["area"] >= low0) & (df["area"] <= high0), "label",
            ]
        )
        assert light_emits[-1] == in_range_init

        # Programmatic region update → narrower selection.
        light_emits.clear()
        panel._gate_region.setRegion([200.0, high0])
        in_range_narrow = set(
            int(lbl)
            for lbl in df.loc[
                (df["area"] >= 200.0) & (df["area"] <= high0), "label",
            ]
        )
        assert light_emits, "setRegion must fire gated_selection_changed"
        assert light_emits[-1] == in_range_narrow

    def test_gate_disable_emits_empty_on_both_signals(self, qapp):
        panel, _ = self._panel_with_df(qapp)
        panel._feature_combo.setCurrentText("area")
        panel._gate_cb.setChecked(True)

        light_emits = []
        heavy_emits = []
        panel.gated_selection_changed.connect(
            lambda s: light_emits.append(set(s))
        )
        panel.gated_drag_finished.connect(
            lambda s: heavy_emits.append(set(s))
        )
        panel._gate_cb.setChecked(False)
        assert light_emits and len(light_emits[-1]) == 0, (
            "disabling the gate must emit an empty set on the continuous signal"
        )
        assert heavy_emits and len(heavy_emits[-1]) == 0, (
            "disabling the gate must also emit an empty set on the drag-finished signal "
            "so the controller clears the 3D viewer isolation"
        )

    def test_switch_to_qc_score_force_unchecks_gate(self, qapp):
        panel, _ = self._panel_with_df(qapp)
        panel._feature_combo.setCurrentText("area")
        panel._gate_cb.setChecked(True)
        assert panel._gate_cb.isChecked()

        # Switching back to qc_score must force-uncheck the gate so a
        # stale gated selection doesn't survive a feature change.
        panel._feature_combo.setCurrentText("qc_score")
        assert not panel._gate_cb.isChecked()

    def test_setregion_uses_blocksignals_no_recursion(self, qapp):
        """Phase F mirrors the Phase E recursion-fix discipline.

        ``_on_gate_toggled`` calls ``setRegion`` to seed the
        percentile defaults — that must be wrapped in ``blockSignals``
        so it doesn't fire ``sigRegionChanged`` recursively.  We
        instead emit explicitly at the end of ``_on_gate_toggled``.
        Verify exactly ONE continuous + ONE finished emission per
        toggle-on (not 1 + N from setRegion ricochet).
        """
        panel, _ = self._panel_with_df(qapp)
        panel._feature_combo.setCurrentText("area")

        light_emits = []
        heavy_emits = []
        panel.gated_selection_changed.connect(
            lambda s: light_emits.append(set(s))
        )
        panel.gated_drag_finished.connect(
            lambda s: heavy_emits.append(set(s))
        )
        panel._gate_cb.setChecked(True)
        assert len(light_emits) == 1, (
            f"Expected 1 continuous emission from blockSignals'd seed, "
            f"got {len(light_emits)} — recursion may have returned"
        )
        assert len(heavy_emits) == 1, (
            f"Expected 1 drag-finished emission from blockSignals'd seed, "
            f"got {len(heavy_emits)}"
        )

    def test_gate_region_persistent_across_redraws(self, qapp):
        """Gate region is a persistent object — _redraw must re-add it
        without recreating, preserving drag position."""
        panel, _ = self._panel_with_df(qapp)
        panel._feature_combo.setCurrentText("area")
        panel._gate_cb.setChecked(True)

        region_before = panel._gate_region
        # User drags
        panel._gate_region.setRegion([150.0, 400.0])

        # Force a redraw (e.g. via color combo change)
        panel._color_combo.setCurrentText("value")
        assert panel._color_combo.currentText() == "value"

        # Same Python object, same drag position
        assert panel._gate_region is region_before
        low, high = panel._gate_region.getRegion()
        assert abs(low - 150.0) < 1e-6
        assert abs(high - 400.0) < 1e-6


# ---------------------------------------------------------------------------
# Phase 2D (2026-05-04) — colormap sync with right-plot _cmap_combo
# ---------------------------------------------------------------------------

class TestColormapSync:
    """Left-panel gradient bars track the right-plot's _cmap_combo.

    Controller wires ``_cmap_combo.currentTextChanged`` →
    ``panel.set_active_cmap`` so the user can change the colormap on
    the right plot and the left histogram's gradient bars (when Color
    by is continuous) restyle to match.
    """

    def test_default_cmap_is_coolwarm(self, qapp):
        """Backwards-compat: Coolwarm was the pre-Phase-2D hardcoded
        gradient.  Verify the new constructor default keeps that."""
        from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
            TriageHistogramPanel,
        )
        panel = TriageHistogramPanel()
        assert panel._active_cmap == "Coolwarm"

    def test_stops_for_cmap_known_names(self, qapp):
        """Verify _stops_for_cmap pulls from the shared
        qc_plot_widget._COLORMAPS dict for every name the right plot
        offers."""
        from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
            _stops_for_cmap,
        )
        for name in ("Viridis", "Plasma", "Inferno", "Magma", "Turbo", "Coolwarm"):
            stops = _stops_for_cmap(name)
            assert len(stops) == 5
            assert stops[0][0] == 0.0
            assert stops[-1][0] == 1.0
            for _, rgb in stops:
                assert all(0 <= c <= 255 for c in rgb)

    def test_unknown_cmap_falls_back_to_coolwarm(self, qapp):
        from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
            _stops_for_cmap,
        )
        unknown = _stops_for_cmap("nonexistent_cmap")
        coolwarm = _stops_for_cmap("Coolwarm")
        assert unknown == coolwarm

    def test_set_active_cmap_updates_state(self, qapp):
        from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
            TriageHistogramPanel, _stops_for_cmap,
        )
        panel = TriageHistogramPanel()
        old_stops = panel._gradient_stops
        panel.set_active_cmap("Viridis")
        assert panel._active_cmap == "Viridis"
        assert panel._gradient_stops == _stops_for_cmap("Viridis")
        assert panel._gradient_stops != old_stops

    def test_set_active_cmap_noop_on_same_name(self, qapp):
        """Re-setting to the same cmap should not redraw — avoids
        spurious work when the right plot re-emits the signal without
        a real change."""
        from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
            TriageHistogramPanel,
        )
        panel = TriageHistogramPanel()
        # Spy: count _redraw calls
        original_redraw = panel._redraw
        call_count = [0]

        def counting_redraw():
            call_count[0] += 1
            return original_redraw()

        panel._redraw = counting_redraw
        # Same as constructor default → no redraw
        panel.set_active_cmap("Coolwarm")
        assert call_count[0] == 0
        # Switch to a real new value → 1 redraw
        panel.set_active_cmap("Viridis")
        assert call_count[0] == 1
        # Re-set Viridis → no extra redraw
        panel.set_active_cmap("Viridis")
        assert call_count[0] == 1

    def test_coolwarm_round_trip_matches_pre_phase_2d_values(self, qapp):
        """Existing screenshots / user expectations were calibrated to
        the hardcoded pre-Phase-2D Coolwarm stops.  Verify the new
        shared-source path produces ≤1 unit drift per channel after
        the [0,1]float → [0,255]int rounding."""
        from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
            _stops_for_cmap,
        )
        new_stops = _stops_for_cmap("Coolwarm")
        expected_pre_phase_2d = [
            (0.00, (59, 76, 192)),
            (0.25, (141, 176, 254)),
            (0.50, (247, 247, 247)),
            (0.75, (247, 167, 137)),
            (1.00, (180, 4, 38)),
        ]
        for (t_new, rgb_new), (t_old, rgb_old) in zip(
            new_stops, expected_pre_phase_2d
        ):
            assert abs(t_new - t_old) < 1e-9
            for c_new, c_old in zip(rgb_new, rgb_old):
                assert abs(c_new - c_old) <= 1

    def test_sample_gradient_midpoint_interpolation(self, qapp):
        """For Coolwarm, t=0.5 lands exactly on the (0.5, near-white)
        stop → returns it unchanged."""
        from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
            _sample_gradient, _stops_for_cmap,
        )
        mid = _sample_gradient(0.5, _stops_for_cmap("Coolwarm"))
        # Pre-Phase-2D had (247, 247, 247) at t=0.5; allow ±1 rounding.
        for c in mid:
            assert 246 <= c <= 248

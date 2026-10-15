"""U0/U1 behavioral tests for curation selection sync coordinator.

Tests exercise real coordinator dispatch logic via lightweight harnesses.
No Qt or napari required — all surfaces are mocked callbacks.

Test plan:
1. test_plot_dispatch_builds_selectionref_with_df_indices
2. test_flagged_list_dispatch_builds_selectionref_with_primary_label
3. test_dispatch_calls_viewer_and_plot_adapters_once
4. test_dispatch_reentrant_guard_prevents_loop
5. test_dispatch_dedupe_identical_selection
6. test_population_missing_df_indices_rejected_per_existing_policy
7. test_navigator_adapter_called_when_annotator_present
"""

import pytest
from unittest.mock import MagicMock, call


# ---------------------------------------------------------------------------
# Import targets — these don't exist yet (fail-first)
# ---------------------------------------------------------------------------

from napari_epicycle.fusion_reconstruction.curation_selection_sync import (
    SelectionRef,
    CurationSelectionSyncCoordinator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coordinator(**adapter_overrides):
    """Build a coordinator with mock adapters for testing."""
    adapters = {
        "apply_viewer_isolation": MagicMock(),
        "apply_viewer_navigation": MagicMock(),
        "apply_plot_highlight": MagicMock(),
        "apply_flagged_list_selection": MagicMock(),
        "apply_navigator_selection": None,  # annotator not open by default
    }
    adapters.update(adapter_overrides)
    coord = CurationSelectionSyncCoordinator(**adapters)
    return coord, adapters


def _make_sel(*, source="plot", label_ids=None, df_indices=None,
              scope="single_patch", patch_key="patchA",
              mode="focus", primary_label_id=None):
    """Shorthand for building a SelectionRef."""
    label_ids = label_ids or [1]
    df_indices = df_indices or []
    primary = primary_label_id if primary_label_id is not None else label_ids[0]
    return SelectionRef(
        scope=scope,
        patch_key=patch_key,
        label_ids=label_ids,
        primary_label_id=primary,
        df_indices=df_indices,
        source=source,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# U0 Tests: SelectionRef construction
# ---------------------------------------------------------------------------

class TestSelectionRefConstruction:
    """Verify SelectionRef dataclass holds all required fields."""

    def test_selectionref_fields(self):
        sel = _make_sel(label_ids=[5], df_indices=[2], source="plot")
        assert sel.scope == "single_patch"
        assert sel.patch_key == "patchA"
        assert sel.label_ids == [5]
        assert sel.primary_label_id == 5
        assert sel.df_indices == [2]
        assert sel.source == "plot"
        assert sel.mode == "focus"
        assert hasattr(sel, "timestamp")

    def test_selectionref_population_scope(self):
        sel = _make_sel(scope="population", df_indices=[0, 3])
        assert sel.scope == "population"


# ---------------------------------------------------------------------------
# U0 Tests: Trace hook (observability)
# ---------------------------------------------------------------------------

class TestTraceHook:
    """Verify coordinator emits trace events when debug is enabled."""

    def test_trace_emitted_on_dispatch(self):
        """Coordinator calls trace_hook on every dispatch if provided."""
        trace_log = []
        coord, adapters = _make_coordinator()
        coord.set_trace_hook(lambda event, sel, extra: trace_log.append(
            (event, sel.source, sel.label_ids)
        ))
        sel = _make_sel(source="plot", label_ids=[7])
        coord.dispatch(sel)
        assert len(trace_log) == 1
        assert trace_log[0][0] == "dispatch"
        assert trace_log[0][1] == "plot"
        assert trace_log[0][2] == [7]

    def test_no_trace_when_hook_not_set(self):
        """No error when trace_hook is None."""
        coord, _ = _make_coordinator()
        sel = _make_sel()
        coord.dispatch(sel)  # should not raise


# ---------------------------------------------------------------------------
# U1 Tests: Coordinator dispatch
# ---------------------------------------------------------------------------

class TestCoordinatorDispatch:
    """Core coordinator dispatch tests."""

    def test_plot_dispatch_builds_selectionref_with_df_indices(self):
        """Plot-sourced selection carries df_indices to viewer adapter."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(source="plot", label_ids=[10, 20], df_indices=[0, 5])
        coord.dispatch(sel)

        adapters["apply_viewer_isolation"].assert_called_once()
        call_sel = adapters["apply_viewer_isolation"].call_args[0][0]
        assert call_sel.label_ids == [10, 20]
        assert call_sel.df_indices == [0, 5]

    def test_flagged_list_dispatch_builds_selectionref_with_primary_label(self):
        """Flagged-list selection carries primary_label_id to navigation."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(
            source="flagged_list",
            label_ids=[42],
            primary_label_id=42,
            df_indices=[7],
        )
        coord.dispatch(sel)

        adapters["apply_viewer_navigation"].assert_called_once()
        call_sel = adapters["apply_viewer_navigation"].call_args[0][0]
        assert call_sel.primary_label_id == 42

    def test_dispatch_calls_viewer_and_plot_adapters_once(self):
        """Each adapter is called exactly once per dispatch."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(label_ids=[1])
        coord.dispatch(sel)

        adapters["apply_viewer_isolation"].assert_called_once()
        adapters["apply_viewer_navigation"].assert_called_once()
        # Plot highlight is NOT called when source is plot (avoid echo)
        # Flagged list selection IS called when source is plot
        adapters["apply_flagged_list_selection"].assert_called_once()

    def test_plot_source_skips_plot_highlight(self):
        """When source is 'plot', don't echo back to plot highlight."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(source="plot", label_ids=[1])
        coord.dispatch(sel)
        adapters["apply_plot_highlight"].assert_not_called()

    def test_flagged_list_source_calls_plot_highlight(self):
        """When source is 'flagged_list', DO highlight in plot."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(source="flagged_list", label_ids=[1])
        coord.dispatch(sel)
        adapters["apply_plot_highlight"].assert_called_once()

    def test_dispatch_reentrant_guard_prevents_loop(self):
        """Nested dispatch calls are silently dropped."""
        coord, adapters = _make_coordinator()
        call_count = {"n": 0}

        def reentrant_adapter(sel):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Simulate feedback loop: adapter dispatches again
                coord.dispatch(sel)

        adapters["apply_viewer_isolation"] = reentrant_adapter
        coord = CurationSelectionSyncCoordinator(**adapters)
        sel = _make_sel(label_ids=[1])
        coord.dispatch(sel)

        # The reentrant call should be dropped
        assert call_count["n"] == 1

    def test_dispatch_dedupe_identical_selection(self):
        """Identical consecutive dispatches are deduped (second is no-op)."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(label_ids=[5], df_indices=[2], source="plot")
        coord.dispatch(sel)
        coord.dispatch(sel)

        # Viewer isolation called only once
        assert adapters["apply_viewer_isolation"].call_count == 1

    def test_dispatch_different_selection_not_deduped(self):
        """Different selections are NOT deduped."""
        coord, adapters = _make_coordinator()
        sel1 = _make_sel(label_ids=[5], source="plot")
        sel2 = _make_sel(label_ids=[6], source="plot")
        coord.dispatch(sel1)
        coord.dispatch(sel2)

        assert adapters["apply_viewer_isolation"].call_count == 2

    def test_population_missing_df_indices_rejected_per_existing_policy(self):
        """Population scope with single label and no df_indices → rejected."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(
            scope="population",
            label_ids=[42],
            df_indices=[],
        )
        coord.dispatch(sel)

        # Viewer isolation should NOT be called
        adapters["apply_viewer_isolation"].assert_not_called()
        adapters["apply_viewer_navigation"].assert_not_called()

    def test_population_with_df_indices_accepted(self):
        """Population scope WITH df_indices → dispatched normally."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(
            scope="population",
            label_ids=[42],
            df_indices=[3],
        )
        coord.dispatch(sel)

        adapters["apply_viewer_isolation"].assert_called_once()

    def test_navigator_adapter_called_when_annotator_present(self):
        """Navigator adapter is called when annotator is open."""
        nav_adapter = MagicMock()
        coord, adapters = _make_coordinator(
            apply_navigator_selection=nav_adapter
        )
        sel = _make_sel(source="plot", label_ids=[10])
        coord.dispatch(sel)

        nav_adapter.assert_called_once()

    def test_navigator_adapter_skipped_when_absent(self):
        """Navigator adapter=None → no error, just skipped."""
        coord, adapters = _make_coordinator(
            apply_navigator_selection=None
        )
        sel = _make_sel(source="plot", label_ids=[10])
        coord.dispatch(sel)  # should not raise

    def test_navigator_source_skips_navigator_adapter(self):
        """When source is 'navigator', don't echo back to navigator."""
        nav_adapter = MagicMock()
        coord, adapters = _make_coordinator(
            apply_navigator_selection=nav_adapter
        )
        sel = _make_sel(source="navigator", label_ids=[10])
        coord.dispatch(sel)

        nav_adapter.assert_not_called()


# ---------------------------------------------------------------------------
# U1 Tests: Clear/empty selection
# ---------------------------------------------------------------------------

class TestClearSelection:
    """Dispatch with empty label_ids clears isolation."""

    def test_empty_labels_clears_isolation(self):
        """Empty label_ids dispatches a clear to viewer isolation."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(label_ids=[], source="plot")
        # Need to override primary since helper defaults to label_ids[0]
        sel = SelectionRef(
            scope="single_patch",
            patch_key="patchA",
            label_ids=[],
            primary_label_id=None,
            df_indices=[],
            source="plot",
            mode="focus",
        )
        coord.dispatch(sel)

        adapters["apply_viewer_isolation"].assert_called_once()
        call_sel = adapters["apply_viewer_isolation"].call_args[0][0]
        assert call_sel.label_ids == []

    def test_clear_resets_dedupe(self):
        """After clear, re-selecting the same label is not deduped."""
        coord, adapters = _make_coordinator()
        sel1 = _make_sel(label_ids=[5], source="plot")
        clear = SelectionRef(
            scope="single_patch", patch_key="patchA",
            label_ids=[], primary_label_id=None,
            df_indices=[], source="plot", mode="focus",
        )
        sel2 = _make_sel(label_ids=[5], source="plot")
        coord.dispatch(sel1)
        coord.dispatch(clear)
        coord.dispatch(sel2)

        # sel1 + clear + sel2 = 3 isolation calls
        assert adapters["apply_viewer_isolation"].call_count == 3


# ---------------------------------------------------------------------------
# WP-4 tests — Navigator bridge isolation mode (Bug H)
# ---------------------------------------------------------------------------


class TestNavigatorBridgeIsolationMode:
    """WP-4: Bridge dispatches correct mode based on navigator isolation state."""

    def test_navigator_bridge_single_label_dispatches_focus(self):
        """Single label, no isolate → mode='focus'."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(
            source="navigator", label_ids=[42], mode="focus",
        )
        coord.dispatch(sel)
        adapters["apply_viewer_isolation"].assert_called_once()
        dispatched = adapters["apply_viewer_isolation"].call_args[0][0]
        assert dispatched.mode == "focus"

    def test_navigator_bridge_multi_label_isolate_dispatches_isolate_mode(self):
        """Multiple labels + isolate=True → mode='isolate'."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(
            source="navigator", label_ids=[42, 99, 101], mode="isolate",
        )
        coord.dispatch(sel)
        adapters["apply_viewer_isolation"].assert_called_once()
        dispatched = adapters["apply_viewer_isolation"].call_args[0][0]
        assert dispatched.mode == "isolate"
        assert dispatched.label_ids == [42, 99, 101]

    def test_navigator_bridge_includes_all_selected_label_ids(self):
        """All selected IDs are present in the dispatched SelectionRef."""
        coord, adapters = _make_coordinator()
        ids = [1, 5, 10, 20]
        sel = _make_sel(
            source="navigator", label_ids=ids, mode="isolate",
        )
        coord.dispatch(sel)
        dispatched = adapters["apply_viewer_isolation"].call_args[0][0]
        assert dispatched.label_ids == ids

    def test_navigator_bridge_isolate_off_dispatches_focus(self):
        """Even with multiple labels, isolate=False → mode='focus'."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(
            source="navigator", label_ids=[42, 99], mode="focus",
        )
        coord.dispatch(sel)
        dispatched = adapters["apply_viewer_isolation"].call_args[0][0]
        assert dispatched.mode == "focus"


# ---------------------------------------------------------------------------
# WP-5 tests — Flagged list → annotator navigation (Bug E)
# ---------------------------------------------------------------------------


class TestFlaggedListNavigation:
    """WP-5: Flagged list click triggers full annotator label switch even
    when Navigator widget is not open."""

    def test_flagged_list_click_dispatches_through_coordinator(self):
        """Flagged list selection dispatches to all adapters."""
        coord, adapters = _make_coordinator(
            apply_navigator_selection=MagicMock(),
        )
        sel = _make_sel(
            source="flagged_list", label_ids=[42], mode="focus",
            primary_label_id=42,
        )
        coord.dispatch(sel)

        # Navigator adapter should be called (source != navigator)
        adapters["apply_navigator_selection"].assert_called_once()

    def test_flagged_list_click_navigates_main_viewer(self):
        """Viewer navigation adapter is called for flagged list source."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(
            source="flagged_list", label_ids=[42], mode="focus",
            primary_label_id=42,
        )
        coord.dispatch(sel)
        adapters["apply_viewer_navigation"].assert_called_once()

    def test_flagged_list_click_with_navigator_closed_still_dispatches(self):
        """When navigator adapter is None, other adapters still fire."""
        coord, adapters = _make_coordinator(
            apply_navigator_selection=None,
        )
        sel = _make_sel(
            source="flagged_list", label_ids=[42], mode="focus",
            primary_label_id=42,
        )
        coord.dispatch(sel)
        # Viewer isolation and navigation still fire
        adapters["apply_viewer_isolation"].assert_called_once()
        adapters["apply_viewer_navigation"].assert_called_once()


# ---------------------------------------------------------------------------
# Isolation sync tests (F2/F3)
# ---------------------------------------------------------------------------


class TestIsolationSync:
    """Verify isolation state syncs between lasso, navigator, and plot."""

    def test_lasso_isolate_mode_passed_to_navigator_adapter(self):
        """Lasso with mode='isolate' should reach navigator adapter with that mode."""
        nav_adapter = MagicMock()
        coord, adapters = _make_coordinator(
            apply_navigator_selection=nav_adapter,
        )
        sel = _make_sel(
            source="plot", label_ids=[1, 2, 3], mode="isolate",
        )
        coord.dispatch(sel)
        nav_adapter.assert_called_once()
        dispatched_sel = nav_adapter.call_args[0][0]
        assert dispatched_sel.mode == "isolate"
        assert dispatched_sel.label_ids == [1, 2, 3]

    def test_navigator_source_skips_navigator_adapter(self):
        """Source='navigator' should NOT echo back to navigator adapter."""
        nav_adapter = MagicMock()
        coord, adapters = _make_coordinator(
            apply_navigator_selection=nav_adapter,
        )
        sel = _make_sel(
            source="navigator", label_ids=[5], mode="isolate",
        )
        coord.dispatch(sel)
        nav_adapter.assert_not_called()

    def test_navigator_source_reaches_plot_highlight(self):
        """Source='navigator' should reach plot highlight adapter."""
        coord, adapters = _make_coordinator()
        sel = _make_sel(
            source="navigator", label_ids=[5, 10], mode="isolate",
        )
        coord.dispatch(sel)
        adapters["apply_plot_highlight"].assert_called_once()
        dispatched_sel = adapters["apply_plot_highlight"].call_args[0][0]
        assert dispatched_sel.label_ids == [5, 10]

    def test_clear_dispatches_to_all_surfaces(self):
        """Empty label_ids should reach viewer_isolation and navigator."""
        nav_adapter = MagicMock()
        coord, adapters = _make_coordinator(
            apply_navigator_selection=nav_adapter,
        )
        sel = SelectionRef(
            scope="single_patch", patch_key="patchA",
            label_ids=[], primary_label_id=None,
            df_indices=[], source="plot", mode="focus",
        )
        coord.dispatch(sel)
        adapters["apply_viewer_isolation"].assert_called_once()
        nav_adapter.assert_called_once()
        # Navigation should be skipped for clear (empty labels)
        adapters["apply_viewer_navigation"].assert_not_called()


# ---------------------------------------------------------------------------
# Phase 1 tests — B1/B3: isolation clear must clean up controller state
# ---------------------------------------------------------------------------


class _FakeUI:
    """Minimal controller UI stand-in for isolation clear tests."""

    def __init__(self):
        self.qc_plot_widget = MagicMock()
        self.qc_plot_widget.highlight_labels = MagicMock()
        self.qc_plot_widget._clear_selection = MagicMock()
        self.sync_viewer_colors_cb = MagicMock()
        self.sync_viewer_colors_cb.isChecked = MagicMock(return_value=False)


class TestAnnotatorIsolationClearedCallback:
    """B1/B3: _on_annotator_isolation_cleared must clear controller
    isolation state so stale _isolated_label_ids don't pollute
    subsequent sync-colors or isolation logic."""

    def _make_controller_stub(self):
        """Create a minimal stub with the real callback method."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
        )
        import types

        ctrl = types.SimpleNamespace()
        ctrl.ui = _FakeUI()
        ctrl._isolated_label_ids = {1, 2, 3}
        ctrl._isolated_df_indices = [0, 1, 2]
        ctrl._colormap_authority = ColormapAuthority()
        ctrl._append_log = lambda msg: None
        ctrl._reset_viewer_label_colors = MagicMock()
        # Step 3 plot pass P3: the one colour path.
        ctrl._apply_plot_colours_to_viewer = MagicMock(return_value=True)
        ctrl._find_active_labels_layer = MagicMock(
            return_value=types.SimpleNamespace(show_selected_label=True)
        )
        ctrl._get_navigator_widget = lambda: None

        ctrl._on_annotator_isolation_cleared = types.MethodType(
            FusionReconstructionController._on_annotator_isolation_cleared,
            ctrl,
        )
        return ctrl

    def test_clears_isolated_label_ids(self):
        """B1: _isolated_label_ids must be empty after annotator clears."""
        ctrl = self._make_controller_stub()
        assert ctrl._isolated_label_ids == {1, 2, 3}

        ctrl._on_annotator_isolation_cleared()

        assert ctrl._isolated_label_ids == set(), (
            "_isolated_label_ids must be cleared when annotator isolation ends"
        )

    def test_clears_isolated_df_indices(self):
        """B1: _isolated_df_indices must be empty after annotator clears."""
        ctrl = self._make_controller_stub()
        assert ctrl._isolated_df_indices == [0, 1, 2]

        ctrl._on_annotator_isolation_cleared()

        assert ctrl._isolated_df_indices == [], (
            "_isolated_df_indices must be cleared when annotator isolation ends"
        )

    def test_clears_plot_highlight(self):
        """Plot highlight should be cleared."""
        ctrl = self._make_controller_stub()
        ctrl._on_annotator_isolation_cleared()
        ctrl.ui.qc_plot_widget.highlight_labels.assert_called_once_with([])

    def test_clears_plot_lasso_selection(self):
        """B3: Plot lasso selection state should also be cleared."""
        ctrl = self._make_controller_stub()
        ctrl._on_annotator_isolation_cleared()
        ctrl.ui.qc_plot_widget._clear_selection.assert_called_once()

    def test_reapplies_colors_after_clear(self):
        """After clearing isolation, viewer colors should be re-applied."""
        ctrl = self._make_controller_stub()
        ctrl.ui.sync_viewer_colors_cb.isChecked.return_value = False
        ctrl._on_annotator_isolation_cleared()
        ctrl._reset_viewer_label_colors.assert_called_once()

    def test_reapplies_the_plots_colours_with_the_tick_on(self):
        """Step 3 plot pass P3: with "Colours in viewer" ticked the clear
        re-applies the plot's colours through the one path (and does not
        reset to napari's colours)."""
        ctrl = self._make_controller_stub()
        ctrl.ui.sync_viewer_colors_cb.isChecked.return_value = True
        ctrl._on_annotator_isolation_cleared()
        ctrl._apply_plot_colours_to_viewer.assert_called_once()
        assert ctrl._isolated_label_ids == set()
        ctrl._reset_viewer_label_colors.assert_not_called()


class TestAnnotatorIsolationClearedAlwaysCleans:
    """_on_annotator_isolation_cleared always runs the full cleanup path,
    including resetting show_selected_label and re-applying colors.
    This handles both lasso-initiated (controller) and checkbox-initiated
    (annotator) isolation."""

    def _make_controller_stub(self, *, isolated_ids=None):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
        )
        import types

        ctrl = types.SimpleNamespace()
        ctrl.ui = _FakeUI()
        ctrl._isolated_label_ids = isolated_ids or set()
        ctrl._isolated_df_indices = []
        ctrl._colormap_authority = ColormapAuthority()
        ctrl._append_log = lambda msg: None
        ctrl._reset_viewer_label_colors = MagicMock()
        # Step 3 plot pass P3: the one colour path.
        ctrl._apply_plot_colours_to_viewer = MagicMock(return_value=True)
        # Fake labels layer with show_selected_label
        ctrl._find_active_labels_layer = MagicMock(
            return_value=types.SimpleNamespace(show_selected_label=True)
        )
        ctrl._get_navigator_widget = lambda: None

        ctrl._on_annotator_isolation_cleared = types.MethodType(
            FusionReconstructionController._on_annotator_isolation_cleared,
            ctrl,
        )
        return ctrl

    def test_always_resets_show_selected_label(self):
        """show_selected_label must be reset even when _isolated_label_ids is empty
        (annotator-initiated isolation doesn't set controller IDs)."""
        ctrl = self._make_controller_stub(isolated_ids=set())
        ctrl._on_annotator_isolation_cleared()

        layer = ctrl._find_active_labels_layer.return_value
        assert layer.show_selected_label is False

    def test_always_reapplies_colors(self):
        """Colors must be re-applied even when _isolated_label_ids was empty."""
        ctrl = self._make_controller_stub(isolated_ids=set())
        ctrl.ui.sync_viewer_colors_cb.isChecked.return_value = False
        ctrl._on_annotator_isolation_cleared()

        ctrl._reset_viewer_label_colors.assert_called_once()

    def test_clears_dirty_state(self):
        """When _isolated_label_ids is non-empty, it gets cleared."""
        ctrl = self._make_controller_stub(isolated_ids={5, 10})
        ctrl._on_annotator_isolation_cleared()

        assert ctrl._isolated_label_ids == set()
        ctrl.ui.qc_plot_widget.highlight_labels.assert_called_once_with([])

    def test_always_clears_plot_highlight(self):
        """Plot highlight is always cleared."""
        ctrl = self._make_controller_stub(isolated_ids=set())
        ctrl._on_annotator_isolation_cleared()

        ctrl.ui.qc_plot_widget.highlight_labels.assert_called_once_with([])
        # Should re-apply colors
        assert (
            ctrl._reset_viewer_label_colors.called
            or ctrl._apply_plot_colours_to_viewer.called
        )


# ---------------------------------------------------------------------------
# Close/relaunch stability — patch switch and close clear all surfaces
# ---------------------------------------------------------------------------


class TestPatchSwitchClearsAllSelectionSurfaces:
    """Patch switch must clear isolation state, plot selection, and navigator
    state BEFORE loading the new patch context."""

    def _make_controller_stub_with_sync(self):
        """Controller stub with coordinator and full selection state."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
            ColormapOwner,
        )
        import types

        ctrl = types.SimpleNamespace()
        ctrl.ui = _FakeUI()
        ctrl._isolated_label_ids = {10, 20, 30}
        ctrl._isolated_df_indices = [0, 1, 2]
        ctrl._colormap_authority = ColormapAuthority()
        ctrl._colormap_authority.claim(ColormapOwner.CONTROLLER_ISOLATION)
        ctrl._append_log = lambda msg: None
        ctrl._reset_viewer_label_colors = MagicMock()
        # Step 3 plot pass P3: the one colour path.
        ctrl._apply_plot_colours_to_viewer = MagicMock(return_value=True)
        ctrl._find_active_labels_layer = MagicMock(return_value=None)
        ctrl._get_navigator_widget = lambda: None

        # Bind _on_annotator_isolation_cleared
        ctrl._on_annotator_isolation_cleared = types.MethodType(
            FusionReconstructionController._on_annotator_isolation_cleared,
            ctrl,
        )
        return ctrl

    def test_patch_switch_clears_all_selection_surfaces_before_load(self):
        """After patch-switch cleanup, all isolation/selection surfaces
        must be clean: controller isolation IDs, colormap authority,
        plot highlight, and plot lasso selection."""
        ctrl = self._make_controller_stub_with_sync()

        # Pre-condition: dirty isolation state
        assert ctrl._isolated_label_ids == {10, 20, 30}
        assert ctrl._isolated_df_indices == [0, 1, 2]

        # The annotator close callback is the unified cleanup path
        ctrl._on_annotator_isolation_cleared()

        # All surfaces must be clean
        assert ctrl._isolated_label_ids == set(), (
            "Isolation label IDs not cleared on patch switch cleanup"
        )
        assert ctrl._isolated_df_indices == [], (
            "Isolation df indices not cleared on patch switch cleanup"
        )
        assert ctrl._colormap_authority.current_owner is None, (
            "Colormap authority must be released after cleanup"
        )
        ctrl.ui.qc_plot_widget.highlight_labels.assert_called_once_with([])
        ctrl.ui.qc_plot_widget._clear_selection.assert_called_once()


class TestCloseAnnotatorClearsPlotState:
    """Annotator close must clear both plot highlight AND lasso selection
    state via the controller callback."""

    def _make_controller_stub(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
        )
        import types

        ctrl = types.SimpleNamespace()
        ctrl.ui = _FakeUI()
        ctrl._isolated_label_ids = set()
        ctrl._isolated_df_indices = []
        ctrl._colormap_authority = ColormapAuthority()
        ctrl._append_log = lambda msg: None
        ctrl._reset_viewer_label_colors = MagicMock()
        # Step 3 plot pass P3: the one colour path.
        ctrl._apply_plot_colours_to_viewer = MagicMock(return_value=True)
        ctrl._find_active_labels_layer = MagicMock(return_value=None)
        ctrl._get_navigator_widget = lambda: None

        ctrl._on_annotator_isolation_cleared = types.MethodType(
            FusionReconstructionController._on_annotator_isolation_cleared,
            ctrl,
        )
        return ctrl

    def test_close_clears_plot_selection_and_highlight(self):
        """On annotator close, both highlight_labels([]) and _clear_selection()
        must be called on the plot widget."""
        ctrl = self._make_controller_stub()

        ctrl._on_annotator_isolation_cleared()

        ctrl.ui.qc_plot_widget.highlight_labels.assert_called_once_with([])
        ctrl.ui.qc_plot_widget._clear_selection.assert_called_once()

    def test_close_clears_plot_even_without_active_isolation(self):
        """Plot must be cleared even when no isolation was active on the
        controller side (annotator-initiated isolation case)."""
        ctrl = self._make_controller_stub()
        # No isolation IDs set
        assert ctrl._isolated_label_ids == set()

        ctrl._on_annotator_isolation_cleared()

        ctrl.ui.qc_plot_widget.highlight_labels.assert_called_once_with([])
        ctrl.ui.qc_plot_widget._clear_selection.assert_called_once()


# ---------------------------------------------------------------------------
# Round 2: Patch switch + close hard-clear must include navigator UI reset
# ---------------------------------------------------------------------------


class TestPatchSwitchClearsNavigatorUI:
    """Patch switch must clear navigator UI state (selection + isolate checkbox)
    in addition to controller isolation state and plot highlight/lasso."""

    def _make_controller_stub_with_navigator(self):
        """Controller stub with a navigator reference via the annotator dock."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
            ColormapOwner,
        )
        import types

        ctrl = types.SimpleNamespace()
        ctrl.ui = _FakeUI()
        ctrl._isolated_label_ids = {10, 20, 30}
        ctrl._isolated_df_indices = [0, 1, 2]
        ctrl._colormap_authority = ColormapAuthority()
        ctrl._colormap_authority.claim(ColormapOwner.CONTROLLER_ISOLATION)
        ctrl._append_log = lambda msg: None
        ctrl._reset_viewer_label_colors = MagicMock()
        # Step 3 plot pass P3: the one colour path.
        ctrl._apply_plot_colours_to_viewer = MagicMock(return_value=True)
        ctrl._find_active_labels_layer = MagicMock(return_value=None)

        # Navigator widget with reset tracking
        nav = MagicMock()
        nav.reset_selection_ui = MagicMock()
        ctrl._navigator_ref = nav

        # Simulate getting navigator via _get_navigator_widget helper
        ctrl._get_navigator_widget = lambda: ctrl._navigator_ref

        ctrl._on_annotator_isolation_cleared = types.MethodType(
            FusionReconstructionController._on_annotator_isolation_cleared,
            ctrl,
        )
        return ctrl, nav

    def test_close_callback_does_not_overclear_navigator_ui(self):
        """_on_annotator_isolation_cleared must NOT call reset_selection_ui.

        R3: Navigator UI reset only happens at lifecycle boundaries via
        _hard_clear_curation_surfaces (patch switch) or the annotator's
        _close_cleanup (close).  The isolation-cleared callback fires on
        every isolate-off toggle — calling reset_selection_ui here would
        wipe the user's current selection context mid-session."""
        ctrl, nav = self._make_controller_stub_with_navigator()

        ctrl._on_annotator_isolation_cleared()

        nav.reset_selection_ui.assert_not_called(), (
            "_on_annotator_isolation_cleared must NOT call navigator "
            "reset_selection_ui — it over-clears during normal use"
        )

    def test_close_callback_clears_without_double_dispatch(self):
        """Close must trigger exactly one clear pathway — not redundant
        clears that cause extra colormap churn."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
        )
        import types

        ctrl = types.SimpleNamespace()
        ctrl.ui = _FakeUI()
        ctrl._isolated_label_ids = {10, 20}
        ctrl._isolated_df_indices = [0, 1]
        ctrl._colormap_authority = ColormapAuthority()
        ctrl._append_log = lambda msg: None
        ctrl._reset_viewer_label_colors = MagicMock()
        # Step 3 plot pass P3: the one colour path.
        ctrl._apply_plot_colours_to_viewer = MagicMock(return_value=True)
        ctrl._find_active_labels_layer = MagicMock(return_value=None)
        ctrl._get_navigator_widget = lambda: None

        ctrl._on_annotator_isolation_cleared = types.MethodType(
            FusionReconstructionController._on_annotator_isolation_cleared,
            ctrl,
        )

        ctrl._on_annotator_isolation_cleared()

        # Plot highlight cleared exactly once
        ctrl.ui.qc_plot_widget.highlight_labels.assert_called_once_with([])
        # Colors re-applied exactly once
        assert ctrl._reset_viewer_label_colors.call_count == 1, (
            f"Colors re-applied {ctrl._reset_viewer_label_colors.call_count} times, "
            f"expected exactly 1"
        )


# ---------------------------------------------------------------------------
# Round 3: Hard-clear helper + isolate-off must not over-clear
# ---------------------------------------------------------------------------


class TestHardClearCurationSurfaces:
    """_hard_clear_curation_surfaces must exist as a unified cleanup helper
    and be called from patch switch to clear all surfaces."""

    def _make_controller_with_hard_clear(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
            ColormapOwner,
        )
        import types

        ctrl = types.SimpleNamespace()
        ctrl.ui = _FakeUI()
        ctrl._isolated_label_ids = {10, 20, 30}
        ctrl._isolated_df_indices = [0, 1, 2]
        ctrl._colormap_authority = ColormapAuthority()
        ctrl._colormap_authority.claim(ColormapOwner.CONTROLLER_ISOLATION)
        ctrl._append_log = lambda msg: None
        ctrl._reset_viewer_label_colors = MagicMock()
        # Step 3 plot pass P3: the one colour path.
        ctrl._apply_plot_colours_to_viewer = MagicMock(return_value=True)
        ctrl._find_active_labels_layer = MagicMock(return_value=None)

        # Navigator with reset tracking
        nav = MagicMock()
        nav.reset_selection_ui = MagicMock()
        ctrl._get_navigator_widget = lambda: nav

        ctrl._hard_clear_curation_surfaces = types.MethodType(
            FusionReconstructionController._hard_clear_curation_surfaces,
            ctrl,
        )
        return ctrl, nav

    def test_switch_patch_runs_hard_clear_all_surfaces_before_load(self):
        """_hard_clear_curation_surfaces must clear isolation IDs, plot
        highlight/lasso, navigator UI state, and reset authority."""
        ctrl, nav = self._make_controller_with_hard_clear()

        ctrl._hard_clear_curation_surfaces()

        # Controller isolation state cleared
        assert ctrl._isolated_label_ids == set()
        assert ctrl._isolated_df_indices == []
        # Authority reset (not just released — full reset for patch switch)
        assert ctrl._colormap_authority.current_owner is None
        # Plot cleared
        ctrl.ui.qc_plot_widget.highlight_labels.assert_called_once_with([])
        ctrl.ui.qc_plot_widget._clear_selection.assert_called_once()
        # Navigator UI reset
        nav.reset_selection_ui.assert_called_once()

    def test_hard_clear_source_present_in_switch_patch(self):
        """_switch_patch_from_3d_annotator must call _hard_clear_curation_surfaces."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        assert "_hard_clear_curation_surfaces" in source, (
            "_switch_patch_from_3d_annotator must call _hard_clear_curation_surfaces"
        )


class TestIsolateOffDoesNotOverclear:
    """Normal isolate-off toggle must NOT clear navigator selection context.

    reset_selection_ui() should only be called at lifecycle boundaries
    (close / patch switch), not on every isolate toggle."""

    def test_isolate_off_does_not_overclear_navigator_selection_context(self):
        """When _on_annotator_isolation_cleared fires during normal use
        (user unchecked Isolate), it must NOT call reset_selection_ui
        on the navigator — that would wipe the user's current selection."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
        )
        import types

        ctrl = types.SimpleNamespace()
        ctrl.ui = _FakeUI()
        ctrl._isolated_label_ids = {5}
        ctrl._isolated_df_indices = [0]
        ctrl._colormap_authority = ColormapAuthority()
        ctrl._append_log = lambda msg: None
        ctrl._reset_viewer_label_colors = MagicMock()
        # Step 3 plot pass P3: the one colour path.
        ctrl._apply_plot_colours_to_viewer = MagicMock(return_value=True)
        ctrl._find_active_labels_layer = MagicMock(return_value=None)

        # Navigator with reset tracking
        nav = MagicMock()
        nav.reset_selection_ui = MagicMock()
        ctrl._get_navigator_widget = lambda: nav

        ctrl._on_annotator_isolation_cleared = types.MethodType(
            FusionReconstructionController._on_annotator_isolation_cleared,
            ctrl,
        )

        ctrl._on_annotator_isolation_cleared()

        nav.reset_selection_ui.assert_not_called(), (
            "Normal isolate-off must NOT call reset_selection_ui — "
            "that over-clears the user's current selection context"
        )

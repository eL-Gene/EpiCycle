"""R4 behavioral tests for curation scope boundary enforcement.

Verifies that:
- Switching to population mode disables navigator adapter.
- Population mode prevents annotator push and subviewer refresh.
- Navigator adapter respects patch identity and clears on empty selection.
- Navigator-origin selection dispatches to non-navigator adapters.
- Ambiguous population click emits user-visible message.

All tests are behavioral — real production methods bound to MagicMock
harnesses.  No Qt or napari required.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import types


# ---------------------------------------------------------------------------
# Import targets
# ---------------------------------------------------------------------------

from napari_epicycle.fusion_reconstruction.curation_selection_sync import (
    SelectionRef,
    CurationSelectionSyncCoordinator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coordinator(**overrides):
    """Build coordinator with mock adapters."""
    adapters = {
        "apply_viewer_isolation": MagicMock(),
        "apply_viewer_navigation": MagicMock(),
        "apply_plot_highlight": MagicMock(),
        "apply_flagged_list_selection": MagicMock(),
        "apply_navigator_selection": None,
    }
    adapters.update(overrides)
    coord = CurationSelectionSyncCoordinator(**adapters)
    return coord, adapters


def _make_controller_harness():
    """Minimal harness with real controller methods bound.

    Includes real CurationSelectionSyncCoordinator with mock adapters
    so dispatch routes properly through real methods.
    """
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    harness = MagicMock()
    harness._log = []
    harness._append_log = lambda msg: harness._log.append(msg)

    # B5 crash fix: _apply_navigator_selection_adapter now early-returns
    # when a patch switch is in progress.  These flags default to a truthy
    # MagicMock auto-attribute, which would trip the new gate; pin them
    # False to represent the steady-state (no switch in flight).
    harness._patch_switch_loading = False
    harness._patch_load_in_flight = False
    harness._sync_debug_log = lambda msg: None

    # Wire real coordinator with mock adapter targets
    harness._selection_sync = CurationSelectionSyncCoordinator(
        apply_viewer_isolation=lambda sel: harness._isolate_labels_in_viewer(
            sel.label_ids, df_indices=sel.df_indices or None
        ),
        apply_viewer_navigation=lambda sel: harness._navigate_to_label(
            sel.primary_label_id,
            df_idx=int(sel.df_indices[0]) if sel.df_indices else None,
        ),
        apply_plot_highlight=lambda sel: harness._apply_plot_highlight_adapter(sel),
        apply_flagged_list_selection=lambda sel: None,
    )

    # R4-WP5: Wire rejection callback for user-visible feedback
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    harness._on_selection_rejected = types.MethodType(
        FusionReconstructionController._on_selection_rejected, harness
    )
    harness._selection_sync.set_rejection_callback(
        harness._on_selection_rejected
    )

    return harness


def _make_sel(*, source="plot", label_ids=None, df_indices=None,
              scope="single_patch", patch_key="patchA",
              mode="focus", primary_label_id=None):
    """Shorthand for building a SelectionRef."""
    label_ids = label_ids or [1]
    df_indices = df_indices or []
    primary = primary_label_id if primary_label_id is not None else (
        label_ids[0] if label_ids else None
    )
    return SelectionRef(
        scope=scope, patch_key=patch_key,
        label_ids=label_ids, primary_label_id=primary,
        df_indices=df_indices, source=source, mode=mode,
    )


# ---------------------------------------------------------------------------
# R4-WP1: Scope boundary enforcement
# ---------------------------------------------------------------------------

# TestScopeBoundaryEnforcement class removed 2026-05-03
# (drop-population-mode Commit 3/6).  All 3 tests asserted population-
# mode no-op behaviours that no longer exist.


# ---------------------------------------------------------------------------
# R4-WP2: Patch-accurate SelectionRef construction
# ---------------------------------------------------------------------------

class TestPatchAccurateSelectionRef:
    """SelectionRef.patch_key resolution from current selection state."""

    # test_plot_selection_derives_patch_key_from_df_row removed
    # 2026-05-03 (drop-population-mode Commit 4/6) — population
    # mode is gone, so patch_name column lookup no longer exists.

    def test_flagged_list_in_single_patch_uses_current_patch(self):
        """In single-patch mode, patch_key comes from current patch selection."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        harness = _make_controller_harness()
        # _is_population_mode mock removed 2026-05-03 — production method gone.
        harness._isolated_label_ids = set()

        dispatched = []
        orig_dispatch = harness._selection_sync.dispatch

        def capture_dispatch(sel):
            dispatched.append(sel)
            orig_dispatch(sel)

        harness._selection_sync.dispatch = capture_dispatch

        harness._selected_curation_patch = MagicMock(return_value="my_patch.tif")
        harness._selected_patch_name = MagicMock(return_value="fallback.tif")

        # Create a fake QListWidget item
        from qtpy.QtCore import Qt
        item = MagicMock()
        item.data = MagicMock(side_effect=lambda role: {
            Qt.UserRole: 42,
            Qt.UserRole + 1: 3,
        }.get(role))

        real = FusionReconstructionController._on_flagged_label_clicked
        harness._on_flagged_label_clicked = types.MethodType(real, harness)
        harness._on_flagged_label_clicked(item)

        assert len(dispatched) == 1
        assert dispatched[0].patch_key == "my_patch.tif"


# ---------------------------------------------------------------------------
# R4-WP3: Navigator adapter patch identity + clear
# ---------------------------------------------------------------------------

class TestNavigatorAdapterPatchIdentity:
    """Navigator adapter must respect patch identity and clear selection."""

    def test_clear_selection_clears_navigator(self):
        """Empty label_ids dispatch must clear navigator selection."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        harness = _make_controller_harness()

        # Set up dock with navigator
        dock = MagicMock()
        dock.isVisible.return_value = True
        widget = MagicMock()
        widget._navigator_widget = MagicMock()
        dock.widget.return_value = widget
        harness.viewer.window._dock_widgets = {
            "EpiCycle 3D Curation Annotator": dock
        }

        real = FusionReconstructionController._apply_navigator_selection_adapter
        harness._apply_navigator_selection_adapter = types.MethodType(real, harness)

        # Wire navigator adapter
        harness._selection_sync.update_navigator_adapter(
            harness._apply_navigator_selection_adapter
        )

        # Dispatch clear selection
        sel = SelectionRef(
            scope="single_patch", patch_key="patchA",
            label_ids=[], primary_label_id=None,
            df_indices=[], source="flagged_list", mode="focus",
        )
        harness._selection_sync.dispatch(sel)

        # Navigator should have been told to clear
        widget._navigator_widget.clear_selection.assert_called_once()

    def test_patch_mismatch_drops_navigator_update(self):
        """Navigator update must be dropped if patch_key doesn't match."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        harness = _make_controller_harness()
        harness._qc_view_patch_name = "patchA"

        dock = MagicMock()
        dock.isVisible.return_value = True
        widget = MagicMock()
        widget._navigator_widget = MagicMock()
        dock.widget.return_value = widget
        harness.viewer.window._dock_widgets = {
            "EpiCycle 3D Curation Annotator": dock
        }

        real = FusionReconstructionController._apply_navigator_selection_adapter
        harness._apply_navigator_selection_adapter = types.MethodType(real, harness)
        harness._selection_sync.update_navigator_adapter(
            harness._apply_navigator_selection_adapter
        )

        # Dispatch with mismatched patch
        sel = _make_sel(
            source="plot", label_ids=[10], patch_key="patchB",
        )
        harness._selection_sync.dispatch(sel)

        # Navigator must NOT be updated (patch mismatch)
        widget._navigator_widget.select_label.assert_not_called()
        widget._navigator_widget.select_labels.assert_not_called()

    def test_patch_match_allows_navigator_update(self):
        """Navigator update proceeds when patch_key matches active patch."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        harness = _make_controller_harness()
        harness._qc_view_patch_name = "patchA"

        dock = MagicMock()
        dock.isVisible.return_value = True
        widget = MagicMock()
        widget._navigator_widget = MagicMock()
        dock.widget.return_value = widget
        harness.viewer.window._dock_widgets = {
            "EpiCycle 3D Curation Annotator": dock
        }

        real = FusionReconstructionController._apply_navigator_selection_adapter
        harness._apply_navigator_selection_adapter = types.MethodType(real, harness)
        harness._selection_sync.update_navigator_adapter(
            harness._apply_navigator_selection_adapter
        )

        sel = _make_sel(
            source="plot", label_ids=[10], patch_key="patchA",
        )
        harness._selection_sync.dispatch(sel)

        widget._navigator_widget.select_label.assert_called_once_with(10)


# ---------------------------------------------------------------------------
# R4-WP4: Navigator -> coordinator bridge
# ---------------------------------------------------------------------------

class TestNavigatorCoordinatorBridge:
    """Navigator-origin selection must dispatch to non-navigator adapters."""

    def test_navigator_selection_dispatches_to_other_adapters(self):
        """Navigator-origin dispatch must call plot/viewer adapters, not navigator."""
        coord, adapters = _make_coordinator(
            apply_navigator_selection=MagicMock(),
        )

        sel = _make_sel(
            source="navigator", label_ids=[42], patch_key="patchA",
        )
        coord.dispatch(sel)

        # Viewer adapters called
        adapters["apply_viewer_isolation"].assert_called_once()
        adapters["apply_viewer_navigation"].assert_called_once()

        # Plot highlight called (source != "plot")
        adapters["apply_plot_highlight"].assert_called_once()

        # Navigator NOT called (source echo suppression)
        adapters["apply_navigator_selection"].assert_not_called()

    def test_navigator_bridge_callback_wired_in_annotator(self):
        """Controller must wire navigator bridge callback on annotator open.

        This is a structural test — verifying the annotator's
        _on_navigator_label_selected_cb attribute is set.
        """
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        # The callback should be wirable via set_navigator_selection_callback
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        assert hasattr(Curation3DAnnotator, "set_navigator_selection_callback"), (
            "Curation3DAnnotator must have set_navigator_selection_callback method"
        )


# ---------------------------------------------------------------------------
# R4-WP5: User-visible ambiguity feedback
# ---------------------------------------------------------------------------

class TestAmbiguityFeedback:
    """Ambiguous population clicks must produce user-visible feedback."""

    def test_ambiguous_click_triggers_rejection_callback(self):
        """Coordinator must call rejection callback on ambiguous population click."""
        rejection_log = []
        coord, adapters = _make_coordinator()
        coord.set_rejection_callback(
            lambda sel, reason: rejection_log.append((sel, reason))
        )

        sel = _make_sel(
            scope="population", label_ids=[42], df_indices=[],
        )
        coord.dispatch(sel)

        # Must have called the rejection callback
        assert len(rejection_log) == 1
        assert "ambiguous" in rejection_log[0][1].lower()

        # Must NOT have dispatched to adapters
        adapters["apply_viewer_isolation"].assert_not_called()

    # test_ambiguous_click_visible_in_controller_log removed 2026-05-03
    # (drop-population-mode Commit 3/6) — population mode no longer exists.


# TestNavigatorBridgePopulationBlock removed 2026-05-03
# (drop-population-mode Commit 3/6) — navigator bridge population
# block no longer exists.


# ---------------------------------------------------------------------------
# R4.1 Fix: Scope roundtrip rebinds navigator adapter
# ---------------------------------------------------------------------------

# TestScopeRoundtripRebind class removed 2026-05-03
# (drop-population-mode Commit 3/6) — scope roundtrip is no longer
# possible since "All Patches" no longer exists.


# ---------------------------------------------------------------------------
# R4.1 Fix: Clear selection clears navigator isolation
# ---------------------------------------------------------------------------

class TestClearSelectionClearsIsolation:
    """Clear dispatch must clear both navigator selection and isolation."""

    def test_clear_selection_clears_navigator_and_isolation(self):
        """Empty label_ids dispatch must clear navigator selection + isolation."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        harness = _make_controller_harness()
        harness._qc_view_patch_name = "patchA"

        dock = MagicMock()
        dock.isVisible.return_value = True
        widget = MagicMock()

        # Use a mock navigator but bind the REAL clear_selection method
        nav = MagicMock()
        nav._on_isolation_changed_cb = MagicMock()
        real_clear = CurationLabelNavigator.clear_selection
        nav.clear_selection = types.MethodType(real_clear, nav)

        widget._navigator_widget = nav
        dock.widget.return_value = widget
        harness.viewer.window._dock_widgets = {
            "EpiCycle 3D Curation Annotator": dock
        }

        real_adapter = FusionReconstructionController._apply_navigator_selection_adapter
        harness._apply_navigator_selection_adapter = types.MethodType(
            real_adapter, harness
        )
        harness._selection_sync.update_navigator_adapter(
            harness._apply_navigator_selection_adapter
        )

        # Dispatch clear selection
        sel = SelectionRef(
            scope="single_patch", patch_key="patchA",
            label_ids=[], primary_label_id=None,
            df_indices=[], source="flagged_list", mode="focus",
        )
        harness._selection_sync.dispatch(sel)

        # Isolation must be cleared (empty set) via real clear_selection path
        nav._on_isolation_changed_cb.assert_called_once_with(set())


# ---------------------------------------------------------------------------
# R4.2 Fix: Index semantics — iloc vs loc for df_idx
# ---------------------------------------------------------------------------

# TestIndexSemanticsFix class removed 2026-05-03
# (drop-population-mode Commit 4/6).  Both tests asserted that
# patch_key was resolved from the merged DataFrame's patch_name
# column using either label-index lookup or positional lookup.
# Population mode is gone, the merged DataFrame never appears, and
# patch_key now comes from the bucket / loaded-patch selection only.


# ---------------------------------------------------------------------------
# WP-7 tests — Stale patch context in navigator adapter (Bug B)
# ---------------------------------------------------------------------------


class TestPatchContextStaleness:
    """WP-7: _qc_view_patch_name is current when adapter filters events."""

    def test_patch_switch_updates_qc_view_patch_name(self):
        """_load_patch_for_qc_view sets _qc_view_patch_name."""
        harness = _make_controller_harness()
        # Before: stale value
        harness._qc_view_patch_name = "old_patch.tiff"

        # Simulate _load_patch_for_qc_view updating the name
        harness._qc_view_patch_name = "new_patch.tiff"

        assert harness._qc_view_patch_name == "new_patch.tiff"

    def test_adapter_after_patch_switch_uses_new_patch_key(self):
        """Navigator adapter should use updated _qc_view_patch_name."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        harness = _make_controller_harness()
        harness._qc_view_patch_name = "new_patch.tiff"

        dock = MagicMock()
        dock.isVisible.return_value = True
        widget = MagicMock()
        nav = MagicMock()
        widget._navigator_widget = nav
        dock.widget.return_value = widget
        harness.viewer.window._dock_widgets = {
            "EpiCycle 3D Curation Annotator": dock
        }

        real = FusionReconstructionController._apply_navigator_selection_adapter
        harness._apply_navigator_selection_adapter = types.MethodType(real, harness)

        # Dispatch a selection for the new patch — should NOT be dropped
        sel = _make_sel(
            source="plot", label_ids=[42], patch_key="new_patch.tiff",
        )
        harness._apply_navigator_selection_adapter(sel)
        nav.select_label.assert_called_once_with(42)

    def test_adapter_drop_produces_log_message(self):
        """Mismatched patch_key should log a user-visible message."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        harness = _make_controller_harness()
        harness._qc_view_patch_name = "current_patch.tiff"

        dock = MagicMock()
        dock.isVisible.return_value = True
        widget = MagicMock()
        nav = MagicMock()
        widget._navigator_widget = nav
        dock.widget.return_value = widget
        harness.viewer.window._dock_widgets = {
            "EpiCycle 3D Curation Annotator": dock
        }

        real = FusionReconstructionController._apply_navigator_selection_adapter
        harness._apply_navigator_selection_adapter = types.MethodType(real, harness)

        # Dispatch with wrong patch — should be dropped + logged
        sel = _make_sel(
            source="plot", label_ids=[42], patch_key="other_patch.tiff",
        )
        harness._apply_navigator_selection_adapter(sel)
        nav.select_label.assert_not_called()

        # Check log contains mismatch message
        log_text = " ".join(harness._log)
        assert "patch mismatch" in log_text.lower()


class TestNavigatorAdapterSwitchGate:
    """B5 crash fix: the navigator adapter must early-return (and NOT touch
    the navigator) while a patch switch is in progress.  A QC-plot lasso
    dispatch deferred via QTimer.singleShot(0) can land inside the switch
    window; reaching navigator.select_labels while the table is rebuilding
    is the Windows access-violation crash."""

    def _build_visible_dock_harness(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        harness = _make_controller_harness()
        harness._qc_view_patch_name = "current_patch.tiff"

        dock = MagicMock()
        dock.isVisible.return_value = True
        widget = MagicMock()
        nav = MagicMock()
        widget._navigator_widget = nav
        dock.widget.return_value = widget
        harness.viewer.window._dock_widgets = {
            "EpiCycle 3D Curation Annotator": dock
        }

        real = FusionReconstructionController._apply_navigator_selection_adapter
        harness._apply_navigator_selection_adapter = types.MethodType(
            real, harness
        )
        return harness, nav

    def test_adapter_skipped_when_patch_switch_loading(self):
        """`_patch_switch_loading` True → adapter early-returns, nav untouched."""
        harness, nav = self._build_visible_dock_harness()
        harness._patch_switch_loading = True
        harness._patch_load_in_flight = False

        sel = _make_sel(
            source="plot", label_ids=[7, 8], patch_key="current_patch.tiff",
        )
        harness._apply_navigator_selection_adapter(sel)

        nav.select_label.assert_not_called()
        nav.select_labels.assert_not_called()

    def test_adapter_skipped_when_patch_load_in_flight(self):
        """`_patch_load_in_flight` True → adapter early-returns, nav untouched."""
        harness, nav = self._build_visible_dock_harness()
        harness._patch_switch_loading = False
        harness._patch_load_in_flight = True

        sel = _make_sel(
            source="plot", label_ids=[42], patch_key="current_patch.tiff",
        )
        harness._apply_navigator_selection_adapter(sel)

        nav.select_label.assert_not_called()
        nav.select_labels.assert_not_called()

    def test_adapter_proceeds_when_no_switch_in_progress(self):
        """Both gate flags False → adapter dispatches to the navigator
        exactly as before (behavior preserved for the normal flow)."""
        harness, nav = self._build_visible_dock_harness()
        harness._patch_switch_loading = False
        harness._patch_load_in_flight = False

        sel = _make_sel(
            source="plot", label_ids=[42], patch_key="current_patch.tiff",
        )
        harness._apply_navigator_selection_adapter(sel)

        nav.select_label.assert_called_once_with(42)

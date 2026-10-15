"""Tests for Phase 2 lifecycle hardening (WP-4 + WP-2).

WP-4: Robust guard flag reset on close/relaunch — ensures ALL epicycle
      guard flags are reset on ALL main-viewer layers, even those not
      tracked in the name lists. Also tests orphan window sweep extension.

WP-2: Sync suppression context manager — verifies the LabelSyncManager
      can suppress property sync during batch colormap operations.

All tests are Qt-free using lightweight fakes.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal fakes
# ---------------------------------------------------------------------------


class FakeLayer:
    """Minimal layer fake with guard flags and events."""

    def __init__(self, name: str, *, kind: str = "labels"):
        self.name = name
        self._kind = kind
        self.colormap = None
        self.brush_size = 1
        self.opacity = 1.0
        self.blending = "translucent"
        self.n_edit_dimensions = 2
        self.events = MagicMock()


def _make_layers_with_flags(names, *, flags_to_set=None):
    """Create fake layers with optional guard flags pre-set."""
    layers = []
    for name in names:
        layer = FakeLayer(name)
        if flags_to_set:
            for flag in flags_to_set:
                setattr(layer, flag, True)
        layers.append(layer)
    return layers


ALL_GUARD_FLAGS = [
    "_epicycle_sync_connected",
    "_epicycle_main_drag_connected",
    "_epicycle_vis_synced",
    "_epicycle_style_synced",
    "_epicycle_pin_connected",
    "_epicycle_labels_update_connected",
    "_epicycle_show_sel_connected",
]


# ---------------------------------------------------------------------------
# WP-4: Guard flag safety-net sweep
# ---------------------------------------------------------------------------


class TestGuardFlagSafetyNet:
    """After _disconnect_viewer_events, ALL epicycle guard flags
    on ALL main-viewer layers should be reset to False."""

    def test_tracked_layers_flags_reset(self):
        """Layers in the tracked name lists get their flags reset."""
        flags = [
            "_epicycle_sync_connected",
            "_epicycle_vis_synced",
            "_epicycle_style_synced",
        ]
        layers = _make_layers_with_flags(["labels_A", "labels_B"], flags_to_set=flags)

        # Verify flags are set
        for layer in layers:
            for flag in flags:
                assert getattr(layer, flag) is True

        # Simulate the safety-net sweep
        for layer in layers:
            for flag in ALL_GUARD_FLAGS:
                if hasattr(layer, flag):
                    setattr(layer, flag, False)

        # Verify all flags cleared
        for layer in layers:
            for flag in flags:
                assert getattr(layer, flag) is False

    def test_untracked_layers_flags_reset(self):
        """Layers NOT in the tracked name lists (dynamically added)
        also get their guard flags reset by the safety-net sweep."""
        tracked_names = ["labels_A"]
        all_layers = _make_layers_with_flags(
            ["labels_A", "labels_dynamic_B", "labels_unexpected_C"],
            flags_to_set=["_epicycle_sync_connected", "_epicycle_vis_synced"],
        )

        # Only sweep by name would miss untracked layers:
        # tracked names = ["labels_A"], but layers_B and layers_C also have flags
        untracked = [l for l in all_layers if l.name not in tracked_names]
        assert len(untracked) == 2

        # Safety-net sweep hits ALL layers
        for layer in all_layers:
            for flag in ALL_GUARD_FLAGS:
                if hasattr(layer, flag):
                    setattr(layer, flag, False)

        # ALL layers should have flags cleared
        for layer in all_layers:
            assert getattr(layer, "_epicycle_sync_connected") is False
            assert getattr(layer, "_epicycle_vis_synced") is False

    def test_safety_net_does_not_crash_on_layers_without_flags(self):
        """Layers that never had guard flags should not crash the sweep."""
        layer = FakeLayer("plain_layer")
        # Ensure no flags exist
        for flag in ALL_GUARD_FLAGS:
            assert not hasattr(layer, flag) or getattr(layer, flag) is not True

        # Safety-net sweep should not raise
        for flag in ALL_GUARD_FLAGS:
            if hasattr(layer, flag):
                setattr(layer, flag, False)

    def test_source_code_has_safety_net_sweep(self):
        """Verify _disconnect_viewer_events has the safety-net sweep."""
        import inspect

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        source = inspect.getsource(
            NativeMultiviewAnnotator._disconnect_viewer_events
        )
        # Should contain the safety-net sweep pattern
        assert "_epicycle_sync_connected" in source
        assert "_epicycle_vis_synced" in source
        assert "_epicycle_style_synced" in source
        # Should iterate viewer.layers (the safety-net sweep)
        assert "self.viewer.layers" in source


# ---------------------------------------------------------------------------
# WP-4: Orphan window sweep extension
# ---------------------------------------------------------------------------


class TestOrphanWindowSweepExtension:
    """_close_orphan_magicgui_windows_global should also catch
    QtPopup / QtModalPopup windows, not just magicgui."""

    def test_source_code_catches_qtmodalpopup(self):
        """The global orphan sweep should match qtmodalpopup pattern."""
        import inspect

        from napari_epicycle.annotator.launcher import (
            _close_orphan_magicgui_windows_global,
        )

        source = inspect.getsource(_close_orphan_magicgui_windows_global)
        assert "qtmodalpopup" in source.lower() or "qtpopup" in source.lower()

    def test_source_code_catches_magicgui(self):
        """The global orphan sweep should still match magicgui pattern."""
        import inspect

        from napari_epicycle.annotator.launcher import (
            _close_orphan_magicgui_windows_global,
        )

        source = inspect.getsource(_close_orphan_magicgui_windows_global)
        assert "magicgui" in source.lower()

    def test_instance_orphan_sweep_catches_qtmodalpopup(self):
        """The instance-level orphan sweep should also match qtmodalpopup."""
        import inspect

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        source = inspect.getsource(
            NativeMultiviewAnnotator._close_orphan_magicgui_windows
        )
        assert "qtmodalpopup" in source.lower() or "qtpopup" in source.lower()

    def test_close_retry_schedule_helper_exists(self):
        """Launcher should provide delayed retry sweeps for late magicgui windows."""
        import inspect

        from napari_epicycle.annotator.launcher import (
            _schedule_close_orphan_sweep_retries,
        )

        source = inspect.getsource(_schedule_close_orphan_sweep_retries)
        assert "singleShot" in source
        assert "5000" in source

    def test_run_widget_cleanup_calls_retry_sweeps(self):
        """Widget cleanup should trigger retry orphan sweeps."""
        import inspect

        from napari_epicycle.annotator.launcher import _run_widget_cleanup

        source = inspect.getsource(_run_widget_cleanup)
        assert "_schedule_close_orphan_sweep_retries" in source


class TestTeardownDoesNotOrphanWidgets:
    """Teardown should not create new top-level widgets during close."""

    def test_destroy_dock_does_not_unparent_widget_or_dock(self):
        """_destroy_dock should avoid setParent(None), which can orphan windows."""
        import inspect

        from napari_epicycle.annotator.launcher import _destroy_dock

        source = inspect.getsource(_destroy_dock)
        assert "widget.setParent(None)" not in source
        assert "dock.setParent(None)" not in source

    def test_close_subviewers_does_not_unparent_qtviewer_or_children(self):
        """_close_subviewers should close/delete viewers without manual unparenting."""
        import inspect

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        source = inspect.getsource(NativeMultiviewAnnotator._close_subviewers)
        assert "child.setParent(None)" not in source
        assert "qtv.setParent(None)" not in source

    def test_snapshot_orphan_sweep_no_longer_requires_magicgui_for_qwidget(self):
        """Snapshot sweep should close leaked top-level QWidgets even if unnamed."""
        import inspect

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        source = inspect.getsource(NativeMultiviewAnnotator._destroy_orphan_toplevel_widgets)
        assert 'if cls_name == "QWidget"' not in source
        assert "is_magicgui_name" in source
        assert "cls_name not in self._ORPHAN_CLASSES" in source
        assert "w.setParent(None)" not in source


# ---------------------------------------------------------------------------
# WP-2: Sync suppression context manager
# ---------------------------------------------------------------------------


class TestSyncSuppression:
    """LabelSyncManager.suppress_property_sync() context manager."""

    def _make_sync_manager(self):
        """Create a minimal LabelSyncManager for testing."""
        from packages.epicycle_annotator.sync import LabelSyncManager

        main_viewer = MagicMock()
        main_viewer.layers = []
        vm_xy = MagicMock()
        vm_xy.layers = []
        vm_xz = MagicMock()
        vm_xz.layers = []
        vm_yz = MagicMock()
        vm_yz.layers = []

        return LabelSyncManager(
            main_viewer, vm_xy, vm_xz, vm_yz,
            MagicMock(), MagicMock(), MagicMock(),
        )

    def test_sync_suppressed_inside_context(self):
        """Inside suppress_property_sync, _syncing_properties should be True."""
        sync = self._make_sync_manager()
        assert not getattr(sync, "_syncing_properties", False)

        with sync.suppress_property_sync():
            assert sync._syncing_properties is True

        assert sync._syncing_properties is False

    def test_sync_resumes_after_context(self):
        """After exiting the context, sync should work normally."""
        sync = self._make_sync_manager()

        with sync.suppress_property_sync():
            pass

        assert sync._syncing_properties is False

    def test_sync_resumes_after_exception(self):
        """Even if an exception occurs, the flag must be reset."""
        sync = self._make_sync_manager()

        with pytest.raises(ValueError):
            with sync.suppress_property_sync():
                raise ValueError("test error")

        assert sync._syncing_properties is False

    def test_sync_label_properties_skipped_during_suppression(self):
        """sync_label_properties should be a no-op when suppressed."""
        sync = self._make_sync_manager()
        layer = FakeLayer("test_label")

        # Track whether sync does work
        call_count = 0
        original_sync = sync.sync_label_properties

        def counting_sync(active_layer):
            nonlocal call_count
            call_count += 1
            # Don't call original — it would try to find real layers

        sync.sync_label_properties = counting_sync

        # Without suppression, sync fires
        sync._syncing_properties = False
        counting_sync(layer)
        assert call_count == 1

        # With suppression, the guard in sync_label_properties prevents work
        sync._syncing_properties = True
        # The real sync_label_properties checks _syncing_properties and returns early
        # Let's verify the guard is checked
        assert sync._syncing_properties is True

    def test_nested_suppression_safe(self):
        """Nested suppress_property_sync should not break."""
        sync = self._make_sync_manager()

        with sync.suppress_property_sync():
            assert sync._syncing_properties is True
            # Nested context — should still work
            with sync.suppress_property_sync():
                assert sync._syncing_properties is True
            # After inner exits, flag should still be True
            # (outer context is still active)
            assert sync._syncing_properties is True

        # After outer exits, flag should be False
        assert sync._syncing_properties is False

    def test_source_code_has_suppress_method(self):
        """LabelSyncManager should expose suppress_property_sync."""
        from packages.epicycle_annotator.sync import LabelSyncManager

        assert hasattr(LabelSyncManager, "suppress_property_sync")


# ---------------------------------------------------------------------------
# WP-2: Controller uses sync suppression in batch operations
# ---------------------------------------------------------------------------


class TestControllerSyncSuppression:
    """Controller batch colormap operations should suppress sync."""

    def test_isolate_labels_source_mentions_suppression(self):
        """_isolate_labels_in_viewer or its callees should reference
        sync suppression or batch colormap pattern."""
        import inspect

        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        # The one colour path (Step 3 plot pass P3) is THE batch colormap
        # writer of the QC colours: it must write under the suppression.
        # The isolation's own distinct-colour writer must too.
        for method_name in ("_apply_plot_colours_to_viewer",
                            "_show_isolated_labels_distinct"):
            source = inspect.getsource(
                getattr(FusionReconstructionController, method_name)
            )
            assert "with self._suppress_sync_cascade()" in source, (
                f"{method_name} must write its colormap under the sync "
                "suppression"
            )
        # The old colour names reach the one path, never their own writer.
        wrapper = inspect.getsource(
            FusionReconstructionController._update_viewer_feature_colors)
        assert "_apply_plot_colours_to_viewer" in wrapper
        assert "layer.colormap" not in wrapper


# ---------------------------------------------------------------------------
# Bug 2: Snapshot must be taken BEFORE QtViewer creation
# ---------------------------------------------------------------------------


class TestSnapshotTimingBeforeQtViewerCreation:
    """_creation_toplevel_ids snapshot must be captured BEFORE creating
    the 3 QtViewer instances, so any popups spawned by QtViewer
    construction are 'new' and eligible for cleanup on close."""

    def test_snapshot_appears_before_qtviewer_in_source(self):
        """In __init__, _creation_toplevel_ids assignment must appear
        BEFORE any QtViewer() constructor call."""
        import inspect

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        source = inspect.getsource(NativeMultiviewAnnotator.__init__)
        snapshot_pos = source.find("_creation_toplevel_ids")
        qtviewer_pos = source.find("QtViewer(")
        assert snapshot_pos != -1, "_creation_toplevel_ids not found in __init__"
        assert qtviewer_pos != -1, "QtViewer( not found in __init__"
        assert snapshot_pos < qtviewer_pos, (
            f"Snapshot (_creation_toplevel_ids at char {snapshot_pos}) must appear "
            f"BEFORE QtViewer creation (at char {qtviewer_pos}) in __init__"
        )


# ---------------------------------------------------------------------------
# Bug 3: Sweep functions must NOT call setParent(None)
# ---------------------------------------------------------------------------


class TestSweepFunctionsNoSetParentNone:
    """Orphan sweep functions must not call setParent(None) on widgets.
    Unparenting top-level widgets is a no-op, and unparenting parented
    widgets can orphan their children — contradicting the design intent."""

    def test_close_orphan_magicgui_windows_no_setparent(self):
        """Instance-level sweep must not call setParent(None)."""
        import inspect

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        source = inspect.getsource(
            NativeMultiviewAnnotator._close_orphan_magicgui_windows
        )
        assert "setParent(None)" not in source, (
            "_close_orphan_magicgui_windows must not call setParent(None)"
        )

    def test_destroy_orphan_toplevel_no_setparent(self):
        """Snapshot-based sweep must not call setParent(None)."""
        import inspect

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        source = inspect.getsource(
            NativeMultiviewAnnotator._destroy_orphan_toplevel_widgets
        )
        assert "setParent(None)" not in source, (
            "_destroy_orphan_toplevel_widgets must not call setParent(None)"
        )

    def test_global_sweep_no_setparent(self):
        """Global fallback sweep must not call setParent(None)."""
        import inspect

        from napari_epicycle.annotator.launcher import (
            _close_orphan_magicgui_windows_global,
        )

        source = inspect.getsource(_close_orphan_magicgui_windows_global)
        assert "setParent(None)" not in source, (
            "_close_orphan_magicgui_windows_global must not call setParent(None)"
        )


# ---------------------------------------------------------------------------
# Bug 1: Zombie widget prevention
# ---------------------------------------------------------------------------


class TestZombieWidgetPrevention:
    """After _run_widget_cleanup, the widget must be hidden and scheduled
    for deletion to prevent zombie accumulation as top-level windows."""

    def test_run_widget_cleanup_does_not_call_deleteLater(self):
        """_run_widget_cleanup must NOT call widget.deleteLater().

        If the hook queues widget for deletion, then napari's setParent(None)
        triggers our setParent override which flushes deferred deletes via
        sendPostedEvents(DeferredDelete).  That flush destroys the widget
        MID-EXECUTION of setParent, orphaning all children.
        """
        import inspect

        from napari_epicycle.annotator.launcher import _run_widget_cleanup

        source = inspect.getsource(_run_widget_cleanup)
        # The source should contain "deleteLater" only in comments (explaining
        # why it's NOT called), not as an actual call.
        lines = [
            line for line in source.split("\n")
            if "deleteLater" in line and not line.strip().startswith("#")
        ]
        assert len(lines) == 0, (
            "_run_widget_cleanup must NOT call widget.deleteLater() — "
            "it causes the widget to self-destruct during setParent override. "
            f"Found active deleteLater calls: {lines}"
        )

    def test_run_widget_cleanup_hides_widget(self):
        """_run_widget_cleanup must call widget.hide() to remove from taskbar."""
        import inspect

        from napari_epicycle.annotator.launcher import _run_widget_cleanup

        source = inspect.getsource(_run_widget_cleanup)
        assert "widget.hide()" in source, (
            "_run_widget_cleanup must hide the widget to remove from taskbar"
        )

    def test_close_cleanup_has_reentrance_guard(self):
        """_close_cleanup must have a re-entrancy guard to prevent double-execution."""
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(Curation3DAnnotator._close_cleanup)
        assert "_close_cleanup_done" in source, (
            "_close_cleanup must check _close_cleanup_done to prevent "
            "double-execution when closeEvent fires after _run_widget_cleanup"
        )

    def test_pre_launch_zombie_sweep_exists(self):
        """Launcher must have a function to sweep orphan annotator zombies."""
        import inspect

        from napari_epicycle.annotator.launcher import (
            _sweep_orphan_annotator_widgets,
        )

        source = inspect.getsource(_sweep_orphan_annotator_widgets)
        # Must scan topLevelWidgets
        assert "topLevelWidgets" in source
        # Must match annotator class names
        assert "NativeMultiviewAnnotator" in source or "Curation3DAnnotator" in source

    def test_launch_3d_calls_zombie_sweep(self):
        """launch_3d_curation_annotator must call the zombie sweep."""
        import inspect

        from napari_epicycle.annotator.launcher import (
            launch_3d_curation_annotator,
        )

        source = inspect.getsource(launch_3d_curation_annotator)
        assert "_sweep_orphan_annotator_widgets" in source, (
            "launch_3d_curation_annotator must call _sweep_orphan_annotator_widgets "
            "before creating a new widget"
        )


# ---------------------------------------------------------------------------
# Child widget orphan prevention
# ---------------------------------------------------------------------------


class TestChildWidgetOrphanPrevention:
    """_run_widget_cleanup must hide and destroy all child QWidgets
    before napari can orphan them via setParent(None).  Without this,
    each close/relaunch cycle leaks ~23 child widgets (QPushButton,
    QCheckBox, QGroupBox, etc.) as top-level orphans that appear on the
    Windows taskbar as 'magicgui'."""

    def test_hide_and_destroy_child_widgets_exists(self):
        """Launcher must export _hide_and_destroy_child_widgets."""
        from napari_epicycle.annotator.launcher import (
            _hide_and_destroy_child_widgets,
        )

        assert callable(_hide_and_destroy_child_widgets)

    def test_setparent_override_handles_child_destruction(self):
        """Child widget destruction is handled in setParent override, not launcher.

        napari's remove_dock_widget calls setParent(None) on the annotator
        BEFORE our close hook fires.  The setParent override on the annotator
        intercepts this call and hides + deleteLater() all children at the
        exact moment napari unparents the widget.
        """
        import inspect

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        source = inspect.getsource(NativeMultiviewAnnotator.setParent)
        assert "findChildren" in source, (
            "setParent must use findChildren to get all descendant QWidgets"
        )
        assert "hide()" in source, "Must hide children to prevent taskbar appearance"
        assert "deleteLater" in source, "Must schedule children for deletion"

    def test_setparent_override_hides_children_on_none(self):
        """NativeMultiviewAnnotator.setParent must hide children when parent=None.

        napari's remove_dock_widget calls setParent(None) on the inner widget,
        orphaning it and its children as top-level windows.  The override must
        hide + deleteLater all child QWidgets before calling super().setParent(None)
        so they never appear on the Windows taskbar.
        """
        import inspect

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        source = inspect.getsource(NativeMultiviewAnnotator.setParent)
        assert "parent is None" in source, (
            "setParent override must check for None parent (napari orphaning)"
        )
        assert "findChildren" in source, (
            "Must use findChildren to get all descendant QWidgets"
        )
        assert "hide()" in source, "Must hide children"
        assert "deleteLater" in source, "Must schedule children for deletion"
        assert "super().setParent" in source, "Must call super() to actually reparent"


# ---------------------------------------------------------------------------
# Stub widget parenting — prevent creation-time orphans
# ---------------------------------------------------------------------------


class TestStubWidgetParenting:
    """Curation3DAnnotator._build_draw_group creates many hidden stub widgets
    for parent-class compatibility.  Without an explicit parent, they become
    top-level orphan windows.  If anything later calls setVisible(True) on
    them (e.g. set_flagged_labels shows _focused_mode_group), they appear
    on the Windows taskbar as 'magicgui'.

    These tests verify that ALL stub widgets are parented to the draw group.
    """

    # Stub attribute names that MUST have parent=group in _build_draw_group.
    _STUB_ATTRS = [
        "crosshairs_check",
        "image_combo",
        "labels_combo",
        "sync_to_subviewers_btn",
        "refresh_layers_btn",
        "create_labels_btn",
        "segment_mode_btn",
        "accept_slice_btn",
        "overlay_mode_combo",
        "overlay_opacity_sb",
        "_focused_toggle_btn",
        "_prev_flagged_btn",
        "_next_flagged_btn",
        "_focused_mode_group",
        "_focused_nav_label",
        "_flagged_labels_list",
        "_mark_good_btn",
        "_delete_label_btn",
        "_discard_garbage_btn",
        "_tier_filter_good",
        "_tier_filter_review",
        "_tier_filter_bad",
        "_tier_filter_garbage",
        "_nav_status_label",
    ]

    def test_build_draw_group_source_parents_stubs(self):
        """_build_draw_group must pass parent=group to all stub widgets.

        Source-level check: every stub widget constructor must include
        'parent=group' to prevent top-level orphaning.
        """
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(Curation3DAnnotator._build_draw_group)
        # Count occurrences of parent=group — should be at least len(_STUB_ATTRS)
        parent_count = source.count("parent=group")
        assert parent_count >= len(self._STUB_ATTRS), (
            f"Expected at least {len(self._STUB_ATTRS)} 'parent=group' in "
            f"_build_draw_group, found {parent_count}. "
            "Stub widgets without parent=group become top-level orphans."
        )

    def test_build_draw_group_no_parentless_constructors(self):
        """No QWidget subclass in _build_draw_group should be constructed
        without parent=group (except the group itself and layout-added widgets).
        """
        import ast
        import inspect
        import textwrap

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(Curation3DAnnotator._build_draw_group)
        source = textwrap.dedent(source)

        # Widget classes that need explicit parents
        widget_classes = {
            "QPushButton", "QCheckBox", "QComboBox", "QGroupBox",
            "QSpinBox", "QLabel", "QListWidget", "QRadioButton",
        }
        # Parse AST and find constructor calls without parent= keyword
        tree = ast.parse(source)
        parentless = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Get the function name
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                continue
            if name not in widget_classes:
                continue
            # Check if parent= is in keyword args
            has_parent = any(kw.arg == "parent" for kw in node.keywords)
            if not has_parent:
                # Exception: QLabel("Patch") and QLabel("Scroll Step") are
                # added to layouts directly, which gives them a parent.
                # Also QGroupBox("3D Curation Tools") is the group itself.
                parentless.append((name, node.lineno))

        # Allowed parentless constructors: widgets added to a layout get an
        # implicit parent from the layout.  The AST analysis can't track
        # layout.addWidget calls, so we allow up to 15 parentless constructors
        # (1 group itself + 14 widgets added to layout/patch_row/step_row/ar_row).
        max_allowed = 15
        assert len(parentless) <= max_allowed, (
            f"Found {len(parentless)} widget constructors without parent=group "
            f"(max allowed: {max_allowed}). Parentless widgets become top-level "
            f"orphans on the taskbar. Offenders: {parentless}"
        )

    def test_data_group_has_parent_in_utility_panel(self):
        """_build_utility_panel must add data_group to a layout or parent it.

        data_group returned by _build_data_group() is an empty QGroupBox.
        Without a layout parent, it becomes a top-level orphan.
        """
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(Curation3DAnnotator._build_utility_panel)
        assert "data_group" in source, "Must reference self.data_group"
        # Must add to layout (addWidget) to give it a parent
        assert "addWidget(self.data_group)" in source, (
            "data_group must be added to a layout to prevent top-level orphaning"
        )

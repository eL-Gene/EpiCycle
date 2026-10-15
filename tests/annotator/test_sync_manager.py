"""Tests for LabelSyncManager — bidirectional sync, edit tracking, properties.

Phase K11d — comprehensive curation test suite.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from napari.layers import Labels

from packages.epicycle_annotator.sync import LabelSyncManager


# ---------------------------------------------------------------------------
# Lightweight fakes (no Qt / napari required)
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def disconnect(self, cb):
        try:
            self._callbacks.remove(cb)
        except ValueError:
            raise ValueError(f"Handler not connected: {cb}")


class _FakeLayer:
    """Minimal Labels-like layer for sync testing."""

    def __init__(self, data, name=""):
        self.data = data
        self.name = name
        self.mode = "paint"
        self.selected_label = 1
        self.show_selected_label = False
        self.brush_size = 1
        self.opacity = 1.0
        self.colormap = "default"
        self.blending = "translucent"
        self.n_edit_dimensions = 2
        self.contour = 0
        self.mouse_drag_callbacks = []
        self.events = types.SimpleNamespace(
            mode=_FakeEvent(),
            selected_label=_FakeEvent(),
            brush_size=_FakeEvent(),
            opacity=_FakeEvent(),
            colormap=_FakeEvent(),
            blending=_FakeEvent(),
            n_edit_dimensions=_FakeEvent(),
            contour=_FakeEvent(),
        )


# Register so isinstance(fake, Labels) returns True
Labels.register(_FakeLayer)


class _FakeViewerModel:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.dims = types.SimpleNamespace(current_step=(0,))
        self.cursor = types.SimpleNamespace(position=(0, 0, 0))


class _FakeLayerList:
    def __init__(self):
        self._layers = []
        self.selection = types.SimpleNamespace(active=None)

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)

    def __contains__(self, name):
        return any(l.name == name for l in self._layers)

    def __getitem__(self, name):
        for l in self._layers:
            if l.name == name:
                return l
        raise KeyError(name)

    def add(self, layer):
        self._layers.append(layer)


def _make_sync_manager():
    """Create a LabelSyncManager with fakes."""
    main = _FakeViewerModel()
    vm_xy = _FakeViewerModel()
    vm_xz = _FakeViewerModel()
    vm_yz = _FakeViewerModel()
    mgr = LabelSyncManager(
        main_viewer=types.SimpleNamespace(layers=main.layers),
        viewer_model_xy=vm_xy,
        viewer_model_xz=vm_xz,
        viewer_model_yz=vm_yz,
        qt_viewer_xy=None,
        qt_viewer_xz=None,
        qt_viewer_yz=None,
    )
    return mgr, main, vm_xy, vm_xz, vm_yz


# ---------------------------------------------------------------------------
# Tests — shared data registration
# ---------------------------------------------------------------------------


class TestSharedDataRegistration:
    def test_register_stores_array(self):
        mgr, _, _, _, _ = _make_sync_manager()
        data = np.zeros((4, 8, 8), dtype=np.int32)
        mgr.register_shared_data("labels", data)
        assert "labels" in mgr.shared_labels
        assert mgr.shared_labels["labels"] is data

    def test_unregister_removes_data(self):
        mgr, _, _, _, _ = _make_sync_manager()
        data = np.zeros((4, 8, 8), dtype=np.int32)
        mgr.register_shared_data("labels", data)
        mgr.unregister_layer("labels")
        assert "labels" not in mgr.shared_labels

    def test_unregister_clears_edited_slices(self):
        mgr, _, _, _, _ = _make_sync_manager()
        data = np.zeros((4, 8, 8), dtype=np.int32)
        mgr.register_shared_data("labels", data)
        mgr.edited_slices["labels"] = {"XY": {1: [0, 1]}, "XZ": {}, "YZ": {}}
        mgr.unregister_layer("labels")
        assert "labels" not in mgr.edited_slices

    def test_unregister_nonexistent_is_noop(self):
        mgr, _, _, _, _ = _make_sync_manager()
        mgr.unregister_layer("nonexistent")  # Should not raise


class TestRenameLayer:
    def test_rename_moves_shared_data(self):
        mgr, _, _, _, _ = _make_sync_manager()
        data = np.zeros((4, 8, 8), dtype=np.int32)
        mgr.register_shared_data("old_name", data)
        mgr.rename_layer("old_name", "new_name")
        assert "new_name" in mgr.shared_labels
        assert "old_name" not in mgr.shared_labels
        assert mgr.shared_labels["new_name"] is data

    def test_rename_moves_edited_slices(self):
        mgr, _, _, _, _ = _make_sync_manager()
        mgr.edited_slices["old_name"] = {"XY": {1: [5]}, "XZ": {}, "YZ": {}}
        mgr.rename_layer("old_name", "new_name")
        assert "new_name" in mgr.edited_slices
        assert "old_name" not in mgr.edited_slices


# ---------------------------------------------------------------------------
# Tests — base label name extraction
# ---------------------------------------------------------------------------


class TestBaseNameExtraction:
    def test_strips_xy_suffix(self):
        mgr, _, _, _, _ = _make_sync_manager()
        assert mgr._base_label_name("seg_XY") == "seg"

    def test_strips_xz_suffix(self):
        mgr, _, _, _, _ = _make_sync_manager()
        assert mgr._base_label_name("seg_XZ") == "seg"

    def test_strips_yz_suffix(self):
        mgr, _, _, _, _ = _make_sync_manager()
        assert mgr._base_label_name("seg_YZ") == "seg"

    def test_no_suffix_unchanged(self):
        mgr, _, _, _, _ = _make_sync_manager()
        assert mgr._base_label_name("seg") == "seg"

    def test_empty_string(self):
        mgr, _, _, _, _ = _make_sync_manager()
        assert mgr._base_label_name("") == ""

    def test_none_returns_empty(self):
        mgr, _, _, _, _ = _make_sync_manager()
        assert mgr._base_label_name(None) == ""


# ---------------------------------------------------------------------------
# Tests — edited slices tracking
# ---------------------------------------------------------------------------


class TestEditedSlicesTracking:
    def test_get_edited_slices_unknown_layer(self):
        mgr, _, _, _, _ = _make_sync_manager()
        result = mgr.get_edited_slices("unknown")
        assert result == {"XY": {}, "XZ": {}, "YZ": {}}

    def test_track_slice_edit_creates_entry(self):
        mgr, _, vm_xy, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        layer.selected_label = 5
        vm_xy.dims.current_step = (2,)
        mgr.track_slice_edit(layer, vm_xy, "XY")
        result = mgr.get_edited_slices("seg")
        assert 5 in result["XY"]
        assert 2 in result["XY"][5]

    def test_track_slice_edit_no_duplicate_index(self):
        mgr, _, vm_xy, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        layer.selected_label = 5
        vm_xy.dims.current_step = (2,)
        mgr.track_slice_edit(layer, vm_xy, "XY")
        mgr.track_slice_edit(layer, vm_xy, "XY")
        result = mgr.get_edited_slices("seg")
        assert result["XY"][5].count(2) == 1

    def test_track_multiple_labels(self):
        mgr, _, vm_xy, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")

        layer.selected_label = 3
        vm_xy.dims.current_step = (0,)
        mgr.track_slice_edit(layer, vm_xy, "XY")

        layer.selected_label = 7
        vm_xy.dims.current_step = (1,)
        mgr.track_slice_edit(layer, vm_xy, "XY")

        result = mgr.get_edited_slices("seg")
        assert 3 in result["XY"]
        assert 7 in result["XY"]
        assert 0 in result["XY"][3]
        assert 1 in result["XY"][7]

    def test_track_skips_label_zero(self):
        mgr, _, vm_xy, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        layer.selected_label = 0
        vm_xy.dims.current_step = (1,)
        mgr.track_slice_edit(layer, vm_xy, "XY")
        result = mgr.get_edited_slices("seg")
        assert len(result["XY"]) == 0

    def test_track_multiple_orientations(self):
        mgr, _, vm_xy, vm_xz, vm_yz = _make_sync_manager()
        layer_xy = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        layer_xz = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XZ")
        layer_xy.selected_label = 1
        layer_xz.selected_label = 1
        vm_xy.dims.current_step = (0,)
        vm_xz.dims.current_step = (3,)
        mgr.track_slice_edit(layer_xy, vm_xy, "XY")
        mgr.track_slice_edit(layer_xz, vm_xz, "XZ")
        result = mgr.get_edited_slices("seg")
        assert 0 in result["XY"][1]
        assert 3 in result["XZ"][1]


# ---------------------------------------------------------------------------
# Tests — sync_to_main_view
# ---------------------------------------------------------------------------


class TestSyncToMainView:
    def test_xy_sync_copies_correct_slice(self):
        mgr, main, vm_xy, _, _ = _make_sync_manager()
        shared = np.zeros((4, 8, 8), dtype=np.int32)
        mgr.register_shared_data("seg", shared)

        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        layer.data[1, 2, 3] = 99
        vm_xy.dims.current_step = (1,)

        mgr.sync_to_main_view(layer, "XY", vm_xy)
        assert shared[1, 2, 3] == 99

    def test_xz_sync_copies_correct_slice(self):
        mgr, main, _, vm_xz, _ = _make_sync_manager()
        shared = np.zeros((4, 8, 8), dtype=np.int32)
        mgr.register_shared_data("seg", shared)

        # In XZ view, layer data is transposed: shape (Y, Z, X) → (8, 4, 8)
        # The sync code does: shared_data[:, idx, :] = layer.data[idx, :, :]
        # so layer.data must have shape (8, 4, 8) for XZ orientation
        xz_data = np.zeros((8, 4, 8), dtype=np.int32)
        xz_data[2, :, :] = 77  # Y-index=2
        layer = _FakeLayer(xz_data, name="seg_XZ")
        vm_xz.dims.current_step = (2,)

        mgr.sync_to_main_view(layer, "XZ", vm_xz)
        np.testing.assert_array_equal(shared[:, 2, :], xz_data[2, :, :])

    def test_sync_unknown_layer_is_noop(self):
        mgr, _, vm_xy, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="unknown_XY")
        mgr.sync_to_main_view(layer, "XY", vm_xy)  # Should not raise

    def test_deferred_refresh_skips_main_layer_refresh(self):
        """When defer_main_refresh=True, main layer's refresh is not called."""
        mgr, main, vm_xy, _, _ = _make_sync_manager()
        shared = np.zeros((4, 8, 8), dtype=np.int32)
        mgr.register_shared_data("seg", shared)

        # Add a main layer that tracks refresh calls
        main_layer = _FakeLayer(shared, name="seg")
        main_layer._refresh_called = False
        original_refresh = lambda: setattr(main_layer, "_refresh_called", True)
        main_layer.refresh = original_refresh
        main.layers.add(main_layer)

        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        vm_xy.dims.current_step = (0,)

        mgr.sync_to_main_view(layer, "XY", vm_xy, defer_main_refresh=True)
        assert main_layer._refresh_called is False


# ---------------------------------------------------------------------------
# Tests — connect_label_sync guard
# ---------------------------------------------------------------------------


class TestConnectLabelSyncGuard:
    def test_connect_once_sets_flag(self):
        mgr, _, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        mgr.connect_label_sync(layer)
        assert getattr(layer, "_epicycle_sync_connected", False) is True

    def test_connect_twice_no_duplicate_handlers(self):
        mgr, _, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        mgr.connect_label_sync(layer)
        n_callbacks_1 = len(layer.events.mode._callbacks)
        mgr.connect_label_sync(layer)
        n_callbacks_2 = len(layer.events.mode._callbacks)
        assert n_callbacks_1 == n_callbacks_2

    def test_connects_all_property_events(self):
        mgr, _, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        mgr.connect_label_sync(layer)
        assert len(layer.events.mode._callbacks) >= 1
        assert len(layer.events.selected_label._callbacks) >= 1
        assert len(layer.events.brush_size._callbacks) >= 1
        assert len(layer.events.opacity._callbacks) >= 1
        assert len(layer.events.colormap._callbacks) >= 1
        assert len(layer.events.blending._callbacks) >= 1


# ---------------------------------------------------------------------------
# Tests — sync_label_properties
# ---------------------------------------------------------------------------


class TestSyncLabelProperties:
    def test_properties_propagated_to_subviewers(self):
        mgr, _, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        # Add subviewer label layers
        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            vm.layers.add(sub)

        active = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        active.brush_size = 10
        active.opacity = 0.5
        active.colormap = "custom_cmap"

        mgr.sync_label_properties(active)

        # 2026-05-17: brush_size is hardcoded to 1 in sync_label_properties
        # regardless of active_layer.brush_size (locked
        # design decision).  opacity / colormap still propagate normally.
        for vm in [vm_xy, vm_xz, vm_yz]:
            for layer in vm.layers:
                assert layer.brush_size == 1
                assert layer.opacity == 0.5
                assert layer.colormap == "custom_cmap"


# ---------------------------------------------------------------------------
# Tests — sync_selected_label
# ---------------------------------------------------------------------------


class TestColormapSyncPropagation:
    """Colormap changes on the main layer propagate to subviewers.

    The ColormapAuthority (not sync.py flags) prevents controller/annotator
    conflicts — sync.py always propagates colormaps unconditionally.
    """

    def test_colormap_sync_propagates(self):
        mgr, _, vm_xy, vm_xz, vm_yz = _make_sync_manager()
        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.colormap = "original"
            vm.layers.add(sub)

        active = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        active.colormap = "new_cmap"

        mgr.sync_label_properties(active)

        for vm in [vm_xy, vm_xz, vm_yz]:
            for layer in vm.layers:
                assert layer.colormap == "new_cmap"

    def test_all_properties_sync_together(self):
        mgr, _, vm_xy, vm_xz, vm_yz = _make_sync_manager()
        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.colormap = "original"
            sub.brush_size = 1
            sub.opacity = 1.0
            vm.layers.add(sub)

        active = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        active.colormap = "qc_cmap"
        active.brush_size = 20
        active.opacity = 0.3

        mgr.sync_label_properties(active)

        # 2026-05-17: brush_size hardcoded to 1.  The
        # remaining properties still propagate normally.
        for vm in [vm_xy, vm_xz, vm_yz]:
            for layer in vm.layers:
                assert layer.brush_size == 1, "brush_size locked at 1"
                assert layer.opacity == 0.3, "opacity should sync"
                assert layer.colormap == "qc_cmap", "colormap should sync"

    def test_contour_syncs_to_subviewers(self):
        # Bumping contour (outline) on the active/main labels layer mirrors to
        # the side views, so outlines show in XY/XZ/YZ (eyeball merges).
        mgr, _, vm_xy, vm_xz, vm_yz = _make_sync_manager()
        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.contour = 0
            vm.layers.add(sub)

        active = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        active.contour = 2
        mgr.sync_label_properties(active)

        for vm in [vm_xy, vm_xz, vm_yz]:
            for layer in vm.layers:
                assert layer.contour == 2, "contour should sync to subviewers"

    def test_contour_zero_syncs_off(self):
        # Setting contour back to 0 (filled) also propagates off.
        mgr, _, vm_xy, vm_xz, vm_yz = _make_sync_manager()
        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.contour = 3
            vm.layers.add(sub)

        active = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        active.contour = 0
        mgr.sync_label_properties(active)

        for vm in [vm_xy, vm_xz, vm_yz]:
            for layer in vm.layers:
                assert layer.contour == 0, "contour=0 (filled) should sync"


class TestSyncSelectedLabel:
    def test_updates_main_and_subviewers(self):
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main_layer.selected_label = 1
        main.layers.add(main_layer)

        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.selected_label = 1
            vm.layers.add(sub)

        source = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        source.selected_label = 42

        mgr.sync_selected_label(source)

        assert main_layer.selected_label == 42
        for vm in [vm_xy, vm_xz, vm_yz]:
            for layer in vm.layers:
                assert layer.selected_label == 42


# ---------------------------------------------------------------------------
# Tests — n_edit_dimensions bidirectional sync (main ↔ subviewers)
# ---------------------------------------------------------------------------


class TestNEditDimensionsSync:
    """n_edit_dimensions must stay in sync between main viewer and subviewers."""

    def test_subviewer_change_propagates_to_main(self):
        """Changing n_edit_dimensions on a subviewer should update the main layer."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main_layer.n_edit_dimensions = 2
        main.layers.add(main_layer)

        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.n_edit_dimensions = 2
            vm.layers.add(sub)

        # Simulate a subviewer changing n_edit_dimensions to 3
        source = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        source.n_edit_dimensions = 3

        mgr.sync_label_properties(source)

        assert main_layer.n_edit_dimensions == 3, (
            "Main layer should be updated when subviewer changes n_edit_dimensions"
        )

    def test_main_change_propagates_to_subviewers(self):
        """Changing n_edit_dimensions on the main layer should update all subviewers."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main_layer.n_edit_dimensions = 3
        main.layers.add(main_layer)

        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.n_edit_dimensions = 2
            vm.layers.add(sub)

        mgr.sync_label_properties(main_layer)

        for vm in [vm_xy, vm_xz, vm_yz]:
            for layer in vm.layers:
                assert layer.n_edit_dimensions == 3, (
                    f"Subviewer {layer.name} should be updated to n_edit_dimensions=3"
                )

    def test_main_event_connected_via_connect_main_label_sync(self):
        """connect_main_label_sync should connect n_edit_dimensions event on main layer."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(main_layer)

        mgr.connect_main_label_sync(main_layer)

        # The n_edit_dimensions event should have a callback
        assert len(main_layer.events.n_edit_dimensions._callbacks) > 0, (
            "connect_main_label_sync should connect n_edit_dimensions event"
        )

    def test_brush_size_event_connected_on_main(self):
        """REGRESSION (D-brush-sync 2026-05-10, narrowed): connect_main_
        label_sync must connect ``events.brush_size`` so changing the
        brush size on the main viewer's slider propagates to subviewers.

        Pre-fix: only ``n_edit_dimensions`` was wired on the main layer.
        Setting brush_size on main fired the event but no listener was
        attached → subviewer brush stayed stale.

        DELIBERATELY NOT WIRED (after 3D crash 2026-05-10): colormap,
        opacity, blending.  See connect_main_label_sync docstring for
        rationale — those properties already have explicit propagation
        paths and the duplicate event-driven path triggered crashes
        during patch teardown.
        """
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()
        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(main_layer)

        mgr.connect_main_label_sync(main_layer)

        # brush_size MUST be wired.
        brush_emitter = getattr(main_layer.events, "brush_size", None)
        assert brush_emitter is not None
        assert len(brush_emitter._callbacks) > 0, (
            "connect_main_label_sync should connect brush_size event"
        )
        # colormap / opacity / blending MUST NOT be wired — duplicating
        # propagation through events triggered a 3D crash during patch
        # teardown and they have explicit propagation paths elsewhere.
        for event_name in ("colormap", "opacity", "blending"):
            emitter = getattr(main_layer.events, event_name, None)
            if emitter is None:
                continue
            assert len(emitter._callbacks) == 0, (
                f"connect_main_label_sync MUST NOT connect {event_name} "
                f"on the main layer — it duplicates explicit propagation "
                f"and crashes during patch teardown."
            )

    def test_main_brush_size_change_propagates_to_subviewers(self):
        """Changing brush_size on the main layer (e.g., via napari's
        slider) should propagate to ALL subviewer label layers."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main_layer.brush_size = 1
        main.layers.add(main_layer)

        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.brush_size = 1
            vm.layers.add(sub)

        # Simulate connect_main_label_sync wiring the events.
        mgr.connect_main_label_sync(main_layer)

        # User adjusts brush_size on main viewer's slider.
        # Mock event firing — sync_label_properties is called with the
        # main layer as the source.  This is what events.brush_size.
        # connect would trigger.
        main_layer.brush_size = 7
        mgr.sync_label_properties(main_layer)

        # 2026-05-17: brush_size is hardcoded to 1 in
        # sync_label_properties (locked design decision).
        # Even if the user/code sets brush_size=7 on the source, the
        # sync propagates 1 to all subviewers.  Unlock path is
        # documented in lock_brush_size_to_1.
        for vm in [vm_xy, vm_xz, vm_yz]:
            for layer in vm.layers:
                assert layer.brush_size == 1, (
                    f"Subviewer {layer.name} brush_size locked at 1 "
                    f"regardless of source; got {layer.brush_size}"
                )


# ---------------------------------------------------------------------------
# Tests — disconnect_label_sync
# ---------------------------------------------------------------------------


class TestDisconnectLabelSync:
    """Verify clean disconnection of label sync event handlers."""

    def test_disconnect_resets_guard_flag(self):
        """After disconnect_label_sync, _epicycle_sync_connected must be False."""
        mgr, _, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        mgr.connect_label_sync(layer)
        assert layer._epicycle_sync_connected is True

        mgr.disconnect_label_sync(layer)
        assert layer._epicycle_sync_connected is False

    def test_disconnect_removes_all_event_handlers(self):
        """After disconnect, no callbacks should remain on any event emitter."""
        mgr, _, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        mgr.connect_label_sync(layer)

        # Verify handlers were connected
        for attr in ("mode", "selected_label", "brush_size", "opacity",
                      "colormap", "blending", "n_edit_dimensions"):
            assert len(getattr(layer.events, attr)._callbacks) >= 1

        mgr.disconnect_label_sync(layer)

        # All handlers should be removed
        for attr in ("mode", "selected_label", "brush_size", "opacity",
                      "colormap", "blending", "n_edit_dimensions"):
            assert len(getattr(layer.events, attr)._callbacks) == 0, (
                f"Event {attr} should have 0 callbacks after disconnect"
            )

    def test_disconnect_then_reconnect_works(self):
        """After disconnect + reconnect, handlers are present and work."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(layer)

        mgr.connect_label_sync(layer)
        mgr.disconnect_label_sync(layer)

        # Reconnect must succeed (flag was reset)
        mgr.connect_label_sync(layer)
        assert layer._epicycle_sync_connected is True
        assert len(layer.events.selected_label._callbacks) == 1

    def test_disconnect_noop_when_not_connected(self):
        """Calling disconnect on a never-connected layer should not raise."""
        mgr, _, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        # Should not raise
        mgr.disconnect_label_sync(layer)
        assert layer._epicycle_sync_connected is False

    def test_no_handler_accumulation_across_cycles(self):
        """Multiple connect/disconnect cycles must not accumulate handlers."""
        mgr, _, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")

        for _ in range(5):
            mgr.connect_label_sync(layer)
            assert len(layer.events.mode._callbacks) == 1
            mgr.disconnect_label_sync(layer)
            assert len(layer.events.mode._callbacks) == 0

    def test_two_managers_disconnect_independently(self):
        """Two separate SyncManagers should disconnect their own handlers."""
        mgr1, _, _, _, _ = _make_sync_manager()
        mgr2, _, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")

        mgr1.connect_label_sync(layer)
        # Reset flag so mgr2 can also connect
        layer._epicycle_sync_connected = False
        mgr2.connect_label_sync(layer)

        # Both added handlers
        assert len(layer.events.mode._callbacks) == 2

        # Disconnect mgr1's handlers
        mgr1.disconnect_label_sync(layer)
        assert len(layer.events.mode._callbacks) == 1

        # The remaining handler is mgr2's
        mgr2.disconnect_label_sync(layer)
        assert len(layer.events.mode._callbacks) == 0


# ---------------------------------------------------------------------------
# Tests — disconnect_main_label_sync
# ---------------------------------------------------------------------------


class TestDisconnectMainLabelSync:
    """Verify clean disconnection of main label drag + n_edit handlers."""

    def test_disconnect_resets_guard_flag(self):
        mgr, main, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(layer)
        mgr.connect_main_label_sync(layer)
        assert layer._epicycle_main_drag_connected is True

        mgr.disconnect_main_label_sync(layer)
        assert layer._epicycle_main_drag_connected is False

    def test_disconnect_removes_drag_callback(self):
        mgr, main, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(layer)
        mgr.connect_main_label_sync(layer)
        assert len(layer.mouse_drag_callbacks) == 1

        mgr.disconnect_main_label_sync(layer)
        assert len(layer.mouse_drag_callbacks) == 0

    def test_disconnect_removes_n_edit_handler(self):
        mgr, main, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(layer)
        mgr.connect_main_label_sync(layer)
        assert len(layer.events.n_edit_dimensions._callbacks) >= 1

        mgr.disconnect_main_label_sync(layer)
        assert len(layer.events.n_edit_dimensions._callbacks) == 0

    def test_disconnect_noop_when_not_connected(self):
        mgr, _, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        mgr.disconnect_main_label_sync(layer)  # Should not raise
        assert layer._epicycle_main_drag_connected is False

    def test_disconnect_then_reconnect_works(self):
        mgr, main, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(layer)

        mgr.connect_main_label_sync(layer)
        mgr.disconnect_main_label_sync(layer)
        mgr.connect_main_label_sync(layer)

        assert layer._epicycle_main_drag_connected is True
        assert len(layer.mouse_drag_callbacks) == 1
        assert len(layer.events.n_edit_dimensions._callbacks) == 1

    def test_no_accumulation_across_cycles(self):
        mgr, main, _, _, _ = _make_sync_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(layer)

        for _ in range(5):
            mgr.connect_main_label_sync(layer)
            assert len(layer.mouse_drag_callbacks) == 1
            mgr.disconnect_main_label_sync(layer)
            assert len(layer.mouse_drag_callbacks) == 0


# ---------------------------------------------------------------------------
# Tests — show_selected_label sync with selected_label
# ---------------------------------------------------------------------------


class TestShowSelectedLabelSync:
    """Verify that show_selected_label is synced atomically alongside
    selected_label, preventing desync between main viewer and subviewers.

    Without this, isolation mode can show different labels in the main
    viewer vs subviewers under lag, because show_selected_label was only
    propagated via a separate manual loop (_propagate_show_selected_label)
    that runs after the sync cascade.
    """

    def test_show_selected_label_propagated_to_subviewers(self):
        """When source has show_selected_label=True, all subviewers must
        also get show_selected_label=True after sync_selected_label."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(main_layer)

        subs = []
        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.show_selected_label = False
            vm.layers.add(sub)
            subs.append(sub)

        # Simulate isolation: source has show_selected_label=True
        main_layer.selected_label = 5
        main_layer.show_selected_label = True
        mgr.sync_selected_label(main_layer)

        for sub in subs:
            assert sub.selected_label == 5, (
                f"{sub.name}: selected_label must match main"
            )
            assert sub.show_selected_label is True, (
                f"{sub.name}: show_selected_label must be synced to True "
                "to prevent isolation desync"
            )

    def test_show_selected_label_false_propagated_on_isolation_off(self):
        """When isolation is turned off (show_selected_label=False), all
        subviewers must also clear show_selected_label."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(main_layer)

        subs = []
        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.show_selected_label = True  # Previously isolated
            vm.layers.add(sub)
            subs.append(sub)

        # Turn off isolation
        main_layer.selected_label = 5
        main_layer.show_selected_label = False
        mgr.sync_selected_label(main_layer)

        for sub in subs:
            assert sub.show_selected_label is False, (
                f"{sub.name}: show_selected_label must be cleared on isolation off"
            )

    def test_selected_label_and_show_selected_label_consistent_after_rapid_switch(self):
        """Simulates rapid label switching: after N calls to sync_selected_label
        with different label_ids + show_selected_label=True, all viewers must
        agree on the LAST label.

        This tests the scenario where the user clicks rapidly in the navigator
        with isolation enabled.
        """
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(main_layer)

        subs = []
        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            vm.layers.add(sub)
            subs.append(sub)

        # Rapid switching: labels 3, 7, 12
        for label_id in [3, 7, 12]:
            main_layer.selected_label = label_id
            main_layer.show_selected_label = True
            mgr.sync_selected_label(main_layer)

        # All must show label 12 (the last one)
        for sub in subs:
            assert sub.selected_label == 12, (
                f"{sub.name}: must show last label (12), not stale"
            )
            assert sub.show_selected_label is True

    def test_subviewer_isolation_propagated_to_main(self):
        """When a subviewer sets show_selected_label, the main layer must
        also be updated (bidirectional sync)."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main_layer.show_selected_label = False
        main.layers.add(main_layer)

        sub_xy = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        vm_xy.layers.add(sub_xy)

        # Subviewer initiates isolation
        sub_xy.selected_label = 9
        sub_xy.show_selected_label = True
        mgr.sync_selected_label(sub_xy)

        assert main_layer.selected_label == 9
        assert main_layer.show_selected_label is True, (
            "Main layer must get show_selected_label=True from subviewer sync"
        )


# ---------------------------------------------------------------------------
# Tests — _suppress_show_selected_sync flag (3D proxy isolation guard)
# ---------------------------------------------------------------------------


class TestSuppressShowSelectedSync:
    """When 3D proxy isolation is active, the main layer has
    show_selected_label=False intentionally (to avoid expensive iso_categorical
    re-render).  sync_selected_label must NOT propagate that False to subviewers.

    The _suppress_show_selected_sync flag on LabelSyncManager controls this:
    when True, sync_selected_label still propagates selected_label but skips
    show_selected_label propagation entirely.
    """

    def test_suppress_flag_defaults_false(self):
        """Flag should default to False — normal sync propagation."""
        mgr, _, _, _, _ = _make_sync_manager()
        assert mgr._suppress_show_selected_sync is False

    def test_sync_propagates_show_sel_when_not_suppressed(self):
        """Baseline: when flag is False, show_selected_label propagates normally."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main_layer.selected_label = 7
        main_layer.show_selected_label = True
        main.layers.add(main_layer)

        subs = []
        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.show_selected_label = False
            vm.layers.add(sub)
            subs.append(sub)

        mgr._suppress_show_selected_sync = False
        mgr.sync_selected_label(main_layer)

        for sub in subs:
            assert sub.selected_label == 7
            assert sub.show_selected_label is True, (
                f"{sub.name}: show_selected_label should propagate when not suppressed"
            )

    def test_sync_skips_show_sel_when_suppressed(self):
        """When flag is True, selected_label still syncs but
        show_selected_label is NOT overwritten on subviewers."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main_layer.selected_label = 7
        main_layer.show_selected_label = False  # Proxy mode: main has False
        main.layers.add(main_layer)

        subs = []
        for vm, suffix in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            sub = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name=f"seg_{suffix}")
            sub.show_selected_label = True  # Subviewers should keep True
            sub.selected_label = 1  # Old value
            vm.layers.add(sub)
            subs.append(sub)

        mgr._suppress_show_selected_sync = True
        mgr.sync_selected_label(main_layer)

        for sub in subs:
            assert sub.selected_label == 7, (
                f"{sub.name}: selected_label must still sync even when suppressed"
            )
            assert sub.show_selected_label is True, (
                f"{sub.name}: show_selected_label must NOT be overwritten "
                "when _suppress_show_selected_sync is True"
            )

    def test_suppress_also_protects_main_layer(self):
        """When a subviewer fires sync_selected_label with suppress ON,
        the main layer's show_selected_label should not be overwritten either."""
        mgr, main, vm_xy, vm_xz, vm_yz = _make_sync_manager()

        main_layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main_layer.selected_label = 1
        main_layer.show_selected_label = False  # Proxy mode
        main.layers.add(main_layer)

        sub_xy = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg_XY")
        sub_xy.selected_label = 7
        sub_xy.show_selected_label = True
        vm_xy.layers.add(sub_xy)

        mgr._suppress_show_selected_sync = True
        mgr.sync_selected_label(sub_xy)

        assert main_layer.selected_label == 7, (
            "selected_label must still sync to main"
        )
        assert main_layer.show_selected_label is False, (
            "Main layer show_selected_label must NOT be overwritten when suppressed"
        )


# -----------------------------------------------------------------------
# disconnect_all — comprehensive handler flush
# -----------------------------------------------------------------------


class TestDisconnectAll:
    """Tests for LabelSyncManager.disconnect_all()."""

    def test_disconnect_all_clears_sync_handlers(self):
        """disconnect_all() flushes _sync_handlers and disconnects events."""
        main, vm_xy, vm_xz, vm_yz = (
            _FakeViewerModel(), _FakeViewerModel(),
            _FakeViewerModel(), _FakeViewerModel(),
        )
        mgr = LabelSyncManager(main, vm_xy, vm_xz, vm_yz, None, None, None)
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(layer)
        mgr.connect_label_sync(layer)
        # Should have handlers connected
        assert len(layer.events.mode._callbacks) == 1
        assert len(mgr._sync_handlers) == 1

        mgr.disconnect_all()

        assert len(mgr._sync_handlers) == 0
        assert len(layer.events.mode._callbacks) == 0
        assert len(layer.events.selected_label._callbacks) == 0

    def test_disconnect_all_clears_main_drag_handlers(self):
        """disconnect_all() flushes _main_drag_handlers dict."""
        main, vm_xy, vm_xz, vm_yz = (
            _FakeViewerModel(), _FakeViewerModel(),
            _FakeViewerModel(), _FakeViewerModel(),
        )
        mgr = LabelSyncManager(main, vm_xy, vm_xz, vm_yz, None, None, None)
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(layer)
        mgr.connect_main_label_sync(layer)
        assert len(mgr._main_drag_handlers) == 1

        mgr.disconnect_all()

        assert len(mgr._main_drag_handlers) == 0

    def test_disconnect_all_idempotent(self):
        """Calling disconnect_all() twice is safe."""
        main, vm_xy, vm_xz, vm_yz = (
            _FakeViewerModel(), _FakeViewerModel(),
            _FakeViewerModel(), _FakeViewerModel(),
        )
        mgr = LabelSyncManager(main, vm_xy, vm_xz, vm_yz, None, None, None)
        mgr.disconnect_all()
        mgr.disconnect_all()  # Should not raise

    def test_no_accumulation_with_disconnect_all_in_cycle(self):
        """Simulating close/relaunch: disconnect_all prevents accumulation."""
        main, vm_xy, vm_xz, vm_yz = (
            _FakeViewerModel(), _FakeViewerModel(),
            _FakeViewerModel(), _FakeViewerModel(),
        )
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="seg")
        main.layers.add(layer)

        for cycle in range(5):
            mgr = LabelSyncManager(main, vm_xy, vm_xz, vm_yz, None, None, None)
            mgr.connect_label_sync(layer)
            # Each cycle should have exactly 1 handler per event
            assert len(layer.events.mode._callbacks) == 1, (
                f"Cycle {cycle}: expected 1 mode handler, got {len(layer.events.mode._callbacks)}"
            )
            # Simulate teardown: per-layer disconnect + flush
            mgr.disconnect_label_sync(layer)
            mgr.disconnect_all()

        assert len(layer.events.mode._callbacks) == 0


# ---------------------------------------------------------------------------
# Tests — clear_edited_slices_for_label
# ---------------------------------------------------------------------------


class TestClearEditedSlicesForLabel:
    """Verify clear_edited_slices_for_label removes entries for a single label."""

    def test_clears_label_from_all_orientations(self):
        mgr, _, _, _, _ = _make_sync_manager()
        mgr.edited_slices["seg"] = {
            "XY": {5: [0, 1, 2]},
            "XZ": {5: [3]},
            "YZ": {5: [4, 5]},
        }
        mgr.clear_edited_slices_for_label("seg", 5)
        result = mgr.get_edited_slices("seg")
        assert 5 not in result["XY"]
        assert 5 not in result["XZ"]
        assert 5 not in result["YZ"]

    def test_preserves_other_labels(self):
        mgr, _, _, _, _ = _make_sync_manager()
        mgr.edited_slices["seg"] = {
            "XY": {5: [0, 1], 10: [2, 3]},
            "XZ": {5: [4]},
            "YZ": {},
        }
        mgr.clear_edited_slices_for_label("seg", 5)
        result = mgr.get_edited_slices("seg")
        assert 5 not in result["XY"]
        assert 10 in result["XY"]
        assert result["XY"][10] == [2, 3]

    def test_handles_numpy_int_keys(self):
        mgr, _, _, _, _ = _make_sync_manager()
        mgr.edited_slices["seg"] = {
            "XY": {np.int64(5): [0, 1]},
            "XZ": {},
            "YZ": {},
        }
        mgr.clear_edited_slices_for_label("seg", 5)
        result = mgr.get_edited_slices("seg")
        for key in result["XY"]:
            assert int(key) != 5

    def test_noop_for_unknown_layer(self):
        mgr, _, _, _, _ = _make_sync_manager()
        # Should not raise
        mgr.clear_edited_slices_for_label("unknown_layer", 5)

    def test_noop_for_absent_label(self):
        mgr, _, _, _, _ = _make_sync_manager()
        mgr.edited_slices["seg"] = {
            "XY": {10: [0]},
            "XZ": {},
            "YZ": {},
        }
        mgr.clear_edited_slices_for_label("seg", 5)
        result = mgr.get_edited_slices("seg")
        assert 10 in result["XY"]

    def test_empty_edited_slices(self):
        mgr, _, _, _, _ = _make_sync_manager()
        mgr.edited_slices["seg"] = {"XY": {}, "XZ": {}, "YZ": {}}
        mgr.clear_edited_slices_for_label("seg", 5)
        result = mgr.get_edited_slices("seg")
        assert result == {"XY": {}, "XZ": {}, "YZ": {}}

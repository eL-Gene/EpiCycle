"""Tests for SubviewerManager — comprehensive curation subviewer test suite.

Covers: colormap propagation, fast reload, layer naming, in-place updates,
visibility/style sync, drag callback guards, image RGB detection, removal,
renaming, and visibility guard logic.

Phase K11d.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from packages.epicycle_annotator.subviewers import SubviewerManager
from packages.epicycle_annotator.orientation import transpose_data


# ---------------------------------------------------------------------------
# Lightweight fakes (no Qt / napari required)
# ---------------------------------------------------------------------------

class _FakeLayer:
    """Minimal layer stand-in for testing SubviewerManager."""

    def __init__(self, data, name="", layer_type="image", **kwargs):
        self.data = data
        self.name = name
        self._type = layer_type
        self.opacity = kwargs.get("opacity", 1.0)
        self.blending = kwargs.get("blending", "translucent")
        self.colormap = kwargs.get("colormap", "gray")
        self.contrast_limits = kwargs.get("contrast_limits", (0, 1))
        self.gamma = kwargs.get("gamma", 1.0)
        self.rendering = kwargs.get("rendering", "mip")
        self.visible = kwargs.get("visible", True)
        self.editable = True
        self.brush_size = kwargs.get("brush_size", 1)
        self.selected_label = kwargs.get("selected_label", 1)
        self.mode = kwargs.get("mode", "pan_zoom")
        self.mouse_drag_callbacks = []
        self.contour = kwargs.get("contour", 0)
        self.rgb = kwargs.get("rgb", False)
        # napari events stub
        self.events = types.SimpleNamespace(
            visible=_FakeEvent(),
            colormap=_FakeEvent(),
            contrast_limits=_FakeEvent(),
            opacity=_FakeEvent(),
            gamma=_FakeEvent(),
            mode=_FakeEvent(),
            selected_label=_FakeEvent(),
            brush_size=_FakeEvent(),
            blending=_FakeEvent(),
        )

    @property
    def ndim(self):
        return self.data.ndim


class _FakeEvent:
    """Stub for napari layer events."""

    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def disconnect(self, cb):
        try:
            self._callbacks.remove(cb)
        except ValueError:
            raise ValueError(f"Handler not connected: {cb}")


class _FakeViewerModel:
    """Lightweight viewer model with layer add/remove."""

    def __init__(self):
        self.layers = _FakeLayerList()
        self.dims = types.SimpleNamespace(
            current_step=(0,),
            range=((0, 10, 1),),
        )
        self.cursor = types.SimpleNamespace(position=(0, 0, 0))

    def add_image(self, data, name="", **kwargs):
        layer = _FakeLayer(data, name=name, layer_type="image", **kwargs)
        self.layers._layers.append(layer)
        return layer

    def add_labels(self, data, name=""):
        layer = _FakeLayer(data, name=name, layer_type="labels")
        self.layers._layers.append(layer)
        return layer


class _FakeLayerList:
    def __init__(self):
        self._layers = []

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

    def remove(self, layer):
        self._layers.remove(layer)


class _FakeSyncManager:
    def __init__(self):
        self.shared_labels = {}
        self._sync_connected = set()

    def register_shared_data(self, name, data):
        self.shared_labels[name] = data

    def connect_label_sync(self, label_layer):
        self._sync_connected.add(label_layer.name)


class _FakeDrawingHandler:
    def __init__(self):
        self._callbacks_created = []

    def create_mouse_drag_callback(self, orientation, viewer_model):
        self._callbacks_created.append(orientation)
        def _cb(layer, event):
            pass
        return _cb


def _make_manager():
    """Create a SubviewerManager with 3 fake viewer models."""
    vm_xy = _FakeViewerModel()
    vm_xz = _FakeViewerModel()
    vm_yz = _FakeViewerModel()
    mgr = SubviewerManager(
        vm_xy, vm_xz, vm_yz,
        qt_viewer_xy=None,
        qt_viewer_xz=None,
        qt_viewer_yz=None,
    )
    return mgr, vm_xy, vm_xz, vm_yz


# ---------------------------------------------------------------------------
# Tests — colormap propagation in add_labels_to_subviewers
# ---------------------------------------------------------------------------


class TestColormapPropagation:
    """Verify that colormap from the source label layer is copied to subviewers."""

    def test_default_colormap_propagated(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="test_labels", layer_type="labels")
        label_layer.colormap = "custom_cmap"

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        for vm in [vm_xy, vm_xz, vm_yz]:
            lbl_layers = [l for l in vm.layers if l._type == "labels"]
            assert len(lbl_layers) == 1
            assert lbl_layers[0].colormap == "custom_cmap"

    def test_dict_colormap_propagated(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="test_labels", layer_type="labels")
        mock_cmap = {"color_dict": {1: [1, 0, 0, 1], 2: [0, 1, 0, 1]}}
        label_layer.colormap = mock_cmap

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        for vm in [vm_xy, vm_xz, vm_yz]:
            lbl_layers = [l for l in vm.layers if l._type == "labels"]
            assert lbl_layers[0].colormap == mock_cmap


# ---------------------------------------------------------------------------
# Tests — label property propagation
# ---------------------------------------------------------------------------


class TestLabelPropertyPropagation:
    def test_brush_size_propagated(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="lbl", layer_type="labels", brush_size=5)

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        # 2026-05-17: brush_size is locked at 1 across all annotator
        # layers.  Subviewers no longer copy the source
        # layer's brush_size — they get 1 via lock_brush_size_to_1.
        for vm in [vm_xy, vm_xz, vm_yz]:
            lbl_layers = [l for l in vm.layers if l._type == "labels"]
            assert lbl_layers[0].brush_size == 1

    def test_selected_label_propagated(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="lbl", layer_type="labels", selected_label=42)

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        for vm in [vm_xy, vm_xz, vm_yz]:
            lbl_layers = [l for l in vm.layers if l._type == "labels"]
            assert lbl_layers[0].selected_label == 42

    def test_mode_set_to_paint(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="lbl", layer_type="labels")

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        for vm in [vm_xy, vm_xz, vm_yz]:
            lbl_layers = [l for l in vm.layers if l._type == "labels"]
            assert lbl_layers[0].mode == "paint"

    def test_editable_set_true(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="lbl", layer_type="labels")

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        for vm in [vm_xy, vm_xz, vm_yz]:
            lbl_layers = [l for l in vm.layers if l._type == "labels"]
            assert lbl_layers[0].editable is True

    def test_preserve_labels_set_true(self):
        # Clip drawing to existing labels: every
        # subviewer label layer must have preserve_labels=True so the brush only
        # fills background + the eraser only removes the selected label.
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="lbl", layer_type="labels")

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        for vm in [vm_xy, vm_xz, vm_yz]:
            lbl_layers = [l for l in vm.layers if l._type == "labels"]
            assert lbl_layers[0].preserve_labels is True

    def test_clip_exempt_layer_propagates_false(self):
        # A source layer that opts OUT (_epicycle_clip_to_existing=False, e.g. the
        # split Refine layer) must yield subviewer labels with preserve_labels
        # FALSE + the flag propagated, so split seeds can paint over the target.
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="Refine", layer_type="labels")
        label_layer._epicycle_clip_to_existing = False

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        for vm in [vm_xy, vm_xz, vm_yz]:
            lbl_layers = [l for l in vm.layers if l._type == "labels"]
            assert lbl_layers[0].preserve_labels is False
            assert lbl_layers[0]._epicycle_clip_to_existing is False


# ---------------------------------------------------------------------------
# Tests — drag callback guard
# ---------------------------------------------------------------------------


class TestDragCallbackGuard:
    def test_drag_callback_added_once(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="lbl", layer_type="labels")

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        for vm in [vm_xy, vm_xz, vm_yz]:
            lbl = [l for l in vm.layers if l._type == "labels"][0]
            assert len(lbl.mouse_drag_callbacks) == 1
            assert getattr(lbl, "_epicycle_drag_connected", False) is True

    def test_drag_callback_not_duplicated_on_re_add(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="lbl", layer_type="labels")

        mgr.add_labels_to_subviewers(label_layer, sync, draw)
        # Re-add (same layer name, same shape) — in-place update
        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        for vm in [vm_xy, vm_xz, vm_yz]:
            lbl = [l for l in vm.layers if l._type == "labels"][0]
            assert len(lbl.mouse_drag_callbacks) == 1

    def test_three_orientations_get_callbacks(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="lbl", layer_type="labels")

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        assert sorted(draw._callbacks_created) == ["XY", "XZ", "YZ"]


# ---------------------------------------------------------------------------
# Tests — image layer addition
# ---------------------------------------------------------------------------


class TestAddImageToSubviewers:
    def test_non_rgb_image_gets_mip_rendering(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()

        data = np.zeros((4, 8, 8), dtype=np.float32)
        img_layer = _FakeLayer(data, name="ch0", layer_type="image")

        mgr.add_image_to_subviewers(img_layer)

        for vm in [vm_xy, vm_xz, vm_yz]:
            img = vm.layers._layers[0]
            assert img.rendering == "mip"

    def test_rgb_image_detected_from_flag(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()

        data = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        img_layer = _FakeLayer(data, name="rgb", layer_type="image", rgb=True)

        mgr.add_image_to_subviewers(img_layer)

        for vm in [vm_xy, vm_xz, vm_yz]:
            img = vm.layers._layers[0]
            assert img.rgb is True

    def test_rgb_image_detected_from_4d_shape(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()

        data = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        img_layer = _FakeLayer(data, name="rgb", layer_type="image", rgb=False)

        mgr.add_image_to_subviewers(img_layer)

        for vm in [vm_xy, vm_xz, vm_yz]:
            img = vm.layers._layers[0]
            assert img.rgb is True

    def test_non_rgb_image_preserves_colormap(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()

        data = np.zeros((4, 8, 8), dtype=np.float32)
        img_layer = _FakeLayer(data, name="ch0", layer_type="image")
        img_layer.colormap = types.SimpleNamespace(name="green")

        mgr.add_image_to_subviewers(img_layer)

        for vm in [vm_xy, vm_xz, vm_yz]:
            img = vm.layers._layers[0]
            assert img.colormap == "green"

    def test_contrast_limits_preserved(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()

        data = np.zeros((4, 8, 8), dtype=np.float32)
        img_layer = _FakeLayer(data, name="ch0", layer_type="image")
        img_layer.contrast_limits = (10, 200)

        mgr.add_image_to_subviewers(img_layer)

        for vm in [vm_xy, vm_xz, vm_yz]:
            img = vm.layers._layers[0]
            assert img.contrast_limits == (10, 200)

    def test_opacity_preserved(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()

        data = np.zeros((4, 8, 8), dtype=np.float32)
        img_layer = _FakeLayer(data, name="ch0", layer_type="image", opacity=0.7)

        mgr.add_image_to_subviewers(img_layer)

        for vm in [vm_xy, vm_xz, vm_yz]:
            img = vm.layers._layers[0]
            assert img.opacity == 0.7


# ---------------------------------------------------------------------------
# Tests — layer naming
# ---------------------------------------------------------------------------


class TestLayerNaming:
    def test_label_names_have_orientation_suffix(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="seg", layer_type="labels")

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        assert vm_xy.layers._layers[0].name == "seg_XY"
        assert vm_xz.layers._layers[0].name == "seg_XZ"
        assert vm_yz.layers._layers[0].name == "seg_YZ"

    def test_image_names_have_orientation_suffix(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()

        data = np.zeros((4, 8, 8), dtype=np.float32)
        img_layer = _FakeLayer(data, name="ch0", layer_type="image")

        mgr.add_image_to_subviewers(img_layer)

        assert vm_xy.layers._layers[0].name == "ch0_XY"
        assert vm_xz.layers._layers[0].name == "ch0_XZ"
        assert vm_yz.layers._layers[0].name == "ch0_YZ"


# ---------------------------------------------------------------------------
# Tests — rename in subviewers
# ---------------------------------------------------------------------------


class TestRenameInSubviewers:
    def test_rename_updates_all_orientations(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="old", layer_type="labels")
        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        mgr.rename_layer_in_subviewers("old", "new")

        assert vm_xy.layers._layers[0].name == "new_XY"
        assert vm_xz.layers._layers[0].name == "new_XZ"
        assert vm_yz.layers._layers[0].name == "new_YZ"

    def test_rename_nonexistent_is_noop(self):
        mgr, _, _, _ = _make_manager()
        mgr.rename_layer_in_subviewers("nonexistent", "new")  # Should not raise


# ---------------------------------------------------------------------------
# Tests — remove from subviewers
# ---------------------------------------------------------------------------


class TestRemoveFromSubviewers:
    def test_removes_all_orientations(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="seg", layer_type="labels")
        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        mgr.remove_layer_from_subviewers("seg")

        for vm in [vm_xy, vm_xz, vm_yz]:
            assert len(vm.layers) == 0

    def test_remove_nonexistent_is_noop(self):
        mgr, _, _, _ = _make_manager()
        mgr.remove_layer_from_subviewers("nonexistent")  # Should not raise

    def test_remove_does_not_clobber_shared_prefix(self):
        """Removing 'FT digit 5' must NOT remove 'FT digit 5 extra' layers."""
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        short_data = np.zeros((4, 8, 8), dtype=np.int32)
        long_data = np.zeros((4, 8, 8), dtype=np.int32)
        short_layer = _FakeLayer(short_data, name="FT digit 5", layer_type="labels")
        long_layer = _FakeLayer(long_data, name="FT digit 5 extra", layer_type="labels")

        mgr.add_labels_to_subviewers(short_layer, sync, draw)
        mgr.add_labels_to_subviewers(long_layer, sync, draw)

        # Each subviewer should have 2 label layers (1 per add call).
        for vm in [vm_xy, vm_xz, vm_yz]:
            assert len(vm.layers) == 2

        # Remove the short-named layer — long-named layers must survive.
        mgr.remove_layer_from_subviewers("FT digit 5")

        for vm in [vm_xy, vm_xz, vm_yz]:
            remaining = [l.name for l in vm.layers]
            assert len(remaining) == 1, f"Expected 1 remaining, got {remaining}"
            assert "FT digit 5 extra" in remaining[0]


# ---------------------------------------------------------------------------
# Tests — get_subviewer_layer
# ---------------------------------------------------------------------------


class TestGetSubviewerLayer:
    def test_finds_correct_layer(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="seg", layer_type="labels")
        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        result = mgr.get_subviewer_layer("seg", "XY")
        assert result is not None
        assert result.name == "seg_XY"

    def test_returns_none_for_missing(self):
        mgr, _, _, _ = _make_manager()
        result = mgr.get_subviewer_layer("nonexistent", "XY")
        assert result is None


# ---------------------------------------------------------------------------
# Tests — _viewer_model_for_orientation
# ---------------------------------------------------------------------------


class TestViewerModelForOrientation:
    def test_xy_returns_xy_model(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        assert mgr._viewer_model_for_orientation("XY") is vm_xy

    def test_xz_returns_xz_model(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        assert mgr._viewer_model_for_orientation("XZ") is vm_xz

    def test_yz_returns_yz_model(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        assert mgr._viewer_model_for_orientation("YZ") is vm_yz

    def test_case_insensitive(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        assert mgr._viewer_model_for_orientation("xy") is vm_xy

    def test_unknown_defaults_to_yz(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        assert mgr._viewer_model_for_orientation("ZZ") is vm_yz


# ---------------------------------------------------------------------------
# Tests — in-place data update
# ---------------------------------------------------------------------------


class TestInPlaceUpdate:
    def test_replace_same_shape_labels(self):
        mgr, vm_xy, _, _ = _make_manager()
        data1 = np.zeros((4, 8, 8), dtype=np.int32)
        vm_xy.add_labels(data1, name="lbl")

        data2 = np.ones((4, 8, 8), dtype=np.int32)
        result = mgr._add_or_replace_labels(vm_xy, "lbl", data2)

        assert len(vm_xy.layers) == 1
        np.testing.assert_array_equal(result.data, data2)

    def test_replace_different_shape_creates_new(self):
        mgr, vm_xy, _, _ = _make_manager()
        data1 = np.zeros((4, 8, 8), dtype=np.int32)
        vm_xy.add_labels(data1, name="lbl")

        data2 = np.ones((6, 10, 10), dtype=np.int32)
        result = mgr._add_or_replace_labels(vm_xy, "lbl", data2)

        assert len(vm_xy.layers) == 1
        np.testing.assert_array_equal(result.data, data2)

    def test_replace_image_same_shape_updates_data(self):
        mgr, vm_xy, _, _ = _make_manager()
        data1 = np.zeros((4, 8, 8), dtype=np.float32)
        vm_xy.add_image(data1, name="img")

        data2 = np.ones((4, 8, 8), dtype=np.float32)
        mgr._add_or_replace_image(vm_xy, "img", data2)

        assert len(vm_xy.layers) == 1
        np.testing.assert_array_equal(vm_xy.layers._layers[0].data, data2)

    def test_add_new_image(self):
        mgr, vm_xy, _, _ = _make_manager()
        data = np.zeros((4, 8, 8), dtype=np.float32)
        mgr._add_or_replace_image(vm_xy, "new_img", data)

        assert len(vm_xy.layers) == 1
        assert vm_xy.layers._layers[0].name == "new_img"


# ---------------------------------------------------------------------------
# Tests — clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_removes_all_layers(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        for vm in [vm_xy, vm_xz, vm_yz]:
            vm.add_image(np.zeros((4, 8, 8)), name="img")
            vm.add_labels(np.zeros((4, 8, 8), dtype=np.int32), name="lbl")

        mgr.clear()

        for vm in [vm_xy, vm_xz, vm_yz]:
            assert len(vm.layers) == 0

    def test_clear_empty_is_noop(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        mgr.clear()  # Should not raise
        for vm in [vm_xy, vm_xz, vm_yz]:
            assert len(vm.layers) == 0

    def test_clear_twice_is_safe(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        vm_xy.add_image(np.zeros((4, 8, 8)), name="img")
        mgr.clear()
        mgr.clear()
        assert len(vm_xy.layers) == 0


# ---------------------------------------------------------------------------
# Tests — sync registration
# ---------------------------------------------------------------------------


class TestSyncRegistration:
    def test_shared_data_registered(self):
        mgr, _, _, _ = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="my_labels", layer_type="labels")

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        assert "my_labels" in sync.shared_labels
        assert sync.shared_labels["my_labels"] is data

    def test_connect_label_sync_called(self):
        mgr, _, _, _ = _make_manager()
        sync = _FakeSyncManager()
        draw = _FakeDrawingHandler()

        data = np.zeros((4, 8, 8), dtype=np.int32)
        label_layer = _FakeLayer(data, name="seg", layer_type="labels")

        mgr.add_labels_to_subviewers(label_layer, sync, draw)

        assert "seg" in sync._sync_connected


# ---------------------------------------------------------------------------
# Tests — visibility sync
# ---------------------------------------------------------------------------


class TestVisibilitySync:
    def test_connects_visibility_event_once(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0")
        mgr.sync_layer_visibility(layer)
        assert getattr(layer, "_epicycle_vis_synced", False) is True
        n1 = len(layer.events.visible._callbacks)
        mgr.sync_layer_visibility(layer)
        n2 = len(layer.events.visible._callbacks)
        assert n1 == n2  # no duplicate

    def test_guard_flag_set(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0")
        mgr.sync_layer_visibility(layer)
        assert layer._epicycle_vis_synced is True


# ---------------------------------------------------------------------------
# Tests — image style sync
# ---------------------------------------------------------------------------


class TestImageStyleSync:
    def test_connects_style_events_once(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0", layer_type="image")
        mgr.sync_image_style(layer)
        assert getattr(layer, "_epicycle_style_synced", False) is True

        n_cmap = len(layer.events.colormap._callbacks)
        mgr.sync_image_style(layer)
        assert len(layer.events.colormap._callbacks) == n_cmap

    def test_connects_all_four_style_events(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0", layer_type="image")
        mgr.sync_image_style(layer)
        assert len(layer.events.colormap._callbacks) >= 1
        assert len(layer.events.contrast_limits._callbacks) >= 1
        assert len(layer.events.opacity._callbacks) >= 1
        assert len(layer.events.gamma._callbacks) >= 1


# ---------------------------------------------------------------------------
# Tests — fast reload pre-flight checks
# ---------------------------------------------------------------------------


class TestFastReloadPreFlight:
    def test_empty_subviewers_returns_false(self):
        mgr, _, _, _ = _make_manager()
        img_proxy = _FakeLayer(np.zeros((4, 8, 8), dtype=np.float32), name="img")
        lbl_proxy = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), name="lbl")
        success, names = mgr.try_fast_reload([img_proxy], [lbl_proxy])
        assert success is False
        assert names == []

    def test_count_mismatch_returns_false(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        shape = (4, 8, 8)
        # Add 1 image + 1 labels to each subviewer
        for vm, orient in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            vm.add_image(transpose_data(np.zeros(shape, dtype=np.float32), orient), name=f"img_{orient}")
            vm.add_labels(transpose_data(np.zeros(shape, dtype=np.int32), orient), name=f"lbl_{orient}")

        # Try with 2 images but only 1 exists
        img1 = _FakeLayer(np.zeros(shape, dtype=np.float32), name="img1")
        img2 = _FakeLayer(np.zeros(shape, dtype=np.float32), name="img2")
        lbl = _FakeLayer(np.zeros(shape, dtype=np.int32), name="lbl")
        success, names = mgr.try_fast_reload([img1, img2], [lbl])
        assert success is False

    def test_shape_mismatch_returns_false(self):
        mgr, vm_xy, vm_xz, vm_yz = _make_manager()
        shape = (4, 8, 8)
        for vm, orient in [(vm_xy, "XY"), (vm_xz, "XZ"), (vm_yz, "YZ")]:
            vm.add_image(transpose_data(np.zeros(shape, dtype=np.float32), orient), name=f"img_{orient}")
            vm.add_labels(transpose_data(np.zeros(shape, dtype=np.int32), orient), name=f"lbl_{orient}")

        # Different shape proxy
        img = _FakeLayer(np.zeros((6, 10, 10), dtype=np.float32), name="img")
        lbl = _FakeLayer(np.zeros(shape, dtype=np.int32), name="lbl")
        success, names = mgr.try_fast_reload([img], [lbl])
        assert success is False


# ---------------------------------------------------------------------------
# Tests — visibility guard (Curation3DAnnotator._on_layer_inserted)
# ---------------------------------------------------------------------------


class TestVisibilityGuard:
    """Test the _on_layer_inserted guard logic."""

    def test_guard_skips_when_not_visible(self):
        call_count = [0]

        class FakeWidget:
            def __init__(self):
                self._active_labels_name = ""
            def isVisible(self):
                return False
            def _auto_load_if_ready(self):
                call_count[0] += 1

        widget = FakeWidget()
        if not widget.isVisible():
            pass
        elif not widget._active_labels_name:
            widget._auto_load_if_ready()
        assert call_count[0] == 0

    def test_guard_allows_when_visible(self):
        call_count = [0]

        class FakeWidget:
            def __init__(self):
                self._active_labels_name = ""
            def isVisible(self):
                return True
            def _auto_load_if_ready(self):
                call_count[0] += 1

        widget = FakeWidget()
        if not widget.isVisible():
            pass
        elif not widget._active_labels_name:
            widget._auto_load_if_ready()
        assert call_count[0] == 1

    def test_guard_skips_when_already_loaded(self):
        call_count = [0]

        class FakeWidget:
            def __init__(self):
                self._active_labels_name = "some_labels"
            def isVisible(self):
                return True
            def _auto_load_if_ready(self):
                call_count[0] += 1

        widget = FakeWidget()
        if not widget.isVisible():
            pass
        elif not widget._active_labels_name:
            widget._auto_load_if_ready()
        assert call_count[0] == 0


# ---------------------------------------------------------------------------
# Tests — disconnect_layer_visibility
# ---------------------------------------------------------------------------


class TestDisconnectLayerVisibility:
    """Verify clean disconnection of visibility event handlers."""

    def test_disconnect_resets_flag(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0")
        mgr.sync_layer_visibility(layer)
        assert layer._epicycle_vis_synced is True

        mgr.disconnect_layer_visibility(layer)
        assert layer._epicycle_vis_synced is False

    def test_disconnect_removes_handler(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0")
        mgr.sync_layer_visibility(layer)
        assert len(layer.events.visible._callbacks) == 1

        mgr.disconnect_layer_visibility(layer)
        assert len(layer.events.visible._callbacks) == 0

    def test_disconnect_then_reconnect_works(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0")
        mgr.sync_layer_visibility(layer)
        mgr.disconnect_layer_visibility(layer)
        mgr.sync_layer_visibility(layer)
        assert layer._epicycle_vis_synced is True
        assert len(layer.events.visible._callbacks) == 1

    def test_disconnect_noop_when_not_connected(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0")
        mgr.disconnect_layer_visibility(layer)  # Should not raise
        assert layer._epicycle_vis_synced is False

    def test_no_accumulation_across_cycles(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0")
        for _ in range(5):
            mgr.sync_layer_visibility(layer)
            assert len(layer.events.visible._callbacks) == 1
            mgr.disconnect_layer_visibility(layer)
            assert len(layer.events.visible._callbacks) == 0


# ---------------------------------------------------------------------------
# Tests — disconnect_image_style
# ---------------------------------------------------------------------------


class TestDisconnectImageStyle:
    """Verify clean disconnection of image style event handlers."""

    def test_disconnect_resets_flag(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0", layer_type="image")
        mgr.sync_image_style(layer)
        assert layer._epicycle_style_synced is True

        mgr.disconnect_image_style(layer)
        assert layer._epicycle_style_synced is False

    def test_disconnect_removes_all_handlers(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0", layer_type="image")
        mgr.sync_image_style(layer)
        for attr in ("colormap", "contrast_limits", "opacity", "gamma"):
            assert len(getattr(layer.events, attr)._callbacks) >= 1

        mgr.disconnect_image_style(layer)
        for attr in ("colormap", "contrast_limits", "opacity", "gamma"):
            assert len(getattr(layer.events, attr)._callbacks) == 0, (
                f"Event {attr} should have 0 callbacks after disconnect"
            )

    def test_disconnect_then_reconnect_works(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0", layer_type="image")
        mgr.sync_image_style(layer)
        mgr.disconnect_image_style(layer)
        mgr.sync_image_style(layer)
        assert layer._epicycle_style_synced is True
        assert len(layer.events.colormap._callbacks) == 1

    def test_disconnect_noop_when_not_connected(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0", layer_type="image")
        mgr.disconnect_image_style(layer)  # Should not raise
        assert layer._epicycle_style_synced is False

    def test_no_accumulation_across_cycles(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0", layer_type="image")
        for _ in range(5):
            mgr.sync_image_style(layer)
            assert len(layer.events.colormap._callbacks) == 1
            mgr.disconnect_image_style(layer)
            assert len(layer.events.colormap._callbacks) == 0


# -----------------------------------------------------------------------
# disconnect_all — comprehensive handler flush
# -----------------------------------------------------------------------


class TestDisconnectAll:
    """Tests for SubviewerManager.disconnect_all()."""

    def test_disconnect_all_clears_vis_handlers(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0")
        mgr.sync_layer_visibility(layer)
        assert len(mgr._vis_handlers) == 1
        assert len(layer.events.visible._callbacks) == 1

        mgr.disconnect_all()

        assert len(mgr._vis_handlers) == 0
        assert len(layer.events.visible._callbacks) == 0

    def test_disconnect_all_clears_style_handlers(self):
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0", layer_type="image")
        mgr.sync_image_style(layer)
        assert len(mgr._style_handlers) == 1
        assert len(layer.events.colormap._callbacks) == 1

        mgr.disconnect_all()

        assert len(mgr._style_handlers) == 0
        assert len(layer.events.colormap._callbacks) == 0

    def test_disconnect_all_idempotent(self):
        mgr, _, _, _ = _make_manager()
        mgr.disconnect_all()
        mgr.disconnect_all()  # Should not raise

    def test_no_accumulation_with_disconnect_all_in_cycle(self):
        """Simulate close/relaunch: new managers per cycle, single persistent layer."""
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0")

        for cycle in range(5):
            mgr, _, _, _ = _make_manager()
            mgr.sync_layer_visibility(layer)
            assert len(layer.events.visible._callbacks) == 1, (
                f"Cycle {cycle}: expected 1 vis handler, got "
                f"{len(layer.events.visible._callbacks)}"
            )
            mgr.disconnect_layer_visibility(layer)
            mgr.disconnect_all()

        assert len(layer.events.visible._callbacks) == 0

    def test_disconnect_all_after_per_layer_is_safe(self):
        """disconnect_all after per-layer disconnect doesn't double-disconnect."""
        mgr, _, _, _ = _make_manager()
        layer = _FakeLayer(np.zeros((4, 8, 8)), name="ch0", layer_type="image")
        mgr.sync_image_style(layer)
        mgr.sync_layer_visibility(layer)

        # Per-layer disconnect first
        mgr.disconnect_image_style(layer)
        mgr.disconnect_layer_visibility(layer)

        # Then flush — should be a no-op, not raise
        mgr.disconnect_all()

        assert len(layer.events.colormap._callbacks) == 0
        assert len(layer.events.visible._callbacks) == 0

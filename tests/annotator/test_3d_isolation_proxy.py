"""Tests for Phase 2 — 3D isolation proxy display layer.

Verifies that when ndisplay=3 and single-label isolation is active,
a small proxy display layer replaces the main layer for 3D rendering,
enabling fast isosurface extraction on a cropped ROI instead of the
full volume.

Source layer stays intact for editing.  Proxy is display-only.

All tests are Qt-free using source inspection + lightweight fakes.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest


# -----------------------------------------------------------------------
# Source inspection: proxy methods exist and are wired
# -----------------------------------------------------------------------


class TestProxyMethodsExist:
    """Structural: proxy helper methods should exist on Curation3DAnnotator."""

    def test_update_isolation_proxy_exists(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        assert hasattr(Curation3DAnnotator, "_update_isolation_proxy"), (
            "_update_isolation_proxy method should exist"
        )

    def test_remove_isolation_proxy_exists(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        assert hasattr(Curation3DAnnotator, "_remove_isolation_proxy"), (
            "_remove_isolation_proxy method should exist"
        )


class TestProxyWiring:
    """Structural: proxy logic should be wired into isolation + cleanup."""

    def test_on_isolation_changed_references_proxy(self):
        """_on_isolation_changed should delegate to proxy in 3D."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_isolation_changed
        )
        assert "_update_isolation_proxy" in source or "_isolation_proxy" in source, (
            "_on_isolation_changed should reference proxy logic for 3D path"
        )

    def test_on_isolation_changed_checks_ndisplay(self):
        """_on_isolation_changed should check ndisplay for proxy decision."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_isolation_changed
        )
        assert "ndisplay" in source, (
            "_on_isolation_changed should check ndisplay for proxy vs "
            "show_selected_label decision"
        )

    def test_close_cleanup_references_proxy(self):
        """_close_cleanup should clean up proxy layer."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._close_cleanup
        )
        assert "_isolation_proxy" in source or "_remove_isolation_proxy" in source, (
            "_close_cleanup should handle proxy cleanup"
        )


# -----------------------------------------------------------------------
# Behavioral tests: proxy creation / update / removal
# -----------------------------------------------------------------------


class _FakeLayerList(list):
    """Minimal fake for napari viewer.layers that supports name lookup."""

    def __contains__(self, item):
        if isinstance(item, str):
            return any(getattr(l, "name", None) == item for l in self)
        return super().__contains__(item)

    def __getitem__(self, key):
        if isinstance(key, str):
            for l in self:
                if getattr(l, "name", None) == key:
                    return l
            raise KeyError(key)
        return super().__getitem__(key)

    def __iter__(self):
        return list.__iter__(self)

    def remove(self, layer):
        list.remove(self, layer)


def _make_proxy_test_env(*, ndisplay=3, label_id=5, volume_shape=(20, 20, 20)):
    """Create a test environment for proxy behavioral tests.

    Returns (annotator_fake, main_layer, viewer).
    """
    from napari.layers import Labels as _Labels
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    # Create volume with a small label
    data = np.zeros(volume_shape, dtype=np.int32)
    # Place label in a small region (5x5x5)
    data[5:10, 5:10, 5:10] = label_id
    data[12:15, 12:15, 12:15] = 99  # Another label

    layer = _Labels(data, name="labels")
    layer.selected_label = label_id

    layers = _FakeLayerList([layer])
    dims = types.SimpleNamespace(ndisplay=ndisplay)
    viewer = types.SimpleNamespace(layers=layers, dims=dims)

    ann = types.SimpleNamespace()
    ann.viewer = viewer
    ann._active_labels_name = "labels"
    ann._isolation_proxy_layer = None
    ann._isolation_proxy_active = False
    ann._isolation_proxy_label_id = None

    ann._layer_by_name = types.MethodType(
        Curation3DAnnotator._layer_by_name, ann
    )
    ann._update_isolation_proxy = types.MethodType(
        Curation3DAnnotator._update_isolation_proxy, ann
    )
    ann._remove_isolation_proxy = types.MethodType(
        Curation3DAnnotator._remove_isolation_proxy, ann
    )

    return ann, layer, viewer


class TestProxyCreation:
    """_update_isolation_proxy should create a proxy with ROI data."""

    def test_creates_proxy_layer(self):
        """Proxy layer should be added to viewer.layers."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)

        assert ann._isolation_proxy_layer is not None
        assert ann._isolation_proxy_active is True
        assert ann._isolation_proxy_label_id == 5

    def test_proxy_is_smaller_than_main(self):
        """Proxy data should be a cropped ROI, not the full volume."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)

        proxy = ann._isolation_proxy_layer
        # Main is 20x20x20 = 8000, proxy should be much smaller
        assert proxy.data.size < main_layer.data.size, (
            f"Proxy ({proxy.data.shape}) should be smaller than main "
            f"({main_layer.data.shape})"
        )

    def test_proxy_contains_target_label(self):
        """Proxy data should contain the target label."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)

        proxy = ann._isolation_proxy_layer
        assert 5 in proxy.data, "Proxy should contain the target label"

    def test_proxy_is_not_editable(self):
        """Proxy layer should be non-editable (display only)."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)

        proxy = ann._isolation_proxy_layer
        assert proxy.editable is False

    def test_proxy_has_show_selected_label(self):
        """Proxy should have show_selected_label=True."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)

        proxy = ann._isolation_proxy_layer
        assert proxy.show_selected_label is True
        assert proxy.selected_label == 5

    def test_main_layer_hidden_when_proxy_active(self):
        """Main layer should be hidden when proxy is active."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)

        assert main_layer.visible is False

    def test_proxy_added_to_viewer_layers(self):
        """Proxy layer should be in viewer.layers."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)

        proxy = ann._isolation_proxy_layer
        assert proxy in viewer.layers

    def test_proxy_has_correct_translate(self):
        """Proxy translate should offset to ROI position in volume."""
        from napari.layers import Labels as _Labels

        # Create volume with label far from origin
        data = np.zeros((50, 50, 50), dtype=np.int32)
        data[30:35, 30:35, 30:35] = 42  # Far from origin
        layer = _Labels(data, name="labels")

        ann, _, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann.viewer.layers = _FakeLayerList([layer])
        ann._update_isolation_proxy(42, layer)

        proxy = ann._isolation_proxy_layer
        translate = list(proxy.translate)
        # Label at [30:35], margin=5 → ROI starts at 25
        assert all(t >= 25 for t in translate), (
            f"Proxy translate {translate} should offset to ROI position"
        )


class TestProxyUpdate:
    """Switching labels should update proxy with new ROI."""

    def test_update_changes_label_id(self):
        """Switching labels should update proxy label tracking."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)
        assert ann._isolation_proxy_label_id == 5

        # Switch to label 99
        ann._update_isolation_proxy(99, main_layer)
        assert ann._isolation_proxy_label_id == 99

    def test_update_reuses_proxy_layer(self):
        """Updating should reuse the same proxy layer, not create a new one."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)
        proxy1 = ann._isolation_proxy_layer

        ann._update_isolation_proxy(99, main_layer)
        proxy2 = ann._isolation_proxy_layer

        assert proxy1 is proxy2, "Should reuse same proxy layer object"

    def test_noop_same_label(self):
        """Updating with same label should be a no-op."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)
        proxy = ann._isolation_proxy_layer
        old_data = proxy.data.copy()

        # Call again with same label
        ann._update_isolation_proxy(5, main_layer)
        # Should still be same proxy
        assert ann._isolation_proxy_layer is proxy


class TestProxyRemoval:
    """_remove_isolation_proxy should clean up and restore main layer."""

    def test_removes_proxy_from_viewer(self):
        """Proxy should be removed from viewer.layers."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)
        assert ann._isolation_proxy_layer in viewer.layers

        ann._remove_isolation_proxy()
        assert ann._isolation_proxy_layer is None
        assert ann._isolation_proxy_active is False

    def test_restores_main_layer_visibility(self):
        """Main layer should be visible again after proxy removal."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)
        assert main_layer.visible is False

        ann._remove_isolation_proxy()
        assert main_layer.visible is True

    def test_remove_when_no_proxy_is_safe(self):
        """Removing when no proxy exists should not raise."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        # No proxy created
        ann._remove_isolation_proxy()  # Should not raise
        assert ann._isolation_proxy_active is False


# -----------------------------------------------------------------------
# Propagation ordering tests (Task 2)
# -----------------------------------------------------------------------


class _OrderTrackingLayer:
    """Records the order of attribute assignments for verification."""

    def __init__(self, name=""):
        self.name = name
        self._selected_label = 1
        self._show_selected_label = False
        self._assignment_order = []

    @property
    def selected_label(self):
        return self._selected_label

    @selected_label.setter
    def selected_label(self, value):
        self._selected_label = value
        self._assignment_order.append(("selected_label", value))

    @property
    def show_selected_label(self):
        return self._show_selected_label

    @show_selected_label.setter
    def show_selected_label(self, value):
        self._show_selected_label = value
        self._assignment_order.append(("show_selected_label", value))


# Register so isinstance checks pass
from napari.layers import Labels as _LabelsReg

_LabelsReg.register(_OrderTrackingLayer)


class TestPropagateShowSelectedLabelOrdering:
    """show_selected_label must be set BEFORE selected_label on subviewers.

    Setting selected_label fires sync events that can cascade.  If
    show_selected_label hasn't been set yet, the sync cascade reads
    the old value and propagates it — stomping the intended state.
    """

    def test_show_selected_set_before_selected_label(self):
        """On each subviewer layer, show_selected_label must be assigned
        before selected_label when both are being set."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        # Build minimal fake environment
        tracking_layers = []
        for suffix in ("XY", "XZ", "YZ"):
            lyr = _OrderTrackingLayer(name=f"seg_{suffix}")
            tracking_layers.append(lyr)

        # Fake sync manager with viewer models containing tracking layers
        sync = types.SimpleNamespace()
        sync._base_label_name = lambda name: name.replace("_XY", "").replace("_XZ", "").replace("_YZ", "")

        fake_vm_xy = types.SimpleNamespace(layers=tracking_layers[0:1])
        fake_vm_xz = types.SimpleNamespace(layers=tracking_layers[1:2])
        fake_vm_yz = types.SimpleNamespace(layers=tracking_layers[2:3])
        sync.viewer_model_xy = fake_vm_xy
        sync.viewer_model_xz = fake_vm_xz
        sync.viewer_model_yz = fake_vm_yz

        ann = types.SimpleNamespace()
        ann._active_labels_name = "seg"
        ann.sync_manager = sync
        ann._propagate_show_selected_label = types.MethodType(
            Curation3DAnnotator._propagate_show_selected_label, ann
        )

        ann._propagate_show_selected_label(True, label_id=42)

        for lyr in tracking_layers:
            order = lyr._assignment_order
            assert len(order) == 2, (
                f"{lyr.name}: expected 2 assignments, got {order}"
            )
            assert order[0][0] == "show_selected_label", (
                f"{lyr.name}: show_selected_label must be set FIRST, "
                f"but got {order[0][0]}"
            )
            assert order[1][0] == "selected_label", (
                f"{lyr.name}: selected_label must be set SECOND, "
                f"but got {order[1][0]}"
            )

    def test_propagate_always_assigns_to_trigger_rerender(self):
        """Even if show_selected_label matches, always assign to force
        napari to fire the property-change event and re-render."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        lyr = _OrderTrackingLayer(name="seg_XY")
        lyr._show_selected_label = True  # Already True

        sync = types.SimpleNamespace()
        sync._base_label_name = lambda name: name.replace("_XY", "").replace("_XZ", "").replace("_YZ", "")
        sync.viewer_model_xy = types.SimpleNamespace(layers=[lyr])
        sync.viewer_model_xz = types.SimpleNamespace(layers=[])
        sync.viewer_model_yz = types.SimpleNamespace(layers=[])

        ann = types.SimpleNamespace()
        ann._active_labels_name = "seg"
        ann.sync_manager = sync
        ann._propagate_show_selected_label = types.MethodType(
            Curation3DAnnotator._propagate_show_selected_label, ann
        )

        ann._propagate_show_selected_label(True, label_id=42)

        # show_selected_label should ALWAYS be assigned (even if same value)
        # to ensure napari fires the re-render event
        show_assigns = [a for a in lyr._assignment_order if a[0] == "show_selected_label"]
        assert len(show_assigns) == 1, (
            f"show_selected_label must always be assigned: {lyr._assignment_order}"
        )


class TestSyncShowSelectedLabelGuard:
    """_sync_show_selected_label must be blocked when suppress flag is set,
    even before _isolation_proxy_active is True."""

    def test_blocked_by_suppress_flag_before_proxy_active(self):
        """The suppress flag on sync_manager should block
        _sync_show_selected_label even if _isolation_proxy_active is False."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        source = inspect.getsource(
            Curation3DAnnotator._sync_show_selected_label
        )
        assert "_suppress_show_selected_sync" in source, (
            "_sync_show_selected_label must check the suppress flag "
            "on sync_manager, not just _isolation_proxy_active"
        )


# -----------------------------------------------------------------------
# Suppress flag lifecycle tests (Task 3)
# -----------------------------------------------------------------------


class TestProxySuppressFlagLifecycle:
    """The suppress flag on sync_manager must be set/cleared in all
    proxy lifecycle paths: activate, remove, close, patch switch."""

    def _make_env_with_sync(self, *, ndisplay=3, label_id=5):
        """Create test env with a real-ish sync manager that has the flag."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        ann, main_layer, viewer = _make_proxy_test_env(
            ndisplay=ndisplay, label_id=label_id
        )
        # Add a sync manager with the suppress flag
        sync = types.SimpleNamespace()
        sync._suppress_show_selected_sync = False
        sync._base_label_name = lambda name: name.replace("_XY", "").replace("_XZ", "").replace("_YZ", "")
        sync.viewer_model_xy = types.SimpleNamespace(layers=[])
        sync.viewer_model_xz = types.SimpleNamespace(layers=[])
        sync.viewer_model_yz = types.SimpleNamespace(layers=[])
        ann.sync_manager = sync

        return ann, main_layer, viewer, sync

    def test_proxy_activate_sets_suppress_flag(self):
        """Creating proxy must set _suppress_show_selected_sync = True."""
        ann, main_layer, viewer, sync = self._make_env_with_sync()
        assert sync._suppress_show_selected_sync is False

        ann._update_isolation_proxy(5, main_layer)

        assert sync._suppress_show_selected_sync is True, (
            "Proxy activation must set suppress flag on sync manager"
        )

    def test_proxy_remove_clears_suppress_flag(self):
        """Removing proxy must clear _suppress_show_selected_sync."""
        ann, main_layer, viewer, sync = self._make_env_with_sync()
        ann._update_isolation_proxy(5, main_layer)
        assert sync._suppress_show_selected_sync is True

        ann._remove_isolation_proxy()

        assert sync._suppress_show_selected_sync is False, (
            "Proxy removal must clear suppress flag"
        )

    def test_remove_without_proxy_clears_flag_safely(self):
        """Remove when no proxy is active should still clear flag."""
        ann, main_layer, viewer, sync = self._make_env_with_sync()
        sync._suppress_show_selected_sync = True  # Force to True

        ann._remove_isolation_proxy()

        assert sync._suppress_show_selected_sync is False


# -----------------------------------------------------------------------
# End-to-end integration tests (Task 4)
# -----------------------------------------------------------------------


class TestEndToEndProxySyncIsolation:
    """Full chain: proxy active → sync fires → subviewer state preserved."""

    def _make_full_env(self):
        """Build env with sync manager, main layer, and subviewer layers."""
        from packages.epicycle_annotator.sync import LabelSyncManager
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        # Main viewer
        data = np.zeros((20, 20, 20), dtype=np.int32)
        data[5:10, 5:10, 5:10] = 5
        data[12:15, 12:15, 12:15] = 99

        main_layer = _FakeLabelsLayer(data, name="seg")

        main_layers = _FakeLayerList([main_layer])
        dims = types.SimpleNamespace(ndisplay=3)
        main_viewer = types.SimpleNamespace(layers=main_layers, dims=dims)

        # Subviewer models with label layers
        sub_layers = {}
        vms = {}
        for suffix in ("XY", "XZ", "YZ"):
            sub = _FakeLabelsLayer(data, name=f"seg_{suffix}")
            sub.show_selected_label = False
            sub.selected_label = 1
            vm = types.SimpleNamespace(layers=[sub])
            sub_layers[suffix] = sub
            vms[suffix] = vm

        sync = LabelSyncManager(
            main_viewer=main_viewer,
            viewer_model_xy=vms["XY"],
            viewer_model_xz=vms["XZ"],
            viewer_model_yz=vms["YZ"],
            qt_viewer_xy=None,
            qt_viewer_xz=None,
            qt_viewer_yz=None,
        )

        return main_layer, sub_layers, sync, main_viewer

    def test_proxy_mode_sync_does_not_stomp_subviewers(self):
        """When suppress flag is ON, firing sync_selected_label with
        main layer (show_selected_label=False) must NOT clear subviewers."""
        main_layer, subs, sync, _ = self._make_full_env()

        # Simulate proxy mode: main has show_sel=False, subviewers have True
        main_layer.show_selected_label = False
        for sub in subs.values():
            sub.show_selected_label = True
            sub.selected_label = 5

        # Set suppress flag (as proxy activation would)
        sync._suppress_show_selected_sync = True

        # Fire sync (as selected_label event would)
        main_layer.selected_label = 5
        sync.sync_selected_label(main_layer)

        for suffix, sub in subs.items():
            assert sub.selected_label == 5, (
                f"seg_{suffix}: selected_label must still sync"
            )
            assert sub.show_selected_label is True, (
                f"seg_{suffix}: show_selected_label must NOT be stomped "
                "by sync when suppress flag is ON"
            )

    def test_2d_mode_sync_propagates_show_sel_normally(self):
        """When suppress flag is OFF (2D mode), sync propagates show_sel normally.
        Regression guard: 2D behavior must not change."""
        main_layer, subs, sync, _ = self._make_full_env()

        main_layer.show_selected_label = True
        main_layer.selected_label = 42
        sync._suppress_show_selected_sync = False  # Normal 2D mode

        sync.sync_selected_label(main_layer)

        for suffix, sub in subs.items():
            assert sub.selected_label == 42
            assert sub.show_selected_label is True, (
                f"seg_{suffix}: 2D sync must propagate show_selected_label normally"
            )

    def test_isolation_clear_allows_sync_to_propagate_false(self):
        """After proxy is removed (suppress cleared), sync should propagate
        show_selected_label=False to clean up subviewer state."""
        main_layer, subs, sync, _ = self._make_full_env()

        # Start in proxy mode
        sync._suppress_show_selected_sync = True
        for sub in subs.values():
            sub.show_selected_label = True

        # Clear proxy (simulates _remove_isolation_proxy)
        sync._suppress_show_selected_sync = False

        # Now sync should propagate False from main
        main_layer.show_selected_label = False
        main_layer.selected_label = 1
        sync.sync_selected_label(main_layer)

        for suffix, sub in subs.items():
            assert sub.show_selected_label is False, (
                f"seg_{suffix}: after suppress cleared, sync must propagate "
                "show_selected_label=False to clean up subviewers"
            )

    def test_suppress_flag_on_sync_then_explicit_propagate_works(self):
        """Full proxy chain: suppress sync, then explicit propagation sets
        show_selected_label=True on subviewers. Mimics the real code path."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        main_layer, subs, sync, _ = self._make_full_env()

        # Suppress sync (proxy mode)
        sync._suppress_show_selected_sync = True

        # Sync fires with main show_sel=False — should not stomp
        main_layer.show_selected_label = False
        main_layer.selected_label = 5
        sync.sync_selected_label(main_layer)

        # Explicit propagation (what _propagate_show_selected_label does)
        for sub in subs.values():
            sub.show_selected_label = True
            sub.selected_label = 5

        # Verify subviewers kept True despite sync
        for suffix, sub in subs.items():
            assert sub.show_selected_label is True
            assert sub.selected_label == 5


class _FakeLabelsLayer:
    """Fake Labels layer for integration tests with sync manager."""

    def __init__(self, data, name=""):
        self.data = data
        self.name = name
        self.mode = "pan_zoom"
        self.selected_label = 1
        self.show_selected_label = False
        self.brush_size = 1
        self.opacity = 1.0
        self.colormap = "default"
        self.blending = "translucent"
        self.n_edit_dimensions = 2
        self.visible = True
        self.editable = True
        self.mouse_drag_callbacks = []
        self.events = types.SimpleNamespace(
            mode=_FakeEvent(),
            selected_label=_FakeEvent(),
            brush_size=_FakeEvent(),
            opacity=_FakeEvent(),
            colormap=_FakeEvent(),
            blending=_FakeEvent(),
            n_edit_dimensions=_FakeEvent(),
        )


class _FakeEvent:
    """Minimal event emitter for integration fakes."""

    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def disconnect(self, cb):
        try:
            self._callbacks.remove(cb)
        except ValueError:
            pass


_LabelsReg.register(_FakeLabelsLayer)


# -----------------------------------------------------------------------
# Subviewer visibility tests — visibility sync stomps subviewers
# -----------------------------------------------------------------------


class TestSubviewerVisibilityDuringProxyIsolation:
    """When proxy isolation hides the main layer (visible=False),
    sync_layer_visibility propagates False to subviewer label layers.

    _propagate_show_selected_label(True) must restore subviewer
    visibility so the isolated label is visible in 2D subviewers.
    """

    def test_propagate_show_true_sets_subviewer_visible(self):
        """_propagate_show_selected_label(True) must set layer.visible=True
        on each subviewer label layer, counteracting visibility sync."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        # Build subviewer layers that are HIDDEN (simulating visibility sync)
        sub_layers = []
        for suffix in ("XY", "XZ", "YZ"):
            lyr = _FakeLabelsLayer(np.zeros((5, 5, 5), dtype=np.int32),
                                   name=f"seg_{suffix}")
            lyr.visible = False  # Simulates visibility sync stomped them
            sub_layers.append(lyr)

        sync = types.SimpleNamespace()
        sync._base_label_name = (
            lambda name: name.replace("_XY", "").replace("_XZ", "").replace("_YZ", "")
        )
        sync.viewer_model_xy = types.SimpleNamespace(layers=sub_layers[0:1])
        sync.viewer_model_xz = types.SimpleNamespace(layers=sub_layers[1:2])
        sync.viewer_model_yz = types.SimpleNamespace(layers=sub_layers[2:3])

        ann = types.SimpleNamespace()
        ann._active_labels_name = "seg"
        ann.sync_manager = sync
        ann._propagate_show_selected_label = types.MethodType(
            Curation3DAnnotator._propagate_show_selected_label, ann
        )

        ann._propagate_show_selected_label(True, label_id=5)

        for lyr in sub_layers:
            assert lyr.visible is True, (
                f"{lyr.name}: must be visible=True after "
                "_propagate_show_selected_label(True)"
            )

    def test_propagate_show_false_does_not_force_visible(self):
        """_propagate_show_selected_label(False) should NOT force
        layer.visible — visibility is managed by the normal sync."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        sub_layers = []
        for suffix in ("XY", "XZ", "YZ"):
            lyr = _FakeLabelsLayer(np.zeros((5, 5, 5), dtype=np.int32),
                                   name=f"seg_{suffix}")
            lyr.visible = True  # Normal state
            sub_layers.append(lyr)

        sync = types.SimpleNamespace()
        sync._base_label_name = (
            lambda name: name.replace("_XY", "").replace("_XZ", "").replace("_YZ", "")
        )
        sync.viewer_model_xy = types.SimpleNamespace(layers=sub_layers[0:1])
        sync.viewer_model_xz = types.SimpleNamespace(layers=sub_layers[1:2])
        sync.viewer_model_yz = types.SimpleNamespace(layers=sub_layers[2:3])

        ann = types.SimpleNamespace()
        ann._active_labels_name = "seg"
        ann.sync_manager = sync
        ann._propagate_show_selected_label = types.MethodType(
            Curation3DAnnotator._propagate_show_selected_label, ann
        )

        ann._propagate_show_selected_label(False)

        # visible should remain as-is (True) — propagate(False) doesn't touch it
        for lyr in sub_layers:
            assert lyr.visible is True, (
                f"{lyr.name}: visible must not be changed by propagate(False)"
            )

    def test_full_chain_proxy_hide_then_propagate_restores_visibility(self):
        """End-to-end: proxy hides main → vis sync hides subviewers →
        _propagate_show_selected_label(True) must restore subviewer
        visibility and set show_selected_label."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        data = np.zeros((20, 20, 20), dtype=np.int32)
        data[5:10, 5:10, 5:10] = 5

        # Build subviewer layers — start visible
        sub_layers = {}
        for suffix in ("XY", "XZ", "YZ"):
            lyr = _FakeLabelsLayer(data, name=f"seg_{suffix}")
            lyr.visible = True
            lyr.show_selected_label = False
            sub_layers[suffix] = lyr

        sync = types.SimpleNamespace()
        sync._base_label_name = (
            lambda name: name.replace("_XY", "").replace("_XZ", "").replace("_YZ", "")
        )
        sync.viewer_model_xy = types.SimpleNamespace(layers=[sub_layers["XY"]])
        sync.viewer_model_xz = types.SimpleNamespace(layers=[sub_layers["XZ"]])
        sync.viewer_model_yz = types.SimpleNamespace(layers=[sub_layers["YZ"]])

        ann = types.SimpleNamespace()
        ann._active_labels_name = "seg"
        ann.sync_manager = sync
        ann._propagate_show_selected_label = types.MethodType(
            Curation3DAnnotator._propagate_show_selected_label, ann
        )

        # Simulate visibility sync stomping subviewers (main_layer.visible=False
        # propagated by sync_layer_visibility)
        for lyr in sub_layers.values():
            lyr.visible = False

        # Now propagate show_selected_label — should restore visibility
        matched = ann._propagate_show_selected_label(True, label_id=5)

        assert matched == 3, f"Expected 3 matched layers, got {matched}"
        for suffix, lyr in sub_layers.items():
            assert lyr.visible is True, (
                f"seg_{suffix}: must be visible after propagate(True)"
            )
            assert lyr.show_selected_label is True, (
                f"seg_{suffix}: show_selected_label must be True"
            )
            assert lyr.selected_label == 5, (
                f"seg_{suffix}: selected_label must be 5"
            )


# -----------------------------------------------------------------------
# Deferred visibility restoration tests — reduce lag on uncheck
# -----------------------------------------------------------------------


class TestDeferredVisibilityRestoration:
    """_remove_isolation_proxy should support deferred visibility restoration
    to avoid expensive 3D re-render before property changes are done."""

    def test_remove_proxy_deferred_keeps_main_hidden(self):
        """With restore_visibility=False, main layer stays hidden."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)
        assert main_layer.visible is False

        ann._remove_isolation_proxy(restore_visibility=False)

        assert main_layer.visible is False, (
            "With restore_visibility=False, main layer must stay hidden"
        )
        assert ann._isolation_proxy_active is False
        assert ann._isolation_proxy_layer is None

    def test_remove_proxy_default_restores_visibility(self):
        """Default behavior (restore_visibility=True) still works."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        ann._update_isolation_proxy(5, main_layer)

        ann._remove_isolation_proxy()  # Default = True

        assert main_layer.visible is True

    def test_remove_proxy_deferred_still_clears_suppress_flag(self):
        """Even when deferring visibility, suppress flag must be cleared."""
        ann, main_layer, viewer = _make_proxy_test_env(ndisplay=3, label_id=5)
        sync = types.SimpleNamespace()
        sync._suppress_show_selected_sync = False
        ann.sync_manager = sync

        ann._update_isolation_proxy(5, main_layer)
        assert sync._suppress_show_selected_sync is True

        ann._remove_isolation_proxy(restore_visibility=False)

        assert sync._suppress_show_selected_sync is False


# -----------------------------------------------------------------------
# Performance: skip refresh on isolation clear
# -----------------------------------------------------------------------


class _RefreshTrackingLayer(_FakeLabelsLayer):
    """Tracks refresh() calls to verify they're skipped when clearing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


_LabelsReg.register(_RefreshTrackingLayer)


class TestPropagateSkipsRefreshOnClear:
    """When show=False (clearing isolation), skip layer.refresh() calls
    to avoid unnecessary subviewer canvas repaints."""

    def _make_env(self):
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        sub_layers = []
        for suffix in ("XY", "XZ", "YZ"):
            lyr = _RefreshTrackingLayer(
                np.zeros((5, 5, 5), dtype=np.int32), name=f"seg_{suffix}"
            )
            sub_layers.append(lyr)

        sync = types.SimpleNamespace()
        sync._base_label_name = (
            lambda name: name.replace("_XY", "").replace("_XZ", "").replace("_YZ", "")
        )
        sync.viewer_model_xy = types.SimpleNamespace(layers=sub_layers[0:1])
        sync.viewer_model_xz = types.SimpleNamespace(layers=sub_layers[1:2])
        sync.viewer_model_yz = types.SimpleNamespace(layers=sub_layers[2:3])

        ann = types.SimpleNamespace()
        ann._active_labels_name = "seg"
        ann.sync_manager = sync
        ann._propagate_show_selected_label = types.MethodType(
            Curation3DAnnotator._propagate_show_selected_label, ann
        )
        return ann, sub_layers

    def test_refresh_called_when_show_true(self):
        """refresh() should be called for each layer when enabling isolation."""
        ann, sub_layers = self._make_env()
        ann._propagate_show_selected_label(True, label_id=5)

        for lyr in sub_layers:
            assert lyr.refresh_count == 1, (
                f"{lyr.name}: refresh should be called once when show=True"
            )

    def test_refresh_skipped_when_show_false(self):
        """refresh() should NOT be called when clearing isolation."""
        ann, sub_layers = self._make_env()
        ann._propagate_show_selected_label(False)

        for lyr in sub_layers:
            assert lyr.refresh_count == 0, (
                f"{lyr.name}: refresh should be skipped when show=False, "
                f"got {lyr.refresh_count} calls"
            )


# -----------------------------------------------------------------------
# Structural: controller colormap propagation to subviewers
# -----------------------------------------------------------------------


class TestControllerColormapPropagation:
    """_suppress_sync_cascade must propagate colormap to annotator
    subviewers after the batch completes."""

    def test_suppress_sync_cascade_calls_propagate(self):
        """After yield, _suppress_sync_cascade should call
        _propagate_colormap_to_annotator_subviewers."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._suppress_sync_cascade
        )
        assert "_propagate_colormap_to_annotator_subviewers" in source, (
            "_suppress_sync_cascade must call "
            "_propagate_colormap_to_annotator_subviewers after batch"
        )

    def test_propagate_method_exists(self):
        """Controller should have _propagate_colormap_to_annotator_subviewers."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        assert hasattr(
            FusionReconstructionController,
            "_propagate_colormap_to_annotator_subviewers",
        )


# -----------------------------------------------------------------------
# Double-application suppression tests
# -----------------------------------------------------------------------


class TestNavigatorIsolationSuppression:
    """When the coordinator dispatches, the navigator must NOT re-trigger
    isolation on the annotator — the controller already applied colormaps."""

    def test_fire_isolation_skipped_when_suppressed(self):
        """_fire_isolation_if_active must return early when
        _suppress_isolation_dispatch is True."""
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )

        source = inspect.getsource(
            CurationLabelNavigator._fire_isolation_if_active
        )
        assert "_suppress_isolation_dispatch" in source, (
            "_fire_isolation_if_active must check _suppress_isolation_dispatch"
        )

    def test_coordinator_adapter_sets_suppress_flag(self):
        """The coordinator's navigator adapter must set
        _suppress_isolation_dispatch around select_labels."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._apply_navigator_selection_adapter
        )
        assert "_suppress_isolation_dispatch" in source, (
            "Navigator adapter must set _suppress_isolation_dispatch "
            "before calling select_labels"
        )


# -----------------------------------------------------------------------
# Performance: deferred 3D re-render via hide-set-show pattern
# -----------------------------------------------------------------------


class TestDeferredIsolationRender:
    """_isolate_labels_in_viewer and _clear_label_isolation should hide
    layers before colormap changes and defer visibility restoration to
    prevent synchronous 3D iso_categorical re-renders."""

    def test_isolate_hides_layers_before_colormap(self):
        """_isolate_labels_in_viewer should hide target layers before
        setting colormaps (structural check)."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._isolate_labels_in_viewer
        )
        # Must hide layers before colormap assignment
        hide_idx = source.find("layer.visible = False")
        # Step 3 plot pass P3: the isolate's colour write is the one path.
        colormap_idx = source.find("_apply_plot_colours_to_viewer")
        assert colormap_idx > 0, (
            "_isolate_labels_in_viewer must colour through the one path"
        )
        assert hide_idx > 0, (
            "_isolate_labels_in_viewer must hide layers (layer.visible = False)"
        )
        assert hide_idx < colormap_idx, (
            "Layers must be hidden BEFORE colormap assignment, "
            "not after"
        )

    def test_isolate_uses_qtimer_for_visibility_restore(self):
        """_isolate_labels_in_viewer should defer visibility restoration
        via QTimer.singleShot to avoid synchronous 3D re-render."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._isolate_labels_in_viewer
        )
        assert "QTimer.singleShot" in source, (
            "_isolate_labels_in_viewer must use QTimer.singleShot for "
            "deferred visibility restoration"
        )

    def test_clear_isolation_hides_layers_before_colormap(self):
        """_clear_label_isolation should hide target layers before
        setting colormaps."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._clear_label_isolation
        )
        hide_idx = source.find("layer.visible = False")
        assert hide_idx > 0, (
            "_clear_label_isolation must hide layers (layer.visible = False)"
        )

    def test_clear_isolation_uses_qtimer_for_visibility_restore(self):
        """_clear_label_isolation should defer visibility restoration
        via QTimer.singleShot."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._clear_label_isolation
        )
        assert "QTimer.singleShot" in source, (
            "_clear_label_isolation must use QTimer.singleShot for "
            "deferred visibility restoration"
        )


# -----------------------------------------------------------------------
# Annotator skips redundant colormap restore when controller handles it
# -----------------------------------------------------------------------


class TestAnnotatorSkipsColormapRestoreForController:
    """When _on_isolation_cleared_cb is set (controller callback exists),
    the annotator should skip its own colormap restore — the controller
    will immediately apply the correct QC feature colormap."""

    def test_isolation_clear_skips_colormap_when_controller_cb_set(self):
        """_on_isolation_changed source should check _on_isolation_cleared_cb
        and skip colormap restore when the controller will handle it."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_isolation_changed
        )
        assert "_controller_will_handle_colormap" in source, (
            "_on_isolation_changed must skip colormap restore when "
            "controller callback is set"
        )

    def test_isolation_clear_still_restores_when_no_controller_cb(self):
        """When no controller callback is set, the annotator must
        restore colormap itself (standalone annotator use case)."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_isolation_changed
        )
        # Must have the fallback path
        assert "not _controller_will_handle_colormap" in source, (
            "_on_isolation_changed must still restore colormap when "
            "no controller callback exists"
        )


# -----------------------------------------------------------------------
# Performance: all colormap paths use deferred 3D re-render
# -----------------------------------------------------------------------


class TestAllColormapPathsDeferred:
    """Every controller method that sets colormaps on visible layers must
    use the hide-set-show + QTimer pattern to avoid synchronous 3D
    iso_categorical re-renders blocking the UI thread."""

    def test_on_annotator_isolation_cleared_hides_layers(self):
        """_on_annotator_isolation_cleared must hide target layers
        before setting colormaps."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._on_annotator_isolation_cleared
        )
        assert "layer.visible = False" in source, (
            "_on_annotator_isolation_cleared must hide layers before "
            "colormap changes"
        )

    def test_on_annotator_isolation_cleared_uses_qtimer(self):
        """_on_annotator_isolation_cleared must defer visibility
        restoration via QTimer."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._on_annotator_isolation_cleared
        )
        assert "QTimer.singleShot" in source, (
            "_on_annotator_isolation_cleared must use QTimer.singleShot "
            "for deferred visibility restoration"
        )

    def test_sync_viewer_colors_toggled_hides_layers(self):
        """_on_sync_viewer_colors_toggled must hide target layers
        before colormap changes."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._on_sync_viewer_colors_toggled
        )
        assert "layer.visible = False" in source, (
            "_on_sync_viewer_colors_toggled must hide layers before "
            "colormap changes"
        )

    def test_sync_viewer_colors_toggled_uses_qtimer(self):
        """_on_sync_viewer_colors_toggled must defer visibility
        restoration via QTimer."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._on_sync_viewer_colors_toggled
        )
        assert "QTimer.singleShot" in source, (
            "_on_sync_viewer_colors_toggled must use QTimer.singleShot "
            "for deferred visibility restoration"
        )

    def test_reset_viewer_label_colors_uses_suppress_cascade(self):
        """_reset_viewer_label_colors must use _suppress_sync_cascade
        to prevent N*3 subviewer cascades."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._reset_viewer_label_colors
        )
        assert "_suppress_sync_cascade" in source, (
            "_reset_viewer_label_colors must use _suppress_sync_cascade "
            "to prevent subviewer cascade"
        )


# -----------------------------------------------------------------------
# Satellite dock cleanup on close/relaunch
# -----------------------------------------------------------------------


class TestSatelliteDockCleanup:
    """Verify that satellite docks (Advanced Features, Multiviewer Settings,
    Label Navigator) are destroyed when the 3D annotator closes."""

    def test_close_cleanup_calls_destroy_satellite_docks(self):
        """_close_cleanup must call _destroy_satellite_docks so floating
        docks don't accumulate across close/relaunch cycles."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(Curation3DAnnotator._close_cleanup)
        assert "_destroy_satellite_docks" in source, (
            "_close_cleanup must call _destroy_satellite_docks to clean up "
            "Advanced Features and Multiviewer Settings docks"
        )

    def test_destroy_satellite_docks_exists_in_launcher(self):
        """_destroy_satellite_docks helper must exist in launcher module."""
        from napari_epicycle.annotator.launcher import (
            _destroy_satellite_docks,
        )

        assert callable(_destroy_satellite_docks)

    def test_destroy_satellite_docks_covers_all_dock_names(self):
        """_destroy_satellite_docks must cover all known satellite dock names."""
        source_mod = inspect.getsource(
            __import__(
                "napari_epicycle.annotator.launcher",
                fromlist=["_destroy_satellite_docks"],
            )._destroy_satellite_docks
        )
        assert "3D Advanced Features" in source_mod or "_ADVANCED_DOCK_NAMES" in source_mod, (
            "Must clean up Advanced Features dock"
        )
        assert "_MULTIVIEWER_SETTINGS_DOCK_NAME" in source_mod, (
            "Must clean up Multiviewer Settings dock"
        )
        assert "_NAVIGATOR_DOCK_NAME" in source_mod, (
            "Must clean up Navigator dock"
        )

    def test_relaunch_cleans_satellite_docks(self):
        """launch_3d_curation_annotator must call _destroy_satellite_docks
        when no visible annotator dock exists (fresh relaunch path)."""
        from napari_epicycle.annotator.launcher import (
            launch_3d_curation_annotator,
        )

        source = inspect.getsource(launch_3d_curation_annotator)
        assert "_destroy_satellite_docks" in source, (
            "launch_3d_curation_annotator must call _destroy_satellite_docks "
            "on fresh relaunch to prevent dock accumulation"
        )

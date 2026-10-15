"""Tests for unified operation view sync — _sync_all_views_after_operation.

Verifies that all operations (interpolation, smooth, split) correctly sync
shared_data changes to the main layer, subviewers, and isolation proxy.

Key design: uses refresh() only — NEVER data=data setter — to avoid
expensive iso_categorical recalculation in 3D mode.

Uses lightweight fakes (no Qt / napari required).
"""

from __future__ import annotations

import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lightweight fakes (no Qt / napari required)
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Minimal event emitter."""

    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def disconnect(self, cb):
        try:
            self._callbacks.remove(cb)
        except ValueError:
            pass


class _FakeEvents:
    """Minimal events container."""

    def __init__(self):
        self.data = _FakeEvent()
        self.mode = _FakeEvent()
        self.selected_label = _FakeEvent()
        self.brush_size = _FakeEvent()
        self.opacity = _FakeEvent()
        self.colormap = _FakeEvent()
        self.blending = _FakeEvent()
        self.name = _FakeEvent()


class _FakeLabelsLayer:
    """Minimal Labels-like layer for sync tests."""

    _IS_LABELS = True

    def __init__(self, data, name="labels"):
        self._data = data
        self.name = name
        self.visible = True
        self.editable = True
        self.mode = "pan_zoom"
        self.selected_label = 1
        self.show_selected_label = False
        self.brush_size = 1
        self.opacity = 1.0
        self.colormap = "auto"  # placeholder for isolation colormap tests
        self.events = _FakeEvents()
        self._refresh_count = 0
        self._data_set_count = 0

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._data_set_count += 1

    def refresh(self):
        self._refresh_count += 1


class _FakeLayerList:
    """Minimal layer list that supports name lookup."""

    def __init__(self, layers=None):
        self._layers = list(layers or [])

    def __contains__(self, name):
        return any(l.name == name for l in self._layers)

    def __getitem__(self, name):
        for l in self._layers:
            if l.name == name:
                return l
        raise KeyError(name)

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)

    def append(self, layer):
        self._layers.append(layer)

    def remove(self, layer):
        self._layers.remove(layer)


class _FakeDims:
    """Minimal dims for ndisplay check."""

    def __init__(self, ndisplay=2):
        self.ndisplay = ndisplay


class _FakeViewerModel:
    """Minimal ViewerModel for subviewers."""

    def __init__(self, layers=None, dims=None):
        self.layers = _FakeLayerList(layers)
        if dims is not None:
            self.dims = dims


class _FakeSyncManager:
    """Minimal sync manager for view sync tests."""

    def __init__(self):
        self.shared_labels: dict[str, np.ndarray] = {}
        self._edited: dict[str, dict] = {}
        self._suppress_show_selected_sync = False

    def get_edited_slices(self, layer_name: str) -> dict:
        return self._edited.setdefault(layer_name, {"XY": {}, "XZ": {}, "YZ": {}})

    def clear_edited_slices_for_label(self, layer_name: str, label_id: int) -> None:
        edited = self._edited.get(layer_name)
        if edited is None:
            return
        for orient_dict in edited.values():
            orient_dict.pop(label_id, None)
            for key in list(orient_dict.keys()):
                if int(key) == label_id:
                    orient_dict.pop(key, None)


def _make_sync_test_annotator(**overrides):
    """Build a fake annotator with _sync_all_views_after_operation bound.

    Creates a (10,10,10) shared volume with labels, a main layer backed
    by the same array, and 3 subviewer label layers.

    Uses duck-typing patched version of the method since our _FakeLabelsLayer
    doesn't pass isinstance(Labels) checks.
    """
    vol = np.zeros((10, 10, 10), dtype=np.int32)
    vol[2:8, 2:8, 2:8] = 5  # label 5 fills center

    sync = _FakeSyncManager()
    sync.shared_labels["labels"] = vol

    # Main layer — shares same array by default
    main_layer = _FakeLabelsLayer(vol, name="labels")

    # Subviewer layers
    sub_xy = _FakeLabelsLayer(vol.copy(), name="labels_XY")
    sub_xz = _FakeLabelsLayer(vol.copy(), name="labels_XZ")
    sub_yz = _FakeLabelsLayer(vol.copy(), name="labels_YZ")

    viewer = _FakeViewerModel([main_layer], dims=_FakeDims(ndisplay=3))
    vm_xy = _FakeViewerModel([sub_xy])
    vm_xz = _FakeViewerModel([sub_xz])
    vm_yz = _FakeViewerModel([sub_yz])

    # Isolation proxy state
    _proxy_update_calls = []

    # Tracking lists for isolation propagation
    _propagate_show_selected_calls = []
    _propagate_colormap_calls = []

    ann = types.SimpleNamespace(
        viewer=viewer,
        viewer_model_xy=vm_xy,
        viewer_model_xz=vm_xz,
        viewer_model_yz=vm_yz,
        sync_manager=sync,

        _active_labels_name="labels",

        # Isolation proxy state
        _isolation_proxy_active=False,
        _isolation_proxy_label_id=None,
        _isolation_proxy_layer=None,

        # Isolation visual state (Fix 1)
        _isolation_active=False,
        _pre_isolation_colormap=None,
        _fragment_refine_roi=None,

        _proxy_update_calls=_proxy_update_calls,
        _propagate_show_selected_calls=_propagate_show_selected_calls,
        _propagate_colormap_calls=_propagate_colormap_calls,
    )

    # Stub _layer_by_name
    def _layer_by_name(layer_list, name):
        for layer in layer_list:
            if getattr(layer, "name", "") == name:
                return layer
        return None
    ann._layer_by_name = _layer_by_name

    # Stub _get_selected_label
    ann._get_selected_label = lambda: main_layer.selected_label

    # Stub _update_isolation_proxy to record calls
    def _update_isolation_proxy(label_id, main_layer, *, force=False, roi_hint=None):
        _proxy_update_calls.append({
            "label_id": label_id,
            "force": force,
            "roi_hint": roi_hint,
        })
    ann._update_isolation_proxy = _update_isolation_proxy

    # Stub _propagate_show_selected_label to record calls
    def _propagate_show_selected_label(show, *, label_id=None):
        _propagate_show_selected_calls.append({
            "show": show,
            "label_id": label_id,
        })
        return 3
    ann._propagate_show_selected_label = _propagate_show_selected_label

    # Stub _propagate_colormap_to_subviewers to record calls
    def _propagate_colormap_to_subviewers(colormap):
        _propagate_colormap_calls.append({"colormap": colormap})
        return 3
    ann._propagate_colormap_to_subviewers = _propagate_colormap_to_subviewers

    # Duck-typed _sync_all_views_after_operation (no isinstance check).
    # Mirrors the real method's logic including skip_3d_refresh (Fix 3)
    # and multi-label isolation re-propagation (Fix 1).
    def _sync_all_views_after_operation(
        self, roi=None, label_id=None, *, skip_3d_refresh=False
    ):
        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        if shared_data is None:
            return
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if main_layer is not None and hasattr(main_layer, "data"):
            if main_layer.data is not shared_data:
                if roi is not None:
                    main_layer.data[roi] = shared_data[roi]
                else:
                    main_layer.data[:] = shared_data
            if not skip_3d_refresh:
                if getattr(main_layer, "visible", True):
                    main_layer.refresh()
                if self._isolation_proxy_active and label_id is not None:
                    roi_hint = getattr(self, "_fragment_refine_roi", None)
                    self._update_isolation_proxy(
                        label_id, main_layer, force=True, roi_hint=roi_hint
                    )
        # Refresh subviewers
        self._refresh_subviewers()

        # Re-propagate isolation visual after data refresh (Fix 1).
        if getattr(self, "_isolation_active", False) or getattr(
            self, "_isolation_proxy_active", False
        ):
            sel_label = label_id or self._get_selected_label()
            if sel_label and sel_label > 0:
                if self._pre_isolation_colormap is not None:
                    self._propagate_colormap_to_subviewers(
                        main_layer.colormap if main_layer and hasattr(main_layer, "colormap") else None
                    )
                else:
                    self._propagate_show_selected_label(True, label_id=sel_label)

    def _refresh_subviewers(self):
        labels_name = self._active_labels_name
        for orient, vm in [
            ("XY", self.viewer_model_xy),
            ("XZ", self.viewer_model_xz),
            ("YZ", self.viewer_model_yz),
        ]:
            sub = self._layer_by_name(vm.layers, f"{labels_name}_{orient}")
            if sub is not None:
                sub.refresh()

    ann._sync_all_views_after_operation = types.MethodType(
        _sync_all_views_after_operation, ann
    )
    ann._refresh_subviewers = types.MethodType(_refresh_subviewers, ann)

    # Apply overrides
    for k, v in overrides.items():
        setattr(ann, k, v)

    return ann


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _get_main_layer(ann):
    return ann._layer_by_name(ann.viewer.layers, "labels")


def _get_sub_layer(ann, orient):
    vm = {
        "XY": ann.viewer_model_xy,
        "XZ": ann.viewer_model_xz,
        "YZ": ann.viewer_model_yz,
    }[orient]
    return ann._layer_by_name(vm.layers, f"labels_{orient}")


# ---------------------------------------------------------------------------
# WP-1: _sync_all_views_after_operation tests
# ---------------------------------------------------------------------------


class TestSyncAllViewsAfterOperation:
    """Verify the unified sync helper correctly propagates data changes."""

    def test_main_layer_refreshed_not_data_setter(self):
        """Main layer should receive refresh() — NEVER data=data setter.

        This is the core bug fix: data=data triggers expensive
        iso_categorical recalculation in 3D mode.
        """
        ann = _make_sync_test_annotator()
        main = _get_main_layer(ann)
        assert main._refresh_count == 0
        assert main._data_set_count == 0

        ann._sync_all_views_after_operation()

        # refresh() called — data setter NOT used
        assert main._refresh_count >= 1
        assert main._data_set_count == 0

    def test_subviewers_get_refreshed(self):
        """All 3 subviewers should get refresh() called."""
        ann = _make_sync_test_annotator()

        ann._sync_all_views_after_operation()

        for orient in ("XY", "XZ", "YZ"):
            sub = _get_sub_layer(ann, orient)
            assert sub._refresh_count >= 1, f"{orient} subviewer not refreshed"

    def test_roi_scoped_sync_copies_only_roi_when_arrays_differ(self):
        """When main_layer.data is a separate array, ROI-scoped sync copies only ROI."""
        ann = _make_sync_test_annotator()
        shared = ann.sync_manager.shared_labels["labels"]
        main = _get_main_layer(ann)

        # Create a separate array for main layer
        main._data = shared.copy()

        # Modify shared_data in ROI
        roi = (slice(2, 5), slice(2, 5), slice(2, 5))
        shared[roi] = 42

        ann._sync_all_views_after_operation(roi=roi)

        # Main layer ROI should be updated
        assert np.all(main.data[roi] == 42)
        # Outside ROI should still have old values
        assert main.data[0, 0, 0] == 0

    def test_full_sync_when_no_roi_and_arrays_differ(self):
        """When main_layer.data differs and no ROI, full volume copy."""
        ann = _make_sync_test_annotator()
        shared = ann.sync_manager.shared_labels["labels"]
        main = _get_main_layer(ann)

        # Separate arrays
        main._data = np.zeros_like(shared)

        ann._sync_all_views_after_operation()

        # Full volume should match
        np.testing.assert_array_equal(main.data, shared)

    def test_noop_when_no_shared_data(self):
        """Should return immediately if shared_data is None."""
        ann = _make_sync_test_annotator()
        ann.sync_manager.shared_labels.clear()

        # Should not raise
        ann._sync_all_views_after_operation()

    def test_isolation_proxy_updated_when_active(self):
        """When isolation proxy is active, update proxy with force=True."""
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
        )

        ann._sync_all_views_after_operation(label_id=5)

        # Proxy should be updated with force=True
        assert len(ann._proxy_update_calls) == 1
        assert ann._proxy_update_calls[0]["label_id"] == 5
        assert ann._proxy_update_calls[0]["force"] is True

    def test_refresh_when_proxy_active_but_no_label_id(self):
        """If proxy is active but no label_id provided, just refresh()."""
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
        )
        main = _get_main_layer(ann)

        ann._sync_all_views_after_operation()  # no label_id

        # Proxy NOT updated (no label_id to update for)
        assert len(ann._proxy_update_calls) == 0
        # refresh() called (not data setter)
        assert main._refresh_count >= 1
        assert main._data_set_count == 0


# ---------------------------------------------------------------------------
# WP-2: Interpolation view sync tests
# ---------------------------------------------------------------------------


class TestInterpolationViewSync:
    """Verify interpolation operations use refresh()-based sync."""

    def test_run_interpolation_syncs_views(self):
        """After interpolation, refresh() should be called on main layer."""
        ann = _make_sync_test_annotator()
        main = _get_main_layer(ann)
        shared = ann.sync_manager.shared_labels["labels"]

        shared[0, 0, 0] = 77
        ann._sync_all_views_after_operation(label_id=5)

        assert main._refresh_count >= 1
        assert main._data_set_count == 0  # No data setter!

    def test_reject_interpolation_restores_and_syncs(self):
        """After rejecting interpolation, restored snapshot synced via refresh()."""
        ann = _make_sync_test_annotator()
        shared = ann.sync_manager.shared_labels["labels"]
        main = _get_main_layer(ann)

        roi = (slice(2, 8), slice(2, 8), slice(2, 8))
        snapshot = shared[roi].copy()
        shared[roi] = 99  # "interpolated"
        shared[roi] = snapshot  # restore

        ann._sync_all_views_after_operation(roi=roi, label_id=5)

        assert main._refresh_count >= 1
        for orient in ("XY", "XZ", "YZ"):
            sub = _get_sub_layer(ann, orient)
            assert sub._refresh_count >= 1

    def test_interpolation_with_proxy_updates_proxy(self):
        """If isolation proxy active, interpolation should update it."""
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
        )

        ann._sync_all_views_after_operation(label_id=5)

        assert len(ann._proxy_update_calls) == 1
        assert ann._proxy_update_calls[0]["force"] is True


# ---------------------------------------------------------------------------
# WP-3: Smooth view sync tests
# ---------------------------------------------------------------------------


class TestSmoothViewSync:
    """Verify smooth operations use refresh()-based sync."""

    def test_preview_smooth_syncs_views(self):
        """After smooth preview, refresh() called — no data setter."""
        ann = _make_sync_test_annotator()
        shared = ann.sync_manager.shared_labels["labels"]
        main = _get_main_layer(ann)

        roi = (slice(3, 7), slice(3, 7), slice(3, 7))
        shared[roi] = 42
        ann._sync_all_views_after_operation(roi=roi, label_id=5)

        assert main._refresh_count >= 1
        assert main._data_set_count == 0

    def test_reject_smooth_restores_and_syncs(self):
        """After rejecting smooth, restored data synced via refresh()."""
        ann = _make_sync_test_annotator()
        shared = ann.sync_manager.shared_labels["labels"]

        roi = (slice(3, 7), slice(3, 7), slice(3, 7))
        snapshot = shared[roi].copy()
        shared[roi] = 42
        shared[roi] = snapshot

        ann._sync_all_views_after_operation(roi=roi, label_id=5)

        main = _get_main_layer(ann)
        assert main._refresh_count >= 1
        assert main._data_set_count == 0

    def test_smooth_with_proxy_active_updates_proxy(self):
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
        )

        ann._sync_all_views_after_operation(roi=None, label_id=5)

        assert len(ann._proxy_update_calls) == 1
        assert ann._proxy_update_calls[0]["force"] is True

    def test_reject_smooth_with_separate_array(self):
        """Reject smooth with separate main layer array — ROI copy + refresh()."""
        ann = _make_sync_test_annotator()
        shared = ann.sync_manager.shared_labels["labels"]
        main = _get_main_layer(ann)

        main._data = shared.copy()

        roi = (slice(3, 7), slice(3, 7), slice(3, 7))
        shared[roi] = 99
        shared[roi] = 5

        ann._sync_all_views_after_operation(roi=roi, label_id=5)

        np.testing.assert_array_equal(main.data[roi], shared[roi])
        assert main._data_set_count == 0  # No data setter


# ---------------------------------------------------------------------------
# WP-4: Split view sync tests
# ---------------------------------------------------------------------------


class TestSplitViewSync:
    """Verify split operations use refresh()-based sync."""

    def test_start_split_syncs_zeroed_hole(self):
        ann = _make_sync_test_annotator()
        shared = ann.sync_manager.shared_labels["labels"]

        shared[shared == 5] = 0

        ann._sync_all_views_after_operation()

        for orient in ("XY", "XZ", "YZ"):
            sub = _get_sub_layer(ann, orient)
            assert sub._refresh_count >= 1

    def test_accept_split_syncs_fragment_data(self):
        ann = _make_sync_test_annotator()
        shared = ann.sync_manager.shared_labels["labels"]

        roi = (slice(2, 8), slice(2, 8), slice(2, 8))
        shared[roi] = 0
        shared[3:5, 3:5, 3:5] = 10
        shared[5:7, 5:7, 5:7] = 11

        ann._sync_all_views_after_operation(roi=roi)

        main = _get_main_layer(ann)
        assert main._refresh_count >= 1
        assert main._data_set_count == 0

    def test_reject_split_restores_original(self):
        ann = _make_sync_test_annotator()
        shared = ann.sync_manager.shared_labels["labels"]

        roi = (slice(2, 8), slice(2, 8), slice(2, 8))
        snapshot = shared[roi].copy()
        shared[roi] = 0
        shared[roi] = snapshot

        ann._sync_all_views_after_operation(roi=roi)

        main = _get_main_layer(ann)
        assert main._refresh_count >= 1
        assert main._data_set_count == 0


# ---------------------------------------------------------------------------
# WP-5: Isolation state management tests
# ---------------------------------------------------------------------------


class TestIsolationDuringSyncOp:
    """Verify isolation proxy is correctly managed during operations."""

    def test_proxy_not_updated_without_label_id(self):
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
        )

        ann._sync_all_views_after_operation(roi=None, label_id=None)

        assert len(ann._proxy_update_calls) == 0

    def test_proxy_updated_with_label_id(self):
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
        )

        ann._sync_all_views_after_operation(label_id=5)

        assert len(ann._proxy_update_calls) == 1
        call = ann._proxy_update_calls[0]
        assert call["label_id"] == 5
        assert call["force"] is True

    def test_proxy_updated_with_different_label_id(self):
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
        )

        ann._sync_all_views_after_operation(label_id=10)

        assert len(ann._proxy_update_calls) == 1
        assert ann._proxy_update_calls[0]["label_id"] == 10

    def test_no_proxy_uses_refresh(self):
        """When proxy NOT active, refresh() called — no data setter."""
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=False,
        )
        main = _get_main_layer(ann)

        ann._sync_all_views_after_operation(label_id=5)

        assert main._refresh_count >= 1
        assert main._data_set_count == 0
        assert len(ann._proxy_update_calls) == 0

    def test_proxy_and_subviewers_both_updated(self):
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
        )

        ann._sync_all_views_after_operation(label_id=5)

        assert len(ann._proxy_update_calls) == 1
        for orient in ("XY", "XZ", "YZ"):
            sub = _get_sub_layer(ann, orient)
            assert sub._refresh_count >= 1

    def test_separate_array_roi_sync_with_proxy(self):
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
        )
        shared = ann.sync_manager.shared_labels["labels"]
        main = _get_main_layer(ann)
        main._data = shared.copy()

        roi = (slice(3, 7), slice(3, 7), slice(3, 7))
        shared[roi] = 42

        ann._sync_all_views_after_operation(roi=roi, label_id=5)

        np.testing.assert_array_equal(main.data[roi], shared[roi])
        assert len(ann._proxy_update_calls) == 1


# ---------------------------------------------------------------------------
# _update_isolation_proxy force parameter tests
# ---------------------------------------------------------------------------


class TestUpdateIsolationProxyForce:
    """Verify the force parameter on _update_isolation_proxy."""

    def test_force_parameter_in_signature(self):
        """Verify _update_isolation_proxy accepts force keyword."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        import inspect
        sig = inspect.signature(Curation3DAnnotator._update_isolation_proxy)
        assert "force" in sig.parameters
        param = sig.parameters["force"]
        assert param.default is False
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# Fix 1: Multi-label isolation colormap re-propagation tests
# ---------------------------------------------------------------------------


class TestMultiLabelIsolationSync:
    """Verify sync correctly dispatches isolation re-propagation.

    Fix 1: When multi-label isolation (DirectLabelColormap) is active,
    _sync_all_views_after_operation should re-propagate the colormap to
    subviewers — not show_selected_label which would hide all but one label.
    """

    def test_multi_label_propagates_colormap_not_show_selected(self):
        """When _pre_isolation_colormap is set, colormap re-propagated."""
        ann = _make_sync_test_annotator(
            _isolation_active=True,
            _pre_isolation_colormap="saved_colormap",
        )
        main = _get_main_layer(ann)
        main.colormap = "multi_label_direct_colormap"

        ann._sync_all_views_after_operation(label_id=5)

        # Colormap propagated — NOT show_selected_label
        assert len(ann._propagate_colormap_calls) == 1
        assert ann._propagate_colormap_calls[0]["colormap"] == "multi_label_direct_colormap"
        assert len(ann._propagate_show_selected_calls) == 0

    def test_single_label_propagates_show_selected_label(self):
        """When _pre_isolation_colormap is None, show_selected_label re-propagated."""
        ann = _make_sync_test_annotator(
            _isolation_active=True,
            _pre_isolation_colormap=None,
        )

        ann._sync_all_views_after_operation(label_id=5)

        # show_selected_label propagated — NOT colormap
        assert len(ann._propagate_show_selected_calls) == 1
        assert ann._propagate_show_selected_calls[0]["show"] is True
        assert ann._propagate_show_selected_calls[0]["label_id"] == 5
        assert len(ann._propagate_colormap_calls) == 0

    def test_no_isolation_no_propagation(self):
        """When isolation is inactive, nothing re-propagated."""
        ann = _make_sync_test_annotator(
            _isolation_active=False,
            _isolation_proxy_active=False,
        )

        ann._sync_all_views_after_operation(label_id=5)

        assert len(ann._propagate_show_selected_calls) == 0
        assert len(ann._propagate_colormap_calls) == 0

    def test_proxy_active_triggers_propagation(self):
        """Isolation proxy active (without _isolation_active) still propagates."""
        ann = _make_sync_test_annotator(
            _isolation_active=False,
            _isolation_proxy_active=True,
            _pre_isolation_colormap=None,
        )

        ann._sync_all_views_after_operation(label_id=5)

        assert len(ann._propagate_show_selected_calls) == 1

    def test_label_id_zero_skips_propagation(self):
        """When selected label is 0 (background), no propagation."""
        ann = _make_sync_test_annotator(
            _isolation_active=True,
        )
        _get_main_layer(ann).selected_label = 0

        ann._sync_all_views_after_operation()

        assert len(ann._propagate_show_selected_calls) == 0


# ---------------------------------------------------------------------------
# Fix 3: skip_3d_refresh tests
# ---------------------------------------------------------------------------


class TestSkip3DRefresh:
    """Verify skip_3d_refresh skips expensive operations but still syncs data."""

    def test_skip_3d_refresh_skips_main_refresh(self):
        """Main layer refresh() skipped when skip_3d_refresh=True."""
        ann = _make_sync_test_annotator()
        main = _get_main_layer(ann)
        assert main._refresh_count == 0

        ann._sync_all_views_after_operation(skip_3d_refresh=True)

        assert main._refresh_count == 0  # NOT called

    def test_skip_3d_refresh_still_refreshes_subviewers(self):
        """Subviewers still get refreshed even with skip_3d_refresh."""
        ann = _make_sync_test_annotator()

        ann._sync_all_views_after_operation(skip_3d_refresh=True)

        for orient in ("XY", "XZ", "YZ"):
            sub = _get_sub_layer(ann, orient)
            assert sub._refresh_count >= 1, f"{orient} not refreshed"

    def test_skip_3d_refresh_still_copies_data(self):
        """Data copy from shared to main still happens."""
        ann = _make_sync_test_annotator()
        shared = ann.sync_manager.shared_labels["labels"]
        main = _get_main_layer(ann)
        main._data = shared.copy()

        roi = (slice(2, 5), slice(2, 5), slice(2, 5))
        shared[roi] = 42

        ann._sync_all_views_after_operation(roi=roi, skip_3d_refresh=True)

        assert np.all(main.data[roi] == 42)

    def test_skip_3d_refresh_skips_proxy_rebuild(self):
        """Proxy not rebuilt when skip_3d_refresh=True."""
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
        )

        ann._sync_all_views_after_operation(label_id=5, skip_3d_refresh=True)

        assert len(ann._proxy_update_calls) == 0

    def test_normal_path_still_refreshes(self):
        """Without skip_3d_refresh, refresh() still fires (regression guard)."""
        ann = _make_sync_test_annotator()
        main = _get_main_layer(ann)

        ann._sync_all_views_after_operation(skip_3d_refresh=False)

        assert main._refresh_count >= 1


# ---------------------------------------------------------------------------
# Fix 5: roi_hint parameter tests
# ---------------------------------------------------------------------------


class TestRoiHintInProxy:
    """Verify roi_hint parameter wiring for proxy bounding-box optimization."""

    def test_roi_hint_parameter_in_signature(self):
        """Verify _update_isolation_proxy accepts roi_hint keyword."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        import inspect
        sig = inspect.signature(Curation3DAnnotator._update_isolation_proxy)
        assert "roi_hint" in sig.parameters
        param = sig.parameters["roi_hint"]
        assert param.default is None
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_sync_passes_fragment_refine_roi_as_hint(self):
        """When _fragment_refine_roi is set, proxy gets it as roi_hint."""
        roi = (slice(2, 8), slice(2, 8), slice(2, 8))
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
            _fragment_refine_roi=roi,
        )

        ann._sync_all_views_after_operation(label_id=5)

        assert len(ann._proxy_update_calls) == 1
        assert ann._proxy_update_calls[0]["roi_hint"] is roi

    def test_sync_passes_none_roi_hint_when_no_cache(self):
        """When _fragment_refine_roi is None, proxy gets roi_hint=None."""
        ann = _make_sync_test_annotator(
            _isolation_proxy_active=True,
            _isolation_proxy_label_id=5,
            _fragment_refine_roi=None,
        )

        ann._sync_all_views_after_operation(label_id=5)

        assert len(ann._proxy_update_calls) == 1
        assert ann._proxy_update_calls[0]["roi_hint"] is None

    def test_fragment_refine_roi_initialized(self):
        """Verify _fragment_refine_roi exists in __init__."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        import inspect
        src = inspect.getsource(Curation3DAnnotator.__init__)
        assert "_fragment_refine_roi" in src

    def test_reset_split_state_clears_fragment_refine_roi(self):
        """_reset_split_state clears _fragment_refine_roi."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        import inspect
        src = inspect.getsource(Curation3DAnnotator._reset_split_state)
        assert "_fragment_refine_roi" in src


# ---------------------------------------------------------------------------
# Fix 2: Fragment refine 3D proxy tests (source-level)
# ---------------------------------------------------------------------------


class TestFragmentRefineIsolation:
    """Verify fragment refine uses single-label isolation consistently."""

    def test_enter_fragment_refine_uses_single_label_isolation(self):
        """_enter_fragment_refine_mode uses _on_isolation_changed for single label."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        import inspect
        src = inspect.getsource(Curation3DAnnotator._enter_fragment_refine_mode)
        assert "_on_isolation_changed" in src

    def test_on_fragment_selected_updates_proxy(self):
        """_on_fragment_selected rebuilds proxy when isolation is active."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        import inspect
        src = inspect.getsource(Curation3DAnnotator._on_fragment_selected)
        assert "_update_isolation_proxy" in src
        assert "_fragment_refine_roi" in src

    def test_accept_smooth_fragment_cycle_updates_proxy(self):
        """_accept_smooth fragment refine auto-advance updates proxy."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        import inspect
        src = inspect.getsource(Curation3DAnnotator._accept_smooth)
        assert "_update_isolation_proxy" in src
        assert "_fragment_refine_roi" in src

    def test_enter_fragment_refine_caches_roi(self):
        """_enter_fragment_refine_mode caches ROI for fast proxy scans."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        import inspect
        src = inspect.getsource(Curation3DAnnotator._enter_fragment_refine_mode)
        assert "_fragment_refine_roi" in src

    def test_enter_fragment_refine_skips_3d_refresh(self):
        """_enter_fragment_refine_mode uses skip_3d_refresh when isolation active."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        import inspect
        src = inspect.getsource(Curation3DAnnotator._enter_fragment_refine_mode)
        assert "skip_3d_refresh" in src

    def test_accept_split_skips_3d_refresh_conditionally(self):
        """_accept_split uses skip_3d_refresh when auto-advance + proxy active."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        import inspect
        src = inspect.getsource(Curation3DAnnotator._accept_split)
        assert "skip_3d_refresh" in src

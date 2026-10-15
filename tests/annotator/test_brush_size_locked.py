"""Tests for the brush_size=1 lock (2026-05-17).

Locked design decision: brush_size is fixed at 1 across
all annotator Labels layers.  These tests verify:

  1. ``lock_brush_size_to_1`` immediately sets brush=1 on a fresh layer.
  2. The snap-back handler reverts user/code changes back to 1.
  3. The lock is idempotent — calling twice doesn't double-attach.
  4. ``sync_label_properties`` always propagates brush=1 regardless of
     ``active_layer.brush_size``.
  5. ``_apply_draw_settings`` always writes brush=1 regardless of
     ``_default_brush_size``.

Tests use plain Python fakes with a minimal napari-event-like API —
no Qt / no GUI required, runs in any pytest invocation.
"""
from __future__ import annotations

import types

import pytest


# ---------------------------------------------------------------------------
# Fake napari Labels layer for unit testing the lock
# ---------------------------------------------------------------------------


class _FakeEmitter:
    """Mimics napari's event emitter — connect handlers + emit."""

    def __init__(self):
        self._handlers = []

    def connect(self, handler):
        self._handlers.append(handler)

    def emit(self, event):
        for h in self._handlers:
            h(event)


class _FakeEvent:
    def __init__(self, source):
        self.source = source


class _FakeEvents:
    def __init__(self):
        self.brush_size = _FakeEmitter()


class _FakeLabels:
    """Minimal stand-in for napari.layers.Labels.

    Setting ``brush_size`` triggers ``events.brush_size.emit`` only when
    the value actually changes — same contract as napari.
    """

    def __init__(self, name="fake", brush_size=10):
        self.name = name
        self._brush_size = brush_size
        self.events = _FakeEvents()

    @property
    def brush_size(self):
        return self._brush_size

    @brush_size.setter
    def brush_size(self, value):
        if value == self._brush_size:
            return  # napari same-value no-op
        self._brush_size = value
        self.events.brush_size.emit(_FakeEvent(self))


# ---------------------------------------------------------------------------
# lock_brush_size_to_1
# ---------------------------------------------------------------------------


class TestLockBrushSizeToOne:

    def test_initial_brush_set_to_1(self):
        from packages.epicycle_annotator.sync import lock_brush_size_to_1
        layer = _FakeLabels(brush_size=10)
        lock_brush_size_to_1(layer)
        assert layer.brush_size == 1

    def test_already_1_no_op(self):
        from packages.epicycle_annotator.sync import lock_brush_size_to_1
        layer = _FakeLabels(brush_size=1)
        # Should not error and should remain 1.
        lock_brush_size_to_1(layer)
        assert layer.brush_size == 1

    def test_snap_back_on_user_change(self):
        from packages.epicycle_annotator.sync import lock_brush_size_to_1
        layer = _FakeLabels(brush_size=10)
        lock_brush_size_to_1(layer)
        assert layer.brush_size == 1
        # Simulate user dragging the slider to 7.
        layer.brush_size = 7
        # Snap-back handler should have reverted it.
        assert layer.brush_size == 1

    def test_idempotent_attach(self):
        """Calling lock twice attaches the handler only once."""
        from packages.epicycle_annotator.sync import lock_brush_size_to_1
        layer = _FakeLabels(brush_size=5)
        lock_brush_size_to_1(layer)
        lock_brush_size_to_1(layer)
        assert len(layer.events.brush_size._handlers) == 1

    def test_flag_set_after_lock(self):
        from packages.epicycle_annotator.sync import lock_brush_size_to_1
        layer = _FakeLabels()
        assert getattr(layer, "_epicycle_brush_locked", False) is False
        lock_brush_size_to_1(layer)
        assert layer._epicycle_brush_locked is True

    def test_repeated_user_drags_all_snap(self):
        """Several user changes all snap back; no perf regression possible
        because the snap is a single int write per actual change."""
        from packages.epicycle_annotator.sync import lock_brush_size_to_1
        layer = _FakeLabels(brush_size=10)
        lock_brush_size_to_1(layer)
        for target in (3, 8, 15, 2, 20):
            layer.brush_size = target
            assert layer.brush_size == 1, (
                f"snap-back failed for value {target}"
            )

    def test_lock_tolerates_missing_events(self):
        """If a fake/test layer has no events.brush_size, lock is a no-op
        rather than crashing.  Defensive — supports MagicMock fakes in
        other test files."""
        from packages.epicycle_annotator.sync import lock_brush_size_to_1

        class _NoEvents:
            brush_size = 10
            name = "noevents"
        layer = _NoEvents()
        # Should not raise — and brush_size is still adjusted to 1.
        lock_brush_size_to_1(layer)
        assert layer.brush_size == 1


# ---------------------------------------------------------------------------
# sync_label_properties hardcoded value
# ---------------------------------------------------------------------------


class _FakeLayerList(list):
    def __contains__(self, name):
        return any(getattr(l, "name", None) == name for l in self)

    def __getitem__(self, key):
        if isinstance(key, str):
            for layer in self:
                if getattr(layer, "name", None) == key:
                    return layer
            raise KeyError(key)
        return list.__getitem__(self, key)


class _FakeViewerModel:
    def __init__(self, layers):
        self.layers = layers


class _FakeMainViewer:
    def __init__(self, layers):
        self.layers = layers


class _ExtFakeLabels(_FakeLabels):
    """Adds opacity / colormap / blending / n_edit_dimensions so the full
    sync_label_properties path runs without AttributeError."""

    def __init__(self, name="fake", brush_size=10):
        super().__init__(name=name, brush_size=brush_size)
        self.opacity = 1.0
        self.colormap = None
        self.blending = "translucent"
        self.n_edit_dimensions = 2


def _build_sync_manager_with_layers():
    from packages.epicycle_annotator.sync import LabelSyncManager
    main_lyr = _ExtFakeLabels(name="curation_3d_X", brush_size=10)
    sub_xy = _ExtFakeLabels(name="curation_3d_X_XY", brush_size=10)
    sub_xz = _ExtFakeLabels(name="curation_3d_X_XZ", brush_size=10)
    sub_yz = _ExtFakeLabels(name="curation_3d_X_YZ", brush_size=10)
    main_viewer = _FakeMainViewer(_FakeLayerList([main_lyr]))
    vm_xy = _FakeViewerModel(_FakeLayerList([sub_xy]))
    vm_xz = _FakeViewerModel(_FakeLayerList([sub_xz]))
    vm_yz = _FakeViewerModel(_FakeLayerList([sub_yz]))
    mgr = LabelSyncManager(
        main_viewer=main_viewer,
        viewer_model_xy=vm_xy, viewer_model_xz=vm_xz, viewer_model_yz=vm_yz,
        qt_viewer_xy=None, qt_viewer_xz=None, qt_viewer_yz=None,
    )
    return mgr, main_lyr, [sub_xy, sub_xz, sub_yz]


def _patch_isinstance_for_labels(monkeypatch):
    """sync_label_properties uses ``isinstance(layer, Labels)`` — patch it
    so our fakes pass."""
    import packages.epicycle_annotator.sync as sync_mod
    monkeypatch.setattr(sync_mod, "Labels", _ExtFakeLabels)


class TestSyncLabelPropertiesHardcodesOne:

    def test_main_gets_brush_1_from_active_sub(self, monkeypatch):
        _patch_isinstance_for_labels(monkeypatch)
        mgr, main_lyr, subs = _build_sync_manager_with_layers()
        # Active = subviewer with brush=10.  Hardcode should propagate 1.
        subs[0]._brush_size = 10
        mgr.sync_label_properties(subs[0])
        assert main_lyr.brush_size == 1
        for sub in subs:
            assert sub.brush_size == 1

    def test_subs_get_brush_1_from_active_main(self, monkeypatch):
        _patch_isinstance_for_labels(monkeypatch)
        mgr, main_lyr, subs = _build_sync_manager_with_layers()
        main_lyr._brush_size = 10  # bypass setter to skip event emit
        mgr.sync_label_properties(main_lyr)
        for sub in subs:
            assert sub.brush_size == 1

    def test_active_layer_value_ignored(self, monkeypatch):
        """Even if active_layer.brush_size is 42, propagated value is 1."""
        _patch_isinstance_for_labels(monkeypatch)
        mgr, main_lyr, subs = _build_sync_manager_with_layers()
        subs[0]._brush_size = 42
        mgr.sync_label_properties(subs[0])
        assert main_lyr.brush_size == 1
        for sub in subs:
            assert sub.brush_size == 1

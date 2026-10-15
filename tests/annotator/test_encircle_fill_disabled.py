"""napari-0.8 right-click encircle-and-fill must not reach subviewer labels.

napari 0.8 gives every Labels layer a brush-stroke overlay whose right-button
press starts an encircle-and-fill stroke.  EpiCycle's subviewer sync is
drag-generator-only, so the fill lands visually and is then silently discarded
on the next reslice -- the user's work is lost with no warning.

The block is applied through the overlay MODEL; napari's own press callback
stays in ``mouse_drag_callbacks`` because the overlay's ``close()`` removes it
unguarded.  That crash path needs a real viewer -- see
``test_encircle_fill_real_viewer.py``.

Resolves a ranked Step 2 defect (Brick A).
The module is shared with Step 3, whose subviewers have the same one-way
sync, so the containment applies there too.
"""

from __future__ import annotations

import types

import numpy as np

from packages.epicycle_annotator.subviewers import (
    SubviewerManager,
    disable_encircle_fill,
)


# ---------------------------------------------------------------------------
# Stand-in for the napari overlay callback (same module + qualname as the real
# one, which is what the identifier matches on).
# ---------------------------------------------------------------------------

def _napari_brush_stroke_press(layer, event):  # pragma: no cover - never run
    raise AssertionError("the napari brush-stroke handler must not survive")


_napari_brush_stroke_press.__module__ = "napari._vispy.overlays.labels_brush_stroke"
_napari_brush_stroke_press.__qualname__ = (
    "_only_when_overlay_enabled.<locals>.decorated_callback"
)


def _epicycle_cb(layer, event):  # pragma: no cover - never run
    pass


class _FakeEvent:
    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def disconnect(self, cb):
        self._callbacks.remove(cb)

    def emit(self):
        for cb in list(self._callbacks):
            cb(None)


class _FakeLabelsLayer:
    """Labels layer with napari-0.8's overlay handler already installed."""

    _type = "labels"

    def __init__(self, data, name=""):
        self.data = data
        self.name = name
        self.visible = True
        self.editable = True
        self.brush_size = 1
        self.selected_label = 1
        self.mode = "pan_zoom"
        self.colormap = "label"
        self.preserve_labels = True
        self.mouse_drag_callbacks = [_napari_brush_stroke_press]
        self.mouse_move_callbacks = []
        self._overlays = {"brush_stroke": types.SimpleNamespace(enabled=True)}
        self.events = types.SimpleNamespace(
            visible=_FakeEvent(),
            colormap=_FakeEvent(),
            mode=_FakeEvent(),
            selected_label=_FakeEvent(),
            brush_size=_FakeEvent(),
        )

    @property
    def ndim(self):
        return self.data.ndim


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------

def test_disables_the_overlay_and_touches_no_callback():
    layer = _FakeLabelsLayer(np.zeros((2, 2, 2), np.int32), "lbl_XY")
    layer.mouse_drag_callbacks.append(_epicycle_cb)

    disable_encircle_fill(layer)

    assert layer._overlays["brush_stroke"].enabled is False
    # napari's own callback MUST survive: its overlay's close() removes it
    # unguarded, so stripping it turns every layer removal into a ValueError.
    assert layer.mouse_drag_callbacks == [_napari_brush_stroke_press, _epicycle_cb]


def test_re_asserts_when_napari_re_enables_on_a_mode_change():
    layer = _FakeLabelsLayer(np.zeros((2, 2, 2), np.int32), "lbl_XY")
    disable_encircle_fill(layer)

    # napari re-enables the overlay when the layer goes back into PAINT
    # (`_mode_setter_helper`, BEFORE events.mode is emitted).
    layer._overlays["brush_stroke"].enabled = True
    layer.events.mode.emit()

    assert layer._overlays["brush_stroke"].enabled is False


def test_is_idempotent():
    layer = _FakeLabelsLayer(np.zeros((2, 2, 2), np.int32), "lbl_XY")
    disable_encircle_fill(layer)
    disable_encircle_fill(layer)
    assert len(layer.events.mode._callbacks) == 1


def test_survives_a_layer_without_events_or_overlays():
    class _Bare:
        pass

    disable_encircle_fill(_Bare())  # must not raise


# ---------------------------------------------------------------------------
# Wiring -- both subviewer Labels-creation sites
# ---------------------------------------------------------------------------

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


class _FakeViewerModel:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.dims = types.SimpleNamespace(current_step=(0,), range=((0, 10, 1),))
        self.cursor = types.SimpleNamespace(position=(0, 0, 0))

    def add_labels(self, data, name=""):
        # napari 0.8 installs the brush-stroke handler at add time.
        layer = _FakeLabelsLayer(data, name=name)
        self.layers._layers.append(layer)
        return layer


class _FakeSync:
    def __init__(self):
        self.shared_labels = {}

    def register_shared_data(self, name, data):
        self.shared_labels[name] = data

    def connect_label_sync(self, layer):
        pass


class _FakeDrawing:
    def create_mouse_drag_callback(self, orientation, viewer_model):
        return _epicycle_cb


def _make_manager():
    vms = [_FakeViewerModel() for _ in range(3)]
    mgr = SubviewerManager(
        *vms, qt_viewer_xy=None, qt_viewer_xz=None, qt_viewer_yz=None
    )
    return mgr, vms


def test_add_labels_to_subviewers_disables_encircle_fill():
    mgr, vms = _make_manager()
    src = _FakeLabelsLayer(np.zeros((4, 8, 8), np.int32), "lbl")

    mgr.add_labels_to_subviewers(src, _FakeSync(), _FakeDrawing())

    for vm in vms:
        lbl = list(vm.layers)[0]
        assert lbl._overlays["brush_stroke"].enabled is False
        # Both napari's and EpiCycle's callbacks are untouched.
        assert _napari_brush_stroke_press in lbl.mouse_drag_callbacks
        assert _epicycle_cb in lbl.mouse_drag_callbacks


def test_fast_reload_disables_encircle_fill_again():
    """FAST reload reuses the existing layers -- re-strip on that path too.

    Uses REAL napari Labels layers (no viewer needed) because
    `try_fast_reload` gates on `isinstance(l, Labels)`.
    """
    from unittest.mock import MagicMock

    from napari.layers import Labels

    from packages.epicycle_annotator.orientation import transpose_data

    data = np.zeros((4, 8, 8), np.int32)
    vms, lbls = [], []
    for orient in ("XY", "XZ", "YZ"):
        lbl = Labels(transpose_data(data, orient), name=f"old_labels_{orient}")
        vm = MagicMock()
        vm.layers = [lbl]
        vms.append(vm)
        lbls.append(lbl)

    mgr = SubviewerManager(*vms, MagicMock(), MagicMock(), MagicMock())

    # Simulate napari re-enabling the overlay between patches.
    for lbl in lbls:
        lbl._overlays["brush_stroke"].enabled = True

    proxy = Labels(data.copy(), name="new_labels")
    ok, _ = mgr.try_fast_reload([], [proxy])
    assert ok is True

    for lbl in lbls:
        assert lbl._overlays["brush_stroke"].enabled is False

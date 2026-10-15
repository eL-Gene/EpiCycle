"""`_apply_edit_constraints` must not un-hide a layer the user hid with `d`.

Journey item #14: the method forced ``layer.visible = allow`` on BOTH
branches, so any post-prediction refresh brought back a labels layer the
user had deliberately hidden with the channel keybinding.

Only the disallowed branch may touch visibility (that hide is the method's
own doing).  Shared module -- Step 3's 3d_curation target mode is covered
below too.
"""

from __future__ import annotations

import types

from napari_epicycle.annotator.native_multiview_widget import (
    NativeMultiviewAnnotator,
)


class _FakeLabelsLayer:
    def __init__(self, name, visible=True):
        self.name = name
        self.visible = visible
        self.editable = True
        self.mode = "paint"


def _make_widget(layers, target_mode):
    w = NativeMultiviewAnnotator.__new__(NativeMultiviewAnnotator)
    w._active_target_mode = target_mode
    w._all_subviewer_label_layers = lambda: layers
    w._apply_edit_constraints = (
        NativeMultiviewAnnotator._apply_edit_constraints.__get__(w)
    )
    return w


def _lateral_layers():
    return [
        _FakeLabelsLayer("p_work_labels_lateral_3d_XY"),
        _FakeLabelsLayer("p_work_labels_axial_xz_3d_XZ"),
        _FakeLabelsLayer("p_work_labels_axial_yz_3d_YZ"),
    ]


def test_hidden_allowed_layer_stays_hidden():
    layers = _lateral_layers()
    layers[0].visible = False  # the user pressed `d`

    _make_widget(layers, "lateral")._apply_edit_constraints()

    assert layers[0].visible is False, (
        "a layer hidden with `d` must not be un-hidden by a refresh"
    )
    # ...and it must stay EDITABLE: nothing re-arms `editable` when the layer
    # comes back, so latching it off here makes paint permanently dead on it
    # (napari's mode setter needs editable).  napari forces PAN_ZOOM while a
    # layer is invisible on its own.
    assert layers[0].editable is True
    assert layers[0].mode == "paint"


def test_visible_allowed_layer_stays_visible_and_editable():
    layers = _lateral_layers()

    _make_widget(layers, "lateral")._apply_edit_constraints()

    assert layers[0].visible is True
    assert layers[0].editable is True
    assert layers[0].mode == "paint"


def test_disallowed_layers_are_still_hidden_and_locked():
    layers = _lateral_layers()

    _make_widget(layers, "lateral")._apply_edit_constraints()

    for layer in layers[1:]:
        assert layer.visible is False
        assert layer.editable is False
        assert layer.mode == "pan_zoom"


def test_axial_target_hides_the_lateral_layer_only():
    layers = _lateral_layers()

    _make_widget(layers, "axial")._apply_edit_constraints()

    assert layers[0].visible is False
    assert layers[1].visible is True and layers[1].editable is True
    assert layers[2].visible is True and layers[2].editable is True


def test_step3_curation_mode_leaves_every_visibility_alone():
    """3D curation allows all views -- and must not resurrect a hidden one."""
    layers = _lateral_layers()
    layers[1].visible = False

    _make_widget(layers, "3d_curation")._apply_edit_constraints()

    assert [l.visible for l in layers] == [True, False, True]
    # Every view is allowed in 3D curation, so every layer stays editable —
    # including one that is transiently invisible under the isolation cascade.
    assert all(l.editable is True for l in layers)
    assert all(l.mode == "paint" for l in layers)

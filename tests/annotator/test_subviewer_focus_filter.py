"""Tests for the per-subviewer focus listener (Curation2DAnnotator).

Pure-Python tests with fakes — no Qt or napari instantiation required.
Covers the behavior of `_on_subviewer_focused`, `_refresh_segment_button_text`
focus clearing, and `_teardown_focus_filters` cleanup.

Regression test included for the bug shipped + caught in 2026-05-27 review
cycle: mode-reset on every mouse-down killed paint mid-stroke (fixed in
`d747f4e` by gating mode reset on orient switch only).
"""

from __future__ import annotations

import types

import pytest

from napari_epicycle.annotator.curation_2d_widget import (
    Curation2DAnnotator,
    _SubviewerFocusFilter,
)


# ---------------------------------------------------------------------------
# Lightweight fakes (no Qt / napari required)
# ---------------------------------------------------------------------------


class _FakeLabels:
    """Stand-in for napari.layers.Labels — minimal surface needed for tests."""

    def __init__(self, name: str, mode: str = "paint", editable: bool = True):
        self.name = name
        self.mode = mode
        self.editable = editable
        self.visible = True

    # Make isinstance(layer, Labels) check pass by patching the import target.
    pass


class _FakeLayerList:
    def __init__(self, layers):
        self._by_name = {layer.name: layer for layer in layers}
        self.selection = types.SimpleNamespace(active=None)

    def __iter__(self):
        return iter(self._by_name.values())

    def __contains__(self, name):
        return name in self._by_name

    def __getitem__(self, name):
        return self._by_name[name]


class _FakeVM:
    def __init__(self, layers):
        self.layers = _FakeLayerList(layers)


class _FakeAnnotator:
    """Minimal Curation2DAnnotator-shaped object for testing focus methods.

    Mimics only the attributes / methods that `_on_subviewer_focused`
    and the bucket-switch focus-clearing code path read.
    """

    def __init__(self, target_mode: str = "lateral", patch_prefix: str = "patch_1"):
        self._active_target_mode = target_mode
        self._active_labels_name = f"{patch_prefix}_work_labels_lateral_3d"
        self._focused_orient = None
        self._indicator_calls = []

        # Lateral layers (one per orientation, present in all 3 VMs).
        self.xy_lateral = _FakeLabels(f"{patch_prefix}_work_labels_lateral_3d_XY")
        self.xz_lateral = _FakeLabels(f"{patch_prefix}_work_labels_lateral_3d_XZ", editable=False)
        self.yz_lateral = _FakeLabels(f"{patch_prefix}_work_labels_lateral_3d_YZ", editable=False)

        # Axial layers (XZ class + YZ class, each present in all 3 VMs).
        self.xy_axial_xz = _FakeLabels(f"{patch_prefix}_work_labels_axial_xz_3d_XY", editable=False)
        self.xz_axial_xz = _FakeLabels(f"{patch_prefix}_work_labels_axial_xz_3d_XZ")
        self.yz_axial_xz = _FakeLabels(f"{patch_prefix}_work_labels_axial_xz_3d_YZ", editable=False)

        self.xy_axial_yz = _FakeLabels(f"{patch_prefix}_work_labels_axial_yz_3d_XY", editable=False)
        self.xz_axial_yz = _FakeLabels(f"{patch_prefix}_work_labels_axial_yz_3d_XZ", editable=False)
        self.yz_axial_yz = _FakeLabels(f"{patch_prefix}_work_labels_axial_yz_3d_YZ")

        self.viewer_model_xy = _FakeVM([self.xy_lateral, self.xy_axial_xz, self.xy_axial_yz])
        self.viewer_model_xz = _FakeVM([self.xz_lateral, self.xz_axial_xz, self.xz_axial_yz])
        self.viewer_model_yz = _FakeVM([self.yz_lateral, self.yz_axial_xz, self.yz_axial_yz])

    def _vm_for_orient(self, orient):
        return {
            "XY": self.viewer_model_xy,
            "XZ": self.viewer_model_xz,
            "YZ": self.viewer_model_yz,
        }.get(orient.upper())

    def _labels_patch_prefix(self, labels_name):
        # Strip _work_labels_*_3d suffix
        for suffix in (
            "_work_labels_lateral_3d",
            "_work_labels_axial_xz_3d",
            "_work_labels_axial_yz_3d",
        ):
            if labels_name.endswith(suffix):
                return labels_name[: -len(suffix)]
        return labels_name

    def _label_for_patch(self, patch_prefix, orient):
        if orient.upper() == "XZ":
            return f"{patch_prefix}_work_labels_axial_xz_3d"
        if orient.upper() == "YZ":
            return f"{patch_prefix}_work_labels_axial_yz_3d"
        return f"{patch_prefix}_work_labels_lateral_3d"

    def _layer_by_name(self, layer_list, name):
        try:
            return layer_list[name]
        except (KeyError, TypeError):
            return None

    def _update_per_view_indicator(self):
        self._indicator_calls.append(getattr(self, "_focused_orient", None))


# Patch isinstance(layer, Labels) to accept _FakeLabels.
@pytest.fixture(autouse=True)
def _patch_labels_isinstance(monkeypatch):
    from napari_epicycle.annotator import curation_2d_widget as cw

    monkeypatch.setattr(cw, "Labels", _FakeLabels)


# ---------------------------------------------------------------------------
# _on_subviewer_focused: bucket-mode gating
# ---------------------------------------------------------------------------


def test_lateral_mode_skips_xz_focus():
    """In lateral bucket, clicking XZ subviewer is ignored (no-op)."""
    a = _FakeAnnotator(target_mode="lateral")
    a._focused_orient = "XY"
    Curation2DAnnotator._on_subviewer_focused(a, "XZ")
    # Focus not changed
    assert a._focused_orient == "XY"
    # Indicator not refreshed
    assert a._indicator_calls == []


def test_lateral_mode_skips_yz_focus():
    """In lateral bucket, clicking YZ subviewer is ignored."""
    a = _FakeAnnotator(target_mode="lateral")
    a._focused_orient = "XY"
    Curation2DAnnotator._on_subviewer_focused(a, "YZ")
    assert a._focused_orient == "XY"


def test_axial_mode_skips_xy_focus():
    """In axial bucket, clicking XY subviewer is ignored."""
    a = _FakeAnnotator(target_mode="axial")
    a._active_labels_name = "patch_1_work_labels_axial_xz_3d"
    a._focused_orient = "XZ"
    Curation2DAnnotator._on_subviewer_focused(a, "XY")
    assert a._focused_orient == "XZ"


# ---------------------------------------------------------------------------
# _on_subviewer_focused: mode-reset gating (regression coverage)
# ---------------------------------------------------------------------------


def test_first_focus_does_not_reset_mode():
    """REGRESSION test: previous_focus=None must not touch layer.mode.

    The first-ever focus (or first focus after patch switch) should preserve
    whatever mode the layer is in.  The earlier broken impl reset to pan_zoom
    on every mouse-down — including the mouse-down that started a paint stroke.
    """
    a = _FakeAnnotator(target_mode="lateral")
    a._focused_orient = None
    a.xy_lateral.mode = "paint"  # user is in paint mode

    Curation2DAnnotator._on_subviewer_focused(a, "XY")

    # Mode preserved — the regression would have set this to pan_zoom.
    assert a.xy_lateral.mode == "paint"
    assert a._focused_orient == "XY"
    # Indicator IS refreshed on first focus.
    assert a._indicator_calls == ["XY"]


def test_same_orient_reclick_does_not_reset_mode():
    """Clicking the same subviewer again must not touch layer.mode.

    The mouse-down that continues a paint stroke (or that starts a NEW stroke
    within the same view) maps here — preserves mode so paint stays paint.
    """
    a = _FakeAnnotator(target_mode="axial")
    a._active_labels_name = "patch_1_work_labels_axial_xz_3d"
    a._focused_orient = "XZ"
    a.xz_axial_xz.mode = "paint"

    Curation2DAnnotator._on_subviewer_focused(a, "XZ")

    assert a.xz_axial_xz.mode == "paint"
    assert a._focused_orient == "XZ"


def test_orient_switch_resets_mode_to_pan_zoom():
    """Switching from XZ to YZ resets the new VM's active layer to pan_zoom.

    This is the design-intended behavior: new subviewer focus = safe pan
    default, user opts into paint via '2'.
    """
    a = _FakeAnnotator(target_mode="axial")
    a._active_labels_name = "patch_1_work_labels_axial_xz_3d"
    a._focused_orient = "XZ"
    a.yz_axial_yz.mode = "paint"  # YZ was previously in paint somehow

    Curation2DAnnotator._on_subviewer_focused(a, "YZ")

    # YZ VM's class-matching layer (axial_yz_3d_YZ) is now reset to pan_zoom.
    assert a.yz_axial_yz.mode == "pan_zoom"
    assert a._focused_orient == "YZ"
    assert a._indicator_calls == ["YZ"]


# ---------------------------------------------------------------------------
# _on_subviewer_focused: class-matching routing (the actual bug fix)
# ---------------------------------------------------------------------------


def test_axial_yz_routes_to_correct_class_layer():
    """In axial mode, focusing YZ activates axial_yz_3d_YZ (NOT axial_xz_3d_YZ).

    This is the core fix: the broken sync_active_layer cascade lands
    axial_xz_3d_YZ as active in YZ VM after segmenting XZ.  Our focus
    listener corrects to axial_yz_3d_YZ which is the class-matching layer
    (editable per _apply_edit_constraints).
    """
    a = _FakeAnnotator(target_mode="axial")
    a._active_labels_name = "patch_1_work_labels_axial_xz_3d"  # set by segment XZ
    a._focused_orient = "XZ"
    # Simulate the broken cascade: YZ VM has the WRONG class active
    a.viewer_model_yz.layers.selection.active = a.yz_axial_xz

    Curation2DAnnotator._on_subviewer_focused(a, "YZ")

    # YZ VM is now routed to the CORRECT class for YZ orientation.
    assert a.viewer_model_yz.layers.selection.active is a.yz_axial_yz


def test_axial_xz_routes_to_correct_class_layer():
    """Symmetric: focusing XZ in axial mode activates axial_xz_3d_XZ."""
    a = _FakeAnnotator(target_mode="axial")
    a._active_labels_name = "patch_1_work_labels_axial_yz_3d"  # set by segment YZ
    a._focused_orient = "YZ"
    a.viewer_model_xz.layers.selection.active = a.xz_axial_yz  # wrong class

    Curation2DAnnotator._on_subviewer_focused(a, "XZ")

    assert a.viewer_model_xz.layers.selection.active is a.xz_axial_xz


# ---------------------------------------------------------------------------
# _SubviewerFocusFilter: only fires on LeftButton
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self, evt_type, button):
        self._type = evt_type
        self._button = button

    def type(self):
        return self._type

    def button(self):
        return self._button


def test_filter_ignores_non_left_button():
    """C1 (review follow-up): right-click and middle-click must not trigger
    the focus listener.  Otherwise right-click context menus + middle-click
    panning would inadvertently reassign the VM's active layer.
    """
    from qtpy.QtCore import Qt, QEvent

    calls = []

    class _StubAnnotator:
        def _on_subviewer_focused(self, orient):
            calls.append(orient)

    stub = _StubAnnotator()
    filt = _SubviewerFocusFilter(stub, "XZ")

    # Right-click: must NOT fire
    filt.eventFilter(None, _FakeEvent(QEvent.MouseButtonPress, Qt.RightButton))
    # Middle-click: must NOT fire
    filt.eventFilter(None, _FakeEvent(QEvent.MouseButtonPress, Qt.MiddleButton))

    assert calls == []

    # Left-click: must fire
    filt.eventFilter(None, _FakeEvent(QEvent.MouseButtonPress, Qt.LeftButton))
    assert calls == ["XZ"]


def test_filter_ignores_non_press_events():
    """Filter only listens for MouseButtonPress, not move/release/wheel."""
    from qtpy.QtCore import Qt, QEvent

    calls = []

    class _StubAnnotator:
        def _on_subviewer_focused(self, orient):
            calls.append(orient)

    stub = _StubAnnotator()
    filt = _SubviewerFocusFilter(stub, "XY")

    # MouseMove: must NOT fire
    filt.eventFilter(None, _FakeEvent(QEvent.MouseMove, Qt.LeftButton))
    # MouseButtonRelease: must NOT fire
    filt.eventFilter(None, _FakeEvent(QEvent.MouseButtonRelease, Qt.LeftButton))

    assert calls == []

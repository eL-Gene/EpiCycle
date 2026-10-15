"""Tests for the 3D curation annotator launcher.

Phase K5.7.
"""

from __future__ import annotations

import importlib
import inspect
import types

import pytest

from napari_epicycle.annotator.launcher import (
    launch_3d_curation_annotator,
    launch_native_multiview_annotator,
)


# ---------------------------------------------------------------------------
# Fake objects for launcher testing (no Qt / napari required)
# ---------------------------------------------------------------------------

class _FakeDock:
    def __init__(self):
        self.shown = False
        self.raised = False
        self._widget = self

    def show(self):
        self.shown = True

    def raise_(self):
        self.raised = True

    def isVisible(self):
        return True

    def widget(self):
        return self._widget

    def windowTitle(self):
        return ""

    # Public API stubs (set_context, set_focused_refinement)
    def set_context(self, **kwargs):
        self._last_context = kwargs

    def set_focused_refinement(self, ids, centroids):
        self._last_flagged = (ids, centroids)


class _FakeWindow:
    def __init__(self):
        self._dock_widgets = {}
        self.add_calls = []

    def add_dock_widget(self, widget, name=None, area=None):
        self.add_calls.append((widget, name, area))
        self._dock_widgets[name] = _FakeDock()


class _FakeViewer:
    def __init__(self):
        self.window = _FakeWindow()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDockNameSeparation:
    def test_dock_names_differ(self):
        """HITL 2D and 3D curation annotators must use different dock names."""
        hitl_sig = inspect.signature(launch_native_multiview_annotator)
        curation_sig = inspect.signature(launch_3d_curation_annotator)
        hitl_default = hitl_sig.parameters["dock_name"].default
        curation_default = curation_sig.parameters["dock_name"].default
        assert hitl_default != curation_default
        assert hitl_default == "EpiCycle 2D Training Annotator"
        assert curation_default == "EpiCycle 3D Curation Annotator"


class TestLaunch3DCuration:
    def test_no_viewer_returns_false(self):
        ok, msg = launch_3d_curation_annotator(None)
        assert ok is False
        assert "Viewer" in msg

    def test_no_window_returns_false(self):
        viewer = types.SimpleNamespace(window=None)
        ok, msg = launch_3d_curation_annotator(viewer)
        assert ok is False

    def test_reuse_existing_dock(self):
        viewer = _FakeViewer()
        dock = _FakeDock()
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock
        ok, msg = launch_3d_curation_annotator(viewer, labels_layer_name="lbl")
        assert ok is True
        assert "already open" in msg
        assert dock.raised is True

    def test_visible_reuse_skips_global_orphan_sweep(self, monkeypatch):
        """Visible dock reuse should not run global orphan sweep.

        Launch-time global sweeps are expensive and contribute to close/relaunch
        churn. Reuse path should wire callbacks and return quickly.
        """
        viewer = _FakeViewer()
        dock = _FakeDock()
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        sweep_calls = []
        monkeypatch.setattr(
            "napari_epicycle.annotator.launcher._close_orphan_magicgui_windows_global",
            lambda: sweep_calls.append(True),
        )

        ok, msg = launch_3d_curation_annotator(viewer, labels_layer_name="lbl")
        assert ok is True
        assert "already open" in msg
        assert sweep_calls == [], (
            "Visible dock reuse should not trigger global orphan sweep"
        )

    def test_launch_new_dock_correct_name(self, monkeypatch):
        class _FakeWidget:
            def __init__(self, viewer):
                self.viewer = viewer
            def set_context(self, **kwargs):
                self.ctx = kwargs

        fake_module = types.SimpleNamespace(Curation3DAnnotator=_FakeWidget)
        monkeypatch.setattr(importlib, "import_module", lambda _name: fake_module)
        viewer = _FakeViewer()
        ok, msg = launch_3d_curation_annotator(viewer, labels_layer_name="test_labels")
        assert ok is True
        assert "Opened 3D curation annotator" in msg
        assert len(viewer.window.add_calls) == 1
        _, name, area = viewer.window.add_calls[0]
        assert name == "EpiCycle 3D Curation Annotator"
        assert area == "right"

    def test_flagged_data_passed_to_reused_dock(self):
        viewer = _FakeViewer()
        dock = _FakeDock()
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock
        launch_3d_curation_annotator(
            viewer,
            flagged_label_ids=[1, 5],
            flagged_centroids={1: (0, 0, 0), 5: (1, 2, 3)},
        )
        assert hasattr(dock, "_last_flagged")
        assert dock._last_flagged[0] == [1, 5]

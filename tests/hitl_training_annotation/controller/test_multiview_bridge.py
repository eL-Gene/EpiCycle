import types

from napari_epicycle.hitl_training_annotation.controller.multiview_bridge import (
    launch_multiview_annotator,
)


class _FakeWindow:
    def __init__(self):
        self._dock_widgets = {}
        self.calls = []

    def add_dock_widget(self, widget, name=None, area=None):
        self.calls.append((widget, name, area))
        self._dock_widgets[name] = widget


class _FakeViewer:
    def __init__(self):
        self.window = _FakeWindow()


def test_launch_multiview_bridge_import_missing(monkeypatch):
    import importlib

    def _fail_import(_name):
        raise ImportError("missing")

    monkeypatch.setattr(importlib, "import_module", _fail_import)
    ok, msg = launch_multiview_annotator(_FakeViewer())
    assert ok is False
    assert "not importable" in msg


def test_launch_multiview_bridge_success(monkeypatch):
    import importlib

    class _Widget:
        def __init__(self, viewer):
            self.viewer = viewer

    fake_mod = types.SimpleNamespace(MultiviewAnnotator=_Widget)
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake_mod)

    viewer = _FakeViewer()
    ok, msg = launch_multiview_annotator(viewer)
    assert ok is True
    assert "Opened" in msg
    assert len(viewer.window.calls) == 1
    _, name, area = viewer.window.calls[0]
    assert name == "EpiCycle Multiview Annotator"
    assert area == "right"


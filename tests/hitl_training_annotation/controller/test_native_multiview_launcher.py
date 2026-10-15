from napari_epicycle.annotator.launcher import launch_native_multiview_annotator


class _FakeDock:
    def __init__(self):
        self.shown = False
        self.raised = False
        self.context_calls = []

    def show(self):
        self.shown = True

    def raise_(self):
        self.raised = True

    def isVisible(self):
        return True

    def set_context(self, **kwargs):
        self.context_calls.append(kwargs)


class _FakeWindow:
    def __init__(self):
        self._dock_widgets = {}
        self.calls = []

    def add_dock_widget(self, widget, name=None, area=None):
        self.calls.append((widget, name, area))
        self._dock_widgets[name] = _FakeDock()


class _FakeViewer:
    def __init__(self):
        self.window = _FakeWindow()


def test_launch_native_multiview_success(monkeypatch):
    import types
    import importlib

    class _Widget:
        def __init__(self, viewer):
            self.viewer = viewer
            self.context_calls = []

        def set_context(self, **kwargs):
            self.context_calls.append(kwargs)

    fake_module = types.SimpleNamespace(NativeMultiviewAnnotator=_Widget)
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake_module)
    viewer = _FakeViewer()
    ok, msg = launch_native_multiview_annotator(
        viewer,
        image_layer_name="img",
        labels_layer_name="lbl",
    )
    assert ok is True
    assert "Opened native multiview annotator." in msg
    assert len(viewer.window.calls) == 1
    _, name, area = viewer.window.calls[0]
    assert name == "EpiCycle 2D Training Annotator"
    assert area == "right"


def test_launch_native_multiview_reuses_existing():
    viewer = _FakeViewer()
    dock = _FakeDock()
    viewer.window._dock_widgets["EpiCycle 2D Training Annotator"] = dock
    ok, msg = launch_native_multiview_annotator(
        viewer,
        image_layer_name="img1",
        labels_layer_name="lbl1",
    )
    assert ok is True
    assert "already open" in msg
    assert dock.raised is True
    assert dock.context_calls

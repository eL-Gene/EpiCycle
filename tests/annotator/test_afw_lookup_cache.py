"""P1 (Add rebuild final fix pass): the Advanced Features dock lookup is cached.

``launcher._find_dock_by_title`` walks ``findChildren(QDockWidget)`` on the
main window -- ~31 ms per call on the owner's session -- and the annotator
asked for the panel 14-20 times per user action.  The resolved (dock, widget)
pair is now cached on the annotator; a read is one attribute read plus a
liveness check, and a dock that has gone (deleted, unparented, or holding a
different widget) falls back to the search.
"""

from __future__ import annotations

from napari_epicycle.annotator import curation_3d_widget as CW
from napari_epicycle.annotator.launcher import _ADVANCED_DOCK_NAMES

C = CW.Curation3DAnnotator
TITLE = _ADVANCED_DOCK_NAMES["3d_curation"]


class _Dock:
    def __init__(self, widget, title=TITLE):
        self._w = widget
        self._title = title
        self._parent = object()

    def windowTitle(self):
        return self._title

    def widget(self):
        return self._w

    def parent(self):
        return self._parent


class _Main:
    def __init__(self, docks):
        self.docks = list(docks)
        self.calls = 0

    def findChildren(self, _cls):
        self.calls += 1
        return list(self.docks)


class _Window:
    def __init__(self, main):
        self._qt_window = main
        self.dock_widgets = {}


class _Viewer:
    def __init__(self, main):
        self.window = _Window(main)


class _Ann:
    _get_advanced_features_widget = C._get_advanced_features_widget
    _invalidate_afw_cache = C._invalidate_afw_cache

    def __init__(self, viewer):
        self.viewer = viewer


def _make():
    panel = object()
    dock = _Dock(panel)
    main = _Main([dock])
    return _Ann(_Viewer(main)), main, dock, panel


def test_one_lookup_per_session_not_per_call():
    ann, main, _dock, panel = _make()
    for _ in range(20):
        assert ann._get_advanced_features_widget() is panel
    assert main.calls == 1


def test_unparented_dock_falls_back_to_the_search():
    ann, main, dock, panel = _make()
    assert ann._get_advanced_features_widget() is panel
    # napari's remove_dock_widget unparents the dock; a new one is added.
    dock._parent = None
    panel2 = object()
    main.docks = [_Dock(panel2)]
    assert ann._get_advanced_features_widget() is panel2
    assert main.calls == 2
    assert ann._get_advanced_features_widget() is panel2
    assert main.calls == 2


def test_dock_holding_a_different_widget_falls_back():
    ann, main, dock, panel = _make()
    ann._get_advanced_features_widget()
    panel2 = object()
    dock._w = panel2
    assert ann._get_advanced_features_widget() is panel2
    assert main.calls == 2


def test_deleted_dock_falls_back(monkeypatch):
    ann, main, dock, panel = _make()
    ann._get_advanced_features_widget()
    monkeypatch.setattr(CW, "_qobject_is_deleted", lambda o: o is dock)
    main.docks = []
    assert ann._get_advanced_features_widget() is None
    assert main.calls == 2
    assert getattr(ann, "_afw_cache", None) is None


def test_no_dock_is_not_cached():
    main = _Main([])
    ann = _Ann(_Viewer(main))
    assert ann._get_advanced_features_widget() is None
    assert ann._get_advanced_features_widget() is None
    assert main.calls == 2


def test_invalidate_clears_the_cache():
    ann, main, _dock, _panel = _make()
    ann._get_advanced_features_widget()
    ann._invalidate_afw_cache()
    ann._get_advanced_features_widget()
    assert main.calls == 2


def test_reset_add_state_and_launcher_cleanup_clear_the_cache():
    import inspect
    from napari_epicycle.annotator import launcher as L

    assert "self._afw_cache = None" in inspect.getsource(C._reset_add_state)
    assert "_afw_cache" in inspect.getsource(L._run_widget_cleanup)


def test_real_qt_deleted_dock_is_seen_as_deleted():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication, QDockWidget, QWidget

    app = QApplication.instance() or QApplication([])
    d = QDockWidget()
    d.setWidget(QWidget())
    assert CW._qobject_is_deleted(d) is False
    from qtpy import sip
    sip.delete(d)
    assert CW._qobject_is_deleted(d) is True
    # A plain Python object is never "deleted".
    assert CW._qobject_is_deleted(object()) is False
    del app

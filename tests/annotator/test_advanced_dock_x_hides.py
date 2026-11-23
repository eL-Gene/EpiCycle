"""napari's title-bar X on the Advanced Features dock only HIDES it.

napari's X calls the dock's ``destroyOnClose()``, which removes the dock from
the window and unparents the panel.  No EpiCycle code runs on that path, so
an open Add / Reshape / Refine edit was stranded: its lock still held and no
Accept / Cancel left on screen.  ``launch_advanced_features_dock`` now
replaces ``destroyOnClose`` on its dock with a plain hide; the panel, its
open edit and its state stay, and "Advanced Features..." shows the same
panel again.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

from napari_epicycle.annotator import curation_3d_widget as CW
from napari_epicycle.annotator import launcher as L
from napari_epicycle.annotator.advanced_features_widget import (
    AdvancedFeaturesWidget,
)

TITLE = L._ADVANCED_DOCK_NAMES["3d_curation"]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _is_deleted(obj) -> bool:
    try:
        from qtpy import sip
    except Exception:  # pragma: no cover
        import sip  # type: ignore
    try:
        return bool(sip.isdeleted(obj))
    except Exception:
        return False


def _flush():
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


class _Window:
    """napari's window over a REAL ``QMainWindow``; docks are napari's own
    ``QtViewerDockWidget`` so the title bar's X is the real button."""

    def __init__(self, viewer, main):
        self._viewer = viewer
        self._qt_window = main
        self._dock_widgets = {}
        self.removed = []
        # napari's dock reaches the viewer through the qt viewer it is given.
        self._qt_viewer = QWidget(main)
        self._qt_viewer.viewer = viewer

    def add_dock_widget(self, widget, name=None, area="right"):
        from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget

        dock = QtViewerDockWidget(self._qt_viewer, widget, name=name, area=area)
        self._qt_window.addDockWidget(dock.qt_area, dock)
        self._dock_widgets[name] = dock
        return dock

    def remove_dock_widget(self, dock):
        # What napari's Window.remove_dock_widget does to a dock.
        self.removed.append(dock)
        for key, value in list(self._dock_widgets.items()):
            if value is dock:
                del self._dock_widgets[key]
        self._qt_window.removeDockWidget(dock)
        inner = dock.widget()
        if inner is not None:
            inner.setParent(None)
        dock.setParent(None)
        dock.deleteLater()


class _Viewer:
    def __init__(self):
        self.main = QMainWindow()
        self.window = _Window(self, self.main)


class _Ann:
    _get_advanced_features_widget = (
        CW.Curation3DAnnotator._get_advanced_features_widget
    )

    def __init__(self, viewer):
        self.viewer = viewer
        self._afw_cache = None


@pytest.fixture
def viewer(qapp):
    """The fake viewer; its window is closed and deleted at teardown."""
    v = _Viewer()
    yield v
    v.main.close()
    v.main.deleteLater()
    qapp.processEvents()
    _flush()


@pytest.fixture
def napari_viewer_class(monkeypatch):
    """napari's ``destroyOnClose`` acts only on a real ``napari.Viewer``;
    let it see the fake one so the unpatched X removes the dock as live."""
    import napari.viewer

    monkeypatch.setattr(napari.viewer, "Viewer", _Viewer)


def _the_dock(viewer, title=TITLE):
    docks = [
        d for d in viewer.main.findChildren(QDockWidget)
        if not _is_deleted(d) and d.windowTitle() == title
    ]
    assert len(docks) == 1
    return docks[0]


def test_launch_installs_the_hide_on_close_hook(qapp, napari_viewer_class, viewer):
    widget, ok, msg = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    assert ok, msg
    dock = _the_dock(viewer)
    assert "destroyOnClose" in vars(dock)
    assert getattr(dock, "_epicycle_hide_on_close", None) is not None


def test_the_x_hides_the_dock_and_keeps_the_panel(qapp, napari_viewer_class, viewer):
    ann = _Ann(viewer)
    widget, ok, msg = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    assert ok, msg
    dock = _the_dock(viewer)
    assert hasattr(dock.title, "close_button")

    dock.title.close_button.click()  # napari's X
    _flush()

    assert viewer.window.removed == []
    assert not _is_deleted(dock)
    assert not _is_deleted(widget)
    assert dock.isHidden()
    assert dock.parent() is viewer.main
    assert dock.widget() is widget
    # The annotator still reaches the hidden panel (and its open edit).
    assert ann._get_advanced_features_widget() is widget


def test_advanced_features_reshows_the_same_hidden_panel(qapp, napari_viewer_class, viewer):
    widget, ok, _msg = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    assert ok
    dock = _the_dock(viewer)
    dock.title.close_button.click()
    _flush()
    assert dock.isHidden()

    again, ok2, msg2 = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    _flush()

    assert ok2, msg2
    assert again is widget
    assert isinstance(again, AdvancedFeaturesWidget)
    assert "already open" in msg2
    assert _the_dock(viewer) is dock
    assert not dock.isHidden()
    # The reused dock keeps its hook; a second X still only hides.
    dock.title.close_button.click()
    _flush()
    assert viewer.window.removed == []
    assert dock.isHidden() and not _is_deleted(widget)


def test_hook_is_installed_once(qapp, napari_viewer_class, viewer):
    L.launch_advanced_features_dock(viewer, mode="3d_curation")
    dock = _the_dock(viewer)
    first = dock.destroyOnClose
    L.launch_advanced_features_dock(viewer, mode="3d_curation")
    assert dock.destroyOnClose is first


def test_float_and_redock_keep_the_hook(qapp, napari_viewer_class, viewer,
                                        monkeypatch):
    """napari rebuilds the title bar on float / re-dock; the new X button
    still finds the override (it calls ``self.parent().destroyOnClose()``)."""
    monkeypatch.setattr(L, "_gl_contexts_shared", lambda: True)
    widget, ok, _msg = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    assert ok
    dock = _the_dock(viewer)
    dock._update_title_bar(is_floating=not dock.isFloating())
    dock.title.close_button.click()
    _flush()
    assert viewer.window.removed == []
    assert dock.isHidden() and not _is_deleted(widget)


def test_napari_x_removes_a_dock_without_the_hook(qapp, napari_viewer_class,
                                                  viewer):
    """Teeth: with no hook, napari's own X removes the dock -- what the hook
    prevents on the 3D curation dock."""
    widget, ok, _msg = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    assert ok
    dock = _the_dock(viewer)
    del dock.destroyOnClose               # napari's own method again
    dock.title.close_button.click()
    assert viewer.window.removed == [dock]


def test_the_2d_dock_gets_no_hook(qapp, napari_viewer_class, viewer):
    """The standalone 2D annotator never tears its Advanced dock down, so a
    hidden one would outlive it: its X keeps napari's own removal."""
    widget, ok, msg = L.launch_advanced_features_dock(viewer, mode="hitl_2d")
    assert ok, msg
    dock = _the_dock(viewer, L._ADVANCED_DOCK_NAMES["hitl_2d"])
    assert "destroyOnClose" not in vars(dock)
    assert getattr(dock, "_epicycle_hide_on_close", None) is None
    dock.title.close_button.click()
    assert viewer.window.removed == [dock]

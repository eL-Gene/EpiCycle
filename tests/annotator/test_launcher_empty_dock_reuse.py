"""A dock left EMPTY by napari's title-bar X must never be "reused".

Closing the Advanced Features dock with napari's X removes the panel from
napari's registry but can leave an empty ``QDockWidget`` with the same title
attached to the main window.  ``_resolve_existing_dock`` found it by title,
``launch_advanced_features_dock`` "reused" it, an empty floating dock appeared
and ``_get_advanced_features_widget()`` answered None -- Reshape / Refine / Add
unreachable until the annotator was relaunched (live driver, part G / F5).

The launchers now purge an empty (or foreign) dock with that title and build
a fresh one as on a first launch; a live dock is still reused.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtWidgets import QApplication, QDockWidget, QLabel, QMainWindow

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


class _Window:
    """napari's window, reduced to what the launcher touches, over a REAL
    ``QMainWindow`` so ``findChildren(QDockWidget)`` is the real search."""

    def __init__(self, main):
        self._qt_window = main
        self._dock_widgets = {}

    def add_dock_widget(self, widget, name=None, area="right"):
        dock = QDockWidget(name, self._qt_window)
        dock.setWidget(widget)
        self._qt_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._dock_widgets[name] = dock
        return dock

    def remove_dock_widget(self, dock):
        for key, value in list(self._dock_widgets.items()):
            if value is dock:
                del self._dock_widgets[key]
        self._qt_window.removeDockWidget(dock)


class _Viewer:
    def __init__(self):
        self.main = QMainWindow()
        self.window = _Window(self.main)


class _Ann:
    _get_advanced_features_widget = (
        CW.Curation3DAnnotator._get_advanced_features_widget
    )

    def __init__(self, viewer):
        self.viewer = viewer
        self._afw_cache = None


def _flush():
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _docks_titled(main, title=TITLE):
    return [
        d for d in main.findChildren(QDockWidget)
        if not _is_deleted(d) and d.windowTitle() == title
    ]


def _leave_empty_dock(viewer):
    """What napari's X leaves behind: a titled dock holding nothing."""
    stale = QDockWidget(TITLE, viewer.main)
    viewer.main.addDockWidget(Qt.RightDockWidgetArea, stale)
    assert stale.widget() is None
    return stale


def test_empty_dock_is_not_reused_a_fresh_panel_is_built(qapp):
    viewer = _Viewer()
    stale = _leave_empty_dock(viewer)

    widget, ok, msg = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    _flush()

    assert ok is True, msg
    assert isinstance(widget, AdvancedFeaturesWidget)
    assert "already open" not in msg
    # The empty dock is gone; the one left holds the new panel.
    assert _is_deleted(stale) or stale not in _docks_titled(viewer.main)
    live = _docks_titled(viewer.main)
    assert len(live) == 1
    assert live[0].widget() is widget


def test_dock_holding_a_foreign_widget_is_not_reused(qapp):
    viewer = _Viewer()
    stale = QDockWidget(TITLE, viewer.main)
    stale.setWidget(QLabel("not the panel"))
    viewer.main.addDockWidget(Qt.RightDockWidgetArea, stale)

    widget, ok, msg = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    _flush()

    assert ok is True, msg
    assert isinstance(widget, AdvancedFeaturesWidget)
    live = _docks_titled(viewer.main)
    assert [d.widget() for d in live] == [widget]


def test_live_dock_is_still_reused(qapp):
    viewer = _Viewer()
    first, ok1, _ = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    second, ok2, msg2 = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    _flush()

    assert ok1 and ok2
    assert second is first
    assert "already open" in msg2
    assert len(_docks_titled(viewer.main)) == 1
    assert not _is_deleted(first)


def test_annotator_finds_the_fresh_panel_after_the_rebuild(qapp):
    viewer = _Viewer()
    ann = _Ann(viewer)
    old, _ok, _ = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    assert ann._get_advanced_features_widget() is old
    old_dock = _docks_titled(viewer.main)[0]

    # napari's X: the panel is taken out and destroyed, the dock stays, empty.
    viewer.window.remove_dock_widget(old_dock)
    old_dock.setWidget(None) if hasattr(old_dock, "setWidget") else None
    old.setParent(None)
    old.deleteLater()
    _flush()
    viewer.main.addDockWidget(Qt.RightDockWidgetArea, old_dock)
    assert old_dock.widget() is None

    new, ok, msg = L.launch_advanced_features_dock(viewer, mode="3d_curation")
    _flush()

    assert ok is True, msg
    assert new is not old
    assert isinstance(new, AdvancedFeaturesWidget)
    assert ann._get_advanced_features_widget() is new


def test_find_dock_by_title_skips_an_empty_dock(qapp):
    viewer = _Viewer()
    _leave_empty_dock(viewer)
    live = viewer.window.add_dock_widget(QLabel("x"), name=TITLE)
    assert L._find_dock_by_title(viewer, TITLE) is live
    assert L._resolve_existing_dock(viewer, TITLE) is live


def test_empty_navigator_dock_is_purged_not_reused(qapp):
    viewer = _Viewer()
    stale = QDockWidget(L._NAVIGATOR_DOCK_NAME, viewer.main)
    viewer.main.addDockWidget(Qt.RightDockWidgetArea, stale)

    built = []

    class _NavAnn:
        _cached_flagged_items = None
        _active_labels_name = ""

        def __init__(self):
            self.viewer = viewer

        def _create_navigator(self):
            w = QLabel("nav")
            built.append(w)
            return w

        def _layer_by_name(self, layers, name):
            return None

    ann = _NavAnn()
    widget, ok, msg = L.launch_curation_label_navigator(viewer, ann)
    _flush()

    assert ok is True, msg
    assert built and widget is built[0]
    assert "already open" not in msg
    docks = _docks_titled(viewer.main, L._NAVIGATOR_DOCK_NAME)
    assert [d.widget() for d in docks] == [widget]

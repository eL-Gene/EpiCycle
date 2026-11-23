"""The floating Advanced Features dock opens INSIDE the main window's screen.

A live run on two monitors stacked vertically (nothing above y = 0) found
the floating dock at (0, -300): its title bar (X, hide), the page header,
S1 and the box card sat above every screen, out of reach of the mouse.
``launch_advanced_features_dock`` floated and resized the dock but never
placed it.  It now puts a NEW floating dock at the main window's top-right
corner, kept inside the available area of the main window's screen; a dock
that is shown again keeps wherever the user put it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from qtpy.QtCore import QRect

from napari_epicycle.annotator import launcher as L

from .test_advanced_dock_x_hides import (  # noqa: F401 - fixtures
    _the_dock,
    napari_viewer_class,
    qapp,
    viewer,
)


@pytest.mark.parametrize("win, avail, size, want", [
    # the top-right corner of the window, inside the screen
    ((100, 50, 900, 700), (0, 0, 2880, 1800), (360, 500), (640, 50)),
    # a window reaching above every screen: pushed down to the screen top
    ((0, -300, 900, 700), (0, 0, 2880, 1800), (360, 500), (540, 0)),
    # a window on the lower monitor: stays on that monitor
    ((200, 1850, 1200, 800), (0, 1800, 2880, 864), (360, 500), (1040, 1850)),
    # a dock taller than the room left below the window top: lifted
    ((0, 1500, 900, 300), (0, 0, 2880, 1800), (360, 500), (540, 1300)),
    # a window hanging off the right edge: the dock stays on screen
    ((2700, 10, 900, 700), (0, 0, 2880, 1800), (360, 500), (2520, 10)),
    # a dock bigger than the screen: its top-left corner stays visible
    ((0, 0, 900, 700), (0, 0, 300, 400), (360, 500), (0, 0)),
])
def test_the_origin_is_the_window_top_right_kept_on_screen(win, avail, size,
                                                          want):
    assert L._floating_dock_origin(win, avail, size) == want


class _Screen:
    def __init__(self, rect):
        self._rect = rect

    def availableGeometry(self):           # noqa: N802 - Qt name
        return self._rect


def _launch(viewer, monkeypatch, avail):
    seen = []

    def fake_screen(widget):
        seen.append(widget)
        return _Screen(avail)

    monkeypatch.setattr(L, "_screen_of", fake_screen)
    widget, ok, msg = L.launch_advanced_features_dock(viewer,
                                                      mode="3d_curation")
    assert ok, msg
    return _the_dock(viewer), seen


def test_a_new_floating_dock_opens_inside_the_main_windows_screen(
        qapp, napari_viewer_class, viewer, monkeypatch):
    viewer.main.setGeometry(0, -300, 900, 700)   # partly above every screen
    avail = QRect(0, 0, 2880, 1800)
    dock, seen = _launch(viewer, monkeypatch, avail)
    assert seen and seen[0] is viewer.main, "the MAIN WINDOW's screen"
    assert dock.isFloating()
    g = dock.frameGeometry()
    assert avail.contains(g.topLeft()), (g, avail)
    assert g.y() >= 0, "the title bar is reachable"
    assert g.x() + g.width() <= avail.x() + avail.width()


def test_the_height_cap_reads_the_same_screen(
        qapp, napari_viewer_class, viewer, monkeypatch):
    viewer.main.setGeometry(0, 0, 900, 700)
    dock, _seen = _launch(viewer, monkeypatch, QRect(0, 0, 1200, 400))
    assert dock.height() <= int(400 * 0.85)
    assert dock.y() >= 0


def test_a_reshown_dock_keeps_where_the_user_put_it(
        qapp, napari_viewer_class, viewer, monkeypatch):
    viewer.main.setGeometry(100, 50, 900, 700)
    dock, _seen = _launch(viewer, monkeypatch, QRect(0, 0, 2880, 1800))
    dock.move(321, 123)                     # the user drags it
    dock.hide()
    L.launch_advanced_features_dock(viewer, mode="3d_curation")
    assert not dock.isHidden()
    assert (dock.x(), dock.y()) == (321, 123)

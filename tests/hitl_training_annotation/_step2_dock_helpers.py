"""Helpers shared by several Step 2 test files: dock a widget in a scroll area
and open an accordion section.  This module is not a test."""
from qtpy.QtWidgets import QScrollArea


def _in_dock(qapp, w, width=576, height=578):
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(w)
    sa.resize(width, height)
    sa.show()
    qapp.processEvents()
    return sa


def _open(qapp, w, index):
    from napari_epicycle.ui_shell.theme import Accordion
    acc = w.findChildren(Accordion)[0]
    acc.setCurrentIndex(index)
    qapp.processEvents()
    qapp.processEvents()

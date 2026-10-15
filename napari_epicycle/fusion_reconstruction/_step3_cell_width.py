"""The two lines under Step 3's cell width ("Typical cell width ... µm + ≈ N pixels at M µm per pixel",
brick R2b part 2).

A small widget kept OUT of ``ui.py`` (brick R3 owns that file while R2b
is built): the dim "≈ N pixels at M µm per pixel" line for the
highlighted / first patch with a voxel size, and the amber warning "No
voxel size — cell width falls back to pixels." while the width is not Auto
and a patch Predict would run has none.  The controller pushes the two
texts through :meth:`CellWidthLines.show`; every sentence is a
``_step3_pr_text`` constant.

``make_cell_width_lines(ui)`` builds it and stores it on the panel as
``ui.pr_cell_width_lines``; R3's ``_build_pr_settings`` places it in the
row right under the width, above batch size.  ``ensure_cell_width_lines(ui)``
returns the panel's own; only for a panel that built none (a test's bare
grid) it puts one in the width's grid itself, spanning both columns of a
new last row.
"""
from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from ..ui_shell.theme import DIM_LABEL_STYLE, PANEL_WARN_STYLE

#: the attribute the panel keeps the widget under
ATTR = "pr_cell_width_lines"


class CellWidthLines(QWidget):
    """The "≈ N pixels" line (dim) over the no-voxel warning (amber); each
    hidden while its text is empty."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("step3CellWidthLines")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.px_label = QLabel(self)
        self.px_label.setWordWrap(True)
        self.px_label.setStyleSheet(DIM_LABEL_STYLE)
        self.warn_label = QLabel(self)
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet(PANEL_WARN_STYLE)
        lay.addWidget(self.px_label)
        lay.addWidget(self.warn_label)
        self.show_lines("", "")

    def show_lines(self, px_text: str, warning: str) -> None:
        self.px_label.setText(px_text)
        self.px_label.setVisible(bool(px_text))
        self.warn_label.setText(warning)
        self.warn_label.setVisible(bool(warning))
        self.setVisible(bool(px_text or warning))


def make_cell_width_lines(ui, parent: Optional[QWidget] = None
                          ) -> CellWidthLines:
    """Build the lines and keep them on *ui* (R3's one call)."""
    lines = CellWidthLines(parent)
    setattr(ui, ATTR, lines)
    return lines


def _grid_holding(widget) -> Optional[QGridLayout]:
    """The grid layout *widget* sits in (searched from its parent's layout
    down), or None."""
    parent = widget.parentWidget() if widget is not None else None
    stack = [parent.layout()] if parent is not None and parent.layout() \
        else []
    while stack:
        lay = stack.pop()
        if isinstance(lay, QGridLayout) and lay.indexOf(widget) >= 0:
            return lay
        for i in range(lay.count()):
            child = lay.itemAt(i).layout()
            if child is not None:
                stack.append(child)
    return None


def ensure_cell_width_lines(ui) -> Optional[CellWidthLines]:
    """The panel's lines; built and placed in the cell width's grid (a new
    last row, both columns) when the panel has none.  None when the width
    box is not in a grid (nothing is placed blindly)."""
    lines = getattr(ui, ATTR, None)
    if lines is not None:
        return lines
    box = getattr(ui, "diameter_sb", None)
    grid = _grid_holding(box)
    if grid is None:
        return None
    lines = make_cell_width_lines(ui, box.parentWidget())
    grid.addWidget(lines, grid.rowCount(), 0, 1, 2)
    return lines

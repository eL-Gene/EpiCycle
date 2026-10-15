"""Shared UI theme constants for EpiCycle.

Provides a consistent visual language across all step widgets.
"""

from pathlib import Path as _Path

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QStyledItemDelegate, QTextBrowser, QVBoxLayout, QWidget,
)


# ── Typography ──
# Monospace for data values (paths, counts, model names) — improves scanability.
MONO_FONT = "font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;"
MONO_LABEL_STYLE = f"color: #aaaaaa; font-size: 11px; {MONO_FONT}"

# ── Accordion widget (replaces QToolBox — immune to napari's global theme) ──


def _qta_icon_or_none(icon_name: str, color: str):
    """Return a qtawesome QIcon, or None if unavailable."""
    if not icon_name:
        return None
    try:
        import qtawesome as qta
        return qta.icon(icon_name, color=color)
    except Exception:
        return None


class AccordionSection(QWidget):
    """A single collapsible section — header button + content panel."""

    def __init__(self, title: str, widget: QWidget, expanded: bool = False,
                 accent: str = "#d4a843", icon_name: str = "",
                 parent: QWidget = None) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = expanded
        self._accent = accent
        self._icon_name = icon_name

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(0)

        self._header = QPushButton(title)
        self._header.setFlat(True)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header.setFixedHeight(34)
        self._header.setIconSize(QSize(16, 16))
        self._header.clicked.connect(self._toggle)
        layout.addWidget(self._header)

        # Wrap content in a scroll area so each tab scrolls independently.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setWidget(widget)
        self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._scroll, 1)

        self._scroll.setVisible(self._expanded)
        self._active = False
        self._apply_style()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._apply_style()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._scroll.setVisible(expanded)
        self._apply_style()

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._scroll.setVisible(self._expanded)
        self._apply_style()

    def _apply_style(self) -> None:
        arrow = "▾" if self._expanded else "▸"
        # "&" is a mnemonic marker on a QPushButton: double a LONE "&" so a
        # title such as Step 4's "Measure & tidy" shows its ampersand, and
        # leave an already-escaped "&&" (Step 1's "Load && Adjust") alone.
        import re as _re
        title = _re.sub(r"(?<!&)&(?!&)", "&&", self._title)
        self._header.setText(f"  {arrow}   {title}")
        if self._active:
            # Use accent color tint for active header background.
            r = int(self._accent[1:3], 16)
            g = int(self._accent[3:5], 16)
            b = int(self._accent[5:7], 16)
            self._header.setStyleSheet(
                f"QPushButton {{"
                f"  background-color: rgba({r},{g},{b}, 0.12);"
                f"  border: 1px solid rgba({r},{g},{b}, 0.20);"
                f"  border-left: 3px solid {self._accent};"
                f"  border-radius: 4px;"
                f"  padding: 8px 12px 8px 8px;"
                f"  font-size: 12px; font-weight: bold;"
                f"  color: #eeeeee; text-align: left;"
                f"}}"
                f"QPushButton:hover {{"
                f"  background-color: rgba({r},{g},{b}, 0.18);"
                f"}}"
            )
            icon_color = "#eeeeee"
        else:
            self._header.setStyleSheet(
                "QPushButton {"
                "  background-color: rgba(255, 255, 255, 0.03);"
                "  border: 1px solid rgba(255, 255, 255, 0.08);"
                "  border-left: 3px solid rgba(255, 255, 255, 0.12);"
                "  border-radius: 4px;"
                "  padding: 8px 12px 8px 8px;"
                "  font-size: 12px; font-weight: bold;"
                "  color: #bbbbbb; text-align: left;"
                "}"
                "QPushButton:hover {"
                "  background-color: rgba(255, 255, 255, 0.06);"
                "  border-left: 3px solid rgba(255, 255, 255, 0.25);"
                "  color: #dddddd;"
                "}"
            )
            icon_color = "#888888"

        # Set icon inside the button via QIcon.
        if self._icon_name:
            qicon = _qta_icon_or_none(self._icon_name, icon_color)
            if qicon is not None:
                self._header.setIcon(qicon)


class Accordion(QWidget):
    """A styled accordion replacing QToolBox — immune to napari's global theme."""

    def __init__(self, accent: str = "#d4a843", parent: QWidget = None) -> None:
        super().__init__(parent)
        self._sections: list[AccordionSection] = []
        self._accent = accent
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(1)
        self._current = -1

    def addItem(self, widget: QWidget, title: str,
                icon_name: str = "") -> None:
        section = AccordionSection(title, widget, expanded=False,
                                   accent=self._accent, icon_name=icon_name,
                                   parent=self)
        section._header.clicked.connect(
            lambda _=False, idx=len(self._sections): self._on_click(idx)
        )
        self._sections.append(section)
        self._layout.addWidget(section)

    def setCurrentIndex(self, index: int) -> None:
        self._current = index
        for i, sec in enumerate(self._sections):
            sec.set_expanded(i == index)
            sec.set_active(i == index)
        self._update_stretch()

    def count(self) -> int:
        return len(self._sections)

    def _on_click(self, index: int) -> None:
        if self._current == index:
            self._sections[index]._toggle()
            if not self._sections[index]._expanded:
                self._current = -1
                self._sections[index].set_active(False)
        else:
            if 0 <= self._current < len(self._sections):
                self._sections[self._current].set_expanded(False)
                self._sections[self._current].set_active(False)
            self._current = index
            self._sections[index].set_expanded(True)
            self._sections[index].set_active(True)
        self._update_stretch()

    def _update_stretch(self) -> None:
        """Give all space to the expanded section, none to collapsed ones."""
        for i, sec in enumerate(self._sections):
            idx = self._layout.indexOf(sec)
            if idx >= 0:
                self._layout.setStretchFactor(sec, 1 if i == self._current else 0)
            # Collapsed sections should be as small as possible.
            if sec._expanded:
                sec.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            else:
                sec.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

# ── Flow overview group (top of each step) ──
FLOW_OVERVIEW_STYLE = """
QGroupBox {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    margin-top: 6px;
    padding: 10px 12px 8px 12px;
    font-size: 11px;
    font-weight: bold;
    color: #aaaaaa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0px 6px;
    color: #999999;
    font-size: 10px;
}
"""

# ── Compact flow hint (borderless alternative to FLOW_OVERVIEW_STYLE) ──
FLOW_HINT_STYLE = (
    "background: rgba(255, 255, 255, 0.015);"
    "border-left: 2px solid rgba(255, 255, 255, 0.08);"
    "border-radius: 2px;"
    "padding: 6px 10px;"
)

# ── Section separator ──
SEPARATOR_STYLE = "background-color: rgba(255, 255, 255, 0.08);"
SEPARATOR_LINE_STYLE = "background-color: #555555; margin: 4px 0px;"

# ── Dim / secondary labels ──
DIM_LABEL_STYLE = "color: #999999; font-size: 10px;"
STATUS_LABEL_STYLE = "color: #aaaaaa; font-size: 11px;"
SECTION_HEADER_STYLE = "font-weight: bold; font-size: 12px; color: #cccccc;"

# ── Flow description text ──
FLOW_TITLE_STYLE = "font-weight: 600; font-size: 11px; color: #cccccc;"
FLOW_SUBTITLE_STYLE = "font-size: 10px; color: #888888;"

# ── Log output (read-only QTextEdit) ──
LOG_OUTPUT_STYLE = (
    "background-color: #222222; color: #dddddd; padding: 6px;"
    f" {MONO_FONT}"
)

# ── Tab widget ──
TAB_WIDGET_STYLE = """
QTabWidget::pane {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 3px;
}
QTabBar::tab {
    padding: 5px 14px;
    font-size: 11px;
    color: #999999;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
    min-width: 60px;
}
QTabBar::tab:selected {
    color: #eeeeee;
    border-bottom: 2px solid #d4a843;
}
QTabBar::tab:hover:!selected {
    color: #cccccc;
    border-bottom: 2px solid rgba(255, 255, 255, 0.15);
}
"""

# ── QCollapsible styling (superqt) ──
COLLAPSIBLE_STYLE = """
QCollapsible > QPushButton {
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: bold;
    color: #bbbbbb;
    text-align: left;
}
QCollapsible > QPushButton:hover {
    background-color: rgba(255, 255, 255, 0.07);
    color: #dddddd;
}
"""

# ── Status readout (monospace value with subtle background) ──
STATUS_READOUT_STYLE = (
    "color: #aaaaaa; font-size: 11px;"
    " font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;"
    " background: rgba(255, 255, 255, 0.03);"
    " border-radius: 3px; padding: 3px 6px;"
)

# ── Canvas / dark preview area ──
CANVAS_AREA_STYLE = (
    "background-color: #111111; color: #bcbcbc; padding: 8px;"
    "border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 4px;"
)

# ── QCollapsible animation duration (ms) ──
COLLAPSIBLE_DURATION = 120

# ── Form input styling (QLineEdit, QComboBox, QSpinBox) ──
INPUT_STYLE = """
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 3px;
    padding: 4px 6px;
    color: #dddddd;
    font-size: 11px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid rgba(255, 255, 255, 0.25);
    background-color: rgba(255, 255, 255, 0.06);
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    color: #666666;
    background-color: rgba(255, 255, 255, 0.02);
}
QComboBox::drop-down {
    border: none;
    padding-right: 4px;
}
"""

# ── QListWidget styling (accent-colored selection) ──
def list_style(accent: str = "#d4a843") -> str:
    r, g, b = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
    return (
        "QListWidget {"
        "  background-color: rgba(255, 255, 255, 0.03);"
        "  border: 1px solid rgba(255, 255, 255, 0.08);"
        "  border-radius: 3px;"
        "  padding: 2px;"
        "  font-size: 11px;"
        "  color: #cccccc;"
        "}"
        "QListWidget::item {"
        "  padding: 3px 6px;"
        "  border-radius: 2px;"
        "}"
        "QListWidget::item:hover {"
        "  background-color: rgba(255, 255, 255, 0.06);"
        "}"
        "QListWidget::item:selected {"
        f"  background-color: rgba({r}, {g}, {b}, 0.25);"
        "  color: #eeeeee;"
        "}"
    )

LIST_STYLE = list_style("#d4a843")
LIST_STYLE_TEAL = list_style("#5ba4cf")

# ── Styled progress bar (accent-colored) ──
def progress_bar_style(color: str = "#d4a843", height: int = 14) -> str:
    return (
        f"QProgressBar {{"
        f"  background: rgba(255, 255, 255, 0.06);"
        f"  border: none;"
        f"  border-radius: 3px;"
        f"  text-align: center;"
        f"  font-size: 9px;"
        f"  color: #cccccc;"
        f"  min-height: {int(height)}px; max-height: {int(height)}px;"
        f"}}"
        f"QProgressBar::chunk {{"
        f"  background: {color};"
        f"  border-radius: 3px;"
        f"}}"
    )

PROGRESS_BAR_STYLE = progress_bar_style("#d4a843")       # amber (Build)
PROGRESS_BAR_STYLE_TEAL = progress_bar_style("#5ba4cf")  # teal (Setup)
PROGRESS_BAR_STYLE_GREEN = progress_bar_style("#6abf69") # green (Apply)

# ── Patch lifecycle status colors ──
# Maps the pipeline progression onto the 3-accent system:
#   gray (inactive) → teal (predicted) → amber (reconstructed)
#   → purple (curating) → green (accepted) → ice-blue (finalized)
#
# To customize: edit hex values here. All list widgets in Step 3
# (patch state, curation bucket, refinement list) read from this dict.
PATCH_STATUS_COLORS = {
    "unprocessed":   "#777777",   # dim gray — awaiting action
    "predicted":     "#5ba4cf",   # teal — analysis/info (Setup accent)
    "reconstructed": "#d4a843",   # amber — in-progress (Build accent)
    "curating":      "#a78bca",   # soft purple — active editing
    "accepted":      "#6abf69",   # green — approved (Apply accent)
    "finalized":     "#80deea",   # ice blue — diamond-hard, locked
}

# qtawesome icons per pipeline status
PATCH_STATUS_ICONS = {
    "unprocessed":   "mdi6.checkbox-blank-outline",
    "predicted":     "mdi6.check",
    "reconstructed": "mdi6.cube-outline",
    "curating":      "mdi6.pencil-outline",
    "accepted":      "mdi6.check-circle-outline",
    "finalized":     "mdi6.diamond-stone",
}

# Statuses that are protected from re-prediction/reconstruction
PROTECTED_STATUSES = frozenset({"accepted", "finalized"})

# Pipeline progress dots
PATCH_STATUS_DOTS = {
    "unprocessed":   "\u25cb",             # ○
    "predicted":     "\u25cf",             # ●
    "reconstructed": "\u25cf\u25cf",       # ●●
    "curating":      "\u25cf\u25cf",       # ●●
    "accepted":      "\u25cf\u25cf\u25cf", # ●●●
    "finalized":     "\u25cf\u25cf\u25cf", # ●●●
}

# ── Stack lifecycle status colors (Step 4: 3D Stack Segmentation) ──
# Where Step 3 and Step 4
# mean the same thing they share Step 3's word, colour and dots (built as
# COPIES of PATCH_STATUS_*, which are not changed); Step 4's own statuses keep
# Step 4's.  A legacy stored ``segmented`` is inferred from its record's keys
# when the state is read (measured with a features CSV,
# reconstructed with a labels path, else failed); ``smoothed`` reads as
# measured; STACK_STATUS_WORDS carries the word on screen.
STACK_STATUS_COLORS = {
    "unprocessed":   PATCH_STATUS_COLORS["unprocessed"],
    "predicted":     PATCH_STATUS_COLORS["predicted"],
    "reconstructed": PATCH_STATUS_COLORS["reconstructed"],
    "qc_extracted":  "#a78bca",   # soft purple -- "measured" (Step 4's own)
    "finalized":     PATCH_STATUS_COLORS["finalized"],
    "failed":        "#e07070",   # red -- Predict / Reconstruct failed
    "segmented":     "#e07070",   # legacy: a Reconstruct that never finished
    "smoothed":      "#a78bca",   # legacy: never written; reads as measured
}

STACK_STATUS_ICONS = {
    "unprocessed":   "mdi6.circle-outline",
    "predicted":     PATCH_STATUS_ICONS["predicted"],
    "reconstructed": PATCH_STATUS_ICONS["reconstructed"],
    "qc_extracted":  "mdi6.chart-scatter-plot",
    "finalized":     PATCH_STATUS_ICONS["finalized"],
    "failed":        "mdi6.close-circle-outline",
    "segmented":     "mdi6.close-circle-outline",
    "smoothed":      "mdi6.chart-scatter-plot",
}

STACK_STATUS_DOTS = {
    "unprocessed":   PATCH_STATUS_DOTS["unprocessed"],
    "predicted":     PATCH_STATUS_DOTS["predicted"],
    "reconstructed": PATCH_STATUS_DOTS["reconstructed"],
    "qc_extracted":  "●●",       # ●●
    "finalized":     PATCH_STATUS_DOTS["finalized"],
    "failed":        "",                   # none: nothing reached
    "segmented":     "",
    "smoothed":      "●●",
}

#: The word on screen for each stored status.
STACK_STATUS_WORDS = {
    "unprocessed":   "unprocessed",
    "predicted":     "predicted",
    "reconstructed": "reconstructed",
    "qc_extracted":  "measured",
    "finalized":     "finalized",
    "failed":        "failed",
    "segmented":     "failed",
    "smoothed":      "measured",
}

# ── Splitter handle ──
SPLITTER_STYLE = """
QSplitter::handle {
    background: rgba(255, 255, 255, 0.06);
    margin: 2px 0px;
}
QSplitter::handle:hover {
    background: rgba(255, 255, 255, 0.15);
}
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical { height: 3px; }
"""

# ── Scrollbar (thin, consistent) ──
SCROLLBAR_STYLE = """
QScrollBar:vertical {
    background: transparent; width: 6px;
}
QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 3px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

# ── Inner QGroupBox styling (inside accordion panels) ──
INNER_GROUP_STYLE = """
QGroupBox {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    margin-top: 14px;
    padding: 8px 8px 6px 8px;
    font-size: 11px;
    font-weight: bold;
    color: #aaaaaa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0px 6px;
    color: #999999;
    font-size: 10px;
}
"""

# Dialog-level input + group-box theming for the Step 2 themed dialogs
# (New Run, Manage Runs, Import Pairs, Import External).
#
# DISTINCT FROM ``INNER_GROUP_STYLE`` above — that one is used by
# ``apply_inner_group_style`` across all 4 step main widgets and is a
# LOAD-BEARING shared style; do not touch it without auditing Steps 1-4.
# This dialog-specific style:
#   * uses amber title color (matches Step 3 qc_extraction inspiration)
#   * 11px title font (matches content size, was 10px in INNER_GROUP_STYLE)
#   * no background-color fill on QGroupBox (kills the gray title pad)
#   * themes QLineEdit / QTextEdit / QSpinBox / QComboBox / radios with
#     coherent dark inputs instead of Qt's default
#   * forces transparent backgrounds on inert wrapper QWidgets
#     (QStackedWidget pages, radio-button containers) which Qt otherwise
#     paints with a gray default that clashes with the dialog surface
#
# Apply with ``dialog.setStyleSheet(DIALOG_FORM_STYLE)`` at the dialog
# level.  Inherits down to all descendants.  Set 2026-05-26 after user
# reported the gray title pad + radio-container clash in screenshots.
DIALOG_FORM_STYLE = """
QDialog {
    background-color: #2b2b2b;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
}
/* For inert wrapper QWidgets (radio button containers, stacked
   page widgets) that pick up Qt's default gray, call
   ``wrapper.setStyleSheet("background: transparent;")`` directly
   in the dialog code at construction time.  Per-widget direct
   styling is more discoverable than a global property selector. */
QGroupBox {
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 4px;
    margin-top: 10px;
    padding: 12px 8px 8px 8px;
    font-size: 11px;
    font-weight: bold;
    color: #cccccc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 6px;
    color: #d4a843;
    font-size: 11px;
    font-weight: bold;
}
QLineEdit, QTextEdit {
    background-color: rgba(0, 0, 0, 0.30);
    color: #eeeeee;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 3px;
    padding: 4px 6px;
    font-size: 11px;
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid rgba(212, 168, 67, 0.55);
}
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: rgba(0, 0, 0, 0.30);
    color: #eeeeee;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 3px;
    padding: 3px 6px;
    font-size: 11px;
    min-height: 18px;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid rgba(212, 168, 67, 0.55);
}
QComboBox QAbstractItemView {
    background: #2a2a2a;
    color: #eeeeee;
    selection-background-color: rgba(212, 168, 67, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.18);
}
QRadioButton, QCheckBox {
    color: #cccccc;
    font-size: 11px;
    background: transparent;
    spacing: 6px;
}
QRadioButton:disabled, QCheckBox:disabled {
    color: #666666;
}
QLabel {
    background: transparent;
}
QStackedWidget, QStackedWidget > QWidget {
    background: transparent;
}
"""

# ── Primary action button (Predict All, Train, Reconstruct, etc.) ──

def _rgba(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def primary_btn_style(accent: str = "#d4a843") -> str:
    """Generate a primary button stylesheet from an accent color."""
    return (
        f"QPushButton {{"
        f"  background-color: {_rgba(accent, 0.25)};"
        f"  border: 1px solid {_rgba(accent, 0.4)};"
        f"  border-radius: 4px; padding: 7px 16px;"
        f"  font-size: 11px; font-weight: bold; color: #eeeeee;"
        f"}}"
        f"QPushButton:hover {{"
        f"  background-color: {_rgba(accent, 0.35)};"
        f"  border: 1px solid {_rgba(accent, 0.55)};"
        f"}}"
        f"QPushButton:pressed {{"
        f"  background-color: {_rgba(accent, 0.45)};"
        f"}}"
        f"QPushButton:disabled {{"
        f"  background-color: {_rgba(accent, 0.08)};"
        f"  border: 1px solid {_rgba(accent, 0.15)};"
        f"  color: #777777;"
        f"}}"
    )


# Pre-built for convenience.
PRIMARY_BTN_STYLE = primary_btn_style("#d4a843")  # amber (Build)
PRIMARY_BTN_STYLE_TEAL = primary_btn_style("#5ba4cf")  # teal (Setup)
PRIMARY_BTN_STYLE_GREEN = primary_btn_style("#6abf69")  # green (Apply)
PRIMARY_BTN_STYLE_CYAN = primary_btn_style("#4ecdc4")  # cyan (Refine Further — distinct from teal)

# ── Secondary action button (Pause, Browse, etc.) ──
SECONDARY_BTN_STYLE = """
QPushButton {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 4px;
    padding: 5px 12px;
    font-size: 10px;
    color: #cccccc;
}
QPushButton:hover {
    background-color: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.18);
}
QPushButton:pressed {
    background-color: rgba(255, 255, 255, 0.15);
}
QPushButton:disabled {
    background-color: rgba(255, 255, 255, 0.02);
    color: #666666;
}
"""

# ── Destructive action button (Delete Label, Discard All Garbage, etc.) ──

def destructive_btn_style(accent: str = "#c62828") -> str:
    """Themed destructive button — keeps red as the danger signal but
    matches the rest of the EpiCycle theme (translucent fill + border).
    Pass a darker red (e.g. ``#8b0000``) for higher-stakes actions."""
    return (
        f"QPushButton {{"
        f"  background-color: {_rgba(accent, 0.22)};"
        f"  border: 1px solid {_rgba(accent, 0.40)};"
        f"  border-radius: 4px; padding: 6px 14px;"
        f"  font-size: 11px; font-weight: bold; color: #f0d4d4;"
        f"}}"
        f"QPushButton:hover {{"
        f"  background-color: {_rgba(accent, 0.35)};"
        f"  border: 1px solid {_rgba(accent, 0.55)};"
        f"  color: #ffffff;"
        f"}}"
        f"QPushButton:pressed {{"
        f"  background-color: {_rgba(accent, 0.50)};"
        f"}}"
        f"QPushButton:disabled {{"
        f"  background-color: {_rgba(accent, 0.06)};"
        f"  border: 1px solid {_rgba(accent, 0.12)};"
        f"  color: #777777;"
        f"}}"
    )


DESTRUCTIVE_BTN_STYLE = destructive_btn_style("#c62828")        # red (Delete)
DESTRUCTIVE_BTN_STYLE_DARK = destructive_btn_style("#8b0000")   # dark red (Discard All)


# ── Toggle button (checkable) — accent fill when ON, neutral when OFF ──

def toggle_btn_style(accent: str = "#d4a843") -> str:
    """Checkable QPushButton style — neutral surface when unchecked,
    accent-tinted when checked.  Use for binary view toggles like
    "Dim Others" or "Snap to Grid" where the visual state should
    self-describe ON/OFF without text suffixes."""
    return (
        f"QPushButton {{"
        f"  background-color: rgba(255, 255, 255, 0.04);"
        f"  border: 1px solid rgba(255, 255, 255, 0.10);"
        f"  border-radius: 4px;"
        f"  padding: 5px 12px;"
        f"  font-size: 11px;"
        f"  color: #cccccc;"
        f"}}"
        f"QPushButton:hover {{"
        f"  background-color: rgba(255, 255, 255, 0.08);"
        f"  border: 1px solid rgba(255, 255, 255, 0.18);"
        f"  color: #eeeeee;"
        f"}}"
        f"QPushButton:checked {{"
        f"  background-color: {_rgba(accent, 0.22)};"
        f"  border: 1px solid {_rgba(accent, 0.45)};"
        f"  color: #eeeeee;"
        f"  font-weight: bold;"
        f"}}"
        f"QPushButton:checked:hover {{"
        f"  background-color: {_rgba(accent, 0.32)};"
        f"  border: 1px solid {_rgba(accent, 0.60)};"
        f"}}"
        f"QPushButton:disabled {{"
        f"  background-color: rgba(255, 255, 255, 0.02);"
        f"  color: #666666;"
        f"}}"
    )


TOGGLE_BTN_STYLE = toggle_btn_style("#d4a843")  # amber (Build/Step 3)


# ---------------------------------------------------------------------------
# ── Curation tools panel (the Step 3 dock) ──
#
# The primitives of the approved Brick 2 panel design.
# Its ``<style>`` block IS the design; every literal below is copied from a
# rule in it, so a reviewer can diff the two by eye:
#
#   .sec    -> PANEL_CARD_STYLE        + make_panel_card()
#   .sec h4 -> PANEL_CARD_HEADER_STYLE + panel_card_header()
#   .row    -> panel_form_label()      (one FIXED label column per section)
#   .combo  -> PANEL_INPUT_STYLE
#   .chev   -> PANEL_CHEVRON_STYLE
#   .chip   -> PANEL_CHIP_STYLE
#   .piece  -> PANEL_LIST_STYLE        + PanelRowDelegate (mono trailing value)
#   .help   -> PANEL_HELP_STYLE
#   .warn   -> PANEL_WARN_STYLE
#   .status -> PANEL_STATUS_PILL_STYLE
#   .btn    -> PANEL_BTN_STYLE   /  .btn.pri -> PANEL_PRIMARY_BTN_STYLE
#   .foot   -> PANEL_FOOTER_STYLE
#
# They are here rather than in the panel so Brick 2b/3 (Refine, Add) can
# reuse them without importing from the widget.
# ---------------------------------------------------------------------------

#: The mockup's palette, by name, so a caller never re-types a literal.
PANEL_CARD_BG = "rgba(255, 255, 255, 0.02)"
PANEL_CARD_BORDER = "rgba(255, 255, 255, 0.06)"
PANEL_FIELD_BG = "#414851"
PANEL_GOLD = "#d4a843"
PANEL_GOLD_HOVER = "#e6c26a"
PANEL_HELP_COLOR = "#888888"
PANEL_ROW_LABEL_COLOR = "#bbbbbb"

#: A card is a plain ``QWidget`` with this object name -- NOT a QFrame:
#: several of the containers it is applied to are pinned to ``QWidget`` by
#: the panel's contract test, and a plain QWidget paints a stylesheet
#: background only with ``WA_StyledBackground`` (which ``make_panel_card``
#: sets).  The ID selector keeps the border off every child.
PANEL_CARD_OBJECT = "epiPanelCard"
PANEL_CARD_STYLE = (
    "QWidget#epiPanelCard {"
    f"  background: {PANEL_CARD_BG};"
    f"  border: 1px solid {PANEL_CARD_BORDER};"
    "  border-radius: 4px;"
    "}"
)

#: `.sec h4` -- small, grey, UPPERCASE, letter-spaced.  Qt stylesheets have
#: no ``letter-spacing``, so ``panel_card_header`` puts that on the QFont.
PANEL_CARD_HEADER_STYLE = (
    "color: #999999; font-size: 10px; font-weight: bold;"
    " background: transparent;"
)
PANEL_CARD_HEADER_SPACING = 0.6

#: `.lbl` -- the fixed label column of a form row.
PANEL_ROW_LABEL_STYLE = (
    f"color: {PANEL_ROW_LABEL_COLOR}; font-size: 11px;"
    " background: transparent;"
)

#: The two column widths the panel uses.  S3's rows are short ("Markers
#: from"); the Advanced knobs are not ("Search padding (voxels)").
PANEL_LABEL_COLUMN = 82
PANEL_ADV_LABEL_COLUMN = 134

#: `.help` -- the grey one-liner under a control.
PANEL_HELP_STYLE = (
    f"color: {PANEL_HELP_COLOR}; font-size: 10px; background: transparent;"
)

#: `.warn` -- the amber block on a result.
PANEL_WARN_STYLE = (
    "background: rgba(212, 168, 67, 0.10);"
    " border: 1px solid rgba(212, 168, 67, 0.35);"
    " border-radius: 3px; padding: 5px 7px;"
    " color: #e6c26a; font-size: 10px;"
)

#: `.warn.bad` -- the RED block: the tool did not do what was asked.
#: Same shape as the amber one so the two read as one family; the colour
#: is the whole difference, because that is the whole distinction.  Amber
#: is a warning about a result you HAVE; red is the absence of one.
PANEL_ERROR_STYLE = (
    "background: rgba(224, 87, 79, 0.10);"
    " border: 1px solid rgba(224, 87, 79, 0.40);"
    " border-radius: 3px; padding: 5px 7px;"
    " color: #e0776f; font-size: 10px;"
)

#: `.status` -- the state pill at the top of every page.
PANEL_STATUS_PILL_OBJECT = "epiStatusPill"
PANEL_STATUS_PILL_STYLE = (
    "QWidget#epiStatusPill {"
    "  background: rgba(255, 255, 255, 0.03);"
    "  border: 1px solid rgba(255, 255, 255, 0.08);"
    "  border-radius: 4px;"
    "}"
)

#: `.foot` -- the fixed two-button footer.
PANEL_FOOTER_OBJECT = "epiPanelFooter"
PANEL_FOOTER_STYLE = (
    "QWidget#epiPanelFooter {"
    "  background: rgba(255, 255, 255, 0.02);"
    "  border-top: 1px solid rgba(255, 255, 255, 0.08);"
    "}"
)

#: `.combo` / `.num` -- a solid field, a small arrow, slim spin buttons.
#: The spin buttons are NARROWED, never removed: hiding them would take
#: away a way of changing the value, which is behaviour, not style.
PANEL_INPUT_STYLE = (
    "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {"
    f"  background-color: {PANEL_FIELD_BG};"
    "  border: 1px solid rgba(255, 255, 255, 0.10);"
    "  border-radius: 3px;"
    "  padding: 2px 6px;"
    "  color: #eeeeee;"
    "  font-size: 11px;"
    "  min-height: 16px;"
    "}"
    "QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {"
    "  border: 1px solid rgba(212, 168, 67, 0.55);"
    "  background-color: #484f59;"
    "}"
    "QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,"
    " QDoubleSpinBox:disabled {"
    "  color: #777777;"
    "  background-color: rgba(255, 255, 255, 0.03);"
    "}"
    "QComboBox::drop-down {"
    "  border: none; width: 14px; padding-right: 2px;"
    "}"
    "QComboBox::down-arrow { width: 9px; height: 9px; }"
    "QComboBox QAbstractItemView {"
    "  background: #2a2d33; color: #eeeeee;"
    "  border: 1px solid rgba(255, 255, 255, 0.18);"
    "  selection-background-color: rgba(212, 168, 67, 0.35);"
    "}"
    "QSpinBox::up-button, QSpinBox::down-button,"
    " QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {"
    "  width: 13px; border: none;"
    "  background: rgba(255, 255, 255, 0.06);"
    "}"
    "QSpinBox::up-button:hover, QSpinBox::down-button:hover,"
    " QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {"
    "  background: rgba(255, 255, 255, 0.12);"
    "}"
)

#: The tick INSIDE a ticked box.  The moment a stylesheet touches
#: ``::indicator`` Qt stops drawing its own glyph, so a rule that only
#: fills the box gold renders a solid gold square and the user cannot
#: tell ticked from "highlighted".  It has to be given an image back --
#: and it has to be a FILE: a `data:` URI in a Qt stylesheet loads
#: nothing at all (measured: zero dark pixels inside the box).
#:
#: NOTE FOR PACKAGING: a wheel/sdist must ship this .svg as package data
#: (``napari_epicycle.ui_shell`` -> ``*.svg``), or an installed copy gets
#: the gold square back.
_PANEL_TICK_SVG = _Path(__file__).with_name("_panel_tick.svg").as_posix()

#: `.cb` -- an 11 px box that fills gold, with a dark tick in it, when it
#: is on.  Set on the panel ROOT so it reaches every tick without a
#: per-widget call.
PANEL_CHECKBOX_STYLE = (
    "QCheckBox { color: #dddddd; font-size: 11px; spacing: 6px; }"
    "QCheckBox:disabled { color: #777777; }"
    "QCheckBox::indicator, QListWidget::indicator {"
    "  width: 11px; height: 11px;"
    "  border: 1px solid #999999;"
    "  border-radius: 2px;"
    "  background: rgba(0, 0, 0, 0.25);"
    "}"
    "QCheckBox::indicator:checked, QListWidget::indicator:checked {"
    f"  background: {PANEL_GOLD}; border: 1px solid {PANEL_GOLD};"
    f'  image: url("{_PANEL_TICK_SVG}");'
    "}"
    "QCheckBox::indicator:disabled, QListWidget::indicator:disabled {"
    "  border: 1px solid #666666; background: rgba(0, 0, 0, 0.15);"
    "}"
    # A DISABLED box must still say whether it is ticked: without this the
    # rule above painted over the checked gold, and the dark tick glyph on
    # the dark background drew a checked box exactly as an empty one
    # (Add live-shot fix F7: "Treat dark space as outside", greyed on a
    # patch with no voxel size, read as unticked).  A dimmed gold.
    "QCheckBox::indicator:checked:disabled,"
    " QListWidget::indicator:checked:disabled {"
    "  background: rgba(212, 168, 67, 0.55);"
    "  border: 1px solid rgba(212, 168, 67, 0.55);"
    f'  image: url("{_PANEL_TICK_SVG}");'
    "}"
)

#: `.rb` -- Step 3 Model Refinement's radio promoted to the theme (Step 2
#: look, brick L; ``fusion_reconstruction.ui.MR_RADIO_STYLE`` stays as it is):
#: an 11 px round indicator with a 1 px #999999 border and a gold dot when
#: it is on.  Set on a panel ROOT beside ``PANEL_CHECKBOX_STYLE``.
PANEL_RADIO_STYLE = (
    "QRadioButton { color: #dddddd; font-size: 11px; spacing: 6px; }"
    "QRadioButton:disabled { color: #777777; }"
    # The size, the radius and the padding ride EVERY state rule: napari's
    # own ``::indicator::checked`` rule (6 px + 5 px padding + an svg) is
    # more specific than a state-less one, so a width given only on the base
    # rule loses to it and the checked dot drew as a big gold square (seen
    # live, Step 2 look brick L).
    "QRadioButton::indicator, QRadioButton::indicator:unchecked,"
    " QRadioButton::indicator:checked, QRadioButton::indicator:disabled,"
    " QRadioButton::indicator:hover {"
    "width:11px;height:11px;border-radius:6px;padding:0px;}"
    "QRadioButton::indicator, QRadioButton::indicator:unchecked {"
    "border:1px solid #999999;background:rgba(0,0,0,0.25);}"
    # napari's checked rule adds ``image: circle.svg; padding: 5px`` of its
    # own: ``image: none`` so the gold dot below is the only mark
    "QRadioButton::indicator:checked{border:1px solid #d4a843;image:none;"
    "background:qradialgradient(cx:0.5,cy:0.5,radius:0.5,fx:0.5,fy:0.5,"
    "stop:0 #1d1e24,stop:0.35 #1d1e24,stop:0.45 #d4a843,stop:1 #d4a843)}"
    "QRadioButton::indicator:disabled{border:1px solid #666666;}"
)

#: The drag handle under a ticklist.  Same surface as a chip, because it
#: is the same weight of control: something small you may touch, that
#: says nothing until you do.  The dots are painted, not a glyph -- a
#: 6 px row has no room for a font's own line spacing.
PANEL_GRIP_STYLE = (
    "QFrame {"
    "  background-color: rgba(255, 255, 255, 0.06);"
    "  border: 1px solid transparent;"
    "  border-radius: 3px;"
    "}"
    "QFrame:hover {"
    "  background-color: rgba(212, 168, 67, 0.25);"
    "  border: 1px solid rgba(212, 168, 67, 0.40);"
    "}"
)

#: How tall that handle is.  It is taken out of the list's own cap, not
#: added to the page: the markers-found state has 7 px of headroom
#: against its two-screen ceiling.
PANEL_GRIP_H = 6

#: `.chev` -- a disclosure row: left-aligned, bold, quiet.
PANEL_CHEVRON_STYLE = (
    "QPushButton {"
    "  background-color: rgba(255, 255, 255, 0.04);"
    "  border: 1px solid rgba(255, 255, 255, 0.08);"
    "  border-radius: 3px;"
    "  padding: 3px 9px;"
    "  font-size: 10px; font-weight: bold;"
    "  color: #bbbbbb;"
    "  text-align: left;"
    "}"
    "QPushButton:hover {"
    "  background-color: rgba(255, 255, 255, 0.07); color: #dddddd;"
    "}"
    "QPushButton:checked {"
    "  background-color: rgba(255, 255, 255, 0.06); color: #dddddd;"
    "}"
    "QPushButton:disabled { color: #666666; }"
)

#: `.chip` -- a small flat toggle / quick action.
PANEL_CHIP_STYLE = (
    "QPushButton {"
    "  background-color: rgba(255, 255, 255, 0.06);"
    "  border: 1px solid transparent;"
    "  border-radius: 3px;"
    "  padding: 2px 8px;"
    "  font-size: 10px;"
    "  color: #cccccc;"
    "  text-align: left;"
    "}"
    "QPushButton:hover { background-color: rgba(255, 255, 255, 0.10); }"
    "QPushButton:checked {"
    "  background-color: rgba(212, 168, 67, 0.20);"
    "  border: 1px solid rgba(212, 168, 67, 0.40);"
    "  color: #eeeeee;"
    "}"
    "QPushButton:disabled { color: #666666; }"
)

#: `.btn` -- the panel's neutral button, on the mockup's solid field grey.
PANEL_BTN_STYLE = (
    "QPushButton {"
    f"  background-color: {PANEL_FIELD_BG};"
    "  border: 1px solid rgba(255, 255, 255, 0.10);"
    "  border-radius: 4px;"
    # 10 px, not the mockup's 12: three of these share one row on the
    # painted-marker block, and the page is CLIPPED, never scrolled,
    # past the viewport's 316 px.
    "  padding: 4px 10px;"
    "  font-size: 11px; font-weight: bold;"
    "  color: #eeeeee;"
    "}"
    "QPushButton:hover { background-color: #4b535e; }"
    "QPushButton:pressed { background-color: #3a4149; }"
    "QPushButton:disabled {"
    "  background-color: rgba(255, 255, 255, 0.03);"
    "  border: 1px solid rgba(255, 255, 255, 0.05);"
    "  color: #777777;"
    "}"
)


def panel_primary_btn_style(accent: str = PANEL_GOLD,
                            hover: str = PANEL_GOLD_HOVER) -> str:
    """`.btn.pri` -- the SOLID accent button of the mockup.

    Distinct from :func:`primary_btn_style`, which tints a translucent
    surface; on this panel the one committing action is the only fully
    saturated thing on screen, which is what makes it findable at 340 px.
    """
    return (
        "QPushButton {"
        f"  background-color: {accent};"
        f"  border: 1px solid {accent};"
        "  border-radius: 4px;"
        "  padding: 5px 12px;"
        "  font-size: 11px; font-weight: bold;"
        "  color: #1e1e1e;"
        "}"
        "QPushButton:hover {"
        f"  background-color: {hover}; border: 1px solid {hover};"
        "}"
        f"QPushButton:pressed {{ background-color: {_rgba(accent, 0.80)}; }}"
        "QPushButton:disabled {"
        f"  background-color: {_rgba(accent, 0.12)};"
        f"  border: 1px solid {_rgba(accent, 0.25)};"
        "  color: #777777;"
        "}"
    )


PANEL_PRIMARY_BTN_STYLE = panel_primary_btn_style()


# ---------------------------------------------------------------------------
# Step 2 look, polish pass (2026-10-12): the literals Step 2's hand-written
# sheets used to spell out, named once.  ADDITIVE -- nothing above changes.
# ---------------------------------------------------------------------------

#: the "ok" green (GPU line, accepted) and the warn amber text -- the same
#: two values the rest of the panel already uses
PANEL_OK_COLOR = "#6abf69"
PANEL_WARN_COLOR = PANEL_GOLD_HOVER
#: a zero count / a disabled word
PANEL_ZERO_COLOR = "#666666"
#: a bright value (the active column of a table) and the red of "Discard"
PANEL_BRIGHT_COLOR = "#eeeeee"
PANEL_ERROR_TEXT_COLOR = "#e0776f"
#: the checked pair of ``PANEL_CHIP_STYLE`` (amber tint + edge), reused by
#: the pill chips and the View switch so there is ONE checked amber
PANEL_CHIP_CHECKED_BG = "rgba(212, 168, 67, 0.20)"
PANEL_CHIP_CHECKED_BORDER = "rgba(212, 168, 67, 0.40)"
#: a section foot's top rule (Step 3's ``PR_FOOT_STYLE`` / ``MR_FOOT_STYLE``)
PANEL_FOOT_BORDER = "#555555"
#: the one progress-bar height of the panel, and the short-dock one
PANEL_PROGRESS_H = 14
PANEL_PROGRESS_H_SHORT = 8
#: a column of a two-column step panel: the splitter handle and the inner
#: margins (Step 3's ``PR_SPLIT_HANDLE`` / ``PR_LEFT_INNER_MARGIN`` /
#: ``PR_RIGHT_INNER_MARGIN``)
PANEL_SPLIT_HANDLE = 6
PANEL_LEFT_INNER_MARGIN = 6
PANEL_RIGHT_INNER_MARGIN = 8


def disclosure_btn_style(object_name: str) -> str:
    """``COLLAPSIBLE_STYLE``'s header-button rules (box, 6 10 padding, 11 px
    bold ``#bbbbbb``) for a plain ``QPushButton`` of *object_name* -- the Loss
    curves header, which is not a superqt ``QCollapsible``."""
    return COLLAPSIBLE_STYLE.replace(
        "QCollapsible > QPushButton", f"QPushButton#{object_name}")


def panel_foot_style(object_name: str) -> str:
    """``QWidget#<name> { border-top: 1px solid #555555; }`` -- the rule
    Step 3's pinned feet carry above their buttons."""
    return (f"QWidget#{object_name} "
            f"{{ border-top: 1px solid {PANEL_FOOT_BORDER}; }}")


#: a table's column head (10 px bold ``#bbbbbb``)
PANEL_COLHEAD_STYLE = (
    f"color: {PANEL_ROW_LABEL_COLOR}; font-size: 10px; font-weight: bold;"
    " background: transparent;"
)
#: a value in a card (11 px ``#dddddd``) -- ``MR_CARD_VALUE_STYLE``
PANEL_VALUE_STYLE = (
    "color: #dddddd; font-size: 11px; background: transparent;"
)
#: a mono caption under a status line (the engine-name grey, 10 px)
PANEL_SLICES_HEADER_STYLE = (
    "color: #999999; font-size: 10px; background: transparent;"
    f" {MONO_FONT}"
)

#: ``SECONDARY_BTN_STYLE`` for a ``QToolButton`` (the Manage menu button): the
#: same fill, border, text and states, no menu arrow
SECONDARY_TOOLBTN_STYLE = (
    SECONDARY_BTN_STYLE.replace("QPushButton", "QToolButton")
    + "QToolButton::menu-indicator { image: none; }"
)

#: the View switch's two segments: ``PANEL_CHIP_STYLE``'s fills on a
#: ``QToolButton``, the checked one amber
PANEL_SEGMENT_STYLE = (
    "QToolButton {"
    "  background-color: rgba(255, 255, 255, 0.06);"
    "  border: 1px solid rgba(255, 255, 255, 0.10);"
    "  padding: 4px 10px; font-size: 11px; font-weight: bold;"
    "  color: #cccccc;"
    "}"
    "QToolButton:hover { background-color: rgba(255, 255, 255, 0.10); }"
    "QToolButton:checked {"
    f"  background-color: {PANEL_CHIP_CHECKED_BG};"
    f"  border: 1px solid {PANEL_CHIP_CHECKED_BORDER};"
    "  color: #eeeeee;"
    "}"
    f"QToolButton:disabled {{ color: {PANEL_ZERO_COLOR};"
    "  background-color: rgba(255, 255, 255, 0.02); }"
)

#: the dropdown list of a combo: the dark surface and the amber selection --
#: ``::item`` rules are needed as well, or the item's own (grey) hover wins
PANEL_COMBO_POPUP_STYLE = (
    "QComboBox QAbstractItemView {"
    "  background: #2a2d33; color: #eeeeee;"
    "  border: 1px solid rgba(255, 255, 255, 0.18);"
    "  selection-background-color: rgba(212, 168, 67, 0.35);"
    "  selection-color: #eeeeee; outline: 0;"
    "}"
    "QComboBox QAbstractItemView::item { min-height: 20px; padding: 2px 6px; }"
    "QComboBox QAbstractItemView::item:selected,"
    " QComboBox QAbstractItemView::item:hover {"
    "  background: rgba(212, 168, 67, 0.35); color: #eeeeee;"
    "}"
)


def panel_list_style(accent: str = PANEL_GOLD) -> str:
    """`.piece` -- one tinted row per item, 3 px apart, no frame noise."""
    r, g, b = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
    return (
        "QListWidget {"
        "  background-color: transparent;"
        "  border: none;"
        "  padding: 0px;"
        "  font-size: 11px;"
        "  color: #dddddd;"
        "  outline: none;"
        "}"
        "QListWidget::item {"
        "  background-color: rgba(255, 255, 255, 0.03);"
        "  border-radius: 3px;"
        "  padding: 2px 5px;"
        "  margin-bottom: 2px;"
        "}"
        "QListWidget::item:hover { background-color: rgba(255, 255, 255, 0.07); }"
        "QListWidget::item:selected {"
        f"  background-color: rgba({r}, {g}, {b}, 0.25);"
        "  color: #eeeeee;"
        "}"
    ) + PANEL_CHECKBOX_STYLE


PANEL_LIST_STYLE = panel_list_style()


def make_panel_card(container: QWidget, *,
                    margins=(9, 6, 9, 6), spacing: int = 3) -> QWidget:
    """Turn *container* into a `.sec` card, in place.

    In place, and not a new QFrame, because the panel's contract test pins
    several of these containers to ``QWidget`` BY TYPE NAME -- a card that
    changed their class would pass every visual review and fail the one
    test that protects the consumers reaching them.

    Returns the same widget so it can be used inline.
    """
    try:
        container.setObjectName(PANEL_CARD_OBJECT)
        container.setAttribute(Qt.WA_StyledBackground, True)
        container.setStyleSheet(PANEL_CARD_STYLE)
        lay = container.layout()
        if lay is not None:
            lay.setContentsMargins(*margins)
            lay.setSpacing(spacing)
    except Exception:
        pass
    return container


def panel_card_header(text: str, parent: QWidget = None) -> QLabel:
    """`.sec h4` -- the card's UPPERCASE, letter-spaced title."""
    from qtpy.QtGui import QFont

    label = QLabel(str(text).upper(), parent)
    label.setStyleSheet(PANEL_CARD_HEADER_STYLE)
    try:
        font = label.font()
        font.setLetterSpacing(QFont.AbsoluteSpacing, PANEL_CARD_HEADER_SPACING)
        font.setBold(True)
        label.setFont(font)
    except Exception:
        pass
    return label


def panel_form_label(text: str, width: int = PANEL_LABEL_COLUMN,
                     parent: QWidget = None) -> QLabel:
    """`.lbl` -- one cell of the FIXED label column of a section.

    Fixed, not preferred: a column that is as wide as each row's own words
    puts every field at a different left edge, which is the single loudest
    difference between a themed panel and a pile of rows.
    """
    label = QLabel(str(text), parent)
    label.setStyleSheet(PANEL_ROW_LABEL_STYLE)
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    label.setFixedWidth(int(width))
    label.setWordWrap(True)
    return label


class PanelRowDelegate(QStyledItemDelegate):
    """Draws a list row's trailing "  <sep>  value" in mono grey.

    The mockup's `.piece` row is a checkbox, a colour swatch, a name that
    takes the slack, and a monospaced count hard against the right edge.
    Qt gives us the first three for free; only the split between the name
    and the value needs painting, so this delegate does exactly that and
    leaves hit-testing, check-state editing and selection to the base --
    the item TEXT is untouched, which keeps it the thing the tests read.
    """

    def __init__(self, parent=None, separator: str = "  ·  ",
                 color: str = "#aaaaaa") -> None:
        super().__init__(parent)
        self._separator = str(separator)
        self._color = str(color)

    def _mono(self, base):
        from qtpy.QtGui import QFont

        font = QFont(base)
        font.setFamily("Consolas")
        return font

    def paint(self, painter, option, index):  # noqa: N802 - Qt naming
        from qtpy.QtCore import QRect
        from qtpy.QtGui import QColor, QFontMetrics
        from qtpy.QtWidgets import (
            QApplication, QStyle, QStyleOptionViewItem,
        )

        try:
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            head, sep, tail = str(opt.text or "").rpartition(self._separator)
            if not sep or not tail:
                super().paint(painter, option, index)
                return
            widget = opt.widget
            style = widget.style() if widget is not None else QApplication.style()
            mono = self._mono(opt.font)
            tail_w = QFontMetrics(mono).horizontalAdvance(tail) + 10
            # 1) the row itself (background, check, swatch) at FULL width,
            #    so the tint and the selection reach the right edge ...
            base = QStyleOptionViewItem(opt)
            base.text = ""
            style.drawControl(QStyle.CE_ItemViewItem, base, painter, widget)
            # 2) ... then the name, in a rect that stops short of the value
            #    so Qt elides the NAME rather than overwriting the number.
            opt.text = head
            opt.rect = QRect(
                opt.rect.left(), opt.rect.top(),
                max(24, opt.rect.width() - tail_w), opt.rect.height(),
            )
            style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)
            painter.save()
            painter.setFont(mono)
            painter.setPen(QColor(self._color))
            painter.drawText(
                QRect(option.rect.right() - tail_w + 2, option.rect.top(),
                      tail_w - 6, option.rect.height()),
                Qt.AlignRight | Qt.AlignVCenter, tail,
            )
            painter.restore()
        except Exception:
            try:
                super().paint(painter, option, index)
            except Exception:
                pass


# ── Tier pill (rounded badge with tier-color tint) ──

def tier_pill_style(tier_color_hex: str) -> str:
    """Stylesheet for a small rounded pill rendering of a tier label.

    Pass the foreground/text color as hex (e.g. ``"#4caf50"`` for the
    "good" tier).  The pill background is a faint tint of the same
    color, the border a slightly stronger one — same visual vocabulary
    as ``AccordionSection`` active state.
    """
    if (
        isinstance(tier_color_hex, str)
        and tier_color_hex.startswith("#")
        and len(tier_color_hex) == 7
    ):
        r = int(tier_color_hex[1:3], 16)
        g = int(tier_color_hex[3:5], 16)
        b = int(tier_color_hex[5:7], 16)
        bg = f"rgba({r}, {g}, {b}, 0.15)"
        border = f"rgba({r}, {g}, {b}, 0.40)"
        fg = tier_color_hex
    else:
        bg = "rgba(255, 255, 255, 0.06)"
        border = "rgba(255, 255, 255, 0.18)"
        fg = "#bbbbbb"
    return (
        f"color: {fg};"
        " font-size: 10px; font-weight: bold;"
        " padding: 2px 8px;"
        f" border: 1px solid {border};"
        " border-radius: 8px;"
        f" background-color: {bg};"
    )


# ── Themed table (QTableWidget — Manage Runs etc.) ──


def themed_table_style(accent: str = "#d4a843") -> str:
    """Stylesheet for a ``QTableWidget`` matching the EpiCycle dark/amber
    theme.  Kills Qt's default gray gridlines + header chrome, applies a
    subtle accent-tinted header bar, alternating row tint, and an
    accent-tinted selection.

    Pair with ``table.setAlternatingRowColors(True)`` and
    ``table.setShowGrid(False)`` at the call site.  Cell content stays
    plain ``QTableWidgetItem`` — no per-cell stylesheet hacks needed.

    Used by: ``_manage_runs_panel.py``.  Designed to fit on top of the
    yellow-tinged dialog background that ``themed_confirm`` /
    ``themed_three_choice`` already produce — no fighting between
    surface and table chrome.
    """
    r, g, b = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
    return (
        "QTableWidget {"
        "  background-color: rgba(255, 255, 255, 0.03);"
        "  border: 1px solid rgba(255, 255, 255, 0.08);"
        "  border-radius: 4px;"
        "  gridline-color: transparent;"
        "  font-size: 11px;"
        "  color: #cccccc;"
        "  selection-background-color: transparent;"
        "}"
        "QTableWidget::item {"
        "  padding: 5px 8px;"
        "  border: none;"
        "}"
        "QTableWidget::item:alternate {"
        "  background-color: rgba(255, 255, 255, 0.02);"
        "}"
        "QTableWidget::item:hover {"
        "  background-color: rgba(255, 255, 255, 0.05);"
        "}"
        "QTableWidget::item:selected {"
        f"  background-color: rgba({r}, {g}, {b}, 0.22);"
        "  color: #eeeeee;"
        "}"
        "QHeaderView::section {"
        f"  background-color: rgba({r}, {g}, {b}, 0.18);"
        f"  color: #eeeeee;"
        "  padding: 6px 8px;"
        "  border: none;"
        f"  border-right: 1px solid rgba({r}, {g}, {b}, 0.10);"
        "  font-size: 10px; font-weight: bold;"
        "  text-transform: uppercase;"
        "}"
        "QHeaderView::section:last { border-right: none; }"
        "QTableCornerButton::section {"
        f"  background-color: rgba({r}, {g}, {b}, 0.18);"
        "  border: none;"
        "}"
    )


THEMED_TABLE_STYLE = themed_table_style("#d4a843")


# ── Pair card (used by Import Pairs + External Import dialogs) ──


def make_pair_card(
    *,
    title: str,
    meta: str,
    img_pixmap=None,
    mask_pixmap=None,
    badge_text: str = "",
    badge_color_hex: str = "#888888",
    checked: bool = False,
    thumb_size: int = 48,
    parent: QWidget = None,
):
    """Build a card widget for one image+mask training pair.

    Layout (left-to-right):  ☐  [img thumb]  [mask thumb]  [title / meta vbox]  [badge]

    The card lives inside a ``QListWidgetItem`` via ``list.setItemWidget(item, card)``
    so the caller iterates ``list.count()`` instead of ``table.rowCount()``.
    Thumbnails come from ``_thumbnails.make_image_thumbnail`` /
    ``make_mask_thumbnail`` (already LRU-cached) — same QPixmap render
    target as the pre-card ``QTableWidgetItem.setIcon`` path.

    Args:
        title: pair label (e.g. ``"patch_2_XY_z0042"``).
        meta: one-line metadata (status, dims, label count, mtime).
        img_pixmap / mask_pixmap: ``QPixmap`` or ``None``.  When ``None``,
            renders a neutral placeholder square so card width stays
            consistent across rows.
        badge_text / badge_color_hex: status pill on the right edge.
            Empty ``badge_text`` hides the badge.
        checked: initial check state of the row's checkbox.
        thumb_size: pixel size of each thumbnail (matches existing
            48×48 convention; honored when rendering placeholders).

    Returns:
        ``(card_widget, checkbox)`` — caller connects ``checkbox.toggled``
        to its selection-tracking dict.
    """
    from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QIcon, QPixmap

    def _as_pixmap(thumb, size):
        """Accept QPixmap or QIcon; return a QPixmap or None.

        2026-05-26 bug fix: ``_thumbnails.make_image_thumbnail`` /
        ``make_mask_thumbnail`` return ``QIcon`` (designed for the old
        ``QTableWidgetItem.setIcon`` render path).  ``QLabel.setPixmap``
        needs a ``QPixmap``.  Without this conversion the previous
        ``isinstance(img_pixmap, QPixmap)`` check silently failed and
        thumbnails rendered as empty placeholder squares.
        """
        if thumb is None:
            return None
        if isinstance(thumb, QPixmap):
            return thumb if not thumb.isNull() else None
        if isinstance(thumb, QIcon):
            pm = thumb.pixmap(size, size)
            return pm if pm is not None and not pm.isNull() else None
        return None

    card = QWidget(parent=parent)
    card.setStyleSheet(
        "QWidget { background-color: transparent; }"
    )
    root = QHBoxLayout(card)
    root.setContentsMargins(8, 6, 8, 6)
    root.setSpacing(10)

    # Checkbox
    checkbox = QCheckBox(parent=card)
    checkbox.setChecked(bool(checked))
    checkbox.setStyleSheet(
        "QCheckBox { spacing: 0; }"
        "QCheckBox::indicator {"
        "  width: 16px; height: 16px;"
        "  border: 1px solid rgba(255, 255, 255, 0.30);"
        "  border-radius: 3px;"
        "  background-color: rgba(255, 255, 255, 0.04);"
        "}"
        "QCheckBox::indicator:checked {"
        "  background-color: rgba(212, 168, 67, 0.65);"
        "  border: 1px solid rgba(212, 168, 67, 0.85);"
        "}"
        "QCheckBox::indicator:hover {"
        "  border: 1px solid rgba(255, 255, 255, 0.50);"
        "}"
    )
    root.addWidget(checkbox, 0, Qt.AlignVCenter)

    # Image thumbnail
    img_label = QLabel(parent=card)
    img_label.setFixedSize(thumb_size, thumb_size)
    img_label.setAlignment(Qt.AlignCenter)
    img_label.setStyleSheet(
        "background-color: rgba(255, 255, 255, 0.08);"
        " border: 1px solid rgba(255, 255, 255, 0.18);"
        " border-radius: 3px;"
    )
    _img_pm = _as_pixmap(img_pixmap, thumb_size)
    if _img_pm is not None:
        img_label.setPixmap(_img_pm)
    root.addWidget(img_label, 0, Qt.AlignVCenter)

    # Mask thumbnail
    mask_label = QLabel(parent=card)
    mask_label.setFixedSize(thumb_size, thumb_size)
    mask_label.setAlignment(Qt.AlignCenter)
    mask_label.setStyleSheet(
        "background-color: rgba(255, 255, 255, 0.08);"
        " border: 1px solid rgba(255, 255, 255, 0.18);"
        " border-radius: 3px;"
    )
    _mask_pm = _as_pixmap(mask_pixmap, thumb_size)
    if _mask_pm is not None:
        mask_label.setPixmap(_mask_pm)
    root.addWidget(mask_label, 0, Qt.AlignVCenter)

    # Title + meta vbox
    text_col = QVBoxLayout()
    text_col.setContentsMargins(0, 0, 0, 0)
    text_col.setSpacing(2)
    title_label = QLabel(title, parent=card)
    title_label.setStyleSheet(
        "color: #eeeeee; font-size: 11px; font-weight: bold;"
        " background: transparent; border: none;"
    )
    text_col.addWidget(title_label)
    meta_label = QLabel(meta, parent=card)
    meta_label.setStyleSheet(
        "color: #999999; font-size: 10px;"
        " background: transparent; border: none;"
    )
    text_col.addWidget(meta_label)
    root.addLayout(text_col, 1)

    # Status badge on the right edge
    if badge_text:
        badge = QLabel(badge_text, parent=card)
        badge.setStyleSheet(tier_pill_style(badge_color_hex))
        badge.setAlignment(Qt.AlignCenter)
        root.addWidget(badge, 0, Qt.AlignVCenter)

    # Stash references on the card so callers can adjust later if needed.
    card._checkbox = checkbox
    card._img_label = img_label
    card._mask_label = mask_label
    card._title_label = title_label
    card._meta_label = meta_label
    return card, checkbox


# Map common pair statuses → badge color (kept here so dialogs stay
# style-free; if a pair has a custom status, dialogs can pass any hex).
PAIR_STATUS_BADGE_COLORS = {
    "accepted":  "#6abf69",   # green
    "staged":    "#d4a843",   # amber
    "committed": "#5ba4cf",   # teal — "Finalized (trained)"
    "finalized": "#80deea",   # ice blue
    "external":  "#a78bca",   # soft purple (external imports)
}


# ── Color swatch (small filled square showing a label's on-screen color) ──


_DEFAULT_LABEL_CMAP = None


def default_label_colormap():
    """Cached napari default label colormap (``label_colormap(49, seed=0.5)``).

    Used as the swatch FALLBACK when the live layer colormap hides a label
    (alpha=0, i.e. during isolation).  Building this colormap is non-trivial
    (random color gen + low-discrepancy image + searchsorted + allocations) and
    its args are CONSTANT, so the result never changes — yet ``swatch_color_for_label``
    rebuilt it on EVERY hidden-label call (~1365x per navigator rebuild during
    isolation ≈ 2.1s of pure waste).  Cache it once.  Also reused by the batch
    helper ``curation_label_navigator._batch_swatch_colors``.
    """
    global _DEFAULT_LABEL_CMAP
    if _DEFAULT_LABEL_CMAP is None:
        from napari.utils.colormaps import label_colormap
        _DEFAULT_LABEL_CMAP = label_colormap(49, seed=0.5)
    return _DEFAULT_LABEL_CMAP


def swatch_color_for_label(label_layer, label_id: int):
    """Sample a napari Labels layer's colormap for ``label_id``.

    Returns a ``QColor``.  Falls back to neutral gray if anything goes
    wrong.  When the layer's current colormap hides the label
    (alpha=0, e.g. during isolation), falls back to napari's default
    label colormap so the swatch still renders an identifiable color.
    """
    from qtpy.QtGui import QColor as _QColor
    import numpy as _np

    fallback = _QColor(204, 204, 204)
    if label_layer is None:
        return fallback
    cmap = getattr(label_layer, "colormap", None)
    if cmap is None or not hasattr(cmap, "map"):
        return fallback
    try:
        rgba = cmap.map(_np.array([int(label_id)], dtype=_np.intp))
    except Exception:
        return fallback
    if rgba is None:
        return fallback
    arr = _np.asarray(rgba).reshape(-1, 4)
    if arr.size == 0:
        return fallback
    r, g, b, a = arr[0]
    if float(a) <= 0.001:
        try:
            default_cmap = default_label_colormap()  # cached — see above
            rgba2 = default_cmap.map(_np.array([int(label_id)], dtype=_np.intp))
            r, g, b, _a = _np.asarray(rgba2).reshape(-1, 4)[0]
        except Exception:
            return fallback
    return _QColor(
        int(_np.clip(float(r) * 255, 0, 255)),
        int(_np.clip(float(g) * 255, 0, 255)),
        int(_np.clip(float(b) * 255, 0, 255)),
    )


class ColorSwatch(QLabel):
    """Small filled square showing a label's actual on-screen color.

    Used in merge dialog cards and label navigator rows for at-a-glance
    visual identification of labels by their viewer color.
    """

    def __init__(self, color, size: int = 14, parent: QWidget = None) -> None:
        super().__init__(parent=parent)
        self._size = size
        self.setFixedSize(size, size)
        self.set_color(color)

    def set_color(self, color) -> None:
        """``color`` is a ``QColor``."""
        self._color = color
        self.setStyleSheet(
            f"background-color: rgba({color.red()}, {color.green()},"
            f" {color.blue()}, 255);"
            " border: 1px solid rgba(255, 255, 255, 0.18);"
            " border-radius: 2px;"
        )


def elide_label(label: QLabel, text: str, max_px: int,
                mode=None) -> str:
    """Put *text* on *label*, shortened with an ellipsis to fit *max_px*, and
    keep the full text reachable as the tooltip.

    A list row is a fixed-width thing with buttons on its right.  A plain
    ``QLabel`` asks for however many pixels its text needs, so one long name
    pushes the whole row -- and therefore the panel -- wider.  Eliding caps what
    the label asks for while losing nothing: the full name is one hover away.

    Returns the text actually shown, so callers can tell whether it was cut.
    """
    from qtpy.QtGui import QFontMetrics

    text = "" if text is None else str(text)
    if mode is None:
        mode = Qt.ElideMiddle
    try:
        shown = QFontMetrics(label.font()).elidedText(text, mode, int(max_px))
    except Exception:          # noqa: BLE001 -- cosmetic; never break a row
        shown = text
    label.setText(shown)
    label.setToolTip(text)
    # The ELISION is what keeps the size hint small -- a QLabel's hint follows
    # the text it holds, and that text is now at most *max_px* wide.  This cap
    # does something narrower: it stops the label stretching past *max_px* on a
    # WIDE dock, so the row's buttons stay put instead of drifting right.  It is
    # not what prevents the row growing.
    label.setMaximumWidth(int(max_px))
    return shown


def apply_inner_group_style(widget: QWidget) -> None:
    """Apply the inner QGroupBox glass style to all QGroupBox children."""
    from qtpy.QtWidgets import QGroupBox
    for grp in widget.findChildren(QGroupBox):
        grp.setStyleSheet(INNER_GROUP_STYLE)


def apply_button_icon(button, icon_name: str, color: str = "#eeeeee",
                      size: int = 16) -> bool:
    """Set a qtawesome icon on a QPushButton.

    Returns True if the icon resolved + was applied, False if it was
    unavailable — so a caller can keep a text fallback rather than leave a
    blank button (existing callers ignore the return, unchanged)."""
    qicon = _qta_icon_or_none(icon_name, color)
    if qicon is not None:
        button.setIcon(qicon)
        button.setIconSize(QSize(size, size))
        return True
    return False


class ComingSoonOverlay(QWidget):
    """A panel with a semi-transparent 'Coming Soon' overlay.

    Wraps *content* (the real panel contents) and overlays a translucent
    gray veil with a centered label.  The underlying widgets are visible
    but non-interactive, signalling that the feature is planned.
    """

    _OVERLAY_STYLE = (
        "background: rgba(30, 30, 35, 0.75);"
        "border-radius: 4px;"
    )
    _LABEL_STYLE = (
        "color: rgba(255, 255, 255, 0.55);"
        "font-size: 13px;"
        "font-weight: bold;"
        "background: transparent;"
    )

    def __init__(self, content: QWidget, label_text: str = "Coming Soon",
                 parent: QWidget = None) -> None:
        super().__init__(parent)
        from qtpy.QtWidgets import QLabel, QStackedLayout
        layout = QStackedLayout(self)
        layout.setStackingMode(QStackedLayout.StackAll)

        # Layer 0: real content (disabled).
        content.setEnabled(False)
        layout.addWidget(content)

        # Layer 1: translucent overlay with centered label.
        overlay = QWidget()
        overlay.setStyleSheet(self._OVERLAY_STYLE)
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.addStretch(1)
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(self._LABEL_STYLE)
        overlay_layout.addWidget(lbl)
        overlay_layout.addStretch(1)
        layout.addWidget(overlay)


# ── Themed confirmation dialog (dark, consistent with napari) ──

_DIALOG_STYLE = """
QDialog {
    background-color: #2b2b2b;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
}
QLabel {
    color: #dddddd;
    font-size: 12px;
}
"""

_DIALOG_BTN_CONFIRM = """
QPushButton {
    background-color: rgba(212, 168, 67, 0.25);
    border: 1px solid rgba(212, 168, 67, 0.40);
    border-radius: 4px;
    color: #eeeeee;
    font-weight: bold;
    padding: 6px 18px;
    min-width: 80px;
}
QPushButton:hover {
    background-color: rgba(212, 168, 67, 0.35);
}
"""

_DIALOG_BTN_CANCEL = """
QPushButton {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 4px;
    color: #bbbbbb;
    padding: 6px 18px;
    min-width: 80px;
}
QPushButton:hover {
    background-color: rgba(255, 255, 255, 0.10);
}
"""


_DIALOG_BTN_CONFIRM_GREEN = """
QPushButton {
    background-color: rgba(106, 191, 105, 0.25);
    border: 1px solid rgba(106, 191, 105, 0.40);
    border-radius: 4px;
    color: #eeeeee;
    font-weight: bold;
    padding: 6px 18px;
    min-width: 80px;
}
QPushButton:hover {
    background-color: rgba(106, 191, 105, 0.35);
}
"""


def themed_confirm(parent, title: str, message: str,
                   confirm_text: str = "Continue",
                   cancel_text: str = "Cancel",
                   accent: str = "amber") -> bool:
    """Show a dark-themed confirmation dialog. Returns True if confirmed.

    Args:
        accent: ``"amber"`` (Build/Step 3), ``"green"`` (Apply/Step 4) or
            ``"red"`` (a destructive confirm: Remove, Reset, Clear 2D
            predictions).
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setStyleSheet(_DIALOG_STYLE)
    dlg.setMinimumWidth(380)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(12)

    # Title
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #eeeeee;")
    layout.addWidget(title_lbl)

    # Message
    msg_lbl = QLabel(message)
    msg_lbl.setWordWrap(True)
    msg_lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
    layout.addWidget(msg_lbl)

    layout.addSpacing(4)

    # Buttons
    btn_row = QHBoxLayout()
    btn_row.addStretch(1)

    cancel_btn = QPushButton(cancel_text)
    cancel_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
    cancel_btn.clicked.connect(dlg.reject)

    confirm_btn = QPushButton(confirm_text)
    btn_style = (
        _DIALOG_BTN_CONFIRM_RED if accent == "red"
        else _DIALOG_BTN_CONFIRM_GREEN if accent == "green"
        else _DIALOG_BTN_CONFIRM
    )
    confirm_btn.setStyleSheet(btn_style)
    confirm_btn.clicked.connect(dlg.accept)
    confirm_btn.setDefault(True)

    btn_row.addWidget(cancel_btn)
    btn_row.addWidget(confirm_btn)
    layout.addLayout(btn_row)

    return dlg.exec_() == QDialog.Accepted


def themed_notice(parent, title: str, message: str,
                  ok_text: str = "OK") -> None:
    """A dark-themed, one-button notice -- ``themed_confirm``'s sibling for
    a sentence that needs no decision (e.g. why Batch Refine cannot open)."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setStyleSheet(_DIALOG_STYLE)
    dlg.setMinimumWidth(380)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(12)

    title_lbl = QLabel(title, parent=dlg)
    title_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #eeeeee;")
    layout.addWidget(title_lbl)

    msg_lbl = QLabel(message, parent=dlg)
    msg_lbl.setWordWrap(True)
    msg_lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
    layout.addWidget(msg_lbl)

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    ok_btn = QPushButton(ok_text, parent=dlg)
    ok_btn.setStyleSheet(_DIALOG_BTN_CONFIRM)
    ok_btn.clicked.connect(dlg.accept)
    ok_btn.setDefault(True)
    btn_row.addWidget(ok_btn)
    layout.addLayout(btn_row)

    dlg.exec_()
    # Parented to the caller, so without this every notice stayed alive,
    # hidden, until the caller died (audit D-L4).
    dlg.deleteLater()


def themed_three_choice(parent, title: str, message: str,
                        primary_text: str = "Primary",
                        secondary_text: str = "Secondary",
                        cancel_text: str = "Cancel",
                        accent: str = "amber") -> str:
    """Show a 3-button dark-themed dialog.

    Used for "fork" decisions where Cancel isn't the only escape — e.g.
    the extraction-staleness gate at training launch:
      [Re-extract stale patches first] [Train anyway] [Cancel]
    or the mixed-settings gate at extraction launch:
      [Re-extract these too] [Keep mixed] [Cancel]

    Returns one of: ``"primary"``, ``"secondary"``, ``"cancel"``.

    The primary button is rendered in the accent color (amber for Step 3,
    green for Step 4); secondary is rendered as a neutral filled button
    (still highlighted vs. cancel); cancel uses the muted cancel style.

    Args:
        accent: ``"amber"`` (Build/Step 3) or ``"green"`` (Apply/Step 4).
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setStyleSheet(_DIALOG_STYLE)
    dlg.setMinimumWidth(460)  # wider — three buttons need more room.

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(12)

    # Title
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #eeeeee;")
    layout.addWidget(title_lbl)

    # Message
    msg_lbl = QLabel(message)
    msg_lbl.setWordWrap(True)
    msg_lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
    layout.addWidget(msg_lbl)

    layout.addSpacing(4)

    # Result holder — we use a list-of-one as a mutable closure cell
    # since lambdas can't directly assign to enclosing-scope names
    # in older Python (still common in our envs).
    result_holder = ["cancel"]

    def _click_primary():
        result_holder[0] = "primary"
        dlg.accept()

    def _click_secondary():
        result_holder[0] = "secondary"
        dlg.accept()

    def _click_cancel():
        result_holder[0] = "cancel"
        dlg.reject()

    # Buttons
    btn_row = QHBoxLayout()
    btn_row.addStretch(1)

    cancel_btn = QPushButton(cancel_text)
    cancel_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
    cancel_btn.clicked.connect(_click_cancel)

    # "Secondary" — neutral filled, between cancel and primary in
    # visual weight.  Reuses the cancel style as a base then could
    # be styled further; for now we accept the cancel-style read so
    # the eye still goes to the primary action.
    secondary_btn = QPushButton(secondary_text)
    secondary_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
    secondary_btn.clicked.connect(_click_secondary)

    primary_btn = QPushButton(primary_text)
    btn_style = (
        _DIALOG_BTN_CONFIRM_GREEN if accent == "green"
        else _DIALOG_BTN_CONFIRM
    )
    primary_btn.setStyleSheet(btn_style)
    primary_btn.clicked.connect(_click_primary)
    primary_btn.setDefault(True)

    btn_row.addWidget(cancel_btn)
    btn_row.addWidget(secondary_btn)
    btn_row.addWidget(primary_btn)
    layout.addLayout(btn_row)

    dlg.exec_()
    return result_holder[0]


# ────────────────────────────────────────────────────────────────────────────
# Help system (2026-05-18)
# ────────────────────────────────────────────────────────────────────────────
#
# `themed_help_dialog` shows a NON-MODAL floating window that renders a
# markdown guide.  Non-modal so users can read the guide while still
# clicking through the UI it describes — the original modal proposal was
# rejected because step-by-step guidance + blocking-modal is incompatible.
#
# `help_button` creates the small `?` icon button that opens the dialog.
# Drop one into each accordion section header to give users an entry
# point.  Each section's `?` opens a different markdown file under
# `docs/guides/`.
#
# Content style: plain biological language, not raw ML/engineering jargon.  Every
# parameter explanation should read like a wet-lab protocol footnote.

# Catalog of open help dialogs, keyed by guide_path, so clicking `?`
# twice doesn't open two copies.  Weak ownership — when a dialog closes,
# the user's QWidget destruction cleans up the reference automatically
# via the finished callback.
_OPEN_HELP_DIALOGS: dict[str, QDialog] = {}


# Embedded CSS for the QTextBrowser markdown render.  Tuned for dark
# theme readability — biologists will spend real time in here, so
# generous spacing, clear hierarchy, biological-doc tone.
def _help_dialog_css(accent: str) -> str:
    accent_dim = _rgba(accent, 0.55)
    accent_bg = _rgba(accent, 0.10)
    accent_border = _rgba(accent, 0.35)
    return f"""
    body {{
        background-color: #1e1e1e;
        color: #b8b8b8;
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        font-size: 12pt;
        line-height: 1.55;
        padding: 6px 12px;
    }}
    h1 {{
        color: #dcdcdc;
        font-size: 18pt;
        font-weight: 600;
        margin-top: 4px;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 2px solid {accent_dim};
    }}
    h2 {{
        color: #d0d0d0;
        font-size: 14pt;
        font-weight: 600;
        margin-top: 22px;
        margin-bottom: 6px;
        padding-bottom: 3px;
        border-bottom: 1px solid {_rgba(accent, 0.25)};
    }}
    h3 {{
        color: #c4c4c4;
        font-size: 12pt;
        font-weight: 600;
        margin-top: 16px;
        margin-bottom: 4px;
    }}
    p {{
        margin: 8px 0;
    }}
    ul, ol {{
        margin: 6px 0 10px 22px;
        padding-left: 4px;
    }}
    li {{
        margin: 3px 0;
    }}
    li p {{
        margin: 2px 0;
    }}
    code {{
        font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;
        background-color: rgba(255, 255, 255, 0.05);
        color: #e0b870;
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 10.5pt;
    }}
    pre {{
        background-color: #161616;
        border-left: 3px solid {accent_dim};
        padding: 10px 14px;
        margin: 10px 0;
        border-radius: 4px;
        font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;
        font-size: 10.5pt;
        color: #b8b8b8;
        white-space: pre-wrap;
    }}
    pre code {{
        background-color: transparent;
        padding: 0;
        color: inherit;
        font-size: inherit;
    }}
    blockquote {{
        background-color: {accent_bg};
        border-left: 3px solid {accent_border};
        padding: 8px 14px;
        margin: 10px 0;
        color: #b4b4b4;
        font-style: normal;
        border-radius: 0 4px 4px 0;
    }}
    blockquote p {{
        margin: 4px 0;
    }}
    strong {{
        color: #d4d4d4;
        font-weight: 600;
    }}
    em {{
        color: #b8b8b8;
        font-style: italic;
    }}
    a {{
        color: {accent};
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    hr {{
        border: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin: 18px 0;
    }}
    table {{
        border-collapse: collapse;
        margin: 10px 0;
    }}
    th, td {{
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 6px 10px;
    }}
    th {{
        background-color: rgba(255, 255, 255, 0.05);
        font-weight: 600;
        color: #d8d8d8;
    }}
    """


def themed_help_dialog(parent, title: str, markdown_path,
                       accent: str = "#d4a843") -> QDialog:
    """Open a non-modal floating help dialog rendering a markdown guide.

    Args:
        parent: the QWidget hosting the help button (typically the
            section header).  Used for window-parent ownership so the
            dialog tracks the main napari window's z-order.
        title: window title (also rendered as the dialog's accent header).
        markdown_path: filesystem path to the `.md` file to render.
            Uses Qt's native ``setMarkdown`` (Qt 5.14+) so no Python
            markdown library dependency is required.
        accent: hex color for header band + accent borders.  Pass
            ``"#d4a843"`` for Step 3 (amber), ``"#6abf69"`` for Step 4
            (green), etc.  Defaults to amber.

    Returns:
        the ``QDialog`` reference.  Caller can ignore it — the helper
        manages lifetime via the ``_OPEN_HELP_DIALOGS`` registry so
        clicking `?` repeatedly raises the existing dialog instead of
        creating duplicates.
    """
    from pathlib import Path as _Path

    key = str(markdown_path)

    # If a dialog for this guide is already open, raise it instead of
    # opening a second one.  User clicked `?` twice probably to find
    # the existing window.
    existing = _OPEN_HELP_DIALOGS.get(key)
    if existing is not None:
        try:
            existing.raise_()
            existing.activateWindow()
            return existing
        except Exception:
            # Existing dialog was destroyed without our finished signal —
            # drop the stale ref and fall through to create a fresh one.
            _OPEN_HELP_DIALOGS.pop(key, None)

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    # Tool window: doesn't grab focus, stays out of the taskbar, but
    # remains a normal movable/resizable window the user can position.
    # CRITICALLY does NOT block input on the parent — this is the
    # whole point of non-modal help.
    dlg.setModal(False)
    flags = (
        Qt.Tool
        | Qt.WindowTitleHint
        | Qt.WindowCloseButtonHint
        | Qt.CustomizeWindowHint
    )
    try:
        dlg.setWindowFlags(flags)
    except Exception:
        pass
    dlg.setStyleSheet(_DIALOG_STYLE)
    dlg.resize(680, 620)
    dlg.setMinimumSize(520, 400)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    # Accent header band — gives the dialog a clear visual tie to the
    # section it belongs to (amber for Step 3, green for Step 4, etc.).
    header = QFrame()
    header.setFixedHeight(38)
    header.setStyleSheet(
        f"QFrame {{ background-color: {_rgba(accent, 0.18)}; "
        f"border-bottom: 1px solid {_rgba(accent, 0.45)}; }}"
    )
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(16, 0, 14, 0)
    header_layout.setSpacing(8)
    header_title = QLabel(title)
    header_title.setStyleSheet(
        f"color: #d8d8d8; font-size: 13px; font-weight: 600; "
        f"background-color: transparent;"
    )
    header_layout.addWidget(header_title)
    header_layout.addStretch(1)
    # qtawesome icon if available, fall back to a colored bullet.
    try:
        import qtawesome as qta
        icon_lbl = QLabel()
        icon_pix = qta.icon(
            "mdi6.help-circle-outline", color=accent
        ).pixmap(QSize(18, 18))
        icon_lbl.setPixmap(icon_pix)
        icon_lbl.setStyleSheet("background-color: transparent;")
        header_layout.addWidget(icon_lbl)
    except Exception:
        pass
    layout.addWidget(header)

    # Body — QTextBrowser renders markdown via Qt's native setMarkdown.
    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    # Apply the typography CSS via document stylesheet so it cascades
    # through the rendered markdown HTML.
    browser.document().setDefaultStyleSheet(_help_dialog_css(accent))
    browser.setStyleSheet(
        "QTextBrowser { background-color: #1e1e1e; border: 0; }"
    )

    md_path = _Path(str(markdown_path))
    if md_path.exists():
        try:
            md_text = md_path.read_text(encoding="utf-8")
        except Exception as exc:
            md_text = (
                f"# Error loading guide\n\n"
                f"Could not read `{md_path}`:\n\n```\n{exc}\n```"
            )
    else:
        md_text = (
            f"# Guide not yet written\n\n"
            f"The markdown file for this section does not exist yet:\n\n"
            f"`{md_path}`\n\n"
            f"This is a placeholder — content is being added section by "
            f"section in the guides folder."
        )
    browser.setMarkdown(md_text)

    layout.addWidget(browser, 1)

    # Footer with Close button — also accent-tinted, consistent visual.
    footer = QFrame()
    footer.setStyleSheet(
        "QFrame { background-color: #232323; "
        "border-top: 1px solid rgba(255, 255, 255, 0.08); }"
    )
    footer_layout = QHBoxLayout(footer)
    footer_layout.setContentsMargins(14, 8, 14, 8)
    footer_layout.addStretch(1)
    close_btn = QPushButton("Close")
    close_btn.setStyleSheet(primary_btn_style(accent))
    close_btn.setMinimumWidth(80)
    close_btn.clicked.connect(dlg.close)
    footer_layout.addWidget(close_btn)
    layout.addWidget(footer)

    # Track the dialog in the registry, drop on close so re-clicks open
    # a fresh window instead of raising a destroyed one.
    _OPEN_HELP_DIALOGS[key] = dlg

    def _on_finished(_result):
        _OPEN_HELP_DIALOGS.pop(key, None)
    dlg.finished.connect(_on_finished)

    dlg.show()
    return dlg


def help_button(parent, guide_path, title: str,
                accent: str = "#d4a843",
                tooltip: str = "Open the guide for this section") -> QPushButton:
    """Create a small `?` icon button that opens a guide dialog on click.

    Drop one into any accordion section header.  The button is sized to
    fit alongside section text (22×22 px) and uses qtawesome's
    ``mdi6.help-circle-outline`` glyph when available, falling back to a
    text `?` if qtawesome isn't installed.

    Args:
        parent: the QWidget host (usually the section header layout's
            owner).  Used as the help dialog's parent.
        guide_path: filesystem path to the markdown guide to open.
        title: window title for the dialog.
        accent: hex color matching the host section's accent.
        tooltip: hover text.
    """
    btn = QPushButton(parent)
    btn.setFixedSize(22, 22)
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setStyleSheet(
        f"QPushButton {{"
        f"  background-color: transparent;"
        f"  border: 1px solid {_rgba(accent, 0.35)};"
        f"  border-radius: 11px;"
        f"  color: {_rgba(accent, 0.80)};"
        f"  font-size: 11px; font-weight: bold;"
        f"  padding: 0px;"
        f"}}"
        f"QPushButton:hover {{"
        f"  background-color: {_rgba(accent, 0.15)};"
        f"  border: 1px solid {_rgba(accent, 0.55)};"
        f"  color: #f0f0f0;"
        f"}}"
        f"QPushButton:pressed {{"
        f"  background-color: {_rgba(accent, 0.25)};"
        f"}}"
    )
    # qtawesome glyph if available; otherwise text fallback.
    try:
        import qtawesome as qta
        btn.setIcon(qta.icon(
            "mdi6.help-circle-outline", color=accent
        ))
        btn.setIconSize(QSize(14, 14))
    except Exception:
        btn.setText("?")

    btn.clicked.connect(
        lambda: themed_help_dialog(parent, title, guide_path, accent=accent)
    )
    return btn


# ---- Step 4 revamp (B1a) ----
# Appended only, so a sibling brick that
# also appends to this file merges cleanly.

#: Step 4's accent (Apply green), as a name.
STEP4_ACCENT = "#6abf69"
#: ``list_style`` in Step 4's accent (Appendix A ``.lst``).
LIST_STYLE_GREEN = list_style(STEP4_ACCENT)

#: The red confirm of ``themed_confirm(accent="red")``.
_DIALOG_BTN_CONFIRM_RED = destructive_btn_style("#c62828") + """
QPushButton { padding: 6px 18px; min-width: 80px; }
"""

# ── Pipeline sentences (Steps 3 and 4) ──
# ONE table for the words both steps show while Predict / Reconstruct run.
# Step 3 renders them with unit="patch" and is byte-identical to its old
# f-strings (pinned by tests/segmentation/test_pipe_sentences_step3_golden.py);
# Step 4 renders them with unit="stack".  ``{units}`` is the plural, from
# ``pipe_units`` ("patch" + "s" would print "patchs").  Counts stay plural
# whatever the number, because Step 3 prints "1 patches" today.
PIPE_SEGMENTING_START = "Segmenting {n} {units} ({slices} slices)..."
PIPE_SEGMENTING_ITEM = "Segmenting {i}/{n}: {name}  {time}"
PIPE_SEGMENTING_VIEW = (
    "Segmenting {unit} {i}/{n}: {name} — {VIEW} ({c}/{t})  {time}")
PIPE_SEGMENTATION_DONE = "Segmentation complete: {s}/{n} {units}"
PIPE_FAILED_SUFFIX = " ({k} failed)"
PIPE_TIME_SUFFIX = "  [{t}]"
PIPE_RECONSTRUCTING_START = "Reconstructing {n} {units}..."
PIPE_RECONSTRUCTING_ITEM = "Reconstructing {i}/{n}: {name}...  {time}"
PIPE_RECONSTRUCTION_DONE = "Reconstruction complete: {s}/{n} {units}"
PIPE_CANCELLING = "Cancelling... will stop after current {unit} completes"
PIPE_ERROR_PREDICT = "[ERROR] Prediction failed for {name}: {msg}"
PIPE_ERROR_RECON = "[ERROR] Reconstruction failed for {name}: {msg}"
#: The running words on a list row (Step 3's patch list, Step 4's stack rows).
ROW_SEGMENTING = "segmenting..."
ROW_RECONSTRUCTING = "reconstructing..."
#: The run combo's one item (Step 3's ``_format_run_picker_label``: the
#: active marker "● " and a count-aware unit).
RUN_ROW_DEFAULT = "● Default  ({n} {units})"
#: The model card's indicator (Step 3's ``_refresh_model_status``).
MODEL_NONE = "(no model loaded — click Change… to pick)"
MODEL_ACTIVE_PAIR = "lat: {lateral}  |  ax: {axial}"

_PIPE_PLURALS = {"patch": "patches", "stack": "stacks"}


def pipe_units(unit: str, n=None) -> str:
    """The plural of *unit* ("patches" / "stacks").  With a count *n* of
    exactly 1 the singular comes back (``RUN_ROW_DEFAULT`` is count-aware;
    the ``PIPE_*`` sentences never pass *n*)."""
    if n is not None and n == 1:
        return unit
    return _PIPE_PLURALS.get(unit, unit + "s")


def pipe_time(elapsed: str, eta: str = "") -> str:
    """Step 3's time part: ``"[{elapsed}"`` + ``" | {eta}]"`` or ``"]"``."""
    return f"[{elapsed}" + (f" | {eta}]" if eta else "]")


# =============================================================================
# The QC plot's palette
# =============================================================================
#
# ``QCPlotWidget(layout="workbench", palette=...)`` takes a ``PlotPalette``
# (defined in ``fusion_reconstruction/qc_plot_widget.py``) OR this plain
# mapping, so the shell never imports a step module.  Step 4 passes no
# palette and keeps the plot's own green default.


def plot_palette(accent: str, *, ok_text: str = None) -> dict:
    """The mapping ``QCPlotWidget(palette=...)`` accepts, built from a
    ``#rrggbb`` hex: ``{"accent", "accent_rgb", "ok_text"}``.

    ``ok_text`` (the colour of an "ok" sentence) defaults to the accent.
    """
    h = str(accent).strip()
    if len(h) != 7 or not h.startswith("#"):
        raise ValueError(f"plot_palette needs a #rrggbb hex, got {accent!r}")
    try:
        rgb = (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))
    except ValueError:
        raise ValueError(
            f"plot_palette needs a #rrggbb hex, got {accent!r}") from None
    return {"accent": h, "accent_rgb": rgb,
            "ok_text": h if ok_text is None else str(ok_text)}


#: Step 3's QC plot in amber.  An "ok" line is NEUTRAL text, never amber
#: (the accent appears only as a tint, an edge or a
#: grip, never as the colour of a sentence).
STEP3_PLOT_PALETTE = plot_palette(PANEL_GOLD, ok_text=PANEL_ROW_LABEL_COLOR)

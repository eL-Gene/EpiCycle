"""The settings column's form helpers, shared by Steps 3 and 4.

Step 4's ``.frm4`` form: a two-column
grid with a 128 px field column, the Advanced row (the plain name over the
grey mono engine name), the tick whose name toggles it, the collapsible with
the engine name on its CLOSED header, the amber lock line, and the section
sub-header with its ``?``.  Moved verbatim from ``segmentation/ui.py``;
``SegmentStack``'s ``_frm4`` / ``_field`` / ``_adv_row`` / ``_collapsible``
/ ``_sub_header`` stay methods that call these functions, and ``ui.py``
keeps ``_TickLabel`` / ``_LockLine`` / ``LOCK_LINE_STYLE`` as names of
what lives here.

Stateless: ``adv_row`` writes the label it builds into the *labels* dict it
is handed (an output) and returns it.  The two object names Step 4's
stylesheets select on are keyword parameters with Step 4's values as
defaults (``step4LockLine``, ``step4EngineName``), so Step 4's panel is
unchanged.  Imports nothing from any step package.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
)
from superqt import QCollapsible

from ._recon_settings_text import STEP4_ADVANCED_LABELS
from .theme import COLLAPSIBLE_DURATION, SECTION_HEADER_STYLE, help_button

FORM_FIELD_W = 128             # .frm4 fields
ENGINE_NAME_STYLE = (
    "color: #777777; font-size: 10px; "
    "font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;")
ZWSP = "\u200b"
#: the lock line's amber (both steps' lock lines are amber)
LOCK_AMBER = "#d4a843"
LOCK_LINE_OBJECT = "step4LockLine"
ENGINE_NAME_OBJECT = "step4EngineName"


def lock_line_style(object_name: str = LOCK_LINE_OBJECT) -> str:
    """The lock line's stylesheet, selecting on *object_name*."""
    return (
        f"QFrame#{object_name} {{ background: rgba(212,168,67,0.08); "
        "border-left: 3px solid #d4a843; }"
        f"QFrame#{object_name} QLabel {{ color: #d4a843; font-size: 11px; "
        "background: transparent; }"
    )


def adv_label_html(widget_name: str, table=None) -> str:
    """The .frm4 label: the plain name over the grey mono engine name.
    *table* is ``widget -> (plain, engine)`` (default: the shared
    ``STEP4_ADVANCED_LABELS``)."""
    import html
    plain, engine = (STEP4_ADVANCED_LABELS if table is None
                     else table)[widget_name]
    # A zero-width space after each "_" lets a long engine name wrap in
    # the narrow column instead of widening it.
    eng = html.escape(engine).replace("_", "_" + ZWSP)
    return (f"{html.escape(plain)}<br><span style=\"{ENGINE_NAME_STYLE}\">"
            f"{eng}</span>")


class TickLabel(QLabel):
    """The plain name to the LEFT of an Advanced tick (the text is kept off
    the box so a long name wraps); clicking the name toggles the tick
    (fidelity F5), as a check box's own text would."""

    def __init__(self, text: str, buddy):
        super().__init__(text)
        self._tick = buddy
        self.setCursor(Qt.PointingHandCursor)

    def mouseReleaseEvent(self, event):
        tick = self._tick
        if (event.button() == Qt.LeftButton and tick is not None
                and tick.isEnabled()):
            tick.toggle()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class LockLine(QFrame):
    """An amber lock line: 8 % amber fill, a 3 px amber left border,
    a lock icon and the running job's lock sentence."""

    def __init__(self, parent=None, *, object_name: str = LOCK_LINE_OBJECT):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setStyleSheet(lock_line_style(object_name))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(6)
        self._icon = QLabel(self)
        try:
            import qtawesome as qta
            self._icon.setPixmap(
                qta.icon("mdi6.lock-outline", color=LOCK_AMBER).pixmap(14, 14))
        except Exception:      # noqa: BLE001
            pass
        self._icon.setAlignment(Qt.AlignTop)
        lay.addWidget(self._icon, 0, Qt.AlignTop)
        self._label = QLabel(self)
        self._label.setWordWrap(True)
        lay.addWidget(self._label, 1)
        self.setVisible(False)

    def setText(self, text: str) -> None:
        self._label.setText(text or "")
        self.setVisible(bool(text))

    def text(self) -> str:
        return self._label.text()


def frm4() -> QGridLayout:
    """The .frm4 grid: a stretching label column, a 128 px field column."""
    g = QGridLayout()
    g.setContentsMargins(0, 0, 0, 0)
    g.setHorizontalSpacing(8)
    g.setVerticalSpacing(5)
    g.setColumnStretch(0, 1)
    g.setColumnMinimumWidth(1, FORM_FIELD_W)
    return g


def field(w):
    """*w* at the .frm4 field width."""
    w.setFixedWidth(FORM_FIELD_W)
    return w


def adv_row(grid: QGridLayout, widget_name: str, widget, labels=None, *,
            table=None, tick_label=None) -> QLabel:
    """One .frm4 row: the plain name over the grey mono engine name on
    the left, the field (128 px) or the tick on the right.  A tick keeps
    its text off the button so a long plain name wraps instead of
    widening the column.  The label is
    written into *labels* (an output dict, when given) and returned;
    *tick_label* is the tick's label class (default ``TickLabel``)."""
    r = grid.rowCount()
    if isinstance(widget, QCheckBox):
        cls = TickLabel if tick_label is None else tick_label
        lbl = cls(adv_label_html(widget_name, table), widget)
    else:
        lbl = QLabel(adv_label_html(widget_name, table))
    lbl.setTextFormat(Qt.RichText)
    lbl.setWordWrap(True)
    grid.addWidget(lbl, r, 0)
    if isinstance(widget, QCheckBox):
        widget.setText("")
        lbl.setBuddy(widget)
        grid.addWidget(widget, r, 1, Qt.AlignLeft | Qt.AlignVCenter)
    else:
        grid.addWidget(field(widget), r, 1)
    if labels is not None:
        labels[widget_name] = lbl
    return lbl


def collapsible(title: str, engine: str = "", *,
                engine_object_name: str = ENGINE_NAME_OBJECT) -> QCollapsible:
    """A collapsible with the right-pointing arrow when closed (F10);
    *engine* puts the grey mono engine name on the CLOSED header line,
    right-aligned (F5), instead of as a line inside the body."""
    c = QCollapsible(title, collapsedIcon="\u25b6")
    c.setDuration(COLLAPSIBLE_DURATION)
    # a tight pitch: superqt's default margins cost ~24 px a header (F4/F5)
    c.layout().setContentsMargins(0, 0, 0, 0)
    c.layout().setSpacing(0)
    if engine:
        btn = c._toggle_btn
        eng = QLabel(engine, btn)
        eng.setObjectName(engine_object_name)
        eng.setStyleSheet(ENGINE_NAME_STYLE)
        eng.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        row = QHBoxLayout(btn)
        row.setSizeConstraint(QHBoxLayout.SetNoConstraint)
        row.setContentsMargins(0, 0, 4, 0)
        row.addStretch(1)
        row.addWidget(eng, 0, Qt.AlignRight | Qt.AlignVCenter)
        c._epicycle_engine_label = eng
    c._epicycle_collapsed_symbol = "\u25b6"
    return c


def sub_header(parent, title: str, guide_path: str = "", *, accent: str,
               help_buttons=None, guide: str = "") -> QHBoxLayout:
    """A section sub-header: the bold title, and a ``?`` opening
    *guide_path* in *accent* when a guide is given (the button is appended
    to *help_buttons* as ``(button, guide)``, an output list)."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    lbl = QLabel(title)
    lbl.setStyleSheet(SECTION_HEADER_STYLE)
    row.addWidget(lbl)
    row.addStretch(1)
    if guide_path:
        btn = help_button(parent, guide_path, title, accent=accent)
        row.addWidget(btn)
        if help_buttons is not None:
            help_buttons.append((btn, guide))
    return row

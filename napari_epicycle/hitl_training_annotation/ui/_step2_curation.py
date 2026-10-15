"""Step 2 Curation, the right column and the run row
(brick P module half).

Builders RECEIVE the widgets ``main_widget.py`` creates (the 21 lock-list
widgets stay literal ``self.x =`` assignments there) and lay them out; they
create only NEW widgets: the run row's ``Manage`` menu button, the View
switch proxy, the Curation progress table, the foot's stage line, the help
rings.  Nothing here imports the controller; every sentence is a
``CUR_*`` constant of ``_step2_text.py``.

Public: :func:`build_run_row`, :class:`ViewSwitch`, :func:`build_progress_table`,
:func:`build_model_block`, :func:`build_pinned_foot`, :func:`build_right_column`,
:func:`compose_stage_line`, :func:`compose_progress_note`,
:func:`compose_inference_summary`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from qtpy.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, QTimer
from qtpy.QtWidgets import (QButtonGroup, QComboBox, QFrame, QGridLayout,
                            QHBoxLayout, QLabel, QMenu, QScrollArea,
                            QSizePolicy, QSplitter, QToolButton,
                            QVBoxLayout, QWidget)

from napari_epicycle.ui_shell.theme import (
    MONO_FONT, MONO_LABEL_STYLE, PANEL_BRIGHT_COLOR, PANEL_CARD_OBJECT,
    PANEL_CARD_STYLE, PANEL_ERROR_TEXT_COLOR, PANEL_GOLD, PANEL_HELP_COLOR,
    PANEL_HELP_STYLE, PANEL_RIGHT_INNER_MARGIN, PANEL_PROGRESS_H, PANEL_PROGRESS_H_SHORT,
    PANEL_ROW_LABEL_COLOR, PANEL_ROW_LABEL_STYLE, PANEL_SEGMENT_STYLE,
    PANEL_ZERO_COLOR, PRIMARY_BTN_STYLE, SCROLLBAR_STYLE, SECONDARY_BTN_STYLE,
    SECONDARY_TOOLBTN_STYLE, SECTION_HEADER_STYLE, help_button,
    panel_foot_style, panel_form_label, progress_bar_style)

from ._step2_text import (  # noqa: F401 -- the words live in one place
    CUR_RUN_LABEL,
    CUR_NEW_RUN_BTN,
    CUR_MANAGE_BTN,
    CUR_MANAGE_RUNS_ACTION,
    CUR_MANAGE_CAPTION,
    CUR_IMPORT_PAIRS_ACTION,
    CUR_IMPORT_EXTERNAL_ACTION,
    CUR_VIEW_HEADER,
    CUR_VIEW_DESC,
    CUR_VIEW_LATERAL,
    CUR_VIEW_AXIAL,
    CUR_PROGRESS_HEADER,
    CUR_PROGRESS_DESC,
    CUR_PROGRESS_NOTE,
    CUR_PROGRESS_NOTE_NO_ROUNDS,
    CUR_MODEL_HEADER,
    CUR_MODEL_DESC,
    CUR_MODEL_USE_LABEL,
    CUR_MODEL_PRETRAINED_LABEL,
    CUR_MODEL_ACTIVE,
    CUR_MODEL_PRETRAINED_NOTE,
    CUR_MODEL_EXTERNAL_NOTE,
    CUR_MODEL_AUTO_UPDATE,
    CUR_MODEL_CHANNEL1_LABEL,
    CUR_MODEL_CHANNEL2_LABEL,
    CUR_MODEL_INFERENCE_TITLE,
    CUR_BTN_OPEN,
    CUR_BTN_ACCEPT,
    CUR_BTN_FINALIZE,
    CUR_BTN_DISCARD,
    CUR_BTN_REVERT,
    CUR_STAGE_READY,
    CUR_STAGE_NO_PATCH,
    CUR_STAGE_NO_SLICES,
    CUR_STAGE_LOCKED,
    CUR_STAGE_SEL_ONE,
    CUR_STAGE_SEL_MANY,
    CUR_STAGE_SEL_NONE,
    STATUS_COLOURS,
    STATUS_ICONS,
    STATUS_ORDER,
    STATUS_WORDS,
)

# Guide file names under docs/guides (H owns the files; the View one is new).
GUIDE_RUN = "step2_round_run.md"
GUIDE_VIEW = "step2_curation_progress.md"  # no step2_view.md
GUIDE_PROGRESS = "step2_curation_progress.md"
GUIDE_MODEL = "step2_model.md"

from napari_epicycle.ui_shell._guides import guides_dir as _guides_dir
_GUIDES_DIR = _guides_dir()
_ACCENT = PANEL_GOLD

#: The status table: stored key, word, colour, mdi6 glyph.
STATUS_ROWS = tuple(
    (key, STATUS_WORDS[key].capitalize(), STATUS_COLOURS[key],
     STATUS_ICONS[key]) for key in STATUS_ORDER)
BUCKETS = ("lateral", "axial")
BUCKET_TITLES = {"lateral": CUR_VIEW_LATERAL, "axial": CUR_VIEW_AXIAL}

#: The right column's scroll floor (the Model Refinement number, ui.py:189).
SCROLL_MIN_H = 80
#: a short dock lets the snapped viewport shrink to the switch + the counts
#: (the Use row scrolls) rather than show a half-cut combo
SCROLL_MIN_H_SHORT = 48
FOOT_BAR_H = PANEL_PROGRESS_H     # the one bar height (8 px in short mode)
FOOT_ICON_PX = 14
TALL_MODE_MIN_PX = 700   # below this the section descs are tooltips
_QWIDGETSIZE_MAX = 16777215   # Qt's "no maximum"
_ZERO_COLOUR = PANEL_ZERO_COLOR
_MONO = MONO_FONT
FOOT_OBJECT = "step2CurationFoot"
#: a snapped viewport keeps this many px below the block it ends on: the
#: viewport comes out ~2 px shorter than the cap on a real (hi-DPI) screen,
#: and a descender must not be cut
SNAP_SLACK = 2


def snap_cap(scroll, cut: int) -> int:
    """The ``maximumHeight`` that makes *scroll*'s VIEWPORT end at *cut*: the
    cut plus whatever the area's own chrome takes (a stylesheet's border or
    padding -- the viewport comes out 2-4 px shorter than the area on a real
    napari window) plus ``SNAP_SLACK``."""
    chrome = max(int(scroll.height()) - int(scroll.viewport().height()), 0)
    return int(cut) + chrome + SNAP_SLACK


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
_LOG_TAG_RE = re.compile(r"^\s*\[[A-Za-z0-9_\- ]+\]\s*")


def strip_log_tag(text) -> str:
    """``"[PATCH] Loading 'x' ..."`` -> ``"Loading 'x' ..."``: the busy words
    are the controller's log lines; the UI shows them without the tag."""
    return _LOG_TAG_RE.sub("", str(text or ""), count=1).strip()


def compose_stage_line(*, locked: bool = False, round_n: Optional[int] = None,
                       bucket: str = "lateral", patch: str = "",
                       n_selected: int = 0, n_slices: Optional[int] = None,
                       busy_text: Optional[str] = None) -> str:
    """The one Curation stage sentence.  Order of precedence:
    the training lock, a running busy phase, no patch, a view with no
    slices, then the ready line with the selection count."""
    bucket = str(bucket or "lateral")
    if locked:
        n = "?" if round_n is None else int(round_n)
        return CUR_STAGE_LOCKED.format(n=n, bucket=bucket)
    if busy_text and strip_log_tag(busy_text):
        return strip_log_tag(busy_text)
    if not patch:
        return CUR_STAGE_NO_PATCH
    if n_slices is not None and int(n_slices) == 0:
        return CUR_STAGE_NO_SLICES
    k = max(0, int(n_selected or 0))
    if k == 0:
        sel = CUR_STAGE_SEL_NONE
    elif k == 1:
        sel = CUR_STAGE_SEL_ONE
    else:
        sel = CUR_STAGE_SEL_MANY.format(k=k)
    return CUR_STAGE_READY.format(patch=patch, bucket=bucket, sel=sel)


def compose_progress_note(trained_rounds: Optional[tuple],
                          next_lateral: Any, next_axial: Any,
                          bucket: str = "lateral") -> str:
    """The dim sentence under the progress table.  ``trained_rounds`` is
    ``(first, last)`` of the ACTIVE bucket's rounds already trained (``bucket``
    is ``"lateral"`` or ``"axial"``), or ``None``."""
    if trained_rounds:
        a, b = trained_rounds
        span = (f"{bucket} round {a}" if a == b
                else f"{bucket} rounds {a}–{b}")
        return CUR_PROGRESS_NOTE.format(trained=span, lat=next_lateral,
                                        ax=next_axial)
    return CUR_PROGRESS_NOTE_NO_ROUNDS.format(lat=next_lateral, ax=next_axial)


def compose_inference_summary(preset: str = "", channels: str = "",
                              flow: Any = None, cellprob: Any = None,
                              diameter: Any = None) -> str:
    """``Balanced · membrane / nuclei · flow 0.4 · cellprob 0.0 · diameter
    30 px`` -- the grey line under the closed Inference disclosure."""
    parts = []
    if preset:
        parts.append(str(preset))
    if channels:
        parts.append(str(channels))
    if flow is not None:
        parts.append(f"flow {float(flow):g}")
    if cellprob is not None:
        parts.append(f"cellprob {float(cellprob):.1f}")
    if diameter is not None:
        parts.append("diameter auto" if int(diameter) == 0
                     else f"diameter {int(diameter)} px")
    return " · ".join(parts)


def _icon(name: str, colour: str):
    try:
        import qtawesome as qta
        return qta.icon(name, color=colour)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small shared pieces
# ---------------------------------------------------------------------------
def _section_header(parent: QWidget, title: str, guide: str = "",
                    guides_dir: Optional[Path] = None) -> QWidget:
    row = QWidget(parent)
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    label = QLabel(title, row)
    label.setStyleSheet(SECTION_HEADER_STYLE)
    row.title_label = label
    lay.addWidget(label)
    lay.addStretch(1)
    if guide:
        lay.addWidget(help_button(
            row, guide_path=str((guides_dir or _GUIDES_DIR) / guide),
            title=title, accent=_ACCENT, tooltip=f"Help: {title}"))
    return row


def _desc(parent: QWidget, text: str) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setStyleSheet(PANEL_HELP_STYLE)
    lbl.setWordWrap(True)
    lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    return lbl


class SnapWatcher(QObject):
    """Calls ``callback()`` ONCE, on the next event-loop pass, whenever a
    watched widget is resized or its layout is re-done.  A block snap measures
    the content's geometry; the first measure (at a resize) is taken before
    the columns have finished laying out, so the snap runs again when they
    have.  The callback is a bound method (a lambda held only by PyQt can lose
    its closure when napari garbage-collects)."""

    _EVENTS = (QEvent.Resize, QEvent.LayoutRequest, QEvent.Show)

    def __init__(self, parent: QObject, callback):
        super().__init__(parent)
        self._callback = callback
        self._pending = False

    def watch(self, *widgets) -> None:
        for w in widgets:
            if w is not None:
                w.installEventFilter(self)

    def eventFilter(self, obj, ev):  # noqa: N802 (Qt API)
        if ev.type() in self._EVENTS and not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._fire)
        return False

    def _fire(self) -> None:
        self._pending = False
        try:
            self._callback()
        except RuntimeError:          # the watched widget was deleted
            pass


class ElidedLabel(QLabel):
    """A label whose text never overflows its column.  ``text()`` is always
    the FULL text (the controller and the tests read it); what is shown is the
    full text when it fits ``max_lines`` lines at the label's width, else ONE
    line elided in the middle with the full text as the tooltip.  With
    ``max_lines == 0`` it is an ordinary word-wrapped label.  Needs a
    horizontally ``Ignored`` size policy, as every label of a column has."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None,
                 max_lines: int = 0, elide=Qt.ElideMiddle):
        super().__init__(parent)
        self._full = ""
        self._max_lines = int(max_lines)
        self._elide = elide
        self._busy = False
        self._tip_ours = False
        self.setText(text)

    def text(self) -> str:  # noqa: D102
        return self._full

    def setText(self, text) -> None:  # noqa: N802 (Qt API)
        self._full = str(text or "")
        self._reflow()

    def set_max_lines(self, n: int) -> None:
        self._max_lines = int(n)
        self._reflow()

    def resizeEvent(self, event):  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._reflow()

    def _reflow(self) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            full = self._full
            elided = False
            if self._max_lines <= 0:
                self.setWordWrap(True)
                QLabel.setText(self, full)
            else:
                self.ensurePolished()
                fm = self.fontMetrics()
                # 2 px of safety: a glyph's last pixel must not meet the
                # viewport edge (the Active line's closing parenthesis)
                w = max(self.contentsRect().width() - 2, 10)
                if self._max_lines == 1:
                    fits = fm.horizontalAdvance(full) <= w
                    wrap = False
                else:
                    r = fm.boundingRect(
                        QRect(0, 0, w, 100000),
                        int(Qt.TextWordWrap | Qt.AlignLeft), full)
                    fits = (r.width() <= w and
                            r.height() <= self._max_lines * fm.lineSpacing()
                            + 1)
                    wrap = fits
                if fits:
                    self.setWordWrap(wrap)
                    QLabel.setText(self, full)
                else:
                    self.setWordWrap(False)
                    QLabel.setText(self, fm.elidedText(full, self._elide, w))
                    elided = True
            if elided:
                self.setToolTip(full)
                self._tip_ours = True
            elif self._tip_ours:
                self.setToolTip("")
                self._tip_ours = False
        finally:
            self._busy = False


def _adopt(widget: QWidget, parent: QWidget) -> None:
    """Re-parent a received widget under ``parent`` (a layout add does it
    too; this is for the hidden ones that go in no layout)."""
    if widget.parent() is not parent:
        widget.setParent(parent)


class _EnabledMirror(QObject):
    """Calls ``callback()`` whenever one of the watched widgets' enabled
    state changes (an event filter: no lambda, no per-widget signal)."""

    def __init__(self, parent: QObject, callback):
        super().__init__(parent)
        self._callback = callback

    def eventFilter(self, obj, ev):  # noqa: N802 (Qt API)
        if ev.type() == QEvent.EnabledChange:
            try:
                self._callback()
            except Exception:
                pass
        return False


# ---------------------------------------------------------------------------
# Run row
# ---------------------------------------------------------------------------
class RunRow(QWidget):
    """``Run:`` picker . New Run . Manage ▾ . ``?``.  The three old buttons
    stay (hidden) so the controller's connections stand; the menu's actions
    ``.click()`` them and mirror their enabled state."""

    def __init__(self, picker, new_run_btn, manage_btn, import_run_btn,
                 import_external_btn, help_btn, parent=None):
        super().__init__(parent)
        self.picker = picker
        self.new_run_btn = new_run_btn
        self.hidden_buttons = (manage_btn, import_run_btn, import_external_btn)
        self._manage_btn = manage_btn
        self._import_run_btn = import_run_btn
        self._import_external_btn = import_external_btn

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        label = QLabel(CUR_RUN_LABEL, self)
        label.setStyleSheet(SECTION_HEADER_STYLE)
        lay.addWidget(label)
        picker.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        # Step 3's picker: the width follows the column, never the text
        picker.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        picker.setMinimumContentsLength(10)
        lay.addWidget(picker, 1)
        new_run_btn.setText(CUR_NEW_RUN_BTN)
        new_run_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        lay.addWidget(new_run_btn)

        for b in self.hidden_buttons:
            _adopt(b, self)
            b.setVisible(False)

        self.manage_menu_button = QToolButton(self)
        self.manage_menu_button.setText(f"{CUR_MANAGE_BTN} ▾")
        self.manage_menu_button.setPopupMode(QToolButton.InstantPopup)
        # Step 3's Manage button: the secondary look, the amber cog
        self.manage_menu_button.setStyleSheet(SECONDARY_TOOLBTN_STYLE)
        _cog = _icon("mdi6.cog-outline", PANEL_GOLD)
        if _cog is not None:
            self.manage_menu_button.setIcon(_cog)
            self.manage_menu_button.setIconSize(QSize(FOOT_ICON_PX,
                                                      FOOT_ICON_PX))
            self.manage_menu_button.setToolButtonStyle(
                Qt.ToolButtonTextBesideIcon)
        self.menu = QMenu(self.manage_menu_button)
        self.act_manage = self.menu.addAction(CUR_MANAGE_RUNS_ACTION)
        self.menu.addSeparator()
        self.act_caption = self.menu.addAction(CUR_MANAGE_CAPTION)
        self.act_caption.setEnabled(False)
        self.act_import_run = self.menu.addAction(CUR_IMPORT_PAIRS_ACTION)
        self.act_import_external = self.menu.addAction(CUR_IMPORT_EXTERNAL_ACTION)
        self.act_manage.triggered.connect(self._click_manage)
        self.act_import_run.triggered.connect(self._click_import_run)
        self.act_import_external.triggered.connect(self._click_import_external)
        self.menu.aboutToShow.connect(self.sync_enabled)
        self.manage_menu_button.setMenu(self.menu)
        lay.addWidget(self.manage_menu_button)
        if help_btn is not None:
            lay.addWidget(help_btn)

        self._mirror = _EnabledMirror(self, self.sync_enabled)
        for b in self.hidden_buttons:
            b.installEventFilter(self._mirror)
        self.sync_enabled()

    # bound methods (no lambdas held by PyQt)
    def _click_manage(self, *_):
        self._manage_btn.click()

    def _click_import_run(self, *_):
        self._import_run_btn.click()

    def _click_import_external(self, *_):
        self._import_external_btn.click()

    def sync_enabled(self) -> None:
        """Mirror the hidden buttons' enabled state onto the menu."""
        self.act_manage.setEnabled(self._manage_btn.isEnabled())
        self.act_import_run.setEnabled(self._import_run_btn.isEnabled())
        self.act_import_external.setEnabled(
            self._import_external_btn.isEnabled())
        self.manage_menu_button.setEnabled(any(
            b.isEnabled() for b in self.hidden_buttons))


def build_run_row(picker, new_run_btn, manage_btn, import_run_btn,
                  import_external_btn, help_btn=None, parent=None) -> RunRow:
    return RunRow(picker, new_run_btn, manage_btn, import_run_btn,
                  import_external_btn, help_btn, parent)


# ---------------------------------------------------------------------------
# View switch
# ---------------------------------------------------------------------------
_SEG_STYLE = PANEL_SEGMENT_STYLE


class ViewSwitch(QWidget):
    """Two-segment proxy over the two hidden radios (D17).  The radios keep
    the state (the controller and the lock list read them); a click on a
    segment ``.click()``s the radio, ``toggled`` mirrors back, the enabled
    state follows."""

    def __init__(self, lateral_radio, axial_radio, parent=None):
        super().__init__(parent)
        self.radios = {"lateral": lateral_radio, "axial": axial_radio}
        for r in self.radios.values():
            _adopt(r, self)
            r.setVisible(False)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.buttons: Dict[str, QToolButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for key, left in (("lateral", True), ("axial", False)):
            b = QToolButton(self)
            b.setText(BUCKET_TITLES[key])
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setStyleSheet(_SEG_STYLE + (
                "QToolButton { border-top-left-radius: 4px;"
                " border-bottom-left-radius: 4px; }" if left else
                "QToolButton { border-top-right-radius: 4px;"
                " border-bottom-right-radius: 4px; border-left: none; }"))
            self._group.addButton(b)
            lay.addWidget(b, 1)
            self.buttons[key] = b
        self.buttons["lateral"].clicked.connect(self._on_lateral_clicked)
        self.buttons["axial"].clicked.connect(self._on_axial_clicked)
        self._mirror = _EnabledMirror(self, self.sync)
        for key, r in self.radios.items():
            r.toggled.connect(self._on_radio_toggled)
            r.installEventFilter(self._mirror)
        self.sync()

    def _on_lateral_clicked(self, *_):
        self.radios["lateral"].click()
        self.sync()

    def _on_axial_clicked(self, *_):
        self.radios["axial"].click()
        self.sync()

    def _on_radio_toggled(self, *_):
        self.sync()

    def sync(self) -> None:
        """Bring both segments to the radios' checked + enabled state."""
        for key, b in self.buttons.items():
            r = self.radios[key]
            b.setEnabled(r.isEnabled())
            if r.isChecked() and not b.isChecked():
                b.setChecked(True)  # the exclusive group unchecks the other
        if not any(r.isChecked() for r in self.radios.values()):
            self._group.setExclusive(False)
            for b in self.buttons.values():
                b.setChecked(False)
            self._group.setExclusive(True)

    def active_bucket(self) -> str:
        return "axial" if self.radios["axial"].isChecked() else "lateral"


# ---------------------------------------------------------------------------
# Curation progress table
# ---------------------------------------------------------------------------
class ProgressTable(QWidget):
    """The 5 x 2 Lateral | Axial table.  Built once; the controller seam
    (c) only calls :meth:`refresh`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName(PANEL_CARD_OBJECT)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(PANEL_CARD_STYLE)
        grid = QGridLayout()
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(0, 1)
        self.grid = grid
        self.header_labels: Dict[str, QLabel] = {}
        for col, bucket in enumerate(BUCKETS, start=1):
            h = QLabel(BUCKET_TITLES[bucket], self)
            h.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(h, 0, col)
            self.header_labels[bucket] = h
        self.row_labels: Dict[str, QWidget] = {}
        self.cells: Dict[tuple, QLabel] = {}
        for r, (key, word, colour, glyph) in enumerate(STATUS_ROWS, start=1):
            cell = QWidget(self)
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(5)
            ic = QLabel(cell)
            ic.setFixedSize(14, 14)
            icon = _icon(glyph, colour)
            if icon is not None:
                ic.setPixmap(icon.pixmap(14, 14))
            else:
                ic.setText("●")
                ic.setStyleSheet(f"color: {colour};")
            wd = QLabel(word, cell)
            wd.setStyleSheet(f"color: {colour}; font-size: 11px;"
                             " background: transparent;")
            cl.addWidget(ic)
            cl.addWidget(wd, 1)
            grid.addWidget(cell, r, 0)
            self.row_labels[key] = wd
            for col, bucket in enumerate(BUCKETS, start=1):
                n = QLabel("0", self)
                n.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(n, r, col)
                self.cells[(key, bucket)] = n
        self.note_label = _desc(self, "")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        # the full table (grid + note) sits in one widget so the compact
        # (short-dock) form can swap it for the one-line summary
        self.table_body = QWidget(self)
        body = QVBoxLayout(self.table_body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(4)
        body.addLayout(grid)
        note_wrap = QHBoxLayout()
        note_wrap.setContentsMargins(8, 0, 8, 6)
        note_wrap.addWidget(self.note_label)
        body.addLayout(note_wrap)
        outer.addWidget(self.table_body)
        # the compact form: icon + count per status of the ACTIVE bucket
        self.summary_row = QWidget(self)
        sh = QHBoxLayout(self.summary_row)
        sh.setContentsMargins(8, 4, 8, 4)
        sh.setSpacing(4)
        self.summary_counts: Dict[str, QLabel] = {}
        for key, word, colour, glyph in STATUS_ROWS:
            ic = QLabel(self.summary_row)
            ic.setFixedSize(14, 14)
            icon = _icon(glyph, colour)
            if icon is not None:
                ic.setPixmap(icon.pixmap(14, 14))
            else:
                ic.setText("●")
                ic.setStyleSheet(f"color: {colour};")
            ic.setToolTip(word)
            n = QLabel("0", self.summary_row)
            n.setToolTip(word)
            sh.addWidget(ic)
            sh.addWidget(n)
            sh.addSpacing(6)
            self.summary_counts[key] = n
        sh.addStretch(1)
        outer.addWidget(self.summary_row)
        self._compact = False
        self.summary_row.setVisible(False)
        self.refresh({}, "lateral", "")

    def set_compact(self, compact: bool) -> None:
        """Short dock: one line of icon + count per status (the active
        bucket) instead of the 5 x 2 table; the table returns in tall mode."""
        self._compact = bool(compact)
        self.table_body.setVisible(not self._compact)
        self.summary_row.setVisible(self._compact)

    def summary_text(self) -> str:
        """The compact line as plain text, ``Predicted 62 | Curating 2``
        (what the icons say), for tests."""
        return " | ".join(
            f"{word} {self.summary_counts[key].text()}"
            for key, word, *_ in STATUS_ROWS)

    def refresh(self, counts: Mapping[str, Mapping[str, int]],
                active_bucket: str = "lateral", note: str = "") -> None:
        """``counts[bucket][status]`` (stored status keys; a missing one is
        0).  The active bucket's column is bold + bright; zeros are grey."""
        active = active_bucket if active_bucket in BUCKETS else "lateral"
        for bucket in BUCKETS:
            on = bucket == active
            by = (counts or {}).get(bucket) or {}
            self.header_labels[bucket].setStyleSheet(
                f"color: {PANEL_BRIGHT_COLOR if on else PANEL_HELP_COLOR};"
                " font-size: 10px;"
                f" font-weight: {'bold' if on else 'normal'};"
                " background: transparent;")
            for key, *_ in STATUS_ROWS:
                try:
                    n = int(by.get(key, 0) or 0)
                except (TypeError, ValueError):
                    n = 0
                lbl = self.cells[(key, bucket)]
                lbl.setText(str(n))
                colour = _ZERO_COLOUR if n == 0 else (
                    PANEL_BRIGHT_COLOR if on else PANEL_ROW_LABEL_COLOR)
                lbl.setStyleSheet(
                    f"color: {colour}; font-size: 11px; {_MONO}"
                    f" font-weight: {'bold' if on else 'normal'};"
                    " background: transparent;")
        self.note_label.setText(note or "")
        self.note_label.setVisible(bool(note))
        by = (counts or {}).get(active) or {}
        for key, *_ in STATUS_ROWS:
            try:
                n = int(by.get(key, 0) or 0)
            except (TypeError, ValueError):
                n = 0
            lbl = self.summary_counts[key]
            lbl.setText(str(n))
            lbl.setStyleSheet(
                f"color: {_ZERO_COLOUR if n == 0 else PANEL_BRIGHT_COLOR};"
                f" font-size: 11px; {_MONO} font-weight: bold;"
                " background: transparent;")
        # the hover is the table as text, read back from its (sanitised) cells
        lines = [f"{word}: {BUCKET_TITLES['lateral']} "
                 f"{self.cells[(key, 'lateral')].text()}"
                 f" · {BUCKET_TITLES['axial']} "
                 f"{self.cells[(key, 'axial')].text()}"
                 for key, word, *_ in STATUS_ROWS]
        self.summary_row.setToolTip(
            "\n".join(lines + ([note] if note else [])))


def build_progress_table(parent=None) -> ProgressTable:
    return ProgressTable(parent)


# ---------------------------------------------------------------------------
# Model block
# ---------------------------------------------------------------------------
class ModelBlock(QWidget):
    """Use / Active / (external path row) / Pretrained / use-newest / the
    two channel combos / the closed Inference disclosure with its grey
    summary.  Receives every widget; creates only labels and rows."""

    def __init__(self, source_combo, active_label, path_edit, browse_btn,
                 pretrained_combo, auto_update_cb, channel1_combo,
                 channel2_combo, inference_collapsible, parent=None):
        super().__init__(parent)
        self.source_combo = source_combo
        self.active_label = active_label
        self.path_edit = path_edit
        self.browse_btn = browse_btn
        self.pretrained_combo = pretrained_combo
        self.inference_collapsible = inference_collapsible
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        self._extra: list = []     # hidden by the compact (short-dock) form
        self._compact = False

        def row(label_text, widget):
            w = QWidget(self)
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            h.addWidget(panel_form_label(label_text, parent=w))
            widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            h.addWidget(widget, 1)
            return w

        self.use_row = row(CUR_MODEL_USE_LABEL, source_combo)
        lay.addWidget(self.use_row)
        active_label.setStyleSheet(PANEL_ROW_LABEL_STYLE)
        # an external model's path never overflows: two lines at most, else
        # one line elided in the middle with the full text on hover
        if hasattr(active_label, "set_max_lines"):
            active_label.set_max_lines(2)
        else:
            active_label.setWordWrap(True)
        active_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(active_label)

        self.path_row = QWidget(self)
        ph = QHBoxLayout(self.path_row)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.setSpacing(6)
        path_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        ph.addWidget(path_edit, 1)
        ph.addWidget(browse_btn)
        lay.addWidget(self.path_row)

        pre_row = row(CUR_MODEL_PRETRAINED_LABEL, pretrained_combo)
        lay.addWidget(pre_row)
        self.pretrained_note = _desc(self, CUR_MODEL_PRETRAINED_NOTE)
        lay.addWidget(self.pretrained_note)
        auto_update_cb.setText(CUR_MODEL_AUTO_UPDATE)
        lay.addWidget(auto_update_cb)
        self._extra += [pre_row, self.pretrained_note, auto_update_cb]

        for text, combo in ((CUR_MODEL_CHANNEL1_LABEL, channel1_combo),
                            (CUR_MODEL_CHANNEL2_LABEL, channel2_combo)):
            lbl = QLabel(text, self)
            lbl.setStyleSheet(PANEL_ROW_LABEL_STYLE)
            lay.addWidget(lbl)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            lay.addWidget(combo)
            self._extra += [lbl, combo]
        lay.addWidget(inference_collapsible)
        self.inference_summary = _desc(self, "")
        lay.addWidget(self.inference_summary)
        self._extra += [inference_collapsible]

        source_combo.currentTextChanged.connect(self._on_source_text)
        self.sync()

    def _on_source_text(self, *_):
        self.sync()

    def attach_use_help(self, button: QWidget) -> None:
        """The Model header's ``?`` beside the Use combo (short mode drops
        the header row; the guide stays one click away)."""
        self.use_row.layout().addWidget(button)

    def snap_rows(self) -> list:
        """The rows a short viewport may end under (Use, Active, the path)."""
        return [self.use_row, self.active_label, self.path_row]

    def is_external(self) -> bool:
        return "external" in str(self.source_combo.currentText()).lower()

    def sync(self) -> None:
        """Show the path row only for External; set the Pretrained note and
        its hover reason from the current ``Use:`` item."""
        ext = self.is_external()
        self.path_row.setVisible(ext and not self._compact)
        note = CUR_MODEL_EXTERNAL_NOTE if ext else CUR_MODEL_PRETRAINED_NOTE
        self.pretrained_note.setText(note)
        self.pretrained_combo.setToolTip(note)
        if ext:
            self.path_edit.setToolTip(str(self.path_edit.text()))

    def set_inference_summary(self, text: str) -> None:
        self.inference_summary.setText(text or "")
        self.inference_summary.setVisible(bool(text) and not self._compact)

    def set_compact(self, compact: bool) -> None:
        """Short dock: only ``Use`` and ``Active`` stay; Pretrained, the
        newest-model checkbox, the channels and Inference return in tall
        mode (the owner's Q1(a))."""
        self._compact = bool(compact)
        for w in self._extra:
            w.setVisible(not self._compact)
        self.inference_summary.setVisible(
            bool(self.inference_summary.text()) and not self._compact)
        self.sync()


def build_model_block(source_combo, active_label, path_edit, browse_btn,
                      pretrained_combo, auto_update_cb, channel1_combo,
                      channel2_combo, inference_collapsible,
                      parent=None) -> ModelBlock:
    return ModelBlock(source_combo, active_label, path_edit, browse_btn,
                      pretrained_combo, auto_update_cb, channel1_combo,
                      channel2_combo, inference_collapsible, parent)


# ---------------------------------------------------------------------------
# Pinned foot
# ---------------------------------------------------------------------------
class PinnedFoot(QWidget):
    """Open 2D annotator (the ONE primary) . Accept / Finalize accepted /
    Discard / Revert in two rows . the busy bar . the stage line.  A sibling
    of the scroll area, never inside it (the Model Refinement shape)."""

    def __init__(self, open_btn, accept_btn, finalize_btn, discard_btn,
                 revert_btn, busy_bar, stage_label=None, parent=None):
        super().__init__(parent)
        self.open_btn = open_btn
        self.buttons = (accept_btn, finalize_btn, discard_btn, revert_btn)
        self.busy_bar = busy_bar
        # Step 3's feet carry a rule above their buttons
        self.setObjectName(FOOT_OBJECT)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(panel_foot_style(FOOT_OBJECT))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(3)

        # the ONE primary: Step 3's translucent amber (D-A)
        open_btn.setText(CUR_BTN_OPEN)
        open_btn.setStyleSheet(PRIMARY_BTN_STYLE)
        open_btn.setProperty("epiPrimary", True)
        open_btn.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        icon = _icon("mdi6.dock-window", PANEL_BRIGHT_COLOR)
        if icon is not None:
            open_btn.setIcon(icon)
        lay.addWidget(open_btn)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        for i, (btn, text, glyph, colour) in enumerate((
                (accept_btn, CUR_BTN_ACCEPT, STATUS_ICONS["accepted"],
                 STATUS_COLOURS["accepted"]),
                (finalize_btn, CUR_BTN_FINALIZE, STATUS_ICONS["staged"],
                 STATUS_COLOURS["staged"]),
                (discard_btn, CUR_BTN_DISCARD, "mdi6.trash-can-outline",
                 PANEL_ERROR_TEXT_COLOR),
                (revert_btn, CUR_BTN_REVERT, "mdi6.undo-variant",
                 PANEL_ROW_LABEL_COLOR))):
            btn.setText(text)
            # one kind of secondary, as Step 3's foot (D-A)
            btn.setStyleSheet(SECONDARY_BTN_STYLE)
            btn.setProperty("epiPrimary", False)
            btn.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            ic = _icon(glyph, colour)
            if ic is not None:
                btn.setIcon(ic)
                btn.setIconSize(QSize(FOOT_ICON_PX, FOOT_ICON_PX))
            grid.addWidget(btn, i // 2, i % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        busy_bar.setStyleSheet(progress_bar_style(PANEL_GOLD, FOOT_BAR_H))
        busy_bar.setFixedHeight(FOOT_BAR_H)
        busy_bar.setTextVisible(False)
        # the bar keeps the full foot width always; the busy words are its
        # tooltip and the stage line's (``busy_label`` is a hidden stub)
        lay.addWidget(busy_bar)
        self.busy_label = QLabel("", self)
        self.busy_label.setVisible(False)

        if stage_label is None:
            stage_label = QLabel("", self)
        self.stage_label = stage_label
        stage_label.setStyleSheet(MONO_LABEL_STYLE)   # as Step 3's feet
        stage_label.setWordWrap(True)
        stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        stage_label.setMinimumHeight(13)
        stage_label.setMaximumHeight(28)
        lay.addWidget(stage_label)

    def set_stage(self, text: str) -> None:
        self.stage_label.setText(text or "")
        self.stage_label.setToolTip(text or "")

    def set_short(self, short: bool) -> None:
        """A short dock: tighter gaps and the 8 px bar (the room buys the
        View / Use rows their place); the tall foot is as built."""
        short = bool(short)
        lay = self.layout()
        lay.setSpacing(2 if short else 3)
        lay.setContentsMargins(0, 2 if short else 4, 0, 0)
        h = PANEL_PROGRESS_H_SHORT if short else PANEL_PROGRESS_H
        self.busy_bar.setStyleSheet(progress_bar_style(PANEL_GOLD, h))
        self.busy_bar.setFixedHeight(h)
        fn = getattr(self.stage_label, "set_max_lines", None)
        if fn is not None:
            fn(1 if short else 0)

    def set_busy(self, text: Optional[str]) -> None:
        """The busy words (log tag stripped) as the bar's tooltip; the
        stage line carries them on screen."""
        self.busy_bar.setToolTip(strip_log_tag(text) if text else "")


def build_pinned_foot(open_btn, accept_btn, finalize_btn, discard_btn,
                      revert_btn, busy_bar, stage_label=None,
                      parent=None) -> PinnedFoot:
    return PinnedFoot(open_btn, accept_btn, finalize_btn, discard_btn,
                      revert_btn, busy_bar, stage_label, parent)


# ---------------------------------------------------------------------------
# The right column
# ---------------------------------------------------------------------------
_THIN_SCROLL = (
    "QScrollArea { background: transparent; border: none; }"
    + SCROLLBAR_STYLE
    + "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical"
    " { background: transparent; }"
)


class RightColumn(QWidget):
    """View switch, progress table, Model block (all in the scroll area)
    over the pinned foot."""

    def __init__(self, view_switch, progress_table, model_block, foot,
                 parent=None, guides_dir: Optional[Path] = None):
        super().__init__(parent)
        self.view_switch = view_switch
        self.progress_table = progress_table
        self.model_block = model_block
        self.foot = foot
        outer = QVBoxLayout(self)
        # Step 3's 8 px between the handle and the right column
        outer.setContentsMargins(PANEL_RIGHT_INNER_MARGIN, 0, 0, 0)
        outer.setSpacing(0)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.scroll.setStyleSheet(_THIN_SCROLL)
        self.scroll.setMinimumHeight(SCROLL_MIN_H)
        content = QWidget()
        self.content = content
        c = QVBoxLayout(content)
        c.setContentsMargins(0, 0, 4, 6)
        c.setSpacing(6)
        self._descs = []
        h = _section_header(content, CUR_VIEW_HEADER, GUIDE_VIEW, guides_dir)
        c.addWidget(h)
        d = _desc(content, CUR_VIEW_DESC)
        self._descs.append((h, d))
        c.addWidget(d)
        # short mode drops the View header row; its ``?`` sits beside the
        # switch (hidden in tall mode, where the header carries it)
        self.switch_row = QWidget(content)
        sw = QHBoxLayout(self.switch_row)
        sw.setContentsMargins(0, 0, 0, 0)
        sw.setSpacing(4)
        sw.addWidget(view_switch, 1)
        self.view_help = help_button(
            self.switch_row,
            guide_path=str((guides_dir or _GUIDES_DIR) / GUIDE_VIEW),
            title=CUR_VIEW_HEADER, accent=_ACCENT,
            tooltip=f"Help: {CUR_VIEW_HEADER}")
        self.view_help.setVisible(False)
        sw.addWidget(self.view_help)
        c.addWidget(self.switch_row)
        c.addSpacing(4)
        self._gap_items = [c.itemAt(c.count() - 1)]
        h = _section_header(content, CUR_PROGRESS_HEADER, GUIDE_PROGRESS,
                            guides_dir)
        c.addWidget(h)
        d = _desc(content, CUR_PROGRESS_DESC)
        self._descs.append((h, d))
        c.addWidget(d)
        c.addWidget(progress_table)
        c.addSpacing(4)
        self._gap_items.append(c.itemAt(c.count() - 1))
        h = _section_header(content, CUR_MODEL_HEADER, GUIDE_MODEL, guides_dir)
        c.addWidget(h)
        d = _desc(content, CUR_MODEL_DESC)
        self._descs.append((h, d))
        c.addWidget(d)
        c.addWidget(model_block)
        # the same for the Model header: its ``?`` joins the Use row
        self.model_help = help_button(
            model_block,
            guide_path=str((guides_dir or _GUIDES_DIR) / GUIDE_MODEL),
            title=CUR_MODEL_HEADER, accent=_ACCENT,
            tooltip=f"Help: {CUR_MODEL_HEADER}")
        self.model_help.setVisible(False)
        if hasattr(model_block, "attach_use_help"):
            model_block.attach_use_help(self.model_help)
        self._content_layout = c
        c.addStretch(1)
        self.scroll.setWidget(content)
        outer.addWidget(self.scroll, 1)
        # a stretch that takes the room a snapped (shorter) scroll area
        # leaves, so the foot stays pinned to the column's bottom
        outer.addStretch(0)
        outer.addWidget(foot, 0)
        self._short = False
        self._snap_watch = SnapWatcher(self, self.snap_viewport)
        self._snap_watch.watch(foot, content, self.scroll.viewport())
        #: the blocks of the scroll content, top to bottom: the viewport is
        #: cut at the bottom of the last one that fits whole (short mode)
        self._blocks = [w for pair in self._descs for w in pair] + [
            self.switch_row, progress_table] + list(
            getattr(model_block, "snap_rows", lambda: [model_block])())

    def snap_viewport(self) -> None:
        """Short dock: end the scroll viewport at a block boundary, so the
        first screenful never shows a half-cut header or row under the pinned
        foot (the scroll bar still reaches the rest).  Tall mode: no cap."""
        if not self._short:
            self.scroll.setMaximumHeight(_QWIDGETSIZE_MAX)
            return
        lay = self.content.layout()
        if lay is not None:
            lay.activate()
        room = self.height() - self.foot.height()
        bottoms = sorted(
            w.mapTo(self.content, QPoint(0, w.height())).y()
            for w in self._blocks
            if not w.isHidden() and w.height() > 0)
        chrome = snap_cap(self.scroll, 0) - SNAP_SLACK
        fits = [b for b in bottoms
                if self.scroll.minimumHeight() <= b
                and b + chrome <= room]
        self.scroll.setMaximumHeight(
            snap_cap(self.scroll, max(fits)) if fits else _QWIDGETSIZE_MAX)

    def resizeEvent(self, event):  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        if self._short:
            QTimer.singleShot(0, self.snap_viewport)

    def showEvent(self, event):  # noqa: N802 (Qt API)
        super().showEvent(event)
        if self._short:
            QTimer.singleShot(0, self.snap_viewport)


def build_right_column(view_switch, progress_table, model_block, foot,
                       parent=None, guides_dir=None) -> RightColumn:
    return RightColumn(view_switch, progress_table, model_block, foot,
                       parent, guides_dir)


def set_short_mode(pairs, short: bool) -> None:
    """In short mode (dock under ``TALL_MODE_MIN_PX``) the
    ``?`` descs are tooltips of their headers; in tall mode they show.
    ``pairs`` is ``[(header_row, desc_label), ...]``."""
    for header, desc in pairs:
        label = getattr(header, "title_label", header)
        desc.setVisible(not short)
        label.setToolTip(desc.text() if short else "")


def _right_set_short(self, short: bool) -> None:
    set_short_mode(self._descs, short)
    # the compact right column (Q1(a)): a one-line progress summary and only
    # Use + Active in the Model block; the full table and block in tall mode
    for part in (self.progress_table, self.model_block):
        fn = getattr(part, "set_compact", None)
        if fn is not None:
            fn(short)
    # the one-line summary sits right under the View switch: the Curation
    # progress header is dropped in short mode (its ``?`` is the View
    # header's -- one guide -- and the table is the summary's hover); so are
    # the View and Model headers (their ``?`` moved beside the switch / the
    # Use combo)
    for i in (0, 1, 2):
        self._descs[i][0].setVisible(not short)
    self.view_help.setVisible(bool(short))
    self.model_help.setVisible(bool(short))
    self.scroll.setMinimumHeight(SCROLL_MIN_H_SHORT if short
                                 else SCROLL_MIN_H)
    self._content_layout.setSpacing(3 if short else 6)
    for it in self._gap_items:           # the 4 px gaps between the blocks
        it.changeSize(0, 0 if short else 4, QSizePolicy.Minimum,
                      QSizePolicy.Fixed)
    self._content_layout.invalidate()
    fn = getattr(self.foot, "set_short", None)
    if fn is not None:
        fn(short)
    self._short = bool(short)
    self.snap_viewport()
    QTimer.singleShot(0, self.snap_viewport)


RightColumn.set_short_mode = _right_set_short


def enclosing_viewport_height(widget) -> int:
    """The height of the scroll viewport that holds ``widget`` (the shell's
    ``widgetResizable`` ``QScrollArea``); 0 when there is none.  A widget
    inside such an area is as tall as max(viewport, its minimum hint), so its
    own ``height()`` cannot say how much room the dock really offers."""
    from qtpy.QtWidgets import QScrollArea
    w = widget.parentWidget() if widget is not None else None
    while w is not None:
        if isinstance(w, QScrollArea):
            return int(w.viewport().height())
        w = w.parentWidget()
    return 0


def enclosing_scroll_area(widget):
    from qtpy.QtWidgets import QScrollArea
    w = widget.parentWidget() if widget is not None else None
    while w is not None:
        if isinstance(w, QScrollArea):
            return w
        w = w.parentWidget()
    return None


class CompactSplitter(QSplitter):
    """A splitter that asks for no more height than its panes' minimums
    (``fusion_reconstruction/ui.py`` ``_CompactSplitter``): the lists' and
    the scroll area's size hints would otherwise grow the whole section."""

    def sizeHint(self):  # noqa: N802 (Qt API)
        return QSize(super().sizeHint().width(),
                     self.minimumSizeHint().height())

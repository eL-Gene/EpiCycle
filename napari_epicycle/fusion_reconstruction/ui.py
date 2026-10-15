"""
Reconstruct + Curate 3D — Accordion UI with two-column layouts.

Phase K: Merged Predict + Reconstruct, QC Analysis, Model Refinement.
Five-section accordion: Session, Predict + Reconstruct, Curation (incl. QC),
Model Refinement, Diagnostics.
"""

import os

from qtpy.QtCore import QPropertyAnimation, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QBrush, QColor, QKeySequence, QPainter, QPalette
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from pathlib import Path as _Path

from ..ui_shell.theme import (
    Accordion, CANVAS_AREA_STYLE, COLLAPSIBLE_DURATION, COLLAPSIBLE_STYLE,
    DIM_LABEL_STYLE, FLOW_HINT_STYLE, FLOW_TITLE_STYLE, FLOW_SUBTITLE_STYLE,
    INPUT_STYLE, LIST_STYLE, LOG_OUTPUT_STYLE, MONO_LABEL_STYLE,
    PATCH_STATUS_COLORS, PATCH_STATUS_DOTS, PATCH_STATUS_ICONS,
    PRIMARY_BTN_STYLE, PROGRESS_BAR_STYLE, PROTECTED_STATUSES, SCROLLBAR_STYLE,
    SECONDARY_BTN_STYLE, SECTION_HEADER_STYLE, SEPARATOR_LINE_STYLE,
    SPLITTER_STYLE, STATUS_LABEL_STYLE, TAB_WIDGET_STYLE,
    apply_button_icon, apply_inner_group_style, help_button,
    PANEL_CHECKBOX_STYLE, PANEL_ERROR_STYLE, PANEL_WARN_STYLE,
    PANEL_HELP_STYLE, PANEL_INPUT_STYLE, PANEL_LABEL_COLUMN,
    PANEL_ROW_LABEL_STYLE, PANEL_GOLD, make_panel_card,
)
from ..ui_shell import theme as _pipe  # shared running words
from ..ui_shell import _recon_settings_text as _RT
from ..ui_shell._recon_settings_widgets import (
    ENGINE_NAME_STYLE, FORM_FIELD_W, LockLine, TickLabel, adv_row,
    collapsible, field, frm4, sub_header,
)
from ..ui_shell._stack_row_delegate import (
    MODE_NARROW, MODE_WIDE, ROLE_RECORD, ROLE_STATUS, ROLE_THUMB,
    ROLE_THUMB_TIP, StackListWidget, StackRowDelegate, list_cap_px,
)
from . import _step3_mr_text as _MRT
from . import _step3_pr_text as _PRT
from . import _step3_rows as _rows
from . import _step3_cell_width as _cw  # the width's two lines (R2b)


# Help-system constants — Step 3 panel `?` buttons resolve guide paths
# relative to this file so editable-install (`pip install -e .`) layouts
# work without hard-coded absolute paths.
from napari_epicycle.ui_shell._guides import guides_dir as _guides_dir
_GUIDES_DIR = _guides_dir()
_STEP3_ACCENT = "#d4a843"  # amber — matches Step 3 primary action accent.
#: the settings column's guide
_PR_SETTINGS_GUIDE = "step3_predict_reconstruct_settings.md"
#: the object name of the grey engine names on Step 3's collapsibles
_PR_ENGINE_OBJECT = "step3EngineName"
#: u-Segment3D's distance-transform methods (the hidden, inert flow-
#: consistency and indirect widgets keep today's item lists)
_DTFORM_OPTIONS = (
    "cellpose_improve", "edt", "fmm", "fmm_skel", "cellpose_skel",
)

# ---------------------------------------------------------------------------
# The Predict + Reconstruct layout (brick R3):
# Step 4's ② numbers, in amber
# ---------------------------------------------------------------------------
PR_SPLIT_LEFT_W = 240
PR_SPLIT_RIGHT_W = 330
PR_SPLIT_LEFT_MIN = 240
PR_SPLIT_RIGHT_MIN = 300
PR_SPLIT_HANDLE = 6
PR_LEFT_INNER_MARGIN = 6       # setContentsMargins(0, 0, 6, 0)
PR_RIGHT_INNER_MARGIN = 8      # setContentsMargins(8, 0, 0, 0)
PR_FOOT_BTN_MIN_W = 60
PR_PROGRESS_H = 14
#: what PANEL_WARN_STYLE / PANEL_ERROR_STYLE add to a line's height (5 px
#: padding and a 1 px border, top and bottom, plus slack)
PR_STATUS_BOX_PAD = 14
#: the two cards' guides (the files keep their names)
_PR_MODEL_GUIDE = "step3_predict_reconstruct_model_source.md"
_PR_PATCHES_GUIDE = "step3_predict_reconstruct_patch_state.md"
#: Step 3's object names for the shared lock line and the pinned foot
_PR_LOCK_OBJECT = "step3LockLine"
_PR_FOOT_OBJECT = "step3PrFoot"
PR_FOOT_STYLE = "QWidget#step3PrFoot { border-top: 1px solid #555555; }"
#: the lat | ax lines are NEUTRAL (amber is a warning).  No
#: padding of their own: the app's QWidget padding is what every other
#: label of the card has, so the lines start where "Active:" starts (V1)
PR_MODEL_LINE_STYLE = "color: #cccccc; font-size: 11px;"
#: the mockup's ``.mcard`` gap (between the card's own lines) and the
#: column's ``.col`` gap (between the cards and around the separator, whose
#: own margin is 4 px) -- STEP3_CONSISTENCY_MOCKUP_v2 board D (V1)
PR_CARD_GAP = 4
PR_COL_GAP = 6
PR_SEP_MARGIN = 4
#: the Patches rows on the shared painter in Step 3's colours: the
#: dark tick glyph on the gold fill (the accent is a delegate keyword, A3),
#: the amber selection tint and PANEL_WARN's text colour for the amber lines
PR_ROW_PALETTE = {
    "sel_bg": QColor(212, 168, 67, 64),
    "chk_mark": QColor("#1d1e24"),
    "warn": QColor("#e6c26a"),
}
#: the accent of Step 3's ticks (A3: filled, with the 35 % halo)
PR_ROW_ACCENT = "#d4a843"
#: the Patches list's floor (Step 4's 60 px) and its preferred height
PR_LIST_MIN_H = 60
PR_LIST_HINT_H = 120
#: the list property Step 3's lock sets (the painter dims the rows)
PR_LOCK_PROPERTY = "step3Locked"
#: Session's eye dims when the foot's View is grey (A2)
SESSION_EYE_LOCK_PROPERTY = "step3SessionEyeLocked"
#: the widest the Pipeline line's Resume link is drawn (it elides past it)
SESSION_RESUME_MAX_W = 220
#: the amber of Session's "without a voxel size" clause (PANEL_WARN's text)
SESSION_WARN_COLOUR = "#e6c26a"
#: Session's patch list takes its rows' height up to this -- four rows of
#: the 128 px picture (A3; it was 300 px at the 64 px picture, and a fixed
#: 120 px cap before V2) -- and never less than the floor below (Step 4's
#: floor for its Session list)
SESSION_LIST_MAX_H = list_cap_px(MODE_WIDE)
SESSION_LIST_MIN_H = 72
#: Session's guide (brick V2: the patch list's ``?``)
_SESSION_GUIDE = "step3_session.md"
#: the mockup's ``.linkish``
PR_LINK_STYLE = "color: #d4a843; font-size: 11px; text-decoration: underline;"
#: F9: the running / complete line's neutral-bright mono, never the dim grey
MR_INFO_STYLE = MONO_LABEL_STYLE.replace("#aaaaaa", "#cfcfcf")
#: the status line's kinds
_PR_STATUS_STYLES = {
    "neutral": MONO_LABEL_STYLE,
    #: F9: a running / complete line, the mockup's ``.pipe.ok``
    "info": MR_INFO_STYLE,
    "warn": PANEL_WARN_STYLE,
    "error": PANEL_ERROR_STYLE,
}

# ---------------------------------------------------------------------------
# The Model Refinement layout: the
# Predict + Reconstruct numbers above, plus its own few
# ---------------------------------------------------------------------------
_MR_LOCK_OBJECT = "step3MrLockLine"
_MR_FOOT_OBJECT = "step3MrFoot"
MR_FOOT_STYLE = "QWidget#step3MrFoot { border-top: 1px solid #555555; }"
_MR_LEFT_FOOT_OBJECT = "step3MrLeftFoot"
MR_LEFT_FOOT_STYLE = (
    "QWidget#step3MrLeftFoot { border-top: 1px solid #555555; }")
#: the room a scroll bar needs, kept beside the content so the rows and
#: chips never sit under it (M9)
MR_SCROLL_GUTTER = 8
#: Cancel never narrower than its word (M12)
MR_CANCEL_MIN_W = 96
#: the list floor: two rows (88-104 px on the 240 px column) show (M9)
MR_LIST_TWO_ROWS_H = 2 * 104 + 8
#: the four guides of the section (the files keep their names)
_MR_PATCHES_GUIDE = "step3_model_refinement_patch_selection.md"
_MR_FINE_TUNING_GUIDE = "step3_model_refinement_fine_tuning.md"
_MR_TRAINING_CONFIG_GUIDE = "step3_model_refinement_training_config.md"
#: All / None: secondary buttons at the secondary height, a narrow floor
MR_SMALL_BTN_W = 40
#: the loss plot's floor in the pinned foot
MR_LOSS_MIN_H = 140      # the user drags the height now (split brick)
MR_SPLIT_TOP_MIN_H = 120  # the settings pane can shrink this far
MR_SPLIT_OPEN_FRAC = 0.55  # the foot's share of the column when Loss opens
#: a result-card value
MR_CARD_VALUE_STYLE = "color: #dddddd; font-size: 11px; background: transparent;"
#: a result-card key (the Curation selection card's keys)
MR_CARD_KEY_STYLE = "color: #999999; font-size: 11px; background: transparent;"
#: the empty-state title (the Curation card's ``_NONE_STYLE``) and the
#: title of a real result
MR_CARD_EMPTY_TITLE_STYLE = (
    "color: #cccccc; font-size: 11px; background: transparent;")
MR_CARD_TITLE_STYLE = (
    "color: #eeeeee; font-size: 11px; font-weight: bold; "
    "background: transparent;")
MR_CARD_OBJECT = "mrResultCard"
#: the Model Refinement panel's own radio look (gold dot when checked); the
#: shared theme has none, and only this panel asks for it
MR_RADIO_STYLE = (
    "QRadioButton::indicator{width:11px;height:11px;"
    "border:1px solid #999999;border-radius:6px;"
    "background:rgba(0,0,0,0.25)} "
    "QRadioButton::indicator:checked{border:1px solid #d4a843;"
    "background:qradialgradient(cx:0.5,cy:0.5,radius:0.5,fx:0.5,fy:0.5,"
    "stop:0 #1d1e24,stop:0.35 #1d1e24,stop:0.45 #d4a843,stop:1 #d4a843)}")


def _mr_card_style(edge: str) -> str:
    """The Curation selection card's look (``_qc_plot_area._card_style``):
    a faint fill and border, a 3 px LEFT edge -- gold when a model exists,
    white 15 % when the card is empty."""
    return (
        f"QWidget#{MR_CARD_OBJECT} {{"
        "  background: rgba(255, 255, 255, 0.03);"
        "  border: 1px solid rgba(255, 255, 255, 0.10);"
        f"  border-left: 3px solid {edge};"
        "  border-radius: 4px;"
        "}")


def _pr_separator(parent) -> QLabel:
    sep = QLabel(parent)
    sep.setFixedHeight(1)
    sep.setStyleSheet(SEPARATOR_LINE_STYLE)
    return sep


def _bracket_words(text: str) -> list:
    """*text*'s words, a ``[...]`` time part kept as ONE word, so the first
    line never ends inside it (``[1:23:45 | 2:10:00]``)."""
    out = []
    held = None
    for word in text.split(" "):
        if held is not None:
            held.append(word)
            if word.endswith("]"):
                out.append(" ".join(held))
                held = None
        elif word.startswith("[") and not word.endswith("]"):
            held = [word]
        else:
            out.append(word)
    if held is not None:        # no closing bracket: the words as they were
        out.extend(held)
    return out


def _two_lines(lbl, text: str, slack: int = 0) -> str:
    """*text* cut to what two wrapped lines of *lbl* can hold (Step 4's
    ``SegmentStack._two_lines``), fitted at the label's CONTENTS width:
    the caller applies the style first, so a warn / error box's padding
    and border are already taken off (R3 audit MED-1)."""
    lbl.ensurePolished()        # the style sheet's font and box, not the default
    width = max(lbl.contentsRect().width() - slack, 200)
    fm = lbl.fontMetrics()
    if fm.horizontalAdvance(text) <= width:
        return text
    words = _bracket_words(text)
    first = ""
    k = 0
    for k, word in enumerate(words):
        trial = (first + " " + word) if first else word
        if first and fm.horizontalAdvance(trial) > width:
            break
        first = trial
    else:
        return text
    rest = " ".join(words[k:])
    return first + "\n" + fm.elidedText(rest, Qt.ElideRight, width)


class _ClickLabel(QLabel):
    """The mockup's ``.linkish`` line: a label that reports a left click
    (the "Tick all unprocessed" link, V1)."""

    clicked = Signal()

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)

    def mouseReleaseEvent(self, event):  # noqa: N802 (Qt API)
        if (event.button() == Qt.LeftButton and self.isEnabled()
                and self.rect().contains(event.pos())):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class _PatchList(StackListWidget):
    """The Patches list (V1): one highlighted row (for View), and Ctrl+A
    asks for every tickable row to be ticked instead of selecting them --
    a single-selection list would otherwise swallow the key."""

    select_all_requested = Signal()

    def sizeHint(self):  # noqa: N802 (Qt API)
        """A modest preferred height: the list has the column's stretch,
        so it fills what the section gives it, but its preferred height
        is what the section's scroll area asks for -- Qt's 192 px made the
        section scroll under a two-line lock line on an 848 px dock (V1)."""
        hint = super().sizeHint()
        return QSize(hint.width(), PR_LIST_HINT_H)

    def keyPressEvent(self, event):  # noqa: N802 (Qt API)
        if event.matches(QKeySequence.SelectAll):
            self.select_all_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _FitCollapsible(QCollapsible):
    """A ``QCollapsible`` that opens to the height its content needs AT ITS
    WIDTH.  superqt sizes the open body from ``sizeHint()``, which a wrapped
    label answers for a much wider box, so a body of word-wrapped lines
    opened clipped (Model Refinement look, B1: Extraction Settings and
    Advanced Parameters at the 240 px column)."""

    def _fitted_height(self):
        lay = self._content.layout()
        width = self._content.width() or self.width()
        if lay is None or width <= 0:
            return None
        if lay.hasHeightForWidth():
            return lay.totalHeightForWidth(width) + 10
        return lay.sizeHint().height() + 10

    def _expand_collapse(self, direction, animate=True, emit=True):
        super()._expand_collapse(direction, animate, emit)
        if direction != QPropertyAnimation.Direction.Forward:
            return
        height = self._fitted_height()
        if height is None:
            return
        if animate:
            self._animation.setEndValue(height)
        else:
            self._content.setMaximumHeight(height)


class _CompactSplitter(QSplitter):
    """A splitter that asks for no more height than its panes' minimums
    (the settings scroll area's size hint would otherwise grow the whole
    section), while still reporting those minimums so the foot is never
    squeezed below its buttons."""

    def sizeHint(self):  # noqa: N802 (Qt API)
        return QSize(super().sizeHint().width(),
                     self.minimumSizeHint().height())


class _GrowCollapsible(_FitCollapsible):
    """A collapsible whose OPEN body takes whatever height its column gives
    (the Loss curves, H4 / split brick): never animated, and the body is
    never capped -- superqt caps it at its size hint (and again on every
    child resize), which left an empty band under the plot."""

    def eventFilter(self, a0, a1):  # noqa: N802 (Qt API)
        return False

    def _expand_collapse(self, direction, animate=True, emit=True):
        if self._locked:
            return
        forward = direction == QPropertyAnimation.Direction.Forward
        self._toggle_btn.setIcon(
            self._expanded_icon if forward else self._collapsed_icon)
        self._toggle_btn.setChecked(forward)
        self._content.setMaximumHeight(16777215 if forward else 0)
        if emit:
            self.toggled.emit(forward)


def _grow_collapsible(title: str, engine: str = "", *,
                      engine_object_name: str = "") -> QCollapsible:
    """``collapsible`` (with the engine name on the header line) on the
    growing class: the Loss curves' open body takes the column's height."""
    c = _GrowCollapsible(title, collapsedIcon="▶")
    c.setDuration(0)
    c.layout().setContentsMargins(0, 0, 0, 0)
    c.layout().setSpacing(0)
    # superqt top-aligns the frame layout, which caps the body at its size
    # hint (the empty band under the plot): let the body take the height
    c.layout().setAlignment(Qt.AlignmentFlag(0))
    c._toggle_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    c.layout().setStretch(1, 1)
    if engine or engine_object_name:
        btn = c._toggle_btn
        eng = QLabel(engine, btn)
        if engine_object_name:
            eng.setObjectName(engine_object_name)
        eng.setStyleSheet(ENGINE_NAME_STYLE)
        eng.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        row = QHBoxLayout(btn)
        row.setSizeConstraint(QHBoxLayout.SetNoConstraint)
        row.setContentsMargins(0, 0, 4, 0)
        row.addStretch(1)
        row.addWidget(eng, 0, Qt.AlignRight | Qt.AlignVCenter)
        c._epicycle_engine_label = eng
    c._epicycle_collapsed_symbol = "▶"
    return c


def _mr_collapsible(title: str) -> QCollapsible:
    """``ui_shell._recon_settings_widgets.collapsible`` on the fitting
    class: the right-pointing arrow when closed, a tight pitch."""
    c = _FitCollapsible(title, collapsedIcon="▶")
    c.setDuration(COLLAPSIBLE_DURATION)
    c.layout().setContentsMargins(0, 0, 0, 0)
    c.layout().setSpacing(0)
    c._epicycle_collapsed_symbol = "▶"
    return c


class _RefineList(_PatchList):
    """Patch Selection's list: the Patches list's behaviour (one
    highlighted row, Ctrl+A asks for every tick) on the Model Refinement
    rows."""


class _MiddleElidedLabel(QLabel):
    """A one-line label elided in the MIDDLE when narrow (a model name's
    end is what tells two rounds apart); ``text()`` stays whole and the
    minimum width is 0, so it never pushes the column sideways."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextFormat(Qt.PlainText)
        self.setWordWrap(False)

    def minimumSizeHint(self):  # noqa: N802 (Qt API)
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        rect = self.contentsRect()
        # QLabel's own indent rule (V1): under a style sheet a label has a
        # frame, and Qt then indents its text by half an "x" -- without it
        # these lines started left of every other label of the card
        m = self.indent()
        if m < 0 and self.frameWidth():
            m = self.fontMetrics().horizontalAdvance("x") // 2 - self.margin()
        if m > 0:
            rect.setLeft(rect.left() + m)
        elided = self.fontMetrics().elidedText(self.text(), Qt.ElideMiddle,
                                               rect.width())
        self.style().drawItemText(
            p, rect, int(self.alignment()), self.palette(), self.isEnabled(),
            elided, QPalette.WindowText)
        p.end()


# The Model Refinement Preset combo's descriptions live in
# ``_step3_mr_text`` (what each preset DOES, one source for the panel and
# the guide).  The controller imports this name from here.
PRESET_DESCRIPTIONS = _MRT.PRESET_DESCRIPTIONS

# Back-compat alias.  Older callers may have imported the underscored
# name; remove once nothing references it.
_PRESET_DESCRIPTIONS = PRESET_DESCRIPTIONS


def _guide_button_of(row: QHBoxLayout, guide_filename: str):
    """The ``?`` button ``_make_subsection_header`` put last in *row*,
    tagged with the guide it opens (``property("guide")``), or ``None``."""
    item = row.itemAt(row.count() - 1) if row.count() else None
    btn = item.widget() if item is not None else None
    if btn is not None:
        btn.setProperty("guide", guide_filename)
    return btn


def _make_subsection_header(parent: QWidget, title: str,
                            guide_filename: str = None) -> QHBoxLayout:
    """Build a subsection header row: [styled label] [stretch] [? button].

    Replaces the previous 3-line idiom:
        _hdr = QLabel(title)
        _hdr.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(_hdr)

    With:
        layout.addLayout(_make_subsection_header(self, title, guide_filename))

    The `?` button is omitted when guide_filename is None (or empty).
    When provided, clicking the button opens the markdown file at
    `docs/guides/<guide_filename>` in a non-modal themed help dialog.
    Missing guide files render a friendly "Guide not yet written"
    placeholder, so it's safe to wire a `?` even before the markdown
    is authored.

    parent is the long-lived QWidget that owns the dialog (typically
    the FusionReconstructionCuration instance — `self` at call site).
    """
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    label = QLabel(title)
    label.setStyleSheet(SECTION_HEADER_STYLE)
    row.addWidget(label)
    row.addStretch(1)
    if guide_filename:
        row.addWidget(help_button(
            parent,
            guide_path=str(_GUIDES_DIR / guide_filename),
            title=title,
            accent=_STEP3_ACCENT,
            tooltip=f"Help: {title}",
        ))
    return row


# ---------------------------------------------------------------------------
# Status palette (matches HITL 2D pattern)
# ---------------------------------------------------------------------------

# Build QColor lookup from theme palette (single source of truth)
_STATUS_COLORS = {
    status: QColor(hex_val)
    for status, hex_val in PATCH_STATUS_COLORS.items()
}
_STATUS_ICONS = PATCH_STATUS_ICONS
_STATUS_DOTS = PATCH_STATUS_DOTS
_PROTECTED_STATUSES = PROTECTED_STATUSES

# Axis column width (CZYX / ZCYX / ZYX all fit in 4) — keeps the monospace
# patch-state rows aligned even when a patch's axes are unknown (blank).
_AXIS_COL_WIDTH = 4


def _format_patch_state_row(
    name: str,
    status: str,
    axes: str | None,
    dots: str,
    max_name: int,
    max_status: int,
) -> str:
    """Build the monospace text for one Patch State row (pure / Qt-free).

    Layout: ``" {name}  {status}  {axis}  {dots}"`` with each column padded
    for alignment.  ``axes`` is the per-patch auto-detected axis order
    (``"CZYX"`` / ``"ZCYX"`` / ``"ZYX"``); when ``None``/empty a blank
    placeholder of the same width is shown so columns stay aligned.

    Factored out of ``refresh_patch_state_list`` so the formatting can be
    unit-tested without a live QApplication.
    """
    axis_str = (axes or "").strip()
    return (
        f" {name.ljust(max_name)}"
        f"  {status.ljust(max_status)}"
        f"  {axis_str.ljust(_AXIS_COL_WIDTH)}"
        f"  {dots}"
    )


def _pr_row_flags(item, locked: bool):
    """A Patches row's flags (V1): enabled and selectable (a protected or
    missing row too, for View); checkable while no Predict / Reconstruct
    runs, when the record is tickable or the row is ticked (a
    a tick can be taken off, never put on)."""
    rec = item.data(ROLE_RECORD) or {}
    flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
    ticked = item.checkState() == Qt.Checked
    if (rec.get("tickable", True) or ticked) and not locked:
        flags |= Qt.ItemIsUserCheckable
    return flags


class _TwoLineLabel(QLabel):
    """A wrapping label that keeps the FULL sentence and shows at most two
    lines of it, re-cut whenever its width changes (a cut made at one width
    would clip at a narrower one)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._full = None

    def set_full(self, text: str) -> None:
        self._full = "" if text is None else str(text)
        self._cut()

    def setText(self, text) -> None:  # noqa: N802 (Qt API)
        self._full = None
        super().setText(text)

    def _cut(self) -> None:
        if self._full is None:
            return
        # a few px of slack: a glyph the metrics under-count must not wrap a
        # cut line into a third, hidden one
        cut = _two_lines(self, self._full, 16)
        if cut != self.text():
            super().setText(cut)

    def resizeEvent(self, ev):  # noqa: N802 (Qt API)
        super().resizeEvent(ev)
        self._cut()


class _MrRunLine(QWidget):
    """Model Refinement's run line, Predict + Reconstruct's run row without
    the combo: "Run:" in the section-header style, the run's name at 11 px,
    the dim tail.  Row min-height 28."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(28)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._head = QLabel(_MRT.MR_RUN_HEAD, self)
        self._head.setStyleSheet(SECTION_HEADER_STYLE)
        self._name = _MiddleElidedLabel(self)
        self._name.setStyleSheet(
            "color: #dddddd; font-size: 11px; background: transparent;")
        self._tail = _MiddleElidedLabel(self)
        self._tail.setStyleSheet(DIM_LABEL_STYLE)
        self._tail.setText(_MRT.MR_RUN_TAIL.strip())
        lay.addWidget(self._head, 0)
        lay.addWidget(self._name, 0)
        lay.addWidget(self._tail, 0)
        lay.addStretch(1)

    def set_run(self, name: str) -> None:
        self._name.setText(_MRT.MR_RUN_NAME.format(name=name or "—"))

    def text(self) -> str:
        return " ".join((self._head.text(), self._name.text(),
                         self._tail.text()))


class FusionReconstructionCuration(QWidget):
    """Reconstruct + Curate 3D accordion widget with two-column layouts."""

    #: a Patches row's ⓘ was clicked: open its model details (R6)
    info_requested = Signal(str)

    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self._build_ui()
        self._apply_theme()

        # Controller wired after UI is fully built
        from .controller import FusionReconstructionController
        self.controller = FusionReconstructionController(widget=self, viewer=viewer)

    def _apply_theme(self) -> None:
        apply_inner_group_style(self)
        # Primary action buttons.
        # 2026-05-17: extract_features_btn promoted to primary alongside
        # commit_triage_btn — both are major user-facing actions on the
        # QC pipeline.  commit_triage_btn was also previously listed in
        # both groups; the duplicate has been removed so primary actually
        # applies (secondary was clobbering it).
        for attr in ("segment_button", "predict_selected_btn",
                     "run_reconstruction_btn", "reconstruct_selected_btn",
                     "open_3d_annotator_button", "finalize_labels_button",
                     "extract_features_btn",
                     "commit_triage_btn",
                     # 2026-05-19: Extract Training Pairs is the primary
                     # gate into Model Refinement — promoted to PRIMARY
                     # amber alongside launch_refinement_btn (Start Training)
                     # so the two key actions on the Refinement panel share
                     # consistent visual weight.
                     # (Model Refinement look, B1: Extract is SECONDARY
                     # now; the section's primaries are Start Training and
                     # Use for Re-Prediction.)
                     "refine_use_for_predict_btn",
                     "launch_refinement_btn"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setStyleSheet(PRIMARY_BTN_STYLE)
        # Secondary buttons.
        # extract_all_features_btn intentionally omitted (hidden 2026-05-17).
        # pca_btn / umap_btn / tsne_btn (2026-05-17) — recompute siblings
        # of cluster_btn.  Secondary because they're cheaper, in-session
        # operations vs. the primary "Extract" pipeline run.
        for attr in ("set_workdir_browse_button", "set_patch_directory_button",
                     "lateral_model_browse", "axial_model_browse",
                     "prediction_pause_btn", "reconstruction_pause_btn",
                     "pipeline_cancel_btn",
                     "view_reconstructed_button",
                     "epicycle_defaults_btn", "useg3d_defaults_btn",
                     "save_curation_btn", "undo_curation_btn",
                     "accept_annotation_button",
                     "qc_settings_btn",
                     "pca_btn", "cluster_btn", "umap_btn", "tsne_btn",
                     # assign_tier_btn / undo_tier_btn: styled by the QC
                     # plot area's action row (Step 3 plot pass P2).
                     "save_qc_csv_btn",
                     "apply_threshold_btn",
                     "extract_training_pairs_btn",
                     "clear_extracted_pairs_btn",
                     "refinement_cancel_btn",
                     "refine_configure_per_orientation_btn",
                     "reset_to_predicted_btn",
                     "refine_select_all_btn", "refine_deselect_all_btn",
                     "refine_apply_preset_btn",
                     "refine_save_config_btn", "refine_load_config_btn",
                     "refine_view_models_btn",
                     "refine_import_browse_btn",
                     "refine_import_browse_btn_axial"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setStyleSheet(SECONDARY_BTN_STYLE)
        # Key action icons.
        _btn_icons = {
            "segment_button": "mdi6.flash",
            "predict_selected_btn": "mdi6.flash",
            "run_reconstruction_btn": "mdi6.cube-outline",
            "reconstruct_selected_btn": "mdi6.cube-outline",
            "open_3d_annotator_button": "mdi6.application-edit-outline",
            "finalize_labels_button": "mdi6.check-all",
            "extract_features_btn": "mdi6.chart-scatter-plot",
            "extract_all_features_btn": "mdi6.chart-scatter-plot",
            "launch_refinement_btn": "mdi6.brain",
            "pipeline_cancel_btn": "mdi6.stop-circle-outline",
            "refinement_cancel_btn": "mdi6.stop-circle-outline",
            "view_reconstructed_button": "mdi6.eye-outline",
            "reset_to_predicted_btn": "mdi6.arrow-u-left-top",
            "extract_training_pairs_btn": "mdi6.file-export-outline",
            "refine_use_for_predict_btn": "mdi6.reload",
            "refine_view_models_btn": "mdi6.archive-search-outline",
            "refine_save_config_btn": "mdi6.content-save-outline",
            "refine_load_config_btn": "mdi6.folder-open-outline",
        }
        for attr, icon_name in _btn_icons.items():
            btn = getattr(self, attr, None)
            if btn is not None:
                apply_button_icon(btn, icon_name)
        # Start Training and Cancel are one pair: the primary's height
        _start = getattr(self, "launch_refinement_btn", None)
        _cancel = getattr(self, "refinement_cancel_btn", None)
        if _start is not None and _cancel is not None:
            _h = _start.sizeHint().height()
            _start.setFixedHeight(_h)
            _cancel.setFixedHeight(_h)
        self.setStyleSheet(
            self.styleSheet() + INPUT_STYLE + LIST_STYLE + SCROLLBAR_STYLE
            + COLLAPSIBLE_STYLE + TAB_WIDGET_STYLE
        )
        for attr in ("prediction_progress", "reconstruction_progress",
                     "pipeline_progress",
                     "refinement_progress", "refine_extraction_progress"):
            bar = getattr(self, attr, None)
            if bar is not None:
                bar.setStyleSheet(PROGRESS_BAR_STYLE)

    # ------------------------------------------------------------------
    # Top-level layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Flow overview (compact hint strip)
        overview = QWidget()
        overview.setStyleSheet(FLOW_HINT_STYLE)
        ov_lay = QVBoxLayout(overview)
        ov_lay.setContentsMargins(0, 0, 0, 0)
        ov_lay.setSpacing(2)
        self.routing_hint_label = QLabel(
            "Select a patch \u2192 predict \u2192 reconstruct \u2192 curate \u2192 finalize"
        )
        self.routing_hint_label.setWordWrap(True)
        self.routing_hint_label.setStyleSheet(FLOW_TITLE_STYLE)
        routing = QLabel("Routing: multi-view 2D \u2192 3D fusion \u2192 curation \u2192 finalized labels.")
        routing.setWordWrap(True)
        routing.setStyleSheet(FLOW_SUBTITLE_STYLE)
        ov_lay.addWidget(self.routing_hint_label)
        ov_lay.addWidget(routing)

        root.addWidget(overview)

        # Accordion — 5 sections
        accordion = Accordion()
        accordion.addItem(self._create_session_panel(), "Session",
                          icon_name="mdi6.folder-cog-outline")
        accordion.addItem(self._create_predict_reconstruct_panel(),
                          "Predict + Reconstruct",
                          icon_name="mdi6.cube-outline")
        accordion.addItem(self._create_curation_panel(), "Curation",
                          icon_name="mdi6.pencil-outline")
        accordion.addItem(self._create_model_refinement_panel(),
                          "Model Refinement",
                          icon_name="mdi6.brain")
        accordion.addItem(self._create_diagnostics_panel(), "Diagnostics",
                          icon_name="mdi6.stethoscope")
        accordion.setCurrentIndex(1)
        self._accordion = accordion
        root.addWidget(accordion, 1)

        self.setLayout(root)

    # ------------------------------------------------------------------
    # Summary dashboard
    # ------------------------------------------------------------------

    def _create_summary_dashboard(self) -> QWidget:
        """The Pipeline status block (A2, board B): the title, "Patch
        counts in this run.", the six counters in a 3 x 2 grid, then ONE
        line "Last activity … · Resume …".  The total's label is kept
        (the controller writes it) and hidden."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        _sep = QLabel("")
        _sep.setFixedHeight(1)
        _sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        outer.addWidget(_sep)

        _hdr = QLabel(_PRT.S3_PIPELINE_TITLE)
        _hdr.setStyleSheet(SECTION_HEADER_STYLE)
        outer.addWidget(_hdr)
        _desc = QLabel(_PRT.S3_PIPELINE_DESC)
        _desc.setWordWrap(True)
        _desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        outer.addWidget(_desc)

        # "Last activity" + "Resume" (2026-05-18 Session redesign; A2: one
        # line UNDER the counters, board B).
        activity_row = QHBoxLayout()
        activity_row.setContentsMargins(0, 2, 0, 4)
        activity_row.setSpacing(6)
        _activity_lbl = QLabel(_PRT.S3_PIPELINE_LAST_ACTIVITY)
        _activity_lbl.setStyleSheet(DIM_LABEL_STYLE)
        self.session_last_activity_value = QLabel("never")
        self.session_last_activity_value.setStyleSheet(MONO_LABEL_STYLE)
        activity_row.addWidget(_activity_lbl)
        activity_row.addWidget(self.session_last_activity_value)
        _dot = QLabel(_PRT.S3_PIPELINE_SEP)
        _dot.setStyleSheet(DIM_LABEL_STYLE)
        activity_row.addWidget(_dot)

        resume_row = activity_row
        _resume_lbl = QLabel(_PRT.S3_PIPELINE_RESUME)
        _resume_lbl.setStyleSheet(DIM_LABEL_STYLE)
        # QPushButton styled as a flat link so the user can see it's
        # actionable.  Default text = "Nothing in progress" when no
        # resume target exists.
        self.session_resume_btn = QPushButton("Nothing in progress")
        self.session_resume_btn.setFlat(True)
        self.session_resume_btn.setCursor(Qt.PointingHandCursor)
        self.session_resume_btn.setEnabled(False)
        self.session_resume_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            "color: #888; font-size: 11px; text-align: left; "
            "padding: 0; margin: 0; }"
            "QPushButton:enabled { color: #d4a843; "
            "text-decoration: underline; }"
            "QPushButton:enabled:hover { color: #eccd6f; }"
        )
        resume_row.addWidget(_resume_lbl)
        resume_row.addWidget(self.session_resume_btn, 1)

        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)

        def _val_label(color_name: str) -> QLabel:
            lbl = QLabel("0")
            lbl.setStyleSheet(f"font-weight: bold; color: {color_name};")
            return lbl

        # 6 counters in a 3x2 grid.  Pre-2026-05-18 the dashboard had 4
        # counters with "Reconstructed" silently aggregating
        # reconstructed + curating + accepted — hid mid-curation work.
        # Now each stage gets its own counter so users can see exactly
        # where the in-flight work is.
        self.summary_patches_value = _val_label("#ccc")
        self.summary_unprocessed_value = _val_label(
            _STATUS_COLORS.get("unprocessed", _STATUS_COLORS["predicted"]).name()
            if _STATUS_COLORS.get("unprocessed") else "#888"
        )
        self.summary_predicted_value = _val_label(_STATUS_COLORS["predicted"].name())
        self.summary_reconstructed_value = _val_label(_STATUS_COLORS["reconstructed"].name())
        self.summary_curating_value = _val_label(_STATUS_COLORS["curating"].name())
        self.summary_accepted_value = _val_label(_STATUS_COLORS["accepted"].name())
        self.summary_finalized_value = _val_label(_STATUS_COLORS["finalized"].name())

        # board B: the six stages, three to a row; the total is the header
        # count line's now (its label kept, hidden, for the controller)
        self.summary_patches_value.setParent(container)
        self.summary_patches_value.setVisible(False)
        values = (self.summary_unprocessed_value,
                  self.summary_predicted_value,
                  self.summary_reconstructed_value,
                  self.summary_curating_value,
                  self.summary_accepted_value,
                  self.summary_finalized_value)
        for i, (word, value) in enumerate(zip(_PRT.S3_PIPELINE_COUNTERS,
                                              values)):
            r, col = divmod(i, 3)
            grid.addWidget(QLabel(word), r, 2 * col)
            grid.addWidget(value, r, 2 * col + 1)

        outer.addLayout(grid)
        outer.addLayout(activity_row)
        return container

    def refresh_summary_dashboard(self, counts: dict) -> None:
        """Update pipeline status dashboard with patch counts by status.

        2026-05-18: now reports each stage separately instead of
        aggregating Reconstructed/Curating/Accepted into one bucket.
        """
        total = sum(counts.values())
        self.summary_patches_value.setText(str(total))

        self.summary_unprocessed_value.setText(str(counts.get("unprocessed", 0)))
        self.summary_predicted_value.setText(str(counts.get("predicted", 0)))
        self.summary_reconstructed_value.setText(str(counts.get("reconstructed", 0)))
        self.summary_curating_value.setText(str(counts.get("curating", 0)))
        self.summary_accepted_value.setText(str(counts.get("accepted", 0)))
        self.summary_finalized_value.setText(str(counts.get("finalized", 0)))

    def refresh_session_activity(self, last_activity_text: str,
                                 resume_text: str, resume_enabled: bool) -> None:
        """Update Last activity timestamp + Resume button display.

        Controller computes the strings (so the UI stays dumb) and
        passes them in.  ``resume_enabled=False`` greys the button
        out and disables the click handler.
        """
        self.session_last_activity_value.setText(last_activity_text)
        # A2 fold: a long patch name is elided (the whole text in the
        # tooltip), so the Resume link never widens the panel
        btn = self.session_resume_btn
        btn.setText(btn.fontMetrics().elidedText(
            resume_text or "", Qt.ElideRight, SESSION_RESUME_MAX_W))
        btn.setToolTip(resume_text or "")
        btn.setEnabled(resume_enabled)

    # ------------------------------------------------------------------
    # Patch list coloring
    # ------------------------------------------------------------------

    def update_patch_list_status(self, patch_statuses: dict, *,
                                 axes_by_patch: dict = None,
                                 info_by_patch: dict = None,
                                 shapes: dict = None,
                                 missing=None,
                                 spacing_by_patch: dict = None,
                                 measured_by_patch: dict = None) -> None:
        """Session's rows (brick V2): each item's record for
        the shared painter (``_step3_rows.build_session_row`` -- the status
        dots + word, the line of what is known, the missing line), its
        status in ``ROLE_STATUS`` and its picture from the in-memory map.
        The items themselves (``Qt.UserRole`` = the name, ``text()`` = the
        file name) are made by the folder load and left as they are.

        R7: *spacing_by_patch* -- ``{patch: {"zyx", "source"}}``, the
        stored ``patch_spacing`` -- gives each row its voxel-size line (amber
        "voxels" for a patch with none); re-applied on every rebuild."""
        lst = self.loaded_patch_list
        spacing_by_patch = spacing_by_patch or {}
        measured_by_patch = measured_by_patch or {}
        axes_by_patch = axes_by_patch or {}
        info_by_patch = info_by_patch or {}
        shapes = shapes or {}
        missing = set(missing or ())
        was = lst.blockSignals(True)
        try:
            for i in range(lst.count()):
                item = lst.item(i)
                name = item.data(Qt.UserRole) or item.text()
                status = patch_statuses.get(name, "unprocessed")
                rec = _rows.build_session_row(
                    name, status, axes_by_patch.get(name),
                    manifest_info=info_by_patch.get(name),
                    source_shape=shapes.get(name), missing=name in missing,
                    spacing=spacing_by_patch.get(name),
                    tier_counts=self._tier_counts_for(name),
                    measured_spacing=measured_by_patch.get(name))
                item.setData(ROLE_RECORD, rec)
                item.setData(ROLE_STATUS, status)
                item.setToolTip(rec.get("row_tip") or "")
                self._apply_patch_thumb(item)
        finally:
            lst.blockSignals(was)
        self._fit_session_list_height()
        lst.viewport().update()

    def _fit_session_list_height(self) -> None:
        """Session's list takes its rows' height, between
        ``SESSION_LIST_MIN_H`` and ``SESSION_LIST_MAX_H`` (it scrolls inside
        itself past that) -- Step 4's ``_fit_session_list_height``."""
        lst = self.loaded_patch_list
        if lst.count() == 0:
            lst.setFixedHeight(SESSION_LIST_MIN_H)
            return
        total = 2 * lst.frameWidth() + 4
        for i in range(lst.count()):
            total += max(lst.sizeHintForRow(i), 0)
        lst.setFixedHeight(max(SESSION_LIST_MIN_H,
                               min(total, SESSION_LIST_MAX_H)))

    # ------------------------------------------------------------------
    # The pictures (brick V2): one map, both lists
    # ------------------------------------------------------------------

    def _thumb_lists(self):
        return [lst for lst in (getattr(self, "loaded_patch_list", None),
                                getattr(self, "_curation_bucket", None),
                                getattr(self, "_patch_state_list", None),
                                getattr(self, "refine_patch_list", None))
                if lst is not None]

    def _apply_patch_thumb(self, item) -> None:
        """Put the map's picture and its tooltip on *item* (``None`` and no
        tooltip when the map has nothing for its patch: the painter draws
        the placeholder)."""
        name = item.data(Qt.UserRole) or item.text()
        pixmap, tip = getattr(self, "_patch_thumbs", {}).get(
            name, (None, None))
        item.setData(ROLE_THUMB, pixmap)
        item.setData(ROLE_THUMB_TIP, tip or None)

    def _apply_patch_thumbs(self, names=None) -> None:
        for lst in self._thumb_lists():
            was = lst.blockSignals(True)
            try:
                for i in range(lst.count()):
                    item = lst.item(i)
                    if names is None or (item.data(Qt.UserRole)
                                         or item.text()) in names:
                        self._apply_patch_thumb(item)
            finally:
                lst.blockSignals(was)
            lst.viewport().update()

    def set_patch_thumbs(self, mapping: dict) -> None:
        """REPLACE the whole picture map -- ``{patch: (QPixmap or None,
        tooltip or None)}`` -- and re-apply it to Session's and Curation's
        rows (a new folder's batch: no picture of an older folder's
        same-named patch survives)."""
        self._patch_thumbs = {str(k): (v[0], v[1])
                              for k, v in (mapping or {}).items()}
        self._apply_patch_thumbs()

    def set_patch_thumb(self, name: str, pixmap=None, tip=None) -> None:
        """One patch's picture (or ``None``: the placeholder) and its
        tooltip, on every row of that patch in both lists."""
        if not hasattr(self, "_patch_thumbs"):
            self._patch_thumbs = {}
        self._patch_thumbs[str(name)] = (pixmap, tip)
        self._apply_patch_thumbs({str(name)})

    # ------------------------------------------------------------------
    # 1. Session panel
    # ------------------------------------------------------------------

    def _build_session_patch_header(self, panel: QWidget,
                                    lay: QVBoxLayout) -> None:
        """The Patches card's header (A2, board B): the title with its
        ``?`` (``step3_session.md``), the description and the header count
        line (``set_session_header_counts``)."""
        row = _make_subsection_header(
            panel, _PRT.S3_SESSION_PATCHES_TITLE, _SESSION_GUIDE)
        self.session_patches_help_btn = _guide_button_of(row,
                                                         _SESSION_GUIDE)
        lay.addLayout(row)
        desc = QLabel(_PRT.S3_SESSION_PATCHES_DESC, panel)
        desc.setWordWrap(True)
        desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        lay.addWidget(desc)
        self.session_header_counts = QLabel(panel)
        self.session_header_counts.setTextFormat(Qt.RichText)
        self.session_header_counts.setWordWrap(True)
        self.session_header_counts.setStyleSheet(FLOW_SUBTITLE_STYLE)
        lay.addWidget(self.session_header_counts)
        self.set_session_header_counts(0)

    def set_session_header_counts(self, n: int, curating: int = 0,
                                  finalized: int = 0,
                                  no_voxel: int = 0) -> None:
        """The header count line; the "without a voxel size" clause in
        the warning amber."""
        import html as _html
        head, amber = _PRT.session_header_counts(n, curating, finalized,
                                                 no_voxel)
        text = _html.escape(head)
        if amber:
            text += (f'<span style="color:{SESSION_WARN_COLOUR}">'
                     f'{_html.escape(amber)}</span>')
        self.session_header_counts.setText(text)

    def set_patch_dir_source(self, defaulted: bool, folder: str = "", *,
                             workspace: bool = True) -> None:
        """The dim sentence beside "Use another folder…": the Step 1
        default, or the chosen folder's name (its full path the tooltip; the
        label elides, so a long name never widens the panel).  Hidden with
        no workspace."""
        lab = self.patch_dir_qualifier
        if defaulted or not folder:
            lab.setText(_PRT.S3_SESSION_USING_DEFAULT)
            lab.setToolTip(folder or _PRT.S3_SESSION_USING_DEFAULT)
        else:
            name = os.path.basename(os.path.normpath(str(folder))) or folder
            lab.setText(_PRT.S3_SESSION_USING_FOLDER.format(folder=name))
            lab.setToolTip(str(folder))
        lab.setVisible(bool(workspace))

    def set_session_eye_locked(self, locked: bool) -> None:
        """The rows' eye dims (and the controller refuses it) exactly when
        the foot's View is grey."""
        lst = self.loaded_patch_list
        lst.setProperty(SESSION_EYE_LOCK_PROPERTY, bool(locked))
        lst.viewport().update()

    def set_tier_counts(self, mapping: dict, *, replace: bool = False) -> None:
        """The tier counts of the rows (A2) --
        ``{patch: tier items or None}`` -- kept in a map that every
        rebuild of either list reads, and put on the rows now.  *replace*
        drops every patch the mapping does not name."""
        if replace or not hasattr(self, "_tier_count_items"):
            self._tier_count_items = {}
        for k, v in (mapping or {}).items():
            self._tier_count_items[str(k)] = v
        # the two WIDE lists by name, and (A3) the PROTECTED rows of the
        # Predict + Reconstruct list, whose narrow rows draw the counts
        # compactly under the chip
        pr = getattr(self, "_patch_state_list", None)
        if pr is not None:
            was = pr.blockSignals(True)
            try:
                for i in range(pr.count()):
                    item = pr.item(i)
                    rec = item.data(ROLE_RECORD)
                    if rec is None:
                        continue
                    name = item.data(Qt.UserRole) or item.text()
                    item.setData(ROLE_RECORD,
                                 self._with_pr_tier_counts(rec, name))
            finally:
                pr.blockSignals(was)
            pr.doItemsLayout()
            pr.viewport().update()
        for lst in (getattr(self, "loaded_patch_list", None),
                    getattr(self, "_curation_bucket", None)):
            if lst is None:
                continue
            was = lst.blockSignals(True)
            try:
                for i in range(lst.count()):
                    item = lst.item(i)
                    rec = item.data(ROLE_RECORD)
                    if rec is None:
                        continue
                    name = item.data(Qt.UserRole) or item.text()
                    item.setData(ROLE_RECORD,
                                 self._with_tier_counts(rec, name))
            finally:
                lst.blockSignals(was)
            lst.doItemsLayout()
            lst.viewport().update()
        self._fit_session_list_height()

    def _tier_counts_for(self, name: str):
        return getattr(self, "_tier_count_items", {}).get(str(name))

    def _with_tier_counts(self, rec: dict, name: str) -> dict:
        rec = dict(rec)
        items = self._tier_counts_for(name)
        if items:
            rec["tier_counts"] = tuple(items)
        else:
            rec.pop("tier_counts", None)
        return rec

    def _with_pr_tier_counts(self, rec: dict, name: str) -> dict:
        """The Predict + Reconstruct row's record with the tier counts of
        its patch on disk -- only a PROTECTED row (accepted / finalized)
        draws them (A3); any other row has none."""
        if rec.get("status") in PROTECTED_STATUSES:
            return self._with_tier_counts(rec, name)
        rec = dict(rec)
        rec.pop("tier_counts", None)
        return rec

    def _build_session_patch_list(self, panel: QWidget,
                                  lay: QVBoxLayout) -> None:
        """Session's patch list (brick V2): painted by
        the shared row painter in ``MODE_WIDE`` with no tick and no eye --
        the picture, the status dots + word, the name and the line of what
        is known -- in Step 3's colours; it dims under a Predict /
        Reconstruct lock like the Patches rows.  Selection is unchanged
        (several rows, the first is the active patch)."""
        self.loaded_patch_list = StackListWidget(panel)
        self.loaded_patch_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.loaded_patch_list.setToolTip(_PRT.S3_SESSION_LIST_HELP)
        # A2 (board B): the eye -- the foot's View for that row, dimmed
        # exactly when View is grey -- and the inline Set spacing…
        self._session_row_delegate = StackRowDelegate(
            MODE_WIDE, self.loaded_patch_list, palette=PR_ROW_PALETTE,
            accent=PR_ROW_ACCENT,
            show_tick=False, show_eye=True, lock_property=PR_LOCK_PROPERTY,
            eye_lock_property=SESSION_EYE_LOCK_PROPERTY,
            dim_when_locked=True, keep_aspect=True,
            help_lines={"row_eye": _PRT.S3_SESSION_EYE_TIP,
                        "row_action": _PRT.S3_SET_SPACING_TIP,
                        "row_tiers": _PRT.S3_SESSION_TIERS_TIP})
        self.loaded_patch_list.setItemDelegate(self._session_row_delegate)
        self.loaded_patch_list.setFixedHeight(SESSION_LIST_MIN_H)
        self.loaded_patch_list.set_empty_lines(_PRT.S3_SESSION_EMPTY_LINES)
        # R7: a right-click opens the row's menu too
        self.loaded_patch_list.setContextMenuPolicy(Qt.CustomContextMenu)
        lay.addWidget(self.loaded_patch_list)
        self._build_session_folder_row(panel, lay)
        self._build_session_spacing_row(panel, lay)
        # A2: board B has no button row under the list for the spacing --
        # it is the amber row's own button and the row menu now; the row
        # is kept (its gate still runs) and hidden
        self._session_spacing_row.setVisible(False)

    def _build_session_folder_row(self, panel: QWidget,
                                  lay: QVBoxLayout) -> None:
        """Under the list (A2, board B): "Use another folder…" (the Browse
        the folder field had) and the dim sentence that says where the
        patches come from.  The path field itself is kept -- the
        controller reads and writes it -- and hidden."""
        row = QWidget(panel)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        self.set_patch_directory_path = QLineEdit(row)
        self.set_patch_directory_path.setPlaceholderText("Path to z-stacks...")
        self.set_patch_directory_path.setVisible(False)
        self.set_patch_directory_button = QPushButton(
            _PRT.S3_SESSION_USE_FOLDER_BTN, row)
        self.set_patch_directory_button.setToolTip(
            _PRT.S3_SESSION_USE_FOLDER_TIP)
        h.addWidget(self.set_patch_directory_button, 0)
        # B3: with no folder picked, this step
        # reads the patches Step 1 extracted into the workspace; A2: the
        # line says which folder is read -- elided, never clipped, and
        # hidden until there is a workspace
        from ._qc_plot_area import ElidedLabel
        self.patch_dir_qualifier = ElidedLabel(row)
        self.patch_dir_qualifier.setText(_PRT.S3_SESSION_USING_DEFAULT)
        self.patch_dir_qualifier.setStyleSheet(FLOW_SUBTITLE_STYLE)
        self.patch_dir_qualifier.setVisible(False)
        h.addWidget(self.patch_dir_qualifier, 1)
        self._session_folder_row = row
        lay.addWidget(row)
        # A2 fold: Session's own one-line status (a refused eye or inline
        # button says why HERE, where the user clicked)
        self.session_status_label = ElidedLabel(panel)
        self.session_status_label.setStyleSheet(DIM_LABEL_STYLE)
        self.session_status_label.setVisible(False)
        lay.addWidget(self.session_status_label)

    def set_session_status(self, text: str) -> None:
        """Session's status line: the sentence (elided, whole in the
        tooltip), hidden when empty."""
        self.session_status_label.setText(text or "")
        self.session_status_label.setVisible(bool(text))

    def _build_session_spacing_row(self, panel: QWidget,
                                   lay: QVBoxLayout) -> None:
        """R7: "Set spacing…" under Session's list, acting
        on the highlighted rows, with its grey reason beside it on screen
        (``set_session_spacing_gate``; Qt shows no tooltip on a disabled
        button).  The controller wires the click."""
        self._session_spacing_row = QWidget(panel)
        row = QHBoxLayout(self._session_spacing_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        panel = self._session_spacing_row
        self.session_set_spacing_btn = QPushButton(_PRT.S3_SET_SPACING_BTN,
                                                   panel)
        self.session_set_spacing_btn.setToolTip(_PRT.S3_SET_SPACING_TIP)
        self.session_set_spacing_btn.setEnabled(False)
        row.addWidget(self.session_set_spacing_btn, 0)
        self.session_set_spacing_note = QLabel(
            _PRT.S3_SET_SPACING_NEEDS_PATCH, panel)
        self.session_set_spacing_note.setWordWrap(True)
        self.session_set_spacing_note.setStyleSheet(FLOW_SUBTITLE_STYLE)
        row.addWidget(self.session_set_spacing_note, 1)
        row.addStretch(0)
        lay.addWidget(self._session_spacing_row)

    def set_session_spacing_gate(self, enabled: bool, reason: str = "") -> None:
        """Set spacing…'s grey state and its reason beside it ("" hides
        the reason: the button is live)."""
        self.session_set_spacing_btn.setEnabled(bool(enabled))
        self.session_set_spacing_note.setText(reason or "")
        self.session_set_spacing_note.setVisible(bool(reason))

    def _create_session_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self._session_panel = panel
        # Workspace (A2, board B: its ? and its words)
        _ws_row = _make_subsection_header(
            panel, _PRT.S3_SESSION_WORKSPACE_TITLE, _SESSION_GUIDE)
        self.session_workspace_help_btn = _guide_button_of(_ws_row,
                                                           _SESSION_GUIDE)
        lay.addLayout(_ws_row)
        _desc = QLabel(_PRT.S3_SESSION_WORKSPACE_DESC)
        _desc.setWordWrap(True)
        _desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        lay.addWidget(_desc)
        self.set_workdir_path = QLineEdit()
        self.set_workdir_path.setPlaceholderText("Select workspace directory...")
        self.set_workdir_browse_button = QPushButton("Browse")
        row = QHBoxLayout()
        row.addWidget(self.set_workdir_path, 1)
        row.addWidget(self.set_workdir_browse_button)
        lay.addLayout(row)

        # Separator
        _sep = QLabel("")
        _sep.setFixedHeight(1)
        _sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        lay.addWidget(_sep)

        # The Patches card (A2, board B): title, ?, description, the
        # header count line, the painted list, then "Use another folder…"
        self._build_session_patch_header(panel, lay)
        self._build_session_patch_list(panel, lay)

        # Separator
        _sep = QLabel("")
        _sep.setFixedHeight(1)
        _sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        lay.addWidget(_sep)

        # Model status: wrapped in ONE container and HIDDEN
        # (the model is chosen in Predict + Reconstruct); the
        # controller still writes the two value labels.
        self._session_model_status = QWidget(panel)
        ms_lay = QVBoxLayout(self._session_model_status)
        ms_lay.setContentsMargins(0, 0, 0, 0)
        ms_lay.setSpacing(6)
        _hdr = QLabel("Model Status")
        _hdr.setStyleSheet(SECTION_HEADER_STYLE)
        ms_lay.addWidget(_hdr)
        _desc = QLabel("Active lateral and axial model sources for 2D prediction.")
        _desc.setWordWrap(True)
        _desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        ms_lay.addWidget(_desc)
        status_form = QFormLayout()
        self.status_lateral_model_value = QLabel("-")
        self.status_lateral_model_value.setStyleSheet(MONO_LABEL_STYLE)
        self.status_axial_model_value = QLabel("-")
        self.status_axial_model_value.setStyleSheet(MONO_LABEL_STYLE)
        status_form.addRow("Lateral:", self.status_lateral_model_value)
        status_form.addRow("Axial:", self.status_axial_model_value)
        ms_lay.addLayout(status_form)
        self._session_model_status.setVisible(False)
        lay.addWidget(self._session_model_status)

        # Pipeline status (A2, board B: under the list, the model pointer
        # after it)
        dash = self._create_summary_dashboard()
        self._session_dashboard = dash
        lay.addWidget(dash)
        self._build_session_model_pointer(panel, lay)

        lay.addStretch(1)
        return panel

    def _build_session_model_pointer(self, panel: QWidget,
                                     lay: QVBoxLayout) -> None:
        """Where the hidden Model Status block was: the dim line that says
        where the model is chosen."""
        self.session_model_pointer = QLabel(_PRT.S3_SESSION_MODEL_POINTER,
                                            panel)
        self.session_model_pointer.setWordWrap(True)
        self.session_model_pointer.setStyleSheet(DIM_LABEL_STYLE)
        lay.addWidget(self.session_model_pointer)

    # ------------------------------------------------------------------
    # 2. Predict + Reconstruct panel (two-column bucket layout)
    # ------------------------------------------------------------------

    def _create_predict_reconstruct_panel(self) -> QWidget:
        """The Predict + Reconstruct section on Step 4's ② layout, in amber.

        Top to bottom: the lock line (hidden while no flag is set; V1 put
        it over the run row, the mockup's board F), the run row (full
        width); the splitter 224 | 330 (6 px
        handle, minimums 200 | 300, not collapsible, stretch 0 : 1).  LEFT:
        the Model card and the Patches card.  RIGHT: a scroll area holding
        R2a's settings column over the pinned foot (Predict Selected ·
        Reconstruct Selected / Cancel · View, the bar, ONE status line).
        The four buttons, the bar and the status label are the same objects
        as before, so ``_CURATION_LOCK_WIDGETS`` and every alias below keep
        pointing at them.  Controller-free (the parity tests build
        ``_build_ui`` without one).
        """
        panel = QWidget()
        self._pr_panel = panel
        # Ticks gold over the whole section (P3)
        panel.setStyleSheet(PANEL_CHECKBOX_STYLE)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(6, 8, 6, 6)
        outer.setSpacing(6)
        self._pr_help_buttons = []
        # The lock line (row 21), at the top of the section as on the
        # mockup's board F and on Step 4's ②: hidden while no flag is set
        self.pr_lock_line = LockLine(panel, object_name=_PR_LOCK_OBJECT)
        outer.addWidget(self.pr_lock_line)

        # --- The run row (Step 3 multi-run): the SAME widgets, full width
        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        _run_lbl = QLabel(_PRT.S3_RUN_LABEL)
        _run_lbl.setStyleSheet(SECTION_HEADER_STYLE)
        run_row.addWidget(_run_lbl)
        self.run_picker_combo = QComboBox()
        self.run_picker_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.run_picker_combo.setMinimumContentsLength(10)
        run_row.addWidget(self.run_picker_combo, 1)
        self.new_run_button = QPushButton(_PRT.S3_NEW_RUN_BTN)
        self.new_run_button.setStyleSheet(SECONDARY_BTN_STYLE)
        apply_button_icon(self.new_run_button, "mdi6.plus-circle-outline",
                          color=_STEP3_ACCENT, size=14)
        run_row.addWidget(self.new_run_button)
        self.manage_runs_button = QPushButton(_PRT.S3_MANAGE_RUNS_BTN)
        self.manage_runs_button.setStyleSheet(SECONDARY_BTN_STYLE)
        apply_button_icon(self.manage_runs_button, "mdi6.cog-outline",
                          color=_STEP3_ACCENT, size=14)
        run_row.addWidget(self.manage_runs_button)
        outer.addLayout(run_row)
        outer.addWidget(_pr_separator(panel))

        # --- The splitter (row 22: Step 4's numbers)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(SPLITTER_STYLE)
        splitter.setHandleWidth(PR_SPLIT_HANDLE)
        splitter.setChildrenCollapsible(False)
        self.pr_splitter = splitter

        # LEFT -- the Model card, then the Patches card
        left = QWidget()
        left.setMinimumWidth(PR_SPLIT_LEFT_MIN)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, PR_LEFT_INNER_MARGIN, 0)
        # V1: the mockup's column gap; the separator keeps its own margin
        left_lay.setSpacing(PR_COL_GAP)
        self.pr_left = left
        self._build_pr_model_card(left, left_lay)
        left_lay.addSpacing(PR_SEP_MARGIN)
        left_lay.addWidget(_pr_separator(left))
        left_lay.addSpacing(PR_SEP_MARGIN)
        self._build_pr_patches_card(left, left_lay)

        # RIGHT -- the settings (they scroll) over the pinned foot
        right = QWidget()
        right.setMinimumWidth(PR_SPLIT_RIGHT_MIN)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(PR_RIGHT_INNER_MARGIN, 0, 0, 0)
        right_lay.setSpacing(0)
        self.pr_right = right
        self.pr_right_scroll = QScrollArea(right)
        self.pr_right_scroll.setWidgetResizable(True)
        self.pr_right_scroll.setFrameShape(QFrame.NoFrame)
        # V1: the settings scroll inside the column, so their preferred
        # height is not the section's -- it takes what the column has
        self.pr_right_scroll.setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Ignored)
        self.pr_right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        content = QWidget()
        self.pr_right_content = content
        c_lay = QVBoxLayout(content)
        c_lay.setContentsMargins(0, 0, 0, 6)
        c_lay.setSpacing(6)
        # --- Settings, moved
        self._build_pr_settings(c_lay)
        c_lay.addStretch(1)
        self.pr_right_scroll.setWidget(content)
        right_lay.addWidget(self.pr_right_scroll, 1)
        self._build_pr_foot(right, right_lay)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([PR_SPLIT_LEFT_W, PR_SPLIT_RIGHT_W])
        outer.addWidget(splitter, 1)

        self._apply_pr_layout_help_lines()

        # ============================================================
        # Backward-compat aliases (controller still uses old names)
        # ============================================================
        self.segment_button = self.predict_selected_btn
        self._predicted_bucket = self._patch_state_list
        self._reconstructed_bucket = self._patch_state_list
        self.run_reconstruction_btn = self.reconstruct_selected_btn
        self.prediction_pause_btn = self.pipeline_cancel_btn
        self.reconstruction_pause_btn = self.pipeline_cancel_btn
        self.prediction_progress = self.pipeline_progress
        self.reconstruction_progress = self.pipeline_progress
        self.prediction_status_label = self.pipeline_status_label
        self.reconstruction_status_label = self.pipeline_status_label
        self.reconstructed_results_list = self._patch_state_list
        self.batch_process_btn = self.predict_selected_btn
        self.batch_pause_btn = self.pipeline_cancel_btn
        self.batch_progress_label = self.pipeline_status_label
        # commit_to_recon_btn and clear_reconstruction_btn removed
        # 2026-05-18 — were hidden dead-code aliases never wired.
        self._recon_column = right  # alias so controller enable/disable doesn't crash
        self._build_pr_legacy_hidden(panel, outer)
        return panel

    def _build_pr_model_card(self, left: QWidget, lay: QVBoxLayout) -> None:
        """The Model card (the mockup's ``.mcard``): header + ``?``, the
        sentence, "Active:" + Change…, then the model lines as ONE block --
        one NEUTRAL line per bucket, elided in the middle, the full name in
        its tooltip, no gap between them.  The card's lines sit
        ``PR_CARD_GAP`` apart, the sentence close under its header (the
        mockup's ``.desc`` margin).  The hidden model-state widgets stay
        built (the controller reads and writes them)."""
        card = QVBoxLayout()
        card.setContentsMargins(0, 0, 0, 0)
        card.setSpacing(PR_CARD_GAP)
        head = QVBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(PR_CARD_GAP - 2)
        head.addLayout(sub_header(
            self, _PRT.S3_MODEL_TITLE, str(_GUIDES_DIR / _PR_MODEL_GUIDE),
            accent=_STEP3_ACCENT, help_buttons=self._pr_help_buttons,
            guide=_PR_MODEL_GUIDE))
        desc = QLabel(_PRT.S3_MODEL_DESC, left)
        desc.setWordWrap(True)
        desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        head.addWidget(desc)
        card.addLayout(head)
        act = QHBoxLayout()
        act.setContentsMargins(0, 0, 0, 0)
        act.setSpacing(8)
        act.addWidget(QLabel(_PRT.S3_MODEL_ACTIVE_LABEL, left))
        act.addStretch(1)
        self.model_change_btn = QPushButton(_PRT.S3_MODEL_CHANGE_BTN, left)
        act.addWidget(self.model_change_btn)
        card.addLayout(act)
        lines = QVBoxLayout()
        lines.setContentsMargins(0, 0, 0, 0)
        lines.setSpacing(0)
        self.pr_model_lat_label = _MiddleElidedLabel(left)
        self.pr_model_ax_label = _MiddleElidedLabel(left)
        for lbl in (self.pr_model_lat_label, self.pr_model_ax_label):
            lbl.setStyleSheet(PR_MODEL_LINE_STYLE)
            lines.addWidget(lbl)
        card.addLayout(lines)
        # B (board D): what the active models were trained on, under the two
        # names; it WRAPS so the channel names stay readable (owner, live
        # shots), the whole sentence is the tooltip; hidden with no model
        # and for a Cellpose built-in
        self.pr_model_trained_label = QLabel("", left)
        self.pr_model_trained_label.setWordWrap(True)
        self.pr_model_trained_label.setVisible(False)
        card.addWidget(self.pr_model_trained_label)
        self.show_pr_model("", "", "", "")
        self._build_pr_hidden_model_state(left, card)
        lay.addLayout(card)

    def _build_pr_patches_card(self, left: QWidget, lay: QVBoxLayout) -> None:
        """The Patches card: header + ``?``, its sentence, the patch
        list painted by the shared row painter in ``MODE_NARROW`` (Step 3's
        palette, no eye: View is the foot's button), the ticked
        lines as one tight block, and the "Tick all unprocessed" link."""
        head = QVBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(PR_COL_GAP - 2)
        head.addLayout(sub_header(
            self, _PRT.S3_PATCHES_TITLE,
            str(_GUIDES_DIR / _PR_PATCHES_GUIDE),
            accent=_STEP3_ACCENT, help_buttons=self._pr_help_buttons,
            guide=_PR_PATCHES_GUIDE))
        desc = QLabel(_PRT.S3_PATCHES_DESC, left)
        desc.setWordWrap(True)
        desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        head.addWidget(desc)
        lay.addLayout(head)
        # Unified patch state list, painted from each item's record
        self._patch_state_list = _PatchList(left)
        # Step 4's floor: the list gives up height before the section
        # scrolls (a lock line on a short screen, V1)
        self._patch_state_list.setMinimumHeight(PR_LIST_MIN_H)
        # row 20: the ticks choose what runs; ONE highlighted row is View's
        self._patch_state_list.setSelectionMode(QListWidget.SingleSelection)
        self._patch_state_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._pr_row_delegate = StackRowDelegate(
            MODE_NARROW, self._patch_state_list,
            help_lines=_PRT.S3_ROW_HELP_LINES, palette=PR_ROW_PALETTE,
            accent=PR_ROW_ACCENT,
            show_eye=False, lock_property=PR_LOCK_PROPERTY,
            eye_lock_property=PR_LOCK_PROPERTY, dim_when_locked=True,
            show_info=True, show_thumb=True, keep_aspect=True)
        # R6: the row's ⓘ, forwarded signal to signal (no lambda)
        self._pr_row_delegate.info_clicked.connect(self.info_requested)
        self._patch_state_list.setItemDelegate(self._pr_row_delegate)
        lay.addWidget(self._patch_state_list, 1)
        ticked = QVBoxLayout()
        ticked.setContentsMargins(0, 0, 0, 0)
        ticked.setSpacing(0)
        self.pr_ticked_predict_label = QLabel(left)
        self.pr_ticked_recon_label = QLabel(left)
        for lbl in (self.pr_ticked_predict_label, self.pr_ticked_recon_label):
            lbl.setWordWrap(True)
            lbl.setStyleSheet(DIM_LABEL_STYLE)
            ticked.addWidget(lbl)
        # under a Predict / Reconstruct lock the ticked lines give way to
        # what still works (the mockup's board F)
        self.pr_locked_list_note = QLabel(_PRT.S3_LOCK_LIST_NOTE, left)
        self.pr_locked_list_note.setWordWrap(True)
        self.pr_locked_list_note.setStyleSheet(DIM_LABEL_STYLE)
        self.pr_locked_list_note.setVisible(False)
        ticked.addWidget(self.pr_locked_list_note)
        lay.addLayout(ticked)
        self.set_pr_ticked_lines(0, 0, 0)
        self.pr_tick_all_link = _ClickLabel(_PRT.S3_TICK_ALL_LINK, left)
        self.pr_tick_all_link.setStyleSheet(PR_LINK_STYLE)
        lay.addWidget(self.pr_tick_all_link)

    def _build_pr_foot(self, right: QWidget, lay: QVBoxLayout) -> None:
        """The pinned foot: the four buttons 2 × 2, the bar and ONE
        status line -- a sibling of the scroll area, never inside it."""
        foot = QWidget(right)
        foot.setObjectName(_PR_FOOT_OBJECT)
        foot.setStyleSheet(PR_FOOT_STYLE)
        self.pr_foot = foot
        f_lay = QVBoxLayout(foot)
        f_lay.setContentsMargins(0, 6, 0, 0)
        f_lay.setSpacing(6)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        self.predict_selected_btn = QPushButton(_PRT.S3_PREDICT_BTN, foot)
        self.reconstruct_selected_btn = QPushButton(_PRT.S3_RECONSTRUCT_BTN,
                                                    foot)
        self.pipeline_cancel_btn = QPushButton(_PRT.S3_CANCEL_BTN, foot)
        self.pipeline_cancel_btn.setEnabled(False)
        self.view_reconstructed_button = QPushButton(_PRT.S3_VIEW_BTN, foot)
        for b in (self.predict_selected_btn, self.reconstruct_selected_btn,
                  self.pipeline_cancel_btn, self.view_reconstructed_button):
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            b.setMinimumWidth(PR_FOOT_BTN_MIN_W)
        grid.addWidget(self.predict_selected_btn, 0, 0)
        grid.addWidget(self.reconstruct_selected_btn, 0, 1)
        grid.addWidget(self.pipeline_cancel_btn, 1, 0)
        grid.addWidget(self.view_reconstructed_button, 1, 1)
        f_lay.addLayout(grid)
        # Unified pipeline progress
        self.pipeline_progress = QProgressBar(foot)
        self.pipeline_progress.setTextVisible(True)
        self.pipeline_progress.setFixedHeight(PR_PROGRESS_H)
        f_lay.addWidget(self.pipeline_progress)
        self.pipeline_status_label = QLabel(foot)
        self.pipeline_status_label.setStyleSheet(MONO_LABEL_STYLE)
        self.pipeline_status_label.setSizePolicy(QSizePolicy.Ignored,
                                                 QSizePolicy.Preferred)
        self.pipeline_status_label.setWordWrap(True)
        f_lay.addWidget(self.pipeline_status_label)
        lay.addWidget(foot)

    def _build_pr_hidden_model_state(self, left: QWidget,
                                     lay: QVBoxLayout) -> None:
        """The model-state widgets the controller still reads and writes,
        never shown (2026-05-21 Stage 3; the hidden model-state
        widgets stay built).  Parented through a hidden container (a parentless QWidget becomes a taskbar orphan).  The indicator
        the controller writes (``model_active_indicator_label``) lives here
        too: the card shows the two neutral bucket lines instead."""
        self._hidden_model_state = QWidget(left)
        self._hidden_model_state.setVisible(False)
        hidden_lay = QVBoxLayout(self._hidden_model_state)
        hidden_lay.setContentsMargins(0, 0, 0, 0)
        hidden_lay.setSpacing(0)
        self.model_active_indicator_label = QLabel(self._hidden_model_state)
        hidden_lay.addWidget(self.model_active_indicator_label)
        self.lateral_model_path = QLineEdit()
        hidden_lay.addWidget(self.lateral_model_path)
        self.axial_model_path = QLineEdit()
        hidden_lay.addWidget(self.axial_model_path)
        # pretrained_combo: hidden state (``_on_pretrained_changed``)
        self.pretrained_combo = QComboBox()
        self.pretrained_combo.addItems(["cyto3", "nuclei"])
        hidden_lay.addWidget(self.pretrained_combo)
        # Browse buttons: no-op state-only widgets the controller wires
        self.lateral_model_browse = QPushButton()
        hidden_lay.addWidget(self.lateral_model_browse)
        self.axial_model_browse = QPushButton()
        hidden_lay.addWidget(self.axial_model_browse)
        # model_source_combo: hidden state for ``_model_source_mode``
        self.model_source_combo = QComboBox()
        self.model_source_combo.addItems(["Auto from HITL", "Import", "Pretrained"])
        self.model_source_combo.setCurrentText("Import")
        self.model_source_combo.currentIndexChanged.connect(
            self._on_model_source_changed,
        )
        hidden_lay.addWidget(self.model_source_combo)
        lay.addWidget(self._hidden_model_state)

    def _build_pr_legacy_hidden(self, panel: QWidget,
                                outer: QVBoxLayout) -> None:
        """Legacy processing mode -- not shown, kept for controller compat."""
        self.processing_mode_combo = QComboBox(parent=panel)
        self.processing_mode_combo.addItems(["Batch All"])
        self.processing_mode_combo.setVisible(False)

    def _apply_pr_layout_help_lines(self) -> None:
        """The run row, the cards and the foot get their help lines as
        tooltips (``S3_LAYOUT_HELP_LINES``)."""
        for name, line in _PRT.S3_LAYOUT_HELP_LINES.items():
            w = getattr(self, name, None)
            if w is not None and hasattr(w, "setToolTip"):
                w.setToolTip(line)

    def show_pr_model(self, lat_name: str, ax_name: str,
                      lat_full: str = "", ax_full: str = "") -> None:
        """The Model card's two neutral lines: one per bucket,
        elided in the middle, the full name as tooltip.  An empty name reads
        ``S3_MODEL_NO_NAME``."""
        none = _PRT.S3_MODEL_NO_NAME
        for lbl, line, tip, name, full in (
                (self.pr_model_lat_label, _PRT.S3_MODEL_LAT_LINE,
                 _PRT.S3_MODEL_LAT_TIP, lat_name, lat_full),
                (self.pr_model_ax_label, _PRT.S3_MODEL_AX_LINE,
                 _PRT.S3_MODEL_AX_TIP, ax_name, ax_full)):
            lbl.setText(line.format(name=name or none))
            lbl.setToolTip(tip.format(name=full or name or none))

    def show_pr_trained(self, text: str, known: bool) -> None:
        """The Model card's trained-channels line (B, board D): always the
        dim style (``DIM_LABEL_STYLE``, the mockup's ``.dim``); *known* is
        kept for the caller's sake, only "" hides it."""
        lbl = self.pr_model_trained_label
        lbl.setText(text or "")
        lbl.setToolTip(text or "")
        lbl.setStyleSheet(DIM_LABEL_STYLE)      # board D's .dim, known or not
        lbl.setVisible(bool(text))

    def set_pr_ticked_lines(self, n: int, k: int, m: int) -> None:
        """The Patches card's two ticked lines: *n* ticked, *k* of them
        protected, *m* ready to reconstruct (V1: the ticks)."""
        text = _PRT.S3_TICKED_PREDICT.format(n=n)
        if k:
            text += _PRT.S3_TICKED_PROTECTED.format(k=k)
        self.pr_ticked_predict_label.setText(text)
        self.pr_ticked_recon_label.setText(
            _PRT.S3_TICKED_RECONSTRUCT.format(m=m, n=n))

    def set_pr_lock_line(self, text: str = "") -> None:
        """The section's lock line (row 21): *text* shows it, "" hides it."""
        self.pr_lock_line.setText(text or "")

    #: the run row and the model card's button: greyed outright under a
    #: Predict / Reconstruct lock (row 21)
    _PR_SURFACE_LOCK_WIDGETS = ("run_picker_combo", "new_run_button",
                                "manage_runs_button", "model_change_btn",
                                "pr_tick_all_link")
    #: the setting inputs greyed one by one (never the ? buttons or the
    #: disclosures of the same column; Step 4's K5)
    _PR_SETTING_INPUT_TYPES = (QAbstractSpinBox, QComboBox, QCheckBox,
                               QLineEdit)

    def _pr_setting_inputs(self):
        """The TOP-LEVEL setting inputs of the right column (Step 4's H1):
        a spin box's or combo's own inner QLineEdit is not one."""
        types = self._PR_SETTING_INPUT_TYPES
        return [w for w in self.pr_right_content.findChildren(QWidget)
                if isinstance(w, types)
                and not isinstance(w.parentWidget(), types)]

    def set_pr_surface_locked(self, locked: bool) -> None:
        """Row 21: while Predict or Reconstruct runs, the run row, Change…,
        "Tick all unprocessed", every setting input (each one's OWN enable
        state, ``WA_ForceDisabled``, remembered and given back on unlock)
        and the ticks (``ItemIsUserCheckable`` off, quietly; the rows dim)
        are grey.  The ? buttons and the disclosures stay live; the rows
        stay clickable for View.  After the unlock the column's own gates
        run again (Step 4's M1)."""
        locked = bool(locked)
        if locked == getattr(self, "_pr_surface_locked", False):
            return
        self._pr_surface_locked = locked
        for name in self._PR_SURFACE_LOCK_WIDGETS:
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(not locked)
        if locked:
            self._pr_setting_prior = [
                (w, not w.testAttribute(Qt.WA_ForceDisabled))
                for w in self._pr_setting_inputs()]
            for w, _on in self._pr_setting_prior:
                w.setEnabled(False)
        else:
            for w, on in getattr(self, "_pr_setting_prior", None) or []:
                try:
                    w.setEnabled(on)
                except RuntimeError:
                    pass
            self._pr_setting_prior = []
            self._update_pr_flow_controls()
            self._update_pr_size_off_note()
        for lbl in (self.pr_ticked_predict_label, self.pr_ticked_recon_label):
            lbl.setVisible(not locked)
        self.pr_locked_list_note.setVisible(locked)
        # the link ticks rows: it gives way too (the mockup's board F)
        self.pr_tick_all_link.setVisible(not locked)
        self._apply_pr_tick_lock()
        print(f"[STEP3-LOCK] surface locked={locked}")

    def _apply_pr_tick_lock(self) -> None:
        """A row's box is checkable only when its record is tickable -- or, when it is ticked, so it can be unticked -- and no
        Predict / Reconstruct runs.  Under ``blockSignals``.  Session's
        rows dim with them (brick V2); they have no ticks, so nothing else
        of Session greys."""
        lst = self._patch_state_list
        locked = bool(getattr(self, "_pr_surface_locked", False))
        session = getattr(self, "loaded_patch_list", None)
        if session is not None:
            session.setProperty(PR_LOCK_PROPERTY, locked)
            session.viewport().update()
        lst.setProperty(PR_LOCK_PROPERTY, locked)
        was = lst.blockSignals(True)
        try:
            for i in range(lst.count()):
                it = lst.item(i)
                it.setFlags(_pr_row_flags(it, locked))
        finally:
            lst.blockSignals(was)
        lst.viewport().update()

    def pr_status_fits(self, text: str, kind: str = "neutral") -> bool:
        """Would *text* show whole in the foot's two lines at its width?"""
        lbl = self.pipeline_status_label
        lbl.setStyleSheet(_PR_STATUS_STYLES.get(kind, MONO_LABEL_STYLE))
        return _two_lines(lbl, str(text)) == str(text)

    def show_pr_status(self, text: str, kind: str = "neutral") -> None:
        """The foot's ONE status line: *kind* neutral
        (mono grey), warn (``PANEL_WARN_STYLE``) or error (red).  It wraps
        to at most two lines; past that the second line is elided and the
        whole sentence is the tooltip."""
        lbl = self.pipeline_status_label
        style = _PR_STATUS_STYLES.get(kind, MONO_LABEL_STYLE)
        lbl.setStyleSheet(style)
        text = "" if text is None else str(text)
        lbl.setToolTip(text)
        lbl.setText(_two_lines(lbl, text))
        boxed = 0 if style is MONO_LABEL_STYLE else PR_STATUS_BOX_PAD
        lbl.setMaximumHeight(2 * lbl.fontMetrics().lineSpacing() + 4 + boxed)
        self._pr_status_kind = kind if kind in _PR_STATUS_STYLES else "neutral"

    # ------------------------------------------------------------------
    # 2b. The settings column
    # ------------------------------------------------------------------

    def _build_pr_settings(self, lay: QVBoxLayout) -> None:
        """The settings column, built from the shared ``.frm4`` helpers
        (``ui_shell/_recon_settings_widgets``) on Step 3's OWN widgets.

        Every widget keeps its name, range, step, decimals and default
        (``batch_sb``'s range is Step 4's 1-1024); plain names
        over grey engine names through ``STEP3_SETTING_KEYS``; the seven
        groups closed; Cellpose's two thresholds, "Compute masks" (OFF),
        the indirect method and the six flow-consistency widgets built and
        HIDDEN.  Until R3 the column sits where the two
        parameter boxes were.  Controller-free (the parity test builds
        ``_build_ui`` without a controller).
        """
        self._pr_engine_labels = {}
        self._pr_settings = QWidget()
        self._pr_settings.setStyleSheet(PANEL_CHECKBOX_STYLE)
        col = QVBoxLayout(self._pr_settings)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        col.addLayout(_make_subsection_header(
            self, _RT.SEG_SETTINGS_TITLE,
            guide_filename=_PR_SETTINGS_GUIDE,
        ))
        self.pr_settings_desc = QLabel(_PRT.S3_SETTINGS_DESC_NO_RUN)
        self.pr_settings_desc.setWordWrap(True)
        self.pr_settings_desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        col.addWidget(self.pr_settings_desc)

        sf = QHBoxLayout()
        sf.setSpacing(8)
        sf.addWidget(QLabel(_RT.SEG_START_FROM_LABEL))
        self.start_from_combo = QComboBox()
        self.start_from_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.start_from_combo.setMinimumContentsLength(10)
        self.start_from_combo.addItem(_RT.SEG_PRESET_EPICYCLE, "epicycle")
        self.start_from_combo.addItem(_RT.SEG_PRESET_USEG3D, "useg3d")
        sf.addWidget(self.start_from_combo, 1)
        col.addLayout(sf)
        self.pr_last_run_note = QLabel(_PRT.S3_LAST_RUN_NOTE)
        self.pr_last_run_note.setWordWrap(True)
        self.pr_last_run_note.setStyleSheet(DIM_LABEL_STYLE)
        self.pr_last_run_note.setVisible(False)
        col.addWidget(self.pr_last_run_note)

        # --- Inference settings (open) ---
        self._pr_inf_collapsible = collapsible(
            _RT.SEG_INFERENCE_TITLE, engine_object_name=_PR_ENGINE_OBJECT)
        inf = QWidget()
        inf_lay = QVBoxLayout(inf)
        inf_lay.setContentsMargins(4, 4, 0, 4)
        inf_lay.setSpacing(6)
        g = frm4()
        self.chan1_combo = QComboBox()
        self.chan1_combo.addItems(list(_PRT.S3_CHAN1_ITEMS))
        self.chan1_combo.setCurrentIndex(2)
        self.chan2_combo = QComboBox()
        self.chan2_combo.addItems(list(_PRT.S3_CHAN2_ITEMS))
        self.chan2_combo.setCurrentIndex(1)
        self.diameter_sb = QDoubleSpinBox()
        self.diameter_sb.setRange(0, 500)
        self.diameter_sb.setSpecialValueText(_RT.SEG_CELL_WIDTH_AUTO)
        self.diameter_sb.setSuffix(_PRT.S3_CELL_WIDTH_SUFFIX)
        self.diameter_sb.setValue(0)
        for r, (text, w) in enumerate(((_RT.SEG_MEMBRANE_LABEL,
                                        self.chan1_combo),
                                       (_RT.SEG_NUCLEI_LABEL,
                                        self.chan2_combo),
                                       (_RT.SEG_CELL_WIDTH_LABEL,
                                        self.diameter_sb))):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            g.addWidget(lbl, r, 0)
            g.addWidget(field(w), r, 1)
        # The width's two lines ("≈ N pixels at M µm per pixel" and the
        # no-voxel warning, R2b part 2) in the row right under it, above
        # batch size; the controller's ``ensure_cell_width_lines`` finds them
        g.addWidget(_cw.make_cell_width_lines(self), 3, 0, 1, 2)
        # Batch size at the TOP level (the owner's 2026-10-06 addition): the
        # plain name over the grey engine name, as an Advanced row
        self.batch_sb = QSpinBox()
        self.batch_sb.setRange(1, 1024)
        self.batch_sb.setValue(128)
        self._pr_adv_row(g, "batch_sb", self.batch_sb)
        inf_lay.addLayout(g)
        ticks = QHBoxLayout()
        self.norm_cb = QCheckBox(_RT.SEG_NORMALIZE_LABEL)
        self.norm_cb.setChecked(True)
        self.rescale_cb = QCheckBox(_RT.SEG_RESCALE_LABEL)
        self.rescale_cb.setChecked(True)
        ticks.addWidget(self.norm_cb, 1)
        ticks.addWidget(self.rescale_cb, 1)
        inf_lay.addLayout(ticks)
        self.pr_unused_knobs_note = QLabel(_PRT.S3_UNUSED_KNOBS_NOTE)
        self.pr_unused_knobs_note.setWordWrap(True)
        self.pr_unused_knobs_note.setStyleSheet(DIM_LABEL_STYLE)
        inf_lay.addWidget(self.pr_unused_knobs_note)
        self._build_pr_hidden_knobs(inf_lay)
        self._pr_inf_collapsible.addWidget(inf)
        self._pr_inf_collapsible.expand(animate=False)
        col.addWidget(self._pr_inf_collapsible)

        # --- Advanced (closed) ---
        self._pr_adv_collapsible = collapsible(
            _RT.SEG_ADVANCED_TITLE, _PRT.S3_ADVANCED_ENGINE,
            engine_object_name=_PR_ENGINE_OBJECT)
        adv = QWidget()
        adv_lay = QVBoxLayout(adv)
        adv_lay.setContentsMargins(4, 4, 0, 4)
        adv_lay.setSpacing(6)
        self._build_pr_advanced(adv_lay)
        self._pr_adv_collapsible.addWidget(adv)
        col.addWidget(self._pr_adv_collapsible)

        # The two old preset buttons: built and HIDDEN; the
        # controller keeps the first wired to the one writer and
        # disconnects nothing from the second because it connects nothing.
        self._pr_hidden_buttons = QWidget()
        self._pr_hidden_buttons.setVisible(False)
        hb = QHBoxLayout(self._pr_hidden_buttons)
        hb.setContentsMargins(0, 0, 0, 0)
        self.epicycle_defaults_btn = QPushButton(_RT.SEG_PRESET_EPICYCLE)
        self.useg3d_defaults_btn = QPushButton(_RT.SEG_PRESET_USEG3D)
        hb.addWidget(self.epicycle_defaults_btn)
        hb.addWidget(self.useg3d_defaults_btn)
        col.addWidget(self._pr_hidden_buttons)

        self._apply_pr_help_lines()
        lay.addWidget(self._pr_settings)

    def _pr_adv_row(self, grid: QGridLayout, widget_name: str, widget):
        """One ``.frm4`` row with Step 3's names (``S3_ADVANCED_LABELS``);
        its label is kept in ``_pr_engine_labels``."""
        return adv_row(grid, widget_name, widget, self._pr_engine_labels,
                       table=_PRT.S3_ADVANCED_LABELS)

    def _build_pr_hidden_knobs(self, lay: QVBoxLayout) -> None:
        """Cellpose's two mask thresholds and "Compute masks", built and
        never shown: the thresholds act only on the
        masks, and the masks are not computed."""
        self._hidden_knobs = QWidget()
        self._hidden_knobs.setVisible(False)
        hk = QVBoxLayout(self._hidden_knobs)
        hk.setContentsMargins(0, 0, 0, 0)
        self.flow_thresh = QDoubleSpinBox()
        self.flow_thresh.setRange(0, 3.0)
        self.flow_thresh.setSingleStep(0.1)
        self.flow_thresh.setValue(0.4)
        self.cellprob_thresh = QDoubleSpinBox()
        self.cellprob_thresh.setRange(-10, 10)
        self.cellprob_thresh.setSingleStep(0.5)
        self.cellprob_thresh.setValue(0.0)
        self.compute_masks_cb = QCheckBox()
        self.compute_masks_cb.setChecked(False)
        hk.addWidget(self.flow_thresh)
        hk.addWidget(self.cellprob_thresh)
        hk.addWidget(self.compute_masks_cb)
        lay.addWidget(self._hidden_knobs)

    def _build_pr_advanced(self, adv_lay: QVBoxLayout) -> None:
        g = frm4()
        self.method_combo = QComboBox()
        self.method_combo.addItem(_RT.SEG_METHOD_DIRECT, "direct")
        self.n_cpu_sb = QSpinBox()
        self.n_cpu_sb.setRange(1, 64)
        self.n_cpu_sb.setValue(6)
        self._pr_adv_row(g, "method_combo", self.method_combo)
        self._pr_adv_row(g, "n_cpu_sb", self.n_cpu_sb)
        adv_lay.addLayout(g)

        join = QLabel(_RT.SEG_JOIN_TITLE)
        join.setStyleSheet(SECTION_HEADER_STYLE)
        adv_lay.addWidget(join)
        join_desc = QLabel(_RT.SEG_JOIN_DESC)
        join_desc.setWordWrap(True)
        join_desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        adv_lay.addWidget(join_desc)

        self._make_pr_group_widgets()
        self._pr_adv_groups = {}
        for plain, engine, names in _RT.STEP4_ADVANCED_GROUPS:
            coll = collapsible(plain, engine,
                               engine_object_name=_PR_ENGINE_OBJECT)
            body = QWidget()
            b_lay = QVBoxLayout(body)
            b_lay.setContentsMargins(4, 2, 0, 4)
            b_lay.setSpacing(5)
            gg = frm4()
            if engine == "size_filters":
                self._pr_adv_row(gg, "skip_postprocess_cb",
                                 self.skip_postprocess_cb)
                # A note, not a warning: dim in Step 3
                self.seg_size_off_note = QLabel(_RT.SEG_SIZE_OFF_NOTE)
                self.seg_size_off_note.setWordWrap(True)
                self.seg_size_off_note.setStyleSheet(DIM_LABEL_STYLE)
                gg.addWidget(self.seg_size_off_note, gg.rowCount(), 0, 1, 2)
                self.seg_size_desc = QLabel(_RT.SEG_SIZE_DESC)
                self.seg_size_desc.setWordWrap(True)
                self.seg_size_desc.setStyleSheet(DIM_LABEL_STYLE)
                gg.addWidget(self.seg_size_desc, gg.rowCount(), 0, 1, 2)
                names = [n for n in names if n != "skip_postprocess_cb"]
            for n in names:
                self._pr_adv_row(gg, n, getattr(self, n))
            if "ps_mask_thresh_sb" in names:
                # why the two fields are grey, on screen (Qt shows no
                # tooltip on a disabled widget)
                self.ps_flows_off_note = QLabel(_RT.PS_FLOWS_OFF_NOTE)
                self.ps_flows_off_note.setWordWrap(True)
                self.ps_flows_off_note.setStyleSheet(DIM_LABEL_STYLE)
                gg.addWidget(self.ps_flows_off_note, gg.rowCount(), 0, 1, 2)
            b_lay.addLayout(gg)
            coll.addWidget(body)
            adv_lay.addWidget(coll)
            self._pr_adv_groups[engine] = coll

        self._build_pr_hidden_recon(adv_lay)

        self.ps_smooth_flows_cb.toggled.connect(self._update_pr_flow_controls)
        self._update_pr_flow_controls()
        self.skip_postprocess_cb.toggled.connect(self._update_pr_size_off_note)
        self._update_pr_size_off_note()

    def _update_pr_flow_controls(self, *_):
        enabled = self.ps_smooth_flows_cb.isChecked()
        self.ps_mask_thresh_sb.setEnabled(enabled)
        self.ps_soft_weight_cb.setEnabled(enabled)
        self.ps_flows_off_note.setVisible(not enabled)

    def _update_pr_size_off_note(self, *_):
        self.seg_size_off_note.setVisible(self.skip_postprocess_cb.isChecked())

    def _make_pr_group_widgets(self) -> None:
        """The 38 group controls + ``do_mp_cb`` + ``skip_postprocess_cb``:
        Step 3's names, ranges, steps, decimals and defaults, unchanged
        (pinned by ``test_step4_useg3d_defaults_parity.py``)."""
        def dspin(lo, hi, step, val, dec=None):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            if dec is not None:
                sb.setDecimals(dec)
            sb.setValue(val)
            return sb

        def ispin(lo, hi, val):
            sb = QSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(val)
            return sb

        def check(val):
            cb = QCheckBox()
            cb.setChecked(val)
            return cb

        # Smooth before joining
        self.ps_smooth_probs_cb = check(False)
        self.ps_smooth_flows_cb = check(False)
        self.ps_sigma_perp_sb = dspin(0, 20.0, 0.5, 3.0)
        self.ps_sigma_inplane_sb = dspin(0, 20.0, 0.5, 1.0)
        self.ps_mask_thresh_sb = dspin(0, 1.0, 0.05, 0.2, 2)
        self.ps_soft_weight_cb = check(True)
        # Follow the flows
        self.gd_decay_sb = dspin(0, 1, 0.001, 0.001, 3)
        self.gd_iters_sb = ispin(1, 1000, 25)
        self.gd_momenta_sb = dspin(0, 1, 0.01, 0.98)
        self.gd_renorm_cb = check(True)
        self.do_mp_cb = check(True)
        self.gd_tile_z_sb = ispin(16, 512, 128)
        self.gd_tile_y_sb = ispin(16, 1024, 256)
        self.gd_tile_x_sb = ispin(16, 1024, 256)
        self.gd_tile_overlap_sb = dspin(0, 0.5, 0.05, 0.25)
        # Combine the probabilities
        self.ccp_ksize_sb = ispin(1, 50, 25)
        self.ccp_alpha_sb = dspin(0, 10, 0.1, 0.5)
        self.ccp_smooth_sigma_sb = dspin(0, 20, 0.1, 0.8)
        self.ccp_threshold_level_sb = ispin(0, 10, 1)
        self.ccp_threshold_n_levels_sb = ispin(2, 20, 3)
        self.ccp_prob_thresh_sb = dspin(-1.0, 1.0, 0.05, -1.0, 2)
        self.ccp_prob_thresh_sb.setSpecialValueText(_RT.SEG_PROB_THRESH_AUTO)
        self.ccp_min_prob_thresh_sb = dspin(0.0, 1.0, 0.05, 0.0, 2)
        self.ccp_apply_one_d_p_cb = check(True)
        # Clean the mask
        self.post_small_obj_sb = ispin(0, 100000, 10000)
        self.post_fill_cb = check(True)
        self.post_closing_sb = ispin(0, 50, 1)
        self.post_dilation_sb = ispin(0, 50, 1)
        self.post_extra_erode_sb = ispin(0, 50, 0)
        # Combine the flows
        self.ccg_ksize_sb = ispin(1, 50, 25)
        self.ccg_alpha_sb = dspin(0, 10, 0.1, 0.5)
        self.ccg_smooth_sigma_sb = dspin(0, 20, 0.5, 1.5)
        self.ccg_post_sigma_sb = dspin(0, 20, 0.1, 0.5)
        # Group into cells
        self.cc_min_area_sb = ispin(0, 10000, 5)
        self.cc_thresh_factor_sb = dspin(-5.0, 5.0, 0.1, 0.0)
        self.cc_smooth_sigma_sb = dspin(0, 20, 0.5, 1.0)
        # Size limits
        self.skip_postprocess_cb = check(True)
        self.sf_min_size_sb = ispin(0, 100000, 200)
        self.sf_max_size_sb = ispin(-1, 10000000, -1)
        self.sf_max_size_sb.setSpecialValueText(_RT.SEG_MAX_SIZE_NONE)
        self.sf_max_size_factor_sb = dspin(0, 100, 1, 10)
        self.sf_do_stats_cb = check(True)

    def _build_pr_hidden_recon(self, adv_lay: QVBoxLayout) -> None:
        """The flow-consistency group and the indirect method: built,
        HIDDEN and inert -- they only ever hold
        ``ReconstructionConfig()``'s values, which the collector still
        reads.  The method list holds only "Direct", so nothing shows the
        indirect container."""
        dtform_options = list(_DTFORM_OPTIONS)
        self._hidden_flow_consistency = QWidget()
        self._hidden_flow_consistency.setVisible(False)
        fc = QVBoxLayout(self._hidden_flow_consistency)
        fc.setContentsMargins(0, 0, 0, 0)
        self.do_flow_remove_cb = QCheckBox()
        self.flow_threshold_sb = QDoubleSpinBox()
        self.flow_threshold_sb.setRange(0, 1)
        self.flow_threshold_sb.setSingleStep(0.05)
        self.flow_threshold_sb.setValue(0.8)
        self.fc_dtform_combo = QComboBox()
        self.fc_dtform_combo.addItems(dtform_options)
        self.fc_edt_percentile_sb = QDoubleSpinBox()
        self.fc_edt_percentile_sb.setRange(0, 1.0)
        self.fc_edt_percentile_sb.setDecimals(3)
        self.fc_edt_percentile_sb.setSingleStep(0.005)
        self.fc_edt_percentile_sb.setValue(0.01)
        self.fc_power_dist_sb = QDoubleSpinBox()
        self.fc_power_dist_sb.setRange(-1, 1.0)
        self.fc_power_dist_sb.setDecimals(2)
        self.fc_power_dist_sb.setSingleStep(0.1)
        self.fc_power_dist_sb.setValue(-1.0)
        self.fc_smooth_skel_sb = QDoubleSpinBox()
        self.fc_smooth_skel_sb.setRange(0, 20)
        self.fc_smooth_skel_sb.setSingleStep(0.5)
        self.fc_smooth_skel_sb.setValue(3.0)
        for w in (self.do_flow_remove_cb, self.flow_threshold_sb,
                  self.fc_dtform_combo, self.fc_edt_percentile_sb,
                  self.fc_power_dist_sb, self.fc_smooth_skel_sb):
            fc.addWidget(w)
        adv_lay.addWidget(self._hidden_flow_consistency)

        self._indirect_method_container = QWidget()
        im = QVBoxLayout(self._indirect_method_container)
        im.setContentsMargins(0, 0, 0, 0)
        self.im_dtform_combo = QComboBox()
        self.im_dtform_combo.addItems(dtform_options)
        self.im_smooth_binary_sb = QDoubleSpinBox()
        self.im_smooth_binary_sb.setRange(0, 20)
        self.im_smooth_binary_sb.setSingleStep(0.5)
        self.im_smooth_binary_sb.setValue(1.0)
        self.im_smooth_skel_sb = QDoubleSpinBox()
        self.im_smooth_skel_sb.setRange(0, 20)
        self.im_smooth_skel_sb.setSingleStep(0.5)
        self.im_smooth_skel_sb.setValue(3.0)
        self.im_edt_percentile_sb = QDoubleSpinBox()
        self.im_edt_percentile_sb.setRange(0, 1.0)
        self.im_edt_percentile_sb.setDecimals(3)
        self.im_edt_percentile_sb.setSingleStep(0.005)
        self.im_edt_percentile_sb.setValue(0.01)
        self.im_power_dist_sb = QDoubleSpinBox()
        self.im_power_dist_sb.setRange(-1, 1.0)
        self.im_power_dist_sb.setDecimals(2)
        self.im_power_dist_sb.setSingleStep(0.1)
        self.im_power_dist_sb.setValue(-1.0)
        for w in (self.im_dtform_combo, self.im_smooth_binary_sb,
                  self.im_smooth_skel_sb, self.im_edt_percentile_sb,
                  self.im_power_dist_sb):
            im.addWidget(w)
        self._indirect_method_container.setVisible(False)
        adv_lay.addWidget(self._indirect_method_container)

    def _apply_pr_help_lines(self) -> None:
        """Every settings control gets its help line as tooltip
        (``S3_HELP_LINES``: the shared lines through the alias map, Step 3's
        own over them).  The hidden knobs have none."""
        for name, line in _PRT.S3_HELP_LINES.items():
            w = getattr(self, name, None)
            if w is not None and hasattr(w, "setToolTip"):
                w.setToolTip(line)

    def show_pr_last_run_note(self, visible: bool, text=None) -> None:
        """The dim note under Start from: shown while "This
        run's last reconstruction" is chosen and its manifest is older
        than R2c.  The same slot carries a refused pick's sentence
        (*text*, a ``_step3_pr_text`` constant); without one it reads
        ``S3_LAST_RUN_NOTE``."""
        self.pr_last_run_note.setText(text or _PRT.S3_LAST_RUN_NOTE)
        self.pr_last_run_note.setVisible(bool(visible))

    def set_pr_settings_desc(self, has_last_run: bool) -> None:
        """The description with or without its run clause."""
        self.pr_settings_desc.setText(
            _PRT.S3_SETTINGS_DESC if has_last_run
            else _PRT.S3_SETTINGS_DESC_NO_RUN)

    # ------------------------------------------------------------------
    # Bucket helpers
    # ------------------------------------------------------------------

    def refresh_patch_state_list(
        self, all_statuses: dict, predicting: set = None,
        reconstructing: set = None, axes_by_patch: dict = None,
        ticked=None, missing=None, info_by_patch: dict = None,
        made_by_by_patch: dict = None, shapes: dict = None,
        spacing_by_patch: dict = None, unsized=None,
    ) -> None:
        """Rebuild the Patches list: one row per patch, painted by the
        shared row painter from the record each item carries in
        ``ROLE_RECORD`` (``_step3_rows.build_row``).

        Every item keeps ``Qt.UserRole`` = the patch name, ``UserRole + 1``
        = its status (or the running kind) and ``text()`` = today's padded
        row (row 24's two goldens).  The rebuild runs under
        ``blockSignals``: it is not a user's tick or selection.

        Parameters
        ----------
        all_statuses : dict
            ``{patch_name: status_str}`` from the state manager.
        predicting, reconstructing : set, optional
            Patches a job is working on (the row shows the running words).
        axes_by_patch : dict, optional
            ``{patch_name: axes_str}`` -- each patch's detected axis order.
        ticked : iterable, optional
            The ticked patches (the run's ``ticked`` key); re-applied here.
        missing : iterable, optional
            Patches whose file is gone (row 27): amber line, no tick.
        info_by_patch : dict, optional
            ``{patch_name: reconstruction manifest payload}`` for the shape
            line; a patch without one gets no shape line.
        shapes, spacing_by_patch : dict, optional
            A3: ``{patch: raw shape}`` and ``{patch: patch_spacing entry}``
            -- the metadata line and the voxel-size line of each row (a
            patch absent from ``spacing_by_patch`` shows the amber
            "voxels"); a caller that passes neither gets the rows as they
            were.
        unsized : set, optional
            B + S1: the patches whose EFFECTIVE voxel size (own, else QC
            Settings') is none -- only they carry the "cell width falls back
            to pixels" tooltip; ``None`` keeps it on every row without a
            stored size.
        made_by_by_patch : dict, optional
            ``{patch_name: (chip text, active)}`` (R6,
            ``_step3_rows.made_by_fields``): the "made by" chip under the
            shape line, amber-bordered when not active, and the ⓘ; a patch
            without one gets neither.
        """
        lst = self._patch_state_list
        _cur = lst.currentItem()
        keep = _cur.data(Qt.UserRole) if (
            _cur is not None and _cur.isSelected()) else None
        locked = bool(getattr(self, "_pr_surface_locked", False))
        predicting = set(predicting or ())
        reconstructing = set(reconstructing or ())
        axes_by_patch = axes_by_patch or {}
        ticked = set(ticked or ())
        missing = set(missing or ())
        info_by_patch = info_by_patch or {}
        made_by_by_patch = made_by_by_patch or {}
        shapes = shapes or {}
        names = sorted(all_statuses.keys())
        max_name = max((len(n) for n in names), default=10)
        max_status = max(
            (len(s) for s in all_statuses.values()), default=12
        )
        _was_blocked = lst.blockSignals(True)
        try:
            lst.clear()
            for name in names:
                status = all_statuses[name]
                running = None
                if name in predicting:
                    running = "segmenting"
                    text = (f" {name.ljust(max_name)}  "
                            f"{_pipe.ROW_SEGMENTING.ljust(max_status)}")
                elif name in reconstructing:
                    running = "reconstructing"
                    text = (f" {name.ljust(max_name)}  "
                            f"{_pipe.ROW_RECONSTRUCTING.ljust(max_status)}")
                else:
                    text = _format_patch_state_row(
                        name, status, axes_by_patch.get(name),
                        _STATUS_DOTS.get(status, ""), max_name, max_status,
                    )
                gone = name in missing
                made_by, made_active = made_by_by_patch.get(name,
                                                            (None, True))
                extra = {}
                if spacing_by_patch is not None:
                    extra["spacing"] = spacing_by_patch.get(name)
                rec = _rows.build_row(
                    name, status, axes_by_patch.get(name),
                    info_by_patch.get(name), ticked=name in ticked,
                    missing=gone, running=running, made_by=made_by,
                    made_by_active=made_active,
                    source_shape=shapes.get(name),
                    pixel_fallback=(unsized is None or name in unsized),
                    **extra)
                rec = self._with_pr_tier_counts(rec, name)
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, name)
                item.setData(Qt.UserRole + 1, running or status)
                item.setData(ROLE_RECORD, rec)
                put = getattr(self, "_apply_patch_thumb", None)
                if put is not None:     # A1: the picture survives a rebuild
                    put(item)
                if rec.get("row_tip"):
                    item.setToolTip(rec["row_tip"])
                item.setCheckState(Qt.Checked if name in ticked
                                   else Qt.Unchecked)
                item.setFlags(_pr_row_flags(item, locked))
                lst.addItem(item)
                if keep is not None and name == keep:
                    # the highlighted row (View's) survives the rebuild
                    lst.setCurrentItem(item)
                    item.setSelected(True)
        finally:
            lst.blockSignals(_was_blocked)

    def refresh_predicted_bucket(
        self, patch_names: list, statuses: dict, predicting: set = None,
    ) -> None:
        """Backward-compat: calls refresh_patch_state_list instead."""
        self.refresh_patch_state_list(statuses, predicting=predicting)

    def refresh_reconstructed_bucket(self, patch_names: list, statuses: dict) -> None:
        """Backward-compat: no-op (refresh_patch_state_list handles both)."""
        pass

    def _on_model_source_changed(self, index: int) -> None:
        """Hidden-state combo signal handler — retained for back-compat.

        2026-05-21 Stage 3: the source combo is now hidden state.  The
        old `_import_widget` / `_pretrained_widget` show/hide logic is
        removed because those containers no longer exist as user-
        facing UI (replaced by the Active model indicator + picker).
        The combo still drives `_on_pretrained_changed`'s mode gate
        via `self._model_source_mode`, so the controller's handler
        still fires when source changes — but this UI-level handler
        has nothing visual to update.
        """
        return

    # ------------------------------------------------------------------
    # 3. Curation panel (bucket-driven, QSplitter two-column)
    # ------------------------------------------------------------------

    def _create_curation_panel(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(SPLITTER_STYLE)

        # -- LEFT: Curate Bucket + Actions + QC Controls (stacked) --
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(6, 6, 6, 6)
        left_lay.setSpacing(4)

        # --- Curate section ---
        left_lay.addLayout(_make_subsection_header(
            self, "Curate",
            guide_filename="step3_curation_curate.md",
        ))
        _desc = QLabel("Open 3D annotator to edit labels and assign triage tiers.")
        _desc.setWordWrap(True)
        _desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        left_lay.addWidget(_desc)

        # D2 (perf push 2026-05-09): workspace-level toggle for
        # downsample-on-load.  Sits above the bucket list so it's
        # visible at a glance and a single click flips the mode.
        # Affects FUTURE patch loads only — existing in-memory patch
        # keeps its current resolution until the user re-clicks it
        # in the bucket below.  Canonical mask is preserved at full
        # resolution on disk via NN-upsample-on-save.
        self.downsample_mode_cb = QCheckBox("Downsample patches 2× on load")
        self.downsample_mode_cb.setToolTip(
            "Halves volumes in all 3 dimensions for snappier interaction "
            "on large stacks.  Curation runs at downsampled resolution; "
            "the canonical mask on disk is preserved at full resolution "
            "via NN-upsample on save.  Re-click the patch in the bucket "
            "below to apply the new mode."
        )
        left_lay.addWidget(self.downsample_mode_cb)

        # Curation bucket (replaces old curation_result_combo); brick V2:
        # painted by the shared row painter with the patch's picture
        self._build_curation_bucket(left, left_lay)

        self.open_3d_annotator_button = QPushButton("Open 3D Curation Annotator")
        self.open_3d_annotator_button.setStyleSheet("font-weight: bold;")
        left_lay.addWidget(self.open_3d_annotator_button)

        # Save / Undo curation row
        curation_action_row = QHBoxLayout()
        self.save_curation_btn = QPushButton("Save Curation")
        self.save_curation_btn.setToolTip(
            "Persist current mask and CSV to disk"
        )
        self.save_curation_btn.setEnabled(False)
        self.undo_curation_btn = QPushButton("Undo")
        self.undo_curation_btn.setToolTip("Undo last curation action")
        self.undo_curation_btn.setEnabled(False)
        self.undo_curation_btn.setMaximumWidth(60)
        curation_action_row.addWidget(self.save_curation_btn)
        curation_action_row.addWidget(self.undo_curation_btn)
        left_lay.addLayout(curation_action_row)

        commit_row = QHBoxLayout()
        self.accept_annotation_button = QPushButton("Accept Volume")
        self.finalize_labels_button = QPushButton("Finalize Labels")
        commit_row.addWidget(self.accept_annotation_button)
        commit_row.addWidget(self.finalize_labels_button)
        left_lay.addLayout(commit_row)

        self.reset_to_predicted_btn = QPushButton("Reset to Predicted")
        self.reset_to_predicted_btn.setToolTip(
            "Move selected curation patches back to predicted status\n"
            "for re-reconstruction with a different model or parameters.\n"
            "Existing reconstruction artifacts will be overwritten."
        )
        left_lay.addWidget(self.reset_to_predicted_btn)

        # Separator
        sep1 = QLabel("")
        sep1.setFixedHeight(1)
        sep1.setStyleSheet(SEPARATOR_LINE_STYLE)
        left_lay.addWidget(sep1)

        # --- QC Feature Controls ---
        left_lay.addLayout(_make_subsection_header(
            self, "QC Features",
            guide_filename="step3_curation_qc_features.md",
        ))
        _desc = QLabel("Extract morphology features, cluster, and filter labels by quality.")
        _desc.setWordWrap(True)
        _desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        left_lay.addWidget(_desc)

        extract_row = QHBoxLayout()
        self.extract_features_btn = QPushButton("Extract")
        self.extract_features_btn.setToolTip("Extract QC features for the selected patch")
        # extract_all_features_btn hidden 2026-05-17 (EDA compute buttons branch) —
        # batch extraction across all reconstructed patches is a once-per-project
        # action that no longer earns its row slot.  Widget kept alive (parented
        # to self) so _QCFeatureBatchWorker + controller wiring stay intact and
        # can be reinstated by un-hiding + re-adding to a layout.
        self.extract_all_features_btn = QPushButton("Extract All", parent=self)
        self.extract_all_features_btn.setToolTip("Extract features for all reconstructed patches")
        self.extract_all_features_btn.setVisible(False)
        self.qc_settings_btn = QPushButton("Settings...")
        self.qc_settings_btn.setToolTip("Configure spacing, erosion, neighbor radius, and score method")
        self.qc_settings_btn.setMaximumWidth(70)
        extract_row.addWidget(self.extract_features_btn, 1)
        extract_row.addWidget(self.qc_settings_btn)

        # Vertical separator between extract group and recompute group.
        # The recompute buttons reuse the in-memory df without re-running
        # the feature pipeline, so visually grouping them apart from
        # Extract clarifies their lower-cost nature.
        compute_sep = QFrame()
        compute_sep.setFrameShape(QFrame.VLine)
        compute_sep.setFrameShadow(QFrame.Sunken)
        compute_sep.setStyleSheet("color: #555;")
        extract_row.addWidget(compute_sep)

        # Recompute buttons — operate on the in-memory df using the
        # current QC settings.  Each filters to active rows (ghost rows
        # excluded — see _run_clustering rationale) and persists to
        # disk on completion.
        self.pca_btn = QPushButton("PCA", parent=self)
        self.pca_btn.setToolTip(
            "Recompute PCA (PC1..PCN) on the in-memory features using "
            "current Settings.  Does not re-extract features."
        )
        self.cluster_btn = QPushButton("Cluster", parent=self)
        self.cluster_btn.setToolTip(
            "Run clustering on the in-memory features using current "
            "Settings (algorithm, n_components, etc.)."
        )
        self.umap_btn = QPushButton("UMAP", parent=self)
        self.umap_btn.setToolTip(
            "Recompute UMAP_1 / UMAP_2 on the in-memory features."
        )
        self.tsne_btn = QPushButton("tSNE", parent=self)
        self.tsne_btn.setToolTip(
            "Recompute tSNE_1 / tSNE_2 on the in-memory features."
        )
        extract_row.addWidget(self.pca_btn)
        extract_row.addWidget(self.cluster_btn)
        extract_row.addWidget(self.umap_btn)
        extract_row.addWidget(self.tsne_btn)
        left_lay.addLayout(extract_row)

        # qc_scope_combo removed 2026-05-03 (drop-population-mode
        # Commit 3/6).  Population mode is gone; the QC view is always
        # single-patch.  The combo, its row, and the "Scope:" label
        # are no longer rendered.

        # QC summary
        from ._qc_plot_text import QC_SUMMARY_NONE, CURATION_STATUS_NONE
        self.qc_summary_label = QLabel(QC_SUMMARY_NONE)
        self.qc_summary_label.setStyleSheet(STATUS_LABEL_STYLE)
        self.qc_summary_label.setWordWrap(True)
        left_lay.addWidget(self.qc_summary_label)

        # Tier assignment widgets constructed here (they must exist before
        # _apply_theme iterates button attrs); the QC plot area takes them
        # into its pinned action row in the right column (Step 3 plot pass
        # P2), which sets their words and their look.
        self.tier_assign_combo = QComboBox(parent=self)
        self.tier_assign_combo.addItems(["curated", "good", "review", "bad", "garbage"])
        self.assign_tier_btn = QPushButton(parent=self)
        self.assign_tier_btn.setEnabled(False)
        self.undo_tier_btn = QPushButton(parent=self)
        self.undo_tier_btn.setEnabled(False)

        # Commit Triage button constructed here, but added BELOW the
        # histogram panel later in this method so it sits at the
        # bottom of the left column (Phase 1.6).
        self.commit_triage_btn = QPushButton("Commit Triage", parent=self)
        self.commit_triage_btn.setToolTip(
            "Discard garbage labels, save tier assignments"
        )
        # NOTE: not added to left_lay yet — see below the histogram panel.
        # Hidden CSV compat widget — see Phase 1.5 fix above.
        self.save_qc_csv_btn = QPushButton(
            "CSV (hidden — see Phase 1.5)", parent=self,
        )
        self.save_qc_csv_btn.setVisible(False)

        # Advanced Filters section retained as instance attributes but
        # NOT added to the layout (2026-05-04, Phase 1.5).  The histogram
        # panel below will absorb threshold/auto-discard workflows in
        # Phase 2 (drag the bad threshold on `area` to filter small
        # artifacts, etc.).  Hidden widgets keep controller callbacks
        # (apply_threshold_btn → _apply_qc_threshold) functional but
        # unreachable until we delete those wiring sites in Phase 3.
        # Parent both to `self` so Qt doesn't garbage-collect them out
        # from under the controller wiring.  They're never shown, but
        # apply_threshold_btn must remain alive for _apply_theme + the
        # controller's clicked.connect to function.
        self._adv_filters_collapsible = QCollapsible(
            "Advanced Filters", parent=self,
        )
        self._adv_filters_collapsible.setVisible(False)
        adv_filt_w = QWidget(parent=self)
        adv_filt_w.setVisible(False)
        adv_filt_lay = QVBoxLayout(adv_filt_w)
        adv_filt_lay.setContentsMargins(4, 4, 4, 4)
        adv_filt_lay.setSpacing(4)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Feature:"))
        self.threshold_feature_combo = QComboBox()
        self.threshold_feature_combo.setToolTip("Select feature to threshold")
        thresh_row.addWidget(self.threshold_feature_combo, 1)
        adv_filt_lay.addLayout(thresh_row)

        range_grid = QGridLayout()
        range_grid.setHorizontalSpacing(6)
        range_grid.setVerticalSpacing(2)
        self.threshold_min_sb = QDoubleSpinBox()
        self.threshold_min_sb.setRange(-1e6, 1e6)
        self.threshold_min_sb.setDecimals(2)
        self.threshold_max_sb = QDoubleSpinBox()
        self.threshold_max_sb.setRange(-1e6, 1e6)
        self.threshold_max_sb.setDecimals(2)
        self.threshold_max_sb.setValue(1e6)
        range_grid.addWidget(QLabel("Min:"), 0, 0)
        range_grid.addWidget(self.threshold_min_sb, 0, 1)
        range_grid.addWidget(QLabel("Max:"), 1, 0)
        range_grid.addWidget(self.threshold_max_sb, 1, 1)
        adv_filt_lay.addLayout(range_grid)

        autodiscard_row = QHBoxLayout()
        autodiscard_row.addWidget(QLabel("Min Vol:"))
        self.min_volume_sb = QDoubleSpinBox()
        self.min_volume_sb.setRange(0, 1e6)
        self.min_volume_sb.setDecimals(0)
        self.min_volume_sb.setValue(0)
        self.min_volume_sb.setSuffix(" vox")
        self.min_volume_sb.setToolTip(
            "Labels below this volume are auto-discarded on Apply Filter.\n"
            "Set to 0 to disable."
        )
        autodiscard_row.addWidget(self.min_volume_sb, 1)
        adv_filt_lay.addLayout(autodiscard_row)

        self.apply_threshold_btn = QPushButton("Apply Filter")
        adv_filt_lay.addWidget(self.apply_threshold_btn)
        # NOTE: _adv_filters_collapsible is NOT added to left_lay.
        # See comment above.

        # Triage section — flat list and history list retained as
        # hidden compatibility widgets so the controller's existing
        # populator (_update_flagged_labels_list, curation_history.addItem
        # at controller.py:2379/2483/3931) still has live targets.
        # 2026-05-04 (Phase 1.5): tabbed wrapper removed; the histogram
        # panel takes the full vertical space at the bottom of the
        # left column with stretch factor 1.
        # Both QListWidgets parented to `self` so Qt keeps them alive
        # for the controller's .clear() / .addItem() calls.  Hidden so
        # they don't render or claim space.
        self._flagged_labels_list = QListWidget(parent=self)
        self._flagged_labels_list.setVisible(False)
        self._flagged_labels_list.setToolTip(
            "Hidden compatibility widget — Phase 1.5 of the triage "
            "redesign.  Phase 2 routes its data into the histogram panel."
        )
        self.curation_history = QListWidget(parent=self)
        self.curation_history.setVisible(False)

        # The triage histogram left this column for the QC plot area's
        # lower half (Step 3 plot pass P2); the patch list
        # above takes the stretch it left (owner, 2026-10-06).

        # Commit Triage row — sits at the bottom of the left column.
        # Workflow argument: the histogram is where you SET tier rules,
        # this button is where you COMMIT those rules destructively.
        commit_row = QHBoxLayout()
        commit_row.addWidget(self.commit_triage_btn)
        left_lay.addLayout(commit_row)

        # -- RIGHT: Status + QC Visualization (full height) --
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(6, 6, 6, 6)
        right_lay.setSpacing(4)

        # Curation status (constructed here so the controller can update
        # it before the right column finishes building, but added to the
        # layout below the QC plot — 2026-05-19 layout polish for
        # consistency with the rest of Step 3 where status text lives
        # near the action it reports on).
        self.curation_status_label = QLabel(CURATION_STATUS_NONE)
        self.curation_status_label.setStyleSheet(STATUS_LABEL_STYLE)
        self.curation_status_label.setWordWrap(True)

        # QC plot area (Step 3 plot pass P2): the
        # see-line + status slot + "Colours in viewer", the Workbench plot
        # in amber over a divider, the Histogram / Selection views, and the
        # pinned action row.  The subtitle sentence is gone:
        # the see-line's status slot carries the "how to select" hint.
        right_lay.addLayout(_make_subsection_header(
            self, "QC Plot",
            guide_filename="step3_curation_qc_plot.md",
        ))
        from ._qc_plot_area import QCPlotArea
        self.qc_plot_area = QCPlotArea(
            right,
            tier_combo=self.tier_assign_combo,
            assign_btn=self.assign_tier_btn,
            undo_btn=self.undo_tier_btn,
        )
        right_lay.addWidget(self.qc_plot_area, 1)
        # The names the controller and the tests have always read.
        self.qc_plot_widget = self.qc_plot_area.qc_plot_widget
        self._triage_histogram_panel = self.qc_plot_area.panel
        self.sync_viewer_colors_cb = self.qc_plot_area.colours_in_viewer_cb

        # Curation busy bar (2026-06-05; redesigned 2026-06-06) — a general
        # "Step 3 busy" indicator shared by every long-running curation op.
        # Renamed from patch_load_progress (it is no longer load-specific).
        # 2026-06-06: PERMANENT (always visible).  While IDLE the bar is a
        # static determinate-empty strip (setRange(0,100) + setValue(0), no
        # marquee).  While BUSY the controller flips it to INDETERMINATE
        # "scan mode" (setRange(0,0)) so the marquee shows "something is
        # happening".  The descriptive op text lives in curation_status_label
        # directly BELOW the bar (moved here next), which doubles as the
        # permanent patch-status line when idle.  Driven by the controller's
        # _busy_show / _busy_progress / _busy_hide helpers.
        self.curation_busy_bar = QProgressBar(parent=self)
        self.curation_busy_bar.setRange(0, 100)
        self.curation_busy_bar.setValue(0)
        self.curation_busy_bar.setTextVisible(True)
        # The op text lives in the status line under the bar; an idle bar
        # must not read "0%" (P2 audit LOW-17; _busy_show sets "" too).
        self.curation_busy_bar.setFormat("")
        self.curation_busy_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.curation_busy_bar.setVisible(True)  # permanent — static when idle
        right_lay.addWidget(self.curation_busy_bar)

        # Curation status label directly BELOW the busy bar (2026-06-06).  This
        # is the single permanent status line: while idle it shows the patch
        # status ("Curating: patch001"); while busy the controller's busy
        # helpers overwrite it with the descriptive per-op text ("Computing
        # PCA…", "Loading patch001 — downsampling…", etc.) and restore the idle
        # text on _busy_hide.  Constructed earlier (so the controller can set
        # text before the column finishes building); added to the layout here.
        right_lay.addWidget(self.curation_status_label)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([340, 680])

        outer.addWidget(splitter)

        # Backward-compat: hidden curation_result_combo (not in layout)
        self.curation_result_combo = QComboBox()

        return panel

    # ------------------------------------------------------------------
    # Curation bucket helper
    # ------------------------------------------------------------------

    def _build_curation_bucket(self, left: QWidget,
                               left_lay: QVBoxLayout) -> None:
        """Curation's box list (brick V2): painted by the shared row painter
        in ``MODE_WIDE`` with no tick and no eye -- today's content (the
        name, the status and its dots, the measured / edited marks, the
        missing line) beside the patch's picture -- in Step 3's colours.
        Selection, the click handler and the right-click menu are as
        before; a missing row's ``⋯`` opens the same menu."""
        self._curation_bucket = StackListWidget(left)
        self._curation_bucket.setSelectionMode(QListWidget.ExtendedSelection)
        self._curation_bucket.setMinimumHeight(80)
        self._curation_row_delegate = StackRowDelegate(
            MODE_WIDE, self._curation_bucket, palette=PR_ROW_PALETTE,
            accent=PR_ROW_ACCENT,
            show_tick=False, show_eye=False, keep_aspect=True,
            wrap_tiers=True,
            help_lines={"row_tiers": _PRT.S3_SESSION_TIERS_TIP})
        self._curation_bucket.setItemDelegate(self._curation_row_delegate)
        left_lay.addWidget(self._curation_bucket, 1)

    def refresh_curation_bucket(self, patch_names: list, statuses: dict,
                                qc_extracted: set = None,
                                edited_patches: set = None,
                                final_curated_patches: set = None,
                                missing: set = None,
                                axes_by_patch: dict = None,
                                info_by_patch: dict = None,
                                shapes: dict = None,
                                spacing_by_patch: dict = None,
                                measured_by_patch: dict = None) -> None:
        """Populate the curation bucket with reconstructed+ patches -- one
        item per patch with ``Qt.UserRole`` = its name, ``text()`` = today's
        padded row and the record the shared painter draws (brick V2,
        ``_step3_rows.build_curation_row``), its picture from the map.

        Args:
            qc_extracted: Set of patch names that have QC feature CSVs on disk.
                Patches in this set get a diamond indicator.
            edited_patches: Set of patch names with curation edits this session.
                Patches in this set get a pencil indicator.
            final_curated_patches: Set of patch names that are finalized + clean.
                Patches in this set get a star indicator.
        """
        self._curation_bucket.clear()
        qc_extracted = qc_extracted or set()
        edited_patches = edited_patches or set()
        final_curated_patches = final_curated_patches or set()
        missing = set(missing or ())
        axes_by_patch = axes_by_patch or {}
        info_by_patch = info_by_patch or {}
        shapes = shapes or {}
        # Column widths for alignment (today's text(), kept)
        max_name = max((len(n) for n in patch_names), default=10)
        max_status = max(
            (len(statuses.get(n, "reconstructed")) for n in patch_names),
            default=12,
        )
        # Max dots width for fixed-width alignment (●●● = 3 chars)
        _max_dots = 3
        bucket = self._curation_bucket
        for name in patch_names:
            status = statuses.get(name, "reconstructed")
            dots = _STATUS_DOTS.get(status, "")
            # Pad dots to fixed width so indicators always align
            dots_padded = dots.ljust(_max_dots)
            tags = []
            if name in qc_extracted:
                tags.append(_PRT.S3_CURATION_TAG_QC)
            if name in edited_patches:
                tags.append(_PRT.S3_CURATION_TAG_EDITED)
            tag_str = f"  {' '.join(tags)}" if tags else ""
            text = (
                f" {name.ljust(max_name)}"
                f"  {status.ljust(max_status)}"
                f"  {dots_padded}{tag_str}"
            )
            gone = name in missing
            if gone:
                # row 27: the row stays, amber, and a click is refused
                text += "  " + _PRT.S3_WARN_SIGN + _PRT.S3_PATCH_MISSING
            extra = {}
            if spacing_by_patch is not None:
                extra["spacing"] = spacing_by_patch.get(name)
            rec = _rows.build_curation_row(
                name, status, qc_extracted=name in qc_extracted,
                edited=name in edited_patches, missing=gone,
                tier_counts=self._tier_counts_for(name),
                axes=axes_by_patch.get(name),
                source_shape=shapes.get(name),
                manifest_info=info_by_patch.get(name),
                measured_spacing=(measured_by_patch or {}).get(name),
                **extra)
            if name == getattr(self, "_loaded_patch_name", None):
                rec["loaded"] = True
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, name)
            item.setData(ROLE_STATUS, status)
            item.setData(ROLE_RECORD, rec)
            if gone:
                item.setToolTip(_PRT.S3_PATCH_MISSING)
            self._apply_patch_thumb(item)
            bucket.addItem(item)

    def set_loaded_patch(self, name) -> None:
        """Mark the Curation row of the patch LOADED in the viewer (A3
        fold; ``None`` clears it).  Kept, so every rebuild of the list
        re-applies it; exactly one row carries it."""
        self._loaded_patch_name = name or None
        bucket = getattr(self, "_curation_bucket", None)
        if bucket is None:
            return
        was = bucket.blockSignals(True)
        try:
            for i in range(bucket.count()):
                item = bucket.item(i)
                rec = item.data(ROLE_RECORD)
                if rec is None:
                    continue
                rec = dict(rec)
                if (self._loaded_patch_name
                        and item.data(Qt.UserRole) == self._loaded_patch_name):
                    rec["loaded"] = True
                else:
                    rec.pop("loaded", None)
                item.setData(ROLE_RECORD, rec)
        finally:
            bucket.blockSignals(was)
        bucket.viewport().update()

    def switch_to_section(self, index: int) -> None:
        """Programmatically open an accordion section by index."""
        if hasattr(self, "_accordion"):
            self._accordion.setCurrentIndex(index)

    def refine_ticked_names(self) -> list:
        """The patch names ticked in Patch Selection, in list order (the
        ticks ARE the extraction scope; Brick B2 switches the controller's
        reads to this)."""
        lst = self.refine_patch_list
        out = []
        for i in range(lst.count()):
            it = lst.item(i)
            if it.checkState() == Qt.Checked:
                name = it.data(Qt.UserRole)
                if name:
                    out.append(str(name))
        return out

    def set_refine_ticks(self, on: bool) -> None:
        """Tick (``True``) every tickable row, or take every tick off."""
        lst = self.refine_patch_list
        state = Qt.Checked if on else Qt.Unchecked
        was = lst.blockSignals(True)
        try:
            for i in range(lst.count()):
                it = lst.item(i)
                rec = it.data(ROLE_RECORD) or {}
                if on and not rec.get("tickable", True):
                    continue
                it.setCheckState(state)
        finally:
            lst.blockSignals(was)
        self._update_refine_ticked_label()
        lst.viewport().update()

    def refine_row_count(self) -> int:
        """How many finalized patches the list shows."""
        return self.refine_patch_list.count()

    def _on_refine_select_all_requested(self) -> None:
        self.set_refine_ticks(True)

    def _update_refine_ticked_label(self, *_args) -> None:
        self.refine_ticked_count_label.setText(_MRT.MR_TICKED_COUNT.format(
            k=len(self.refine_ticked_names()), n=self.refine_row_count()))

    def set_refine_lock_line(self, text: str = "") -> None:
        """The Model Refinement section's lock line: *text* shows it, ""
        hides it."""
        self.refine_lock_line.setText(text or "")

    def set_refine_run_line(self, run_name: str) -> None:
        """The read-only run line under the lock line."""
        self.refine_run_line.set_run(run_name)

    def set_refine_will_train(self, n_lateral: int, n_axial: int,
                              n_patches: int) -> None:
        """The "Will train on" line and, under it, the few-pairs block."""
        self.pairs_summary_label.setText(
            _MRT.will_train_text(n_lateral, n_axial, n_patches))
        # F10: the block is about the LATERAL model, so it shows only when
        # the Mode trains lateral; none at all gets its own sentence
        try:
            mode = str(self.refinement_mode_combo.currentText())
        except Exception:      # noqa: BLE001
            mode = ""
        lateral_trains = "Axial" not in mode
        n_lat, n_ax = int(n_lateral), int(n_axial)
        text = ""
        if lateral_trains and n_lat + n_ax > 0:
            if n_lat == 0:
                text = _MRT.MR_NO_LATERAL_PAIRS
            elif n_lat < _MRT.FEW_PAIRS_LATERAL:
                text = _MRT.MR_FEW_PAIRS_WARN.format(n=n_lat)
        self.refine_few_pairs_label.setText(text)
        self.refine_few_pairs_label.setVisible(bool(text))

    def show_refine_status(self, text: str, kind: str = "neutral") -> None:
        """The foot's status line: *kind* neutral (mono grey), warn or
        error (the Predict + Reconstruct styles)."""
        lbl = self.refinement_status_label
        style = _PR_STATUS_STYLES.get(kind, MONO_LABEL_STYLE)
        lbl.setStyleSheet(style)
        text = "" if text is None else str(text)
        lbl.setToolTip(text)
        # at most TWO lines (M6): past that the second is elided and the
        # whole sentence is the tooltip -- the foot never grows the section
        lbl.setText(_two_lines(lbl, text))
        boxed = 0 if style in (MONO_LABEL_STYLE, MR_INFO_STYLE)             else PR_STATUS_BOX_PAD
        lbl.setMaximumHeight(2 * lbl.fontMetrics().lineSpacing() + 4 + boxed)

    def show_refine_extraction_status(self, text: str) -> None:
        """The line under Extract: at most TWO lines (the same cut as the
        foot's), the whole sentence its tooltip."""
        lbl = self.refine_extraction_status_label
        text = "" if text is None else str(text)
        lbl.setToolTip(text)
        lbl.set_full(text)
        lbl.setMaximumHeight(2 * lbl.fontMetrics().lineSpacing() + 4)

    def set_refine_loss_names(self, text: str = "") -> None:
        """Kept for the controller's calls; the header suffix stays empty
        (the stats line in the plot already names the models)."""
        eng = getattr(self.refine_loss_collapsible,
                      "_epicycle_engine_label", None)
        if eng is not None:
            eng.setText("")

    def set_refine_card(self, title: str = "", *, trained_on: str = "",
                        started_from: str = "", lateral_loss: str = "",
                        axial_loss: str = "", took: str = "",
                        new_run_hint: str = "") -> None:
        """Fill the result card; a field left "" hides its row, an empty
        *title* shows the empty sentence."""
        self.refine_card_title.setText(title or _MRT.MR_OUT_EMPTY)
        has_model = any((trained_on, started_from, lateral_loss, axial_loss,
                         took))
        self.refine_card_title.setStyleSheet(
            MR_CARD_TITLE_STYLE if has_model else MR_CARD_EMPTY_TITLE_STYLE)
        self.refine_card.setStyleSheet(_mr_card_style(
            PANEL_GOLD if has_model else "rgba(255, 255, 255, 0.15)"))
        for lbl, text in ((self.refine_card_trained_on, trained_on),
                          (self.refine_card_started_from, started_from),
                          (self.refine_card_lateral_loss, lateral_loss),
                          (self.refine_card_axial_loss, axial_loss),
                          (self.refine_card_took, took)):
            lbl.setText(text or "")
            for w in self._mr_card_rows[lbl]:
                w.setVisible(bool(text))
        self.refine_card_new_run_hint.setText(
            new_run_hint or _MRT.MR_NEW_RUN_HINT_NONE)

    def refresh_refinement_patch_list(
        self, statuses: dict, extraction_info: dict = None,
        stale_stems: "set | None" = None,
        mixed_settings_stems: "set | None" = None,
        axes_by_patch: "dict | None" = None,
        info_by_patch: "dict | None" = None,
        shapes: "dict | None" = None,
    ) -> None:
        """Populate the Model Refinement patch list (finalized patches only).

        One row per patch on the shared narrow painter (brick B1,
        ``_step3_rows.build_refine_row``): the tick, the picture, the status
        and one chip line from the extraction manifest; the ⏱ stale and ⚠
        mixed-settings markers keep their lines and tooltips.  The ticks of
        patches still listed survive the rebuild.

        Args:
            statuses: {patch_name: status} dict, pre-filtered to finalized.
            extraction_info: {patch_stem: {"spatial": N, "axial": N, ...}}
                from the extraction manifest (the folder on disk is still
                ``spatial``; the screen says lateral).
            stale_stems: patch stems whose source labels_3d.tif on disk is
                NEWER than the per-patch source_mtime stored at extraction.
            mixed_settings_stems: patch stems whose per-patch extraction /
                cleaning snapshot differs from the current UI knob values.
            axes_by_patch, info_by_patch, shapes: M8, the Session rows'
                sources (``{patch: axes}``, ``{patch: reconstruction
                manifest}``, ``{patch: raw shape}``): the row's shape line
                ("81×256×256 · 839 labels") and its axes on the status line.
        """
        axes_by_patch = axes_by_patch or {}
        info_by_patch = info_by_patch or {}
        shapes = shapes or {}
        extraction_info = extraction_info or {}
        stale_stems = stale_stems or set()
        mixed_settings_stems = mixed_settings_stems or set()
        lst = self.refine_patch_list
        keep = set(self.refine_ticked_names())
        was = lst.blockSignals(True)
        try:
            lst.clear()
            for name in sorted(statuses):
                status = statuses[name]
                stem = _Path(name).stem
                ext = extraction_info.get(stem, {})
                rec = _rows.build_refine_row(
                    name, status, axes_by_patch.get(name),
                    n_lateral=int(ext.get("spatial", 0)),
                    n_axial=int(ext.get("axial", 0)),
                    ticked=name in keep, stale=stem in stale_stems,
                    mixed=stem in mixed_settings_stems,
                    manifest_info=info_by_patch.get(name),
                    source_shape=shapes.get(name))
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, name)
                item.setData(Qt.UserRole + 1, status)
                item.setData(ROLE_RECORD, rec)
                self._apply_patch_thumb(item)
                if rec.get("row_tip"):
                    item.setToolTip(rec["row_tip"])
                item.setCheckState(Qt.Checked if name in keep
                                   else Qt.Unchecked)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable
                              | Qt.ItemIsUserCheckable)
                lst.addItem(item)
        finally:
            lst.blockSignals(was)
        self._update_refine_ticked_label()
        lst.viewport().update()

    def set_refine_surface_locked(self, locked: bool) -> None:
        """M2: the rows dim under any operation, as Predict + Reconstruct's
        do (``PR_LOCK_PROPERTY`` read by the row painter)."""
        lst = self.refine_patch_list
        lst.setProperty(PR_LOCK_PROPERTY, bool(locked))
        lst.viewport().update()

    # ------------------------------------------------------------------
    # 4. Model Refinement panel (the Predict + Reconstruct layout)
    # ------------------------------------------------------------------

    @staticmethod
    def _mr_squeeze(combo: QComboBox) -> QComboBox:
        """A combo that never widens its column: it shows what fits and
        elides the rest (a long model name must not scroll the section)."""
        combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(6)
        combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        return combo

    @staticmethod
    def _mr_label(text: str, parent: QWidget) -> QLabel:
        """A form row's label: a plain ``QLabel`` at the app font, as
        Predict + Reconstruct's "Start from" row has it."""
        return QLabel(text, parent)

    def _mr_row(self, parent: QWidget, label: QLabel, *fields,
                stretch_last: bool = True) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(label)
        for i, f in enumerate(fields):
            row.addWidget(f, 1 if i == 0 else 0)
        return row

    def _mr_help_header(self, parent: QWidget, title: str,
                        guide: str = "") -> QHBoxLayout:
        """A sub-header on the SAME call and style as Curation's
        (``_make_subsection_header``): the bold title, the ``?`` on the
        right where a guide exists."""
        row = _make_subsection_header(self, title, guide or None)
        if guide:
            btn = _guide_button_of(row, guide)
            self._mr_help_buttons.append((btn, guide))
        return row

    def _create_model_refinement_panel(self) -> QWidget:
        """The Model Refinement section on the Predict + Reconstruct layout
        (Brick B1).

        Top to bottom: the lock line (hidden), the run line, a separator,
        then the splitter 240 | 330.  LEFT: Patch Selection (the narrow
        row painter, ticks as check states), the extraction block and the
        collapsed Extraction Settings.  RIGHT: a scroll area (Fine-Tuning,
        Training Configuration, Model Output's result card) over the
        pinned foot (Start Training · Cancel, the bar, ONE status line, the
        Loss curves).  Every widget the controller binds keeps its name.
        Controller-free (the parity tests build ``_build_ui`` without one).
        """
        panel = QWidget()
        self._mr_panel = panel
        # as Predict + Reconstruct and Curation: the checkbox sheet only
        # (the combos and spin boxes take the root INPUT_STYLE) plus this
        # panel's gold radios
        panel.setStyleSheet(PANEL_CHECKBOX_STYLE + MR_RADIO_STYLE)
        self._mr_help_buttons = []
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(6, 8, 6, 6)
        outer.setSpacing(6)

        self.refine_lock_line = LockLine(panel, object_name=_MR_LOCK_OBJECT)
        outer.addWidget(self.refine_lock_line)
        self.refine_run_line = _MrRunLine(panel)
        self.refine_run_line.set_run("")
        outer.addWidget(self.refine_run_line)
        outer.addWidget(_pr_separator(panel))

        splitter = QSplitter(Qt.Horizontal, panel)
        splitter.setStyleSheet(SPLITTER_STYLE)
        splitter.setHandleWidth(PR_SPLIT_HANDLE)
        splitter.setChildrenCollapsible(False)
        self.refine_splitter = splitter

        # LEFT -- Patch Selection and the extraction block
        left = QWidget(splitter)
        left.setMinimumWidth(PR_SPLIT_LEFT_MIN)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, PR_LEFT_INNER_MARGIN, 0)
        left_lay.setSpacing(0)       # the clip edge IS the foot's top line
        self.refine_left = left
        self._build_mr_patch_selection(left, left_lay)
        self._build_mr_extraction(left, left_lay)

        # RIGHT -- the settings (they scroll) over the pinned foot
        right = QWidget(splitter)
        right.setMinimumWidth(PR_SPLIT_RIGHT_MIN)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(PR_RIGHT_INNER_MARGIN, 0, 0, 0)
        right_lay.setSpacing(0)
        self.refine_right = right
        self._mr_right_lay = right_lay
        self.refine_right_scroll = QScrollArea(right)
        self.refine_right_scroll.setWidgetResizable(True)
        self.refine_right_scroll.setFrameShape(QFrame.NoFrame)
        self.refine_right_scroll.setSizePolicy(QSizePolicy.Expanding,
                                               QSizePolicy.Ignored)
        self.refine_right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        content = QWidget()
        self.refine_right_content = content
        c_lay = QVBoxLayout(content)
        c_lay.setContentsMargins(0, 0, 0, 6)
        c_lay.setSpacing(6)
        self._build_mr_fine_tuning(content, c_lay)
        c_lay.addWidget(_pr_separator(content))
        self._build_mr_training_config(content, c_lay)
        c_lay.addWidget(_pr_separator(content))
        self._build_mr_output(content, c_lay)
        c_lay.addStretch(1)
        self.refine_right_scroll.setWidget(content)
        self.refine_right_scroll.setMinimumHeight(MR_SPLIT_TOP_MIN_H)
        self._build_mr_foot(right, right_lay)
        # the settings over the foot on a vertical splitter (split brick):
        # the owner drags how much of the column the Loss plot takes
        rsplit = _CompactSplitter(Qt.Vertical, right)
        rsplit.setStyleSheet(SPLITTER_STYLE)
        rsplit.setHandleWidth(PR_SPLIT_HANDLE)
        rsplit.setChildrenCollapsible(False)
        rsplit.addWidget(self.refine_right_scroll)
        rsplit.addWidget(self.refine_foot)
        rsplit.setStretchFactor(0, 1)
        rsplit.setStretchFactor(1, 0)
        self.refine_right_splitter = rsplit
        self._mr_split_dragged = False
        self._mr_split_programmatic = False
        rsplit.splitterMoved.connect(self._on_mr_split_moved)
        right_lay.addWidget(rsplit, 1)
        self._on_mr_loss_toggled(False)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([PR_SPLIT_LEFT_W, PR_SPLIT_RIGHT_W])
        outer.addWidget(splitter, 1)
        return panel

    def _build_mr_patch_selection(self, left: QWidget,
                                  lay: QVBoxLayout) -> None:
        """Patch Selection: header + ``?``, the sentence, the list on the
        shared narrow painter (ticks are CHECK STATES; one highlighted
        row), All / None and the count, the "Will train on" line and the
        few-pairs block."""
        head = QVBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(PR_COL_GAP - 2)
        head.addLayout(self._mr_help_header(
            left, _MRT.MR_PATCHES_TITLE, _MR_PATCHES_GUIDE))
        desc = QLabel(_MRT.MR_PATCHES_DESC, left)
        desc.setWordWrap(True)
        desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        head.addWidget(desc)
        lay.addLayout(head)
        lay.addSpacing(PR_COL_GAP)

        # H1: everything below the sentence scrolls INSIDE the column (the
        # list, All / None, the lines, Extraction Settings); the section
        # never grows past the dock.  Ignored vertically, as the right
        # column's scroll area is.
        scroll = QScrollArea(left)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        s_lay = QVBoxLayout(body)
        s_lay.setContentsMargins(0, 0, MR_SCROLL_GUTTER, 0)
        s_lay.setSpacing(PR_COL_GAP)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)
        self.refine_left_scroll = scroll
        self.refine_left_content = body
        self._mr_left_lay = s_lay
        lay = s_lay

        lst = _RefineList(left)
        self.refine_patch_list = lst
        lst.setMinimumHeight(max(PR_LIST_MIN_H, MR_LIST_TWO_ROWS_H))
        lst.setSelectionMode(QListWidget.SingleSelection)
        lst.set_empty_lines(_MRT.MR_EMPTY_LINES)
        lst.setToolTip(_MRT.MR_LIST_HELP)
        self._mr_row_delegate = StackRowDelegate(
            MODE_NARROW, lst,
            help_lines={"row_tick": _MRT.MR_ROW_TICK_HELP},
            palette=PR_ROW_PALETTE, accent=PR_ROW_ACCENT,
            show_eye=False, lock_property=PR_LOCK_PROPERTY,
            eye_lock_property=PR_LOCK_PROPERTY, dim_when_locked=True,
            show_info=False, show_thumb=True, keep_aspect=True)
        lst.setItemDelegate(self._mr_row_delegate)
        lst.select_all_requested.connect(self._on_refine_select_all_requested)
        lst.itemChanged.connect(self._update_refine_ticked_label)
        lay.addWidget(lst, 1)

        btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(6)
        self.refine_select_all_btn = QPushButton(_MRT.MR_ALL_BTN, left)
        self.refine_select_all_btn.setToolTip(_MRT.MR_ALL_TIP)
        self.refine_deselect_all_btn = QPushButton(_MRT.MR_NONE_BTN, left)
        self.refine_deselect_all_btn.setToolTip(_MRT.MR_NONE_TIP)
        for b in (self.refine_select_all_btn, self.refine_deselect_all_btn):
            b.setMinimumWidth(MR_SMALL_BTN_W)
            btns.addWidget(b)
        btns.addStretch(1)
        self.refine_ticked_count_label = QLabel(
            _MRT.MR_TICKED_COUNT.format(k=0, n=0), left)
        self.refine_ticked_count_label.setStyleSheet(DIM_LABEL_STYLE)
        btns.addWidget(self.refine_ticked_count_label)
        lay.addLayout(btns)

        self.pairs_summary_label = QLabel(_MRT.MR_WILL_TRAIN_NOTHING, left)
        self.pairs_summary_label.setStyleSheet(STATUS_LABEL_STYLE)
        self.pairs_summary_label.setWordWrap(True)
        self.pairs_summary_label.setSizePolicy(QSizePolicy.Ignored,
                                               QSizePolicy.Preferred)
        lay.addWidget(self.pairs_summary_label)
        self.refine_few_pairs_label = QLabel("", left)
        self.refine_few_pairs_label.setStyleSheet(PANEL_WARN_STYLE)
        self.refine_few_pairs_label.setWordWrap(True)
        self.refine_few_pairs_label.setSizePolicy(QSizePolicy.Ignored,
                                                  QSizePolicy.Preferred)
        self.refine_few_pairs_label.setVisible(False)
        lay.addWidget(self.refine_few_pairs_label)

    def _build_mr_extraction(self, left: QWidget, lay: QVBoxLayout) -> None:
        """Extract (its own Cancel) + Clear… on one row, the bar, the
        reason / time line, then the collapsed Extraction Settings."""
        scroll_lay = self._mr_left_lay
        foot = QWidget(left)
        foot.setObjectName(_MR_LEFT_FOOT_OBJECT)
        foot.setStyleSheet(MR_LEFT_FOOT_STYLE)
        self.refine_left_foot = foot
        lay = QVBoxLayout(foot)           # the pinned LEFT foot (H1)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(6)
        outer_left = left.layout()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.extract_training_pairs_btn = QPushButton(_MRT.MR_EXTRACT_BTN,
                                                      left)
        self.extract_training_pairs_btn.setToolTip(_MRT.MR_EXTRACT_TIP)
        self.clear_extracted_pairs_btn = QPushButton(_MRT.MR_CLEAR_BTN, left)
        self.clear_extracted_pairs_btn.setToolTip(_MRT.MR_CLEAR_TIP)
        for b in (self.extract_training_pairs_btn,
                  self.clear_extracted_pairs_btn):
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            b.setMinimumWidth(PR_FOOT_BTN_MIN_W)
        row.addWidget(self.extract_training_pairs_btn, 3)
        row.addWidget(self.clear_extracted_pairs_btn, 1)
        lay.addLayout(row)
        self.refine_extraction_progress = QProgressBar(left)
        self.refine_extraction_progress.setTextVisible(False)
        self.refine_extraction_progress.setValue(0)
        self.refine_extraction_progress.setFixedHeight(PR_PROGRESS_H)
        lay.addWidget(self.refine_extraction_progress)
        self.refine_extraction_status_label = _TwoLineLabel("", left)
        self.refine_extraction_status_label.setStyleSheet(MONO_LABEL_STYLE)
        self.refine_extraction_status_label.setWordWrap(True)
        self.refine_extraction_status_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.refine_extraction_status_label)
        outer_left.addWidget(foot)

        coll = _mr_collapsible(_MRT.MR_EXTRACT_SETTINGS_TITLE)
        self.refine_extraction_settings = coll
        body = QWidget()
        b_lay = QVBoxLayout(body)
        b_lay.setContentsMargins(4, 4, 0, 4)
        b_lay.setSpacing(6)
        self.refine_per_orientation_summary_label = QLabel(
            _MRT.MR_PER_ORIENTATION_DEFAULT, body)
        self.refine_per_orientation_summary_label.setWordWrap(True)
        self.refine_per_orientation_summary_label.setStyleSheet(
            MONO_LABEL_STYLE)
        self.refine_per_orientation_summary_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        b_lay.addWidget(self.refine_per_orientation_summary_label)
        self.refine_configure_per_orientation_btn = QPushButton(
            _MRT.MR_CONFIGURE_BTN, body)
        self.refine_configure_per_orientation_btn.setToolTip(
            _MRT.MR_CONFIGURE_TIP)
        self.refine_configure_per_orientation_btn.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Fixed)
        b_lay.addWidget(self.refine_configure_per_orientation_btn)
        split_lbl = QLabel(_MRT.MR_SPLIT_TITLE, body)
        split_lbl.setStyleSheet(PANEL_ROW_LABEL_STYLE
                                + " font-weight: bold;")
        b_lay.addWidget(split_lbl)
        grid = frm4()
        self.refine_train_frac_sb = QDoubleSpinBox()
        self.refine_train_frac_sb.setRange(0.5, 1.0)
        self.refine_train_frac_sb.setDecimals(2)
        self.refine_train_frac_sb.setSingleStep(0.05)
        self.refine_train_frac_sb.setValue(0.8)
        self.refine_train_frac_sb.setToolTip(
            "Fraction of extracted slices used for training.\n"
            "The remainder is held out for validation loss monitoring.\n"
            "E.g. 0.8 = 80% train, 20% validation.")
        self.refine_chunk_size_sb = QSpinBox()
        self.refine_chunk_size_sb.setRange(1, 100)
        self.refine_chunk_size_sb.setValue(10)
        self.refine_chunk_size_sb.setToolTip(
            "Number of contiguous slices that stay together as a unit\n"
            "during train/val splitting. E.g. chunk_size=10 means slices\n"
            "0-9 form one block, 10-19 another, etc. Each block goes\n"
            "entirely to train or val — never split across both.")
        for r, (text, w) in enumerate((
                (_MRT.MR_TRAIN_FRAC_LABEL, self.refine_train_frac_sb),
                (_MRT.MR_CHUNK_LABEL, self.refine_chunk_size_sb))):
            lbl = QLabel(text, body)
            lbl.setStyleSheet(PANEL_ROW_LABEL_STYLE)
            grid.addWidget(lbl, r, 0)
            grid.addWidget(field(w), r, 1)
        b_lay.addLayout(grid)
        self.refine_spatial_split_cb = QCheckBox(_MRT.MR_SPATIAL_CB, body)
        self.refine_spatial_split_cb.setChecked(True)
        self.refine_spatial_split_cb.setToolTip(
            "Split train/val by contiguous spatial blocks instead of\n"
            "random shuffle. Prevents data leakage from adjacent slices\n"
            "appearing in both train and val sets. Chunk size controls\n"
            "how many adjacent slices stay together as a unit.")
        b_lay.addWidget(self.refine_spatial_split_cb)
        self.refine_split_summary_label = QLabel(_MRT.MR_SPLIT_SUMMARY, body)
        self.refine_split_summary_label.setStyleSheet(STATUS_LABEL_STYLE)
        b_lay.addWidget(self.refine_split_summary_label)
        coll.addWidget(body)
        scroll_lay.addWidget(coll)

    def _build_mr_fine_tuning(self, content: QWidget,
                              lay: QVBoxLayout) -> None:
        """Fine-Tuning: Mode (the VISIBLE combo, ``PANEL_INPUT_STYLE``),
        its one-sentence hint, Init, the Source / Pretrained / Import rows
        and their Axial twins (shown per Init and Mode by the controller),
        and the Model name."""
        head = QVBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(PR_COL_GAP - 2)
        head.addLayout(self._mr_help_header(
            content, _MRT.MR_FT_TITLE, _MR_FINE_TUNING_GUIDE))
        desc = QLabel(_MRT.MR_FT_DESC, content)
        desc.setWordWrap(True)
        desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        head.addWidget(desc)
        lay.addLayout(head)

        self.refinement_mode_combo = self._mr_squeeze(QComboBox(content))
        self.refinement_mode_combo.setToolTip(_MRT.MR_MODE_TIP)
        self.refinement_mode_combo.addItems(list(_MRT.MR_MODE_ITEMS))
        lay.addLayout(self._mr_row(
            content, self._mr_label(_MRT.MR_MODE_LABEL, content),
            self.refinement_mode_combo))
        self.refinement_mode_hint = QLabel(
            _MRT.mode_hint(_MRT.MR_MODE_ITEMS[0], _MRT.MR_HINT_START_HITL,
                           _MRT.MR_HINT_START_HITL, _MRT.MR_HINT_NAME_AUTO),
            content)
        self.refinement_mode_hint.setWordWrap(True)
        self.refinement_mode_hint.setStyleSheet(FLOW_SUBTITLE_STYLE)
        self.refinement_mode_hint.setSizePolicy(QSizePolicy.Ignored,
                                                QSizePolicy.Preferred)
        lay.addWidget(self.refinement_mode_hint)

        self.refine_init_combo = self._mr_squeeze(QComboBox(content))
        self.refine_init_combo.addItems(list(_MRT.MR_INIT_ITEMS))
        self.refine_init_combo.setToolTip(_MRT.MR_INIT_TIP)
        lay.addLayout(self._mr_row(
            content, self._mr_label(_MRT.MR_INIT_LABEL, content),
            self.refine_init_combo))

        # Source -- the primary row (lateral / shared) and its axial twin
        def _combo_row(attr_row, attr_label, attr_combo, label, tip,
                       items=None, visible=True):
            row_w = QWidget(content)
            setattr(self, attr_row, row_w)
            r_lay = QHBoxLayout(row_w)
            r_lay.setContentsMargins(0, 0, 0, 0)
            r_lay.setSpacing(8)
            lbl = self._mr_label(label, row_w)
            setattr(self, attr_label, lbl)
            combo = self._mr_squeeze(QComboBox(row_w))
            if items:
                combo.addItems(list(items))
            combo.setToolTip(tip)
            setattr(self, attr_combo, combo)
            r_lay.addWidget(lbl)
            r_lay.addWidget(combo, 1)
            row_w.setVisible(visible)
            lay.addWidget(row_w)

        pretrained = ("cyto3", "cyto2", "cyto", "nuclei")
        _combo_row("refine_source_row", "refine_source_label",
                   "refinement_source_combo", _MRT.MR_SOURCE_LABEL,
                   _MRT.MR_SOURCE_TIP)
        _combo_row("refine_source_row_axial", "refine_source_label_axial",
                   "refinement_source_combo_axial", _MRT.MR_AXIAL_LABEL,
                   _MRT.MR_SOURCE_AXIAL_TIP, visible=False)
        _combo_row("refine_pretrained_row", "refine_pretrained_label",
                   "refine_pretrained_combo", _MRT.MR_PRETRAINED_LABEL,
                   _MRT.MR_PRETRAINED_TIP, pretrained, visible=False)
        _combo_row("refine_pretrained_row_axial",
                   "refine_pretrained_label_axial",
                   "refine_pretrained_combo_axial", _MRT.MR_AXIAL_LABEL,
                   _MRT.MR_PRETRAINED_AXIAL_TIP, pretrained, visible=False)

        def _import_row(attr_row, attr_label, attr_path, attr_btn,
                        placeholder):
            row_w = QWidget(content)
            setattr(self, attr_row, row_w)
            r_lay = QHBoxLayout(row_w)
            r_lay.setContentsMargins(0, 0, 0, 0)
            r_lay.setSpacing(8)
            lbl = self._mr_label(
                _MRT.MR_IMPORT_LABEL if attr_label == "refine_import_label"
                else _MRT.MR_AXIAL_LABEL, row_w)
            setattr(self, attr_label, lbl)
            path = QLineEdit(row_w)
            path.setPlaceholderText(placeholder)
            path.setReadOnly(True)
            path.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            setattr(self, attr_path, path)
            btn = QPushButton(_MRT.MR_BROWSE_BTN, row_w)
            setattr(self, attr_btn, btn)
            r_lay.addWidget(lbl)
            r_lay.addWidget(path, 1)
            r_lay.addWidget(btn)
            row_w.setVisible(False)
            lay.addWidget(row_w)

        _import_row("refine_import_row", "refine_import_label",
                    "refine_import_path", "refine_import_browse_btn",
                    _MRT.MR_IMPORT_PLACEHOLDER)
        _import_row("refine_import_row_axial", "refine_import_label_axial",
                    "refine_import_path_axial",
                    "refine_import_browse_btn_axial",
                    _MRT.MR_IMPORT_AXIAL_PLACEHOLDER)

        self.refine_model_name = QLineEdit(content)
        self.refine_model_name.setPlaceholderText(
            _MRT.MR_MODEL_NAME_PLACEHOLDER)
        self.refine_model_name.setSizePolicy(QSizePolicy.Ignored,
                                             QSizePolicy.Fixed)
        lay.addLayout(self._mr_row(
            content, self._mr_label(_MRT.MR_MODEL_NAME_LABEL, content),
            self.refine_model_name))

    def _build_mr_training_config(self, content: QWidget,
                                  lay: QVBoxLayout) -> None:
        """Training Configuration: Preset + Apply, what the preset does,
        Epochs / Learning rate / Batch size, the collapsed Advanced
        Parameters on ONE aligned grid and Save / Load Config."""
        head = QVBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(PR_COL_GAP - 2)
        head.addLayout(self._mr_help_header(
            content, _MRT.MR_TC_TITLE, _MR_TRAINING_CONFIG_GUIDE))
        desc = QLabel(_MRT.MR_TC_DESC, content)
        desc.setWordWrap(True)
        desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        head.addWidget(desc)
        lay.addLayout(head)

        self.refine_preset_combo = self._mr_squeeze(QComboBox(content))
        self.refine_preset_combo.addItems(
            [_MRT.MR_PRESET_DISPLAY.get(k, k) for k in _MRT.MR_PRESET_ITEMS])
        self.refine_preset_combo.setToolTip(_MRT.MR_PRESET_TIP)
        self.refine_apply_preset_btn = QPushButton(_MRT.MR_APPLY_BTN, content)
        self.refine_apply_preset_btn.setToolTip(_MRT.MR_APPLY_TIP)
        lay.addLayout(self._mr_row(
            content, self._mr_label(_MRT.MR_PRESET_LABEL, content),
            self.refine_preset_combo, self.refine_apply_preset_btn))
        self.refine_preset_description = QLabel("", content)
        self.refine_preset_description.setWordWrap(True)
        self.refine_preset_description.setStyleSheet(PANEL_HELP_STYLE)
        self.refine_preset_description.setSizePolicy(QSizePolicy.Ignored,
                                                     QSizePolicy.Preferred)
        self.refine_preset_description.setText(
            PRESET_DESCRIPTIONS.get("default", ""))
        lay.addWidget(self.refine_preset_description)

        self.refine_epochs_sb = QSpinBox()
        self.refine_epochs_sb.setRange(1, 5000)
        self.refine_epochs_sb.setValue(100)
        self.refine_epochs_sb.setToolTip(
            "Number of training iterations through the full dataset.\n"
            "Higher = more learning, but diminishing returns + "
            "overfitting risk on small datasets.\n"
            "Typical EpiCycle values: 100 (default + quick check), "
            "500 (finetune / thorough start), 1000 (final run).\n"
            "Cellpose's underlying default is 2000 — large because "
            "Cellpose was designed for fresh training, not iterative "
            "HITL refinement.")
        self.refine_lr_sb = QDoubleSpinBox()
        self.refine_lr_sb.setRange(1e-6, 1.0)
        self.refine_lr_sb.setDecimals(6)
        self.refine_lr_sb.setSingleStep(0.0001)
        self.refine_lr_sb.setValue(0.01)
        self.refine_lr_sb.setToolTip(
            "How fast the model updates its weights per epoch.\n"
            "  • Higher (0.01): faster learning but unstable / can "
            "overshoot.  EpiCycle's default.\n"
            "  • Lower (0.001, 10× smaller): more stable, slower.  "
            "Use when fine-tuning a pretrained model so you don't "
            "unlearn its prior knowledge.\n"
            "(Cellpose's underlying `train_seg` default is 0.005; "
            "EpiCycle overrides to 0.01 for SGD.)")
        self.refine_batch_sb = QSpinBox()
        self.refine_batch_sb.setRange(1, 128)
        self.refine_batch_sb.setValue(16)
        self.refine_batch_sb.setToolTip(
            "Number of training images processed simultaneously per "
            "step.\n"
            "  • Higher (32, 64): faster training + more stable "
            "gradients but uses more GPU memory.\n"
            "  • Lower (8, 4): smaller memory footprint, noisier "
            "gradients.  Use on small GPUs or for very dense slices.\n"
            "If you hit CUDA out-of-memory errors, reduce this first.")
        # the same grid as Predict + Reconstruct's Inference settings: a
        # stretching label column, the 128 px field pinned right
        top_grid = frm4()
        for r, (text, w) in enumerate(((_MRT.MR_EPOCHS_LABEL,
                                        self.refine_epochs_sb),
                                       (_MRT.MR_LR_LABEL, self.refine_lr_sb),
                                       (_MRT.MR_BATCH_LABEL,
                                        self.refine_batch_sb))):
            lbl = QLabel(text, content)
            lbl.setToolTip(w.toolTip())
            top_grid.addWidget(lbl, r, 0)
            top_grid.addWidget(field(w), r, 1)
        lay.addLayout(top_grid)

        adv = _mr_collapsible(_MRT.MR_ADVANCED_TITLE)
        self.refine_advanced = adv
        adv_w = QWidget()
        adv_lay = QVBoxLayout(adv_w)
        adv_lay.setContentsMargins(4, 4, 0, 4)
        adv_lay.setSpacing(6)
        g = frm4()
        self.refine_wd_sb = QDoubleSpinBox()
        self.refine_wd_sb.setRange(0.0, 1.0)
        self.refine_wd_sb.setDecimals(6)
        self.refine_wd_sb.setSingleStep(0.00001)
        self.refine_wd_sb.setValue(1e-5)
        self.refine_wd_sb.setToolTip(
            "L2 regularization strength.  Penalizes large weights to "
            "discourage overfitting on small datasets.\n"
            "Cellpose default: 1e-5.  Raise (e.g., 1e-4) only if you "
            "see severe overfitting (train loss dropping but val loss "
            "rising).")
        self.refine_momentum_sb = QDoubleSpinBox()
        self.refine_momentum_sb.setRange(0.0, 1.0)
        self.refine_momentum_sb.setDecimals(3)
        self.refine_momentum_sb.setSingleStep(0.01)
        self.refine_momentum_sb.setValue(0.9)
        self.refine_momentum_sb.setToolTip(
            "SGD momentum — accumulates a fraction of past gradients "
            "to smooth out per-step updates.\n"
            "0.9 is the Cellpose default and rarely needs changing.  "
            "Only matters when 'Use SGD' is enabled.")
        self.refine_min_masks_sb = QSpinBox()
        self.refine_min_masks_sb.setRange(0, 1000)
        self.refine_min_masks_sb.setValue(0)
        self.refine_min_masks_sb.setToolTip(
            "Minimum masks per training image (0 = no filter)")
        self.refine_chan1_combo = QComboBox()
        self.refine_chan1_combo.addItems(
            ["0: gray", "1: red", "2: green", "3: blue"])
        self.refine_chan1_combo.setCurrentIndex(2)  # default: green
        self.refine_chan1_combo.setToolTip(
            "The channel containing the objects you want segmented.\n"
            "  • For cells: typically the membrane / cytoplasm stain.\n"
            "  • For nuclei segmentation: the nuclear stain (DAPI, "
            "Hoechst, etc.).\n"
            "  • Use '0: gray' to collapse all channels into "
            "grayscale.\n\n"
            "IMPORTANT: must match the channel ordering used at predict "
            "time.")
        self.refine_chan2_combo = QComboBox()
        self.refine_chan2_combo.addItems(
            ["0: none", "1: red", "2: green", "3: blue"])
        self.refine_chan2_combo.setCurrentIndex(1)  # default: red
        self.refine_chan2_combo.setToolTip(
            "Optional auxiliary channel that helps Cellpose separate "
            "touching objects.\n"
            "  • Typically the nuclear channel when segmenting cells.\n"
            "  • Leave on '0: none' if not using.")
        self.refine_seed_sb = QSpinBox()
        self.refine_seed_sb.setRange(0, 99999)
        self.refine_seed_sb.setValue(0)
        self.refine_seed_sb.setToolTip(
            "Random seed for reproducible training (data shuffling, "
            "weight init, augmentation).\n"
            "  • 0: no seed — every run picks different randomness; "
            "use for normal runs.\n"
            "  • Non-zero (e.g., 42): deterministic — repeated runs "
            "with the same data + config produce identical models.  "
            "Use when debugging or benchmarking training changes.")
        self.refine_normalize_cb = QCheckBox("", adv_w)
        self.refine_normalize_cb.setChecked(True)
        self.refine_normalize_cb.setToolTip(
            "Normalize image intensity per-image before training.\n"
            "Usually should match whatever was used during HITL training.")
        self.refine_rescale_cb = QCheckBox("", adv_w)
        self.refine_rescale_cb.setChecked(True)
        self.refine_rescale_cb.setToolTip(
            "Rescale images so cells match the model's expected diameter.\n"
            "Enables scale invariance. Keep on unless cell sizes are "
            "uniform.")
        self.refine_sgd_cb = QCheckBox("", adv_w)
        self.refine_sgd_cb.setChecked(True)
        self.refine_sgd_cb.setToolTip(
            "Use SGD optimizer with momentum instead of RAdam.\n"
            "SGD can be more stable for fine-tuning with small datasets.")
        self.refine_compute_flows_cb = QCheckBox("", adv_w)
        self.refine_compute_flows_cb.setChecked(False)
        self.refine_compute_flows_cb.setToolTip(
            "Recompute flow fields from masks during training")
        rows = (
            ("Weight decay:", self.refine_wd_sb),
            ("Momentum:", self.refine_momentum_sb),
            ("Min train masks:", self.refine_min_masks_sb),
            ("Channel to segment:", self.refine_chan1_combo),
            ("Auxiliary channel (optional):", self.refine_chan2_combo),
            ("Seed:", self.refine_seed_sb),
            ("Normalize input", self.refine_normalize_cb),
            ("Rescale to training diameter", self.refine_rescale_cb),
            ("Use SGD (instead of RAdam)", self.refine_sgd_cb),
            ("Recompute flows", self.refine_compute_flows_cb),
        )
        for r, (text, w) in enumerate(rows):
            if isinstance(w, QCheckBox):
                lbl = TickLabel(text, w)
                lbl.setToolTip(w.toolTip())
                g.addWidget(w, r, 1, Qt.AlignLeft | Qt.AlignVCenter)
            else:
                lbl = QLabel(text, adv_w)
                lbl.setToolTip(w.toolTip())
                g.addWidget(field(w), r, 1)
            lbl.setStyleSheet(PANEL_ROW_LABEL_STYLE)
            lbl.setWordWrap(True)
            g.addWidget(lbl, r, 0)
        # Device: the label on the grid, the two radios in the field cell
        r = len(rows)
        dev_lbl = QLabel("Device:", adv_w)
        dev_lbl.setStyleSheet(PANEL_ROW_LABEL_STYLE)
        dev_cell = QWidget(adv_w)
        dev_lay = QHBoxLayout(dev_cell)
        dev_lay.setContentsMargins(0, 0, 0, 0)
        dev_lay.setSpacing(8)
        self.refine_device_gpu = QRadioButton("GPU", dev_cell)
        self.refine_device_gpu.setChecked(True)
        self.refine_device_gpu.setToolTip(
            "Train on CUDA-capable GPU.  Required for reasonable "
            "training speed; even small datasets can take hours on "
            "CPU.  Falls back gracefully if CUDA isn't available.")
        self.refine_device_cpu = QRadioButton("CPU", dev_cell)
        self.refine_device_cpu.setToolTip(
            "Train on CPU.  Only useful for debugging / testing on "
            "machines without a GPU.  Expect 10-100× slower than GPU.")
        device_group = QButtonGroup(adv_w)
        device_group.addButton(self.refine_device_gpu)
        device_group.addButton(self.refine_device_cpu)
        dev_lay.addWidget(self.refine_device_gpu)
        dev_lay.addWidget(self.refine_device_cpu)
        dev_lay.addStretch(1)
        dev_cell.setFixedWidth(FORM_FIELD_W)
        g.addWidget(dev_lbl, r, 0)
        g.addWidget(dev_cell, r, 1)
        adv_lay.addLayout(g)
        adv.addWidget(adv_w)
        lay.addWidget(adv)

        cfg = QHBoxLayout()
        cfg.setContentsMargins(0, 0, 0, 0)
        cfg.setSpacing(6)
        self.refine_save_config_btn = QPushButton(_MRT.MR_SAVE_CONFIG_BTN,
                                                  content)
        self.refine_save_config_btn.setToolTip(
            "Save the current training hyperparameters (epochs, LR, "
            "batch, channels, etc.) to a JSON file.  Reproducible: "
            "load it later to restore the exact same config.")
        self.refine_load_config_btn = QPushButton(_MRT.MR_LOAD_CONFIG_BTN,
                                                  content)
        self.refine_load_config_btn.setToolTip(
            "Load training hyperparameters from a previously-saved "
            "JSON file.  Overwrites the current spinbox / combo / "
            "checkbox values.")
        for b in (self.refine_save_config_btn, self.refine_load_config_btn):
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            b.setMinimumWidth(PR_FOOT_BTN_MIN_W)
            cfg.addWidget(b, 1)
        lay.addLayout(cfg)

    def _build_mr_output(self, content: QWidget, lay: QVBoxLayout) -> None:
        """Model Output: the result card (title, five facts, the folder),
        Use for Re-Prediction + View Refined Models, and the New Run
        sentence.  Built hidden: ``refine_pack_status_label``."""
        head = QVBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(PR_COL_GAP - 2)
        head.addLayout(self._mr_help_header(content, _MRT.MR_OUT_TITLE))
        desc = QLabel(_MRT.MR_OUT_DESC, content)
        desc.setWordWrap(True)
        desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        head.addWidget(desc)
        lay.addLayout(head)

        card = QWidget(content)
        c_lay = QVBoxLayout(card)
        c_lay.setContentsMargins(9, 6, 9, 6)
        c_lay.setSpacing(4)
        card.setObjectName(MR_CARD_OBJECT)
        card.setAttribute(Qt.WA_StyledBackground, True)
        card.setStyleSheet(_mr_card_style("rgba(255, 255, 255, 0.15)"))
        self.refine_card = card
        self.refine_card_title = QLabel(_MRT.MR_OUT_EMPTY, card)
        self.refine_card_title.setWordWrap(True)
        self.refine_card_title.setStyleSheet(MR_CARD_EMPTY_TITLE_STYLE)
        self.refine_card_title.setSizePolicy(QSizePolicy.Ignored,
                                             QSizePolicy.Preferred)
        c_lay.addWidget(self.refine_card_title)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        grid.setColumnMinimumWidth(0, PANEL_LABEL_COLUMN)
        grid.setColumnStretch(1, 1)
        self._mr_card_rows = {}
        facts = (
            ("refine_card_trained_on", _MRT.MR_CARD_TRAINED_ON, False),
            ("refine_card_started_from", _MRT.MR_CARD_STARTED_FROM, True),
            ("refine_card_lateral_loss", _MRT.MR_CARD_LATERAL_LOSS, False),
            ("refine_card_axial_loss", _MRT.MR_CARD_AXIAL_LOSS, False),
            ("refine_card_took", _MRT.MR_CARD_TOOK, False),
        )
        for r, (attr, key, elide) in enumerate(facts):
            k = QLabel(key, card)
            k.setStyleSheet(MR_CARD_KEY_STYLE)
            k.setAlignment(Qt.AlignTop)
            if elide:
                v = _MiddleElidedLabel(card)
            else:
                v = QLabel("", card)
                v.setWordWrap(True)
                v.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            v.setStyleSheet(MR_CARD_VALUE_STYLE)
            setattr(self, attr, v)
            grid.addWidget(k, r, 0, Qt.AlignTop)
            grid.addWidget(v, r, 1)
            k.setVisible(False)
            v.setVisible(False)
            self._mr_card_rows[v] = (k, v)
        r = len(facts)
        folder_key = QLabel(_MRT.MR_CARD_FOLDER, card)
        folder_key.setStyleSheet(MR_CARD_KEY_STYLE)
        self.refinement_output_label = _MiddleElidedLabel(card)
        self.refinement_output_label.setText(_MRT.MR_CARD_FOLDER_EMPTY)
        self.refinement_output_label.setStyleSheet(MONO_LABEL_STYLE)
        grid.addWidget(folder_key, r, 0, Qt.AlignTop)
        grid.addWidget(self.refinement_output_label, r, 1)
        c_lay.addLayout(grid)

        btns = QVBoxLayout()            # H3: the doors stack, never clip
        btns.setContentsMargins(0, 2, 0, 0)
        btns.setSpacing(6)
        self.refine_use_for_predict_btn = QPushButton(_MRT.MR_USE_BTN, card)
        self.refine_use_for_predict_btn.setToolTip(_MRT.MR_USE_TIP)
        self.refine_view_models_btn = QPushButton(_MRT.MR_VIEW_MODELS_BTN,
                                                  card)
        self.refine_view_models_btn.setToolTip(_MRT.MR_VIEW_MODELS_TIP)
        for b in (self.refine_use_for_predict_btn,
                  self.refine_view_models_btn):
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            b.setMinimumWidth(PR_FOOT_BTN_MIN_W)
        btns.addWidget(self.refine_use_for_predict_btn)
        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.addWidget(self.refine_view_models_btn)
        view_row.addStretch(1)
        self.refine_view_models_btn.setSizePolicy(QSizePolicy.Maximum,
                                                  QSizePolicy.Fixed)
        btns.addLayout(view_row)
        c_lay.addLayout(btns)
        self.refine_card_new_run_hint = QLabel(_MRT.MR_NEW_RUN_HINT_NONE,
                                               card)
        self.refine_card_new_run_hint.setWordWrap(True)
        self.refine_card_new_run_hint.setStyleSheet(PANEL_HELP_STYLE)
        self.refine_card_new_run_hint.setSizePolicy(QSizePolicy.Ignored,
                                                    QSizePolicy.Preferred)
        c_lay.addWidget(self.refine_card_new_run_hint)
        lay.addWidget(card)

        # built, hidden, never shown: nothing writes a model-pack file now
        self.refine_pack_status_label = QLabel("", content)
        self.refine_pack_status_label.setStyleSheet(STATUS_LABEL_STYLE)
        self.refine_pack_status_label.setWordWrap(True)
        self.refine_pack_status_label.setVisible(False)
        lay.addWidget(self.refine_pack_status_label)

    def _build_mr_foot(self, right: QWidget, lay: QVBoxLayout) -> None:
        """The pinned foot: Start Training · Cancel, the bar, ONE wrapping
        status line and the collapsed Loss curves -- a sibling of the scroll
        area, never inside it."""
        foot = QWidget(right)
        foot.setObjectName(_MR_FOOT_OBJECT)
        foot.setStyleSheet(MR_FOOT_STYLE)
        self.refine_foot = foot
        f_lay = QVBoxLayout(foot)
        f_lay.setContentsMargins(0, 6, 0, 0)
        f_lay.setSpacing(6)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.launch_refinement_btn = QPushButton(_MRT.MR_START_BTN, foot)
        self.launch_refinement_btn.setToolTip(_MRT.MR_START_TIP)
        self.refinement_cancel_btn = QPushButton(_MRT.MR_CANCEL_BTN, foot)
        self.refinement_cancel_btn.setToolTip(_MRT.MR_CANCEL_TIP)
        self.refinement_cancel_btn.setEnabled(False)
        for b in (self.launch_refinement_btn, self.refinement_cancel_btn):
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            b.setMinimumWidth(PR_FOOT_BTN_MIN_W)
        self.refinement_cancel_btn.setMinimumWidth(MR_CANCEL_MIN_W)
        # Minimum, not Ignored: an Ignored button of stretch 0 collapses to
        # nothing and is pushed off the row instead of keeping its width
        self.refinement_cancel_btn.setSizePolicy(QSizePolicy.Minimum,
                                                 QSizePolicy.Fixed)
        row.addWidget(self.launch_refinement_btn, 1)     # M12
        row.addWidget(self.refinement_cancel_btn, 0)
        f_lay.addLayout(row)
        self.refinement_progress = QProgressBar(foot)
        self.refinement_progress.setTextVisible(False)    # blank when idle
        self.refinement_progress.setFixedHeight(PR_PROGRESS_H)
        f_lay.addWidget(self.refinement_progress)
        self.refinement_status_label = QLabel("", foot)
        self.refinement_status_label.setStyleSheet(MONO_LABEL_STYLE)
        self.refinement_status_label.setWordWrap(True)
        self.refinement_status_label.setSizePolicy(QSizePolicy.Ignored,
                                                   QSizePolicy.Preferred)
        f_lay.addWidget(self.refinement_status_label)

        loss = _grow_collapsible(_MRT.MR_LOSS_TITLE, _MRT.MR_LOSS_SUFFIX,
                                 engine_object_name=_PR_ENGINE_OBJECT)
        loss.setParent(foot)
        self.refine_loss_collapsible = loss
        try:
            from napari_epicycle.hitl_training_annotation.ui.loss_plot_widget import (
                LossPlotWidget,
            )
            self.refinement_loss_plot = LossPlotWidget()
        except Exception:
            self.refinement_loss_plot = QLabel(
                "Loss curve\n(pyqtgraph not available)")
            self.refinement_loss_plot.setAlignment(Qt.AlignCenter)
            self.refinement_loss_plot.setStyleSheet(CANVAS_AREA_STYLE)
        self.refinement_loss_plot.setMinimumHeight(MR_LOSS_MIN_H)
        self.refinement_loss_plot.setToolTip(_MRT.MR_LOSS_EMPTY)
        self.refinement_loss_plot.setSizePolicy(QSizePolicy.Expanding,
                                                QSizePolicy.Expanding)
        loss.addWidget(self.refinement_loss_plot)
        f_lay.addWidget(loss)
        loss.toggled.connect(self._on_mr_loss_toggled)

    def _on_mr_split_moved(self, *_args) -> None:
        """A drag of the handle: from now on the user's sizes stand."""
        if not getattr(self, "_mr_split_programmatic", False):
            self._mr_split_dragged = True

    def _mr_foot_collapsed_h(self) -> int:
        """The foot's height with the Loss curves closed (buttons, bar,
        status, header) -- the least the bottom pane may take."""
        foot = self.refine_foot
        lay = foot.layout()
        loss = self.refine_loss_collapsible
        m = lay.contentsMargins()
        h = m.top() + m.bottom()
        for i in range(lay.count()):
            item = lay.itemAt(i)
            w = item.widget()
            if w is loss:
                h += loss._toggle_btn.sizeHint().height()
            elif w is not None:
                h += (w.heightForWidth(foot.width())
                      if w.hasHeightForWidth() and foot.width() > 0
                      else w.sizeHint().height())
            else:
                h += item.sizeHint().height()
            h += lay.spacing()
        return h

    def _apply_mr_split_sizes(self, opened: bool) -> None:
        sp = getattr(self, "refine_right_splitter", None)
        if sp is None:
            return
        total = sp.height() - sp.handleWidth()
        if total <= 0:
            return
        if opened:
            bottom = int(total * MR_SPLIT_OPEN_FRAC)
        else:
            bottom = self._mr_foot_collapsed_h()
        bottom = max(bottom, self.refine_foot.minimumSizeHint().height())
        self._mr_split_programmatic = True
        try:
            sp.setSizes([max(total - bottom, 0), bottom])
        finally:
            self._mr_split_programmatic = False

    def _on_mr_loss_toggled(self, opened: bool) -> None:
        """An OPEN Loss curves fills the bottom pane of the right column's
        splitter; closed, the foot is just its buttons / bar / status."""
        loss = self.refine_loss_collapsible
        loss.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding if opened else QSizePolicy.Maximum)
        try:
            loss.content().setSizePolicy(
                QSizePolicy.Preferred,
                QSizePolicy.Expanding if opened else QSizePolicy.Preferred)
        except Exception:      # noqa: BLE001
            pass
        self._mr_loss_open = bool(opened)
        sp = getattr(self, "refine_right_splitter", None)
        if sp is None:
            return
        sp.setStretchFactor(0, 1)
        sp.setStretchFactor(1, 1 if opened else 0)
        self.refine_foot.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding if opened else QSizePolicy.Maximum)
        if not self._mr_split_dragged:
            self._apply_mr_split_sizes(bool(opened))
            # the pane may not be laid out yet (first show): redo once
            QTimer.singleShot(0, self._reapply_mr_split_sizes)

    def _reapply_mr_split_sizes(self) -> None:
        if not self._mr_split_dragged:
            self._apply_mr_split_sizes(self._mr_loss_open)

    # ------------------------------------------------------------------
    # 5. Diagnostics panel
    # ------------------------------------------------------------------

    def _create_diagnostics_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # GPU status
        _hdr = QLabel("System")
        _hdr.setStyleSheet(SECTION_HEADER_STYLE)
        lay.addWidget(_hdr)
        self.gpu_status_label = QLabel("GPU: checking...")
        self.gpu_status_label.setStyleSheet(MONO_LABEL_STYLE)
        lay.addWidget(self.gpu_status_label)

        _sep = QLabel("")
        _sep.setFixedHeight(1)
        _sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        lay.addWidget(_sep)

        # Log output
        _hdr = QLabel("Log Output")
        _hdr.setStyleSheet(SECTION_HEADER_STYLE)
        lay.addWidget(_hdr)
        _desc = QLabel("Pipeline events, reconstruction progress, and error messages.")
        _desc.setWordWrap(True)
        _desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        lay.addWidget(_desc)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(LOG_OUTPUT_STYLE)
        self.log_output.setMinimumHeight(220)
        lay.addWidget(self.log_output, 1)

        return panel

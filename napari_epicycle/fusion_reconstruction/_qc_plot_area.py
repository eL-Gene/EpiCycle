"""Step 3's QC plot area: the right column of Curation.

``QCPlotArea`` stacks, top to bottom:

* the **see-line** -- what the plot shows (``{n} labels · {g} ghosts ·
  {sel} · coloured by {x}``), a separator, the **status slot** (what just
  happened, coloured by kind) and the **"Colours in viewer"** tick;
* the **plot** (``QCPlotWidget`` in the Workbench layout, amber, Step 3's
  five tiers and ghosts) above a **divider** (a copy of the Workbench's,
  with its rules: a top stop that counts a wrapped axes row, a floor for
  the lower area, a first-show reset, a double-click reset, the plot
  height kept when the chrome changes);
* the **lower area**: two word-tabs "Histogram · Selection" over a stack
  that HIDES its non-current page (a ``QStackedWidget`` answers with its
  tallest page), holding the triage histogram and the
  Selection card;
* the pinned **action row**: ``Assign {n} selected to [tier] [Assign]
  [Undo assign] [✕ Clear]``.

The controller pushes every selection change through ONE entry,
:meth:`QCPlotArea.selection_changed` (``source=`` says which door); the
area never connects to the plot's ``labels_selected`` itself, and the card
always reads the plot's selection LIVE.  Every word comes from
``_qc_plot_text`` (an AST test refuses a literal in a ``setText``).
"""

from __future__ import annotations

import html
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from qtpy.QtCore import QEvent, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QPainter, QPalette
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QSplitterHandle,
    QVBoxLayout,
    QWidget,
)

from ..ui_shell import theme
from . import _qc_plot_text as T
from .qc_plot_widget import (
    STEP3_SCHEME,
    WB_PLOT_MIN_H,
    QCPlotWidget,
    _wb_toggle_style,
    as_plot_palette,
)
from .triage_histogram_panel import TriageHistogramPanel

#: The plot's height after a divider reset.
PLOT_DEFAULT_PX = 270
#: Room the lower area gets above its floor after a reset.
LOWER_SLACK_PX = 30
#: The divider's thickness (the Workbench's).
HANDLE_PX = 9
#: Room between the plot's legend and the divider.
PINNED_BOTTOM_PX = 4
#: The five tiers in triage order (the action row's list, the card's chips).
TIERS = tuple(STEP3_SCHEME.buckets[:5])
#: The catch-all bucket (a label with no tier).
UNASSIGNED = STEP3_SCHEME.unassigned
#: Rows that are ghosts (the plot's own rule, ``_get_ghost_mask``).
GHOST_STATUSES = ("deleted", "superseded")

#: Status-slot colours by kind.
STATUS_INFO_COLOUR = "#9a9a9a"
STATUS_WARN_COLOUR = theme.PANEL_GOLD_HOVER
STATUS_ERROR_COLOUR = "#e0776f"   # PANEL_ERROR_STYLE's text colour
#: The status slot never takes more than this share of the see-line.
STATUS_MAX_SHARE = 0.40

_SEE_STYLE = "color: #cccccc; font-size: 12px; background: transparent;"
_STATUS_STYLE = "color: {c}; font-size: 11px; background: transparent;"
_SEP_STYLE = "background: #555555;"
_ACTION_LABEL_STYLE = (
    "QLabel { color: #dddddd; font-size: 12px; background: transparent; }"
    # locked or greyed: dim like the buttons beside it (P2 audit LOW-16)
    "QLabel:disabled { color: #6f6f6f; }")
_ACTION_ROW_STYLE = (
    "QWidget#qcActionRow { border-top: 1px solid #555555;"
    " background: transparent; }")
_COUNT_STYLE = ("color: #eeeeee; font-size: 13px; font-weight: bold;"
                " background: transparent;")
_NONE_STYLE = "color: #cccccc; font-size: 11px; background: transparent;"
_CHIPS_STYLE = "color: #bbbbbb; font-size: 11px; background: transparent;"
_MSG_STYLE = "color: #cfcfcf; font-size: 11px; background: transparent;"
_TOTALS_STYLE = "color: #999999; font-size: 11px; background: transparent;"
_ASSIGN_STYLE = (theme.PRIMARY_BTN_STYLE
                 + "QPushButton { padding: 4px 12px; }")


def _card_style(edge: str) -> str:
    return (
        "QWidget#qcSelectionCard {"
        "  background: rgba(255, 255, 255, 0.03);"
        "  border: 1px solid rgba(255, 255, 255, 0.10);"
        f"  border-left: 3px solid {edge};"
        "  border-radius: 4px;"
        "}")


_CARD_EDGE_ON = theme.PANEL_GOLD
_CARD_EDGE_NONE = "rgba(255, 255, 255, 0.15)"


# ---------------------------------------------------------------------------
# Pure helpers (Qt-free)
# ---------------------------------------------------------------------------

def _rgb_hex(rgb) -> str:
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _num(n) -> str:
    return f"{int(n):,}"


def selection_card_parts(tiers: Iterable) -> List[Tuple[str, int, str]]:
    """``[(tier, n, "#rrggbb"), ...]`` for the Selection card's chips and mix
    bar: the five tiers always, in triage order; ``unassigned`` (a label
    with no tier, or an unknown word) only when some label is in it.

    >>> selection_card_parts(["good", "good", "", "bad"])[:2]
    [('curated', 0, '#00bcd4'), ('good', 2, '#50c878')]
    """
    counts = {b: 0 for b in STEP3_SCHEME.buckets}
    for t in tiers:
        counts[STEP3_SCHEME.bucket_of(t)] += 1
    out = []
    for b in STEP3_SCHEME.buckets:
        if b == UNASSIGNED and counts[b] == 0:
            continue
        out.append((b, counts[b], _rgb_hex(STEP3_SCHEME.rgb[b])))
    return out


def _word(bucket: str) -> str:
    return T.UNMARKED_WORD if bucket == UNASSIGNED else bucket


def _active_mask(df) -> np.ndarray:
    if df is None:
        return np.zeros(0, dtype=bool)
    if "label_status" not in df.columns:
        return np.ones(len(df), dtype=bool)
    return ~df["label_status"].isin(GHOST_STATUSES).to_numpy()


def totals_line(df) -> str:
    """The card's totals line over the whole patch: every tier of the
    active labels, then the ghosts (``""`` with no table).

    "All labels: curated 842 · good 0 · review 0 · bad 0 · garbage 0, plus
    837 ghosts"
    """
    if df is None:
        return ""
    active = _active_mask(df)
    tiers = (df["qc_tier"].to_numpy()[active].tolist()
             if "qc_tier" in df.columns else [""] * int(active.sum()))
    parts = T.SEE_SEP.join(
        T.CARD_TIER_COUNT.format(tier=_word(b), n=_num(n))
        for b, n, _ in selection_card_parts(tiers))
    ghosts = int((~active).sum())
    if ghosts == 0:
        return T.CARD_TOTALS_NO_GHOSTS.format(parts=parts)
    return T.count_text(ghosts, T.CARD_TOTALS, T.CARD_TOTALS_ONE_GHOST,
                        parts=parts)


def see_line_text(counts: dict) -> str:
    """The see-line from ``QCPlotWidget.see_counts()``: labels · selected ·
    ghosts, the least important part last so the elision eats it first
    (``SEE_LINE``; P2 audit MED-4).

    The ghosts are ALWAYS counted, Ghosts toggle on or off: they are rows
    of the table whether the plot draws them or not, while the legend lists
    only what is drawn (P2 audit LOW-12)."""
    labels = int(counts.get("labels", 0))
    ghosts = int(counts.get("ghosts", 0))
    sel = int(counts.get("selected", 0))
    parts = [
        T.count_text(labels, T.SEE_LABELS, T.SEE_LABELS_ONE),
        T.SEE_SELECTED.format(n=_num(sel)) if sel else T.SEE_NONE_SELECTED,
        T.count_text(ghosts, T.SEE_GHOSTS, T.SEE_GHOSTS_ONE),
    ]
    return T.SEE_SEP.join(parts)


def _fmt_value(v: float) -> str:
    a = abs(v)
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{v:.3g}"


def card_message(kind: str, ids: Sequence[int], *, tiers: Sequence[str] = (),
                 gate=None) -> str:
    """The card's source line for a selection of ``ids`` made by ``kind``
    (``""`` for a kind that says nothing, or no selection)."""
    n = len(ids)
    if not n:
        return ""
    if kind == "lasso":
        return T.count_text(n, T.CARD_MSG_LASSO, T.CARD_MSG_LASSO_ONE)
    if kind == "legend":
        tier = " + ".join(_word(t) for t in tiers) if tiers else ""
        return T.count_text(n, T.CARD_MSG_LEGEND, T.CARD_MSG_LEGEND_ONE,
                            tier=tier)
    if kind == "click":
        return T.CARD_MSG_CLICK.format(id=int(ids[0]))
    if kind == "echo":
        return T.count_text(n, T.CARD_MSG_ECHO, T.CARD_MSG_ECHO_ONE)
    if kind == "kept":
        return T.count_text(n, T.CARD_MSG_KEPT, T.CARD_MSG_KEPT_ONE)
    if kind == "violin":
        group = " + ".join(_word(t) for t in tiers) if tiers else ""
        return T.count_text(n, T.CARD_MSG_VIOLIN, T.CARD_MSG_VIOLIN_ONE,
                            group=group)
    if kind == "gate" and gate is not None:
        feature, lo, hi = gate
        return T.count_text(n, T.CARD_MSG_GATE, T.CARD_MSG_GATE_ONE,
                            feature=feature, lo=_fmt_value(lo),
                            hi=_fmt_value(hi))
    return ""


#: Which selection sources switch the lower area.
SWITCH_TO_SELECTION = ("lasso", "legend", "kept")
NO_SWITCH = ("click", "echo", "gate", "refresh")
SOURCES = ("plot", "clear") + SWITCH_TO_SELECTION + NO_SWITCH


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------

class ElidedLabel(QLabel):
    """A one-line label that ELIDES when narrow, while ``text()`` stays the
    full sentence (the tooltip carries it too).  Its minimum width is 0, so
    it never pushes the row sideways."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextFormat(Qt.PlainText)
        self.setWordWrap(False)

    def setText(self, text):  # noqa: N802 (Qt API)
        text = "" if text is None else str(text)
        super().setText(text)
        self.setToolTip(text)
        self.updateGeometry()

    def minimumSizeHint(self):  # noqa: N802
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        rect = self.contentsRect()
        elided = self.fontMetrics().elidedText(self.text(), Qt.ElideRight,
                                               rect.width())
        self.style().drawItemText(
            p, rect, int(self.alignment()), self.palette(), self.isEnabled(),
            elided, QPalette.WindowText)
        p.end()


class MixBar(QWidget):
    """The Selection card's tier-mix bar: one coloured share per tier
    (a copy of the Workbench's ``_TierMixBar``)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._shares: list = []
        self.setFixedHeight(10)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_counts(self, counts: list):
        """``counts`` = ``[(colour, n), ...]``; an empty list clears it."""
        self._shares = [(c, int(n)) for c, n in counts if int(n) > 0]
        self.update()

    def shares(self) -> list:
        return list(self._shares)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(255, 255, 255, 15))
        total = sum(n for _, n in self._shares)
        x = 0.0
        for colour, n in self._shares:
            if total <= 0:
                break
            dw = w * n / total
            p.fillRect(int(round(x)), 0, int(round(x + dw)) - int(round(x)), h,
                       QColor(colour))
            x += dw
        p.end()


class DividerHandle(QSplitterHandle):
    """The divider between the plot and the lower area (a copy of the
    Workbench's handle): 9 px, a grip that takes the accent on hover; a
    double click asks for a reset."""

    reset_requested = Signal()

    def __init__(self, orientation, parent, accent: str = theme.PANEL_GOLD):
        super().__init__(orientation, parent)
        self._accent = accent
        self.setAttribute(Qt.WA_Hover, True)
        self.setMouseTracking(True)
        self._hover = False

    def enterEvent(self, event):  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.reset_requested.emit()
        event.accept()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        w = self.width()
        p.fillRect(0, 0, w, self.height(), QColor(255, 255, 255, 8))
        p.fillRect(0, 0, w, 1, QColor("#555555"))
        grip = (QColor(self._accent) if self._hover
                else QColor(255, 255, 255, 72))
        p.fillRect(w // 2 - 18, 4, 36, 2, grip)
        p.end()


class DividerSplitter(QSplitter):
    """A vertical QSplitter whose handle is a :class:`DividerHandle`."""

    reset_requested = Signal()

    def __init__(self, orientation, parent=None, *,
                 accent: str = theme.PANEL_GOLD):
        self._accent = accent
        super().__init__(orientation, parent)

    def createHandle(self):  # noqa: N802
        handle = DividerHandle(self.orientation(), self,
                               getattr(self, "_accent", theme.PANEL_GOLD))
        handle.reset_requested.connect(self.reset_requested)
        return handle


def _layout_holds(lay, widget) -> bool:
    for i in range(lay.count()):
        item = lay.itemAt(i)
        wd = item.widget()
        if wd is not None and (wd is widget or wd.isAncestorOf(widget)):
            return True
        sub = item.layout()
        if sub is not None and _layout_holds(sub, widget):
            return True
    return False


def layout_height_without(lay, width: int, skip) -> int:
    """``lay``'s height for ``width`` with the widget ``skip`` (and only it)
    counted as 0 px; hidden widgets and empty items drop out, as QBoxLayout
    drops them (a copy of the Workbench's ``_layout_height_without``)."""
    m = lay.contentsMargins()
    inner = max(1, width - m.left() - m.right())
    total = m.top() + m.bottom()
    n = 0
    for i in range(lay.count()):
        item = lay.itemAt(i)
        wd = item.widget()
        if wd is not None and wd.isHidden():
            continue
        if item.isEmpty() and item.layout() is None:
            continue
        n += 1
        if wd is skip:
            continue
        sub = None
        if wd is not None and wd.isAncestorOf(skip):
            sub = wd.layout()
        elif wd is None and item.layout() is not None:
            if _layout_holds(item.layout(), skip):
                sub = item.layout()
        if sub is not None:
            total += layout_height_without(sub, inner, skip)
        elif item.hasHeightForWidth():
            total += item.heightForWidth(inner)
        else:
            total += item.sizeHint().height()
    spacing = max(0, lay.spacing())
    return int(total + spacing * max(0, n - 1))


# ---------------------------------------------------------------------------
# The Selection card
# ---------------------------------------------------------------------------

class SelectionCard(QWidget):
    """``{n} labels selected`` · a five-tier mix bar · chips · a message ·
    the totals line; with nothing selected, ``CARD_NONE`` on a neutral
    edge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("qcSelectionCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.count_label = QLabel(self)
        self.count_label.setStyleSheet(_COUNT_STYLE)
        row.addWidget(self.count_label)
        self.mix_bar = MixBar(self)
        row.addWidget(self.mix_bar, 1)
        lay.addLayout(row)
        self.none_label = QLabel(self)
        self.none_label.setWordWrap(True)
        self.none_label.setStyleSheet(_NONE_STYLE)
        lay.addWidget(self.none_label)
        self.chips_label = QLabel(self)
        self.chips_label.setTextFormat(Qt.RichText)
        self.chips_label.setWordWrap(True)
        self.chips_label.setStyleSheet(_CHIPS_STYLE)
        lay.addWidget(self.chips_label)
        self.message_label = QLabel(self)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(_MSG_STYLE)
        lay.addWidget(self.message_label)
        self.totals_label = QLabel(self)
        self.totals_label.setWordWrap(True)
        self.totals_label.setStyleSheet(_TOTALS_STYLE)
        lay.addWidget(self.totals_label)
        self._message_kind = "info"
        self._has_selection = False
        self.show_parts([], "", "")

    # -- what the card shows ------------------------------------------------
    def has_selection(self) -> bool:
        return self._has_selection

    def show_parts(self, parts, message: str, totals: str, *,
                   n: Optional[int] = None):
        """``parts`` from :func:`selection_card_parts` over the selected
        labels (``[]`` = nothing selected)."""
        count = sum(k for _, k, _ in parts) if n is None else int(n)
        self._has_selection = count > 0
        if self._has_selection:
            self.count_label.setText(T.count_text(
                count, T.CARD_SELECTED, T.CARD_SELECTED_ONE))
            self.count_label.show()
            self.mix_bar.set_counts([(c, k) for _, k, c in parts])
            self.mix_bar.show()
            self.none_label.hide()
            chips = "&nbsp;&nbsp;&nbsp;".join(
                f'<span style="color:{c}">&#9679;</span>&nbsp;'
                + html.escape(T.CARD_TIER_COUNT.format(tier=_word(t),
                                                       n=_num(k)))
                for t, k, c in parts)
            self.chips_label.setText(chips)
            self.chips_label.show()
        else:
            self.count_label.hide()
            self.mix_bar.set_counts([])
            self.mix_bar.hide()
            self.none_label.setText(T.CARD_NONE)
            self.none_label.show()
            self.chips_label.hide()
        self.set_message(message, "info")
        self.totals_label.setText(totals)
        self.totals_label.setVisible(bool(totals))
        self.setStyleSheet(_card_style(
            _CARD_EDGE_ON if self._has_selection else _CARD_EDGE_NONE))

    def set_message(self, text: str, kind: str = "info"):
        """The message line: neutral text, or a ``PANEL_WARN_STYLE`` block
        for ``kind="warn"`` (something refused)."""
        if kind not in T.STATUS_KINDS:
            raise ValueError(f"card message kind must be one of "
                             f"{T.STATUS_KINDS}, got {kind!r}")
        self._message_kind = kind
        self.message_label.setText(text or "")
        if kind == "warn":
            self.message_label.setStyleSheet(theme.PANEL_WARN_STYLE)
        elif kind == "error":
            self.message_label.setStyleSheet(theme.PANEL_ERROR_STYLE)
        else:
            self.message_label.setStyleSheet(_MSG_STYLE)
        self.message_label.setVisible(bool(text))

    def message(self) -> Tuple[str, str]:
        return self.message_label.text(), self._message_kind


# ---------------------------------------------------------------------------
# The area
# ---------------------------------------------------------------------------

class QCPlotArea(QWidget):
    """The right column of Step 3's Curation.

    Public API:

    * attributes ``qc_plot_widget``, ``panel`` (the histogram),
      ``card``, ``see_label``, ``status_label``, ``colours_in_viewer_cb``,
      ``action_row``, ``assign_label``, ``tier_combo``, ``assign_btn``,
      ``undo_btn``, ``clear_btn``, ``tab_histogram``, ``tab_selection``;
    * :meth:`selection_changed` (the ONE door the controller pushes),
      :meth:`refresh_card`, :meth:`set_dataframe`, :meth:`reset`;
    * :meth:`show_histogram` / :meth:`show_selection` / :meth:`current_view`;
    * :meth:`set_locked`, :meth:`set_enabled_state`,
      :meth:`set_card_message`, :meth:`refresh_see_line`;
    * the divider: :meth:`reset_divider`, :meth:`keep_plot_height`,
      :meth:`pinned_chrome_height`, :meth:`pinned_minimum_px`,
      :meth:`lower_minimum_px`.
    """

    def __init__(self, parent=None, *, tier_combo: Optional[QComboBox] = None,
                 assign_btn: Optional[QPushButton] = None,
                 undo_btn: Optional[QPushButton] = None):
        super().__init__(parent)
        self._palette = as_plot_palette(theme.STEP3_PLOT_PALETTE)
        H = T.STEP3_PLOT_HELP
        self._df = None
        self._locked = False
        self._has_df = False
        self._can_undo = False
        self._view = "histogram"
        self._msg_source = ""
        # the Violin view (Lasso turned off there, back on after)
        self._violin_shown = False
        self._lasso_before_violin = False
        self._see_sentence = T.SEE_NO_PATCH
        self._status = ("", "info")
        # A receipt ("Assigned…", "Undid…") is up: the next selection,
        # plot-type, Colour by or Ghosts change clears it (P2 audit MED-6).
        self._receipt = False
        self._pending_receipt = False
        self._last_ids: Tuple[int, ...] = ()
        self._divider_was_reset = False
        self._divider_user_moved = False
        self._last_chrome = None
        self._pinned_w = -1

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # --- row 2: the see-line ------------------------------------------
        see_row = QHBoxLayout()
        see_row.setContentsMargins(0, 0, 0, 0)
        see_row.setSpacing(10)
        self.see_label = ElidedLabel(self)
        self.see_label.setStyleSheet(_SEE_STYLE)
        self.see_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        see_row.addWidget(self.see_label, 1)
        self._see_sep = QFrame(self)
        self._see_sep.setFixedSize(1, 14)
        self._see_sep.setStyleSheet(_SEP_STYLE)
        see_row.addWidget(self._see_sep)
        self.status_label = ElidedLabel(self)
        self.status_label.setSizePolicy(QSizePolicy.Preferred,
                                        QSizePolicy.Preferred)
        self.status_label.setStyleSheet(_STATUS_STYLE.format(
            c=STATUS_INFO_COLOUR))
        see_row.addWidget(self.status_label)
        self.colours_in_viewer_cb = QCheckBox(T.COLOURS_IN_VIEWER, self)
        self.colours_in_viewer_cb.setChecked(True)
        self.colours_in_viewer_cb.setStyleSheet(theme.PANEL_CHECKBOX_STYLE)
        self.colours_in_viewer_cb.setToolTip(H["Colours in viewer"])
        self.colours_in_viewer_cb.setSizePolicy(QSizePolicy.Fixed,
                                                QSizePolicy.Fixed)
        see_row.addWidget(self.colours_in_viewer_cb)
        root.addLayout(see_row)

        # --- rows 3-7: the plot over the divider over the lower area -------
        self._splitter = DividerSplitter(Qt.Vertical, self,
                                         accent=self._palette.accent)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(HANDLE_PX)
        self._splitter.reset_requested.connect(self.reset_divider)
        root.addWidget(self._splitter, 1)

        pinned = QWidget(self._splitter)
        pinned.setObjectName("qcPlotPinned")
        pin_lay = QVBoxLayout(pinned)
        # 4 px under the legend, so it never touches the divider's line.
        pin_lay.setContentsMargins(0, 0, 0, PINNED_BOTTOM_PX)
        pin_lay.setSpacing(0)
        self._pinned = pinned
        self.qc_plot_widget = QCPlotWidget(
            pinned, layout="workbench", texts=T.STEP3_PLOT_TEXTS,
            palette=theme.STEP3_PLOT_PALETTE, tier_scheme=STEP3_SCHEME)
        pin_lay.addWidget(self.qc_plot_widget, 1)
        # The plot clamps at WB_PLOT_MAX_H: the spare room goes here.
        pin_lay.addStretch(0)

        lower = QWidget(self._splitter)
        lower.setObjectName("qcPlotLower")
        low_lay = QVBoxLayout(lower)
        low_lay.setContentsMargins(0, 0, 0, 0)
        low_lay.setSpacing(6)
        self._lower = lower
        tabs = QHBoxLayout()
        tabs.setContentsMargins(0, 0, 0, 0)
        tabs.setSpacing(6)
        toggle = _wb_toggle_style(self._palette)
        self.tab_histogram = QPushButton(T.TAB_HISTOGRAM, lower)
        self.tab_selection = QPushButton(T.TAB_SELECTION, lower)
        self._tab_group = QButtonGroup(self)
        self._tab_group.setExclusive(True)
        for btn, tip in ((self.tab_histogram, H["Histogram tab"]),
                         (self.tab_selection, H["Selection tab"])):
            btn.setCheckable(True)
            btn.setStyleSheet(toggle)
            btn.setToolTip(tip)
            btn.setFocusPolicy(Qt.TabFocus)
            self._tab_group.addButton(btn)
            tabs.addWidget(btn)
        # A fixed-width count slot: the tab never changes width.
        fm = self.tab_selection.fontMetrics()
        self.tab_selection.setMinimumWidth(
            fm.horizontalAdvance(T.TAB_SELECTION_COUNT.format(n="88,888"))
            + 2 * 8 + 8)
        self.tab_histogram.clicked.connect(self.show_histogram)
        self.tab_selection.clicked.connect(self.show_selection)
        tabs.addStretch(1)
        low_lay.addLayout(tabs)

        pages = QWidget(lower)
        pages_lay = QVBoxLayout(pages)
        pages_lay.setContentsMargins(0, 0, 0, 0)
        pages_lay.setSpacing(0)
        self._pages = pages
        self.panel = TriageHistogramPanel(pages)
        self._apply_panel_help()
        pages_lay.addWidget(self.panel, 1)
        self._card_page = QWidget(pages)
        card_lay = QVBoxLayout(self._card_page)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)
        self.card = SelectionCard(self._card_page)
        card_lay.addWidget(self.card)
        card_lay.addStretch(1)
        pages_lay.addWidget(self._card_page, 1)
        low_lay.addWidget(pages, 1)

        self._splitter.addWidget(pinned)
        self._splitter.addWidget(lower)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        for i in range(self._splitter.count()):
            handle = self._splitter.handle(i)
            if handle is not None:
                handle.setToolTip(H["Divider"])

        # --- row 9: the pinned action row ---------------------------------
        row = QWidget(self)
        row.setObjectName("qcActionRow")
        row.setAttribute(Qt.WA_StyledBackground, True)
        row.setStyleSheet(_ACTION_ROW_STYLE)
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 6, 0, 0)
        row_lay.setSpacing(6)
        self.action_row = row
        self.assign_label = QLabel(row)
        self.assign_label.setStyleSheet(_ACTION_LABEL_STYLE)
        self.assign_label.setToolTip(H["Assign"])
        row_lay.addWidget(self.assign_label)
        if tier_combo is None:
            tier_combo = QComboBox()
            tier_combo.addItems(list(TIERS))
        self.tier_combo = tier_combo
        tier_combo.setParent(row)
        tier_combo.setStyleSheet(theme.PANEL_INPUT_STYLE)
        tier_combo.setToolTip(H["Assign"])
        tier_combo.setMinimumWidth(0)
        tier_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        tier_combo.setMinimumContentsLength(6)
        row_lay.addWidget(tier_combo, 1)
        if assign_btn is None:
            assign_btn = QPushButton()
        self.assign_btn = assign_btn
        assign_btn.setParent(row)
        assign_btn.setText(T.ACTION_ASSIGN)
        assign_btn.setToolTip(H["Assign"])
        assign_btn.setStyleSheet(_ASSIGN_STYLE)
        assign_btn.setMaximumWidth(16777215)
        row_lay.addWidget(assign_btn)
        if undo_btn is None:
            undo_btn = QPushButton()
        self.undo_btn = undo_btn
        undo_btn.setParent(row)
        undo_btn.setText(T.ACTION_UNDO)
        undo_btn.setToolTip(H["Undo assign"])
        undo_btn.setStyleSheet(theme.PANEL_BTN_STYLE)
        undo_btn.setMaximumWidth(16777215)
        row_lay.addWidget(undo_btn)
        clear = self.qc_plot_widget.take_clear_button()
        self.clear_btn = clear
        clear.setParent(row)
        clear.setText(T.ACTION_CLEAR)
        clear.setToolTip(H["Clear"])
        clear.setStyleSheet(theme.PANEL_BTN_STYLE)
        clear.show()
        row_lay.addWidget(clear)
        root.addWidget(row)
        self.status_label.setToolTip(H["Status"])
        # The legend's ghosts entry is plain text and cannot carry its own
        # tooltip: the legend line carries it (P2 audit LOW-14).
        self.qc_plot_widget._legend_label.setToolTip(H["Ghosts legend"])

        # --- signals --------------------------------------------------------
        pw = self.qc_plot_widget
        pw.stats_changed.connect(self.refresh_see_line)
        pw.colour_by_changed.connect(self._on_chrome_maybe_changed)
        pw.ghosts_toggled.connect(self._on_ghosts_toggled)
        pw.status_changed.connect(self._on_status_changed)
        pw.plot_type_combo.currentIndexChanged.connect(
            self._on_chrome_maybe_changed)
        pw.set_selection_kept_callback(self._on_selection_kept)
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        self._keep_timer = QTimer(self)
        self._keep_timer.setSingleShot(True)
        self._keep_timer.setInterval(0)
        self._keep_timer.timeout.connect(self.keep_plot_height)
        self._min_timer = QTimer(self)
        self._min_timer.setSingleShot(True)
        self._min_timer.setInterval(0)
        self._min_timer.timeout.connect(self._update_pinned_minimum)
        self._pinned.installEventFilter(self)
        # Every chrome change (a field shown, the gradient legend, a wrapped
        # row) ends in a layout request on the plot widget -- also the ones
        # no signal announces (set_features showing Scale / Colours): keep
        # the plot's height then (coalesced, one loop turn later).
        self.qc_plot_widget.installEventFilter(self)

        # Floors: the lower area at the tab strip + the histogram's own
        # minimum (derived, never a hand-added constant).
        self._lower.setMinimumHeight(self.lower_minimum_px())
        self.show_histogram()
        self.reset(T.PLOT_NO_PATCH, T.SEE_NO_PATCH)

    # ------------------------------------------------------------------
    # Help on the histogram's controls (the area's words, STEP3_PLOT_HELP)
    # ------------------------------------------------------------------
    def _apply_panel_help(self):
        H = T.STEP3_PLOT_HELP
        p = self.panel
        for attr, key in (("_feature_combo", "Triage by"),
                          ("_color_combo", "Colour bars by"),
                          ("_reset_btn", "Reset lines"),
                          ("_gate_cb", "Gate")):
            w = getattr(p, attr, None)
            if w is not None:
                w.setToolTip(H[key])

    # ------------------------------------------------------------------
    # The lower area's two views
    # ------------------------------------------------------------------
    def show_histogram(self):
        self._show_view("histogram")

    def show_selection(self):
        self._show_view("selection")

    def _show_view(self, view: str):
        self._view = view
        hist = view == "histogram"
        self.panel.setVisible(hist)
        self._card_page.setVisible(not hist)
        for btn, on in ((self.tab_histogram, hist),
                        (self.tab_selection, not hist)):
            if btn.isChecked() != on:
                btn.setChecked(on)

    def current_view(self) -> str:
        """``"histogram"`` or ``"selection"``."""
        return self._view

    # ------------------------------------------------------------------
    # The ONE selection door
    # ------------------------------------------------------------------
    def selection_changed(self, source: str = "plot"):
        """The controller says the selection changed, and through which
        door:

        * ``"lasso"`` / ``"legend"`` / ``"kept"`` -> the Selection view;
        * ``"clear"`` -> the Histogram view, the status cleared;
        * ``"click"`` / ``"echo"`` / ``"gate"`` / ``"refresh"`` -> no switch;
        * ``"plot"`` -> read off the plot (``selection_kind()``).

        Always: the card, the tab count, the see-line and the action row's
        label are rebuilt from the plot's LIVE selection; a selection that
        did not come from the gate unticks Gate silently.
        """
        if source not in SOURCES:
            raise ValueError(f"unknown selection source {source!r}")
        pw = self.qc_plot_widget
        ids = pw.selected_label_ids()
        if tuple(ids) != self._last_ids:
            self._last_ids = tuple(ids)
            self._clear_receipt()
        kind = source
        if source == "plot":
            k = pw.selection_kind()
            kind = k if (k and ids) else "clear"
        if kind == "clear":
            self._msg_source = ""
            pw.set_status("", "info")
            self.show_histogram()
        elif kind in SWITCH_TO_SELECTION and ids:
            self.show_selection()
        if kind != "refresh":
            self._msg_source = kind if ids else ""
            if (self._msg_source == "legend"
                    and pw.plot_type_combo.currentText() == "Violin"):
                self._msg_source = "violin"
        if ids and kind not in ("gate", "refresh", "clear"):
            self.panel.untick_gate_silently()
        self.refresh_card()

    def _on_selection_kept(self):
        """A plot-type switch kept the selection."""
        self.selection_changed("kept")

    def refresh_card(self):
        """Rebuild the card, the tab count, the see-line and the action
        row's label from the live selection and the table the host set."""
        pw = self.qc_plot_widget
        ids = pw.selected_label_ids()
        df = self._df
        parts = []
        if ids and df is not None and "label" in df.columns:
            active = _active_mask(df)
            labels = df["label"].to_numpy()
            pick = active & np.isin(labels, np.asarray(ids, dtype=np.int64))
            tiers = (df["qc_tier"].to_numpy()[pick].tolist()
                     if "qc_tier" in df.columns else [""] * int(pick.sum()))
            parts = selection_card_parts(tiers)
        elif ids:
            parts = selection_card_parts([""] * len(ids))
        n = len(ids)
        message = ""
        if ids:
            # the legend's words of whatever column was picked ("Border",
            # "C3", "good") -- a non-tier pick read "Legend: , 153 labels."
            tiers_picked = pw.picked_category_words()
            gate = self.panel.gate_range() if self._msg_source == "gate" else None
            message = card_message(self._msg_source, ids, tiers=tiers_picked,
                                   gate=gate)
        self.card.show_parts(parts, message, totals_line(df), n=n)
        self.tab_selection.setText(
            T.TAB_SELECTION_COUNT.format(n=_num(n)) if n else T.TAB_SELECTION)
        self.assign_label.setText(
            T.ACTION_ASSIGN_LABEL.format(n=_num(n)) if n
            else T.ACTION_ASSIGN_LABEL_NONE)
        self._apply_enables()
        self.refresh_see_line()

    def set_card_message(self, text: str, kind: str = "info"):
        """Write the card's message line (``"warn"``: an amber block)."""
        self.card.set_message(text, kind)

    # ------------------------------------------------------------------
    # The see-line and the status slot
    # ------------------------------------------------------------------
    def refresh_see_line(self, *_):
        """The see-line and the status slot, from ONE ``see_counts()``
        read (it runs on every plot recount)."""
        counts = self.qc_plot_widget.see_counts()
        has_rows = (counts["labels"] + counts["ghosts"]) > 0
        if has_rows:
            self.see_label.setText(see_line_text(counts))
        else:
            self.see_label.setText(self._see_sentence)
        self._sync_plot_controls(has_rows)
        self._render_status(counts)

    def _on_status_changed(self, text: str, kind: str):
        self._status = (text or "", kind or "info")
        self._receipt = bool(self._pending_receipt and text)
        self._pending_receipt = False
        self._render_status()

    def set_receipt(self, text: str):
        """Show an action's receipt in the status slot ("Assigned 15 labels
        to review.", "Undid the last assignment."); it stays until the next
        selection, plot-type, Colour by or Ghosts change."""
        self._pending_receipt = True
        self.qc_plot_widget.set_status(text, "info")

    def _clear_receipt(self):
        if self._receipt:
            self._receipt = False
            self.qc_plot_widget.set_status("", "info")

    def _render_status(self, counts: Optional[dict] = None):
        text, kind = self._status
        if not text and not self._locked:
            # (no "Lasso or click…" while the lock greys Lasso, MED-6)
            if counts is None:
                counts = self.qc_plot_widget.see_counts()
            has_rows = (counts["labels"] + counts["ghosts"]) > 0
            if has_rows and not counts["selected"]:
                if not self._on_violin():
                    text, kind = T.STATUS_IDLE_HINT, "info"
                elif self.qc_plot_widget.violin_groups_drawn():
                    text, kind = T.STATUS_IDLE_HINT_VIOLIN, "info"
                # (no violin drawn: VIOLIN_NO_VALUES is on the plot)
        colour = {"warn": STATUS_WARN_COLOUR,
                  "error": STATUS_ERROR_COLOUR}.get(kind, STATUS_INFO_COLOUR)
        self.status_label.setStyleSheet(_STATUS_STYLE.format(c=colour))
        self.status_label.setText(text)
        self.status_label.setToolTip(text or T.STEP3_PLOT_HELP["Status"])
        self.status_label.setVisible(bool(text))
        self._see_sep.setVisible(bool(text))
        self._status_kind_shown = kind

    def status_shown(self) -> Tuple[str, str]:
        """``(text, kind)`` the status slot shows now (the idle hint too)."""
        return self.status_label.text(), getattr(self, "_status_kind_shown",
                                                 "info")

    def _on_violin(self) -> bool:
        return self.qc_plot_widget.plot_type_combo.currentText() == "Violin"

    def _sync_plot_controls(self, has_rows: bool):
        """The toolbar and the axes row are greyed with nothing to plot
        (empty states; Ghosts stays ticked, dimmed).

        On Violin, Lasso and Density draw nothing: they stay IN PLACE but
        greyed (the plot hides Lasso there, and the toolbar shifted), the
        idle hint says to click a violin, and a Lasso left on from a map is
        turned off for the Violin and back on when the user leaves it
        (left on, it swallowed every violin click)."""
        pw = self.qc_plot_widget
        violin = self._on_violin()
        lasso = getattr(pw, "_select_btn", None)
        if lasso is not None:
            if lasso.isHidden():
                lasso.setVisible(True)
            if violin and not self._violin_shown:
                self._lasso_before_violin = lasso.isChecked()
                if lasso.isChecked():
                    lasso.setChecked(False)
            elif not violin and self._violin_shown:
                if self._lasso_before_violin and not lasso.isChecked():
                    lasso.setChecked(True)
                self._lasso_before_violin = False
        self._violin_shown = violin
        for name in ("_wb_seg", "_select_btn", "_density_btn", "_history_cb",
                     "_wb_x_field", "_wb_y_field", "_wb_colour_field",
                     "_wb_scale_field", "_wb_cmap_field"):
            w = getattr(pw, name, None)
            on = has_rows and not (violin and name in ("_select_btn",
                                                       "_density_btn"))
            if w is not None and w.isEnabled() != on:
                w.setEnabled(on)

    def _on_ghosts_toggled(self, shown: bool):
        self._clear_receipt()
        self.panel.set_ghosts_visible(bool(shown))
        self.refresh_see_line()
        self._schedule_keep_plot()

    def _on_chrome_maybe_changed(self, *_):
        self._clear_receipt()
        self.refresh_see_line()
        self._schedule_keep_plot()

    # ------------------------------------------------------------------
    # The table, the lock, the enables, the reset
    # ------------------------------------------------------------------
    def set_dataframe(self, df):
        """The table the card counts (``None``: no patch / no features)."""
        self._df = df
        self._has_df = df is not None and len(df) > 0
        self.refresh_card()

    def dataframe(self):
        return self._df

    def set_locked(self, locked: bool):
        """Grey the histogram's controls and lines, the action row (the
        Clear proxy re-enables Clear on every selection change, so the
        CONTAINER is disabled) -- the tabs, the divider and the colours
        tick stay live (lifecycle)."""
        self._locked = bool(locked)
        self.panel.set_locked(self._locked)
        self.action_row.setEnabled(not self._locked)
        self._render_status()

    def is_locked(self) -> bool:
        return self._locked

    def set_enabled_state(self, has_df: bool, has_selection: bool,
                          can_undo: bool):
        """The action row's honest gates: the tier list with a
        table, Assign with a table and a selection, Undo assign with a
        table and something to undo.  The lock sits on top (the row)."""
        self._has_df = bool(has_df)
        self._can_undo = bool(can_undo)
        self.tier_combo.setEnabled(self._has_df)
        self.assign_btn.setEnabled(self._has_df and bool(has_selection))
        self.undo_btn.setEnabled(self._has_df and self._can_undo)

    def _apply_enables(self):
        has_sel = bool(self.qc_plot_widget.selected_label_ids())
        self.assign_btn.setEnabled(self._has_df and has_sel)

    def reset(self, sentence_plot: str = T.PLOT_NO_PATCH,
              sentence_see: str = T.SEE_NO_PATCH):
        """An empty area (a patch switch starts, a run switch, Browse, a
        patch without features): no rows, no status, the Histogram view,
        an empty card, ``sentence_see`` on the see-line and
        ``sentence_plot`` in the plot.  Never ``set_features(empty)``,
        which leaves a stale legend."""
        pw = self.qc_plot_widget
        self._see_sentence = sentence_see or ""
        self._receipt = self._pending_receipt = False
        self._last_ids = ()
        pw.set_empty_sentence(sentence_plot)
        pw.clear()
        pw.set_status("", "info")
        self.panel.set_data(None)
        self._msg_source = ""
        self.card.set_message("", "info")
        self.show_histogram()
        self.set_dataframe(None)
        self.set_enabled_state(False, False, False)

    # ------------------------------------------------------------------
    # The divider (the Workbench's rules)
    # ------------------------------------------------------------------
    def lower_minimum_px(self) -> int:
        """The lower area's floor: the tab strip + the histogram panel's
        own minimum + their spacing, WITH room for the panel's one-line
        note whether or not it shows now -- the note appears
        on qc_score, and a floor without it let the note overprint the
        histogram's axis title (live driver, P2).  Derived, never a
        hand-added constant, and the same in both views so the divider
        never jumps."""
        tabs = max(self.tab_histogram.sizeHint().height(),
                   self.tab_selection.sizeHint().height())
        spacing = max(0, self._lower.layout().spacing())
        note = getattr(self.panel, "_note_label", None)
        if note is None or note.isHidden() or not hasattr(self, "_panel_bare"):
            # measured with the note hidden (at build, and at every show
            # while it is hidden), then the note's line reserved on top
            self._panel_bare = self.panel.minimumSizeHint().height()
            if note is not None and not note.isHidden():
                self._panel_bare -= (note.minimumSizeHint().height()
                                     + max(0, self.panel.layout().spacing()))
        reserve = 0
        if note is not None:
            m = note.contentsMargins()
            reserve = (note.fontMetrics().height() + m.top() + m.bottom()
                       + max(0, self.panel.layout().spacing()))
        return int(tabs + spacing + self._panel_bare + reserve)

    def pinned_chrome_height(self, width: Optional[int] = None) -> int:
        """The plot part's height WITHOUT the plot canvas at ``width``:
        toolbar, axes row (wrapped rows count), the legend lines."""
        pw = self.qc_plot_widget
        w = width if width is not None else self._pinned.width()
        return layout_height_without(self._pinned.layout(), max(1, int(w)),
                                     pw._plot_widget)

    def pinned_minimum_px(self, width: Optional[int] = None) -> int:
        """The divider's top stop: the chrome at ``width`` + the plot's
        150 px floor (W2: a wrapped axes row counts)."""
        return self.pinned_chrome_height(width) + WB_PLOT_MIN_H

    def plot_default_px(self) -> int:
        return PLOT_DEFAULT_PX

    def _update_pinned_minimum(self):
        pinned = self._pinned
        if pinned.width() <= 0:
            return
        need = self.pinned_minimum_px(pinned.width())
        if need != pinned.minimumHeight():
            pinned.setMinimumHeight(need)
            self.updateGeometry()

    def minimumSizeHint(self):  # noqa: N802
        """What the area ASKS its section for: room for the plot at its
        default 270 px and the lower area at its floor + 30 px (P2 audit
        HIGH-2 -- with the histogram gone from the left column nothing else
        asks the Curation section for height, so the column got exactly
        its floors and the plot sat at 150).  The DIVIDER's floors are
        unchanged: the plot can still be dragged down to 150."""
        hint = super().minimumSizeHint()
        extra = (PLOT_DEFAULT_PX - WB_PLOT_MIN_H) + LOWER_SLACK_PX
        return QSize(hint.width(), hint.height() + extra)

    def reset_divider(self):
        """Double-click (and the first show): the lower area at its floor
        plus 30 px, the plot the rest, capped at 270 px."""
        self._update_pinned_minimum()
        total = sum(self._splitter.sizes())
        if total <= 0:
            return
        chrome = self.pinned_chrome_height()
        floor = self.lower_minimum_px()
        top = min(chrome + PLOT_DEFAULT_PX, total - floor - LOWER_SLACK_PX)
        top = max(self._pinned.minimumHeight(), top)
        top = min(top, max(0, total - floor))
        self._splitter.setSizes([top, max(0, total - top)])
        self._divider_was_reset = True
        self._divider_user_moved = False
        self._last_chrome = chrome

    def _on_splitter_moved(self, *_):
        self._divider_user_moved = True
        self._last_chrome = self.pinned_chrome_height()

    def _schedule_keep_plot(self, *_):
        if not self._keep_timer.isActive():
            self._keep_timer.start()

    def keep_plot_height(self):
        """The plot keeps its height when a chrome line appears or goes:
        the top pane grows or shrinks by the change and the lower area
        gives way, never below its floor."""
        self._update_pinned_minimum()
        if not self._divider_was_reset or self._last_chrome is None:
            return
        chrome = self.pinned_chrome_height()
        delta = chrome - self._last_chrome
        if delta == 0:
            # the chrome did not move -- but the area may have grown (its
            # section answering a bigger minimum): give the plot its 270
            self._restore_default_plot()
            return
        sizes = self._splitter.sizes()
        if len(sizes) != 2:
            return
        top, bottom = sizes
        floor = self.lower_minimum_px()
        if delta > 0:
            give = min(delta, max(0, bottom - floor))
        elif top - self._last_chrome < PLOT_DEFAULT_PX:
            # The chrome shrank (e.g. Scale / Colours went) under a plot
            # shorter than its default: keep the top pane, so the PLOT
            # grows -- up to 270 px -- instead of the lower area.
            give = -max(0, (top - chrome) - PLOT_DEFAULT_PX)
        else:
            give = delta
        if give:
            self._splitter.setSizes([top + give, bottom - give])
        self._last_chrome = chrome
        self._restore_default_plot()

    def _restore_default_plot(self):
        """While the user has not dragged the divider, a plot under its
        default 270 px gets it back as soon as the area has the room (the
        area's minimum grows with the chrome, and the section answers one
        layout pass later)."""
        if (not self._divider_was_reset or self._divider_user_moved
                or self._last_chrome is None):
            return
        sizes = self._splitter.sizes()
        if len(sizes) != 2:
            return
        top, bottom = sizes
        chrome = self.pinned_chrome_height()
        want = chrome + PLOT_DEFAULT_PX - top
        room = bottom - self.lower_minimum_px()
        if want > 0 and room > 0:
            give = min(want, room)
            self._splitter.setSizes([top + give, bottom - give])

    def _first_divider_reset(self):
        if self._divider_was_reset or not self.isVisible():
            return
        need = (self.pinned_chrome_height() + WB_PLOT_MIN_H
                + self.lower_minimum_px())
        if sum(self._splitter.sizes()) >= need:
            self.reset_divider()

    def eventFilter(self, obj, event):  # noqa: N802
        if (obj is self.qc_plot_widget
                and event.type() == QEvent.LayoutRequest):
            self._schedule_keep_plot()
        if (obj is self._pinned and event.type() == QEvent.Resize
                and event.size().width() != self._pinned_w):
            self._pinned_w = event.size().width()
            if not self._min_timer.isActive():
                self._min_timer.start()
        return super().eventFilter(obj, event)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        # The floor again now the styles are polished (font sizes).
        self._lower.setMinimumHeight(self.lower_minimum_px())
        self._first_divider_reset()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if event.size().height() != event.oldSize().height():
            self._schedule_keep_plot()
        self.status_label.setMaximumWidth(
            max(0, int(event.size().width() * STATUS_MAX_SHARE)))
        self._first_divider_reset()

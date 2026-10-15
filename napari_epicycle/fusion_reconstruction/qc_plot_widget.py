"""Interactive QC visualization widget using pyqtgraph.

Supports three plot types:
- Histogram: feature value distribution with threshold lines
- Scatter: two-feature scatter plot with click-to-select
- PCA: 2D PCA projection of all numeric features

Points are colored by a user-selected feature (continuous colormap) or
by QC status after thresholding (green/orange/red categorical).

Scatter and PCA plots support freeform lasso selection for cross-modal
filtering: drag to draw a lasso around points, and all enclosed label IDs
are emitted via ``labels_selected``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from qtpy.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
)
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

# --- Palette ---------------------------------------------------------------
_BG_COLOR = "#1a1a2e"
_GRID_ALPHA = 20
_PASS_COLOR = (80, 200, 120)        # green
_FLAGGED_COLOR = (255, 160, 40)     # warm orange
_EXCLUDED_COLOR = (240, 60, 60)     # red
_DEFAULT_COLOR = (140, 150, 170)    # cool gray-blue

_STATUS_COLORS = {
    "pass": _PASS_COLOR,
    "flagged": _FLAGGED_COLOR,
    "excluded": _EXCLUDED_COLOR,
}

_TIER_COLORS = {
    "curated": (0, 188, 212),    # cyan/teal
    "good": (80, 200, 120),      # green
    "review": (255, 200, 60),    # yellow-orange
    "bad": (255, 120, 40),       # warm orange
    "garbage": (240, 60, 60),    # red
}

# --- Colormaps (5-stop, normalized 0-1 RGB) --------------------------------
_COLORMAPS = {
    "Viridis": np.array([
        [0.267, 0.004, 0.329],
        [0.231, 0.322, 0.545],
        [0.129, 0.569, 0.549],
        [0.369, 0.789, 0.384],
        [0.992, 0.906, 0.145],
    ]),
    "Plasma": np.array([
        [0.051, 0.031, 0.529],
        [0.494, 0.012, 0.659],
        [0.800, 0.278, 0.471],
        [0.976, 0.584, 0.251],
        [0.941, 0.976, 0.129],
    ]),
    "Inferno": np.array([
        [0.001, 0.000, 0.014],
        [0.341, 0.063, 0.431],
        [0.733, 0.216, 0.329],
        [0.976, 0.549, 0.039],
        [0.988, 1.000, 0.643],
    ]),
    "Magma": np.array([
        [0.001, 0.000, 0.014],
        [0.318, 0.071, 0.486],
        [0.718, 0.216, 0.475],
        [0.984, 0.533, 0.380],
        [0.988, 0.992, 0.749],
    ]),
    "Turbo": np.array([
        [0.190, 0.071, 0.231],
        [0.157, 0.733, 0.925],
        [0.635, 0.988, 0.235],
        [0.984, 0.502, 0.133],
        [0.478, 0.016, 0.012],
    ]),
    "Coolwarm": np.array([
        [0.231, 0.298, 0.753],
        [0.553, 0.690, 0.996],
        [0.969, 0.969, 0.969],
        [0.969, 0.655, 0.537],
        [0.706, 0.016, 0.149],
    ]),
}

# Backward-compat alias used by older code paths
_COLORMAP_STOPS = _COLORMAPS["Viridis"]

_COMBO_STYLE = (
    "QComboBox {"
    "  font-size: 10px; padding: 2px 6px;"
    "  background: #1e1e2e; color: #e0e0e0; border: 1px solid rgba(255,255,255,0.10);"
    "  border-radius: 3px;"
    "}"
    "QComboBox:hover { border-color: rgba(106,191,105,0.40); }"
    "QComboBox::drop-down { border: none; }"
    "QComboBox QAbstractItemView {"
    "  background: #2b2b2b; color: #e0e0e0;"
    "  selection-background-color: rgba(106,191,105,0.25);"
    "}"
)

_LABEL_STYLE = "color: #999999; font-size: 10px; font-weight: 500;"

_SELECT_BTN_STYLE = (
    "QPushButton { font-size: 13px; padding: 1px 4px; background: #1e1e2e; "
    "color: #999; border: 1px solid rgba(255,255,255,0.10); border-radius: 3px; }"
    "QPushButton:hover { border-color: rgba(106,191,105,0.30); }"
    "QPushButton:checked { background: rgba(106,191,105,0.15); color: #6abf69; "
    "border-color: rgba(106,191,105,0.40); }"
)


# --- The Step 4 Workbench layout (opt-in: ``layout="workbench"``) ----------
# Nothing below is used by the classic layout
# Step 3 builds; ``tests/fusion_reconstruction/
# test_qc_plot_classic_layout_golden.py`` holds the classic layout still.

#: The layouts ``QCPlotWidget`` can be built in.
PLOT_LAYOUTS = ("classic", "workbench")

#: The plugin dock's neutral surface (the navy ``_BG_COLOR`` goes).
_WB_BG = "#23242b"

#: Unassigned (no tier) gets a colour and a legend entry in the Workbench
#: only (the mockup's ``#8b909a``).  Step 3's colours are not
#: touched.
_WB_UNASSIGNED_RGB = (139, 144, 154)
#: Step 4's one word for "no mark", in the legend too.
_WB_UNASSIGNED = "unmarked"
#: The Workbench's three tier buckets, in legend order, and their colours.
#: ONE rule for the plot and the controller's card (audit E1-5):
#: "garbage" is bad; empty, NaN, "review", "curated" are unmarked.
WB_TIER_BUCKETS = ("good", "bad", "unmarked")
_WB_BUCKET_RGB = {"good": _TIER_COLORS["good"], "bad": _TIER_COLORS["bad"],
                  "unmarked": _WB_UNASSIGNED_RGB}
#: A plot-type switch keeps the selection in the Workbench (audit X1-2):
#: the kept rows ride this sentinel category.
_WB_KEPT = ("__wb_kept__", "")


@dataclass(frozen=True)
class TierScheme:
    """How the Workbench layout groups ``qc_tier`` values into buckets.  Read only under ``layout="workbench"``.

    ``buckets`` is the legend order; its LAST entry is the catch-all a
    tier outside ``buckets`` (empty, NaN, an unknown word) falls into, and
    ``aliases`` sends a tier to another bucket first (Step 4: garbage is
    bad).  ``rgb`` gives each bucket its colour.  ``unassigned_word`` is
    the catch-all's legend word (a ``texts["UNMARKED_WORD"]`` wins).
    ``ghost_entry``: the legend counts the ghosts (plain text) while they
    are shown.  ``ghosts_default_on``: the ghosts toggle starts ticked.
    ``ghosts_behind``: ghosts are drawn small, without an outline, faint
    and behind the active points.  ``show_empty_unassigned``: the legend
    lists the catch-all even when no row is in it.
    """

    name: str
    buckets: Tuple[str, ...]
    rgb: Mapping[str, Tuple[int, int, int]] = field(hash=False)
    unassigned_word: str
    ghost_entry: bool
    ghosts_default_on: bool
    ghosts_behind: bool
    aliases: Mapping[str, str] = field(default_factory=dict, hash=False)
    show_empty_unassigned: bool = True
    #: The legend writes each entry's WORD in its colour (Step 4).  False:
    #: only the dot carries the colour, the word is neutral ``#cccccc``
    #: (Step 3: tier colours never as text).
    colour_words: bool = True
    #: The Violin view (fix/step3-plot-violin): one violin per group of
    #: ACTIVE labels only (ghosts are no group), groups the Colour by names
    #: when it is a categorical with data, else the QC tiers, the legend
    #: the groups' legend, the ticks in the legend's words and no bottom
    #: title (the axes note names the grouping).  Off: Step 4's violin,
    #: unchanged.
    violin_active_groups: bool = False

    def __post_init__(self):
        if not self.buckets:
            raise ValueError("a TierScheme needs at least one bucket")
        missing = [b for b in self.buckets if b not in self.rgb]
        if missing:
            raise ValueError(f"TierScheme {self.name!r}: no colour for {missing}")
        bad = [a for a, b in self.aliases.items() if b not in self.buckets]
        if bad:
            raise ValueError(f"TierScheme {self.name!r}: alias to no bucket {bad}")

    @property
    def unassigned(self) -> str:
        """The catch-all bucket (the last one)."""
        return self.buckets[-1]

    def bucket_of(self, tier) -> str:
        """The bucket of one ``qc_tier`` value."""
        t = "" if tier is None else str(tier)
        t = self.aliases.get(t, t)
        return t if t in self.buckets else self.unassigned

    def buckets_of(self, df) -> np.ndarray:
        """The bucket of every row of ``df`` (all the catch-all with no
        ``qc_tier`` column; empty with no table)."""
        if df is None:
            return np.array([], dtype=object)
        if "qc_tier" not in df.columns:
            return np.array([self.unassigned] * len(df), dtype=object)
        return np.array([self.bucket_of(t) for t in df["qc_tier"].tolist()],
                        dtype=object)

    def colours(self) -> Dict[str, Tuple[int, int, int]]:
        """``{bucket: rgb}`` in legend order."""
        return {b: tuple(self.rgb[b]) for b in self.buckets}


#: Step 4's scheme: good, bad (garbage counts as bad), unmarked; no ghosts
#: entry, ghosts OFF and drawn as they always were.
STEP4_SCHEME = TierScheme(
    name="step4",
    buckets=WB_TIER_BUCKETS,
    rgb=_WB_BUCKET_RGB,
    unassigned_word=_WB_UNASSIGNED,
    ghost_entry=False,
    ghosts_default_on=False,
    ghosts_behind=False,
    aliases={"garbage": "bad"},
)

#: Step 3's colour for a label with no tier (the plot's default grey).
_STEP3_UNASSIGNED_RGB = (140, 150, 170)

#: Step 3's scheme: the five tiers in triage order, then unassigned (only
#: listed when some label has no tier); the ghosts counted in the legend,
#: on by default, drawn behind.
STEP3_SCHEME = TierScheme(
    name="step3",
    buckets=("curated", "good", "review", "bad", "garbage", "unassigned"),
    rgb={**_TIER_COLORS, "unassigned": _STEP3_UNASSIGNED_RGB},
    unassigned_word="unassigned",
    ghost_entry=True,
    ghosts_default_on=True,
    ghosts_behind=True,
    show_empty_unassigned=False,
    colour_words=False,
    violin_active_groups=True,
)


def wb_tier_bucket(tier) -> str:
    """The Step 4 bucket of one ``qc_tier`` value (the ONE rule)."""
    return STEP4_SCHEME.bucket_of(tier)


def wb_tier_buckets(df) -> np.ndarray:
    """The bucket of every row of ``df`` (all unmarked with no column)."""
    return STEP4_SCHEME.buckets_of(df)

@dataclass(frozen=True)
class PlotPalette:
    """The accent the Workbench layout is drawn in.  Defined HERE, not in ``ui_shell.theme``, so this module never
    imports the shell; ``theme.plot_palette(accent)`` builds the mapping
    form, which ``QCPlotWidget(palette=...)`` accepts as well.

    ``accent`` is a ``#rrggbb`` hex, ``accent_rgb`` the same colour as
    three ints, ``ok_text`` the colour a host uses for an "ok" sentence.
    """

    accent: str
    accent_rgb: Tuple[int, int, int]
    ok_text: str


#: The Workbench's own palette, Step 4's green: every default-palette style
#: below is byte-identical to the strings Step 4 shipped with
#: (``tests/fusion_reconstruction/golden/qc_plot_workbench.json``).
DEFAULT_PLOT_PALETTE = PlotPalette("#6abf69", (106, 191, 105), "#6abf69")


def as_plot_palette(palette) -> PlotPalette:
    """``None`` -> the default; a :class:`PlotPalette` as it is; a mapping
    with ``accent`` / ``accent_rgb`` (/ ``ok_text``) -> a PlotPalette."""
    if palette is None:
        return DEFAULT_PLOT_PALETTE
    if isinstance(palette, PlotPalette):
        return palette
    if isinstance(palette, Mapping):
        rgb = tuple(int(c) for c in palette["accent_rgb"])
        if len(rgb) != 3 or not all(0 <= c <= 255 for c in rgb):
            raise ValueError(f"accent_rgb must be three 0-255 ints, got {rgb!r}")
        accent = str(palette["accent"])
        return PlotPalette(accent, rgb, str(palette.get("ok_text", accent)))
    raise TypeError(
        f"palette must be a PlotPalette or a mapping, got {type(palette).__name__}")


def _accent(p: PlotPalette, alpha: str) -> str:
    """``rgba(r,g,b,alpha)`` in the accent, written the way the shipped
    Workbench strings write it (no spaces)."""
    r, g, b = p.accent_rgb
    return f"rgba({r},{g},{b},{alpha})"


def _wb_combo_style(p: PlotPalette) -> str:
    """The Workbench's combo boxes, hover and selection in the accent."""
    return (
        "QComboBox {"
        "  font-size: 11px; padding: 2px 6px; min-height: 20px;"
        "  background: rgba(255,255,255,0.04); color: #dddddd;"
        "  border: 1px solid rgba(255,255,255,0.10); border-radius: 3px;"
        "}"
        f"QComboBox:hover {{ border-color: {_accent(p, '0.40')}; }}"
        "QComboBox:disabled { color: #666666; background: rgba(255,255,255,0.02); }"
        "QComboBox QAbstractItemView {"
        "  background: #2b2d35; color: #dddddd;"
        f"  selection-background-color: {_accent(p, '0.25')};"
        "}"
    )


def _wb_seg_style(p: PlotPalette) -> str:
    """The plot-type segmented row (mockup ``.seg``): the accent only on
    the pressed one.

    The ``:disabled`` / ``:checked:disabled`` rules (a greyed row, its
    pressed segment dimmed, not lit) are APPENDED only for a palette other
    than the default -- the same gate as :func:`_wb_toggle_style`, so
    Step 4's shipped string does not move.
    """
    style = (
        "QPushButton {"
        "  background: rgba(255,255,255,0.04); color: #cccccc; font-size: 11px;"
        "  border: 1px solid rgba(255,255,255,0.10); border-radius: 0px;"
        "  padding: 0px 9px; min-height: 24px; max-height: 24px;"
        "}"
        "QPushButton:hover { background: rgba(255,255,255,0.08); }"
        "QPushButton:checked {"
        f"  background: {_accent(p, '0.28')}; color: #ffffff; font-weight: bold;"
        "}"
    )
    if p != DEFAULT_PLOT_PALETTE:
        style += (
            "QPushButton:disabled { color:#666666; }"
            "QPushButton:checked:disabled { background: "
            f"{_accent(p, '0.10')}; color:#888888; font-weight:bold; }}"
        )
    return style


def _wb_toggle_style(p: PlotPalette) -> str:
    """Lasso / Density / ghosts toggles with words (mockup ``.tg``).

    A ``:checked:disabled`` rule (a greyed ticked toggle dimmed, not lit) is APPENDED only for a palette other than the default:
    added to the default it would change Step 4's shipped string, which
    the Workbench golden pins byte for byte.  Step 4 is therefore drawn
    exactly as before; Step 3's amber toggles get the rule.
    """
    style = (
        "QPushButton, QCheckBox {"
        "  background: rgba(255,255,255,0.05); color: #cccccc; font-size: 11px;"
        "  border: 1px solid rgba(255,255,255,0.10); border-radius: 4px;"
        "  padding: 0px 8px; min-height: 24px; max-height: 24px;"
        "}"
        "QPushButton:checked, QCheckBox:checked {"
        f"  background: {_accent(p, '0.25')}; color: #ffffff;"
        f"  border-color: {_accent(p, '0.50')};"
        "}"
        "QPushButton:disabled, QCheckBox:disabled { color: #666666; }"
        "QCheckBox { spacing: 0px; }"
        "QCheckBox::indicator { width: 0px; height: 0px; border: none;"
        "  background: none; image: none; margin: 0px; }"
    )
    if p != DEFAULT_PLOT_PALETTE:
        style += (
            "QPushButton:checked:disabled, QCheckBox:checked:disabled {"
            f"  background: {_accent(p, '0.10')}; color: #888888;"
            f"  border-color: {_accent(p, '0.25')};"
            "}"
        )
    return style


#: The default-palette styles under their shipped names, for any reader
#: outside this module (Step 4 is drawn from these exact strings).
_WB_COMBO_STYLE = _wb_combo_style(DEFAULT_PLOT_PALETTE)

_WB_LABEL_STYLE = "color: #bbbbbb; font-size: 12px;"

#: The plot-type segmented row (mockup ``.seg``), default palette.
_WB_SEG_STYLE = _wb_seg_style(DEFAULT_PLOT_PALETTE)

#: Lasso / Density / Removed toggles with words, default palette.
_WB_TOGGLE_STYLE = _wb_toggle_style(DEFAULT_PLOT_PALETTE)

#: "✕ Clear", which the Workbench moves into its selection card.
_WB_CLEAR_STYLE = (
    "QPushButton {"
    "  background: rgba(255,255,255,0.04); color: #bbbbbb; font-size: 11px;"
    "  border: 1px solid rgba(255,255,255,0.10); border-radius: 4px;"
    "  padding: 0px 8px; min-height: 22px; max-height: 22px;"
    "}"
    "QPushButton:disabled { color: #666666; }"
)

#: The empty-map text is wrapped to this width.
_WB_EMPTY_TEXT_WIDTH = 360

#: The plot area clamps (the pane heights clamp the plot to
#: 150-560 px).
WB_PLOT_MIN_H = 150
WB_PLOT_MAX_H = 560


class FlowLayout(QLayout):
    """A layout that places its items left to right and wraps them.

    Qt has no wrapping row; the Workbench's axes row (X · Y · Colour by ·
    Scale · Colours) must wrap at 578 px instead of truncating.
    Height-for-width, so a parent asks for the wrapped height at its width.
    """

    def __init__(self, parent=None, *, h_spacing: int = 10, v_spacing: int = 6):
        super().__init__(parent)
        self._items: list = []
        self._h = h_spacing
        self._v = v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def insertWidget(self, index, widget):  # noqa: N802
        """Insert ``widget`` at ``index`` (``addWidget`` appends)."""
        self.addChildWidget(widget)
        item = QWidgetItem(widget)
        index = max(0, min(index, len(self._items)))
        self._items.insert(index, item)
        self.invalidate()

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, apply=True)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect, *, apply: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = eff.x(), eff.y(), 0
        for item in self._items:
            wd = item.widget()
            if wd is not None and wd.isHidden():
                continue
            hint = item.sizeHint()
            w = min(hint.width(), max(1, eff.width()))
            if x > eff.x() and x + w > eff.right() + 1:
                x = eff.x()
                y += line_h + self._v
                line_h = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), QSize(w, hint.height())))
            x += w + self._h
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


#: The words the Workbench layout needs, passed in at construction
#: (``QCPlotWidget(layout="workbench", texts=...)``) so this Step 3 module
#: never imports Step 4's sentence table.  ``help`` maps a control's name
#: to its help line (at least "Plot type", "Lasso", "Density", "Removed",
#: "Clear", "Colour by", "Scale", "Colours").
WB_TEXT_KEYS = ("help", "UMAP_EMPTY", "TSNE_EMPTY", "MAP_BUSY_EMPTY",
                "VIOLIN_GROUPS", "MAP_AXES")
WB_HELP_KEYS = ("Plot type", "Lasso", "Density", "Removed", "Clear",
                "Colour by", "Scale", "Colours")


def _fmt_num(v: float) -> str:
    """The mockup's number format for a legend end."""
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


class _ShownAsEnabled:
    """Workbench: stands in for ``_clear_sel_btn`` so every classic
    ``setVisible(x)`` greys or un-greys the always-shown "✕ Clear"."""

    def __init__(self, button):
        self._button = button

    def setVisible(self, visible):  # noqa: N802
        self._button.setEnabled(bool(visible))

    def isVisible(self):  # noqa: N802
        return self._button.isEnabled()

    def isHidden(self):  # noqa: N802
        return not self._button.isEnabled()

    def __getattr__(self, name):
        return getattr(self._button, name)


class _GradientBar(QWidget):
    """A 140 x 9 px colour-scale bar (mockup ``.grad``)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stops = None
        self.setFixedSize(140, 9)

    def set_stops(self, stops):
        self._stops = None if stops is None else np.asarray(stops)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        if self._stops is None or len(self._stops) == 0:
            return
        p = QPainter(self)
        grad = QLinearGradient(0, 0, self.width(), 0)
        n = len(self._stops)
        for i, c in enumerate(self._stops):
            grad.setColorAt(i / max(1, n - 1), QColor(
                int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(0, 0, self.width(), self.height(), 2, 2)
        p.end()


class _GradientLegend(QWidget):
    """name · lo · gradient · hi · scale word (the mockup's feature legend)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(6)
        style = "color: #bbbbbb; font-size: 11px;"
        self.name_label = QLabel("", self)
        self.lo_label = QLabel("", self)
        self.bar = _GradientBar(self)
        self.hi_label = QLabel("", self)
        self.scale_label = QLabel("", self)
        for lab in (self.name_label, self.lo_label, self.hi_label):
            lab.setStyleSheet(style)
        self.scale_label.setStyleSheet("color: #888888; font-size: 11px;")
        for w in (self.name_label, self.lo_label, self.bar, self.hi_label,
                  self.scale_label):
            lay.addWidget(w)
        lay.addStretch(1)

    def set_legend(self, name, lo, hi, stops, scale):
        self.name_label.setText(name)
        self.lo_label.setText(lo)
        self.hi_label.setText(hi)
        self.bar.set_stops(stops)
        self.scale_label.setText(scale)


class _NotifyingLabel(QLabel):
    """A hidden QLabel that tells its reader when its text is set.

    In the Workbench layout the plot's own "N labels" count is not drawn;
    the Workbench's "what you see" line recounts whenever the plot writes
    it.  Every ``_stats_label.setText`` call site stays as it is.
    """

    text_set = Signal()

    def setText(self, text):  # noqa: N802
        super().setText(text)
        self.text_set.emit()


class _StatusNotifier(QObject):
    """Workbench only: carries ``QCPlotWidget.status_changed``.

    A plain ``QObject`` (never a widget, so the widget tree does not grow)
    holding the signal, so ``QCPlotWidget`` gains no class-level signal --
    the classic layout golden pins that list.
    """

    changed = Signal(str, str)   # (text, kind)


def _sorted_unique_cluster_ids(values) -> list:
    """Return sorted unique cluster IDs as ints, skipping NaN / pd.NA / None.

    2026-05-17 fix: ``cluster_id`` is set to ``pd.NA`` on ghost rows
    by ``_run_clustering`` (so ghosts don't participate in clustering).
    ``sorted(set(values))`` crashes on ``pd.NA`` because its ``__lt__``
    raises ``TypeError: boolean value of NA is ambiguous``.  This
    helper filters non-coercible values at the int-conversion step
    so the legend / color-map code in the plot widget stays clean.
    """
    out = []
    seen = set()
    for v in values:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            # NaN, pd.NA, None — skip
            continue
        if iv not in seen:
            seen.add(iv)
            out.append(iv)
    return sorted(out)


def _apply_colormap(values: np.ndarray, stops: np.ndarray = None) -> np.ndarray:
    """Map 0-1 normalized values to RGBA colors using colormap stops."""
    if stops is None:
        stops = _COLORMAPS["Viridis"]
    t = np.clip(values, 0, 1)[:, np.newaxis]
    n_stops = len(stops)
    seg = t * (n_stops - 1)
    idx = np.clip(seg.astype(int), 0, n_stops - 2)
    frac = seg - idx
    c0 = stops[idx.ravel()]
    c1 = stops[(idx + 1).ravel()]
    rgb = c0 + frac * (c1 - c0)
    alpha = np.full((len(values), 1), 0.85)
    return np.hstack([rgb, alpha]).astype(np.float32)


class _SelectionLasso(pg.GraphicsObject):
    """Freeform lasso overlay in data coordinates for point selection."""

    def __init__(self):
        super().__init__()
        self._path = QPainterPath()
        self._pen = pg.mkPen(
            color=(100, 180, 255, 200), width=1.5, style=Qt.DashLine,
        )
        self._brush = QBrush(QColor(100, 180, 255, 30))

    def add_point(self, x: float, y: float):
        self.prepareGeometryChange()
        if self._path.elementCount() == 0:
            self._path.moveTo(x, y)
        else:
            self._path.lineTo(x, y)
        self.update()

    def close_path(self):
        self.prepareGeometryChange()
        if self._path.elementCount() > 0:
            self._path.closeSubpath()
        self.update()

    def reset(self):
        self.prepareGeometryChange()
        self._path = QPainterPath()
        self.update()

    def boundingRect(self):
        if self._path.elementCount() > 0:
            return self._path.boundingRect()
        return QRectF()

    def paint(self, painter, *args):
        if self._path.elementCount() > 0:
            painter.setPen(self._pen)
            painter.setBrush(self._brush)
            painter.drawPath(self._path)


# NOTE: `_UMAPWorker` (background-thread UMAP compute) was removed in
# Phase A.5 (2026-05-04).  The widget no longer auto-computes UMAP when
# the user picks the UMAP plot type — embeddings are computed during
# Step 3 Extract (via the QC Settings dialog's "Compute UMAP/t-SNE"
# checkboxes) and ride into per-patch CSVs as ``UMAP_1``/``UMAP_2`` and
# ``tSNE_1``/``tSNE_2`` columns.  ``_hydrate_embedding_caches()`` reads
# those columns into ``_umap_cache``/``_tsne_cache`` whenever a new df
# is set.  Step 4's EDA workbench keeps its own controller-driven
# Compute UMAP / Compute t-SNE buttons that write the caches directly,
# so this widget remains a passive renderer.


class QCPlotWidget(QWidget):
    """Embeddable pyqtgraph-based QC feature visualization."""

    labels_selected = Signal(list)              # label IDs clicked/selected
    tier_thresholds_changed = Signal(tuple)     # (t1, t2, t3) for multi-threshold
    tier_drag_finished = Signal(tuple)           # emitted on mouse release after drag
    viewer_color_requested = Signal(str, str, str)  # (color_by_feature, colormap_name, norm_mode)
    category_selected = Signal(str, str)        # (column_name, category_value)

    # Phase E (2026-05-04) — "Histogram" removed from the right plot.
    # The left-panel triage_histogram_panel.py is the canonical
    # histogram view for qc_score (draggable tier lines + Coolwarm
    # gradient + tier chips) — having a second histogram on the right
    # was duplicate functionality and a recursion hot-spot.  The
    # follow-up dead-code sweep (2026-05-04) removed the unreachable
    # _draw_histogram / _draw_gradient_histogram /
    # _draw_stacked_histogram / _compute_stacked_bins /
    # _handle_histogram_bar_click methods plus the _thresh_line widget,
    # _on_thresh_moved slot, and threshold_changed signal.
    PLOT_TYPES = ["Scatter", "PCA", "UMAP", "t-SNE", "Violin"]

    def __init__(self, parent: Optional[QWidget] = None, *,
                 layout: str = "classic", texts: Optional[dict] = None,
                 palette=None, tier_scheme=None):
        if layout not in PLOT_LAYOUTS:
            raise ValueError(
                f"QCPlotWidget layout must be one of {PLOT_LAYOUTS}, "
                f"got {layout!r}")
        if layout == "workbench":
            missing = [k for k in WB_TEXT_KEYS if k not in (texts or {})]
            missing += [f"help[{k}]" for k in WB_HELP_KEYS
                        if k not in (texts or {}).get("help", {})]
            if missing:
                raise ValueError(
                    f"layout='workbench' needs texts for {missing}")
        super().__init__(parent)
        #: Workbench only: the words it shows, supplied by the caller.
        self._wb_texts = dict(texts or {})
        #: ``"classic"`` (Step 3, byte-identical) or ``"workbench"`` (the
        #: Step 4 Workbench).  Read only by the opt-in branches.
        self._layout_mode = layout
        #: Workbench only: the accent the
        #: workbench styles are drawn in.  ``palette`` is READ only for
        #: the workbench layout; the classic layout ignores it entirely.
        self._palette = (as_plot_palette(palette) if layout == "workbench"
                         else DEFAULT_PLOT_PALETTE)
        #: Workbench only: how ``qc_tier`` groups into legend buckets
        #: (STEP4_SCHEME unless the host passes another).  The classic
        #: layout never reads it.
        if layout == "workbench" and tier_scheme is not None:
            if not isinstance(tier_scheme, TierScheme):
                raise TypeError("tier_scheme must be a TierScheme, got "
                                f"{type(tier_scheme).__name__}")
            self._scheme = tier_scheme
        else:
            self._scheme = STEP4_SCHEME
        #: Workbench only: the sentence an empty plot shows (None: none),
        #: the last status pushed, and the host's kept-selection hook.
        self._wb_empty_sentence: Optional[str] = None
        self._wb_status: Tuple[str, str] = ("", "info")
        self._wb_selection_kept_cb: Optional[Callable[[], None]] = None
        #: Workbench only: the map whose compute is running (empty text
        #: reads MAP_BUSY_EMPTY), and the input each map was made from.
        self._wb_map_busy: Optional[str] = None
        self._wb_map_inputs: Dict[str, str] = {}

        # State
        self._df = None                          # pandas DataFrame
        self._qc_statuses: Dict[int, str] = {}   # label_id -> status
        self._scatter_label_ids: List[int] = []   # label IDs in scatter order
        self._scatter_to_df_index: List[int] = []   # scatter pos → full-df row index
        self._clicked_df_index: Optional[int] = None  # scatter position of last single-clicked point
        self._lasso_points: list = []                # lasso polygon points [(x, y), ...]
        self._selected_indices: set = set()       # indices of selected points
        self._pca_features: Optional[List[str]] = None  # curated PCA feature list
        self._pca_cache: Optional[np.ndarray] = None  # cached PCA projection
        self._pc_projected: Optional[np.ndarray] = None  # full PCA for PC coloring
        # Embedding caches.  Hydrated from persisted columns
        # (UMAP_1/UMAP_2, tSNE_1/tSNE_2) by ``_hydrate_embedding_caches``
        # whenever a new df is set, OR written directly by Step 4's
        # controller via the EDA Compute UMAP / Compute t-SNE buttons.
        self._umap_cache: Optional[np.ndarray] = None
        self._tsne_cache: Optional[np.ndarray] = None
        self._selected_categories: set = set()   # set of (col_name, cat_value) tuples
        self._selected_category_df_indices: list = []   # DataFrame row indices for categories

        # Layout
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # --- Top bar row 1: plot type + select + axes ---
        row1 = QHBoxLayout()
        row1.setContentsMargins(4, 4, 4, 0)
        row1.setSpacing(6)

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(self.PLOT_TYPES)
        self.plot_type_combo.setMaximumWidth(90)
        self.plot_type_combo.setStyleSheet(_COMBO_STYLE)
        self.plot_type_combo.currentIndexChanged.connect(self._on_plot_type_changed)

        self._select_btn = QPushButton("\u27E1")  # ⟡ lasso symbol
        self._select_btn.setCheckable(True)
        self._select_btn.setMaximumWidth(26)
        self._select_btn.setToolTip("Lasso select \u2014 drag to select points")
        self._select_btn.setStyleSheet(_SELECT_BTN_STYLE)

        self._clear_sel_btn = QPushButton("\u2715")  # ✕ clear
        self._clear_sel_btn.setMaximumWidth(26)
        self._clear_sel_btn.setToolTip("Clear selection")
        self._clear_sel_btn.setStyleSheet(_SELECT_BTN_STYLE)
        self._clear_sel_btn.setVisible(False)
        self._clear_sel_btn.clicked.connect(self._clear_selection)

        self._density_btn = QPushButton("\u2248")  # ≈ density symbol
        self._density_btn.setCheckable(True)
        self._density_btn.setMaximumWidth(26)
        self._density_btn.setToolTip("Toggle density contours")
        self._density_btn.setStyleSheet(_SELECT_BTN_STYLE)
        self._density_btn.toggled.connect(self._redraw)

        self._history_cb = QCheckBox("\U0001F47B")  # 👻 ghost emoji
        self._history_cb.setToolTip(
            "Show History — display deleted/superseded labels\n"
            "as ghost points in scatter/PCA/UMAP plots"
        )
        self._history_cb.setStyleSheet("color: #999; font-size: 10px;")
        self._history_cb.setChecked(True)
        self._history_cb.toggled.connect(self._redraw)

        # Phase A.5: "From PCA" checkbox removed.  UMAP's pca_dims is
        # now controlled by the QC Settings dialog's "PCA dimensions"
        # field for Step 3, and by the Embedding tab for Step 4.

        self._x_label = QLabel("Feature")
        self._x_label.setStyleSheet(_LABEL_STYLE)
        self._feature_x_combo = QComboBox()
        self._feature_x_combo.setMinimumWidth(100)
        self._feature_x_combo.setMaximumWidth(140)
        self._feature_x_combo.setStyleSheet(_COMBO_STYLE)
        self._feature_x_combo.currentIndexChanged.connect(self._redraw)

        self._y_label = QLabel("vs")
        self._y_label.setStyleSheet(_LABEL_STYLE)
        self._feature_y_combo = QComboBox()
        self._feature_y_combo.setMinimumWidth(100)
        self._feature_y_combo.setMaximumWidth(140)
        self._feature_y_combo.setStyleSheet(_COMBO_STYLE)
        self._feature_y_combo.setVisible(False)
        self._feature_y_combo.currentIndexChanged.connect(self._redraw)

        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #999; font-size: 10px;")
        self._stats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row1.addWidget(self.plot_type_combo)
        row1.addWidget(self._select_btn)
        row1.addWidget(self._clear_sel_btn)
        row1.addWidget(self._density_btn)
        row1.addWidget(self._history_cb)
        row1.addWidget(self._x_label)
        row1.addWidget(self._feature_x_combo)
        row1.addWidget(self._y_label)
        row1.addWidget(self._feature_y_combo)
        row1.addStretch(1)
        row1.addWidget(self._stats_label)
        root.addLayout(row1)

        # --- Top bar row 2: color-by + colormap selectors ---
        row2 = QHBoxLayout()
        row2.setContentsMargins(4, 0, 4, 0)
        row2.setSpacing(6)

        self._color_label = QLabel("Color by")
        self._color_label.setStyleSheet(_LABEL_STYLE)
        self._color_combo = QComboBox()
        self._color_combo.setMinimumWidth(100)
        self._color_combo.setMaximumWidth(180)
        self._color_combo.setEditable(True)
        self._color_combo.setInsertPolicy(QComboBox.NoInsert)
        self._color_combo.setStyleSheet(_COMBO_STYLE)
        self._color_combo.activated.connect(self._redraw)
        # Substring search: type anywhere in the name to filter
        _completer = self._color_combo.completer()
        if _completer is not None:
            _completer.setFilterMode(Qt.MatchContains)
            from qtpy.QtWidgets import QCompleter
            _completer.setCompletionMode(QCompleter.PopupCompletion)

        self._cmap_label = QLabel("Colormap")
        self._cmap_label.setStyleSheet(_LABEL_STYLE)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(list(_COLORMAPS.keys()))
        self._cmap_combo.setMaximumWidth(90)
        self._cmap_combo.setStyleSheet(_COMBO_STYLE)
        self._cmap_combo.currentIndexChanged.connect(self._redraw)

        self._norm_label = QLabel("Norm")
        self._norm_label.setStyleSheet(_LABEL_STYLE)
        self._norm_combo = QComboBox()
        self._norm_combo.addItems(["Linear", "Log", "Rank"])
        self._norm_combo.setCurrentText("Rank")
        self._norm_combo.setMaximumWidth(70)
        self._norm_combo.setStyleSheet(_COMBO_STYLE)
        self._norm_combo.setToolTip(
            "Linear: raw min-max (good for bounded 0-1 features)\n"
            "Log: log1p transform (good for skewed features like area)\n"
            "Rank: percentile rank (distribution-agnostic, uniform spread)"
        )
        self._norm_combo.currentIndexChanged.connect(self._redraw)

        self._colorbar_label = QLabel("")
        self._colorbar_label.setStyleSheet("color: #999; font-size: 9px;")

        row2.addWidget(self._color_label)
        row2.addWidget(self._color_combo)
        row2.addWidget(self._norm_label)
        row2.addWidget(self._norm_combo)
        row2.addWidget(self._cmap_label)
        row2.addWidget(self._cmap_combo)
        row2.addWidget(self._colorbar_label)
        row2.addStretch(1)
        root.addLayout(row2)

        # --- Plot area ---
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(_BG_COLOR)
        self._plot_widget.showGrid(x=True, y=True, alpha=_GRID_ALPHA / 255)

        plot_item = self._plot_widget.plotItem
        plot_item.setContentsMargins(4, 4, 4, 4)
        plot_item.getViewBox().setDefaultPadding(0.04)

        for axis_name in ("bottom", "left"):
            ax = self._plot_widget.getAxis(axis_name)
            ax.setTextPen(pg.mkPen(color=(120, 120, 130), width=1))
            ax.setPen(pg.mkPen(color=(50, 50, 60), width=1))
            ax.setStyle(
                tickLength=-5,
                tickTextOffset=4,
                autoExpandTextSpace=True,
                autoReduceTextSpace=True,
            )
            if axis_name == "bottom":
                ax.setHeight(24)
            else:
                ax.setWidth(42)

        self._plot_widget.setLabel("bottom", "")
        self._plot_widget.setLabel("left", "")
        plot_item.setTitle(None)

        # Multi-threshold tier lines (3 draggable lines for qc_score histogram)
        _tier_line_colors = [
            (80, 200, 120, 200),   # good/review boundary — green
            (255, 160, 40, 200),   # review/bad boundary — orange
            (240, 60, 60, 200),    # bad/garbage boundary — red
        ]
        self._tier_lines: list = []
        for color in _tier_line_colors:
            line = pg.InfiniteLine(
                angle=90, movable=True,
                pen=pg.mkPen(color=color, width=2, style=Qt.DashLine),
            )
            line.setVisible(False)
            line.sigPositionChanged.connect(self._on_tier_line_moved)
            line.sigPositionChangeFinished.connect(self._on_tier_line_released)
            self._plot_widget.addItem(line, ignoreBounds=True)
            self._tier_lines.append(line)

        # Selection rectangle overlay (data-coordinate rect for drag-select)
        self._selection_lasso = _SelectionLasso()
        self._plot_widget.addItem(self._selection_lasso, ignoreBounds=True)

        root.addWidget(self._plot_widget, 1)

        # --- Bottom legend ---
        self._legend_label = QLabel("")
        self._legend_label.setStyleSheet(
            "color: #999; font-size: 9px; padding: 2px 4px;"
        )
        self._legend_label.setWordWrap(True)
        self._legend_label.setMaximumHeight(36)
        self._legend_label.setTextFormat(Qt.RichText)
        self._legend_label.setOpenExternalLinks(False)
        self._legend_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self._legend_label.linkActivated.connect(self._on_legend_clicked)
        root.addWidget(self._legend_label)

        # Click handling for scatter (nearest-point click)
        self._plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # Event filter on viewport for lasso drag-select
        self._plot_widget.viewport().installEventFilter(self)

        if self._layout_mode == "workbench":
            self._build_workbench_layout(root, row1, row2)

    # --- The Workbench layout (opt-in) --------------------------

    def _build_workbench_layout(self, root, row1, row2):
        """Re-home the classic top bars into the Workbench's toolbar.

        Called only for ``layout="workbench"``.  Every widget the handlers
        read (``plot_type_combo``, ``_select_btn``, ``_feature_x_combo``…)
        stays the SAME object; only its text, style and place change, and
        the hidden ``plot_type_combo`` is still the truth every handler
        reads (the segmented buttons drive it).  Every word comes in
        through ``texts`` (this Step 3 module never imports Step 4's text).
        """
        H = self._wb_texts["help"]
        big = 16777215  # QWIDGETSIZE_MAX

        for row in (row1, row2):
            while row.count():
                row.takeAt(0)
            root.removeItem(row)
            row.setParent(None)

        # The count moves to the Workbench's "what you see" line.
        old = self._stats_label
        old.hide()
        old.setParent(None)
        old.deleteLater()
        self._stats_label = _NotifyingLabel("", self)
        self._stats_label.hide()
        #: Workbench only: emitted whenever the plot recounts or reselects.
        self.stats_changed = self._stats_label.text_set
        #: Workbench only: ``status_changed(text, kind)`` from
        #: :meth:`set_status` (an INSTANCE signal, like ``stats_changed``).
        self._wb_status_notifier = _StatusNotifier(self)
        self.status_changed = self._wb_status_notifier.changed
        #: Workbench only: the ghosts toggle's ``toggled(bool)`` and the
        #: Colour by combo's ``currentTextChanged(str)`` (instance aliases,
        #: the ``stats_changed`` pattern) for the host's see-line.
        self.ghosts_toggled = self._history_cb.toggled
        self.colour_by_changed = self._color_combo.currentTextChanged

        # --- toolbar: the five plot types as a segmented row + three toggles
        P = self._palette
        self.plot_type_combo.hide()
        self.plot_type_combo.setStyleSheet(_wb_combo_style(P))
        seg = QWidget(self)
        seg_lay = QHBoxLayout(seg)
        seg_lay.setContentsMargins(0, 0, 0, 0)
        seg_lay.setSpacing(0)
        self._wb_type_group = QButtonGroup(self)
        self._wb_type_group.setExclusive(True)
        self._wb_type_buttons: Dict[str, QPushButton] = {}
        for i, name in enumerate(self.PLOT_TYPES):
            b = QPushButton(name, seg)
            b.setCheckable(True)
            b.setStyleSheet(_wb_seg_style(P))
            b.setToolTip(H["Plot type"])
            b.setFocusPolicy(Qt.TabFocus)
            # Sized for the BOLD face the pressed button uses, so pressing
            # one never widens it (and never shifts its neighbours).
            bold = QFont(b.font())
            bold.setPixelSize(11)
            bold.setBold(True)
            b.setMinimumWidth(
                QFontMetrics(bold).horizontalAdvance(name) + 2 * 9 + 6)
            self._wb_type_group.addButton(b, i)
            seg_lay.addWidget(b)
            self._wb_type_buttons[name] = b
        self._wb_type_group.buttonClicked.connect(self._wb_on_type_button)
        self.plot_type_combo.currentIndexChanged.connect(
            self._wb_sync_type_buttons)
        self._wb_seg = seg

        ghosts_word = self._wb_word("GHOSTS_WORD", "Removed")
        for btn, text, key in ((self._select_btn, "Lasso", "Lasso"),
                               (self._density_btn, "Density", "Density"),
                               (self._history_cb, ghosts_word, "Removed")):
            btn.setText(text)
            btn.setMaximumWidth(big)
            btn.setStyleSheet(_wb_toggle_style(P))
            btn.setToolTip(H[key])
        # Step 4's mockup opens with "Removed" off (the classic plot keeps
        # its ghosts on); Step 3's scheme opens with them on.
        # Blocked: the redraw its toggle asks for would run before the
        # axes row exists.
        was = self._history_cb.blockSignals(True)
        self._history_cb.setChecked(self._scheme.ghosts_default_on)
        self._history_cb.blockSignals(was)

        # "✕ Clear" is always on screen in the Workbench, greyed with no
        # selection: every classic ``_clear_sel_btn.setVisible(x)`` becomes
        # setEnabled(x) through a small proxy.
        clear = self._clear_sel_btn
        clear.setText("✕ Clear")
        clear.setMaximumWidth(big)
        clear.setStyleSheet(_WB_CLEAR_STYLE)
        clear.setToolTip(H["Clear"])
        clear.setVisible(True)
        clear.setEnabled(False)
        self._wb_clear_btn = clear
        self._clear_sel_btn = _ShownAsEnabled(clear)

        # The toolbar wraps too (mockup ``.tool`` is a wrapping row), so a
        # wide font never pushes the dock sideways.
        tool = FlowLayout(h_spacing=8, v_spacing=6)
        tool.addWidget(seg)
        tool.addWidget(self._select_btn)
        tool.addWidget(self._density_btn)
        tool.addWidget(self._history_cb)
        # "✕ Clear" waits here until the Workbench takes it into its
        # selection card (take_clear_button), so it never floats unplaced.
        tool.addWidget(clear)
        self._wb_tool_row = tool

        # --- the one axes row that wraps: X · Y · Colour by · Scale · Colours
        def combo_fit(combo, chars):
            combo.setStyleSheet(_wb_combo_style(P))
            combo.setMaximumWidth(big)
            combo.setMinimumWidth(0)
            combo.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(chars)

        def field(label, combo, tip):
            box = QWidget(self)
            lay = QHBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            label.setStyleSheet(_WB_LABEL_STYLE)
            label.setToolTip(tip)
            combo.setToolTip(tip)
            lay.addWidget(label)
            lay.addWidget(combo)
            return box

        self._x_label.setText("X")
        self._y_label.setText("Y")
        self._color_label.setText("Colour by")
        self._norm_label.setText("Scale")
        self._cmap_label.setText("Colours")
        combo_fit(self._feature_x_combo, 15)
        combo_fit(self._feature_y_combo, 15)
        combo_fit(self._color_combo, 15)
        combo_fit(self._norm_combo, 6)
        combo_fit(self._cmap_combo, 8)
        self._y_label.setVisible(True)
        self._feature_y_combo.setVisible(True)
        self._wb_x_field = field(self._x_label, self._feature_x_combo, "")
        self._wb_y_field = field(self._y_label, self._feature_y_combo, "")
        self._wb_colour_field = field(self._color_label, self._color_combo,
                                      H["Colour by"])
        self._wb_scale_field = field(self._norm_label, self._norm_combo,
                                     H["Scale"])
        self._wb_cmap_field = field(self._cmap_label, self._cmap_combo,
                                    H["Colours"])
        self._wb_axes_note = QLabel("", self)
        self._wb_axes_note.setStyleSheet(_WB_LABEL_STYLE)
        # The range moves to the gradient legend under the plot (mockup);
        # the classic range label stays alive, written, and never shown.
        self._colorbar_label.hide()

        flow = FlowLayout(h_spacing=10, v_spacing=6)
        for w in (self._wb_x_field, self._wb_y_field, self._wb_axes_note,
                  self._wb_colour_field, self._wb_scale_field,
                  self._wb_cmap_field):
            flow.addWidget(w)
        #: Workbench only: the wrapping axes row (the Workbench adds its PC
        #: pickers here through ``add_axes_widget``).
        self.axes_flow = flow

        bars = QVBoxLayout()
        bars.setContentsMargins(0, 0, 0, 0)
        bars.setSpacing(6)
        bars.addLayout(tool)
        bars.addLayout(flow)
        root.insertLayout(0, bars)
        root.setSpacing(6)

        # The feature-colour legend as the mockup draws it: name · lo ·
        # a 140 px gradient · hi · the scale word, under the plot.
        self._wb_gradient = _GradientLegend(self)
        self._wb_gradient.hide()
        root.addWidget(self._wb_gradient)

        self._plot_widget.setBackground(_WB_BG)
        self._plot_widget.setMinimumHeight(WB_PLOT_MIN_H)
        self._plot_widget.setMaximumHeight(WB_PLOT_MAX_H)

        # currentTextChanged, not currentIndexChanged: the colour combo is
        # editable, and an editable combo's setCurrentText only sets the text.
        self._color_combo.currentTextChanged.connect(self._wb_sync_axes)
        self._feature_x_combo.currentIndexChanged.connect(self._wb_sync_axes)
        self._wb_sync_type_buttons()
        self._wb_sync_axes()

    def _wb_on_type_button(self, button):
        """A segmented button drives the (hidden) plot-type combo."""
        idx = self._wb_type_group.id(button)
        if idx >= 0 and idx != self.plot_type_combo.currentIndex():
            self.plot_type_combo.setCurrentIndex(idx)

    def _wb_sync_type_buttons(self, *_):
        """The pressed segmented button follows the combo (the controller
        sets the plot type through ``plot_type_combo`` too)."""
        name = self.plot_type_combo.currentText()
        btn = self._wb_type_buttons.get(name)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)

    def _wb_is_feature_colour(self) -> bool:
        colour = self._color_combo.currentText()
        return bool(colour) and colour not in self._CATEGORICAL_COLOR_BY

    def _wb_sync_axes(self, *_):
        """Show the axes-row fields the current plot type and colour use."""
        if self._layout_mode != "workbench":
            return
        t = self._wb_texts
        pt = self.plot_type_combo.currentText()
        self._wb_x_field.setVisible(pt in ("Scatter", "Violin"))
        self._x_label.setText("X" if pt == "Scatter" else "Feature")
        self._wb_y_field.setVisible(pt == "Scatter")
        colour = self._color_combo.currentText()
        is_feature = self._wb_is_feature_colour()
        self._wb_scale_field.setVisible(is_feature)
        self._wb_cmap_field.setVisible(is_feature)
        note = ""
        if pt == "Violin":
            by = colour if colour in self._CATEGORICAL_COLOR_BY else "QC status"
            if self._wb_violin_on():
                by = self._wb_violin_by()
            note = t["VIOLIN_GROUPS"].format(
                feat=self._feature_x_combo.currentText(), by=by)
        elif pt in ("UMAP", "t-SNE"):
            cache = self._umap_cache if pt == "UMAP" else self._tsne_cache
            made_from = self._wb_map_inputs.get(pt, "")
            if cache is not None and made_from:
                note = t["MAP_AXES"].format(map=pt, input=made_from)
        self._wb_axes_note.setText(note)
        self._wb_axes_note.setVisible(bool(note))
        self.axes_flow.invalidate()

    def _wb_update_legend(self):
        """Categorical colour: the clickable legend line.  A feature: the
        gradient legend (name · lo · gradient · hi · scale)."""
        if not self._wb_is_feature_colour() or self._df is None:
            self._wb_gradient.hide()
            self._legend_label.show()
            return
        name = self._color_combo.currentText()
        values = None
        if name.startswith("PC") and name[2:].isdigit():
            i = int(name[2:]) - 1
            if (self._pc_projected is not None
                    and i < self._pc_projected.shape[1]):
                values = self._pc_projected[:, i]
        elif name in self._df.columns:
            values = self._df[name].to_numpy()
        if values is None:
            self._wb_gradient.hide()
            self._legend_label.show()
            return
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        lo, hi = (float(v.min()), float(v.max())) if len(v) else (0.0, 0.0)
        self._wb_gradient.set_legend(
            name, _fmt_num(lo), _fmt_num(hi), self._active_cmap_stops(),
            self._norm_combo.currentText())
        if (self._wb_violin_on()
                and self.plot_type_combo.currentText() == "Violin"):
            # Step 3's violin: the groups' legend for the violins, and the
            # gradient under it -- the key of the viewer's colours.
            self._legend_label.show()
        else:
            self._legend_label.hide()
        self._wb_gradient.show()

    def _wb_draw_empty_map(self, kind: str):
        """The Workbench's empty UMAP / t-SNE text: it names the
        Map tab instead of Step 3's QC Settings dialog.  No axes are drawn
        (the previous plot's titles and ticks must not linger)."""
        t = self._wb_texts
        if self._wb_map_busy == kind:
            sentence = t["MAP_BUSY_EMPTY"].format(map=kind)
        else:
            sentence = t["UMAP_EMPTY"] if kind == "UMAP" else t["TSNE_EMPTY"]
        self._wb_draw_sentence(sentence)

    def _wb_draw_sentence(self, sentence: str):
        """One centred ``"title|body"`` sentence, no axes (the empty map's
        drawing, shared with :meth:`set_empty_sentence`)."""
        title, _, body = sentence.partition("|")
        item = self._plot_widget.getPlotItem()
        # (no setLabel here: PlotItem.setLabel shows its axis again)
        item.hideAxis("bottom")
        item.hideAxis("left")
        text_item = pg.TextItem(color=(170, 170, 170), anchor=(0.5, 0.5))
        text_item.setText(title + "\n" + body)
        text_item.setTextWidth(_WB_EMPTY_TEXT_WIDTH)
        if self._palette == DEFAULT_PLOT_PALETTE:
            # Step 4's empty map, drawn exactly as it always was.
            text_item.setFont(pg.QtGui.QFont("sans-serif", 10))
        else:
            # A host palette (Step 3): each line centred inside the
            # wrapping width, in the application's own font.  Gated so
            # Step 4's pixels do not move.
            from qtpy.QtWidgets import QApplication

            doc = text_item.textItem.document()
            opt = doc.defaultTextOption()
            opt.setAlignment(Qt.AlignHCenter)
            doc.setDefaultTextOption(opt)
            text_item.setFont(QApplication.font())
            if hasattr(text_item, "updateTextPos"):
                text_item.updateTextPos()
        self._plot_widget.addItem(text_item)
        text_item.setPos(0, 0)
        self._plot_widget.setXRange(-1, 1)
        self._plot_widget.setYRange(-1, 1)

    # --- Workbench hooks ----------------------------------------------------

    def _require_workbench(self, what: str):
        if self._layout_mode != "workbench":
            raise RuntimeError(f"{what} needs layout='workbench'")

    def take_clear_button(self) -> QPushButton:
        """Workbench: hand "✕ Clear" to the selection card.

        The plot keeps greying it and un-greying it exactly as it used to
        hide and show it; only its place changes.
        """
        self._require_workbench("take_clear_button")
        self._wb_tool_row.removeWidget(self._wb_clear_btn)
        return self._wb_clear_btn

    def add_axes_widget(self, widget: QWidget, *, before_colour: bool = True):
        """Workbench: put ``widget`` into the wrapping axes row (the PC
        pickers go before "Colour by")."""
        self._require_workbench("add_axes_widget")
        index = self.axes_flow.count()
        if before_colour:
            for i in range(self.axes_flow.count()):
                if self.axes_flow.itemAt(i).widget() is self._wb_colour_field:
                    index = i
                    break
        self.axes_flow.insertWidget(index, widget)

    def set_status(self, text: str, kind: str = "info"):
        """Say what just happened.

        Workbench: emits ``status_changed(text, kind)`` for the host's
        status slot (``kind`` is ``"info"``, ``"warn"`` or ``"error"``).
        Classic: writes the plot's own count label, as the controller's
        status writes do today.
        """
        if kind not in ("info", "warn", "error"):
            raise ValueError(
                f"status kind must be info / warn / error, got {kind!r}")
        text = "" if text is None else str(text)
        if self._layout_mode != "workbench":
            self._stats_label.setText(text)
            return
        self._wb_status = (text, kind)
        self.status_changed.emit(text, kind)

    def status(self) -> Tuple[str, str]:
        """Workbench: the last ``(text, kind)`` :meth:`set_status` pushed."""
        self._require_workbench("status")
        return self._wb_status

    def set_empty_sentence(self, text: Optional[str]):
        """Workbench: the ``"title|body"`` sentence an EMPTY plot (no table,
        or a table with no rows) shows in its middle, axes hidden; ``None``
        shows nothing.  Redraws only while the plot is empty."""
        self._require_workbench("set_empty_sentence")
        self._wb_empty_sentence = None if not text else str(text)
        if self._df is None or self._df.empty:
            self._redraw()

    def set_selection_kept_callback(self, cb: Optional[Callable[[], None]]):
        """Workbench: ``cb()`` runs after a plot-type switch KEPT the
        selection (the rows ride along as a frozen pick), so the host can
        show it.  ``None`` removes it."""
        self._require_workbench("set_selection_kept_callback")
        if cb is not None and not callable(cb):
            raise TypeError("the kept-selection callback must be callable")
        self._wb_selection_kept_cb = cb

    def _on_selection_kept(self):
        """Workbench: a plot-type switch kept the selection.  A no-op
        unless the host set a callback; a failing callback is reported,
        never raised into the plot switch."""
        cb = self._wb_selection_kept_cb
        if cb is None:
            return
        try:
            cb()
        except Exception as exc:  # noqa: BLE001 -- the switch must finish
            print(f"[QC-PLOT] kept-selection callback failed: {exc!r}")

    def set_map_busy(self, kind: Optional[str]):
        """Workbench (B3b): the map now computing, or ``None``.  Its empty
        plot then reads MAP_BUSY_EMPTY instead of UMAP_EMPTY / TSNE_EMPTY."""
        self._require_workbench("set_map_busy")
        self._wb_map_busy = kind
        if self.plot_type_combo.currentText() == kind or kind is None:
            self._redraw()

    def _wb_unassigned_rgba(self) -> np.ndarray:
        r, g, b = self._scheme.rgb[self._scheme.unassigned]
        return np.array([r / 255, g / 255, b / 255, 0.85], dtype=np.float32)

    def _wb_buckets(self, df) -> np.ndarray:
        """The bucket of every row of ``df`` under this plot's scheme."""
        return self._scheme.buckets_of(df)

    def _wb_active_mask(self, df=None) -> np.ndarray:
        """Workbench: the rows still active (``label_status``), so a
        removed cell is in no count and no pick (re-audit MED-2)."""
        df = self._df if df is None else df
        if df is None:
            return np.zeros(0, bool)
        if "label_status" not in df.columns:
            return np.ones(len(df), bool)
        return (df["label_status"].fillna("active").astype(str)
                == "active").to_numpy()

    def _wb_word(self, key: str, default: str) -> str:
        return (self._wb_texts or {}).get(key, default)

    def _wb_unassigned_mask(self) -> np.ndarray:
        """Rows in the catch-all bucket (Step 4: unmarked)."""
        return self._wb_buckets(self._df) == self._scheme.unassigned

    def _wb_tier_legend_colours(self) -> dict:
        """The Workbench's tier legend, in the scheme's order (Step 4:
        good, bad, unmarked)."""
        return self._scheme.colours()

    def viewer_colours(self):
        """Workbench only: ``(label ids, RGBA float32)`` -- the colours this
        plot draws for its active rows (deleted / superseded rows are its
        ghosts) under the current Colour by, Scale and Colours.  Step 4's
        viewer takes them as they are, so the plot and the viewer can never
        disagree (owner question 2026-10-05: tiers, a missing column, NaN,
        a constant feature, noise alpha).  Empty with no table."""
        if self._layout_mode != "workbench":
            raise RuntimeError("viewer_colours needs layout='workbench'")
        if self._df is None or self._df.empty:
            return (np.zeros(0, dtype=np.int64),
                    np.zeros((0, 4), dtype=np.float32))
        ghost = self._get_ghost_mask()
        active = self._df if ghost is None else self._df[~ghost]
        rgba = np.asarray(self._get_point_colors_for(active),
                          dtype=np.float32)
        return active["label"].to_numpy(dtype=np.int64), rgba

    #: Colour by words -> the table column they read (the viewer's hover).
    _WB_COLOUR_BY_COLUMN = {
        "": "qc_status",
        "QC status": "qc_status",
        "QC tier": "qc_tier",
        "Cluster": "cluster_id",
        "Patch": "patch_name",
        "Border cell": "is_border_cell",
        "Topology": "euler_number",
    }

    def colour_by_column_values(self):
        """Workbench only: ``(column, label ids, values)`` -- the table
        column the current Colour by reads and its value on every row
        :meth:`viewer_colours` returns, in the same order (Step 3's viewer
        hover).  ``column`` is ``None`` and
        ``values`` empty when Colour by names no column of the table (a PC,
        a fallback, "QC status" with no ``qc_status``).  Step 4 never calls
        it."""
        if self._layout_mode != "workbench":
            raise RuntimeError(
                "colour_by_column_values needs layout='workbench'")
        if self._df is None or self._df.empty:
            return None, np.zeros(0, dtype=np.int64), np.zeros(0)
        ghost = self._get_ghost_mask()
        active = self._df if ghost is None else self._df[~ghost]
        ids = active["label"].to_numpy(dtype=np.int64)
        name = (self._color_combo.currentText()
                if self._color_combo.count() > 0 else "")
        col = self._WB_COLOUR_BY_COLUMN.get(name, name)
        if col not in active.columns:
            return None, ids, np.zeros(0)
        return col, ids, active[col].to_numpy()

    def set_colour_by(self, name: str, *, colours: Optional[str] = None,
                      scale: Optional[str] = None) -> bool:
        """Workbench only: pick ``name`` in Colour by the way a user does --
        the combo, then one redraw (which also asks for the viewer's
        colours, under the "Colours in viewer" tick).  False when the combo
        does not offer ``name``.

        ``colours`` (a Colours item, e.g. "Coolwarm") and ``scale`` (a Scale
        item: "Linear" / "Log" / "Rank") are set first, signals blocked, so
        the whole change is still ONE redraw; an item the combo does not
        offer refuses the call (False) and nothing changes.  Called without
        them (Step 4's calls) the path is the one it always was.
        """
        if self._layout_mode != "workbench":
            raise RuntimeError("set_colour_by needs layout='workbench'")
        idx = self._color_combo.findText(name)
        if idx < 0:
            return False
        extra = []
        for combo, item in ((self._cmap_combo, colours),
                            (self._norm_combo, scale)):
            if item is None:
                continue
            j = combo.findText(item)
            if j < 0:
                return False
            extra.append((combo, j))
        for combo, j in extra:
            was = combo.blockSignals(True)
            combo.setCurrentIndex(j)
            combo.blockSignals(was)
        self._color_combo.setCurrentIndex(idx)
        self._redraw()
        return True

    def ghosts_shown(self) -> bool:
        """Workbench: whether the ghosts toggle is ticked."""
        self._require_workbench("ghosts_shown")
        return bool(self._history_cb.isChecked())

    def see_counts(self) -> dict:
        """Workbench: what the host's see-line says, read live.

        ``labels`` -- rows that are not ghosts (``labels + ghosts`` is the
        table's length); ``ghosts`` -- the ghost-mask sum (deleted /
        superseded rows); ``selected`` -- ``len(selected_label_ids())``;
        ``colour_by`` -- the Colour by text, "QC status" when empty.
        Read here because ``_update_stats`` keeps a selection count instead
        of recounting while a selection is held.
        """
        self._require_workbench("see_counts")
        df = self._df
        if df is None:
            n_rows = n_ghosts = 0
        else:
            mask = self._get_ghost_mask()
            n_rows = len(df)
            n_ghosts = int(mask.sum()) if mask is not None else 0
        return {
            "labels": n_rows - n_ghosts,
            "ghosts": n_ghosts,
            "selected": len(self.selected_label_ids()),
            "colour_by": self._color_combo.currentText() or "QC status",
        }

    def selection_kind(self) -> str:
        """Workbench: the kind of selection held now -- ``"lasso"``,
        ``"legend"``, ``"click"``, ``"kept"`` (carried across a plot-type
        switch) or ``""`` (none).  Same priority as
        :meth:`get_selected_df_indices`.  A ``highlight_labels`` echo reads
        as ``"lasso"`` (the plot cannot tell them apart; the host knows
        which door it pushed through)."""
        self._require_workbench("selection_kind")
        if self._selected_indices:
            return "lasso"
        if self._selected_category_df_indices:
            if _WB_KEPT in self._selected_categories:
                return "kept"
            return "legend"
        if self._clicked_df_index is not None:
            return "click"
        return ""

    def _wb_bucket_colors(self, df) -> np.ndarray:
        colors = np.zeros((len(df), 4), dtype=np.float32)
        buckets = self._wb_buckets(df)
        for name, (r, g, b) in self._scheme.colours().items():
            m = buckets == name
            if m.any():
                colors[m] = [r / 255, g / 255, b / 255, 0.85]
        return colors

    def _wb_frozen_mask_for(self, df, n: int):
        """Workbench: a category pick is FROZEN to the rows it selected
        (audit X1-1), so a mark that moves those rows to another tier
        keeps them highlighted.  ``df`` may be the active subset."""
        idx = self._selected_category_df_indices
        if not idx or self._df is None:
            return None
        full = np.zeros(len(self._df), dtype=bool)
        full[[i for i in idx if 0 <= i < len(self._df)]] = True
        if df is self._df or len(df) == len(self._df):
            out = full[:n] if n <= len(full) else None
        else:
            pos = self._df.index.get_indexer(df.index)
            if (pos < 0).any():
                return None
            out = full[pos]
        return out if out is not None and out.any() else None

    def _wb_live_positions(self, key) -> list:
        """The active rows a legend entry covers NOW."""
        col, val = key
        if self._df is None:
            return []
        if col == "qc_tier":
            m = self._wb_buckets(self._df) == val
        elif col in self._df.columns:
            try:
                typed = int(val)
            except (ValueError, TypeError):
                try:
                    typed = float(val)
                except (ValueError, TypeError):
                    typed = val
            m = (self._df[col] == typed).to_numpy()
        else:
            return []
        return [int(i) for i in np.where(m & self._wb_active_mask())[0]]

    def _wb_keep_selection_as_rows(self):
        """Workbench plot-type switch (audit X1-2): the selection becomes
        frozen rows instead of being cleared."""
        if self._df is None:
            return False
        if (not self._selected_indices and self._clicked_df_index is None
                and self._selected_category_df_indices):
            # Already positions (a pick or a kept selection): pass them
            # through as they are (re-audit LOW-4).
            pos = [int(p) for p in self._selected_category_df_indices]
        else:
            rows = list(self.get_selected_df_indices())
            if not rows:
                return False
            pos = self._df.index.get_indexer(rows)
            pos = [int(p) for p in pos if p >= 0]
        if not pos:
            return False
        self._selected_indices.clear()
        self._lasso_points.clear()
        self._selection_lasso.reset()
        self._clicked_df_index = None
        if not self._selected_categories:
            self._selected_categories = {_WB_KEPT}
        self._selected_category_df_indices = pos
        return True

    def refresh_marks(self, df):
        """Workbench (B3b): the tier column changed, nothing else did.

        Redraws with the selection, the maps and the view kept (a mark
        never clears a map,; the card keeps what it
        shows), then asks the viewer for the new colours.
        """
        self._require_workbench("refresh_marks")
        if self._df is not None and len(df) != len(self._df):
            self.set_features(df, preserve_caches=True)
            return
        self._df = df
        self._redraw(_lightweight=True)
        color_by = (self._color_combo.currentText()
                    if self._color_combo.count() > 0 else "")
        cmap_name = (self._cmap_combo.currentText()
                     if self._cmap_combo.count() > 0 else "Viridis")
        norm_mode = (self._norm_combo.currentText()
                     if self._norm_combo.count() > 0 else "Linear")
        self.viewer_color_requested.emit(color_by, cmap_name, norm_mode)

    def set_map_input(self, kind: str, made_from: str):
        """Workbench: what a map was made from, for MAP_AXES."""
        self._require_workbench("set_map_input")
        self._wb_map_inputs[kind] = made_from or ""
        self._wb_sync_axes()

    # --- Public API ---------------------------------------------------------

    def set_features(self, df, *, preserve_caches: bool = False):
        """Load a feature DataFrame and populate selectors.

        Clears any active lasso/category selection and re-syncs combos.
        Use ``update_data()`` instead when only row values changed
        (e.g., label deletion) but the column structure is the same.

        Args:
            df: Feature DataFrame.
            preserve_caches: If True, keep existing PCA/UMAP/t-SNE caches
                (used when restoring pre-injected caches from disk).
        """
        self._df = df
        self._clicked_df_index = None
        if self._selected_indices or self._selected_categories:
            self._selected_indices.clear()
            self._selected_categories.clear()
            self._selected_category_df_indices = []
            self._clear_sel_btn.setVisible(False)
            self.labels_selected.emit([])  # clear controller isolation
        if not preserve_caches:
            self._pca_cache = None
            self._umap_cache = None
            self._tsne_cache = None
        # Phase A.5: hydrate UMAP/t-SNE caches from persisted columns
        # if Step 3 wrote them during Extract.  Runs after invalidation
        # so a previously stale in-memory cache is replaced — never
        # silently kept alongside fresh column data.
        self._hydrate_embedding_caches()
        self._sync_feature_combos()
        self._redraw()

    def update_data(self, df):
        """Update the DataFrame without clearing selection or combos.

        Use this after row-level changes (label deletion, tier assignment,
        status update) where the column set hasn't changed.  Preserves the
        user's current lasso/category selection, Color-by, X/Y axis, and
        plot type choices.  Invalidates PCA/UMAP caches since row data
        changed.

        Lasso selection is preserved by converting scatter-position indices
        to label IDs before the redraw, then mapping back afterwards.
        """
        # Convert position-based selection to label IDs before data changes
        preserved_label_ids = None
        if self._selected_indices and self._scatter_label_ids:
            preserved_label_ids = set()
            for i in self._selected_indices:
                if 0 <= i < len(self._scatter_label_ids):
                    preserved_label_ids.add(int(self._scatter_label_ids[i]))

        self._df = df
        self._pca_cache = None
        self._umap_cache = None
        self._tsne_cache = None
        # Phase A.5: re-hydrate from persisted columns after invalidation.
        self._hydrate_embedding_caches()

        # Clear position indices — they'll be rebuilt after redraw
        self._selected_indices.clear()
        self._redraw()

        # Re-map preserved label IDs back to new scatter positions
        if preserved_label_ids and self._scatter_label_ids:
            new_indices = set()
            for i, lid in enumerate(self._scatter_label_ids):
                if int(lid) in preserved_label_ids:
                    new_indices.add(i)
            if new_indices:
                self._selected_indices = new_indices
                self._clear_sel_btn.setVisible(True)
                n_sel = len(new_indices)
                self._stats_label.setText(
                    f"{n_sel} selected of {len(self._scatter_label_ids)}"
                )
                self._redraw()  # re-render with highlights restored

    def set_qc_statuses(self, statuses: Dict[int, str]):
        """Set per-label QC statuses for coloring."""
        self._qc_statuses = dict(statuses)
        self._redraw()

    def set_pca_features(self, features: Optional[List[str]]):
        """Set the feature list used for PCA projection."""
        self._pca_features = features
        self._pca_cache = None  # invalidate PCA cache
        # Phase A.5: PCA-feature changes invalidate the in-memory PCA
        # cache but do NOT invalidate persisted UMAP/t-SNE columns —
        # those were computed on the dialog's chosen feature set, not
        # on the workbench's pca_features.  Re-hydrate after the
        # PCA cache reset so the embedding survives feature swaps.
        self._umap_cache = None
        self._tsne_cache = None
        self._hydrate_embedding_caches()

    def set_pc_color_options(self, n_components: int):
        """Add PC1..PCN entries to the color-by dropdown.

        Called after PCA computation. Removes old PC entries first.
        """
        combo = self._color_combo
        combo.blockSignals(True)
        # Remove old PC entries + separator
        for i in range(combo.count() - 1, -1, -1):
            text = combo.itemText(i)
            if text.startswith("PC") or text.startswith("──"):
                combo.removeItem(i)
        # Add separator + new entries
        if n_components > 0:
            combo.addItem("── PCs ──")
            # Make separator non-selectable
            idx = combo.count() - 1
            combo.model().item(idx).setEnabled(False)
            for i in range(n_components):
                combo.addItem(f"PC{i + 1}")
        combo.blockSignals(False)

    def clear(self):
        """Remove all data and reset."""
        self._umap_cache = None
        self._tsne_cache = None
        self._df = None
        self._qc_statuses.clear()
        self._scatter_label_ids.clear()
        self._scatter_to_df_index.clear()
        self._clicked_df_index = None
        self._selected_indices.clear()
        self._selected_categories.clear()
        self._selected_category_df_indices = []
        self._selection_lasso.reset()
        self._lasso_points.clear()
        self._plot_widget.clear()
        for tl in self._tier_lines:
            self._plot_widget.addItem(tl, ignoreBounds=True)
            tl.setVisible(False)
        self._plot_widget.addItem(self._selection_lasso, ignoreBounds=True)
        self._stats_label.setText("")
        self._legend_label.setText("")
        self._colorbar_label.setText("")
        if self._layout_mode == "workbench":
            # no stale gradient legend, a greyed
            # Clear, and the empty sentence (if the host set one).
            self._wb_gradient.hide()
            self._legend_label.show()
            self._clear_sel_btn.setVisible(False)
            self._redraw()

    def reset_tier_lines(self):
        """Reset tier line state.

        Phase E removed ``Histogram`` from ``PLOT_TYPES`` so the right
        plot's tier lines never become visible — the canonical tier
        lines now live on ``triage_histogram_panel``.  This method is
        kept as a no-op-ish hook because the controller still calls it
        on patch switch (controller.py:~4150); leaving the call site
        untouched avoids churn outside the qc_plot_widget surface.
        """
        for line in self._tier_lines:
            line.setVisible(False)

    # --- Colormap helpers ---------------------------------------------------

    def _active_cmap_stops(self) -> np.ndarray:
        """Return the stops array for the currently selected colormap."""
        name = (
            self._cmap_combo.currentText()
            if self._cmap_combo.count() > 0
            else "Viridis"
        )
        return _COLORMAPS.get(name, _COLORMAPS["Viridis"])

    # --- Lasso selection ----------------------------------------------------

    #: Minimum distance (in data coords) between consecutive lasso points.
    _LASSO_MIN_DIST = 0.0

    def eventFilter(self, obj, event):
        """Intercept mouse events on plot viewport for lasso selection."""
        if obj is not self._plot_widget.viewport():
            return super().eventFilter(obj, event)

        etype = event.type()

        # Right-click anywhere clears selection (works even without lasso mode)
        if etype == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            if self._selected_indices:
                self._clear_selection()
                return True

        if not self._select_btn.isChecked():
            return super().eventFilter(obj, event)
        if not self._scatter_label_ids:
            return super().eventFilter(obj, event)

        if etype == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            vb = self._plot_widget.plotItem.vb
            scene_pos = self._plot_widget.mapToScene(event.pos())
            view_pos = vb.mapSceneToView(scene_pos)
            # Start a new lasso
            self._lasso_points.clear()
            self._selection_lasso.reset()
            pt = (view_pos.x(), view_pos.y())
            self._lasso_points.append(pt)
            self._selection_lasso.add_point(*pt)
            self._selection_lasso.setVisible(True)
            return True

        if etype == QEvent.MouseMove and self._lasso_points:
            vb = self._plot_widget.plotItem.vb
            scene_pos = self._plot_widget.mapToScene(event.pos())
            view_pos = vb.mapSceneToView(scene_pos)
            pt = (view_pos.x(), view_pos.y())
            # Decimation: skip if too close to last point
            lx, ly = self._lasso_points[-1]
            dist = ((pt[0] - lx) ** 2 + (pt[1] - ly) ** 2) ** 0.5
            if dist > self._LASSO_MIN_DIST:
                self._lasso_points.append(pt)
                self._selection_lasso.add_point(*pt)
            return True

        if (
            etype == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
            and self._lasso_points
        ):
            self._selection_lasso.close_path()
            self._finish_lasso_select()
            return True

        return super().eventFilter(obj, event)

    def _finish_lasso_select(self):
        """Find scatter points inside the lasso polygon."""
        from qtpy.QtCore import QTimer

        x_data, y_data = self._get_scatter_coords()

        if x_data is None or len(self._lasso_points) < 3:
            self._selection_lasso.reset()
            self._lasso_points.clear()
            self._redraw(_lightweight=True)
            return

        from matplotlib.path import Path as MplPath

        verts = list(self._lasso_points)
        mpl_path = MplPath(verts)
        points = np.column_stack([x_data, y_data])
        inside = mpl_path.contains_points(points)

        n_scatter = len(self._scatter_label_ids)
        self._selected_indices = set(
            int(i) for i in np.where(inside)[0] if int(i) < n_scatter
        )
        self._clicked_df_index = None  # lasso overrides single-click
        self._selection_lasso.reset()
        self._lasso_points.clear()
        if self._layout_mode == "workbench":
            # A lasso replaces a legend pick or a kept selection (X1-1).
            self._selected_categories.clear()
            self._selected_category_df_indices = []

        if self._selected_indices:
            selected_ids = [
                int(self._scatter_label_ids[i]) for i in sorted(self._selected_indices)
            ]
            self._stats_label.setText(
                f"{len(selected_ids)} selected of {len(self._scatter_label_ids)}"
            )
            self._clear_sel_btn.setVisible(True)
            # _lightweight=True: update plot visuals (selected point
            # highlight) without emitting viewer_color_requested.
            # The deferred labels_selected signal below triggers the
            # coordinator which applies isolation-aware coloring —
            # emitting viewer_color_requested here would redundantly
            # call _update_viewer_feature_colors with pre-isolation
            # state (all labels colored, immediately overwritten).
            self._redraw(_lightweight=True)
            # Defer expensive viewer isolation to after plot repaint
            QTimer.singleShot(0, lambda: self.labels_selected.emit(selected_ids))
        else:
            self._selected_indices.clear()
            self._clear_sel_btn.setVisible(False)
            self._redraw(_lightweight=True)
            QTimer.singleShot(0, lambda: self.labels_selected.emit([]))

    def get_selected_df_indices(self) -> list:
        """Return DataFrame row indices of currently selected points.

        Translates internal scatter-position indices through
        ``_scatter_to_df_index`` so the caller always receives indices
        valid for the full (unfiltered) DataFrame, even when ghost rows
        have been excluded from the rendered scatter plot.

        Priority: lasso selection > category selection > single-click.
        """
        mapping = self._scatter_to_df_index
        if self._selected_indices:
            if mapping:
                return sorted(
                    mapping[i] for i in self._selected_indices
                    if i < len(mapping)
                )
            return sorted(self._selected_indices)
        if self._selected_category_df_indices:
            return self._selected_category_df_indices
        if self._clicked_df_index is not None:
            if mapping and self._clicked_df_index < len(mapping):
                return [mapping[self._clicked_df_index]]
            return [self._clicked_df_index]
        return []

    def selected_label_ids(self) -> list:
        """Label ids of the points the plot shows highlighted RIGHT NOW.

        Read live from the plot's own selection state, never cached, so it
        also follows ``highlight_labels`` -- which rewrites the selection
        without emitting ``labels_selected`` (a navigator or annotator
        selection repaints the plot that way).  Same priority as
        :meth:`get_selected_df_indices` (lasso > category > click); each
        index is read against the ``label`` column exactly as the
        Assign-tier path reads it (positionally).  Sorted, de-duplicated
        ints; ``[]`` with no DataFrame or no selection.
        """
        df = self._df
        if df is None or "label" not in getattr(df, "columns", ()):
            return []
        labels = df["label"].to_numpy()
        out = set()
        for i in self.get_selected_df_indices():
            try:
                i = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(labels):
                out.add(int(labels[i]))
        return sorted(out)

    def _clear_selection(self):
        """Clear lasso/category selection, restore normal view, and notify controller."""
        self._selected_indices.clear()
        self._clicked_df_index = None
        self._selected_categories.clear()
        self._selected_category_df_indices = []
        self._selection_lasso.reset()
        self._lasso_points.clear()
        self._clear_sel_btn.setVisible(False)
        self._stats_label.setText("")
        # _lightweight=True: update plot visuals without emitting
        # viewer_color_requested — the deferred labels_selected signal
        # below triggers the coordinator which handles colormap restore.
        self._redraw(_lightweight=True)
        # Defer expensive viewer isolation clearing to after plot repaint
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, lambda: self.labels_selected.emit([]))

    def highlight_labels(self, label_ids: list):
        """Programmatically highlight specific labels in the plot.

        Maps label IDs to scatter point indices and applies the same
        visual treatment as lasso selection (enlarged, bright outline).
        Pass an empty list to clear the highlight.
        """
        # Clear single-click state — programmatic highlight replaces it.
        self._clicked_df_index = None

        if not label_ids:
            self._selected_indices.clear()
            self._selected_categories.clear()
            self._selected_category_df_indices = []
            self._clear_sel_btn.setVisible(False)
            self._stats_label.setText("")
            self._redraw()
            return

        id_set = set(int(lid) for lid in label_ids)
        matched = set()
        for i, lid in enumerate(self._scatter_label_ids):
            if int(lid) in id_set:
                matched.add(i)
        if matched:
            self._selected_indices = matched
            self._selected_categories.clear()
            self._selected_category_df_indices = []
            n = len(matched)
            self._stats_label.setText(
                f"{n} selected of {len(self._scatter_label_ids)}"
            )
            self._clear_sel_btn.setVisible(True)
            self._redraw()

    def _on_legend_clicked(self, link: str):
        """Handle click on a legend swatch link."""
        if not link.startswith("cat://"):
            return
        parts = link[6:].split("/", 1)
        if len(parts) != 2:
            return
        col_name, cat_value = parts[0], parts[1]
        # Detect Ctrl modifier for additive multi-category selection
        from qtpy.QtWidgets import QApplication
        additive = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        self._toggle_category_selection(col_name, cat_value, additive=additive)

    def _toggle_category_selection(self, col_name: str, cat_value: str, *, additive: bool = False):
        """Toggle category selection with optional additive (Ctrl+click) mode.

        Without Ctrl: replaces selection (or toggles off if same single category).
        With Ctrl: adds/removes category from the current multi-selection.
        """
        # Clear single-click state — category selection replaces it.
        self._clicked_df_index = None

        wb_tier = (self._layout_mode == "workbench" and col_name == "qc_tier")
        if self._df is None or (col_name not in self._df.columns
                                and not wb_tier):
            return
        self._selected_categories.discard(_WB_KEPT)

        from qtpy.QtCore import QTimer

        key = (col_name, cat_value)

        if additive:
            # Ctrl+click: toggle this category in/out of multi-selection
            if key in self._selected_categories:
                self._selected_categories.discard(key)
            else:
                self._selected_categories.add(key)
            # Clear lasso — categories and lasso are mutually exclusive
            self._selected_indices.clear()
            self._lasso_points.clear()
            self._selection_lasso.reset()
        else:
            # Normal click: replace or toggle off
            if (self._selected_categories == {key}
                    and self._layout_mode == "workbench"
                    and self._wb_live_positions(key)
                    != list(self._selected_category_df_indices)):
                # Re-audit LOW-6: after a mark the pick's frozen rows are
                # not this entry's cells any more: re-pick the live ones
                # (none -> the empty pick below clears it).
                self._selected_categories = {key}
            elif self._selected_categories == {key}:
                # Toggle off — same single category clicked again
                self._selected_categories.clear()
            else:
                self._selected_categories = {key}
            # Clear lasso
            self._selected_indices.clear()
            self._lasso_points.clear()
            self._selection_lasso.reset()

        # Rebuild combined df indices and label IDs from all selected categories
        if not self._selected_categories:
            self._selected_category_df_indices = []
            self._clear_sel_btn.setVisible(False)
            self._stats_label.setText("")
            self._redraw()
            self.category_selected.emit("", "")
            QTimer.singleShot(0, lambda: self.labels_selected.emit([]))
            return

        # Union all selected category masks
        combined_mask = np.zeros(len(self._df), dtype=bool)
        for sc_col, sc_val in self._selected_categories:
            if self._layout_mode == "workbench" and sc_col == "qc_tier":
                # The Workbench's buckets: the card's rule (E1-5).
                combined_mask |= self._wb_buckets(self._df) == sc_val
                continue
            if sc_col not in self._df.columns:
                continue
            col = self._df[sc_col]
            # 2026-05-17 fix: robust type coercion that handles BOTH
            # numpy int/float dtypes AND pandas nullable dtypes
            # (Int64, Float64) — the previous `col.dtype in (...)`
            # check only matched the lowercase numpy dtypes, so the
            # cluster_id column (which becomes Int64 when ghosts
            # were initialized with pd.NA) silently failed comparison
            # and the mask was all-False → no cells highlighted on
            # legend click.
            #
            # Strategy: try int → float → string, in that order.
            # Falls through gracefully for any column type
            # (cluster_id, is_border_cell, qc_tier, patch_name, etc.).
            try:
                typed_val = int(sc_val)
            except (ValueError, TypeError):
                try:
                    typed_val = float(sc_val)
                except (ValueError, TypeError):
                    typed_val = sc_val
            combined_mask |= (col == typed_val).values

        if self._layout_mode == "workbench":
            combined_mask &= self._wb_active_mask()   # MED-2
        self._selected_category_df_indices = list(np.where(combined_mask)[0])
        label_ids = [int(lid) for lid in self._df.loc[combined_mask, "label"].values]

        if label_ids:
            n_cats = len(self._selected_categories)
            if n_cats == 1:
                _only = next(iter(self._selected_categories))
                self._stats_label.setText(f"{len(label_ids)} labels in '{_only[1]}'")
            else:
                self._stats_label.setText(f"{len(label_ids)} labels in {n_cats} categories")
            self._clear_sel_btn.setVisible(True)
            self._redraw()
            # Emit first selected category for breakdown table compatibility
            first = next(iter(self._selected_categories))
            self.category_selected.emit(first[0], first[1])
            _ids = label_ids
            QTimer.singleShot(0, lambda: self.labels_selected.emit(_ids))
        else:
            self._selected_categories.clear()
            self._selected_category_df_indices = []
            self._redraw()
            self.category_selected.emit("", "")
            QTimer.singleShot(0, lambda: self.labels_selected.emit([]))

    def _get_scatter_coords(self):
        """Return (x, y) arrays for current scatter/PCA plot data.

        Returns only **active** (non-ghost) coordinates so that positional
        indices are consistent with ``_scatter_label_ids`` and
        ``_scatter_to_df_index``, which are built from the ghost-filtered
        ``active_df`` during rendering.
        """
        if self._df is None:
            return None, None

        ghost_mask = self._get_ghost_mask()

        plot_type = self.plot_type_combo.currentText()
        if plot_type == "Scatter":
            feat_x = self._feature_x_combo.currentText()
            feat_y = self._feature_y_combo.currentText()
            if feat_x in self._df.columns and feat_y in self._df.columns:
                x = self._df[feat_x].values.astype(np.float64)
                y = self._df[feat_y].values.astype(np.float64)
                if ghost_mask is not None:
                    active = ~ghost_mask
                    x, y = x[active], y[active]
                return x, y
        elif plot_type == "PCA":
            projected = self._get_pca_projection()
            if projected.shape[1] >= 2:
                x, y = projected[:, 0], projected[:, 1]
                if ghost_mask is not None:
                    active = ~ghost_mask
                    x, y = x[active], y[active]
                return x, y
        elif plot_type == "UMAP":
            if self._umap_cache is not None and self._umap_cache.shape[1] >= 2:
                x, y = self._umap_cache[:, 0], self._umap_cache[:, 1]
                if ghost_mask is not None:
                    active = ~ghost_mask
                    x, y = x[active], y[active]
                return x, y
        elif plot_type == "t-SNE":
            if self._tsne_cache is not None and self._tsne_cache.shape[1] >= 2:
                x, y = self._tsne_cache[:, 0], self._tsne_cache[:, 1]
                if ghost_mask is not None:
                    active = ~ghost_mask
                    x, y = x[active], y[active]
                return x, y
        return None, None

    # --- Internal -----------------------------------------------------------

    def _sync_feature_combos(self):
        """Populate feature selectors from DataFrame columns.

        Preserves the user's current X, Y, and Color-by selections across
        DataFrame updates (e.g., after label deletion) as long as the
        previously selected item still exists in the new option list.
        """
        from .qc_features import NUMERIC_FEATURES

        # Save current selections before clearing
        prev_x = self._feature_x_combo.currentText()
        prev_y = self._feature_y_combo.currentText()
        prev_color = self._color_combo.currentText()

        for combo in (self._feature_x_combo, self._feature_y_combo, self._color_combo):
            combo.blockSignals(True)
            combo.clear()

        if self._df is not None:
            # Use NUMERIC_FEATURES as preferred order, then append any extra
            # numeric columns from the DataFrame (e.g., per-channel intensity
            # features from Step 4) that aren't in the hardcoded list.
            known = [c for c in NUMERIC_FEATURES if c in self._df.columns]
            known_set = set(known)
            # Exclude non-feature columns
            _EXCLUDE = {"label", "patch_name", "stack_name", "label_status",
                        "feature_state", "qc_tier"}
            extra = [
                c for c in self._df.select_dtypes(include="number").columns
                if c not in known_set
                and c not in _EXCLUDE
                and not c.startswith("bbox-")
                and not c.startswith("inertia_tensor")
            ]
            cols = known + sorted(extra)
            self._feature_x_combo.addItems(cols)
            self._feature_y_combo.addItems(cols)
            # Color-by options: categorical first, then continuous features
            color_options = ["QC status"]
            if ("qc_tier" in self._df.columns
                    or self._layout_mode == "workbench"):
                # The Workbench offers "QC tier" before the first mark:
                # every cell is then unassigned (grey).
                color_options.append("QC tier")
            if "cluster_id" in self._df.columns:
                color_options.append("Cluster")
            if "patch_name" in self._df.columns:
                color_options.append("Patch")
            if "is_border_cell" in self._df.columns:
                color_options.append("Border cell")
            if "euler_number" in self._df.columns:
                color_options.append("Topology")
            color_options.extend(cols)
            self._color_combo.addItems(color_options)

            # Restore previous selections (fall back to defaults if missing)
            x_idx = self._feature_x_combo.findText(prev_x)
            if x_idx >= 0:
                self._feature_x_combo.setCurrentIndex(x_idx)

            y_idx = self._feature_y_combo.findText(prev_y)
            if y_idx >= 0:
                self._feature_y_combo.setCurrentIndex(y_idx)
            elif len(cols) >= 2:
                self._feature_y_combo.setCurrentIndex(1)

            color_idx = self._color_combo.findText(prev_color)
            if color_idx >= 0:
                self._color_combo.setCurrentIndex(color_idx)

        for combo in (self._feature_x_combo, self._feature_y_combo, self._color_combo):
            combo.blockSignals(False)

    def _on_plot_type_changed(self, _index):
        """Update visibility of axis selectors based on plot type."""
        plot_type = self.plot_type_combo.currentText()
        is_scatter = plot_type == "Scatter"
        is_violin = plot_type == "Violin"
        is_embedding = plot_type in ("PCA", "UMAP", "t-SNE")

        self._y_label.setVisible(is_scatter)
        self._feature_y_combo.setVisible(is_scatter)
        self._select_btn.setVisible(not is_violin)
        # Phase A.5: _umap_from_pca_cb removed (UMAP pre-reduction is
        # now controlled at compute time, not from the plot widget).

        if is_scatter:
            self._x_label.setText("X axis")
        elif is_embedding:
            self._x_label.setText("")
        else:
            self._x_label.setText("Feature")

        kept = False
        if (self._layout_mode == "workbench"
                and self._wb_keep_selection_as_rows()):
            kept = True   # X1-2: the Workbench keeps the selection across plots
        elif self._selected_indices or self._selected_categories:
            self._selected_indices.clear()
            self._selected_categories.clear()
            self._selected_category_df_indices = []
            self._clear_sel_btn.setVisible(False)
            self.labels_selected.emit([])  # clear controller isolation
        self._redraw()
        if kept:
            self._on_selection_kept()

    def _redraw(self, *, _lightweight=False):
        """Full redraw based on current plot type.

        Args:
            _lightweight: If True, skip autoRange and viewer color sync.
                Used during threshold line dragging for smooth interaction.
        """
        if self._layout_mode == "workbench":
            self._wb_sync_axes()
            item = self._plot_widget.getPlotItem()
            item.showAxis("bottom")
            item.showAxis("left")
        self._plot_widget.clear()
        for tl in self._tier_lines:
            self._plot_widget.addItem(tl, ignoreBounds=True)
            # Phase E removed Histogram from PLOT_TYPES — right-plot
            # tier lines never become visible; canonical tier lines
            # live on triage_histogram_panel.
            tl.setVisible(False)
        self._plot_widget.addItem(self._selection_lasso, ignoreBounds=True)
        self._scatter_label_ids.clear()
        self._scatter_to_df_index.clear()

        # Reset axis ticks — Violin sets custom category ticks that persist
        for axis_name in ("bottom", "left"):
            self._plot_widget.getAxis(axis_name).setTicks(None)

        if self._df is None or self._df.empty:
            self._stats_label.setText("")
            if (self._layout_mode == "workbench"
                    and self._wb_empty_sentence):
                self._wb_draw_sentence(self._wb_empty_sentence)
            return

        plot_type = self.plot_type_combo.currentText()

        if plot_type == "Scatter":
            self._draw_scatter()
        elif plot_type == "PCA":
            self._draw_pca()
        elif plot_type == "UMAP":
            self._draw_umap()
        elif plot_type == "t-SNE":
            self._draw_tsne()
        elif plot_type == "Violin":
            self._draw_violin()

        self._update_stats()
        self._update_legend()
        if self._layout_mode == "workbench":
            self._wb_update_legend()

        if not _lightweight:
            # Auto-fit view to new data (reset any previous zoom/pan)
            self._plot_widget.getViewBox().autoRange()

            # Notify controller to sync viewer label colors.
            # Always emit — the controller's coloring methods are isolation-aware
            # and will apply feature colors to only isolated labels when active.
            color_by = self._color_combo.currentText() if self._color_combo.count() > 0 else ""
            cmap_name = self._cmap_combo.currentText() if self._cmap_combo.count() > 0 else "Viridis"
            norm_mode = self._norm_combo.currentText() if self._norm_combo.count() > 0 else "Linear"
            self.viewer_color_requested.emit(color_by, cmap_name, norm_mode)

    def _normalize_values(self, values: np.ndarray) -> np.ndarray:
        """Normalize values to [0, 1] using the current normalization mode.

        Modes:
        - Linear: raw min-max normalization
        - Log: log1p transform then min-max
        - Rank: percentile rank (uniform spread, distribution-agnostic)
        """
        n = len(values)
        mode = (
            self._norm_combo.currentText()
            if self._norm_combo.count() > 0
            else "Linear"
        )
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return np.zeros(n)

        v = values.copy()

        if mode == "Log":
            v[finite_mask] = np.log1p(np.clip(v[finite_mask], 0, None))
        elif mode == "Rank":
            from scipy.stats import rankdata
            ranked = np.zeros(n)
            ranked[finite_mask] = rankdata(v[finite_mask]) / finite_mask.sum()
            ranked[~finite_mask] = 0.0
            return ranked

        vmin = v[finite_mask].min()
        vmax = v[finite_mask].max()
        if vmax > vmin:
            normed = (v - vmin) / (vmax - vmin)
        else:
            normed = np.full(n, 0.5)
        normed[~finite_mask] = 0.0
        return normed

    def _get_point_colors(self) -> np.ndarray:
        """Return RGBA color array based on color-by selector (full DataFrame)."""
        return self._get_point_colors_for(self._df)

    def _get_point_colors_for(self, df) -> np.ndarray:
        """Return RGBA color array based on color-by selector for *df*.

        Works on an arbitrary DataFrame (e.g. active-only subset).
        """
        n = len(df)
        color_by = (
            self._color_combo.currentText()
            if self._color_combo.count() > 0
            else ""
        )

        default_rgba = np.array([_DEFAULT_COLOR[0] / 255, _DEFAULT_COLOR[1] / 255,
                                 _DEFAULT_COLOR[2] / 255, 0.85], dtype=np.float32)

        # Use QC status coloring
        if color_by == "QC status" or not color_by:
            colors = np.zeros((n, 4), dtype=np.float32)
            self._colorbar_label.setText("")
            if "qc_status" in df.columns:
                status_vals = df["qc_status"].values
                colors[:] = default_rgba
                for status, c in _STATUS_COLORS.items():
                    mask = status_vals == status
                    if mask.any():
                        colors[mask] = [c[0] / 255, c[1] / 255, c[2] / 255, 0.85]
            elif self._qc_statuses:
                label_ids = df["label"].values
                colors[:] = default_rgba
                for status, c in _STATUS_COLORS.items():
                    rgba = np.array([c[0] / 255, c[1] / 255, c[2] / 255, 0.85], dtype=np.float32)
                    mask = np.array([self._qc_statuses.get(int(lid), "") == status for lid in label_ids])
                    if mask.any():
                        colors[mask] = rgba
            else:
                colors[:, :3] = np.array(_DEFAULT_COLOR) / 255
                colors[:, 3] = 0.7
            return colors

        if color_by == "QC tier" and self._layout_mode == "workbench":
            return self._wb_bucket_colors(df)

        # Use QC tier coloring
        if color_by == "QC tier" and "qc_tier" in df.columns:
            colors = np.zeros((n, 4), dtype=np.float32)
            tiers = df["qc_tier"].values
            colors[:] = default_rgba
            for tier, c in _TIER_COLORS.items():
                mask = tiers == tier
                if mask.any():
                    colors[mask] = [c[0] / 255, c[1] / 255, c[2] / 255, 0.85]
            return colors

        # Use cluster coloring
        if color_by == "Cluster" and "cluster_id" in df.columns:
            colors = np.zeros((n, 4), dtype=np.float32)
            cluster_ids = df["cluster_id"].values
            # Filter NaN / pd.NA (ghost rows) before sorting.
            unique_ids = _sorted_unique_cluster_ids(cluster_ids)
            palette = self._PATCH_PALETTE
            for cid in unique_ids:
                mask = cluster_ids == cid
                if cid == -1:
                    colors[mask] = [0.4, 0.4, 0.4, 0.5]
                else:
                    colors[mask] = palette[int(cid) % len(palette)]
            return colors

        # Use topology (euler number) categorical coloring
        if color_by == "Topology" and "euler_number" in df.columns:
            colors = np.zeros((n, 4), dtype=np.float32)
            euler = df["euler_number"].values
            colors[euler == 1] = [100/255, 200/255, 100/255, 0.85]  # Solid
            colors[euler >= 2] = [180/255, 130/255, 255/255, 0.85]  # Cavity
            colors[euler == 0] = [255/255, 180/255, 50/255, 0.85]   # Tunnel
            colors[euler < 0] = [255/255, 80/255, 80/255, 0.85]     # Complex
            return colors

        # Use border cell categorical coloring
        if color_by == "Border cell" and "is_border_cell" in df.columns:
            colors = np.zeros((n, 4), dtype=np.float32)
            vals = df["is_border_cell"].values
            colors[vals == 0] = [100 / 255, 180 / 255, 255 / 255, 0.85]  # blue
            colors[vals == 1] = [255 / 255, 100 / 255, 100 / 255, 0.85]  # red
            return colors

        # Use per-patch categorical coloring
        if color_by == "Patch" and "patch_name" in df.columns:
            colors = np.zeros((n, 4), dtype=np.float32)
            patches = df["patch_name"].values
            unique_patches = list(dict.fromkeys(patches))
            palette = self._PATCH_PALETTE
            for idx, p in enumerate(unique_patches):
                mask = patches == p
                colors[mask] = palette[idx % len(palette)]
            return colors

        # Use PC coloring (from cached PCA projection)
        if color_by.startswith("PC") and color_by[2:].isdigit():
            pc_idx = int(color_by[2:]) - 1  # PC1 → index 0
            if (self._pc_projected is not None
                    and pc_idx < self._pc_projected.shape[1]
                    and self._pc_projected.shape[0] == n):
                values = self._pc_projected[:, pc_idx].astype(np.float64)
                normed = self._normalize_values(values)
                vmin = values.min()
                vmax = values.max()
                self._colorbar_label.setText(
                    f"  {vmin:.1f} \u2500\u2500 {vmax:.1f}"
                )
                return _apply_colormap(normed, self._active_cmap_stops())

        # Use continuous feature coloring
        if color_by in df.columns:
            values = df[color_by].values.astype(np.float64)
            normed = self._normalize_values(values)
            finite_mask = np.isfinite(values)
            if finite_mask.any():
                vmin = values[finite_mask].min()
                vmax = values[finite_mask].max()
                self._colorbar_label.setText(
                    f"  {vmin:.1f} \u2500\u2500 {vmax:.1f}"
                )
                return _apply_colormap(normed, self._active_cmap_stops())

        # Fallback
        colors = np.zeros((n, 4), dtype=np.float32)
        colors[:, :3] = np.array(_DEFAULT_COLOR) / 255
        colors[:, 3] = 0.7
        return colors

    def _get_status_colors(self) -> np.ndarray:
        """Return RGBA colors from QC status column."""
        n = len(self._df)
        colors = np.zeros((n, 4), dtype=np.float32)
        self._colorbar_label.setText("")

        default_rgba = np.array([_DEFAULT_COLOR[0] / 255, _DEFAULT_COLOR[1] / 255,
                                 _DEFAULT_COLOR[2] / 255, 0.85], dtype=np.float32)

        if "qc_status" in self._df.columns:
            # Vectorized: build lookup, map via numpy
            status_vals = self._df["qc_status"].values
            colors[:] = default_rgba
            for status, c in _STATUS_COLORS.items():
                mask = status_vals == status
                if mask.any():
                    colors[mask] = [c[0] / 255, c[1] / 255, c[2] / 255, 0.85]
        elif self._qc_statuses:
            label_ids = self._df["label"].values
            colors[:] = default_rgba
            for status, c in _STATUS_COLORS.items():
                rgba = np.array([c[0] / 255, c[1] / 255, c[2] / 255, 0.85], dtype=np.float32)
                mask = np.array([self._qc_statuses.get(int(lid), "") == status for lid in label_ids])
                if mask.any():
                    colors[mask] = rgba
        else:
            colors[:, :3] = np.array(_DEFAULT_COLOR) / 255
            colors[:, 3] = 0.7

        return colors

    def _get_tier_colors(self) -> np.ndarray:
        """Return categorical RGBA colors from QC tier column."""
        if self._layout_mode == "workbench":
            return self._wb_bucket_colors(self._df)
        n = len(self._df)
        colors = np.zeros((n, 4), dtype=np.float32)
        tiers = self._df["qc_tier"].values

        default_rgba = np.array([_DEFAULT_COLOR[0] / 255, _DEFAULT_COLOR[1] / 255,
                                 _DEFAULT_COLOR[2] / 255, 0.85], dtype=np.float32)
        colors[:] = default_rgba
        for tier, c in _TIER_COLORS.items():
            mask = tiers == tier
            if mask.any():
                colors[mask] = [c[0] / 255, c[1] / 255, c[2] / 255, 0.85]

        return colors

    def _get_cluster_colors(self) -> np.ndarray:
        """Return categorical RGBA colors from cluster_id column."""
        n = len(self._df)
        colors = np.zeros((n, 4), dtype=np.float32)
        cluster_ids = self._df["cluster_id"].values
        # Filter NaN / pd.NA (ghost rows) before sorting.
        unique_ids = _sorted_unique_cluster_ids(cluster_ids)
        palette = self._PATCH_PALETTE

        for cid in unique_ids:
            mask = cluster_ids == cid
            if cid == -1:
                colors[mask] = [0.4, 0.4, 0.4, 0.5]
            else:
                colors[mask] = palette[int(cid) % len(palette)]

        return colors

    # 12 distinct categorical colors for per-patch coloring
    _PATCH_PALETTE = np.array([
        [0.12, 0.47, 0.71, 0.85],  # blue
        [1.00, 0.50, 0.05, 0.85],  # orange
        [0.17, 0.63, 0.17, 0.85],  # green
        [0.84, 0.15, 0.16, 0.85],  # red
        [0.58, 0.40, 0.74, 0.85],  # purple
        [0.55, 0.34, 0.29, 0.85],  # brown
        [0.89, 0.47, 0.76, 0.85],  # pink
        [0.50, 0.50, 0.50, 0.85],  # gray
        [0.74, 0.74, 0.13, 0.85],  # olive
        [0.09, 0.75, 0.81, 0.85],  # cyan
        [0.94, 0.89, 0.26, 0.85],  # yellow
        [0.36, 0.72, 0.36, 0.85],  # lime
    ], dtype=np.float32)

    def _get_patch_colors(self) -> np.ndarray:
        """Return categorical RGBA colors per patch name."""
        n = len(self._df)
        colors = np.zeros((n, 4), dtype=np.float32)
        patches = self._df["patch_name"].values
        unique_patches = list(dict.fromkeys(patches))  # preserve order
        palette = self._PATCH_PALETTE

        for idx, p in enumerate(unique_patches):
            mask = patches == p
            colors[mask] = palette[idx % len(palette)]

        return colors

    def _apply_selection_dimming(self, colors: np.ndarray, n: int, cat_mask):
        """Dim alpha channel for unselected points (vectorized)."""
        if self._selected_indices:
            sel_arr = np.zeros(n, dtype=bool)
            for idx in self._selected_indices:
                if 0 <= idx < n:
                    sel_arr[idx] = True
            colors[~sel_arr, 3] *= 0.35
        elif cat_mask is not None:
            colors[~cat_mask, 3] *= 0.25

    def _safe_align_colors(self, colors: np.ndarray, n: int) -> np.ndarray:
        """Coerce ``colors`` to shape ``(n, 4)`` so brush count matches
        point count.

        Defensive guard added 2026-05-04: if a caller's color array
        was computed against a different row count than the active
        scatter set (e.g. stale cache, df mutation between fetch and
        plot), fall back to a uniform default rather than crashing
        ``pyqtgraph.ScatterPlotItem`` with a brush-count mismatch.
        Logs once via ``print`` so the upstream cause is still
        visible during diagnosis without erroring the workflow.
        """
        if colors is None or len(colors) != n:
            print(
                f"[QC-PLOT] color size mismatch (got "
                f"{0 if colors is None else len(colors)}, "
                f"expected {n}); falling back to default brush. "
                "Likely a stale color/feature cache against a "
                "changed df — please report if this happens often."
            )
            fallback = np.zeros((n, 4), dtype=np.float32)
            fallback[:, :3] = np.array(_DEFAULT_COLOR) / 255.0
            fallback[:, 3] = 0.7
            return fallback
        return colors

    def _colors_to_brushes(self, colors: np.ndarray) -> list:
        """Convert RGBA float array to list of QBrush."""
        rgba_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
        return [pg.mkBrush(int(r), int(g), int(b), int(a))
                for r, g, b, a in rgba_u8]

    def _selection_mask(self, n: int) -> np.ndarray:
        """Return boolean array of selected indices."""
        mask = np.zeros(n, dtype=bool)
        for idx in self._selected_indices:
            if 0 <= idx < n:
                mask[idx] = True
        return mask

    def _point_sizes(self, n: int) -> np.ndarray:
        """Return per-point sizes, enlarged for selected points."""
        if not self._selected_indices:
            return np.full(n, 7, dtype=np.float32)
        sizes = np.full(n, 5, dtype=np.float32)
        sizes[self._selection_mask(n)] = 11
        return sizes

    def _point_pens(self, n: int) -> list:
        """Return per-point outline pens, highlighted for selected."""
        default_pen = pg.mkPen(color=(255, 255, 255, 30), width=0.5)
        if not self._selected_indices:
            return [default_pen] * n
        sel_pen = pg.mkPen(color=(100, 180, 255, 220), width=1.5)
        dim_pen = pg.mkPen(color=(255, 255, 255, 15), width=0.5)
        mask = self._selection_mask(n)
        pens = [dim_pen] * n
        for i in np.where(mask)[0]:
            pens[i] = sel_pen
        return pens

    def _point_sizes_with_cat(self, n: int, cat_mask) -> np.ndarray:
        """Sizes considering lasso *and* category selection."""
        if self._selected_indices:
            return self._point_sizes(n)
        if cat_mask is not None:
            sizes = np.full(n, 5, dtype=np.float32)
            sizes[cat_mask] = 11
            return sizes
        return self._point_sizes(n)

    def _point_pens_with_cat(self, n: int, cat_mask) -> list:
        """Pens considering lasso *and* category selection."""
        if self._selected_indices:
            return self._point_pens(n)
        if cat_mask is not None:
            sel_pen = pg.mkPen(color=(100, 180, 255, 220), width=1.5)
            dim_pen = pg.mkPen(color=(255, 255, 255, 15), width=0.5)
            pens = [dim_pen] * n
            for i in np.where(cat_mask)[0]:
                pens[i] = sel_pen
            return pens
        return self._point_pens(n)

    # Categorical color-by modes that trigger stacked histogram
    _CATEGORICAL_COLOR_BY = {"QC status", "QC tier", "Cluster", "Patch", "Border cell", "Topology"}

    # Ghost point color: visible gray with sufficient alpha
    _GHOST_COLOR = (140, 140, 160, 130)
    #: A scheme with ``ghosts_behind`` (Step 3): 7 px, brush alpha 110 and a
    #: faint 1 px pen -- an "x" is ALL outline, so a pen-less one draws
    #: nothing (P2 audit MED-5).
    _GHOST_BEHIND_SIZE = 7
    _GHOST_BEHIND_ALPHA = 110
    _GHOST_BEHIND_PEN = (140, 140, 160, 120)

    def _get_ghost_mask(self) -> Optional[np.ndarray]:
        """Return boolean mask of ghost (deleted/superseded) rows, or None."""
        if self._df is None or "label_status" not in self._df.columns:
            return None
        statuses = self._df["label_status"].values
        mask = (statuses == "deleted") | (statuses == "superseded")
        if not mask.any():
            return None
        return mask

    def _add_ghost_scatter(self, x: np.ndarray, y: np.ndarray,
                           ghost_mask: np.ndarray) -> None:
        """Add a separate non-interactive scatter for ghost points."""
        gx = x[ghost_mask]
        gy = y[ghost_mask]
        if len(gx) == 0:
            return
        # getattr: test fakes built from SimpleNamespace call this method
        # unbound and carry only the classic attributes.
        if (getattr(self, "_layout_mode", None) == "workbench"
                and self._scheme.ghosts_behind):
            # Owner: small, faint, BEHIND the active points, so
            # ghosts no longer read as "all bad"; a thin faint pen, because
            # an "x" with no pen is not drawn at all.
            r, g, b, _a = self._GHOST_COLOR
            behind = pg.ScatterPlotItem(
                x=gx, y=gy,
                size=self._GHOST_BEHIND_SIZE,
                brush=pg.mkBrush(r, g, b, self._GHOST_BEHIND_ALPHA),
                pen=pg.mkPen(color=self._GHOST_BEHIND_PEN, width=1.0),
                symbol="x",
            )
            behind.setZValue(-1)
            self._plot_widget.addItem(behind)
            return
        gc = self._GHOST_COLOR
        brush = pg.mkBrush(*gc)
        pen = pg.mkPen(color=(200, 200, 200, 100), width=1.0)
        ghost_scatter = pg.ScatterPlotItem(
            x=gx, y=gy,
            size=10,
            brush=brush,
            pen=pen,
            symbol="x",
        )
        self._plot_widget.addItem(ghost_scatter)

    def _get_categorical_info(self, color_by):
        """Return (column_name, {category: (r,g,b)} dict) for a categorical color-by."""
        if color_by == "QC status" and "qc_status" in self._df.columns:
            return "qc_status", {
                "pass": _PASS_COLOR,
                "flagged": _FLAGGED_COLOR,
                "excluded": _EXCLUDED_COLOR,
            }
        if color_by == "QC tier" and self._layout_mode == "workbench":
            return "qc_tier", self._wb_tier_legend_colours()
        if color_by == "QC tier" and "qc_tier" in self._df.columns:
            return "qc_tier", _TIER_COLORS
        if color_by == "Cluster" and "cluster_id" in self._df.columns:
            palette = self._PATCH_PALETTE
            # Filter NaN / pd.NA (ghost rows) before sorting.
            unique_ids = _sorted_unique_cluster_ids(
                self._df["cluster_id"].values
            )
            cmap = {}
            for cid in unique_ids:
                if cid == -1:
                    cmap[cid] = (100, 100, 100)  # noise = gray
                else:
                    c = palette[int(cid) % len(palette)]
                    cmap[cid] = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            return "cluster_id", cmap
        if color_by == "Patch" and "patch_name" in self._df.columns:
            palette = self._PATCH_PALETTE
            unique_patches = list(dict.fromkeys(self._df["patch_name"].values))
            cmap = {}
            for i, p in enumerate(unique_patches):
                c = palette[i % len(palette)]
                cmap[p] = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            return "patch_name", cmap
        if color_by == "Border cell" and "is_border_cell" in self._df.columns:
            return "is_border_cell", {
                0: (100, 180, 255),   # Interior = blue
                1: (255, 100, 100),   # Border = red
            }
        if color_by == "Topology" and "euler_number" in self._df.columns:
            import numpy as _np
            # Derive euler_category column if not present or stale
            euler = self._df["euler_number"].values
            cats = _np.where(euler == 1, "Solid",
                   _np.where(euler >= 2, "Has cavity",
                   _np.where(euler == 0, "Has tunnel", "Complex")))
            self._df["euler_category"] = cats
            _topo_colors = {
                "Solid": (100, 200, 100),
                "Has cavity": (180, 130, 255),
                "Has tunnel": (255, 180, 50),
                "Complex": (255, 80, 80),
            }
            categories = {}
            for cat in self._df["euler_category"].unique():
                categories[cat] = _topo_colors.get(cat, (150, 150, 150))
            return "euler_category", categories
        return None, {}

    def _get_selected_category_mask(self) -> Optional[np.ndarray]:
        """Return boolean mask for rows matching all selected categories (union), or None."""
        return self._get_selected_category_mask_for(self._df, len(self._df) if self._df is not None else 0)

    def _get_selected_category_mask_for(self, df, n: int) -> Optional[np.ndarray]:
        """Return boolean mask for rows matching selected categories in *df*.

        2026-05-17 fix: same robust type coercion as
        ``_toggle_category_selection``.  The previous dtype check
        only matched numpy dtypes (`np.int64`, `np.float64`, etc.),
        so the pandas-nullable `Int64` cluster_id column silently
        fell through to string comparison → all-False mask → cluster
        legend click highlighted cells in the viewer (via emit)
        but not the points in the scatter (this mask is what
        `_apply_selection_dimming` / `_point_sizes_with_cat` /
        `_point_pens_with_cat` consume).
        """
        if not self._selected_categories or df is None:
            return None
        if self._layout_mode == "workbench":
            return self._wb_frozen_mask_for(df, n)
        combined = np.zeros(n, dtype=bool)
        for col_name, cat_value in self._selected_categories:
            if col_name not in df.columns:
                continue
            col = df[col_name]
            # Try int → float → string in order; handles both numpy
            # and pandas nullable dtypes (Int64, Float64) uniformly.
            try:
                typed_val = int(cat_value)
            except (ValueError, TypeError):
                try:
                    typed_val = float(cat_value)
                except (ValueError, TypeError):
                    typed_val = cat_value
            combined |= (col == typed_val).values
        return combined if combined.any() else None

    def _draw_scatter(self):
        feat_x = self._feature_x_combo.currentText()
        feat_y = self._feature_y_combo.currentText()
        if not feat_x or not feat_y:
            return
        if feat_x not in self._df.columns or feat_y not in self._df.columns:
            return

        x_all = self._df[feat_x].values.astype(np.float64)
        y_all = self._df[feat_y].values.astype(np.float64)

        # Separate ghost (deleted/superseded) from active points.
        ghost_mask = self._get_ghost_mask()
        if ghost_mask is not None:
            active_mask = ~ghost_mask
            x = x_all[active_mask]
            y = y_all[active_mask]
            active_df = self._df[active_mask]
        else:
            x, y, active_df = x_all, y_all, self._df

        n = len(x)
        self._scatter_label_ids = active_df["label"].values.tolist()
        self._scatter_to_df_index = list(active_df.index)

        colors = self._get_point_colors_for(active_df)
        colors = self._safe_align_colors(colors, n)
        cat_mask = self._get_selected_category_mask_for(active_df, n)
        self._apply_selection_dimming(colors, n, cat_mask)

        scatter = pg.ScatterPlotItem(
            x=x, y=y,
            size=self._point_sizes_with_cat(n, cat_mask),
            brush=self._colors_to_brushes(colors),
            pen=self._point_pens_with_cat(n, cat_mask),
        )
        self._plot_widget.addItem(scatter)

        # Ghost points (if Show History is checked).
        if ghost_mask is not None and self._history_cb.isChecked():
            self._add_ghost_scatter(x_all, y_all, ghost_mask)

        if self._density_btn.isChecked():
            self._draw_density_contours(x, y)
        self._plot_widget.setLabel(
            "bottom", feat_x, **{"color": "#999", "font-size": "10px"},
        )
        self._plot_widget.setLabel(
            "left", feat_y, **{"color": "#999", "font-size": "10px"},
        )

    def _get_pca_projection(self) -> np.ndarray:
        """Return cached PCA projection, computing if needed.

        Defensive shape check (added 2026-05-04): callers that bypass
        ``set_features`` (e.g. controllers that do ``pw._df = df``
        directly during tier-drag redraws) can leave a stale cache
        with a different row count than the current df.  Re-compute
        when the shape disagrees so downstream draws never crash on
        ``brush vs n`` mismatches.
        """
        df = self._df
        n_expected = len(df) if df is not None else 0
        if (
            self._pca_cache is None
            or self._pca_cache.shape[0] != n_expected
        ):
            from .qc_features import compute_pca
            self._pca_cache = compute_pca(
                df, n_components=2, features=self._pca_features,
            )
        return self._pca_cache

    def _draw_pca(self):
        projected = self._get_pca_projection()
        if projected.shape[1] < 2:
            return

        x_all = projected[:, 0]
        y_all = projected[:, 1]

        # Separate ghost from active points.
        ghost_mask = self._get_ghost_mask()
        if ghost_mask is not None:
            active_mask = ~ghost_mask
            x = x_all[active_mask]
            y = y_all[active_mask]
            active_df = self._df[active_mask]
        else:
            x, y, active_df = x_all, y_all, self._df

        n = len(x)
        self._scatter_label_ids = active_df["label"].values.tolist()
        self._scatter_to_df_index = list(active_df.index)

        colors = self._get_point_colors_for(active_df)
        colors = self._safe_align_colors(colors, n)
        cat_mask = self._get_selected_category_mask_for(active_df, n)
        self._apply_selection_dimming(colors, n, cat_mask)

        scatter = pg.ScatterPlotItem(
            x=x, y=y,
            size=self._point_sizes_with_cat(n, cat_mask),
            brush=self._colors_to_brushes(colors),
            pen=self._point_pens_with_cat(n, cat_mask),
        )
        self._plot_widget.addItem(scatter)

        # Ghost points (if Show History is checked).
        if ghost_mask is not None and self._history_cb.isChecked():
            self._add_ghost_scatter(x_all, y_all, ghost_mask)

        if self._density_btn.isChecked():
            self._draw_density_contours(x, y)
        self._plot_widget.setLabel(
            "bottom", "PC 1", **{"color": "#999", "font-size": "10px"},
        )
        self._plot_widget.setLabel(
            "left", "PC 2", **{"color": "#999", "font-size": "10px"},
        )

    # Phase A.5 (2026-05-04): UMAP background-compute machinery removed.
    # Previously deleted methods: _on_umap_pca_toggled, _cancel_umap_thread,
    # _cleanup_orphan, _start_umap_compute, _cleanup_umap_refs,
    # _on_umap_finished, _on_umap_error.  See the comment above
    # `class QCPlotWidget` for the new contract: this widget is a passive
    # renderer for embeddings precomputed by Step 3 (CSV columns) or
    # Step 4 (controller-side EDA Compute UMAP / Compute t-SNE buttons).

    def _hydrate_embedding_caches(self) -> None:
        """Populate ``_umap_cache``/``_tsne_cache`` from persisted columns.

        Step 3's QC Settings dialog has opt-in "Compute UMAP" /
        "Compute t-SNE" checkboxes; when checked, the extraction worker
        writes ``UMAP_1``/``UMAP_2`` and/or ``tSNE_1``/``tSNE_2``
        columns into the per-patch CSV.  Loading that CSV brings the
        columns back into ``self._df``; this helper turns them into
        the same ``(n_rows, 2)`` ndarray shape that Step 4's controller
        compute path produces, so the existing ``_draw_umap``/
        ``_draw_tsne`` rendering paths work unchanged.

        Called from ``set_features``, ``update_data``, and
        ``set_pca_features`` AFTER the in-memory caches are reset.
        Quietly does nothing when ``self._df`` is ``None``/empty or the
        columns are absent (i.e. the user did not opt in to embedding
        computation in the dialog).
        """
        df = self._df
        if df is None or df.empty:
            return
        if {"UMAP_1", "UMAP_2"}.issubset(df.columns):
            self._umap_cache = df[["UMAP_1", "UMAP_2"]].values.astype(
                np.float64,
            )
        if {"tSNE_1", "tSNE_2"}.issubset(df.columns):
            self._tsne_cache = df[["tSNE_1", "tSNE_2"]].values.astype(
                np.float64,
            )
        # Phase E (2026-05-04) — hydrate PCA cache from persisted
        # PC1/PC2 columns so the PCA plot mode renders without
        # re-fitting.  Step 3 writes these via compute_pca_columns
        # at extract time; Step 4 CSVs don't contain them and this
        # block silently skips, leaving _get_pca_projection's
        # fallback fresh-fit path intact for Step 4.
        if {"PC1", "PC2"}.issubset(df.columns):
            self._pca_cache = df[["PC1", "PC2"]].values.astype(
                np.float64,
            )

    def _draw_umap(self):
        # Phase A.5: when cache is None (user did not opt in to UMAP
        # in the dialog AND Step 4 has not run Compute UMAP), show a
        # passive placeholder.  No background thread is started here.
        if self._umap_cache is None:
            if self._layout_mode == "workbench":
                self._wb_draw_empty_map("UMAP")
                return
            text_item = pg.TextItem(
                "Compute UMAP from the QC Settings dialog\n"
                "(or the Embedding tab in Step 4)",
                color=(160, 170, 190), anchor=(0.5, 0.5),
            )
            text_item.setFont(pg.QtGui.QFont("sans-serif", 12))
            self._plot_widget.addItem(text_item)
            text_item.setPos(0, 0)
            self._plot_widget.setXRange(-1, 1)
            self._plot_widget.setYRange(-1, 1)
            return

        projected = self._umap_cache
        if projected.shape[1] < 2:
            return

        x_all = projected[:, 0]
        y_all = projected[:, 1]

        # Separate ghost from active points.
        ghost_mask = self._get_ghost_mask()
        if ghost_mask is not None:
            active_mask = ~ghost_mask
            x = x_all[active_mask]
            y = y_all[active_mask]
            active_df = self._df[active_mask]
        else:
            x, y, active_df = x_all, y_all, self._df

        n = len(x)
        self._scatter_label_ids = active_df["label"].values.tolist()
        self._scatter_to_df_index = list(active_df.index)

        colors = self._get_point_colors_for(active_df)
        colors = self._safe_align_colors(colors, n)
        cat_mask = self._get_selected_category_mask_for(active_df, n)
        self._apply_selection_dimming(colors, n, cat_mask)

        scatter = pg.ScatterPlotItem(
            x=x, y=y,
            size=self._point_sizes_with_cat(n, cat_mask),
            brush=self._colors_to_brushes(colors),
            pen=self._point_pens_with_cat(n, cat_mask),
        )
        self._plot_widget.addItem(scatter)

        # Ghost points (if Show History is checked).
        if ghost_mask is not None and self._history_cb.isChecked():
            self._add_ghost_scatter(x_all, y_all, ghost_mask)

        if self._density_btn.isChecked():
            self._draw_density_contours(x, y)
        # Phase A.5: "(from PCA)" suffix removed alongside the
        # _umap_from_pca_cb checkbox.  The PCA pre-reduction now
        # always happens at compute time (dialog or Step 4
        # controller) and is no longer toggleable from the plot.
        self._plot_widget.setLabel(
            "bottom", "UMAP 1", **{"color": "#999", "font-size": "10px"},
        )
        self._plot_widget.setLabel(
            "left", "UMAP 2", **{"color": "#999", "font-size": "10px"},
        )

    def _draw_tsne(self):
        """Draw t-SNE scatter plot from cached projection."""
        if self._tsne_cache is None:
            if self._layout_mode == "workbench":
                self._wb_draw_empty_map("t-SNE")
                return
            text_item = pg.TextItem(
                "Compute t-SNE from the QC Settings dialog\n"
                "(or the Embedding tab in Step 4)",
                color=(160, 170, 190), anchor=(0.5, 0.5),
            )
            text_item.setFont(pg.QtGui.QFont("sans-serif", 12))
            self._plot_widget.addItem(text_item)
            text_item.setPos(0, 0)
            self._plot_widget.setXRange(-1, 1)
            self._plot_widget.setYRange(-1, 1)
            return

        projected = self._tsne_cache
        if projected.shape[1] < 2:
            return

        x_all = projected[:, 0]
        y_all = projected[:, 1]

        ghost_mask = self._get_ghost_mask()
        if ghost_mask is not None:
            active_mask = ~ghost_mask
            x = x_all[active_mask]
            y = y_all[active_mask]
            active_df = self._df[active_mask]
        else:
            x, y, active_df = x_all, y_all, self._df

        n = len(x)
        self._scatter_label_ids = active_df["label"].values.tolist()
        self._scatter_to_df_index = list(active_df.index)

        colors = self._get_point_colors_for(active_df)
        colors = self._safe_align_colors(colors, n)
        cat_mask = self._get_selected_category_mask_for(active_df, n)
        self._apply_selection_dimming(colors, n, cat_mask)

        scatter = pg.ScatterPlotItem(
            x=x, y=y,
            size=self._point_sizes_with_cat(n, cat_mask),
            brush=self._colors_to_brushes(colors),
            pen=self._point_pens_with_cat(n, cat_mask),
        )
        self._plot_widget.addItem(scatter)

        if ghost_mask is not None and self._history_cb.isChecked():
            self._add_ghost_scatter(x_all, y_all, ghost_mask)

        if self._density_btn.isChecked():
            self._draw_density_contours(x, y)
        self._plot_widget.setLabel(
            "bottom", "t-SNE 1", **{"color": "#999", "font-size": "10px"},
        )
        self._plot_widget.setLabel(
            "left", "t-SNE 2", **{"color": "#999", "font-size": "10px"},
        )

    def _draw_violin(self):
        """Draw per-category violin plots for the selected feature."""
        if self._wb_violin_on():
            self._wb_draw_violin()
            return
        from scipy.stats import gaussian_kde

        feature = self._feature_x_combo.currentText()
        if not feature or feature not in self._df.columns:
            return

        color_by = (
            self._color_combo.currentText()
            if self._color_combo.count() > 0
            else ""
        )

        cat_col, cat_colors = self._get_categorical_info(
            color_by if color_by in self._CATEGORICAL_COLOR_BY else "QC status"
        )
        if cat_col is None or cat_col not in self._df.columns:
            return

        categories = list(cat_colors.keys())
        grouped = self._df.groupby(cat_col)

        # Determine selected categories for highlighting
        sel_cats = set()
        for sc_col, sc_val in self._selected_categories:
            if sc_col == cat_col:
                sel_cats.add(sc_val)

        for i, cat_value in enumerate(categories):
            color_rgb = cat_colors.get(cat_value, (140, 150, 170))
            r, g, b = int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])

            try:
                group = grouped.get_group(cat_value)
            except KeyError:
                continue

            vals = group[feature].dropna().values.astype(np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue

            is_sel = bool(sel_cats) and str(cat_value) in sel_cats
            fill_alpha = 140 if is_sel else 70
            edge_alpha = 220 if is_sel else 140

            # KDE for violin shape
            try:
                kde = gaussian_kde(vals)
            except (np.linalg.LinAlgError, ValueError):
                continue

            y_grid = np.linspace(vals.min(), vals.max(), 60)
            density = kde(y_grid)
            # Normalize density to a max half-width of 0.35
            max_d = density.max()
            if max_d > 0:
                density = density / max_d * 0.35

            # Draw mirrored violin shape as filled polygon
            x_left = i - density
            x_right = i + density
            # Build polygon: left side (bottom→top) + right side (top→bottom)
            poly_x = np.concatenate([x_left, x_right[::-1]])
            poly_y = np.concatenate([y_grid, y_grid[::-1]])

            fill = pg.FillBetweenItem(
                pg.PlotCurveItem(x=x_left, y=y_grid),
                pg.PlotCurveItem(x=x_right, y=y_grid),
                brush=pg.mkBrush(r, g, b, fill_alpha),
            )
            self._plot_widget.addItem(fill)

            # Violin outline
            self._plot_widget.addItem(pg.PlotCurveItem(
                x=x_left, y=y_grid,
                pen=pg.mkPen(color=(r, g, b, edge_alpha), width=1.2),
            ))
            self._plot_widget.addItem(pg.PlotCurveItem(
                x=x_right, y=y_grid,
                pen=pg.mkPen(color=(r, g, b, edge_alpha), width=1.2),
            ))

            # Box overlay: Q1–Q3 + median
            q1 = np.percentile(vals, 25)
            median = np.percentile(vals, 50)
            q3 = np.percentile(vals, 75)
            box_w = 0.08

            # IQR box
            box = pg.BarGraphItem(
                x0=[i - box_w], x1=[i + box_w],
                y0=[q1], height=[q3 - q1],
                brush=pg.mkBrush(r, g, b, 180),
                pen=pg.mkPen(color=(255, 255, 255, 100), width=1),
            )
            self._plot_widget.addItem(box)

            # Median line
            self._plot_widget.addItem(pg.PlotCurveItem(
                x=[i - box_w * 1.5, i + box_w * 1.5],
                y=[median, median],
                pen=pg.mkPen(color=(255, 255, 255, 220), width=2),
            ))

        # X axis labels
        labels = []
        for cat_value in categories:
            if cat_col == "cluster_id":
                labels.append("noise" if cat_value == -1 else f"C{cat_value}")
            else:
                labels.append(str(cat_value))

        x_axis = self._plot_widget.getAxis("bottom")
        x_axis.setTicks([[(i, labels[i]) for i in range(len(labels))]])
        self._plot_widget.setLabel(
            "left", feature, **{"color": "#999", "font-size": "10px"},
        )
        self._plot_widget.setLabel(
            "bottom", cat_col.replace("_", " "), **{"color": "#999", "font-size": "10px"},
        )

    def _draw_density_contours(self, x: np.ndarray, y: np.ndarray):
        """Overlay 2D KDE density contours on the current scatter/PCA plot."""
        from scipy.stats import gaussian_kde

        # Need enough finite points for KDE
        mask = np.isfinite(x) & np.isfinite(y)
        xf, yf = x[mask], y[mask]
        if len(xf) < 6:
            return

        try:
            kde = gaussian_kde(np.vstack([xf, yf]))
        except (np.linalg.LinAlgError, ValueError):
            return  # singular covariance — skip contours

        # Evaluate KDE on a grid
        n_grid = 80
        pad = 0.05
        x_range = xf.max() - xf.min()
        y_range = yf.max() - yf.min()
        if x_range < 1e-12 or y_range < 1e-12:
            return
        xg = np.linspace(xf.min() - pad * x_range, xf.max() + pad * x_range, n_grid)
        yg = np.linspace(yf.min() - pad * y_range, yf.max() + pad * y_range, n_grid)
        xx, yy = np.meshgrid(xg, yg)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(n_grid, n_grid)

        # Contour levels — percentiles of density values
        d_sorted = np.sort(density.ravel())
        levels = [
            d_sorted[int(len(d_sorted) * p)]
            for p in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
        ]
        # Remove duplicate levels
        levels = sorted(set(levels))

        # Draw contours using skimage (unambiguous row/col convention)
        from skimage.measure import find_contours

        dx = (xg[-1] - xg[0]) / (n_grid - 1)
        dy = (yg[-1] - yg[0]) / (n_grid - 1)
        alphas = np.linspace(40, 150, len(levels)).astype(int)
        for i, level in enumerate(levels):
            contours = find_contours(density, level)
            pen = pg.mkPen(color=(200, 210, 230, int(alphas[i])), width=1.0)
            for contour in contours:
                if len(contour) < 2:
                    continue
                # find_contours returns (row, col) — row maps to Y, col maps to X
                data_x = xg[0] + contour[:, 1] * dx
                data_y = yg[0] + contour[:, 0] * dy
                curve = pg.PlotCurveItem(x=data_x, y=data_y, pen=pen)
                self._plot_widget.addItem(curve)

    def _update_stats(self):
        if self._df is None or self._df.empty:
            self._stats_label.setText("")
            return
        # Preserve selection count message if points are selected
        if self._selected_indices:
            return
        ghost_mask = self._get_ghost_mask()
        n_total = len(self._df)
        n_ghost = int(ghost_mask.sum()) if ghost_mask is not None else 0
        n = n_total - n_ghost
        ghost_str = f"  +{n_ghost} ghost" if n_ghost else ""
        if "qc_status" in self._df.columns:
            active_df = self._df[~ghost_mask] if ghost_mask is not None else self._df
            passed = int((active_df["qc_status"] == "pass").sum())
            flagged = int((active_df["qc_status"] == "flagged").sum())
            self._stats_label.setText(
                f"{n} labels  |  {passed} pass  |  {flagged} flagged{ghost_str}"
            )
        else:
            self._stats_label.setText(f"{n} labels{ghost_str}")

    def _wb_legend_html(self, color_by: str):
        """Workbench categorical legend: "{colour by}:" then one entry per
        category with its count (the mockup); groups read G1, G2..."""
        cat_col, cat_colors = self._get_categorical_info(
            color_by if color_by else "QC status")
        if cat_col is None:
            return ""
        scheme = self._scheme
        if cat_col == "qc_tier":
            values = self._wb_buckets(self._df)
        elif cat_col in self._df.columns:
            values = self._df[cat_col].to_numpy()
        else:
            values = np.array([])
        active = self._wb_active_mask()
        if len(values) == len(active):
            values = values[active]        # removed cells count nowhere
        items = ['<span style="color:#888888;">{}:</span>'.format(
            color_by or "QC status")]
        for cat_value, (r, g, b) in cat_colors.items():
            if cat_col == "cluster_id":
                display = (self._wb_word("NOISE_WORD", "noise")
                           if cat_value == -1 else
                           self._wb_word("GROUP_WORD", "G{n}").format(
                               n=int(cat_value) + 1, id=int(cat_value)))
            elif cat_col == "qc_tier" and cat_value == scheme.unassigned:
                display = self._wb_word("UNMARKED_WORD",
                                        scheme.unassigned_word)
            elif cat_col == "is_border_cell":
                display = {0: "Interior", 1: "Border"}.get(cat_value,
                                                           str(cat_value))
            else:
                display = str(cat_value)
            try:
                count = int((values == cat_value).sum())
            except Exception:  # noqa: BLE001 -- a count, never a crash
                count = 0
            if (cat_col == "qc_tier" and cat_value == scheme.unassigned
                    and count == 0 and not scheme.show_empty_unassigned):
                continue
            is_sel = (cat_col, str(cat_value)) in self._selected_categories
            weight = "bold" if is_sel else "normal"
            deco = "underline" if is_sel else "none"
            if scheme.colour_words:
                items.append(
                    '<a href="cat://{col}/{val}" style="color:rgb({r},{g},{b});'
                    ' font-weight:{w}; text-decoration:{d};">\u25cf&nbsp;{disp}'
                    '</a>&nbsp;<span style="color:#888888;">{n}</span>'.format(
                        col=cat_col, val=cat_value, r=int(r), g=int(g),
                        b=int(b), w=weight, d=deco, disp=display, n=count))
            else:
                # The dot in the entry's colour, the word neutral, both
                # inside the one link so a click on either picks it.
                items.append(
                    '<a href="cat://{col}/{val}" style="color:#cccccc;'
                    ' font-weight:{w}; text-decoration:{d};">'
                    '<span style="color:rgb({r},{g},{b});">\u25cf</span>'
                    '&nbsp;{disp}</a>&nbsp;<span style="color:#888888;">'
                    '{n}</span>'.format(
                        col=cat_col, val=cat_value, r=int(r), g=int(g),
                        b=int(b), w=weight, d=deco, disp=display, n=count))
        ghosts = self._wb_ghost_legend_entry()
        if ghosts:
            items.append(ghosts)
        return " &nbsp; ".join(items)

    # --- The Violin view under ``TierScheme.violin_active_groups`` --------

    def _wb_violin_on(self) -> bool:
        """The Step 3 violin (the scheme's ``violin_active_groups``)."""
        return (self._layout_mode == "workbench"
                and bool(getattr(self._scheme, "violin_active_groups",
                                 False)))

    def _wb_violin_by(self) -> str:
        """The Colour by word the violins group by: the Colour by itself
        when it is a categorical whose column is in the table, else
        "QC tier" (a feature, or "QC status" with no ``qc_status``, used
        to group by nothing and draw an empty plot)."""
        colour = (self._color_combo.currentText()
                  if self._color_combo.count() > 0 else "")
        if (self._df is not None and colour in self._CATEGORICAL_COLOR_BY
                and colour != "QC tier"):
            col, cats = self._get_categorical_info(colour)
            if col is not None and col in self._df.columns and cats:
                return colour
        return "QC tier"

    def _wb_category_word(self, col, value) -> str:
        """A category's on-screen word -- the legend's (``value`` may be
        the column's own value or its ``str``, as a pick stores it)."""
        scheme = self._scheme
        if col == "cluster_id":
            try:
                cid = int(float(value))
            except (TypeError, ValueError):
                return str(value)
            if cid == -1:
                return self._wb_word("NOISE_WORD", "noise")
            return self._wb_word("GROUP_WORD", "G{n}").format(n=cid + 1,
                                                              id=cid)
        if col == "qc_tier" and str(value) == scheme.unassigned:
            return self._wb_word("UNMARKED_WORD", scheme.unassigned_word)
        if col == "is_border_cell":
            v = str(value)
            if v in ("1", "1.0", "True"):
                return "Border"
            if v in ("0", "0.0", "False"):
                return "Interior"
        return str(value)

    def _wb_violin_groups(self):
        """``(cat_col, [(value, word, rgb, mask), ...])`` -- one group per
        category that holds at least one ACTIVE label (ghosts are no
        group), in legend order.  The drawing and the click read it, so a
        click lands on the group it shows."""
        if self._df is None or self._df.empty:
            return None, []
        import pandas as pd

        cat_col, cat_colors = self._get_categorical_info(self._wb_violin_by())
        if cat_col is None:
            return None, []
        if cat_col == "qc_tier":
            keys = pd.Series(self._wb_buckets(self._df), index=self._df.index)
        elif cat_col in self._df.columns:
            keys = self._df[cat_col]
        else:
            return cat_col, []
        active = self._wb_active_mask()
        groups = []
        for value, rgb in cat_colors.items():
            try:
                eq = keys == value
                if hasattr(eq, "fillna"):
                    eq = eq.fillna(False)
                m = np.asarray(eq, dtype=bool) & active
            except Exception:  # noqa: BLE001 -- a group, never a crash
                continue
            if m.any():
                groups.append((value, self._wb_category_word(cat_col, value),
                               rgb, m))
        return cat_col, groups

    def _wb_draw_violin(self):
        """The Step 3 violin: :meth:`_wb_violin_groups`' groups, drawn the
        way the classic violin draws one (KDE outline, IQR box, median);
        a group with one value (or no spread) is its median line."""
        from scipy.stats import gaussian_kde
        import pandas as pd

        self._wb_violin_drawn = set()
        feature = self._feature_x_combo.currentText()
        if not feature or feature not in self._df.columns:
            return
        cat_col, groups = self._wb_violin_groups()
        values_all = pd.to_numeric(self._df[feature], errors="coerce"
                                   ).to_numpy(dtype=np.float64)
        sel_cats = {v for c, v in self._selected_categories if c == cat_col}
        drawn = 0
        for i, (value, _word, rgb, mask) in enumerate(groups):
            vals = values_all[mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            drawn += 1
            self._wb_violin_drawn.add(i)
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            is_sel = bool(sel_cats) and str(value) in sel_cats
            fill_alpha = 140 if is_sel else 70
            edge_alpha = 220 if is_sel else 140
            box_w = 0.08
            kde = None
            if len(vals) >= 2 and float(np.ptp(vals)) > 0:
                try:
                    kde = gaussian_kde(vals)
                except (np.linalg.LinAlgError, ValueError):
                    kde = None
            if kde is not None:
                y_grid = np.linspace(vals.min(), vals.max(), 60)
                density = kde(y_grid)
                max_d = density.max()
                if max_d > 0:
                    density = density / max_d * 0.35
                x_left = i - density
                x_right = i + density
                self._plot_widget.addItem(pg.FillBetweenItem(
                    pg.PlotCurveItem(x=x_left, y=y_grid),
                    pg.PlotCurveItem(x=x_right, y=y_grid),
                    brush=pg.mkBrush(r, g, b, fill_alpha),
                ))
                for xs in (x_left, x_right):
                    self._plot_widget.addItem(pg.PlotCurveItem(
                        x=xs, y=y_grid,
                        pen=pg.mkPen(color=(r, g, b, edge_alpha), width=1.2),
                    ))
                q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
                self._plot_widget.addItem(pg.BarGraphItem(
                    x0=[i - box_w], x1=[i + box_w], y0=[q1], height=[q3 - q1],
                    brush=pg.mkBrush(r, g, b, 180),
                    pen=pg.mkPen(color=(255, 255, 255, 100), width=1),
                ))
            median = float(np.median(vals))
            self._plot_widget.addItem(pg.PlotCurveItem(
                x=[i - box_w * 1.5, i + box_w * 1.5], y=[median, median],
                pen=pg.mkPen(color=(255, 255, 255, 220) if kde is not None
                             else (r, g, b, 255), width=2),
            ))
        if drawn == 0:
            sentence = self._wb_word("VIOLIN_NO_VALUES", "")
            if sentence:
                self._wb_draw_sentence(sentence.format(feat=feature))
            return
        self._plot_widget.getAxis("bottom").setTicks(
            [[(i, grp[1]) for i, grp in enumerate(groups)]])
        self._plot_widget.setLabel(
            "left", feature, **{"color": "#999", "font-size": "10px"})
        # No bottom title: it overprinted the middle group's word, and the
        # axes note already reads "{feature} by {groups}".
        self._plot_widget.setLabel("bottom", "")

    def violin_groups_drawn(self) -> set:
        """Workbench: the positions of the violins the last Step 3 violin
        drawing put on screen (a group whose feature values are all NaN
        draws nothing and takes no click); empty off the violin."""
        if self.plot_type_combo.currentText() != "Violin":
            return set()
        return set(getattr(self, "_wb_violin_drawn", set()))

    def picked_category_words(self) -> List[str]:
        """Workbench: the words of the categories picked now (a legend or
        a violin pick), as the legend writes them; ``[]`` for none or a
        selection kept across a plot-type switch."""
        self._require_workbench("picked_category_words")
        return [self._wb_category_word(col, val)
                for col, val in sorted(self._selected_categories, key=str)
                if (col, val) != _WB_KEPT]

    def _wb_plot_draws_ghosts(self) -> bool:
        """Whether the current plot type draws ghost points: Scatter and
        PCA do, a UMAP / t-SNE only once its map exists, a Violin never."""
        pt = self.plot_type_combo.currentText()
        if pt in ("Scatter", "PCA"):
            return True
        cache = {"UMAP": self._umap_cache, "t-SNE": self._tsne_cache}.get(pt)
        return (cache is not None and getattr(cache, "ndim", 0) == 2
                and cache.shape[1] >= 2)

    def _wb_ghost_legend_entry(self) -> str:
        """The legend's ghosts entry -- a grey "×", the word and the count,
        PLAIN text (ghosts cannot be selected, so no ``cat://`` link).
        Only for a scheme with ``ghost_entry``, a ``GHOSTS_LEGEND`` word,
        the ghosts shown and at least one ghost."""
        if not self._scheme.ghost_entry:
            return ""
        word = self._wb_word("GHOSTS_LEGEND", None)
        if (not word or not self._history_cb.isChecked()
                or not self._wb_plot_draws_ghosts()):
            return ""
        mask = self._get_ghost_mask()
        n = int(mask.sum()) if mask is not None else 0
        if n == 0:
            return ""
        r, g, b, _a = self._GHOST_COLOR
        return ('<span style="color:rgb({r},{g},{b});">\u00d7&nbsp;{w}'
                '</span>&nbsp;<span style="color:#888888;">{n}</span>'
                ).format(r=r, g=g, b=b, w=word, n=n)

    def _update_legend(self):
        color_by = (
            self._color_combo.currentText()
            if self._color_combo.count() > 0
            else ""
        )
        if (self._wb_violin_on() and self._df is not None
                and self.plot_type_combo.currentText() == "Violin"):
            self._legend_label.setText(
                self._wb_legend_html(self._wb_violin_by()))
            self._colorbar_label.setText("")
            return
        if (self._layout_mode == "workbench" and self._df is not None
                and (color_by in self._CATEGORICAL_COLOR_BY or not color_by)):
            self._legend_label.setText(self._wb_legend_html(color_by))
            self._colorbar_label.setText("")
            return

        # Categorical modes — render clickable <a href> links
        if color_by in self._CATEGORICAL_COLOR_BY or not color_by:
            cat_col, cat_colors = self._get_categorical_info(
                color_by if color_by else "QC status"
            )
            if cat_col is None:
                self._legend_label.setText("")
                return

            items = []
            for cat_value, color_rgb in cat_colors.items():
                r = int(color_rgb[0])
                g = int(color_rgb[1])
                b = int(color_rgb[2])

                # Display name mapping for human-readable legend labels
                _DISPLAY_NAMES = {
                    "is_border_cell": {0: "Interior", 1: "Border"},
                    # euler_category already has string values — no mapping needed
                }
                if cat_col == "cluster_id":
                    display = "noise" if cat_value == -1 else f"C{cat_value}"
                elif cat_col in _DISPLAY_NAMES:
                    display = _DISPLAY_NAMES[cat_col].get(cat_value, str(cat_value))
                else:
                    display = str(cat_value)

                # Bold + underline for all selected categories
                is_sel = (cat_col, str(cat_value)) in self._selected_categories
                weight = "bold" if is_sel else "normal"
                deco = "underline" if is_sel else "none"

                items.append(
                    f'<a href="cat://{cat_col}/{cat_value}" '
                    f'style="color:rgb({r},{g},{b}); font-weight:{weight}; '
                    f'text-decoration:{deco};">'
                    f'\u25cf&nbsp;{display}</a>'
                )
            self._legend_label.setText(" &nbsp; ".join(items))
            self._colorbar_label.setText("")
        else:
            # Continuous colormap legend using active colormap
            stops = self._active_cmap_stops()
            blocks = [
                f'<span style="color:rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})">'
                f'\u2588</span>'
                for c in stops
            ]
            self._legend_label.setText(
                f"{''.join(blocks)}  {color_by}"
            )

    def _on_tier_line_moved(self, _line):
        """Emit tier thresholds when any tier line is dragged."""
        vals = tuple(line.value() for line in self._tier_lines)
        self.tier_thresholds_changed.emit(vals)

    def _on_tier_line_released(self, _line):
        """Emit final thresholds on mouse release — triggers viewer color sync."""
        vals = tuple(line.value() for line in self._tier_lines)
        self.tier_drag_finished.emit(vals)

    def _on_mouse_clicked(self, event):
        """Handle click on scatter/PCA (nearest point) or histogram bar."""
        if self._df is None:
            return
        # In lasso select mode, the event filter handles everything
        if self._select_btn.isChecked() and not (
                self._wb_violin_on()
                and self.plot_type_combo.currentText() == "Violin"):
            return

        pos = event.scenePos()
        vb = self._plot_widget.plotItem.vb
        if not self._plot_widget.sceneBoundingRect().contains(pos):
            return

        mouse_point = vb.mapSceneToView(pos)
        mx, my = mouse_point.x(), mouse_point.y()

        plot_type = self.plot_type_combo.currentText()

        # Violin mode — click nearest violin to toggle category
        if plot_type == "Violin":
            self._handle_violin_click(mx)
            return

        # Scatter / PCA — click nearest point
        if not self._scatter_label_ids:
            return

        x_data, y_data = self._get_scatter_coords()
        if x_data is None:
            return

        # Normalize distances by axis range
        x_range = x_data.max() - x_data.min() if x_data.max() > x_data.min() else 1.0
        y_range = y_data.max() - y_data.min() if y_data.max() > y_data.min() else 1.0
        dist = ((x_data - mx) / x_range) ** 2 + ((y_data - my) / y_range) ** 2
        nearest_idx = int(np.argmin(dist))
        label_id = self._scatter_label_ids[nearest_idx]

        # Store the clicked row index for get_selected_df_indices().
        self._clicked_df_index = nearest_idx

        # Clear any active lasso isolation first, then navigate
        if self._selected_indices:
            self._selected_indices.clear()
            self._clear_sel_btn.setVisible(False)
            self.labels_selected.emit([])  # clear controller isolation
        self.labels_selected.emit([int(label_id)])

    def _handle_violin_click(self, mx):
        """Identify nearest violin by x-position and toggle that category."""
        if self._wb_violin_on():
            cat_col, groups = self._wb_violin_groups()
            i = int(np.floor(float(mx) + 0.5))
            if (not groups or not 0 <= i < len(groups)
                    or i not in self.violin_groups_drawn()):
                return     # beside every violin, or a group drawn empty
            self._toggle_category_selection(cat_col, str(groups[i][0]))
            return
        color_by = (
            self._color_combo.currentText()
            if self._color_combo.count() > 0
            else ""
        )
        cat_col, cat_colors = self._get_categorical_info(
            color_by if color_by in self._CATEGORICAL_COLOR_BY else "QC status"
        )
        if cat_col is None:
            return
        categories = list(cat_colors.keys())
        if not categories:
            return
        nearest_idx = int(round(mx))
        nearest_idx = max(0, min(nearest_idx, len(categories) - 1))
        self._toggle_category_selection(cat_col, str(categories[nearest_idx]))

"""Real-time training loss plot widget using pyqtgraph.

Supports three display modes:
- Overlay: all prior rounds as faded lines, current round bold
- Current: only the latest round
- Per-round: dropdown selector to view a specific round

Live updates during training via ``append_live_point()``.

Round grading (Step 2 look pass, ``grade_by_round``; off by default, so Step 3
and the old Step 2 look are untouched).  With the flag on:

* ids are read as ``{family}_{n}`` (``lateral_0``, ``axial_2``, ``foo/axial``);
  the live id ``round_NN`` carries no family and takes ``set_live_family()``'s;
* the legend uses the disk numbering: ``lat r0 . lat r1 . lat r3 (live) . ax r0``;
* hue = the family's (lateral accent, axial ``_PAL_AXIAL_RGB``); inside a
  family rounds are graded by AGE (0 = newest, or the live one)::

      age   alpha   width  style
      0     100 %   2.0    solid   (bold in the legend)
      1      70 %   1.4    solid
      2      55 %   1.4    dashed
      3      40 %   1.4    dotted
      4+     30 %   1.0    solid   (faint)

  so any number of rounds (1..7 and more) reads; the table is ``GRADE_TABLE``;
* Overlay draws BOTH families; the family that is not the bold one (the live
  family, else ``set_view_filter``'s, else the newest stored round's) is drawn
  at 30 % alpha throughout.  Current round / Per round keep the view filter;
* the Overlay / Current round / Per round combo stays visible;
* when idle the y axis is fitted to the TAIL: the first ``_TAIL_SKIP_FRAC`` of
  every drawn curve is ignored when choosing the range (early epochs clip at
  the top); while a round is live the axis auto-ranges.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

# --- Palette ------------------------------------------------------------------
_ROUND_COLORS = [
    (100, 180, 255),   # soft blue
    (130, 220, 160),   # soft green
    (255, 170, 100),   # soft orange
    (200, 140, 255),   # soft purple
    (255, 130, 150),   # soft red
    (100, 220, 220),   # soft teal
    (220, 200, 100),   # soft gold
    (180, 180, 180),   # gray
]

_LIVE_COLOR = (80, 200, 255)        # bright cyan for the live curve
_LIVE_PEN_WIDTH = 2.0
_HISTORY_PEN_WIDTH = 1.3
_HISTORY_ALPHA = 100                # 0-255 transparency for prior rounds
_BG_COLOR = "#1a1a1a"
_GRID_ALPHA = 30

# Palette look (Step 3 Model Refinement; matches the QC plot's workbench look).
_PAL_BG_COLOR = "#23242b"
_PAL_GRID_ALPHA = 28                # the owner keeps the gridlines visible
_PAL_AXIS_PEN = (50, 50, 60)
_PAL_TEXT_PEN = (120, 120, 130)
_PAL_VAL_ALPHA = 153                # 60 % of 255
_PAL_LATERAL_WIDTH = 1.6
_PAL_AXIAL_WIDTH = 1.4
#: the second hue: the lateral model draws in the accent, the axial one here
_PAL_AXIAL_RGB = (159, 180, 216)      # #9fb4d8
#: the combo's look, the QC plot's (11 px, bg .04, border .10, radius 3)
_PAL_COMBO_STYLE = (
    "QComboBox {"
    "  font-size: 11px; padding: 2px 6px; min-height: 20px;"
    "  background: rgba(255,255,255,0.04); color: #dddddd;"
    "  border: 1px solid rgba(255,255,255,0.10); border-radius: 3px;"
    "}"
    "QComboBox:disabled { color: #666666;"
    " background: rgba(255,255,255,0.02); }"
    "QComboBox QAbstractItemView {"
    "  background: #2b2d35; color: #dddddd; }"
)
_DEFAULT_COMBO_STYLE = "QComboBox { font-size: 10px; padding: 1px 4px; }"

# --- Round grading (Step 2 look) ----------------------------------------------
#: (alpha 0-255, width px, "solid" | "dash" | "dot") for age 0, 1, 2, 3, 4+
GRADE_TABLE = (
    (255, 2.0, "solid"),
    (179, 1.4, "solid"),
    (140, 1.4, "dash"),
    (102, 1.4, "dot"),
    (77, 1.0, "solid"),
)
_OTHER_FAMILY_ALPHA = 77            # 30 % of 255, Overlay only
_TAIL_SKIP_FRAC = 0.10              # leading share of each curve ignored by the y fit
_TAIL_PAD = 0.06                    # y padding around the tail range
_FAMILY_WORDS = {"lateral": "lat", "axial": "ax"}
_FAMILY_ORDER = ("lateral", "axial")


def grade_for_age(age: int) -> Tuple[int, float, str]:
    """``(alpha, width, style)`` for a round ``age`` rounds behind the newest."""
    return GRADE_TABLE[max(0, min(int(age), len(GRADE_TABLE) - 1))]


def family_of(round_id: str) -> Optional[str]:
    """``"lateral"`` / ``"axial"`` from the id (a ``/axial`` suffix wins),
    ``None`` for an id with no family (the live ``round_NN``)."""
    base, _, suffix = str(round_id).partition("/")
    if suffix in _FAMILY_WORDS:
        return suffix
    for fam in _FAMILY_WORDS:
        if base.startswith(fam + "_"):
            return fam
    return None


def round_number(round_id: str) -> Optional[int]:
    """The disk round number in the id (``lateral_3`` -> 3, ``round_03`` -> 3)."""
    m = re.search(r"(\d+)$", str(round_id).partition("/")[0])
    return int(m.group(1)) if m else None


def display_round(n: int) -> int:
    """The number shown on screen: the disk's (a pass-through)."""
    return n


def tail_y_range(curves, skip: float = _TAIL_SKIP_FRAC):
    """``(lo, hi)`` over the tails of ``curves`` (first ``skip`` share of each
    ignored), padded; ``None`` when there is nothing finite."""
    vals = []
    for c in curves:
        a = np.asarray(list(c), dtype=float)
        a = a[int(len(a) * skip):]
        a = a[np.isfinite(a)]
        if a.size:
            vals.append((float(a.min()), float(a.max())))
    if not vals:
        return None
    lo = min(v[0] for v in vals)
    hi = max(v[1] for v in vals)
    pad = (hi - lo) * _TAIL_PAD or 0.05 * max(abs(hi), 1e-6)
    return lo - pad, hi + pad


class LossPlotWidget(QWidget):
    """Embeddable pyqtgraph-based loss curve viewer."""

    MODES = ["Overlay", "Current Round", "Per Round"]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # State ---
        self._round_curves: Dict[str, List[float]] = {}
        self._round_order: List[str] = []
        self._live_x: List[float] = []
        self._live_y: List[float] = []
        self._live_round_id: Optional[str] = None
        self._view_filter: Optional[str] = None  # "lateral" or "axial" or None (show all)
        self._palette: Optional[dict] = None
        self._round_vals: Dict[str, List[tuple]] = {}
        self._live_vx: List[float] = []
        self._live_vy: List[float] = []
        self._live_n_epochs: Optional[int] = None
        self.grade_by_round = False
        self._live_family: Optional[str] = None
        self._tail_src: List[List[float]] = []

        # Layout ---
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # --- Top bar: mode + round selector + stats ---
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(2, 2, 2, 0)
        top_bar.setSpacing(4)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.MODES)
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.setToolTip("Loss plot display mode")
        self.mode_combo.setMaximumWidth(110)
        self.mode_combo.setStyleSheet(
            "QComboBox { font-size: 10px; padding: 1px 4px; }"
        )
        self.mode_combo.currentIndexChanged.connect(self._redraw)

        self._round_selector = QComboBox()
        self._round_selector.setToolTip("Select round to view")
        self._round_selector.setMaximumWidth(90)
        self._round_selector.setStyleSheet(
            "QComboBox { font-size: 10px; padding: 1px 4px; }"
        )
        self._round_selector.setVisible(False)
        self._round_selector.currentIndexChanged.connect(self._redraw)

        self._stats_label = QLabel("No training data")
        self._stats_label.setStyleSheet("color: #888; font-size: 10px;")
        self._stats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top_bar.addWidget(self.mode_combo)
        top_bar.addWidget(self._round_selector)
        top_bar.addStretch(1)
        top_bar.addWidget(self._stats_label)
        root.addLayout(top_bar)

        # --- Plot area (stacked: empty state overlay + actual plot) ---
        plot_container = QWidget()
        self._stack = QStackedLayout(plot_container)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.setStackingMode(QStackedLayout.StackAll)

        # Empty state placeholder
        self._empty_label = QLabel("Loss curves will appear here\nduring training")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            f"background: {_BG_COLOR}; color: #555; font-size: 11px;"
            "border-radius: 4px;"
        )

        # pyqtgraph plot
        self._plot_widget = pg.PlotWidget()
        plot_item = self._plot_widget.plotItem
        # Remove the default title
        plot_item.setTitle(None)

        # Crosshair
        self._vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color=(255, 255, 255, 30), width=1, style=Qt.DashLine),
        )
        self._plot_widget.addItem(self._vline, ignoreBounds=True)
        self._vline.setVisible(False)
        self._plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._stack.addWidget(self._plot_widget)
        self._stack.addWidget(self._empty_label)

        root.addWidget(plot_container, 1)

        # --- Bottom legend ---
        self._legend_label = QLabel("")
        self._legend_label.setStyleSheet(
            "color: #777; font-size: 9px; padding: 1px 3px 0px 3px;"
        )
        self._legend_label.setWordWrap(True)
        self._legend_label.setMaximumHeight(18)
        root.addWidget(self._legend_label)

        # Initial state
        self._apply_style(None)
        self._show_empty(True)

    # --- Styling --------------------------------------------------------------

    def _apply_style(self, palette: Optional[dict]):
        """Apply the default (Step 2) look, or the palette look."""
        pw = self._plot_widget
        plot_item = pw.plotItem
        if palette is None:
            pw.setBackground(_BG_COLOR)
            pw.showGrid(x=True, y=True, alpha=_GRID_ALPHA / 255)
            plot_item.setContentsMargins(0, 0, 0, 0)
            plot_item.getViewBox().setDefaultPadding(0.02)
            for axis_name in ("bottom", "left"):
                ax = pw.getAxis(axis_name)
                ax.setTextPen(pg.mkPen(color=(140, 140, 140), width=1))
                ax.setPen(pg.mkPen(color=(60, 60, 60), width=1))
                ax.setStyle(
                    tickLength=-4,
                    tickTextOffset=3,
                    autoExpandTextSpace=False,
                    autoReduceTextSpace=True,
                )
                ax.setHeight(20) if axis_name == "bottom" else ax.setWidth(35)
            pw.setLabel("bottom", "")
            pw.setLabel("left", "")
            self._empty_label.setStyleSheet(
                f"background: {_BG_COLOR}; color: #555; font-size: 11px;"
                "border-radius: 4px;"
            )
            self._legend_label.setStyleSheet(
                "color: #777; font-size: 9px; padding: 1px 3px 0px 3px;"
            )
            self.mode_combo.setStyleSheet(_DEFAULT_COMBO_STYLE)
            self._round_selector.setStyleSheet(_DEFAULT_COMBO_STYLE)
            return
        from qtpy.QtGui import QFont
        self.mode_combo.setStyleSheet(_PAL_COMBO_STYLE)
        self._round_selector.setStyleSheet(_PAL_COMBO_STYLE)
        pw.setBackground(_PAL_BG_COLOR)
        pw.showGrid(x=True, y=True, alpha=_PAL_GRID_ALPHA / 255)
        plot_item.setContentsMargins(4, 4, 4, 4)
        plot_item.getViewBox().setDefaultPadding(0.04)
        font = QFont()
        font.setPixelSize(11)
        for axis_name in ("bottom", "left"):
            ax = pw.getAxis(axis_name)
            ax.setTextPen(pg.mkPen(color=_PAL_TEXT_PEN, width=1))
            ax.setPen(pg.mkPen(color=_PAL_AXIS_PEN, width=1))
            ax.setTickFont(font)
            ax.setStyle(
                tickLength=-5,
                tickTextOffset=4,
                autoExpandTextSpace=True,
                autoReduceTextSpace=True,
            )
            if axis_name == "bottom":
                ax.setHeight(22)
            else:
                ax.setWidth(40)
        # no axis titles (M11): the legend says what the lines are
        pw.setLabel("bottom", "")
        pw.setLabel("left", "")
        self._empty_label.setStyleSheet(
            f"background: {_PAL_BG_COLOR}; color: #555; font-size: 11px;"
            "border-radius: 4px;"
        )
        self._legend_label.setStyleSheet(
            "color: #999999; font-size: 9px; padding: 1px 3px 0px 3px;"
        )

    #: the palette (Step 3) look's words for MODES[1:]; Step 2 keeps its own
    _PAL_MODE_WORDS = ("Current model", "Per model")

    def _mode(self) -> str:
        """The canonical mode name (the combo's SHOWN words differ under a
        palette)."""
        i = self.mode_combo.currentIndex()
        return self.MODES[i] if 0 <= i < len(self.MODES) else self.MODES[0]

    def _set_mode_words(self, palette: bool) -> None:
        words = (self._PAL_MODE_WORDS if palette else tuple(self.MODES[1:]))
        for i, w in enumerate(words, start=1):
            self.mode_combo.setItemText(i, w)

    def set_palette(self, palette: Optional[dict]):
        """Adopt a ``plot_palette`` mapping (``accent``, ``accent_rgb``,
        ``ok_text``).  ``None`` restores the default (Step 2) look."""
        self._palette = dict(palette) if palette else None
        self._set_mode_words(bool(self._palette) and not self.grade_by_round)
        # Step 3 has one behaviour (Overlay): no mode / round combo
        self.mode_combo.setVisible(not self._palette or self.grade_by_round)
        if self._palette and not self.grade_by_round:
            self._round_selector.setVisible(False)
        self._apply_style(self._palette)
        self._redraw()

    def _accent(self):
        pal = getattr(self, "_palette", None)
        if not pal:
            return _LIVE_COLOR
        rgb = pal.get("accent_rgb")
        if rgb is None:
            h = str(pal["accent"]).lstrip("#")
            rgb = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        return tuple(int(v) for v in rgb)

    def _vals(self) -> Dict[str, List[tuple]]:
        # __dict__ access: tolerates a bare ``__new__`` instance (tests).
        return self.__dict__.setdefault("_round_vals", {})

    def _round_color(self, round_id: str, fallback_index: int):
        """Palette: one hue per MODEL -- the lateral model in the accent,
        the axial model in the second hue -- whatever the bundle (an older
        bundle only fades).  No palette: today's per-index cycle."""
        if self.__dict__.get("grade_by_round"):
            return (_PAL_AXIAL_RGB if self._family(round_id) == "axial"
                    else self._accent())
        if not getattr(self, "_palette", None):
            return _ROUND_COLORS[fallback_index % len(_ROUND_COLORS)]
        if str(round_id).endswith("/axial"):
            return _PAL_AXIAL_RGB
        return self._accent()

    def _newest_bundle(self) -> Optional[str]:
        """The bundle (the id's prefix) drawn at full strength: the live
        one, else the newest stored."""
        if self._live_round_id:
            return str(self._live_round_id).split("/", 1)[0]
        order = self._filtered_rounds()
        return order[-1].split("/", 1)[0] if order else None

    def _history_width(self, round_id: str) -> float:
        if getattr(self, "_palette", None):
            if round_id.endswith("/axial"):
                return _PAL_AXIAL_WIDTH
            return _PAL_LATERAL_WIDTH
        return _HISTORY_PEN_WIDTH

    # --- Public API -----------------------------------------------------------

    def set_display_mode(self, mode: str):
        """Set display mode programmatically."""
        idx = self.MODES.index(mode) if mode in self.MODES else 0
        self.mode_combo.setCurrentIndex(idx)

    def set_grade_by_round(self, on: bool = True):
        """Switch the Step 2 round-age grading on / off (module docstring)."""
        self.grade_by_round = bool(on)
        self.set_palette(self._palette)      # combo visibility, words, style

    def set_stats_hidden(self, hidden: bool = True):
        """Hide the plot's own stats line (Step 2's Loss header already
        carries the same caption, so it would be said twice)."""
        self._stats_label.setVisible(not hidden)

    def set_live_family(self, bucket: Optional[str]):
        """The family (``"lateral"`` / ``"axial"``) of the live ``round_NN``
        curve, which carries none in its id.  Cleared when the round ends."""
        self._live_family = bucket if bucket in _FAMILY_WORDS else None
        self._redraw()

    def set_idle_family(self, bucket: Optional[str]):
        """The family drawn bold (and named in the header caption) while no
        round is live: the bucket the next Train writes (Step 2's Writes
        line), not whichever family stored the newest round.  ``None``
        returns to the newest stored round's family."""
        bucket = bucket if bucket in _FAMILY_WORDS else None
        if bucket == self.__dict__.get("_idle_family"):
            return
        self._idle_family = bucket
        self._redraw()

    def _family(self, round_id: str) -> str:
        """The id's family; an id without one (the live ``round_NN``) takes
        ``set_live_family``'s, else the view filter's, else lateral."""
        fam = family_of(round_id)
        if fam:
            return fam
        live = self.__dict__.get("_live_family")
        return live or self.__dict__.get("_view_filter") or "lateral"

    def _bold_family(self) -> str:
        """The family drawn at full strength in Overlay."""
        if self._live_round_id:
            return self._family(self._live_round_id)
        if self._view_filter:
            return self._view_filter
        idle = self.__dict__.get("_idle_family")
        if idle:
            return idle
        if self._round_order:
            return self._family(self._round_order[-1])
        return "lateral"

    def _label(self, round_id: str, live: bool = False) -> str:
        """``lat r3`` / ``ax r0`` (+ `` (live)``); a ``/view`` suffix is dropped."""
        n = round_number(round_id)
        word = _FAMILY_WORDS[self._family(round_id)]
        out = word if n is None else f"{word} r{display_round(n)}"
        return out + (" (live)" if live else "")

    def _ages(self) -> Dict[str, int]:
        """Age per stored id inside its family (0 = newest round number; with
        a live curve every stored round of its family is one older)."""
        ages: Dict[str, int] = {}
        live_fam = (self._family(self._live_round_id)
                    if self._live_round_id else None)
        for fam in _FAMILY_ORDER:
            ids = [r for r in self._round_order if self._family(r) == fam]
            nums = sorted({round_number(r) or 0 for r in ids}, reverse=True)
            shift = 1 if fam == live_fam else 0
            for r in ids:
                ages[r] = nums.index(round_number(r) or 0) + shift
        return ages

    def _overlay_rounds(self) -> List[str]:
        """Overlay's ids: both families under grading, else the filtered ones."""
        return (list(self._round_order) if self.grade_by_round
                else self._filtered_rounds())

    def grading_caption(self) -> str:
        """``last: lat r3 . train 0.62 . 5 rounds`` for the Loss header."""
        ids = self._overlay_rounds()
        if not ids:
            return ""
        # the caption names the bold family's newest round (its own line
        # is the one drawn bold), else the newest stored overall
        bold = self._bold_family()
        mine = [r for r in ids if self._family(r) == bold]
        last = mine[-1] if mine else ids[-1]
        losses = self._round_curves.get(last) or []
        n = len({(self._family(r), round_number(r)) for r in ids})
        out = f"last: {self._label(last)}"
        if losses:
            out += f" \u00b7 train {losses[-1]:.2f}"
        return out + f" \u00b7 {n} round" + ("" if n == 1 else "s")

    def set_view_filter(self, bucket: Optional[str]):
        """Filter displayed curves by view bucket ('lateral' or 'axial').

        Pass None to show all curves (no filter).
        """
        self._view_filter = bucket
        self._sync_round_selector()
        self._redraw()

    def _filtered_rounds(self) -> List[str]:
        """Return round IDs matching the active view filter."""
        if not self._view_filter:
            return list(self._round_order)
        prefix = self._view_filter + "_"
        return [r for r in self._round_order if r.startswith(prefix)]

    def add_round_curve(self, round_id: str, losses: List[float],
                        val_losses: Optional[List[tuple]] = None):
        """Store a completed round's loss curve and redraw.

        ``val_losses`` is an optional sparse list of ``(epoch, val)`` pairs.
        """
        self._round_curves[round_id] = list(losses)
        if val_losses:
            self._vals()[round_id] = [(float(e), float(v)) for e, v in val_losses]
        else:
            self._vals().pop(round_id, None)
        if round_id not in self._round_order:
            self._round_order.append(round_id)
        self._sync_round_selector()
        self._show_empty(False)
        self._redraw()

    def begin_live_round(self, round_id: str,
                         n_epochs: Optional[int] = None):
        """Start a new live training session.  ``n_epochs`` (palette look)
        fixes the x axis to 0..n_epochs while it runs, so the curve grows
        into a stable frame."""
        self._live_n_epochs = int(n_epochs) if n_epochs else None
        self._live_round_id = round_id
        self._live_x = []
        self._live_y = []
        self._live_vx = []
        self._live_vy = []
        self._show_empty(False)
        self._redraw()
        self._apply_live_range()

    def append_live_point(self, epoch: int, loss: float,
                          val_loss: Optional[float] = None):
        """Add a single epoch data point during live training.

        ``val_loss`` is appended only when not ``None`` (sampled epochs).
        """
        self._live_x.append(float(epoch))
        self._live_y.append(loss)
        if val_loss is not None:
            self._live_vx.append(float(epoch))
            self._live_vy.append(float(val_loss))
        self._update_live_curve()
        self._update_stats()
        if self._palette:
            self._update_legend()

    def finish_live_round(self):
        """Clear live state without promoting to stored curve.

        The caller should follow up with ``add_round_curve()`` using the
        complete loss array from the backend (which may differ from the
        sampled points streamed during training).
        """
        self._live_round_id = None
        self._live_x = []
        self._live_y = []
        self._live_vx = []
        self._live_vy = []
        self._live_plot_item = None
        self._live_family = None
        self._release_live_range()
        self._redraw()

    def cancel_live_round(self):
        """Discard an in-progress live round (e.g. on error)."""
        self._live_round_id = None
        self._live_x = []
        self._live_y = []
        self._live_vx = []
        self._live_vy = []
        self._live_plot_item = None
        self._live_family = None
        self._release_live_range()
        self._redraw()

    def clear_all(self):
        """Reset all stored curves."""
        self._round_curves.clear()
        self._round_order.clear()
        self._vals().clear()
        self._live_x = []
        self._live_y = []
        self._live_vx = []
        self._live_vy = []
        self._live_round_id = None
        self._live_family = None
        self._sync_round_selector()
        self._show_empty(True)
        self._redraw()

    def load_round_history(self, losses_by_round: Dict[str, Any]):
        """Bulk-load prior round losses (e.g. from saved state).

        Accepts either:
        - {round_id: [loss_values]} (single view)
        - {round_id: {view_type: [loss_values]}} (multi-view — loads all views)
        """
        self._round_curves.clear()
        self._round_order.clear()
        self._vals().clear()
        for round_id, entry in losses_by_round.items():
            # Keys in `losses_by_round` are bucket-keyed (`"lateral_N"` /
            # `"axial_N"`) since the per-bucket state write in the
            # controller.  Use them verbatim so `_filtered_rounds` (prefix
            # match on `"lateral_"` / `"axial_"`) matches both live-trained
            # and restored curves.  Pre-2026-05-26 this prepended a
            # `"round_"` prefix that silently broke the view filter after
            # any switch-and-reload cycle.
            key = str(round_id)
            if isinstance(entry, dict):
                # Multi-view: add each view as a separate curve.
                for view_type, losses in entry.items():
                    if isinstance(losses, (list, np.ndarray)) and len(losses) > 0:
                        label = f"{key}/{view_type}"
                        self._round_curves[label] = [float(v) for v in losses]
                        self._round_order.append(label)
            else:
                losses = entry
                if isinstance(losses, (list, np.ndarray)) and len(losses) > 0:
                    self._round_curves[key] = [float(v) for v in losses]
                    self._round_order.append(key)
        self._sync_round_selector()
        has_data = bool(self._round_order)
        self._show_empty(not has_data)
        self._redraw()

    # --- Internal drawing -----------------------------------------------------

    _palette = None
    grade_by_round = False
    _live_family = None
    _idle_family = None
    _live_plot_item = None
    _live_val_item = None
    _live_end_item = None

    def _apply_live_range(self):
        """While a palette-look run is live, x is 0..n_epochs (fixed)."""
        n = getattr(self, "_live_n_epochs", None)
        if not (n and getattr(self, "_palette", None)):
            return
        vb = self._plot_widget.plotItem.getViewBox()
        vb.enableAutoRange(axis="x", enable=False)
        vb.setXRange(0, n, padding=0.03)

    def _release_live_range(self):
        self._live_n_epochs = None
        try:
            self._plot_widget.plotItem.getViewBox().enableAutoRange(
                axis="x", enable=True)
        except Exception:      # noqa: BLE001
            pass

    def _show_empty(self, empty: bool):
        """Toggle between empty placeholder and live plot."""
        if empty:
            self._stack.setCurrentWidget(self._empty_label)
        else:
            self._stack.setCurrentWidget(self._plot_widget)

    def _sync_round_selector(self):
        self._round_selector.blockSignals(True)
        self._round_selector.clear()
        for rid in self._filtered_rounds():
            self._round_selector.addItem(
                self._label(rid) if self.grade_by_round else rid)
        filtered = self._filtered_rounds()
        if filtered:
            self._round_selector.setCurrentIndex(len(filtered) - 1)
        self._round_selector.blockSignals(False)

    def _redraw(self):
        """Full redraw based on current mode."""
        self._plot_widget.clear()
        self._plot_widget.addItem(self._vline, ignoreBounds=True)
        self._live_plot_item = None
        self._live_val_item = None
        self._live_end_item = None

        mode = self._mode()
        self._tail_src = []
        self._round_selector.setVisible(
            mode == "Per Round"
            and (self.grade_by_round or not getattr(self, "_palette", None)))

        if mode == "Overlay":
            self._draw_overlay()
        elif mode == "Current Round":
            self._draw_current()
        elif mode == "Per Round":
            self._draw_per_round()

        if self._live_round_id and self._live_x:
            self._add_live_curve_item()

        self._apply_tail_fit()
        self._update_stats()
        self._update_legend()

    def _apply_tail_fit(self):
        """Grading look, idle: y fitted to the tail of what is drawn."""
        if not self.grade_by_round:
            return
        vb = self._plot_widget.plotItem.getViewBox()
        rng = None if self._live_round_id else tail_y_range(self._tail_src)
        if rng is None:
            vb.enableAutoRange(axis="y", enable=True)
            return
        vb.enableAutoRange(axis="y", enable=False)
        vb.setYRange(rng[0], rng[1], padding=0)

    def _graded_pen(self, round_id: str, age: int, bold_fam: str):
        alpha, width, style = grade_for_age(age)
        if self._family(round_id) != bold_fam:
            alpha = _OTHER_FAMILY_ALPHA
        qs = {"solid": Qt.SolidLine, "dash": Qt.DashLine,
              "dot": Qt.DotLine}[style]
        return pg.mkPen(color=(*self._round_color(round_id, 0), alpha),
                        width=width, style=qs), alpha

    def _draw_overlay(self):
        if self.grade_by_round:
            ages = self._ages()
            bold = self._bold_family()
            for rid in self._round_order:
                losses = self._round_curves[rid]
                pen, alpha = self._graded_pen(rid, ages.get(rid, 0), bold)
                self._plot_widget.plot(np.arange(len(losses)), losses,
                                       pen=pen, name=rid)
                self._draw_round_val(rid, self._round_color(rid, 0), alpha)
                self._tail_src.append(losses)
            return
        filtered = self._filtered_rounds()
        newest = self._newest_bundle() if self._palette else None
        for i, round_id in enumerate(filtered):
            losses = self._round_curves[round_id]
            color = self._round_color(round_id, i)
            # the newest bundle draws at full strength, only OLDER fade (M4)
            full = (newest is not None
                    and round_id.split("/", 1)[0] == newest)
            alpha = 255 if full else _HISTORY_ALPHA
            pen = pg.mkPen(color=(*color, alpha),
                           width=self._history_width(round_id))
            x = np.arange(len(losses))
            self._plot_widget.plot(x, losses, pen=pen, name=round_id)
            self._draw_round_val(round_id, color, alpha)

    def _draw_current(self):
        filtered = self._filtered_rounds()
        if filtered:
            latest = filtered[-1]
            losses = self._round_curves[latest]
            color = self._round_color(latest, len(filtered) - 1)
            pen = pg.mkPen(color=(*color, 220), width=_LIVE_PEN_WIDTH)
            x = np.arange(len(losses))
            self._plot_widget.plot(x, losses, pen=pen)
            self._draw_round_val(latest, color, 220)
            self._tail_src.append(losses)

    def _draw_per_round(self):
        filtered = self._filtered_rounds()
        idx = self._round_selector.currentIndex()
        if 0 <= idx < len(filtered):
            round_id = filtered[idx]
            losses = self._round_curves[round_id]
            color = self._round_color(round_id, idx)
            pen = pg.mkPen(color=(*color, 220), width=_LIVE_PEN_WIDTH)
            x = np.arange(len(losses))
            self._plot_widget.plot(x, losses, pen=pen)
            self._draw_round_val(round_id, color, 220)
            self._tail_src.append(losses)

    def _draw_round_val(self, round_id: str, color, alpha: int):
        """Dashed polyline + a marker per sample, only when val was given."""
        vals = self._vals().get(round_id)
        if not vals:
            return
        vx = np.array([e for e, _ in vals])
        vy = np.array([v for _, v in vals])
        a = int(alpha * 0.6) if alpha >= 200 else alpha
        pen = pg.mkPen(color=(*color, a), width=1.3, style=Qt.DashLine)
        self._plot_widget.plot(
            vx, vy, pen=pen, symbol="o", symbolSize=4,
            symbolPen=None, symbolBrush=(*color, a),
        )

    def _live_color(self):
        if getattr(self, "_palette", None) and self._live_round_id:
            return self._round_color(self._live_round_id, 0)
        return self._accent()

    def _add_live_curve_item(self):
        pen = pg.mkPen(color=self._live_color(), width=_LIVE_PEN_WIDTH)
        x = np.array(self._live_x)
        y = np.array(self._live_y)
        self._live_plot_item = self._plot_widget.plot(x, y, pen=pen)
        if getattr(self, "_palette", None):
            c = self._live_color()
            self._live_end_item = self._plot_widget.plot(
                x[-1:], y[-1:], pen=None, symbol="o", symbolSize=7,
                symbolPen=None, symbolBrush=c,
            )
            self._live_val_item = self._plot_widget.plot(
                np.array(self._live_vx), np.array(self._live_vy),
                pen=pg.mkPen(color=(*c, _PAL_VAL_ALPHA), width=_LIVE_PEN_WIDTH,
                             style=Qt.DashLine),
            )

    def _update_live_curve(self):
        if self._live_plot_item is not None:
            self._live_plot_item.setData(
                np.array(self._live_x), np.array(self._live_y)
            )
            if self._live_end_item is not None:
                self._live_end_item.setData(
                    np.array(self._live_x[-1:]), np.array(self._live_y[-1:])
                )
            if self._live_val_item is not None:
                self._live_val_item.setData(
                    np.array(self._live_vx), np.array(self._live_vy)
                )
        else:
            self._add_live_curve_item()

    def _palette_stats(self) -> str:
        """Panel words (M5): "{bundle} · lateral · live" while it runs,
        "{bundle} · both models" once finished -- no round(s), no ids."""
        if self._live_round_id and self._live_y:
            bundle, _, bucket = str(self._live_round_id).partition("/")
            words = [bundle] + ([bucket] if bucket else []) + ["live"]
            return " \u00b7 ".join(words)
        newest = self._newest_bundle()
        if newest is None:
            return "No training data"
        buckets = {r.partition("/")[2] for r in self._filtered_rounds()
                   if r.split("/", 1)[0] == newest}
        if {"lateral", "axial"} <= buckets:
            what = "both models"
        elif "axial" in buckets:
            what = "axial model"
        elif "lateral" in buckets:
            what = "lateral model"
        else:
            return newest
        return f"{newest} \u00b7 {what}"

    def _update_stats(self):
        if self.grade_by_round:
            self._stats_label.setText(
                self.grading_caption() or "No training data")
            return
        if getattr(self, "_palette", None):
            self._stats_label.setText(self._palette_stats())
            return
        filtered = self._filtered_rounds()
        parts = []
        if self._live_round_id and self._live_y:
            epoch = int(self._live_x[-1]) if self._live_x else 0
            loss = self._live_y[-1]
            parts.append(f"Epoch {epoch} | loss: {loss:.4f}")
        elif filtered:
            latest = filtered[-1]
            losses = self._round_curves[latest]
            if losses:
                parts.append(f"{latest}: {losses[-1]:.4f}")
        if filtered:
            parts.append(f"{len(filtered)} round(s)")
        self._stats_label.setText(" \u00b7 ".join(parts) if parts else "No training data")

    def _palette_models(self) -> list:
        """``[(rgb, bucket, train, val)]`` of the newest bundle: its stored
        passes, then the live one."""
        newest = self._newest_bundle()
        out = []
        if newest is None:
            return out
        live = self._live_round_id if (self._live_round_id
                                       and self._live_y) else None
        for rid in self._filtered_rounds():
            if rid.split("/", 1)[0] != newest or rid == live:
                continue
            losses = self._round_curves.get(rid) or []
            if not losses:
                continue
            v = self._vals().get(rid)
            out.append((self._round_color(rid, len(out)),
                        rid.partition("/")[2], losses[-1],
                        v[-1][1] if v else None))
        if live and str(live).split("/", 1)[0] == newest:
            out.append((self._live_color(), str(live).partition("/")[2],
                        self._live_y[-1],
                        self._live_vy[-1] if self._live_vy else None))
        return out

    def _palette_legend(self) -> str:
        """One model: dot train 0.41 / dashes validation 0.44 (last
        values).  Two models: one entry each, "— lateral 0.41 / 0.71 ·
        — axial 0.38 / 0.63 · - - validation", each dash in its
        model's colour."""
        models = self._palette_models()

        def num(x):
            return f'<span style="color:#cccccc">{x:.2f}</span>'

        if len(models) >= 2 and len({m[1] for m in models}) >= 2:
            parts = []
            for c, bucket, tr, va in models:
                col = f"rgb({c[0]},{c[1]},{c[2]})"
                val = "" if va is None else f" / {num(va)}"
                parts.append(f'<span style="color:{col}">—</span> '
                             f'{bucket} {num(tr)}{val}')
            parts.append('<span style="color:#999999">- -</span> validation')
            return " · ".join(parts)
        if self._live_round_id and self._live_y:
            c = self._live_color()
            tr = self._live_y[-1]
            va = self._live_vy[-1] if self._live_vy else None
        else:
            filtered = self._filtered_rounds()
            if not filtered:
                return ""
            rid = filtered[-1]
            c = self._round_color(rid, len(filtered) - 1)
            losses = self._round_curves.get(rid) or []
            if not losses:
                return ""
            tr = losses[-1]
            v = self._vals().get(rid)
            va = v[-1][1] if v else None
        col = f"rgb({c[0]},{c[1]},{c[2]})"
        out = f'<span style="color:{col}">—</span> train {num(tr)}'
        if va is not None:
            out += (f' · <span style="color:{col}">- -</span> '
                    f'validation {num(va)}')
        return out

    def _graded_legend(self) -> str:
        """Overlay: one entry per round (swatch in its grade, the bold line's
        label in bold); Current round / Per round: the one drawn."""
        bg = (35, 36, 43)
        glyph = {"solid": "\u2501", "dash": "\u254c\u254c",
                 "dot": "\u00b7\u00b7\u00b7"}

        def entry(rid, age, bold_fam, live=False):
            alpha, _w, style = grade_for_age(age)
            if self._family(rid) != bold_fam:
                alpha = _OTHER_FAMILY_ALPHA
            c = self._round_color(rid, 0)
            c = tuple(int(bg[i] + (c[i] - bg[i]) * alpha / 255)
                      for i in range(3))
            label = self._label(rid, live)
            if age == 0 and self._family(rid) == bold_fam:
                label = f"<b>{label}</b>"
            return (f'<span style="color:rgb({c[0]},{c[1]},{c[2]})">'
                    f'{glyph[style]}</span> {label}')

        bold = self._bold_family()
        items = []
        if self._mode() == "Overlay":
            ages = self._ages()
            order = sorted(self._round_order, key=lambda r: (
                _FAMILY_ORDER.index(self._family(r)), round_number(r) or 0))
            items = [entry(r, ages.get(r, 0), bold) for r in order]
            if self._live_round_id:
                items.append(entry(self._live_round_id, 0, bold, True))
        elif self._live_round_id:
            fam = self._family(self._live_round_id)
            items = [entry(self._live_round_id, 0, fam, True)]
        else:
            ids = self._filtered_rounds()
            idx = self._round_selector.currentIndex()
            if self._mode() == "Per Round" and 0 <= idx < len(ids):
                rid = ids[idx]
            else:
                rid = ids[-1] if ids else None
            if rid is not None:
                items = [entry(rid, 0, self._family(rid))]
        return "  ".join(items)

    def _update_legend(self):
        if self.grade_by_round:
            self._legend_label.setMaximumHeight(44)
            self._legend_label.setText(self._graded_legend())
            return
        if getattr(self, "_palette", None):
            self._legend_label.setText(self._palette_legend())
            return
        mode = self._mode()
        filtered = self._filtered_rounds()
        if mode == "Overlay" and (filtered or self._live_round_id):
            items = []
            for i, rid in enumerate(filtered):
                c = _ROUND_COLORS[i % len(_ROUND_COLORS)]
                items.append(
                    f'<span style="color:rgb({c[0]},{c[1]},{c[2]})">\u2501</span> {rid}'
                )
            if self._live_round_id:
                c = _LIVE_COLOR
                items.append(
                    f'<span style="color:rgb({c[0]},{c[1]},{c[2]})">\u2501</span> '
                    f'{self._live_round_id} (live)'
                )
            self._legend_label.setText("  ".join(items))
        elif self._live_round_id:
            c = _LIVE_COLOR
            self._legend_label.setText(
                f'<span style="color:rgb({c[0]},{c[1]},{c[2]})">\u2501</span> '
                f'{self._live_round_id} (live)'
            )
        else:
            self._legend_label.setText("")

    def _on_mouse_moved(self, pos):
        vb = self._plot_widget.plotItem.vb
        if self._plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            self._vline.setPos(mouse_point.x())
            self._vline.setVisible(True)
        else:
            self._vline.setVisible(False)

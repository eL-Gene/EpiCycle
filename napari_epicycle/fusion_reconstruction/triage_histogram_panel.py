"""The Step 3 triage histogram: tier-stacked bars + draggable tier lines.

A standalone widget -- it shares no code with ``qc_plot_widget.py``, so
Step 4's EDA Workbench is untouched.  ``Triage by`` picks the X-axis
feature; ``Colour bars by`` stacks the bars by a categorical column
(``qc_tier`` by default) or colours them by a continuous one.  With
``Triage by = qc_score`` three draggable tier lines re-sort the labels
that are not curated (``thresholds_changed`` while dragging,
``tier_drag_finished`` on release); on any other feature ``Gate`` draws
a band whose labels feed the same selection the lasso uses
(``gated_selection_changed`` per pixel, ``gated_drag_finished`` on
release).  The panel owns no Assign button: the curation panel's tier
combo + Assign stays the one commit path.

Nothing on screen comes from made-up data (Step 3 plot pass): with no DataFrame the panel shows ``HIST_EMPTY``, a feature with
no finite values shows ``HIST_NO_VALUES``, and a data push never emits a
gate signal (``set_data`` turns Gate off with ``untick_gate_silently``).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont, QTextOption
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..ui_shell.theme import (
    PANEL_BTN_STYLE,
    PANEL_CHECKBOX_STYLE,
    PANEL_HELP_STYLE,
    PANEL_INPUT_STYLE,
    PANEL_ROW_LABEL_STYLE,
)


# ---------------------------------------------------------------------------
# Words -- every sentence and tooltip the panel shows.  (The plot pass's
# brick P2 moves these into ``_qc_plot_text.py``.)
# ---------------------------------------------------------------------------

#: No DataFrame yet: no patch loaded, or a patch without QC features.
HIST_EMPTY = "No QC data yet."
#: A DataFrame whose chosen feature has no finite values.
HIST_NO_VALUES = "No values for {feature} on this patch."
#: The note line while Triage by is qc_score.
HIST_NOTE_DRAG = (
    "Dragging a line re-sorts every label that is not curated, "
    "hand assignments included."
)
#: The note line when every active label is curated.
HIST_NOTE_ALL_CURATED = "All {n} labels are curated; the lines change nothing."

#: The control words.
HIST_TRIAGE_BY = "Triage by"
HIST_COLOUR_BARS_BY = "Colour bars by"
HIST_RESET_LINES = "Reset lines"
HIST_GATE = "Gate"
#: The left axis title.
HIST_COUNT_AXIS = "count"
#: The one gradient entry of "Colour bars by": each bar coloured by its
#: position on the Triage-by axis.
HIST_VALUE_ITEM = "value"

#: One plain tooltip per control (keyed by the control's own word).
HIST_HELP = {
    HIST_TRIAGE_BY: (
        "The feature along the bottom of the histogram. With qc_score, "
        "three lines sort the labels into good, review, bad and garbage; "
        "with any other feature, tick Gate to select a range."
    ),
    HIST_COLOUR_BARS_BY: (
        "What the bars are coloured by. A tier or cluster column stacks "
        "the bars by group; \"value\" colours each bar by its position on "
        "the Triage by axis."
    ),
    HIST_RESET_LINES: (
        "Put the three tier lines back at the quartiles of qc_score on "
        "this patch, and re-sort the labels that are not curated."
    ),
    HIST_GATE: (
        "Select every label whose value lies inside a band. Drag the band "
        "or its edges; the selected labels can then be assigned to a tier."
    ),
    HIST_VALUE_ITEM: (
        "Colour each bar by its position on the Triage by axis."
    ),
}

# ---------------------------------------------------------------------------
# Look
# ---------------------------------------------------------------------------

#: The plot surface (the plugin dock's neutral surface, as the plot's).
HIST_BG = "#23242b"
#: The plot's floor (the lower area's height budget).
HIST_PLOT_MIN_H = 130
#: The one label size inside the plot (axis titles, ticks, empty text).
HIST_LABEL_PX = 11
#: The gate band takes the selection colour (the lasso's), not cyan --
#: cyan is the curated tier.
GATE_COLOUR = "#59b3ff"
GATE_RGB = (0x59, 0xB3, 0xFF)
#: The ghost stack's grey (the plot's ghost colour).
GHOST_GREY = (140, 140, 160)
#: The dark halo drawn under each tier line, so a green line still reads
#: on green bars: 1 px each side of the 2 px line.
HALO_RGB = (30, 30, 36)
HALO_WIDTH = 4
#: Wrap width of the empty-state sentence, in pixels.
_EMPTY_TEXT_WIDTH = 260


# Match qc_plot_widget._TIER_COLORS for visual consistency across the
# Step 3 UI.  Values are RGB 0-255 tuples; hex variants used for the
# tier pill style helper.
_TIER_COLORS_RGB = {
    "curated": (0, 188, 212),    # cyan/teal
    "good": (80, 200, 120),      # green
    "review": (255, 200, 60),    # yellow-orange
    "bad": (255, 120, 40),       # warm orange
    "garbage": (240, 60, 60),    # red
    # Catch-all for any tier value not in the canonical 5 (NaN,
    # "unassigned", "flagged", custom labels, etc.).  Without this,
    # untiered rows disappear from the stacked histogram, distorting
    # the visible shape — see the bug fix of 2026-05-04.
    "unassigned": (140, 150, 170),  # cool gray-blue
    # Committed / discarded garbage (label_status deleted / superseded).  Shown
    # as a grey "ghost" stack so the user can SEE what they removed on
    # Commit Triage — it stays until a re-extraction drops the ghost rows
    # (mirrors the scatter's ghost points).  2026-06-09.  Hidden with
    # ``set_ghosts_visible(False)``.
    "ghost": GHOST_GREY,
}

# Bottom-up stacking order in the histogram.  Drawing the highest-
# severity tier (garbage) first means it sits at the bottom and good/
# curated stack on top — visually similar to a tier "ladder".
# "unassigned" sits at the very bottom as a foundation.
# "ghost" (committed/discarded garbage) sits at the very bottom as a gray
# foundation; "unassigned" above it; then the active tiers.
_TIER_STACK_ORDER = ["ghost", "unassigned", "garbage", "bad", "review", "good", "curated"]

# Z-order for the persistent tier InfiniteLines + gate LinearRegionItem.
# Crash-fix follow-up (2026-06-08): the crash-fix made the tier lines +
# gate region persistent (added once, never removed on redraw) and
# switched _redraw to a selective bar-only clear.  That fix is leak-free
# + crash-safe, but it introduced a z-order regression: because the
# persistent lines are added ONCE (early, in __init__) while bars are
# re-addItem'd AFTER them on every redraw, the bars (default zValue 0,
# ~78% opaque brush) painted ON TOP of the tier lines + gate region —
# washing out the draggable lines.  Pre-crash-fix, _redraw did
# clear()+re-add-lines-LAST, which kept the lines on top by insertion
# order.  Setting an explicit zValue here restores that pre-fix render
# regardless of insertion order: bars stay at the default z=0, so
# z=10 lines/region always paint above them.
_LINE_Z = 10

# Shared colormap source (2026-05-04).
# Imported from qc_plot_widget so the left panel's gradient bars
# stay in sync with the right plot's _cmap_combo selection.  When
# the user changes the right combo, controller calls
# panel.set_active_cmap(name) which re-keys this gradient and
# triggers _redraw.
def _stops_for_cmap(name: str) -> list:
    """Return a 5-stop gradient list ``[(t, (r, g, b)), ...]`` for
    the named colormap.  Pulls from ``qc_plot_widget._COLORMAPS``
    (single source of truth, [0, 1] floats); converts to [0, 255]
    ints with evenly-spaced ``t`` values.

    Falls back to Coolwarm when ``name`` is unknown — matches the
    original hardcoded default.
    """
    from .qc_plot_widget import _COLORMAPS
    stops_arr = _COLORMAPS.get(name, _COLORMAPS["Coolwarm"])
    n = len(stops_arr)
    if n < 2:
        return [(0.0, (128, 128, 128))]
    return [
        (i / (n - 1), tuple(int(round(c * 255)) for c in stops_arr[i]))
        for i in range(n)
    ]


def _sample_gradient(t: float, stops: list) -> tuple:
    """Sample a gradient ``[(t_i, (r, g, b)), ...]`` at parameter
    ``t`` in [0, 1].  Linear interpolation between adjacent stops.
    """
    t = max(0.0, min(1.0, float(t)))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(int(c0[k] + f * (c1[k] - c0[k])) for k in range(3))
    return stops[-1][1]


class _ElidedLabel(QLabel):
    """One neutral line, elided on the right to the width it is given
    (re-elided on every resize); the full sentence is its tooltip."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setWordWrap(False)
        # Ignored horizontally: the line never widens the panel.
        self.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Fixed)

    def full_text(self) -> str:
        return self._full

    def set_full_text(self, text: str) -> None:
        self._full = text or ""
        self.setToolTip(self._full)
        self._re_elide()

    def _re_elide(self) -> None:
        if not self._full:
            QLabel.setText(self, "")
            return
        self.ensurePolished()
        width = max(0, self.contentsRect().width())
        QLabel.setText(self, self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideRight, width))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._re_elide()


class TriageHistogramPanel(QWidget):
    """Tier-stacked histogram + draggable thresholds for triage.

    Public API: ``set_data(df)`` (``None`` = the empty state),
    ``set_active_cmap``, ``set_ghosts_visible``, ``untick_gate_silently``,
    ``reset_tier_lines``, ``get_thresholds``, ``empty_sentence`` and the
    five signals below.
    """

    # Emitted continuously while the user drags any tier line (Phase 2C).
    # Payload is the sorted (t1, t2, t3) tuple of all 3 line positions.
    # Mirrors qc_plot_widget.tier_thresholds_changed exactly.
    thresholds_changed = Signal(tuple)

    # Emitted ONCE on drag-release so the controller can do the
    # heavier viewer color sync that's skipped during continuous drag.
    # Mirrors qc_plot_widget.tier_drag_finished exactly.
    tier_drag_finished = Signal(tuple)

    # Declared for a bin click; nothing emits it today.
    bin_clicked = Signal(int, int)  # (bin_index, label_count)

    # Phase F (2026-05-04) — multi-feature gating signals.
    # Two signals to keep continuous-drag cheap.
    #
    # gated_selection_changed: fires on EVERY drag pixel.  Controller
    # routes ONLY to the right plot's selection bucket + a lightweight
    # redraw (visual highlight of dimmed points).  No 3D viewer
    # colormap rebuild on this path — too expensive to fire per-pixel.
    #
    # gated_drag_finished: fires ONCE on drag release.  Controller
    # then runs the full _on_qc_plot_label_selected pipeline (3D
    # viewer DirectLabelColormap, navigator highlight, subviewer
    # mirror).
    #
    # This mirrors the qc_plot_widget pattern of
    # tier_thresholds_changed (continuous) + tier_drag_finished
    # (release) which solved the same lag problem for tier-line drags.
    gated_selection_changed = Signal(object)
    gated_drag_finished = Signal(object)

    # Categorical columns supported as Color by options.  Anything in
    # this set will draw a stacked histogram colored by the column's
    # values; other (numeric) columns draw a continuous gradient.
    # Column names match the canonical EpiCycle DataFrame schema —
    # see qc_plot_widget.py:1017-1027 for the right plot's mapping.
    _CATEGORICAL_COLOR_BY = (
        "qc_tier",
        "qc_status",
        "cluster_id",
        "is_border_cell",
        "euler_number",
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._df = None  # set by set_data; None = the empty state
        # The host's lock (Step 3 plot pass P2): every control greyed and
        # the lines / band frozen; ``_sync_controls`` honours it.
        self._locked = False
        # The ghost stack is drawn unless set_ghosts_visible(False).
        self._ghosts_visible = True
        # The sentence on screen in an empty state, else None.
        self._empty_sentence = None
        self._tier_lines: list = []
        # Crash-fix (2026-06-08): per-redraw histogram BAR items are
        # tracked here so _redraw can remove ONLY the bars and leave the
        # persistent tier InfiniteLines + gate region untouched on the
        # canvas.  Pre-fix _redraw called _plot_widget.clear() which
        # tore out EVERYTHING — including the tier lines — and then
        # re-addItem'd the same line objects.  When that redraw was
        # triggered SYNCHRONOUSLY from a tier line's sigPositionChanged
        # (i.e. mid-drag), removing + re-adding the line whose handler
        # was still on the stack re-entrantly mutated the pyqtgraph
        # scene → native Windows access violation.  Selective bar-only
        # removal keeps the lines continuously in the scene.
        self._bar_items: list = []
        # Phase 2D (2026-05-04) — gradient colormap shared with the
        # right plot's _cmap_combo.  Default Coolwarm matches
        # pre-Phase-2D behaviour; controller calls set_active_cmap
        # when the user switches the right combo.
        self._active_cmap = "Coolwarm"
        self._gradient_stops = _stops_for_cmap(self._active_cmap)
        self._build_ui()
        # Phase 2C: tier lines are persistent objects.  Created here
        # once; re-attached to the plot on each redraw.  Drag state +
        # signal connections survive across redraws naturally.
        self._create_tier_lines()
        # Phase F (2026-05-04): gate region (pg.LinearRegionItem) for
        # non-qc_score features.  Created persistent here, hidden by
        # default; addItem'd by _redraw whenever a non-qc_score
        # feature is selected.
        self._create_gate_region()
        self._redraw()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(PANEL_ROW_LABEL_STYLE)
        return label

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        # No margins: the host's height budget counts only the rows.
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # --- Combo row: Triage by + Colour bars by + Reset lines + Gate --
        combo_row = QHBoxLayout()
        combo_row.setSpacing(4)
        combo_row.addWidget(self._field_label(HIST_TRIAGE_BY))
        self._feature_combo = QComboBox()
        # Starts EMPTY: set_data fills it from the DataFrame's columns.
        self._feature_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self._feature_combo.setToolTip(HIST_HELP[HIST_TRIAGE_BY])
        combo_row.addWidget(self._feature_combo, 1)

        combo_row.addWidget(self._field_label(HIST_COLOUR_BARS_BY))
        self._color_combo = QComboBox()
        # Seeded with the default only; set_data fills the real choices
        # (the categorical columns present, then the one "value" entry)
        # and keeps qc_tier when it is there.
        self._color_combo.addItem("qc_tier")
        self._color_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self._color_combo.setToolTip(HIST_HELP[HIST_COLOUR_BARS_BY])
        combo_row.addWidget(self._color_combo, 1)

        # Reset lines and Gate share one slot (Rule A): Reset lines with
        # data on qc_score, Gate with data on any other feature.
        self._reset_btn = QPushButton(HIST_RESET_LINES)
        self._reset_btn.setStyleSheet(PANEL_BTN_STYLE)
        self._reset_btn.setToolTip(HIST_HELP[HIST_RESET_LINES])
        self._reset_btn.setVisible(False)  # no data yet
        combo_row.addWidget(self._reset_btn)

        # Gate (2026-05-04): shown only when there is data and the X-axis
        # feature is NOT qc_score (the tier lines handle that case).  When
        # ticked, a band brackets the histogram and the labels in
        # [low, high] populate the same selection bucket the lasso uses.
        self._gate_cb = QCheckBox(HIST_GATE)
        self._gate_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self._gate_cb.setToolTip(HIST_HELP[HIST_GATE])
        self._gate_cb.setVisible(False)  # no data yet
        combo_row.addWidget(self._gate_cb)
        lay.addLayout(combo_row)

        # --- Histogram plot ----------------------------------------
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumHeight(HIST_PLOT_MIN_H)
        self._plot_widget.setBackground(HIST_BG)
        self._plot_widget.showGrid(x=False, y=True, alpha=0.15)
        tick_font = QFont()
        tick_font.setPixelSize(HIST_LABEL_PX)
        for name in ("bottom", "left"):
            axis = self._plot_widget.getAxis(name)
            axis.setPen(pg.mkPen("#555"))
            axis.setTextPen(pg.mkPen("#999"))
            axis.setTickFont(tick_font)
        # Pan + zoom on X only — Y zoom isn't useful for histograms.
        self._plot_widget.setMouseEnabled(x=True, y=False)

        lay.addWidget(self._plot_widget, 1)

        # The empty-state sentence, drawn the way the plot draws its own
        # (centred, axes hidden).  Persistent; shown by _show_empty.
        self._empty_text = pg.TextItem(color=(170, 170, 170), anchor=(0.5, 0.5))
        empty_font = QFont()
        empty_font.setPixelSize(HIST_LABEL_PX)
        self._empty_text.setFont(empty_font)
        self._empty_text.setTextWidth(_EMPTY_TEXT_WIDTH)
        # Centre the words inside the wrap box, not only the box.
        option = QTextOption(Qt.AlignmentFlag.AlignHCenter)
        option.setWrapMode(QTextOption.WrapMode.WordWrap)
        self._empty_text.textItem.document().setDefaultTextOption(option)
        self._empty_text.setVisible(False)
        self._plot_widget.addItem(self._empty_text, ignoreBounds=True)

        # --- Note line --------------------------
        # What a tier-line drag does: one neutral help line, elided, the
        # full sentence as its tooltip; hidden when there is nothing to say.
        self._note_label = _ElidedLabel()
        self._note_label.setStyleSheet(PANEL_HELP_STYLE)
        self._note_label.setVisible(False)
        lay.addWidget(self._note_label)

        # Combo signal hookups — redraw on change.
        self._feature_combo.currentTextChanged.connect(self._redraw)
        self._feature_combo.currentTextChanged.connect(
            self._on_feature_changed_for_gate,
        )
        self._color_combo.currentTextChanged.connect(self._redraw)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        self._gate_cb.toggled.connect(self._on_gate_toggled)

    # ------------------------------------------------------------------
    # Redraw
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        """Re-render the histogram from the current combo state and df.

        With no DataFrame (or no feature to triage by) the panel shows
        ``HIST_EMPTY``; a feature with no finite values shows
        ``HIST_NO_VALUES``.  Nothing is ever drawn from made-up data.
        """
        feature = self._feature_combo.currentText()
        color_by = self._color_combo.currentText() or "qc_tier"

        if self._df is None or not feature or feature not in self._df.columns:
            self._show_empty(HIST_EMPTY)
            return
        all_values, all_tiers, _xr = self._values_from_df(feature)
        if all_values is None or len(all_values) == 0:
            self._show_empty(HIST_NO_VALUES.format(feature=feature))
            return
        if self._ghosts_visible:
            shown = np.ones(len(all_values), dtype=bool)
        else:
            # Ghosts hidden: the ghost rows leave every bar (not only the
            # grey stack), so the bars count the active labels alone.
            shown = all_tiers != "ghost"
        values, tiers = all_values[shown], all_tiers[shown]
        if len(values) == 0:
            self._show_empty(HIST_NO_VALUES.format(feature=feature))
            return
        self._hide_empty()

        # Match QCPlotWidget._draw_histogram exactly: bin count is
        # derived from data size, edges from np.histogram on the actual
        # values (no pre-clipping).  Pre-fix the panel used a hardcoded
        # 30 bins on linspace edges over a forced [0,1] range, producing
        # a different shape from the right plot for the same feature.
        #
        # Bins, edges and the x-range come from the FULL set (ghosts
        # included) and only the shown values are counted, so hiding the
        # ghosts is a pure subtraction of the grey stack and the tier
        # lines stay in view.
        n_bins = min(50, max(10, len(all_values) // 5))
        _counts_unused, edges = np.histogram(all_values, bins=n_bins)
        bin_idx = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
        # Refresh x_range from actual edges in case np.histogram chose
        # something different from min/max (it doesn't, but explicit).
        x_range = (float(edges[0]), float(edges[-1]))

        # Crash-fix (2026-06-08): remove ONLY the per-redraw bar items,
        # NOT the whole scene.  Pre-fix this was self._plot_widget.clear()
        # which removed the persistent tier InfiniteLines + gate region
        # too, forcing a re-addItem at the bottom of this method.  When
        # _redraw is reached synchronously from a tier line's
        # sigPositionChanged handler (mid-drag → thresholds_changed →
        # controller → _push_to_triage_histogram → set_data → _redraw),
        # removing + re-adding the dragged line while its own handler is
        # on the stack re-entrantly mutates the pyqtgraph scene graph →
        # native Windows access violation.  Selective bar removal leaves
        # the tier lines (and the gate region) continuously in the scene,
        # so they are NEVER torn out mid-drag.  Bars still rebuild every
        # tick so the live recolor the user values is preserved.
        self._clear_bars()

        if color_by == "qc_tier":
            # Stacked tier histogram — anything not in the canonical 5
            # tiers (incl. NaN, "flagged", custom labels) falls into
            # the "unassigned" gray stack so the total bar height equals
            # the underlying value count.  Pre-fix those rows were
            # silently skipped, distorting the visible shape.
            tier_counts = {t: np.zeros(n_bins, dtype=int) for t in _TIER_STACK_ORDER}
            for i, t in zip(bin_idx, tiers):
                if t in tier_counts:
                    tier_counts[t][i] += 1
                else:
                    tier_counts["unassigned"][i] += 1
            self._draw_stacked_bars(edges, tier_counts)
        elif (
            color_by in self._CATEGORICAL_COLOR_BY
            and self._df is not None
            and color_by in self._df.columns
        ):
            # Generic categorical stacking — handles cluster_id,
            # is_border_cell, euler_number, qc_status uniformly.
            # Cycles through a Set2-style palette for unique values.
            cat_palette = [
                (102, 194, 165),
                (252, 141, 98),
                (141, 160, 203),
                (231, 138, 195),
                (166, 216, 84),
                (255, 217, 47),
                (229, 196, 148),
                (179, 179, 179),
            ]
            # Perf (2026-06-08): per-column .values + numpy row-mask, no wide
            # df[~mask] copy (see _values_from_df).
            #
            # Every per-row array here is cut with the SAME finite mask
            # _values_from_df uses, then with ``shown``, so it lines up
            # with ``bin_idx`` row for row.  Ghost rows go to the grey
            # stack (they are already out of ``shown`` when hidden).
            df = self._df
            finite = np.isfinite(df[feature].values.astype(np.float64))
            cat_vals = df[color_by].values[finite][shown]
            if "label_status" in df.columns:
                is_ghost = np.isin(
                    df["label_status"].values[finite],
                    ["deleted", "superseded"],
                )[shown]
            else:
                is_ghost = np.zeros(len(cat_vals), dtype=bool)
            # Map each unique active category to a palette slot.
            unique_cats = list(dict.fromkeys(cat_vals[~is_ghost].tolist()))
            cat_to_slot = {c: i % len(cat_palette) for i, c in enumerate(unique_cats)}
            cat_counts: dict = {}
            ghost_counts = np.zeros(n_bins, dtype=int)
            for i, c, g in zip(bin_idx, cat_vals, is_ghost):
                if g:
                    ghost_counts[i] += 1
                    continue
                key = cat_to_slot[c]
                cat_counts.setdefault(
                    key, np.zeros(n_bins, dtype=int),
                )[i] += 1
            self._draw_categorical_stacked(
                edges, cat_counts, cat_palette, ghost_counts=ghost_counts,
            )
        else:
            # Continuous gradient — bin color from bin midpoint normalized
            # to the value range.  Used when Color by matches the X-axis
            # feature (e.g. "Triage by qc_score, Color by qc_score") or
            # when the chosen Color by is not categorical.
            counts = np.zeros(n_bins, dtype=int)
            for i in bin_idx:
                counts[i] += 1
            self._draw_gradient_bars(edges, counts, x_range)

        # Tier lines: visible only when X-axis is qc_score (Rule A).
        # Lines are PERSISTENT pyqtgraph objects (created once in
        # __init__) — _attach_tier_lines re-adds them to the plot
        # without recreating, preserving user drag state across
        # redraws.  Quartile defaults are applied only when a line
        # is currently invisible (first show or after reset).
        if feature == "qc_score":
            # Phase 2C+ (2026-05-04): defaults come from the saved
            # qc_tier column when present — so lines naturally snap to
            # whatever tier boundaries the patch was last calibrated
            # to (committed user calibration, in-session drag positions
            # re-applied by the controller, or quartile-derived from
            # _auto_assign_qc_tiers at extract time).  Falls back to
            # data quartiles (over every finite row, ghosts included,
            # whether or not the ghost stack is shown), then to
            # proportional defaults when there are fewer than 4 values.
            tier_derived = self._derive_thresholds_from_qc_tier()
            if tier_derived is not None:
                defaults = tier_derived
            elif len(all_values) >= 4:
                q25, q50, q75 = np.percentile(all_values, [25, 50, 75])
                defaults = (float(q25), float(q50), float(q75))
            else:
                xspan = x_range[1] - x_range[0]
                defaults = (
                    x_range[0] + 0.15 * xspan,
                    x_range[0] + 0.40 * xspan,
                    x_range[0] + 0.65 * xspan,
                )
            self._attach_tier_lines(defaults, visible=True)
            # Crash-fix (2026-06-08): the gate region is mutually
            # exclusive with the tier lines (Rule A).  Since _redraw no
            # longer clears the scene, the region may still be attached
            # from a prior non-qc_score redraw — just hide it (it is
            # ignoreBounds=True so it never affects autoRange).  Do NOT
            # remove it: that would be the same remove/re-add churn we
            # are eliminating, and the gate region is never the dragged
            # object during a qc_score-feature drag anyway.
            if self._gate_region is not None:
                self._gate_region.setVisible(False)
        else:
            # Hide tier lines for non-qc_score X axes (Rule A).  They
            # stay attached to the scene (just invisible) — they are
            # never removed/re-added, which is the crash we are fixing.
            for line in self._tier_lines:
                line.setVisible(False)
            self._sync_halos()
            # Phase F (2026-05-04) — gate region belongs on this canvas.
            # Crash-fix (2026-06-08): add only if not already attached
            # (the scene is no longer cleared each redraw, so the region
            # persists once added — preserving the user's drag position).
            # ignoreBounds=True so the region's outer rect doesn't
            # affect autoRange.  Its visibility is governed by the gate
            # checkbox.
            if self._gate_region is not None:
                plot_item = self._plot_widget.getPlotItem()
                existing = plot_item.items if plot_item is not None else []
                if self._gate_region not in existing:
                    self._plot_widget.addItem(
                        self._gate_region, ignoreBounds=True,
                    )
                self._gate_region.setVisible(self._gate_cb.isChecked())

        self._update_note(feature)
        self._sync_controls()

        # Padding around X range so the leftmost/rightmost bars aren't clipped.
        pad = 0.02 * (x_range[1] - x_range[0])
        self._plot_widget.setXRange(x_range[0] - pad, x_range[1] + pad, padding=0.0)
        # An empty state pinned the Y range; the bars own it again.
        self._plot_widget.enableAutoRange(axis="y")
        label_css = {"color": "#999", "font-size": f"{HIST_LABEL_PX}px"}
        self._plot_widget.setLabel("bottom", feature, **label_css)
        self._plot_widget.setLabel("left", HIST_COUNT_AXIS, **label_css)

    # ------------------------------------------------------------------
    # Empty state + note line
    # ------------------------------------------------------------------

    def _show_empty(self, sentence: str) -> None:
        """Clear bars, lines, band and axis labels; show ``sentence``.

        The tier lines are hidden, so the next drawn redraw re-derives
        their positions (the ``reset_tier_lines`` rule).
        """
        self._clear_bars()
        for line in self._tier_lines:
            line.setVisible(False)
        self._sync_halos()
        if getattr(self, "_gate_region", None) is not None:
            self._gate_region.setVisible(False)
        # Nothing to pan or zoom.
        self._plot_widget.setMouseEnabled(x=False, y=False)
        item = self._plot_widget.getPlotItem()
        for name in ("bottom", "left"):
            # AxisItem.setLabel (not PlotItem.setLabel, which shows the
            # axis again).
            item.getAxis(name).setLabel("")
            item.hideAxis(name)
        self._empty_text.setText(sentence)
        self._empty_text.setPos(0, 0)
        self._empty_text.setVisible(True)
        self._plot_widget.setXRange(-1, 1, padding=0.0)
        self._plot_widget.setYRange(-1, 1, padding=0.0)
        self._empty_sentence = sentence
        self._set_note("")
        self._sync_controls()

    def _hide_empty(self) -> None:
        if self._empty_sentence is None and not self._empty_text.isVisible():
            return
        self._empty_text.setVisible(False)
        self._empty_sentence = None
        item = self._plot_widget.getPlotItem()
        item.showAxis("bottom")
        item.showAxis("left")
        # Pan + zoom on X only — Y zoom isn't useful for histograms.
        self._plot_widget.setMouseEnabled(x=True, y=False)

    def empty_sentence(self):
        """The sentence on screen in an empty state, else ``None``."""
        return self._empty_sentence

    def _set_note(self, text: str) -> None:
        self._note_label.set_full_text(text)
        self._note_label.setVisible(bool(text))

    def _sync_controls(self) -> None:
        """The combo row follows the state (silent: never emits).

        No data: both combos disabled (the empty sentence is the reason),
        Reset lines and Gate hidden.  Bars drawn: Reset lines on qc_score,
        Gate on any other feature (they share one slot, Rule A).  A
        feature with no values: both hidden, the combos stay live -- the
        way out is the Triage by combo itself.
        """
        has_df = self._df is not None
        drawn = has_df and self._empty_sentence is None
        feature = self._feature_combo.currentText()
        live = not self._locked
        self._feature_combo.setEnabled(has_df and live)
        self._color_combo.setEnabled(has_df and live)
        self._reset_btn.setVisible(drawn and feature == "qc_score")
        self._gate_cb.setVisible(drawn and feature not in ("", "qc_score"))
        self._reset_btn.setEnabled(live)
        self._gate_cb.setEnabled(live)
        # Only on a change: _sync_controls runs on every drag tick.
        for line in self._tier_lines:
            if bool(getattr(line, "movable", True)) != live:
                line.setMovable(live)
        region = getattr(self, "_gate_region", None)
        if region is not None and bool(getattr(region, "movable", True)) != live:
            region.setMovable(live)

    def _update_note(self, feature: str) -> None:
        """The note line under the plot: while the
        feature is qc_score, ``HIST_NOTE_ALL_CURATED`` when every active
        (non-ghost) label is curated, else ``HIST_NOTE_DRAG``; nothing for
        any other feature."""
        df = self._df
        if feature != "qc_score" or df is None:
            self._set_note("")
            return
        if "label_status" in df.columns:
            keep = ~df["label_status"].isin(["deleted", "superseded"]).to_numpy()
        else:
            keep = np.ones(len(df), dtype=bool)
        n_active = int(keep.sum())
        if n_active == 0:
            self._set_note("")
            return
        if "qc_tier" in df.columns:
            active_tiers = df["qc_tier"].values[keep]
            all_curated = bool(np.all(active_tiers == "curated"))
        else:
            all_curated = False
        if all_curated:
            self._set_note(HIST_NOTE_ALL_CURATED.format(n=f"{n_active:,}"))
        else:
            self._set_note(HIST_NOTE_DRAG)

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _clear_bars(self) -> None:
        """Remove ONLY the per-redraw histogram bar items from the plot.

        Crash-fix (2026-06-08): replaces the old blanket
        ``_plot_widget.clear()`` in ``_redraw``.  ``clear()`` removed
        every scene item — including the persistent tier
        ``InfiniteLine``s and the gate ``LinearRegionItem`` — which
        forced a re-``addItem`` of those same objects each redraw.  When
        the redraw fired synchronously from a tier line's
        ``sigPositionChanged`` (mid-drag), tearing the dragged line out
        of the scene and re-adding it while its handler was still on the
        stack caused a re-entrant pyqtgraph scene mutation → native
        Windows access violation.

        Only the ``BarGraphItem``s created by the ``_draw_*`` helpers are
        tracked in ``self._bar_items`` and removed here.  ``removeItem``
        is a no-op when the item is no longer in the plot, so this is
        safe even if pyqtgraph already dropped one.  The tier lines and
        gate region are left exactly where they are.
        """
        for bar in self._bar_items:
            self._plot_widget.removeItem(bar)
        self._bar_items = []

    def _draw_stacked_bars(
        self, edges: np.ndarray, tier_counts: dict[str, np.ndarray],
    ) -> None:
        """Render a per-tier stacked histogram."""
        n_bins = len(edges) - 1
        bottoms = np.zeros(n_bins, dtype=float)
        for tier in _TIER_STACK_ORDER:
            counts = tier_counts.get(tier)
            if counts is None or counts.sum() == 0:
                continue
            r, g, b = _TIER_COLORS_RGB[tier]
            bar = pg.BarGraphItem(
                x0=edges[:-1], x1=edges[1:],
                y0=bottoms, height=counts.astype(float),
                brush=pg.mkBrush(r, g, b, 200),
                pen=pg.mkPen(color=(20, 20, 20, 120), width=0.6),
            )
            self._plot_widget.addItem(bar)
            self._bar_items.append(bar)
            bottoms = bottoms + counts.astype(float)

    def _draw_categorical_stacked(
        self, edges: np.ndarray, cat_counts: dict, palette: list,
        ghost_counts=None,
    ) -> None:
        """Stacked histogram colored by categorical key (e.g. cluster ID),
        on a grey ghost stack when ``ghost_counts`` has any rows."""
        n_bins = len(edges) - 1
        bottoms = np.zeros(n_bins, dtype=float)
        if ghost_counts is not None and ghost_counts.sum() > 0:
            r, g, b = GHOST_GREY
            bar = pg.BarGraphItem(
                x0=edges[:-1], x1=edges[1:],
                y0=bottoms, height=ghost_counts.astype(float),
                brush=pg.mkBrush(r, g, b, 200),
                pen=pg.mkPen(color=(20, 20, 20, 120), width=0.6),
            )
            self._plot_widget.addItem(bar)
            self._bar_items.append(bar)
            bottoms = bottoms + ghost_counts.astype(float)
        for cat_id in sorted(cat_counts.keys()):
            counts = cat_counts[cat_id]
            if counts.sum() == 0:
                continue
            r, g, b = palette[cat_id % len(palette)]
            bar = pg.BarGraphItem(
                x0=edges[:-1], x1=edges[1:],
                y0=bottoms, height=counts.astype(float),
                brush=pg.mkBrush(r, g, b, 200),
                pen=pg.mkPen(color=(20, 20, 20, 120), width=0.6),
            )
            self._plot_widget.addItem(bar)
            self._bar_items.append(bar)
            bottoms = bottoms + counts.astype(float)

    def set_active_cmap(self, name: str) -> None:
        """Phase 2D (2026-05-04) — sync gradient with right-plot cmap.

        Called by the controller when the right plot's ``_cmap_combo``
        text changes (or once at wire-up time for the initial sync).
        Stores the new colormap name, re-derives stops via
        :func:`_stops_for_cmap`, and triggers :meth:`_redraw` so any
        gradient bars currently on the canvas refresh to the new
        colormap.

        No-op when ``name`` is the same as the current setting (avoids
        spurious redraws when the right plot re-emits the signal
        without a real change).
        """
        if name == self._active_cmap:
            return
        self._active_cmap = name
        self._gradient_stops = _stops_for_cmap(name)
        # Redraw so existing gradient bars pick up the new stops.
        # Only meaningful when Color by is continuous; categorical
        # paths ignore _gradient_stops.
        self._redraw()

    def _draw_gradient_bars(
        self, edges: np.ndarray, counts: np.ndarray, x_range: tuple,
    ) -> None:
        """Draw bars colored by a continuous gradient on the bin center."""
        span = x_range[1] - x_range[0]
        for i in range(len(counts)):
            if counts[i] == 0:
                continue
            mid = 0.5 * (edges[i] + edges[i + 1])
            t = (mid - x_range[0]) / span if span > 0 else 0.5
            # Phase 2D — use the active gradient (synced with right
            # plot's _cmap_combo via set_active_cmap) instead of the
            # pre-Phase-2D hardcoded Coolwarm.
            r, g, b = _sample_gradient(t, self._gradient_stops)
            bar = pg.BarGraphItem(
                x0=[edges[i]], x1=[edges[i + 1]], height=[float(counts[i])],
                brush=pg.mkBrush(r, g, b, 200),
                pen=pg.mkPen(color=(20, 20, 20, 120), width=0.6),
            )
            self._plot_widget.addItem(bar)
            self._bar_items.append(bar)

    def _create_tier_lines(self) -> None:
        """Create the 3 persistent threshold InfiniteLines (Phase 2C).

        Lines are created ONCE in __init__, hidden initially, and added
        to the plot once by ``_attach_tier_lines``.  A redraw never
        removes them (it removes only the bars, see ``_clear_bars``),
        which preserves:
          - drag state (the InfiniteLine remembers its position)
          - signal connections (sigPositionChanged etc. stay wired)

        Each line has a dark halo beneath it (``_tier_halos``: a second,
        wider, unmovable line in ``HALO_RGB`` at z - 1) so a green line
        reads on green bars; ``_sync_halos`` keeps it on its line.

        Per ``qc_features.assign_qc_tiers``' docstring, EpiCycle's
        convention is **higher qc_score = WORSE**, so:
            leftmost  (t1) → green   — good / review boundary
            middle    (t2) → orange  — review / bad boundary
            rightmost (t3) → red     — bad / garbage boundary
        """
        line_colors = (
            (80, 200, 120),    # green  — good / review boundary
            (255, 160, 40),    # orange — review / bad boundary
            (240, 60, 60),     # red    — bad / garbage boundary
        )
        self._tier_lines = []
        self._tier_halos = []
        for color in line_colors:
            halo = pg.InfiniteLine(
                pos=0.0, angle=90, movable=False,
                pen=pg.mkPen(color=HALO_RGB + (255,), width=HALO_WIDTH),
            )
            halo.setVisible(False)
            halo.setZValue(_LINE_Z - 1)
            # Never takes a click or a hover from the line above it.
            halo.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            halo.setAcceptHoverEvents(False)
            self._tier_halos.append(halo)
            line = pg.InfiniteLine(
                pos=0.0, angle=90, movable=True,
                pen=pg.mkPen(color=color + (220,), width=2),
                hoverPen=pg.mkPen(color=color + (255,), width=3),
            )
            line.setVisible(False)  # positioned + shown by _redraw
            # Crash-fix follow-up (2026-06-08): paint above the bars.
            # See _LINE_Z rationale — persistent lines are added once in
            # __init__ but bars are re-addItem'd after them each redraw,
            # so without an explicit z the ~78%-opaque bars wash out the
            # draggable lines.  z=10 > bars' default z=0 → lines on top.
            line.setZValue(_LINE_Z)
            # The halo follows first, then the controller hears the move.
            line.sigPositionChanged.connect(self._sync_halos)
            line.sigPositionChanged.connect(self._on_tier_line_moved)
            line.sigPositionChangeFinished.connect(self._on_tier_line_released)
            self._tier_lines.append(line)

    # ------------------------------------------------------------------
    # Phase F (2026-05-04) — multi-feature gate via pg.LinearRegionItem
    # ------------------------------------------------------------------

    def _create_gate_region(self) -> None:
        """Create the persistent gate-band ``pg.LinearRegionItem``.

        Mirrors the Step 4 pattern (``segmentation/analysis_widget.py``)
        but in the selection colour ``GATE_COLOUR`` (the lasso's), and
        with split signals so the continuous drag stays cheap:

          - ``sigRegionChanged``        → ``_on_gate_region_changed``
            (per-pixel, light: right-plot selection bucket only)
          - ``sigRegionChangeFinished`` → ``_on_gate_region_finished``
            (once on release, heavy: 3D viewer colormap + navigator
            + subviewer sync)

        The region's two boundaries are individually draggable
        (resize); the middle area drags as a unit (move).  Inside is
        tinted with a low-alpha brush in the selection colour.
        Default hidden; shown by ``_on_gate_toggled`` when the user
        checks the gate while the X-axis is a non-qc_score feature.
        """
        self._gate_region = pg.LinearRegionItem(
            values=[0.0, 1.0],
            brush=pg.mkBrush(*GATE_RGB, 40),
            pen=pg.mkPen(color=GATE_RGB + (220,), width=1),
            hoverBrush=pg.mkBrush(*GATE_RGB, 60),
            movable=True,
        )
        # Crash-fix follow-up (2026-06-08): paint above the bars — same
        # rationale as the tier lines (see _LINE_Z).  The region is added
        # once and persists across selective-clear redraws; without an
        # explicit z the bars re-added after it would obscure the gate
        # band.  z=10 > bars' default z=0 → gate region on top.
        self._gate_region.setZValue(_LINE_Z)
        self._gate_region.setVisible(False)
        self._gate_region.sigRegionChanged.connect(
            self._on_gate_region_changed,
        )
        self._gate_region.sigRegionChangeFinished.connect(
            self._on_gate_region_finished,
        )

    def _on_feature_changed_for_gate(self, feature: str) -> None:
        """Toggle gate-checkbox visibility when X-axis feature changes.

        Gate is mutually exclusive with the three tier lines — when the
        feature is qc_score, the tier lines own the visualization; the
        gate checkbox hides and force-unchecks (which clears the
        selection so a stale gate doesn't survive a feature switch).
        A switch to another gateable feature with Gate ticked re-seats the
        band.  A data push fills the combo under blockSignals, so it
        never reaches this handler.
        """
        is_qc_score = feature == "qc_score"
        self._sync_controls()
        if is_qc_score and self._gate_cb.isChecked():
            # Force uncheck — fires _on_gate_toggled(False) which
            # clears the gate selection cleanly.
            self._gate_cb.setChecked(False)
        elif (
            self._gate_cb.isChecked()
            and self._df is not None
            and feature in self._df.columns
        ):
            # A switch between two gateable features: the band still
            # holds the OLD feature's values.  Re-seat it at the new
            # feature's 10th / 90th percentiles and emit (a user action),
            # so the plot does not keep the old gated ids.
            self._on_gate_toggled(True)

    def untick_gate_silently(self) -> None:
        """Turn Gate off and hide its band WITHOUT emitting anything.

        For a selection that did not come from the gate (a lasso, a
        legend pick, a patch switch): the gate's own untick emits an empty
        set on both gate signals, which would clear that selection.
        """
        self._gate_cb.blockSignals(True)
        try:
            self._gate_cb.setChecked(False)
        finally:
            self._gate_cb.blockSignals(False)
        if getattr(self, "_gate_region", None) is not None:
            self._gate_region.setVisible(False)

    def _compute_gated_label_ids(self) -> set:
        """Return the set of label IDs whose feature value is within
        the current gate region.  Empty set if gate is disabled or
        unconfigurable (qc_score / no df)."""
        if not self._gate_cb.isChecked():
            return set()
        if self._df is None:
            return set()
        feature = self._feature_combo.currentText()
        if feature == "qc_score" or feature not in self._df.columns:
            return set()
        low, high = self._gate_region.getRegion()
        low, high = float(low), float(high)
        if low > high:
            low, high = high, low
        df = self._df
        if "label_status" in df.columns:
            active = df[~df["label_status"].isin(["deleted", "superseded"])]
        else:
            active = df
        if "label" not in active.columns or feature not in active.columns:
            return set()
        values = active[feature].values.astype(float)
        finite = np.isfinite(values)
        mask = finite & (values >= low) & (values <= high)
        return {int(lid) for lid in active.loc[mask, "label"].values}

    def _on_gate_toggled(self, checked: bool) -> None:
        """Show/hide gate region + emit empty set when disabled.

        On enable: positions the region at the 10th / 90th percentiles
        of the current feature's active rows, then fires both signals
        once so the controller picks up the initial selection.
        ``blockSignals`` discipline around ``setRegion`` to avoid the
        recursion bug pattern fixed in ``e9fa9e4`` / ``e4944c9``.

        On disable: hides the region and emits empty sets on BOTH
        signals so the controller clears the right plot AND the 3D
        viewer isolation cleanly.
        """
        feature = self._feature_combo.currentText()
        gateable = (
            feature != "qc_score"
            and self._df is not None
            and feature in self._df.columns
        )
        self._gate_region.setVisible(checked and gateable)

        if not (checked and gateable):
            # Clear: empty set on both signals so the controller's
            # heavy and light handlers both reset their state.
            self.gated_selection_changed.emit(set())
            self.gated_drag_finished.emit(set())
            return

        # Position the region at 10th / 90th percentiles of the
        # active population for the current feature.
        df = self._df
        if "label_status" in df.columns:
            active = df[~df["label_status"].isin(["deleted", "superseded"])]
        else:
            active = df
        if len(active) < 2:
            self._gate_region.setVisible(False)
            self.gated_selection_changed.emit(set())
            self.gated_drag_finished.emit(set())
            return
        values = active[feature].values.astype(float)
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            self._gate_region.setVisible(False)
            self.gated_selection_changed.emit(set())
            self.gated_drag_finished.emit(set())
            return
        low_default = float(np.percentile(finite, 10))
        high_default = float(np.percentile(finite, 90))

        self._gate_region.blockSignals(True)
        try:
            self._gate_region.setRegion([low_default, high_default])
        finally:
            self._gate_region.blockSignals(False)

        # Seed both signals so the controller initialises BOTH the
        # right-plot bucket AND the 3D viewer isolation.
        ids = self._compute_gated_label_ids()
        self.gated_selection_changed.emit(ids)
        self.gated_drag_finished.emit(ids)

    def _on_gate_region_changed(self, _region) -> None:
        """Continuous (per-pixel) drag handler — emits the LIGHT
        signal so the controller updates only the right-plot
        selection bucket.  Heavy 3D viewer / navigator / subviewer
        sync is deferred to ``_on_gate_region_finished``.
        """
        self.gated_selection_changed.emit(self._compute_gated_label_ids())

    def _on_gate_region_finished(self, _region) -> None:
        """Drag-release handler — emits the HEAVY signal so the
        controller runs the full ``_on_qc_plot_label_selected``
        pipeline (3D viewer DirectLabelColormap rebuild, navigator
        highlight, subviewer mirror).  Mirrors the
        tier_drag_finished pattern that solved the same lag problem
        for tier-line drags.
        """
        self.gated_drag_finished.emit(self._compute_gated_label_ids())

    def _attach_tier_lines(self, thresholds: tuple, visible: bool) -> None:
        """Re-attach the persistent tier lines to the plot, with the
        given (t1, t2, t3) positions and visibility per Rule A.

        Lines that are currently invisible get repositioned to the
        provided defaults; lines that are visible (i.e. user has
        dragged them or they were positioned on a previous redraw)
        keep their current values.  Mirrors qc_plot_widget's
        ``if not line.isVisible(): line.setValue(default)`` pattern.

        Phase C debug fix (2026-05-04): wrap ``line.setValue`` in
        ``blockSignals(True)/False``.  Without this, pyqtgraph's
        ``InfiniteLine.setPos`` synchronously fires
        ``sigPositionChanged`` → ``_on_tier_line_moved`` → emits
        ``thresholds_changed`` → controller's
        ``_on_tier_thresholds_changed`` → ``_update_flagged_labels_list``
        → ``_push_to_triage_histogram`` → ``set_data`` → ``_redraw``
        → back into ``_attach_tier_lines``.  That recursion crashed
        Python with a Windows access violation in deepcopy after
        ~10+ levels deep on user's test machine (faulthandler trace
        2026-05-04).  ``_on_reset_clicked`` uses the same pattern.


        Crash-fix (2026-06-08): this method is now IDEMPOTENT w.r.t. the
        scene.  ``_redraw`` no longer calls ``_plot_widget.clear()`` (it
        does selective bar-only removal via ``_clear_bars``), so the
        persistent tier lines stay in the plot across redraws.  Adding a
        line is therefore guarded by a membership check — we NEVER remove
        a line first (removing + re-adding the dragged line while its own
        ``sigPositionChanged`` handler is on the stack is the re-entrant
        scene mutation that crashed Python with a native access
        violation).  We only ever (a) add a line not yet in the scene,
        (b) reposition invisible lines under ``blockSignals`` (the
        2026-05-04 recursion guard), and (c) update visibility.  A line
        already in the scene is left exactly where it is.
        """
        plot_item = self._plot_widget.getPlotItem()
        existing = plot_item.items if plot_item is not None else []
        for line, halo, default_val in zip(
            self._tier_lines, self._tier_halos, thresholds,
        ):
            # Add only if not already attached — never remove + re-add,
            # which is the mid-drag re-entrant scene mutation crash.
            if halo not in existing:
                self._plot_widget.addItem(halo)
            if line not in existing:
                self._plot_widget.addItem(line)
            if not line.isVisible():
                line.blockSignals(True)
                try:
                    line.setValue(float(default_val))
                finally:
                    line.blockSignals(False)
            line.setVisible(visible)
        self._sync_halos()

    def _sync_halos(self, *_args) -> None:
        """Put each halo on its line: same position, same visibility."""
        for line, halo in zip(self._tier_lines, self._tier_halos):
            value = float(line.value())
            if float(halo.value()) != value:
                halo.setValue(value)
            if halo.isVisible() != line.isVisible():
                halo.setVisible(line.isVisible())

    def _on_tier_line_moved(self, _line) -> None:
        """Slot fired continuously while user drags any tier line.

        Emits ``thresholds_changed`` with the sorted (t1, t2, t3)
        triple.  The controller's ``_on_tier_thresholds_changed``
        slot (shared with the right plot) re-runs assign_qc_tiers
        on the DataFrame and triggers UI refresh.
        """
        thresholds = tuple(sorted(float(line.value()) for line in self._tier_lines))
        self.thresholds_changed.emit(thresholds)

    def _on_tier_line_released(self, _line) -> None:
        """Slot fired once when the user releases the drag.  Emits
        ``tier_drag_finished`` so the controller can do the heavier
        viewer color sync that's skipped during continuous dragging.
        """
        thresholds = tuple(sorted(float(line.value()) for line in self._tier_lines))
        self.tier_drag_finished.emit(thresholds)

    # ------------------------------------------------------------------
    # Reset lines
    # ------------------------------------------------------------------

    def _on_reset_clicked(self) -> None:
        """Reposition tier lines at quartiles of the current feature
        and re-tier the DataFrame to match.

        Phase 2C+ semantics: tier lines normally derive from the saved
        qc_tier column (so they reflect each patch's own tier state).
        Reset is the explicit override that says "ignore saved state,
        snap to quartiles of THIS data."  Implementation:
          1. Compute (q25, q50, q75) of the current feature values.
          2. Set each tier line directly via setValue (signals blocked
             so we emit a single clean signal pair at the end).
          3. Emit ``thresholds_changed`` + ``tier_drag_finished`` so
             the controller re-runs ``assign_qc_tiers(df, quartiles)``,
             clears the per-patch threshold map slot for this patch,
             and refreshes both plots.
        """
        if self._df is None:
            return
        feature = self._feature_combo.currentText()
        if feature != "qc_score":
            return  # Rule A: lines hidden for non-qc_score X axes
        values, _tiers, _xr = self._values_from_df(feature)
        if values is None or len(values) < 4:
            return
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        quartiles = (float(q25), float(q50), float(q75))
        for line, val in zip(self._tier_lines, quartiles):
            line.blockSignals(True)
            line.setValue(val)
            line.blockSignals(False)
        self._sync_halos()
        # Emit once with the final triple (sorted ascending) so the
        # controller can re-tier + refresh both plots in one shot.
        self.thresholds_changed.emit(quartiles)
        self.tier_drag_finished.emit(quartiles)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, df) -> None:
        """Receive the current DataFrame (``None`` = the empty state).

        Pushes a fresh DataFrame to the panel and triggers a redraw.
        Combos auto-populate from ``df.columns``, preserving the user's
        prior selection when its column is still present.

        ``None`` (a patch switch, a patch with no QC features) clears
        bars, lines, band and axis labels and shows ``HIST_EMPTY``.

        Gate: ``None`` and a NEW df (a different object) turn Gate off
        WITHOUT emitting (``untick_gate_silently``) -- the gate's own
        untick emits an empty set, which would clear a lasso made in the
        plot.  The same df pushed again keeps the gate.

        Tier-line position handling (2026-05-04):
          - New df (first call, or a different object) → derive lines
            from the saved ``qc_tier`` column (or fall back to quartiles).
          - Same df (re-tier drag triggered) → preserve user-dragged
            positions (the controller already mutated qc_tier with the
            new thresholds; the InfiniteLine objects persist across
            redraws).
          - Reset lines button → reset to the current data's quartiles.
        Detected via ``df is self._df``.

        The "derive from saved qc_tier" rule means each patch's lines
        naturally reflect whatever tier state was last assigned to it
        — quartile defaults from auto-tier-on-extract for fresh
        patches, user-saved boundaries for committed-triage patches,
        or in-session drag positions when the controller re-applied
        ``self._patch_tier_thresholds`` on patch switch.
        """
        if df is None:
            self._df = None
            self.untick_gate_silently()
            self.reset_tier_lines()
            self._redraw()
            return

        is_new_df = df is not self._df
        if is_new_df:
            self.untick_gate_silently()
        self._df = df
        if is_new_df:
            # Derive line positions from the saved qc_tier column so
            # patch switches reflect each patch's own tier state.  Hide
            # lines first so _attach_tier_lines (called inside _redraw)
            # treats them as "needs default" — the default is now
            # tier-derived rather than quartile-derived, computed below.
            self.reset_tier_lines()
        self._populate_combos_from_df(df)
        # The combos were filled under blockSignals, so the feature may
        # have changed without _on_feature_changed_for_gate running;
        # _redraw ends with _sync_controls, which puts the row right.
        self._redraw()

    def set_locked(self, locked: bool) -> None:
        """The host's lock (Step 3 plot pass P2):
        ``True`` greys Triage by, Colour bars by, Reset lines and Gate and
        freezes the tier lines and the gate band; ``False`` gives them back
        as the data allows.  Silent: nothing is emitted."""
        locked = bool(locked)
        if locked == self._locked:
            return
        self._locked = locked
        self._sync_controls()

    def is_locked(self) -> bool:
        """Whether :meth:`set_locked` holds the panel."""
        return self._locked

    def gate_range(self):
        """``(feature, low, high)`` of the gate band while Gate is ticked
        and shown, else ``None`` (the Selection card's "Gate:" line)."""
        region = getattr(self, "_gate_region", None)
        if (region is None or not self._gate_cb.isChecked()
                or not region.isVisible()):
            return None
        low, high = region.getRegion()
        return (self._feature_combo.currentText(), float(low), float(high))

    def set_ghosts_visible(self, visible: bool) -> None:
        """Draw (default) or leave out the ghost rows (committed /
        discarded labels) -- driven by the plot's Ghosts toggle.  The
        tier lines and the threshold maths are not affected."""
        visible = bool(visible)
        if visible == self._ghosts_visible:
            return
        self._ghosts_visible = visible
        self._redraw()

    def _derive_thresholds_from_qc_tier(self) -> tuple:
        """Return (t1, t2, t3) implied by the current df's qc_tier column.

        EpiCycle convention (per ``qc_features.assign_qc_tiers``):
        higher qc_score = WORSE.  So:
            t1 = max qc_score among 'good' rows
            t2 = max qc_score among 'review' rows
            t3 = max qc_score among 'bad' rows

        Returns ``None`` when the column is missing or any of the 3
        tier slots has no rows — caller falls back to quartiles.
        """
        if self._df is None or "qc_tier" not in self._df.columns:
            return None
        df = self._df
        if "qc_score" not in df.columns:
            return None
        # Perf (2026-06-08): numpy on the two needed columns instead of FOUR
        # full-DataFrame copies (df[~mask] + df[df.qc_tier==t] x3) — a minor
        # per-tick win (see _values_from_df), not the post-extract lag fix.
        if "label_status" in df.columns:
            keep = ~df["label_status"].isin(["deleted", "superseded"]).to_numpy()
        else:
            keep = np.ones(len(df), dtype=bool)
        qc_tier = df["qc_tier"].values[keep]
        qc_score = df["qc_score"].values[keep].astype(np.float64)
        if len(qc_score) == 0:
            return None

        thresholds = []
        for tier_name in ("good", "review", "bad"):
            m = (qc_tier == tier_name)
            if not m.any():
                return None  # gap in tier coverage — use quartiles instead
            thresholds.append(float(np.nanmax(qc_score[m])))
        return tuple(sorted(thresholds))

    def reset_tier_lines(self) -> None:
        """Hide tier lines so the next redraw repositions them at the
        current data's quartile defaults.  Mirrors
        ``qc_plot_widget.reset_tier_lines`` for symmetry.  Public for
        the controller to call on explicit user actions if needed.
        """
        for line in self._tier_lines:
            line.setVisible(False)
        self._sync_halos()

    def _populate_combos_from_df(self, df) -> None:
        """Refresh the Triage by / Color by combo items from df columns.

        Skips obviously non-feature columns (label IDs, status flags
        better suited as Color by).  Preserves the user's current
        selection when the column is still present.
        """
        # Numeric columns only for Triage by (X-axis) — exclude
        # categorical / id columns.
        skip_x = {"label", "label_status", "qc_status", "qc_tier",
                  "feature_state", "patch_name", "cluster",
                  "cluster_id", "border_cell"}
        try:
            import numpy as _np
            numeric_cols = [
                c for c in df.columns
                if c not in skip_x and _np.issubdtype(df[c].dtype, _np.number)
            ]
        except Exception:
            numeric_cols = [c for c in df.columns if c not in skip_x]

        # Make sure qc_score is first if present.
        if "qc_score" in numeric_cols:
            numeric_cols.remove("qc_score")
            numeric_cols.insert(0, "qc_score")

        prev_x = self._feature_combo.currentText()
        self._feature_combo.blockSignals(True)
        self._feature_combo.clear()
        self._feature_combo.addItems(numeric_cols)
        if prev_x in numeric_cols:
            self._feature_combo.setCurrentText(prev_x)
        elif "qc_score" in numeric_cols:
            self._feature_combo.setCurrentText("qc_score")
        self._feature_combo.blockSignals(False)

        # Colour bars by — the canonical EpiCycle categorical columns
        # present in the df, then ONE gradient entry (HIST_VALUE_ITEM):
        # each bar coloured by its position on the Triage-by axis.
        # Numeric columns are not offered: the gradient never read them.
        color_options: list = [
            col for col in self._CATEGORICAL_COLOR_BY if col in df.columns
        ]
        color_options.append(HIST_VALUE_ITEM)

        prev_c = self._color_combo.currentText()
        self._color_combo.blockSignals(True)
        self._color_combo.clear()
        self._color_combo.addItems(color_options)
        self._color_combo.setItemData(
            color_options.index(HIST_VALUE_ITEM),
            HIST_HELP[HIST_VALUE_ITEM],
            Qt.ItemDataRole.ToolTipRole,
        )
        if prev_c in color_options:
            self._color_combo.setCurrentText(prev_c)
        elif "qc_tier" in color_options:
            self._color_combo.setCurrentText("qc_tier")
        self._color_combo.blockSignals(False)

    def _values_from_df(self, feature: str) -> tuple:
        """Extract (values, tiers, x_range) from the stored DataFrame.

        Filters to active rows (excludes ghost rows where
        ``label_status`` is "deleted" or "superseded") and finite
        values only.  ``tiers`` aligns with ``values``; uses
        ``"unassigned"`` for missing qc_tier entries.
        """
        if self._df is None:
            return None, None, None
        df = self._df
        if feature not in df.columns:
            return None, None, None
        # Perf (2026-06-08): operate on individual column ``.values`` with a
        # numpy row-mask instead of copying the full (~100-col) DataFrame via
        # ``df[~mask]``.  A minor win in the per-tick redraw (avoids a wide copy)
        # — NOT the post-extract triage-lag fix (that is the switch-in-place CSV
        # reload in the controller).  A row is "active" when it is
        # not a deleted/superseded ghost.
        # 2026-06-09: INCLUDE committed/ghost rows (label_status deleted /
        # superseded) so committed garbage shows as a gray "ghost" stack instead
        # of vanishing — it stays until a re-extraction drops the ghost rows.
        # (Previously this filtered ghosts out via a `keep` mask.)  Active rows
        # keep their qc_tier; ghost rows are tagged "ghost" (gray) below.
        all_values = df[feature].values.astype(np.float64)
        finite_mask = np.isfinite(all_values)
        values = all_values[finite_mask]
        if len(values) == 0:
            return None, None, None
        if "qc_tier" in df.columns:
            tiers_raw = df["qc_tier"].values[finite_mask]
            # Coerce NaN / None / falsy strings to "unassigned" — the
            # canonical 5 tiers are matched by string equality, anything
            # else falls through to the gray fallback bucket.
            tiers = np.array([
                str(t) if (t is not None and not (isinstance(t, float) and np.isnan(t)))
                else "unassigned"
                for t in tiers_raw
            ])
        else:
            tiers = np.array(["unassigned"] * len(values))
        # Tag committed/ghost rows "ghost" so they stack GRAY (overriding their
        # original tier colour) — these are the cells discarded on Commit Triage.
        if "label_status" in df.columns:
            status = df["label_status"].values[finite_mask]
            ghost_mask = np.isin(status, ["deleted", "superseded"])
            if ghost_mask.any():
                # np.array inferred `tiers`' FIXED-WIDTH string dtype from the
                # active tier strings, which can be <U4 (e.g. all "good"/"bad").
                # Assigning "ghost" (5 chars) into a <U4 array silently TRUNCATES
                # to "ghos" → it'd miss the "ghost" stack and fall into
                # "unassigned" (blue, not gray).  Widen to <U16 first.
                tiers = tiers.astype("<U16")
                tiers[ghost_mask] = "ghost"
        # X range: derive from the actual min/max of values so the
        # histogram shape mirrors QCPlotWidget.  Pre-fix bug (2026-05-04):
        # qc_score was hardcoded to [0, 1], pile-driving real values
        # outside that range into the edge bins via np.clip — distorted
        # shape.  Real qc_score is a composite (PCA + ranks), unbounded.
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0
        x_range = (lo, hi)
        return values, tiers, x_range

    def get_thresholds(self) -> tuple:
        """Return the current (garbage_bad, bad_review, review_good) line positions."""
        if not self._tier_lines:
            return (0.0, 0.0, 0.0)
        return tuple(float(line.value()) for line in self._tier_lines)

"""Floating Label Navigator for 3D curation — triage command center.

Launched from the 3D Curation Annotator's utility panel via the
"Label Navigator..." button.  Provides a table-based label list with
tier filters, sorting, triage actions, and auto-advance for rapid
label-by-label review.

Phase K — progressive disclosure extraction from Curation3DAnnotator.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QRect, QSize, Qt
from qtpy.QtGui import QColor, QFont, QPainter
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_COLORS = {
    "curated": "#00bcd4",
    "good": "#4caf50",
    "review": "#ffc83c",
    "bad": "#ff7828",
    "garbage": "#f03c3c",
}

TIER_ORDER = {"curated": 0, "good": 1, "review": 2, "bad": 3, "garbage": 4}

# Tier-action buttons (Mark Curated / Good / Review / Bad / Garbage):
# - Flag-only — set qc_tier on the label's df row.  No volume mutation.
# - Mirrors TIER_COLORS as an ACCENT on dark background so the button
#   echoes the badge color without dominating the panel.  Idle = dark
#   bg with tier-colored 1px border + tier-colored text.  Hover = bg
#   lightens slightly, text turns white.  Pressed = solid tier color,
#   confirming the action with a clear flash.
def _tier_button_style(tier: str) -> str:
    base = TIER_COLORS.get(tier, "#888888")
    return (
        f"QPushButton {{"
        f" background-color: #2b2b2b;"
        f" color: {base};"
        f" border: 1px solid {base};"
        f" border-radius: 4px;"
        f" padding: 4px 8px;"
        f" font-weight: 600;"
        f"}}"
        f"QPushButton:hover {{"
        f" background-color: #3a3a3a;"
        f" color: #ffffff;"
        f"}}"
        f"QPushButton:pressed {{"
        f" background-color: {base};"
        f" color: #ffffff;"
        f"}}"
        f"QPushButton:disabled {{"
        f" background-color: #2b2b2b;"
        f" border-color: #4a4a4a;"
        f" color: #4a4a4a;"
        f"}}"
    )


# Column model — _COL_COLOR is a small swatch showing the label's actual
# on-screen color, sourced from the active Labels layer's colormap.
# All other columns are keyed by symbolic constants below; updating
# _COLUMNS auto-shifts indices via range(...).
_COLUMNS = ("", "ID", "Area", "Z", "Score", "Tier")
_COL_COLOR, _COL_ID, _COL_AREA, _COL_Z, _COL_SCORE, _COL_TIER = range(6)

_SORT_OPTIONS = ["Tier", "Area (desc)", "Z (asc)", "ID (asc)", "Score (asc)"]


# ---------------------------------------------------------------------------
# Stylesheets — match the EpiCycle dark theme via shared helpers.
# ---------------------------------------------------------------------------

_TABLE_STYLE = """
QTableWidget {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    gridline-color: transparent;
    font-size: 11px;
    color: #cccccc;
    selection-background-color: rgba(212, 168, 67, 0.25);
    selection-color: #eeeeee;
}
QTableWidget::item {
    padding: 4px 6px;
    border: none;
}
QTableWidget::item:hover {
    background-color: rgba(255, 255, 255, 0.05);
}
QTableWidget::item:selected {
    background-color: rgba(212, 168, 67, 0.25);
    color: #eeeeee;
}
QHeaderView::section {
    background-color: rgba(255, 255, 255, 0.04);
    color: #aaaaaa;
    padding: 5px 8px;
    border: none;
    border-bottom: 1px solid rgba(255, 255, 255, 0.10);
    font-size: 10px;
    font-weight: bold;
}
QHeaderView::section:hover {
    background-color: rgba(255, 255, 255, 0.07);
    color: #cccccc;
}
"""

_HEADER_TITLE_STYLE = (
    "font-weight: bold; font-size: 12px; color: #eeeeee; padding: 2px 0;"
)
_SUMMARY_STYLE = "color: #aaaaaa; font-size: 11px;"
_SECTION_LABEL_STYLE = "color: #aaaaaa; font-size: 11px;"

# Tier filter checkboxes: keep the tier's color on the text, but also
# tint the checkbox indicator so the unchecked state still reads as
# "this filter is for THAT tier".
def _tier_filter_checkbox_style(tier_color: str) -> str:
    if tier_color.startswith("#") and len(tier_color) == 7:
        r = int(tier_color[1:3], 16)
        g = int(tier_color[3:5], 16)
        b = int(tier_color[5:7], 16)
    else:
        r, g, b = 200, 200, 200
    return (
        f"QCheckBox {{"
        f"  color: {tier_color};"
        f"  font-size: 11px; font-weight: bold;"
        f"  spacing: 4px;"
        f"}}"
        f"QCheckBox::indicator {{"
        f"  width: 12px; height: 12px;"
        f"  border: 1px solid rgba({r},{g},{b}, 0.55);"
        f"  border-radius: 2px;"
        f"  background: rgba({r},{g},{b}, 0.10);"
        f"}}"
        f"QCheckBox::indicator:checked {{"
        f"  background: rgba({r},{g},{b}, 0.65);"
        f"  border: 1px solid rgba({r},{g},{b}, 0.85);"
        f"}}"
        f"QCheckBox::indicator:hover {{"
        f"  border: 1px solid rgba({r},{g},{b}, 0.85);"
        f"}}"
    )


_PLAIN_CHECKBOX_STYLE = """
QCheckBox {
    color: #cccccc;
    font-size: 11px;
    spacing: 4px;
}
QCheckBox::indicator {
    width: 12px; height: 12px;
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-radius: 2px;
    background: rgba(255, 255, 255, 0.04);
}
QCheckBox::indicator:checked {
    background: rgba(212, 168, 67, 0.55);
    border: 1px solid rgba(212, 168, 67, 0.85);
}
QCheckBox::indicator:hover {
    border: 1px solid rgba(255, 255, 255, 0.40);
}
"""

# Recalculate QC button — kept on its own teal accent because it
# represents a re-analysis action distinct from the green/amber/red
# cluster of triage actions.  Re-themed to match primary_btn_style.
_RECALC_QC_STYLE_KEY = "#5ba4cf"  # teal — Setup accent


# ---------------------------------------------------------------------------
# Item delegates — paint Color + Tier cells directly during the table's
# paintEvent, no per-row widgets.
# ---------------------------------------------------------------------------
#
# _rebuild_table runs on every filter/sort/items-set, so any per-row widget
# creation cost multiplies into noticeable lag with hundreds of labels.
# Delegates eliminate that cost entirely: paint() is invoked only for
# visible cells, and there is no QWidget tree to construct, layout, style,
# or destroy on rebuild.
#
# Custom data is stored on the item via Qt.UserRole + small offsets:
#   _ROLE_TIER_TEXT — tier name (str), read by _TierPillDelegate
#   _ROLE_SWATCH_COLOR — label color (QColor), read by _SwatchDelegate
# DisplayRole is left as the original payload (or empty string) so default
# text rendering is suppressed via initStyleOption.text = "".

_ROLE_TIER_TEXT = Qt.UserRole + 1
_ROLE_SWATCH_COLOR = Qt.UserRole + 1  # different column, no conflict

# Cached pen colors for swatch outline — Qt-style equivalents of
# rgba(255, 255, 255, 0.18) used elsewhere for border accents.
_SWATCH_OUTLINE = QColor(255, 255, 255, 46)


def _batch_swatch_colors(label_layer, label_ids):
    """Return ``{label_id: QColor}`` for *label_ids* in one colormap map.

    [NAV-PERF] 2026-06-05 — ``swatch_color_for_label`` (theme.py) was
    previously called once PER ROW inside ``_rebuild_table``; each call did
    ``cmap.map(np.array([id]))`` (a single-element colormap map) plus an
    alpha≤0 fallback path.  Over ~1900 rows that is ~1900 single-element
    maps.  This helper maps ALL requested ids in ONE ``cmap.map`` call and
    builds a dict the rebuild loop indexes per row.

    The per-id COLOR is byte-for-byte identical to ``swatch_color_for_label``:
      * Same neutral-gray fallback (204,204,204) when the layer / colormap
        is missing or the batched ``map`` raises (in which case we fall
        back to the per-id helper, which raises and returns fallback the
        same way the original did).
      * Same alpha≤0.001 → napari default-label-colormap fallback, but
        done by delegating those specific ids back to
        ``swatch_color_for_label`` (preserving its try/except → fallback
        behavior exactly).  This path is only reachable during isolation
        and affects only the hidden ids, so per-id there is cheap.
      * Same float→255 clip / int rounding.

    Returns an empty dict on any unexpected failure so callers fall back
    to the original per-row path.
    """
    from napari_epicycle.ui_shell.theme import swatch_color_for_label
    import numpy as _np

    fallback = QColor(204, 204, 204)
    result = {}
    if not label_ids:
        return result

    # Match swatch_color_for_label's guard order exactly: a missing layer
    # or non-mappable colormap means EVERY id is the neutral-gray fallback.
    if label_layer is None:
        return {int(i): fallback for i in label_ids}
    cmap = getattr(label_layer, "colormap", None)
    if cmap is None or not hasattr(cmap, "map"):
        return {int(i): fallback for i in label_ids}

    ids = [int(i) for i in label_ids]
    try:
        arr = _np.asarray(
            cmap.map(_np.array(ids, dtype=_np.intp))
        ).reshape(-1, 4)
    except Exception:
        # Batched map failed — defer to the per-id helper, which applies
        # the same try/except → fallback contract for each id.
        return {i: swatch_color_for_label(label_layer, i) for i in ids}

    if arr.shape[0] != len(ids):
        # Shape mismatch — be safe, per-id the whole set.
        return {i: swatch_color_for_label(label_layer, i) for i in ids}

    hidden_ids = []
    for row_i, lid in enumerate(ids):
        r, g, b, a = arr[row_i]
        if float(a) <= 0.001:
            # Isolation/hidden-label path — the live colormap hides this label
            # (alpha 0).  DEFER it to ONE batched map through the cached default
            # label colormap below, instead of a per-id swatch_color_for_label
            # call.  During isolation this branch is the MAJORITY of rows, and
            # per-id was rebuilding the default colormap each time → ~2.1s per
            # rebuild.  Batched here it's O(1).
            hidden_ids.append(lid)
            continue
        result[lid] = QColor(
            int(_np.clip(float(r) * 255, 0, 255)),
            int(_np.clip(float(g) * 255, 0, 255)),
            int(_np.clip(float(b) * 255, 0, 255)),
        )

    if hidden_ids:
        # Map ALL hidden labels in ONE call through the CACHED default label
        # colormap — byte-identical to swatch_color_for_label's per-id fallback
        # (same colormap, same r/g/b extraction, alpha ignored) but O(1) instead
        # of O(hidden).  On any failure, fall back to the per-id helper so the
        # behaviour is never worse than before.
        try:
            from napari_epicycle.ui_shell.theme import default_label_colormap
            _dcmap = default_label_colormap()
            _darr = _np.asarray(
                _dcmap.map(_np.array(hidden_ids, dtype=_np.intp))
            ).reshape(-1, 4)
            if _darr.shape[0] != len(hidden_ids):
                raise ValueError("default-cmap batch shape mismatch")
            for _j, lid in enumerate(hidden_ids):
                dr, dg, db, _da = _darr[_j]
                result[lid] = QColor(
                    int(_np.clip(float(dr) * 255, 0, 255)),
                    int(_np.clip(float(dg) * 255, 0, 255)),
                    int(_np.clip(float(db) * 255, 0, 255)),
                )
        except Exception:
            for lid in hidden_ids:
                result[lid] = swatch_color_for_label(label_layer, lid)
    return result


class _SwatchDelegate(QStyledItemDelegate):
    """Paint a small filled square in the Color column.

    Reads the QColor from the item's ``_ROLE_SWATCH_COLOR`` role.
    No widgets per row — the delegate's ``paint`` is invoked only
    for visible cells.
    """

    _SIZE = 12

    def paint(self, painter, option, index):
        # Cell background (selection / hover) via the platform style.
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""  # suppress default text rendering
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)

        color = index.data(_ROLE_SWATCH_COLOR)
        if color is None:
            return
        if not isinstance(color, QColor):
            try:
                color = QColor(color)
            except Exception:
                return

        size = self._SIZE
        rect = option.rect
        x = rect.x() + (rect.width() - size) // 2
        y = rect.y() + (rect.height() - size) // 2
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(x, y, size, size, color)
        painter.setPen(_SWATCH_OUTLINE)
        painter.drawRect(x, y, size - 1, size - 1)
        painter.restore()


class _TierPillDelegate(QStyledItemDelegate):
    """Paint the tier as a compact rounded pill, left-aligned in the cell.

    Reads the tier string from the item's ``_ROLE_TIER_TEXT`` role.
    Pill geometry: text width + 16px horizontal padding, height = font
    height + 4px, vertically centred, 6px from cell left edge.  Same
    visual vocabulary as the merge dialog cards.
    """

    _LEFT_INSET = 6
    _H_PAD = 8  # padding on each side of the text
    _V_PAD = 2

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)

        tier = index.data(_ROLE_TIER_TEXT)
        if not tier:
            return

        color_hex = TIER_COLORS.get(tier, "#bbbbbb")
        base = QColor(color_hex)
        bg = QColor(base.red(), base.green(), base.blue(), 38)    # ≈0.15 alpha
        border = QColor(base.red(), base.green(), base.blue(), 102)  # ≈0.40 alpha

        # Bold 10px-ish font for the pill text.
        f = QFont(option.font)
        f.setBold(True)
        f.setPointSize(max(8, f.pointSize() - 1))
        fm = painter.fontMetrics() if option.font.pointSize() == f.pointSize() else None
        # We need metrics for the new font — set it on the painter first.
        painter.save()
        painter.setFont(f)
        fm = painter.fontMetrics()

        text_w = fm.horizontalAdvance(tier)
        pill_h = fm.height() + 2 * self._V_PAD
        pill_w = text_w + 2 * self._H_PAD

        rect = option.rect
        pill_x = rect.x() + self._LEFT_INSET
        pill_y = rect.y() + (rect.height() - pill_h) // 2
        pill_rect = QRect(pill_x, pill_y, pill_w, pill_h)

        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(bg)
        painter.setPen(border)
        painter.drawRoundedRect(pill_rect, 8, 8)

        painter.setPen(base)
        painter.drawText(pill_rect, Qt.AlignCenter, tier)
        painter.restore()


class CurationLabelNavigator(QWidget):
    """Floating label navigation + triage panel for 3D curation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label Navigator")

        # Internal state
        self._items: list[dict] = []
        # Switch-safety: set True in closeEvent when the dock is
        # being torn down; reset False in reset_selection_ui on (re)use.
        # A deferred rebuild (QTimer.singleShot(0, _do_pending_rebuild))
        # queued just before teardown checks this FIRST so it can't touch a
        # mid-destruction Qt table — the unclosed door on the B5 crash.
        self._destroyed: bool = False
        self._auto_advance: bool = False
        # Main napari Labels layer — used for the swatch column to
        # sample each row's actual on-screen color.  May stay None in
        # tests; missing layer falls back to neutral gray swatches.
        self._main_label_layer = None

        # Painting badge — current `selected_label` on the main layer.
        # Tracked separately from the table's selected row because the
        # paint brush's active label can diverge (e.g., auto-increment
        # after enclosed-fill bumps to max+1, but the table still shows
        # the row the user clicked).  Updated via ``set_paint_label``
        # and the auto-subscribed ``layer.events.selected_label``
        # listener.  Default 0 = background (eraser).
        self._paint_label: int = 0
        # (layer, callback) for clean disconnect when set_label_layer
        # is called with a new layer or with None at teardown.
        self._selected_label_handler = None

        # Render Selected (Option B) — track what's currently rendered
        # in the viewer so the Render button can light up when the
        # user's selection diverges (multi-select waiting to render).
        # Updated after every successful isolation fire.
        self._last_rendered_label_ids: set = set()

        # Callbacks (injected via set_callbacks)
        self._on_navigate_cb = None
        self._on_select_label_cb = None
        # Unified tier-flag callback: on_mark_tier(label_id, tier_str).
        # Replaces the old separate on_mark_good + on_delete callbacks —
        # the 5 tier buttons (Mark Curated/Good/Review/Bad/Garbage) all
        # dispatch through this one entry point.  Flag-only — does not
        # zero the label in the volume.  Volume destruction happens via
        # the separate Discard All Garbage button (on_discard_garbage).
        self._on_mark_tier_cb = None
        self._on_discard_garbage_cb = None
        self._on_merge_cb = None
        self._on_open_merge_dialog_cb = None
        self._on_focused_mode_changed_cb = None
        self._on_show_selected_label_changed_cb = None
        self._on_isolation_changed_cb = None
        self._on_scroll_subviewer_cb = None
        self._on_recalculate_cb = None
        # New Label button callback — fires when the user
        # clicks "New Label" in the action row.  Receiver bumps
        # selected_label to max(volume) + 1 across all viewers (and
        # ensures paint mode is active on subviewers).
        self._on_new_label_cb = None
        # Delete Label button callback (Task A, 2026-06-01) — fires
        # when the user clicks "Delete Label" in the sidebar.  Receiver
        # snapshots the label's ROI, zeros it in shared_data, syncs
        # all views, and pushes a CurationUndoEntry via the controller
        # so the destructive action is undoable through the canonical
        # curation-undo system (same path used by interp / split /
        # smooth / merge / discard).  See Curation3DAnnotator.
        # _on_delete_label_from_navigator for the canonical handler.
        self._on_delete_label_cb = None
        # "Isolate while painting" toggle
        # — receiver flips the paint-isolation feature on the
        # annotator.  Forwarded via on_paint_label_changed when
        # layer.events.selected_label fires (i.e., when the brush
        # changes targets).
        self._on_paint_isolation_toggled_cb = None
        self._on_paint_label_changed_cb = None
        # Batch Refine — receiver (the annotator) opens the floating
        # Batch Refine dialog when the sidebar button is clicked.
        self._on_batch_refine_cb = None
        # Fired on any table selection change (manual or programmatic) so
        # the annotator can keep the Batch Refine "Selected" count live.
        self._on_selection_changed_cb = None
        # Op-state predicate (2026-05-16) — receiver
        # returns True if any refinement op is staged (interp / split
        # / smooth preview).  The three user-initiated isolation entry
        # points (`_on_isolate_toggled`, `_fire_isolation_if_active`,
        # `_on_render_selected_clicked`) consult this and bail out
        # silently with a transient status message when True, so the
        # user can't accidentally clobber an in-flight op's isolation
        # state.  Wired by `Curation3DAnnotator._wire_navigator_callbacks`
        # to `_is_op_active_for_navigator` (which syncs refine_state
        # before reading `any_op_active`).
        self._is_op_active_cb = None
        # Transient status timer for the brief "Finish current op first"
        # message shown in the editing header when an isolation gate
        # fires.  Lazily created on first use.
        self._op_active_status_timer = None

        self._build_ui()
        self._wire_signals()

    def _curation_debug_enabled(self) -> bool:
        import os
        return os.environ.get("EPICYCLE_CURATION_DEBUG") == "1"

    def _curation_debug(self, msg: str) -> None:
        if self._curation_debug_enabled():
            print(f"[CURATION-DEBUG] {msg}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        from napari_epicycle.ui_shell.theme import (
            DESTRUCTIVE_BTN_STYLE,
            DESTRUCTIVE_BTN_STYLE_DARK,
            INPUT_STYLE,
            PRIMARY_BTN_STYLE_GREEN,
            PRIMARY_BTN_STYLE_TEAL,
            SECONDARY_BTN_STYLE,
            SEPARATOR_LINE_STYLE,
            TOGGLE_BTN_STYLE,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # --- Editing header ---
        self._editing_header = QLabel("")
        self._editing_header.setStyleSheet(_HEADER_TITLE_STYLE)
        layout.addWidget(self._editing_header)

        # --- Summary line ---
        self._summary_label = QLabel("0 labels")
        self._summary_label.setStyleSheet(_SUMMARY_STYLE)
        layout.addWidget(self._summary_label)

        # --- Search row ---
        # Multi-ID search: comma-separated IDs (whitespace tolerant).
        # When non-empty the search overrides tier filters so a label
        # can always be found regardless of which tier filters are off.
        # ENTER commits: single-ID navigates (single-fire), multi-ID
        # selects all matches (then user clicks Render Selected).
        # ESC clears.
        from qtpy.QtWidgets import QLineEdit
        search_row = QHBoxLayout()
        search_row.setSpacing(6)
        search_lbl = QLabel("\U0001F50D Find ID(s):")  # 🔍
        search_lbl.setStyleSheet(_SECTION_LABEL_STYLE)
        search_row.addWidget(search_lbl)
        self._search_field = QLineEdit()
        self._search_field.setPlaceholderText("e.g.  47   or   12, 47, 89")
        self._search_field.setStyleSheet(INPUT_STYLE)
        self._search_field.setToolTip(
            "Comma-separated label IDs.  Filters the table to matching "
            "rows (overrides tier filters).  ENTER navigates / selects, "
            "ESC clears."
        )
        search_row.addWidget(self._search_field, 1)
        layout.addLayout(search_row)

        # --- Filter row + sort ---
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)

        self._tier_filter_curated = QCheckBox("C")
        self._tier_filter_curated.setChecked(True)
        self._tier_filter_curated.setStyleSheet(
            _tier_filter_checkbox_style(TIER_COLORS["curated"])
        )
        self._tier_filter_curated.setToolTip("Show curated-tier labels")

        self._tier_filter_good = QCheckBox("G")
        self._tier_filter_good.setChecked(True)
        self._tier_filter_good.setStyleSheet(
            _tier_filter_checkbox_style(TIER_COLORS["good"])
        )
        self._tier_filter_good.setToolTip("Show good-tier labels")

        self._tier_filter_review = QCheckBox("R")
        self._tier_filter_review.setChecked(True)
        self._tier_filter_review.setStyleSheet(
            _tier_filter_checkbox_style(TIER_COLORS["review"])
        )
        self._tier_filter_review.setToolTip("Show review-tier labels")

        self._tier_filter_bad = QCheckBox("B")
        self._tier_filter_bad.setChecked(True)
        self._tier_filter_bad.setStyleSheet(
            _tier_filter_checkbox_style(TIER_COLORS["bad"])
        )
        self._tier_filter_bad.setToolTip("Show bad-tier labels")

        self._tier_filter_garbage = QCheckBox("Ga")
        self._tier_filter_garbage.setChecked(True)
        self._tier_filter_garbage.setStyleSheet(
            _tier_filter_checkbox_style(TIER_COLORS["garbage"])
        )
        self._tier_filter_garbage.setToolTip("Show garbage-tier labels")

        filter_row.addWidget(self._tier_filter_curated)
        filter_row.addWidget(self._tier_filter_good)
        filter_row.addWidget(self._tier_filter_review)
        filter_row.addWidget(self._tier_filter_bad)
        filter_row.addWidget(self._tier_filter_garbage)
        filter_row.addStretch(1)

        sort_lbl = QLabel("Sort:")
        sort_lbl.setStyleSheet(_SECTION_LABEL_STYLE)
        filter_row.addWidget(sort_lbl)
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(_SORT_OPTIONS)
        self._sort_combo.setMaximumWidth(140)
        self._sort_combo.setStyleSheet(INPUT_STYLE)
        filter_row.addWidget(self._sort_combo)

        layout.addLayout(filter_row)

        # --- Table ---
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(False)  # we handle sorting ourselves
        self._table.setFocusPolicy(Qt.StrongFocus)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(False)
        self._table.setStyleSheet(_TABLE_STYLE)

        # Item delegates — paint Color + Tier cells directly without
        # creating a QWidget per row.  See _SwatchDelegate /
        # _TierPillDelegate above.  Stored on the navigator so they
        # are not garbage-collected.
        self._swatch_delegate = _SwatchDelegate(self._table)
        self._tier_delegate = _TierPillDelegate(self._table)
        self._table.setItemDelegateForColumn(_COL_COLOR, self._swatch_delegate)
        self._table.setItemDelegateForColumn(_COL_TIER, self._tier_delegate)

        # Column widths
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COL_COLOR, QHeaderView.Fixed)
        header.resizeSection(_COL_COLOR, 24)
        header.setSectionResizeMode(_COL_ID, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_AREA, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_Z, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_SCORE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_TIER, QHeaderView.Stretch)

        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 2026-06-01 sidebar-right layout: all action buttons (tier
        # triage, + New Label, Merge Into\u2026, Prev/Next) live in a
        # vertical sidebar to the right of the table instead of in
        # rows below it.  All button object identities, signals,
        # tooltips, and styles are preserved by reference \u2014 only the
        # parent container changes, so every downstream callback +
        # signal wiring + test that references *.btn attributes keeps
        # working.  Top rows above the table (search / filter+sort)
        # are deliberately UNTOUCHED.
        self._mark_good_btn = QPushButton("Mark Good")
        self._mark_good_btn.setToolTip(
            "Mark selected label as good quality (tier=good)"
        )
        self._mark_good_btn.setStyleSheet(_tier_button_style("good"))

        self._mark_review_btn = QPushButton("Mark Review")
        self._mark_review_btn.setToolTip(
            "Mark selected label for further review (tier=review)"
        )
        self._mark_review_btn.setStyleSheet(_tier_button_style("review"))

        self._mark_bad_btn = QPushButton("Mark Bad")
        self._mark_bad_btn.setToolTip(
            "Mark selected label as bad \u2014 flagged for attention, "
            "NOT deleted (tier=bad)"
        )
        self._mark_bad_btn.setStyleSheet(_tier_button_style("bad"))

        self._mark_garbage_btn = QPushButton("Mark Garbage")
        self._mark_garbage_btn.setToolTip(
            "Mark selected label as garbage \u2014 flagged for deletion "
            "in the commit-triage flow (flag-only, no immediate "
            "volume mutation) (tier=garbage)"
        )
        self._mark_garbage_btn.setStyleSheet(_tier_button_style("garbage"))

        self._mark_curated_btn = QPushButton("Mark Curated")
        self._mark_curated_btn.setToolTip(
            "Mark selected label as curated \u2014 confirmed perfect "
            "after manual inspection (tier=curated)"
        )
        self._mark_curated_btn.setStyleSheet(_tier_button_style("curated"))

        # "New Label" creates a fresh label slot by bumping
        # selected_label to max(volume) + 1.  Same effect as napari's
        # default "M" key, but discoverable as a UI affordance \u2014
        # especially for new users who don't know the keyboard
        # shortcut.  PRIMARY_BTN_STYLE_TEAL distinguishes it as a
        # "create" action distinct from the tier-marking buttons.
        # (\u00a71.4, 2026-05-16)
        self._new_label_btn = QPushButton("+ New Label")
        self._new_label_btn.setToolTip(
            "Set selected_label to max(volume) + 1 \u2014 a fresh label "
            "ready to paint into a new cell.  The row is registered "
            "in this list only after you paint with it."
        )
        self._new_label_btn.setStyleSheet(PRIMARY_BTN_STYLE_TEAL)

        # Delete Label button (Task A, 2026-06-01).  Destructive \u2014
        # zeroes the currently selected label in the volume.  Lives
        # directly UNDER + New Label so create/destroy are vertically
        # paired.  Undo flows through the canonical curation-undo
        # system (snapshot ROI -> mutate -> sync -> commit) so Ctrl+Z
        # restores the label exactly like interp/split/smooth reject.
        # DESTRUCTIVE_BTN_STYLE matches the other destructive button
        # (Mark Garbage / Discard) styling already in the navigator.
        self._delete_label_btn = QPushButton("Delete Label")
        self._delete_label_btn.setToolTip(
            "Zero the selected label in the volume (destructive).  "
            "Use the controller's Undo to restore.  Disabled while a "
            "refinement op (interpolation / split / smooth / merge) "
            "or uncommitted paint stroke is in flight."
        )
        self._delete_label_btn.setStyleSheet(DESTRUCTIVE_BTN_STYLE)

        self._merge_btn = QPushButton("Merge Into\u2026")
        self._merge_btn.setToolTip(
            "Merge selected label into another label (relabel)"
        )
        self._merge_btn.setStyleSheet(SECONDARY_BTN_STYLE)

        # Batch Refine \u2014 launches the floating Batch Refine dialog
        # (apply an ordered recipe of geometry-only ops across many
        # labels).  Gated to a clean state by the annotator on open.
        self._batch_refine_btn = QPushButton("\u2699 Batch Refine\u2026")
        from packages.epicycle_annotator.batch_refine import (
            NAV_BATCH_REFINE_TIP,
        )

        self._batch_refine_btn.setToolTip(NAV_BATCH_REFINE_TIP)
        self._batch_refine_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        self._batch_refine_btn.clicked.connect(self._on_batch_refine_clicked)

        self._prev_btn = QPushButton("\u25c0")
        self._prev_btn.setToolTip("Previous visible label")
        self._prev_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        self._next_btn = QPushButton("\u25b6")
        self._next_btn.setToolTip("Next visible label")
        self._next_btn.setStyleSheet(SECONDARY_BTN_STYLE)

        # Helper for sidebar separators (matches existing theme style).
        def _sidebar_sep() -> QFrame:
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(SEPARATOR_LINE_STYLE)
            sep.setFixedHeight(1)
            return sep

        # --- Sidebar container ---
        # Fixed-width vertical sidebar with grouped triage actions.
        # Group order (top to bottom), separated by horizontal lines:
        #   1. Routine triage    : Good / Review / Bad / Garbage
        #   2. Final-confirm     : Curated  (separated to slow mis-clicks)
        #   3. Label management  : + New Label  (future Delete Label here)
        #   4. Navigation/relabel: Merge Into\u2026, then Prev / Next side-by-side
        # Fixed width keeps the table column the dominant horizontal element.
        self._sidebar_widget = QWidget()
        self._sidebar_widget.setFixedWidth(150)
        sidebar_layout = QVBoxLayout(self._sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(4)

        # Group 1: Routine triage
        sidebar_layout.addWidget(self._mark_good_btn)
        sidebar_layout.addWidget(self._mark_review_btn)
        sidebar_layout.addWidget(self._mark_bad_btn)
        sidebar_layout.addWidget(self._mark_garbage_btn)
        sidebar_layout.addWidget(_sidebar_sep())

        # Group 2: Final-confirm (visually separated to slow mis-clicks)
        sidebar_layout.addWidget(self._mark_curated_btn)
        sidebar_layout.addWidget(_sidebar_sep())

        # Group 3: Label management — create on top, destroy underneath
        # so create/destroy are vertically paired.  Delete Label uses
        # DESTRUCTIVE_BTN_STYLE; undo flows through canonical curation
        # undo (snapshot/mutate/commit) wired in Curation3DAnnotator.
        sidebar_layout.addWidget(self._new_label_btn)
        sidebar_layout.addWidget(self._delete_label_btn)
        sidebar_layout.addWidget(_sidebar_sep())

        # Group 4: Navigation/relabel
        sidebar_layout.addWidget(self._merge_btn)
        prev_next_row = QHBoxLayout()
        prev_next_row.setSpacing(2)
        prev_next_row.addWidget(self._prev_btn, 1)
        prev_next_row.addWidget(self._next_btn, 1)
        sidebar_layout.addLayout(prev_next_row)

        # Group 5: Batch tools
        sidebar_layout.addWidget(_sidebar_sep())
        sidebar_layout.addWidget(self._batch_refine_btn)

        # Pin sidebar contents to the top; empty space falls below.
        sidebar_layout.addStretch(1)

        # --- Middle split: table (LEFT, stretch=1) + sidebar (RIGHT, fixed) ---
        middle_split = QHBoxLayout()
        middle_split.setContentsMargins(0, 0, 0, 0)
        middle_split.setSpacing(6)
        middle_split.addWidget(self._table, 1)
        middle_split.addWidget(self._sidebar_widget, 0)
        layout.addLayout(middle_split, 1)

        # Discard All Garbage + Recalculate QC buttons removed 2026-05-10
        # per user UX feedback: garbage cleanup happens via the
        # commit-triage flow; QC recalculation is invoked elsewhere in
        # the workflow.  Navigator stays focused on per-label triage.

        # --- Bottom: display option widget creation (added to 2-row
        # layout further below).  Widget creation is unchanged from
        # pre-rethink; only the parent layout differs (now 2 rows of 3
        # widgets each in a collapsible "Display options" container).

        # "Dim Others" \u2014 toggle that dims non-flagged labels in subviewers
        # so the active triage list visually pops while surrounding tissue
        # stays faintly visible for context.  Visual state carries ON/OFF
        # via TOGGLE_BTN_STYLE accent fill \u2014 no text suffix needed.
        self._focused_mode_btn = QPushButton("Dim Others")
        self._focused_mode_btn.setCheckable(True)
        self._focused_mode_btn.setToolTip(
            "Dim labels that are NOT in your triage list \u2014 keep context "
            "visible but de-emphasised so flagged labels stand out."
        )
        self._focused_mode_btn.setStyleSheet(TOGGLE_BTN_STYLE)

        self._isolate_cb = QCheckBox("Isolate")
        self._isolate_cb.setToolTip(
            "Show only selected labels in all viewers "
            "(Ctrl+Click to multi-select)"
        )
        self._isolate_cb.setStyleSheet(_PLAIN_CHECKBOX_STYLE)

        self._auto_advance_cb = QCheckBox("Auto-advance")
        self._auto_advance_cb.setToolTip(
            "Automatically navigate to next label after triage action"
        )
        self._auto_advance_cb.setStyleSheet(_PLAIN_CHECKBOX_STYLE)

        # "Isolate while painting" —
        # opt-in toggle that auto-isolates whatever the brush's
        # selected_label currently is.  Tracks brush state via the
        # layer.events.selected_label subscription already installed
        # by set_label_layer for the label badge; forwards changes to
        # the annotator's on_paint_label_changed callback.  Off by
        # default; persists for the session.
        self._paint_isolate_cb = QCheckBox("Isolate while painting")
        self._paint_isolate_cb.setToolTip(
            "Show only the label currently selected by the brush.  "
            "Auto-follows when you click New Label, switch labels via "
            "the navigator, or paint + auto-increment.  Useful for "
            "isolating a newly-painted cell while you build it up.  "
            "Independent of the main Isolate checkbox — last "
            "isolation request wins."
        )
        self._paint_isolate_cb.setStyleSheet(_PLAIN_CHECKBOX_STYLE)

        # Right-click → select: when ON, right-clicking a label in any
        # subviewer focuses that label's row in this navigator (scroll
        # + select + trigger isolate-if-on).  Gated server-side by the
        # annotator's _right_click_navigate_allowed() guard so it
        # silently no-ops during any refinement op, while the merge
        # dialog is open, while in paint/erase mode, on background
        # clicks, etc.  Default ON, session-only (resets on relaunch).
        self._right_click_select_cb = QCheckBox("Right-click → select")
        self._right_click_select_cb.setChecked(True)
        self._right_click_select_cb.setToolTip(
            "Right-click a label in any subviewer to focus it in this "
            "list (scroll + select).  Disabled automatically during "
            "refinement ops (interp / split / smooth), while the Merge "
            "dialog is open, and while painting."
        )
        self._right_click_select_cb.setStyleSheet(_PLAIN_CHECKBOX_STYLE)

        # "Render Selected" button — Option B for multi-isolation perf.
        # Ctrl/Shift+click in the table only updates SELECTION; isolation
        # does not auto-fire for multi (the rebuild-per-click stack was
        # the root cause of the ctrl+click multi-iso lag).  Instead the
        # user accumulates the selection and clicks this button to fire
        # ONE render with the full set — same pattern as lasso.
        # Single click (no modifier) still auto-fires immediately so
        # the dominant single-label browsing UX is unchanged.
        # Visual state:
        #   disabled: 0/1 row selected, OR Isolate is off, OR
        #             selection == last_rendered (in sync)
        #   enabled accent: selection != last_rendered (waiting to render)
        self._render_selected_btn = QPushButton("Render Selected")
        self._render_selected_btn.setToolTip(
            "Render the current multi-selection in the viewer (one shot).\n"
            "Lights up when your selection differs from what's rendered.\n"
            "For single-label clicks, isolation auto-fires — you only "
            "need this for multi-select via Ctrl/Shift+click."
        )
        self._render_selected_btn.setStyleSheet(TOGGLE_BTN_STYLE)
        self._render_selected_btn.setCheckable(True)  # uses :checked accent
        self._render_selected_btn.setEnabled(False)

        # 2026-06-01 sidebar-right layout: the 6 display widgets are
        # preserved EXACTLY (object identities, signals, tooltips,
        # styles) but wrapped to 2 rows of 3 widgets each instead of
        # the original 1 row of 6.  Two-row wrap is the actual width-
        # reducing change at the bottom — the sidebar-right move alone
        # doesn't shrink the dock if display options still forces a
        # 700px-wide row.
        # Row 1 (Dim/Isolate-style):   Dim Others, Isolate, Paint-iso
        # Row 2 (selection/advance):   Render Sel, RC select, Auto-adv
        _display_separator = QFrame()
        _display_separator.setFrameShape(QFrame.HLine)
        _display_separator.setStyleSheet(SEPARATOR_LINE_STYLE)
        _display_separator.setFixedHeight(1)
        layout.addWidget(_display_separator)

        # Collapsible header: ▾ / ▸ chevron toggle.  Default expanded
        # so users see the toggles they currently use; can collapse to
        # save vertical space.
        self._display_options_toggle = QToolButton()
        self._display_options_toggle.setText("▾ Display options")
        self._display_options_toggle.setCheckable(True)
        self._display_options_toggle.setChecked(True)  # default expanded
        self._display_options_toggle.setToolTip(
            "Show / hide isolation + selection toggles "
            "(Dim Others / Isolate / Paint-iso / Render Selected / "
            "Right-click select / Auto-advance)."
        )
        self._display_options_toggle.setStyleSheet(
            "QToolButton {"
            "  background: transparent;"
            "  border: none;"
            "  color: #cccccc;"
            "  font-size: 11px;"
            "  padding: 2px 0px;"
            "  text-align: left;"
            "}"
            "QToolButton:hover { color: #ffffff; }"
        )
        layout.addWidget(self._display_options_toggle)

        # Container for the 6 display widgets in 2 rows of 3.
        # Visibility toggled by _display_options_toggle.  All 6 widget
        # objects, signal connections, tooltips, and styles are
        # unchanged — only the layout container differs.
        self._display_options_container = QWidget()
        _display_inner_layout = QVBoxLayout(self._display_options_container)
        _display_inner_layout.setContentsMargins(0, 0, 0, 0)
        _display_inner_layout.setSpacing(4)

        _display_row_1 = QHBoxLayout()
        _display_row_1.setSpacing(8)
        _display_row_1.addWidget(self._focused_mode_btn)
        _display_row_1.addWidget(self._isolate_cb)
        _display_row_1.addWidget(self._paint_isolate_cb)
        _display_row_1.addStretch(1)
        _display_inner_layout.addLayout(_display_row_1)

        _display_row_2 = QHBoxLayout()
        _display_row_2.setSpacing(8)
        _display_row_2.addWidget(self._render_selected_btn)
        _display_row_2.addWidget(self._right_click_select_cb)
        _display_row_2.addWidget(self._auto_advance_cb)
        _display_row_2.addStretch(1)
        _display_inner_layout.addLayout(_display_row_2)

        layout.addWidget(self._display_options_container)

        def _on_display_toggle_clicked(checked: bool):
            self._display_options_container.setVisible(bool(checked))
            self._display_options_toggle.setText(
                "▾ Display options" if checked else "▸ Display options"
            )
        self._display_options_toggle.toggled.connect(_on_display_toggle_clicked)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        self._table.cellClicked.connect(self._on_row_clicked)
        # Tier buttons all dispatch through _on_mark_tier_clicked(tier).
        # lambda captures the tier string at connect-time.
        self._mark_curated_btn.clicked.connect(
            lambda: self._on_mark_tier_clicked("curated")
        )
        self._mark_good_btn.clicked.connect(
            lambda: self._on_mark_tier_clicked("good")
        )
        self._mark_review_btn.clicked.connect(
            lambda: self._on_mark_tier_clicked("review")
        )
        self._mark_bad_btn.clicked.connect(
            lambda: self._on_mark_tier_clicked("bad")
        )
        self._mark_garbage_btn.clicked.connect(
            lambda: self._on_mark_tier_clicked("garbage")
        )
        self._merge_btn.clicked.connect(self._on_merge_clicked)
        self._new_label_btn.clicked.connect(self._on_new_label_clicked)
        self._delete_label_btn.clicked.connect(self._on_delete_label_clicked)
        self._prev_btn.clicked.connect(self._navigate_prev)
        self._next_btn.clicked.connect(self._navigate_next)
        self._focused_mode_btn.toggled.connect(self._on_focused_mode_toggled)
        self._isolate_cb.toggled.connect(self._on_isolate_toggled)
        self._paint_isolate_cb.toggled.connect(
            self._on_paint_isolate_toggled
        )
        self._sort_combo.currentIndexChanged.connect(lambda _: self._apply_sort())
        self._tier_filter_curated.toggled.connect(lambda _: self._apply_filter())
        self._tier_filter_good.toggled.connect(lambda _: self._apply_filter())
        self._tier_filter_review.toggled.connect(lambda _: self._apply_filter())
        self._tier_filter_bad.toggled.connect(lambda _: self._apply_filter())
        self._tier_filter_garbage.toggled.connect(lambda _: self._apply_filter())
        # Render Selected (Option B) — explicit fire for multi-selection
        self._render_selected_btn.clicked.connect(self._on_render_selected_clicked)
        # Selection changes update the Render button enable / accent state
        # A BOUND METHOD, never a lambda (2026-09-27): a lambda here is held
        # ONLY by PyQt's slot proxy on the selection model, and when that
        # model's wrapper sits in a garbage cycle (a dead _rebuild_table
        # frame kept alive by numba's first-call compile), the gc.collect()
        # napari runs on every layer removal CLEARS the lambda's closure
        # while PyQt still calls it -> access violation "line ??? in
        # <lambda>" (the 06-09 / 09-07 / 09-15 navigator dumps).
        sel_model = self._table.selectionModel()
        if sel_model is not None:
            sel_model.selectionChanged.connect(self._on_sel_model_changed)
        # Search field — text changes filter the table; ENTER commits;
        # ESC clears (handled in eventFilter / keyPressEvent below).
        self._search_field.textChanged.connect(
            lambda _: self._apply_filter()
        )
        self._search_field.returnPressed.connect(self._on_search_enter)
        # Arrow key navigation on the table
        self._table.installEventFilter(self)
        # ESC handling on the search field
        self._search_field.installEventFilter(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_callbacks(
        self,
        *,
        on_navigate=None,
        on_select_label=None,
        on_mark_tier=None,
        on_discard_garbage=None,
        on_merge=None,
        on_open_merge_dialog=None,
        on_focused_mode_changed=None,
        on_show_selected_label_changed=None,
        on_isolation_changed=None,
        on_scroll_subviewer=None,
        on_recalculate=None,
        on_new_label=None,
        on_delete_label=None,
        on_paint_isolation_toggled=None,
        on_paint_label_changed=None,
        on_batch_refine=None,
        on_selection_changed=None,
        is_op_active=None,
    ):
        """Inject all callbacks at once.

        ``on_mark_tier(label_id, tier_str) -> None`` is the unified
        tier-flag callback fired by the five tier-action buttons
        (Curated / Good / Review / Bad / Garbage).  Flag-only — no
        volume mutation.  Receiver should update the label's qc_tier
        in the QC features df and push the refreshed row list back to
        the navigator.

        ``on_discard_garbage(list[int]) -> None`` is the destructive
        path — receiver zeros all garbage-tier labels in the volume.
        This is the ONLY destructive button in the navigator (post-
        2026-05-10 redesign that replaced the per-label "Delete Label"
        button with the flag-only "Garbage" tier action).

        ``on_merge`` is the legacy single-label callback signature
        ``(keep_id, absorb_id) -> None``. As of the merge-dialog rewrite,
        the click handler no longer fires this callback — it routes through
        ``on_open_merge_dialog`` instead. The field is preserved for
        backward compatibility with existing test stubs and any
        external caller that still provides it. New code should set
        ``on_open_merge_dialog``.

        ``on_open_merge_dialog(keep_id) -> None`` is fired when the
        user clicks Merge Into in the navigator. The receiver is
        responsible for opening the floating MergeDialog (typically
        the curation_3d_widget which has access to the labels volume
        for neighbor discovery).
        """
        self._on_navigate_cb = on_navigate
        self._on_select_label_cb = on_select_label
        self._on_mark_tier_cb = on_mark_tier
        self._on_discard_garbage_cb = on_discard_garbage
        self._on_merge_cb = on_merge
        self._on_open_merge_dialog_cb = on_open_merge_dialog
        self._on_focused_mode_changed_cb = on_focused_mode_changed
        self._on_show_selected_label_changed_cb = on_show_selected_label_changed
        self._on_isolation_changed_cb = on_isolation_changed
        self._on_scroll_subviewer_cb = on_scroll_subviewer
        self._on_recalculate_cb = on_recalculate
        self._on_new_label_cb = on_new_label
        self._on_delete_label_cb = on_delete_label
        self._on_paint_isolation_toggled_cb = on_paint_isolation_toggled
        self._on_paint_label_changed_cb = on_paint_label_changed
        self._on_batch_refine_cb = on_batch_refine
        self._on_selection_changed_cb = on_selection_changed
        self._is_op_active_cb = is_op_active
        self._curation_debug(
            "navigator callbacks updated "
            f"navigate={bool(on_navigate)} "
            f"select={bool(on_select_label)} "
            f"isolate={bool(on_isolation_changed)}"
        )

    def set_label_layer(self, layer) -> None:
        """Set the napari Labels layer used to sample swatch colors
        and to drive the "Painting: N" badge in the editing header.

        The swatch column samples each row's actual on-screen color
        from the layer's colormap.  Keep this updated whenever the
        active labels layer changes (patch switch, annotator
        relaunch).  Pass ``None`` to fall back to neutral gray
        swatches (e.g. during teardown).

        Also subscribes to ``layer.events.selected_label`` so the
        Painting badge tracks the active brush label.  Disconnects
        from the previous layer first (clean swap on patch switch /
        annotator relaunch).  Safe to call with ``None``.
        """
        # Disconnect any previous selected_label subscription first
        # — applies to the OLD layer, not the new one.
        if self._selected_label_handler is not None:
            old_layer, old_cb = self._selected_label_handler
            try:
                old_layer.events.selected_label.disconnect(old_cb)
            except (AttributeError, ValueError, RuntimeError):
                # napari may already have torn down the layer's event
                # emitter (e.g., during dock close).  Idempotent —
                # nothing to do.
                pass
            self._selected_label_handler = None

        self._main_label_layer = layer

        if layer is not None:
            try:
                cb = self._on_layer_selected_label_changed
                layer.events.selected_label.connect(cb)
                self._selected_label_handler = (layer, cb)
                # Initialize from the layer's current state so the
                # badge is correct even before the first event fires.
                self._paint_label = int(
                    getattr(layer, "selected_label", 0) or 0
                )
            except (AttributeError, TypeError):
                # Layer doesn't expose events.selected_label (tests
                # using a partial fake, or a non-Labels layer).  No
                # subscription possible — leave _paint_label at 0.
                pass
        else:
            # set_label_layer(None) on teardown — reset to default.
            self._paint_label = 0

        # Refresh the header so the badge reflects the new state
        # immediately (esp. on patch switch where selected_label may
        # change without a separate event firing).
        try:
            self._update_editing_header()
        except (AttributeError, RuntimeError):
            # Header widget may not exist yet during early init.
            pass

    def _on_layer_selected_label_changed(self, _event=None) -> None:
        """Handler for the main layer's ``events.selected_label``.

        Reads the current ``selected_label`` from the tracked layer
        and refreshes the Painting badge.  Also forwards the new
        label to the annotator's ``on_paint_label_changed`` callback
        (used by the "Isolate while painting" toggle).

        Defensive — bails if the layer has been torn down between
        event emission and dispatch.
        """
        layer = self._main_label_layer
        if layer is None:
            return
        try:
            new_label = int(getattr(layer, "selected_label", 0) or 0)
        except (AttributeError, ValueError, TypeError):
            return
        self._paint_label = new_label
        try:
            self._update_editing_header()
        except (AttributeError, RuntimeError):
            pass
        # Forward to annotator for paint-isolation tracking.  The
        # annotator's callback no-ops if its toggle is OFF, so this
        # forward is cheap when paint-isolation isn't active.
        cb = getattr(self, "_on_paint_label_changed_cb", None)
        if cb is not None:
            try:
                cb(new_label)
            except Exception:
                pass

    def _on_paint_isolate_toggled(self, checked: bool) -> None:
        """Forward the "Isolate while painting" checkbox state to the
        annotator's ``on_paint_isolation_toggled`` callback
        companion, 2026-05-16).  Annotator manages the actual
        isolation state machine; navigator just owns the UI.
        """
        cb = getattr(self, "_on_paint_isolation_toggled_cb", None)
        if cb is None:
            return
        try:
            cb(bool(checked))
        except Exception:
            pass

    def set_paint_label(self, label_id: int) -> None:
        """Programmatic setter for the Painting badge.

        Normally the badge updates itself via the
        ``layer.events.selected_label`` subscription installed by
        ``set_label_layer``.  This setter exists for callers that
        don't have a real napari Labels layer (tests) or for
        explicit refresh after a set-then-emit race.
        """
        self._paint_label = max(0, int(label_id))
        try:
            self._update_editing_header()
        except (AttributeError, RuntimeError):
            pass

    def set_items(self, items: list) -> None:
        """Replace the label list (rebuild is coalesced — see below).

        Parameters
        ----------
        items : list of dict
            Each dict has keys: label_id, tier, centroid, area.

        Performance — single-flight rebuild coalescing (2026-05-10):
        Merge / split / discard-garbage cascades fire set_items 2-4
        times in rapid succession within the same event loop tick
        (one call per affected label_status / qc_tier update).  Each
        rebuild touches ~1k+ rows (custom-delegate paints, color
        swatches, tier badges) and runs in 50-80ms.  Letting them all
        run in series = 200-300ms perceived lag on every Accept.

        Fix: store the latest items, schedule a deferred rebuild via
        QTimer.singleShot(0, ...) only if one is not already pending.
        Subsequent set_items calls in the same tick just update
        _items — the pending rebuild then picks up the latest data
        and runs ONCE.

        Trade-off: a 1-tick (~0-16ms) delay between data update and
        visual refresh.  Imperceptible on accept flows; the rebuild
        still runs synchronously when no cascade is active (single
        scheduled-then-fired QTimer = effectively immediate).
        """
        self._items = list(items)
        self._schedule_rebuild()

    def remove_item(self, label_id: int) -> None:
        """Remove a single label from the table."""
        self._items = [it for it in self._items if it.get("label_id") != label_id]
        self._schedule_rebuild()

    def _schedule_rebuild(self) -> None:
        """Coalesce multiple data-update calls into one deferred rebuild.

        See ``set_items`` docstring for the rationale.  Pattern is
        idempotent: if a rebuild is already pending, this just returns
        — the pending callback will pick up the latest ``self._items``
        when it fires.
        """
        # Switch-safety: never arm a new deferred rebuild once the
        # navigator is being torn down — the timer would fire on a destroyed
        # widget.
        if getattr(self, "_destroyed", False):
            return
        if getattr(self, "_rebuild_pending", False):
            return
        self._rebuild_pending = True
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, self._do_pending_rebuild)

    def _do_pending_rebuild(self) -> None:
        """QTimer callback — clears the pending flag and runs the
        actual rebuild against whatever ``self._items`` holds now.

        Defensive: if the navigator widget was closed/destroyed between
        the schedule and the fire, the underlying Qt widget is
        sip-deleted and ANY attribute access raises RuntimeError
        ("wrapped C/C++ object has been deleted").  Catch + silent
        no-op so a closed-during-cascade dock doesn't crash the
        process.
        """
        # Switch-safety: the strongest guard.  ``_destroyed`` is a
        # plain Python attribute set in ``closeEvent`` BEFORE the C++ object
        # is destroyed, so reading it does not call into Qt C++ state.  This
        # catches the dangerous window the RuntimeError probe below misses —
        # a C++ object mid-destruction whose Python wrapper is not yet
        # sip-deleted (access would touch freed memory → Windows access
        # violation, the unclosed door on the B5 crash).
        try:
            if self._destroyed:
                return
        except (RuntimeError, AttributeError):
            return  # wrapper already deleted
        try:
            self._rebuild_pending = False
        except (RuntimeError, AttributeError):
            return  # widget destroyed
        try:
            # Cheap aliveness probe: accessing _table is what
            # _rebuild_table does first.  Probe it here to fail fast
            # with a clean RuntimeError rather than mid-rebuild.
            _ = self._table.rowCount()
        except (RuntimeError, AttributeError):
            print(
                "[RT-COALESCE] deferred rebuild SKIPPED — navigator "
                "widget destroyed between schedule and fire",
                flush=True,
            )
            return
        print(
            "[RT-COALESCE] deferred rebuild firing — picking up latest items",
            flush=True,
        )
        try:
            self._rebuild_table()
        except (RuntimeError, AttributeError) as _exc:
            print(
                f"[RT-COALESCE] rebuild raised mid-flight: {_exc!r} "
                f"— widget likely destroyed during rebuild",
                flush=True,
            )

    def closeEvent(self, event) -> None:
        """Switch-safety: mark the navigator as torn down.

        Fires both on a user dock-close (X) and on the controller's
        ``_destroy_dock`` (which calls ``widget.close()`` before
        ``deleteLater``).  Setting ``_destroyed`` here — while the Qt
        object is still fully alive — lets a pending deferred rebuild
        short-circuit cleanly when it fires on the next event-loop tick,
        instead of racing the C++ destruction.  Reset to ``False`` in
        ``reset_selection_ui`` when the dock is (re)used.
        """
        self._destroyed = True
        try:
            super().closeEvent(event)
        except (RuntimeError, AttributeError):
            pass

    def update_item_tier(self, label_id: int, new_tier: str) -> None:
        """Update a single item's tier display."""
        for it in self._items:
            if it.get("label_id") == label_id:
                it["tier"] = new_tier
                break
        self._schedule_rebuild()

    def current_label_id(self):
        """Return the label_id of the currently selected table row, or None."""
        row_idx = self._table.currentRow()
        if row_idx < 0:
            return None
        item = self._table.item(row_idx, _COL_ID)
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        if data:
            return data.get("label_id")
        return None

    def is_right_click_select_enabled(self) -> bool:
        """True iff the "Right-click → select" checkbox is checked.

        Polled by the annotator's subviewer right-click callback as
        one of the gating guards (see Curation3DAnnotator.
        _right_click_navigate_allowed).  When False the right-click
        is silently ignored.
        """
        cb = getattr(self, "_right_click_select_cb", None)
        return bool(cb is not None and cb.isChecked())

    def select_label(self, label_id: int) -> None:
        """Programmatically select a single label, scroll, and navigate."""
        for row_idx in range(self._table.rowCount()):
            item = self._table.item(row_idx, _COL_ID)
            if item is not None:
                data = item.data(Qt.UserRole)
                if data and data.get("label_id") == label_id:
                    # 2026-09-07 crash fix — mirror the 2026-06-09
                    # `_rebuild_table` / `select_labels` discipline:
                    # `selectRow` emits the selection MODEL's OWN
                    # `selectionChanged` (a separate QObject —
                    # `self._table.blockSignals` does NOT cover it), which
                    # re-enters `_on_table_selection_changed` from INSIDE
                    # Qt's selection mutation.  The 2026-09-07 access
                    # violation's innermost Python frame was exactly that
                    # lambda, called from this `selectRow`.  Block the
                    # selection-model signals around the mutation and
                    # re-fire the handler ONCE on the now-stable table —
                    # `_on_table_selection_changed` is idempotent (a
                    # render-button state refresh + a debounced Batch
                    # Refine count callback), so behaviour is preserved.
                    sel_model = self._table.selectionModel()
                    if sel_model is not None:
                        sel_model.blockSignals(True)
                    try:
                        self._table.selectRow(row_idx)
                    finally:
                        if sel_model is not None:
                            sel_model.blockSignals(False)
                    if sel_model is not None:
                        self._on_table_selection_changed()
                    self._table.scrollToItem(item)
                    self._on_row_clicked(row_idx)
                    return

    def scroll_to_label(self, label_id: int) -> None:
        """Scroll the table to ``label_id`` and visually select the row
        WITHOUT firing the on_row_clicked callbacks (navigate /
        isolation / select).

        Used for visual feedback when a row is added programmatically
        (e.g., auto-register on new-label paint) — the user gets
        "your new label landed HERE" without the navigator hijacking
        their current paint focus / view.

        Silently no-ops if ``label_id`` isn't in the current table
        (filtered out, not yet registered, etc.).
        """
        for row_idx in range(self._table.rowCount()):
            item = self._table.item(row_idx, _COL_ID)
            if item is None:
                continue
            data = item.data(Qt.UserRole)
            if data and data.get("label_id") == label_id:
                # blockSignals around setCurrentItem prevents the
                # cellClicked → _on_row_clicked cascade from firing
                # navigation / isolation callbacks.
                self._table.blockSignals(True)
                try:
                    self._table.scrollToItem(item)
                    self._table.setCurrentItem(item)
                finally:
                    self._table.blockSignals(False)
                return

    def select_labels(self, label_ids) -> None:
        """Programmatically select multiple labels and trigger isolation."""
        from qtpy.QtCore import QItemSelectionModel

        # B5 crash fix: a QC-plot lasso dispatch (deferred via
        # QTimer.singleShot(0)) can land mid patch-switch while the table
        # is being rebuilt.  If a coalesced rebuild is pending, the table
        # is about to be torn down + repopulated — the lasso selection is
        # STALE (belongs to the old patch).  Dropping it here is correct;
        # the rebuild repopulates the new patch's rows with no selection.
        if getattr(self, "_rebuild_pending", False):
            return

        label_id_set = set(label_ids)
        sel_model = self._table.selectionModel()

        # B5 crash fix: clearSelection() + select() loop + setCurrentIndex
        # below re-enter `_on_table_selection_changed` via the selection
        # MODEL's OWN signals (a separate QObject — `self._table.
        # blockSignals` does NOT cover them).  During a patch switch the
        # table may be rebuilding, and that re-entry on a half-built /
        # being-destroyed table is the Windows access-violation crash.
        # Mirror the `_rebuild_table` pattern (1997-2001): block the
        # selection-model signals around the mutation, then re-fire the
        # handler ONCE on the now-stable table to preserve the normal
        # behavior (render-button arm + Batch Refine "Selected" count).
        first_row = None
        sel_model.blockSignals(True)
        try:
            sel_model.clearSelection()

            for row_idx in range(self._table.rowCount()):
                item = self._table.item(row_idx, _COL_ID)
                if item is not None:
                    data = item.data(Qt.UserRole)
                    if data and data.get("label_id") in label_id_set:
                        idx = self._table.model().index(row_idx, 0)
                        sel_model.select(
                            idx,
                            QItemSelectionModel.Select | QItemSelectionModel.Rows,
                        )
                        if first_row is None:
                            first_row = row_idx

            if first_row is not None:
                # B4 fix: Use setCurrentIndex with NoUpdate to move the cursor
                # without clearing multi-selection. Qt's setCurrentCell()
                # internally fires ClearAndSelect which wipes the selection
                # we just built above.
                sel_model.setCurrentIndex(
                    self._table.model().index(first_row, 0),
                    QItemSelectionModel.NoUpdate,
                )
        finally:
            sel_model.blockSignals(False)

        if first_row is not None:
            self._table.scrollToItem(self._table.item(first_row, _COL_ID))
            # B4 fix: Don't call _on_row_clicked — it can disrupt multi-
            # selection in ExtendedSelection mode. Instead, fire the
            # navigate/select callbacks directly for the first row.
            item = self._table.item(first_row, _COL_ID)
            if item is not None:
                data = item.data(Qt.UserRole)
                if data:
                    label_id = data.get("label_id")
                    centroid = data.get("centroid")
                    self._update_editing_header()
                    if centroid is not None and self._on_navigate_cb:
                        try:
                            self._on_navigate_cb(label_id, centroid)
                        except Exception:
                            pass
                    if label_id is not None and self._on_select_label_cb:
                        try:
                            self._on_select_label_cb(label_id)
                        except Exception:
                            pass

        # B5 crash fix: re-fire the selection-changed handler ONCE on the
        # now-stable table — the blocked selection-model signal would
        # normally have armed the Render button + updated the Batch Refine
        # "Selected" count.  Guard the re-fire too: if a rebuild snuck in,
        # the table is unstable again and the rebuild will refresh state.
        if not getattr(self, "_rebuild_pending", False):
            self._on_table_selection_changed()

        # B5 fix: Fire isolation exactly once (not twice — _on_row_clicked
        # also called it before this fix).
        self._fire_isolation_if_active()

    def reset_selection_ui(self) -> None:
        """Silently reset selection and isolate UI state without firing callbacks.

        Used during close cleanup and dock reuse to clear stale state from
        a previous annotator session.  Unlike ``clear_selection()``, this
        method does NOT fire the isolation callback — it only touches the
        UI widgets so the dock is visually clean for the next session.
        """
        # Switch-safety: this is the canonical (re)use entry point
        # (launcher reuse path + close cleanup both call it).  Clearing
        # ``_destroyed`` here re-arms deferred rebuilds for a reused dock.
        # On the close path, ``closeEvent`` runs AFTER this (via the later
        # ``_destroy_dock`` → ``widget.close()``), so the flag still ends up
        # True for a torn-down dock.  These two assignments are first so they
        # always run even if the body below raises.  ``_rebuild_pending`` is
        # cleared too: a stale True (from a previous session whose timer was
        # dropped) would otherwise wedge ``_schedule_rebuild`` forever on a
        # reused dock.  Safe to clear here — this is not in an update cascade,
        # so it does not defeat the per-tick rebuild coalescing.
        self._destroyed = False
        self._rebuild_pending = False
        sel_model = self._table.selectionModel()
        if sel_model is not None:
            # B5 crash fix: block the selection-model's own signals around
            # clearSelection so it can't re-enter _on_table_selection_changed
            # on a table that's being torn down during a switch/teardown.
            sel_model.blockSignals(True)
            try:
                sel_model.clearSelection()
            finally:
                sel_model.blockSignals(False)
        self._table.setCurrentCell(-1, -1)
        self._editing_header.setText("")
        # Uncheck Isolate silently — blockSignals prevents _on_isolate_toggled
        self._isolate_cb.blockSignals(True)
        self._isolate_cb.setChecked(False)
        self._isolate_cb.blockSignals(False)

    def clear_selection(self) -> None:
        """Programmatically clear the current table selection and isolation."""
        sel_model = self._table.selectionModel()
        if sel_model is not None:
            # B5 crash fix: block the selection-model's own signals around
            # clearSelection so it can't re-enter _on_table_selection_changed
            # on a table that's being torn down during a switch/teardown.
            sel_model.blockSignals(True)
            try:
                sel_model.clearSelection()
            finally:
                sel_model.blockSignals(False)
        self._table.setCurrentCell(-1, -1)
        # F4: Uncheck Isolate silently (before firing callback to avoid
        # the toggled signal re-entering _on_isolate_toggled).
        self.set_isolate_checked(False)
        # R4.1: Also clear isolation state if active
        if self._on_isolation_changed_cb is not None:
            try:
                self._on_isolation_changed_cb(set())
            except Exception:
                pass

    def select_next_triage(self) -> bool:
        """Select the first visible label with a triage tier (review/bad/garbage).

        Iterates visible table rows from top, finds the first row whose tier
        is ``review``, ``bad``, or ``garbage``, selects it, and fires the
        navigation callback.

        Returns
        -------
        bool
            True if a triage label was found and selected, False otherwise.
        """
        _TRIAGE_TIERS = {"review", "bad", "garbage"}
        for row_idx in range(self._table.rowCount()):
            item = self._table.item(row_idx, _COL_ID)
            if item is None:
                continue
            data = item.data(Qt.UserRole)
            if data and data.get("tier") in _TRIAGE_TIERS:
                self._table.selectRow(row_idx)
                self._table.scrollToItem(item)
                self._on_row_clicked(row_idx)
                return True
        return False

    def selected_label_ids(self) -> set:
        """Return label_ids for ALL currently selected rows."""
        result = set()
        sel_model = self._table.selectionModel()
        if sel_model is None:
            return result
        for model_idx in sel_model.selectedRows():
            row_idx = model_idx.row()
            item = self._table.item(row_idx, _COL_ID)
            if item is not None:
                data = item.data(Qt.UserRole)
                if data and data.get("label_id") is not None:
                    result.add(data["label_id"])
        return result

    # ------------------------------------------------------------------
    # Private — isolation helpers
    # ------------------------------------------------------------------

    def _is_op_active(self) -> bool:
        """Predicate: is a refinement op currently staged on the annotator?

        Calls the injected ``_is_op_active_cb`` callback (wired by
        ``Curation3DAnnotator._wire_navigator_callbacks`` to read
        ``_refine_state.any_op_active``).  Returns False if no
        callback is set (e.g., during tests that don't wire op state).
        """
        if self._is_op_active_cb is None:
            return False
        try:
            return bool(self._is_op_active_cb())
        except Exception:
            return False

    def _show_op_active_status(self, msec: int = 2000) -> None:
        """Show a brief 'finish current op first' message in the editing
        header.  After ``msec`` milliseconds, restore the normal
        ``_update_editing_header`` content.

        Triggered when a user-initiated isolation action is blocked by
        the op-state gate.  Keeps the user informed
        instead of silently swallowing the click.
        """
        try:
            from qtpy.QtCore import QTimer
        except ImportError:
            return
        self._editing_header.setText(
            "⚠ Finish or reject current refinement op first"
        )
        if self._op_active_status_timer is None:
            self._op_active_status_timer = QTimer(self)
            self._op_active_status_timer.setSingleShot(True)
            self._op_active_status_timer.timeout.connect(
                self._update_editing_header
            )
        self._op_active_status_timer.start(msec)

    def _fire_isolation_if_active(self) -> None:
        """If isolate is ON, fire the isolation callback with all selected IDs.

        Skipped when ``_suppress_isolation_dispatch`` is True — set by the
        coordinator to prevent double-application when the controller
        already applied isolation colormaps.

        **Option B (multi-isolation perf):** auto-fire is restricted to
        single-row selections.  For 2+ selected rows, the user must
        click "Render Selected" to commit — this avoids the rebuild-
        per-click stack that made ctrl+click feel laggy.  The Render
        button glows when the current selection differs from
        ``_last_rendered_label_ids``.

        Programmatic paths (lasso → coordinator → ``select_labels()``)
        already set ``_suppress_isolation_dispatch=True`` before
        triggering selection, so the multi-row gate here doesn't
        affect lasso behavior.
        """
        import time as _time

        if getattr(self, "_suppress_isolation_dispatch", False):
            return
        if not self._isolate_cb.isChecked():
            # Isolation off: nothing to fire, but Render button state
            # may need to update (e.g., user just toggled isolate off).
            self._update_render_button_state()
            return

        # Op-state gate: if a refinement op is staged,
        # firing isolation here would mutate the proxy/colormap state
        # the op relies on for its preview.  Bail out + show status.
        if self._is_op_active():
            print(
                f"[ISO-GATE] _fire_isolation_if_active BLOCKED "
                f"selected_ids={sorted(int(x) for x in self.selected_label_ids())} "
                f"isolate_checked=True "
                f"op_active=True"
            )
            self._show_op_active_status()
            self._update_render_button_state()
            return

        ids = self.selected_label_ids()
        # Option B: only auto-fire for single-row selections.  For
        # multi-row, defer to the explicit Render Selected button.
        if len(ids) > 1:
            self._update_render_button_state()
            return

        if self._on_isolation_changed_cb:
            _t0 = _time.perf_counter()
            try:
                self._on_isolation_changed_cb(ids)
                # Track what's actually rendered — any subsequent
                # multi-select divergence will light up the Render
                # Selected button.
                self._last_rendered_label_ids = set(ids)
            except Exception:
                pass
            _dt = _time.perf_counter() - _t0
            if _dt > 0.03:
                print(
                    f"[ISO-TRACE] navigator fire_isolation "
                    f"dt={_dt:.3f}s"
                )
        self._update_render_button_state()

    def _on_render_selected_clicked(self) -> None:
        """Manually fire isolation with the current multi-selection.

        Option B's "commit" action — the user has accumulated rows via
        Ctrl/Shift+click and is ready to render them all at once.
        Does not respect ``_suppress_isolation_dispatch`` (this is an
        explicit user action; the suppression flag is for programmatic
        paths only).
        """
        import time as _time

        if not self._isolate_cb.isChecked():
            return
        # Op-state gate: same rationale as
        # `_fire_isolation_if_active` — explicit Render click during
        # a staged op would clobber the op's isolation state.
        if self._is_op_active():
            print(
                f"[ISO-GATE] _on_render_selected_clicked BLOCKED "
                f"selected_ids={sorted(int(x) for x in self.selected_label_ids())} "
                f"op_active=True"
            )
            self._show_op_active_status()
            return
        ids = self.selected_label_ids()
        if not ids:
            return
        if self._on_isolation_changed_cb:
            _t0 = _time.perf_counter()
            try:
                self._on_isolation_changed_cb(ids)
                self._last_rendered_label_ids = set(ids)
            except Exception:
                pass
            _dt = _time.perf_counter() - _t0
            if _dt > 0.03:
                print(
                    f"[ISO-TRACE] navigator render_selected "
                    f"dt={_dt:.3f}s n={len(ids)}"
                )
        self._update_render_button_state()

    def _on_sel_model_changed(self, _selected, _deselected) -> None:
        """``selectionChanged`` slot.  A bound method on purpose: it has no
        closure for the garbage collector to clear (see _connect_signals)."""
        self._on_table_selection_changed()

    def _on_table_selection_changed(self) -> None:
        """Table selection changed (manual Ctrl/Shift-click OR programmatic
        ``select_labels`` from a QC-plot lasso).  Updates the Render button
        and notifies any listener (the annotator refreshes the Batch Refine
        "Selected" count off this — it is the one signal BOTH selection
        gestures funnel through)."""
        self._update_render_button_state()
        cb = getattr(self, "_on_selection_changed_cb", None)
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    def _update_render_button_state(self) -> None:
        """Update the Render Selected button enabled / accent state.

        Disabled when:
          - 0 or 1 row selected (single-row auto-fires; nothing to do)
          - Isolate checkbox is off (rendering not active)
          - Selection == last_rendered (already in sync)

        Enabled with accent (TOGGLE_BTN_STYLE :checked state) when
        selection != last_rendered — i.e., the user has accumulated a
        new multi-selection waiting to render.
        """
        btn = getattr(self, "_render_selected_btn", None)
        if btn is None:
            return
        try:
            ids = self.selected_label_ids()
            isolate_on = self._isolate_cb.isChecked()
            in_sync = ids == self._last_rendered_label_ids
            should_enable = isolate_on and len(ids) >= 2 and not in_sync
            btn.setEnabled(should_enable)
            # Accent fill via :checked state (TOGGLE_BTN_STYLE).  We
            # toggle the checked state to drive the visual but
            # programmatically (no toggled signal — clicked is the
            # action signal).
            btn.blockSignals(True)
            btn.setChecked(should_enable)
            btn.blockSignals(False)
        except Exception:
            pass

    def _get_selected_row_indices(self) -> set:
        """Return the set of currently selected row indices."""
        sel_model = self._table.selectionModel()
        if sel_model is None:
            return set()
        return {idx.row() for idx in sel_model.selectedRows()}

    # ------------------------------------------------------------------
    # Private — UI updates
    # ------------------------------------------------------------------

    def _update_summary(self, tier_counts=None) -> None:
        """Update the summary label with tier counts."""
        if tier_counts is None:
            tier_counts = {
                "curated": 0, "good": 0, "review": 0, "bad": 0, "garbage": 0,
            }
            for row_idx in range(self._table.rowCount()):
                item = self._table.item(row_idx, _COL_ID)
                if item is None:
                    continue
                data = item.data(Qt.UserRole)
                if data:
                    t = data.get("tier", "review")
                    if t in tier_counts:
                        tier_counts[t] += 1

        total = sum(tier_counts.values())
        parts = []
        for t in ("curated", "good", "review", "bad", "garbage"):
            if tier_counts[t] > 0:
                color = TIER_COLORS.get(t, "#ffffff")
                parts.append(
                    f'<span style="color:{color}">{tier_counts[t]} {t}</span>'
                )
        if parts:
            summary = f"{total} labels: {', '.join(parts)}"
        else:
            summary = f"{total} labels"
        self._summary_label.setText(summary)
        # Discard All Garbage button removed 2026-05-10; the garbage
        # count is already shown in the summary text above.

    def _update_editing_header(self) -> None:
        """Update the editing header.

        Format: ``Painting: <N>  •  Editing: Label <lid> (<tier>)  Z=<z>``
        — Painting prefix tracks the active brush label
        (``selected_label`` on the main Labels layer); Editing suffix
        shows the row currently selected in the table.  The two can
        diverge (e.g., auto-increment bumps Painting to max+1 after
        an enclosed-fill stroke while the table still shows the
        previously-clicked row).

        When ``_paint_label == 0`` the prefix shows
        ``"Painting: background (0)"`` so the user can tell that
        left-click would erase rather than paint a real label.
        """
        # Painting prefix — always shown when a layer is connected.
        if self._paint_label == 0:
            paint_text = "Painting: background (0)"
        else:
            paint_text = f"Painting: {self._paint_label}"

        # Editing suffix — only when a row is selected in the table.
        editing_text = ""
        row_idx = self._table.currentRow()
        if row_idx >= 0:
            item = self._table.item(row_idx, _COL_ID)
            if item is not None:
                data = item.data(Qt.UserRole)
                if data:
                    lid = data.get("label_id", "?")
                    tier = data.get("tier", "?")
                    centroid = data.get("centroid", (0, 0, 0))
                    z = centroid[0]
                    sel_count = len(self._get_selected_row_indices())
                    if sel_count > 1:
                        editing_text = (
                            f"Editing: Label {lid} ({tier})  Z={z}  "
                            f"[{sel_count} selected]"
                        )
                    else:
                        editing_text = (
                            f"Editing: Label {lid} ({tier})  Z={z}"
                        )

        if editing_text:
            self._editing_header.setText(f"{paint_text}  •  {editing_text}")
        else:
            self._editing_header.setText(paint_text)

    # ------------------------------------------------------------------
    # Private — filter + sort
    # ------------------------------------------------------------------

    def _apply_filter(self) -> None:
        """Rebuild the table when filter checkboxes change."""
        self._rebuild_table()

    def _apply_sort(self) -> None:
        """Rebuild the table when sort combo changes."""
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        """Clear and repopulate the table from ``self._items``.

        Applies both tier filtering and sorting.  Avoids
        ``setRowHidden()`` which renders as white stripes under
        napari's dark Qt theme — instead we only insert rows that
        pass the active tier filter.
        """
        # Remember current multi-selection so we can restore after rebuild
        prev_selected_ids = self.selected_label_ids()
        prev_current_id = self.current_label_id()

        # --- Search filter (overrides tier filter when non-empty) ---
        search_ids = self._parse_search_ids()

        # --- Tier filter ---
        filter_map = {
            "curated": self._tier_filter_curated.isChecked(),
            "good": self._tier_filter_good.isChecked(),
            "review": self._tier_filter_review.isChecked(),
            "bad": self._tier_filter_bad.isChecked(),
            "garbage": self._tier_filter_garbage.isChecked(),
        }
        if search_ids:
            # Search overrides tier filter — user wants to find a
            # specific ID regardless of which tiers are toggled on.
            visible = [
                it for it in self._items
                if int(it.get("label_id", -1)) in search_ids
            ]
        else:
            visible = [
                it for it in self._items
                if filter_map.get(it.get("tier", "review"), True)
            ]

        # --- Sort ---
        sort_text = self._sort_combo.currentText()
        if sort_text == "Tier":
            visible.sort(
                key=lambda d: (
                    TIER_ORDER.get(d.get("tier", "review"), 99),
                    d.get("label_id", 0),
                )
            )
        elif sort_text == "Area (desc)":
            visible.sort(key=lambda d: d.get("area", 0), reverse=True)
        elif sort_text == "Z (asc)":
            visible.sort(
                key=lambda d: (
                    d.get("centroid", (0, 0, 0))[0],
                    d.get("label_id", 0),
                )
            )
        elif sort_text == "ID (asc)":
            visible.sort(key=lambda d: d.get("label_id", 0))
        elif sort_text == "Score (asc)":
            visible.sort(
                key=lambda d: (
                    d.get("qc_score") if d.get("qc_score") is not None else float("inf"),
                    d.get("label_id", 0),
                )
            )

        # --- Populate table (only visible rows) ---
        # Hot path — runs on every filter/sort/items-set toggle.  All
        # custom rendering (Color swatch + Tier pill) is delegate-based
        # (see _SwatchDelegate / _TierPillDelegate above) so per row we
        # only create QTableWidgetItem instances; the actual painting
        # happens lazily for visible cells only.
        #
        # 2026-05-10 crash fix: a three-part lockdown around
        # ``setRowCount(0)`` prevents the access violation observed
        # when a merge-dialog Accept fires while rapid checkbox-driven
        # isolation cascades have queued paint events targeting custom
        # delegate instances:
        #
        #   1. blockSignals(True): suppress currentChanged /
        #      selectionChanged signals during the rebuild — without
        #      this, the selection cascade can re-enter our handlers
        #      while items are being destroyed/created.
        #   2. selectionModel().clearSelection(): tear down the
        #      selection in a controlled order BEFORE setRowCount(0)
        #      starts destroying items.  Pending paint events that
        #      targeted "selected" item state then have nothing to
        #      paint against → no access violation.
        #   3. setUpdatesEnabled(False): coalesce visual repaints
        #      (existing optimization; kept).
        from napari_epicycle.ui_shell.theme import swatch_color_for_label
        import time as _navperf_time

        _navperf_total_t0 = _navperf_time.perf_counter()
        prev_row_count = self._table.rowCount()
        prev_sel_size = (
            len(self._table.selectionModel().selectedRows())
            if self._table.selectionModel() is not None
            else 0
        )
        print(
            f"[RT-FIX] _rebuild_table ENTER visible={len(visible)} "
            f"prev_rows={prev_row_count} prev_sel_size={prev_sel_size}",
            flush=True,
        )

        # [NAV-PERF] 2026-06-05 — batch the swatch colormap map ONCE for
        # every visible label_id instead of calling swatch_color_for_label
        # (a single-element cmap.map) per row.  Result is a
        # {label_id: QColor} dict the loop indexes; per-id color is
        # byte-for-byte identical (see _batch_swatch_colors docstring).  If
        # the batch helper fails for any id (returns missing key) we fall
        # back to the original per-row swatch_color_for_label call below.
        _swatch_build_dt = 0.0
        _swatch_color_map = {}
        try:
            _swatch_t0 = _navperf_time.perf_counter()
            _visible_ids = [int(d["label_id"]) for d in visible]
            _swatch_color_map = _batch_swatch_colors(
                self._main_label_layer, _visible_ids
            )
            _swatch_build_dt = (
                _navperf_time.perf_counter() - _swatch_t0
            ) * 1000.0
        except Exception:
            _swatch_color_map = {}

        self._table.setUpdatesEnabled(False)
        self._table.blockSignals(True)
        try:
            # Clear selection BEFORE destroying items.
            sel_model = self._table.selectionModel()
            if sel_model is not None:
                # Block the SELECTION MODEL's OWN signals.  `self._table.
                # blockSignals(True)` above does NOT cover them — the selection
                # model is a separate QObject — so without this, clearSelection
                # re-enters `_on_table_selection_changed` → `_on_selection_changed_cb`
                # mid-rebuild (the re-entry the 2026-05-10 lockdown comment claims
                # to suppress but never actually did).
                sel_model.blockSignals(True)
                try:
                    sel_model.clearSelection()
                finally:
                    sel_model.blockSignals(False)
            print("[RT-FIX] selection cleared, about to setRowCount(0)", flush=True)
            self._table.setRowCount(0)
            print(
                f"[RT-FIX] setRowCount(0) survived — building {len(visible)} new rows",
                flush=True,
            )
            # [NAV-PERF] 2026-06-05 — size the table ONCE up front instead
            # of insertRow() in the loop (which forces incremental model
            # resizing per row).  Rows are then filled with setItem().
            self._table.setRowCount(len(visible))

            # Italic font + subtle dark-amber background for UNTRIAGED
            # rows — labels with ``qc_tier == ""`` (the canonical
            # "newly painted, no tier decision yet" state set when a new label is painted).
            # Changed 2026-05-16 from feature_state="stale" criterion:
            # that signal fired for too many rows (split fragments,
            # post-interpolate-accept ghosts, etc., all get stale on
            # feature recompute) which visually overwhelmed.  The
            # qc_tier=="" criterion is narrower and matches the user's
            # mental model of "this is a new label I haven't curated."
            from qtpy.QtGui import QFont, QBrush
            _UNTRIAGED_FONT = QFont()
            _UNTRIAGED_FONT.setItalic(True)
            # Subtle dark-amber tint — distinct from the dark navigator
            # background (~#2b2b2b) but not loud.  Pairs with italic
            # for a "needs your attention" cue.
            _UNTRIAGED_BG = QBrush(QColor("#3a342a"))
            _UNTRIAGED_TOOLTIP = (
                "Newly painted label, not yet triaged.  Click Preview "
                "/ Accept on the Refinement controls to commit it as "
                "curated, or mark a tier directly from the buttons "
                "above."
            )

            for row_idx, data in enumerate(visible):
                label_id = data["label_id"]
                tier = data.get("tier", "review")
                centroid = data.get("centroid", (0, 0, 0))
                area = data.get("area", 0)
                z = centroid[0]
                # Untriaged = newly-painted labels with empty qc_tier
                # (set in controller._on_annotator_new_label_painted).
                # See the comment above _UNTRIAGED_FONT for why we
                # switched from the feature_state criterion.
                is_untriaged = str(tier) == ""

                # --- Color: stored as QColor on the item; delegate paints ---
                # [NAV-PERF] index the pre-batched color map; on a miss
                # (helper failed for this id) fall back to the original
                # per-row call so the swatch color stays identical.
                color = _swatch_color_map.get(int(label_id))
                if color is None:
                    color = swatch_color_for_label(
                        self._main_label_layer, int(label_id)
                    )
                color_item = QTableWidgetItem()
                color_item.setData(_ROLE_SWATCH_COLOR, color)
                color_item.setFlags(
                    color_item.flags() & ~Qt.ItemIsEditable
                )
                if is_untriaged:
                    color_item.setBackground(_UNTRIAGED_BG)
                    color_item.setToolTip(_UNTRIAGED_TOOLTIP)
                self._table.setItem(row_idx, _COL_COLOR, color_item)

                id_item = QTableWidgetItem()
                id_item.setData(Qt.DisplayRole, label_id)
                id_item.setData(Qt.UserRole, data)
                id_item.setForeground(QColor(TIER_COLORS.get(tier, "#ffffff")))
                if is_untriaged:
                    id_item.setFont(_UNTRIAGED_FONT)
                    id_item.setBackground(_UNTRIAGED_BG)
                    id_item.setToolTip(_UNTRIAGED_TOOLTIP)
                self._table.setItem(row_idx, _COL_ID, id_item)

                area_item = QTableWidgetItem()
                area_item.setData(Qt.DisplayRole, area)
                if is_untriaged:
                    area_item.setFont(_UNTRIAGED_FONT)
                    area_item.setBackground(_UNTRIAGED_BG)
                    area_item.setToolTip(_UNTRIAGED_TOOLTIP)
                self._table.setItem(row_idx, _COL_AREA, area_item)

                z_item = QTableWidgetItem()
                z_item.setData(Qt.DisplayRole, int(z))
                if is_untriaged:
                    z_item.setFont(_UNTRIAGED_FONT)
                    z_item.setBackground(_UNTRIAGED_BG)
                    z_item.setToolTip(_UNTRIAGED_TOOLTIP)
                self._table.setItem(row_idx, _COL_Z, z_item)

                score = data.get("qc_score")
                score_item = QTableWidgetItem()
                # Render em-dash ("—") for missing scores so de novo
                # rows (qc_score is None or NaN — features not yet
                # extracted) display cleanly instead of the literal
                # string "nan".  Controller already converts NaN → None
                # in _build_flagged_label_items.
                if score is not None:
                    import math
                    try:
                        score_f = float(score)
                        if math.isnan(score_f):
                            score_item.setData(Qt.DisplayRole, "—")
                        else:
                            score_item.setData(
                                Qt.DisplayRole, round(score_f, 3)
                            )
                    except (TypeError, ValueError):
                        score_item.setData(Qt.DisplayRole, "—")
                else:
                    score_item.setData(Qt.DisplayRole, "—")
                if is_untriaged:
                    score_item.setFont(_UNTRIAGED_FONT)
                    score_item.setBackground(_UNTRIAGED_BG)
                    score_item.setToolTip(_UNTRIAGED_TOOLTIP)
                self._table.setItem(row_idx, _COL_SCORE, score_item)

                # --- Tier: stored as text on the item; delegate paints pill ---
                # Untriaged state: tooltip + bg tint (delegate paints
                # the pill so italic font wouldn't render).  An empty
                # qc_tier shows no pill text by design — that's the
                # visual cue on the Tier column.
                tier_item = QTableWidgetItem()
                tier_item.setData(_ROLE_TIER_TEXT, tier)
                tier_item.setFlags(
                    tier_item.flags() & ~Qt.ItemIsEditable
                )
                if is_untriaged:
                    tier_item.setBackground(_UNTRIAGED_BG)
                    tier_item.setToolTip(_UNTRIAGED_TOOLTIP)
                self._table.setItem(row_idx, _COL_TIER, tier_item)
        finally:
            self._table.blockSignals(False)
            self._table.setUpdatesEnabled(True)
            print(
                f"[RT-FIX] _rebuild_table EXIT new_rows={self._table.rowCount()}",
                flush=True,
            )
            # [NAV-PERF] confirm the gain — this rebuild is fired via
            # QTimer.singleShot so it was previously un-timed.
            _navperf_total_dt = (
                _navperf_time.perf_counter() - _navperf_total_t0
            ) * 1000.0
            print(
                f"[NAV-PERF] _rebuild_table n_visible={len(visible)} "
                f"swatch_batch={_swatch_build_dt:.1f}ms "
                f"total={_navperf_total_dt:.1f}ms",
                flush=True,
            )

        # --- Update summary from ALL items (not just visible) ---
        tier_counts = {"curated": 0, "good": 0, "review": 0, "bad": 0, "garbage": 0}
        has_garbage = False
        for it in self._items:
            t = it.get("tier", "review")
            if t in tier_counts:
                tier_counts[t] += 1
            if t == "garbage":
                has_garbage = True
        self._update_summary(tier_counts)
        # Discard All Garbage + Recalculate QC buttons removed
        # 2026-05-10 — their setEnabled gating is no longer needed.

        # --- Restore multi-selection ---
        from qtpy.QtCore import QItemSelectionModel

        sel_model = self._table.selectionModel()
        first_match = None
        for row_idx in range(self._table.rowCount()):
            item = self._table.item(row_idx, _COL_ID)
            if item is not None:
                d = item.data(Qt.UserRole)
                if d and d.get("label_id") in prev_selected_ids:
                    idx = self._table.model().index(row_idx, 0)
                    sel_model.select(
                        idx,
                        QItemSelectionModel.Select | QItemSelectionModel.Rows,
                    )
                    if d.get("label_id") == prev_current_id:
                        first_match = row_idx
                    elif first_match is None:
                        first_match = row_idx
        if first_match is not None:
            self._table.setCurrentCell(first_match, 0)

        # Search auto-scroll: when there's a non-empty search and the
        # previous selection wasn't restored at a search-matched row,
        # scroll to the topmost row in the visible (filtered) table.
        # Cheap: just scrollToItem(row 0) since the table is already
        # filtered to search matches.
        if search_ids and self._table.rowCount() > 0:
            top_item = self._table.item(0, _COL_ID)
            if top_item is not None:
                try:
                    self._table.scrollToItem(top_item)
                except Exception:
                    pass

        self._update_editing_header()
        # Selection may have changed during rebuild; refresh Render
        # Selected button accent state.
        self._update_render_button_state()

    # ------------------------------------------------------------------
    # Private — navigation
    # ------------------------------------------------------------------

    def _on_row_clicked(self, row_idx: int, _col: int = 0) -> None:
        """Navigate to the clicked label's centroid."""
        import time as _time

        _t_total = _time.perf_counter()

        item = self._table.item(row_idx, _COL_ID)
        if item is None:
            return
        data = item.data(Qt.UserRole)
        if data is None:
            return

        label_id = data.get("label_id")
        centroid = data.get("centroid")
        self._curation_debug(
            "row click "
            f"row={row_idx} label={label_id} "
            f"selected_ids={sorted(int(x) for x in self.selected_label_ids())} "
            f"isolate_checked={self._isolate_cb.isChecked()} "
            f"cb_navigate={bool(self._on_navigate_cb)} "
            f"cb_select={bool(self._on_select_label_cb)} "
            f"cb_isolate={bool(self._on_isolation_changed_cb)}"
        )

        self._update_editing_header()

        _t0 = _time.perf_counter()
        if centroid is not None and self._on_navigate_cb:
            try:
                self._on_navigate_cb(label_id, centroid)
            except Exception:
                pass
        _t_nav = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        if label_id is not None and self._on_select_label_cb:
            try:
                self._on_select_label_cb(label_id)
            except Exception:
                pass
        _t_select = _time.perf_counter() - _t0

        # Fire isolation update with ALL selected labels
        _t0 = _time.perf_counter()
        self._fire_isolation_if_active()
        _t_isolate = _time.perf_counter() - _t0

        # Restore focus to the table — subviewer updates from callbacks
        # can steal focus, breaking subsequent Up/Down arrow navigation.
        self._table.setFocus()

        _dt_total = _time.perf_counter() - _t_total
        if _dt_total > 0.03:
            print(
                f"[ISO-TRACE] navigator row_click "
                f"label={label_id} "
                f"navigate={_t_nav:.3f}s "
                f"select={_t_select:.3f}s "
                f"isolate={_t_isolate:.3f}s "
                f"total={_dt_total:.3f}s"
            )

    def _navigate_next(self) -> None:
        """Move to next row. In multi-select, cycle within selected rows only."""
        total = self._table.rowCount()
        if total == 0:
            return
        selected = self._get_selected_row_indices()
        current = self._table.currentRow()
        if len(selected) > 1:
            sorted_sel = sorted(selected)
            after = [r for r in sorted_sel if r > current]
            next_row = after[0] if after else sorted_sel[0]
            self._table.setCurrentCell(next_row, 0)
        else:
            next_row = (current + 1) % total
            self._table.selectRow(next_row)
        self._on_row_clicked(next_row)

    def _navigate_prev(self) -> None:
        """Move to previous row. In multi-select, cycle within selected rows only."""
        total = self._table.rowCount()
        if total == 0:
            return
        selected = self._get_selected_row_indices()
        current = self._table.currentRow()
        if current < 0:
            current = 0
        if len(selected) > 1:
            sorted_sel = sorted(selected)
            before = [r for r in sorted_sel if r < current]
            prev_row = before[-1] if before else sorted_sel[-1]
            self._table.setCurrentCell(prev_row, 0)
        else:
            prev_row = (current - 1) % total
            self._table.selectRow(prev_row)
        self._on_row_clicked(prev_row)

    def _advance_to_next_visible(self) -> None:
        """Auto-advance to the next actionable row after a triage action.

        Skips rows with tier in {good, curated, garbage} since those are
        either resolved or deferred — the user wants the next *fresh*
        review/bad item.  Wraps around once and stays put if nothing
        actionable remains.

        Pre-fix this method assumed the just-actioned row had been
        eager-removed, so it just clamped currentRow() to the new
        bounds.  Now that we keep the row in the table (tier badge
        flipped), we must skip past it explicitly.
        """
        total = self._table.rowCount()
        if total == 0:
            self._update_editing_header()
            return

        actionable_tiers = {"review", "bad"}
        cur = max(self._table.currentRow(), 0)

        # Walk forward from current+1, wrapping once.  Stop at cur to
        # avoid infinite loop if no actionable rows are left.
        for offset in range(1, total + 1):
            idx = (cur + offset) % total
            if idx == cur and offset > 0:
                # Wrapped fully without finding — stay put.
                break
            item = self._table.item(idx, _COL_ID)
            if item is None:
                continue
            data = item.data(Qt.UserRole)
            if data and data.get("tier") in actionable_tiers:
                self._table.selectRow(idx)
                self._on_row_clicked(idx)
                return

        # No more actionable rows — keep selection where it is and
        # refresh the editing header so the count is current.
        self._update_editing_header()

    # ------------------------------------------------------------------
    # Private — triage actions
    # ------------------------------------------------------------------

    def _on_mark_tier_clicked(self, tier: str) -> None:
        """Set the current label's tier via ``_on_mark_tier_cb`` and
        optionally advance.

        Flag-only — the receiver (typically the controller) updates the
        label's ``qc_tier`` in the QC features df and pushes the full
        refreshed row list back to the navigator.  Per the post-2026-
        05-05 docstring on the previous mark-good handler: we trust the
        controller to push back the new state and DO NOT eager-remove
        the row here.  Auto-advance (if enabled) moves to the next
        actionable row in the filtered/sorted view.

        Replaces the previous ``_on_mark_good_clicked`` +
        ``_on_delete_clicked`` handlers — both used the same pattern,
        differing only in which callback to invoke and which tier to
        flag.  The 5 tier buttons now share this single dispatcher with
        the tier string captured in the connect-time lambda.
        """
        lid = self.current_label_id()
        if lid is None:
            return
        if self._on_mark_tier_cb:
            try:
                self._on_mark_tier_cb(lid, tier)
            except Exception:
                pass
        if self._auto_advance_cb.isChecked():
            self._advance_to_next_visible()

    def _on_merge_clicked(self) -> None:
        """Open the floating Merge Dialog for the currently selected label.

        UX direction: the selected label is the KEEP (target). The dialog
        surfaces adjacent labels (via graduated dilation) and lets the user
        check which to absorb. Replaces the previous QInputDialog flow
        which required typing the target label ID by hand.

        Row removal + auto-advance happen later, in the controller's
        post-merge callback (after the user clicks Accept in the dialog),
        so a Cancel still leaves the navigator state intact.
        """
        keep_id = self.current_label_id()
        if keep_id is None:
            return
        if self._on_open_merge_dialog_cb is not None:
            try:
                self._on_open_merge_dialog_cb(keep_id)
            except Exception:
                pass

    def _on_new_label_clicked(self) -> None:
        """User clicked the New Label button — bump selected_label
        to max(volume) + 1 across all viewers.

        Delegates to the annotator (via ``on_new_label`` callback) so
        the navigator stays decoupled from the layer state.  No-op
        if no callback is wired or if the callback raises.
        """
        cb = self._on_new_label_cb
        if cb is None:
            return
        try:
            cb()
        except Exception:
            pass

    def _on_delete_label_clicked(self) -> None:
        """User clicked the Delete Label button — destructively zero
        the selected label in the volume.

        Delegates to the annotator (via ``on_delete_label`` callback)
        which performs the snapshot + mutation + sync + commit-to-undo
        chain.  Auto-advance (if enabled) moves to the next visible
        row after the destructive action — same UX as the tier-flag
        buttons (Mark Good / Bad / etc).  No-op if no callback is
        wired or if the callback raises.
        """
        lid = self.current_label_id()
        if lid is None:
            return
        cb = self._on_delete_label_cb
        if cb is None:
            return
        try:
            cb(lid)
        except Exception:
            pass
        if self._auto_advance_cb.isChecked():
            self._advance_to_next_visible()

    def _on_batch_refine_clicked(self) -> None:
        """User clicked Batch Refine — ask the annotator to open the
        floating Batch Refine dialog.  The annotator gates the open on a
        clean refinement state (``can_start_batch_refine``); no-op if no
        callback is wired or it raises."""
        cb = getattr(self, "_on_batch_refine_cb", None)
        if cb is None:
            return
        try:
            cb()
        except Exception:
            pass

    def set_label_action_buttons_enabled(self, enabled: bool) -> None:
        """Enable / disable BOTH the + New Label and Delete Label
        buttons together.

        Called by the annotator at every refinement-op state transition
        (and at uncommitted-paint-stroke / merge-dialog open / close)
        so the buttons visually disable instead of failing silently or
        racing the staged op.  When ``enabled`` is False the tooltips
        update too so the user knows why.

        Pairs with ``set_merge_enabled`` which already gates the Merge
        Into button on the same predicate.
        """
        enabled = bool(enabled)
        self._new_label_btn.setEnabled(enabled)
        self._delete_label_btn.setEnabled(enabled)
        if enabled:
            self._new_label_btn.setToolTip(
                "Set selected_label to max(volume) + 1 — a fresh label "
                "ready to paint into a new cell.  The row is registered "
                "in this list only after you paint with it."
            )
            self._delete_label_btn.setToolTip(
                "Zero the selected label in the volume (destructive).  "
                "Use the controller's Undo to restore.  Disabled while "
                "a refinement op (interpolation / split / smooth / "
                "merge) or uncommitted paint stroke is in flight."
            )
        else:
            disabled_msg = (
                "Disabled while a refinement op is active "
                "(interpolation / split / smooth / merge dialog) or "
                "an uncommitted paint stroke is in flight.  Accept / "
                "Reject / Clear Edits / close dialog to re-enable."
            )
            self._new_label_btn.setToolTip(disabled_msg)
            self._delete_label_btn.setToolTip(disabled_msg)

    # ``_on_discard_garbage_clicked`` and ``_on_recalc_clicked`` removed
    # 2026-05-10 per user UX feedback — the corresponding buttons are
    # gone.  The ``_on_discard_garbage_cb`` and ``_on_recalculate_cb``
    # fields remain in set_callbacks for back-compat; the controller
    # may still invoke discard-garbage via the commit-triage flow which
    # bypasses the navigator entirely.

    # ------------------------------------------------------------------
    # Private — focused mode
    # ------------------------------------------------------------------

    def _on_focused_mode_toggled(self, enabled: bool) -> None:
        """Notify the focused-mode callback.

        Visual state (button accent fill when ON, neutral when OFF) is
        carried by ``TOGGLE_BTN_STYLE`` via the button's ``checked``
        pseudo-state — no text suffix needed.  The button label stays
        ``"Dim Others"`` in both states.
        """
        if self._on_focused_mode_changed_cb:
            try:
                self._on_focused_mode_changed_cb(enabled)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private — isolate on select
    # ------------------------------------------------------------------

    def set_isolate_checked(self, checked: bool) -> None:
        """Set the Isolate checkbox state without firing callbacks.

        Used by the coordinator to sync the checkbox state after lasso
        selection or clear events.  Uses ``blockSignals`` to suppress
        the ``toggled`` signal that would otherwise trigger
        ``_on_isolate_toggled``.
        """
        self._isolate_cb.blockSignals(True)
        self._isolate_cb.setChecked(checked)
        self._isolate_cb.blockSignals(False)

    def set_merge_enabled(self, enabled: bool) -> None:
        """Enable/disable the Merge Into button.

        Called by the annotator at every refinement-op state transition
        so the button visually disables (rather than failing silently
        on click) while interp/split/smooth is staged.  Tooltip updates
        too so the user knows why.
        """
        self._merge_btn.setEnabled(bool(enabled))
        if enabled:
            self._merge_btn.setToolTip(
                "Merge selected label into another label (relabel)"
            )
        else:
            self._merge_btn.setToolTip(
                "Disabled while a refinement op is active "
                "(interpolation / split / smooth). "
                "Accept or Reject the current op first."
            )

    def _on_isolate_toggled(self, checked: bool) -> None:
        """Toggle multi-label isolation via colormap."""
        self._curation_debug(
            "isolate toggled "
            f"checked={checked} "
            f"selected_ids={sorted(int(x) for x in self.selected_label_ids())} "
            f"cb_isolate={bool(self._on_isolation_changed_cb)}"
        )
        # Op-state gate: refuse the toggle while a
        # refinement op is staged, revert the checkbox to its prior
        # state silently, and show a transient status message.  The
        # checkbox stays at the value the user TRIED to set until we
        # blockSignals + revert, so the revert is reading "what was
        # this before the user clicked" = `not checked`.
        if self._is_op_active():
            print(
                f"[ISO-GATE] _on_isolate_toggled BLOCKED "
                f"user_tried_to_set={checked} "
                f"reverting_to={not checked} "
                f"op_active=True"
            )
            self._isolate_cb.blockSignals(True)
            self._isolate_cb.setChecked(not checked)
            self._isolate_cb.blockSignals(False)
            self._show_op_active_status()
            self._update_render_button_state()
            return
        if self._on_isolation_changed_cb:
            try:
                ids = self.selected_label_ids() if checked else set()
                self._on_isolation_changed_cb(ids)
                # Track render state for the Render Selected button.
                self._last_rendered_label_ids = set(ids)
            except Exception:
                pass
        elif self._on_show_selected_label_changed_cb:
            # Legacy fallback
            try:
                self._on_show_selected_label_changed_cb(checked)
            except Exception:
                pass
        # Refresh Render Selected button state (becomes disabled when
        # Isolate is off; may light up if Isolate goes on with multi-
        # selection already present).
        self._update_render_button_state()

    # ------------------------------------------------------------------
    # Event filter — arrow key navigation on the table
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        """Intercept arrow keys on the table + ESC on the search field.

        Table:
          Up/Down → navigate labels (prev/next).
          Left/Right → scroll the hovered subviewer via callback.
        Search field:
          ESC → clear the search.
        """
        if obj is self._table and event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Down:
                self._navigate_next()
                return True
            elif key == Qt.Key_Up:
                self._navigate_prev()
                return True
            elif key in (Qt.Key_Left, Qt.Key_Right):
                if self._on_scroll_subviewer_cb:
                    direction = "right" if key == Qt.Key_Right else "left"
                    try:
                        self._on_scroll_subviewer_cb(direction)
                    except Exception:
                        pass
                return True  # always consume to prevent table column navigation
        if (
            obj is getattr(self, "_search_field", None)
            and event.type() == QEvent.KeyPress
            and event.key() == Qt.Key_Escape
        ):
            self._search_field.clear()  # textChanged → _apply_filter
            return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Private — search helpers
    # ------------------------------------------------------------------

    def _parse_search_ids(self) -> set:
        """Parse the search field into a set of label IDs.

        Comma-separated, whitespace tolerant.  Non-numeric tokens are
        ignored silently (so "47abc, 12" yields ``{12}``).  Returns
        ``set()`` for an empty / whitespace-only field.
        """
        try:
            raw = self._search_field.text() if self._search_field else ""
        except Exception:
            return set()
        if not raw or not raw.strip():
            return set()
        out = set()
        for tok in raw.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.add(int(tok))
            except ValueError:
                # Silently skip non-numeric tokens.
                pass
        return out

    def _on_search_enter(self) -> None:
        """ENTER pressed in the search field.

        Single-ID match: navigate to that label (single-fire isolation
        if Isolate is on, same as clicking the row).

        Multi-ID match: select all matched rows (then user clicks
        Render Selected to fire isolation as one render).

        No matches: no-op.
        """
        ids = self._parse_search_ids()
        if not ids:
            return
        # Find which IDs are actually present in the visible table
        # (search composes with row visibility — though search overrides
        # tier filter, so all parsed IDs should be visible if they
        # exist in self._items at all).
        present_rows = {}  # label_id -> row_idx
        for row_idx in range(self._table.rowCount()):
            item = self._table.item(row_idx, _COL_ID)
            if item is None:
                continue
            data = item.data(Qt.UserRole)
            if data:
                lid = data.get("label_id")
                if lid in ids:
                    present_rows[lid] = row_idx
        if not present_rows:
            return
        if len(present_rows) == 1:
            # Single match → navigate (full single-click semantics).
            row_idx = next(iter(present_rows.values()))
            self._on_row_clicked(row_idx)
        else:
            # Multi match → select all matched rows.  Auto-fire is
            # gated by _fire_isolation_if_active (multi-row → no fire),
            # so this just sets up the selection for the user to commit
            # via Render Selected.
            from qtpy.QtCore import QItemSelectionModel
            sel_model = self._table.selectionModel()
            if sel_model is None:
                return
            # B5 crash fix: block the selection-model's own signals around
            # clearSelection so it can't re-enter _on_table_selection_changed
            # on a table being torn down during a switch.  The select() loop
            # below re-fires selectionChanged with the real multi-selection,
            # and the explicit _update_* calls at the end re-arm the UI.
            sel_model.blockSignals(True)
            try:
                sel_model.clearSelection()
            finally:
                sel_model.blockSignals(False)
            first_row = None
            for lid, row_idx in present_rows.items():
                idx = self._table.model().index(row_idx, 0)
                sel_model.select(
                    idx,
                    QItemSelectionModel.Select | QItemSelectionModel.Rows,
                )
                if first_row is None:
                    first_row = row_idx
            if first_row is not None:
                self._table.setCurrentCell(first_row, 0)
                self._table.scrollToItem(self._table.item(first_row, _COL_ID))
            self._update_editing_header()
            self._update_render_button_state()

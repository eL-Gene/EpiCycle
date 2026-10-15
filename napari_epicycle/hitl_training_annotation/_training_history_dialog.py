"""Phase B + C — Themed non-modal dialog showing training history per
bucket.

Pulls per-round data via ``_training_history.read_full_history`` and
displays it as two parallel lifelines per bucket (data + model).  The
data line is always continuous (pairs accumulate); the model line
breaks when ``init_mode != "inherited"`` (user restarted training).


Non-modal: ``setModal(False) + show()`` — user can keep dialog open
while interacting with the main napari window.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFont, QPainter, QPen
from qtpy.QtWidgets import (
    QDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QTextEdit, QVBoxLayout, QWidget,
)

from ._training_history import (
    RoundHistoryEntry, read_full_history, has_continuous_lineage_from_prior,
)


# ---------------------------------------------------------------------------
# Pure-Python rendering helpers (unit-testable without Qt instantiation)
# ---------------------------------------------------------------------------

def format_round_summary(entry: RoundHistoryEntry) -> str:
    """One-line text summary of a round.  Used as a tooltip / fallback
    text rendering in Phase B + in tooltips in Phase C."""
    parts = [f"round_{entry.round_idx:02d}"]
    init_disp = entry.init_mode
    if entry.init_path:
        # For non-inherited modes, the path is meaningful.
        import os
        path_str = entry.init_path
        # Shorten long paths to basename for display.
        if "/" in path_str or "\\" in path_str:
            path_str = os.path.basename(path_str.rstrip("/\\"))
        init_disp = f"{entry.init_mode}: {path_str}" if entry.init_mode != "inherited" else f"inherited ({path_str})"
    parts.append(init_disp)
    parts.append(f"ch{entry.channels_used}")
    parts.append(f"{entry.n_train_pairs} pairs")
    parts.append(f"ep={entry.n_epochs}")
    if entry.finished_at:
        # First 16 chars: "YYYY-MM-DDTHH:MM"
        parts.append(entry.finished_at[:16].replace("T", " "))
    return "  ·  ".join(parts)


def format_history_for_bucket(
    entries: List[RoundHistoryEntry], bucket: str,
) -> str:
    """Render a bucket's history as a multi-line text block.

    Used in Phase B (placeholder rendering before the QPainter
    timeline lands).  Also used as the accessible-text fallback /
    clipboard export from the timeline widget.
    """
    if not entries:
        return f"({bucket}: no training rounds yet)"
    lineage = has_continuous_lineage_from_prior(entries)
    lines: List[str] = []
    for idx, entry in enumerate(entries):
        line = format_round_summary(entry)
        # Mark broken lineage with a "↺ reset" tag.
        if idx > 0 and not lineage[idx]:
            line = "↺ reset  " + line
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase C — Parallel-line timeline widget
# ---------------------------------------------------------------------------

class _TimelineWidget(QWidget):
    """Custom-painted parallel-line timeline for one bucket.

    Layout (height ≈ 100 px):

        Round:    00      01      02      03

        Data:     ●───────●───────●───────●     (always solid)
        Model:    ●───────●       ●───────●     (broken on init != inherited)

    Emits ``round_clicked(int)`` when user clicks a dot.
    """
    round_clicked = Signal(int)

    # Layout constants
    LEFT_PADDING = 80           # space for "Data:" / "Model:" labels
    RIGHT_PADDING = 20
    TOP_PADDING = 18            # space for round-number axis
    DOT_RADIUS = 6
    DATA_LINE_Y_OFFSET = 25     # below the top padding
    MODEL_LINE_Y_OFFSET = 60

    # Colors
    AMBER = QColor("#d4a843")
    DIM_GRAY = QColor("#666666")
    TEXT = QColor("#cccccc")
    LABEL_TEXT = QColor("#888888")
    BG = QColor("transparent")

    def __init__(self, entries: List[RoundHistoryEntry], parent=None):
        super().__init__(parent)
        self._entries = list(entries)
        self._lineage = has_continuous_lineage_from_prior(self._entries)
        self.setMinimumHeight(110)
        self.setMinimumWidth(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

    def set_entries(self, entries: List[RoundHistoryEntry]) -> None:
        self._entries = list(entries)
        self._lineage = has_continuous_lineage_from_prior(self._entries)
        self.update()

    # ------------------------------------------------------------------
    # Geometry helpers (testable separately from paintEvent)
    # ------------------------------------------------------------------

    def _dot_x(self, idx: int) -> int:
        """X-pixel center of the dot for round at index ``idx``."""
        if not self._entries:
            return self.LEFT_PADDING
        usable_w = max(1, self.width() - self.LEFT_PADDING - self.RIGHT_PADDING)
        if len(self._entries) == 1:
            return self.LEFT_PADDING + usable_w // 2
        step = usable_w / max(1, len(self._entries) - 1)
        return int(self.LEFT_PADDING + idx * step)

    def _hit_test(self, mouse_x: int, mouse_y: int) -> int:
        """Return the round_idx the mouse is over (by hit-test radius),
        or -1 if no dot was hit.
        """
        data_y = self.TOP_PADDING + self.DATA_LINE_Y_OFFSET
        model_y = self.TOP_PADDING + self.MODEL_LINE_Y_OFFSET
        for i in range(len(self._entries)):
            dx = self._dot_x(i)
            if abs(mouse_x - dx) > self.DOT_RADIUS + 2:
                continue
            for dy in (data_y, model_y):
                if abs(mouse_y - dy) <= self.DOT_RADIUS + 2:
                    return i
        return -1

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event):
        if not self._entries:
            painter = QPainter(self)
            painter.setPen(self.LABEL_TEXT)
            painter.drawText(self.rect(), Qt.AlignCenter, "(no rounds yet)")
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        data_y = self.TOP_PADDING + self.DATA_LINE_Y_OFFSET
        model_y = self.TOP_PADDING + self.MODEL_LINE_Y_OFFSET

        # Row labels.
        label_font = QFont()
        label_font.setPointSize(9)
        painter.setFont(label_font)
        painter.setPen(self.LABEL_TEXT)
        painter.drawText(8, self.TOP_PADDING - 2, "Round:")
        painter.drawText(8, data_y + 4, "Data:")
        painter.drawText(8, model_y + 4, "Model:")

        # Round numbers along the top.
        for i in range(len(self._entries)):
            dx = self._dot_x(i)
            painter.setPen(self.TEXT)
            painter.drawText(dx - 12, self.TOP_PADDING - 2, f"{i:02d}")

        # Data lifeline — always continuous solid amber line.
        data_pen = QPen(self.AMBER, 2)
        painter.setPen(data_pen)
        if len(self._entries) > 1:
            for i in range(len(self._entries) - 1):
                x1 = self._dot_x(i) + self.DOT_RADIUS
                x2 = self._dot_x(i + 1) - self.DOT_RADIUS
                painter.drawLine(x1, data_y, x2, data_y)

        # Model lifeline — broken where lineage[i] == False (i > 0).
        model_pen = QPen(self.AMBER, 2)
        painter.setPen(model_pen)
        for i in range(1, len(self._entries)):
            if not self._lineage[i]:
                continue  # skip broken segment
            x1 = self._dot_x(i - 1) + self.DOT_RADIUS
            x2 = self._dot_x(i) - self.DOT_RADIUS
            painter.drawLine(x1, model_y, x2, model_y)

        # Dots — data row (amber filled).
        painter.setPen(QPen(self.AMBER, 1))
        painter.setBrush(self.AMBER)
        for i in range(len(self._entries)):
            painter.drawEllipse(
                self._dot_x(i) - self.DOT_RADIUS, data_y - self.DOT_RADIUS,
                self.DOT_RADIUS * 2, self.DOT_RADIUS * 2,
            )

        # Dots — model row.  Filled amber when continuous incoming link;
        # outlined amber (hollow) when this round starts a new lineage.
        for i in range(len(self._entries)):
            is_lineage_start = (i == 0) or (not self._lineage[i])
            painter.setPen(QPen(self.AMBER, 2))
            if is_lineage_start and i > 0:
                # Hollow dot — marks a model-lineage break.
                painter.setBrush(Qt.NoBrush)
            else:
                painter.setBrush(self.AMBER)
            painter.drawEllipse(
                self._dot_x(i) - self.DOT_RADIUS, model_y - self.DOT_RADIUS,
                self.DOT_RADIUS * 2, self.DOT_RADIUS * 2,
            )

        # Pair counts under data dots.
        small_font = QFont()
        small_font.setPointSize(8)
        painter.setFont(small_font)
        painter.setPen(self.LABEL_TEXT)
        for i, entry in enumerate(self._entries):
            painter.drawText(
                self._dot_x(i) - 12, data_y + 18, f"{entry.n_train_pairs}p"
            )

        # Init labels under model dots (only when broken, shows source).
        for i, entry in enumerate(self._entries):
            if i == 0 or not self._lineage[i]:
                label = entry.init_mode
                if entry.init_path:
                    import os
                    label = os.path.basename(entry.init_path.rstrip("/\\")) or entry.init_mode
                painter.drawText(
                    self._dot_x(i) - 30, model_y + 18, label[:14],
                )

        painter.end()

    def mouseMoveEvent(self, event):
        idx = self._hit_test(event.x(), event.y())
        if idx >= 0:
            entry = self._entries[idx]
            self.setToolTip(format_round_summary(entry))
        else:
            self.setToolTip("")
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            idx = self._hit_test(event.x(), event.y())
            if idx >= 0:
                self.round_clicked.emit(self._entries[idx].round_idx)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Phase B — Themed dialog shell
# ---------------------------------------------------------------------------

class TrainingHistoryDialog(QDialog):
    """Non-modal dialog showing training history per bucket.

    Layout:
        Title bar
        ┌─ Lateral ─────────┐
        │  timeline widget   │
        │  detail panel      │
        └────────────────────┘
        ┌─ Axial ────────────┐
        │  timeline widget   │
        │  detail panel      │
        └────────────────────┘
        [Refresh]  [Close]
    """

    def __init__(
        self,
        *,
        workspace,
        run_id: str,
        display_name: Optional[str] = None,
        use_config_callback: Optional[Callable[[RoundHistoryEntry], None]] = None,
        active_bucket_callback: Optional[Callable[[], str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._workspace = workspace
        self._run_id = run_id
        self._display_name = display_name or run_id
        self._use_config_callback = use_config_callback
        # 2026-05-27 user feedback: clicks should only apply config for
        # the bucket matching the current view target.  Without this
        # gate, clicking an axial round while on lateral view would
        # push axial round's params to the shared widgets — confusing.
        # Inactive bucket dots still update the detail panel (read-only
        # inspection) but skip the use_config callback.
        self._active_bucket_callback = active_bucket_callback
        self._lateral_entries: List[RoundHistoryEntry] = []
        self._axial_entries: List[RoundHistoryEntry] = []

        self.setModal(False)
        self.setWindowTitle(f"Training History — {self._display_name}")
        self.setMinimumWidth(640)
        self.setMinimumHeight(440)

        # Apply themed style (same DIALOG_FORM_STYLE used by other Step 2 dialogs).
        try:
            from napari_epicycle.ui_shell.theme import DIALOG_FORM_STYLE
            self.setStyleSheet(DIALOG_FORM_STYLE)
        except Exception:
            pass  # style is cosmetic; never crash on missing theme

        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        title = QLabel(f"Training History — {self._display_name}", parent=self)
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #eeeeee;")
        root.addWidget(title)

        subtitle = QLabel(
            "Top line = data accumulation (always continuous).  "
            "Bottom line = model lineage (broken where the round restarted "
            "from a new base).  Hover a dot for details, click to open "
            "manifest actions.",
            parent=self,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        root.addWidget(subtitle)

        # Lateral group
        self._lateral_group = lat_group = QGroupBox("Lateral", parent=self)
        lat_v = QVBoxLayout(lat_group)
        lat_v.setContentsMargins(8, 14, 8, 8)
        lat_v.setSpacing(4)
        self._lateral_timeline = _TimelineWidget([], parent=lat_group)
        self._lateral_timeline.round_clicked.connect(
            lambda idx: self._on_round_clicked("lateral", idx)
        )
        lat_v.addWidget(self._lateral_timeline)
        self._lateral_detail = QTextEdit(parent=lat_group)
        self._lateral_detail.setReadOnly(True)
        self._lateral_detail.setMaximumHeight(120)
        self._lateral_detail.setStyleSheet(
            "background-color: rgba(0,0,0,0.30); color: #cccccc; "
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 10px;"
        )
        lat_v.addWidget(self._lateral_detail)
        root.addWidget(lat_group)

        # Axial group
        self._axial_group = ax_group = QGroupBox("Axial", parent=self)
        ax_v = QVBoxLayout(ax_group)
        ax_v.setContentsMargins(8, 14, 8, 8)
        ax_v.setSpacing(4)
        self._axial_timeline = _TimelineWidget([], parent=ax_group)
        self._axial_timeline.round_clicked.connect(
            lambda idx: self._on_round_clicked("axial", idx)
        )
        ax_v.addWidget(self._axial_timeline)
        self._axial_detail = QTextEdit(parent=ax_group)
        self._axial_detail.setReadOnly(True)
        self._axial_detail.setMaximumHeight(120)
        self._axial_detail.setStyleSheet(self._lateral_detail.styleSheet())
        ax_v.addWidget(self._axial_detail)
        root.addWidget(ax_group)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._refresh_btn = QPushButton("Refresh", parent=self)
        self._refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(self._refresh_btn)
        self._close_btn = QPushButton("Close", parent=self)
        self._close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

    def refresh(self):
        """Re-read manifests from disk and re-render timelines."""
        history = read_full_history(self._workspace, self._run_id)
        self._lateral_entries = history.get("lateral", [])
        self._axial_entries = history.get("axial", [])
        self._lateral_timeline.set_entries(self._lateral_entries)
        self._axial_timeline.set_entries(self._axial_entries)
        # Populate detail panels with full-text history.
        self._lateral_detail.setPlainText(
            format_history_for_bucket(self._lateral_entries, "lateral")
        )
        self._axial_detail.setPlainText(
            format_history_for_bucket(self._axial_entries, "axial")
        )
        # Visual cue: append "(active)" / "(inactive — click to inspect)"
        # to group-box titles so user knows which bucket Use-config
        # clicks will land on.  Only renders when an active_bucket
        # callback was wired.
        active_bucket = self._resolve_active_bucket()
        if active_bucket is not None:
            lat_active = (active_bucket == "lateral")
            ax_active = (active_bucket == "axial")
            self._lateral_group.setTitle(
                "Lateral  ● active" if lat_active
                else "Lateral  (inactive — inspection only)"
            )
            self._axial_group.setTitle(
                "Axial  ● active" if ax_active
                else "Axial  (inactive — inspection only)"
            )

    def _on_round_clicked(self, bucket: str, round_idx: int):
        entries = self._lateral_entries if bucket == "lateral" else self._axial_entries
        entry = next((e for e in entries if e.round_idx == round_idx), None)
        if entry is None:
            return
        # Highlight the selected round in the detail panel — this is
        # read-only inspection, allowed for BOTH buckets regardless of
        # the active view target.
        detail = self._lateral_detail if bucket == "lateral" else self._axial_detail
        active_bucket = self._resolve_active_bucket()
        is_active = (active_bucket == bucket) or active_bucket is None
        applied_note = "" if is_active else (
            f"\n\n⚠ Click outside the active view ({active_bucket}).  "
            f"Config NOT applied to widgets — switch to {bucket} view to apply."
        )
        detail.setPlainText(
            format_round_summary(entry) + "\n\n"
            + "Manifest: " + entry.manifest_path
            + (f"\nWeights: {entry.weights_filename}" if entry.weights_filename else "")
            + (f"\nDuration: {entry.duration_seconds:.1f}s" if entry.duration_seconds else "")
            + (f"\nCellpose: {entry.cellpose_version}" if entry.cellpose_version else "")
            + applied_note
        )
        # Invoke "Use this config" callback ONLY when the clicked bucket
        # matches the current view target.  User feedback 2026-05-27:
        # "it should only be clickable on the respective mode".
        if not is_active:
            return
        if self._use_config_callback is not None:
            try:
                self._use_config_callback(entry)
            except Exception as exc:
                print(f"[HISTORY-CALLBACK] use_config_callback failed: "
                      f"{type(exc).__name__}: {exc}")

    def _resolve_active_bucket(self) -> Optional[str]:
        """Return the active bucket name via the callback, or None if no
        callback was wired (in which case all buckets are treated as
        active — backward-compat for callers that don't pass the param)."""
        cb = self._active_bucket_callback
        if cb is None:
            return None
        try:
            v = cb()
            return v if v in ("lateral", "axial") else None
        except Exception:
            return None

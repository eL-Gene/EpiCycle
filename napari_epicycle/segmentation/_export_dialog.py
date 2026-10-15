"""The Step 4 Export window (state D2).

A ``QDialog`` in the ``themed_confirm`` family (``_DIALOG_STYLE``, Close in
``_DIALOG_BTN_CANCEL``), 480 px wide, non-modal, parented to the Step 4
panel.  It is built ONCE at panel construction and kept hidden (grill 2
U1-4), so the controller's ``self.ui.export_meshes_btn`` /
``export_features_btn`` / ``export_status_label`` exist when
``_connect_signals`` runs; Export… only shows it.

The window computes nothing: the controller pushes the rows (who can go and
why), the loaded stack for the meshes, the busy state and the status
sentences.  Ticks are the user's for ONE export only:
every ``set_rows`` puts every stack that can go back to ticked.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from qtpy.QtCore import QEvent, QRect, QSize, Qt, Signal
from qtpy.QtGui import QColor, QFont, QFontMetrics, QPainter
from qtpy.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QSizePolicy, QStyle, QStyledItemDelegate,
    QToolTip, QVBoxLayout, QWidget,
)

from ..ui_shell.theme import (
    DIM_LABEL_STYLE, FLOW_SUBTITLE_STYLE, LIST_STYLE_GREEN,
    PRIMARY_BTN_STYLE_GREEN, SECONDARY_BTN_STYLE, SEPARATOR_LINE_STYLE,
    STATUS_LABEL_STYLE, STEP4_ACCENT, _DIALOG_BTN_CANCEL, _DIALOG_STYLE,
    apply_button_icon,
)
from . import _step4_text as T
from ._stack_row_delegate import (CHECK_DISABLED_STYLE, StackRowDelegate,
                                  _icon_pixmap)

EXPORT_W = 480                 # today's Extract Features width
ROW_H = 22
WHY_MAX_W = 170                # .xrow .why: flex 0 1 170px
LIST_MAX_ROWS = 8
ROLE_ROW = Qt.UserRole + 2
_AMBER = QColor("#d4a843")
_NAME = QColor("#dddddd")
_NAME_OFF = QColor("#777777")
_SEL_BG = QColor(106, 191, 105, 64)


def _show_tip(pos, text, view) -> None:
    """``QToolTip.showText`` (a module name, so a test can watch it)."""
    QToolTip.showText(pos, text, view)


TITLE_STYLE = "font-weight: bold; font-size: 13px; color: #eeeeee;"
COUNT_STYLE = "color: #aaaaaa; font-size: 11px;"


class ElidedLabel(QLabel):
    """A one-line label that elides its text to its width; the whole text
    is its tooltip unless another tooltip is given (a file's full path)."""

    def __init__(self, parent=None, *, elide=Qt.ElideRight,
                 align=Qt.AlignLeft | Qt.AlignVCenter):
        super().__init__(parent)
        self._full = ""
        self._elide = elide
        self.setAlignment(align)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

    def setText(self, text, tip=None) -> None:
        self._full = str(text or "")
        self.setToolTip(self._full if tip is None else tip)
        self._refit()

    def text(self) -> str:
        return self._full

    def _refit(self) -> None:
        self.ensurePolished()
        width = max(self.width(), 1)
        QLabel.setText(self, self.fontMetrics().elidedText(
            self._full, self._elide, width))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refit()

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def sizeHint(self):
        self.ensurePolished()
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 2,
                     super().sizeHint().height())


class WrapStatusLabel(QLabel):
    """The window's status line (R1-3: every sentence label in the
    window word-wraps): the whole sentence on screen, a path (or any other
    tooltip) passed as *tip*."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def setText(self, text, tip=None) -> None:
        text = str(text or "")
        QLabel.setText(self, text)
        self.setToolTip(text if tip is None else tip)


class ExportRowDelegate(QStyledItemDelegate):
    """One ``.xrow``: tick · status icon · name · status word · an amber
    reason (at most 170 px, right-aligned).  A row that cannot go has a
    greyed tick and a grey name; a click anywhere on a row that can go
    toggles its tick."""

    def _fonts(self, base: QFont):
        name = QFont(base)
        name.setPixelSize(11)
        word = QFont("Consolas")
        word.setStyleHint(QFont.Monospace)
        word.setPixelSize(11)
        why = QFont(base)
        why.setPixelSize(10)
        return name, word, why

    def sizeHint(self, option, index):
        return QSize(max(option.rect.width(), 100), ROW_H)

    def layout(self, rect: QRect, row: Dict[str, Any], base: QFont) -> dict:
        _name, word_f, why_f = self._fonts(base)
        x, y, h = rect.x() + 6, rect.y(), rect.height()
        g = {"tick": QRect(x, y + (h - 13) // 2, 13, 13)}
        x += 13 + 8
        g["icon"] = QRect(x, y + (h - 14) // 2, 14, 14)
        x += 14 + 8
        right = rect.right() - 6
        why = row.get("why") or ""
        why_w = min(QFontMetrics(why_f).horizontalAdvance(why), WHY_MAX_W) \
            if why else 0
        g["why"] = QRect(right - why_w, y, why_w, h)
        word = f"{row.get('dots') or ''} {row.get('word') or ''}".strip()
        word_w = QFontMetrics(word_f).horizontalAdvance(word)
        wr = g["why"].left() - (8 if why_w else 0)
        g["word"] = QRect(wr - word_w, y, word_w, h)
        g["name"] = QRect(x, y, max(g["word"].left() - 8 - x, 10), h)
        return g

    def paint(self, painter: QPainter, option, index):
        row = index.data(ROLE_ROW) or {}
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = option.rect
        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, _SEL_BG)
        g = self.layout(rect, row, option.font)
        name_f, word_f, why_f = self._fonts(option.font)
        can_go = bool(row.get("can_go"))
        StackRowDelegate._paint_tick(
            painter, g["tick"],
            index.data(Qt.CheckStateRole) == Qt.Checked, can_go)
        pm = _icon_pixmap(row.get("icon") or "mdi6.circle-outline",
                          row.get("colour") or "#777777", 14)
        if pm is not None:
            painter.drawPixmap(g["icon"].topLeft(), pm)
        painter.setFont(name_f)
        painter.setPen(_NAME if can_go else _NAME_OFF)
        painter.drawText(g["name"], int(Qt.AlignLeft | Qt.AlignVCenter),
                         QFontMetrics(name_f).elidedText(
                             row.get("name") or "", Qt.ElideMiddle,
                             g["name"].width()))
        painter.setFont(word_f)
        painter.setPen(QColor(row.get("colour") or "#777777"))
        word = f"{row.get('dots') or ''} {row.get('word') or ''}".strip()
        painter.drawText(g["word"], int(Qt.AlignLeft | Qt.AlignVCenter), word)
        why = row.get("why") or ""
        if why:
            painter.setFont(why_f)
            painter.setPen(_AMBER)
            painter.drawText(g["why"], int(Qt.AlignRight | Qt.AlignVCenter),
                             QFontMetrics(why_f).elidedText(
                                 why, Qt.ElideRight, g["why"].width()))
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if event.type() == QEvent.MouseButtonRelease:
            row = index.data(ROLE_ROW) or {}
            if not row.get("can_go"):
                return True
            state = index.data(Qt.CheckStateRole)
            model.setData(index, Qt.Unchecked if state == Qt.Checked
                          else Qt.Checked, Qt.CheckStateRole)
            return True
        if event.type() in (QEvent.MouseButtonPress,
                            QEvent.MouseButtonDblClick):
            return True
        return super().editorEvent(event, model, option, index)

    def helpEvent(self, event, view, option, index):
        if event is None or event.type() != QEvent.ToolTip:
            return super().helpEvent(event, view, option, index)
        _show_tip(event.globalPos(), T.STEP4_HELP_LINES["export_row_tick"],
                  view)
        return True


class Step4ExportDialog(QDialog):
    """The Export window.  ``features_requested`` /
    ``meshes_requested`` are the two exports; ``ticks_changed`` fires when
    the user takes a tick off (or puts it back)."""

    features_requested = Signal()
    meshes_requested = Signal()
    ticks_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(T.EXPORT_WINDOW_TITLE)
        self.setModal(False)
        self.setStyleSheet(_DIALOG_STYLE + LIST_STYLE_GREEN
                           + CHECK_DISABLED_STYLE)
        self.setFixedWidth(EXPORT_W)
        self._busy = False
        self._can_go = 0
        self._mesh_stack: Optional[str] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 14, 20, 16)
        lay.setSpacing(8)

        self.title_label = QLabel(T.EXPORT_WINDOW_TITLE)
        self.title_label.setStyleSheet(TITLE_STYLE)
        lay.addWidget(self.title_label)
        self.desc_label = self._wrap(T.EXPORT_DESC, FLOW_SUBTITLE_STYLE)
        lay.addWidget(self.desc_label)
        self.count_label = self._wrap("", COUNT_STYLE)
        lay.addWidget(self.count_label)

        self.export_rows_list = QListWidget()
        self.export_rows_list.setItemDelegate(ExportRowDelegate(
            self.export_rows_list))
        self.export_rows_list.setSelectionMode(
            QAbstractItemView.NoSelection)
        self.export_rows_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        self.export_rows_list.setUniformItemSizes(True)
        self.export_rows_list.itemChanged.connect(self._on_tick_changed)
        lay.addWidget(self.export_rows_list)

        self.export_features_btn = QPushButton(
            T.say(T.EXPORT_FEATURES_BTN, k=0))
        self.export_features_btn.setStyleSheet(PRIMARY_BTN_STYLE_GREEN)
        apply_button_icon(self.export_features_btn,
                          "mdi6.database-export-outline", size=14)
        self.export_features_btn.clicked.connect(self._on_features_clicked)
        lay.addWidget(self.export_features_btn)
        self.what_label = self._wrap(T.EXPORT_FEATURES_WHAT, DIM_LABEL_STYLE)
        lay.addWidget(self.what_label)
        self.export_status_label = WrapStatusLabel()
        self.export_status_label.setStyleSheet(STATUS_LABEL_STYLE)
        lay.addWidget(self.export_status_label)
        self.last_label = self._wrap("", DIM_LABEL_STYLE)
        self.last_label.setVisible(False)
        lay.addWidget(self.last_label)

        sep = QLabel("")
        sep.setFixedHeight(1)
        sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        lay.addWidget(sep)

        mesh_row = QHBoxLayout()
        mesh_row.setSpacing(8)
        self.export_meshes_btn = QPushButton(T.EXPORT_MESHES_BTN)
        self.export_meshes_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        apply_button_icon(self.export_meshes_btn, "mdi6.cube-scan",
                          color=STEP4_ACCENT, size=14)
        self.export_meshes_btn.clicked.connect(self._on_meshes_clicked)
        mesh_row.addWidget(self.export_meshes_btn)
        self.mesh_target_label = ElidedLabel()
        self.mesh_target_label.setStyleSheet(DIM_LABEL_STYLE)
        mesh_row.addWidget(self.mesh_target_label, 1)
        lay.addLayout(mesh_row)
        self.mesh_what_label = self._wrap("", DIM_LABEL_STYLE)
        lay.addWidget(self.mesh_what_label)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        self.close_btn = QPushButton(T.EXPORT_CLOSE_BTN)
        self.close_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
        self.close_btn.clicked.connect(self.hide)
        close_row.addWidget(self.close_btn)
        lay.addLayout(close_row)

        self.set_rows([])
        self.set_mesh_target(None, "")
        self.hide()

    @staticmethod
    def _wrap(text: str, style: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(style)
        return lbl

    # -- rows --------------------------------------------------

    def set_rows(self, rows: List[Dict[str, Any]], *,
                 keep_unticked: bool = False) -> None:
        """Every reconstructed stack, one line each.  A stack that can go is
        ticked (every time the window opens: ticks are not remembered); a refresh while it is open (``keep_unticked``) keeps
        the ticks the user took off."""
        lw = self.export_rows_list
        off = set()
        if keep_unticked:
            for i in range(lw.count()):
                it = lw.item(i)
                if it.checkState() != Qt.Checked:
                    off.add(it.data(Qt.UserRole))
        lw.blockSignals(True)
        try:
            lw.clear()
            for row in rows:
                item = QListWidgetItem()
                item.setData(Qt.UserRole, row.get("name"))
                item.setData(ROLE_ROW, dict(row))
                flags = Qt.ItemIsEnabled
                if row.get("can_go"):
                    flags |= Qt.ItemIsUserCheckable
                item.setFlags(flags)
                ticked = row.get("can_go") and row.get("name") not in off
                item.setData(Qt.CheckStateRole,
                             Qt.Checked if ticked else Qt.Unchecked)
                lw.addItem(item)
        finally:
            lw.blockSignals(False)
        self._can_go = sum(1 for r in rows if r.get("can_go"))
        self.count_label.setText(T.say(T.EXPORT_FINALIZED_COUNT,
                                       k=self._can_go, n=len(rows)))
        n_shown = max(1, min(len(rows), LIST_MAX_ROWS))
        lw.setFixedHeight(n_shown * ROW_H + 2 * lw.frameWidth() + 4)
        self._refresh_features()
        self._fit_height()

    def _fit_height(self) -> None:
        """The window takes its content's height at its fixed width (a
        word-wrapped label's size hint is its one-line width, so
        ``adjustSize`` alone leaves the rows spread apart)."""
        lay = self.layout()
        if lay is not None and lay.hasHeightForWidth():
            h = lay.heightForWidth(EXPORT_W)
        else:
            h = self.sizeHint().height()
        self.setFixedHeight(max(h, 1))

    def ticked_names(self) -> List[str]:
        lw = self.export_rows_list
        out = []
        for i in range(lw.count()):
            it = lw.item(i)
            row = it.data(ROLE_ROW) or {}
            if row.get("can_go") and it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _on_tick_changed(self, _item) -> None:
        self._refresh_features()
        self.ticks_changed.emit()

    def _refresh_features(self) -> None:
        """Export features: its count, and why it is grey."""
        k = len(self.ticked_names())
        self.export_features_btn.setText(T.say(T.EXPORT_FEATURES_BTN, k=k))
        reason = ""
        if self._busy:
            reason = getattr(self, "_busy_reason", "") or T.EXPORT_RUNNING
        elif self._can_go == 0:
            reason = T.EXPORT_BLOCK
        elif k == 0:
            reason = T.EXPORT_NONE_TICKED
        self.export_features_btn.setEnabled(not reason)
        if reason:
            self.export_status_label.setText(reason)
        elif self.export_status_label.text() in (
                T.EXPORT_BLOCK, T.EXPORT_NONE_TICKED, T.EXPORT_RUNNING,
                T.EXPORT_ROWS_CHANGED) or (
                self.export_status_label.text()
                and self.export_status_label.text()
                == getattr(self, "_last_busy_reason", "")):
            self.export_status_label.setText("")
        if self.layout() is not None:
            self._fit_height()      # R1-3: a wrapped status line

    # -- meshes (9) ---------------------------------------------

    def set_mesh_target(self, stack: Optional[str], folder: str) -> None:
        self._mesh_stack = stack or None
        if stack:
            self.mesh_target_label.setText(
                T.EXPORT_MESH_TARGET.format(stack=stack))
            self.mesh_what_label.setText(
                T.EXPORT_MESH_WHAT.format(folder=folder))
        else:
            self.mesh_target_label.setText(T.EXPORT_MESH_NONE)
            self.mesh_what_label.setText(
                T.EXPORT_MESH_WHAT.format(folder=folder))
        self.export_meshes_btn.setEnabled(bool(stack) and not self._busy)
        self._fit_height()

    # -- busy, status, last export -------------------------------------------

    def set_busy(self, busy: bool, reason: str = "") -> None:
        """While an export runs every control is grey except Close.  B2: any other Step 4 job greys them too, and *reason* -- that
        job's ``STEP4_LOCK_*`` sentence -- is the status line instead of
        ``EXPORT_RUNNING`` (I-9: the reason is on screen)."""
        self._busy = bool(busy)
        self._busy_reason = reason if busy else ""
        if reason:
            self._last_busy_reason = reason
        self.export_rows_list.setEnabled(not busy)
        self.export_meshes_btn.setEnabled(bool(self._mesh_stack)
                                          and not busy)
        self._refresh_features()

    def show_status(self, text: str, tip: Optional[str] = None) -> None:
        self.export_status_label.setText(text, tip)
        self._fit_height()

    def show_last(self, text: str) -> None:
        self.last_label.setText(text)
        self.last_label.setVisible(bool(text))
        self._fit_height()

    def _on_features_clicked(self) -> None:
        self.features_requested.emit()

    def _on_meshes_clicked(self) -> None:
        self.meshes_requested.emit()

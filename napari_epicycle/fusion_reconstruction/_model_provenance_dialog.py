"""The model details window ("On
screen", brick R6a), opened by a row's ⓘ in the Predict + Reconstruct list.

A small window in the ``themed_notice`` family (``ui_shell/theme.py``: the
same dark dialog, title line, button styles).  It lists EVERY key of every
view GENERICALLY -- a key no code has heard of shows with no code change,
nested dicts as ``a.b`` (``_model_provenance.flatten_items``) -- then the
block's own keys, then an editable ``notes`` field.

* Save hands the notes to the caller's ``on_save(notes)`` (a bound method
  of the caller); the window does no file I/O.  ``on_save`` answers
  ``None`` / ``True`` (saved), a sentence (shown as it is) or ``False``
  (``save_failed``); an exception is caught and logged.
* ``locked=True`` (row 21's lock: Predict / Reconstruct running) greys Save
  and writes the reason ON SCREEN -- Qt shows no tooltip on a disabled
  widget; ``set_locked`` follows the lock while the window is open.
* Every widget is parented; no lambda is connected; the window deletes
  itself on close (``WA_DeleteOnClose``).
* It opens ``PREF_W`` wide (at most ``MAX_W``, to fit the 566 px dock)
  and at most ``MAX_H`` tall; the listing scrolls.  Long paths wrap (a zero-width space
  after each separator), the raw value is the label's tooltip.

Every word arrives in ``texts`` (``_step3_pr_text.DIALOG_TEXTS``).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ..ui_shell.theme import (
    _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM, _DIALOG_STYLE, DIM_LABEL_STYLE,
    PANEL_ERROR_STYLE, PANEL_HELP_STYLE, SCROLLBAR_STYLE,
)
from ..ui_shell._recon_settings_widgets import ZWSP
from ._model_provenance import VIEW_NAMES, display_value, flatten_items

#: the widest the window opens: the 566 px dock
MAX_W = 560
#: the tallest it opens; the listing scrolls beyond
MAX_H = 620
#: the narrowest, as ``themed_notice``
MIN_W = 380
#: the width it opens at: room for a key column and a value column
#: without wrapping every path (inside ``MAX_W``)
PREF_W = 520
#: an unbroken run longer than this gets a break opportunity
_RUN = 18
#: characters after which a value may wrap
_BREAK_AFTER = "/\\._-:,"

_SECTION_STYLE = "font-weight: bold; font-size: 11px; color: #eeeeee;"
_KIND_STYLE = "color: #bbbbbb; font-size: 10px;"
_KEY_STYLE = DIM_LABEL_STYLE
_VALUE_STYLE = "color: #dddddd; font-size: 10px;"
_NOTES_STYLE = (
    "QPlainTextEdit { background-color: #3a3f47;"
    " border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 3px;"
    " color: #eeeeee; font-size: 11px; padding: 3px; }"
    "QPlainTextEdit:focus { border: 1px solid rgba(212, 168, 67, 0.55); }"
)
_DISABLED_BTN = (
    "QPushButton:disabled { background-color: rgba(255, 255, 255, 0.03);"
    " border: 1px solid rgba(255, 255, 255, 0.06); color: #777777; }"
)


def _breakable(text: str) -> str:
    """*text* with a zero-width space after each separator and inside any
    long unbroken run, so a path wraps in a narrow window."""
    out: List[str] = []
    run = 0
    for ch in text:
        out.append(ch)
        if ch.isspace():
            run = 0
            continue
        run += 1
        if ch in _BREAK_AFTER or run >= _RUN:
            out.append(ZWSP)
            run = 0
    return "".join(out)


class ModelProvenanceDialog(QDialog):
    """``ModelProvenanceDialog(parent, block, *, title, texts, on_save,
    locked=False)``; the caller shows it (non-modal) and keeps no reference
    it needs after close."""

    def __init__(self, parent, block, *, title: str,
                 texts: Dict[str, Any],
                 on_save: Callable[[str], Any],
                 locked: bool = False):
        super().__init__(parent)
        self._texts = dict(texts or {})
        self._on_save = on_save
        self._block = block if isinstance(block, dict) else {}
        self._shown: List[Tuple[str, str, str]] = []

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(title)
        self.setStyleSheet(_DIALOG_STYLE + SCROLLBAR_STYLE)
        self.setMinimumWidth(MIN_W)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        self.title_label = QLabel(_breakable(title), parent=self)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #eeeeee;")
        root.addWidget(self.title_label)

        # ── the listing ────────────────────────────────────────────────
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { background: transparent; }")
        body = QWidget(self.scroll)
        body.setObjectName("modelProvenanceBody")
        body.setStyleSheet(
            "QWidget#modelProvenanceBody { background: transparent; }")
        self._body_layout = QVBoxLayout(body)
        self._body_layout.setContentsMargins(0, 0, 6, 0)
        self._body_layout.setSpacing(8)
        self._build_listing(body)
        self._body_layout.addStretch(1)
        self.scroll.setWidget(body)
        root.addWidget(self.scroll, 1)

        # ── the notes ──────────────────────────────────────────────────
        notes_lbl = QLabel(self._t("notes"), parent=self)
        notes_lbl.setStyleSheet(_SECTION_STYLE)
        root.addWidget(notes_lbl)
        self.notes_edit = QPlainTextEdit(self)
        self.notes_edit.setPlaceholderText(self._t("notes_hint"))
        self.notes_edit.setStyleSheet(_NOTES_STYLE)
        self.notes_edit.setFixedHeight(64)
        notes = self._block.get("notes")
        self.notes_edit.setPlainText("" if notes is None else str(notes))
        root.addWidget(self.notes_edit)

        # ── the reason Save is grey, and the save result: on screen ────
        self.reason_label = QLabel(self._t("locked"), parent=self)
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet(PANEL_HELP_STYLE)
        root.addWidget(self.reason_label)
        self.status_label = QLabel("", parent=self)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(PANEL_HELP_STYLE)
        self.status_label.hide()
        root.addWidget(self.status_label)

        # ── the buttons ────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addStretch(1)
        self.close_btn = QPushButton(self._t("close"), self)
        self.close_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
        self.close_btn.clicked.connect(self.close)
        self.save_btn = QPushButton(self._t("save"), self)
        self.save_btn.setStyleSheet(_DIALOG_BTN_CONFIRM + _DISABLED_BTN)
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._on_save_clicked)
        row.addWidget(self.close_btn)
        row.addWidget(self.save_btn)
        root.addLayout(row)

        self.set_locked(locked)
        self._fit()

    # ── words ──────────────────────────────────────────────────────────
    def _t(self, key: str) -> str:
        value = self._texts.get(key, "")
        return value if isinstance(value, str) else ""

    def _kind_word(self, kind) -> str:
        kinds = self._texts.get("kinds")
        if isinstance(kinds, dict) and kind in kinds:
            return str(kinds[kind])
        return "" if kind is None else str(kind)

    # ── the listing ────────────────────────────────────────────────────
    def _section(self, parent: QWidget, heading: str,
                 sub: Optional[str] = None
                 ) -> Tuple[QWidget, QGridLayout]:
        head = QLabel(_breakable(heading), parent=parent)
        head.setWordWrap(True)
        head.setStyleSheet(_SECTION_STYLE)
        self._body_layout.addWidget(head)
        if sub:
            sub_lbl = QLabel(_breakable(sub), parent=parent)
            sub_lbl.setWordWrap(True)
            sub_lbl.setStyleSheet(_KIND_STYLE)
            self._body_layout.addWidget(sub_lbl)
        holder = QWidget(parent)
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 3)
        self._body_layout.addWidget(holder)
        return holder, grid

    def _rows(self, holder: QWidget, grid: QGridLayout, section: str,
              mapping) -> None:
        empty = self._t("empty")
        for i, (key, value) in enumerate(flatten_items(mapping)):
            text = display_value(value, empty=empty)
            k = QLabel(_breakable(key), parent=holder)
            k.setWordWrap(True)
            k.setStyleSheet(_KEY_STYLE)
            k.setAlignment(Qt.AlignmentFlag.AlignTop
                           | Qt.AlignmentFlag.AlignLeft)
            v = QLabel(_breakable(text), parent=holder)
            v.setWordWrap(True)
            v.setStyleSheet(_VALUE_STYLE)
            v.setAlignment(Qt.AlignmentFlag.AlignTop
                           | Qt.AlignmentFlag.AlignLeft)
            v.setToolTip(text)
            grid.addWidget(k, i, 0)
            grid.addWidget(v, i, 1)
            self._shown.append((section, key, text))

    def _build_listing(self, body: QWidget) -> None:
        views = self._block.get("views")
        if not isinstance(views, dict):
            views = None
        if not self._block or views is None:
            note = QLabel(self._t("no_block"), parent=body)
            note.setWordWrap(True)
            note.setStyleSheet(_KIND_STYLE)
            self._body_layout.addWidget(note)
        if views is not None:
            order = [n for n in VIEW_NAMES if n in views] + \
                [n for n in views if n not in VIEW_NAMES]
            for name in VIEW_NAMES:
                if name not in views:
                    order.append(name)
            for name in order:
                view = views.get(name)
                heading = self._t(name) or str(name)
                if not isinstance(view, dict):
                    self._section(body, heading, self._t("view_missing"))
                    continue
                holder, grid = self._section(
                    body, heading, self._kind_word(view.get("kind")))
                self._rows(holder, grid, str(name), view)
        record = {k: v for k, v in self._block.items()
                  if k not in ("views", "notes")}
        if record:
            holder, grid = self._section(body, self._t("record"))
            self._rows(holder, grid, "record", record)

    def shown_items(self) -> List[Tuple[str, str, str]]:
        """``[(section, dotted key, shown value)]`` as listed (for tests and
        the driver)."""
        return list(self._shown)

    # ── the lock ───────────────────────────────────────────────────────
    def set_locked(self, locked: bool) -> None:
        """Save greyed, with its reason on screen, while *locked*."""
        locked = bool(locked)
        self.save_btn.setEnabled(not locked)
        self.reason_label.setHidden(not locked)

    # ── Save ───────────────────────────────────────────────────────────
    def _on_save_clicked(self) -> None:
        if not self.save_btn.isEnabled():
            return
        notes = self.notes_edit.toPlainText()
        try:
            answer = self._on_save(notes)
        except Exception as exc:      # noqa: BLE001
            print(f"[PROVENANCE] notes save raised: {type(exc).__name__}: "
                  f"{exc}")
            answer = False
        if answer is None or answer is True:
            self._show_status(self._t("saved"), failed=False)
        elif isinstance(answer, str) and answer:
            self._show_status(answer, failed=True)
        else:
            self._show_status(self._t("save_failed"), failed=True)

    def _show_status(self, text: str, *, failed: bool) -> None:
        self.status_label.setStyleSheet(
            PANEL_ERROR_STYLE if failed else PANEL_HELP_STYLE)
        self.status_label.setText(text)
        self.status_label.show()

    # ── size ───────────────────────────────────────────────────────────
    def _fit(self) -> None:
        hint = self.sizeHint()
        body = self.scroll.widget()
        want_h = hint.height()
        if body is not None:
            want_h += max(0, body.sizeHint().height()
                          - self.scroll.sizeHint().height())
        self.resize(min(max(MIN_W, PREF_W, hint.width()), MAX_W),
                    min(max(want_h, 200), MAX_H))

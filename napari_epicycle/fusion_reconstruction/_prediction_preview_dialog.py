"""The prediction window of a Predict + Reconstruct row, and its loader.

Opened by the row's eye on a predicted patch (or "Show prediction…" in the
row's menu).  A small non-modal window in the model details window's family
(the same dark dialog, title line and buttons):

* a two-line provenance (the run, the model, when; the channel codes as the
  channel boxes name them);
* two dropdowns, Pass (XY / XZ / YZ) and Map (Probability / Flows), opening
  on XY + Probability -- a choice hands ``on_choice(pass, kind)`` to the
  caller, which reads ONLY that file on a worker and answers with
  :meth:`set_map`;
* two panes, "What the model saw" (the patch rebuilt from the recorded
  channel codes) and the chosen map, each in a 248 px box with the
  picture's aspect kept and its pixels square;
* one plane slider whose axis follows the pass (z for XY, y for XZ, x for
  YZ), counted from 0 as napari counts; moves are drawn on a 0 ms
  single-shot timer, so a fast drag draws once per event-loop turn, and
  each plane is thinned to about twice the pane's size before it is
  coloured, so a move costs the same at any patch size;
* under the map its key (the probability's fixed −6 … 6, the flows'
  colour wheel);
* the reading line -- as tall as the longer of its two sentences, so a
  Map change never squeezes the panes -- and in the foot "Model details…"
  (``on_details()``; hidden for a patch with no made-by record) and Close.

The window sizes itself by its layout's height for its width, again
whenever a read fills it (a wrapped provenance line), never past the
screen it is on.

The window does no file I/O and holds no array file open: every array it
gets is a plain 8-bit volume in memory.  Every widget is parented; no
lambda is connected; it deletes itself on close.  Every word arrives in
``texts`` (``_step3_pr_text.PRED_TEXTS``).

:class:`PredictionPreviewWorker` runs ONE loader function of
``_prediction_preview`` on a ``QThread`` and answers ``loaded`` or
``failed``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from qtpy.QtCore import QObject, QRect, Qt, QTimer, Signal
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from ..ui_shell.theme import (
    _DIALOG_BTN_CANCEL, _DIALOG_STYLE, INPUT_STYLE,
)
from . import _prediction_preview as PP

#: the window's width: the 566 px dock's room, as the model details window
MAX_W = 560
#: each pane's square box
PANE_PX = 248
#: the gap between the two panes
PANE_GAP = 16

_TITLE_STYLE = "font-weight: bold; font-size: 13px; color: #eeeeee;"
_LINE_STYLE = "color: #aaaaaa; font-size: 11px;"
_LABEL_STYLE = "color: #cccccc; font-size: 11px;"
_HEAD_STYLE = "font-weight: bold; font-size: 11px; color: #eeeeee;"
_BOX_STYLE = (
    "QLabel { background-color: #000000; color: #999999; font-size: 10px;"
    " border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 3px; }")
_CAPTION_STYLE = (
    "color: #999999; font-size: 10px;"
    " font-family: Consolas, 'Cascadia Mono', monospace;")
_NOTE_STYLE = "color: #999999; font-size: 10px;"
_SLIDER_LABEL_STYLE = (
    "color: #bbbbbb; font-size: 11px;"
    " font-family: Consolas, 'Cascadia Mono', monospace;")
_SLIDER_STYLE = (
    "QSlider::groove:horizontal { height: 4px; border-radius: 2px;"
    " background: rgba(255, 255, 255, 0.15); }"
    "QSlider::handle:horizontal { width: 12px; height: 12px;"
    " margin: -4px 0; border-radius: 6px; background: #d4a843; }")

_LUT: Optional[np.ndarray] = None


def magma_lut() -> np.ndarray:
    """napari's magma as a ``(256, 3)`` uint8 table, built once."""
    global _LUT
    if _LUT is None:
        try:
            from napari.utils.colormaps import ensure_colormap
            rgba = np.asarray(ensure_colormap("magma").map(
                np.linspace(0.0, 1.0, 256)), dtype=np.float64)
            _LUT = np.clip(np.rint(rgba[:, :3] * 255.0), 0, 255).astype(
                np.uint8)
        except Exception as exc:      # noqa: BLE001 -- a grey ramp then
            print(f"[STEP3-PRED] magma colour table unavailable: {exc!r}")
            ramp = np.arange(256, dtype=np.uint8)
            _LUT = np.stack([ramp, ramp, ramp], axis=-1)
    return _LUT


def to_pixmap(rgb: np.ndarray) -> QPixmap:
    """A ``QPixmap`` of a ``uint8 (rows, cols, 3)`` picture.  The array is
    made contiguous, the row stride is given (3 x cols bytes: Qt would
    otherwise pad each row to 4 bytes) and the image is copied before the
    array can go away."""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    img = QImage(arr.data, cols, rows, 3 * cols,
                 QImage.Format.Format_RGB888).copy()
    del arr
    return QPixmap.fromImage(img)


class PredictionPreviewWorker(QObject):
    """Runs ``fn(*args)`` once (``run``) and answers ``loaded(result)`` or
    ``failed((kind, detail))`` -- *kind* ``"too_large"`` (*detail* the
    patch's Z, Y, X) or ``"error"`` (*detail* the error's text)."""

    loaded = Signal(object)
    failed = Signal(object)

    def __init__(self, fn: Callable, args: Tuple):
        super().__init__()
        self._fn = fn
        self._args = tuple(args)

    def run(self) -> None:
        try:
            result = self._fn(*self._args)
        except PP.PreviewTooLarge as exc:
            self.failed.emit(("too_large", exc.shape))
            return
        except Exception as exc:      # noqa: BLE001 -- said in the window
            print(f"[STEP3-PRED] read failed: {type(exc).__name__}: {exc}")
            self.failed.emit(("error", f"{type(exc).__name__}: {exc}"))
            return
        self.loaded.emit(result)


class PredictionPreviewDialog(QDialog):
    """``PredictionPreviewDialog(parent, *, title, texts, on_choice,
    on_details, details=True)``; the caller shows it (non-modal)."""

    def __init__(self, parent, *, title: str, texts: Dict[str, Any],
                 on_choice: Callable[[str, str], Any],
                 on_details: Callable[[], Any], details: bool = True):
        super().__init__(parent)
        self._texts = dict(texts or {})
        self._on_choice = on_choice
        self._on_details = on_details
        self._image: Optional[Dict[str, Any]] = None
        self._map: Optional[Dict[str, Any]] = None
        self._shape: Optional[Tuple[int, int, int]] = None
        self._pass = PP.PASSES[0]
        self._kind = PP.KINDS[0]
        self._map_message = ""
        #: the last two pictures drawn (the tests and the driver read them)
        self.last_left: Optional[np.ndarray] = None
        self.last_right: Optional[np.ndarray] = None

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(title)
        self.setStyleSheet(_DIALOG_STYLE + INPUT_STYLE)
        self.setMinimumWidth(2 * PANE_PX + PANE_GAP + 24)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(8)

        self.title_label = QLabel(title, parent=self)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(_TITLE_STYLE)
        root.addWidget(self.title_label)
        self.line1_label = QLabel(self._t("loading"), parent=self)
        self.line2_label = QLabel("", parent=self)
        for lbl in (self.line1_label, self.line2_label):
            lbl.setWordWrap(True)
            lbl.setStyleSheet(_LINE_STYLE)
            root.addWidget(lbl)
        self.line2_label.hide()

        # ── Pass / Map ─────────────────────────────────────────────────
        row = QHBoxLayout()
        row.setSpacing(8)
        pass_lbl = QLabel(self._t("pass_label"), parent=self)
        map_lbl = QLabel(self._t("map_label"), parent=self)
        for lbl in (pass_lbl, map_lbl):
            lbl.setStyleSheet(_LABEL_STYLE)
        self.pass_combo = QComboBox(self)
        for key in PP.PASSES:
            self.pass_combo.addItem(self._t(f"pass_{key}"), key)
        self.map_combo = QComboBox(self)
        self.map_combo.addItem(self._t("map_prob"), "probs")
        self.map_combo.addItem(self._t("map_flows"), "flows")
        row.addWidget(pass_lbl)
        row.addWidget(self.pass_combo)
        row.addSpacing(10)
        row.addWidget(map_lbl)
        row.addWidget(self.map_combo)
        row.addStretch(1)
        root.addLayout(row)
        self.pass_combo.currentIndexChanged.connect(self._on_combo_changed)
        self.map_combo.currentIndexChanged.connect(self._on_combo_changed)

        # ── the two panes ──────────────────────────────────────────────
        panes = QHBoxLayout()
        panes.setSpacing(PANE_GAP)
        panes.addStretch(1)
        (self.left_head, self.left_box, self.left_caption,
         left) = self._pane(self._t("left"))
        (self.right_head, self.right_box, self.right_caption,
         right) = self._pane("")
        self.left_note = QLabel(self._t("left_contrast"), parent=left)
        self.left_note.setWordWrap(True)
        self.left_note.setStyleSheet(_NOTE_STYLE)
        left.layout().addWidget(self.left_note)
        panes.addWidget(left)
        panes.addWidget(right)
        panes.addStretch(1)
        root.addLayout(panes)

        # ── the plane slider ───────────────────────────────────────────
        srow = QHBoxLayout()
        srow.setSpacing(12)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setStyleSheet(_SLIDER_STYLE)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider_label = QLabel("", parent=self)
        self.slider_label.setStyleSheet(_SLIDER_LABEL_STYLE)
        self.slider_label.setMinimumWidth(70)
        self.slider_label.setAlignment(Qt.AlignmentFlag.AlignRight
                                       | Qt.AlignmentFlag.AlignVCenter)
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.slider_label)
        root.addLayout(srow)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(0)
        self._render_timer.timeout.connect(self._render)
        #: a second size check once the new texts are laid out (a label's
        #: wrapped height is known only after the event loop turns)
        self._grow_timer = QTimer(self)
        self._grow_timer.setSingleShot(True)
        self._grow_timer.setInterval(0)
        self._grow_timer.timeout.connect(self._grow)
        self.slider.valueChanged.connect(self._on_slider_moved)

        self.reading_label = QLabel(self._t("reading"), parent=self)
        self.reading_label.setWordWrap(True)
        self.reading_label.setStyleSheet(_NOTE_STYLE)
        self.reading_label.setAlignment(Qt.AlignmentFlag.AlignLeft
                                        | Qt.AlignmentFlag.AlignTop)
        root.addWidget(self.reading_label)

        # ── the foot ───────────────────────────────────────────────────
        foot = QHBoxLayout()
        foot.addStretch(1)
        self.details_btn = QPushButton(self._t("details"), self)
        self.details_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
        self.details_btn.clicked.connect(self._on_details_clicked)
        self.details_btn.setVisible(bool(details))
        self.close_btn = QPushButton(self._t("close"), self)
        self.close_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
        self.close_btn.setDefault(True)
        self.close_btn.clicked.connect(self.close)
        foot.addWidget(self.details_btn)
        foot.addWidget(self.close_btn)
        root.addLayout(foot)

        self._set_right_head()
        self.set_loading()
        self._fit()

    # ── words ──────────────────────────────────────────────────────────
    def _t(self, key: str) -> str:
        value = self._texts.get(key, "")
        return value if isinstance(value, str) else ""

    def _pane(self, head_text: str):
        holder = QWidget(self)
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        head = QLabel(head_text, parent=holder)
        head.setStyleSheet(_HEAD_STYLE)
        box = QLabel("", parent=holder)
        box.setFixedSize(PANE_PX, PANE_PX)
        box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.setWordWrap(True)
        box.setStyleSheet(_BOX_STYLE)
        caption = QLabel("", parent=holder)
        caption.setWordWrap(True)
        caption.setStyleSheet(_CAPTION_STYLE)
        holder.setFixedWidth(PANE_PX)
        lay.addWidget(head)
        lay.addWidget(box)
        lay.addWidget(caption)
        lay.addStretch(1)
        return head, box, caption, holder

    def _short_pass(self) -> str:
        shorts = self._texts.get("pass_short")
        if isinstance(shorts, dict) and self._pass in shorts:
            return str(shorts[self._pass])
        return self._pass.upper()

    def _set_right_head(self) -> None:
        kind_word = self._t("map_prob") if self._kind == "probs" \
            else self._t("map_flows")
        short = self._short_pass()
        self.right_head.setText(self._t("right").format(
            **{"map": kind_word, "pass": short}))
        probs = self._kind == "probs"
        self.right_caption.setText(self._t("prob_key") if probs
                                   else self._t("flows_key"))
        self.reading_label.setText(self._t("reading") if probs
                                   else self._t("reading_flows"))

    # ── what the caller tells the window ───────────────────────────────
    def current_choice(self) -> Tuple[str, str]:
        return self._pass, self._kind

    def set_loading(self) -> None:
        """Both panes say "Loading…" (the first read)."""
        for box in (self.left_box, self.right_box):
            box.clear()
            box.setText(self._t("loading"))

    def set_loading_map(self) -> None:
        """The right pane says which pass is loading; the dropdowns stay
        live (a newer choice wins)."""
        self._map = None
        self._map_message = self._t("loading_map").format(
            **{"pass": self._short_pass()})
        self.right_box.clear()
        self.right_box.setText(self._map_message)

    def set_image(self, payload: Dict[str, Any]) -> None:
        """The first read's answer: the provenance, the model's input (or
        the shape-changed sentence) and the first map."""
        lines = payload.get("lines") or {}
        self.line1_label.setText(str(lines.get("line1") or ""))
        line2 = str(lines.get("line2") or "")
        self.line2_label.setText(line2)
        self.line2_label.setVisible(bool(line2))
        self._image = payload.get("image")
        shape = payload.get("shape")
        if shape:
            self._shape = tuple(int(s) for s in shape)
        if self._image is None and payload.get("shape_changed"):
            self.left_box.clear()
            self.left_box.setText(self._t("shape_changed"))
            self.left_caption.setText("")
        else:
            self.left_caption.setText(str(payload.get("caption") or ""))
        the_map = payload.get("map")
        if the_map is not None:
            self.set_map(the_map, reset_slider=True)
        else:
            self._reset_slider()
            self._render_now()
        self._grow()
        self._grow_timer.start()

    def set_map(self, the_map: Dict[str, Any], *,
                reset_slider: bool = False) -> None:
        """A map read for the CURRENT choice (the caller drops an older
        one)."""
        self._map = the_map
        self._map_message = ""
        shape = the_map.get("shape")
        if shape and self._shape is None:
            self._shape = tuple(int(s) for s in shape)
        if reset_slider:
            self._reset_slider()
        self._render_now()

    def set_failed(self, text: str) -> None:
        """The first read failed: both panes carry the sentence, and there
        is nothing to choose."""
        self.line1_label.setText("")
        for box in (self.left_box, self.right_box):
            box.clear()
            box.setText(str(text))
        self.slider.setEnabled(False)
        self.pass_combo.setEnabled(False)
        self.map_combo.setEnabled(False)

    def set_map_failed(self, text: str) -> None:
        """A map read failed, or was refused: the right pane says why."""
        self._map = None
        self._map_message = str(text)
        self.right_box.clear()
        self.right_box.setText(self._map_message)

    # ── the dropdowns and the slider ───────────────────────────────────
    def _on_combo_changed(self, _index: int) -> None:
        new_pass = self.pass_combo.currentData() or PP.PASSES[0]
        new_kind = self.map_combo.currentData() or PP.KINDS[0]
        pass_changed = new_pass != self._pass
        if new_pass == self._pass and new_kind == self._kind:
            return
        self._pass, self._kind = str(new_pass), str(new_kind)
        self._set_right_head()
        if pass_changed:
            self._reset_slider()
        self.set_loading_map()
        self._render_now()
        try:
            self._on_choice(self._pass, self._kind)
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP3-PRED] choice failed: {type(exc).__name__}: {exc}")

    def _reset_slider(self) -> None:
        """The slider's range follows the pass; it starts in the middle."""
        if self._shape is None:
            return
        n = PP.plane_count(self._shape, self._pass)
        was = self.slider.blockSignals(True)
        try:
            self.slider.setRange(0, max(n - 1, 0))
            self.slider.setValue(n // 2)
        finally:
            self.slider.blockSignals(was)
        self.slider.setEnabled(n > 1)
        self._set_slider_label()

    def _set_slider_label(self) -> None:
        if self._shape is None:
            self.slider_label.setText("")
            return
        n = PP.plane_count(self._shape, self._pass)
        self.slider_label.setText(self._t("slider").format(
            axis=PP.SLIDER_AXIS[self._pass], i=self.slider.value(),
            n=max(n - 1, 0)))

    def _on_slider_moved(self, _value: int) -> None:
        self._set_slider_label()
        self._render_timer.start()

    def _render_now(self) -> None:
        self._render_timer.stop()
        self._render()

    def _render(self) -> None:
        i = self.slider.value()
        room = 2 * PANE_PX
        img = self._image
        if img is not None:
            seg = PP.plane(img["seg"], self._pass, i)
            step = PP.thin_step(seg.shape, room)
            helper = img.get("helper")
            helper = None if helper is None else PP.thin(
                PP.plane(helper, self._pass, i), step)
            self.last_left = PP.render_image(PP.thin(seg, step), helper)
            self._show(self.left_box, self.last_left)
        m = self._map
        if m is not None:
            if m.get("kind") == "flows":
                down, across = PP.flow_plane(m["comps"], self._pass, i)
                step = PP.thin_step(down.shape, room)
                self.last_right = PP.render_flow(
                    PP.thin(down, step), PP.thin(across, step),
                    m["lo"], m["hi"])
            else:
                prob = PP.plane(m["prob"], self._pass, i)
                self.last_right = PP.render_prob(
                    PP.thin(prob, PP.thin_step(prob.shape, room)),
                    magma_lut())
            self._show(self.right_box, self.last_right)

    @staticmethod
    def _show(box: QLabel, rgb: np.ndarray) -> None:
        pm = to_pixmap(rgb)
        room = PANE_PX - 2
        up = pm.width() < room and pm.height() < room
        box.setText("")
        box.setPixmap(pm.scaled(
            room, room, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation if up
            else Qt.TransformationMode.SmoothTransformation))

    # ── the foot ───────────────────────────────────────────────────────
    def _on_details_clicked(self) -> None:
        try:
            self._on_details()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP3-PRED] details failed: {type(exc).__name__}: {exc}")

    # ── size ───────────────────────────────────────────────────────────
    def _reading_room(self, width: int) -> int:
        """The reading line's height: the taller of its two sentences
        wrapped at the line's width in a window *width* wide."""
        m = self.layout().contentsMargins()
        room = max(width - m.left() - m.right(), 100)
        self.reading_label.ensurePolished()      # the style sheet's font
        fm = self.reading_label.fontMetrics()
        wrap = Qt.TextFlag.TextWordWrap
        flags = int(getattr(wrap, "value", wrap))
        return max(fm.boundingRect(QRect(0, 0, room, 100000), flags,
                                   text).height()
                   for text in (self._t("reading"), self._t("reading_flows"),
                                " "))

    def _needed_height(self, width: int) -> int:
        lay = self.layout()
        lay.invalidate()          # the labels' new texts, measured afresh
        if lay.hasHeightForWidth():
            return lay.totalHeightForWidth(width)
        return self.sizeHint().height()

    def _fit(self) -> None:
        hint = self.sizeHint()
        width = min(max(hint.width(), self.minimumWidth()), MAX_W)
        self.reading_label.setFixedHeight(self._reading_room(width))
        self.resize(width, self._needed_height(width))
        self._fit_to_screen()

    def _grow(self) -> None:
        """Taller when what the window now shows needs it (a provenance
        line that wraps); never shorter, so nothing jumps."""
        need = self._needed_height(self.width())
        if need > self.height():
            self.resize(self.width(), need)
            self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """Inside the available area of the screen it is shown on (a
        second monitor's scale and size are known only once it is
        shown)."""
        screen = self.screen()
        if screen is None:
            return
        area = screen.availableGeometry()
        w = min(self.width(), max(area.width() - 20, 200))
        h = min(self.height(), max(area.height() - 40, 200))
        if (w, h) != (self.width(), self.height()):
            self.resize(w, h)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_to_screen()

    def closeEvent(self, event):
        """The arrays go with the window (it is deleted later)."""
        self._render_timer.stop()
        self._grow_timer.stop()
        self._image = None
        self._map = None
        self.last_left = None
        self.last_right = None
        super().closeEvent(event)

from __future__ import annotations

import json
import os
from functools import lru_cache
from html import escape
from pathlib import Path as _Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from qtpy.QtGui import QColor, QImage, QPixmap

from packages.epicycle_preprocessing import recipe as _recipe
from packages.epicycle_preprocessing import registry as _op_registry

from ..ui_shell.theme import (
    Accordion, CANVAS_AREA_STYLE, DIM_LABEL_STYLE,
    FLOW_HINT_STYLE, FLOW_TITLE_STYLE, FLOW_SUBTITLE_STYLE,
    INPUT_STYLE, LIST_STYLE, LIST_STYLE_TEAL, LOG_OUTPUT_STYLE,
    MONO_LABEL_STYLE,
    PROGRESS_BAR_STYLE_TEAL, SCROLLBAR_STYLE,
    PRIMARY_BTN_STYLE_TEAL, SECONDARY_BTN_STYLE, SECTION_HEADER_STYLE,
    SEPARATOR_LINE_STYLE, SEPARATOR_STYLE, SPLITTER_STYLE,
    apply_button_icon, apply_inner_group_style, elide_label, help_button,
    toggle_btn_style,
)
from .controller import (
    DataPreparationController,
    ZOOM_DEPTH_LABELS,
    ZOOM_DEPTH_MODES,
    _channel_display_hex,
    LIBRARY_ROLE_REL_PATH,
    LIBRARY_ROLE_INTERACTIVE,
)
from ._paths import step1_patches_dir

#: Step 1 = Setup band = teal (`workflow_shell.py:49-56`).  Steps 2–3
#: are amber, 4–5 green.  Chrome is teal / grey / red only — green and magenta
#: are fluorophore LUTs, i.e. data, not chrome.
_STEP1_ACCENT = "#5ba4cf"

#: Resolved relative to this file so an editable install works without an
#: absolute path.  Same convention as Step 3 (`fusion_reconstruction/ui.py:55`).
from napari_epicycle.ui_shell._guides import guides_dir as _guides_dir
_GUIDES_DIR = _guides_dir()

#: The `+ add op` menu is the only QMenu in the repo, and `theme.py` has no
#: QMenu rule — so an unstyled menu renders in the OS light palette inside a
#: dark dock.  Kept local because this phase leaves `theme.py` alone;
#: its proper home is there, alongside the other shared styles.
_MENU_STYLE = (
    "QMenu {"
    "  background-color: #2b2b2b;"
    "  border: 1px solid rgba(255, 255, 255, 0.12);"
    "  padding: 4px; font-size: 11px; color: #dddddd;"
    "}"
    "QMenu::item { padding: 4px 22px 4px 12px; border-radius: 3px; }"
    "QMenu::item:selected { background-color: rgba(91, 164, 207, 0.28); }"
    "QMenu::item:disabled {"
    "  color: #888888; font-weight: bold; padding: 6px 8px 2px 6px;"
    "}"
    "QMenu::separator {"
    "  height: 1px; background: rgba(255,255,255,0.10); margin: 4px 6px;"
    "}"
)

#: The ④ Snapshot split button (`QToolButton`, MenuButtonPopup).  Mirrors the
#: teal secondary look (`SECONDARY_BTN_STYLE`) but targets `QToolButton` and
#: carries a menu-indicator arrow separator so the ▾ half reads as a split.
_SNAPSHOT_BTN_STYLE = (
    "QToolButton {"
    "  background-color: rgba(255, 255, 255, 0.05);"
    "  border: 1px solid rgba(255, 255, 255, 0.10);"
    "  border-radius: 4px; padding: 5px 10px;"
    "  font-size: 10px; color: #cccccc;"
    "}"
    "QToolButton:hover {"
    "  background-color: rgba(255, 255, 255, 0.10);"
    "  border: 1px solid rgba(255, 255, 255, 0.18);"
    "}"
    "QToolButton:pressed { background-color: rgba(255, 255, 255, 0.15); }"
    "QToolButton:disabled {"
    "  background-color: rgba(255, 255, 255, 0.02); color: #666666;"
    "}"
    "QToolButton::menu-button {"
    "  border-left: 1px solid rgba(255, 255, 255, 0.12); width: 14px;"
    "}"
    # Constrain the ▾ indicator — unstyled Qt draws a chunky native arrow.
    "QToolButton::menu-arrow { width: 7px; height: 7px; }"
    "QToolButton::menu-arrow:open { top: 1px; left: 1px; }"
)

#: The `+ Add step` menu + the whole recipe engine are now driven by the REAL,
#: pure registry in `packages/epicycle_preprocessing`: one key-based
#: table that feeds BOTH the menu (`registry.by_bucket()`) and the executor, so
#: the UI can never offer an op the engine cannot run.  The old UI-side
#: `_OP_SUITE` placeholder is gone (Brick 1b).  `percentile_normalize` is
#: deliberately ABSENT from the registry too — it is the implicit,
#: always-first, non-removable working-space stage.


#: The three box statuses, encoded THREE times — colour, line glyph and row
#: text — so status survives colourblindness (M4).
_BOX_STATUS = {
    "new":      (_STEP1_ACCENT, "─", "new"),
    "staged":   ("#80deea",     "═", "staged"),
    "captured": ("#8a9bae",     "┈", "captured"),
}

#: One legend string, shared by both Box List instances ("one
#: widget, one row grammar, one status palette").
_BOX_LEGEND = "─ new    ═ staged    ┈ captured"

#: ④ row action buttons (Brick 2/4): the live location-eye + capture + the
#: (placeholder) delete.  Neutral idle style + a magenta "eye on" style that
#: matches the location box colour drawn in the source volume.
_ROW_ACTION_STYLE = (
    "QPushButton { border: 1px solid rgba(255,255,255,0.10);"
    " border-radius: 3px; color: #b8b8b8; background: transparent; }"
    "QPushButton:hover { border-color: #3fb6b6; color: #ffffff; }"
    "QPushButton:disabled { color: #5c5c5c;"
    " border-color: rgba(255,255,255,0.05); }")
_ROW_EYE_ON_STYLE = (
    "QPushButton { border: 1px solid #d451dc;"
    " border-radius: 3px; color: #ff66ee;"
    " background: rgba(212,81,220,0.14); }"
    "QPushButton:hover { border-color: #ff66ee; color: #ffffff; }")
#: The capture button stays LIT CYAN while that patch's inspector is OPEN
#: (not just on hover) — a persistent "this one's open" cue.
_ROW_CAPTURE_ON_STYLE = (
    "QPushButton { border: 1px solid #3fb6b6;"
    " border-radius: 3px; color: #5fd6d6;"
    " background: rgba(63,182,182,0.16); }"
    "QPushButton:hover { border-color: #5fd6d6; color: #ffffff; }")
#: Delete is a DESTRUCTIVE action → red/orange hover, not the neutral cyan.
_ROW_TRASH_STYLE = (
    "QPushButton { border: 1px solid rgba(255,255,255,0.10);"
    " border-radius: 3px; color: #b8b8b8; background: transparent; }"
    "QPushButton:hover { border-color: #e0664b; color: #ff8a6a;"
    " background: rgba(224,102,75,0.12); }"
    "QPushButton:disabled { color: #5c5c5c;"
    " border-color: rgba(255,255,255,0.05); }")

#: One ceiling for every box-size control.  The geometry slider (0..2048), the
#: geometry spinbox (0..100000) and `bbox_size_spinbox` (1..4096) used to
#: disagree, which made the slider⇄spinbox pair lossy above 2048 and left the
#: two size models without a shared upper bound.
_GEOM_MAX = 4096

#: ②'s per-channel card strip scrolls inside a themed Box-List FRAME, so the
#: QScrollArea itself must be TRANSPARENT + borderless and let the frame draw
#: the box.  This is the merge-dialog `_SCROLL_AREA_STYLE` idiom
#: (`annotator/merge_dialog.py:197`), inlined because ui.py may not make a
#: cross-step import.
_RECIPE_SCROLL_STYLE = (
    "QScrollArea { background: transparent; border: none; }"
    "QScrollArea > QWidget > QWidget { background: transparent; }"
    + SCROLLBAR_STYLE
)


# ---------------------------------------------------------------------------
# ④ Library — sidecar detail + saved-patch thumbnail (F2, cached, cheap)
# ---------------------------------------------------------------------------
# A captured/on-disk patch has no in-memory pixels, so its row detail comes from
# the FROZEN `_metadata.json` sidecar (no pixel read) and its thumbnail from a
# mid-Z slice of the small patch TIFF.  Both are LRU-cached by (path, mtime) so
# a scroll / refresh never re-reads.  Defensive throughout: a missing / corrupt
# file yields "" (detail) or None (thumbnail), never an exception.


def _step1_patches_dir(ws: str) -> str:
    """The canonical Step-1 patches directory under a workspace.

    B3 folded the inline duplicate into the
    module that owns the layout; this thin wrapper keeps the `str` return
    every call site here expects.  `._paths` is same-package, so the
    no-cross-step-import rule is untouched."""
    return str(step1_patches_dir(ws))


@lru_cache(maxsize=1024)
def _sidecar_detail_for_path(path: str, mtime_ns: int) -> str:
    """`{Y}×{X} · {C}ch · Z{lo}–{hi}` from a patch `_metadata.json` sidecar.

    Derived from `cropped_shape` (Y=shape[-2], X=shape[-1]; Z=shape[1] for a
    4D CZYX patch, shape[0] for a 3D ZYX patch) + `channel_names` (C=len), with
    the truncated `selected_z_range` preferred for the Z segment when present.
    NO pixel read.  Returns "" on missing / corrupt / non-dict sidecar.  Cached
    by (path, mtime_ns) — the mtime key invalidates a rewritten sidecar."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return ""
    if not isinstance(meta, dict):
        return ""
    shape = meta.get("cropped_shape")
    chans = meta.get("channel_names")
    bits = []
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        bits.append(f"{shape[-2]}×{shape[-1]}")
    if isinstance(chans, (list, tuple)) and chans:
        bits.append(f"{len(chans)}ch")
    zr = meta.get("selected_z_range")
    if isinstance(zr, (list, tuple)) and len(zr) == 2:
        bits.append(f"Z{int(zr[0])}–{int(zr[1])}")
    elif isinstance(shape, (list, tuple)) and len(shape) == 4:
        bits.append(f"Z{int(shape[1])}")
    elif isinstance(shape, (list, tuple)) and len(shape) == 3:
        bits.append(f"Z{int(shape[0])}")
    return " · ".join(bits)


def _thumb_norm8(arr: np.ndarray) -> np.ndarray:
    """Percentile-clip a 2D plane to uint8 [0, 255] for display."""
    a = arr.astype(np.float32)
    lo = float(np.percentile(a, 1))
    hi = float(np.percentile(a, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _thumb_stride(arr: np.ndarray, size: int) -> np.ndarray:
    """Stride-downsample a 2D plane so its longest edge is ~<= *size* (fast,
    good enough for a tiny thumbnail)."""
    h, w = arr.shape[:2]
    if h <= size and w <= size:
        return arr
    sh = max(1, h // size)
    sw = max(1, w // size)
    return arr[::sh, ::sw]


def _compose_patch_rgb(arr: np.ndarray, size: int):
    """Reduce a saved patch array to an (H, W, 3) uint8 RGB preview.

    * 4D CZYX  → mid-Z slice, up to the first 3 channels mapped to R/G/B.
    * 3D ZYX   → grayscale max-intensity projection over Z.
    * 2D YX    → grayscale as-is.
    Returns None for any other shape."""
    arr = np.asarray(arr)
    if arr.ndim == 4:                       # (C, Z, Y, X)
        mid = arr.shape[1] // 2
        planes = [arr[c, mid] for c in range(min(int(arr.shape[0]), 3))]
        gray = len(planes) == 1
    elif arr.ndim == 3:                     # (Z, Y, X) single channel
        planes = [arr.max(axis=0)]
        gray = True
    elif arr.ndim == 2:                     # (Y, X)
        planes = [arr]
        gray = True
    else:
        return None
    planes = [_thumb_norm8(_thumb_stride(np.asarray(p), size)) for p in planes]
    h, w = planes[0].shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if gray:
        out[..., 0] = out[..., 1] = out[..., 2] = planes[0]
    else:
        for i, p in enumerate(planes[:3]):
            out[..., i] = p
    return out


def _read_patch_midz_planes(path: str, max_channels: int = 3):
    """Read ONLY the mid-Z Y×X plane of up to `max_channels` channels from a
    saved patch TIFF — a few PAGES via ``key=``, never the whole volume (the
    thumbnail RAM guard: a large ROI crop must not load multi-GB into RAM just
    to draw a preview).  Returns an ``(n, Y, X)`` array (n ≤ max_channels), or
    None.  Assumes the persisted patch axis order (``CZYX`` for multi-channel,
    else ``ZYX`` / ``YX``), which is exactly what ``persist_patch`` writes."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = series.axes or ""
        dims = dict(zip(axes, series.shape))
        n_c = int(dims.get("C", 1))
        n_z = int(dims.get("Z", 1))
        n_ch = max(1, min(n_c, int(max_channels)))
        z_mid = n_z // 2
        if "Z" in axes and "C" in axes:      # CZYX: page(c) = c*Z + z_mid
            keys = [c * n_z + z_mid for c in range(n_ch)]
        elif "Z" in axes:                    # ZYX: the single mid-Z page
            keys = [z_mid]
        elif "C" in axes:                    # CYX (no Z): one page per channel
            keys = list(range(n_ch))
        else:                                # YX (2D)
            keys = [0]
        raw = tf.asarray(key=keys if len(keys) > 1 else keys[0])
    planes = np.asarray(raw)
    if planes.ndim == 2:                     # a single page -> (1, Y, X)
        planes = planes[None]
    return planes


@lru_cache(maxsize=256)
def _patch_thumbnail_pixmap(path: str, mtime_ns: int, size: int):
    """A `size`×`size` QPixmap preview of a saved patch TIFF, or None.

    Reads ONLY the mid-Z plane of up to 3 channels (`_read_patch_midz_planes`,
    a few pages — never the whole volume) and composes them to RGB (≤3 channels
    → R/G/B; 1 → grayscale).  Cached by (path, mtime_ns, size) so scrolling ④
    never re-reads.  Returns None on any read/compose failure — the caller then
    shows the empty placeholder."""
    try:
        planes = _read_patch_midz_planes(path)
    except Exception:
        return None
    if planes is None or planes.ndim != 3 or planes.shape[0] == 0:
        return None
    try:
        chans = [_thumb_norm8(_thumb_stride(planes[i], int(size)))
                 for i in range(min(int(planes.shape[0]), 3))]
    except Exception:
        return None
    if not chans:
        return None
    h, w = chans[0].shape[:2]
    rgb_c = np.zeros((h, w, 3), dtype=np.uint8)
    if len(chans) == 1:
        rgb_c[..., 0] = rgb_c[..., 1] = rgb_c[..., 2] = chans[0]
    else:
        for i, p in enumerate(chans[:3]):
            rgb_c[..., i] = p
    rgb_c = np.ascontiguousarray(rgb_c)
    qimg = QImage(rgb_c.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    pm = QPixmap.fromImage(qimg.copy())     # .copy() detaches from the buffer
    if pm.isNull():
        return None
    scaled = pm.scaled(int(size), int(size), Qt.KeepAspectRatio,
                       Qt.SmoothTransformation)
    # Round the corners so the preview matches its rounded card (the square
    # pixmap otherwise overpaints the container's rounded corners).
    return _round_pixmap(scaled, 4)


def _round_pixmap(pm, radius: int):
    """A copy of *pm* with rounded corners (transparent outside the radius), so
    it sits flush inside a rounded card.  Best-effort — returns *pm* unchanged on
    any failure."""
    try:
        from qtpy.QtGui import QPainter, QPainterPath
        from qtpy.QtCore import QRectF
        w, h = pm.width(), pm.height()
        if w <= 0 or h <= 0:
            return pm
        rounded = QPixmap(w, h)
        rounded.fill(Qt.transparent)
        painter = QPainter(rounded)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            path = QPainterPath()
            path.addRoundedRect(QRectF(0, 0, w, h), float(radius), float(radius))
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, pm)
        finally:
            painter.end()
        return rounded
    except Exception:
        return pm


# Row-name budgets.  A list row is a fixed-width thing with a button cluster on
# its right; a name longer than this is elided (full text on hover) rather than
# allowed to push the buttons off the edge.
ROW_NAME_MAX_PX = 190
# Right margin of a box row, and the gap between the name and the dimensions.
_DIM_PAD_PX, _DIM_GAP_PX = 10, 8
# Below this the name is not worth painting at all, so the dims stand down.
_DIM_MIN_NAME_PX = 40


def reserved_dim_width(dim_text_width: int, row_width: int,
                       pad: int = _DIM_PAD_PX, gap: int = _DIM_GAP_PX) -> int:
    """Pixels to reserve at the RIGHT of a box row for its ``ZxYxX`` dimensions.

    ``0`` means "do not paint the dimensions": either there are none, or the row
    is too narrow to carry both and the name wins.  Pure arithmetic, so the
    "dims drawn ON TOP of the name" bug is testable without a paint device.
    """
    if dim_text_width <= 0:
        return 0
    reserve = int(dim_text_width) + pad + gap
    if int(row_width) - reserve < _DIM_MIN_NAME_PX:
        return 0
    return reserve


class _BoxDimDelegate(QStyledItemDelegate):
    """Paints a dimmed ``Z×Y×X`` voxel size (stored in ``Qt.UserRole+8``)
    right-aligned after a ③ box row's name — self-documenting rows WITHOUT
    touching ``item.text()`` (every text reader stays intact).  A no-op when the
    role is unset (e.g. the legacy list path), so it is safe on the shared list.
    """

    def paint(self, painter, option, index):
        dim = index.data(Qt.UserRole + 8)
        if not dim:
            super().paint(painter, option, index)
            return
        # Reserve a column for the dims and ELIDE THE NAME to what is left,
        # then let the base delegate paint the row across its FULL width.
        # Painting the name unelided and the dims on top of it (what this used
        # to do) made the two overlap on any row whose name reached the right
        # edge; narrowing the base delegate's rect instead would fix the overlap
        # but leave the selection / hover strip short of the dims column.
        dim = str(dim)
        reserve = reserved_dim_width(
            option.fontMetrics.horizontalAdvance(dim), option.rect.width())
        if not reserve:
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = option.fontMetrics.elidedText(
            opt.text, Qt.ElideMiddle,
            max(0, option.rect.width() - reserve - _DIM_PAD_PX))
        super().paint(painter, opt, index)
        painter.save()
        painter.setPen(QColor("#7a8290"))
        painter.drawText(option.rect.adjusted(0, 0, -_DIM_PAD_PX, 0),
                         Qt.AlignRight | Qt.AlignVCenter, dim)
        painter.restore()


class BoxListPanel(QWidget):
    """The Box List — the linchpin, used by BOTH ③ and ④.

    The separate "Extracted ROIs" list is gone, and ④'s
    gallery is **the same component** as ③'s: "one widget, one row grammar, one
    status palette".  ③ shows all boxes; ④ filters to those with pixels and
    draws larger thumbnails.  Those are the only differences.

    The inner list is a plain `QListWidget` because the shell requires it: ④'s
    instance is exposed as `extracted_roi_list`, and the shell calls `.model()`
    for three signals plus `.currentItem()`, `.count()` and `.item(0).text()`
    on it.

    **`item.setText(name)` is always set, even though a widget is overlaid.**
    Seven controller sites key into `extracted_patches` off `item.text()`
    (`controller.py:776, 917, 936, 963, 984, 1093, 1118`) and
    `workflow_shell._read_active_roi` reads it too — a `setItemWidget` row that
    leaves the text empty breaks plot, snap, Set Range and export *silently*.
    """

    def __init__(self, owner, thumb: int = 0, absent_status: str = "captured",
                 row_actions: str = "box", count_noun: str = "extractable",
                 parent=None):
        super().__init__(parent)
        self._owner = owner
        self._thumb = thumb
        # The action cluster is PANEL-SCOPED, because one
        # component serves two panels and the glyph means different things in
        # each: in ③ the bin deletes a *box*; in ④ it discards a staged patch
        # or guard-deletes a captured one.  v4.0 justified consolidating the
        # export on "🗑 is already in the row" — which was false for captured
        # rows, and left nothing saying how a staged patch gets discarded.
        self._row_actions = row_actions
        self._count_noun = count_noun
        # What a row means when its name is NOT in `extracted_patches`.  In ③
        # that is a freshly drawn box (`new`); in ④ it is a patch already on
        # disk (`captured`).  Without this every ③ row rendered grey/`captured`
        # — which in this panel's vocabulary means locked and non-extractable,
        # the opposite of what Extract will do to it.
        self._absent_status = absent_status
        # When True, `refresh()` keeps the count / empty / select-all in sync but
        # does NOT overlay the per-row card widget.  ③'s box tool sets this so
        # each row shows exactly ONE (native) checkbox and InternalMove drag-
        # reorder works — the card's own QCheckBox was the SECOND tick, and a
        # setItemWidget editor breaks the drag.  ④'s instance
        # leaves it False and keeps the rich row cards.
        self._suppress_row_widgets = False

        # ④ location-eye + capture callbacks (Brick 2/4), wired by the controller
        # via `set_location_eye` / `set_capture_callback`.  None until wired, so a
        # row's 👁 / capture button renders disabled (③ never sets these).
        self._loc_eye_toggle = None     # fn(patch_name) -> toggle its location
        self._loc_eye_state = None      # fn(patch_name) -> bool (currently shown)
        self._capture_cb = None         # fn(patch_name) -> open capture view
        self._capture_open_state = None # fn(patch_name) -> bool (inspector open?)
        self._delete_cb = None          # fn(patch_name) -> delete that patch
        self._double_click_cb = None    # fn(patch_name) -> double-click a row
        # ④ per-patch box styling (P2 Brick 2), wired by the controller via
        # `set_style_callback`.  `_style_click_cb(name, global_pos)` opens the
        # style popup; `_swatch_color_fn(name) -> (r,g,b)` (0..1) tints the row
        # swatch to the patch's resolved box colour.
        self._style_click_cb = None
        self._swatch_color_fn = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        head = QHBoxLayout()
        self._head = head
        head.setSpacing(6)
        self.select_all_check = QCheckBox("all")
        self.select_all_check.setToolTip("Tick or untick every box")
        head.addWidget(self.select_all_check)
        head.addStretch(1)
        self.count_label = QLabel("0")
        self.count_label.setStyleSheet(DIM_LABEL_STYLE)
        head.addWidget(self.count_label)
        lay.addLayout(head)

        self.list = QListWidget()
        self.list.setStyleSheet(LIST_STYLE_TEAL)
        self.list.setMinimumHeight(120)
        lay.addWidget(self.list, 1)

        self.empty_label = QLabel("")
        self.empty_label.setWordWrap(True)
        self.empty_label.setStyleSheet(DIM_LABEL_STYLE)
        lay.addWidget(self.empty_label)

        self.legend_label = QLabel(_BOX_LEGEND)
        self.legend_label.setStyleSheet(DIM_LABEL_STYLE)
        lay.addWidget(self.legend_label)

        self.select_all_check.toggled.connect(self._toggle_all)
        self.list.itemDoubleClicked.connect(self._on_item_double_clicked)
        for sig in (self.list.model().rowsInserted,
                    self.list.model().rowsRemoved,
                    self.list.model().modelReset):
            sig.connect(self.refresh)

    def set_empty_text(self, text: str) -> None:
        self.empty_label.setText(text)
        self.refresh()

    def set_location_eye(self, toggle_cb, state_fn) -> None:
        """Wire ④'s per-row 👁 to the controller (Brick 2): ``toggle_cb(name)``
        shows/hides that patch's location box in the source volume;
        ``state_fn(name) -> bool`` reports whether it is currently shown (drives
        the eye icon + magenta highlight)."""
        self._loc_eye_toggle = toggle_cb
        self._loc_eye_state = state_fn
        self.refresh()

    def set_style_callback(self, click_cb, color_fn) -> None:
        """Wire ④'s per-row colour swatch (P2 Brick 2): ``click_cb(name,
        global_pos)`` opens the box-style popup; ``color_fn(name) -> (r, g, b)``
        (0..1) gives the patch's resolved box colour to tint the chip."""
        self._style_click_cb = click_cb
        self._swatch_color_fn = color_fn
        self.refresh()

    def _make_row_swatch(self, lay, name) -> None:
        """A 16px colour chip = the patch's resolved box colour; click → the
        per-patch box-style popup.  Only in the ④ ('patch') panel."""
        sw = QToolButton()
        sw.setFixedSize(16, 16)
        sw.setCursor(Qt.PointingHandCursor)
        sw.setToolTip("Box colour + style for this patch")
        rgb = None
        if self._swatch_color_fn is not None:
            try:
                rgb = self._swatch_color_fn(name)
            except Exception:
                rgb = None
        if rgb is not None:
            r, g, b = (int(round(min(1.0, max(0.0, float(c))) * 255)) for c in rgb[:3])
        else:
            r, g, b = 210, 90, 190   # magenta fallback (overlay default)
        sw.setStyleSheet(
            f"QToolButton {{ background-color: rgb({r},{g},{b});"
            "  border: 1px solid rgba(255,255,255,0.35); border-radius: 3px; }"
            "QToolButton:hover { border: 1px solid rgba(255,255,255,0.7); }")
        if self._style_click_cb is not None:
            sw.clicked.connect(
                lambda _=False, n=name, b=sw: self._style_click_cb(
                    n, b.mapToGlobal(b.rect().bottomLeft())))
        else:
            sw.setEnabled(False)
        lay.addWidget(sw, 0)

    def set_header_left(self, *widgets) -> None:
        """Insert *widgets* at the LEFT of the header row — before the (hidden-in-
        ④) 'all' checkbox, the stretch, and the 'N patch(es)' count.  ④ uses this
        to place its "Show locations" + Snapshot controls INLINE with the count
        (one row), instead of on a separate strip above the list."""
        for i, wdg in enumerate(widgets):
            if wdg is not None:
                self._head.insertWidget(i, wdg)

    def set_capture_callback(self, cb, open_state=None) -> None:
        """Wire ④'s per-row capture button (Brick 4): ``cb(name)`` opens the
        capture view for that saved patch.  ``open_state(name) -> bool`` (optional)
        reports whether that patch's inspector is currently OPEN, so the button
        stays LIT CYAN while it is (not just on hover)."""
        self._capture_cb = cb
        self._capture_open_state = open_state
        self.refresh()

    def set_double_click_callback(self, cb) -> None:
        """Wire a ④ row DOUBLE-click: ``cb(name)`` highlights that patch's box in
        the source volume (the signal is connected once in ``__init__``)."""
        self._double_click_cb = cb

    def _on_item_double_clicked(self, item) -> None:
        if self._double_click_cb is not None and item is not None:
            try:
                self._double_click_cb(item.text())
            except Exception:
                pass

    def set_delete_callback(self, cb) -> None:
        """Wire ④'s per-row delete (bin) button: ``cb(name)`` deletes that saved
        patch (the controller confirms, then removes its registry row + on-disk
        TIFF/sidecar + session stage + box signature)."""
        self._delete_cb = cb
        self.refresh()

    def _toggle_all(self, checked: bool) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.refresh()

    def _sync_select_all(self) -> None:
        """Make the header follow its rows, tri-state.

        It started UNCHECKED while `make_checkable` ticked every row, so at rest
        the header contradicted all of them and the user's first click on "all"
        was a visible no-op (measured: the same tick set before and after).
        Partially-ticked is now the third state rather than being rounded to
        one of the two.
        """
        tickable = [self.list.item(i) for i in range(self.list.count())
                    if self.list.item(i).flags() & Qt.ItemIsUserCheckable]
        ticked = sum(1 for it in tickable if it.checkState() == Qt.Checked)
        box = self.select_all_check
        was = box.blockSignals(True)     # cosmetic sync must not re-enter
        try:
            box.setTristate(bool(tickable) and 0 < ticked < len(tickable))
            box.setCheckState(
                Qt.Checked if tickable and ticked == len(tickable)
                else Qt.PartiallyChecked if ticked
                else Qt.Unchecked)
        finally:
            box.blockSignals(was)

    def checked_names(self) -> list:
        """Names of the ticked rows — what `Extract checked` acts on."""
        return [self.list.item(i).text() for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def _tickable(self, name: str) -> bool:
        """Captured rows are NOT tickable, in EITHER panel.

        In ③ because captured boxes are non-extractable; in ④ because
        they are already on disk, so "export" is a no-op or an overwrite of a
        committed patch — which is what "captured locked" has always meant.
        Consequence: `[☑ all]` and the count both scope to the rest.
        """
        return self._status_of(name) != "captured"

    def make_checkable(self) -> None:
        """③: give every tickable box row a native check state (its "Extract
        checked" tick), and lock the captured ones.

        ④ (Review & Library) has **no tick UI** any more — Extract writes
        straight to disk, so there is nothing to "export checked".  A plain
        ``addItem`` (and the old export logic) leaves a native check indicator on
        each row, and the row-card widget sits ON TOP of it — so it shows but is
        unclickable.  Strip BOTH the checkable flag AND the CheckStateRole from
        every ④ row so no indicator renders.  (The checkbox mechanism is retained
        for a future ④ multi-select feature — P2 restructure — just disabled.)
        """
        if self._row_actions == "patch":
            for i in range(self.list.count()):
                item = self.list.item(i)
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                item.setData(Qt.CheckStateRole, None)   # remove the indicator
            return
        for i in range(self.list.count()):
            item = self.list.item(i)
            # ④ rows belonging to a NON-loaded source (or legacy source_id == "")
            # are non-interactive: never tickable, even if a stale in-memory
            # stage from a prior source would make `_status_of` read "staged"
            # (which would otherwise re-arm the tick + let Export write a patch
            # from a source that is no longer loaded).  Item-based, so it needs
            # the provenance `_refresh_library_from_registry` stamped.
            non_interactive = (self._row_actions == "patch"
                               and item.data(LIBRARY_ROLE_INTERACTIVE) is False)
            if non_interactive or not self._tickable(item.text()):
                # Re-evaluated every refresh, NOT once: a row's status changes
                # under it (staged → captured at export), and a patch that has
                # been committed must stop being exportable at that moment.
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                continue
            # MEASURED: `addItem(str)` already carries `ItemIsUserCheckable`,
            # but leaves `checkState()` at Unchecked — so guarding on the FLAG
            # never fires and every row reads as unticked.  Guard on an explicit
            # init marker instead, so a row the user deliberately unticks is not
            # re-checked by the next refresh.
            if item.data(Qt.UserRole + 7) is None:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                item.setData(Qt.UserRole + 7, True)

    def refresh(self, *_args) -> None:
        """Re-render every row, and keep the count and empty state honest.

        Rebuilt wholesale so a row can never describe a different box than the
        one its buttons act on.
        """
        self.make_checkable()
        n = self.list.count()
        visible = sum(1 for i in range(n) if not self.list.item(i).isHidden())
        self._sync_select_all()
        # Panel-aware count.  ④ (`row_actions == "patch"`) has no tick column any
        # more (F1 writes Extract straight to disk), so it simply reports the
        # TOTAL number of patches on disk.  ③ still names what its tick column is
        # FOR ("2 extractable"), scoped to the tickable subset —
        # captured rows are excluded, being untickable.
        if self._row_actions == "patch":
            text = f"{n} patch(es)" if n else "0"
        else:
            ready = sum(1 for i in range(n)
                        if self._tickable(self.list.item(i).text()))
            text = f"{ready} {self._count_noun}" if n else "0"
        if visible != n:
            text += f" · {visible} of {n} shown"
        self.count_label.setText(text)
        self.empty_label.setVisible(n == 0)
        # Legend hidden: the new/staged/captured status glyphs it explained were
        # dropped 2026-08-18 (④ patches are always on disk; ③ boxes are always
        # "new" this-session), so the legend is now noise in both panels.
        self.legend_label.setVisible(False)
        if self._suppress_row_widgets:
            # ③'s box tool owns rendering: one native checkbox per row + drag
            # reorder.  The row-card overlay below would add a second checkbox
            # and break InternalMove, so stop here (count/empty already synced).
            return
        for i in range(n):
            item = self.list.item(i)
            # Key on CONTENT, not the name.  The row is derived from
            # `_status_of`/`_detail_of`, which read `extracted_patches`; keying
            # on the name meant Set Range never updated the row and a staged
            # box kept saying "full Z" forever.
            eye = (bool(self._loc_eye_state(item.text()))
                   if self._loc_eye_state else None)
            # Include the resolved swatch colour so a box-STYLE change (P2 Brick 2)
            # re-tints the chip without needing an eye toggle to move the key.
            swatch = None
            if self._swatch_color_fn is not None:
                try:
                    c = self._swatch_color_fn(item.text())
                    swatch = tuple(round(float(x), 4) for x in c) if c else None
                except Exception:
                    swatch = None
            # Capture-inspector open state so the camera button re-styles cyan
            # (lit) the moment that patch's inspector opens / closes.
            capopen = (bool(self._capture_open_state(item.text()))
                       if self._capture_open_state else None)
            key = (item.text(), self._status_of(item.text()),
                   self._detail_of(item.text()),
                   item.checkState() == Qt.Checked, eye, swatch, capopen)
            if (self.list.itemWidget(item) is None
                    or item.data(Qt.UserRole + 9) != key):
                row = self._make_row(item)
                item.setSizeHint(row.sizeHint())
                self.list.setItemWidget(item, row)
                item.setData(Qt.UserRole + 9, key)

    def _status_of(self, name: str) -> str:
        """A patch held in memory is `staged`; one only on disk is `captured`.

        Read from the controller's existing `extracted_patches` dict — a read,
        not a logic change.
        """
        patches = getattr(self._owner.controller, "extracted_patches", {}) or {}
        return "staged" if name in patches else self._absent_status

    def _detail_of(self, name: str) -> str:
        """`512² · Z12–62 · 2ch`, from the in-memory patch where one exists.

        A ④ Library row with no in-memory data (a captured/on-disk patch — the
        cross-session case) falls back to the FROZEN sidecar (no pixel read).
        ③'s box list never reaches this fallback: `_suppress_row_widgets`
        short-circuits `refresh` before `_make_row` / `_detail_of` are called."""
        patches = getattr(self._owner.controller, "extracted_patches", {}) or {}
        info = patches.get(name)
        if info is not None:
            # In-memory staged patch — the historical path, unchanged.
            data = info.get("data")
            bits = []
            shape = getattr(data, "shape", None)
            if shape:
                bits.append(f"{shape[-2]}×{shape[-1]}")
                if len(shape) >= 4:
                    bits.append(f"{shape[0]}ch")
            zr = info.get("selected_z_range")
            bits.append(f"Z{int(zr[0])}–{int(zr[1])}" if zr and len(zr) == 2
                        else "full Z")
            return " · ".join(bits)
        if self._row_actions == "patch":
            return self._detail_from_sidecar(name)
        return ""

    def _detail_from_sidecar(self, name: str) -> str:
        """Size/channel/Z detail for a captured ④ row, read from the patch's
        `_metadata.json` sidecar under the workspace (LRU-cached; NO pixel
        read).  "" on missing workspace / sidecar (row shows just its name)."""
        try:
            ws = self._owner.output_file_path.text().strip()
        except Exception:
            return ""
        if not ws:
            return ""
        path = os.path.join(_step1_patches_dir(ws), f"{name}_metadata.json")
        try:
            mtime_ns = os.stat(path).st_mtime_ns
        except OSError:
            return ""
        return _sidecar_detail_for_path(path, int(mtime_ns))

    def _thumb_pixmap_for(self, item):
        """A cached QPixmap for a ④ Library row's saved patch TIFF, or None.

        Reads the TIFF path from the item's registry provenance
        (`LIBRARY_ROLE_REL_PATH`), falling back to the canonical patches-dir
        layout.  None (→ empty placeholder) when no workspace is set, the file
        is missing, or the read fails."""
        try:
            ws = self._owner.output_file_path.text().strip()
        except Exception:
            return None
        if not ws:
            return None
        rel = item.data(LIBRARY_ROLE_REL_PATH)
        if rel:
            path = os.path.join(ws, str(rel))
        else:
            path = os.path.join(_step1_patches_dir(ws), f"{item.text()}.tiff")
        try:
            mtime_ns = os.stat(path).st_mtime_ns
        except OSError:
            return None
        return _patch_thumbnail_pixmap(path, int(mtime_ns), int(self._thumb))

    def _make_row(self, item) -> QWidget:
        """`☑ · swatch · line-style · [thumb] · name · detail · status`."""
        name = item.text()
        status = self._status_of(name)
        # Status badges (new/staged/captured) were DROPPED (user decision
        # 2026-08-18): ④ patches are always on disk, so "staged"/"captured" was
        # noise and even flipped for the same patch across a reopen.  `status` is
        # still read below for the bin tooltip + action set, but no swatch/glyph/
        # badge is drawn.
        # ④ rows from a non-loaded source (or legacy source_id == "") are shown
        # but not interactive — no checkbox + greyed (parity with make_checkable,
        # which locks the same rows).  `is False` (not just falsy) so a row not
        # yet stamped by the registry refresh (role unset) keeps the pre-F2
        # behaviour.
        non_interactive = (self._row_actions == "patch"
                           and item.data(LIBRARY_ROLE_INTERACTIVE) is False)

        row = QWidget()
        # MEASURED: without an opaque background the delegate paints the item's
        # own text UNDER this overlay, so every row showed its raw name behind
        # the card's labels.  The text cannot be dropped — seven controller
        # sites and the shell key off `item.text()` — so the row covers it.
        row.setAutoFillBackground(True)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)

        # D1 (MEASURED): the item's built-in check indicator is UNCLICKABLE once
        # `setItemWidget` registers an editor for the index —
        # `QAbstractItemView::edit()` short-circuits the delegate's
        # `editorEvent`.  Ticks were therefore all-or-nothing via the header.
        # A real QCheckBox in the row is the `_import_pairs_dialog` pattern
        # used elsewhere in the plugin.
        #
        # Eye-on-the-LEFT (2026-08-19): EVERY ④ row carries the LIVE location-eye
        # (Brick 2) in the LEFT cell and NEVER a checkbox — F1 made Extract write
        # straight to disk, so ④'s old "export checked → Library" tick is dead
        # (nothing to commit).  ③'s box tool is `_suppress_row_widgets`, so its
        # rows never reach `_make_row` at all.  The `elif` QCheckBox branch below
        # is therefore now effectively DEAD (④ never checkboxes; ③ suppresses row
        # widgets) — left in place as harmless reference, not reachable.
        eye_on_left = (self._row_actions == "patch")
        if eye_on_left:
            self._make_row_eye(lay, name, non_interactive)
        elif self._tickable(name) and not non_interactive:
            check = QCheckBox()
            check.setToolTip("Include this box in the next "
                             + ("export" if self._row_actions == "patch"
                                else "extract"))
            check.setChecked(item.checkState() == Qt.Checked)
            check.toggled.connect(
                lambda on, it=item: it.setCheckState(
                    Qt.Checked if on else Qt.Unchecked))
            lay.addWidget(check, 0)
        else:
            # A blank of the same width, so the name column still lines up
            # across a mixed list instead of the captured rows shifting left.
            spacer = QLabel("")
            spacer.setFixedWidth(QCheckBox().sizeHint().width())
            spacer.setToolTip("captured — already on disk, nothing to commit")
            lay.addWidget(spacer, 0)

        # ④ per-patch box-style swatch (P2 Brick 2), after the eye cell.
        if self._row_actions == "patch" and not non_interactive:
            self._make_row_swatch(lay, name)

        if self._thumb:
            thumb = QLabel("")
            thumb.setFixedSize(self._thumb, self._thumb)
            thumb.setStyleSheet(CANVAS_AREA_STYLE)
            # ④ Library rows show a real RGB preview of the saved patch TIFF
            # (mid-Z slice, cached).  A missing / unreadable TIFF keeps the empty
            # placeholder.  ③'s box list has no thumb (`thumb=0`), so never here.
            if self._row_actions == "patch":
                pm = self._thumb_pixmap_for(item)
                if pm is not None:
                    thumb.setPixmap(pm)
                    thumb.setAlignment(Qt.AlignCenter)
            lay.addWidget(thumb, 0)

        label = QLabel()
        label.setStyleSheet("font-size: 11px; color: #dddddd;")
        elide_label(label, name, ROW_NAME_MAX_PX)   # full name on hover
        lay.addWidget(label, 1)

        detail = QLabel(self._detail_of(name))
        detail.setStyleSheet(DIM_LABEL_STYLE)
        lay.addWidget(detail, 0)

        # The row grammar's last cell (and drawn in the artifact) --
        # PANEL-SCOPED, because the same glyph commits different
        # things in the two panels.  ③: geometry actions on a *box*, and a
        # captured row shows look-and-hide only (the structural lock, expressed
        # in the UI).  ④: actions on a *patch* -- and its bin is where a staged
        # patch is discarded now that the detail pane's Discard is gone.
        if self._row_actions == "patch":
            # ④: LIVE capture button (Brick 4) + a placeholder delete; the
            # location-eye rides here only when it is NOT already on the left
            # (a tickable/staged row keeps its checkbox-left, eye-right).
            self._make_patch_row_actions(
                lay, name, non_interactive, include_eye=not eye_on_left)
        else:
            # ③: box actions — disabled placeholders (box model pending).
            if status != "captured":
                actions = (("✎", "Edit this box"), ("⧉", "Duplicate"),
                           ("👁", "Show / hide"), ("🗑", "Delete this box"))
            else:
                actions = (("👁", "Show / hide"),)
            for glyph, tip in actions:
                btn = QPushButton(glyph)
                btn.setFixedSize(17, 17)
                btn.setToolTip(tip)
                btn.setEnabled(False)
                btn.setStyleSheet(
                    "QPushButton { border: 1px solid rgba(255,255,255,0.08);"
                    " border-radius: 3px; font-size: 9px; color: #888888;"
                    " background: transparent; }")
                lay.addWidget(btn, 0)
                # v1: these do nothing (the box model behind them was never
                # built), so they are HIDDEN rather than shown greyed-out on
                # every row.  Built, not deleted -- turning them back on is one
                # line once they have something to commit.
                btn.setVisible(False)

        # Cross-source / legacy rows: grey the whole card + say why.  The item's
        # selectability was already cleared by `_refresh_library_from_registry`;
        # this is the visual half.
        if non_interactive:
            row.setEnabled(False)
            # An explicit `color:` in the label stylesheet wins over Qt's
            # disabled palette, so dim the name directly to make "greyed" read.
            label.setStyleSheet("font-size: 11px; color: #6a6a6a;")
            label.setToolTip(
                f"{name} — saved from another source; load that source to "
                "select or export it")

        return row

    def _make_row_eye(self, lay, name, non_interactive) -> None:
        """The LIVE 👁 location-eye for a ④ patch row (Brick 2): toggles whether
        that patch's bounding box is drawn in the source volume.  mdi icon (open
        eye when shown, with a magenta highlight; struck-through eye when
        hidden), ASCII fallback — never an emoji."""
        eye = QPushButton()
        eye.setFixedSize(22, 22)
        eye_on = bool(self._loc_eye_state and self._loc_eye_state(name))
        if not apply_button_icon(
                eye, "mdi6.eye" if eye_on else "mdi6.eye-off-outline"):
            eye.setText("O" if eye_on else "-")          # ASCII, not emoji
        eye.setToolTip(
            "Hide this patch's location box"
            if eye_on else
            "Show where this patch sits in the source volume")
        eye_wired = (self._loc_eye_toggle is not None) and not non_interactive
        eye.setEnabled(eye_wired)
        eye.setStyleSheet(
            _ROW_EYE_ON_STYLE if (eye_wired and eye_on) else _ROW_ACTION_STYLE)
        if eye_wired:
            eye.clicked.connect(
                lambda _=False, n=name: self._loc_eye_toggle(n))
        lay.addWidget(eye, 0)

    def _make_patch_row_actions(self, lay, name, non_interactive,
                                include_eye=True) -> None:
        """④ patch-row RIGHT action cluster: (optionally) the 👁 location-eye
        (Brick 2), a LIVE capture button (Brick 4), and a placeholder delete
        (F4).  Uses mdi icons (not emoji).  `include_eye` is False when the eye
        already sits in the row's LEFT cell (a captured row)."""
        if include_eye:
            self._make_row_eye(lay, name, non_interactive)

        # capture — open this patch in the capture inspector (render + save PNG).
        cap = QPushButton()
        cap.setFixedSize(22, 22)
        if not apply_button_icon(cap, "mdi6.camera-outline"):
            cap.setText("[]")
        cap.setToolTip("Open this patch in the capture view (render + save image)")
        cap_wired = (self._capture_cb is not None) and not non_interactive
        cap.setEnabled(cap_wired)
        cap_open = bool(self._capture_open_state
                        and self._capture_open_state(name))
        cap.setToolTip(
            "Capture view is OPEN for this patch" if cap_open
            else "Open this patch in the capture view (render + save image)")
        cap.setStyleSheet(
            _ROW_CAPTURE_ON_STYLE if (cap_wired and cap_open)
            else _ROW_ACTION_STYLE)
        if cap_wired:
            cap.clicked.connect(lambda _=False, n=name: self._capture_cb(n))
        lay.addWidget(cap, 0)

        # delete — remove this patch (registry row + TIFF/sidecar + stage),
        # wired to the controller's confirm-then-delete when a callback is set.
        trash = QPushButton()
        trash.setFixedSize(22, 22)
        if not apply_button_icon(trash, "mdi6.trash-can-outline"):
            trash.setText("x")
        del_wired = (getattr(self, "_delete_cb", None) is not None
                     and not non_interactive)
        trash.setToolTip(
            "Delete this patch — removes it from disk and the library"
            if del_wired else "Delete this patch — unavailable")
        trash.setEnabled(del_wired)
        trash.setStyleSheet(_ROW_TRASH_STYLE)   # destructive → red/orange hover
        if del_wired:
            trash.clicked.connect(lambda _=False, n=name: self._delete_cb(n))
        lay.addWidget(trash, 0)


class DataPreparationWidget(QWidget):
    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self._init_ui()

    # ------------------------------------------------------------------
    # Shell helpers
    # ------------------------------------------------------------------

    def _make_subsection_header(self, title: str,
                                guide_filename: str = None) -> QHBoxLayout:
        """Subsection header row: [styled label] [stretch] [? button].

        Copied locally rather than imported from `fusion_reconstruction/ui.py:110`
        — `ui.py` stays free of cross-step imports; the helper's proper
        home is `theme.py`, a later shared-code change.

        A **method**, not a module function, so it can register each guide on
        `self._help_guides` — that makes the set of wired `?` buttons
        inspectable without walking the widget tree.: a header row carries **only** title + stretch + `?`. No
        action ever enters a header row.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        label = QLabel(title)
        label.setStyleSheet(SECTION_HEADER_STYLE)
        row.addWidget(label)
        row.addStretch(1)
        # A `?` whose guide file does not exist opens a "not written yet" page,
        # which is worse than no `?` at all -- so the button is only built when
        # the guide is really on disk.  `_help_guides` therefore lists exactly
        # the buttons the user can see (asserted in test_ui.py).
        if guide_filename and (_GUIDES_DIR / guide_filename).is_file():
            self._help_guides.append(guide_filename)
            row.addWidget(help_button(
                self,
                guide_path=str(_GUIDES_DIR / guide_filename),
                title=title,
                accent=_STEP1_ACCENT,
                tooltip=f"Help: {title}",
            ))
        return row

    @staticmethod
    def _separator() -> QLabel:
        """The 1px rule this file repeats in several places."""
        sep = QLabel("")
        sep.setFixedHeight(1)
        sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        return sep

    def sync_show_locations_master(self) -> None:
        """Make the ④ "Show patch locations" master checkbox follow the per-row
        eyes, tri-state (Brick 2): checked when ALL current-source patches are
        shown, partial when SOME, unchecked when none.  Reads the controller's
        current-source patch set + `_visible_patch_locations`."""
        chk = getattr(self, "show_locations_check", None)
        ctl = getattr(self, "controller", None)
        if chk is None or ctl is None:
            return
        try:
            names = ctl._current_source_patch_names()
        except Exception:
            names = []
        visible = getattr(ctl, "_visible_patch_locations", set()) or set()
        total = len(names)
        shown = sum(1 for n in names if n in visible)
        was = chk.blockSignals(True)      # a cosmetic sync must not re-toggle
        try:
            chk.setTristate(bool(total) and 0 < shown < total)
            chk.setCheckState(
                Qt.Checked if total and shown == total
                else Qt.PartiallyChecked if shown
                else Qt.Unchecked)
        finally:
            chk.blockSignals(was)

    def _build_snapshot_button(self) -> QToolButton:
        """④ Snapshot split button: a `QToolButton` in MenuButtonPopup mode.  The
        MAIN half saves a WYSIWYG TIFF of the CURRENT main-canvas view (exactly as
        shown — whatever ndisplay / channels / boxes) via
        ``viewer.screenshot(canvas_only=True, scale=N)``; the ▾ half opens a small
        popover for the resolution scale factor.  The scale bar is the LIVE
        "Scale bar" control's — a WYSIWYG shot captures it as shown.  The popover
        widget IS the session-level persistence — the controller reads the scale
        factor at click time."""
        btn = QToolButton()
        btn.setText("Snapshot")
        btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn.setPopupMode(QToolButton.MenuButtonPopup)
        btn.setStyleSheet(_SNAPSHOT_BTN_STYLE)
        apply_button_icon(btn, "mdi6.monitor-screenshot")
        btn.setToolTip(
            "Save a TIFF of the current main view exactly as shown (scale bar "
            "included if it's on).  Use ▾ for the export resolution.")

        menu = QMenu(btn)
        menu.setStyleSheet(_MENU_STYLE)
        pop = QWidget()
        play = QVBoxLayout(pop)
        play.setContentsMargins(8, 8, 8, 8)
        play.setSpacing(6)

        # The scale bar (on/off + colour + corner + styling) is the LIVE
        # "Scale bar" control's now — a WYSIWYG snapshot captures it verbatim, so
        # this popover carries only the export resolution.
        hint = QLabel("Scale bar? use the Scale bar button — it's captured as shown.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a9bae; font-size: 9px;")
        play.addWidget(hint)

        scale_row = QHBoxLayout()
        scale_row.setSpacing(6)
        scale_row.addWidget(QLabel("scale factor"))
        self.snapshot_scale_spin = QSpinBox()
        self.snapshot_scale_spin.setRange(1, 8)
        self.snapshot_scale_spin.setValue(2)
        self.snapshot_scale_spin.setSuffix("×")
        self.snapshot_scale_spin.setToolTip(
            "Render the figure at this multiple of the on-screen canvas "
            "resolution (clamped so the result fits the GPU texture limit).")
        scale_row.addWidget(self.snapshot_scale_spin, 1)
        play.addLayout(scale_row)

        action = QWidgetAction(menu)
        action.setDefaultWidget(pop)
        menu.addAction(action)
        btn.setMenu(menu)
        self.snapshot_button = btn
        return btn

    def _build_scalebar_button(self) -> QToolButton:
        """④ header LIVE scale-bar control: a checkable `QToolButton` in
        MenuButtonPopup mode.  The MAIN half toggles a live napari scale bar in
        the main viewer (labelled in **µm** when a voxel size is set, pixels
        otherwise — the label rides `layer.units`, set by the physical display);
        the ▾ half opens a styling popover (length auto/fixed, font size, colour,
        box, ticks, corner).  The controller reads these widgets in
        ``_apply_live_scale_bar`` and re-applies on any change."""
        btn = QToolButton()
        btn.setText("Scale bar")
        btn.setCheckable(True)
        btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn.setPopupMode(QToolButton.MenuButtonPopup)
        btn.setStyleSheet(_SNAPSHOT_BTN_STYLE)
        apply_button_icon(btn, "mdi6.ruler")
        btn.setToolTip(
            "Show a live scale bar in the main viewer.  Reads µm from the voxel "
            "size (pixels when none is set).  Use ▾ to style it.")

        menu = QMenu(btn)
        menu.setStyleSheet(_MENU_STYLE)
        pop = QWidget()
        play = QVBoxLayout(pop)
        play.setContentsMargins(8, 8, 8, 8)
        play.setSpacing(6)

        self.scalebar_auto_length_check = QCheckBox("Auto length")
        self.scalebar_auto_length_check.setChecked(True)
        self.scalebar_auto_length_check.setToolTip(
            "Let napari pick a round bar length.  Uncheck to fix it below.")
        play.addWidget(self.scalebar_auto_length_check)

        len_row = QHBoxLayout()
        len_row.setSpacing(6)
        len_row.addWidget(QLabel("length"))
        self.scalebar_length_spin = QDoubleSpinBox()
        self.scalebar_length_spin.setRange(0.01, 100000.0)
        self.scalebar_length_spin.setDecimals(2)
        self.scalebar_length_spin.setValue(50.0)
        self.scalebar_length_spin.setSuffix(" µm")
        self.scalebar_length_spin.setEnabled(False)   # follows Auto
        self.scalebar_length_spin.setToolTip(
            "Fixed bar length in display units (µm when a voxel size is set).")
        len_row.addWidget(self.scalebar_length_spin, 1)
        play.addLayout(len_row)

        font_row = QHBoxLayout()
        font_row.setSpacing(6)
        font_row.addWidget(QLabel("font size"))
        self.scalebar_font_spin = QSpinBox()
        self.scalebar_font_spin.setRange(4, 40)
        self.scalebar_font_spin.setValue(10)
        font_row.addWidget(self.scalebar_font_spin, 1)
        play.addLayout(font_row)

        unit_row = QHBoxLayout()
        unit_row.setSpacing(6)
        unit_row.addWidget(QLabel("units"))
        self.scalebar_unit_combo = QComboBox()
        # "auto" follows the layer's own units (µm when a voxel size is set,
        # pixels otherwise); the others FORCE the label — napari converts the
        # length into that unit when the layer carries a physical spacing.
        self.scalebar_unit_combo.addItems(["auto", "µm", "nm", "mm", "pixel"])
        self.scalebar_unit_combo.setToolTip(
            "Unit shown on the bar.  'auto' uses the source's voxel units (µm "
            "once a voxel size is set); the others force the label.")
        unit_row.addWidget(self.scalebar_unit_combo, 1)
        play.addLayout(unit_row)

        col_row = QHBoxLayout()
        col_row.setSpacing(6)
        col_row.addWidget(QLabel("colour"))
        self.scalebar_color_combo = QComboBox()
        self.scalebar_color_combo.addItems(["light", "dark"])
        col_row.addWidget(self.scalebar_color_combo, 1)
        play.addLayout(col_row)

        self.scalebar_box_check = QCheckBox("Box background")
        self.scalebar_box_check.setChecked(False)
        play.addWidget(self.scalebar_box_check)

        self.scalebar_ticks_check = QCheckBox("Ticks")
        self.scalebar_ticks_check.setChecked(True)
        play.addWidget(self.scalebar_ticks_check)

        pos_row = QHBoxLayout()
        pos_row.setSpacing(6)
        pos_row.addWidget(QLabel("corner"))
        self.scalebar_position_combo = QComboBox()
        self.scalebar_position_combo.addItems(
            ["bottom right", "bottom left", "top right", "top left"])
        pos_row.addWidget(self.scalebar_position_combo, 1)
        play.addLayout(pos_row)

        action = QWidgetAction(menu)
        action.setDefaultWidget(pop)
        menu.addAction(action)
        btn.setMenu(menu)
        # All scale-bar signal wiring (the Auto→grey-out + the live re-apply)
        # lives in the controller __init__; the spin starts disabled (Auto is on).
        self.scalebar_toggle_button = btn
        return btn

    def _log(self, prefix: str, msg: str) -> None:
        """Append a `[PREFIX] message` line to ⑤ Diagnostics, mirrored to stdout.

        Best-effort: safe before the pane exists (during construction) and after
        Qt deletes it, so no caller needs a guard.  The stdout mirror is
        deliberate — the repo-wide convention is `[PREFIX-N]` console
        diagnostics the user can read while testing.
        """
        # `[HH:MM:SS] message` — Step 3's exact format
        # (`fusion_reconstruction/controller.py:18621-18623`), so the three logs
        # read the same when a user pastes one into a bug report.
        from datetime import datetime
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [{prefix}] {msg}"
        # Encode-safe mirror: a non-ASCII char (e.g. the ⟨processed⟩ marker's
        # angle brackets) on a legacy-codepage stdout (Windows cp1252) raises
        # UnicodeEncodeError from `print`.  A DIAGNOSTIC must never crash its
        # caller — this bug aborted the streaming processed-store render mid-way
        # (processed layers added but `_processed_available` never set + no clean
        # swap → the "4 layers, dark Processed button" state).  Fall back to an
        # ASCII-safe rendering, then swallow.
        try:
            print(line)
        except Exception:
            try:
                print(line.encode("ascii", "replace").decode("ascii"))
            except Exception:
                pass
        pane = getattr(self, "log_output", None)
        if pane is None:
            return
        try:
            pane.append(line)
        except RuntimeError:
            pass  # widget already deleted by Qt

    def _init_ui(self):
        """Initialize staged Project Setup UI and wire controller callbacks."""
        # Must exist before any panel is built — `_make_subsection_header`
        # appends to it during construction.
        self._help_guides: list[str] = []
        # True once the user clicks a section header; a deliberate choice then
        # wins over progressive default-open for the rest of the visit.
        self._section_chosen_by_user = False
        # Reentrancy guards, one per concern, never merged.
        self._source_sync_active = False    # ① list ⇄ ② combo selection
        self._source_cards_active = False   # ① card re-render
        self._channel_sel_active = False    # ③ selection normalisation
        # ③ inclusion source of truth: channel display-name → included in the
        # extract.  Real per-row checkboxes read+write this dict; the default
        # for an unseen name is True, so a fresh load extracts every channel.
        self._channel_included: dict[str, bool] = {}
        # True only after a GENUINE user voxel edit (valueChanged), cleared by a
        # restore / reset — so the voxel `editingFinished` save
        # fires only on a real edit, never on a focus-out of an unchanged field.
        self._voxel_dirty = False

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # The flow/header strip is DELIBERATELY UNCHANGED (user decision
        # 2026-07-26).  The design mockup renames its stages and adds a counter strip
        # to this band; both are deferred.  The consequence is that the summary
        # dashboard lives in ① Session instead of the header — recorded in
        # `_create_session_panel` so the deviation is not silent.
        main_layout.addWidget(self._create_flow_overview_group())

        # Five sections: ①–④ are the flow stages, ⑤ Diagnostics is off-flow —
        # the shape Steps 2 and 3 already have.  Titles carry `&&` because
        # `Accordion.addItem` puts them straight into a QPushButton, where a
        # lone `&` is swallowed as a mnemonic.
        accordion = Accordion(accent=_STEP1_ACCENT)
        accordion.addItem(self._create_session_panel(), "Session",
                          icon_name="mdi6.folder-outline")
        accordion.addItem(self._create_load_adjust_panel(),
                          "Load && Adjust",
                          icon_name="mdi6.tune")
        accordion.addItem(self._create_draw_extract_panel(),
                          "Draw && Extract",
                          icon_name="mdi6.crop")
        accordion.addItem(self._create_review_library_panel(),
                          "Review && Library",
                          icon_name="mdi6.file-export-outline")
        accordion.addItem(self._create_diagnostics_panel(), "Diagnostics",
                          icon_name="mdi6.console")
        accordion.setCurrentIndex(0)
        self.accordion = accordion
        main_layout.addWidget(accordion, 1)

        self.setLayout(main_layout)
        # MUST stay after every section is built and before the controller:
        # `_apply_theme` dereferences widget attrs unguarded, and the
        # controller's __init__ connects to widgets.
        self._apply_theme()

        self.controller = DataPreparationController(widget=self, viewer=self.viewer)
        self._connect_signals()

    def open_review_library_section(self) -> None:
        """Expand the ④ Review & Library accordion section — the ③→④ bridge a
        successful Extract calls so the new patches are visible.  Index 3 in the
        `addItem` order (Session · Load && Adjust · Draw && Extract ·
        Review && Library · Diagnostics); best-effort, never raises."""
        try:
            self.accordion.setCurrentIndex(3)
        except Exception:
            pass

    def _apply_theme(self) -> None:
        apply_inner_group_style(self)
        # Primary action buttons (teal accent for Step 1).
        for btn in (self.extract_roi, self.save_reviewed_roi,
                    self.export_all_ready_roi, self.load_source_button,
                    self.apply_preprocess_button):
            btn.setStyleSheet(PRIMARY_BTN_STYLE_TEAL)
        # Secondary buttons.
        for btn in (self.browse_button, self.free_bbox_button,
                    self.fixed_bbox_button, self.clear_shapes_button,
                    self.auto_select_zrange_button, self.set_select_zrange_button,
                    self.source_browse_button,
                    self.unload_source_button, self.locate_source_button,
                    self.prepare_largedata_button,
                    self.largedata_settings_button,
                    self.test_region_button, self.voxel_reset_button,
                    self.add_source_button,
                    self.edit_bbox_button, self.undo_box_button,
                    self.redo_box_button,
                    self.box_add_button,
                    self.box_delete_button,
                    self.place_preview_button, self.recenter_preview_button):
            btn.setStyleSheet(SECONDARY_BTN_STYLE)
        # ③'s S/M/L size presets are a LIST — theme them to match the other
        # secondary box-tool buttons (they otherwise shipped unstyled).  The
        # restored 128³/256³/512³ absolute presets + custom Set button ride the
        # same style.
        for _b in (list(self.box_preset_buttons)
                   + [self.box_custom_apply_button]):
            _b.setStyleSheet(SECONDARY_BTN_STYLE)
        # S/M/L are single-glyph buttons pinned narrow; SECONDARY's 12px side
        # padding leaves ~4px for the letter and CLIPS it to a garbage glyph
        # (2026-07-27 user report).  Keep SECONDARY's look, override only the
        # padding so the letter fits + centres.
        for _b in self.box_size_preset_buttons:
            _b.setStyleSheet(
                SECONDARY_BTN_STYLE + "QPushButton { padding: 4px 2px; }")
        # The ② preview-region S/M/L presets are the same single-glyph shape —
        # give them the same narrow padding so the letter fits + centres.
        for _b in self.preview_size_preset_buttons:
            _b.setStyleSheet(
                SECONDARY_BTN_STYLE + "QPushButton { padding: 4px 2px; }")
        # `transpose_button` is CHECKABLE and was getting SECONDARY_BTN_STYLE,
        # which has no `:checked` rule — so it had **no visible ON state**.
        # `box3d_mode_button` is checkable too (box-mode ON/OFF).
        self.transpose_button.setStyleSheet(toggle_btn_style(_STEP1_ACCENT))
        self.box3d_mode_button.setStyleSheet(toggle_btn_style(_STEP1_ACCENT))
        # Key action icons.
        apply_button_icon(self.extract_roi, "mdi6.package-variant")
        apply_button_icon(self.save_reviewed_roi, "mdi6.tray-arrow-up")
        apply_button_icon(self.export_all_ready_roi, "mdi6.export-variant")
        # Data browser icons.
        apply_button_icon(self.source_browse_button, "mdi6.folder-search-outline")
        apply_button_icon(self.load_source_button, "mdi6.eye-plus-outline")
        apply_button_icon(self.unload_source_button, "mdi6.eye-minus-outline")
        apply_button_icon(self.prepare_largedata_button, "mdi6.layers-triple-outline")
        apply_button_icon(self.largedata_settings_button, "mdi6.tune-variant")
        apply_button_icon(self.transpose_button, "mdi6.swap-horizontal")
        apply_button_icon(self.clear_shapes_button, "mdi6.delete-sweep-outline")
        self.setStyleSheet(self.styleSheet() + INPUT_STYLE + LIST_STYLE + SCROLLBAR_STYLE)
        # Both bars carry the step accent.  `progress_bar_style("#4a9ede")` was
        # an off-palette near-miss on `_STEP1_ACCENT` and is retired.
        for bar in (self.progress_bar, self.source_progress_bar,
                    self.export_progress_bar):
            bar.setStyleSheet(PROGRESS_BAR_STYLE_TEAL)
        # Step 1's lists take the teal variant, not the amber default that the
        # shared LIST_STYLE carries.  (The ② recipe is no longer a
        # QListWidget — it is the per-channel column scroller, styled inline.)
        for lst in (self.source_file_list, self.extracted_roi_list,
                    self.image_layer_combo):
            lst.setStyleSheet(LIST_STYLE_TEAL)

    def _create_flow_overview_group(self):
        hint = QWidget()
        hint.setStyleSheet(FLOW_HINT_STYLE)
        layout = QVBoxLayout(hint)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        flow = QLabel(
            "Workspace \u2192 Load Data \u2192 ROI Capture \u2192 Review & Export"
        )
        flow.setWordWrap(True)
        flow.setStyleSheet(FLOW_TITLE_STYLE)

        sub = QLabel(
            "Set workspace, load source TIFFs, draw ROI bounding boxes, "
            "extract patches, trim Z-range, then export."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(FLOW_SUBTITLE_STYLE)

        layout.addWidget(flow)
        layout.addWidget(sub)
        return hint

    # ------------------------------------------------------------------
    # ① Session
    # ------------------------------------------------------------------

    def _create_session_panel(self):
        """① Session — workspace + summary.

        Scope: the family shell only.  Source cards, alias and the
        card's size/dtype/axes line land next.

        **Deliberate deviation from the design mockup, recorded here rather than left
        silent:** the mockup puts the `new · staged · captured` counter strip in
        the header band under the flow strip.  The header is frozen by user
        decision (2026-07-26), so the strip lives here instead.  The cost is
        real and worth naming — the accordion is exclusive, so these counters
        are invisible while you work in ③.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Workspace directory
        layout.addLayout(self._make_subsection_header(
            "Workspace", "step1_session.md"))

        helper = QLabel(
            "Choose a workspace folder where exported ROIs and metadata will be written."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(helper)

        self.output_file_label = QLabel("")  # kept for backward compat
        self.output_file_label.setVisible(False)
        self.output_file_path = QLineEdit()
        self.output_file_path.setPlaceholderText("Select workspace directory...")
        self.output_file_path.textChanged.connect(self._update_workspace_status)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setToolTip("Select or create a workspace folder for this project")
        self.browse_button.clicked.connect(self.browse_output_file)

        path_row = QHBoxLayout()
        path_row.addWidget(self.output_file_path)
        path_row.addWidget(self.browse_button)

        layout.addWidget(self.output_file_label)
        layout.addLayout(path_row)

        # Workspace status
        self.workspace_status_label = QLabel("")
        self.workspace_status_label.setWordWrap(True)
        self.workspace_status_label.setStyleSheet(DIM_LABEL_STYLE)
        layout.addWidget(self.workspace_status_label)

        layout.addWidget(self._separator())

        # ── Sources ── header carries [+ Add], THEN the description — matches
        # the ① mockup (the action sits in the Sources subsection header) and
        # fixes the prior order, where the title rendered below its own desc.
        src_header = self._make_subsection_header(
            "Sources", "step1_session_sources.md")
        self.add_source_button = QPushButton("+ Add")
        self.add_source_button.setToolTip(
            "Browse a folder of TIFF stacks and register them with this project")
        self.add_source_button.setMaximumWidth(64)
        src_header.insertWidget(src_header.count() - 1, self.add_source_button)
        layout.addLayout(src_header)

        _sdesc = QLabel(
            "Image stacks registered in this workspace. Adding a source does "
            "not extract anything — sources persist with 0 patches."
        )
        _sdesc.setWordWrap(True)
        _sdesc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(_sdesc)

        self.source_folder_path = QLineEdit()
        self.source_folder_path.setPlaceholderText("Source data folder...")
        self.source_folder_path.setReadOnly(True)
        # The mockup's ① has no folder field; `+ Add` is the affordance.  Kept
        # for the controller (`:162, :166, :1069`), hidden like its Browse twin.
        self.source_folder_path.setVisible(False)
        # `+ Add` above IS the browse affordance the mockup draws.  This older
        # button is the same slot; kept alive because the controller binds the name, which must
        # survive and `_apply_theme` styles it unguarded, but hidden so ① does
        # not show two identical actions.
        self.source_browse_button = QPushButton("Browse")
        self.source_browse_button.setToolTip(
            "Select folder containing TIFF stacks")
        self.source_browse_button.setVisible(False)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(6)
        folder_row.addWidget(self.source_folder_path, 1)
        folder_row.addWidget(self.source_browse_button, 0)
        layout.addLayout(folder_row)

        # Source CARDS.  `source_file_list` stays a QListWidget — frozen type,
        # and `browse_source_folder` populates it — but each row is rendered as
        # a card via `setItemWidget`, the `_import_pairs_dialog` pattern
        # Item text is left untouched: `on_source_file_selected`
        # and `load_source` read `item.data(Qt.UserRole)`, and
        # `_refresh_source_file_list` writes the text.
        self.source_file_list = QListWidget()
        self.source_file_list.setMinimumHeight(120)
        self.source_file_list.setStyleSheet(LIST_STYLE_TEAL)
        # Brick 4a a5: the same "Find the original file..." action, on the row
        # itself.  (2) rows are never greyed (only (4) rows are), so this is
        # reachable whether or not the source is currently loaded.
        self.source_file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.source_file_list.customContextMenuRequested.connect(
            self._on_source_row_context_menu)
        layout.addWidget(self.source_file_list, 1)

        self.source_empty_label = QLabel(
            "No sources yet — Browse a folder of TIFFs.")
        self.source_empty_label.setWordWrap(True)
        self.source_empty_label.setStyleSheet(DIM_LABEL_STYLE)
        layout.addWidget(self.source_empty_label)

        # Alias — drives the patch prefix.
        alias_row = QHBoxLayout()
        alias_row.setSpacing(6)
        alias_row.addWidget(QLabel("Name:"))
        self.display_name_edit = QLineEdit()
        self.display_name_edit.setPlaceholderText("Short name for this source…")
        self.display_name_edit.setToolTip(
            "Becomes the prefix on every patch extracted from this source")
        alias_row.addWidget(self.display_name_edit, 1)
        layout.addLayout(alias_row)

        self.alias_hint_label = QLabel("")
        self.alias_hint_label.setWordWrap(True)
        self.alias_hint_label.setStyleSheet(DIM_LABEL_STYLE)
        layout.addWidget(self.alias_hint_label)

        # Channels: READ-ONLY here with a pointer to ③, which owns identity.
        # ① cannot own them — the channel count is only knowable AFTER the load
        # that happens in ②, so a first visit would have nothing to populate.
        self.session_channels_label = QLabel(
            "Channels: — · named in Draw & Extract")
        self.session_channels_label.setWordWrap(True)
        self.session_channels_label.setStyleSheet(DIM_LABEL_STYLE)
        layout.addWidget(self.session_channels_label)

        layout.addWidget(self._separator())

        # ── Session Summary ──
        layout.addLayout(self._make_subsection_header(
            "Session Summary", "step1_session_summary.md"))

        self.summary_new_label = self._make_counter_chip("new", _STEP1_ACCENT)
        self.summary_staged_label = self._make_counter_chip("staged", "#80deea")
        self.summary_captured_label = self._make_counter_chip(
            "captured", "#8a9bae")

        counter_row = QHBoxLayout()
        counter_row.setSpacing(6)
        for chip in (self.summary_new_label, self.summary_staged_label,
                     self.summary_captured_label):
            counter_row.addWidget(chip)
        counter_row.addStretch(1)
        layout.addLayout(counter_row)

        layout.addStretch(1)
        return panel

    def _make_source_card(self, item) -> QWidget:
        """One ① source card: alias · original filename · detail · size.

        Built fresh rather than reusing `make_pair_card`, which renders a
        placeholder mask square this single-image case has no use for.

        Reads only what the controller already puts on the item — the stem in
        `UserRole + 1` and the display text `_refresh_source_file_list` composes
        (filename + shape + patch-count badge, keyed on the immutable
        `source_id`).  RGB-composite thumbnails and the decompressed
        size/dtype/axes line need the extended `scan_tiff_folder` header pass, so they are omitted rather than faked.
        """
        text = item.text() or ""
        stem = item.data(Qt.UserRole + 1) or (text.split(" ")[0] if text else "")
        path = item.data(Qt.UserRole) or ""

        card = QWidget()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(1)

        alias = QLabel(str(stem) or "source")
        alias.setWordWrap(True)
        alias.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #dddddd;")
        lay.addWidget(alias)

        if path:
            origin = QLabel(f"↳ {os.path.basename(path)}")
            origin.setWordWrap(True)
            origin.setStyleSheet(DIM_LABEL_STYLE)
            origin.setToolTip(path)
            lay.addWidget(origin)
            if not os.path.isfile(path):
                # Moved/deleted source — the path is the immutable key, so say
                # so rather than failing silently later.
                gone = QLabel("⚠ source moved — relocate")
                gone.setStyleSheet("font-size: 10px; color: #d4a843;")
                lay.addWidget(gone)

        # `_refresh_source_file_list` writes a two-line string with a LEADING
        # space, so splitting on " " gave "" and the "detail" line was the whole
        # text — the alias rendered twice on one card.
        detail = text.split("\n", 1)[1].strip() if "\n" in text else ""
        if detail:
            info = QLabel(detail)
            info.setWordWrap(True)
            info.setStyleSheet(DIM_LABEL_STYLE)
            lay.addWidget(info)

        # decompressed size + dtype live HERE, on the card, in
        # warning colour when the stack will not fit.  This is where the load
        # plan line and `override ▾` went when they were deleted — without it
        # their removal is a net loss of information, not a relocation.
        route = self._route_for_source(path)
        if route is not None:
            from .logic import format_bytes
            bits = [format_bytes(route["nbytes"])]
            if route.get("dtype"):
                bits.append(route["dtype"])
            if route.get("axes"):
                bits.append(str(route["axes"]))
            note = "" if route["eager"] else (
                "  ⚠ larger than free memory — memory-mappable"
                if route["view"] == "MEMMAP"
                else "  ⚠ larger than free memory")
            size = QLabel(" · ".join(bits) + note)
            size.setWordWrap(True)
            size.setStyleSheet(
                DIM_LABEL_STYLE if route["eager"]
                else "font-size: 10px; color: #d4a843;")
            lay.addWidget(size)

        return card

    def _route_for_source(self, path):
        """Header-only load route for a source, cached per path.

        Cached because the card must not re-read the
        header on every selection.
        """
        if not path:
            return None
        cache = getattr(self, "_route_cache", None)
        if cache is None:
            cache = self._route_cache = {}
        if path in cache:
            return cache[path]
        scanned = getattr(self.controller, "_scanned_files", None) or []
        info = next((f for f in scanned if f.get("path") == path), None)
        if info is None or not info.get("nbytes"):
            cache[path] = None
            return None
        try:
            import psutil
            available = psutil.virtual_memory().available
        except Exception:
            available = None
        from .logic import plan_load_route
        shape = info.get("shape") or ()
        n_ch = min(shape[0], shape[1]) if len(shape) >= 4 else 1
        route = plan_load_route(info["nbytes"], available, n_ch,
                                compression=info.get("compression"))
        route["dtype"] = info.get("dtype")
        route["axes"] = info.get("axes")
        cache[path] = route
        return route

    def invalidate_route_cache(self) -> None:
        """Drop cached header routes.

        Called when the source folder is re-browsed: the cache is keyed by path,
        so a file replaced in place -- or a different folder holding the same
        filename -- would otherwise keep the stale numbers.
        """
        self._route_cache = {}

    def confirm_non_eager_load(self, path, route) -> bool:
        """Blocking decision dialog when the route is not eager.

        Deliberately NOT a "load anyway" prompt: `override` was deleted outright
        because it put "ignore the memory guard" one click from the allocation
        the guard exists to prevent.  The choices are memory-mapped loading
        (uncompressed files only) and Cancel; the OME-Zarr re-save button is
        built but hidden until the write path lands (release hide list
        2026-10-12).

        Returns True only if the caller may proceed — today, never.
        """
        from qtpy.QtWidgets import QDialog, QDialogButtonBox
        from .logic import format_bytes

        dlg = QDialog(self)
        dlg.setWindowTitle("This stack is larger than memory")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        head = QLabel(os.path.basename(path))
        head.setStyleSheet(SECTION_HEADER_STYLE)
        lay.addWidget(head)

        body = QLabel(
            f"Decompressed it is {format_bytes(route['nbytes'])}, and "
            f"processing it would peak near {format_bytes(route['proc_peak'])} "
            "— more than this machine has free.\n\n"
            "Loading it whole would freeze napari."
        )
        body.setWordWrap(True)
        body.setStyleSheet(FLOW_SUBTITLE_STYLE)
        lay.addWidget(body)

        detail = QLabel(
            "It is stored uncompressed, so it can be read memory-mapped: pages "
            "stay on disk and only what you look at is paged in. No conversion "
            "needed."
            if route.get("view") == "MEMMAP" else
            "It is compressed, so it cannot be memory-mapped, and Step 1 "
            "cannot load it whole on this machine."
        )
        detail.setWordWrap(True)
        detail.setStyleSheet(FLOW_SUBTITLE_STYLE)
        lay.addWidget(detail)

        mappable = route.get("view") == "MEMMAP"
        self._non_eager_choice = None

        if mappable:
            # A real capability, not a stub: `tifffile.memmap` needs no new
            # backend.  Opt-in rather than automatic, because a memory-mapped
            # array behaves differently downstream and the user should know
            # which one they got.
            mmap_btn = QPushButton("Load memory-mapped")
            mmap_btn.setToolTip(
                "Read the file in place, without copying it into RAM")
            mmap_btn.setStyleSheet(PRIMARY_BTN_STYLE_TEAL)

            def _choose_mmap():
                self._non_eager_choice = "memmap"
                dlg.accept()

            mmap_btn.clicked.connect(_choose_mmap)
            lay.addWidget(mmap_btn)
        else:
            convert = QPushButton("Re-save as OME-Zarr…")
            convert.setEnabled(False)
            # release hide list 2026-10-12: the chunked write path is not
            # built; hidden (kept, reversible) rather than a dead button.
            convert.setVisible(False)
            convert.setToolTip(
                "The chunked write path is not built yet")
            convert.setStyleSheet(SECONDARY_BTN_STYLE)
            lay.addWidget(convert)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Cancel).setText("Cancel load")
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        dlg.setStyleSheet(self.styleSheet())
        dlg.exec_()
        return self._non_eager_choice == "memmap"

    def _rebuild_source_cards(self, *_args) -> None:
        """Render every source row as a card.

        Re-runs on `dataChanged` as well as insert/remove/reset, because
        `browse_source_folder:162-168` adds items with **empty text** and
        `_refresh_source_file_list` fills it in a second pass — without the
        `dataChanged` hook the cards would show blank details forever.
        """
        if getattr(self, "_source_cards_active", False):
            return
        self._source_cards_active = True
        try:
            lst = self.source_file_list
            for i in range(lst.count()):
                item = lst.item(i)
                if (lst.itemWidget(item) is not None
                        and item.data(Qt.UserRole + 8) == item.text()):
                    continue
                card = self._make_source_card(item)
                item.setSizeHint(card.sizeHint())
                lst.setItemWidget(item, card)
                item.setData(Qt.UserRole + 8, item.text())
            self.source_empty_label.setVisible(lst.count() == 0)
        finally:
            self._source_cards_active = False

    def _update_alias_hint(self, *_args) -> None:
        """State the prefix the alias will actually produce.

        Mirrors `controller.extract_patches:643-650`, including its empty-name
        `else` branch, so the user sees the consequence before extracting rather
        than after.
        """
        alias = self.display_name_edit.text().strip()
        prefix = f"{alias}_patch" if alias else "patch"
        # Numbered from where extraction will actually start, so the hint agrees
        # with ③'s row labels and with `find_next_patch_index`.
        start = self._next_patch_index(prefix)
        self.alias_hint_label.setText(
            f"Patches will be named {prefix}_{start}, {prefix}_{start + 1}, …")

    @staticmethod
    def _make_counter_chip(caption: str, color: str) -> QLabel:
        """One box-status counter for the summary strip.

        Three statuses only — `new · staged · captured`, the vocabulary
        exactly.  There is deliberately **no fourth `ready` counter**: the old
        `ready` meant "has a `selected_z_range`", which now belongs to ④'s
        detail pane.
        """
        chip = QLabel(f"0 {caption}")
        chip.setAlignment(Qt.AlignCenter)
        chip.setMinimumWidth(78)
        chip.setStyleSheet(
            f"QLabel {{"
            f"  background: rgba(255, 255, 255, 0.04);"
            f"  border: 1px solid {color}55;"
            f"  border-left: 3px solid {color};"
            f"  border-radius: 3px;"
            f"  padding: 4px 8px;"
            f"  font-size: 10px; color: #cccccc;"
            f"}}"
        )
        chip._epicycle_caption = caption
        return chip

    def refresh_summary_dashboard(self, counts: dict) -> None:
        """Update the ① counter strip.

        Takes `{"new": int, "staged": int, "captured": int}`; missing keys read
        as 0, so a partial dict, `{}` or `None` is safe.
        """
        counts = counts or {}
        for chip in (self.summary_new_label, self.summary_staged_label,
                     self.summary_captured_label):
            caption = getattr(chip, "_epicycle_caption", "")
            chip.setText(f"{int(counts.get(caption, 0))} {caption}")

    # ------------------------------------------------------------------
    # ② Load & Adjust (two-column splitter)
    # ------------------------------------------------------------------

    def _create_load_adjust_panel(self):
        """② Load & Adjust — data browser (left) + preprocess (right).

        Left = Source + Voxel size + Extract-from; right = the preprocessing
        recipe; ONE shared labelled bar spans the bottom.

        **The split is 4/6, not 50/50** — a 50/50 split of the
        *whole* section is wrong; the chain needs the wider half", and recipe
        rows carry index + name + params + scope + ⚙ + three buttons.
        """
        panel = QWidget()
        root = QVBoxLayout(panel)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(SPLITTER_STYLE)
        splitter.addWidget(self._create_data_browser_column())
        splitter.addWidget(self._create_preprocessing_column())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([400, 600])
        root.addWidget(splitter, 1)

        # ── ONE SHARED BAR, full width at the section bottom.
        # It serves both Load and Apply, so it MUST name what is running — a
        # nameless marquee is exactly the "which one is moving?" problem the
        # shared-bar exception was justified against.  Driven only by
        # `_busy_show`/`_busy_hide`.
        root.addWidget(self._separator())
        # The Load button itself doubles as Cancel while a read is in flight
        # (`controller.on_load_or_cancel_clicked` dispatches), so the bar stays
        # full-width with no separate Cancel button beside it.
        self.source_progress_bar = QProgressBar()
        self.source_progress_bar.setRange(0, 1)
        self.source_progress_bar.setValue(0)
        self.source_progress_bar.setTextVisible(True)
        self.source_progress_bar.setFormat("")
        root.addWidget(self.source_progress_bar)

        # Distinct from `source_status_label`, which stays in the left column as
        # the load RESULT line the controller already writes.
        self.busy_status_label = QLabel("— idle")
        self.busy_status_label.setWordWrap(True)
        self.busy_status_label.setStyleSheet(DIM_LABEL_STYLE)
        root.addWidget(self.busy_status_label)
        return panel

    # ------------------------------------------------------------------
    # ② shared busy bar — the `_busy_show(label)` contract
    # ------------------------------------------------------------------

    def _busy_show(self, label: str) -> None:
        """Put ②'s shared bar into scan mode and NAME the running operation.

        Mirrors Step 3's `_busy_show` (`fusion_reconstruction/controller.py:11518`),
        including its deliberate choice of scan mode over a fake percentage
        where there is no real denominator.  The label is mandatory:
        naming the op is the condition on which ② is allowed a shared bar at all.
        """
        self.source_progress_bar.setRange(0, 0)
        self.source_progress_bar.setFormat(label)
        self.source_progress_bar.setVisible(True)
        self.busy_status_label.setText(label)

    def _busy_progress(self, label: str, done: int, total: int) -> None:
        """Real percentage where a denominator exists (slices, chunks)."""
        if total <= 0:
            self._busy_show(label)
            return
        self.source_progress_bar.setRange(0, total)
        self.source_progress_bar.setValue(done)
        self.source_progress_bar.setFormat(f"{label}  %p%")
        self.busy_status_label.setText(label)

    def _busy_hide(self, idle: str = "— idle") -> None:
        """Return the bar to rest.  Always reachable from a terminal funnel so
        a failed op cannot leave the bar spinning."""
        self.source_progress_bar.setRange(0, 1)
        self.source_progress_bar.setValue(0)
        self.source_progress_bar.setFormat("")
        self.busy_status_label.setText(idle)

    def _refresh_prepare_largedata_enabled(self, *_args) -> None:
        """Enable "Prepare for large-data viewing" only while a source file is
        selected.  A no-op while the ② surface is busy (the busy lock — and the
        Prepare-as-Cancel flip — own the button state then, so this must not
        fight them)."""
        if getattr(getattr(self, "controller", None), "_step1_busy", False):
            return
        try:
            has_selection = self.source_file_list.currentItem() is not None
            # Brick 4a a4: Prepare re-reads the ORIGINAL file block-wise, so it
            # is off while Step 1 is serving the prepared copy of a source whose
            # original it can't find.
            if self._raw_missing():
                self.prepare_largedata_button.setEnabled(False)
                self.prepare_largedata_button.setToolTip(
                    "Point Step 1 back at the original image file first "
                    "(“Find the original file…”)")
                return
            self.prepare_largedata_button.setEnabled(has_selection)
            self.prepare_largedata_button.setToolTip(
                self._PREPARE_LARGEDATA_TOOLTIP)
        except Exception:
            pass

    def _call_locate_source(self, source_path=None) -> None:
        """Route "Find the original file..." to the controller, guarded.

        The row context menu already checked the handler exists; the button used
        to call it straight off `self.controller` and would raise on any
        controller that does not carry it (a stub, a partially-built widget in a
        test).  Same guard, one place (Brick 4c item 7)."""
        fn = getattr(getattr(self, "controller", None),
                     "locate_source_for_row", None)
        if callable(fn):
            try:
                fn(source_path=source_path)
            except Exception:      # noqa: BLE001 -- never break the (2) surface
                pass

    def _on_source_row_context_menu(self, pos) -> None:
        """Right-click on a (2) source row -> "Find the original file...".

        Routed through the controller so the decision matrix + the confirm modal
        live in exactly one place."""
        try:
            item = self.source_file_list.itemAt(pos)
            if item is None:
                return
            menu = QMenu(self.source_file_list)
            menu.setStyleSheet(_MENU_STYLE)
            act = menu.addAction("Find the original file\u2026")
            chosen = menu.exec_(self.source_file_list.mapToGlobal(pos))
            if chosen is not act:
                return
            # Pass the ROW's path; the controller owns every registry lookup
            # (ui.py deliberately never imports `_state`).
            self._call_locate_source(item.data(Qt.UserRole))
        except Exception:      # noqa: BLE001 -- a context menu is never critical
            pass

    def _raw_missing(self) -> bool:
        """True while the controller is serving the loaded source from its
        prepared copy because the original file can't be found."""
        return bool(getattr(getattr(self, "controller", None),
                            "_raw_missing", False))

    def refresh_raw_missing_gates(self) -> None:
        """Re-apply the two "needs the original file" button gates (Prepare and
        “Process whole volume”).  Cheap + idempotent — the controller calls
        it whenever the raw-missing state flips."""
        self._refresh_prepare_largedata_enabled()
        try:
            self._apply_process_button_gate(
                getattr(self, "_process_gate_has_channels", False))
        except Exception:
            pass

    def _create_data_browser_column(self):
        """② left column: Source → Voxel size → Extract from.

        The folder picker, source list and alias live in ① — the old
        fragmented Workspace/Source-browser split is merged into one Session panel, and
        Ownership is settled: ① is the *browser*, ② is where Load happens.
        """
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(6, 6, 14, 6)
        layout.setSpacing(6)

        # --- Source ---
        layout.addLayout(self._make_subsection_header(
            "Source", "step1_load_settings.md"))

        _desc = QLabel(
            "Load a registered source. Order is confirmed here — if the image "
            "looks scrambled, change it and load again."
        )
        _desc.setWordWrap(True)
        _desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(_desc)

        # ② OWNS THE PICKER (settled v3.7) — a two-way proxy over
        # `source_file_list`'s current row, which is what
        # `controller.load_source:188` actually reads, so the combo cannot
        # disagree with the button beside it.  Needed because the accordion is
        # exclusive: ①'s list and this Load button are never co-visible.
        source_row = QHBoxLayout()
        source_row.setSpacing(6)
        source_row.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self._SOURCE_COMBO_HINT = "Which source to load. Register sources in Session."
        self.source_combo.setToolTip(self._SOURCE_COMBO_HINT)
        # The box is pinned narrow (below), so a long name is elided in the
        # CLOSED box -- mirror the current item's tooltip onto the box itself so
        # hovering it still shows the full name and path.
        self.source_combo.currentIndexChanged.connect(
            self._sync_source_combo_tooltip)
        # Sizing (the Step-2 `main_widget.py` / Step-3 `curation_3d_widget.py`
        # fix): Qt's default `AdjustToContentsOnFirstShow` locks the combo to the
        # LONGEST item on first paint, and that width then becomes this section's
        # minimum -- it propagates to the splitter and steals width from
        # everything to its right.  Pin a minimum contents length instead: long
        # source names elide in the box and show in full in the popup, which is
        # free to be wider.
        self.source_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.source_combo.setMinimumContentsLength(12)
        self.source_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        source_row.addWidget(self.source_combo, 1)
        layout.addLayout(source_row)

        # Axis order \u2014 the SINGLE visible editor for the field.
        # `transpose_button` stays alive but hidden: its name is frozen and
        # `controller.py:242` reads `.isChecked()`, so the combo drives it
        # rather than sitting beside it as a second editor.
        order_row = QHBoxLayout()
        order_row.setSpacing(6)
        order_row.addWidget(QLabel("Order:"))
        self.axis_order_combo = QComboBox()
        # Two entries, not three: `transpose_button` is a boolean, so a `ZYX`
        # option would resolve identically to `CZYX` and promise a mode the
        # loader does not have (a manifest axes field that lies is worse than none).
        self.axis_order_combo.addItems(["CZYX", "ZCYX"])
        self.axis_order_combo.setToolTip(
            "Axis order of the file. If the loaded image looks scrambled, "
            "change this and load again."
        )
        # Same rationale as `source_combo` above.
        self.axis_order_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.axis_order_combo.setMinimumContentsLength(6)
        order_row.addWidget(self.axis_order_combo, 1)
        # NOT "detected" \u2014 no detection code exists yet, and asserting
        # an unverified axis order as the exact failure to avoid ("the manifest
        # `axes` field is a lie").  The real header pass is
        # detection work still to come.
        self.axis_order_hint = QLabel("not detected \u2014 confirm")
        self.axis_order_hint.setWordWrap(True)   # wrap, never widen the row
        self.axis_order_hint.setStyleSheet(DIM_LABEL_STYLE)
        order_row.addWidget(self.axis_order_hint, 0)
        layout.addLayout(order_row)

        self.transpose_button = QPushButton("Transpose Axes")
        self.transpose_button.setToolTip("Swap axis 0 and 1 (e.g. ZCYX \u2192 CZYX)")
        self.transpose_button.setCheckable(True)
        self.transpose_button.setVisible(False)
        layout.addWidget(self.transpose_button)

        # Actions, after the selectors.
        self.load_source_button = QPushButton("Load")
        self.load_source_button.setToolTip(
            "Load the selected TIFF into the viewer, splitting channels into "
            "separate layers"
        )
        self.unload_source_button = QPushButton("Unload")
        self.unload_source_button.setToolTip(
            "Remove the loaded source layers from the viewer")
        # Brick 4a a5: point this source back at its original image file after
        # it was renamed / moved / left behind by a workspace copy.  Always
        # present (not only in the raw-missing state) so it is reachable when
        # the user simply KNOWS a file moved.
        self.locate_source_button = QPushButton("Find the original file\u2026")
        self.locate_source_button.setToolTip(
            "Point this source back at its original image file \u2014 use it "
            "after the file was renamed, moved, or left behind when the "
            "workspace was copied")
        layout.addWidget(self.load_source_button)
        layout.addWidget(self.unload_source_button)
        layout.addWidget(self.locate_source_button)

        # "Prepare for large-data viewing" — convert this source into a
        # chunked, multiscale image so it loads + pans smoothly without using
        # all the machine's memory, then display it lazily.  Disabled until a
        # source file is picked (enablement follows the list selection below).
        self.prepare_largedata_button = QPushButton(
            "Prepare for large-data viewing")
        self.prepare_largedata_button.setToolTip(
            "Convert this source into a chunked multiscale image so it loads + "
            "pans smoothly without using all your memory. Best for stacks too "
            "large to open whole. Runs in the background; the box tool + "
            "extract then work on the converted image.")
        self.prepare_largedata_button.setEnabled(False)
        layout.addWidget(self.prepare_largedata_button)

        # "Large-data settings…" — choose a resource-adaptive preset (Safe /
        # Balanced / Fast) or hand-set every convert knob, persisted per
        # workspace.  Feeds num_workers/chunk/downsample/min_size into the
        # Prepare convert above.  Always enabled (it's a settings dialog); it
        # seeds from the machine-adaptive default when nothing is saved yet.
        self.largedata_settings_button = QPushButton("Large-data settings…")
        self.largedata_settings_button.setToolTip(
            "Tune how much of your machine the large-data conversion uses "
            "(parallel workers, chunk size, overview levels). Presets scale to "
            "your CPU + memory automatically; Custom lets you set every value. "
            "Saved per workspace.")
        layout.addWidget(self.largedata_settings_button)

        # PROGRESSIVE 3D (experimental): stream a sharper 3D view in as the user
        # zooms (the vendored napari-#9067 seam), instead of the coarse-only 3D
        # render napari shows by default.  Default OFF (opt-in): the shipping
        # default is the proven crash-free NATIVE multiscale path + the manual
        # "3D detail" slider below; checking this opts a power user into the
        # experimental streaming engine.  Kept default-off because the vendored
        # engine monkeypatches vispy's GLIR flush process-wide and its 3D<->2D
        # transitions are GL-timing fragile — so the default experience never
        # carries that risk.  The controller's `progressive_wanted` gate also
        # requires `progressive.progressive_available()`, so a checked box on an
        # install missing the engine still fails CLOSED to the plain path.
        self.progressive_3d_checkbox = QCheckBox("Progressive 3D (experimental)")
        self.progressive_3d_checkbox.setChecked(False)
        self.progressive_3d_checkbox.setToolTip(
            "Experimental / off by default. Streams high-res 3D detail in as you "
            "zoom, instead of a coarse-only 3D view. Leave unchecked to use the "
            "proven plain view + the 3D-detail slider; check to opt in.")
        # NB: availability is NOT probed here.  Probing would eagerly import the
        # ~6k-LOC vendored engine at Step-1 widget construction — on the main
        # thread, for EVERY session, incl. users who never touch large data
        # (Step 1 is the entry screen).  Instead the checkbox is enabled
        # optimistically and the controller's `progressive_wanted()` gate checks
        # `progressive_available()` LAZILY at large-data display time (when the
        # engine is imported anyway), failing CLOSED to the plain path if it's
        # unavailable.  Keeps plugin startup fast.
        layout.addWidget(self.progressive_3d_checkbox)

        # ZOOM DEPTH (progressive-3D only): controls how much of the tissue
        # THICKNESS the sharper 3D region covers when you zoom in.  By default
        # (Off) that region is a thin camera-centred slab of Z, so the lateral
        # view can grow while depth stays shallow.  Higher percentages show more
        # of the depth (Z) but a narrower lateral window (they trade GPU memory
        # from width to depth); Full = the whole depth; Cube = a balanced
        # isotropic box (depth ≈ width).  Hidden unless a prepared multiscale
        # store is shown in 3D with Progressive 3D ON (the controller shows/syncs
        # it via `configure_zoom_depth` / `hide_zoom_depth`); default Off, and a
        # LIVE change (no re-load / re-prepare).  Labels/modes: ZOOM_DEPTH_LABELS.
        _zd_row = QWidget()
        _zd_lay = QHBoxLayout(_zd_row)
        _zd_lay.setContentsMargins(0, 0, 0, 0)
        _zd_lay.setSpacing(6)
        _zd_caption = QLabel("Zoom depth")
        _zd_lay.addWidget(_zd_caption)
        self.zoom_depth_combo = QComboBox()
        self.zoom_depth_combo.addItems(list(ZOOM_DEPTH_LABELS))
        self.zoom_depth_combo.setCurrentIndex(0)             # Off
        self.zoom_depth_combo.setToolTip(
            "When you zoom in, how much of the tissue THICKNESS (depth / Z) the "
            "sharpened region covers. Off = a thin slab. Higher % = see more of "
            "the depth but a narrower lateral window (and more GPU memory). "
            "Full = the whole depth. Cube = a balanced isotropic box "
            "(depth ≈ width).")
        _zd_lay.addWidget(self.zoom_depth_combo, 1)
        _zd_row.setVisible(False)
        self.zoom_depth_row = _zd_row
        layout.addWidget(_zd_row)

        # 3D DETAIL row: pins a sharper pyramid level in the 3D view (napari
        # renders only the COARSEST level in 3D by default -> blurry).  Hidden
        # until a prepared multiscale store is displayed; the controller fills
        # the slider range + readout via `configure_3d_detail`.  Left = sharper
        # (heavier), right = coarser (lighter).  Only affects the 3D view — 2D
        # pan/zoom keeps napari's automatic level picking.
        self.threed_detail_row = QWidget()
        _td = QVBoxLayout(self.threed_detail_row)
        _td.setContentsMargins(0, 2, 0, 2)
        _td.setSpacing(2)
        _td_top = QHBoxLayout()
        _td_top.setSpacing(6)
        _td_caption = QLabel("3D detail")
        _td_caption.setToolTip(
            "Sharpen the 3D view by pinning a finer pyramid level. In 3D, napari "
            "shows the coarsest (blurriest) level by default. Sharper levels use "
            "more GPU memory; the finest level loads the full volume. Only "
            "affects 3D — 2D pan/zoom is unchanged.")
        _td_top.addWidget(_td_caption)
        self.threed_detail_slider = QSlider(Qt.Horizontal)
        self.threed_detail_slider.setMinimum(0)
        self.threed_detail_slider.setMaximum(0)
        self.threed_detail_slider.setToolTip(
            "Left = sharper (more memory) · Right = coarser (lighter)")
        _td_top.addWidget(self.threed_detail_slider, 1)
        _td.addLayout(_td_top)
        self.threed_detail_label = QLabel("")
        self.threed_detail_label.setStyleSheet("font-size: 10px; color: #9aa0a6;")
        self.threed_detail_label.setWordWrap(True)
        _td.addWidget(self.threed_detail_label)
        self.threed_detail_row.setVisible(False)
        layout.addWidget(self.threed_detail_row)

        # THE LOAD RESULT LINE.  This row already
        # exists: `source_status_label` carries four coloured states the
        # controller writes (`:191` amber, `:203` red, `:232` red, `:279`
        # green).  Extend its content; never build a second row and orphan this
        # wiring.  It stays HERE, next to Load \u2014 the shared bar at the panel
        # foot has its own `busy_status_label`.
        self.source_status_label = QLabel("\u2014 not loaded")
        self.source_status_label.setWordWrap(True)
        self.source_status_label.setStyleSheet(DIM_LABEL_STYLE)
        layout.addWidget(self.source_status_label)

        layout.addWidget(self._separator())

        # --- Voxel size: metadata ONLY, no Apply ---
        layout.addLayout(self._make_subsection_header(
            "Voxel size", "step1_load_voxel.md"))

        _vdesc = QLabel(
            "Physical spacing. It defines the units every box and resample "
            "step uses. It is never applied to the viewer; recording it with "
            "each patch needs the spacing reader."
        )
        _vdesc.setWordWrap(True)
        _vdesc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(_vdesc)

        # ONE source of truth for both the initial values and `\u21ba`, so reset
        # cannot disagree with what was shipped.  1 \u00b5m isotropic until
        # `read_voxel_spacing()` exists \u2014 NOT the mockup's illustrative 2/1/1,
        # which would have the panel claim it detected a spacing it never read.
        self._detected_voxel_size = (1.0, 1.0, 1.0)
        vz, vy, vx = self._detected_voxel_size
        self.voxel_z_spin = self._make_voxel_spin(vz)
        self.voxel_y_spin = self._make_voxel_spin(vy)
        self.voxel_x_spin = self._make_voxel_spin(vx)

        vox_row = QHBoxLayout()
        vox_row.setSpacing(4)
        for axis, spin in (("Z", self.voxel_z_spin), ("Y", self.voxel_y_spin),
                           ("X", self.voxel_x_spin)):
            tag = QLabel(axis)
            tag.setStyleSheet(DIM_LABEL_STYLE)
            vox_row.addWidget(tag)
            vox_row.addWidget(spin, 1)
        self.voxel_reset_button = QPushButton("\u21ba")
        self.voxel_reset_button.setFixedSize(24, 24)
        self.voxel_reset_button.setToolTip("Reset to the values read from the file")
        vox_row.addWidget(self.voxel_reset_button, 0)
        layout.addLayout(vox_row)

        # `voxel_size_source: detected | user` \u2014 in a tool whose selling point
        # is provenance, a hand-edited spacing must be distinguishable from a
        # read one.
        self.voxel_source_label = QLabel("\u2014 load a source to read spacing")
        self.voxel_source_label.setWordWrap(True)
        self.voxel_source_label.setStyleSheet(DIM_LABEL_STYLE)
        layout.addWidget(self.voxel_source_label)

        self.voxel_aspect_label = QLabel("")
        self.voxel_aspect_label.setWordWrap(True)
        self.voxel_aspect_label.setStyleSheet(DIM_LABEL_STYLE)
        layout.addWidget(self.voxel_aspect_label)

        layout.addWidget(self._separator())

        # --- Raw / Processed view ---  (Brick A: kill the additive-blend overlay;
        # one representation at a time via a single segmented toggle.  Governs the
        # VIEW now; will govern EXTRACT in Brick B — hence the "sets view + extract"
        # note + the ③ mirror.)  Replaces the old "Extract from" radio pair.
        layout.addLayout(self._make_subsection_header(
            "Raw / Processed", "step1_load_preprocess.md"))

        # A two-button segmented toggle (exclusive).  "Processed" is greyed until a
        # processed version exists for the loaded source THIS session; the
        # controller lights it up via `_set_view_toggle_state` after Process.
        self.view_raw_button = QPushButton("Raw")
        self.view_raw_button.setCheckable(True)
        self.view_raw_button.setChecked(True)
        self.view_raw_button.setToolTip(
            "Show the raw source.  \\ toggles raw / processed."
        )
        self.view_processed_button = QPushButton("Processed")
        self.view_processed_button.setCheckable(True)
        self.view_processed_button.setEnabled(False)
        self.view_processed_button.setToolTip(
            "Show the processed result (available after Process whole volume).  "
            "\\ toggles raw / processed."
        )
        self._view_toggle_group = QButtonGroup(self)
        self._view_toggle_group.setExclusive(True)
        self._view_toggle_group.addButton(self.view_raw_button)
        self._view_toggle_group.addButton(self.view_processed_button)
        for _b in (self.view_raw_button, self.view_processed_button):
            _b.setStyleSheet(toggle_btn_style(_STEP1_ACCENT))

        view_row = QHBoxLayout()
        view_row.setSpacing(0)
        view_row.addWidget(self.view_raw_button)
        view_row.addWidget(self.view_processed_button)
        view_row.addStretch(1)
        layout.addLayout(view_row)

        _vnote = QLabel("sets view + extract")
        _vnote.setWordWrap(True)
        _vnote.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(_vnote)

        # Region compare-splice tool — present-but-disabled until Brick C
        # wires the comparator.
        self.compare_raw_processed_button = QPushButton(
            "⧉ Compare raw / processed")
        self.compare_raw_processed_button.setEnabled(False)
        self.compare_raw_processed_button.setStyleSheet(SECONDARY_BTN_STYLE)
        self.compare_raw_processed_button.setToolTip(
            "Coming soon — splice a region of raw and processed to compare them."
        )
        layout.addWidget(self.compare_raw_processed_button)
        # v1: "Coming soon" is not a feature.  Hidden, not deleted -- the
        # controller binds this by name.
        self.compare_raw_processed_button.setVisible(False)

        layout.addStretch(1)
        return column

    @staticmethod
    def _make_voxel_spin(default: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        # 3 decimals + a " um" suffix + the step arrows need real room; without a
        # floor the three spins collapse to unreadable slivers whenever the
        # section is squeezed.
        spin.setMinimumWidth(72)
        spin.setRange(0.001, 1000.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        spin.setValue(default)
        spin.setSuffix(" \u00b5m")
        return spin

    def _update_voxel_aspect(self, *_args) -> None:
        """State the anisotropy in words.

        Step 1 never writes `layer.scale`, so the canvas cannot
        show that Z is coarser than XY.  Saying it here is the cheap half of
        what the test-region preview does properly.
        """
        z = self.voxel_z_spin.value()
        xy = min(self.voxel_y_spin.value(), self.voxel_x_spin.value())
        if xy <= 0:
            self.voxel_aspect_label.setText("")
        elif abs(z / xy - 1.0) < 0.01:
            self.voxel_aspect_label.setText("isotropic")
        elif z > xy:
            self.voxel_aspect_label.setText(
                f"aspect ratio: Z is {z / xy:.3g}\u00d7 coarser than XY")
        else:
            self.voxel_aspect_label.setText(
                f"aspect ratio: Z is {xy / z:.3g}\u00d7 finer than XY")

    def _mark_voxel_edited(self, *_args) -> None:
        # A genuine user edit (valueChanged) \u2014 mark dirty so
        # `_on_voxel_editing_finished` persists it, and flip the provenance label.
        self._voxel_dirty = True
        self.voxel_source_label.setText("edited by hand \u00b7 \u21ba to restore detected")

    def _on_voxel_editing_finished(self, *_args) -> None:
        """Voxel spin ``editingFinished`` \u2014 persist ONLY on a real edit.

        ``editingFinished`` also fires on a focus-out with no change, and on a
        programmatic restore's blocked ``setValue`` the dirty flag stays clear, so
        gating on ``_voxel_dirty`` prevents a spurious ``detected -> user`` type
        flip."""
        if not getattr(self, "_voxel_dirty", False):
            return
        self._voxel_dirty = False
        cb = getattr(self.controller, "on_voxel_edited", None)
        if cb is not None:
            cb()

    def set_detected_voxel_size(self, z, y, x) -> None:
        """Set the DETECTED baseline the ``\u21ba`` reset restores \u2014
        the header/store spacing, NOT ``(1, 1, 1)``.  Does not touch the spins."""
        try:
            self._detected_voxel_size = (float(z), float(y), float(x))
        except (TypeError, ValueError):
            self._detected_voxel_size = (1.0, 1.0, 1.0)

    def apply_voxel_size(self, z, y, x, source="detected") -> None:
        """Push a resolved voxel into the spins + set the honest provenance label.
        Blocked ``setValue`` so it does not fire the edit/save
        signals or mark the field dirty."""
        for spin, value in ((self.voxel_z_spin, z), (self.voxel_y_spin, y),
                            (self.voxel_x_spin, x)):
            spin.blockSignals(True)
            try:
                spin.setValue(float(value))
            except (TypeError, ValueError):
                pass
            spin.blockSignals(False)
        self._voxel_dirty = False
        self._set_voxel_source_label(source)
        self._update_voxel_aspect()

    def _set_voxel_source_label(self, source) -> None:
        """Honest voxel provenance line (``user`` / ``detected`` / ``unknown``)."""
        if source == "user":
            self.voxel_source_label.setText("your value \u00b7 \u21ba to restore detected")
        elif source == "detected":
            self.voxel_source_label.setText("detected from file")
        else:
            self.voxel_source_label.setText("\u2014 unknown (1 \u00b5m assumed)")

    def _reset_voxel_to_detected(self) -> None:
        z, y, x = self._detected_voxel_size
        for spin, value in ((self.voxel_z_spin, z), (self.voxel_y_spin, y),
                            (self.voxel_x_spin, x)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._voxel_dirty = False
        # Honest label: a real detected baseline says "detected", a 1/1/1 baseline
        # is "unknown" (never silently presented as a detected spacing).
        is_unit = all(abs(float(v) - 1.0) < 1e-9 for v in (z, y, x))
        self._set_voxel_source_label("unknown" if is_unit else "detected")
        self._update_voxel_aspect()

    def _update_extract_source_hint(self, *_args) -> None:
        """Mirror the Raw / Processed selection next to \u2462's Extract button.

        The exclusive accordion means the toggle in \u2461 and the Extract button in
        \u2462 can never be seen together, so the state has to travel.

        The hint used to end "nothing on disk until Review & Library ›
        Export", which was simply untrue: Extract WRITES each patch (TIFF +
        sidecar + registry entry) as it goes, and ④ lists what is already
        on disk - there is no separate Export step to reach.  Say what
        actually happens instead.
        """
        hint = getattr(self, "extract_source_hint", None)
        if hint is None:
            return
        which = ("processed"
                 if getattr(self, "view_processed_button", None) is not None
                 and self.view_processed_button.isChecked()
                 else "raw")
        hint.setText(
            f"— extracts from {which} · saved to disk right away"
        )

    def _on_view_toggle(self, button, checked) -> None:
        """Route the Raw / Processed segmented toggle to the controller's
        mutually-exclusive VIEW swap (Brick A).  Fires only on the button that
        BECAME checked; the controller hides one set + shows the other (no
        unload/reload)."""
        if not checked:
            return
        ctrl = getattr(self, "controller", None)
        if button is self.view_processed_button:
            fn = getattr(ctrl, "_show_processed_view", None)
        else:
            fn = getattr(ctrl, "_show_raw_view", None)
        if callable(fn):
            fn()
        self._update_extract_source_hint()

    def _set_view_toggle_state(self, has_processed: bool,
                               showing_processed: bool) -> None:
        """Reflect the controller's AUTHORITATIVE view state onto \u2461's Raw /
        Processed toggle (Brick A): enable/grey the Processed button, and check
        the segment that matches which set is actually on the canvas (read off
        `.visible` by the controller).  Sets the checked state with
        signals BLOCKED so this never re-enters the controller.

        The Compare button stays disabled in Brick A (present-but-disabled until
        Brick C wires the comparator)."""
        self.view_processed_button.setEnabled(bool(has_processed))
        target = (self.view_processed_button
                  if (has_processed and showing_processed)
                  else self.view_raw_button)
        blocked = (self._view_toggle_group, self.view_raw_button,
                   self.view_processed_button)
        for w in blocked:
            w.blockSignals(True)
        try:
            target.setChecked(True)
        finally:
            for w in blocked:
                w.blockSignals(False)
        self._update_extract_source_hint()

    def _create_preprocessing_column(self):
        """② right column: the per-channel COLUMN recipe builder.

        Locked via the v2 interactive prototype: one COLUMN per detected channel,
        side-by-side in a horizontal scroller (min ~210px each), each column its
        own ordered list of intensity op chips + a per-column "+ Add step".  The
        single shared, linked GEOMETRY step (isotropic rescale) renders as a
        blue-tinted chip INSIDE every column at one shared boundary index — add /
        edit / move / remove syncs across every column because it is ONE
        `GeometryStep` (the geometry-interleave rule).

        The panel holds + edits a `Recipe` (packages.epicycle_preprocessing.recipe);
        the columns are a PROJECTION rebuilt WHOLESALE from it, so a ✕/▲▼ can
        never act on a stale index.

        Brick 1b: the UI is real + clickable but the ops are STUBBED — Preview /
        Apply only LOG the flattened recipe (`recipe.flatten_for_channel`); no op
        math, no bake, no extract routing (Brick 2/3).  The menu reads the REAL
        `packages/epicycle_preprocessing` registry, not a placeholder.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 6, 6, 6)
        layout.setSpacing(6)

        layout.addLayout(self._make_subsection_header(
            "Preprocess", "step1_load_preprocess.md"))

        helper = QLabel(
            "One column per channel. Steps run top to bottom. The blue "
            "resample chip is shared — it applies to every channel identically."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(helper)

        # ── ②'s body is a VERTICAL SPLITTER: the per-channel card strip on top,
        # the before→after plot below (user 2026-08-07 — "half card, half
        # histogram").  A thin, subtle handle lets the biologist trade card room
        # for plot room; neither pane can collapse to nothing.
        body_split = QSplitter(Qt.Vertical)
        body_split.setStyleSheet(SPLITTER_STYLE)
        body_split.setChildrenCollapsible(False)
        body_split.setHandleWidth(5)

        # -- TOP pane: the themed Box-List FRAME holding the horizontal card
        # strip.  The columns live in a plain container QWidget with an HBox;
        # `_refresh_recipe_view` rebuilds that HBox WHOLESALE from the recipe.
        # The container is what the ② busy-lock disables (one name → every
        # per-column control locks; the single `add_op_button` it replaces is
        # removed from that loop).
        box_frame = QFrame()
        box_frame.setObjectName("recipeBoxFrame")
        box_frame.setAttribute(Qt.WA_StyledBackground, True)
        box_frame.setFrameShape(QFrame.NoFrame)
        box_frame.setMinimumHeight(150)
        box_frame.setStyleSheet(
            "#recipeBoxFrame { background: #202020;"
            " border: 1px solid rgba(255,255,255,0.10); border-radius: 6px; }")
        box_lay = QVBoxLayout(box_frame)
        box_lay.setContentsMargins(2, 2, 2, 2)
        box_lay.setSpacing(2)

        # Frame header row = channel count (right-aligned, echoing the Box List
        # count grammar) + the empty-state message (shown only when 0 channels).
        head = QHBoxLayout()
        head.setContentsMargins(10, 6, 10, 2)
        head.setSpacing(6)
        self.recipe_empty_label = QLabel(
            "No channels yet — load a source in this section to build a "
            "per-channel recipe."
        )
        self.recipe_empty_label.setWordWrap(True)
        self.recipe_empty_label.setStyleSheet(DIM_LABEL_STYLE)
        head.addWidget(self.recipe_empty_label, 1)
        self.recipe_channel_count_label = QLabel("0 channels")
        self.recipe_channel_count_label.setStyleSheet(DIM_LABEL_STYLE)
        head.addWidget(self.recipe_channel_count_label, 0,
                       Qt.AlignRight | Qt.AlignVCenter)
        box_lay.addLayout(head)

        self.recipe_columns_container = QWidget()
        self._recipe_columns_layout = QHBoxLayout(self.recipe_columns_container)
        self._recipe_columns_layout.setContentsMargins(11, 11, 11, 11)
        self._recipe_columns_layout.setSpacing(11)
        self._recipe_columns_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.recipe_scroll = QScrollArea()
        self.recipe_scroll.setWidgetResizable(True)
        self.recipe_scroll.setWidget(self.recipe_columns_container)
        self.recipe_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.recipe_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Transparent + borderless so the FRAME draws the box, not the scroll —
        # the merge-dialog `_SCROLL_AREA_STYLE` idiom.  The cards scroll
        # HORIZONTALLY when there are more channels than fit across.
        self.recipe_scroll.setStyleSheet(_RECIPE_SCROLL_STYLE)
        box_lay.addWidget(self.recipe_scroll, 1)
        body_split.addWidget(box_frame)

        # -- BOTTOM pane: a CONTAINER that stacks the before→after plot (stretch
        # 1) OVER the Region-editor card (fixed).  This is the "plot cedes height
        # on reveal" mechanism (preferred): when the region-editor body
        # reveals inside the card, the card grows by its own sizeHint and the
        # PLOT — its stretch-1 sibling — gives up exactly that height.  The top
        # card strip is governed by the splitter (doesn't move) and the pinned
        # footer lives BELOW body_split (doesn't move).
        bottom_pane = QWidget()
        bottom_lay = QVBoxLayout(bottom_pane)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(6)
        # Sized so the plot (150px min) + an EXPANDED card (~150px) both fit
        # without clipping; the splitter can still trade this pane's share with
        # the top card strip.
        bottom_pane.setMinimumHeight(300)

        # raw→processed intensity histogram FRAME — a REAL pyqtgraph plot themed
        # like the QC plots (`qc_plot_widget.py:408`).  Layout top→bottom: a
        # controls row (channel selector + Log-Y / Raw-DN view toggles), the plot
        # (with the dim hint overlaid centred until a Preview draws), and a factual
        # stat read-out line.  All view controls act on the ALREADY-COMPUTED
        # per-channel entries — none of them recompute (Preview owns compute).
        self.recipe_histogram_area = QFrame()
        self.recipe_histogram_area.setObjectName("recipePlotFrame")
        self.recipe_histogram_area.setAttribute(Qt.WA_StyledBackground, True)
        self.recipe_histogram_area.setFrameShape(QFrame.NoFrame)
        self.recipe_histogram_area.setMinimumHeight(180)
        self.recipe_histogram_area.setStyleSheet(
            "#recipePlotFrame { background: #202020;"
            " border: 1px solid rgba(255,255,255,0.10); border-radius: 6px; }")
        hist_lay = QVBoxLayout(self.recipe_histogram_area)
        hist_lay.setContentsMargins(6, 6, 6, 6)
        hist_lay.setSpacing(4)

        # ── histogram state (the entries are cached so channel / view switches are
        # INSTANT — no recompute).
        self._recipe_hist_entries = []
        self._recipe_hist_selected_channel = 0
        self._recipe_hist_stale = False
        self._recipe_hist_lo = -0.1     # matches controller._PREVIEW_HIST_LO
        self._recipe_hist_hi = 1.1      # matches controller._PREVIEW_HIST_HI
        self._recipe_hist_default_hint = (
            "Draw a region, then Preview to compare raw vs processed intensity.")

        # ── Controls row: channel selector (left) + view toggles (right).
        hist_controls = QWidget()
        hcl = QHBoxLayout(hist_controls)
        hcl.setContentsMargins(0, 0, 0, 0)
        hcl.setSpacing(6)

        # Channel selector — exclusive checkable "tabs"; the CHECKED (accent-tinted)
        # one is the active channel whose before/after is shown.  Only the selected
        # channel is drawn — never an all-channel overlay.
        self.recipe_hist_channel_bar = QWidget()
        self._recipe_hist_channel_layout = QHBoxLayout(self.recipe_hist_channel_bar)
        self._recipe_hist_channel_layout.setContentsMargins(0, 0, 0, 0)
        self._recipe_hist_channel_layout.setSpacing(3)
        self._recipe_hist_channel_group = QButtonGroup(self.recipe_hist_channel_bar)
        self._recipe_hist_channel_group.setExclusive(True)
        self._recipe_hist_channel_buttons = []
        hcl.addWidget(self.recipe_hist_channel_bar, 0)
        hcl.addStretch(1)

        # Log-Y (default ON) — sparse bright tails stay visible next to the pile-up;
        # empty bins render as 0 (log1p), never -inf.
        self.recipe_hist_logy_toggle = QPushButton("Log Y")
        self.recipe_hist_logy_toggle.setCheckable(True)
        self.recipe_hist_logy_toggle.setChecked(True)
        self.recipe_hist_logy_toggle.setStyleSheet(toggle_btn_style(_STEP1_ACCENT))
        self.recipe_hist_logy_toggle.setToolTip(
            "Log-scale the height axis (default) so sparse bright tails stay "
            "visible next to the tall pile-up. Off = a straight scale.  "
            "Height is voxels per million voxels, so the raw and processed "
            "curves stay comparable even when a resize step changes how many "
            "voxels there are.")
        self.recipe_hist_logy_toggle.toggled.connect(
            lambda _c: self._redraw_recipe_histogram())
        hcl.addWidget(self.recipe_hist_logy_toggle, 0)

        # Raw-DN x-axis — same curves, ticks relabelled to raw detector counts.
        self.recipe_hist_rawaxis_toggle = QPushButton("Raw DN")
        self.recipe_hist_rawaxis_toggle.setCheckable(True)
        self.recipe_hist_rawaxis_toggle.setChecked(False)
        self.recipe_hist_rawaxis_toggle.setStyleSheet(toggle_btn_style(_STEP1_ACCENT))
        self.recipe_hist_rawaxis_toggle.setToolTip(
            "Label the x-axis in raw detector counts (DN) instead of normalized "
            "intensity. Same curves — only the tick labels change "
            "(raw = u·(p99.5−p0.1) + p0.1).\n\n"
            "Caveat: these raw-DN ticks are exact only for the raw (before) "
            "curve. Remapping ops (CLAHE / gamma / background / bleach) rescale "
            "intensities, so the processed curve's working-space values no "
            "longer correspond to those raw brightnesses.")
        self.recipe_hist_rawaxis_toggle.toggled.connect(
            lambda _c: self._redraw_recipe_histogram())
        hcl.addWidget(self.recipe_hist_rawaxis_toggle, 0)

        hist_lay.addWidget(hist_controls, 0)

        # ── Plot holder: the pyqtgraph plot with the hint overlaid centred.
        plot_holder = QWidget()
        plot_grid = QGridLayout(plot_holder)
        plot_grid.setContentsMargins(0, 0, 0, 0)

        self._recipe_plot_widget = pg.PlotWidget()
        self._recipe_plot_widget.setBackground("#1a1a2e")
        self._recipe_plot_widget.showGrid(x=True, y=True, alpha=20 / 255)
        _plot_item = self._recipe_plot_widget.plotItem
        _plot_item.setContentsMargins(4, 4, 4, 4)
        for _axis_name in ("bottom", "left"):
            _ax = self._recipe_plot_widget.getAxis(_axis_name)
            _ax.setTextPen(pg.mkPen(color=(120, 120, 130), width=1))
            _ax.setPen(pg.mkPen(color=(50, 50, 60), width=1))
        self._recipe_plot_widget.setLabel(
            "bottom", "normalized intensity (0 = p0.1, 1 = p99.5 of raw)")
        self._recipe_plot_widget.setLabel("left", "voxels per million voxels")
        _plot_item.setTitle(None)
        plot_grid.addWidget(self._recipe_plot_widget, 0, 0)

        self.recipe_histogram_hint = QLabel(self._recipe_hist_default_hint)
        self.recipe_histogram_hint.setAlignment(Qt.AlignCenter)
        self.recipe_histogram_hint.setWordWrap(True)
        self.recipe_histogram_hint.setStyleSheet(
            "color: #6f6f6f; font-size: 11px; background: transparent;"
            " border: none;")
        # Let clicks pass through to the plot beneath.
        self.recipe_histogram_hint.setAttribute(
            Qt.WA_TransparentForMouseEvents, True)
        # Overlaid in the SAME grid cell, added last so it paints ON TOP.
        plot_grid.addWidget(self.recipe_histogram_hint, 0, 0, Qt.AlignCenter)
        hist_lay.addWidget(plot_holder, 1)

        # ── Stat read-out — factual numbers only (no mean/SD/SNR, no composite
        # "contrast improved" score).  Hidden until a preview is drawn.
        self.recipe_hist_stat_label = QLabel("")
        self.recipe_hist_stat_label.setWordWrap(True)
        self.recipe_hist_stat_label.setStyleSheet(
            "color: #9aa0a6; font-size: 10px; background: transparent;"
            " border: none;")
        self.recipe_hist_stat_label.setVisible(False)
        hist_lay.addWidget(self.recipe_hist_stat_label, 0)

        # Controls hidden until the first draw.
        self.recipe_hist_channel_bar.setVisible(False)
        self.recipe_hist_logy_toggle.setVisible(False)
        self.recipe_hist_rawaxis_toggle.setVisible(False)

        # stretch 1 → the plot area is the pane's elastic element; it cedes height
        # to the fixed card when the editor body reveals.
        bottom_lay.addWidget(self.recipe_histogram_area, 1)

        # ── Region-editor CARD (Preview v2 Piece 2): a bordered frame that is the
        # PERMANENT home of the preview-region controls, themed like the box /
        # plot frames.  COLLAPSED (no region) it shows only the "Place preview
        # region" CTA; EXPANDED (region placed) the editor body replaces it
        # (`preview_region_group`: header + Recenter/S-M-L + the 2×2 sliders).
        # The FRAME is ALWAYS visible; only the body toggles — the whole reveal
        # runs through `set_preview_region_active`, which also owns the box + its
        # overlay via the controller.
        self.preview_region_card = QFrame()
        self.preview_region_card.setObjectName("recipeRegionCard")
        self.preview_region_card.setAttribute(Qt.WA_StyledBackground, True)
        self.preview_region_card.setFrameShape(QFrame.NoFrame)
        self.preview_region_card.setStyleSheet(
            "#recipeRegionCard { background: #202020;"
            " border: 1px solid rgba(255,255,255,0.10); border-radius: 6px; }")
        card_lay = QVBoxLayout(self.preview_region_card)
        card_lay.setContentsMargins(10, 8, 10, 8)
        card_lay.setSpacing(6)

        # Collapsed CTA — the ONLY thing the card shows until a region is placed.
        # It hides (via `set_preview_region_active`) once the editor body reveals,
        # so the card reads as "just Place" collapsed and "the editor" expanded.
        place_row = QHBoxLayout()
        place_row.setSpacing(6)
        self.place_preview_button = QPushButton("▭ Place a region to preview")
        self.place_preview_button.setToolTip(
            "Drop a preview region box at the current view centre. Preview then "
            "computes the recipe on THIS region — move it to a representative "
            "area (dense nuclei, a fold, a dim corner).")
        self.place_preview_button.clicked.connect(self._on_place_preview_clicked)
        place_row.addWidget(self.place_preview_button, 0)
        place_row.addStretch(1)
        card_lay.addLayout(place_row)

        # The grouped editor is ONE container so it reveals / collapses as a unit
        # (`set_preview_region_active`).  Hidden until a region is placed; lives
        # INSIDE the always-visible card.
        self.preview_region_group = QWidget()
        prg = QVBoxLayout(self.preview_region_group)
        prg.setContentsMargins(0, 0, 0, 0)
        prg.setSpacing(4)

        # Card HEADER row: "Region editor" title + Exit (only visible once a
        # region is placed, since the whole group hides when collapsed).
        head_row = QHBoxLayout()
        head_row.setSpacing(6)
        region_title = QLabel("Region editor")
        region_title.setStyleSheet(
            "color: #cfcfcf; font-size: 11px; font-weight: 600;")
        head_row.addWidget(region_title, 0)
        head_row.addStretch(1)
        self.exit_preview_button = QPushButton("✕ Exit preview")
        self.exit_preview_button.setToolTip(
            "Remove the preview region + any preview layers and collapse these "
            "controls.")
        self.exit_preview_button.clicked.connect(self._on_exit_preview_clicked)
        head_row.addWidget(self.exit_preview_button, 0)
        prg.addLayout(head_row)

        # Recenter + S/M/L size presets.
        pr_row = QHBoxLayout()
        pr_row.setSpacing(6)
        self.recenter_preview_button = QPushButton("⌖ Recenter")
        self.recenter_preview_button.setToolTip(
            "Move the preview region to the current view centre (keeps its size).")
        self.recenter_preview_button.clicked.connect(
            self._on_recenter_preview_clicked)
        pr_row.addWidget(self.recenter_preview_button, 0)

        self.preview_size_preset_buttons = []
        for _label, _frac, _tip in (
                ("S", 0.125, "Small — 1/8 of the field"),
                ("M", 0.25, "Medium — 1/4 of the field"),
                ("L", 0.5, "Large — 1/2 of the field")):
            _b = QPushButton(_label)
            _b.setToolTip(
                f"{_tip}. Resizes the preview region and sets the next size.")
            _b.clicked.connect(
                lambda _c=False, f=_frac: self._on_preview_size_clicked(f))
            self.preview_size_preset_buttons.append(_b)
            pr_row.addWidget(_b, 0)
        pr_row.addStretch(1)
        prg.addLayout(pr_row)

        # X/Y/Z position + Z-depth sliders in a 2×2 GRID — dragging one MOVES /
        # RESIZES the box LIVE (the controller clears any shown preview so it
        # can't disagree with a moved box; the user re-clicks Preview to
        # recompute).  LEFT half (cols 0-2) = X (row0) over Y (row1); RIGHT half
        # (cols 4-6) = Z-position (row0) over Z-depth (row1); each track expands
        # full-width within its half.  Z-position is inert while the box spans all
        # Z, so it greys until Z-depth < full.
        sld_grid = QGridLayout()
        sld_grid.setContentsMargins(2, 0, 2, 0)
        sld_grid.setHorizontalSpacing(6)
        sld_grid.setVerticalSpacing(3)
        self._preview_slider_grid = sld_grid

        def _add_pr_slider(row, col0, tag, tip):
            cap = QLabel(tag)
            cap.setStyleSheet(DIM_LABEL_STYLE)
            cap.setFixedWidth(56)
            cap.setToolTip(tip)
            slider = QSlider(Qt.Horizontal)
            slider.setToolTip(tip)
            val = QLabel("—")
            val.setStyleSheet("font-size: 10px; color: #9aa0a6;")
            val.setFixedWidth(60)   # wider: shows "value / max" (e.g. 128 / 512)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            sld_grid.addWidget(cap, row, col0)
            sld_grid.addWidget(slider, row, col0 + 1)
            sld_grid.addWidget(val, row, col0 + 2)
            return slider, val

        # LEFT half: X over Y.
        self.preview_x_slider, self.preview_x_value_label = _add_pr_slider(
            0, 0, "X position", "Move the preview region along X (width).")
        self.preview_y_slider, self.preview_y_value_label = _add_pr_slider(
            1, 0, "Y position", "Move the preview region along Y (height).")
        # RIGHT half: Z-position over Z-depth.
        self.preview_z_slider, self.preview_z_value_label = _add_pr_slider(
            0, 4, "Z position",
            "Move the preview region along Z (depth). Inert — and greyed — while "
            "the region spans the whole Z stack.")
        self.preview_zdepth_slider, self.preview_zdepth_value_label = \
            _add_pr_slider(
                1, 4, "Z depth",
                "How many Z slices the region spans. Full depth = the whole stack "
                "(the default), which is why Z position is greyed there.")
        # Each slider track expands within its half; a thin gutter (col 3)
        # separates the two halves.
        sld_grid.setColumnStretch(1, 1)
        sld_grid.setColumnStretch(5, 1)
        sld_grid.setColumnMinimumWidth(3, 14)
        prg.addLayout(sld_grid)

        # Programmatic-sync guard: `set_preview_region_active` seeds slider values
        # from the box, and those `setValue` calls must NOT be read back as user
        # moves (else a feedback loop + spurious stale-token bumps).
        self._syncing_preview_sliders = False
        self.preview_x_slider.valueChanged.connect(self._on_preview_x_changed)
        self.preview_y_slider.valueChanged.connect(self._on_preview_y_changed)
        self.preview_z_slider.valueChanged.connect(self._on_preview_z_changed)
        self.preview_zdepth_slider.valueChanged.connect(
            self._on_preview_zdepth_changed)
        # The full (Z, Y, X) shape the sliders range over — set on reveal.
        self._preview_full_shape = None

        # ── Preview — the region editor's PROMINENT bottom action (button-
        # hierarchy pass).  Preview is region-scoped + ITERATIVE (clicked many
        # times as the region is moved/resized), so it belongs INSIDE the card,
        # below the sliders, separated by a hairline.  Styled prominent-but-tonal
        # (SECONDARY_BTN_STYLE) — the primary action WITHIN the card, yet
        # deliberately LIGHTER than the lone accent Apply in the pinned footer
        # (Apply is the recipe-scoped, one-shot terminal bake).  It reveals /
        # collapses with `preview_region_group`, so it is only visible once a
        # region is placed (intended).  Enable stays owned by
        # `_refresh_preview_button_state`.
        prg.addWidget(self._separator())
        self.test_region_button = QPushButton("↻ Preview this region")
        self.test_region_button.setStyleSheet(SECONDARY_BTN_STYLE)
        self.test_region_button.clicked.connect(self._on_preview_clicked)
        prg.addWidget(self.test_region_button)

        self.preview_region_group.setVisible(False)
        card_lay.addWidget(self.preview_region_group)
        # stretch 0 → the card is fixed; it takes its sizeHint and the plot cedes.
        bottom_lay.addWidget(self.preview_region_card, 0)

        body_split.addWidget(bottom_pane)

        # 50/50 to start; both panes stretch equally on resize.
        body_split.setStretchFactor(0, 1)
        body_split.setStretchFactor(1, 1)
        body_split.setSizes([1000, 1000])
        layout.addWidget(body_split, 1)

        # ── PINNED footer: the recipe-order hint over the LONE accent Apply
        # button, then the status line.  Added to the PANEL's top-level layout
        # AFTER body_split, so it is a SIBLING of the splitter — never a
        # descendant of body_split or the region card.  It therefore stays at the
        # panel bottom and does NOT move whether or not a region is placed.
        # Preview moved INTO the region card (region-scoped +
        # iterative); "Process whole volume" stays here as the single
        # recipe-scoped, terminal run — the ONE accent-filled button in the panel,
        # full-width on its own row.  It runs the recipe over the WHOLE volume on
        # a worker → fixed-scale uint16, holds the result in memory and DISPLAYS
        # it (WYSIWYG); NO on-disk persistence yet (Brick 1b-1).  While a run is
        # in flight the button doubles as Cancel.  Its enable + tooltip stay owned
        # by `_refresh_recipe_view`.
        foot = QVBoxLayout()
        foot.setSpacing(4)
        self.recipe_order_hint = QLabel("")
        self.recipe_order_hint.setStyleSheet(DIM_LABEL_STYLE)
        foot.addWidget(self.recipe_order_hint, 0)

        self.apply_preprocess_button = QPushButton("Process whole volume")
        self.apply_preprocess_button.setStyleSheet(PRIMARY_BTN_STYLE_TEAL)
        self.apply_preprocess_button.clicked.connect(self._on_apply_clicked)
        foot.addWidget(self.apply_preprocess_button, 0)
        layout.addLayout(foot)

        self.preprocess_status_label = QLabel("— not applied")
        self.preprocess_status_label.setWordWrap(True)
        self.preprocess_status_label.setStyleSheet(DIM_LABEL_STYLE)
        layout.addWidget(self.preprocess_status_label)

        # The recipe is the persisted + executed source of truth; the columns are
        # a projection of it.  Starts EMPTY; `update_image_layer_list`
        # reconciles it to the detected channel count when a source loads.
        self._recipe = _recipe.Recipe(channels=[], geometry=None)
        self._recipe_channel_names: list[str] = []
        # Preview stays disabled until the controller reports a placed preview
        # REGION box (Preview v2 Piece 2).  Init BEFORE the first
        # `_refresh_recipe_view`, which reads it via `_refresh_preview_button_state`.
        self._preview_region_ready = False
        self._refresh_recipe_view()
        return panel

    # ------------------------------------------------------------------
    # ② per-channel column recipe builder
    # ------------------------------------------------------------------

    #: Canonical geometry-method tokens (recipe config) ↔ display labels.
    _GEO_METHOD_LABELS = {
        "nearest": "Nearest", "linear": "Linear",
        "cubic": "Cubic", "area": "Area-average",
    }

    #: Tooltip restored on the Prepare button when nothing is blocking it.
    _PREPARE_LARGEDATA_TOOLTIP = (
        "Convert this source into a chunked multiscale image so it loads + "
        "pans smoothly without using all your memory. Best for stacks too "
        "large to open whole. Runs in the background; the box tool + "
        "extract then work on the converted image.")

    def _apply_process_button_gate(self, has_channels: bool) -> None:
        """Enable/disable “Process whole volume” and set the matching tooltip.

        Two reasons it can be off: no channels loaded, or (Brick 4a a4) the
        original image file can't be found — the whole-volume run streams the
        RAW file block-wise, so the prepared copy is not a substitute."""
        self._process_gate_has_channels = bool(has_channels)
        if self._raw_missing():
            self.apply_preprocess_button.setEnabled(False)
            self.apply_preprocess_button.setToolTip(
                "Point Step 1 back at the original image file first "
                "(“Find the original file…”)")
            return
        self.apply_preprocess_button.setEnabled(bool(has_channels))
        self.apply_preprocess_button.setToolTip(
            "Run the recipe over the whole volume → uint16, shown in the viewer "
            "and held in memory (not saved to disk yet)"
            if has_channels else "Load a source with channels first")

    def _refresh_recipe_view(self) -> None:
        """Rebuild the per-channel columns WHOLESALE from `self._recipe`.

        A wholesale rebuild (not a patch) means a displayed ✕/▲▼ can never act on
        a stale index — removing channel-0's step 1 can't leave a later handler
        closed over the wrong op (the stale-index trap).  Columns come from the
        DETECTED channel names; the recipe's per-channel lists are indexed to
        match (reconciled in `update_image_layer_list`).
        """
        lay = self._recipe_columns_layout
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        names = self._recipe_channel_names
        has_channels = bool(names)
        self.recipe_empty_label.setVisible(not has_channels)
        self.recipe_scroll.setVisible(has_channels)
        # Hide the "N channels" count in the empty state so it isn't redundant
        # with the "No channels yet…" empty label.
        self.recipe_channel_count_label.setVisible(has_channels)
        # The plot FRAME is ALWAYS mounted in the vertical splitter now (it holds
        # a real pyqtgraph plot), so it no longer hides on the empty state.
        n = len(names)
        self.recipe_channel_count_label.setText(
            f"{n} channel" + ("" if n == 1 else "s"))

        for ci in range(len(names)):
            # INVARIANT: add each card TOP-aligned so all headers line up — a
            # shorter card must never drop to vertical-centre when a sibling
            # grows (pairs with the layout-level AlignLeft|AlignTop above).
            lay.addWidget(self._make_recipe_column(ci), 0, Qt.AlignTop)
        lay.addStretch(1)

        # Preview + Process need a recipe to act on.  Preview is LIVE (computes
        # on the placed region → before/after histogram, Brick 2b); "Process
        # whole volume" runs the recipe over the FULL volume on a worker →
        # fixed-scale uint16, held in memory + displayed (Brick 1b-1; no on-disk
        # save yet).  Preview additionally requires a placed preview REGION
        # (Preview v2 Piece 2) — see `_refresh_preview_button_state`.  Their
        # tooltips must not disagree.
        self._refresh_preview_button_state()
        self._apply_process_button_gate(has_channels)

        self.recipe_order_hint.setText(self._fiji_hint_text())

    def _fiji_hint_text(self) -> str:
        """"order = Fiji default" when channel 0 flattens to the proven Fiji
        order (Gaussian → isotropic → background); else "" (the default)."""
        if not self._recipe.channels:
            return ""
        keys = [s.key for s in _recipe.flatten_for_channel(self._recipe, 0)]
        return "order = Fiji default" if keys[:3] == [
            "gaussian", "isotropic", "background"] else ""

    def _make_recipe_column(self, ci: int) -> QWidget:
        """One channel column: header (LUT swatch · name · chN · clear-✕) + the
        ordered chips (intensity + the shared geometry chip at the boundary) + a
        per-column "+ Add step"."""
        names = self._recipe_channel_names
        name = names[ci] if ci < len(names) else f"ch{ci}"
        # Channel identity colour = the channel's CANVAS colour, keyed off
        # `_SOURCE_DISPLAY_CMAPS` (same index the source layers use), so the card
        # header band + swatch match what the user sees on the canvas.
        lut = _channel_display_hex(ci)
        ops = (self._recipe.channels[ci]
               if ci < len(self._recipe.channels) else [])

        col = QFrame()
        col.setObjectName("recipeCard")
        col.setAttribute(Qt.WA_StyledBackground, True)
        col.setFrameShape(QFrame.NoFrame)
        col.setMinimumWidth(210)
        # Hug content vertically so the card ends at its last step (not a tall
        # empty stripe) — the empty space sits BELOW all cards, per the mockup.
        col.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        # Channel identity = a vivid LUT-coloured HEADER band (below) + a subtle
        # full border (user 2026-08-07: the thin all-around colour read as muted;
        # the coloured header pops).  The card rounds at 6px, and QSS radius does
        # NOT clip children — so the header rounds its TOP corners and the
        # "+ Add step" foot rounds its BOTTOM corners, or a child's square corner
        # pokes past the rounded card (the corner-nub bug).
        r, g, b = int(lut[1:3], 16), int(lut[3:5], 16), int(lut[5:7], 16)
        col.setStyleSheet(
            "#recipeCard { background: #2b2b2b;"
            " border: 1px solid rgba(255,255,255,0.10); border-radius: 6px; }"
        )
        cl = QVBoxLayout(col)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        # -- header: the vivid LUT-coloured band = channel identity --
        head = QWidget()
        head.setObjectName("recipeHd")
        head.setStyleSheet(
            f"#recipeHd {{ background: rgba({r}, {g}, {b}, 0.20);"
            f" border-bottom: 2px solid {lut};"
            " border-top-left-radius: 5px; border-top-right-radius: 5px; }}"
        )
        hl = QHBoxLayout(head)
        hl.setContentsMargins(9, 8, 6, 7)
        hl.setSpacing(7)
        swatch = QLabel("")
        swatch.setFixedSize(11, 11)
        swatch.setStyleSheet(f"background: {lut}; border-radius: 2px; border: none;")
        hl.addWidget(swatch, 0)
        nm = QLabel(name)
        nm.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #eeeeee; border: none;")
        nm.setToolTip(name)
        hl.addWidget(nm, 1)
        tag = QLabel(f"ch{ci}")
        tag.setStyleSheet(DIM_LABEL_STYLE + " border: none;")
        hl.addWidget(tag, 0)
        if ops:
            clr = QPushButton("✕")
            clr.setFixedSize(18, 18)
            clr.setToolTip("Clear this channel's steps")
            clr.setStyleSheet(SECONDARY_BTN_STYLE)
            clr.clicked.connect(
                lambda _c=False, i=ci: self._clear_channel_steps(i))
            hl.addWidget(clr, 0)
        cl.addWidget(head)

        # -- body: interleave the intensity ops with the shared geometry chip at
        #    the shared boundary index (geometry-interleave).  `i` is the op's
        #    index WITHIN the channel's intensity list, not its displayed
        #    position, so a chip's handler acts on the right op.
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(4, 4, 4, 5)
        bl.setSpacing(2)
        geom = self._recipe.geometry
        boundary = None
        if geom is not None:
            try:
                b = int(geom.boundary_index)
            except (TypeError, ValueError):
                b = 0
            boundary = max(0, min(b, len(ops)))
        for i in range(len(ops)):
            if boundary is not None and i == boundary:
                bl.addWidget(self._make_geometry_chip())
            bl.addWidget(self._make_intensity_chip(ci, i, ops[i]))
        if boundary is not None and boundary >= len(ops):
            bl.addWidget(self._make_geometry_chip())
        if not ops and geom is None:
            empty = QLabel("no steps")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet(
                "color: #6f6f6f; font-size: 10px; font-style: italic; border: none;")
            bl.addWidget(empty)
        cl.addWidget(body)

        # -- "+ Add step" --
        add = QPushButton("+ Add step")
        add.setToolTip(
            "Add an intensity op to this channel, or the shared geometry step "
            "to all channels")
        add.setStyleSheet(SECONDARY_BTN_STYLE)
        add.clicked.connect(
            lambda _c=False, i=ci, b=add: self._show_add_op_menu(i, b))
        addwrap = QWidget()
        addwrap.setObjectName("recipeFoot")
        addwrap.setAttribute(Qt.WA_StyledBackground, True)
        # Round the BOTTOM corners to match the card (QSS radius does not clip
        # children — this stops the foot's square corners poking the rounded
        # card, the same corner-nub fix the header applies at the top).
        addwrap.setStyleSheet(
            "#recipeFoot { background: transparent;"
            " border-bottom-left-radius: 5px; border-bottom-right-radius: 5px; }")
        awl = QVBoxLayout(addwrap)
        awl.setContentsMargins(6, 3, 6, 8)
        awl.addWidget(add)
        cl.addWidget(addwrap)
        return col

    def _make_intensity_chip(self, ci: int, i: int, step) -> QWidget:
        """One intensity op chip: name · inline param knob(s) · ▲▼ · ✕."""
        op = _op_registry.find(step.key)
        label = op.label if op is not None else step.key
        chip = QWidget()
        chip.setObjectName("opChip")
        chip.setStyleSheet(
            "#opChip { border-radius: 4px; }"
            "#opChip:hover { background: rgba(255,255,255,0.05); }"
        )
        lay = QHBoxLayout(chip)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(5)

        nm = QLabel(label)
        nm.setStyleSheet("font-size: 12px; color: #dddddd; border: none;")
        nm.setToolTip(label)
        lay.addWidget(nm, 1)

        # Param knob(s) — one per ParamSpec.  Each writes back to the step's
        # params dict WITHOUT a re-render (the knob IS the display, so there is
        # no second readout to keep synced — the stale-index trap only bites
        # STRUCTURAL edits, which do rebuild).
        #
        # ONE param → inline (keeps single-param ops like Gaussian / Background
        # compact).  TWO+ params → STACK them VERTICALLY in a sub-column so a
        # multi-param op (CLAHE clip+kernel, Bleach, and any FUTURE op) stays
        # SLIM instead of doubling the card WIDTH (user request 2026-08-14).
        # This is the ONE builder every op routes through, so the slim layout
        # applies to every current + future op automatically.
        if op is not None and op.params:
            knobs = [
                self._make_param_knob(
                    ci, i, p, step.params.get(p.name, p.default))
                for p in op.params
            ]
            knobs = [k for k in knobs if k is not None]
            if len(knobs) == 1:
                lay.addWidget(knobs[0], 0)
            elif len(knobs) >= 2:
                stack = QWidget()
                sv = QVBoxLayout(stack)
                sv.setContentsMargins(0, 0, 0, 0)
                sv.setSpacing(2)
                for k in knobs:
                    sv.addWidget(k, 0)
                lay.addWidget(stack, 0)

        lay.addWidget(self._make_move_buttons(
            lambda: self._move_intensity_step(ci, i, -1),
            lambda: self._move_intensity_step(ci, i, 1)), 0)

        rm = QPushButton("✕")
        rm.setFixedSize(18, 18)
        rm.setToolTip("Remove this step")
        rm.setStyleSheet(SECONDARY_BTN_STYLE)
        rm.clicked.connect(
            lambda _c=False, c=ci, ii=i: self._remove_intensity_step(c, ii))
        lay.addWidget(rm, 0)
        return chip

    def _make_param_knob(self, ci: int, i: int, pspec, value):
        """A compact `[name][field]` inline editor for one intensity op param.

        Numeric params → spin box with the unit as a suffix; choice params → a
        combo of the declared choices.  Returns None for an unknown type.
        """
        wrap = QWidget()
        wl = QHBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(2)
        lbl = QLabel(pspec.name)
        lbl.setStyleSheet("font-size: 10px; color: #8f8f8f; border: none;")
        lbl.setToolTip(pspec.name)
        wl.addWidget(lbl, 0)

        if pspec.type == "int":
            field = QSpinBox()
            field.setRange(0, 100000)
            try:
                field.setValue(int(value))
            except (TypeError, ValueError):
                field.setValue(int(pspec.default))
            if pspec.unit:
                field.setSuffix(f" {pspec.unit}")
            field.setFixedWidth(64)
            field.valueChanged.connect(
                lambda v, c=ci, ii=i, k=pspec.name:
                self._set_intensity_param(c, ii, k, int(v)))
        elif pspec.type == "float":
            field = QDoubleSpinBox()
            field.setRange(0.0, 100000.0)
            field.setDecimals(3)
            field.setSingleStep(0.1)
            try:
                field.setValue(float(value))
            except (TypeError, ValueError):
                field.setValue(float(pspec.default))
            if pspec.unit:
                field.setSuffix(f" {pspec.unit}")
            field.setFixedWidth(72)
            field.valueChanged.connect(
                lambda v, c=ci, ii=i, k=pspec.name:
                self._set_intensity_param(c, ii, k, float(v)))
        elif pspec.type == "choice":
            field = QComboBox()
            field.addItems([str(x) for x in (pspec.choices or ())])
            idx = field.findText(str(value))
            if idx >= 0:
                field.setCurrentIndex(idx)
            field.setFixedWidth(72)
            field.currentTextChanged.connect(
                lambda t, c=ci, ii=i, k=pspec.name:
                self._set_intensity_param(c, ii, k, t))
        else:
            return None
        field.setStyleSheet(INPUT_STYLE)
        field.setToolTip(pspec.name)
        wl.addWidget(field, 0)
        return wrap

    def _make_move_buttons(self, up_fn, down_fn) -> QWidget:
        """A tight vertical ▲▼ pair used by both intensity + geometry chips."""
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        for glyph, fn, tip in (("▲", up_fn, "Move earlier"),
                               ("▼", down_fn, "Move later")):
            b = QPushButton(glyph)
            b.setFixedSize(16, 12)
            b.setToolTip(tip)
            b.setStyleSheet(
                "QPushButton { border: none; background: transparent;"
                " color: #8f8f8f; font-size: 9px; padding: 0; }"
                "QPushButton:hover { color: #dddddd; }")
            b.clicked.connect(lambda _c=False, f=fn: f())
            vl.addWidget(b)
        return w

    def _make_geometry_chip(self) -> QWidget:
        """The ONE shared, linked geometry chip — blue-tinted, rendered inside
        EVERY column at the shared boundary index.  ⚙ opens the shared
        config; ▲▼ moves the boundary in all columns; ✕ removes it from all."""
        chip = QWidget()
        chip.setObjectName("geoChip")
        chip.setStyleSheet(
            "#geoChip { background: rgba(91,164,207,0.10);"
            " border: 1px solid rgba(91,164,207,0.45); border-radius: 4px; }"
        )
        outer = QVBoxLayout(chip)
        outer.setContentsMargins(6, 5, 6, 6)
        outer.setSpacing(3)

        top = QHBoxLayout()
        top.setSpacing(6)
        chain = QLabel("⇄")
        chain.setStyleSheet(
            f"color: {_STEP1_ACCENT}; font-size: 12px; border: none;")
        top.addWidget(chain, 0)
        gname = QLabel(self._geometry_display_name())
        gname.setStyleSheet(
            "color: #cfeafa; font-size: 12px; font-weight: 600; border: none;")
        top.addWidget(gname, 1)
        gear = QPushButton("⚙")
        gear.setFixedSize(18, 18)
        gear.setToolTip("Configure the shared resample (applies to all channels)")
        gear.setStyleSheet(SECONDARY_BTN_STYLE)
        gear.clicked.connect(lambda _c=False: self._show_geometry_config_dialog())
        top.addWidget(gear, 0)
        top.addWidget(self._make_move_buttons(
            lambda: self._move_geometry(-1),
            lambda: self._move_geometry(1)), 0)
        rm = QPushButton("✕")
        rm.setFixedSize(18, 18)
        rm.setToolTip("Remove the shared resample from all channels")
        rm.setStyleSheet(SECONDARY_BTN_STYLE)
        rm.clicked.connect(lambda _c=False: self._remove_geometry())
        top.addWidget(rm, 0)
        outer.addLayout(top)

        summary = QLabel(self._geometry_summary())
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #8ec2de; font-size: 10px; border: none;")
        outer.addWidget(summary)

        linktag = QLabel("⇄ linked · all channels")
        linktag.setStyleSheet("color: #7fb8d6; font-size: 9px; border: none;")
        outer.addWidget(linktag)
        return chip

    # ---- geometry config helpers ----

    def _geometry_config(self) -> dict:
        """The current geometry config, defaulted for any missing key."""
        geom = self._recipe.geometry
        base = _recipe.default_geometry_config()
        if geom is not None and isinstance(geom.config, dict):
            base.update(geom.config)
        return base

    def _geometry_display_name(self) -> str:
        return ("Isotropic rescale"
                if self._geometry_config().get("mode") == "iso"
                else "Rescale (per-axis)")

    def _geometry_summary(self) -> str:
        cfg = self._geometry_config()
        method = self._GEO_METHOD_LABELS.get(
            str(cfg.get("method", "linear")).lower(),
            str(cfg.get("method", "linear")))
        if cfg.get("mode") == "iso":
            return f"isotropic → {float(cfg.get('target_um', 1.0)):g} µm · {method}"
        factors = list(cfg.get("factors", (1.0, 1.0, 1.0))) + [1.0, 1.0, 1.0]
        z, y, x = factors[:3]
        return (f"rescale  Z {float(z):g} · Y {float(y):g} · X {float(x):g}"
                f"  · {method}")

    def _method_label_to_key(self, label: str) -> str:
        for key, disp in self._GEO_METHOD_LABELS.items():
            if disp == label:
                return key
        return "linear"

    # ---- the "+ Add step" menu (registry-driven) ----

    def _make_add_op_menu(self, ci: int) -> QMenu:
        """Build (but do not exec) the registry-driven `+ Add step` menu for
        channel `ci`.  Factored out of `_show_add_op_menu` so the enabled /
        "— soon" grouping is testable without an `exec_`.

        Grouped by the registry's buckets.  v1 shows ONLY ops that actually
        run: a menu two-thirds full of disabled "— soon" entries advertises a
        toolkit that does not exist, and the pick is refused anyway by
        `_add_op_from_menu`'s re-check.  A Geometry pick sets/moves the shared
        linked chip (all channels); an intensity pick appends to channel `ci`.
        The label → key mapping goes through the OpSpec (steps bind by key).
        Group headings are DISABLED QActions (not `addSection`, which the Windows
        Aero style does not render)."""
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_STYLE)
        menu.setAttribute(Qt.WA_DeleteOnClose)
        first = True
        for bucket, ops in _op_registry.by_bucket().items():
            ops = [op for op in ops if op.implemented]
            if not ops:
                continue          # a bucket with nothing runnable is not a group
            if not first:
                menu.addSeparator()
            first = False
            heading = menu.addAction(
                bucket + (" · all channels" if bucket == "Geometry" else ""))
            heading.setEnabled(False)
            for op in ops:
                action = menu.addAction(op.label)
                # closure-free (final fix pass U4): the pick rides on the
                # action and the slot is a bound method -- a lambda on a
                # QAction nothing keeps is the navigator's GC crash class
                action.setData((int(ci), str(op.key)))
                action.triggered.connect(self._on_add_op_action)
        return menu

    def _on_add_op_action(self, _checked=False) -> None:
        """A `+ Add step` menu pick: ``(channel, op key)`` is the triggering
        action's data (set in `_make_add_op_menu`)."""
        act = self.sender()
        try:
            ci, key = act.data()
        except Exception:      # noqa: BLE001 -- not one of our actions
            return
        self._add_op_from_menu(int(ci), str(key))

    def _show_add_op_menu(self, ci: int, anchor=None) -> None:
        """Open the `+ Add step` menu for channel `ci` under its button."""
        menu = self._make_add_op_menu(ci)
        if isinstance(anchor, QPushButton):
            menu.exec_(anchor.mapToGlobal(anchor.rect().bottomLeft()))
        else:
            menu.exec_(self.recipe_scroll.mapToGlobal(
                self.recipe_scroll.rect().topLeft()))

    def _add_op_from_menu(self, ci: int, key: str) -> None:
        """Route a menu pick: geometry → the shared linked step (all channels);
        intensity → append to channel `ci`."""
        op = _op_registry.find(key)
        if op is None or not op.implemented:
            return
        if op.kind == "geometry":
            self._set_geometry(op, ci)
        else:
            self._append_intensity_step(ci, op)

    # ---- recipe mutators (each rebuilds + persists) ----

    def _ensure_channel(self, ci: int) -> None:
        while len(self._recipe.channels) <= ci:
            self._recipe.channels.append([])

    def _append_intensity_step(self, ci: int, op) -> None:
        self._ensure_channel(ci)
        params = {p.name: p.default for p in op.params}
        self._recipe.channels[ci].append(
            _recipe.IntensityStep(key=op.key, params=params))
        self._log("STEP1-RECIPE", f"ch{ci}: + {op.key}")
        self._refresh_recipe_view()
        self._notify_recipe_edited()

    def _set_geometry(self, op, ci: int) -> None:
        """Add or re-key the ONE shared geometry step.  A fresh add appends it at
        the end of the column it was added from; an existing step keeps its
        config + boundary."""
        if self._recipe.geometry is None:
            ops_len = (len(self._recipe.channels[ci])
                       if ci < len(self._recipe.channels) else 0)
            self._recipe.geometry = _recipe.GeometryStep(
                key=op.key, boundary_index=ops_len,
                config=_recipe.default_geometry_config())
        else:
            self._recipe.geometry.key = op.key
        self._log("STEP1-RECIPE", "+ shared geometry (all channels)")
        self._refresh_recipe_view()
        self._notify_recipe_edited()

    def _move_intensity_step(self, ci: int, i: int, delta: int) -> None:
        if ci >= len(self._recipe.channels):
            return
        ops = self._recipe.channels[ci]
        target = i + delta
        if 0 <= i < len(ops) and 0 <= target < len(ops):
            ops[i], ops[target] = ops[target], ops[i]
            self._refresh_recipe_view()
            self._notify_recipe_edited()

    def _remove_intensity_step(self, ci: int, i: int) -> None:
        if ci >= len(self._recipe.channels):
            return
        ops = self._recipe.channels[ci]
        if 0 <= i < len(ops):
            removed = ops.pop(i)
            self._log("STEP1-RECIPE", f"ch{ci}: − {removed.key}")
            self._refresh_recipe_view()
            self._notify_recipe_edited()

    def _clear_channel_steps(self, ci: int) -> None:
        if ci < len(self._recipe.channels) and self._recipe.channels[ci]:
            self._recipe.channels[ci] = []
            self._log("STEP1-RECIPE", f"ch{ci}: cleared")
            self._refresh_recipe_view()
            self._notify_recipe_edited()

    def _set_intensity_param(self, ci: int, i: int, key: str, value) -> None:
        """Write a param WITHOUT a re-render (the knob is the sole display)."""
        if ci < len(self._recipe.channels):
            ops = self._recipe.channels[ci]
            if 0 <= i < len(ops):
                ops[i].params[key] = value
                self._notify_recipe_edited()

    def _move_geometry(self, delta: int) -> None:
        """Move the shared boundary — re-segments EVERY channel identically."""
        geom = self._recipe.geometry
        if geom is None:
            return
        max_len = max((len(ch) for ch in self._recipe.channels), default=0)
        try:
            b = int(geom.boundary_index)
        except (TypeError, ValueError):
            b = 0
        geom.boundary_index = max(0, min(b + delta, max_len))
        self._refresh_recipe_view()
        self._notify_recipe_edited()

    def _remove_geometry(self) -> None:
        if self._recipe.geometry is not None:
            self._recipe.geometry = None
            self._log("STEP1-RECIPE", "− shared geometry")
            self._refresh_recipe_view()
            self._notify_recipe_edited()

    def _show_geometry_config_dialog(self) -> None:
        """The shared resample config editor — mode (Isotropic | Per-axis),
        target µm OR Z·Y·X factors, method.  Edits apply to the ONE shared step,
        so they sync to every column.  `mode` is stored canonically as
        "iso" | "aniso" (radio Isotropic → "iso", Per-axis → "aniso")."""
        if self._recipe.geometry is None:
            return
        cfg = self._geometry_config()
        dialog = QDialog(self)
        dialog.setWindowTitle("Resample — shared across all channels")
        dialog.setStyleSheet(
            "QDialog { background: #2b2b2b; }"
            "QLabel { color: #cccccc; font-size: 11px; border: none; }")
        form = QVBoxLayout(dialog)
        form.setContentsMargins(14, 12, 14, 12)
        form.setSpacing(9)

        note = QLabel("Geometry applies to every channel identically.")
        note.setStyleSheet("color: #8f8f8f; font-size: 10px; border: none;")
        form.addWidget(note)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        iso_radio = QRadioButton("Isotropic")
        aniso_radio = QRadioButton("Per-axis")
        grp = QButtonGroup(dialog)
        grp.addButton(iso_radio)
        grp.addButton(aniso_radio)
        (iso_radio if cfg.get("mode") == "iso" else aniso_radio).setChecked(True)
        mode_row.addWidget(iso_radio)
        mode_row.addWidget(aniso_radio)
        mode_row.addStretch(1)
        form.addLayout(mode_row)

        # Isotropic row.
        iso_w = QWidget()
        iso_l = QHBoxLayout(iso_w)
        iso_l.setContentsMargins(0, 0, 0, 0)
        iso_l.setSpacing(6)
        iso_l.addWidget(QLabel("Target"))
        tgt = QDoubleSpinBox()
        tgt.setRange(0.001, 100000.0)
        tgt.setDecimals(3)
        tgt.setSingleStep(0.1)
        tgt.setValue(float(cfg.get("target_um", 1.0)))
        tgt.setSuffix(" µm")
        tgt.setStyleSheet(INPUT_STYLE)
        iso_l.addWidget(tgt)
        iso_l.addStretch(1)
        form.addWidget(iso_w)

        # Per-axis row.
        aniso_w = QWidget()
        aniso_l = QHBoxLayout(aniso_w)
        aniso_l.setContentsMargins(0, 0, 0, 0)
        aniso_l.setSpacing(6)
        aniso_l.addWidget(QLabel("Factors"))
        factors = list(cfg.get("factors", (1.0, 1.0, 1.0))) + [1.0, 1.0, 1.0]
        axis_spins = {}
        for ax, val in zip(("Z", "Y", "X"), factors[:3]):
            aniso_l.addWidget(QLabel(ax))
            sp = QDoubleSpinBox()
            sp.setRange(0.001, 100000.0)
            sp.setDecimals(3)
            sp.setSingleStep(0.1)
            sp.setValue(float(val))
            sp.setFixedWidth(66)
            sp.setStyleSheet(INPUT_STYLE)
            aniso_l.addWidget(sp)
            axis_spins[ax] = sp
        aniso_l.addStretch(1)
        form.addWidget(aniso_w)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        method = QComboBox()
        method.addItems(list(self._GEO_METHOD_LABELS.values()))
        method.setCurrentText(self._GEO_METHOD_LABELS.get(
            str(cfg.get("method", "linear")).lower(), "Linear"))
        method.setStyleSheet(INPUT_STYLE)
        method_row.addWidget(method)
        method_row.addStretch(1)
        form.addLayout(method_row)

        def _sync_mode() -> None:
            iso = iso_radio.isChecked()
            iso_w.setVisible(iso)
            aniso_w.setVisible(not iso)
        iso_radio.toggled.connect(lambda _c=False: _sync_mode())
        _sync_mode()

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(SECONDARY_BTN_STYLE)
        cancel.clicked.connect(dialog.reject)
        done = QPushButton("Done")
        done.setStyleSheet(PRIMARY_BTN_STYLE_TEAL)
        done.clicked.connect(dialog.accept)
        btns.addWidget(cancel)
        btns.addWidget(done)
        form.addLayout(btns)

        if dialog.exec_() and self._recipe.geometry is not None:
            self._recipe.geometry.config = {
                "mode": "iso" if iso_radio.isChecked() else "aniso",
                "target_um": float(tgt.value()),
                "factors": (float(axis_spins["Z"].value()),
                            float(axis_spins["Y"].value()),
                            float(axis_spins["X"].value())),
                "method": self._method_label_to_key(method.currentText()),
            }
            self._refresh_recipe_view()
            self._notify_recipe_edited()

    # ---- Preview (Brick 2b — live compute) / Apply (Brick 3 — still a stub) ----

    def _on_preview_clicked(self) -> None:
        """Preview delegates to the controller, which computes the recipe on a
        test crop off the main thread + draws the before/after histogram (Brick
        2b).  Falls back to the flatten LOG only on a bare widget with no
        controller (keeps the widget usable in isolation)."""
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "on_preview_recipe", None)
        if callable(cb):
            cb()
        else:
            self._log_flattened_recipe("PREVIEW")

    def _on_apply_clicked(self) -> None:
        """"Process whole volume" → run the recipe over the WHOLE volume on a
        worker → in-memory uint16 + a WYSIWYG ⟨processed⟩ render (Brick 1b-1).
        The button doubles as Cancel while a run is in flight; the SINGLE dispatch
        (Process-vs-Cancel on `_processing`) lives on the controller
        (`on_process_or_cancel_clicked`) so it can't drift from a second copy here.
        Falls back to the flatten LOG only on a bare widget with no controller."""
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "on_process_or_cancel_clicked", None)
        if callable(cb):
            cb()
        else:
            self._log_flattened_recipe("APPLY")

    # ---- Preview REGION controls (Preview v2 Piece 2) — delegate to controller --

    def _on_place_preview_clicked(self) -> None:
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "place_preview_region", None)
        if callable(cb):
            cb()

    def _on_recenter_preview_clicked(self) -> None:
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "recenter_preview_region", None)
        if callable(cb):
            cb()

    def _on_preview_size_clicked(self, frac) -> None:
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "set_preview_size_fraction", None)
        if callable(cb):
            cb(frac)

    def _on_exit_preview_clicked(self) -> None:
        """"Exit preview" — the single graceful teardown of the ② preview region
        (drops the box + any preview layers + collapses these controls)."""
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "_teardown_preview", None)
        if callable(cb):
            cb()

    # -- X/Y/Z position + Z-depth sliders (revealed after Place) --------------
    #    A live drag MOVES / RESIZES the box + clears the (now stale) preview; the
    #    user re-clicks Preview to recompute (we never recompute per tick).  Each
    #    handler is a no-op during a programmatic seed (`_syncing_preview_sliders`).

    def _on_preview_x_changed(self, value) -> None:
        self._update_preview_slider_value_labels()
        if getattr(self, "_syncing_preview_sliders", False):
            return
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "set_preview_center", None)
        if callable(cb):
            cb(cx=int(value))

    def _on_preview_y_changed(self, value) -> None:
        self._update_preview_slider_value_labels()
        if getattr(self, "_syncing_preview_sliders", False):
            return
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "set_preview_center", None)
        if callable(cb):
            cb(cy=int(value))

    def _on_preview_z_changed(self, value) -> None:
        self._update_preview_slider_value_labels()
        if getattr(self, "_syncing_preview_sliders", False):
            return
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "set_preview_center", None)
        if callable(cb):
            cb(cz=int(value))

    def _on_preview_zdepth_changed(self, value) -> None:
        self._update_preview_slider_value_labels()
        # Z-position is inert while the box spans all Z — grey it off the depth.
        self._refresh_preview_zpos_enabled()
        if getattr(self, "_syncing_preview_sliders", False):
            return
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "set_preview_z_depth", None)
        if callable(cb):
            cb(int(value))

    def _update_preview_slider_value_labels(self) -> None:
        """Mirror each slider's current value AND its axis max into its small
        value label — e.g. ``128 / 512``.  The max is read cheaply off the
        slider's own range (seeded by `_sync_preview_sliders_from_box` from the
        full (Z, Y, X) shape), so the readout doubles as a slice-index scale."""
        for slider, label in (
                (getattr(self, "preview_x_slider", None),
                 getattr(self, "preview_x_value_label", None)),
                (getattr(self, "preview_y_slider", None),
                 getattr(self, "preview_y_value_label", None)),
                (getattr(self, "preview_z_slider", None),
                 getattr(self, "preview_z_value_label", None)),
                (getattr(self, "preview_zdepth_slider", None),
                 getattr(self, "preview_zdepth_value_label", None))):
            if slider is not None and label is not None:
                label.setText(f"{int(slider.value())} / {int(slider.maximum())}")

    def _refresh_preview_zpos_enabled(self) -> None:
        """Grey the Z-position slider while the box spans the FULL Z depth (it is
        inert there); enable it the moment Z-depth < full.  Wired off the depth
        slider's value (both live drags and the programmatic seed)."""
        shape = getattr(self, "_preview_full_shape", None)
        zslider = getattr(self, "preview_z_slider", None)
        dslider = getattr(self, "preview_zdepth_slider", None)
        if shape is None or zslider is None or dslider is None:
            return
        full_depth = int(shape[0])
        zslider.setEnabled(int(dslider.value()) < full_depth)

    def set_preview_region_active(self, active: bool, full_shape=None) -> None:
        """The controller reveals (or collapses) the grouped preview-region editor.

        On reveal it also seeds the slider ranges + values from the CURRENT
        preview box (read off the controller) and *full_shape* (a (Z, Y, X)
        tuple), and greys Z-position while the box spans all Z.  This is separate
        from `set_preview_region_ready`, which owns the Preview-button enable."""
        active = bool(active)
        grp = getattr(self, "preview_region_group", None)
        if grp is not None:
            grp.setVisible(active)
        # The card is the editor's permanent home: COLLAPSED it shows only the
        # "Place preview region" CTA; EXPANDED the editor body replaces it, so
        # the CTA hides while a region is placed (the card FRAME stays visible
        # either way).
        place = getattr(self, "place_preview_button", None)
        if place is not None:
            place.setVisible(not active)
        if active and full_shape is not None:
            self._preview_full_shape = tuple(int(s) for s in full_shape)
            self._sync_preview_sliders_from_box()
        elif not active:
            self._preview_full_shape = None

    def _sync_preview_sliders_from_box(self) -> None:
        """Seed the X/Y/Z + Z-depth sliders from the controller's preview box +
        the stored full (Z, Y, X) shape, WITHOUT firing the live-move callbacks
        (guarded by `_syncing_preview_sliders`)."""
        shape = getattr(self, "_preview_full_shape", None)
        ctrl = getattr(self, "controller", None)
        box = getattr(ctrl, "_preview_box", None)
        if shape is None or box is None:
            return
        Z, Y, X = (int(s) for s in shape)
        self._syncing_preview_sliders = True
        try:
            # Ranges first, so the subsequent `setValue` clamps into range.
            self.preview_x_slider.setRange(0, max(0, X - 1))
            self.preview_y_slider.setRange(0, max(0, Y - 1))
            self.preview_z_slider.setRange(0, max(0, Z - 1))
            self.preview_zdepth_slider.setRange(1, max(1, Z))
            self.preview_x_slider.setValue(int(box.cx))
            self.preview_y_slider.setValue(int(box.cy))
            self.preview_z_slider.setValue(int(box.cz))
            self.preview_zdepth_slider.setValue(int(box.sz))
        finally:
            self._syncing_preview_sliders = False
        self._update_preview_slider_value_labels()
        self._refresh_preview_zpos_enabled()

    def set_preview_region_ready(self, ready: bool) -> None:
        """The controller reports whether a preview REGION box has been placed.
        Preview stays disabled (with a hint) until one exists (Preview v2 Piece
        2)."""
        self._preview_region_ready = bool(ready)
        self._refresh_preview_button_state()

    def _refresh_preview_button_state(self) -> None:
        """Enable Preview only when there is a recipe (channels) AND a placed
        preview region; otherwise disable it with the reason as a tooltip."""
        btn = getattr(self, "test_region_button", None)
        if btn is None:
            return
        has_channels = bool(getattr(self, "_recipe_channel_names", None))
        ready = bool(getattr(self, "_preview_region_ready", False))
        btn.setEnabled(has_channels and ready)
        if not has_channels:
            tip = "Load a source with channels first"
        elif not ready:
            tip = "Place a preview region first"
        else:
            tip = ("Preview the recipe on the placed region and draw the "
                   "before → after intensity histogram (approximate — CLAHE / "
                   "bleach / geometry differ on the whole volume)")
        btn.setToolTip(tip)

    def set_preview_status(self, message: str, ok: bool = True,
                           advisories=None) -> None:
        """Themed ② status line (not a traceback) for Preview results / errors.
        Mirrors `set_export_status`'s green/amber convention.

        *advisories* are per-run caveats (this region cannot show CLAHE / bleach
        honestly; the whole-volume run will refuse this recipe on this source).
        They are NOT failures: a preview that rendered and drew its histogram
        succeeded.  They used to be appended by re-calling this with
        ``ok=False``, which repainted the whole line amber so a working preview
        looked broken.  Each rides its own amber "⚠ " line UNDER the green
        success line instead.
        """
        color = "#6abf69" if ok else "#d4a843"
        notes = [str(a) for a in (advisories or []) if str(a).strip()]
        if notes:
            # `escape(quote=False)` keeps apostrophes and quotes literal so the
            # sentences read as written; only markup characters are neutralised.
            body = "".join(
                f'<div style="color:#d4a843;">⚠ {escape(n, quote=False)}</div>'
                for n in notes)
            self.preprocess_status_label.setText(
                f'<div style="color:{color};">{escape(message, quote=False)}'
                f'</div>{body}')
        else:
            self.preprocess_status_label.setText(message)
        self.preprocess_status_label.setStyleSheet(
            f"font-size: 10px; color: {color};")

    def set_recipe_histograms(self, entries) -> None:
        """Cache the per-channel raw→processed intensity histograms + draw the
        SELECTED channel (Brick 2b, corrected).

        *entries* is the list the controller builds — each a
        :class:`PreviewHistEntry`
        ``(index, name, lut, before_counts, after_counts, bin_edges, stats)``:

        - ``before`` = the RAW crop through the LINEAR half of percentile-normalize
          (``u``), UNCLIPPED — so the pixels normalize clips show as tails past the
          0 / 1 rails (identical before/after ⇔ values truly unchanged; the old
          "normalize-only looks like a no-op" bug is structurally impossible).
        - ``after``  = the ``execute`` output, already in the ``u`` frame.

        Both curves share ONE fixed bin frame (``[-0.1, 1.1]``); the channel
        selector, Log-Y and Raw-DN toggles all switch the VIEW of these cached
        entries with NO recompute.  An EMPTY list re-shows the hint (empty state).
        """
        self._recipe_hist_stale = False
        entries = list(entries or [])
        self._recipe_hist_entries = entries

        if not entries:
            self._clear_recipe_plot()
            self.recipe_histogram_hint.setText(self._recipe_hist_default_hint)
            self.recipe_histogram_hint.setVisible(True)
            self.recipe_hist_channel_bar.setVisible(False)
            self.recipe_hist_logy_toggle.setVisible(False)
            self.recipe_hist_rawaxis_toggle.setVisible(False)
            self.recipe_hist_stat_label.setVisible(False)
            return

        self.recipe_histogram_hint.setVisible(False)
        self.recipe_hist_logy_toggle.setVisible(True)
        self.recipe_hist_rawaxis_toggle.setVisible(True)
        self._rebuild_recipe_channel_bar(entries)
        self.recipe_hist_channel_bar.setVisible(True)
        self._redraw_recipe_histogram()

    def _clear_recipe_plot(self) -> None:
        """Clear the plot + drop the legend (a scene child not wiped by
        ``plot.clear()``) + hide the title.  Shared by the empty state + each
        redraw so a stale legend never keeps dead rows."""
        plot = self._recipe_plot_widget
        plot.clear()
        legend = getattr(self, "_recipe_plot_legend", None)
        if legend is not None:
            try:
                legend.scene().removeItem(legend)
            except Exception:
                pass
            self._recipe_plot_legend = None
        # Reset pyqtgraph's own back-reference: addLegend() hands back the EXISTING
        # (detached) legend unless this is cleared, so it vanishes after redraw 1.
        try:
            plot.plotItem.legend = None
        except Exception:
            pass
        try:
            plot.setTitle(None)
        except Exception:
            pass

    def _recipe_hist_selected_entry(self):
        """The cached entry for the active channel (index clamped), or None."""
        entries = getattr(self, "_recipe_hist_entries", None) or []
        if not entries:
            return None
        idx = int(getattr(self, "_recipe_hist_selected_channel", 0))
        if idx < 0 or idx >= len(entries):
            idx = 0
            self._recipe_hist_selected_channel = 0
        return entries[idx]

    def _rebuild_recipe_channel_bar(self, entries) -> None:
        """Rebuild the exclusive channel "tabs" from *entries*, preserving the
        active index when still valid (else default to channel 0)."""
        layout = self._recipe_hist_channel_layout
        group = self._recipe_hist_channel_group
        for b in list(self._recipe_hist_channel_buttons):
            try:
                group.removeButton(b)
                layout.removeWidget(b)
                b.setParent(None)
                b.deleteLater()
            except Exception:
                pass
        self._recipe_hist_channel_buttons = []

        n = len(entries)
        if self._recipe_hist_selected_channel >= n:
            self._recipe_hist_selected_channel = 0

        for i, entry in enumerate(entries):
            name = getattr(entry, "name", entry[1])
            lut = getattr(entry, "lut", entry[2])
            b = QPushButton(str(name))
            b.setCheckable(True)
            b.setStyleSheet(toggle_btn_style(lut))
            b.setToolTip(
                f"Show the raw→processed histogram for channel “{name}”.")
            b.setChecked(i == self._recipe_hist_selected_channel)
            b.clicked.connect(
                lambda _c=False, idx=i: self._on_recipe_hist_channel_selected(idx))
            group.addButton(b, i)
            layout.addWidget(b, 0)
            self._recipe_hist_channel_buttons.append(b)

    def _on_recipe_hist_channel_selected(self, idx) -> None:
        """A channel tab was clicked — switch the SHOWN channel (INSTANT; reads the
        cached entries, never recomputes).  A re-click of the active tab is a
        no-op (kept checked by the exclusive group)."""
        idx = int(idx)
        if idx == getattr(self, "_recipe_hist_selected_channel", 0):
            if 0 <= idx < len(self._recipe_hist_channel_buttons):
                self._recipe_hist_channel_buttons[idx].setChecked(True)
            return
        self._recipe_hist_selected_channel = idx
        # Keep the tab check-state in sync for a PROGRAMMATIC switch too (a real
        # click already toggled it; setChecked on an exclusive-group button
        # auto-unchecks the siblings).
        if 0 <= idx < len(self._recipe_hist_channel_buttons):
            self._recipe_hist_channel_buttons[idx].setChecked(True)
        self._redraw_recipe_histogram()

    def _redraw_recipe_histogram(self) -> None:
        """Draw the ACTIVE channel's cached before/after curves + clip markers +
        stat read-out, honouring the Log-Y and Raw-DN view toggles.  Pure view —
        no recompute; safe to call on any channel / toggle switch.

        Does NOT clear ``_recipe_hist_stale``: only a fresh Preview result
        (``set_recipe_histograms``) clears stale.  So a channel-tab / Log-Y /
        Raw-DN toggle AFTER a recipe edit re-plots the cached (stale) curves but
        RE-APPLIES the greyed opacity + "re-Preview" hint at the end — the stale
        state persists visibly across view interactions (was silently undone)."""
        self._clear_recipe_plot()
        entry = self._recipe_hist_selected_entry()
        if entry is None:
            return
        plot = self._recipe_plot_widget
        name = getattr(entry, "name", entry[1])
        lut = getattr(entry, "lut", entry[2])
        before = np.asarray(getattr(entry, "before_counts", entry[3]),
                            dtype=np.float64)
        after = np.asarray(getattr(entry, "after_counts", entry[4]),
                           dtype=np.float64)
        edges = np.asarray(getattr(entry, "bin_edges", entry[5]),
                           dtype=np.float64)
        stats = getattr(entry, "stats", entry[6] if len(entry) > 6 else {}) or {}

        plot.setTitle(
            f"Intensity histogram — raw (unclipped) vs processed  ·  {name}",
            color=(200, 200, 205), size="9pt")
        try:
            self._recipe_plot_legend = plot.addLegend(
                offset=(-8, 8), labelTextColor=(200, 200, 205))
        except Exception:
            self._recipe_plot_legend = None

        # Log-Y (default): plot log1p(count) so empty bins render as 0, never -inf.
        log_y = self.recipe_hist_logy_toggle.isChecked()
        y_before = np.log1p(before) if log_y else before
        y_after = np.log1p(after) if log_y else after

        # BEFORE = muted / neutral; AFTER = channel LUT accent.  Both stepped-filled
        # semi-transparent (stepMode → x = bin EDGES, len = y + 1); after on top.
        neutral = pg.mkColor(165, 170, 180)
        before_pen = pg.mkColor(neutral); before_pen.setAlpha(210)
        before_fill = pg.mkColor(neutral); before_fill.setAlpha(55)
        acc = pg.mkColor(lut)
        after_pen = pg.mkColor(acc); after_pen.setAlpha(240)
        after_fill = pg.mkColor(acc); after_fill.setAlpha(95)

        plot.plot(
            edges, y_before, stepMode="center", fillLevel=0,
            brush=pg.mkBrush(before_fill),
            pen=pg.mkPen(before_pen, width=1),
            name="raw (unclipped)")
        plot.plot(
            edges, y_after, stepMode="center", fillLevel=0,
            brush=pg.mkBrush(after_fill),
            pen=pg.mkPen(after_pen, width=2),
            name="processed")

        self._add_recipe_clip_markers(plot)
        self._apply_recipe_hist_view_range()
        self._apply_recipe_hist_x_axis(stats)
        self._recipe_plot_widget.setLabel(
            "left",
            "log(1 + voxels per million voxels)" if log_y
            else "voxels per million voxels")

        self.recipe_hist_stat_label.setText(self._format_recipe_stat(stats))
        self.recipe_hist_stat_label.setVisible(True)

        # A geometry (resample) op changes the voxel COUNT.  Both curves are now
        # DENSITIES (each normalized by its own voxel count — C#7), so the
        # processed curve no longer sits uniformly lower purely because of the
        # re-grid; say what the axis is instead of apologising for it.
        recipe = getattr(self, "_recipe", None)
        if recipe is not None and getattr(recipe, "geometry", None) is not None:
            self.recipe_hist_stat_label.setText(
                self.recipe_hist_stat_label.text()
                + "  ·  resize changes how many voxels there are — both curves "
                "are shown as voxels per million voxels, so they stay "
                "comparable")

        # If a recipe edit greyed this histogram, a view toggle just re-plotted
        # it at full opacity — re-apply the stale visuals so it stays visibly
        # stale until a fresh Preview (`set_recipe_histograms`) clears the flag.
        if getattr(self, "_recipe_hist_stale", False):
            self._apply_recipe_stale_visuals()

    def _apply_recipe_hist_view_range(self) -> None:
        """Frame the plot every redraw: X pinned to the FIXED intensity window
        ``[_recipe_hist_lo, _recipe_hist_hi]`` (a hair of padding), Y auto-ranged
        to the data — so the whole histogram shows on the FIRST Preview and every
        redraw WITHOUT the user clicking pyqtgraph's auto-range "A" button.

        X auto-range is disabled (the frame is fixed by design — the curves share
        one bin frame); Y auto-range stays ON so a tall pile-up, the Log-Y ↔ Raw
        counts swap, and channel switches always fit vertically.  Explicitly
        re-enabling Y auto-range also forces pyqtgraph to resolve the range
        immediately (the "range set before first paint doesn't stick" race that
        left the first Preview zoomed into the default ``[0, 1] × [0, 1]``).
        Never raises (a headless / fake plot)."""
        plot = self._recipe_plot_widget
        lo = float(getattr(self, "_recipe_hist_lo", -0.1))
        hi = float(getattr(self, "_recipe_hist_hi", 1.1))
        try:
            vb = plot.getViewBox()
        except Exception:
            vb = None
        try:
            if vb is not None:
                vb.enableAutoRange(axis="x", enable=False)
            plot.setXRange(lo, hi, padding=0.02)
            if vb is not None:
                vb.enableAutoRange(axis="y", enable=True)
            else:
                plot.enableAutoRange(axis="y")
        except Exception:
            pass

    def _apply_recipe_stale_visuals(self) -> None:
        """Grey the drawn curves + show the "re-Preview" hint / stat text WITHOUT
        touching ``_recipe_hist_stale``.  Shared by ``mark_recipe_preview_stale``
        (which sets the flag) and the tail of ``_redraw_recipe_histogram`` (which
        keeps a stale view stale across channel / Log-Y / Raw-DN toggles)."""
        for item in self._recipe_plot_widget.listDataItems():
            try:
                item.setOpacity(0.30)
            except Exception:
                pass
        self.recipe_histogram_hint.setText("recipe changed — re-Preview")
        self.recipe_histogram_hint.setVisible(True)
        if getattr(self, "recipe_hist_stat_label", None) is not None:
            self.recipe_hist_stat_label.setText(
                "recipe changed — re-Preview to refresh")

    def _add_recipe_clip_markers(self, plot) -> None:
        """Vertical clip lines at ``u = 0`` (p0.1) and ``u = 1`` (p99.5) + faint
        shading of the clipped zones (``u < 0`` and ``u > 1``).  Re-added each
        redraw (``plot.clear()`` drops added items)."""
        lo = float(getattr(self, "_recipe_hist_lo", -0.1))
        hi = float(getattr(self, "_recipe_hist_hi", 1.1))
        shade = pg.mkBrush(120, 120, 130, 40)
        for span in ([lo, 0.0], [1.0, hi]):
            try:
                region = pg.LinearRegionItem(
                    values=span, movable=False, brush=shade)
                region.setZValue(-20)
                for line in region.lines:
                    line.setPen(pg.mkPen(None))
                plot.addItem(region)
            except Exception:
                pass
        line_pen = pg.mkPen(210, 130, 95, 170, width=1, style=Qt.DashLine)
        for xpos, txt in ((0.0, "clip low · p0.1"),
                          (1.0, "clip high · p99.5")):
            try:
                vln = pg.InfiniteLine(
                    pos=xpos, angle=90, pen=line_pen, label=txt,
                    labelOpts={"color": (190, 150, 130), "position": 0.9,
                               "movable": False})
                vln.setZValue(-10)
                plot.addItem(vln)
            except Exception:
                pass

    def _apply_recipe_hist_x_axis(self, stats) -> None:
        """Relabel the x-axis ticks per the Raw-DN toggle — SAME curves, different
        tick labels.  Normalized (default): auto ticks + a ``u`` label.  Raw-DN:
        ticks at ``u`` positions labelled with raw DN (``u·(p_hi−p_lo)+p_lo``)."""
        ax = self._recipe_plot_widget.getAxis("bottom")
        if self.recipe_hist_rawaxis_toggle.isChecked():
            p_lo = float(stats.get("p_lo", 0.0))
            p_hi = float(stats.get("p_hi", 1.0))
            span = (p_hi - p_lo)
            major = [(u, self._fmt_dn(p_lo + u * span))
                     for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
            try:
                ax.setTicks([major, []])
            except Exception:
                pass
            self._recipe_plot_widget.setLabel("bottom", "raw intensity (DN)")
        else:
            try:
                ax.setTicks(None)   # restore auto ticks
            except Exception:
                pass
            self._recipe_plot_widget.setLabel(
                "bottom", "normalized intensity (0 = p0.1, 1 = p99.5 of raw)")

    @staticmethod
    def _fmt_dn(v) -> str:
        """Format a raw DN value compactly (int-ish for large, 2-dp for small)."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "?"
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 1:
            return f"{v:.1f}"
        return f"{v:.2f}"

    def _format_recipe_stat(self, stats) -> str:
        """The factual read-out line — raw range, clip fractions, median shift.
        NO mean / SD / SNR, NO composite score (design: factual only).

        A degenerate crop (``stats["degenerate"]``) has NO percentile spread —
        ``percentile_normalize`` collapses it to all-zeros — so the usual
        ``(p0.1/p99.5)`` range + "0% clipped" line would be an outright lie.
        Render an honest note instead (the histogram falls back to a min-max
        frame purely to avoid a div-by-zero; it is not what normalize does)."""
        if stats.get("degenerate"):
            return ("degenerate intensity range — normalize collapses this "
                    "region to 0 (preview region only)")
        p_lo = self._fmt_dn(stats.get("p_lo", 0.0))
        p_hi = self._fmt_dn(stats.get("p_hi", 0.0))
        lo = float(stats.get("pct_clipped_low", 0.0))
        hi = float(stats.get("pct_clipped_high", 0.0))
        mb = float(stats.get("median_before_u", 0.0))
        ma = float(stats.get("median_after_u", 0.0))
        return (f"raw {p_lo}–{p_hi} (p0.1/p99.5)  ·  "
                f"clipped {lo:.1f}%↓ {hi:.1f}%↑  ·  "
                f"median {mb:.2f}→{ma:.2f}  ·  preview region only")

    def mark_recipe_preview_stale(self) -> None:
        """Grey the shown histogram + show a "recipe changed — re-Preview" hint
        WITHOUT recomputing (the controller calls this on a recipe-card edit).  A
        no-op when no preview histogram is drawn.  The stale flag persists across
        channel / Log-Y / Raw-DN toggles (see ``_redraw_recipe_histogram``) until
        a fresh Preview clears it."""
        if not self._recipe_plot_widget.listDataItems():
            return
        self._recipe_hist_stale = True
        self._apply_recipe_stale_visuals()

    def _log_flattened_recipe(self, verb: str) -> None:
        """Stub (Brick 1b): LOG the flattened recipe per channel — no op math,
        no bake, no extract routing (Brick 2/3)."""
        n = len(self._recipe_channel_names)
        if n == 0:
            self._log("PREPROC-STUB", f"{verb}: no channels loaded")
            return
        for ci in range(n):
            if ci >= len(self._recipe.channels):
                continue
            keys = [s.key for s in _recipe.flatten_for_channel(self._recipe, ci)]
            self._log("PREPROC-STUB",
                      f"{verb} ch{ci} ({self._recipe_channel_names[ci]}): {keys}")
        self._log("PREPROC-STUB", f"{verb}: engine not wired yet (Brick 2/3)")
        self.preprocess_status_label.setText(
            f"— {verb.lower()} logged the recipe (engine not wired yet)")

    # ---- recipe persistence (widget ⇄ controller) ----

    def get_recipe_json(self) -> dict:
        """The current recipe as a JSON-safe dict for persistence.  NEVER None —
        clearing a recipe yields an empty-but-present recipe so `upsert_source`'s
        skip-None does not leave a stale recipe on disk."""
        return _recipe.to_json(self._recipe)

    def set_recipe_from_json(self, raw) -> None:
        """Restore a persisted recipe into the panel.  Loaded verbatim (the
        detected channel count may be unknown at source-select, before load);
        `update_image_layer_list` validates it against the real channel count."""
        self._recipe = _recipe.from_json(raw)
        n = len(self._recipe_channel_names)
        if n:
            self._recipe = _recipe.normalize_recipe(self._recipe, n)
        self._refresh_recipe_view()

    def _notify_recipe_edited(self) -> None:
        """Persist the current recipe against the selected source (debounced,
        controller-side).  Best-effort: a no-op before the controller exists."""
        ctrl = getattr(self, "controller", None)
        cb = getattr(ctrl, "on_recipe_edited", None)
        if callable(cb):
            cb()

    def _update_workspace_status(self, path: str = "") -> None:
        """Update workspace status indicator when path changes."""
        import os
        if not path:
            path = self.output_file_path.text()
        if not path.strip():
            self.workspace_status_label.setText("")
            return
        # Guarded: `isdir` + `listdir` on a disconnected network
        # path can raise or stall, and this runs from the workspace box's
        # `textChanged`, i.e. on the UI thread while the user types.
        try:
            exists = os.path.isdir(path)
        except OSError:
            exists = False
        if exists:
            try:
                entries = os.listdir(path)
            except OSError:
                entries = []
            tiffs = [f for f in entries if f.endswith((".tiff", ".tif"))]
            jsons = [f for f in entries if f.endswith("_metadata.json")]
            if tiffs:
                self.workspace_status_label.setText(
                    f"\u2714  Existing workspace \u2014 {len(tiffs)} patch(es), "
                    f"{len(jsons)} metadata file(s)"
                )
                self.workspace_status_label.setStyleSheet(
                    "font-size: 10px; color: #6abf69;"
                )
            else:
                self.workspace_status_label.setText(
                    "\u2714  Directory exists \u2014 empty (ready for export)"
                )
                self.workspace_status_label.setStyleSheet(
                    "font-size: 10px; color: #5ba4cf;"
                )
        else:
            self.workspace_status_label.setText(
                "\u26a0  Directory does not exist \u2014 will be created on export"
            )
            self.workspace_status_label.setStyleSheet(
                "font-size: 10px; color: #d4a843;"
            )

    # ------------------------------------------------------------------
    # ③ Draw & Extract — the PRIMARY panel
    # ------------------------------------------------------------------

    def _create_draw_extract_panel(self):
        """③ Draw & Extract — channels · draw · extract.

        Scope: shell only, body unrewritten.  The 2-column rebuild
        (channel rows left over the Box List, box editor right) comes later.

        """
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(6, 8, 6, 6)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        # ③ had no subsection header at all; the primary panel should be
        # labelled like its siblings.
        controls_layout.addLayout(self._make_subsection_header(
            "Channels", "step1_draw_channels.md"))

        self.image_layer_label = QLabel(
            "Name, colour, and the order they are stacked."
        )
        self.image_layer_label.setWordWrap(True)
        self.image_layer_label.setStyleSheet(FLOW_SUBTITLE_STYLE)

        # ③ OWNS CHANNEL IDENTITY: include · order · LUT ·
        # name, one ROW per channel.  `image_layer_combo` stays the frozen
        # MultiSelection QListWidget the controller reads
        # (`extract_patches:558`), with each row a card carrying order buttons.
        #
        # The old label read "selection order matters".  Measured behaviour:
        # `selectedItems()` returns SELECTION order, so click order genuinely
        # did affect the stack — but invisibly, with nothing on screen showing
        # it.  `_normalize_channel_selection` re-applies the selection in row
        # order after every change, so the visible row order IS the stack
        # order and the ↑↓ buttons mean what they appear to mean.
        self.image_layer_combo = QListWidget()
        self.image_layer_combo.setSelectionMode(QListWidget.MultiSelection)

        # "no user-facing layer picker in the new design".  Hidden, not
        # deleted — `controller.py:498,528` still reads the combo, and the name is frozen:
        # the controller binds it.  Same pattern as `transpose_button`.
        self.bbox_layer_label = QLabel("Bounding box layer:")
        self.bbox_layer_label.setVisible(False)
        self.bbox_layer_combo = QComboBox()
        self.bbox_layer_combo.setVisible(False)

        # Pre-redesign row.  The design carries the fixed size ON the Fixed button and in
        # Geometry's Size presets, not in its own labelled spinbox.  Frozen names
        # (`start_fixed_drawing` reads the spinbox), so hidden not deleted — the
        # pattern already used for `transpose_button` and `bbox_layer_combo`.
        self.bbox_size_label = QLabel("Fixed ROI size:")
        self.bbox_size_label.setVisible(False)
        self.bbox_size_spinbox = QSpinBox()
        self.bbox_size_spinbox.setRange(1, 4096)
        self.bbox_size_spinbox.setValue(256)
        self.bbox_size_spinbox.setSingleStep(8)
        self.bbox_size_spinbox.setVisible(False)

        self.free_bbox_button = QPushButton("▭ Free")
        self.free_bbox_button.setToolTip(
            "Sticky: drag on the image to draw rectangles. Each one is a patch."
        )
        self.fixed_bbox_button = QPushButton(
            f"⊹ Fixed {self.bbox_size_spinbox.value()}³")
        self.fixed_bbox_button.setToolTip(
            "Click on the image to drop a box at the fixed size. Change the "
            "size with Geometry's presets."
        )
        self.edit_bbox_button = QPushButton("✎ Edit")
        self.edit_bbox_button.setCheckable(True)
        self.edit_bbox_button.setEnabled(False)
        self.edit_bbox_button.setToolTip(
            "Select, move and resize existing boxes — needs the box geometry "
            "model"
        )

        # Undo/redo (M1).  These are in v1 because the redesign's own new
        # gestures — slider-driven geometry, drag-resize, delete — make an undo
        # stack mandatory rather than nice.  The stack itself is.
        self.undo_box_button = QPushButton("⟲ Undo")
        self.undo_box_button.setEnabled(False)
        self.undo_box_button.setToolTip(
            "Undo the last box change — needs the box undo stack"
        )
        self.redo_box_button = QPushButton("⟳ Redo")
        self.redo_box_button.setEnabled(False)
        self.redo_box_button.setToolTip(
            "Redo the last undone box change — needs the box undo stack"
        )

        # ── ③ box tool (Phase 1, EpiCycle-Viz model) ──────────────────────
        # The Free/Fixed 2D-rectangle draw is RETIRED (design D2): boxes are
        # created only via "Add box" (volume centre) or "Click-to-place" (a
        # click in 2D drops one at the cursor voxel).  The retired draw buttons
        # are HIDDEN, not deleted — their names are frozen (`_connect_signals`
        # wires them, `_set_step1_busy`/`_apply_theme` name them literally), the
        # same "hide not delete" pattern used for `transpose_button`.
        for _b in (self.free_bbox_button, self.fixed_bbox_button,
                   self.edit_bbox_button, self.undo_box_button,
                   self.redo_box_button):
            _b.setVisible(False)

        self.box3d_mode_button = QPushButton("📦 Box mode")
        self.box3d_mode_button.setCheckable(True)
        self.box3d_mode_button.setToolTip(
            "Place ROI boxes to extract. Toggle ON to place / edit; toggle OFF "
            "keeps the placed boxes visible + movable (only the drop-guide goes "
            "away). Use napari's own 2D/3D button (bottom-left of the viewer) to "
            "inspect in 3D.")
        self.box_add_button = QPushButton("➕ Add box")
        self.box_add_button.setToolTip(
            "Drop a box at the CURRENT view (navigate/zoom there first), sized "
            "by the S/M/L preset. Full depth by default — narrow Z in Geometry.")
        self.box_delete_button = QPushButton("🗑 Delete box")
        self.box_delete_button.setToolTip("Remove the selected box.")
        self.box_place_check = QCheckBox("⌖ Click-to-place")
        self.box_place_check.setToolTip(
            "In 2D, click a slice to drop a new box at the cursor (same S/M/L "
            "size).")

        # ── Size presets: the STICKY size the next
        # Add / click-to-place uses, as a fraction of the field.  Clicking one
        # also resizes the selected box so the choice is visible immediately.
        self.box_size_preset_buttons = []
        for _label, _frac, _tip in (
                ("S", 0.125, "Small — 1/8 of the field"),
                ("M", 0.25, "Medium — 1/4 of the field"),
                ("L", 0.5, "Large — 1/2 of the field")):
            _b = QPushButton(_label)
            _b.setToolTip(
                f"{_tip}. Sets the size for the next Add box / click-to-place, "
                "and resizes the selected box.")
            _b.clicked.connect(
                lambda _c=False, f=_frac: self.controller.set_box_size_fraction(f))
            self.box_size_preset_buttons.append(_b)

        # The design writes this as `[📦 Extract checked]`, but the mockup's glyphs are
        # stand-ins for icons: this button already gets `mdi6.package-variant`
        # from `_apply_theme`, and shipping both renders two box glyphs.  The
        # draw buttons below DO keep their glyphs — they have no qtawesome icon.
        self.extract_roi = QPushButton("Extract checked")
        self.extract_roi.setToolTip(
            "Extract the ticked, not-yet-captured boxes. The ticked channels "
            "become the patch's channels."
        )

        self.progress_bar = QProgressBar()

        # Drawing mode status.
        self.draw_mode_label = QLabel("Mode: Idle")
        self.draw_mode_label.setStyleSheet(FLOW_SUBTITLE_STYLE)

        controls_layout.addWidget(self.image_layer_label)
        controls_layout.addWidget(self.image_layer_combo, 1)

        # Bulk include toggles over the per-row checkboxes.  Inclusion lives in
        # `self._channel_included`; these set every current name True / False,
        # rebuild the rows so the checkboxes reflect it, and refresh the readout.
        self.channel_select_all_button = QPushButton("Select all")
        self.channel_select_all_button.setToolTip(
            "Include every channel in the extract")
        self.channel_select_all_button.setStyleSheet(SECONDARY_BTN_STYLE)
        self.channel_deselect_all_button = QPushButton("Deselect all")
        self.channel_deselect_all_button.setToolTip(
            "Exclude every channel — then tick just the ones to extract")
        self.channel_deselect_all_button.setStyleSheet(SECONDARY_BTN_STYLE)
        _chan_btn_row = QHBoxLayout()
        _chan_btn_row.setSpacing(4)
        _chan_btn_row.addWidget(self.channel_select_all_button)
        _chan_btn_row.addWidget(self.channel_deselect_all_button)
        _chan_btn_row.addStretch(1)
        controls_layout.addLayout(_chan_btn_row)

        self.channel_empty_label = QLabel(
            "No channels — load a source in Load & Adjust.")
        self.channel_empty_label.setWordWrap(True)
        self.channel_empty_label.setStyleSheet(DIM_LABEL_STYLE)
        controls_layout.addWidget(self.channel_empty_label)

        # The order the extract will ACTUALLY use, read exactly the way
        # `controller.extract_patches:558-559` reads it, so the readout cannot
        # promise an order the extract will not honour.
        self.channel_order_label = QLabel("")
        self.channel_order_label.setWordWrap(True)
        self.channel_order_label.setStyleSheet(DIM_LABEL_STYLE)
        controls_layout.addWidget(self.channel_order_label)
        controls_layout.addWidget(self.bbox_layer_label)
        controls_layout.addWidget(self.bbox_layer_combo)
        controls_layout.addWidget(self.bbox_size_label)
        controls_layout.addWidget(self.bbox_size_spinbox)

        controls_layout.addWidget(self._separator())

        # ── Draw & ROIs — the second left-column subsection.  The design puts
        # Channels at ~25% over Draw+ROIs at ~75%, which the stretch factors on
        # the two lists express.
        controls_layout.addLayout(self._make_subsection_header(
            # SINGLE ampersand: `_make_subsection_header` builds a QLabel, which
            # renders `&&` literally.  Only QPushButton and menu text process
            # mnemonics — that is why the accordion titles need `&&` and these
            # do not.
            "Draw & ROIs", "step1_draw_boxes.md"))

        _ddesc = QLabel(
            "Draw boxes, then extract them. Boxes stay editable until exported "
            "in Review & Library."
        )
        _ddesc.setWordWrap(True)
        _ddesc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        controls_layout.addWidget(_ddesc)

        self.clear_shapes_button = QPushButton("Clear")
        self.clear_shapes_button.setToolTip("Remove all drawn ROI shapes")

        # [▭Free][⊹Fix 256³][✎Edit] then ⟲Undo ⟳Redo.
        draw_row = QHBoxLayout()
        draw_row.setSpacing(4)
        for btn in (self.free_bbox_button, self.fixed_bbox_button,
                    self.edit_bbox_button):
            draw_row.addWidget(btn)
        controls_layout.addLayout(draw_row)

        undo_row = QHBoxLayout()
        undo_row.setSpacing(4)
        for btn in (self.undo_box_button, self.redo_box_button,
                    self.clear_shapes_button):
            undo_row.addWidget(btn)
        controls_layout.addLayout(undo_row)

        # ── ③ box-tool controls (Phase 1) — the create + view affordances that
        # replace the retired 2D draw.  `📦 Box mode` places / edits boxes in the
        # CURRENT view; use napari's own 2D/3D toggle to inspect in 3D (the custom
        # 2D⇄3D button was dropped as redundant, 2026-08-17).
        box_mode_row = QHBoxLayout()
        box_mode_row.setSpacing(4)
        box_mode_row.addWidget(self.box3d_mode_button)
        controls_layout.addLayout(box_mode_row)

        box_create_row = QHBoxLayout()
        box_create_row.setSpacing(4)
        box_create_row.addWidget(self.box_add_button)
        # The S/M/L sticky-size presets sit right beside Add.
        _size_lbl = QLabel("size")
        _size_lbl.setStyleSheet(DIM_LABEL_STYLE)
        box_create_row.addWidget(_size_lbl)
        for _b in self.box_size_preset_buttons:
            _b.setFixedWidth(28)
            box_create_row.addWidget(_b)
        box_create_row.addWidget(self.box_delete_button)
        box_create_row.addWidget(self.box_place_check)
        controls_layout.addLayout(box_create_row)

        _esc = QLabel("Full depth by default — narrow Z in Geometry. "
                      "Drag rows to reorder.")
        # WORD-WRAP (Part B fix 2026-07-27): without it a long single line
        # forces a wide minimumSizeHint on the LEFT column, pinning the ③
        # splitter so it could not be dragged.  Wrapping lets the column shrink
        # to its button rows and the splitter balance freely.  The 2026-07-27
        # polish also TRIMMED the copy (was ~100 chars, wrapped to an awkward
        # 2 lines with dead space) — the "refine later in Review & Library"
        # half was redundant with ③'s subtitle just above.
        _esc.setWordWrap(True)
        _esc.setStyleSheet(DIM_LABEL_STYLE)
        controls_layout.addWidget(_esc)

        controls_layout.addWidget(self.draw_mode_label)
        # v1: vestigial ("Mode: Idle" for a tool whose mode is already obvious
        # from the Box-mode button).  Hidden, not deleted -- `_set_draw_mode`
        # still writes to it.
        self.draw_mode_label.setVisible(False)

        # ③'s Box List — the same component ④ uses, no thumbnails here because
        # the column is narrow (③ shows all boxes, ④ filters to
        # those with pixels and draws them larger).
        self.draw_box_list = BoxListPanel(self, thumb=0, absent_status="new",
                                          row_actions="box",
                                          count_noun="extractable")
        self.draw_box_list.set_empty_text(
            "No boxes yet — turn on 3D box mode, then Add box or Click-to-place.")
        self.box_list = self.draw_box_list.list
        # C1: dimmed per-row Z×Y×X size (painted from Qt.UserRole+8; item text
        # stays the pure patch name).
        self.box_list.setItemDelegate(_BoxDimDelegate(self.box_list))
        controls_layout.addWidget(self.draw_box_list, 3)

        controls_layout.addWidget(self.extract_roi)
        controls_layout.addWidget(self.progress_bar)

        # ②'s `Extract from` mirrored here, because the exclusive accordion
        # means the radio and the button it governs are never co-visible.
        self.extract_source_hint = QLabel("")
        self.extract_source_hint.setWordWrap(True)
        self.extract_source_hint.setStyleSheet(DIM_LABEL_STYLE)
        controls_layout.addWidget(self.extract_source_hint)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(SPLITTER_STYLE)
        controls_layout.setContentsMargins(2, 2, 12, 2)
        splitter.addWidget(controls)
        splitter.addWidget(self._create_box_editor_column())
        # Item E / feedback #4: the box-mode + preset buttons widened the left
        # column and squeezed the geometry editor.  Give the right (editor)
        # column the larger share so the Center/Size sliders + presets fit.
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([430, 570])
        layout.addWidget(splitter, 1)
        return panel

    def _create_box_editor_column(self):
        """③ right column: the box Geometry editor.

        The geometry controls stay disabled until a box is selected: they are
        projections of a `BoxRec` source of truth, driven live by the box tool
        in 3D box mode.  Shown rather than omitted because the design draws them and
        the real widgets are wanted with an honest state.

        **Render / style moved to ④ (2026-07-27):** the colour / thickness /
        opacity controls are now LIVE (they drive the vispy box overlay) and
        live in ④ Review & Library beside the provenance off-ramp — see
        `_create_review_library_panel`.  Removing them from here also un-pins
        this column (Part B), leaving Geometry the room it needs.
        """
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(10, 6, 6, 6)
        layout.setSpacing(6)

        # ── Geometry ──
        geom_header = self._make_subsection_header(
            "Geometry", "step1_draw_geometry.md")
        layout.addLayout(geom_header)

        self.geometry_target_label = QLabel("no box selected")
        self.geometry_target_label.setStyleSheet(DIM_LABEL_STYLE)
        layout.addWidget(self.geometry_target_label)

        _gdesc = QLabel(
            "The selected box, in voxels. Drag on the image or type an exact "
            "value."
        )
        _gdesc.setWordWrap(True)
        _gdesc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(_gdesc)

        # Slider PAIRED with a spinbox per axis (M6): a slider cannot hit an
        # exact voxel, a spinbox cannot be dragged.  `_box_geom_slider_for` maps
        # each geometry spin -> its paired slider so `set_box_geometry_maxima`
        # can lift BOTH ceilings off the loaded source (else a >4096 source
        # clamps + silently crops on a geometry edit).
        self._box_geom_slider_for = {}
        (center_box, self.box_center_x_spin, self.box_center_y_spin,
         self.box_center_z_spin) = self._make_geometry_group(
            "Center", ("X", "Y", "Z"), 0)
        layout.addWidget(center_box)
        (size_box, self.box_size_w_spin, self.box_size_h_spin,
         self.box_size_d_spin) = self._make_geometry_group(
            "Size", ("W", "H", "D"), 256)
        layout.addWidget(size_box)

        # ── Absolute-size presets (v2, feedback #6): 128³/256³/512³ + custom.
        # RESTORED from the retired cube presets, but with SAFE semantics — they
        # set the NEXT box's absolute XY voxel size (sticky) via
        # `set_box_size_voxels` and resize the active box WITHOUT touching its Z
        # (the ³ is nominal: full depth by default, narrow Z in the Size group).
        # `_apply_box_preset` (the old W/H/D writer) is retained for its own
        # tests but is no longer wired to these buttons.
        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        preset_row.addWidget(QLabel("size"))
        self.box_preset_buttons = []
        for size in (128, 256, 512):
            btn = QPushButton(f"{size}³")
            btn.setFixedWidth(48)
            btn.setToolTip(
                f"Set the next box's XY size to {size} voxels (full depth by "
                f"default — narrow Z in the Size group). Resizes the selected "
                f"box too.")
            btn.clicked.connect(
                lambda _c=False, s=size: self.controller.set_box_size_voxels(s))
            self.box_preset_buttons.append(btn)
            preset_row.addWidget(btn)
        # Custom absolute size: a voxel field + Set (same sticky size model).
        self.box_custom_size_spin = QSpinBox()
        self.box_custom_size_spin.setRange(1, _GEOM_MAX)
        self.box_custom_size_spin.setValue(256)
        self.box_custom_size_spin.setFixedWidth(72)
        self.box_custom_size_spin.setToolTip(
            "A custom absolute XY voxel size for the next box.")
        self.box_custom_apply_button = QPushButton("Set")
        self.box_custom_apply_button.setToolTip(
            "Use this custom XY voxel size for the next box (and resize the "
            "selected one).")
        self.box_custom_apply_button.clicked.connect(
            lambda: self.controller.set_box_size_voxels(
                self.box_custom_size_spin.value()))
        preset_row.addWidget(self.box_custom_size_spin)
        preset_row.addWidget(self.box_custom_apply_button)
        self.box_lock_size_check = QCheckBox("lock")
        self.box_lock_size_check.setToolTip(
            "Freeze the SELECTED box — disable its move/resize until you "
            "uncheck.  Per-box; a new box starts unlocked.")
        # Disabled until a box is selected in box mode; the box tool enables it
        # on select/add and reflects the selected box's own lock state.
        self.box_lock_size_check.setEnabled(False)
        # Lambda-deferred (like the S/M/L presets): `self.controller` is attached
        # AFTER this build method, so bind the access at toggle time, not now.
        self.box_lock_size_check.toggled.connect(
            lambda checked: self.controller.set_box_locked(checked))
        preset_row.addWidget(self.box_lock_size_check)
        preset_row.addStretch(1)
        layout.addLayout(preset_row)

        # ── At-edge behavior (Brick 1) ──
        # A live geometry edit that pushes a box past the volume edge either
        # SHRINKS it onto the rendered bounds (default — extraction unchanged,
        # the handle never lags the drawn box) or KEEPS its size and slides it
        # inward (deliberately changes the edge extraction).  Same segmented
        # look as the ② Raw/Processed toggle; lambda-deferred because
        # `self.controller` attaches AFTER this build method.
        edge_row = QHBoxLayout()
        edge_row.setSpacing(0)
        _edge_label = QLabel("At edge:")
        _edge_label.setStyleSheet(DIM_LABEL_STYLE)
        edge_row.addWidget(_edge_label)
        edge_row.addSpacing(6)
        self.box_edge_shrink_button = QPushButton("Shrink")
        self.box_edge_shrink_button.setCheckable(True)
        self.box_edge_shrink_button.setChecked(True)
        self.box_edge_shrink_button.setToolTip(
            "Shrink a box that reaches the volume edge to fit — the extracted "
            "region is unchanged and the handle stays on the drawn box "
            "(default).")
        self.box_edge_keep_button = QPushButton("Keep size")
        self.box_edge_keep_button.setCheckable(True)
        self.box_edge_keep_button.setToolTip(
            "Keep the box's size and slide it inward off the edge — this "
            "changes the extracted region near an edge.")
        self._box_edge_group = QButtonGroup(self)
        self._box_edge_group.setExclusive(True)
        self._box_edge_group.addButton(self.box_edge_shrink_button)
        self._box_edge_group.addButton(self.box_edge_keep_button)
        for _b in (self.box_edge_shrink_button, self.box_edge_keep_button):
            _b.setStyleSheet(toggle_btn_style(_STEP1_ACCENT))
        self.box_edge_shrink_button.clicked.connect(
            lambda: self.controller.set_box_edge_mode("shrink"))
        self.box_edge_keep_button.clicked.connect(
            lambda: self.controller.set_box_edge_mode("keep"))
        edge_row.addWidget(self.box_edge_shrink_button)
        edge_row.addWidget(self.box_edge_keep_button)
        edge_row.addStretch(1)
        layout.addLayout(edge_row)

        self.box_keys_hint = QLabel("keys ←/→ nudge 1 vx · shift 10")
        self.box_keys_hint.setWordWrap(True)   # Part B: don't pin the editor column
        self.box_keys_hint.setStyleSheet(DIM_LABEL_STYLE)
        layout.addWidget(self.box_keys_hint)

        # The geometry spins are DISABLED until a box is selected in 3D box
        # mode; the box tool enables them then (`_set_geometry_enabled`) and
        # loads the SELECTED box's own centre/size into them — so the selected
        # box is always the edit target and editing one box never touches
        # another (per-box independence, bug fix 2026-07-27).  (The Render/style
        # widgets moved to ④ and are wired live — see `_create_review_library_panel`.)
        self._box_geometry_widgets = [
            self.box_center_x_spin, self.box_center_y_spin,
            self.box_center_z_spin, self.box_size_w_spin,
            self.box_size_h_spin, self.box_size_d_spin,
        ]
        for w in self._box_geometry_widgets:
            w.setEnabled(False)
            # APPEND, never replace.  The blanket assignment used to wipe
            # tooltips set moments earlier — the presets lost
            # "Set W, H and D to 256" and lock lost its explanation.
            existing = w.toolTip()
            reason = ("The selected box's centre / size as a BoxRec (voxels) — "
                      "editable in 3D box mode once a box is selected")
            w.setToolTip(f"{existing} — {reason}" if existing else reason)

        layout.addStretch(1)
        return column

    @staticmethod
    def _make_slider(low: int, high: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        return slider

    def _make_geometry_group(self, title: str, axes, default: int):
        """A `Center`/`Size` sub-box: one slider + spinbox row per axis."""
        box = QGroupBox(title)
        grid = QGridLayout(box)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)
        spins = []
        for row, axis in enumerate(axes):
            tag = QLabel(axis)
            tag.setStyleSheet(DIM_LABEL_STYLE)
            tag.setFixedWidth(14)
            # ONE domain for both views.  MEASURED: with the slider at
            # 0..2048 and the spin at 0..100000 the pair terminated but was
            # LOSSY — typing 5000 came back as 2048 — which defeats the stated
            # reason for having both ("a slider cannot hit an exact voxel, a
            # spinbox cannot be dragged").  `_GEOM_MAX` also matches
            # `bbox_size_spinbox`'s ceiling so the two size models agree.
            slider = self._make_slider(0, _GEOM_MAX, default)
            spin = QSpinBox()
            spin.setRange(0, _GEOM_MAX)
            spin.setValue(default)
            spin.setFixedWidth(72)
            # Two views of one value.
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            grid.addWidget(tag, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(spin, row, 2)
            spins.append(spin)
            # Remember the paired slider so the maxima can be lifted off the
            # source shape together with the spin's (guarded — the map may not
            # exist for older construction orders).
            slider_map = getattr(self, "_box_geom_slider_for", None)
            if slider_map is not None:
                slider_map[spin] = slider
        return (box, *spins)

    def set_box_geometry_maxima(self, shape_zyx) -> None:
        """Lift the box geometry spin + slider maxima off the loaded source
        shape ``(Z, Y, X)`` so a source deeper/taller/wider than the 4096
        build-time ceiling is not silently cropped when a geometry value is
        edited: Z -> center_z / size_d (D), Y -> center_y / size_h (H),
        X -> center_x / size_w (W).  The box tool calls this when box mode
        starts.  Called BEFORE the geometry handlers are connected, so lowering
        a ceiling below a stale value cannot fire a spurious box resize."""
        if not shape_zyx or len(shape_zyx) != 3:
            return
        try:
            z, y, x = (int(s) for s in shape_zyx)
        except (TypeError, ValueError):
            return
        for spin, hi in ((self.box_center_z_spin, z), (self.box_size_d_spin, z),
                         (self.box_center_y_spin, y), (self.box_size_h_spin, y),
                         (self.box_center_x_spin, x), (self.box_size_w_spin, x)):
            hi = max(1, int(hi))
            slider = self._box_geom_slider_for.get(spin)
            # Raise the slider first so the spin's cross-connected setValue can
            # reach the new ceiling instead of being clamped back by the slider.
            if slider is not None:
                slider.setMaximum(hi)
            spin.setMaximum(hi)

    def _apply_box_preset(self, size: int) -> None:
        """RETIRED cube preset (128/256/512) — sets W/H/D to a cube and the
        fixed-draw size, unless size is locked.

        The buttons that call this are now hidden (`setVisible(False)`): they
        write D (Z), which narrows the box's depth now that Z is editable, and
        duplicate the S/M/L fraction presets.  The method is retained for the
        frozen-name contract + its tests; the S/M/L presets
        (`set_box_size_fraction`) are the live size model.
        """
        if self.box_lock_size_check.isChecked():
            self._log("STEP1-BOX", f"preset {size}³ ignored — size is locked")
            return
        for spin in (self.box_size_w_spin, self.box_size_h_spin,
                     self.box_size_d_spin):
            spin.setValue(size)
        self.set_fixed_box_size(size)

    def set_fixed_box_size(self, size: int) -> None:
        """Point `bbox_size_spinbox` — what the draw path reads — at `size`.

        The button's LABEL is part of the contract: it is built once from the
        construction-time value, so leaving it stale would show `256³` while
        drawing a 512 box.
        """
        spin = self.bbox_size_spinbox
        size = max(spin.minimum(), min(int(size), spin.maximum()))
        spin.setValue(size)
        self.fixed_bbox_button.setText(f"⊹ Fixed {size}³")
        self._log("STEP1-BOX", f"fixed draw size = {size}³")

    # ------------------------------------------------------------------
    # ④ Review & Library
    # ------------------------------------------------------------------

    def _create_review_library_panel(self):
        """④ Review & Library — patch list · depth plot · export.

        Order: header + description → search/sort → master/detail
        → separator → bulk Export (primary) → progress → status → separator →
        Provenance figure as an off-ramp.  Export at the BOTTOM because
        Z-truncate lives in the detail pane; on top it would mean setting a
        Z-range in the middle of the panel and scrolling *up* to commit it.

        **ONE export control (c #14), and `save_reviewed_roi` carries it.**
        Not because of its old label — `_on_export_clicked` used to export only
        `currentItem()` — but because it is the name the SHELL logs "ROI export
        triggered" on (`workflow_shell.py:1418`), so retiring it would silently
        stop the shell's history recording exports.  Its handler was rewritten
        to read the tick column; `export_all_ready_roi` is retained-but-retired
        (constructed, styled, `setVisible(False)`) because `_apply_theme`
        dereferences it unguarded and `_set_step1_busy` names it literally.

        *(An earlier revision of this docstring described the opposite — the two
        buttons "swapped" so this method needed zero controller edits.  That was
        the pre-amendment design and it read as an instruction to revert #14,
        sitting directly above the code implementing it.)*

        The depth plot is NOT rewritten — it already ships
        per-channel intensity-vs-Z curves, a draggable `SpanSelector`, shaded
        head/tail and snap-to-signal at `controller.py:770-904`.
        """
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(6, 8, 6, 6)
        outer.setSpacing(6)

        # ── Header + description + search/sort span the FULL panel (the
        # master/detail split line falls after the search row).
        lib_header = self._make_subsection_header(
            "Patch library", "step1_review_library.md")
        outer.addLayout(lib_header)

        self.extracted_roi_label = QLabel(
            "Every patch extracted from this workspace, saved on disk. Select "
            "one to see its depth profile, trim it, or open the capture view."
        )
        self.extracted_roi_label.setWordWrap(True)
        self.extracted_roi_label.setStyleSheet(FLOW_SUBTITLE_STYLE)
        outer.addWidget(self.extracted_roi_label)

        find_row = QHBoxLayout()
        find_row.setSpacing(6)
        self.patch_search_edit = QLineEdit()
        self.patch_search_edit.setPlaceholderText("Search…")
        self.patch_search_edit.setClearButtonEnabled(True)
        find_row.addWidget(self.patch_search_edit, 1)
        self.patch_sort_combo = QComboBox()
        # Only the orders the data can actually honour — see _sort_patch_list.
        self.patch_sort_combo.addItems(["name A–Z", "name Z–A"])
        find_row.addWidget(self.patch_sort_combo, 0)
        outer.addLayout(find_row)

        # ── Master / detail (draggable splitter, mirroring ②/③) ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(6)

        # ④'s Box List — THE SAME COMPONENT as ③'s, just with
        # larger thumbnails.  `extracted_roi_list` remains the inner
        # QListWidget, keeping the frozen name AND type the shell depends on.
        self.library_box_list = BoxListPanel(self, thumb=64,
                                            absent_status="captured",
                                            row_actions="patch",
                                            count_noun="patches")
        # ④ has no per-row tick column any more (F1 writes Extract straight to
        # disk), so a "select all" master is meaningless — hide it.  ③'s
        # `draw_box_list.select_all_check` stays visible + functional.
        self.library_box_list.select_all_check.setVisible(False)
        self.library_box_list.set_empty_text(
            "No patches yet — draw boxes in Draw & Extract, then Extract.")
        self.extracted_roi_list = self.library_box_list.list

        # "Show locations" + the Snapshot split button go INLINE in the list's
        # header row (left of the "N patch(es)" count) — ONE clean row, not a
        # separate strip above the list.
        self.show_locations_check = QCheckBox("Show locations")
        self.show_locations_check.setTristate(True)
        self.show_locations_check.setToolTip(
            "Draw every saved patch's bounding box in the source volume so you "
            "can see where each patch came from.  Use a row's eye to show just "
            "one.")
        self.library_box_list.set_header_left(
            self.show_locations_check, self._build_snapshot_button(),
            self._build_scalebar_button())

        left_layout.addWidget(self.library_box_list, 1)

        # ── Detail pane: plot → Z-truncate → details → per-patch actions,
        # the same grammar as the panel itself, nested.
        detail = QWidget()
        det = QVBoxLayout(detail)
        # Left margin mirrors the left column's 10px right margin so the detail
        # controls get breathing room from the splitter handle (not flush).
        det.setContentsMargins(10, 0, 0, 0)
        det.setSpacing(6)

        self.selected_roi_label = QLabel("Selected ROI: -")
        self.selected_roi_label.setWordWrap(True)   # long patch names wrap
        self.selected_roi_label.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #dddddd;")
        det.addWidget(self.selected_roi_label)

        _pdesc = QLabel("signal at each depth")
        _pdesc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        det.addWidget(_pdesc)

        self.plot_area = QWidget()
        # BOUNDED, not Expanding.  The plot is one item in a top-to-bottom
        # detail pane (plot → Z-truncate → details → actions).  Left Expanding
        # with the only stretch factor it absorbed every spare pixel and pushed
        # the controls that act on it to the bottom of a tall dock — which is
        # why changing `minimumHeight` alone changed nothing on screen.
        self.plot_area.setMinimumHeight(160)
        self.plot_area.setMaximumHeight(240)
        self.plot_area.setMinimumWidth(350)
        self.plot_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.plot_area.setStyleSheet(CANVAS_AREA_STYLE)
        self.plot_area.setLayout(QVBoxLayout())
        det.addWidget(self.plot_area, 0)

        det.addLayout(self._make_subsection_header(
            "Z-truncate", "step1_review_zrange.md"))

        # Numeric start/end beside the draggable band — a span cannot hit an
        # exact slice.  These WRITE the range; leaving them enabled and
        # inert above a live `Set Range`, as the previous attempt did, silently
        # commits whatever the span last held instead of what was typed.
        z_row = QHBoxLayout()
        z_row.setSpacing(6)
        self.ztrunc_start_spin = QSpinBox()
        self.ztrunc_start_spin.setRange(0, 100000)
        self.ztrunc_end_spin = QSpinBox()
        self.ztrunc_end_spin.setRange(0, 100000)
        z_row.addWidget(self.ztrunc_start_spin, 1)
        z_row.addWidget(QLabel("to"), 0)
        z_row.addWidget(self.ztrunc_end_spin, 1)
        self.auto_select_zrange_button = QPushButton("snap")
        self.auto_select_zrange_button.setToolTip(
            "Snap the range to the slices that carry signal")
        z_row.addWidget(self.auto_select_zrange_button, 0)
        det.addLayout(z_row)
        _zhint = QLabel("drag the shaded edges on the plot")
        _zhint.setStyleSheet(DIM_LABEL_STYLE)
        det.addWidget(_zhint)

        # The depth profile auto-draws when a patch is selected (UX
        # consult), so the old "Plot Intensity" button was redundant and removed.
        # Attr name is FROZEN (the controller wiring test depends on it); the LABEL
        # is "Trim" — a destructive crop-in-place of the patch, not the old inert
        # "record the range" no-op.
        self.set_select_zrange_button = QPushButton("Trim")
        self.set_select_zrange_button.setToolTip(
            "Crop this saved patch to the selected Z range — overwrites the "
            "patch on disk (can't be undone)")
        z_actions = QHBoxLayout()
        z_actions.setSpacing(6)
        z_actions.addWidget(self.set_select_zrange_button)
        det.addLayout(z_actions)

        det.addWidget(self._separator())

        _dhdr = QLabel("details")
        _dhdr.setStyleSheet(DIM_LABEL_STYLE)
        det.addWidget(_dhdr)

        self.selected_zrange_label = QLabel("Selected Z-range: -")
        self.selected_zrange_label.setStyleSheet(DIM_LABEL_STYLE)
        det.addWidget(self.selected_zrange_label)

        # Key/value rows, as the artifact draws them: size · from · status.
        kv = QGridLayout()
        kv.setContentsMargins(0, 0, 0, 0)
        kv.setHorizontalSpacing(10)
        kv.setVerticalSpacing(2)
        self.patch_detail_values = {}
        for r, key in enumerate(("size", "from", "status")):
            k = QLabel(key)
            k.setStyleSheet(DIM_LABEL_STYLE)
            v = QLabel("—")
            v.setStyleSheet("font-size: 10px; color: #dddddd;")
            v.setWordWrap(True)
            kv.addWidget(k, r, 0)
            kv.addWidget(v, r, 1)
            self.patch_detail_values[key] = v
        det.addLayout(kv)
        # Frozen-ish single-line form kept for any caller that still writes it.
        self.patch_details_label = QLabel("— select a patch")
        self.patch_details_label.setVisible(False)
        det.addWidget(self.patch_details_label)

        # No detail-pane hint here: patches are written to disk at Extract (there
        # is no tick column and both export buttons are retired/hidden), so the
        # old "tick rows to choose what exports" copy was doubly false.  A patch
        # is deleted with its row 🗑 and trimmed with the Trim button above.

        # NOT frozen and NOT connected — an earlier comment here claimed both,
        # and neither is true (it appears in no frozen-names list and nothing in the
        # repo connects it).  Kept only as an inert placeholder for the discard
        # that the redesign relocates into ④'s row action cluster; it will be
        # DELETED when that cluster becomes live, not wired up here.
        self.discard_patch_button = QPushButton("Discard")
        self.discard_patch_button.setVisible(False)
        det.addWidget(self.discard_patch_button)

        # Spare height collects here rather than stretching the plot.
        det.addStretch(1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(SPLITTER_STYLE)
        splitter.addWidget(left)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([300, 640])
        outer.addWidget(splitter, 1)

        # Retained-but-retired (F1): ③ Extract now writes each patch to disk
        # immediately, so the old two-step "Export checked → Library" control is
        # hidden.  Kept alive (not deleted) because `_set_step1_busy`'s literal
        # widget list + the `save_reviewed_roi.clicked` connect + the SHELL's
        # "ROI export triggered" log (`workflow_shell.py:1418`) still reference
        # the name; mirrors how `export_all_ready_roi` is retired just below.  Both
        # are `setVisible(False)` so they occupy no space at the ④ bottom.
        self.save_reviewed_roi = QPushButton("Export checked → Library")
        self.save_reviewed_roi.setToolTip(
            "Write every ticked patch to the workspace. Names are allocated "
            "here, at export, so a patch's number depends on when it is "
            "committed."
        )
        self.save_reviewed_roi.setVisible(False)
        outer.addWidget(self.save_reviewed_roi)

        # Retained-but-retired: kept alive for `_set_step1_busy`'s literal list
        # and the `hasattr` connect at `controller.py:88`, no longer visible.
        self.export_all_ready_roi = QPushButton("Export All Ready ROIs")
        self.export_all_ready_roi.setVisible(False)
        outer.addWidget(self.export_all_ready_roi)

        # ④ action feedback lives together at the bottom of the RIGHT (detail)
        # column: the result text (green success / red failure via
        # `_set_export_status`) above a slim progress bar that scans during a
        # depth-profile load / capture launch / extract.  The old full-width
        # bottom strip (separator + status line) is gone.
        self.export_status_label = QLabel("—")
        self.export_status_label.setWordWrap(True)
        self.export_status_label.setStyleSheet(DIM_LABEL_STYLE)
        det.addWidget(self.export_status_label)

        self.export_progress_bar = QProgressBar()
        self.export_progress_bar.setRange(0, 1)
        self.export_progress_bar.setValue(0)
        self.export_progress_bar.setMaximumHeight(8)
        self.export_progress_bar.setTextVisible(False)
        det.addWidget(self.export_progress_bar)

        # The former "Box appearance" off-ramp (global colour / thickness /
        # opacity + the hidden provenance-figure knobs) was AXED in the ④ P2
        # rework (Brick 1): box styling becomes per-patch (Brick 2), and the
        # location toggle + Snapshot now live in the count row above.  ③'s box
        # tool keeps its own built-in default style.
        return panel

    # ------------------------------------------------------------------
    # ④ helpers
    # ------------------------------------------------------------------

    def _filter_patch_list(self, term: str = "") -> None:
        """Search over ④'s list.

        Widget-side only.  The "live, empty and honest" rule covers
        widgets with **no backend**; ④'s list is populated today, so a search
        box that ignored typing would be a broken control, not an honest one.
        """
        needle = (term or self.patch_search_edit.text()).strip().lower()
        for i in range(self.extracted_roi_list.count()):
            item = self.extracted_roi_list.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())
        # The count says "N of M shown" — filter-dependent, so refresh it here.
        self.library_box_list.refresh()

    def _sort_patch_list(self, *_args) -> None:
        """Sort ④'s list.

        MEASURED bug, fixed: with the old three-item combo
        (`newest first / name A–Z / name Z–A`) index 0 fell into the ascending
        branch, so "newest first" silently sorted alphabetically and A–Z / Z–A
        were **inverted** (`descending = index == 1` against A–Z at index 1).
        The combo now offers only the two orders the data supports — nothing
        stores a creation date, so "newest first" was unimplementable, and
        sorting by name destroys the insertion order it would have to fall back
        on.  `date added` is part of ④'s row grammar and lands with it.
        """
        descending = self.patch_sort_combo.currentIndex() == 1
        self.extracted_roi_list.sortItems(
            Qt.DescendingOrder if descending else Qt.AscendingOrder)
        self.library_box_list.refresh()
        self._filter_patch_list()

    def _on_ztrunc_spin_changed(self, *_args) -> None:
        """Type an exact range and have `Set Range` honour it.

        `_on_set_zrange_clicked` commits `controller.selected_z_range`, which is
        otherwise written only by the matplotlib span's `onselect`
        (`controller.py:847-850`).  Without this, typing 12/62 and pressing Set
        Range stores whatever the span last held — wrong data, silently, into
        the frozen sidecar.
        """
        lo = int(self.ztrunc_start_spin.value())
        hi = int(self.ztrunc_end_spin.value())
        if hi < lo:
            return
        self.controller.selected_z_range = (lo, hi)
        self.selected_zrange_label.setText(f"Selected Z-range: {lo} to {hi}")

    def _refresh_patch_details(self, *_args) -> None:
        """Fill the details block for the selected patch.

        Reads the controller's existing `extracted_patches` — a read, not a
        logic change.
        """
        item = self.extracted_roi_list.currentItem()
        values = getattr(self, "patch_detail_values", {})
        if item is None:
            for v in values.values():
                v.setText("—")
            self.patch_details_label.setText("— select a patch")
            return
        name = item.text()
        detail = self.library_box_list._detail_of(name)
        # Post-F1 every patch is written to disk AT Extract — there is no separate
        # export step and ④ has no tick/lock UI, so the old "in memory until
        # exported" / "locked" notes were false.  Both statuses are on disk.
        note = "saved on disk"
        source = self.display_name_edit.text().strip() or "—"
        if "size" in values:
            values["size"].setText(detail or "—")
            values["from"].setText(source)
            values["status"].setText(note)
        self.patch_details_label.setText(f"{detail} · {note}")

    # ------------------------------------------------------------------
    # ⑤ Diagnostics (off-flow)
    # ------------------------------------------------------------------

    def _create_diagnostics_panel(self):
        """⑤ Diagnostics — a read-only log pane, at parity with Steps 2/3/4.

        `log_output` matches Step 3's `_create_diagnostics_panel`
        (`fusion_reconstruction/ui.py:3213`) so the family reads the same.  This
        is where Step 1's messages surface instead of stdout, which a user
        running napari from a shortcut never sees.  Step 3's Diagnostics carries
        no `?` button; Step 1 matches.
        """
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # PARITY WITH S2/S3/S4.  Steps 3 and 4 are both
        # `System` → separator → `Log Output`, and Step 2 uses the same
        # `Log Output` heading.  An earlier pass renamed this "Console" from the
        # design bench's wording — but the bench is a mockup and the shipped
        # steps are the family, and the shipped steps are the authority.
        _hdr = QLabel("System")
        _hdr.setStyleSheet(SECTION_HEADER_STYLE)
        lay.addWidget(_hdr)
        # Steps 3/4 show GPU here.  Step 1 has no GPU operation, so it shows the
        # number its own guard actually decides on: available memory.
        self.system_status_label = QLabel("Memory: checking…")
        self.system_status_label.setStyleSheet(MONO_LABEL_STYLE)
        lay.addWidget(self.system_status_label)
        self._refresh_system_status()

        lay.addWidget(self._separator())

        _hdr = QLabel("Log Output")
        _hdr.setStyleSheet(SECTION_HEADER_STYLE)
        lay.addWidget(_hdr)

        _desc = QLabel(
            "Source loading, extraction and export events, plus errors that "
            "would otherwise only reach the console."
        )
        _desc.setWordWrap(True)
        _desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        lay.addWidget(_desc)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(LOG_OUTPUT_STYLE)
        self.log_output.setMinimumHeight(220)
        lay.addWidget(self.log_output, 1)

        return panel

    def _refresh_system_status(self) -> None:
        """The ⑤ System line — parity with Steps 3/4's GPU readout.

        Step 1's equivalent fact is available memory, because that is what the
        load guard decides on: showing it means the number behind a refused load
        is visible rather than only in the guard's message.
        """
        try:
            import psutil
            from .logic import format_bytes
            vm = psutil.virtual_memory()
            self.system_status_label.setText(
                f"Memory: {format_bytes(vm.available)} free of "
                f"{format_bytes(vm.total)}")
        except Exception as exc:
            self.system_status_label.setText(f"Memory: unavailable ({exc})")

    # ------------------------------------------------------------------
    # Progressive default-open
    # ------------------------------------------------------------------

    def _mark_section_chosen(self, *_args) -> None:
        self._section_chosen_by_user = True

    def apply_progressive_default_open(self) -> None:
        """Open ① when nothing is set up, ③ once there is something to draw on.

        A constant default would be wrong: ③ is the primary panel in the
        steady state, but on a first visit — no workspace, no source — it has
        nothing to act on.  Evaluated on step-enter and again after a successful
        Load.  A manual section click wins for the rest of the visit.
        """
        if self._section_chosen_by_user:
            return
        has_workspace = bool(self.output_file_path.text().strip())
        has_pixels = self.image_layer_combo.count() > 0
        target = 2 if (has_workspace and has_pixels) else 0
        if self.accordion._current != target:
            self.accordion.setCurrentIndex(target)

    def browse_output_file(self):
        base_dir = QFileDialog.getExistingDirectory(
            self, "Select Workspace Folder", ""
        )
        if not base_dir:
            return
        # The controller owns the whole change (B5).  It closes the previous
        # project — source layers, overlays, identity, the (2) Sources list —
        # REFUSES while a Step-1 job owns the current folder (returning False),
        # and on an allowed change writes this box itself and then publishes to
        # the shell.  The box must move BEFORE the publish, because the shell
        # recomputes divergence by reading it; doing it here, after the call,
        # made Step 1 look diverged from itself (lifecycle audit).
        #
        # The workspace IS the chosen folder — everything Step 1 writes goes
        # under artifacts/step1/ automatically, so there is no subfolder prompt.
        # Browse is also the only door that publishes, never the `textChanged`
        # chain — that fires on adopt and on the start-up pre-fill too.
        handler = getattr(self.controller, "on_workspace_browsed", None)
        if callable(handler):
            handler(base_dir)      # False = a job is running; box left alone
        else:                      # no controller handler (tests / bare UI)
            self.output_file_path.setText(base_dir)

    def _connect_signals(self):
        self.free_bbox_button.clicked.connect(self.start_free_drawing)
        self.fixed_bbox_button.clicked.connect(self.start_fixed_drawing)
        # ③ Extract runs on a worker (F1); the button doubles as Cancel while a
        # run is in flight (`on_extract_or_cancel_clicked` dispatches).
        self.extract_roi.clicked.connect(
            self.controller.on_extract_or_cancel_clicked)
        self.clear_shapes_button.clicked.connect(self.controller.clear_shapes)
        # ── ③ box tool (Phase 1).  The box tool owns its own viewer wiring
        # (Shapes layer, click-to-place callback, geometry-spin + list-row
        # connections); the widget only routes these buttons to the controller.
        self.box3d_mode_button.toggled.connect(
            self.controller.on_box_mode_toggled)
        self.box_add_button.clicked.connect(self.controller.add_box)
        self.box_delete_button.clicked.connect(self.controller.delete_box)
        self.box_place_check.toggled.connect(self.controller.on_place_toggled)
        # ④ Render/style: the global "Box appearance" section was axed (Brick 1);
        # per-patch styling arrives in Brick 2.  ③'s box tool keeps its built-in
        # default style, so there is no global style control to wire here.
        # Data browser signals.
        self.source_browse_button.clicked.connect(self.controller.browse_source_folder)
        self.source_file_list.itemClicked.connect(self.controller.on_source_file_selected)
        self.load_source_button.clicked.connect(
            self.controller.on_load_or_cancel_clicked)
        self.unload_source_button.clicked.connect(self.controller.unload_source)
        # Brick 4a a5: relink the loaded source (the row context menu passes an
        # explicit id; the button means "the one on screen").
        self.locate_source_button.clicked.connect(
            lambda *_a: self._call_locate_source())
        # "Prepare for large-data viewing": the button doubles as Cancel while
        # the background convert runs (same dispatcher shape as Load).  Enabled
        # only while a source file is selected — driven off the source list's
        # selection (the ② source_combo proxy routes through `setCurrentRow`, so
        # the list signals cover it too).
        self.prepare_largedata_button.clicked.connect(
            self.controller.on_prepare_or_cancel_clicked)
        # "Large-data settings…" opens the preset / power-user config dialog.
        self.largedata_settings_button.clicked.connect(
            self.controller.open_largedata_config_dialog)
        self.threed_detail_slider.valueChanged.connect(
            self._on_3d_detail_slider_changed)
        # Progressive-3D toggle: re-display the current store in the chosen mode
        # (controller no-ops when nothing is loaded).
        self.progressive_3d_checkbox.toggled.connect(
            self.controller.on_progressive_3d_toggled)
        # "Zoom depth" combo: LIVE — the controller sets the seam's per-draw depth
        # mode (no re-load).  `_connect_signals` runs AFTER widget construction,
        # so the initial setCurrentIndex(0) above does NOT fire this.
        self.zoom_depth_combo.currentIndexChanged.connect(
            self.controller.on_zoom_depth_changed)
        self.source_file_list.currentItemChanged.connect(
            self._refresh_prepare_largedata_enabled)
        # Keep the Order combo pointed at the SELECTED source (click OR keyboard),
        # so Prepare can orient it without a full Load.  `itemClicked` (above,
        # -> on_source_file_selected) covers mouse only; this covers keyboard nav.
        self.source_file_list.currentItemChanged.connect(
            self.controller._on_source_current_item_changed)
        self.source_file_list.itemSelectionChanged.connect(
            self._refresh_prepare_largedata_enabled)
        self._refresh_prepare_largedata_enabled()
        # A manual section click disables progressive default-open for the
        # visit, so the UI never yanks the user out of a section they chose.
        for section in self.accordion._sections:
            section._header.clicked.connect(self._mark_section_chosen)

        # ── ② wiring.  Widget-side only: no controller state is touched, which
        # is all this phase does (connections, not logic).  The
        # per-column "+ Add step" buttons + Preview/Apply are connected inline
        # in `_create_preprocessing_column` / `_make_recipe_column` (they are
        # rebuilt wholesale, so a static connect here would go stale).
        # The Order combo is the single visible editor; it drives the hidden,
        # frozen `transpose_button` that `controller.py:242` actually reads.
        self.axis_order_combo.currentTextChanged.connect(
            self._on_axis_order_changed)
        # `activated` fires only on USER interaction (not the programmatic
        # detection setter), so it is the signal that marks a deliberate
        # override — recorded per-source so a re-load keeps the user's Order.
        self.axis_order_combo.activated.connect(self._on_axis_order_user_picked)
        for spin in (self.voxel_z_spin, self.voxel_y_spin, self.voxel_x_spin):
            spin.valueChanged.connect(self._update_voxel_aspect)
            spin.valueChanged.connect(self._mark_voxel_edited)
            # editingFinished (focus-out / Enter) — the SAVE signal:
            # NOT per-keystroke valueChanged.  Gated on `_voxel_dirty`.
            spin.editingFinished.connect(self._on_voxel_editing_finished)
        # Reset restores the DETECTED baseline (widget-side) AND persists the
        # cleared override (controller-side).  Connection order matters: the
        # widget reset runs FIRST (pushes the current baseline into the spins),
        # then the controller runs and has the LAST word — for a user-override
        # source whose select-time baseline was the (1,1,1) placeholder it may
        # lazily re-detect the header and push the real spacing back into the
        # spins via `apply_voxel_size`, so the ↺ shows the re-detected value.
        self.voxel_reset_button.clicked.connect(self._reset_voxel_to_detected)
        self.voxel_reset_button.clicked.connect(self.controller.on_voxel_reset)
        self._view_toggle_group.buttonToggled.connect(self._on_view_toggle)

        # ── ④ wiring.
        self.patch_search_edit.textChanged.connect(self._filter_patch_list)
        self.patch_sort_combo.currentIndexChanged.connect(self._sort_patch_list)
        for spin in (self.ztrunc_start_spin, self.ztrunc_end_spin):
            spin.valueChanged.connect(self._on_ztrunc_spin_changed)
        self.extracted_roi_list.itemSelectionChanged.connect(
            self._refresh_patch_details)

        # ── ③ wiring.  Inclusion is the per-row checkbox (→ `_channel_included`);
        # the checkbox `toggled` handler drives the readout.  The old
        # `itemSelectionChanged` wires stay (harmless visual focus / row-order
        # normalisation), but inclusion no longer rides on selection.
        self.image_layer_combo.itemSelectionChanged.connect(
            self._normalize_channel_selection)
        self.image_layer_combo.itemSelectionChanged.connect(
            self._update_channel_order_readout)
        self.channel_select_all_button.clicked.connect(self._select_all_channels)
        self.channel_deselect_all_button.clicked.connect(
            self._deselect_all_channels)

        # ── ① wiring.  `+ Add` reuses the existing folder browse rather than
        # introducing a second code path.
        self.add_source_button.clicked.connect(
            self.controller.browse_source_folder)
        self.display_name_edit.textChanged.connect(self._update_alias_hint)
        # editingFinished (focus-out / Enter) is the alias SAVE signal
        # — NOT the per-keystroke textChanged, which only updates the
        # hint.  A programmatic `setText` (restore) does not emit it.
        self.display_name_edit.editingFinished.connect(
            self.controller.on_display_name_edited)
        # `on_source_file_selected` auto-fills the alias on an ① click; record
        # it so `_on_source_combo_picked` can tell auto from hand-typed.
        self.source_file_list.itemClicked.connect(self._note_auto_alias)
        self.source_file_list.currentRowChanged.connect(
            self._on_source_row_changed)
        self.source_combo.activated.connect(self._on_source_combo_picked)
        # `dataChanged` matters: `browse_source_folder` adds items with EMPTY
        # text and `_refresh_source_file_list` fills it in a second pass.
        for sig in (self.source_file_list.model().rowsInserted,
                    self.source_file_list.model().rowsRemoved,
                    self.source_file_list.model().modelReset,
                    self.source_file_list.model().dataChanged):
            sig.connect(self._refresh_source_dependents)
        # The alias drives ③'s provisional box names, so a rename re-labels them.
        self.display_name_edit.textChanged.connect(self.refresh_draw_box_list)

        # ── Initial state, once every connection is in place.
        self._refresh_source_dependents()
        self._update_alias_hint()
        self._rebuild_channel_rows()
        self._update_channel_order_readout()
        self._update_extract_source_hint()
        self._update_voxel_aspect()

    # ------------------------------------------------------------------
    # ① list ⇄ ② combo — a two-way proxy over ONE selection
    # ------------------------------------------------------------------

    def _note_auto_alias(self, *_args) -> None:
        """Remember the alias the controller just auto-derived (see D4 in
        `_on_source_combo_picked`)."""
        self._auto_alias = self.display_name_edit.text().strip()

    def _refresh_source_dependents(self, *_args) -> None:
        self._rebuild_source_cards()
        self._rebuild_source_combo()

    def _rebuild_source_combo(self) -> None:
        """Mirror ①'s source list into ②'s combo.

        The combo LABEL is the source's saved alias when it is a
        registered source, else the stem (`Qt.UserRole + 1`, set at insert by
        `browse_source_folder`) — read rather than `item.text()`, which is still
        empty when `rowsInserted` fires.  The `ToolTipRole` carries the full
        original filename + path so a long/aliased name is still identifiable
        (combo elision keeps the visible label short).
        """
        if getattr(self, "_source_sync_active", False):
            return
        self._source_sync_active = True
        try:
            # `{normalized_path: alias}` for the registered sources (one state
            # read for the whole rebuild); empty when no controller / workspace.
            alias_map = {}
            cb = getattr(getattr(self, "controller", None), "source_alias_map", None)
            if cb is not None:
                try:
                    alias_map = cb() or {}
                except Exception:
                    alias_map = {}
            self.source_combo.clear()
            for i in range(self.source_file_list.count()):
                item = self.source_file_list.item(i)
                stem = item.data(Qt.UserRole + 1) or item.text() or f"item {i}"
                path = item.data(Qt.UserRole)
                alias = None
                if path:
                    alias = alias_map.get(
                        os.path.normcase(os.path.normpath(str(path))))
                self.source_combo.addItem(str(alias or stem))
                idx = self.source_combo.count() - 1
                tip = f"{stem}\n{path}" if path else str(stem)
                self.source_combo.setItemData(idx, tip, Qt.ToolTipRole)
            # MEASURED: QComboBox auto-selects index 0 on populate, while the
            # list's currentRow is still -1 after a Browse.  A guarded
            # `if 0 <= row` therefore leaves the combo naming a source that
            # `load_source:202` reports as "No file selected".  Set it
            # unconditionally so the two cannot disagree.
            row = self.source_file_list.currentRow()
            self.source_combo.setCurrentIndex(
                row if 0 <= row < self.source_combo.count() else -1)
            # `setCurrentIndex` to the SAME index emits nothing, so refresh the
            # box tooltip here too (the item tips were just rebuilt).
            self._sync_source_combo_tooltip()
        finally:
            self._source_sync_active = False

    def _sync_source_combo_tooltip(self, *_args) -> None:
        """Put the CURRENT item's tooltip (name + path) on the closed combo, so
        an elided name is one hover away from being readable.  Falls back to the
        standing hint when nothing is selected."""
        combo = getattr(self, "source_combo", None)
        if combo is None:
            return
        tip = combo.itemData(combo.currentIndex(), Qt.ToolTipRole)
        combo.setToolTip(str(tip) if tip
                         else getattr(self, "_SOURCE_COMBO_HINT", ""))

    def _on_source_row_changed(self, row: int) -> None:
        """① list → ② combo."""
        if getattr(self, "_source_sync_active", False):
            return
        self._source_sync_active = True
        try:
            if 0 <= row < self.source_combo.count():
                self.source_combo.setCurrentIndex(row)
        finally:
            self._source_sync_active = False

    def _on_source_combo_picked(self, index: int) -> None:
        """② combo → ① list.

        Writes the list's CURRENT ROW (what `controller.load_source` reads), which
        fires ``currentItemChanged`` -> the controller's SINGLE selection funnel
        (`_on_source_current_item_changed`) BEFORE ``currentRowChanged``.  That
        funnel does the FULL restore (alias / axis order / voxel), so the combo
        route no longer re-derives the alias itself — switching A→B restores B's
        SAVED setup exactly like a mouse/keyboard select (no A-alias-on-B
        provenance corruption).  The `_source_sync_active` guard covers only the
        list↔combo mirror; the funnel deliberately ignores it (verified: the
        mirror writes the COMBO, never the list's current item)."""
        if getattr(self, "_source_sync_active", False):
            return
        self._source_sync_active = True
        try:
            if 0 <= index < self.source_file_list.count():
                self.source_file_list.setCurrentRow(index)
        finally:
            self._source_sync_active = False

    def _on_axis_order_changed(self, text: str) -> None:
        """Order combo → the hidden transpose flag the controller reads.

        `load_source` swaps axes 0 and 1 when `transpose_button.isChecked()`
        (`controller.py:242`), which is exactly ZCYX → CZYX.  Mapping the combo
        onto it keeps ONE editor for the field while leaving the
        frozen widget and its consumer untouched.
        """
        self.transpose_button.setChecked(text == "ZCYX")
        self.axis_order_hint.setText(
            "not detected — confirm" if text != "ZYX" else "single channel")
        self._log("STEP1-AXES", f"order set to {text}")

    def apply_axis_order(self, order, hint) -> None:
        """Set the Order combo + hint from the controller's L0 axis detection.

        *order* is ``"CZYX"`` / ``"ZCYX"`` / ``"ZYX"``; a 3D ``"ZYX"`` source
        maps onto the combo's ``CZYX`` / no-transpose state — the combo has only
        ``CZYX`` and ``ZCYX`` (frozen) and a 3D stack needs no swap.
        The combo stays the single user-confirmable editor: setting it
        here drives the hidden transpose flag via ``_on_axis_order_changed`` but
        does NOT emit ``activated``, so it is never mistaken for a user override.
        The explicit *hint* is applied AFTER the combo, replacing the generic
        one the combo's own signal writes, with the detection detail.
        """
        combo_value = "ZCYX" if str(order).upper() == "ZCYX" else "CZYX"
        self.axis_order_combo.setCurrentText(combo_value)
        self.axis_order_hint.setText(hint)
        self.axis_order_hint.setStyleSheet(DIM_LABEL_STYLE)

    def _on_axis_order_user_picked(self, *_args) -> None:
        """User deliberately picked an Order (combo ``activated``) — tell the
        controller so a re-load of this source keeps the choice."""
        cb = getattr(self.controller, "note_axis_order_override", None)
        if cb is not None:
            cb()

    def configure_3d_detail(self, n_levels: int, default_level: int) -> None:
        """Show + range the 3D-detail slider for a freshly-displayed multiscale
        store of *n_levels* levels, parked at *default_level* (the coarsest, so
        the initial 3D view matches napari's default — no surprise memory hit).

        Setting the value fires ``valueChanged`` -> the controller applies the
        level + writes the readout, so no separate initial-apply call is needed.
        Range/value are set under a signal block, then value is set once to fire
        exactly one apply."""
        s = self.threed_detail_slider
        s.blockSignals(True)
        s.setMinimum(0)
        s.setMaximum(max(0, int(n_levels) - 1))
        s.blockSignals(False)
        self.threed_detail_row.setVisible(True)
        # Fire exactly one valueChanged so the controller applies + labels.  If
        # the value is already `default_level`, force a manual apply.
        if s.value() == int(default_level):
            self._on_3d_detail_slider_changed(int(default_level))
        else:
            s.setValue(int(default_level))

    def hide_3d_detail(self) -> None:
        """Hide the 3D-detail row (no multiscale store displayed)."""
        if hasattr(self, "threed_detail_row"):
            self.threed_detail_row.setVisible(False)

    def configure_zoom_depth(self, mode) -> None:
        """Show the "Zoom depth" combo AND sync its selection to *mode* (the
        controller's ``_zoom_depth_wanted`` source-of-truth: None / a float /
        "cube").

        Mirrors ``configure_3d_detail``'s "sync-on-show" pattern so the combo can
        never drift from the actual render mode across a store change / any reset
        path.  ``blockSignals`` so the programmatic ``setCurrentIndex`` does NOT
        fire ``currentIndexChanged`` -> ``on_zoom_depth_changed`` (which would
        re-enter the controller during a refresh)."""
        if not hasattr(self, "zoom_depth_combo"):
            return
        idx = ZOOM_DEPTH_MODES.index(mode) if mode in ZOOM_DEPTH_MODES else 0
        self.zoom_depth_combo.blockSignals(True)
        self.zoom_depth_combo.setCurrentIndex(idx)
        self.zoom_depth_combo.blockSignals(False)
        if hasattr(self, "zoom_depth_row"):
            self.zoom_depth_row.setVisible(True)

    def hide_zoom_depth(self) -> None:
        """Hide the "Zoom depth" combo row (2D / progressive OFF)."""
        if hasattr(self, "zoom_depth_row"):
            self.zoom_depth_row.setVisible(False)

    def set_3d_detail_label(self, text: str) -> None:
        if hasattr(self, "threed_detail_label"):
            self.threed_detail_label.setText(text)

    def _on_3d_detail_slider_changed(self, value: int) -> None:
        """Slider moved (or configured) — forward to the controller, which sets
        ``locked_data_level`` on the multiscale layers + writes the readout."""
        cb = getattr(self.controller, "on_3d_detail_changed", None)
        if cb is not None:
            cb(int(value))

    def update_image_layer_list(self, layer_names):
        """Called by the controller whenever the viewer's Image layers change.

        Feeds BOTH ③'s channel picker AND ②'s per-channel preprocessing columns:
        the channel count is only knowable AFTER a load, and `_on_layers_changed`
        already tracks exactly that.

        Selection is preserved BY NAME, so a layer appearing or disappearing
        elsewhere cannot silently move which channel a row describes.
        """
        # Inclusion now lives in the name-keyed `self._channel_included` dict
        # (③ owns include + order), so it survives this rebuild for free.
        # Reconcile the dict to the new name set: KEEP every existing value (a
        # deliberate untick stays False and is NEVER silently re-checked),
        # DEFAULT any new name to True, and PRUNE names that disappeared.
        # `_on_layers_changed` fires on every insert/remove/rename — including
        # `_ensure_shapes_layer`'s — so this is what stops a Draw Free from
        # clearing the include-set and breaking Extract.
        lst = self.image_layer_combo
        self._channel_included = {
            n: self._channel_included.get(n, True) for n in layer_names}
        # Preserve the hand-ordering across the rebuild: already-ordered names
        # keep their order, new names append.  Selection is not read for
        # inclusion any more; it may remain only as harmless visual focus.
        previous_order = [lst.item(i).text() for i in range(lst.count())]
        ordered = ([n for n in previous_order if n in layer_names]
                   + [n for n in layer_names if n not in previous_order])
        lst.blockSignals(True)
        lst.clear()
        lst.addItems(ordered)
        lst.blockSignals(False)

        # ② preprocessing: reconcile the per-source recipe to the detected
        # channel set (bind-by-INDEX) and rebuild the columns.  Guarded so a
        # controller call before the ② panel is built is a no-op.
        if hasattr(self, "_recipe_columns_layout"):
            self._sync_recipe_channels(layer_names)
        self._rebuild_channel_rows()
        self._update_channel_order_readout()
        # ① shows the channels read-only, with a pointer to ③.
        summary = " ".join(f"▪{n}" for n in layer_names) or "—"
        self.session_channels_label.setText(
            f"Channels: {summary} · named in Draw & Extract")

    def _sync_recipe_channels(self, layer_names) -> None:
        """Reconcile `self._recipe` to the DETECTED channel set + rebuild the
        columns.  The recipe binds to channel INDEX, so a channel-count
        change validates it against the new count via `normalize_recipe` (drops
        out-of-range channels, coerces params) — but ONLY when channels are
        present, so an unload (0 channels) leaves the recipe intact in memory for
        a re-load (it stays on disk regardless)."""
        names = list(layer_names)
        self._recipe_channel_names = names
        if names:
            self._recipe = _recipe.normalize_recipe(self._recipe, len(names))
        self._refresh_recipe_view()

    # ------------------------------------------------------------------
    # ③ channel picker
    # ------------------------------------------------------------------

    def _rebuild_channel_rows(self) -> None:
        """Render one row per channel: `☑ · LUT · name · chN · [↑][↓]`.

        Ordering the ROWS is what controls the stack order, because
        `extract_patches` reads `selectedItems()` and Qt returns those in row
        order.  Rows are rebuilt wholesale so a button can never act on a
        different channel than the one its row shows.
        """
        lst = self.image_layer_combo
        for i in range(lst.count()):
            item = lst.item(i)
            if lst.itemWidget(item) is not None and item.data(
                    Qt.UserRole + 9) == item.text():
                continue
            row = self._make_channel_row(i, item.text())
            item.setSizeHint(row.sizeHint())
            lst.setItemWidget(item, row)
            item.setData(Qt.UserRole + 9, item.text())
        self.channel_empty_label.setVisible(lst.count() == 0)

    def _make_channel_row(self, index: int, name: str) -> QWidget:
        row = QWidget()
        # Opaque background so the QListWidget delegate's own item text (the
        # channel name, needed by `checked_channel_names`) does not ghost behind
        # this overlay row — same fix as the box row `_make_row` (L1).
        row.setAutoFillBackground(True)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(4, 1, 4, 1)
        lay.setSpacing(6)

        # Real per-row include checkbox (all checked by default).  D1 (ui.py
        # :389, MEASURED): a native QListWidgetItem check is UNCLICKABLE once
        # `setItemWidget` registers an editor for the row, so the
        # `_import_pairs_dialog` pattern — a real QCheckBox in the row — is
        # used.  Its state is READ FROM + WRITTEN TO `self._channel_included`
        # (name-keyed), so it survives a row rebuild and a reorder.  State is
        # set BEFORE the `toggled` connect so the programmatic `setChecked`
        # cannot fire the handler during construction.
        check = QCheckBox()
        check.setToolTip("Include this channel in the extract")
        # Rounded amber indicator (the app's polished checkbox look, from
        # theme.make_pair_card) so the widget checkbox reads as intentional +
        # on-theme rather than a bare default square (P4 — match the box-list feel).
        check.setStyleSheet(
            "QCheckBox { spacing: 0; }"
            "QCheckBox::indicator {"
            "  width: 14px; height: 14px;"
            "  border: 1px solid rgba(255, 255, 255, 0.30);"
            "  border-radius: 3px;"
            "  background-color: rgba(255, 255, 255, 0.04);"
            "}"
            "QCheckBox::indicator:checked {"
            "  background-color: rgba(212, 168, 67, 0.65);"
            "  border: 1px solid rgba(212, 168, 67, 0.85);"
            "}"
            "QCheckBox::indicator:hover {"
            "  border: 1px solid rgba(255, 255, 255, 0.50);"
            "}"
        )
        check.setChecked(self._channel_included.get(name, True))
        check.toggled.connect(
            lambda on, nm=name: self._on_channel_included_toggled(nm, on))
        lay.addWidget(check, 0)

        swatch = QLabel("")
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(
            f"background-color: "
            f"{_channel_display_hex(index)};"
            f" border-radius: 2px;")
        swatch.setToolTip("Display colour for this channel")
        lay.addWidget(swatch, 0)

        label = QLabel()
        label.setStyleSheet("font-size: 11px; color: #dddddd;")
        elide_label(label, name, ROW_NAME_MAX_PX)   # full name on hover
        lay.addWidget(label, 1)

        tag = QLabel(f"ch{index}")
        tag.setStyleSheet(DIM_LABEL_STYLE)
        lay.addWidget(tag, 0)

        for glyph, icon_name, tip, delta in (
                ("↑", "mdi6.chevron-up", "Stack earlier", -1),
                ("↓", "mdi6.chevron-down", "Stack later", 1)):
            btn = QPushButton(glyph)
            btn.setFixedSize(20, 20)
            btn.setToolTip(tip)
            btn.setStyleSheet(SECONDARY_BTN_STYLE)
            # Real chevron icon so the button isn't an empty "tofu" box (the ↑/↓
            # glyphs are missing in the UI font).  Clear the text glyph ONLY when
            # the icon actually resolved — a blank button is worse than tofu.
            if apply_button_icon(btn, icon_name, color="#dddddd", size=14):
                btn.setText("")
            btn.clicked.connect(
                lambda _c=False, i=index, d=delta: self._move_channel(i, d))
            lay.addWidget(btn, 0)
        return row

    def _move_channel(self, index: int, delta: int) -> None:
        """Reorder a channel, preserving its ticked state.

        `takeItem`/`insertItem` drops the selection, so it is captured and
        restored — otherwise reordering would silently un-include the channel.
        """
        lst = self.image_layer_combo
        target = index + delta
        if not (0 <= index < lst.count() and 0 <= target < lst.count()):
            return
        was_selected = lst.item(index).isSelected()
        item = lst.takeItem(index)
        lst.insertItem(target, item)
        item.setSelected(was_selected)
        for i in range(lst.count()):
            lst.item(i).setData(Qt.UserRole + 9, None)
        self._rebuild_channel_rows()
        self._normalize_channel_selection()
        self._update_channel_order_readout()

    def _normalize_channel_selection(self) -> None:
        """Force the selection model's order to match ROW order.

        MEASURED, not assumed: `QListWidget.selectedItems()` returns items in
        the order they were ADDED TO THE SELECTION, not in row order.  Selecting
        `nuc` then `cad` yields `['nuc', 'cad']`, and physically moving a row
        leaves `selectedIndexes()` as `[1, 0]`.

        That matters because `controller.extract_patches:558-559` builds the
        channel stack straight from `selectedItems()`.  So the ↑↓ buttons would
        reorder the *display* and change nothing about the data — the exact
        class of lie the explicit ordering exists to remove.

        Re-applying the selection in row order makes what you see the order you
        get, without touching the controller.  Signals are blocked so the
        re-selection does not re-enter through `itemSelectionChanged`.
        """
        lst = self.image_layer_combo
        if getattr(self, "_channel_sel_active", False):
            return
        self._channel_sel_active = True
        try:
            chosen = {i.text() for i in lst.selectedItems()}
            if not chosen:
                return
            lst.blockSignals(True)
            lst.clearSelection()
            for i in range(lst.count()):
                if lst.item(i).text() in chosen:
                    lst.item(i).setSelected(True)
            lst.blockSignals(False)
        finally:
            self._channel_sel_active = False

    def _on_channel_included_toggled(self, name: str, checked: bool) -> None:
        """A channel's include checkbox was toggled — persist to the name-keyed
        `self._channel_included` (the ONE inclusion source of truth) and refresh
        the order readout."""
        self._channel_included[name] = bool(checked)
        self._update_channel_order_readout()

    def checked_channel_names(self) -> list[str]:
        """The INCLUDED channel names IN ROW ORDER — the ONE inclusion+order
        reader the extract uses.

        Row order is item order (independent of check / selection state); a name
        is included when `self._channel_included` says so (default True for an
        unseen name).  For the same included set + row order this returns the
        identical list the old `selectedItems()`-in-row-order returned, so
        everything downstream (`_selected_channel_names`,
        `get_combined_image_stack`, `channel_pairs`, the per-patch `"channels"`
        sidecar) is byte-identical — only the DEFAULT (all-included) and the
        GESTURE (tick vs select) changed."""
        lst = self.image_layer_combo
        return [lst.item(r).text() for r in range(lst.count())
                if self._channel_included.get(lst.item(r).text(), True)]

    def _set_all_channels_included(self, included: bool) -> None:
        """Bulk-set every CURRENT channel's inclusion, then re-render so the
        checkboxes reflect it and the readout refreshes.  Names are read from
        the list in row order, so a channel added since the dict last synced is
        still covered.  Forces a full row rebuild (clears the cached row key)
        so each freshly built checkbox re-reads the dict."""
        lst = self.image_layer_combo
        for r in range(lst.count()):
            self._channel_included[lst.item(r).text()] = included
            lst.item(r).setData(Qt.UserRole + 9, None)
        self._rebuild_channel_rows()
        self._update_channel_order_readout()

    def _select_all_channels(self, *_args) -> None:
        self._set_all_channels_included(True)

    def _deselect_all_channels(self, *_args) -> None:
        self._set_all_channels_included(False)

    def _update_channel_order_readout(self, *_args) -> None:
        """Show the channels the extract will actually stack, in row order.

        Derived from `checked_channel_names()` — the SAME inclusion+order the
        extract reads (`controller._selected_channel_names` /
        `get_combined_image_stack`) — so it cannot disagree with what gets
        stacked.  The "No channels ticked" prompt shows ONLY when the included
        set is empty.
        """
        names = self.checked_channel_names()
        self.channel_order_label.setText(
            "Extraction order: " + ", ".join(names) if names
            else "No channels ticked — the extract needs at least one."
        )

    def channel_pairs(self) -> list:
        """The ONE ordered `[(channel_label, layer_name)]` list.

        `channel_names` and the stacked array must come from a single ordered
        pair-list or they can drift.  `channel_label` is kept distinct from
        `layer.name` (4) so a session-indexed layer name is never
        written into the sidecar as a channel identity.  Reads the INCLUDED
        channels in row order (`checked_channel_names()`), the same inclusion
        the extract uses.
        """
        return [(name, name) for name in self.checked_channel_names()]

    def update_shape_layer_list(self, layer_names):
        self.bbox_layer_combo.clear()
        self.bbox_layer_combo.addItems(layer_names)

    def checked_patch_names(self):
        """④'s export scope — the ticked, VISIBLE rows (c #14).

        Returns a list, **empty when nothing is ticked** — never `None`.  It
        used to return `None` there so the caller could fall back to the
        selected row, which reinstated `currentItem()` as a second scope: after
        deliberately unticking every row you could still press Export and have
        it write the highlighted patch.  #14 rules that out by name ("④ using a
        different mechanism (selection, …)"), and this is a permanent write, so
        an empty tick set must mean *nothing*, not *something else*.

        **Hidden rows are excluded.**  The search box hides rows without
        touching their check state, so a filtered list would otherwise export
        patches the user cannot see — the exact opposite of "scope always
        visible".
        """
        lst = self.extracted_roi_list
        return [lst.item(i).text() for i in range(lst.count())
                if lst.item(i).checkState() == Qt.Checked
                and not lst.item(i).isHidden()]

    def box_row_count(self) -> int:
        """How many rows ③'s Box List currently shows.

        The tick filter is POSITIONAL, so the controller must be able to check
        that the list and the Shapes layer agree before trusting it.
        """
        return self.draw_box_list.list.count()

    def checked_box_indices(self):
        """Row indices ticked in ③'s Box List, for `Extract checked`.

        Returns `None` when the list is empty, so a session that never opened ③
        extracts everything exactly as before rather than silently extracting
        nothing.
        """
        lst = self.draw_box_list.list
        if lst.count() == 0:
            return None
        return {i for i in range(lst.count())
                if lst.item(i).checkState() == Qt.Checked}

    def refresh_draw_box_list(self, *_args) -> None:
        """Rebuild ③'s Box List from the live Shapes layer.

        Boxes are never counted anywhere today (drawn-box counting is
        new work), so this reads the layer directly.  Names are provisional
        `{prefix}_N` previews using the same rule
        `controller.extract_patches:643-650` will apply — so the row you tick is
        labelled with the name the patch will actually get.
        """
        # Phase-1 box tool owns the list once any ③ box exists: its BoxRecs are
        # the source of truth, so the legacy Shapes-layer rebuild below must not
        # clobber the box-tool rows (an alias edit still re-labels them, via the
        # tool's own refresh).
        box_tool = getattr(getattr(self, "controller", None), "box_tool", None)
        if box_tool is not None and (box_tool.is_active() or box_tool.has_boxes()):
            # Defer while box mode is ON even with zero boxes, so the legacy
            # branch never watches the box tool's own Shapes layer (a `redraw`
            # emits `events.data`, which would otherwise re-enter this on every
            # slider tick — a per-drag rebuild storm).
            box_tool.refresh_list()
            return

        layer = self._current_shapes_layer()
        n = 0 if layer is None else len(getattr(layer, "data", []) or [])
        self._watch_shapes_layer(layer)
        lst = self.draw_box_list.list

        alias = self.display_name_edit.text().strip()
        prefix = f"{alias}_patch" if alias else "patch"
        # Numbered from where extraction will ACTUALLY start: `extract_patches`
        # uses `find_next_patch_index` (cumulative per source), so
        # counting from 1 mislabels every row on a workspace that already holds
        # patches.
        start = self._next_patch_index(prefix)
        wanted = [f"{prefix}_{start + i}" for i in range(n)]

        # Skip the rebuild only when the NAMES already match.  An equal count
        # can still mean different boxes (a delete plus an add in one event),
        # and the names change whenever the alias or the workspace does.
        if [lst.item(i).text() for i in range(lst.count())] == wanted:
            self.draw_box_list.refresh()
            return

        # Preserve unticks by POSITION, not by name.  Two MEASURED failures of
        # the by-name version, both silently re-including a box the user had
        # excluded — and `extract_patches` filters on `checked_box_indices`,
        # which is positional, so position is also the consistent key:
        #   • every rebuild RENUMBERS (`start` moves with the workspace), so
        #     after an export the names all change and no untick matches;
        #   • editing the alias changes every name at once, same result.
        unchecked = {i for i in range(lst.count())
                     if lst.item(i).checkState() != Qt.Checked}
        lst.clear()
        for i, name in enumerate(wanted):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if i in unchecked else Qt.Checked)
            # MEASURED: without the marker, `addItem` fires `rowsInserted` →
            # `BoxListPanel.refresh` → `make_checkable`, which sees an
            # uninitialised item and forces it back to Checked — so the untick
            # above survived a bare `refresh()` but was reverted the moment the
            # user drew another box.  Two mechanisms wrote the same state and
            # the wrong one won.
            item.setData(Qt.UserRole + 7, True)
            lst.addItem(item)
        self.draw_box_list.refresh()

    def _current_shapes_layer(self):
        """The ROI Shapes layer, READ-ONLY — never creating one.

        `controller.get_selected_shapes_layer()` calls `_ensure_shapes_layer()`,
        which CREATES a layer.  Driving the box-list refresh through it would
        drop an empty "ROI Shapes" layer into the shared viewer the moment a
        source is clicked — before any workspace or image exists — and re-enter
        `_on_layers_changed`, which then clears the channel selection.
        """
        try:
            from napari.layers import Shapes as _Shapes
        except Exception:  # pragma: no cover - napari always present in app
            return None
        layers = getattr(self.viewer, "layers", None) or []
        try:
            return next((l for l in layers if isinstance(l, _Shapes)), None)
        except Exception as exc:
            self._log("STEP1-DRAW", f"could not read shapes layers: {exc}")
            return None

    def _next_patch_index(self, prefix: str) -> int:
        """Where the next extracted patch's numbering starts.

        Delegates to the same pure helper `extract_patches` uses, so the row
        label and the allocated name cannot diverge.  Falls back to 1 when
        there is no workspace yet.
        """
        workspace = self.output_file_path.text().strip()
        if not workspace:
            return 1
        try:
            from .logic import find_next_patch_index
            return int(find_next_patch_index(workspace, prefix))
        except Exception as exc:
            self._log("STEP1-DRAW", f"could not read next patch index: {exc}")
            return 1

    def _watch_shapes_layer(self, layer) -> None:
        """Subscribe once to a Shapes layer's data changes.

        This is the module's first `layer.events.data` subscription.  The
        per-layer `_epicycle_step1_watched` flag is the repo's standard guard
        against handler accumulation — reconnecting on every draw-mode switch
        would fire the rebuild N times per edit.
        """
        if layer is None or getattr(layer, "_epicycle_step1_watched", False):
            return
        try:
            layer.events.data.connect(self.refresh_draw_box_list)
            layer._epicycle_step1_watched = True
        except Exception as exc:
            self._log("STEP1-DRAW", f"could not watch shapes layer: {exc}")

    def _set_draw_mode(self, mode: str) -> None:
        colors = {"Idle": "#888888", "Free Draw": "#5ba4cf", "Fixed": "#d4a843"}
        color = colors.get(mode.split(" ")[0], "#888888")
        self.draw_mode_label.setText(f"Mode: {mode}")
        self.draw_mode_label.setStyleSheet(f"font-size: 10px; color: {color};")

    def start_free_drawing(self):
        self._disconnect_mouse_callback()
        layer = self.controller.get_selected_shapes_layer()
        if layer:
            layer.mode = "add_rectangle"
            self._watch_shapes_layer(layer)
            self.refresh_draw_box_list()
            self._set_draw_mode("Free Draw \u2014 drag to draw rectangles")

    def start_fixed_drawing(self):
        layer = self.controller.get_selected_shapes_layer()
        if not layer:
            self._log("STEP1-DRAW", "No valid Shapes layer selected.")
            return

        layer.mode = "pan_zoom"
        self._disconnect_mouse_callback()
        self._watch_shapes_layer(layer)
        self.refresh_draw_box_list()
        self.mouse_callback = self._make_fixed_box_callback(layer)
        self.viewer.mouse_drag_callbacks.append(self.mouse_callback)
        size = self.bbox_size_spinbox.value()
        self._set_draw_mode(f"Fixed ({size}px) \u2014 click to place")

    def _disconnect_mouse_callback(self):
        if hasattr(self, "mouse_callback"):
            try:
                self.viewer.mouse_drag_callbacks.remove(self.mouse_callback)
            except ValueError:
                pass
            self.mouse_callback = None

    def _make_fixed_box_callback(self, layer):
        def callback(viewer, event):
            if layer.mode != "pan_zoom":
                return

            start = event.position
            yield
            end = event.position

            if not np.allclose(start, end, atol=1e-1):
                return

            size = self.bbox_size_spinbox.value()
            half = size / 2
            y, x = end[-2], end[-1]

            y0, y1 = y - half, y + half
            x0, x1 = x - half, x + half

            # Clamp to image bounds if an image layer exists.
            for lyr in viewer.layers:
                if hasattr(lyr, "data") and hasattr(lyr.data, "shape"):
                    sh = lyr.data.shape
                    if len(sh) >= 2:
                        img_h, img_w = sh[-2], sh[-1]
                        y0 = max(0, y0)
                        y1 = min(img_h, y1)
                        x0 = max(0, x0)
                        x1 = min(img_w, x1)
                        break

            rect = np.array(
                [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]
            )
            layer.add(rect, shape_type="rectangle")

        return callback

    def set_roi_queue_summary(self, total: int, conditioned: int) -> None:
        """Frozen widget method — the controller calls it, so the NAME
        must survive; the label it used to write does not.

        It printed `"Queue: N ROI(s), M ready"`, and the redesign retires `ready`
        outright: it meant "has a `selected_z_range`", which is now ④'s detail
        pane's business, not a top-level count.  The body is retargeted to the
        ① dashboard, which speaks the `new · staged · captured` vocabulary.
        `conditioned` is deliberately unused — that is the retired counter.
        """
        # All three counters, not just staged: `new` is what is drawn but not
        # extracted, `captured` what is on disk.  Previously only `staged` ever
        # moved, so two thirds of the strip was permanently zero.
        drawn = self.draw_box_list.list.count()
        captured = max(0, self.extracted_roi_list.count() - int(total))
        self.refresh_summary_dashboard({
            "new": max(0, drawn - int(total)),
            "staged": int(total),
            "captured": captured,
        })
        self.library_box_list.refresh()

    def set_export_status(self, message: str, success: bool = True) -> None:
        color = "#6abf69" if success else "#e07070"
        self.export_status_label.setText(message)
        self.export_status_label.setStyleSheet(f"font-size: 10px; color: {color};")

    def set_selected_roi_status(self, roi_name: str, z_range) -> None:
        self.selected_roi_label.setText(f"Selected ROI: {roi_name or '-'}")
        if z_range and len(z_range) == 2:
            self.selected_zrange_label.setText(
                f"Selected Z-range: {int(z_range[0])} to {int(z_range[1])}"
            )
            self.seed_ztrunc_spins(z_range)
        else:
            self.selected_zrange_label.setText("Selected Z-range: -")

    def seed_ztrunc_spins(self, z_range) -> None:
        """Point the Z-truncate spinboxes AT the range they are editing.

        They were write-only: nothing ever wrote them FROM the patch or from
        the matplotlib span, so both sat at 0 while the label read
        "12 to 62" — and one nudge of either spin then wrote `(0, 1)` into
        `controller.selected_z_range`, which `Set Range` commits and
        `_export_patch_by_name` slices the TIFF with.  A 1-slice crop from a
        stray click on a control that LOOKED like it held the real range.

        Signals are blocked so seeding cannot write back through
        `_on_ztrunc_spin_changed` — that would defeat the span the seed came
        from, which is the mirror-image of the same bug.
        """
        if not z_range or len(z_range) != 2:
            return
        lo, hi = int(z_range[0]), int(z_range[1])
        for spin, value in ((self.ztrunc_start_spin, lo),
                            (self.ztrunc_end_spin, hi)):
            if value > spin.maximum():
                spin.setMaximum(value)
            was = spin.blockSignals(True)
            try:
                spin.setValue(value)
            finally:
                spin.blockSignals(was)

from __future__ import annotations

import html
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Mapping
from typing import Callable, Dict, Optional

import psutil
import qtawesome as qta
from qtpy.QtCore import QSize, QTimer, Qt
from qtpy.QtGui import QFont, QFontMetrics
from qtpy.QtWidgets import (
    QDockWidget,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from napari.viewer import Viewer

from ..data_preparation.ui import DataPreparationWidget
from ..fusion_reconstruction.ui import FusionReconstructionCuration
from ..hitl_training_annotation.ui.main_widget import TrainingAnnotationHITL2D
from ..segmentation.ui import SegmentStack
from ..tiled_segmentation.ui import LargeScaleSegmentation
from . import _workspace_facts


@dataclass(frozen=True)
class _WorkflowStep:
    step_id: str
    title: str
    subtitle: str
    section: str
    widget_factory: Callable[[Viewer], QWidget]
    group: str = "pipeline"  # "pipeline" or "utility"


# Section header colors — muted to sit well on napari's dark theme.
_SECTION_COLORS: Dict[str, str] = {
    "Setup": "#5ba4cf",       # teal
    "Build": "#d4a843",       # amber
    "Apply": "#6abf69",       # green
    "Visualization": "#a78bca",  # purple (future)
    "Animations": "#5b9fd4",     # blue (future)
    "Tools": "#8a9bae",          # slate (future)
}

_SECTION_ICONS: Dict[str, str] = {
    "Setup": "mdi6.tune-variant",
    "Build": "mdi6.layers-triple-outline",
    "Apply": "mdi6.play-circle-outline",
    "Visualization": "mdi6.eye-outline",
    "Animations": "mdi6.filmstrip",
    "Tools": "mdi6.wrench-outline",
}

_STEP_ICONS: Dict[str, str] = {
    "project_setup": "mdi6.crosshairs",
    "roi_hitl_2d": "mdi6.refresh",
    "reconstruct_curate_3d": "mdi6.cube-outline",
    "apply_3d": "mdi6.table-row",
    "whole_tissue": "mdi6.arrow-expand-all",
}


#: Shared-workspace banner accents.
#: No new colour: red is today's warning red, green is the Apply
#: section green, amber is the Build section amber.
#: Both the shared-workspace key and the explicit "shared" alias resolve to
#: the ONE `_workspace_facts.SHARED_ACCENT` constant, so the accent can be
#: swapped (green -> teal) by editing that single line.  Read through a
#: function, not bound at import, so the swap needs no restart of anything
#: that already imported this module.
_BANNER_BASE_COLORS: Dict[str, str] = {
    "red": "#e07070",
    "amber": "#d4a843",
}


class _BannerPalette(Mapping):
    """The banner palette, with the shared accent resolved on every read.

    Reading it live (rather than baking the hex in at import) is what makes
    the accent a genuinely one-line change: everything that paints a shared
    strip goes through `_workspace_facts.SHARED_ACCENT` at the moment it
    paints, so editing that constant is the whole switch.
    """

    #: `_workspace_facts.SHARED_ACCENT_KEY` plus the explicit role name.
    _SHARED_KEYS = ("green", "shared")

    def __getitem__(self, key):
        if key in self._SHARED_KEYS:
            return _workspace_facts.SHARED_ACCENT
        return _BANNER_BASE_COLORS[key]

    def __contains__(self, key):
        return key in self._SHARED_KEYS or key in _BANNER_BASE_COLORS

    def __iter__(self):
        yield from self._SHARED_KEYS
        yield from _BANNER_BASE_COLORS

    def __len__(self):
        return len(self._SHARED_KEYS) + len(_BANNER_BASE_COLORS)


_BANNER_COLORS = _BannerPalette()

#: The step-panel background the banner sits on — used to dim the qualifier
#: text, since Qt rich text has no `opacity`.
_BANNER_BG = "#2e3138"


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' + alpha (0.0–1.0) to 'rgba(r,g,b,a)' for Qt CSS."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _blend(fg_hex: str, bg_hex: str, alpha: float) -> str:
    """`fg` at `alpha` over `bg`, as '#rrggbb' — a stand-in for CSS opacity."""
    out = []
    for i in (1, 3, 5):
        f = int(fg_hex[i:i + 2], 16)
        b = int(bg_hex[i:i + 2], 16)
        out.append(int(round(f * alpha + b * (1.0 - alpha))))
    return "#" + "".join(f"{v:02x}" for v in out)


class _BannerLabel(QLabel):
    """One coloured strip under the step title.

    Same shape as the pre-B1 hardcoded warning line — 3 px left edge, an 8 %
    tint of the same colour, 11 px text — but the colour is a parameter and
    the path inside it is **middle-elided** so the drive and the folder stay
    readable in a narrow panel.  The elision is recomputed on every resize;
    the untruncated path is always the tooltip and the text is selectable
    with the mouse.
    """

    _MONO = "'Consolas','Cascadia Mono','Courier New',monospace"

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__("", parent)
        self._text = ""
        self._path = ""
        self._tooltip = ""
        self._color_key = "red"
        self.setWordWrap(False)
        self.setTextFormat(Qt.RichText)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # A non-wrapping QLabel reports its FULL text width as its minimum
        # size hint, so a long path would force the whole dock panel wider
        # (measured: ~610 px, ratcheting on every resize).  The strip is
        # allowed to be as narrow as the panel wants — it elides instead.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setVisible(False)

    def minimumSizeHint(self):      # noqa: N802 (Qt override)
        return QSize(60, super().minimumSizeHint().height())

    def set_strip(self, text: str, color_key: str, path: str = "",
                  tooltip: str = "") -> None:
        """Show `text` in `color_key`; `path` (a substring of it) may elide.

        Early-returns when nothing changed: the 3 s state poll calls this on
        every tick, and an unconditional `setText` wipes whatever the user
        has selected with the mouse.
        """
        color_key = color_key if color_key in _BANNER_COLORS else "red"
        if (text == self._text and color_key == self._color_key
                and (tooltip or path or text) == self._tooltip):
            return
        self._text = text or ""
        self._path = path or ""
        self._color_key = color_key
        color = _BANNER_COLORS[self._color_key]
        self.setStyleSheet(
            f"color: {color}; font-size: 11px;"
            f"padding: 6px 12px; margin: 0px;"
            f"background: {_rgba(color, 0.08)};"
            f"border-left: 3px solid {color};"
        )
        self._tooltip = tooltip or self._path or self._text
        self.setToolTip(self._tooltip)
        self.setVisible(bool(self._text))
        self._render()

    def clear_strip(self) -> None:
        if not self._text:
            return
        self._text = ""
        self._path = ""
        self._tooltip = ""
        self.setText("")
        self.setToolTip("")
        self.setVisible(False)

    # -- rendering ---------------------------------------------------------
    def resizeEvent(self, event) -> None:      # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if not self._text:
            self.setText("")
            return
        if not self._path or self._path not in self._text:
            self.setText(html.escape(self._text))
            return
        head, _, tail = self._text.partition(self._path)
        metrics = QFontMetrics(self.font())
        # The path is DRAWN in the monospace face, which is wider than the UI
        # font: measuring it with `self.font()` under-counts and the line
        # overflows the panel.  Measure each run in the face it is painted in.
        mono_font = QFont(self.font())
        mono_font.setFamilies(
            ["Consolas", "Cascadia Mono", "Courier New", "monospace"])
        mono = QFontMetrics(mono_font)

        shown, tail_shown = self._path, tail
        if self.width() > 1:
            # 24 px of horizontal padding + 3 px border + a little slack.
            avail = self.width() - 34
            head_w = metrics.horizontalAdvance(head)
            min_path = 40
            # The qualifier gives way first — the path is the information.
            tail_budget = max(0, avail - head_w - min_path)
            if metrics.horizontalAdvance(tail) > tail_budget:
                tail_shown = metrics.elidedText(
                    tail, Qt.ElideRight, tail_budget)
            path_budget = max(
                min_path,
                avail - head_w - metrics.horizontalAdvance(tail_shown),
            )
            shown = mono.elidedText(self._path, Qt.ElideMiddle, path_budget)
        dim = _blend(_BANNER_COLORS[self._color_key], _BANNER_BG, 0.72)
        self.setText(
            f"{html.escape(head)}"
            f"<span style=\"font-family:{self._MONO}\">{html.escape(shown)}</span>"
            f"<span style=\"color:{dim}\">{html.escape(tail_shown)}</span>"
        )


class _NavItem(QPushButton):
    """A single clickable nav item styled with its section's color."""

    def __init__(self, label: str, color_hex: str, step_id: str = "",
                 icon_name: str = "", disabled: bool = False,
                 parent: QWidget = None) -> None:
        super().__init__(label, parent)
        self.step_id = step_id
        self._color_hex = color_hex
        self._icon_name = icon_name
        self._active = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        self.setCheckable(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(28)
        self.setIconSize(QSize(16, 16))
        if disabled:
            self.setEnabled(False)
        self._apply_style()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._apply_style()

    def _apply_style(self) -> None:
        c = self._color_hex
        shared = (
            "  border-radius: 4px;"
            "  text-align: left; padding: 4px 8px 4px 10px;"
            "  font-size: 11px;"
            "  margin: 0px 8px 0px 24px;"
        )
        if self._active:
            self.setStyleSheet(
                f"QPushButton {{"
                f"  background: {_rgba(c, 0.20)}; color: {c};"
                f"  border-left: 3px solid {c};"
                f"  font-weight: bold;"
                f"  {shared}"
                f"}}"
                f"QPushButton:hover {{ background: {_rgba(c, 0.30)}; }}"
            )
        elif self.isEnabled():
            self.setStyleSheet(
                f"QPushButton {{"
                f"  background: transparent; color: #cccccc;"
                f"  border: none;"
                f"  {shared}"
                f"}}"
                f"QPushButton:hover {{ background: rgba(255,255,255,0.06); }}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton {{"
                f"  background: transparent; color: #555555;"
                f"  border: none;"
                f"  {shared}"
                f"}}"
            )
        # Apply qtawesome icon colored to match state.
        if self._icon_name:
            if self._active:
                icon_color = c
            elif self.isEnabled():
                icon_color = "#999999"
            else:
                icon_color = "#555555"
            try:
                self.setIcon(qta.icon(self._icon_name, color=icon_color))
            except Exception:
                pass


class _NavSection(QWidget):
    """A collapsible section with a colored header and child items."""

    item_clicked = None  # set externally

    def __init__(self, name: str, color_hex: str, icon_name: str = "",
                 collapsed: bool = False, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._name = name
        self._color_hex = color_hex
        self._icon_name = icon_name
        self._expanded = not collapsed
        self._active = False
        self._items: list[_NavItem] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header: single button with icon inside + chevron label.
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(0)

        self._header = QPushButton(name.upper())
        self._header.setFlat(True)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header.setFixedHeight(30)
        self._header.setIconSize(QSize(18, 18))
        self._header.clicked.connect(self.toggle)
        header_row.addWidget(self._header, 1)

        self._chevron = QLabel()
        self._chevron.setFixedSize(20, 30)
        self._chevron.setAlignment(Qt.AlignCenter)
        header_row.addWidget(self._chevron)

        self._header_container = QWidget(self)
        self._header_container.setLayout(header_row)
        layout.addWidget(self._header_container)

        # Collapsible body.
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(1)
        layout.addWidget(self._body)

        self._body.setVisible(self._expanded)
        self._apply_header_style()

    def add_item(self, item: _NavItem) -> None:
        self._items.append(item)
        self._body_layout.addWidget(item)

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._apply_header_style()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._apply_header_style()

    def _apply_header_style(self) -> None:
        c = self._color_hex
        if self._active:
            container_bg = _rgba(c, 0.10)
        else:
            container_bg = "transparent"
        # Tint the entire header row (icon + text + chevron).
        self._header_container.setStyleSheet(
            f"background: {container_bg}; border-radius: 4px;"
        )
        self._header.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent; color: {c};"
            f"  border: none; border-radius: 4px;"
            f"  text-align: left; padding: 4px 8px;"
            f"  font-size: 12px; font-weight: bold;"
            f"}}"
            f"QPushButton:hover {{ background: {_rgba(c, 0.08)}; }}"
        )
        self._header.setText(self._name.upper())

        # Section icon inside button.
        if self._icon_name:
            try:
                self._header.setIcon(qta.icon(self._icon_name, color=c))
            except Exception:
                pass
        # Chevron indicator.
        chevron_icon_name = "fa5s.chevron-down" if self._expanded else "fa5s.chevron-right"
        try:
            chev = qta.icon(chevron_icon_name, color="#666666")
            self._chevron.setPixmap(chev.pixmap(QSize(10, 10)))
        except Exception:
            self._chevron.setText("▾" if self._expanded else "▸")


@dataclass(frozen=True)
class _StepIntent:
    objective: str
    outputs: str
    success: str
    actions: tuple[str, ...]


def _make_resource_bar(color: str) -> QProgressBar:
    """Create a slim, tqdm-style progress bar with the given accent color."""
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setFixedHeight(14)
    bar.setTextVisible(True)
    bar.setFormat("%p%")
    bar.setStyleSheet(
        f"QProgressBar {{"
        f"  background: rgba(255, 255, 255, 0.06);"
        f"  border: none;"
        f"  border-radius: 3px;"
        f"  text-align: center;"
        f"  font-size: 9px;"
        f"  color: #cccccc;"
        f"}}"
        f"QProgressBar::chunk {{"
        f"  background: {color};"
        f"  border-radius: 3px;"
        f"}}"
    )
    return bar


def _read_gpu_utilization() -> Optional[int]:
    """Query GPU utilization % via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


class _StatusCard(QWidget):
    """Compact at-a-glance project status + live resource monitors."""

    #: Scoped with an object name (B1): the un-scoped version cascaded into
    #: every child QLabel, so each text row drew its own border + tint.  The
    #: `#epicycleStatusCard` selector paints the box only.
    _CARD_STYLE = (
        "QWidget#epicycleStatusCard {"
        "  background: rgba(255, 255, 255, 0.025);"
        "  border-radius: 4px;"
        "  border: 1px solid rgba(255, 255, 255, 0.06);"
        "}"
    )
    _LABEL_DIM = "color: #888888; font-size: 10px;"
    _LABEL_BRIGHT = "color: #dddddd; font-size: 11px;"

    #: The little rounded box each TEXT row sits in.  Pre-B1 this came for
    #: free: `_CARD_STYLE` was set on the card with no selector, so Qt
    #: cascaded it into every child QLabel.  B1 scoped the card to
    #: `QWidget#epicycleStatusCard` to stop the border also landing on the
    #: RAM / CPU / GPU captions -- and the chips went with it.  They are back,
    #: drawn deliberately and addressed by each row's OWN object name, so the
    #: bars stay exactly as they are and nothing cascades anywhere.
    _ROW_CHIP = (
        "  background: rgba(255, 255, 255, 0.025);"
        "  border: 1px solid rgba(255, 255, 255, 0.06);"
        "  border-radius: 4px;"
        "  padding: 3px 6px;"
    )
    _WORKSPACE_STYLE = (
        "QLabel#epicycleWorkspaceRow {"
        + _ROW_CHIP +
        "  color: #eeeeee; font-size: 12px; font-weight: bold;"
        "}"
    )
    #: Same red as the banner's state-1 line (mockup legend).
    _WORKSPACE_NONE_STYLE = (
        "QLabel#epicycleWorkspaceRow {"
        + _ROW_CHIP +
        "  color: #e07070; font-size: 12px; font-weight: bold;"
        "}"
    )
    _STATUS_ROW_STYLE = (
        "QLabel#epicycleStatusRow {"
        + _ROW_CHIP +
        "  color: #dddddd; font-size: 11px;"
        "}"
    )

    def __init__(self) -> None:
        super().__init__()
        self._labels: Dict[str, QLabel] = {}
        self._bars: Dict[str, QProgressBar] = {}
        self._gpu_available: Optional[bool] = None
        self._build_ui()
        self._start_resource_polling()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        card = QWidget()
        card.setObjectName("epicycleStatusCard")
        card.setStyleSheet(self._CARD_STYLE)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(6)

        # ── Workspace name (prominent) ──
        self._labels["workspace"] = QLabel("-")
        self._labels["workspace"].setObjectName("epicycleWorkspaceRow")
        self._labels["workspace"].setStyleSheet(self._WORKSPACE_STYLE)
        self._labels["workspace"].setWordWrap(True)
        card_layout.addWidget(self._labels["workspace"])

        # ── Compact status line ──
        self._labels["status"] = QLabel("-")
        self._labels["status"].setObjectName("epicycleStatusRow")
        self._labels["status"].setStyleSheet(self._STATUS_ROW_STYLE)
        self._labels["status"].setWordWrap(True)
        card_layout.addWidget(self._labels["status"])

        # The old "Lat: - | Ax: -" models row is gone (B1): it was blank
        # most of the time and described a step that no longer exists.

        # ── Resource bars ──
        card_layout.addSpacing(4)

        # RAM — full-width row
        ram_row = QHBoxLayout()
        ram_row.setSpacing(6)
        ram_lbl = QLabel("RAM")
        ram_lbl.setStyleSheet(self._LABEL_DIM)
        ram_lbl.setFixedWidth(36)
        self._bars["ram"] = _make_resource_bar("#ff6b9d")
        ram_row.addWidget(ram_lbl)
        ram_row.addWidget(self._bars["ram"])
        card_layout.addLayout(ram_row)

        # CPU + GPU — side by side
        cpu_gpu_row = QHBoxLayout()
        cpu_gpu_row.setSpacing(8)

        cpu_lbl = QLabel("CPU")
        cpu_lbl.setStyleSheet(self._LABEL_DIM)
        cpu_lbl.setFixedWidth(28)
        self._bars["cpu"] = _make_resource_bar("#00d4aa")

        gpu_lbl = QLabel("GPU")
        gpu_lbl.setStyleSheet(self._LABEL_DIM)
        gpu_lbl.setFixedWidth(28)
        self._bars["gpu"] = _make_resource_bar("#ffd93d")

        cpu_gpu_row.addWidget(cpu_lbl)
        cpu_gpu_row.addWidget(self._bars["cpu"])
        cpu_gpu_row.addSpacing(4)
        cpu_gpu_row.addWidget(gpu_lbl)
        cpu_gpu_row.addWidget(self._bars["gpu"])
        card_layout.addLayout(cpu_gpu_row)

        root.addWidget(card)

    def _start_resource_polling(self) -> None:
        self._resource_timer = QTimer(self)
        self._resource_timer.setInterval(2000)
        self._resource_timer.timeout.connect(self._poll_resources)
        self._resource_timer.start()
        # Initial read.
        self._poll_resources()

    def _poll_resources(self) -> None:
        try:
            self._bars["cpu"].setValue(int(psutil.cpu_percent(interval=None)))
        except Exception:
            pass
        try:
            mem = psutil.virtual_memory()
            self._bars["ram"].setValue(int(mem.percent))
        except Exception:
            pass
        # GPU — lazy-detect availability, then poll via nvidia-smi.
        if self._gpu_available is None:
            gpu_val = _read_gpu_utilization()
            self._gpu_available = gpu_val is not None
            if gpu_val is not None:
                self._bars["gpu"].setValue(gpu_val)
            else:
                self._bars["gpu"].setValue(0)
                self._bars["gpu"].setFormat("n/a")
                self._bars["gpu"].setEnabled(False)
        elif self._gpu_available:
            gpu_val = _read_gpu_utilization()
            if gpu_val is not None:
                self._bars["gpu"].setValue(gpu_val)

    def update_values(self, values: Dict[str, str]) -> None:
        """Paint the two text rows (B1).

        Row 1 is the SHARED workspace's folder name (bold), with the full
        path — plus any per-step divergences — as its tooltip, or a red
        "No workspace".  Row 2 is one line that is true on whichever step
        the user is on.  The RAM / CPU / GPU bars are untouched.
        """
        workspace = values.get("workspace_path", "") or ""
        status_line = values.get("status_line", "") or ""
        diverged_note = values.get("diverged_note", "") or ""

        if workspace and workspace != "-":
            display_name = (
                os.path.basename(workspace.rstrip("/\\")) or workspace)
            self._labels["workspace"].setStyleSheet(self._WORKSPACE_STYLE)
            self._labels["workspace"].setText(display_name)
            tooltip = workspace
            if diverged_note:
                tooltip = f"{workspace}\n\n{diverged_note}"
            self._labels["workspace"].setToolTip(tooltip)
        else:
            self._labels["workspace"].setStyleSheet(
                self._WORKSPACE_NONE_STYLE)
            self._labels["workspace"].setText("No workspace")
            self._labels["workspace"].setToolTip("")

        self._labels["status"].setText(status_line)


class _HistoryDrawer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        toolbar = QHBoxLayout()

        title = QLabel("Run + Artifact History")
        title.setStyleSheet("font-weight: bold;")
        self.clear_button = QPushButton("Clear")

        toolbar.addWidget(title)
        toolbar.addStretch(1)
        toolbar.addWidget(self.clear_button)

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(70)
        self.history_list.setMaximumHeight(110)

        root.addLayout(toolbar)
        root.addWidget(self.history_list)

        self.clear_button.clicked.connect(self.history_list.clear)

    def add_event(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_list.insertItem(0, f"[{timestamp}] {message}")


class EpiCycleWorkflowShell(QWidget):
    """Phase J workflow shell.

    Keeps existing step widgets/controllers, but rehosts them inside the new shell:
    - left sidebar (workflow navigator + global state)
    - center step workspace
    - bottom history drawer
    """

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # ── The one shared workspace ──
        # `_workspace_path` is now AUTHORITATIVE, not a scrape cache: it is
        # written only by `set_workspace` (from a step's Browse) and by the
        # remembered-workspace suggestion read below.
        self._workspace_path = ""
        self._workspace_source_step = ""
        #: step_id -> "browsed" (the user picked it there) | "adopted"
        #: (inherited from the shared workspace).  A step with no entry has
        #: no path of its own.  This is what makes a step amber or not.
        self._step_path_origin: Dict[str, str] = {}
        #: step_id -> its own path, while it differs from the shared one
        self._diverged_steps: Dict[str, str] = {}
        #: (step_id, its path, the shared path) already announced
        self._diverged_logged: set = set()
        #: Adopt only ever runs on a user-driven step entry, never during the
        #: `_select_step` that `__init__` performs.
        self._shell_ready = False
        # A SUGGESTION, kept SEPARATE from the shared workspace: it pre-fills
        # Step 1's box and nothing else.  A workspace becomes *shared* only
        # when the user browses to it, in any step, so
        # a remembered path must not make the shell claim one.  No adopt, no
        # state load, no source restore, no workspace I/O.
        self._suggested_workspace = _workspace_facts.load_last_workspace()
        self._sidebar_dock_name = "EpiCycle Workflow Sidebar"
        self._history_dock_name = "EpiCycle Run + Artifact History"
        self._sidebar_container: Optional[QWidget] = None
        self._sidebar_dock = None
        self._history_dock = None
        self._aux_docks_ready = False
        self._step_specs = self._build_step_specs()
        self._step_widgets: Dict[str, QWidget] = {}
        self._step_index_by_id: Dict[str, int] = {}
        self._step_intents = self._build_step_intents()
        self._active_step_id: Optional[str] = None
        self._viewer_canvas_snapshot: Optional[Dict[str, object]] = None
        self._subviewer_steps = {"roi_hitl_2d", "reconstruct_curate_3d", "apply_3d"}
        # Map each step to its section's accent color for themed selection.
        self._step_section_color: Dict[str, str] = {}
        for step in self._step_specs:
            self._step_section_color[step.step_id] = _SECTION_COLORS.get(
                step.section, "#2d6aa0"
            )
        # Populated by _build_left_panel.
        self._nav_sections: Dict[str, _NavSection] = {}
        self._nav_items: Dict[str, _NavItem] = {}

        # Global layer lifecycle manager — tracks layer ownership by context.
        from .layer_lifecycle_manager import LayerLifecycleManager
        self.layer_manager = LayerLifecycleManager(napari_viewer)

        self._build_ui()
        self._wire_state_signals()
        self._init_state_polling()
        # Every step controller now exists and the startup `_select_step` has
        # run, so a step entry from here on is the user's doing: open the
        # adopt gate.
        self._shell_ready = True

    def _build_step_specs(self):
        return [
            _WorkflowStep(
                step_id="project_setup",
                title="Project Setup & ROI Extraction",
                subtitle="Create or open a workspace and extract ROI patches.",
                section="Setup",
                widget_factory=lambda viewer: DataPreparationWidget(viewer=viewer),
            ),
            _WorkflowStep(
                step_id="roi_hitl_2d",
                title="2D Model HITL Bootstrapping",
                subtitle="Iteratively curate slices and train lateral/axial round models.",
                section="Build",
                widget_factory=lambda viewer: TrainingAnnotationHITL2D(viewer=viewer),
            ),
            _WorkflowStep(
                step_id="reconstruct_curate_3d",
                title="3D Reconstruction & Curation",
                subtitle="Predict, reconstruct, QC-analyze, and curate 3D volumes.",
                section="Build",
                widget_factory=lambda viewer: FusionReconstructionCuration(viewer=viewer),
            ),
            # Finalize Model Pack — archived; model consolidation will be
            # absorbed into 3D Reconstruction & Curation.
            _WorkflowStep(
                step_id="apply_3d",
                title="3D Stack Segmentation",
                subtitle="Run stack-level 3D segmentation with selected model pack.",
                section="Apply",
                widget_factory=lambda viewer: SegmentStack(viewer=viewer),
            ),
            _WorkflowStep(
                step_id="whole_tissue",
                title="3D Scaled Segmentation",
                subtitle="Configure tiled segmentation jobs for larger volumes.",
                section="Apply",
                widget_factory=lambda viewer: LargeScaleSegmentation(viewer=viewer),
            ),
        ]

    def _build_step_intents(self) -> Dict[str, _StepIntent]:
        return {
            "project_setup": _StepIntent(
                objective="Define project workspace and capture ROI assets that downstream steps depend on.",
                outputs="Workspace path + ROI patch set + ROI metadata.",
                success="ROIs are extracted, reviewed, and export target is valid.",
                actions=(
                    "Select image layers and ROI drawing layer.",
                    "Draw/free or fixed-size ROI boxes and extract patches.",
                    "Review ROI list and export to the project workspace.",
                ),
            ),
            "roi_hitl_2d": _StepIntent(
                objective="Curate representative slices and run round-based HITL model updates.",
                outputs="Round training artifacts + inference outputs + benchmark summaries.",
                success="At least one completed round with accepted/refined annotations.",
                actions=(
                    "Set working directory and load a stack/view.",
                    "Segment current slices and curate accepts/rejects.",
                    "Stage slices and run round training + benchmarking.",
                ),
            ),
            "reconstruct_curate_3d": _StepIntent(
                objective="Predict, reconstruct, QC-analyze, curate, and optionally refine models from 3D volumes.",
                outputs="3D reconstruction manifests + QC feature CSVs + curated labels + refined models.",
                success="Reconstruction completes, QC passes, and curated labels are finalized.",
                actions=(
                    "Predict and reconstruct (step-by-step or batch all patches).",
                    "Extract QC features, threshold-filter, and triage flagged labels.",
                    "Curate results and finalize labels.",
                    "Extract training pairs from curated 3D and fine-tune models.",
                ),
            ),
            "finalize_model": _StepIntent(
                objective="Promote the best round artifacts into a reusable model pack.",
                outputs="Final model pack registry artifact.",
                success="Pack is generated and appears in workspace model registry.",
                actions=(
                    "Review candidate workspace models + benchmark context.",
                    "Choose lateral/axial model pair or imported models.",
                    "Finalize and validate the model pack artifact.",
                ),
            ),
            "apply_3d": _StepIntent(
                objective="Run stack-level segmentation using a finalized model pack.",
                outputs="Segmentation run manifest + output labels volume.",
                success="Run completes and output can be loaded/reviewed in viewer.",
                actions=(
                    "Set workspace, input stack, and model pack.",
                    "Run segmentation job.",
                    "Review run artifact and load selected output.",
                ),
            ),
            "whole_tissue": _StepIntent(
                objective="Configure and launch large-scale tiled segmentation jobs.",
                outputs="Job spec/bundle + runtime outputs and run records.",
                success="Job artifacts are generated and outputs are trackable/loadable.",
                actions=(
                    "Provide input/model and select local or SLURM mode.",
                    "Set tile/overlap/runtime profile.",
                    "Generate/run job and review outputs.",
                ),
            ),
        }

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        center_panel = self._build_center_panel()
        root.addWidget(center_panel, 1)

        self.history_drawer = _HistoryDrawer()

        self.state_panel = _StatusCard()
        self._sidebar_container = self._build_sidebar_container()
        QTimer.singleShot(0, self._ensure_aux_docks_attached)

        first_step_id = self._step_specs[0].step_id
        self._select_step(first_step_id)
        # AFTER the startup selection, deliberately: Step 1's `on_step_enter`
        # has already run against an empty box, so the remembered path cannot
        # drag a source restore or a migration prompt into start-up.
        self._prefill_step1_from_suggestion()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._ensure_aux_docks_attached()

    def _ensure_aux_docks_attached(self) -> None:
        if self._aux_docks_ready:
            return
        self._cleanup_duplicate_docks(self._sidebar_dock_name)
        self._cleanup_duplicate_docks(self._history_dock_name)
        self._attach_sidebar_dock()
        self._attach_history_dock()

        host_dock = self._find_host_dock()
        history_dock = self._resolve_history_dock()
        if host_dock is not None and history_dock is not None and host_dock is not history_dock:
            self._place_history_below_host(host_dock, history_dock)
            self._set_default_history_split(host_dock, history_dock)
            self._aux_docks_ready = True
        else:
            # no retries; showEvent + next user open keeps behavior deterministic.
            self._aux_docks_ready = True

        # Defer sidebar reorder so napari's own docks are fully settled first.
        if self._sidebar_dock is not None:
            QTimer.singleShot(100, self._position_sidebar_top_left)

    @staticmethod
    def _make_h_divider() -> QWidget:
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #555555;")
        return line

    def _build_sidebar_container(self) -> QWidget:
        # Inner content widget.
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(4, 4, 4, 4)
        inner_layout.setSpacing(6)

        # Status card on top.
        inner_layout.addWidget(self.state_panel)

        # Full-width divider.
        inner_layout.addWidget(self._make_h_divider())

        # Workflow nav tree.
        workflow_panel = self._build_left_panel()
        inner_layout.addWidget(workflow_panel, 1)

        # Wrap in scroll area so small screens can scroll.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(280)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical {"
            "  background: transparent; width: 6px;"
            "}"
            "QScrollBar::handle:vertical {"
            "  background: rgba(255, 255, 255, 0.15);"
            "  border-radius: 3px; min-height: 20px;"
            "}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
            "  height: 0px;"
            "}"
        )
        return scroll

    def _dock_registry(self):
        window = getattr(self.viewer, "window", None)
        if window is None:
            return None
        registry = getattr(window, "dock_widgets", None)
        if registry is None:
            registry = getattr(window, "_dock_widgets", None)
        return registry

    def _find_docks_by_title(self, title: str):
        qt_main = getattr(getattr(self.viewer, "window", None), "_qt_window", None)
        if qt_main is None:
            return []
        docks = []
        try:
            for dock in qt_main.findChildren(QDockWidget):
                if str(dock.windowTitle()) == str(title):
                    docks.append(dock)
        except Exception:
            return []
        return docks

    def _cleanup_duplicate_docks(self, title: str) -> None:
        docks = self._find_docks_by_title(title)
        if len(docks) <= 1:
            return
        keep = docks[0]
        for dock in docks[1:]:
            try:
                dock.hide()
                dock.setParent(None)
                dock.deleteLater()
            except Exception:
                continue
        if title == self._sidebar_dock_name:
            self._sidebar_dock = keep
        if title == self._history_dock_name:
            self._history_dock = keep

    def _resolve_sidebar_dock(self):
        docks = self._find_docks_by_title(self._sidebar_dock_name)
        return docks[0] if docks else None

    def _resolve_history_dock(self):
        docks = self._find_docks_by_title(self._history_dock_name)
        return docks[0] if docks else None

    def _attach_sidebar_dock(self) -> None:
        if self._sidebar_container is None:
            return
        window = getattr(self.viewer, "window", None)
        if window is None:
            return
        existing = self._resolve_sidebar_dock()
        if existing is not None:
            self._sidebar_dock = existing
            return
        try:
            self._sidebar_dock = window.add_dock_widget(
                self._sidebar_container,
                name=self._sidebar_dock_name,
                area="left",
            )
            if self._sidebar_dock is not None and hasattr(self._sidebar_dock, "setMinimumWidth"):
                self._sidebar_dock.setMinimumWidth(280)
        except Exception:
            self._sidebar_dock = None

    def _position_sidebar_top_left(self) -> None:
        """Reorder docks so the sidebar sits at the top of the left dock area,
        above napari's built-in 'layer controls' and 'layer list' docks."""
        if self._sidebar_dock is None:
            return
        qt_main = getattr(
            getattr(self.viewer, "window", None), "_qt_window", None
        )
        if qt_main is None:
            return
        try:
            # Resolve raw QDockWidgets by title (napari returns proxies
            # from add_dock_widget, but splitDockWidget needs the real Qt
            # objects which findChildren provides).
            sidebar_real = None
            layer_controls = None
            for dock in qt_main.findChildren(QDockWidget):
                title = dock.windowTitle()
                if title == self._sidebar_dock_name:
                    sidebar_real = dock
                elif title == "layer controls":
                    layer_controls = dock

            if sidebar_real is None or layer_controls is None:
                return

            # Target order (top→bottom): sidebar → layer controls → layer list
            # splitDockWidget(A, B, Vertical) splits A's space: A on top, B below.
            # Step 1: move sidebar into controls' space (controls top, sidebar below)
            qt_main.splitDockWidget(layer_controls, sidebar_real, Qt.Vertical)
            # Step 2: flip — split sidebar's space: sidebar top, controls below
            qt_main.splitDockWidget(sidebar_real, layer_controls, Qt.Vertical)
        except Exception:
            return

    def _attach_history_dock(self) -> None:
        window = getattr(self.viewer, "window", None)
        if window is None:
            return
        existing = self._resolve_history_dock()
        if existing is not None:
            self._history_dock = existing
            try:
                if hasattr(existing, "show"):
                    existing.show()
            except Exception:
                pass
            return
        try:
            self._history_dock = window.add_dock_widget(
                self.history_drawer,
                name=self._history_dock_name,
                area="right",
            )
            if self._history_dock is not None:
                if hasattr(self._history_dock, "setMinimumHeight"):
                    self._history_dock.setMinimumHeight(90)
                if hasattr(self._history_dock, "setMaximumHeight"):
                    self._history_dock.setMaximumHeight(170)
        except Exception:
            self._history_dock = None

    def _find_host_dock(self):
        """Find the QDockWidget that contains this shell widget."""
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QDockWidget):
                return parent
            parent = parent.parentWidget()
        return None

    def _place_history_below_host(self, host_dock, history_dock) -> None:
        if host_dock is None or history_dock is None:
            return
        if host_dock is history_dock:
            return
        qt_main = getattr(getattr(self.viewer, "window", None), "_qt_window", None)
        if qt_main is None:
            return
        try:
            qt_main.splitDockWidget(host_dock, history_dock, Qt.Vertical)
        except Exception:
            return

    def _set_default_history_split(self, host_dock, history_dock) -> None:
        if host_dock is None or history_dock is None:
            return
        qt_main = getattr(getattr(self.viewer, "window", None), "_qt_window", None)
        if qt_main is None:
            return
        try:
            qt_main.resizeDocks([host_dock, history_dock], [760, 130], Qt.Vertical)
        except Exception:
            return

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._nav_sections: Dict[str, _NavSection] = {}
        self._nav_items: Dict[str, _NavItem] = {}

        # ── Pipeline sections (expanded) ──
        current_section = None
        for step in self._step_specs:
            if step.section != current_section:
                current_section = step.section
                color = _SECTION_COLORS.get(step.section, "#aaaaaa")
                icon_name = _SECTION_ICONS.get(step.section, "")
                section_w = _NavSection(step.section, color, icon_name,
                                        collapsed=False, parent=panel)
                self._nav_sections[step.section] = section_w
                layout.addWidget(section_w)

            color = self._step_section_color.get(step.step_id, "#aaaaaa")
            step_icon = _STEP_ICONS.get(step.step_id, "")
            # QPushButton treats '&' as mnemonic prefix — escape for display.
            btn_title = step.title.replace("&", "&&")
            item = _NavItem(btn_title, color, step.step_id,
                            icon_name=step_icon, parent=panel)
            item.clicked.connect(
                lambda _checked=False, sid=step.step_id: self._select_step(sid)
            )
            section_w.add_item(item)
            self._nav_items[step.step_id] = item

        # release hide list 2026-10-12: Step 5 (3D Scaled Segmentation) is not
        # released; hidden, still built so step indices never shift.
        if "whole_tissue" in self._nav_items:
            self._nav_items["whole_tissue"].setVisible(False)

        # ── Full-width divider ── (release hide list 2026-10-12: the
        # placeholder sections below are hidden, so no divider / spacing)
        divider = self._make_h_divider()
        layout.addWidget(divider)
        divider.setVisible(False)

        # ── Utility sections (collapsed, disabled placeholders) ──
        utility_sections = [
            ("Visualization", ["Data Explorer", "Analysis Dashboard"]),
            ("Animations", ["Timelapse Builder", "Fly-Through"]),
            ("Tools", ["3D Model Training", "Multiview Annotator"]),
        ]
        for sec_name, items in utility_sections:
            color = _SECTION_COLORS.get(sec_name, "#aaaaaa")
            icon_name = _SECTION_ICONS.get(sec_name, "")
            section_w = _NavSection(sec_name, color, icon_name,
                                    collapsed=True, parent=panel)
            self._nav_sections[sec_name] = section_w
            for label in items:
                item = _NavItem(label, color, disabled=True, parent=panel)
                section_w.add_item(item)
            layout.addWidget(section_w)
            section_w.setVisible(False)  # release hide list 2026-10-12

        layout.addStretch(1)
        return panel

    # Steps with full internal accordion structure — hide subtitle + intent.
    _self_describing_steps = {
        "project_setup", "roi_hitl_2d", "reconstruct_curate_3d",
        "apply_3d", "whole_tissue",
    }

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header bar: section-colored accent + title + subtitle ──
        self._header_bar = QWidget()
        self._header_bar.setFixedHeight(48)
        header_layout = QHBoxLayout(self._header_bar)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        # Left accent stripe (colored by active section).
        self._header_accent = QWidget()
        self._header_accent.setFixedWidth(4)
        self._header_accent.setStyleSheet("background: #888888;")

        # Title + subtitle stacked.
        header_text = QWidget()
        header_text_layout = QVBoxLayout(header_text)
        header_text_layout.setContentsMargins(12, 6, 12, 6)
        header_text_layout.setSpacing(2)

        self.step_title = QLabel("")
        self.step_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #eeeeee;"
        )
        self.step_subtitle = QLabel("")
        self.step_subtitle.setWordWrap(True)
        self.step_subtitle.setStyleSheet(
            "font-size: 10px; color: #999999;"
        )

        header_text_layout.addWidget(self.step_title)
        header_text_layout.addWidget(self.step_subtitle)

        header_layout.addWidget(self._header_accent)
        header_layout.addWidget(header_text, 1)

        # ── Banner strips under the step title ──
        # Strip 1 carries the workspace state (red / green / amber); strip 2
        # carries an ordinary advisory ("No completed HITL training round…")
        # underneath it, so each keeps its own colour and the advisory can
        # come and go without moving the workspace line around.
        self.guidance_label = _BannerLabel()
        self.advisory_label = _BannerLabel()

        # ── Intent group (hidden for self-describing steps) ──
        self.intent_group = QGroupBox("Step Intent")
        self.intent_group.setStyleSheet(
            "QGroupBox { font-size: 11px; color: #aaaaaa; }"
        )
        intent_layout = QFormLayout(self.intent_group)
        intent_layout.setContentsMargins(12, 8, 12, 8)
        intent_layout.setVerticalSpacing(4)
        self.intent_objective = QLabel("")
        self.intent_objective.setWordWrap(True)
        self.intent_outputs = QLabel("")
        self.intent_outputs.setWordWrap(True)
        self.intent_success = QLabel("")
        self.intent_success.setWordWrap(True)
        self.intent_actions = QLabel("")
        self.intent_actions.setWordWrap(True)
        for lbl in (self.intent_objective, self.intent_outputs,
                     self.intent_success, self.intent_actions):
            lbl.setStyleSheet("font-size: 10px; color: #cccccc;")
        intent_layout.addRow("Objective:", self.intent_objective)
        intent_layout.addRow("Outputs:", self.intent_outputs)
        intent_layout.addRow("Success:", self.intent_success)
        intent_layout.addRow("Key actions:", self.intent_actions)

        # ── Step content stack ──
        self.step_stack = QStackedWidget()
        self.step_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        for step in self._step_specs:
            legacy_widget = step.widget_factory(self.viewer)
            self._step_widgets[step.step_id] = legacy_widget

            # Inject global layer manager into each controller.
            controller = getattr(legacy_widget, "controller", None)
            if controller is not None:
                controller.layer_lifecycle_manager = self.layer_manager
                # Step 2 training-lock (2026-05-27): inject workflow_shell
                # ref so the Step 2 controller can lock the side-nav
                # during training.  Other step controllers may also
                # benefit later (Step 3/4/5 long-running ops) — the ref is
                # injected universally; only Step 2 currently uses it.
                controller._workflow_shell = self

            container = self._build_step_container(step, legacy_widget)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            scroll.setWidget(container)

            index = self.step_stack.addWidget(scroll)
            self._step_index_by_id[step.step_id] = index

        layout.addWidget(self._header_bar)
        layout.addWidget(self.guidance_label)
        layout.addWidget(self.advisory_label)
        # Intent group removed from layout — too verbose, clutters the chrome.
        # Widgets kept alive so _update_step_intent doesn't error.
        self.intent_group.setVisible(False)
        layout.addWidget(self.step_stack, 1)
        return panel

    def _build_step_container(self, step: _WorkflowStep, legacy_widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # These steps already have full internal structure, avoid redundant shell wrappers.
        if step.step_id in self._self_describing_steps:
            layout.addWidget(legacy_widget)
            return container

        if self._show_shell_hints(step.step_id):
            section_hints = self._build_section_hints(step.step_id)

            inputs_group = QGroupBox("Inputs")
            inputs_layout = QVBoxLayout(inputs_group)
            inputs_hint = QLabel(section_hints["inputs"])
            inputs_hint.setWordWrap(True)
            inputs_layout.addWidget(inputs_hint)

            config_group = QGroupBox("Config")
            config_layout = QVBoxLayout(config_group)
            config_hint = QLabel(section_hints["config"])
            config_hint.setWordWrap(True)
            config_layout.addWidget(config_hint)

            layout.addWidget(inputs_group)
            layout.addWidget(config_group)

        work_group = QGroupBox("Run / Preview / Review / Commit")
        work_layout = QVBoxLayout(work_group)
        work_layout.addWidget(legacy_widget)

        layout.addWidget(work_group)
        return container

    def _show_shell_hints(self, step_id: str) -> bool:
        return step_id not in self._self_describing_steps

    def _build_section_hints(self, step_id: str) -> Dict[str, str]:
        hints: Dict[str, Dict[str, str]] = {
            "project_setup": {
                "inputs": "Dataset layers, ROI draw layers, and export workspace target.",
                "config": "ROI extraction size, selection mode, and review/export settings.",
            },
            "roi_hitl_2d": {
                "inputs": "Working directory, selected patch stack, and orthoview selection.",
                "config": "Training parameters, model type/device, and curation refinement settings.",
            },
            "reconstruct_curate_3d": {
                "inputs": "Working directory, selected stacks, model artifacts, and QC feature scope.",
                "config": "Prediction, reconstruction, QC thresholds, and model refinement parameters.",
            },
            "finalize_model": {
                "inputs": "Workspace model artifacts, benchmark summaries, and imported model references.",
                "config": "Lateral/axial model choices and final pack notes.",
            },
            "apply_3d": {
                "inputs": "Working directory, input stack, and model pack.",
                "config": "Run naming and execution options.",
            },
            "whole_tissue": {
                "inputs": "Working directory, input volume, and model artifact.",
                "config": "Tile size/overlap and runtime profile (local/slurm settings).",
            },
        }
        return hints.get(
            step_id,
            {
                "inputs": "Set required inputs for this step.",
                "config": "Configure step parameters before running.",
            },
        )

    def _apply_nav_highlight(self, step_id: str, section: str) -> None:
        """Apply section-themed highlight to the active step and its header."""
        # Reset all items and sections.
        for item in self._nav_items.values():
            item.set_active(False)
        for sec_w in self._nav_sections.values():
            sec_w.set_active(False)

        # Activate the selected item and its section.
        item = self._nav_items.get(step_id)
        if item is not None:
            item.set_active(True)
        sec_w = self._nav_sections.get(section)
        if sec_w is not None:
            sec_w.set_active(True)

    def _get_step_controller(self, step_id: str):
        """Return the controller for a given step, or None."""
        widget = self._step_widgets.get(step_id)
        return getattr(widget, "controller", None)

    def lock_step_nav(self, locked: bool, owner: str = "default") -> None:
        """Lock/unlock the workflow-shell side-nav buttons.

        Owner-keyed: the nav is locked
        while the set of owners is not empty, and a call adds or removes
        ONLY its own owner, so one step's job ending can never release
        another step's lock.  Steps 2 and 3 call it without ``owner`` (the
        ``"default"`` owner, unchanged); Step 4 passes ``owner="step4"``.

        Step 2 training-lock (2026-05-27): called from
        ``TrainingAnnotationController._set_training_ui_locked``.  While
        locked, all pipeline-section step buttons (Project Setup / ROI +
        HITL 2D / Reconstruct + Curate 3D / Apply 3D / Whole Tissue) are
        disabled so the user cannot navigate away from the running step.
        The current step's content stays visible; only sideways navigation
        is blocked.

        Utility-section items (Visualization / Animations / Tools) are
        already disabled at idle so we leave them alone.

        Idempotent: calling lock_step_nav(True) twice is a no-op; calling
        lock_step_nav(False) restores items to enabled regardless of
        prior state.

        Robust against UI churn: if `self._nav_items` doesn't exist yet
        (called before _build_left_panel), this is a no-op.

        Audit fix (2026-05-27, second-pass review #3): Qt's
        ``setEnabled(False)`` does NOT cancel mouse-click events that
        are already queued in the event loop.  A user clicking a step-
        nav button at the exact tick training launches could still
        navigate away.  Belt-and-suspenders:
          1. ``setEnabled(False)`` blocks NEW clicks
          2. ``blockSignals(True)`` cancels queued clicked() signal emission
          3. Setting ``self._step_nav_locked`` so ``_select_step`` can
             early-return if a slot fires anyway (e.g. via programmatic
             ``item.clicked.emit()``)
        """
        owners = getattr(self, "_step_nav_owners", None)
        if not isinstance(owners, set):
            owners = set()
            self._step_nav_owners = owners
        owner = str(owner or "default")
        if locked:
            owners.add(owner)
        else:
            owners.discard(owner)
        locked = bool(owners)
        items = getattr(self, "_nav_items", None)
        if not items:
            return
        enabled = not bool(locked)
        # Audit fix: drive a shell-side flag _select_step honors as a
        # defensive guard (in case the signal-block + setEnabled both
        # somehow get bypassed).
        self._step_nav_locked = bool(locked)
        print(f"[STEP2-LOCK] workflow_shell.lock_step_nav owner={owner} "
              f"locked={locked} owners={sorted(owners)} items={len(items)}")
        for step_id, item in items.items():
            if hasattr(item, "setEnabled"):
                try:
                    item.setEnabled(enabled)
                except Exception as exc:
                    print(f"[STEP2-LOCK] failed to set step={step_id!r} "
                          f"enabled={enabled}: {type(exc).__name__}: {exc}")
            # Audit fix: blockSignals cancels queued clicked() emission
            # so a click in flight at the lock-instant doesn't fire the
            # _select_step slot post-lock.
            if hasattr(item, "blockSignals"):
                try:
                    item.blockSignals(bool(locked))
                except Exception as exc:
                    print(f"[STEP2-LOCK] failed to blockSignals step={step_id!r} "
                          f"locked={locked}: {type(exc).__name__}: {exc}")

    def _select_step(self, step_id: str) -> None:
        if step_id not in self._step_index_by_id:
            return

        # Audit fix (2026-05-27, second-pass review #3): defensive guard
        # against queued click events during a training-lock window.
        # `lock_step_nav` already does setEnabled(False) + blockSignals,
        # but if either is bypassed (programmatic emit, race window,
        # future code path), this guard refuses the navigation.
        if getattr(self, "_step_nav_locked", False):
            print(f"[STEP2-LOCK] _select_step({step_id!r}) BLOCKED — "
                  "step nav is locked (training in flight)")
            return

        previous_step = self._active_step_id
        step = next((s for s in self._step_specs if s.step_id == step_id), None)
        if step is None:
            return

        # --- Step exit hook ---
        if previous_step and previous_step != step_id:
            prev_ctrl = self._get_step_controller(previous_step)
            if prev_ctrl is not None and hasattr(prev_ctrl, "on_step_exit"):
                try:
                    stay = prev_ctrl.on_step_exit() is False
                except Exception:
                    stay = False
                if stay:
                    # A step may veto
                    # its own exit by returning False (Step 4 does when its
                    # save before leaving failed).  Every other step returns
                    # None, so nothing changes for them.
                    print(f"[STEP-NAV] {previous_step!r} kept: its exit "
                          f"hook refused the change to {step_id!r}")
                    return

        index = self._step_index_by_id[step_id]
        self.step_stack.setCurrentIndex(index)

        self.step_title.setText(step.title)
        self.step_subtitle.setText(step.subtitle)
        # Hide subtitle for steps with full internal structure.
        self.step_subtitle.setVisible(step_id not in self._self_describing_steps)
        # Color the header accent stripe to match the active section.
        accent_color = self._step_section_color.get(step_id, "#888888")
        self._header_accent.setStyleSheet(f"background: {accent_color};")
        self._refresh_step_guidance(step_id)
        self._update_step_intent(step_id)

        self._apply_nav_highlight(step_id, step.section)

        self._sync_viewer_canvas_mode(step_id=step_id, previous_step=previous_step)
        self._active_step_id = step_id

        new_ctrl = self._get_step_controller(step_id)

        # --- Shared-workspace adopt ---
        # Before `on_step_enter`, so the step enters with its workspace
        # already set.  `_adopt_into_step` is gated on `_shell_ready` (so it
        # cannot fire during __init__'s own `_select_step`, when the
        # controllers are half-wired), skips a step the user browsed
        # elsewhere, and RE-adopts a step still holding a path it merely
        # inherited from a workspace that has since moved.
        self._adopt_into_step(step_id)
        self._refresh_divergence()

        # --- Step enter hook ---
        if new_ctrl is not None and hasattr(new_ctrl, "on_step_enter"):
            try:
                new_ctrl.on_step_enter()
            except Exception:
                pass

        # Re-render the banner now that adopt + enter have had their say.
        self._refresh_step_guidance(step_id)

        self.history_drawer.add_event(f"Opened step: {step.title}")
        self.refresh_global_state()

    def _sync_viewer_canvas_mode(self, step_id: str, previous_step: Optional[str]) -> None:
        entering_subviewer_step = step_id in self._subviewer_steps
        leaving_subviewer_step = previous_step in self._subviewer_steps if previous_step else False

        if entering_subviewer_step:
            self._enter_subviewer_canvas_mode(step_id)
        elif leaving_subviewer_step:
            self._restore_viewer_canvas_mode()

    def _capture_canvas_snapshot(self) -> None:
        if self._viewer_canvas_snapshot is not None:
            return
        dims = getattr(self.viewer, "dims", None)
        if dims is None:
            return
        try:
            self._viewer_canvas_snapshot = {
                "ndisplay": int(getattr(dims, "ndisplay", 2)),
                "order": tuple(getattr(dims, "order", ())),
            }
        except Exception:
            self._viewer_canvas_snapshot = None

    def _enter_subviewer_canvas_mode(self, step_id: str) -> None:
        self._capture_canvas_snapshot()

        if step_id == "roi_hitl_2d":
            widget = self._step_widgets.get(step_id)
            controller = getattr(widget, "controller", None)
            apply_fn = getattr(controller, "apply_active_view_to_viewer", None)
            if callable(apply_fn):
                apply_fn()
                self.history_drawer.add_event("Subviewer canvas mode aligned to ROI + HITL 2D active view")

    def _restore_viewer_canvas_mode(self) -> None:
        if self._viewer_canvas_snapshot is None:
            return
        snapshot = dict(self._viewer_canvas_snapshot)
        self._viewer_canvas_snapshot = None

        dims = getattr(self.viewer, "dims", None)
        if dims is None:
            return
        try:
            if "ndisplay" in snapshot:
                wanted = int(snapshot["ndisplay"])
                # Only ASSIGN when it actually differs.  napari's ndisplay setter
                # is not free on a same-value set: it fires the dims event chain,
                # which on 0.8 rebuilds every layer's visual (a 2D<->3D texture
                # swap on a multiscale source).  A step re-enter that returns to
                # the mode it left in was paying that for nothing.
                if int(getattr(dims, "ndisplay", -1)) != wanted:
                    self.viewer.dims.ndisplay = wanted
        except Exception:
            pass
        try:
            order = tuple(snapshot.get("order", ()))
            if order and len(order) == getattr(dims, "ndim", len(order)):
                self.viewer.dims.order = order
        except Exception:
            pass

    def _update_step_intent(self, step_id: str) -> None:
        intent = self._step_intents.get(step_id)
        if intent is None:
            self.intent_objective.setText("-")
            self.intent_outputs.setText("-")
            self.intent_success.setText("-")
            self.intent_actions.setText("-")
            return

        self.intent_objective.setText(intent.objective)
        self.intent_outputs.setText(intent.outputs)
        self.intent_success.setText(intent.success)
        self.intent_actions.setText("\n".join(f"- {entry}" for entry in intent.actions))

    def _build_step_guidance(self, step_id: str) -> str:
        """The ORDINARY advisory for a step, or "" — the banner's second strip.

        B1: the "no workspace" wording moved into the workspace strip
        (`_workspace_facts.banner_strips`), so this returns only the real
        advisories.  The round advisory now reads the workspace's TRAINED
        rounds from DISK instead of Step 2's live
        `current_round`, which is empty until Step 2 has been opened in this
        session — the reason a workspace with three trained rounds still
        showed "No completed HITL training round detected yet…".
        """
        workspace = self._read_workspace_path()
        if not workspace:
            # Nothing shared yet: no advisory, and — just as important — no
            # disk read at start-up.
            return ""
        # An advisory describes the workspace THIS STEP is on.  Reading it off
        # the shared path was wrong for a diverged step: Step 3 sitting on an
        # old-layout folder was told about the shared workspace's state file
        # instead of its own, so the "press Browse" line never appeared where
        # it was needed.  (Found by the B2 live run.)
        workspace = self._read_step_workspace_path(step_id) or workspace

        # B2: adopt deliberately shows NO modal, so the two
        # things Browse would have said out loud are said here instead.
        if step_id == "roi_hitl_2d":
            hint = _workspace_facts.step2_layout_hint(workspace)
            if hint:
                return hint
            if not _workspace_facts.step2_has_runs(workspace):
                return _workspace_facts.MSG_STEP2_NO_RUN

        if step_id == "reconstruct_curate_3d":
            if _workspace_facts.step3_pre_v3(workspace):
                # Louder than the round advisory: until the user Browses,
                # Step 3 is reading this workspace and writing nothing.
                return _workspace_facts.MSG_STEP3_PRE_V3

        if step_id in {"reconstruct_curate_3d", "finalize_model", "apply_3d",
                       "whole_tissue"}:
            rounds = _workspace_facts.trained_rounds(workspace)
            if rounds.get("lateral", 0) == 0 and rounds.get("axial", 0) == 0:
                return _workspace_facts.MSG_NO_TRAINED_MODEL

        has_model = bool(self._read_model_choice("lateral")
                         or self._read_model_choice("axial"))
        # Step 4 picks its own model on its model card;
        # the stale "Finalize Model" advisory is for Step 5 only.
        if step_id in {"whole_tissue"} and not has_model:
            return ("No final model is selected yet. "
                    "Continue to Finalize Model if needed.")

        return ""

    # ------------------------------------------------------------------
    # Shared workspace
    # ------------------------------------------------------------------

    #: Short names for the sidebar's one status line.
    _STEP_SHORT_NAMES: Dict[str, str] = {
        "project_setup": "Step 1",
        "roi_hitl_2d": "Step 2",
        "reconstruct_curate_3d": "Step 3",
        "apply_3d": "Step 4",
        "whole_tissue": "Step 5",
    }

    #: Where each step keeps its own workspace path.  This is the table the
    #: pre-B1 `_read_workspace_path` scrape encoded; it survives as the
    #: per-step lookup used by the banner and by divergence detection.
    _STEP_WORKSPACE_FIELDS: Dict[str, str] = {
        "project_setup": "output_file_path",
        "roi_hitl_2d": "set_workdir_path",
        "reconstruct_curate_3d": "set_workdir_path",
        "finalize_model": "workdir_line",
        "apply_3d": "workdir_line",
        "whole_tissue": "workdir_line",
    }

    @staticmethod
    def _same_path(a: str, b: str) -> bool:
        if not a or not b:
            return False
        try:
            return (os.path.normcase(os.path.normpath(a))
                    == os.path.normcase(os.path.normpath(b)))
        except Exception:      # noqa: BLE001
            return a == b

    def _read_step_workspace_path(self, step_id: str) -> str:
        """That step's OWN workspace path, straight from its line-edit."""
        attr = self._STEP_WORKSPACE_FIELDS.get(step_id)
        if not attr:
            return ""
        widget = self._step_widgets.get(step_id)
        text_widget = getattr(widget, attr, None)
        text_method = getattr(text_widget, "text", None)
        if not callable(text_method):
            return ""
        try:
            return text_method().strip()
        except Exception:      # noqa: BLE001
            return ""

    def set_workspace(self, path: str, source_step: str = "") -> None:
        """Publish a workspace the user chose with Browse in `source_step`.

        The symmetric rule: **the most recent Browse in ANY step sets
        the shared workspace.**  No step is privileged — users come back to
        the step they left off in and set the folder there.

        Called ONLY from the four Browse handlers — never from a
        `textChanged` chain, so an adopt or the start-up pre-fill can never
        masquerade as a user choice.

        What happens:
          * the browsing step is marked ``browsed`` (it is green by
            definition — its path IS the shared workspace);
          * an unchanged path early-returns after that marking, so browsing a
            diverged step back to the shared folder turns it green again;
          * otherwise the shared workspace moves, is remembered on disk, and
            every step whose path was merely ``adopted`` (or is empty)
            re-adopts it — immediately for the step on screen, and on entry
            for the rest;
          * a step the user had browsed SOMEWHERE ELSE keeps its own folder
            and becomes diverged (amber).  Divergence is not permanent: it
            ends as soon as the shared workspace comes back to that path, or
            the user browses that step to the shared one.
        """
        if not path:
            return
        try:
            path = os.path.abspath(path)
        except Exception:      # noqa: BLE001
            pass

        if source_step:
            self._step_path_origin[source_step] = "browsed"

        if self._same_path(path, self._workspace_path):
            self._refresh_divergence()
            self.refresh_global_state()
            self._refresh_step_guidance()
            return

        previous = self._workspace_path
        self._workspace_path = path
        self._workspace_source_step = source_step
        self._suggested_workspace = path
        _workspace_facts.clear_cache()
        _workspace_facts.save_last_workspace(path)
        print(f"[WORKSPACE] shared workspace set from {source_step or '?'}: "
              f"{path}")

        # Re-adopt the step the user is looking at, so its banner + path box
        # follow immediately; the others follow on entry (`_select_step`).
        if self._active_step_id and self._active_step_id != source_step:
            self._adopt_into_step(self._active_step_id, previous=previous)

        self._refresh_divergence()
        self.refresh_global_state()
        self._refresh_step_guidance()

    def _adopt_into_step(self, step_id: str, previous: str = "") -> bool:
        """Give `step_id` the shared workspace, if it is free to take it.

        A step takes the shared workspace when it has no path of its own, or
        when it is still holding a path it only INHERITED (`previous`, the
        workspace that was shared before).  A step the user browsed
        somewhere else is never overwritten — it stays amber
        until the user browses it again.

        Returns True when the step's controller actually adopted.
        """
        if not self._shell_ready or not self._workspace_path:
            return False
        ctrl = self._get_step_controller(step_id)
        if ctrl is None:
            return False
        adopt = getattr(ctrl, "adopt_workspace", None)
        if not callable(adopt):
            return False

        step_path = self._read_step_workspace_path(step_id)
        origin = self._step_path_origin.get(step_id, "")
        if step_path:
            if origin == "browsed":
                return False
            if self._same_path(step_path, self._workspace_path):
                return False
            # An inherited, now-stale path: hand the controller the OLD path
            # so it can tell "still holding what I adopted" from "the user
            # changed this behind my back".
            previous = previous or step_path

        try:
            try:
                adopt(self._workspace_path, previous=previous)
            except TypeError:
                # A controller on the older single-argument contract.
                adopt(self._workspace_path)
        except Exception as exc:      # noqa: BLE001
            print(f"[WORKSPACE] adopt failed for {step_id}: {exc}")
            return False

        if self._same_path(self._read_step_workspace_path(step_id),
                           self._workspace_path):
            self._step_path_origin[step_id] = "adopted"
            return True
        return False

    def _refresh_divergence(self) -> None:
        """Recompute which steps are amber, and log each change once.

        A step is diverged only while the user BROWSED it somewhere other
        than the shared workspace.  The entry is pruned the moment its path
        matches again (N6) — divergence is a state, not a life sentence.
        """
        shared = self._workspace_path
        current: Dict[str, str] = {}
        if shared:
            for step_id, origin in self._step_path_origin.items():
                if origin != "browsed":
                    continue
                step_path = self._read_step_workspace_path(step_id)
                if step_path and not self._same_path(step_path, shared):
                    current[step_id] = step_path
        for step_id, step_path in current.items():
            key = (step_id, os.path.normcase(step_path),
                   os.path.normcase(shared))
            if key in self._diverged_logged:
                continue
            self._diverged_logged.add(key)
            print(f"[WORKSPACE] {step_id} is on {step_path} — diverged from "
                  f"the shared workspace {shared}")
        self._diverged_steps = current

    def _prefill_step1_from_suggestion(self) -> None:
        """Show the remembered workspace in Step 1's box — and do nothing else.

        `blockSignals` keeps the pre-fill out of Step 1's `textChanged` chain
        (source restore, patch-count refresh, migration timer), so start-up
        performs no I/O inside the workspace.  The path
        becomes real the moment the user enters a step or presses Browse.
        """
        if not self._suggested_workspace:
            return
        widget = self._step_widgets.get("project_setup")
        line = getattr(widget, "output_file_path", None)
        if line is None or not hasattr(line, "setText"):
            return
        try:
            if line.text().strip():
                return
            line.blockSignals(True)
            try:
                line.setText(self._suggested_workspace)
            finally:
                line.blockSignals(False)
        except Exception as exc:      # noqa: BLE001
            print(f"[WORKSPACE] could not pre-fill the remembered "
                  f"workspace: {exc}")
            return
        print(f"[WORKSPACE] remembered workspace shown (nothing loaded): "
              f"{self._suggested_workspace}")
        self._refresh_step_guidance()
        self.refresh_global_state()

    def _workspace_is_missing(self, path: str) -> bool:
        """True when a local-drive workspace path no longer exists (I7/I13)."""
        if not path:
            return False
        try:
            if not _workspace_facts._is_local_drive_path(path):
                return False
            return not os.path.isdir(path)
        except Exception:      # noqa: BLE001
            return False

    def _refresh_step_guidance(self, step_id: Optional[str] = None) -> None:
        """Repaint both banner strips for the active step.

        Called after adopt + `on_step_enter` in `_select_step`, from
        `set_workspace`, and from `refresh_global_state` — so a Browse turns
        the line green immediately, without a step switch.
        """
        step_id = step_id or self._active_step_id
        if step_id is None or not hasattr(self, "guidance_label"):
            return
        shared = self._workspace_path
        step_path = self._read_step_workspace_path(step_id)
        # The remembered path pre-filled into Step 1's box is a SUGGESTION,
        # not a folder anyone has opened (F1): while nothing is shared it
        # must still read "Choose a workspace folder to begin.", or start-up
        # would claim a workspace the user never picked.
        if (not shared and step_path and self._suggested_workspace
                and self._same_path(step_path, self._suggested_workspace)):
            step_path = ""
        strips = _workspace_facts.banner_strips(
            step_id,
            shared,
            step_path,
            advisory=self._build_step_guidance(step_id),
            missing=self._workspace_is_missing(step_path or shared),
            browsed=self._step_path_origin.get(step_id) == "browsed",
        )
        labels = [self.guidance_label, self.advisory_label]
        # A diverged step's line names two paths and can be clipped in a
        # narrow panel, so its tooltip spells both of them out.
        tooltip = ""
        if shared and step_path and not self._same_path(shared, step_path):
            tooltip = f"This step: {step_path}\nShared workspace: {shared}"
        for index, (label, strip) in enumerate(zip(labels, strips)):
            label.set_strip(*strip, tooltip=(tooltip if index == 0 else ""))
        for label in labels[len(strips):]:
            label.clear_strip()

    def _read_active_step_status(self, step_id: Optional[str]) -> str:
        """One sidebar line that is true on whichever step the user is on.

        Read-only, and cheap: the Step 2 round counts and the Step 1 patch
        count come from `_workspace_facts`, which caches on an mtime, so the
        3 s poll costs a `stat`.
        """
        # Before the shell is ready nothing has been chosen and nothing may
        # be read from disk — not even a patch count (N1).
        if not getattr(self, "_shell_ready", False):
            return ""
        workspace = self._read_workspace_path()
        if not workspace:
            short = self._STEP_SHORT_NAMES.get(step_id, "Step 1")
            return f"{short}  ·  nothing loaded"

        if step_id == "project_setup":
            # `allow_network` only once a Browse has proved the path is
            # reachable — an unreachable UNC path must not stall the UI.
            n = _workspace_facts.patch_count(
                workspace, allow_network=bool(self._workspace_source_step))
            return f"Step 1  ·  {n} patch{'' if n == 1 else 'es'}"

        if step_id == "roi_hitl_2d":
            rounds = _workspace_facts.trained_rounds(workspace)
            ctrl = self._get_step_controller("roi_hitl_2d")
            run = (getattr(ctrl, "active_run_id", "") or "").strip()
            head = f"Step 2  ·  run {run}" if run else "Step 2"
            return (f"{head}  ·  lateral {rounds.get('lateral', 0)}"
                    f"  ·  axial {rounds.get('axial', 0)}")

        if step_id == "reconstruct_curate_3d":
            ctrl = self._get_step_controller("reconstruct_curate_3d")
            patch = (getattr(ctrl, "_selected_curation_patch", "")
                     or getattr(ctrl, "_selected_patch_name", "") or "")
            patch = patch.strip() if isinstance(patch, str) else ""
            if not patch:
                return "Step 3  ·  no patch selected"
            status = ""
            manager = getattr(ctrl, "state_manager", None)
            getter = getattr(manager, "get_patch_status", None)
            if callable(getter):
                try:
                    status = getter(patch) or ""
                except Exception:      # noqa: BLE001
                    status = ""
            return (f"Step 3  ·  {patch}  ·  {status}" if status
                    else f"Step 3  ·  {patch}")

        if step_id == "apply_3d":
            widget = self._step_widgets.get("apply_3d")
            listw = getattr(widget, "stacks_list", None)
            count = 0
            try:
                count = int(listw.count()) if listw is not None else 0
            except Exception:      # noqa: BLE001
                count = 0
            return f"Step 4  ·  {count} stack{'' if count == 1 else 's'}"

        title = self.step_title.text().strip() if hasattr(self, "step_title") else ""
        return title or "-"

    def _diverged_note(self) -> str:
        """The sidebar tooltip's "Browsed elsewhere:" lines, or ""."""
        if not self._diverged_steps:
            return ""
        titles = {s.step_id: s.title for s in self._step_specs}
        lines = [
            f"Browsed elsewhere: {titles.get(sid, sid)} → {path}"
            for sid, path in sorted(self._diverged_steps.items())
        ]
        return "\n".join(lines)

    def _init_state_polling(self) -> None:
        self._state_timer = QTimer(self)
        self._state_timer.setInterval(3000)
        self._state_timer.timeout.connect(self.refresh_global_state)
        self._state_timer.start()
        self.refresh_global_state()

    def _wire_state_signals(self) -> None:
        # Prefer signal-driven updates, keep polling as fallback.
        watch_fields = [
            ("project_setup", "output_file_path"),
            ("project_setup", "extracted_roi_list"),
            ("roi_hitl_2d", "set_workdir_path"),
            ("roi_hitl_2d", "data_to_add_list"),
            ("reconstruct_curate_3d", "set_workdir_path"),
            ("finalize_model", "workdir_line"),
            ("finalize_model", "lateral_model_combo"),
            ("finalize_model", "axial_model_combo"),
            ("finalize_model", "pack_list"),
            ("apply_3d", "workdir_line"),
            ("whole_tissue", "workdir_line"),
            ("whole_tissue", "jobs_list"),
        ]

        for step_id, attr in watch_fields:
            widget = self._step_widgets.get(step_id)
            if widget is None:
                continue
            control = getattr(widget, attr, None)
            if control is None:
                continue

            for signal_name in ("textChanged", "currentTextChanged", "itemSelectionChanged"):
                signal = getattr(control, signal_name, None)
                if signal is not None:
                    signal.connect(self._refresh_global_state_slot)

            model = getattr(control, "model", None)
            if callable(model):
                item_model = model()
                if item_model is not None:
                    item_model.rowsInserted.connect(self._refresh_global_state_slot)
                    item_model.rowsRemoved.connect(self._refresh_global_state_slot)
                    item_model.modelReset.connect(self._refresh_global_state_slot)

        self._wire_button_history()

    def _wire_button_history(self) -> None:
        button_events = [
            ("project_setup", "extract_roi", "ROI extraction triggered"),
            ("project_setup", "save_reviewed_roi", "ROI export triggered"),
            ("roi_hitl_2d", "update_and_train_button", "HITL training triggered"),
            # ("roi_hitl_2d", "run_cross_round_benchmark_button", ...) — removed
            # 2026-09-07: the Benchmarking panel is behind ComingSoonOverlay and
            # the button is unwired + disabled, so logging "Round benchmark
            # triggered" would have been a lie.
            ("roi_hitl_2d", "finalize_labels_button", "HITL label finalize triggered"),
            ("reconstruct_curate_3d", "segment_button", "Predict All patches triggered"),
            ("reconstruct_curate_3d", "prediction_pause_btn", "Prediction pause requested"),
            ("reconstruct_curate_3d", "run_reconstruction_btn", "Reconstruct All patches triggered"),
            ("reconstruct_curate_3d", "reconstruction_pause_btn", "Reconstruction pause requested"),
            ("reconstruct_curate_3d", "finalize_labels_button", "Reconstruction finalize triggered"),
            ("reconstruct_curate_3d", "extract_features_btn", "QC feature extraction triggered"),
            ("reconstruct_curate_3d", "extract_all_features_btn", "QC feature extraction (all patches) triggered"),
            ("reconstruct_curate_3d", "apply_threshold_btn", "QC threshold filter applied"),
            ("reconstruct_curate_3d", "save_qc_csv_btn", "QC CSV export triggered"),
            ("reconstruct_curate_3d", "extract_training_pairs_btn", "Training pair extraction triggered"),
            ("reconstruct_curate_3d", "launch_refinement_btn", "Model refinement training triggered"),
            ("finalize_model", "finalize_btn", "Final model pack generation triggered"),
            ("whole_tissue", "generate_job_btn", "Whole tissue job generation triggered"),
        ]

        for step_id, attr, message in button_events:
            widget = self._step_widgets.get(step_id)
            if widget is None:
                continue
            button = getattr(widget, attr, None)
            clicked = getattr(button, "clicked", None)
            if clicked is not None:
                clicked.connect(lambda _checked=False, msg=message: self.history_drawer.add_event(msg))
                clicked.connect(self._refresh_global_state_slot)

    def _refresh_global_state_slot(self, *_args) -> None:
        self.refresh_global_state()

    def refresh_global_state(self) -> None:
        workspace = self._read_workspace_path()
        active_step = self.step_title.text().strip()

        self.state_panel.update_values(
            {
                "workspace_path": workspace,
                "active_step": active_step,
                "status_line": self._read_active_step_status(
                    self._active_step_id),
                "diverged_note": self._diverged_note(),
            }
        )
        # The banner reads the same state; keeping it on this one refresh
        # path means a Browse repaints it without a step switch.
        self._refresh_step_guidance()

    def _read_workspace_path(self) -> str:
        """The shared workspace — a plain getter.

        The pre-B1 write-once scrape over the step line-edits is gone: it
        went stale the moment a step browsed elsewhere and it could never
        tell a shared workspace from a diverged one.  The per-step lookup it
        encoded lives on as `_read_step_workspace_path`.
        """
        return self._workspace_path

    def _read_active_roi(self) -> str:
        widget = self._step_widgets.get("project_setup")
        if widget is None:
            return ""
        roi_list = getattr(widget, "extracted_roi_list", None)
        if roi_list is None:
            return ""
        current = roi_list.currentItem()
        if current is not None:
            return current.text()
        if roi_list.count() > 0:
            return roi_list.item(0).text()
        return ""

    def _read_round_number(self) -> str:
        widget = self._step_widgets.get("roi_hitl_2d")
        if widget is None:
            return ""
        controller = getattr(widget, "controller", None)
        if controller is None:
            return ""
        round_idx = getattr(controller, "current_round", None)
        if round_idx is None:
            return ""
        return str(round_idx)

    def _read_staged_sample_count(self) -> str:
        widget = self._step_widgets.get("roi_hitl_2d")
        if widget is None:
            return ""
        staged_list = getattr(widget, "data_to_add_list", None)
        if staged_list is None:
            return ""
        return str(staged_list.count())

    def _read_model_choice(self, mode: str) -> str:
        widget = self._step_widgets.get("finalize_model")
        if widget is None:
            return ""
        combo_attr = "lateral_model_combo" if mode == "lateral" else "axial_model_combo"
        combo = getattr(widget, combo_attr, None)
        text_method = getattr(combo, "currentText", None)
        if not callable(text_method):
            return ""
        value = text_method().strip()
        if value == "<none>":
            return ""
        return value

    def _read_curation_status(self):
        for step_id in ("roi_hitl_2d", "reconstruct_curate_3d"):
            widget = self._step_widgets.get(step_id)
            if widget is None:
                continue
            controller = getattr(widget, "controller", None)
            if controller is None:
                continue
            sessions = getattr(controller, "curation_sessions", None)
            if sessions is None:
                continue
            active = getattr(sessions, "active", None)
            if active is None:
                continue

            source = getattr(active, "source_artifact", "")
            workflows = getattr(active, "workflows_used", [])
            workflow_summary = ", ".join(workflows) if workflows else "workflow"
            label = f"{source} ({workflow_summary})" if source else workflow_summary
            return label, "active"

        return "", "idle"

    def _read_latest_artifact_status(self):
        candidates = [
            ("finalize_model", "pack_list", "status_label"),
            ("whole_tissue", "jobs_list", "status_label"),
        ]

        for step_id, list_attr, status_attr in candidates:
            widget = self._step_widgets.get(step_id)
            if widget is None:
                continue
            list_widget = getattr(widget, list_attr, None)
            status_widget = getattr(widget, status_attr, None)
            if list_widget is None:
                continue
            if list_widget.count() <= 0:
                continue
            latest = list_widget.item(0).text()
            status = ""
            status_text = getattr(status_widget, "text", None)
            if callable(status_text):
                status = status_text().strip()
            return latest, status

        return "", ""

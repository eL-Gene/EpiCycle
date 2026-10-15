"""
3D Stack Segmentation — the Step 4 panel.

Four sections on the shared Accordion in Step 4's green: ① Session,
② Segment (a pinned two-column splitter under the run row), ③ Measure &
tidy (today's QC Analysis content until brick B1b) and Diagnostics.

Every sentence on this panel is a named constant of ``_step4_text`` or of
the shared pipeline block in ``ui_shell/theme.py``.  The three stack
lists stay ``QListWidget``s (D-a) painted by one ``StackRowDelegate``; the
ticks of ① and ② are mirrored here and reported as ``ticks_changed``.
"""

from qtpy.QtCore import QPoint, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QAbstractSpinBox,
)
from superqt import QCollapsible

from ..ui_shell.theme import (
    Accordion, COLLAPSIBLE_DURATION, COLLAPSIBLE_STYLE,
    DIM_LABEL_STYLE, FLOW_HINT_STYLE, FLOW_TITLE_STYLE,
    FLOW_SUBTITLE_STYLE, INPUT_STYLE, LIST_STYLE_GREEN, LOG_OUTPUT_STYLE,
    MONO_LABEL_STYLE, PRIMARY_BTN_STYLE_GREEN, PROGRESS_BAR_STYLE_GREEN,
    MODEL_NONE, RUN_ROW_DEFAULT, SCROLLBAR_STYLE, SECONDARY_BTN_STYLE,
    SECTION_HEADER_STYLE, SEPARATOR_LINE_STYLE, SPLITTER_STYLE,
    STATUS_LABEL_STYLE, STEP4_ACCENT,
    STACK_STATUS_COLORS, STACK_STATUS_DOTS, STACK_STATUS_ICONS,
    apply_button_icon, apply_inner_group_style, destructive_btn_style,
    help_button, pipe_units,
)
from . import _step4_text as T
from ._step4_rows import norm_path as _norm
from ._export_dialog import ElidedLabel as _ElidedLabel
from ._export_dialog import Step4ExportDialog
from ..ui_shell._recon_settings_widgets import (
    ENGINE_NAME_STYLE, FORM_FIELD_W, ZWSP, LockLine, TickLabel,
    adv_label_html, adv_row, collapsible, field, frm4, lock_line_style,
    sub_header,
)
from ._stack_row_delegate import (
    MODE_MEASURE, MODE_NARROW, MODE_WIDE, ROLE_NAME, ROLE_RECORD,
    ROLE_STATUS, ROLE_THUMB, StackListWidget, StackRowDelegate,
    CHECK_DISABLED_STYLE, list_cap_px,
)


# ---------------------------------------------------------------------------
# Status palette (from theme — single source of truth)
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    status: QColor(hex_val)
    for status, hex_val in STACK_STATUS_COLORS.items()
}
_STATUS_ICONS = STACK_STATUS_ICONS
_STATUS_DOTS = STACK_STATUS_DOTS

# Layout numbers
SPLIT_LEFT_W = 224
SPLIT_RIGHT_W = 330
SPLIT_LEFT_MIN = 200
SPLIT_RIGHT_MIN = 300
LEFT_INNER_MARGIN = 6          # setContentsMargins(0, 0, 6, 0)
RIGHT_INNER_MARGIN = 8         # setContentsMargins(8, 0, 0, 0)
#: ① list: four rows of the 128 px picture (A3; it was 310 px, the
#: mockup's six rows at 32 px), then it scrolls inside itself (F4)
SESSION_LIST_MAX_H = list_cap_px(MODE_WIDE)

GUIDE_SESSION = "step4_session.md"
GUIDE_SEGMENT = "step4_segment.md"
GUIDE_MEASURE = "step4_measure_tidy.md"

_AMBER = "#d4a843"
LOCK_LINE_STYLE = lock_line_style()
AMBER_NOTE_STYLE = "color: #d4a843; font-size: 11px;"
AMBER_DIM_STYLE = "color: #d4a843; font-size: 10px;"
_FOOT_BTN_PAD = "QPushButton { padding: 7px 6px; }"
# ③ (Appendix A .hchip / .scard6 / .spill)
CHIP_STYLE = "color: #999999; font-size: 10px;"
CHIP_WARN_STYLE = "color: #d4a843; font-size: 10px;"
MT_TITLE_DIM_STYLE = "font-weight: bold; font-size: 12px; color: #666666;"
STACK_CARD_STYLE = (
    "QFrame#step4StackCard { background: rgba(255,255,255,0.03); "
    "border: 1px solid rgba(255,255,255,0.08); border-radius: 3px; }")
CARD_NAME_STYLE = (
    "color: #dddddd; font-size: 11px; "
    "font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;")
CARD_DIM_NAME_STYLE = "color: #999999; font-size: 11px;"
CARD_LINE_STYLE = "color: #999999; font-size: 10px;"
PILL_AMBER_STYLE = (
    "color: #d4a843; border: 1px solid rgba(212,168,67,0.55); "
    "border-radius: 8px; padding: 0px 7px; font-size: 10px;")
PILL_GREY_STYLE = (
    "color: #aaaaaa; border: 1px solid rgba(255,255,255,0.14); "
    "border-radius: 8px; padding: 0px 7px; font-size: 10px;")
_EXPORT_BTN_PAD = "QPushButton { padding: 2px 10px; }"
#: Medium = today's σ in voxels (S4u:1176-1183); Light half, Strong double.
SIGMA_MEDIUM = (3.0, 1.5)
SMOOTH_FACTORS = (0.5, 1.0, 2.0)
SIGMA_VOX_RANGE = (0.1, 10.0)
QC_BAR_PERCENT = "%p%"          # Qt's own percentage (Smooth, meshes)
_REVIEW_COLOURS = ("#6abf69", "#e07070", "#aaaaaa")
#: ③ rows keep the record as built here; ROLE_RECORD is what is painted.
ROLE_BASE = Qt.UserRole + 5


def _guide_path(name: str) -> str:
    from napari_epicycle.ui_shell._guides import guide_path
    return str(guide_path(name))


def rich_amber(dim: str, amber: str) -> str:
    """Rich text: the dim part, then an amber span (① status, ② ticked
    line).  Both parts are constants; ``&``/``<`` are escaped."""
    import html
    out = html.escape(dim or "")
    if amber:
        out += f"<span style='color:{_AMBER}'>{html.escape(amber)}</span>"
    return out


class _TickLabel(TickLabel):
    """``ui_shell._recon_settings_widgets.TickLabel`` under Step 4's own
    class name (the lift keeps the panel's widget tree unchanged)."""


class _LockLine(LockLine):
    """``ui_shell._recon_settings_widgets.LockLine`` (object name
    ``step4LockLine``) under Step 4's own class name."""


class SegmentStack(QWidget):
    """Step 4 — 3D Stack Segmentation with QC analysis."""

    #: Names of the ticked stacks, after a tick in ① or ②.
    ticks_changed = Signal(list)
    #: A row's eye (``StackRowDelegate``): the stack's name.
    stack_eye_clicked = Signal(str)
    #: A ① row's ``⋯`` menu: the stack's name and where to open the menu.
    stack_menu_requested = Signal(str, QPoint)
    #: A ③ row was clicked: load that stack.
    mt_load_requested = Signal(str)

    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self._mt_unsaved = None
        self._build_ui()
        self._apply_theme()

        from .controller import SegmentStackController
        self.controller = SegmentStackController(widget=self, viewer=viewer)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        apply_inner_group_style(self)
        # Primary action buttons (Step 4's green)
        for attr in ("predict_btn", "reconstruct_btn",
                     "extract_features_btn", "save_tiers_btn",
                     "finalize_btn"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setStyleSheet(PRIMARY_BTN_STYLE_GREEN)
        for attr in ("predict_btn", "reconstruct_btn", "save_tiers_btn",
                     "finalize_btn"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setStyleSheet(PRIMARY_BTN_STYLE_GREEN + _FOOT_BTN_PAD)
        # Secondary buttons
        for attr in ("browse_workdir_btn", "add_folder_btn",
                     "set_spacing_btn", "new_run_btn", "manage_runs_btn",
                     "cancel_btn", "view_btn",
                     "feature_reference_btn", "open_analysis_btn",
                     "delete_bad_btn", "undo_btn", "smooth_all_btn",
                     "fix_by_hand_btn"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setStyleSheet(SECONDARY_BTN_STYLE)
        self.export_btn.setStyleSheet(SECONDARY_BTN_STYLE + _EXPORT_BTN_PAD)
        # Reset only: red (.btnx, destructive_btn_style)
        self.reset_original_btn.setStyleSheet(destructive_btn_style())
        # Button icons (today's _btn_icons names; View = eye)
        _btn_icons = {
            "predict_btn": "mdi6.brain",
            "reconstruct_btn": "mdi6.cube-outline",
            "cancel_btn": "mdi6.stop-circle-outline",
            "view_btn": "mdi6.eye-outline",
            "extract_features_btn": "mdi6.chart-scatter-plot",
            "save_tiers_btn": "mdi6.content-save-outline",
            "finalize_btn": "mdi6.check-all",
        }
        for attr, icon_name in _btn_icons.items():
            btn = getattr(self, attr, None)
            if btn is not None:
                apply_button_icon(btn, icon_name, size=14)
        for attr, icon_name in (("add_folder_btn", "mdi6.folder-plus-outline"),
                                ("set_spacing_btn", "mdi6.ruler"),
                                ("new_run_btn", "mdi6.plus-circle-outline"),
                                ("manage_runs_btn", "mdi6.cog-outline"),
                                ("open_analysis_btn", "mdi6.chart-line"),
                                ("feature_reference_btn",
                                 "mdi6.information-outline"),
                                ("delete_bad_btn", "mdi6.delete-outline"),
                                ("undo_btn", "mdi6.undo"),
                                ("smooth_all_btn", "mdi6.blur"),
                                ("fix_by_hand_btn", "mdi6.brush-outline"),
                                ("export_btn",
                                 "mdi6.database-export-outline")):
            btn = getattr(self, attr, None)
            if btn is not None:
                apply_button_icon(btn, icon_name, color=STEP4_ACCENT, size=14)
        apply_button_icon(self.reset_original_btn, "mdi6.restore",
                          color="#ff6b6b", size=14)
        self.setStyleSheet(
            self.styleSheet() + INPUT_STYLE + LIST_STYLE_GREEN
            + SCROLLBAR_STYLE + COLLAPSIBLE_STYLE + CHECK_DISABLED_STYLE
        )
        # Progress bars (green accent)
        for attr in ("run_progress", "qc_progress"):
            bar = getattr(self, attr, None)
            if bar is not None:
                bar.setStyleSheet(PROGRESS_BAR_STYLE_GREEN)

    # ------------------------------------------------------------------
    # Top-level layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        root.addWidget(self._build_flow_overview())
        self._help_buttons = []

        accordion = Accordion(accent=STEP4_ACCENT)
        accordion.addItem(self._build_session_panel(), T.SECTION_SESSION,
                          icon_name="mdi6.folder-cog-outline")
        accordion.addItem(self._build_segment_panel(), T.SECTION_SEGMENT,
                          icon_name="mdi6.play-box-outline")
        accordion.addItem(self._build_measure_panel(), T.SECTION_MEASURE,
                          icon_name="mdi6.chart-scatter-plot")
        accordion.addItem(self._build_diagnostics_panel(),
                          T.SECTION_DIAGNOSTICS,
                          icon_name="mdi6.stethoscope")
        accordion.setCurrentIndex(0)
        root.addWidget(accordion, 1)
        self.accordion = accordion
        # The Export window: built once, kept hidden; the controller
        # reads its buttons through these attributes (grill 2 U1-4).
        self.export_dialog = Step4ExportDialog(self)
        self.export_features_btn = self.export_dialog.export_features_btn
        self.export_meshes_btn = self.export_dialog.export_meshes_btn
        self.export_status_label = self.export_dialog.export_status_label
        self.export_rows_list = self.export_dialog.export_rows_list
        self._apply_help_lines()
        self._wire_lists()
        # Remember a pick on currentIndexChanged, connected HERE -- before
        # the controller exists -- so it runs before the controller's own
        # handler, which may rebuild the combos from the remembered choice
        # (the owner's "can't set the channels", 2026-09-29: Qt emits
        # currentIndexChanged before activated, and remembering only on
        # activated let every rebuild snap the pick back).  Rebuilds run
        # under blockSignals, so they never overwrite the choice.
        self.chan1_combo.currentIndexChanged.connect(self._remember_chan1)
        self.chan2_combo.currentIndexChanged.connect(self._remember_chan2)

    def _build_flow_overview(self):
        hint = QWidget()
        hint.setStyleSheet(FLOW_HINT_STYLE)
        layout = QVBoxLayout(hint)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        flow = QLabel(T.FLOW_TITLE)
        flow.setWordWrap(True)
        flow.setStyleSheet(FLOW_TITLE_STYLE)

        sub = QLabel(T.FLOW_SUBTITLE)
        sub.setWordWrap(True)
        sub.setStyleSheet(FLOW_SUBTITLE_STYLE)

        layout.addWidget(flow)
        layout.addWidget(sub)
        return hint

    # ------------------------------------------------------------------
    # Small builders
    # ------------------------------------------------------------------

    def _sub_header(self, title: str, guide: str = "") -> QHBoxLayout:
        return sub_header(self, title, _guide_path(guide) if guide else "",
                          accent=STEP4_ACCENT,
                          help_buttons=self._help_buttons, guide=guide)

    @staticmethod
    def _desc(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(FLOW_SUBTITLE_STYLE)
        return lbl

    @staticmethod
    def _sep() -> QLabel:
        sep = QLabel("")
        sep.setFixedHeight(1)
        sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        return sep

    @staticmethod
    def _dim(text: str = "", *, wrap: bool = True) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(wrap)
        lbl.setStyleSheet(DIM_LABEL_STYLE)
        return lbl

    # ------------------------------------------------------------------
    # ① Session
    # ------------------------------------------------------------------

    def _build_session_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 8, 6, 12)
        layout.setSpacing(5)

        self.session_lock_line = _LockLine(panel)
        layout.addWidget(self.session_lock_line)

        # Workspace
        layout.addLayout(self._sub_header(T.SESSION_WORKSPACE_TITLE,
                                          GUIDE_SESSION))
        layout.addWidget(self._desc(T.SESSION_WORKSPACE_DESC))
        self.workdir_line = QLineEdit()
        self.workdir_line.setPlaceholderText(T.SESSION_WORKSPACE_PLACEHOLDER)
        self.browse_workdir_btn = QPushButton(T.BROWSE_BTN)
        row = QHBoxLayout()
        row.addWidget(self.workdir_line, 1)
        row.addWidget(self.browse_workdir_btn)
        layout.addLayout(row)

        layout.addWidget(self._sep())

        # Stacks
        layout.addLayout(self._sub_header(T.SESSION_STACKS_TITLE,
                                          GUIDE_SESSION))
        layout.addWidget(self._desc(T.SESSION_STACKS_DESC))
        self.session_stacks_status = QLabel(T.STACKS_NONE)
        self.session_stacks_status.setStyleSheet(STATUS_LABEL_STYLE)
        self.session_stacks_status.setWordWrap(True)
        self.session_stacks_status.setTextFormat(Qt.RichText)
        layout.addWidget(self.session_stacks_status)

        self.session_stacks_list = StackListWidget()
        self.session_stacks_list.setSelectionMode(QListWidget.SingleSelection)
        self._session_delegate = StackRowDelegate(
            MODE_WIDE, self.session_stacks_list, keep_aspect=True,
            help_lines=T.STEP4_HELP_LINES, wide_line2=T.ROW_WIDE_LINE2)
        self.session_stacks_list.setItemDelegate(self._session_delegate)
        self.session_stacks_list.set_empty_lines(
            (T.STACKS_EMPTY_1, T.STACKS_EMPTY_2))
        self.session_stacks_list.setMinimumHeight(72)
        self.session_stacks_list.setFixedHeight(72)
        layout.addWidget(self.session_stacks_list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.add_folder_btn = QPushButton(T.ADD_FOLDER_BTN)
        self.set_spacing_btn = QPushButton(T.SET_SPACING_BTN)
        self.session_ticked_label = self._dim(
            T.TICKED_COUNT.format(n=0), wrap=False)
        self.session_ticked_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        btn_row.addWidget(self.add_folder_btn)
        btn_row.addWidget(self.set_spacing_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.session_ticked_label)
        layout.addLayout(btn_row)
        self.set_spacing_note = self._dim(T.SET_SPACING_NEEDS_STACK)
        layout.addWidget(self.set_spacing_note)

        layout.addWidget(self._sep())

        # Pipeline status
        layout.addLayout(self._sub_header(T.PIPELINE_TITLE))
        layout.addWidget(self._desc(T.PIPELINE_DESC))
        counts = QGridLayout()
        counts.setHorizontalSpacing(8)
        counts.setVerticalSpacing(4)
        self.pipeline_count_labels = {}
        pairs = list(T.PIPELINE_COUNT_LABELS) + [
            ("failed", T.PIPELINE_FAILED_LABEL)]
        for i, (status, word) in enumerate(pairs):
            name = QLabel(word)
            num = QLabel(T.fmt_count(0))
            num.setStyleSheet(
                f"color: {STACK_STATUS_COLORS.get(status, '#777777')}; "
                "font-weight: 600;")
            r, c = divmod(i, 3)
            counts.addWidget(name, r, 2 * c)
            counts.addWidget(num, r, 2 * c + 1)
            self.pipeline_count_labels[status] = (name, num)
        counts.setColumnStretch(6, 1)
        layout.addLayout(counts)
        self.pipeline_count_labels["failed"][0].setVisible(False)
        self.pipeline_count_labels["failed"][1].setVisible(False)
        self.session_model_pointer = self._dim(T.SESSION_MODEL_POINTER)
        layout.addWidget(self.session_model_pointer)

        layout.addStretch(1)
        return panel

    # ------------------------------------------------------------------
    # ② Segment
    # ------------------------------------------------------------------

    def _build_segment_panel(self):
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(6, 8, 6, 6)
        outer.setSpacing(6)

        self.seg_lock_line = _LockLine(panel)
        outer.addWidget(self.seg_lock_line)

        # Run row (Step 3's run_row, Step 4's accent)
        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        _run_lbl = QLabel(T.RUN_ROW_LABEL)
        _run_lbl.setStyleSheet(SECTION_HEADER_STYLE)
        run_row.addWidget(_run_lbl)
        self.run_picker_combo = QComboBox()
        self.run_picker_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.run_picker_combo.setMinimumContentsLength(10)
        self.run_picker_combo.addItem(
            RUN_ROW_DEFAULT.format(n=0, units=pipe_units("stack", 0)))
        run_row.addWidget(self.run_picker_combo, 1)
        self.new_run_btn = QPushButton(T.NEW_RUN_BTN)
        self.new_run_btn.setEnabled(False)
        run_row.addWidget(self.new_run_btn)
        self.manage_runs_btn = QPushButton(T.MANAGE_RUNS_BTN)
        self.manage_runs_btn.setEnabled(False)
        run_row.addWidget(self.manage_runs_btn)
        outer.addLayout(run_row)
        self.run_row_note = self._dim(T.RUN_ROW_AFTER_RELEASE)
        self.run_row_note.setAlignment(Qt.AlignRight)
        outer.addWidget(self.run_row_note)
        outer.addWidget(self._sep())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(SPLITTER_STYLE)
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)
        self.seg_splitter = splitter

        # ============================================================
        # LEFT COLUMN — model card, then the stacks
        # ============================================================
        left = QWidget()
        left.setMinimumWidth(SPLIT_LEFT_MIN)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, LEFT_INNER_MARGIN, 0)
        left_lay.setSpacing(4)
        self.seg_left = left

        left_lay.addLayout(self._sub_header(T.SEG_MODEL_TITLE, GUIDE_SEGMENT))
        left_lay.addWidget(self._desc(T.SEG_MODEL_DESC))
        act = QHBoxLayout()
        act.setSpacing(8)
        act.addWidget(QLabel(T.SEG_ACTIVE_LABEL))
        act.addStretch(1)
        self.model_change_btn = QPushButton(T.MODEL_CHANGE_BTN)
        act.addWidget(self.model_change_btn)
        left_lay.addLayout(act)
        self.model_active_indicator_label = QLabel(MODEL_NONE)
        self.model_active_indicator_label.setWordWrap(True)
        self.model_active_indicator_label.setStyleSheet(
            "color: #888; font-size: 11px; padding: 2px 0px;")
        left_lay.addWidget(self.model_active_indicator_label)
        self.model_trained_label = self._dim(T.SEG_MODEL_TRAINED_UNKNOWN)
        left_lay.addWidget(self.model_trained_label)

        # Hidden model-state widgets (Step 3's _hidden_model_state pattern,
        # session restore and _get_model_paths keep their callers.
        self._hidden_model_state = QWidget(left)
        self._hidden_model_state.setVisible(False)
        hidden_lay = QVBoxLayout(self._hidden_model_state)
        hidden_lay.setContentsMargins(0, 0, 0, 0)
        self.model_source_combo = QComboBox()
        self.model_source_combo.addItems(["Import", "Import", "Pretrained"])
        self.model_source_combo.setCurrentIndex(1)
        hidden_lay.addWidget(self.model_source_combo)
        self.lateral_model_path = QLineEdit()
        hidden_lay.addWidget(self.lateral_model_path)
        self.axial_model_path = QLineEdit()
        hidden_lay.addWidget(self.axial_model_path)
        self.pretrained_combo = QComboBox()
        try:
            from ..fusion_reconstruction.refinement_registry_dialog import (
                PRETRAINED_MODELS,
            )
            self.pretrained_combo.addItems([n for n, _d in PRETRAINED_MODELS])
        except Exception:      # noqa: BLE001
            self.pretrained_combo.addItems(["cyto3", "nuclei"])
        hidden_lay.addWidget(self.pretrained_combo)
        left_lay.addWidget(self._hidden_model_state)

        left_lay.addWidget(self._sep())
        left_lay.addLayout(self._sub_header(T.SEG_STACKS_TITLE, GUIDE_SEGMENT))
        left_lay.addWidget(self._desc(T.SEG_STACKS_DESC))

        self.stacks_list = StackListWidget()
        self.stacks_list.setSelectionMode(QListWidget.SingleSelection)
        self._seg_delegate = StackRowDelegate(
            MODE_NARROW, self.stacks_list, show_thumb=True, keep_aspect=True,
            help_lines=T.STEP4_HELP_LINES, wide_line2=T.ROW_WIDE_LINE2)
        self.stacks_list.setItemDelegate(self._seg_delegate)
        self.stacks_list.setMinimumHeight(60)
        left_lay.addWidget(self.stacks_list, 1)

        self.seg_ticked_label = QLabel(T.SEG_TICKED_LINE.format(n=0))
        self.seg_ticked_label.setTextFormat(Qt.RichText)
        self.seg_ticked_label.setWordWrap(True)
        self.seg_ticked_label.setStyleSheet(DIM_LABEL_STYLE)
        left_lay.addWidget(self.seg_ticked_label)

        # ============================================================
        # RIGHT COLUMN — settings (scrolls) over the pinned foot
        # ============================================================
        right = QWidget()
        right.setMinimumWidth(SPLIT_RIGHT_MIN)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(RIGHT_INNER_MARGIN, 0, 0, 0)
        right_lay.setSpacing(0)
        self.seg_right = right

        self.seg_right_scroll = QScrollArea()
        self.seg_right_scroll.setWidgetResizable(True)
        self.seg_right_scroll.setFrameShape(QFrame.NoFrame)
        self.seg_right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        content = QWidget()
        self.seg_right_content = content
        c_lay = QVBoxLayout(content)
        c_lay.setContentsMargins(0, 0, 0, 6)
        c_lay.setSpacing(6)
        self._build_segment_settings(c_lay)
        c_lay.addStretch(1)
        self.seg_right_scroll.setWidget(content)
        right_lay.addWidget(self.seg_right_scroll, 1)

        foot = QWidget()
        foot.setObjectName("step4SegFoot")
        foot.setStyleSheet(
            "QWidget#step4SegFoot { border-top: 1px solid #555555; }")
        self.seg_foot = foot
        f_lay = QVBoxLayout(foot)
        f_lay.setContentsMargins(0, 6, 0, 0)
        f_lay.setSpacing(6)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        self.predict_btn = QPushButton(T.PREDICT_BTN)
        self.reconstruct_btn = QPushButton(T.RECONSTRUCT_BTN)
        self.cancel_btn = QPushButton(T.CANCEL_RUN_BTN)
        self.cancel_btn.setEnabled(False)
        self.view_btn = QPushButton(T.VIEW_BTN)
        for b in (self.predict_btn, self.reconstruct_btn, self.cancel_btn,
                  self.view_btn):
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            b.setMinimumWidth(60)
        grid.addWidget(self.predict_btn, 0, 0)
        grid.addWidget(self.reconstruct_btn, 0, 1)
        grid.addWidget(self.cancel_btn, 1, 0)
        grid.addWidget(self.view_btn, 1, 1)
        f_lay.addLayout(grid)
        self.run_progress = QProgressBar()
        self.run_progress.setValue(0)
        self.run_progress.setTextVisible(True)
        self.run_progress.setFormat("")        # F10: no text while idle
        self.run_progress.setFixedHeight(14)
        f_lay.addWidget(self.run_progress)
        self.run_status_label = QLabel(T.STEP4_SEG_IDLE.format(n=0))
        self.run_status_label.setStyleSheet(MONO_LABEL_STYLE)
        self.run_status_label.setSizePolicy(QSizePolicy.Ignored,
                                            QSizePolicy.Preferred)
        # F3: the running sentence wraps to at most two lines
        self.run_status_label.setWordWrap(True)
        f_lay.addWidget(self.run_status_label)
        right_lay.addWidget(foot)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SPLIT_LEFT_W, SPLIT_RIGHT_W])
        outer.addWidget(splitter, 1)
        return panel

    def _frm4(self) -> QGridLayout:
        return frm4()

    @staticmethod
    def _field(w):
        return field(w)

    def _form_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        return lbl

    def _adv_row(self, grid: QGridLayout, widget_name: str, widget):
        """One .frm4 row (``ui_shell._recon_settings_widgets.adv_row``);
        its label is kept in ``_adv_engine_labels``."""
        adv_row(grid, widget_name, widget, self._adv_engine_labels,
                tick_label=_TickLabel)

    def _collapsible(self, title: str, engine: str = "") -> QCollapsible:
        """``ui_shell._recon_settings_widgets.collapsible``."""
        return collapsible(title, engine)

    def _build_segment_settings(self, lay: QVBoxLayout) -> None:
        self._adv_engine_labels = {}
        lay.addLayout(self._sub_header(T.SEG_SETTINGS_TITLE, GUIDE_SEGMENT))
        lay.addWidget(self._desc(T.SEG_SETTINGS_DESC))
        sf = QHBoxLayout()
        sf.setSpacing(8)
        sf.addWidget(QLabel(T.SEG_START_FROM_LABEL))
        self.start_from_combo = QComboBox()
        self.start_from_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.start_from_combo.setMinimumContentsLength(10)
        self.start_from_combo.addItem(T.SEG_PRESET_EPICYCLE, "epicycle")
        self.start_from_combo.addItem(T.SEG_PRESET_USEG3D, "useg3d")
        sf.addWidget(self.start_from_combo, 1)
        lay.addLayout(sf)

        # --- Inference settings (open) ---
        self._seg_collapsible = self._collapsible(T.SEG_INFERENCE_TITLE)
        inf = QWidget()
        inf_lay = QVBoxLayout(inf)
        inf_lay.setContentsMargins(4, 4, 0, 4)
        inf_lay.setSpacing(6)
        g = self._frm4()
        self.chan1_combo = QComboBox()
        self.chan2_combo = QComboBox()
        self.chan2_combo.addItem(T.SEG_CHANNEL_NONE)
        self.cell_width_um_sb = QDoubleSpinBox()
        self.cell_width_um_sb.setRange(0.0, 500.0)
        self.cell_width_um_sb.setDecimals(1)
        self.cell_width_um_sb.setSingleStep(0.5)
        self.cell_width_um_sb.setValue(0.0)
        self.cell_width_um_sb.setSpecialValueText(T.SEG_CELL_WIDTH_AUTO)
        self.cell_width_um_sb.setSuffix(T.SEG_CELL_WIDTH_SUFFIX)
        for r, (text, w) in enumerate(((T.SEG_MEMBRANE_LABEL, self.chan1_combo),
                                       (T.SEG_NUCLEI_LABEL, self.chan2_combo),
                                       (T.SEG_CELL_WIDTH_LABEL,
                                        self.cell_width_um_sb))):
            g.addWidget(self._form_label(text), r, 0)
            g.addWidget(self._field(w), r, 1)
        self.cell_width_px_label = self._dim("")
        self.cell_width_px_label.setAlignment(Qt.AlignRight)
        self.cell_width_px_label.setVisible(False)
        g.addWidget(self.cell_width_px_label, 3, 0, 1, 2)
        inf_lay.addLayout(g)
        ticks = QHBoxLayout()
        self.normalize_cb = QCheckBox(T.SEG_NORMALIZE_LABEL)
        self.normalize_cb.setChecked(True)
        self.rescale_cb = QCheckBox(T.SEG_RESCALE_LABEL)
        self.rescale_cb.setChecked(True)
        ticks.addWidget(self.normalize_cb, 1)
        ticks.addWidget(self.rescale_cb, 1)
        inf_lay.addLayout(ticks)
        self._seg_collapsible.addWidget(inf)
        self._seg_collapsible.expand(animate=False)
        lay.addWidget(self._seg_collapsible)

        # --- Advanced (closed) ---
        self._adv_collapsible = self._collapsible(T.SEG_ADVANCED_TITLE)
        adv = QWidget()
        adv_lay = QVBoxLayout(adv)
        adv_lay.setContentsMargins(4, 4, 0, 4)
        adv_lay.setSpacing(6)
        self._build_advanced(adv_lay)
        self._adv_collapsible.addWidget(adv)
        lay.addWidget(self._adv_collapsible)

    def _build_advanced(self, adv_lay: QVBoxLayout) -> None:
        # The two hidden knobs: built, never shown, so
        # the config keeps reading their defaults.
        self._hidden_knobs = QWidget()
        self._hidden_knobs.setVisible(False)
        hk = QVBoxLayout(self._hidden_knobs)
        hk.setContentsMargins(0, 0, 0, 0)
        self.flow_thresh_sb = QDoubleSpinBox()
        self.flow_thresh_sb.setRange(0, 5.0)
        self.flow_thresh_sb.setSingleStep(0.05)
        self.flow_thresh_sb.setDecimals(2)
        self.flow_thresh_sb.setValue(0.40)
        self.cellprob_sb = QDoubleSpinBox()
        self.cellprob_sb.setRange(-6.0, 6.0)
        self.cellprob_sb.setSingleStep(0.5)
        self.cellprob_sb.setDecimals(2)
        self.cellprob_sb.setValue(0.00)
        hk.addWidget(self.flow_thresh_sb)
        hk.addWidget(self.cellprob_sb)
        adv_lay.addWidget(self._hidden_knobs)

        g = self._frm4()
        self.batch_size_sb = QSpinBox()
        self.batch_size_sb.setRange(1, 1024)
        self.batch_size_sb.setValue(128)
        self.method_combo = QComboBox()
        self.method_combo.addItem(T.SEG_METHOD_DIRECT, "direct")
        self.n_cpu_sb = QSpinBox()
        self.n_cpu_sb.setRange(1, 64)
        self.n_cpu_sb.setValue(6)
        self._adv_row(g, "batch_size_sb", self.batch_size_sb)
        self._adv_row(g, "method_combo", self.method_combo)
        self._adv_row(g, "n_cpu_sb", self.n_cpu_sb)
        adv_lay.addLayout(g)

        adv_lay.addLayout(self._sub_header(T.SEG_JOIN_TITLE))
        adv_lay.addWidget(self._desc(T.SEG_JOIN_DESC))

        self._make_group_widgets()
        self._adv_groups = {}
        for plain, engine, names in T.STEP4_ADVANCED_GROUPS:
            coll = self._collapsible(plain, engine)
            body = QWidget()
            b_lay = QVBoxLayout(body)
            b_lay.setContentsMargins(4, 2, 0, 4)
            b_lay.setSpacing(5)
            gg = self._frm4()
            if engine == "size_filters":
                self._adv_row(gg, "skip_postprocess_cb",
                              self.skip_postprocess_cb)
                self.seg_size_off_note = QLabel(T.SEG_SIZE_OFF_NOTE)
                self.seg_size_off_note.setWordWrap(True)
                self.seg_size_off_note.setStyleSheet(AMBER_NOTE_STYLE)
                gg.addWidget(self.seg_size_off_note, gg.rowCount(), 0, 1, 2)
                self.seg_size_desc = self._dim(T.SEG_SIZE_DESC)
                gg.addWidget(self.seg_size_desc, gg.rowCount(), 0, 1, 2)
                names = [n for n in names if n != "skip_postprocess_cb"]
            for n in names:
                self._adv_row(gg, n, getattr(self, n))
            if "ps_mask_thresh_sb" in names:
                # F7: why the two fields are grey, on screen (Qt shows no
                # tooltip on a disabled widget)
                self.ps_flows_off_note = self._dim(T.PS_FLOWS_OFF_NOTE)
                gg.addWidget(self.ps_flows_off_note, gg.rowCount(), 0, 1, 2)
            b_lay.addLayout(gg)
            coll.addWidget(body)
            adv_lay.addWidget(coll)
            self._adv_groups[engine] = coll

        # Behaviour of the controls themselves (today's)
        self.ps_smooth_flows_cb.toggled.connect(self._update_flow_controls)
        self._update_flow_controls()
        self.skip_postprocess_cb.toggled.connect(self._update_size_off_note)
        self._update_size_off_note()

    def _update_flow_controls(self, *_):
        enabled = self.ps_smooth_flows_cb.isChecked()
        self.ps_mask_thresh_sb.setEnabled(enabled)
        self.ps_soft_weight_cb.setEnabled(enabled)
        note = getattr(self, "ps_flows_off_note", None)
        if note is not None:
            note.setVisible(not enabled)

    def _update_size_off_note(self, *_):
        self.seg_size_off_note.setVisible(self.skip_postprocess_cb.isChecked())

    def _make_group_widgets(self) -> None:
        """The 38 group controls + ``do_mp_cb`` + ``skip_postprocess_cb``,
        with today's names, ranges, steps and defaults (Step 3's, pinned by
        test_step4_useg3d_defaults_parity.py)."""
        def dspin(lo, hi, step, val, dec=None):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            if dec is not None:
                sb.setDecimals(dec)
            sb.setValue(val)
            return sb

        def ispin(lo, hi, val):
            sb = QSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(val)
            return sb

        def check(val):
            cb = QCheckBox()
            cb.setChecked(val)
            return cb

        # Smooth before joining
        self.ps_smooth_probs_cb = check(False)
        self.ps_smooth_flows_cb = check(False)
        self.ps_sigma_perp_sb = dspin(0, 20.0, 0.5, 3.0)
        self.ps_sigma_inplane_sb = dspin(0, 20.0, 0.5, 1.0)
        self.ps_mask_thresh_sb = dspin(0, 1.0, 0.05, 0.2, 2)
        self.ps_soft_weight_cb = check(True)
        # Follow the flows
        self.gd_decay_sb = dspin(0, 1, 0.001, 0.001, 3)
        self.gd_iters_sb = ispin(1, 1000, 25)
        self.gd_momenta_sb = dspin(0, 1, 0.01, 0.98)
        self.gd_renorm_cb = check(True)
        self.do_mp_cb = check(True)
        self.gd_tile_z_sb = ispin(16, 512, 128)
        self.gd_tile_y_sb = ispin(16, 1024, 256)
        self.gd_tile_x_sb = ispin(16, 1024, 256)
        self.gd_tile_overlap_sb = dspin(0, 0.5, 0.05, 0.25)
        # Combine the probabilities
        self.ccp_ksize_sb = ispin(1, 50, 25)
        self.ccp_alpha_sb = dspin(0, 10, 0.1, 0.5)
        self.ccp_smooth_sigma_sb = dspin(0, 20, 0.1, 0.8)
        self.ccp_threshold_level_sb = ispin(0, 10, 1)
        self.ccp_threshold_n_levels_sb = ispin(2, 20, 3)
        self.ccp_prob_thresh_sb = dspin(-1.0, 1.0, 0.05, -1.0, 2)
        self.ccp_prob_thresh_sb.setSpecialValueText(T.SEG_PROB_THRESH_AUTO)
        self.ccp_min_prob_thresh_sb = dspin(0.0, 1.0, 0.05, 0.0, 2)
        self.ccp_apply_one_d_p_cb = check(True)
        # Clean the mask
        self.post_small_obj_sb = ispin(0, 100000, 10000)
        self.post_fill_cb = check(True)
        self.post_closing_sb = ispin(0, 50, 1)
        self.post_dilation_sb = ispin(0, 50, 1)
        self.post_extra_erode_sb = ispin(0, 50, 0)
        # Combine the flows
        self.ccg_ksize_sb = ispin(1, 50, 25)
        self.ccg_alpha_sb = dspin(0, 10, 0.1, 0.5)
        self.ccg_smooth_sigma_sb = dspin(0, 20, 0.5, 1.5)
        self.ccg_post_sigma_sb = dspin(0, 20, 0.1, 0.5)
        # Group into cells
        self.cc_min_area_sb = ispin(0, 10000, 5)
        self.cc_thresh_factor_sb = dspin(-5.0, 5.0, 0.1, 0.0)
        self.cc_smooth_sigma_sb = dspin(0, 20, 0.5, 1.0)
        # Size limits
        self.skip_postprocess_cb = check(True)
        self.sf_min_size_sb = ispin(0, 100000, 200)
        self.sf_max_size_sb = ispin(-1, 10000000, -1)
        self.sf_max_size_sb.setSpecialValueText(T.SEG_MAX_SIZE_NONE)
        self.sf_max_size_factor_sb = dspin(0, 100, 1, 10)
        self.sf_do_stats_cb = check(True)

    def _apply_help_lines(self) -> None:
        """Every visible ① / ② control gets its help line as tooltip."""
        for name, line in T.STEP4_HELP_LINES.items():
            w = getattr(self, name, None)
            if w is not None and hasattr(w, "setToolTip"):
                w.setToolTip(line)

    # ------------------------------------------------------------------
    # ③ Measure & tidy: two pinned columns, like ②
    # ------------------------------------------------------------------

    def _mt_header(self, title: str, chip) -> QHBoxLayout:
        """A block's header row: the title, its ``?``, a right-aligned
        status chip (``.sub.s6``)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        lbl = QLabel(title)
        lbl.setStyleSheet(SECTION_HEADER_STYLE)
        row.addWidget(lbl)
        self._mt_titles.append(lbl)
        btn = help_button(self, _guide_path(GUIDE_MEASURE), title,
                          accent=STEP4_ACCENT)
        row.addWidget(btn)
        self._help_buttons.append((btn, GUIDE_MEASURE))
        if chip is not None:
            row.addWidget(chip, 1)
        else:
            row.addStretch(1)
        return row

    @staticmethod
    def _chip() -> "_ElidedLabel":
        chip = _ElidedLabel(align=Qt.AlignRight | Qt.AlignVCenter)
        chip.setStyleSheet(CHIP_STYLE)
        return chip

    @staticmethod
    def _grow(*buttons) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for b in buttons:
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            b.setMinimumWidth(60)
            row.addWidget(b, 1)
        return row

    def _build_measure_panel(self):
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(6, 8, 6, 6)
        outer.setSpacing(6)

        self.mt_lock_line = _LockLine(panel)
        outer.addWidget(self.mt_lock_line)
        self._mt_titles = []

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(SPLITTER_STYLE)
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)
        self.mt_splitter = splitter

        # ============================================================
        # LEFT COLUMN — the reconstructed stacks, the stack card
        # ============================================================
        left = QWidget()
        left.setMinimumWidth(SPLIT_LEFT_MIN)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, LEFT_INNER_MARGIN, 0)
        left_lay.setSpacing(6)
        self.mt_left = left

        left_lay.addLayout(self._sub_header(T.MT_STACKS_TITLE, GUIDE_MEASURE))
        left_lay.addWidget(self._desc(T.MT_STACKS_DESC))

        self.qc_stacks_list = StackListWidget()
        self.qc_stacks_list.setSelectionMode(QListWidget.SingleSelection)
        self.qc_stacks_list.setMinimumHeight(60)
        self._qc_delegate = StackRowDelegate(
            MODE_MEASURE, self.qc_stacks_list, show_thumb=True,
            keep_aspect=True,
            help_lines=T.STEP4_HELP_LINES, wide_line2=T.ROW_WIDE_LINE2)
        self.qc_stacks_list.setItemDelegate(self._qc_delegate)
        left_lay.addWidget(self.qc_stacks_list, 1)

        card = QFrame()
        card.setObjectName("step4StackCard")
        card.setStyleSheet(STACK_CARD_STYLE)
        c_lay = QVBoxLayout(card)
        c_lay.setContentsMargins(8, 6, 8, 6)
        c_lay.setSpacing(3)
        self.stack_card_name = _ElidedLabel(elide=Qt.ElideMiddle)
        self.stack_card_name.setStyleSheet(CARD_NAME_STYLE)
        c_lay.addWidget(self.stack_card_name)
        pill_row = QHBoxLayout()
        pill_row.setContentsMargins(0, 0, 0, 0)
        self.stack_card_pill = QLabel()
        self.stack_card_pill.setStyleSheet(PILL_AMBER_STYLE)
        self.stack_card_pill.setVisible(False)
        pill_row.addWidget(self.stack_card_pill)
        pill_row.addStretch(1)
        c_lay.addLayout(pill_row)
        self.stack_card_line = QLabel()
        self.stack_card_line.setTextFormat(Qt.RichText)
        self.stack_card_line.setWordWrap(True)
        self.stack_card_line.setStyleSheet(CARD_LINE_STYLE)
        c_lay.addWidget(self.stack_card_line)
        self.stack_card_note = QLabel()
        self.stack_card_note.setWordWrap(True)
        self.stack_card_note.setStyleSheet(AMBER_DIM_STYLE)
        self.stack_card_note.setVisible(False)
        c_lay.addWidget(self.stack_card_note)
        self.stack_card = card
        left_lay.addWidget(card)

        self.mt_not_recon_label = self._dim()
        self.mt_not_recon_label.setVisible(False)
        left_lay.addWidget(self.mt_not_recon_label)

        # ============================================================
        # RIGHT COLUMN — four blocks (scrolls) over the pinned foot
        # ============================================================
        right = QWidget()
        right.setMinimumWidth(SPLIT_RIGHT_MIN)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(RIGHT_INNER_MARGIN, 0, 0, 0)
        right_lay.setSpacing(0)
        self.mt_right = right

        self.mt_right_scroll = QScrollArea()
        self.mt_right_scroll.setWidgetResizable(True)
        self.mt_right_scroll.setFrameShape(QFrame.NoFrame)
        self.mt_right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        content = QWidget()
        self.mt_right_content = content
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(6)

        # 1. Measure
        self.mt_chip_measure = self._chip()
        lay.addLayout(self._mt_header(T.MT_MEASURE_TITLE,
                                      self.mt_chip_measure))
        lay.addWidget(self._desc(T.MEASURE_DESC))
        self.extract_features_btn = QPushButton(T.MEASURE_BTN_FIRST)
        self.extract_features_btn.setSizePolicy(QSizePolicy.Ignored,
                                                QSizePolicy.Fixed)
        lay.addWidget(self.extract_features_btn)
        self.open_analysis_btn = QPushButton(T.OPEN_WORKBENCH_BTN)
        self.feature_reference_btn = QPushButton(T.WHAT_MEASURED_BTN)
        lay.addLayout(self._grow(self.open_analysis_btn,
                                 self.feature_reference_btn))
        lay.addWidget(self._sep())

        # 2. Review
        self.mt_chip_review = QLabel()
        self.mt_chip_review.setTextFormat(Qt.RichText)
        self.mt_chip_review.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.mt_chip_review.setSizePolicy(QSizePolicy.Ignored,
                                          QSizePolicy.Preferred)
        self.mt_chip_review.setStyleSheet(CHIP_STYLE)
        lay.addLayout(self._mt_header(T.MT_REVIEW_TITLE, self.mt_chip_review))
        lay.addWidget(self._desc(T.REVIEW_DESC))
        self.delete_bad_btn = QPushButton(T.REMOVE_BAD_BTN)
        self.undo_btn = QPushButton(T.UNDO_BTN)
        lay.addLayout(self._grow(self.delete_bad_btn, self.undo_btn))
        lay.addWidget(self._sep())

        # 3. Tidy
        self.mt_chip_tidy = self._chip()
        lay.addLayout(self._mt_header(T.MT_TIDY_TITLE, self.mt_chip_tidy))
        lay.addWidget(self._desc(T.TIDY_DESC))
        self.smooth_strength_combo = QComboBox()
        self.smooth_strength_combo.addItems(list(T.SMOOTH_STRENGTHS))
        self.smooth_strength_combo.setCurrentIndex(1)       # Medium
        self.smooth_all_btn = QPushButton(T.SMOOTH_OUTLINES_BTN)
        lay.addLayout(self._grow(self.smooth_strength_combo,
                                 self.smooth_all_btn))
        lay.addWidget(self._sep())

        # 4. Fix by hand (after the release: always grey, no handler)
        self.mt_chip_fix = self._chip()
        self.mt_chip_fix.setText(T.MT_CHIP_FIX_LATER)
        lay.addLayout(self._mt_header(T.MT_FIX_TITLE, self.mt_chip_fix))
        lay.addWidget(self._desc(T.FIX_DESC))
        self.fix_by_hand_btn = QPushButton(T.FIX_BY_HAND_BTN)
        self.fix_by_hand_btn.setEnabled(False)
        lay.addLayout(self._grow(self.fix_by_hand_btn))

        # 5. Advanced (closed): smoothing in µm, channels, Reset
        self._mt_adv_collapsible = self._collapsible(T.MT_ADV_TITLE,
                                                     engine=T.MT_ADV_HINT)
        adv = QWidget()
        adv_lay = QVBoxLayout(adv)
        adv_lay.setContentsMargins(4, 4, 4, 4)
        adv_lay.setSpacing(6)
        grid = self._frm4()
        self.smooth_sigma_z_sb = QDoubleSpinBox()
        self.smooth_sigma_xy_sb = QDoubleSpinBox()
        for r, (label, box) in enumerate(
                ((T.TIDY_SIGMA_Z_LABEL, self.smooth_sigma_z_sb),
                 (T.TIDY_SIGMA_XY_LABEL, self.smooth_sigma_xy_sb))):
            box.setDecimals(2)
            lbl = QLabel(label)
            lbl.setWordWrap(True)
            grid.addWidget(lbl, r, 0)
            grid.addWidget(self._field(box), r, 1)
        self.smooth_preserve_topology_cb = QCheckBox()
        self.smooth_preserve_topology_cb.setChecked(True)
        topo = _TickLabel(T.TIDY_TOPOLOGY_LABEL,
                          self.smooth_preserve_topology_cb)
        topo.setWordWrap(True)
        grid.addWidget(topo, 2, 0)
        grid.addWidget(self.smooth_preserve_topology_cb, 2, 1,
                       Qt.AlignLeft | Qt.AlignVCenter)
        adv_lay.addLayout(grid)

        _ch = QLabel(T.MT_CHANNELS_TITLE)
        _ch.setStyleSheet(SECTION_HEADER_STYLE)
        adv_lay.addWidget(_ch)
        adv_lay.addWidget(self._desc(T.MT_CHANNELS_DESC))
        axes_row = QHBoxLayout()
        axes_row.addWidget(QLabel(T.MT_CHANNELS_AXES))
        self.qc_axes_label = QLabel(T.MT_CHANNELS_AXES_UNKNOWN)
        self.qc_axes_label.setStyleSheet(DIM_LABEL_STYLE)
        axes_row.addWidget(self.qc_axes_label, 1)
        adv_lay.addLayout(axes_row)
        self._channel_container = QVBoxLayout()
        self._channel_container.setSpacing(2)
        adv_lay.addLayout(self._channel_container)
        self._channel_rows: list = []

        self.reset_original_btn = QPushButton(T.RESET_BTN)
        adv_lay.addWidget(self.reset_original_btn)
        self._mt_adv_collapsible.addWidget(adv)
        lay.addWidget(self._mt_adv_collapsible)
        lay.addStretch(1)
        self.mt_right_scroll.setWidget(content)
        right_lay.addWidget(self.mt_right_scroll, 1)

        # -- the pinned foot (.rfoot.f6) --
        foot = QWidget()
        foot.setObjectName("step4MtFoot")
        foot.setStyleSheet(
            "QWidget#step4MtFoot { border-top: 1px solid #555555; }")
        self.mt_foot = foot
        f_lay = QVBoxLayout(foot)
        f_lay.setContentsMargins(0, 6, 0, 0)
        f_lay.setSpacing(5)
        _fin = QLabel(T.FINISH_TITLE)
        _fin.setStyleSheet(SECTION_HEADER_STYLE)
        f_lay.addWidget(_fin)
        f_lay.addWidget(self._desc(T.FINISH_DESC))
        self.save_tiers_btn = QPushButton(T.SAVE_BTN)
        self.finalize_btn = QPushButton(T.FINALIZE_BTN)
        f_lay.addLayout(self._grow(self.save_tiers_btn, self.finalize_btn))
        self.finalize_note = _ElidedLabel()
        self.finalize_note.setStyleSheet(AMBER_NOTE_STYLE)
        self.finalize_note.setVisible(False)
        f_lay.addWidget(self.finalize_note)
        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.setSpacing(8)
        self.qc_progress = QProgressBar()
        self.qc_progress.setRange(0, 100)
        self.qc_progress.setValue(0)
        self.qc_progress.setTextVisible(True)
        self.qc_progress.setFormat("")          # blank before any job
        self.qc_progress.setFixedHeight(14)
        bar_row.addWidget(self.qc_progress, 1)
        self.export_btn = QPushButton(T.EXPORT_OPEN_BTN)
        self.export_btn.setFixedHeight(22)
        bar_row.addWidget(self.export_btn)
        f_lay.addLayout(bar_row)
        self.qc_status_line = _ElidedLabel()
        self.qc_status_line.setStyleSheet(STATUS_LABEL_STYLE)
        f_lay.addWidget(self.qc_status_line)
        right_lay.addWidget(foot)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SPLIT_LEFT_W, SPLIT_RIGHT_W])
        outer.addWidget(splitter, 1)

        self._sigma_vox = [SIGMA_MEDIUM[0], SIGMA_MEDIUM[1]]
        self._sigma_scale = [1.0, 1.0]
        self._sigma_in_um = False
        self._sigma_quiet = False
        self._show_sigmas()
        self.smooth_strength_combo.activated.connect(self._on_strength_chosen)
        self.smooth_sigma_z_sb.valueChanged.connect(self._on_sigma_edited)
        self.smooth_sigma_xy_sb.valueChanged.connect(self._on_sigma_edited)
        self.qc_stacks_list.itemClicked.connect(self._on_mt_row_clicked)
        self.show_stack_card()
        return panel

    # ------------------------------------------------------------------
    # Diagnostics panel
    # ------------------------------------------------------------------

    def _build_diagnostics_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # GPU status
        _hdr = QLabel("System")
        _hdr.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(_hdr)
        self.gpu_status_label = QLabel("GPU: checking...")
        self.gpu_status_label.setStyleSheet(MONO_LABEL_STYLE)
        layout.addWidget(self.gpu_status_label)
        self._detect_gpu()

        _sep = QLabel("")
        _sep.setFixedHeight(1)
        _sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        layout.addWidget(_sep)

        # Log output
        _hdr = QLabel("Log Output")
        _hdr.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(_hdr)
        _desc = QLabel(T.DIAG_LOG_DESC)
        _desc.setWordWrap(True)
        _desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(_desc)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(LOG_OUTPUT_STYLE)
        self.log_output.setMinimumHeight(220)
        layout.addWidget(self.log_output, 1)
        return panel

    def _detect_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self.gpu_status_label.setText(f"GPU: {name}")
                self.gpu_status_label.setStyleSheet(
                    MONO_LABEL_STYLE.rstrip(";") + "; color: #4caf50;"
                )
            else:
                self.gpu_status_label.setText("GPU: not available (CPU only)")
                self.gpu_status_label.setStyleSheet(
                    MONO_LABEL_STYLE.rstrip(";") + "; color: #ff9800;"
                )
        except ImportError:
            self.gpu_status_label.setText("GPU: torch not installed")
            self.gpu_status_label.setStyleSheet(
                MONO_LABEL_STYLE.rstrip(";") + "; color: #ff9800;"
            )
        except Exception:
            self.gpu_status_label.setText("GPU: unknown")
            self.gpu_status_label.setStyleSheet(MONO_LABEL_STYLE)


    # ------------------------------------------------------------------
    # The three stack lists
    # ------------------------------------------------------------------

    def _stack_lists(self):
        return (self.session_stacks_list, self.stacks_list,
                self.qc_stacks_list)

    def set_tier_counts(self, mapping: dict, *, replace: bool = False) -> None:
        """The bucket counts of the measured rows (A3) --
        ``{stack name: ((word, n, (r, g, b)), ...) or None}`` -- kept in a
        map every rebuild of ① and ③ reads, and put on the rows now.  ②
        (what to segment) shows none.  *replace* drops every stack the
        mapping does not name."""
        if replace or not hasattr(self, "_tier_count_items"):
            self._tier_count_items = {}
        for k, v in (mapping or {}).items():
            self._tier_count_items[str(k)] = v
        for lw in (self.session_stacks_list, self.qc_stacks_list):
            lw.blockSignals(True)
            try:
                for i in range(lw.count()):
                    it = lw.item(i)
                    rec = it.data(ROLE_RECORD)
                    if rec is None:
                        continue
                    it.setData(ROLE_RECORD, self._with_tier_counts(
                        rec, it.data(ROLE_NAME) or rec.get("name")))
            finally:
                lw.blockSignals(False)
            lw.doItemsLayout()
            lw.viewport().update()
        self._fit_session_list_height()

    def set_loaded_stack(self, name) -> None:
        """Mark (3)'s row of the stack LOADED in the viewer (A3 fold;
        ``None`` clears it).  Kept, so the list's rebuilds re-apply it;
        exactly one row carries it, and (1) / (2) never do."""
        self._loaded_stack_name = name or None
        lw = self.qc_stacks_list
        lw.blockSignals(True)
        try:
            for i in range(lw.count()):
                it = lw.item(i)
                rec = it.data(ROLE_RECORD)
                if rec is None:
                    continue
                it.setData(ROLE_RECORD, self._with_loaded(
                    rec, it.data(ROLE_NAME)))
        finally:
            lw.blockSignals(False)
        lw.viewport().update()

    def _with_loaded(self, rec: dict, name) -> dict:
        rec = dict(rec)
        if name and name == getattr(self, "_loaded_stack_name", None):
            rec["loaded"] = True
        else:
            rec.pop("loaded", None)
        return rec

    def _with_tier_counts(self, rec: dict, name) -> dict:
        rec = dict(rec)
        items = getattr(self, "_tier_count_items", {}).get(str(name))
        if items:
            rec["tier_counts"] = tuple(items)
            rec["tier_words"] = True        # dot · number · word, wrapped
        else:
            rec.pop("tier_counts", None)
            rec.pop("tier_words", None)
        return rec

    def refresh_stack_lists(self, rows, *, ticked=()) -> None:
        """Rebuild the three lists from immutable row records, under
        ``blockSignals``.  ① and ② list every row and are checkable
        (a row that cannot run is built without ``ItemIsUserCheckable``);
        ③ lists the reconstructed-or-later rows and is not checkable.  The
        highlighted row of each list and the thumbnails (A3: on all three
        lists) survive."""
        ticked = set(ticked or ())
        carried = {}
        for lw0 in self._stack_lists():
            for i in range(lw0.count()):
                it = lw0.item(i)
                pm = it.data(ROLE_THUMB)
                rec = it.data(ROLE_RECORD) or {}
                if pm is not None and rec.get("path"):
                    carried.setdefault(_norm(rec["path"]), pm)
        lists = (
            (self.session_stacks_list, [r for r in rows], True),
            (self.stacks_list, [r for r in rows], True),
            (self.qc_stacks_list, [r for r in rows if r.get("measure_listed")],
             False),
        )
        for lw, lrows, checkable in lists:
            cur = lw.currentItem()
            keep = cur.data(ROLE_NAME) if cur is not None else None
            thumbs = carried
            lw.blockSignals(True)
            sel = lw.selectionModel()
            if sel is not None:
                sel.blockSignals(True)
            try:
                lw.clear()
                for rec in lrows:
                    item = QListWidgetItem()
                    item.setData(ROLE_NAME, rec.get("name"))
                    item.setData(ROLE_STATUS, rec.get("stored_status"))
                    if lw is self.qc_stacks_list:
                        item.setData(ROLE_BASE, dict(rec))
                        item.setData(ROLE_RECORD, self._with_loaded(
                            self._with_tier_counts(
                                self._mt_row_record(rec), rec.get("name")),
                            rec.get("name")))
                    elif lw is self.session_stacks_list:
                        item.setData(ROLE_RECORD, self._with_tier_counts(
                            dict(rec), rec.get("name")))
                    else:
                        item.setData(ROLE_RECORD, dict(rec))
                    flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                    is_ticked = rec.get("name") in ticked
                    # a greyed row that is ticked shows its
                    # tick and can be UNticked; it is never tickable
                    if (checkable and (rec.get("tickable", True)
                                       or is_ticked)
                            and not getattr(self, "_surface_locked",
                                            False)):
                        flags |= Qt.ItemIsUserCheckable
                    item.setFlags(flags)
                    if checkable:
                        item.setData(
                            Qt.CheckStateRole,
                            Qt.Checked if is_ticked else Qt.Unchecked)
                    pm = thumbs.get(_norm(rec.get("path") or ""))
                    if pm is not None:
                        item.setData(ROLE_THUMB, pm)
                    lw.addItem(item)
                    if keep is not None and rec.get("name") == keep:
                        lw.setCurrentItem(item)
            finally:
                if sel is not None:
                    sel.blockSignals(False)
                lw.blockSignals(False)
            lw.doItemsLayout()
        self._fit_session_list_height()
        print(f"[STEP4-UI] lists: {len(rows)} rows, {len(ticked)} ticked")

    def update_row(self, name: str, record: dict) -> None:
        """Replace one row's record in ① and ② (a running row's progress),
        without rebuilding the lists."""
        for lw in (self.session_stacks_list, self.stacks_list):
            for i in range(lw.count()):
                it = lw.item(i)
                if it.data(ROLE_NAME) == name:
                    lw.blockSignals(True)
                    try:
                        rec = dict(record)
                        if lw is self.session_stacks_list:
                            rec = self._with_tier_counts(rec, name)
                        it.setData(ROLE_RECORD, rec)
                    finally:
                        lw.blockSignals(False)
            # A1-9: a progress tick repaints; the row's height is unchanged
            lw.viewport().update()

    def set_row_thumbnail(self, stack_path: str, pixmap) -> bool:
        """Put a thumbnail on the rows of *stack_path* in ①, ② and ③ (rows
        are keyed by path, never by name).  False when no ① row has that
        path."""
        key = _norm(stack_path)
        found = False
        for lw in self._stack_lists():
            for i in range(lw.count()):
                it = lw.item(i)
                rec = it.data(ROLE_RECORD) or {}
                if _norm(rec.get("path") or "") == key:
                    lw.blockSignals(True)
                    try:
                        it.setData(ROLE_THUMB, pixmap)
                    finally:
                        lw.blockSignals(False)
                    lw.viewport().update()
                    found = found or lw is self.session_stacks_list
        return found

    def _fit_session_list_height(self) -> None:
        """① scrolls as a section, so its list takes its rows' height (up to
        ``SESSION_LIST_MAX_H``; it scrolls inside itself past that)."""
        lw = self.session_stacks_list
        if lw.count() == 0:
            lw.setFixedHeight(72)
            return
        total = 2 * lw.frameWidth() + 4
        for i in range(lw.count()):
            total += max(lw.sizeHintForRow(i), 0)
        lw.setFixedHeight(max(72, min(total, self._session_list_cap())))

    def _session_list_cap(self) -> int:
        """``SESSION_LIST_MAX_H``, less the lock line while one shows (B2
        owner recommendation (a)): the list gives up exactly the room the
        lock line takes, so Pipeline status stays on screen (F4) idle AND
        under a lock."""
        line = self.session_lock_line
        if line.isHidden() or not line.text():
            return SESSION_LIST_MAX_H
        parent = line.parentWidget()
        width = max((parent.width() if parent is not None else 566) - 12,
                    200)
        lay = line.layout()
        h = (lay.totalHeightForWidth(width) if lay is not None
             and lay.hasHeightForWidth() else line.sizeHint().height())
        spacing = 0
        playout = parent.layout() if parent is not None else None
        if playout is not None:
            spacing = max(playout.spacing(), 0)
        return max(72, SESSION_LIST_MAX_H - h - spacing)

    def ticked_names(self) -> list:
        """The ticked names, from ②'s list (the two are mirrored)."""
        out = []
        for i in range(self.stacks_list.count()):
            it = self.stacks_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(ROLE_NAME))
        return out

    def _on_row_tick_changed(self, item) -> None:
        """One bound slot for ① and ②: mirror the tick into the other
        list under ``blockSignals`` and report the ticked names."""
        src = self.sender()
        other = (self.stacks_list if src is self.session_stacks_list
                 else self.session_stacks_list)
        name = item.data(ROLE_NAME)
        state = item.checkState()
        rec = item.data(ROLE_RECORD) or {}
        # a greyed row, once unticked, cannot be ticked again
        drop = state != Qt.Checked and not rec.get("tickable", True)
        for lw in [x for x in (src, other) if x is not None]:
            lw.blockSignals(True)
            try:
                for i in range(lw.count()):
                    it = lw.item(i)
                    if it.data(ROLE_NAME) != name:
                        continue
                    if lw is other:
                        # the enum through setData: ``setCheckState``
                        # stores a bare int under PyQt6, which the row
                        # painter's ``Qt.Checked`` compare never matches
                        it.setData(Qt.CheckStateRole, state)
                    if drop:
                        it.setFlags(it.flags() & ~Qt.ItemIsUserCheckable)
            finally:
                lw.blockSignals(False)
        other.viewport().update()
        print(f"[STEP4-UI] tick {name} -> {state == Qt.Checked}")
        self.ticks_changed.emit(self.ticked_names())

    def _on_row_eye(self, name: str) -> None:
        self.stack_eye_clicked.emit(name)

    def _on_row_menu(self, name: str, pos) -> None:
        self.stack_menu_requested.emit(name, pos)

    def _wire_lists(self) -> None:
        self.session_stacks_list.itemChanged.connect(self._on_row_tick_changed)
        self.stacks_list.itemChanged.connect(self._on_row_tick_changed)
        for d in (self._session_delegate, self._seg_delegate):
            d.eye_clicked.connect(self._on_row_eye)
        self._session_delegate.menu_requested.connect(self._on_row_menu)
        # A3 fold: a right-click opens the same row menu as the ⋯
        self.session_stacks_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_stacks_list.customContextMenuRequested.connect(
            self._on_session_context)

    def _on_session_context(self, pos) -> None:
        lw = self.session_stacks_list
        item = lw.itemAt(pos)
        if item is None:
            return
        self._on_row_menu(item.data(ROLE_NAME),
                          lw.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------
    # ① and ② setters (the controller computes, the panel shows)
    # ------------------------------------------------------------------

    def show_session_summary(self, main: str, amber: str, counts: dict,
                             n_ticked: int, spacing_note: str = "") -> None:
        """①'s status line, pipeline counts, ticked count and Set spacing's
        reason line."""
        self.session_stacks_status.setText(rich_amber(main, amber))
        for status, (lbl, num) in self.pipeline_count_labels.items():
            num.setText(T.fmt_count(counts.get(status, 0)))
            if status == "failed":
                shown = counts.get("failed", 0) > 0
                lbl.setVisible(shown)
                num.setVisible(shown)
        self.session_ticked_label.setText(T.TICKED_COUNT.format(n=n_ticked))
        self.set_spacing_note.setText(spacing_note)
        self.set_spacing_note.setVisible(bool(spacing_note))
        self.set_spacing_btn.setEnabled(
            not spacing_note
            and not getattr(self, "_surface_locked", False))

    def show_run_row(self, n_stacks: int) -> None:
        self.run_picker_combo.setItemText(
            0, RUN_ROW_DEFAULT.format(n=n_stacks,
                                      units=pipe_units("stack", n_stacks)))

    def show_model_card(self, text: str, colour: str, trained: str,
                        missing: bool = False) -> None:
        """The ``Active:`` line and the trained-on line; a model
        whose files are missing turns the second line amber."""
        self.model_active_indicator_label.setText(text)
        if missing:
            colour = _AMBER          # F10: MODEL_MISSING turns the card amber
        self.model_active_indicator_label.setStyleSheet(
            f"color: {colour}; font-size: 11px; padding: 2px 0px;")
        self.model_trained_label.setText(trained)
        self.model_trained_label.setStyleSheet(
            AMBER_NOTE_STYLE if missing else DIM_LABEL_STYLE)

    def show_ticked_line(self, dim: str, amber: str) -> None:
        if dim and amber:
            dim = dim + T.SEG_TICKED_JOIN
        self.seg_ticked_label.setText(rich_amber(dim, amber))

    def show_seg_status(self, text: str, *, active: bool = False) -> None:
        """②'s status, green while a job runs.  It wraps to at most two
        lines (F3); past that the second line is elided and the whole
        sentence is the tooltip."""
        lbl = self.run_status_label
        style = MONO_LABEL_STYLE if not active else (
            MONO_LABEL_STYLE.rstrip(";") + "; color: #6abf69;")
        lbl.setStyleSheet(style)
        lbl.setToolTip(text)
        lbl.setText(self._two_lines(lbl, text))
        lbl.setMaximumHeight(2 * lbl.fontMetrics().lineSpacing() + 4)

    @staticmethod
    def _two_lines(lbl, text: str) -> str:
        """*text* cut to what two wrapped lines of *lbl* can hold."""
        width = max(lbl.width(), 200)
        lbl.ensurePolished()        # the style sheet's font, not the default
        fm = lbl.fontMetrics()
        if fm.horizontalAdvance(text) <= width:
            return text
        words = text.split(" ")
        first = ""
        k = 0
        for k, word in enumerate(words):
            trial = (first + " " + word) if first else word
            if first and fm.horizontalAdvance(trial) > width:
                break
            first = trial
        else:
            return text
        rest = " ".join(words[k:])
        return first + "\n" + fm.elidedText(rest, Qt.ElideRight, width)

    def show_cell_width_px(self, text: str) -> None:
        self.cell_width_px_label.setText(text)
        self.cell_width_px_label.setVisible(bool(text))

    def set_lock_lines(self, text: str = "") -> None:
        """The same job sentence in every section's lock line, or none."""
        for line in (self.session_lock_line, self.seg_lock_line,
                     self.mt_lock_line):
            line.setText(text)
        self._fit_session_list_height()     # (a): ① keeps its foot on screen

    #: ① and ② widgets the lock greys outright; their only
    #: other enable rule is Set spacing…'s, re-applied on unlock.
    _SURFACE_LOCK_WIDGETS = ("workdir_line", "browse_workdir_btn",
                             "add_folder_btn", "set_spacing_btn",
                             "run_picker_combo", "model_change_btn")

    #: The setting inputs the lock greys one by one (K5: never the ?
    #: buttons or the Advanced disclosures in the same column).
    _SETTING_INPUT_TYPES = (QAbstractSpinBox, QComboBox, QCheckBox,
                            QLineEdit)

    def _setting_inputs(self):
        """The TOP-LEVEL setting inputs of ②'s right column (H1): a spin
        box's or combo's own inner QLineEdit is not one -- disabling it
        separately left a spin box that no longer took typing."""
        types = self._SETTING_INPUT_TYPES
        return [w for w in self.seg_right_content.findChildren(QWidget)
                if isinstance(w, types)
                and not isinstance(w.parentWidget(), types)]

    def set_surface_locked(self, locked: bool, eye_live: bool = False
                           ) -> None:
        """B2: while a Step 4 job runs, ① and ②'s lock set is grey --
        the workspace row, Add folder, Set spacing…, the run row, the model
        card's Change…, every setting input (each one's own enable state
        is remembered and comes back on unlock; the ? buttons and the
        Advanced disclosures stay live, K5) and the ticks
        (``ItemIsUserCheckable`` off, quietly).  *eye_live*: the rows' eye
        is live (Predict / Reconstruct only, K3).  The lock line in each
        section is the reason on screen (I-9).  Predict / Reconstruct /
        Cancel / View follow their own gates."""
        locked = bool(locked)
        eye_locked = locked and not eye_live
        for lw in (self.session_stacks_list, self.stacks_list):
            if bool(lw.property("step4EyeLocked")) != eye_locked:
                lw.setProperty("step4EyeLocked", eye_locked)
                lw.viewport().update()
        if locked == getattr(self, "_surface_locked", False):
            return
        self._surface_locked = locked
        for name in self._SURFACE_LOCK_WIDGETS:
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(not locked)
        if not locked:
            # Set spacing…'s own rule: grey while its note shows.
            self.set_spacing_btn.setEnabled(
                not self.set_spacing_note.text())
        if locked:
            # H1: each widget's OWN state (WA_ForceDisabled), not
            # isEnabled(), which a disabled parent also turns False; kept
            # with the widget object itself, never an id()
            self._setting_prior = [
                (w, not w.testAttribute(Qt.WA_ForceDisabled))
                for w in self._setting_inputs()]
            for w, _on in self._setting_prior:
                w.setEnabled(False)
        else:
            for w, on in getattr(self, "_setting_prior", None) or []:
                try:
                    w.setEnabled(on)
                except RuntimeError:
                    pass
            self._setting_prior = []
            # M1: re-run the column's own gates, so nothing set during
            # the lock is frozen by a blind restore
            self._update_flow_controls()
            self._update_size_off_note()
        self._apply_tick_lock()
        print(f"[STEP4-LOCK] surface locked={locked}")

    def _apply_tick_lock(self) -> None:
        """The ① and ② ticks follow the lock: a row's box is checkable only
        when its record is tickable (or, greyed but ticked, so
        it can be unticked) AND no job runs.  Under
        ``blockSignals``, so no tick is reported."""
        locked = bool(getattr(self, "_surface_locked", False))
        for lw in (self.session_stacks_list, self.stacks_list):
            lw.setProperty("step4Locked", locked)   # the delegate dims
            lw.blockSignals(True)
            try:
                for i in range(lw.count()):
                    it = lw.item(i)
                    rec = it.data(ROLE_RECORD) or {}
                    flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                    ticked = it.checkState() == Qt.Checked
                    if (rec.get("tickable", True) or ticked) and not locked:
                        flags |= Qt.ItemIsUserCheckable       # a greyed row can still be unticked
                    it.setFlags(flags)
            finally:
                lw.blockSignals(False)
            lw.viewport().update()

    # ------------------------------------------------------------------
    # ③ setters (the controller computes, the panel shows)
    # ------------------------------------------------------------------

    def _on_mt_row_clicked(self, item) -> None:
        name = item.data(ROLE_NAME) if item is not None else None
        if name:
            print(f"[STEP4-UI] (3) row clicked: {name}")
            self.mt_load_requested.emit(name)

    def select_mt_row(self, name) -> None:
        """Highlight the loaded stack's ③ row (none: clear), quietly."""
        lw = self.qc_stacks_list
        lw.blockSignals(True)
        sel = lw.selectionModel()
        if sel is not None:
            sel.blockSignals(True)
        try:
            lw.clearSelection()
            for i in range(lw.count()):
                it = lw.item(i)
                if name and it.data(ROLE_NAME) == name:
                    lw.setCurrentItem(it)
                    it.setSelected(True)
                    break
        finally:
            if sel is not None:
                sel.blockSignals(False)
            lw.blockSignals(False)
        lw.viewport().update()

    def _mt_row_record(self, rec: dict) -> dict:
        """③'s row: the narrow row with ``ROW_UNSAVED`` on the loaded stack
        while it has unsaved work, and only the last-run warning."""
        out = dict(rec)
        head = T.ROW_LAST_RUN_FAILED.split("{", 1)[0]
        warns = [w for w in rec.get("warns_narrow") or ()
                 if w.startswith(head)]
        if self._mt_unsaved and rec.get("name") == self._mt_unsaved:
            warns.append(T.ROW_UNSAVED)
        out["warns_narrow"] = tuple(warns)
        return out

    def set_mt_unsaved(self, name) -> None:
        """The loaded stack carries "Edited, not saved" (or no row does)."""
        if name == self._mt_unsaved:
            return
        self._mt_unsaved = name
        lw = self.qc_stacks_list
        lw.blockSignals(True)
        try:
            for i in range(lw.count()):
                it = lw.item(i)
                base = it.data(ROLE_BASE) or it.data(ROLE_RECORD) or {}
                it.setData(ROLE_RECORD, self._with_loaded(
                    self._with_tier_counts(self._mt_row_record(base),
                                           it.data(ROLE_NAME)),
                    it.data(ROLE_NAME)))
        finally:
            lw.blockSignals(False)
        lw.doItemsLayout()

    def show_stack_card(self, name: str = "", pill: str = "", *,
                        pill_grey: bool = False, line: str = "",
                        line_amber: str = "", note: str = "") -> None:
        """The stack card (left column): name, pill, the dim line
        (its amber tail), an amber note.  No name: ``MT_NOTHING_LOADED``
        and Save's reason ``SAVE_NO_STACK``."""
        if not name:
            self.stack_card_name.setStyleSheet(CARD_DIM_NAME_STYLE)
            self.stack_card_name.setText(T.MT_NOTHING_LOADED)
            self.stack_card_pill.setVisible(False)
            self.stack_card_line.setText(rich_amber(T.SAVE_NO_STACK, ""))
            self.stack_card_note.setVisible(False)
            return
        self.stack_card_name.setStyleSheet(CARD_NAME_STYLE)
        self.stack_card_name.setText(name)
        self.stack_card_pill.setText(pill)
        self.stack_card_pill.setStyleSheet(
            PILL_GREY_STYLE if pill_grey else PILL_AMBER_STYLE)
        self.stack_card_pill.setVisible(bool(pill))
        self.stack_card_line.setText(rich_amber(line, line_amber))
        self.stack_card_note.setText(note)
        self.stack_card_note.setVisible(bool(note))

    def show_mt_chips(self, *, measure: str = "", measure_warn: bool = False,
                      review=None, tidy: str = "") -> None:
        """The three header chips; ``review`` is ``(good, bad, unmarked)``
        or None (not measured: no chip)."""
        self.mt_chip_measure.setStyleSheet(
            CHIP_WARN_STYLE if measure_warn else CHIP_STYLE)
        self.mt_chip_measure.setText(measure)
        self.mt_chip_review.setText(self.review_chip_html(review))
        self.mt_chip_tidy.setText(tidy)

    @staticmethod
    def review_chip_html(review) -> str:
        """``MT_CHIP_REVIEW`` with its three numbers in mono, coloured good
        green, bad red, unmarked grey (the v7 D drawing); a sentence (the
        stack is not measured, ``MT_CHIP_NEVER``) is shown as it is."""
        import html
        if isinstance(review, str):
            return html.escape(review)
        if not review:
            return ""
        import html
        mono = ("font-family: Consolas, 'Cascadia Mono', 'Courier New', "
                "monospace;")
        spans = {
            key: (f"<span style=\"color:{col}; {mono}\">"
                  f"{html.escape(T.fmt_count(val))}</span>")
            for key, val, col in zip(("g", "b", "u"), review, _REVIEW_COLOURS)
        }
        return html.escape(T.MT_CHIP_REVIEW).format(**spans)

    def set_mt_titles_dimmed(self, dimmed: bool) -> None:
        """The four block titles dim with their controls under a lock (the
        v7 D3 drawing dims the whole right column)."""
        style = MT_TITLE_DIM_STYLE if dimmed else SECTION_HEADER_STYLE
        for lbl in self._mt_titles:
            lbl.setStyleSheet(style)

    def show_not_reconstructed(self, text: str) -> None:
        self.mt_not_recon_label.setText(text)
        self.mt_not_recon_label.setVisible(bool(text))

    def show_finalize_note(self, text: str) -> None:
        """The foot's one amber sentence (``FINALIZE_NEEDS*``) or none."""
        self.finalize_note.setText(text)
        self.finalize_note.setVisible(bool(text))

    def show_mt_status(self, text: str, tip=None) -> None:
        """③'s one-line status (elided; the whole sentence is its tooltip)."""
        self.qc_status_line.setText(text, tip)

    def show_qc_bar(self, *, value=None, text=None, busy=None,
                    percent=False) -> None:
        """③'s one bar: ``busy`` = indeterminate; ``text`` its
        shown text (``BUSY_MEASURING``, ``QC_BAR_IDLE``, or blank);
        ``percent`` shows Qt's own percentage instead."""
        bar = self.qc_progress
        if busy is True:
            bar.setRange(0, 0)
        elif busy is False and bar.maximum() == 0:
            bar.setRange(0, 100)
        if value is not None:
            if bar.maximum() == 0:
                bar.setRange(0, 100)
            bar.setValue(int(value))
        if percent:
            bar.setFormat(QC_BAR_PERCENT)
        elif text is not None:
            bar.setFormat(text)

    def set_measure_button_text(self, text: str) -> None:
        self.extract_features_btn.setText(text)

    # -- the smoothing distances (Advanced, the strength combo) ------

    def set_sigma_spacing(self, zyx=None) -> None:
        """Show σ in µm for a stack with a voxel size (σ × the spacing on
        that axis), in voxels otherwise.  The engine always gets voxels."""
        scale = None
        if zyx:
            try:
                z, _y, x = (float(v) for v in zyx)
                scale = [z, x]
            except (TypeError, ValueError):
                scale = None
        self._sigma_in_um = scale is not None
        self._sigma_scale = scale or [1.0, 1.0]
        self._show_sigmas()

    def smooth_sigmas_voxels(self):
        """``(σ_z, σ_xy)`` in voxels, what the engine takes."""
        return float(self._sigma_vox[0]), float(self._sigma_vox[1])

    def _show_sigmas(self) -> None:
        self._sigma_quiet = True
        try:
            suffix = (T.SIGMA_SUFFIX_UM if self._sigma_in_um
                      else T.SIGMA_SUFFIX_VOX)
            pairs = ((self.smooth_sigma_z_sb, self._sigma_vox[0],
                      self._sigma_scale[0]),
                     (self.smooth_sigma_xy_sb, self._sigma_vox[1],
                      self._sigma_scale[1]))
            for box, vox, scale in pairs:
                box.setSuffix(suffix)
                box.setRange(SIGMA_VOX_RANGE[0] * scale,
                             SIGMA_VOX_RANGE[1] * scale)
                box.setSingleStep(0.1 if self._sigma_in_um else 0.5)
                box.setValue(vox * scale)
        finally:
            self._sigma_quiet = False

    def _on_strength_chosen(self, idx: int) -> None:
        if idx < 0 or idx >= len(SMOOTH_FACTORS):
            return                                  # "Custom" chosen again
        f = SMOOTH_FACTORS[idx]
        self._sigma_vox = [SIGMA_MEDIUM[0] * f, SIGMA_MEDIUM[1] * f]
        combo = self.smooth_strength_combo
        custom = combo.findText(T.SMOOTH_STRENGTH_CUSTOM)
        if custom >= 0:
            combo.blockSignals(True)
            try:
                combo.removeItem(custom)
                combo.setCurrentIndex(idx)
            finally:
                combo.blockSignals(False)
        self._show_sigmas()
        print(f"[STEP4-UI] smoothing strength {idx}: sigma {self._sigma_vox}")

    def _on_sigma_edited(self, *_):
        if self._sigma_quiet:
            return
        self._sigma_vox = [
            self.smooth_sigma_z_sb.value() / (self._sigma_scale[0] or 1.0),
            self.smooth_sigma_xy_sb.value() / (self._sigma_scale[1] or 1.0)]
        combo = self.smooth_strength_combo
        combo.blockSignals(True)
        try:
            if combo.findText(T.SMOOTH_STRENGTH_CUSTOM) < 0:
                combo.addItem(T.SMOOTH_STRENGTH_CUSTOM)
            combo.setCurrentIndex(combo.findText(T.SMOOTH_STRENGTH_CUSTOM))
        finally:
            combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Channel combos: blank stays blank
    # ------------------------------------------------------------------

    _CHOICE = "_epicycle_channel_choice"

    def update_channel_combos(self, n_channels: int, axes: str = "",
                              channel_names: list = None,
                              roles: dict = None) -> None:
        """Items ``"{i}: {name}"`` for ``i`` in ``0..n-1`` (``chan2`` keeps
        "none" first).  A combo restores the last name actually CHOSEN (a
        Qt property), never ``currentText()``, so a blank membrane channel
        stays blank through any number of refreshes."""
        n_channels = max(int(n_channels or 0), 0)
        names = list(channel_names or [])
        names += [None] * (n_channels - len(names))
        roles = dict(roles or {})

        def item(i):
            # a stack's layer name when one names the channel,
            # else the model's role, else "ch{i}"
            name, role = names[i], roles.get(i)
            if name and role:
                return T.SEG_CHANNEL_ITEM_ROLE.format(i=i, name=name,
                                                      role=role)
            if name:
                return T.SEG_CHANNEL_ITEM.format(i=i, name=name)
            if role:
                return T.SEG_CHANNEL_ITEM_ROLE_ONLY.format(i=i, role=role)
            return T.SEG_CHANNEL_ITEM.format(
                i=i, name=T.SEG_CHANNEL_NAME.format(i=i))

        items = [item(i) for i in range(n_channels)]
        for combo, head in ((self.chan1_combo, []),
                            (self.chan2_combo, [T.SEG_CHANNEL_NONE])):
            choice = combo.property(self._CHOICE)
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(head + items)
                idx = -1
                if choice is not None:
                    idx = self._index_of_choice(combo, choice, bool(head))
                if idx < 0 and head:
                    idx = 0            # chan2: "none"
                combo.setCurrentIndex(idx)
            finally:
                combo.blockSignals(False)

    @staticmethod
    def _index_of_choice(combo, choice, has_none: bool) -> int:
        """The item of a remembered choice: an int channel index (-1 =
        "none" for chan2)."""
        try:
            choice = int(choice)
        except (TypeError, ValueError):
            return -1
        if has_none:
            idx = choice + 1
        else:
            idx = choice
        return idx if 0 <= idx < combo.count() else -1

    def set_channel_choice(self, combo, channel) -> None:
        """Choose *channel* (an array index; ``None`` = "none" for chan2)
        and remember it."""
        has_none = combo is self.chan2_combo
        value = -1 if channel is None else int(channel)
        combo.setProperty(self._CHOICE, value)
        idx = self._index_of_choice(combo, value, has_none)
        if idx < 0 and has_none:
            idx = 0
        combo.setCurrentIndex(idx)

    def _remember_chan1(self, idx: int) -> None:
        self.chan1_combo.setProperty(self._CHOICE, idx if idx >= 0 else None)

    def _remember_chan2(self, idx: int) -> None:
        self.chan2_combo.setProperty(self._CHOICE,
                                     idx - 1 if idx >= 0 else None)

    def chosen_channels(self):
        """``(membrane_index or -1, nuclei_index or None)``."""
        c1 = self.chan1_combo.currentIndex()
        c2 = self.chan2_combo.currentIndex()
        return c1, (c2 - 1 if c2 > 0 else None)

    # ------------------------------------------------------------------
    # Channel row helpers (QC section)
    # ------------------------------------------------------------------

    def set_channel_rows(self, n_channels: int, axes_label: str = "") -> None:
        """Populate the intensity channels checklist.

        Called by the controller when a stack is loaded and the number of
        channels is known.
        """
        # Clear existing rows
        for cb, le in self._channel_rows:
            cb.setParent(None)
            le.setParent(None)
        self._channel_rows.clear()
        while self._channel_container.count():
            item = self._channel_container.takeAt(0)
            sub = item.layout() if item is not None else None
            if sub is not None:
                sub.setParent(None)

        if axes_label:
            self.qc_axes_label.setText(axes_label)

        for i in range(n_channels):
            row = QHBoxLayout()
            cb = QCheckBox(T.MT_CHANNEL_TICK)
            cb.setChecked(True)
            le = QLineEdit(T.SEG_CHANNEL_NAME.format(i=i))
            le.setMaximumWidth(120)
            row.addWidget(cb)
            row.addWidget(le, 1)
            self._channel_container.addLayout(row)
            self._channel_rows.append((cb, le))

    def get_checked_channels(self) -> list:
        """Return list of (channel_index, channel_name) for checked channels."""
        result = []
        for i, (cb, le) in enumerate(self._channel_rows):
            if cb.isChecked():
                name = le.text().strip() or T.SEG_CHANNEL_NAME.format(i=i)
                result.append((i, name))
        return result

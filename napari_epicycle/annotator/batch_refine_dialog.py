"""Floating Batch Refine dialog for the 3D curation workflow.

A non-modal, "dumb view" QWidget that lets the user pick a *target* set
of labels and build an ordered *recipe* of geometry-only refine steps
(SDF Smooth / Dilate-Erode / Fill Holes), then drives the
Preview / Run / Accept / Reject flow via **injected callbacks**.

Architecture (view-only):
- This widget owns NO volume data, NO worker, and NO threading.  The
  annotator (a later commit) injects the callbacks and drives the
  dialog's visual state via ``set_state`` / ``set_progress`` /
  ``set_status`` / ``set_preview_summary``.
- Buttons fire the injected callbacks; the dialog does NOT change its
  own logical state in response — the annotator calls ``set_state``
  after its work.  (It may optimistically disable the just-clicked
  button to prevent double-fire.)

Lifecycle mirrors ``annotator/merge_dialog.py``:
``setWindowFlag(Qt.Tool, True)``, non-modal ``.show()``, a
``_clean_close`` guard so a programmatic close (``mark_clean_close()``)
does NOT fire ``on_close`` while an X-button / parent-destroy close
does.
"""
from __future__ import annotations

from typing import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from packages.epicycle_annotator.batch_refine import (
    BATCH_CHAIN_LABEL,
    BATCH_CHAIN_NONE,
    BATCH_CHAIN_TIP,
    BATCH_CHANNEL_LABEL,
    BATCH_CHANNEL_TIP,
    BATCH_COUNT,
    BATCH_ORPHANS_RUNNING,
    BATCH_PREVIEW_BTN,
    BATCH_PREVIEW_BTN_NONE,
    BATCH_PREVIEW_NO_OBJECT,
    BATCH_PREVIEW_TIP,
    BATCH_REASON_PREVIEW_ONLY,
    BATCH_REASON_RUN_ONLY,
    BATCH_RECIPE_TITLE,
    BATCH_ROWS_OFF,
    BATCH_RUN_BTN,
    BATCH_SAVED_STEPS_LABEL,
    BATCH_KEEP_TIER_TIP,
    BATCH_LOAD_FAILED,
    BATCH_PLOT_EMPTY,
    BATCH_PLOT_NONE,
    BATCH_PLOT_OTHER_PATCH,
    BATCH_SAVE_FAILED,
    BATCH_SAVE_NO_NAME,
    BATCH_ADD_STEP,
    BATCH_CHAIN_UNREADABLE,
    BATCH_KEEP_TIER_LABEL,
    BATCH_RECENT_TIP,
    BATCH_SAVE_NO_WORKSPACE,
    BATCH_SAVE_PROMPT_LABEL,
    BATCH_SAVE_PROMPT_TITLE,
    BATCH_SAVED_TIP,
    BATCH_SAVE_OVERWRITING,
    BATCH_STEP_INVALID,
    BATCH_TARGET_ALL,
    BATCH_TARGET_PLOT,
    BATCH_SUMMARY,
    BATCH_TARGET_ADDED,
    BATCH_TARGET_SELECTED,
    BATCH_WARNING_BANNER,
    RefineStep,
    objects_word,
    summary_lines,
)
from napari_epicycle.ui_shell.theme import (
    DESTRUCTIVE_BTN_STYLE,
    INPUT_STYLE,
    PRIMARY_BTN_STYLE,
    PRIMARY_BTN_STYLE_GREEN,
    PROGRESS_BAR_STYLE,
    SCROLLBAR_STYLE,
    SECONDARY_BTN_STYLE,
)

# -----------------------------------------------------------------
# Theme — local style constants (mirrors merge_dialog.py)
# -----------------------------------------------------------------

_STEP3_ACCENT = "#d4a843"  # amber, Step 3 (Build)

_DIALOG_BG_STYLE = (
    "QWidget#BatchRefineDialogRoot {"
    "  background-color: #2b2b2b;"
    "  border: 1px solid rgba(255, 255, 255, 0.12);"
    "  border-radius: 6px;"
    "}"
)

_HEADER_TITLE_STYLE = (
    "font-weight: bold; font-size: 13px; color: #eeeeee; padding: 2px 0;"
)
_SECTION_TITLE_STYLE = (
    "font-weight: bold; font-size: 11px; color: #cccccc;"
    " letter-spacing: 1px; padding: 4px 0 2px 0;"
)
_HELP_STYLE = "color: #999999; font-size: 11px;"

# Amber warning banner — Step 3 accent fill + bold text.
_WARNING_STYLE = (
    "QLabel#BatchWarning {"
    "  background-color: rgba(212, 168, 67, 0.12);"
    "  border: 1px solid rgba(212, 168, 67, 0.30);"
    "  border-radius: 4px;"
    "  color: #e8c878;"
    "  font-size: 11px;"
    "  padding: 7px 9px;"
    "}"
)

_COUNT_STYLE = (
    "color: #e8c878; font-size: 12px; font-weight: bold; padding: 2px 0;"
)
_STATUS_STYLE = "color: #cccccc; font-size: 11px; padding: 2px 0;"
_SUMMARY_STYLE = (
    "color: #6abf69; font-size: 12px; font-weight: bold; padding: 2px 0;"
)

_ROW_FRAME_STYLE = (
    "QFrame#RecipeRow {"
    "  background-color: rgba(255, 255, 255, 0.03);"
    "  border: 1px solid rgba(255, 255, 255, 0.07);"
    "  border-radius: 4px;"
    "}"
)
_ROW_NUM_STYLE = (
    "color: #e8c878; font-size: 12px; font-weight: bold;"
    " font-family: 'Consolas', 'Cascadia Mono', monospace;"
    " min-width: 16px;"
)
_ICON_BTN_STYLE = (
    "QPushButton {"
    "  background-color: rgba(255, 255, 255, 0.05);"
    "  border: 1px solid rgba(255, 255, 255, 0.10);"
    "  border-radius: 3px; padding: 2px 6px;"
    "  font-size: 11px; color: #cccccc; min-width: 20px;"
    "}"
    "QPushButton:hover { background-color: rgba(255, 255, 255, 0.12); }"
    "QPushButton:disabled { color: #555555; }"
)

_SCROLL_AREA_STYLE = (
    "QScrollArea { background: transparent; border: none; }"
    "QScrollArea > QWidget > QWidget { background: transparent; }"
    + SCROLLBAR_STYLE
)

# Method combo entries → RefineStep.method strings.
_METHOD_DISPLAY = ["SDF Smooth", "Dilate/Erode", "Fill Holes"]
_METHOD_KEYS = ["sdf", "dilate_erode", "fill_holes"]
_AXES = ["3D", "XY", "XZ", "YZ"]


# -----------------------------------------------------------------
# Recipe row
# -----------------------------------------------------------------


class _RecipeRow(QFrame):
    """One step in the recipe — number + method combo + swappable params
    + reorder (▲/▼) + remove (✕).

    Params swap via a per-row ``QStackedWidget`` keyed by method index.
    Pure view: emits user intent (reorder / remove) through injected
    callbacks; builds a :class:`RefineStep` on demand via :meth:`to_step`.
    """

    def __init__(
        self,
        on_remove: Callable[["_RecipeRow"], None],
        on_move_up: Callable[["_RecipeRow"], None],
        on_move_down: Callable[["_RecipeRow"], None],
        initial_method: str = "sdf",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._on_remove = on_remove
        self._on_move_up = on_move_up
        self._on_move_down = on_move_down

        self.setObjectName("RecipeRow")
        self.setStyleSheet(_ROW_FRAME_STYLE)
        self.setFrameShape(QFrame.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self._num_lbl = QLabel("1", parent=self)
        self._num_lbl.setStyleSheet(_ROW_NUM_STYLE)
        layout.addWidget(self._num_lbl)

        self._method_combo = QComboBox(parent=self)
        self._method_combo.addItems(_METHOD_DISPLAY)
        self._method_combo.setStyleSheet(INPUT_STYLE)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        layout.addWidget(self._method_combo)

        # Param stack — one page per method, in _METHOD_KEYS order.
        self._param_stack = QStackedWidget(parent=self)
        self._param_stack.setStyleSheet("background: transparent;")
        self._build_sdf_page()
        self._build_dilate_page()
        self._build_fill_page()
        layout.addWidget(self._param_stack, stretch=1)

        # Reorder + remove controls.
        self._up_btn = QPushButton("▲", parent=self)  # ▲
        self._up_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._up_btn.setToolTip("Move step up")
        self._up_btn.clicked.connect(lambda: self._on_move_up(self))
        layout.addWidget(self._up_btn)

        self._down_btn = QPushButton("▼", parent=self)  # ▼
        self._down_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._down_btn.setToolTip("Move step down")
        self._down_btn.clicked.connect(lambda: self._on_move_down(self))
        layout.addWidget(self._down_btn)

        self._remove_btn = QPushButton("✕", parent=self)  # ✕
        self._remove_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._remove_btn.setToolTip("Remove step")
        self._remove_btn.clicked.connect(lambda: self._on_remove(self))
        layout.addWidget(self._remove_btn)

        # Select the initial method (defaults to SDF).
        try:
            idx = _METHOD_KEYS.index(initial_method)
        except ValueError:
            idx = 0
        self._method_combo.setCurrentIndex(idx)
        self._param_stack.setCurrentIndex(idx)

    # ---- param pages ----

    def _build_sdf_page(self) -> None:
        page = QWidget(parent=self._param_stack)
        page.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(QLabel("σz", parent=page))
        self._sdf_sigma_z = QDoubleSpinBox(parent=page)
        self._sdf_sigma_z.setRange(0.1, 10.0)
        self._sdf_sigma_z.setSingleStep(0.1)
        self._sdf_sigma_z.setValue(1.5)
        self._sdf_sigma_z.setStyleSheet(INPUT_STYLE)
        lay.addWidget(self._sdf_sigma_z)

        lay.addWidget(QLabel("σxy", parent=page))
        self._sdf_sigma_xy = QDoubleSpinBox(parent=page)
        self._sdf_sigma_xy.setRange(0.1, 10.0)
        self._sdf_sigma_xy.setSingleStep(0.1)
        self._sdf_sigma_xy.setValue(1.5)
        self._sdf_sigma_xy.setStyleSheet(INPUT_STYLE)
        lay.addWidget(self._sdf_sigma_xy)

        self._sdf_topo = QCheckBox("topo", parent=page)
        self._sdf_topo.setChecked(True)
        self._sdf_topo.setToolTip("Preserve topology (skeleton restoration).")
        lay.addWidget(self._sdf_topo)
        lay.addStretch(1)

        self._param_stack.addWidget(page)  # index 0

    def _build_dilate_page(self) -> None:
        page = QWidget(parent=self._param_stack)
        page.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(QLabel("radius", parent=page))
        self._de_radius = QSpinBox(parent=page)
        self._de_radius.setRange(-10, 10)
        self._de_radius.setSingleStep(1)
        self._de_radius.setValue(1)
        self._de_radius.setToolTip("Positive = dilate, negative = erode.")
        self._de_radius.setStyleSheet(INPUT_STYLE)
        lay.addWidget(self._de_radius)

        lay.addWidget(QLabel("axis", parent=page))
        self._de_axis = QComboBox(parent=page)
        self._de_axis.addItems(_AXES)
        self._de_axis.setStyleSheet(INPUT_STYLE)
        lay.addWidget(self._de_axis)
        lay.addStretch(1)

        self._param_stack.addWidget(page)  # index 1

    def _build_fill_page(self) -> None:
        page = QWidget(parent=self._param_stack)
        page.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(QLabel("axis", parent=page))
        self._fill_axis = QComboBox(parent=page)
        self._fill_axis.addItems(_AXES)
        self._fill_axis.setStyleSheet(INPUT_STYLE)
        lay.addWidget(self._fill_axis)
        lay.addStretch(1)

        self._param_stack.addWidget(page)  # index 2

    # ---- signal handlers ----

    def _on_method_changed(self, idx: int) -> None:
        self._param_stack.setCurrentIndex(idx)

    # ---- public API ----

    def set_step_number(self, n: int) -> None:
        self._num_lbl.setText(str(n))

    def set_reorder_enabled(self, *, can_up: bool, can_down: bool) -> None:
        self._up_btn.setEnabled(can_up)
        self._down_btn.setEnabled(can_down)

    def set_editable(self, editable: bool) -> None:
        """Lock/unlock the whole row (used during running / staged)."""
        self._method_combo.setEnabled(editable)
        self._param_stack.setEnabled(editable)
        self._up_btn.setEnabled(editable)
        self._down_btn.setEnabled(editable)
        self._remove_btn.setEnabled(editable)

    def method_key(self) -> str:
        return _METHOD_KEYS[self._method_combo.currentIndex()]

    def to_step(self) -> RefineStep:
        """Build a :class:`RefineStep` from the current row widgets."""
        key = self.method_key()
        if key == "sdf":
            params = {
                "sigma_z": float(self._sdf_sigma_z.value()),
                "sigma_xy": float(self._sdf_sigma_xy.value()),
                "preserve_topology": bool(self._sdf_topo.isChecked()),
            }
        elif key == "dilate_erode":
            params = {
                "radius": int(self._de_radius.value()),
                "axis": self._de_axis.currentText(),
            }
        else:  # fill_holes
            params = {"axis": self._fill_axis.currentText()}
        return RefineStep(method=key, params=params)

    def apply_step(self, step: RefineStep) -> None:
        """Populate the row widgets FROM a :class:`RefineStep` (inverse of
        :meth:`to_step`).  Used by ``BatchRefineDialog.set_recipe`` to reload
        a recipe into the builder.  Missing params fall back to the existing
        widget defaults; unknown methods are ignored (clamp to SDF)."""
        key = getattr(step, "method", "sdf")
        try:
            idx = _METHOD_KEYS.index(key)
        except ValueError:
            idx = 0
        self._method_combo.setCurrentIndex(idx)
        self._param_stack.setCurrentIndex(idx)
        p = dict(getattr(step, "params", {}) or {})
        if key == "sdf":
            if "sigma_z" in p:
                self._sdf_sigma_z.setValue(float(p["sigma_z"]))
            if "sigma_xy" in p:
                self._sdf_sigma_xy.setValue(float(p["sigma_xy"]))
            if "preserve_topology" in p:
                self._sdf_topo.setChecked(bool(p["preserve_topology"]))
        elif key == "dilate_erode":
            if "radius" in p:
                self._de_radius.setValue(int(p["radius"]))
            if "axis" in p:
                self._de_axis.setCurrentText(str(p["axis"]))
        elif key == "fill_holes":
            if "axis" in p:
                self._fill_axis.setCurrentText(str(p["axis"]))


# -----------------------------------------------------------------
# Dialog
# -----------------------------------------------------------------


class BatchRefineDialog(QWidget):
    """Floating, non-modal Batch Refine view.

    A dumb view: it builds a target spec + an ordered recipe and exposes
    them via :meth:`target_spec` / :meth:`recipe`, firing injected
    callbacks for Preview / Run / Cancel / Accept / Reject.  The owning
    annotator drives the visual state via :meth:`set_state`,
    :meth:`set_progress`, :meth:`set_status`, and
    :meth:`set_preview_summary`.

    States (driven by :meth:`set_state`):
        ``"idle"``     target + recipe editable; Preview / Run shown.
        ``"running"``  target + recipe locked; progress + Cancel shown.
        ``"staged"``   summary + Reject / Accept shown.
    """

    def __init__(
        self,
        *,
        count_provider: Callable[[dict], int],
        all_count: int,
        selected_count: int,
        available_tiers: list,
        on_preview_one: Callable[[], None],
        on_run: Callable[[], None],
        on_cancel: Callable[[], None],
        on_accept: Callable[[], None],
        on_reject: Callable[[], None],
        on_close: Callable[[], None],
        on_save_recipe: Callable[[str], None] | None = None,
        can_save_recipe: bool = False,
        on_chain_changed: Callable[[], None] | None = None,
        added_targets: bool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("BatchRefineDialogRoot")
        self.setWindowTitle("Batch Refine")
        self.setWindowFlag(Qt.Tool, True)  # tool window — on top, no taskbar
        self.setStyleSheet(_DIALOG_BG_STYLE)
        self.setMinimumWidth(540)
        self.setMinimumHeight(380)

        self._count_provider = count_provider
        self._all_count = int(all_count)
        self._selected_count = int(selected_count)
        self._plot_count = 0
        # Whether the patch has a QC plot at all (set by set_target_counts);
        # decides which reason the count line shows at a plot count of 0.
        self._plot_available = True
        # The plot shows ANOTHER patch (UX audit M6) -- a third reason.
        self._plot_other_patch = False
        self._available_tiers = list(available_tiers)
        self._on_preview_one_cb = on_preview_one
        self._on_run_cb = on_run
        self._on_cancel_cb = on_cancel
        self._on_accept_cb = on_accept
        self._on_reject_cb = on_reject
        self._on_close_cb = on_close
        self._on_save_recipe_cb = on_save_recipe
        self._can_save_recipe = bool(can_save_recipe)
        # Backing maps for the Recent / Saved dropdowns: combo-index →
        # recipe_json (set by set_recipe_history).  Index 0 is the
        # "(none)" placeholder in both → maps start at index 1.
        self._recent_recipes: list = []
        self._saved_recipes: dict = {}
        # The Refine-tab recipe row: combo index ->
        # (name, parsed Recipe | None).  Index 0 is the "none" placeholder.
        self._on_chain_changed_cb = on_chain_changed
        # Opened on explicit targets (Add's "Refine all ▸"): the Selected
        # radio names them "Added this session (N)".
        self._selected_template = (
            BATCH_TARGET_ADDED if added_targets else BATCH_TARGET_SELECTED)
        self._chain_items: list = []
        self._chain_needs_channel = False
        # Why Run / Preview are refused (the reason line), the objects still
        # finishing from a stopped chain run, and the preview object.
        self._run_reason = ""
        self._preview_reason = ""
        self._orphans_running = 0
        # None = never pushed (a standalone dialog): Preview behaves as it
        # always did.  The annotator pushes the viewer's selection.
        self._preview_object = None

        # closeEvent guard. ``mark_clean_close()`` sets this True so a
        # programmatic close does NOT fire ``on_close``.  An X-button
        # close / parent-destroy leaves it False → closeEvent fires
        # ``on_close`` so the annotator can treat it as a reject/teardown.
        self._clean_close = False

        self._rows: list[_RecipeRow] = []
        self._tier_checks: dict[str, QCheckBox] = {}
        self._state = "idle"

        self._build_ui()
        self._update_target_count()
        self.set_state("idle")

    # ---- UI construction --------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        # Header.
        title = QLabel("Batch Refine", parent=self)
        title.setStyleSheet(_HEADER_TITLE_STYLE)
        outer.addWidget(title)

        # Always-visible warning banner.
        warning = QLabel(BATCH_WARNING_BANNER, parent=self)
        warning.setObjectName("BatchWarning")
        warning.setStyleSheet(_WARNING_STYLE)
        warning.setWordWrap(True)
        outer.addWidget(warning)

        # ---- TARGET section ----
        self._target_section = QWidget(parent=self)
        self._target_section.setStyleSheet("background: transparent;")
        target_lay = QVBoxLayout(self._target_section)
        target_lay.setContentsMargins(0, 0, 0, 0)
        target_lay.setSpacing(4)

        target_title = QLabel("TARGET", parent=self._target_section)
        target_title.setStyleSheet(_SECTION_TITLE_STYLE)
        target_lay.addWidget(target_title)

        self._target_group = QButtonGroup(self)
        self._radio_all = QRadioButton(
            BATCH_TARGET_ALL.format(n=self._all_count),
            parent=self._target_section,
        )
        self._radio_selected = QRadioButton(
            self._selected_template.format(n=self._selected_count),
            parent=self._target_section,
        )
        # The ids the QC plot highlights right now, read live by the
        # annotator at every count refresh and at Run.
        self._radio_plot = QRadioButton(
            BATCH_TARGET_PLOT.format(n=self._plot_count),
            parent=self._target_section,
        )
        self._radio_tier = QRadioButton("By tier", parent=self._target_section)
        self._radio_all.setChecked(True)
        for rb in (
            self._radio_all, self._radio_selected, self._radio_plot,
            self._radio_tier,
        ):
            self._target_group.addButton(rb)
            # Only the radio that became CHECKED recounts: `toggled` fires
            # on the one unchecked too, which resolved twice (UX audit L6).
            rb.toggled.connect(self._on_radio_toggled)
            target_lay.addWidget(rb)

        # Tier checkbox row (enabled only in "By tier" mode).
        self._tier_row = QWidget(parent=self._target_section)
        self._tier_row.setStyleSheet("background: transparent;")
        tier_lay = QHBoxLayout(self._tier_row)
        tier_lay.setContentsMargins(20, 0, 0, 0)
        tier_lay.setSpacing(10)
        for tier in self._available_tiers:
            cb = QCheckBox(str(tier), parent=self._tier_row)
            cb.toggled.connect(self._on_target_changed)
            tier_lay.addWidget(cb)
            self._tier_checks[str(tier)] = cb
        tier_lay.addStretch(1)
        target_lay.addWidget(self._tier_row)

        # Live count.
        self._count_lbl = QLabel(BATCH_COUNT.format(k=objects_word(0)),
                                 parent=self._target_section)
        self._count_lbl.setStyleSheet(_COUNT_STYLE)
        # A long reason wraps instead of widening the dialog (UX audit M3).
        self._count_lbl.setWordWrap(True)
        target_lay.addWidget(self._count_lbl)

        outer.addWidget(self._target_section)

        # ---- RECIPE section ----
        self._recipe_section = QWidget(parent=self)
        self._recipe_section.setStyleSheet("background: transparent;")
        recipe_lay = QVBoxLayout(self._recipe_section)
        recipe_lay.setContentsMargins(0, 0, 0, 0)
        recipe_lay.setSpacing(6)

        recipe_title = QLabel(BATCH_RECIPE_TITLE, parent=self._recipe_section)
        recipe_title.setStyleSheet(_SECTION_TITLE_STYLE)
        recipe_lay.addWidget(recipe_title)

        # ---- The Refine-tab recipe row ----
        chain_row = QHBoxLayout()
        chain_row.setSpacing(6)
        chain_row.addWidget(QLabel(BATCH_CHAIN_LABEL, parent=self._recipe_section))
        self._chain_combo = QComboBox(parent=self._recipe_section)
        self._chain_combo.setStyleSheet(INPUT_STYLE)
        self._chain_combo.setToolTip(BATCH_CHAIN_TIP)
        self._chain_combo.addItem(BATCH_CHAIN_NONE)
        self._chain_combo.currentIndexChanged.connect(self._on_chain_selected)
        chain_row.addWidget(self._chain_combo, stretch=1)
        recipe_lay.addLayout(chain_row)

        # The recipe's steps in the Refine page's own words (read-only).
        self._chain_summary_lbl = QLabel("", parent=self._recipe_section)
        self._chain_summary_lbl.setStyleSheet(_STATUS_STYLE)
        self._chain_summary_lbl.setWordWrap(True)
        self._chain_summary_lbl.setVisible(False)
        recipe_lay.addWidget(self._chain_summary_lbl)

        # The channel the recipe's wall / brightness steps read.
        self._chain_channel_row = QWidget(parent=self._recipe_section)
        self._chain_channel_row.setStyleSheet("background: transparent;")
        ch_lay = QHBoxLayout(self._chain_channel_row)
        ch_lay.setContentsMargins(0, 0, 0, 0)
        ch_lay.setSpacing(6)
        ch_lay.addWidget(QLabel(BATCH_CHANNEL_LABEL, parent=self._chain_channel_row))
        self._chain_channel_combo = QComboBox(parent=self._chain_channel_row)
        self._chain_channel_combo.setStyleSheet(INPUT_STYLE)
        self._chain_channel_combo.setToolTip(BATCH_CHANNEL_TIP)
        self._chain_channel_combo.activated.connect(self._on_chain_channel_picked)
        ch_lay.addWidget(self._chain_channel_combo, stretch=1)
        self._chain_channel_row.setVisible(False)
        recipe_lay.addWidget(self._chain_channel_row)

        # In the builder's place while a recipe is chosen.
        self._rows_off_lbl = QLabel(BATCH_ROWS_OFF, parent=self._recipe_section)
        self._rows_off_lbl.setStyleSheet(_HELP_STYLE)
        self._rows_off_lbl.setWordWrap(True)
        self._rows_off_lbl.setVisible(False)
        recipe_lay.addWidget(self._rows_off_lbl)

        # ---- Recipe history row (Recent / Saved / Save) ----
        # Reload-only: picking a recipe repopulates the rows below; it does
        # NOT touch the target radios.  Recent = in-session (most-recent
        # first); Saved = persisted to the workspace via the controller.
        history_row = QHBoxLayout()
        history_row.setSpacing(6)

        history_row.addWidget(QLabel("Recent", parent=self._recipe_section))
        self._recent_combo = QComboBox(parent=self._recipe_section)
        self._recent_combo.setStyleSheet(INPUT_STYLE)
        self._recent_combo.setToolTip(BATCH_RECENT_TIP)
        self._recent_combo.activated.connect(self._on_recent_selected)
        history_row.addWidget(self._recent_combo, stretch=1)

        history_row.addWidget(
            QLabel(BATCH_SAVED_STEPS_LABEL, parent=self._recipe_section))
        self._saved_combo = QComboBox(parent=self._recipe_section)
        self._saved_combo.setStyleSheet(INPUT_STYLE)
        self._saved_combo.setToolTip(BATCH_SAVED_TIP)
        self._saved_combo.activated.connect(self._on_saved_selected)
        history_row.addWidget(self._saved_combo, stretch=1)

        self._save_recipe_btn = QPushButton("Save recipe…", parent=self._recipe_section)
        self._save_recipe_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        self._save_recipe_btn.setToolTip(
            "Name + persist the current recipe to this workspace."
        )
        self._save_recipe_btn.clicked.connect(self._on_save_recipe_clicked)
        # Disabled when no controller wired the save callback (standalone).
        self._save_recipe_btn.setEnabled(
            self._can_save_recipe and self._on_save_recipe_cb is not None
        )
        history_row.addWidget(self._save_recipe_btn)

        recipe_lay.addLayout(history_row)
        # Seed the placeholders so the combos render before set_recipe_history.
        self._recent_combo.addItem("Recent recipes…")
        self._saved_combo.addItem("Saved recipes…")

        self._scroll = QScrollArea(parent=self._recipe_section)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(_SCROLL_AREA_STYLE)
        self._scroll.setMinimumHeight(120)

        self._rows_container = QWidget(parent=self._scroll)
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        self._rows_layout.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._rows_container)
        recipe_lay.addWidget(self._scroll, stretch=1)

        # Add-step + preset controls.
        recipe_btn_row = QHBoxLayout()
        recipe_btn_row.setSpacing(8)
        self._add_btn = QPushButton("+ Add step", parent=self._recipe_section)
        self._add_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        self._add_btn.clicked.connect(self._on_add_step_clicked)
        recipe_btn_row.addWidget(self._add_btn)
        recipe_btn_row.addStretch(1)
        self._preset_btn = QPushButton("Smooth+Fill", parent=self._recipe_section)
        self._preset_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        self._preset_btn.setToolTip("Replace recipe with [SDF Smooth, Fill Holes].")
        self._preset_btn.clicked.connect(self._on_preset_clicked)
        recipe_btn_row.addWidget(self._preset_btn)
        recipe_lay.addLayout(recipe_btn_row)

        # Keep-tier option (lives in the recipe section → auto-locked during
        # a run / staged preview).  When checked, refined labels stay in
        # their current QC tier instead of being reclassified to "curated".
        self._keep_tier_cb = QCheckBox(
            BATCH_KEEP_TIER_LABEL, parent=self._recipe_section,
        )
        self._keep_tier_cb.setToolTip(BATCH_KEEP_TIER_TIP)
        self._keep_tier_cb.setStyleSheet("color: #cccccc; font-size: 11px;")
        recipe_lay.addWidget(self._keep_tier_cb)

        outer.addWidget(self._recipe_section, stretch=1)

        # Why Run / Preview are refused: the batch blocker, or the
        # objects still finishing from a stopped chain run.  On screen, since
        # Qt shows no tooltip on a disabled button.
        self._chain_reason_lbl = QLabel("", parent=self)
        self._chain_reason_lbl.setStyleSheet(_COUNT_STYLE)
        self._chain_reason_lbl.setWordWrap(True)
        self._chain_reason_lbl.setVisible(False)
        outer.addWidget(self._chain_reason_lbl)

        # Seed one default SDF row.
        self._append_row("sdf")

        # ---- Run controls (bottom) ----
        self._status_lbl = QLabel("", parent=self)
        self._status_lbl.setStyleSheet(_STATUS_STYLE)
        self._status_lbl.setWordWrap(True)
        outer.addWidget(self._status_lbl)

        self._summary_lbl = QLabel("", parent=self)
        self._summary_lbl.setStyleSheet(_SUMMARY_STYLE)
        self._summary_lbl.setWordWrap(True)
        outer.addWidget(self._summary_lbl)

        self._progress = QProgressBar(parent=self)
        self._progress.setStyleSheet(PROGRESS_BAR_STYLE)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        outer.addWidget(self._progress)

        # Button row.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._preview_btn = QPushButton(BATCH_PREVIEW_BTN_NONE, parent=self)
        self._preview_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        self._preview_btn.setToolTip(BATCH_PREVIEW_TIP)
        self._preview_btn.clicked.connect(self._on_preview_clicked)
        btn_row.addWidget(self._preview_btn)

        self._run_btn = QPushButton(BATCH_RUN_BTN.format(n=objects_word(0)),
                                    parent=self)
        self._run_btn.setStyleSheet(PRIMARY_BTN_STYLE)
        self._run_btn.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel", parent=self)
        self._cancel_btn.setStyleSheet(DESTRUCTIVE_BTN_STYLE)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self._cancel_btn)

        btn_row.addStretch(1)

        self._reject_btn = QPushButton("Reject", parent=self)
        self._reject_btn.setStyleSheet(DESTRUCTIVE_BTN_STYLE)
        self._reject_btn.clicked.connect(self._on_reject_clicked)
        btn_row.addWidget(self._reject_btn)

        self._accept_btn = QPushButton("Accept", parent=self)
        self._accept_btn.setStyleSheet(PRIMARY_BTN_STYLE_GREEN)
        self._accept_btn.clicked.connect(self._on_accept_clicked)
        btn_row.addWidget(self._accept_btn)

        outer.addLayout(btn_row)

        # A combo's minimum width is its LONGEST item by default, so a long
        # recipe name or summary widened the whole dialog (UX audit M3 on
        # the same page).  They shrink to a few characters and elide.
        for combo in (self._recent_combo, self._saved_combo,
                      self._chain_combo, self._chain_channel_combo):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)

    # ---- recipe row management --------------------------------

    def _append_row(self, method: str) -> _RecipeRow:
        row = _RecipeRow(
            on_remove=self._on_row_remove,
            on_move_up=self._on_row_move_up,
            on_move_down=self._on_row_move_down,
            initial_method=method,
            parent=self._rows_container,
        )
        self._rows.append(row)
        self._rows_layout.addWidget(row)
        self._renumber_rows()
        return row

    def _renumber_rows(self) -> None:
        n = len(self._rows)
        for i, row in enumerate(self._rows):
            row.set_step_number(i + 1)
            row.set_reorder_enabled(can_up=i > 0, can_down=i < n - 1)
        # Run is only meaningful with ≥1 step.
        self._refresh_run_enabled()

    def _on_add_step_clicked(self) -> None:
        self._append_row("sdf")

    def _on_preset_clicked(self) -> None:
        """Replace the recipe with the [SDF, Fill Holes] preset."""
        for row in list(self._rows):
            self._remove_row(row)
        self._append_row("sdf")
        self._append_row("fill_holes")

    def _on_row_remove(self, row: _RecipeRow) -> None:
        self._remove_row(row)
        self._renumber_rows()

    def _remove_row(self, row: _RecipeRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()

    def _on_row_move_up(self, row: _RecipeRow) -> None:
        idx = self._rows.index(row)
        if idx <= 0:
            return
        self._rows[idx - 1], self._rows[idx] = self._rows[idx], self._rows[idx - 1]
        self._reinsert_rows()

    def _on_row_move_down(self, row: _RecipeRow) -> None:
        idx = self._rows.index(row)
        if idx >= len(self._rows) - 1:
            return
        self._rows[idx + 1], self._rows[idx] = self._rows[idx], self._rows[idx + 1]
        self._reinsert_rows()

    def _reinsert_rows(self) -> None:
        """Re-pack rows into the layout in their (reordered) list order."""
        for row in self._rows:
            self._rows_layout.removeWidget(row)
        for row in self._rows:
            self._rows_layout.addWidget(row)
        self._renumber_rows()

    # ---- target handling --------------------------------------

    def _on_radio_toggled(self, checked) -> None:
        if checked:
            self._on_target_changed()

    def _on_target_changed(self, *_args) -> None:
        self._tier_row.setEnabled(self._radio_tier.isChecked())
        self._update_target_count()

    def _update_target_count(self) -> None:
        # Only while idle (UX audit L7): the 500 ms poll pushes the preview
        # object, and a recount then rewrote the count line and Run's text
        # under a running or staged batch.  `set_state("idle")` redraws
        # the line from the last count.
        if self._state != "idle":
            return
        try:
            k = int(self._count_provider(self.target_spec()))
        except Exception:
            k = 0
        self._last_count = k
        self._render_count_line()
        self._fire_chain_changed()

    def _render_count_line(self) -> None:
        """The count line, Run's text and the buttons from the LAST count
        -- no resolve."""
        k = int(getattr(self, "_last_count", 0) or 0)
        if k == 0 and self._radio_plot.isChecked():
            # Run is disabled at 0; this line is its reason.
            if not self._plot_available:
                reason = BATCH_PLOT_NONE
            elif self._plot_other_patch:
                reason = BATCH_PLOT_OTHER_PATCH
            else:
                reason = BATCH_PLOT_EMPTY
            self._count_lbl.setText(reason)
        else:
            self._count_lbl.setText(BATCH_COUNT.format(k=objects_word(k)))
        if self._preview_object == 0:
            # Preview's reason, on the count line.
            self._count_lbl.setText(
                self._count_lbl.text() + "\n" + BATCH_PREVIEW_NO_OBJECT)
        if hasattr(self, "_run_btn"):
            self._run_btn.setText(BATCH_RUN_BTN.format(n=objects_word(k)))
        self._refresh_run_enabled()

    def _refresh_run_enabled(self) -> None:
        # Run only when idle, count > 0, at least one recipe step (or a
        # chosen Refine recipe), nothing refusing it and no stopped chain
        # run still finishing.  Preview likewise, on one object.
        count = getattr(self, "_last_count", 0)
        has_steps = (self.chain_name() is not None if hasattr(self, "_chain_combo")
                     else False) or len(self._rows) > 0
        idle = self._state == "idle"
        orphans = int(getattr(self, "_orphans_running", 0) or 0) > 0
        can_run = (idle and count > 0 and has_steps and not orphans
                   and not getattr(self, "_run_reason", ""))
        if hasattr(self, "_run_btn"):
            self._run_btn.setEnabled(can_run)
        if hasattr(self, "_preview_btn") and idle:
            self._preview_btn.setEnabled(
                has_steps and getattr(self, "_preview_object", None) != 0
                and not orphans and not getattr(self, "_preview_reason", ""))

    # ---- button handlers --------------------------------------
    # These fire the injected callbacks.  The dialog does NOT change its
    # own logical state — the annotator calls set_state afterwards.  It
    # may optimistically disable the just-clicked button to prevent a
    # double-fire before the annotator responds.

    def _on_preview_clicked(self) -> None:
        self._preview_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._on_preview_one_cb()

    def _on_run_clicked(self) -> None:
        self._run_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)
        self._on_run_cb()

    def _on_cancel_clicked(self) -> None:
        self._cancel_btn.setEnabled(False)
        self._on_cancel_cb()

    def _on_reject_clicked(self) -> None:
        self._reject_btn.setEnabled(False)
        self._accept_btn.setEnabled(False)
        self._on_reject_cb()

    def _on_accept_clicked(self) -> None:
        self._accept_btn.setEnabled(False)
        self._reject_btn.setEnabled(False)
        self._on_accept_cb()

    # ---- public API (annotator-facing) ------------------------

    def keep_original_tier(self) -> bool:
        """True if refined labels should stay in their current QC tier
        (the user opted out of reclassifying to 'curated')."""
        return self._keep_tier_cb.isChecked()

    def target_spec(self) -> dict:
        """Return the current target selection.

        ``{"mode": "all"|"selected"|"plot"|"tier", "tiers": [str, ...]}``
        ``tiers`` is the list of checked tier names (only meaningful in
        ``"tier"`` mode, but always returned for completeness).
        """
        if self._radio_selected.isChecked():
            mode = "selected"
        elif self._radio_plot.isChecked():
            mode = "plot"
        elif self._radio_tier.isChecked():
            mode = "tier"
        else:
            mode = "all"
        tiers = [name for name, cb in self._tier_checks.items() if cb.isChecked()]
        return {"mode": mode, "tiers": tiers}

    def recipe(self) -> list | None:
        """Build the ordered recipe of validated :class:`RefineStep`.

        Returns ``None`` (and calls :meth:`set_status` with the error) if
        any row produces an invalid step, so the caller can abort.
        """
        steps: list[RefineStep] = []
        for i, row in enumerate(self._rows):
            step = row.to_step()
            try:
                step.validate()
            except ValueError as exc:
                # A fixed sentence on screen; the raw reason to the log.
                print(f"[BATCH] step {i + 1} invalid: {exc}")
                self.set_status(BATCH_STEP_INVALID.format(k=i + 1))
                return None
            steps.append(step)
        return steps

    # ---- recipe reload (history) ------------------------------

    def set_recipe(self, recipe) -> None:
        """Repopulate the recipe builder FROM an ordered recipe (the INVERSE
        of :meth:`recipe`).

        Accepts a ``list[RefineStep]`` OR a list of step dicts (the
        round-trippable JSON form) — dicts are coerced via
        ``RefineStep.from_dict``.  Clears the existing rows and adds one
        :class:`_RecipeRow` per step, setting each row's method + params.

        Reload-only: does NOT touch the target radios / tier checkboxes.
        A no-op (empty / invalid) recipe leaves the builder untouched
        and sets a status; only fires when ``idle``.
        """
        if self._state != "idle":
            return
        steps = self._coerce_recipe(recipe)
        if not steps:
            self.set_status(BATCH_LOAD_FAILED)
            return
        # Clear existing rows, then add one per step.
        for row in list(self._rows):
            self._remove_row(row)
        for step in steps:
            row = self._append_row(step.method)
            row.apply_step(step)
        self._renumber_rows()
        self._update_target_count()

    @staticmethod
    def _coerce_recipe(recipe) -> list:
        """Normalize ``recipe`` (list of RefineStep OR dicts) → list[RefineStep].

        Returns ``[]`` on anything unparseable (invalid / unknown method) so
        the caller degrades gracefully rather than raising into the UI.
        """
        if not recipe:
            return []
        out: list[RefineStep] = []
        for item in recipe:
            try:
                if isinstance(item, RefineStep):
                    item.validate()
                    out.append(item)
                else:
                    out.append(RefineStep.from_dict(item))
            except (ValueError, TypeError, AttributeError):
                return []
        return out

    def set_recipe_history(self, recent, saved) -> None:
        """(Re)populate the Recent (session) + Saved (persisted) dropdowns.

        ``recent`` is a list of recipe_json (list[dict]); ``saved`` is a
        ``{name: recipe_json}`` dict.  Each combo gets a placeholder at
        index 0 (so selecting it is a no-op) followed by one human-readable
        label per recipe.  The backing maps preserve the recipe_json so a
        selection can be reloaded via :meth:`set_recipe`.
        """
        # --- Recent ---
        self._recent_recipes = [list(r) for r in (recent or [])]
        self._recent_combo.blockSignals(True)
        self._recent_combo.clear()
        self._recent_combo.addItem("Recent recipes…")
        for recipe_json in self._recent_recipes:
            self._recent_combo.addItem(self._recipe_label(recipe_json))
        self._recent_combo.setCurrentIndex(0)
        self._recent_combo.blockSignals(False)

        # --- Saved ---
        self._saved_recipes = dict(saved or {})
        self._saved_combo.blockSignals(True)
        self._saved_combo.clear()
        self._saved_combo.addItem("Saved recipes…")
        for name in self._saved_recipes:
            self._saved_combo.addItem(str(name))
        self._saved_combo.setCurrentIndex(0)
        self._saved_combo.blockSignals(False)

    @staticmethod
    def _recipe_label(recipe_json) -> str:
        """Build a short human label for a recipe_json (method + key params).

        E.g. ``[{"method":"sdf","params":{"sigma_z":2.0,"sigma_xy":1.5}},
        {"method":"fill_holes","params":{"axis":"3D"}}]`` →
        ``"SDF σz=2.0 + Fill Holes"``.  Defensive against malformed entries.
        """
        parts = []
        for step in (recipe_json or []):
            try:
                method = step.get("method", "?")
                p = step.get("params", {}) or {}
            except AttributeError:
                parts.append("?")
                continue
            if method == "sdf":
                sz = p.get("sigma_z")
                parts.append(f"SDF σz={sz}" if sz is not None else "SDF")
            elif method == "dilate_erode":
                r = p.get("radius")
                axis = p.get("axis", "3D")
                if r is not None:
                    verb = "Dilate" if r > 0 else ("Erode" if r < 0 else "D/E")
                    parts.append(f"{verb} r={r} {axis}")
                else:
                    parts.append("Dilate/Erode")
            elif method == "fill_holes":
                parts.append("Fill Holes")
            else:
                parts.append(str(method))
        return " + ".join(parts) if parts else "(empty recipe)"

    def _on_recent_selected(self, idx: int) -> None:
        """Recent combo activated → reload the chosen recipe (idx 0 = no-op)."""
        if idx <= 0 or idx - 1 >= len(self._recent_recipes):
            return
        self.set_recipe(self._recent_recipes[idx - 1])
        # Reset to the placeholder so re-picking the same entry re-fires.
        self._recent_combo.blockSignals(True)
        self._recent_combo.setCurrentIndex(0)
        self._recent_combo.blockSignals(False)

    def _on_saved_selected(self, idx: int) -> None:
        """Saved combo activated → reload the chosen recipe (idx 0 = no-op)."""
        if idx <= 0:
            return
        names = list(self._saved_recipes.keys())
        if idx - 1 >= len(names):
            return
        self.set_recipe(self._saved_recipes[names[idx - 1]])
        self._saved_combo.blockSignals(True)
        self._saved_combo.setCurrentIndex(0)
        self._saved_combo.blockSignals(False)

    def _on_save_recipe_clicked(self) -> None:
        """Save-recipe button → validate the current recipe, prompt for a
        name, and fire the injected save callback.

        Refuses (status only) if the recipe is invalid/empty or no save
        callback is wired.  The callback persists + the annotator refreshes
        the Saved dropdown via ``set_recipe_history``.
        """
        if self._on_save_recipe_cb is None:
            self.set_status(BATCH_SAVE_NO_WORKSPACE)
            return
        # Validate before prompting — refuse to save an invalid/empty recipe.
        recipe = self.recipe()
        if not recipe:
            if recipe is None:
                # recipe() already set an invalid-step status.
                return
            self.set_status(BATCH_ADD_STEP)
            return
        name, ok = QInputDialog.getText(
            self, BATCH_SAVE_PROMPT_TITLE, BATCH_SAVE_PROMPT_LABEL
        )
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            self.set_status(BATCH_SAVE_NO_NAME)
            return
        # Warn (status) on overwrite — the store overwrites collisions.
        if name in self._saved_recipes:
            self.set_status(BATCH_SAVE_OVERWRITING.format(name=name))
        try:
            self._on_save_recipe_cb(name)
        except Exception as exc:  # noqa: BLE001 — never raise into the button
            print(f"[BATCH] save failed: {exc!r}")
            self.set_status(BATCH_SAVE_FAILED)

    def set_state(self, state: str) -> None:
        """Toggle sections/buttons for ``"idle"`` / ``"running"`` / ``"staged"``.

        idle    — target + recipe editable; [Preview][Run] shown;
                  progress + accept/reject + cancel hidden.
        running — target + recipe locked; progress + [Cancel] shown;
                  preview/run + accept/reject hidden.
        staged  — target + recipe locked; summary + [Reject][Accept] shown;
                  preview/run/progress/cancel hidden.
        """
        if state not in ("idle", "running", "staged"):
            raise ValueError(
                f"Unknown state {state!r}; expected idle/running/staged."
            )
        self._state = state
        editable = state == "idle"

        # Sections.
        self._target_section.setEnabled(editable)
        self._recipe_section.setEnabled(editable)
        for row in self._rows:
            row.set_editable(editable)
        if editable:
            self._renumber_rows()  # restore reorder-button enable logic
            self._tier_row.setEnabled(self._radio_tier.isChecked())
            self._apply_chain_mode()

        # Button + progress visibility.
        is_idle = state == "idle"
        is_running = state == "running"
        is_staged = state == "staged"

        self._preview_btn.setVisible(is_idle)
        self._run_btn.setVisible(is_idle)
        self._cancel_btn.setVisible(is_running)
        self._progress.setVisible(is_running)
        self._reject_btn.setVisible(is_staged)
        self._accept_btn.setVisible(is_staged)
        self._summary_lbl.setVisible(is_staged)

        # Re-enable freshly shown buttons (clear optimistic disables).
        if is_idle:
            if hasattr(self, "_count_lbl"):
                # What the poll skipped while busy (the preview object).
                self._render_count_line()
            self._refresh_run_enabled()
            self._status_lbl.setText("")
            self._summary_lbl.setText("")
            # Leave any marquee (set_busy) for a determinate, empty bar.
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
        elif is_running:
            self._cancel_btn.setEnabled(True)
        elif is_staged:
            self._reject_btn.setEnabled(True)
            self._accept_btn.setEnabled(True)

    def set_progress(self, done: int, total: int) -> None:
        """Update the progress bar.  ``total <= 0`` resets to 0%."""
        if total <= 0:
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            return
        self._progress.setRange(0, int(total))
        self._progress.setValue(int(done))

    def set_busy(self) -> None:
        """Show the bar as a marquee (Qt's indeterminate ``setRange(0, 0)``)
        for the stages with no count: preparing and writing.
        ``set_progress`` makes it determinate again."""
        self._progress.setRange(0, 0)

    def set_status(self, text: str) -> None:
        """Set the status line (e.g. "Refining object 37 of 142")."""
        self._status_lbl.setText(str(text))

    def set_preview_summary(
        self,
        n_objects: int,
        n_changed: int,
        *,
        emptied=(),
        absent=(),
        stopped=None,
    ) -> None:
        """Set the staged-state summary: ``BATCH_SUMMARY``, then one line
        per non-empty list -- ``stopped`` (``{id: (step, method word)}``,
        one line per distinct step: they keep the steps before it),
        ``emptied`` (left exactly as they were, a step would have emptied
        them) and ``absent`` (no longer in the volume) -- at most 8 ids per
        line, then "…"."""
        lines = [BATCH_SUMMARY.format(n=objects_word(n_objects),
                                      v=int(n_changed))]
        lines += summary_lines(emptied=emptied, absent=absent,
                               stopped=stopped)
        self._summary_lbl.setText("\n".join(lines))

    def refresh_count(self, *, force: bool | None = None) -> None:
        """Re-query the live target count.

        Called by the annotator when the navigator selection or the QC
        plot's highlighted set changes while the dialog is open, so
        "Selected" / "Plot selection" reflect the latest selection without
        the user having to re-toggle the radio.  Only those two depend on a
        live selection, so All / By-tier are skipped.
        """
        if self._state == "idle" and (
            force
            or self._radio_selected.isChecked()
            or self._radio_plot.isChecked()
        ):
            self._update_target_count()

    def set_target_counts(
        self,
        *,
        all: int | None = None,  # noqa: A002 -- the radio's own name
        selected: int | None = None,
        plot: int | None = None,
        plot_available: bool | None = None,
        plot_other_patch: bool | None = None,
    ) -> None:
        """Rewrite the target radios' texts with fresh counts.

        Additive: a keyword that is not passed leaves its radio alone.
        ``plot_available`` says whether this patch has a QC plot at all;
        it picks the count line's reason (``BATCH_PLOT_NONE`` vs
        ``BATCH_PLOT_EMPTY``) when the plot radio is checked at 0.
        """
        if all is not None:
            self._all_count = int(all)
            self._radio_all.setText(BATCH_TARGET_ALL.format(n=self._all_count))
        if selected is not None:
            self._selected_count = int(selected)
            self._radio_selected.setText(
                self._selected_template.format(n=self._selected_count)
            )
        if plot is not None:
            self._plot_count = int(plot)
            self._radio_plot.setText(
                BATCH_TARGET_PLOT.format(n=self._plot_count)
            )
        if plot_other_patch is not None:
            self._plot_other_patch = bool(plot_other_patch)
        if plot_available is not None or plot_other_patch is not None:
            if plot_available is not None:
                self._plot_available = bool(plot_available)
            if self._state == "idle" and self._radio_plot.isChecked():
                self._update_target_count()

    # ---- the Refine-tab recipe ------------

    def set_chain_recipes(self, store, *, problem: str | None = None,
                          no_workspace: bool | None = None) -> None:
        """(Re)fill the Refine-tab recipe combo from the Refine recipe store.

        ``store`` is ``{name: stored json}`` (the Refine tab's file, read
        through the controller).  An unreadable recipe is a DISABLED item
        whose text is the reason; a ``problem`` reading the file is one
        disabled item; ``no_workspace`` disables the combo and puts the
        reason on the reason line.  The current choice is kept when its
        name is still there and readable.
        """
        from packages.epicycle_annotator import refine_recipe as rr

        keep = self.chain_name()
        combo = self._chain_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(BATCH_CHAIN_NONE)
        self._chain_items = [(None, None)]
        model = combo.model()
        self._no_workspace = bool(no_workspace)
        if no_workspace:
            combo.setEnabled(False)
        else:
            combo.setEnabled(True)
            if problem:
                combo.addItem(rr.FILE_UNREADABLE)
                self._chain_items.append((None, None))
                model.item(combo.count() - 1).setEnabled(False)
            else:
                for name, value in dict(store or {}).items():
                    rec, why = rr.recipe_from_json(value)
                    if rec is None:
                        # A fixed item text; the reader's reason (which may
                        # carry exception text) goes to the console.
                        print(f"[BATCH] Refine recipe {name!r} cannot be "
                              f"read: {why}")
                        combo.addItem(BATCH_CHAIN_UNREADABLE.format(name=name))
                        self._chain_items.append((str(name), None))
                        model.item(combo.count() - 1).setEnabled(False)
                        continue
                    combo.addItem("{} — {}".format(
                        name, rr.steps_summary(rec.steps)))
                    self._chain_items.append((str(name), rec))
        idx = 0
        if keep is not None:
            for i, (name, rec) in enumerate(self._chain_items):
                if name == keep and rec is not None:
                    idx = i
                    break
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)
        self._apply_chain_mode()
        self._refresh_reason_line()

    def select_chain(self, name) -> bool:
        """Choose the readable recipe called ``name``; False if absent."""
        for i, (n, rec) in enumerate(self._chain_items):
            if n == name and rec is not None:
                self._chain_combo.setCurrentIndex(i)
                return True
        return False

    def chain_name(self):
        """The chosen Refine recipe's name, or ``None`` (build steps here)."""
        idx = self._chain_combo.currentIndex()
        if idx <= 0 or idx >= len(self._chain_items):
            return None
        name, rec = self._chain_items[idx]
        return name if rec is not None else None

    def chain_recipe(self):
        """The chosen recipe (``refine_recipe.Recipe``), or ``None``."""
        idx = self._chain_combo.currentIndex()
        if idx <= 0 or idx >= len(self._chain_items):
            return None
        return self._chain_items[idx][1]

    def _on_chain_selected(self, _idx=None) -> None:
        self._apply_chain_mode()
        self._update_target_count()

    def _apply_chain_mode(self) -> None:
        """Show the recipe's rows, or the builder -- never both."""
        from packages.epicycle_annotator import refine_recipe as rr

        rec = self.chain_recipe()
        chain = rec is not None
        self._scroll.setVisible(not chain)
        self._add_btn.setVisible(not chain)
        self._preset_btn.setVisible(not chain)
        self._rows_off_lbl.setVisible(chain)
        self._chain_summary_lbl.setVisible(chain)
        idle = self._state == "idle"
        self._recent_combo.setEnabled(idle and not chain)
        self._saved_combo.setEnabled(idle and not chain)
        self._save_recipe_btn.setEnabled(
            idle and not chain and self._can_save_recipe
            and self._on_save_recipe_cb is not None)
        needs = bool(chain) and any(
            rr.step_needs_channel(st.method, st.params) for st in rec.steps)
        self._chain_needs_channel = needs
        self._chain_channel_row.setVisible(needs)
        self._chain_summary_lbl.setText(
            "\n".join(self.chain_step_lines(rec)) if chain else "")

    @staticmethod
    def chain_step_lines(rec) -> list:
        """One line per step in the Refine page's own words: the method,
        and for a wall-building step the wall type's item text and the
        ticked preparation boxes' text.  No engine token reaches the
        screen."""
        from packages.epicycle_annotator import refine_recipe as rr
        from napari_epicycle.annotator.advanced_features_widget import (
            PREP_TICK_LABELS,
            REFINE_CONTOUR_MODE_LABELS,
            REFINE_WALL_KINDS,
            REFINE_WALLS_FROM_LABELS,
        )

        out = []
        for k, st in enumerate(getattr(rec, "steps", ()) or (), start=1):
            p = dict(st.params or {})
            word = rr.METHOD_WORD.get(st.method, st.method)
            parts = []
            builds_walls = st.method == "settle" or (
                st.method == "contour" and p.get("mode") == "snap")
            if st.method == "contour":
                mi = rr.CONTOUR_MODES.index(p.get("mode", "snap"))
                word = "{} ({})".format(word, REFINE_CONTOUR_MODE_LABELS[mi])
            if builds_walls:
                kind = p.get("wall_kind")
                if kind in REFINE_WALL_KINDS:
                    parts.append("walls: {}".format(
                        REFINE_WALLS_FROM_LABELS[REFINE_WALL_KINDS.index(kind)]))
            if st.method in ("settle", "contour"):
                ticks = dict(p.get("prep_ticks") or {})
                ticked = [PREP_TICK_LABELS[t] for t in rr.PREP_TICKS
                          if ticks.get(t) and t in PREP_TICK_LABELS]
                if ticked:
                    parts.append("preparation: {}".format(", ".join(ticked)))
            line = "{}. {}".format(k, word)
            if parts:
                line += " — " + "; ".join(parts)
            out.append(line)
        return out

    # ---- the channel the recipe reads (blank stays blank) -------------

    _CHANNEL_CHOICE_PROP = "_epicycle_channel_choice"

    def set_chain_channels(self, names, *, preferred: str | None = None) -> None:
        """Fill the channel combo from the viewer's image layers.

        The last name actually CHOSEN is remembered on the combo (the
        ``_epicycle_channel_choice`` property, as every channel combo); a
        refill restores it or selects NOTHING, so a blank combo stays blank
        across refreshes.  On a first fill ``preferred`` -- the Refine page's
        channel -- is selected when listed, else nothing.
        """
        combo = self._chain_channel_combo
        prev = ""
        try:
            prev = str(combo.property(self._CHANNEL_CHOICE_PROP) or "")
        except Exception:
            prev = ""
        if not prev and preferred:
            prev = str(preferred)
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([str(n) for n in (names or [])])
        idx = combo.findText(prev) if prev else -1
        combo.setCurrentIndex(idx)
        if prev:
            combo.setProperty(self._CHANNEL_CHOICE_PROP, prev)
        combo.blockSignals(False)
        self._fire_chain_changed()

    def _on_chain_channel_picked(self, _idx=None) -> None:
        combo = self._chain_channel_combo
        if combo.currentIndex() >= 0 and combo.currentText():
            combo.setProperty(self._CHANNEL_CHOICE_PROP, combo.currentText())
        self._fire_chain_changed()

    def chain_channel(self) -> str:
        """The chosen channel's layer name, or ``""`` -- never a fallback."""
        combo = self._chain_channel_combo
        if combo.currentIndex() < 0:
            return ""
        return str(combo.currentText() or "")

    def _fire_chain_changed(self) -> None:
        cb = getattr(self, "_on_chain_changed_cb", None)
        if cb is None or not hasattr(self, "_chain_combo"):
            return
        try:
            cb()
        except Exception as exc:  # noqa: BLE001 -- never raise into a combo
            print(f"[BATCH] chain re-check failed: {exc!r}")

    # ---- why Run / Preview are refused --------------------------------

    def set_chain_block(self, *, run: str | None = None,
                        preview: str | None = None) -> None:
        """The batch blocker's sentences for Run and for Preview.

        Both refused by the same sentence -> the line names no button;
        only Run refused -> ``"Run: "`` + its sentence; only Preview ->
        ``"Preview: "``.  The refused buttons are disabled; the line is
        their reason.
        """
        self._run_reason = str(run or "")
        self._preview_reason = str(preview or "")
        self._refresh_reason_line()
        self._refresh_run_enabled()

    def set_orphans_running(self, *, n: int | None = None) -> None:
        """``n`` objects from a stopped chain run still computing: Run and
        Preview wait, and the reason line says so."""
        self._orphans_running = int(n or 0)
        self._refresh_reason_line()
        self._refresh_run_enabled()

    def set_preview_object(self, *, obj_id: int | None = None) -> None:
        """The object "Preview" would run on -- the viewer's selection."""
        self._preview_object = int(obj_id or 0) if (obj_id or 0) > 0 else 0
        if self._preview_object:
            self._preview_btn.setText(
                BATCH_PREVIEW_BTN.format(id=self._preview_object))
        else:
            self._preview_btn.setText(BATCH_PREVIEW_BTN_NONE)
        self._update_target_count()

    def reason_text(self) -> str:
        return self._chain_reason_lbl.text()

    def _refresh_reason_line(self) -> None:
        from packages.epicycle_annotator.refine_recipe import NO_WORKSPACE

        run = getattr(self, "_run_reason", "")
        prev = getattr(self, "_preview_reason", "")
        if int(getattr(self, "_orphans_running", 0) or 0) > 0:
            text = BATCH_ORPHANS_RUNNING.format(
                n=objects_word(self._orphans_running))
        elif run and prev and run == prev:
            text = run
        elif run and prev:
            text = BATCH_REASON_RUN_ONLY + run + "\n" + \
                BATCH_REASON_PREVIEW_ONLY + prev
        elif run:
            text = BATCH_REASON_RUN_ONLY + run
        elif prev:
            text = BATCH_REASON_PREVIEW_ONLY + prev
        elif getattr(self, "_no_workspace", False):
            text = NO_WORKSPACE
        else:
            text = ""
        self._chain_reason_lbl.setText(text)
        self._chain_reason_lbl.setVisible(bool(text))

    def mark_clean_close(self) -> None:
        """Flag the next ``close()`` as programmatic so ``on_close`` does
        NOT fire (used by the annotator when it tears the dialog down)."""
        self._clean_close = True

    # ---- lifecycle --------------------------------------------

    def closeEvent(self, event) -> None:
        """Fire ``on_close`` for non-clean closes (X button / parent
        destroy / dock close).  ``mark_clean_close()`` suppresses this so
        a programmatic teardown does not double-fire.

        Mirrors ``MergeDialog.closeEvent`` — defends against the parent
        annotator being mid-teardown (``_is_closing``) so we don't touch
        partially-destroyed state.
        """
        if not self._clean_close:
            parent = self.parent()
            if getattr(parent, "_is_closing", False):
                self._clean_close = True
            else:
                try:
                    self._on_close_cb()
                except Exception:
                    pass
                self._clean_close = True  # idempotent — no double-fire
        super().closeEvent(event)

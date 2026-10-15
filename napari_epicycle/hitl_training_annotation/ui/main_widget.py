from qtpy.QtCore import QEvent, QTimer, Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QTabWidget,
    QToolBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from . import _step2_curation as _C
from . import _step2_text as _T
from . import _step2_lists as L
from . import _step2_training as _TR
from ._step2_lists import (
    PatchLists, PatchPictures, RunSummaryCard, build_curation_patch_list,
    build_session_list, LOCK_PROPERTY as _LIST_LOCK_PROPERTY,
)
from .loss_plot_widget import LossPlotWidget
from ..controller.training_annotation_controller import TrainingAnnotationController
from ...ui_shell.theme import (
    Accordion, COLLAPSIBLE_DURATION, STEP3_PLOT_PALETTE, COLLAPSIBLE_STYLE, ComingSoonOverlay,
    DIM_LABEL_STYLE, FLOW_HINT_STYLE, FLOW_TITLE_STYLE, FLOW_SUBTITLE_STYLE,
    INPUT_STYLE, LIST_STYLE, LOG_OUTPUT_STYLE, MONO_FONT, MONO_LABEL_STYLE,
    PANEL_CHECKBOX_STYLE, PANEL_COMBO_POPUP_STYLE, PANEL_GOLD,
    PANEL_HELP_STYLE, PANEL_LEFT_INNER_MARGIN, PANEL_OK_COLOR,
    PANEL_RADIO_STYLE, PANEL_ROW_LABEL_STYLE, PANEL_WARN_COLOR,
    PANEL_ZERO_COLOR,
    PRIMARY_BTN_STYLE, PROGRESS_BAR_STYLE, SCROLLBAR_STYLE,
    SECONDARY_BTN_STYLE, SECTION_HEADER_STYLE, SEPARATOR_LINE_STYLE,
    PANEL_SPLIT_HANDLE, SPLITTER_STYLE, STATUS_LABEL_STYLE, STATUS_READOUT_STYLE,
    TAB_WIDGET_STYLE, apply_button_icon, apply_inner_group_style,
    help_button,
)
from ...ui_shell._recon_settings_widgets import (
    ENGINE_NAME_STYLE, FORM_FIELD_W, LockLine, collapsible as _disclosure)
from ...ui_shell._stack_row_delegate import MODE_NARROW, list_cap_px, thumb_px

# ---------------------------------------------------------------------------
# Phase 8e-B (2026-05-24) — Step 2 help system
# ---------------------------------------------------------------------------
# `?` button on each section header opens a biologist-friendly markdown
# guide.  Mirrors the Step 3 pattern (themed_help_dialog + help_button +
# guides under docs/guides/).  Guides live at docs/guides/step2_*.md.

from pathlib import Path as _Path
from napari_epicycle.ui_shell._guides import guides_dir as _guides_dir
_STEP2_GUIDES_DIR = _guides_dir()
_STEP2_ACCENT = PANEL_GOLD

#: the left column's width from which the filter chips carry their words
#: ("Predicted 62"); narrower, a chip is its icon and count (the word is the
#: tooltip).  Keyed on the COLUMN, not on the dock's height (polish D-C).
CHIP_WORDS_MIN_W = 330

_COMBO_POPUP_STYLE = PANEL_COMBO_POPUP_STYLE


def _section_header_with_help(title: str, guide_filename: str = "",
                               parent=None) -> QHBoxLayout:
    """Build a section header row with optional `?` help button.

    Returns a QHBoxLayout to addLayout(...) into the parent column.
    Empty ``guide_filename`` skips the help button (header only).
    """
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    label = QLabel(title, parent=parent)
    label.setStyleSheet(SECTION_HEADER_STYLE)
    row.addWidget(label)
    row.addStretch(1)
    if guide_filename:
        row.addWidget(help_button(
            parent,
            guide_path=str(_STEP2_GUIDES_DIR / guide_filename),
            title=title,
            accent=_STEP2_ACCENT,
            tooltip=f"Help: {title}",
        ))
    return row


class TrainingAnnotationHITL2D(QWidget):
    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self.setLayout(self._create_main_layout())
        self._apply_theme()
        self.controller = TrainingAnnotationController(widget=self, viewer=viewer)

    def _apply_theme(self) -> None:
        apply_inner_group_style(self)
        # Primary action buttons.
        # (Curation's foot buttons are styled by ``_step2_curation``'s
        # ``PinnedFoot`` -- brick P; ``launch_multiview_button`` is its one
        # primary.)
        # (Training's foot buttons are styled by ``_step2_training``: Train is
        # the one solid-gold primary, Cancel the secondary -- brick T.)
        self.train_cancel_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        # Secondary buttons.
        for attr in ("set_workdir_browse_button", "train_patch_browse_button",
                     "train_patch_refresh_button",
                     "imported_model_browse_button", "starting_checkpoint_browse",
                     "view_results_button",
                     "apply_inference_params_button",
                     "save_and_set_config_button", "load_config_button"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setStyleSheet(SECONDARY_BTN_STYLE)
        # Key action icons.
        # Filter button icons — match simplified status palette.
        for _btn, _key in ((self.filter_predicted_btn, "predicted"),
                           (self.filter_curating_btn, "curating"),
                           (self.filter_accepted_btn, "accepted"),
                           (self.filter_staged_btn, "staged")):
            apply_button_icon(_btn, _T.STATUS_ICONS[_key],
                              color=_T.STATUS_COLOURS[_key])
        self.setStyleSheet(
            self.styleSheet() + INPUT_STYLE + LIST_STYLE + SCROLLBAR_STYLE
            + COLLAPSIBLE_STYLE + TAB_WIDGET_STYLE
            # Step 2 look (brick L): Step 3's gold ticks and radios on the
            # ROOT, so all four sections get them without a per-widget call
            + PANEL_CHECKBOX_STYLE + PANEL_RADIO_STYLE
            # S2d: the root INPUT_STYLE combo gets Step 3's dark popup
            + _COMBO_POPUP_STYLE
        )
        self.training_progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self._apply_curation_room()

    def _create_main_layout(self):
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        root.addWidget(self._create_flow_overview_group())
        root.addWidget(self._create_control_rail(), 1)

        self._parent_hidden_stubs()
        return root

    def _parent_hidden_stubs(self) -> None:
        """Hidden compat stubs the builders made without a parent: give each
        ``self`` as its parent (kept hidden) so none is a taskbar orphan."""
        for _name, _w in list(vars(self).items()):
            if (isinstance(_w, QWidget) and _w.parentWidget() is None
                    and _w is not self):
                _w.setParent(self)
                _w.setVisible(False)

    def _create_flow_overview_group(self):
        hint = QWidget()
        hint.setStyleSheet(FLOW_HINT_STYLE)
        layout = QVBoxLayout(hint)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel(_T.FLOW_HINT)
        title.setWordWrap(True)
        title.setStyleSheet(FLOW_TITLE_STYLE)
        routing = QLabel("Routing: XY \u2192 lateral, XZ/YZ \u2192 pooled axial.")
        routing.setWordWrap(True)
        routing.setStyleSheet(FLOW_SUBTITLE_STYLE)

        layout.addWidget(title)
        layout.addWidget(routing)
        return hint

    def _create_control_rail(self):
        accordion = Accordion()
        accordion.addItem(self._create_session_panel(), "Session",
                          icon_name="mdi6.folder-cog-outline")
        accordion.addItem(self._create_curation_panel(), "Curation",
                          icon_name="mdi6.pencil-outline")
        accordion.addItem(self._create_training_panel(), "Training",
                          icon_name="mdi6.brain")
        # Release-hardening 2026-09-07 (bucket-switch item #8): Benchmarking
        # is not wired up for v1 -- it was a "coming soon" veil over a
        # permanently disabled button, which reads as a broken feature.  The
        # panel is still BUILT (so none of its child widgets become
        # parentless taskbar orphans) and simply hidden; drop the
        # setVisible(False) below to bring it back.
        accordion.addItem(self._create_benchmarking_panel(), "Benchmarking",
                          icon_name="mdi6.speedometer")
        self._benchmarking_section_index = accordion.count() - 1
        accordion.addItem(self._create_diagnostics_panel(), "Diagnostics",
                          icon_name="mdi6.stethoscope")
        try:
            accordion._sections[self._benchmarking_section_index].setVisible(False)
        except Exception:
            pass
        accordion.setCurrentIndex(1)
        # Step 2 look (brick L): the two patch lists share ONE tally.
        self._patch_pictures = PatchPictures()
        self._patch_lists = PatchLists(
            self.session_patch_list, self.train_patch_list,
            self._patch_pictures, on_summary=self._set_session_summary)
        self._patch_pictures._on_meta = self._patch_lists.refill_last
        # Step 2 look (brick A): the Slices list and Training's staged list
        # share ONE picture manager (separate state objects inside it)
        self.slice_pictures = L.SlicePictures(on_meta=self._on_slice_meta)
        self.slice_pictures.lists = [self.segment_slice_data_list,
                                     self.data_to_add_list]
        self._slice_meta_timer = QTimer(self)
        self._slice_meta_timer.setSingleShot(True)
        self._slice_meta_timer.setInterval(L.META_COALESCE_MS)
        self._slice_meta_timer.timeout.connect(self._repaint_staged_rows)
        self._slice_workspace = None
        self._last_staged_payload = None
        return accordion

    def _set_session_summary(self, text: str) -> None:
        lbl = getattr(self, "session_patches_summary", None)
        if lbl is not None:
            lbl.setText(text)

    def _create_session_panel(self):
        """Session: the lock line,
        Workspace, Patch folder, the Patches list (Step 3's wide rows,
        read-only) and the Run summary card as the last block.  The view
        radios moved to Curation (``_create_curation_config_column``); the
        eleven "Round Status" rows are gone -- the controller still writes
        its ``status_*`` labels, which are kept (hidden, or inside the Run
        summary card)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # The lock line is the FIRST row (driven by on_training_locked).
        self.session_lock_line = LockLine(
            panel, object_name="step2SessionLockLine")
        layout.addWidget(self.session_lock_line)

        # --- Workspace ---
        layout.addLayout(_section_header_with_help(
            _T.SESSION_WORKSPACE_HEADER, _T.GUIDE_SESSION, parent=panel))
        layout.addWidget(self._dim_desc(_T.SESSION_WORKSPACE_DESC, panel))
        self.set_workdir_path = QLineEdit(panel)
        self.set_workdir_path.setPlaceholderText(
            _T.SESSION_WORKSPACE_PLACEHOLDER)
        self.set_workdir_browse_button = QPushButton("Browse", panel)
        workdir_row = QHBoxLayout()
        workdir_row.setSpacing(6)
        workdir_row.addWidget(self.set_workdir_path, 1)
        workdir_row.addWidget(self.set_workdir_browse_button, 0)
        layout.addLayout(workdir_row)

        _sep = QLabel("", panel)
        _sep.setFixedHeight(1)
        _sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        layout.addWidget(_sep)

        # --- Patch folder ---
        layout.addLayout(_section_header_with_help(
            _T.SESSION_PATCH_FOLDER_HEADER, _T.GUIDE_SESSION, parent=panel))
        layout.addWidget(self._dim_desc(_T.SESSION_PATCH_FOLDER_DESC, panel))
        self.load_training_path_patch = QLineEdit(panel)
        self.load_training_path_patch.setPlaceholderText(
            _T.SESSION_PATCH_FOLDER_PLACEHOLDER)
        self.train_patch_browse_button = QPushButton("Browse", panel)
        # 2026-05-26 audit-J MEDIUM fix: Refresh next to Browse -- re-scan
        # after a Step 1 re-extraction without re-Browsing.  Idempotent.
        self.train_patch_refresh_button = QPushButton("Refresh", panel)
        self.train_patch_refresh_button.setToolTip(
            "Re-scan the patch directory.  Use this after running Step 1 "
            "re-extraction to pick up newly-extracted patches without "
            "re-Browsing.  Idempotent — safe to click any time."
        )
        patch_row = QHBoxLayout()
        patch_row.setSpacing(6)
        patch_row.addWidget(self.load_training_path_patch, 1)
        patch_row.addWidget(self.train_patch_browse_button, 0)
        patch_row.addWidget(self.train_patch_refresh_button, 0)
        layout.addLayout(patch_row)
        # B3: when no folder has been picked the
        # box shows the patches Step 1 extracted; this faint line says so
        # (the controller shows / hides it; the count is UI-side).
        self.patch_dir_qualifier = QLabel(
            _T.SESSION_PATCH_FOLDER_IDLE, panel)
        self.patch_dir_qualifier.setWordWrap(True)
        self.patch_dir_qualifier.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(self.patch_dir_qualifier)

        _sep = QLabel("", panel)
        _sep.setFixedHeight(1)
        _sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        layout.addWidget(_sep)

        # --- Patches (Step 3's wide rows, read-only) ---
        layout.addLayout(_section_header_with_help(
            _T.SESSION_PATCHES_HEADER, _T.GUIDE_SESSION, parent=panel))
        layout.addWidget(self._dim_desc(_T.SESSION_PATCHES_DESC, panel))
        self.session_patches_summary = QLabel("", panel)
        self.session_patches_summary.setWordWrap(True)
        self.session_patches_summary.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(self.session_patches_summary)
        self.session_patch_list = build_session_list(panel)
        layout.addWidget(self.session_patch_list)

        # --- Run summary (the last block) ---
        layout.addLayout(_section_header_with_help(
            _T.RUN_SUMMARY_HEADER, _T.GUIDE_SESSION, parent=panel))
        layout.addWidget(self._dim_desc(_T.RUN_SUMMARY_DESC, panel))
        # The controller's own labels (it writes them in
        # ``_refresh_round_status``): kept; four sit inside the card.
        self.status_run_value = QLabel("default")
        self.status_total_pairs_value = QLabel("0")
        self.status_pairs_lateral_value = QLabel("0")
        self.status_pairs_axial_value = QLabel("0")
        self.run_summary_card = RunSummaryCard(
            panel, run_value=self.status_run_value,
            total_value=self.status_total_pairs_value,
            lateral_pairs=self.status_pairs_lateral_value,
            axial_pairs=self.status_pairs_axial_value)
        self.round_status_group = self.run_summary_card
        layout.addWidget(self.run_summary_card)
        self.status_run_value.setToolTip(
            "Active training run.  Step 2 supports multiple runs per "
            "workspace (e.g. one for nuclei, one for membrane).  Switch "
            "between them with the run picker at the top of Curation."
        )

        # The labels the controller writes that no longer show (hidden, but
        # present and parented): CTRL _refresh_round_status.
        for _name, _init in (
                ("status_workspace_value", "-"),
                ("status_round_value", "0"),
                ("status_view_value", "Lateral (XY)"),
                ("status_segmented_value", "0"),
                ("status_accepted_value", "0"),
                ("status_staged_lateral_value", "0"),
                ("status_staged_axial_value", "0"),
                ("status_staged_value", "0"),
                ("status_last_checkpoint_value", "None"),
                ("status_gpu_value", "Checking...")):
            _lbl = QLabel(_init, panel)
            _lbl.setVisible(False)
            setattr(self, _name, _lbl)

        # Detect GPU availability at init (kept for the hidden label).
        self._detect_gpu()

        layout.addStretch(1)
        return panel

    @staticmethod
    def _dim_desc(text: str, parent) -> QLabel:
        """A faint wrapped description line under a header."""
        lbl = QLabel(text, parent)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(FLOW_SUBTITLE_STYLE)
        return lbl

    # ------------------------------------------------------------------
    # Hooks the controller calls (getattr-guarded there
    # section 4.1 (f)).  Every one is safe to call with nothing built.
    # ------------------------------------------------------------------
    def on_session_patches(self, payload) -> None:
        """Hook 4 (end of the controller's ``_refresh_patch_list_badges``):
        fill Session's list and Curation's Patches list from ONE tally, the
        Run summary's round cells and the folder line's count."""
        lists = getattr(self, "_patch_lists", None)
        if lists is None:
            return
        lists.fill(payload)
        card = getattr(self, "run_summary_card", None)
        if card is not None:
            card.update_from(payload)
        n = lists.session.count()
        if payload.get("stack_dir") and n:
            self.patch_dir_qualifier.setText(
                _T.SESSION_PATCH_FOLDER_QUALIFIER.format(n=n))

    def on_shutdown(self) -> None:
        """Hook 2 (the controller's ``shutdown()``): stop the thumbnail
        batches (bounded wait; a thread still running stays parked)."""
        for name in ("_patch_pictures", "slice_pictures"):
            pics = getattr(self, name, None)
            if pics is not None:
                try:
                    pics.stop()
                except Exception as exc:      # noqa: BLE001 -- never break shutdown
                    print(f"[STEP2-THUMB] stop failed: {exc}", flush=True)

    def on_slice_rows(self, payload) -> None:
        """Hook (``_refresh_sample_queue`` / ``_sync_data_to_add_list``): paint the Slices
        rows (or, ``staged``, Training's staged rows) from the records, set the five chips' counts and the
        Slices header, and ask for the pictures the cache lacks.  Never adds,
        removes or retexts a row."""
        pics = getattr(self, "slice_pictures", None)
        if pics is None:
            return
        records = payload.get("records") or {}
        workspace = str(payload.get("workspace") or "")
        if self._slice_workspace is not None \
                and workspace != self._slice_workspace:
            pics.forget_workspace()        # ids are only unique per workspace
        self._slice_workspace = workspace
        staged = bool(payload.get("staged"))
        if staged:
            self._last_staged_payload = payload
            L.apply_slice_records(self.data_to_add_list, records, staged=True,
                                  metas=pics.meta)
        else:
            L.apply_slice_records(self.segment_slice_data_list, records)
            self._refresh_slice_chips(records, payload)
            self._refresh_slices_header(payload)
        pics.refresh(workspace, records, payload.get("ids") or (),
                     stack_paths=payload.get("stack_paths"),
                     loaded_patch=str(payload.get("loaded_patch") or ""))

    def on_slice_written(self, sample_id) -> None:
        """Hook (the controller's debounced mask write): re-draw that
        slice's picture 1.5 s after the last write."""
        pics = getattr(self, "slice_pictures", None)
        if pics is not None:
            pics.note_mask_written(str(sample_id))

    def on_workspace_opened(self, workspace, sample_ids=()) -> None:
        """Hook (the controller's quiet workspace load): delete the slice
        pictures of samples no run has any more (a short daemon thread; the
        slices folder only)."""
        from .. import _step2_thumbs as TH
        pics = getattr(self, "slice_pictures", None)
        self._slice_sweep_thread = TH.sweep_slices_async(
            str(workspace or ""), list(sample_ids or ()),
            in_flight=getattr(pics, "inflight", None))

    def forget_patch(self, patch_name) -> None:
        """Forget a patch (file name or stem): its pictures leave memory
        and the two thumbs folders."""
        ctrl = getattr(self, "controller", None)
        ws = str(getattr(ctrl, "working_dir", "") or "")
        if not ws:
            return
        stem = str(patch_name).rsplit(".", 1)[0] if str(patch_name).lower(
        ).endswith((".tif", ".tiff")) else str(patch_name)
        for name, fn in (("_patch_pictures", "forget_patch"),
                         ("slice_pictures", "forget_patch")):
            pics = getattr(self, name, None)
            if pics is not None:
                arg = patch_name if name == "_patch_pictures" else stem
                getattr(pics, fn)(ws, arg)

    def _on_slice_meta(self) -> None:
        """A slice's meta arrived (worker thread -> main): coalesce, then
        repaint the staged rows (their label counts)."""
        timer = getattr(self, "_slice_meta_timer", None)
        if timer is not None:
            timer.start()

    def _repaint_staged_rows(self) -> None:
        payload = getattr(self, "_last_staged_payload", None)
        pics = getattr(self, "slice_pictures", None)
        if payload is None or pics is None:
            return
        L.apply_slice_records(self.data_to_add_list,
                              payload.get("records") or {}, staged=True,
                              metas=pics.meta)

    _CHIP_BUTTONS = (("all", "filter_all_btn"),
                     ("predicted", "filter_predicted_btn"),
                     ("curating", "filter_curating_btn"),
                     ("accepted", "filter_accepted_btn"),
                     ("staged", "filter_staged_btn"))

    def _apply_chip_mode(self, short: bool) -> None:
        """Narrow column: a chip with an icon shows the count only (its word is
        the tooltip); ``All`` has no icon and keeps its word.  Wide: ``Word n``.
        (The argument is "compact", from the column's width.)"""
        self._chips_compact = bool(short)
        texts = getattr(self, "_chip_texts", None)
        counts = getattr(self, "_chip_counts", None) or {}
        if not texts:
            return
        for key, name in self._CHIP_BUTTONS:
            btn = getattr(self, name, None)
            if btn is None:
                continue
            compact = self._chips_compact and key != "all"
            btn.setText(str(int(counts.get(key, 0))) if compact else texts[key])
            btn.setToolTip(texts[key] if compact else "")

    def _refresh_slice_chips(self, records, payload) -> None:
        """The five chips' ``Word n`` texts (and a dim look at zero)."""
        counts = L.filter_chip_counts(
            records, bucket=str(payload.get("bucket") or "lateral"),
            patch=str(payload.get("patch") or ""))
        texts = L.filter_chip_texts(counts)
        self._chip_texts, self._chip_counts = texts, counts
        # the stage line's "no slices" is about the records (patch + view),
        # not about a filter that hides them all
        self._slice_total = counts.get("all", 0)
        for key, name in (("all", "filter_all_btn"),
                          ("predicted", "filter_predicted_btn"),
                          ("curating", "filter_curating_btn"),
                          ("accepted", "filter_accepted_btn"),
                          ("staged", "filter_staged_btn")):
            btn = getattr(self, name, None)
            if btn is None:
                continue
            compact = getattr(self, "_chips_compact", False) and key != "all"
            btn.setText(str(int(counts.get(key, 0))) if compact else texts[key])
            btn.setToolTip(texts[key] if compact else "")
            zero = counts.get(key, 0) == 0
            if btn.property("zero") != zero:
                btn.setProperty("zero", zero)
                btn.style().unpolish(btn)
                btn.style().polish(btn)

    def _refresh_slices_header(self, payload) -> None:
        """``WholeScale_patch_4 . lat`` beside the Slices header (full text
        on hover; blank with no patch loaded)."""
        lbl = getattr(self, "slices_header_patch", None)
        if lbl is None:
            return
        patch = str(payload.get("loaded_patch") or "")
        bucket = str(payload.get("bucket") or "")
        word = _T.SLICES_VIEW_WORDS.get(bucket, bucket)
        text = _T.SLICES_HEADER_PATCH.format(patch=patch, view=word) \
            if patch else ""
        lbl.setText(text)
        lbl.setToolTip(text)

    def resizeEvent(self, event):  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._apply_curation_room()

    def _room_height(self) -> int:
        """The height the dock really offers: the enclosing shell scroll
        viewport's (this widget is max(viewport, minimum hint) tall inside
        it); this widget's own height when there is no scroll area."""
        vp = _C.enclosing_viewport_height(self)
        return vp if vp > 0 else self.height()

    def _watch_room_viewport(self) -> None:
        area = _C.enclosing_scroll_area(self)
        if area is not None and getattr(self, "_room_area", None) is not area:
            self._room_area = area
            area.viewport().installEventFilter(self)

    def showEvent(self, event):  # noqa: N802 (Qt API)
        super().showEvent(event)
        self._apply_curation_room()

    def _patch_list_min_h(self, short: bool) -> int:
        """The Patches list's floor: tall mode as built (a row a little
        cropped); a short dock shows ONE WHOLE row at the 56 px picture."""
        if short:
            return list_cap_px(MODE_NARROW, 1) - (
                thumb_px(MODE_NARROW) - L.SHORT_PATCH_THUMB_PX)
        return min(list_cap_px(MODE_NARROW, 1) + 8, 70)

    def _chips_compact_for_width(self) -> bool:
        col = getattr(self, "_chip_column", None)
        return col is not None and col.width() < CHIP_WORDS_MIN_W

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if (obj is getattr(self, "_chip_column", None)
                and event.type() == QEvent.Resize):
            compact = self._chips_compact_for_width()
            if compact != getattr(self, "_chips_compact", None):
                self._apply_chip_mode(compact)
        area = getattr(self, "_room_area", None)
        if (area is not None and obj is area.viewport()
                and event.type() == QEvent.Resize):
            self._apply_curation_room()
        return super().eventFilter(obj, event)

    def _apply_curation_room(self) -> None:
        """Below the tall-mode height the section descs are
        tooltips of their headers (the room is ~218 px under the run row)."""
        self._watch_room_viewport()
        short = self._room_height() < _C.TALL_MODE_MIN_PX
        if getattr(self, "_curation_short", None) == short:
            return
        self._curation_short = short
        pairs = list(getattr(self, "_left_descs", []))
        _C.set_short_mode(pairs, short)
        right = getattr(self, "curation_right_column", None)
        if right is not None:
            right.set_short_mode(short)
        # Q3: a short dock favours Slices (Patches at its one-row minimum,
        # the loaded patch is also named in the Slices header); a tall one
        # starts them even.  Only at the mode change, so a drag persists.
        vsplit = getattr(self, "curation_left_splitter", None)
        if vsplit is not None:
            vsplit.setSizes([1, 1000] if short else [1, 1])
        # the `?` rings clear the horizontal handle: a tall dock keeps 4 px
        # above and below it, a short one has no pixel to spare
        gap = 0 if short else 4
        pl = getattr(self, "_patch_pane_layout", None)
        sl = getattr(self, "_slice_pane_layout", None)
        if pl is not None:
            pl.setContentsMargins(0, 0, 0, gap)
        if sl is not None:
            sl.setContentsMargins(0, gap, 0, 0)
        # fix pass 4: Training's descriptions become header tooltips, the
        # Slices chips icon + count (the word in the tooltip)
        tp = getattr(self, "training_panel", None)
        if tp is not None and hasattr(tp, "set_short_mode"):
            tp.set_short_mode(short)
        # polish D-E: a short dock tightens the Curation panel's own chrome
        # (the rule under the run row, the gaps, the margins, the Slices
        # floor) so the lock line fits above the columns
        root = getattr(self, "_curation_root_layout", None)
        if root is not None:
            root.setSpacing(2 if short else 4)
            root.setContentsMargins(6, 1 if short else 2, 6,
                                    1 if short else 2)
        sep = getattr(self, "_curation_sep", None)
        if sep is not None:
            sep.setVisible(not short)
        slist = getattr(self, "segment_slice_data_list", None)
        if slist is not None:
            slist.setMinimumHeight(30 if short else 40)
        # polish D-D: Curation's Patches rows draw the 56 px picture and the
        # list's floor is one whole such row
        pl = getattr(self, "_patch_lists", None)
        if pl is not None:
            pl.set_short(short)
        plist = getattr(self, "train_patch_list", None)
        if plist is not None:
            plist.setMinimumHeight(self._patch_list_min_h(short))
        self._apply_chip_mode(self._chips_compact_for_width())

    def on_busy(self, text=None) -> None:
        """Hook 5 (``_begin_busy`` / ``_end_busy``): words on the Curation
        foot's bar and in its stage line."""
        self._curation_busy_text = str(text) if text else None
        foot = getattr(self, "curation_foot", None)
        if foot is not None:
            foot.set_busy(self._curation_busy_text)
        self.refresh_stage_line()

    def refresh_stage_line(self) -> None:
        """The Curation foot's one stage sentence; called
        from the ends of the controller's ``_refresh_sample_queue`` and
        ``_jump_to_selected_queue_item``, ``on_busy`` and the lock hook."""
        foot = getattr(self, "curation_foot", None)
        if foot is None:
            return
        ctrl = getattr(self, "controller", None)
        # the bucket the table and the lists use (``selected_view``), not the
        # radios: a patch load sets selected_view without touching them
        view = getattr(ctrl, "selected_view", None)
        if view:
            bucket = "lateral" if str(view).upper() == "XY" else "axial"
        else:
            switch = getattr(self, "curation_view_switch", None)
            bucket = switch.active_bucket() if switch is not None else "lateral"
        lst = getattr(self, "segment_slice_data_list", None)
        try:
            n_sel = len(lst.selectedItems()) if lst is not None else 0
            n_slices = lst.count() if lst is not None else None
            total = getattr(self, "_slice_total", None)
            if total is not None and n_slices == 0 and total > 0:
                n_slices = total      # every chip off: the slices exist
        except Exception:      # noqa: BLE001 -- a list mid-rebuild
            n_sel, n_slices = 0, None
        lock = getattr(self, "_curation_lock", None) or (False, None, "")
        self._apply_foot_gate(bool(lock[0]), n_slices)
        foot.set_stage(_C.compose_stage_line(
            locked=bool(lock[0]), round_n=lock[1],
            bucket=lock[2] or bucket,
            patch=str(getattr(ctrl, "loaded_patch_name", "") or ""),
            n_selected=n_sel, n_slices=n_slices,
            busy_text=getattr(self, "_curation_busy_text", None)))

    _FOOT_GATED = ("accept_annotation_button", "stage_current_view_button",
                   "reject_annotation_button", "unstage_selected_button")

    def _apply_foot_gate(self, locked: bool, n_slices) -> None:
        """S2c (a presentation gate; the controller's own gate is untouched):
        with NO slices in the view, Accept / Finalize / Discard / Revert are
        disabled; they light again when slices appear.  Under the training lock
        the controller's lock list owns them (this never touches them)."""
        if locked:
            self._foot_gated = False
            return
        empty = n_slices is not None and int(n_slices) == 0
        gated = bool(getattr(self, "_foot_gated", False))
        if empty == gated:
            return
        for name in self._FOOT_GATED:
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(not empty)
        self._foot_gated = empty

    def _on_view_radio_toggled(self, *_args) -> None:
        """The Run summary's pair count is for the active bucket: follow the
        View switch (a patch load follows through the Session hook)."""
        card = getattr(self, "run_summary_card", None)
        if card is not None:
            card.set_active_bucket(
                "axial" if self.get_view_axial.isChecked() else "lateral")

    def _refresh_inference_summary(self, *_args) -> None:
        """The grey line under the closed Inference disclosure."""
        block = getattr(self, "curation_model_block", None)
        if block is None:
            return

        def _word(combo):
            t = str(combo.currentText())
            return t.split(": ", 1)[1] if ": " in t else t

        ch2 = _word(self.inference_channel2_combo)
        chans = _word(self.inference_channel1_combo)
        if ch2.strip().lower() != "none":
            chans = f"{chans} / {ch2}"
        block.set_inference_summary(_C.compose_inference_summary(
            self.inference_preset_combo.currentText(), chans,
            self.inference_flow_thresh.value(),
            self.inference_cellprob_thresh.value(),
            self.inference_diameter_sb.value()))

    def on_training_locked(self, locked: bool, round_name: str = "",
                           bucket: str = "") -> None:
        """Hook 1 (driven by brick T's seam): Session's amber lock line, the
        read-only list dimmed, Browse / Refresh off -- and Curation's lock
        line, dimmed lists and ``Locked`` stage line."""
        n_round = _T.display_round(round_name)
        self._curation_lock = (bool(locked), n_round, bucket or "")
        cline = getattr(self, "curation_lock_line", None)
        if cline is not None:
            cline.setText(_T.CUR_LOCK_LINE.format(
                n=n_round if n_round is not None else "?", bucket=bucket)
                if locked else "")
        for name in ("train_patch_list", "segment_slice_data_list",
                     "data_to_add_list"):
            lst_c = getattr(self, name, None)
            if lst_c is not None:
                lst_c.setProperty(_LIST_LOCK_PROPERTY, bool(locked))
                lst_c.viewport().update()
        self.refresh_stage_line()
        line = getattr(self, "session_lock_line", None)
        if line is not None:
            n = _T.display_round(round_name)
            line.setText(_T.SESSION_LOCK_LINE.format(
                n=n if n is not None else "?", bucket=bucket) if locked
                else "")
        lst = getattr(self, "session_patch_list", None)
        if lst is not None:
            lst.setProperty(_LIST_LOCK_PROPERTY, bool(locked))
            lst.viewport().update()
        for name in ("train_patch_browse_button",
                     "train_patch_refresh_button"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(not locked)
        # ... and Training's own lock line, the setup column greyed per
        # widget, Train . Cancel (brick T)
        panel = getattr(self, "training_panel", None)
        if panel is not None:
            panel.on_training_locked(locked, round_name, bucket)

    def on_train_started(self, bucket: str, round_name: str) -> None:
        """Hook 3 (``_train_model``, beside ``begin_live_round``): the Loss
        collapsible opens on the live round, Train greys, Cancel lights, the
        stage line says "starting"."""
        panel = getattr(self, "training_panel", None)
        if panel is not None:
            panel.on_train_started(bucket, round_name)

    def on_train_finished(self) -> None:
        """Hook 6 (after the finished / cancelled round's curve is in the
        plot): the Loss collapsible closes back to its idle header."""
        panel = getattr(self, "training_panel", None)
        if panel is not None:
            panel.on_train_finished()

    def on_training_epoch(self, epoch, epochs, loss) -> None:
        """Seam (d): one epoch line from ``_on_training_progress`` -- the
        running stage line and the Loss header's live caption."""
        panel = getattr(self, "training_panel", None)
        if panel is not None:
            panel.on_epoch(epoch, epochs, loss)

    def _detect_gpu(self):
        """Detect GPU availability and update the status label."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self.status_gpu_value.setText(name)
                self.status_gpu_value.setStyleSheet(
                    f"font-weight: bold; color: {PANEL_OK_COLOR};")
            else:
                self.status_gpu_value.setText("CPU only")
                self.status_gpu_value.setStyleSheet(
                    f"color: {PANEL_WARN_COLOR};")
        except ImportError:
            self.status_gpu_value.setText("torch not installed")
            self.status_gpu_value.setStyleSheet(DIM_LABEL_STYLE)
        except Exception:
            self.status_gpu_value.setText("Unknown")
            self.status_gpu_value.setStyleSheet(DIM_LABEL_STYLE)

    def _create_curation_panel(self):
        """Curation (brick P): the lock line, the
        run row, then a horizontal splitter -- LEFT the Patches / Slices
        lists, RIGHT the View switch, Curation progress and Model blocks
        over the pinned foot.  The widgets the controller touches stay
        literal ``self.x = ...`` assignments; ``_step2_curation`` lays them
        out."""
        panel = QWidget()
        root = QVBoxLayout(panel)
        # the run row's "Run:" and the left column's "Patches" share a left
        # edge (polish D-F); the columns' own margins are the splitter's gap
        root.setContentsMargins(6, 2, 6, 2)
        root.setSpacing(4)
        self._curation_root_layout = root       # a short dock tightens it

        self.curation_lock_line = LockLine(panel)
        root.addWidget(self.curation_lock_line)

        # Run row (Phase 8a multi-run UI, brick P look): Run: picker, New
        # Run, Manage menu, ``?``.  The three old buttons stay as hidden
        # attributes the menu actions click, so the connections stand.
        self.run_picker_combo = QComboBox(parent=panel)
        self.run_picker_combo.setMinimumWidth(80)
        self.run_picker_combo.setToolTip(
            "Active training run.  Switch to load a different campaign's "
            "state (data, models, channels)."
        )
        self.new_run_button = QPushButton("New Run", parent=panel)
        apply_button_icon(self.new_run_button, "mdi6.plus-circle-outline",
                          color=PANEL_GOLD, size=14)
        self.new_run_button.setToolTip(
            "Create a new training run (campaign) in this workspace.  "
            "Useful for bootstrapping a separate model for nuclei, membrane, "
            "or experimenting with different segmentation styles."
        )
        self.import_pairs_button = QPushButton("Import Pairs", parent=panel)
        self.import_pairs_button.setToolTip(
            "Copy accepted training pairs FROM ANOTHER RUN in this workspace "
            "into the active run.  Useful for seeding a new run with curated "
            "samples from an existing one (e.g. take round_3 lateral pairs "
            "from 'default' and start training a focused model on them in "
            "'nuclei_v1').  This is intra-workspace; for files from outside "
            "EpiCycle use 'Import External' instead."
        )
        self.import_external_button = QPushButton("Import External",
                                                  parent=panel)
        self.import_external_button.setToolTip(
            "Import 2D image+mask pairs FROM OUTSIDE EpiCycle into this run.  "
            "Useful when you have curated training data from another tool "
            "and want to reuse it here.  You specify the pair filename "
            "convention (e.g. *_img.tif + *_mask.tif).  This is for files "
            "in arbitrary folders; for pairs from another EpiCycle run use "
            "'Import Pairs' instead."
        )
        self.manage_runs_button = QPushButton("Manage", parent=panel)
        self.manage_runs_button.setToolTip(
            "Manage runs: edit metadata, archive, unarchive, delete.  "
            "Delete is blocked if other runs depend on this one as their "
            "init source."
        )
        run_help = help_button(
            panel,
            guide_path=str(_STEP2_GUIDES_DIR / _C.GUIDE_RUN),
            title="Run",
            accent=_STEP2_ACCENT,
            tooltip="Help: Run (multi-run, switching, importing pairs)",
        )
        self.curation_run_row = _C.build_run_row(
            self.run_picker_combo, self.new_run_button,
            self.manage_runs_button, self.import_pairs_button,
            self.import_external_button, run_help, parent=panel)
        root.addWidget(self.curation_run_row)

        # Separator below run row
        _sep = QLabel("", parent=panel)
        _sep.setFixedHeight(1)
        _sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        root.addWidget(_sep)
        self._curation_sep = _sep               # hidden in a short dock

        # Step 3's horizontal splitter numbers (fusion_reconstruction/ui.py
        # :93-97): left min 240, right min 300, handle 6, stretch 0:1.
        splitter = _C.CompactSplitter(Qt.Horizontal)
        splitter.setStyleSheet(SPLITTER_STYLE)
        splitter.setHandleWidth(PANEL_SPLIT_HANDLE)
        splitter.setChildrenCollapsible(False)
        left = self._create_curation_patches_column()
        right = self._create_curation_config_column()
        left.setMinimumWidth(240)   # Step 3's 6 px gap before the handle
        right.setMinimumWidth(290)  # and its 8 px after (content 282)
        splitter.addWidget(left)
        splitter.addWidget(right)
        # the left column starts at its minimum and takes a third of any
        # extra width (the mockup shows ~400 px of 976); the chips wrap
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([240, 332])
        self.curation_splitter = splitter

        root.addWidget(splitter, 1)
        return panel

    def _create_curation_patches_column(self):
        """Left column: a vertical splitter, Patches over Slices (each pane keeps one row at the real 576 px
        dock, so the column's minimum height fits the ~218 px the section
        leaves under the run row)."""
        column = QWidget()
        outer = QVBoxLayout(column)
        outer.setContentsMargins(0, 2, PANEL_LEFT_INNER_MARGIN, 2)
        outer.setSpacing(0)
        vsplit = _C.CompactSplitter(Qt.Vertical)
        vsplit.setStyleSheet(SPLITTER_STYLE)
        vsplit.setHandleWidth(6)
        vsplit.setChildrenCollapsible(False)
        patch_pane = QWidget()
        layout = QVBoxLayout(patch_pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._left_descs = []

        # --- Patches ---
        _ph = _section_header_with_help(
            "Patches", guide_filename="step2_patches.md", parent=patch_pane,
        )
        layout.addLayout(_ph)

        patch_hint = QLabel(_T.CURATION_PATCHES_DESC, patch_pane)
        patch_hint.setWordWrap(True)
        patch_hint.setStyleSheet(FLOW_SUBTITLE_STYLE)
        self._left_descs.append((_ph.itemAt(0).widget(), patch_hint))
        # Step 2 look (brick L): Step 3's narrow rows (80 px picture, name,
        # dims, the five status counts, the strip, the loaded glow); a
        # QListWidget subclass, so every controller access keeps working.
        self.train_patch_list = build_curation_patch_list(patch_pane)
        # one whole row at the real 576 px dock (56 px picture in a short
        # dock, see ``_patch_list_min_h``); the splitter hands out any room
        self.train_patch_list.setMinimumHeight(self._patch_list_min_h(False))
        layout.addWidget(patch_hint)
        layout.addWidget(self.train_patch_list, 1)

        self._patch_pane_layout = layout
        vsplit.addWidget(patch_pane)

        # --- Slices (per-patch navigator) ---
        slice_pane = QWidget()
        layout = QVBoxLayout(slice_pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._slice_pane_layout = layout
        _sh = _section_header_with_help(
            "Slices", guide_filename="step2_slices.md", parent=slice_pane,
        )
        # the loaded patch + view, right-aligned before the `?` (mockup)
        self.slices_header_patch = QLabel("", slice_pane)
        self.slices_header_patch.setStyleSheet(L.SLICES_HEADER_PATCH_STYLE)
        self.slices_header_patch.setSizePolicy(QSizePolicy.Ignored,
                                               QSizePolicy.Preferred)
        self.slices_header_patch.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        _sh.insertWidget(2, self.slices_header_patch, 100)
        layout.addLayout(_sh)

        # Filter chips: icon + word + count, wrapping (two rows at 578 px).
        # Still checkable buttons -- the controller reads ``isChecked()``.
        from ...fusion_reconstruction.qc_plot_widget import FlowLayout
        filter_row = FlowLayout(h_spacing=3, v_spacing=4)
        _chip_colours = dict(_T.STATUS_COLOURS)
        _chip_colours["all"] = _T.FILTER_ALL_COLOUR

        self.filter_all_btn = QPushButton("All", slice_pane)
        self.filter_all_btn.setCheckable(True)
        self.filter_all_btn.setChecked(True)
        self.filter_predicted_btn = QPushButton("Predicted", slice_pane)
        self.filter_predicted_btn.setCheckable(True)
        self.filter_curating_btn = QPushButton("Curating", slice_pane)
        self.filter_curating_btn.setCheckable(True)
        self.filter_accepted_btn = QPushButton("Accepted", slice_pane)
        self.filter_accepted_btn.setCheckable(True)
        # "Finalized" = ready for training (internally "staged" + "committed").
        self.filter_staged_btn = QPushButton("Finalized", slice_pane)
        self.filter_staged_btn.setCheckable(True)
        for key, btn in (("all", self.filter_all_btn),
                         ("predicted", self.filter_predicted_btn),
                         ("curating", self.filter_curating_btn),
                         ("accepted", self.filter_accepted_btn),
                         ("staged", self.filter_staged_btn)):
            btn.setStyleSheet(L.filter_chip_style(_chip_colours[key]))
            btn.setCursor(Qt.PointingHandCursor)
            filter_row.addWidget(btn)

        # Hidden -- committed is merged into Finalized for display.
        self.filter_committed_btn = QPushButton(slice_pane)
        self.filter_committed_btn.setCheckable(True)
        self.filter_committed_btn.setVisible(False)

        layout.addLayout(filter_row)

        # Unified slice list with multi-select: narrow rows, a 56 px picture
        # of the actual slice with its labels (brick A).  A literal
        # assignment, as the lock-list source test reads this file.
        self.segment_slice_data_list = L.build_slice_list(slice_pane)
        self.segment_slice_data_list.setMinimumHeight(40)
        layout.addWidget(self.segment_slice_data_list, 2)

        # View button hidden — clicking a slice auto-navigates via
        # _jump_to_selected_queue_item (connected to itemSelectionChanged).
        # (Accept / Finalize / Discard / Revert live in the right column's
        # pinned foot, built by _create_curation_config_column.)
        self.view_results_button = QPushButton("View", column)
        self.view_results_button.setVisible(False)

        # Backward-compatible aliases (controller references these).
        self.accepted_queue_list = self.segment_slice_data_list
        self.staged_queue_list = self.segment_slice_data_list
        self.sample_queue_list = self.segment_slice_data_list

        # Hidden stub for workflow_shell backward compatibility.
        self.finalize_labels_button = QPushButton("Finalize Labels")
        self.finalize_labels_button.setVisible(False)

        vsplit.addWidget(slice_pane)
        vsplit.setStretchFactor(0, 1)
        vsplit.setStretchFactor(1, 1)
        outer.addWidget(vsplit, 1)
        self.curation_left_splitter = vsplit
        # the chips' words follow THIS column's width (CHIP_WORDS_MIN_W)
        self._chip_column = column
        column.installEventFilter(self)
        return column


    def _create_curation_config_column(self):
        """Right column (brick P): View switch, Curation progress and Model
        blocks in a scroll area over the pinned foot (Open 2D annotator,
        Accept / Finalize accepted / Discard / Revert, the busy bar, the
        stage line).  The widgets the controller touches are created here
        as literal attributes; ``_step2_curation`` lays them out."""
        column = QWidget()          # parent of the widgets built below

        # --- View: two hidden radios hold the state; the proxy clicks them --
        self.get_view_lateral = QRadioButton(_T.CUR_VIEW_LATERAL, column)
        self.get_view_axial = QRadioButton(_T.CUR_VIEW_AXIAL, column)
        self.get_view_lateral.setChecked(True)
        self.curation_view_switch = _C.ViewSwitch(
            self.get_view_lateral, self.get_view_axial)
        self.get_view_axial.toggled.connect(self._on_view_radio_toggled)

        # --- Curation progress: built once, the controller only refreshes ---
        self.curation_progress_table = _C.build_progress_table()
        self.curation_progress_container = self.curation_progress_table
        self.curation_progress_layout = self.curation_progress_table.layout()

        # Backward-compat labels (hidden, controller may still reference these).
        _compat = []
        for _name, _init in (("slice_summary_label", ""),
                             ("summary_round_value", "Round 0"),
                             ("summary_slices_value", "0"),
                             ("summary_masks_value", "Round 0"),
                             ("summary_total_rois_value", "0"),
                             ("summary_accepted_value", "0"),
                             ("summary_staged_value", "0")):
            _lbl = QLabel(_init)
            _lbl.setVisible(False)
            setattr(self, _name, _lbl)
            _compat.append(_lbl)

        # --- Model ---
        self.model_source_combo = QComboBox(parent=column)
        self.model_source_combo.addItems(
            [
                "Auto (Recommended)",
                "Round model",
                "External model file",
                "Cellpose pretrained",
            ]
        )
        self.model_source_combo.setToolTip(
            "Which model the PREDICT button uses (this is for PREDICTION only "
            "— it does NOT change what TRAINING starts from; that's the Base "
            "Model in Training Setup).\n\n"
            " • Auto (Recommended): a trained round checkpoint for this view if "
            "one exists, else the External model file if set, else the Cellpose "
            "Pretrained below.\n"
            " • Round model: ONLY a checkpoint trained in a prior round of this "
            "run — can't predict if none exists yet.\n"
            " • External model file: a .model checkpoint file on disk that "
            "you point to (remembered for this run).\n"
            " • Cellpose pretrained: a stock Cellpose builtin (the Pretrained "
            "dropdown below — cyto3/nuclei/...)."
        )

        self.imported_model_path = QLineEdit(parent=column)
        self.imported_model_path.setPlaceholderText("Path to a model file on disk...")
        self.imported_model_browse_button = QPushButton("Browse", parent=column)

        self.curation_pretrained_combo = QComboBox(parent=column)
        self.curation_pretrained_combo.addItems(
            [
                "cyto3", "nuclei", "cyto2_cp3", "tissuesect_cp3",
                "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3",
                "yeast_phase_cp3", "bact_flou_cp3", "deepbacs_cp3",
                "cyto", "CPx", "transformer_cp3",
            ]
        )
        # Auto-update toggle: when checked, training auto-switches to "auto"
        # model source so the newly trained checkpoint is used immediately.
        self.auto_update_model_cb = QCheckBox(_T.CUR_MODEL_AUTO_UPDATE,
                                              parent=column)
        self.auto_update_model_cb.setChecked(True)
        self.auto_update_model_cb.setToolTip(
            "When checked, model source automatically switches to 'Auto' after\n"
            "training completes so the new checkpoint is used for predictions.\n"
            "Uncheck to keep your current model source setting after training."
        )

        # Hidden stub — model source is auto-applied on combo change.
        self.apply_model_source_button = QPushButton("Apply Model Source")
        self.apply_model_source_button.setVisible(False)
        _compat.append(self.apply_model_source_button)

        # Active model: one neutral line; the controller
        # overwrites it on every refresh.  The old caption label stays as a
        # hidden attribute.
        self.active_model_label = QLabel("Active model:")
        self.active_model_label.setVisible(False)
        _compat.append(self.active_model_label)
        self.active_model_summary_value = _C.ElidedLabel(
            _T.CUR_MODEL_ACTIVE_LINE.format(lat="-", ax="-"))
        # Backward-compatible hidden labels.
        self.active_model_source_value = QLabel("-")
        self.active_lateral_model_value = QLabel("-")
        self.active_axial_model_value = QLabel("-")
        for _lbl in (self.active_model_source_value,
                     self.active_lateral_model_value,
                     self.active_axial_model_value):
            _lbl.setVisible(False)
            _compat.append(_lbl)

        # Inference parameters (collapsible)
        # 2026-05-26 layout port from Step 3 (fusion_reconstruction/ui.py:691-796):
        # 2-col QGridLayout sections separated by 1px gray lines, biology-
        # domain channel labels with tooltips, themed inputs.  Widget
        # attribute names preserved so controller signal wiring is unchanged.
        self._inference_collapsible = _disclosure(_T.CUR_MODEL_INFERENCE_TITLE)
        self.curation_inference_group = QWidget()
        infer_v = QVBoxLayout(self.curation_inference_group)
        infer_v.setContentsMargins(8, 8, 8, 8)
        infer_v.setSpacing(6)

        # ----- Preset (full-width row) -----
        self.inference_preset_combo = QComboBox()
        self.inference_preset_combo.addItems(["Balanced", "Conservative", "Aggressive", "Custom"])
        self.inference_preset_combo.setToolTip(
            "Tune the threshold defaults for different prediction styles:\n"
            "  • Balanced — default thresholds (flow=0.4, cellprob=0.0).\n"
            "  • Conservative — stricter; fewer false positives, may miss dim cells.\n"
            "  • Aggressive — lower thresholds; catches more cells, may add noise.\n"
            "  • Custom — manual values below."
        )
        preset_grid = QGridLayout()
        preset_grid.setHorizontalSpacing(6)
        preset_grid.setVerticalSpacing(4)
        preset_grid.addWidget(QLabel("Preset:"), 0, 0)
        preset_grid.addWidget(self.inference_preset_combo, 0, 1)
        infer_v.addLayout(preset_grid)

        _inf_sep_a = QLabel("")
        _inf_sep_a.setFixedHeight(1)
        _inf_sep_a.setStyleSheet(SEPARATOR_LINE_STYLE)
        infer_v.addWidget(_inf_sep_a)

        # ----- Channels -----
        self.inference_channel1_combo = QComboBox()
        self.inference_channel1_combo.addItems(["0: gray", "1: red", "2: green", "3: blue"])
        self.inference_channel1_combo.setCurrentIndex(2)  # default: green (membrane)
        self.inference_channel2_combo = QComboBox()
        self.inference_channel2_combo.addItems(["0: none", "1: red", "2: green", "3: blue"])
        self.inference_channel2_combo.setCurrentIndex(1)  # default: red (nuclei)
        # 2026-05-26 horizontal-resize fix (mirrors Advanced Parameters):
        # AdjustToMinimumContentsLengthWithIcon + Expanding policy means
        # the combo width stays bounded by the layout column, doesn't
        # force the dock to widen when long channel names come in via
        # _refresh_inference_channel_options.
        for _cb in (self.inference_channel1_combo, self.inference_channel2_combo):
            _cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            _cb.setMinimumContentsLength(8)
            _cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        _inf_chan1_label = QLabel(_T.CUR_MODEL_CHANNEL1_LABEL, self.curation_inference_group)
        _inf_chan1_label.setVisible(False)
        _inf_chan1_label.setToolTip(
            "The channel containing the objects you want segmented.\n"
            "  • For cells: typically the membrane / cytoplasm stain.\n"
            "  • For nuclei: the nuclear stain (DAPI, Hoechst, etc.).\n"
            "  • Use '0: gray' to collapse all channels into grayscale.\n\n"
            "When a patch is loaded, items show your viewer's channel "
            "names (e.g. '1: DAPI'); otherwise Cellpose's RGB defaults."
        )
        self.inference_channel1_combo.setToolTip(_inf_chan1_label.toolTip())
        _inf_chan2_label = QLabel(_T.CUR_MODEL_CHANNEL2_LABEL, self.curation_inference_group)
        _inf_chan2_label.setVisible(False)
        _inf_chan2_label.setToolTip(
            "Optional auxiliary channel that helps Cellpose separate "
            "touching objects.\n"
            "  • Typically the nuclear channel when segmenting cells.\n"
            "  • Leave on '0: none' if not using."
        )
        self.inference_channel2_combo.setToolTip(_inf_chan2_label.toolTip())
        # (the two channel combos live in the Model block, lifted out of
        # this disclosure)

        # ----- Thresholds -----
        self.inference_flow_thresh = QDoubleSpinBox()
        self.inference_flow_thresh.setRange(0.0, 3.0)
        self.inference_flow_thresh.setDecimals(3)
        self.inference_flow_thresh.setSingleStep(0.05)
        self.inference_flow_thresh.setValue(0.4)
        self.inference_flow_thresh.setToolTip(
            "Cellpose flow error threshold.  Higher = more permissive "
            "(keeps more masks).  Default 0.4."
        )
        self.inference_cellprob_thresh = QDoubleSpinBox()
        self.inference_cellprob_thresh.setRange(-10.0, 10.0)
        self.inference_cellprob_thresh.setDecimals(3)
        self.inference_cellprob_thresh.setSingleStep(0.1)
        self.inference_cellprob_thresh.setValue(0.0)
        self.inference_cellprob_thresh.setToolTip(
            "Cell probability threshold.  Lower = more masks (catches "
            "dim cells but may add false positives).  Default 0."
        )
        self.inference_diameter_sb = QSpinBox()
        self.inference_diameter_sb.setRange(0, 512)
        self.inference_diameter_sb.setValue(0)
        self.inference_diameter_sb.setSpecialValueText("Auto")
        self.inference_diameter_sb.setToolTip(
            "Expected object diameter in pixels.  0 = Auto (Cellpose "
            "estimates from the model + data)."
        )
        # 2026-05-27: moved from Advanced Parameters where it was an
        # orphan widget never reaching any Cellpose call.  Now wired to
        # the inference SegmentationConfig.niter argument via
        # _serialize_config + controller's prediction path.
        self.inference_niter_dynamics = QSpinBox()
        self.inference_niter_dynamics.setRange(0, 10000)
        self.inference_niter_dynamics.setValue(0)
        self.inference_niter_dynamics.setSpecialValueText("Auto")
        self.inference_niter_dynamics.setToolTip(
            "Number of dynamics simulation iterations during mask "
            "recovery.  After the network predicts flow vectors, "
            "Cellpose simulates particles following those vectors back "
            "to cell centers — this is the iteration count for that "
            "simulation.\n\n"
            " • Auto (0): Cellpose picks ~2× diameter (good default).\n"
            " • Higher: needed for unusually elongated or large cells "
            "where the default doesn't iterate enough to reach centers.\n"
            " • Lower: faster, fine for round/small cells."
        )
        thresh_grid = QGridLayout()
        thresh_grid.setHorizontalSpacing(6)
        thresh_grid.setVerticalSpacing(4)
        for i, (label, widget) in enumerate([
            ("Flow threshold:", self.inference_flow_thresh),
            ("Cellprob threshold:", self.inference_cellprob_thresh),
            ("Diameter (px):", self.inference_diameter_sb),
            ("Dynamics iterations:", self.inference_niter_dynamics),
        ]):
            thresh_grid.addWidget(QLabel(label), i, 0)
            thresh_grid.addWidget(widget, i, 1)
        infer_v.addLayout(thresh_grid)

        _inf_sep_c = QLabel("")
        _inf_sep_c.setFixedHeight(1)
        _inf_sep_c.setStyleSheet(SEPARATOR_LINE_STYLE)
        infer_v.addWidget(_inf_sep_c)

        # ----- Toggles + Apply (2-col checkbox grid) -----
        self.inference_normalize_cb = QCheckBox("Normalize")
        self.inference_normalize_cb.setChecked(True)
        self.inference_normalize_cb.setToolTip(
            "Per-channel intensity normalization (Cellpose default: ON)."
        )
        self.inference_rescale_cb = QCheckBox("Resample dynamics")
        self.inference_rescale_cb.setChecked(True)
        self.inference_rescale_cb.setToolTip(
            "Resample flow dynamics at the model's expected diameter (default: ON)."
        )
        self.inference_invert_cb = QCheckBox("Invert image")
        self.inference_invert_cb.setToolTip(
            "Invert image intensity before prediction.  Use for dark-cells-"
            "on-bright-background data (e.g. some phase contrast)."
        )
        toggle_grid = QGridLayout()
        toggle_grid.setHorizontalSpacing(12)
        toggle_grid.setVerticalSpacing(2)
        toggle_grid.addWidget(self.inference_normalize_cb, 0, 0)
        toggle_grid.addWidget(self.inference_rescale_cb, 0, 1)
        toggle_grid.addWidget(self.inference_invert_cb, 1, 0)
        infer_v.addLayout(toggle_grid)

        self.apply_inference_params_button = QPushButton("Apply Inference Params")
        self.apply_inference_params_button.setToolTip(
            "Push the parameters above into the active inference config.  "
            "Affects the next 'Predict' run."
        )
        infer_v.addWidget(self.apply_inference_params_button)

        self._inference_collapsible.addWidget(self.curation_inference_group)


        self.curation_model_block = _C.build_model_block(
            self.model_source_combo, self.active_model_summary_value,
            self.imported_model_path, self.imported_model_browse_button,
            self.curation_pretrained_combo, self.auto_update_model_cb,
            self.inference_channel1_combo, self.inference_channel2_combo,
            self._inference_collapsible)
        # The grey summary under the closed disclosure follows the knobs
        # (UI-side connections to a bound method: no controller seam).
        for _sig in (self.inference_preset_combo.currentTextChanged,
                     self.inference_channel1_combo.currentTextChanged,
                     self.inference_channel2_combo.currentTextChanged,
                     self.inference_flow_thresh.valueChanged,
                     self.inference_cellprob_thresh.valueChanged,
                     self.inference_diameter_sb.valueChanged):
            _sig.connect(self._refresh_inference_summary)
        self._refresh_inference_summary()

        # --- The pinned foot: the five buttons the controller wires ---
        self.accept_annotation_button = QPushButton("Accept", parent=column)
        # 2026-05-25 Group F: clarify the two-step Accept -> Finalize flow.
        # Users were confused because "Accept" set status="accepted" but the
        # cherry-pick dialog's default filter is "Finalized only (staged +
        # committed)" — so accepted pairs were filtered out + appeared
        # missing.  Tooltip spells out that "Finalize Accepted" is the
        # second step that promotes accepted -> staged ("finalized").
        self.accept_annotation_button.setToolTip(
            "Mark the selected slice(s) as ACCEPTED — their masks are "
            "written to disk for this round.\n\n"
            "Next step: click 'Finalize Accepted' to promote accepted "
            "slices to FINALIZED, which makes them eligible for training "
            "in the next round AND for cherry-pick import from another run."
        )
        self.reject_annotation_button = QPushButton("Discard", parent=column)
        self.stage_current_view_button = QPushButton("Finalize Accepted", parent=column)
        self.stage_current_view_button.setToolTip(
            "Promote ALL accepted slices in the current view from "
            "ACCEPTED to FINALIZED ('staged' internally).  This is the "
            "step that:\n"
            " • makes pairs visible to cherry-pick imports from other "
            "runs (default filter is 'Finalized only')\n"
            " • includes them in the next training round\n\n"
            "Idempotent — clicking again with no new accepted slices is "
            "a no-op."
        )
        self.unstage_selected_button = QPushButton("Revert", parent=column)
        self.unstage_selected_button.setToolTip(
            "Reopen the selected slice(s) for editing — sends them back to "
            "the editable 'curating' state regardless of current status "
            "(accepted / finalized / committed).\n\n"
            "This is the deliberate way to edit a finalized slice: the "
            "annotator soft-blocks edits on accepted/finalized slices, so "
            "you revert here first, edit, then re-Accept.\n\n"
            "Committed slices (already used to train a round) prompt for "
            "confirmation before reverting."
        )

        self.launch_multiview_button = QPushButton("Open 2D annotator",
                                                   parent=column)
        # The busy bar (Step 2 hardening 2026-09-07): PERMANENT indicator
        # for every long-running Step 2 operation.  Never hidden: idle it is
        # a static determinate-empty strip (setRange(0, 100) + setValue(0));
        # while busy the controller flips it to scan mode (setRange(0, 0)).
        # Brick P: it sits in the right column's pinned foot, with the busy
        # words (``on_busy``) and the stage line under it.
        # `training_progress_bar` stays TRAINING-only.
        self.patch_busy_bar = QProgressBar(parent=column)
        self.patch_busy_bar.setTextVisible(False)
        self.patch_busy_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.patch_busy_bar.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.patch_busy_bar.setRange(0, 100)
        self.patch_busy_bar.setValue(0)
        self.patch_busy_bar.setVisible(True)   # permanent — static when idle
        self.curation_stage_label = _C.ElidedLabel("", column)
        self.curation_foot = _C.build_pinned_foot(
            self.launch_multiview_button, self.accept_annotation_button,
            self.stage_current_view_button, self.reject_annotation_button,
            self.unstage_selected_button, self.patch_busy_bar,
            self.curation_stage_label)

        right = _C.build_right_column(
            self.curation_view_switch, self.curation_progress_table,
            self.curation_model_block, self.curation_foot,
            guides_dir=_STEP2_GUIDES_DIR)
        self.curation_right_column = right

        # --- SAM backward-compat stubs (hidden) ---
        self.sam_device_gpu = QRadioButton("Use GPU")
        self.sam_device_gpu.setVisible(False)
        self.sam_device_cpu = QRadioButton("Use CPU")
        self.sam_device_cpu.setChecked(True)
        self.sam_device_cpu.setVisible(False)
        self.select_sam_model_combo = QComboBox()
        self.select_sam_model_combo.addItems(["vit_h", "vit_l", "vit_b"])
        self.select_sam_model_combo.setVisible(False)
        self.load_sam_model_button = QPushButton("Load SAM Model")
        self.load_sam_model_button.setVisible(False)
        self.use_points = QCheckBox("Points")
        self.use_points.setVisible(False)
        self.use_box = QCheckBox("Boxes")
        self.use_box.setVisible(False)
        self.keep_old_results = QCheckBox("Keep Original Predictions")
        self.keep_old_results.setVisible(False)
        self.refine_results_button = QPushButton("Refine Selected Results")
        self.refine_results_button.setVisible(False)
        self.curation_sam_toggle = QPushButton("Show SAM + Refinement")
        self.curation_sam_toggle.setCheckable(True)
        self.curation_sam_toggle.setVisible(False)

        # Hidden backward-compat stubs.
        self.review_tabs = QTabWidget()
        self.review_tabs.setVisible(False)
        self.queue_filter_combo = QComboBox()
        self.queue_filter_combo.setVisible(False)


        # Hidden compat widgets: parent them so none is a taskbar orphan.
        for _w in _compat:
            _w.setParent(right)
            _w.setVisible(False)
        return right

    def bind_advanced_features_dock(self, adv_widget) -> None:
        """Rebind SAM attributes to the real Advanced Features dock widgets."""
        for attr in (
            "sam_device_gpu",
            "sam_device_cpu",
            "select_sam_model_combo",
            "load_sam_model_button",
            "use_points",
            "use_box",
            "keep_old_results",
            "refine_results_button",
        ):
            real = getattr(adv_widget, attr, None)
            if real is not None:
                setattr(self, attr, real)

    def _create_training_panel(self):
        """Training (brick T): LEFT the staged
        list + the honest count, RIGHT the setup column, both in scroll areas
        over a pinned foot (Train . Cancel, the bar, the stage line, the Loss
        collapsible) on a vertical splitter, the lock line first.  Every
        widget the controller or the lock list touches is still a literal
        ``self.<name> = ...`` assignment HERE (``test_training_safety`` reads
        this file's source); ``_step2_training`` places them."""
        setup = self._create_training_setup_widgets()
        # -- the staged list + the foot ----------------------------------
        self.dataset_staging = QLabel("")
        # Step 2 look (brick A): painted rows with a 56 px picture of the
        # finalized slice; ``item(i).text()`` stays the sample id.
        self.data_to_add_list = L.build_staged_list(None)
        # (its floor, 60 px = Step 3's PR_LIST_MIN_H, is set once, by
        # ``build_staged_column``)
        self.update_and_train_button = QPushButton("Train")
        self.train_cancel_btn = QPushButton("Cancel")
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setTextVisible(False)
        self.training_stage_label = _TR.ElidedLabel("")
        self.loss_plot = LossPlotWidget()
        self.loss_plot.setMinimumHeight(_TR.LOSS_MIN_H)
        # Keep legacy attribute for backward compatibility
        self.running_training_loss_plot = self.loss_plot
        def _help(title, guide):
            return lambda parent: help_button(
                parent, guide_path=str(_STEP2_GUIDES_DIR / guide),
                title=title, accent=_STEP2_ACCENT,
                tooltip=f"Help: {title}")

        panel = QWidget()
        setup_col = _TR.build_setup_column(
            parent=panel, **setup,
            history_button=self.training_history_button,
            help_factory=_help("Training setup", "step2_training_setup.md"))
        # widgets the training lock greys that the controller's lock list does
        # not own, each restored to its OWN prior state on unlock
        setup_col.lock_extra = [
            self.batch_sb, self.device_gpu, self.device_cpu, self.lr_db,
            self.wd_db, self.minmask_sb, self.chan1_combo, self.chan2_combo,
            self.norm_cb, self.rescale_cb, self.flow_cb,
            self.save_and_set_config_button, self.load_config_button,
            # audit M1: a Starting model picked mid-round would flip the
            # (greyed) Mode combo and so the NEXT round's mode
            self.model_selector]
        staged_col = _TR.build_staged_column(
            self.data_to_add_list, self.dataset_staging, parent=panel,
            help_factory=_help("Staged for training", "step2_round_run.md"))
        foot = _TR.build_training_foot(
            self.update_and_train_button, self.train_cancel_btn,
            self.training_progress_bar, self.training_stage_label,
            self.loss_plot, self.loss_plot.mode_combo, parent=panel)
        # the plot takes Step 3's amber palette: lateral in the accent, axial
        # the second hue (D5); ``set_grade_by_round`` (the builder) keeps it
        self.loss_plot.set_palette(STEP3_PLOT_PALETTE)
        # the plot's own view combo now sits in the Loss header; its words
        # follow the panel's words (Overlay / Current round / Per round) -- set AFTER
        # the builder (``set_grade_by_round`` re-applies the plot's own words)
        _mode = self.loss_plot.mode_combo
        for _i, _w in enumerate(_T.LOSS_MODE_WORDS):
            if _i < _mode.count():
                _mode.setItemText(_i, _w)
        # the two hidden stand-ins get the panel as their parent (no orphans)
        self._init_hint_label.setParent(panel)
        self.advanced_params_toggle.setParent(panel)
        # the idle sentence until the first controller refresh
        foot.set_stage_line(_T.compose_training_stage_line())
        self.training_setup_column = setup_col
        self.training_staged_column = staged_col
        self.training_foot = foot
        self.training_panel = _TR.build_training_panel(
            setup_col, staged_col, foot, parent=panel)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.training_panel, 1)
        return panel

    def _create_training_setup_widgets(self) -> dict:
        """The setup column's widgets (literal ``self.x =`` lines), returned
        as the keyword arguments of ``build_setup_column``."""
        # 2026-05-26 model-name display refactor: two read-only labels show
        # the exact filename each bucket's next Train writes (Cellpose's
        # on-disk filename is locked to ``{run_id}_{bucket}_round_NN``).
        # Wiring: controller._refresh_model_name_display recomputes on every
        # _refresh_round_status call; the colour is neutral (look rule 1).
        self.model_name_lateral_label = QLabel("—")
        self.model_name_axial_label = QLabel("—")
        for _ml in (self.model_name_lateral_label, self.model_name_axial_label):
            _ml.setStyleSheet(
                f"color: {PANEL_ZERO_COLOR}; font-size: 11px;"
                f" {MONO_FONT}"
                " background: transparent; border: none;"
                " padding-left: 2px;"
            )
            _ml.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.epochs_sb = QSpinBox()
        self.epochs_sb.setRange(1, 10000)
        self.epochs_sb.setValue(500)
        self.batch_sb = QSpinBox()
        self.batch_sb.setRange(1, 4096)
        self.batch_sb.setValue(8)
        self.batch_sb.setSingleStep(4)

        self.device_cpu = QRadioButton("CPU")
        self.device_gpu = QRadioButton("GPU")
        self.device_gpu.setChecked(True)

        self.model_selector = QComboBox()
        self.model_selector.addItems(
            [
                "scratch",
                "cyto3",
                "nuclei",
                "cyto2_cp3",
                "tissuesect_cp3",
                "livecell_cp3",
                "yeast_PhC_cp3",
                "yeast_BF_cp3",
                "yeast_phase_cp3",
                "bact_flou_cp3",
                "deepbacs_cp3",
                "cyto",
                "CPx",
                "transformer_cp3",
            ]
        )
        self.model_selector.setToolTip(
            "Starting model — the Cellpose weights TRAINING starts from when "
            "no file is set below.  (This drives TRAINING; it is separate "
            "from the 'Model source' dropdown in Curation, which only "
            "chooses what the Predict button uses.)\n\n"
            " • scratch = random initialization — true from-zero training.\n"
            " • cyto3 / nuclei / cyto2_cp3 / ... = fine-tune from that stock "
            "Cellpose model.\n\n"
            "To start training from an OUTSIDE model file or a prior "
            "checkpoint, put it in the file field below — it OVERRIDES "
            "this starting model (that is NOT 'scratch').  In the continue "
            "mode, round_01 onwards automatically inherits the previous "
            "round's model and this starting model is ignored."
        )

        # Starting file — for fine-tuning from a prior model or imported one.
        self.starting_checkpoint_path = QLineEdit()
        self.starting_checkpoint_path.setToolTip(
            "Optional .model file to fine-tune training FROM — an OUTSIDE model "
            "or a prior round's checkpoint.  When set it OVERRIDES the "
            "starting model (training fine-tunes from this file, not from "
            "scratch).  Leave empty to use the starting model.  Ignored in "
            "the continue mode from round_01 on (the prior round's model is "
            "used automatically)."
        )
        self.starting_checkpoint_path.setPlaceholderText("(optional) path to .model file...")
        self.starting_checkpoint_browse = QPushButton("Browse")

        # The controller still writes this label; the ONE hint under the Mode
        # combo mirrors its warning, so it stays present but hidden.
        self.starting_checkpoint_status = QLabel("No checkpoint — will use Base Model")
        self.starting_checkpoint_status.setStyleSheet(MONO_LABEL_STYLE)
        self.starting_checkpoint_status.setWordWrap(True)
        self.starting_checkpoint_status.setVisible(False)

        # 2026-05-27 init-transparency hint -- the controller's
        # ``_refresh_init_hint`` writes ``training_setup_column.set_hint``;
        # this label is the stub-compatible fallback and stays hidden.
        self._init_hint_label = QLabel("")
        self._init_hint_label.setWordWrap(True)
        self._init_hint_label.setVisible(False)

        # --- Advanced (collapsible; closed by the builder) ---
        self._training_adv_collapsible = _disclosure(_T.ADVANCED_TITLE)

        self.advanced_params_group = QWidget()
        adv_layout = QVBoxLayout(self.advanced_params_group)
        adv_layout.setContentsMargins(8, 8, 8, 8)
        adv_layout.setSpacing(6)

        def _plain(text):
            lab = QLabel(text)
            lab.setStyleSheet(PANEL_ROW_LABEL_STYLE)
            return lab

        def _note(text):
            lab = QLabel(text)
            lab.setWordWrap(True)
            lab.setStyleSheet(PANEL_HELP_STYLE)
            lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            return lab

        def _field_row(plain, engine, widget):
            """Plain name over the engine name on the left, the field
            (128 px) on the right."""
            row = QHBoxLayout()
            row.setSpacing(6)
            names = QVBoxLayout()
            names.setSpacing(0)
            names.addWidget(_plain(plain))
            if engine:
                eng = QLabel(engine)
                eng.setStyleSheet(ENGINE_NAME_STYLE)
                names.addWidget(eng)
            row.addLayout(names, 1)
            widget.setFixedWidth(FORM_FIELD_W)
            row.addWidget(widget)
            return row

        # ----- Training hyperparameters -----
        self.lr_db = QDoubleSpinBox()
        self.lr_db.setDecimals(6)
        self.lr_db.setRange(1e-6, 1.0)
        self.lr_db.setSingleStep(0.0001)
        self.lr_db.setValue(0.01)
        self.lr_db.setToolTip("Learning rate (Cellpose default: 0.01).")
        self.wd_db = QDoubleSpinBox()
        self.wd_db.setDecimals(6)
        self.wd_db.setRange(0.0, 1.0)
        self.wd_db.setValue(0.0001)
        self.wd_db.setToolTip("L2 weight decay (Cellpose default: 0.0001).")
        self.minmask_sb = QSpinBox()
        self.minmask_sb.setRange(0, 100)
        self.minmask_sb.setValue(5)
        self.minmask_sb.setToolTip(_T.ADV_MIN_OBJECTS_NOTE
                                   + "  Cellpose default: 5.")
        adv_layout.addLayout(_field_row(
            _T.ADV_LEARNING_RATE, _T.ADV_ENGINE_NAMES["learning_rate"],
            self.lr_db))
        adv_layout.addLayout(_field_row(
            _T.ADV_WEIGHT_DECAY, _T.ADV_ENGINE_NAMES["weight_decay"],
            self.wd_db))
        adv_layout.addLayout(_field_row(
            _T.ADV_MIN_OBJECTS, _T.ADV_ENGINE_NAMES["min_train_masks"],
            self.minmask_sb))
        adv_layout.addWidget(_note(_T.ADV_MIN_OBJECTS_NOTE))

        _sep_a = QLabel("")
        _sep_a.setFixedHeight(1)
        _sep_a.setStyleSheet(SEPARATOR_LINE_STYLE)
        adv_layout.addWidget(_sep_a)

        # ----- Channels (mirrors Step 3 chan1/chan2 wording + tooltips) -----
        self.chan1_combo = QComboBox()
        self.chan1_combo.addItems(["0: gray", "1: red", "2: green", "3: blue"])
        self.chan1_combo.setCurrentIndex(2)  # default: green (membrane channel)
        self.chan2_combo = QComboBox()
        self.chan2_combo.addItems(["0: none", "1: red", "2: green", "3: blue"])
        self.chan2_combo.setCurrentIndex(1)  # default: red (nuclear channel)
        # 2026-05-26 horizontal-resize fix: keep a combo from growing the
        # column to its longest item.
        for _cb in (self.chan1_combo, self.chan2_combo):
            _cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            _cb.setMinimumContentsLength(8)
            _cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        _chan_tip1 = (
            "The channel containing the objects you want segmented.\n"
            "  • For cells: typically the membrane / cytoplasm stain.\n"
            "  • For nuclei segmentation: the nuclear stain (DAPI, Hoechst, etc.).\n"
            "  • Use '0: gray' to collapse all channels into grayscale.\n\n"
            "When a patch is loaded, items show your viewer's channel "
            "names (e.g. '1: DAPI'); otherwise Cellpose's RGB defaults."
        )
        _chan_tip2 = (
            "Optional helper channel that helps Cellpose separate "
            "touching objects.\n"
            "  • Typically the nuclear channel when segmenting cells.\n"
            "  • Leave on '0: none' if not using."
        )
        self.chan1_combo.setToolTip(_chan_tip1)
        self.chan2_combo.setToolTip(_chan_tip2)
        chan_grid = QGridLayout()
        chan_grid.setHorizontalSpacing(6)
        chan_grid.setVerticalSpacing(4)
        chan_grid.setColumnStretch(0, 0)   # label column: hug content
        chan_grid.setColumnStretch(1, 1)   # combo column: absorb extra width
        _l1 = _plain(_T.ADV_CHANNEL1)
        _l1.setToolTip(_chan_tip1)
        _l2 = _plain(_T.ADV_CHANNEL2)
        _l2.setToolTip(_chan_tip2)
        chan_grid.addWidget(_l1, 0, 0)
        chan_grid.addWidget(self.chan1_combo, 0, 1)
        chan_grid.addWidget(_l2, 1, 0)
        chan_grid.addWidget(self.chan2_combo, 1, 1)
        adv_layout.addLayout(chan_grid)

        _sep_b = QLabel("")
        _sep_b.setFixedHeight(1)
        _sep_b.setStyleSheet(SEPARATOR_LINE_STYLE)
        adv_layout.addWidget(_sep_b)

        # ----- Boolean toggles -----
        # Only widgets that actually reach Cellpose's train_seg API are
        # surfaced (2026-05-27 cleanup: Augment, Compute cellprob, the
        # inference thresholds, niter_dynamics and the mode stub were dead).
        self.norm_cb = QCheckBox(_T.ADV_NORMALIZE)
        self.norm_cb.setChecked(True)
        self.norm_cb.setToolTip(
            "Per-channel intensity normalization using 1st/99th percentile "
            "rescale to [0,1].  Cellpose default: ON.  Disable only if "
            "your inputs are already normalized."
        )
        self.rescale_cb = QCheckBox(_T.ADV_RESCALE)
        self.rescale_cb.setChecked(True)
        self.rescale_cb.setToolTip(
            "Auto-resample each image to match Cellpose's expected cell "
            "diameter.  Default: ON — almost always correct."
        )
        self.flow_cb = QCheckBox(_T.ADV_FLOWS)
        self.flow_cb.setChecked(True)
        self.flow_cb.setToolTip(_T.ADV_FLOWS_NOTE)
        toggle_grid = QGridLayout()
        toggle_grid.setHorizontalSpacing(12)
        toggle_grid.setVerticalSpacing(2)
        toggle_grid.addWidget(self.norm_cb, 0, 0)
        toggle_grid.addWidget(self.rescale_cb, 0, 1)
        toggle_grid.addWidget(self.flow_cb, 1, 0)
        adv_layout.addLayout(toggle_grid)
        adv_layout.addWidget(_note(_T.ADV_FLOWS_NOTE))

        # Save/Load config -- power feature, inside the Advanced section.
        self.save_and_set_config_button = QPushButton("Save Config")
        self.load_config_button = QPushButton("Load Config")
        cfg_actions = QHBoxLayout()
        cfg_actions.setSpacing(6)
        cfg_actions.addWidget(self.save_and_set_config_button)
        cfg_actions.addWidget(self.load_config_button)
        adv_layout.addLayout(cfg_actions)

        # Training history button -- themed icon, no emoji; the builder
        # places it in the setup header.
        self.training_history_button = QPushButton("History")
        self.training_history_button.setToolTip(
            "Open the Training History dialog (non-modal).  Shows every "
            "round's init source, channels, pair count, and model "
            "lineage across both buckets."
        )
        self.training_history_button.setStyleSheet(SECONDARY_BTN_STYLE)
        apply_button_icon(self.training_history_button, "mdi6.history",
                          color=PANEL_GOLD)

        self._training_adv_collapsible.addWidget(self.advanced_params_group)
        self._training_adv_collapsible.expand(animate=False)

        # Hidden stub for backward compat (controller may reference this).
        self.advanced_params_toggle = QPushButton("Show Advanced Parameters")
        self.advanced_params_toggle.setCheckable(True)
        self.advanced_params_toggle.setVisible(False)

        # Training mode: the controller identifies the mode by the key words
        # "cumulative" / "scratch", so both stay in the new words (D15).
        self.training_mode_combo = QComboBox()
        self.training_mode_combo.addItems([_T.MODE_CONTINUE, _T.MODE_SCRATCH])
        self.training_mode_combo.setToolTip(
            "Both modes train on the SAME data — every accepted + staged pair "
            "from prior rounds plus this round's staged pairs.  The only "
            "difference is which weights each round STARTS from:\n\n"
            " • Continue from the last round (cumulative): fine-tune from "
            "the PREVIOUS round's model (round_01 onwards chains — the "
            "starting model is ignored).  Faster convergence; the model "
            "keeps what it learned.\n"
            " • Start again from the starting model (from scratch): every "
            "round restarts from the starting model / file instead of the "
            "prior round.  What that means depends on the starting model:\n"
            "      – 'scratch' = random init every round (true from-zero).\n"
            "      – 'cyto3' / 'nuclei' / ...  = fine-tune that stock "
            "Cellpose model fresh every round.\n"
            "      – a file set  = fine-tune from that outside/prior file "
            "every round.\n\n"
            "Per bucket — choosing 'from scratch' for one view doesn't affect "
            "the other bucket's model."
        )
        return dict(
            model_name_lateral_label=self.model_name_lateral_label,
            model_name_axial_label=self.model_name_axial_label,
            training_mode_combo=self.training_mode_combo,
            model_selector=self.model_selector,
            starting_checkpoint_path=self.starting_checkpoint_path,
            starting_checkpoint_browse=self.starting_checkpoint_browse,
            starting_checkpoint_status=self.starting_checkpoint_status,
            epochs_sb=self.epochs_sb, batch_sb=self.batch_sb,
            device_gpu=self.device_gpu, device_cpu=self.device_cpu,
            advanced_collapsible=self._training_adv_collapsible,
        )

    def _create_benchmarking_panel(self):
        """Optional cross-round benchmarking — compare model performance across rounds."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        helper = QLabel(
            "Compare model performance across training rounds. "
            "Run benchmarks on held-out slices to track improvement and "
            "identify when additional rounds yield diminishing returns."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(helper)

        # Placeholder controls.  NOT wired to the controller (journey #15):
        # the panel is behind ComingSoonOverlay and the existing
        # `_run_benchmark` does not do what the text above promises — it
        # compares final TRAINING losses across rounds and writes a JSON
        # this panel never reads, with no held-out prediction and no mask
        # quality.  Keeping the veil and dropping the connection is the
        # honest half: no button that silently writes a file nobody sees.
        self.run_cross_round_benchmark_button = QPushButton("Run Benchmark")
        self.run_cross_round_benchmark_button.setStyleSheet(PRIMARY_BTN_STYLE)
        self.run_cross_round_benchmark_button.setEnabled(False)

        bench_results = QLabel("No benchmark results yet.")
        bench_results.setStyleSheet(STATUS_LABEL_STYLE)
        bench_results.setWordWrap(True)

        layout.addWidget(self.run_cross_round_benchmark_button)
        layout.addWidget(bench_results)
        layout.addStretch(1)

        return ComingSoonOverlay(panel)

    def _create_diagnostics_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # GPU status
        # both headers carry a ``?`` for step2_diagnostics.md
        self.diag_system_header = _section_header_with_help(
            _T.DIAG_SYSTEM_HEADER, guide_filename="step2_diagnostics.md",
            parent=panel)
        layout.addLayout(self.diag_system_header)
        _sdesc = QLabel(_T.DIAG_SYSTEM_DESC)
        _sdesc.setWordWrap(True)
        _sdesc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(_sdesc)
        self.gpu_status_label = QLabel("GPU: checking...")
        self.gpu_status_label.setStyleSheet(MONO_LABEL_STYLE)
        layout.addWidget(self.gpu_status_label)
        self._detect_gpu_for_diagnostics()

        _sep = QLabel("")
        _sep.setFixedHeight(1)
        _sep.setStyleSheet(SEPARATOR_LINE_STYLE)
        layout.addWidget(_sep)

        # Log output
        self.diag_log_header = _section_header_with_help(
            _T.DIAG_LOG_HEADER, guide_filename="step2_diagnostics.md",
            parent=panel)
        layout.addLayout(self.diag_log_header)
        _desc = QLabel(_T.DIAG_LOG_DESC)
        _desc.setWordWrap(True)
        _desc.setStyleSheet(FLOW_SUBTITLE_STYLE)
        layout.addWidget(_desc)

        self.training_log_output = QTextEdit()
        self.training_log_output.setReadOnly(True)
        self.training_log_output.setStyleSheet(LOG_OUTPUT_STYLE)
        self.training_log_output.setMinimumHeight(220)
        layout.addWidget(self.training_log_output, 1)
        return panel

    def _detect_gpu_for_diagnostics(self):
        """Detect GPU for the diagnostics panel status label."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self.gpu_status_label.setText(f"GPU: {name}")
                self.gpu_status_label.setStyleSheet(
                    MONO_LABEL_STYLE.rstrip(";") + f"; color: {PANEL_OK_COLOR};"
                )
            else:
                self.gpu_status_label.setText("GPU: not available (CPU only)")
                self.gpu_status_label.setStyleSheet(
                    MONO_LABEL_STYLE.rstrip(";") + f"; color: {PANEL_WARN_COLOR};"
                )
        except ImportError:
            self.gpu_status_label.setText("GPU: torch not installed")
            self.gpu_status_label.setStyleSheet(
                MONO_LABEL_STYLE.rstrip(";") + f"; color: {PANEL_WARN_COLOR};"
            )
        except Exception:
            self.gpu_status_label.setText("GPU: unknown")
            self.gpu_status_label.setStyleSheet(MONO_LABEL_STYLE)

    def _toggle_advanced_params(self, checked):
        # Legacy stub — Advanced Parameters now uses QCollapsible.
        pass

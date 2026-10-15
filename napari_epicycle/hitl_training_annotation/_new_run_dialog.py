"""Themed Qt dialog for creating a new Step 2 training run (Phase 8a).

UI surface for ``_run_management.create_run_on_disk`` + the controller's
``_create_new_run`` wrapper.  Sections:

* Identity: slug (live-validated), display_name, target combo,
  description, backend radio (cellpose enabled; others greyed)
* Initialization: visual cards (Phase 8a MVP = scratch +
  pretrained only; Run model / Step 3 refined / Browse deferred to 8b)
* Channels: snapshot from viewer
* Preview: directory tree + action buttons

The dialog is Qt-dependent and NOT covered by the pure-Python test
suite for ``_run_management``.  Sub-pieces (slug validation,
collision suggestion, result-dict assembly) are tested in
``tests/hitl_training_annotation/test_new_run_dialog.py``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import _paths as _step2_paths
from packages.epicycle_io.step2_run_manifest import (
    ENABLED_BACKENDS,
    init_spec_browse,
    init_spec_inherited,
    init_spec_pretrained,
    init_spec_scratch,
    init_spec_step3_refined,
)


# ----------------------------------------------------------------------------
# Common targets list (UX hint — doesn't constrain the stored value)
# ----------------------------------------------------------------------------

_COMMON_TARGETS = [
    "(none)",
    "general",
    "nuclei",
    "membrane",
    "cytoplasm",
    "actin",
    "whole cell",
    "mitochondria",
    "(custom)",
]

# ----------------------------------------------------------------------------
# Built-in Cellpose pretrained models (UX dropdown — actual list maintained
# by Cellpose; we surface the most common ones)
# ----------------------------------------------------------------------------

_CELLPOSE_PRETRAINED = [
    "cyto3",
    "cyto2",
    "cyto",
    "nuclei",
    "tissuenet",
    "livecell",
]


# ----------------------------------------------------------------------------
# Validation helpers (pure — used by tests too)
# ----------------------------------------------------------------------------

def validate_slug_input(
    slug: str,
    existing_slugs: Iterable[str],
) -> tuple[bool, str]:
    """Validate a user-typed slug; return (is_valid, message).

    Pure function — no Qt.  Wraps ``_paths.validate_run_id`` (format
    check) + collision check against existing run IDs.

    Collision check is CASE-INSENSITIVE because the underlying
    ``mkdir(exist_ok=False)`` call in ``create_run_on_disk`` will fail
    on Windows / macOS (case-insensitive filesystems) even when the
    case differs — without case-folded comparison, the dialog would
    show "✓ Available" then creation would fail with a confusing
    "directory already exists" error.  Audit-found CRITICAL fix
    2026-05-23.

    Args:
        slug: user input.
        existing_slugs: iterable of slugs already on disk.

    Returns:
        (True, "Available") on success.
        (False, <reason>) on any failure.  Message is user-facing.
    """
    if not slug:
        return False, "Required"
    try:
        _step2_paths.validate_run_id(slug)
    except ValueError as e:
        # 2026-05-25 terminology sweep: validator now emits "Run handle"
        # in error messages directly, so no string-replace needed.  Keep
        # a generic stripper for any legacy callers that may still
        # surface "run_id " in their own custom error wording.
        msg = str(e).replace("run_id ", "Run handle ")
        return False, msg
    slug_lower = slug.lower()
    existing_lower = {s.lower() for s in existing_slugs}
    if slug_lower in existing_lower:
        return False, f"'{slug}' already exists in this workspace"
    return True, "Available"


def suggest_alternate_slug(
    slug: str,
    existing_slugs: Iterable[str],
) -> str:
    """If slug collides, suggest a numeric suffix that doesn't.

    e.g., "nuclei_v1" → "nuclei_v2" (or "nuclei_v3" if v2 also taken).
    For slugs not ending in a digit, appends "_2".

    Returns:
        Suggested slug.  If input slug has no collision, returns it
        unchanged.
    """
    existing = set(existing_slugs)
    if slug not in existing:
        return slug

    # Parse trailing digits
    import re
    m = re.search(r"(\d+)$", slug)
    if m:
        base = slug[:m.start()]
        n = int(m.group(1)) + 1
    else:
        base = slug + "_"
        n = 2

    while f"{base}{n}" in existing:
        n += 1
    return f"{base}{n}"


def derive_display_name_from_slug(slug: str) -> str:
    """Auto-suggest a display name from a slug.

    e.g., "nuclei_v1" → "Nuclei v1"; "membrane-test" → "Membrane Test"
    """
    parts = slug.replace("_", " ").replace("-", " ").split()
    return " ".join(p.capitalize() for p in parts)


# ----------------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------------

class NewRunDialog(QDialog):
    """Themed dialog for creating a new Step 2 training run.

    Usage::

        dlg = NewRunDialog(existing_slugs=existing, parent=self.ui)
        if dlg.exec_():
            result = dlg.get_result()
            controller._create_new_run(**result)

    Result dict shape::

        {
            "slug": str,
            "display_name": str,
            "target": str,
            "description": str,
            "backend": str,
            "init_spec": dict,            # from init_spec_* builders
            "channels_snapshot": list[str] | None,
            "notes": str,
            "activate": bool,
        }
    """

    def __init__(
        self,
        existing_slugs: Optional[Iterable[str]] = None,
        channels_snapshot: Optional[List[str]] = None,
        run_model_sources: Optional[List[Dict]] = None,
        step3_refined_sources: Optional[List[Dict]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._existing_slugs: Set[str] = set(existing_slugs or [])
        self._channels_snapshot: List[str] = list(channels_snapshot or [])
        self._run_model_sources = list(run_model_sources or [])
        self._step3_refined_sources = list(step3_refined_sources or [])
        self._browsed_external_path: str = ""
        self._result: Optional[dict] = None
        self._activate: bool = True  # which Create button was clicked

        self.setWindowTitle("Create New Training Run")
        from napari_epicycle.ui_shell.theme import (
            DIALOG_FORM_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        self._dialog_style = DIALOG_FORM_STYLE
        self._cancel_style = _DIALOG_BTN_CANCEL
        self._confirm_style = _DIALOG_BTN_CONFIRM

        self.setStyleSheet(self._dialog_style)
        self.setMinimumWidth(560)
        self.setMinimumHeight(620)

        self._build_ui()
        self._wire_signals()
        # Trigger initial validation pass so OK button reflects empty-slug state
        self._update_validation()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        # Title
        title = QLabel("Create New Training Run", parent=self)
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #eeeeee;")
        root.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "Each run is a training campaign with its own data, channels, "
            "and model.  Use a new run to bootstrap a separate model "
            "(e.g. nuclei, membrane) without disturbing existing runs.",
            parent=self,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        root.addWidget(subtitle)
        root.addSpacing(4)

        # ===== § Identity ===== (2026-05-26 theming sweep: QGroupBox,
        # styled at dialog level via DIALOG_FORM_STYLE — no per-group
        # setStyleSheet needed)
        identity_group = QGroupBox("Identity", parent=self)
        identity_form = QFormLayout(identity_group)
        identity_form.setContentsMargins(8, 14, 8, 8)
        identity_form.setSpacing(6)

        # Run handle (was "Slug" pre-2026-05-24 — biologists found "slug" too web-dev-y)
        self._slug_edit = QLineEdit(parent=identity_group)
        self._slug_edit.setPlaceholderText("nuclei_v1")
        self._slug_edit.setMaxLength(_step2_paths.RUN_ID_MAX_LENGTH)
        self._slug_edit.setToolTip(
            "Short identifier used as the run's folder name on disk.  Letters, "
            "digits, underscores, and dashes only.  Distinct from 'Display "
            "name' which can be any text."
        )
        identity_form.addRow("Run handle:", self._slug_edit)

        # Slug status (live)
        self._slug_status = QLabel("", parent=identity_group)
        self._slug_status.setStyleSheet("font-size: 10px;")
        self._slug_status.setMinimumHeight(14)
        identity_form.addRow("", self._slug_status)

        # Suggestion link (only shown on collision)
        self._slug_suggestion = QPushButton("", parent=identity_group)
        self._slug_suggestion.setFlat(True)
        self._slug_suggestion.setStyleSheet(
            "QPushButton { color: #6abf69; font-size: 10px; "
            "text-align: left; padding: 0; border: none; }"
            "QPushButton:hover { color: #8cdb8c; text-decoration: underline; }"
        )
        self._slug_suggestion.setVisible(False)
        identity_form.addRow("", self._slug_suggestion)

        # Display name
        self._display_name_edit = QLineEdit(parent=identity_group)
        self._display_name_edit.setPlaceholderText("Auto-suggested from run handle")
        identity_form.addRow("Display name:", self._display_name_edit)

        # Target (combo + free text)
        target_widget = QWidget(parent=identity_group)
        target_widget.setStyleSheet("background: transparent;")
        target_row = QHBoxLayout(target_widget)
        target_row.setContentsMargins(0, 0, 0, 0)
        target_row.setSpacing(6)
        from qtpy.QtWidgets import QComboBox
        self._target_combo = QComboBox(parent=target_widget)
        self._target_combo.addItems(_COMMON_TARGETS)
        self._target_combo.setEditable(True)
        target_row.addWidget(self._target_combo, 1)
        identity_form.addRow("Target:", target_widget)

        # Description (multiline)
        self._description_edit = QTextEdit(parent=identity_group)
        self._description_edit.setMaximumHeight(60)
        self._description_edit.setPlaceholderText(
            "Optional notes about this run's purpose."
        )
        identity_form.addRow("Description:", self._description_edit)

        # Backend (radio)
        backend_widget = QWidget(parent=identity_group)
        backend_widget.setStyleSheet("background: transparent;")
        backend_row = QHBoxLayout(backend_widget)
        backend_row.setContentsMargins(0, 0, 0, 0)
        backend_row.setSpacing(12)
        self._backend_group = QButtonGroup(backend_widget)
        self._backend_radios = {}
        for label, value, enabled in [
            ("Cellpose", "cellpose", True),
            ("Omnipose", "omnipose", False),
            ("StarDist", "stardist", False),
            ("Custom U-Net", "unet2d", False),
        ]:
            rb = QRadioButton(label, parent=backend_widget)
            rb.setEnabled(enabled and value in ENABLED_BACKENDS)
            if value == "cellpose":
                rb.setChecked(True)
            if not rb.isEnabled():
                rb.setToolTip("Coming in a future release")
                rb.setStyleSheet("QRadioButton { color: #666; }")
            backend_row.addWidget(rb)
            self._backend_group.addButton(rb)
            self._backend_radios[value] = rb
        backend_row.addStretch(1)
        identity_form.addRow("Backend:", backend_widget)

        root.addWidget(identity_group)

        # ===== § Initialization (Phase 8b — 5 modes) =====
        # 2026-05-26 theming sweep: wrap in QGroupBox (style at dialog level).
        init_widget = QGroupBox("Initialization", parent=self)
        init_layout = QVBoxLayout(init_widget)
        init_layout.setContentsMargins(8, 14, 8, 8)
        init_layout.setSpacing(8)

        # Init mode radio row — 5 options (Phase 8b adds 3 to MVP's 2)
        mode_widget = QWidget(parent=init_widget)
        mode_grid = QHBoxLayout(mode_widget)
        mode_grid.setContentsMargins(0, 0, 0, 0)
        mode_grid.setSpacing(8)

        from qtpy.QtWidgets import QStackedWidget, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView

        self._init_mode_group = QButtonGroup(mode_widget)
        self._init_scratch_rb = QRadioButton("⊘ Scratch", parent=mode_widget)
        self._init_scratch_rb.setToolTip("Random weights — start fresh.")
        self._init_scratch_rb.setChecked(True)
        self._init_run_model_rb = QRadioButton("↻ Run model", parent=mode_widget)
        self._init_run_model_rb.setToolTip(
            "Init from another run's trained model in this workspace."
        )
        self._init_step3_rb = QRadioButton("⚙ Step 3 refined", parent=mode_widget)
        self._init_step3_rb.setToolTip(
            "Init from a Step 3 refinement bundle in this workspace."
        )
        self._init_pretrained_rb = QRadioButton("⊕ Pretrained", parent=mode_widget)
        self._init_pretrained_rb.setToolTip(
            "Init from a Cellpose built-in model (cyto3, nuclei, etc.)."
        )
        self._init_browse_rb = QRadioButton("📁 Browse", parent=mode_widget)
        self._init_browse_rb.setToolTip(
            "Init from any weights file on disk."
        )
        for rb in (self._init_scratch_rb, self._init_run_model_rb,
                   self._init_step3_rb, self._init_pretrained_rb,
                   self._init_browse_rb):
            self._init_mode_group.addButton(rb)
            mode_grid.addWidget(rb)
        mode_grid.addStretch(1)
        # Disable empty sources (no run models / no Step 3 refined)
        if not self._run_model_sources:
            self._init_run_model_rb.setEnabled(False)
            self._init_run_model_rb.setToolTip(
                "No other runs have trained models in this workspace yet."
            )
        if not self._step3_refined_sources:
            self._init_step3_rb.setEnabled(False)
            self._init_step3_rb.setToolTip(
                "No Step 3 refinement bundles in this workspace yet."
            )
        init_layout.addWidget(mode_widget)

        # Stacked widget — sub-form swaps based on selected mode
        self._init_stack = QStackedWidget(parent=init_widget)

        # Page 0: Scratch (no sub-options)
        scratch_page = QWidget(parent=self._init_stack)
        scratch_page.setStyleSheet("background: transparent;")
        scratch_v = QVBoxLayout(scratch_page)
        scratch_v.setContentsMargins(0, 0, 0, 0)
        scratch_lbl = QLabel(
            "Random Cellpose initialization.  No prior weights — model trains "
            "from scratch.  Slowest convergence but no inherited biases.",
            parent=scratch_page,
        )
        scratch_lbl.setWordWrap(True)
        scratch_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        scratch_v.addWidget(scratch_lbl)
        scratch_v.addStretch(1)
        self._init_stack.addWidget(scratch_page)

        # Page 1: Run model — table of (run, bucket, round)
        run_model_page = QWidget(parent=self._init_stack)
        run_model_page.setStyleSheet("background: transparent;")
        run_model_v = QVBoxLayout(run_model_page)
        run_model_v.setContentsMargins(0, 0, 0, 0)
        run_model_v.setSpacing(4)
        run_model_hint = QLabel(
            "Pick a trained model from another run in this workspace.",
            parent=run_model_page,
        )
        run_model_hint.setStyleSheet("color: #aaa; font-size: 10px;")
        run_model_v.addWidget(run_model_hint)
        self._run_model_table = QTableWidget(parent=run_model_page)
        self._run_model_table.setColumnCount(4)
        self._run_model_table.setHorizontalHeaderLabels(
            ["Run", "Bucket", "Round", "Last modified"]
        )
        self._run_model_table.setMaximumHeight(140)
        self._run_model_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._run_model_table.setSelectionMode(QTableWidget.SingleSelection)
        self._run_model_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._run_model_table.verticalHeader().setVisible(False)
        # 2026-05-26 theming sweep — match Manage Runs styling.
        from napari_epicycle.ui_shell.theme import themed_table_style
        self._run_model_table.setShowGrid(False)
        self._run_model_table.setAlternatingRowColors(True)
        self._run_model_table.setStyleSheet(themed_table_style("#d4a843"))
        hdr = self._run_model_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._populate_run_model_table()
        run_model_v.addWidget(self._run_model_table)
        self._init_stack.addWidget(run_model_page)

        # Page 2: Step 3 refined — table of (bundle, bucket)
        step3_page = QWidget(parent=self._init_stack)
        step3_page.setStyleSheet("background: transparent;")
        step3_v = QVBoxLayout(step3_page)
        step3_v.setContentsMargins(0, 0, 0, 0)
        step3_v.setSpacing(4)
        step3_hint = QLabel(
            "Pick a Step 3 refinement bundle.  Refined models tend to be "
            "more polished than HITL training output.",
            parent=step3_page,
        )
        step3_hint.setWordWrap(True)
        step3_hint.setStyleSheet("color: #aaa; font-size: 10px;")
        step3_v.addWidget(step3_hint)
        self._step3_table = QTableWidget(parent=step3_page)
        self._step3_table.setColumnCount(3)
        self._step3_table.setHorizontalHeaderLabels(
            ["Bundle", "Bucket", "Last modified"]
        )
        self._step3_table.setMaximumHeight(140)
        self._step3_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._step3_table.setSelectionMode(QTableWidget.SingleSelection)
        self._step3_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._step3_table.verticalHeader().setVisible(False)
        # 2026-05-26 theming sweep — match Manage Runs styling.
        self._step3_table.setShowGrid(False)
        self._step3_table.setAlternatingRowColors(True)
        self._step3_table.setStyleSheet(themed_table_style("#d4a843"))
        hdr = self._step3_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._populate_step3_table()
        step3_v.addWidget(self._step3_table)
        self._init_stack.addWidget(step3_page)

        # Page 3: Pretrained (existing combo)
        pretrained_page = QWidget(parent=self._init_stack)
        pretrained_page.setStyleSheet("background: transparent;")
        pretrained_v = QVBoxLayout(pretrained_page)
        pretrained_v.setContentsMargins(0, 0, 0, 0)
        pretrained_v.setSpacing(4)
        pretrained_lbl = QLabel(
            "Pick a Cellpose built-in pretrained model.  Most users start "
            "with cyto3 (general) or nuclei (for DAPI-style channels).",
            parent=pretrained_page,
        )
        pretrained_lbl.setWordWrap(True)
        pretrained_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        pretrained_v.addWidget(pretrained_lbl)
        self._pretrained_combo = QComboBox(parent=pretrained_page)
        self._pretrained_combo.addItems(_CELLPOSE_PRETRAINED)
        pretrained_v.addWidget(self._pretrained_combo)
        pretrained_v.addStretch(1)
        self._init_stack.addWidget(pretrained_page)

        # Page 4: Browse external
        browse_page = QWidget(parent=self._init_stack)
        browse_page.setStyleSheet("background: transparent;")
        browse_v = QVBoxLayout(browse_page)
        browse_v.setContentsMargins(0, 0, 0, 0)
        browse_v.setSpacing(4)
        browse_hint = QLabel(
            "Pick any weights file on disk (e.g. a model from another "
            "workspace or a manually-saved checkpoint).",
            parent=browse_page,
        )
        browse_hint.setWordWrap(True)
        browse_hint.setStyleSheet("color: #aaa; font-size: 10px;")
        browse_v.addWidget(browse_hint)
        browse_btn_row = QHBoxLayout()
        self._browse_btn = QPushButton("Browse...", parent=browse_page)
        self._browse_btn.setStyleSheet(self._cancel_style)
        self._browsed_label = QLabel("(no file selected)", parent=browse_page)
        self._browsed_label.setStyleSheet(
            "color: #aaccee; font-size: 10px; "
            "font-family: 'Courier New', monospace;"
        )
        self._browsed_label.setWordWrap(True)
        browse_btn_row.addWidget(self._browse_btn)
        browse_btn_row.addWidget(self._browsed_label, 1)
        browse_v.addLayout(browse_btn_row)
        browse_v.addStretch(1)
        self._init_stack.addWidget(browse_page)

        init_layout.addWidget(self._init_stack)
        root.addWidget(init_widget)

        # ===== § Channels ===== (2026-05-26 theming sweep)
        channels_group = QGroupBox("Channels", parent=self)
        channels_v = QVBoxLayout(channels_group)
        channels_v.setContentsMargins(8, 14, 8, 8)
        channels_v.setSpacing(4)
        if self._channels_snapshot:
            ch_text = ", ".join(self._channels_snapshot)
            ch_msg = f"Viewer channels: {ch_text}"
        else:
            ch_msg = "No channels detected in viewer (or no viewer running)."
        ch_label = QLabel(ch_msg, parent=channels_group)
        ch_label.setWordWrap(True)
        ch_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        channels_v.addWidget(ch_label)
        root.addWidget(channels_group)

        # ===== § Preview =====
        preview_group = QGroupBox("Preview", parent=self)
        preview_v = QVBoxLayout(preview_group)
        preview_v.setContentsMargins(8, 14, 8, 8)
        preview_v.setSpacing(4)
        self._preview_label = QLabel("(run handle not yet entered)", parent=preview_group)
        self._preview_label.setStyleSheet(
            "color: #aaccee; font-family: 'Courier New', monospace; "
            "font-size: 11px;"
        )
        self._preview_label.setWordWrap(True)
        preview_v.addWidget(self._preview_label)
        root.addWidget(preview_group)

        root.addStretch(1)

        # ===== Button row =====
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self._cancel_btn = QPushButton("Cancel", parent=self)
        self._cancel_btn.setStyleSheet(self._cancel_style)
        btn_row.addWidget(self._cancel_btn)

        self._create_only_btn = QPushButton("Create only", parent=self)
        self._create_only_btn.setStyleSheet(self._cancel_style)
        self._create_only_btn.setToolTip(
            "Create the run but stay on the current run.  Useful for "
            "prepping multiple runs ahead of time."
        )
        btn_row.addWidget(self._create_only_btn)

        self._create_activate_btn = QPushButton("Create + Activate", parent=self)
        self._create_activate_btn.setStyleSheet(self._confirm_style)
        self._create_activate_btn.setDefault(True)
        btn_row.addWidget(self._create_activate_btn)

        root.addLayout(btn_row)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text, parent=self)
        lbl.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #d4a843; "
            "padding-top: 4px;"
        )
        return lbl

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _wire_signals(self):
        self._display_name_user_edited = False
        self._slug_edit.textChanged.connect(self._on_slug_changed)
        self._display_name_edit.textEdited.connect(self._on_display_name_changed)
        self._slug_suggestion.clicked.connect(self._apply_suggestion)
        # 5-mode init picker — swap stacked widget pages
        self._init_scratch_rb.toggled.connect(
            lambda checked: checked and self._init_stack.setCurrentIndex(0)
        )
        self._init_run_model_rb.toggled.connect(
            lambda checked: checked and self._init_stack.setCurrentIndex(1)
        )
        self._init_step3_rb.toggled.connect(
            lambda checked: checked and self._init_stack.setCurrentIndex(2)
        )
        self._init_pretrained_rb.toggled.connect(
            lambda checked: checked and self._init_stack.setCurrentIndex(3)
        )
        self._init_browse_rb.toggled.connect(
            lambda checked: checked and self._init_stack.setCurrentIndex(4)
        )
        self._browse_btn.clicked.connect(self._on_browse_clicked)
        self._cancel_btn.clicked.connect(self.reject)
        self._create_only_btn.clicked.connect(self._on_create_only)
        self._create_activate_btn.clicked.connect(self._on_create_activate)

    def _populate_run_model_table(self):
        """Fill the Run model table with available source models."""
        from qtpy.QtWidgets import QTableWidgetItem
        sources = [s for s in self._run_model_sources if s.get("exists")]
        self._run_model_table.setRowCount(len(sources))
        for row, src in enumerate(sources):
            # 1b#7: `display_label` carries the "inference failed" marker.
            display = (
                src.get("display_label")
                or src.get("run_display_name")
                or src["run_id"]
            )
            mtime = src.get("mtime_iso") or "-"
            items = [
                QTableWidgetItem(display),
                QTableWidgetItem(src["bucket"]),
                QTableWidgetItem(src["round_id"]),
                QTableWidgetItem(mtime[:10] if len(mtime) >= 10 else mtime),
            ]
            for col, it in enumerate(items):
                self._run_model_table.setItem(row, col, it)
            # Stash the dict on the first cell for later retrieval
            items[0].setData(Qt.UserRole, src)
        if sources:
            self._run_model_table.selectRow(0)

    def _populate_step3_table(self):
        """Fill the Step 3 refined table with available source bundles."""
        from qtpy.QtWidgets import QTableWidgetItem
        sources = [s for s in self._step3_refined_sources if s.get("exists")]
        self._step3_table.setRowCount(len(sources))
        for row, src in enumerate(sources):
            mtime = src.get("mtime_iso") or "-"
            items = [
                QTableWidgetItem(src["bundle_name"]),
                QTableWidgetItem(src["bucket"]),
                QTableWidgetItem(mtime[:10] if len(mtime) >= 10 else mtime),
            ]
            for col, it in enumerate(items):
                self._step3_table.setItem(row, col, it)
            items[0].setData(Qt.UserRole, src)
        if sources:
            self._step3_table.selectRow(0)

    def _on_browse_clicked(self):
        """Open file picker for external weights file."""
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pick weights file",
            "",
            "All Files (*.*)",  # Cellpose weights have no consistent extension
        )
        if not path:
            return
        self._browsed_external_path = path
        # Show just the filename if path is long; full path in tooltip
        from pathlib import Path as _Path
        name = _Path(path).name or path
        self._browsed_label.setText(name)
        self._browsed_label.setToolTip(path)

    def _on_slug_changed(self, text: str):
        # Auto-fill display_name as user types the slug.  Track whether the
        # user has manually edited display_name — if so, stop auto-syncing.
        # Audit-feedback UX fix 2026-05-23: previously only the PLACEHOLDER
        # was updated, which felt subtle and many users never noticed.
        if not getattr(self, "_display_name_user_edited", False):
            derived = derive_display_name_from_slug(text) if text else ""
            self._display_name_edit.blockSignals(True)
            try:
                self._display_name_edit.setText(derived)
            finally:
                self._display_name_edit.blockSignals(False)
        self._update_validation()
        self._update_preview()

    def _on_display_name_changed(self, text: str):
        """Track that user has manually edited display_name so we stop
        auto-syncing from slug.  Reset when display_name is cleared."""
        if not text:
            self._display_name_user_edited = False
        else:
            # Only mark user-edited if the text DIFFERS from what slug-derive
            # would produce — that way programmatic setText in _on_slug_changed
            # doesn't trip the flag.
            derived = derive_display_name_from_slug(self._slug_edit.text())
            if text != derived:
                self._display_name_user_edited = True

    def _apply_suggestion(self):
        suggested = self._slug_suggestion.text().replace("Try '", "").rstrip("'")
        if suggested:
            self._slug_edit.setText(suggested)

    def _update_validation(self):
        slug = self._slug_edit.text().strip()
        is_valid, message = validate_slug_input(slug, self._existing_slugs)
        if is_valid:
            self._slug_status.setText(f"✓ {message}")
            self._slug_status.setStyleSheet("color: #6abf69; font-size: 10px;")
            self._slug_suggestion.setVisible(False)
        else:
            prefix = "⚠" if "already exists" in message else "✗"
            self._slug_status.setText(f"{prefix} {message}")
            color = "#d4a843" if "already exists" in message else "#e57373"
            self._slug_status.setStyleSheet(f"color: {color}; font-size: 10px;")
            # Show suggestion on collision
            if "already exists" in message and slug:
                suggested = suggest_alternate_slug(slug, self._existing_slugs)
                self._slug_suggestion.setText(f"Try '{suggested}'")
                self._slug_suggestion.setVisible(True)
            else:
                self._slug_suggestion.setVisible(False)
        # Toggle create buttons
        self._create_only_btn.setEnabled(is_valid)
        self._create_activate_btn.setEnabled(is_valid)

    def _update_preview(self):
        slug = self._slug_edit.text().strip()
        if not slug:
            self._preview_label.setText("(run handle not yet entered)")
            return
        # Resolve target from the form (falls back to <target> placeholder
        # if user hasn't picked one)
        target_text = self._target_combo.currentText().strip()
        if target_text in ("(none)", "(custom)") or not target_text:
            target_text = "<unset>"
        self._preview_label.setText(
            f"artifacts/step2/runs/{slug}/\n"
            f"  ├─ run_manifest.json    (cellpose, target={target_text})\n"
            f"  ├─ lateral/             (empty until first training)\n"
            f"  └─ axial/               (empty until first training)"
        )

    def _get_selected_run_model_source(self) -> Optional[Dict]:
        """Return the dict stashed on the selected Run model row, or None."""
        rows = self._run_model_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self._run_model_table.item(rows[0].row(), 0)
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _get_selected_step3_source(self) -> Optional[Dict]:
        """Return the dict stashed on the selected Step 3 row, or None."""
        rows = self._step3_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self._step3_table.item(rows[0].row(), 0)
        if item is None:
            return None
        return item.data(Qt.UserRole)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def _on_create_only(self):
        self._activate = False
        self._submit()

    def _on_create_activate(self):
        self._activate = True
        self._submit()

    def _submit(self):
        slug = self._slug_edit.text().strip()
        is_valid, _ = validate_slug_input(slug, self._existing_slugs)
        if not is_valid:
            return  # button should be disabled, but defensive

        display_name = self._display_name_edit.text().strip()
        if not display_name:
            display_name = derive_display_name_from_slug(slug)

        target_raw = self._target_combo.currentText().strip()
        if target_raw in ("(none)", "(custom)"):
            target = "" if target_raw == "(none)" else ""
        else:
            target = target_raw

        description = self._description_edit.toPlainText().strip()

        backend = "cellpose"
        for value, rb in self._backend_radios.items():
            if rb.isChecked():
                backend = value
                break

        # Resolve init_spec per selected mode (Phase 8b — 5 modes).
        # Audit-found 2026-05-23: validate selections before submit.
        # Previously these silently fell back to scratch + console print,
        # so user got an unexpected scratch run.  Now block submit + show
        # inline error.
        from qtpy.QtWidgets import QMessageBox
        if self._init_run_model_rb.isChecked():
            src = self._get_selected_run_model_source()
            if src is None:
                QMessageBox.warning(
                    self, "No model selected",
                    "Picked 'Run model' but no row selected in the table.\n"
                    "Please pick a row or switch to a different init mode."
                )
                return
            init_spec = init_spec_inherited(
                source_run_id=src["run_id"],
                source_bucket=src["bucket"],
                source_round_id=src["round_id"],
            )
        elif self._init_step3_rb.isChecked():
            src = self._get_selected_step3_source()
            if src is None:
                QMessageBox.warning(
                    self, "No bundle selected",
                    "Picked 'Step 3 refined' but no row selected in the table.\n"
                    "Please pick a row or switch to a different init mode."
                )
                return
            init_spec = init_spec_step3_refined(
                source_step3_bundle=src["bundle_name"],
                source_step3_bucket=src["bucket"],
            )
        elif self._init_pretrained_rb.isChecked():
            pretrained = self._pretrained_combo.currentText().strip() or "cyto3"
            init_spec = init_spec_pretrained(pretrained_name=pretrained)
        elif self._init_browse_rb.isChecked():
            if not self._browsed_external_path:
                QMessageBox.warning(
                    self, "No file selected",
                    "Picked 'Browse' but no weights file selected.\n"
                    "Please click Browse... and pick a file, or switch "
                    "to a different init mode."
                )
                return
            init_spec = init_spec_browse(
                external_path=self._browsed_external_path,
            )
        else:
            init_spec = init_spec_scratch()

        self._result = {
            "slug": slug,
            "display_name": display_name,
            "target": target,
            "description": description,
            "backend": backend,
            "init_spec": init_spec,
            "channels_snapshot": list(self._channels_snapshot),
            "notes": "",
            "activate": self._activate,
        }
        self.accept()

    def get_result(self) -> Optional[dict]:
        """Return the result dict from a successful submit, or None if cancelled."""
        return self._result


__all__ = [
    "NewRunDialog",
    "validate_slug_input",
    "suggest_alternate_slug",
    "derive_display_name_from_slug",
]

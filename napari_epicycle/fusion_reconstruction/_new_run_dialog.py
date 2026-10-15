"""Themed Qt dialog for creating a new Step 3 reconstruction/curation run.

The Step 3 mirror of ``hitl_training_annotation/_new_run_dialog.py``
(Phase 4a — port the run-management dialogs).  UI surface for
``_run_management.create_run_on_disk`` + the (Phase 4b) controller
``_create_new_step3_run`` wrapper.

ADAPTATIONS from the Step 2 original (see module docstring of
``packages/epicycle_io/step3_run_manifest.py`` for the why):

* **4-mode init picker, NO scratch.**  Step 3 ALWAYS starts from a model
  (it segments + reconstructs) — there is no random-init "scratch" mode.
  The 4 modes are ``pretrained`` / ``step2_run`` / ``step3_refined`` /
  ``browse``, each driving a ``QStackedWidget`` page.  A radio is
  disabled when its source list is empty; the dialog defaults to the
  first ENABLED mode.
* **No channels snapshot section.**  Step 3 runs reuse the workspace's
  patch stacks; there is no per-run channel campaign like Step 2.
* **"Activate after create" checkbox** instead of Step 2's two Create
  buttons (Create only / Create + Activate).
* **Reserved-slug guard** rejects ``"default"`` (the implicit FLAT
  default run has no context dir).

The Qt dialog itself is NOT covered by the pure-Python test suite (Qt
teardown crashes the process).  The pure
pieces — ``validate_slug_input``, ``suggest_alternate_slug``,
``derive_display_name_from_slug``, ``build_init_spec_for_mode`` — ARE
tested in ``tests/fusion_reconstruction/test_step3_new_run_dialog.py``
and are importable WITHOUT constructing the QDialog.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import _paths as _step3_paths
from ._migrate_step3_runs import DEFAULT_RUN_ID
from packages.epicycle_io.step3_run_manifest import (
    init_spec_browse,
    init_spec_pretrained,
    init_spec_step2_run,
    init_spec_step3_refined,
)


# ----------------------------------------------------------------------------
# Suggested target vocabulary (UX hint — doesn't constrain the stored value;
# free text is accepted).  Mirrors VALID_TARGETS + "(none)"/"(custom)" UX
# sentinels.  VALID_TARGETS already contains "cells".
# ----------------------------------------------------------------------------

_TARGET_CHOICES = [
    "(none)",
    "cells",
    "nuclei",
    "membrane",
    "general",
    "(custom)",
]

# Built-in Cellpose pretrained models surfaced in the Pretrained combo when
# the caller doesn't pass an explicit list.
_CELLPOSE_PRETRAINED = [
    "cyto3",
    "cyto2",
    "cyto",
    "nuclei",
    "tissuenet",
    "livecell",
]

# Slugs that may never be used for a named run (reserved identities).
_RESERVED_SLUGS = frozenset({DEFAULT_RUN_ID})

# Init-mode identifiers (match step3_run_manifest VALID_INIT_MODES).
_MODE_PRETRAINED = "pretrained"
_MODE_STEP2_RUN = "step2_run"
_MODE_STEP3_REFINED = "step3_refined"
_MODE_BROWSE = "browse"


# ----------------------------------------------------------------------------
# Validation helpers (pure — used by tests too; NO Qt)
# ----------------------------------------------------------------------------

def validate_slug_input(
    slug: str,
    existing_slugs: Iterable[str],
) -> tuple[bool, str]:
    """Validate a user-typed run handle; return ``(is_valid, message)``.

    Pure function — no Qt.  Wraps ``_paths.validate_run_id`` (format
    check) + a reserved-slug guard (``"default"``) + a CASE-FOLDED
    collision check against existing run handles.

    The collision + reserved checks are case-insensitive because
    ``create_run_on_disk``'s ``mkdir(exist_ok=False)`` fails on
    Windows / macOS (case-insensitive filesystems) even when the case
    differs — without case-folding, the dialog would show
    "✓ Available" then creation would fail.

    Returns:
        ``(True, "Available")`` on success; ``(False, <reason>)`` on any
        failure.  Message is user-facing.
    """
    if not slug:
        return False, "Required"
    try:
        _step3_paths.validate_run_id(slug)
    except ValueError as e:
        msg = str(e).replace("run_id ", "Run handle ")
        return False, msg
    slug_lower = slug.lower()
    if slug_lower in {r.lower() for r in _RESERVED_SLUGS}:
        return False, f"'{slug}' is reserved (the implicit default run)"
    existing_lower = {s.lower() for s in existing_slugs}
    if slug_lower in existing_lower:
        return False, f"'{slug}' already exists in this workspace"
    return True, "Available"


def suggest_alternate_slug(
    slug: str,
    existing_slugs: Iterable[str],
) -> str:
    """If ``slug`` collides, suggest a numeric-suffix variant that doesn't.

    e.g. ``"nuclei_v1"`` → ``"nuclei_v2"``.  For slugs not ending in a
    digit, appends ``"_2"``.  Returns the input unchanged if it has no
    collision.
    """
    existing = set(existing_slugs)
    if slug not in existing:
        return slug

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

    e.g. ``"nuclei_v1"`` → ``"Nuclei V1"``; ``"membrane-test"`` →
    ``"Membrane Test"``.
    """
    parts = slug.replace("_", " ").replace("-", " ").split()
    return " ".join(p.capitalize() for p in parts)


def build_init_spec_for_mode(
    mode: str,
    *,
    pretrained_name: Optional[str] = None,
    step2_run_id: Optional[str] = None,
    step3_bundle: Optional[str] = None,
    step3_run_id: Optional[str] = None,
    browse_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the ``created_from`` init-spec for ``mode``.

    Pure dispatch onto the ``step3_run_manifest`` builders so the dialog's
    submit path and the tests share ONE source of truth.

    ``step3_run_id`` (Phase 5 audit HIGH-1B) disambiguates a
    ``step3_refined`` bundle's owning run so resolution pins to exactly
    ``(run_id, bundle)`` instead of a name-scan.  ``None`` / the flat default
    leaves the key off the spec.

    Raises:
        ValueError: unknown ``mode`` or a builder rejects its args
            (missing name / id / path).
    """
    if mode == _MODE_PRETRAINED:
        return init_spec_pretrained(pretrained_name=pretrained_name or "")
    if mode == _MODE_STEP2_RUN:
        return init_spec_step2_run(source_step2_run_id=step2_run_id or "")
    if mode == _MODE_STEP3_REFINED:
        return init_spec_step3_refined(
            source_step3_bundle=step3_bundle or "",
            source_step3_run_id=step3_run_id or None,
        )
    if mode == _MODE_BROWSE:
        return init_spec_browse(external_path=browse_path or "")
    raise ValueError(f"unknown init mode {mode!r}")


# ----------------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------------

class NewRunDialog(QDialog):
    """Themed dialog for creating a new Step 3 run.

    Usage::

        dlg = NewRunDialog(
            existing_slugs=existing,
            step2_run_sources=step2_sources,
            step3_refined_sources=step3_sources,
            parent=self.ui,
        )
        if dlg.exec_():
            result = dlg.get_result()
            controller._create_new_step3_run(**result)

    Result dict shape::

        {
            "slug": str,
            "display_name": str,
            "target": str,
            "description": str,
            "backend": str,            # "cellpose" (only enabled today)
            "init_spec": dict,         # from step3_run_manifest builders
            "notes": str,
            "activate": bool,
        }

    Source-list dict shapes (each list optional — an empty list greys the
    corresponding init mode):

    * ``step2_run_sources``  — ``{"run_id", "display_name"?, "target"?,
      "mtime_iso"?}``
    * ``step3_refined_sources`` — ``{"bundle_name", "run_id"?, "bucket"?,
      "mtime_iso"?}`` (one per bundle/bucket; the dialog collapses these to
      one row per owning-run + bundle — the ``run_id`` disambiguates a
      same-named bundle across runs, and the bucket is inert)
    * ``pretrained_names`` — ``list[str]`` (Cellpose built-in names)
    """

    def __init__(
        self,
        existing_slugs: Optional[Iterable[str]] = None,
        pretrained_names: Optional[List[str]] = None,
        step2_run_sources: Optional[List[Dict]] = None,
        step3_refined_sources: Optional[List[Dict]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._existing_slugs: Set[str] = set(existing_slugs or [])
        self._pretrained_names = list(pretrained_names or _CELLPOSE_PRETRAINED)
        self._step2_run_sources = list(step2_run_sources or [])
        self._step3_refined_sources = list(step3_refined_sources or [])
        self._browsed_external_path: str = ""
        self._result: Optional[dict] = None

        self.setWindowTitle("Create New Step 3 Run")
        from napari_epicycle.ui_shell.theme import (
            DIALOG_FORM_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        self._dialog_style = DIALOG_FORM_STYLE
        self._cancel_style = _DIALOG_BTN_CANCEL
        self._confirm_style = _DIALOG_BTN_CONFIRM

        self.setStyleSheet(self._dialog_style)
        self.setMinimumWidth(560)
        self.setMinimumHeight(600)

        self._build_ui()
        self._wire_signals()
        self._select_default_mode()
        self._update_validation()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        title = QLabel("Create New Step 3 Run", parent=self)
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #eeeeee;")
        root.addWidget(title)

        subtitle = QLabel(
            "Each run reconstructs + curates the workspace's patches with "
            "its own model.  Use a new run to try a different model "
            "(e.g. cells vs nuclei) without disturbing existing runs.",
            parent=self,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        root.addWidget(subtitle)
        root.addSpacing(4)

        # ===== § Identity =====
        identity_group = QGroupBox("Identity", parent=self)
        identity_form = QFormLayout(identity_group)
        identity_form.setContentsMargins(8, 14, 8, 8)
        identity_form.setSpacing(6)

        self._slug_edit = QLineEdit(parent=identity_group)
        self._slug_edit.setPlaceholderText("cells_v1")
        self._slug_edit.setMaxLength(_step3_paths.RUN_ID_MAX_LENGTH)
        self._slug_edit.setToolTip(
            "Short identifier used as the run's folder name on disk.  "
            "Letters, digits, underscores, and dashes only.  Distinct from "
            "'Display name' which can be any text."
        )
        identity_form.addRow("Run handle:", self._slug_edit)

        self._slug_status = QLabel("", parent=identity_group)
        self._slug_status.setStyleSheet("font-size: 10px;")
        self._slug_status.setMinimumHeight(14)
        identity_form.addRow("", self._slug_status)

        self._slug_suggestion = QPushButton("", parent=identity_group)
        self._slug_suggestion.setFlat(True)
        self._slug_suggestion.setStyleSheet(
            "QPushButton { color: #6abf69; font-size: 10px; "
            "text-align: left; padding: 0; border: none; }"
            "QPushButton:hover { color: #8cdb8c; text-decoration: underline; }"
        )
        self._slug_suggestion.setVisible(False)
        identity_form.addRow("", self._slug_suggestion)

        self._display_name_edit = QLineEdit(parent=identity_group)
        self._display_name_edit.setPlaceholderText("Auto-suggested from run handle")
        identity_form.addRow("Display name:", self._display_name_edit)

        # Target (combo + free text)
        target_widget = QWidget(parent=identity_group)
        target_widget.setStyleSheet("background: transparent;")
        target_row = QHBoxLayout(target_widget)
        target_row.setContentsMargins(0, 0, 0, 0)
        target_row.setSpacing(6)
        self._target_combo = QComboBox(parent=target_widget)
        self._target_combo.addItems(_TARGET_CHOICES)
        self._target_combo.setEditable(True)
        target_row.addWidget(self._target_combo, 1)
        identity_form.addRow("Target:", target_widget)

        self._description_edit = QTextEdit(parent=identity_group)
        self._description_edit.setMaximumHeight(60)
        self._description_edit.setPlaceholderText(
            "Optional notes about this run's purpose."
        )
        identity_form.addRow("Description:", self._description_edit)

        # Backend (radio) — only Cellpose enabled today
        backend_widget = QWidget(parent=identity_group)
        backend_widget.setStyleSheet("background: transparent;")
        backend_row = QHBoxLayout(backend_widget)
        backend_row.setContentsMargins(0, 0, 0, 0)
        backend_row.setSpacing(12)
        from packages.epicycle_io.step3_run_manifest import ENABLED_BACKENDS
        self._backend_group = QButtonGroup(backend_widget)
        self._backend_radios = {}
        for label, value in [
            ("Cellpose", "cellpose"),
            ("StarDist", "stardist"),
            ("Omnipose", "omnipose"),
        ]:
            rb = QRadioButton(label, parent=backend_widget)
            rb.setEnabled(value in ENABLED_BACKENDS)
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

        # ===== § Initialization (4 modes — NO scratch) =====
        init_widget = QGroupBox("Initialization", parent=self)
        init_layout = QVBoxLayout(init_widget)
        init_layout.setContentsMargins(8, 14, 8, 8)
        init_layout.setSpacing(8)

        mode_widget = QWidget(parent=init_widget)
        mode_grid = QHBoxLayout(mode_widget)
        mode_grid.setContentsMargins(0, 0, 0, 0)
        mode_grid.setSpacing(8)

        self._init_mode_group = QButtonGroup(mode_widget)
        self._init_pretrained_rb = QRadioButton("⊕ Pretrained", parent=mode_widget)
        self._init_pretrained_rb.setToolTip(
            "Init from a Cellpose built-in model (cyto3, nuclei, etc.)."
        )
        self._init_step2_rb = QRadioButton("↻ Step 2 run", parent=mode_widget)
        self._init_step2_rb.setToolTip(
            "Init from a Step 2 trained run-model in this workspace."
        )
        self._init_step3_rb = QRadioButton("⚙ Step 3 refined", parent=mode_widget)
        self._init_step3_rb.setToolTip(
            "Init from a Step 3 refinement bundle in this workspace."
        )
        self._init_browse_rb = QRadioButton("📁 Browse", parent=mode_widget)
        self._init_browse_rb.setToolTip(
            "Init from any weights file on disk."
        )
        # Order: pretrained, step2_run, step3_refined, browse
        for rb in (self._init_pretrained_rb, self._init_step2_rb,
                   self._init_step3_rb, self._init_browse_rb):
            self._init_mode_group.addButton(rb)
            mode_grid.addWidget(rb)
        mode_grid.addStretch(1)

        # Disable empty-source modes (pretrained + browse are always enabled)
        if not self._step2_run_sources:
            self._init_step2_rb.setEnabled(False)
            self._init_step2_rb.setToolTip(
                "No Step 2 trained run-models in this workspace yet."
            )
        if not self._step3_refined_sources:
            self._init_step3_rb.setEnabled(False)
            self._init_step3_rb.setToolTip(
                "No Step 3 refinement bundles in this workspace yet."
            )
        init_layout.addWidget(mode_widget)

        # Stacked sub-forms (index order matches the radio order)
        self._init_stack = QStackedWidget(parent=init_widget)

        # Page 0: Pretrained combo
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
        self._pretrained_combo.addItems(self._pretrained_names)
        pretrained_v.addWidget(self._pretrained_combo)
        pretrained_v.addStretch(1)
        self._init_stack.addWidget(pretrained_page)

        # Page 1: Step 2 run — table of (run, target)
        step2_page = QWidget(parent=self._init_stack)
        step2_page.setStyleSheet("background: transparent;")
        step2_v = QVBoxLayout(step2_page)
        step2_v.setContentsMargins(0, 0, 0, 0)
        step2_v.setSpacing(4)
        step2_hint = QLabel(
            "Pick a Step 2 trained run-model.  The run's model is copied "
            "into this new Step 3 run (copy-on-reuse).",
            parent=step2_page,
        )
        step2_hint.setWordWrap(True)
        step2_hint.setStyleSheet("color: #aaa; font-size: 10px;")
        step2_v.addWidget(step2_hint)
        self._step2_table = QTableWidget(parent=step2_page)
        self._step2_table.setColumnCount(3)
        self._step2_table.setHorizontalHeaderLabels(
            ["Run", "Target", "Last modified"]
        )
        self._configure_source_table(self._step2_table)
        self._populate_step2_table()
        step2_v.addWidget(self._step2_table)
        self._init_stack.addWidget(step2_page)

        # Page 2: Step 3 refined — table of (bundle, bucket)
        step3_page = QWidget(parent=self._init_stack)
        step3_page.setStyleSheet("background: transparent;")
        step3_v = QVBoxLayout(step3_page)
        step3_v.setContentsMargins(0, 0, 0, 0)
        step3_v.setSpacing(4)
        step3_hint = QLabel(
            "Pick a Step 3 refinement bundle.  Refined models tend to be "
            "more polished than raw HITL training output.",
            parent=step3_page,
        )
        step3_hint.setWordWrap(True)
        step3_hint.setStyleSheet("color: #aaa; font-size: 10px;")
        step3_v.addWidget(step3_hint)
        self._step3_table = QTableWidget(parent=step3_page)
        self._step3_table.setColumnCount(3)
        self._step3_table.setHorizontalHeaderLabels(
            ["Bundle", "Run", "Last modified"]
        )
        self._configure_source_table(self._step3_table)
        self._populate_step3_table()
        step3_v.addWidget(self._step3_table)
        self._init_stack.addWidget(step3_page)

        # Page 3: Browse external
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

        # Activate checkbox + button row
        self._activate_checkbox = QCheckBox(
            "Activate this run after creating it", parent=self
        )
        self._activate_checkbox.setChecked(True)
        self._activate_checkbox.setToolTip(
            "When checked, the new run becomes the active run immediately.  "
            "Uncheck to create the run but stay on the current one (useful "
            "for prepping several runs ahead of time)."
        )
        root.addWidget(self._activate_checkbox)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel", parent=self)
        self._cancel_btn.setStyleSheet(self._cancel_style)
        btn_row.addWidget(self._cancel_btn)
        self._create_btn = QPushButton("Create Run", parent=self)
        self._create_btn.setStyleSheet(self._confirm_style)
        self._create_btn.setDefault(True)
        btn_row.addWidget(self._create_btn)
        root.addLayout(btn_row)

    def _configure_source_table(self, table: QTableWidget):
        """Shared styling for the step2 / step3 source tables."""
        from napari_epicycle.ui_shell.theme import themed_table_style
        table.setMaximumHeight(140)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(themed_table_style("#d4a843"))
        hdr = table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, table.columnCount()):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)

    # ------------------------------------------------------------------
    # Signals + mode selection
    # ------------------------------------------------------------------

    def _wire_signals(self):
        self._display_name_user_edited = False
        self._slug_edit.textChanged.connect(self._on_slug_changed)
        self._display_name_edit.textEdited.connect(self._on_display_name_changed)
        self._target_combo.currentTextChanged.connect(
            lambda _t: self._update_preview()
        )
        self._slug_suggestion.clicked.connect(self._apply_suggestion)
        self._init_pretrained_rb.toggled.connect(
            lambda c: c and self._init_stack.setCurrentIndex(0)
        )
        self._init_step2_rb.toggled.connect(
            lambda c: c and self._init_stack.setCurrentIndex(1)
        )
        self._init_step3_rb.toggled.connect(
            lambda c: c and self._init_stack.setCurrentIndex(2)
        )
        self._init_browse_rb.toggled.connect(
            lambda c: c and self._init_stack.setCurrentIndex(3)
        )
        self._browse_btn.clicked.connect(self._on_browse_clicked)
        self._cancel_btn.clicked.connect(self.reject)
        self._create_btn.clicked.connect(self._submit)

    def _select_default_mode(self):
        """Check the first ENABLED init radio (pretrained is always enabled)."""
        for rb, idx in (
            (self._init_pretrained_rb, 0),
            (self._init_step2_rb, 1),
            (self._init_step3_rb, 2),
            (self._init_browse_rb, 3),
        ):
            if rb.isEnabled():
                rb.setChecked(True)
                self._init_stack.setCurrentIndex(idx)
                return

    def _populate_step2_table(self):
        sources = list(self._step2_run_sources)
        self._step2_table.setRowCount(len(sources))
        for row, src in enumerate(sources):
            display = src.get("display_name") or src.get("run_id", "")
            target = src.get("target") or "—"
            mtime = src.get("mtime_iso") or "-"
            items = [
                QTableWidgetItem(str(display)),
                QTableWidgetItem(str(target)),
                QTableWidgetItem(mtime[:10] if len(mtime) >= 10 else mtime),
            ]
            for col, it in enumerate(items):
                self._step2_table.setItem(row, col, it)
            items[0].setData(Qt.UserRole, src)
        if sources:
            self._step2_table.selectRow(0)

    def _populate_step3_table(self):
        # Phase 5 audit HIGH-1B: the cross-run source list yields one entry
        # per (bundle, BUCKET).  Collapse to ONE row per (owning-run, bundle)
        # — the bucket selection is inert (resolution returns the full
        # lateral+axial pair), and the owning RUN is what disambiguates a
        # same-named bundle across runs.  Each row stores both bundle_name +
        # run_id in UserRole so _submit can build the run-pinned init_spec.
        rows = self._group_step3_sources(self._step3_refined_sources)
        self._step3_table.setRowCount(len(rows))
        for row, src in enumerate(rows):
            mtime = src.get("mtime_iso") or "-"
            items = [
                QTableWidgetItem(str(src.get("bundle_name", ""))),
                QTableWidgetItem(self._format_run_label(src.get("run_id"))),
                QTableWidgetItem(mtime[:10] if len(mtime) >= 10 else mtime),
            ]
            for col, it in enumerate(items):
                self._step3_table.setItem(row, col, it)
            items[0].setData(Qt.UserRole, src)
        if rows:
            self._step3_table.selectRow(0)

    @staticmethod
    def _group_step3_sources(sources: Iterable[Dict]) -> List[Dict]:
        """Collapse per-(bundle, bucket) source entries to one per (run, bundle).

        Preserves first-seen order; keeps the latest ``mtime_iso`` across the
        collapsed buckets.  Pure (no Qt) so the submit path + tests can reuse
        it.  Each output dict: ``{bundle_name, run_id, mtime_iso}``.
        """
        grouped: Dict[tuple, Dict] = {}
        order: List[tuple] = []
        for src in sources or []:
            if not isinstance(src, dict):
                continue
            key = (src.get("run_id"), src.get("bundle_name", ""))
            mt = src.get("mtime_iso")
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = {
                    "bundle_name": src.get("bundle_name", ""),
                    "run_id": src.get("run_id"),
                    "mtime_iso": mt,
                }
                order.append(key)
            elif mt and (existing["mtime_iso"] is None or mt > existing["mtime_iso"]):
                existing["mtime_iso"] = mt
        return [grouped[k] for k in order]

    @staticmethod
    def _format_run_label(run_id: Optional[str]) -> str:
        """Display label for a bundle's owning run (flat default → '(default)')."""
        return str(run_id) if run_id else "(default)"

    def _on_browse_clicked(self):
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick weights file", "", "All Files (*.*)"
        )
        if not path:
            return
        self._browsed_external_path = path
        from pathlib import Path as _Path
        name = _Path(path).name or path
        self._browsed_label.setText(name)
        self._browsed_label.setToolTip(path)

    def _on_slug_changed(self, text: str):
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
        if not text:
            self._display_name_user_edited = False
        else:
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
            if "already exists" in message and slug:
                suggested = suggest_alternate_slug(slug, self._existing_slugs)
                self._slug_suggestion.setText(f"Try '{suggested}'")
                self._slug_suggestion.setVisible(True)
            else:
                self._slug_suggestion.setVisible(False)
        self._create_btn.setEnabled(is_valid)

    def _update_preview(self):
        slug = self._slug_edit.text().strip()
        if not slug:
            self._preview_label.setText("(run handle not yet entered)")
            return
        target_text = self._target_combo.currentText().strip()
        if target_text in ("(none)", "(custom)") or not target_text:
            target_text = "<unset>"
        self._preview_label.setText(
            f"artifacts/step3/runs/{slug}/\n"
            f"  ├─ run_manifest.json    (cellpose, target={target_text})\n"
            f"  ├─ qc/                  (empty until first extraction)\n"
            f"  └─ <patch>/             (one per reconstructed patch)"
        )

    def _get_selected_step2_source(self) -> Optional[Dict]:
        rows = self._step2_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self._step2_table.item(rows[0].row(), 0)
        return item.data(Qt.UserRole) if item is not None else None

    def _get_selected_step3_source(self) -> Optional[Dict]:
        rows = self._step3_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self._step3_table.item(rows[0].row(), 0)
        return item.data(Qt.UserRole) if item is not None else None

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def _submit(self):
        slug = self._slug_edit.text().strip()
        is_valid, _ = validate_slug_input(slug, self._existing_slugs)
        if not is_valid:
            return  # button should be disabled, but defensive

        display_name = self._display_name_edit.text().strip()
        if not display_name:
            display_name = derive_display_name_from_slug(slug)

        target_raw = self._target_combo.currentText().strip()
        target = "" if target_raw in ("(none)", "(custom)") else target_raw

        description = self._description_edit.toPlainText().strip()

        backend = "cellpose"
        for value, rb in self._backend_radios.items():
            if rb.isChecked():
                backend = value
                break

        from qtpy.QtWidgets import QMessageBox
        if self._init_step2_rb.isChecked():
            src = self._get_selected_step2_source()
            if src is None:
                QMessageBox.warning(
                    self, "No run selected",
                    "Picked 'Step 2 run' but no row selected in the table.\n"
                    "Please pick a row or switch to a different init mode."
                )
                return
            init_spec = build_init_spec_for_mode(
                _MODE_STEP2_RUN, step2_run_id=src["run_id"],
            )
        elif self._init_step3_rb.isChecked():
            src = self._get_selected_step3_source()
            if src is None:
                QMessageBox.warning(
                    self, "No bundle selected",
                    "Picked 'Step 3 refined' but no row selected in the "
                    "table.\nPlease pick a row or switch to a different "
                    "init mode."
                )
                return
            init_spec = build_init_spec_for_mode(
                _MODE_STEP3_REFINED,
                step3_bundle=src["bundle_name"],
                step3_run_id=src.get("run_id"),
            )
        elif self._init_browse_rb.isChecked():
            if not self._browsed_external_path:
                QMessageBox.warning(
                    self, "No file selected",
                    "Picked 'Browse' but no weights file selected.\n"
                    "Please click Browse... and pick a file, or switch "
                    "to a different init mode."
                )
                return
            init_spec = build_init_spec_for_mode(
                _MODE_BROWSE, browse_path=self._browsed_external_path,
            )
        else:
            pretrained = self._pretrained_combo.currentText().strip() or "cyto3"
            init_spec = build_init_spec_for_mode(
                _MODE_PRETRAINED, pretrained_name=pretrained,
            )

        self._result = {
            "slug": slug,
            "display_name": display_name,
            "target": target,
            "description": description,
            "backend": backend,
            "init_spec": init_spec,
            "notes": "",
            "activate": bool(self._activate_checkbox.isChecked()),
        }
        self.accept()

    def get_result(self) -> Optional[dict]:
        """Return the result dict from a successful submit, or None."""
        return self._result


__all__ = [
    "NewRunDialog",
    "validate_slug_input",
    "suggest_alternate_slug",
    "derive_display_name_from_slug",
    "build_init_spec_for_mode",
]

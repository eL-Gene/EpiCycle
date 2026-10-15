"""Themed Qt dialog for the Step 2 layout migration prompt.

Phase 2b of the Step 2 directory restructure.  Sibling to
``_migrate_step2_layout.py`` — that module owns the pure logic; this
one owns the user-facing prompt + execute trigger.

Auto-prompt flow:

    workspace load
         │
         ▼
    detect_layout_version
         │
         ├── NEW ────▶ proceed normally
         │
         ├── UNKNOWN ──▶ proceed normally
         │
         └── LEGACY / MIXED
              │
              ▼
         is_migration_skipped sentinel?  ─── Yes ──▶ proceed normally
              │
              No
              ▼
         prompt_migration ──▶ Cancel / Skip Session / Don't Ask Again / Migrate Now
                                  │           │              │              │
                                continue   in-memory    persist sentinel  backup + execute
                                            no-op                              │
                                                                              ▼
                                                                       success / failure dialog

The dialog is self-contained — caller passes in the workspace path
and a parent widget, gets back a :class:`UserChoice` enum.  Side
effects: optional sentinel write + (if user picks Migrate) calls
``create_backup`` + ``execute_migration`` from the layout module.

This module is Qt-dependent (uses ``qtpy``) and therefore NOT covered
by the pure-Python test suite at
``tests/hitl_training_annotation/test_migrate_step2_layout.py``.  The
underlying migration logic is fully unit-tested there.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from . import _migrate_step2_layout as mlogic


class UserChoice(str, Enum):
    """User's response to the migration prompt."""

    MIGRATE = "migrate"
    SKIP_SESSION = "skip_session"
    SKIP_FOREVER = "skip_forever"
    CANCEL = "cancel"  # set when hard_errors block migration


# ----------------------------------------------------------------------------
# Theme integration
# ----------------------------------------------------------------------------

def _dialog_styles():
    """Pull dialog stylesheets from theme.py at call time (avoid circular import)."""
    from napari_epicycle.ui_shell.theme import (
        _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
    )
    return _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM


# ----------------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------------

def prompt_and_migrate_if_needed(
    parent: Optional[QWidget],
    workspace: Path,
) -> Tuple[UserChoice, Optional[mlogic.ExecutionResult]]:
    """One-call entry point: detect, prompt, execute if user agrees.

    Returns ``(choice, exec_result)`` tuple:

    * ``choice`` is the user's response (or auto-determined if no
      migration was needed).
    * ``exec_result`` is the migration execution result if user picked
      MIGRATE; ``None`` otherwise.

    Side effects:
        * Persists the skip-sentinel if user picked SKIP_FOREVER
        * Creates backup + executes migration if user picked MIGRATE
        * Shows success/failure dialog after execute
    """
    workspace = Path(workspace)
    print(f"[STEP2-MIGRATE] prompt_and_migrate_if_needed: workspace={workspace}")

    # Layout detection
    from . import _paths as p
    layout = p.detect_layout_version(workspace)
    if layout in (p.LAYOUT_VERSION_NEW, p.LAYOUT_VERSION_UNKNOWN):
        print(f"[STEP2-MIGRATE] no prompt needed (layout={layout})")
        return UserChoice.SKIP_SESSION, None

    # Sentinel check — user has previously opted out
    if mlogic.is_migration_skipped(workspace):
        print("[STEP2-MIGRATE] skipped (sentinel file present)")
        return UserChoice.SKIP_FOREVER, None

    # Build plan + prompt
    plan = mlogic.plan_migration(workspace)
    if plan.is_empty:
        print("[STEP2-MIGRATE] empty plan, no prompt")
        return UserChoice.SKIP_SESSION, None

    print(f"[STEP2-MIGRATE] showing migration dialog "
          f"({plan.total_operations} planned operations)")
    choice = _show_dialog(parent, workspace, plan)
    print(f"[STEP2-MIGRATE] user chose: {choice.value}")

    # Execute if requested
    if choice != UserChoice.MIGRATE:
        return choice, None

    # Backup before execute
    print("[STEP2-MIGRATE] user picked Migrate; creating backup + executing")
    backup = mlogic.create_backup(workspace)
    result = mlogic.execute_migration(plan, backup)
    _show_post_execute_dialog(parent, result)
    return choice, result


# ----------------------------------------------------------------------------
# Dialog construction (internal)
# ----------------------------------------------------------------------------

def prompt_migration(
    parent: Optional[QWidget],
    workspace: Path,
) -> UserChoice:
    """Show the migration prompt and return the user's choice.

    Does NOT execute the migration — the caller invokes
    ``mlogic.create_backup`` + ``mlogic.execute_migration`` itself.
    Use :func:`prompt_and_migrate_if_needed` for the one-call flow.
    """
    plan = mlogic.plan_migration(workspace)
    if plan.is_empty:
        return UserChoice.SKIP_SESSION
    return _show_dialog(parent, workspace, plan)


def _show_dialog(
    parent: Optional[QWidget],
    workspace: Path,
    plan: mlogic.MigrationPlan,
) -> UserChoice:
    """Build + show the themed dialog.  Internal."""
    dialog_style, btn_cancel_style, btn_confirm_style = _dialog_styles()

    dlg = QDialog(parent)
    dlg.setWindowTitle("Step 2 Workspace Layout Migration")
    dlg.setStyleSheet(dialog_style)
    dlg.setMinimumWidth(680)
    dlg.setMinimumHeight(500)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(12)

    # Title
    title_lbl = QLabel("Step 2 Workspace Layout Migration")
    title_lbl.setStyleSheet(
        "font-weight: bold; font-size: 14px; color: #eeeeee;"
    )
    layout.addWidget(title_lbl)

    # Body explanation
    body_lbl = QLabel(
        "This workspace is using the legacy Step 2 (HITL 2D) directory "
        "layout.  EpiCycle now consolidates Step 2 artifacts under "
        "<code>artifacts/step2/runs/{run_id}/{lateral,axial}/</code> "
        "to support multiple concurrent training runs (nuclei, membrane, etc.) "
        "and to align with Step 3's organization.<br/><br/>"
        "<b>What will happen:</b> Files are <i>moved</i> (not copied) "
        "to their new locations.  A full backup is saved at "
        "<code>.pre_step2_restructure_&lt;timestamp&gt;/</code> in your "
        "workspace root before any moves — you can roll back at any "
        "time by restoring from that directory.<br/><br/>"
        "Existing training data lands in a <code>default</code> run. "
        "You can create additional runs later via the Step 2 UI."
    )
    body_lbl.setWordWrap(True)
    body_lbl.setTextFormat(Qt.RichText)
    body_lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
    layout.addWidget(body_lbl)

    # Plan summary (one-liner)
    summary_lbl = QLabel(_format_plan_summary(plan))
    summary_lbl.setStyleSheet(
        "color: #d8b478; font-size: 11px; font-weight: bold;"
    )
    layout.addWidget(summary_lbl)

    # Hard errors box (red, blocks migration if present)
    if plan.has_hard_errors:
        errors_frame = _build_errors_box(plan.hard_errors)
        layout.addWidget(errors_frame)

    # Warnings box (yellow, doesn't block)
    if plan.warnings:
        warn_frame = _build_warnings_box(plan.warnings)
        layout.addWidget(warn_frame)

    # Plan details (scrollable)
    details_widget = _build_plan_details(plan)
    details_scroll = QScrollArea()
    details_scroll.setWidget(details_widget)
    details_scroll.setWidgetResizable(True)
    details_scroll.setMinimumHeight(220)
    details_scroll.setStyleSheet(
        "QScrollArea { background-color: #1e1e1e; border: 1px solid #444; }"
    )
    layout.addWidget(details_scroll, 1)

    # Buttons
    result: list = [UserChoice.SKIP_SESSION]

    def _pick(choice):
        def _h():
            result[0] = choice
            dlg.accept()
        return _h

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)

    if plan.has_hard_errors:
        # Hard errors block migration — only Cancel button available
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(btn_cancel_style)
        cancel_btn.setToolTip(
            "Migration is blocked by hard errors above.  Resolve them "
            "and try again."
        )
        cancel_btn.clicked.connect(_pick(UserChoice.CANCEL))
        cancel_btn.setDefault(True)
        btn_row.addWidget(cancel_btn)
    else:
        # Normal three-choice flow
        skip_forever_btn = QPushButton("Don't Ask Again")
        skip_forever_btn.setStyleSheet(btn_cancel_style)
        skip_forever_btn.setToolTip(
            "Skip migration for this workspace permanently.  To re-enable "
            f"the prompt later, delete the file "
            f"`{mlogic.MIGRATION_SKIPPED_SENTINEL}` from the workspace root."
        )
        skip_forever_btn.clicked.connect(_pick(UserChoice.SKIP_FOREVER))

        skip_session_btn = QPushButton("Skip For This Session")
        skip_session_btn.setStyleSheet(btn_cancel_style)
        skip_session_btn.setToolTip(
            "Don't migrate now.  You'll be prompted again next time the "
            "workspace is opened."
        )
        skip_session_btn.clicked.connect(_pick(UserChoice.SKIP_SESSION))

        migrate_btn = QPushButton("Migrate Now")
        migrate_btn.setStyleSheet(btn_confirm_style)
        migrate_btn.setToolTip(
            "Back up the workspace and execute the migration plan above."
        )
        migrate_btn.setDefault(True)
        migrate_btn.clicked.connect(_pick(UserChoice.MIGRATE))

        btn_row.addWidget(skip_forever_btn)
        btn_row.addWidget(skip_session_btn)
        btn_row.addWidget(migrate_btn)

    layout.addLayout(btn_row)

    dlg.exec_()

    # Persist the skip-forever sentinel if chosen
    if result[0] == UserChoice.SKIP_FOREVER:
        mlogic.mark_migration_skipped(workspace)

    return result[0]


# ----------------------------------------------------------------------------
# Post-execute dialog (success / failure / auto-rolled-back)
# ----------------------------------------------------------------------------

def _show_post_execute_dialog(
    parent: Optional[QWidget],
    result: mlogic.ExecutionResult,
) -> None:
    """Themed message-box-style dialog summarizing migration outcome."""
    if result.success:
        msg = QMessageBox(parent)
        msg.setWindowTitle("Step 2 Migration Complete")
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            f"Migration completed successfully.\n\n"
            f"{len(result.completed_steps)} operations applied.\n"
            f"Backup saved at: {result.backup_dir}\n\n"
            "You may delete the backup directory once you've verified "
            "everything works as expected."
        )
        msg.exec_()
        return

    # Failure path
    msg = QMessageBox(parent)
    msg.setWindowTitle("Step 2 Migration Failed")
    msg.setIcon(QMessageBox.Critical)

    text = (
        f"Migration FAILED.\n\n"
        f"Error: {result.error or '(unknown)'}\n"
        f"Failed at step: {result.failed_step or '(unknown)'}\n\n"
    )
    if result.auto_rolled_back:
        text += (
            "Auto-rollback SUCCEEDED — your workspace has been restored "
            "to its pre-migration state from the backup.\n\n"
            "Please report this failure with the error message above."
        )
    else:
        text += (
            "Auto-rollback was NOT triggered or also failed.  Your "
            f"workspace may be in a partial state.  Backup is at:\n"
            f"  {result.backup_dir}\n\n"
            "Restore manually if needed."
        )
    msg.setText(text)
    msg.exec_()


# ----------------------------------------------------------------------------
# Plan rendering helpers
# ----------------------------------------------------------------------------

def _format_plan_summary(plan: mlogic.MigrationPlan) -> str:
    """One-line summary of the plan for the dialog header."""
    parts = []
    if plan.moves:
        parts.append(f"{len(plan.moves)} file move(s)")
    if plan.splits:
        parts.append(f"{len(plan.splits)} metadata split(s)")
    if plan.state_wrap:
        parts.append("1 state-JSON wrap (v2 → v3)")
    if plan.run_manifest_init:
        parts.append("1 run-manifest init (default run)")
    if plan.legacy_deletes:
        parts.append(f"{len(plan.legacy_deletes)} legacy delete(s)")
    if not parts:
        return "No operations planned."
    return "Plan: " + " + ".join(parts) + "."


def _build_warnings_box(warnings: list) -> QFrame:
    """Yellow-styled warnings panel.  Non-blocking."""
    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setStyleSheet(
        "QFrame { "
        "background-color: rgba(218, 165, 32, 0.10); "
        "border: 1px solid rgba(218, 165, 32, 0.50); "
        "border-radius: 4px; "
        "padding: 6px; "
        "}"
    )
    inner = QVBoxLayout(frame)
    inner.setContentsMargins(8, 6, 8, 6)
    inner.setSpacing(4)
    header = QLabel(f"Warnings ({len(warnings)}):")
    header.setStyleSheet("color: #e0c87a; font-weight: bold; font-size: 11px;")
    inner.addWidget(header)
    for w in warnings:
        lbl = QLabel(f"  • {w}")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #d8b478; font-size: 11px;")
        inner.addWidget(lbl)
    return frame


def _build_errors_box(errors: list) -> QFrame:
    """Red-styled hard-errors panel.  Blocks migration."""
    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setStyleSheet(
        "QFrame { "
        "background-color: rgba(220, 60, 60, 0.10); "
        "border: 1px solid rgba(220, 60, 60, 0.60); "
        "border-radius: 4px; "
        "padding: 6px; "
        "}"
    )
    inner = QVBoxLayout(frame)
    inner.setContentsMargins(8, 6, 8, 6)
    inner.setSpacing(4)
    header = QLabel(
        f"Hard errors ({len(errors)}) — migration is BLOCKED until resolved:"
    )
    header.setStyleSheet("color: #ff8080; font-weight: bold; font-size: 11px;")
    inner.addWidget(header)
    for e in errors:
        lbl = QLabel(f"  • {e}")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #ffaaaa; font-size: 11px;")
        inner.addWidget(lbl)
    return frame


def _build_plan_details(plan: mlogic.MigrationPlan) -> QWidget:
    """Scrollable enumeration of every operation in the plan."""
    body = QWidget()
    body.setStyleSheet("background-color: #1e1e1e;")
    vbox = QVBoxLayout(body)
    vbox.setContentsMargins(8, 8, 8, 8)
    vbox.setSpacing(2)

    if plan.moves:
        vbox.addWidget(_section_header(f"File moves ({len(plan.moves)})"))
        for mv in plan.moves:
            vbox.addWidget(_format_move_line(mv))

    if plan.splits:
        vbox.addWidget(_section_header(f"Metadata splits ({len(plan.splits)})"))
        for sp in plan.splits:
            vbox.addWidget(_format_split_line(sp))

    if plan.state_wrap:
        vbox.addWidget(_section_header("State JSON"))
        info = QLabel(
            f"  • Wrap & rename:  "
            f"<code>{plan.state_wrap.src.name}</code>  →  "
            f"<code>{plan.state_wrap.dst.name}</code> "
            "(schema v2 → v3, per-run state hoisted into runs.default)"
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        info.setStyleSheet("color: #c8c8c8; font-size: 10px;")
        vbox.addWidget(info)

    if plan.run_manifest_init:
        vbox.addWidget(_section_header("Run manifest"))
        info = QLabel(
            f"  • Initialize:  "
            f"<code>runs/{plan.run_manifest_init.run_id}/run_manifest.json</code> "
            "(backend=cellpose, target=None, display_name=\"Default\")"
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        info.setStyleSheet("color: #c8c8c8; font-size: 10px;")
        vbox.addWidget(info)

    if plan.legacy_deletes:
        vbox.addWidget(_section_header(
            f"Legacy deletes ({len(plan.legacy_deletes)})"
        ))
        for d in plan.legacy_deletes:
            lbl = QLabel(f"  • <code>{d.path.name}</code>  ({d.purpose})")
            lbl.setTextFormat(Qt.RichText)
            lbl.setStyleSheet("color: #c8c8c8; font-size: 10px;")
            lbl.setWordWrap(True)
            vbox.addWidget(lbl)

    vbox.addStretch(1)
    return body


def _section_header(text: str) -> QLabel:
    """Section sub-header for the plan details body."""
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "color: #d8b478; font-weight: bold; font-size: 11px; "
        "padding-top: 6px;"
    )
    return lbl


def _format_move_line(mv: mlogic.FileMove) -> QLabel:
    """One-line representation of a FileMove."""
    src_short = _short_path(mv.src)
    dst_short = _short_path(mv.dst)
    text = (
        f"  • <code>{src_short}</code>  →  "
        f"<code>{dst_short}</code> "
        f"<span style='color: #888;'>({mv.purpose})</span>"
    )
    lbl = QLabel(text)
    lbl.setTextFormat(Qt.RichText)
    lbl.setStyleSheet("color: #c8c8c8; font-size: 10px;")
    lbl.setWordWrap(True)
    return lbl


def _format_split_line(sp: mlogic.MetadataSplit) -> QLabel:
    """One-line representation of a MetadataSplit."""
    src_short = _short_path(sp.src)
    text = (
        f"  • <code>{src_short}</code>  →  per-bucket "
        f"<code>{sp.kind}.json</code> "
        f"<span style='color: #888;'>({sp.purpose})</span>"
    )
    lbl = QLabel(text)
    lbl.setTextFormat(Qt.RichText)
    lbl.setStyleSheet("color: #c8c8c8; font-size: 10px;")
    lbl.setWordWrap(True)
    return lbl


def _short_path(path: Path, max_segments: int = 6) -> str:
    """Truncate a path to the last `max_segments` components for display."""
    parts = path.parts
    if len(parts) <= max_segments:
        return str(path)
    return ".../" + "/".join(parts[-max_segments:])


# ----------------------------------------------------------------------------
# Public API surface
# ----------------------------------------------------------------------------

__all__ = [
    "UserChoice",
    "prompt_and_migrate_if_needed",
    "prompt_migration",
]

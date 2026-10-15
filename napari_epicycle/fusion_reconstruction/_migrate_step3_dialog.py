"""Themed Qt dialog for the Step 3 layout migration prompt.

Phase 2 of the directory restructure (2026-05-20).  Sibling to
``_migrate_step3_layout.py`` — that module owns the pure logic; this
one owns the user-facing prompt.

Auto-prompt flow:

  workspace load
       │
       ▼
  is_migration_needed?  ─── No ──▶ proceed normally
       │
      Yes
       │
       ▼
  is_migration_skipped?  ─── Yes ──▶ proceed normally
       │
       No
       ▼
  Show :func:`prompt_migration` ──▶ Migrate Now / Skip / Don't Ask
                                          │              │           │
                                       execute        in-memory    mark_migration_skipped
                                                       flag

The dialog is self-contained — caller passes in the workspace path
and a parent widget, gets back a :class:`UserChoice` enum.  No side
effects beyond the optional ``Don't ask again`` sentinel write.

This module is Qt-dependent (uses ``qtpy``) and therefore is NOT
covered by the pure-Python test suite at
``tests/fusion_reconstruction/test_migrate_step3_layout.py``.  The
underlying logic (plan_migration, execute_migration, etc.) is fully
unit-tested in the sibling module.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from pathlib import Path
from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from . import _migrate_step3_layout as mlogic


class UserChoice(str, Enum):
    """User's response to the migration prompt."""

    MIGRATE = "migrate"
    SKIP_SESSION = "skip_session"
    SKIP_FOREVER = "skip_forever"


# Borrow theme.py's dialog stylesheet constants.  We pull them at
# call time rather than at import to keep this module light + avoid
# circular imports during plugin discovery.
def _dialog_styles():
    from napari_epicycle.ui_shell.theme import (
        _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
    )
    return _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM


def prompt_migration(parent: Optional[QWidget],
                     workspace: Path) -> UserChoice:
    """Show the migration prompt and return the user's choice.

    Side effects:
        * If user picks ``SKIP_FOREVER``: writes the skip sentinel.
        * Otherwise: no filesystem changes.

    The CALLER is responsible for invoking
    :func:`_migrate_step3_layout.create_backup` +
    :func:`execute_migration` if the choice is ``MIGRATE``.  This
    function just gathers the user's intent.
    """
    plan = mlogic.plan_migration(workspace)
    if plan.is_empty:
        # Nothing to migrate — return SKIP_SESSION so callers can
        # uniformly treat "no action needed" the same as "user
        # deferred".
        return UserChoice.SKIP_SESSION

    return _show_dialog(parent, workspace, plan)


def _show_dialog(parent: Optional[QWidget],
                 workspace: Path,
                 plan: mlogic.MigrationPlan) -> UserChoice:
    """Build + show the themed dialog.  Internal."""
    dialog_style, btn_cancel_style, btn_confirm_style = _dialog_styles()

    dlg = QDialog(parent)
    dlg.setWindowTitle("Step 3 Workspace Layout Migration")
    dlg.setStyleSheet(dialog_style)
    dlg.setMinimumWidth(640)
    dlg.setMinimumHeight(480)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(12)

    # ── Title ──
    title_lbl = QLabel("Step 3 Workspace Layout Migration")
    title_lbl.setStyleSheet(
        "font-weight: bold; font-size: 14px; color: #eeeeee;"
    )
    layout.addWidget(title_lbl)

    # ── Body explanation ──
    body_lbl = QLabel(
        "This workspace is using the legacy Step 3 directory layout. "
        "EpiCycle now consolidates Step 3 artifacts under "
        "<code>artifacts/step3/</code> so that re-reconstruction, "
        "training, and curation can share one predictable structure.<br/><br/>"
        "<b>What will happen:</b> Files are <i>moved</i> (not copied) "
        "to their new locations.  A full backup is saved at "
        "<code>.legacy_pre_restructure_&lt;timestamp&gt;/</code> in your "
        "workspace before any file moves — you can roll back at any "
        "time by restoring from that directory."
    )
    body_lbl.setWordWrap(True)
    body_lbl.setTextFormat(Qt.RichText)
    body_lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
    layout.addWidget(body_lbl)

    # ── Plan summary ──
    summary_text = _format_plan_summary(plan)
    summary_lbl = QLabel(summary_text)
    summary_lbl.setStyleSheet(
        "color: #d8b478; font-size: 11px; font-weight: bold;"
    )
    layout.addWidget(summary_lbl)

    # ── Warnings (if any) ──
    if plan.warnings:
        warn_frame = _build_warnings_box(plan.warnings)
        layout.addWidget(warn_frame)

    # ── Plan details (scrollable) ──
    details_widget = _build_plan_details(plan)
    details_scroll = QScrollArea()
    details_scroll.setWidget(details_widget)
    details_scroll.setWidgetResizable(True)
    details_scroll.setMinimumHeight(220)
    details_scroll.setStyleSheet(
        "QScrollArea { background-color: #1e1e1e; border: 1px solid #444; }"
    )
    layout.addWidget(details_scroll, 1)  # stretch

    # ── Buttons ──
    result: list = [UserChoice.SKIP_SESSION]

    def _pick(choice):
        def _h():
            result[0] = choice
            dlg.accept()
        return _h

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)

    skip_forever_btn = QPushButton("Don't Ask Again")
    skip_forever_btn.setStyleSheet(btn_cancel_style)
    skip_forever_btn.setToolTip(
        "Skip migration for this workspace permanently.  To re-enable "
        "the prompt later, delete the file `.step3_migration_skipped` "
        "from the workspace root."
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

    # Persist the skip-forever sentinel if chosen.
    if result[0] == UserChoice.SKIP_FOREVER:
        mlogic.mark_migration_skipped(workspace)

    return result[0]


def _format_plan_summary(plan: mlogic.MigrationPlan) -> str:
    """One-line summary of the plan for the dialog header."""
    n_moves = len(plan.moves)
    n_lifts = len(plan.cellpose_lifts)
    state_rewrite = "1 state-JSON rewrite" if plan.legacy_state_path else None
    parts = []
    if n_moves:
        parts.append(f"{n_moves} file/directory move(s)")
    if n_lifts:
        parts.append(f"{n_lifts} Cellpose model lift(s)")
    if state_rewrite:
        parts.append(state_rewrite)
    if not parts:
        return "No operations planned."
    return "Plan: " + " + ".join(parts) + "."


def _build_warnings_box(warnings: list) -> QFrame:
    """Build a styled warnings panel."""
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
    header = QLabel("Warnings:")
    header.setStyleSheet(
        "color: #e0c87a; font-weight: bold; font-size: 11px;"
    )
    inner.addWidget(header)
    for w in warnings:
        lbl = QLabel(f"  • {w}")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #d8b478; font-size: 11px;")
        inner.addWidget(lbl)
    return frame


def _build_plan_details(plan: mlogic.MigrationPlan) -> QWidget:
    """Scrollable enumeration of every move + lift in the plan."""
    body = QWidget()
    body.setStyleSheet("background-color: #1e1e1e;")
    vbox = QVBoxLayout(body)
    vbox.setContentsMargins(8, 8, 8, 8)
    vbox.setSpacing(2)

    if plan.moves:
        vbox.addWidget(_section_header("File / directory moves"))
        for mv in plan.moves:
            vbox.addWidget(_format_move_line(mv))

    if plan.cellpose_lifts:
        vbox.addWidget(_section_header(
            "Cellpose weight lifts (post-move)"
        ))
        for lift in plan.cellpose_lifts:
            vbox.addWidget(_format_lift_line(lift))

    if plan.legacy_state_path:
        vbox.addWidget(_section_header("State JSON"))
        info = QLabel(
            f"  • Rewrite & rename:  "
            f"<code>{plan.legacy_state_path.name}</code>  →  "
            f"<code>step3_state.json</code> "
            "(layout_version=2, schema_version=2, paths rewritten, "
            "reconstruction_run_id dropped)"
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        info.setStyleSheet("color: #c8c8c8; font-size: 10px;")
        vbox.addWidget(info)

    vbox.addStretch(1)
    return body


def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "color: #d8b478; font-weight: bold; font-size: 11px; "
        "margin-top: 6px;"
    )
    return lbl


def _format_move_line(mv: mlogic.FileMove) -> QLabel:
    """One line per move — concise src → dst."""
    src_short = _shorten_path(mv.src, mv.dst)
    dst_short = _shorten_path(mv.dst, mv.src)
    text = (
        f"  • <span style='color: #d0d0d0;'>{mv.purpose}</span><br/>"
        f"    <code>{src_short}</code>  →  <code>{dst_short}</code>"
    )
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextFormat(Qt.RichText)
    lbl.setStyleSheet("color: #a8a8a8; font-size: 10px;")
    return lbl


def _format_lift_line(lift: mlogic.CellposeLift) -> QLabel:
    """One line per Cellpose lift."""
    text = (
        f"  • <span style='color: #d0d0d0;'>"
        f"{lift.bundle_name} / {lift.bucket}</span><br/>"
        f"    <code>{lift.src.name}</code>  →  bundle root "
        f"(if present after move)"
    )
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextFormat(Qt.RichText)
    lbl.setStyleSheet("color: #a8a8a8; font-size: 10px;")
    return lbl


def _shorten_path(path: Path, _ref: Path) -> str:
    """Trim a long path to its tail for compact display.  Returns
    the last 3 components joined by ``/`` — enough to disambiguate
    while staying readable."""
    parts = path.parts
    if len(parts) <= 3:
        return str(path)
    return ".../" + "/".join(parts[-3:])


# ---------------------------------------------------------------------------
# High-level convenience: prompt + execute (used by workspace-load wiring).
# ---------------------------------------------------------------------------


def prompt_and_migrate_if_needed(parent: Optional[QWidget],
                                 workspace: Path) -> bool:
    """Top-level entry point for auto-prompt-on-workspace-load.

    Returns:
        ``True``  if migration was needed AND completed successfully.
        ``False`` if no migration needed, user skipped, or migration
                  failed (caller may want to consult logs).

    Caller is expected to refresh its in-memory state ONLY when this
    returns ``True`` (the workspace path layout has changed).
    """
    workspace = Path(workspace)

    if not mlogic.is_migration_needed(workspace):
        return False

    if mlogic.is_migration_skipped(workspace):
        return False

    choice = prompt_migration(parent, workspace)
    if choice != UserChoice.MIGRATE:
        return False

    # Build a fresh plan (the plan from the prompt is recomputed to
    # capture any last-second filesystem state changes).
    plan = mlogic.plan_migration(workspace)
    if plan.is_empty:
        return False

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = mlogic.create_backup(workspace, timestamp=timestamp)
    result = mlogic.execute_migration(plan, backup_dir=backup_dir)

    if not result.success:
        # Show a follow-up dialog with the error + rollback offer.
        _show_failure_dialog(parent, result, workspace)
        return False

    _show_success_dialog(parent, result)
    return True


def _show_success_dialog(parent: Optional[QWidget],
                        result: mlogic.ExecutionResult) -> None:
    """Show a brief success confirmation."""
    dialog_style, _, btn_confirm_style = _dialog_styles()
    dlg = QDialog(parent)
    dlg.setWindowTitle("Migration Complete")
    dlg.setStyleSheet(dialog_style)
    dlg.setMinimumWidth(440)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(12)

    title = QLabel("Migration Complete")
    title.setStyleSheet(
        "font-weight: bold; font-size: 13px; color: #aee5a5;"
    )
    layout.addWidget(title)

    msg = QLabel(
        f"Successfully executed {len(result.completed_steps)} step(s).<br/>"
        f"Backup retained at: <code>{result.backup_dir}</code><br/><br/>"
        "You can delete the backup directory once you've confirmed the "
        "workspace is functioning normally."
    )
    msg.setWordWrap(True)
    msg.setTextFormat(Qt.RichText)
    msg.setStyleSheet("color: #cccccc; font-size: 11px;")
    layout.addWidget(msg)

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    ok_btn = QPushButton("OK")
    ok_btn.setStyleSheet(btn_confirm_style)
    ok_btn.setDefault(True)
    ok_btn.clicked.connect(dlg.accept)
    btn_row.addWidget(ok_btn)
    layout.addLayout(btn_row)

    dlg.exec_()


def _show_failure_dialog(parent: Optional[QWidget],
                        result: mlogic.ExecutionResult,
                        workspace: Path) -> None:
    """Show a failure dialog with the option to roll back from backup."""
    dialog_style, btn_cancel_style, btn_confirm_style = _dialog_styles()
    dlg = QDialog(parent)
    dlg.setWindowTitle("Migration Failed")
    dlg.setStyleSheet(dialog_style)
    dlg.setMinimumWidth(540)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(12)

    title = QLabel("Migration Failed")
    title.setStyleSheet(
        "font-weight: bold; font-size: 13px; color: #ef9a9a;"
    )
    layout.addWidget(title)

    if result.auto_rolled_back:
        body_text = (
            f"The migration could not be completed.<br/><br/>"
            f"<b>Failed at:</b> {result.failed_step or '(unknown)'}<br/>"
            f"<b>Error:</b> {result.error or '(unknown)'}<br/><br/>"
            f"<b>Steps completed before failure:</b> "
            f"{len(result.completed_steps)}<br/>"
            f"<b>Backup retained at:</b> "
            f"<code>{result.backup_dir}</code><br/><br/>"
            "<b>The workspace was automatically rolled back to its "
            "pre-migration state.</b>  You can investigate the error "
            "and retry the migration, or leave the workspace on the "
            "legacy layout for now."
        )
    else:
        body_text = (
            f"The migration could not be completed.<br/><br/>"
            f"<b>Failed at:</b> {result.failed_step or '(unknown)'}<br/>"
            f"<b>Error:</b> {result.error or '(unknown)'}<br/><br/>"
            f"<b>Steps completed before failure:</b> "
            f"{len(result.completed_steps)}<br/>"
            f"<b>Backup available at:</b> "
            f"<code>{result.backup_dir}</code><br/><br/>"
            "Choose <b>Roll Back</b> to restore the workspace from the "
            "backup, or <b>Leave As-Is</b> to keep the partially-migrated "
            "state for manual inspection."
        )

    msg = QLabel(body_text)
    msg.setWordWrap(True)
    msg.setTextFormat(Qt.RichText)
    msg.setStyleSheet("color: #cccccc; font-size: 11px;")
    layout.addWidget(msg)

    user_action: list = ["leave"]

    def _do_rollback():
        user_action[0] = "rollback"
        dlg.accept()

    def _do_leave():
        user_action[0] = "leave"
        dlg.accept()

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)

    leave_btn = QPushButton("OK" if result.auto_rolled_back else "Leave As-Is")
    leave_btn.setStyleSheet(btn_cancel_style)
    leave_btn.clicked.connect(_do_leave)

    rollback_btn = QPushButton("Roll Back")
    rollback_btn.setStyleSheet(btn_confirm_style)
    rollback_btn.setDefault(True)
    rollback_btn.setVisible(not result.auto_rolled_back)
    rollback_btn.clicked.connect(_do_rollback)

    btn_row.addWidget(leave_btn)
    btn_row.addWidget(rollback_btn)
    layout.addLayout(btn_row)

    dlg.exec_()

    if user_action[0] == "rollback" and result.backup_dir is not None:
        try:
            mlogic.rollback_migration(result.backup_dir, workspace)
        except Exception as e:
            # Best-effort — surface the rollback failure but don't
            # cascade exceptions out of the dialog.
            print(f"[STEP3-MIGRATE] Rollback failed: {e}")


__all__ = [
    "UserChoice",
    "prompt_migration",
    "prompt_and_migrate_if_needed",
]

"""Themed Qt dialog for the Step 2 split→flat layout migration prompt.

Phase C.3 of the flat-layout work.  Sibling to
``_migrate_split_to_flat.py`` — that module owns the pure logic; this
one owns the user-facing prompt + execute trigger.

Auto-prompt flow (called from controller's workspace-load hook):

    workspace load
         │
         ▼
    plan_split_to_flat_migration  ── empty plan ──▶ proceed normally
         │
         not empty / has work to do
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

This module is Qt-dependent (uses ``qtpy``) and therefore NOT covered
by the pure-Python test suite at
``tests/hitl_training_annotation/test_migrate_split_to_flat.py``.
The underlying migration logic is fully unit-tested there.
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

from . import _migrate_split_to_flat as mlogic


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
    """One-call entry point: plan, prompt, execute if user agrees.

    Returns ``(choice, exec_result)`` tuple:

    * ``choice`` is the user's response (or auto-determined if no
      migration was needed).
    * ``exec_result`` is the migration execution result if user picked
      MIGRATE; ``None`` otherwise.

    Side effects:
        * Persists the skip-sentinel if user picked SKIP_FOREVER
        * Creates backup + executes migration if user picked MIGRATE
        * Shows success/failure dialog after execute

    The PLAN is computed once at the top and reused for both the dialog
    display (operation count summary) AND the execute step (so the user
    sees exactly what runs).
    """
    workspace = Path(workspace)
    print(f"[STEP2-SPLIT-TO-FLAT] prompt_and_migrate_if_needed: workspace={workspace}")

    # Plan first — if nothing to do, skip silently
    try:
        plan = mlogic.plan_split_to_flat_migration(workspace)
    except Exception as e:
        print(f"[STEP2-SPLIT-TO-FLAT] plan_split_to_flat_migration raised "
              f"({type(e).__name__}: {e}); skipping prompt")
        return UserChoice.SKIP_SESSION, None

    if plan.is_empty:
        print(f"[STEP2-SPLIT-TO-FLAT] no migration needed "
              f"(total_operations={plan.total_operations})")
        return UserChoice.SKIP_SESSION, None

    if plan.has_hard_errors:
        # Show error dialog + skip — user must resolve manually
        _show_hard_error_dialog(parent, plan)
        return UserChoice.CANCEL, None

    # Sentinel check
    if mlogic.is_migration_skipped(workspace):
        print(f"[STEP2-SPLIT-TO-FLAT] user previously opted out "
              f"(sentinel present); skipping")
        return UserChoice.SKIP_SESSION, None

    # Prompt the user
    choice = _prompt_migration(parent, plan)
    print(f"[STEP2-SPLIT-TO-FLAT] user choice = {choice.value}")

    if choice == UserChoice.SKIP_SESSION or choice == UserChoice.CANCEL:
        return choice, None

    if choice == UserChoice.SKIP_FOREVER:
        try:
            mlogic.mark_migration_skipped(workspace)
        except Exception as e:
            # Brick D / journey #13 (2026-09-07): a "Don't ask again" that
            # could not be written used to fail silently — the prompt then
            # reappeared every session with no explanation.
            print(f"[STEP2-SPLIT-TO-FLAT] could not save the "
                  f"'Don't ask again' choice: {e}")
            _show_sentinel_write_failed(parent, e)
        return choice, None

    # MIGRATE: backup → execute → result dialog
    try:
        backup_dir = mlogic.create_backup(workspace)
        print(f"[STEP2-SPLIT-TO-FLAT] backup created at {backup_dir}")
    except Exception as e:
        _show_result_dialog(
            parent, success=False,
            error=f"Backup creation failed: {type(e).__name__}: {e}",
        )
        return choice, None

    result = mlogic.execute_split_to_flat_migration(
        plan, backup_dir, auto_rollback_on_failure=True,
    )
    _show_result_dialog(
        parent,
        success=result.success,
        error=result.error,
        completed_steps=result.completed_steps,
        auto_rolled_back=result.auto_rolled_back,
        backup_dir=result.backup_dir,
    )
    return choice, result


# ----------------------------------------------------------------------------
# Hard-error blocking dialog
# ----------------------------------------------------------------------------

def _show_sentinel_write_failed(parent, error):
    """Tell the user their "Don't ask again" could not be saved (#13).

    Best-effort: in a headless run (no QApplication) the console line
    already written is the whole story.
    """
    try:
        from qtpy.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is None:
            return
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Could not save your choice")
        box.setText(
            "EpiCycle could not record \"Don't ask again\" for this "
            "workspace."
        )
        box.setInformativeText(
            f"{type(error).__name__}: {error}\n\n"
            "Nothing was migrated and nothing was lost, but this prompt "
            "will appear again the next time you open this workspace."
        )
        box.setStandardButtons(QMessageBox.Ok)
        box.exec_() if hasattr(box, "exec_") else box.exec()
    except Exception as exc:  # pragma: no cover - dialog is best-effort
        print(f"[STEP2-SPLIT-TO-FLAT] sentinel-failure dialog unavailable: "
              f"{exc}")


def _show_hard_error_dialog(
    parent: Optional[QWidget], plan: mlogic.SplitToFlatPlan,
) -> None:
    """Show a blocking dialog explaining the hard errors and what to
    do.  No execute option — user must resolve manually."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Step 2 Migration — Cannot Proceed")
    box.setText(
        "The split→flat migration plan has errors that prevent automatic "
        "migration.\n\n"
        "Resolve the conflicts manually + reopen the workspace to retry."
    )
    box.setDetailedText("\n\n".join(plan.hard_errors))
    box.setStandardButtons(QMessageBox.Ok)
    try:
        _style, _, _ = _dialog_styles()
        box.setStyleSheet(_style)
    except Exception:
        pass
    box.exec_()


# ----------------------------------------------------------------------------
# Main prompt dialog
# ----------------------------------------------------------------------------

def _prompt_migration(
    parent: Optional[QWidget], plan: mlogic.SplitToFlatPlan,
) -> UserChoice:
    """Show the main 4-button prompt.  Returns user's choice."""
    dlg = _MigrationPromptDialog(parent, plan)
    dlg.exec_()
    return dlg.choice


class _MigrationPromptDialog(QDialog):
    """Themed prompt offering 4 choices: Cancel, Skip Session, Don't
    Ask Again, Migrate Now."""

    def __init__(self, parent: Optional[QWidget], plan: mlogic.SplitToFlatPlan):
        super().__init__(parent)
        self.plan = plan
        self.choice: UserChoice = UserChoice.CANCEL  # default if dialog is X'd
        self.setWindowTitle("Step 2 Layout Migration Available")
        self.setMinimumWidth(680)
        self.setModal(True)

        try:
            style, btn_cancel, btn_confirm = _dialog_styles()
            self.setStyleSheet(style)
        except Exception:
            style = btn_cancel = btn_confirm = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Header
        header = QLabel(
            "<b>This workspace can be upgraded to the new flat layout.</b>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Body — explain what migration does
        body_lines = [
            f"<p>Found <b>{len(plan.round_moves)}</b> training round(s) "
            f"using the legacy split layout (<code>bucket/rounds/round_NN/</code> "
            f"+ <code>bucket/models/round_NN/</code>).</p>",
            "<p>Migration will:</p>",
            "<ul>",
            "<li>Move round data into one dir per round "
            "(<code>bucket/round_NN/{accepted, predictions, model}/</code>)</li>",
            "<li>Rename trained-model files to include the round number "
            "(<code>{run}_{bucket}_round_NN</code>)</li>",
            "<li>Update schema 2.x manifests + state JSON to match</li>",
            "<li>Create a full backup at "
            "<code>{workspace}/.pre_step2_flat_layout_&lt;timestamp&gt;/</code> "
            "before any changes</li>",
            "<li>Auto-rollback to the backup if anything fails</li>",
            "</ul>",
            f"<p>Total operations: <b>{plan.total_operations}</b></p>",
        ]
        if plan.warnings:
            body_lines.append(
                f"<p><i>Warnings ({len(plan.warnings)} — non-blocking):</i></p>"
                "<ul>"
            )
            for w in plan.warnings[:5]:
                body_lines.append(f"<li>{w}</li>")
            if len(plan.warnings) > 5:
                body_lines.append(f"<li>... and {len(plan.warnings) - 5} more</li>")
            body_lines.append("</ul>")
        if plan.is_mixed_layout:
            body_lines.append(
                "<p><i>This workspace is partially migrated already. "
                "Re-running migration completes the remaining rounds.</i></p>"
            )

        body = QLabel("".join(body_lines))
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        layout.addWidget(body)

        # Footer help text
        footer = QLabel(
            "<i>You can run training without migrating — the dual-path "
            "readers will find your existing data.  But migrating once "
            "cleans up the directory structure permanently.</i>"
        )
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #888;")
        layout.addWidget(footer)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        b_skip_session = QPushButton("Skip This Session")
        b_skip_session.setStyleSheet(btn_cancel)
        b_skip_session.clicked.connect(
            lambda: self._set_choice_and_close(UserChoice.SKIP_SESSION)
        )
        btn_row.addWidget(b_skip_session)

        b_skip_forever = QPushButton("Don't Ask Again")
        b_skip_forever.setStyleSheet(btn_cancel)
        b_skip_forever.clicked.connect(
            lambda: self._set_choice_and_close(UserChoice.SKIP_FOREVER)
        )
        btn_row.addWidget(b_skip_forever)

        b_migrate = QPushButton("Migrate Now")
        b_migrate.setStyleSheet(btn_confirm)
        b_migrate.clicked.connect(
            lambda: self._set_choice_and_close(UserChoice.MIGRATE)
        )
        btn_row.addWidget(b_migrate)

        layout.addLayout(btn_row)

    def _set_choice_and_close(self, choice: UserChoice) -> None:
        self.choice = choice
        self.accept()


# ----------------------------------------------------------------------------
# Result dialog (post-execute)
# ----------------------------------------------------------------------------

def _show_result_dialog(
    parent: Optional[QWidget],
    *,
    success: bool,
    error: Optional[str] = None,
    completed_steps: Optional[list] = None,
    auto_rolled_back: bool = False,
    backup_dir: Optional[Path] = None,
) -> None:
    """Show a success or failure dialog after the migration executes."""
    box = QMessageBox(parent)
    if success:
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Step 2 Migration — Complete")
        n_steps = len(completed_steps) if completed_steps else 0
        box.setText(
            f"Migration complete.\n\n"
            f"Completed {n_steps} step(s) — your workspace is now on "
            f"the flat layout.\n\n"
            f"The backup at:\n  {backup_dir}\n"
            f"can be deleted once you've verified everything works."
        )
        if completed_steps:
            box.setDetailedText("\n".join(completed_steps))
    else:
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Step 2 Migration — Failed")
        rolled_back = (
            "\n\nThe workspace was automatically rolled back to its "
            "pre-migration state from the backup."
            if auto_rolled_back
            else "\n\nThe workspace may be in an inconsistent state — "
                 f"manually restore from the backup at:\n  {backup_dir}"
        )
        box.setText(
            f"Migration failed.\n\n"
            f"Error: {error or 'unknown'}"
            f"{rolled_back}"
        )
        if completed_steps:
            box.setDetailedText(
                "Completed before failure:\n" + "\n".join(completed_steps)
            )
    box.setStandardButtons(QMessageBox.Ok)
    try:
        _style, _, _ = _dialog_styles()
        box.setStyleSheet(_style)
    except Exception:
        pass
    box.exec_()

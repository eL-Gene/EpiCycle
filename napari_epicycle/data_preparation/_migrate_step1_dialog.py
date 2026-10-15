"""Themed Qt dialog for the Step 1 layout-migration auto-prompt.

Sibling to ``_migrate_step1_layout.py`` (which owns the pure logic). This
module owns the user-facing prompt + execute trigger, mirroring the Step 2
``_migrate_split_to_flat_dialog.py`` pattern.

Auto-prompt flow (called from the controller's workspace-load hook):

    workspace set → should_offer_migration? → prompt
        Skip This Session / Don't Ask Again / Migrate Now
                              │               │
                       mark_migration_skipped  execute_migration
                                               (backup + auto-rollback)
                                                     → result dialog

Qt-dependent → NOT covered by the pure-Python test suite. The underlying
migration logic + the ``should_offer_migration`` decision ARE unit-tested
in ``tests/data_preparation/test_migrate.py``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import _migrate_step1_layout as mlogic


class UserChoice(str, Enum):
    MIGRATE = "migrate"
    SKIP_SESSION = "skip_session"
    SKIP_FOREVER = "skip_forever"


def _dialog_styles():
    """Pull dialog stylesheets from theme.py at call time (avoid circular import)."""
    try:
        from napari_epicycle.ui_shell.theme import (
            _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        return _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM
    except Exception:
        return "", "", ""


def prompt_and_migrate_if_needed(
    parent: Optional[QWidget], workspace,
) -> Tuple[UserChoice, Optional[mlogic.ExecutionResult]]:
    """Plan → prompt → execute if the user agrees. Returns (choice, result).

    Side effects: persists the skip marker on SKIP_FOREVER; on MIGRATE runs
    ``execute_migration`` (which backs up + auto-rolls-back) and shows a
    success/failure dialog.
    """
    workspace = Path(workspace)
    try:
        plan = mlogic.plan_migration(workspace)
    except Exception as e:
        print(f"[STEP1-MIGRATE] plan_migration raised ({type(e).__name__}: {e}); skipping")
        return UserChoice.SKIP_SESSION, None

    if plan.is_empty or mlogic.is_migration_skipped(workspace):
        return UserChoice.SKIP_SESSION, None

    choice = _prompt_migration(parent, plan)
    print(f"[STEP1-MIGRATE] user choice = {choice.value}")

    if choice == UserChoice.SKIP_SESSION:
        return choice, None

    if choice == UserChoice.SKIP_FOREVER:
        try:
            mlogic.mark_migration_skipped(workspace)
        except OSError as e:
            print(f"[STEP1-MIGRATE] failed to persist skip marker: {e}")
        return choice, None

    # MIGRATE — execute_migration handles backup + auto-rollback internally.
    try:
        result = mlogic.execute_migration(plan)
    except Exception as e:
        _show_result_dialog(parent, success=False, error=f"{type(e).__name__}: {e}")
        return choice, None
    _show_result_dialog(parent, success=True, result=result)
    return choice, result


def _prompt_migration(parent, plan: mlogic.MigrationPlan) -> UserChoice:
    dlg = _MigrationPromptDialog(parent, plan)
    dlg.exec_()
    return dlg.choice


class _MigrationPromptDialog(QDialog):
    """Themed 3-choice prompt: Skip This Session / Don't Ask Again / Migrate Now."""

    def __init__(self, parent: Optional[QWidget], plan: mlogic.MigrationPlan):
        super().__init__(parent)
        self.plan = plan
        self.choice: UserChoice = UserChoice.SKIP_SESSION  # default if X'd
        self.setWindowTitle("Step 1 Layout Migration Available")
        self.setMinimumWidth(640)
        self.setModal(True)

        style, btn_cancel, btn_confirm = _dialog_styles()
        self.setStyleSheet(style)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        header = QLabel("<b>This workspace can be tidied into the new layout.</b>")
        header.setWordWrap(True)
        layout.addWidget(header)

        n = len(plan.patches)
        body_lines = [
            f"<p>Found <b>{n}</b> extracted patch(es) stored loosely at the "
            "workspace root.</p>",
            "<p>Migration will:</p>",
            "<ul>",
            "<li>Move each patch (TIFF + <code>_metadata.json</code>) into "
            "<code>artifacts/step1/patches/</code></li>",
            "<li>Build the <code>step1_state.json</code> registry from the sidecars</li>",
            "<li>Create a full backup first + auto-rollback if anything fails</li>",
            "</ul>",
        ]
        if plan.conflicts:
            body_lines.append(
                f"<p><i>{len(plan.conflicts)} item(s) will be skipped "
                "(name conflict or ambiguous) and left where they are.</i></p>"
            )
        body = QLabel("".join(body_lines))
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        layout.addWidget(body)

        footer = QLabel(
            "<i>This is optional — Steps 2 and 3 can still browse to your patches "
            "either way. Migrating once cleans up the workspace permanently.</i>"
        )
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #888;")
        layout.addWidget(footer)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        b_skip = QPushButton("Skip This Session")
        b_skip.setStyleSheet(btn_cancel)
        b_skip.clicked.connect(lambda: self._choose(UserChoice.SKIP_SESSION))
        btn_row.addWidget(b_skip)

        b_never = QPushButton("Don't Ask Again")
        b_never.setStyleSheet(btn_cancel)
        b_never.clicked.connect(lambda: self._choose(UserChoice.SKIP_FOREVER))
        btn_row.addWidget(b_never)

        b_migrate = QPushButton("Migrate Now")
        b_migrate.setStyleSheet(btn_confirm)
        b_migrate.clicked.connect(lambda: self._choose(UserChoice.MIGRATE))
        btn_row.addWidget(b_migrate)

        layout.addLayout(btn_row)

    def _choose(self, choice: UserChoice) -> None:
        self.choice = choice
        self.accept()


def _show_result_dialog(
    parent: Optional[QWidget], *, success: bool,
    result: Optional[mlogic.ExecutionResult] = None, error: Optional[str] = None,
) -> None:
    box = QMessageBox(parent)
    if success:
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Step 1 Migration — Complete")
        moved = result.moved if result else 0
        msg = (
            f"Migration complete.\n\n"
            f"Moved {moved} patch(es) into artifacts/step1/patches/."
        )
        skipped = (result.conflicts + result.skipped) if result else []
        if skipped:
            msg += (
                f"\n\n{len(skipped)} item(s) were skipped (conflict/ambiguous) "
                "and left at the workspace root for you to resolve manually."
            )
        if result and result.backup_dir:
            msg += f"\n\nA backup was kept at:\n  {result.backup_dir}"
        box.setText(msg)
    else:
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Step 1 Migration — Failed")
        box.setText(
            f"Migration failed and was automatically rolled back.\n\n"
            f"Error: {error or 'unknown'}\n\n"
            f"Your patches are unchanged at the workspace root."
        )
    box.setStandardButtons(QMessageBox.Ok)
    style, _, _ = _dialog_styles()
    box.setStyleSheet(style)
    box.exec_()

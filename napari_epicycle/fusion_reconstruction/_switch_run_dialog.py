"""Themed Qt dialogs for the Step 3 run-switch safety prompts (Phase 4a).

The Step 3 mirror of ``hitl_training_annotation/_switch_run_dialog.py``.
Two dialogs:

* :func:`show_switch_blocked` — modal error when a Step 3 operation is in
  flight (extraction / reconstruction / save / commit); user can't
  proceed, only "OK" to dismiss.
* :func:`show_switch_confirm` — modal warn dialog when the 3D curation
  annotator is open or there are unsaved curation edits; "Cancel" /
  "Switch + Reload".

Both use themed styles from ``ui_shell.theme`` for visual consistency
with the other Step 3 dialogs (refinement registry, merge dialog).

Terminology: user-facing strings prefer the ``display_name`` of each
run; the handle (a.k.a. ``run_id`` / ``slug`` internally) is surfaced as
a parenthetical when both are known.  When no display_name is provided,
the handle stands in alone.

The Qt-constructing functions are not unit-tested (Qt teardown);
:func:`_format_run_label` is a pure helper covered by
``tests/fusion_reconstruction/test_step3_switch_run_dialog.py``.
"""

from __future__ import annotations

from typing import List, Optional

from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _format_run_label(run_id: str, display_name: Optional[str]) -> str:
    """Format a run for user-facing display.

    Prefers ``"Display Name (handle: rid)"`` when both exist + differ.
    Falls back to the bare handle when no display_name is provided or it
    matches the handle.
    """
    rid = (run_id or "").strip()
    dn = (display_name or "").strip()
    if dn and dn != rid:
        return f"{dn} (handle: {rid})"
    return rid


def show_switch_blocked(
    parent: Optional[QWidget],
    reasons: List[str],
    old_run_id: str,
    new_run_id: str,
    old_display_name: Optional[str] = None,
    new_display_name: Optional[str] = None,
) -> None:
    """Show a hard-block dialog (operation in flight); no proceed option."""
    from napari_epicycle.ui_shell.theme import _DIALOG_STYLE, _DIALOG_BTN_CANCEL

    old_label = _format_run_label(old_run_id, old_display_name)
    new_label = _format_run_label(new_run_id, new_display_name)

    dlg = QDialog(parent)
    dlg.setWindowTitle("Cannot Switch Runs")
    dlg.setStyleSheet(_DIALOG_STYLE)
    dlg.setMinimumWidth(440)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(10)

    title = QLabel("Cannot switch runs right now", parent=dlg)
    title.setStyleSheet("font-weight: bold; font-size: 14px; color: #e57373;")
    layout.addWidget(title)

    subtitle = QLabel(
        f"Tried to switch from {old_label} to {new_label}.",
        parent=dlg,
    )
    subtitle.setStyleSheet("color: #bbbbbb; font-size: 11px;")
    subtitle.setWordWrap(True)
    layout.addWidget(subtitle)

    for reason in reasons:
        msg = QLabel(reason, parent=dlg)
        msg.setWordWrap(True)
        msg.setStyleSheet("color: #cccccc; font-size: 11px; padding: 4px 0;")
        layout.addWidget(msg)

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    ok_btn = QPushButton("OK", parent=dlg)
    ok_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
    ok_btn.clicked.connect(dlg.reject)
    btn_row.addWidget(ok_btn)
    layout.addLayout(btn_row)

    dlg.exec_()


def show_switch_confirm(
    parent: Optional[QWidget],
    warnings: List[str],
    old_run_id: str,
    new_run_id: str,
    old_display_name: Optional[str] = None,
    new_display_name: Optional[str] = None,
) -> bool:
    """Show a warn-with-confirm dialog; return True if user proceeds.

    Args:
        parent: parent widget.
        warnings: list of human-readable warning strings.
        old_run_id: current active run handle.
        new_run_id: target run handle.
        old_display_name: display name for the current run (optional —
            falls back to handle).
        new_display_name: display name for the target run (optional —
            falls back to handle).

    Returns:
        True if user clicked "Switch + Reload"; False if Cancel / closed.
    """
    from napari_epicycle.ui_shell.theme import (
        _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
    )

    old_label = _format_run_label(old_run_id, old_display_name)
    new_label = _format_run_label(new_run_id, new_display_name)

    dlg = QDialog(parent)
    dlg.setWindowTitle("Switch Run?")
    dlg.setStyleSheet(_DIALOG_STYLE)
    dlg.setMinimumWidth(460)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(10)

    title = QLabel(f"Switch {old_label} → {new_label}?", parent=dlg)
    title.setStyleSheet("font-weight: bold; font-size: 14px; color: #d4a843;")
    title.setWordWrap(True)
    layout.addWidget(title)

    for warning in warnings:
        msg = QLabel("⚠  " + warning, parent=dlg)
        msg.setWordWrap(True)
        msg.setStyleSheet("color: #dddddd; font-size: 11px; padding: 4px 0;")
        layout.addWidget(msg)

    info = QLabel(
        "Your saved curation remains on disk and will be there when you "
        f"switch back to {old_label}.  The 3D curation annotator will "
        f"close and auto-reopen on {new_label}'s last-used patch (if any).",
        parent=dlg,
    )
    info.setWordWrap(True)
    info.setStyleSheet(
        "color: #888888; font-size: 10px; font-style: italic; padding-top: 6px;"
    )
    layout.addWidget(info)

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    cancel_btn = QPushButton("Cancel", parent=dlg)
    cancel_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
    cancel_btn.clicked.connect(dlg.reject)
    btn_row.addWidget(cancel_btn)

    confirm_btn = QPushButton("Switch + Reload Annotator", parent=dlg)
    confirm_btn.setStyleSheet(_DIALOG_BTN_CONFIRM)
    confirm_btn.setDefault(True)
    confirm_btn.clicked.connect(dlg.accept)
    btn_row.addWidget(confirm_btn)
    layout.addLayout(btn_row)

    return dlg.exec_() == QDialog.Accepted


__all__ = [
    "_format_run_label",
    "show_switch_blocked",
    "show_switch_confirm",
]

"""Step 3 Manage Runs panel (Phase 4a — card list + per-row actions).

The Step 3 mirror of ``hitl_training_annotation/_manage_runs_panel.py``.
Themed Qt dialog showing all runs in the active workspace.  Per-row
actions: Switch / Edit metadata / Archive / Unarchive / Delete.

Data flow:

* ``_run_management.list_run_summaries(workspace)`` → cards (one per run)
* User picks a card, clicks an action button
* Action handler calls a controller-injected callback
  (``on_switch`` / ``on_edit`` / ``on_archive`` / ``on_unarchive`` /
  ``on_delete``)
* For Delete, runs ``find_dependents_callback`` first.  Step 3 uses
  copy-on-reuse for models (the source is COPIED into the new run), so a
  dependent does NOT actually need the source on disk — the dependency
  warning is therefore a SOFT warn ("Proceed anyway"), not a hard block
  like Step 2.

ADAPTATIONS from the Step 2 original:

* Cards show the Step-3 ``patch_counts`` (reconstructed / curated /
  accepted / finalized / total) rather than Step 2's lateral/axial round
  counts + accepted-pair count.
* **DEFAULT-RUN GUARD:** the implicit FLAT default run (``"default"``)
  has no manifest dir, so Archive / Delete / Edit are DISABLED for it.
* Dependency guard on Delete is a soft warn (see above), not a block.

The action-enablement decision is extracted into the pure
:func:`_action_enablement` helper so it is unit-testable without a Qt
display.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ._migrate_step3_runs import DEFAULT_RUN_ID


# Action enum for the result dict returned to the caller.
ACTION_NONE = "none"
ACTION_SWITCH = "switch"
ACTION_EDIT = "edit"
ACTION_ARCHIVE = "archive"
ACTION_UNARCHIVE = "unarchive"
ACTION_DELETE = "delete"


# ----------------------------------------------------------------------------
# Pure decision helpers (no Qt — unit-tested directly)
# ----------------------------------------------------------------------------

def _action_enablement(
    run_id: str,
    active_run_id: str,
    is_default: bool,
    is_archived: bool = False,
) -> Dict[str, bool]:
    """Decide which per-run actions are enabled for a given run.

    Pure function — drives the Manage Runs button states without a Qt
    display, and is the single source of truth the dialog delegates to.

    Rules (Step 3):

    * **Switch** — any run that is NOT the active run.
    * **Edit / Archive / Delete** — DISABLED for the FLAT default run
      (``run_id == "default"``): it is synthesized, has no manifest, and
      cannot be archived / deleted / edited.
    * **Edit** — any NAMED run.
    * **Archive** — a named, non-active, not-already-archived run (you
      must switch off a run before archiving it).
    * **Unarchive** — a named, currently-archived run.
    * **Delete** — a named, non-active run (force the user to switch off
      the run they're working on before deleting it).
    * **Clean intermediates** — ANY selected run, including the FLAT default
      run (which has its own ``prediction_cache/``) and the active run (the
      cache is regenerable via re-PREDICT).  This is the SIMPLER enablement
      path (design choice): rather than pre-scanning each run for cache on
      every selection, the button is always enabled and the controller
      handler no-ops with an info modal when there is nothing to clean.

    Returns:
        ``{"switch", "edit", "archive", "unarchive", "delete", "clean"}`` →
        bool.
    """
    is_active = (run_id == active_run_id)
    named = not is_default
    return {
        "switch": not is_active,
        "edit": named,
        "archive": named and (not is_active) and (not is_archived),
        "unarchive": named and is_archived,
        "delete": named and (not is_active),
        "clean": True,
    }


def _should_warn_dependents(dependents: Optional[List[str]]) -> bool:
    """True iff the delete flow should surface a dependency warning.

    Step 3's copy-on-reuse makes this a SOFT warn (not a block) — so the
    decision is simply "are there any dependents to mention?".
    """
    return bool(dependents)


# ----------------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------------

class ManageRunsPanel(QDialog):
    """Card-list panel for managing all Step 3 runs in a workspace."""

    def __init__(
        self,
        run_summaries: List[Dict],
        active_run_id: str,
        on_switch: Optional[Callable[[str], None]] = None,
        on_edit: Optional[Callable[[str, Dict], bool]] = None,
        on_archive: Optional[Callable[[str], bool]] = None,
        on_unarchive: Optional[Callable[[str], bool]] = None,
        on_delete: Optional[Callable[[str], bool]] = None,
        on_clean_intermediates: Optional[Callable[[str], bool]] = None,
        find_dependents_callback: Optional[Callable[[str], List[str]]] = None,
        on_convert_layout: Optional[Callable[[], bool]] = None,
        is_unified: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._summaries = list(run_summaries)
        self._active_run_id = active_run_id
        self._on_switch = on_switch
        self._on_edit = on_edit
        self._on_archive = on_archive
        self._on_unarchive = on_unarchive
        self._on_delete = on_delete
        self._on_clean_intermediates = on_clean_intermediates
        self._find_dependents = find_dependents_callback
        # Phase 7: workspace-level "Convert to unified layout" migration.
        self._on_convert_layout = on_convert_layout
        self._is_unified = bool(is_unified)
        self._summaries_provider: Optional[Callable[[], List[Dict]]] = None

        from napari_epicycle.ui_shell.theme import (
            DIALOG_FORM_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        self._cancel_style = _DIALOG_BTN_CANCEL
        self._confirm_style = _DIALOG_BTN_CONFIRM

        self.setWindowTitle("Manage Step 3 Runs")
        self.setStyleSheet(DIALOG_FORM_STYLE)
        self.setMinimumWidth(720)
        self.setMinimumHeight(440)

        self._build_ui()
        self._wire_signals()
        self._refresh_table()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        title = QLabel("Manage Step 3 Runs", parent=self)
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #eeeeee;")
        root.addWidget(title)

        subtitle = QLabel(
            "Edit, archive, or delete reconstruction/curation runs in this "
            "workspace.  Active runs are marked with ●.  Archived runs are "
            "hidden from the picker but preserved on disk.  The implicit "
            "default run cannot be edited, archived, or deleted.",
            parent=self,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        root.addWidget(subtitle)

        # Filter + search row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        filter_row.addWidget(QLabel("Filter:", parent=self))
        self._filter_combo = QComboBox(parent=self)
        self._filter_combo.addItems(["Active", "Archived", "All"])
        filter_row.addWidget(self._filter_combo)
        filter_row.addSpacing(12)
        filter_row.addWidget(QLabel("Search:", parent=self))
        self._search_edit = QLineEdit(parent=self)
        self._search_edit.setPlaceholderText("filter by handle / name / target")
        filter_row.addWidget(self._search_edit, 1)
        self._refresh_btn = QPushButton("Refresh", parent=self)
        self._refresh_btn.setStyleSheet(self._cancel_style)
        self._refresh_btn.setToolTip(
            "Re-read run metadata from disk.  Use after creating, "
            "reconstructing, or curating in another part of the UI."
        )
        filter_row.addWidget(self._refresh_btn)
        root.addLayout(filter_row)

        self._list = QListWidget(parent=self)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setStyleSheet(
            "QListWidget {"
            "  background-color: rgba(255, 255, 255, 0.03);"
            "  border: 1px solid rgba(255, 255, 255, 0.08);"
            "  border-radius: 4px;"
            "  padding: 2px;"
            "}"
            "QListWidget::item {"
            "  border-bottom: 1px solid rgba(255, 255, 255, 0.04);"
            "}"
            "QListWidget::item:last { border-bottom: none; }"
            "QListWidget::item:hover {"
            "  background-color: rgba(255, 255, 255, 0.04);"
            "}"
            "QListWidget::item:selected {"
            "  background-color: rgba(212, 168, 67, 0.18);"
            "}"
        )
        root.addWidget(self._list, 1)

        self._selection_label = QLabel("(no row selected)", parent=self)
        self._selection_label.setStyleSheet("color: #aaccee; font-size: 11px;")
        root.addWidget(self._selection_label)

        # Action button row
        action_row = QHBoxLayout()
        self._switch_btn = QPushButton("Switch to", parent=self)
        self._switch_btn.setStyleSheet(self._cancel_style)
        self._switch_btn.setEnabled(False)
        self._switch_btn.setToolTip(
            "Make the selected run the active run.  Closes this panel + "
            "reloads the new run's state.  Disabled if you're already on "
            "this run."
        )
        self._edit_btn = QPushButton("Edit Metadata", parent=self)
        self._edit_btn.setStyleSheet(self._cancel_style)
        self._edit_btn.setEnabled(False)
        self._edit_btn.setToolTip(
            "Edit the run's display name, target, and description.  The "
            "run handle (folder name) + backend stay immutable.  Disabled "
            "for the default run."
        )
        self._archive_btn = QPushButton("Archive", parent=self)
        self._archive_btn.setStyleSheet(self._cancel_style)
        self._archive_btn.setEnabled(False)
        self._archive_btn.setToolTip(
            "Hide this run from the active picker.  Models + data stay on "
            "disk.  Unarchive any time.  Disabled for the default run."
        )
        self._unarchive_btn = QPushButton("Unarchive", parent=self)
        self._unarchive_btn.setStyleSheet(self._cancel_style)
        self._unarchive_btn.setEnabled(False)
        self._unarchive_btn.setVisible(False)
        self._unarchive_btn.setToolTip(
            "Restore this archived run to the active picker."
        )
        self._delete_btn = QPushButton("Delete", parent=self)
        self._delete_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: rgba(229, 115, 115, 0.20);"
            "  border: 1px solid rgba(229, 115, 115, 0.45);"
            "  border-radius: 4px;"
            "  color: #ffcccc;"
            "  padding: 6px 18px;"
            "  min-width: 80px;"
            "}"
            "QPushButton:hover {"
            "  background-color: rgba(229, 115, 115, 0.30);"
            "}"
            "QPushButton:disabled {"
            "  background-color: rgba(229, 115, 115, 0.08);"
            "  color: #886666;"
            "}"
        )
        self._delete_btn.setEnabled(False)
        self._delete_btn.setToolTip(
            "PERMANENTLY remove this run from disk + state.  Includes all "
            "reconstructed + curated patch outputs for the run.  Disabled "
            "for the default + active run.  Cannot be undone."
        )
        self._clean_btn = QPushButton("Clean intermediates", parent=self)
        self._clean_btn.setStyleSheet(self._cancel_style)
        self._clean_btn.setEnabled(False)
        self._clean_btn.setToolTip(
            "Free disk space by deleting this run's cached cellpose "
            "prediction outputs (flows/probs) for its RECONSTRUCTED patches. "
            "Reconstructions + curation are kept; only the regenerable "
            "prediction cache is removed.  Re-PREDICT to re-reconstruct."
        )
        # Hide entirely if no clean callback was injected (back-compat).
        self._clean_btn.setVisible(self._on_clean_intermediates is not None)
        action_row.addWidget(self._switch_btn)
        action_row.addWidget(self._edit_btn)
        action_row.addWidget(self._archive_btn)
        action_row.addWidget(self._unarchive_btn)
        action_row.addWidget(self._delete_btn)
        action_row.addWidget(self._clean_btn)
        # Phase 7: workspace-level (selection-INDEPENDENT) "Convert to
        # unified layout" button.  Deliberately NOT registered in
        # _update_action_buttons — its enablement depends only on whether the
        # workspace is already unified, never on the selected row.
        self._convert_btn = QPushButton("Convert to unified layout", parent=self)
        self._convert_btn.setStyleSheet(self._cancel_style)
        if self._is_unified:
            self._convert_btn.setEnabled(False)
            self._convert_btn.setToolTip("Already unified")
        else:
            self._convert_btn.setToolTip(
                "Move the default run's data into runs/default/ (the symmetric "
                "nested layout).  A backup is taken first.  Optional + one-way."
            )
        # Hide the button entirely if no migration callback was injected.
        self._convert_btn.setVisible(self._on_convert_layout is not None)
        action_row.addWidget(self._convert_btn)
        action_row.addStretch(1)
        self._close_btn = QPushButton("Close", parent=self)
        self._close_btn.setStyleSheet(self._cancel_style)
        action_row.addWidget(self._close_btn)
        root.addLayout(action_row)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _wire_signals(self):
        self._filter_combo.currentIndexChanged.connect(self._refresh_table)
        self._search_edit.textChanged.connect(self._refresh_table)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._switch_btn.clicked.connect(self._on_switch_clicked)
        self._edit_btn.clicked.connect(self._on_edit_clicked)
        self._archive_btn.clicked.connect(self._on_archive_clicked)
        self._unarchive_btn.clicked.connect(self._on_unarchive_clicked)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        self._clean_btn.clicked.connect(self._on_clean_intermediates_clicked)
        self._close_btn.clicked.connect(self.accept)
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        self._convert_btn.clicked.connect(self._on_convert_clicked)

    def _on_refresh_clicked(self):
        self._reload_from_callback()

    def _on_convert_clicked(self):
        """Workspace-level: convert the default run to the nested layout.

        Confirms (destructive-ish — moves real reconstruction data, but a
        backup is taken first), invokes the controller-injected
        ``on_convert_layout`` callback, then refreshes the panel.  On
        success the button disables (already-unified).
        """
        if self._on_convert_layout is None or self._is_unified:
            return
        if not self._confirm(
            "Convert to unified layout?",
            "This MOVES your default-run data (reconstructions, QC, curation, "
            "prediction cache, refinement) into runs/default/ — the symmetric "
            "nested layout.\n\nA full backup is taken first, and the migration "
            "rolls back automatically if anything fails.  The 3D annotator "
            "must be closed and no Step-3 operation may be running.\n\n"
            "Continue?",
            confirm_text="Convert",
        ):
            return
        try:
            ok = bool(self._on_convert_layout())
        except Exception as e:  # noqa: BLE001
            self._show_error("Convert failed", str(e))
            return
        if ok:
            self._is_unified = True
            self._convert_btn.setEnabled(False)
            self._convert_btn.setToolTip("Already unified")
            self._reload_from_callback()

    def showEvent(self, event):
        """Refresh-on-show so a re-opened panel never shows a stale snapshot."""
        super().showEvent(event)
        if getattr(self, "_summaries_provider", None) is not None:
            try:
                self._reload_from_callback()
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Manage Runs auto-refresh failed: {e}")

    # ------------------------------------------------------------------
    # Card population
    # ------------------------------------------------------------------

    def _is_default(self, summary: Dict) -> bool:
        return str(summary.get("run_id", "")) == DEFAULT_RUN_ID

    def _make_run_card(self, summary: Dict) -> QWidget:
        """Build a card widget for one run.

        Layout:
          ●  {run_handle}                         [active | archived]
             {display_name}
             {target} · recon:N cur:N acc:N fin:N · total:N · {last_activity}
        """
        from napari_epicycle.ui_shell.theme import tier_pill_style

        is_active = summary.get("is_active", False)
        archived = summary.get("archived", False)
        run_handle = str(summary.get("run_id", ""))
        display_name = str(summary.get("display_name", "")) or run_handle
        target = str(summary.get("target", "")) or "—"
        counts = summary.get("patch_counts") or {}
        recon = counts.get("reconstructed", 0)
        cur = counts.get("curated", 0)
        acc = counts.get("accepted", 0)
        fin = counts.get("finalized", 0)
        total = counts.get("total", 0)
        mtime = summary.get("last_activity_iso") or "—"
        mtime_short = mtime[:10] if len(mtime) >= 10 else mtime

        card = QWidget(parent=self._list)
        card.setStyleSheet("QWidget { background-color: transparent; }")
        row = QHBoxLayout(card)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(10)

        dot = QLabel("●" if is_active else " ", parent=card)
        dot.setFixedWidth(14)
        dot.setStyleSheet(
            "color: #d4a843; font-size: 14px; font-weight: bold;"
            " background: transparent; border: none;"
            if is_active else
            "color: transparent; background: transparent; border: none;"
        )
        row.addWidget(dot, 0, Qt.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        handle_label = QLabel(run_handle, parent=card)
        handle_color = "#888888" if archived else "#eeeeee"
        handle_label.setStyleSheet(
            f"color: {handle_color}; font-size: 12px; font-weight: bold;"
            " background: transparent; border: none;"
        )
        text_col.addWidget(handle_label)

        display_label = QLabel(display_name, parent=card)
        display_color = "#666666" if archived else "#bbbbbb"
        display_label.setStyleSheet(
            f"color: {display_color}; font-size: 11px;"
            " background: transparent; border: none;"
        )
        text_col.addWidget(display_label)

        meta = (
            f"target: {target}  ·  recon:{recon} cur:{cur} acc:{acc} "
            f"fin:{fin}  ·  total:{total}  ·  {mtime_short}"
        )
        meta_label = QLabel(meta, parent=card)
        meta_color = "#555555" if archived else "#999999"
        meta_label.setStyleSheet(
            f"color: {meta_color}; font-size: 10px;"
            " background: transparent; border: none;"
        )
        text_col.addWidget(meta_label)

        row.addLayout(text_col, 1)

        if archived:
            badge = QLabel("archived", parent=card)
            badge.setStyleSheet(tier_pill_style("#888888"))
        elif is_active:
            badge = QLabel("active", parent=card)
            badge.setStyleSheet(tier_pill_style("#d4a843"))
        else:
            badge = QLabel("", parent=card)
            badge.setStyleSheet("background: transparent; border: none;")
        badge.setAlignment(Qt.AlignCenter)
        row.addWidget(badge, 0, Qt.AlignVCenter)

        return card

    def _refresh_table(self, *args):
        """Re-populate from self._summaries with current filter + search."""
        filter_mode = self._filter_combo.currentText()
        search = self._search_edit.text().strip().lower()

        rows = []
        for s in self._summaries:
            archived = s.get("archived", False)
            if filter_mode == "Active" and archived:
                continue
            if filter_mode == "Archived" and not archived:
                continue
            if search:
                haystack = " ".join([
                    str(s.get("run_id", "")),
                    str(s.get("display_name", "")),
                    str(s.get("target", "")),
                ]).lower()
                if search not in haystack:
                    continue
            rows.append(s)

        self._list.clear()
        for s in rows:
            card = self._make_run_card(s)
            item = QListWidgetItem(self._list)
            item.setSizeHint(card.sizeHint())
            item.setData(Qt.UserRole, s)
            self._list.addItem(item)
            self._list.setItemWidget(item, card)

        self._list.clearSelection()
        self._update_action_buttons(None)

    def _selected_summary(self) -> Optional[Dict]:
        items = self._list.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.UserRole)

    def _on_selection_changed(self):
        summary = self._selected_summary()
        self._update_action_buttons(summary)
        if summary is None:
            self._selection_label.setText("(no row selected)")
        else:
            slug = summary.get("run_id", "?")
            display = (summary.get("display_name") or "").strip() or slug
            active_suffix = " — currently ACTIVE" if summary.get("is_active") else ""
            archived_suffix = " (archived)" if summary.get("archived") else ""
            if display != slug:
                label = f"Selected: {display} (handle: {slug})"
            else:
                label = f"Selected: {slug}"
            self._selection_label.setText(
                f"{label}{active_suffix}{archived_suffix}"
            )

    def _update_action_buttons(self, summary: Optional[Dict]):
        if summary is None:
            self._switch_btn.setEnabled(False)
            self._edit_btn.setEnabled(False)
            self._archive_btn.setEnabled(False)
            self._archive_btn.setVisible(True)
            self._unarchive_btn.setVisible(False)
            self._unarchive_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            self._clean_btn.setEnabled(False)
            return
        run_id = str(summary.get("run_id", ""))
        is_archived = bool(summary.get("archived", False))
        enablement = _action_enablement(
            run_id,
            self._active_run_id,
            is_default=self._is_default(summary),
            is_archived=is_archived,
        )
        self._switch_btn.setEnabled(enablement["switch"])
        self._edit_btn.setEnabled(enablement["edit"])
        self._archive_btn.setEnabled(enablement["archive"])
        self._archive_btn.setVisible(not is_archived)
        self._unarchive_btn.setEnabled(enablement["unarchive"])
        self._unarchive_btn.setVisible(is_archived)
        self._delete_btn.setEnabled(enablement["delete"])
        self._clean_btn.setEnabled(enablement["clean"])

    def _on_double_click(self, _item):
        summary = self._selected_summary()
        if summary is None or summary.get("is_active"):
            return
        self._on_switch_clicked()

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _on_switch_clicked(self):
        summary = self._selected_summary()
        if summary is None or self._on_switch is None:
            return
        slug = summary["run_id"]
        try:
            self._on_switch(slug)
        except Exception as e:  # noqa: BLE001
            self._show_error("Switch failed", str(e))
            return
        self.accept()

    def _on_edit_clicked(self):
        summary = self._selected_summary()
        if summary is None or self._on_edit is None:
            return
        slug = summary["run_id"]
        other_names = [
            (s.get("display_name") or "")
            for s in self._summaries
            if s.get("run_id") != slug
        ]
        dlg = EditRunMetadataDialog(
            slug=slug,
            current_display_name=summary.get("display_name", slug),
            current_target=summary.get("target", "") or "",
            current_backend=summary.get("backend", "cellpose"),
            current_description=summary.get("description", "") or "",
            existing_display_names=other_names,
            parent=self,
        )
        if not dlg.exec_():
            return
        updates = dlg.get_updates()
        if not updates:
            return
        try:
            ok = self._on_edit(slug, updates)
        except Exception as e:  # noqa: BLE001
            self._show_error("Edit failed", str(e))
            return
        if ok:
            self._reload_from_callback()

    def _on_archive_clicked(self):
        summary = self._selected_summary()
        if summary is None or self._on_archive is None:
            return
        slug = summary["run_id"]
        display = (summary.get("display_name") or "").strip() or slug
        label = f"{display} (handle: {slug})" if display != slug else slug
        if not self._confirm(
            "Archive run?",
            f"Archive {label}?  It'll be hidden from the active picker but "
            f"preserved on disk.  You can unarchive any time.",
            confirm_text="Archive",
        ):
            return
        try:
            ok = self._on_archive(slug)
        except Exception as e:  # noqa: BLE001
            self._show_error("Archive failed", str(e))
            return
        if ok:
            self._reload_from_callback()

    def _on_unarchive_clicked(self):
        summary = self._selected_summary()
        if summary is None or self._on_unarchive is None:
            return
        slug = summary["run_id"]
        try:
            ok = self._on_unarchive(slug)
        except Exception as e:  # noqa: BLE001
            self._show_error("Unarchive failed", str(e))
            return
        if ok:
            self._reload_from_callback()

    def _on_delete_clicked(self):
        summary = self._selected_summary()
        if summary is None or self._on_delete is None:
            return
        slug = summary["run_id"]
        display = (summary.get("display_name") or "").strip() or slug
        label = f"{display} (handle: {slug})" if display != slug else slug

        # Provenance dependency check.  Step 3 uses copy-on-reuse, so a
        # dependent does NOT need the source on disk — this is a SOFT
        # warn folded into the confirm body, NOT a hard block.
        dependents: List[str] = []
        if self._find_dependents is not None:
            try:
                dependents = list(self._find_dependents(slug) or [])
            except Exception:  # noqa: BLE001
                dependents = []

        counts = summary.get("patch_counts") or {}
        recon = counts.get("reconstructed", 0)
        cur = counts.get("curated", 0)
        acc = counts.get("accepted", 0)
        fin = counts.get("finalized", 0)

        body = (
            f"Permanently delete run {label}?\n\n"
            f"This will remove:\n"
            f"  • {recon} reconstructed + {cur} in-curation patch(es)\n"
            f"  • {acc} accepted + {fin} finalized patch(es)\n"
            f"  • The run's QC features + run manifest\n\n"
        )
        if _should_warn_dependents(dependents):
            body += (
                f"NOTE: {len(dependents)} other run(s) record this run as "
                f"their init source:\n"
                + "\n".join(f"  • {d}" for d in dependents[:10])
                + (f"\n  ... and {len(dependents) - 10} more"
                   if len(dependents) > 10 else "")
                + "\n\nStep 3 copies the model into each run at create "
                "time, so those runs keep working — but their recorded "
                "lineage will point at a deleted run.\n\n"
            )
        body += "This action CANNOT be undone."

        if not self._confirm(
            "DELETE run?",
            body,
            confirm_text="Delete forever",
            destructive=True,
        ):
            return
        try:
            ok = self._on_delete(slug)
        except Exception as e:  # noqa: BLE001
            self._show_error("Delete failed", str(e))
            return
        if ok:
            self._reload_from_callback()

    def _on_clean_intermediates_clicked(self):
        """Free the selected run's reclaimable prediction cache.

        The CONFIRM dialog (with the size estimate) + the "nothing to clean"
        info modal are built in the CONTROLLER handler, which has the cache
        summary — so the panel just invokes the injected callback and
        refreshes the cards on success.
        """
        summary = self._selected_summary()
        if summary is None or self._on_clean_intermediates is None:
            return
        slug = summary["run_id"]
        try:
            ok = self._on_clean_intermediates(slug)
        except Exception as e:  # noqa: BLE001
            self._show_error("Clean intermediates failed", str(e))
            return
        if ok:
            self._reload_from_callback()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reload_from_callback(self):
        """Re-fetch summaries via the injected provider + refresh cards."""
        if callable(self._summaries_provider):
            try:
                self._summaries = self._summaries_provider()
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Manage Runs summaries reload failed: {e}")
        self._refresh_table()

    def set_summaries_provider(self, provider: Callable[[], List[Dict]]):
        """Inject a 0-arg callable that returns fresh summaries."""
        self._summaries_provider = provider

    def _show_error(self, title: str, body: str):
        from napari_epicycle.ui_shell.theme import _DIALOG_STYLE, _DIALOG_BTN_CANCEL
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setStyleSheet(_DIALOG_STYLE)
        dlg.setMinimumWidth(480)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        title_lbl = QLabel(title, parent=dlg)
        title_lbl.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #e57373;"
        )
        layout.addWidget(title_lbl)
        body_lbl = QLabel(body, parent=dlg)
        body_lbl.setWordWrap(True)
        body_lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
        layout.addWidget(body_lbl)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok_btn = QPushButton("OK", parent=dlg)
        ok_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)
        dlg.exec_()

    def _confirm(self, title: str, body: str, confirm_text: str = "Confirm",
                 destructive: bool = False) -> bool:
        from napari_epicycle.ui_shell.theme import (
            _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setStyleSheet(_DIALOG_STYLE)
        dlg.setMinimumWidth(440)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        title_lbl = QLabel(title, parent=dlg)
        color = "#e57373" if destructive else "#d4a843"
        title_lbl.setStyleSheet(
            f"font-weight: bold; font-size: 14px; color: {color};"
        )
        layout.addWidget(title_lbl)
        body_lbl = QLabel(body, parent=dlg)
        body_lbl.setWordWrap(True)
        body_lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
        layout.addWidget(body_lbl)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("Cancel", parent=dlg)
        cancel_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
        cancel_btn.setDefault(True)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel_btn)
        confirm_btn = QPushButton(confirm_text, parent=dlg)
        if destructive:
            confirm_btn.setStyleSheet(
                "QPushButton {"
                "  background-color: rgba(229, 115, 115, 0.25);"
                "  border: 1px solid rgba(229, 115, 115, 0.50);"
                "  border-radius: 4px; color: #ffeeee;"
                "  font-weight: bold; padding: 6px 18px; min-width: 80px;"
                "} QPushButton:hover { background-color: rgba(229, 115, 115, 0.35); }"
            )
        else:
            confirm_btn.setStyleSheet(_DIALOG_BTN_CONFIRM)
        confirm_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(confirm_btn)
        layout.addLayout(btn_row)
        return dlg.exec_() == QDialog.Accepted


# ----------------------------------------------------------------------------
# Sub-dialog: Edit Metadata
# ----------------------------------------------------------------------------

class EditRunMetadataDialog(QDialog):
    """Sub-dialog for editing display_name / target / description.

    Slug + backend are shown but greyed (immutable).
    """

    def __init__(
        self,
        *,
        slug: str,
        current_display_name: str,
        current_target: str,
        current_backend: str,
        current_description: str = "",
        existing_display_names: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._slug = slug
        self._original_display_name = current_display_name
        self._original_target = current_target
        self._original_description = current_description
        _names = list(existing_display_names or [])
        self._other_display_names_lower = {
            n.strip().lower() for n in _names
            if n and n.strip().lower() != (current_display_name or "").strip().lower()
        }
        self._updates: Dict = {}

        from napari_epicycle.ui_shell.theme import (
            _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        self._cancel_style = _DIALOG_BTN_CANCEL
        self._confirm_style = _DIALOG_BTN_CONFIRM

        _title_display = (current_display_name or "").strip() or slug
        self.setWindowTitle(f"Edit Run: {_title_display}")
        self.setStyleSheet(_DIALOG_STYLE)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel(
            f"Edit Metadata — {_title_display} (handle: {slug})",
            parent=self,
        )
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #eeeeee;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Run handle + backend are immutable (they define the run's "
            "identity).  Display name, target, and description can be "
            "freely edited.",
            parent=self,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setSpacing(8)
        slug_lbl = QLabel(slug, parent=self)
        slug_lbl.setStyleSheet("color: #888888; font-family: monospace;")
        form.addRow("Run handle:", slug_lbl)
        backend_lbl = QLabel(current_backend, parent=self)
        backend_lbl.setStyleSheet("color: #888888;")
        form.addRow("Backend:", backend_lbl)
        self._display_name_edit = QLineEdit(current_display_name, parent=self)
        form.addRow("Display name:", self._display_name_edit)
        self._target_edit = QLineEdit(current_target, parent=self)
        self._target_edit.setPlaceholderText("e.g. cells, nuclei, membrane")
        form.addRow("Target:", self._target_edit)
        self._description_edit = QTextEdit(parent=self)
        self._description_edit.setMaximumHeight(80)
        self._description_edit.setPlaceholderText(
            "Optional notes about this run's purpose."
        )
        if self._original_description:
            self._description_edit.setPlainText(self._original_description)
        form.addRow("Description:", self._description_edit)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel", parent=self)
        self._cancel_btn.setStyleSheet(self._cancel_style)
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)
        self._save_btn = QPushButton("Save Changes", parent=self)
        self._save_btn.setStyleSheet(self._confirm_style)
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self._save_btn)
        layout.addLayout(btn_row)

    def _on_save(self):
        display_name = self._display_name_edit.text().strip()
        if not display_name:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Display name required",
                "Display name cannot be empty.  Use the run handle if you "
                "want the human-readable name to match the filesystem name.",
            )
            return

        if display_name.lower() in self._other_display_names_lower:
            from qtpy.QtWidgets import QMessageBox
            res = QMessageBox.question(
                self, "Display name already in use",
                f"Another run already has the display name "
                f"'{display_name}'.  Two runs with the same display name "
                f"will look identical in the Run picker — only the handle "
                f"distinguishes them.\n\nUse this name anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                return

        target = self._target_edit.text().strip()
        description = self._description_edit.toPlainText().strip()

        self._updates = {}
        if display_name != self._original_display_name:
            self._updates["display_name"] = display_name
        if target != self._original_target:
            self._updates["target"] = target
        if description != getattr(self, "_original_description", ""):
            self._updates["description"] = description

        self.accept()

    def get_updates(self) -> Dict:
        return self._updates


__all__ = [
    "ManageRunsPanel",
    "EditRunMetadataDialog",
    "_action_enablement",
    "_should_warn_dependents",
]

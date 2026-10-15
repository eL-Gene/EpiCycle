"""Phase 8c — Manage Runs panel (table view + per-row actions).

Themed Qt dialog showing all runs in the active workspace.  Per-row
actions: Switch / Edit metadata / Archive / Unarchive / Delete.

Data flow:

* `_run_management.list_run_summaries(workspace)` → table rows
* User picks a row, clicks an action button
* Action handler calls into `_run_management` helpers (archive_run,
  unarchive_run, edit_run_metadata, delete_run_from_disk +
  remove_run_from_state_json)
* For Delete, runs `find_runs_inheriting_from` first — if any dependents
  exist, BLOCK the delete with a themed dialog listing them

Layout (text mockup):

  ┌─ Manage Runs ─────────────────────────────────────────────────────┐
  │  Filter: [Active ▼]   Search: [______]                            │
  │                                                                    │
  │  ┌───┬──────────┬──────────────┬─────────┬──────┬──────┬────────┐ │
  │  │ ● │ Run ID   │ Display Name │ Target  │ Lat  │ Ax   │ Status │ │
  │  ├───┼──────────┼──────────────┼─────────┼──────┼──────┼────────┤ │
  │  │ ● │ default  │ Default      │ general │  3   │  2   │ Active │ │
  │  │   │ nuclei_v1│ Nuclei v1    │ nuclei  │  1   │  0   │ Active │ │
  │  └───┴──────────┴──────────────┴─────────┴──────┴──────┴────────┘ │
  │                                                                    │
  │  Selected: 'default'                                               │
  │  [ Edit Metadata ]  [ Archive ]  [ Delete ]              [ Close ] │
  └───────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# Action enum for the result dict returned to the caller (controller decides
# whether to refresh picker / switch active / etc.)
ACTION_NONE = "none"
ACTION_SWITCH = "switch"
ACTION_EDIT = "edit"
ACTION_ARCHIVE = "archive"
ACTION_UNARCHIVE = "unarchive"
ACTION_DELETE = "delete"


class ManageRunsPanel(QDialog):
    """Table-based panel for managing all runs in a workspace."""

    def __init__(
        self,
        *,
        run_summaries: List[Dict],
        active_run_id: str,
        # Callbacks the dialog calls back into the controller for side-effects.
        # All take (slug) → outcome (bool ok or raises).
        on_switch: Optional[Callable[[str], None]] = None,
        on_edit: Optional[Callable[[str, Dict], bool]] = None,
        on_archive: Optional[Callable[[str], bool]] = None,
        on_unarchive: Optional[Callable[[str], bool]] = None,
        on_delete: Optional[Callable[[str], bool]] = None,
        find_dependents_callback: Optional[Callable[[str], List[str]]] = None,
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
        self._find_dependents = find_dependents_callback

        from napari_epicycle.ui_shell.theme import (
            DIALOG_FORM_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        self._cancel_style = _DIALOG_BTN_CANCEL
        self._confirm_style = _DIALOG_BTN_CONFIRM

        self.setWindowTitle("Manage Runs")
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

        title = QLabel("Manage Runs", parent=self)
        title.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #eeeeee;"
        )
        root.addWidget(title)

        subtitle = QLabel(
            "Edit, archive, or delete training runs in this workspace.  "
            "Active runs are marked with ●.  Archived runs are hidden from "
            "the dropdown but preserved on disk.",
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
        # 2026-05-25 audit fix (Group C): explicit Refresh button.  Panel
        # otherwise shows stale counts if user imports / trains / etc.
        # in another modal cycle while this dialog stays cached.
        # showEvent also re-fetches so manual click is rarely needed.
        self._refresh_btn = QPushButton("Refresh", parent=self)
        self._refresh_btn.setStyleSheet(self._cancel_style)
        self._refresh_btn.setToolTip(
            "Re-read run metadata from disk.  Use after creating, importing, "
            "or training in another part of the UI."
        )
        filter_row.addWidget(self._refresh_btn)
        root.addLayout(filter_row)

        # 2026-05-26 theming sweep: run table replaced by a card list,
        # matching the Import Pairs / External Import dialog aesthetic.
        # Per user request — wanted Manage Runs to share the card style
        # they liked in Import Pairs.  Each card shows handle / display
        # name / meta line (target · L:N A:M · Npairs · last activity) /
        # status badge.  Selection model preserved (single, click to
        # select, doubleClick still wired).
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

        # Selection status label
        self._selection_label = QLabel("(no row selected)", parent=self)
        self._selection_label.setStyleSheet("color: #aaccee; font-size: 11px;")
        root.addWidget(self._selection_label)

        # Action button row
        action_row = QHBoxLayout()
        self._switch_btn = QPushButton("Switch to", parent=self)
        self._switch_btn.setStyleSheet(self._cancel_style)
        self._switch_btn.setEnabled(False)
        self._switch_btn.setToolTip(
            "Make the selected run the active run.  Same as picking it "
            "in the Curation header dropdown — closes this panel + "
            "reloads the new run's state (patch, slices, model, etc.).  "
            "Disabled if you're already on this run."
        )
        self._edit_btn = QPushButton("Edit Metadata", parent=self)
        self._edit_btn.setStyleSheet(self._cancel_style)
        self._edit_btn.setEnabled(False)
        self._edit_btn.setToolTip(
            "Edit the run's display name, target (e.g. 'nuclei', "
            "'membrane'), and description.  Useful for renaming a run "
            "without touching its filesystem folder.  The run handle "
            "(folder name) + backend stay immutable."
        )
        self._archive_btn = QPushButton("Archive", parent=self)
        self._archive_btn.setStyleSheet(self._cancel_style)
        self._archive_btn.setEnabled(False)
        self._archive_btn.setToolTip(
            "Hide this run from the active dropdown.  Models + data stay "
            "on disk; Step 3's picker still sees them.  Use for old runs "
            "you don't want cluttering the active list but might revisit.  "
            "Unarchive any time."
        )
        self._unarchive_btn = QPushButton("Unarchive", parent=self)
        self._unarchive_btn.setStyleSheet(self._cancel_style)
        self._unarchive_btn.setEnabled(False)
        self._unarchive_btn.setVisible(False)
        self._unarchive_btn.setToolTip(
            "Restore this archived run to the active dropdown."
        )
        self._delete_btn = QPushButton("Delete", parent=self)
        # Use a red-tinted style for delete (destructive)
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
            "training data, model checkpoints, and curated pairs for the "
            "run.  Blocked if any other run was initialized from this one "
            "as its starting checkpoint.  Cannot be undone."
        )
        action_row.addWidget(self._switch_btn)
        action_row.addWidget(self._edit_btn)
        action_row.addWidget(self._archive_btn)
        action_row.addWidget(self._unarchive_btn)
        action_row.addWidget(self._delete_btn)
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
        self._close_btn.clicked.connect(self.accept)
        # 2026-05-25 audit fix (Group C): explicit Refresh wiring.
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)

    def _on_refresh_clicked(self):
        """User-triggered refresh — re-fetch summaries from disk."""
        self._reload_from_callback()

    def showEvent(self, event):
        """Override to refresh-on-show.

        2026-05-25 audit fix (Group C): without this, opening the panel
        a second time in one session shows the snapshot from the FIRST
        open.  Cheap to call (filesystem scan via list_run_summaries)
        and removes the stale-window class of bug entirely.
        """
        super().showEvent(event)
        if getattr(self, "_summaries_provider", None) is not None:
            try:
                self._reload_from_callback()
            except Exception as e:
                print(f"[WARN] Manage Runs auto-refresh failed: {e}")

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _make_run_card(self, summary: Dict) -> QWidget:
        """Build a card widget for one run.

        Layout:
          ●  {run_handle}                         [active | archived]
             {display_name}
             {target} · L:N A:M · N pairs · {last_activity}

        ● glyph is amber when active, hidden otherwise.  Archived runs
        render with dimmed text + an "archived" pill.  Selected state
        is handled by the QListWidget item background tint (see
        stylesheet at construction).
        """
        from napari_epicycle.ui_shell.theme import tier_pill_style

        is_active = summary.get("is_active", False)
        archived = summary.get("archived", False)
        run_handle = str(summary.get("run_id", ""))
        display_name = str(summary.get("display_name", "")) or run_handle
        target = str(summary.get("target", "")) or "—"
        lat = summary.get("lateral_round_count", 0)
        ax = summary.get("axial_round_count", 0)
        pairs = summary.get("accepted_pair_count", 0)
        mtime = summary.get("last_activity_iso") or "—"
        mtime_short = mtime[:10] if len(mtime) >= 10 else mtime

        card = QWidget(parent=self._list)
        card.setStyleSheet("QWidget { background-color: transparent; }")
        row = QHBoxLayout(card)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(10)

        # Active dot
        dot = QLabel("●" if is_active else " ", parent=card)
        dot.setFixedWidth(14)
        dot.setStyleSheet(
            "color: #d4a843; font-size: 14px; font-weight: bold;"
            " background: transparent; border: none;"
            if is_active else
            "color: transparent; background: transparent; border: none;"
        )
        row.addWidget(dot, 0, Qt.AlignVCenter)

        # Title + meta vbox
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

        meta = f"target: {target}  ·  L:{lat} A:{ax}  ·  {pairs} pair(s)  ·  {mtime_short}"
        meta_label = QLabel(meta, parent=card)
        meta_color = "#555555" if archived else "#999999"
        meta_label.setStyleSheet(
            f"color: {meta_color}; font-size: 10px;"
            " background: transparent; border: none;"
        )
        text_col.addWidget(meta_label)

        row.addLayout(text_col, 1)

        # Status badge on the right
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
        """Re-populate from self._summaries with current filter + search.

        Method name kept as ``_refresh_table`` for back-compat with
        existing callers (filter_combo / search_edit / refresh_btn
        signals all wired here).  Render target is now QListWidget +
        per-row card widgets, not a 9-column table.
        """
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
            # Stash the full summary dict on the item for retrieval
            item.setData(Qt.UserRole, s)
            self._list.addItem(item)
            self._list.setItemWidget(item, card)

        # Clear selection + reset action buttons
        self._list.clearSelection()
        self._update_action_buttons(None)

    def _selected_summary(self) -> Optional[Dict]:
        """Return the full summary dict for the currently-selected row."""
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
            # 2026-05-25 terminology sweep: display_name primary, handle
            # as parenthetical (only when distinct from display).
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
            self._unarchive_btn.setVisible(False)
            self._unarchive_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            return
        is_active = summary.get("is_active", False)
        archived = summary.get("archived", False)
        # Can switch to: any non-active run (regardless of archived status)
        self._switch_btn.setEnabled(not is_active)
        # Can edit: any run
        self._edit_btn.setEnabled(True)
        # Can archive: active (not currently-active runs only — must
        # switch off first) + not already archived
        self._archive_btn.setEnabled(not is_active and not archived)
        self._archive_btn.setVisible(not archived)
        # Can unarchive: archived runs
        self._unarchive_btn.setEnabled(archived)
        self._unarchive_btn.setVisible(archived)
        # Can delete: any run except the currently-active one (force user
        # to switch off first; protects against deleting the run they're
        # working on by accident)
        self._delete_btn.setEnabled(not is_active)

    def _on_double_click(self, _item):
        """Double-click a row = switch to it (if not active)."""
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
        except Exception as e:
            self._show_error("Switch failed", str(e))
            return
        # Switch closes this panel + caller refreshes
        self.accept()

    def _on_edit_clicked(self):
        summary = self._selected_summary()
        if summary is None or self._on_edit is None:
            return
        slug = summary["run_id"]
        # 2026-05-25 Group D: pass other runs' display names so the
        # Edit dialog can soft-warn on collision.
        other_names = [
            (s.get("display_name") or "")
            for s in self._summaries
            if s.get("run_id") != slug
        ]
        # Open EditRunMetadataDialog (sub-dialog)
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
        except Exception as e:
            self._show_error("Edit failed", str(e))
            return
        if ok:
            # Re-read summaries + refresh table
            self._reload_from_callback()

    def _on_archive_clicked(self):
        summary = self._selected_summary()
        if summary is None or self._on_archive is None:
            return
        slug = summary["run_id"]
        # 2026-05-25 terminology sweep: display_name primary.
        display = (summary.get("display_name") or "").strip() or slug
        label = f"{display} (handle: {slug})" if display != slug else slug
        if not self._confirm(
            "Archive run?",
            f"Archive {label}?  It'll be hidden from the active "
            f"dropdown but preserved on disk.  You can unarchive any "
            f"time.  Models trained in this run remain accessible via "
            f"Step 3's picker (it scans archived runs too).",
            confirm_text="Archive",
        ):
            return
        try:
            ok = self._on_archive(slug)
        except Exception as e:
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
        except Exception as e:
            self._show_error("Unarchive failed", str(e))
            return
        if ok:
            self._reload_from_callback()

    def _on_delete_clicked(self):
        summary = self._selected_summary()
        if summary is None or self._on_delete is None:
            return
        slug = summary["run_id"]
        # 2026-05-25 audit-1 LOW: use display_name + handle parens
        # consistent with sibling Delete + Archive dialogs.
        display = (summary.get("display_name") or "").strip() or slug
        label = f"{display} (handle: {slug})" if display != slug else slug

        # Provenance dependency check FIRST
        if self._find_dependents is not None:
            try:
                dependents = self._find_dependents(slug)
            except Exception:
                dependents = []
            if dependents:
                self._show_error(
                    "Cannot delete — other runs depend on this",
                    f"Run {label} is referenced by {len(dependents)} other "
                    f"run(s) as their initialization source:\n\n"
                    + "\n".join(f"  • {d}" for d in dependents[:10])
                    + (f"\n  ... and {len(dependents) - 10} more"
                       if len(dependents) > 10 else "")
                    + "\n\nDelete those runs first (or change their init "
                    "source) before deleting this one.",
                )
                return

        # Strong confirm
        lat = summary.get("lateral_round_count", 0)
        ax = summary.get("axial_round_count", 0)
        # display + label already computed above (used by both the
        # provenance-block error and this confirm dialog).
        # 2026-09-07: delete now MOVES the run to artifacts/step2/.trash/ —
        # the old "permanently / CANNOT be undone / Delete forever" wording
        # was simply false, and scared people off a reversible action.
        if not self._confirm(
            "Delete run?",
            f"Delete run {label}?  It will disappear from your run list, "
            f"along with {lat} lateral and {ax} axial training rounds, their "
            f"models, and all curated 2D pairs.\n\n"
            f"The folder is moved to artifacts/step2/.trash/ — you can drag "
            f"it back if you change your mind, or delete it there to free "
            f"the disk space.",
            confirm_text="Delete",
            destructive=True,
        ):
            return
        try:
            ok = self._on_delete(slug)
        except Exception as e:
            self._show_error("Delete failed", str(e))
            return
        if ok:
            self._reload_from_callback()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reload_from_callback(self):
        """Re-fetch summaries + refresh.  Set externally via set_run_summaries."""
        # The controller updates the dialog's summaries after each action
        # by calling refresh_summaries(); the panel re-populates from
        # self._summaries.  This indirection lets the controller stay
        # responsible for fresh-state fetches.
        if hasattr(self, "_summaries_provider") and callable(self._summaries_provider):
            try:
                self._summaries = self._summaries_provider()
            except Exception as e:
                print(f"[WARN] Manage Runs summaries reload failed: {e}")
        self._refresh_table()

    def set_summaries_provider(self, provider: Callable[[], List[Dict]]):
        """Inject a 0-arg callable that returns fresh summaries.  Used by
        the controller so action handlers can self-refresh without a full
        re-init."""
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
        cancel_btn.setDefault(True)  # Cancel is the safe default
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel_btn)
        confirm_btn = QPushButton(confirm_text, parent=dlg)
        if destructive:
            # Red destructive style
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
        # 2026-05-25 Group D: case-folded set of OTHER runs' display
        # names — used to soft-warn the user when they try to rename
        # this run to a name another run is already using.
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

        # 2026-05-25 terminology sweep: prefer display_name in title +
        # show handle as parenthetical.  Internal var stays `slug` for
        # code clarity (matches disk manifest field name).
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
        # Run handle (read-only)
        slug_lbl = QLabel(slug, parent=self)
        slug_lbl.setStyleSheet("color: #888888; font-family: monospace;")
        form.addRow("Run handle:", slug_lbl)
        # Backend (read-only)
        backend_lbl = QLabel(current_backend, parent=self)
        backend_lbl.setStyleSheet("color: #888888;")
        form.addRow("Backend:", backend_lbl)
        # Display name (editable)
        self._display_name_edit = QLineEdit(current_display_name, parent=self)
        form.addRow("Display name:", self._display_name_edit)
        # Target (editable)
        self._target_edit = QLineEdit(current_target, parent=self)
        self._target_edit.setPlaceholderText("e.g. nuclei, membrane, (none)")
        form.addRow("Target:", self._target_edit)
        # Description (editable, multiline)
        self._description_edit = QTextEdit(parent=self)
        self._description_edit.setMaximumHeight(80)
        self._description_edit.setPlaceholderText(
            "Optional notes about this run's purpose."
        )
        if self._original_description:
            self._description_edit.setPlainText(self._original_description)
        form.addRow("Description:", self._description_edit)
        layout.addLayout(form)

        # Buttons
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
            # display_name is required — refuse to save with empty
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Display name required",
                "Display name cannot be empty.  Use the run handle if you "
                "want the human-readable name to match the filesystem name.",
            )
            return

        # 2026-05-25 Group D: soft-warn on display-name collision with
        # another run.  Two runs with the same display_name render
        # identically in the dropdown — confusing.  Soft-warn (not
        # hard-block) so the user can override if intentional.
        if display_name.lower() in self._other_display_names_lower:
            from qtpy.QtWidgets import QMessageBox
            res = QMessageBox.question(
                self, "Display name already in use",
                f"Another run already has the display name "
                f"'{display_name}'.  Two runs with the same display "
                f"name will look identical in the Run dropdown — only "
                f"the handle distinguishes them.\n\nUse this name "
                f"anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                return

        target = self._target_edit.text().strip()
        description = self._description_edit.toPlainText().strip()

        # Only include fields the user actually changed (audit-fix
        # 2026-05-24): avoid wasted disk writes when Save is clicked
        # without changes.  description was previously unconditional but
        # that re-wrote the manifest on every Save.
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
]

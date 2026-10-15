"""Themed Qt dialog for Phase 8d — Import pairs from another run.

User flow:

1. Open dialog (from Curation header [⤵ Import Pairs] button)
2. Pick source run from dropdown (excludes active run)
3. Pick source round + bucket from second dropdown (auto-filled
   based on source run)
4. Table populates with all accepted pairs from source/round/bucket
5. Default selection: all pairs checked
6. Pick target bucket + round (defaults to next available round)
7. Click Import → backend copies + writes provenance
8. Result toast shows count of copied/skipped/errored
"""

from __future__ import annotations

from typing import Dict, List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ImportPairsDialog(QDialog):
    """Cherry-pick + import accepted pairs from one run into another."""

    def __init__(
        self,
        *,
        active_run_id: str,
        run_summaries: List[Dict],
        list_run_rounds_callback,
        list_source_pairs_callback,
        suggest_target_round_callback=None,
        active_display_name: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Args:
            active_run_id: the run that pairs will be imported INTO.
            run_summaries: from controller._list_runs_for_ui(); each dict
                has run_id + display_name + lateral_round_count +
                axial_round_count + archived + is_active.
            list_run_rounds_callback: ``(run_id) -> List[round-dict]``.
                Used to populate the source-round dropdown.  See
                ``_import_pairs.list_source_run_rounds``.
            list_source_pairs_callback: ``(run_id, bucket, round_idx) ->
                List[pair-dict]``.  Used to populate the pair table.
            suggest_target_round_callback: optional
                ``(bucket) -> int`` — returns the next available round
                index for the active run + bucket.  Defaults to None
                (user picks manually).
            active_display_name: human-readable name of the active run
                (target).  2026-05-25 terminology sweep — used in
                subtitle + target header so user sees the display name
                instead of the raw handle.  Falls back to handle when
                not provided.
            parent: parent widget.
        """
        super().__init__(parent)
        self._active_run_id = active_run_id
        # 2026-05-25 terminology sweep: prefer display_name in subtitle +
        # target header.
        _dn = (active_display_name or "").strip()
        self._active_display_name = _dn if _dn else active_run_id
        self._run_summaries = run_summaries
        self._list_run_rounds = list_run_rounds_callback
        self._list_source_pairs = list_source_pairs_callback
        self._suggest_target_round = suggest_target_round_callback
        self._result: Optional[dict] = None
        self._current_source_pairs: List[Dict] = []

        from napari_epicycle.ui_shell.theme import (
            DIALOG_FORM_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        self._cancel_style = _DIALOG_BTN_CANCEL
        self._confirm_style = _DIALOG_BTN_CONFIRM

        self.setWindowTitle("Import Pairs from Another Run")
        self.setStyleSheet(DIALOG_FORM_STYLE)
        self.setMinimumWidth(620)
        self.setMinimumHeight(540)

        self._build_ui()
        self._wire_signals()
        self._populate_source_run_combo()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        title = QLabel("Import Pairs from Another Run", parent=self)
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #eeeeee;")
        root.addWidget(title)

        # 2026-05-25 terminology sweep: prefer display_name + show handle
        # as parenthetical only when distinct.
        if self._active_display_name != self._active_run_id:
            _target_str = f"{self._active_display_name} (handle: {self._active_run_id})"
        else:
            _target_str = self._active_run_id
        subtitle = QLabel(
            f"Cherry-pick accepted training pairs from another run in this "
            f"workspace and copy them into {_target_str}.  "
            f"Useful when you've curated good samples in one run and want "
            f"to seed a new run with them.",
            parent=self,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        root.addWidget(subtitle)

        # ===== Source ===== (QGroupBox sectioning per Step 3 inspiration,
        # styled at dialog level via DIALOG_FORM_STYLE)
        src_group = QGroupBox("Source", parent=self)
        src_form = QFormLayout(src_group)
        src_form.setContentsMargins(8, 14, 8, 8)
        src_form.setSpacing(6)

        self._source_run_combo = QComboBox(parent=src_group)
        self._source_run_combo.setToolTip(
            "The run you're copying pairs FROM.  Active run is hidden "
            "(would be a no-op).  Archived runs are also hidden."
        )
        src_form.addRow("Source run:", self._source_run_combo)

        # Run summary line — populates when user picks a source run.
        # Shows: display_name · target · backend · last activity.
        self._source_run_info = QLabel("(no source selected)", parent=src_group)
        self._source_run_info.setStyleSheet(
            "color: #888888; font-size: 10px; font-style: italic; "
            "padding-left: 2px;"
        )
        self._source_run_info.setWordWrap(True)
        src_form.addRow("", self._source_run_info)

        self._source_round_combo = QComboBox(parent=src_group)
        self._source_round_combo.setToolTip(
            "Which round + bucket of the source run to pull pairs from.  "
            "Each entry shows the pair count.  Empty rounds are hidden."
        )
        src_form.addRow("Source round / bucket:", self._source_round_combo)

        root.addWidget(src_group)

        # ===== Pair selection =====
        pairs_group = QGroupBox("Pairs", parent=self)
        pairs_v = QVBoxLayout(pairs_group)
        pairs_v.setContentsMargins(8, 14, 8, 8)
        pairs_v.setSpacing(6)

        # Status filter (Phase 8d polish 2026-05-24): default to finalized
        # only (staged + committed) per user request — "only the image
        # data that has the finalized should be importable by default".
        # Toggle lets the user widen the filter.
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Show pairs at status:", parent=pairs_group))
        self._status_filter_combo = QComboBox(parent=pairs_group)
        self._status_filter_combo.addItem("Finalized only (staged + committed)",
                                           {"staged", "committed"})
        self._status_filter_combo.addItem("Accepted + finalized",
                                           {"accepted", "staged", "committed"})
        self._status_filter_combo.addItem("All (incl. still-curating)", None)
        self._status_filter_combo.setToolTip(
            "Filter source pairs by their training status:\n\n"
            "• Finalized only (default): pairs you've staged or already "
            "trained on.  Safest for importing — these are samples you've "
            "fully signed off.\n\n"
            "• Accepted + finalized: also includes pairs you've accepted "
            "but not yet staged for training.\n\n"
            "• All: includes pairs still being curated.  Use only if you "
            "know what you're pulling in."
        )
        status_row.addWidget(self._status_filter_combo)
        status_row.addStretch(1)
        pairs_v.addLayout(status_row)

        pair_row = QHBoxLayout()
        self._select_all_btn = QPushButton("Select All", parent=pairs_group)
        self._select_all_btn.setStyleSheet(self._cancel_style)
        self._select_none_btn = QPushButton("Select None", parent=pairs_group)
        self._select_none_btn.setStyleSheet(self._cancel_style)
        self._selection_count_label = QLabel("0 selected", parent=pairs_group)
        self._selection_count_label.setStyleSheet(
            "color: #aaccee; font-size: 11px; padding-left: 8px;"
        )
        pair_row.addWidget(self._select_all_btn)
        pair_row.addWidget(self._select_none_btn)
        pair_row.addWidget(self._selection_count_label)
        pair_row.addStretch(1)
        pairs_v.addLayout(pair_row)

        # 2026-05-26 theming sweep: pair table replaced by a card list.
        # Each row is now a make_pair_card(...) widget wrapping
        # checkbox + img/mask thumbs + label + meta + status badge.
        # External API of this dialog is unchanged — only the render
        # target swaps.  See theme.make_pair_card for the card layout.
        self._pair_list = QListWidget(parent=pairs_group)
        self._pair_list.setSelectionMode(QListWidget.NoSelection)
        self._pair_list.setMinimumHeight(300)
        self._pair_list.setStyleSheet(
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
        )
        pairs_v.addWidget(self._pair_list)
        root.addWidget(pairs_group)

        # ===== Target =====
        # 2026-05-25 terminology sweep: display_name primary.
        target_group = QGroupBox(f"Target — {_target_str}", parent=self)
        target_form = QFormLayout(target_group)
        target_form.setContentsMargins(8, 14, 8, 8)
        target_form.setSpacing(6)

        self._target_bucket_combo = QComboBox(parent=target_group)
        self._target_bucket_combo.addItems(["lateral", "axial"])
        target_form.addRow("Target bucket:", self._target_bucket_combo)

        self._target_round_spin = QSpinBox(parent=target_group)
        self._target_round_spin.setRange(0, 99)
        self._target_round_spin.setValue(0)
        target_form.addRow("Target round index:", self._target_round_spin)

        self._overwrite_check = QCheckBox(
            "Overwrite existing pairs (default: skip duplicates)",
            parent=target_group,
        )
        self._overwrite_check.setToolTip(
            "If a pair with the same label already exists in the target round "
            "(from a previous import OR native curation), the default is to "
            "skip it.  Check this to overwrite the existing pair with the "
            "source's version.  Affects file copy only — provenance metadata "
            "is always updated to reflect the most recent import."
        )
        target_form.addRow("", self._overwrite_check)

        root.addWidget(target_group)

        root.addStretch(1)

        # ===== Buttons =====
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel", parent=self)
        self._cancel_btn.setStyleSheet(self._cancel_style)
        self._import_btn = QPushButton("Import", parent=self)
        self._import_btn.setStyleSheet(self._confirm_style)
        self._import_btn.setDefault(True)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._import_btn)
        root.addLayout(btn_row)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text, parent=self)
        lbl.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #d4a843; padding-top: 4px;"
        )
        return lbl

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate_source_run_combo(self):
        """Fill source-run combo with everything EXCEPT the active run."""
        self._source_run_combo.clear()
        non_active = [
            s for s in self._run_summaries
            if s["run_id"] != self._active_run_id and not s.get("archived")
        ]
        if not non_active:
            self._source_run_combo.addItem("(no other runs)", "")
            self._source_run_combo.setEnabled(False)
            self._import_btn.setEnabled(False)
            return
        for s in non_active:
            label = f"{s.get('display_name') or s['run_id']}"
            self._source_run_combo.addItem(label, s["run_id"])
        # Trigger first round-list populate
        self._on_source_run_changed(0)

    def _on_source_run_changed(self, _index: int):
        """When user picks a source run, populate the round/bucket combo
        + the info-line under the combo."""
        run_id = self._source_run_combo.currentData()
        self._source_round_combo.clear()
        if not run_id:
            self._source_run_info.setText("(no source selected)")
            return

        # Build info line from run_summaries (Phase 8d polish 2026-05-24)
        summary = next(
            (s for s in self._run_summaries if s["run_id"] == run_id), None
        )
        if summary is not None:
            target = summary.get("target", "") or "(no target)"
            backend = summary.get("backend", "?")
            lat = summary.get("lateral_round_count", 0)
            ax = summary.get("axial_round_count", 0)
            last = summary.get("last_activity_iso") or "—"
            last_short = last[:10] if len(last) >= 10 else last
            self._source_run_info.setText(
                f"target: {target}  ·  backend: {backend}  ·  "
                f"lateral rounds: {lat}  ·  axial rounds: {ax}  ·  "
                f"last activity: {last_short}"
            )
        else:
            self._source_run_info.setText("(summary unavailable)")
        try:
            rounds = self._list_run_rounds(run_id)
        except Exception as e:
            self._source_round_combo.addItem(f"(error: {e})", None)
            return
        if not rounds:
            self._source_round_combo.addItem("(no rounds with pairs)", None)
            self._import_btn.setEnabled(False)
            return
        for r in rounds:
            label = f"{r['round_id']} / {r['bucket']} ({r['pair_count']} pairs)"
            self._source_round_combo.addItem(label, r)
        self._on_source_round_changed(0)
        self._import_btn.setEnabled(True)

    def _on_source_round_changed(self, _index: int):
        """When source round changes, populate the pair list."""
        round_dict = self._source_round_combo.currentData()
        self._pair_list.clear()
        self._current_source_pairs = []
        if not round_dict:
            self._update_selection_count()
            return
        run_id = self._source_run_combo.currentData()
        try:
            pairs = self._list_source_pairs(
                run_id, round_dict["bucket"], round_dict["round_idx"]
            )
        except Exception as e:
            print(f"[WARN] cherry-pick dialog: failed to list pairs from "
                  f"run={run_id!r}: {type(e).__name__}: {e}")
            self._update_selection_count()
            return
        self._current_source_pairs = pairs

        # Suggest target bucket = source bucket (most common case: copy
        # lateral pairs into lateral round)
        try:
            self._target_bucket_combo.setCurrentText(round_dict["bucket"])
        except Exception:
            pass

        # Suggest target round if callback provided
        if self._suggest_target_round is not None:
            try:
                suggested = self._suggest_target_round(round_dict["bucket"])
                if isinstance(suggested, int) and 0 <= suggested <= 99:
                    self._target_round_spin.setValue(suggested)
            except Exception:
                pass

        # Stash the unfiltered list for status-filter re-renders
        self._all_pairs_for_round = pairs
        # Render with current status filter applied
        self._render_pair_table()

    def _render_pair_table(self):
        """Render the pair list using the current status filter.

        Audit-fix 2026-05-24 (C): preserves checkbox state across
        status-filter re-renders.  Previously every re-render rebuilt
        all checkboxes as checked, silently re-checking labels the user
        had explicitly excluded.

        2026-05-26 theming sweep: render target is now a QListWidget of
        make_pair_card(...) widgets instead of a 6-column QTableWidget.
        The check-state capture + selection-count + label-lookup
        invariants are preserved; only the iteration shape changed
        from (row, col) tuples to list items.
        """
        # Capture current checkbox state (label → isChecked) BEFORE clear.
        # The card stores its label via item.setData(Qt.UserRole, label).
        prev_check_state = {}
        for i in range(self._pair_list.count()):
            item = self._pair_list.item(i)
            card = self._pair_list.itemWidget(item)
            if card is None or item is None:
                continue
            label = item.data(Qt.UserRole)
            cb = getattr(card, "_checkbox", None)
            if cb is not None and label is not None:
                prev_check_state[label] = cb.isChecked()
        self._prev_check_state = prev_check_state

        pairs = getattr(self, "_all_pairs_for_round", []) or []
        status_filter = self._status_filter_combo.currentData()
        if isinstance(status_filter, set):
            visible_pairs = [p for p in pairs
                              if p.get("status", "accepted") in status_filter]
        else:
            visible_pairs = pairs  # "All" → no filter
        self._current_source_pairs = visible_pairs  # what import sees on accept

        # Audit-fix 2026-05-24 (B): empty-filter hint so user sees why
        # the list is blank (e.g., "Finalized only" + no staged pairs).
        if not visible_pairs and pairs:
            filter_name = self._status_filter_combo.currentText()
            self._selection_count_label.setText(
                f"0 / 0 matching '{filter_name}'  ·  "
                f"{len(pairs)} pair(s) hidden by filter — try widening above"
            )
        # Phase 8d Tier 3 polish 2026-05-24: thumbnails for img + mask
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail, make_mask_thumbnail,
        )
        from napari_epicycle.ui_shell.theme import (
            make_pair_card, PAIR_STATUS_BADGE_COLORS,
        )

        self._pair_list.clear()
        for p in visible_pairs:
            label = p["label"]
            img_thumb = make_image_thumbnail(
                p["img_path"], size=72, channel_axis="first",
            )
            mask_thumb = make_mask_thumbnail(p["mask_path"], size=72)

            status_raw = p.get("status", "accepted")
            # 2026-05-25 Group F: render internal status string as a
            # user-facing label.  Internal value (predicted/curating/
            # accepted/staged/committed) goes via _STATUS_DISPLAY_MAP to
            # match the vocabulary in the status-filter combo above the
            # list ("Finalized only (staged + committed)").  Previous
            # behavior showed the raw internal string which confused
            # users — they'd click "Finalize Accepted" expecting the
            # column to say "Finalized" but saw "staged" instead.
            status_display = _STATUS_DISPLAY_MAP.get(status_raw, status_raw)
            badge_color = PAIR_STATUS_BADGE_COLORS.get(status_raw, "#888888")
            size_str = _fmt_bytes(p.get("size_bytes", 0))
            meta = f"{status_display}  ·  {size_str}"

            prev = self._prev_check_state.get(label, True)
            card, checkbox = make_pair_card(
                title=label,
                meta=meta,
                img_pixmap=img_thumb,
                mask_pixmap=mask_thumb,
                badge_text=status_display,
                badge_color_hex=badge_color,
                checked=bool(prev),
                thumb_size=72,
                parent=self._pair_list,
            )
            checkbox.stateChanged.connect(self._update_selection_count)
            card.setToolTip(
                f"Image: {p['img_path']}\nMask: {p['mask_path']}\n"
                f"Internal status: {status_raw}"
            )

            item = QListWidgetItem(self._pair_list)
            item.setSizeHint(card.sizeHint())
            item.setData(Qt.UserRole, label)
            self._pair_list.addItem(item)
            self._pair_list.setItemWidget(item, card)

        self._update_selection_count()

    def _update_selection_count(self, *args):
        count = sum(1 for cb in self._iter_checkboxes() if cb.isChecked())
        total = self._pair_list.count()
        self._selection_count_label.setText(f"{count} / {total} selected")
        self._import_btn.setEnabled(count > 0)

    def _iter_checkboxes(self):
        for i in range(self._pair_list.count()):
            item = self._pair_list.item(i)
            card = self._pair_list.itemWidget(item)
            cb = getattr(card, "_checkbox", None) if card else None
            if cb is not None:
                yield cb

    def _select_all(self):
        for cb in self._iter_checkboxes():
            cb.setChecked(True)

    def _select_none(self):
        for cb in self._iter_checkboxes():
            cb.setChecked(False)

    # ------------------------------------------------------------------
    # Signals + submit
    # ------------------------------------------------------------------

    def _wire_signals(self):
        self._source_run_combo.currentIndexChanged.connect(self._on_source_run_changed)
        self._source_round_combo.currentIndexChanged.connect(self._on_source_round_changed)
        self._status_filter_combo.currentIndexChanged.connect(
            self._on_status_filter_changed
        )
        self._select_all_btn.clicked.connect(self._select_all)
        self._select_none_btn.clicked.connect(self._select_none)
        self._cancel_btn.clicked.connect(self.reject)
        self._import_btn.clicked.connect(self._on_import_clicked)

    def _on_status_filter_changed(self, *_args):
        """Re-render the pair table when the user changes the status filter."""
        if hasattr(self, "_all_pairs_for_round"):
            self._render_pair_table()

    def _on_import_clicked(self):
        # Collect selected labels.  Each card's item stores its label
        # via setData(Qt.UserRole, label) at render time — no column
        # lookup needed post-card-list-conversion.
        selected_labels: List[str] = []
        for i in range(self._pair_list.count()):
            item = self._pair_list.item(i)
            card = self._pair_list.itemWidget(item)
            cb = getattr(card, "_checkbox", None) if card else None
            if cb is not None and cb.isChecked():
                label = item.data(Qt.UserRole)
                if label:
                    selected_labels.append(label)
        if not selected_labels:
            return

        round_dict = self._source_round_combo.currentData()
        if not round_dict:
            return

        source_bucket = round_dict["bucket"]
        target_bucket = self._target_bucket_combo.currentText()

        # Cross-bucket warning (audit-found CRITICAL 2026-05-23): lateral
        # slices are 2D XY, axial are XZ/YZ — copying across buckets would
        # silently corrupt the training dataset (Cellpose would train on
        # mis-oriented data); views map XY -> lateral and XZ/YZ -> one pooled
        # axial bucket.  Block with explicit confirm to prevent foot-guns.
        if source_bucket != target_bucket:
            from qtpy.QtWidgets import QMessageBox
            confirmed = self._confirm_cross_bucket(source_bucket, target_bucket)
            if not confirmed:
                return

        self._result = {
            "source_run_id": self._source_run_combo.currentData(),
            "source_bucket": source_bucket,
            "source_round_idx": round_dict["round_idx"],
            "target_bucket": target_bucket,
            "target_round_idx": int(self._target_round_spin.value()),
            "selected_labels": selected_labels,
            "overwrite_collisions": self._overwrite_check.isChecked(),
        }
        self.accept()

    def _confirm_cross_bucket(self, source_bucket: str, target_bucket: str) -> bool:
        """Hard-block confirm for cross-bucket import (lateral ↔ axial)."""
        from napari_epicycle.ui_shell.theme import (
            _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Cross-bucket import — are you sure?")
        dlg.setStyleSheet(_DIALOG_STYLE)
        dlg.setMinimumWidth(480)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        title = QLabel(
            f"⚠  Source = {source_bucket}, Target = {target_bucket}",
            parent=dlg,
        )
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #e57373;")
        layout.addWidget(title)
        body = QLabel(
            "Lateral pairs contain XY slices; axial pairs contain XZ/YZ slices.  "
            "Copying lateral → axial (or vice versa) means Cellpose will train on "
            "mis-oriented data, silently corrupting the dataset.\n\n"
            "Only proceed if you understand the implication — e.g., you're "
            "manually augmenting an axial set with cross-orientation samples "
            "as a known experiment.",
            parent=dlg,
        )
        body.setWordWrap(True)
        body.setStyleSheet("color: #cccccc; font-size: 11px;")
        layout.addWidget(body)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("Cancel", parent=dlg)
        cancel_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
        cancel_btn.setDefault(True)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel_btn)
        proceed_btn = QPushButton("Proceed anyway", parent=dlg)
        proceed_btn.setStyleSheet(_DIALOG_BTN_CONFIRM)
        proceed_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(proceed_btn)
        layout.addLayout(btn_row)
        return dlg.exec_() == QDialog.Accepted

    def get_result(self) -> Optional[dict]:
        return self._result


def _fmt_bytes(n: int) -> str:
    """Human-friendly byte formatting."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n // 1024} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.1f} GB"


# 2026-05-25 Group F: user-facing labels for the internal sample_records
# status strings.  Mirrors the vocabulary used in the dialog's status-
# filter combo at line ~175 ("Finalized only (staged + committed)").
# Internal values stay as-is in sample_records (we don't rename the
# status machine); only the displayed string changes.  Tooltip on each
# Status cell shows the raw internal value for power-users.
_STATUS_DISPLAY_MAP = {
    "predicted": "Predicted",
    "curating":  "Curating",
    "accepted":  "Accepted",
    "staged":    "Finalized",
    "committed": "Finalized (trained)",
}


__all__ = ["ImportPairsDialog"]

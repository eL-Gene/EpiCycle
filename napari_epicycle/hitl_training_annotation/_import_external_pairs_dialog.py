"""Phase 8f — themed Qt dialog for importing external 2D image/mask pairs.

User flow:
1. Click `[📥 Import External]` in Curation header
2. Folder picker → user picks source folder
3. Suffix inputs: img + mask suffix (text fields with [Auto] suggest button)
4. Live preview: count of detected pairs + orphans + sample basenames
5. Target: bucket + round picker (defaults to active bucket's current round)
6. Optional: relabel basenames to EpiCycle convention (default ON)
7. Optional: overwrite collisions
8. Import → backend copies + writes inherited_from_external provenance

Per user feedback 2026-05-24: suffix is user-specified (not auto-detected
silently) for safety, with optional [Auto] button to suggest a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import _import_external_pairs as _ext


class ImportExternalPairsDialog(QDialog):
    """Dialog for importing external image/mask pairs into a run."""

    def __init__(
        self,
        *,
        active_run_id: str,
        suggest_target_round_callback: Optional[Callable[[str], int]] = None,
        active_display_name: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._active_run_id = active_run_id
        # 2026-05-25 terminology sweep: prefer display_name in subtitle.
        _dn = (active_display_name or "").strip()
        self._active_display_name = _dn if _dn else active_run_id
        self._suggest_target_round = suggest_target_round_callback
        self._result: Optional[dict] = None
        self._current_folder: str = ""
        self._current_pairs: List[Dict] = []

        # 2026-05-25 Group G: 300ms debounce on suffix-field typing.
        # Previously every keystroke triggered a full _rescan_pairs +
        # thumbnail rebuild.  Typing "img" decoded everything 3x.  Now:
        # consecutive keystrokes collapse to a single rescan after the
        # user pauses for 300ms.  setSingleShot + start() resets the
        # timer on each call — standard Qt debounce idiom.
        from qtpy.QtCore import QTimer
        self._suffix_debounce_timer = QTimer(self)
        self._suffix_debounce_timer.setSingleShot(True)
        self._suffix_debounce_timer.timeout.connect(self._rescan_pairs)

        from napari_epicycle.ui_shell.theme import (
            DIALOG_FORM_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        self._cancel_style = _DIALOG_BTN_CANCEL
        self._confirm_style = _DIALOG_BTN_CONFIRM

        self.setWindowTitle("Import External Pairs")
        self.setStyleSheet(DIALOG_FORM_STYLE)
        self.setMinimumWidth(640)
        self.setMinimumHeight(580)

        self._build_ui()
        self._wire_signals()
        # Initial state — buttons disabled until folder picked
        self._update_import_state()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        title = QLabel("Import External Pairs", parent=self)
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #eeeeee;")
        root.addWidget(title)

        # 2026-05-25 terminology sweep: prefer display_name + handle as parens.
        if self._active_display_name != self._active_run_id:
            _target_str = f"{self._active_display_name} (handle: {self._active_run_id})"
        else:
            _target_str = self._active_run_id
        subtitle = QLabel(
            f"Import 2D image+mask pairs from outside EpiCycle into run "
            f"{_target_str}.  Useful when you have curated "
            f"training data from another tool and want to reuse it here.",
            parent=self,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        root.addWidget(subtitle)

        # 2026-05-26 theming sweep: each major section wrapped in
        # QGroupBox styled via DIALOG_FORM_STYLE at dialog level.
        # Three logical groups: Source / Detected pairs / Target.

        # ===== Source (folder + pair convention + channel layout) =====
        src_group = QGroupBox("Source", parent=self)
        src_v = QVBoxLayout(src_group)
        src_v.setContentsMargins(8, 14, 8, 8)
        src_v.setSpacing(6)

        folder_row = QHBoxLayout()
        self._folder_label = QLabel("(no folder selected)", parent=src_group)
        self._folder_label.setStyleSheet(
            "color: #aaccee; font-size: 11px; "
            "font-family: 'Courier New', monospace;"
        )
        self._folder_label.setWordWrap(True)
        folder_row.addWidget(self._folder_label, 1)
        self._browse_btn = QPushButton("Browse...", parent=src_group)
        self._browse_btn.setStyleSheet(self._cancel_style)
        folder_row.addWidget(self._browse_btn)
        src_v.addLayout(folder_row)

        conv_form = QFormLayout()
        conv_form.setSpacing(6)
        img_row = QHBoxLayout()
        self._img_suffix_edit = QLineEdit("_img.tif", parent=src_group)
        self._img_suffix_edit.setPlaceholderText("e.g. _img.tif, _raw.tif, .tif")
        img_row.addWidget(self._img_suffix_edit, 1)
        self._auto_btn = QPushButton("Auto", parent=src_group)
        self._auto_btn.setStyleSheet(self._cancel_style)
        self._auto_btn.setToolTip(
            "Scan the source folder + suggest the most likely suffix "
            "convention based on what's there."
        )
        img_row.addWidget(self._auto_btn)
        conv_form.addRow("Image suffix:", img_row)

        self._mask_suffix_edit = QLineEdit("_mask.tif", parent=src_group)
        self._mask_suffix_edit.setPlaceholderText("e.g. _mask.tif, _label.tif, _gt.tif")
        conv_form.addRow("Mask suffix:", self._mask_suffix_edit)
        src_v.addLayout(conv_form)

        conv_hint = QLabel(
            "Pairs are detected by basename matching: e.g. for img suffix "
            "'_img.tif' + mask suffix '_mask.tif', file 'patch_42_img.tif' "
            "pairs with 'patch_42_mask.tif'.",
            parent=src_group,
        )
        conv_hint.setWordWrap(True)
        conv_hint.setStyleSheet("color: #888888; font-size: 10px; font-style: italic;")
        src_v.addWidget(conv_hint)

        # Channel layout (Phase 8f Tier 2 polish 2026-05-24) — channel-axis
        # radio.  EpiCycle is channel-first per controller.py:449
        # (n_channels = input_stack.shape[0]).
        from qtpy.QtWidgets import QButtonGroup
        ch_form = QFormLayout()
        ch_form.setSpacing(6)
        self._channel_axis_group = QButtonGroup(self)
        ch_axis_row = QHBoxLayout()
        self._channel_axis_first = QRadioButton(
            "Channels first (C, H, W)", parent=src_group
        )
        self._channel_axis_first.setChecked(True)
        self._channel_axis_first.setToolTip(
            "Your source images have channels as the FIRST axis — e.g., "
            "shape (2, 1024, 1024) for [nuclei, membrane].  This is "
            "EpiCycle's native convention; no transpose needed."
        )
        self._channel_axis_last = QRadioButton(
            "Channels last (H, W, C)", parent=src_group
        )
        self._channel_axis_last.setToolTip(
            "Your source images have channels as the LAST axis — e.g., "
            "shape (1024, 1024, 3) for RGB or similar.  Common from "
            "QuPath / Fiji exports.  EpiCycle will transpose each image "
            "to channels-first on copy."
        )
        self._channel_axis_none = QRadioButton(
            "2D / grayscale (no channel axis)", parent=src_group
        )
        self._channel_axis_none.setToolTip(
            "Your source images are single-channel 2D arrays — e.g., "
            "shape (1024, 1024).  No transpose needed."
        )
        for rb in (self._channel_axis_first,
                   self._channel_axis_last,
                   self._channel_axis_none):
            self._channel_axis_group.addButton(rb)
            ch_axis_row.addWidget(rb)
        ch_axis_row.addStretch(1)
        ch_axis_widget = QWidget(parent=src_group)
        ch_axis_widget.setStyleSheet("background: transparent;")
        ch_axis_widget.setLayout(ch_axis_row)
        ch_form.addRow("Source channel axis:", ch_axis_widget)
        src_v.addLayout(ch_form)
        ch_hint = QLabel(
            "EpiCycle's Cellpose pipeline expects channels-first.  "
            "Channel-last sources are auto-transposed on copy.  Detected "
            "shape per pair is shown in the Dimensions column below.",
            parent=src_group,
        )
        ch_hint.setWordWrap(True)
        ch_hint.setStyleSheet("color: #888888; font-size: 10px; font-style: italic;")
        src_v.addWidget(ch_hint)

        root.addWidget(src_group)

        # ===== Detected pairs =====
        pairs_group = QGroupBox("Detected pairs", parent=self)
        pairs_v = QVBoxLayout(pairs_group)
        pairs_v.setContentsMargins(8, 14, 8, 8)
        pairs_v.setSpacing(6)

        self._preview_status = QLabel(
            "Pick a source folder to scan for pairs.", parent=pairs_group,
        )
        self._preview_status.setStyleSheet("color: #aaccee; font-size: 11px;")
        pairs_v.addWidget(self._preview_status)

        # 2026-05-25 Group E: per-row checkboxes (parity with cherry-pick).
        select_row = QHBoxLayout()
        self._select_all_btn = QPushButton("Select All", parent=pairs_group)
        self._select_all_btn.setStyleSheet(self._cancel_style)
        self._select_none_btn = QPushButton("Select None", parent=pairs_group)
        self._select_none_btn.setStyleSheet(self._cancel_style)
        self._selection_count_label = QLabel("0 selected", parent=pairs_group)
        self._selection_count_label.setStyleSheet(
            "color: #aaccee; font-size: 11px; padding-left: 8px;"
        )
        select_row.addWidget(self._select_all_btn)
        select_row.addWidget(self._select_none_btn)
        select_row.addWidget(self._selection_count_label)
        select_row.addStretch(1)
        pairs_v.addLayout(select_row)

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
        target_group = QGroupBox(
            f"Target — '{self._active_run_id}'", parent=self,
        )
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

        self._relabel_check = QCheckBox(
            "Rename pairs to EpiCycle's filename convention (recommended)",
            parent=target_group,
        )
        self._relabel_check.setChecked(True)
        self._relabel_check.setToolTip(
            "EpiCycle's slice list expects filenames like "
            "'{stack}_{VIEW}_z{N:04d}_img.tif'.  With this ON (default), "
            "your source pairs are copied with renamed basenames like "
            "'ext_{your_basename}_XY_z0000_img.tif' so they parse + show "
            "up correctly in the Slices list.\n\n"
            "Uncheck only if your source filenames ALREADY follow the "
            "EpiCycle pattern (rare for data from outside the plugin)."
        )
        target_form.addRow("", self._relabel_check)

        self._overwrite_check = QCheckBox(
            "Overwrite existing pairs (default: skip duplicates)",
            parent=target_group,
        )
        self._overwrite_check.setToolTip(
            "If a pair with the same target label already exists in the "
            "destination round (from a previous import or native curation), "
            "the default is to SKIP it (preserve what's there).  Check this "
            "to overwrite the existing pair with the source's version."
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
            "font-weight: bold; font-size: 12px; color: #d4a843; "
            "padding-top: 4px;"
        )
        return lbl

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _wire_signals(self):
        self._browse_btn.clicked.connect(self._on_browse_clicked)
        self._auto_btn.clicked.connect(self._on_auto_clicked)
        self._img_suffix_edit.textChanged.connect(self._on_suffix_changed)
        self._mask_suffix_edit.textChanged.connect(self._on_suffix_changed)
        # Audit-fix 2026-05-24 (A): channel-axis radio change should
        # re-render thumbnails so the preview matches what'll be copied.
        self._channel_axis_group.buttonClicked.connect(
            lambda *_: self._rescan_pairs() if self._current_folder else None
        )
        self._target_bucket_combo.currentTextChanged.connect(
            self._on_target_bucket_changed
        )
        self._cancel_btn.clicked.connect(self.reject)
        self._import_btn.clicked.connect(self._on_import_clicked)
        # 2026-05-25 Group E: per-row selection.  2026-05-26 theming
        # sweep: signals now wired per-card via stateChanged in the
        # populate loop, not a single itemChanged on the table.
        self._select_all_btn.clicked.connect(self._on_select_all)
        self._select_none_btn.clicked.connect(self._on_select_none)

    def _iter_card_checkboxes(self):
        """Yield (item, checkbox) for every card in _pair_list."""
        for i in range(self._pair_list.count()):
            item = self._pair_list.item(i)
            card = self._pair_list.itemWidget(item)
            cb = getattr(card, "_checkbox", None) if card else None
            if cb is not None:
                yield item, cb

    def _on_select_all(self):
        """Check every card."""
        for _item, cb in self._iter_card_checkboxes():
            cb.setChecked(True)
        self._update_selection_count()
        self._update_import_state()

    def _on_select_none(self):
        """Uncheck every card."""
        for _item, cb in self._iter_card_checkboxes():
            cb.setChecked(False)
        self._update_selection_count()
        self._update_import_state()

    def _on_pair_check_changed(self, *_args):
        """Wired per-card from stateChanged in populate loop."""
        self._update_selection_count()
        self._update_import_state()

    def _update_selection_count(self):
        """Refresh the '<N>/<total> selected' label."""
        total = self._pair_list.count()
        n_checked = sum(1 for _i, cb in self._iter_card_checkboxes() if cb.isChecked())
        self._selection_count_label.setText(
            f"{n_checked}/{total} selected"
        )

    def _on_browse_clicked(self):
        from qtpy.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(
            self, "Pick source folder", ""
        )
        if not folder:
            return
        self._current_folder = folder
        self._folder_label.setText(folder)
        self._folder_label.setToolTip(folder)
        self._rescan_pairs()

    def _on_auto_clicked(self):
        if not self._current_folder:
            self._preview_status.setText(
                "Pick a source folder first, then click Auto."
            )
            return
        try:
            img_suf, mask_suf = _ext.suggest_pair_convention(self._current_folder)
            self._img_suffix_edit.setText(img_suf)
            self._mask_suffix_edit.setText(mask_suf)
            # _on_suffix_changed will fire from setText and trigger rescan
        except Exception as e:
            self._preview_status.setText(f"Auto-detect failed: {e}")

    def _on_suffix_changed(self, _text: str):
        # 2026-05-25 Group G: route through the 300ms debounce timer so
        # typing "img" triggers a single rescan, not three.  start()
        # resets the timer if already running — that's the debounce.
        # No-op when no folder is picked (rescan would be a no-op anyway).
        if self._current_folder:
            self._suffix_debounce_timer.start(300)

    def done(self, result):
        """Stop the debounce timer before the dialog closes.

        2026-05-25 audit-A HIGH fix: without this, if the user types
        in the suffix field and clicks Cancel within 300ms, the
        QTimer fires AFTER the dialog has been hidden/destroyed and
        calls `_rescan_pairs` on a stale Qt object — RuntimeError on
        the C++ side ("wrapped C/C++ object has been deleted") or
        silent stale widget reads.  Override both accept + reject by
        gating QDialog.done() which is called by both paths.
        """
        try:
            if self._suffix_debounce_timer.isActive():
                self._suffix_debounce_timer.stop()
        except Exception as e:
            pass
        super().done(result)

    def _on_target_bucket_changed(self, bucket_text: str):
        if self._suggest_target_round is not None:
            try:
                suggested = self._suggest_target_round(bucket_text)
                if isinstance(suggested, int) and 0 <= suggested <= 99:
                    self._target_round_spin.setValue(suggested)
            except Exception:
                pass

    def _rescan_pairs(self):
        """Re-scan the source folder + update preview table.

        2026-05-25 audit-A MEDIUM fix: re-entry guard.  The progress-
        feedback `processEvents()` call inside the table-build loop can
        deliver new signals (suffix textChanged, button clicks, etc.).
        Without this guard, a new `_rescan_pairs` could start while the
        previous one is mid-build → corrupted table state, double row
        counts, or `_current_pairs` reset mid-iteration.  The guard
        makes the second call a no-op; the debounce will re-fire after
        the current call returns.
        """
        if getattr(self, "_populating_pair_table", False):
            return
        img_suf = self._img_suffix_edit.text().strip()
        mask_suf = self._mask_suffix_edit.text().strip()
        if not img_suf or not mask_suf:
            self._preview_status.setText(
                "Enter both image + mask suffixes."
            )
            self._pair_list.clear()
            self._current_pairs = []
            self._update_import_state()
            return
        if img_suf == mask_suf:
            self._preview_status.setText(
                "⚠ Image + mask suffixes must differ."
            )
            self._pair_list.clear()
            self._current_pairs = []
            self._update_import_state()
            return

        try:
            pairs, orphan_imgs, orphan_masks = _ext.enumerate_pairs(
                self._current_folder, img_suf, mask_suf,
            )
        except Exception as e:
            self._preview_status.setText(f"Scan failed: {e}")
            self._pair_list.clear()
            self._current_pairs = []
            self._update_import_state()
            return

        self._current_pairs = pairs
        n_pairs = len(pairs)
        n_orphans = len(orphan_imgs) + len(orphan_masks)
        status_bits = [f"{n_pairs} pair(s) detected"]
        if orphan_imgs:
            status_bits.append(f"{len(orphan_imgs)} orphan image(s)")
        if orphan_masks:
            status_bits.append(f"{len(orphan_masks)} orphan mask(s)")
        self._preview_status.setText("   ·   ".join(status_bits))

        # Audit-fix 2026-05-24 (G): heuristic warning if the detected shape
        # of the first pair doesn't match the chosen channel-axis radio.
        # Prevents silent training-data corruption from a wrong axis pick.
        if pairs:
            axis_warning = self._check_axis_consistency(pairs[0]["img_path"])
            if axis_warning:
                self._preview_status.setText(
                    self._preview_status.text() + "\n" + axis_warning
                )
                self._preview_status.setStyleSheet(
                    "color: #d4a843; font-size: 11px; font-weight: bold;"
                )
            else:
                self._preview_status.setStyleSheet(
                    "color: #aaccee; font-size: 11px;"
                )

        # Populate list — 2026-05-25 Group G: NO row cap (was [:50] —
        # which silently dropped pairs 51+ from import despite showing
        # "60 detected").  Thumbnail cache in _thumbnails.py handles
        # the resulting cold-cache cost (~10-20ms/pair first time,
        # cache-hit instant on rescan).
        # 2026-05-26 theming sweep: render target swapped to QListWidget
        # of pair cards; checkbox state lives on each card's QCheckBox,
        # wired via stateChanged → _on_pair_check_changed.
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail, make_mask_thumbnail,
        )
        from napari_epicycle.ui_shell.theme import (
            make_pair_card, PAIR_STATUS_BADGE_COLORS,
        )
        from qtpy.QtWidgets import QApplication as _QApp
        n_pairs = len(pairs)
        self._pair_list.clear()
        # Determine channel_axis from the current radio selection so the
        # thumbnail picks the right channel for display.  Default "first".
        if hasattr(self, "_channel_axis_first") and self._channel_axis_last.isChecked():
            thumb_channel_axis = "last"
        elif hasattr(self, "_channel_axis_none") and self._channel_axis_none.isChecked():
            thumb_channel_axis = "none"
        else:
            thumb_channel_axis = "first"
        # 2026-05-25 Group G: progress feedback for cold-cache folders.
        # For folders with >50 pairs, periodically update the status
        # label + processEvents so the UI doesn't appear frozen.
        _progress_status_base = self._preview_status.text()
        _progress_threshold = 50
        # 2026-05-25 audit-A MEDIUM: _populating_pair_table is a re-entry
        # guard for _rescan_pairs (still fires from suffix-debounce while
        # processEvents() is running here).  Preserved post-card-list
        # conversion since the guard is in _rescan_pairs (line top), not
        # the populate loop itself.
        self._populating_pair_table = True
        try:
            for row, p in enumerate(pairs):
                img_name = Path(p["img_path"]).name
                mask_name = Path(p["mask_path"]).name
                dims_str = self._read_dims(p["img_path"])
                img_thumb = make_image_thumbnail(
                    p["img_path"], size=72, channel_axis=thumb_channel_axis,
                )
                mask_thumb = make_mask_thumbnail(p["mask_path"], size=72)

                # External-import pairs are always rendered with the
                # external badge — they're not yet tagged with a workflow
                # status (that happens post-import).  Soft-purple tint
                # matches PAIR_STATUS_BADGE_COLORS["external"].
                badge_color = PAIR_STATUS_BADGE_COLORS["external"]
                meta = f"{dims_str}"

                card, checkbox = make_pair_card(
                    title=p["basename"],
                    meta=meta,
                    img_pixmap=img_thumb,
                    mask_pixmap=mask_thumb,
                    badge_text="external",
                    badge_color_hex=badge_color,
                    checked=True,  # default checked = natural select-all start
                    thumb_size=72,
                    parent=self._pair_list,
                )
                checkbox.stateChanged.connect(self._on_pair_check_changed)
                card.setToolTip(
                    f"Image: {img_name}\nMask: {mask_name}\nDimensions: {dims_str}"
                )

                item = QListWidgetItem(self._pair_list)
                item.setSizeHint(card.sizeHint())
                # Stash the row index so _on_import_clicked can map
                # back to self._current_pairs[i] without iterating cards.
                item.setData(Qt.UserRole, row)
                self._pair_list.addItem(item)
                self._pair_list.setItemWidget(item, card)

                # Progress feedback: only for large folders, every 25 rows
                if n_pairs > _progress_threshold and (row + 1) % 25 == 0:
                    self._preview_status.setText(
                        f"{_progress_status_base}  (decoding "
                        f"thumbnails: {row + 1}/{n_pairs})"
                    )
                    _QApp.processEvents()
        finally:
            self._populating_pair_table = False
        # Restore final status text after the progress updates.
        if n_pairs > _progress_threshold:
            self._preview_status.setText(_progress_status_base)
        self._update_selection_count()

        self._update_import_state()

    def _check_axis_consistency(self, img_path: str) -> str:
        """Audit-fix 2026-05-24 (G): heuristic warn if detected shape
        doesn't match the selected channel-axis radio.

        Returns:
            Empty string if shape is consistent with selected axis.
            Warning string otherwise (rendered amber in the preview).
        """
        try:
            import tifffile
            with tifffile.TiffFile(img_path) as tif:
                shape = tif.pages[0].shape
        except Exception:
            return ""
        if self._channel_axis_first.isChecked():
            chosen = "first"
        elif self._channel_axis_last.isChecked():
            chosen = "last"
        else:
            chosen = "none"

        # Heuristic: channel axis is the SMALLEST dim of a 3D array.
        # For 2D arrays, any declaration is fine (no channels to interpret).
        if len(shape) == 2:
            # Mildly nudge user toward 2D radio if they picked first/last,
            # but don't block — 2D pass-through is harmless either way.
            if chosen == "first" or chosen == "last":
                return (
                    f"ℹ  Source is 2D ({tuple(shape)}) — consider picking "
                    f"'2D / grayscale' for clarity (but 'Channels {chosen}' "
                    f"will also work since there are no channels to "
                    f"transpose)."
                )
            return ""
        if len(shape) == 3:
            if chosen == "none":
                return (
                    f"⚠  Source has 3 dims ({tuple(shape)}) but you picked "
                    f"'2D / grayscale'.  Pick 'Channels first' or "
                    f"'Channels last' depending on layout."
                )
            # Smallest dim heuristic — channels usually 1-5, spatial much larger
            smallest = min(shape)
            smallest_idx = shape.index(smallest)
            # If chosen='first' but smallest is at index -1, likely mis-picked
            if chosen == "first" and smallest_idx == len(shape) - 1 and smallest <= 10:
                return (
                    f"⚠  Detected shape {tuple(shape)} — channels look "
                    f"like the LAST axis (size {smallest}), but you picked "
                    f"'Channels first'.  Check the radio above."
                )
            if chosen == "last" and smallest_idx == 0 and smallest <= 10:
                return (
                    f"⚠  Detected shape {tuple(shape)} — channels look "
                    f"like the FIRST axis (size {smallest}), but you picked "
                    f"'Channels last'.  Check the radio above."
                )
            return ""
        if len(shape) > 3:
            return (
                f"⚠  Source has {len(shape)} dimensions ({tuple(shape)}).  "
                f"EpiCycle's 2D Cellpose pipeline expects 2D (H, W) or 3D "
                f"(C, H, W) input.  Volumetric data may not import correctly."
            )
        return ""

    def _read_dims(self, tiff_path: str) -> str:
        """Read tiff shape via header only (fast — no full image load).

        Returns a string like "(2, 1024, 1024) uint16" for display in the
        Dimensions column.  Falls back to '?' on read failure.
        """
        try:
            import tifffile
            with tifffile.TiffFile(tiff_path) as tif:
                shape = tif.pages[0].shape
                dtype = tif.pages[0].dtype
            return f"{tuple(shape)} {dtype}"
        except Exception:
            return "?"

    def _update_import_state(self):
        # 2026-05-25 Group E: require >=1 checked pair, not just >=1 detected.
        n_checked = sum(1 for _i, cb in self._iter_card_checkboxes() if cb.isChecked())
        ok = (
            bool(self._current_folder)
            and n_checked > 0
            and self._img_suffix_edit.text().strip()
            and self._mask_suffix_edit.text().strip()
            and self._img_suffix_edit.text() != self._mask_suffix_edit.text()
        )
        self._import_btn.setEnabled(ok)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def _on_import_clicked(self):
        if not self._current_pairs:
            return
        # Phase 8f Tier 2 polish 2026-05-24: read channel-axis radio
        if self._channel_axis_first.isChecked():
            source_channel_axis = "first"
        elif self._channel_axis_last.isChecked():
            source_channel_axis = "last"
        else:
            source_channel_axis = "none"
        target_bucket = self._target_bucket_combo.currentText()

        # 2026-05-25 Group E: filter to checked-only.  2026-05-26 theming
        # sweep: read checkbox state from cards.  Each card item stores
        # its row index (in self._current_pairs) via setData(Qt.UserRole)
        # at populate time — same mapping invariant as the pre-card
        # table but explicit instead of implicit.
        selected_pairs = []
        for item, cb in self._iter_card_checkboxes():
            if not cb.isChecked():
                continue
            row_idx = item.data(Qt.UserRole)
            if isinstance(row_idx, int) and 0 <= row_idx < len(self._current_pairs):
                selected_pairs.append(self._current_pairs[row_idx])
        n_pairs = len(selected_pairs)
        if n_pairs == 0:
            return

        # 2026-05-25 audit CRITICAL — orientation-confirm gate.  External
        # Import has no source-bucket metadata (cherry-pick has one — the
        # source run's bucket — and we compare-and-warn on mismatch).
        # Here, the user is providing arbitrary 2D image+mask pairs from
        # an external tool.  We tag them XY (lateral bucket) or XZ
        # (axial bucket); XZ and YZ share one pooled axial bucket.  If the
        # user accidentally imports YZ slices into axial-as-XZ, Cellpose
        # trains on mis-oriented data with NO recoverable signal.
        # Mirror the cherry-pick _confirm_cross_bucket pattern: explicit
        # themed confirm with the orientation assumption spelled out.
        if not self._confirm_orientation(target_bucket, n_pairs):
            return

        self._result = {
            "folder": self._current_folder,
            "img_suffix": self._img_suffix_edit.text().strip(),
            "mask_suffix": self._mask_suffix_edit.text().strip(),
            # 2026-05-25 Group E: pass ONLY checked pairs, not all
            # detected pairs.  Previous behavior was all-or-nothing.
            "pair_records": selected_pairs,
            "target_bucket": target_bucket,
            "target_round_idx": int(self._target_round_spin.value()),
            "relabel_basename": self._relabel_check.isChecked(),
            "overwrite_collisions": self._overwrite_check.isChecked(),
            "source_channel_axis": source_channel_axis,
        }
        self.accept()

    def _confirm_orientation(self, target_bucket: str, n_pairs: int) -> bool:
        """Themed confirm dialog spelling out the orientation assumption.

        Returns True if user clicks Confirm, False if they Cancel.

        Background (2026-05-25 audit finding): External Import has no
        per-pair orientation metadata — the user's external tool may
        have produced XY, XZ, or YZ slices.  We tag the lateral bucket
        as XY, the axial bucket as XZ (views map to buckets as XY -> lateral
        and XZ/YZ -> one pooled axial bucket).  If the user imports YZ slices into the lateral
        bucket, Cellpose trains on mis-oriented data with no recovery.
        This dialog forces the user to acknowledge the assumption.
        """
        from napari_epicycle.ui_shell.theme import (
            _DIALOG_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )

        if str(target_bucket).lower() == "lateral":
            assumed_view = "XY"
            warn_kind = "lateral (XY) slices"
            wrong_kind = "XZ or YZ"
            view_explanation = "XY"
        else:
            # 2026-05-25 audit-2 L5: axial bucket pools both XZ and YZ
            # (views map XY -> lateral and XZ/YZ -> pooled axial).  Previous "tag as XZ"
            # wording was misleading for users with YZ data — the
            # underlying mapping is "axial bucket accepts XZ + YZ
            # interchangeably for training".
            assumed_view = "XZ or YZ"
            warn_kind = "axial (XZ or YZ) slices"
            wrong_kind = "XY"
            view_explanation = "axial (XZ/YZ pooled)"

        dlg = QDialog(self)
        dlg.setWindowTitle("Confirm Orientation")
        dlg.setStyleSheet(_DIALOG_STYLE)
        dlg.setMinimumWidth(480)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel(
            f"Confirm orientation: importing as {warn_kind}",
            parent=dlg,
        )
        title.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #d4a843;"
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        msg = QLabel(
            f"You're importing {n_pairs} pair(s) into the "
            f"'{target_bucket}' bucket.  EpiCycle will tag them as "
            f"<b>{view_explanation}</b> slices (per the locked view "
            f"routing: lateral=XY, axial=XZ/YZ pooled).",
            parent=dlg,
        )
        msg.setWordWrap(True)
        msg.setTextFormat(Qt.RichText)
        msg.setStyleSheet("color: #dddddd; font-size: 11px; padding: 4px 0;")
        layout.addWidget(msg)

        warn = QLabel(
            f"⚠  If these pairs are actually <b>{wrong_kind}</b> slices, "
            f"Cellpose will train on mis-oriented data with no warning "
            f"and the resulting model will be broken.  There is no "
            f"automatic recovery — you'd have to discard the round and "
            f"retrain from scratch.",
            parent=dlg,
        )
        warn.setWordWrap(True)
        warn.setTextFormat(Qt.RichText)
        warn.setStyleSheet(
            "color: #e57373; font-size: 11px; padding: 4px 0;"
        )
        layout.addWidget(warn)

        info = QLabel(
            f"Click Confirm only if you're certain these are "
            f"{assumed_view} slices.",
            parent=dlg,
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "color: #888888; font-size: 10px; font-style: italic; "
            "padding-top: 6px;"
        )
        layout.addWidget(info)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("Cancel", parent=dlg)
        cancel_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
        # 2026-05-25 audit-2 H2 fix: Cancel is the safe default so
        # pressing Enter on a half-read dialog does NOT trigger the
        # destructive import action.  Previously confirm_btn was
        # setDefault(True), which silently turned Enter into "yes, run
        # Cellpose on the mis-oriented slices".
        cancel_btn.setDefault(True)
        cancel_btn.setAutoDefault(True)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel_btn)
        confirm_btn = QPushButton(
            f"Confirm — these are {view_explanation} slices", parent=dlg,
        )
        confirm_btn.setStyleSheet(_DIALOG_BTN_CONFIRM)
        confirm_btn.setAutoDefault(False)
        confirm_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(confirm_btn)
        layout.addLayout(btn_row)

        return dlg.exec_() == QDialog.Accepted

    def get_result(self) -> Optional[dict]:
        return self._result


__all__ = ["ImportExternalPairsDialog"]

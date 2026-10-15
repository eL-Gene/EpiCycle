"""Floating merge dialog for the 3D curation workflow.

Replaces the bare ``QInputDialog.getInt()`` previously used in the
Label Navigator's "Merge Into..." action. Surfaces adjacent labels
(discovered via graduated dilation), supports multi-select, and
runs through a two-button (Merge Selected → Accept) flow with full
ROI snapshot rollback on Cancel.

The Navigator asks the cross-op gating predicate
(``can_open_merge_dialog``) before opening this dialog.

Architecture:
- ``MergeDialogState`` is a Qt-free dataclass capturing the dialog's
  state machine (selecting → merged → accepted/cancelled). Fully
  testable in isolation.
- ``MergeDialog`` is the QWidget that owns a state instance and
  delegates state transitions to it. Qt-heavy parts (cards,
  checkboxes, button enable, keyboard shortcuts) live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QKeyEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from napari_epicycle.annotator.curation_label_navigator import TIER_COLORS
from napari_epicycle.ui_shell.theme import (
    ColorSwatch,
    PRIMARY_BTN_STYLE,
    PRIMARY_BTN_STYLE_GREEN,
    SCROLLBAR_STYLE,
    SECONDARY_BTN_STYLE,
    swatch_color_for_label,
    tier_pill_style,
)


# -----------------------------------------------------------------
# State machine — Qt-free
# -----------------------------------------------------------------


@dataclass
class MergeDialogState:
    """State machine for ``MergeDialog``.

    Captures the dialog's lifecycle (``selecting`` → ``merged``),
    the set of label IDs the user has checked for merging, and the
    snapshot needed for rollback if the user clicks Cancel post-merge.

    Fully Qt-free; tested in ``tests/annotator/test_merge_dialog.py``.
    """

    keep_id: int
    neighbor_info: list[dict] = field(default_factory=list)
    label_metadata: dict = field(default_factory=dict)

    checked_ids: set[int] = field(default_factory=set)
    stage: str = "selecting"  # "selecting" | "merged"
    snapshot: Any = None  # (roi_slices, old_roi_data) post-merge
    absorbed_ids: list[int] = field(default_factory=list)

    @property
    def isolation_set(self) -> set[int]:
        """Labels to isolate in viewers: keep label + currently checked."""
        return {self.keep_id} | self.checked_ids

    def can_merge_selected(self) -> bool:
        """[Merge Selected] button is enabled iff in selecting stage
        with at least one neighbor checked."""
        return self.stage == "selecting" and len(self.checked_ids) >= 1

    def can_accept(self) -> bool:
        """[Accept] button is enabled iff merge has been performed."""
        return self.stage == "merged"

    def is_empty(self) -> bool:
        """No neighbors found — dialog renders placeholder row."""
        return len(self.neighbor_info) == 0

    def toggle_check(self, label_id: int) -> None:
        """Toggle a neighbor's checked state. The keep_id cannot be
        toggled (it is always implicitly part of the isolation set
        but never an absorb target)."""
        if label_id == self.keep_id:
            return
        if label_id in self.checked_ids:
            self.checked_ids.discard(label_id)
        else:
            self.checked_ids.add(label_id)

    def transition_to_merged(self, snapshot: Any, absorbed_ids: list[int]) -> None:
        """Move from selecting → merged. Stores the rollback snapshot
        and the IDs that were absorbed."""
        if self.stage != "selecting":
            raise RuntimeError(
                f"transition_to_merged() called in stage '{self.stage}'; "
                "only 'selecting' is valid."
            )
        self.snapshot = snapshot
        self.absorbed_ids = list(absorbed_ids)
        self.stage = "merged"

    def reset_to_selecting(self) -> None:
        """Used after a Cancel-post-merge volume rollback completes."""
        self.stage = "selecting"
        self.snapshot = None
        self.absorbed_ids = []


# -----------------------------------------------------------------
# Theme
# -----------------------------------------------------------------

_STEP3_ACCENT = "#d4a843"  # amber, Step 3 (Build) — matches AccordionSection active state.

_DIALOG_BG_STYLE = (
    "QWidget#MergeDialogRoot {"
    "  background-color: #2b2b2b;"
    "  border: 1px solid rgba(255, 255, 255, 0.12);"
    "  border-radius: 6px;"
    "}"
)

_HEADER_TITLE_STYLE = (
    "font-weight: bold; font-size: 13px; color: #eeeeee;"
    " padding: 2px 0;"
)
_HEADER_HELP_STYLE = "color: #999999; font-size: 11px;"

_CARD_BASE_STYLE = (
    "QFrame#NeighborCard {"
    "  background-color: rgba(255, 255, 255, 0.03);"
    "  border: 1px solid rgba(255, 255, 255, 0.06);"
    "  border-left: 3px solid rgba(255, 255, 255, 0.06);"
    "  border-radius: 4px;"
    "}"
    "QFrame#NeighborCard:hover {"
    "  background-color: rgba(255, 255, 255, 0.06);"
    "  border: 1px solid rgba(255, 255, 255, 0.12);"
    "  border-left: 3px solid rgba(255, 255, 255, 0.18);"
    "}"
)

_CARD_CHECKED_STYLE = (
    "QFrame#NeighborCard {"
    "  background-color: rgba(212, 168, 67, 0.12);"
    "  border: 1px solid rgba(212, 168, 67, 0.20);"
    "  border-left: 3px solid #d4a843;"
    "  border-radius: 4px;"
    "}"
    "QFrame#NeighborCard:hover {"
    "  background-color: rgba(212, 168, 67, 0.18);"
    "}"
)

_CARD_FOCUS_RING_STYLE = (
    " QFrame#NeighborCard {"
    "  border-color: rgba(212, 168, 67, 0.55);"
    " }"
)

_LABEL_ID_STYLE = (
    "color: #eeeeee; font-size: 12px; font-weight: bold;"
    " font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;"
)
_META_STYLE = (
    "color: #aaaaaa; font-size: 11px;"
    " font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;"
)
_META_DIM_STYLE = (
    "color: #777777; font-size: 11px;"
    " font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;"
)
_PLACEHOLDER_STYLE = (
    "color: #777777; font-size: 11px; font-style: italic;"
    " padding: 24px;"
)

_SCROLL_AREA_STYLE = (
    "QScrollArea { background: transparent; border: none; }"
    "QScrollArea > QWidget > QWidget { background: transparent; }"
    + SCROLLBAR_STYLE
)


def _tier_pill_style_for(tier: str) -> str:
    return tier_pill_style(TIER_COLORS.get(tier, "#bbbbbb"))


# -----------------------------------------------------------------
# Neighbor card
# -----------------------------------------------------------------


class _NeighborCard(QFrame):
    """One row in the neighbor list — checkbox + swatch + ID + metadata + tier pill.

    Clicking anywhere on the card (other than the checkbox itself)
    also toggles the checkbox so the entire card is a hit target.
    """

    def __init__(
        self,
        label_id: int,
        info: dict,
        meta: dict,
        swatch_color: QColor,
        on_toggle: Callable[[int, bool], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._label_id = label_id
        self._on_toggle = on_toggle
        self._focused = False

        self.setObjectName("NeighborCard")
        self.setStyleSheet(_CARD_BASE_STYLE)
        self.setFrameShape(QFrame.NoFrame)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        # Checkbox
        self._checkbox = QCheckBox(parent=self)
        self._checkbox.toggled.connect(self._handle_checkbox_toggled)
        layout.addWidget(self._checkbox)

        # Color swatch
        self._swatch = ColorSwatch(swatch_color, parent=self)
        layout.addWidget(self._swatch)

        # ID — bold, monospace
        id_lbl = QLabel(f"ID {label_id}", parent=self)
        id_lbl.setStyleSheet(_LABEL_ID_STYLE)
        id_lbl.setMinimumWidth(70)
        layout.addWidget(id_lbl)

        # Contact voxels (NEW — was dropped in the table version)
        contact = info.get("contact_voxels")
        contact_lbl = QLabel(
            f"contact {int(contact):,}" if contact is not None else "contact —",
            parent=self,
        )
        contact_lbl.setStyleSheet(_META_STYLE)
        contact_lbl.setToolTip(
            "Number of voxel contacts between this label and the keep label."
        )
        layout.addWidget(contact_lbl)

        # Radius
        r = info.get("first_radius")
        r_lbl = QLabel(
            f"r {int(r)}" if r is not None else "r —",
            parent=self,
        )
        r_lbl.setStyleSheet(_META_STYLE)
        r_lbl.setToolTip(
            "First dilation radius (voxels) at which this label became "
            "adjacent to the keep label."
        )
        layout.addWidget(r_lbl)

        # Area
        area = meta.get("area")
        area_lbl = QLabel(
            f"{int(area):,} vox" if area is not None else "— vox",
            parent=self,
        )
        area_lbl.setStyleSheet(_META_STYLE if area is not None else _META_DIM_STYLE)
        area_lbl.setToolTip("Label volume in voxels.")
        layout.addWidget(area_lbl)

        # Centroid Z
        z = meta.get("centroid_z")
        z_lbl = QLabel(
            f"z {z:.0f}" if z is not None else "z —",
            parent=self,
        )
        z_lbl.setStyleSheet(_META_STYLE if z is not None else _META_DIM_STYLE)
        z_lbl.setToolTip("Label centroid Z slice index.")
        layout.addWidget(z_lbl)

        layout.addStretch(1)

        # Tier pill (right edge)
        tier = meta.get("tier") or "unknown"
        self._tier_lbl = QLabel(tier, parent=self)
        self._tier_lbl.setStyleSheet(_tier_pill_style_for(tier))
        self._tier_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._tier_lbl)

    # ---- public API ----

    def is_checked(self) -> bool:
        return self._checkbox.isChecked()

    def set_checked(self, checked: bool) -> None:
        # Avoid re-firing the toggle callback when we drive this
        # programmatically (e.g. keyboard Space).
        self._checkbox.blockSignals(True)
        self._checkbox.setChecked(checked)
        self._checkbox.blockSignals(False)
        self._apply_style()
        # Programmatic check still needs to notify the dialog so it
        # updates state + isolation. blockSignals only blocks the Qt
        # signal — we call the callback directly here.
        self._on_toggle(self._label_id, checked)

    def toggle(self) -> None:
        self._checkbox.toggle()  # fires toggled() → _handle_checkbox_toggled

    def set_focused(self, focused: bool) -> None:
        self._focused = focused
        self._apply_style()

    @property
    def label_id(self) -> int:
        return self._label_id

    # ---- internals ----

    def _handle_checkbox_toggled(self, checked: bool) -> None:
        self._apply_style()
        self._on_toggle(self._label_id, checked)

    def _apply_style(self) -> None:
        base = _CARD_CHECKED_STYLE if self._checkbox.isChecked() else _CARD_BASE_STYLE
        if self._focused:
            base = base + _CARD_FOCUS_RING_STYLE
        self.setStyleSheet(base)

    def mousePressEvent(self, event) -> None:
        # Click anywhere on the card toggles the checkbox.
        if event.button() == Qt.LeftButton:
            self._checkbox.toggle()
            event.accept()
            return
        super().mousePressEvent(event)


# -----------------------------------------------------------------
# Widget
# -----------------------------------------------------------------


class MergeDialog(QWidget):
    """Floating, non-modal merge dialog.

    Launched from the Label Navigator's "Merge Into..." action. Lets
    the user select adjacent labels (discovered via graduated
    dilation) to absorb into the keep label.

    The dialog is non-modal so the user can pan/zoom napari viewers
    while it is open — selection updates the viewers' isolation set
    in real time.
    """

    def __init__(
        self,
        keep_label_id: int,
        neighbor_info: list[dict],
        label_metadata: dict,
        *,
        on_merge: Callable[[int, list[int]], Any],
        on_accept: Callable[[int, list[int]], None],
        on_isolation_changed: Callable[[set[int]], None],
        on_cancel: Callable[[], None],
        main_label_layer=None,
        parent: QWidget | None = None,
    ):
        """Construct the merge dialog.

        Parameters mostly unchanged from the table-based predecessor.
        ``main_label_layer`` is new — used to sample real on-screen
        label colors for the swatch column. May be ``None`` in tests
        (swatches fall back to neutral gray).

        Callbacks (all required, all fire before any UI close):

        on_merge(keep_id, absorb_ids) -> snapshot
            Fires when the user clicks [Merge Selected]. The receiver
            does the volume relabel and returns a (roi_slices,
            old_roi_data) tuple for rollback. The dialog stores it in
            its state and transitions to the "merged" stage.

        on_accept(keep_id, absorb_ids) -> None
            Fires when the user clicks [Accept] (post-merge). The
            receiver runs the post-commit chain: controller callback
            for DataFrame updates, Navigator row removal, undo-stack
            commit. The dialog closes after this callback returns.

        on_isolation_changed(label_ids_set) -> None
            Fires whenever the checked set changes. The receiver
            updates the viewers' isolation set live.

        on_cancel() -> None
            Fires on [Cancel] or [X] close. If state is "merged" the
            receiver must restore the volume from the snapshot stored
            in dialog.state.snapshot. The dialog closes after.
        """
        super().__init__(parent)
        self.setObjectName("MergeDialogRoot")
        self.setWindowTitle(f"Merge Into Label {keep_label_id}")
        self.setWindowFlag(Qt.Tool, True)  # tool window — stays on top, no taskbar
        self.setStyleSheet(_DIALOG_BG_STYLE)
        self.setMinimumWidth(520)
        self.setMinimumHeight(280)

        self._state = MergeDialogState(
            keep_id=keep_label_id,
            neighbor_info=list(neighbor_info),
            label_metadata=dict(label_metadata),
        )
        self._on_merge_cb = on_merge
        self._on_accept_cb = on_accept
        self._on_isolation_changed_cb = on_isolation_changed
        self._on_cancel_cb = on_cancel
        self._main_label_layer = main_label_layer

        # closeEvent guard. The Accept and Cancel button handlers set
        # this to True after firing their callbacks; closeEvent then
        # treats those closes as "already handled". For ANY OTHER way
        # the dialog can close — window X button, parent destroyed,
        # napari dock close — closeEvent fires the cancel callback
        # itself so the volume is rolled back and the multi-label
        # isolation proxies are torn down. Without this guard, an
        # X-close (or parent destroy) leaves the per-label sub-proxies
        # orphaned in the viewer, which subsequent renders (especially
        # iso_categorical mode switches) can crash on.
        self._clean_close = False

        # label_id → card widget, for keyboard navigation + state lookup.
        self._cards: dict[int, _NeighborCard] = {}
        self._card_order: list[int] = []  # focus traversal order
        self._focused_idx: int = -1

        self._build_ui()

        # Force-isolate the keep label as soon as the dialog opens.
        self._fire_isolation_changed()

    # ---- UI construction --------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        # Header — keep label + swatch
        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        header_swatch = ColorSwatch(
            swatch_color_for_label(self._main_label_layer, self._state.keep_id),
            parent=self,
        )
        header_row.addWidget(header_swatch)
        header_title = QLabel(f"Merge Into Label {self._state.keep_id}", parent=self)
        header_title.setStyleSheet(_HEADER_TITLE_STYLE)
        header_row.addWidget(header_title)
        header_row.addStretch(1)
        outer.addLayout(header_row)

        # Help text
        help_label = QLabel(
            "Select adjacent labels to absorb into the keep label."
            "  ↑/↓ to navigate, Space to toggle, Esc to cancel.",
            parent=self,
        )
        help_label.setStyleSheet(_HEADER_HELP_STYLE)
        help_label.setWordWrap(True)
        outer.addWidget(help_label)

        # Card list (scrollable)
        self._scroll = QScrollArea(parent=self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(_SCROLL_AREA_STYLE)

        self._card_container = QWidget(parent=self._scroll)
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(6)
        self._card_layout.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._card_container)
        outer.addWidget(self._scroll, stretch=1)

        self._populate_cards()

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._merge_btn = QPushButton("Merge Selected", parent=self)
        self._merge_btn.setStyleSheet(PRIMARY_BTN_STYLE)
        self._merge_btn.clicked.connect(self._on_merge_selected_clicked)
        btn_row.addWidget(self._merge_btn)

        self._accept_btn = QPushButton("Accept", parent=self)
        self._accept_btn.setStyleSheet(PRIMARY_BTN_STYLE_GREEN)
        self._accept_btn.clicked.connect(self._on_accept_clicked)
        btn_row.addWidget(self._accept_btn)

        btn_row.addStretch(1)

        self._cancel_btn = QPushButton("Cancel", parent=self)
        self._cancel_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self._cancel_btn)

        outer.addLayout(btn_row)

        self._refresh_button_states()

    def _populate_cards(self) -> None:
        """Rebuild the card list from neighbor_info + label_metadata.

        If neighbor_info is empty, render a single placeholder line.
        """
        # Clear any existing cards.
        while self._card_layout.count():
            item = self._card_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._cards.clear()
        self._card_order.clear()
        self._focused_idx = -1

        if self._state.is_empty():
            placeholder = QLabel(
                "No adjacent labels found within 3 voxels.",
                parent=self._card_container,
            )
            placeholder.setStyleSheet(_PLACEHOLDER_STYLE)
            placeholder.setAlignment(Qt.AlignCenter)
            self._card_layout.addWidget(placeholder)
            return

        for info in self._state.neighbor_info:
            label_id = int(info["label_id"])
            meta = self._state.label_metadata.get(label_id, {})
            swatch_color = swatch_color_for_label(
                self._main_label_layer, label_id
            )
            card = _NeighborCard(
                label_id=label_id,
                info=info,
                meta=meta,
                swatch_color=swatch_color,
                on_toggle=self._on_card_toggled,
                parent=self._card_container,
            )
            self._card_layout.addWidget(card)
            self._cards[label_id] = card
            self._card_order.append(label_id)

        # Focus the first card so Space works without a prior click.
        if self._card_order:
            self._set_focus_idx(0)

    # ---- Signal handlers --------------------------------------

    def _on_card_toggled(self, label_id: int, checked: bool) -> None:
        """Sync state, refresh isolation + button states."""
        if checked:
            self._state.checked_ids.add(label_id)
        else:
            self._state.checked_ids.discard(label_id)
        self._fire_isolation_changed()
        self._refresh_button_states()

    def _on_merge_selected_clicked(self) -> None:
        if not self._state.can_merge_selected():
            return
        absorbed = sorted(self._state.checked_ids)
        # Callback returns the snapshot tuple (roi_slices, old_roi_data)
        # so we can roll back on Cancel.
        snapshot = self._on_merge_cb(self._state.keep_id, absorbed)
        self._state.transition_to_merged(snapshot, absorbed)
        self._refresh_button_states()
        # Visually de-emphasise the cards: disable the checkboxes so
        # the user can't toggle further selection during the merged
        # stage. They must Accept or Cancel from here.
        for card in self._cards.values():
            card.setEnabled(False)

    def _on_accept_clicked(self) -> None:
        if not self._state.can_accept():
            return
        # Accept is final — fire on_accept so the annotator can run the
        # post-commit chain (controller callback for DataFrame updates,
        # Navigator row removal, undo-stack commit). Then close.
        # Mark _clean_close BEFORE firing the callback so an exception
        # mid-callback still results in a clean close (no double-fire
        # via closeEvent).
        self._clean_close = True
        self._on_accept_cb(self._state.keep_id, list(self._state.absorbed_ids))
        self.close()

    def _on_cancel_clicked(self) -> None:
        # Cancel rolls back if a merge was performed; the on_cancel
        # callback is responsible for restoring the volume from
        # self._state.snapshot if we're post-merge.
        # Set _clean_close BEFORE firing so closeEvent doesn't re-fire
        # the cancel callback.
        self._clean_close = True
        self._on_cancel_cb()
        self.close()

    def closeEvent(self, event) -> None:
        """Catch every other close path (window X, parent destroyed,
        Qt teardown) and run the cancel callback so isolation proxies
        and volume snapshot are cleaned up. The Accept/Cancel button
        handlers set ``_clean_close = True`` so they aren't double-handled.

        Without this override, X-closing the dialog leaves the per-label
        multi-label sub-proxies orphaned in the viewer (created by the
        annotator's ``_apply_multi_label_proxy`` during candidate
        ticking) and the volume snapshot never rolls back if a merge
        was already performed. Subsequent operations — particularly
        ``ndisplay`` switches that trigger napari iso_categorical
        re-rendering — can crash on the orphaned proxy state.
        """
        if not self._clean_close:
            # Defensive: if the parent annotator is mid-teardown, its
            # own _close_cleanup already disposes isolation/proxies and
            # firing cancel_cb here would touch partially-destroyed state
            # (labels_combo, subviewers, etc.). The annotator's teardown
            # path is the source of truth in that case.
            parent = self.parent()
            if getattr(parent, "_is_closing", False):
                self._clean_close = True
            else:
                try:
                    self._on_cancel_cb()
                except Exception:
                    pass
                self._clean_close = True  # idempotent — no double-fire
        super().closeEvent(event)

    # ---- Helpers -----------------------------------------------

    def _fire_isolation_changed(self) -> None:
        if self._on_isolation_changed_cb is not None:
            self._on_isolation_changed_cb(self._state.isolation_set)

    def _refresh_button_states(self) -> None:
        self._merge_btn.setEnabled(self._state.can_merge_selected())
        self._accept_btn.setEnabled(self._state.can_accept())
        # Cancel is always enabled (acts as escape/rollback).

    def _set_focus_idx(self, idx: int) -> None:
        if not self._card_order:
            self._focused_idx = -1
            return
        idx = max(0, min(idx, len(self._card_order) - 1))
        # Clear focus on previously focused card.
        if 0 <= self._focused_idx < len(self._card_order):
            old_id = self._card_order[self._focused_idx]
            old_card = self._cards.get(old_id)
            if old_card is not None:
                old_card.set_focused(False)
        self._focused_idx = idx
        new_id = self._card_order[idx]
        new_card = self._cards.get(new_id)
        if new_card is not None:
            new_card.set_focused(True)
            self._scroll.ensureWidgetVisible(new_card)

    # ---- Keyboard ----------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Up/Down move focus; Space toggles the focused card; Esc cancels."""
        key = event.key()
        if key == Qt.Key_Escape:
            self._on_cancel_clicked()
            return
        if key in (Qt.Key_Down, Qt.Key_J):
            self._set_focus_idx(self._focused_idx + 1)
            return
        if key in (Qt.Key_Up, Qt.Key_K):
            self._set_focus_idx(self._focused_idx - 1)
            return
        if key == Qt.Key_Space:
            if 0 <= self._focused_idx < len(self._card_order):
                label_id = self._card_order[self._focused_idx]
                card = self._cards.get(label_id)
                if card is not None and card.isEnabled():
                    card.toggle()
                    return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            # Enter triggers the primary action: Merge if we can,
            # else Accept if we can. Otherwise no-op.
            if self._state.can_accept():
                self._on_accept_clicked()
                return
            if self._state.can_merge_selected():
                self._on_merge_selected_clicked()
                return
        super().keyPressEvent(event)

    # ---- API --------------------------------------------------

    @property
    def state(self) -> MergeDialogState:
        """Read-only access for tests + integration callers."""
        return self._state

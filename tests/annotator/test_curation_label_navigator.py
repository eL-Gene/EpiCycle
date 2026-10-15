"""Tests for CurationLabelNavigator — floating triage command center.

Phase K — TDD tests written BEFORE implementation.

Tests use lightweight fakes (no Qt/napari required). They test the public API
surface of the label navigator:
  - set_items() populates a table with columns ID/Area/Z/Tier
  - Tier filter checkboxes toggle visibility of table rows
  - Sort combo reorders rows (Tier / Area desc / Z asc / ID asc)
  - Row click calls on_navigate callback with (label_id, centroid)
  - Prev/Next skip hidden rows, wrap around
  - Mark Good / Delete / Discard All Garbage call callbacks correctly
  - Auto-advance after triage when checkbox is checked
  - Editing header and summary label update on selection/data changes
  - remove_item() and update_item_tier() for incremental updates
"""

from __future__ import annotations

import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lightweight fakes — no Qt required
# ---------------------------------------------------------------------------


class _FakeCheckBox:
    """Minimal stand-in for QCheckBox."""

    def __init__(self, text="", checked=True):
        self._text = text
        self._checked = checked

    def isChecked(self):
        return self._checked

    def setChecked(self, v):
        self._checked = v

    def text(self):
        return self._text


class _FakeButton:
    """Minimal stand-in for QPushButton."""

    def __init__(self, text=""):
        self._text = text
        self._enabled = True
        self._visible = True
        self._checkable = False
        self._checked = False

    def isEnabled(self):
        return self._enabled

    def setEnabled(self, v):
        self._enabled = v

    def isVisible(self):
        return self._visible

    def setVisible(self, v):
        self._visible = v

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t

    def setCheckable(self, v):
        self._checkable = v

    def isChecked(self):
        return self._checked

    def setChecked(self, v):
        self._checked = v


class _FakeLabel:
    """Minimal stand-in for QLabel."""

    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t


class _FakeCombo:
    """Minimal stand-in for QComboBox."""

    def __init__(self):
        self._items = []
        self._current_index = -1

    def addItems(self, items):
        self._items.extend(items)
        if self._current_index < 0 and self._items:
            self._current_index = 0

    def currentText(self):
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index]
        return ""

    def setCurrentText(self, t):
        if t in self._items:
            self._current_index = self._items.index(t)

    def currentIndex(self):
        return self._current_index

    def setCurrentIndex(self, idx):
        self._current_index = idx


class _FakeTableRow:
    """One row of data in the fake table."""

    def __init__(
        self,
        label_id,
        area,
        z,
        tier,
        centroid,
        hidden=False,
        qc_score=None,
        feature_state="",
    ):
        self.label_id = label_id
        self.area = area
        self.z = z
        self.tier = tier
        self.centroid = centroid
        self.hidden = hidden
        self.qc_score = qc_score
        self.feature_state = feature_state
        # Untriaged decoration tracked by the fake — production
        # decorates ID / Area / Z / Score cells with italic font +
        # dark-amber background when qc_tier == "" (newly-painted,
        # not yet triaged).  See navigator._rebuild_table.
        # Pre-2026-05-16 the criterion was feature_state == "stale"
        # but that caught too many rows (split fragments, post-accept
        # ghosts, etc.).  qc_tier == "" is narrower and matches the
        # user's "this is a new label" mental model.
        self.is_untriaged = str(tier) == ""
        # Legacy alias for back-compat with tests written against
        # the old criterion (now also reads qc_tier).  Removed in a
        # later cleanup.
        self.is_stale = self.is_untriaged
        # Full item dict for UserRole storage
        self.item_data = {
            "label_id": label_id,
            "tier": tier,
            "centroid": centroid,
            "area": area,
            "qc_score": qc_score,
            "feature_state": feature_state,
        }


class _FakeTableWidget:
    """Minimal stand-in for QTableWidget — stores rows as _FakeTableRow.

    Supports multi-selection via _selected_rows set (for ExtendedSelection).
    """

    def __init__(self):
        self._rows: list[_FakeTableRow] = []
        self._selected_row = -1
        self._selected_rows: set[int] = set()
        self._focus_count = 0

    def rowCount(self):
        return len(self._rows)

    def addRow(self, row: _FakeTableRow):
        self._rows.append(row)

    def clear(self):
        self._rows.clear()
        self._selected_row = -1
        self._selected_rows.clear()

    def row(self, idx):
        if 0 <= idx < len(self._rows):
            return self._rows[idx]
        return None

    def isRowHidden(self, idx):
        if 0 <= idx < len(self._rows):
            return self._rows[idx].hidden
        return True

    def setRowHidden(self, idx, hidden):
        if 0 <= idx < len(self._rows):
            self._rows[idx].hidden = hidden

    def selectedRow(self):
        return self._selected_row

    def selectRow(self, idx):
        """Select a single row (clears multi-selection, like plain click)."""
        self._selected_row = idx
        self._selected_rows = {idx} if 0 <= idx < len(self._rows) else set()

    def selectedRows(self) -> set:
        """Return all selected row indices (for multi-select)."""
        return set(self._selected_rows)

    def setCurrentCell(self, row, col):
        """Move focus without changing selection (for multi-select nav)."""
        self._selected_row = row

    def addToSelection(self, idx):
        """Toggle a row in/out of selection (Ctrl+Click simulation)."""
        if 0 <= idx < len(self._rows):
            if idx in self._selected_rows:
                self._selected_rows.discard(idx)
            else:
                self._selected_rows.add(idx)
            self._selected_row = idx

    def removeRow(self, idx):
        if 0 <= idx < len(self._rows):
            self._rows.pop(idx)
            self._selected_rows.discard(idx)
            # Shift down indices above removed row
            self._selected_rows = {
                (r - 1 if r > idx else r) for r in self._selected_rows
            }
            if self._selected_row >= len(self._rows):
                self._selected_row = len(self._rows) - 1

    def sortRows(self, key_func, reverse=False):
        self._rows.sort(key=key_func, reverse=reverse)

    def setFocus(self):
        self._focus_count += 1

    def setFocusPolicy(self, policy):
        pass


# ---------------------------------------------------------------------------
# Build a minimal CurationLabelNavigator-like object for testing
# ---------------------------------------------------------------------------

TIER_COLORS = {
    "curated": "#00bcd4",
    "good": "#4caf50",
    "review": "#ffc83c",
    "bad": "#ff7828",
    "garbage": "#f03c3c",
}

TIER_ORDER = {"curated": 0, "good": 1, "review": 2, "bad": 3, "garbage": 4}


def _make_fake_navigator(*, callbacks=None):
    """Create a fake navigator object that mirrors the expected API.

    This simulates the methods and attributes that CurationLabelNavigator
    will have. Tests validate the API contract that the implementation must
    satisfy.
    """
    nav = types.SimpleNamespace()

    # --- UI widgets ---
    nav._editing_header = _FakeLabel("")
    nav._summary_label = _FakeLabel("")
    nav._table = _FakeTableWidget()

    # Tier filter checkboxes
    nav._tier_filter_curated = _FakeCheckBox("C", checked=True)
    nav._tier_filter_good = _FakeCheckBox("G", checked=True)
    nav._tier_filter_review = _FakeCheckBox("R", checked=True)
    nav._tier_filter_bad = _FakeCheckBox("B", checked=True)
    nav._tier_filter_garbage = _FakeCheckBox("Ga", checked=True)

    # Sort combo
    nav._sort_combo = _FakeCombo()
    nav._sort_combo.addItems(["Tier", "Area (desc)", "Z (asc)", "ID (asc)", "Score (asc)"])

    # Action buttons
    nav._mark_good_btn = _FakeButton("Mark Good")
    nav._delete_label_btn = _FakeButton("Delete Label")
    nav._discard_garbage_btn = _FakeButton("Discard All Garbage")
    nav._discard_garbage_btn.setEnabled(False)
    nav._new_label_btn = _FakeButton("New Label")
    nav._prev_btn = _FakeButton("Prev")
    nav._next_btn = _FakeButton("Next")

    # Recalculate QC button
    nav._recalc_qc_btn = _FakeButton("Recalculate QC")
    nav._recalc_qc_btn.setEnabled(False)

    # Focused mode + auto-advance
    nav._focused_mode_btn = _FakeButton("Focused Mode: OFF")
    nav._focused_mode_btn.setCheckable(True)
    nav._auto_advance_cb = _FakeCheckBox("Auto-advance after triage", checked=False)

    # --- Callbacks ---
    nav._on_navigate_cb = None
    nav._on_select_label_cb = None
    # Unified tier-flag callback for the 5 tier buttons (post-2026-05-10).
    nav._on_mark_tier_cb = None
    # Legacy aliases (back-compat for tests still using on_mark_good /
    # on_delete kwargs) — wired in the callback application block below
    # to route through _on_mark_tier_cb.
    nav._on_mark_good_cb = None
    nav._on_delete_cb = None
    nav._on_discard_garbage_cb = None
    nav._on_focused_mode_changed_cb = None
    nav._on_new_label_cb = None
    nav._on_paint_isolation_toggled_cb = None
    nav._on_paint_label_changed_cb = None

    # Isolate checkbox
    nav._isolate_cb = _FakeCheckBox("Isolate", checked=False)
    nav._on_show_selected_label_changed_cb = None
    nav._on_isolation_changed_cb = None

    # Scroll subviewer callback (hover-based Left/Right routing)
    nav._on_scroll_subviewer_cb = None

    # Recalculate callback
    nav._on_recalculate_cb = None

    # Apply callbacks if provided
    if callbacks:
        nav._on_navigate_cb = callbacks.get("on_navigate")
        nav._on_select_label_cb = callbacks.get("on_select_label")
        # Prefer the new unified on_mark_tier; fall back to combining
        # legacy on_mark_good + on_delete kwargs into a tier-aware
        # wrapper for back-compat with older test fixtures.
        nav._on_mark_tier_cb = callbacks.get("on_mark_tier")
        if nav._on_mark_tier_cb is None:
            legacy_good = callbacks.get("on_mark_good")
            legacy_delete = callbacks.get("on_delete")
            if legacy_good is not None or legacy_delete is not None:
                def _wrap_tier(lid, tier, _g=legacy_good, _d=legacy_delete):
                    if tier == "good" and _g is not None:
                        _g(lid)
                    elif tier == "garbage" and _d is not None:
                        # Pre-2026-05-10 the Delete button bundled
                        # volume-zeroing into this callback; post-
                        # redesign the Garbage button is flag-only.
                        # Legacy tests still observe the lid-only call.
                        _d(lid)
                nav._on_mark_tier_cb = _wrap_tier
        nav._on_mark_good_cb = callbacks.get("on_mark_good")  # legacy mirror
        nav._on_delete_cb = callbacks.get("on_delete")  # legacy mirror
        nav._on_discard_garbage_cb = callbacks.get("on_discard_garbage")
        nav._on_focused_mode_changed_cb = callbacks.get("on_focused_mode_changed")
        nav._on_show_selected_label_changed_cb = callbacks.get(
            "on_show_selected_label_changed"
        )
        nav._on_isolation_changed_cb = callbacks.get("on_isolation_changed")
        nav._on_scroll_subviewer_cb = callbacks.get("on_scroll_subviewer")
        nav._on_recalculate_cb = callbacks.get("on_recalculate")
        nav._on_new_label_cb = callbacks.get("on_new_label")
        nav._on_paint_isolation_toggled_cb = callbacks.get("on_paint_isolation_toggled")
        nav._on_paint_label_changed_cb = callbacks.get("on_paint_label_changed")

    # --- Internal state ---
    nav._items = []

    # Painting badge — tracks layer.selected_label.  Default 0 = bg.
    nav._paint_label = 0
    # (layer, callback) for clean disconnect when set_label_layer
    # is called with a new layer or with None at teardown.
    nav._selected_label_handler = None

    return nav


# ===========================================================================
# Test helpers — simulate the methods the implementation will provide
# ===========================================================================


def _set_callbacks(nav, **kwargs):
    """Simulate CurationLabelNavigator.set_callbacks()."""
    nav._on_navigate_cb = kwargs.get("on_navigate")
    nav._on_select_label_cb = kwargs.get("on_select_label")
    nav._on_mark_tier_cb = kwargs.get("on_mark_tier")
    if nav._on_mark_tier_cb is None and kwargs.get("on_mark_good") is not None:
        _legacy = kwargs["on_mark_good"]
        def _wrap_good(lid, tier, _f=_legacy):
            if tier == "good":
                _f(lid)
        nav._on_mark_tier_cb = _wrap_good
    nav._on_mark_good_cb = kwargs.get("on_mark_good")  # legacy mirror
    nav._on_delete_cb = kwargs.get("on_delete")  # legacy mirror
    nav._on_discard_garbage_cb = kwargs.get("on_discard_garbage")
    nav._on_focused_mode_changed_cb = kwargs.get("on_focused_mode_changed")
    nav._on_show_selected_label_changed_cb = kwargs.get(
        "on_show_selected_label_changed"
    )
    nav._on_isolation_changed_cb = kwargs.get("on_isolation_changed")
    nav._on_scroll_subviewer_cb = kwargs.get("on_scroll_subviewer")
    nav._on_recalculate_cb = kwargs.get("on_recalculate")
    nav._on_new_label_cb = kwargs.get("on_new_label")
    nav._on_paint_isolation_toggled_cb = kwargs.get("on_paint_isolation_toggled")
    nav._on_paint_label_changed_cb = kwargs.get("on_paint_label_changed")


def _set_items(nav, items):
    """Simulate CurationLabelNavigator.set_items(items).

    items: list of dicts with keys: label_id, tier, centroid, area
    """
    nav._items = list(items)
    _rebuild_table(nav)


def _rebuild_table(nav):
    """Rebuild the table from nav._items, applying filter + sort.

    Mirrors the real implementation: only rows that pass the tier filter
    are inserted into the table (no setRowHidden).
    """
    prev_selected_ids = _selected_label_ids(nav)
    prev_current_id = _current_label_id(nav)

    # --- Filter ---
    filter_map = {
        "curated": nav._tier_filter_curated.isChecked(),
        "good": nav._tier_filter_good.isChecked(),
        "review": nav._tier_filter_review.isChecked(),
        "bad": nav._tier_filter_bad.isChecked(),
        "garbage": nav._tier_filter_garbage.isChecked(),
    }
    visible = [
        it for it in nav._items
        if filter_map.get(it.get("tier", "review"), True)
    ]

    # --- Sort ---
    sort_text = nav._sort_combo.currentText()
    if sort_text == "Tier":
        visible.sort(
            key=lambda d: (TIER_ORDER.get(d.get("tier", "review"), 99), d.get("label_id", 0))
        )
    elif sort_text == "Area (desc)":
        visible.sort(key=lambda d: d.get("area", 0), reverse=True)
    elif sort_text == "Z (asc)":
        visible.sort(key=lambda d: (d.get("centroid", (0, 0, 0))[0], d.get("label_id", 0)))
    elif sort_text == "ID (asc)":
        visible.sort(key=lambda d: d.get("label_id", 0))
    elif sort_text == "Score (asc)":
        visible.sort(
            key=lambda d: (
                d.get("qc_score") if d.get("qc_score") is not None else float("inf"),
                d.get("label_id", 0),
            )
        )

    # --- Populate table (only visible rows) ---
    nav._table.clear()
    for it in visible:
        row = _FakeTableRow(
            it["label_id"],
            it.get("area", 0),
            it.get("centroid", (0, 0, 0))[0],
            it.get("tier", "review"),
            it.get("centroid", (0, 0, 0)),
            qc_score=it.get("qc_score"),
            feature_state=it.get("feature_state", ""),
        )
        nav._table.addRow(row)

    # --- Summary from ALL items ---
    tier_counts = {"curated": 0, "good": 0, "review": 0, "bad": 0, "garbage": 0}
    has_garbage = False
    for it in nav._items:
        t = it.get("tier", "review")
        if t in tier_counts:
            tier_counts[t] += 1
        if t == "garbage":
            has_garbage = True
    _update_summary(nav, tier_counts)
    nav._discard_garbage_btn.setEnabled(has_garbage)
    nav._recalc_qc_btn.setEnabled(tier_counts.get("curated", 0) > 0)

    # --- Restore multi-selection ---
    first_match = None
    for i in range(nav._table.rowCount()):
        r = nav._table.row(i)
        if r and r.label_id in prev_selected_ids:
            nav._table._selected_rows.add(i)
            if r.label_id == prev_current_id:
                first_match = i
            elif first_match is None:
                first_match = i
    if first_match is not None:
        nav._table._selected_row = first_match
    _update_editing_header(nav)


def _update_summary(nav, tier_counts=None):
    """Update the summary label with tier counts."""
    if tier_counts is None:
        tier_counts = {"curated": 0, "good": 0, "review": 0, "bad": 0, "garbage": 0}
        for it in nav._items:
            t = it.get("tier", "review")
            if t in tier_counts:
                tier_counts[t] += 1

    total = sum(tier_counts.values())
    parts = []
    for t in ("curated", "good", "review", "bad", "garbage"):
        if tier_counts[t] > 0:
            color = TIER_COLORS.get(t, "#ffffff")
            parts.append(
                f'<span style="color:{color}">{tier_counts[t]} {t}</span>'
            )
    summary = f"{total} labels: {', '.join(parts)}" if parts else f"{total} labels"
    nav._summary_label.setText(summary)

    # Update discard button text with garbage count
    gc = tier_counts.get("garbage", 0)
    if gc > 0:
        nav._discard_garbage_btn._text = f"Discard All Garbage ({gc})"
    else:
        nav._discard_garbage_btn._text = "Discard All Garbage"


def _apply_filter(nav):
    """Rebuild the table when filter checkboxes change."""
    _rebuild_table(nav)


def _apply_sort(nav):
    """Rebuild the table when sort combo changes."""
    _rebuild_table(nav)


def _current_label_id(nav):
    """Return the label_id of the currently selected table row, or None."""
    idx = nav._table.selectedRow()
    if idx < 0:
        return None
    r = nav._table.row(idx)
    return r.label_id if r else None


def _selected_label_ids(nav):
    """Return label_ids for ALL currently selected rows."""
    result = set()
    for idx in nav._table.selectedRows():
        r = nav._table.row(idx)
        if r is not None:
            result.add(r.label_id)
    return result


def _select_label(nav, label_id):
    """Programmatically select and scroll to the given label."""
    for i in range(nav._table.rowCount()):
        r = nav._table.row(i)
        if r and r.label_id == label_id:
            nav._table.selectRow(i)
            _on_row_clicked(nav, i)
            return


def _select_labels(nav, label_ids):
    """Programmatically select multiple labels and trigger isolation.

    B4 fix: Don't call _on_row_clicked (which disrupts multi-selection
    in real Qt). Instead, directly navigate to the first row and fire
    callbacks without touching selection state.
    B5 fix: Only call _fire_isolation_if_active once at the end.
    """
    label_id_set = set(label_ids)
    nav._table._selected_rows.clear()
    first_row = None
    for i in range(nav._table.rowCount()):
        r = nav._table.row(i)
        if r and r.label_id in label_id_set:
            nav._table._selected_rows.add(i)
            if first_row is None:
                first_row = i
    if first_row is not None:
        nav._table._selected_row = first_row
        # Navigate to first row WITHOUT calling _on_row_clicked
        # (which would fire _fire_isolation_if_active again = B5 double-fire)
        r = nav._table.row(first_row)
        if r is not None:
            _update_editing_header(nav)
            if nav._on_navigate_cb:
                nav._on_navigate_cb(r.label_id, r.centroid)
            if nav._on_select_label_cb:
                nav._on_select_label_cb(r.label_id)
    # B5: Fire isolation exactly once at the end
    _fire_isolation_if_active(nav)


def _fire_isolation_if_active(nav):
    """If isolate is ON, fire the isolation callback with all selected IDs."""
    if not nav._isolate_cb.isChecked():
        return
    if nav._on_isolation_changed_cb:
        try:
            nav._on_isolation_changed_cb(_selected_label_ids(nav))
        except Exception:
            pass


def _update_editing_header(nav):
    """Update the editing header — Painting prefix + Editing suffix.

    Mirrors the production implementation in
    ``curation_label_navigator.py::_update_editing_header``.  Format:
    ``Painting: N  •  Editing: Label N (tier)  Z=z``.  Paint=0 shows
    ``"Painting: background (0)"``.  Editing suffix only when a row
    is selected.
    """
    paint = int(getattr(nav, "_paint_label", 0))
    if paint == 0:
        paint_text = "Painting: background (0)"
    else:
        paint_text = f"Painting: {paint}"

    editing_text = ""
    lid = _current_label_id(nav)
    if lid is not None:
        idx = nav._table.selectedRow()
        r = nav._table.row(idx)
        if r:
            sel_count = len(nav._table.selectedRows())
            if sel_count > 1:
                editing_text = (
                    f"Editing: Label {r.label_id} ({r.tier})  Z={r.z}  "
                    f"[{sel_count} selected]"
                )
            else:
                editing_text = (
                    f"Editing: Label {r.label_id} ({r.tier})  Z={r.z}"
                )

    if editing_text:
        nav._editing_header.setText(f"{paint_text}  •  {editing_text}")
    else:
        nav._editing_header.setText(paint_text)


def _set_paint_label(nav, label_id):
    """Mirror of CurationLabelNavigator.set_paint_label."""
    nav._paint_label = max(0, int(label_id))
    _update_editing_header(nav)


def _click_new_label(nav):
    """Mirror of CurationLabelNavigator._on_new_label_clicked."""
    cb = nav._on_new_label_cb
    if cb is None:
        return
    try:
        cb()
    except Exception:
        pass


def _on_row_clicked(nav, row_idx):
    """Simulate clicking a row — navigate and select.

    In the real Qt implementation, the current cell always moves to the
    clicked row.  We use setCurrentCell (not selectRow) to avoid clearing
    multi-selection — Qt's ExtendedSelection handles selection changes
    before cellClicked fires.
    """
    nav._table.setCurrentCell(row_idx, 0)
    r = nav._table.row(row_idx)
    if r is None:
        return
    _update_editing_header(nav)
    if nav._on_navigate_cb:
        nav._on_navigate_cb(r.label_id, r.centroid)
    if nav._on_select_label_cb:
        nav._on_select_label_cb(r.label_id)
    # Fire isolation update with ALL selected labels
    _fire_isolation_if_active(nav)
    # Restore focus to table (mirrors real implementation)
    nav._table.setFocus()


def _navigate_next(nav):
    """Move to next row. In multi-select, cycle within selected rows only."""
    total = nav._table.rowCount()
    if total == 0:
        return
    selected = nav._table.selectedRows()
    current = nav._table.selectedRow()
    if len(selected) > 1:
        sorted_sel = sorted(selected)
        after = [r for r in sorted_sel if r > current]
        next_row = after[0] if after else sorted_sel[0]
        nav._table.setCurrentCell(next_row, 0)
    else:
        next_row = (current + 1) % total
        nav._table.selectRow(next_row)
    _on_row_clicked(nav, next_row)


def _navigate_prev(nav):
    """Move to previous row. In multi-select, cycle within selected rows only."""
    total = nav._table.rowCount()
    if total == 0:
        return
    selected = nav._table.selectedRows()
    current = nav._table.selectedRow()
    if current < 0:
        current = 0
    if len(selected) > 1:
        sorted_sel = sorted(selected)
        before = [r for r in sorted_sel if r < current]
        prev_row = before[-1] if before else sorted_sel[-1]
        nav._table.setCurrentCell(prev_row, 0)
    else:
        prev_row = (current - 1) % total
        nav._table.selectRow(prev_row)
    _on_row_clicked(nav, prev_row)


def _on_mark_tier_clicked(nav, tier):
    """Mark the currently selected label with the given tier (flag-only).

    Post-2026-05-10: replaces the previous _on_mark_good_clicked /
    _on_delete_clicked simulators.  The 5 tier buttons (curated, good,
    review, bad, garbage) all dispatch through this one simulator with
    the tier captured at click time.
    """
    lid = _current_label_id(nav)
    if lid is None:
        return
    if nav._on_mark_tier_cb:
        nav._on_mark_tier_cb(lid, tier)
    _remove_item(nav, lid)
    if nav._auto_advance_cb.isChecked():
        _advance_to_next_visible(nav)


def _on_mark_good_clicked(nav):
    """Legacy simulator: marks current label as good."""
    _on_mark_tier_clicked(nav, "good")


def _on_delete_clicked(nav):
    """Legacy simulator: previously zeroed + flagged garbage; the
    flag-only path now goes through _on_mark_tier_clicked('garbage')."""
    _on_mark_tier_clicked(nav, "garbage")


def _on_discard_garbage_clicked(nav):
    """Discard all garbage-tier labels."""
    garbage_ids = [
        it["label_id"] for it in nav._items if it.get("tier") == "garbage"
    ]
    if not garbage_ids:
        return

    nav._items = [it for it in nav._items if it.get("tier") != "garbage"]
    _rebuild_table(nav)

    if nav._on_discard_garbage_cb:
        nav._on_discard_garbage_cb(garbage_ids)


def _on_recalc_clicked(nav):
    """Collect curated label IDs and fire recalculate callback."""
    curated_ids = [
        it["label_id"] for it in nav._items if it.get("tier") == "curated"
    ]
    if not curated_ids:
        return
    if nav._on_recalculate_cb:
        nav._on_recalculate_cb(curated_ids)


def _remove_item(nav, label_id):
    """Remove a single label from the items and rebuild."""
    nav._items = [it for it in nav._items if it.get("label_id") != label_id]
    _rebuild_table(nav)


def _update_item_tier(nav, label_id, new_tier):
    """Update a single item's tier and rebuild."""
    for it in nav._items:
        if it.get("label_id") == label_id:
            it["tier"] = new_tier
            break
    _rebuild_table(nav)


def _advance_to_next_visible(nav):
    """Auto-advance to the next row after a triage action."""
    total = nav._table.rowCount()
    if total == 0:
        nav._table.selectRow(-1)
        _update_editing_header(nav)
        return
    idx = min(max(nav._table.selectedRow(), 0), total - 1)
    _on_row_clicked(nav, idx)


def _select_next_triage(nav):
    """Select the first visible row with a triage tier (review/bad/garbage).

    Returns True if found, False otherwise.
    """
    _TRIAGE_TIERS = {"review", "bad", "garbage"}
    for i in range(nav._table.rowCount()):
        r = nav._table.row(i)
        if r and r.tier in _TRIAGE_TIERS:
            nav._table.selectRow(i)
            _on_row_clicked(nav, i)
            return True
    return False


# ===========================================================================
# Test data fixtures
# ===========================================================================


@pytest.fixture
def sample_items():
    """Sample flagged label items with all 4 tiers."""
    return [
        {"label_id": 5, "tier": "good", "centroid": (2, 4, 7), "area": 2031, "qc_score": 0.95},
        {"label_id": 42, "tier": "review", "centroid": (5, 2, 8), "area": 1204, "qc_score": 0.62},
        {"label_id": 17, "tier": "bad", "centroid": (3, 1, 4), "area": 389, "qc_score": 0.31},
        {"label_id": 91, "tier": "garbage", "centroid": (7, 3, 6), "area": 52, "qc_score": 0.08},
    ]


@pytest.fixture
def mixed_items():
    """Larger item set with multiple garbage labels."""
    return [
        {"label_id": 1, "tier": "good", "centroid": (0, 0, 0), "area": 3000, "qc_score": 0.92},
        {"label_id": 2, "tier": "good", "centroid": (1, 1, 1), "area": 2800, "qc_score": 0.88},
        {"label_id": 3, "tier": "review", "centroid": (2, 2, 2), "area": 1500, "qc_score": 0.55},
        {"label_id": 4, "tier": "bad", "centroid": (3, 3, 3), "area": 400, "qc_score": 0.22},
        {"label_id": 5, "tier": "garbage", "centroid": (4, 4, 4), "area": 30, "qc_score": 0.05},
        {"label_id": 6, "tier": "garbage", "centroid": (5, 5, 5), "area": 15, "qc_score": 0.03},
    ]


# ===========================================================================
# Tests — set_items populates table
# ===========================================================================


class TestSetItems:
    """Verify set_items populates the table with all items."""

    def test_populates_table_with_all_items(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        assert nav._table.rowCount() == 4

    def test_row_stores_label_id(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # After default sort (Tier), good comes first
        r0 = nav._table.row(0)
        assert r0.label_id == 5  # good tier

    def test_row_stores_tier(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        r0 = nav._table.row(0)
        assert r0.tier == "good"

    def test_row_stores_centroid(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        r0 = nav._table.row(0)
        assert r0.centroid == (2, 4, 7)

    def test_row_stores_area(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        r0 = nav._table.row(0)
        assert r0.area == 2031

    def test_row_stores_z_from_centroid(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        r0 = nav._table.row(0)
        assert r0.z == 2  # centroid[0] for label 5

    def test_replaces_previous_items(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        assert nav._table.rowCount() == 4
        _set_items(nav, [sample_items[0]])
        assert nav._table.rowCount() == 1

    def test_empty_items_clears_table(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _set_items(nav, [])
        assert nav._table.rowCount() == 0

    def test_discard_button_enabled_when_garbage_exists(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        assert nav._discard_garbage_btn.isEnabled() is True

    def test_discard_button_disabled_when_no_garbage(self):
        nav = _make_fake_navigator()
        items = [
            {"label_id": 1, "tier": "good", "centroid": (0, 0, 0), "area": 100},
        ]
        _set_items(nav, items)
        assert nav._discard_garbage_btn.isEnabled() is False


# ===========================================================================
# Tests — Summary label
# ===========================================================================


class TestSummaryLabel:
    """Verify summary label shows tier counts."""

    def test_shows_all_tier_counts(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        text = nav._summary_label.text()
        assert "4 labels" in text
        assert "1 good" in text
        assert "1 review" in text
        assert "1 bad" in text
        assert "1 garbage" in text

    def test_shows_mixed_counts(self, mixed_items):
        nav = _make_fake_navigator()
        _set_items(nav, mixed_items)
        text = nav._summary_label.text()
        assert "6 labels" in text
        assert "2 good" in text
        assert "2 garbage" in text

    def test_empty_items(self):
        nav = _make_fake_navigator()
        _set_items(nav, [])
        text = nav._summary_label.text()
        assert "0 labels" in text

    def test_discard_button_shows_garbage_count(self, mixed_items):
        nav = _make_fake_navigator()
        _set_items(nav, mixed_items)
        assert "(2)" in nav._discard_garbage_btn.text()


# ===========================================================================
# Tests — Tier filter
# ===========================================================================


class TestTierFilter:
    """Verify tier filter checkboxes exclude rows from the table."""

    def test_all_checked_shows_all(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        assert nav._table.rowCount() == 4

    def test_uncheck_garbage_excludes_garbage(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._tier_filter_garbage.setChecked(False)
        _apply_filter(nav)
        assert nav._table.rowCount() == 3
        tiers = {nav._table.row(i).tier for i in range(nav._table.rowCount())}
        assert "garbage" not in tiers

    def test_uncheck_good_excludes_good(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._tier_filter_good.setChecked(False)
        _apply_filter(nav)
        assert nav._table.rowCount() == 3
        tiers = {nav._table.row(i).tier for i in range(nav._table.rowCount())}
        assert "good" not in tiers

    def test_uncheck_multiple_tiers(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._tier_filter_good.setChecked(False)
        nav._tier_filter_garbage.setChecked(False)
        _apply_filter(nav)
        assert nav._table.rowCount() == 2
        tiers = {nav._table.row(i).tier for i in range(nav._table.rowCount())}
        assert tiers == {"review", "bad"}

    def test_uncheck_all_empties_table(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._tier_filter_good.setChecked(False)
        nav._tier_filter_review.setChecked(False)
        nav._tier_filter_bad.setChecked(False)
        nav._tier_filter_garbage.setChecked(False)
        _apply_filter(nav)
        assert nav._table.rowCount() == 0

    def test_recheck_restores_rows(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._tier_filter_garbage.setChecked(False)
        _apply_filter(nav)
        assert nav._table.rowCount() == 3
        # Recheck
        nav._tier_filter_garbage.setChecked(True)
        _apply_filter(nav)
        assert nav._table.rowCount() == 4


# ===========================================================================
# Tests — Sorting
# ===========================================================================


class TestSorting:
    """Verify sort combo reorders table rows."""

    def test_sort_by_tier_groups_tiers(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # Default sort is Tier
        tiers = [nav._table.row(i).tier for i in range(nav._table.rowCount())]
        assert tiers == ["good", "review", "bad", "garbage"]

    def test_sort_by_area_descending(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._sort_combo.setCurrentText("Area (desc)")
        _apply_sort(nav)
        areas = [nav._table.row(i).area for i in range(nav._table.rowCount())]
        assert areas == sorted(areas, reverse=True)

    def test_sort_by_z_ascending(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._sort_combo.setCurrentText("Z (asc)")
        _apply_sort(nav)
        zs = [nav._table.row(i).z for i in range(nav._table.rowCount())]
        assert zs == sorted(zs)

    def test_sort_by_id_ascending(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._sort_combo.setCurrentText("ID (asc)")
        _apply_sort(nav)
        ids = [nav._table.row(i).label_id for i in range(nav._table.rowCount())]
        assert ids == sorted(ids)

    def test_sort_by_score_ascending(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._sort_combo.setCurrentText("Score (asc)")
        _apply_sort(nav)
        scores = [nav._table.row(i).qc_score for i in range(nav._table.rowCount())]
        assert scores == sorted(scores)

    def test_sort_by_score_none_values_last(self):
        """Labels without qc_score should sort to the end."""
        items = [
            {"label_id": 1, "tier": "review", "centroid": (0, 0, 0), "area": 100, "qc_score": 0.5},
            {"label_id": 2, "tier": "review", "centroid": (1, 1, 1), "area": 200, "qc_score": None},
            {"label_id": 3, "tier": "review", "centroid": (2, 2, 2), "area": 300, "qc_score": 0.1},
        ]
        nav = _make_fake_navigator()
        _set_items(nav, items)
        nav._sort_combo.setCurrentText("Score (asc)")
        _apply_sort(nav)
        ids = [nav._table.row(i).label_id for i in range(nav._table.rowCount())]
        # 0.1, 0.5 first; None last
        assert ids == [3, 1, 2]

    def test_sort_preserves_filter(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._tier_filter_garbage.setChecked(False)
        _apply_filter(nav)
        nav._sort_combo.setCurrentText("Area (desc)")
        _apply_sort(nav)
        # Garbage rows should not be in the table after re-sort
        assert nav._table.rowCount() == 3
        tiers = {nav._table.row(i).tier for i in range(nav._table.rowCount())}
        assert "garbage" not in tiers


# ===========================================================================
# Tests — Row click navigation
# ===========================================================================


class TestRowClickNavigation:
    """Verify clicking a row calls navigate and select callbacks."""

    def test_click_calls_navigate_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: calls.append((lid, c))}
        )
        _set_items(nav, sample_items)
        _on_row_clicked(nav, 0)
        assert len(calls) == 1
        assert calls[0][0] == nav._table.row(0).label_id

    def test_click_calls_select_label_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_select_label": lambda lid: calls.append(lid)}
        )
        _set_items(nav, sample_items)
        _on_row_clicked(nav, 1)
        assert len(calls) == 1
        assert calls[0] == nav._table.row(1).label_id

    def test_click_passes_centroid(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: calls.append((lid, c))}
        )
        _set_items(nav, sample_items)
        _on_row_clicked(nav, 0)
        r0 = nav._table.row(0)
        assert calls[0][1] == r0.centroid

    def test_click_updates_editing_header(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _on_row_clicked(nav, 0)
        r0 = nav._table.row(0)
        header = nav._editing_header.text()
        assert f"Label {r0.label_id}" in header
        assert f"({r0.tier})" in header
        assert f"Z={r0.z}" in header

    def test_click_selects_row(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _on_row_clicked(nav, 2)
        assert nav._table.selectedRow() == 2


# ===========================================================================
# Tests — Prev/Next navigation
# ===========================================================================


class TestPrevNextNavigation:
    """Verify Prev/Next skip hidden rows and wrap around."""

    def test_next_advances(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        _navigate_next(nav)
        assert nav._table.selectedRow() == 1

    def test_next_wraps_around(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(3)  # Last row
        _navigate_next(nav)
        assert nav._table.selectedRow() == 0

    def test_prev_goes_back(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(2)
        _navigate_prev(nav)
        assert nav._table.selectedRow() == 1

    def test_prev_wraps_around(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        _navigate_prev(nav)
        assert nav._table.selectedRow() == 3

    def test_next_with_filter_excludes_filtered(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # Exclude review — table has: good, bad, garbage (3 rows)
        nav._tier_filter_review.setChecked(False)
        _apply_filter(nav)
        assert nav._table.rowCount() == 3
        nav._table.selectRow(0)  # good
        _navigate_next(nav)
        # Next row is bad (review excluded from table entirely)
        r = nav._table.row(nav._table.selectedRow())
        assert r.tier == "bad"

    def test_prev_with_filter_excludes_filtered(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # Exclude review — table has: good, bad, garbage (3 rows)
        nav._tier_filter_review.setChecked(False)
        _apply_filter(nav)
        # Start at bad (index 1 in filtered table)
        nav._table.selectRow(1)
        _navigate_prev(nav)
        # Prev row is good (review excluded from table entirely)
        r = nav._table.row(nav._table.selectedRow())
        assert r.tier == "good"

    def test_next_calls_navigate_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: calls.append(lid)}
        )
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        _navigate_next(nav)
        assert len(calls) == 1


# ===========================================================================
# Tests — Triage actions
# ===========================================================================


class TestTriageActions:
    """Verify Mark Good, Delete, and Discard All Garbage."""

    def test_mark_good_calls_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_mark_good": lambda lid: calls.append(lid)}
        )
        _set_items(nav, sample_items)
        _select_label(nav, 42)  # review
        _on_mark_good_clicked(nav)
        assert len(calls) == 1
        assert calls[0] == 42

    def test_mark_good_removes_from_table(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _select_label(nav, 42)
        _on_mark_good_clicked(nav)
        assert nav._table.rowCount() == 3
        remaining_ids = [nav._table.row(i).label_id for i in range(3)]
        assert 42 not in remaining_ids

    def test_delete_calls_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_delete": lambda lid: calls.append(lid)}
        )
        _set_items(nav, sample_items)
        _select_label(nav, 17)  # bad
        _on_delete_clicked(nav)
        assert len(calls) == 1
        assert calls[0] == 17

    def test_delete_removes_from_table(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _select_label(nav, 17)
        _on_delete_clicked(nav)
        assert nav._table.rowCount() == 3
        remaining_ids = [nav._table.row(i).label_id for i in range(3)]
        assert 17 not in remaining_ids

    def test_discard_garbage_calls_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_discard_garbage": lambda ids: calls.append(ids)}
        )
        _set_items(nav, sample_items)
        _on_discard_garbage_clicked(nav)
        assert len(calls) == 1
        assert 91 in calls[0]

    def test_discard_garbage_removes_garbage_rows(self, mixed_items):
        nav = _make_fake_navigator()
        _set_items(nav, mixed_items)
        assert nav._table.rowCount() == 6
        _on_discard_garbage_clicked(nav)
        assert nav._table.rowCount() == 4
        remaining_tiers = {nav._table.row(i).tier for i in range(4)}
        assert "garbage" not in remaining_tiers

    def test_discard_garbage_callback_receives_all_ids(self, mixed_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_discard_garbage": lambda ids: calls.append(ids)}
        )
        _set_items(nav, mixed_items)
        _on_discard_garbage_clicked(nav)
        assert set(calls[0]) == {5, 6}

    def test_discard_button_disabled_after_discard(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        assert nav._discard_garbage_btn.isEnabled() is True
        _on_discard_garbage_clicked(nav)
        assert nav._discard_garbage_btn.isEnabled() is False

    def test_no_callback_does_not_crash(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _select_label(nav, 42)
        # No callbacks set — should not raise
        _on_mark_good_clicked(nav)
        _on_delete_clicked(nav)
        _on_discard_garbage_clicked(nav)

    def test_no_selection_does_not_crash(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # No row selected — should not raise
        _on_mark_good_clicked(nav)
        _on_delete_clicked(nav)


# ===========================================================================
# Tests — Auto-advance
# ===========================================================================


class TestAutoAdvance:
    """Verify auto-advance moves to next visible row after triage action."""

    def test_auto_advance_after_mark_good(self, sample_items):
        nav_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: nav_calls.append(lid)}
        )
        _set_items(nav, sample_items)
        nav._auto_advance_cb.setChecked(True)
        _select_label(nav, 5)  # good (first row)
        _on_mark_good_clicked(nav)
        # Should have auto-advanced to the next row
        assert nav._table.selectedRow() >= 0
        assert _current_label_id(nav) is not None
        assert _current_label_id(nav) != 5

    def test_auto_advance_after_delete(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._auto_advance_cb.setChecked(True)
        _select_label(nav, 42)  # review
        _on_delete_clicked(nav)
        assert nav._table.selectedRow() >= 0
        assert _current_label_id(nav) != 42

    def test_no_advance_when_unchecked(self, sample_items):
        nav_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: nav_calls.append(lid)}
        )
        _set_items(nav, sample_items)
        nav._auto_advance_cb.setChecked(False)
        _select_label(nav, 5)  # good (first row)
        _on_mark_good_clicked(nav)
        # Navigate should NOT have been called for auto-advance
        # (only the initial select_label might trigger it)
        # After removing, no new selection should happen
        # The row was removed, so selectedRow may be invalid


# ===========================================================================
# Tests — Editing header display
# ===========================================================================


class TestEditingHeader:
    """Verify editing header shows current label info."""

    def test_shows_label_id_and_tier(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _on_row_clicked(nav, 0)
        r0 = nav._table.row(0)
        header = nav._editing_header.text()
        assert f"Label {r0.label_id}" in header
        assert r0.tier in header

    def test_shows_z_position(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _on_row_clicked(nav, 0)
        r0 = nav._table.row(0)
        header = nav._editing_header.text()
        assert f"Z={r0.z}" in header

    def test_painting_badge_when_no_selection(self):
        """With no row selected and default paint=0, the header shows
        only the Painting badge (no Editing suffix).  Updated 2026-05-16
        — pre-, the header was empty in this state; post-the
        Painting badge is always visible."""
        nav = _make_fake_navigator()
        _set_items(nav, [])
        _update_editing_header(nav)
        assert nav._editing_header.text() == "Painting: background (0)"

    def test_updates_on_navigation(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _on_row_clicked(nav, 0)
        header1 = nav._editing_header.text()
        _on_row_clicked(nav, 1)
        header2 = nav._editing_header.text()
        assert header1 != header2


# ===========================================================================
# Tests — Painting badge
# ===========================================================================


class TestPaintingBadge:
    """The editing header shows a ``Painting: N`` prefix that tracks the
    main layer's ``selected_label``.  Decoupled from the table row
    selection so it reflects the actual brush state — which can
    diverge from the table after auto-increment post-enclosed-fill."""

    def test_default_paint_is_background_zero(self):
        nav = _make_fake_navigator()
        _set_items(nav, [])
        _update_editing_header(nav)
        assert nav._editing_header.text() == "Painting: background (0)"

    def test_set_paint_label_updates_header(self):
        nav = _make_fake_navigator()
        _set_paint_label(nav, 47)
        assert "Painting: 47" in nav._editing_header.text()

    def test_set_paint_label_zero_shows_background(self):
        nav = _make_fake_navigator()
        _set_paint_label(nav, 47)
        _set_paint_label(nav, 0)
        assert "background (0)" in nav._editing_header.text()

    def test_set_paint_label_negative_clamped_to_zero(self):
        """Defensive: ``max(0, int(...))`` in production normalizes
        negative inputs to background.  Bad-data resilience."""
        nav = _make_fake_navigator()
        _set_paint_label(nav, -5)
        assert "background (0)" in nav._editing_header.text()

    def test_paint_and_editing_combined_when_row_selected(
        self, sample_items
    ):
        """When a row is selected AND paint_label is set, both pieces
        show with a separator."""
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _set_paint_label(nav, 47)
        _on_row_clicked(nav, 0)
        text = nav._editing_header.text()
        assert "Painting: 47" in text
        assert "Editing: Label" in text
        assert "  •  " in text

    def test_paint_label_independent_of_table_selection(
        self, sample_items
    ):
        """Painting badge tracks brush state; Editing suffix tracks
        table — they can diverge (e.g., auto-increment after stroke)."""
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _on_row_clicked(nav, 0)
        # Brush bumps to a new label that's NOT in the items list yet
        _set_paint_label(nav, 999)
        text = nav._editing_header.text()
        assert "Painting: 999" in text
        # Editing portion still reflects the clicked row
        r0 = nav._table.row(0)
        assert f"Label {r0.label_id}" in text

    def test_paint_label_persists_across_row_clicks(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _set_paint_label(nav, 123)
        _on_row_clicked(nav, 0)
        _on_row_clicked(nav, 1)
        # Row click updates Editing portion but should NOT clear paint
        assert "Painting: 123" in nav._editing_header.text()


# ===========================================================================
# Tests — Production class has the expected API (smoke check)
# ===========================================================================


class TestProductionPaintLabelAPI:
    """Smoke check that the production CurationLabelNavigator class
    exposes the ``set_paint_label`` method.  The fakes mirror it, but
    we need to confirm the production class is in sync."""

    def test_production_has_set_paint_label(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        assert hasattr(CurationLabelNavigator, "set_paint_label"), (
            "Production CurationLabelNavigator must define set_paint_label"
        )
        assert callable(CurationLabelNavigator.set_paint_label)

    def test_production_has_layer_event_handler(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        assert hasattr(
            CurationLabelNavigator, "_on_layer_selected_label_changed"
        ), (
            "Production must define _on_layer_selected_label_changed "
            "for set_label_layer to wire the events.selected_label "
            "subscription correctly."
        )


# ===========================================================================
# Tests — set_label_layer subscribes to layer.events.selected_label
# ===========================================================================


class _FakeEventEmitter:
    """Stand-in for napari's vispy-derived event emitter."""

    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def disconnect(self, cb):
        if cb in self._callbacks:
            self._callbacks.remove(cb)
        else:
            raise ValueError("callback not connected")

    def emit(self, event=None):
        for cb in list(self._callbacks):
            cb(event)

    def __contains__(self, cb):
        return cb in self._callbacks


class _FakeLayerEvents:
    def __init__(self):
        self.selected_label = _FakeEventEmitter()


class _FakeLayer:
    """Minimal layer with the events emitter API set_label_layer needs."""

    def __init__(self, selected_label=1):
        self.selected_label = selected_label
        self.events = _FakeLayerEvents()


class TestSetLabelLayerSubscription:
    """``set_label_layer`` must subscribe to ``events.selected_label`` so
    the Painting badge tracks the brush state without explicit
    ``set_paint_label`` calls.  Also tests clean disconnect on swap
    (patch switch) and on teardown (``set_label_layer(None)``)."""

    def test_set_label_layer_subscribes(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        # Build a navigator instance just enough to call set_label_layer
        # without going through __init__ (which needs Qt).
        nav = CurationLabelNavigator.__new__(CurationLabelNavigator)
        nav._main_label_layer = None
        nav._paint_label = 0
        nav._selected_label_handler = None
        nav._editing_header = None  # update_editing_header will guard
        # Stub _update_editing_header to be a no-op so we don't need
        # the full Qt widget tree.
        nav._update_editing_header = lambda: None
        nav._on_paint_label_changed_cb = None

        layer = _FakeLayer(selected_label=42)
        nav.set_label_layer(layer)

        assert len(layer.events.selected_label._callbacks) == 1
        assert nav._paint_label == 42

    def test_set_label_layer_disconnects_previous(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        nav = CurationLabelNavigator.__new__(CurationLabelNavigator)
        nav._main_label_layer = None
        nav._paint_label = 0
        nav._selected_label_handler = None
        nav._update_editing_header = lambda: None
        nav._on_paint_label_changed_cb = None

        old_layer = _FakeLayer(selected_label=10)
        new_layer = _FakeLayer(selected_label=20)

        nav.set_label_layer(old_layer)
        assert len(old_layer.events.selected_label._callbacks) == 1

        nav.set_label_layer(new_layer)
        # Old layer must be disconnected
        assert len(old_layer.events.selected_label._callbacks) == 0
        # New layer subscribed
        assert len(new_layer.events.selected_label._callbacks) == 1
        assert nav._paint_label == 20

    def test_set_label_layer_none_disconnects_and_resets(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        nav = CurationLabelNavigator.__new__(CurationLabelNavigator)
        nav._main_label_layer = None
        nav._paint_label = 0
        nav._selected_label_handler = None
        nav._update_editing_header = lambda: None
        nav._on_paint_label_changed_cb = None

        layer = _FakeLayer(selected_label=99)
        nav.set_label_layer(layer)
        assert nav._paint_label == 99

        nav.set_label_layer(None)
        assert len(layer.events.selected_label._callbacks) == 0
        assert nav._main_label_layer is None
        assert nav._paint_label == 0

    def test_event_emission_updates_paint_label(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        nav = CurationLabelNavigator.__new__(CurationLabelNavigator)
        nav._main_label_layer = None
        nav._paint_label = 0
        nav._selected_label_handler = None
        nav._update_editing_header = lambda: None
        nav._on_paint_label_changed_cb = None

        layer = _FakeLayer(selected_label=1)
        nav.set_label_layer(layer)
        assert nav._paint_label == 1

        # Brush autobumps to new label after enclosed-fill
        layer.selected_label = 75
        layer.events.selected_label.emit()
        assert nav._paint_label == 75

    def test_set_label_layer_idempotent_on_same_layer(self):
        """Calling set_label_layer twice with the SAME layer must not
        accumulate handlers — important for patch-switch reload paths
        that re-wire on every flagged-items push."""
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        nav = CurationLabelNavigator.__new__(CurationLabelNavigator)
        nav._main_label_layer = None
        nav._paint_label = 0
        nav._selected_label_handler = None
        nav._update_editing_header = lambda: None
        nav._on_paint_label_changed_cb = None

        layer = _FakeLayer(selected_label=5)
        nav.set_label_layer(layer)
        nav.set_label_layer(layer)
        # Should be only ONE handler — the second call disconnects
        # the first then re-subscribes (net: still one).
        assert len(layer.events.selected_label._callbacks) == 1


# ===========================================================================
# Tests — New Label button
# ===========================================================================


class TestScrollToLabelSilent:
    """``scroll_to_label`` is the silent counterpart of ``select_label``:
    finds the row, scrolls + selects it, but does NOT fire the
    on_row_clicked callbacks.  Used by the auto-pan after
    new-label registration so the user sees the new row land without
    the navigator hijacking the paint focus."""

    def test_production_has_scroll_to_label(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        assert hasattr(CurationLabelNavigator, "scroll_to_label")
        assert callable(CurationLabelNavigator.scroll_to_label)

    def test_production_has_controller_scroller(self):
        """The controller-side helper that defers + dispatches the
        scroll."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        assert hasattr(
            FusionReconstructionController, "_scroll_nav_to_label"
        )

    def test_silent_method_signature(self):
        """``scroll_to_label`` must NOT call _on_row_clicked / dispatch
        any callbacks.  Source-level smoke check that the silent
        implementation doesn't have these patterns."""
        import inspect
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        src = inspect.getsource(CurationLabelNavigator.scroll_to_label)
        # Sanity: must scroll
        assert "scrollToItem" in src
        # Critically: must NOT CALL the callback dispatch.  Check for
        # the call pattern specifically, not the bare name (which
        # appears in the docstring as a reference).
        assert "self._on_row_clicked(" not in src, (
            "scroll_to_label must be silent — do not call "
            "_on_row_clicked or it'll re-trigger navigation and "
            "isolation, defeating the purpose."
        )
        # Must blockSignals to avoid Qt cellClicked side-effects
        assert "blockSignals" in src


class TestNewLabelButton:
    """New Label button bumps selected_label to max(volume)+1 via the
    ``on_new_label`` callback.  Navigator stays decoupled — receiver
    (the annotator) handles the actual label-ID computation and
    cross-viewer propagation."""

    def test_button_present_in_fake(self):
        nav = _make_fake_navigator()
        assert hasattr(nav, "_new_label_btn")

    def test_click_fires_callback(self):
        called = []
        nav = _make_fake_navigator(callbacks={
            "on_new_label": lambda: called.append("fired"),
        })
        _click_new_label(nav)
        assert called == ["fired"]

    def test_click_with_no_callback_is_noop(self):
        nav = _make_fake_navigator()
        # No on_new_label callback wired — must not raise
        _click_new_label(nav)

    def test_callback_exception_is_swallowed(self):
        def _bad():
            raise RuntimeError("simulated annotator bug")
        nav = _make_fake_navigator(callbacks={
            "on_new_label": _bad,
        })
        # Must not raise — _on_new_label_clicked catches everything
        _click_new_label(nav)

    def test_set_callbacks_stores_on_new_label(self):
        nav = _make_fake_navigator()
        marker = lambda: None
        _set_callbacks(nav, on_new_label=marker)
        assert nav._on_new_label_cb is marker


# ===========================================================================
# Tests — Production class has the New Label API (smoke)
# ===========================================================================


class TestProductionNewLabelAPI:
    """Smoke checks that production exposes the New Label surface."""

    def test_navigator_has_new_label_button_attr(self):
        """Production navigator must expose _new_label_btn so the
        styling / lifecycle paths (close cleanup, signal disconnect)
        can find it by name."""
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        # Walk __init__ source — we can't instantiate without Qt, but
        # the attribute assignment must be in the source for any
        # paint/destroy path to find it.
        import inspect
        src = inspect.getsource(CurationLabelNavigator)
        assert "_new_label_btn" in src, (
            "CurationLabelNavigator must create _new_label_btn in "
            "_build_ui (the discoverable UI affordance for paint "
            "auto-increment)"
        )

    def test_navigator_has_click_handler(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        assert hasattr(CurationLabelNavigator, "_on_new_label_clicked")
        assert callable(CurationLabelNavigator._on_new_label_clicked)

    def test_annotator_has_new_label_forwarder(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        assert hasattr(
            Curation3DAnnotator, "_on_new_label_from_navigator"
        ), (
            "Curation3DAnnotator must define _on_new_label_from_navigator "
            "to handle the Navigator's New Label button — uses the "
            "existing _next_label_id_after_accept + _set_selected_label_all "
            "infrastructure."
        )


# ===========================================================================
# Tests — Isolate while painting toggle (companion, 2026-05-16)
# ===========================================================================


class TestPaintIsolationCallbackWiring:
    """The navigator's "Isolate while painting" checkbox forwards its
    state to the annotator's ``on_paint_isolation_toggled`` callback.
    Layer-event subscription also forwards each brush change to
    ``on_paint_label_changed``."""

    def test_set_callbacks_stores_toggle_cb(self):
        nav = _make_fake_navigator()
        marker = lambda enabled: None
        _set_callbacks(nav, on_paint_isolation_toggled=marker)
        assert nav._on_paint_isolation_toggled_cb is marker

    def test_set_callbacks_stores_label_changed_cb(self):
        nav = _make_fake_navigator()
        marker = lambda lid: None
        _set_callbacks(nav, on_paint_label_changed=marker)
        assert nav._on_paint_label_changed_cb is marker


class TestPaintIsolationLayerEventForward:
    """Verify the navigator's existing layer.events.selected_label
    subscription forwards to ``on_paint_label_changed`` so the
    annotator can re-isolate on every brush change.  Reuses the
    bare-instance pattern from TestSetLabelLayerSubscription."""

    def _make_bare_nav(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        nav = CurationLabelNavigator.__new__(CurationLabelNavigator)
        nav._main_label_layer = None
        nav._paint_label = 0
        nav._selected_label_handler = None
        nav._update_editing_header = lambda: None
        nav._on_paint_label_changed_cb = None
        return nav

    def test_event_forwards_to_paint_label_changed_cb(self):
        nav = self._make_bare_nav()
        received = []
        nav._on_paint_label_changed_cb = lambda lid: received.append(lid)
        layer = _FakeLayer(selected_label=1)
        nav.set_label_layer(layer)
        # Clear any forwards from initial set_label_layer
        received.clear()
        layer.selected_label = 99
        layer.events.selected_label.emit()
        assert received == [99]

    def test_event_no_callback_does_not_crash(self):
        nav = self._make_bare_nav()
        # _on_paint_label_changed_cb stays None
        layer = _FakeLayer(selected_label=5)
        nav.set_label_layer(layer)
        layer.events.selected_label.emit()

    def test_event_callback_exception_swallowed(self):
        nav = self._make_bare_nav()
        def _bad(_lid):
            raise RuntimeError("simulated annotator bug")
        nav._on_paint_label_changed_cb = _bad
        layer = _FakeLayer(selected_label=1)
        nav.set_label_layer(layer)
        layer.events.selected_label.emit()


class TestPaintIsolationProductionAPI:
    """Smoke checks that production exposes the required surface."""

    def test_annotator_has_toggle_handler(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        assert hasattr(
            Curation3DAnnotator, "_on_paint_isolation_toggled"
        )

    def test_annotator_has_label_changed_handler(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        assert hasattr(
            Curation3DAnnotator, "_on_paint_label_changed"
        )

    def test_navigator_has_paint_isolate_checkbox_in_source(self):
        import inspect
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        src = inspect.getsource(CurationLabelNavigator)
        assert "_paint_isolate_cb" in src
        assert "Isolate while painting" in src


class TestPaintIsolationAnnotatorBehavior:
    """Behavioral tests on Curation3DAnnotator's paint-isolation
    handlers.  Bare-instance pattern: set just the state fields needed
    and stub _on_isolation_changed to record calls."""

    def _make_bare_annotator(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
        ann._paint_isolation_enabled = False
        ann._paint_isolation_active = False
        ann._paint_isolation_label_id = None
        ann._active_labels_name = "test_layer"
        ann._isolation_calls = []
        ann._on_isolation_changed = lambda ids: ann._isolation_calls.append(
            set(ids)
        )
        ann._fake_layer = types.SimpleNamespace(selected_label=47)
        ann._layer_by_name = lambda layers, name: (
            ann._fake_layer if name == "test_layer" else None
        )
        ann.viewer = types.SimpleNamespace(layers=[])
        return ann

    def test_toggle_on_with_real_label_fires_isolation(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        ann._fake_layer.selected_label = 47
        Curation3DAnnotator._on_paint_isolation_toggled(ann, True)
        assert ann._paint_isolation_enabled is True
        assert ann._paint_isolation_active is True
        assert ann._paint_isolation_label_id == 47
        assert ann._isolation_calls == [{47}]

    def test_toggle_on_with_background_does_not_fire(self):
        """selected_label == 0 (background) waits for a real label
        before firing isolation."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        ann._fake_layer.selected_label = 0
        Curation3DAnnotator._on_paint_isolation_toggled(ann, True)
        assert ann._paint_isolation_enabled is True
        assert ann._paint_isolation_active is False
        assert ann._isolation_calls == []

    def test_toggle_off_clears_isolation(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        ann._fake_layer.selected_label = 47
        Curation3DAnnotator._on_paint_isolation_toggled(ann, True)
        Curation3DAnnotator._on_paint_isolation_toggled(ann, False)
        assert ann._paint_isolation_enabled is False
        assert ann._paint_isolation_active is False
        assert ann._paint_isolation_label_id is None
        assert ann._isolation_calls == [{47}, set()]

    def test_toggle_off_when_not_active_no_op(self):
        """Don't clobber navigator-driven isolation that might be
        active independently."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        Curation3DAnnotator._on_paint_isolation_toggled(ann, False)
        assert ann._isolation_calls == []

    def test_idempotent_toggle_same_state(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        Curation3DAnnotator._on_paint_isolation_toggled(ann, False)
        assert ann._isolation_calls == []

    def test_label_changed_fires_when_enabled(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        ann._paint_isolation_enabled = True
        Curation3DAnnotator._on_paint_label_changed(ann, 99)
        assert ann._isolation_calls == [{99}]
        assert ann._paint_isolation_active is True
        assert ann._paint_isolation_label_id == 99

    def test_label_changed_skipped_when_disabled(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        Curation3DAnnotator._on_paint_label_changed(ann, 99)
        assert ann._isolation_calls == []
        assert ann._paint_isolation_active is False

    def test_label_changed_skipped_for_background(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        ann._paint_isolation_enabled = True
        Curation3DAnnotator._on_paint_label_changed(ann, 0)
        assert ann._isolation_calls == []

    def test_label_changed_deduplicates_same_label(self):
        """Same label firing repeatedly only triggers one isolation
        call — saves the ~245ms cost on redundant events."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        ann._paint_isolation_enabled = True
        for _ in range(5):
            Curation3DAnnotator._on_paint_label_changed(ann, 42)
        assert ann._isolation_calls == [{42}]

    def test_label_changed_re_fires_on_different_label(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        ann._paint_isolation_enabled = True
        Curation3DAnnotator._on_paint_label_changed(ann, 42)
        Curation3DAnnotator._on_paint_label_changed(ann, 99)
        Curation3DAnnotator._on_paint_label_changed(ann, 100)
        assert ann._isolation_calls == [{42}, {99}, {100}]
        assert ann._paint_isolation_label_id == 100

    def test_label_changed_handles_invalid_input(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator()
        ann._paint_isolation_enabled = True
        Curation3DAnnotator._on_paint_label_changed(ann, "not-an-int")
        assert ann._isolation_calls == []


# ===========================================================================
# Tests — Untriaged-row indicator (qc_tier == "")
# Updated 2026-05-16: criterion changed from feature_state to qc_tier
# because feature_state == "stale" caught too many rows (split
# fragments, post-accept ghosts, etc.).  qc_tier == "" matches the
# user's "newly painted, not yet curated" mental model.
# ===========================================================================


class TestUntriagedRowIndicator:
    """Rows with empty qc_tier (canonical "newly painted, no tier
    decision yet" state) get visually distinct treatment:
    italic font + dark-amber background + tooltip in production.

    Other rows — including curated rows with feature_state="stale"
    (which are common after curation operations) — are NOT marked.
    """

    def test_untriaged_flag_propagated_to_table_row(self):
        nav = _make_fake_navigator()
        _set_items(nav, [
            {"label_id": 7, "tier": "", "centroid": (5, 10, 10),
             "area": 100},
            {"label_id": 8, "tier": "curated", "centroid": (5, 20, 20),
             "area": 200},
        ])
        rows = [nav._table.row(i) for i in range(nav._table.rowCount())]
        flag_map = {r.label_id: r.is_untriaged for r in rows}
        assert flag_map[7] is True   # empty qc_tier
        assert flag_map[8] is False  # curated

    def test_curated_with_stale_features_NOT_marked(self):
        """Critical: curated rows with feature_state == "stale" must
        NOT be marked.  This was the bug observed 2026-05-16 — every
        curated label was italicized after a curation session.  The
        new criterion (qc_tier == "") narrows the indicator to truly
        untriaged rows."""
        nav = _make_fake_navigator()
        _set_items(nav, [
            {"label_id": 7, "tier": "curated", "centroid": (0, 0, 0),
             "area": 100, "feature_state": "stale"},
        ])
        row = nav._table.row(0)
        assert row.is_untriaged is False, (
            "curated rows must NOT be marked even when "
            "feature_state == 'stale'"
        )

    def test_review_tier_NOT_marked(self):
        """A review-tier row (the default fallback) is also not
        untriaged.  Only explicitly-empty qc_tier marks the row."""
        nav = _make_fake_navigator()
        _set_items(nav, [
            {"label_id": 7, "tier": "review", "centroid": (0, 0, 0),
             "area": 100},
        ])
        row = nav._table.row(0)
        assert row.is_untriaged is False

    def test_each_tier_value_check(self):
        """Sweep all known tier values — only empty string marks."""
        nav = _make_fake_navigator()
        _set_items(nav, [
            {"label_id": 1, "tier": "curated", "centroid": (0, 0, 0), "area": 1},
            {"label_id": 2, "tier": "good", "centroid": (0, 0, 0), "area": 1},
            {"label_id": 3, "tier": "review", "centroid": (0, 0, 0), "area": 1},
            {"label_id": 4, "tier": "bad", "centroid": (0, 0, 0), "area": 1},
            {"label_id": 5, "tier": "garbage", "centroid": (0, 0, 0), "area": 1},
            {"label_id": 6, "tier": "", "centroid": (0, 0, 0), "area": 1},
        ])
        rows = {r.label_id: r.is_untriaged for r in (
            nav._table.row(i) for i in range(nav._table.rowCount())
        )}
        assert rows[1] is False
        assert rows[2] is False
        assert rows[3] is False
        assert rows[4] is False
        assert rows[5] is False
        assert rows[6] is True

    def test_item_data_carries_tier_and_feature_state(self):
        """The full item dict stored on the row preserves both tier
        AND feature_state for downstream consumers (filters, exports,
        any future indicator that needs both signals)."""
        nav = _make_fake_navigator()
        _set_items(nav, [
            {"label_id": 7, "tier": "", "centroid": (0, 0, 0),
             "area": 1, "feature_state": "stale"},
        ])
        row = nav._table.row(0)
        assert row.item_data.get("feature_state") == "stale"
        assert row.item_data.get("tier") == ""


# ===========================================================================
# Tests — Controller _build_flagged_label_items includes feature_state
# ===========================================================================


class TestControllerItemDictIncludesFeatureState:
    """Smoke check: the controller path that builds items for the
    navigator must include feature_state in the dict so the navigator
    can render the indicator."""

    def test_controller_method_emits_feature_state(self):
        """Inspect the source of _build_flagged_label_items to confirm
        it constructs items with a feature_state key.  We can't easily
        instantiate the controller without the full app, so this is a
        source-level smoke check — same pattern as the
        TestProductionNewLabelAPI tests above."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._build_flagged_label_items
        )
        assert '"feature_state"' in src, (
            "_build_flagged_label_items must include feature_state in "
            "the item dict so the navigator can render the stale-row "
            "indicator."
        )


# ===========================================================================
# Tests — remove_item and update_item_tier
# ===========================================================================


class TestRemoveAndUpdate:
    """Verify incremental update methods."""

    def test_remove_item_removes_row(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _remove_item(nav, 42)
        assert nav._table.rowCount() == 3
        remaining = [nav._table.row(i).label_id for i in range(3)]
        assert 42 not in remaining

    def test_remove_updates_summary(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _remove_item(nav, 91)  # garbage
        text = nav._summary_label.text()
        assert "3 labels" in text
        assert "garbage" not in text

    def test_remove_nonexistent_is_noop(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _remove_item(nav, 999)
        assert nav._table.rowCount() == 4

    def test_update_item_tier(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _update_item_tier(nav, 42, "good")
        # Find the row with label 42 and check tier
        for i in range(nav._table.rowCount()):
            r = nav._table.row(i)
            if r.label_id == 42:
                assert r.tier == "good"
                break

    def test_update_tier_updates_summary(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _update_item_tier(nav, 42, "good")
        text = nav._summary_label.text()
        assert "2 good" in text

    def test_update_tier_respects_filter(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._tier_filter_good.setChecked(False)
        _apply_filter(nav)
        # Update review→good, now it should be excluded from table
        _update_item_tier(nav, 42, "good")
        remaining_ids = [nav._table.row(i).label_id for i in range(nav._table.rowCount())]
        assert 42 not in remaining_ids

    def test_remove_garbage_disables_discard_button(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        assert nav._discard_garbage_btn.isEnabled() is True
        _remove_item(nav, 91)  # only garbage item
        assert nav._discard_garbage_btn.isEnabled() is False


# ===========================================================================
# Tests — current_label_id
# ===========================================================================


class TestCurrentLabelId:
    """Verify current_label_id returns selected row's label."""

    def test_returns_none_when_no_selection(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        assert _current_label_id(nav) is None

    def test_returns_label_after_selection(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _select_label(nav, 42)
        assert _current_label_id(nav) == 42

    def test_returns_label_after_row_click(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _on_row_clicked(nav, 2)
        r2 = nav._table.row(2)
        assert _current_label_id(nav) == r2.label_id

    def test_select_label_selects_correct_row(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _select_label(nav, 91)
        assert _current_label_id(nav) == 91


# ===========================================================================
# Helper — isolate toggle
# ===========================================================================


def _on_isolate_toggled(nav, checked):
    """Simulate toggling the isolate checkbox."""
    nav._isolate_cb.setChecked(checked)
    if nav._on_show_selected_label_changed_cb:
        nav._on_show_selected_label_changed_cb(checked)


# ===========================================================================
# Helper — arrow key simulation
# ===========================================================================


def _handle_key_press(nav, key):
    """Simulate arrow key press on the table.

    Mirrors the event filter in the real implementation:
    Up/Down → navigate labels, Left/Right → scroll hovered subviewer.
    """
    if key == "Down":
        _navigate_next(nav)
    elif key == "Up":
        _navigate_prev(nav)
    elif key in ("Left", "Right"):
        cb = getattr(nav, "_on_scroll_subviewer_cb", None)
        if cb:
            try:
                cb(key.lower())
            except Exception:
                pass


# ===========================================================================
# Tests — Isolate on select
# ===========================================================================


class TestIsolateOnSelect:
    """Verify isolate checkbox calls callback to toggle show_selected_label."""

    def test_isolate_default_unchecked(self):
        nav = _make_fake_navigator()
        assert nav._isolate_cb.isChecked() is False

    def test_toggle_on_calls_callback_with_true(self):
        calls = []
        nav = _make_fake_navigator(
            callbacks={
                "on_show_selected_label_changed": lambda v: calls.append(v)
            }
        )
        _on_isolate_toggled(nav, True)
        assert calls == [True]

    def test_toggle_off_calls_callback_with_false(self):
        calls = []
        nav = _make_fake_navigator(
            callbacks={
                "on_show_selected_label_changed": lambda v: calls.append(v)
            }
        )
        _on_isolate_toggled(nav, True)
        _on_isolate_toggled(nav, False)
        assert calls == [True, False]

    def test_no_callback_no_crash(self):
        nav = _make_fake_navigator()
        # No callbacks set — should not raise
        _on_isolate_toggled(nav, True)
        _on_isolate_toggled(nav, False)

    def test_checkbox_state_updates(self):
        nav = _make_fake_navigator()
        _on_isolate_toggled(nav, True)
        assert nav._isolate_cb.isChecked() is True
        _on_isolate_toggled(nav, False)
        assert nav._isolate_cb.isChecked() is False


# ===========================================================================
# Tests — Arrow key navigation
# ===========================================================================


class TestArrowKeyNavigation:
    """Verify Up/Down arrow keys navigate the table like Prev/Next buttons."""

    def test_down_arrow_advances(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        _handle_key_press(nav, "Down")
        assert nav._table.selectedRow() == 1

    def test_up_arrow_goes_back(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(2)
        _handle_key_press(nav, "Up")
        assert nav._table.selectedRow() == 1

    def test_down_arrow_wraps_around(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(3)  # last row
        _handle_key_press(nav, "Down")
        assert nav._table.selectedRow() == 0

    def test_up_arrow_wraps_around(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        _handle_key_press(nav, "Up")
        assert nav._table.selectedRow() == 3

    def test_down_arrow_triggers_navigate_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: calls.append(lid)}
        )
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        _handle_key_press(nav, "Down")
        assert len(calls) == 1

    def test_up_arrow_triggers_navigate_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: calls.append(lid)}
        )
        _set_items(nav, sample_items)
        nav._table.selectRow(1)
        _handle_key_press(nav, "Up")
        assert len(calls) == 1
        assert calls[0] == nav._table.row(0).label_id

    def test_down_arrow_triggers_select_label_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_select_label": lambda lid: calls.append(lid)}
        )
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        _handle_key_press(nav, "Down")
        assert len(calls) == 1

    def test_arrows_work_with_filter(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # Exclude review — table has: good, bad, garbage (3 rows)
        nav._tier_filter_review.setChecked(False)
        _apply_filter(nav)
        nav._table.selectRow(0)  # good
        _handle_key_press(nav, "Down")
        r = nav._table.row(nav._table.selectedRow())
        assert r.tier == "bad"  # review excluded, next is bad


# ===========================================================================
# Tests — Focus restoration after navigation
# ===========================================================================


class TestFocusRestoration:
    """Verify table reclaims focus after callbacks (subviewers can steal it)."""

    def test_row_click_restores_focus(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        before = nav._table._focus_count
        _on_row_clicked(nav, 0)
        assert nav._table._focus_count > before

    def test_navigate_next_restores_focus(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        before = nav._table._focus_count
        _navigate_next(nav)
        assert nav._table._focus_count > before

    def test_navigate_prev_restores_focus(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(1)
        before = nav._table._focus_count
        _navigate_prev(nav)
        assert nav._table._focus_count > before

    def test_arrow_key_restores_focus(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        before = nav._table._focus_count
        _handle_key_press(nav, "Down")
        assert nav._table._focus_count > before


# ===========================================================================
# Tests — Left/Right scroll routing to subviewer
# ===========================================================================


class TestLeftRightScrollRouting:
    """Verify Left/Right arrow keys on navigator table call scroll callback."""

    def test_right_key_calls_scroll_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_scroll_subviewer": lambda d: calls.append(d)}
        )
        _set_items(nav, sample_items)
        _handle_key_press(nav, "Right")
        assert calls == ["right"]

    def test_left_key_calls_scroll_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_scroll_subviewer": lambda d: calls.append(d)}
        )
        _set_items(nav, sample_items)
        _handle_key_press(nav, "Left")
        assert calls == ["left"]

    def test_no_callback_no_error(self, sample_items):
        nav = _make_fake_navigator()  # no scroll callback
        _set_items(nav, sample_items)
        _handle_key_press(nav, "Right")  # should not raise

    def test_multiple_presses(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_scroll_subviewer": lambda d: calls.append(d)}
        )
        _set_items(nav, sample_items)
        _handle_key_press(nav, "Right")
        _handle_key_press(nav, "Right")
        _handle_key_press(nav, "Left")
        assert calls == ["right", "right", "left"]

    def test_left_right_dont_navigate_labels(self, sample_items):
        nav_calls = []
        nav = _make_fake_navigator(
            callbacks={
                "on_navigate": lambda lid, c: nav_calls.append(lid),
                "on_scroll_subviewer": lambda d: None,
            }
        )
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        _handle_key_press(nav, "Right")
        assert nav._table.selectedRow() == 0  # unchanged
        assert nav_calls == []  # no navigate callback


# ===========================================================================
# K13.1 — Curated tier in Navigator
# ===========================================================================

class TestCuratedTierFilter:
    """Curated tier checkbox, filtering, sorting, and summary."""

    @pytest.fixture()
    def curated_items(self):
        return [
            {"label_id": 1, "tier": "curated", "centroid": (10, 20, 30), "area": 500},
            {"label_id": 2, "tier": "good", "centroid": (15, 25, 35), "area": 400},
            {"label_id": 3, "tier": "review", "centroid": (20, 30, 40), "area": 300},
            {"label_id": 4, "tier": "bad", "centroid": (25, 35, 45), "area": 200},
            {"label_id": 5, "tier": "garbage", "centroid": (30, 40, 50), "area": 100},
        ]

    def test_curated_items_visible_by_default(self, curated_items):
        """Curated checkbox is checked by default, so curated rows appear."""
        nav = _make_fake_navigator()
        _set_items(nav, curated_items)
        assert nav._table.rowCount() == 5
        # Curated row should be in the table
        label_ids = [nav._table.row(i).label_id for i in range(nav._table.rowCount())]
        assert 1 in label_ids

    def test_uncheck_curated_excludes_curated(self, curated_items):
        """Unchecking 'C' hides curated rows."""
        nav = _make_fake_navigator()
        _set_items(nav, curated_items)
        assert nav._table.rowCount() == 5

        nav._tier_filter_curated.setChecked(False)
        _rebuild_table(nav)
        assert nav._table.rowCount() == 4
        label_ids = [nav._table.row(i).label_id for i in range(nav._table.rowCount())]
        assert 1 not in label_ids

    def test_curated_appears_in_summary(self, curated_items):
        """Summary text includes curated count."""
        nav = _make_fake_navigator()
        _set_items(nav, curated_items)
        summary = nav._summary_label.text()
        assert "1 curated" in summary

    def test_curated_sorts_before_good(self, curated_items):
        """When sorted by Tier, curated appears before good (TIER_ORDER=0)."""
        nav = _make_fake_navigator()
        nav._sort_combo._items = ["Tier", "Area (desc)", "Z (asc)", "ID (asc)", "Score (asc)"]
        nav._sort_combo._index = 0  # "Tier"
        _set_items(nav, curated_items)
        # First row should be curated (tier order 0)
        first = nav._table.row(0)
        assert first.tier == "curated"
        second = nav._table.row(1)
        assert second.tier == "good"

    def test_update_item_tier_to_curated(self, curated_items):
        """update_item_tier can change a label's tier to curated."""
        nav = _make_fake_navigator()
        _set_items(nav, curated_items)
        # Change label 3 (review) to curated
        _update_item_tier(nav, label_id=3, new_tier="curated")
        # Verify internal state
        item = next(it for it in nav._items if it["label_id"] == 3)
        assert item["tier"] == "curated"
        # Verify summary updated
        summary = nav._summary_label.text()
        assert "2 curated" in summary


# ===================================================================
# Recalculate QC button
# ===================================================================


class TestRecalculateButton:
    """Recalculate QC button behaviour — enabled when curated labels exist."""

    def test_recalc_button_disabled_when_no_curated(self):
        """Button disabled when no curated labels present."""
        nav = _make_fake_navigator()
        items = [
            {"label_id": 1, "tier": "good", "centroid": (10, 20, 30), "area": 500},
            {"label_id": 2, "tier": "review", "centroid": (15, 25, 35), "area": 400},
        ]
        _set_items(nav, items)
        assert not nav._recalc_qc_btn.isEnabled()

    def test_recalc_button_enabled_when_curated_exists(self):
        """Button enabled when at least one curated label exists."""
        nav = _make_fake_navigator()
        items = [
            {"label_id": 1, "tier": "curated", "centroid": (10, 20, 30), "area": 500},
            {"label_id": 2, "tier": "good", "centroid": (15, 25, 35), "area": 400},
        ]
        _set_items(nav, items)
        assert nav._recalc_qc_btn.isEnabled()

    def test_recalc_click_fires_callback_with_curated_ids(self):
        """Clicking recalculate fires callback with list of curated label IDs."""
        received = []
        nav = _make_fake_navigator(callbacks={"on_recalculate": lambda ids: received.extend(ids)})
        items = [
            {"label_id": 1, "tier": "curated", "centroid": (10, 20, 30), "area": 500},
            {"label_id": 2, "tier": "good", "centroid": (15, 25, 35), "area": 400},
            {"label_id": 3, "tier": "curated", "centroid": (20, 30, 40), "area": 300},
        ]
        _set_items(nav, items)
        _on_recalc_clicked(nav)
        assert sorted(received) == [1, 3]

    def test_recalc_button_disabled_after_no_curated(self):
        """Button becomes disabled after curated items are removed."""
        nav = _make_fake_navigator()
        items = [
            {"label_id": 1, "tier": "curated", "centroid": (10, 20, 30), "area": 500},
            {"label_id": 2, "tier": "good", "centroid": (15, 25, 35), "area": 400},
        ]
        _set_items(nav, items)
        assert nav._recalc_qc_btn.isEnabled()
        # Change curated back to good
        _update_item_tier(nav, label_id=1, new_tier="good")
        assert not nav._recalc_qc_btn.isEnabled()


# ===================================================================
# Tier-colored ID cells
# ===================================================================


class TestTierColoredIDCells:
    """ID column text should be colored by tier for visual scanning."""

    def test_id_cell_foreground_matches_tier(self):
        """Each row's ID cell tier matches its stored tier."""
        nav = _make_fake_navigator()
        items = [
            {"label_id": 1, "tier": "curated", "centroid": (10, 20, 30), "area": 500},
            {"label_id": 2, "tier": "good", "centroid": (15, 25, 35), "area": 400},
            {"label_id": 3, "tier": "review", "centroid": (20, 30, 40), "area": 300},
            {"label_id": 4, "tier": "bad", "centroid": (25, 35, 45), "area": 200},
            {"label_id": 5, "tier": "garbage", "centroid": (30, 40, 50), "area": 100},
        ]
        _set_items(nav, items)
        # Each row's tier is stored in the row data — verify tiers are correct
        for i in range(nav._table.rowCount()):
            row = nav._table.row(i)
            assert row.tier in TIER_COLORS

    def test_curated_id_cell_is_cyan(self):
        """Curated label's tier color is cyan (#00bcd4)."""
        assert TIER_COLORS["curated"] == "#00bcd4"


# ===================================================================
# Multi-select — basic selection
# ===================================================================


class TestMultiSelect:
    """Verify Ctrl+Click multi-select and isolation callback."""

    def test_ctrl_click_adds_to_selection(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # Plain click row 0 (selects only row 0)
        nav._table.selectRow(0)
        assert nav._table.selectedRows() == {0}
        # Ctrl+Click row 2 (adds to selection)
        nav._table.addToSelection(2)
        assert nav._table.selectedRows() == {0, 2}

    def test_ctrl_click_toggles_off(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        nav._table.addToSelection(2)
        assert nav._table.selectedRows() == {0, 2}
        # Ctrl+Click row 2 again to deselect
        nav._table.addToSelection(2)
        assert nav._table.selectedRows() == {0}

    def test_plain_click_clears_multi_select(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.addToSelection(0)
        nav._table.addToSelection(1)
        nav._table.addToSelection(2)
        assert len(nav._table.selectedRows()) == 3
        # Plain click (selectRow) clears multi and selects one
        nav._table.selectRow(1)
        assert nav._table.selectedRows() == {1}

    def test_isolation_fires_with_all_selected_ids(self, sample_items):
        isolation_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_isolation_changed": lambda ids: isolation_calls.append(ids)}
        )
        _set_items(nav, sample_items)
        nav._isolate_cb.setChecked(True)
        # Multi-select rows 0 and 2
        nav._table.selectRow(0)
        nav._table.addToSelection(2)
        _fire_isolation_if_active(nav)
        assert len(isolation_calls) == 1
        r0 = nav._table.row(0)
        r2 = nav._table.row(2)
        assert isolation_calls[0] == {r0.label_id, r2.label_id}

    def test_isolation_not_fired_when_isolate_off(self, sample_items):
        isolation_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_isolation_changed": lambda ids: isolation_calls.append(ids)}
        )
        _set_items(nav, sample_items)
        nav._isolate_cb.setChecked(False)
        nav._table.addToSelection(0)
        nav._table.addToSelection(1)
        _fire_isolation_if_active(nav)
        assert len(isolation_calls) == 0

    def test_selected_label_ids_returns_correct_set(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.addToSelection(0)
        nav._table.addToSelection(3)
        ids = _selected_label_ids(nav)
        r0 = nav._table.row(0)
        r3 = nav._table.row(3)
        assert ids == {r0.label_id, r3.label_id}


# ===================================================================
# Multi-select — navigation cycling
# ===================================================================


class TestMultiSelectNavigation:
    """Verify Up/Down cycles within multi-selected rows only."""

    def test_next_cycles_within_selected(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # Select rows 0 and 2
        nav._table.addToSelection(0)
        nav._table.addToSelection(2)
        nav._table._selected_row = 0  # Start at row 0
        _navigate_next(nav)
        assert nav._table.selectedRow() == 2

    def test_next_wraps_within_selected(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.addToSelection(0)
        nav._table.addToSelection(2)
        nav._table._selected_row = 2  # Start at row 2
        _navigate_next(nav)
        assert nav._table.selectedRow() == 0  # Wraps to first selected

    def test_prev_cycles_within_selected(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.addToSelection(0)
        nav._table.addToSelection(2)
        nav._table._selected_row = 2
        _navigate_prev(nav)
        assert nav._table.selectedRow() == 0

    def test_prev_wraps_within_selected(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.addToSelection(0)
        nav._table.addToSelection(2)
        nav._table._selected_row = 0
        _navigate_prev(nav)
        assert nav._table.selectedRow() == 2  # Wraps to last selected

    def test_single_select_still_cycles_all(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(0)  # Single select
        _navigate_next(nav)
        assert nav._table.selectedRow() == 1  # Next in full table

    def test_multi_select_nav_preserves_selection(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.addToSelection(0)
        nav._table.addToSelection(2)
        nav._table._selected_row = 0
        _navigate_next(nav)
        # Selection set should be unchanged (setCurrentCell preserves it)
        assert nav._table.selectedRows() == {0, 2}

    def test_nav_fires_navigate_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: calls.append(lid)}
        )
        _set_items(nav, sample_items)
        nav._table.addToSelection(0)
        nav._table.addToSelection(2)
        nav._table._selected_row = 0
        _navigate_next(nav)
        assert len(calls) == 1
        assert calls[0] == nav._table.row(2).label_id


# ===================================================================
# Multi-select — rebuild preserves selection
# ===================================================================


class TestMultiSelectRebuild:
    """Verify table rebuild preserves multi-selection."""

    def test_rebuild_preserves_multi_selection(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # Multi-select labels 5 and 17
        for i in range(nav._table.rowCount()):
            r = nav._table.row(i)
            if r.label_id in (5, 17):
                nav._table._selected_rows.add(i)
        # Rebuild (e.g., sort change)
        nav._sort_combo.setCurrentText("ID (asc)")
        _rebuild_table(nav)
        # Labels 5 and 17 should still be selected
        ids = _selected_label_ids(nav)
        assert 5 in ids
        assert 17 in ids

    def test_rebuild_drops_filtered_out_labels(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # Multi-select all labels
        for i in range(nav._table.rowCount()):
            nav._table._selected_rows.add(i)
        # Filter out garbage
        nav._tier_filter_garbage.setChecked(False)
        _rebuild_table(nav)
        ids = _selected_label_ids(nav)
        assert 91 not in ids  # garbage label gone


# ===================================================================
# Multi-select — select_labels (programmatic multi-select)
# ===================================================================


class TestSelectLabels:
    """Verify select_labels() for post-split auto-selection."""

    def test_select_labels_selects_multiple(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _select_labels(nav, [5, 17])
        ids = _selected_label_ids(nav)
        assert ids == {5, 17}

    def test_select_labels_navigates_to_first(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: calls.append(lid)}
        )
        _set_items(nav, sample_items)
        _select_labels(nav, [42, 17])
        # Should navigate to first matching row (tier-sorted: review=42 before bad=17)
        assert len(calls) == 1
        assert calls[0] == 42

    def test_select_labels_triggers_isolation(self, sample_items):
        isolation_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_isolation_changed": lambda ids: isolation_calls.append(ids)}
        )
        _set_items(nav, sample_items)
        nav._isolate_cb.setChecked(True)
        _select_labels(nav, [5, 91])
        assert len(isolation_calls) >= 1
        assert {5, 91} == isolation_calls[-1]

    def test_select_labels_no_isolation_when_off(self, sample_items):
        isolation_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_isolation_changed": lambda ids: isolation_calls.append(ids)}
        )
        _set_items(nav, sample_items)
        nav._isolate_cb.setChecked(False)
        _select_labels(nav, [5, 91])
        assert len(isolation_calls) == 0


# ===================================================================
# B5 crash fix — selection-model blockSignals guards
# ===================================================================


class _RecordingSelectionModel:
    """Fake QItemSelectionModel that records blockSignals() calls and
    asserts clearSelection() is only ever called while signals are
    blocked.  Mirrors the production guard pattern in _rebuild_table."""

    def __init__(self):
        self.events: list = []  # ordered log of operations
        self._blocked = False
        self.clear_while_unblocked = False  # invariant violation flag

    def blockSignals(self, flag):
        self._blocked = bool(flag)
        self.events.append(("blockSignals", bool(flag)))

    def clearSelection(self):
        if not self._blocked:
            self.clear_while_unblocked = True
        self.events.append(("clearSelection",))

    def select(self, idx, flags):
        self.events.append(("select",))

    def setCurrentIndex(self, idx, flags):
        self.events.append(("setCurrentIndex",))


class _MiniIndex:
    def __init__(self, row):
        self._row = row

    def row(self):
        return self._row


class _MiniModel:
    def index(self, row, col):
        return _MiniIndex(row)


class _MiniTable:
    """Minimal QTableWidget stand-in for driving the PRODUCTION
    select_labels / clear_selection / reset_selection_ui methods."""

    def __init__(self, sel_model, n_rows=0):
        self._sel_model = sel_model
        self._n_rows = n_rows

    def selectionModel(self):
        return self._sel_model

    def rowCount(self):
        return self._n_rows

    def item(self, row, col):
        return None  # no rows match → first_row stays None

    def model(self):
        return _MiniModel()

    def scrollToItem(self, item):
        pass

    def setCurrentCell(self, row, col):
        pass


def _make_bare_nav_with_recording_selmodel(n_rows=0):
    """Build a SimpleNamespace with the minimum attributes the three
    production methods touch, plus a recording selection model."""
    sel_model = _RecordingSelectionModel()
    nav = types.SimpleNamespace()
    nav._table = _MiniTable(sel_model, n_rows=n_rows)
    nav._rebuild_pending = False
    nav._editing_header = _FakeLabel("")
    nav._isolate_cb = _FakeCheckBox("Isolate", checked=True)
    # reset_selection_ui calls _isolate_cb.blockSignals — _FakeCheckBox
    # doesn't define it; attach a no-op on the instance.
    nav._isolate_cb.blockSignals = lambda flag: None
    nav._on_isolation_changed_cb = None
    nav._suppress_isolation_dispatch = True  # skip isolation dispatch
    # Track re-fire of the selection-changed handler.
    nav._table_selection_changed_calls = 0
    return nav, sel_model


class TestSelectionModelBlockSignalsGuards:
    """B5: select_labels / clear_selection / reset_selection_ui must wrap
    their clearSelection() in sel_model.blockSignals(True/False) so the
    selection-model's own signals can't re-enter
    _on_table_selection_changed while the table is being rebuilt during
    a patch switch (the Windows access-violation crash)."""

    def _bind(self, nav, method_name):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        real = getattr(CurationLabelNavigator, method_name)
        setattr(nav, method_name, types.MethodType(real, nav))

    def test_select_labels_blocks_signals_around_clear(self):
        nav, sel = _make_bare_nav_with_recording_selmodel(n_rows=0)
        # Re-fire target + isolation helper — bind as no-ops.
        nav._on_table_selection_changed = lambda: setattr(
            nav, "_table_selection_changed_calls",
            nav._table_selection_changed_calls + 1,
        )
        nav._fire_isolation_if_active = lambda: None
        self._bind(nav, "select_labels")

        nav.select_labels([1, 2, 3])

        # clearSelection happened, and ONLY while signals were blocked.
        assert ("clearSelection",) in sel.events
        assert not sel.clear_while_unblocked
        # blockSignals(True) precedes the clearSelection, blockSignals(False) follows.
        names = sel.events
        i_true = names.index(("blockSignals", True))
        i_clear = names.index(("clearSelection",))
        i_false = names.index(("blockSignals", False))
        assert i_true < i_clear < i_false

    def test_select_labels_refires_selection_changed_when_stable(self):
        """Behavior preserved: with no rebuild pending, the blocked
        selectionChanged is re-fired ONCE so the render button arms +
        batch count updates."""
        nav, sel = _make_bare_nav_with_recording_selmodel(n_rows=0)
        nav._on_table_selection_changed = lambda: setattr(
            nav, "_table_selection_changed_calls",
            nav._table_selection_changed_calls + 1,
        )
        nav._fire_isolation_if_active = lambda: None
        self._bind(nav, "select_labels")

        nav.select_labels([1])

        assert nav._table_selection_changed_calls == 1

    def test_select_labels_bails_when_rebuild_pending(self):
        """A stale lasso mid-switch (rebuild pending) is dropped — no
        selection-model mutation, no re-fire."""
        nav, sel = _make_bare_nav_with_recording_selmodel(n_rows=0)
        nav._rebuild_pending = True
        nav._on_table_selection_changed = lambda: setattr(
            nav, "_table_selection_changed_calls",
            nav._table_selection_changed_calls + 1,
        )
        nav._fire_isolation_if_active = lambda: None
        self._bind(nav, "select_labels")

        nav.select_labels([1, 2])

        assert sel.events == []  # never touched the selection model
        assert nav._table_selection_changed_calls == 0

    def test_clear_selection_blocks_signals_around_clear(self):
        nav, sel = _make_bare_nav_with_recording_selmodel(n_rows=0)
        nav.set_isolate_checked = lambda checked: None
        self._bind(nav, "clear_selection")

        nav.clear_selection()

        assert ("clearSelection",) in sel.events
        assert not sel.clear_while_unblocked
        i_true = sel.events.index(("blockSignals", True))
        i_clear = sel.events.index(("clearSelection",))
        i_false = sel.events.index(("blockSignals", False))
        assert i_true < i_clear < i_false

    def test_reset_selection_ui_blocks_signals_around_clear(self):
        nav, sel = _make_bare_nav_with_recording_selmodel(n_rows=0)
        self._bind(nav, "reset_selection_ui")

        nav.reset_selection_ui()

        assert ("clearSelection",) in sel.events
        assert not sel.clear_while_unblocked
        i_true = sel.events.index(("blockSignals", True))
        i_clear = sel.events.index(("clearSelection",))
        i_false = sel.events.index(("blockSignals", False))
        assert i_true < i_clear < i_false


# ===================================================================
# Multi-select — select_label triggers callbacks
# ===================================================================


class TestSelectLabelCallbacks:
    """Verify select_label() (single) triggers navigation callbacks."""

    def test_select_label_fires_navigate(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: calls.append((lid, c))}
        )
        _set_items(nav, sample_items)
        _select_label(nav, 42)
        assert len(calls) == 1
        assert calls[0][0] == 42

    def test_select_label_fires_select_callback(self, sample_items):
        calls = []
        nav = _make_fake_navigator(
            callbacks={"on_select_label": lambda lid: calls.append(lid)}
        )
        _set_items(nav, sample_items)
        _select_label(nav, 17)
        assert calls == [17]

    def test_select_label_updates_current_row(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        _select_label(nav, 91)
        assert _current_label_id(nav) == 91


# ===================================================================
# Multi-select — editing header shows count
# ===================================================================


class TestMultiSelectEditingHeader:
    """Verify editing header shows selection count when multiple selected."""

    def test_header_shows_count_for_multi(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.addToSelection(0)
        nav._table.addToSelection(1)
        nav._table.addToSelection(2)
        _update_editing_header(nav)
        text = nav._editing_header.text()
        assert "[3 selected]" in text

    def test_header_no_count_for_single(self, sample_items):
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        nav._table.selectRow(0)
        _update_editing_header(nav)
        text = nav._editing_header.text()
        assert "selected" not in text


# ===========================================================================
# Tests — select_next_triage
# ===========================================================================


class TestSelectNextTriage:
    """Verify select_next_triage finds the first review/bad/garbage label."""

    def test_finds_first_review_label(self, sample_items):
        """With default tier sort, first triage label should be 'review'."""
        nav = _make_fake_navigator()
        _set_items(nav, sample_items)
        # Default sort is Tier: good(5), review(42), bad(17), garbage(91)
        result = _select_next_triage(nav)
        assert result is True
        assert _current_label_id(nav) == 42  # review tier

    def test_finds_bad_when_no_review(self):
        """When no review labels exist, should find the first bad label."""
        items = [
            {"label_id": 1, "tier": "good", "centroid": (0, 0, 0), "area": 100},
            {"label_id": 2, "tier": "curated", "centroid": (1, 1, 1), "area": 200},
            {"label_id": 3, "tier": "bad", "centroid": (2, 2, 2), "area": 50},
            {"label_id": 4, "tier": "garbage", "centroid": (3, 3, 3), "area": 10},
        ]
        nav = _make_fake_navigator()
        _set_items(nav, items)
        result = _select_next_triage(nav)
        assert result is True
        assert _current_label_id(nav) == 3  # bad tier

    def test_finds_garbage_when_only_triage(self):
        """When only garbage triage labels exist, should find them."""
        items = [
            {"label_id": 1, "tier": "good", "centroid": (0, 0, 0), "area": 100},
            {"label_id": 2, "tier": "garbage", "centroid": (1, 1, 1), "area": 10},
        ]
        nav = _make_fake_navigator()
        _set_items(nav, items)
        result = _select_next_triage(nav)
        assert result is True
        assert _current_label_id(nav) == 2

    def test_returns_false_when_only_good_curated(self):
        """No triage labels remain — should return False."""
        items = [
            {"label_id": 1, "tier": "good", "centroid": (0, 0, 0), "area": 100},
            {"label_id": 2, "tier": "curated", "centroid": (1, 1, 1), "area": 200},
        ]
        nav = _make_fake_navigator()
        _set_items(nav, items)
        result = _select_next_triage(nav)
        assert result is False

    def test_returns_false_on_empty_table(self):
        """Empty table — should return False."""
        nav = _make_fake_navigator()
        _set_items(nav, [])
        result = _select_next_triage(nav)
        assert result is False

    def test_fires_navigate_callback(self, sample_items):
        """Should fire the navigate callback when triage label found."""
        navigated = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: navigated.append((lid, c))}
        )
        _set_items(nav, sample_items)
        _select_next_triage(nav)
        assert len(navigated) == 1
        assert navigated[0][0] == 42  # review label

    def test_respects_tier_filter(self):
        """When review/bad are filtered out, should skip them."""
        items = [
            {"label_id": 1, "tier": "good", "centroid": (0, 0, 0), "area": 100},
            {"label_id": 2, "tier": "review", "centroid": (1, 1, 1), "area": 50},
            {"label_id": 3, "tier": "garbage", "centroid": (2, 2, 2), "area": 10},
        ]
        nav = _make_fake_navigator()
        # Filter out review tier
        nav._tier_filter_review.setChecked(False)
        _set_items(nav, items)
        result = _select_next_triage(nav)
        assert result is True
        assert _current_label_id(nav) == 3  # garbage (review is filtered out)


# ===========================================================================
# Tests — set_isolate_checked (silent checkbox sync)
# ===========================================================================


def _set_isolate_checked(nav, checked: bool):
    """Call the real set_isolate_checked logic (mirrors implementation).

    Sets checkbox state WITHOUT firing the isolation callback.
    """
    # Simulate blockSignals behavior: just set the checkbox, don't fire callback
    old_cb = nav._on_isolation_changed_cb
    nav._on_isolation_changed_cb = None
    nav._isolate_cb.setChecked(checked)
    nav._on_isolation_changed_cb = old_cb


class TestSetIsolateChecked:
    """F1: set_isolate_checked changes checkbox state without firing callbacks."""

    def test_set_checked_true_changes_state(self):
        nav = _make_fake_navigator()
        assert nav._isolate_cb.isChecked() is False
        _set_isolate_checked(nav, True)
        assert nav._isolate_cb.isChecked() is True

    def test_set_checked_false_changes_state(self):
        nav = _make_fake_navigator()
        nav._isolate_cb.setChecked(True)
        _set_isolate_checked(nav, False)
        assert nav._isolate_cb.isChecked() is False

    def test_set_checked_does_not_fire_isolation_callback(self):
        isolation_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_isolation_changed": lambda ids: isolation_calls.append(ids)}
        )
        _set_isolate_checked(nav, True)
        assert len(isolation_calls) == 0, (
            "set_isolate_checked should NOT fire isolation callback"
        )

    def test_set_checked_does_not_fire_legacy_callback(self):
        legacy_calls = []
        nav = _make_fake_navigator(
            callbacks={
                "on_show_selected_label_changed": lambda v: legacy_calls.append(v)
            }
        )
        _set_isolate_checked(nav, True)
        assert len(legacy_calls) == 0


# ===========================================================================
# Tests — clear_selection unchecks Isolate (F4)
# ===========================================================================


def _clear_selection(nav):
    """Simulate clear_selection: clear table selection + uncheck Isolate."""
    nav._table._selected_rows.clear()
    nav._table._selected_row = -1
    # F4: Also uncheck Isolate silently
    _set_isolate_checked(nav, False)
    # Fire isolation callback with empty set (clearing)
    if nav._on_isolation_changed_cb is not None:
        try:
            nav._on_isolation_changed_cb(set())
        except Exception:
            pass


class TestClearSelectionUnchecksIsolate:
    """F4: clear_selection should uncheck the Isolate checkbox."""

    def test_clear_unchecks_isolate(self):
        nav = _make_fake_navigator()
        nav._isolate_cb.setChecked(True)
        _clear_selection(nav)
        assert nav._isolate_cb.isChecked() is False

    def test_clear_fires_empty_isolation(self):
        isolation_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_isolation_changed": lambda ids: isolation_calls.append(ids)}
        )
        nav._isolate_cb.setChecked(True)
        _clear_selection(nav)
        assert set() in isolation_calls


# ===========================================================================
# Tests — select_labels multi-select (B4) and single _fire_isolation (B5)
# ===========================================================================


class TestSelectLabelsMultiSelect:
    """B4: select_labels should select ALL matching rows, not just one.
    B5: _fire_isolation_if_active should fire exactly once, not twice."""

    def _make_items(self):
        return [
            {"label_id": 1, "tier": "good", "centroid": (10, 10, 10), "area": 100},
            {"label_id": 2, "tier": "good", "centroid": (20, 20, 20), "area": 200},
            {"label_id": 3, "tier": "review", "centroid": (30, 30, 30), "area": 300},
            {"label_id": 4, "tier": "bad", "centroid": (40, 40, 40), "area": 400},
            {"label_id": 5, "tier": "good", "centroid": (50, 50, 50), "area": 500},
        ]

    def test_multi_select_all_rows_present(self):
        """select_labels([1,3,5]) should have rows for 1, 3, and 5 selected."""
        nav = _make_fake_navigator()
        _set_items(nav, self._make_items())
        _select_labels(nav, [1, 3, 5])

        selected_ids = _selected_label_ids(nav)
        assert selected_ids == {1, 3, 5}

    def test_multi_select_preserves_all_after_completion(self):
        """After select_labels completes, selectedRows should match all requested labels."""
        nav = _make_fake_navigator()
        _set_items(nav, self._make_items())
        _select_labels(nav, [2, 4])

        selected_ids = _selected_label_ids(nav)
        assert selected_ids == {2, 4}

    def test_multi_select_first_row_is_current(self):
        """The first matching row (in table order) should be the current row."""
        nav = _make_fake_navigator()
        _set_items(nav, self._make_items())
        # Default sort is Tier: good(1,2,5) → review(3) → bad(4)
        # So selecting [3, 5]: label 5 at row 2 comes first, label 3 at row 3
        _select_labels(nav, [3, 5])

        current = nav._table.selectedRow()
        row = nav._table.row(current)
        assert row is not None
        # Label 5 is the first match in tier-sorted order
        assert row.label_id == 5

    def test_multi_select_fires_navigate_for_first(self):
        """select_labels should fire navigate callback for the first row."""
        nav_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: nav_calls.append((lid, c))}
        )
        _set_items(nav, self._make_items())
        _select_labels(nav, [2, 4])

        # Should navigate to first selected (label 2)
        assert len(nav_calls) >= 1
        assert nav_calls[0][0] == 2

    def test_multi_select_fires_select_label_for_first(self):
        """select_labels should fire select_label callback for the first row."""
        sel_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_select_label": lambda lid: sel_calls.append(lid)}
        )
        _set_items(nav, self._make_items())
        _select_labels(nav, [2, 4])

        assert len(sel_calls) >= 1
        assert sel_calls[0] == 2

    def test_isolation_fires_exactly_once(self):
        """B5: _fire_isolation_if_active should be called exactly once, not twice."""
        isolation_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_isolation_changed": lambda ids: isolation_calls.append(ids)}
        )
        _set_items(nav, self._make_items())
        nav._isolate_cb.setChecked(True)

        _select_labels(nav, [1, 3, 5])

        # Should fire exactly once with all 3 IDs
        assert len(isolation_calls) == 1
        assert isolation_calls[0] == {1, 3, 5}

    def test_isolation_fires_with_all_selected_ids(self):
        """When isolation is active, the callback should receive ALL selected label IDs."""
        isolation_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_isolation_changed": lambda ids: isolation_calls.append(ids)}
        )
        _set_items(nav, self._make_items())
        nav._isolate_cb.setChecked(True)

        _select_labels(nav, [2, 3])

        # The isolation callback should get {2, 3}
        assert any(ids == {2, 3} for ids in isolation_calls)

    def test_single_label_still_works(self):
        """select_labels with a single label should work as before."""
        nav_calls = []
        nav = _make_fake_navigator(
            callbacks={"on_navigate": lambda lid, c: nav_calls.append(lid)}
        )
        _set_items(nav, self._make_items())
        _select_labels(nav, [3])

        assert 3 in nav_calls
        selected_ids = _selected_label_ids(nav)
        assert selected_ids == {3}


# ===========================================================================
# 5-button tier system tests (post-2026-05-10 redesign)
# ===========================================================================


class TestTierButtonSystem:
    """Verify the 5 tier buttons (Curated/Good/Review/Bad/Garbage) each
    fire on_mark_tier(label_id, tier_str) flag-only.

    Pre-redesign the navigator had a single "Mark Good" + a destructive
    "Delete Label" button.  Post-redesign all 5 tiers have a dedicated
    flag-only button — destruction is centralized in Discard All Garbage.
    """

    def _items(self):
        return [
            {"label_id": 7, "tier": "review", "z": 1, "area": 100, "score": 0.5,
             "centroid": [1.0, 1.0, 1.0]},
            {"label_id": 8, "tier": "review", "z": 2, "area": 200, "score": 0.6,
             "centroid": [2.0, 2.0, 2.0]},
        ]

    def _make_nav_with_tier_cb(self, calls):
        return _make_fake_navigator(
            callbacks={
                "on_mark_tier": lambda lid, tier: calls.append((lid, tier))
            }
        )

    def test_mark_curated_fires_callback(self):
        calls = []
        nav = self._make_nav_with_tier_cb(calls)
        _set_items(nav, self._items())
        nav._table.selectRow(0)  # selects label 7
        _on_mark_tier_clicked(nav, "curated")
        assert calls == [(7, "curated")]

    def test_mark_good_fires_callback(self):
        calls = []
        nav = self._make_nav_with_tier_cb(calls)
        _set_items(nav, self._items())
        nav._table.selectRow(0)
        _on_mark_tier_clicked(nav, "good")
        assert calls == [(7, "good")]

    def test_mark_review_fires_callback(self):
        calls = []
        nav = self._make_nav_with_tier_cb(calls)
        _set_items(nav, self._items())
        nav._table.selectRow(0)
        _on_mark_tier_clicked(nav, "review")
        assert calls == [(7, "review")]

    def test_mark_bad_fires_callback(self):
        calls = []
        nav = self._make_nav_with_tier_cb(calls)
        _set_items(nav, self._items())
        nav._table.selectRow(0)
        _on_mark_tier_clicked(nav, "bad")
        assert calls == [(7, "bad")]

    def test_mark_garbage_fires_callback_flag_only(self):
        """Garbage button must be flag-only — no implicit volume mutation.

        Regression guard: pre-2026-05-10 the Delete Label button
        zeroed the label in the volume AND set tier=garbage in one
        step.  After the redesign, the Garbage button only fires the
        flag-callback; actual zeroing happens via Discard All Garbage.
        """
        calls = []
        nav = self._make_nav_with_tier_cb(calls)
        _set_items(nav, self._items())
        nav._table.selectRow(0)
        _on_mark_tier_clicked(nav, "garbage")
        # Callback fired with the correct tier — only thing tier
        # button does.  No second callback fired (no volume path).
        assert calls == [(7, "garbage")]

    def test_unknown_tier_does_not_crash(self):
        """Belt-and-suspenders: if a bogus tier string reaches the
        click handler the simulator still forwards it (validation lives
        in the controller's _on_annotator_mark_tier which has a
        whitelist)."""
        calls = []
        nav = self._make_nav_with_tier_cb(calls)
        _set_items(nav, self._items())
        nav._table.selectRow(0)
        _on_mark_tier_clicked(nav, "not_a_real_tier")
        # The navigator's job is to forward; semantics happen
        # downstream.
        assert calls == [(7, "not_a_real_tier")]

    def test_no_label_selected_no_callback(self):
        calls = []
        nav = self._make_nav_with_tier_cb(calls)
        _set_items(nav, self._items())
        # No row selected (table.currentRow returns -1 in fake).
        nav._table._current_row = -1
        _on_mark_tier_clicked(nav, "good")
        assert calls == []

    def test_no_callback_set_does_not_crash(self):
        nav = _make_fake_navigator()  # no callbacks dict
        _set_items(nav, self._items())
        nav._table.selectRow(0)
        # Should silently no-op when _on_mark_tier_cb is None.
        _on_mark_tier_clicked(nav, "good")  # no assertion needed

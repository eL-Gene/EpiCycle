"""Tests for the Label Navigator sidebar-right layout (2026-06-01,
feature/navigator-layout-rethink).

Verifies the layout-only refactor that moves the tier triage + create
+ navigation buttons OUT of horizontal rows below the table and INTO
a fixed-width vertical sidebar on the RIGHT of the table.  Top rows
(search / filter+sort) are preserved verbatim; display options at the
bottom are kept but wrapped from 1 row of 6 widgets to 2 rows of 3.

All buttons remain self.* attributes by identity so existing signal
wiring, callbacks, tooltips, and styles are untouched.

Tests are source-inspect — no Qt instantiation required.
"""
from __future__ import annotations

import inspect

from napari_epicycle.annotator.curation_label_navigator import (
    CurationLabelNavigator,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def test_sidebar_widget_exists_with_fixed_width():
    """The sidebar must be a QWidget with a fixed width — not stretchy
    — so the table's column space stays predictable."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "self._sidebar_widget = QWidget()" in src, (
        "Sidebar must be self._sidebar_widget (QWidget) so subclasses + "
        "tests can find it."
    )
    assert "self._sidebar_widget.setFixedWidth(150)" in src, (
        "Sidebar should be fixed at ~150px wide so the table fills the "
        "rest of the dock."
    )


def test_sidebar_has_vertical_layout():
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "sidebar_layout = QVBoxLayout(self._sidebar_widget)" in src, (
        "Sidebar's internal layout must be a QVBoxLayout."
    )


def test_sidebar_contains_all_action_buttons():
    """All 9 action buttons must be added to the sidebar layout."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    for attr in (
        "_mark_good_btn", "_mark_review_btn", "_mark_bad_btn",
        "_mark_garbage_btn", "_mark_curated_btn",
        "_new_label_btn", "_merge_btn",
    ):
        assert f"sidebar_layout.addWidget(self.{attr})" in src, (
            f"sidebar_layout must add self.{attr}"
        )
    # Prev / Next are nested in a horizontal sub-row inside the sidebar.
    assert "prev_next_row.addWidget(self._prev_btn" in src
    assert "prev_next_row.addWidget(self._next_btn" in src


def test_sidebar_has_three_separators_between_groups():
    """Three separator lines split the four sidebar groups:
    (Triage) — (Curated) — (Create) — (Navigate)."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    # Helper-pattern: _sidebar_sep() factory called from sidebar_layout.
    assert src.count("sidebar_layout.addWidget(_sidebar_sep())") == 3, (
        "Sidebar must have exactly 3 separator lines (between routine "
        "triage / Curated / + New Label / navigation groups)."
    )


def test_sidebar_separator_uses_shared_theme_constant():
    """Separators reuse theme.SEPARATOR_LINE_STYLE rather than a
    module-local constant — consistency with other accordions."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "SEPARATOR_LINE_STYLE" in src
    # No module-local separator style.
    assert "_NAV_SEPARATOR_STYLE" not in src


def test_sidebar_pinned_to_top_with_addstretch():
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "sidebar_layout.addStretch(1)" in src, (
        "Sidebar must end with addStretch(1) so buttons pin to the top "
        "and empty space falls below."
    )


# ---------------------------------------------------------------------------
# Middle split: table + sidebar
# ---------------------------------------------------------------------------


def test_middle_split_uses_qhboxlayout():
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "middle_split = QHBoxLayout()" in src
    assert "middle_split.addWidget(self._table, 1)" in src, (
        "Table goes on the LEFT with stretch=1 (fills available width)."
    )
    assert "middle_split.addWidget(self._sidebar_widget, 0)" in src, (
        "Sidebar goes on the RIGHT with stretch=0 (fixed width)."
    )


def test_no_table_addwidget_directly_on_main_layout():
    """The table must be inside middle_split, NOT added directly to
    the main layout (which would put it above the sidebar instead of
    beside it)."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    # The only line that adds the table should be the middle_split one.
    assert "layout.addWidget(self._table" not in src, (
        "self._table must NOT be added directly to the main layout — "
        "it must go through middle_split so the sidebar sits beside "
        "it."
    )


# ---------------------------------------------------------------------------
# Top rows untouched
# ---------------------------------------------------------------------------


def test_search_row_untouched():
    """Search row (Find ID(s)) remains in the main layout above the
    middle split."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "search_row = QHBoxLayout()" in src
    assert "search_row.addWidget(self._search_field" in src
    assert "layout.addLayout(search_row)" in src


def test_filter_row_untouched():
    """Filter row (tier filter checkboxes + sort combo) remains in the
    main layout above the middle split."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "filter_row = QHBoxLayout()" in src
    assert "self._tier_filter_good" in src
    assert "self._sort_combo" in src


# ---------------------------------------------------------------------------
# Display options: 2 rows of 3 widgets each
# ---------------------------------------------------------------------------


def test_display_options_wrapped_to_two_rows():
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "_display_row_1 = QHBoxLayout()" in src
    assert "_display_row_2 = QHBoxLayout()" in src
    # Each row holds 3 widgets.
    assert "_display_row_1.addWidget(self._focused_mode_btn)" in src
    assert "_display_row_1.addWidget(self._isolate_cb)" in src
    assert "_display_row_1.addWidget(self._paint_isolate_cb)" in src
    assert "_display_row_2.addWidget(self._render_selected_btn)" in src
    assert "_display_row_2.addWidget(self._right_click_select_cb)" in src
    assert "_display_row_2.addWidget(self._auto_advance_cb)" in src


def test_display_options_toggle_and_container_exist():
    """Collapsible toggle + container widget must still exist (was in
    place from prior rethink; sidebar move preserves it)."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "self._display_options_toggle" in src
    assert "self._display_options_container" in src
    assert "QToolButton" in src


def test_display_options_toggle_defaults_expanded():
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "setChecked(True)  # default expanded" in src


def test_display_options_toggle_chevron_text():
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "▾ Display options" in src
    assert "▸ Display options" in src


# ---------------------------------------------------------------------------
# Identity / signal preservation
# ---------------------------------------------------------------------------


def test_all_buttons_still_constructed_as_attributes():
    """The layout rethink must preserve every button as a self.*
    attribute — downstream callbacks + tests reference these by name."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    for btn in (
        "_mark_good_btn", "_mark_review_btn", "_mark_bad_btn",
        "_mark_garbage_btn", "_mark_curated_btn",
        "_new_label_btn", "_merge_btn", "_prev_btn", "_next_btn",
    ):
        assert f"self.{btn} = QPushButton" in src, (
            f"self.{btn} must still be constructed in _build_ui"
        )


def test_new_label_button_has_plus_prefix():
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert 'QPushButton("+ New Label")' in src, (
        "+ New Label should have the + prefix to distinguish create "
        "vs flag actions."
    )


def test_no_legacy_grid_or_action_row_layout():
    """Negative assertion — the prior 2x3 grid + action_row layout
    artifacts are GONE."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "QGridLayout" not in src
    assert "tier_grid" not in src
    assert "bottom_row" not in src


# ---------------------------------------------------------------------------
# Forward-compat: Delete Label slot
# ---------------------------------------------------------------------------


def test_delete_label_placement_documented():
    """A commented-out line in the sidebar shows where future Delete
    Label (Task A) will land — directly under + New Label so create/
    destroy are vertically paired."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "_delete_label_btn" in src, (
        "_build_ui must contain a commented-out placeholder for the "
        "future Delete Label button"
    )
    assert "sidebar_layout.addWidget(self._delete_label_btn)" in src, (
        "Placeholder should be the EXACT line that adds Delete to the "
        "sidebar so future agents copy-paste"
    )

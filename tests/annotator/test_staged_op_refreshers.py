"""The buttons that follow an open Add, the load lock, and the navigator's
highlight snap-back: each does its job, checked on the real methods."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QAbstractItemView, QTableWidget, QTableWidgetItem

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K
from napari_epicycle.annotator.curation_label_navigator import (
    _COL_ID,
    CurationLabelNavigator as N,
)

from .add_widget_fixture import qapp  # noqa: F401 - the fixture
from .test_add_click_markers import h  # noqa: F401 - the fixture


# --- the Add exits grey and un-grey New / Delete / Merge -----------------

def test_reset_add_state_calls_both_refreshers(h):  # noqa: F811
    calls = []
    h._refresh_label_action_buttons_state = lambda: calls.append("labels")
    h._refresh_merge_button_state = lambda: calls.append("merge")
    h._add_running = True
    h._reset_add_state()
    assert h._add_running is False
    assert sorted(calls) == ["labels", "merge"]


def test_add_accept_finally_calls_the_refreshers_after_the_lock_release():
    src = inspect.getsource(K._accept_add)
    lock = src.index("self._fire_add_mode_lock(False)")
    assert src.index("self._add_refresh_label_buttons()", lock) > lock
    assert "self._add_running = False" in src[:lock]


def test_add_start_refreshes_the_buttons(h):  # noqa: F811
    helper = inspect.getsource(K._add_refresh_label_buttons)
    assert "_refresh_label_action_buttons_state" in helper
    assert "_refresh_merge_button_state" in helper


# --- the load lock asks the shared gate instead of enabling blindly ------

def test_unlock_refreshes_instead_of_enabling_new_delete_merge():
    log = []
    nav = SimpleNamespace(
        set_label_action_buttons_enabled=lambda v: log.append(("nav", v)),
        set_merge_enabled=lambda v: log.append(("navmerge", v)),
        set_mark_buttons_enabled=lambda v: None)
    f = SimpleNamespace(
        _navigator_widget=nav, _load_locked=True,
        _is_op_active_for_navigator=lambda: False,
        _restore_ready_to_draw_state=lambda: None,
        _lock_subviewers_for_preview=lambda: None,
        _refresh_label_action_buttons_state=lambda: log.append("labels"),
        _refresh_merge_button_state=lambda: log.append("merge"))
    K.set_load_locked(f, False)
    assert "labels" in log and "merge" in log
    assert ("nav", True) not in log and ("navmerge", True) not in log


# --- the real snap-back ---------------------------------------------------

def _table(qapp, ids):
    t = QTableWidget(len(ids), _COL_ID + 1)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.ExtendedSelection)
    for r, lid in enumerate(ids):
        it = QTableWidgetItem(str(lid))
        it.setData(Qt.UserRole, {"label_id": lid})
        t.setItem(r, _COL_ID, it)
    return t


def _nav(t):
    n = SimpleNamespace(_table=t, fired=0)
    n._on_table_selection_changed = lambda: setattr(n, "fired", n.fired + 1)
    return n


def test_snap_back_selects_only_the_anchor_row_and_fires_once(qapp):
    t = _table(qapp, [4, 5, 6])
    t.selectRow(1)
    t.selectionModel().select(t.model().index(2, 0),
                              t.selectionModel().Select | t.selectionModel().Rows)
    n = _nav(t)
    N._snap_back_selection(n, 4)
    rows = sorted({i.row() for i in t.selectionModel().selectedRows()})
    assert rows == [0]
    assert n.fired == 1


def test_snap_back_with_no_matching_row_clears_the_highlight(qapp):
    t = _table(qapp, [4, 5])
    t.selectRow(1)
    n = _nav(t)
    N._snap_back_selection(n, 0)
    assert t.selectionModel().selectedRows() == []
    assert n.fired == 1

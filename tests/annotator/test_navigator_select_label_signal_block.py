"""Regression tests for ``CurationLabelNavigator.select_label`` (2026-09-07).

The 2026-09-07 access violation's innermost Python frame was the
``selectionModel().selectionChanged`` lambda, re-entered from INSIDE
``self._table.selectRow(row_idx)`` in ``select_label``.  ``select_labels``
(plural) and ``_rebuild_table`` already blocked the selection MODEL's own
signals around their mutations (the 2026-06-09 / B5 fixes); the singular
path did not.

These tests bind the REAL method to a Qt-free fake — no QApplication, safe
in the annotator batch.
"""
from __future__ import annotations

import pytest

from qtpy.QtCore import Qt

from napari_epicycle.annotator.curation_label_navigator import (
    CurationLabelNavigator,
    _COL_ID,
)

_select_label = CurationLabelNavigator.select_label


class _FakeItem:
    def __init__(self, label_id):
        self._data = {"label_id": label_id}

    def data(self, role):
        assert role == Qt.UserRole
        return self._data


class _FakeSelModel:
    def __init__(self, log):
        self._log = log
        self.blocked = False

    def blockSignals(self, v):
        self.blocked = bool(v)
        self._log.append(("blockSignals", bool(v)))


class _FakeTable:
    def __init__(self, label_ids, log, sel_model=True):
        self._items = [_FakeItem(l) for l in label_ids]
        self._log = log
        self._sel = _FakeSelModel(log) if sel_model else None

    def rowCount(self):
        return len(self._items)

    def item(self, row, col):
        assert col == _COL_ID
        return self._items[row]

    def selectionModel(self):
        return self._sel

    def selectRow(self, row):
        blocked = self._sel.blocked if self._sel is not None else None
        self._log.append(("selectRow", row, blocked))

    def scrollToItem(self, item):
        self._log.append(("scrollToItem",))


class _FakeNav:
    def __init__(self, label_ids, sel_model=True):
        self.log: list = []
        self._table = _FakeTable(label_ids, self.log, sel_model=sel_model)
        self.handler_calls = 0
        self.row_clicks: list = []

    def _on_table_selection_changed(self):
        self.handler_calls += 1
        self.log.append(("handler",))

    def _on_row_clicked(self, row_idx):
        self.row_clicks.append(row_idx)
        self.log.append(("row_clicked", row_idx))


def _run(nav, label_id):
    _select_label(nav, label_id)


class TestSelectLabelSignalBlock:
    def test_selection_model_signals_blocked_during_selectRow(self):
        nav = _FakeNav([10, 20, 30])
        _run(nav, 20)
        select = next(e for e in nav.log if e[0] == "selectRow")
        assert select == ("selectRow", 1, True), (
            "selectRow must run with the selection model's signals blocked"
        )

    def test_signals_unblocked_afterwards(self):
        nav = _FakeNav([10, 20, 30])
        _run(nav, 20)
        assert nav._table.selectionModel().blocked is False

    def test_handler_fired_exactly_once_after_the_mutation(self):
        nav = _FakeNav([10, 20, 30])
        _run(nav, 30)
        assert nav.handler_calls == 1
        order = [e[0] for e in nav.log]
        assert order.index("selectRow") < order.index("handler")

    def test_row_clicked_still_dispatched(self):
        """Navigate / isolation must still run — the deferral of the whole
        select is the caller's job, not this method's."""
        nav = _FakeNav([10, 20, 30])
        _run(nav, 10)
        assert nav.row_clicks == [0]

    def test_unknown_label_is_a_noop(self):
        nav = _FakeNav([10, 20, 30])
        _run(nav, 99)
        assert nav.log == []
        assert nav.handler_calls == 0
        assert nav.row_clicks == []

    def test_missing_selection_model_does_not_raise(self):
        nav = _FakeNav([10, 20, 30], sel_model=False)
        _run(nav, 20)
        assert nav.row_clicks == [1]
        # no selection model -> no explicit handler re-fire
        assert nav.handler_calls == 0

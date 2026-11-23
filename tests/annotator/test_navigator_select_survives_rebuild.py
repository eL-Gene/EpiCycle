"""Rows an isolation selects must survive the navigator's deferred rebuild.

The commit callback rebuilds the table one event-loop turn later; a selection
made before that turn was applied to the old rows and dropped.  While a
rebuild is pending the ids are kept and the rebuild applies them.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _nav(qapp, n=8):
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator)

    nav = CurationLabelNavigator()
    nav.set_items([
        {"label_id": i, "tier": "good", "centroid": (i, i, i), "area": 10 * i}
        for i in range(1, n + 1)
    ])
    for _ in range(5):
        qapp.processEvents()
    return nav


def test_selection_made_while_a_rebuild_is_pending_survives_it(qapp):
    nav = _nav(qapp)
    fired = []
    nav._on_table_selection_changed = lambda: fired.append(1)
    nav.set_items([
        {"label_id": i, "tier": "good", "centroid": (i, i, i), "area": 10 * i}
        for i in range(1, 9)
    ])
    assert nav._rebuild_pending
    nav.select_rows_silently({2, 5, 7})
    assert nav._pending_select_ids == {2, 5, 7}
    assert fired == []                       # nothing applied to the old rows
    for _ in range(5):
        qapp.processEvents()
    assert nav.selected_label_ids() == {2, 5, 7}
    assert fired                             # the handler ran after the rebuild
    assert nav._pending_select_ids is None


def test_selection_without_a_pending_rebuild_is_immediate(qapp):
    nav = _nav(qapp)
    nav.select_rows_silently({3, 4})
    assert nav.selected_label_ids() == {3, 4}


def test_a_plain_rebuild_keeps_a_multi_selection(qapp):
    nav = _nav(qapp)
    nav.select_rows_silently({1, 2, 6})
    nav._rebuild_table()
    assert nav.selected_label_ids() == {1, 2, 6}


def test_the_pending_ids_replace_the_old_selection(qapp):
    nav = _nav(qapp)
    nav.select_rows_silently({1})
    nav._rebuild_pending = True
    nav.select_rows_silently({4, 5})
    nav._rebuild_pending = False
    nav._rebuild_table()
    assert nav.selected_label_ids() == {4, 5}


def test_clear_draw_snapshot_survives_a_missing_button():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator as K)

    f = SimpleNamespace(
        _clear_edits_btn=None,
        _refresh_merge_button_state=lambda: None,
        _refresh_label_action_buttons_state=lambda: None)
    K._clear_draw_snapshot(f)
    assert f._draw_snapshot is None


def test_a_row_that_does_not_exist_yet_is_selected_by_the_next_rebuild(qapp):
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator)

    nav = CurationLabelNavigator()
    nav.set_items([{"label_id": i, "tier": "good", "centroid": (i, i, i),
                    "area": 10} for i in (108, 690, 691)])
    for _ in range(5):
        qapp.processEvents()
    nav.select_rows_silently({250, 108})
    assert nav.selected_label_ids() == {108}            # what exists now
    nav.set_items([{"label_id": i, "tier": "good", "centroid": (i, i, i),
                    "area": 10} for i in (108, 250)])
    for _ in range(5):
        qapp.processEvents()
    assert nav.selected_label_ids() == {108, 250}

"""G, B, Ctrl+Z and Esc follow the buttons.

The keys are QShortcuts on the Workbench dock only
(``Qt.WidgetWithChildrenShortcut``), and every key's slot checks its
button's enable state: while a button is grey its key does nothing.
"""
from __future__ import annotations

import pytest
from qtpy.QtCore import Qt

from napari_epicycle.segmentation import _workbench_text as T

from ._workbench_ctrl import pump, select, table, wired


@pytest.fixture
def cw(tmp_path):
    c, wb = wired(tmp_path, table(20, tiers=[""] * 20))
    yield c, wb
    wb.deleteLater()
    pump()


def _shortcut(wb, seq):
    for sc in wb._shortcuts:
        if sc.key().toString() == seq:
            return sc
    raise AssertionError(seq)


def _tier(c, label):
    return c._qc_features_df.set_index("label").loc[label, "qc_tier"]


def test_four_keys_scoped_to_the_dock(cw):
    _, wb = cw
    keys = sorted(sc.key().toString() for sc in wb._shortcuts)
    assert keys == ["B", "Ctrl+Z", "Esc", "G"]
    for sc in wb._shortcuts:
        assert sc.context() == Qt.WidgetWithChildrenShortcut
        assert sc.parent() is wb


def test_g_and_b_do_nothing_while_their_buttons_are_grey(cw):
    c, wb = cw
    assert not wb.mark_good_btn.isEnabled()          # nothing selected
    _shortcut(wb, "G").activated.emit()
    _shortcut(wb, "B").activated.emit()
    assert c._mark_history == []
    assert wb.sel_msg_label.text() == T.MARK_NEEDS_SEL   # X1-4: the reason only


def test_g_marks_good_and_ctrl_z_undoes(cw):
    c, wb = cw
    select(c, [5])
    _shortcut(wb, "G").activated.emit()
    assert _tier(c, 5) == "good"
    _shortcut(wb, "B").activated.emit()
    assert _tier(c, 5) == "bad"
    _shortcut(wb, "Ctrl+Z").activated.emit()
    assert _tier(c, 5) == "good"


def test_ctrl_z_does_nothing_with_an_empty_history(cw):
    c, wb = cw
    select(c, [5])
    assert not wb.undo_mark_btn.isEnabled()
    _shortcut(wb, "Ctrl+Z").activated.emit()
    assert wb.sel_msg_label.text() == T.UNDO_EMPTY       # X1-4: the reason only


def test_keys_do_nothing_under_the_lock(cw):
    c, wb = cw
    select(c, [5])
    wb.set_workbench_lock("Save")
    _shortcut(wb, "G").activated.emit()
    assert _tier(c, 5) == ""
    wb.set_workbench_lock(None)
    _shortcut(wb, "G").activated.emit()
    assert _tier(c, 5) == "good"


def test_keys_stay_live_while_a_map_runs(cw):
    c, wb = cw
    select(c, [6])
    wb.set_map_running("UMAP")
    assert not wb.compute_umap_btn.isEnabled()
    _shortcut(wb, "B").activated.emit()
    assert _tier(c, 6) == "bad"                     # I-17: marks stay live


def test_esc_clears_the_selection_only_when_clear_is_live(cw):
    c, wb = cw
    pw = wb.qc_plot_widget
    pw._color_combo.setCurrentText("QC tier")
    pw._toggle_category_selection("qc_tier", "unmarked")
    pump()
    assert c._selected_label_ids
    wb.show()
    pump()
    assert wb.clear_selection_btn.isEnabled()
    _shortcut(wb, "Esc").activated.emit()
    pump()
    assert c._selected_label_ids == []
    wb.hide()

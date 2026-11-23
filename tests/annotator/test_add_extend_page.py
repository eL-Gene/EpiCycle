"""Extending an object, on the page: rows per object, the current one.

The marker list shows one row per OBJECT -- its number, its seed count
when it grew from several markers, and a status word for the seeds that
made nothing at the last result (not connected, nothing bright here, inside
a kept object).  The result list carries the same seed count and status.
Alt+click on an object's drawn voxels (or its ball), or a click on its row,
makes it the current object; the current object's row is bold and S1 names
it while Place markers is on.  The ready line and the Markers title count
objects, not markers.  A refused (not connected) seed is drawn hollow, in
its own red, in the main view and the side copies.  The hint line replaces
the click help, and the Place markers tooltip names the keys.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_nuclei_box import (  # noqa: F401
    NUC_A,
    NUC_B,
    _active,
    _finish,
    _fire,
    lv,
    off,
)

#: a second spot on nucleus A, outside the first click's ball
A2 = (12, 22, 22)
MARKERS = (AP.LOOK_MARKERS, None)
GUIDE = (pathlib.Path(__file__).resolve().parents[2] / "docs" / "guides"
         / "annotator_3d_add.md")


def _ids(hh):
    return [int(m["id"]) for m in hh._add_click_markers]


def _preview(hh):
    hh._preview_add()
    _finish(hh)
    assert hh._add_preview_result is not None, hh._add_transient


# --------------------------------------------------------------------------- #
# the words
# --------------------------------------------------------------------------- #
def test_the_new_words_word_for_word():
    # the hint names every gesture, Ctrl+Z among them; Esc stays in the
    # tooltip and the guide
    assert AP.ADD_HELP_CLICK_KEYS == (
        "Click: a new object · Shift+click: add to the bold one · "
        "Alt+click it or its row: make it bold · Ctrl+click: remove · "
        "Ctrl+Z: undo last")
    assert AP.ADD_ROW_SEEDS == "{n} seeds"
    # short status words that fit a row beside edge / small ...
    assert AP.ADD_ROW_NOT_CONNECTED == "apart"
    assert AP.ADD_ROW_DIM == "dim"
    assert AP.ADD_ROW_KEPT == "on kept"
    # ... the long form in the row's tooltip
    assert AP.ADD_ROW_STATUS_LONG == {
        "apart": "not connected", "dim": "nothing bright here",
        "on kept": "inside a kept object"}
    assert AP.status_tip(["apart", "dim"]) == (
        "Seeds that made nothing: apart (not connected), dim (nothing "
        "bright here).")
    assert AP.status_tip([]) == ""
    # the S1 current-object tail is gone: the hint names Shift+click and
    # the bold row names the current object
    assert not hasattr(AP, "ADD_S1_CURRENT_TAIL")
    assert not hasattr(AP, "current_tail")


def test_s1_counts_objects_marked():
    assert AP.ADD_S1_BOX_CLICKS == (
        "Box {z} × {y} × {x} voxels · {n} objects marked")
    assert AP.ADD_S1_BOX_CLICKS_ONE == (
        "Box {z} × {y} × {x} voxels · 1 object marked")
    assert AP.ADD_S1_PATCH_CLICKS == (
        "Whole patch {z} × {y} × {x} voxels · {n} objects marked")
    assert AP.ADD_S1_PATCH_CLICKS_ONE == (
        "Whole patch {z} × {y} × {x} voxels · 1 object marked")
    assert AP.ADD_S1_TICK_ONE_MARKER == (
        "Tick at least one object to grow from — all {n} are unticked.")
    assert AP.ADD_S1_TICK_ONE_MARKER_ONE == (
        "Tick the object to grow from — it is unticked.")
    assert AP.ADD_S1_TICK_ONE_NUCLEUS == (
        "Tick at least one object to find its nucleus — all {n} are "
        "unticked.")
    assert AP.ADD_S1_TICK_ONE_NUCLEUS_ONE == (
        "Tick the object to find its nucleus — it is unticked.")


def test_the_row_texts():
    assert AP.object_row_text(412, (12, 20, 305)) == (
        AP.marker_row_text(412, (12, 20, 305)))
    assert AP.object_row_text(412, (12, 20, 305), 3) == "Object 412  ·  3 seeds"
    assert AP.candidate_row_text(412, 1234) == "Object 412  ·  1,234"
    assert AP.candidate_row_text(412, 1234, 2) == (
        "Object 412  ·  2 seeds  ·  1,234")
    assert AP.candidate_row_text(412) == "Object 412"


def test_the_status_words_come_once_each_in_order():
    assert AP.seed_status_words([None, None]) == []
    assert AP.seed_status_words(["kept", "apart", "dim", "apart"]) == [
        AP.ADD_ROW_NOT_CONNECTED, AP.ADD_ROW_DIM, AP.ADD_ROW_KEPT]


def test_s1_never_names_the_current_object():
    """The current object is named by its bold row and the hint line, not
    by S1: the ready line and the found line read as they do with no
    object current (the found line keeps why rows started unticked)."""
    ready = dict(has_box=True, preset="nuclei", channel="n",
                 box_shape=(40, 125, 125), look_for=AP.LOOK_MARKERS,
                 n_clicks=7, n_ticked=7, place_markers_on=True)
    assert AP.s1_sentence(dict(ready, current_id=412)) == AP.box_line(
        (40, 125, 125), preset="nuclei", n_clicks=7, n_ticked=7,
        look_for=AP.LOOK_MARKERS)
    some = dict(ready, n_ticked=5, current_id=412)
    assert AP.s1_sentence(some) == AP.box_line(
        (40, 125, 125), preset="nuclei", n_clicks=7, n_ticked=5,
        look_for=AP.LOOK_MARKERS)
    cand = dict(candidates=(12, 2, 2, 1, 9), place_markers_on=True)
    assert AP.s1_sentence(dict(cand, current_id=412)) == AP.found_line(
        12, k_edge=2, k_small=2, k_both=1)
    cand6 = dict(candidates=(12, 0, 0, 0, 10, 2), place_markers_on=True)
    assert AP.s1_sentence(dict(cand6, current_id=412)) == AP.found_line(
        12, k_plain=2)


def test_every_new_sentence_is_in_the_guide():
    """Each sentence the page shows is in the guide's words (the guide's
    quotes are held to the constants by the guide-parity test)."""
    guide = " ".join(GUIDE.read_text(encoding="utf-8").split())
    assert "## Clicking" in GUIDE.read_text(encoding="utf-8")
    for text in ((AP.ADD_HELP_CLICK_KEYS,
                  AP.ADD_ROW_SEEDS.format(n=3),
                  AP.ADD_ROW_NOT_CONNECTED, AP.ADD_ROW_DIM, AP.ADD_ROW_KEPT,
                  AP.ADD_S1_NOT_CONNECTED.format(id=12))
                 + tuple(AP.ADD_ROW_STATUS_LONG.values())):
        assert text in guide, text


# --------------------------------------------------------------------------- #
# the orchestrator: counts per object, the current object, statuses
# --------------------------------------------------------------------------- #
def test_the_push_counts_objects_and_names_the_current_one(off):
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(A2, shift=True)
    assert off._add_marker_click(NUC_B)
    a, _a, b = _ids(off)
    # the markers travel one per seed; the count is per object
    assert len(off.adv.last("click_markers")) == 3
    assert off.adv.last("click_count") == 2
    assert off.adv.last("click_current") == b
    st = off._add_s1_state()
    assert st["n_clicks"] == 2 and st["n_ticked"] == 2
    off._on_add_click_tick(a, False)
    assert off._add_s1_state()["n_ticked"] == 1
    assert off.adv.last("click_count") == 2


def test_the_ready_line_counts_objects_and_names_the_current_one(off):
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(A2, shift=True)
    assert off._add_marker_click(NUC_B)
    b = _ids(off)[-1]
    text = off.adv.last("status_text")
    assert text.endswith("2 objects marked"), text
    assert off.adv.last("click_current") == b
    # Esc: no object current, the same line
    assert off._add_esc_current()
    assert off.adv.last("click_current") == 0
    assert off.adv.last("status_text") == text


def test_the_found_line_names_no_current_object(off):
    assert off._add_marker_click(NUC_A)
    _preview(off)
    text = off.adv.last("status_text")
    assert text.startswith(AP.ADD_S1_FOUND_ONE), text
    assert "Shift+click" not in text and "#" not in text, text


def test_a_pick_clears_the_not_connected_sentence(off):
    """After "That spot is not touching #A", picking another object (its
    row, or Alt+click) takes the sentence off: it was about A."""
    assert off._add_marker_click(NUC_A)
    _preview(off)
    a = _ids(off)[0]
    assert off._add_marker_click(NUC_B, shift=True)
    _preview(off)
    assert off._add_preview_res.apart_ids == [a]
    said = AP.ADD_S1_NOT_CONNECTED.format(id=a)
    assert off.adv.last("status_text") == said
    # a row pick
    assert off._add_pick_object(a)
    assert off._add_transient is None
    assert off.adv.last("status_text") != said
    # ... and an Alt+click
    off._add_transient = said
    assert off._add_marker_click(NUC_A, alt=True)
    assert off._add_transient is None
    assert off.adv.last("status_text") != said


def test_a_pick_keeps_any_other_sentence(off):
    assert off._add_marker_click(NUC_A)
    other = AP.ADD_S1_CLICK_OUTSIDE
    off._add_transient = other
    assert off._add_pick_object(_ids(off)[0])
    assert off._add_transient == other


def test_alt_click_on_a_drawn_object_makes_it_current(off):
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(NUC_B)
    _preview(off)
    a, b = _ids(off)
    assert off._add_current_rec()["id"] == b
    e0 = off._add_markers_epoch
    n0 = len(FakeThread.instances)
    # A2 lies on nucleus A's drawn voxels, outside its ball
    assert not off._add_in_ball(A2, off._add_click_markers[0])
    assert off._add_marker_click(A2, alt=True)
    assert off._add_current_rec()["id"] == a
    assert _ids(off) == [a, b]                       # nothing placed
    assert off._add_markers_epoch == e0              # no engine input
    assert len(FakeThread.instances) == n0
    assert off._add_preview_result is not None       # nothing dropped
    assert off.adv.last("click_current") == a
    # the next Shift+click extends A
    assert off._add_marker_click((12, 18, 18), shift=True)
    assert _ids(off) == [a, b, a]


def test_alt_click_on_a_ball_makes_it_current(off):
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(NUC_B)
    a = _ids(off)[0]
    assert off._add_marker_click(NUC_A, alt=True)
    assert off._add_current_rec()["id"] == a
    assert len(off._add_click_markers) == 2


def test_alt_click_on_nothing_changes_nothing(off):
    assert off._add_marker_click(NUC_A)
    cur = off._add_current_rec()
    assert not off._add_marker_click((14, 24, 8), alt=True)
    assert off._add_current_rec() is cur


def test_a_row_click_makes_the_object_current(lv):
    assert lv._add_marker_click(NUC_A)
    assert lv._add_marker_click(A2, shift=True)
    assert lv._add_marker_click(NUC_B)
    _fire(lv)
    _finish(lv)
    a = _ids(lv)[0]
    e0 = lv._add_markers_epoch
    assert lv._add_pick_object(a)
    # the object's last seed holds the reference
    assert lv._add_current_rec() is lv._add_click_markers[1]
    assert lv._add_markers_epoch == e0
    assert not _active(lv), "picking is no marker change"
    assert lv.adv.last("click_current") == a
    assert not lv._add_pick_object(9999)
    assert lv._add_current_rec()["id"] == a


def test_a_refused_seed_is_drawn_hollow_and_red_and_named_on_its_row(off):
    assert off._add_marker_click(NUC_A)
    _preview(off)
    a = _ids(off)[0]
    assert off._add_marker_click(NUC_B, shift=True)
    _preview(off)
    assert off._add_preview_res.apart_ids == [a]
    layer = off._add_marker_layer
    # hollow: the middle is empty, the shell is the refused value
    assert layer.data[NUC_B] == 0
    z, y, x = NUC_B
    shell = layer.data[z - 3:z + 4, y - 3:y + 4, x - 3:x + 4]
    assert (shell == AP.ADD_BALL_APART_VALUE).any()
    assert set(np.unique(shell).tolist()) <= {0, AP.ADD_BALL_APART_VALUE}
    # the other seed keeps its object's colour
    assert layer.data[NUC_A] == AP.ball_value(a)
    # its own red, in the main view and the side copies
    red = np.asarray(AP.ADD_BALL_APART_RGBA, np.float32)
    targets = [layer] + [off.subviewer_manager.get_subviewer_layer(
        layer.name, o) for o in ("XY", "XZ", "YZ")]
    for t in targets:
        got = np.asarray(t.colormap.map(
            np.array([AP.ADD_BALL_APART_VALUE])))[0]
        assert np.allclose(got, red), t.name
        # the object colours are napari's own, unchanged
        from napari.utils.colormaps import label_colormap

        cyc = label_colormap(49, seed=0.5)
        ids = np.arange(1, 50)
        assert np.allclose(t.colormap.map(ids), cyc.map(ids)), t.name
    # the rows: the marker list and the result list name it
    assert off.adv.last("click_status") == {a: [AP.ADD_ROW_NOT_CONNECTED]}
    assert off.adv.last("candidate_status") == {
        a: [AP.ADD_ROW_NOT_CONNECTED]}
    assert off.adv.last("candidate_seeds") == {a: 2}
    # Undo last takes it back: no hollow ball, no status
    off._add_click_undo()
    assert not (layer.data == AP.ADD_BALL_APART_VALUE).any()
    assert off.adv.last("click_status") == {}
    _preview(off)
    assert off.adv.last("candidate_status") == {}


def test_a_dim_seed_of_an_object_is_named_on_its_row(off):
    assert off._add_marker_click(NUC_A)
    a = _ids(off)[0]
    assert off._add_marker_click((14, 24, 8), shift=True)       # off stain
    _preview(off)
    assert off.adv.last("click_status") == {a: [AP.ADD_ROW_DIM]}
    assert off.adv.last("candidate_seeds") == {a: 2}


def test_live_the_refused_seed_is_hollow_once_its_result_lands(lv):
    assert lv._add_marker_click(NUC_A)
    _fire(lv)
    _finish(lv)
    assert lv._add_marker_click(NUC_B, shift=True)
    layer = lv._add_marker_layer
    assert layer.data[NUC_B] != 0                    # a ball until told
    _fire(lv)
    _finish(lv)
    assert layer.data[NUC_B] == 0
    assert (layer.data == AP.ADD_BALL_APART_VALUE).any()


# --------------------------------------------------------------------------- #
# the panel
# --------------------------------------------------------------------------- #
def _markers_page(w):
    reset_add(w)
    w.update_add_ui_state(look_for_items=[], box_active=True)
    keys = list(w._add_look_keys)
    w.add_look_for_combo.setCurrentIndex(keys.index(MARKERS))


def _back(w):
    w.add_look_for_combo.setCurrentIndex(0)
    w.update_add_ui_state(box_active=False, whole_patch=False,
                          click_count=0, click_markers=[], click_status={},
                          click_current=0, candidate_seeds={},
                          candidate_status={}, candidate_ids=[],
                          candidate_sizes={}, candidate_edge_ids=set())
    w.add_markers_btn.setChecked(False)
    reset_add(w)


SEEDS = [(9, (1, 2, 3), True), (9, (1, 2, 8), True), (10, (4, 5, 6), True)]


def _rows(lst):
    from qtpy.QtCore import Qt

    return {lst.item(i).data(Qt.UserRole): lst.item(i)
            for i in range(lst.count())}


def test_the_marker_list_has_one_row_per_object(widget):
    _markers_page(widget)
    try:
        widget.update_add_ui_state(click_count=2, click_markers=SEEDS)
        lst = widget.add_click_list
        assert lst.count() == 2
        rows = _rows(lst)
        assert rows[9].text() == "Object 9  ·  2 seeds"
        assert rows[10].text() == AP.marker_row_text(10, (4, 5, 6))
        # the Markers title counts objects
        title = str(widget.add_markers_btn.property(
            "_epicycle_chevron_title"))
        assert title == AP.markers_summary(AP.SOURCE_CLICKS, n_clicks=2), (
            title)
    finally:
        _back(widget)


def test_an_untick_on_an_object_row_counts_one_object(widget):
    from qtpy.QtCore import Qt

    _markers_page(widget)
    got = []
    widget._on_add_click_tick_cb = lambda i, t: got.append((i, t))
    try:
        widget.update_add_ui_state(click_count=2, click_markers=SEEDS)
        _rows(widget.add_click_list)[9].setCheckState(Qt.Unchecked)
        assert got == [(9, False)]
        assert widget._add_click_ticked == 1
        assert widget.add_click_count_label.text() == AP.click_count_text(
            2, n_ticked=1)
    finally:
        widget._on_add_click_tick_cb = None
        _back(widget)


def test_a_row_carries_its_status_words_as_chips(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_CHIPS_ROLE,
    )

    _markers_page(widget)
    try:
        widget.update_add_ui_state(
            click_count=2, click_markers=SEEDS,
            click_status={9: [AP.ADD_ROW_NOT_CONNECTED]})
        rows = _rows(widget.add_click_list)
        assert rows[9].data(ADD_CHIPS_ROLE) == [AP.ADD_ROW_NOT_CONNECTED]
        assert not rows[10].data(ADD_CHIPS_ROLE)
        long = AP.status_tip([AP.ADD_ROW_NOT_CONNECTED])
        assert long in rows[9].toolTip()
        assert long not in rows[10].toolTip()
        # a status alone, without a rebuild
        widget.update_add_ui_state(click_status={})
        row = _rows(widget.add_click_list)[9]
        assert not row.data(ADD_CHIPS_ROLE)
        assert long not in row.toolTip()
    finally:
        _back(widget)


def test_the_current_object_s_row_is_bold(widget):
    _markers_page(widget)
    try:
        widget.update_add_ui_state(click_count=2, click_markers=SEEDS,
                                   click_current=9)
        rows = _rows(widget.add_click_list)
        assert rows[9].font().bold() and not rows[10].font().bold()
        widget.update_add_ui_state(click_current=10)
        rows = _rows(widget.add_click_list)
        assert rows[10].font().bold() and not rows[9].font().bold()
        widget.update_add_ui_state(click_current=0)
        assert not any(r.font().bold()
                       for r in _rows(widget.add_click_list).values())
        # a rebuild keeps it
        widget.update_add_ui_state(click_current=9)
        widget.update_add_ui_state(click_count=2, click_markers=SEEDS)
        assert _rows(widget.add_click_list)[9].font().bold()
    finally:
        _back(widget)


@pytest.mark.parametrize("attr", ["add_click_list", "add_candidate_list"])
def test_a_row_click_picks_its_object_and_a_tick_does_not(widget, attr):
    from qtpy.QtCore import Qt

    _markers_page(widget)
    got = []
    widget.set_add_callbacks(on_click_pick=got.append)
    try:
        widget.update_add_ui_state(click_count=2, click_markers=SEEDS)
        widget.update_add_ui_state(candidate_ids=[9, 10],
                                   candidate_sizes={9: 50, 10: 60},
                                   candidate_edge_ids=set())
        lst = getattr(widget, attr)
        item = _rows(lst)[9]
        lst.itemPressed.emit(item)
        lst.itemClicked.emit(item)
        assert got == [9]
        # a click on the tick box: the tick changes between press and
        # release, so it is a tick, not a pick
        lst.itemPressed.emit(item)
        item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked
                           else Qt.Checked)
        lst.itemClicked.emit(item)
        assert got == [9]
    finally:
        widget._on_add_click_pick_cb = None
        _back(widget)


def test_a_real_click_on_a_row_picks_it(widget):
    from qtpy.QtCore import QPoint, Qt
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QApplication

    _markers_page(widget)
    got = []
    widget.set_add_callbacks(on_click_pick=got.append)
    widget.resize(340, 900)
    widget.show()
    widget.focus_tab("Add")
    try:
        widget.update_add_ui_state(click_count=2, click_markers=SEEDS)
        widget.add_markers_btn.setChecked(True)
        for _ in range(3):
            QApplication.processEvents()
        lst = widget.add_click_list
        assert not lst.isHidden()
        r = lst.visualItemRect(_rows(lst)[10])
        QTest.mouseClick(lst.viewport(), Qt.LeftButton, Qt.NoModifier,
                         QPoint(r.right() - 20, r.center().y()))
        assert got == [10]
        assert _rows(lst)[10].checkState() == Qt.Checked
    finally:
        widget._on_add_click_pick_cb = None
        widget.hide()
        _back(widget)


def test_the_result_rows_carry_the_seed_count_status_and_bold(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_CHIPS_ROLE,
    )

    _markers_page(widget)
    try:
        widget.update_add_ui_state(
            candidate_seeds={9: 2},
            candidate_status={9: [AP.ADD_ROW_NOT_CONNECTED]},
            click_current=9,
            candidate_ids=[9, 10], candidate_sizes={9: 1234, 10: 60},
            candidate_edge_ids={10})
        rows = _rows(widget.add_candidate_list)
        assert rows[9].text() == "Object 9  ·  2 seeds  ·  1,234"
        assert rows[10].text() == "Object 10  ·  60"
        assert AP.ADD_ROW_NOT_CONNECTED in rows[9].data(ADD_CHIPS_ROLE)
        assert rows[10].data(ADD_CHIPS_ROLE) == ["edge"]
        assert AP.status_tip([AP.ADD_ROW_NOT_CONNECTED]) in rows[9].toolTip()
        # the status word draws before edge / small (they drop first)
        widget.update_add_ui_state(
            candidate_status={10: [AP.ADD_ROW_DIM]},
            candidate_ids=[9, 10], candidate_sizes={9: 1234, 10: 60},
            candidate_edge_ids={10})
        rows = _rows(widget.add_candidate_list)
        assert rows[10].data(ADD_CHIPS_ROLE) == [AP.ADD_ROW_DIM, "edge"]
        tip = rows[10].toolTip()
        assert tip.count(AP.status_tip([AP.ADD_ROW_DIM])) == 1, tip
        # a restyle alone does not repeat it
        widget.update_add_ui_state(click_current=10)
        widget.update_add_ui_state(click_current=9)
        rows = _rows(widget.add_candidate_list)
        tip = rows[10].toolTip()
        assert tip.count(AP.status_tip([AP.ADD_ROW_DIM])) == 1, tip
        assert rows[9].font().bold() and not rows[10].font().bold()
        widget.update_add_ui_state(click_current=10)
        rows = _rows(widget.add_candidate_list)
        assert rows[10].font().bold() and not rows[9].font().bold()
    finally:
        _back(widget)


def test_a_five_digit_row_elides_and_never_widens_the_page(widget):
    from qtpy.QtWidgets import QApplication

    _markers_page(widget)
    widget.resize(340, 350)
    widget.show()
    widget.focus_tab("Add")
    try:
        widget.add_markers_btn.setChecked(True)
        # objects of several seeds (an object of one seed keeps the
        # marker row and its place, as before)
        seeds = [(65520, (1234, 1234, 1234), True),
                 (65520, (1234, 1234, 1240), True),
                 (65521, (1234, 1234, 1250), True),
                 (65521, (1234, 1234, 1260), True)]
        widget.update_add_ui_state(
            click_count=2, click_markers=seeds,
            click_status={65520: [AP.ADD_ROW_NOT_CONNECTED, AP.ADD_ROW_DIM,
                                  AP.ADD_ROW_KEPT]},
            click_current=65520)
        for _ in range(3):
            QApplication.processEvents()
        lst = widget.add_click_list
        assert lst.count() == 2
        assert lst.horizontalScrollBar().maximum() == 0
        content = widget._panel_slots["Add"]["content"]
        scroll = widget._panel_slots["Add"]["scroll"]
        assert content.minimumSizeHint().width() <= scroll.width()
        # the result list too
        widget.update_add_ui_state(
            candidate_seeds={65520: 2},
            candidate_status={65520: [AP.ADD_ROW_NOT_CONNECTED]},
            candidate_ids=[65520, 65521],
            candidate_sizes={65520: 123456, 65521: 99999},
            candidate_edge_ids=set())
        for _ in range(3):
            QApplication.processEvents()
        assert widget.add_candidate_list.horizontalScrollBar().maximum() == 0
        assert content.minimumSizeHint().width() <= scroll.width()
    finally:
        widget.hide()
        _back(widget)


def test_the_hint_line_replaces_the_click_help(widget):
    _markers_page(widget)
    try:
        widget.add_markers_btn.setChecked(True)
        assert widget._add_source_help.text() == AP.ADD_HELP_CLICK_KEYS
        assert not widget._add_source_help.isHidden()
        # it stays at every marker count (it names Shift+click and Ctrl+Z,
        # which only matter once a marker is placed)
        for n, seeds in ((2, SEEDS), (1, SEEDS[:1]), (0, [])):
            widget.update_add_ui_state(click_count=n, click_markers=seeds)
            assert not widget._add_source_help.isHidden(), n
            assert widget._add_source_help.text() == AP.ADD_HELP_CLICK_KEYS
            # the height it costs: the preset's own help line goes while
            # markers are placed (it says what Nuclei finds, not how the
            # markers work)
            assert widget._add_preset_help.isHidden() is (n > 0), n
    finally:
        _back(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(marker_sources=[
        (AP.ADD_ITEM_RUN.format(name="r") + AP.ADD_ITEM_RAW_SUFFIX, True,
         ("labels", "r")),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))], box_active=True)
    try:
        widget.add_marker_source_combo.setCurrentIndex(2)
        widget.add_markers_btn.setChecked(True)
        assert widget._add_source_help.text() == AP.ADD_HELP_CLICK_KEYS
        # Cells keeps its preset help with markers placed
        widget.update_add_ui_state(click_count=2, click_markers=SEEDS)
        assert not widget._add_preset_help.isHidden()
    finally:
        widget.set_add_preset("nuclei")
        _back(widget)


def test_the_hint_line_stays_over_a_result(widget):
    """With a result on screen (from the first click onward with "Update
    as I click") the gesture line stays: extending happens then."""
    _markers_page(widget)
    try:
        widget.add_markers_btn.setChecked(True)
        widget.update_add_ui_state(click_count=2, click_markers=SEEDS)
        widget.update_add_ui_state(candidate_ids=[9, 10],
                                   candidate_sizes={9: 50, 10: 60},
                                   candidate_edge_ids=set())
        assert not widget._add_result_container.isHidden()
        assert not widget._add_source_help.isHidden()
        assert widget._add_source_help.text() == AP.ADD_HELP_CLICK_KEYS
    finally:
        _back(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(marker_sources=[
        (AP.ADD_ITEM_RUN.format(name="r") + AP.ADD_ITEM_RAW_SUFFIX, True,
         ("labels", "r")),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))], box_active=True)
    try:
        widget.add_marker_source_combo.setCurrentIndex(2)
        widget.add_markers_btn.setChecked(True)
        widget.update_add_ui_state(click_count=2, click_markers=SEEDS)
        widget.update_add_ui_state(candidate_ids=[9, 10],
                                   candidate_sizes={9: 50, 10: 60},
                                   candidate_edge_ids=set())
        assert not widget._add_source_help.isHidden()
        assert widget._add_source_help.text() == AP.ADD_HELP_CLICK_KEYS
    finally:
        widget.set_add_preset("nuclei")
        _back(widget)


def test_undo_last_names_ctrl_z(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    want = "Removes the last marker placed (Ctrl + Z in a side view)."
    assert ADD_HELP_LINES["add_click_undo_btn"] == want
    assert widget.add_click_undo_btn.toolTip() == want


def test_the_place_markers_tooltip_names_the_keys(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    tip = ADD_HELP_LINES["add_place_markers_btn"]
    assert widget.add_place_markers_btn.toolTip() == tip
    for key in ("Shift + click", "Alt + click", "Ctrl + click",
                "Ctrl + Z", "Esc"):
        assert key in tip, key
    lst = ADD_HELP_LINES["add_click_list"]
    assert "row" in lst and "current" in lst
    # the constructor's own tooltip says the same (set before the help
    # lines are applied)
    import inspect

    from napari_epicycle.annotator import advanced_features_widget as AFW

    src = inspect.getsource(AFW.AdvancedFeaturesWidget)
    assert re.search(r'add_place_markers_btn\.setToolTip\(\s*ADD_HELP_LINES'
                     r'\["add_place_markers_btn"\]\)', src)


# --------------------------------------------------------------------------- #
# real fonts (offscreen Qt here has none): measured in a child process
# --------------------------------------------------------------------------- #
_FONT_DIR = "C:/Windows/Fonts"

_CHILD_HEAD = """
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
import napari_epicycle.annotator.advanced_features_widget as AFW
from packages.epicycle_annotator import add_preview as AP
"""

_HINT_CHILD = _CHILD_HEAD + """
w = AFW.AdvancedFeaturesWidget(mode="3d_curation")
w.resize(340, 900)
w.show()
for _ in range(5):
    app.processEvents()
w.focus_tab("Add")
w.refresh_add_image_combo(["nuc", "mem"])
w.update_add_ui_state(look_for_items=[], box_active=True)
keys = list(w._add_look_keys)
w.add_look_for_combo.setCurrentIndex(keys.index((AP.LOOK_MARKERS, None)))
w.add_markers_btn.setChecked(True)
w.update_add_ui_state(click_count=2, click_markers=[
    (9, (1, 2, 3), True), (9, (1, 2, 8), True), (10, (4, 5, 6), True)])
for _ in range(10):
    app.processEvents()
lb = w._add_source_help
fm = lb.fontMetrics()
out = {"font": app.font().family(), "shown": lb.text(),
       "visible": lb.isVisible(), "room": lb.contentsRect().width(),
       "line": fm.lineSpacing(),
       "one_line": fm.horizontalAdvance(AP.ADD_HELP_CLICK_KEYS),
       "h": {str(px): lb.heightForWidth(px) for px in range(276, 313, 4)}}
print("HINTJSON" + json.dumps(out), flush=True)
os._exit(0)
"""

_CHIP_CHILD = _CHILD_HEAD + """
calls = []
real = AFW.add_row_layout
def spy(text_w, name_w, chip_ws, tail_w, **kw):
    n = real(text_w, name_w, chip_ws, tail_w, **kw)
    calls.append(n)
    return n
AFW.add_row_layout = spy
w = AFW.AdvancedFeaturesWidget(mode="3d_curation")
w.resize(322, 900)
w.show()
for _ in range(5):
    app.processEvents()
w.focus_tab("Add")
w.refresh_add_image_combo(["nuc", "mem"])
w.update_add_ui_state(look_for_items=[], box_active=True)
keys = list(w._add_look_keys)
w.add_look_for_combo.setCurrentIndex(keys.index((AP.LOOK_MARKERS, None)))
out = {"font": app.font().family()}
for word in (AP.ADD_ROW_NOT_CONNECTED, AP.ADD_ROW_DIM, AP.ADD_ROW_KEPT):
    calls.clear()
    w.update_add_ui_state(candidate_seeds={65520: 3},
                          candidate_status={65520: [word]},
                          candidate_ids=[65520],
                          candidate_sizes={65520: 1234},
                          candidate_edge_ids={65520})
    for _ in range(5):
        app.processEvents()
    lst = w.add_candidate_list
    lst.viewport().grab()
    it = lst.item(0)
    out[word] = {"text": it.text(), "chips": it.data(AFW.ADD_CHIPS_ROLE),
                 "shown": calls[0] if calls else None,
                 "vp": lst.viewport().width()}
print("CHIPJSON" + json.dumps(out), flush=True)
os._exit(0)
"""


def _child(code, tag):
    import json
    import os
    import subprocess
    import sys

    repo = pathlib.Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", "FONT_DIR = {!r}\n".format(_FONT_DIR) + code],
        cwd=str(repo), env=env, capture_output=True, text=True, timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(tag)]
    assert lines, (proc.returncode, proc.stdout[-1500:], proc.stderr[-3000:])
    got = json.loads(lines[-1][len(tag):])
    print("[ADD-REALFONT] {}".format(json.dumps(got)))
    return got


def test_the_hint_is_two_lines_with_real_fonts():
    """The hint line, with markers placed: on the page, and two lines at
    every label width from 276 px (the label's room in a 312 px wide page)
    to 312 px.  Narrower than 276 px it takes three lines."""
    got = _child(_HINT_CHILD, "HINTJSON")
    assert got["shown"] == AP.ADD_HELP_CLICK_KEYS, got
    assert got["visible"], got
    assert got["one_line"] > 312, got           # one line would not fit
    for px, h in got["h"].items():
        assert round(h / got["line"]) == 2, (px, h, got)


def test_a_status_chip_draws_on_a_wide_result_row_with_real_fonts():
    """The widest result row of an object of several seeds -- "Object
    65520  ·  3 seeds  ·  1,234" -- at the dock's width: its status word
    draws (it comes first; edge drops before it)."""
    got = _child(_CHIP_CHILD, "CHIPJSON")
    for word in (AP.ADD_ROW_NOT_CONNECTED, AP.ADD_ROW_DIM, AP.ADD_ROW_KEPT):
        row = got[word]
        assert row["text"] == "Object 65520  ·  3 seeds  ·  1,234", row
        assert row["chips"][0] == word, row
        assert row["shown"] is not None and row["shown"] >= 1, (word, row)


# --------------------------------------------------------------------------- #
# one refused-seed store; the two pick paths agree
# --------------------------------------------------------------------------- #
def _refused(hh):
    """A on nucleus A, then a Shift+click on nucleus B: refused."""
    assert hh._add_marker_click(NUC_A)
    _preview(hh)
    a = _ids(hh)[0]
    assert hh._add_marker_click(NUC_B, shift=True)
    _preview(hh)
    assert hh._add_preview_res.apart_ids == [a]
    assert (hh._add_marker_layer.data == AP.ADD_BALL_APART_VALUE).any()
    assert hh.adv.last("click_status") == {a: [AP.ADD_ROW_NOT_CONNECTED]}
    return a


def test_a_plain_click_on_the_refused_voxel_leaves_the_new_object_plain(off):
    """The refusal says "click it plainly": on the refused seed's own
    voxel that replaces it with a new object, which carries no refused
    mark -- no hollow ball, no chip -- although it sits where the refused
    seed sat."""
    a = _refused(off)
    layer = off._add_marker_layer
    assert off._add_marker_click(NUC_B)
    new = off._add_click_markers[-1]
    b = int(new["id"])
    assert b != a and tuple(new["zyx"]) == NUC_B
    assert not off._add_apart_rec(new)
    assert not (layer.data == AP.ADD_BALL_APART_VALUE).any()
    assert layer.data[NUC_B] == AP.ball_value(b)
    assert off.adv.last("click_status") == {}


def test_a_row_pick_refuses_an_unticked_object(off):
    """An unticked object's balls are off the view: a row click does not
    make it current (Alt+click cannot either), so no Shift+click adds an
    unseen seed to it."""
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(NUC_B)
    a, b = _ids(off)
    off._on_add_click_tick(a, False)
    assert not off._add_pick_object(a)
    assert off._add_current_rec()["id"] == b
    assert not off._add_marker_click(NUC_A, alt=True)
    assert off._add_current_rec()["id"] == b


def test_a_row_pick_needs_the_clicks_to_be_the_markers(off):
    """With another marker source (Anywhere here), the result's ids are
    not the click markers' ids: a row click there picks nothing."""
    assert off._add_marker_click(NUC_A)
    assert off._add_marker_click(NUC_B)
    a = _ids(off)[0]
    cur = off._add_current
    off.adv.look = (AP.LOOK_ANYWHERE, None)
    assert not off._add_pick_object(a)
    assert off._add_current is cur


def test_removing_an_object_s_first_seed_drops_its_refused_marks(off):
    """Ctrl+click on an object's first seed: the seed refused against it is
    now the object's only seed, so it is a plain ball with no chip."""
    a = _refused(off)
    layer = off._add_marker_layer
    assert off._add_marker_click(NUC_A, ctrl=True)
    assert [tuple(m["zyx"]) for m in off._add_click_markers] == [NUC_B]
    assert not off._add_apart_rec(off._add_click_markers[0])
    assert not (layer.data == AP.ADD_BALL_APART_VALUE).any()
    assert layer.data[NUC_B] == AP.ball_value(a)
    assert off.adv.last("click_status") == {}


def test_cells_a_refusal_names_its_refused_seed(h):
    """Cells: a refused run still says which seed was not connected, on
    its row and as a hollow ball, as Nuclei does."""
    import types

    A, B = (10, 15, 10), (10, 15, 20)
    assert h._add_marker_click(A)
    assert h._add_marker_click(B, shift=True)
    a = _ids(h)[0]
    # the refused answer of a run of these markers (the run itself is the
    # engine's test): handed to the page as a finished run's
    h._add_is_stale = lambda *_a, **_k: False
    res = AP.AddPreviewResult(roi=None, preset="cells")
    res.status = "refused"
    res.refusal = AP.ADD_S1_FAILED
    res.seed_zyx = [A, B]
    res.seed_why = [None, "apart"]
    res.apart_ids = [a]
    h._on_add_finished(types.SimpleNamespace(result=res))
    assert h.adv.last("click_status") == {a: [AP.ADD_ROW_NOT_CONNECTED]}
    assert (h._add_marker_layer.data == AP.ADD_BALL_APART_VALUE).any()


def test_a_push_with_markers_restyles_the_result_rows_too(widget):
    """A push that carries the markers AND a new current object makes the
    result list's bold row follow, not only the marker list's."""
    _markers_page(widget)
    try:
        widget.update_add_ui_state(click_current=9, candidate_ids=[9, 10],
                                   candidate_sizes={9: 50, 10: 60},
                                   candidate_edge_ids=set())
        assert _rows(widget.add_candidate_list)[9].font().bold()
        widget.update_add_ui_state(
            click_count=2, click_markers=SEEDS, click_current=10,
            candidate_status={10: [AP.ADD_ROW_DIM]})
        rows = _rows(widget.add_candidate_list)
        assert rows[10].font().bold() and not rows[9].font().bold()
        assert rows[10].data(
            __import__("napari_epicycle.annotator.advanced_features_widget",
                       fromlist=["ADD_CHIPS_ROLE"]).ADD_CHIPS_ROLE) == [
            AP.ADD_ROW_DIM]
    finally:
        _back(widget)

"""Nuclei ▸ "At the markers I place": the page and the orchestrator.

The page half (a real panel, offscreen): the item is the last of Look for
nuclei and is read by its key; picking it shows the Markers card (Place
markers, the Ball, the list) and folds the histogram; a Look-for change
turns Place markers off; a Walls card left open on Cells never shows on
Nuclei; the Cells cards and help lines are what they were.

The orchestrator half (the Qt-free harness): the gather sends the ticked
points, their reserved ids, the Ball and the true spacing only when Look
for nuclei is the markers -- never on Anywhere, even with the source's
fallback reading clicks; S1's marker rows; why a marker made nothing, on
its row after a Preview and after a refusal; the marks cleared wherever the
kept-object marks are; the balls shown with the markers only.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread, Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_click_markers import h  # noqa: F401

MARKERS = (AP.LOOK_MARKERS, None)


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def _pick_markers(w):
    keys = list(w._add_look_keys)
    w.add_look_for_combo.setCurrentIndex(keys.index(MARKERS))


def _back(w):
    w.add_look_for_combo.setCurrentIndex(0)
    w.update_add_ui_state(box_active=False, whole_patch=False,
                          click_count=0, click_markers=[],
                          click_absorbed_ids=[], default_spacing=False,
                          brightness_flat=False, candidate_ids=[],
                          candidate_sizes={}, candidate_edge_ids=set())
    w.add_markers_btn.setChecked(False)
    w.add_walls_btn.setChecked(False)
    reset_add(w)


def _histogram(w):
    w.update_add_ui_state(histogram_data={
        "values": list(np.linspace(0.0, 1.0, 200)), "n_bins": 32})


def _plot(w):
    return (getattr(w, "_add_plot_widget", None)
            or getattr(w, "_add_plot_placeholder", None))


def _run_items():
    return [(AP.look_inside_text("default", "cells"), True,
             (AP.LOOK_INSIDE, "default"), False)]


def test_the_markers_item_is_last_and_read_by_its_key(widget):
    from qtpy.QtCore import Qt

    reset_add(widget)
    c = widget.add_look_for_combo
    for items in ([], _run_items()):
        widget.update_add_ui_state(look_for_items=items)
        last = c.count() - 1
        assert c.itemText(last) == AP.ADD_LOOK_MARKERS
        assert widget._add_look_keys[last] == MARKERS
        assert c.itemData(last, Qt.ToolTipRole) == AP.ADD_LOOK_MARKERS
        assert c.model().item(last).isEnabled()
        assert c.itemText(0) == AP.ADD_LOOK_ANYWHERE
    assert c.count() == 3
    c.setCurrentIndex(c.count() - 1)
    try:
        assert widget.get_add_look_for() == MARKERS
        assert widget.add_look_markers()
        assert not widget.add_look_inside()
        # a run listing keeps the markers picked
        widget.update_add_ui_state(look_for_items=[])
        assert widget.get_add_look_for() == MARKERS
    finally:
        _back(widget)
    widget.update_add_ui_state(look_for_items=[])
    assert not widget.add_look_markers()


def test_a_new_page_offers_the_item_before_any_run_is_listed(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget()
    try:
        c = w.add_look_for_combo
        assert [c.itemText(i) for i in range(c.count())] == [
            AP.ADD_LOOK_ANYWHERE, AP.ADD_LOOK_MARKERS]
        assert w.get_add_look_for() == (AP.LOOK_ANYWHERE, None)
    finally:
        w.setParent(None)
        w.deleteLater()


def test_the_combo_s_help_names_the_markers(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    reset_add(widget)
    assert ADD_HELP_LINES["add_look_for_combo"] == AP.ADD_HELP_LOOK_FOR
    assert widget.add_look_for_combo.toolTip() == AP.ADD_HELP_LOOK_FOR


def test_picking_markers_shows_the_markers_card_and_folds_the_histogram(
        widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    _histogram(widget)
    assert not _plot(widget).isHidden()
    _pick_markers(widget)
    try:
        assert not widget._add_cells_container.isHidden()
        assert not widget.add_markers_btn.isHidden()
        assert widget.add_markers_btn.isChecked()
        assert not widget._add_markers_container.isHidden()
        assert not widget._add_click_rows.isHidden()
        assert not widget.add_place_markers_btn.isHidden()
        assert not widget._add_click_radius_row.isHidden()
        # the Cells parts of the card are hidden
        for w in (widget._add_source_row, widget._add_nuclei_rows,
                  widget.add_run_count_label,
                  widget._add_membrane_only_label, widget.add_walls_btn,
                  widget._add_walls_container):
            assert w.isHidden(), w
        # no marker yet: the clicks' hint line (it replaced the Nuclei help)
        assert not widget._add_source_help.isHidden()
        assert widget._add_source_help.text() == AP.ADD_HELP_CLICK_KEYS
        # the histogram folds; the threshold stays, its "drag the red line"
        # help goes with the histogram
        assert _plot(widget).isHidden()
        assert not widget._add_threshold_row.isHidden()
        assert not widget.add_clahe_cb.isHidden()
        assert widget._add_threshold_help.text() == AP.ADD_HELP_THRESHOLD
        assert widget._add_threshold_help.isHidden()
        # Separate touching objects shows (the markers' on / off) with the
        # markers' own Spacing in place of Anywhere's
        assert not widget.add_separate_cb.isHidden()
        assert widget._add_spacing_row.isHidden()
        assert not widget._add_marker_spacing_row.isHidden()
        assert not widget._add_anywhere_adv.isHidden()
        # an Auto status still shows
        widget.update_add_ui_state(brightness_flat=True)
        assert not widget._add_threshold_help.isHidden()
        assert widget._add_threshold_help.text() == (
            AP.ADD_AUTO_NOTHING_TO_SPLIT)
        widget.update_add_ui_state(brightness_auto=float("nan"))
        assert not widget._add_threshold_help.isHidden()
        assert widget._add_threshold_help.text() == AP.ADD_AUTO_PENDING
        # one marker placed: the hint line stays, the list shows
        widget.update_add_ui_state(click_count=1,
                                   click_markers=[(9, (1, 2, 3), True)])
        assert not widget._add_source_help.isHidden()
        assert widget._add_source_help.text() == AP.ADD_HELP_CLICK_KEYS
        assert not widget.add_click_list.isHidden()
        # over candidates: the click rows and the hint line (extending
        # happens over a result), not the list
        widget.update_add_ui_state(candidate_ids=[9],
                                   candidate_sizes={9: 10},
                                   candidate_edge_ids=set())
        assert not widget._add_click_rows.isHidden()
        assert widget.add_click_list.isHidden()
        assert not widget._add_source_help.isHidden()
    finally:
        _back(widget)


def test_back_on_anywhere_the_page_is_what_it_was(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    _histogram(widget)
    _pick_markers(widget)
    widget.add_look_for_combo.setCurrentIndex(0)
    try:
        # the container holds only the Tissue outline card's chevron here,
        # closed: no Markers card, no Walls card, nothing open
        box = widget._add_cells_container
        assert not box.isHidden()
        lay = box.layout()
        shown = [lay.itemAt(i).widget() for i in range(lay.count())
                 if lay.itemAt(i).widget() is not None
                 and not lay.itemAt(i).widget().isHidden()]
        assert shown == [widget.add_nuc_outline_btn]
        assert widget.add_markers_btn.isHidden()
        assert widget._add_markers_container.isHidden()
        assert widget.add_walls_btn.isHidden()
        assert not widget.add_nuc_outline_btn.isChecked()
        assert widget._add_nuc_outline_container.isHidden()
        assert not _plot(widget).isHidden()
        assert not widget._add_threshold_help.isHidden()
        assert not widget.add_separate_cb.isHidden()
        assert not widget._add_spacing_row.isHidden()
        assert widget._add_marker_spacing_row.isHidden()
        assert "look_for" not in widget.get_add_params()
        assert "marker_spacing" not in widget.get_add_params()
    finally:
        _back(widget)


def test_walls_open_on_cells_shows_no_walls_body_on_nuclei_anywhere(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    _histogram(widget)
    widget.set_add_preset("cells")
    widget.add_walls_btn.setChecked(True)
    assert not widget._add_walls_container.isHidden()
    try:
        widget.set_add_preset("nuclei")
        assert widget._add_walls_container.isHidden()
        assert widget._add_markers_container.isHidden()
        assert not _plot(widget).isHidden()
        # on the markers: Markers opens, Walls stays hidden
        _pick_markers(widget)
        assert not widget._add_markers_container.isHidden()
        assert widget._add_walls_container.isHidden()
        assert not widget.add_walls_btn.isVisibleTo(widget)
        # back on Cells the user's Walls card is open, alone
        widget.set_add_preset("cells")
        assert widget.add_walls_btn.isChecked()
        assert not widget._add_walls_container.isHidden()
        assert widget._add_markers_container.isHidden()
    finally:
        widget.set_add_preset("nuclei")
        _back(widget)


@pytest.mark.parametrize("pick", [1, 0])
def test_the_source_help_stays_on_cells_for_every_source(widget, pick):
    """Cells' help line under the From row shows for the nuclei and the run
    source as it always did."""
    reset_add(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(marker_sources=[
        (AP.ADD_ITEM_RUN.format(name="r") + AP.ADD_ITEM_RAW_SUFFIX, True,
         ("labels", "r")),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))], box_active=True)
    try:
        widget.add_marker_source_combo.setCurrentIndex(pick)
        widget.add_markers_btn.setChecked(True)
        assert not widget._add_source_help.isHidden()
        assert not widget._add_source_row.isHidden()
        assert widget._add_source_help.text() != AP.ADD_HELP_NUCLEI_CLICKS
    finally:
        widget.set_add_preset("nuclei")
        _back(widget)


def test_a_look_for_change_turns_place_markers_off_on_the_page(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=_run_items(), box_active=True)
    _pick_markers(widget)
    try:
        widget.add_place_markers_btn.setChecked(True)
        widget.add_look_for_combo.setCurrentIndex(0)
        assert not widget.add_place_markers_btn.isChecked()
        widget.update_add_ui_state(place_markers_on=True)
        _pick_markers(widget)
        assert not widget.add_place_markers_btn.isChecked()
    finally:
        _back(widget)
        widget.update_add_ui_state(look_for_items=[])


def test_the_markers_title_and_the_advanced_summary(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    _pick_markers(widget)
    try:
        widget.update_add_ui_state(
            click_count=3, click_markers=[(i, (1, 2, 3), True)
                                          for i in (4, 5, 6)])
        title = str(widget.add_markers_btn.property(
            "_epicycle_chevron_title"))
        assert title == AP.markers_summary(AP.SOURCE_CLICKS, n_clicks=3)
        adv = str(widget._add_advanced_toggle.property(
            "_epicycle_chevron_title"))
        # read as Anywhere's, with the markers' own Spacing
        assert adv.startswith("Advanced: separate touching · spacing {}"
                              .format(AP.ADD_MARKER_SPACING_DEFAULT)), adv
        widget.add_separate_cb.setChecked(False)
        adv = str(widget._add_advanced_toggle.property(
            "_epicycle_chevron_title"))
        # unticked, the markers' Spacing is hidden and not in the summary
        assert adv.startswith("Advanced: one per blob"), adv
        assert "spacing" not in adv, adv
    finally:
        _back(widget)


def test_a_marker_row_says_why_it_made_nothing(widget):
    from qtpy.QtCore import Qt

    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    _pick_markers(widget)
    try:
        widget.update_add_ui_state(
            click_count=3, click_markers=[(i, (1, 2, 3), True)
                                          for i in (4, 5, 6)],
            click_absorbed_ids=[4],
            click_absorbed_tips={5: AP.ADD_MARKER_DIM_TIP,
                                 6: AP.ADD_MARKER_SMALL_TIP})
        lst = widget.add_click_list
        rows = {lst.item(i).data(Qt.UserRole): lst.item(i)
                for i in range(lst.count())}
        assert rows[4].toolTip() == AP.ADD_MARKER_ABSORBED_TIP
        assert rows[5].toolTip() == AP.ADD_MARKER_DIM_TIP
        assert rows[6].toolTip() == AP.ADD_MARKER_SMALL_TIP
        for i in (4, 5, 6):
            assert rows[i].foreground().color().name() == "#777777"
        # the marks cleared: the rows read the list's help again
        widget.update_add_ui_state(click_absorbed_ids=[],
                                   click_absorbed_tips={})
        rows = {lst.item(i).data(Qt.UserRole): lst.item(i)
                for i in range(lst.count())}
        for i in (4, 5, 6):
            assert rows[i].toolTip() == ADD_HELP_LINES["add_click_list"]
            assert rows[i].foreground().color().name() == "#eeeeee"
    finally:
        _back(widget)


def test_the_default_spacing_amber_shows_on_the_markers(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True,
                               default_spacing=True)
    try:
        assert widget.add_spacing_note.isHidden()        # Anywhere: never
        _pick_markers(widget)
        assert not widget.add_spacing_note.isHidden()
        assert widget.add_spacing_note.text() == (
            AP.ADD_NOTE_DEFAULT_SPACING_BOXED)
    finally:
        _back(widget)


def test_markers_on_a_whole_patch_show_the_card(widget):
    """A whole patch shows the Markers card as a box does: Place markers,
    the Ball, the click help, the threshold's Auto line."""
    def _seen(w):
        return dict(
            card=w._add_cells_container.isHidden(),
            body=w._add_markers_container.isHidden(),
            rows=w._add_click_rows.isHidden(),
            place=w.add_place_markers_btn.isHidden(),
            help=w._add_source_help.isHidden(),
            help_text=w._add_source_help.text(),
            threshold=w._add_threshold_help.isHidden(),
            threshold_text=w._add_threshold_help.text())

    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    _pick_markers(widget)
    try:
        box = _seen(widget)
        widget.update_add_ui_state(whole_patch=True)
        whole = _seen(widget)
        assert not whole["card"] and not whole["body"]
        assert not whole["rows"] and not whole["place"]
        assert whole["help_text"] == AP.ADD_HELP_CLICK_KEYS
        assert whole == box
        # a marker placed on the whole patch is listed as on a box
        widget.update_add_ui_state(click_count=1,
                                   click_markers=[(9, (1, 2, 3), True)])
        assert not widget.add_click_list.isHidden()
        assert widget.add_click_list.count() == 1
    finally:
        _back(widget)


def test_inside_and_cells_never_show_the_nuclei_help(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=_run_items(), box_active=True)
    widget.add_look_for_combo.setCurrentIndex(1)
    try:
        assert widget.add_look_inside()
        assert widget._add_cells_container.isHidden()
        assert widget._add_source_help.text() != AP.ADD_HELP_NUCLEI_CLICKS
    finally:
        _back(widget)
        widget.update_add_ui_state(look_for_items=[])
    # Cells, every source, Markers open: never the Nuclei markers' help
    widget.set_add_preset("cells")
    widget.update_add_ui_state(marker_sources=[
        (AP.ADD_ITEM_RUN.format(name="r") + AP.ADD_ITEM_RAW_SUFFIX, True,
         ("labels", "r")),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))], box_active=True)
    try:
        widget.add_markers_btn.setChecked(True)
        for pick in range(widget.add_marker_source_combo.count()):
            widget.add_marker_source_combo.setCurrentIndex(pick)
            for n in (0, 1):
                widget.update_add_ui_state(
                    click_count=n,
                    click_markers=[(9, (1, 2, 3), True)][:n])
                assert widget._add_source_help.text() != (
                    AP.ADD_HELP_NUCLEI_CLICKS), (pick, n)
    finally:
        widget.set_add_preset("nuclei")
        _back(widget)


# --------------------------------------------------------------------------- #
# the orchestrator (the Qt-free harness)
# --------------------------------------------------------------------------- #
def _nuclei_harness(monkeypatch, *, look=MARKERS, spacing=(1.0, 1.0, 1.0),
                    centres=((10, 10, 10),),
                    box=(slice(0, 20), slice(0, 30), slice(0, 30))):
    install_fakes(monkeypatch)
    vol = np.zeros((20, 30, 30), np.int32)
    zz, yy, xx = np.indices(vol.shape)
    img = np.zeros(vol.shape, np.float64)
    for cz, cy, cx in centres:
        img[((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= 16] = 1.0
    hh = Harness(vol, images={"nuc": img}, spacing=spacing)
    hh.adv.look = look
    hh.adv.params["click_radius_um"] = 1.5
    hh.place_box(box)
    hh._add_threshold_value = 0.5
    hh._add_click_markers = [
        {"zyx": (10, 10, 10), "id": 1, "ticked": True},
        {"zyx": (10, 22, 22), "id": 2, "ticked": True},
        {"zyx": (3, 3, 3), "id": 3, "ticked": False}]
    return hh


def _run(hh):
    hh._preview_add()
    assert hh._add_busy, hh._add_transient
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    return hh._add_preview_res


def test_anywhere_sends_no_points_even_when_the_source_reads_clicks(
        monkeypatch):
    hh = _nuclei_harness(monkeypatch, look=(AP.LOOK_ANYWHERE, None))
    assert hh.adv.source == ("paint", None)        # the trap
    kw, why = hh._add_gather_preview()
    assert why is None
    assert "click_points" not in kw and "reserved_ids" not in kw
    assert "look_for" not in kw["params"]
    assert "click_radius_um" not in kw["params"]
    assert kw["spacing"] == hh._add_spacing()


def test_the_markers_send_the_ticked_points_their_ids_and_the_ball(
        monkeypatch):
    hh = _nuclei_harness(monkeypatch, spacing=(2.0, 0.5, 0.5))
    kw, why = hh._add_gather_preview()
    assert why is None
    assert kw["click_points"] == [(10, 10, 10), (10, 22, 22)]
    assert kw["reserved_ids"] == [1, 2]
    p = kw["params"]
    assert p["look_for"] == AP.LOOK_MARKERS
    assert p["click_radius_um"] == 1.5
    assert p["threshold"] == 0.5
    # the drawn balls' spacing, not the layer scale Anywhere reads
    assert kw["spacing"] == (2.0, 0.5, 0.5)
    assert kw["spacing"] != hh._add_spacing()


def test_inside_each_cell_sends_no_points(monkeypatch):
    from .test_add_derive_wiring import _harness

    hi, _img, _cells = _harness(monkeypatch)
    hi._add_click_markers = [{"zyx": (2, 2, 2), "id": 50, "ticked": True}]
    kw, why = hi._add_gather_preview()
    assert why is None, why
    assert "click_points" not in kw and "reserved_ids" not in kw


def test_a_look_for_change_turns_place_markers_off_and_clicks_place_nothing(
        monkeypatch):
    hh = _nuclei_harness(monkeypatch)
    hh._add_place_markers_on = True
    hh.adv.look = (AP.LOOK_ANYWHERE, None)
    hh._on_add_look_for_changed(hh.adv.look)
    assert hh._add_place_markers_on is False
    assert hh.adv.last("place_markers_on") is False
    before = list(hh._add_click_markers)
    assert hh._add_marker_click((10, 15, 15)) is False
    assert hh._add_click_markers == before


def _s1(hh):
    return AP.s1_sentence(hh._add_s1_state())


def test_s1_on_the_markers(monkeypatch):
    hh = _nuclei_harness(monkeypatch)
    hh._add_click_markers = []
    assert AP.preview_block(hh._add_s1_state()) == AP.ADD_S1_NO_MARKERS
    hh._add_place_markers_on = True
    assert _s1(hh) == AP.ADD_S1_NO_MARKERS_ON
    hh._add_click_markers = [{"zyx": (10, 10, 10), "id": 1, "ticked": False},
                             {"zyx": (10, 22, 22), "id": 2, "ticked": False}]
    assert _s1(hh) == AP.nuclei_tick_one_line(2)
    hh._add_click_markers[0]["ticked"] = True
    assert _s1(hh) == AP.box_line((20, 30, 30), preset="nuclei", n_clicks=2,
                                  n_ticked=1, look_for=AP.LOOK_MARKERS)
    hh._add_whole_patch = True
    # a whole patch takes the markers: its own ready line, and the gather
    # sends the ticked points as on a box
    assert _s1(hh) == AP.box_line((20, 30, 30), preset="nuclei", n_clicks=2,
                                  n_ticked=1, look_for=AP.LOOK_MARKERS,
                                  whole_patch=True)
    assert _s1(hh).startswith("Whole patch ")
    kw, why = hh._add_gather_preview()
    assert why is None, why
    assert kw["click_points"] == [(10, 10, 10)]
    assert kw["reserved_ids"] == [1]
    assert kw["params"]["look_for"] == AP.LOOK_MARKERS
    assert kw["params"]["whole_patch"] is True


def test_a_whole_patch_gives_the_nucleus_a_box_gives(monkeypatch):
    """The same ticked markers make the same nucleus on a whole patch as on
    a box strictly inside the patch (no engine change; the threshold is a
    number, so the only difference is the roi): the whole patch's result,
    cut to the box, is the box's result.  On a whole patch the image is the
    layer's own array, read on the worker -- the crop helper (a float64
    copy of the patch on the main thread) is never called -- and the
    box-info gather measures the whole patch for Auto and the histogram."""
    box_roi = (slice(4, 16), slice(4, 18), slice(4, 18))
    one = [{"zyx": (10, 10, 10), "id": 1, "ticked": True}]
    box = _nuclei_harness(monkeypatch, box=box_roi)
    box._add_click_markers = [dict(m) for m in one]
    assert box._add_roi == box_roi
    res_box = _run(box)
    whole = _nuclei_harness(monkeypatch, box=box_roi)
    whole._add_click_markers = [dict(m) for m in one]
    whole._start_add_whole()
    assert whole._add_whole_patch
    assert whole._add_roi == tuple(slice(0, n) for n in (20, 30, 30))
    info = whole._add_gather_info()
    assert info is not None and info["image_roi"] is not None
    # the layer's own array, read on the worker (no copy of the patch on
    # the main thread)
    assert info["image_roi"] is whole._refine_channel_source("nuc")
    crops = []
    real_crop = whole._add_channel_crop
    whole._add_channel_crop = (
        lambda *a, **k: crops.append(a) or real_crop(*a, **k))
    kw, why = whole._add_gather_preview()
    assert why is None, why
    assert kw["image_roi"] is whole._refine_channel_source("nuc")
    res_whole = _run(whole)
    assert crops == [], "a whole patch copies no crop on the main thread"
    assert res_box is not None and res_whole is not None
    assert list(res_whole.ids) == list(res_box.ids) == [1]
    assert res_whole.result.shape == (20, 30, 30)
    assert res_box.result.shape == (12, 14, 14)
    assert np.array_equal(res_whole.result[box_roi], res_box.result)
    outside = np.array(res_whole.result, copy=True)
    outside[box_roi] = 0
    assert not outside.any(), "nothing made outside the box's space"
    assert res_whole.sizes == res_box.sizes


def test_a_click_on_a_whole_patch_places_a_marker(h):
    """Nuclei ▸ the markers on a whole patch: Place markers stays on and a
    click inside the patch places a marker."""
    h.adv.preset = "nuclei"
    h.adv.look = MARKERS
    h._start_add_whole()
    assert h._add_whole_patch
    assert h._add_place_markers_on is True
    assert h.adv.last("place_markers_on") is not False
    # outside the box from before the whole patch, inside the patch
    assert h._add_marker_click((19, 29, 29))
    assert [tuple(m["zyx"]) for m in h._add_markers()] == [(19, 29, 29)]
    s = h._add_s1_state()
    assert s["whole_patch"] and s["n_clicks"] == 1
    # one ticked marker: the whole patch is ready to preview
    assert AP.preview_block(s) is None


def test_a_dim_marker_is_marked_after_the_preview_and_counted(monkeypatch):
    hh = _nuclei_harness(monkeypatch)
    res = _run(hh)
    assert res is not None and res.status == "ok"
    assert list(res.ids) == [1]
    assert hh._add_marker_why == {2: "dim"}
    push = hh._add_marker_push()
    assert push["click_absorbed_tips"] == {2: AP.ADD_MARKER_DIM_TIP}
    assert hh.adv.last("click_absorbed_tips") == {2: AP.ADD_MARKER_DIM_TIP}
    notes = hh.adv.last("notes")
    assert notes and AP.ADD_NOTE_MARKERS_DIM_ONE in notes[0][1]


def test_a_refusal_still_marks_the_rows(monkeypatch):
    hh = _nuclei_harness(monkeypatch)
    hh._add_click_markers = [{"zyx": (10, 22, 22), "id": 1, "ticked": True},
                             {"zyx": (3, 25, 25), "id": 2, "ticked": True}]
    assert _run(hh) is None                 # a refusal keeps no result
    assert hh._add_transient == AP.ADD_S1_MARKERS_NONE_ON_STAIN
    assert hh._add_marker_why == {1: "dim", 2: "dim"}
    assert hh.adv.last("click_absorbed_tips") == {
        1: AP.ADD_MARKER_DIM_TIP, 2: AP.ADD_MARKER_DIM_TIP}


def test_a_cells_refusal_is_unchanged(monkeypatch):
    """Cells' refusal path keeps no marks (it never did)."""
    hh = _nuclei_harness(monkeypatch)
    hh._add_absorbed_ids = set()
    hh._add_marker_why = {}

    class _Res:
        status = "refused"
        refusal = AP.ADD_S1_ALL_FILTERED
        refusal_reason = ""
        preset = "cells"
        absorbed_ids = [1]
        marker_why = {2: "dim"}

        def log_line(self):
            return "[ADD] test"

    hh.adv.preset = "cells"
    # as launched (the stale guard reads the launch's labels, patch, run
    # and box)
    hh._add_launch_labels_name = hh._active_labels_name
    hh._add_launch_patch = hh.ctx["patch_name"]
    hh._add_launch_run_epoch = hh.ctx["run_epoch"]
    hh._add_launch_roi = tuple(hh._add_roi)

    class _Out:
        result = _Res()

    hh._on_add_finished(_Out())
    assert hh._add_transient == AP.ADD_S1_ALL_FILTERED
    assert hh._add_absorbed_ids == set()
    assert hh._add_marker_why == {}


def test_the_marks_are_cleared_at_every_exit_and_a_reused_id_has_no_tip(
        monkeypatch):
    hh = _nuclei_harness(monkeypatch)
    _run(hh)
    assert hh._add_marker_why
    hh._reject_add()
    assert hh._add_marker_why == {} and hh._add_absorbed_ids == set()
    # a new box: a marker that reserves the old id carries no old tip
    hh.place_box((slice(0, 20), slice(0, 30), slice(0, 30)))
    hh._add_click_markers = [{"zyx": (10, 22, 22), "id": 2, "ticked": True}]
    push = hh._add_marker_push()
    assert push["click_absorbed_tips"] == {}
    assert push["click_absorbed_ids"] == []


def test_the_marks_are_cleared_when_a_box_closes_back_to_the_review(
        monkeypatch):
    hh = _nuclei_harness(monkeypatch)
    hh._add_marker_why = {2: "dim"}
    hh._add_absorbed_ids = {1}
    hh._add_resume_review()
    assert hh._add_marker_why == {} and hh._add_absorbed_ids == set()


def test_the_marks_shift_with_a_rebase(monkeypatch):
    hh = _nuclei_harness(monkeypatch)
    hh._add_markers_base = 1
    hh._add_marker_why = {2: "dim"}
    hh._add_absorbed_ids = {3}
    assert hh._add_rebase_markers(11)
    assert hh._add_marker_why == {12: "dim"}
    assert hh._add_absorbed_ids == {13}


@pytest.mark.parametrize("change", ["preset", "look"])
def test_a_preset_or_look_for_change_drops_the_marks_and_keeps_the_markers(
        monkeypatch, change):
    hh = _nuclei_harness(monkeypatch)
    hh._add_marker_why = {2: "dim"}
    hh._add_absorbed_ids = {1}
    records = [dict(m) for m in hh._add_click_markers]
    if change == "preset":
        hh.adv.preset = "cells"
        hh._on_add_preset_changed("cells")
        assert [dict(m) for m in hh._add_click_markers] == records
        hh.adv.preset = "nuclei"
        hh._on_add_preset_changed("nuclei")
    else:
        hh.adv.look = (AP.LOOK_ANYWHERE, None)
        hh._on_add_look_for_changed(hh.adv.look)
        hh.adv.look = MARKERS
        hh._on_add_look_for_changed(hh.adv.look)
    # the same ids and ticks, the marks gone
    assert [dict(m) for m in hh._add_click_markers] == records
    assert hh._add_marker_why == {} and hh._add_absorbed_ids == set()
    assert hh.adv.last("click_absorbed_tips") == {}


def test_a_preset_change_with_no_marks_pushes_no_marker_list(monkeypatch):
    """The marker list is pushed again only when a preset change dropped
    marks; with none it is not touched."""
    hh = _nuclei_harness(monkeypatch)
    hh._add_marker_why = {}
    hh._add_absorbed_ids = set()

    def _lists():
        return sum(1 for kw in hh.adv.pushes if "click_markers" in kw)

    before = _lists()
    hh.adv.preset = "cells"
    hh._on_add_preset_changed("cells")
    hh.adv.preset = "nuclei"
    hh._on_add_preset_changed("nuclei")
    assert _lists() == before
    # with marks it is pushed, the marks gone
    hh._add_marker_why = {2: "dim"}
    hh._on_add_preset_changed("nuclei")
    assert _lists() == before + 1
    assert hh.adv.last("click_absorbed_tips") == {}


# --------------------------------------------------------------------------- #
# a removed marker takes its marks with it: the id it frees, reserved again
# by the next marker, starts clean
# --------------------------------------------------------------------------- #
#: a nucleus in the click-marker fixture's box, and two spots on dim stain
NUC_AT = (12, 20, 20)
DIM_A, DIM_B = (12, 8, 22), (14, 24, 8)


def _nuclei_clicks(h):
    """The click-marker fixture on Nuclei ▸ the markers, with a nucleus in
    its "nuc" channel."""
    from napari.layers import Image

    vol = h.sync_manager.shared_labels["L"]
    zz, yy, xx = np.indices(vol.shape)
    img = np.zeros(vol.shape, np.float64)
    img[((zz - NUC_AT[0]) ** 2 + (yy - NUC_AT[1]) ** 2
         + (xx - NUC_AT[2]) ** 2) <= 16] = 1.0
    h.viewer.layers.append(Image(img, name="nuc"))
    h.adv.preset = "nuclei"
    h.adv.look = MARKERS
    h.adv.params["click_radius_um"] = 1.0
    h._add_threshold_value = 0.5
    return h


def _ids(h):
    return [int(m["id"]) for m in h._add_click_markers]


def _clean(h, mid):
    """``mid`` carries no mark: no reason, not absorbed, a normal row."""
    why = getattr(h, "_add_marker_why", None) or {}
    absorbed = getattr(h, "_add_absorbed_ids", None) or set()
    assert mid not in why, why
    assert mid not in absorbed, absorbed
    push = h._add_marker_push()
    assert mid not in push["click_absorbed_tips"], push
    assert mid not in push["click_absorbed_ids"], push
    assert mid not in (h.adv.last("click_absorbed_tips") or {})
    assert mid not in (h.adv.last("click_absorbed_ids") or [])


def _place_dim_last_and_preview(h):
    assert h._add_marker_click(NUC_AT)
    assert h._add_marker_click(DIM_A)
    res = _run(h)
    assert res is not None and res.status == "ok", h._add_transient
    gone = _ids(h)[-1]
    assert h._add_marker_why == {gone: "dim"}
    assert h.adv.last("click_absorbed_tips") == {gone: AP.ADD_MARKER_DIM_TIP}
    return gone


def _place_again(h, gone, at=DIM_B):
    assert h._add_marker_click(at)
    new = _ids(h)[-1]
    assert new == gone                      # the freed id, reserved again
    _clean(h, new)


def test_undo_last_takes_its_marks_and_a_reused_id_starts_clean(h):  # noqa: F811
    _nuclei_clicks(h)
    gone = _place_dim_last_and_preview(h)
    h._add_click_undo()
    assert gone not in _ids(h)
    assert gone not in h._add_marker_why
    _place_again(h, gone)


def test_ctrl_click_takes_its_marks_and_a_reused_id_starts_clean(h):  # noqa: F811
    _nuclei_clicks(h)
    gone = _place_dim_last_and_preview(h)
    assert h._add_marker_click(DIM_A, ctrl=True)
    assert gone not in _ids(h)
    assert gone not in h._add_marker_why
    _place_again(h, gone)


def test_remove_all_takes_the_marks_and_a_reused_id_starts_clean(h):  # noqa: F811
    _nuclei_clicks(h)
    assert h._add_marker_click(DIM_A)        # the first id is the dim one
    assert h._add_marker_click(NUC_AT)
    res = _run(h)
    assert res is not None and res.status == "ok", h._add_transient
    first = _ids(h)[0]
    assert h._add_marker_why == {first: "dim"}
    h._add_click_clear()
    assert h._add_click_markers == []
    assert h._add_marker_why == {} and h._add_absorbed_ids == set()
    _place_again(h, first)


def test_a_box_move_drop_takes_its_marks_and_a_reused_id_starts_clean(h):  # noqa: F811
    _nuclei_clicks(h)
    gone = _place_dim_last_and_preview(h)
    # the box moves off DIM_A (y 8): that marker is dropped
    h.place_box((slice(2, 18), slice(14, 28), slice(2, 28)))
    h._add_keep_clicks_in_box()
    assert gone not in _ids(h)
    assert gone not in h._add_marker_why
    _place_again(h, gone)


def test_a_refusal_then_remove_all_and_a_reused_id_starts_clean(h):  # noqa: F811
    _nuclei_clicks(h)
    assert h._add_marker_click(DIM_A)
    assert h._add_marker_click(DIM_B)
    assert _run(h) is None                   # nothing on stain: refused
    assert h._add_transient == AP.ADD_S1_MARKERS_NONE_ON_STAIN
    ids = _ids(h)
    assert h._add_marker_why == {i: "dim" for i in ids}
    h._add_click_clear()
    assert h._add_marker_why == {}
    _place_again(h, ids[0], at=(14, 24, 24))


def test_cells_undo_last_takes_the_kept_mark_and_a_reused_id_starts_clean(
        h):  # noqa: F811
    """The same slip on Cells' click markers: a marker the last Preview
    found on a kept object, undone, frees its id without its mark."""
    assert h.adv.preset == "cells"
    assert h._add_marker_click((12, 20, 20))
    assert h._add_marker_click((12, 8, 22))
    gone = _ids(h)[-1]
    h._add_absorbed_ids = {gone}             # as the last Preview left it
    h._add_click_undo()
    assert gone not in h._add_absorbed_ids
    _place_again(h, gone)


def test_a_clicked_nucleus_at_the_box_edge_starts_unticked(monkeypatch):
    """A clicked nucleus is flagged like any candidate: at the box edge it
    starts unticked, and S1 says why."""
    hh = _nuclei_harness(monkeypatch, centres=((10, 10, 10), (10, 25, 10)))
    hh._add_click_markers = [{"zyx": (10, 10, 10), "id": 1, "ticked": True},
                             {"zyx": (10, 25, 10), "id": 2, "ticked": True}]
    res = _run(hh)
    assert res.status == "ok"
    assert sorted(res.ids) == [1, 2]
    assert set(res.edge_ids) == {2}
    assert 2 in set(hh.adv.last("candidate_edge_ids"))
    hh.adv.checked = [1]
    assert _s1(hh) == AP.found_line(2, k_edge=1)


# --------------------------------------------------------------------------- #
# the balls
# --------------------------------------------------------------------------- #
def test_the_balls_show_on_nuclei_only_with_the_markers(h):  # noqa: F811
    assert h._add_marker_click((12, 20, 20))
    layer = h._add_marker_layer
    assert layer.visible
    h.adv.preset = "nuclei"
    h.adv.look = (AP.LOOK_ANYWHERE, None)
    h._on_add_preset_changed("nuclei")
    assert not layer.visible
    h.adv.look = MARKERS
    h._on_add_look_for_changed(h.adv.look)
    assert layer.visible
    for orient in ("XY", "XZ", "YZ"):
        sub = h.subviewer_manager.get_subviewer_layer(layer.name, orient)
        assert sub is None or sub.visible
    # a whole patch: they follow Cells' rule (shown)
    h._add_whole_patch = True
    h._add_update_ball_visibility()
    assert layer.visible
    h._add_whole_patch = False
    h.adv.look = (AP.LOOK_ANYWHERE, None)
    h._on_add_look_for_changed(h.adv.look)
    assert not layer.visible
    # the record is kept through all of it
    assert len(h._add_click_markers) == 1

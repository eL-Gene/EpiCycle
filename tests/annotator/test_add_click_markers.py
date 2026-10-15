"""Click markers and the ball layer.

Refusals (on a kept object / outside the box / busy), Ctrl+click removal,
a box move keeping and dropping clicks, the ball layer flagged before it is
appended and mirrored as VIEWS, the side views' active layer put back,
Place markers unchecked on tab / preset / source change, the balls hidden
on Cells -> Nuclei and shown again.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.orientation import transpose_data

from .add_harness import Harness, install_fakes, marker_zyx


class _Sel:
    def __init__(self):
        self.active = None


class _Layers(list):
    def __init__(self):
        super().__init__()
        self.selection = _Sel()

    def append(self, layer):
        super().append(layer)
        self.selection.active = layer        # napari makes it active


class _VM:
    def __init__(self):
        self.layers = _Layers()


class _SubMgr:
    """Mimics add_labels_to_subviewers: transposed VIEWS, made active."""

    def __init__(self, h):
        self.h = h
        self.copies = {}
        self.appended_flagged = []

    def add_labels_to_subviewers(self, layer, sync_manager=None,
                                 drawing_handler=None):
        from napari.layers import Labels

        for orient, vm in (("XY", self.h.viewer_model_xy),
                           ("XZ", self.h.viewer_model_xz),
                           ("YZ", self.h.viewer_model_yz)):
            c = Labels(transpose_data(layer.data, orient),
                       name=f"{layer.name}_{orient}")
            vm.layers.append(c)
            self.copies[(layer.name, orient)] = c

    def get_subviewer_layer(self, name, orient):
        return self.copies.get((name, orient))

    def remove_layer_from_subviewers(self, name):
        for key in [k for k in self.copies if k[0] == name]:
            del self.copies[key]


class _ViewerLayers(_Layers):
    def append(self, layer):
        # the flag must already be set when the layer is inserted
        _ViewerLayers.flag_at_append.append(
            getattr(layer, "_epicycle_temporary", False))
        super().append(layer)


_ViewerLayers.flag_at_append = []


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    vol = np.zeros((20, 30, 30), np.int32)
    vol[5:9, 5:9, 5:9] = 7
    hh = Harness(vol, spacing=(0.5, 0.5, 0.5))
    hh.viewer.layers = _ViewerLayers()
    hh.viewer_model_xy, hh.viewer_model_xz, hh.viewer_model_yz = (
        _VM(), _VM(), _VM())
    hh.subviewer_manager = _SubMgr(hh)
    hh.drawing_handler = None
    main_copies = {}
    from napari.layers import Labels

    for orient, vm in (("XY", hh.viewer_model_xy), ("XZ", hh.viewer_model_xz),
                       ("YZ", hh.viewer_model_yz)):
        c = Labels(transpose_data(vol, orient), name=f"L_{orient}")
        vm.layers.append(c)
        hh.subviewer_manager.copies[("L", orient)] = c
        main_copies[orient] = c
    hh.main_copies = main_copies
    hh.adv.preset = "cells"
    hh.place_box((slice(2, 18), slice(2, 28), slice(2, 28)))
    hh._add_place_markers_on = True
    # a second click here is a separate click, not a double click's second
    # press (C-L3 quietly drops those; test_add_audit_fixes_cd covers it)
    hh._add_double_click_s = lambda: 0.0
    _ViewerLayers.flag_at_append = []
    return hh


def test_a_click_on_empty_space_places_a_marker_and_a_ball(h):
    assert h._add_marker_click((12, 20, 20))
    assert marker_zyx(h) == [(12, 20, 20)]
    layer = h._add_marker_layer
    # The ball's value draws the colour of the id it
    # reserved (the volume's max is 7, so the first marker is object 8)
    assert layer is not None and layer.data[12, 20, 20] == AP.ball_value(8)
    assert layer.data.dtype == np.uint8
    assert h.adv.last("click_count") == 1


def test_a_click_on_a_kept_object_is_refused_with_its_id(h):
    assert not h._add_marker_click((6, 6, 6))
    assert h._add_transient == AP.ADD_S1_CLICK_ON_KEPT.format(id=7)
    assert h._add_click_markers == []


def test_a_click_outside_the_box_is_refused(h):
    assert not h._add_marker_click((0, 0, 0))
    assert h._add_transient == AP.ADD_S1_CLICK_OUTSIDE


def test_clicks_count_only_while_place_markers_is_on(h):
    h._add_place_markers_on = False
    assert not h._add_marker_click((12, 20, 20))
    assert h._add_click_markers == []


def test_a_click_is_refused_while_busy(h):
    h._add_busy = True
    assert not h._add_marker_click((12, 20, 20))
    assert h._add_transient == AP.ADD_S1_BUSY_REFUSED


def test_ctrl_click_removes_the_nearest_ball_containing_the_point(h):
    h._add_marker_click((12, 20, 20))
    h._add_marker_click((12, 10, 20))
    assert not h._add_marker_click((12, 15, 15), ctrl=True)   # in no ball
    assert h._add_marker_click((12, 21, 20), ctrl=True)
    assert marker_zyx(h) == [(12, 10, 20)]


def test_undo_last_and_clear(h):
    h._add_marker_click((12, 20, 20))
    h._add_marker_click((12, 10, 20))
    h._add_click_undo()
    assert marker_zyx(h) == [(12, 20, 20)]
    h._add_click_clear()
    assert h._add_click_markers == []
    assert not h._add_marker_layer.data.any()


def test_a_box_move_keeps_inside_clicks_and_drops_the_others(h):
    h._add_marker_click((12, 20, 20))
    h._add_marker_click((12, 5, 20))
    h._on_add_box_changed(12, 20, 20, 10, 10, 10)
    assert marker_zyx(h) == [(12, 20, 20)]
    assert h._add_transient == AP.ADD_S1_CLICKS_DROPPED_ONE


def test_the_ball_layer_is_flagged_before_it_is_appended(h):
    h._add_marker_click((12, 20, 20))
    assert _ViewerLayers.flag_at_append == [True]
    assert h._add_marker_layer.editable is False


def test_the_side_copies_are_views_of_the_main_array(h):
    h._add_marker_click((12, 20, 20))
    for orient in ("XY", "XZ", "YZ"):
        sub = h.subviewer_manager.get_subviewer_layer(
            h._ADD_BALL_LAYER, orient)
        assert np.shares_memory(sub.data, h._add_marker_layer.data)


def test_the_side_views_active_layer_is_restored(h):
    h._add_marker_click((12, 20, 20))
    assert h.viewer_model_xy.layers.selection.active is h.main_copies["XY"]
    assert h.viewer_model_xz.layers.selection.active is h.main_copies["XZ"]
    assert h.viewer_model_yz.layers.selection.active is h.main_copies["YZ"]


def test_balls_hidden_on_nuclei_and_shown_again_on_cells(h):
    h._add_marker_click((12, 20, 20))
    h.adv.preset = "nuclei"
    h._on_add_preset_changed("nuclei")
    assert h._add_marker_layer.visible is False
    assert marker_zyx(h) == [(12, 20, 20)], "clicks are kept"
    h.adv.preset = "cells"
    h._on_add_preset_changed("cells")
    assert h._add_marker_layer.visible is True


@pytest.mark.parametrize("change", ["preset", "source", "tab"])
def test_place_markers_is_unchecked(h, change):
    if change == "preset":
        h._on_add_preset_changed("cells")
    elif change == "source":
        h._on_add_source_changed(("paint", None))
    else:
        h._refresh_all_image_combos = lambda: None
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        Curation3DAnnotator._on_tool_tab_changed(h)
    assert h._add_place_markers_on is False
    assert h.adv.last("place_markers_on") is False


def test_no_click_acts_off_the_add_tab(h, monkeypatch):
    monkeypatch.setattr(type(h), "_add_tab_current", lambda self: False)
    assert not h._add_marker_click((12, 20, 20))


def test_clicks_reach_the_preview_box_local(h):
    h._add_marker_click((12, 20, 20))
    h._add_info = AP.AddBoxInfo(empty_share=1.0)
    h.adv.params["dark_outside"] = False
    h.adv.wall["channel"] = "mem"
    from napari.layers import Image

    h.viewer.layers.append(Image(np.random.rand(20, 30, 30), name="mem"))
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert kw["click_points"] == [(12, 20, 20)]


# --------------------------------------------------------------------------- #
# A5 -- reserved ids, colours, ticks
# --------------------------------------------------------------------------- #
A, B, C = (12, 20, 20), (12, 10, 20), (12, 20, 10)


def _ids(h):
    return [m["id"] for m in h._add_click_markers]


def _place(h, *points):
    for p in points:
        assert h._add_marker_click(p), p


def test_each_marker_reserves_the_next_id(h):
    _place(h, A, B, C)
    assert _ids(h) == [8, 9, 10]          # the volume's max is 7
    assert all(m["ticked"] for m in h._add_click_markers)
    assert h.adv.last("click_markers") == [(8, A, True), (9, B, True),
                                           (10, C, True)]


def test_the_id_floor_is_honoured(h):
    h._get_label_id_floor_cb = lambda: 400
    _place(h, A)
    assert _ids(h) == [401]


def test_a_middle_removal_leaves_a_gap_and_the_top_id_is_freed(h):
    _place(h, A, B, C)
    assert h._add_marker_click(B, ctrl=True)
    assert _ids(h) == [8, 10], "a gap; #3 keeps its id (its colour)"
    _place(h, (12, 10, 5))
    assert _ids(h) == [8, 10, 11]
    h._add_click_undo()
    assert _ids(h) == [8, 10]
    assert h._add_marker_click(C, ctrl=True)      # the top one
    _place(h, (12, 10, 5))
    assert _ids(h) == [8, 9], "removing the last marker frees the top id"
    h._add_click_clear()
    _place(h, B)
    assert _ids(h) == [8], "Clear resets"


def test_each_ball_draws_the_colour_of_its_id(h):
    h._get_label_id_floor_cb = lambda: 60
    _place(h, A, B)
    data = h._add_marker_layer.data
    assert data[A] == AP.ball_value(61) == 12
    assert data[B] == AP.ball_value(62) == 13


def test_an_unticked_ball_is_not_drawn_and_nothing_else_is_rewritten(h):
    """Invariant 23: a tick change rewrites only the affected balls."""
    _place(h, A, B, C)
    data = h._add_marker_layer.data
    data[2, 2, 2] = 99                         # a sentinel no ball covers
    h._on_add_click_tick(9, False)
    assert data[B] == 0 and not (data == AP.ball_value(9)).any()
    assert data[A] == AP.ball_value(8) and data[C] == AP.ball_value(10)
    assert data[2, 2, 2] == 99, "a full-volume rewrite would have zeroed it"
    h._on_add_click_tick(9, True)
    assert data[B] == AP.ball_value(9)
    assert data[2, 2, 2] == 99


def test_an_unticked_marker_keeps_its_id_and_the_others_their_colours(h):
    _place(h, A, B, C)
    h._on_add_click_tick(9, False)
    assert [(m["id"], m["ticked"]) for m in h._add_click_markers] == [
        (8, True), (9, False), (10, True)]
    assert h.adv.last("click_markers")[1] == (9, B, False)


def test_ctrl_click_hits_drawn_balls_only(h):
    _place(h, A)
    h._on_add_click_tick(8, False)
    assert not h._add_marker_click(A, ctrl=True)
    assert _ids(h) == [8]


def test_a_click_inside_any_marker_ball_is_refused_as_the_same_spot(h):
    _place(h, A)
    assert not h._add_marker_click((12, 21, 20))
    assert h._add_transient == AP.ADD_S1_CLICK_SAME_SPOT.format(id=8)
    h._on_add_click_tick(8, False)             # unticked still counts
    assert not h._add_marker_click(A)
    assert _ids(h) == [8]


def test_ids_rebase_when_next_label_moves(h, capsys):
    _place(h, A, B)
    h._add_click_undo()
    _place(h, C)                                # ids 8, 9 (C took 9)
    h._on_add_click_tick(8, False)
    h._get_label_id_floor_cb = lambda: 20       # next_label 8 -> 21
    _place(h, (12, 10, 5))
    assert [(m["id"], m["ticked"]) for m in h._add_click_markers] == [
        (21, False), (22, True), (23, True)], "shifted, ticks kept"
    assert "[ADD] marker ids re-based" in capsys.readouterr().out
    data = h._add_marker_layer.data
    assert data[C] == AP.ball_value(22) and data[A] == 0


def _cells_ready(h):
    from napari.layers import Image

    h._add_info = AP.AddBoxInfo(empty_share=1.0)
    h.adv.params["dark_outside"] = False
    h.adv.wall["channel"] = "mem"
    h.viewer.layers.append(Image(np.random.rand(20, 30, 30), name="mem"))


def test_the_preview_gets_only_the_ticked_points_and_their_ids(h):
    _place(h, A, B, C)
    h._on_add_click_tick(9, False)
    _cells_ready(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert kw["click_points"] == [A, C]
    assert kw["reserved_ids"] == [8, 10]


def test_the_preview_rebases_the_ids_first(h, capsys):
    _place(h, A, C)
    h._get_label_id_floor_cb = lambda: 30
    _cells_ready(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert kw["reserved_ids"] == [31, 32] == _ids(h)
    assert kw["next_label"] == 31
    assert "[ADD] marker ids re-based" in capsys.readouterr().out
    assert h._add_marker_layer.data[C] == AP.ball_value(32)


def test_ticks_follow_ids_through_remove_undo_and_a_box_move(h):
    """Invariant 21."""
    _place(h, A, B, C)
    h._on_add_click_tick(10, False)
    assert h._add_marker_click(A, ctrl=True)
    assert [(m["id"], m["ticked"]) for m in h._add_click_markers] == [
        (9, True), (10, False)]
    h._on_add_box_changed(12, 20, 12, 10, 10, 10)     # keeps C only
    assert [(m["id"], m["ticked"]) for m in h._add_click_markers] == [
        (10, False)]
    assert h._add_transient == AP.ADD_S1_CLICKS_DROPPED_ONE


def test_a_box_move_counts_the_unticked_markers_it_drops(h):
    _place(h, A, B, C)
    h._on_add_click_tick(9, False)
    h._on_add_box_changed(12, 20, 20, 6, 6, 6)       # keeps A only
    assert _ids(h) == [8]
    assert h._add_transient == AP.ADD_S1_CLICKS_DROPPED.format(n=2)


def test_s1_counts_the_ticked_markers_and_blocks_on_none(h):
    _cells_ready(h)
    _place(h, A, B)
    h._on_add_click_tick(9, False)
    assert "1 of 2 ticked" in h.adv.last("status_text")
    assert h.adv.last("click_markers")[1][2] is False
    h._on_add_click_tick(8, False)
    assert h.adv.last("status_text") == AP.tick_one_marker_line(2)
    assert h.adv.last("status_kind") == "warn"
    assert h.adv.last("preview_enabled") is False


def test_a_tick_change_bumps_the_epoch_and_drops_the_preview(h):
    _place(h, A, B)
    e0 = h._add_markers_epoch
    h._add_preview_result = np.zeros((2, 2, 2), np.int32)
    h._on_add_click_tick(9, False)
    assert h._add_markers_epoch == e0 + 1
    assert h._add_preview_result is None
    h._on_add_click_tick(9, False)             # no change, no bump
    assert h._add_markers_epoch == e0 + 1


@pytest.mark.parametrize("change", [
    lambda h: h._add_marker_click(C),
    lambda h: h._add_marker_click(A, ctrl=True),
    lambda h: h._add_click_undo(),
    lambda h: h._add_click_clear(),
    lambda h: h._on_add_click_tick(8, False),
    lambda h: h._on_add_box_changed(12, 20, 12, 10, 10, 10),
])
def test_every_marker_change_bumps_the_epoch(h, change):
    _place(h, A, B)
    e0 = h._add_markers_epoch
    change(h)
    assert h._add_markers_epoch > e0


def test_a_bare_tuple_is_normalised_into_a_record(h):
    h._add_click_markers = [A, B]
    assert h._add_markers() == [{"zyx": A, "id": 8, "ticked": True},
                                {"zyx": B, "id": 9, "ticked": True}]

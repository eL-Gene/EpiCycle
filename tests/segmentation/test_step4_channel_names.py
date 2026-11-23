"""The channel names ② shows.

History: owner live test 2026-09-29 ("the channel names should sync to the
drop downs, so we know what the nuclei / cell membrane channels are") put
the model's TRAINING-time names first -- another dataset's names
(WholeScale_ch0 on any stack).  Owner live test 2026-10-05 (BUG B): "it is
just stuck if you load or select new data / preview new data; the dropdown
names don't really change".

Owner-approved rules:

* The training-time names are gone from the items.
* The boxes count as Cellpose and Steps 2 and 3 do (2026-10-27, "keep it
  consistent with what step3/step2 does"): the first box offers
  "0: gray" then one item per channel, the second "0: none" then the
  same, and an item's index is its Cellpose code.
* With no stack's layers to name them, an item is Step 3's own word for
  the code ("1: red", "2: green", "3: blue").
* When the stack loaded in ③ -- or else the stack PREVIEWED with View or
  the eye -- has channel layers in the viewer, their layer names are used
  after the code: "{code}: {name}" (code k = the stack's channel k - 1).
  The names always belong to ONE stack (never a previewed stack's mixed
  with the loaded one's), and follow a rename.
* The model's role for a channel ("segment" / "helper") is the item's
  TOOLTIP, never its text: in the text it was cut off by the narrow field.
* The Model card names each trained channel by the item its box shows for
  it ("segment 2: green, helper 1: red", Step 3's words), and follows a
  load or a rename.
* The chosen channel never changes on a refresh.

A Step 2 round model still reports its training CHANNELS (codes) from
``{model}_training_config.json`` (the model card's "Trained on ..." and the
pre-pick).  Fixtures are shaped like the owner's ``artifacts/step2/runs/
Cell/lateral/round_01/model/`` (read on 2026-09-29, never written).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump, tick


def _step2_round(root, bucket="lateral", rnd=1, channels=(2, 1),
                 used=("2: WholeScale_ch1", "1: WholeScale_ch0")):
    name = f"Cell_{bucket}_round_{rnd:02d}"
    d = (root / "artifacts" / "step2" / "runs" / "Cell" / bucket
         / f"round_{rnd:02d}" / "model")
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(b"weights")
    (d / f"{name}_training_config.json").write_text(json.dumps({
        "model_name": name,
        "hyperparameters": {"channels": list(channels)}}), encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "bucket": bucket,
        "training": {"channels_used": list(used)}}), encoding="utf-8")
    return d / name


def test_a_step2_round_model_reports_its_training_channels(tmp_path):
    w = _step2_round(tmp_path)
    assert M.read_trained_channels(w) == [2, 1]


def test_the_training_time_names_are_no_longer_read():
    """Decision 58: the dead reader is gone."""
    assert not hasattr(M, "read_trained_channel_names")


def test_nothing_recorded_stays_unknown(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "weights").write_bytes(b"w")
    assert M.read_trained_channels(d / "weights") is None


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"), channels=2)
    w = panel(qapp, ws)
    tick(w, qapp, "P1.tif")
    lat = _step2_round(ws, "lateral")
    ax = _step2_round(ws, "axial")
    w.controller._set_active_model({
        "lateral": str(lat), "axial": str(ax), "kind": "step2_run",
        "lateral_label": "Cell · round_01", "axial_label": "Cell · round_01"})
    pump(qapp)
    w._ws = ws
    yield w
    w.controller.viewer = None
    close(w)


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _tips(combo):
    from qtpy.QtCore import Qt
    return [combo.itemData(i, Qt.ToolTipRole) for i in range(combo.count())]


def _role_only():
    """The items with nothing shown: the code words of Step 3 (the roles
    are in the tooltips)."""
    return ["0: gray", "1: red", "2: green"]


def test_with_nothing_shown_the_items_carry_the_models_roles_as_tips(qapp, w):
    assert _items(w.chan1_combo) == _role_only()
    # exactly Step 3's own items, for the codes this stack offers
    from napari_epicycle.ui_shell import _recon_settings_text as RT
    assert _items(w.chan1_combo) == list(RT.SEG_CODE_ITEMS_SEGMENT[:3])
    assert _items(w.chan2_combo) == list(RT.SEG_CODE_ITEMS_HELPER[:3])
    # the role is the item's tooltip, never its text; "0: gray" has none
    assert _tips(w.chan1_combo) == [None, T.SEG_ROLE_TIP_HELPER,
                                    T.SEG_ROLE_TIP_SEGMENT]
    assert _tips(w.chan1_combo)[1:] == [
        "This model was trained with this channel as its helper.",
        "This model was trained to segment this channel."]
    # chan2 has "0: none" first, with no tip
    assert _items(w.chan2_combo) == [T.SEG_CHANNEL_NONE, "1: red", "2: green"]
    assert T.SEG_CHANNEL_NONE == "0: none"
    assert _tips(w.chan2_combo) == [None, T.SEG_ROLE_TIP_HELPER,
                                    T.SEG_ROLE_TIP_SEGMENT]
    # the model's codes are pre-picked as they are: segment 2, helper 1
    assert w.chosen_channels() == (2, 1)
    # the card names each channel by the item its box shows: Step 3's words
    assert w.model_trained_label.text() == (
        "Trained on segment 2: green, helper 1: red.")


def test_a_channel_with_no_role_reads_its_code(qapp, w):
    w.controller._trained_codes = None
    w.controller._refresh_channel_combos()
    assert _items(w.chan1_combo) == ["0: gray", "1: red", "2: green"]
    assert _tips(w.chan1_combo) == [None, None, None]


def _viewer(c):
    from napari.components import ViewerModel
    c.viewer = ViewerModel()
    return c.viewer


def _load(c, name="P1.tif"):
    lab = np.zeros((4, 8, 8), np.int32)
    raw = {"data": np.zeros((2, 4, 8, 8), np.uint8), "ch_axis": 0,
           "n_channels": 2, "axes": "CZYX"}
    c._apply_loaded(name, {"labels": lab, "n_cells": 0, "backup": "",
                           "raw": raw, "source_path": ""})


def test_a_loaded_stack_names_the_items_and_a_rename_follows(qapp, w):
    c = w.controller
    _viewer(c)
    _load(c)
    pump(qapp)
    chans = [x for x in c._stack_layers
             if getattr(x, "_epicycle_channel_index", None) is not None]
    assert len(chans) == 2
    # Step 4's own default layer names ("step4 · ch0") name nothing: the
    # items keep the code words, never "1: ch0" (a position beside a code)
    assert [str(x.name) for x in chans] == ["step4 · ch0", "step4 · ch1"]
    assert _items(w.chan1_combo) == _role_only()
    assert _items(w.chan2_combo) == ["0: none", "1: red", "2: green"]
    assert _tips(w.chan1_combo)[1] == T.SEG_ROLE_TIP_HELPER
    assert w.model_trained_label.text() == (
        "Trained on segment 2: green, helper 1: red.")
    chans[1].name = "step4 · Actin"
    pump(qapp)
    assert w.chan1_combo.itemText(2) == T.SEG_CHANNEL_ITEM.format(
        code=2, name="Actin")
    assert w.chan1_combo.itemText(2) == "2: Actin"
    assert _tips(w.chan1_combo)[2] == T.SEG_ROLE_TIP_SEGMENT
    assert w.chan1_combo.itemText(1) == "1: red"   # still the default name
    assert w.chosen_channels() == (2, 1)
    # the card follows the rename: it names the channel as its box does
    assert w.model_trained_label.text() == (
        "Trained on segment 2: Actin, helper 1: red.")


def _write_source(w, name):
    src = w._ws / "src" / name
    src.parent.mkdir(exist_ok=True)
    tifffile.imwrite(str(src), np.zeros((2, 4, 8, 8), np.uint8),
                     metadata={"axes": "CZYX"})
    return src


def test_a_preview_names_the_items_and_its_removal_restores_them(qapp, w):
    c = w.controller
    viewer = _viewer(c)
    src = _write_source(w, "P2.tif")
    c._preview_selected_stack("P2.tif", {"source_stack_path": str(src)})
    pump(qapp)
    # a preview's default layer names ("step4 · P2 ch0 [preview]") name
    # nothing either: the code words stay
    previews = [x for x in viewer.layers
                if getattr(x, "_epicycle_channel_index", None) is not None]
    assert previews and all("P2 ch" in str(x.name) for x in previews)
    assert _items(w.chan1_combo) == _role_only()
    assert _tips(w.chan1_combo)[1] == T.SEG_ROLE_TIP_HELPER
    [x for x in previews if x._epicycle_channel_index == 1][0].name = (
        "step4 · Ecad [preview]")
    pump(qapp)
    items = _items(w.chan1_combo)
    assert items[2] == "2: Ecad" and "(" not in items[2]
    assert items[1] == "1: red"
    c._on_stack_eye("P2.tif")                      # the eye hides it
    pump(qapp)
    assert _items(w.chan1_combo) == _role_only()
    assert w.chosen_channels() == (2, 1)


def test_the_loaded_stack_wins_over_a_preview(qapp, w):
    """Never a previewed stack's names mixed with the loaded one's."""
    c = w.controller
    viewer = _viewer(c)
    _load(c, "P1.tif")
    [x for x in c._stack_layers
     if getattr(x, "_epicycle_channel_index", None) == 0][0].name = \
        "step4 · DAPI"
    src = _write_source(w, "P2.tif")
    c._preview_selected_stack("P2.tif", {"source_stack_path": str(src)})
    pump(qapp)
    items = _items(w.chan1_combo)
    assert items[1] == T.SEG_CHANNEL_ITEM.format(code=1, name="DAPI")
    assert items[1] == "1: DAPI"
    assert _tips(w.chan1_combo)[1] == T.SEG_ROLE_TIP_HELPER
    assert "P2" not in items[2]
    assert len(viewer.layers) >= 5


def test_a_blank_segment_pick_stays_blank_with_names(qapp, w):
    c = w.controller
    w.set_channel_choice(w.chan1_combo, 5)      # a channel that is gone
    c._refresh_channel_combos()
    c._refresh_channel_combos()
    assert w.chan1_combo.currentIndex() == -1


def test_a_one_channel_stack_offers_gray_and_its_one_channel(qapp, tmp_path):
    """A one-channel (ZYX) stack: the boxes offer "0: gray" / "0: none" and
    code 1, in Step 3's words, and a model trained on that channel says so
    on the card and in the item's tooltip; its code 1 is pre-picked."""
    ws = make_workspace(tmp_path, patches=("P1.tif",), channels=1)
    w = panel(qapp, ws)
    try:
        tick(w, qapp, "P1.tif")
        lat = _step2_round(ws, "lateral", channels=(1, 0),
                           used=("1: WholeScale_ch0",))
        ax = _step2_round(ws, "axial", channels=(1, 0),
                          used=("1: WholeScale_ch0",))
        w.controller._set_active_model({
            "lateral": str(lat), "axial": str(ax), "kind": "step2_run",
            "lateral_label": "Cell · round_01",
            "axial_label": "Cell · round_01"})
        pump(qapp)
        assert _items(w.chan1_combo) == ["0: gray", "1: red"]
        assert _items(w.chan2_combo) == ["0: none", "1: red"]
        assert _tips(w.chan1_combo) == [None, T.SEG_ROLE_TIP_SEGMENT]
        assert w.chosen_channels() == (1, 0)
        assert w.model_trained_label.text() == "Trained on segment 1: red."
        assert "channels" not in w.model_trained_label.text()
    finally:
        w.controller.viewer = None
        close(w)

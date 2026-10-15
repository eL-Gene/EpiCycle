"""The channel names ② shows.

History: owner live test 2026-09-29 ("the channel names should sync to the
drop downs, so we know what the nuclei / cell membrane channels are") put
the model's TRAINING-time names first -- another dataset's names
(WholeScale_ch0 on any stack).  Owner live test 2026-10-05 (BUG B): "it is
just stuck if you load or select new data / preview new data; the dropdown
names don't really change".

Owner-approved rules:

* The training-time names are gone from the items.
* With no stack's layers to name them, an item is the index plus the
  model's role when the model records one: ``SEG_CHANNEL_ITEM_ROLE_ONLY``
  "{i}: {role} (model)"; a channel with no role reads ``SEG_CHANNEL_ITEM``
  "{i}: ch{i}".
* When the stack loaded in ③ -- or else the stack PREVIEWED with View or
  the eye -- has channel layers in the viewer, their layer names are used:
  "{i}: {name} ({role})".  The names always belong to ONE stack (never a
  previewed stack's mixed with the loaded one's), and follow a rename.
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


def _role_only():
    return [T.SEG_CHANNEL_ITEM_ROLE_ONLY.format(i=0,
                                                role=T.SEG_MODEL_ROLE_NUCLEI),
            T.SEG_CHANNEL_ITEM_ROLE_ONLY.format(i=1,
                                                role=T.SEG_MODEL_ROLE_MEMBRANE)]


def test_with_nothing_shown_the_items_are_the_models_roles(qapp, w):
    assert _items(w.chan1_combo) == _role_only()
    assert _items(w.chan1_combo)[0] == "0: nuclei (model)"
    # the model's channels are pre-picked: membrane 1, nuclei 0
    assert w.chosen_channels() == (1, 0)
    assert w.model_trained_label.text() != T.SEG_MODEL_TRAINED_UNKNOWN


def test_a_channel_with_no_role_reads_its_number(qapp, w):
    w.controller._trained_codes = None
    w.controller._refresh_channel_combos()
    assert _items(w.chan1_combo) == [
        T.SEG_CHANNEL_ITEM.format(i=0, name=T.SEG_CHANNEL_NAME.format(i=0)),
        T.SEG_CHANNEL_ITEM.format(i=1, name=T.SEG_CHANNEL_NAME.format(i=1))]


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
    assert w.chan1_combo.itemText(0) == T.SEG_CHANNEL_ITEM_ROLE.format(
        i=0, name="ch0", role=T.SEG_MODEL_ROLE_NUCLEI)
    chans[1].name = "step4 · Actin"
    pump(qapp)
    assert w.chan1_combo.itemText(1) == T.SEG_CHANNEL_ITEM_ROLE.format(
        i=1, name="Actin", role=T.SEG_MODEL_ROLE_MEMBRANE)
    assert w.chosen_channels() == (1, 0)


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
    items = _items(w.chan1_combo)
    assert items != _role_only()
    assert "P2" in items[0] and "(nuclei)" in items[0]
    previews = [x for x in viewer.layers
                if getattr(x, "_epicycle_channel_index", None) == 1]
    previews[0].name = "step4 · Ecad [preview]"
    pump(qapp)
    assert w.chan1_combo.itemText(1).startswith("1: Ecad")
    c._on_stack_eye("P2.tif")                      # the eye hides it
    pump(qapp)
    assert _items(w.chan1_combo) == _role_only()
    assert w.chosen_channels() == (1, 0)


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
    assert items[0] == T.SEG_CHANNEL_ITEM_ROLE.format(
        i=0, name="DAPI", role=T.SEG_MODEL_ROLE_NUCLEI)
    assert "P2" not in items[1]
    assert len(viewer.layers) >= 5


def test_a_blank_membrane_pick_stays_blank_with_names(qapp, w):
    c = w.controller
    w.set_channel_choice(w.chan1_combo, 5)      # a channel that is gone
    c._refresh_channel_combos()
    c._refresh_channel_combos()
    assert w.chan1_combo.currentIndex() == -1

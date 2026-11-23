"""Step 4's channel boxes say a model's roles in TOOLTIPS, and count by
Cellpose code.

The role a model records for a channel ("segment" / "helper") is the
item's tooltip, never its text: in the text it was cut off by the narrow
field.  The boxes count as Steps 2 and 3 do: the first offers "0: gray"
then one item per channel, the second "0: none" then the same, and an
item's index IS the Cellpose code.  For one to three channels the model's
codes are pre-picked as they are, ``chosen_channels`` reads the codes, and
the codes the run saves are the picked codes, the same numbers the old
position boxes saved for the same channel (position p was saved as
p + 1).  A gray model (first code 0) was trained on no one channel, so no
item carries a role tooltip and the card names none.
"""
from __future__ import annotations

import json

import pytest
from qtpy.QtCore import Qt

from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.ui_shell import _recon_settings_text as RT

from ._step4_panel import close, make_workspace, panel, pump, tick

#: (channel count, the model's Cellpose codes)
CASES = [(1, (1, 0)), (2, (2, 1)), (3, (3, 1))]


def _model(root, bucket, channels):
    d = root / "models" / f"{bucket}_{channels[0]}{channels[1]}"
    d.mkdir(parents=True, exist_ok=True)
    w = d / f"Cell_{bucket}_round_01"
    w.write_bytes(b"weights")
    (d / f"{w.name}_training_config.json").write_text(json.dumps({
        "hyperparameters": {"channels": list(channels)}}), encoding="utf-8")
    return w


def _panel(qapp, tmp_path, n, codes):
    ws = make_workspace(tmp_path, patches=("P1.tif",), channels=n)
    w = panel(qapp, ws)
    tick(w, qapp, "P1.tif")
    w.controller._set_active_model({
        "lateral": str(_model(ws, "lateral", codes)),
        "axial": str(_model(ws, "axial", codes)),
        "kind": "step2_run",
        "lateral_label": "Cell · round_01", "axial_label": "Cell · round_01"})
    pump(qapp)
    return w


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _tips(combo):
    return [combo.itemData(i, Qt.ToolTipRole) for i in range(combo.count())]


def _plain(n):
    """Step 3's own items for codes 0 .. n (no stack names a channel)."""
    return list(RT.SEG_CODE_ITEMS_SEGMENT[:n + 1])


@pytest.mark.parametrize("n,codes", CASES)
def test_the_items_are_codes_and_the_roles_are_tooltips(qapp, tmp_path,
                                                         n, codes):
    w = _panel(qapp, tmp_path, n, codes)
    try:
        # "0: gray" + one item per channel; chan2 "0: none" first
        assert _items(w.chan1_combo) == _plain(n)
        assert _items(w.chan2_combo) == list(RT.SEG_CODE_ITEMS_HELPER[:n + 1])
        # the roles are tooltips on the trained codes' items only
        want = [None] * (n + 1)
        want[codes[0]] = T.SEG_ROLE_TIP_SEGMENT
        if codes[1]:
            want[codes[1]] = T.SEG_ROLE_TIP_HELPER
        assert _tips(w.chan1_combo) == want
        assert _tips(w.chan2_combo) == want
        assert T.SEG_ROLE_TIP_SEGMENT == (
            "This model was trained to segment this channel.")
        assert T.SEG_ROLE_TIP_HELPER == (
            "This model was trained with this channel as its helper.")
        # the model's codes are pre-picked as they are: index == code
        assert w.chosen_channels() == codes
        assert w.chan1_combo.currentIndex() == codes[0]
        assert w.chan2_combo.currentIndex() == codes[1]
    finally:
        w.controller.viewer = None
        close(w)


@pytest.mark.parametrize("n,codes", CASES)
def test_every_pick_saves_its_code_the_same_number_as_before(qapp, tmp_path,
                                                             n, codes):
    w = _panel(qapp, tmp_path, n, codes)
    c = w.controller
    try:
        for c1 in range(n + 1):
            for c2 in range(n + 1):
                w.set_channel_choice(w.chan1_combo, c1)
                w.set_channel_choice(w.chan2_combo, c2)
                c._refresh_channel_combos()           # a rebuild keeps it
                assert w.chosen_channels() == (c1, c2)
                cfg = c._collect_seg_config()
                assert list(cfg.channels) == [c1, c2], (c1, c2)
        # the old position boxes saved position p as p + 1 and "none" as
        # 0: every channel the old boxes offered saves the same code now
        for p in range(n):
            for q in [None] + list(range(n)):
                old = [p + 1, 0 if q is None else q + 1]
                w.set_channel_choice(w.chan1_combo, p + 1)
                w.set_channel_choice(w.chan2_combo,
                                     0 if q is None else q + 1)
                assert list(c._collect_seg_config().channels) == old
    finally:
        w.controller.viewer = None
        close(w)


def test_a_gray_model_names_no_channel_and_picks_gray(qapp, tmp_path):
    w = _panel(qapp, tmp_path, 2, (0, 2))
    try:
        assert _items(w.chan1_combo) == _plain(2)
        assert _tips(w.chan1_combo) == [None, None, None]
        assert _tips(w.chan2_combo) == [None, None, None]
        assert w.model_trained_label.text() == (
            "Trained on one grayscale image (all channels averaged).")
        # the model's own codes: gray, as it was trained
        assert w.chosen_channels() == (0, 2)
        assert w.chan1_combo.currentText() == T.SEG_CHANNEL_GRAY == "0: gray"
        assert list(w.controller._collect_seg_config().channels) == [0, 2]
    finally:
        w.controller.viewer = None
        close(w)

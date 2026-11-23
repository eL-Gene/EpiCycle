"""Step 4's two channel boxes count by Cellpose code, like Steps 2 and 3.

Owner, 2026-10-27: "keep it consistent with what step3/step2 does".

* The first box offers "0: gray" (every channel averaged), the second
  "0: none", then one item per channel of the largest ticked stack; in both
  boxes an item's index IS the Cellpose code (code k = the stack's channel
  k - 1).  With no stack shown an item is Step 3's own word for the code
  ("1: red", "2: green", "3: blue"), past the third "4: ch3"; a shown
  stack's layer names follow the code ("1: DAPI").
* Predict passes the picked codes as they are, so the same picks give
  Cellpose the same two numbers in Step 3 and in Step 4, and the "Step 3
  run" preset hands Step 4 the run's own codes.
* The Model card names a model's channels in Step 3's card's words.
* Nothing saved changes meaning.  Step 4 never saved a box's position:
  what it saves is the Cellpose list it ran with (``predicted_with.seg``
  in the state file, ``channels`` in the cache meta), which the old
  position boxes already converted to codes.  So an old state file is read
  as it is, nothing is shifted on load or on save, and no marker is needed.
* A one-channel (ZYX) stack offers "0: gray" / "0: none" and code 1.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.ui_shell import _recon_settings_text as RT

from ._step4_panel import close, make_workspace, panel, pump, tick


# ---------------------------------------------------------------- Qt-free

@pytest.mark.parametrize("helper", [False, True])
def test_with_no_stack_shown_the_items_are_step3s_code_words(helper):
    words = RT.SEG_CODE_ITEMS_HELPER if helper else RT.SEG_CODE_ITEMS_SEGMENT
    for code, word in enumerate(words):
        assert M.channel_item(code, None, helper=helper) == word
    assert M.channel_item(4, None, helper=helper) == "4: ch3"
    assert T.SEG_CHANNEL_GRAY == "0: gray"
    assert T.SEG_CHANNEL_NONE == "0: none"


def test_a_shown_stacks_names_follow_the_code():
    names = ["DAPI", "Membrane", None]
    assert M.channel_item(0, names) == "0: gray"
    assert M.channel_item(0, names, helper=True) == "0: none"
    assert M.channel_item(1, names) == "1: DAPI"
    assert M.channel_item(2, names, helper=True) == "2: Membrane"
    assert M.channel_item(3, names) == "3: blue"     # a channel no layer names


@pytest.mark.parametrize("c1", range(4))
@pytest.mark.parametrize("c2", range(4))
def test_the_step4_card_says_what_step3s_card_says(c1, c2):
    """With no stack shown, Step 4's card names a model's channels in the
    words of Step 3's card, for every pair of codes."""
    from napari_epicycle.fusion_reconstruction._step3_rows import trained_line
    rec = {"kind": "refined", "trained_channels": [c1, c2], "run_id": None}
    step3, known = trained_line(rec, dict(rec))
    assert known
    assert M.trained_channels_text([c1, c2]) == step3


def test_the_card_example():
    assert M.trained_channels_text([2, 1]) == (
        "Trained on segment 2: green, helper 1: red.")
    assert M.trained_channels_text([2, 1], ["DAPI", "Membrane"]) == (
        "Trained on segment 2: Membrane, helper 1: DAPI.")
    assert M.trained_channels_text([1, 0]) == "Trained on segment 1: red."
    assert M.trained_channels_text(None) == T.SEG_MODEL_TRAINED_UNKNOWN
    assert M.trained_channels_text(["x"]) == T.SEG_MODEL_TRAINED_UNKNOWN


def test_one_rule_for_how_many_channels_a_pick_needs():
    """Code k needs k channels; gray needs one, whatever the helper (Cellpose
    averages every channel and ignores the helper under gray)."""
    assert M.channels_needed([0, 0]) == 1
    assert M.channels_needed([0, 2]) == 1
    assert M.channels_needed([0, 3]) == 1
    assert M.channels_needed([1, 0]) == 1
    assert M.channels_needed([2, 1]) == 2
    assert M.channels_needed([1, 3]) == 3
    assert M.channels_needed(None) is None


# --------------------------------------------- Step 3 and Step 4 agree (Qt)

@pytest.fixture
def step3_ui(qapp):
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    yield w
    w.deleteLater()


@pytest.fixture(params=[1, 2, 3], ids=["1ch", "2ch", "3ch"])
def step4(qapp, tmp_path, request):
    """The Step 4 panel on a stack of 1, 2 or 3 channels (the one-channel
    stack is the ZYX fixture)."""
    from tests.fusion_reconstruction._one_channel import write_zyx_patch
    n = request.param
    ws = make_workspace(tmp_path, patches=("P1.tif",), channels=max(n, 2))
    if n == 1:
        write_zyx_patch(ws / "artifacts" / "step1" / "patches", "P1.tif")
    w = panel(qapp, ws)
    w.controller._trained_codes = None          # a built-in model
    tick(w, qapp, "P1.tif")
    w.controller._refresh_channel_combos()
    pump(qapp)
    w._ws = ws
    w._n = n
    yield w
    w.controller.viewer = None
    close(w)


def _step3_channels(ui):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    return list(C._collect_seg_config(SimpleNamespace(ui=ui)).channels)


def _pick_text(combo, text):
    i = combo.findText(text)
    assert i >= 0, text
    combo.setCurrentIndex(i)


def test_step3_and_step4_give_cellpose_the_same_channels(qapp, step3_ui,
                                                         step4):
    """The same item picked in both steps' boxes gives Cellpose the same
    channel list, for a stack of one, two and three channels."""
    s3, w, n = step3_ui, step4, step4._n
    # Step 4 shows Step 3's own items, as far as the stack has channels
    assert [w.chan1_combo.itemText(i) for i in range(w.chan1_combo.count())
            ] == [s3.chan1_combo.itemText(i) for i in range(n + 1)]
    assert [w.chan2_combo.itemText(i) for i in range(w.chan2_combo.count())
            ] == [s3.chan2_combo.itemText(i) for i in range(n + 1)]
    seen = 0
    for t1 in RT.SEG_CODE_ITEMS_SEGMENT[:n + 1]:
        for t2 in RT.SEG_CODE_ITEMS_HELPER[:n + 1]:
            for ui in (s3, w):
                _pick_text(ui.chan1_combo, t1)
                _pick_text(ui.chan2_combo, t2)
            got4 = list(w.controller._collect_seg_config().channels)
            assert got4 == _step3_channels(s3), (t1, t2)
            assert w.chosen_channels() == tuple(got4)
            seen += 1
    assert seen == (n + 1) ** 2


def test_with_nothing_recorded_the_first_box_starts_blank_not_gray(
        qapp, step4):
    """A model that records no channels: the first box starts blank and
    Predict asks for a pick; "0: gray" is offered but never chosen for you,
    and the helper starts at "0: none"."""
    w = step4
    assert w.chan1_combo.currentIndex() == -1
    assert w.chan1_combo.currentText() == ""
    assert w.chan1_combo.itemText(0) == T.SEG_CHANNEL_GRAY
    assert w.chosen_channels() == (-1, 0)
    w.controller._refresh_channel_combos()       # a refresh keeps it blank
    assert w.chosen_channels() == (-1, 0)


def test_the_step3_preset_runs_with_the_step3_runs_codes(qapp, step4):
    """The "Step 3 run" preset hands Step 4 the run's Cellpose codes, and
    Predict passes them unchanged, gray included."""
    c = step4.controller
    n = step4._n
    for codes in ([2, 1], [1, 0], [3, 2], [0, 0], [0, 2]):
        if max(codes) > n:
            continue                    # a code this stack does not offer
        c._apply_step3_channels(codes)
        assert step4.chosen_channels() == tuple(codes)
        assert list(c._collect_seg_config().channels) == codes


# ------------------------------------------------- one channel (ZYX) (Qt)

def test_a_one_channel_stack_offers_gray_and_code_1(qapp, tmp_path):
    from tests.fusion_reconstruction._one_channel import write_zyx_patch
    ws = make_workspace(tmp_path, patches=("P1.tif",), channels=2)
    write_zyx_patch(ws / "artifacts" / "step1" / "patches", "P1.tif")
    w = panel(qapp, ws)
    try:
        w.controller._trained_codes = None
        tick(w, qapp, "P1.tif")
        w.controller._refresh_channel_combos()
        pump(qapp)
        items1 = [w.chan1_combo.itemText(i)
                  for i in range(w.chan1_combo.count())]
        items2 = [w.chan2_combo.itemText(i)
                  for i in range(w.chan2_combo.count())]
        assert items1 == ["0: gray", "1: red"]
        assert items2 == ["0: none", "1: red"]
        c = w.controller
        for c1 in (0, 1):
            w.set_channel_choice(w.chan1_combo, c1)
            w.set_channel_choice(w.chan2_combo, 0)
            assert c._channels_needed() == 1          # it can run
            assert list(c._collect_seg_config().channels) == [c1, 0]
    finally:
        w.controller.viewer = None
        close(w)


# ------------------------------------- what is saved does not move (no Qt)

def _old_state(ws):
    """A state file written before the boxes counted by code: Step 4 saved
    the Cellpose list it ran with ([2, 1]: segment the second channel,
    helper the first), never a box position."""
    data = {
        "schema_version": 2,
        "workspace_root": str(ws),
        "stacks": {"P1.tif": {
            "status": "predicted",
            "predicted_with": {"kind": "pretrained", "lateral": "cyto3",
                               "axial": "cyto3",
                               "seg": {"channels": [2, 1]}},
        }},
        "session": {"start_from": "epicycle", "ticked": ["P1.tif"]},
        "active_stack": None,
    }
    (ws / "step4_state.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def _saved_channels(ws):
    data = json.loads((ws / "step4_state.json").read_text(encoding="utf-8"))
    return data["stacks"]["P1.tif"]["predicted_with"]["seg"]["channels"]


def _keys(x, out=None):
    out = set() if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            out.add(k)
            _keys(v, out)
    elif isinstance(x, list):
        for v in x:
            _keys(v, out)
    return out


def test_an_old_state_file_is_read_as_it_is_and_never_shifted(tmp_path):
    from napari_epicycle.segmentation.state_manager import StackStateManager
    ws = tmp_path / "ws"
    ws.mkdir()
    _old_state(ws)
    for _ in range(3):                 # load + save, three times over
        sm = StackStateManager()
        sm.load(str(ws))
        rec = sm.get_record("P1.tif")
        assert rec["predicted_with"]["seg"]["channels"] == [2, 1]
        assert sm.save(str(ws))
        assert _saved_channels(ws) == [2, 1]
    data = json.loads((ws / "step4_state.json").read_text(encoding="utf-8"))
    # no box position and no marker was ever written
    assert not {k for k in _keys(data) if "chan" in k.lower()} - {"channels"}
    assert data["schema_version"] == 2


def test_a_panel_on_an_old_state_keeps_its_saved_codes(qapp, tmp_path):
    """Opening and saving the panel on an old state leaves its saved
    Cellpose list as it is, and the session gains no channel key."""
    ws = make_workspace(tmp_path, patches=("P1.tif",), channels=2)
    _old_state(ws)
    w = panel(qapp, ws)
    try:
        w.set_channel_choice(w.chan1_combo, 1)
        w.controller._save_session_metadata()
        pump(qapp)
    finally:
        w.controller.viewer = None
        close(w)
    assert _saved_channels(ws) == [2, 1]
    data = json.loads((ws / "step4_state.json").read_text(encoding="utf-8"))
    assert not [k for k in _keys(data["session"]) if "chan" in k.lower()]

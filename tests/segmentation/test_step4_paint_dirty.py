"""B2: a hand paint on the labels layer is unsaved work (the B1b hand-off;
9).

napari's ``paint`` event sets ``_labels_dirty``, holds the edit time until
Save, bumps the epoch, shows "Edited, not saved" in ③, makes the
measurements stale, and a load, a Browse or leaving Step 4 then asks first.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"))
    w = panel(qapp, ws)
    c = w.controller
    run = ws / "artifacts" / "step4" / "runs" / "r1"
    run.mkdir(parents=True)
    lab = np.zeros((4, 8, 8), np.uint32)
    lab[1, 1:3, 1:3] = 1
    for name in ("P1.tif", "P2.tif"):
        path = run / f"{name}_labels_3d.tif"
        tifffile.imwrite(str(path), lab)
        c.state_manager.set_stack_status(name, "reconstructed",
                                         labels_path=str(path))
    c._refresh_stacks()
    c._apply_loaded("P1.tif", {"labels": lab.copy(), "n_cells": 1,
                               "backup": "", "raw": None, "source_path": ""})
    pump(qapp)
    yield w
    close(w)


def test_a_paint_is_unsaved_work(qapp, w):
    c = w.controller
    assert not c._has_unsaved_changes
    epoch = c._labels_epoch
    c._on_labels_painted()
    assert c._labels_dirty is True
    assert c._has_unsaved_changes
    assert c._labels_epoch == epoch + 1
    assert c._pending_edits.get("labels_changed_at")
    assert c._pending_edits.get("_stack") == "P1.tif"
    pump(qapp)
    assert w.save_tiers_btn.isEnabled()               # something to save
    assert w.stack_card_pill.text() == T.MT_STACK_PILL_UNSAVED


def test_a_paint_then_a_load_asks_first(qapp, w):
    c = w.controller
    c._on_labels_painted()
    asked = []
    c._ask_unsaved = lambda stack: asked.append(stack) or "stay"
    c._request_load("P2.tif")
    assert asked == ["P1.tif"]
    assert c._current_stack_name == "P1.tif"          # Stay kept it


def test_a_paint_then_leaving_step4_asks(qapp, w, monkeypatch):
    c = w.controller
    c._on_labels_painted()
    asked = []
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm",
                        lambda *a, **k: asked.append(a[1]) or False)
    c.on_step_exit()
    assert asked == [T.STEP_EXIT_TITLE]


def test_a_paint_with_nothing_loaded_changes_nothing(qapp, w):
    c = w.controller
    c._labels_volume = None
    c._current_stack_name = None
    c._labels_dirty = False
    c._on_labels_painted()
    assert c._labels_dirty is False


def test_a_data_swap_is_not_a_paint(qapp, w):
    """``events.data`` (our own ``layer.data = …``) only bumps the epoch."""
    c = w.controller
    epoch = c._labels_epoch
    c._on_labels_layer_edited()
    assert c._labels_epoch == epoch + 1
    assert c._labels_dirty is False

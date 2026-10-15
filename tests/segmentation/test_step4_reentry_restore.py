"""Owner live test (2026-10-05), BUG A: "can't really load it even if you
click on the card inside the box list".

Trace (confirmed by these tests failing first): leaving Step 4 clears its
layers (F15); coming back re-added only the labels layer from memory -- the
image channels are not kept in memory, so they stayed gone -- and a click
on the loaded stack's ③ row returned at "is already loaded".  With one
stack there was no other row to force a reload.

Now, on re-entry and on a click of the loaded row, a stack whose layers are
missing is RESTORED: the labels layer from memory (unsaved edits kept, no
unsaved-work prompt), the image channels re-read on the ``_StackLoadWorker``
(kind "restore": no labels read, no backup) with ``_load_in_flight`` raised,
tagged with their channel index again.  A click with everything present
still does nothing.  A REAL napari ``ViewerModel``, ``LayerLifecycleManager``
and ``QThread``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from ._step4_panel import close, make_workspace, panel, pump

napari = pytest.importorskip("napari")


def _wait(qapp, cond, timeout=20.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.005)
    return cond()


@pytest.fixture
def setup(qapp, tmp_path, monkeypatch):
    from napari.components import ViewerModel

    from napari_epicycle.ui_shell import theme
    from napari_epicycle.ui_shell.layer_lifecycle_manager import (
        LayerLifecycleManager,
    )
    ws = make_workspace(tmp_path, patches=("P1.tif",), channels=2)
    w = panel(qapp, ws)
    c = w.controller
    viewer = ViewerModel()
    c.viewer = viewer
    c.layer_lifecycle_manager = LayerLifecycleManager(viewer)
    run = ws / "artifacts" / "step4" / "runs" / "r1"
    run.mkdir(parents=True)
    lab = np.zeros((6, 24, 24), np.uint32)
    lab[1:3, 2:5, 2:5] = 1
    tifffile.imwrite(str(run / "labels_3d.tif"), lab)
    c.state_manager.set_stack_status(
        "P1.tif", "reconstructed", labels_path=str(run / "labels_3d.tif"))
    c._refresh_stacks()
    pump(qapp)
    # leaving with unsaved work: "Leave Without Saving"
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: False)
    yield w, c, viewer
    c.viewer = None
    close(w)
    pump(qapp)
    for th in getattr(c, "_retired_threads", []):
        th.wait(3000)


def _load(qapp, c):
    c._request_load("P1.tif")
    assert _wait(qapp, lambda: not c._load_in_flight)
    assert c._current_stack_name == "P1.tif"


def _channel_layers(viewer):
    return [lyr for lyr in viewer.layers
            if getattr(lyr, "_epicycle_channel_index", None) is not None]


def _labels_layers(viewer):
    return [lyr for lyr in viewer.layers
            if lyr.__class__.__name__ == "Labels"]


def test_re_entry_brings_the_image_channels_back(qapp, setup):
    w, c, viewer = setup
    _load(qapp, c)
    assert len(_channel_layers(viewer)) == 2
    c.on_step_exit()
    assert len(viewer.layers) == 0
    c.on_step_enter()
    assert _wait(qapp, lambda: not c._load_in_flight)
    chans = _channel_layers(viewer)
    assert sorted(x._epicycle_channel_index for x in chans) == [0, 1]
    assert len(_labels_layers(viewer)) == 1


def test_a_click_on_the_loaded_row_restores_missing_layers(qapp, setup):
    w, c, viewer = setup
    _load(qapp, c)
    for lyr in list(viewer.layers):                 # gone some other way
        if getattr(lyr, "_epicycle_channel_index", None) is not None:
            viewer.layers.remove(lyr)
    c._request_load("P1.tif")
    assert _wait(qapp, lambda: not c._load_in_flight)
    assert len(_channel_layers(viewer)) == 2


def test_a_click_with_everything_present_does_nothing(qapp, setup):
    w, c, viewer = setup
    _load(qapp, c)
    before = list(viewer.layers)
    c._request_load("P1.tif")
    assert not c._load_in_flight
    assert list(viewer.layers) == before


def test_unsaved_edits_survive_the_round_trip(qapp, setup):
    w, c, viewer = setup
    _load(qapp, c)
    vol = c._labels_volume
    vol[4, 10:12, 10:12] = 7                       # a paint
    c._on_labels_painted()
    assert c._labels_dirty
    c.on_step_exit()
    c.on_step_enter()
    assert _wait(qapp, lambda: not c._load_in_flight)
    assert c._labels_dirty is True
    assert c._labels_volume is vol
    lab = c._find_labels_layer()
    assert lab is not None and lab.data is vol
    assert int(lab.data[4, 10, 10]) == 7


def test_two_exit_enter_cycles_do_not_accumulate(qapp, setup):
    w, c, viewer = setup
    _load(qapp, c)
    n = len(viewer.layers)
    for _ in range(2):
        c.on_step_exit()
        c.on_step_enter()
        assert _wait(qapp, lambda: not c._load_in_flight)
        assert len(viewer.layers) == n
    live = [x for x in c._stack_layers if any(x is y for y in viewer.layers)]
    assert len(c._stack_layers) == len(live) == n


def test_the_eye_works_after_re_entry(qapp, setup):
    w, c, viewer = setup
    _load(qapp, c)
    c.on_step_exit()
    c.on_step_enter()
    assert _wait(qapp, lambda: not c._load_in_flight)
    n = len(viewer.layers)
    c._on_stack_eye("P1.tif")                      # show the preview
    assert len(viewer.layers) > n
    c._on_stack_eye("P1.tif")                      # and hide it again
    assert len(viewer.layers) == n


def test_re_entry_puts_the_workbench_colours_back_on_the_labels(qapp, setup):
    """W4 (owner live test 2026-10-05): "Colours in viewer" is ticked, the
    labels layer re-added on re-entry must carry the Workbench's colours
    again (not napari's default colormap until the Workbench is opened)."""
    import pandas as pd

    from napari.utils.colormaps import DirectLabelColormap

    from napari_epicycle.fusion_reconstruction.qc_features import (
        PCA_FEATURES, save_patch_csv,
    )
    w, c, viewer = setup
    run = Path(c.state_manager.get_record("P1.tif")["labels_path"]).parent
    lab = np.zeros((6, 24, 24), np.uint32)
    for i in range(8):
        lab[1:3, 2 * i:2 * i + 2, 2:5] = i + 1
    tifffile.imwrite(str(run / "labels_3d.tif"), lab)
    rng = np.random.RandomState(0)
    cols = {"label": np.arange(1, 9)}
    for f in list(PCA_FEATURES)[:6]:
        cols[f] = rng.lognormal(size=8)
    cols["qc_score"] = np.linspace(0.0, 1.0, 8)
    cols["is_border_cell"] = np.zeros(8, bool)
    csv = run / "P1_features.csv"
    save_patch_csv(pd.DataFrame(cols), str(csv))
    c.state_manager.update_record("P1.tif", qc_features_csv=str(csv))
    c.state_manager.set_stack_status("P1.tif", "qc_extracted")
    _load(qapp, c)
    pump(qapp)
    assert c._sync_cb is not None and c._sync_cb.isChecked()
    lyr = c._find_labels_layer()
    assert isinstance(lyr.colormap, DirectLabelColormap)
    before = {k: np.asarray(v).tolist()
              for k, v in lyr.colormap.color_dict.items() if k}
    assert before
    c.on_step_exit()
    c.on_step_enter()
    assert _wait(qapp, lambda: not c._load_in_flight)
    pump(qapp)
    lyr = c._find_labels_layer()
    assert lyr is not None
    assert isinstance(lyr.colormap, DirectLabelColormap), type(lyr.colormap)
    after = {k: np.asarray(v).tolist()
             for k, v in lyr.colormap.color_dict.items() if k}
    assert after == before


def test_re_entry_recolours_the_labels_once(qapp, setup, monkeypatch):
    """Audit LOW-4: one colour pass per re-entry (each re-maps the shown
    labels on the main thread), not one after the labels and another after
    the image re-read."""
    w, c, viewer = setup
    _load(qapp, c)
    calls = []
    monkeypatch.setattr(c, "_on_sync_toggled",
                        lambda checked: calls.append(checked))
    monkeypatch.setattr(type(c), "_sync_cb", property(
        lambda self: type("CB", (), {"isChecked": lambda s: True})()))
    c.on_step_exit()
    c.on_step_enter()
    assert _wait(qapp, lambda: not c._load_in_flight)
    pump(qapp)
    assert len(_channel_layers(viewer)) == 2          # the image came back
    assert calls == [True]

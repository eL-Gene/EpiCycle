"""F15: Step 4's layers carry the ``step4 · ``
prefix, are registered under the ``"step4"`` context by their ACTUAL names
(a multichannel add returns a list; napari renames on a collision), and
clearing removes only Step 4's own layers -- never another step's ``ch0``.

A real napari ``ViewerModel`` (no canvas) and a real
``LayerLifecycleManager``.
"""
from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.segmentation import controller as C

from ._step4_panel import close, make_workspace, panel, pump

napari = pytest.importorskip("napari")


@pytest.fixture
def setup(qapp, tmp_path):
    from napari.components import ViewerModel

    from napari_epicycle.ui_shell.layer_lifecycle_manager import (
        LayerLifecycleManager,
    )
    ws = make_workspace(tmp_path, patches=("P1.tif",))
    w = panel(qapp, ws)
    c = w.controller
    viewer = ViewerModel()
    c.viewer = viewer
    c.layer_lifecycle_manager = LayerLifecycleManager(viewer)
    yield w, c, viewer
    c.viewer = None
    close(w)


def _load(c, n_ch=2):
    lab = np.zeros((4, 8, 8), np.int32)
    lab[1, 1:3, 1:3] = 1
    raw = {"data": np.zeros((n_ch, 4, 8, 8), np.uint8), "ch_axis": 0,
           "n_channels": n_ch, "axes": "CZYX"}
    c._apply_loaded("P1.tif", {"labels": lab, "n_cells": 1, "backup": "",
                               "raw": raw, "source_path": ""})


def test_every_step4_layer_carries_the_prefix_and_is_registered(setup):
    _w, c, viewer = setup
    _load(c)
    names = [layer.name for layer in viewer.layers]
    assert names and all(n.startswith(C.STEP4_LAYER_PREFIX) for n in names)
    reg = c.layer_lifecycle_manager._registry.get("step4", [])
    assert sorted(reg) == sorted(names)
    assert c._find_labels_layer() is not None


def test_another_steps_ch0_survives_a_step4_clear(setup):
    _w, c, viewer = setup
    viewer.add_image(np.zeros((4, 8, 8)), name="ch0")        # Step 3's
    viewer.add_image(np.zeros((4, 8, 8)), name="stack_image")
    _load(c)
    c._clear_stack_layers()
    left = [layer.name for layer in viewer.layers]
    assert left == ["ch0", "stack_image"]


def test_a_name_collision_registers_the_actual_name(setup):
    _w, c, viewer = setup
    # an earlier Step 4 layer name taken by someone else -> napari renames
    viewer.add_image(np.zeros((4, 8, 8)),
                     name=C.STEP4_LAYER_PREFIX + "labels")
    _load(c)
    reg = set(c.layer_lifecycle_manager._registry.get("step4", []))
    lab = c._find_labels_layer()
    assert lab is not None and lab.name in reg
    assert lab.name != C.STEP4_LAYER_PREFIX + "labels"
    c.layer_lifecycle_manager.clear_step("step4")
    assert [layer.name for layer in viewer.layers] == [
        C.STEP4_LAYER_PREFIX + "labels"]


def test_step_exit_clears_only_step4s_layers(setup):
    _w, c, viewer = setup
    viewer.add_image(np.zeros((4, 8, 8)), name="ch1")
    _load(c)
    c.on_step_exit()
    assert [layer.name for layer in viewer.layers] == ["ch1"]


def test_preview_layers_carry_the_prefix(setup, tmp_path):
    import tifffile
    _w, c, viewer = setup
    src = tmp_path / "S.tif"
    tifffile.imwrite(str(src), np.zeros((2, 4, 8, 8), np.uint8),
                     metadata={"axes": "CZYX"})
    c._preview_selected_stack("S.tif", {"source_stack_path": str(src)})
    names = [layer.name for layer in viewer.layers]
    assert names and all(n.startswith(C.STEP4_LAYER_PREFIX) for n in names)
    reg = set(c.layer_lifecycle_manager._registry.get("step4", []))
    assert set(names) <= reg
    c._on_stack_eye("S.tif")                 # the eye hides it again
    assert len(viewer.layers) == 0


def test_re_entry_brings_the_loaded_stacks_labels_back(setup):
    _w, c, viewer = setup
    _load(c)
    c.on_step_exit()
    assert c._find_labels_layer() is None
    c.on_step_enter()
    lab = c._find_labels_layer()
    assert lab is not None and lab.name.startswith(C.STEP4_LAYER_PREFIX)
    assert lab.data is c._labels_volume

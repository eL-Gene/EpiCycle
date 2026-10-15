"""Qt-free tests for ``packages.epicycle_annotator.labels_render``.

The napari>=0.7 hidden-layer placeholder hazard (see the module docstring):
after a re-show, a vispy colormap node built against the placeholder slice
must be rebuilt; a healthy node must be left alone (zero cost).
"""
from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")
from napari.layers import Labels  # noqa: E402
from napari.utils.colormaps import DirectLabelColormap, label_colormap  # noqa: E402

from packages.epicycle_annotator.labels_render import (  # noqa: E402
    heal_labels_render,
    labels_colormap_node_is_stale,
    rebuild_stale_labels_colormap,
)


class _Cmap:
    pass


class DirectLabelVispyColormap(_Cmap):  # name is what the check reads
    pass


class LabelVispyColormap(_Cmap):
    pass


class _FakeNode:
    def __init__(self, cmap):
        self.cmap = cmap


class _FakeVispyLayer:
    def __init__(self, cmap):
        self.node = _FakeNode(cmap)
        self.rebuilds = 0

    def _on_colormap_change(self, event=None):
        self.rebuilds += 1
        self.node.cmap = DirectLabelVispyColormap()


class _FakeCanvas:
    def __init__(self):
        self.layer_to_visual = {}


class _FakeQtViewer:
    def __init__(self):
        self.canvas = _FakeCanvas()


def _direct(ids):
    d = {int(i): np.array([0.2, 0.4, 0.6, 1.0], dtype=np.float32) for i in ids}
    d[None] = np.array([0, 0, 0, 0], dtype=np.float32)
    return DirectLabelColormap(color_dict=d)


def _labels(dtype=np.uint32, visible=True):
    data = np.zeros((4, 8, 8), dtype=dtype)
    data[1:3, 2:6, 2:6] = 7
    data[0, 0, 0] = 300
    layer = Labels(data, name="lab", visible=visible)
    layer.colormap = _direct([7, 300])
    return layer


def _wire(layer, node_cmap):
    qtv = _FakeQtViewer()
    vl = _FakeVispyLayer(node_cmap)
    qtv.canvas.layer_to_visual[layer] = vl
    return qtv, vl


# --------------------------------------------------------------------------
# labels_colormap_node_is_stale
# --------------------------------------------------------------------------

def test_healthy_direct_node_is_not_stale():
    layer = _labels()
    qtv, _ = _wire(layer, DirectLabelVispyColormap())
    assert labels_colormap_node_is_stale(layer, qtv) is False


def test_auto_shader_on_direct_uint32_is_stale():
    layer = _labels()
    qtv, _ = _wire(layer, LabelVispyColormap())
    assert labels_colormap_node_is_stale(layer, qtv) is True


def test_small_int_data_legitimately_uses_auto_shader():
    layer = _labels(dtype=np.uint16)
    qtv, _ = _wire(layer, LabelVispyColormap())
    assert labels_colormap_node_is_stale(layer, qtv) is False


def test_cyclic_colormap_is_never_reported_stale():
    layer = _labels()
    layer.colormap = label_colormap(49, seed=0.5)
    qtv, _ = _wire(layer, LabelVispyColormap())
    assert labels_colormap_node_is_stale(layer, qtv) is False


def test_hidden_layer_is_undetermined():
    layer = _labels(visible=False)
    qtv, _ = _wire(layer, LabelVispyColormap())
    assert labels_colormap_node_is_stale(layer, qtv) is None


def test_missing_visual_or_viewer_is_undetermined():
    layer = _labels()
    assert labels_colormap_node_is_stale(layer, None) is None
    assert labels_colormap_node_is_stale(layer, _FakeQtViewer()) is None


def test_non_labels_is_undetermined():
    assert labels_colormap_node_is_stale(object(), _FakeQtViewer()) is None


# --------------------------------------------------------------------------
# rebuild_stale_labels_colormap / heal_labels_render
# --------------------------------------------------------------------------

def test_rebuild_only_when_stale():
    layer = _labels()
    qtv, vl = _wire(layer, LabelVispyColormap())
    assert rebuild_stale_labels_colormap(layer, qtv) is True
    assert vl.rebuilds == 1
    # second pass: node now consistent -> untouched
    assert rebuild_stale_labels_colormap(layer, qtv) is False
    assert vl.rebuilds == 1


def test_healthy_layer_costs_nothing():
    layer = _labels()
    qtv, vl = _wire(layer, DirectLabelVispyColormap())
    assert rebuild_stale_labels_colormap(layer, qtv) is False
    assert vl.rebuilds == 0


def test_heal_counts_and_prints(capsys):
    a = _labels()
    b = _labels()
    qa, va = _wire(a, LabelVispyColormap())
    qb, vb = _wire(b, DirectLabelVispyColormap())
    n = heal_labels_render([(a, qa), (b, qb), (None, None)], tag="unit")
    assert n == 1
    assert va.rebuilds == 1 and vb.rebuilds == 0
    assert "[RENDER-HEAL] unit" in capsys.readouterr().out


def test_heal_is_silent_when_nothing_to_do(capsys):
    a = _labels()
    qa, _ = _wire(a, DirectLabelVispyColormap())
    assert heal_labels_render([(a, qa)], tag="unit") == 0
    assert "[RENDER-HEAL]" not in capsys.readouterr().out


def test_rebuild_never_raises_on_broken_visual():
    layer = _labels()
    qtv, vl = _wire(layer, LabelVispyColormap())

    def boom(event=None):
        raise RuntimeError("gl context gone")

    vl._on_colormap_change = boom
    assert rebuild_stale_labels_colormap(layer, qtv) is False

"""The labels key under a single-cell Isolate also toggles the proxy in 2D.

The single-cell isolation proxy is the layer the main viewer shows while
Isolate is on, in 2D as well as 3D.  The labels key therefore routes to the
proxy plus the three side-view label layers whenever the proxy is a live
layer of the viewer, whatever the display mode, and falls back to the
default route when it is not.

Qt-free apart from one bare QWidget: the annotator is built without its
heavy constructor and given fake layers.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication, QWidget  # noqa: E402

from napari_epicycle.annotator.curation_3d_widget import (  # noqa: E402
    Curation3DAnnotator,
)


class _Layer:
    def __init__(self, name):
        self.name = name
        self.visible = True


class _VM:
    def __init__(self, orient, layers):
        self._orient = orient
        self.layers = list(layers)
        self.dims = types.SimpleNamespace(ndisplay=2)


class _Bare(Curation3DAnnotator):
    def __init__(self):          # skip the full annotator build
        QWidget.__init__(self)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _by_name(layers, name):
    for lyr in layers:
        if getattr(lyr, "name", None) == name:
            return lyr
    return None


def _rig(app, *, ndisplay, proxy_in_viewer):
    main = _Layer("labels_3d")
    proxy = _Layer("_epicycle_isolation_proxy")
    subs = {o: _Layer(f"labels_3d_{o}") for o in ("XY", "XZ", "YZ")}
    ann = _Bare()
    ann._refine_layer = None
    ann._multi_label_proxy_layers = {}
    ann._isolation_proxy_layer = proxy
    ann._isolation_proxy_label_id = 5
    ann._active_labels_names = ["labels_3d"]
    ann._add_running = False
    ann._add_candidate_layer = None
    ann.viewer = types.SimpleNamespace(
        layers=[main] + ([proxy] if proxy_in_viewer else []),
        dims=types.SimpleNamespace(ndisplay=ndisplay))
    ann.viewer_model_xy = _VM("XY", [subs["XY"]])
    ann.viewer_model_xz = _VM("XZ", [subs["XZ"]])
    ann.viewer_model_yz = _VM("YZ", [subs["YZ"]])
    ann._layer_by_name = _by_name
    ann._orientation_for_vm = lambda vm: vm._orient
    return ann, main, proxy, subs


def test_2d_isolate_routes_to_the_proxy_and_the_side_views(app):
    ann, main, proxy, subs = _rig(app, ndisplay=2, proxy_in_viewer=True)
    targets = ann._get_toggle_targets_for_name("labels_3d")
    assert proxy in targets
    for s in subs.values():
        assert s in targets
    assert main not in targets


def test_3d_isolate_is_unchanged(app):
    ann, main, proxy, subs = _rig(app, ndisplay=3, proxy_in_viewer=True)
    targets = ann._get_toggle_targets_for_name("labels_3d")
    assert proxy in targets
    for s in subs.values():
        assert s in targets
    assert main not in targets


@pytest.mark.parametrize("nd", [2, 3])
def test_a_proxy_not_in_the_viewer_takes_the_default_route(app, nd):
    ann, main, proxy, subs = _rig(app, ndisplay=nd, proxy_in_viewer=False)
    targets = ann._get_toggle_targets_for_name("labels_3d")
    assert main in targets
    for s in subs.values():
        assert s in targets
    assert proxy not in targets

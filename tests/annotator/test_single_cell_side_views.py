"""The side views follow every single-cell isolation entry.

A Refine Further reject rebuilds the isolated cell's proxy without going
through the isolation call, and the side views kept the fragment's one-cell
map (so the cell drew nothing in XY / XZ / YZ).  A hidden isolation stays
hidden when the isolated cell changes; Render Selected with two cells shows
the side views again.  Real Labels layers and ViewerModels.
"""
from __future__ import annotations

import types
from types import SimpleNamespace

import numpy as np

from napari.components import ViewerModel

from napari_epicycle.annotator.curation_3d_widget import (
    Curation3DAnnotator as K)

_BIND = ("_update_isolation_proxy", "_propagate_colormap_to_subviewers",
         "_propagate_show_selected_label", "_layer_by_name")


def _env():
    data = np.zeros((20, 20, 20), np.uint32)
    data[2:6, 2:6, 2:6] = 250
    data[10:14, 10:14, 10:14] = 690
    data[2:5, 12:16, 12:16] = 691
    viewer = ViewerModel()
    main = viewer.add_labels(data, name="labels")
    sides = {}
    models = {}
    for tag in ("xy", "xz", "yz"):
        vm = ViewerModel()
        sides[tag] = vm.add_labels(np.zeros((4, 4, 4), np.uint32),
                                   name="labels")
        models[tag] = vm
    sync = SimpleNamespace(
        viewer_model_xy=models["xy"], viewer_model_xz=models["xz"],
        viewer_model_yz=models["yz"],
        _base_label_name=lambda n: n, _suppress_show_selected_sync=False)
    w = SimpleNamespace(
        viewer=viewer, sync_manager=sync, _active_labels_name="labels",
        _isolation_proxy_layer=None, _isolation_proxy_active=False,
        _isolation_proxy_label_id=None, _multi_label_proxy_layers={},
        _multi_label_proxy_set=None, _plot_cmap=None, _plot_cmap_pull_cb=None,
        main=main, sides=sides)
    for name in _BIND:
        setattr(w, name, types.MethodType(getattr(K, name), w))
    return w


def _alpha(layer, lid):
    return float(np.asarray(layer.colormap.map(np.array([lid])))[0][3])


def test_a_rebuilt_proxy_gives_the_side_views_the_cells_own_map():
    w = _env()
    w._update_isolation_proxy(250, w.main)
    w._update_isolation_proxy(690, w.main, force=True)      # fragment mode
    assert all(_alpha(s, 250) == 0 for s in w.sides.values())
    w._update_isolation_proxy(250, w.main, force=True)      # the reject
    for s in w.sides.values():
        assert _alpha(s, 250) > 0
        assert s.show_selected_label is True
        assert s.selected_label == 250


def test_a_new_proxy_for_another_cell_also_syncs_the_side_views():
    w = _env()
    w._update_isolation_proxy(250, w.main)
    w._update_isolation_proxy(691, w.main)
    for s in w.sides.values():
        assert _alpha(s, 691) > 0


def test_a_hidden_isolation_stays_hidden_when_the_cell_changes():
    w = _env()
    w._update_isolation_proxy(250, w.main)
    w._isolation_proxy_layer.visible = False        # the hide key
    for s in w.sides.values():
        s.visible = False
    w._update_isolation_proxy(691, w.main)
    assert w._isolation_proxy_layer.visible is False
    assert all(s.visible is False for s in w.sides.values())
    for s in w.sides.values():
        assert _alpha(s, 691) > 0                   # the map is still right


def test_a_visible_isolation_shows_the_side_views_for_the_new_cell():
    w = _env()
    w._update_isolation_proxy(250, w.main)
    for s in w.sides.values():
        s.visible = False
    w._update_isolation_proxy(691, w.main)
    assert all(s.visible for s in w.sides.values())


def test_render_selected_with_two_cells_shows_the_side_views():
    from .test_isolation_colormap_guard import _fake

    ann = _fake(ndisplay=3)
    sides = []
    for _ in range(3):
        vm = ViewerModel()
        sides.append(vm.add_labels(np.zeros((4, 4, 4), np.uint32),
                                   name="labels"))
        sides[-1].visible = False
    ann.sync_manager = SimpleNamespace(
        viewer_model_xy=SimpleNamespace(layers=[sides[0]]),
        viewer_model_xz=SimpleNamespace(layers=[sides[1]]),
        viewer_model_yz=SimpleNamespace(layers=[sides[2]]),
        _base_label_name=lambda n: n, _suppress_show_selected_sync=False)
    ann._propagate_multi_label_colormap_to_subviewers = lambda s: 0
    ann._on_isolation_changed({1, 2})
    assert all(s.visible for s in sides)


def test_render_selected_keeps_a_hidden_isolation_hidden():
    from .test_isolation_colormap_guard import _fake

    ann = _fake(ndisplay=3)
    side = ViewerModel().add_labels(np.zeros((4, 4, 4), np.uint32),
                                    name="labels")
    side.visible = False
    ann.sync_manager = SimpleNamespace(
        viewer_model_xy=SimpleNamespace(layers=[side]),
        viewer_model_xz=SimpleNamespace(layers=[]),
        viewer_model_yz=SimpleNamespace(layers=[]),
        _base_label_name=lambda n: n, _suppress_show_selected_sync=False)
    ann._propagate_multi_label_colormap_to_subviewers = lambda s: 0
    ann._multi_label_proxy_layers = {1: SimpleNamespace(visible=False)}
    ann._multi_label_proxy_set = frozenset({1})
    ann._on_isolation_changed({1, 2})
    assert side.visible is False


def _op_env():
    w = _env()
    w.viewer.dims.ndisplay = 3
    w.sync_manager.shared_labels = {"labels": w.main.data}
    w._isolation_active = True
    w._isolation_label_set = None
    w._fragment_refine_roi = None
    w._pre_isolation_colormap = None
    w._refresh_label_data_in_subviewers = lambda: None
    w._get_selected_label = lambda: 250
    for name in ("_sync_all_views_after_operation", "_iso_refresh",
                 "_label_crop", "_iso_live_ids"):
        setattr(w, name, types.MethodType(getattr(K, name), w))
    return w


def test_the_sync_after_a_rejected_fragment_edit_keeps_the_cell_drawn():
    w = _op_env()
    w._update_isolation_proxy(250, w.main)
    w._update_isolation_proxy(690, w.main, force=True)   # Refine Further
    w._update_isolation_proxy(250, w.main, force=True)   # Reject
    w._sync_all_views_after_operation(label_id=250)
    for s in w.sides.values():
        assert _alpha(s, 250) > 0
    assert _alpha(w.main, 250) > 0
    # main is masked under the proxy: a cell that is not isolated is clear
    assert _alpha(w.main, 691) == 0
    assert _alpha(w.main, 690) == 0
    w._sync_all_views_after_operation(label_id=250)      # e.g. a smooth preview
    for s in w.sides.values():
        assert _alpha(s, 250) > 0

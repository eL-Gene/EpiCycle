"""The isolation shows the colour the QC plot gives a cell.

A single-cell proxy, a multi-cell sub-proxy and the side views' colour map all
take the plot's colour for the cell; when the plot has no colour for an id (a
cell painted a moment ago) or the Colour box is off, napari's own colour is
used, so the cell is visible at once.  Qt-free; real napari Labels layers.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from napari_epicycle.annotator import curation_3d_widget as mod
from napari_epicycle.annotator.curation_3d_widget import (
    Curation3DAnnotator as K,
    _one_label_colormap,
    _plot_colour_for,
)

RED = np.array([1, 0, 0, 1], np.float32)
GREEN = np.array([0, 1, 0, 1], np.float32)
BLUE = np.array([0, 0, 1, 0.5], np.float32)
PLOT = {1: RED, 2: GREEN, 3: BLUE}


class _Layers(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            for lyr in self:
                if getattr(lyr, "name", None) == key:
                    return lyr
            raise KeyError(key)
        return super().__getitem__(key)


def _widget(plot=PLOT):
    from napari.layers import Labels

    data = np.zeros((6, 6, 6), np.uint32)
    data[1:3, 1:3, 1:3] = 1
    data[3:5, 1:3, 1:3] = 2
    data[1:3, 3:5, 3:5] = 3
    main = Labels(data, name="labels")
    w = SimpleNamespace(
        viewer=SimpleNamespace(layers=_Layers([main])),
        _plot_cmap=plot, _plot_cmap_pull_cb=None,
        _multi_label_proxy_layers={}, _isolation_active=True,
        _isolation_proxy_active=False, _isolation_proxy_label_id=None,
        _isolation_proxy_layer=None, _multi_label_proxy_set=None,
        _isolation_label_set=None, _active_labels_name="labels",
        main=main, calls=[],
    )
    w._layer_by_name = lambda layers, name: layers[name]
    w._propagate_colormap_to_subviewers = (
        lambda cm: w.calls.append(("SUB", cm)) or 3)
    w._propagate_show_selected_label = (
        lambda v, **k: w.calls.append(("SHOWSEL", v, k.get("label_id"))))
    w._propagate_multi_label_colormap_to_subviewers = (
        lambda ids: K._propagate_multi_label_colormap_to_subviewers(w, ids))
    w._apply_multi_isolation_colormap = (
        lambda m, ids: w.calls.append(("MULTI", frozenset(ids))))
    return w


def _rgba(cmap, lid):
    return np.asarray(cmap.map(np.array([lid])))[0]


@pytest.mark.parametrize("ids", [[1], [1, 2], [1, 2, 3]])
def test_the_colour_for_is_the_plots(ids):
    got = _plot_colour_for(_widget(), ids)
    for lid in ids:
        assert np.allclose(got[lid], PLOT[lid])


def test_an_id_the_plot_lacks_takes_napari_colour_and_is_visible():
    got = _plot_colour_for(_widget(), [1, 99])
    assert np.allclose(got[1], RED)
    assert got[99][3] > 0


def test_colour_box_off_uses_napari_colours():
    got = _plot_colour_for(_widget(plot=None), [1])
    assert not np.allclose(got[1], RED)
    assert got[1][3] > 0


def test_the_pull_callback_is_asked_once():
    w = _widget(plot=mod._PLOT_CMAP_UNSET)
    asked = []
    w._plot_cmap_pull_cb = lambda: asked.append(1) or PLOT
    _plot_colour_for(w, [1])
    _plot_colour_for(w, [2])
    assert asked == [1]
    assert w._plot_cmap is PLOT


def test_the_one_label_map_shows_only_that_cell():
    cm = _one_label_colormap(_widget(), 2)
    assert np.allclose(_rgba(cm, 2), GREEN)
    assert _rgba(cm, 1)[3] == 0
    assert _rgba(cm, 77)[3] == 0


def test_a_sub_proxy_wears_the_plot_colour():
    w = _widget()
    roi = w.main.data.copy()
    w.viewer.layers = _Layers([w.main])
    for lid in (1, 2, 3):
        K._build_single_label_subproxy(
            w, w.main, lid, crop=((slice(0, 6),) * 3, roi))
        sub = w._multi_label_proxy_layers[lid]
        assert np.allclose(_rgba(sub.colormap, lid), PLOT[lid])
        assert sub.show_selected_label is True


def test_the_side_views_multi_map_uses_the_plot_colours():
    w = _widget()
    K._propagate_multi_label_colormap_to_subviewers(w, {1, 2, 3})
    (_, cm), = [c for c in w.calls if c[0] == "SUB"]
    for lid in (1, 2, 3):
        assert np.allclose(_rgba(cm, lid), PLOT[lid])


def test_recolour_single_proxy_follows_a_new_plot_colour():
    from napari.layers import Labels

    w = _widget()
    proxy = Labels(w.main.data.copy(), name="p")
    proxy.selected_label = 2
    w._isolation_proxy_layer = proxy
    w._isolation_proxy_active = True
    w._isolation_proxy_label_id = 2
    K._iso_recolour(w)
    assert np.allclose(_rgba(proxy.colormap, 2), GREEN)
    w._plot_cmap = {2: BLUE}
    first = proxy.colormap
    K._iso_recolour(w)
    assert proxy.colormap is not first
    assert np.allclose(_rgba(proxy.colormap, 2), BLUE)
    assert proxy.show_selected_label is True
    assert ("SHOWSEL", True, 2) in w.calls
    assert any(c[0] == "SUB" for c in w.calls)


def test_recolour_multi_proxies_and_side_views():
    from napari.layers import Labels

    w = _widget()
    subs = {lid: Labels(w.main.data.copy(), name=f"s{lid}") for lid in (1, 3)}
    for lid, sub in subs.items():
        sub.selected_label = lid
    w._multi_label_proxy_layers = subs
    w._multi_label_proxy_set = frozenset({1, 3})
    w._plot_cmap = {1: BLUE, 3: RED}
    K._iso_recolour(w)
    assert np.allclose(_rgba(subs[1].colormap, 1), BLUE)
    assert np.allclose(_rgba(subs[3].colormap, 3), RED)
    (_, cm), = [c for c in w.calls if c[0] == "SUB"]
    assert np.allclose(_rgba(cm, 1), BLUE)


def test_recolour_colour_map_mode_clears_the_cache_first():
    w = _widget()
    w._isolation_label_set = frozenset({1, 2, 3})
    seen = []
    w._apply_multi_isolation_colormap = (
        lambda m, ids: seen.append((w._isolation_label_set, frozenset(ids))))
    K._iso_recolour(w)
    assert seen == [(None, frozenset({1, 2, 3}))]


def test_recolour_does_nothing_without_isolation():
    w = _widget()
    w._isolation_active = False
    K._iso_recolour(w)
    assert w.calls == []

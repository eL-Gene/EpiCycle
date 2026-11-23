"""Follow-ups to the isolation colour family.

* the cell isolated after a recolour is visible in the side views;
* 2D single-cell isolation follows a Colour-by change;
* the colour map the isolation puts back is rebuilt by a recolour;
* the navigator names the Isolate-box untick;
* a new patch makes the annotator forget the plot colours.
Qt-free; real napari Labels layers.
"""
from __future__ import annotations

import types
from types import SimpleNamespace

import numpy as np

from napari_epicycle.annotator import curation_3d_widget as mod
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K

from .test_isolation_plot_colour import (
    BLUE, GREEN, PLOT, RED, _rgba, _widget)

NEW_PLOT = {1: BLUE, 2: GREEN, 3: RED}


def test_the_next_isolated_cell_is_visible_in_the_side_views_after_a_recolour():
    from napari.layers import Labels

    w = _widget()
    w.viewer.dims = SimpleNamespace(ndisplay=3)
    w._isolation_active = True
    w._isolation_proxy_layer = Labels(w.main.data.copy(), name="p")
    w._isolation_proxy_layer.selected_label = 1
    w._isolation_proxy_active = True
    w._isolation_proxy_label_id = 1
    w._pre_isolation_colormap = None
    w._colormap_authority = None
    w._on_navigator_label_selected_cb = None
    w._isolation_bridge_dispatched = True
    w._on_isolation_cleared_cb = None
    w._update_isolation_proxy = lambda lid, main: (
        setattr(w, "_isolation_proxy_label_id", lid))
    w._remove_isolation_proxy = lambda **k: None
    w._plot_cmap = {1: RED}
    K._iso_recolour(w)                 # side views now hold a one-label map
    w._plot_cmap = {1: RED, 2: GREEN}
    w.calls.clear()
    K._on_isolation_changed(w, {2})
    maps = [c[1] for c in w.calls if c[0] == "SUB"]
    assert maps, "the side views were not given a colour map"
    assert np.allclose(_rgba(maps[-1], 2), GREEN)
    order = [c[0] for c in w.calls]
    assert order.index("SUB") < order.index("SHOWSEL")


def test_2d_single_isolation_follows_a_colour_by_change():
    w = _widget()
    w.main.selected_label = 2
    w.main.show_selected_label = True
    w._plot_cmap = {1: RED, 2: GREEN, 3: BLUE}
    K._iso_recolour(w)
    assert np.allclose(_rgba(w.main.colormap, 2), GREEN)
    first = w.main.colormap
    w._plot_cmap = {2: RED}
    K._iso_recolour(w)
    assert w.main.colormap is not first
    assert np.allclose(_rgba(w.main.colormap, 2), RED)
    assert w.main.show_selected_label is True
    (_, side), = [c for c in w.calls if c[0] == "SUB"][-1:]
    assert side is w.main.colormap
    assert ("SHOWSEL", True, 2) in w.calls


def test_2d_single_isolation_with_the_colour_box_off_uses_napari_colours():
    w = _widget()
    w.main.selected_label = 2
    w.main.show_selected_label = True
    w._plot_cmap = None
    K._iso_recolour(w)
    assert w.main.show_selected_label is True
    assert _rgba(w.main.colormap, 2)[3] > 0


def test_a_recolour_rebuilds_the_colour_map_the_isolation_puts_back():
    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapAuthority)

    w = _widget()
    w._iso_refresh_saved_colormap = types.MethodType(
        K._iso_refresh_saved_colormap, w)
    w._colormap_authority = ColormapAuthority()
    old = object()
    w._pre_isolation_colormap = old
    w._multi_label_proxy_set = frozenset({1})
    w._multi_label_proxy_layers = {}
    w._plot_cmap = NEW_PLOT
    K._iso_recolour(w)
    assert w._pre_isolation_colormap is not old
    for lid in (1, 2, 3):
        assert np.allclose(_rgba(w._pre_isolation_colormap, lid), NEW_PLOT[lid])
    assert w._colormap_authority.restore() is w._pre_isolation_colormap


def test_a_suspended_edit_gets_the_new_colours_on_main_and_side_views():
    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapAuthority, ColormapOwner)

    w = _widget()
    w._iso_refresh_saved_colormap = types.MethodType(
        K._iso_refresh_saved_colormap, w)
    auth = ColormapAuthority()
    auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
    w._colormap_authority = auth
    w._isolation_active = False
    w._pre_isolation_colormap = None
    w._plot_cmap = NEW_PLOT
    w.main.visible = False
    K._iso_recolour(w)
    assert np.allclose(_rgba(w.main.colormap, 1), BLUE)
    assert any(c[0] == "SUB" for c in w.calls)


# ---------------------------------------------------------------- navigator
def _nav(cb):
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator as N)

    nav = SimpleNamespace(
        _on_isolation_changed_cb=cb, _on_show_selected_label_changed_cb=None,
        _curation_debug=lambda m: None, selected_label_ids=lambda: {4, 5},
        _is_op_active=lambda: False, _update_render_button_state=lambda: None,
        _last_rendered_label_ids=set())
    nav._on_isolate_toggled = lambda checked: N._on_isolate_toggled(nav, checked)
    return nav


def test_the_navigator_names_the_untick():
    calls = []
    nav = _nav(lambda ids, **k: calls.append((set(ids), k)))
    nav._on_isolate_toggled(True)
    nav._on_isolate_toggled(False)
    assert calls == [({4, 5}, {}), (set(), {"reason": "box_untick"})]


def test_the_navigator_untick_works_with_a_one_argument_callback():
    calls = []
    nav = _nav(lambda ids: calls.append(set(ids)))
    nav._on_isolate_toggled(False)
    assert calls == [set()]


# ------------------------------------------------------------- patch switch
def test_a_patch_switch_makes_the_annotator_forget_the_plot_colours():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C)
    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapAuthority)

    c = object.__new__(C)
    widget = SimpleNamespace(_plot_cmap={1: RED})
    c._get_curation_annotator_widget = lambda: widget
    c._colormap_authority = ColormapAuthority()
    c._isolated_label_ids = set()
    c._isolated_df_indices = []
    c.ui = SimpleNamespace(qc_plot_widget=None)
    try:
        C._hard_clear_curation_surfaces(c)
    except Exception:
        pass            # the rest of the clear needs a full controller
    assert widget._plot_cmap is mod._PLOT_CMAP_UNSET


# ------------------------------------------------------ re-audit follow-ups
def test_suspending_a_one_cell_isolation_gives_the_side_views_the_full_map():
    w = _widget()
    w._iso_live_ids = types.MethodType(K._iso_live_ids, w)
    w._remove_isolation_proxy = lambda **k: None
    w._colormap_authority = None
    w._pre_isolation_colormap = None
    w._isolation_active = True
    w._isolation_proxy_active = True
    w._isolation_proxy_label_id = 1
    w._plot_cmap = PLOT
    K._iso_suspend(w)
    maps = [c[1] for c in w.calls if c[0] == "SUB"]
    assert maps
    assert _rgba(maps[-1], 3)[3] > 0          # a cell that was not isolated
    assert np.allclose(_rgba(maps[-1], 3), BLUE)


def test_a_visible_main_is_not_recoloured_during_a_suspended_edit():
    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapAuthority, ColormapOwner)

    w = _widget()
    w._iso_refresh_saved_colormap = types.MethodType(
        K._iso_refresh_saved_colormap, w)
    auth = ColormapAuthority()
    auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
    w._colormap_authority = auth
    w._isolation_active = False
    w._pre_isolation_colormap = None
    w._plot_cmap = NEW_PLOT
    w.main.visible = True
    before = w.main.colormap
    K._iso_refresh_saved_colormap(w)
    assert w.main.colormap is before
    assert any(c[0] == "SUB" for c in w.calls)
    w.main.visible = False
    K._iso_refresh_saved_colormap(w)
    assert w.main.colormap is not before


def _skip_widget(live, plot, selected=None):
    w = _widget(plot=plot)
    w._iso_live_ids = lambda: frozenset(live)
    if selected is not None:
        w.main.show_selected_label = True
        w.main.selected_label = selected
    return w


def test_unchanged_check_compares_the_whole_map_when_nothing_is_live():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C)

    same = {k: v.copy() for k, v in PLOT.items()}
    assert C._plot_colours_unchanged_for(_skip_widget([], PLOT), same)
    other = dict(same)
    other[3] = RED
    assert not C._plot_colours_unchanged_for(_skip_widget([], PLOT), other)


def test_unchanged_check_uses_the_selected_cell_in_2d_single_mode():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C)

    other = dict(PLOT)
    other[3] = RED                       # only a cell that is not selected
    w = _skip_widget([], PLOT, selected=1)
    assert C._plot_colours_unchanged_for(w, other)
    other[1] = BLUE
    assert not C._plot_colours_unchanged_for(w, other)


def test_the_controller_push_is_silent_for_a_bare_stand_in():
    from napari_epicycle.fusion_reconstruction import controller as ctl

    bare = SimpleNamespace()
    assert ctl._push_plot_cmap_to_annotator(bare, {1: RED}) is None
    assert ctl._push_plot_cmap_to_annotator(bare, None) is None
    assert ctl._read_plot_colour_dict(bare) is None

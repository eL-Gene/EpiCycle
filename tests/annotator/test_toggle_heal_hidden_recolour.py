"""Hide / unhide (the labels key) during an isolation.

* the labels key re-shows every layer it hid with its render repaired, so
  side views that moved while hidden do not draw every label;
* a recolour while the user has hidden the labels leaves them hidden;
* under a one-cell 3D isolation a stray show of main still draws only that
  cell.

Real launcher on an offscreen napari viewer, real Labels layers.
"""
from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import pytest

pytest.importorskip("napari")

PLOT = {
    1: np.array([1, 0, 0, 1], np.float32),
    2: np.array([0, 1, 0, 1], np.float32),
    3: np.array([0, 0, 1, 1], np.float32),
    4: np.array([1, 1, 0, 1], np.float32),
}


def _pump(n=30):
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    for _ in range(n):
        app.processEvents()
        time.sleep(0.005)


class _Rig:
    pass


@pytest.fixture(scope="module")
def rig():
    import napari
    from napari.layers import Labels
    from qtpy.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    try:
        viewer = napari.Viewer(show=False)
    except Exception as exc:    # no GL context (offscreen CI)
        pytest.skip(f"needs a GL context: {exc!r}")
    img = np.random.rand(20, 40, 40).astype(np.float32)
    lab = np.zeros((20, 40, 40), np.uint32)
    lab[2:8, 2:10, 2:10] = 1
    lab[2:8, 20:30, 20:30] = 2
    lab[10:18, 5:15, 25:35] = 3
    lab[10:18, 25:35, 5:15] = 4
    viewer.add_image(img, name="img")
    viewer.add_labels(lab, name="labels")
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    from napari_epicycle.annotator.launcher import launch_3d_curation_annotator
    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapAuthority)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok, msg = launch_3d_curation_annotator(
            viewer, image_layer_name="img", labels_layer_name="labels")
        _pump(50)
        w = None
        for dw in viewer.window._dock_widgets.values():
            try:
                inner = dw.widget()
            except Exception:
                continue
            for c in [inner] + list(inner.findChildren(Curation3DAnnotator)):
                if isinstance(c, Curation3DAnnotator):
                    w = c
        assert ok and w is not None, msg
        _pump(100)
        if not w._active_labels_names:
            w.image_combo.setCurrentText("img")
            w.labels_combo.setCurrentText("labels")
            w._load_selected_layers()
            _pump(100)
        w.set_colormap_authority(ColormapAuthority())
        w._plot_cmap = dict(PLOT)
        viewer.dims.ndisplay = 3
        _pump(30)
    keys = []
    for _v, k in w._channel_keybindings:
        if k not in keys:
            keys.append(k)
    r = _Rig()
    r.viewer, r.w, r.key, r.buf = viewer, w, keys[-1], buf
    r.main = viewer.layers["labels"]
    r.Labels = Labels
    yield r
    with contextlib.suppress(Exception):
        viewer.close()


def _sides(r):
    out = []
    for vm in (r.w.viewer_model_xy, r.w.viewer_model_xz, r.w.viewer_model_yz):
        for lyr in vm.layers:
            if isinstance(lyr, r.Labels) and lyr.name.startswith("labels"):
                out.append(lyr)
    return out


def _qv_for(r, layer):
    sm = r.w.subviewer_manager
    for vm, qv in ((r.w.viewer_model_xy, sm.qt_viewer_xy),
                   (r.w.viewer_model_xz, sm.qt_viewer_xz),
                   (r.w.viewer_model_yz, sm.qt_viewer_yz)):
        if layer in vm.layers:
            return qv


def _toggle(r):
    fn = None
    for kb, f in r.viewer.keymap.items():
        if str(kb).lower() == r.key.lower():
            fn = f
    assert fn is not None
    with contextlib.redirect_stdout(r.buf):
        fn(r.viewer)
        _pump(40)


def _isolate(r, ids):
    with contextlib.redirect_stdout(r.buf):
        r.w._on_isolation_changed(ids)
        _pump(80)


def _end(r):
    with contextlib.redirect_stdout(r.buf):
        r.w._on_isolation_changed([])
        _pump(60)
    if not r.main.visible:
        _toggle(r)


def _step_sides(r):
    with contextlib.redirect_stdout(r.buf):
        for vm in (r.w.viewer_model_xy, r.w.viewer_model_xz, r.w.viewer_model_yz):
            cs = list(vm.dims.current_step)
            cs[0] = (cs[0] + 1) % 5 + 1
            with contextlib.suppress(Exception):
                vm.dims.current_step = tuple(cs)
        _pump(40)


def _poke(r):
    with contextlib.redirect_stdout(r.buf):
        for lyr in _sides(r):
            cur = lyr.selected_label
            lyr.selected_label = cur + 1
            lyr.selected_label = cur
            lyr.show_selected_label = lyr.show_selected_label
        _pump(30)


def _recolour(r):
    for k in PLOT:
        PLOT[k] = PLOT[k].copy()
        PLOT[k][0] = (PLOT[k][0] + 0.37) % 1.0
    r.w._plot_cmap = dict(PLOT)
    with contextlib.redirect_stdout(r.buf):
        r.w._iso_recolour()
        _pump(40)


def _alpha(cm, lid):
    return float(np.asarray(cm.map(np.array([lid])))[0][3])


@pytest.mark.parametrize("ids", [[1], [1, 2]])
def test_unhide_after_the_side_views_moved_leaves_no_stale_node(rig, ids):
    from packages.epicycle_annotator.labels_render import (
        labels_colormap_node_is_stale)

    _isolate(rig, ids)
    _toggle(rig)                       # hide
    _step_sides(rig)
    _poke(rig)
    if len(ids) > 1:
        _recolour(rig)
    calls = []
    real = rig.w._heal_label_render

    def spy(*a, **k):
        calls.append((a, k))
        return real(*a, **k)

    rig.w._heal_label_render = spy
    try:
        _toggle(rig)                   # show
    finally:
        del rig.w._heal_label_render
    try:
        assert calls, "the show branch never repaired the render"
        for lyr in _sides(rig):
            assert lyr.visible
            assert labels_colormap_node_is_stale(lyr, _qv_for(rig, lyr)) is not True
    finally:
        _end(rig)


def test_recolour_while_hidden_keeps_main_hidden_with_three_cells(rig):
    _isolate(rig, [1, 2, 3])
    _toggle(rig)
    assert not rig.main.visible
    _recolour(rig)
    assert not rig.main.visible, "the recolour re-showed a main the user hid"
    _toggle(rig)
    assert rig.main.visible
    assert _alpha(rig.main.colormap, 4) == 0
    _end(rig)


def test_recolour_while_hidden_keeps_one_cell_views_hidden(rig):
    _isolate(rig, [1])
    _toggle(rig)
    proxy = rig.w._isolation_proxy_layer
    assert proxy is not None and not proxy.visible
    _recolour(rig)
    assert not proxy.visible
    assert all(not lyr.visible for lyr in _sides(rig))
    _toggle(rig)
    assert proxy.visible
    assert all(lyr.visible for lyr in _sides(rig))
    _end(rig)


def test_a_stray_show_of_main_during_one_cell_isolation_draws_only_that_cell(rig):
    _isolate(rig, [1])
    try:
        rig.main.visible = True
        assert _alpha(rig.main.colormap, 4) == 0 or (
            rig.main.colormap.use_selection
            and rig.main.colormap.selection == 1)
        assert _alpha(rig.main.colormap, 1) > 0
    finally:
        rig.main.visible = True
        _end(rig)


# --------------------------------------------------------------------------- #
# the Add candidate's three side copies (real ViewerModels, offscreen)
# --------------------------------------------------------------------------- #
def test_add_candidate_side_copies_are_healed(monkeypatch):
    """Hide the Add result with the labels key, press None, scroll the XY
    view, show it again: the repair after the show covers the candidate's
    three side copies, each with its own view's Qt viewer.

    Real: the ``ViewerModel`` objects, the ``Labels`` layers, the subviewer
    and sync managers, the labels-key toggle and ``_heal_label_render``.
    Not real: the Qt viewers (no GL context offscreen; stand-ins), so the
    test records what reaches the repair instead of a vispy node."""
    from packages.epicycle_annotator import labels_render
    from packages.epicycle_annotator import add_preview as AP

    from .add_harness import install_fakes
    from .add_layer_rig import LayerRig, blocks, result_for

    install_fakes(monkeypatch)
    shape = (12, 40, 40)
    h = LayerRig(np.zeros(shape, np.int32), ndisplay=3)
    h.adv.preset = "cells"
    roi = tuple(slice(0, s) for s in shape)
    h.place_box(roi)
    h._add_click_markers = []
    res = blocks(shape, 4, first=100, block=8)
    out = result_for(roi, res, edge=(100, 101))
    with contextlib.redirect_stdout(io.StringIO()):
        h._apply_add_result(out)
        h._setup_channel_keybindings()
    cand = h._add_candidate_layer
    sides = h.side_copies()
    assert cand is not None and all(s is not None for s in sides)
    fn = None
    for kb, f in h.viewer.keymap.items():
        if str(kb).lower() == "a":
            fn = f
    assert fn is not None

    with contextlib.redirect_stdout(io.StringIO()):
        fn(h.viewer)                           # hide
    assert not cand.visible and not any(s.visible for s in sides)
    with contextlib.redirect_stdout(io.StringIO()):
        h.adv.tick([])                         # None while hidden
        cs = list(h.viewer_model_xy.dims.current_step)
        cs[0] = (cs[0] + 3) % shape[0]
        h.viewer_model_xy.dims.current_step = tuple(cs)

    calls = []
    real = labels_render.heal_labels_render

    def spy(pairs, *, tag=""):
        pairs = list(pairs)
        calls.append((tag, pairs))
        return real(pairs, tag=tag)

    monkeypatch.setattr(labels_render, "heal_labels_render", spy)
    with contextlib.redirect_stdout(io.StringIO()):
        fn(h.viewer)                           # show
    assert cand.visible and all(s.visible for s in sides)
    got = [p for tag, ps in calls if tag == "channel-toggle" for p in ps]
    assert got, "the show branch never repaired the render"
    for side, qv in zip(sides, (h.qt_viewer_xy, h.qt_viewer_xz,
                                h.qt_viewer_yz)):
        mine = [q for lyr, q in got if lyr is side]
        assert mine == [qv], (side.name, mine)
    # and the None state is what the side copies show: every object faint
    for s in sides:
        for i in out.ids:
            assert float(s.colormap.color_dict[int(i)][3]) == pytest.approx(
                AP.ADD_UNTICKED_ALPHA)
    with contextlib.redirect_stdout(io.StringIO()):
        h._remove_add_candidate()


def test_the_repair_reaches_no_other_side_layer(monkeypatch):
    """Only the Add result's own side copies join the repair: a Refine
    layer (or an isolation layer) of a side view passed with the toggle's
    targets is left to its own code, exactly as before -- the repair sees
    the active labels' side copies, each with its own view."""
    from napari.layers import Labels

    from packages.epicycle_annotator import labels_render

    from .add_harness import install_fakes
    from .add_layer_rig import LayerRig, blocks, result_for

    install_fakes(monkeypatch)
    shape = (12, 40, 40)
    h = LayerRig(np.zeros(shape, np.int32), ndisplay=3)
    vms = (h.viewer_model_xy, h.viewer_model_xz, h.viewer_model_yz)
    qts = (h.qt_viewer_xy, h.qt_viewer_xz, h.qt_viewer_yz)
    copies = []
    for vm, o in zip(vms, ("XY", "XZ", "YZ")):
        lyr = Labels(np.zeros((12, 40), np.int32), name="L_" + o)
        vm.layers.append(lyr)
        copies.append(lyr)
    refine = Labels(np.zeros((12, 40), np.int32), name="Refine_XY")
    h.viewer_model_xy.layers.append(refine)
    calls = []
    real = labels_render.heal_labels_render

    def spy(pairs, *, tag=""):
        pairs = list(pairs)
        calls.append(pairs)
        return real(pairs, tag=tag)

    monkeypatch.setattr(labels_render, "heal_labels_render", spy)
    want = list(zip(copies, qts))

    def _got():
        return [(lyr, q) for lyr, q in calls[-1]]

    h._heal_label_render(None, tag="channel-toggle",
                         extra_layers=[refine] + copies)
    assert [(a.name, b) for a, b in _got()] == [
        (a.name, b) for a, b in want]
    # with an Add result up its copies join; Refine still does not
    h.adv.preset = "cells"
    roi = tuple(slice(0, s) for s in shape)
    h.place_box(roi)
    h._add_click_markers = []
    with contextlib.redirect_stdout(io.StringIO()):
        h._apply_add_result(result_for(
            roi, blocks(shape, 4, first=100, block=8)))
    sides = h.side_copies()
    h._heal_label_render(None, tag="channel-toggle",
                         extra_layers=[refine] + copies + sides)
    names = [a.name for a, _b in _got()]
    assert "Refine_XY" not in names
    assert sorted(names) == sorted([c.name for c in copies]
                                   + [s.name for s in sides])
    with contextlib.redirect_stdout(io.StringIO()):
        h._remove_add_candidate()

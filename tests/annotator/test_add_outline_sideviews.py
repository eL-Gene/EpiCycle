"""The tissue outline in the three side views (Add page, Cells preset).

"Show the tissue outline" draws the outline's skin in the main view and, as
a copy, in each of the XY / XZ / YZ side views, so scrolling the slices
shows where the tissue edge was drawn.  Pinned here, through the real
orchestrator on the Qt-free harness with a real subviewer manager over three
real napari viewer models:

* the copies exist in all three views while Show is ticked and are views
  of the same array, never new volumes; their look is their own (cyan with
  a clear zero, half opacity, painted over the slice, so the outline shows
  on bright stain), while the main view keeps its faint additive cyan;
* one drop of the outline pays napari's garbage collection once;
* the draw paints "Drawing the tissue outline…" before it holds the main
  thread, and the legend names the side views only when all three hold it;
* they hide with the main layer while candidates are up;
* they go (hidden first, before the main layer) on untick, No outline,
  Cancel, Accept, "+ Add another" (and come back with it), Done, a patch
  switch, a run switch, a close, a sweep of the temporary layers -- and on
  an exit that finds the main layer already gone;
* Stop and a Clear never leave a copy behind that the main view lacks;
* they are Image layers: never a labels-key target, never a labels-sync
  member, never brush-locked;
* the legend under the tick says where it is drawn.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from napari.components import ViewerModel
from napari.layers import Image, Labels

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.subviewers import SubviewerManager
from napari_epicycle.annotator import curation_3d_widget as CW
from napari_epicycle.annotator import native_multiview_widget as NM

from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_outline_wiring import _land_info, _show, h  # noqa: F401

NAME = "guide_add_outline"
ORIENTS = ("XY", "XZ", "YZ")


@pytest.fixture
def hs(h):  # noqa: F811 - the outline harness, with real side views
    """``h`` with three real side viewer models behind the real subviewer
    manager, and a temp-layer stub that builds a real Image layer."""
    vms = {o: ViewerModel() for o in ORIENTS}
    shape = h.sync_manager.shared_labels["L"].shape
    for o, vm in vms.items():
        # the main labels' side copy, the layer the side views paint on
        vm.add_labels(np.zeros(shape, np.int32), name=f"L_{o}")
    h.viewer_model_xy = vms["XY"]
    h.viewer_model_xz = vms["XZ"]
    h.viewer_model_yz = vms["YZ"]
    h.subviewer_manager = SubviewerManager(
        vms["XY"], vms["XZ"], vms["YZ"], None, None, None)
    h.vms = vms
    h.log = []

    def _temp(key, array, *, kind="labels", name=None, image_kw=None,
              **_placement):
        kw = dict(image_kw or {})
        visible = kw.pop("visible", True)
        layer = Image(array, name=name, visible=visible, **kw)
        layer._epicycle_temporary = True
        h.viewer.layers.append(layer)
        layer.visible = True                   # built hidden, then shown
        h.added.append((key, array, dict(kind=kind, image_kw=image_kw),
                        layer))
        h.log.append(("main_added", key))
        return layer

    def _remove(key):
        h.removed.append(key)
        h.log.append(("main_removed", key, len(_copies(h))))
        for layer in list(h.viewer.layers):
            if getattr(layer, "name", None) == NAME and key == "add_outline":
                layer.visible = False
                h.viewer.layers.remove(layer)

    h._add_temp_layer = _temp
    h._remove_temp_layer = _remove
    for o, vm in vms.items():
        vm.layers.events.removed.connect(
            lambda e, o=o: h.log.append(
                ("side_removed", o, e.value.name, bool(e.value.visible))))
    return h


def _copies(hh):
    out = {}
    for o, vm in hh.vms.items():
        for layer in vm.layers:
            if layer.name == f"{NAME}_{o}":
                out[o] = layer
    return out


def _assert_none(hh, why):
    assert _copies(hh) == {}, why
    assert hh._add_outline_layer is None, why


def _assert_drawn(hh):
    main = hh._add_outline_layer
    assert main is not None
    copies = _copies(hh)
    assert set(copies) == set(ORIENTS), copies
    return main, copies


def _assert_hidden_then_removed_side_first(hh):
    """Every side copy was hidden when it was removed, and the side copies
    left before the main layer did.

    The "hidden" half is what the subviewer manager's removal does itself
    (it hides each side layer before removing it); the page's own hide
    loop before that call is defensive and is not what this proves.  The
    "side first" half is the page's (main-first fails eight exits)."""
    side = [e for e in hh.log if e[0] == "side_removed" and e[2].startswith(
        NAME)]
    assert {e[1] for e in side} == set(ORIENTS), hh.log
    assert all(e[3] is False for e in side), "hidden before removal"
    mains = [e for e in hh.log if e[0] == "main_removed"
             and e[1] == "add_outline"]
    for m in mains:
        assert m[2] == 0, "the side copies go before the main layer"


def _preview(hh):
    hh._preview_add()
    th = hh._add_thread
    th.run()
    th.end()
    return hh._add_preview_res


def _accept(hh):
    res = _preview(hh)
    assert res is not None and res.ok, getattr(res, "refusal", None)
    hh.adv.checked = list(res.ids)
    hh._on_add_committed_cb = None
    hh._accept_add()
    assert hh._add_review


# --------------------------------------------------------------------------- #
# drawn in all three side views
# --------------------------------------------------------------------------- #
def test_show_draws_a_copy_in_each_side_view(hs):
    _show(hs)
    main, copies = _assert_drawn(hs)
    shell = hs._add_cached_outline(shell=True)
    for o, sub in copies.items():
        assert isinstance(sub, Image) and not isinstance(sub, Labels)
        assert sub.data.dtype == np.uint8
        assert tuple(sub.contrast_limits) == (0.0, 1.0)
        assert sub.visible is True
        assert getattr(sub, "_epicycle_temporary", False)
        # the same array, seen along the view's own axis -- not a copy
        assert np.shares_memory(sub.data, shell), o
        assert sub.data.base is shell or sub.data is shell, o
    # each view slices the volume along its own axis (the standard
    # side-view transposes)
    assert copies["XY"].data.shape == shell.shape
    assert copies["XZ"].data.shape == (shell.shape[1], shell.shape[0],
                                       shell.shape[2])
    assert copies["YZ"].data.shape == (shell.shape[2], shell.shape[1],
                                       shell.shape[0])
    assert np.array_equal(copies["YZ"].data, shell.transpose(2, 1, 0))
    # one copy per view, and a refresh adds none
    n = len(hs.added)
    hs._add_refresh_ui()
    assert len(hs.added) == n
    for o, vm in hs.vms.items():
        assert sum(layer.name == f"{NAME}_{o}" for layer in vm.layers) == 1


def test_the_side_copies_paint_cyan_over_the_slice_and_the_main_view_keeps_its_tint(
        hs):
    """In a slice the outline sits on the tissue's outermost voxels, often
    bright stain, where an additive 0.3 cyan clips to white.  The side
    copies paint cyan over the slice at half opacity with a clear zero;
    the main view (3D, over a dark background) keeps its faint additive
    cyan."""
    _show(hs)
    main, copies = _assert_drawn(hs)
    assert abs(main.opacity - 0.3) < 1e-9
    assert str(main.blending) == "additive"
    assert main.colormap.name == "cyan"
    for o, sub in copies.items():
        assert abs(sub.opacity - 0.5) < 1e-9, o
        assert str(sub.blending) == "translucent", o
        rgba = np.asarray(sub.colormap.map(np.array([0.0, 1.0])))
        assert rgba[0][3] == 0.0, "zero is transparent"
        assert np.allclose(rgba[1], [0.0, 1.0, 1.0, 1.0]), "one is cyan"
        # the main layer's own colormap is untouched by the side look
        assert main.colormap.name == "cyan"
    # over a bright slice the outline's pixels move away from white
    bright = np.array([0.95, 0.95, 0.95])
    cyan = np.array([0.0, 1.0, 1.0])
    over = 0.5 * cyan + 0.5 * bright
    added = np.clip(bright + 0.3 * cyan, 0.0, 1.0)
    assert np.abs(over - bright).sum() > 3 * np.abs(added - bright).sum()


def test_a_drop_pays_naparis_garbage_collection_once(monkeypatch):
    """The side copies' removal and the main layer's removal nest in one
    batch, so one drop runs one real collection (two before)."""
    import napari._vispy.canvas as canvas_mod

    from packages.epicycle_annotator import layer_batch as LB

    real = {"n": 0}

    class _Counting:
        @staticmethod
        def collect(*_a, **_k):
            real["n"] += 1
            return 0

    monkeypatch.setattr(LB, "_real_gc", _Counting())
    shape = (12, 40, 40)
    main_vm = ViewerModel()
    vms = [ViewerModel() for _ in ORIENTS]
    # napari's canvas runs a collection at every layer removal; the batch
    # swaps that module attribute for a counter
    for vm in [main_vm] + vms:
        vm.layers.events.removed.connect(
            lambda _e: canvas_mod.gc.collect())
    w = types.SimpleNamespace(
        viewer=main_vm, subviewer_manager=SubviewerManager(
            *vms, None, None, None),
        sync_manager=None, _active_labels_name="", _temp_guide_layers={})
    w._layer_by_name = lambda layers, name: next(
        (la for la in layers if la.name == name), None)
    for n in ("_add_drop_outline_layer", "_add_drop_outline_side",
              "_add_outline_side_copies", "_remove_temp_layer"):
        setattr(w, n, types.MethodType(
            getattr(CW.Curation3DAnnotator, n), w))

    def draw():
        m = Image(np.zeros(shape, np.uint8), name=NAME)
        m._epicycle_temporary = True
        main_vm.layers.append(m)
        w._temp_guide_layers["add_outline"] = m
        w._add_outline_layer = m
        w._add_outline_layer_key = ("k",)
        w.subviewer_manager.add_image_to_subviewers(m)

    draw()
    real["n"] = 0
    w._add_drop_outline_layer()
    assert real["n"] == 1, real
    assert [la.name for la in main_vm.layers] == []
    assert all(len(vm.layers) == 0 for vm in vms)
    # teeth: two separate drops are two collections (the counter counts)
    draw()
    real["n"] = 0
    w._add_drop_outline_side()
    w._remove_temp_layer("add_outline")
    assert real["n"] == 2, real


def test_the_draw_paints_its_words_before_it_holds_the_main_thread(hs):
    """With the skin already measured (a re-tick), nothing else is on
    screen while the layers are added: S1 and the legend say "Drawing the
    tissue outline…" and are painted first; the page's refresh then
    takes the words back."""
    _show(hs)
    hs.adv.show.on = False
    hs._on_add_show_outline(False)
    _assert_none(hs, "untick")
    painted = []

    class _Label:
        def __init__(self, name):
            self.name = name

        def repaint(self):
            painted.append(self.name)

    hs.adv.add_status_label = _Label("s1")
    hs.adv._add_show_outline_legend = _Label("legend")
    seen = {}
    temp = hs._add_temp_layer

    def _spy(key, array, **kw):
        last = [p for p in hs.adv.pushes if "status_text" in p][-1]
        seen["s1"] = last["status_text"]
        seen["kind"] = last.get("status_kind")
        seen["drawing"] = hs.adv.last("outline_drawing")
        seen["painted"] = list(painted)
        return temp(key, array, **kw)

    hs._add_temp_layer = _spy
    hs.adv.show.on = True
    hs._on_add_show_outline(True)            # cached: drawn at once
    _assert_drawn(hs)
    assert seen["s1"] == AP.ADD_OUTLINE_DRAWING
    assert seen["kind"] == "busy"
    assert seen["drawing"] is True
    assert seen["painted"] == ["s1", "legend"]
    # after: the words are lifted and S1 is the page's own again
    assert hs.adv.last("outline_drawing") is False
    assert hs.adv.last("status_text") != AP.ADD_OUTLINE_DRAWING
    # a refresh with nothing to draw paints nothing
    painted.clear()
    hs._add_refresh_ui()
    assert painted == []


def test_the_legend_names_the_side_views_only_when_all_three_hold_it(hs):
    _show(hs)
    assert hs.adv.last("outline_in_side_views") is True
    hs._add_drop_outline_layer()

    def _boom(_layer):
        raise RuntimeError("no side views")

    hs.subviewer_manager.add_image_to_subviewers = _boom
    hs._add_refresh_ui()
    assert hs._add_outline_layer is not None, "the main view still has it"
    assert _copies(hs) == {}
    assert hs.adv.last("outline_drawn") is True
    assert hs.adv.last("outline_in_side_views") is False


def test_a_partial_set_of_side_copies_is_dropped(hs):
    """Two of three is not three: the partial copies go, so the legend can
    say "main view only" truthfully."""
    _show(hs)
    hs._add_drop_outline_layer()
    real_add = hs.subviewer_manager.add_image_to_subviewers

    def _two(layer):
        real_add(layer)
        hs.vms["YZ"].layers.remove(f"{NAME}_YZ")

    hs.subviewer_manager.add_image_to_subviewers = _two
    hs._add_refresh_ui()
    assert _copies(hs) == {}
    assert hs.adv.last("outline_in_side_views") is False


def test_the_side_views_keep_their_labels_copy_active(hs):
    _show(hs)
    for o, vm in hs.vms.items():
        assert vm.layers.selection.active.name == f"L_{o}", o


def test_the_outline_stays_on_screen_over_candidates_in_every_view(hs):
    # a result is on screen after every click with "Update as I click", and
    # the outline is what the cells grow to: it is never hidden behind them
    _show(hs)
    main, copies = _assert_drawn(hs)
    hs._add_preview_result = np.zeros((1, 1, 1), np.int32)
    hs._add_refresh_ui()
    assert main.visible is True
    assert all(c.visible is True for c in copies.values())
    hs._add_preview_result = None
    hs._add_refresh_ui()
    assert main.visible is True
    assert all(c.visible is True for c in _copies(hs).values())


def test_a_new_outline_replaces_the_copies_never_stacks_them(hs):
    _show(hs)
    first = _copies(hs)
    # another outline (its key changed): dropped, then drawn again
    hs._add_drop_outline_layer()
    hs._add_refresh_ui()
    second = _copies(hs)
    assert set(second) == set(ORIENTS)
    assert all(second[o] is not first[o] for o in ORIENTS)
    for o, vm in hs.vms.items():
        assert sum(layer.name == f"{NAME}_{o}" for layer in vm.layers) == 1


# --------------------------------------------------------------------------- #
# every exit removes them, hidden first, before the main layer
# --------------------------------------------------------------------------- #
def test_untick_removes_the_copies(hs):
    _show(hs)
    hs.log.clear()
    hs.adv.show.on = False
    hs._on_add_show_outline(False)
    _assert_none(hs, "untick")
    _assert_hidden_then_removed_side_first(hs)


def _close_cleanup(hh):
    """The annotator's real close teardown on the harness; the steps that
    need a live Qt viewer (event disconnects, arrow keys) are stubbed."""
    for name in ("_disconnect_viewer_events", "_disconnect_label_handlers",
                 "_restore_main_viewer_arrow_keys",
                 "_abandon_seed_find_worker", "_abandon_reshape_worker"):
        setattr(hh, name, lambda *a, **k: None)
    for name in ("_smooth_running", "_split_running"):
        setattr(hh, name, False)
    hh._interp_snapshot = None
    hh._on_isolation_cleared_cb = None
    hh._navigator_dock = None
    hh._navigator_widget = None
    try:
        CW.Curation3DAnnotator._close_cleanup(hh)
    except Exception:  # noqa: BLE001 - the harness is too bare to finish
        pass          # the Add teardown runs first, inside its own guard


@pytest.mark.parametrize("exit_", ["outline_none", "cancel", "switch",
                                   "close", "run_switch", "sweep"])
def test_every_exit_removes_the_copies(hs, exit_, monkeypatch):
    _show(hs)
    _assert_drawn(hs)
    hs.log.clear()
    if exit_ == "outline_none":
        hs.adv.params["outline"] = "none"
        hs._on_add_outline_changed("none")
    elif exit_ == "cancel":
        hs._reject_add()
    elif exit_ == "switch":
        hs._teardown_add_for_switch()
    elif exit_ == "close":
        # the annotator's close with the box up: the real teardown
        assert hs._add_running
        _close_cleanup(hs)
    elif exit_ == "run_switch":
        # a run switch closes the annotator through the launcher's widget
        # cleanup, which runs the same close teardown
        from napari_epicycle.annotator import launcher as LA

        monkeypatch.setattr(LA, "_schedule_close_orphan_sweep_retries",
                            lambda: None)
        hs._close_cleanup = lambda: _close_cleanup(hs)
        LA._run_widget_cleanup(hs)
    else:
        hs._temp_guide_layers = {}
        CW.Curation3DAnnotator._remove_all_temp_layers(hs)
    assert _copies(hs) == {}, exit_
    side = [e for e in hs.log if e[0] == "side_removed"
            and e[2].startswith(NAME)]
    assert {e[1] for e in side} == set(ORIENTS), exit_
    assert all(e[3] is False for e in side), "hidden before removal"
    if exit_ != "sweep":
        _assert_none(hs, exit_)
        _assert_hidden_then_removed_side_first(hs)


def test_accept_removes_the_copies_and_add_another_draws_them_again(hs):
    _show(hs)
    hs.log.clear()
    _accept(hs)
    _assert_none(hs, "Accept")
    _assert_hidden_then_removed_side_first(hs)
    assert hs.adv.show.on is True, "Accept keeps the tick"
    hs._add_another()
    _land_info(hs)
    hs._add_refresh_ui()
    _assert_drawn(hs)
    # and leaving that box ends it in every view again
    hs.log.clear()
    hs._reject_add()
    _assert_none(hs, "Cancel after + Add another")
    _assert_hidden_then_removed_side_first(hs)


def test_done_removes_a_copy_left_behind_by_a_main_layer_already_gone(hs):
    """Defensive: the copies are looked up by name, so an exit finds them
    even when the main layer's references were already dropped."""
    _show(hs)
    _accept(hs)
    _assert_none(hs, "Accept")
    # a stray set of copies with no main layer (the references gone)
    stray = Image(np.zeros((12, 40, 40), np.uint8), name=NAME)
    hs.subviewer_manager.add_image_to_subviewers(stray)
    assert set(_copies(hs)) == set(ORIENTS)
    assert hs._add_outline_layer is None
    hs.log.clear()
    hs._add_primary()                         # Done (REVIEW)
    assert not hs._add_review
    assert _copies(hs) == {}
    side = [e for e in hs.log if e[0] == "side_removed"]
    assert {e[1] for e in side} == set(ORIENTS)
    assert all(e[3] is False for e in side), "hidden before removal"


def test_the_drop_with_nothing_drawn_touches_no_view(hs):
    hs.log.clear()
    hs._add_drop_outline_layer()
    hs._add_drop_outline_layer()
    assert hs.log == []
    for o, vm in hs.vms.items():
        assert [layer.name for layer in vm.layers] == [f"L_{o}"]


def test_stop_leaves_no_copy_the_main_view_lacks(hs):
    _show(hs)
    hs._preview_add()
    th = hs._add_thread
    hs._stop_add_preview()
    th.run()
    th.end()
    hs._add_refresh_ui()
    assert hs._add_preview_res is None, "stopped: nothing from this box"
    assert hs._add_transient == AP.ADD_S1_STOPPED
    # Stop is not an exit: the outline stays in every view, as it was
    main = hs._add_outline_layer
    assert main is not None
    copies = _copies(hs)
    assert set(copies) == set(ORIENTS)
    assert main.visible is True
    assert all(c.visible is True for c in copies.values())


def _clear(hs, zyx_slices, label_id):
    labels = hs.sync_manager.shared_labels["L"]
    labels[zyx_slices] = label_id
    hs._apply_add_clear(AP.clear_plan(
        labels=labels, mode="box",
        box=(slice(0, 12), slice(0, 20), slice(0, 20))))
    assert not (labels == label_id).any(), "cleared"
    hs._add_transient = None
    _land_info(hs)
    hs._add_refresh_ui()


def test_a_clear_under_the_shrink_wrap_keeps_the_same_copies(hs):
    """The shrink-wrap reads the stain only: a Clear does not change it, so
    the outline and its three copies stay as they are."""
    _show(hs)
    before = _copies(hs)
    _clear(hs, (slice(2, 6), slice(2, 8), slice(2, 8)), 7)
    after = _copies(hs)
    assert set(after) == set(ORIENTS)
    assert all(after[o] is before[o] for o in ORIENTS)


def test_a_clear_under_the_cleared_outline_redraws_without_stacking(hs):
    """On the outline of the objects you cleared, a second Clear changes
    the outline: the old copies go (hidden first) and one new copy per view
    is drawn."""
    hs.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    _clear(hs, (slice(2, 6), slice(2, 8), slice(2, 8)), 7)
    hs.adv.params["outline"] = "cleared"
    hs._on_add_outline_changed("cleared")
    hs.adv.show.on = True
    hs._on_add_show_outline(True)
    _land_info(hs)
    hs._add_refresh_ui()
    before = _copies(hs)
    assert set(before) == set(ORIENTS)
    hs.log.clear()
    _clear(hs, (slice(6, 10), slice(10, 16), slice(10, 16)), 9)
    _land_info(hs)
    hs._add_refresh_ui()
    after = _copies(hs)
    assert set(after) == set(ORIENTS)
    assert all(after[o] is not before[o] for o in ORIENTS), "redrawn"
    for o, vm in hs.vms.items():
        assert sum(layer.name == f"{NAME}_{o}" for layer in vm.layers) == 1
    _assert_hidden_then_removed_side_first(hs)


# --------------------------------------------------------------------------- #
# an Image layer: no labels key, no labels sync, no brush lock
# --------------------------------------------------------------------------- #
def test_the_copies_are_never_a_labels_key_target(hs):
    _show(hs)
    copies = list(_copies(hs).values())
    # the labels key while Add runs: the base targets plus Add's own
    hs._active_labels_names = ["L"]
    hs._orientation_for_vm = types.MethodType(
        NM.NativeMultiviewAnnotator._orientation_for_vm, hs)
    base = NM.NativeMultiviewAnnotator._get_toggle_targets_for_name(hs, "L")
    targets = list(base) + list(hs._add_toggle_targets())
    for sub in copies:
        assert all(t is not sub for t in targets)
    assert all(t is not hs._add_outline_layer for t in targets)


def test_the_copies_are_image_layers_never_brush_locked(hs):
    """Image layers: the brush lock is never put on them, and the label
    sync never sees them (every side-view sync loop takes Labels layers
    only, and the image path of the subviewer manager never registers a
    layer with the sync)."""
    _show(hs)
    for o, sub in _copies(hs).items():
        assert isinstance(sub, Image) and not isinstance(sub, Labels)
        assert not getattr(sub, "_epicycle_brush_locked", False)


# --------------------------------------------------------------------------- #
# the words
# --------------------------------------------------------------------------- #
def test_the_legend_says_it_is_drawn_in_the_side_views(widget):  # noqa: F811
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    assert AP.ADD_HELP_SHOW_OUTLINE == (
        "Drawn in the main view and the three side views.")
    assert "side views" in ADD_HELP_LINES["add_show_outline_cb"]
    assert "not in the" not in ADD_HELP_LINES["add_show_outline_cb"]
    reset_add(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(box_active=True)
    widget.add_outline_combo.setCurrentIndex(
        AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
    widget.add_show_outline_cb.setChecked(True)
    try:
        widget.update_add_ui_state(outline_pending=False, outline_drawn=True)
        assert widget._add_show_outline_legend.text() == (
            AP.ADD_HELP_SHOW_OUTLINE)
        assert widget.add_show_outline_cb.toolTip() == ADD_HELP_LINES[
            "add_show_outline_cb"]
        # while the layers are added
        widget.update_add_ui_state(outline_drawing=True)
        assert widget._add_show_outline_legend.text() == (
            AP.ADD_OUTLINE_DRAWING)
        # the side views could not take it: the legend says main only
        widget.update_add_ui_state(outline_drawing=False,
                                   outline_in_side_views=False)
        assert widget._add_show_outline_legend.text() == (
            AP.ADD_SHOW_OUTLINE_MAIN_ONLY)
        widget.update_add_ui_state(outline_in_side_views=True)
        assert widget._add_show_outline_legend.text() == (
            AP.ADD_HELP_SHOW_OUTLINE)
    finally:
        widget.update_add_ui_state(outline_drawing=False,
                                   outline_in_side_views=True)
        widget.add_show_outline_cb.setChecked(False)
        reset_add(widget)


def test_a_page_refresh_does_not_show_the_layers_again(hs):
    """napari's ``visible = True`` refreshes the layer every time; a page
    refresh with nothing changed sets no visibility, in any view."""
    _show(hs)
    main, copies = _assert_drawn(hs)
    fired = []
    for layer in [main] + list(copies.values()):
        layer.events.visible.connect(lambda e, n=layer.name: fired.append(n))
    hs._add_refresh_ui()
    hs._add_refresh_ui()
    assert fired == []
    # a result on screen changes nothing either: the outline stays drawn
    hs._add_preview_result = np.zeros((1, 1, 1), np.int32)
    hs._add_refresh_ui()
    assert fired == []
    # a real change (hidden by hand) still reaches all four
    for layer in [main] + list(copies.values()):
        layer.visible = False
    fired.clear()
    hs._add_refresh_ui()
    assert len(fired) == 4

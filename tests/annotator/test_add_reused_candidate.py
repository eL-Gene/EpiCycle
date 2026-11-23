"""The Add candidate is made once per box and redrawn in place.

The first Preview of a box builds the candidate ``Labels`` layer and its
three side-view copies (transposed views of the same array).  Every later
Preview writes only the changed voxels into that SAME array -- the old
result's box zeroed, the new one written -- gives the layer its two fresh
colour maps and shows it.  An input change (a click, a threshold, a preset)
zeros the drawn result and hides the four layers; it no longer removes
them.  They go only at Close box, Accept, Discard, a patch switch and close.

* the second render keeps the four layer objects and appends nothing;
* what is drawn is what a fresh build of the same result draws (the array
  and both colour maps), for a box that moved and one that shrank;
* no ``.data`` reassignment on any of the four after the build, no undo
  history;
* invalidation zeros and hides, the page and the labels key read "drawn",
  not "the layer exists";
* the read-only lock holds after redraws, hides and the toggles;
* every exit still removes the four.

Real napari layers through ``add_layer_rig``.
"""

from __future__ import annotations

import contextlib
import io
import types

import numpy as np
import pytest

from napari_epicycle.annotator import curation_3d_widget as CW
from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.orientation import transpose_data

from .add_harness import install_fakes
from .add_layer_rig import LayerRig, blocks, result_for

SHAPE = (12, 40, 40)
NAME = "_epicycle_add_candidate"

ROI_A = (slice(0, 12), slice(0, 24), slice(0, 24))
ROI_MOVED = (slice(0, 12), slice(16, 40), slice(16, 40))
ROI_SHRUNK = (slice(2, 10), slice(4, 16), slice(4, 16))


def _new_rig():
    vol = np.zeros(SHAPE, np.int32)
    h = LayerRig(vol, ndisplay=3)
    h.adv.preset = "cells"
    h.place_box(tuple(slice(0, s) for s in SHAPE))
    h._add_click_markers = []
    h._add_markers_base = 500
    return h


def _drop(h):
    with contextlib.suppress(Exception), contextlib.redirect_stdout(
            io.StringIO()):
        h._remove_add_candidate()


@pytest.fixture
def rig(monkeypatch):
    install_fakes(monkeypatch)
    h = _new_rig()
    yield h
    _drop(h)


def _res(roi, first, n, block, edge=()):
    shape = tuple(s.stop - s.start for s in roi)
    return result_for(roi, blocks(shape, n, first=first, block=block),
                      edge=edge)


def _a():
    return _res(ROI_A, 100, 4, 6, edge=(100,))


def _moved():
    return _res(ROI_MOVED, 200, 3, 6, edge=(201,))


def _shrunk():
    return _res(ROI_SHRUNK, 300, 2, 4)


def _box():
    """A result on the whole placed box (Accept refuses any other roi)."""
    return _res(tuple(slice(0, s) for s in SHAPE), 400, 4, 8)


def _apply(h, out):
    with contextlib.redirect_stdout(io.StringIO()):
        h._apply_add_result(out)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _four(h):
    return [h._add_candidate_layer] + h.side_copies()


def _expected(out):
    full = np.zeros(SHAPE, np.int32)
    full[tuple(out.roi)] = out.result
    return full


def _cmap_equal(a, b):
    da, db = dict(a.color_dict), dict(b.color_dict)
    assert set(da) == set(db)
    for k in da:
        np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]))


# --------------------------------------------------------------------------- #
# (a) the same four objects, nothing appended
# --------------------------------------------------------------------------- #
def test_a_second_render_keeps_the_four_layers_and_appends_nothing(rig):
    _apply(rig, _a())
    first = _four(rig)
    assert all(x is not None for x in first)
    vms = (rig.viewer, rig.viewer_model_xy, rig.viewer_model_xz,
           rig.viewer_model_yz)
    lengths = [len(vm.layers) for vm in vms]
    inserted = []
    for vm in vms:
        vm.layers.events.inserted.connect(lambda e: inserted.append(e))
    _quiet(rig._invalidate_add_preview)
    _apply(rig, _moved())
    _apply(rig, _shrunk())
    again = _four(rig)
    assert all(x is y for x, y in zip(again, first))
    assert [len(vm.layers) for vm in vms] == lengths
    assert inserted == []


def test_the_second_render_line_says_diff(rig, capsys):
    capsys.readouterr()
    _apply_loud = rig._apply_add_result
    _apply_loud(_a())
    text = capsys.readouterr().out
    assert any(ln.startswith("[ADD-PERF] render ") and "path=build" in ln
               for ln in text.splitlines()), text
    _quiet(rig._invalidate_add_preview)
    capsys.readouterr()
    _apply_loud(_moved())
    text = capsys.readouterr().out
    lines = [ln for ln in text.splitlines()
             if ln.startswith("[ADD-PERF] render ")]
    assert len(lines) == 1 and "path=diff" in lines[0], text
    assert not any(ln.startswith("[ADD-PERF] remove ")
                   for ln in text.splitlines())


# --------------------------------------------------------------------------- #
# (b) what is drawn equals a fresh build
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("second", [_moved, _shrunk],
                         ids=["moved", "shrunk"])
@pytest.mark.parametrize("hide_between", [True, False],
                         ids=["after_a_click", "straight"])
def test_the_redraw_equals_a_fresh_build(monkeypatch, rig, second,
                                         hide_between):
    _apply(rig, _a())
    if hide_between:
        _quiet(rig._invalidate_add_preview)
    out = second()
    _apply(rig, out)
    fresh = _new_rig()
    try:
        _apply(fresh, second())
        want = _expected(out)
        cand = rig._add_candidate_layer
        np.testing.assert_array_equal(np.asarray(cand.data), want)
        np.testing.assert_array_equal(
            np.asarray(fresh._add_candidate_layer.data), want)
        for o, mine, theirs in zip(("XY", "XZ", "YZ"), rig.side_copies(),
                                   fresh.side_copies()):
            np.testing.assert_array_equal(np.asarray(mine.data),
                                          transpose_data(want, o))
            _cmap_equal(mine.colormap, theirs.colormap)
        _cmap_equal(cand.colormap, fresh._add_candidate_layer.colormap)
        assert all(x.visible for x in _four(rig))
        assert rig._add_drawn_roi == tuple(out.roi)
        assert rig._add_colours_token == fresh._add_colours_token
    finally:
        _drop(fresh)


def test_the_side_copies_stay_views_of_the_one_array(rig):
    _apply(rig, _a())
    _apply(rig, _moved())
    base = rig._add_candidate_layer.data
    for s in rig.side_copies():
        assert np.shares_memory(np.asarray(s.data), base)


# --------------------------------------------------------------------------- #
# (c) no .data reassignment after the build; (d) no history
# --------------------------------------------------------------------------- #
def test_the_data_setter_is_never_called_after_the_build(monkeypatch, rig):
    from napari.layers import Labels

    _apply(rig, _a())
    four = {id(x) for x in _four(rig)}
    real = Labels.__dict__["data"]
    sets = []

    def _set(self, value):
        sets.append(id(self))
        real.fset(self, value)

    monkeypatch.setattr(Labels, "data", property(real.fget, _set))
    for out in (_moved(), _shrunk(), _a()):
        _quiet(rig._invalidate_add_preview)
        _apply(rig, out)
        _apply(rig, out)                    # straight, no invalidation
    _quiet(rig._on_add_contour_toggled, True)
    _quiet(rig._on_add_contour_toggled, False)
    assert not four & set(sets)


def test_no_undo_history_on_the_candidate(rig):
    _apply(rig, _a())
    _apply(rig, _moved())
    for lyr in _four(rig):
        assert lyr._undo_history.maxlen == 0
        assert lyr._redo_history.maxlen == 0
    cand = rig._add_candidate_layer
    idx = (np.array([0]), np.array([0]), np.array([0]))
    cand._save_history((idx, np.array([0]), 7))  # what a paint would record
    assert len(cand._undo_history) == 0


# --------------------------------------------------------------------------- #
# (e) invalidation zeros and hides; the predicates read "drawn"
# --------------------------------------------------------------------------- #
def test_invalidation_zeros_and_hides_and_keeps_the_four(rig, capsys):
    _apply(rig, _a())
    first = _four(rig)
    capsys.readouterr()
    rig._invalidate_add_preview()
    text = capsys.readouterr().out
    assert _four(rig) == first
    assert first[0] in rig.viewer.layers
    assert not any(x.visible for x in first)
    assert not np.any(first[0].data)
    assert rig._add_drawn_roi is None
    assert rig._add_preview_result is None
    assert rig._add_drawn_token is None and rig._add_colours_token is None
    assert NAME in rig.sync_manager.shared_labels
    assert not any(ln.startswith("[ADD-PERF] remove ")
                   for ln in text.splitlines())
    hides = [ln for ln in text.splitlines()
             if ln.startswith("[ADD-PERF] hide ")]
    assert len(hides) == 1, text


def test_the_list_is_cleared_once_when_a_preview_was_on_screen(rig):
    _apply(rig, _a())
    rig.adv.pushes.clear()
    _quiet(rig._invalidate_add_preview)
    cleared = [kw for kw in rig.adv.pushes if "candidate_ids" in kw]
    assert cleared == [cleared[0]] and cleared[0]["candidate_ids"] == []
    rig.adv.pushes.clear()
    _quiet(rig._invalidate_add_preview)    # nothing drawn: no push
    assert [kw for kw in rig.adv.pushes if "candidate_ids" in kw] == []


def test_a_preset_switch_after_a_click_paints_no_removing_words(rig):
    _apply(rig, _a())
    _quiet(rig._invalidate_add_preview)
    rig.adv.pushes.clear()
    rig.adv.preset = "nuclei"
    _quiet(rig._on_add_preset_changed, "nuclei")
    assert AP.ADD_S1_REMOVING_RESULT not in [
        kw.get("status_text") for kw in rig.adv.pushes]


def test_the_labels_key_skips_a_hidden_candidate(rig):
    _apply(rig, _a())
    four = _four(rig)
    targets = rig._add_toggle_targets()
    assert all(any(t is x for t in targets) for x in four)
    _quiet(rig._invalidate_add_preview)
    targets = rig._add_toggle_targets()
    assert not any(t is x for t in targets for x in four)


def test_the_annotator_routes_no_undrawn_candidate(monkeypatch):
    """The real hook: after the session, a kept but undrawn candidate adds
    no targets to the labels key."""
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator,
    )

    class _L:
        def __init__(self, name):
            self.name = name

    a = CW.Curation3DAnnotator.__new__(CW.Curation3DAnnotator)
    a._active_labels_names = ["L"]
    a._add_running = False
    a._add_candidate_layer = _L(NAME)
    a._add_marker_layer = None
    a._add_drawn_roi = None
    a._refine_layer = None
    a._multi_label_proxy_layers = None
    a._isolation_proxy_layer = None
    a.subviewer_manager = None
    a.viewer = types.SimpleNamespace(
        dims=types.SimpleNamespace(ndisplay=3), layers=[])
    monkeypatch.setattr(NativeMultiviewAnnotator,
                        "_get_toggle_targets_for_name",
                        lambda self, name: ["L"])
    assert a._add_toggle_targets() == []
    assert _quiet(a._get_toggle_targets_for_name, "L") == ["L"]
    a._add_drawn_roi = ROI_A
    assert [t.name for t in a._add_toggle_targets()] == [NAME]


# --------------------------------------------------------------------------- #
# (f) the lock holds
# --------------------------------------------------------------------------- #
def _assert_locked(h):
    cand = h._add_candidate_layer
    assert cand.editable is False
    assert int(cand.brush_size) == 1
    for s in h.side_copies():
        assert s.editable is False, s.name
        assert s.mode == "pan_zoom", (s.name, s.mode)
        assert int(s.brush_size) == 1, s.name


def test_the_build_lock_still_reads_after_redraws_and_hides(rig):
    """Pins the lock the build sets, read again after redraws, hides and
    the toggles.  No path of a box session writes ``editable`` or ``mode``
    on the four layers (napari's own hide / show parks and puts back the
    same pan-zoom mode, firing no event), so this cannot catch a later
    unlock -- only a build that stops locking."""
    _apply(rig, _a())
    _assert_locked(rig)
    for out in (_moved(), _shrunk(), _a()):
        _quiet(rig._invalidate_add_preview)
        _assert_locked(rig)
        _apply(rig, out)
        _assert_locked(rig)
    _quiet(rig._on_add_place_markers_toggled, True)
    _assert_locked(rig)
    _quiet(rig._on_add_place_markers_toggled, False)
    _quiet(rig._on_add_contour_toggled, True)
    _assert_locked(rig)
    _quiet(rig._invalidate_add_preview)
    _apply(rig, _moved())
    _assert_locked(rig)
    _quiet(rig._on_add_contour_toggled, False)
    _assert_locked(rig)


def test_the_outline_stays_through_a_redraw(rig):
    import warnings

    _apply(rig, _a())
    _quiet(rig._on_add_contour_toggled, True)
    _quiet(rig._invalidate_add_preview)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            CW._add_outline_filter(False)      # only the redraw's own guard
            _apply(rig, _moved())
        assert rig._add_candidate_layer.contour == 1
        assert [w for w in rec
                if CW._CONTOUR_3D_WARNING in str(w.message)] == []
    finally:
        _quiet(rig._on_add_contour_toggled, False)
        CW._add_outline_filter(False)


# --------------------------------------------------------------------------- #
# (g) every exit still removes the four, also after a hide
# --------------------------------------------------------------------------- #
def _left_behind(h):
    out = []
    for vm in (h.viewer, h.viewer_model_xy, h.viewer_model_xz,
               h.viewer_model_yz):
        out += [lyr.name for lyr in vm.layers
                if str(lyr.name).startswith(NAME)]
    return out


def _discard(h):
    h._add_secondary()


def _accept(h):
    h._on_add_committed_cb = None
    h._accept_add()


def _close_box(h):
    h._add_preview_result = None
    h._add_secondary()


def _patch_switch(h):
    h._teardown_add_for_switch()


@pytest.mark.parametrize("leave", [_discard, _accept, _close_box,
                                   _patch_switch],
                         ids=lambda f: f.__name__.strip("_"))
def test_every_exit_removes_the_four(rig, leave):
    _apply(rig, _a())
    _quiet(rig._invalidate_add_preview)
    _apply(rig, _box())
    cand = rig._add_candidate_layer
    assert len(_left_behind(rig)) == 4
    _quiet(leave, rig)
    assert rig._add_candidate_layer is None
    assert _left_behind(rig) == []
    assert id(cand) not in rig.sync_manager._sync_handlers
    assert NAME not in rig.sync_manager.shared_labels
    assert rig._add_drawn_roi is None


def test_a_render_after_an_exit_builds_again(rig):
    _apply(rig, _a())
    first = rig._add_candidate_layer
    _quiet(_discard, rig)
    _apply(rig, _moved())
    assert rig._add_candidate_layer is not first
    np.testing.assert_array_equal(np.asarray(rig._add_candidate_layer.data),
                                  _expected(_moved()))


# --------------------------------------------------------------------------- #
# the plugin's viewer is a proxy: a lookup by name never returns the layer
# --------------------------------------------------------------------------- #
def test_through_the_plugin_viewer_proxy_the_redraw_takes_the_diff_path(
        monkeypatch, capsys):
    """napari hands a plugin its viewer wrapped in ``PublicOnlyProxy``;
    ``viewer.layers[name]`` then returns a new proxy object on every call,
    so a by-name identity test never matched the kept candidate and every
    Preview rebuilt it (seen live: two path=build renders, a remove between).
    The hidden candidate, Place markers off, then a render: the same four
    layers, one path=diff line, no remove."""
    from napari.utils._proxies import PublicOnlyProxy

    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator,
    )

    install_fakes(monkeypatch)
    h = _new_rig()
    h.viewer = PublicOnlyProxy(h.viewer)
    # the annotator's own lookup (the harness's walks the list instead)
    h._layer_by_name = types.MethodType(
        NativeMultiviewAnnotator._layer_by_name, h)
    try:
        # the trap holds
        assert h._layer_by_name(h.viewer.layers, "L") is not h.main_layer
        _apply(h, _a())
        first = _four(h)
        _quiet(h._on_add_place_markers_toggled, True)
        _quiet(h._invalidate_add_preview)                   # a click
        _quiet(h._on_add_place_markers_toggled, False)
        assert not first[0].visible
        capsys.readouterr()
        out = _moved()
        h._apply_add_result(out)
        text = capsys.readouterr().out
        renders = [ln for ln in text.splitlines()
                   if ln.startswith("[ADD-PERF] render ")]
        assert len(renders) == 1 and "path=diff" in renders[0], text
        assert not any(ln.startswith("[ADD-PERF] remove ")
                       for ln in text.splitlines()), text
        assert all(x is y for x, y in zip(_four(h), first))
        assert all(x.visible for x in first)
        np.testing.assert_array_equal(np.asarray(first[0].data),
                                      _expected(out))
    finally:
        _drop(h)


# --------------------------------------------------------------------------- #
# hardening: a failed redraw rebuilds; a hide with nothing drawn is silent;
# the history stays off; the outline filter stays
# --------------------------------------------------------------------------- #
def test_a_failed_redraw_rebuilds_and_the_rows_come(rig, capsys):
    _apply(rig, _a())
    _quiet(rig._invalidate_add_preview)
    first = rig._add_candidate_layer

    def _boom(*_a, **_k):
        raise RuntimeError("a side copy that cannot refresh")

    rig.side_copies()[0].refresh = _boom
    rig.adv.pushes.clear()
    capsys.readouterr()
    out = _moved()
    rig._apply_add_result(out)
    text = capsys.readouterr().out
    assert "[ADD] candidate redraw failed, rebuilding: " in text, text
    renders = [ln for ln in text.splitlines()
               if ln.startswith("[ADD-PERF] render ")]
    assert renders and "path=build" in renders[-1], text
    cand = rig._add_candidate_layer
    assert cand is not None and cand is not first
    np.testing.assert_array_equal(np.asarray(cand.data), _expected(out))
    assert rig._add_drawn_roi == tuple(out.roi)
    assert rig._add_drawn_token is not None
    pushed = [kw["candidate_ids"] for kw in rig.adv.pushes
              if "candidate_ids" in kw]
    assert pushed and sorted(pushed[-1]) == sorted(out.ids)


def test_an_invalidation_with_nothing_drawn_is_silent(rig, capsys):
    _apply(rig, _a())
    _quiet(rig._invalidate_add_preview)          # drawn -> hidden
    events = []
    for lyr in _four(rig):
        lyr.events.visible.connect(lambda e: events.append(e))
    capsys.readouterr()
    rig._invalidate_add_preview()                # nothing drawn now
    text = capsys.readouterr().out
    assert events == []
    assert not any(ln.startswith("[ADD-PERF] hide ")
                   for ln in text.splitlines()), text


def test_the_history_stays_off_after_an_editable_flip(rig):
    _apply(rig, _a())
    cand = rig._add_candidate_layer
    for lyr in _four(rig):
        lyr.editable = True
        lyr.editable = False                     # napari resets history
    assert cand._undo_history.maxlen != 0, "napari no longer resets: no teeth"
    _quiet(rig._invalidate_add_preview)
    _apply(rig, _moved())
    for lyr in _four(rig):
        assert lyr._undo_history.maxlen == 0, lyr.name
        assert lyr._redo_history.maxlen == 0, lyr.name


def test_a_redraw_that_puts_the_outline_back_keeps_its_filter(rig):
    import warnings

    _apply(rig, _a())
    _quiet(rig._invalidate_add_preview)
    cand = rig._add_candidate_layer
    cand.contour = 0                             # the layer lost the outline
    rig._add_contour_on = True                   # ... the page still wants it
    CW._add_outline_filter(False)
    try:
        _apply(rig, _moved())
        assert cand.contour == 1
        entry = CW._ADD_OUTLINE_FILTER
        assert entry is not None
        # by identity: an equal entry another test left behind is not ours
        assert any(f is entry for f in warnings.filters),             "the outline filter was dropped at the redraw's exit"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            CW._add_outline_filter(False)        # back in front of "always"
            CW._add_outline_filter(True)
            cand.refresh()
        assert [w for w in rec
                if CW._CONTOUR_3D_WARNING in str(w.message)] == []
    finally:
        _quiet(rig._on_add_contour_toggled, False)
        CW._add_outline_filter(False)

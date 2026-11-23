"""A result whose objects are unticked is never an empty screen.

* the candidate layer holds EVERY object of the result, ticked or not;
* ticked objects: their row swatch's colour, opaque, in the main view and
  the three side views;
* unticked objects: invisible in the main view (napari's 3D labels drawing
  stops at the first visible label, so a faint shell would hide the ticked
  objects inside it) and faint in the three side views;
* a tick, All or None changes colours only -- never the layer's data;
* the side views keep their faint colours through an outline toggle and an
  opacity / blending change (the property sync copies the main colours);
* Accept writes exactly the ticked objects;
* every exit leaves no candidate layer, side copy or handler behind;
* the first direct colour map of a process is paid once, on a thread that
  nothing waits for.

Real napari ``Labels`` layers and ``ViewerModel`` objects, the real
``SubviewerManager`` / ``LabelSyncManager`` (``connect_label_sync``) and the
annotator's own render / remove code, through ``add_layer_rig.LayerRig``.
Offscreen there is no GL context: the side views' Qt viewers are stand-ins.
"""

from __future__ import annotations

import contextlib
import io
import threading

import numpy as np
import pytest

from napari_epicycle.annotator import curation_3d_widget as CW
from packages.epicycle_annotator import add_preview as AP

from .add_harness import Harness, install_fakes
from .add_layer_rig import LayerRig, blocks, result_for

SHAPE = (12, 40, 40)
EXISTING = 7
#: the result: four objects, the first two at the box edge (they start
#: unticked), the other two ticked
IDS = (100, 101, 102, 103)
EDGE = (100, 101)
TICKED = (102, 103)
NAME = "_epicycle_add_candidate"


def _swatch(i):
    """The row swatch's colour (napari's default cyclic colours)."""
    from napari.utils.colormaps import label_colormap

    return np.asarray(label_colormap(49, seed=0.5).map(
        np.array([int(i)], dtype=np.intp))[0], np.float32)


def _rgba(layer, i):
    """The colour the layer draws id *i* in (an id its map does not list
    takes the map's ``None`` colour)."""
    from napari.utils.colormaps import DirectLabelColormap

    cm = layer.colormap
    assert isinstance(cm, DirectLabelColormap), type(cm).__name__
    assert layer.colormap is layer._direct_colormap   # direct mode, not auto
    if int(i) in cm.color_dict:
        return np.asarray(cm.color_dict[int(i)], np.float32)
    return np.asarray(cm.color_dict[None], np.float32)


def _check(h, ticked, *, side_alpha=None):
    """Main: ticked opaque, the rest alpha 0; the three side copies:
    ticked opaque, the rest at the faint alpha.  Colours = the swatches."""
    side_alpha = AP.ADD_UNTICKED_ALPHA if side_alpha is None else side_alpha
    cand = h._add_candidate_layer
    sides = h.side_copies()
    assert all(s is not None for s in sides)
    for i in IDS:
        want = _swatch(i)
        on = int(i) in set(int(t) for t in ticked)
        got = _rgba(cand, i)
        if on:
            np.testing.assert_allclose(got[:3], want[:3], atol=1e-6)
        assert got[3] == pytest.approx(1.0 if on else 0.0), (i, got)
        for s in sides:
            got = _rgba(s, i)
            np.testing.assert_allclose(got[:3], want[:3], atol=1e-6)
            assert got[3] == pytest.approx(1.0 if on else side_alpha), (
                s.name, i, got)
    for layer in [cand] + sides:
        assert float(_rgba(layer, 0)[3]) == 0.0


@pytest.fixture
def rig(monkeypatch):
    install_fakes(monkeypatch)
    vol = np.zeros(SHAPE, np.int32)
    vol[0:2, 0:3, 0:3] = EXISTING
    h = LayerRig(vol, ndisplay=3)
    h.adv.preset = "cells"
    h.place_box(tuple(slice(0, s) for s in SHAPE))
    h._add_click_markers = []
    h._add_markers_base = 500
    yield h
    with contextlib.suppress(Exception), contextlib.redirect_stdout(
            io.StringIO()):
        h._remove_add_candidate()


def _apply(h):
    res = blocks(SHAPE, 4, first=IDS[0], block=8)
    res[0:2, 0:3, 0:3] = 0                    # not over the kept object
    out = result_for(tuple(slice(0, s) for s in SHAPE), res, edge=EDGE)
    assert sorted(out.ids) == list(IDS)
    with contextlib.redirect_stdout(io.StringIO()):
        h._apply_add_result(out)
    assert sorted(h.adv.checked) == list(TICKED)    # the rows as they start
    return out


# --------------------------------------------------------------------------- #
# the colours
# --------------------------------------------------------------------------- #
def test_faint_in_side_views_only(rig):
    h = rig
    _apply(h)
    cand = h._add_candidate_layer
    assert cand is not None and cand.name == NAME
    # the layer really holds the unticked objects
    assert {int(v) for v in np.unique(cand.data) if v} == set(IDS)
    for s in h.side_copies():
        assert {int(v) for v in np.unique(np.asarray(s.data)) if v} == set(IDS)
    _check(h, TICKED)


def test_the_faint_colour_maps_by_themselves():
    class _W:
        pass

    main, side = CW._add_candidate_colormaps(_W(), IDS, TICKED)
    # the main map lists the ticked ids only: an unticked id takes the
    # clear None colour (a shorter map: cheaper to build and to draw)
    assert set(main.color_dict) - {None, 0} == set(TICKED)
    assert set(side.color_dict) - {None, 0} == set(IDS)
    for i in IDS:
        on = i in TICKED
        assert side.color_dict[i][3] == pytest.approx(
            1.0 if on else AP.ADD_UNTICKED_ALPHA)
        np.testing.assert_allclose(side.color_dict[i][:3], _swatch(i)[:3],
                                   atol=1e-6)
        if on:
            assert main.color_dict[i][3] == pytest.approx(1.0)
            np.testing.assert_allclose(main.color_dict[i][:3],
                                       _swatch(i)[:3], atol=1e-6)
    for cm in (main, side):
        assert cm.color_dict[0][3] == 0 and cm.color_dict[None][3] == 0
    # nothing ticked: still a direct map (None is clear, not napari's
    # default black), so the layer stays in direct mode
    main, _side = CW._add_candidate_colormaps(_W(), IDS, ())
    assert set(main.color_dict) == {None, 0}
    assert float(main.color_dict[None][3]) == 0.0


def test_a_tick_all_and_none_change_colours_never_data(rig):
    h = rig
    _apply(h)
    cand = h._add_candidate_layer
    data = cand.data
    before = np.array(data, copy=True)
    sides = h.side_copies()
    with contextlib.redirect_stdout(io.StringIO()):
        h.adv.tick([100, 102, 103])           # one tick
    _check(h, (100, 102, 103))
    with contextlib.redirect_stdout(io.StringIO()):
        h.adv.tick(list(IDS))                 # All
    _check(h, IDS)
    with contextlib.redirect_stdout(io.StringIO()):
        h.adv.tick([])                        # None
    _check(h, ())
    assert h._add_candidate_layer is cand, "a tick rebuilt the layer"
    assert h.side_copies() == sides
    assert cand.data is data
    np.testing.assert_array_equal(np.asarray(cand.data), before)
    assert h.adv.last("status_text") == AP.ADD_S1_TICK_ONE


def _count_side_colour_writes(h):
    """Count colour-map writes on each side copy made from now on (the map
    a copy is built with is not a write)."""
    counts = {}

    def _on_insert(event):
        layer = event.value
        if not str(layer.name).startswith(NAME):
            return
        counts[layer.name] = 0

        def _bump(_e, name=layer.name):
            counts[name] += 1
        layer.events.colormap.connect(_bump)

    for vm in (h.viewer_model_xy, h.viewer_model_xz, h.viewer_model_yz):
        vm.layers.events.inserted.connect(_on_insert)
    return counts


@pytest.mark.parametrize("outline", [False, True])
def test_side_copies_are_coloured_once_per_preview(rig, outline):
    """A Preview writes each side copy's colour map twice after making it:
    the mirror gives it the main map, then it gets the side map -- with
    the outline on too: the outline is written first (written after, its
    property sync copied the main map back over the side map and the
    keeper put the side map back: four writes)."""
    h = rig
    h._add_contour_on = outline
    counts = _count_side_colour_writes(h)
    _apply(h)
    assert len(counts) == 3, counts
    assert all(n == 2 for n in counts.values()), counts
    _check(h, TICKED)
    if outline:
        assert all(s.contour == 1 for s in h.side_copies())


def test_side_maps_survive_property_sync(rig):
    h = rig
    _apply(h)
    with contextlib.redirect_stdout(io.StringIO()):
        h._on_add_contour_toggled(True)       # the outline tick
    assert all(s.contour == 1 for s in h.side_copies())
    _check(h, TICKED)
    h._add_candidate_layer.opacity = 0.5
    assert all(s.opacity == pytest.approx(0.5) for s in h.side_copies())
    _check(h, TICKED)
    h._add_candidate_layer.blending = "additive"
    _check(h, TICKED)
    with contextlib.redirect_stdout(io.StringIO()):
        h._on_add_contour_toggled(False)
        h.adv.tick([])                        # None, with the outline off
    _check(h, ())


# --------------------------------------------------------------------------- #
# Accept and the exits
# --------------------------------------------------------------------------- #
def test_accept_writes_only_ticked_rows(rig):
    h = rig
    _apply(h)
    with contextlib.redirect_stdout(io.StringIO()):
        h.adv.tick([100, 102, 103])           # one edge row ticked, 101 not
        h._on_add_committed_cb = None
        h._accept_add()
    vol = h.sync_manager.shared_labels["L"]
    got = {int(v) for v in np.unique(vol) if v} - {EXISTING}
    assert got == {100, 102, 103}


def _left_behind(h):
    out = []
    for vm in (h.viewer, h.viewer_model_xy, h.viewer_model_xz,
               h.viewer_model_yz):
        out += [lyr.name for lyr in vm.layers
                if str(lyr.name).startswith(NAME)]
    return out


def _props_handlers(layer):
    """How many times the candidate's own property handler is connected."""
    n = 0
    for ev in ("colormap", "contour", "opacity", "blending", "brush_size",
               "n_edit_dimensions"):
        for cb in getattr(layer.events, ev).callbacks:
            name = cb[1] if isinstance(cb, tuple) else getattr(
                cb, "__name__", "")
            if name == "_on_add_candidate_props":
                n += 1
    return n


def _exit_discard(h):
    h._add_secondary()                        # Discard (a result is up)


def _exit_accept(h):
    h._on_add_committed_cb = None
    h._accept_add()


def _exit_close_box(h):
    h._reject_add()                           # Close box / Done


def _exit_preset_switch(h):
    h._on_add_preset_changed("nuclei")


def _exit_patch_switch(h):
    h._teardown_add_for_switch()


def _exit_annotator_close(h):
    # the Add part of the annotator's close cleanup with a box up
    h._reject_add()
    h._add_forget_clears()


def _exit_reset(h):
    # the Add part of the close cleanup with no box (review / idle)
    h._reset_add_state()


@pytest.mark.parametrize("leave", [
    _exit_discard, _exit_accept, _exit_close_box,
    _exit_patch_switch, _exit_annotator_close, _exit_reset,
], ids=lambda f: f.__name__[len("_exit_"):])
def test_nothing_left_behind(rig, leave):
    h = rig
    _apply(h)
    cand = h._add_candidate_layer
    assert _props_handlers(cand) == 6         # one per property event
    assert id(cand) in h.sync_manager._sync_handlers
    assert len(_left_behind(h)) == 4
    with contextlib.redirect_stdout(io.StringIO()):
        leave(h)
    assert h._add_candidate_layer is None
    assert _left_behind(h) == []
    assert _props_handlers(cand) == 0
    assert id(cand) not in h.sync_manager._sync_handlers
    assert NAME not in h.sync_manager.shared_labels
    assert getattr(h, "_add_drawn_token", None) is None
    assert getattr(h, "_add_colours_token", None) is None


def test_a_preset_switch_zeros_and_hides(rig):
    """A preset switch is an input change, not an exit: the four layers
    stay (made once per box), zeroed and hidden, still watched and synced;
    nothing of the result is left drawn."""
    h = rig
    _apply(h)
    cand = h._add_candidate_layer
    with contextlib.redirect_stdout(io.StringIO()):
        _exit_preset_switch(h)
    assert h._add_candidate_layer is cand
    assert len(_left_behind(h)) == 4
    assert not cand.visible and not any(s.visible for s in h.side_copies())
    assert not np.any(cand.data)
    assert _props_handlers(cand) == 6
    assert id(cand) in h.sync_manager._sync_handlers
    assert getattr(h, "_add_drawn_roi", None) is None
    assert getattr(h, "_add_drawn_token", None) is None
    assert getattr(h, "_add_colours_token", None) is None


# --------------------------------------------------------------------------- #
# the first direct colour map of a process
# --------------------------------------------------------------------------- #
def _fresh_warm(monkeypatch, safe=(True, "a thread-safe layer (test)")):
    monkeypatch.setattr(CW, "_ADD_CMAP_WARM",
                        {"thread": None, "dtypes": set(), "safe": safe})


def test_colormap_warm_up_once_off_main(monkeypatch):
    from napari.utils.colormaps import colormap as napari_cm

    install_fakes(monkeypatch)
    _fresh_warm(monkeypatch)
    gate = threading.Event()
    calls = []
    real = napari_cm._cast_labels_data_to_texture_dtype_direct

    def spy(data, cmap):
        calls.append((threading.current_thread() is threading.main_thread(),
                      np.asarray(data).dtype, np.asarray(data).ndim,
                      int(cmap._num_unique_colors)))
        gate.wait(20)
        return real(data, cmap)

    monkeypatch.setattr(napari_cm, "_cast_labels_data_to_texture_dtype_direct",
                        spy)
    h = Harness(np.zeros((6, 16, 16), np.uint32))
    with contextlib.redirect_stdout(io.StringIO()):
        h._start_add()
    th = CW._ADD_CMAP_WARM["thread"]
    assert th is not None, "the warm-up was not started at the first box"
    assert th.is_alive(), "the box waited for the warm-up"
    with contextlib.redirect_stdout(io.StringIO()):
        h._reject_add()
        h._start_add()                        # a second box: no second job
        h._add_ensure_cmap_warm()
    assert CW._ADD_CMAP_WARM["thread"] is th
    gate.set()
    th.join(60)
    assert not th.is_alive()
    assert calls
    assert not any(on_main for on_main, *_ in calls)
    assert {dt for _m, dt, _n, _u in calls} == {np.dtype(np.uint32)}
    # the ranks a draw passes: the 3D main view and the 2D side views
    assert {n for _m, _dt, n, _u in calls} >= {2, 3}
    # the texture size an Add map can need: up to 254 colours (49 cyclic
    # colours, faint or opaque, never more)
    uniq = {u for *_x, u in calls}
    assert max(uniq) <= 254
    with contextlib.redirect_stdout(io.StringIO()):
        h._reject_add()


def test_the_warm_up_is_per_label_type(monkeypatch):
    """A patch with another label data type gets its own warm-up; the same
    type never twice."""
    from napari.utils.colormaps import colormap as napari_cm

    install_fakes(monkeypatch)
    _fresh_warm(monkeypatch)
    seen = []
    monkeypatch.setattr(napari_cm, "_cast_labels_data_to_texture_dtype_direct",
                        lambda data, cmap: seen.append(np.asarray(data).dtype))
    monkeypatch.setattr(napari_cm, "_cast_labels_data_to_texture_dtype_auto",
                        lambda data, cmap: None)
    threads = []
    for dt in (np.int32, np.int32, np.uint32, np.uint32):
        h = Harness(np.zeros((6, 16, 16), dt))
        with contextlib.redirect_stdout(io.StringIO()):
            h._add_ensure_cmap_warm()
        th = CW._ADD_CMAP_WARM["thread"]
        if th is not None and th not in threads:
            threads.append(th)
    for th in threads:
        th.join(60)
    assert len(threads) == 2
    assert set(seen) == {np.dtype(np.int32), np.dtype(np.uint32)}


def test_no_warm_up_without_a_thread_safe_numba_layer(monkeypatch, capsys):
    install_fakes(monkeypatch)
    _fresh_warm(monkeypatch, safe=None)
    monkeypatch.setattr(CW, "_add_cmap_warm_safe",
                        lambda: (False, "numba would use workqueue"))
    h = Harness(np.zeros((6, 16, 16), np.int32))
    h._add_ensure_cmap_warm()
    h._add_ensure_cmap_warm()
    out = capsys.readouterr().out
    assert CW._ADD_CMAP_WARM["thread"] is None
    assert out.count("[ADD] colour warm-up skipped") == 1
    assert "workqueue" in out


@pytest.mark.parametrize("layer,initialised,loadable,want", [
    ("default", None, {"omp"}, True),          # the usual conda install
    ("default", None, {"tbb", "omp"}, True),
    ("default", None, set(), False),           # it would fall to workqueue
    ("workqueue", None, {"tbb", "omp"}, False),
    ("omp", None, {"omp"}, True),
    ("tbb", None, {"omp"}, False),             # numba would refuse to start
    ("threadsafe", None, {"omp"}, True),
    ("forksafe", None, set(), False),          # workqueue
    ("something", None, {"omp"}, False),       # unknown: never
    ("default", "workqueue", {"omp"}, False),  # already running workqueue
    ("default", "omp", set(), True),           # already running omp
    ("default", "raise", {"omp"}, False),      # a check that raises
])
def test_warm_up_safety_rule(monkeypatch, layer, initialised, loadable,
                             want):
    from numba.core import config
    from numba.np.ufunc import parallel as par

    monkeypatch.setattr(config, "THREADING_LAYER", layer)
    monkeypatch.setattr(config, "THREADING_LAYER_PRIORITY",
                        ["tbb", "omp", "workqueue"])
    if initialised in (None, "raise"):
        monkeypatch.setattr(par, "_is_initialized", False)
    else:
        monkeypatch.setattr(par, "_is_initialized", True)
        monkeypatch.setattr(par, "_threading_layer", initialised)

    def _loadable(name):
        if initialised == "raise":
            raise RuntimeError("an import that raises")
        return name in loadable

    monkeypatch.setattr(CW, "_add_numba_pool_loadable", _loadable)
    ok, why = CW._add_cmap_warm_safe()
    assert ok is want, why
    assert isinstance(why, str) and why


_WQ_SCRIPT = '''
import numpy as np
import napari_epicycle.annotator.curation_3d_widget as CW
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.colormaps import colormap as ncm


class _L:
    data = np.zeros((4, 4, 4), np.int32)


class _W:
    def _add_main_label_layer(self):
        return _L()


CW.Curation3DAnnotator._add_ensure_cmap_warm(_W())
clear = np.zeros(4, np.float32)
d = {None: clear, 0: clear}
for i in range(1, 30):
    d[i] = np.array([i / 30, 0.5, 0.5, 1.0], np.float32)
cm = DirectLabelColormap(color_dict=d)
big = (np.arange(40 * 64 * 64) % 29).astype(np.int32).reshape(40, 64, 64)
for _ in range(5):
    ncm._cast_labels_data_to_texture_dtype_direct(big, cm)
    ncm._cast_labels_data_to_texture_dtype_direct(big[3], cm)
import numba
th = CW._ADD_CMAP_WARM["thread"]
print("WARM-THREAD", th is not None, numba.threading_layer())
'''


def test_the_warm_up_never_aborts_under_workqueue(tmp_path):
    """The concurrent case that aborted the process (numba's workqueue
    layer is not thread-safe): the main thread draws direct colour maps
    while the warm-up would compile.  With workqueue forced, the real
    ``_add_ensure_cmap_warm`` skips it and the process ends normally."""
    import os
    import pathlib
    import subprocess
    import sys

    repo = pathlib.Path(__file__).resolve().parents[2]
    script = tmp_path / "wq.py"
    script.write_text(_WQ_SCRIPT, encoding="utf-8")
    env = dict(os.environ)
    env["NUMBA_THREADING_LAYER"] = "workqueue"
    env["PYTHONPATH"] = str(repo)
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run([sys.executable, "-X", "utf8", str(script)],
                          cwd=str(repo), env=env, capture_output=True,
                          text=True, timeout=600)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-3000:]
    assert "WARM-THREAD False workqueue" in out, out[-3000:]
    assert "[ADD] colour warm-up skipped" in out, out[-3000:]


def test_no_warm_up_without_napari_s_function(monkeypatch, capsys):
    from napari.utils.colormaps import colormap as napari_cm

    install_fakes(monkeypatch)
    _fresh_warm(monkeypatch)
    monkeypatch.delattr(napari_cm, "_cast_labels_data_to_texture_dtype_direct")
    h = Harness(np.zeros((6, 16, 16), np.int32))
    h._add_ensure_cmap_warm()
    h._add_ensure_cmap_warm()
    out = capsys.readouterr().out
    assert CW._ADD_CMAP_WARM["thread"] is None
    assert out.count("[ADD] colour warm-up skipped") == 1

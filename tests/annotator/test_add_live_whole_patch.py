"""Update as I click on a whole patch: Nuclei ▸ At the markers I place.

With the tick on, a whole patch runs as you click too: the patch is
prepared once (its bright pieces) and each click solves only the pieces
whose markers changed; the drawn result is updated by the changed places
only.  Every run on a whole patch with the tick on -- a Preview press too
-- goes this way, and a Preview pressed while the drawn result already
answers the markers runs nothing.  With the tick off a whole patch runs as
it always did.  A box is unchanged in both.  Cells on a whole patch still
waits for Preview.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_incremental as AI
from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator import add_segment as AS

from .add_harness import FakeThread
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_nuclei_box import (
    NUC_A,
    NUC_B,
    _active,
    _FakeTimer,
    _fire,
    _finish,
    _nuclei,
)


class _Spies:
    def __init__(self, monkeypatch):
        self.prepares = 0
        self.one_go = 0
        self.patches = []
        real_prep = AI.prepare_markers_box
        real_seg = AS.segment_at_markers

        def prep(*a, **k):
            self.prepares += 1
            return real_prep(*a, **k)

        def seg(*a, **k):
            self.one_go += 1
            return real_seg(*a, **k)

        monkeypatch.setattr(AI, "prepare_markers_box", prep)
        monkeypatch.setattr(AS, "segment_at_markers", seg)


def _launches(hh):
    got = []
    real = hh._launch_add_worker

    def spy(kwargs, *a, **k):
        got.append(kwargs)
        return real(kwargs, *a, **k)

    hh._launch_add_worker = spy
    return got


def _whole(hh, *, live):
    _nuclei(hh, live=live)
    hh._start_add_whole()
    assert hh._add_whole_patch
    return hh


@pytest.fixture
def wl(h, monkeypatch):  # noqa: F811
    import qtpy.QtCore

    _FakeTimer.made = []
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    hh = _whole(h, live=True)
    hh.spies = _Spies(monkeypatch)
    hh.launched = _launches(hh)
    return hh


@pytest.fixture
def wo(h, monkeypatch):  # noqa: F811
    import qtpy.QtCore

    _FakeTimer.made = []
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    hh = _whole(h, live=False)
    hh.spies = _Spies(monkeypatch)
    hh.launched = _launches(hh)
    return hh


def _click_and_draw(hh, at):
    assert hh._add_marker_click(at)
    _fire(hh)
    _finish(hh)
    assert hh._add_preview_result is not None, hh._add_transient
    return hh._add_preview_res


def _drawn(hh):
    """What the candidate layer shows: every draw replayed -- the whole
    result, or only the changed places written over the one before."""
    shown = None
    for (result, _roi, _c), patches in zip(hh.rendered,
                                           hh.rendered_patches):
        if patches is None:
            shown = np.array(result, copy=True)
        else:
            for box, values in patches:
                shown[box] = values
    return shown


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_a_whole_patch_runs_live_for_nuclei_markers_only(wl):
    assert wl._add_live_possible() is False      # no marker yet
    assert wl._add_marker_click(NUC_A)
    assert wl._add_live_possible()
    assert _active(wl)
    s = wl._add_s1_state()
    assert "not updated as you click" not in AP.s1_sentence(s)
    # Cells ▸ Click markers on a whole patch runs as you click only with
    # walls the click fill redoes; others wait for Preview
    wl.adv.preset = "cells"
    wl.adv.source = (AP.SOURCE_CLICKS, None)
    wl.adv.wall = dict(wl.adv.wall, wall_kind="membrane")
    assert wl._add_live_possible()
    wl.adv.wall = dict(wl.adv.wall, wall_kind="frangi")
    assert not wl._add_live_possible()


def test_the_nuclei_gate_sentence_is_gone():
    assert not hasattr(AP, "ADD_S1_LIVE_OVER_GATE_PATCH_NUCLEI")


def test_the_gate_sentence_stays_for_cells_only():
    s = {"has_patch": True, "has_box": True, "preset": "nuclei",
         "look_for": AP.LOOK_MARKERS, "channel": "nuc",
         "box_shape": (20, 30, 30), "whole_patch": True, "n_clicks": 1,
         "n_ticked": 1, "place_markers_on": True, "live_on": True,
         "source": AP.SOURCE_CLICKS}
    assert AP.s1_sentence(s) != AP.ADD_S1_LIVE_OVER_GATE_PATCH
    assert "not updated as you click" not in AP.s1_sentence(s)
    cells = dict(s, preset="cells", wall_channel="mem")
    assert AP.s1_sentence(cells) == AP.ADD_S1_LIVE_OVER_GATE_PATCH


# --------------------------------------------------------------------------- #
# live on: prepared once, solved per click, drawn by the changed places
# --------------------------------------------------------------------------- #
def test_a_click_runs_with_no_preview_through_the_engine(wl):
    res = _click_and_draw(wl, NUC_A)
    assert wl.spies.prepares == 1 and wl.spies.one_go == 0
    assert res.marker_solution is not None and res.marker_stage is not None
    assert res.whole_patch and res.edge_ids == set()
    i = int(res.ids[0])
    assert res.result[NUC_A] == i
    vals, counts = np.unique(res.result, return_counts=True)
    assert res.sizes == {int(v): int(n) for v, n in zip(vals, counts) if v}
    assert np.array_equal(_drawn(wl), res.result)
    # the first run copied the patch's labels once; it prepared on the
    # worker
    assert wl.launched[0]["existing_roi"] is not None


def test_the_second_click_reuses_the_stage_and_draws_patches_only(wl):
    first = _click_and_draw(wl, NUC_A)
    second = _click_and_draw(wl, NUC_B)
    assert wl.spies.prepares == 1, "the stage is prepared once"
    kw = wl.launched[-1]
    assert kw["existing_roi"] is None, "no copy of the patch's labels"
    assert kw["marker_stage"] is first.marker_stage
    assert kw["marker_previous"] is first.marker_solution
    assert not second.marker_diff.full
    assert wl.rendered_patches[-1] is not None
    assert len(wl.rendered_patches[-1]) >= 1
    assert np.array_equal(_drawn(wl), second.result)
    assert len(second.ids) == 2


def _runs():
    return len(FakeThread.instances)


def _settle(hh):
    """The run a change scheduled: the quiet, the run, its finish."""
    _fire(hh)
    _finish(hh)
    assert hh._add_preview_result is not None, hh._add_transient
    return hh._add_preview_res


def test_a_preview_while_the_drawn_result_is_current_runs_nothing(wl):
    """With the smoothing on (the page's default): a Preview while the drawn
    result answers the markers runs nothing; a change of the smoothing, of
    Smallest object or a Shift seed each runs again (Smallest object with
    a new prepare, the smoothing and the seed on the prepared patch)."""
    wl.adv.params["auto_smooth"] = True
    wl.adv.params["smooth_sigma"] = 1.0
    res = _click_and_draw(wl, NUC_A)
    assert res.marker_solution.smooth_sigma == 1.0
    assert wl._add_marker_drawn_current()
    n = _runs()
    wl._preview_add()
    assert _runs() == n, "Preview ran again"
    assert not _active(wl)
    assert wl._add_preview_res is res
    assert np.array_equal(_drawn(wl), res.result)
    # the smoothing: a run on the prepared patch
    wl.adv.params["smooth_sigma"] = 1.5
    wl._on_add_advanced_changed()
    assert not wl._add_marker_drawn_current()
    sm = _settle(wl)
    assert _runs() == n + 1
    assert sm is not res and sm.marker_solution.smooth_sigma == 1.5
    assert wl.spies.prepares == 1
    wl._preview_add()
    assert _runs() == n + 1, "Preview ran again"
    # Smallest object: in the prepared patch's key -- prepared again
    wl.adv.params["min_size"] = 3
    wl._on_add_advanced_changed()
    assert not wl._add_marker_drawn_current()
    ms = _settle(wl)
    assert _runs() == n + 2
    assert ms is not sm and wl.spies.prepares == 2
    wl._preview_add()
    assert _runs() == n + 2, "Preview ran again"
    # a Shift seed on the current object: a run on the prepared patch
    seed = (NUC_A[0], NUC_A[1] + 3, NUC_A[2])
    assert wl._add_marker_click(seed, shift=True)
    assert not wl._add_marker_drawn_current()
    sd = _settle(wl)
    assert _runs() == n + 3
    assert len(sd.marker_solution.points) == 2
    assert wl.spies.prepares == 2
    wl._preview_add()
    assert _runs() == n + 3, "Preview ran again"


def test_a_manual_preview_with_the_tick_on_goes_through_the_engine(wl):
    assert wl._add_marker_click(NUC_A)
    wl._add_live_timer.stop()
    wl._preview_add()
    _finish(wl)
    res = wl._add_preview_res
    assert res is not None and res.marker_solution is not None
    assert wl.spies.one_go == 0 and wl.spies.prepares == 1


# --------------------------------------------------------------------------- #
# live off: today's path, byte for byte
# --------------------------------------------------------------------------- #
def test_with_the_tick_off_a_whole_patch_runs_as_before(wo):
    assert wo._add_marker_click(NUC_A)
    assert wo._add_marker_click(NUC_B)
    assert not _active(wo)
    wo._preview_add()
    _finish(wo)
    res = wo._add_preview_res
    assert res is not None
    assert wo.spies.one_go == 1 and wo.spies.prepares == 0
    assert getattr(res, "marker_solution", None) is None
    kw = dict(wo.launched[-1])
    assert "marker_stage" not in kw and "marker_previous" not in kw
    again = AP.run_add_preview(**kw)
    assert np.array_equal(again.result, res.result)
    assert again.ids == res.ids and again.sizes == res.sizes


def test_the_tick_on_after_a_tick_off_preview_keeps_the_result(wo):
    """Nuclei's rule does not change with the tick: the result a tick-off
    Preview drew stays when it comes on, and the next click runs through
    the engine."""
    assert wo._add_marker_click(NUC_A)
    wo._preview_add()
    _finish(wo)
    drawn = wo._add_preview_res
    assert drawn is not None and drawn.ok
    n = len(FakeThread.instances)
    wo._on_add_live_toggled(True)
    assert wo._add_preview_res is drawn
    assert wo._add_preview_result is not None
    assert not _active(wo) and len(FakeThread.instances) == n
    assert wo._add_marker_click(NUC_B)
    _fire(wo)
    assert wo.launched[-1].get("marker_engine")
    _finish(wo)
    res = wo._add_preview_res
    assert res is not drawn and res.marker_solution is not None


# --------------------------------------------------------------------------- #
# Accept, and when the prepared stage is let go
# --------------------------------------------------------------------------- #
def test_accept_writes_the_drawn_objects_and_drops_the_stage(wl):
    wl.adv.params["clear_border"] = False
    res = _click_and_draw(wl, NUC_A)
    assert wl._add_marker_cache is not None
    vol = wl.sync_manager.shared_labels["L"]
    wl._accept_add()
    i = int(res.ids[0])
    assert int(vol[NUC_A]) == i
    assert int(np.count_nonzero(vol == i)) == int(res.sizes[i])
    assert getattr(wl, "_add_marker_cache", None) is None


def test_a_setting_in_the_stage_drops_it(wl):
    _click_and_draw(wl, NUC_A)
    assert wl._add_marker_cache is not None
    wl._add_threshold_value = 0.6
    wl._add_prune_caches()
    assert wl._add_marker_cache is None


def test_close_box_and_a_patch_switch_drop_the_stage(wl):
    _click_and_draw(wl, NUC_A)
    assert wl._add_marker_cache is not None
    wl._add_forget_clears()                    # a patch switch / a close
    assert wl._add_marker_cache is None
    _click_and_draw(wl, NUC_B)
    assert wl._add_marker_cache is not None
    wl._reject_add()                           # Close box
    assert getattr(wl, "_add_marker_cache", None) is None


def test_a_stale_run_keeps_its_solution_as_the_next_previous(wl):
    _click_and_draw(wl, NUC_A)
    assert wl._add_marker_click(NUC_B)
    _fire(wl)
    th = FakeThread.instances[-1]
    # the smoothing changes while it runs: its answer is stale, its stage
    # and solution are still good
    wl.adv.params["smooth_sigma"] = 2.0
    wl._invalidate_add_preview()
    th.run()
    th.end()
    kept = wl._add_marker_cache["solution"]
    assert len(kept.points) == 2
    if _active(wl):
        _fire(wl)
    _finish(wl)
    assert wl.launched[-1]["marker_previous"] is kept
    assert wl.spies.prepares == 1


def test_the_tick_off_then_on_solves_the_whole_patch_again(wl):
    _click_and_draw(wl, NUC_A)
    wl._on_add_live_toggled(False)
    assert getattr(wl, "_add_marker_cache", None) is None
    wl._on_add_live_toggled(True)
    assert wl._add_marker_click(NUC_B)
    _fire(wl)
    _finish(wl)
    res = wl._add_preview_res
    assert res.marker_diff.full
    assert wl.spies.prepares == 2


def test_the_redraw_writes_only_the_changed_places(h):  # noqa: F811
    """The candidate layer's array: the changed places are written, every
    other voxel is left as it is (a mark outside them stays)."""
    from types import SimpleNamespace

    roi = (slice(0, 6), slice(0, 8), slice(0, 8))
    data = np.full((6, 8, 8), 9, np.int32)
    cand = SimpleNamespace(data=data, contour=0, colormap=None, visible=True,
                           refresh=lambda: None)
    h._add_drawn_roi = roi
    h._add_candidate_side_copies = lambda: []
    h._add_put_side_colours = lambda: None
    h._add_contour_on = False
    new = np.zeros((6, 8, 8), np.int32)
    new[1:3, 1:3, 1:3] = 4
    patches = (((slice(1, 3), slice(1, 3), slice(1, 3)), new[1:3, 1:3, 1:3]),
               ((slice(5, 6), slice(7, 8), slice(7, 8)), new[5:6, 7:8, 7:8]))
    h._add_redraw_candidate(cand, new, roi, (object(), object()),
                            patches=patches)
    assert (data[1:3, 1:3, 1:3] == 4).all() and data[5, 7, 7] == 0
    rest = np.ones(data.shape, bool)
    rest[1:3, 1:3, 1:3] = False
    rest[5, 7, 7] = False
    assert (data[rest] == 9).all()
    # without patches: the whole result
    h._add_redraw_candidate(cand, new, roi, (object(), object()))
    assert np.array_equal(data, new)



# --------------------------------------------------------------------------- #
# the changed places into the candidate's own dtype (the patch's labels are
# uint32 or uint16; the engine's patches are int32)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [np.uint32, np.uint16, np.int32],
                         ids=["uint32", "uint16", "int32"])
def test_patch_write_into_a_uint32_candidate(h, dtype):  # noqa: F811
    from types import SimpleNamespace

    roi = (slice(0, 6), slice(0, 8), slice(0, 8))
    data = np.full((6, 8, 8), 9, dtype)
    cand = SimpleNamespace(data=data, contour=0, colormap=None, visible=True,
                           refresh=lambda: None)
    h._add_drawn_roi = roi
    h._add_candidate_side_copies = lambda: []
    h._add_put_side_colours = lambda: None
    h._add_contour_on = False
    new = np.zeros((6, 8, 8), AI._result_dtype([4]))
    new[1:3, 1:3, 1:3] = 4
    patches = (((slice(1, 3), slice(1, 3), slice(1, 3)),
                new[1:3, 1:3, 1:3].copy()),
               ((slice(5, 6), slice(7, 8), slice(7, 8)),
                new[5:6, 7:8, 7:8].copy()))
    h._add_redraw_candidate(cand, new, roi, (object(), object()),
                            patches=patches)
    assert data.dtype == dtype
    assert (data[1:3, 1:3, 1:3] == 4).all() and data[5, 7, 7] == 0
    rest = np.ones(data.shape, bool)
    rest[1:3, 1:3, 1:3] = False
    rest[5, 7, 7] = False
    assert (data[rest] == 9).all()


def test_three_live_clicks_on_a_uint32_candidate_never_rebuild(wl):
    """The real candidate redraw on a uint32 layer: every click after the
    first writes its changed places; the four layers are never taken down
    and built again."""
    import types
    from types import SimpleNamespace

    from napari_epicycle.annotator import curation_3d_widget as CW

    vol = wl.sync_manager.shared_labels["L"]
    data = np.zeros(vol.shape, np.uint32)
    cand = SimpleNamespace(data=data, contour=0, colormap=None, visible=True,
                           refresh=lambda: None,
                           name="_epicycle_add_candidate")
    wl.viewer.layers.append(cand)
    wl._add_candidate_layer = cand
    wl._add_drawn_roi = None
    wl._add_candidate_reusable = lambda kept, main: kept is cand
    wl._add_candidate_side_copies = lambda: []
    wl._add_put_side_colours = lambda: None
    wl._add_contour_on = False
    rebuilt = []

    def no_rebuild():
        rebuilt.append(True)
        raise AssertionError("the candidate was taken down to rebuild")

    wl._remove_add_candidate = no_rebuild
    writes = []
    real_redraw = wl._add_redraw_candidate

    def redraw(c, result, roi, colours, patches=None):
        writes.append(patches is not None)
        return real_redraw(c, result, roi, colours, patches=patches)

    wl._add_redraw_candidate = redraw
    wl._render_add_candidate = types.MethodType(
        CW.Curation3DAnnotator._render_add_candidate, wl)
    third = (NUC_A[0], NUC_A[1], NUC_A[2] + 3)
    for at in (NUC_A, NUC_B, third):
        res = _click_and_draw(wl, at)
        assert np.array_equal(data[tuple(wl._add_preview_roi)], res.result)
    assert rebuilt == []
    assert writes == [False, True, True], writes
    assert wl._add_candidate_layer is cand


# --------------------------------------------------------------------------- #
# the tick going off while a whole-patch run is in flight
# --------------------------------------------------------------------------- #
def test_the_tick_off_mid_run_keeps_no_stage_when_it_lands(wl):
    assert wl._add_marker_click(NUC_A)
    _fire(wl)
    th = FakeThread.instances[-1]
    wl._on_add_live_toggled(False)
    assert wl._add_marker_cache is None
    th.run()
    th.end()
    assert wl._add_marker_cache is None, "stage kept with the tick off"


def test_the_tick_off_mid_manual_engine_run_keeps_no_stage(wl):
    assert wl._add_marker_click(NUC_A)
    wl._add_live_timer.stop()
    wl._preview_add()
    th = FakeThread.instances[-1]
    wl._on_add_live_toggled(False)
    th.run()
    th.end()
    assert wl._add_marker_cache is None, "stage kept with the tick off"


# --------------------------------------------------------------------------- #
# what is held while a result is on screen
# --------------------------------------------------------------------------- #
def test_a_result_holds_no_second_solution(wl):
    """The result on screen names the solution it started from without
    keeping it: once nothing else holds the first solution it is freed."""
    import gc
    import weakref

    first = _click_and_draw(wl, NUC_A)
    ref = weakref.ref(first.marker_solution)
    second = _click_and_draw(wl, NUC_B)
    assert not second.marker_diff.full
    assert wl.rendered_patches[-1] is not None, "the patches path is kept"
    assert not isinstance(getattr(second, "marker_previous", None),
                          AI.MarkerSolution)
    # the test's own holds: its result, the launches' kwargs, the threads
    del first
    wl.launched.clear()
    FakeThread.instances.clear()
    gc.collect()
    assert ref() is None, "the first solution is still held"
    assert wl._add_marker_cache["solution"] is second.marker_solution


@pytest.mark.parametrize("leave", ["cells", "anywhere"])
def test_leaving_the_nuclei_markers_drops_the_stage(wl, leave):
    _click_and_draw(wl, NUC_A)
    assert wl._add_marker_cache is not None
    if leave == "cells":
        wl.adv.preset = "cells"
    else:
        wl.adv.look = (AP.LOOK_ANYWHERE, None)
    wl._add_prune_caches()
    assert wl._add_marker_cache is None


def test_the_stage_stays_while_the_nuclei_markers_stay(wl):
    _click_and_draw(wl, NUC_A)
    wl._add_prune_caches()
    assert wl._add_marker_cache is not None


# --------------------------------------------------------------------------- #
# the console: one line per engine run
# --------------------------------------------------------------------------- #
def _engine_lines(out):
    return [ln for ln in out.splitlines()
            if ln.startswith("[ADD] markers on the whole patch:")]


def test_each_engine_run_prints_one_line(wl, capsys):
    _click_and_draw(wl, NUC_A)
    out = capsys.readouterr().out
    lines = _engine_lines(out)
    assert len(lines) == 1, out
    assert "prepared" in lines[0] and "full" in lines[0], lines[0]
    assert "pieces solved" in lines[0] and " ms" in lines[0], lines[0]
    second = _click_and_draw(wl, NUC_B)
    out = capsys.readouterr().out
    lines = _engine_lines(out)
    assert len(lines) == 1, out
    n = len(second.marker_diff.patches)
    assert "reused" in lines[0], lines[0]
    assert "{} pieces solved".format(
        second.marker_diff.n_pieces_solved) in lines[0], lines[0]
    assert "{} patches written".format(n) in lines[0], lines[0]


# --------------------------------------------------------------------------- #
# the strip's line on a whole patch
# --------------------------------------------------------------------------- #
def test_a_live_run_on_a_whole_patch_says_reading_the_patch(wl):
    from napari_epicycle.annotator.reshape_worker import stage_text

    assert wl._add_marker_click(NUC_A)
    _fire(wl)
    assert wl.adv.last("measuring") == stage_text("add_box_patch")
    wl._on_add_progress("add_box")
    assert wl.adv.last("measuring") == stage_text("add_box_patch")
    wl._add_refresh_ui()
    assert wl.adv.last("measuring") == stage_text("add_box_patch")
    wl._on_add_progress("add_find")
    assert wl.adv.last("measuring") == stage_text("add_find")
    _finish(wl)


# --------------------------------------------------------------------------- #
# the slow-patch amber with the tick on
# --------------------------------------------------------------------------- #
def test_a_large_whole_patch_with_the_tick_on_says_the_first_update(
        wl, monkeypatch):
    monkeypatch.setattr(AP, "ADD_SLOW_PATCH_VOXELS", 10)
    n = int(np.prod(wl.sync_manager.shared_labels["L"].shape))
    m = AP.millions(n)
    live = AP.ADD_NOTE_PATCH_SLOW_LIVE.format(m=m)
    assert wl._add_derive_note_text() == live
    assert AP.nuclei_patch_slow_line(n, live=True) == live
    assert "Preview" not in live and "Stop" not in live
    # the tick off: the Preview's wait, as before
    wl._add_live_enabled = False
    assert wl._add_derive_note_text() == AP.nuclei_patch_slow_line(n)
    assert wl._add_derive_note_text() == (
        AP.ADD_NOTE_PATCH_SLOW_PLAIN.format(m=m))
    # the tick on but Anywhere: no update as you click there
    wl._add_live_enabled = True
    wl.adv.look = (AP.LOOK_ANYWHERE, None)
    assert wl._add_derive_note_text() == AP.nuclei_patch_slow_line(n)


#: the label's fonts: offscreen Qt in this process has none, so the note is
#: measured in a child process with the system's
_FONT_DIR = "C:/Windows/Fonts"

_NOTE_CHILD = """
import json, os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
from napari_epicycle.annotator.advanced_features_widget import (
    AdvancedFeaturesWidget)
from packages.epicycle_annotator import add_preview as AP
w = AdvancedFeaturesWidget(mode="3d_curation")
w.resize(340, 900)
w.show()
for _ in range(5):
    app.processEvents()
w.focus_tab("Add")
lb = w._add_derive_note
lb.setText("x")
lb.setVisible(True)
for _ in range(3):
    app.processEvents()
out = {"one": lb.heightForWidth(300),
       "line": lb.fontMetrics().lineSpacing(), "rows": []}
for n in (AP.ADD_SLOW_PATCH_VOXELS + 1, 99999999, 999999999, 9999999999):
    # the label's width at the 312 px page is about 286 px
    for width in range(272, 308, 4):
        lb.setText(AP.nuclei_patch_slow_line(n))
        plain = lb.heightForWidth(width)
        lb.setText(AP.nuclei_patch_slow_line(n, live=True))
        out["rows"].append([n, width, plain, lb.heightForWidth(width)])
print("NOTEJSON" + json.dumps(out))
w.close()
"""


def test_the_live_slow_note_is_two_lines_at_the_page_width():
    """The amber with the tick on takes two lines at the page's widths --
    no more than the one with the tick off."""
    import json
    import os
    import pathlib
    import subprocess
    import sys

    repo = pathlib.Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    code = "FONT_DIR = {!r}\n".format(_FONT_DIR) + _NOTE_CHILD
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(repo),
                          env=env, capture_output=True, text=True,
                          timeout=300)
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.startswith("NOTEJSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:], proc.stderr[-3000:])
    got = json.loads(lines[-1][len("NOTEJSON"):])
    two = got["one"] + got["line"]
    for n, width, plain, live in got["rows"]:
        assert live <= plain, (n, width, plain, live)
        assert live <= two, (n, width, live, two)

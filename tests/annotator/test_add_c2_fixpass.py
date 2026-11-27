"""Cells on the Add page after the second look: what the page does, through
the real code paths.

Pinned here: the outline's skin through the real Accept and "+ Add
another"; the skin is never a labels-key target; the solver check's answer
arriving late, and missing, and never waited for on the main thread; Stop
while a whole patch's background is measured; the whole Flow 2 (Whole patch, Clear
every object, the outline follows) through the real page's rules and the
orchestrator; the outline worker's error; the outline and the wall picture
kept per patch, not per session; a late outline answer for another patch;
the main Undo during the review; the review's chip rows first; the
whole-patch words; the Accept timing line; "Showing the result…" painted
first; the run count, the Show legend, the dark rows' and the edge rows'
tooltips, the Advanced summary's full text; the slower wait before a whole
patch's outline is measured again; the nuclei channel cut on the worker.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import region as R

from .add_harness import Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_outline_wiring import (  # noqa: F401
    _CellsAdv,
    _land_info,
    _show,
    _world,
    h,
)
from .test_reshape_region import _threshold

ISO = (1.0, 1.0, 1.0)


def _measure(hh):
    """Ask the box info and let it land (the outline is measured there)."""
    hh._request_add_box_info()
    _land_info(hh)
    hh._add_refresh_ui()


def _preview(hh):
    hh._preview_add()
    th = hh._add_thread
    th.run()
    th.end()
    return hh._add_preview_res


# --------------------------------------------------------------------------- #
# the outline's skin: Accept, "+ Add another", never a labels-key target
# --------------------------------------------------------------------------- #
def test_accept_drops_the_skin_keeps_the_tick_and_add_another_redraws(h):
    _show(h)
    assert h._add_outline_layer is not None
    res = _preview(h)
    assert res is not None and res.ok, getattr(res, "refusal", None)
    h.adv.checked = list(res.ids)
    h._on_add_committed_cb = None
    h.removed.clear()
    n_added = len(h.added)
    h._accept_add()
    assert h._add_review
    assert "add_outline" in h.removed, "Accept drops the skin"
    assert h._add_outline_layer is None
    assert h.adv.show.on is True, "the tick is kept for + Add another"
    h._add_another()
    _land_info(h)
    h._add_refresh_ui()
    assert len(h.added) > n_added, "+ Add another draws it again"
    assert h.added[-1][0] == "add_outline"


def test_the_skin_is_never_a_labels_key_target(h):
    """The labels key toggles the candidate and the click balls while Add
    runs, never the outline's skin -- and the skin is an Image layer."""
    _show(h)
    layer = h._add_outline_layer
    assert layer is not None
    assert layer not in h._add_toggle_targets()
    _key, _array, kw, _layer = h.added[-1]
    assert kw["kind"] == "image"


def test_the_sync_does_not_add_a_second_skin_when_it_is_re_entered(h):
    """Appending a layer fires the layer list's events; a sync re-entered
    from there while the first layer is being added adds nothing."""
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h.adv.show.on = True
    h._on_add_show_outline(True)
    _land_info(h)
    h._add_drop_outline_layer()
    h.added.clear()
    real = h._add_temp_layer

    def reentering(key, array, **kw):
        h._add_sync_outline_layer()          # the layer list's event
        return real(key, array, **kw)

    h._add_temp_layer = reentering
    h._add_sync_outline_layer()
    assert len(h.added) == 1


def test_a_sweep_of_the_temporary_layers_forgets_the_skin(h):
    from napari_epicycle.annotator import curation_3d_widget as CW

    _show(h)
    assert h._add_outline_layer is not None
    h._temp_guide_layers = {}
    h.subviewer_manager = None
    CW.Curation3DAnnotator._remove_all_temp_layers(h)
    assert h._add_outline_layer is None and h._add_outline_layer_key is None
    h.added.clear()
    h._add_refresh_ui()
    assert h.added and h.added[-1][0] == "add_outline", "drawn again"


# --------------------------------------------------------------------------- #
# the solver check
# --------------------------------------------------------------------------- #
class _Outcome:
    def __init__(self, result):
        self.result = result


def test_the_solver_answer_arriving_late_greys_the_whole_patch(h,
                                                              monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as CW

    monkeypatch.setitem(CW._ADD_NUMBA, "ok", None)
    monkeypatch.setitem(CW._ADD_NUMBA, "why", None)
    monkeypatch.setitem(CW._ADD_NUMBA_JOB, "thread", object())  # asking
    h._start_add_whole()                      # pressed before the answer
    assert h._add_whole_patch
    CW._add_numba_answered(_Outcome(("missing", "numba is not installed")))
    h._on_add_numba_finished()
    assert h.adv.last("numba_ok") is False
    assert h.adv.last("numba_why") == "missing"
    assert h.adv.last("status_text") == AP.ADD_WHOLE_NEEDS_NUMBA
    assert h.adv.last("preview_enabled") is False


def test_the_solver_check_runs_on_its_own_thread_never_waited_for(
        monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as CW
    from packages.epicycle_annotator.reshape import compete as CP

    install_fakes(monkeypatch)
    monkeypatch.setitem(CW._ADD_NUMBA, "ok", None)
    monkeypatch.setitem(CW._ADD_NUMBA, "why", None)
    monkeypatch.setitem(CW._ADD_NUMBA_JOB, "thread", None)
    monkeypatch.setitem(CW._ADD_NUMBA_JOB, "worker", None)

    def missing():
        raise ImportError(CP.NUMBA_MISSING)

    monkeypatch.setattr(CP, "_kernels", missing)
    ch, cells = _world()
    hh = Harness(np.zeros(cells.shape, np.int32), images=ch)
    hh.adv = _CellsAdv()
    hh._add_run_guide = cells
    hh._start_add()
    th = CW._ADD_NUMBA_JOB["thread"]
    assert th is not None, "asked on a worker"
    # Cancel before the answer: the page does not wait for the check
    hh._reject_add()
    assert th.quit_called is False
    th.run()                                  # the answer lands later
    th.end()
    assert CW._ADD_NUMBA["ok"] is False and CW._ADD_NUMBA["why"] == "missing"
    # a whole patch here greys Preview and S1 says why (Whole stays live)
    assert AP.preview_block(dict(has_box=True, preset="cells",
                                 whole_patch=True, numba_ok=False,
                                 numba_why="missing")) == (
        AP.ADD_WHOLE_NEEDS_NUMBA)


def test_a_refused_set_up_is_the_answer_for_the_process(h, monkeypatch):
    """A Preview whose growth engine cannot be set up says why and keeps
    the answer, so the next Preview does not try the set-up again."""
    from napari_epicycle.annotator import curation_3d_widget as CW

    monkeypatch.setitem(CW._ADD_NUMBA, "why", None)
    calls = []

    def broken(*, announce=None):
        calls.append(1)
        raise RuntimeError("no locator")

    monkeypatch.setattr(R, "compete_is_warm", lambda: False)
    monkeypatch.setattr(R, "warm_up_compete", broken)
    h.adv.params["growth"] = "compete"
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h.adv.params["outline"] = "none"
    _measure(h)
    res = _preview(h)
    assert res is None
    assert h._add_transient == AP.ADD_NO_NUMBA_CACHE
    assert CW._ADD_NUMBA["ok"] is False and CW._ADD_NUMBA["why"] == "cache"
    assert calls == [1]
    h._add_refresh_ui()
    assert h.adv.last("numba_ok") is False


# --------------------------------------------------------------------------- #
# Stop on a whole patch
# --------------------------------------------------------------------------- #
@pytest.fixture
def hc(monkeypatch):
    install_fakes(monkeypatch)
    ch, cells = _world()
    hh = Harness(np.zeros(cells.shape, np.int32), images=ch)
    hh.adv = _CellsAdv()
    hh.ctx["spacing_source"] = "qc_settings"
    hh._add_run_guide = cells
    hh._add_dark_cache = {hh._add_dark_key(): _threshold(ch)}
    return hh


def test_stop_while_the_background_is_measured_writes_nothing(hc,
                                                              monkeypatch):
    """Stop pressed while the fill measures where the background starts:
    the stop lands before the next slab, the fill never runs, nothing is
    written."""
    hc._start_add_whole()
    _land_info(hc)
    before = hc.sync_manager.shared_labels["L"].copy()
    real = R.reshape_empty_region
    real_radii = R._background_radii
    seen = []

    def radii(*a, **k):
        hc._stop_add_preview()              # Stop while it is measured
        seen.append("radii")
        return real_radii(*a, **k)

    def wrapped(*a, progress=None, **k):
        def spy(stage, i, n):
            seen.append(stage)
            if progress is not None:
                progress(stage, i, n)
        return real(*a, progress=spy, **k)

    monkeypatch.setattr(R, "_background_radii", radii)
    monkeypatch.setattr(R, "reshape_empty_region", wrapped)
    hc._preview_add()
    th = hc._add_thread
    th.run()
    th.end()
    assert "radii" in seen, seen
    assert "add_cut" not in seen, "the fill never started"
    assert hc._add_preview_result is None
    assert hc._add_transient == AP.ADD_S1_STOPPED_PATCH
    assert np.array_equal(hc.sync_manager.shared_labels["L"], before)


# --------------------------------------------------------------------------- #
# Flow 2 through the real page and the orchestrator
# --------------------------------------------------------------------------- #
def test_flow_2_whole_patch_clear_every_object_the_outline_follows(
        widget, monkeypatch):
    install_fakes(monkeypatch)
    ch, cells = _world()
    labels = cells.copy()                     # a patch fully drawn
    hh = Harness(labels, images=ch)
    hh.adv = widget
    hh.ctx["spacing_source"] = "qc_settings"
    hh._add_run_guide = cells
    hh._add_dark_cache = {hh._add_dark_key(): _threshold(ch)}
    hh._add_session_memory = {"last_preset": "cells"}
    reset_add(widget)
    widget.refresh_add_image_combo(["membrane", "nuclei"])
    widget.set_add_preset("cells")
    widget.clear_add_touched()
    run = ("labels", "cells")
    entries = [(AP.ADD_ITEM_RUN.format(name="cells"), True, run),
               (AP.ADD_ITEM_CLICKS, True, ("paint", None))]
    # the patch's other runs, as the workspace would list them
    hh._add_marker_source_entries = lambda: list(entries)
    widget.set_add_marker_sources(entries)
    widget.add_marker_source_combo.setCurrentIndex(0)
    widget.add_wall_channel_combo.setCurrentIndex(
        widget.add_wall_channel_combo.findText("membrane"))
    live = set()
    hh._on_discard_garbage_cb = lambda ids, **kw: live.add(
        kw["extra"]["add_clear_seq"])
    hh._get_live_add_clears_cb = lambda: set(live)
    try:
        hh._start_add_whole()
        assert hh._add_whole_patch
        # the page's own rules: a whole patch shows Grow by (the box's pick
        # kept), and an untouched Tissue outline is the shrink-wrap (no
        # Clear yet)
        assert not widget._add_grow_row.isHidden()
        assert widget.get_add_grow() == AP.GROW_FLOOD
        assert widget.get_add_outline() == AP.OUTLINE_SHRINK
        assert hh._add_outline_mode() == AP.OUTLINE_SHRINK
        _land_info(hh)
        assert hh._add_outline_cache["key"][0] == "shrink"
        # Clear every object in the patch
        hh._apply_add_clear(AP.clear_plan(labels=labels, mode="patch"))
        assert not labels.any()
        hh._add_transient = None          # the Clear's own line, read
        hh._add_refresh_ui()
        # the untouched pick follows: the outline of the objects cleared
        assert widget.get_add_outline() == AP.OUTLINE_CLEARED
        assert hh._add_outline_mode() == AP.OUTLINE_CLEARED
        assert hh._add_cached_outline() is None, "measured again"
        assert widget._add_ui.get("outline_pending") is True
        _land_info(hh)
        hh._add_refresh_ui()
        cache = hh._add_outline_cache
        assert cache["key"][0] == "cleared"
        assert np.array_equal(cache["outline"],
                              AP.measure_outline(
                                  {"mode": "cleared",
                                   "shape": labels.shape,
                                   "cleared": list(
                                       hh._add_clear_masks.values())},
                                  spacing=ISO))
        status = widget.add_status_label.text()
        assert status.startswith("Whole patch "), status
        assert widget.add_preview_btn.isEnabled()
    finally:
        hh._reject_add()
        widget.set_add_marker_sources([])
        reset_add(widget)


# --------------------------------------------------------------------------- #
# the outline worker's error
# --------------------------------------------------------------------------- #
def test_an_outline_worker_error_is_said_and_never_sticks(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h.adv.show.on = True
    h._request_add_box_info()
    w = h._add_info_worker
    assert getattr(w, "_epicycle_outline_job", False)
    h.sender = w
    h._on_add_info_error("MemoryError: 0.68 GiB")
    h.sender = None
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_S1_OUTLINE_FAILED
    assert h.adv.last("outline_refused") is True
    # another outline is measured again
    h.adv.params["outline"] = "none"
    h._add_refresh_ui()
    assert h.adv.last("status_text") != AP.ADD_S1_OUTLINE_FAILED


# --------------------------------------------------------------------------- #
# the caches: per patch, not per session
# --------------------------------------------------------------------------- #
def test_the_outline_and_the_walls_survive_cancel_on_the_same_patch(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    _measure(h)
    outline = h._add_cached_outline()
    assert outline is not None
    h._add_elev_cache = {"key": h._add_elev_key(), "elev": np.zeros(3)}
    h._reject_add()                           # Cancel
    assert h._add_elev_cache is not None, "the walls are the patch's"
    cache = h._add_outline_cache
    assert cache["outline"] is None and cache["packed"] is not None
    # the next session on this patch: the worker unpacks it, nothing is
    # measured again
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    kw = h._add_gather_info()
    assert kw["outline_job"]["mode"] == "packed"
    _measure(h)
    assert np.array_equal(h._add_cached_outline(), outline)
    # another patch: both go
    h._teardown_add_for_switch()
    assert h._add_outline_cache is None and h._add_elev_cache is None


def test_the_caches_drop_when_their_inputs_change(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    _measure(h)
    assert h._add_outline_cache is not None
    h._add_elev_cache = {"key": h._add_elev_key(), "elev": np.zeros(3)}
    h.adv.params["bridge_um"] = 12.0          # the shrink-wrap's input
    h._add_refresh_ui()
    assert h._add_outline_cache is None
    assert h._add_elev_cache is not None
    h.adv.wall = dict(h.adv.wall, chain="denoise")   # the walls' input
    h._add_refresh_ui()
    assert h._add_elev_cache is None


def test_another_outline_pick_keeps_the_cache(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    _measure(h)
    key = h._add_outline_cache["key"]
    h.adv.params["outline"] = "none"
    h._add_refresh_ui()
    assert h._add_outline_cache["key"] == key


def test_a_late_outline_answer_for_another_patch_is_dropped(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h._request_add_box_info()
    th = h._add_info_thread
    h.ctx["patch_name"] = "Q"                # switched while it ran
    h._add_forget_clears()
    th.run()
    th.end()
    assert getattr(h, "_add_outline_cache", None) is None


# --------------------------------------------------------------------------- #
# the main Undo during the review
# --------------------------------------------------------------------------- #
def _review(h, sessions):
    h._add_session = [dict(e) for e in sessions]
    h._add_piece_ids = [i for e in sessions for i in e["ids"]]
    h._add_review = True
    h._add_running = False


def test_the_main_undo_takes_the_last_accept_off_the_review(h):
    _review(h, [{"ids": [10, 11], "sizes": {10: 5, 11: 6}},
                {"ids": [20], "sizes": {20: 7}}])
    h._add_after_accept_undone(20)
    assert h._add_review
    assert h._add_piece_ids == [10, 11]
    assert h.adv.last("piece_ids") == [10, 11]
    assert h.adv.last("status_text") == AP.ADD_S1_UNDO_TOOK_BACK
    # an Accept the review does not list: nothing moves
    h._add_after_accept_undone(999)
    assert h._add_piece_ids == [10, 11]
    # the last one: the review closes
    h._add_after_accept_undone(10)
    assert not h._add_review and h._add_session == []
    assert h.adv.last("review_active") is False
    assert h.adv.last("status_text") == AP.ADD_S1_UNDO_TOOK_BACK_ALL


def test_undo_all_is_the_pages_own_and_the_hook_stays_quiet(h):
    _review(h, [{"ids": [10], "sizes": {10: 5}}])
    calls = []

    def undo_session(n):
        # the controller pops each Accept through the main Undo path
        h._add_after_accept_undone(10)
        calls.append(n)
        return n

    h._on_add_undo_session_cb = undo_session
    h._add_confirm = lambda text: True
    h._undo_add_all()
    assert calls == [1]
    assert h.adv.last("status_text") == AP.ADD_UNDO_DONE


def test_the_review_lists_the_chip_rows_first(h):
    flags = {30: ("two",), 12: ("cut",)}
    _review(h, [{"ids": [5, 12, 30, 7], "sizes": {},
                 "flags": flags, "owners": {30: 1, 31: 1, 12: 2},
                 "owner_run": "A"},
                {"ids": [31], "sizes": {}, "flags": {31: ("two",)},
                 "owners": {31: 1}, "owner_run": "A"}])
    chips = h._add_review_chips(h._add_piece_ids)
    order = h._add_review_order(h._add_piece_ids, chips.get("piece_chips"))
    assert order[:3] == [30, 31, 12]
    assert order[3:] == [5, 7]


# --------------------------------------------------------------------------- #
# whole-patch words, the Accept line, "Showing the result…"
# --------------------------------------------------------------------------- #
def test_the_whole_patch_says_patch_not_box(hc):
    from napari_epicycle.annotator.reshape_worker import stage_text

    hc._start_add_whole()
    hc._on_add_progress("add_box", 0, 4)
    assert hc.adv.pushes[-1]["stage_text"] == stage_text("add_box_patch")
    assert "patch" in stage_text("add_box_patch")
    s = hc._add_s1_state()
    assert AP.measuring_line(s) == AP.ADD_S1_MEASURING_PATCH
    hc._add_busy = True
    hc._add_refresh_ui()
    assert hc.adv.last("status_text") == AP.ADD_S1_WORKING_PATCH
    hc._add_busy = False
    hc._on_add_stopped()
    assert hc._add_transient == AP.ADD_S1_STOPPED_PATCH


def test_accept_prints_its_main_thread_steps_and_shows_first(h, capsys):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h.adv.params["outline"] = "none"
    _measure(h)
    texts = []
    real = h._push_add_ui

    def spy(**kw):
        if "status_text" in kw:
            texts.append(kw["status_text"])
        return real(**kw)

    h._push_add_ui = spy
    res = _preview(h)
    assert res is not None and res.ok
    assert AP.ADD_S1_SHOWING in texts, "painted before the result is drawn"
    assert texts.index(AP.ADD_S1_SHOWING) < len(texts) - 1
    h.adv.checked = list(res.ids)
    h._on_add_committed_cb = None
    capsys.readouterr()
    h._accept_add()
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith(
        "[ADD-PERF] accept")]
    assert line, out
    for step in ("mask=", "write=", "stats=", "commit=", "sync=",
                 "review=", "refresh=", "total="):
        assert step in line[0], (step, line[0])


# --------------------------------------------------------------------------- #
# the slower wait before a whole patch's outline is measured again
# --------------------------------------------------------------------------- #
class _Timer:
    started = []

    def __init__(self, *_a):
        self.timeout = self

    def setSingleShot(self, _b):            # noqa: N802 - Qt name
        pass

    def connect(self, _fn):
        pass

    def start(self, ms):
        _Timer.started.append(ms)


def test_a_whole_patch_outline_waits_longer_before_it_is_measured(
        hc, monkeypatch):
    import qtpy.QtCore

    from napari_epicycle.annotator import curation_3d_widget as CW

    monkeypatch.setattr(qtpy.QtCore, "QTimer", _Timer)
    _Timer.started = []
    hc.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    CW.Curation3DAnnotator._schedule_add_histogram(hc)
    hc._start_add_whole()
    CW.Curation3DAnnotator._schedule_add_histogram(hc)
    assert _Timer.started == [150, 300]


# --------------------------------------------------------------------------- #
# the nuclei channel's Auto is cut on the worker
# --------------------------------------------------------------------------- #
def test_the_nuclei_channel_goes_to_the_worker_uncut(hc):
    hc.adv.source = ("nuclei_masks", None)
    hc.adv.nuclei_channel = "nuclei"
    hc.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    kw = hc._add_gather_info()
    assert kw["nuclei_roi"] is None
    assert kw["nuclei_source"] is hc.viewer.layers[1].data
    info = AP.add_box_info(**kw)
    want = AP.add_box_info(**dict(kw, nuclei_source=None, nuclei_roi=np.array(
        hc.viewer.layers[1].data[kw["roi"]], np.float64)))
    assert info.nuclei_auto is not None
    assert info.nuclei_auto == want.nuclei_auto


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def test_the_page_says_patch_in_the_pending_run_count(widget):
    reset_add(widget)
    widget.set_add_preset("cells")
    try:
        widget.update_add_ui_state(box_active=True, whole_patch=True,
                                   run_count=())
        assert widget.add_run_count_label.text() == (
            AP.ADD_RUN_COUNT_PATCH_PENDING)
        widget.update_add_ui_state(whole_patch=False, run_count=())
        assert widget.add_run_count_label.text() == AP.ADD_RUN_COUNT_PENDING
    finally:
        widget.update_add_ui_state(box_active=False)
        reset_add(widget)


def test_the_show_legend_says_what_is_true(widget):
    reset_add(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(box_active=True)
    widget.add_outline_combo.setCurrentIndex(
        AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
    widget.add_show_outline_cb.setChecked(True)
    leg = widget._add_show_outline_legend
    try:
        widget.update_add_ui_state(outline_pending=True)
        assert leg.text() == AP.ADD_OUTLINE_PENDING
        # measured, but the skin not drawn yet
        widget.update_add_ui_state(outline_pending=False,
                                   outline_drawn=False)
        assert leg.text() == AP.ADD_OUTLINE_PENDING
        widget.update_add_ui_state(outline_drawn=True)
        assert leg.text() == AP.ADD_HELP_SHOW_OUTLINE
        # stays drawn over candidates (the legend does not change)
        widget.update_add_ui_state(candidate_ids=[5],
                                   candidate_sizes={5: 10})
        assert leg.text() == AP.ADD_HELP_SHOW_OUTLINE
        widget.update_add_ui_state(candidate_ids=[], candidate_sizes={})
        # refused
        widget.update_add_ui_state(outline_refused=True)
        assert leg.text() == AP.ADD_SHOW_OUTLINE_REFUSED
    finally:
        widget.update_add_ui_state(outline_refused=False,
                                   outline_drawn=True, box_active=False)
        widget.add_show_outline_cb.setChecked(False)
        reset_add(widget)


def test_grey_rows_carry_their_reason_in_their_tooltip(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    reset_add(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(box_active=True)
    try:
        widget.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        assert not widget.add_dark_outside_cb.isEnabled()
        assert widget.add_dark_outside_cb.toolTip() == AP.ADD_DARK_BY_OUTLINE
        assert widget.add_dark_clearance_sb.toolTip() == (
            AP.ADD_DARK_BY_OUTLINE)
        widget.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        assert widget.add_dark_outside_cb.toolTip() == (
            ADD_HELP_LINES["add_dark_outside_cb"])
        # the edge rows on a whole patch: hidden, not grey
        widget.update_add_ui_state(whole_patch=True)
        assert widget.add_clear_border_cb.isHidden()
        assert widget._add_edge_zone_row.isHidden()
        widget.update_add_ui_state(whole_patch=False)
        assert not widget.add_clear_border_cb.isHidden()
        assert widget.add_clear_border_cb.isEnabled()
        assert widget.add_clear_border_cb.toolTip() == (
            ADD_HELP_LINES["add_clear_border_cb"])
    finally:
        widget.update_add_ui_state(box_active=False, whole_patch=False)
        reset_add(widget)


def test_one_amber_slot_on_a_cells_box(widget):
    """A derive note takes the one amber slot; the spacing note and the
    dark-off note give way (the spacing note moves to the Advanced help
    slot, the dark-off reason is there already)."""
    reset_add(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(box_active=True)
    widget.add_outline_combo.setCurrentIndex(
        AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
    widget.add_dark_outside_cb.setChecked(False)
    ambers = (widget.add_spacing_note, widget.add_dark_off_note,
              widget._add_derive_note)
    try:
        widget.update_add_ui_state(default_spacing=True, derive_note="")
        shown = [w for w in ambers if not w.isHidden()]
        assert shown == [widget.add_dark_off_note]
        assert AP.ADD_NOTE_DEFAULT_SPACING_BOXED in (
            widget._add_advanced_help.text())
        widget.update_add_ui_state(derive_note=AP.ADD_NOTE_HALFSIZE_BOXED)
        shown = [w for w in ambers if not w.isHidden()]
        assert shown == [widget._add_derive_note]
        assert AP.ADD_DARK_OFF in widget._add_advanced_help.text()
        assert AP.ADD_NOTE_DEFAULT_SPACING_BOXED in (
            widget._add_advanced_help.text())
        # on a whole patch the dark-off note says patch, not box faces
        widget.update_add_ui_state(derive_note="", whole_patch=True)
        assert widget.add_dark_off_note.text() == (
            AP.ADD_NOTE_DARK_OFF_LEAK_WHOLE)
        assert AP.ADD_DARK_OFF not in widget._add_advanced_help.text()
    finally:
        widget.update_add_ui_state(default_spacing=False, derive_note="",
                                   whole_patch=False, box_active=False)
        widget.add_dark_outside_cb.setChecked(True)
        reset_add(widget)


def test_the_advanced_summary_keeps_its_full_text_in_the_tooltip(widget):
    reset_add(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(box_active=True)
    try:
        widget.add_grow_combo.setCurrentIndex(
            AP.GROW_KEYS.index(AP.GROW_SPREAD))
        widget._refresh_add_summaries()
        tip = widget._add_advanced_toggle.toolTip()
        # the whole summary rides in the tooltip, its last part included
        # (the smoothing); the ball sits on the Markers card, never here
        assert widget.add_auto_smooth_cb.isChecked()
        tail = "smooth {:.1f}".format(widget.add_smooth_sigma_sb.value())
        assert "membrane included" in tip, tip
        assert tip.rstrip().endswith(tail), tip
        assert "ball" not in tip, tip
    finally:
        widget.add_grow_combo.setCurrentIndex(0)
        widget.update_add_ui_state(box_active=False)
        reset_add(widget)

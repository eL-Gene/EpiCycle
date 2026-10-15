"""The Add preview worker and box-info worker lifecycles (invariants 9-11, B2.3).

The real orchestrator methods on a Qt-free harness (``add_harness``); the
thread is a ``FakeThread`` whose ``started`` / ``finished`` the test emits,
and the worker is the real ``RefineWorker`` (``run()`` called synchronously,
no pytest-qt).
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread, Harness, install_fakes, nuclei_volume
from .test_add_nuclei_golden import box_of


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    img, labels = nuclei_volume()
    hh = Harness(labels, images={"nuc": img})
    hh.place_box(box_of(labels.shape))
    return hh


def _launch(h):
    h._preview_add()
    assert h._add_busy
    return FakeThread.instances[-1], h._add_worker


# --------------------------------------------------------------------------- #
# a normal run
# --------------------------------------------------------------------------- #
def test_a_preview_runs_and_lands_and_busy_holds_until_the_thread_ends(h):
    th, w = _launch(h)
    th.run()
    assert h._add_preview_result is not None
    assert h._add_preview_roi == tuple(h._add_roi)
    assert h._add_busy, "busy until the thread is gone (4.6)"
    assert h.adv.last("preview_enabled") is False
    th.end()
    assert not h._add_busy
    assert h._add_thread is None
    assert h.adv.last("preview_enabled") is True
    assert h.rendered and h.rendered[-1][1] == tuple(h._add_roi)


def test_the_event_reaching_fn_is_the_workers_own(h, monkeypatch):
    seen = {}

    def spy(**kw):
        seen["ev"] = kw.get("cancel_event")
        return AP.AddPreviewResult(status="cancelled", roi=kw["roi"])

    monkeypatch.setattr(AP, "run_add_preview", spy)
    th, w = _launch(h)
    th.run()
    assert seen["ev"] is w.cancel_event


def test_every_run_exit_emits(h, monkeypatch):
    def boom(**kw):
        raise ValueError("raw text that must not reach the page")

    monkeypatch.setattr(AP, "run_add_preview", boom)
    th, _w = _launch(h)
    th.run()
    assert h._add_transient == AP.ADD_S1_FAILED
    assert "raw text" not in (h.adv.last("status_text") or "")


# --------------------------------------------------------------------------- #
# stale (the five legs + the session)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("leg", ["token", "labels", "patch", "epoch", "roi",
                                 "session"])
def test_a_stale_finish_is_discarded(h, leg):
    th, w = _launch(h)
    if leg == "token":
        h.sender = w
        h._add_token += 1
    elif leg == "labels":
        h._active_labels_name = "other"
    elif leg == "patch":
        h.ctx["patch_name"] = "Q"
    elif leg == "epoch":
        h.ctx["run_epoch"] = 2
    elif leg == "roi":
        h._add_roi = (slice(0, 5),) * 3
    elif leg == "session":
        h._add_running = False
    th.run()
    assert h._add_preview_result is None
    assert h.rendered == []


# --------------------------------------------------------------------------- #
# Stop
# --------------------------------------------------------------------------- #
def test_stop_before_run_emits_and_writes_nothing(h):
    th, w = _launch(h)
    h._stop_add_preview()
    assert h._add_stopping and w.cancel_event.is_set()
    th.run()                                   # pre-run cancel: finished(None)
    assert h._add_transient == AP.ADD_S1_STOPPED
    assert h._add_preview_result is None
    assert h._add_busy                         # until the thread is gone
    th.end()
    assert not h._add_busy
    assert h.adv.last("preview_enabled") is True


def test_stop_during_run_routes_by_the_flag_not_the_status(h, monkeypatch):
    """A Stop inside an unpolled span comes back ``ok``: routed to Stopped."""
    def ok_anyway(**kw):
        h._stop_add_preview()                  # lands while fn runs
        return AP.AddPreviewResult(status="ok", roi=kw["roi"],
                                   result=np.ones((2, 2, 2), np.int32),
                                   ids=[1], sizes={1: 8})

    monkeypatch.setattr(AP, "run_add_preview", ok_anyway)
    th, _w = _launch(h)
    th.run()
    assert h._add_transient == AP.ADD_S1_STOPPED
    assert h._add_preview_result is None
    assert h.rendered == []


def test_stop_is_a_state_no_wait(h):
    th, _w = _launch(h)
    h._stop_add_preview()
    assert th.quit_called is False, "Stop must not join the thread"


def test_an_error_after_stop_is_stopped_not_failed(h, monkeypatch):
    def boom(**kw):
        h._stop_add_preview()
        raise RuntimeError("x")

    monkeypatch.setattr(AP, "run_add_preview", boom)
    th, _w = _launch(h)
    th.run()
    assert h._add_transient == AP.ADD_S1_STOPPED


def test_stop_then_preview_waits_for_the_thread(h):
    th, _w = _launch(h)
    h._stop_add_preview()
    th.run()
    n = len(FakeThread.instances)
    h._preview_add()
    assert len(FakeThread.instances) == n, "no second worker yet"
    assert h._add_transient == AP.ADD_S1_BUSY_REFUSED
    th.end()
    h._preview_add()
    assert len(FakeThread.instances) == n + 1


# --------------------------------------------------------------------------- #
# launch failure, close mid-preview
# --------------------------------------------------------------------------- #
def test_a_start_that_raises_releases_busy_and_keeps_the_session(h):
    FakeThread.fail_start = True
    h._preview_add()
    assert not h._add_busy
    assert h._add_transient == AP.ADD_S1_FAILED
    assert h._add_running


def test_close_mid_preview_cancels_and_a_late_finish_is_stale(h):
    th, w = _launch(h)
    h._reject_add()
    assert w.cancel_event.is_set()
    assert h._add_thread is None and not h._add_busy
    assert th.quit_called
    th.run()                                   # the late answer
    assert h._add_preview_result is None
    assert h.rendered == []


def test_an_unstoppable_thread_is_parked(h):
    th, w = _launch(h)
    FakeThread.stuck = True
    h._add_abandon_workers()
    assert (th, w) in h._refine_parked_threads
    assert th.finished.cbs == []


# --------------------------------------------------------------------------- #
# every busy refusal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("call", [
    lambda h: h._on_add_box_changed(10, 10, 10, 4, 4, 4),
    lambda h: h._recenter_add_box_on_crosshair(),
    lambda h: h._set_add_box_size_fraction(0.25),
    lambda h: h._on_add_clahe_toggled(True),
    lambda h: h._on_add_threshold_changed(0.3),
    lambda h: h._on_add_advanced_changed(),
    lambda h: h._on_add_otsu_reset(),
    lambda h: h._preview_add(),
    # B2 fix pass L-L1: the panel disables these while busy; belt and braces
    lambda h: h._on_add_preset_changed("cells"),
    lambda h: h._on_add_source_changed(("paint", None)),
])
def test_every_change_is_refused_while_busy(h, call):
    th, _w = _launch(h)
    box = h._add_box
    roi = h._add_roi
    call(h)
    assert h._add_box == box and h._add_roi == roi
    assert h._add_transient == AP.ADD_S1_BUSY_REFUSED
    assert not _w.cancel_event.is_set(), "a refusal cancels nothing"


def test_busy_refused_gives_way_at_the_end_of_the_run(h):
    th, _w = _launch(h)
    h._recenter_add_box_on_crosshair()
    th.run()
    th.end()
    assert h._add_transient != AP.ADD_S1_BUSY_REFUSED


# --------------------------------------------------------------------------- #
# the box-info worker
# --------------------------------------------------------------------------- #
@pytest.fixture
def cells(monkeypatch):
    install_fakes(monkeypatch)
    from .test_reshape_region import _tissue

    mem, nuc, cells = _tissue()
    hh = Harness(np.zeros_like(cells), images={"mem": mem, "nuc": nuc},
                 spacing=(0.5, 0.5, 0.5))
    hh.adv.preset = "cells"
    hh.adv.source = ("labels", "run1")
    hh._add_run_guide = cells.copy()
    hh.place_box(tuple(slice(0, n) for n in cells.shape))
    return hh


def test_one_info_thread_at_a_time_and_a_dirty_relaunch(cells):
    cells._request_add_box_info()
    assert len(FakeThread.instances) == 1
    cells._request_add_box_info()
    cells._request_add_box_info()
    assert len(FakeThread.instances) == 1
    assert cells._add_info_dirty
    th = FakeThread.instances[0]
    th.run()
    th.end()
    assert len(FakeThread.instances) == 2, "relaunched ONCE"
    assert not cells._add_info_dirty


def test_the_info_answer_lands_and_caches_the_dark_threshold(cells):
    cells._request_add_box_info()
    th = FakeThread.instances[0]
    th.run()
    assert cells._add_info is not None
    assert cells._add_info.run_count == (8, 8)
    assert cells._add_cached_dark() is not None
    assert cells.adv.last("run_count") == (8, 8)


def test_a_stale_info_answer_is_dropped(cells):
    cells._request_add_box_info()
    th = FakeThread.instances[0]
    cells._add_roi = (slice(0, 4),) * 3        # the box moved
    th.run()
    assert cells._add_info is None


def test_the_info_worker_is_cleaned_at_every_exit(cells):
    cells._request_add_box_info()
    w = cells._add_info_worker
    cells._reject_add()
    assert cells._add_info_thread is None
    assert w.cancel_event.is_set()


def test_preview_waits_for_the_dark_threshold(cells):
    cells._add_info = AP.AddBoxInfo(empty_share=1.0, run_count=(8, 8))
    cells._preview_add()
    assert not cells._add_busy
    assert cells._add_transient == AP.ADD_S1_MEASURING


def test_a_cells_preview_runs_through_the_worker(cells):
    cells._request_add_box_info()
    FakeThread.instances[0].run()
    FakeThread.instances[0].end()
    cells._preview_add()
    assert cells._add_busy, cells._add_transient
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert cells._add_preview_result is not None
    res = cells._add_preview_res
    assert res.ok and res.source == "labels"
    assert min(res.ids) >= 1


def test_preview_is_the_fifth_batch_door(h):
    """A batch's finish or Reject rewrites the volume the preview
    would measure, so Preview is refused while Batch Refine runs / stages."""
    from packages.epicycle_annotator.batch_refine import ADD_BATCH_BUSY

    h._batch_staged = True
    h._preview_add()
    assert FakeThread.instances == []
    assert h.adv.last("status_text") == ADD_BATCH_BUSY


# --------------------------------------------------------------------------- #
# B4.2 -- the Brightness histogram and its Auto land with the box info
# --------------------------------------------------------------------------- #
def _otsu_of_box(h):
    from skimage.filters import threshold_otsu

    img = h.viewer.layers[0].data
    return float(threshold_otsu(np.asarray(img)[h._add_roi].astype(np.float64)))


def _land_info(h):
    h._request_add_box_info()
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    return th


def test_the_histogram_and_auto_land_with_the_info(h):
    assert h.adv.last("histogram_data") is None
    _land_info(h)
    hist = h.adv.last("histogram_data")
    assert hist is not None and sum(hist["counts"]) > 0
    assert len(hist["edges"]) == AP.ADD_HIST_BINS + 1
    want = _otsu_of_box(h)
    assert h._add_info.brightness_auto == want
    assert h.adv.last("brightness_auto") == want
    # an untouched threshold follows the landed Auto, at full precision
    assert h._add_threshold_value == want
    assert h.adv.last("threshold_value") == want


def test_nothing_is_measured_on_the_main_thread(h, monkeypatch):
    """The real `_compute_add_histogram` and the Auto button compute
    nothing: they show pending / read the landed value."""
    from napari_epicycle.annotator import curation_3d_widget as CW
    from packages.epicycle_annotator import add_segment

    def boom(*a, **k):
        raise AssertionError("measured on the main thread")

    monkeypatch.setattr(add_segment, "add_threshold_space", boom)
    monkeypatch.setattr(add_segment, "_resolve_threshold", boom)
    CW.Curation3DAnnotator._compute_add_histogram(h, reset_threshold=True)
    assert h.adv.last("histogram_data") == {"counts": [], "edges": []}
    assert np.isnan(h.adv.last("brightness_auto"))
    h._on_add_otsu_reset()                  # pending: nothing to read
    assert h._add_threshold_value is None


def test_auto_reads_the_landed_value(h):
    _land_info(h)
    h._add_threshold_value = 0.01           # the user moved the line
    h._on_add_otsu_reset()
    assert h._add_threshold_value == _otsu_of_box(h)


def test_a_typed_threshold_is_not_overwritten_by_a_landing(h):
    h._add_threshold_value = 0.42           # typed / dragged before it landed
    _land_info(h)
    assert h._add_threshold_value == 0.42
    assert h.adv.last("histogram_data") is not None   # the bars still land


def test_a_dirty_relaunch_refreshes_the_histogram(h):
    h._request_add_box_info()
    first = FakeThread.instances[-1]
    # Local contrast turned on while the first job runs: a new value space
    h.adv.params["clahe"] = True
    h._add_threshold_value = None
    h._request_add_box_info()
    assert h._add_info_dirty
    first.run()
    # the first answer was measured in the OLD space: its bars and Auto
    # are not shown
    assert h.adv.last("histogram_data") is None
    assert h._add_threshold_value is None
    first.end()
    second = FakeThread.instances[-1]
    assert second is not first
    second.run()
    second.end()
    hist = h.adv.last("histogram_data")
    assert hist is not None and hist["edges"][-1] <= 1.0 + 1e-9
    assert h._add_threshold_value == h._add_info.brightness_auto


def test_a_stale_histogram_is_dropped(h):
    h._request_add_box_info()
    th = FakeThread.instances[-1]
    h._add_roi = (slice(0, 5),) * 3            # the box moved
    th.run()
    assert h.adv.last("histogram_data") is None
    assert h._add_threshold_value is None


# --------------------------------------------------------------------------- #
# A5 -- the markers epoch; a tick while busy
# --------------------------------------------------------------------------- #
def test_a_finish_after_the_markers_changed_is_discarded(h, capsys):
    """Invariant 22."""
    th, _w = _launch(h)
    h._add_markers_epoch = getattr(h, "_add_markers_epoch", 0) + 1
    th.run()
    assert h._add_preview_result is None
    assert h.rendered == []
    assert "markers changed" in capsys.readouterr().out


def test_a_tick_while_busy_is_refused_and_the_list_pushed_back(h):
    h._add_click_markers = [{"zyx": (5, 5, 5), "id": 900, "ticked": True}]
    th, w = _launch(h)
    e0 = getattr(h, "_add_markers_epoch", 0)
    h._on_add_click_tick(900, False)
    assert h._add_click_markers[0]["ticked"] is True
    assert h._add_transient == AP.ADD_S1_BUSY_REFUSED
    assert h.adv.last("click_markers") == [(900, (5, 5, 5), True)]
    assert getattr(h, "_add_markers_epoch", 0) == e0
    assert not w.cancel_event.is_set(), "a refusal cancels nothing"

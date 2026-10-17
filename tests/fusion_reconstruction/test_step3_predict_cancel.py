"""Step 3 Predict: Cancel stops after the current slice.

Everything is synchronous and untimed: the worker's ``run()`` is called
directly with direct (same-thread) signal connections, and the fake
``predict_fn`` reaches the worker through a holder so it can click Cancel
from inside a slice report.  The controller slots run on a bare controller
(``__new__``) with a recording status line.
"""

from __future__ import annotations

import time

import pytest

from napari_epicycle.fusion_reconstruction import _step3_pr_text as _pr_text
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
    _SerialPredictionWorker,
)
from napari_epicycle.ui_shell import theme as _theme


def _cfgs(n):
    return [{"patch_name": f"p{i + 1}"} for i in range(n)]


class _Run:
    """One worker, its fake predict function and a record of its signals."""

    def __init__(self, n_patches, slices, script=None):
        self.holder = {}
        self.events = []
        self.calls = []
        self.slices = slices
        self.script = script or (lambda name, view, cur: None)
        self.worker = _SerialPredictionWorker(self._predict, _cfgs(n_patches))
        self.holder["w"] = self.worker
        w = self.worker
        w.patch_started.connect(lambda n, i, t: self.events.append(("started", n)))
        w.patch_completed.connect(lambda n, r: self.events.append(("done", n)))
        w.error.connect(lambda n, m: self.events.append(("error", n)))
        w.view_progress.connect(
            lambda n, v, c, t: self.events.append(("view", n, c, t)))
        w.cancelled.connect(lambda d, t: self.events.append(("cancelled", d, t)))
        w.all_completed.connect(lambda: self.events.append(("all",)))

    def _predict(self, progress_cb, patch_name, **_kw):
        self.calls.append(patch_name)
        for cur in range(1, self.slices):
            self.script(patch_name, "xy", cur)
            progress_cb("xy", cur, self.slices)
        # the end-of-view report, as the real segmenter sends it
        self.script(patch_name, "xy", self.slices)
        progress_cb("xy", self.slices, self.slices)
        return {"views": {}}

    def kinds(self):
        return [e[0] for e in self.events]


def test_a_cancel_inside_a_slice_stops_the_patch_and_the_run():
    def script(name, view, cur):
        if cur == 3:
            run.holder["w"].request_pause()

    run = _Run(2, 10, script)
    run.worker.run()
    assert run.calls == ["p1"]
    kinds = run.kinds()
    assert "error" not in kinds and "done" not in kinds
    assert ("cancelled", 0, 2) in run.events
    # cancelled arrives before all_completed
    assert kinds[-2:] == ["cancelled", "all"]
    # no slice report was emitted after the click
    assert max(e[2] for e in run.events if e[0] == "view") == 2


def test_a_cancel_between_patches_keeps_the_finished_one():
    run = _Run(2, 4)
    run.worker.patch_completed.connect(
        lambda n, r: run.holder["w"].request_pause())
    run.worker.run()
    assert run.calls == ["p1"]
    assert run.kinds().count("done") == 1
    assert run.kinds()[-2:] == ["cancelled", "all"]
    assert ("cancelled", 1, 2) in run.events


def test_no_cancel_means_no_cancelled_signal():
    run = _Run(2, 4)
    run.worker.run()
    assert run.calls == ["p1", "p2"]
    assert "cancelled" not in run.kinds()
    assert run.kinds()[-1] == "all"
    assert run.kinds().count("done") == 2


def test_a_cancel_in_the_last_slice_of_the_last_patch_lets_it_finish():
    def script(name, view, cur):
        if name == "p2" and cur == 4:      # the last slice report is 4 of 4
            run.holder["w"].request_pause()

    run = _Run(2, 4, script)
    run.worker.run()
    assert run.kinds().count("done") == 2
    assert "cancelled" not in run.kinds()
    assert run.kinds()[-1] == "all"


def test_the_end_of_view_report_never_raises():
    holder = {}
    seen = []

    def fn(progress_cb, patch_name, **_kw):
        holder["w"].request_pause()
        progress_cb("xy", 5, 5)     # already complete: must not raise
        seen.append("reached")
        return {}

    holder["w"] = _SerialPredictionWorker(fn, _cfgs(1))
    holder["w"].run()
    assert seen == ["reached"]


def test_an_error_then_a_cancel_counts_only_finished_patches():
    class _Boom(_Run):
        def _predict(self, progress_cb, patch_name, **kw):
            self.calls.append(patch_name)
            if patch_name == "p1":
                raise RuntimeError("bad patch")
            self.calls.pop()        # the base class records the call itself
            return super()._predict(progress_cb, patch_name, **kw)

    def script(name, view, cur):
        if name == "p2" and cur == 2:
            run.holder["w"].request_pause()

    run = _Boom(3, 6, script)
    run.worker.run()
    assert ("error", "p1") in run.events
    assert run.calls == ["p1", "p2"]
    assert "done" not in run.kinds()
    assert ("cancelled", 0, 3) in run.events
    assert run.kinds()[-2:] == ["cancelled", "all"]


# ---------------------------------------------------------------- controller


class _Bar:
    def __init__(self):
        self.value = None
        self.fmt = None

    def setValue(self, v):
        self.value = v

    def maximum(self):
        return 100

    def setFormat(self, f):
        self.fmt = f


class _Btn:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, v):
        self.enabled = v


class _Ui:
    def __init__(self):
        self.pipeline_progress = _Bar()
        self.pipeline_cancel_btn = _Btn()
        self.predict_selected_btn = _Btn()
        self.reconstruct_selected_btn = _Btn()


class _Worker:
    def __init__(self, requested=False):
        self._cancel_requested = requested
        self.paused = 0

    def request_pause(self):
        self.paused += 1
        self._cancel_requested = True


def _ctl(requested=False):
    c = C.__new__(C)
    c.ui = _Ui()
    c._run_epoch = 0
    c._serial_pred_launch_epoch = 0
    c._predicting_patches = {"p2"}
    c._pipeline_cancelled = None
    c._pipeline_errors = 0
    c._pipeline_patch_count = 3
    c._pipeline_start_time = time.monotonic()
    c._pipeline_slices_done = 0
    c._pipeline_total_slices = 30
    c._prediction_running = True
    c._reconstruction_running = False
    c._serial_prediction_worker = _Worker(requested)
    c.status = []
    c.log = []
    c._pr_status = lambda text, kind="neutral": c.status.append((text, kind))
    c._append_log = c.log.append
    c._refresh_patch_state_list = lambda: c.log.append("<refresh>")
    c._refresh_step3_lock = lambda: None
    return c


def test_the_cancelled_summary_reads_done_of_total_and_logs_it():
    c = _ctl()
    c._on_serial_pred_cancelled(1, 3)
    c._on_serial_pred_complete()
    text, kind = c.status[-1]
    assert text.startswith("Segmentation cancelled: 1/3 patches done")
    assert c.log[0] == text
    assert kind == "warn"
    assert c._pipeline_cancelled is None
    assert c._predicting_patches == set()
    assert "<refresh>" in c.log
    assert c.ui.pipeline_progress.value is None      # the bar is not jumped


def test_the_cancelled_summary_carries_the_failed_suffix():
    c = _ctl()
    c._pipeline_errors = 1
    c._on_serial_pred_cancelled(0, 3)
    c._on_serial_pred_complete()
    text, _kind = c.status[-1]
    assert text.startswith(
        _pr_text.S3_PRED_CANCELLED.format(done=0, total=3))
    assert _theme.PIPE_FAILED_SUFFIX.format(k=1) in text


def test_a_normal_finish_is_unchanged_and_jumps_the_bar():
    c = _ctl()
    c._on_serial_pred_complete()
    text, _kind = c.status[-1]
    assert text.startswith("Segmentation complete")
    assert c.ui.pipeline_progress.value == 100


def test_the_cancel_click_says_slice_and_disables_the_button():
    c = _ctl()
    w = c._serial_prediction_worker
    c._cancel_prediction()
    assert w.paused == 1
    assert c.status[-1][0] == _theme.PIPE_CANCELLING.format(unit="slice")
    assert c.ui.pipeline_cancel_btn.enabled is False
    assert c.log[-1] == "Cancel requested: stopping after the current slice"


def test_view_progress_keeps_the_cancelling_line_but_moves_the_bar():
    cancelling = _theme.PIPE_CANCELLING.format(unit="slice")
    c = _ctl(requested=True)
    c._pr_job_line = cancelling
    c._on_serial_pred_view_progress("p2", "xy", 4, 10)
    assert c.status == []                  # the line is already showing
    assert c.ui.pipeline_progress.value == 4
    # another writer replaced the line: it is put back, never a progress line
    c3 = _ctl(requested=True)
    c3._pr_job_line = "something else"
    c3._on_serial_pred_view_progress("p2", "xy", 5, 10)
    assert c3.status == [(cancelling, "neutral")]
    assert c3.ui.pipeline_progress.value == 5
    c2 = _ctl(requested=False)
    c2._on_serial_pred_view_progress("p2", "xy", 4, 10)
    assert len(c2.status) == 1


def test_a_queued_patch_started_cannot_overwrite_the_cancelling_line():
    cancelling = _theme.PIPE_CANCELLING.format(unit="slice")
    c = _ctl(requested=True)
    c._pr_job_line = cancelling
    c._on_serial_pred_started("p3", 2, 3)
    assert c.status == []
    assert "p3" in c._predicting_patches
    c3 = _ctl(requested=True)
    c3._pr_job_line = "something else"
    c3._on_serial_pred_started("p3", 2, 3)
    assert c3.status == [(cancelling, "neutral")]
    c2 = _ctl(requested=False)
    c2._on_serial_pred_started("p3", 2, 3)
    assert len(c2.status) == 1


@pytest.mark.parametrize("requested,expected", [(False, True), (True, False)])
def test_the_cancel_button_stays_off_once_cancel_was_asked(requested, expected):
    c = _ctl(requested)
    c._update_button_states()
    assert c.ui.pipeline_cancel_btn.enabled is expected


def test_the_complete_slot_greys_cancel_and_a_late_click_keeps_the_summary():
    c = _ctl()
    c.ui.pipeline_cancel_btn.enabled = True
    c._on_serial_pred_complete()
    assert c.ui.pipeline_cancel_btn.enabled is False
    summary = c.status[-1]
    # a click that still reaches the worker (button already grey in the UI,
    # but the slot may be queued) only matters if it rewrites the line; the
    # summary must be the last write before the thread finishes
    assert summary[0].startswith("Segmentation complete")


def test_a_run_switch_clears_the_cancelled_flag_without_a_summary():
    c = _ctl()
    c._on_serial_pred_cancelled(1, 3)
    c._run_epoch = 1                    # the run changed mid-prediction
    c._on_serial_pred_complete()
    assert c._pipeline_cancelled is None
    assert c.status == []

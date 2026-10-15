"""Batch Refine progress: stages, ETA, marquee, summary.

- ``run_batch_recipe(stage_cb=)`` reports preparing -> refining -> writing,
  and "refining" arrives BEFORE the first compute (the ETA clock starts
  there, on the main thread);
- ``BatchRefineWorker.stage`` carries it; every signal in order, driven by a
  direct ``worker.run()`` (the ``test_refine_recipe_worker.py`` pattern);
- ``eta_text`` is pure: its two thresholds and its rounding;
- the dialog's marquee (``set_busy``) and the staged summary lines;
- the annotator's handlers put the sentences on screen.

Plain ``QApplication`` (pytest-qt is not installed).
"""
import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from packages.epicycle_annotator import batch_refine as br  # noqa: E402
from packages.epicycle_annotator.batch_refine import (  # noqa: E402
    BATCH_CANCELLING,
    BATCH_COUNT,
    BATCH_ETA_MINUTES,
    BATCH_ETA_SECONDS,
    BATCH_FAILED,
    BATCH_NONE_CHANGED,
    BATCH_STAGE_PREPARING,
    BATCH_STAGE_REFINING,
    BATCH_STAGE_WRITING,
    BATCH_SUMMARY,
    BATCH_SUMMARY_ABSENT,
    BATCH_SUMMARY_EMPTIED,
    BATCH_WARNING_BANNER,
    ETA_MIN_DONE,
    ETA_MIN_SECONDS,
    ETA_MINUTES_FROM,
    RefineStep,
    eta_text,
    run_batch_recipe,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def widgets(qapp):
    made = []
    yield made
    for w in made:
        try:
            if hasattr(w, "mark_clean_close"):
                w.mark_clean_close()
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    qapp.processEvents()


def _vol(n=5):
    v = np.zeros((12, 12, 12 * n), dtype=np.int32)
    for i in range(n):
        v[3:9, 3:9, 12 * i + 3:12 * i + 9] = i + 1
    return v


def _recipe():
    return [RefineStep("dilate_erode", {"radius": 1, "axis": "3D"})]


# ------------------------------------------------------------------ ETA


class TestEtaText:
    def test_the_constants(self):
        assert ETA_MIN_DONE == 2
        assert ETA_MIN_SECONDS == 2
        assert ETA_MINUTES_FROM == 90

    def test_nothing_before_two_objects(self):
        assert eta_text(10.0, 1, 100) == ""

    def test_nothing_before_two_seconds(self):
        assert eta_text(1.9, 5, 100) == ""

    def test_nothing_when_done(self):
        assert eta_text(10.0, 10, 10) == ""

    def test_seconds_round_up_to_ten(self):
        # 4 s for 2 objects -> 2 s each -> 8 left -> 16 s -> "20 s"
        assert eta_text(4.0, 2, 10) == BATCH_ETA_SECONDS.format(s=20)
        # exactly 10 s left stays 10
        assert eta_text(2.0, 2, 12) == BATCH_ETA_SECONDS.format(s=10)
        # 1 s left still says 10, never 0
        assert eta_text(2.0, 2, 3) == BATCH_ETA_SECONDS.format(s=10)

    def test_minutes_from_ninety_seconds(self):
        # 89 s left -> seconds (rounded up to 90)
        assert eta_text(2.0, 2, 91) == BATCH_ETA_SECONDS.format(s=90)
        # 90 s left -> minutes, rounded up: 2 min
        assert eta_text(2.0, 2, 92) == BATCH_ETA_MINUTES.format(m=2)
        # 120 s -> 2 min; 122 s -> 3 min
        assert eta_text(2.0, 2, 122) == BATCH_ETA_MINUTES.format(m=2)
        assert eta_text(2.0, 2, 124) == BATCH_ETA_MINUTES.format(m=3)


# ------------------------------------------------------- the pure stages


class TestStageOrder:
    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_preparing_refining_progress_writing(self, n_workers):
        events = []
        run_batch_recipe(
            _vol(5), [1, 2, 3, 4, 5], _recipe(), n_workers=n_workers,
            progress_cb=lambda d, t: events.append(("progress", d, t)),
            stage_cb=lambda s, d, t: events.append(("stage", s, d, t)),
        )
        assert events[0] == ("stage", "preparing", 0, 0)
        assert events[1] == ("stage", "refining", 0, 5)
        progress = [e for e in events if e[0] == "progress"]
        assert progress and progress[-1] == ("progress", 5, 5)
        assert events[-1] == ("stage", "writing", 0, 5)
        # "refining" comes before ANY compute tick
        first_progress = events.index(progress[0])
        assert events.index(("stage", "refining", 0, 5)) < first_progress

    def test_refining_before_the_first_compute(self, monkeypatch):
        events = []
        real = br._compute_one

        def spy(*a, **k):
            events.append("compute")
            return real(*a, **k)

        monkeypatch.setattr(br, "_compute_one", spy)
        run_batch_recipe(
            _vol(2), [1, 2], _recipe(), n_workers=1,
            stage_cb=lambda s, d, t: events.append(s),
        )
        assert events.index("refining") < events.index("compute")

    def test_a_cancel_before_start_only_prepares(self):
        import threading

        ev = threading.Event()
        ev.set()
        stages = []
        res = run_batch_recipe(
            _vol(2), [1, 2], _recipe(), cancel_event=ev,
            stage_cb=lambda s, d, t: stages.append(s),
        )
        assert res.cancelled
        assert stages == ["preparing"]

    def test_no_stage_cb_is_byte_identical(self):
        a = run_batch_recipe(_vol(3), [1, 2, 3], _recipe(), n_workers=1)
        b = run_batch_recipe(
            _vol(3), [1, 2, 3], _recipe(), n_workers=1,
            stage_cb=lambda *a: None,
        )
        assert np.array_equal(a.working_volume, b.working_volume)
        assert a.touched_ids == b.touched_ids
        assert a.n_changed_total == b.n_changed_total


# ------------------------------------------------------------ the worker


class TestWorkerSignals:
    def test_signal_order_on_a_direct_run(self, qapp):
        from napari_epicycle.annotator.batch_refine_worker import (
            BatchRefineWorker,
        )

        w = BatchRefineWorker(_vol(3), [1, 2, 3], _recipe(), n_workers=1)
        events = []
        w.stage.connect(lambda s, d, t: events.append(("stage", s, d, t)))
        w.progress.connect(lambda d, t: events.append(("progress", d, t)))
        w.finished.connect(lambda r: events.append(("finished",)))
        w.error.connect(lambda m: events.append(("error", m)))
        w.run()
        assert events[0] == ("stage", "preparing", 0, 0)
        assert events[1] == ("stage", "refining", 0, 3)
        assert ("progress", 3, 3) in events
        assert events[-2] == ("stage", "writing", 0, 3)
        assert events[-1] == ("finished",)
        assert not any(e[0] == "error" for e in events)


# ------------------------------------------------------------- the dialog


def _dialog(widgets):
    from napari_epicycle.annotator.batch_refine_dialog import BatchRefineDialog

    dlg = BatchRefineDialog(
        count_provider=lambda spec: 7,
        all_count=7,
        selected_count=0,
        available_tiers=["good"],
        on_preview_one=lambda: None,
        on_run=lambda: None,
        on_cancel=lambda: None,
        on_accept=lambda: None,
        on_reject=lambda: None,
        on_close=lambda: None,
    )
    widgets.append(dlg)
    return dlg


class TestDialog:
    def test_marquee_then_determinate(self, widgets):
        dlg = _dialog(widgets)
        dlg.set_state("running")
        dlg.set_busy()
        assert dlg._progress.minimum() == 0 and dlg._progress.maximum() == 0
        dlg.set_progress(2, 9)
        assert dlg._progress.maximum() == 9 and dlg._progress.value() == 2
        dlg.set_busy()
        assert dlg._progress.maximum() == 0

    def test_idle_leaves_the_marquee(self, widgets):
        dlg = _dialog(widgets)
        dlg.set_state("running")
        dlg.set_busy()
        dlg.set_state("idle")
        assert dlg._progress.maximum() > 0

    def test_count_line_says_objects(self, widgets):
        dlg = _dialog(widgets)
        assert dlg._count_lbl.text() == BATCH_COUNT.format(k="7 objects")
        assert "objects" in dlg._count_lbl.text()

    def test_warning_banner(self, widgets):
        from qtpy.QtWidgets import QLabel

        dlg = _dialog(widgets)
        texts = [lbl.text() for lbl in dlg.findChildren(QLabel)]
        assert BATCH_WARNING_BANNER in texts
        assert "objects" in BATCH_WARNING_BANNER

    def test_summary_alone(self, widgets):
        dlg = _dialog(widgets)
        dlg.set_preview_summary(142, 11800)
        assert dlg._summary_lbl.text() == BATCH_SUMMARY.format(
            n="142 objects", v=11800)
        assert "objects" in dlg._summary_lbl.text()
        assert "11,800" in dlg._summary_lbl.text()

    def test_summary_lists_emptied_and_absent(self, widgets):
        dlg = _dialog(widgets)
        dlg.set_preview_summary(3, 40, emptied=[14, 57], absent=[91])
        lines = dlg._summary_lbl.text().split("\n")
        assert lines[0] == BATCH_SUMMARY.format(n="3 objects", v=40)
        assert lines[1] == BATCH_SUMMARY_EMPTIED.format(n="2 objects", ids="14, 57")
        assert lines[2] == BATCH_SUMMARY_ABSENT.format(n="1 object", ids="91")

    def test_at_most_eight_ids_then_an_ellipsis(self, widgets):
        dlg = _dialog(widgets)
        dlg.set_preview_summary(1, 1, emptied=list(range(1, 12)))
        line = dlg._summary_lbl.text().split("\n")[1]
        assert line == BATCH_SUMMARY_EMPTIED.format(
            n="11 objects", ids="1, 2, 3, 4, 5, 6, 7, 8, …")

    def test_the_summary_keywords_are_keyword_only(self):
        import inspect

        from napari_epicycle.annotator.batch_refine_dialog import (
            BatchRefineDialog,
        )

        sig = inspect.signature(BatchRefineDialog.set_preview_summary)
        for name in ("emptied", "absent"):
            assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------- the annotator


class _Dlg:
    def __init__(self):
        self.status = []
        self.progress = []
        self.busy = 0
        self.states = []
        self.summary = None

    def set_status(self, t):
        self.status.append(t)

    def set_progress(self, d, t):
        self.progress.append((d, t))

    def set_busy(self):
        self.busy += 1

    def set_state(self, s):
        self.states.append(s)

    def set_preview_summary(self, *a, **k):
        self.summary = (a, k)


class _Fake(types.SimpleNamespace):
    def __getattr__(self, name):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        fn = Curation3DAnnotator.__dict__.get(name)
        if callable(fn):
            return fn.__get__(self)
        raise AttributeError(name)


def _ann(monkeypatch, clock):
    from napari_epicycle.annotator import curation_3d_widget as cw

    monkeypatch.setattr(cw._time, "monotonic", lambda: clock["t"])
    return _Fake(
        _batch_refine_dialog=_Dlg(),
        _batch_n_targets=10,
        _batch_eta_t0=None,
        _batch_worker=None,
        _batch_thread=None,
        _batch_running=True,
        _batch_snapshot=None,
        _active_labels_name="lbl",
        _batch_labels_name="lbl",
        _fire_batch_refine_lock=lambda active: None,
        _restore_ready_to_draw_state=lambda: None,
        _sync_refine_state=lambda: None,
        _batch_thread_cleanup=lambda: None,
    )


class TestAnnotatorHandlers:
    def test_preparing_is_a_marquee(self, monkeypatch):
        ann = _ann(monkeypatch, {"t": 0.0})
        ann._on_batch_worker_stage("preparing", 0, 0)
        dlg = ann._batch_refine_dialog
        assert dlg.busy == 1
        assert dlg.status[-1] == BATCH_STAGE_PREPARING.format(n="10 objects")

    def test_refining_starts_the_clock_and_eta_follows(self, monkeypatch):
        clock = {"t": 100.0}
        ann = _ann(monkeypatch, clock)
        dlg = ann._batch_refine_dialog
        ann._on_batch_worker_stage("refining", 0, 10)
        assert dlg.progress[-1] == (0, 10)
        assert dlg.status[-1] == BATCH_STAGE_REFINING.format(done=0, total=10)
        clock["t"] = 101.0
        ann._on_batch_worker_progress(1, 10)
        assert dlg.status[-1] == BATCH_STAGE_REFINING.format(done=1, total=10)
        clock["t"] = 104.0  # 4 s for 2 -> 16 s left -> "20 s"
        ann._on_batch_worker_progress(2, 10)
        assert dlg.status[-1] == (
            BATCH_STAGE_REFINING.format(done=2, total=10)
            + " · " + BATCH_ETA_SECONDS.format(s=20)
        )

    def test_writing_is_a_marquee(self, monkeypatch):
        ann = _ann(monkeypatch, {"t": 0.0})
        ann._on_batch_worker_stage("writing", 0, 10)
        dlg = ann._batch_refine_dialog
        assert dlg.busy == 1
        assert dlg.status[-1] == BATCH_STAGE_WRITING.format(n="10 objects")

    def test_error_is_the_fixed_sentence(self, monkeypatch, capsys):
        ann = _ann(monkeypatch, {"t": 0.0})
        ann._on_batch_worker_error("KeyError: 'x'\nTraceback ...")
        dlg = ann._batch_refine_dialog
        assert dlg.status[-1] == BATCH_FAILED
        assert "KeyError" in capsys.readouterr().out  # raw text to the log

    def test_cancel_sentence(self, monkeypatch):
        ann = _ann(monkeypatch, {"t": 0.0})
        ann._on_batch_cancel()
        assert ann._batch_refine_dialog.status[-1] == BATCH_CANCELLING

    def test_nothing_changed_names_the_emptied(self, monkeypatch):
        ann = _ann(monkeypatch, {"t": 0.0})
        res = br.BatchResult(
            working_volume=np.zeros((2, 2, 2), np.int32), touched_ids=[],
            union_roi=None, n_processed=2, n_changed_total=0,
            cancelled=False, emptied_ids=[4, 9], absent_ids=[])
        ann._on_batch_worker_finished(res)
        text = ann._batch_refine_dialog.status[-1]
        assert text.startswith(BATCH_NONE_CHANGED)
        assert BATCH_SUMMARY_EMPTIED.format(n="2 objects", ids="4, 9") in text

    def test_nothing_changed_plain(self, monkeypatch):
        ann = _ann(monkeypatch, {"t": 0.0})
        res = br.BatchResult(
            working_volume=np.zeros((2, 2, 2), np.int32), touched_ids=[],
            union_roi=None, n_processed=2, n_changed_total=0,
            cancelled=False)
        ann._on_batch_worker_finished(res)
        assert ann._batch_refine_dialog.status[-1] == BATCH_NONE_CHANGED

    def test_staged_summary_carries_emptied_and_absent(self, monkeypatch):
        ann = _ann(monkeypatch, {"t": 0.0})
        vol = np.zeros((4, 4, 4), np.int32)
        ann.sync_manager = types.SimpleNamespace(shared_labels={"lbl": vol})
        ann._batch_snapshot = vol.copy()
        ann._sync_all_views_after_operation = lambda roi=None: None
        work = vol.copy()
        work[0, 0, 0] = 1
        res = br.BatchResult(
            working_volume=work, touched_ids=[1],
            union_roi=(slice(0, 1),) * 3, n_processed=3, n_changed_total=1,
            cancelled=False, emptied_ids=[2], absent_ids=[3])
        ann._on_batch_worker_finished(res)
        args, kw = ann._batch_refine_dialog.summary
        assert args == (1, 1)
        assert kw == {"emptied": [2], "absent": [3]}
        assert ann._batch_refine_dialog.states[-1] == "staged"

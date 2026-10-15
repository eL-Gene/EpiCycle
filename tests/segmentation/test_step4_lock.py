"""F4 / I-8 / I-13 (brick B0 part): the flags, the ``try/finally`` rule and
the thread-finished backstop, on REAL ``QThread``s.

B0 adds the flags and their backstops; the greying
of the lock set, the owner-keyed step-nav lock and the lock lines are B2
(their tests join this file then).  A plain ``QApplication`` is pumped
until a deadline -- no pytest-qt.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from napari_epicycle.segmentation import _step4_lock as L
from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.segmentation import controller as C

from ._step4_fakes import controller


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


def _pump(app, until, deadline=20.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(until())


def _join(ctrl):
    for th in getattr(ctrl, "_retired_threads", []):
        th.wait(3000)


def test_every_flag_starts_false(tmp_path):
    ctrl = controller(tmp_path)
    for flag in L.ALL_FLAGS:
        assert getattr(ctrl, flag) is False
    assert L.is_locked(ctrl) is False


def test_any_block_flag_locks_and_the_map_flag_does_not(tmp_path):
    ctrl = controller(tmp_path)
    for flag in L.STEP4_BLOCK_FLAGS:
        setattr(ctrl, flag, True)
        assert L.is_locked(ctrl) and L.lock_reason(ctrl) == flag
        setattr(ctrl, flag, False)
    ctrl._eda_running = True
    assert not L.is_locked(ctrl)


def test_import_error_in_predict_releases_the_lock(qapp, tmp_path,
                                                   monkeypatch):
    """Today the import sat outside any try: an ImportError emitted
    neither ``error`` nor ``all_completed`` and the thread never quit."""
    import packages.epicycle_reconstruction.logic as logic

    monkeypatch.delattr(logic, "segment_views")
    ctrl = controller(tmp_path)
    ctrl._seg_start_time = time.monotonic()
    # B1a: after a run the buttons follow their gates, so give
    # Predict what it needs: a model, a membrane channel, a ticked stack.
    from napari_epicycle.segmentation import _step4_models as M
    ctrl._active_model = M.model_snapshot("cyto3", "cyto3",
                                          kind="pretrained")
    ctrl.ui.model_source_combo.setCurrentIndex(2)
    ctrl.state_manager.register_stacks(["s.tif"])
    # re-audit N3: a stack with no resolved axes cannot run, so give the
    # record the axes a real listing would have read
    ctrl.state_manager.set_stack_status(
        "s.tif", "unprocessed", axes="CZYX", stack_shape=[2, 6, 24, 24],
        n_channels=2)
    ctrl.state_manager.set_ticked(["s.tif"])
    ctrl.ui.predict_btn.setEnabled(False)
    ctrl._launch_prediction_worker([{
        "stack_name": "s.tif", "source_stack_path": "x.tif",
        "lateral_model_path": "L", "axial_model_path": "A"}])
    assert ctrl._predict_running is True
    assert _pump(qapp, lambda: not ctrl._predict_running)
    assert ctrl._seg_thread is None
    assert ctrl.ui.predict_btn.isEnabled()
    assert not ctrl.ui.cancel_btn.isEnabled()
    assert any("Prediction failed for s.tif" in l
               for l in ctrl.ui.log_output.lines)
    assert ctrl.state_manager.get_stack_status("s.tif") == "failed"
    assert ctrl.state_manager.get_record("s.tif")["failed_stage"] == "predict"
    _join(ctrl)


def test_a_reconstruct_worker_that_raises_releases_the_lock(qapp, tmp_path):
    class Boom:
        def run_real(self, **kw):
            raise RuntimeError("u-Segment3D fell over")

    ctrl = controller(tmp_path)
    ctrl._seg_start_time = time.monotonic()
    ctrl.state_manager.set_stack_status("s.tif", "finalized", labels_path="L")
    ctrl._launch_reconstruction_worker([{
        "stack_name": "s.tif", "working_dir": str(tmp_path),
        "source_stack_path": "x.tif", "lateral_model_path": "L",
        "axial_model_path": "A", "cached_views": {"xy": None}}],
        pipeline=Boom())
    assert _pump(qapp, lambda: not ctrl._reconstruct_running)
    # A stack with a 3D result keeps it: the failure is recorded only.
    rec = ctrl.state_manager.get_record("s.tif")
    assert rec["status"] == "finalized"
    assert rec["failed_stage"] == "reconstruct"
    _join(ctrl)


def test_a_failure_with_nothing_later_reads_failed(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.state_manager.set_stack_status("s.tif", "predicted")
    ctrl._on_recon_error("s.tif", "boom")
    assert ctrl.state_manager.get_stack_status("s.tif") == "failed"


def test_browse_and_adopt_refuse_while_a_job_runs(tmp_path):
    ctrl = controller(tmp_path)
    ctrl._predict_running = True
    other = tmp_path / "other"
    other.mkdir()
    ctrl.working_dir = ""
    ctrl.adopt_workspace(str(other))
    assert ctrl.working_dir == ""
    ctrl._browse_workdir()            # returns before any file dialog
    assert T.LOCK_WORKSPACE_REFUSED in ctrl.ui.log_output.lines


def test_the_segmentation_worker_emits_all_completed_in_finally(tmp_path):
    """No QThread needed: ``run`` itself must end in ``all_completed``."""
    class Boom:
        def run_real(self, **kw):
            raise KeyboardInterrupt      # a BaseException, not an Exception

    worker = C._StackSegmentationWorker(
        [{"stack_name": "s.tif", "working_dir": str(tmp_path),
          "source_stack_path": "x", "lateral_model_path": "L",
          "axial_model_path": "A", "cached_views": {}}], Boom())
    done = []
    worker.all_completed.connect(lambda: done.append(1))
    worker.run()
    assert done == [1]


def _blocked_worker(event):
    """A QObject worker that blocks on *event*."""
    from qtpy.QtCore import QObject, Signal

    class W(QObject):
        finished = Signal(object, str, int)
        error = Signal(str, str, int)
        csv_saved = Signal(str, str)
        progress = Signal(int)
        log = Signal(str)

        def run(self):
            event.wait(20)
            self.error.emit("done", "none.tif", -1)

    return W()


def test_step4_never_releases_a_lock_it_did_not_take(qapp, tmp_path):
    """Audit L1: a second launch while the first thread runs is refused,
    and nothing the refused launch leaves behind ends the live job's lock;
    gc.collect() while the first thread runs does not abort."""
    import gc
    import threading

    ctrl = controller(tmp_path)
    first_go, second_go = threading.Event(), threading.Event()
    first = _blocked_worker(first_go)
    assert ctrl._launch_measure_worker(first, 1) is not None
    live = ctrl._qc_thread
    second = _blocked_worker(second_go)
    assert ctrl._launch_measure_worker(second, 2) is None     # refused
    assert ctrl._qc_thread is live
    assert ctrl._measure_token == 1
    gc.collect()
    qapp.processEvents()
    assert ctrl._measure_running is True
    first_go.set()
    assert _pump(qapp, lambda: not ctrl._measure_running)
    second_go.set()
    _join(ctrl)


def test_a_finish_from_a_replaced_thread_keeps_the_live_flag(tmp_path):
    ctrl = controller(tmp_path)
    live, stale = object(), object()
    ctrl._qc_thread = live
    ctrl._measure_running = True
    ctrl._on_step4_thread_finished("_measure_running", "_qc", stale, None)
    assert ctrl._measure_running is True and ctrl._qc_thread is live


def test_every_action_refuses_while_locked(tmp_path):
    ctrl = controller(tmp_path)
    ctrl._smooth_running = True
    ctrl._current_stack_name = "s.tif"
    for name in ("_run_prediction", "_run_reconstruction",
                 "_extract_features", "_smooth_all", "_load_reconstruction",
                 "_delete_bad_labels", "_export_all_stacks",
                 "_export_meshes"):
        assert getattr(ctrl, name)() is None, name
    assert ctrl._save_tier_assignments() is False
    assert ctrl._finalize() is False
    refused = [l for l in ctrl.ui.log_output.lines if "refused" in l]
    assert len(refused) == 10

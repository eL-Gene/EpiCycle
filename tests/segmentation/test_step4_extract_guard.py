"""F1 / I-1: a Measure finish for another stack or another job changes nothing.

F1 (the thread-finished backstop).  The guard
tests are Qt-free; the backstop test runs a REAL ``QThread`` under a plain
``QApplication`` (a fake thread that emits ``finished`` by
itself hides a thread that never quits).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from ._step4_fakes import controller


def _vol():
    v = np.zeros((4, 6, 6), np.int32)
    v[1:3, 1:3, 1:3] = 1
    v[1:3, 3:5, 3:5] = 2
    return v


def _df(labels, border=()):
    return pd.DataFrame({"label": list(labels),
                         "is_border_cell": [1 if l in border else 0
                                            for l in labels]})


@pytest.fixture
def ctrl(tmp_path):
    c = controller(tmp_path)
    c._current_stack_name = "B.tif"
    c._labels_volume = _vol()
    c._qc_features_df = _df([1, 2])
    c.state_manager.set_stack_status("A.tif", "reconstructed",
                                     labels_path=str(tmp_path / "A" / "l.tif"))
    c.state_manager.set_stack_status("B.tif", "qc_extracted",
                                     labels_path=str(tmp_path / "B" / "l.tif"))
    c._measure_token = 7
    return c


def _snapshot(c):
    return (c._labels_volume.copy(), c._qc_features_df.copy(),
            c.state_manager.get_record("A.tif"),
            c.state_manager.get_record("B.tif"))


def test_a_finish_for_another_stack_touches_nothing(ctrl, tmp_path):
    before = _snapshot(ctrl)
    ctrl._pending_border_removal = True
    ctrl._on_extract_worker_finished(_df([1, 2, 3], border=(1,)), "A.tif", 7)
    after = _snapshot(ctrl)
    assert np.array_equal(before[0], after[0])
    assert before[1].equals(after[1])
    assert before[2] == after[2] and before[3] == after[3]
    assert "border_cells_path" not in ctrl.state_manager.get_record("A.tif")
    assert not (tmp_path / "A").exists()           # no border_cells.tif
    assert any("discarded" in l for l in ctrl.ui.log_output.lines)


def test_a_finish_from_an_older_job_touches_nothing(ctrl):
    before = _snapshot(ctrl)
    ctrl._on_extract_worker_finished(_df([1]), "B.tif", 6)
    after = _snapshot(ctrl)
    assert np.array_equal(before[0], after[0])
    assert before[1].equals(after[1])


def test_the_eda_cache_is_not_invalidated_by_a_stale_finish(ctrl, tmp_path):
    eda = tmp_path / "artifacts" / "step4" / "eda"
    eda.mkdir(parents=True)
    (eda / "B.tif_eda.json").write_text("{}", encoding="utf-8")
    ctrl._on_extract_worker_finished(_df([1]), "A.tif", 7)
    assert (eda / "B.tif_eda.json").exists()


def test_a_stale_error_does_not_write_the_live_status(ctrl):
    ctrl._on_extract_worker_error("boom", "A.tif", 7)
    assert ctrl.ui.qc_status_line.sets == []


# ------------------------------------------------ the backstop (real Qt)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


def _pump(app, until, deadline=15.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(until())


def test_stale_finish_still_clears_the_flag(qapp, ctrl):
    from qtpy.QtCore import QObject, Signal

    class StaleWorker(QObject):
        finished = Signal(object, str, int)
        error = Signal(str, str, int)
        csv_saved = Signal(str, str)
        progress = Signal(int)
        log = Signal(str)

        def run(self):
            self.finished.emit(_df([9]), "A.tif", 1)     # stale on both

    worker = StaleWorker()
    before = _snapshot(ctrl)
    ctrl.ui.extract_features_btn.setEnabled(False)
    ctrl._launch_measure_worker(worker, 7)
    assert ctrl._measure_running is True
    assert _pump(qapp, lambda: not ctrl._measure_running)
    assert ctrl._qc_thread is None
    assert ctrl.ui.extract_features_btn.isEnabled()      # backstop
    after = _snapshot(ctrl)
    assert np.array_equal(before[0], after[0])
    assert before[1].equals(after[1])
    for th in getattr(ctrl, "_retired_threads", []):
        th.wait(2000)


def test_the_real_measure_worker_always_ends_its_thread(qapp, ctrl, tmp_path):
    """A real ``_StackQCFeatureWorker`` whose labels cannot be read emits
    ``error`` with its stack and token, and the thread quits."""
    from napari_epicycle.segmentation.controller import _StackQCFeatureWorker

    worker = _StackQCFeatureWorker(
        labels_path=str(tmp_path / "missing.tif"), stack_name="B.tif",
        csv_path=str(tmp_path / "x.csv"), token=7)
    ctrl._measure_launch = {"token": 7, "epoch": ctrl._labels_epoch}
    ctrl._launch_measure_worker(worker, 7)
    assert _pump(qapp, lambda: not ctrl._measure_running)
    assert any(s.startswith("Measuring B.tif failed")
               for s in ctrl.ui.qc_status_line.sets)
    for th in getattr(ctrl, "_retired_threads", []):
        th.wait(2000)


# ------------------------------------------- audit D2 / D3 / L2 (pending)


def _launch_state(ctrl, tmp_path, token, stack="B.tif"):
    """What _run_feature_extraction records at launch, for a fake job."""
    from napari_epicycle.segmentation.controller import _measure_pending
    qc = tmp_path / "artifacts" / "step4" / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    csv = str(qc / f"{stack}_features.csv")
    adj = str(qc / f"{stack}_adjacency.csv")
    ctrl._measure_pendings = {token: [(_measure_pending(csv, token), csv),
                                      (_measure_pending(adj, token), adj)]}
    ctrl._measure_launch = {"token": token, "epoch": ctrl._labels_epoch,
                            "stack": stack, "spacing": None,
                            "z_orientation": "basal_down"}
    return csv, _measure_pending(csv, token)


def test_a_stale_finish_leaves_the_saved_marks_on_disk(ctrl, tmp_path):
    csv, pending = _launch_state(ctrl, tmp_path, 7)
    saved = "label,qc_tier\n1,good\n2,bad\n"
    open(csv, "w", encoding="utf-8").write(saved)
    open(pending, "w", encoding="utf-8").write("label\n1\n2\n")
    ctrl._current_stack_name = "other.tif"            # the user moved on
    ctrl._on_extract_worker_finished(_df([1, 2]), "B.tif", 7)
    assert open(csv, encoding="utf-8").read() == saved
    assert not os.path.exists(pending)


def test_a_current_finish_moves_the_pending_csv_in(ctrl, tmp_path):
    ctrl._open_analysis_dock = lambda: None
    csv, pending = _launch_state(ctrl, tmp_path, 7)
    open(pending, "w", encoding="utf-8").write("label\n1\n2\n")
    ctrl._on_extract_worker_finished(_df([1, 2]), "B.tif", 7)
    assert os.path.exists(csv) and not os.path.exists(pending)
    assert ctrl.state_manager.get_record("B.tif")["qc_features_csv"] == csv


def test_csv_saved_writes_no_record(ctrl, tmp_path):
    """Audit D3: a reset stack must not get a CSV pointer back."""
    ctrl.state_manager.set_stack_status("A.tif", "reconstructed")
    ctrl._on_csv_saved("A.tif", str(tmp_path / "x.pending"))
    assert "qc_features_csv" not in ctrl.state_manager.get_record("A.tif")


def test_an_epoch_bump_makes_the_finish_stale(ctrl, tmp_path):
    before = _snapshot(ctrl)
    _launch_state(ctrl, tmp_path, 7)
    ctrl._labels_epoch += 1                       # e.g. a Smooth landed
    ctrl._on_extract_worker_finished(_df([1]), "B.tif", 7)
    assert before[1].equals(ctrl._qc_features_df)


def test_the_worker_writes_the_launch_marks_into_its_pending_csv(tmp_path):
    import pandas as pd
    from napari_epicycle.segmentation.controller import (
        _StackQCFeatureWorker, _apply_tier_map, _measure_pending,
    )
    old = tmp_path / "old.csv"
    old.write_text("label,qc_tier\n1,bad\n2,\n", encoding="utf-8")
    w = _StackQCFeatureWorker(labels_path="", stack_name="s", csv_path="",
                              tier_csv=str(old))
    tiers = w._launch_tiers()
    assert tiers == {1: "bad"}
    df = pd.DataFrame({"label": [1, 2, 3]})
    assert _apply_tier_map(df, tiers) == 1
    assert list(df["qc_tier"]) == ["bad", "", ""]
    assert _measure_pending("a.csv", 3) == "a.csv.3.pending"


def test_the_worker_gets_a_copy_of_the_volume(ctrl, tmp_path, monkeypatch):
    import napari_epicycle.segmentation.controller as C
    got = {}

    class Spy:
        def __init__(self, **kw):
            got.update(kw)

    monkeypatch.setattr(C, "_StackQCFeatureWorker", Spy)
    ctrl._launch_measure_worker = lambda worker, token: None
    ctrl.state_manager.update_record("B.tif", labels_path="L")
    ctrl._run_feature_extraction(skip_probability=True)
    assert got["labels_array"] is not ctrl._labels_volume
    assert np.array_equal(got["labels_array"], ctrl._labels_volume)

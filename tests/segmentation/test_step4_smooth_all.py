"""F2 / I-1 / I-2 / I-3: Smooth All through Batch Refine's executor.

F2, D-b (the output changes voxel
for voxel, accepted).  The worker's ``run`` is driven directly for the
maths; one case runs it on a REAL ``QThread`` (I-13).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.segmentation.controller import _SmoothBatchWorker

from ._step4_fakes import controller


def _volume():
    """An interior blob, a neighbour, and one thin single-voxel label."""
    v = np.zeros((40, 48, 48), np.uint32)
    zz, yy, xx = np.mgrid[:40, :48, :48]
    v[((zz - 20) ** 2 + (yy - 18) ** 2 + (xx - 18) ** 2) <= 36] = 1
    v[((zz - 20) ** 2 + (yy - 30) ** 2 + (xx - 30) ** 2) <= 25] = 2
    v[20, 40, 5] = 3                          # thin: smoothing empties it
    return v


def _run(worker):
    got = {}
    worker.finished.connect(lambda r, s, t: got.update(result=r, stack=s,
                                                       token=t))
    worker.error.connect(lambda m, s, t: got.update(error=m))
    worker.run()
    return got


def test_the_volume_passed_in_is_never_mutated():
    vol = _volume()
    before = vol.copy()
    got = _run(_SmoothBatchWorker(vol, 3.0, 1.5, True, stack_name="s",
                                  token=4))
    assert "result" in got, got.get("error")
    assert np.array_equal(vol, before)                          # I-2
    assert got["result"].working_volume is not vol
    assert got["stack"] == "s" and got["token"] == 4


def test_a_thin_label_survives_and_no_id_is_lost():
    vol = _volume()
    got = _run(_SmoothBatchWorker(vol, 3.0, 1.5, True))
    res = got["result"]
    assert 3 in res.emptied_ids
    assert set(np.unique(res.working_volume)) == set(np.unique(vol))  # I-3
    assert res.working_volume[20, 40, 5] == 3


def test_an_interior_label_equals_batch_refine_voxel_for_voxel():
    from packages.epicycle_annotator.batch_refine import (
        RefineStep, run_batch_recipe,
    )
    vol = _volume()
    got = _run(_SmoothBatchWorker(vol, 2.0, 1.0, True))
    ref = run_batch_recipe(
        vol, [1, 2, 3],
        [RefineStep("sdf", {"sigma_z": 2.0, "sigma_xy": 1.0,
                            "preserve_topology": True})],
        empty="skip")
    assert np.array_equal(got["result"].working_volume == 1,
                          ref.working_volume == 1)


# ---------------------------------------------------- the finish handler


class _Result:
    def __init__(self, vol, emptied=()):
        self.working_volume = vol
        self.touched_ids = [1]
        self.n_changed_total = 5
        self.emptied_ids = list(emptied)
        self.cancelled = False


@pytest.fixture
def ctrl(tmp_path):
    c = controller(tmp_path)
    c._current_stack_name = "s.tif"
    c._labels_volume = _volume()
    c._smooth_token = 3
    c._smooth_launch = {"token": 3, "stack": "s.tif",
                        "volume": c._labels_volume, "epoch": c._labels_epoch}
    return c


def test_the_finish_swaps_in_the_copy_and_keeps_the_undo_slot(ctrl, tmp_path):
    old = ctrl._labels_volume
    new = old.copy()
    new[0, 0, 0] = 1
    ctrl._on_smooth_worker_finished(_Result(new, emptied=[3]), "s.tif", 3)
    assert ctrl._labels_volume is new
    assert ctrl._undo_slot["volume"] is old
    assert ctrl._labels_dirty and ctrl._has_unsaved_changes
    assert ctrl._pending_edits["n_thin_kept"] == 1
    assert ctrl._pending_edits["smoothed_at"]
    # Held until Save: the record is not written by the edit (G1-9).
    assert "smoothed_at" not in ctrl.state_manager.get_record("s.tif")
    assert not list(tmp_path.rglob("labels_3d_smoothed.tif"))


def test_a_finish_after_a_stack_switch_is_discarded(ctrl):
    old = ctrl._labels_volume
    ctrl._current_stack_name = "other.tif"
    ctrl._on_smooth_worker_finished(_Result(old.copy()), "s.tif", 3)
    assert ctrl._labels_volume is old
    assert ctrl._undo_slot is None and not ctrl._labels_dirty


def test_a_finish_after_an_edit_is_discarded(ctrl):
    old = ctrl._labels_volume
    ctrl._labels_epoch += 1                  # e.g. Remove bad cells ran
    ctrl._on_smooth_worker_finished(_Result(old.copy()), "s.tif", 3)
    assert ctrl._labels_volume is old


def test_the_failure_sentence(ctrl):
    ctrl._on_smooth_worker_error("boom", "s.tif", 3)
    assert ctrl.ui.qc_status_line.sets[-1] == \
        T.MT_SMOOTH_FAILED.format(stack="s.tif")


# --------------------------------------------------------- a real thread


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


def test_the_smooth_worker_on_a_real_thread(qapp, ctrl):
    worker = _SmoothBatchWorker(ctrl._labels_volume, 2.0, 1.0, True,
                                stack_name="s.tif", token=3)
    import pandas as pd
    before = ctrl._labels_volume
    # Smooth All is gated on features (L7: the backstop refreshes gating).
    ctrl._qc_features_df = pd.DataFrame({"label": [1, 2, 3]})
    ctrl.ui.smooth_all_btn.setEnabled(False)
    ctrl._launch_smooth_worker(worker)
    t0 = time.monotonic()
    while ctrl._smooth_running and time.monotonic() - t0 < 60:
        qapp.processEvents()
        time.sleep(0.01)
    assert not ctrl._smooth_running
    assert ctrl._labels_volume is not before
    assert ctrl.ui.smooth_all_btn.isEnabled()
    for th in getattr(ctrl, "_retired_threads", []):
        th.wait(3000)


def test_a_hand_edit_on_the_layer_makes_the_smooth_stale(ctrl):
    """Audit L9: the layer's data / paint events bump the epoch."""
    old = ctrl._labels_volume
    ctrl._on_labels_layer_edited(None)
    ctrl._on_smooth_worker_finished(_Result(old.copy()), "s.tif", 3)
    assert ctrl._labels_volume is old


def test_the_smooth_launch_lets_go_of_the_volume(ctrl):
    """Audit L8."""
    ctrl._restore_smooth_widgets()
    assert ctrl._smooth_launch["volume"] is None


# ------------------------------------------------ F16: was_smoothed (B1b)


def _dialog_input(ctrl, monkeypatch, record_fields, pending=None):
    """The ``was_smoothed`` the Extract Features dialog is opened with."""
    import napari_epicycle.segmentation.extract_dialog as ED
    got = {}

    class _Dlg:
        def __init__(self, *a, **k):
            got.update(k)

        def exec_(self):
            return 0                       # Rejected: nothing runs

    monkeypatch.setattr(ED, "ExtractFeaturesDialog", _Dlg)
    ctrl.state_manager.set_stack_status("s.tif", "qc_extracted",
                                        **record_fields)
    ctrl._pending_edits = dict(pending or {})
    ctrl._extract_features()
    return got["was_smoothed"]


@pytest.mark.parametrize("fields,pending,expected", [
    ({}, None, False),                                     # never smoothed
    ({"smoothed_at": "2026-09-28T10"}, None, True),        # never measured
    ({"smoothed_at": "2026-09-28T10",
      "measured_at": "2026-09-28T11"}, None, False),       # measured after
    ({"smoothed_at": "2026-09-28T12",
      "measured_at": "2026-09-28T11"}, None, True),        # smoothed after
    ({"measured_at": "2026-09-28T11"},
     {"smoothed_at": "2026-09-28T12", "_stack": "s.tif"}, True),  # held
])
def test_was_smoothed_reads_the_timestamps(ctrl, monkeypatch, fields,
                                           pending, expected):
    """F16: ``smoothed_at is not None and (measured_at is None or
    smoothed_at > measured_at)``; today's status ``smoothed`` is never set."""
    assert _dialog_input(ctrl, monkeypatch, fields, pending) is expected

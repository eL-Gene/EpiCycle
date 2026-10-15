"""Tests for BatchRefineWorker — the QThread wrapper around run_batch_recipe.

A plain ``QApplication`` fixture guarantees Qt is up (pytest-qt is NOT
installed, nor required; ``QT_QPA_PLATFORM=offscreen``).  The worker's
``run()`` is synchronous, so we call it directly and capture its signals
via directly-connected slots (delivered synchronously on the same thread)
-- the ``test_refine_recipe_worker.py`` pattern.  No QtViewer/vispy here,
so no teardown-crash risk.
"""
import os

import numpy as np
import pytest

from packages.epicycle_annotator.batch_refine import BatchResult, RefineStep
from napari_epicycle.annotator.batch_refine_worker import BatchRefineWorker


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _vol():
    v = np.zeros((6, 10, 10), dtype=np.int32)
    v[2:4, 3:6, 3:6] = 1
    v[2:4, 3:6, 6:9] = 2
    return v


def test_worker_emits_finished_batchresult(qapp):
    w = BatchRefineWorker(
        _vol(), [1, 2], [RefineStep("dilate_erode", {"radius": 1})], n_workers=1
    )
    got = []
    w.finished.connect(lambda r: got.append(r))
    w.run()
    assert len(got) == 1
    assert isinstance(got[0], BatchResult)
    assert got[0].cancelled is False
    assert got[0].touched_ids  # dilate changed something


def test_worker_emits_progress_per_label(qapp):
    w = BatchRefineWorker(
        _vol(),
        [1, 2],
        [RefineStep("sdf", {"sigma_z": 1.5, "sigma_xy": 1.5})],
        n_workers=1,
    )
    progress = []
    w.progress.connect(lambda d, t: progress.append((d, t)))
    w.run()
    assert progress, "expected at least one progress emit"
    assert progress[-1][0] == progress[-1][1]  # final done == total


def test_worker_cancel_before_run_leaves_volume_unmodified(qapp):
    vol = _vol()
    w = BatchRefineWorker(
        vol, [1, 2], [RefineStep("dilate_erode", {"radius": 2})], n_workers=1
    )
    w.cancel()
    got = []
    w.finished.connect(lambda r: got.append(r))
    w.run()
    assert got[0].cancelled is True
    assert np.array_equal(got[0].working_volume, vol)


def test_worker_emits_error_on_bad_recipe(qapp):
    w = BatchRefineWorker(_vol(), [1], [RefineStep("bogus", {})], n_workers=1)
    errors = []
    w.error.connect(lambda m: errors.append(m))
    finished = []
    w.finished.connect(lambda r: finished.append(r))
    w.run()
    assert len(errors) == 1
    assert not finished  # error path does not also emit finished

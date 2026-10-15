"""A fresh Measure is never stale (live driver 2026-09-29, MED).

Trace.  ``_measure_stale(rec)`` answered True
whenever ``_labels_dirty`` was set.  ``_labels_dirty`` is written by a paint
(``_on_labels_painted``), Remove bad cells, Undo, a Smooth's finish -- and
by ``_on_extract_finished`` itself: with "remove border cells" on, it zeroes
the border cells in the volume (``_has_unsaved_changes = True``) BEFORE it
records ``measured_at``.  The CSV was computed on exactly that volume, so
right after a Measure the stack read "stale" and the card "Unsaved edits ·
measurements stale".  Every label edit also bumps ``_labels_epoch``; the
Measure now records the epoch it describes (after the border removal), and
unsaved edits make the measurements stale only when the epoch has moved
since.  The border-zeroed volume IS an unsaved edit (the file on disk still
has the border cells), so "Unsaved edits" stays -- without "measurements
stale".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif",))
    w = panel(qapp, ws)
    c = w.controller
    run = ws / "artifacts" / "step4" / "runs" / "r1"
    run.mkdir(parents=True)
    lab = np.zeros((4, 8, 8), np.uint32)
    lab[1, 1:3, 1:3] = 1          # an interior cell
    lab[0, 0:2, 0:2] = 2          # a border cell
    tifffile.imwrite(str(run / "labels_3d.tif"), lab)
    csv = run / "P1_features.csv"
    c.state_manager.set_stack_status(
        "P1.tif", "reconstructed", labels_path=str(run / "labels_3d.tif"),
        qc_features_csv=str(csv))
    c._refresh_stacks()
    c._apply_loaded("P1.tif", {"labels": lab.copy(), "n_cells": 2,
                               "backup": "", "raw": None, "source_path": ""})
    pump(qapp)
    w._csv = csv
    yield w
    close(w)


def _measure(c, border=True):
    df = pd.DataFrame({"label": [1, 2], "volume": [4.0, 4.0],
                       "is_border_cell": [0, 1]})
    c._pending_border_removal = border
    c._on_extract_finished(df, "P1.tif")


def test_a_fresh_measure_with_border_removal_is_not_stale(qapp, w):
    c = w.controller
    _measure(c)
    pump(qapp)
    rec = c.state_manager.get_record("P1.tif")
    assert c._labels_dirty is True             # the border cells went
    assert not c._measure_stale(rec)
    assert w.stack_card_pill.text() == T.MT_STACK_PILL_UNSAVED_FRESH
    assert w.finalize_btn.isEnabled()


def test_an_edit_after_the_measure_makes_it_stale(qapp, w):
    c = w.controller
    _measure(c)
    c._on_labels_painted()                     # a paint after the Measure
    pump(qapp)
    rec = c.state_manager.get_record("P1.tif")
    assert c._measure_stale(rec)
    assert w.stack_card_pill.text() == T.MT_STACK_PILL_UNSAVED


def test_without_border_removal_nothing_is_unsaved(qapp, w):
    c = w.controller
    _measure(c, border=False)
    pump(qapp)
    rec = c.state_manager.get_record("P1.tif")
    assert c._labels_dirty is False
    assert not c._measure_stale(rec)

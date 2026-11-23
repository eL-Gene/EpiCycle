"""The Step 4 row follows the measured file: a first Measure starts the tier
count, and a Save after the table shrank (Remove bad cells) updates the
row's "(N cells)" line.  Run one file at a time with ``-o addopts=""``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile
from qtpy.QtCore import Qt

from ._step4_panel import close, make_workspace, panel, pump

ROLE_RECORD = Qt.UserRole + 2


@pytest.fixture
def w(qapp, tmp_path, monkeypatch):
    ws = make_workspace(tmp_path, patches=("P1.tif",))
    w = panel(qapp, ws)
    c = w.controller
    run = ws / "artifacts" / "step4" / "runs" / "r1"
    run.mkdir(parents=True)
    lab = np.zeros((4, 8, 8), np.uint32)
    lab[1, 1:3, 1:3] = 1
    lab[2, 4:6, 4:6] = 2
    lab[3, 6:8, 0:2] = 3
    tifffile.imwrite(str(run / "labels_3d.tif"), lab)
    csv = run / "P1_features.csv"
    c.state_manager.set_stack_status(
        "P1.tif", "reconstructed", labels_path=str(run / "labels_3d.tif"),
        qc_features_csv=str(csv), stack_shape=[4, 8, 8], axes="ZYX")
    c._refresh_stacks()
    c._apply_loaded("P1.tif", {"labels": lab.copy(), "n_cells": 3,
                               "backup": "", "raw": None, "source_path": ""})
    pump(qapp)
    w._csv = csv
    yield w
    close(w)


def _row_text(w, name="P1.tif"):
    lw = w.stacks_list
    for i in range(lw.count()):
        if lw.item(i).data(Qt.UserRole) == name:
            from napari_epicycle.segmentation import _step4_rows as R
            rec = lw.item(i).data(ROLE_RECORD)
            return rec, R
    raise AssertionError("row not listed")


def test_a_first_measure_restarts_the_tier_counts(qapp, w, monkeypatch):
    c = w.controller
    assert c._qc_features_df is None          # no earlier table
    calls = []
    monkeypatch.setattr(c, "_restart_tier_counts_if_csv_changed",
                        lambda: calls.append("counts"))
    c._on_extract_finished(pd.DataFrame({"label": [1, 2, 3],
                                         "volume": [4.0, 4.0, 4.0]}), "P1.tif")
    assert calls == ["counts"]


def test_a_save_after_cells_were_removed_updates_the_rows_count(
        qapp, w, monkeypatch):
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)
    c = w.controller
    c._on_extract_finished(pd.DataFrame({"label": [1, 2, 3],
                                         "volume": [4.0, 4.0, 4.0],
                                         "qc_tier": ["good", "bad", "good"]}),
                           "P1.tif")
    pump(qapp)
    assert c.state_manager.get_record("P1.tif")["measured_cells"] == 3
    assert "3 cells" in _row_text(w)[0]["shape_text"]
    c._current_stack_name = "P1.tif"
    c._delete_bad_labels()                  # marks the bad row deleted
    assert c._save_tier_assignments() is True
    pump(qapp)
    rec = c.state_manager.get_record("P1.tif")
    assert rec["measured_cells"] == 2
    assert "2 cells" in _row_text(w)[0]["shape_text"]

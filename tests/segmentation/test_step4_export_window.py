"""The Export window (owner decisions 7, 18, 19,
28; F14, F20 export side; I-19).

The real panel with its controller.  Only finalized stacks go, each other
row says why on screen; a stack edited after its Finalize is refused
(``EXPORT_ROW_WHY_EDITED``) and its rows never reach the table; a stack in
voxels goes with the amber ``EXPORT_ROW_UNITS_VOXELS`` and ``meta_*``
columns from ``measured_spacing``; ticks are for one export only; the
meshes are made from the LOADED stack.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_rows as R
from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump

NAMES = ("A.tif", "B.tif", "C.tif", "D.tif", "E.tif")
UM = {"zyx": [1.0, 0.5, 0.5], "source": "step1"}


def _stack(ws, c, name, status, *, spacing=UM, measured=UM, n=3,
           finalize=True):
    run = ws / "artifacts" / "step4" / "runs" / Path(name).stem
    run.mkdir(parents=True, exist_ok=True)
    lab = run / "labels_3d.tif"
    tifffile.imwrite(str(lab), np.zeros((2, 3, 3), np.uint32))
    fields = {"labels_path": str(lab)}
    if status in ("qc_extracted", "finalized"):
        csv = ws / "artifacts" / "step4" / "qc" / f"{name}_features.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"label": list(range(1, n + 1)), "area": [5.0] * n,
                      "label_status": ["active"] * n}).to_csv(csv,
                                                               index=False)
        fields.update(qc_features_csv=str(csv), measured_spacing=measured,
                      measured_at="2026-09-28T12:00:00+00:00")
    fields["spacing"] = spacing
    fields["made_with"] = M.model_snapshot("cyto3", "cyto3",
                                           kind="pretrained")
    c.state_manager.set_stack_status(name, status, **fields)
    if status == "finalized" and finalize:
        c.state_manager.update_record(
            name, finalized_fingerprint=M.finalized_fingerprint(
                str(lab), fields["qc_features_csv"]))


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=NAMES)
    w = panel(qapp, ws)
    c = w.controller
    _stack(ws, c, "A.tif", "finalized")
    _stack(ws, c, "B.tif", "finalized", spacing=None, measured=None)
    _stack(ws, c, "C.tif", "qc_extracted")
    _stack(ws, c, "D.tif", "reconstructed")
    c._refresh_stacks()
    pump(qapp)
    w._ws = ws
    yield w
    w.export_dialog.hide()
    close(w)


def _rows(w):
    lw = w.export_rows_list
    return {lw.item(i).data(0x0100): lw.item(i) for i in range(lw.count())}


def test_finalized_only_with_reasons_on_screen(qapp, w):
    w.controller._open_export_window()
    pump(qapp)
    rows = _rows(w)
    assert set(rows) == {"A.tif", "B.tif", "C.tif", "D.tif"}  # E: not recon
    rec = {n: it.data(0x0100 + 2) for n, it in rows.items()}
    assert rec["A.tif"]["can_go"] and rec["A.tif"]["why"] == ""
    assert rec["B.tif"]["can_go"]
    assert rec["B.tif"]["why"] == T.EXPORT_ROW_UNITS_VOXELS
    assert not rec["C.tif"]["can_go"]
    assert rec["C.tif"]["why"] == T.EXPORT_ROW_WHY_FINALIZE
    assert rec["D.tif"]["why"] == T.EXPORT_ROW_WHY_MEASURE
    from qtpy.QtCore import Qt
    assert rows["A.tif"].checkState() == Qt.Checked
    assert not rows["C.tif"].flags() & Qt.ItemIsUserCheckable
    assert w.export_dialog.count_label.text() == \
        T.say(T.EXPORT_FINALIZED_COUNT, k=2, n=4)
    assert w.export_features_btn.text() == T.say(T.EXPORT_FEATURES_BTN, k=2)
    assert w.export_features_btn.isEnabled()


def test_nothing_finalized_blocks(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("C.tif",))
    w = panel(qapp, ws)
    try:
        _stack(ws, w.controller, "C.tif", "qc_extracted")
        w.controller._open_export_window()
        pump(qapp)
        assert not w.export_features_btn.isEnabled()
        assert w.export_status_label.text() == T.EXPORT_BLOCK
    finally:
        w.export_dialog.hide()
        close(w)


def test_ticks_are_for_one_export_only(qapp, w):
    from qtpy.QtCore import Qt
    c = w.controller
    c._open_export_window()
    pump(qapp)
    rows = _rows(w)
    rows["A.tif"].setCheckState(Qt.Unchecked)
    rows["B.tif"].setCheckState(Qt.Unchecked)
    assert not w.export_features_btn.isEnabled()
    assert w.export_status_label.text() == T.EXPORT_NONE_TICKED
    w.export_dialog.hide()
    c._open_export_window()                        # opened again
    pump(qapp)
    rows = _rows(w)
    assert rows["A.tif"].checkState() == Qt.Checked
    assert rows["B.tif"].checkState() == Qt.Checked


def test_a_stack_edited_after_finalize_is_refused(qapp, w, monkeypatch):
    """U1-3 / I-19: Remove bad cells + Save after Finalize changes the
    files Finalize saw; the row is off and the table leaves it out."""
    c = w.controller
    lab = Path(c.state_manager.get_record("A.tif")["labels_path"])
    time.sleep(0.02)
    tifffile.imwrite(str(lab), np.ones((2, 3, 3), np.uint32))
    c._open_export_window()
    pump(qapp)
    row = _rows(w)["A.tif"].data(0x0100 + 2)
    assert not row["can_go"] and row["why"] == T.EXPORT_ROW_WHY_EDITED
    out = w._ws / "out.csv"
    from qtpy.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    c._export_all_stacks()
    df = pd.read_csv(out)
    assert set(df["stack_name"]) == {"B.tif"}


def test_a_spacing_change_after_finalize_is_refused(qapp, w):
    c = w.controller
    c.state_manager.update_record(
        "A.tif", spacing={"zyx": [2.0, 0.5, 0.5], "source": "user"})
    c._refresh_export_window()
    row = _rows(w)["A.tif"].data(0x0100 + 2)
    assert row["why"] == T.EXPORT_ROW_WHY_EDITED


def test_the_meta_columns_and_mixed_units(qapp, w, monkeypatch):
    c = w.controller
    out = w._ws / "artifacts" / "step4" / "exports" / "all.csv"
    from qtpy.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    c._open_export_window()
    pump(qapp)
    c._export_all_stacks()
    df = pd.read_csv(out, keep_default_na=False)
    for col in R.META_COLUMNS:
        assert col in df.columns
    a = df[df["stack_name"] == "A.tif"].iloc[0]
    b = df[df["stack_name"] == "B.tif"].iloc[0]
    assert a["meta_units"] == "um" and float(a["meta_spacing_z"]) == 1.0
    assert b["meta_units"] == "voxels" and b["meta_spacing_z"] == ""
    assert a["meta_model"] == M.model_short_label(
        M.model_snapshot("cyto3", "cyto3", kind="pretrained"))
    assert len(df) == 6
    done = T.say(T.EXPORT_FEATURES_DONE, k=2, n=6, file="all.csv")
    assert w.export_status_label.text() == done
    assert w.export_status_label.toolTip() == str(out)   # the path: tooltip
    last = w.export_dialog.last_label.text()
    assert last.endswith(" · 2 stacks · all.csv")
    assert c._export_running is False
    rec = c.state_manager.get_session_metadata()["last_export"]
    assert rec["n_stacks"] == 2 and rec["path"] == str(out)


def test_the_meta_columns_come_from_measure_time():
    """F14 / G1-2: ``measured_spacing``, never ``spacing``."""
    df = pd.DataFrame({"stack_name": ["A.tif"], "area": [1.0]})
    out = R.add_meta_columns(df, {"A.tif": {
        "spacing": {"zyx": [9.0, 9.0, 9.0]},
        "measured_spacing": {"zyx": [1.0, 0.5, 0.25]}}})
    assert list(out.loc[0, ["meta_spacing_z", "meta_spacing_y",
                            "meta_spacing_x"]]) == [1.0, 0.5, 0.25]


def test_meshes_are_made_from_the_loaded_stack(qapp, w, monkeypatch):
    c = w.controller
    c._apply_loaded("C.tif", {"labels": np.zeros((2, 3, 3), np.uint32),
                              "n_cells": 3, "backup": "", "raw": None,
                              "source_path": ""})
    launched = []

    class _Proc:
        def __init__(self, args, **k):
            launched.append(args)
            self.stdout = self.stderr = None

        def poll(self):
            return 0

    import subprocess
    monkeypatch.setattr(subprocess, "Popen", _Proc)
    c._open_export_window()
    w.export_meshes_btn.click()
    assert launched and "C.tif" in str(launched[0][1])
    assert c._mesh_running
    assert w.mt_lock_line.text() == T.STEP4_LOCK_EXPORT
    assert not w.export_features_btn.isEnabled()
    assert w.export_dialog.close_btn.isEnabled()
    c._mesh_timer.stop()
    c._on_mesh_poll()                               # the process ended
    assert not c._mesh_running
    assert w.export_status_label.text() == T.EXPORT_MESH_FAILED


def test_a_failed_save_stops_the_mesh_export(qapp, w, monkeypatch):
    c = w.controller
    c._apply_loaded("C.tif", {"labels": np.zeros((2, 3, 3), np.uint32),
                              "n_cells": 3, "backup": "", "raw": None,
                              "source_path": ""})
    c._labels_dirty = True
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)
    c.state_manager.save_blocked = True
    import subprocess
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: pytest.fail("meshes from old cells"))
    c._export_meshes()
    assert not c._mesh_running
    assert w.export_status_label.text() == T.EXPORT_MESH_FAILED


def test_export_rows_pure():
    """Q1: a legacy finalized stack (no fingerprint) with no
    ``finalized_at`` to compare its files with does not go."""
    row = R.export_row("x.tif", {"status": "finalized",
                                 "qc_features_csv": __file__,
                                 "measured_spacing": UM, "spacing": UM})
    assert not row["can_go"] and row["why"] == T.EXPORT_ROW_WHY_EDITED


def test_an_open_window_follows_the_panel(qapp, w):
    """B1-3: an edit made while the window is open reaches its rows; the
    ticks the user took off stay off."""
    from qtpy.QtCore import Qt
    c = w.controller
    c._open_export_window()
    pump(qapp)
    _rows(w)["B.tif"].setCheckState(Qt.Unchecked)
    lab = Path(c.state_manager.get_record("A.tif")["labels_path"])
    time.sleep(0.02)
    tifffile.imwrite(str(lab), np.ones((2, 3, 3), np.uint32))
    c._refresh_mt_panel()
    pump(qapp)
    rows = _rows(w)
    assert rows["A.tif"].data(0x0100 + 2)["why"] == T.EXPORT_ROW_WHY_EDITED
    assert rows["B.tif"].checkState() == Qt.Unchecked


def test_a_ticked_stack_that_can_no_longer_go_is_said(qapp, w, monkeypatch):
    """B1-3: never dropped silently; the list refreshes and says so."""
    c = w.controller
    c._open_export_window()
    pump(qapp)
    lab = Path(c.state_manager.get_record("A.tif")["labels_path"])
    time.sleep(0.02)
    tifffile.imwrite(str(lab), np.ones((2, 3, 3), np.uint32))
    from qtpy.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(
        lambda *a, **k: pytest.fail("exported a changed list")))
    dlg = w.export_dialog
    # the window still held A as ticked when Export features was pressed
    monkeypatch.setattr(dlg, "ticked_names", lambda: ["A.tif", "B.tif"])
    c._export_all_stacks()
    assert w.export_status_label.text() == T.EXPORT_ROWS_CHANGED
    assert _rows(w)["A.tif"].data(0x0100 + 2)["why"] == \
        T.EXPORT_ROW_WHY_EDITED


def test_the_loaded_stacks_unsaved_edits_are_named(qapp, w):
    """G1-2: the row says its unsaved edits are not in the table."""
    c = w.controller
    c._apply_loaded("A.tif", {"labels": np.zeros((2, 3, 3), np.uint32),
                              "n_cells": 3, "backup": "", "raw": None,
                              "source_path": ""})
    c._labels_dirty = True
    c._open_export_window()
    pump(qapp)
    row = _rows(w)["A.tif"].data(0x0100 + 2)
    assert row["can_go"] and row["why"] == T.EXPORT_ROW_UNSAVED


def test_the_default_file_is_dated_and_counts_the_stacks(qapp, w,
                                                         monkeypatch):
    """Decided 2026-09-29: the default name is
    features_<YYYY-MM-DD>_<n>stacks.csv under artifacts/step4/exports/, so
    a re-export never overwrites the previous table."""
    import datetime as _dt
    c = w.controller
    seen = []
    from qtpy.QtWidgets import QFileDialog

    def _ask(parent, title, default, flt):
        seen.append(default)
        return "", ""                      # cancelled: nothing written

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(_ask))
    c._open_export_window()
    pump(qapp)
    c._export_all_stacks()
    today = _dt.date.today().isoformat()
    assert seen, "no save dialog"
    assert Path(seen[0]) == (w._ws / "artifacts" / "step4" / "exports"
                             / T.EXPORT_DEFAULT_NAME.format(date=today, n=2))
    assert Path(seen[0]).name == f"features_{today}_2stacks.csv"


def test_the_mesh_done_sentence_stays_on_screen(qapp, tmp_path):
    """Live driver 2026-09-29: _finish_mesh_export wrote the done sentence
    and then refreshed ③, whose Export-window refresh overwrote it with
    EXPORT_BLOCK ('Finalize a stack first.')."""
    import io
    ws = make_workspace(tmp_path, patches=("A.tif",))
    w = panel(qapp, ws)
    try:
        c = w.controller
        c._open_export_window()
        pump(qapp)
        c._mesh_job = {"stack": "A.tif"}

        class _Proc:
            stdout = io.StringIO("DONE:5\nSKIPPED:0\nSKIPPED_IDS:\n")
            stderr = io.StringIO("")

        c._finish_mesh_export(_Proc(), 0, tmp_path, 5, tmp_path / "x.py")
        pump(qapp)
        assert w.export_status_label.text() == T.say(
            T.EXPORT_MESH_DONE, n=5, stack="A.tif")
    finally:
        w.export_dialog.hide()
        close(w)

"""A3: Step 4 takes the shared sizes,
the overlay on measured stacks, the metadata / voxel lines on its narrow
rows, the accent tick and the bucket counts.  Qt-backed through the real
panel; run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari_epicycle.segmentation import _step4_rows as R
from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.segmentation import _stack_thumbs as ST

from ._step4_panel import close, make_workspace, panel, pump

ENTRY = {"name": "s.tif", "path": "", "origin": "step1", "conflict": False}
ROLE_THUMB = Qt.UserRole + 3
ROLE_RECORD = Qt.UserRole + 2


def _rec(**kw):
    base = {"status": "unprocessed", "axes": "CZYX",
            "stack_shape": [2, 6, 24, 24], "n_channels": 2,
            "spacing": {"zyx": [2.0, 0.5, 0.5], "source": "step1"}}
    base.update(kw)
    return base


# -- the record --------------------------------------------------------------

def test_a_narrow_row_carries_the_voxel_line():
    row = R.build_row(ENTRY, _rec())
    assert row["vox_text"] == row["spacing_text"] and not row["vox_warn"]
    none = R.build_row(ENTRY, _rec(spacing=None))
    assert none["vox_text"] == T.ROW_SPACING_NONE and none["vox_warn"]


def test_a_measured_row_names_its_labels_and_its_csv():
    row = R.build_row(ENTRY, _rec(status="qc_extracted", n_cells=4,
                                  labels_path="C:/x/l.tif",
                                  qc_features_csv="C:/x/f.csv"))
    assert row["labels_path"] == "C:/x/l.tif"
    assert row["features_csv"] == "C:/x/f.csv"
    plain = R.build_row(ENTRY, _rec(labels_path="C:/x/l.tif"))
    assert plain["labels_path"] == "" and plain["features_csv"] == ""


# -- the panel ---------------------------------------------------------------

@pytest.fixture
def ws(tmp_path):
    return make_workspace(tmp_path, patches=("P1.tif", "P2.tif"))


def _item_thumbs(lw):
    return {lw.item(i).data(Qt.UserRole): lw.item(i).data(ROLE_THUMB)
            for i in range(lw.count())}


def _pm():
    from qtpy.QtGui import QColor, QPixmap
    pm = QPixmap(8, 8)
    pm.fill(QColor("#336699"))
    return pm


def test_the_lists_take_the_shared_constants_and_the_green_accent(qapp, ws):
    from napari_epicycle.ui_shell.theme import STEP4_ACCENT
    from napari_epicycle.segmentation import ui as U
    from napari_epicycle.ui_shell import _stack_row_delegate as D
    w = panel(qapp, ws)
    try:
        for dlg, mode in ((w._session_delegate, D.MODE_WIDE),
                          (w._seg_delegate, D.MODE_NARROW),
                          (w._qc_delegate, D.MODE_MEASURE)):
            assert dlg._palette["accent"].name().lower() == STEP4_ACCENT
        assert w._seg_delegate._show_thumb and w._qc_delegate._show_thumb
        assert U.SESSION_LIST_MAX_H == D.list_cap_px(D.MODE_WIDE)
    finally:
        close(w)


def test_one_picture_reaches_all_three_lists_and_survives_a_rebuild(
        qapp, ws, monkeypatch):
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (None, None))
    w = panel(qapp, ws)
    try:
        c = w.controller
        # P1 is measured, so ③ lists it too
        c.state_manager.set_stack_status("P1.tif", "qc_extracted",
                                         labels_path="x", n_cells=3)
        c._refresh_stacks()
        pump(qapp)
        path = next(r["path"] for r in c._rows if r["name"] == "P1.tif")
        assert w.set_row_thumbnail(path, _pm())
        for lw in (w.session_stacks_list, w.stacks_list, w.qc_stacks_list):
            assert _item_thumbs(lw)["P1.tif"] is not None, lw
        c._refresh_stacks()
        pump(qapp)
        for lw in (w.session_stacks_list, w.stacks_list, w.qc_stacks_list):
            assert _item_thumbs(lw)["P1.tif"] is not None, lw
    finally:
        close(w)


def test_a_measured_stack_asks_for_the_overlay(qapp, ws, monkeypatch):
    import tifffile
    started = []
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (started.append((a, k)) or
                                         (None, None)))
    w = panel(qapp, ws)
    try:
        lab = ws / "labels.tif"
        tifffile.imwrite(str(lab), np.ones((6, 24, 24), np.uint16))
        c = w.controller
        c.state_manager.set_stack_status("P1.tif", "reconstructed",
                                         labels_path=str(lab))
        c._refresh_stacks()
        c._start_thumbs()
        reqs = {r.name: r for r in started[-1][0][1]}
        assert reqs["P1.tif"].labels_path == str(lab)
        assert reqs["P1.tif"].labels_mtime_ns == os.stat(lab).st_mtime_ns
        assert reqs["P2.tif"].labels_path == ""
        # the pictures are drawn at the wide size
        assert started[-1][0][3] == ST.THUMB_WIDE_PX
    finally:
        close(w)


def test_new_labels_restart_the_pictures(qapp, ws, monkeypatch):
    import tifffile
    started = []
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (started.append((a, k)) or
                                         (None, None)))
    w = panel(qapp, ws)
    try:
        c = w.controller
        n0 = len(started)
        lab = ws / "labels.tif"
        tifffile.imwrite(str(lab), np.ones((6, 24, 24), np.uint16))
        c.state_manager.set_stack_status("P1.tif", "reconstructed",
                                         labels_path=str(lab))
        c._refresh_stacks()
        pump(qapp)
        assert len(started) == n0 + 1, "labels appeared: draw them"
        c._refresh_stacks()
        pump(qapp)
        assert len(started) == n0 + 1, "nothing changed: no new worker"
    finally:
        close(w)


# -- the bucket counts -------------------------------------------------------

def _write_csv(path, tiers):
    import pandas as pd
    pd.DataFrame({"label": range(len(tiers)),
                  "qc_tier": tiers}).to_csv(path, index=False)


def _wait(qapp, pred, secs=20):
    import time
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs and not pred():
        qapp.processEvents()
        time.sleep(0.01)
    return pred()


def _records(lw):
    return {lw.item(i).data(Qt.UserRole): lw.item(i).data(ROLE_RECORD)
            for i in range(lw.count())}


def test_measured_stacks_show_good_bad_unmarked_off_the_main_thread(
        qapp, ws, monkeypatch):
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (None, None))
    w = panel(qapp, ws)
    try:
        c = w.controller
        csv = ws / "f.csv"
        _write_csv(csv, ["good", "good", "garbage", "bad", None, "review"])
        c.state_manager.set_stack_status("P1.tif", "qc_extracted",
                                         labels_path="x", n_cells=6,
                                         qc_features_csv=str(csv))
        c._refresh_stacks()
        assert _wait(qapp, lambda: "tier_counts" in
                     _records(w.session_stacks_list)["P1.tif"])
        for lw in (w.session_stacks_list, w.qc_stacks_list):
            items = _records(lw)["P1.tif"]["tier_counts"]
            assert [(t[0], t[1]) for t in items] == [
                ("good", 2), ("bad", 2), ("unmarked", 2)]
        # ③ draws the words too (dot, number, word), ① always did
        assert _records(w.qc_stacks_list)["P1.tif"]["tier_words"] is True
        # a stack that is not measured shows nothing; ② never does
        assert "tier_counts" not in _records(w.session_stacks_list)["P2.tif"]
        assert "tier_counts" not in _records(w.stacks_list)["P1.tif"]
        # the Workbench's colours
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            _WB_BUCKET_RGB)
        rgb = {t[0]: t[2] for t in
               _records(w.session_stacks_list)["P1.tif"]["tier_counts"]}
        assert rgb["good"] == tuple(_WB_BUCKET_RGB["good"][:3])
        # the counts survive a rebuild of the rows, with no new worker
        c._refresh_stacks()
        pump(qapp)
        assert "tier_counts" in _records(w.session_stacks_list)["P1.tif"]
    finally:
        close(w)


def test_a_changed_csv_is_counted_again(qapp, ws, monkeypatch):
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (None, None))
    w = panel(qapp, ws)
    try:
        c = w.controller
        csv = ws / "f.csv"
        _write_csv(csv, ["good"])
        c.state_manager.set_stack_status("P1.tif", "qc_extracted",
                                         labels_path="x", n_cells=1,
                                         qc_features_csv=str(csv))
        c._refresh_stacks()
        assert _wait(qapp, lambda: "tier_counts" in
                     _records(w.session_stacks_list)["P1.tif"])
        _write_csv(csv, ["good", "bad", "bad"])
        os.utime(csv, (1_900_000_000, 1_900_000_000))
        c._refresh_stacks()
        assert _wait(qapp, lambda: [t[1] for t in _records(
            w.session_stacks_list)["P1.tif"].get("tier_counts", ())]
            == [1, 2, 0])
    finally:
        close(w)


# -- the loaded-row indicator ---------------------------------------------------

def _loaded_names(lw):
    return [lw.item(i).data(Qt.UserRole) for i in range(lw.count())
            if (lw.item(i).data(ROLE_RECORD) or {}).get("loaded")]


def test_one_measure_row_is_loaded_and_none_after_unload(qapp, ws, monkeypatch):
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (None, None))
    w = panel(qapp, ws)
    try:
        c = w.controller
        for n in ("P1.tif", "P2.tif"):
            c.state_manager.set_stack_status(n, "reconstructed",
                                             labels_path="x", n_cells=1)
        c._refresh_stacks()
        lw = w.qc_stacks_list
        assert _loaded_names(lw) == []
        c._current_stack_name = "P2.tif"
        c._sync_loaded_stack_marker()
        assert _loaded_names(lw) == ["P2.tif"]
        c._refresh_stacks()                              # a rebuild
        assert _loaded_names(lw) == ["P2.tif"]
        lw.setCurrentRow(0)                              # a selection change
        assert _loaded_names(lw) == ["P2.tif"]
        c._current_stack_name = "P1.tif"                 # a switch
        c._sync_loaded_stack_marker()
        assert _loaded_names(lw) == ["P1.tif"]
        c._current_stack_name = None                     # unload
        c._sync_loaded_stack_marker()
        assert _loaded_names(lw) == []
        assert _loaded_names(w.session_stacks_list) == []   # ① never marks
    finally:
        close(w)

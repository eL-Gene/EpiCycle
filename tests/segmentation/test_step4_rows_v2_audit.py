"""A3 audit fold (Step 4 side): Save as a trigger, a REAL shutdown of the
counter, the stale-batch guard, the raw-fallback tail, one thumbnail worker
on a first load, one "no voxel size" on a narrow row.  Run one file at a time
with ``-o addopts=""``.
"""
from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari_epicycle.segmentation import _step4_rows as R
from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.segmentation import _stack_thumbs as ST

from ._step4_panel import close, make_workspace, panel, pump

ROLE_RECORD = Qt.UserRole + 2
ENTRY = {"name": "s.tif", "path": "", "origin": "step1", "conflict": False}


@pytest.fixture
def ws(tmp_path):
    return make_workspace(tmp_path, patches=("P1.tif", "P2.tif"))


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


def test_a_save_restarts_the_counts_and_the_pictures(tmp_path, monkeypatch):
    """M1: Save is a trigger."""
    import pandas as pd
    import tifffile
    from ._step4_fakes import controller
    ctrl = controller(tmp_path)
    run = tmp_path / "artifacts" / "step4" / "runs" / "r"
    run.mkdir(parents=True)
    labels = run / "labels_3d.tif"
    tifffile.imwrite(str(labels), np.zeros((3, 4, 4), np.uint32))
    csv = tmp_path / "artifacts" / "step4" / "qc" / "s.tif_features.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("label\n7\n", encoding="utf-8")
    ctrl.state_manager.set_stack_status("s.tif", "qc_extracted",
                                        labels_path=str(labels),
                                        qc_features_csv=str(csv))
    ctrl._current_stack_name = "s.tif"
    ctrl._labels_volume = np.zeros((3, 4, 4), np.uint32)
    ctrl._qc_features_df = pd.DataFrame({"label": [7], "qc_tier": ["good"]})
    ctrl._labels_dirty = True
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)
    calls = []
    monkeypatch.setattr(ctrl, "_restart_tier_counts_if_csv_changed",
                        lambda: calls.append("counts"), raising=False)
    monkeypatch.setattr(ctrl, "_restart_thumbs_if_labels_changed",
                        lambda: calls.append("thumbs"), raising=False)
    assert ctrl._save_tier_assignments() is True
    assert sorted(calls) == ["counts", "thumbs"]


def test_shutdown_stops_a_running_counter_and_a_late_result_lands_nowhere(
        qapp, ws, monkeypatch):
    """M4: a REAL running batch, blocked, then shutdown (the old test never
    started a batch)."""
    import threading
    from napari_epicycle.fusion_reconstruction import _step3_tier_counts as TC
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (None, None))
    gate = threading.Event()
    real = TC.count_batch

    def blocking(*a, **k):
        gate.wait(20)
        return real(*a, **k)

    monkeypatch.setattr(TC, "count_batch", blocking)
    w = panel(qapp, ws)
    try:
        c = w.controller
        csv = ws / "f.csv"
        _write_csv(csv, ["good", "bad"])
        c.state_manager.set_stack_status("P1.tif", "qc_extracted",
                                         labels_path="x", n_cells=2,
                                         qc_features_csv=str(csv))
        c._refresh_stacks()
        state = c._tier_counts_state
        assert _wait(qapp, state.running), "a batch must be running"
        worker, token = state.worker, c._tier_counts_token
        c.shutdown()
        assert c._tier_counts_token is None
        assert state.thread is None and state.worker is None
        gate.set()
        worker.counted.emit((token, c.working_dir),
                            {"P1.tif": {"good": 1, "bad": 1, "unmarked": 0}})
        pump(qapp)
        assert "tier_counts" not in _records(w.session_stacks_list)["P1.tif"]
    finally:
        gate.set()
        close(w)


def test_a_refresh_after_shutdown_does_not_raise(qapp, ws, monkeypatch):
    """L2: the token is None after shutdown."""
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
        c.shutdown()
        c._refresh_stacks()                    # a queued refresh, late
    finally:
        close(w)


def test_a_stale_batch_or_workspace_changes_no_row(qapp, ws, monkeypatch):
    """M5: the guard in ``_on_tier_counts_counted``."""
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (None, None))
    w = panel(qapp, ws)
    try:
        c = w.controller
        csv = ws / "f.csv"
        _write_csv(csv, ["good", "good"])
        c.state_manager.set_stack_status("P1.tif", "qc_extracted",
                                         labels_path="x", n_cells=2,
                                         qc_features_csv=str(csv))
        c._refresh_stacks()
        c._tier_counts_token = 5
        res = {"P1.tif": {"good": 2, "bad": 0, "unmarked": 0}}
        c._on_tier_counts_counted((4, c.working_dir), res)     # old token
        c._on_tier_counts_counted((5, "A"), res)               # wrong ws
        for lw in (w.session_stacks_list, w.qc_stacks_list):
            rec = _records(lw).get("P1.tif") or {}
            assert not any(t[1] == 2 for t in rec.get("tier_counts", ()))
        c._on_tier_counts_counted((5, c.working_dir), res)     # the live one
        assert any(t[1] == 2 for t in
                   _records(w.session_stacks_list)["P1.tif"]["tier_counts"])
    finally:
        close(w)


def test_the_raw_fallback_is_told_by_the_name_tail():
    """L1: a stack called ``a.l1`` is not an overlaid picture."""
    raw = ST.thumb_png_path("/c", "a.l1.tif", "/x/a.l1.tif", 5, 6)
    lab = ST.thumb_png_path("/c", "a.l1.tif", "/x/a.l1.tif", 5, 6,
                            labels_mtime_ns=9)
    assert ST.has_labels_part(lab) and not ST.has_labels_part(raw)
    assert not ST.has_labels_part("")


def test_the_step4_thumb_site_uses_the_tail(qapp, tmp_path, monkeypatch):
    """L1, Step 4's site: unreadable labels on a stack named ``a.l1``."""
    ws = make_workspace(tmp_path, patches=("a.l1.tif",))
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (None, None))
    w = panel(qapp, ws)
    try:
        c = w.controller
        row = next(r for r in c._rows if r["name"] == "a.l1.tif")
        key = R.norm_path(row["path"])
        c._thumb_requested = {key: (1, 2, 77)}
        seen = []
        real = c._thumb_cache.load

        def spy(*a, **k):
            seen.append(k.get("extra"))
            return real(*a, **k)

        monkeypatch.setattr(c._thumb_cache, "load", spy)
        raw = ST.thumb_png_path(tmp_path, "a.l1.tif", row["path"], 1, 2)
        c._on_thumb_ready("a.l1.tif", row["path"], str(raw), b"")
        assert seen == [0]
    finally:
        close(w)


def test_the_overlay_colour_table_is_built_once():
    assert ST.label_rgb_table() is ST.label_rgb_table()


def test_first_load_starts_one_thumbnail_worker(qapp, ws, monkeypatch):
    """L4: not two (the labels restart and the explicit start)."""
    import tifffile
    started = []
    monkeypatch.setattr(ST, "start_thumb_worker",
                        lambda *a, **k: (started.append(1) or (None, None)))
    w = panel(qapp, ws)
    try:
        c = w.controller
        lab = ws / "labels.tif"
        tifffile.imwrite(str(lab), np.ones((6, 24, 24), np.uint16))
        c.state_manager.set_stack_status("P1.tif", "reconstructed",
                                         labels_path=str(lab))
        c._thumb_labels_seen = {}
        started.clear()
        c._rescan_stacks(persist=False)
        pump(qapp)
        assert len(started) == 1, started
    finally:
        close(w)


def test_a_narrow_row_says_no_voxel_size_once():
    """L7."""
    rec = {"status": "unprocessed", "axes": "CZYX",
           "stack_shape": [2, 6, 24, 24], "n_channels": 2, "spacing": None}
    row = R.build_row(ENTRY, rec)
    assert row["vox_text"] == T.ROW_SPACING_NONE and row["vox_warn"]
    assert T.ROW_WARN_NO_SPACING_SHORT not in row["warns_narrow"]

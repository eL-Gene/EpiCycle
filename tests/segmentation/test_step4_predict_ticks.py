"""Predict Selected and Reconstruct Selected over the ticks (F6, F10): every ticked stack with its OWN axes and channels, a
failure counted in Step 3's done sentence, "failed" as a row status."""
from __future__ import annotations

import time

import numpy as np
import pytest

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.ui_shell import theme as TH

from ._step4_panel import close, make_workspace, panel, pump, tick


def _ready(w, qapp):
    c = w.controller
    c._on_model_use({"_kind": "pretrained", "name": "cyto3"})
    w.set_channel_choice(w.chan1_combo, 0)
    w.set_channel_choice(w.chan2_combo, None)
    pump(qapp)
    return c


@pytest.fixture
def ws(tmp_path):
    return make_workspace(tmp_path, patches=("P1.tif", "P2.tif", "P3.tif"),
                          zcyx=("P2.tif",))


def test_every_ticked_stack_gets_its_own_config(qapp, ws, monkeypatch):
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        seen = []
        monkeypatch.setattr(c, "_launch_prediction_worker", seen.extend)
        tick(w, qapp, "P1.tif", "P2.tif", "P3.tif")
        w.cell_width_um_sb.setValue(8.0)
        assert w.predict_btn.isEnabled()
        w.predict_btn.click()
        pump(qapp)
        assert [cfg["stack_name"] for cfg in seen] == [
            "P1.tif", "P2.tif", "P3.tif"]
        axes = {cfg["stack_name"]: cfg["axes"] for cfg in seen}
        assert axes == {"P1.tif": "CZYX", "P2.tif": "ZCYX", "P3.tif": "CZYX"}
        for cfg in seen:
            assert cfg["seg_config"].channels == [1, 0]
            # 8 µm at 0.5 µm per pixel: 16 px, from the stack's own spacing
            assert cfg["seg_config"].diameter == 16.0
            assert cfg["model_snapshot"]["kind"] == "pretrained"
            assert cfg["cache_meta"]["axes"] == cfg["axes"]
        assert any(T.LOG_PREDICT_START.split(" {")[0] in line
                   for line in w.log_output.toPlainText().splitlines())
    finally:
        close(w)


def test_only_ticked_runnable_stacks_run(qapp, tmp_path, monkeypatch):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"),
                        one_channel=("P2.tif",))
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        w.set_channel_choice(w.chan2_combo, 1)   # nuclei = channel 1
        pump(qapp)
        seen = []
        monkeypatch.setattr(c, "_launch_prediction_worker", seen.extend)
        tick(w, qapp, "P1.tif", "P2.tif")
        w.predict_btn.click()
        pump(qapp)
        # P2 has one channel and cannot run with two: skipped, named
        assert [cfg["stack_name"] for cfg in seen] == ["P1.tif"]
        assert "P2.tif" in w.log_output.toPlainText()
    finally:
        close(w)


def _fake_views(z=6, y=24, x=24):
    views = {}
    for v, shape in (("xy", (z, y, x)), ("xz", (y, z, x)),
                     ("yz", (x, z, y))):
        views[v] = (np.zeros((2,) + shape, np.float32),
                    np.zeros(shape, np.float32), None)
    return views


def test_a_failure_is_counted_and_shown(qapp, ws, monkeypatch):
    """A real worker on a real QThread: P2's prediction raises."""
    import packages.epicycle_reconstruction.logic as logic

    def fake_segment_views(image, lat, ax, cfg, axes="ZCYX", progress_cb=None):
        if image.shape[1] == 2 and axes == "ZCYX":
            raise RuntimeError("the GPU fell over")
        if progress_cb:
            progress_cb("xy", 3, 6)
        return _fake_views()

    monkeypatch.setattr(logic, "segment_views", fake_segment_views)
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        tick(w, qapp, "P1.tif", "P2.tif", "P3.tif")
        w.predict_btn.click()
        deadline = time.monotonic() + 30
        while c._predict_running and time.monotonic() < deadline:
            pump(qapp)
            time.sleep(0.02)
        pump(qapp, 10)
        assert not c._predict_running
        status = w.run_status_label.toolTip() or w.run_status_label.text()
        head = TH.PIPE_SEGMENTATION_DONE.format(s=2, n=3, units="stacks")
        assert status.startswith(head + TH.PIPE_FAILED_SUFFIX.format(k=1))
        row = next(r for r in c._rows if r["name"] == "P2.tif")
        assert row["word"] == "failed"
        assert c.state_manager.get_record("P2.tif")["failed_stage"] == \
            "predict"
        assert "the GPU fell over" in w.log_output.toPlainText()
        assert "the GPU fell over" not in status   # raw text: log only
        for n in ("P1.tif", "P3.tif"):
            rec = c.state_manager.get_record(n)
            assert rec["status"] == "predicted"
            assert rec["predicted_with"]["kind"] == "pretrained"
            assert rec["predicted_axes"] == "CZYX"
    finally:
        close(w)


def test_reconstruct_takes_the_ticked_stacks_with_a_complete_cache(
        qapp, ws, monkeypatch):
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        snap = c._model_snapshot()
        for n in ("P1.tif", "P3.tif"):
            c._on_prediction_completed(n, {
                "views": _fake_views(), "source_stack_path": str(
                    ws / "artifacts/step1/patches" / n),
                "model_snapshot": snap,
                "cache_meta": {"model": snap, "axes": "CZYX"}})
        pump(qapp)
        seen = []
        monkeypatch.setattr(c, "_launch_reconstruction_worker",
                            lambda cfgs, pipeline=None: seen.extend(cfgs))
        tick(w, qapp, "P1.tif", "P2.tif", "P3.tif")
        assert w.reconstruct_btn.isEnabled()
        w.reconstruct_btn.click()
        pump(qapp)
        assert [cfg["stack_name"] for cfg in seen] == ["P1.tif", "P3.tif"]
        for cfg in seen:
            assert cfg["require_cache"] is True
            assert M.same_model(cfg["made_with"], snap)
        assert T.LOG_NO_CACHE.format(stack="P2.tif") in \
            w.log_output.toPlainText()
    finally:
        close(w)


def test_the_running_row_and_the_view_sentence(qapp, ws):
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        tick(w, qapp, "P1.tif")
        c._begin_run("predict", [{"stack_name": "P1.tif"}], 36)
        c._on_stack_started("P1.tif", 0, 1)
        c._on_seg_slice_progress("P1.tif", "xz", 5, 24)
        pump(qapp)
        row = next(r for r in c._rows if r["name"] == "P1.tif")
        assert row["word"] == TH.ROW_SEGMENTING
        text = w.run_status_label.toolTip()
        assert text.startswith("Segmenting stack 1/1: P1.tif — XZ (5/24)  [")
    finally:
        c._running = {}
        close(w)


def test_an_old_npy_record_is_not_shown(qapp, ws):
    """F12: ``.npy`` cannot run; an old record stays in the file, unshown."""
    w = panel(qapp, ws)
    try:
        c = w.controller
        c.state_manager.register_stacks(["old.npy"])
        c._refresh_stacks()
        pump(qapp)
        assert "old.npy" not in [r["name"] for r in c._rows]
        assert c.state_manager.get_record("old.npy") is not None
    finally:
        close(w)


def test_clear_2d_predictions_returns_to_unprocessed(qapp, ws, monkeypatch):
    """F9: a red confirm, the cache deleted, back to unprocessed."""
    import napari_epicycle.ui_shell.theme as th
    asked = []
    monkeypatch.setattr(th, "themed_confirm",
                        lambda *a, **k: asked.append(k.get("accent")) or True)
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        snap = c._model_snapshot()
        c._on_prediction_completed("P1.tif", {
            "views": _fake_views(), "model_snapshot": snap,
            "source_stack_path": str(ws / "artifacts/step1/patches/P1.tif"),
            "cache_meta": {"model": snap, "axes": "CZYX"}})
        assert c.state_manager.get_stack_status("P1.tif") == "predicted"
        c._clear_predictions("P1.tif")
        pump(qapp)
        assert asked == ["red"]
        assert c.state_manager.get_stack_status("P1.tif") == "unprocessed"
        assert "predicted_with" not in c.state_manager.get_record("P1.tif")
        assert not M.find_prediction_cache(c._prediction_cache_dir(),
                                           "P1.tif")["complete"]
    finally:
        close(w)


def test_cancel_says_step3s_sentence_then_the_true_count(qapp, ws):
    """F10: the cancelling line, then the done line with the true count."""
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)

        class _W:
            def request_cancel(self):
                self.cancelled = True

        c._seg_worker = _W()
        c._begin_run("predict", [{"stack_name": n} for n in
                                 ("P1.tif", "P2.tif", "P3.tif")], 30)
        c._cancel_run()
        assert w.run_status_label.toolTip() == TH.PIPE_CANCELLING.format(
            unit="stack")
        c._run_done = 1
        c._on_all_stacks_completed()
        assert w.run_status_label.toolTip().startswith(
            TH.PIPE_SEGMENTATION_DONE.format(s=1, n=3, units="stacks"))
    finally:
        c._seg_worker = None
        close(w)


def test_the_first_tick_after_an_adopt_is_kept_and_saved(qapp, ws):
    """A tick is a user action that saves: after a read-only adopt it
    loads the state in write mode, registers the stacks and keeps the tick."""
    from napari_epicycle.segmentation.ui import SegmentStack
    w = SegmentStack(viewer=None)
    try:
        c = w.controller
        c.adopt_workspace(str(ws))
        pump(qapp)
        assert c.state_manager.get_all_stack_statuses() == {}
        tick(w, qapp, "P2.tif")
        assert c.state_manager.get_ticked() == ["P2.tif"]
        assert set(c.state_manager.get_all_stack_statuses()) == {
            "P1.tif", "P2.tif", "P3.tif"}
        assert (ws / "step4_state.json").exists()     # the root (dec. 55)
        assert "P2.tif" in w.ticked_names()
    finally:
        close(w)


def test_add_a_folder_lists_it_and_remembers_it(qapp, ws, tmp_path,
                                                monkeypatch):
    """Add a folder… lists the folder's .tif/.tiff; session.stack_folders
    keeps it, and a restore lists it again."""
    import napari_epicycle.segmentation.controller as C

    from ._step4_panel import write_stack
    extra = tmp_path / "more"
    write_stack(extra / "F1.tif")
    monkeypatch.setattr(C.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(extra)))
    w = panel(qapp, ws)
    try:
        w.add_folder_btn.click()
        pump(qapp)
        c = w.controller
        assert "F1.tif" in [r["name"] for r in c._rows]
        assert c.state_manager.get_session_metadata()["stack_folders"] == [
            str(extra)]
        assert c.state_manager.get_record("F1.tif")["origin"] == "folder"
    finally:
        close(w)
    w = panel(qapp, ws)
    try:
        assert "F1.tif" in [r["name"] for r in w.controller._rows]
    finally:
        close(w)

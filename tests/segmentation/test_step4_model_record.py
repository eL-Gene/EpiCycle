"""F7 / F24 / I-6: the model on the record, the cache meta, the cache names.

F24.  Qt-free (the workers' ``run`` is
called directly; their signals are plain Qt signals with direct slots).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import controller as C

from ._step4_fakes import controller


def _views(shape=(2, 3, 3)):
    flows = np.zeros((3,) + shape, np.float32)
    probs = np.zeros(shape, np.float32)
    masks = np.zeros(shape, np.int32)
    return {v: (flows, probs, masks) for v in ("xy", "xz", "yz")}


SNAP = {"lateral": "D:/m/lat.pt", "axial": "D:/m/ax.pt",
        "lateral_label": "lat.pt", "axial_label": "ax.pt",
        "kind": "external", "run_id": None, "bundle": None,
        "lateral_rel": None, "axial_rel": None}


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    src = w / "stack.tif"
    src.write_bytes(b"12345")
    return w


# ------------------------------------------------------------- Predict


def test_the_worker_carries_the_launch_time_snapshot(monkeypatch, tmp_path):
    import packages.epicycle_reconstruction.logic as logic
    import tifffile

    monkeypatch.setattr(logic, "segment_views",
                        lambda *a, **k: _views())
    monkeypatch.setattr(tifffile, "imread", lambda p: np.zeros((2, 3, 3)))
    cfg = {"stack_name": "s.tif", "source_stack_path": "x.tif",
           "lateral_model_path": "L", "axial_model_path": "A",
           "model_snapshot": dict(SNAP), "cache_meta": {"axes": "ZYX"}}
    worker = C._StackPredictionWorker([cfg])
    got = []
    worker.stack_completed.connect(lambda n, r: got.append((n, r)))
    cfg["model_snapshot"]["lateral"] = "changed after launch"
    worker.run()
    assert got and got[0][1]["model_snapshot"]["lateral"] == "D:/m/lat.pt"


def test_predict_writes_predicted_with_fingerprint_and_meta(ws):
    ctrl = controller(ws)
    src = str(ws / "stack.tif")
    ctrl._on_prediction_completed("stack.tif", {
        "views": _views(), "source_stack_path": src,
        "model_snapshot": dict(SNAP),
        "cache_meta": {"channels": [1, 0], "axes": "ZYX",
                       "diameter_px": 30.0},
    })
    rec = ctrl.state_manager.get_record("stack.tif")
    assert rec["status"] == "predicted"
    assert rec["predicted_with"] == SNAP                       # I-6
    assert rec["source_fingerprint"] == M.file_fingerprint(src)
    meta = M.read_cache_meta(ctrl._prediction_cache_dir(), "stack.tif")
    assert meta["model"] == SNAP and meta["axes"] == "ZYX"
    assert meta["channels"] == [1, 0]
    found = M.find_prediction_cache(ctrl._prediction_cache_dir(), "stack.tif")
    assert found["kind"] == "name" and found["complete"]


def test_predict_again_keeps_a_3d_status(ws):
    ctrl = controller(ws)
    ctrl.state_manager.set_stack_status("stack.tif", "finalized",
                                        labels_path="L")
    ctrl._on_prediction_completed("stack.tif", {
        "views": _views(), "source_stack_path": str(ws / "stack.tif"),
        "model_snapshot": dict(SNAP), "cache_meta": {}})
    rec = ctrl.state_manager.get_record("stack.tif")
    assert rec["status"] == "finalized"
    assert rec["predicted_with"] == SNAP


def test_the_meta_is_deleted_first_and_written_last(ws, monkeypatch):
    ctrl = controller(ws)
    ctrl._on_prediction_completed("stack.tif", {
        "views": _views(), "source_stack_path": str(ws / "stack.tif"),
        "model_snapshot": dict(SNAP), "cache_meta": {}})
    cache = ctrl._prediction_cache_dir()
    assert M.cache_meta_path(cache, "stack.tif").exists()
    real_save = np.save
    calls = {"n": 0}

    def flaky(path, arr, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_save(path, arr, *a, **k)

    monkeypatch.setattr(C.np, "save", flaky)
    with pytest.raises(OSError):
        ctrl._persist_prediction_cache(_views(), "stack.tif", {"model": SNAP})
    # A half-written cache has no meta, so Reconstruct refuses it.
    assert not M.cache_meta_path(cache, "stack.tif").exists()
    assert M.find_prediction_cache(cache, "stack.tif")["complete"] is False


# ------------------------------------------------------------ Reconstruct


def test_reconstruct_records_made_with_from_the_meta(ws):
    ctrl = controller(ws)
    run = ws / "artifacts" / "step4" / "runs" / "r1"
    run.mkdir(parents=True)
    labels = run / "labels_3d.tif"
    labels.write_bytes(b"x")
    ctrl.state_manager.set_stack_status("stack.tif", "failed",
                                        failed_stage="reconstruct")
    ctrl._on_stack_completed("stack.tif", {
        "labels_path": str(labels), "source_stack_path": str(ws / "stack.tif"),
        "made_with": dict(SNAP), "num_cells_final": 12})
    rec = ctrl.state_manager.get_record("stack.tif")
    assert rec["status"] == "reconstructed"
    assert rec["made_with"] == SNAP                             # I-6
    assert rec["n_cells"] == 12
    assert "failed_stage" not in rec


def test_a_legacy_cache_gives_made_with_null(ws):
    ctrl = controller(ws)
    run = ws / "artifacts" / "step4" / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "labels_3d.tif").write_bytes(b"x")
    ctrl._on_stack_completed("stack.tif", {
        "labels_path": str(run / "labels_3d.tif"),
        "source_stack_path": str(ws / "stack.tif"), "made_with": None})
    assert ctrl.state_manager.get_record("stack.tif")["made_with"] is None


def test_the_worker_never_re_predicts_on_an_incomplete_cache(tmp_path):
    class _Service:
        called = False

        def run_real(self, **kw):
            _Service.called = True
            return {}

    cfg = {"stack_name": "s.tif", "working_dir": str(tmp_path),
           "source_stack_path": "x.tif", "lateral_model_path": "L",
           "axial_model_path": "A", "require_cache": True,
           "cache_files": {"xy": ("a", "b", None)}}
    worker = C._StackSegmentationWorker([cfg], _Service())
    errors, done = [], []
    worker.error.connect(lambda n, m: errors.append((n, m)))
    worker.all_completed.connect(lambda: done.append(True))
    worker.run()
    assert not _Service.called
    assert errors and errors[0][0] == "s.tif"
    assert done == [True]


# ----------------------------------------------------------- cache names


def test_cache_keyed_by_file_name(tmp_path):
    for name in ("a.tif", "a.tiff"):
        for v in M.VIEWS:
            for k in ("flows", "probs"):
                np.save(M.cache_array_path(tmp_path, name, v, k),
                        np.zeros(1))
    (M.cache_meta_path(tmp_path, "a.tif")).write_text("{}", encoding="utf-8")
    a = M.find_prediction_cache(tmp_path, "a.tif")
    b = M.find_prediction_cache(tmp_path, "a.tiff")
    assert a["complete"] and not b["complete"]      # a.tiff has no meta
    assert all("a.tif_" in Path(p[0]).name for p in a["files"].values())


def test_a_legacy_stem_cache_is_still_read(tmp_path):
    for v in M.VIEWS:
        for k in ("flows", "probs"):
            np.save(tmp_path / f"stack_{v}_{k}.npy", np.zeros(1))
    found = M.find_prediction_cache(tmp_path, "stack.tif")
    assert found["kind"] == "legacy" and found["complete"]
    assert found["meta"] is None


def test_clearing_stack_leaves_stack_2(ws):
    ctrl = controller(ws)
    for name in ("stack.tif", "stack_2.tif"):
        ctrl.state_manager.register_stacks([name])
        ctrl._persist_prediction_cache(_views(), name, {"model": SNAP})
    ctrl._clear_prediction_cache_for("stack.tif")
    cache = ctrl._prediction_cache_dir()
    assert M.find_prediction_cache(cache, "stack.tif")["kind"] is None
    assert M.find_prediction_cache(cache, "stack_2.tif")["complete"]


def test_same_model_compares_paths_never_labels():
    a = dict(SNAP)
    b = dict(SNAP, lateral_label="something else")
    assert M.same_model(a, b)
    c = dict(SNAP, lateral="D:/m/other.pt")
    assert not M.same_model(a, c)
    rel_a = dict(SNAP, lateral="C:/x/ws/m/l.pt", lateral_rel="m/l.pt",
                 axial="C:/x/ws/m/a.pt", axial_rel="m/a.pt")
    rel_b = dict(SNAP, lateral="E:/copy/ws/m/l.pt", lateral_rel="m/l.pt",
                 axial="E:/copy/ws/m/a.pt", axial_rel="m/a.pt")
    assert M.same_model(rel_a, rel_b)


def test_model_snapshot_from_the_live_widgets(ws):
    ctrl = controller(ws)
    lat = ws / "models" / "lat.pt"
    lat.parent.mkdir()
    lat.write_bytes(b"x")
    ctrl.ui.model_source_combo.setCurrentIndex(1)
    ctrl.ui.lateral_model_path.setText(str(lat))
    ctrl.ui.axial_model_path.setText(str(lat))
    snap = ctrl._model_snapshot()
    assert snap["kind"] == "external"
    assert snap["lateral_rel"] == "models/lat.pt"
    ctrl.ui.model_source_combo.setCurrentIndex(2)
    snap = ctrl._model_snapshot()
    assert snap["kind"] == "pretrained" and snap["lateral"] == "cyto3"


def test_a_new_predict_leaves_no_old_view_behind(ws):
    """Audit D11: a view the new Predict does not write cannot survive."""
    ctrl = controller(ws)
    ctrl._persist_prediction_cache(_views(), "stack.tif", {"model": SNAP})
    cache = ctrl._prediction_cache_dir()
    partial = {"xy": _views()["xy"]}
    ctrl._persist_prediction_cache(partial, "stack.tif", {"model": SNAP})
    assert not M.cache_array_path(cache, "stack.tif", "xz", "flows").exists()
    assert M.find_prediction_cache(cache, "stack.tif")["complete"] is False

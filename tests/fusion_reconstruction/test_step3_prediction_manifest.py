"""Step 3's prediction manifest (R2c; row
26).

Predict writes ``prediction_cache/{stem}_prediction_manifest.json`` per
patch: every ``SegmentationConfig`` field as Cellpose got it, the axes,
the views, the patch's pixel number, the resize rule, and two empty slots
(``spacing`` for R7, ``model_provenance`` for R6).  It is written in the
worker AFTER the arrays, atomically (mkstemp + fsync + ``os.replace``);
the cache restore never needs it; every path that deletes a patch's cache
arrays deletes it too.

A real panel + controller (``FusionReconstructionCuration(viewer=None)``),
``segment_views`` faked.  Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _step3_manifests as S  # noqa: E402
from packages.epicycle_training.config import SegmentationConfig  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def w(app):
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    widget = FusionReconstructionCuration(viewer=None)
    yield widget
    timer = getattr(widget.controller.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    widget.deleteLater()
    app.processEvents()


def _open(w, ws: Path):
    c = w.controller
    c.working_dir = str(ws)
    c.state_manager.active_run_id = "default"
    c.state_manager.load(str(ws))
    return c


def _views():
    f = np.ones((2, 3, 4, 4), np.float32)
    p = np.ones((3, 4, 4), np.float32)
    return {v: (f, p, None) for v in ("xy", "xz", "yz")}


@pytest.fixture
def fake_segment(monkeypatch):
    import packages.epicycle_reconstruction.logic as logic
    seen = []

    def fake(image, lat, ax, seg_config, axes="ZCYX", progress_cb=None):
        seen.append(seg_config)
        return _views()

    monkeypatch.setattr(logic, "segment_views", fake)
    return seen


def _patch(ws: Path, name="P1.tif"):
    import tifffile
    path = ws / name
    tifffile.imwrite(str(path), np.zeros((4, 8, 8), np.uint8))
    return path


def _seg(**kw):
    base = dict(channels=[2, 1], normalize=True, rescale=None,
                cellprob_threshold=0.0, flow_threshold=0.4, diameter=None,
                batch_size=64, compute_masks=False)
    base.update(kw)
    return SegmentationConfig(**base)


def _predict(c, ws, name="P1.tif", seg=None):
    path = _patch(ws, name)
    return c._predict_single_patch(
        patch_name=name, stack_path=str(path), lateral_path="lat",
        axial_path="ax", seg_config=seg or _seg(), axes="CZYX")


def _manifest_path(c, name="P1.tif"):
    return S.prediction_manifest_path(c._prediction_cache_dir(), name)


# ---------------------------------------------------------------- content

def test_predict_writes_the_manifest(w, tmp_path, fake_segment):
    c = _open(w, tmp_path)
    seg = _seg(diameter=23.0, rescale=True)
    _predict(c, tmp_path, seg=seg)
    path = _manifest_path(c)
    assert path.name == "P1_prediction_manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m["manifest_version"] == 1
    assert m["patch_name"] == "P1.tif"
    assert m["predicted_at"]
    assert m["axes"] == "ZYX"                        # resolved per patch
    assert m["views"] == ["xy", "xz", "yz"]
    assert set(m["seg_config"]) == {
        f.name for f in dataclasses.fields(SegmentationConfig)}
    assert m["seg_config"] == json.loads(json.dumps(dataclasses.asdict(seg)))
    assert m["diameter_px"] == 23.0
    assert m["rescale_rule"] == S.RESCALE_FROM_WIDTH
    assert m["spacing"] is None                      # R7
    assert m["model_provenance"] is None             # R6


@pytest.mark.parametrize("rescale,diameter,rule,factor", [
    (True, None, S.RESCALE_NONE, 1.0),
    (None, None, S.RESCALE_TRAINED, None),
    (None, 30.0, S.RESCALE_FROM_WIDTH, None),
    (True, 30.0, S.RESCALE_FROM_WIDTH, None),
])
def test_the_resize_rule_is_recorded(rescale, diameter, rule, factor):
    m = S.build_prediction_manifest(
        patch_name="P", axes="ZYX",
        seg_config=_seg(rescale=rescale, diameter=diameter))
    assert m["rescale_rule"] == rule
    assert m["rescale_factor"] == factor
    assert m["seg_config"]["rescale"] == rescale     # None included


def test_the_manifest_is_json_safe_with_numpy_values():
    seg = _seg(diameter=np.float32(12.5), batch_size=np.int64(16),
               channels=[np.int64(2), np.int64(1)],
               cellprob_threshold=np.float64(0.5))
    m = S.build_prediction_manifest(patch_name="P", axes="ZYX",
                                    seg_config=seg,
                                    cell_width_um=np.float64(4.0))
    back = json.loads(json.dumps(m))                 # no default= needed
    assert back["seg_config"]["batch_size"] == 16
    assert back["seg_config"]["channels"] == [2, 1]
    assert back["diameter_px"] == 12.5
    assert back["cell_width_um"] == 4.0
    assert type(m["seg_config"]["batch_size"]) is int


def test_the_config_given_is_not_changed(w, tmp_path, fake_segment):
    c = _open(w, tmp_path)
    seg = _seg(diameter=11.0)
    before = dataclasses.asdict(seg)
    _predict(c, tmp_path, seg=seg)
    assert dataclasses.asdict(seg) == before


# ---------------------------------------------------------------- written last, atomically

def test_it_is_written_after_every_array(w, tmp_path, fake_segment,
                                         monkeypatch):
    c = _open(w, tmp_path)
    order = []
    real_save, real_write = np.save, S.write_json_atomic

    def save(path, arr, *a, **kw):
        order.append(("array", Path(path).name))
        return real_save(path, arr, *a, **kw)

    def write(path, data):
        cache = Path(path).parent
        # every array is on disk when the manifest is written
        assert len(list(cache.glob("P1_*_flows.npy"))) == 3
        assert len(list(cache.glob("P1_*_probs.npy"))) == 3
        order.append(("manifest", Path(path).name))
        return real_write(path, data)

    monkeypatch.setattr(np, "save", save)
    monkeypatch.setattr(S, "write_json_atomic", write)
    _predict(c, tmp_path)
    assert order[-1] == ("manifest", "P1_prediction_manifest.json")
    assert sum(1 for k, _ in order if k == "array") == 6


def test_the_write_is_mkstemp_fsync_replace(tmp_path, monkeypatch):
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(S.os, "fsync",
                        lambda fd: (calls.append("fsync"), real_fsync(fd)))
    monkeypatch.setattr(S.os, "replace",
                        lambda a, b: (calls.append(("replace", Path(a).parent
                                                    == Path(b).parent)),
                                      real_replace(a, b)))
    target = tmp_path / "x_prediction_manifest.json"
    S.write_json_atomic(target, {"a": 1})
    assert calls == ["fsync", ("replace", True)]
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


def test_a_failed_write_leaves_the_old_file_and_no_temp(tmp_path,
                                                        monkeypatch):
    target = tmp_path / "x_prediction_manifest.json"
    S.write_json_atomic(target, {"old": True})

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(S.json, "dump", boom)
    with pytest.raises(OSError):
        S.write_json_atomic(target, {"new": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == [target.name]


def test_a_failed_manifest_write_leaves_no_manifest_beside_new_arrays(
        w, tmp_path, monkeypatch):
    """R2c audit LOW-1: the stem's OLD manifest is unlinked before the
    first array is saved, so a persist that dies at the manifest leaves no
    record beside the new arrays -- never the old run's record."""
    c = _open(w, tmp_path)
    c._persist_prediction_cache(_views(), patch_name="P1.tif",
                                manifest={"run": "old"})
    assert _manifest_path(c).exists()

    def boom(path, data):
        raise OSError("disk full")

    monkeypatch.setattr(S, "write_json_atomic", boom)
    with pytest.raises(OSError):
        c._persist_prediction_cache(_views(), patch_name="P1.tif",
                                    manifest={"run": "new"})
    assert not _manifest_path(c).exists()
    assert len(list(c._prediction_cache_dir().glob("P1_*.npy"))) == 6


def test_a_half_done_array_save_leaves_no_manifest(w, tmp_path,
                                                   monkeypatch):
    """The old manifest is gone before the first ``np.save``."""
    c = _open(w, tmp_path)
    c._persist_prediction_cache(_views(), patch_name="P1.tif",
                                manifest={"run": "old"})
    real_save, n = np.save, []

    def save(path, arr, *a, **kw):
        n.append(1)
        if len(n) == 2:
            raise OSError("disk full")
        return real_save(path, arr, *a, **kw)

    monkeypatch.setattr(np, "save", save)
    with pytest.raises(OSError):
        c._persist_prediction_cache(_views(), patch_name="P1.tif",
                                    manifest={"run": "new"})
    assert not _manifest_path(c).exists()


# ---------------------------------------------------------------- the restore

def test_the_restore_does_not_need_it(w, tmp_path, fake_segment):
    c = _open(w, tmp_path)
    _predict(c, tmp_path)
    _manifest_path(c).unlink()
    back = c._try_restore_prediction_cache("P1.tif")
    assert back is not None and set(back) == {"xy", "xz", "yz"}


def test_a_persist_without_a_manifest_writes_none(w, tmp_path):
    """Callers that pass no manifest (the cache layout is unchanged)."""
    c = _open(w, tmp_path)
    c._persist_prediction_cache(_views(), patch_name="P1.tif")
    assert not _manifest_path(c).exists()
    assert len(list(c._prediction_cache_dir().iterdir())) == 6


def test_the_state_scan_does_not_count_it_as_a_prediction(w, tmp_path):
    c = _open(w, tmp_path)
    cache = c._prediction_cache_dir()
    cache.mkdir(parents=True)
    S.write_json_atomic(S.prediction_manifest_path(cache, "P1.tif"), {})
    c.state_manager._state["patches"]["P1.tif"] = {"status": "unprocessed"}
    c.state_manager.scan_existing_artifacts(str(tmp_path))
    assert c.state_manager._state["patches"]["P1.tif"]["status"] == \
        "unprocessed"


# ---------------------------------------------------------------- deleted with the arrays

def _two_patches(c, ws):
    for name in ("A.tif", "B.tif"):
        _predict(c, ws, name)
    cache = c._prediction_cache_dir()
    assert S.prediction_manifest_path(cache, "A.tif").exists()
    assert S.prediction_manifest_path(cache, "B.tif").exists()
    return cache


def test_clearing_a_patchs_cache_deletes_its_manifest(w, tmp_path,
                                                      fake_segment):
    c = _open(w, tmp_path)
    cache = _two_patches(c, tmp_path)
    c._clear_prediction_cache("A.tif")
    assert not any(p.name.startswith("A_") for p in cache.iterdir())
    assert S.prediction_manifest_path(cache, "B.tif").exists()
    assert len(list(cache.glob("B_*.npy"))) == 6


def test_clean_intermediates_deletes_the_manifest_with_the_arrays(
        w, tmp_path, fake_segment):
    from napari_epicycle.fusion_reconstruction._run_management import (
        clean_run_intermediates,
    )
    c = _open(w, tmp_path)
    cache = _two_patches(c, tmp_path)
    c.state_manager._state["patches"] = {
        "A.tif": {"status": "reconstructed"}, "B.tif": {"status": "predicted"}}
    c.state_manager.save()
    clean_run_intermediates(tmp_path, "default", run_path_id=None)
    assert not any(p.name.startswith("A_") for p in cache.iterdir())
    # a merely predicted patch keeps its cache AND its manifest
    assert S.prediction_manifest_path(cache, "B.tif").exists()


def test_every_deleter_of_the_arrays_globs_the_stem():
    """The two deleters of a patch's arrays remove ``{stem}_*`` -- the
    manifest's name is inside that pattern by construction."""
    import inspect

    from napari_epicycle.fusion_reconstruction import _run_management as RM
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    assert 'glob(f"{stem}_*")' in inspect.getsource(C._clear_prediction_cache)
    assert 'glob(f"{stem}_*")' in inspect.getsource(RM.clean_run_intermediates)
    assert S.prediction_manifest_path("c", "P1.tif").name.startswith("P1_")


# ---------------------------------------------------------------- reading it back

def test_read_prediction_manifest_never_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert S.read_prediction_manifest(bad) is None
    assert S.read_prediction_manifest(tmp_path / "missing.json") is None
    other = tmp_path / "other.json"
    other.write_text("[1, 2]", encoding="utf-8")
    assert S.read_prediction_manifest(other) is None


def test_a_config_is_rebuilt_from_its_record():
    seg = _seg(diameter=19.0, rescale=True)
    rec = S.build_prediction_manifest(patch_name="P", axes="ZYX",
                                      seg_config=seg)["seg_config"]
    rec["unknown_future_key"] = 1
    back = S.seg_config_from_record(rec, SegmentationConfig)
    assert back == seg
    assert S.seg_config_from_record(None, SegmentationConfig) is None

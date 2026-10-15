"""The reconstruction manifest records every parameter (R2c).

``packages/epicycle_reconstruction/logic.py``'s ``manifest.json`` becomes
version 2: ``seg_config`` = every ``SegmentationConfig`` field,
``recon_config`` = every ``ReconstructionConfig`` field, plus
``extra_manifest`` keys that never overwrite a core key.  Additive: every
key a version-1 manifest had keeps its name and value, so every reader of
the old shape reads the new one unchanged.  Step 4 calls the same writer,
and its "Step 3 run" preset reads more groups from a v2 manifest.

Qt-free except the ``run_real`` wiring test (fakes u-Segment3D).
"""
from __future__ import annotations

import dataclasses
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from packages.epicycle_reconstruction import logic as L
from packages.epicycle_reconstruction.config import ReconstructionConfig
from packages.epicycle_training.config import SegmentationConfig


# ---------------------------------------------------------------- helpers

def v1_manifest(seg_config, recon_config, **core):
    """The manifest as ``logic.py`` wrote it before R2c (frozen copy)."""
    return {
        "run_id": core["run_id"],
        "generated_at": core["generated_at"],
        "mode": "real",
        "source_stack_path": core["source_stack_path"],
        "labels_path": core["labels_path"],
        "labels_shape": list(core["labels_shape"]),
        "num_cells_raw": core["num_cells_raw"],
        "num_cells_final": core["num_cells_final"],
        "axes": core["axes"],
        "seg_config": {
            "channels": list(seg_config.channels) if seg_config else [0, 0],
        },
        "recon_config": {
            "method": recon_config.method,
            "skip_postprocess": recon_config.skip_postprocess,
            "pre_smooth_probs": recon_config.pre_smooth_probs,
            "pre_smooth_flows": recon_config.pre_smooth_flows,
            "pre_smooth_sigma_perp": recon_config.pre_smooth_sigma_perp,
            "pre_smooth_sigma_inplane": recon_config.pre_smooth_sigma_inplane,
            "flow_mask_threshold": recon_config.flow_mask_threshold,
            "flow_soft_weighting": recon_config.flow_soft_weighting,
            "combine_cell_gradients": recon_config.combine_cell_gradients,
            "postprocess_binary": recon_config.postprocess_binary,
            "gradient_descent": recon_config.gradient_descent,
            "flow_consistency": recon_config.flow_consistency,
        },
    }


CORE = dict(run_id="20261006T120000Z", generated_at="2026-10-06T12:00:00",
            source_stack_path="D:/ws/patches/P1.tif",
            labels_path="D:/ws/artifacts/step3/runs/P1.tif/labels_3d.tif",
            labels_shape=(4, 5, 6), num_cells_raw=7, num_cells_final=6,
            axes="CZYX")


def _seg():
    return SegmentationConfig(channels=[3, 2], normalize=False, rescale=None,
                              cellprob_threshold=0.5, flow_threshold=0.6,
                              diameter=17.0, batch_size=32,
                              compute_masks=False)


def _recon():
    rc = ReconstructionConfig(n_cpu=3, skip_postprocess=False,
                              pre_smooth_probs=True)
    rc.combine_cell_probs["ksize"] = 11
    rc.connected_component["min_area"] = 9
    rc.size_filters["min_size"] = 321
    rc.gradient_descent["n_iter"] = 40
    return rc


def v2(seg=None, recon=None, extra=None):
    return L.build_real_manifest(seg_config=seg if seg is not None else _seg(),
                                 recon_config=recon or _recon(),
                                 extra_manifest=extra, **CORE)


def _contains(big, small, path="") -> list:
    """Every key of *small* is in *big* with the same value (deep)."""
    bad = []
    for k, v in small.items():
        if k not in big:
            bad.append(path + k)
        elif isinstance(v, dict) and isinstance(big[k], dict):
            bad += _contains(big[k], v, path + k + ".")
        elif big[k] != v:
            bad.append(f"{path}{k}: {big[k]!r} != {v!r}")
    return bad


# ---------------------------------------------------------------- the shape

def test_v2_records_its_version():
    assert L.RECON_MANIFEST_VERSION == 2
    assert v2()["manifest_version"] == 2


def test_every_segmentation_field_is_recorded():
    seg = _seg()
    got = v2(seg=seg)["seg_config"]
    assert set(got) == {f.name for f in dataclasses.fields(SegmentationConfig)}
    assert got == json.loads(json.dumps(dataclasses.asdict(seg)))
    assert got["channels"] == [3, 2]
    assert got["rescale"] is None          # the value Cellpose got, None too


@pytest.mark.parametrize("rescale", [True, None, 0.5])
def test_rescale_is_the_value_given(rescale):
    seg = dataclasses.replace(_seg(), rescale=rescale)
    assert v2(seg=seg)["seg_config"]["rescale"] == rescale


def test_every_reconstruction_field_is_recorded():
    rc = _recon()
    got = v2(recon=rc)["recon_config"]
    assert set(got) == {f.name for f in dataclasses.fields(ReconstructionConfig)}
    assert got == json.loads(json.dumps(dataclasses.asdict(rc)))
    for added in ("n_cpu", "save_binary", "save_prob3d", "save_flow3d",
                  "binary_erode", "binary_kernel_size", "binary_fill_holes",
                  "combine_cell_probs", "connected_component", "size_filters",
                  "indirect_method"):
        assert added in got, added
    assert got["n_cpu"] == 3
    assert got["combine_cell_probs"]["ksize"] == 11


def test_every_v1_key_keeps_its_name_and_value():
    seg, rc = _seg(), _recon()
    old = json.loads(json.dumps(v1_manifest(seg, rc, **CORE)))
    new = json.loads(json.dumps(L.build_real_manifest(
        seg_config=seg, recon_config=rc, **CORE)))
    assert _contains(new, old) == []
    assert set(new) - set(old) == {"manifest_version"}
    assert set(new["recon_config"]) > set(old["recon_config"])


def test_numpy_values_are_written_as_plain_json():
    seg = SegmentationConfig(channels=[np.int64(2), np.int64(1)],
                             diameter=np.float32(12.5),
                             batch_size=np.int64(64),
                             cellprob_threshold=np.float64(0.25))
    rc = ReconstructionConfig(n_cpu=np.int64(4))
    rc.gradient_descent["momenta"] = np.float32(0.5)
    rc.gradient_descent["tile_shape"] = np.array([8, 16, 16])
    m = L.build_real_manifest(seg_config=seg, recon_config=rc,
                              extra_manifest={"x": np.float64(1.5)}, **CORE)
    text = json.dumps(m)                   # no default= needed
    back = json.loads(text)
    assert back["seg_config"]["channels"] == [2, 1]
    assert back["seg_config"]["diameter"] == 12.5
    assert back["seg_config"]["batch_size"] == 64
    assert back["recon_config"]["n_cpu"] == 4
    assert back["recon_config"]["gradient_descent"]["tile_shape"] == [8, 16, 16]
    assert back["x"] == 1.5
    assert type(m["recon_config"]["n_cpu"]) is int


def test_path_values_are_written_as_plain_json(tmp_path):
    """R2c audit LOW-3: the pure function's output is JSON-safe, as its
    docstring says, when a caller hands it ``Path`` objects."""
    core = dict(CORE, source_stack_path=tmp_path / "P1.tif",
                labels_path=tmp_path / "labels.tif", run_id="r")
    m = L.build_real_manifest(seg_config=_seg(), recon_config=_recon(),
                              **core)
    back = json.loads(json.dumps(m))       # no default= needed
    assert back["source_stack_path"] == str(tmp_path / "P1.tif")
    assert back["labels_path"] == str(tmp_path / "labels.tif")
    assert type(m["source_stack_path"]) is str


# ---------------------------------------------------------------- extra keys

def test_extra_keys_are_written_as_given():
    extra = {"cell_width_um": 8.5, "notes": {"a": [1, 2]}}
    m = v2(extra=extra)
    assert m["cell_width_um"] == 8.5
    assert m["notes"] == {"a": [1, 2]}


@pytest.mark.parametrize("key", ["run_id", "seg_config", "recon_config",
                                 "labels_path", "manifest_version", "mode",
                                 "source_stack_path"])
def test_an_extra_key_never_overwrites_a_core_key(key, capsys):
    base = v2()
    m = v2(extra={key: "CLOBBER", "kept": 1})
    assert m[key] == base[key]
    assert m["kept"] == 1
    out = capsys.readouterr().out
    assert "[RECON-MANIFEST]" in out and repr(key) in out


def test_no_extra_is_no_extra_key():
    assert set(v2()) == set(v2(extra={}))


# ---------------------------------------------------------------- run_real

@pytest.fixture
def fake_useg3d(monkeypatch):
    """u-Segment3D as a fake: params with the groups run_real updates, an
    aggregation that returns a tiny labelled volume."""
    groups = ("combine_cell_gradients", "combine_cell_probs",
              "postprocess_binary", "gradient_descent", "connected_component",
              "indirect_method")
    params = types.ModuleType("segment3D.parameters")
    params.get_2D_to_3D_aggregation_params = lambda: {g: {} for g in groups}
    params.get_postprocess_segmentation_params = lambda: {
        "flow_consistency": {}, "size_filters": {}}
    useg = types.ModuleType("segment3D.usegment3d")

    def aggregate(probs, gradients, params):
        seg = np.zeros((4, 5, 6), np.int32)
        seg[1:3, 1:3, 1:3] = 1
        return seg, (np.zeros((4, 5, 6), np.float32), None)

    useg.aggregate_2D_to_3D_segmentation_direct_method = aggregate
    useg.postprocess_3D_cell_segmentation = (
        lambda seg, **kw: (seg, None))
    pkg = types.ModuleType("segment3D")
    pkg.parameters, pkg.usegment3d = params, useg
    monkeypatch.setitem(sys.modules, "segment3D", pkg)
    monkeypatch.setitem(sys.modules, "segment3D.parameters", params)
    monkeypatch.setitem(sys.modules, "segment3D.usegment3d", useg)


def _views():
    f = np.zeros((2, 4, 5, 6), np.float32)
    p = np.zeros((4, 5, 6), np.float32)
    return {v: (f, p, None) for v in ("xy", "xz", "yz")}


def test_run_real_writes_v2_with_the_extra_keys(tmp_path, fake_useg3d):
    svc = L.ReconstructionPipelineService()
    out = svc.run_real(
        working_dir=str(tmp_path), source_stack_path=str(tmp_path / "P1.tif"),
        lateral_model_path="lat", axial_model_path="ax",
        seg_config=_seg(), recon_config=_recon(), cached_views=_views(),
        patch_name="P1.tif", reconstructions_root=str(tmp_path / "runs"),
        extra_manifest={"cell_width_um": 9.0, "run_id": "NOPE"})
    m = json.loads(Path(out["manifest_path"]).read_text(encoding="utf-8"))
    assert m["manifest_version"] == 2
    assert m["run_id"] == out["run_id"] != "NOPE"
    assert m["cell_width_um"] == 9.0
    assert m["seg_config"]["batch_size"] == 32
    assert m["recon_config"]["size_filters"]["min_size"] == 321
    assert m["num_cells_final"] == 1


def test_run_real_without_extras_is_still_v2(tmp_path, fake_useg3d):
    """Step 4's call (no ``extra_manifest``) gains the keys too."""
    out = L.ReconstructionPipelineService().run_real(
        working_dir=str(tmp_path), source_stack_path=str(tmp_path / "S.tif"),
        lateral_model_path="lat", axial_model_path="ax",
        cached_views=_views(), output_base=str(tmp_path / "step4"))
    m = json.loads(Path(out["manifest_path"]).read_text(encoding="utf-8"))
    assert m["manifest_version"] == 2
    assert m["recon_config"] == json.loads(json.dumps(
        dataclasses.asdict(ReconstructionConfig())))


# ---------------------------------------------------------------- readers

def _write(ws: Path, manifest, patch="P1.tif"):
    d = ws / "artifacts" / "step3" / "runs" / patch
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d / "manifest.json"


def _both(ws: Path):
    """The same reconstruction written v1 (old shape) and v2."""
    seg, rc = _seg(), _recon()
    core = dict(CORE, source_stack_path=str(ws / "patches" / "P1.tif"))
    return (json.loads(json.dumps(v1_manifest(seg, rc, **core))),
            json.loads(json.dumps(L.build_real_manifest(
                seg_config=seg, recon_config=rc, **core))))


@pytest.mark.parametrize("which", [0, 1])
def test_step3s_last_run_reader(tmp_path, which):
    from napari_epicycle.fusion_reconstruction.controller import (
        _newest_recon_manifest,
    )
    m = _both(tmp_path)[which]
    path = _write(tmp_path, m)
    got = _newest_recon_manifest(tmp_path / "artifacts" / "step3" / "runs")
    assert got["path"] == str(path)
    assert got["seg_config"]["channels"] == [3, 2]
    assert got["recon_config"]["gradient_descent"]["n_iter"] == 40
    assert got["manifest_version"] == (None if which == 0 else 2)


def test_the_shared_preset_reads_v1_keys_the_same_from_v2(tmp_path):
    from napari_epicycle.ui_shell._recon_presets import step3_recon_preset
    old, new = _both(tmp_path)
    p1 = step3_recon_preset(old["recon_config"])
    p2 = step3_recon_preset(new["recon_config"])
    assert {k: p2[k] for k in p1} == p1          # nothing v1 gave changed


def test_step4s_step3_preset_is_more_exact_on_v2(tmp_path):
    """Effects: Step 4's "Step 3 run" preset applies every group a
    manifest records, so a v2 manifest also fills the probability, the
    grouping and the size groups; ``n_cpu`` stays unread by Step 4."""
    from napari_epicycle.segmentation import _step4_models as M
    old, new = _both(tmp_path)
    p1 = M.step3_recon_preset(old["recon_config"])
    p2 = M.step3_recon_preset(new["recon_config"])
    gained = set(p2) - set(p1)
    assert {"ccp_ksize_sb", "cc_min_area_sb", "sf_min_size_sb",
            "sf_max_size_sb", "ccp_prob_thresh_sb"} <= gained
    assert p2["ccp_ksize_sb"] == 11
    assert p2["cc_min_area_sb"] == 9
    assert p2["sf_min_size_sb"] == 321
    assert "n_cpu_sb" not in p2
    assert not any(k.startswith(("cc_", "ccp_", "sf_")) for k in p1)
    # and through Step 4's own reader of the active Step 3 run
    _write(tmp_path, new)
    found = M.newest_step3_manifest(str(tmp_path))
    assert found["channels"] == [3, 2]
    assert M.step3_recon_preset(found["recon_config"]) == p2


@pytest.mark.parametrize("which", [0, 1])
def test_step3s_state_scan_reads_both(tmp_path, which):
    from napari_epicycle.fusion_reconstruction.state_manager import (
        ReconstructionStateManager,
    )
    m = _both(tmp_path)[which]
    _write(tmp_path, m)
    sm = ReconstructionStateManager()
    sm.load(str(tmp_path))
    sm._state["patches"]["P1.tif"] = {"status": "unprocessed"}
    sm.scan_existing_artifacts(str(tmp_path))
    rec = sm._state["patches"]["P1.tif"]
    assert rec["status"] == "reconstructed"
    assert rec["reconstruction_run_id"] == CORE["run_id"]
    timer = getattr(sm, "_save_timer", None)
    if timer is not None:
        timer.stop()


@pytest.mark.parametrize("which", [0, 1])
def test_step4s_run_store_reads_both(tmp_path, which):
    from napari_epicycle.segmentation.io import StackSegmentationRunStore
    m = _both(tmp_path)[which]
    d = tmp_path / "artifacts" / "step4" / "runs" / "r1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    store = StackSegmentationRunStore()
    (path,) = store.list_manifests(str(tmp_path))
    payload = store.load_manifest(path)
    assert Path(payload["source_stack_path"]).name == "P1.tif"
    assert payload["labels_path"] == CORE["labels_path"]
    assert payload["run_id"] == CORE["run_id"]

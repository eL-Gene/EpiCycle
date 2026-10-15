"""R6a: the pure half of model provenance.

* the kind table on synthetic bundles -- a Step 2 round model, one started
  from a refined bundle or another run, a Step 3 refined bundle (default and
  named run), a Cellpose built-in name, a stray file, a missing file, an
  unreadable manifest;
* the per-view record and the block's shape;
* unknown keys kept on read and on every rewrite (a foreign key at every
  level);
* the atomic notes write and the backfill;
* ``made_by_text`` / ``short_name`` and ``is_active_model``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from napari_epicycle.fusion_reconstruction import _model_provenance as P
from napari_epicycle.fusion_reconstruction import _step3_pr_text as PT
from napari_epicycle.fusion_reconstruction._step3_manifests import (
    build_prediction_manifest,
)
from packages.epicycle_training.step2_model_manifest import (
    build_step2_model_manifest,
)


# ── fixtures ────────────────────────────────────────────────────────────────


def _write(path: Path, data=b"weights") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (bytes, bytearray)):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def step2_model(ws: Path, run="cells", bucket="lateral", rnd=2,
                channels=(2, 1), **manifest_kw) -> Path:
    name = f"{run}_{bucket}_round_{rnd:02d}"
    folder = ws / "artifacts" / "step2" / "runs" / run / bucket / \
        f"round_{rnd:02d}" / "model"
    w = _write(folder / name)
    _write(folder / f"{name}_training_config.json",
           json.dumps({"hyperparameters": {"channels": list(channels)}}))
    kw = dict(run_id=run, bucket=bucket, round_number=rnd,
              source_workspace=str(ws), weights_filename=name,
              final_loss_train=0.42, n_train_pairs=37, n_epochs=100,
              channels_used=["membrane", "nuclei"],
              env_cellpose_version="3.1.1.2")
    kw.update(manifest_kw)
    _write(folder / "manifest.json",
           json.dumps(build_step2_model_manifest(**kw)))
    return w


def refined_bundle(ws: Path, name="mybundle", run=None) -> Path:
    from napari_epicycle.fusion_reconstruction._refinement_bundle import (
        write_or_merge_bundle_manifest,
    )
    base = ws / "artifacts" / "step3"
    if run:
        base = base / "runs" / run
    folder = base / "refinement" / "models" / name
    for b in ("lateral", "axial"):
        _write(folder / b)
        _write(folder / f"{b}_training_config.json",
               json.dumps({"hyperparameters": {"channels": [0, 0]}}))
        write_or_merge_bundle_manifest(
            folder, name, b, init_mode="step2_run", init_path="x",
            final_train_loss=0.3, n_train_pairs=12, n_epochs=50)
    return folder / "lateral"


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    return w


# ── the kind table ──────────────────────────────────────────────────────────


def test_the_kinds_are_the_spec_s_six():
    assert P.MODEL_KINDS == ("pretrained", "bootstrap", "domain_specialized",
                             "refined", "external", "unknown")
    assert set(PT.KIND_WORDS) == set(P.MODEL_KINDS)


def test_a_step2_round_model_is_bootstrap(ws):
    assert P.model_kind(step2_model(ws), workspace=str(ws)) == "bootstrap"


def test_a_cross_bucket_init_stays_bootstrap(ws):
    w = step2_model(ws, bucket="axial", init_mode="cross_bucket",
                    inherited_from_bucket="lateral")
    assert P.model_kind(w, workspace=str(ws)) == "bootstrap"


def test_a_step2_model_from_a_refined_bundle_is_domain_specialized(ws):
    w = step2_model(ws, init_mode="step3_refined",
                    inherited_from_step3_bundle="mybundle",
                    inherited_from_step3_bucket="lateral")
    assert P.model_kind(w, workspace=str(ws)) == "domain_specialized"


def test_a_step2_model_from_another_run_is_domain_specialized(ws):
    w = step2_model(ws, run="second", init_mode="inherited",
                    inherited_from_run="cells",
                    inherited_from_run_bucket="lateral",
                    inherited_from_run_round_id="round_02")
    assert P.model_kind(w, workspace=str(ws)) == "domain_specialized"


def test_a_browsed_init_inside_a_refined_bundle_is_domain_specialized(ws):
    init = refined_bundle(ws)
    w = step2_model(ws, init_mode="browse", init_path=str(init),
                    inherited_from_external_path=str(init),
                    inherited_from_external_label="mine")
    assert P.model_kind(w, workspace=str(ws)) == "domain_specialized"


def test_a_refined_bundle_is_refined(ws):
    assert P.model_kind(refined_bundle(ws), workspace=str(ws)) == "refined"
    named = refined_bundle(ws, name="b2", run="cells")
    assert P.model_kind(named, workspace=str(ws)) == "refined"


@pytest.mark.parametrize("name", ["cyto3", "nuclei", "cyto2", "cyto"])
def test_a_builtin_name_is_pretrained(ws, name):
    assert P.model_kind(name, workspace=str(ws)) == "pretrained"


def test_step3s_pretrained_catalog_is_known():
    from napari_epicycle.fusion_reconstruction.refinement_registry_dialog \
        import PRETRAINED_MODELS
    for name, _desc in PRETRAINED_MODELS:
        assert name in P.CELLPOSE_BUILTIN_NAMES


def test_a_stray_file_is_external(ws, tmp_path):
    outside = _write(tmp_path / "elsewhere" / "my_model.pth")
    inside = _write(ws / "models_i_found" / "my_model")
    assert P.model_kind(outside, workspace=str(ws)) == "external"
    assert P.model_kind(inside, workspace=str(ws)) == "external"


@pytest.mark.parametrize("path", ["", None, "notabuiltin"])
def test_nothing_or_an_unknown_name_is_unknown(ws, path):
    assert P.model_kind(path, workspace=str(ws)) == "unknown"


def test_a_missing_file_is_unknown(ws):
    assert P.model_kind(ws / "gone" / "model", workspace=str(ws)) == "unknown"
    rec = P.view_record(ws / "gone" / "model", workspace=str(ws))
    assert rec["kind"] == "unknown"
    assert rec["extra"]["error"] == PT.ERR_MODEL_MISSING


def test_an_unreadable_manifest_decides_from_the_path(ws):
    w = step2_model(ws)
    (w.parent / "manifest.json").write_text("{not json", encoding="utf-8")
    rec = P.view_record(w, workspace=str(ws))
    assert rec["kind"] == "bootstrap"
    assert rec["extra"]["error"] == PT.ERR_MANIFEST_UNREADABLE
    assert rec["run_id"] == "cells" and rec["round"] == 2


def test_a_legacy_step2_round_folder_is_bootstrap(ws):
    w = _write(ws / "models" / "round_03" / "lateral_model")
    rec = P.view_record(w, workspace=str(ws))
    assert rec["kind"] == "bootstrap" and rec["round"] == 3


def test_the_kind_never_raises(ws):
    for weird in (object(), 12, b"\x00", ws / ("x" * 400)):
        assert P.model_kind(weird, workspace=str(ws)) in P.MODEL_KINDS
        assert P.view_record(weird, workspace=str(ws))["kind"] in \
            P.MODEL_KINDS


# ── the view record ─────────────────────────────────────────────────────────

VIEW_KEYS = ["backend", "kind", "name", "path", "run_id", "round",
             "fingerprint", "trained_channels", "extra"]


def test_a_step2_view_record(ws):
    w = step2_model(ws)
    rec = P.view_record(w, workspace=str(ws))
    assert list(rec) == VIEW_KEYS
    assert rec["backend"] == "cellpose"
    assert rec["path"] == \
        "artifacts/step2/runs/cells/lateral/round_02/model/" \
        "cells_lateral_round_02"
    assert rec["name"] == "cells · lateral · round_02"
    assert rec["run_id"] == "cells" and rec["round"] == 2
    assert rec["fingerprint"] == {"size": 7,
                                  "mtime_ns": os.stat(w).st_mtime_ns}
    assert rec["trained_channels"] == [2, 1]
    ex = rec["extra"]
    assert ex["run_id"] == "cells" and ex["bucket"] == "lateral"
    assert ex["round_number"] == 2
    assert ex["init"]["mode"] == "scratch"
    assert ex["training"] == {"n_train_pairs": 37, "n_epochs": 100,
                              "channels_used": ["membrane", "nuclei"]}
    assert ex["final_loss"]["train"] == 0.42
    assert ex["environment"]["cellpose_version"] == "3.1.1.2"
    assert ex["model_manifest"] == \
        "artifacts/step2/runs/cells/lateral/round_02/model/manifest.json"
    assert "error" not in ex
    json.dumps(rec)


def test_a_domain_specialized_record_names_its_init(ws):
    w = step2_model(ws, init_mode="step3_refined",
                    inherited_from_step3_bundle="mybundle",
                    inherited_from_step3_bucket="lateral")
    ex = P.view_record(w, workspace=str(ws))["extra"]
    assert ex["init"]["mode"] == "step3_refined"
    assert ex["inherited_from"] == {"step3_bundle": "mybundle",
                                    "step3_bucket": "lateral"}


def test_a_refined_view_record(ws):
    w = refined_bundle(ws, name="b2", run="cells")
    rec = P.view_record(w, workspace=str(ws))
    assert rec["kind"] == "refined" and rec["name"] == "b2"
    assert rec["run_id"] == "cells" and rec["round"] is None
    assert rec["trained_channels"] == [0, 0]
    ex = rec["extra"]
    assert ex["bundle"] == "b2" and ex["bucket"] == "lateral"
    assert ex["init"]["mode"] == "step2_run"
    assert ex["training"]["n_train_pairs"] == 12
    assert ex["final_loss"]["train"] == 0.3
    assert ex["model_manifest"].endswith("refinement/models/b2/manifest.json")
    assert P.view_record(refined_bundle(ws), workspace=str(ws))["run_id"] \
        == "default"


def test_a_pretrained_view_record(ws):
    rec = P.view_record("cyto3", workspace=str(ws))
    assert rec == {"backend": "cellpose", "kind": "pretrained",
                   "name": "cyto3", "path": "cyto3", "run_id": None,
                   "round": None, "fingerprint": None,
                   "trained_channels": None, "extra": {}}


def test_an_outside_path_stays_absolute(ws, tmp_path):
    w = _write(tmp_path / "elsewhere" / "m.pth")
    rec = P.view_record(w, workspace=str(ws))
    assert rec["kind"] == "external" and rec["name"] == "m.pth"
    assert Path(rec["path"]) == w


def test_the_backend_is_a_keyword(ws):
    assert P.view_record("cyto3", workspace=str(ws),
                         backend="omnipose")["backend"] == "omnipose"


# ── the block ───────────────────────────────────────────────────────────────

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


def test_the_block_shape(ws):
    lat = step2_model(ws)
    ax = step2_model(ws, bucket="axial")
    b = P.provenance_block(lat, ax, workspace=str(ws), now=NOW)
    assert list(b) == ["schema", "recorded_at", "views", "notes"]
    assert b["schema"] == 1 == P.PROVENANCE_SCHEMA
    assert b["recorded_at"] == NOW.isoformat()
    assert list(b["views"]) == ["lateral", "axial"]
    assert b["views"]["axial"]["extra"]["bucket"] == "axial"
    assert b["notes"] == ""
    json.dumps(b)


def test_an_empty_axial_takes_the_lateral(ws):
    b = P.provenance_block("cyto3", "", workspace=str(ws), now=NOW)
    assert b["views"]["axial"]["name"] == "cyto3"


def test_now_defaults_to_the_clock(ws):
    b = P.provenance_block("cyto3", "cyto3", workspace=str(ws))
    assert datetime.fromisoformat(b["recorded_at"]).tzinfo is not None


# ── unknown keys kept ───────────────────────────────────────────────────────


def _foreign_manifest(ws):
    m = build_prediction_manifest(patch_name="p01.tif", axes="ZCYX",
                                  seg_config=None)
    block = P.provenance_block("cyto3", "cyto3", workspace=str(ws), now=NOW)
    block["from_the_future"] = {"x": 1}
    block["views"]["lateral"]["omnipose_flow"] = 0.4
    block["views"]["lateral"]["extra"]["deep"] = {"k": [1, 2]}
    block["views"]["volumetric"] = {"kind": "refined", "name": "3d"}
    m["model_provenance"] = block
    m["a_key_from_later"] = {"nested": True}
    return m


def test_read_block_keeps_every_key(ws):
    m = _foreign_manifest(ws)
    got = P.read_block(m)
    assert got == m["model_provenance"]
    assert got is not m["model_provenance"]          # a copy
    got["views"]["lateral"]["extra"]["deep"]["k"].append(3)
    assert m["model_provenance"]["views"]["lateral"]["extra"]["deep"] == \
        {"k": [1, 2]}


@pytest.mark.parametrize("bad", [None, {}, {"model_provenance": None},
                                 {"model_provenance": "x"}, [], "str"])
def test_read_block_of_nothing_is_none(bad):
    assert P.read_block(bad) is None


def test_write_notes_keeps_every_other_key_and_is_atomic(ws, monkeypatch):
    path = ws / "cache" / "p01_prediction_manifest.json"
    m = _foreign_manifest(ws)
    _write(path, json.dumps(m))
    calls = []
    from napari_epicycle.fusion_reconstruction import _step3_manifests as SM
    real = SM.write_json_atomic

    def spy(p, data):
        calls.append(Path(p))
        real(p, data)

    monkeypatch.setattr(SM, "write_json_atomic", spy)
    ok, problem = P.write_notes(path, "imaged on the old scope")
    assert ok and problem is None
    assert calls == [path]
    got = json.loads(path.read_text(encoding="utf-8"))
    expect = json.loads(json.dumps(m))
    expect["model_provenance"]["notes"] = "imaged on the old scope"
    assert got == expect
    assert not [p for p in path.parent.iterdir() if p.suffix == ".tmp"]


def test_write_notes_refuses_without_a_block(ws):
    path = _write(ws / "m.json", json.dumps({"model_provenance": None,
                                              "keep": 1}))
    ok, problem = P.write_notes(path, "x")
    assert not ok and problem
    assert json.loads(path.read_text(encoding="utf-8")) == \
        {"model_provenance": None, "keep": 1}


@pytest.mark.parametrize("content", [None, "{broken", "[1, 2]"])
def test_write_notes_never_raises_and_never_clobbers(ws, content):
    path = ws / "m.json"
    if content is not None:
        _write(path, content)
    ok, problem = P.write_notes(path, "x")
    assert not ok and isinstance(problem, str) and problem
    if content is not None:
        assert path.read_text(encoding="utf-8") == content


def test_a_reconstruction_manifest_copy_takes_notes_too(ws):
    """Reconstruct carries the block as ``extra_manifest`` (R2c): the same
    write works on the reconstruction manifest's copy."""
    recon = {"manifest_version": 2, "seg_config": {}, "recon_config": {},
             "model_provenance": P.provenance_block(
                 "cyto3", "cyto3", workspace=str(ws), now=NOW),
             "prediction_manifest": "prediction_cache/p01_prediction_"
                                    "manifest.json"}
    path = _write(ws / "runs" / "p01" / "manifest.json", json.dumps(recon))
    assert P.write_notes(path, "n") == (True, None)
    got = json.loads(path.read_text(encoding="utf-8"))
    assert got["model_provenance"]["notes"] == "n"
    assert got["prediction_manifest"] == recon["prediction_manifest"]


def _recon_pair(ws):
    """Two reconstruction manifests of the default run: A older, B newer,
    each with its own recon_config and a provenance block."""
    root = ws / "artifacts" / "step3" / "runs"
    paths = {}
    for name, iters, age in (("A.tif", 11, 2000), ("B.tif", 22, 1000)):
        p = _write(root / name / "manifest.json", json.dumps({
            "manifest_version": 2, "seg_config": {"diameter": None},
            "recon_config": {"gradient_descent": {"n_iter": iters}},
            "model_provenance": P.provenance_block(
                "cyto3", "cyto3", workspace=str(ws), now=NOW)}))
        t = 1_700_000_000 - age
        os.utime(p, (t, t))
        paths[name] = p
    return root, paths


def test_a_note_keeps_the_manifest_mtime(ws):
    """B MED-1: a note is not a new reconstruction -- the manifest's mtime
    is put back after the atomic replace, so 'the newest reconstruction'
    never jumps to the patch whose note was edited."""
    root, paths = _recon_pair(ws)
    before = os.stat(paths["A.tif"]).st_mtime_ns
    assert P.write_notes(paths["A.tif"], "edited later") == (True, None)
    assert os.stat(paths["A.tif"]).st_mtime_ns == before
    got = json.loads(paths["A.tif"].read_text(encoding="utf-8"))
    assert got["model_provenance"]["notes"] == "edited later"


def test_a_note_on_the_older_patch_leaves_the_newest_reader_alone(ws):
    root, paths = _recon_pair(ws)
    assert P.write_notes(paths["A.tif"], "n") == (True, None)
    from napari_epicycle.fusion_reconstruction.controller import (
        _newest_recon_manifest,
    )
    newest = _newest_recon_manifest(root)
    assert Path(newest["path"]) == paths["B.tif"]
    from napari_epicycle.segmentation._step4_models import (
        newest_step3_manifest,
    )
    s4 = newest_step3_manifest(str(ws))
    assert s4["recon_config"]["gradient_descent"]["n_iter"] == 22


def test_a_backfill_into_an_existing_manifest_keeps_its_mtime(ws):
    path = ws / "artifacts" / "step3" / "runs" / "C.tif" / "manifest.json"
    _write(path, json.dumps({"manifest_version": 2, "recon_config": {},
                             "model_provenance": None}))
    os.utime(path, (1_600_000_000, 1_600_000_000))
    before = os.stat(path).st_mtime_ns
    assert P.backfill_manifest(path, P.unknown_block(now=NOW)) == (True, None)
    assert os.stat(path).st_mtime_ns == before
    assert P.read_block(json.loads(path.read_text(encoding="utf-8")))


# ── backfill ────────────────────────────────────────────────────────────────


def test_backfill_writes_a_minimal_manifest(ws):
    path = ws / "cache" / "old_prediction_manifest.json"
    block = P.unknown_block(now=NOW)
    assert P.backfill_manifest(path, block) == (True, None)
    got = json.loads(path.read_text(encoding="utf-8"))
    assert got == {"manifest_version": 1, "backfilled": True,
                   "model_provenance": block}
    assert P.read_block(got) == block
    assert P.made_by_text(block) == PT.MADE_BY_UNKNOWN
    assert P.write_notes(path, "later") == (True, None)


def test_backfill_into_an_r2c_manifest_keeps_its_keys(ws):
    """A patch predicted between R2c and R6 has a manifest whose block is
    ``null``: the block is added, nothing else changes."""
    path = ws / "cache" / "p_prediction_manifest.json"
    m = build_prediction_manifest(patch_name="p.tif", axes="ZYX",
                                  seg_config=None)
    m["foreign"] = [1]
    _write(path, json.dumps(m))
    block = P.unknown_block(now=NOW)
    assert P.backfill_manifest(path, block) == (True, None)
    got = json.loads(path.read_text(encoding="utf-8"))
    assert got["model_provenance"] == block
    assert got[P.BACKFILLED_BLOCK_KEY] is True
    for k, v in m.items():
        if k != "model_provenance":
            assert got[k] == v


def test_backfill_never_overwrites_a_recorded_block(ws):
    path = ws / "m.json"
    m = _foreign_manifest(ws)
    _write(path, json.dumps(m))
    ok, problem = P.backfill_manifest(path, P.unknown_block(now=NOW))
    assert not ok and problem
    assert json.loads(path.read_text(encoding="utf-8")) == \
        json.loads(json.dumps(m))


def test_backfill_never_clobbers_an_unreadable_file(ws):
    path = _write(ws / "m.json", "{broken")
    ok, problem = P.backfill_manifest(path, P.unknown_block(now=NOW))
    assert not ok and problem
    assert path.read_text(encoding="utf-8") == "{broken"


# ── the chip words ──────────────────────────────────────────────────────────


def test_made_by_text(ws):
    b = P.provenance_block(step2_model(ws), "", workspace=str(ws), now=NOW)
    assert P.short_name(b) == "cells · lateral · round_02"
    assert P.made_by_text(b) == "made by cells · lateral · round_02"
    assert P.made_by_text(None) == PT.MADE_BY_UNKNOWN
    assert P.made_by_text({"views": {}}) == PT.MADE_BY_UNKNOWN
    assert P.short_name(None) == ""


def test_a_long_name_is_elided(ws):
    b = P.provenance_block("cyto3", "cyto3", workspace=str(ws), now=NOW)
    b["views"]["lateral"]["name"] = "a" * 60
    s = P.short_name(b)
    assert len(s) == PT.SHORT_NAME_MAX and s.endswith(PT.ELLIPSIS)
    assert P.made_by_text(b) == PT.MADE_BY.format(short=s)


def test_pick_block_prefers_the_reconstruction_copy(ws):
    a = P.provenance_block("cyto3", "", workspace=str(ws), now=NOW)
    b = P.provenance_block("nuclei", "", workspace=str(ws), now=NOW)
    assert P.pick_block({"model_provenance": a}, {"model_provenance": b}) \
        == a
    assert P.pick_block({"model_provenance": None},
                        {"model_provenance": b}) == b
    assert P.pick_block(None, None) is None


# ── is the active model ─────────────────────────────────────────────────────


def test_is_active_model_by_relative_path(ws, tmp_path):
    lat, ax = step2_model(ws), step2_model(ws, bucket="axial")
    b = P.provenance_block(lat, ax, workspace=str(ws), now=NOW)
    assert P.is_active_model(b, str(lat), str(ax), str(ws))
    assert not P.is_active_model(b, str(ax), str(ax), str(ws))
    assert not P.is_active_model(None, str(lat), str(ax), str(ws))
    # a copied workspace compares equal (relative paths)
    import shutil
    ws2 = tmp_path / "copy"
    shutil.copytree(ws, ws2)
    lat2 = ws2 / lat.relative_to(ws)
    ax2 = ws2 / ax.relative_to(ws)
    assert P.is_active_model(b, str(lat2), str(ax2), str(ws2))


def test_is_active_model_outside_and_builtin(ws, tmp_path):
    out = _write(tmp_path / "elsewhere" / "m.pth")
    b = P.provenance_block(out, "", workspace=str(ws), now=NOW)
    assert P.is_active_model(b, str(out), "", str(ws))
    assert P.is_active_model(b, str(out).upper() if os.name == "nt"
                             else str(out), str(out), str(ws))
    c = P.provenance_block("cyto3", "cyto3", workspace=str(ws), now=NOW)
    assert P.is_active_model(c, "cyto3", "cyto3", str(ws))
    assert not P.is_active_model(c, "nuclei", "cyto3", str(ws))


# ── flattening for the window ───────────────────────────────────────────────


def test_flatten_items_is_generic():
    items = P.flatten_items({"a": 1, "b": {"c": None, "d": {"e": [1, 2]}},
                             "f": []})
    assert items == [("a", 1), ("b.c", None), ("b.d.e", [1, 2]), ("f", [])]
    assert P.display_value(None, empty="—") == "—"
    assert P.display_value([1, "x"], empty="—") == "1, x"
    assert P.display_value([], empty="—") == "—"
    assert P.display_value(True, empty="—") == "true"
    assert P.display_value(0.5, empty="—") == "0.5"


def test_the_module_is_qt_free():
    import ast
    src = Path(P.__file__).read_text(encoding="utf-8")
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert not any(m.startswith(("qtpy", "PyQt", "napari", "superqt"))
                   for m in mods), mods
    assert not any("segmentation" in m for m in mods), mods

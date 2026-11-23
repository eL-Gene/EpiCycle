"""The pairing check inside the QC feature extraction.

The group reads the other run's labels file, pairs the two runs and writes
five match columns into the feature table; without another run nothing
in the extraction changes.  Refusals (another stack file, another shape, an
unreadable file) leave the table without match columns and say why in the
log and the summary.  The provenance comes back from the subprocess in a
sidecar and is recorded in the extraction-context file.

Synthetic volumes only; no QApplication.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_epicycle.fusion_reconstruction import _qc_pairing_text as T
from napari_epicycle.fusion_reconstruction import qc_pairing_group as G
from napari_epicycle.fusion_reconstruction._qc_extract_core import (
    run_full_qc_extraction,
)
from napari_epicycle.fusion_reconstruction.qc_features import (
    load_patch_csv,
    save_patch_csv,
)
from napari_epicycle.segmentation import feature_registry as FR

SHAPE = (24, 40, 40)
MATCH_COLS = ("match_count", "match_id", "match_overlap",
                "match_shared", "match_minor_ratio")


# --------------------------------------------------------------------------- #
# volumes
# --------------------------------------------------------------------------- #
def _cells():
    """Five 6x6x6 cubes, ids 1..5, one per y band (the subprocess tests'
    layout)."""
    lab = np.zeros(SHAPE, np.int32)
    for i in range(5):
        y0 = i * 8 + 1
        lab[2:8, y0:y0 + 6, 2:8] = i + 1
    return lab


def _nuclei():
    """A 3x3x3 nucleus in cells 1-3 (ids 11-13), none in cell 4, two in
    cell 5 (ids 15 and 16, side by side)."""
    nuc = np.zeros(SHAPE, np.uint16)
    for i in range(3):
        y0 = i * 8 + 1
        nuc[3:6, y0 + 1:y0 + 4, 3:6] = 11 + i
    y0 = 4 * 8 + 1
    nuc[3:6, y0:y0 + 3, 3:6] = 15
    nuc[3:6, y0 + 3:y0 + 6, 3:6] = 16
    return nuc


def _cfg(tmp_path, name="a", **extra):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    lab = d / "labels_3d.tif"
    tifffile.imwrite(str(lab), _cells())
    cfg = dict(
        labels_path=str(lab), source_stack_path="", prob_path="",
        csv_path=str(d / "patch001_features.csv"), patch_name="patch001",
        spacing=None, erosion_iters=2, max_radius=5,
        composite_method="zscore", seg_channel=0, pca_features=None,
        n_pcs=5, compute_umap=False, umap_min_dist=0.10,
        umap_n_neighbors=15, compute_tsne=False, tsne_perplexity=30.0,
        channel_configs=[], exclude_border=False, invert_z=False,
        metric="euclidean", patch_status="reconstructed",
    )
    cfg.update(extra)
    return cfg


def _match_file(tmp_path, arr=None, name="nuclei_labels_3d.tif"):
    p = tmp_path / name
    tifffile.imwrite(str(p), _nuclei() if arr is None else arr)
    return p


def _bundle(path, **kw):
    return G.make_match_bundle(path=str(path), run="nuclei",
                                 run_display_name="Nuclei", **kw)


# --------------------------------------------------------------------------- #
# the group in the registry
# --------------------------------------------------------------------------- #
def test_the_group_is_defined_but_not_in_the_registry():
    assert FR.CROSS_RUN_PAIRING.name == "cross_run_pairing"
    assert FR.CROSS_RUN_PAIRING.requires == frozenset(
        {"labels", "match_labels"})
    assert all(g.name != "cross_run_pairing" for g in FR.FEATURE_REGISTRY)
    assert all(g.name != "cross_run_pairing" for g in FR.build_registry())


def test_the_group_goes_right_after_core_morphology_once():
    reg = [g for g in FR.build_registry() if g.name != "composite"]
    out = FR.with_cross_run_pairing(reg)
    names = [g.name for g in out]
    assert names[0] == "core_morphology"
    assert names[1] == "cross_run_pairing"
    assert names[2:] == [g.name for g in reg[1:]]
    assert [g.name for g in FR.with_cross_run_pairing(out)] == names
    assert [g.name for g in reg] == [g.name for g in reg]  # not changed


def test_run_extraction_skips_the_group_without_match_labels():
    reg = FR.with_cross_run_pairing(
        [g for g in FR.FEATURE_REGISTRY if g.name == "core_morphology"])
    df = FR.run_extraction(reg, {"labels": _cells()})
    assert len(df) == 5
    assert not any(c.startswith("match_") for c in df.columns)


# --------------------------------------------------------------------------- #
# the group body
# --------------------------------------------------------------------------- #
def _core_df():
    reg = [g for g in FR.FEATURE_REGISTRY if g.name == "core_morphology"]
    return FR.run_extraction(reg, {"labels": _cells()})


def test_the_group_writes_five_typed_columns(tmp_path, capsys):
    b = _bundle(_match_file(tmp_path))
    df = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                             match_labels=b)
    for c in G.INT_COLUMNS:
        assert df[c].dtype == np.int64, c
    for c in G.FLOAT_COLUMNS:
        assert df[c].dtype == np.float64, c
    row = df.set_index("label")
    assert row.loc[1, "match_count"] == 1
    assert row.loc[1, "match_id"] == 11
    assert row.loc[1, "match_overlap"] == pytest.approx(1.0)
    assert row.loc[1, "match_shared"] == 1
    assert np.isnan(row.loc[1, "match_minor_ratio"])
    # no nucleus: measured, no link
    assert row.loc[4, "match_count"] == 0
    assert row.loc[4, "match_id"] == 0
    assert row.loc[4, "match_overlap"] == 0.0
    # two nuclei of equal share: the lower id, a minor ratio of 1
    assert row.loc[5, "match_count"] == 2
    assert row.loc[5, "match_id"] == 15
    assert row.loc[5, "match_minor_ratio"] == pytest.approx(1.0)
    out = b[G.OUTCOME_KEY]
    assert out["refused"] is None
    assert out["floor_active_vx"] == 20 and out["floor_other_vx"] == 20
    assert out["link_floor"] == pytest.approx(G.LINK_FLOOR)
    assert out["links"] == 5
    assert out["file"] == "nuclei_labels_3d.tif"
    json.dumps(out)                       # JSON-safe
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("[QC-PAIR]")][-1]
    assert "other=nuclei file=nuclei_labels_3d.tif" in line
    assert "floor_active=20vx floor_other=20vx link=0.2 links=5" in line
    assert line.endswith("ms") and "wall=" in line


def test_the_link_floor_and_factor_are_the_decided_constants():
    assert G.LINK_FLOOR == 0.20
    assert G.SIZE_FACTOR == 0.1
    b = G.make_match_bundle(path="x.tif", run="r")
    assert b["link_floor"] == 0.20 and b["factor"] == 0.1


def test_rows_are_matched_on_the_label_value_with_the_table_as_authority():
    labels = _cells()
    other = _nuclei()
    from napari_epicycle.fusion_reconstruction.qc_pairing import pair_runs
    table = pair_runs(labels, other, link_floor=0.2, floor_active=20,
                      floor_other=20)
    # the table lacks label 2 (dropped), is reordered, and holds a label
    # the volume never had (99)
    df = pd.DataFrame({"label": np.array([5, 99, 1, 3, 4], np.int32)})
    G.write_match_columns(df, table)
    assert list(df["label"]) == [5, 99, 1, 3, 4]
    for c in G.INT_COLUMNS:
        assert df[c].dtype == np.int64, c        # not NaN-poisoned
    got = df.set_index("label")
    assert got.loc[5, "match_id"] == 15
    assert got.loc[1, "match_id"] == 11
    assert got.loc[3, "match_id"] == 13
    assert got.loc[99, "match_count"] == 0
    assert got.loc[99, "match_id"] == 0
    assert len(df) == 5


def test_a_wide_id_matches_the_int32_label_column():
    labels = np.zeros((4, 6, 6), np.uint32)
    labels[1:3, 1:4, 1:4] = 2 ** 31 + 5           # wraps to a negative int32
    other = np.zeros((4, 6, 6), np.uint16)
    other[1:3, 1:4, 1:4] = 7
    from napari_epicycle.fusion_reconstruction.qc_pairing import pair_runs
    table = pair_runs(labels, other, link_floor=0.2, floor_active=1,
                      floor_other=1)
    df = pd.DataFrame({"label": labels.max().astype(np.int64)
                       .astype(np.int32).reshape(1)})
    G.write_match_columns(df, table)
    assert int(df.loc[0, "match_id"]) == 7


def test_a_shape_refusal_leaves_the_table_without_match_columns(tmp_path):
    p = _match_file(tmp_path, np.zeros((24, 40, 39), np.uint16))
    b = _bundle(p)
    df = _core_df()
    before = list(df.columns)
    out = G.cross_run_pairing(labels=_cells(), df=df, match_labels=b)
    assert out is df and list(out.columns) == before
    assert b[G.OUTCOME_KEY]["refused"] == (
        "The match columns were skipped: run 'Nuclei' has labels of "
        "24 x 40 x 39 voxels for this patch, and this run has 24 x 40 x 40.")


def test_the_manifest_shape_refuses_before_any_read(tmp_path):
    b = _bundle(tmp_path / "missing.tif", other_labels_shape=[24, 40, 41])
    out = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                              match_labels=b)
    assert not any(c.startswith("match_") for c in out.columns)
    assert b[G.OUTCOME_KEY]["refused"] == T.REFUSE_SHAPE.format(
        other="Nuclei", a="24 x 40 x 41", b="24 x 40 x 40")


def test_the_manifest_shape_is_compared_on_its_last_three_axes(tmp_path):
    # a manifest that records a channel axis in front still matches
    b = _bundle(_match_file(tmp_path), other_labels_shape=[1, *SHAPE])
    out = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                              match_labels=b)
    assert b[G.OUTCOME_KEY]["refused"] is None
    assert "match_count" in out.columns


def test_a_different_stack_file_refuses(tmp_path):
    b = _bundle(_match_file(tmp_path), active_stack_name="wound_1.tif",
                other_stack_name="wound_2.tif")
    out = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                              match_labels=b)
    assert not any(c.startswith("match_") for c in out.columns)
    assert b[G.OUTCOME_KEY]["refused"] == (
        "The match columns were skipped: run 'Nuclei' was made from a "
        "different stack file (wound_2.tif, this run: wound_1.tif).")


def test_stack_names_differing_only_in_case_are_the_same_file(tmp_path):
    b = _bundle(_match_file(tmp_path), active_stack_name="Wound_1.TIF",
                other_stack_name="wound_1.tif")
    out = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                              match_labels=b)
    assert b[G.OUTCOME_KEY]["refused"] is None
    assert "match_count" in out.columns


def test_a_failing_comparison_refuses_with_its_reason(tmp_path,
                                                      monkeypatch, capsys):
    def boom(active, other, **kw):
        raise MemoryError("Unable to allocate 2.1 GiB")

    monkeypatch.setattr(G, "pair_runs", boom)
    b = _bundle(_match_file(tmp_path))
    df = _core_df()
    before = list(df.columns)
    out = G.cross_run_pairing(labels=_cells(), df=df, match_labels=b)
    assert out is df and list(out.columns) == before
    assert b[G.OUTCOME_KEY]["refused"] == (
        "The match columns were skipped: run 'Nuclei' could not be "
        "compared (Unable to allocate 2.1 GiB).")
    assert "comparison failed" in capsys.readouterr().out


def test_a_cancel_inside_the_comparison_is_not_a_refusal(tmp_path,
                                                         monkeypatch):
    from napari_epicycle.fusion_reconstruction.qc_pairing import (
        PairingCancelled,
    )

    def stop(active, other, **kw):
        raise PairingCancelled()

    monkeypatch.setattr(G, "pair_runs", stop)
    b = _bundle(_match_file(tmp_path))
    with pytest.raises(PairingCancelled):
        G.cross_run_pairing(labels=_cells(), df=_core_df(), match_labels=b)
    assert G.OUTCOME_KEY not in b


def test_a_failure_half_way_leaves_no_match_column(tmp_path, monkeypatch):
    def half(df, table):
        df["match_count"] = 0
        raise ValueError("a bad id")

    monkeypatch.setattr(G, "write_match_columns", half)
    b = _bundle(_match_file(tmp_path))
    out = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                              match_labels=b)
    assert not any(c.startswith("match_") for c in out.columns)
    assert "a bad id" in b[G.OUTCOME_KEY]["refused"]


def test_an_unreadable_file_refuses(tmp_path):
    p = tmp_path / "broken.tif"
    p.write_bytes(b"not a tiff")
    b = _bundle(p)
    out = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                              match_labels=b)
    assert not any(c.startswith("match_") for c in out.columns)
    assert b[G.OUTCOME_KEY]["refused"] == T.REFUSE_UNREADABLE.format(
        other="Nuclei")


# --------------------------------------------------------------------------- #
# the whole extraction
# --------------------------------------------------------------------------- #
def test_without_a_match_the_csv_is_byte_identical(tmp_path, monkeypatch):
    seen = []
    real = FR.run_extraction

    def spy(registry, available, **kw):
        seen.append(([g.name for g in registry], sorted(available)))
        return real(registry, available, **kw)

    monkeypatch.setattr(FR, "run_extraction", spy)
    cfg_a = _cfg(tmp_path, "a")
    cfg_b = _cfg(tmp_path, "b", match_labels=None)
    sa = run_full_qc_extraction(**cfg_a)
    sb = run_full_qc_extraction(**cfg_b)
    assert Path(cfg_a["csv_path"]).read_bytes() == \
        Path(cfg_b["csv_path"]).read_bytes()
    assert sa["pairing"] is None and sb["pairing"] is None
    cols = list(load_patch_csv(cfg_a["csv_path"]).columns)
    assert not any(c.startswith("match_") for c in cols)
    today = [g.name for g in FR.build_registry() if g.name != "composite"]
    assert seen[0][0] == today and seen[1][0] == today
    assert "match_labels" not in seen[0][1]


def test_with_a_match_the_csv_carries_the_five_columns(tmp_path):
    p = _match_file(tmp_path)
    logs = []
    cfg = _cfg(tmp_path, "a", match_labels=_bundle(p))
    bundle_in = dict(cfg["match_labels"])
    summary = run_full_qc_extraction(**cfg, log=logs.append)
    assert cfg["match_labels"] == bundle_in     # the caller's dict untouched
    df = load_patch_csv(cfg["csv_path"])
    for c in MATCH_COLS:
        assert c in df.columns, c
    assert list(df.columns).index("match_count") > \
        list(df.columns).index("area")
    for c in G.INT_COLUMNS:
        assert df[c].dtype == np.int64, c
    for c in G.FLOAT_COLUMNS:
        assert df[c].dtype == np.float64, c
    assert int(df.set_index("label").loc[5, "match_count"]) == 2
    pairing = summary["pairing"]
    assert pairing["floor_active_vx"] == 20 and pairing["refused"] is None
    assert not any("skipped" in m for m in logs)


def test_a_refusal_reaches_the_log_and_the_summary(tmp_path):
    p = _match_file(tmp_path, np.zeros((24, 40, 39), np.uint16))
    logs = []
    cfg = _cfg(tmp_path, "a", match_labels=_bundle(p))
    summary = run_full_qc_extraction(**cfg, log=logs.append)
    sentence = T.REFUSE_SHAPE.format(other="Nuclei", a="24 x 40 x 39",
                                     b="24 x 40 x 40")
    assert summary["pairing"]["refused"] == sentence
    assert sentence in logs
    cols = list(load_patch_csv(cfg["csv_path"]).columns)
    assert not any(c.startswith("match_") for c in cols)


def test_the_subprocess_writes_the_pairing_sidecar(tmp_path):
    from napari_epicycle.fusion_reconstruction import (
        _qc_extract_subprocess as sub,
    )
    p = _match_file(tmp_path)
    cfg = _cfg(tmp_path, "a", match_labels=_bundle(p))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    assert sub.main(["prog", str(config_path)]) == 0
    side = sub._pairing_path(str(config_path))
    got = json.loads(side.read_text(encoding="utf-8"))
    assert got["run"] == "nuclei"
    assert got["floor_active_vx"] == 20 and got["floor_other_vx"] == 20
    assert got["refused"] is None


def test_the_subprocess_writes_null_without_a_match(tmp_path):
    from napari_epicycle.fusion_reconstruction import (
        _qc_extract_subprocess as sub,
    )
    cfg = _cfg(tmp_path, "a")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    assert sub.main(["prog", str(config_path)]) == 0
    side = sub._pairing_path(str(config_path))
    assert json.loads(side.read_text(encoding="utf-8")) is None


# --------------------------------------------------------------------------- #
# the CSV round trip
# --------------------------------------------------------------------------- #
def test_the_csv_keeps_int64_and_turns_float_after_a_blank(tmp_path):
    df = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                             match_labels=_bundle(_match_file(tmp_path)))
    p = tmp_path / "x_features.csv"
    save_patch_csv(df, p)
    back = load_patch_csv(p)
    for c in G.INT_COLUMNS:
        assert back[c].dtype == np.int64, c
    mask = back["label"] == 1
    assert G.blank_match_columns(back, mask)
    save_patch_csv(back, p)
    again = load_patch_csv(p)
    assert again["match_count"].dtype == np.float64
    assert again["match_shared"].dtype == np.float64
    assert again["match_id"].dtype == np.int64
    row = again.set_index("label").loc[1]
    assert np.isnan(row["match_count"]) and row["match_id"] == 0
    assert np.isnan(row["match_overlap"])
    assert again.set_index("label").loc[5, "match_count"] == 2.0


def test_match_status_is_never_written(tmp_path):
    df = pd.DataFrame({"label": [1, 2], "area": [3.0, 4.0],
                       "match_status": ["one match", "no match"]})
    p = tmp_path / "x_features.csv"
    save_patch_csv(df, p)
    assert "match_status" not in load_patch_csv(p).columns
    assert "match_status" in df.columns          # the table is not changed


def test_blank_is_a_no_op_without_match_columns():
    df = pd.DataFrame({"label": [1], "area": [2.0]})
    assert not G.blank_match_columns(df, df["label"] == 1)
    assert list(df.columns) == ["label", "area"]


# --------------------------------------------------------------------------- #
# provenance and staleness
# --------------------------------------------------------------------------- #
def test_the_context_fields_of_a_measured_and_a_refused_run():
    prov = {"patch_file_size": 10, "patch_file_mtime": 1.5,
            "match_run": "nuclei", "match_run_display_name": "Nuclei",
            "match_file": "artifacts/step3/x.tif",
            "match_file_mtime": 2.5, "match_file_size": 99,
            "link_floor": 0.2}
    ok = G.pairing_context_fields(prov, {"link_floor": 0.2,
                                         "floor_active_vx": 336,
                                         "floor_other_vx": 40,
                                         "refused": None})
    assert ok == {"patch_file_size": 10, "patch_file_mtime": 1.5,
                  "match_run": "nuclei", "match_run_display_name":
                  "Nuclei", "match_file": "artifacts/step3/x.tif",
                  "match_file_mtime": 2.5, "match_file_size": 99,
                  "link_floor": 0.2, "floor_active_vx": 336,
                  "floor_other_vx": 40}
    no = G.pairing_context_fields(prov, {"refused": "why"})
    assert no["match_run"] == "nuclei" and no["match_skipped"] == "why"
    assert "match_file_mtime" not in no
    alone = G.pairing_context_fields({"patch_file_size": 1,
                                      "patch_file_mtime": 2.0}, None)
    assert alone == {"patch_file_size": 1, "patch_file_mtime": 2.0}


def test_the_sidecar_records_the_pairing_fields(tmp_path):
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        read_extraction_context, stale_reason, write_extraction_context,
    )
    csv = tmp_path / "p_features.csv"
    fields = {"match_run": "nuclei", "floor_active_vx": 12,
              "patch_file_size": 5, "spacing": "must not win"}
    write_extraction_context(csv, spacing=(1.0, 1.0, 1.0),
                             downsample_factor=1, pairing=fields)
    ctx = read_extraction_context(csv)
    assert ctx["match_run"] == "nuclei" and ctx["floor_active_vx"] == 12
    assert ctx["patch_file_size"] == 5
    assert ctx["spacing"] == [1.0, 1.0, 1.0]
    assert stale_reason(ctx, current_spacing=(1.0, 1.0, 1.0),
                        current_downsample=1) is None


def test_match_stale_reason_on_a_touched_file(tmp_path):
    p = tmp_path / "nuc.tif"
    p.write_bytes(b"1234")
    st = G.file_stat(p)
    ctx = {"match_run": "nuclei", "match_run_display_name": "Nuclei",
           "match_file_mtime": st["mtime"], "match_file_size": st["size"]}
    assert G.match_stale_reason(ctx, os.stat(p)) is None
    assert G.match_stale_reason(ctx, G.file_stat(p)) is None
    later = time.time() + 5
    os.utime(p, (later, later))
    assert G.match_stale_reason(ctx, os.stat(p)) == (
        "Run 'Nuclei' has changed since these match values were measured. "
        "Extract again to refresh them.")
    os.utime(p, (st["mtime"], st["mtime"]))
    p.write_bytes(b"12345")
    os.utime(p, (st["mtime"], st["mtime"]))
    assert G.match_stale_reason(ctx, os.stat(p)) is not None   # the size
    assert G.match_stale_reason(ctx, None) is not None         # gone
    assert G.match_stale_reason({"match_run": "n"}, os.stat(p)) is None
    assert G.match_stale_reason(None, os.stat(p)) is None


# --------------------------------------------------------------------------- #
# which run, which file
# --------------------------------------------------------------------------- #
PATCH = "patch_001.tif"


def _workspace(tmp_path, *, nuclei_stack="patch_001.tif",
               nuclei_labels=True, archived=False):
    """Default run (flat, cells) + run ``nuclei`` (named), one patch."""
    from napari_epicycle.fusion_reconstruction import _paths
    ws = tmp_path / "ws"
    step3 = ws / "artifacts" / "step3"
    cells_dir = step3 / "runs" / PATCH
    cells_dir.mkdir(parents=True)
    tifffile.imwrite(str(cells_dir / "labels_3d.tif"), _cells())
    (cells_dir / "manifest.json").write_text(json.dumps({
        "source_stack_path": "C:/elsewhere/patches/patch_001.tif",
        "labels_shape": list(SHAPE)}), encoding="utf-8")
    run_root = step3 / "runs" / "nuclei"
    run_root.mkdir(parents=True)
    (run_root / "run_manifest.json").write_text(json.dumps({
        "run_id": "nuclei", "display_name": "Nuclei DAPI",
        "archived_at": "2026-01-01T00:00:00Z" if archived else None,
        "backend": "cellpose"}), encoding="utf-8")
    nd = run_root / "reconstructions" / PATCH
    nd.mkdir(parents=True)
    if nuclei_labels:
        tifffile.imwrite(str(nd / "labels_3d.tif"), _nuclei())
    else:
        # only a curated file: the check reads reconstructions only
        cur = run_root / "curation" / PATCH
        cur.mkdir(parents=True)
        tifffile.imwrite(str(cur / "labels_curated.tif"), _nuclei())
    (nd / "manifest.json").write_text(json.dumps({
        "source_stack_path": f"D:/copy/ws/patches/{nuclei_stack}",
        "labels_shape": list(SHAPE)}), encoding="utf-8")
    state = {"schema_version": 3, "active_run_id": "default",
             "runs": {"default": {"patches": {PATCH: {"status":
                                                       "reconstructed"}}},
                      "nuclei": {"patches": {PATCH: {"status":
                                                      "reconstructed"}}}}}
    sp = Path(_paths.state_json_path(ws))
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(state), encoding="utf-8")
    return ws


def test_the_other_run_is_listed_with_its_reconstruction_file(tmp_path):
    ws = _workspace(tmp_path)
    runs = G.list_match_runs(ws, PATCH, "default")
    assert [r["run"] for r in runs] == ["nuclei"]
    assert runs[0]["display_name"] == "Nuclei DAPI"
    assert runs[0]["path"].replace("\\", "/").endswith(
        f"runs/nuclei/reconstructions/{PATCH}/labels_3d.tif")
    # from the nuclei run, the default run is the other one
    other = G.list_match_runs(ws, PATCH, "nuclei")
    assert [r["run"] for r in other] == ["default"]


def test_a_run_with_only_a_curated_file_is_not_listed(tmp_path):
    ws = _workspace(tmp_path, nuclei_labels=False)
    assert G.list_match_runs(ws, PATCH, "default") == []


def test_an_archived_run_is_not_listed(tmp_path):
    ws = _workspace(tmp_path, archived=True)
    assert G.list_match_runs(ws, PATCH, "default") == []


def test_resolve_builds_the_bundle_and_the_provenance(tmp_path):
    ws = _workspace(tmp_path)
    bundle, note, prov = G.resolve_match(
        ws, PATCH, "nuclei", active_run_id="default", active_path_id=None)
    assert note is None
    assert bundle["run"] == "nuclei"
    assert bundle["run_display_name"] == "Nuclei DAPI"
    # file names only: a copied workspace keeps its absolute paths
    assert bundle["active_stack_name"] == "patch_001.tif"
    assert bundle["other_stack_name"] == "patch_001.tif"
    assert bundle["other_labels_shape"] == list(SHAPE)
    assert bundle["link_floor"] == 0.20
    json.dumps(bundle)
    assert prov["match_run"] == "nuclei"
    assert prov["match_file"] == (
        f"artifacts/step3/runs/nuclei/reconstructions/{PATCH}/labels_3d.tif")
    st = os.stat(bundle["path"])
    assert prov["match_file_size"] == st.st_size
    assert prov["match_file_mtime"] == pytest.approx(st.st_mtime)
    assert G.match_file_path(prov, ws) == str(ws / prov["match_file"])


def test_resolve_a_run_without_labels_gives_the_log_line(tmp_path):
    ws = _workspace(tmp_path, nuclei_labels=False)
    bundle, note, prov = G.resolve_match(
        ws, PATCH, "nuclei", active_run_id="default", active_path_id=None)
    assert bundle is None and prov is None
    # the run's name, not its slug
    assert note == ("Run 'Nuclei DAPI' cannot be compared for this patch "
                    "(no labels, archived or deleted); match columns "
                    "skipped.")
    gone = G.resolve_match(ws, PATCH, "deleted_run",
                             active_run_id="default", active_path_id=None)
    assert gone[1] == T.NO_LABELS.format(other="deleted_run")
    assert G.resolve_match(ws, PATCH, None, active_run_id="default",
                             active_path_id=None) == (None, None, None)


def test_a_different_stack_name_refuses_through_the_group(tmp_path):
    ws = _workspace(tmp_path, nuclei_stack="patch_002.tif")
    bundle, _, _ = G.resolve_match(
        ws, PATCH, "nuclei", active_run_id="default", active_path_id=None)
    G.cross_run_pairing(labels=_cells(), df=_core_df(), match_labels=bundle)
    assert bundle[G.OUTCOME_KEY]["refused"] == T.REFUSE_STACK.format(
        other="Nuclei DAPI", other_file="patch_002.tif",
        this_file="patch_001.tif")


def test_the_listing_matches_the_reshape_guide_picker(tmp_path):
    from napari_epicycle.annotator.reshape_guide import list_guide_sources
    for kw in ({}, {"archived": True}):
        ws = _workspace(tmp_path / str(len(kw)), **kw)
        for active in ("default", "nuclei"):
            ours = [r["run"] for r in G.list_match_runs(ws, PATCH, active)]
            guide = [g.run_id for g in list_guide_sources(
                ws, PATCH, exclude_run_id=active) if g.kind == "run"]
            assert ours == guide, (kw, active)


def test_the_listing_never_reads_a_run_s_last_activity(tmp_path,
                                                       monkeypatch):
    from napari_epicycle.fusion_reconstruction import _run_management as RM

    def no_scan(*a, **k):
        raise AssertionError("a run's files were scanned")

    monkeypatch.setattr(RM, "_latest_mtime", no_scan)
    monkeypatch.setattr(RM, "list_run_summaries", no_scan)
    ws = _workspace(tmp_path)
    listing = G.read_run_listing(ws)
    assert [r["run"] for r in listing["runs"]] == ["default", "nuclei"]
    assert G.run_display_name(listing, "nuclei") == "Nuclei DAPI"
    bundle, note, _ = G.resolve_match(
        ws, PATCH, "nuclei", active_run_id="default", active_path_id=None,
        listing=listing)
    assert note is None and bundle["run"] == "nuclei"


def test_a_pre_read_listing_is_used_as_given(tmp_path):
    ws = _workspace(tmp_path)
    listing = {"nested": False, "runs": [
        {"run": "default", "display_name": "Default", "archived": False},
        {"run": "nuclei", "display_name": "Nuclei DAPI", "archived": True}]}
    # the listing says archived: not offered, though the manifest says not
    assert G.list_match_runs(ws, PATCH, "default", listing=listing) == []


def test_the_explicit_none_choice_is_recorded():
    out = G.pairing_context_fields(
        {"patch_file_size": 3, "patch_file_mtime": 1.0,
         "match_run": None}, None)
    assert out == {"patch_file_size": 3, "patch_file_mtime": 1.0,
                   "match_run": None}
    never = G.pairing_context_fields(
        {"patch_file_size": 3, "patch_file_mtime": 1.0}, None)
    assert "match_run" not in never


def test_the_log_line_of_a_comparison():
    assert T.pairing_log_line(None) is None
    assert T.pairing_log_line({"display_name": "Nuclei", "links": 7,
                               "refused": None}) == (
        "Compared with run 'Nuclei': 7 links.")
    assert T.pairing_log_line({"refused": "x"}) == "x"


def test_active_label_rows_leaves_ghost_rows_alone():
    df = pd.DataFrame({"label": [1, 1, 2],
                       "label_status": ["active", "superseded", "active"]})
    assert list(G.active_label_rows(df, 1)) == [True, False, False]
    assert list(G.active_label_rows(df, [1, 2])) == [True, False, True]
    plain = pd.DataFrame({"label": [1, 2]})
    assert list(G.active_label_rows(plain, 2)) == [False, True]


def test_values_predate_edits_on_disk_reads_blanked_rows():
    assert not G.values_predate_edits_on_disk(pd.DataFrame({"label": [1]}))
    assert not G.values_predate_edits_on_disk(
        pd.DataFrame({"match_count": [0, 1]}))
    assert G.values_predate_edits_on_disk(
        pd.DataFrame({"match_count": [0, np.nan]}))

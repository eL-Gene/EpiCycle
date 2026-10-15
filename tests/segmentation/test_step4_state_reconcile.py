"""Every load reconciles the state with the disk (live driver, 2026-09-29).

"Reconstruct again" deletes the old run and saves; that save rotates the
PREVIOUS state into the backup, so the backup is always one save behind a
delete.  A recovery "from the last backup" then pointed at a deleted run,
and the real new run was orphaned.  Qt-free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.segmentation import state_manager as SM
from napari_epicycle.segmentation.io import StackSegmentationRunStore

from ._step4_fakes import controller


def _run(ws: Path, run_id: str, stack="s.tif", cells=5) -> Path:
    run = ws / "artifacts" / "step4" / "runs" / run_id
    run.mkdir(parents=True)
    labels = run / "labels_3d.tif"
    labels.write_bytes(b"x")
    (run / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "labels_path": str(labels),
        "source_stack_path": str(ws / "stacks" / stack),
        "num_cells_final": cells}), encoding="utf-8")
    return labels


def _ctrl(ws):
    ctrl = controller(ws)
    ctrl.run_store = StackSegmentationRunStore()
    return ctrl


def test_the_drivers_scenario_recovers_the_new_run(tmp_path, monkeypatch):
    """Delete run -> save -> corrupt -> recover from backup: the record
    points at the NEW run, status reconstructed, no missing paths."""
    import napari_epicycle.segmentation.controller as C
    import napari_epicycle.ui_shell.theme as theme

    monkeypatch.setattr(theme, "themed_notice", lambda *a, **k: None)
    ws = tmp_path / "ws"
    old_labels = _run(ws, "20260929T030000Z")
    csv = ws / "artifacts" / "step4" / "qc" / "s.tif_features.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("label\n1\n", encoding="utf-8")
    ctrl = _ctrl(ws)
    ctrl.state_manager.set_stack_status(
        "s.tif", "qc_extracted", labels_path=str(old_labels),
        qc_features_csv=str(csv), measured_at="t")
    ctrl._persist_state()
    # Reconstruct again: a new run, the old one deleted, then a save
    # (which rotates the qc_extracted record into the backup).
    new_labels = _run(ws, "20260929T033841Z")
    ctrl._on_stack_completed("s.tif", {
        "labels_path": str(new_labels), "source_stack_path": "",
        "made_with": None})
    assert not old_labels.exists() and not csv.exists()
    bak = json.loads(SM.step4_backup_path(str(ws)).read_text(encoding="utf-8"))
    assert bak["stacks"]["s.tif"]["labels_path"] == str(old_labels)
    # The state file is corrupted; a Browse recovers from the backup.
    SM.step4_state_path(str(ws)).write_text("garbage", encoding="utf-8")
    monkeypatch.setattr(C.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(ws)))
    ctrl2 = _ctrl(ws)
    ctrl2._browse_workdir()
    rec = ctrl2.state_manager.get_record("s.tif")
    assert rec["status"] == "reconstructed"
    assert rec["labels_path"] == str(new_labels)
    for key in ("labels_path", "prob_path", "qc_features_csv",
                "original_labels_path", "border_cells_path"):
        assert not rec.get(key) or Path(rec[key]).exists(), key
    assert any(T.state_reconciled_line(1) in l
               for l in ctrl2.ui.log_output.lines)
    # ... and it is saved.
    on_disk = json.loads(SM.step4_state_path(str(ws)).read_text(
        encoding="utf-8"))
    assert on_disk["stacks"]["s.tif"]["labels_path"] == str(new_labels)


def test_a_gone_result_drops_its_keys_and_re_derives_the_status(tmp_path):
    """(a): no cache -> unprocessed; a complete cache -> predicted."""
    ws = tmp_path
    stacks = {
        "a.tif": {"status": "finalized", "labels_path": str(ws / "no.tif"),
                  "qc_features_csv": "c", "made_with": {}, "n_cells": 3,
                  "finalized_at": "t", "source_stack_path": "keep"},
        "b.tif": {"status": "reconstructed",
                  "labels_path": str(ws / "no2.tif")},
    }
    cache = ws / "cache"
    cache.mkdir()
    for v in M.VIEWS:
        for k in ("flows", "probs"):
            np.save(M.cache_array_path(cache, "b.tif", v, k), np.zeros(1))
    M.cache_meta_path(cache, "b.tif").write_text("{}", encoding="utf-8")
    changes = M.reconcile_records(stacks, measure_keys=SM._QC_METADATA_KEYS,
                                  cache_dir=cache, manifests=[])
    assert set(changes) == {"a.tif", "b.tif"}
    assert stacks["a.tif"] == {"status": "unprocessed",
                               "source_stack_path": "keep"}
    assert stacks["b.tif"]["status"] == "predicted"


def test_gone_measure_files_demote_to_reconstructed(tmp_path):
    """(b): labels there, CSV gone -> measure keys dropped only."""
    labels = tmp_path / "l.tif"
    labels.write_bytes(b"x")
    stacks = {"s.tif": {"status": "finalized", "labels_path": str(labels),
                        "qc_features_csv": str(tmp_path / "gone.csv"),
                        "finalized_at": "t", "measured_at": "t",
                        "original_labels_path": "o"}}
    changes = M.reconcile_records(stacks, measure_keys=SM._QC_METADATA_KEYS,
                                  cache_dir=None, manifests=[])
    assert "s.tif" in changes
    rec = stacks["s.tif"]
    assert rec["status"] == "reconstructed"
    assert rec["labels_path"] == str(labels)
    assert rec["original_labels_path"] == "o"
    assert not set(SM._QC_METADATA_KEYS) & set(rec)


def test_an_orphaned_run_is_adopted_newest_first(tmp_path):
    """(c): a record with no live labels takes the newest manifest."""
    ws = tmp_path
    newer = _run(ws, "20260929T040000Z", cells=9)
    older = _run(ws, "20260929T030000Z", cells=4)
    manifests = [json.loads((newer.parent / "manifest.json").read_text()),
                 json.loads((older.parent / "manifest.json").read_text())]
    stacks = {}
    changes = M.reconcile_records(stacks, measure_keys=SM._QC_METADATA_KEYS,
                                  cache_dir=None, manifests=manifests)
    assert changes and stacks["s.tif"]["labels_path"] == str(newer)
    assert stacks["s.tif"]["n_cells"] == 9
    # A live record is never replaced by a manifest.
    stacks2 = {"s.tif": {"status": "qc_extracted", "labels_path": str(older)}}
    assert M.reconcile_records(stacks2, measure_keys=SM._QC_METADATA_KEYS,
                               cache_dir=None, manifests=manifests) == {}
    assert stacks2["s.tif"]["labels_path"] == str(older)


def test_reconcile_never_deletes_and_adopt_writes_nothing(tmp_path):
    ws = tmp_path / "ws"
    labels = _run(ws, "20260929T030000Z")
    SM.step4_state_path(str(ws)).parent.mkdir(parents=True, exist_ok=True)
    SM.step4_state_path(str(ws)).write_text(json.dumps({
        "schema_version": 2, "session": {},
        "stacks": {"s.tif": {"status": "qc_extracted",
                             "labels_path": str(ws / "gone.tif")}}}),
        encoding="utf-8")
    before = SM.step4_state_path(str(ws)).read_bytes()
    listing = sorted(str(p) for p in ws.rglob("*"))
    ctrl = _ctrl(ws)
    ctrl.working_dir = ""
    ctrl.adopt_workspace(str(ws))
    assert ctrl.state_manager.get_record("s.tif")["labels_path"] == str(labels)
    assert SM.step4_state_path(str(ws)).read_bytes() == before
    assert sorted(str(p) for p in ws.rglob("*")) == listing


@pytest.mark.parametrize("shape, axes, expected", [
    ((81, 2, 240, 240), "ZCYX", 561),
    ((2, 81, 240, 240), "CZYX", 561),
    ((81, 240, 240), "ZYX", 561),
    ((81, 240, 240), "", 561),
    ((240, 240), "YX", None),
])
def test_the_predict_slice_count(shape, axes, expected):
    """The work is Z + Y + X (81 + 240 + 240), not the first dim x 3."""
    assert M.prediction_slice_count(shape, axes) == expected


def test_the_reconciled_line_counts_right():
    """Integration fix: the driver saw "1 stacks were ..."."""
    assert T.state_reconciled_line(1) == (
        "Updated 1 stack to match the files on disk.")
    assert T.state_reconciled_line(3) == (
        "Updated 3 stacks to match the files on disk.")

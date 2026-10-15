"""Decided 2026-09-29:
``step4_state.json`` lives at the workspace ROOT, beside ``step1_state.json``,
``step2_state.json`` and ``step3_state.json``.

Read order: the root file -> the old ``artifacts/step4/step4_state.json``
(moved once: written to the root, the old copy renamed aside as
``step4_state.json.moved-<stamp>``) -> the ``.bak`` (only when the file
read was corrupt) -> the legacy ``stack_segmentation_state.json``.
Backups stay in ``artifacts/step4/state_backup/``.
"""
from __future__ import annotations

import json
from pathlib import Path

from napari_epicycle.segmentation import state_manager as SM


def _root(ws):
    return Path(ws) / "step4_state.json"


def _old(ws):
    return Path(ws) / "artifacts" / "step4" / "step4_state.json"


def _v2(stacks):
    return {"schema_version": 2, "session": {}, "stacks": stacks}


def _write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")


def test_the_state_path_is_the_workspace_root(tmp_path):
    assert SM.step4_state_path(str(tmp_path)) == _root(tmp_path)


def test_a_fresh_workspace_writes_the_root_only(tmp_path):
    m = SM.StackStateManager()
    m.load(str(tmp_path))
    m.register_stacks(["a.tif"])
    assert m.save()
    assert _root(tmp_path).exists()
    assert not _old(tmp_path).exists()


def test_an_artifacts_copy_moves_to_the_root_once(tmp_path):
    _write(_old(tmp_path), _v2({"a.tif": {"status": "predicted"}}))
    m = SM.StackStateManager()
    m.load(str(tmp_path))
    assert m.get_stack_status("a.tif") == "predicted"
    assert _root(tmp_path).exists()
    assert not _old(tmp_path).exists()
    moved = list(_old(tmp_path).parent.glob("step4_state.json.moved-*"))
    assert len(moved) == 1
    back = json.loads(_root(tmp_path).read_text(encoding="utf-8"))
    assert back["stacks"]["a.tif"]["status"] == "predicted"


def test_adopt_reads_the_artifacts_copy_and_moves_nothing(tmp_path):
    _write(_old(tmp_path), _v2({"a.tif": {"status": "predicted"}}))
    m = SM.StackStateManager()
    m.load(str(tmp_path), write=False)
    assert m.get_stack_status("a.tif") == "predicted"
    assert _old(tmp_path).exists() and not _root(tmp_path).exists()


def test_the_root_wins_when_both_exist(tmp_path):
    _write(_root(tmp_path), _v2({"a.tif": {"status": "reconstructed"}}))
    _write(_old(tmp_path), _v2({"a.tif": {"status": "predicted"}}))
    m = SM.StackStateManager()
    m.load(str(tmp_path))
    assert m.get_stack_status("a.tif") == "reconstructed"
    assert _old(tmp_path).exists()          # left alone: the root won


def test_the_legacy_fallback_still_works(tmp_path):
    (tmp_path / "stack_segmentation_state.json").write_text(json.dumps({
        "schema_version": 1, "stacks": {"a.tif": {"status": "predicted"}}}),
        encoding="utf-8")
    m = SM.StackStateManager()
    m.load(str(tmp_path))
    assert m.get_stack_status("a.tif") == "predicted"
    assert m.loaded_from == "legacy"


def test_a_corrupt_root_is_quarantined_beside_it_then_the_backup(tmp_path):
    _root(tmp_path).write_text("{torn", encoding="utf-8")
    _write(Path(tmp_path) / "artifacts" / "step4" / "state_backup"
           / "step4_state.json.bak", _v2({"a.tif": {"status": "predicted"}}))
    m = SM.StackStateManager()
    m.load(str(tmp_path))
    assert m.loaded_from == "backup"
    assert list(Path(tmp_path).glob("step4_state.json.corrupt-*"))


def test_the_step1_guard_reads_the_root_file(tmp_path):
    from napari_epicycle.data_preparation._downstream import downstream_uses
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(_root(ws), _v2({"P1.tif": {"status": "reconstructed"}}))
    assert downstream_uses(str(ws), "P1").step4_kinds == ("reconstruction",)


def test_the_step1_guard_still_reads_an_unmoved_artifacts_copy(tmp_path):
    from napari_epicycle.data_preparation._downstream import downstream_uses
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(_old(ws), _v2({"P1.tif": {"status": "predicted"}}))
    assert downstream_uses(str(ws), "P1").step4_kinds == ("prediction",)

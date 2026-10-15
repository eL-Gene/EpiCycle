"""F3 / : ``step4_state.json`` (schema 2), quarantine, backup, legacy.

Invariants I-4 and I-15.  Qt-free.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.segmentation import state_manager as SM
from napari_epicycle.segmentation.state_manager import StackStateManager


def _new(ws) -> Path:
    return Path(ws) / "step4_state.json"       # the root


def _bak(ws) -> Path:
    return (Path(ws) / "artifacts" / "step4" / "state_backup"
            / "step4_state.json.bak")


def _legacy(ws) -> Path:
    return Path(ws) / "stack_segmentation_state.json"


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else json.dumps(data)
    path.write_text(text, encoding="utf-8")


def _v1(**stacks):
    return {"schema_version": 1, "stacks": stacks,
            "session": {"input_stacks_dir": "D:/stacks"},
            "active_stack": None, "last_updated": "2026-01-01T00:00:00+00:00"}


def _v2(**stacks):
    return {"schema_version": 2, "stacks": stacks, "session": {},
            "active_stack": None, "last_updated": None,
            "migrated_from_legacy_at": None}


# ---------------------------------------------------------------- paths


def test_the_state_file_lives_at_the_workspace_root(tmp_path):
    assert StackStateManager.STATE_FILENAME == "step4_state.json"
    assert SM.LEGACY_STATE_FILENAME == "stack_segmentation_state.json"
    assert SM.step4_state_path(str(tmp_path)) == _new(tmp_path)
    assert SM.step4_backup_path(str(tmp_path)) == _bak(tmp_path)


def test_a_fresh_workspace_is_an_empty_schema_2_state(tmp_path):
    mgr = StackStateManager()
    state = mgr.load(str(tmp_path))
    assert state["schema_version"] == 2
    assert state["stacks"] == {}
    assert mgr.save_blocked is False
    assert mgr.take_notices() == []


def test_save_writes_the_new_file_and_never_the_legacy_one(tmp_path):
    _write(_legacy(tmp_path), _v1(a={"status": "predicted"}))
    before = _legacy(tmp_path).read_bytes()
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    mgr.set_stack_status("a", "reconstructed")
    assert mgr.save() is True
    assert _new(tmp_path).exists()
    assert _legacy(tmp_path).read_bytes() == before      # I-15
    data = json.loads(_new(tmp_path).read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["migrated_from_legacy_at"]
    assert data["stacks"]["a"]["status"] == "reconstructed"


def test_a_good_file_missing_keys_is_merged_over_the_empty_state(tmp_path):
    _write(_new(tmp_path), {"schema_version": 2})
    mgr = StackStateManager()
    state = mgr.load(str(tmp_path))
    assert state["stacks"] == {} and state["session"] == {}
    assert mgr.take_notices() == []


# ------------------------------------------------------------ quarantine


@pytest.mark.parametrize("bad", [
    "{ broken json !!!",
    "[1, 2, 3]",
    json.dumps({"schema_version": 3, "stacks": {}}),       # a future version
    json.dumps({"schema_version": 1, "stacks": {}}),       # wrong file version
])
def test_an_unreadable_file_is_set_aside_never_overwritten(tmp_path, bad):
    _write(_new(tmp_path), bad)
    mgr = StackStateManager()
    state = mgr.load(str(tmp_path))
    assert state["stacks"] == {}
    corrupt = list(_new(tmp_path).parent.glob("step4_state.json.corrupt-*"))
    assert len(corrupt) == 1
    assert corrupt[0].read_text(encoding="utf-8") == bad         # I-4
    notices = mgr.take_notices()
    assert notices == [T.STATE_STARTED_EMPTY.format(file=corrupt[0].name)]
    assert mgr.save() is True        # the next save writes a fresh file
    assert corrupt[0].read_text(encoding="utf-8") == bad


def test_a_bad_file_falls_back_to_the_backup_first(tmp_path):
    _write(_bak(tmp_path), _v2(a={"status": "finalized"}))
    _write(_legacy(tmp_path), _v1(a={"status": "predicted"}))
    _write(_new(tmp_path), "garbage")
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    assert mgr.get_stack_status("a") == "finalized"
    corrupt = next(_new(tmp_path).parent.glob("step4_state.json.corrupt-*"))
    assert mgr.take_notices() == [
        T.STATE_RESTORED_FROM_BACKUP.format(file=corrupt.name)]


def test_then_the_legacy_file(tmp_path):
    _write(_bak(tmp_path), "also garbage")
    _write(_legacy(tmp_path), _v1(a={"status": "predicted"}))
    _write(_new(tmp_path), "garbage")
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    assert mgr.get_stack_status("a") == "predicted"
    corrupt = next(_new(tmp_path).parent.glob("step4_state.json.corrupt-*"))
    assert mgr.take_notices() == [
        T.STATE_RESTORED_FROM_LEGACY.format(file=corrupt.name)]


def test_a_failing_rename_blocks_every_save(tmp_path, monkeypatch):
    _write(_new(tmp_path), "garbage")
    real_replace = os.replace

    def refuse(src, dst, *a, **k):
        if ".corrupt-" in str(dst):
            raise PermissionError("held by another process")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(SM.os, "replace", refuse)
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    assert mgr.save_blocked is True
    assert T.STATE_SAVE_BLOCKED in mgr.take_notices()
    mgr.set_stack_status("a", "predicted")
    assert mgr.save() is False
    assert _new(tmp_path).read_text(encoding="utf-8") == "garbage"   # I-4


def test_adopt_mode_reports_and_renames_nothing(tmp_path):
    _write(_new(tmp_path), "garbage")
    listing = sorted(p.name for p in _new(tmp_path).parent.iterdir())
    mgr = StackStateManager()
    state = mgr.load(str(tmp_path), write=False)
    assert state["stacks"] == {}
    assert sorted(p.name for p in _new(tmp_path).parent.iterdir()) == listing
    assert mgr.save_blocked is True and mgr.needs_write_load is True
    assert mgr.save() is False
    assert _new(tmp_path).read_text(encoding="utf-8") == "garbage"
    # The first user action re-loads in write mode, which sets it aside.
    mgr.load(str(tmp_path))
    assert mgr.save_blocked is False and mgr.needs_write_load is False
    assert list(_new(tmp_path).parent.glob("step4_state.json.corrupt-*"))


# ---------------------------------------------------------------- legacy


def test_a_corrupt_legacy_file_is_left_alone(tmp_path):
    _write(_legacy(tmp_path), "{ nope")
    mgr = StackStateManager()
    state = mgr.load(str(tmp_path))
    assert state["stacks"] == {}
    assert _legacy(tmp_path).read_text(encoding="utf-8") == "{ nope"
    assert T.STATE_LEGACY_UNREADABLE in mgr.take_logs()
    assert not list(tmp_path.glob("*.corrupt-*"))


def test_a_newer_legacy_file_is_named_once(tmp_path):
    _write(_legacy(tmp_path), _v1(a={"status": "predicted"}))
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    mgr.save()
    mgr.take_notices()
    # An older build writes the legacy file again, later.
    later = _v1(a={"status": "finalized"})
    later["last_updated"] = "2999-01-01T00:00:00+00:00"
    _write(_legacy(tmp_path), later)
    mgr2 = StackStateManager()
    mgr2.load(str(tmp_path))
    assert mgr2.take_notices() == [T.STATE_LEGACY_NEWER]
    assert mgr2.get_stack_status("a") == "predicted"    # not merged
    mgr2.save()
    mgr3 = StackStateManager()
    mgr3.load(str(tmp_path))
    assert mgr3.take_notices() == []                     # once


# ---------------------------------------------------------- the backup


def test_save_rotates_the_previous_good_file_to_the_backup(tmp_path):
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    mgr.set_stack_status("a", "predicted")
    mgr.save()
    first = _new(tmp_path).read_bytes()
    mgr.set_stack_status("a", "reconstructed")
    mgr.save()
    assert _bak(tmp_path).read_bytes() == first


def test_a_failed_rotation_never_blocks_the_save(tmp_path, monkeypatch):
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    mgr.save()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(SM.shutil, "copy2", boom)
    mgr.set_stack_status("a", "predicted")
    assert mgr.save() is True
    assert json.loads(_new(tmp_path).read_text(encoding="utf-8"))[
        "stacks"]["a"]["status"] == "predicted"


def test_save_fsyncs_before_the_replace(tmp_path, monkeypatch):
    order = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(SM.os, "fsync",
                        lambda fd: (order.append("fsync"), real_fsync(fd)))
    monkeypatch.setattr(SM.os, "replace",
                        lambda a, b: (order.append("replace"),
                                      real_replace(a, b)))
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    mgr.save()
    assert order == ["fsync", "replace"]


def test_a_failed_write_raises_and_leaves_the_old_file(tmp_path, monkeypatch):
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    mgr.set_stack_status("a", "predicted")
    mgr.save()
    good = _new(tmp_path).read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(SM.json, "dump", boom)
    mgr.set_stack_status("a", "reconstructed")
    with pytest.raises(OSError):
        mgr.save()
    assert _new(tmp_path).read_bytes() == good
    assert not list(_new(tmp_path).parent.glob(".step4_state_*.tmp"))


# ------------------------------------------------------------- migration


def test_v1_migration_table(tmp_path):
    lat = tmp_path / "lat.pt"
    ax = tmp_path / "ax.pt"
    lat.write_bytes(b"x")
    ax.write_bytes(b"x")
    pack = tmp_path / "pack.json"
    pack.write_text(json.dumps({"lateral_model": str(lat),
                                "axial_model": str(ax)}), encoding="utf-8")
    v1 = _v1(
        seg={"status": "segmented", "labels_path": "L"},
        smo={"status": "smoothed", "qc_features_csv": "C"},
        rec={"status": "reconstructed", "extra": 7},
    )
    v1["session"] = {"input_stacks_dir": "D:/stacks", "model_source": 0,
                     "model_pack_path": str(pack)}
    out = SM.migrate_v1_to_v2(v1)
    assert out["schema_version"] == 2
    assert out["session"]["stack_folders"] == ["D:/stacks"]
    am = out["session"]["active_model"]
    assert am["kind"] == "external"
    assert am["lateral"] == str(lat) and am["axial"] == str(ax)
    # Audit D7: a "segmented" record with an earlier result keeps it.
    assert out["stacks"]["seg"]["status"] == "reconstructed"
    assert out["stacks"]["seg"]["failed_stage"] == "reconstruct"
    assert out["stacks"]["seg"]["labels_path"] == "L"
    assert out["stacks"]["smo"]["status"] == "qc_extracted"
    assert out["stacks"]["rec"] == {"status": "reconstructed", "extra": 7}
    # Lossless: the old session keys stay for today's widgets.
    assert out["session"]["input_stacks_dir"] == "D:/stacks"
    assert v1["stacks"]["seg"]["status"] == "segmented"    # input untouched


@pytest.mark.parametrize("session, expected_kind", [
    ({"model_source": 1, "lateral_model_path": "A.pt",
      "axial_model_path": "B.pt"}, "external"),
    ({"model_source": 2}, None),
    ({"model_source": 0, "model_pack_path": "missing.json"}, None),
])
def test_v1_model_source_migration(session, expected_kind):
    v1 = _v1()
    v1["session"] = session
    am = SM.migrate_v1_to_v2(v1)["session"]["active_model"]
    if expected_kind is None:
        assert am is None
    else:
        assert am["kind"] == expected_kind
        assert am["lateral"] == "A.pt" and am["axial"] == "B.pt"
        assert am["lateral_label"] == "A.pt" and am["axial_label"] == "B.pt"


# ---------------------------------------------------------- accessors


def test_get_record_is_a_copy(tmp_path):
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    mgr.set_stack_status("a", "predicted", labels_path="L")
    rec = mgr.get_record("a")
    rec["labels_path"] = "changed"
    assert mgr.get_record("a")["labels_path"] == "L"
    assert mgr.get_record("missing") == {}


def test_clear_stack_keys_and_ticks_and_model(tmp_path):
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    mgr.set_stack_status("a", "qc_extracted", qc_features_csv="C", x=1)
    mgr.clear_stack_keys("a", ["qc_features_csv", "nope"])
    assert "qc_features_csv" not in mgr.get_record("a")
    assert mgr.get_record("a")["x"] == 1
    mgr.set_ticked(["a", "b"])
    assert mgr.get_ticked() == ["a", "b"]
    snap = {"lateral": "L", "axial": "A", "kind": "external"}
    mgr.set_active_model(snap)
    assert mgr.get_active_model() == snap
    mgr.save()
    mgr2 = StackStateManager()
    mgr2.load(str(tmp_path))
    assert mgr2.get_ticked() == ["a", "b"]
    assert mgr2.get_active_model() == snap


def test_update_record_keeps_the_status(tmp_path):
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    mgr.set_stack_status("a", "reconstructed")
    mgr.update_record("a", n_cells=5)
    assert mgr.get_stack_status("a") == "reconstructed"
    assert mgr.get_record("a")["n_cells"] == 5


def test_failed_is_a_valid_status_and_has_theme_entries():
    from napari_epicycle.ui_shell.theme import (
        STACK_STATUS_COLORS, STACK_STATUS_DOTS, STACK_STATUS_ICONS,
    )
    assert "failed" in SM.VALID_STATUSES
    assert STACK_STATUS_COLORS["failed"] == "#e07070"
    assert STACK_STATUS_ICONS["failed"] == "mdi6.close-circle-outline"
    assert STACK_STATUS_DOTS["failed"] == ""


def test_widened_qc_keys_are_all_cleared_by_reset(tmp_path):
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    keys = {k: "x" for k in SM._QC_METADATA_KEYS}
    mgr.set_stack_status("a", "finalized", labels_path="L", **keys)
    mgr.reset_to_reconstructed("a")
    rec = mgr.get_record("a")
    assert rec["status"] == "reconstructed"
    assert rec["labels_path"] == "L"
    assert not set(SM._QC_METADATA_KEYS) & set(rec)
    for k in ("qc_features_csv", "border_cells_path", "smoothed_cache_path",
              "measured_spacing", "finalized_fingerprint", "smoothed_at",
              "n_thin_kept", "measured_at", "measured_cells",
              "measured_z_orientation", "finalized_at"):
        assert k in SM._QC_METADATA_KEYS


# ------------------------------------------------------------ the audit


@pytest.mark.parametrize("rec, status", [
    ({"status": "segmented"}, "failed"),
    ({"status": "segmented", "labels_path": "L"}, "reconstructed"),
    ({"status": "segmented", "labels_path": "L", "qc_features_csv": "C"},
     "qc_extracted"),
])
def test_a_segmented_record_infers_its_status(rec, status):
    out = SM.migrate_v1_to_v2(_v1(s=rec))["stacks"]["s"]
    assert out["status"] == status
    assert out["failed_stage"] == "reconstruct"


@pytest.mark.parametrize("bad", [
    {"schema_version": 2, "stacks": [], "session": {}},
    {"schema_version": 2, "stacks": {}, "session": "oops"},
])
def test_a_wrong_typed_file_is_quarantined_not_emptied(tmp_path, bad):
    _write(_new(tmp_path), bad)
    mgr = StackStateManager()
    mgr.load(str(tmp_path))
    kept = list(_new(tmp_path).parent.glob("step4_state.json.corrupt-*"))
    assert len(kept) == 1
    assert json.loads(kept[0].read_text(encoding="utf-8")) == bad


def test_the_backup_only_holds_a_file_that_parses(tmp_path):
    _write(_bak(tmp_path), _v2(a={"status": "finalized"}))
    good = _bak(tmp_path).read_bytes()
    _write(_new(tmp_path), "garbage")
    mgr = StackStateManager()
    mgr.load(str(tmp_path), write=False)       # adopt: nothing renamed
    mgr.save_blocked = False                   # force a save over garbage
    mgr.save()
    assert _bak(tmp_path).read_bytes() == good

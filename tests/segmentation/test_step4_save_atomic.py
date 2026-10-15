"""F11 / F20 / F23: Save and Finalize write atomically, report, and stop.

F11, F20 (record side), F23, (edit timestamps
held until Save).  Qt-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_text as T

from ._step4_fakes import controller


@pytest.fixture
def yes(monkeypatch):
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)


@pytest.fixture
def loaded(tmp_path):
    ctrl = controller(tmp_path)
    run = tmp_path / "artifacts" / "step4" / "runs" / "r"
    run.mkdir(parents=True)
    labels = run / "labels_3d.tif"
    old = np.zeros((3, 4, 4), np.uint32)
    old[1, 1, 1] = 7
    tifffile.imwrite(str(labels), old)
    csv = tmp_path / "artifacts" / "step4" / "qc" / "s.tif_features.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("label\n7\n", encoding="utf-8")
    ctrl.state_manager.set_stack_status("s.tif", "qc_extracted",
                                        labels_path=str(labels),
                                        qc_features_csv=str(csv))
    ctrl._current_stack_name = "s.tif"
    new = old.copy()
    new[2, 2, 2] = 9
    ctrl._labels_volume = new
    ctrl._qc_features_df = pd.DataFrame({"label": [7, 9],
                                         "qc_tier": ["good", "bad"]})
    ctrl._labels_dirty = True
    ctrl._pending_edits = {"labels_changed_at": "T1", "smoothed_at": "T0",
                           "n_thin_kept": 2, "_stack": "s.tif"}
    return ctrl, labels, csv, old


def test_save_writes_both_files_and_the_held_timestamps(loaded, yes):
    ctrl, labels, csv, _old = loaded
    assert ctrl._save_tier_assignments() is True
    assert tifffile.imread(str(labels))[2, 2, 2] == 9
    assert "9" in csv.read_text(encoding="utf-8")
    rec = ctrl.state_manager.get_record("s.tif")
    assert rec["labels_changed_at"] == "T1"
    assert rec["smoothed_at"] == "T0" and rec["n_thin_kept"] == 2
    assert not ctrl._has_unsaved_changes and ctrl._pending_edits == {}
    assert not list(labels.parent.glob("*.tmp"))
    assert ctrl._save_in_flight is False


def test_an_exception_mid_write_leaves_the_old_file_intact(loaded, yes,
                                                           monkeypatch):
    ctrl, labels, csv, old = loaded
    before_csv = csv.read_text(encoding="utf-8")
    import napari_epicycle.fusion_reconstruction.qc_features as QF

    def boom(df, path):
        raise OSError("disk full")

    monkeypatch.setattr(QF, "save_patch_csv", boom)
    assert ctrl._save_tier_assignments() is False
    assert np.array_equal(tifffile.imread(str(labels)), old)
    assert csv.read_text(encoding="utf-8") == before_csv
    assert not list(labels.parent.glob("*.tmp"))
    assert ctrl._has_unsaved_changes                    # still dirty
    assert ctrl._pending_edits["labels_changed_at"] == "T1"
    assert "labels_changed_at" not in ctrl.state_manager.get_record("s.tif")
    assert ctrl.ui.qc_status_line.sets[-1] == \
        T.MT_SAVE_FAILED.format(stack="s.tif")
    assert ctrl._save_in_flight is False


def test_a_cancelled_save_returns_false(loaded, monkeypatch):
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: False)
    ctrl, labels, _csv, old = loaded
    assert ctrl._save_tier_assignments() is False
    assert np.array_equal(tifffile.imread(str(labels)), old)


def test_a_blocked_state_file_refuses_before_writing(loaded, yes):
    ctrl, labels, _csv, old = loaded
    ctrl.state_manager.save_blocked = True
    assert ctrl._save_tier_assignments(confirm=False) is False
    assert np.array_equal(tifffile.imread(str(labels)), old)
    assert ctrl._has_unsaved_changes


def test_failed_save_keeps_dirty_and_stops_the_load(loaded, monkeypatch):
    """Load's Save stops on a failed Save."""
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_three_choice",
                        lambda *a, **k: "primary")
    ctrl, labels, _csv, _old = loaded
    ctrl.state_manager.save_blocked = True
    vol = ctrl._labels_volume
    ctrl._request_load("other.tif")
    assert ctrl._current_stack_name == "s.tif"
    assert ctrl._labels_volume is vol
    assert ctrl._has_unsaved_changes
    assert not ctrl._load_in_flight


def test_finalize_records_the_fingerprint(loaded, yes):
    ctrl, labels, csv, _old = loaded
    assert ctrl._finalize() is True
    rec = ctrl.state_manager.get_record("s.tif")
    assert rec["status"] == "finalized"
    assert rec["finalized_at"]
    fp = rec["finalized_fingerprint"]
    assert fp == M.finalized_fingerprint(str(labels), str(csv))
    assert fp["labels"][0] == labels.stat().st_size
    assert rec["labels_changed_at"] == "T1"


def test_a_failed_finalize_is_not_finalized(loaded, yes, monkeypatch):
    ctrl, _labels, _csv, _old = loaded
    import tifffile as tf

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(tf, "imwrite", boom)
    assert ctrl._finalize() is False
    assert ctrl.state_manager.get_stack_status("s.tif") == "qc_extracted"
    assert "finalized_fingerprint" not in ctrl.state_manager.get_record("s.tif")


# ------------------------------------------------------------ the audit


def test_a_discard_never_carries_edits_to_another_stack(loaded, yes,
                                                        tmp_path):
    """Audit D1: Remove on A, Discard, then B's Save writes none of A's
    held timestamps."""
    ctrl, _labels, _csv, _old = loaded
    ctrl._has_unsaved_changes = False           # "Discard & Continue"
    assert ctrl._pending_edits == {} and ctrl._undo_slot is None
    ctrl._pending_edits = {"labels_changed_at": "T9", "_stack": "s.tif"}
    ctrl._current_stack_name = "b.tif"
    run = tmp_path / "artifacts" / "step4" / "runs" / "b"
    run.mkdir(parents=True)
    ctrl.state_manager.set_stack_status("b.tif", "qc_extracted",
                                        labels_path=str(run / "l.tif"))
    ctrl._qc_features_df = None
    ctrl._labels_dirty = True
    assert ctrl._save_tier_assignments(confirm=False) is True
    rec = ctrl.state_manager.get_record("b.tif")
    for key in ("labels_changed_at", "smoothed_at", "n_thin_kept"):
        assert key not in rec


def test_the_stack_is_re_checked_after_the_modal(loaded, monkeypatch):
    """Audit L3: the question ran the event loop and the stack changed."""
    import napari_epicycle.ui_shell.theme as theme
    ctrl, labels, _csv, old = loaded

    def confirm(*a, **k):
        ctrl._current_stack_name = None          # e.g. Reconstruct unloaded it
        return True

    monkeypatch.setattr(theme, "themed_confirm", confirm)
    assert ctrl._save_tier_assignments() is False
    assert np.array_equal(tifffile.imread(str(labels)), old)
    ctrl._current_stack_name = "s.tif"
    assert ctrl._finalize() is False
    assert ctrl.state_manager.get_stack_status("s.tif") == "qc_extracted"


def test_a_failed_csv_replace_puts_the_labels_back(loaded, yes, monkeypatch):
    """Audit L4: Windows refuses the CSV replace -> labels unchanged."""
    import napari_epicycle.segmentation.controller as C
    ctrl, labels, csv, old = loaded
    before = labels.read_bytes()
    real = C.os.replace

    csv_before = csv.read_bytes()

    def replace(src, dst, *a, **k):
        # Refuse ONLY the temp -> CSV replace, so the restore can run.
        if str(dst) == str(csv) and str(src).endswith(".tmp"):
            raise PermissionError("held by Excel")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(C.os, "replace", replace)
    assert ctrl._save_tier_assignments() is False
    monkeypatch.setattr(C.os, "replace", real)
    assert labels.read_bytes() == before
    assert csv.read_bytes() == csv_before
    assert ctrl._has_unsaved_changes
    for folder in (labels.parent, csv.parent):
        assert not list(folder.glob("*.prev"))
        assert not list(folder.glob("*.tmp"))


def test_a_failed_restore_is_said_in_the_log(loaded, yes, monkeypatch):
    """Re-audit L-a: a .prev that cannot be put back is named."""
    import napari_epicycle.segmentation.controller as C
    ctrl, labels, csv, _old = loaded
    real = C.os.replace

    def replace(src, dst, *a, **k):
        if str(dst) == str(csv) and str(src).endswith(".tmp"):
            raise PermissionError("held by Excel")
        if str(src).endswith(".prev"):
            raise PermissionError("still held")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(C.os, "replace", replace)
    assert ctrl._save_tier_assignments() is False
    assert any(l.startswith("[ERROR] could not put back") and ".prev" in l
               for l in ctrl.ui.log_output.lines)


def test_a_save_that_writes_nothing_fails(loaded, yes):
    """Audit L5: dirty labels with no labels file is not a save."""
    ctrl, _labels, _csv, _old = loaded
    ctrl.state_manager.update_record("s.tif", labels_path="")
    assert ctrl._save_tier_assignments() is False
    assert ctrl._has_unsaved_changes


def test_a_failed_persist_restores_the_record(loaded, yes, monkeypatch):
    """Audit D5: status still qc_extracted, no fingerprint."""
    ctrl, _labels, _csv, _old = loaded
    monkeypatch.setattr(ctrl.state_manager, "save", lambda *a, **k: False)
    assert ctrl._finalize() is False
    rec = ctrl.state_manager.get_record("s.tif")
    assert rec["status"] == "qc_extracted"
    assert "finalized_fingerprint" not in rec
    assert "labels_changed_at" not in rec

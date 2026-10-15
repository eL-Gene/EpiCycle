"""F5 / F19 / I-5 / I-16: Reset and Reconstruct-again delete the old result.

F5 and F19 (delete, never
rename or keep, as Step 3 does).  Qt-free: ``themed_confirm`` is answered
by the test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from napari_epicycle.segmentation import state_manager as SM

from ._step4_fakes import controller


@pytest.fixture
def yes(monkeypatch):
    import napari_epicycle.ui_shell.theme as theme
    asked = []
    monkeypatch.setattr(theme, "themed_confirm",
                        lambda parent, title, message, **k:
                        asked.append(message) or True)
    return asked


def _measured_stack(ws: Path, name="s.tif"):
    """A stack with a run folder, an original, and every derived file."""
    run = ws / "artifacts" / "step4" / "runs" / "20260101T000000Z"
    run.mkdir(parents=True)
    labels = run / "labels_3d.tif"
    original = run / "labels_3d_original.tif"
    edited = np.zeros((3, 4, 4), np.uint32)
    edited[1, 1, 1] = 2
    pristine = np.zeros((3, 4, 4), np.uint32)
    pristine[1, 1, 1] = 2
    pristine[2, 2, 2] = 5
    tifffile.imwrite(str(labels), edited)
    tifffile.imwrite(str(original), pristine)
    qc = ws / "artifacts" / "step4" / "qc"
    qc.mkdir(parents=True)
    csv = qc / f"{name}_features.csv"
    for p in (csv, qc / f"{name}_adjacency.csv",
              qc / f"{name}_features_raw.csv"):
        p.write_text("label\n1\n", encoding="utf-8")
    eda = ws / "artifacts" / "step4" / "eda"
    eda.mkdir(parents=True)
    for ext in (".npz", ".json"):
        (eda / f"{name}_eda{ext}").write_bytes(b"x")
    border = run / "border_cells.tif"
    smoothed = run / "labels_3d_smoothed.tif"
    border.write_bytes(b"x")
    smoothed.write_bytes(b"x")
    rec = dict(labels_path=str(labels), original_labels_path=str(original),
               qc_features_csv=str(csv), border_cells_path=str(border),
               smoothed_cache_path=str(smoothed), measured_at="t",
               measured_cells=3,
               measured_spacing={"zyx": [1, 2, 3], "source": "user"},
               measured_z_orientation="basal_down", smoothed_at="t",
               n_thin_kept=0, finalized_at="t",
               finalized_fingerprint={"labels": None, "csv": None},
               qc_tier_map={}, channel_name_map={})
    return run, rec, pristine


def _derived(ws, run, name="s.tif"):
    qc = ws / "artifacts" / "step4" / "qc"
    eda = ws / "artifacts" / "step4" / "eda"
    return [qc / f"{name}_features.csv", qc / f"{name}_adjacency.csv",
            qc / f"{name}_features_raw.csv", eda / f"{name}_eda.npz",
            eda / f"{name}_eda.json", run / "border_cells.tif",
            run / "labels_3d_smoothed.tif"]


# ----------------------------------------------------------------- Reset


def test_reset_clears_every_measure_key_and_deletes_the_files(tmp_path, yes):
    ctrl = controller(tmp_path)
    run, rec, pristine = _measured_stack(tmp_path)
    ctrl.state_manager.set_stack_status("s.tif", "finalized", **rec)
    ctrl._current_stack_name = "s.tif"
    ctrl._labels_volume = np.ones((3, 4, 4), np.uint32)
    ctrl._pending_edits = {"smoothed_at": "t"}
    ctrl._undo_slot = {"kind": "smooth"}
    ctrl._labels_dirty = True
    ctrl._reset_to_original()
    got = ctrl.state_manager.get_record("s.tif")
    assert got["status"] == "reconstructed"
    assert not set(SM._QC_METADATA_KEYS) & set(got)              # I-5
    assert got["labels_path"] == rec["labels_path"]
    assert got["original_labels_path"] == rec["original_labels_path"]
    for p in _derived(tmp_path, run):
        assert not p.exists(), p
    assert not list(tmp_path.rglob("*.corrupt*"))                # never renamed
    assert np.array_equal(tifffile.imread(rec["labels_path"]), pristine)
    assert np.array_equal(ctrl._labels_volume, pristine)
    assert ctrl._pending_edits == {} and ctrl._undo_slot is None
    assert not ctrl._has_unsaved_changes
    assert not Path(rec["labels_path"] + ".tmp").exists()
    assert yes and "Reset s.tif" in yes[0]
    assert "3D meshes" in yes[0]                            # re-audit L-c


def test_the_next_measure_merges_no_border_file(tmp_path, yes):
    ctrl = controller(tmp_path)
    run, rec, _ = _measured_stack(tmp_path)
    ctrl.state_manager.set_stack_status("s.tif", "qc_extracted", **rec)
    ctrl._current_stack_name = "s.tif"
    ctrl._labels_volume = np.zeros((3, 4, 4), np.uint32)
    ctrl._reset_to_original()
    rec2 = ctrl.state_manager.get_record("s.tif")
    assert "border_cells_path" not in rec2
    assert "qc_features_csv" not in rec2        # the next Load reads no CSV


def test_a_failed_reset_changes_nothing(tmp_path, yes, monkeypatch):
    ctrl = controller(tmp_path)
    run, rec, _ = _measured_stack(tmp_path)
    ctrl.state_manager.set_stack_status("s.tif", "qc_extracted", **rec)
    ctrl._current_stack_name = "s.tif"
    vol = np.zeros((3, 4, 4), np.uint32)
    ctrl._labels_volume = vol
    import napari_epicycle.segmentation.controller as C

    def boom(*a, **k):
        raise PermissionError("locked")

    monkeypatch.setattr(C.os, "replace", boom)
    before = tifffile.imread(rec["labels_path"])
    ctrl._reset_to_original()
    assert ctrl._labels_volume is vol
    assert ctrl.state_manager.get_record("s.tif")["qc_features_csv"]
    assert np.array_equal(tifffile.imread(rec["labels_path"]), before)
    assert ctrl.ui.qc_status_line.sets[-1].startswith("Resetting s.tif failed")


def test_reset_is_refused_while_a_job_runs(tmp_path, yes):
    ctrl = controller(tmp_path)
    run, rec, _ = _measured_stack(tmp_path)
    ctrl.state_manager.set_stack_status("s.tif", "qc_extracted", **rec)
    ctrl._current_stack_name = "s.tif"
    ctrl._measure_running = True
    ctrl._reset_to_original()
    assert ctrl.state_manager.get_stack_status("s.tif") == "qc_extracted"
    assert not yes


# ------------------------------------------------------ Reconstruct again


def test_reconstruct_again_drops_border_and_csv(tmp_path):
    ctrl = controller(tmp_path)
    run, rec, _ = _measured_stack(tmp_path)
    ctrl.state_manager.set_stack_status("s.tif", "finalized", **rec)
    new_run = tmp_path / "artifacts" / "step4" / "runs" / "20260202T000000Z"
    new_run.mkdir()
    new_labels = new_run / "labels_3d.tif"
    new_labels.write_bytes(b"x")
    ctrl._on_stack_completed("s.tif", {
        "labels_path": str(new_labels),
        "source_stack_path": str(tmp_path / "s.tif"),
        "made_with": None, "num_cells_final": 4})
    got = ctrl.state_manager.get_record("s.tif")
    assert got["status"] == "reconstructed"
    assert got["labels_path"] == str(new_labels)
    assert not set(SM._QC_METADATA_KEYS) & set(got)             # I-16
    assert "original_labels_path" not in got
    assert not run.exists()                     # the old run folder is gone
    for p in _derived(tmp_path, run):
        assert not p.exists(), p
    assert new_labels.exists()


def test_reconstruct_again_unloads_the_loaded_stack(tmp_path):
    ctrl = controller(tmp_path)
    run, rec, _ = _measured_stack(tmp_path)
    ctrl.state_manager.set_stack_status("s.tif", "qc_extracted", **rec)
    ctrl._current_stack_name = "s.tif"
    ctrl._labels_volume = np.zeros((3, 4, 4), np.uint32)
    new_run = tmp_path / "artifacts" / "step4" / "runs" / "new"
    new_run.mkdir()
    (new_run / "labels_3d.tif").write_bytes(b"x")
    ctrl._on_stack_completed("s.tif", {
        "labels_path": str(new_run / "labels_3d.tif"),
        "source_stack_path": "", "made_with": None})
    assert ctrl._current_stack_name is None and ctrl._labels_volume is None


def test_a_first_reconstruct_deletes_nothing(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.state_manager.set_stack_status("s.tif", "predicted")
    run = tmp_path / "artifacts" / "step4" / "runs" / "r"
    run.mkdir(parents=True)
    (run / "labels_3d.tif").write_bytes(b"x")
    ctrl._on_stack_completed("s.tif", {
        "labels_path": str(run / "labels_3d.tif"), "source_stack_path": "",
        "made_with": None})
    assert (run / "labels_3d.tif").exists()


def test_an_old_run_outside_the_runs_folder_is_never_deleted(tmp_path):
    ctrl = controller(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "labels_3d.tif").write_bytes(b"x")
    ctrl.state_manager.set_stack_status(
        "s.tif", "reconstructed",
        labels_path=str(elsewhere / "labels_3d.tif"))
    new_run = tmp_path / "artifacts" / "step4" / "runs" / "n"
    new_run.mkdir(parents=True)
    (new_run / "labels_3d.tif").write_bytes(b"x")
    ctrl._on_stack_completed("s.tif", {
        "labels_path": str(new_run / "labels_3d.tif"),
        "source_stack_path": "", "made_with": None})
    assert (elsewhere / "labels_3d.tif").exists()


# ------------------------------------------------------------ the audit


def test_a_shared_run_folder_is_never_deleted(tmp_path):
    """Audit D4: two stacks reconstructed in the same second share one
    runs/{run_id}/ folder; Reconstruct A again must keep B's files."""
    ctrl = controller(tmp_path)
    shared = tmp_path / "artifacts" / "step4" / "runs" / "20260101T000000Z"
    shared.mkdir(parents=True)
    (shared / "labels_3d.tif").write_bytes(b"a")
    (shared / "b_probabilities.npy").write_bytes(b"b")
    ctrl.state_manager.set_stack_status(
        "a.tif", "reconstructed", labels_path=str(shared / "labels_3d.tif"))
    ctrl.state_manager.set_stack_status(
        "b.tif", "reconstructed", labels_path=str(shared / "labels_3d.tif"),
        prob_path=str(shared / "b_probabilities.npy"))
    new = tmp_path / "artifacts" / "step4" / "runs" / "new"
    new.mkdir()
    (new / "labels_3d.tif").write_bytes(b"x")
    ctrl._on_stack_completed("a.tif", {
        "labels_path": str(new / "labels_3d.tif"), "source_stack_path": "",
        "made_with": None})
    assert (shared / "b_probabilities.npy").exists()
    assert any("kept" in l for l in ctrl.ui.log_output.lines)


def test_a_missing_new_result_deletes_nothing(tmp_path):
    """Audit D15 / re-audit M2: a failure, never a status write."""
    ctrl = controller(tmp_path)
    run, rec, _ = _measured_stack(tmp_path)
    ctrl.state_manager.set_stack_status("s.tif", "qc_extracted", **rec)
    ctrl._on_stack_completed("s.tif", {
        "labels_path": str(tmp_path / "nowhere" / "labels_3d.tif"),
        "source_stack_path": "", "made_with": None})
    assert run.exists()
    for p in _derived(tmp_path, run):
        assert p.exists(), p
    got = ctrl.state_manager.get_record("s.tif")
    # A stack with an earlier result keeps it and shows the failure.
    assert got["status"] == "qc_extracted"
    assert got["labels_path"] == rec["labels_path"]
    assert got["qc_features_csv"] == rec["qc_features_csv"]
    assert got["failed_stage"] == "reconstruct"


def test_a_missing_first_result_reads_failed(tmp_path):
    """Re-audit M2: nothing earlier -> failed, no labels_path written."""
    ctrl = controller(tmp_path)
    ctrl.state_manager.set_stack_status("s.tif", "predicted")
    ctrl._on_stack_completed("s.tif", {
        "labels_path": str(tmp_path / "nowhere.tif"),
        "source_stack_path": "", "made_with": None})
    got = ctrl.state_manager.get_record("s.tif")
    assert got["status"] == "failed" and "labels_path" not in got


def test_reset_deletes_the_stack_meshes(tmp_path, yes):
    """Audit D9: no stale meshes after Reset."""
    ctrl = controller(tmp_path)
    run, rec, _ = _measured_stack(tmp_path)
    meshes = tmp_path / "artifacts" / "step4" / "meshes" / "s.tif"
    meshes.mkdir(parents=True)
    (meshes / "cell_1.obj").write_text("v 0 0 0", encoding="utf-8")
    other = tmp_path / "artifacts" / "step4" / "meshes" / "t.tif"
    other.mkdir()
    ctrl.state_manager.set_stack_status("s.tif", "qc_extracted", **rec)
    ctrl._current_stack_name = "s.tif"
    ctrl._labels_volume = np.zeros((3, 4, 4), np.uint32)
    ctrl._pending_border_removal = True
    ctrl._reset_to_original()
    assert not meshes.exists() and other.exists()
    assert ctrl._pending_border_removal is False

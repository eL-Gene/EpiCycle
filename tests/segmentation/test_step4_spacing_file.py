"""S1: Step 4 reads and writes the root ``patch_spacing.json`` it shares with
Step 3.  Qt-free."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile

from napari_epicycle.data_preparation import _voxel_spacing as VS

from ._step4_fakes import controller


def _tiff(path: Path, spacing_um=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((3, 4, 4), np.uint8)
    if spacing_um is None:
        tifffile.imwrite(str(path), data)
    else:
        z, y, x = spacing_um
        tifffile.imwrite(str(path), data, imagej=True,
                         resolution=(1.0 / x, 1.0 / y),
                         metadata={"spacing": z, "unit": "um"})


def _ws(tmp_path, spacing_um=(2.0, 0.3, 0.3)):
    ws = tmp_path / "ws"
    _tiff(ws / "stacks" / "a.tif", spacing_um=spacing_um)
    return ws, ws / "stacks"


def test_a_resolved_file_value_is_written_to_the_file(tmp_path):
    ws, folder = _ws(tmp_path)
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(folder), persist=True)
    got = VS.read_patch_spacing_file(str(ws))
    assert got["stacks/a.tif"]["source"] == "file"
    assert got["stacks/a.tif"]["zyx"] == ctrl.state_manager.get_record(
        "a.tif")["spacing"]["zyx"]


def test_a_records_user_value_is_copied_into_the_file(tmp_path):
    ws, folder = _ws(tmp_path)
    ctrl = controller(ws)
    ctrl.state_manager.set_stack_status(
        "a.tif", "unprocessed",
        spacing={"zyx": [5.0, 1.0, 1.0], "source": "user"})
    ctrl._scan_stacks_directory(str(folder), persist=True)
    assert VS.read_patch_spacing_file(str(ws))["stacks/a.tif"] == {
        "zyx": [5.0, 1.0, 1.0], "source": "user",
        "set_at": VS.read_patch_spacing_file(str(ws))[
            "stacks/a.tif"]["set_at"]}


def test_a_differing_file_user_entry_wins_and_is_logged(tmp_path, capsys):
    ws, folder = _ws(tmp_path)
    VS.write_patch_spacing_entry(str(ws), "stacks/a.tif", (7, 2, 2), "user")
    ctrl = controller(ws)
    ctrl.state_manager.set_stack_status(
        "a.tif", "unprocessed",
        spacing={"zyx": [5.0, 1.0, 1.0], "source": "user"})
    ctrl._scan_stacks_directory(str(folder), persist=True)
    rec = ctrl.state_manager.get_record("a.tif")
    assert rec["spacing"] == {"zyx": [7.0, 2.0, 2.0], "source": "user"}
    assert VS.read_patch_spacing_file(str(ws))["stacks/a.tif"]["zyx"] == \
        [7.0, 2.0, 2.0]
    assert "[STEP4-SPACING]" in capsys.readouterr().out


def test_step3s_user_value_reaches_a_record_with_none(tmp_path):
    ws, folder = _ws(tmp_path)
    VS.write_patch_spacing_entry(str(ws), "stacks/a.tif", (7, 2, 2), "user")
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(folder), persist=True)
    assert ctrl.state_manager.get_record("a.tif")["spacing"] == {
        "zyx": [7.0, 2.0, 2.0], "source": "user"}


def test_set_spacing_writes_the_file(tmp_path):
    ws, folder = _ws(tmp_path)
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(folder), persist=True)
    assert ctrl._apply_spacing(
        ["a.tif"], {"zyx": [4.0, 0.5, 0.5], "z_orientation": "basal_down",
                    "axes_override": ""}, confirm=False)
    entry = VS.read_patch_spacing_file(str(ws))["stacks/a.tif"]
    assert (entry["source"], entry["zyx"]) == ("user", [4.0, 0.5, 0.5])
    # and a later scan keeps it over the file's tags
    ctrl._scan_stacks_directory(str(folder), persist=True)
    assert ctrl.state_manager.get_record("a.tif")["spacing"]["zyx"] == \
        [4.0, 0.5, 0.5]


def test_the_adopt_writes_nothing_but_reads_the_users_value(tmp_path):
    ws, folder = _ws(tmp_path)
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(folder), persist=False)
    assert not VS.patch_spacing_path(str(ws)).exists()
    VS.write_patch_spacing_entry(str(ws), "stacks/a.tif", (7, 2, 2), "user")
    mtime = VS.patch_spacing_path(str(ws)).stat().st_mtime_ns
    ctrl2 = controller(ws)
    ctrl2._scan_stacks_directory(str(folder), persist=False)
    assert ctrl2._transient_meta["a.tif"]["spacing"]["zyx"] == [7.0, 2.0, 2.0]
    assert VS.patch_spacing_path(str(ws)).stat().st_mtime_ns == mtime


def test_the_two_steps_round_trip_one_voxel_size(tmp_path):
    from napari_epicycle.fusion_reconstruction.state_manager import (
        ReconstructionStateManager,
    )
    ws, folder = _ws(tmp_path)
    m3 = ReconstructionStateManager()
    m3.load(str(ws))
    m3.set_spacing_stack_dir(str(folder))
    m3.set_patch_spacing("a.tif", (6, 1, 1), "user")        # Step 3 sets it
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(folder), persist=True)
    assert ctrl.state_manager.get_record("a.tif")["spacing"] == {
        "zyx": [6.0, 1.0, 1.0], "source": "user"}
    ctrl._apply_spacing(                                     # Step 4 changes
        ["a.tif"], {"zyx": [8.0, 2.0, 2.0], "z_orientation": "basal_down",
                    "axes_override": ""}, confirm=False)
    assert m3.get_patch_spacing("a.tif") == {"zyx": [8.0, 2.0, 2.0],
                                             "source": "user"}


# ------------------------------------------------------- S1 audit folds


def test_h3_step_enter_rereads_the_shared_file(tmp_path):
    ws, folder = _ws(tmp_path)
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(folder), persist=True)
    assert ctrl.state_manager.get_record("a.tif")["spacing"]["source"] == \
        "file"
    # Step 3 sets a size meanwhile
    VS.write_patch_spacing_entry(str(ws), "stacks/a.tif", (7, 2, 2), "user")
    ctrl.on_step_enter()
    assert ctrl.state_manager.get_record("a.tif")["spacing"] == {
        "zyx": [7.0, 2.0, 2.0], "source": "user"}
    # Measure reads the same record
    assert ctrl._spacing_for("a.tif") is not None


def test_h3_step_enter_reads_but_never_writes(tmp_path):
    ws, folder = _ws(tmp_path)
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(folder), persist=True)
    path = VS.patch_spacing_path(str(ws))
    before = path.read_bytes()
    ctrl.on_step_enter()
    assert path.read_bytes() == before


def test_m2_a_failed_write_does_not_commit_the_record(tmp_path, monkeypatch):
    ws, folder = _ws(tmp_path)
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(folder), persist=True)
    old = dict(ctrl.state_manager.get_record("a.tif")["spacing"])
    monkeypatch.setattr(VS, "write_patch_spacing_entry",
                        lambda *a, **k: (_ for _ in ()).throw(
                            PermissionError("locked")))
    ok = ctrl._apply_spacing(
        ["a.tif"], {"zyx": [4.0, 0.5, 0.5], "z_orientation": "basal_down",
                    "axes_override": ""}, confirm=False)
    assert ok is False
    assert ctrl.state_manager.get_record("a.tif")["spacing"] == old
    from napari_epicycle.segmentation import _step4_text as T
    assert any(T.SPACING_SAVE_FAILED in line
               for line in ctrl.ui.log_output.lines)
    monkeypatch.undo()
    ctrl._scan_stacks_directory(str(folder), persist=True)
    assert ctrl.state_manager.get_record("a.tif")["spacing"]["zyx"] == \
        old["zyx"]                         # not silently reverted either way


def test_m8_the_old_to_new_change_is_said_when_the_files_value_takes_over(
        tmp_path, capsys):
    from napari_epicycle.segmentation import _step4_text as T
    ws, folder = _ws(tmp_path)
    ctrl = controller(ws)
    ctrl.state_manager.set_stack_status(
        "a.tif", "qc_extracted",
        spacing={"zyx": [5.0, 1.0, 1.0], "source": "user"},
        measured_spacing={"zyx": [5.0, 1.0, 1.0], "source": "user"})
    VS.write_patch_spacing_entry(str(ws), "stacks/a.tif", (7, 2, 2), "user")
    ctrl._scan_stacks_directory(str(folder), persist=True)
    out = capsys.readouterr().out
    assert "[STEP4-SPACING] a.tif:" in out and "7.0" in out
    assert any(T.MT_SPACING_CHANGED in line
               for line in ctrl.ui.log_output.lines)


def test_m9_the_file_wins_line_reaches_diagnostics(tmp_path):
    ws, folder = _ws(tmp_path)
    VS.write_patch_spacing_entry(str(ws), "stacks/a.tif", (7, 2, 2), "user")
    ctrl = controller(ws)
    ctrl.state_manager.set_stack_status(
        "a.tif", "unprocessed",
        spacing={"zyx": [5.0, 1.0, 1.0], "source": "user"})
    ctrl._scan_stacks_directory(str(folder), persist=True)
    assert any("[STEP4-SPACING]" in line and "wins" in line
               for line in ctrl.ui.log_output.lines)


def test_m3_the_adopt_never_renames_a_corrupt_file(tmp_path):
    ws, folder = _ws(tmp_path)
    open(VS.patch_spacing_path(str(ws)), "w").write("{broken")
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(folder), persist=False)
    assert VS.patch_spacing_path(str(ws)).exists()

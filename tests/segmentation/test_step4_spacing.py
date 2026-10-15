"""Where a stack's voxel size comes from, and the record field.

Qt-free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation._state import voxel_size_for_patch_path
from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_text as T

from ._step4_fakes import controller


def _step1_state(kind="raw", voxel=(2.0, 0.5, 0.5), store=None):
    src = {"source_id": "src_0", "path": "D:/raw.tif", "voxel_size": voxel}
    if store:
        src["processed_store"] = {"path": store, "format": "ome-zarr"}
    return {
        "schema_version": 2,
        "sources": [src],
        "patches": [{"patch_name": "p1", "relative_path": "patches/p1.tif",
                     "sidecar_path": "patches/p1.json",
                     "source_id": "src_0", "source_kind": kind}],
    }


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


# ------------------------------------------------------------ the S1 helper


def test_the_step1_helper_gives_a_raw_patch_its_source_voxel():
    info = voxel_size_for_patch_path(_step1_state(), "patches\\P1.tif")
    assert info == {"voxel_size": [2.0, 0.5, 0.5], "source_kind": "raw",
                    "processed_store": None}


def test_the_step1_helper_never_gives_a_processed_patch_the_raw_voxel():
    info = voxel_size_for_patch_path(
        _step1_state(kind="processed", store="artifacts/step1/x.zarr"),
        "patches/p1.tif")
    assert info["voxel_size"] is None
    assert info["processed_store"] == "artifacts/step1/x.zarr"


def test_the_step1_helper_unknown_and_unit_voxels():
    assert voxel_size_for_patch_path(_step1_state(), "other.tif") is None
    info = voxel_size_for_patch_path(_step1_state(voxel=(1, 1, 1)),
                                     "patches/p1.tif")
    assert info["voxel_size"] is None
    assert voxel_size_for_patch_path({"patches": "garbage"}, "x") is None


# ----------------------------------------------------------- the resolver


def test_rule_1_a_user_value_is_never_overwritten(tmp_path):
    rec = {"spacing": {"zyx": [3.0, 1.0, 1.0], "source": "user"}}
    got = M.resolve_stack_spacing(rec, _step1_state(),
                                  str(tmp_path / "patches/p1.tif"),
                                  str(tmp_path))
    assert got == {"zyx": [3.0, 1.0, 1.0], "source": "user"}


def test_rule_2_step1_wins_over_the_file(tmp_path):
    p = tmp_path / "patches" / "p1.tif"
    _tiff(p, spacing_um=(5.0, 0.25, 0.25))
    got = M.resolve_stack_spacing({}, _step1_state(), str(p), str(tmp_path))
    assert got == {"zyx": [2.0, 0.5, 0.5], "source": "step1"}


def test_rule_2_a_processed_patch_falls_to_its_own_tags(tmp_path):
    p = tmp_path / "patches" / "p1.tif"
    _tiff(p, spacing_um=(5.0, 0.25, 0.25))
    state = _step1_state(kind="processed", store="artifacts/step1/none.zarr")
    got = M.resolve_stack_spacing({}, state, str(p), str(tmp_path))
    assert got["source"] == "file"
    assert got["zyx"] == pytest.approx([5.0, 0.25, 0.25])


def test_rule_3_the_file_and_rule_4_none(tmp_path):
    p = tmp_path / "f.tif"
    _tiff(p, spacing_um=(2.0, 0.3, 0.3))
    got = M.resolve_stack_spacing({}, None, str(p), str(tmp_path))
    assert got["source"] == "file"
    q = tmp_path / "g.tif"
    _tiff(q)
    assert M.resolve_stack_spacing({}, None, str(q), str(tmp_path)) is None


def test_adopt_reads_no_file(tmp_path, monkeypatch):
    # The resolver lives in ``data_preparation/_voxel_spacing`` since R2b and
    # reads ITS module's ``_read_voxel_spacing``; ``M`` only re-exports it, so
    # a patch on ``M`` would reach nothing.  Patch where the code reads.
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    p = tmp_path / "f.tif"
    _tiff(p, spacing_um=(2.0, 0.3, 0.3))
    calls = []
    # The stub answers a value the TIFF does NOT carry, so the result can
    # only come from the stub, never from a real read of the file.
    monkeypatch.setattr(VS, "_read_voxel_spacing",
                        lambda path: calls.append(path) or (7.0, 0.7, 0.7))
    assert M.resolve_stack_spacing({}, None, str(p), str(tmp_path),
                                   read_file=False) is None
    assert calls == []
    # the patch reaches the code: with read_file the stub is what is read
    assert M.resolve_stack_spacing({}, None, str(p), str(tmp_path)) == \
        {"zyx": [7.0, 0.7, 0.7], "source": "file"}
    assert calls == [str(p)]


def test_read_step1_state_never_renames_a_corrupt_file(tmp_path):
    (tmp_path / "step1_state.json").write_text("{ bad", encoding="utf-8")
    assert M.read_step1_state(str(tmp_path)) is None
    assert (tmp_path / "step1_state.json").read_text(encoding="utf-8") == "{ bad"


# --------------------------------------------------------- the record


def test_a_scan_writes_spacing_and_origin_on_the_record(tmp_path):
    ws = tmp_path / "ws"
    folder = ws / "stacks"
    _tiff(folder / "a.tif", spacing_um=(2.0, 0.3, 0.3))
    _tiff(folder / "b.tif")
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(folder), persist=True)
    a = ctrl.state_manager.get_record("a.tif")
    b = ctrl.state_manager.get_record("b.tif")
    assert a["spacing"]["source"] == "file"
    assert a["origin"] == "folder"
    assert b.get("spacing") is None


def test_a_rescan_never_changes_a_measured_stack_silently(tmp_path):
    ws = tmp_path / "ws"
    folder = ws / "stacks"
    _tiff(folder / "a.tif", spacing_um=(2.0, 0.3, 0.3))
    ctrl = controller(ws)
    ctrl.state_manager.set_stack_status(
        "a.tif", "qc_extracted",
        measured_spacing={"zyx": [9.0, 9.0, 9.0], "source": "user"})
    ctrl._scan_stacks_directory(str(folder), persist=True)
    rec = ctrl.state_manager.get_record("a.tif")
    assert rec["measured_spacing"] == {"zyx": [9.0, 9.0, 9.0],
                                       "source": "user"}
    assert rec["spacing"]["source"] == "file"
    assert any(T.MT_SPACING_CHANGED in line
               for line in ctrl.ui.log_output.lines)


def test_the_adopt_scan_registers_nothing(tmp_path):
    ws = tmp_path / "ws"
    _tiff(ws / "stacks" / "a.tif", spacing_um=(2.0, 0.3, 0.3))
    ctrl = controller(ws)
    ctrl._scan_stacks_directory(str(ws / "stacks"), persist=False)
    assert ctrl.state_manager.get_record("a.tif") == {}


# ------------------------------------------------------------ the audit


def test_same_spacing_compares_numbers_not_sources():
    """Audit D6."""
    a = {"zyx": [2.0, 0.5, 0.5], "source": "file"}
    b = {"zyx": [2.0, 0.5, 0.5000000001], "source": "user"}
    assert M.same_spacing(a, b)
    assert not M.same_spacing(a, {"zyx": [2.0, 0.6, 0.5], "source": "file"})
    assert M.same_spacing(None, None) and not M.same_spacing(a, None)


def test_a_user_set_unit_spacing_is_kept(tmp_path):
    """Audit D10: 1/1/1 chosen by the user is not "unknown"."""
    p = tmp_path / "f.tif"
    _tiff(p, spacing_um=(2.0, 0.3, 0.3))
    rec = {"spacing": {"zyx": [1.0, 1.0, 1.0], "source": "user"}}
    assert M.resolve_stack_spacing(rec, None, str(p), str(tmp_path)) ==         {"zyx": [1.0, 1.0, 1.0], "source": "user"}


def test_a_rescan_with_the_same_numbers_says_nothing(tmp_path):
    ws = tmp_path / "ws"
    _tiff(ws / "stacks" / "a.tif", spacing_um=(2.0, 0.5, 0.5))
    ctrl = controller(ws)
    ctrl.state_manager.set_stack_status(
        "a.tif", "qc_extracted",
        measured_spacing={"zyx": [2.0, 0.5, 0.5], "source": "user"})
    ctrl._scan_stacks_directory(str(ws / "stacks"), persist=True)
    assert not any(T.MT_SPACING_CHANGED in l
                   for l in ctrl.ui.log_output.lines)


def test_same_spacing_to_half_a_spin_box_step():
    """Re-audit M3: a file's 0.10833 equals the 0.108 a spin box holds."""
    a = {"zyx": [1.0, 0.10833, 0.10833], "source": "file"}
    b = {"zyx": [1.0, 0.108, 0.108], "source": "user"}
    assert M.same_spacing(a, b)
    assert not M.same_spacing(a, {"zyx": [1.0, 0.109, 0.109]})

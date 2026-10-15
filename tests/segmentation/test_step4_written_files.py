"""Files Step 4 writes (owner's Wound_patch_1 export, 2026-09-29).

* A mesh ``.obj`` was written with the platform encoding and newlines, and
  its header held an em-dash: every file began ``# EpiCycle cell mesh \\x97``
  (cp1252) with ``\\r\\n`` -- a UTF-8 OBJ reader chokes.  The header is
  ASCII now and the file UTF-8 with ``\\n``.
* ``save_patch_csv`` passes ``encoding="utf-8"`` explicitly.  pandas'
  default was already UTF-8, so the merged export with its "·" in
  ``meta_model`` round-trips (checked on the owner's file); the explicit
  argument is byte-identical for the ASCII CSVs Step 3 writes.
* Cells under 10 voxels get no mesh.  The export now says how many and
  logs which (``EXPORT_MESH_DONE_SKIPPED``); before, 701 cells gave "693
  meshes" and the 8 were never named.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from napari_epicycle.fusion_reconstruction.qc_features import (
    export_all_cell_meshes, export_cell_mesh_obj, save_patch_csv,
)
from napari_epicycle.segmentation import _step4_text as T


def _vol():
    vol = np.zeros((12, 12, 12), np.int32)
    vol[2:8, 2:8, 2:8] = 1          # a cube: a mesh
    vol[10, 10, 9:11] = 2           # 2 voxels: no mesh
    vol[10, 1, 1:4] = 3             # 3 voxels: no mesh
    return vol


def test_an_obj_is_ascii_utf8_with_unix_newlines(tmp_path):
    p = tmp_path / "cell_1.obj"
    export_cell_mesh_obj(_vol(), 1, p)
    raw = p.read_bytes()
    raw.decode("ascii")                         # raises on \x97
    assert b"\r\n" not in raw
    assert raw.startswith(b"# EpiCycle cell mesh - label 1\n")


def test_a_middle_dot_round_trips_through_read_csv(tmp_path):
    df = pd.DataFrame({"label": [1, 2],
                       "meta_model": ["Cell · round_02 / Cell · round_01"] * 2})
    p = tmp_path / "all.csv"
    save_patch_csv(df, p)
    back = pd.read_csv(p)                       # no encoding argument
    assert back["meta_model"].tolist() == df["meta_model"].tolist()
    p.read_bytes().decode("utf-8")


def test_an_ascii_csv_is_byte_identical_to_the_old_writer(tmp_path):
    df = pd.DataFrame({"label": [1, 2, 3], "volume": [1.5, 2.25, 3.0],
                       "qc_tier": ["good", "", "bad"]})
    old = tmp_path / "old.csv"
    df.to_csv(old, index=False)                 # the call before the fix
    new = tmp_path / "new.csv"
    save_patch_csv(df, new)
    assert new.read_bytes() == old.read_bytes()


def test_the_mesh_export_names_the_cells_it_skipped(tmp_path):
    manifest = export_all_cell_meshes(_vol(), tmp_path, "S.tif")
    assert manifest["label"].tolist() == [1]
    assert manifest.attrs["skipped_labels"] == [2, 3]


def test_the_done_sentence_names_the_skipped_count():
    line = T.EXPORT_MESH_DONE_SKIPPED.format(n=693, stack="Wound_patch_1.tiff",
                                             k=8)
    assert "693" in line and "8" in line and "10 voxels" in line


def test_the_finished_export_says_the_skipped_cells(qapp, tmp_path):
    """The subprocess's SKIPPED / SKIPPED_IDS lines reach the Export
    window's status and the log."""
    import io
    from ._step4_fakes import controller
    c = controller(tmp_path)
    lines = []
    c._export_status = lambda text, tip=None: lines.append(text)
    c._mesh_job = {"stack": "Wound_patch_1.tiff"}

    class _Proc:
        stdout = io.StringIO("DONE:693\nSKIPPED:8\n"
                             "SKIPPED_IDS:2,153,340,401,411,428,459,627\n")
        stderr = io.StringIO("")

    c._finish_mesh_export(_Proc(), 0, tmp_path, 701, tmp_path / "x.py")
    assert lines[-1] == T.say(T.EXPORT_MESH_DONE_SKIPPED, n=693,
                              stack="Wound_patch_1.tiff", k=8)
    assert any("153" in l and "627" in l for l in c.ui.log_output.lines)


def test_no_skipped_cells_keeps_the_plain_sentence(qapp, tmp_path):
    import io
    from ._step4_fakes import controller
    c = controller(tmp_path)
    lines = []
    c._export_status = lambda text, tip=None: lines.append(text)
    c._mesh_job = {"stack": "S.tif"}

    class _Proc:
        stdout = io.StringIO("DONE:3\nSKIPPED:0\nSKIPPED_IDS:\n")
        stderr = io.StringIO("")

    c._finish_mesh_export(_Proc(), 0, tmp_path, 3, tmp_path / "x.py")
    assert lines[-1] == T.say(T.EXPORT_MESH_DONE, n=3, stack="S.tif")

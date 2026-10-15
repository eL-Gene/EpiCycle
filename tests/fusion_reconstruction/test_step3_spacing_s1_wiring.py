"""S1 wired: the dialog's line and the row tails through a real panel +
controller, and the Step 3 value landing in patch_spacing.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.data_preparation import _voxel_spacing as VS  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def w(app):
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    widget = FusionReconstructionCuration(viewer=None)
    yield widget
    timer = getattr(widget.controller.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    widget.deleteLater()
    app.processEvents()


def _load(w, ws: Path, monkeypatch):
    import tifffile
    c = w.controller
    monkeypatch.setattr(c, "_start_step3_thumbs", lambda *a, **k: None)
    c.working_dir = str(ws)
    c.state_manager.active_run_id = "default"
    c.state_manager.load(str(ws))
    folder = ws / "patches"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("A.tif", "B.tif"):
        tifffile.imwrite(str(folder / name), np.zeros((4, 8, 8), np.uint8))
    c._load_stack_dir(str(folder), persist=False)
    return c


def _record(lst, name):
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    for i in range(lst.count()):
        if lst.item(i).text() == name:
            return lst.item(i).data(ROLE_RECORD)


def test_a_user_value_lands_in_the_root_file(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    c.state_manager.set_patch_spacing("A.tif", (2, .5, .5), "user")
    got = VS.read_patch_spacing_file(str(tmp_path))
    assert got["patches/A.tif"]["source"] == "user"
    # Step 4 writing the same key is seen by Step 3 at once
    VS.write_patch_spacing_entry(str(tmp_path), "patches/A.tif", (3, 1, 1),
                                 "user")
    assert c.state_manager.get_patch_spacing("A.tif")["zyx"] == [3.0, 1.0, 1.0]


def test_the_dialog_line_follows_the_patch_in_view(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    c.state_manager.set_patch_spacing("A.tif", (2, .5, .5), "user")
    c._qc_view_patch_name = "A.tif"
    assert "2.00 × 0.50 × 0.50 µm (set by you)" in c._extract_spacing_lines()[0][0]
    c._qc_view_patch_name = "B.tif"
    text, warn = c._extract_spacing_lines()[0]
    assert warn and "No voxel size" in text
    c._qc_view_patch_name = ""
    assert c._extract_spacing_lines()[0][0].startswith("2 patches: 1 with")


def test_the_session_row_shows_the_tail(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    c.state_manager.set_patch_spacing("A.tif", (2, .5, .5), "user")
    c.state_manager._state["patches"].setdefault("A.tif", {}).update(
        {"status": "reconstructed",
         "measured_spacing": {"zyx": [1.0, 0.25, 0.25]}})
    c._refresh_patch_list_colors()
    rec = _record(w.loaded_patch_list, "A.tif")
    assert rec["line3_tail"].endswith("1.00 × 0.25 × 0.25 µm")
    assert "line3_tail" not in _record(w.loaded_patch_list, "B.tif")


# ------------------------------------------------------- S1 audit folds


def _set_measured(c, name, zyx):
    c.state_manager._state["patches"].setdefault(name, {}).update(
        {"status": "reconstructed",
         "measured_spacing": {"zyx": list(zyx)}})


def test_h2_a_patch_measured_at_the_fallback_has_no_tail(w, tmp_path,
                                                         monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    c._qc_settings = dict(c._qc_settings or {}, spacing=(1.0, 0.25, 0.25))
    _set_measured(c, "B.tif", (1.0, 0.25, 0.25))     # B has no own size
    _set_measured(c, "A.tif", (1.0, 0.25, 0.25))
    c.state_manager.set_patch_spacing("A.tif", (2, .5, .5), "user")
    c._refresh_patch_list_colors()
    assert "line3_tail" not in _record(w.loaded_patch_list, "B.tif")
    assert "line3_tail" in _record(w.loaded_patch_list, "A.tif")
    # the fallback changes: B's features are now stale against it
    c._qc_settings = dict(c._qc_settings, spacing=(3.0, 1.0, 1.0))
    c._refresh_patch_list_colors()
    assert "line3_tail" in _record(w.loaded_patch_list, "B.tif")


def test_m5_the_dialog_line_uses_the_patch_extract_works_on(w, tmp_path,
                                                            monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    c.state_manager.set_patch_spacing("A.tif", (2, .5, .5), "user")
    c._qc_view_patch_name = ""
    monkeypatch.setattr(c, "_selected_curation_patch", lambda: "A.tif")
    text, warn = c._extract_spacing_lines()[0]
    assert text.startswith("This patch: 2.00") and not warn


def test_m6_the_no_size_sentence_names_the_tags_first():
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as T
    assert "tags" in T.S3_EXTRACT_SPACING_NONE
    assert "then the fallback" in T.S3_EXTRACT_SPACING_NONE


def test_m7_the_rows_are_redrawn_after_the_size_is_recorded():
    import inspect
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    src = inspect.getsource(C._on_qc_extract_finished)
    assert src.index("self._record_extraction_spacing(") < \
        src.rindex("self._refresh_patch_state_list()")


def test_m7_a_stale_tail_goes_with_the_record(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    c.state_manager.set_patch_spacing("A.tif", (2, .5, .5), "user")
    _set_measured(c, "A.tif", (1.0, 0.25, 0.25))
    c._refresh_patch_list_colors()
    assert "line3_tail" in _record(w.loaded_patch_list, "A.tif")
    c._record_extraction_spacing("A.tif")           # re-extraction finished
    c._refresh_patch_list_colors()
    assert "line3_tail" not in _record(w.loaded_patch_list, "A.tif")


def test_m2_a_failed_write_is_said_on_screen(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    shown = []
    from napari_epicycle.fusion_reconstruction import controller as CT
    monkeypatch.setattr(CT, "themed_notice",
                        lambda parent, title, msg: shown.append(msg))
    monkeypatch.setattr(VS, "write_patch_spacing_entry",
                        lambda *a, **k: (_ for _ in ()).throw(
                            PermissionError("locked")))

    class _Dlg:
        def exec_(self): return True
        def values(self): return {"zyx": [2, .5, .5]}
        def deleteLater(self): pass

    monkeypatch.setattr(c, "_make_set_spacing_dialog", lambda *a, **k: _Dlg())
    monkeypatch.setattr(CT, "themed_confirm", lambda *a, **k: True)
    assert c._open_set_spacing(["A.tif"]) is False
    assert shown and "could not be saved" in shown[0]


def test_l6_open_qc_settings_passes_the_line(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    import napari_epicycle.fusion_reconstruction.qc_extraction_dialog as D
    seen = {}

    class _Fake:
        Accepted = 1
        spacing_auto_btn = type("B", (), {"clicked": type(
            "S", (), {"connect": lambda *a: None})()})()

        def __init__(self, *a, **k):
            seen.update(k)

        def exec_(self):
            return 0

    monkeypatch.setattr(D, "QCExtractionDialog", _Fake)
    c._qc_view_patch_name = "A.tif"
    c.state_manager.set_patch_spacing("A.tif", (2, .5, .5), "user")
    c._open_qc_settings()
    assert seen["patch_spacing_lines"][0][0].startswith("This patch: 2.00")


def test_m3_a_pre_v3_adopt_writes_no_spacing_file(w, tmp_path, monkeypatch):
    import tifffile
    c = w.controller
    monkeypatch.setattr(c, "_start_step3_thumbs", lambda *a, **k: None)
    ws = tmp_path
    folder = ws / "patches"
    folder.mkdir()
    tifffile.imwrite(str(folder / "A.tif"), np.zeros((4, 8, 8), np.uint8))
    _step1_with_voxel(ws)
    (ws / "reconstruction_state.json").write_text(
        json.dumps({"schema_version": 1, "patches": {}}), encoding="utf-8")
    (ws / "step3_state.json").write_text(
        json.dumps({"schema_version": 1, "patches": {},
                    "loaded_stack_dir": str(folder)}), encoding="utf-8")
    c.working_dir = str(ws)
    c._adopt_workspace_quiet(str(ws))
    assert not VS.patch_spacing_path(str(ws)).exists()


def _step1_with_voxel(ws):
    from napari_epicycle.data_preparation._paths import step1_state_path
    path = step1_state_path(str(ws))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 2,
        "sources": [{"source_id": "sa", "path": "D:/a.tif",
                     "voxel_size": [2.0, 0.5, 0.5]}],
        "patches": [{"patch_name": "A", "relative_path": "patches/A.tif",
                     "sidecar_path": "patches/A.json", "source_id": "sa",
                     "source_kind": "raw"}]}), encoding="utf-8")

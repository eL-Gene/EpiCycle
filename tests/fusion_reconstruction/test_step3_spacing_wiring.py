"""Per-patch voxel size, WIRED (brick R7w).

* Resolution + persistence: the folder load, a run switch and step enter
  resolve every listed patch with ``read_file=False`` on the main thread and
  store it (``apply_resolved_spacing``); a ``user`` value is never
  overwritten.  Predict's and the QC extraction's workers resolve again with
  ``read_file=True`` for a patch still without one and hand the result back;
  it is stored on the MAIN thread.
* Session's rows: the spacing line per source, amber "voxels" with the
  no-tags hint as the row's tooltip.
* Set spacing…: a button under Session's list (the highlighted rows) and the
  row menu; the shared dialog with its two rows hidden; the confirm names
  the patches whose features would go stale; ``set_patch_spacing(...,
  "user")``; greyed under the Predict / Reconstruct lock with its reason on
  screen.
* The readers: ``_patch_spacing`` = the patch's own, else QC Settings', else
  None -- the single and batch extraction, the per-patch sidecar written and
  read, the annotator context, the cell width; ``measured_spacing`` at the
  extraction's finish.

A real panel + controller.  Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _step3_pr_text as P  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_rows as R  # noqa: E402

VOXEL_A = [2.0, 0.5, 0.5]        # patch A: Step 1 knows it
TAGS_B = (3.0, 0.25, 0.25)       # patch B: only its file's tags (when faked)


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


def _step1(ws: Path, voxel_b=None):
    from napari_epicycle.data_preparation._paths import step1_state_path
    state = {
        "schema_version": 2,
        "sources": [
            {"source_id": "sa", "path": "D:/a.tif", "voxel_size": VOXEL_A},
            {"source_id": "sb", "path": "D:/b.tif", "voxel_size": voxel_b},
        ],
        "patches": [
            {"patch_name": "A", "relative_path": "patches/A.tif",
             "sidecar_path": "patches/A.json", "source_id": "sa",
             "source_kind": "raw"},
            {"patch_name": "B", "relative_path": "patches/B.tif",
             "sidecar_path": "patches/B.json", "source_id": "sb",
             "source_kind": "raw"},
        ],
    }
    path = step1_state_path(str(ws))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def _load(w, ws: Path, monkeypatch, voxel_b=None):
    """A workspace with A (Step 1 knows it) and B (nobody does), loaded
    through the REAL folder load (the picture worker stubbed out)."""
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
    _step1(ws, voxel_b)
    c._load_stack_dir(str(folder), persist=False)
    return c


def _records(lst):
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    out = {}
    for i in range(lst.count()):
        it = lst.item(i)
        out[it.text()] = (it.data(ROLE_RECORD), it.toolTip())
    return out


def _highlight(w, *names):
    lst = w.loaded_patch_list
    lst.clearSelection()
    for i in range(lst.count()):
        if lst.item(i).text() in names:
            lst.item(i).setSelected(True)


class _MainThreadOnly:
    """Records the thread of every ``tifffile`` open and every voxel-tag
    read; ``main()`` = those made on the main thread."""

    def __init__(self, monkeypatch):
        import tifffile
        from napari_epicycle.data_preparation import _voxel_spacing as VS
        self.calls = []
        spy = self
        real_tf, real_read = tifffile.TiffFile, tifffile.imread

        class Counting(real_tf):
            def __init__(self, arg, *a, **k):
                spy.calls.append(("TiffFile", threading.current_thread()))
                super().__init__(arg, *a, **k)

        def imread(*a, **k):
            spy.calls.append(("imread", threading.current_thread()))
            return real_read(*a, **k)

        def tags(path):
            spy.calls.append(("tags", threading.current_thread()))
            return None

        monkeypatch.setattr(tifffile, "TiffFile", Counting)
        monkeypatch.setattr(tifffile, "imread", imread)
        monkeypatch.setattr(VS, "_read_voxel_spacing", tags)

    def main(self):
        return [k for k, t in self.calls if t is threading.main_thread()]


# ================================================================ resolution

def test_the_folder_load_resolves_and_persists(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    sm = c.state_manager
    assert sm.get_patch_spacing("A.tif") == {"zyx": VOXEL_A,
                                             "source": "step1"}
    assert sm.get_patch_spacing("B.tif") is None
    # workspace-level, in the saved file
    sm.save()
    from napari_epicycle.fusion_reconstruction import _paths
    on_disk = json.loads((tmp_path / _paths.STEP3_STATE_FILENAME)
                         .read_text(encoding="utf-8"))
    assert on_disk["patch_spacing"]["A.tif"]["source"] == "step1"


def test_the_main_thread_resolve_opens_no_file(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    spy = _MainThreadOnly(monkeypatch)
    c._resolve_listed_spacing("test")
    c._refresh_patch_list_colors()
    c._patch_spacing("B.tif")
    c._update_cell_width_line()
    c._on_annotator_reshape_context()
    c._refresh_session_spacing_gate()
    assert spy.main() == []


def test_a_user_value_survives_every_re_resolve(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    sm = c.state_manager
    sm.set_patch_spacing("A.tif", [5.0, 1.0, 1.0], "user")
    c._resolve_listed_spacing("again")
    c._load_stack_dir(str(tmp_path / "patches"), persist=False)
    assert sm.get_patch_spacing("A.tif") == {"zyx": [5.0, 1.0, 1.0],
                                             "source": "user"}
    # and the resolver's rule 1 sees it (``_patch_spacing_record``)
    assert c._patch_spacing_record("A.tif")["spacing"]["source"] == "user"
    assert c._patch_spacing("A.tif")["zyx"] == [5.0, 1.0, 1.0]


def test_step_enter_picks_up_a_voxel_size_step1_got_meanwhile(
        w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    assert c.state_manager.get_patch_spacing("B.tif") is None
    _step1(tmp_path, voxel_b=[1.0, 0.3, 0.3])
    c.on_step_enter()
    assert c.state_manager.get_patch_spacing("B.tif") == {
        "zyx": [1.0, 0.3, 0.3], "source": "step1"}
    recs = _records(w.loaded_patch_list)
    assert recs["B.tif"][0]["line3"] == P.spacing_row_text(
        {"zyx": [1.0, 0.3, 0.3], "source": "step1"})


def test_the_three_triggers_each_resolve(w, tmp_path, monkeypatch):
    from napari_epicycle.fusion_reconstruction import _paths
    c = _load(w, tmp_path, monkeypatch)
    folder = tmp_path / "patches"
    (tmp_path / _paths.STEP3_STATE_FILENAME).write_text(json.dumps({
        "schema_version": 3, "active_run_id": "default",
        "loaded_stack_dir": str(folder),
        "runs": {"default": {"patches": {}},
                 "cells": {"patches": {}, "display_name": "cells"}}}),
        encoding="utf-8")
    c.state_manager.active_run_id = ""
    c.state_manager.load(str(tmp_path))
    seen = []
    real = c._resolve_listed_spacing

    def spy(why=""):
        seen.append(why)
        return real(why)

    monkeypatch.setattr(c, "_resolve_listed_spacing", spy)
    c._load_stack_dir(str(folder), persist=False)
    c.on_step_enter()
    c._switch_active_run("cells")
    assert "folder load" in seen
    assert "step enter" in seen
    assert "run switch" in seen
    # workspace-level: the run switch keeps A's value
    assert c.state_manager.get_patch_spacing("A.tif") == {
        "zyx": VOXEL_A, "source": "step1"}


# ================================================================ the workers

@pytest.fixture
def fake_segment(monkeypatch):
    import packages.epicycle_reconstruction.logic as logic

    def fake(image, lat, ax, seg_config, axes="ZCYX", progress_cb=None):
        f = np.zeros((2, 3, 4, 4), np.float32)
        p = np.zeros((3, 4, 4), np.float32)
        return {v: (f, p, None) for v in ("xy", "xz", "yz")}

    monkeypatch.setattr(logic, "segment_views", fake)


def test_predicts_worker_reads_the_tags_and_the_main_thread_stores_them(
        w, tmp_path, monkeypatch, fake_segment):
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    from napari_epicycle.fusion_reconstruction import _step3_manifests as S
    c = _load(w, tmp_path, monkeypatch)
    sm = c.state_manager
    writes = []
    real_set = sm.set_patch_spacing

    def set_spy(*a, **k):
        writes.append(threading.current_thread())
        return real_set(*a, **k)

    monkeypatch.setattr(sm, "set_patch_spacing", set_spy)
    reads = []

    def tags(path):
        reads.append(threading.current_thread())
        return TAGS_B

    monkeypatch.setattr(VS, "_read_voxel_spacing", tags)
    out = {}

    def run():
        out["r"] = c._predict_single_patch(
            patch_name="B.tif",
            stack_path=str(tmp_path / "patches" / "B.tif"),
            lateral_path="lat", axial_path="ax",
            seg_config=c._collect_seg_config(), axes="CZYX",
            cell_width_um=0.0, spacing=None, workspace=str(tmp_path))

    th = threading.Thread(target=run, name="predict-worker")
    th.start()
    th.join()
    found = {"zyx": list(TAGS_B), "source": "file"}
    assert out["r"]["spacing_found"] == found
    assert reads and all(t is not threading.main_thread() for t in reads)
    assert writes == []                      # nothing stored by the worker
    assert sm.get_patch_spacing("B.tif") is None
    manifest = json.loads(S.prediction_manifest_path(
        c._prediction_cache_dir(), "B.tif").read_text(encoding="utf-8"))
    assert manifest["spacing"] == found      # R7's slot, filled
    c._pipeline_slices_done = 0              # what the launch sets
    c._on_serial_pred_patch_done("B.tif", out["r"])
    assert sm.get_patch_spacing("B.tif") == found
    assert writes and all(t is threading.main_thread() for t in writes)
    rec = _records(w.loaded_patch_list)["B.tif"][0]
    assert rec["line3"] == P.spacing_row_text(found)
    assert rec["line3_warn"] is False


def test_predict_takes_qc_settings_only_after_the_files_tags(
        w, tmp_path, monkeypatch, fake_segment):
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    monkeypatch.setattr(VS, "_read_voxel_spacing", lambda p: None)
    c = _load(w, tmp_path, monkeypatch)
    qc = {"zyx": [1.0, 0.25, 0.25], "source": "qc_settings"}
    r = c._predict_single_patch(
        patch_name="B.tif", stack_path=str(tmp_path / "patches" / "B.tif"),
        lateral_path="lat", axial_path="ax",
        seg_config=c._collect_seg_config(), axes="CZYX",
        cell_width_um=8.0, spacing=None, workspace=str(tmp_path),
        fallback_spacing=qc)
    assert r["spacing_found"] is None
    assert r["diameter"] == 32.0               # 8 µm at 0.25 µm per pixel


def test_predicts_launch_passes_the_own_value_and_the_fallback(
        w, tmp_path, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C

    class _W:
        made = []

        def __init__(self, fn, configs):
            self.configs = configs
            _W.made.append(self)
            raise RuntimeError("stop here")

    monkeypatch.setattr(C, "_SerialPredictionWorker", _W)
    c = _load(w, tmp_path, monkeypatch)
    c._qc_settings["spacing"] = (1.0, 0.4, 0.4)
    c.state_manager._state["ticked"] = ["A.tif", "B.tif"]
    c._refresh_patch_state_list()
    w.lateral_model_path.setText("lat")
    w.axial_model_path.setText("ax")
    c._predict_all()
    cfg = {x["patch_name"]: x for x in _W.made[-1].configs}
    assert cfg["A.tif"]["spacing"] == {"zyx": VOXEL_A, "source": "step1"}
    assert cfg["B.tif"]["spacing"] is None
    assert cfg["B.tif"]["fallback_spacing"] == {
        "zyx": [1.0, 0.4, 0.4], "source": "qc_settings"}


def _extract_kwargs_for(c, w, name, monkeypatch, tmp_path):
    """Highlight ONLY *name* in Curation (as a click does) and run the real
    ``_extract_features_current`` with the launch spied."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QListWidgetItem
    bucket = w._curation_bucket
    if bucket.count() == 0:
        for n in ("A.tif", "B.tif"):
            it = QListWidgetItem(n)
            it.setData(Qt.UserRole, n)
            bucket.addItem(it)
    bucket.clearSelection()
    for i in range(bucket.count()):
        if bucket.item(i).data(Qt.UserRole) == name:
            bucket.setCurrentItem(bucket.item(i))
            bucket.item(i).setSelected(True)
    labels = tmp_path / f"{name}_labels.tif"
    labels.write_bytes(b"x")
    c._qc_view_patch_name = ""
    launched = []
    monkeypatch.setattr(c, "_get_active_labels_path", lambda: str(labels))
    monkeypatch.setattr(c, "_get_active_source_stack_path", lambda: "")
    monkeypatch.setattr(c, "_get_active_seg_channel", lambda: 0)
    monkeypatch.setattr(c, "_get_active_prob_path", lambda: "")
    monkeypatch.setattr(c, "_busy_show", lambda *a, **k: None)
    monkeypatch.setattr(c, "_launch_qc_extract_subprocess",
                        lambda kw: launched.append(dict(kw)) or True)
    c._extract_features_current()
    return launched[0] if launched else None


def test_the_single_extraction_takes_the_highlighted_patchs_spacing(
        w, tmp_path, monkeypatch):
    """Item 12a (the driver's r7 finding): A has a user value, B none; with
    ONLY B highlighted in Curation the launch kwargs are B's, with QC
    Settings' value and the file read asked for.  (The driver highlighted
    B without clearing A on an ExtendedSelection list, so the first
    selected row -- A -- was extracted.)"""
    c = _load(w, tmp_path, monkeypatch)
    c._qc_settings["spacing"] = (1.0, 1.0, 1.0)
    c.state_manager.set_patch_spacing("A.tif", [1.5, 0.4, 0.4], "user")
    ka = _extract_kwargs_for(c, w, "A.tif", monkeypatch, tmp_path)
    assert ka["patch_name"] == "A.tif"
    assert ka["spacing"] == [1.5, 0.4, 0.4] and not ka["spacing_from_file"]
    kb = _extract_kwargs_for(c, w, "B.tif", monkeypatch, tmp_path)
    assert kb["patch_name"] == "B.tif"
    assert kb["spacing"] == [1.0, 1.0, 1.0]
    assert kb["spacing_from_file"] is True


def test_the_extraction_subprocess_hands_back_what_it_read(tmp_path,
                                                          monkeypatch):
    from napari_epicycle.fusion_reconstruction import _qc_extract_core as core
    from napari_epicycle.fusion_reconstruction import (
        _qc_extract_subprocess as sub,
    )
    found = {"zyx": list(TAGS_B), "source": "file"}
    monkeypatch.setattr(core, "run_full_qc_extraction",
                        lambda **k: {"csv_path": "x.csv", "rows": 1,
                                     "adjacency_list": [],
                                     "spacing_found": found})
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"patch_name": "B.tif"}), encoding="utf-8")
    assert sub.main(["x", str(cfg)]) == 0
    assert json.loads(Path(str(cfg) + ".spacing.json").read_text(
        encoding="utf-8")) == found


def test_the_core_reads_the_tags_only_when_asked(tmp_path, monkeypatch):
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    from napari_epicycle.fusion_reconstruction import _qc_extract_core as core
    monkeypatch.setattr(VS, "_read_voxel_spacing", lambda p: TAGS_B)
    assert core.resolve_patch_spacing_from_file(
        str(tmp_path / "B.tif"), str(tmp_path)) == {
            "zyx": list(TAGS_B), "source": "file"}
    assert core.resolve_patch_spacing_from_file("", str(tmp_path)) is None


def test_the_subprocess_result_is_stored_on_the_main_thread(
        w, tmp_path, monkeypatch):
    import napari_epicycle.fusion_reconstruction.qc_features as QF
    c = _load(w, tmp_path, monkeypatch)
    found = {"zyx": list(TAGS_B), "source": "file"}
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    Path(str(cfg) + ".spacing.json").write_text(json.dumps(found),
                                                encoding="utf-8")
    c._qc_proc = object()
    c._qc_proc_reader = None
    c._qc_proc_config_path = str(cfg)
    c._qc_proc_patch_name = "B.tif"
    c._qc_proc_workspace = c.working_dir
    c._qc_proc_csv_path = str(tmp_path / "B_features.csv")
    c._qc_proc_cancelled = False
    import pandas as pd
    monkeypatch.setattr(QF, "load_patch_csv", lambda p: pd.DataFrame())
    finished = []
    monkeypatch.setattr(c, "_on_qc_extract_finished",
                        lambda df, pn: finished.append(pn))
    monkeypatch.setattr(c, "_on_qc_csv_saved", lambda *a: None)
    c._on_qc_proc_done(0, "")
    assert finished == ["B.tif"]
    assert c.state_manager.get_patch_spacing("B.tif") == found
    assert c._extraction_spacing_used("B.tif") == found
    assert not Path(str(cfg) + ".spacing.json").exists()


def test_a_spacing_from_another_workspace_is_never_stored(
        w, tmp_path, monkeypatch):
    """A MED-1: an extraction launched in workspace A whose spacing_found
    lands after ``working_dir`` moved to B writes NOTHING (no stored value,
    no launch stash) -- the guard compares the LAUNCH workspace, not the
    current one with itself."""
    ws_a = tmp_path / "A_ws"
    ws_b = tmp_path / "B_ws"
    ws_b.mkdir(parents=True)
    c = _load(w, ws_b, monkeypatch)
    before = c.state_manager.get_patch_spacing("B.tif")
    found = {"zyx": list(TAGS_B), "source": "file"}
    c._qc_launch_spacing = {}
    c._on_qc_spacing_found("B.tif", found, str(ws_a))
    assert c.state_manager.get_patch_spacing("B.tif") == before
    assert "B.tif" not in (c._qc_launch_spacing or {})
    # the same result in its own workspace is stored
    c._on_qc_spacing_found("B.tif", found, str(ws_b))
    assert c.state_manager.get_patch_spacing("B.tif") == found


def test_the_subprocess_result_of_another_workspace_is_dropped(
        w, tmp_path, monkeypatch):
    import pandas as pd

    import napari_epicycle.fusion_reconstruction.qc_features as QF
    c = _load(w, tmp_path / "now", monkeypatch)
    before = c.state_manager.get_patch_spacing("B.tif")
    found = {"zyx": list(TAGS_B), "source": "file"}
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    Path(str(cfg) + ".spacing.json").write_text(json.dumps(found),
                                                encoding="utf-8")
    c._qc_proc = object()
    c._qc_proc_reader = None
    c._qc_proc_config_path = str(cfg)
    c._qc_proc_patch_name = "B.tif"
    c._qc_proc_workspace = str(tmp_path / "launched_here")
    c._qc_proc_csv_path = str(tmp_path / "B_features.csv")
    c._qc_proc_cancelled = False
    monkeypatch.setattr(QF, "load_patch_csv", lambda p: pd.DataFrame())
    monkeypatch.setattr(c, "_on_qc_extract_finished", lambda df, pn: None)
    monkeypatch.setattr(c, "_on_qc_csv_saved", lambda *a: None)
    c._on_qc_proc_done(0, "")
    assert c.state_manager.get_patch_spacing("B.tif") == before
    assert c._qc_proc_workspace == ""


def test_the_single_worker_emits_its_launch_workspace(monkeypatch, tmp_path):
    import napari_epicycle.fusion_reconstruction.controller as C
    from napari_epicycle.fusion_reconstruction import _qc_extract_core as core
    import napari_epicycle.fusion_reconstruction.qc_features as QF
    import pandas as pd
    found = {"zyx": list(TAGS_B), "source": "file"}
    monkeypatch.setattr(core, "run_full_qc_extraction",
                        lambda **k: {"spacing_found": found,
                                     "adjacency_list": []})
    monkeypatch.setattr(QF, "load_patch_csv", lambda p: pd.DataFrame())
    worker = C._QCFeatureWorker("l.tif", "B.tif", "b.csv",
                                spacing_from_file=True,
                                spacing_workspace=str(tmp_path / "A_ws"))
    got = []
    worker.spacing_found.connect(lambda n, f, ws: got.append((n, f, ws)))
    worker.run()
    assert got == [("B.tif", found, str(tmp_path / "A_ws"))]


# ================================================================ Session's rows

def test_the_row_line_per_source_and_the_amber_voxels():
    for source in ("step1", "file", "user"):
        rec = R.build_session_row(
            "p.tif", "unprocessed", "ZYX",
            spacing={"zyx": [1.0, 0.33, 0.33], "source": source})
        assert rec["line3"] == P.S3_ROW_SPACING_BY_SOURCE[source].format(
            z="1.00", y="0.33", x="0.33")
        assert rec["line3_warn"] is False
        assert rec["menu"] is True
        assert rec["row_tip"] is None
    none = R.build_session_row("p.tif", "unprocessed", "ZYX", spacing=None)
    assert none["line3"] == P.S3_ROW_SPACING_NONE == "voxels"
    assert none["line3_warn"] is True
    assert none["row_tip"] == P.S3_SPACING_NO_TAGS_HINT
    # a missing patch keeps its own tooltip
    gone = R.build_session_row("p.tif", "unprocessed", spacing=None,
                               missing=True)
    assert gone["row_tip"] == P.S3_PATCH_MISSING


def test_the_rows_show_the_line_and_re_apply_it_on_every_rebuild(
        w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    recs = _records(w.loaded_patch_list)
    assert recs["A.tif"][0]["line3"] == P.spacing_row_text(
        {"zyx": VOXEL_A, "source": "step1"})
    assert recs["B.tif"][0]["line3"] == "voxels"
    assert recs["B.tif"][0]["line3_warn"] is True
    assert recs["B.tif"][1] == P.S3_SPACING_NO_TAGS_HINT
    c.state_manager.set_patch_spacing("B.tif", [1.0, 0.5, 0.5], "user")
    c._refresh_patch_list_colors()
    assert _records(w.loaded_patch_list)["B.tif"][0]["line3"] == \
        P.S3_ROW_SPACING_USER.format(z="1.00", y="0.50", x="0.50")


def test_the_delegate_paints_the_third_line_and_step4_rows_are_unchanged(
        app):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont
    from napari_epicycle.ui_shell import _stack_row_delegate as D
    d = D.StackRowDelegate(D.MODE_WIDE, show_tick=False, show_eye=False)
    base = {"name": "p.tif", "status": "unprocessed", "word": "x",
            "line2": "ZYX"}
    g0 = d.layout(QRect(0, 0, 400, 0), dict(base), QFont())
    g1 = d.layout(QRect(0, 0, 400, 0), dict(base, line3="voxels",
                                            line3_warn=True), QFont())
    assert "l3w" not in g0["lines"] and "l3w" in g1["lines"]
    assert g1["lines"]["l3w"].top() >= g1["lines"]["l2"].bottom()


# ================================================================ Set spacing…

class _FakeDialog:
    made = []

    def __init__(self, names, record, accept=True, zyx=(5.0, 1.0, 1.0)):
        self.names, self.record = list(names), record
        self._accept, self._zyx = accept, list(zyx)
        _FakeDialog.made.append(self)

    def exec_(self):
        return 1 if self._accept else 0

    def values(self):
        return {"zyx": list(self._zyx)}

    def deleteLater(self):
        pass


def test_the_button_opens_the_shared_dialog_with_both_rows_hidden(
        w, tmp_path, monkeypatch):
    from napari_epicycle.ui_shell import _set_spacing_dialog as SD
    c = _load(w, tmp_path, monkeypatch)
    made = []
    monkeypatch.setattr(SD.SetSpacingDialog, "exec_",
                        lambda self: made.append(self) or 0)
    _highlight(w, "A.tif", "B.tif")
    assert w.session_set_spacing_btn.isEnabled()
    w.session_set_spacing_btn.click()
    (dlg,) = made
    assert dlg.axes_combo is None and dlg.z_orientation_combo is None
    assert dlg.windowTitle() == P.S3_SPACING_DIALOG_TITLE
    # every highlighted patch, named (not only the first / current row)
    assert dlg.desc_label.text() == P.S3_SPACING_DIALOG_DESC_NAMED.format(
        n=2, names="A.tif, B.tif")
    # prefilled from the FIRST highlighted patch
    assert [dlg.spacing_z_sb.value(), dlg.spacing_y_sb.value(),
            dlg.spacing_x_sb.value()] == VOXEL_A
    assert c.state_manager.get_patch_spacing("B.tif") is None   # cancelled


def test_nothing_highlighted_greys_the_button_with_its_reason(
        w, tmp_path, monkeypatch):
    _load(w, tmp_path, monkeypatch)
    _highlight(w)
    assert not w.session_set_spacing_btn.isEnabled()
    assert w.session_set_spacing_note.text() == P.S3_SET_SPACING_NEEDS_PATCH
    assert not w.session_set_spacing_note.isHidden()
    _highlight(w, "B.tif")
    assert w.session_set_spacing_btn.isEnabled()
    assert w.session_set_spacing_note.isHidden()


def test_the_row_menu_offers_set_spacing_for_the_row(w, tmp_path,
                                                    monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    opened = []
    monkeypatch.setattr(c, "_open_set_spacing", lambda n: opened.append(n))
    texts = []

    def pick(menu, gpos):
        acts = menu.actions()
        texts.extend(a.text() for a in acts)
        return acts[0]

    monkeypatch.setattr(c, "_exec_menu", pick)
    _highlight(w, "A.tif")
    from qtpy.QtCore import QPoint
    w._session_row_delegate.menu_requested.emit("B.tif", QPoint(0, 0))
    assert texts == [P.S3_MENU_SET_SPACING]
    assert opened == [["B.tif"]]          # not highlighted: that row alone
    _highlight(w, "A.tif", "B.tif")
    c._on_session_row_menu("B.tif", QPoint(0, 0))
    assert opened[-1] == ["A.tif", "B.tif"]   # one of the highlighted


def test_ok_stores_user_values_and_the_confirm_names_the_stale_patches(
        w, tmp_path, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    c = _load(w, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "_get_qc_extracted_patches",
                        lambda: {"A.tif"})
    monkeypatch.setattr(c, "_make_set_spacing_dialog",
                        lambda n, r: _FakeDialog(n, r))
    bodies = []
    monkeypatch.setattr(C, "themed_confirm",
                        lambda parent, title, body, **k:
                        bodies.append((title, body, k)) or False)
    assert c._open_set_spacing(["A.tif", "B.tif"]) is False   # refused
    assert bodies[-1][0] == P.S3_SPACING_CHANGE_TITLE
    assert bodies[-1][1] == P.spacing_confirm_text(["A.tif"])
    assert bodies[-1][2]["confirm_text"] == P.S3_SPACING_APPLY_BTN
    assert c.state_manager.get_patch_spacing("B.tif") is None
    monkeypatch.setattr(C, "themed_confirm",
                        lambda parent, title, body, **k: True)
    assert c._open_set_spacing(["A.tif", "B.tif"]) is True
    for n in ("A.tif", "B.tif"):
        assert c.state_manager.get_patch_spacing(n) == {
            "zyx": [5.0, 1.0, 1.0], "source": "user"}
    assert _records(w.loaded_patch_list)["B.tif"][0]["line3"] == \
        P.S3_ROW_SPACING_USER.format(z="5.00", y="1.00", x="1.00")


def test_stale_is_judged_against_what_the_features_were_measured_with(
        w, tmp_path, monkeypatch):
    """B LOW-4: measured at A, the stored value since moved to B (a Step 1
    re-resolve); the user picks A again -> NOT stale; picks B -> stale."""
    c = _load(w, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "_get_qc_extracted_patches", lambda: {"A.tif"})
    c.state_manager.set_measured_spacing(
        "A.tif", {"zyx": VOXEL_A, "source": "step1"})
    # the value every reader takes NOW (a Step 1 re-resolve moved it)
    monkeypatch.setattr(c, "_patch_spacing", lambda n, *a, **k: {
        "zyx": [4.0, 1.0, 1.0], "source": "step1"})
    assert c._spacing_stale_names(["A.tif"], VOXEL_A) == []
    assert c._spacing_stale_names(["A.tif"], [4.0, 1.0, 1.0]) == ["A.tif"]


def test_stale_falls_back_to_the_patchs_own_sidecar(w, tmp_path,
                                                     monkeypatch):
    """No measured_spacing recorded: the per-patch extraction-context
    sidecar's spacing is what the features were measured with."""
    c = _load(w, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "_get_qc_extracted_patches", lambda: {"A.tif"})
    c._qc_launch_spacing = {"A.tif": {"zyx": VOXEL_A, "source": "step1"}}
    c._write_qc_extraction_context("A.tif")
    c.state_manager.set_measured_spacing("A.tif", None)
    # the value every reader takes NOW (a Step 1 re-resolve moved it)
    monkeypatch.setattr(c, "_patch_spacing", lambda n, *a, **k: {
        "zyx": [4.0, 1.0, 1.0], "source": "step1"})
    assert c._spacing_stale_names(["A.tif"], VOXEL_A) == []
    assert c._spacing_stale_names(["A.tif"], [4.0, 1.0, 1.0]) == ["A.tif"]


def test_an_unreadable_step1_state_keeps_a_stored_step1_value(
        w, tmp_path, monkeypatch, capsys):
    """B LOW-5: a stored ``step1`` value is dropped only when Step 1's state
    was READ and no longer has the patch -- never because the file could
    not be read (one console line says so)."""
    from napari_epicycle.data_preparation._paths import step1_state_path
    c = _load(w, tmp_path, monkeypatch)
    want = {"zyx": VOXEL_A, "source": "step1"}
    assert c.state_manager.get_patch_spacing("A.tif") == want
    path = step1_state_path(str(tmp_path))
    path.write_text("{broken", encoding="utf-8")
    capsys.readouterr()
    c._resolve_listed_spacing("t")
    assert c.state_manager.get_patch_spacing("A.tif") == want
    out = capsys.readouterr().out
    assert out.count("could not be read") == 1
    # gone too: kept
    path.unlink()
    c._resolve_listed_spacing("t")
    assert c.state_manager.get_patch_spacing("A.tif") == want
    # read, and A is no longer in it: dropped
    _step1(tmp_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["patches"] = [p for p in state["patches"] if p["patch_name"] != "A"]
    path.write_text(json.dumps(state), encoding="utf-8")
    c._resolve_listed_spacing("t")
    assert c.state_manager.get_patch_spacing("A.tif") is None


def test_no_stale_patch_means_no_confirm(w, tmp_path, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    c = _load(w, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "_get_qc_extracted_patches", lambda: set())
    monkeypatch.setattr(c, "_make_set_spacing_dialog",
                        lambda n, r: _FakeDialog(n, r))
    asked = []
    monkeypatch.setattr(C, "themed_confirm",
                        lambda *a, **k: asked.append(a) or True)
    assert c._open_set_spacing(["B.tif"]) is True
    assert asked == []
    assert c.state_manager.get_patch_spacing("B.tif")["source"] == "user"


def test_a_spacing_change_fires_the_staleness_notice_for_the_patch_in_view(
        w, tmp_path, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    c = _load(w, tmp_path, monkeypatch)
    c._qc_launch_spacing = {"A.tif": {"zyx": VOXEL_A, "source": "step1"}}
    c._write_qc_extraction_context("A.tif")
    c._qc_view_patch_name = "A.tif"
    monkeypatch.setattr(c, "_get_qc_extracted_patches", lambda: {"A.tif"})
    monkeypatch.setattr(c, "_make_set_spacing_dialog",
                        lambda n, r: _FakeDialog(n, r))
    monkeypatch.setattr(C, "themed_confirm", lambda *a, **k: True)
    shown = []
    monkeypatch.setattr(c, "_maybe_show_qc_staleness_notification",
                        lambda pn="": shown.append((pn,
                                                    c._qc_extraction_stale)))
    assert c._open_set_spacing(["A.tif"]) is True
    assert shown and shown[0][0] == "A.tif" and shown[0][1]


def test_greyed_under_the_predict_lock_with_its_reason(w, tmp_path,
                                                      monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    _highlight(w, "A.tif")
    # live BEFORE the lock (else "greyed under the lock" proves nothing)
    assert w.session_set_spacing_btn.isEnabled()
    made = []
    monkeypatch.setattr(c, "_make_set_spacing_dialog",
                        lambda n, r: made.append(n) or _FakeDialog(n, r))
    c._prediction_running = True
    try:
        c._refresh_pr_lock()
        assert not w.session_set_spacing_btn.isEnabled()
        assert w.session_set_spacing_note.text() == P.S3_SET_SPACING_LOCKED
        assert c._open_set_spacing(["A.tif"]) is False
        assert made == []
        from qtpy.QtCore import QPoint
        acts = []
        monkeypatch.setattr(c, "_exec_menu",
                            lambda m, p: acts.extend(m.actions()))
        c._on_session_row_menu("A.tif", QPoint(0, 0))
        assert acts and not acts[0].isEnabled()
    finally:
        c._prediction_running = False
    c._refresh_pr_lock()
    assert w.session_set_spacing_btn.isEnabled()


# ================================================================ the readers

def test_a_none_qc_spacing_is_no_fallback_at_every_reader(
        w, tmp_path, monkeypatch):
    """The coordinator's driver finding: on a workspace where QC Settings
    were never set the key EXISTS with ``None``; the sidecar write and the
    staleness check used ``tuple(get("spacing", default))`` and raised."""
    from napari_epicycle.fusion_reconstruction import _paths
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        read_extraction_context,
    )
    c = _load(w, tmp_path, monkeypatch)
    c._qc_settings = {"spacing": None}
    assert c._qc_fallback_spacing() is None
    assert c._patch_spacing("B.tif") is None          # voxel units
    c._qc_launch_spacing = {}
    c._write_qc_extraction_context("B.tif")           # no TypeError
    assert read_extraction_context(_paths.qc_features_csv(
        c.working_dir, "B", run_id=None))["spacing"] == [1.0, 1.0, 1.0]
    assert c._check_qc_extraction_staleness("B.tif") is None
    c._write_qc_extraction_context("A.tif")
    assert c._check_qc_extraction_staleness("A.tif") is None
    assert c._on_annotator_reshape_context()["spacing"] == (1.0, 1.0, 1.0)


def test_set_spacing_acts_on_every_highlighted_row(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    opened = []
    monkeypatch.setattr(c, "_open_set_spacing", lambda n: opened.append(n))
    lst = w.loaded_patch_list
    _highlight(w, "A.tif", "B.tif")
    lst.setCurrentRow(1)                     # the current row is B ...
    for i in range(lst.count()):             # ... and both stay selected
        lst.item(i).setSelected(True)
    c._on_set_spacing()
    assert opened == [["A.tif", "B.tif"]]



def test_patch_spacing_falls_back_to_qc_settings_then_none(
        w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    c._qc_settings["spacing"] = None
    assert c._patch_spacing("B.tif") is None
    c._qc_settings["spacing"] = (1.0, 0.4, 0.4)
    assert c._patch_spacing("B.tif") == {"zyx": [1.0, 0.4, 0.4],
                                         "source": "qc_settings"}
    assert c._patch_spacing("A.tif") == {"zyx": VOXEL_A, "source": "step1"}


def test_the_single_extraction_takes_the_patchs_spacing(w, tmp_path,
                                                        monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    labels = tmp_path / "labels.tif"
    labels.write_bytes(b"x")
    got = []
    monkeypatch.setattr(c, "_get_active_labels_path", lambda: str(labels))
    monkeypatch.setattr(c, "_get_active_source_stack_path",
                        lambda: str(tmp_path / "patches" / "B.tif"))
    monkeypatch.setattr(c, "_get_active_seg_channel", lambda: 0)
    monkeypatch.setattr(c, "_get_active_prob_path", lambda: "")
    monkeypatch.setattr(c, "_busy_show", lambda *a: None)
    monkeypatch.setattr(c, "_launch_qc_extract_subprocess",
                        lambda kw: got.append(kw) or True)
    c._qc_settings["spacing"] = (1.0, 0.4, 0.4)
    for name in ("A.tif", "B.tif"):
        monkeypatch.setattr(c, "_selected_curation_patch", lambda n=name: n)
        c._extract_features_current()
    a, b = got
    assert a["spacing"] == VOXEL_A and a["spacing_from_file"] is False
    assert b["spacing"] == [1.0, 0.4, 0.4] and b["spacing_from_file"] is True
    assert b["spacing_workspace"] == str(tmp_path)


def test_the_batch_extraction_carries_one_spacing_per_patch(
        w, tmp_path, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    c = _load(w, tmp_path, monkeypatch)
    labels = tmp_path / "labels.tif"
    labels.write_bytes(b"x")
    c.state_manager._state["patches"] = {
        "A.tif": {"status": "reconstructed"},
        "B.tif": {"status": "reconstructed"}}
    monkeypatch.setattr(c, "_resolve_patch_payload", lambda n: {
        "labels_path": str(labels),
        "source_stack_path": str(tmp_path / "patches" / n),
        "seg_config": {"channels": [0, 0]}})
    made = []

    class _BW:
        def __init__(self, configs, **kw):
            made.append((configs, kw))
            raise RuntimeError("stop here")

    monkeypatch.setattr(C, "_QCFeatureBatchWorker", _BW)
    c._extract_features_all()
    configs, kw = made[-1]
    cfg = {x["patch_name"]: x for x in configs}
    assert cfg["A.tif"]["spacing"] == VOXEL_A
    assert cfg["A.tif"]["spacing_from_file"] is False
    assert cfg["B.tif"]["spacing"] is None            # voxels, unless...
    assert cfg["B.tif"]["spacing_from_file"] is True  # ...the file says


def test_the_batch_worker_uses_each_entrys_spacing(monkeypatch, tmp_path):
    import napari_epicycle.fusion_reconstruction.controller as C
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    monkeypatch.setattr(VS, "_read_voxel_spacing", lambda p: TAGS_B)
    import napari_epicycle.segmentation.feature_registry as FR
    seen = []

    def stop(registry, available, progress_cb=None):
        seen.append(available.get("spacing"))
        raise RuntimeError("enough")

    monkeypatch.setattr(FR, "run_extraction", stop)
    import tifffile
    lab = tmp_path / "l.tif"
    tifffile.imwrite(str(lab), np.ones((2, 4, 4), np.uint16))
    found = []
    # one worker per entry: the fake stops a worker at its first patch
    for entry in (
            {"labels_path": str(lab), "patch_name": "A", "csv_path": "a.csv",
             "spacing": [2.0, 0.5, 0.5]},
            {"labels_path": str(lab), "patch_name": "B", "csv_path": "b.csv",
             "spacing": None, "spacing_from_file": True,
             "source_stack_path": str(tmp_path / "B.tif"),
             "spacing_workspace": str(tmp_path)}):
        worker = C._QCFeatureBatchWorker([entry], spacing=(9.0, 9.0, 9.0))
        worker.spacing_found.connect(
            lambda n, f, ws: found.append((n, f, ws)))
        worker.run()
    assert seen == [[2.0, 0.5, 0.5], list(TAGS_B)]
    # the launch workspace rides the signal (A MED-1)
    assert found == [("B", {"zyx": list(TAGS_B), "source": "file"},
                      str(tmp_path))]


def test_the_sidecar_is_written_and_read_per_patch(w, tmp_path, monkeypatch):
    from napari_epicycle.fusion_reconstruction import _paths
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        read_extraction_context,
    )
    c = _load(w, tmp_path, monkeypatch)
    c._qc_launch_spacing = {"A.tif": {"zyx": VOXEL_A, "source": "step1"},
                            "B.tif": None}
    c._write_qc_extraction_context("A.tif")
    c._write_qc_extraction_context("B.tif")
    csv = lambda n: _paths.qc_features_csv(c.working_dir, n,  # noqa: E731
                                           run_id=None)
    assert read_extraction_context(csv("A"))["spacing"] == VOXEL_A
    assert read_extraction_context(csv("B"))["spacing"] == [1.0, 1.0, 1.0]
    # each patch is compared with ITS OWN record
    assert c._check_qc_extraction_staleness("A.tif") is None
    assert c._check_qc_extraction_staleness("B.tif") is None
    c.state_manager.set_patch_spacing("A.tif", [4.0, 1.0, 1.0], "user")
    assert "spacing changed" in c._check_qc_extraction_staleness("A.tif")
    assert c._check_qc_extraction_staleness("B.tif") is None


def test_measured_spacing_is_recorded_at_the_extractions_finish(
        w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    c._qc_launch_spacing = {"A.tif": {"zyx": VOXEL_A, "source": "step1"},
                            "B.tif": None}
    c._on_qc_batch_patch_done("A.tif", 5)
    c._on_qc_batch_patch_done("B.tif", 5)
    assert c.state_manager.get_measured_spacing("A.tif") == {
        "zyx": VOXEL_A, "source": "step1"}
    assert c.state_manager.get_measured_spacing("B.tif") is None
    # a value an extraction read from the file is what was measured with
    c._on_qc_spacing_found("B.tif", {"zyx": list(TAGS_B), "source": "file"},
                           c.working_dir)
    c._record_extraction_spacing("B.tif")
    assert c.state_manager.get_measured_spacing("B.tif")["zyx"] == \
        list(TAGS_B)


def test_the_annotator_context_takes_the_patchs_spacing(w, tmp_path,
                                                        monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    c._qc_view_patch_name = "A.tif"
    assert c._on_annotator_reshape_context()["spacing"] == tuple(VOXEL_A)
    c._qc_view_patch_name = "B.tif"
    c._qc_settings["spacing"] = (1.0, 0.4, 0.4)
    assert c._on_annotator_reshape_context()["spacing"] == (1.0, 0.4, 0.4)
    c._qc_settings["spacing"] = None
    assert c._on_annotator_reshape_context()["spacing"] == (1.0, 1.0, 1.0)


def test_the_cell_width_reads_the_same_spacing(w, tmp_path, monkeypatch):
    c = _load(w, tmp_path, monkeypatch)
    c.state_manager._state["ticked"] = ["B.tif"]
    c._refresh_patch_state_list()
    w.diameter_sb.setValue(8.0)
    c.state_manager.set_patch_spacing("B.tif", [1.0, 0.5, 0.5], "user")
    assert c._patch_spacing("B.tif") == {"zyx": [1.0, 0.5, 0.5],
                                         "source": "user"}
    from napari_epicycle.fusion_reconstruction import _step3_cell_width as CW
    seen = []
    real = CW.ensure_cell_width_lines

    def spy(ui):
        out = real(ui)

        class _L:
            def show_lines(self, px, warn):
                seen.append((px, warn))
                out.show_lines(px, warn)
        return _L()

    monkeypatch.setattr(CW, "ensure_cell_width_lines", spy)
    c._update_cell_width_line()
    assert seen[-1] == (P.S3_CELL_WIDTH_PX.format(px=16, um="0.50"), "")

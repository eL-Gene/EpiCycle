"""R6's WIRING of model provenance (brick R6w).

* Predict: the block built ONCE at launch from the snapshotted model paths
  rides every config and fills the prediction manifest's
  ``model_provenance`` slot.
* Reconstruct: ``extra_manifest`` carries the block and the prediction
  manifest's workspace-relative path, with R2c's extras intact.
* The rows: the "made by" chip under the shape line (the reconstruction
  manifest's copy first), amber-bordered "— not the active model" when it
  is not the active model, the unknown sentence for an old patch; the
  prediction manifest read once and kept until its mtime moves.
* The ⓘ: a click opens ONE details window per patch (a second raises it);
  Save writes the notes atomically into both manifests, every other key
  kept, or backfills an old patch; Save follows the lock; the windows go on
  a folder change, a run switch / workspace load, a step exit and shutdown.
* Step 4's painter is unchanged (``show_info`` off, no ``chips_top``).

Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _model_provenance as MP  # noqa: E402
from napari_epicycle.fusion_reconstruction import _paths  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_manifests as S  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_pr_text as T  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_rows as R  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _pump(app, n=3):
    from qtpy.QtCore import QCoreApplication, QEvent
    for _ in range(n):
        app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


STATUSES = {"a.tif": "unprocessed", "b.tif": "predicted",
            "c.tif": "predicted", "d.tif": "predicted",
            "e.tif": "reconstructed"}


def _block(model, ws=""):
    return MP.provenance_block(model, model, workspace=ws,
                               now="2026-10-07T10:00:00+00:00")


def _write_pred(c, name, block, **extra):
    data = {"manifest_version": 1, "patch_name": name,
            "seg_config": {"diameter": None}, "cell_width_um": 8.0,
            "keep_me": {"nested": [1, 2]}, "model_provenance": block}
    data.update(extra)
    path = S.prediction_manifest_path(c._prediction_cache_dir(), name)
    S.write_json_atomic(path, data)
    return path


def _write_recon(c, name, block):
    path = _paths.patch_run_manifest_path(
        c.working_dir, name, run_id=c._active_run_path_id())
    S.write_json_atomic(path, {
        "manifest_version": 2, "labels_shape": [4, 8, 8],
        "num_cells_final": 12, "other": "kept", "model_provenance": block})
    return path


@pytest.fixture
def full(app, tmp_path):
    import tifffile

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration(viewer=None)
    c = w.controller
    c.working_dir = str(tmp_path)
    c.state_manager.active_run_id = "default"
    c.state_manager.load(str(tmp_path))
    folder = tmp_path / "patches"
    folder.mkdir()
    for name in STATUSES:
        tifffile.imwrite(str(folder / name), np.zeros((4, 8, 8), np.uint8))
    c.loaded_stack_dir = str(folder)
    c.state_manager._state["patches"] = {
        n: {"status": s} for n, s in STATUSES.items()}
    w.lateral_model_path.setText("cyto3")
    w.axial_model_path.setText("cyto3")
    # b: made by the active model; c: by another; d: nothing recorded;
    # e: reconstructed -- its manifest's copy (cyto3) wins over a prediction
    # manifest re-written later by another model
    _write_pred(c, "b.tif", _block("cyto3"))
    _write_pred(c, "c.tif", _block("nuclei"))
    _write_pred(c, "e.tif", _block("nuclei"))
    _write_recon(c, "e.tif", _block("cyto3"))
    c._recompute_missing_patches()
    c._refresh_patch_state_list()
    yield w, c, folder
    try:
        c._close_provenance_windows()
    except Exception:
        pass
    timer = getattr(c.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    w.deleteLater()
    _pump(app)


def _rec(w, name):
    from qtpy.QtCore import Qt

    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    lst = w._patch_state_list
    for i in range(lst.count()):
        if lst.item(i).data(Qt.UserRole) == name:
            return lst.item(i).data(ROLE_RECORD)
    raise KeyError(name)


# ------------------------------------------------------------ Predict

class _Sig:
    def connect(self, *a, **k):
        pass


class _FakeThread:
    def __init__(self, *a, **k):
        self.started = self.finished = _Sig()

    def start(self):
        pass

    def quit(self):
        pass

    def isRunning(self):
        return False


class _FakeWorker:
    made = []

    def __init__(self, fn, configs):
        self.patch_started = self.patch_completed = _Sig()
        self.view_progress = self.all_completed = self.error = _Sig()
        self.cancelled = _Sig()
        self.configs = configs
        _FakeWorker.made.append(self)

    def moveToThread(self, t):
        pass

    def run(self):
        pass


def _views():
    f = np.ones((2, 3, 4, 4), np.float32)
    p = np.ones((3, 4, 4), np.float32)
    return {v: (f, p, None) for v in ("xy", "xz", "yz")}


@pytest.fixture
def fake_segment(monkeypatch):
    import packages.epicycle_reconstruction.logic as logic

    def fake(image, lat, ax, seg_config, axes="ZCYX", progress_cb=None):
        return _views()

    monkeypatch.setattr(logic, "segment_views", fake)


def test_the_block_is_built_once_at_launch_and_lands_in_the_manifest(
        full, monkeypatch, fake_segment):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, folder = full
    built = []
    real = MP.provenance_block

    def spy(lat, ax, **kw):
        built.append((lat, ax))
        return real(lat, ax, **kw)

    monkeypatch.setattr(C._mp, "provenance_block", spy)
    monkeypatch.setattr(C, "_SerialPredictionWorker", _FakeWorker)
    monkeypatch.setattr(C, "QThread", _FakeThread)
    _FakeWorker.made.clear()
    c._pr_set_item_ticked(_item(w, "a.tif"), True)
    c._pr_set_item_ticked(_item(w, "d.tif"), True)
    monkeypatch.setattr(C, "themed_confirm", lambda *a, **k: True)
    c._predict_all()
    c._prediction_running = False
    assert built == [("cyto3", "cyto3")]            # once, at launch
    assert c._serial_prediction_worker is _FakeWorker.made[-1]
    cfgs = _FakeWorker.made[-1].configs
    assert {cf["patch_name"] for cf in cfgs} == {"a.tif", "d.tif"}
    block = cfgs[0]["model_provenance"]
    assert all(cf["model_provenance"] is block for cf in cfgs)
    assert block["views"]["lateral"]["kind"] == "pretrained"
    # the worker's body writes it into the manifest's slot
    cfg = dict(cfgs[0])
    cfg["cell_width_um"] = None
    c._predict_single_patch(**cfg)
    m = json.loads(S.prediction_manifest_path(
        c._prediction_cache_dir(), cfg["patch_name"]).read_text("utf-8"))
    assert m["model_provenance"] == json.loads(json.dumps(block))
    assert m["seg_config"]                            # R2c's record kept


def test_the_launch_snapshot_wins_over_a_model_changed_mid_run(
        full, fake_segment, tmp_path):
    """The worker writes the block it was handed, not the fields' text."""
    w, c, folder = full
    block = _block("cyto3")
    w.lateral_model_path.setText("nuclei")
    c._predict_single_patch(
        patch_name="a.tif", stack_path=str(folder / "a.tif"),
        lateral_path="cyto3", axial_path="cyto3",
        seg_config=c._collect_seg_config(), axes="CZYX",
        model_provenance=block)
    m = MP.read_manifest(S.prediction_manifest_path(
        c._prediction_cache_dir(), "a.tif"))
    assert m["model_provenance"]["views"]["lateral"]["name"] == "cyto3"


# ------------------------------------------------------------ Reconstruct

def test_reconstruct_carries_the_block_and_the_manifest_path(full):
    w, c, _f = full
    seg = c._collect_seg_config()
    recorded, extra = c._recon_recorded_seg("b.tif", seg, 8.0)
    assert extra["model_provenance"] == _block("cyto3")
    rel = extra["prediction_manifest"]
    assert not os.path.isabs(rel) and rel.endswith(
        "b_prediction_manifest.json")
    assert (Path(c.working_dir) / rel).is_file()
    # R2c's extras intact
    assert extra["seg_config_source"] == "prediction"
    assert extra["cell_width_um"] == 8.0


def test_reconstruct_from_settings_keeps_r2cs_three_extras(full):
    w, c, _f = full
    # a prediction manifest with a block but no inference record (a
    # backfilled one): the settings are recorded, the block still carried
    path = S.prediction_manifest_path(c._prediction_cache_dir(), "d.tif")
    MP.backfill_manifest(path, MP.unknown_block(now="2026-10-07"))
    _r, extra = c._recon_recorded_seg("d.tif", c._collect_seg_config(), 7.5)
    assert extra["cell_width_um"] == 7.5
    assert extra["seg_config_source"] == "settings"
    assert extra["diameter_from"] == "settings"
    assert extra["model_provenance"]["views"]["lateral"]["kind"] == "unknown"
    assert extra["prediction_manifest"].endswith(
        "d_prediction_manifest.json")


def test_no_prediction_manifest_carries_nothing_new(full):
    w, c, _f = full
    _r, extra = c._recon_recorded_seg("a.tif", c._collect_seg_config(), 7.5)
    assert "model_provenance" not in extra
    assert "prediction_manifest" not in extra
    assert extra == {"cell_width_um": 7.5, "seg_config_source": "settings",
                     "diameter_from": "settings"}


def test_run_real_gets_the_merged_extras(full, monkeypatch, fake_segment):
    w, c, folder = full
    c._predict_single_patch(
        patch_name="a.tif", stack_path=str(folder / "a.tif"),
        lateral_path="cyto3", axial_path="cyto3",
        seg_config=c._collect_seg_config(), axes="CZYX",
        model_provenance=_block("cyto3"))
    seen = {}

    def run_real(**kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(c.reconstruction_service, "run_real", run_real)
    c._reconstruct_single_patch(
        patch_name="a.tif", stack_path=str(folder / "a.tif"),
        working_dir=c.working_dir, lateral_path="cyto3", axial_path="cyto3",
        seg_config=c._collect_seg_config(), recon_config=None,
        cell_width_um=8.0)
    extra = seen["extra_manifest"]
    assert extra["model_provenance"] == _block("cyto3")
    assert extra["prediction_manifest"].endswith("a_prediction_manifest.json")
    assert extra["seg_config_source"] == "prediction"
    assert "cell_width_um" in extra


# ------------------------------------------------------------ the chip

def test_the_chip_per_case(full):
    w, c, _f = full
    b, cc, d, e = (_rec(w, n) for n in ("b.tif", "c.tif", "d.tif", "e.tif"))
    assert b["chips_top"] == ((T.MADE_BY.format(short="cyto3"), False),)
    assert cc["chips_top"] == ((T.S3_MADE_BY_NOT_ACTIVE.format(
        made_by=T.MADE_BY.format(short="nuclei")), True),)
    assert d["chips_top"] == ((T.MADE_BY_UNKNOWN, False),)
    # the reconstruction manifest's copy wins over the prediction manifest
    assert e["chips_top"] == ((T.MADE_BY.format(short="cyto3"), False),)
    for rec in (b, cc, d, e):
        assert rec["info"] is True
    a = _rec(w, "a.tif")                     # never predicted: no chip
    assert "chips_top" not in a and not a.get("info")


def test_a_model_change_moves_the_amber_border(full):
    w, c, _f = full
    w.lateral_model_path.setText("nuclei")
    w.axial_model_path.setText("nuclei")
    c._refresh_patch_state_list()
    assert _rec(w, "c.tif")["chips_top"][0][1] is False
    assert _rec(w, "b.tif")["chips_top"][0][1] is True


def test_a_missing_row_keeps_the_chip_and_gives_its_slot_to_the_menu(full):
    w, c, folder = full
    os.remove(folder / "b.tif")
    c._recompute_missing_patches()
    c._refresh_patch_state_list()
    rec = _rec(w, "b.tif")
    assert rec["chips_top"] and rec["info"] is False and rec["menu"]


def test_a_row_being_predicted_has_no_chip(full):
    w, c, _f = full
    c._predicting_patches.add("b.tif")
    try:
        c._refresh_patch_state_list()
        assert "chips_top" not in _rec(w, "b.tif")
    finally:
        c._predicting_patches.discard("b.tif")


def test_build_row_and_made_by_fields_are_pure():
    blk = _block("cyto3")
    assert R.made_by_fields(blk, "cyto3", "cyto3") == (
        T.MADE_BY.format(short="cyto3"), True)
    assert R.made_by_fields(blk, "nuclei", "") == (
        T.MADE_BY.format(short="cyto3"), False)
    assert R.made_by_fields(None, "cyto3", "cyto3") == (
        T.MADE_BY_UNKNOWN, True)
    rec = R.build_row("p.tif", "predicted", made_by="made by x",
                      made_by_active=False)
    assert rec["chips_top"] == (("made by x — not the active model", True),)
    assert rec["chips"] == ()                 # the has-curation slot apart
    assert "chips_top" not in R.build_row("p.tif", "predicted")


def test_the_prediction_manifest_is_read_once_until_it_changes(
        full, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _f = full
    reads = []
    real = MP.read_manifest

    def spy(path):
        reads.append(Path(str(path)).name)
        return real(path)

    monkeypatch.setattr(C._mp, "read_manifest", spy)
    c._refresh_patch_state_list()
    c._refresh_patch_state_list()
    assert reads == []                        # cached by the first build
    path = S.prediction_manifest_path(c._prediction_cache_dir(), "c.tif")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    c._refresh_patch_state_list()
    c._refresh_patch_state_list()
    assert reads == ["c_prediction_manifest.json"]   # once per change


def test_a_paint_reads_no_file(full, monkeypatch, app):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _f = full
    reads = []
    monkeypatch.setattr(C._mp, "read_manifest",
                        lambda p: reads.append(p))
    w._patch_state_list.resize(260, 600)
    w._patch_state_list.show()
    w._patch_state_list.repaint()
    _pump(app)
    assert reads == []


# ------------------------------------------------------------ the painter

def _layout(w, name):
    from qtpy.QtCore import QRect, Qt
    lst = w._patch_state_list
    d = lst.itemDelegate()
    for i in range(lst.count()):
        idx = lst.model().index(i, 0)
        if idx.data(Qt.UserRole) == name:
            rec = idx.data(Qt.UserRole + 2)
            return d, idx, d.layout(QRect(0, 0, 260, 0), rec, lst.font())
    raise KeyError(name)


def test_the_chip_sits_under_the_shape_line_and_the_info_in_the_eye_slot(
        full):
    w, c, _f = full
    d, _i, g = _layout(w, "e.tif")
    l3 = g["lines"]["l3"]
    chip = g["chips_top"][0][0]
    # A3: the voxel-size line, when the row has one, sits between them
    above = g["lines"].get("l3n", l3)
    assert chip.top() >= above.bottom() and chip.top() - above.bottom() <= 3
    assert "info" in g and "eye" not in g
    assert g["axis"].right() < g["info"].left()
    # a chip costs the row a line -- or fits under the 80 px picture (A3)
    _d, _i, g_a = _layout(w, "a.tif")
    assert g["height"] >= g_a["height"]
    assert chip.bottom() < g["height"]


def test_a_click_on_the_info_glyph_emits_its_name(full):
    from qtpy.QtCore import QEvent, QPointF, QRect, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QStyleOptionViewItem
    w, c, _f = full
    got = []

    class Sink:
        def take(self, name):
            got.append(name)

    sink = Sink()
    d, idx, _g = _layout(w, "b.tif")
    d.info_clicked.connect(sink.take)
    opt = QStyleOptionViewItem()
    opt.rect = QRect(0, 0, 260, _g["height"])
    opt.font = w._patch_state_list.font()
    pos = QPointF(_g["info"].center())
    ev = QMouseEvent(QEvent.MouseButtonRelease, pos, pos, Qt.LeftButton,
                     Qt.LeftButton, Qt.NoModifier)
    assert d.editorEvent(ev, idx.model(), opt, idx)
    assert got == ["b.tif"]
    assert d.help_key_at(opt, idx, _g["info"].center()) == "row_info"
    assert d.help_key_at(opt, idx,
                         _g["chips_top"][0][0].center()) == "row_made_by"
    assert d._help_lines["row_info"] == T.S3_ROW_HELP_LINES["row_info"]


def test_step4s_painter_keeps_no_info_and_no_top_chip(app):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont

    from napari_epicycle.ui_shell._stack_row_delegate import (
        MODE_NARROW, StackRowDelegate,
    )
    d = StackRowDelegate(MODE_NARROW)
    rec = {"name": "s.tif", "status": "predicted", "axes": "ZYX",
           "info": True, "chips": (("x", False),)}
    g = d.layout(QRect(0, 0, 260, 0), rec, QFont())
    assert "info" not in g and "eye" in g
    assert "chips_top" not in g


# ------------------------------------------------------------ the window

def _item(w, name):
    from qtpy.QtCore import Qt
    lst = w._patch_state_list
    return next(lst.item(i) for i in range(lst.count())
                if lst.item(i).data(Qt.UserRole) == name)


def _open(w, c, app, name):
    w.info_requested.emit(name)
    _pump(app)
    return c._prov_windows[name][0]


def test_the_info_opens_one_window_and_a_second_click_raises_it(full, app):
    from napari_epicycle.fusion_reconstruction._model_provenance_dialog \
        import ModelProvenanceDialog
    w, c, _f = full
    dlg = _open(w, c, app, "b.tif")
    assert isinstance(dlg, ModelProvenanceDialog)
    assert dlg.windowTitle() == T.PROV_TITLE.format(patch="b.tif")
    assert dlg.parent() is w
    assert dlg.isVisible()
    again = _open(w, c, app, "b.tif")
    assert again is dlg and len(c._prov_windows) == 1
    other = _open(w, c, app, "c.tif")
    assert other is not dlg and len(c._prov_windows) == 2
    # its own Close deletes it; the next click opens a new one
    dlg.close()
    _pump(app)
    fresh = _open(w, c, app, "b.tif")
    assert fresh is not dlg


def test_the_window_shows_the_reconstruction_copy(full, app):
    w, c, _f = full
    dlg = _open(w, c, app, "e.tif")
    shown = {(s, k): v for s, k, v in dlg.shown_items()}
    assert shown[("lateral", "name")] == "cyto3"


def test_save_writes_the_notes_into_both_files_keeping_every_key(
        full, app, monkeypatch):
    w, c, _f = full
    writes = []
    real = S.write_json_atomic

    def spy(path, data):
        writes.append(Path(str(path)).name)
        return real(path, data)

    monkeypatch.setattr(S, "write_json_atomic", spy)
    pm = S.prediction_manifest_path(c._prediction_cache_dir(), "e.tif")
    rm = _paths.patch_run_manifest_path(c.working_dir, "e.tif",
                                        run_id=c._active_run_path_id())
    before_pm, before_rm = MP.read_manifest(pm), MP.read_manifest(rm)
    dlg = _open(w, c, app, "e.tif")
    dlg.notes_edit.setPlainText("seeded from wound-v2")
    dlg.save_btn.click()
    after_pm, after_rm = MP.read_manifest(pm), MP.read_manifest(rm)
    assert after_pm["model_provenance"]["notes"] == "seeded from wound-v2"
    assert after_rm["model_provenance"]["notes"] == "seeded from wound-v2"
    for before, after in ((before_pm, after_pm), (before_rm, after_rm)):
        before["model_provenance"]["notes"] = "seeded from wound-v2"
        assert after == before                   # nothing else moved
    assert sorted(writes) == sorted([pm.name, rm.name])   # atomic writer
    assert dlg.status_label.text() == T.PROV_SAVED
    assert not list(pm.parent.glob("*.tmp"))


def test_save_backfills_an_old_patch(full, app):
    w, c, _f = full
    pm = S.prediction_manifest_path(c._prediction_cache_dir(), "d.tif")
    assert not pm.exists()
    dlg = _open(w, c, app, "d.tif")
    dlg.notes_edit.setPlainText("old run, cyto2 I think")
    dlg.save_btn.click()
    m = MP.read_manifest(pm)
    assert m[MP.BACKFILLED_KEY] is True
    assert m["model_provenance"]["views"]["lateral"]["kind"] == "unknown"
    assert m["model_provenance"]["notes"] == "old run, cyto2 I think"
    # the chip still says unknown, and a second save edits, not re-creates
    c._refresh_patch_state_list()
    assert _rec(w, "d.tif")["chips_top"][0][0] == T.MADE_BY_UNKNOWN
    dlg.notes_edit.setPlainText("cyto2")
    dlg.save_btn.click()
    assert MP.read_manifest(pm)["model_provenance"]["notes"] == "cyto2"


def test_save_is_greyed_under_predict_and_follows_the_lock(full, app):
    w, c, _f = full
    dlg = _open(w, c, app, "b.tif")
    assert dlg.save_btn.isEnabled()
    c._prediction_running = True
    try:
        c._refresh_pr_lock()
        assert not dlg.save_btn.isEnabled()
        assert not dlg.reason_label.isHidden()
        other = _open(w, c, app, "c.tif")        # opened under the lock
        assert not other.save_btn.isEnabled()
        # a save that slips through is refused, nothing written
        assert c._prov_windows["b.tif"][1].save("x") == T.PROV_LOCKED
    finally:
        c._prediction_running = False
    c._refresh_pr_lock()
    assert dlg.save_btn.isEnabled()


def test_a_save_after_a_run_switch_writes_nothing(full, app):
    w, c, _f = full
    dlg = _open(w, c, app, "b.tif")
    saver = c._prov_windows["b.tif"][1]
    c.state_manager.active_run_id = "other-run"
    try:
        assert saver.save("x") == T.PROV_SAVE_GONE
    finally:
        c.state_manager.active_run_id = "default"
    pm = S.prediction_manifest_path(c._prediction_cache_dir(), "b.tif")
    assert MP.read_manifest(pm)["model_provenance"]["notes"] == ""
    dlg.close()


@pytest.mark.parametrize("how", ["folder", "workspace_or_run", "exit",
                                 "shutdown"])
def test_the_windows_close_on_change(full, app, how):
    from napari_epicycle.fusion_reconstruction.controller import (
        _prov_window_alive,
    )
    w, c, folder = full
    dlgs = [_open(w, c, app, n) for n in ("b.tif", "e.tif")]
    if how == "folder":
        c._load_stack_dir(str(folder), persist=False)
    elif how == "workspace_or_run":
        # the call _switch_active_run and _browse_workdir both make
        c._reset_workspace_scoped_caches()
    elif how == "exit":
        c.on_step_exit()
    else:
        c.shutdown()
    _pump(app, 5)
    assert c._prov_windows == {}
    assert not any(_prov_window_alive(d, visible=False) for d in dlgs)


def test_the_run_switch_and_the_browse_call_the_closer():
    """``_switch_active_run`` and a workspace load (``_browse_workdir`` ->
    ``_adopt_workspace_quiet``, also the shell's adopt) reset the workspace
    caches, and that reset closes the windows (the parametrised test
    above)."""
    import inspect

    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    for fn in (C._switch_active_run, C._adopt_workspace_quiet):
        assert "_reset_workspace_scoped_caches()" in inspect.getsource(fn)
    assert "_adopt_workspace_quiet(" in inspect.getsource(C._browse_workdir)
    assert "_close_provenance_windows()" in inspect.getsource(
        C._reset_workspace_scoped_caches)


def test_no_lambda_on_a_signal_in_the_wiring():
    import ast
    src = (Path(__file__).resolve().parents[2] / "napari_epicycle" /
           "fusion_reconstruction" / "controller.py").read_text("utf-8")
    tree = ast.parse(src)
    wanted = {"_on_pr_info_requested", "_save_provenance_notes",
              "_close_provenance_windows", "_live_provenance_windows",
              "_set_provenance_windows_locked", "_pr_row_made_by",
              "_pr_prediction_block", "_recon_provenance_extra"}
    seen = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in wanted:
            seen.add(fn.name)
            assert not any(isinstance(n, ast.Lambda) for n in ast.walk(fn))
    assert seen == wanted


def test_the_provenance_words_live_in_the_one_text_module():
    import importlib.util
    assert importlib.util.find_spec(
        "napari_epicycle.fusion_reconstruction._step3_provenance_text"
    ) is None
    assert MP.PT is T

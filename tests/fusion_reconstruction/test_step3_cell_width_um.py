"""Step 3's cell width in µm (R2b part 2).

* Step 3 builds ONE ``SegmentationConfig`` per run; each patch's Predict
  replaces its ``diameter`` with THAT patch's pixel number, on a copy --
  ``max(1, round(width / mean Y-X voxel size))`` (Step 4's rule).
* Auto (0) passes ``diameter=None``, as before.
* The voxel size: resolved on the main thread with ``read_file=False``
  (Step 1's registry; no file opened), again on the worker with
  ``read_file=True`` when the main thread found none.  None at all: the
  number is taken as pixels, the log names the patch, and while the width
  is not Auto the row says "No voxel size — cell width falls back to
  pixels.".
* Under the width the dim "≈ N pixels at M µm per pixel" for the
  highlighted patch (else the first patch Predict would run).
* V1 (row 20): the patches Predict runs are the TICKED ones; ``_open``
  ticks every unprocessed patch -- what the old "nothing selected" route
  ran -- so these tests keep their meaning (re-baselined in V1).
* ``diameter_px`` lands in the prediction manifest per patch.

A real panel + controller; ``segment_views`` faked.  Run one file at a time
with ``-o addopts=""``.
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

from napari_epicycle.fusion_reconstruction import _step3_manifests as S  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_pr_text as P  # noqa: E402

VOXEL_A = [2.0, 0.5, 0.5]      # patch A: Step 1 knows it
# patch B: Step 1 has no voxel size for its source


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


def _step1(ws: Path):
    from napari_epicycle.data_preparation._paths import step1_state_path
    state = {
        "schema_version": 2,
        "sources": [
            {"source_id": "sa", "path": "D:/a.tif", "voxel_size": VOXEL_A},
            {"source_id": "sb", "path": "D:/b.tif", "voxel_size": None},
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


def _open(w, ws: Path, statuses=None, ticked=None):
    import tifffile
    c = w.controller
    c.working_dir = str(ws)
    c.state_manager.active_run_id = "default"
    c.state_manager.load(str(ws))
    folder = ws / "patches"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("A.tif", "B.tif"):
        tifffile.imwrite(str(folder / name), np.zeros((4, 8, 8), np.uint8))
    c.loaded_stack_dir = str(folder)
    st = statuses or {"A.tif": "unprocessed", "B.tif": "unprocessed"}
    c.state_manager._state["patches"] = {
        n: {"status": s} for n, s in st.items()}
    # V1: tick what the old "nothing selected" route ran
    c.state_manager._state["ticked"] = sorted(
        ticked if ticked is not None
        else [n for n, s in st.items() if s == "unprocessed"])
    _step1(ws)
    c._refresh_patch_state_list()
    return c


@pytest.fixture
def fake_segment(monkeypatch):
    import packages.epicycle_reconstruction.logic as logic
    seen = []

    def fake(image, lat, ax, seg_config, axes="ZCYX", progress_cb=None):
        seen.append(seg_config)
        f = np.zeros((2, 3, 4, 4), np.float32)
        p = np.zeros((3, 4, 4), np.float32)
        return {v: (f, p, None) for v in ("xy", "xz", "yz")}

    monkeypatch.setattr(logic, "segment_views", fake)
    return seen


def _seg(c):
    return c._collect_seg_config()


def _predict(c, ws, name, um, spacing=None):
    return c._predict_single_patch(
        patch_name=name, stack_path=str(ws / "patches" / name),
        lateral_path="lat", axial_path="ax", seg_config=_seg(c),
        axes="CZYX", cell_width_um=um, spacing=spacing,
        workspace=str(ws))


def _pm(c, name):
    return json.loads(S.prediction_manifest_path(
        c._prediction_cache_dir(), name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the rule

@pytest.mark.parametrize("um,zyx,want", [
    (8.0, [2.0, 0.5, 0.5], 16),
    (8.0, [2.0, 0.4, 0.6], 16),          # the mean of Y and X
    (0.3, [2.0, 0.5, 0.5], 1),           # never below one pixel
    (10.0, [1.0, 0.33, 0.33], 30),
])
def test_the_pixel_number_is_steps4s_rule(um, zyx, want):
    from napari_epicycle.segmentation import _step4_models as M
    spacing = {"zyx": zyx, "source": "step1"}
    assert S.cell_width_px(um, spacing) == want
    assert S.cell_width_px(um, spacing) == M.cell_width_px(um, spacing)
    assert S.patch_diameter(um, spacing) == (float(want),
                                             S.DIAMETER_FROM_SPACING)


def test_auto_and_no_voxel():
    assert S.patch_diameter(0, {"zyx": VOXEL_A}) == (None, S.DIAMETER_AUTO)
    assert S.patch_diameter(None, None) == (None, S.DIAMETER_AUTO)
    assert S.patch_diameter(12.5, None) == (12.5, S.DIAMETER_AS_PIXELS)


# ---------------------------------------------------------------- per patch, in the worker

def test_each_patch_gets_its_own_pixel_number(w, tmp_path, fake_segment):
    c = _open(w, tmp_path)
    w.diameter_sb.setValue(8.0)
    seg = _seg(c)
    before = seg.diameter
    out_a = _predict(c, tmp_path, "A.tif", 8.0,
                     {"zyx": VOXEL_A, "source": "step1"})
    out_b = _predict(c, tmp_path, "B.tif", 8.0,
                     {"zyx": [2.0, 0.25, 0.25], "source": "file"})
    assert [s.diameter for s in fake_segment] == [16.0, 32.0]
    assert seg.diameter == before                 # the run config: a copy
    assert out_a["diameter"] == 16.0 and out_b["diameter"] == 32.0
    m = _pm(c, "A.tif")
    assert m["diameter_px"] == 16.0
    assert m["seg_config"]["diameter"] == 16.0
    assert m["cell_width_um"] == 8.0
    assert m["diameter_from"] == S.DIAMETER_FROM_SPACING
    assert m["rescale_rule"] == S.RESCALE_FROM_WIDTH
    # R7's slot, filled by R7w: the voxel size this Predict used
    assert m["spacing"] == {"zyx": VOXEL_A, "source": "step1"}
    assert _pm(c, "B.tif")["diameter_px"] == 32.0


def test_auto_passes_none(w, tmp_path, fake_segment, monkeypatch):
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    reads = []
    monkeypatch.setattr(VS, "_read_voxel_spacing",
                        lambda p: reads.append(p))
    c = _open(w, tmp_path)
    w.diameter_sb.setValue(0)
    out = _predict(c, tmp_path, "B.tif", 0.0, None)
    assert fake_segment[-1].diameter is None
    assert out["diameter_from"] == S.DIAMETER_AUTO
    assert _pm(c, "B.tif")["diameter_px"] is None
    # R7w re-baselined: Auto needs no voxel size, but a patch with
    # none of its own has its file's tags read on the worker (once) so the
    # value can be stored -- the worker reads the file anyway
    assert [Path(p).name for p in reads] == ["B.tif"]
    assert out["spacing_found"] is None


def test_no_voxel_size_runs_the_number_as_pixels(w, tmp_path, fake_segment):
    c = _open(w, tmp_path)
    out = _predict(c, tmp_path, "B.tif", 12.0, None)
    assert fake_segment[-1].diameter == 12.0
    assert out["diameter_from"] == S.DIAMETER_AS_PIXELS
    m = _pm(c, "B.tif")
    assert m["diameter_px"] == 12.0
    assert m["diameter_from"] == S.DIAMETER_AS_PIXELS
    # the log names the patch, on the main thread, when the patch is done
    c._serial_pred_launch_epoch = getattr(c, "_run_epoch", 0)
    c._pipeline_slices_done = 0
    c._on_serial_pred_patch_done("B.tif", out)
    assert P.S3_LOG_WIDTH_IN_PIXELS.format(patch="B.tif", px="12") in \
        w.log_output.toPlainText()


def test_the_worker_reads_the_files_tags_when_the_main_thread_had_none(
        w, tmp_path, fake_segment, monkeypatch):
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    reads = []

    def tags(path):
        reads.append((str(path), threading.current_thread().name))
        return (2.0, 0.25, 0.25)

    monkeypatch.setattr(VS, "_read_voxel_spacing", tags)
    c = _open(w, tmp_path)
    out = _predict(c, tmp_path, "B.tif", 8.0, None)
    assert out["diameter"] == 32.0
    assert out["diameter_from"] == S.DIAMETER_FROM_SPACING
    assert [Path(p).name for p, _ in reads] == ["B.tif"]


def test_a_legacy_call_keeps_the_config_as_given(w, tmp_path, fake_segment):
    """No ``cell_width_um`` (a caller from before R2b): nothing replaced."""
    c = _open(w, tmp_path)
    w.diameter_sb.setValue(9.0)
    c._predict_single_patch(
        patch_name="A.tif", stack_path=str(tmp_path / "patches" / "A.tif"),
        lateral_path="l", axial_path="a", seg_config=_seg(c), axes="CZYX")
    assert fake_segment[-1].diameter == 9.0


# ---------------------------------------------------------------- the launch, on the main thread

class _FakeThread:
    def __init__(self, *a, **k):
        from types import SimpleNamespace
        sig = SimpleNamespace(connect=lambda *a, **k: None)
        self.started = self.finished = sig

    def start(self):
        pass

    def quit(self):
        pass


class _FakeWorker:
    made = []

    def __init__(self, fn, configs):
        from types import SimpleNamespace
        sig = SimpleNamespace(connect=lambda *a, **k: None)
        self.patch_started = self.patch_completed = sig
        self.view_progress = self.all_completed = self.error = sig
        self.cancelled = sig
        self.configs = configs
        _FakeWorker.made.append(self)

    def moveToThread(self, t):
        pass

    def run(self):
        pass


def test_predict_resolves_spacing_with_read_file_false_on_the_main_thread(
        w, tmp_path, monkeypatch):
    """Spies ``resolve_stack_spacing``: every launch-time call passes
    ``read_file=False`` on the main thread.  (Whether a file is really
    opened is the next test's job.)"""
    import napari_epicycle.fusion_reconstruction.controller as C
    calls = []
    real = C.resolve_stack_spacing

    def spy(record, step1, path, ws="", *, read_file=True):
        calls.append((Path(path).name, read_file,
                      threading.current_thread() is threading.main_thread()))
        return real(record, step1, path, ws, read_file=read_file)

    monkeypatch.setattr(C, "resolve_stack_spacing", spy)
    monkeypatch.setattr(C, "_SerialPredictionWorker", _FakeWorker)
    monkeypatch.setattr(C, "QThread", _FakeThread)
    _FakeWorker.made.clear()
    c = _open(w, tmp_path)
    w.lateral_model_path.setText("lat")
    w.axial_model_path.setText("ax")
    w.diameter_sb.setValue(8.0)
    c._predict_all()
    c._prediction_running = False
    (worker,) = _FakeWorker.made
    assert c._serial_prediction_worker is worker
    cfg = {x["patch_name"]: x for x in worker.configs}
    assert cfg["A.tif"]["spacing"] == {"zyx": VOXEL_A, "source": "step1"}
    assert cfg["B.tif"]["spacing"] is None
    assert cfg["A.tif"]["cell_width_um"] == 8.0
    assert cfg["A.tif"]["workspace"] == str(tmp_path)
    # the run config holds the number typed; the worker replaces it per patch
    assert cfg["A.tif"]["seg_config"].diameter == 8.0
    launch = [x for x in calls if x[0] in ("A.tif", "B.tif")]
    assert launch and all(rf is False and main for _, rf, main in launch)


class _TiffOpens:
    """``tifffile.TiffFile`` counting the paths it opens."""

    def __init__(self, monkeypatch):
        import tifffile
        self.paths = []
        real = tifffile.TiffFile
        spy = self

        class Counting(real):
            def __init__(self, arg, *a, **k):
                spy.paths.append(Path(str(arg)).name)
                super().__init__(arg, *a, **k)

        monkeypatch.setattr(tifffile, "TiffFile", Counting)

    def count(self, name):
        return self.paths.count(name)


def test_predict_opens_no_file_for_the_spacing_on_the_main_thread(
        w, tmp_path, monkeypatch):
    """R2c audit LOW-5, a real ``tifffile`` spy: ``_predict_all`` on the
    main thread opens each patch ONCE -- the pre-existing slice count
    (``tifffile.TiffFile(...).series[0].shape``) -- and the voxel-size code
    adds no open, for a patch Step 1 knows (A) and one it does not (B,
    whose file the resolver WOULD read with ``read_file=True``)."""
    import napari_epicycle.fusion_reconstruction.controller as C
    monkeypatch.setattr(C, "_SerialPredictionWorker", _FakeWorker)
    monkeypatch.setattr(C, "QThread", _FakeThread)
    _FakeWorker.made.clear()
    c = _open(w, tmp_path)
    w.lateral_model_path.setText("lat")
    w.axial_model_path.setText("ax")
    w.diameter_sb.setValue(8.0)
    opens = _TiffOpens(monkeypatch)
    c._predict_all()
    c._prediction_running = False
    assert _FakeWorker.made, "Predict did not launch"
    assert c._serial_prediction_worker is _FakeWorker.made[-1]
    # one open per patch: the slice count, nothing from the spacing code
    assert opens.count("A.tif") == 1, opens.paths
    assert opens.count("B.tif") == 1, opens.paths
    # the spy sees the resolver's own reads (teeth): with read_file=True
    # the patch Step 1 does not know IS opened
    before = opens.count("B.tif")
    C.resolve_stack_spacing(None, None, str(tmp_path / "patches" / "B.tif"),
                            str(tmp_path), read_file=True)
    assert opens.count("B.tif") == before + 1


# ---------------------------------------------------------------- the two lines

def _lines(w):
    return w.pr_cell_width_lines


def test_the_lines_sit_under_the_width_in_its_grid(w, tmp_path):
    from qtpy.QtWidgets import QGridLayout

    from napari_epicycle.fusion_reconstruction import _step3_cell_width as CW
    _open(w, tmp_path)
    w.diameter_sb.setValue(8.0)
    lines = _lines(w)
    grid = CW._grid_holding(w.diameter_sb)
    assert isinstance(grid, QGridLayout)
    assert grid.indexOf(lines) >= 0
    r_box = grid.getItemPosition(grid.indexOf(w.diameter_sb))[0]
    r_lines, col, _rs, span = grid.getItemPosition(grid.indexOf(lines))
    assert r_lines > r_box and (col, span) == (0, 2)
    # the panel's own (R3) is used when it has one: never a second
    assert CW.ensure_cell_width_lines(w) is lines


def test_the_panel_places_the_lines_directly_under_the_width_row(w,
                                                                tmp_path):
    """R3's ``_build_pr_settings`` builds the lines itself, in the row right
    under the width and ABOVE batch size; the controller's ensure path then
    finds them and builds no second copy."""
    from napari_epicycle.fusion_reconstruction import _step3_cell_width as CW
    # built by the panel, before the controller ever asks
    lines = getattr(w, CW.ATTR, None)
    assert isinstance(lines, CW.CellWidthLines)
    grid = CW._grid_holding(w.diameter_sb)
    assert grid is not None and grid.indexOf(lines) >= 0
    r_box = grid.getItemPosition(grid.indexOf(w.diameter_sb))[0]
    r_batch = grid.getItemPosition(grid.indexOf(w.batch_sb))[0]
    r_lines, col, _rs, span = grid.getItemPosition(grid.indexOf(lines))
    assert (r_lines, col, span) == (r_box + 1, 0, 2)
    assert r_batch == r_lines + 1
    # the controller's ensure path (and a second call) reuse it
    c = _open(w, tmp_path)
    w.diameter_sb.setValue(8.0)
    c._update_cell_width_line()
    assert CW.ensure_cell_width_lines(w) is lines
    assert w.findChildren(CW.CellWidthLines) == [lines]
    assert not lines.warn_label.isHidden()


def test_the_px_line_and_the_warning(w, tmp_path):
    c = _open(w, tmp_path)
    w.diameter_sb.setValue(8.0)
    lines = _lines(w)
    assert lines.px_label.text() == P.S3_CELL_WIDTH_PX.format(px=16,
                                                              um="0.50")
    assert not lines.px_label.isHidden()
    # B (no voxel size) would run too: the warning
    assert lines.warn_label.text() == P.S3_NO_VOXEL_WARNING
    assert not lines.warn_label.isHidden()
    # only A ticked: no warning
    from qtpy.QtCore import Qt
    lst = w._patch_state_list
    for i in range(lst.count()):
        it = lst.item(i)
        it.setCheckState(Qt.Checked if it.data(0x0100) == "A.tif"
                         else Qt.Unchecked)
    assert lines.warn_label.isHidden()
    # Auto: neither line
    w.diameter_sb.setValue(0)
    assert lines.isHidden()


def test_the_highlighted_patch_wins_and_b_alone_shows_only_the_warning(
        w, tmp_path):
    c = _open(w, tmp_path, ticked=["B.tif"])
    lst = w._patch_state_list
    b = next(lst.item(i) for i in range(lst.count())
             if lst.item(i).data(0x0100) == "B.tif")
    lst.setCurrentItem(b)
    b.setSelected(True)
    w.diameter_sb.setValue(8.0)
    lines = _lines(w)
    assert lines.px_label.isHidden()          # B has no voxel size
    assert lines.warn_label.text() == P.S3_NO_VOXEL_WARNING


def test_the_lines_open_no_file(w, tmp_path, monkeypatch):
    """The main thread never opens a TIFF for the cell width (Step 4's
    ``test_step4_session_thumbs`` spy)."""
    import tifffile

    from napari_epicycle.data_preparation import _voxel_spacing as VS
    c = _open(w, tmp_path)
    opened = []
    monkeypatch.setattr(tifffile, "TiffFile",
                        lambda *a, **k: opened.append(a) or 1 / 0)
    monkeypatch.setattr(tifffile, "imread",
                        lambda *a, **k: opened.append(a) or 1 / 0)
    monkeypatch.setattr(VS, "_read_voxel_spacing",
                        lambda p: opened.append(p) or 1 / 0)
    for v in (8.0, 9.5, 0.0, 12.0):
        w.diameter_sb.setValue(v)
    c._update_cell_width_line()
    assert opened == []
    assert c._patch_spacing("B.tif", str(tmp_path / "patches" / "B.tif"),
                            c._step1_state_for_spacing()) is None
    assert opened == []


def test_the_reconstruct_launch_carries_the_width(w, tmp_path, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C

    class _Rec(_FakeWorker):
        def __init__(self, fn, configs):
            super().__init__(fn, configs)
            from types import SimpleNamespace
            self.log_message = SimpleNamespace(connect=lambda *a, **k: None)

    monkeypatch.setattr(C, "_SerialReconstructionWorker", _Rec)
    monkeypatch.setattr(C, "QThread", _FakeThread)
    _FakeWorker.made.clear()
    c = _open(w, tmp_path, {"A.tif": "predicted"}, ticked=["A.tif"])
    w.lateral_model_path.setText("lat")
    w.axial_model_path.setText("ax")
    w.diameter_sb.setValue(7.5)
    c._reconstruct_selected()
    c._reconstruction_running = False
    (worker,) = _FakeWorker.made
    assert worker.configs[0]["cell_width_um"] == 7.5


def test_the_line_refresher_never_raises(w, tmp_path, monkeypatch, capsys):
    """R2c audit LOW-6: ``_update_cell_width_line`` is a slot (the width
    box, the list's selection); a slot must never raise -- the whole body
    is guarded, not only the placing of the lines."""
    c = _open(w, tmp_path)
    w.diameter_sb.setValue(8.0)

    def boom():
        raise RuntimeError("list gone")

    monkeypatch.setattr(c, "_cell_width_patches", boom)
    c._update_cell_width_line()            # no exception
    assert "[STEP3-WIDTH]" in capsys.readouterr().out

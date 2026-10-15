"""The Patches card's rows on the shared row painter, and the ticks.

``_step3_rows.build_row`` is pure; ``refresh_patch_state_list`` sets every
item's ``ROLE_RECORD`` and keeps ``Qt.UserRole`` = the name, ``UserRole + 1``
= the status and ``text()`` = today's padded row; the list is painted by a
``StackRowDelegate`` in ``MODE_NARROW`` with Step 3's palette and no eye.
Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os

import pytest
from tests.fusion_reconstruction._platform import windows_font_metrics

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _P():
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    return P


def _R():
    from napari_epicycle.fusion_reconstruction import _step3_rows as R
    return R


def _bare():
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    return w


@pytest.fixture(scope="module")
def ui(app):
    w = _bare()
    yield w
    w.deleteLater()


MANIFEST = {"labels_shape": [124, 240, 240], "num_cells_final": 4653,
            "num_cells_raw": 4700}


# ------------------------------------------------------------ build_row

def test_an_unprocessed_row_is_a_circle_and_tickable():
    rec = _R().build_row("p.tif", "unprocessed", "CZYX")
    assert rec["icon"] == "mdi6.circle-outline"
    assert rec["word"] == "unprocessed" and rec["dots"] == "○"
    assert rec["axes"] == "CZYX"
    assert rec["tickable"] is True
    assert rec["shape_text"] == ""            # no manifest, no shape line
    assert rec["warns_narrow"] == () and rec["notes_narrow"] == ()
    assert not rec.get("menu")


def test_the_shape_line_comes_from_the_manifest():
    P = _P()
    rec = _R().build_row("p.tif", "reconstructed", "ZCYX", MANIFEST)
    assert rec["shape_text"] == P.S3_ROW_SHAPE.format(
        z=124, y=240, x=240, n="4,653")
    assert rec["shape_text"] == "124×240×240 · 4,653 labels"
    no_count = _R().shape_line({"labels_shape": [5, 6, 7]})
    assert no_count == "5×6×7"
    assert _R().shape_line({"labels_shape": "bad"}) == ""
    assert _R().shape_line(None) == ""


@pytest.mark.parametrize("status", ["accepted", "finalized"])
def test_a_protected_row_says_so_and_cannot_be_ticked(status):
    P = _P()
    rec = _R().build_row("p.tif", status, "CZYX", MANIFEST)
    assert rec["tickable"] is False
    assert rec["notes_narrow"] == (P.S3_ROW_PROTECTED,)
    assert rec["row_tip"] == P.S3_ROW_PROTECTED_TIP.format(status=status)
    assert rec["dim_name"] is False       # the mockup: a protected name reads


def test_a_missing_row_keeps_its_place_amber_and_untickable():
    P = _P()
    rec = _R().build_row("p.tif", "reconstructed", "CZYX", MANIFEST,
                         missing=True)
    assert rec["tickable"] is False
    assert rec["warns_narrow"] == (P.S3_WARN_SIGN + P.S3_PATCH_MISSING,)
    assert rec["menu"] is True                 # Forget lives in its menu
    assert rec["icon"] == "mdi6.alert-outline"
    assert rec["icon_colour"] == _R().WARN_COLOUR
    assert rec["dim_name"] is True
    assert rec["row_tip"] == P.S3_PATCH_MISSING


@pytest.mark.parametrize("kind,word", [("segmenting", "segmenting..."),
                                        ("reconstructing",
                                         "reconstructing...")])
def test_a_running_row_shows_the_shared_running_words(kind, word):
    rec = _R().build_row("p.tif", "predicted", "CZYX", running=kind)
    assert rec["word"] == word and rec["dots"] == ""
    assert rec["icon"] == "mdi6.timer-sand"


def test_a_ticked_curating_row_carries_the_has_curation_chip():
    P = _P()
    rec = _R().build_row("p.tif", "curating", "CZYX", ticked=True)
    assert rec["chips"] == ((P.S3_CHIP_HAS_CURATION, True),)
    assert _R().build_row("p.tif", "curating", "CZYX")["chips"] == ()


# ------------------------------------------------------------ the list

def test_the_list_is_painted_by_the_shared_painter(ui):
    from napari_epicycle.ui_shell._stack_row_delegate import (
        MODE_NARROW, StackListWidget, StackRowDelegate,
    )
    lst = ui._patch_state_list
    assert isinstance(lst, StackListWidget)
    d = lst.itemDelegate()
    assert isinstance(d, StackRowDelegate)
    assert d.mode == MODE_NARROW
    assert d._show_eye is False                   # View is the foot's
    assert d._lock_property == "step3Locked"
    # gold ticks, amber selection tint, PANEL_WARN text
    assert d._palette["accent"].name() == "#d4a843"
    assert d._palette["warn"].name() == "#e6c26a"
    assert d._palette["sel_bg"].alpha() == 64


def test_every_rebuild_sets_the_record_and_keeps_name_status_text(ui):
    from qtpy.QtCore import Qt

    from napari_epicycle.fusion_reconstruction.ui import (
        _format_patch_state_row,
    )
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    rows = {"a.tif": "unprocessed", "b.tif": "reconstructed",
            "c.tif": "finalized"}
    ui.refresh_patch_state_list(dict(rows), axes_by_patch={"b.tif": "ZCYX"},
                                info_by_patch={"b.tif": MANIFEST})
    lst = ui._patch_state_list
    assert lst.count() == 3
    for i in range(3):
        item = lst.item(i)
        name = item.data(Qt.UserRole)
        assert item.data(Qt.UserRole + 1) == rows[name]
        assert item.text() == _format_patch_state_row(
            name, rows[name], "ZCYX" if name == "b.tif" else None,
            {"unprocessed": "○", "reconstructed": "●●",
             "finalized": "●●●"}[rows[name]], 5, 13)
        rec = item.data(ROLE_RECORD)
        assert rec["name"] == name
    b = [lst.item(i) for i in range(3) if lst.item(i).data(Qt.UserRole)
         == "b.tif"][0]
    assert b.data(ROLE_RECORD)["shape_text"] == "124×240×240 · 4,653 labels"


def test_a_protected_row_stays_selectable_for_view(ui):
    from qtpy.QtCore import Qt
    ui.refresh_patch_state_list({"c.tif": "finalized"})
    item = ui._patch_state_list.item(0)
    assert item.flags() & Qt.ItemIsSelectable
    assert item.toolTip() == _P().S3_ROW_PROTECTED_TIP.format(
        status="finalized")


@windows_font_metrics
def test_the_rows_are_taller_than_a_text_line_and_follow_the_record(app,
                                                                     ui):
    """The painter sizes the row from the record: a row with the shape
    line and the protected note is taller than a bare one."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QStyleOptionViewItem
    ui.refresh_patch_state_list({"a.tif": "unprocessed",
                                 "c.tif": "finalized"},
                                info_by_patch={"c.tif": MANIFEST})
    lst = ui._patch_state_list
    d = lst.itemDelegate()
    opt = QStyleOptionViewItem()
    opt.font = lst.font()
    hs = {}
    for i in range(lst.count()):
        idx = lst.model().index(i, 0)
        hs[idx.data(Qt.UserRole)] = d.sizeHint(opt, idx).height()
    assert hs["c.tif"] > hs["a.tif"] > 20


def test_the_row_help_lines_are_steps_3s(ui):
    d = ui._patch_state_list.itemDelegate()
    assert d._help_lines == _P().S3_ROW_HELP_LINES


# ------------------------------------------------------------ the ticks
# (row 20) a real panel + controller on a temporary workspace

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


class _FakeWorker:
    made = []

    def __init__(self, fn, configs):
        self.patch_started = self.patch_completed = _Sig()
        self.view_progress = self.all_completed = self.error = _Sig()
        self.log_message = _Sig()
        self.configs = configs
        _FakeWorker.made.append(self)

    def moveToThread(self, t):
        pass

    def run(self):
        pass


@pytest.fixture
def full(app, tmp_path):
    import numpy as np
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
    statuses = {"a.tif": "unprocessed", "b.tif": "predicted",
                "c.tif": "curating", "d.tif": "finalized",
                "e.tif": "unprocessed"}
    for name in statuses:
        tifffile.imwrite(str(folder / name), np.zeros((4, 8, 8), np.uint8))
    c.loaded_stack_dir = str(folder)
    c.state_manager._state["patches"] = {
        n: {"status": s} for n, s in statuses.items()}
    w.lateral_model_path.setText("cyto3")
    w.axial_model_path.setText("cyto3")
    c._recompute_missing_patches()
    c._refresh_patch_state_list()
    yield w, c, folder
    timer = getattr(c.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    w.deleteLater()
    app.processEvents()


def _item(w, name):
    from qtpy.QtCore import Qt
    lst = w._patch_state_list
    return next(lst.item(i) for i in range(lst.count())
                if lst.item(i).data(Qt.UserRole) == name)


def _tick(w, *names):
    from qtpy.QtCore import Qt
    for name in names:
        _item(w, name).setCheckState(Qt.Checked)


def _ticked(w):
    return set(w.controller._ticked_patch_names())


def test_nothing_ticked_greys_both_and_says_so(full):
    w, c, _f = full
    assert _ticked(w) == set()
    c._refresh_pr_foot_gates(keep_status=False)
    assert not w.predict_selected_btn.isEnabled()
    assert not w.reconstruct_selected_btn.isEnabled()
    assert w.pipeline_status_label.toolTip() == _P().S3_GATE_NOTHING_TICKED
    _tick(w, "a.tif")                         # live with a tick
    assert w.predict_selected_btn.isEnabled()


def test_predict_runs_the_ticks_and_nothing_without_one(full, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _f = full
    monkeypatch.setattr(C, "_SerialPredictionWorker", _FakeWorker)
    monkeypatch.setattr(C, "QThread", _FakeThread)
    _FakeWorker.made.clear()
    c._predict_all()                          # nothing ticked
    assert _FakeWorker.made == []             # the old route is gone
    _tick(w, "a.tif")
    c._predict_all()
    c._prediction_running = False
    (worker,) = _FakeWorker.made
    assert [x["patch_name"] for x in worker.configs] == ["a.tif"]


def test_reconstruct_runs_the_ready_ticks(full, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _f = full
    monkeypatch.setattr(C, "_SerialReconstructionWorker", _FakeWorker)
    monkeypatch.setattr(C, "QThread", _FakeThread)
    _FakeWorker.made.clear()
    _tick(w, "a.tif", "b.tif")
    c._reconstruct_selected()
    c._reconstruction_running = False
    (worker,) = _FakeWorker.made
    assert [x["patch_name"] for x in worker.configs] == ["b.tif"]


def test_the_ticks_are_saved_per_run_and_survive_every_rebuild(full):
    w, c, _f = full
    _tick(w, "a.tif", "c.tif")
    assert c.state_manager.get_ticked() == ["a.tif", "c.tif"]
    # a status change and a rebuild (the busiest caller path)
    c.state_manager._state["patches"]["a.tif"]["status"] = "predicted"
    c._refresh_patch_state_list()
    assert _ticked(w) == {"a.tif", "c.tif"}
    # the run's stored list is what a rebuild shows (a run switch)
    c.state_manager._state["ticked"] = ["e.tif"]
    c._refresh_patch_state_list()
    assert _ticked(w) == {"e.tif"}


def test_ticks_are_kept_after_a_run(full, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _f = full
    monkeypatch.setattr(C, "_SerialPredictionWorker", _FakeWorker)
    monkeypatch.setattr(C, "QThread", _FakeThread)
    _tick(w, "a.tif")
    c._predict_all()
    c.state_manager._state["patches"]["a.tif"]["status"] = "predicted"
    c._prediction_running = False
    c._refresh_step3_lock()
    c._refresh_patch_state_list()
    assert _ticked(w) == {"a.tif"}


def test_a_protected_row_can_be_unticked_never_ticked(full):
    from qtpy.QtCore import Qt
    w, c, _f = full
    d = _item(w, "d.tif")
    assert not (d.flags() & Qt.ItemIsUserCheckable)
    assert _item(w, "a.tif").flags() & Qt.ItemIsUserCheckable
    d.setCheckState(Qt.Checked)               # a programmatic / Space tick
    assert _ticked(w) == set()                # taken back
    # a tick stored before the patch was finalized can be taken off
    c.state_manager._state["ticked"] = ["d.tif"]
    c._refresh_patch_state_list()
    d = _item(w, "d.tif")
    assert d.flags() & Qt.ItemIsUserCheckable
    d.setCheckState(Qt.Unchecked)
    assert _ticked(w) == set()
    assert c.state_manager.get_ticked() == []


def test_the_link_ticks_every_unprocessed_patch(full):
    w, c, _f = full
    w.pr_tick_all_link.clicked.emit()
    assert _ticked(w) == {"a.tif", "e.tif"}
    assert c.state_manager.get_ticked() == ["a.tif", "e.tif"]


def test_ctrl_a_ticks_every_tickable_row(app, full):
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    w, c, _f = full
    QTest.keyClick(w._patch_state_list, Qt.Key_A, Qt.ControlModifier)
    app.processEvents()
    assert _ticked(w) == {"a.tif", "b.tif", "c.tif", "e.tif"}  # not d


def test_the_rerun_confirm_names_each_patch_and_its_status(full,
                                                           monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _f = full
    seen = []

    def confirm(parent, title, msg, **k):
        seen.append((title, msg, k))
        return False

    monkeypatch.setattr(C, "themed_confirm", confirm)
    monkeypatch.setattr(C, "_SerialPredictionWorker", _FakeWorker)
    monkeypatch.setattr(C, "QThread", _FakeThread)
    _tick(w, "a.tif", "b.tif", "c.tif")
    c._predict_all()
    (title, msg, kw) = seen[0]
    P = _P()
    assert title == P.S3_REPREDICT_TITLE
    assert P.S3_RERUN_LINE.format(patch="b.tif", status="predicted") in msg
    assert P.S3_RERUN_LINE.format(patch="c.tif", status="curating") in msg
    assert "a.tif" not in msg                 # no result yet
    assert kw.get("accent") == "red"


def test_ticking_a_curating_row_puts_the_has_curation_chip_on(full):
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    w, c, _f = full
    assert _item(w, "c.tif").data(ROLE_RECORD)["chips"] == ()
    _tick(w, "c.tif")
    assert _item(w, "c.tif").data(ROLE_RECORD)["chips"] == (
        (_P().S3_CHIP_HAS_CURATION, True),)


def test_the_ticked_lines_read_the_ticks(full):
    w, c, _f = full
    c.state_manager._state["ticked"] = ["d.tif"]
    c._refresh_patch_state_list()
    _tick(w, "a.tif", "b.tif")
    assert w.pr_ticked_predict_label.text() == \
        "Predict: 3 ticked · 1 protected"
    assert w.pr_ticked_recon_label.text() == \
        "Reconstruct: 1 ready of 3 ticked"


def test_the_highlighted_row_survives_a_rebuild(full):
    w, c, _f = full
    lst = w._patch_state_list
    it = _item(w, "b.tif")
    lst.setCurrentItem(it)
    it.setSelected(True)
    c._refresh_patch_state_list()
    sel = lst.selectedItems()
    assert [x.data(0x0100) for x in sel] == ["b.tif"]


# ------------------------------------------------------------ missing files

def test_a_missing_patch_keeps_its_row_amber_and_untickable(full):
    from qtpy.QtCore import Qt

    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    w, c, folder = full
    (folder / "b.tif").unlink()
    c._recompute_missing_patches()
    c._refresh_patch_state_list()
    b = _item(w, "b.tif")
    rec = b.data(ROLE_RECORD)
    assert rec["missing"] and not rec["tickable"]
    assert rec["warns_narrow"] == (_P().S3_WARN_SIGN
                                   + _P().S3_PATCH_MISSING,)
    assert not (b.flags() & Qt.ItemIsUserCheckable)
    # and in the Curation bucket (b is predicted: not listed there; c is)
    (folder / "c.tif").unlink()
    c._recompute_missing_patches()
    c._refresh_patch_state_list()
    bucket = w._curation_bucket
    texts = {bucket.item(i).data(Qt.UserRole): bucket.item(i).text()
             for i in range(bucket.count())}
    assert texts["c.tif"].endswith(_P().S3_PATCH_MISSING)
    assert not texts["d.tif"].endswith(_P().S3_PATCH_MISSING)


def test_a_click_on_a_missing_patch_in_curation_is_refused(full,
                                                         monkeypatch):
    w, c, folder = full
    (folder / "c.tif").unlink()
    c._recompute_missing_patches()
    c._refresh_patch_state_list()
    called = []
    monkeypatch.setattr(c, "_selected_curation_patch", lambda: "c.tif")
    monkeypatch.setattr(c, "_switch_to_curation_patch",
                        lambda name: called.append(name))
    c._on_curation_patch_clicked(None)
    assert called == []
    assert w.curation_status_label.text() == _P().S3_PATCH_MISSING


def test_missing_is_one_stat_per_patch_on_the_triggers_not_per_rebuild(
        full, monkeypatch):
    import os as _os
    w, c, folder = full
    calls = []
    real = _os.path.exists

    def spy(p):
        calls.append(str(p))
        return real(p)

    monkeypatch.setattr(_os.path, "exists", spy)
    c._refresh_patch_state_list()
    assert not [p for p in calls if p.startswith(str(folder))]
    c._recompute_missing_patches()
    assert len([p for p in calls if p.startswith(str(folder))]) == 5


def test_predict_finds_a_file_gone_since_the_load(full, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, folder = full
    monkeypatch.setattr(C, "_SerialPredictionWorker", _FakeWorker)
    monkeypatch.setattr(C, "QThread", _FakeThread)
    _FakeWorker.made.clear()
    _tick(w, "a.tif", "e.tif")
    (folder / "e.tif").unlink()
    c._predict_all()
    c._prediction_running = False
    (worker,) = _FakeWorker.made
    assert [x["patch_name"] for x in worker.configs] == ["a.tif"]
    assert "e.tif" in c._missing_patches

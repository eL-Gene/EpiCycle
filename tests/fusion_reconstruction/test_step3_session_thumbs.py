"""Session's patch rows with their pictures ((R5), item 3, rows 3, 23, 30; brick V2).

Session's ``loaded_patch_list`` is painted by the shared row painter in
``MODE_WIDE`` with ``show_tick=False`` (and no eye): thumbnail · name and the
status dots + word · the line "{axes} · {z}×{y}×{x} · {c} ch · {n} labels"
(each part only when known, never from a TIFF opened on the main thread).
The pictures come from V2a's engine on the shared worker; three triggers
(the folder load, a run switch, step enter), one batch per trigger, a stale
picture dropped, stopped on step exit and at shutdown.  ONE channel draws
grey, two draw magenta + green -- proven end to end through the real list.
Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _D():
    from napari_epicycle.ui_shell import _stack_row_delegate as D
    return D


def _P():
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    return P


def _R():
    from napari_epicycle.fusion_reconstruction import _step3_rows as R
    return R


def pump(app, n=5):
    for _ in range(n):
        app.processEvents()


# ------------------------------------------------------------ the painter

def _layout(delegate, rec, width=400):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont
    return delegate.layout(QRect(0, 0, width, 0), rec, QFont())


def test_show_tick_off_takes_the_tick_away_and_moves_the_picture_left(app):
    D = _D()
    rec = {"name": "p.tif", "word": "curating", "axes": "CZYX"}
    with_tick = _layout(D.StackRowDelegate(D.MODE_WIDE), rec)
    no_tick = _layout(D.StackRowDelegate(D.MODE_WIDE, show_tick=False), rec)
    assert "tick" in with_tick and "tick" not in no_tick
    assert no_tick["thumb"].left() < with_tick["thumb"].left()
    assert no_tick["thumb"].left() == D.PAD_X_WIDE


def test_step4s_wide_row_is_unchanged_by_the_new_keywords(app):
    """Step 4 passes none of them: tick, eye and menu all there, at the
    same places (the paint golden holds the pixels)."""
    D = _D()
    rec = {"name": "p.tif", "word": "segmented", "axes": "CZYX"}
    g = _layout(D.StackRowDelegate(D.MODE_WIDE), rec)
    assert {"tick", "thumb", "eye", "menu"} <= set(g)
    assert g["thumb"].left() == D.PAD_X_WIDE + D.TICK + 8
    assert g["menu"].right() < 400 and g["eye"].right() < g["menu"].left()
    assert g["mid"].right() < g["eye"].left()


def test_a_wide_row_without_the_eye_has_no_eye_and_a_menu_only_when_asked(
        app):
    D = _D()
    d = D.StackRowDelegate(D.MODE_WIDE, show_tick=False, show_eye=False)
    plain = _layout(d, {"name": "p.tif", "word": "curating"})
    assert "eye" not in plain and "menu" not in plain
    # the middle runs to the right edge
    assert plain["mid"].right() >= 400 - D.PAD_X_WIDE - 2
    asked = _layout(d, {"name": "p.tif", "word": "curating", "menu": True})
    assert "eye" not in asked and "menu" in asked
    assert asked["mid"].right() < asked["menu"].left()


def test_a_record_line2_replaces_the_template(app):
    """Step 3 hands its own line 2 (only the parts it knows); Step 4's
    template is read only when the record carries none."""
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont, QImage, QPainter
    D = _D()

    class _Ix:
        def __init__(self, rec):
            self.rec = rec

        def data(self, role):
            from qtpy.QtCore import Qt
            return self.rec if role == D.ROLE_RECORD else (
                Qt.Unchecked if role == Qt.CheckStateRole else None)

        def flags(self):
            from qtpy.QtCore import Qt
            return Qt.ItemIsEnabled

    drawn = []

    class _Spy(D.StackRowDelegate):
        pass

    d = _Spy(D.MODE_WIDE, show_tick=False, show_eye=False,
             wide_line2="{axes} · {c} ch{spacing}")
    img = QImage(400, 60, QImage.Format_ARGB32)
    p = QPainter(img)
    real = p.drawText

    from qtpy.QtWidgets import QStyleOptionViewItem
    opt = QStyleOptionViewItem()
    opt.rect = QRect(0, 0, 400, 40)
    opt.font = QFont()

    class _P2:
        """A painter proxy that records the texts drawn."""
        def __getattr__(self, n):
            return getattr(p, n)

        def drawText(self, *a):
            drawn.append(a[-1])
            return real(*a)

    d.paint(_P2(), opt, _Ix({"name": "p.tif", "word": "curating",
                             "axes": "CZYX", "line2": "CZYX · 2 ch"}))
    d.paint(_P2(), opt, _Ix({"name": "p.tif", "word": "curating",
                             "axes": "ZYX", "n_channels": 1}))
    p.end()
    assert "CZYX · 2 ch" in drawn
    assert "ZYX · 1 ch" in drawn          # the template, as before


def test_the_picture_has_its_own_tooltip_only_when_the_row_carries_one(app):
    from qtpy.QtCore import QPoint, QRect
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import QListWidget, QListWidgetItem
    from qtpy.QtWidgets import QStyleOptionViewItem
    D = _D()
    lw = QListWidget()
    d = D.StackRowDelegate(D.MODE_WIDE, lw, show_tick=False, show_eye=False)
    lw.setItemDelegate(d)
    it = QListWidgetItem("p.tif")
    it.setData(D.ROLE_RECORD, {"name": "p.tif"})
    lw.addItem(it)
    opt = QStyleOptionViewItem()
    opt.rect = QRect(0, 0, 400, 40)
    opt.font = QFont()
    idx = lw.model().index(0, 0)
    g = d.layout(opt.rect, {"name": "p.tif"}, opt.font)
    centre = g["thumb"].center()
    assert d.help_key_at(opt, idx, centre) is None
    it.setData(D.ROLE_THUMB_TIP, "The middle slice.")
    assert d.help_key_at(opt, idx, centre) == "row_thumb"
    assert d.help_key_at(opt, idx, QPoint(300, 10)) is None
    lw.deleteLater()


def test_an_empty_line2_leaves_no_blank_line_before_the_amber_one(app):
    D = _D()
    d = D.StackRowDelegate(D.MODE_WIDE, show_tick=False, show_eye=False)
    warn = ("x",)
    g = _layout(d, {"name": "p", "line2": "", "warns_wide": warn})
    assert g["warns"][0][0].top() == g["lines"]["l1"].bottom() + 1 + D.GAP
    full = _layout(d, {"name": "p", "line2": "CZYX", "warns_wide": warn})
    assert full["warns"][0][0].top() > g["warns"][0][0].top()
    # Step 4's records carry no "line2": its line 2 is always laid out
    s4 = _layout(D.StackRowDelegate(D.MODE_WIDE), {"name": "p",
                                                   "warns_wide": warn})
    assert s4["lines"]["l2"].height() > 0


def _painted(delegate, pixmap):
    """The 32 px square the delegate paints for *pixmap*, as an image."""
    from qtpy.QtCore import QRect, Qt
    from qtpy.QtGui import QColor, QImage, QPainter
    img = QImage(40, 40, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    delegate._paint_thumb(p, QRect(4, 4, 32, 32), pixmap,
                          keep_aspect=delegate._keep_aspect)
    p.end()
    return img


def test_a_slice_that_is_not_square_keeps_its_aspect_on_step3_rows(app):
    from qtpy.QtGui import QColor, QPixmap
    D = _D()
    wide = QPixmap(64, 32)                    # 2:1, as a 240 x 120 slice
    wide.fill(QColor("#00ff00"))
    kept = _painted(D.StackRowDelegate(D.MODE_WIDE, keep_aspect=True), wide)
    # fitted: green in the middle row, the dark square above and below
    assert kept.pixelColor(20, 20).green() == 255
    assert kept.pixelColor(20, 6).green() < 100
    # Step 4 (the default) fills the square as before
    filled = _painted(D.StackRowDelegate(D.MODE_WIDE), wide)
    assert filled.pixelColor(20, 6).green() == 255


# ------------------------------------------------------------ the record

def test_a_session_record_says_only_what_is_known():
    P, R = _P(), _R()
    rec = R.build_session_row("p.tif", "curating", "CZYX")
    assert rec["line2"] == "CZYX"                    # channels unknown
    assert rec["word"] == "curating" and rec["dots"] == "●●"
    rec = R.build_session_row("p.tif", "unprocessed", "ZYX")
    assert rec["line2"] == P.S3_SESSION_CHANNELS.format(axes="ZYX", c=1) \
        or rec["line2"] == "ZYX · 1 ch"
    assert rec["line2"] == "ZYX · 1 ch"
    rec = R.build_session_row("p.tif", "reconstructed", "ZCYX",
                              source_shape=(81, 2, 256, 256),
                              manifest_info={"labels_shape": [81, 256, 256],
                                             "num_cells_final": 1650})
    assert rec["line2"] == "ZCYX · 81×256×256 · 2 ch · 1,650 labels"
    assert rec["n_channels"] == 2
    # a shape that does not fit the axes is not believed
    rec = R.build_session_row("p.tif", "unprocessed", "CZYX",
                              source_shape=(5, 6, 7))
    assert rec["line2"] == "CZYX"
    # the manifest alone gives the shape and the labels, not the channels
    rec = R.build_session_row("p.tif", "reconstructed", "CZYX",
                              manifest_info={"labels_shape": [4, 5, 6]})
    assert rec["line2"] == "CZYX · 4×5×6"


def test_a_missing_session_row_is_amber_with_the_forget_menu():
    P, R = _P(), _R()
    rec = R.build_session_row("p.tif", "reconstructed", "CZYX", missing=True)
    assert rec["warns_wide"] == (P.S3_WARN_SIGN + P.S3_PATCH_MISSING,)
    assert rec["menu"] is True and rec["dim_name"] is True
    assert rec["row_tip"] == P.S3_PATCH_MISSING


# ------------------------------------------------------------ the list

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


@pytest.fixture
def ui(app):
    w = _bare()
    yield w
    w.deleteLater()
    app.processEvents()


def _fill(ui, names):
    """What ``_load_stack_dir`` does to the list: one item per file."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QListWidgetItem
    ui.loaded_patch_list.clear()
    for n in names:
        it = QListWidgetItem(n)
        it.setData(Qt.UserRole, n)
        ui.loaded_patch_list.addItem(it)


def _pm(colour="#336699"):
    from qtpy.QtGui import QColor, QPixmap
    pm = QPixmap(8, 8)
    pm.fill(QColor(colour))
    return pm


def test_session_list_is_painted_wide_with_no_tick_and_the_eye(ui):
    D = _D()
    lst = ui.loaded_patch_list
    assert isinstance(lst, D.StackListWidget)
    d = lst.itemDelegate()
    assert isinstance(d, D.StackRowDelegate) and d.mode == D.MODE_WIDE
    # A2 (board B): the eye is the foot's View for that row
    assert d._show_tick is False and d._show_eye is True
    assert d._keep_aspect is True
    from qtpy.QtWidgets import QAbstractItemView
    assert lst.selectionMode() == QAbstractItemView.ExtendedSelection
    assert lst.toolTip() == _P().S3_SESSION_LIST_HELP


def test_session_rows_carry_records_and_keep_name_and_text(ui):
    from qtpy.QtCore import Qt
    D = _D()
    _fill(ui, ["a.tif", "b.tif", "c.npy"])
    ui.update_patch_list_status(
        {"a.tif": "curating", "b.tif": "unprocessed"},
        axes_by_patch={"a.tif": "ZCYX", "b.tif": "ZYX"},
        info_by_patch={"a.tif": {"labels_shape": [4, 8, 8],
                                 "num_cells_final": 12}},
        shapes={"a.tif": [4, 2, 8, 8]}, missing={"b.tif"})
    lst = ui.loaded_patch_list
    recs = {lst.item(i).data(Qt.UserRole): lst.item(i).data(D.ROLE_RECORD)
            for i in range(lst.count())}
    assert [lst.item(i).text() for i in range(3)] == ["a.tif", "b.tif",
                                                      "c.npy"]
    assert recs["a.tif"]["line2"] == "ZCYX · 4×8×8 · 2 ch · 12 labels"
    assert recs["a.tif"]["word"] == "curating"
    assert recs["b.tif"]["warns_wide"] and recs["b.tif"]["menu"] is True
    assert recs["c.npy"]["word"] == "unprocessed"     # no record: the default
    assert lst.item(0).data(D.ROLE_STATUS) == "curating"


def test_the_pictures_are_reapplied_on_every_rebuild(ui):
    from qtpy.QtCore import Qt
    D, P = _D(), _P()
    _fill(ui, ["a.tif", "b.tif"])
    ui.set_patch_thumbs({"a.tif": (None, P.S3_THUMB_PENDING),
                         "b.tif": (None, P.S3_THUMB_PENDING)})
    lst = ui.loaded_patch_list
    assert lst.item(0).data(D.ROLE_THUMB) is None
    assert lst.item(0).data(D.ROLE_THUMB_TIP) == P.S3_THUMB_PENDING
    ui.set_patch_thumb("a.tif", _pm(), P.S3_THUMB_TIP)
    assert lst.item(0).data(D.ROLE_THUMB) is not None
    assert lst.item(0).data(D.ROLE_THUMB_TIP) == P.S3_THUMB_TIP
    # a record refresh and a whole rebuild keep it
    ui.update_patch_list_status({"a.tif": "curating", "b.tif": "curating"})
    assert lst.item(0).data(D.ROLE_THUMB) is not None
    _fill(ui, ["a.tif", "b.tif"])
    ui.update_patch_list_status({"a.tif": "curating", "b.tif": "curating"})
    assert lst.item(0).data(D.ROLE_THUMB) is not None
    assert lst.item(1).data(D.ROLE_THUMB) is None
    # Curation's list gets the same picture
    ui.refresh_curation_bucket(["a.tif", "b.tif"],
                               {"a.tif": "curating", "b.tif": "curating"})
    b = ui._curation_bucket
    got = {b.item(i).data(Qt.UserRole): b.item(i).data(D.ROLE_THUMB)
           for i in range(b.count())}
    assert got["a.tif"] is not None and got["b.tif"] is None
    # a new folder's map replaces the old one entirely
    ui.set_patch_thumbs({})
    assert lst.item(0).data(D.ROLE_THUMB) is None
    assert lst.item(0).data(D.ROLE_THUMB_TIP) is None
    assert b.item(0).data(D.ROLE_THUMB) is None


def test_session_rows_dim_under_predict_and_nothing_else_greys(ui):
    from napari_epicycle.fusion_reconstruction import ui as U
    _fill(ui, ["a.tif"])
    ui.update_patch_list_status({"a.tif": "reconstructed"})
    lst = ui.loaded_patch_list
    assert not lst.property(U.PR_LOCK_PROPERTY)
    ui.set_pr_surface_locked(True)
    assert lst.property(U.PR_LOCK_PROPERTY) is True
    assert lst.isEnabled()
    assert lst.itemDelegate()._dim_when_locked is True
    assert lst.itemDelegate()._lock_property == U.PR_LOCK_PROPERTY
    ui.set_pr_surface_locked(False)
    assert not lst.property(U.PR_LOCK_PROPERTY)


def test_the_session_list_takes_its_rows_height_up_to_a_cap(ui, app):
    from napari_epicycle.fusion_reconstruction import ui as U
    _fill(ui, ["a.tif"])
    ui.update_patch_list_status({"a.tif": "curating"})
    one = ui.loaded_patch_list.height()
    _fill(ui, [f"p{i}.tif" for i in range(30)])
    ui.update_patch_list_status({})
    many = ui.loaded_patch_list.height()
    assert one < many <= U.SESSION_LIST_MAX_H
    assert many == U.SESSION_LIST_MAX_H


# ------------------------------------------------------------ the controller

def _vol_grey():
    import numpy as np
    v = np.zeros((4, 16, 16), np.uint16)
    v[:, 4:12, 4:12] = 1000
    v[:, 6:10, 6:10] = 3000
    return v


def _vol_two():
    """ZCYX, C = 2: channel 0 on the left half, channel 1 on the right."""
    import numpy as np
    v = np.zeros((4, 2, 16, 16), np.uint16)
    v[:, 0, :, :8] = 4000
    v[:, 1, :, 8:] = 4000
    v[:, 0, 0, 0] = 1      # not flat, so the normalisation has a range
    v[:, 1, 0, 15] = 1
    return v


def _write_tif(path, arr, axes):
    import tifffile
    tifffile.imwrite(str(path), arr, metadata={"axes": axes})


@pytest.fixture
def ctl(app, tmp_path):
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
    _write_tif(folder / "g.tif", _vol_grey(), "ZYX")
    _write_tif(folder / "c.tif", _vol_two(), "ZCYX")
    (folder / "n.npy").write_bytes(b"")
    yield w, c, folder
    c.shutdown()
    timer = getattr(c.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    w.deleteLater()
    app.processEvents()


class _FakeStart:
    """Stands in for ``_step3_thumbs.start_step3_thumbs``: records each
    batch and starts nothing."""

    def __init__(self):
        self.calls = []

    def __call__(self, parent, requests, cache_dir, on_ready, on_failed, *,
                 state=None, on_finished=None, on_error=None):
        self.calls.append({"requests": list(requests), "ready": on_ready,
                           "failed": on_failed, "finished": on_finished,
                           "cache_dir": cache_dir})
        return state

    def started(self):
        return [c for c in self.calls if c["requests"]]


@pytest.fixture
def fake_start(monkeypatch):
    from napari_epicycle.fusion_reconstruction import _step3_thumbs as S3
    f = _FakeStart()
    monkeypatch.setattr(S3, "start_step3_thumbs", f)
    return f


def _thumbs(lst):
    from qtpy.QtCore import Qt
    D = _D()
    return {lst.item(i).data(Qt.UserRole): (lst.item(i).data(D.ROLE_THUMB),
                                            lst.item(i).data(D.ROLE_THUMB_TIP))
            for i in range(lst.count())}


def _records(lst):
    from qtpy.QtCore import Qt
    D = _D()
    return {lst.item(i).data(Qt.UserRole): lst.item(i).data(D.ROLE_RECORD)
            for i in range(lst.count())}


def _png(colour="#336699"):
    from qtpy.QtCore import QBuffer, QByteArray, QIODevice
    from qtpy.QtGui import QColor, QImage
    img = QImage(8, 8, QImage.Format_RGB32)
    img.fill(QColor(colour))
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


def test_the_folder_load_asks_for_every_listed_tiff_with_its_own_axes(
        ctl, fake_start):
    w, c, folder = ctl
    P = _P()
    c._load_stack_dir(str(folder), persist=False)
    started = fake_start.started()
    assert len(started) == 1                      # one batch for the trigger
    reqs = {r.name: r for r in started[0]["requests"]}
    assert sorted(reqs) == ["c.tif", "g.tif"]     # the .npy is skipped
    assert reqs["g.tif"].axes == "ZYX" and reqs["c.tif"].axes == "ZCYX"
    assert Path(started[0]["cache_dir"]) == (
        Path(c.working_dir) / "artifacts" / "step3" / "thumbs")
    # the rows wait with a placeholder and say so
    s = _thumbs(w.loaded_patch_list)
    assert s["g.tif"] == (None, P.S3_THUMB_PENDING)
    assert s["n.npy"] == (None, None)
    # the header read recorded the raw shape: the row says how many channels
    recs = _records(w.loaded_patch_list)
    assert recs["c.tif"]["line2"] == "ZCYX · 4×16×16 · 2 ch"
    assert recs["g.tif"]["line2"] == "ZYX · 4×16×16 · 1 ch"


def test_each_trigger_starts_at_most_one_batch(ctl, fake_start, tmp_path):
    import json
    from napari_epicycle.fusion_reconstruction import _paths
    w, c, folder = ctl
    (tmp_path / _paths.STEP3_STATE_FILENAME).write_text(json.dumps({
        "schema_version": 3, "active_run_id": "default",
        "loaded_stack_dir": str(folder),
        "runs": {"default": {"patches": {}},
                 "cells": {"patches": {}, "display_name": "cells"}}}),
        encoding="utf-8")
    c.state_manager.active_run_id = ""
    c.state_manager.load(str(tmp_path))
    c._load_stack_dir(str(folder), persist=False)
    assert len(fake_start.started()) == 1          # the folder load
    n = len(fake_start.calls)
    c.on_step_enter()                              # step enter
    assert len(fake_start.calls) == n              # kept: it is drawing
    c._switch_active_run("cells")                  # a run switch
    assert c.state_manager.get_active_run_id() == "cells"
    # the switch's folder load and its own trigger: one batch, not two
    assert len(fake_start.started()) <= 2
    switched = len(fake_start.calls)
    assert switched - n <= 1
    # a finished batch's pictures are in memory: the next trigger starts
    # nothing at all
    last = fake_start.started()[-1]
    for r in last["requests"]:
        last["ready"](r.name, r.path, "", _png())
    last["finished"]()
    before = len(fake_start.started())
    c.on_step_enter()
    assert len(fake_start.started()) == before
    assert all(t[0] is not None for nm, t in _thumbs(w.loaded_patch_list)
               .items() if nm.endswith(".tif"))


def test_an_older_batchs_picture_never_lands_on_a_new_row_of_the_same_name(
        ctl, fake_start, tmp_path):
    w, c, folder = ctl
    c._load_stack_dir(str(folder), persist=False)
    old = fake_start.started()[-1]
    old_path = next(r.path for r in old["requests"] if r.name == "g.tif")
    other = tmp_path / "other"
    other.mkdir()
    _write_tif(other / "g.tif", _vol_grey()[:, ::-1].copy(), "ZYX")
    c._load_stack_dir(str(other), persist=False)
    new = fake_start.started()[-1]
    new_path = next(r.path for r in new["requests"] if r.name == "g.tif")
    old["ready"]("g.tif", old_path, "", _png("#ff0000"))     # older batch
    assert _thumbs(w.loaded_patch_list)["g.tif"][0] is None
    new["ready"]("g.tif", old_path, "", _png("#ff0000"))     # wrong file
    assert _thumbs(w.loaded_patch_list)["g.tif"][0] is None
    old["failed"]("g.tif", old_path, "x")                    # older failure
    assert _thumbs(w.loaded_patch_list)["g.tif"][1] == _P().S3_THUMB_PENDING
    new["ready"]("g.tif", new_path, "", _png("#00ff00"))
    pm = _thumbs(w.loaded_patch_list)["g.tif"][0]
    assert pm is not None
    assert pm.toImage().pixelColor(2, 2).green() == 255


def test_every_rebuild_reapplies_the_pictures_from_memory(ctl, fake_start):
    w, c, folder = ctl
    c._load_stack_dir(str(folder), persist=False)
    c.state_manager.set_patch_status("c.tif", "reconstructed")
    c._refresh_patch_state_list()
    b = fake_start.started()[-1]
    for r in b["requests"]:
        b["ready"](r.name, r.path, "", _png())
    b["finished"]()
    for fn in (c._refresh_patch_state_list, c._refresh_patch_list_colors):
        fn()
        assert _thumbs(w.loaded_patch_list)["c.tif"][0] is not None
        assert _thumbs(w._curation_bucket)["c.tif"][0] is not None
    # the folder loaded again: the list is rebuilt, the pictures come from
    # memory, no batch is started
    n = len(fake_start.started())
    c._load_stack_dir(str(folder), persist=False)
    assert len(fake_start.started()) == n
    assert _thumbs(w.loaded_patch_list)["c.tif"][0] is not None
    assert _thumbs(w.loaded_patch_list)["c.tif"][1] == _P().S3_THUMB_TIP
    assert _thumbs(w._curation_bucket)["c.tif"][0] is not None


def test_a_failure_says_why_on_the_row_once_on_the_console_not_retried(
        ctl, fake_start, capsys):
    from napari_epicycle.ui_shell import _stack_thumbs as T
    from napari_epicycle.fusion_reconstruction import _step3_thumbs as S3
    w, c, folder = ctl
    c._load_stack_dir(str(folder), persist=False)
    b = fake_start.started()[-1]
    path = next(r.path for r in b["requests"] if r.name == "g.tif")
    capsys.readouterr()
    b["failed"]("g.tif", path, T.THUMB_FAIL_UNREADABLE)
    tip = S3.thumb_tooltip_for_failure(T.THUMB_FAIL_UNREADABLE)
    assert "patch" in tip and "stack" not in tip
    assert _thumbs(w.loaded_patch_list)["g.tif"] == (None, tip)
    out = [ln for ln in capsys.readouterr().out.splitlines()
           if ln.startswith("[STEP3-THUMB] g.tif:")]
    assert len(out) == 1
    b["finished"]()
    n = len(fake_start.started())
    c._start_step3_thumbs("again")
    later = fake_start.started()[n:]
    assert not any(r.name == "g.tif" for call in later
                   for r in call["requests"])
    assert _thumbs(w.loaded_patch_list)["g.tif"] == (None, tip)


def test_step_exit_and_shutdown_stop_the_batch_idempotently(app, ctl):
    w, c, folder = ctl
    c._load_stack_dir(str(folder), persist=False)
    state = c._s3_thumb_state
    token = c._s3_thumb_token
    c.on_step_exit()
    assert state.thread is None and state.worker is None
    assert c._s3_thumb_token > token
    c.on_step_exit()                                  # again: nothing
    c.shutdown()
    c.shutdown()
    assert state.thread is None and c._s3_thumb_batch is None


def test_a_read_only_adopt_writes_no_picture_into_the_workspace(ctl):
    w, c, folder = ctl
    c.state_manager.suppress_save = True
    try:
        d = c._s3_thumb_cache_dir()
        assert Path(c.working_dir) not in d.parents
    finally:
        c.state_manager.suppress_save = False
    assert c._s3_thumb_cache_dir() == (Path(c.working_dir) / "artifacts"
                                       / "step3" / "thumbs")


def _wait(app, cond, limit=30.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < limit:
        app.processEvents()
        time.sleep(0.02)
    return cond()


def test_the_main_thread_never_reads_pixels_for_a_picture(app, ctl,
                                                         monkeypatch):
    import tifffile
    w, c, folder = ctl
    calls = []
    real = tifffile.TiffFile.__init__

    def spy(self, *a, **k):
        calls.append(threading.current_thread() is threading.main_thread())
        return real(self, *a, **k)

    # the folder load's header reads (the axes) are the list's, done first
    c._load_stack_dir(str(folder), persist=False)
    _wait(app, lambda: c._s3_thumb_batch is None or c._s3_thumb_batch.done)
    c._s3_thumb_cache = None
    c._s3_thumb_failed = {}
    for p in (Path(c.working_dir) / "artifacts/step3/thumbs").glob("*.png"):
        p.unlink()
    monkeypatch.setattr(tifffile.TiffFile, "__init__", spy)
    c._start_step3_thumbs("spy")
    assert _wait(app, lambda: all(
        t[0] is not None for nm, t in _thumbs(w.loaded_patch_list).items()
        if nm.endswith(".tif")))
    assert calls and not any(calls), calls


def _rgb(pm):
    img = pm.toImage()
    return [(img.pixelColor(x, y).red(), img.pixelColor(x, y).green(),
             img.pixelColor(x, y).blue())
            for y in range(0, img.height(), 4)
            for x in range(0, img.width(), 4)]


def test_one_channel_draws_inferno_and_two_draw_magenta_and_green(app, ctl):
    """The owner's rule, end to end: a 3-D ``ZYX`` patch and a 4-D C = 2
    patch, the real worker, both real lists, the PNG each row was given,
    decoded."""
    w, c, folder = ctl
    c._load_stack_dir(str(folder), persist=False)
    for n in ("g.tif", "c.tif"):
        c.state_manager.set_patch_status(n, "reconstructed")
    c._refresh_patch_state_list()
    lists = (w.loaded_patch_list, w._curation_bucket)
    assert _wait(app, lambda: all(
        _thumbs(lst)[n][0] is not None for lst in lists
        for n in ("g.tif", "c.tif")))
    for lst in lists:
        one = _rgb(_thumbs(lst)["g.tif"][0])
        assert all(r - g >= -12 for r, g, _b in one), one[:5]  # inferno's red
        assert not all(r == g == b for r, g, b in one)
        assert max(r for r, _g, _b in one) > 100         # not blank
        two = _rgb(_thumbs(lst)["c.tif"][0])
        magenta = [p for p in two if p[0] > 150 and p[2] > 150 and p[1] < 60]
        green = [p for p in two if p[1] > 150 and p[0] < 60 and p[2] < 60]
        assert magenta and green, two[:8]
        assert not any(r == g == b and r > 30 for r, g, b in two)

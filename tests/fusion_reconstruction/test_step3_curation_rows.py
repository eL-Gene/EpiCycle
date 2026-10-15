"""Curation's box list on the shared row painter, with the picture
(the owner's wish of 2026-10-07, brick
V2).

The rows keep today's content -- the name, the status and its dots, the
measured / edited marks and the missing line -- painted in ``MODE_WIDE``
with no tick and no eye, the picture beside them.  Every item keeps
``Qt.UserRole`` = the name and ``text()`` = today's padded row; the list
keeps ``ExtendedSelection``, its click handler and V1's refusal of a missing
patch.  Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os

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


NAMES = ["a.tif", "b.tif", "c.tif"]
STATUSES = {"a.tif": "reconstructed", "b.tif": "curating",
            "c.tif": "finalized"}


def _rows(ui):
    from qtpy.QtCore import Qt
    D = _D()
    b = ui._curation_bucket
    return [(b.item(i).data(Qt.UserRole), b.item(i).text(),
             b.item(i).data(D.ROLE_RECORD)) for i in range(b.count())]


def test_the_box_list_is_painted_wide_with_no_tick_and_no_eye(ui):
    from qtpy.QtWidgets import QAbstractItemView
    D = _D()
    b = ui._curation_bucket
    assert isinstance(b, D.StackListWidget)
    d = b.itemDelegate()
    assert isinstance(d, D.StackRowDelegate) and d.mode == D.MODE_WIDE
    assert d._show_tick is False and d._show_eye is False
    assert d._keep_aspect is True
    assert b.selectionMode() == QAbstractItemView.ExtendedSelection
    # the old monospace text layout is gone: the painter draws the row
    assert "monospace" not in b.styleSheet()


def test_the_rows_keep_todays_content(ui):
    P = _P()
    ui.refresh_curation_bucket(NAMES, STATUSES, qc_extracted={"b.tif"},
                               edited_patches={"b.tif", "c.tif"})
    rows = _rows(ui)
    assert [r[0] for r in rows] == NAMES
    # text() is today's padded row (name, status, dots, marks)
    a_text, b_text = rows[0][1], rows[1][1]
    assert a_text.startswith(" a.tif") and "reconstructed" in a_text
    assert P.S3_CURATION_TAG_QC in b_text and "curating" in b_text
    recs = {r[0]: r[2] for r in rows}
    assert recs["a.tif"]["word"] == "reconstructed"
    assert recs["a.tif"]["dots"] == "●●" and recs["a.tif"]["line2"] == ""
    assert recs["b.tif"]["line2"] == (P.S3_CURATION_TAG_QC + P.S3_LINE_SEP
                                      + P.S3_CURATION_TAG_EDITED)
    assert recs["c.tif"]["line2"] == P.S3_CURATION_TAG_EDITED
    assert recs["c.tif"]["word"] == "finalized"


def test_a_missing_patch_keeps_its_row_amber_with_the_menu(ui):
    P = _P()
    ui.refresh_curation_bucket(NAMES, STATUSES, missing={"a.tif"})
    recs = {r[0]: r[2] for r in _rows(ui)}
    assert recs["a.tif"]["warns_wide"] == (P.S3_WARN_SIGN
                                           + P.S3_PATCH_MISSING,)
    assert recs["a.tif"]["menu"] is True and recs["a.tif"]["dim_name"]
    assert P.S3_PATCH_MISSING in _rows(ui)[0][1]      # today's text too
    assert ui._curation_bucket.item(0).toolTip() == P.S3_PATCH_MISSING


def test_the_picture_lands_on_the_box_row_and_survives_a_refresh(ui):
    from qtpy.QtGui import QColor, QPixmap
    D, P = _D(), _P()
    ui.refresh_curation_bucket(NAMES, STATUSES)
    pm = QPixmap(8, 8)
    pm.fill(QColor("#ff00ff"))
    ui.set_patch_thumb("b.tif", pm, P.S3_THUMB_TIP)
    b = ui._curation_bucket
    assert b.item(1).data(D.ROLE_THUMB) is not None
    assert b.item(1).data(D.ROLE_THUMB_TIP) == P.S3_THUMB_TIP
    ui.refresh_curation_bucket(NAMES, STATUSES)
    assert b.item(1).data(D.ROLE_THUMB) is not None
    assert b.item(0).data(D.ROLE_THUMB) is None


def test_a_row_is_at_least_the_pictures_height(ui, app):
    D = _D()
    ui.refresh_curation_bucket(NAMES, STATUSES)
    b = ui._curation_bucket
    b.resize(320, 300)
    ui.show()
    app.processEvents()
    assert b.sizeHintForRow(0) >= D.thumb_px(D.MODE_WIDE) + 2 * D.PAD_Y
    ui.hide()


# ------------------------------------------------------------ the controller

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
    tifffile.imwrite(str(folder / "b.tif"), np.zeros((4, 8, 8), np.uint8))
    c.loaded_stack_dir = str(folder)
    c.state_manager._state["patches"] = {
        "a.tif": {"status": "reconstructed"},
        "b.tif": {"status": "reconstructed"}}
    c._recompute_missing_patches()
    c._refresh_patch_state_list()
    yield w, c
    timer = getattr(c.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    c.shutdown()
    w.deleteLater()
    app.processEvents()


def test_a_click_on_a_missing_patch_is_still_refused(full, monkeypatch):
    from qtpy.QtCore import Qt
    w, c = full
    P = _P()
    assert "a.tif" in c._missing_patches
    switched = []
    monkeypatch.setattr(c, "_switch_to_curation_patch",
                        lambda name, **k: switched.append(name))
    b = w._curation_bucket
    item = next(b.item(i) for i in range(b.count())
                if b.item(i).data(Qt.UserRole) == "a.tif")
    b.clearSelection()
    item.setSelected(True)
    c._on_curation_patch_clicked(item)
    assert switched == []
    assert w.curation_status_label.text() == P.S3_CURATION_MISSING_REFUSED
    good = next(b.item(i) for i in range(b.count())
                if b.item(i).data(Qt.UserRole) == "b.tif")
    b.clearSelection()
    good.setSelected(True)
    c._on_curation_patch_clicked(good)
    assert switched == ["b.tif"]


def test_the_painted_menu_of_a_missing_box_row_offers_forget(full,
                                                            monkeypatch):
    """The ``⋯`` the painter draws on a missing row reaches V1's menu."""
    from qtpy.QtCore import QPoint
    w, c = full
    asked = []
    monkeypatch.setattr(c, "_pr_row_menu",
                        lambda name, gpos: asked.append(name))
    # R7w re-baselined: Session's ⋯ is Session's own menu (Set spacing…,
    # then Forget on a missing row), not V1's
    session = []
    monkeypatch.setattr(c, "_session_row_menu",
                        lambda name, gpos: session.append(name))
    w._curation_bucket.itemDelegate().menu_requested.emit("a.tif", QPoint())
    w.loaded_patch_list.itemDelegate().menu_requested.emit("a.tif", QPoint())
    assert asked == ["a.tif"]
    assert session == ["a.tif"]

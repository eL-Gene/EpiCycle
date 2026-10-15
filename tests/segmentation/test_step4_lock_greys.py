"""B2 fix pass R1-5 / R1-8: what the lock greys must LOOK grey.

A disabled CHECKED box keeps its tick in a dimmed blue (the
PANEL_CHECKBOX rule), and a locked row dims its name and its ``⋯`` glyph,
not only its tick.  Pixel samples on offscreen renders.
"""
from __future__ import annotations

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QStyle, QStyleOptionButton

from napari_epicycle.segmentation._stack_row_delegate import (
    CHECK_DISABLED_BLUE,
)

from ._step4_panel import close, make_workspace, panel, pump


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"))
    w = panel(qapp, ws)
    yield w
    close(w)


def _lock(w, qapp, on):
    w.controller._set_step4_flag("_measure_running", on)
    pump(qapp)


def _indicator_colour(cb):
    opt = QStyleOptionButton()
    cb.initStyleOption(opt)
    r = cb.style().subElementRect(QStyle.SE_CheckBoxIndicator, opt, cb)
    img = cb.grab().toImage()
    return QColor(img.pixel(r.left() + 2, r.top() + 2))


def test_a_greyed_checked_box_is_dimmed_blue(qapp, w):
    w.accordion.setCurrentIndex(1)          # ② open: the box is laid out
    pump(qapp)
    cb = w.normalize_cb
    cb.setChecked(True)
    _lock(w, qapp, True)
    assert not cb.isEnabled()
    got = _indicator_colour(cb)
    want = QColor(CHECK_DISABLED_BLUE)
    assert abs(got.red() - want.red()) <= 12
    assert abs(got.green() - want.green()) <= 12
    assert abs(got.blue() - want.blue()) <= 12
    _lock(w, qapp, False)
    assert _indicator_colour(cb).name() != want.name()


def _region_brightness(lw, row, key):
    """Mean brightness of one part of a row, from the delegate's own
    layout (``key`` = the name line "l1" or the ``⋯`` "menu")."""
    it = lw.item(row)
    rect = lw.visualItemRect(it)
    delegate = lw.itemDelegate()
    from napari_epicycle.segmentation._stack_row_delegate import ROLE_RECORD
    g = delegate.layout(rect, it.data(ROLE_RECORD) or {}, lw.font())
    r = g["lines"]["l1"] if key == "l1" else g[key]
    img = lw.viewport().grab().toImage()
    total, n = 0, 0
    for x in range(r.left(), r.right()):
        for y in range(r.top(), r.bottom()):
            c = QColor(img.pixel(QPoint(x, y)))
            total += max(c.red(), c.green(), c.blue())
            n += 1
    return total / max(n, 1)


@pytest.mark.parametrize("which", ["session_stacks_list", "stacks_list"])
def test_a_locked_row_dims_its_name(qapp, w, which):
    lw = getattr(w, which)
    before = _region_brightness(lw, 0, "l1")
    _lock(w, qapp, True)
    after = _region_brightness(lw, 0, "l1")
    _lock(w, qapp, False)
    assert after < before * 0.9, (before, after)


def test_a_locked_rows_menu_glyph_dims(qapp, w):
    lw = w.session_stacks_list
    before = _region_brightness(lw, 0, "menu")
    _lock(w, qapp, True)
    after = _region_brightness(lw, 0, "menu")
    _lock(w, qapp, False)
    assert after < before * 0.8, (before, after)


@pytest.mark.parametrize("which", ["session_stacks_list", "stacks_list"])
def test_the_eye_dims_under_measure_and_not_under_predict(qapp, w, which):
    """UI re-audit LOW-8 (K3): the eye is refused, and dimmed, under any
    job but Predict / Reconstruct."""
    lw = getattr(w, which)
    before = _region_brightness(lw, 0, "eye")
    w.controller._set_step4_flag("_predict_running", True)
    pump(qapp)
    during_run = _region_brightness(lw, 0, "eye")
    w.controller._set_step4_flag("_predict_running", False)
    _lock(w, qapp, True)                          # _measure_running
    during_measure = _region_brightness(lw, 0, "eye")
    _lock(w, qapp, False)
    # the narrow ② row's eye box also holds a bit of the (dimmed) name, so
    # "unchanged under Predict" is asserted on the wide ① row only
    if which == "session_stacks_list":
        assert abs(during_run - before) < before * 0.05, (before,
                                                          during_run)
    assert during_measure < during_run * 0.8, (during_run, during_measure)


def test_every_help_button_stays_live_under_a_lock(qapp, w):
    """UI re-audit LOW-6/7: the ? buttons of ② and ③ open their guides
    under every job (the guides say so)."""
    _lock(w, qapp, True)
    try:
        assert w._help_buttons
        for btn, _guide in w._help_buttons:
            assert btn.isEnabled(), _guide
    finally:
        _lock(w, qapp, False)


def test_a_greyed_checked_workbench_box_is_dimmed_blue(qapp):
    """UI re-audit: the Workbench's own check boxes follow the same rule."""
    from napari_epicycle.segmentation.analysis_widget import (
        AnalysisControlsWidget,
    )
    aw = AnalysisControlsWidget()
    try:
        aw.resize(900, 700)
        aw.show()
        pump(qapp)
        cb = aw.sync_viewer_colors_cb
        cb.setChecked(True)
        cb.setEnabled(False)
        pump(qapp)
        got = _indicator_colour(cb)
        want = QColor(CHECK_DISABLED_BLUE)
        assert abs(got.blue() - want.blue()) <= 12, got.name()
        assert abs(got.red() - want.red()) <= 12, got.name()
    finally:
        aw.hide()
        aw.deleteLater()

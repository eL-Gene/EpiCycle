"""The mirrored tick PAINTS ticked (PyQt6).

① (Session) and ② (Segment) show the same stacks, and a tick in one is
mirrored into the other by ``SegmentStack._on_row_tick_changed``.  The row
painter (``StackRowDelegate.paint``) reads ``index.data(Qt.CheckStateRole)``
and compares it with ``Qt.Checked``.  ``QListWidgetItem.setCheckState`` is a
C++ inline that stores ``static_cast<int>(state)``, so under PyQt6 the role
then holds a bare ``2``, which never equals the strict ``Qt.CheckState``
enum: the mirrored row painted an EMPTY box although ``checkState()`` said
Checked.  Qt's own keyboard toggle (Space, ``QStyledItemDelegate``) stores a
bare int on the SOURCE row the same way.

Every tick here goes through the real view: a mouse click on the painted
tick box (the delegate's ``editorEvent``) or a Space key press, then the
real ``itemChanged`` -> ``_on_row_tick_changed``.
"""
from __future__ import annotations

import pytest

from ._step4_panel import close, make_workspace, panel, pump

pytest.importorskip("qtpy")


def _row(lw, name):
    from qtpy.QtCore import Qt
    for i in range(lw.count()):
        it = lw.item(i)
        if it.data(Qt.UserRole) == name:
            return it
    raise AssertionError(f"no row {name!r}")


def _tick_rect(w, lw, it):
    """The tick box's rect in the view, from the delegate's own layout."""
    from qtpy.QtWidgets import QStyleOptionViewItem
    from napari_epicycle.segmentation._stack_row_delegate import ROLE_RECORD
    dlg = lw.itemDelegate()
    opt = QStyleOptionViewItem()
    opt.rect = lw.visualItemRect(it)
    g = dlg.layout(opt.rect, it.data(ROLE_RECORD) or {}, lw.font())
    assert "tick" in g, "this list paints no tick box"
    return g["tick"]


def _click_tick(qapp, w, lw, it):
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    lw.scrollToItem(it)
    pump(qapp)
    r = _tick_rect(w, lw, it)
    QTest.mouseClick(lw.viewport(), Qt.LeftButton,
                     Qt.KeyboardModifier.NoModifier, r.center())
    pump(qapp)


def _painted_checked(monkeypatch, lw, it) -> bool:
    """What the delegate PAINTS for *it*'s tick box (``_paint_tick``'s
    ``checked``), through the real ``paint``."""
    from qtpy.QtGui import QColor, QImage, QPainter
    from qtpy.QtWidgets import QStyle, QStyleOptionViewItem
    from napari_epicycle.segmentation import _stack_row_delegate as D

    seen = []

    def record(painter, r, checked, enabled, **_kw):
        seen.append(bool(checked))

    monkeypatch.setattr(D.StackRowDelegate, "_paint_tick",
                        staticmethod(record))
    dlg = lw.itemDelegate()
    idx = lw.indexFromItem(it)
    opt = QStyleOptionViewItem()
    lw.initViewItemOption(opt)
    opt.rect = lw.visualItemRect(it)
    opt.widget = lw
    opt.state |= QStyle.StateFlag.State_Enabled
    img = QImage(max(opt.rect.width(), 1), max(opt.rect.height(), 1),
                 QImage.Format.Format_ARGB32)
    img.fill(QColor(38, 41, 48))
    p = QPainter(img)
    try:
        dlg.paint(p, opt, idx)
    finally:
        p.end()
    monkeypatch.undo()
    assert seen, "the delegate painted no tick box"
    return seen[-1]


def _stored_is_checked(it) -> bool:
    """The delegate's own comparison, on the stored role data."""
    from qtpy.QtCore import Qt
    return it.data(Qt.CheckStateRole) == Qt.Checked


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path)
    w = panel(qapp, ws)
    yield w
    close(w)


@pytest.mark.parametrize("src_attr, dst_attr", [
    ("session_stacks_list", "stacks_list"),
    ("stacks_list", "session_stacks_list"),
])
def test_a_clicked_tick_paints_ticked_in_the_mirrored_list(
        qapp, w, monkeypatch, src_attr, dst_attr):
    from qtpy.QtCore import Qt
    src, dst = getattr(w, src_attr), getattr(w, dst_attr)
    s_it, d_it = _row(src, "P2.tif"), _row(dst, "P2.tif")
    assert d_it.checkState() != Qt.Checked
    assert _painted_checked(monkeypatch, dst, d_it) is False

    _click_tick(qapp, w, src, s_it)

    assert s_it.checkState() == Qt.Checked, "the click did not tick"
    assert "P2.tif" in w.ticked_names()
    assert d_it.checkState() == Qt.Checked            # the mirror ran
    assert _stored_is_checked(d_it), (
        f"mirrored role holds {d_it.data(Qt.CheckStateRole)!r}")
    assert _painted_checked(monkeypatch, dst, d_it) is True
    assert _painted_checked(monkeypatch, src, s_it) is True

    # and the untick mirrors and paints empty
    _click_tick(qapp, w, src, s_it)
    assert d_it.checkState() == Qt.Unchecked
    assert "P2.tif" not in w.ticked_names()
    assert _painted_checked(monkeypatch, dst, d_it) is False
    assert _painted_checked(monkeypatch, src, s_it) is False


def test_a_tick_ticked_back_and_forth_across_the_lists_paints_right(
        qapp, w, monkeypatch):
    from qtpy.QtCore import Qt
    a, b = w.session_stacks_list, w.stacks_list
    a_it, b_it = _row(a, "P1.tif"), _row(b, "P1.tif")
    _click_tick(qapp, w, a, a_it)          # tick in ①
    _click_tick(qapp, w, b, b_it)          # untick in ② (mirrored row)
    assert a_it.checkState() == Qt.Unchecked
    assert _painted_checked(monkeypatch, a, a_it) is False
    _click_tick(qapp, w, b, b_it)          # tick again in ②
    assert a_it.checkState() == Qt.Checked
    assert _painted_checked(monkeypatch, a, a_it) is True
    assert _painted_checked(monkeypatch, b, b_it) is True


def test_a_space_key_tick_paints_ticked_in_both_lists(qapp, w, monkeypatch):
    """Qt's own keyboard toggle stores a bare int on the SOURCE row; the
    painter must still read it as ticked."""
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    src, dst = w.stacks_list, w.session_stacks_list
    s_it, d_it = _row(src, "P3.tif"), _row(dst, "P3.tif")
    src.setFocus()
    src.setCurrentItem(s_it)
    pump(qapp)
    QTest.keyClick(src.viewport(), Qt.Key_Space)
    pump(qapp)
    if s_it.checkState() != Qt.Checked:
        QTest.keyClick(src, Qt.Key_Space)
        pump(qapp)
    assert s_it.checkState() == Qt.Checked, "Space did not tick"
    assert d_it.checkState() == Qt.Checked
    assert _painted_checked(monkeypatch, src, s_it) is True
    assert _painted_checked(monkeypatch, dst, d_it) is True


def test_the_painter_reads_every_stored_form(qapp, w, monkeypatch):
    """Enum, bare int (PyQt6's ``setCheckState``), partial and None."""
    from qtpy.QtCore import Qt
    lw = w.stacks_list
    it = _row(lw, "P1.tif")
    lw.blockSignals(True)
    try:
        for value, want in ((Qt.Checked, True), (2, True),
                            (Qt.Unchecked, False), (0, False),
                            (Qt.PartiallyChecked, False), (1, False),
                            (None, False)):
            # Qt keeps the OLD variant when the new one compares equal
            # (int 2 == enum Checked), so clear the role first: the form
            # under test is then really the one stored
            it.setData(Qt.CheckStateRole, None)
            it.setData(Qt.CheckStateRole, value)
            stored = it.data(Qt.CheckStateRole)
            if value is not None:
                assert type(stored) is type(value), (value, stored)
            assert _painted_checked(monkeypatch, lw, it) is want, value
    finally:
        it.setData(Qt.CheckStateRole, Qt.Unchecked)
        lw.blockSignals(False)

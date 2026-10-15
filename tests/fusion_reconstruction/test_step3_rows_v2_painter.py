"""A3: the shared
painter -- picture sizes per mode, the centred accent tick, the narrow
row's metadata / voxel / tier lines.  Geometry only (no pixel hashes; the
pixel contracts live in the goldens).  One file at a time, ``-o addopts=""``.
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


def _layout(delegate, rec, width=420):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont
    return delegate.layout(QRect(0, 0, width, 0), rec, QFont())


REC = {"name": "p.tif", "word": "curating", "axes": "CZYX",
       "shape_text": "147x240x240 - 2 ch"}
TIERS = (("good", 117, (106, 191, 105)), ("review", 119, (212, 168, 67)),
         ("bad", 119, (230, 100, 100)), ("garbage", 122, (150, 150, 150)),
         ("curated", 365, (90, 200, 220)))


def test_picture_rect_follows_the_mode_constants(app):
    from napari_epicycle.ui_shell import _stack_thumbs as T
    D = _D()
    w = _layout(D.StackRowDelegate(D.MODE_WIDE), REC)
    n = _layout(D.StackRowDelegate(D.MODE_NARROW, show_thumb=True), REC)
    m = _layout(D.StackRowDelegate(D.MODE_MEASURE, show_thumb=True), REC)
    assert w["thumb"].width() == w["thumb"].height() == T.THUMB_WIDE_PX
    assert n["thumb"].width() == n["thumb"].height() == T.THUMB_NARROW_PX
    assert m["thumb"].width() == m["thumb"].height() == T.THUMB_NARROW_PX
    assert w["height"] >= T.THUMB_WIDE_PX + 2 * D.PAD_Y
    assert n["height"] >= T.THUMB_NARROW_PX + 2 * D.PAD_Y
    assert D.thumb_px(D.MODE_WIDE) == 128 and D.thumb_px(D.MODE_NARROW) == 80


def test_a_measure_row_has_no_tick_but_a_picture_when_asked(app):
    D = _D()
    m = _layout(D.StackRowDelegate(D.MODE_MEASURE, show_thumb=True), REC)
    assert "tick" not in m and "thumb" in m
    plain = _layout(D.StackRowDelegate(D.MODE_MEASURE), REC)
    assert "thumb" not in plain


@pytest.mark.parametrize("mode", ["wide", "narrow"])
def test_the_tick_is_centred_vertically_on_the_row(app, mode):
    D = _D()
    rec = dict(REC, tier_counts=TIERS if mode == "wide" else ())
    g = _layout(D.StackRowDelegate(mode, show_thumb=True), rec)
    t = g["tick"]
    assert t.width() == t.height() == D.TICK
    mid_row = g["height"] / 2.0
    assert abs((t.top() + t.bottom() + 1) / 2.0 - mid_row) <= 1.0
    # the picture is centred the same way, and the text block sits on it
    th = g["thumb"]
    assert abs((th.top() + th.bottom() + 1) / 2.0 - mid_row) <= 1.0


def test_text_block_is_centred_on_the_picture_when_shorter(app):
    D = _D()
    g = _layout(D.StackRowDelegate(D.MODE_WIDE), {"name": "p.tif",
                                                  "word": "x"})
    l1, th = g["lines"]["l1"], g["thumb"]
    block_top, block_bot = l1.top(), g["lines"]["l2"].bottom()
    centre = (block_top + block_bot + 1) / 2.0
    assert abs(centre - (th.top() + th.bottom() + 1) / 2.0) <= 3.0


def test_tick_paints_accent_fill_and_halo_when_checked(app):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QColor, QImage, QPainter
    D = _D()
    d = D.StackRowDelegate(D.MODE_WIDE, accent="#d4a843")
    img = QImage(30, 30, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    d._paint_tick(p, QRect(7, 7, 16, 16), True, True,
                 accent=d._palette["accent"])
    p.end()
    centre = QColor.fromRgba(img.pixel(15, 7 + 3))          # inside the box, off the mark
    assert abs(centre.red() - 0xd4) <= 2 and abs(centre.green() - 0xa8) <= 2
    halo = QColor.fromRgba(img.pixel(5, 15))                # 2 px outside the box
    assert 60 <= halo.alpha() <= 120 and halo.red() > 150     # ~35 %
    off = QImage(30, 30, QImage.Format_ARGB32)
    off.fill(QColor(0, 0, 0, 0))
    p = QPainter(off)
    d._paint_tick(p, QRect(7, 7, 16, 16), False, True,
                 accent=d._palette["accent"])
    p.end()
    assert QColor.fromRgba(off.pixel(15, 10)).alpha() == 0          # hollow
    border = QColor.fromRgba(off.pixel(8, 15))
    assert border.alpha() > 200 and abs(border.red() - 0x5a) <= 4  # grey 2 px
    assert QColor.fromRgba(off.pixel(5, 15)).alpha() == 0           # no halo


def test_step4_default_accent_is_the_step4_green_not_blue(app):
    from napari_epicycle.ui_shell.theme import STEP4_ACCENT
    D = _D()
    d = D.StackRowDelegate(D.MODE_WIDE)
    assert d._palette["accent"].name().lower() == STEP4_ACCENT.lower()
    assert d._palette["accent"].name().lower() != "#4560c4"


def test_narrow_row_carries_the_voxel_line_and_the_tier_line(app):
    D = _D()
    d = D.StackRowDelegate(D.MODE_NARROW, show_thumb=True)
    base = _layout(d, REC, width=240)
    rec = dict(REC, vox_text="0.52 x 0.52 x 0.52 um (from Step 1)",
               tier_counts=TIERS)
    g = _layout(d, rec, width=240)
    assert "l3n" in g["lines"] and g["lines"]["l3n"].top() > \
        g["lines"]["l3"].top()
    assert g["tier_draw"] and "tiers" in g["lines"]
    assert g["lines"]["tiers"].top() >= g["lines"]["l3n"].bottom()
    assert g["height"] > base["height"]
    # compact: dot + number only; every tier fits by wrapping, none dropped
    assert len(g["tier_draw"]) == 5 and not g["tiers_elided"]
    ys = {r[1].top() for r in g["tier_draw"]}
    assert len(ys) >= 2                       # wrapped onto more than one line
    for dot, nr, wr, num, word, rgb in g["tier_draw"]:
        assert dot.right() <= g["mid"].right() + g["mid"].width()


def test_narrow_row_without_the_new_keys_is_unchanged(app):
    D = _D()
    a = _layout(D.StackRowDelegate(D.MODE_NARROW), REC, width=240)
    assert "l3n" not in a["lines"] and "tiers" not in a["lines"]


def test_a_narrow_row_that_asks_for_words_draws_dot_number_word_wrapped(app):
    D = _D()
    d = D.StackRowDelegate(D.MODE_MEASURE, show_thumb=True)
    items = (("good", 117, (1, 2, 3)), ("bad", 119, (4, 5, 6)),
             ("unmarked", 365, (7, 8, 9)))
    # (offscreen fonts are wide: 320 px stands for the live column)
    g = _layout(d, dict(REC, tier_counts=items, tier_words=True), width=320)
    assert [t[4] for t in g["tier_draw"]] == ["good", "bad", "unmarked"]
    for dot, nr, wr, num, word, rgb in g["tier_draw"]:
        assert wr.width() > 0 and wr.left() > nr.right()
    assert not g["tiers_elided"]
    w2 = _layout(d, dict(REC, tier_counts=items, tier_words=True), width=250)
    assert len({t[1].top() for t in w2["tier_draw"]}) >= 2    # wrapped
    # without the key the row stays compact (P+R's protected rows)
    c = _layout(d, dict(REC, tier_counts=items), width=240)
    assert [t[4] for t in c["tier_draw"]] == ["", "", ""]


# -- the fold: Curation's tier line wraps; one order everywhere ---------------

def _five():
    from napari_epicycle.fusion_reconstruction import _step3_tier_counts as TC
    cols = {t: (i, i, i) for i, t in enumerate(TC.TIERS)}
    return TC.tier_items({t: 100 + i for i, t in enumerate(TC.TIERS)}, cols)


def test_every_tier_line_follows_the_one_order(app):
    from napari_epicycle.fusion_reconstruction import _step3_tier_counts as TC
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    assert TC.TIERS == ("curated", "good", "review", "bad", "garbage")
    items = _five()
    assert [i[0] for i in items] == list(TC.TIERS)
    D = _D()
    for mode, kw, width in ((D.MODE_WIDE, {}, 640),
                            (D.MODE_WIDE, {"wrap_tiers": True}, 640),
                            (D.MODE_NARROW, {}, 420)):
        g = _layout(D.StackRowDelegate(mode, **kw), dict(REC,
                    tier_counts=items, tier_words=True), width)
        assert [w for _b, w in g["tier_items"]] == list(TC.TIERS)[
            :len(g["tier_items"])]
    # the tooltip lists them in the same order
    tip = P.S3_SESSION_TIERS_TIP
    pos = [tip.index(t) for t in TC.TIERS]
    assert pos == sorted(pos)


def test_curation_wraps_its_tier_line_instead_of_dropping_tiers(app):
    D = _D()
    d = D.StackRowDelegate(D.MODE_WIDE, show_tick=False, show_eye=False,
                           keep_aspect=True, wrap_tiers=True)
    plain = D.StackRowDelegate(D.MODE_WIDE, show_tick=False, show_eye=False,
                               keep_aspect=True)
    rec = dict(REC, tier_counts=_five())
    wide = _layout(d, rec, 700)
    assert len(wide["tier_draw"]) == 5 and wide["tier_draw"][0][4] == "curated"
    assert len({t[1].top() for t in wide["tier_draw"]}) == 1   # one line
    narrow = _layout(d, rec, 300)
    assert len(narrow["tier_draw"]) == 5 and not narrow["tiers_elided"]
    assert len({t[1].top() for t in narrow["tier_draw"]}) >= 2  # wrapped
    assert narrow["tier_draw"][0][4] == "curated"               # words kept
    for dot, nr, wr, *_ in narrow["tier_draw"]:
        assert max(nr.right(), wr.right()) <= narrow["lines"]["tiers"].right()
    # the same row in a delegate that does not wrap drops tiers with "…"
    assert _layout(plain, rec, 300)["tiers_elided"] is True
    # a column too narrow for one tier's words falls back to dot + number
    tiny = _layout(d, rec, 150)
    assert len(tiny["tier_draw"]) == 5 and tiny["tier_draw"][0][4] == ""


def test_curation_list_at_the_splitters_minimum_clips_nothing(app):
    from qtpy.QtWidgets import QWidget
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration)
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    try:
        w.resize(566, 900)
        w.show()
        app.processEvents()
        w.refresh_curation_bucket(["a.tif"], {"a.tif": "curating"})
        w.set_tier_counts({"a.tif": _five()})
        lst = w._curation_bucket
        item = lst.item(0)
        left = lst.parentWidget()
        # the splitter's smallest left column
        minw = max(left.minimumSizeHint().width(), 120)
        lst.setFixedWidth(minw)
        app.processEvents()
        rec = item.data(ROLE_RECORD)
        g = lst.itemDelegate().layout(
            lst.visualItemRect(item).adjusted(0, 0, 0, 0)
            .__class__(0, 0, lst.viewport().width(), 0), rec, lst.font())
        line = g["lines"]["tiers"]
        assert len(g["tier_draw"]) == 5 and not g["tiers_elided"]
        for dot, nr, wr, *_ in g["tier_draw"]:
            assert dot.left() >= line.left()
            assert max(nr.right(), wr.right()) <= line.right()
        assert g["height"] > 0
    finally:
        w.close()
        w.deleteLater()


# -- the fold: the loaded-row indicator ---------------------------------------

def _paint_row(delegate, rec, selected=False, w=300, h=None):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QColor, QFont, QImage, QPainter
    from qtpy.QtWidgets import QStyle, QStyleOptionViewItem
    from qtpy.QtGui import QStandardItem, QStandardItemModel
    model = QStandardItemModel()
    it = QStandardItem()
    it.setData(rec, 0x100 + 2)
    model.appendRow(it)
    g = delegate.layout(QRect(0, 0, w, 0), rec, QFont())
    h = h or g["height"]
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(QColor(38, 41, 48))
    opt = QStyleOptionViewItem()
    opt.rect = QRect(0, 0, w, h)
    opt.font = QFont()
    opt.state |= QStyle.StateFlag.State_Enabled
    if selected:
        opt.state |= QStyle.StateFlag.State_Selected
    p = QPainter(img)
    delegate.paint(p, opt, model.index(0, 0))
    p.end()
    return img


def test_the_loaded_row_has_an_accent_bar_and_glow_that_selection_keeps(app):
    from qtpy.QtGui import QColor
    D = _D()
    d = D.StackRowDelegate(D.MODE_WIDE, show_tick=False, show_eye=False,
                           accent="#d4a843")
    plain = _paint_row(d, dict(REC))
    loaded = _paint_row(d, dict(REC, loaded=True))
    mid = loaded.height() // 2
    bar = QColor.fromRgba(loaded.pixel(0, mid))
    assert (bar.red(), bar.green(), bar.blue()) == (0xd4, 0xa8, 0x43)
    assert QColor.fromRgba(loaded.pixel(1, mid)).red() == 0xd4   # 2 px wide
    assert QColor.fromRgba(loaded.pixel(2, mid)).red() != 0xd4 or \
        QColor.fromRgba(loaded.pixel(3, mid)).red() != 0xd4
    # the faint glow: the row's empty right side is lighter than the plain one
    gx = loaded.width() - 20
    assert QColor.fromRgba(loaded.pixel(gx, mid)) != \
        QColor.fromRgba(plain.pixel(gx, mid))
    assert QColor.fromRgba(plain.pixel(0, mid)).red() != 0xd4
    # a selected loaded row still carries the bar
    sel = _paint_row(d, dict(REC, loaded=True), selected=True)
    assert QColor.fromRgba(sel.pixel(0, mid)).red() == 0xd4


def test_the_loaded_bar_takes_the_lists_accent(app):
    from qtpy.QtGui import QColor
    D = _D()
    d = D.StackRowDelegate(D.MODE_MEASURE, show_thumb=True)   # Step 4 green
    img = _paint_row(d, dict(REC, loaded=True))
    c = QColor.fromRgba(img.pixel(0, img.height() // 2))
    assert (c.red(), c.green(), c.blue()) == (0x6a, 0xbf, 0x69)


# -- the audit fold: H1 the narrow voxel line, L8 tooltips -------------------

VOX = "0.52 × 0.52 × 0.52 µm (from Step 1)"


def test_a_narrow_voxel_line_draws_the_numbers_wrapped_not_cut(app):
    D = _D()
    d = D.StackRowDelegate(D.MODE_NARROW, show_thumb=True)
    g = _layout(d, dict(REC, vox_text=VOX), width=240)
    assert g["vox_shown"] == "0.52 × 0.52 × 0.52 µm"       # no source
    r = g["lines"]["l3n"]
    from qtpy.QtGui import QFont, QFontMetrics
    fm = QFontMetrics(d._fonts(QFont())["word_small"])
    assert r.height() >= fm.height()
    # a column too narrow for one line gets a taller, wrapped rectangle
    tight = _layout(d, dict(REC, vox_text=VOX), width=170)
    assert tight["lines"]["l3n"].height() > fm.height()
    # the amber word has no source part and is untouched
    none = _layout(d, dict(REC, vox_text="voxels", vox_warn=True), width=240)
    assert none["vox_shown"] == "voxels"


def test_the_narrow_voxel_tooltip_carries_the_source(app):
    from qtpy.QtCore import QPoint
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import QStyleOptionViewItem
    from qtpy.QtCore import QRect
    D = _D()
    d = D.StackRowDelegate(D.MODE_NARROW, show_thumb=True)
    rec = dict(REC, vox_text=VOX)
    g = _layout(d, rec, width=240)
    opt = QStyleOptionViewItem()
    opt.rect = QRect(0, 0, 240, g["height"])
    opt.font = QFont()

    class Idx:
        @staticmethod
        def data(role):
            return rec if role == D.ROLE_RECORD else None
    pos = g["lines"]["l3n"].center()
    assert d.help_key_at(opt, Idx, pos) == "row_vox"
    assert d.tip_text(opt, Idx, pos) == VOX


def test_a_measure_rows_tier_line_has_its_own_tooltip(app):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import QStyleOptionViewItem
    D = _D()
    d = D.StackRowDelegate(D.MODE_MEASURE, show_thumb=True)
    rec = dict(REC, tier_counts=_five(), tier_words=True)
    g = _layout(d, rec, width=420)
    opt = QStyleOptionViewItem()
    opt.rect = QRect(0, 0, 420, g["height"])
    opt.font = QFont()

    class Idx:
        @staticmethod
        def data(role):
            return rec if role == D.ROLE_RECORD else None
    box, word = g["tier_items"][1]
    assert d.help_key_at(opt, Idx, box.center()) == "row_tiers"
    assert d.tip_text(opt, Idx, box.center()) == word
    assert d.help_key_at(opt, Idx, g["lines"]["l1"].center()) == \
        "mt_row_click"


def test_pr_rows_have_a_tier_tooltip():
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    assert P.S3_ROW_HELP_LINES["row_tiers"] == P.S3_SESSION_TIERS_TIP

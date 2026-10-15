"""A2 : Session's and Curation's rows gain the tier counts, and an
amber Session row the inline "Set spacing…" button -- additive record keys
(``tier_counts``, ``row_action``) the shared painter draws; Step 4 never
sets them, so its rows are byte-identical (``test_step4_lift_goldens.py``).
Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

ITEMS = (("good", 412, (80, 200, 120)), ("review", 96, (255, 200, 60)),
         ("bad", 31, (255, 120, 40)), ("garbage", 12, (240, 60, 60)))


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _D():
    from napari_epicycle.ui_shell import _stack_row_delegate as D
    return D


def _R():
    from napari_epicycle.fusion_reconstruction import _step3_rows as R
    return R


def _P():
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    return P


def _layout(d, rec, width=420):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont
    return d.layout(QRect(0, 0, width, 0), rec, QFont())


def _session_delegate():
    D = _D()
    return D.StackRowDelegate(D.MODE_WIDE, show_tick=False, show_eye=True,
                              keep_aspect=True)


# -- the builders ----------------------------------------------------------------

def test_session_row_carries_the_tier_counts_only_when_given():
    R = _R()
    assert "tier_counts" not in R.build_session_row("p.tif", "curating",
                                                    "CZYX")
    rec = R.build_session_row("p.tif", "curating", "CZYX", tier_counts=ITEMS)
    assert rec["tier_counts"] == ITEMS
    assert "tier_counts" not in R.build_session_row(
        "p.tif", "curating", "CZYX", tier_counts=None)


def test_curation_row_carries_the_tier_counts_only_when_given():
    R = _R()
    assert "tier_counts" not in R.build_curation_row("p.tif", "curating")
    rec = R.build_curation_row("p.tif", "curating", qc_extracted=True,
                               tier_counts=ITEMS)
    assert rec["tier_counts"] == ITEMS


def test_an_amber_session_row_has_no_inline_button():
    P, R = _P(), _R()
    rec = R.build_session_row("p.tif", "unprocessed", "CZYX")
    assert rec["line3"] == P.S3_ROW_SPACING_NONE and rec["line3_warn"]
    # Set spacing… is on the menu and the right-click only
    assert "row_action" not in rec and rec["menu"] is True
    known = R.build_session_row(
        "p.tif", "unprocessed", "CZYX",
        spacing={"zyx": [1.0, 0.33, 0.33], "source": "user"})
    assert "row_action" not in known
    gone = R.build_session_row("p.tif", "unprocessed", "CZYX", missing=True)
    assert "row_action" not in gone          # its menu offers Forget


def test_the_new_keywords_are_keyword_only():
    import inspect
    R = _R()
    for fn in (R.build_session_row, R.build_curation_row):
        p = inspect.signature(fn).parameters["tier_counts"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None


# -- the painter -------------------------------------------------------------------

def test_the_tier_line_sits_under_the_last_line_and_adds_height(app):
    d = _session_delegate()
    R = _R()
    base = R.build_session_row("p.tif", "curating", "CZYX",
                               spacing={"zyx": [1, 1, 1], "source": "user"})
    # a wide list (the mockup's 578 px column; offscreen fonts are wide)
    plain = _layout(d, base, width=640)
    with_counts = _layout(d, dict(base, tier_counts=ITEMS), width=640)
    tl = with_counts["lines"]["tiers"]
    assert tl.top() >= with_counts["lines"]["l3w"].bottom()
    # A3: the 128 px picture sets the row's height, so the line costs none
    # while the text block is the shorter one -- but it is inside the row
    assert with_counts["height"] >= plain["height"]
    assert tl.bottom() < with_counts["height"]
    assert with_counts["height"] >= 128 + 2 * _D().PAD_Y
    assert "tiers" not in plain["lines"]
    # one hit-box per tier drawn, left to right, inside the line
    boxes = with_counts["tier_items"]
    assert [b[1] for b in boxes] == ["good", "review", "bad", "garbage"]
    xs = [b[0].left() for b in boxes]
    assert xs == sorted(xs) and boxes[-1][0].right() <= tl.right()


def test_a_narrow_row_drops_the_tiers_that_do_not_fit(app):
    d = _session_delegate()
    rec = dict(_R().build_session_row("p.tif", "curating", "CZYX"),
               tier_counts=ITEMS)
    narrow = _layout(d, rec, width=170)
    assert len(narrow["tier_items"]) < 4
    assert narrow["tiers_elided"] is True
    for r, _w in narrow["tier_items"]:
        assert r.right() <= narrow["lines"]["tiers"].right()


def test_the_action_button_follows_the_amber_word(app):
    d = _session_delegate()
    # (Session no longer sets it; the painter still supports it)
    rec = dict(_R().build_session_row("p.tif", "unprocessed", "CZYX"),
               row_action=("set_spacing", _P().S3_SET_SPACING_BTN))
    g = _layout(d, rec)
    a = g["action"]
    l3 = g["lines"]["l3w"]
    assert a.left() > l3.left()
    assert l3.top() - 2 <= a.top() and a.bottom() <= l3.bottom() + 4
    no = _layout(d, _R().build_session_row(
        "p.tif", "unprocessed", "CZYX",
        spacing={"zyx": [1, 1, 1], "source": "user"}))
    assert "action" not in no


def test_a_click_on_the_action_emits_name_and_key(app):
    from qtpy.QtCore import QEvent, QPoint, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QListWidgetItem
    D = _D()
    lst = D.StackListWidget()
    d = _session_delegate()
    lst.setItemDelegate(d)
    lst.resize(420, 300)
    it = QListWidgetItem("p.tif")
    it.setData(D.ROLE_NAME, "p.tif")
    it.setData(D.ROLE_RECORD,
               dict(_R().build_session_row("p.tif", "unprocessed", "CZYX"),
                    row_action=("set_spacing", _P().S3_SET_SPACING_BTN)))
    lst.addItem(it)
    got = []

    class Sink:
        def hit(self, name, key):
            got.append((name, key))
    sink = Sink()
    d.action_clicked.connect(sink.hit)
    from qtpy.QtWidgets import QStyleOptionViewItem
    opt = QStyleOptionViewItem()
    opt.rect = lst.visualItemRect(it)
    opt.font = lst.font()
    opt.widget = lst
    idx = lst.indexFromItem(it)
    g = d.layout(opt.rect, idx.data(D.ROLE_RECORD), opt.font)
    pos = g["action"].center()
    from qtpy.QtCore import QPointF
    ev = QMouseEvent(QEvent.MouseButtonRelease, QPointF(pos), Qt.LeftButton,
                     Qt.LeftButton, Qt.NoModifier)
    assert d.editorEvent(ev, lst.model(), opt, idx) is True
    assert got == [("p.tif", "set_spacing")]
    assert d.help_key_at(opt, idx, pos) == "row_action"
    lst.deleteLater()


def test_step4_rows_have_none_of_the_new_parts(app):
    D = _D()
    rec = {"name": "s.tif", "word": "predicted", "axes": "ZCYX",
           "n_channels": 2}
    g = _layout(D.StackRowDelegate(D.MODE_WIDE), rec)
    assert "action" not in g and "tiers" not in g["lines"]
    assert g.get("tier_items", []) == []

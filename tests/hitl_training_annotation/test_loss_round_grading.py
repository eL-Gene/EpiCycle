"""LossPlotWidget round grading (Step 2 look, brick T loss adapter)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")

from napari_epicycle.hitl_training_annotation.ui import loss_plot_widget as lpw

PAL = {"accent": "#d9a441", "accent_rgb": (217, 164, 65), "ok_text": "#cccccc"}
AMBER = (217, 164, 65)
BLUE = (159, 180, 216)


@pytest.fixture(scope="module")
def _app():
    from qtpy.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _make(_app, grade=True):
    w = lpw.LossPlotWidget()
    w.set_palette(PAL)
    if grade:
        w.set_grade_by_round(True)
    return w


@pytest.fixture
def widget(_app):
    w = _make(_app)
    yield w
    w.deleteLater()


def _pens(w):
    out = []
    for i in w._plot_widget.plotItem.items:
        if hasattr(i, "getData") and i is not w._vline:
            p = i.opts["pen"]
            out.append(None if p is None else (
                p.color().getRgb(), round(p.widthF(), 3), int(p.style().value)))
    return out


# --- pure helpers --------------------------------------------------------------

def test_family_and_number():
    assert lpw.family_of("lateral_0") == "lateral"
    assert lpw.family_of("axial_2") == "axial"
    assert lpw.family_of("lateral_1/axial") == "axial"      # suffix honoured
    assert lpw.family_of("foo/axial") == "axial"
    assert lpw.family_of("round_03") is None
    assert lpw.round_number("lateral_0") == 0
    assert lpw.round_number("axial_2") == 2
    assert lpw.round_number("axial_2/xz") == 2
    assert lpw.round_number("round_03") == 3
    assert lpw.round_number("weird") is None
    assert lpw.display_round(4) == 4


def test_grade_table_for_1_to_7_rounds():
    # n rounds -> ages n-1 ... 0 ; (alpha, width, style) by age
    expect = {0: (255, 2.0, "solid"), 1: (179, 1.4, "solid"),
              2: (140, 1.4, "dash"), 3: (102, 1.4, "dot"),
              4: (77, 1.0, "solid")}
    for n in range(1, 8):
        for age in range(n):
            assert lpw.grade_for_age(age) == expect[min(age, 4)]
    # strictly lighter with age, never below the faint grade
    alphas = [lpw.grade_for_age(a)[0] for a in range(7)]
    assert alphas == sorted(alphas, reverse=True) and alphas[-1] == 77


def test_tail_y_range_ignores_early_epochs():
    curve = [9.0] + [1.0] * 9 + [0.5] * 10        # first 10 % = 2 points
    lo, hi = lpw.tail_y_range([curve])
    assert hi < 2.0 and lo < 0.5 <= hi
    assert lpw.tail_y_range([]) is None
    assert lpw.tail_y_range([[]]) is None


# --- ids and the live family ---------------------------------------------------

def test_live_id_family_from_set_live_family(widget):
    widget.begin_live_round("round_03")
    assert widget._family("round_03") == "lateral"          # default
    widget.set_live_family("axial")
    assert widget._family("round_03") == "axial"
    assert widget._label("round_03", live=True) == "ax r3 (live)"
    widget.cancel_live_round()
    assert widget._live_family is None


def test_live_family_survives_begin_live_round(widget):
    widget.set_live_family("axial")             # hook may run before begin
    widget.begin_live_round("round_01")
    assert widget._family("round_01") == "axial"


# --- grading in Overlay --------------------------------------------------------

def _load(w, ids, n=20):
    for rid in ids:
        w.add_round_curve(rid, [1.0 - 0.02 * i for i in range(n)])


def test_overlay_grades_by_age_and_dims_other_family(widget):
    _load(widget, ["lateral_0", "lateral_1", "lateral_2", "lateral_3",
                   "lateral_4", "axial_0", "axial_1"])
    widget.set_view_filter("lateral")
    pens = dict(zip(widget._round_order, _pens(widget)))
    assert pens["lateral_4"] == ((*AMBER, 255), 2.0, 1)
    assert pens["lateral_3"] == ((*AMBER, 179), 1.4, 1)
    assert pens["lateral_2"] == ((*AMBER, 140), 1.4, 2)     # DashLine
    assert pens["lateral_1"] == ((*AMBER, 102), 1.4, 3)     # DotLine
    assert pens["lateral_0"] == ((*AMBER, 77), 1.0, 1)
    # the other family: its hue, 30 % throughout
    assert pens["axial_0"][0] == (*BLUE, 77)
    assert pens["axial_1"][0] == (*BLUE, 77)


def test_overlay_draws_both_families_even_with_filter(widget):
    _load(widget, ["lateral_0", "axial_0"])
    widget.set_view_filter("axial")
    assert len(_pens(widget)) == 2
    pens = dict(zip(widget._round_order, _pens(widget)))
    assert pens["axial_0"][0] == (*BLUE, 255)               # filter picks the bold
    assert pens["lateral_0"][0] == (*AMBER, 77)


def test_live_curve_is_bold_and_stored_ones_age(widget):
    _load(widget, ["lateral_0", "lateral_1"])
    widget.set_live_family("lateral")
    widget.begin_live_round("round_02")
    widget.append_live_point(0, 0.9)
    widget.append_live_point(1, 0.8)
    pens = _pens(widget)
    assert pens[0][0] == (*AMBER, 140) and pens[1][0] == (*AMBER, 179)
    assert pens[2][1] == 2.0                                # the live line


def test_other_family_untouched_in_current_and_per_round(widget):
    _load(widget, ["lateral_0", "axial_0", "axial_1"])
    widget.set_view_filter("axial")
    widget.set_display_mode("Current Round")
    assert _pens(widget) == [((*BLUE, 220), 2.0, 1)]
    widget.set_display_mode("Per Round")
    assert _pens(widget) == [((*BLUE, 220), 2.0, 1)]
    assert widget._round_selector.count() == 2              # axial only
    assert widget._round_selector.itemText(0) == "ax r0"


# --- legend, combo, caption ----------------------------------------------------

def _plain(html):
    import re
    return re.sub(r"<[^>]+>", "", html)


def test_legend_strings(widget):
    _load(widget, ["lateral_0", "lateral_1", "axial_0", "axial_1/xz"])
    widget.set_live_family("lateral")
    widget.begin_live_round("round_02")
    widget.append_live_point(0, 0.9)
    text = _plain(widget._legend_label.text())
    for w in ("lat r0", "lat r1", "lat r2 (live)", "ax r0", "ax r1"):
        assert w in text
    assert "/xz" not in text and "lateral_" not in text
    assert text.index("lat r0") < text.index("lat r2 (live)")
    assert "<b>lat r2 (live)</b>" in widget._legend_label.text()


def test_combo_visible_under_flag_hidden_without(_app):
    w = _make(_app, grade=False)
    assert w.mode_combo.isHidden()
    w.set_grade_by_round(True)
    assert not w.mode_combo.isHidden()
    assert w.mode_combo.itemText(1) == "Current Round"      # not "Current model"
    w.set_grade_by_round(False)
    assert w.mode_combo.isHidden()
    assert w.mode_combo.itemText(1) == "Current model"


def test_caption(widget):
    assert widget.grading_caption() == ""
    widget.add_round_curve("lateral_0", [1.0, 0.62])
    widget.add_round_curve("lateral_1", [1.0, 0.62])
    assert widget.grading_caption() == "last: lat r1 · train 0.62 · 2 rounds"


# --- y fit ---------------------------------------------------------------------

def test_y_fitted_to_tail_when_idle_and_auto_when_live(widget):
    widget.add_round_curve("lateral_0", [50.0] + [1.0] * 19)
    lo, hi = widget._plot_widget.plotItem.getViewBox().viewRange()[1]
    assert hi < 5.0                                          # the 50 is clipped
    widget.begin_live_round("round_01")
    assert widget._plot_widget.plotItem.getViewBox().autoRangeEnabled()[1]


# --- flag off: byte-identical pens ---------------------------------------------

def _fixture_three(w):
    for rid in ("lateral_0", "axial_0", "lateral_1"):
        w.add_round_curve(rid, [1.0, 0.8, 0.6])


# captured from the pre-change widget (flag off), pen = (rgba, width, style)
_BASELINE = {
    (False, "Overlay"): [((100, 180, 255, 100), 1.3, 1),
                         ((130, 220, 160, 100), 1.3, 1),
                         ((255, 170, 100, 100), 1.3, 1)],
    (False, "Current Round"): [((255, 170, 100, 220), 2.0, 1)],
    (False, "Per Round"): [((255, 170, 100, 220), 2.0, 1)],
    (True, "Overlay"): [((217, 164, 65, 100), 1.6, 1),
                        ((217, 164, 65, 100), 1.6, 1),
                        ((217, 164, 65, 255), 1.6, 1)],
    (True, "Current Round"): [((217, 164, 65, 220), 2.0, 1)],
    (True, "Per Round"): [((217, 164, 65, 220), 2.0, 1)],
}


@pytest.mark.parametrize("palette", [False, True])
@pytest.mark.parametrize("mode", ["Overlay", "Current Round", "Per Round"])
def test_flag_off_pens_byte_identical(_app, palette, mode):
    w = lpw.LossPlotWidget()
    if palette:
        w.set_palette(PAL)
    assert w.grade_by_round is False
    _fixture_three(w)
    w.set_display_mode(mode)
    assert _pens(w) == _BASELINE[(palette, mode)]
    w.deleteLater()

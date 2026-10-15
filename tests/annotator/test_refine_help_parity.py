"""Refine help parity with Reshape.

Reshape's rule, applied to Refine and held against the REAL widget, not
against a list typed by hand:

* every input control on the Refine page is explained -- by a grey
  ``HELP_LINES`` line rendered under it, or, where a line would push a
  measured state over its ceiling, by a tooltip, and then it
  must be NAMED in ``TOOLTIP_ONLY`` with its reason.  Set equality both
  ways, so a new control cannot slip in unexplained and a stale exemption
  cannot outlive its control;
* every item of the method, wall-type and solver combos carries an
  ``ITEM_HELP`` sentence, as its tooltip in the open list;
* the collapsed description equals the accepted wording **verbatim**, kept
  as a fixture beside this test (``fixtures/refine_about_description.md``);
* the ``?`` opens the guide retitled "Refining the shape of one object";
* no help line pushes a measured state over its ceiling;
* the state-9 sentence table has one sentence per method, keyed on it;
* the guide names every row label the page shows, and none it retired.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("qtpy")

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "docs" / "guides" / "annotator_3d_refine.md"
SPEC = Path(__file__).parent / "fixtures" / "refine_about_description.md"


def test_the_description_fixture_exists():
    assert SPEC.is_file() and SPEC.stat().st_size > 0


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover - CI without Qt
            pytest.skip("cannot create a QApplication: {}".format(exc))
    return app


@pytest.fixture
def widget(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    try:
        w.setParent(None)
    except Exception:
        pass


def _M():
    from napari_epicycle.annotator import advanced_features_widget as M

    return M


#: The controls that are explained by a TOOLTIP and not by a grey line, and
#: why.  Anything on the page that is neither in HELP_LINES nor here fails.
TOOLTIP_ONLY = {
    # the page's own chrome -- what a grey line would repeat is already on
    # screen next to it
    "refine_about_btn": "it opens the description, which IS the text",
    "refine_prep_btn": "a fold-out; its title lists what is ticked",
    "refine_advanced_btn": "a fold-out with nothing to explain but its name",
    "refine_show_walls_cb": "the legend under it is the explanation, and "
                            "says why when it is grey",
    "smooth_preview_btn": "pinned strip of fixed height; when it is grey "
                          "the status line says why",
    "refine_cancel_preview_btn": "pinned strip of fixed height; the stage "
                                 "line reads Stopping... while it works",
    # Refine recipes
    "refine_recipes_btn": "the pinned strip has no room for a line; every "
                          "greyed menu item says why in its own text",
    "smooth_accept_btn": "the footer's summary line says what both do",
    "smooth_reject_btn": "the footer's summary line says what both do",
    "refine_prob_cmap_combo": "it shares the overlay tick's row and line",
    # a sentence per ITEM instead of one line
    "refine_method_combo": "the method's description sits under it, and "
                           "each item has its own sentence in the list",
    "refine_contour_mode_combo": "the line under it swaps with the mode",
    "refine_contour_slices_combo": "the line under it swaps with the choice",
    # the contour's rows: that page is the tallest state, 751 of its 763 px
    # ceiling, so one more grey line there would push it over
    "refine_contour_axis_combo": "its four items explain themselves; the "
                                 "page has no room for a line",
    "refine_contour_share_sb": "the sampler's line just below covers it",
    "refine_contour_nslices_sb": "the sampler's line just below covers it",
    "refine_contour_limit_sb": "its line came off when the prepare card's "
                               "sentence went on; the page has no room",
    "refine_contour_iterations_sb": "the page has no room for a line",
    "refine_contour_balloon_sb": "its line is live, recomputed from the "
                                 "push and the gate",
    "refine_lambda1_sb": "the page has no room for a line",
    "refine_lambda2_sb": "the page has no room for a line",
}

INPUT_CLASSES = ("QComboBox", "QSpinBox", "QDoubleSpinBox", "QCheckBox",
                 "QPushButton")


def _refine_controls(widget):
    """``{attr: widget}`` for every input control on the Refine page."""
    from qtpy import QtWidgets

    page = widget._panel_slots["Refine"]["page"]
    classes = tuple(getattr(QtWidgets, c) for c in INPUT_CLASSES)
    on_page = set()
    for cls in classes:
        for w in page.findChildren(cls):
            on_page.add(id(w))
    out = {}
    for attr in dir(widget):
        if attr.startswith("__"):
            continue
        try:
            obj = getattr(widget, attr)
        except Exception:
            continue
        if isinstance(obj, classes) and id(obj) in on_page:
            out.setdefault(attr, obj)
    return out, on_page


class TestEveryControlIsExplained:
    def test_every_control_on_the_page_has_a_name(self, widget):
        """An unnamed control cannot be held to anything below.  The `?`
        is the one exemption -- it is built inline by `help_button`."""
        controls, on_page = _refine_controls(widget)
        named = {id(w) for w in controls.values()}
        from qtpy import QtWidgets

        page = widget._panel_slots["Refine"]["page"]
        stray = []
        for cls in INPUT_CLASSES:
            for w in page.findChildren(getattr(QtWidgets, cls)):
                if id(w) in named:
                    continue
                if w.toolTip() == "Help: Refine":
                    continue
                stray.append((cls, w.text() if hasattr(w, "text") else ""))
        assert not stray, stray

    def test_each_one_has_a_line_or_a_named_tooltip(self, widget):
        M = _M()
        controls, _ = _refine_controls(widget)
        missing = sorted(a for a in controls
                         if a not in M.HELP_LINES and a not in TOOLTIP_ONLY)
        assert not missing, "unexplained Refine control(s): {}".format(missing)

    def test_no_stale_exemption(self, widget):
        controls, _ = _refine_controls(widget)
        stale = sorted(set(TOOLTIP_ONLY) - set(controls))
        assert not stale, "TOOLTIP_ONLY names no control: {}".format(stale)

    def test_no_control_is_both(self):
        both = sorted(set(TOOLTIP_ONLY) & set(_M().HELP_LINES))
        assert not both, both

    def test_every_tooltip_only_control_has_a_tooltip(self, widget):
        controls, _ = _refine_controls(widget)
        bare = sorted(a for a in TOOLTIP_ONLY
                      if not controls[a].toolTip().strip())
        assert not bare, "no tooltip on: {}".format(bare)

    def test_every_line_is_rendered_in_a_real_label(self, widget):
        from qtpy.QtWidgets import QLabel

        M = _M()
        controls, _ = _refine_controls(widget)
        page = widget._panel_slots["Refine"]["page"]
        texts = {lb.text() for lb in page.findChildren(QLabel)}
        unrendered = sorted(a for a in controls
                            if a in M.HELP_LINES
                            and M.HELP_LINES[a] not in texts)
        assert not unrendered, unrendered

    def test_a_lined_control_s_tooltip_is_not_a_second_story(self, widget):
        """Where a control has both, they say the same thing -- two
        different sentences about one knob is how one of them goes stale
        (the Fill Holes and the two weights did, B3)."""
        M = _M()
        controls, _ = _refine_controls(widget)
        for attr, w in controls.items():
            tip = w.toolTip()
            if attr in M.HELP_LINES and tip:
                assert tip == M.HELP_LINES[attr], attr


class TestItemHelp:
    @pytest.mark.parametrize("attr", ["refine_method_combo",
                                      "refine_wall_kind_combo",
                                      "refine_solver_combo"])
    def test_every_item_carries_its_sentence_as_a_tooltip(self, widget, attr):
        from qtpy.QtCore import Qt

        M = _M()
        combo = getattr(widget, attr)
        texts = M.ITEM_HELP[attr]
        assert len(texts) == combo.count()
        for i in range(combo.count()):
            assert combo.itemData(i, Qt.ToolTipRole) == texts[i], (attr, i)

    def test_there_is_one_method_sentence_per_method(self):
        M = _M()
        assert len(M.ITEM_HELP["refine_method_combo"]) == len(
            M.REFINE_METHOD_KEYS)

    def test_a_greyed_guided_grow_carries_its_reason_then_its_help(
            self, widget):
        from qtpy.QtCore import Qt

        M = _M()
        idx = M.REFINE_METHOD_INDEX["guided_grow"]
        combo = widget.refine_method_combo
        widget.set_refine_guided_grow_available(False)
        assert combo.itemData(idx, Qt.ToolTipRole) == (
            M.REFINE_GUIDED_GROW_NO_PROB)
        widget.set_refine_guided_grow_available(True)
        assert combo.itemData(idx, Qt.ToolTipRole) == (
            M.ITEM_HELP["refine_method_combo"][idx])

    def test_a_greyed_solver_carries_its_reason_then_its_help(self, widget):
        from qtpy.QtCore import Qt

        M = _M()
        idx = M.REFINE_SOLVERS.index("label_diffusion")
        combo = widget.refine_solver_combo
        widget.set_refine_solver_available(False, M.REFINE_SOLVER_MISSING)
        assert combo.itemData(idx, Qt.ToolTipRole) == M.REFINE_SOLVER_MISSING
        assert widget._refine_solver_note.text() == M.REFINE_SOLVER_MISSING
        widget.set_refine_solver_available(True)
        assert combo.itemData(idx, Qt.ToolTipRole) == (
            M.ITEM_HELP["refine_solver_combo"][idx])


def _spec_11_1():
    text = SPEC.read_text(encoding="utf-8")
    sec = text.split("### 11.1 The description, verbatim", 1)[1]
    sec = sec.split("\n## ", 1)[0]
    quotes = [ln[2:].strip() for ln in sec.splitlines()
              if ln.startswith("> ")]
    return quotes[0], quotes[1]


def _unmark(s):
    return s.replace("**", "")


class TestTheDescriptionIsTheAcceptedWordingVerbatim:
    def test_the_visible_line(self):
        line, _body = _spec_11_1()
        assert _M().REFINE_ABOUT_LINE == _unmark(line)

    def test_the_body_s_first_paragraph(self):
        _line, first = _spec_11_1()
        body = _M().REFINE_ABOUT_BODY.split("\n\n")[0]
        body = (body.replace("&#8212;", "—")
                .replace("<b>", "**").replace("</b>", "**"))
        assert body == first

    def test_the_body_names_the_seven_methods_in_combo_order(self):
        M = _M()
        para = M.REFINE_ABOUT_BODY.split("\n\n")[1]
        names = re.findall(r"<b>(.*?)</b>", para)
        want = [t.split(" (")[0] for _i, t, _k in M.REFINE_METHOD_KEYS]
        assert names == want

    def test_it_is_collapsed_and_the_line_wraps(self, widget):
        assert widget.refine_about_btn.isChecked() is False
        assert widget._refine_about_body.isHidden() is True
        assert widget._refine_about_line.wordWrap() is True
        widget.refine_about_btn.setChecked(True)
        assert widget._refine_about_body.isHidden() is False
        assert widget._refine_about_body.wordWrap() is True

    def test_the_description_is_not_a_help_line(self):
        """UX-round 11's precedent: the chevron and the walls tick are
        tooltips, never HELP_LINES entries."""
        M = _M()
        assert "refine_about_btn" not in M.HELP_LINES
        assert "refine_show_walls_cb" not in M.HELP_LINES


class TestTheQuestionMark:
    def test_it_opens_the_retitled_guide(self, widget, monkeypatch):
        from qtpy.QtWidgets import QPushButton

        from napari_epicycle.ui_shell import theme

        seen = []
        monkeypatch.setattr(
            theme, "themed_help_dialog",
            lambda parent, title, path, accent=None: seen.append(
                (title, path)))
        page = widget._panel_slots["Refine"]["page"]
        btn, = [b for b in page.findChildren(QPushButton)
                if b.toolTip() == "Help: Refine"]
        btn.click()
        (title, path), = seen
        assert Path(path) == GUIDE
        assert title.startswith("Refining the shape of one object")
        first = GUIDE.read_text(encoding="utf-8").splitlines()[0]
        assert first == "# Refining the shape of one object"


class TestNoHelpLineBreachesACeiling:
    """The ceilings are OWNER-ACCEPTED and are not raised for a
    help line -- a line that does not fit is a tooltip instead."""

    def _classes(self):
        from tests.annotator.test_panel_skeleton import (
            TestTheRefinePageAfterBrickB1 as B1,
            TestTheRefinePageAfterBrickB2 as B2,
        )

        return B1(), B2()

    def _measure(self, b1, widget, app):
        from tests.annotator.test_panel_skeleton import _slots

        app.processEvents()
        return (_slots(widget, "Refine")["content"].sizeHint().height(),
                b1._screen_h(widget))

    def test_the_contour_2d_state_every_control_on(self, widget):
        b1, b2 = self._classes()
        app = b1._shown(widget)
        try:
            b2._contour_2d_every_control_on(widget)
            content, screen = self._measure(b1, widget, app)
            print("[B3-HEIGHT] contour 2D every control on: {} / {} = "
                  "{:.3f}".format(content, screen, content / screen))
            assert content <= b2.REFINE_CONTOUR_2D_SCREENS * screen
        finally:
            widget.hide()

    def test_settle_prepare_open(self, widget):
        b1, _b2 = self._classes()
        app = b1._shown(widget)
        try:
            b1._settle_prep_open(widget)
            content, screen = self._measure(b1, widget, app)
            print("[B3-HEIGHT] Settle prepare open: {} / {} = {:.3f}".format(
                content, screen, content / screen))
            assert content <= b1.REFINE_SETTLE_PREP_SCREENS * screen
        finally:
            widget.hide()

    @pytest.mark.parametrize("key", ["sdf", "dilate_erode", "guided_grow",
                                     "fill_holes", "flood_fill"])
    def test_the_five_short_pages_stay_under_the_lower_ceiling(
            self, widget, key):
        """The pages that GAINED lines in B3, each measured against the
        lower of the two ceilings, with Guided Grow greyed on every page
        but its own (a greyed method cannot be shown)."""
        M = _M()
        b1, _b2 = self._classes()
        app = b1._shown(widget)
        try:
            widget.set_refine_guided_grow_available(key == "guided_grow")
            widget.refine_method_combo.setCurrentIndex(
                M.REFINE_METHOD_INDEX[key])
            content, screen = self._measure(b1, widget, app)
            print("[B3-HEIGHT] {}: {} / {} = {:.3f}".format(
                key, content, screen, content / screen))
            assert content <= b1.REFINE_SETTLE_PREP_SCREENS * screen
        finally:
            widget.hide()


class TestTheNothingMovedTableIsComplete:
    def test_one_sentence_per_method(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        M = _M()
        table = Curation3DAnnotator._REFINE_NOTHING_MOVED
        assert set(table) == {k for _i, _t, k in M.REFINE_METHOD_KEYS}
        for key, text in table.items():
            assert text.startswith("Nothing moved"), key
            assert text.endswith("."), key
        assert len(set(table.values())) == len(table), "a shared sentence"

    def test_the_contour_s_reasons_each_have_their_own(self):
        from napari_epicycle.annotator import curation_3d_widget as C

        reasons = (C.REFINE_CONTOUR_PUSH_BELOW_GATE,
                   C.REFINE_CONTOUR_AT_LIMIT, C.REFINE_CONTOUR_FENCED)
        assert len(set(reasons)) == 3
        assert all(r.startswith("Nothing moved") for r in reasons)


class TestTheGuideMatchesThePage:
    def _guide(self):
        return GUIDE.read_text(encoding="utf-8")

    def test_every_row_label_on_the_page_is_named_in_the_guide(self, widget):
        """The guide is the only place a knob's meaning is written down at
        length, so every row the page can show is in it."""
        from qtpy.QtWidgets import QCheckBox, QLabel

        from napari_epicycle.ui_shell.theme import PANEL_ROW_LABEL_STYLE

        guide = self._guide()
        page = widget._panel_slots["Refine"]["page"]
        # the fixed label column of every row (`panel_form_label`)
        names = {lb.text() for lb in page.findChildren(QLabel)
                 if lb.styleSheet() == PANEL_ROW_LABEL_STYLE}
        names |= {cb.text() for cb in page.findChildren(QCheckBox)}
        # the method and mode names, as the combos list them
        M = _M()
        names |= {t for _i, t, _k in M.REFINE_METHOD_KEYS}
        names |= set(M.REFINE_CONTOUR_MODE_LABELS)
        names.discard("")
        assert len(names) > 20, sorted(names)
        missing = sorted(n for n in names if n not in guide)
        assert not missing, "the guide never names: {}".format(missing)

    # "Image weight" is NOT here: the troubleshooting section names it on
    # purpose, to say where it went (B3 audit).
    RETIRED = ("MorphGAC", "MorphACWE", "SDF Smooth", "Preserve topology",
               "Distance clamp", "cell", "Source picker",
               "Drawing is off", "cannot be chained",
               "nothing is written until")

    @pytest.mark.parametrize("word", RETIRED)
    def test_the_guide_names_nothing_that_was_retired(self, word):
        text = self._guide()
        pattern = r"\b{}\b".format(re.escape(word))
        assert not re.search(pattern, text), word
